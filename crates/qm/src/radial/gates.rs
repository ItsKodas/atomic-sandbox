//! The transferability gates T1–T7 (spec §4.8; T8 is Task 5's). CPU, f64, atomic units: energies in
//! Hartree, u = rR in u-form. Every threshold is a field of `LEDGER.gates` — universal, never per element
//! (CLAUDE.md rule 1); a miss refuses the candidate (the recipe then tries the next local, then the next f),
//! never loosens a threshold (rule 6).
//!
//! # Spec §4.8, the table
//! | gate | what | threshold |
//! |---|---|---|
//! | T1 eigenvalues (a regression: the reference state is an exact eigenstate of the KB Hamiltonian with the same screening) | KB pseudo-atom vs AE, reference configuration, every valence (n, l) | < 1 mHa (`t1_eig`) |
//! | T2 norm | ∫₀^{r_c} u² equal AE vs PP (TM's construction, as a regression) | < 1e-10 (`t2_norm`) |
//! | T3 excitation / transfer energies | AE vs KB-PP total-energy *differences* between the reference configuration and each of: the +1 ion; the s → p promotion; the −1 anion if bound in both; the spin-polarised Hund ground multiplet | each ΔE within 5 mHa (`t3_delta_e`) |
//! | T4 eigenvalues in the T3 configurations | every valence eigenvalue | < 5 mHa (`t4_eig`) |
//! | T5 log derivatives, in energy units | δE_equiv(E) at r_d = 1.2·max_l r_c,l (`t5_rd_factor`), E ∈ [ε_v,min − 1 Ha, ε_v,max + 1 Ha] (`t5_window`) on a 10 mHa mesh (`t5_mesh`), the projector channels and the local channel (never an ungenerated l) | max_E \|δE_equiv\| < 20 mHa (`t5_de_equiv`) |
//! | T6 hardness (Teter, PRB 48, 5031 (1993)) | ∂ε_l/∂f_l′ at ±0.25 e (`t6_dq`), AE vs PP | < 10 mHa/e (`t6_hardness`) |
//! | T7 ghost | G1, G2 of §4.6; G2's count also in the KB atom of every T3 configuration (per spin), against that configuration's AE atom's bound spectrum above the core | as stated |
//!
//! T1 covers every channel with a reference state, occupied or not (Task 3 review N4: under the
//! polynomial local Be's unoccupied 2p is a projector that nothing else gates). G1 and G2 at the
//! reference run in `kb::ghost_gates` before these gates; T7 adds the count per T3 configuration.
//!
//! # T5: why energy units, and the Wronskian form (review S4; plan review M5, Spec question 12)
//! For two regular solutions u_E, u_E′ of −½u″ + (V_l − E)u = 0, d/dr[u_E u_E′′ − u_E′ u_E′] =
//! 2(E − E′)u_E u_E′. Integrating to r_d and dividing by u_E(r_d)u_E′(r_d):
//!   D_E′ − D_E = 2(E − E′)∫₀^{r_d}u_E u_E′ dr / (u_E u_E′)(r_d),  so  ∂D/∂E = −(2/u(r_d)²)∫₀^{r_d}u² dr
//! (D = u′/u; the norm-conservation identity: equal norms inside r_d ⇒ equal ∂D/∂E at ε_l, Hamann,
//! Schlüter & Chiang 1979). A mismatch δD is therefore worth δE = δD·u(r_d)²/(2∫₀^{r_d}u²) — the same
//! meaning for every channel and element, unlike a radian threshold. That first form keeps D^PP's pole
//! where u_PP(r_d) = 0; the gated quantity is the pole-free Wronskian form
//!   **δE_equiv(E) = ½·W(û_AE, û_PP)(r_d) = ½[û_AE û′_PP − û′_AE û_PP](r_d)**,
//! each regular solution normalised to ∫₀^{r_d} û² dr = 1. With w = u(r_d)²/∫₀^{r_d}u², W(û_AE, û_PP) =
//! û_AE û_PP (D^PP − D^AE) = (D^PP − D^AE)·sgn·√(w_AE w_PP), i.e. the first form times √(w_PP/w_AE) =
//! 1 + O(δw): the two agree to third order at ε_l (where D, D′ and so w match). |δE_equiv| is finite (û
//! finite, norms positive) and independent of the sign of either û (W is bilinear). Perturbatively it is
//! ⟨û_AE|V_PP − V_AE|û_PP⟩ on [0, r_d] (the Wronskian identity again, with two potentials at one E); the
//! KB terms cancel beyond β's support r_β ≤ r_d. The rejected difference form ½(D_PP w_PP − D_AE w_AE) is
//! not first-order equal: the identity gives D′w = −2 for each potential at every E, so that form's slope
//! at ε_l is ½D(w′_PP − w′_AE) with w′ = 2D″/D′², and norm conservation matches D and D′ there, not D″.
//! At E = ε_l, û_AE = û_PP beyond r_c (same norm inside r_d), so δE_equiv(ε_l) = 0 to the discretisation.
//! Here û_AE is the regular solution in the reference AE potential (Coulomb start), û_PP the regular KB
//! solution (`kb::kb_regular`, pole-free) in V_loc + the reference screening; û′ is the 5-point stencil at
//! r_d = r[nearest(t5_rd_factor·max r_c)] and the norm `LogGrid::partial` to that index.
//!
//! # The configurations (T3, T4, T7)
//! - cation: one electron out of the highest valence shell (the occupied channel with the highest ε);
//! - s→p: one electron from the valence s to the valence p, when both channels exist, the s holds ≥ 1 and
//!   the p has room (the p channel may be the unoccupied one: Be 2s¹2p¹);
//! - anion: one electron into the lowest channel shell with room; kept only if bound in both atoms;
//! - Hund: the spin-polarised `ground_state` multiplet (the §4.1 scan); the PP gets its valence
//!   occupations per spin.
//! All but Hund are spin-restricted, like the reference; occupations move ½ e per spin. The AE atom of a
//! configuration is the core full in both spins plus the channels' (n, l) at the configuration's
//! occupations (`AtomOptions::fixed`). A configuration that cannot be built (Ne: no room in p for s→p,
//! no shell with room for the anion) is listed in `GateReport::skipped` with its reason (rule 7).
//! The anion arm (Task 1 re-review R1-N1): only `RadialError::Unbound { after_backtracking: true, .. }`
//! — of the AE atom, or of the KB atom (`solve_kb_atom` has the same fallback and confirmation) — files the
//! anion as "not bound in both" in `skipped`; any other error (no convergence, an impossible occupation, a
//! non-genuine flag) fails the gate: a solver failure is never filed as physics. Every integer anion
//! Z = 1–18 is unbound in the spherical LSDA atom (Task 1, measured): no −1/r tail, self-interaction
//! (Shore, Rose & Zaremba, Phys. Rev. B 15, 2858 (1977), "Failure of the local exchange approximation in
//! the evaluation of the H⁻ ground state"; bibliographic data verified via Crossref,
//! DOI 10.1103/PhysRevB.15.2858, 2026-10-15 — the paper's text was not read; the claim cited is its
//! title's). It is reported in the record as an LSDA property, never confined into binding.
//!
//! # T7 per configuration (Task 3 review S3)
//! For E_KB < 0 the GSS margin ε₀^loc − ε_l is at most 1/|E_KB| (variational), so a near-singular negative
//! E_KB that passes at the reference can grow a ghost under a transfer configuration's screening, within
//! mHa of the valence, where T4 need not see it. So in every T3 configuration that both atoms bind, per
//! spin and for every l ≤ l_max: the exact inertia count n_KB(top) of the configuration's KB atom (its own
//! output potential) must not exceed the configuration's AE atom's bound states of that l above its core
//! below top = min(0, ε_v,max + `g2_window`) (ε_v,max: the configuration's highest occupied valence AE
//! eigenvalue, both spins). As in G2: more is a ghost (refused); fewer is recorded (`T7Row`).
//!
//! # T6
//! ∂ε_l/∂f_l′ by a central difference over f_l′ ± `t6_dq` (spin-restricted, ½ per spin), AE vs PP, for
//! every occupied l and l′. A full shell cannot take +dq (Pauli): one-sided over [f − 2dq, f] there.
//!
//! # Method-convergence literals (review S2 practice)
//! 1e-12 electrons of slack in T6's full-shell test; 1e-9 electrons in the Hund valence-count check.
use super::atom::{ground_state, solve_atom, AtomOptions, AtomResult, RadialError, ShellOcc, SpinMode};
use super::kb::{kb_count_below, kb_regular, lowest, reference_config, solve_kb_atom, KbAtom, ValenceConfig};
use super::ledger::LEDGER;
use super::numerov::{bound_state, g_coeff, outward, regular_start};
use super::recipe::PseudoTable;

/// One T3 configuration: ΔE = E(config) − E(reference) in each atom, and T4's worst valence eigenvalue error.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferRow { pub name: String, pub de_ae: f64, pub de_pp: f64, pub max_eig_error: f64 }

/// T7's count in one T3 configuration, spin and l: the AE bound levels above the core below `top` and the
/// KB inertia count there (n_kb ≤ ae.len(), else refused; fewer is recorded, not a ghost).
#[derive(Clone, Debug, PartialEq)]
pub struct T7Row { pub config: String, pub spin: u8, pub l: u8, pub top: f64, pub ae: Vec<f64>, pub n_kb: usize }

/// The gate numbers: T1 worst |Δε| (Ha), T2 worst norm difference, T3 rows, T4 worst |Δε| over the T3
/// configurations, T5 (l, max |δE_equiv|), T6 (l, l′, AE slope, PP slope) in Ha/e, T7 rows, and what was
/// skipped with its reason (Default: the empty report of a table that has not been through the gates —
/// `recipe::ghost_checked_candidate`, tests only).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GateReport {
    pub t1: f64, pub t2: f64, pub t3: Vec<TransferRow>, pub t4: f64, pub t5: Vec<(u8, f64)>,
    pub t6: Vec<(u8, u8, f64, f64)>, pub t7: Vec<T7Row>, pub skipped: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TransferConfig { pub name: &'static str, pub ae: AtomOptions, pub pp: ValenceConfig, pub spin: SpinMode }

/// AE shells: the core full in both spins plus the valence channels' (n, l) at `val[l]`.
fn ae_shells(t: &PseudoTable, val: &ValenceConfig) -> Vec<ShellOcc> {
    let mut v: Vec<ShellOcc> = t.partition.core.iter().map(|s| ShellOcc { n: s.n, l: s.l, occ: [0.5 * s.occ, 0.5 * s.occ] }).collect();
    for (l, occ) in val.occ.iter().enumerate() {
        if occ[0] + occ[1] <= 0.0 { continue; }
        let n = t.partition.channels.iter().find(|c| c.l as usize == l).expect("a channel for every occupied l").n;
        v.push(ShellOcc { n, l: l as u8, occ: *occ });
    }
    v
}

fn moved(base: &ValenceConfig, from: Option<usize>, to: Option<usize>) -> ValenceConfig {
    let mut c = base.clone();
    if let Some(l) = from { c.occ[l][0] -= 0.5; c.occ[l][1] -= 0.5; }
    if let Some(l) = to { c.occ[l][0] += 0.5; c.occ[l][1] += 0.5; }
    c
}

fn cap(l: usize) -> f64 { 2.0 * (2 * l + 1) as f64 }

fn total(c: &ValenceConfig, l: usize) -> f64 { c.occ[l][0] + c.occ[l][1] }

/// The T3 configurations (module doc), in the order cation, s→p, anion, Hund; `hund` is the AE
/// spin-polarised ground state, whose valence occupations per spin the PP's Hund configuration takes.
pub fn transfer_configs(t: &PseudoTable, hund: &AtomResult) -> Vec<TransferConfig> {
    let base = reference_config(&t.partition);
    let chans = &t.partition.channels;
    let homo = chans.iter().filter(|c| c.occupied).max_by(|a, b| a.eps.total_cmp(&b.eps)).expect("valence").l as usize;
    let restricted = |name, pp: ValenceConfig| TransferConfig { name, ae: AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &pp)), pp, spin: SpinMode::Restricted };
    let mut out = vec![restricted("cation", moved(&base, Some(homo), None))];
    let has = |l: usize| chans.iter().any(|c| c.l as usize == l);
    if has(0) && has(1) && total(&base, 0) >= 1.0 && total(&base, 1) <= cap(1) - 1.0 {
        out.push(restricted("s→p", moved(&base, Some(0), Some(1))));
    }
    let eps_of = |l: usize| chans.iter().find(|c| c.l as usize == l).expect("a channel").eps;
    let room = (0..base.occ.len()).filter(|&l| has(l) && total(&base, l) < cap(l)).min_by(|&a, &b| eps_of(a).total_cmp(&eps_of(b)));
    if let Some(l) = room { out.push(restricted("anion", moved(&base, None, Some(l)))); }
    // Hund: the AE spin-polarised multiplet and the same valence occupations per spin for the PP
    let mut pp = ValenceConfig { occ: vec![[0.0; 2]; base.occ.len()] };
    for o in hund.orbitals.iter().filter(|o| t.partition.valence.iter().any(|v| v.n == o.n && v.l == o.l)) {
        pp.occ[o.l as usize] = o.occ;
    }
    out.push(TransferConfig { name: "Hund", ae: AtomOptions::aufbau(SpinMode::Polarised, hund.multiplicity), pp, spin: SpinMode::Polarised });
    out
}

/// Why a configuration the rules name was not built (rule 7: visible, never silent).
fn not_built(t: &PseudoTable, built: &[TransferConfig]) -> Vec<String> {
    let base = reference_config(&t.partition);
    let has = |l: usize| t.partition.channels.iter().any(|c| c.l as usize == l);
    let mut out = Vec::new();
    if !built.iter().any(|c| c.name == "s→p") {
        out.push(if !has(0) || !has(1) { "s→p: no s or no p channel".to_string() }
                 else if total(&base, 0) < 1.0 { "s→p: the valence s holds no electron".to_string() }
                 else { "s→p: the valence p shell is full".to_string() });
    }
    if !built.iter().any(|c| c.name == "anion") { out.push("anion: no channel shell has room".to_string()); }
    out
}

/// The AE regular solution at e in the reference potential, u-form, to i_end.
fn ae_regular(t: &PseudoTable, l: u8, e: f64, i_end: usize) -> Vec<f64> {
    let g = &t.grid;
    let gc = g_coeff(g, &t.reference.v[0], l, e);
    let (y0, y1) = regular_start(g, l, t.z as f64);
    outward(&gc, y0, y1, i_end, g.delta).0.iter().enumerate().map(|(i, y)| y * g.r[i].sqrt()).collect()
}

/// δE_equiv(e) = ½·W(û_AE, û_PP)(r_d) = ½[û_AE û′_PP − û′_AE û_PP](r_d), each regular solution
/// normalised to ∫₀^{r_d} û² dr = 1, at r_d = LEDGER.gates.t5_rd_factor · max r_c (module doc;
/// Spec question 12 as the plan review ruled).
pub fn delta_e_equiv(t: &PseudoTable, l: u8, e: f64) -> f64 {
    let g = &t.grid;
    let r_d = LEDGER.gates.t5_rd_factor * t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max);
    let i_d = g.nearest(r_d);
    let v_pp: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
    let proj = t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
    // (û, û′) at r_d
    let at_rd = |u: &[f64]| {
        let norm = g.partial(&u.iter().map(|x| x * x).chain(std::iter::repeat(0.0)).take(g.len()).collect::<Vec<_>>(), i_d).sqrt();
        let du = (u[i_d - 2] - 8.0 * u[i_d - 1] + 8.0 * u[i_d + 1] - u[i_d + 2]) / (12.0 * g.delta * g.r[i_d]);
        (u[i_d] / norm, du / norm)
    };
    let (ae, dae) = at_rd(&ae_regular(t, l, e, i_d + 2));
    let (pp, dpp) = at_rd(&kb_regular(g, &v_pp, proj, l, e, i_d + 2));
    0.5 * (ae * dpp - dae * pp)
}

/// T7's count in one configuration (module doc): per spin and l ≤ l_max, n_KB(top) against the AE levels
/// above the core below top.
fn t7_counts(t: &PseudoTable, name: &str, ae: &AtomResult, pp: &KbAtom, spin: SpinMode) -> Result<Vec<T7Row>, String> {
    let g = &t.grid;
    let part = &t.partition;
    let valence = |o: &&super::atom::Orbital| part.valence.iter().any(|v| v.n == o.n && v.l == o.l);
    let eps_v_max = ae.orbitals.iter().filter(valence)
        .flat_map(|o| (0..2).filter(|&s| o.occ[s] > 0.0).map(|s| o.eps[s]).collect::<Vec<_>>())
        .fold(f64::NEG_INFINITY, f64::max);
    let top = (eps_v_max + LEDGER.gates.g2_window).min(0.0);
    let spins: &[usize] = if spin == SpinMode::Restricted { &[0] } else { &[0, 1] };
    let mut rows = Vec::new();
    for &s in spins {
        for l in 0..=part.l_max {
            let n_core_l = part.core.iter().filter(|c| c.l == l).count();
            let levels: Vec<f64> = (n_core_l..).map_while(|k| bound_state(g, &ae.v[s], l, k, t.z as f64, None))
                .map(|b| b.e).take_while(|&e| e < top).collect();
            let proj = t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
            let n_kb = kb_count_below(g, &pp.v[s], proj, l, top);
            if n_kb > levels.len() {
                return Err(format!("T7 {name} spin {s}, l = {l}: {n_kb} KB bound states below {top:.4} Ha, the all-electron atom {} ({levels:?}): an extra state (ghost)", levels.len()));
            }
            rows.push(T7Row { config: name.to_string(), spin: s as u8, l, top, ae: levels, n_kb });
        }
    }
    Ok(rows)
}

fn genuinely_unbound<T>(r: &Result<T, RadialError>) -> bool {
    matches!(r, Err(RadialError::Unbound { after_backtracking: true, .. }))
}

/// T1–T7 on a candidate that passed G1/G2 (module doc); the first miss refuses it with the gate's name.
pub fn run_gates(t: &PseudoTable) -> Result<GateReport, String> {
    let gates = &LEDGER.gates;
    let g = &t.grid;
    let input = t.kb.input();
    let base = reference_config(&t.partition);
    let proj_of = |l: u8| t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
    let pp_ref = solve_kb_atom(g, &input, &base, SpinMode::Restricted).map_err(|e| format!("reference KB atom: {e}"))?;
    // T1 (a regression: the reference is an exact eigenstate of the KB Hamiltonian), every channel with a
    // reference state — an unoccupied one through the lowest KB level of its l in the reference potential
    let mut t1 = 0.0f64;
    for c in &t.partition.channels {
        let e = if c.occupied { pp_ref.eps[c.l as usize][0] } else {
            lowest(g, &pp_ref.v[0], proj_of(c.l), c.l, f64::NAN).ok_or_else(|| format!("T1: the unoccupied l = {} reference state is not bound by the KB atom", c.l))?
        };
        t1 = t1.max((e - c.eps).abs());
    }
    if t1 >= gates.t1_eig { return Err(format!("T1: {:.3} mHa", t1 * 1e3)); }
    // T2 (norm conservation, TM's construction, as a regression)
    let t2 = t.channels.iter().zip(&t.partition.channels).map(|(tm, c)| {
        let sq = |u: &[f64]| u.iter().map(|x| x * x).collect::<Vec<_>>();
        (g.partial(&sq(&tm.u), tm.i_c) - g.partial(&sq(&c.u), tm.i_c)).abs()
    }).fold(0.0, f64::max);
    if t2 >= gates.t2_norm { return Err(format!("T2: norm error {t2:.2e}")); }
    // T3 / T4 / T7
    let hund = ground_state(t.z, SpinMode::Polarised).map_err(|e| format!("Hund atom: {e}"))?;
    let (mut t3, mut t4, mut t7) = (Vec::new(), 0.0f64, Vec::new());
    let configs = transfer_configs(t, &hund);
    let mut skipped = not_built(t, &configs);
    for cfg in &configs {
        if cfg.name == "Hund" {
            // the polarised ground state must be the partition's core plus valence (true for Z ≤ 18)
            let n_v: f64 = cfg.pp.occ.iter().map(|o| o[0] + o[1]).sum();
            if (n_v - t.z_v).abs() > 1e-9 {
                return Err(format!("T3 Hund: the polarised ground state holds {n_v} electrons in the partition's valence shells, Z_v = {}", t.z_v));
            }
        }
        let ae = if cfg.name == "Hund" { Ok(hund.clone()) } else { solve_atom(t.z, &cfg.ae) };
        let pp = solve_kb_atom(g, &input, &cfg.pp, cfg.spin);
        let (ae, pp) = match (ae, pp) {
            (Ok(ae), Ok(pp)) => (ae, pp),
            // not both bound (the arm above), and each atom either bound or genuinely unbound (R1-N1)
            (ae, pp) if cfg.name == "anion" && (ae.is_ok() || genuinely_unbound(&ae)) && (pp.is_ok() || genuinely_unbound(&pp)) => {
                let say = |r: Result<String, RadialError>| r.unwrap_or_else(|e| e.to_string());
                skipped.push(format!("anion: not bound in both — LSDA (no −1/r tail, self-interaction); all-electron: {}; pseudo: {}",
                                     say(ae.map(|a| format!("bound, E = {:.6} Ha", a.energy.total))),
                                     say(pp.map(|p| format!("bound, E = {:.6} Ha", p.energy)))));
                continue;
            }
            (ae, pp) => return Err(format!("T3 {}: all-electron {:?} / pseudo {:?}", cfg.name, ae.err(), pp.err())),
        };
        let de_ae = ae.energy.total - t.reference.energy.total;
        let de_pp = pp.energy - pp_ref.energy;
        let mut worst = 0.0f64;
        for o in ae.orbitals.iter().filter(|o| t.partition.valence.iter().any(|v| v.n == o.n && v.l == o.l)) {
            for s in 0..2 { if o.occ[s] > 0.0 { worst = worst.max((o.eps[s] - pp.eps[o.l as usize][s]).abs()); } }
        }
        if (de_ae - de_pp).abs() >= gates.t3_delta_e { return Err(format!("T3 {}: ΔE_AE {de_ae:.6} vs ΔE_PP {de_pp:.6}", cfg.name)); }
        if worst >= gates.t4_eig { return Err(format!("T4 {}: eigenvalue error {:.3} mHa", cfg.name, worst * 1e3)); }
        t4 = t4.max(worst);
        t3.push(TransferRow { name: cfg.name.to_string(), de_ae, de_pp, max_eig_error: worst });
        t7.extend(t7_counts(t, cfg.name, &ae, &pp, cfg.spin)?);
    }
    // T5: projector channels and the local channel (never an ungenerated l)
    let occ_eps = || t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.eps);
    let e_min = occ_eps().fold(f64::INFINITY, f64::min) - gates.t5_window;
    let e_max = occ_eps().fold(f64::NEG_INFINITY, f64::max) + gates.t5_window;
    let steps = ((e_max - e_min) / gates.t5_mesh).round() as usize;
    let mut t5 = Vec::new();
    for c in &t.partition.channels {
        let worst = (0..=steps).map(|k| delta_e_equiv(t, c.l, e_min + k as f64 * gates.t5_mesh).abs()).fold(0.0, f64::max);
        if worst >= gates.t5_de_equiv { return Err(format!("T5 l = {}: max δE_equiv {:.2} mHa", c.l, worst * 1e3)); }
        t5.push((c.l, worst));
    }
    // T6: ∂ε_l/∂f_l′ by ±t6_dq (restricted), AE vs PP
    let mut t6 = Vec::new();
    let occupied: Vec<u8> = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.l).collect();
    for &lp in &occupied {
        let shifted = |dq: f64| { let mut c = base.clone(); c.occ[lp as usize][0] += 0.5 * dq; c.occ[lp as usize][1] += 0.5 * dq; c };
        // a full shell cannot take +dq (Pauli): one-sided difference over [f − 2dq, f] there
        let full = total(&base, lp as usize) + gates.t6_dq > cap(lp as usize) + 1e-12;
        let (hi, lo) = if full { (0.0, -2.0 * gates.t6_dq) } else { (gates.t6_dq, -gates.t6_dq) };
        let (cp, cm) = (shifted(hi), shifted(lo));
        let ae_p = solve_atom(t.z, &AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &cp))).map_err(|e| format!("T6: {e}"))?;
        let ae_m = solve_atom(t.z, &AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &cm))).map_err(|e| format!("T6: {e}"))?;
        let pp_p = solve_kb_atom(g, &input, &cp, SpinMode::Restricted).map_err(|e| format!("T6: {e}"))?;
        let pp_m = solve_kb_atom(g, &input, &cm, SpinMode::Restricted).map_err(|e| format!("T6: {e}"))?;
        for &l in &occupied {
            let n = t.partition.channels.iter().find(|c| c.l == l).expect("a channel").n;
            let ae_eps = |a: &AtomResult| a.orbitals.iter().find(|o| o.n == n && o.l == l).expect("occupied").eps[0];
            let slope_ae = (ae_eps(&ae_p) - ae_eps(&ae_m)) / (hi - lo);
            let slope_pp = (pp_p.eps[l as usize][0] - pp_m.eps[l as usize][0]) / (hi - lo);
            if (slope_ae - slope_pp).abs() >= gates.t6_hardness { return Err(format!("T6 ∂ε_{l}/∂f_{lp}: AE {slope_ae:.5} PP {slope_pp:.5}")); }
            t6.push((l, lp, slope_ae, slope_pp));
        }
    }
    Ok(GateReport { t1, t2, t3, t4, t5, t6, t7, skipped })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec §8 row 4, first failing test: AE vs KB-PP total-energy differences between the reference
    /// and the +1 ion, the s → p promotion and the spin-polarised Hund multiplet, each within 5 mHa.
    #[test]
    fn oxygen_transfer_energies_within_5mha() {
        let t = crate::radial::generate(8).expect("O generates");
        let names: Vec<&str> = t.gates.t3.iter().map(|r| r.name.as_str()).collect();
        for want in ["cation", "s→p", "Hund"] { assert!(names.contains(&want), "missing {want}: {names:?}"); }
        for row in &t.gates.t3 {
            eprintln!("O {}: ΔE_AE {:.6} ΔE_PP {:.6} (Δ {:+.3} mHa), T4 {:.3} mHa", row.name, row.de_ae, row.de_pp, (row.de_pp - row.de_ae) * 1e3, row.max_eig_error * 1e3);
            assert!((row.de_ae - row.de_pp).abs() < 5e-3, "{}: ΔE_AE {:.6} ΔE_PP {:.6}", row.name, row.de_ae, row.de_pp);
            assert!(row.max_eig_error < 5e-3, "{}: T4 eigenvalue error {:.6}", row.name, row.max_eig_error);
        }
        eprintln!("O skipped: {:?}", t.gates.skipped);
    }

    /// δE_equiv vanishes at the reference eigenvalue (beyond r_c the normalised û_AE and û_PP coincide
    /// there, so their Wronskian is zero; norm conservation then makes the mismatch second order in
    /// E − ε_l, spec §4.8 T5): at ε_l it is far below the 20 mHa gate.
    #[test]
    fn log_derivative_mismatch_vanishes_at_the_reference_energy() {
        let t = crate::radial::generate(8).unwrap();
        for c in t.partition.channels.iter().filter(|c| c.occupied) {
            let d = delta_e_equiv(&t, c.l, c.eps);
            eprintln!("O l = {}: δE_equiv(ε_l = {:.6}) = {d:.3e}", c.l, c.eps);
            assert!(d.abs() < 1e-4, "l = {}: δE_equiv(ε_l) = {d:.3e}", c.l);
        }
    }
}
