//! The recipe: Z in, a gated Kleinman–Bylander pseudopotential out (spec §4.4; CLAUDE.md rule 1 — the
//! only per-element input is Z, every constant is a field of `LEDGER`). CPU, f64, atomic units.
//!
//! # Spec §4.4, the rule this module implements
//! - r_c,l = f·r_max,l, r_max,l the outermost maximum of |u_l^AE| (beyond the outermost node, so u^PP is
//!   nodeless; a capped unoccupied channel (§4.2 item 3) may have r_c inside its own r_max, as Be's 2p
//!   does — allowed for a nodeless state; `tm_channel` refuses any r_c inside a node).
//! - f is not per element: it is taken from the universal ladder `LEDGER.f_ladder` = {1.6, 1.4, 1.2, 1.0},
//!   softest first, and the first f for which the potential passes every gate is kept. If none passes,
//!   the element is refused at this recipe version (`Refusal::NoLadderEntryPasses`, every tried entry
//!   listed with its reason; never worked around by hand).
//! - The local channel, in the universal order `LEDGER.local_order`: (1) the highest generated channel's
//!   V^ion (first row and Al–Ar: p, since the 3d is unbound and dropped — §4.2 item 3); (2) the smooth
//!   polynomial local, −Z_v/r + τ_c beyond r_loc = min_l r_c,l, the even quartic matched C² inside
//!   (`kb::polynomial_local`; τ_c the full core's Hartree tail, Task 2 review S3). The first candidate
//!   that passes the ghost gates at the current f is used; the remaining channels become projectors.
//!   Loop order: f outer, local inner — a softer f with the second local beats a harder f with the first.
//! - Gates: G1 and G2 (`kb::ghost_gates`), then T1–T7 (`gates::run_gates`, Task 4) on every candidate;
//!   then the dimer gate (Task 11) for Z ≤ `LEDGER.dimer_z_max`; T8 (h_req, Task 5) on the accepted entry
//!   only. The accepted entry carries its `GenerationRecord`.
//! - Molecular gate on f (review S6b; Task 11, `generate_with`): for any element whose all-electron σ → 0
//!   reference is affordable (Z ≤ `LEDGER.dimer_z_max`), the homonuclear dimer against the element's own
//!   all-electron curve is an additional gate on f, in the same descending order, applied after the atomic
//!   gates of the entry pass. It is a 3-D computation the radial code cannot run, so the caller supplies it
//!   ([`DimerGate`]; `providers::pseudo::GridDimerGate`). A table for such an element generated without it
//!   is **provisional** (`PseudoTable::provisional`): reported, and refused by every `IonicSetup::build`
//!   (`IonicError::Provisional`), so the ungated and gated potentials of one element never stand in for
//!   each other under one `recipe_hash` (§4.9, §11 M6; switching the gate on moved `recipe_hash` through
//!   `ledger::DIMER_PROCEDURE`, the gated species' own identity — `ledger` module doc, "Two identities").
//!
//! # The dimer gate (spec §4.4; Spec question 9 ruling) — atomic units, Hartree and Bohr
//! Two nuclei of charge Z at ±R/2 on x; E(R) the Born–Oppenheimer energy; F(R) = F_x on the nucleus at +R/2
//! = −∂E/∂x₁ = −dE/dR (positive = repulsive). The caller locates, for the pseudised dimer and for the
//! element's own all-electron dimer, R_e with F(R_e) = 0 and F′(R_e) < 0 inside the bracket
//! `LEDGER.dimer_bracket`·r_peak (r_peak: the outermost occupied valence maximum of the AE reference, so
//! the bracket is f-independent), k = E″(R_e) = −F′(R_e) and ω = √(k/μ), μ = m/2 (the most abundant
//! isotope's mass — the one per-element datum allowed). The all-electron F at each R is the σ → 0 limit
//! of F(R; σ) = F₀ + bσ² + cσ³ + dσ⁴ (`scf::extrapolate_sigma`; the R-derivative of the energy ansatz has
//! the same σ dependence). Pass ⇔ |R_e^PP − R_e^AE| < `dimer_re` and |ω^PP/ω^AE − 1| < `dimer_omega_rel`
//! (the §6.4 numbers). An AE dimer with no force zero in the bracket (F not + at the inner end and − at the
//! outer) makes the gate **not applicable** for that element — a derived outcome recorded with the bond
//! forces at both bracket ends (He: LSDA's He₂ well lies near 5 Bohr, outside [1, 4]·r_peak(1s)) — provided the
//! pseudised dimer's repulsive wall agrees with the AE one at the outer end (`DIMER_PROCEDURE.wall_rel`, Task 11 review
//! Minor-2), else the entry fails. The end-sign test presumes a monotone repulsive AE curve in the bracket when it has no
//! sign change (true for Z ≤ `dimer_z_max` = 2: H₂ is bound, He₂ repulsive throughout); an even number of roots would
//! be missed — a reason the gate's bound stays at Z ≤ 2. A
//! pseudised dimer that is unbound where the AE one is bound fails the entry; a gate that cannot compute
//! (a solver error) fails the entry with its reason.
//! - The ladder's outputs (Z → f, local, r_c,l, Z_v, E_KB, GSS, h_req) are pinned in `references.md` by
//!   the validation task, so a change to any universal constant that moves an element shows in review.
//!
//! # Inputs fixed here
//! The reference is the spin-restricted ground state (spec §4.3), PW92. The partial core is LFC's
//! (`nlcc::partial_core`, spec §4.5: r_nlc from the AE valence density, so f-independent and computed once
//! per element); it enters the unscreening's and the KB atom's v_xc only. τ_c always uses the *full* core
//! (ρ_pc never enters V_H; Task 2 review S3), and `kb_form`'s derived support r_β moves out to r_nlc by
//! itself where r_nlc > max r_c (no new constant).
use super::atom::{ground_state, AtomResult, SpinMode};
use super::filter;
use super::gates::{run_gates, GateReport};
use super::grid::LogGrid;
use super::kb::{core_tail, ghost_gates, kb_form, GhostReport, KbPotential};
use super::ledger::{LocalCandidate, LEDGER};
use super::nlcc::{partial_core_at, PartialCore};
use super::partition::{core_density, partition, valence_density, Partition, Refusal};
use super::record::{record_of, GenerationRecord};
use super::tm::{tm_channels, unscreen, TmChannel};
use std::sync::Arc;
use std::time::Instant;

/// A generated, gated pseudopotential with everything it was built from. `screen` is the reference
/// screening V_H[ρ_v^PP] + v_xc[ρ_v^PP + ρ_pc] (= V_l^scr − V_l^ion, the same for every l). `gates` holds
/// T1–T7's numbers, `record` the generation record (filled on the entry `generate` accepts).
#[derive(Clone, Debug)]
pub struct PseudoTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub f: f64, pub local: LocalCandidate, pub kb: KbPotential,
    pub channels: Vec<TmChannel>, pub screen: Vec<f64>, pub partition: Partition,
    pub reference: Arc<AtomResult>, pub grid: LogGrid, pub ghost: GhostReport,
    pub partial_core: PartialCore, pub gates: GateReport, pub record: GenerationRecord,
    /// Z ≤ `LEDGER.dimer_z_max` generated without the dimer gate (plan review M6): reported, never usable —
    /// `IonicSetup::build` refuses it (`IonicError::Provisional`).
    pub provisional: bool,
}

/// The dimer gate's two minima (spec §4.4): (R_e Bohr, ω a.u.) of the pseudised dimer and of the element's own
/// all-electron dimer at σ → 0, both on the same grid by the same procedure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimerPair { pub pp: (f64, f64), pub ae: (f64, f64) }

/// Why the dimer gate returned no pair. The plan's interface was `Result<DimerPair, String>` with `Err` read as
/// "not applicable"; a solver failure must not read as that, so the three outcomes are told apart.
#[derive(Clone, Debug, PartialEq)]
pub enum DimerError {
    /// The all-electron dimer has no force zero in the bracket: the gate does not apply to this element (the
    /// string carries the PP and AE bond forces at both bracket ends, plan review Q9(ii)).
    NotApplicable(String),
    /// The AE dimer is bound in the bracket and the pseudised one is not: this ladder entry fails the gate.
    PseudoUnbound(String),
    /// The AE dimer has no force zero in the bracket, and the pseudised dimer's repulsive wall disagrees with the AE one
    /// at the outer bracket end beyond the procedure's `wall_rel` (Task 11 review Minor-2: "not applicable" must not hide a
    /// bad table): this ladder entry fails.
    WallMismatch(String),
    /// The 3-D computation failed (a solver error): this ladder entry fails, with the reason.
    Failed(String),
}

/// The molecular gate on f (module doc, "The dimer gate"; review S6b; Spec question 9): the caller computes the
/// pseudised homonuclear dimer and the element's own all-electron σ → 0 dimer on the grid.
pub trait DimerGate { fn dimers(&mut self, table: &PseudoTable) -> Result<DimerPair, DimerError>; }

/// Z in, potential out (spec §4.4) without the dimer gate: for Z ≤ `LEDGER.dimer_z_max` the table is
/// provisional (module doc) — reported, never usable. `generate_with` with a gate makes the usable one.
pub fn generate(z: u8) -> Result<PseudoTable, Refusal> { generate_with(z, None) }

/// The whole ladder (Tasks 3–5 and 11): reference atom → partition → partial core → for each (f, local) in
/// descending f: TM/KB/G1/G2/T1–T7 (`candidate`), then the dimer gate for Z ≤ `LEDGER.dimer_z_max`; the first
/// passing entry gets h_req, h_eb and its filter rows; none → refused with every reason.
pub fn generate_with(z: u8, dimer: Option<&mut dyn DimerGate>) -> Result<PseudoTable, Refusal> {
    generate_at_crossover_with(z, LEDGER.nlcc_crossover, dimer)
}

/// `generate` with the NLCC crossover c given (`nlcc::partial_core_at`). Only `LEDGER.nlcc_crossover` makes the recipe's
/// table: the other rungs of `LEDGER.nlcc_crossover_ladder` exist for the ladder's selection table (spec §4.9; the record's
/// `recipe_hash` is the build's, so a table made at another c is a study table and must not be used by an ionic setup).
pub fn generate_at_crossover(z: u8, crossover: f64) -> Result<PseudoTable, Refusal> { generate_at_crossover_with(z, crossover, None) }

/// [`generate_at_crossover`] with the dimer gate (module doc).
pub fn generate_at_crossover_with(z: u8, crossover: f64, mut dimer: Option<&mut dyn DimerGate>) -> Result<PseudoTable, Refusal> {
    let start = Instant::now();
    let reference = Arc::new(ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?);
    let part = partition(&reference)?;
    // LFC (spec §4.5): from the AE core and valence densities, independent of f
    let pc = partial_core_at(&reference.grid, &core_density(&reference, &part), &valence_density(&reference, &part), crossover)
        .map_err(|message| Refusal::PartialCore { z, message })?;
    let mut tried = Vec::new();
    for &f in LEDGER.f_ladder.iter() {
        for &local in LEDGER.local_order.iter() {
            let mut t = match candidate(&reference, &part, f, local, &pc) {
                Ok(t) => t,
                Err(why) => { tried.push(format!("f = {f}, {local:?}: {why}")); continue; }
            };
            // the molecular gate on f (Task 11), after the entry's atomic gates, in the same descending order
            if t.z <= LEDGER.dimer_z_max && !dimer_verdict(&mut t, dimer.as_deref_mut()) {
                tried.push(format!("f = {f}, {local:?}: dimer gate {}", t.record.dimer));
                continue;
            }
            // T8 (spec §4.8; Task 5): the filter ladder and h_req of the accepted entry
            let (h_req, rows) = filter::h_req(&t);
            t.record.h_req = h_req;
            t.record.h_eb = filter::h_eb(&rows);
            t.record.filter = rows;
            t.record.seconds = start.elapsed().as_secs_f64();
            return Ok(t);
        }
    }
    Err(Refusal::NoLadderEntryPasses { z, tried })
}

/// Fills `record.dimer` (module doc, "The dimer gate"); false ⇔ this ladder entry fails the gate (the ladder moves
/// on). Without a gate the table is provisional (plan review M6): reported, never usable — `IonicSetup::build`
/// refuses it, so an ungated table can never stand in for the gated one under the same `recipe_hash`.
fn dimer_verdict<'g>(t: &mut PseudoTable, gate: Option<&mut (dyn DimerGate + 'g)>) -> bool {
    let Some(g) = gate else {
        t.provisional = true;
        t.record.dimer = "provisional (dimer gate not run)".into();
        return true;
    };
    match g.dimers(t) {
        Ok(p) => {
            let (dr, dw) = (p.pp.0 - p.ae.0, p.pp.1 / p.ae.1 - 1.0);
            let ok = dr.abs() < LEDGER.dimer_re && dw.abs() < LEDGER.dimer_omega_rel;
            t.record.dimer = format!("{}: PP R_e {:.4} Bohr, ω {:.6} a.u. vs AE σ→0 R_e {:.4} Bohr, ω {:.6} a.u. (ΔR_e {:+.4} Bohr, ω ratio − 1 {:+.4}; bounds {}, {})",
                                     if ok { "pass" } else { "fail" }, p.pp.0, p.pp.1, p.ae.0, p.ae.1, dr, dw, LEDGER.dimer_re, LEDGER.dimer_omega_rel);
            ok
        }
        Err(DimerError::NotApplicable(why)) => { t.record.dimer = format!("not applicable: {why}"); true }
        Err(DimerError::PseudoUnbound(why)) => { t.record.dimer = format!("fail: the pseudised dimer is unbound where the all-electron one is bound: {why}"); false }
        Err(DimerError::WallMismatch(why)) => { t.record.dimer = format!("fail: no AE minimum in the bracket, and the pseudised repulsive wall disagrees: {why}"); false }
        Err(DimerError::Failed(why)) => { t.record.dimer = format!("fail: the dimer gate could not run: {why}"); false }
    }
}

/// One ladder entry: TM channels at f, unscreened with the partial core, the KB form with `local`, G1/G2,
/// then T1–T7 (`gates::run_gates`).
pub(crate) fn candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, pc: &PartialCore) -> Result<PseudoTable, String> {
    let mut t = ghost_checked_candidate(reference, part, f, local, pc)?;
    t.gates = run_gates(&t)?;
    t.record = record_of(&t, 0.0);
    Ok(t)
}

/// The construction and G1/G2 only (Task 3's stage): `gates` and `record` are empty defaults. For tests of
/// the construction; `generate` never returns such a table.
pub(crate) fn ghost_checked_candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, pc: &PartialCore) -> Result<PseudoTable, String> {
    let g = &reference.grid;
    let channels = tm_channels(reference, part, f)?;
    let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
    let un = unscreen(g, part.z_v, &channels, &occ, &pc.rho_pc);
    let screen: Vec<f64> = (0..g.len()).map(|i| channels[0].v_scr[i] - un.v_ion[0][i]).collect();
    // τ_c from the *full* core (never ρ_pc, Task 2 review S3): N_c = Z − Z_v
    let tau_c = core_tail(g, &core_density(reference, part), part.n_core as f64);
    let kb = kb_form(g, part.z, part.z_v, &channels, &un, local, pc.rho_pc.clone(), &tau_c);
    let ghost = ghost_gates(g, &kb, &screen, part)?;
    Ok(PseudoTable { z: part.z, z_v: part.z_v, n_core: part.n_core, f, local, kb, channels, screen,
                     partition: part.clone(), reference: reference.clone(), grid: g.clone(), ghost,
                     partial_core: pc.clone(), gates: GateReport::default(), record: GenerationRecord::default(), provisional: false })
}

#[cfg(test)]
mod tests {
    use super::ghost_checked_candidate;
    use crate::radial::nlcc::partial_core;
    use crate::radial::partition::valence_density;
    use crate::radial::kb::{core_tail, kb_count_below, kb_form, reference_config, solve_kb_atom, GssVerdict};
    use crate::radial::numerov::bound_state;
    use crate::radial::partition::{core_density, partition, Refusal};
    use crate::radial::tm::{tm_channels, unscreen};
    use crate::radial::{ground_state, LocalCandidate, SpinMode, LEDGER};
    use std::sync::Arc;

    /// Task 3 review, ruling (c): the polynomial-local O at f = 1.6 is a case-B ghost — E_KB(s) < 0 (the
    /// near-singular −415.6 Ha) with ε₀^loc < ε_s on the reference screening — and G1 refuses it. For
    /// E_KB < 0 the ghost-free window is ε_l < ε₀^loc ≤ ε_l + 1/|E_KB| (variational), 2.4 mHa wide here.
    /// The ruling's case is LFC's c = 1 core (Task 3); at the recipe's crossover (-4, c = ¼) the same candidate is still a
    /// ghost and refused, by G2's inertia count (a third s state at −0.606 Ha, measured) instead of G1.
    #[test]
    fn polynomial_local_oxygen_is_a_case_b_ghost() {
        let reference = Arc::new(ground_state(8, SpinMode::Restricted).unwrap());
        let part = partition(&reference).unwrap();
        let rho_c = core_density(&reference, &part);
        let at_ledger = partial_core(&reference.grid, &rho_c, &valence_density(&reference, &part)).unwrap();
        let e = ghost_checked_candidate(&reference, &part, 1.6, LocalCandidate::Polynomial, &at_ledger).expect_err("a ghost at the recipe's c");
        assert!(e.starts_with("G1: ghost") || e.starts_with("G2:"), "{e}");
        let pc = crate::radial::partial_core_at(&reference.grid, &rho_c, &valence_density(&reference, &part), 1.0).unwrap();
        let err = ghost_checked_candidate(&reference, &part, 1.6, LocalCandidate::Polynomial, &pc).expect_err("a case-B ghost");
        assert!(err.starts_with("G1: ghost in l = 0"), "{err}");
        let g = &reference.grid;
        let chans = tm_channels(&reference, &part, 1.6).unwrap();
        let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
        let un = unscreen(g, part.z_v, &chans, &occ, &pc.rho_pc);
        let kb = kb_form(g, 8, part.z_v, &chans, &un, LocalCandidate::Polynomial, pc.rho_pc.clone(), &core_tail(g, &rho_c, part.n_core as f64));
        let screen: Vec<f64> = (0..g.len()).map(|i| chans[0].v_scr[i] - un.v_ion[0][i]).collect();
        let v: Vec<f64> = kb.v_loc.iter().zip(&screen).map(|(a, b)| a + b).collect();
        let e0 = bound_state(g, &v, 0, 0, 0.0, None).unwrap().e;
        let (e_kb, eps_s) = (kb.projectors[0].e_kb, part.channels[0].eps);
        eprintln!("O f = 1.6 polynomial: E_KB(s) {e_kb:.4}, ε₀^loc {e0:.6} < ε_s {eps_s:.6}; {err}");
        assert!(e_kb < 0.0 && e0 < eps_s, "case B: E_KB {e_kb}, ε₀^loc {e0}, ε_s {eps_s}");
    }

    /// Round 1 regression: the inertia count is monotone up to ε = 0. The review's probe read the
    /// polynomial-local Al at f = 1.0 as missing the AE 4s (count 1 vs 2 at 0); the count was in fact
    /// falling from 2 to 1 between −1e-3 and −1e-4 Ha — impossible — because u_i's WKB-ratio start was not the
    /// Dirichlet solution n_loc's Sturm count assumes once the decay length exceeds the box. With the
    /// Dirichlet start the KB s channel binds the 4s (G2 roots 3s and ≈ the AE 4s), counts 2 below 0 on a
    /// log sweep up to 0, and nothing is missing.
    #[test]
    fn inertia_count_is_monotone_to_the_continuum_edge() {
        let reference = Arc::new(ground_state(13, SpinMode::Restricted).unwrap());
        let part = partition(&reference).unwrap();
        let pc = partial_core(&reference.grid, &core_density(&reference, &part), &valence_density(&reference, &part)).unwrap();
        let t = ghost_checked_candidate(&reference, &part, 1.0, LocalCandidate::Polynomial, &pc).expect("G1 and G2 pass");
        let (ae4s, roots) = (part.spectrum[0][1], &t.ghost.g2_roots[0].1);
        eprintln!("Al f = 1.0 polynomial: G2 s roots {roots:?} (AE 4s {ae4s:.7}); g2_missing {:?}", t.ghost.g2_missing);
        assert!(t.ghost.g2_missing.is_empty());
        assert_eq!(roots.len(), 2);
        let g = &t.grid;
        let v: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
        let p = &t.kb.projectors[0];
        let mut last = 0;
        for e in [-0.5, -0.1, -0.02, -0.0125, -0.0121, -0.012, -0.01, -1e-3, -1e-4, -1e-5, -1e-6, -1e-8, 0.0] {
            let c = kb_count_below(g, &v, Some((p.e_kb, &p.beta)), 0, e);
            assert!(c >= last, "count fell from {last} to {c} at ε = {e}");
            last = c;
        }
        assert_eq!(last, 2);
    }

    /// A gate of fixed answers for the mechanics test: `answers[k]` answers the k-th call; records the (f, local) asked.
    struct Scripted { answers: Vec<Result<super::DimerPair, super::DimerError>>, asked: Vec<(f64, LocalCandidate)> }
    impl super::DimerGate for Scripted {
        fn dimers(&mut self, t: &super::PseudoTable) -> Result<super::DimerPair, super::DimerError> {
            self.asked.push((t.f, t.local));
            self.answers.get(self.asked.len() - 1).cloned().unwrap_or_else(|| panic!("asked {} times", self.asked.len()))
        }
    }

    /// Task 11 (spec §4.4, plan review M6): the dimer gate's place in the ladder, on scripted answers (the 3-D gate is
    /// the providers'; `providers::pseudo` tests it on the GPU). Without a gate H is provisional; a failing answer moves
    /// the ladder to the next entry that passes the atomic gates, and the accepted entry's record says "pass" and is
    /// usable; "not applicable" accepts the first entry; a gate that cannot run, or a pseudised dimer that is unbound,
    /// fails every entry (refused, with the reason); for Z above `dimer_z_max` the gate is never asked.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only (several H ladders); cargo test -p qm --release")]
    fn the_dimer_gate_runs_inside_the_ladder() {
        use super::{generate, generate_with, DimerError, DimerPair};
        let ungated = generate(1).expect("H passes the atomic gates");
        assert!(ungated.provisional && ungated.record.dimer == "provisional (dimer gate not run)", "{}", ungated.record.dimer);
        let same = DimerPair { pp: (1.45, 0.02), ae: (1.45, 0.02) };
        let long = DimerPair { pp: (1.45 + 2.0 * LEDGER.dimer_re, 0.02), ae: (1.45, 0.02) };
        // fail the first entry the gate sees, pass the next
        let mut g = Scripted { answers: vec![Ok(long), Ok(same)], asked: Vec::new() };
        let t = generate_with(1, Some(&mut g)).expect("H passes at a later entry");
        assert_eq!(g.asked.len(), 2);
        assert_eq!(g.asked[0], (ungated.f, ungated.local), "the gate is asked first at the entry the atomic gates accept");
        assert_eq!((t.f, t.local), g.asked[1]);
        assert!(!t.provisional && t.record.dimer.starts_with("pass: "), "{}", t.record.dimer);
        assert!(!t.record.filter.is_empty(), "T8 ran on the accepted entry");
        // not applicable: the first entry is accepted and usable
        let mut g = Scripted { answers: vec![Err(DimerError::NotApplicable("no force zero".into()))], asked: Vec::new() };
        let t = generate_with(1, Some(&mut g)).expect("not applicable accepts");
        assert_eq!((t.f, t.local, t.provisional), (ungated.f, ungated.local, false));
        assert_eq!(t.record.dimer, "not applicable: no force zero");
        // a gate that cannot run, or an unbound pseudised dimer, fails every entry: refused with the reason
        let mut g = Scripted { answers: (0..16).map(|_| Err(DimerError::Failed("device lost".into()))).collect(), asked: Vec::new() };
        let e = generate_with(1, Some(&mut g)).expect_err("every entry fails");
        assert!(e.to_string().contains("dimer gate fail: the dimer gate could not run: device lost"), "{e}");
        let mut g = Scripted { answers: (0..16).map(|_| Err(DimerError::PseudoUnbound("F > 0 throughout".into()))).collect(), asked: Vec::new() };
        let e = generate_with(1, Some(&mut g)).expect_err("every entry fails");
        assert!(e.to_string().contains("unbound where the all-electron one is bound"), "{e}");
        // above the bound the gate is never asked and the table is usable
        let mut g = Scripted { answers: Vec::new(), asked: Vec::new() };
        let o = generate_with(8, Some(&mut g)).expect("O generates");
        assert!(g.asked.is_empty() && !o.provisional && !generate(8).unwrap().provisional);
        assert!(o.record.dimer.starts_with("not applicable (Z above"), "{}", o.record.dimer);
    }

    /// C–Ne generate on the ladder with every gate (G1, G2, T1–T7 since Task 4; measured: C f = 1.2
    /// polynomial local, N 1.2, O 1.4, F 1.4, Ne 1.6 — T5 moves C–F off f = 1.6, `pseudo_gen` pins the
    /// records), N_c = 2, f from the ladder.
    #[test]
    fn second_row_generates_on_the_ladder() {
        for z in 6u8..=10 {
            let t = super::generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
            assert!(LEDGER.f_ladder.contains(&t.f));
            assert_eq!(t.n_core, 2);
            eprintln!("Z = {z}: f = {}, local {:?}, r_c = {:?}", t.f, t.local, t.channels.iter().map(|c| (c.l, c.r_c)).collect::<Vec<_>>());
        }
    }

    /// Task 3's stage of the ladder — the first (f, local) whose construction passes G1 and G2, with the
    /// LFC partial core — for the report below. T1–T7 refuse some of these elements (Task 4: T5 refuses Be,
    /// B and Al–Cl at every entry); that outcome is `generate`'s and is pinned by `pseudo_gen`'s generation
    /// records, not here.
    fn ghost_ladder(z: u8) -> Result<super::PseudoTable, Refusal> {
        let reference = Arc::new(ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?);
        let part = partition(&reference)?;
        let pc = partial_core(&reference.grid, &core_density(&reference, &part), &valence_density(&reference, &part))
            .map_err(|message| Refusal::PartialCore { z, message })?;
        let mut tried = Vec::new();
        for &f in LEDGER.f_ladder.iter() {
            for &local in LEDGER.local_order.iter() {
                match ghost_checked_candidate(&reference, &part, f, local, &pc) {
                    Ok(t) => return Ok(t),
                    Err(why) => tried.push(format!("f = {f}, {local:?}: {why}")),
                }
            }
        }
        Err(Refusal::NoLadderEntryPasses { z, tried })
    }

    /// The ladder over H–Ar at Task 3's gates (G1, G2; `ghost_ladder`), reported for the physics review and
    /// pinned: Li, Na and Mg are refused by the partition (two valence s states); every other Z passes G1/G2
    /// (a refusal fails the test, to be seen in review); every table has its β supports within the first
    /// grid point beyond max(r_c, r_nlc) (the S3 consequence, whatever the local); G2's root count equals the
    /// all-electron spectrum above the core below its window (Partition::spectrum) for every l ≤ l_max.
    /// Printed per element: f, local, r_c, r_loc, r_cmp, each projector's E_KB, GSS and β support, the G1
    /// verdict with ε₀^loc, ε₁^loc on the reference screening, the KB atom's eigenvalue per occupied channel
    /// against the reference, and the G2 roots against the AE spectrum; for a refusal, every tried entry.
    #[test]
    fn ladder_report_h_through_ar() {
        for z in 1u8..=18 {
            let t = match ghost_ladder(z) {
                Ok(t) => t,
                Err(e @ Refusal::TwoStatesSameL { .. }) => {
                    assert!([3u8, 11, 12].contains(&z), "Z = {z}: unexpected partition refusal {e}");
                    eprintln!("Z = {z:2}: refused by the partition: {e}");
                    continue;
                }
                // every Z the partition accepts generates at this task's gates (measured: all at f = 1.6 with
                // the highest-channel local); a ladder refusal is an outcome for review, so it fails here
                Err(e) => panic!("Z = {z}: refused by the ladder: {e}"),
            };
            assert!(![3u8, 11, 12].contains(&z), "Z = {z} must be refused by the partition");
            let g = &t.grid;
            let i_from = t.channels.iter().map(|c| c.i_c).max().unwrap().max(t.partial_core.r_nlc.map_or(0, |r| g.nearest(r))) + 1;
            eprintln!("Z = {z:2}: f = {}, local {:?} (l {:?}), Z_v {}, N_c {}, r_c {:?}, r_loc {:.4}, r_cmp {:.4}, first point beyond max r_c {:.4}",
                      t.f, t.local, t.kb.local_l, t.z_v, t.n_core, t.channels.iter().map(|c| (c.l, (c.r_c * 1e4).round() / 1e4)).collect::<Vec<_>>(),
                      t.kb.r_loc, t.kb.r_cmp, g.r[i_from]);
            let v_ref: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
            for p in &t.kb.projectors {
                let e0 = bound_state(g, &v_ref, p.l, 0, 0.0, None).map(|b| b.e);
                let e1 = bound_state(g, &v_ref, p.l, 1, 0.0, None).map(|b| b.e);
                let verdict = &t.ghost.g1.iter().find(|(l, _)| *l == p.l).unwrap().1;
                let eps = t.partition.channels.iter().find(|c| c.l == p.l).unwrap().eps;
                eprintln!("    projector l = {}: E_KB {:+.6} Ha, GSS {:+.6} Ha, β support {:.4} Bohr; G1 {} (ε_l {eps:.6}, ε₀^loc {e0:?}, ε₁^loc {e1:?})",
                          p.l, p.e_kb, p.gss, p.support(g), if *verdict == GssVerdict::GhostFree { "ghost-free" } else { "GHOST" });
                assert!(p.support(g) <= g.r[i_from], "Z = {z}, l = {}: β support {:.4} beyond {:.4}", p.l, p.support(g), g.r[i_from]);
            }
            assert!(t.ghost.g2_missing.is_empty(), "Z = {z}: AE states the pseudo-atom misses: {:?}", t.ghost.g2_missing);
            let atom = solve_kb_atom(g, &t.kb.input(), &reference_config(&t.partition), SpinMode::Restricted).unwrap();
            for c in t.partition.channels.iter().filter(|c| c.occupied) {
                let e = atom.eps[c.l as usize][0];
                eprintln!("    KB atom l = {}: {e:.10} vs reference {:.10} (Δ {:+.2e})", c.l, c.eps, e - c.eps);
            }
            eprintln!("    G2: max |Δε| {:.2e}; KB atom energy {:.8} Ha", t.ghost.g2_max_eig_error, atom.energy);
            let top = (t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max) + LEDGER.gates.g2_window).min(0.0);
            for (l, roots) in &t.ghost.g2_roots {
                let ae: Vec<f64> = t.partition.spectrum[*l as usize].iter().copied().filter(|&e| e < top).collect();
                let proj = t.kb.projectors.iter().find(|p| p.l == *l).map(|p| (p.e_kb, p.beta.as_slice()));
                assert_eq!(roots.len(), ae.len(), "Z = {z}, l = {l}");
                assert_eq!(kb_count_below(g, &atom.v[0], proj, *l, top), ae.len(), "Z = {z}, l = {l}");
                let d: Vec<String> = roots.iter().zip(&ae).map(|(a, b)| format!("{a:.6} (AE {b:.6}, Δ {:+.1e})", a - b)).collect();
                eprintln!("    G2 l = {l}: {} root(s) below {top:.3} Ha: {d:?}", roots.len());
            }
        }
    }
}
