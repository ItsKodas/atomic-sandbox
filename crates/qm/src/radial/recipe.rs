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
//! - Gates at this task: G1 and G2 (`kb::ghost_gates`). Task 4 adds T1–T7 to `candidate` (and the LFC
//!   partial core in place of the full core); Task 11 the dimer gate.
//! - Molecular gate on f (review S6b; Task 11 implements it, not this task): for any element whose
//!   all-electron σ → 0 reference is affordable (Z ≤ `LEDGER.dimer_z_max`), the homonuclear dimer against
//!   the element's own all-electron curve (R_e within `dimer_re`, ω within `dimer_omega_rel`) is an
//!   additional gate on f, in the same descending order. It is a 3-D computation supplied by the
//!   providers. A table for such an element generated without it is provisional and no ionic setup may
//!   use it, so the ungated and gated potentials of one element never share a `recipe_hash` (§4.9, §11
//!   M6). The bracket for the dimer minimum is `dimer_bracket`·r_peak of the outermost valence shell.
//! - The ladder's outputs (Z → f, local, r_c,l, Z_v, E_KB, GSS, h_req) are pinned in `references.md` by
//!   the validation task, so a change to any universal constant that moves an element shows in review.
//!
//! # Inputs fixed here
//! The reference is the spin-restricted ground state (spec §4.3), PW92. At Task 3 the partial core is the
//! full core (Task 2's r_nlc → 0 limit); τ_c always uses the full core (ρ_pc never enters V_H).
use super::atom::{ground_state, AtomResult, SpinMode};
use super::grid::LogGrid;
use super::kb::{core_tail, ghost_gates, kb_form, GhostReport, KbPotential};
use super::ledger::{LocalCandidate, LEDGER};
use super::partition::{core_density, partition, Partition, Refusal};
use super::tm::{tm_channels, unscreen, TmChannel};
use std::sync::Arc;

/// A generated, gated pseudopotential with everything it was built from. `screen` is the reference
/// screening V_H[ρ_v^PP] + v_xc[ρ_v^PP + ρ_pc] (= V_l^scr − V_l^ion, the same for every l).
#[derive(Clone, Debug)]
pub struct PseudoTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub f: f64, pub local: LocalCandidate, pub kb: KbPotential,
    pub channels: Vec<TmChannel>, pub screen: Vec<f64>, pub partition: Partition,
    pub reference: Arc<AtomResult>, pub grid: LogGrid, pub ghost: GhostReport,
}

/// Z in, potential out (spec §4.4): f from `LEDGER.f_ladder` softest first, local candidates in
/// `LEDGER.local_order`; the first (f, local) passing every gate is kept; none → refused at this recipe
/// version with the reasons listed.
pub fn generate(z: u8) -> Result<PseudoTable, Refusal> {
    let reference = Arc::new(ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?);
    let part = partition(&reference)?;
    // Task 3: the full core as the partial core (Task 2's r_nlc → 0 limit); Task 4 replaces this line by
    // nlcc::partial_core (LFC, spec §4.5)
    let rho_pc = core_density(&reference, &part);
    let mut tried = Vec::new();
    for &f in LEDGER.f_ladder.iter() {
        for &local in LEDGER.local_order.iter() {
            match candidate(&reference, &part, f, local, &rho_pc) {
                Ok(t) => return Ok(t),
                Err(why) => tried.push(format!("f = {f}, {local:?}: {why}")),
            }
        }
    }
    Err(Refusal::NoLadderEntryPasses { z, tried })
}

/// One ladder entry: TM channels at f, unscreened with `rho_pc`, the KB form with `local`, the gates.
pub(crate) fn candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, rho_pc: &[f64]) -> Result<PseudoTable, String> {
    let g = &reference.grid;
    let channels = tm_channels(reference, part, f)?;
    let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
    let un = unscreen(g, part.z_v, &channels, &occ, rho_pc);
    let screen: Vec<f64> = (0..g.len()).map(|i| channels[0].v_scr[i] - un.v_ion[0][i]).collect();
    // τ_c from the *full* core (whatever the partial core): N_c = Z − Z_v
    let tau_c = core_tail(g, &core_density(reference, part), part.n_core as f64);
    let kb = kb_form(g, part.z, part.z_v, &channels, &un, local, rho_pc.to_vec(), &tau_c);
    let ghost = ghost_gates(g, &kb, &screen, part)?;
    Ok(PseudoTable { z: part.z, z_v: part.z_v, n_core: part.n_core, f, local, kb, channels, screen,
                     partition: part.clone(), reference: reference.clone(), grid: g.clone(), ghost })
}

#[cfg(test)]
mod tests {
    use crate::radial::kb::{kb_count_below, reference_config, solve_kb_atom, GssVerdict};
    use crate::radial::numerov::bound_state;
    use crate::radial::partition::Refusal;
    use crate::radial::{SpinMode, LEDGER};

    /// Task 3 ladder (G1 + G2 only; Task 4 adds T1–T7): C–Ne generate, nothing refused, f from the ladder.
    #[test]
    fn second_row_generates_on_the_ladder() {
        for z in 6u8..=10 {
            let t = super::generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
            assert!(LEDGER.f_ladder.contains(&t.f));
            assert_eq!(t.n_core, 2);
            eprintln!("Z = {z}: f = {}, local {:?}, r_c = {:?}", t.f, t.local, t.channels.iter().map(|c| (c.l, c.r_c)).collect::<Vec<_>>());
        }
    }

    /// The ladder over H–Ar at this task's gates (G1, G2), reported for the physics review and pinned:
    /// Li, Na and Mg are refused by the partition (two valence s states); every other Z generates (a
    /// ladder refusal fails the test, to be seen in review); every generated table has its β supports within the first grid point beyond max r_c (the
    /// S3 consequence, whatever the local); G2's root count equals the all-electron spectrum above the
    /// core below its window (Partition::spectrum) for every l ≤ l_max. Printed per element: f, local,
    /// r_c, r_loc, r_cmp, each projector's E_KB, GSS and β support, the G1 verdict with ε₀^loc, ε₁^loc on
    /// the reference screening, the KB atom's eigenvalue per occupied channel against the reference, and
    /// the G2 roots against the AE spectrum; for a refusal, every tried entry.
    #[test]
    fn ladder_report_h_through_ar() {
        for z in 1u8..=18 {
            let t = match super::generate(z) {
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
            let i_from = t.channels.iter().map(|c| c.i_c).max().unwrap() + 1;
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
