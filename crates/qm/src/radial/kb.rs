//! The Kleinman–Bylander separable form, the radial KB atom and the ghost-state gates G1/G2 (spec §4.4
//! item 2, §4.6, §11 Q4/Q5; Kleinman & Bylander, Phys. Rev. Lett. 48, 1425 (1982); Gonze, Stumpf &
//! Scheffler, Phys. Rev. B 44, 8503 (1991)). CPU, f64, atomic units: V is the electron's potential
//! energy (Hartree), u = rR in u-form with ∫u² dr = 1, ε < 0 for a bound state. Every radial operator
//! below acts on one angular momentum l; "V" without a subscript is the channel's full local potential
//! V_loc + screening (the centrifugal l(l+1)/2r² is added by `numerov::g_coeff`).
//!
//! # The local candidates (spec §4.4; `LEDGER.local_order`)
//! 1. `HighestChannel`: V_loc = V_{l_top}^ion, l_top the highest *generated* channel, `channels.last()`
//!    (not `Partition::l_max`, which counts dropped channels — Task 2 review S2).
//! 2. `Polynomial` (Task 2 review S3): V_loc(r) = T(r) := −Z_v/r + τ_c(r) for r ≥ r_loc, τ_c = V_H[ρ_c] −
//!    N_c/r the frozen core's Hartree tail (spec §4.3; ≡ 0 without a core; [`core_tail`]), r_loc = min_l
//!    r_c,l (a grid point); inside, the even quartic a₀ + a₂r² + a₄r⁴ matched C² at r_loc. With V′ = 2a₂r +
//!    4a₄r³ and V″ = 2a₂ + 12a₄r², V″ − V′/r = 8a₄r², so at r = r_loc
//!      a₄ = (T″ − T′/r)/(8r²),  a₂ = (T′ − 4a₄r³)/(2r),  a₀ = T − a₂r² − a₄r⁴.
//!    For τ_c = 0 (T′ = Z_v/r², T″ = −2Z_v/r³) this is a₄ = −3Z_v/(8r⁵), a₂ = 5Z_v/(4r³), a₀ = −15Z_v/(8r).
//!    Why τ_c: every V_l^ion is −Z_v/r + τ_c beyond max r_c (spec §4.3), so with this tail δV_l vanishes
//!    there to the unscreening identity's residual (≈ 1e-12) and β_l is as compact as under the
//!    highest-channel local; a −Z_v/r tail alone leaves δV_l = τ_c and spreads each β (the review's probe:
//!    O 1.41 → 1.97 Bohr, Al 4.25 → 5.58; measured here at f = 1.6, s projector: O 1.39 → 2.14, Al 4.18 →
//!    `polynomial_local_carries_the_core_tail`). Interface: τ_c is passed as a grid table (the
//!    caller builds it from the *full* core density — ρ_pc never enters V_H); T′ and T″ at r_loc are its
//!    5-point grid derivatives (`LogGrid::derivative`, O(δ⁴); τ_c is smooth).
//!
//! # The KB form (spec §4.6, review S1)
//! For every generated channel l other than the local one: δV_l = V_l^ion − V_loc, β_l = δV_l·u_l^PP
//! (u-form), E_l^KB = 1/∫u_l δV_l u_l dr, and the radial operator V_NL = E_l^KB |β_l⟩⟨β_l|. On the
//! reference state V_NL u_l = E_KB β_l ⟨β_l|u_l⟩ = β_l = δV_l u_l: the semilocal operator's action,
//! exactly. V_NL is invariant under u → λu (β → λβ, E_KB → E_KB/λ²); ∫u² = 1 (the TM u^PP is norm-
//! conserving to the normalised u^AE) pins the reported E_KB. The GSS energy ∫β²/∫uβ = ⟨u|δV²|u⟩/⟨u|δV|u⟩
//! has E_KB's sign. Task 6's grid β_lm carries √((2l+1)/4π) so that it is the same operator in 3-D.
//!
//! # The radial KB equation (integro-differential; spec §4.6)
//!   −½u″ + [l(l+1)/2r² + V(r)]u + E_KB β⟨β|u⟩ = εu.
//! With c = E_KB⟨β|u⟩ this is −½u″ + (V_eff − ε)u = −cβ. For r ≤ r_m (β ≡ 0 beyond, to 1e-12 relative)
//! u = a·u_h + c·u_p: u_h the regular homogeneous solution, u_p the regular particular solution of
//! −½u_p″ + (V_eff − ε)u_p = −β (Numerov with the source S = 2r^{3/2}β, `numerov::outward_sourced`); beyond,
//! u = b·u_i, u_i decaying. Continuity of value and slope at r_m, with b eliminated, gives
//! a·W(u_h,u_i) + c·W(u_p,u_i) = 0 (W(f,g) = fg′ − f′g); self-consistency c = E_KB⟨β|a·u_h + c·u_p⟩ gives
//! a·E_KB⟨β|u_h⟩ + c·(E_KB⟨β|u_p⟩ − 1) = 0. Hence M (a, c)ᵀ = 0 with
//!   M = [[W(u_h,u_i), W(u_p,u_i)], [E_KB⟨β|u_h⟩, E_KB⟨β|u_p⟩ − 1]],
//! entries finite at every ε (no poles); the KB eigenvalues are the zeros of det M. W is taken in x = ln r
//! (W_x = r·W_r, the factor common to row 1). r_m (`matching_index`) is fixed per projector: an
//! ε-dependent r_m would make det M jump.
//!
//! # The Green-function form and the exact count (plan review Q5)
//! G = (H_loc − ε)⁻¹ (regular at 0, decaying): since (H_loc − ε)u_p = −β, Gβ = −(u_p − κu_h) with
//! κ = W(u_p,u_i)/W(u_h,u_i) (the combination with the decaying tail). Therefore
//!   s(ε) := −1/E_KB − ⟨β|G|β⟩ = −1/E_KB + ⟨β|u_p⟩ − κ⟨β|u_h⟩,   E_KB·W(u_h,u_i)·s(ε) = det M.
//! s is invariant under rescaling u_h or u_i (det M and W(u_h,u_i) scale alike), so the positive rescalings
//! of the Numerov runs (overflow guards, the ε-dependent inward start) never flip it. det M is continuous:
//! at a local eigenvalue W(u_h,u_i) → 0 while s has a pole, with a finite product. Haynsworth inertia
//! additivity on the bordered operator B = [[H_loc − ε, β], [βᵀ, −1/E_KB]]: the Schur complement of the
//! corner is H_loc − ε + E_KB ββᵀ = H_KB − ε, that of H_loc − ε is s(ε), so
//!   n₋(B) = [E_KB > 0] + n_KB(ε) = n_loc(ε) + [s(ε) < 0]   ⇒   n_KB(ε) = n_loc(ε) + [s(ε) < 0] − [E_KB > 0],
//! n_loc(ε) the local channel's eigenvalue count below ε (Sturm: the nodes of u_h on (0, r_ie), r_ie =
//! `shooting_end`; it counts the Dirichlet problem on [0, r_ie], so u_i must vanish at r_ie — `green` starts
//! it there, round 1: a WKB-ratio start broke monotonicity near ε = 0). It is exact up to the two discretisations of each local eigenvalue
//! (the node jump and W's zero, each bisected to 1e-13), which differ by ≤ 2e-12 Ha (review probe: blips of
//! that width in the count at C, O, Al, Ar local levels — harmless unless a local eigenvalue sits within
//! 1e-12 of a G2 window top or a bisection midpoint lands in one). A mesh of det M can miss two roots
//! inside one cell, so the mesh only locates roots (for the eigenvalue comparison) and G2 asserts that
//! both agree.
//!
//! # Stability: det M is evaluated through the Green-function kernel
//! The brief's M is the right function but the wrong numerics at deep energies. u_h and u_p both grow as
//! e^{κr} (κ = √(2(V − ε))) through β's support, and det M = −W(u_h,u_i) + E_KB[W(u_h,u_i)⟨β|u_p⟩ −
//! W(u_p,u_i)⟨β|u_h⟩] cancels their common growing part: a relative loss ≈ e^{2κ r_c}. Measured on the
//! review's ghost case B (O s, a −562 Ha well): det changes sign hundreds of times near −592 Ha, where
//! there is no eigenvalue (e^{2κr_c} ≈ e^{95}). The same function, from the kernel G(r, r′) =
//! −(2/W)u_h(r<)u_i(r>) of (H_loc − ε)⁻¹ (W = W_r(u_h, u_i); check: u = (2/W)[u_i∫₀^r u_h f + u_h∫_r^∞ u_i f]
//! has −½u″ + (V_eff − ε)u = −f), is
//!   ⟨β|G|β⟩ = −(4/W)∫₀^∞ β u_i C_h dr,  C_h(r) = ∫₀^r β u_h dr′,
//!   E_KB·W·s(ε) = −W + 4E_KB∫β u_i C_h dr   (= det M/r_m, M's first row being r_m·W_r in x),
//! which contains only the products u_h(r′)u_i(r), r′ ≤ r, bounded at any depth. `kb_green` evaluates it
//! (u_i run inward to the first grid point); `kb_eigenvalues` and `kb_count_below` use it, and `kb_state`
//! builds the eigenvector from the same kernel, u ∝ Gβ (M's null vector a·u_h + c·u_p cancels the same
//! growing parts, and at a root of the stable form M's two rows disagree by the discretisation —
//! measured 1.1e-5 on the oscillator). W itself is constant but is taken where the solutions are
//! smoothest in x (least |g|): at r_m they change by e^{±√g δ} per step and the 5-point derivative's
//! relative error (√g δ)⁴/30 reached 3e-3 on the oscillator — harmless inside M (row 1 carries it in both
//! entries), not in s. `kb_regular` (Task 4's T5, at r_d ≈ 1.2 r_c with a compact β) keeps M's second row,
//! as the brief specifies (plan review Q12).
//!
//! # GSS (G1), re-derived from interlacing, and Spec question 4
//! A rank-one perturbation interlaces: for E_KB > 0, ε_k^loc ≤ ε_k^KB ≤ ε_{k+1}^loc; for E_KB < 0,
//! ε_{k−1}^loc ≤ ε_k^KB ≤ ε_k^loc. The reference ε_l is a KB eigenvalue by construction. A ghost (GSS's
//! sense) is a KB level below ε_l.
//! - E_KB < 0: if ε_l < ε₀^loc then ε₁^KB ≥ ε₀^loc > ε_l, so ε_l = ε₀^KB, no ghost; if ε₀^loc < ε_l then
//!   ε₀^KB ≤ ε₀^loc < ε_l, a ghost.
//! - E_KB > 0: if ε_l < ε₁^loc then ε₁^KB ≥ ε₁^loc > ε_l, so ε_l = ε₀^KB, no ghost; if ε₁^loc < ε_l then
//!   ε₀^KB ≤ ε₁^loc < ε_l, a ghost.
//! Hence GSS: ghost-free iff (E_KB < 0 ∧ ε_l < ε₀^loc) ∨ (E_KB > 0 ∧ ε₀^loc < ε_l < ε₁^loc); an unbound
//! local level is the continuum edge, 0. The variational principle restricts the failures (Spec question
//! 4, plan review Q4): ⟨φ|H_loc|φ⟩ = ε_l − ⟨φ|δV|φ⟩ ≥ ε₀^loc, so ε₀^loc > ε_l forces E_KB < 0 (ghost-free)
//! and E_KB > 0 forces ε₀^loc < ε_l. The two constructible ghosts are (A) E_KB > 0 with ε₁^loc < ε_l and
//! (B) E_KB < 0 with ε₀^loc < ε_l; in both the interlacing above puts a KB level below ε_l, so G2's exact
//! count at ε_l + window is ≥ 2 where the channel expects 1 (`deliberate_bad_local_is_caught_as_ghost`).
//!
//! # The radial KB atom (G2; the T-gates of Task 4)
//! SCF on ρ_v (spin densities, Pulay/DIIS of `atom`, the same tolerance and measure): per channel and spin
//! the lowest KB eigenvalue (`lowest`: bisection on the exact count) of V_σ = V_loc + V_H[ρ_v] +
//! v_xc,σ[ρ_v + ρ_pc] (each spin ½ρ_pc), occupied per `ValenceConfig`. Start ρ_in = 0, as `atom` does
//! (deviation from the brief, which started from the bare pseudo-ion's full output density: it
//! over-screens and unbinds the p shell of C–Ne at step 1, measured). At the reference configuration the screening is the unscreening's own, so the TM
//! states are self-consistent and the eigenvalues are the reference ones. Energy (output density, the
//! `scf.rs` discipline): T_s + E_NL = Σfε − Σ_σ∫ρ_σ V_σ^in, E = T_s + E_NL + ∫ρV_loc + E_H + E_xc[ρ_v + ρ_pc]
//! (Neumaier sums, rule 4).
//!
//! # G2 (spec §4.6, plan review Q5)
//! The SCF KB atom at the reference configuration reproduces each occupied channel's eigenvalue to
//! `LEDGER.gates.g2_eig`, and for every l ≤ `Partition::l_max` the exact count below top = min(0, ε_v,max
//! + `g2_window`) does not exceed the number of the *all-electron* reference's bound states of that l above
//! its core below top (`Partition::spectrum`); the `g2_mesh` roots must number the count (else the mesh
//! missed a pair: reported as a numerical miss, not a ghost verdict). More than the AE atom is a ghost
//! (GSS's sense: a state the separable form adds) and refuses; fewer is an AE state the pseudo-atom does
//! not bind — a transferability miss at the continuum edge, where a count has no energy tolerance — and is
//! recorded in `GhostReport::g2_missing` for T5 (window to ε_v,max + 1 Ha) and the generation record
//! (Task 3 review, ruling (a)/S1; measured: polynomial-local Al and Si at f = 1.0 miss the AE 4s). A dropped
//! l (no bound AE state) must hold none. Deviation from the plan's brief, which expected the generated
//! channels of l only.
//! The (n+1)s (rule 7): the neutral LSDA potential is short-range (no −1/r tail, self-interaction), so it
//! has no Rydberg series; for l = 0 (no centrifugal barrier) a well as deep as an atom's still binds one
//! level more than it occupies — C 3s −0.0061, O 3s −0.0045, Ne 3s −0.0024, Al 4s −0.0121, Si 4s −0.0138 Ha,
//! unchanged to ≤ 3e-5 Ha with the grid continued from 60 to 480 Bohr (review probe; Be 3s, decay length
//! 20 Bohr, moves by 2.6e-5). It is a real bound state of the LSDA functional and an artefact of it: the
//! real atom has a full Rydberg series, far deeper. The KB potentials reproduce it (worst 6.4e-4 Ha, Ar);
//! it is reported, never gated on its energy.
//!
//! # The projector support r_β (spec §4.6; Task 3 review, ruling (b))
//! Every δV_l vanishes analytically beyond r_β = max(max_l r_c,l, r_nlc), where each V_l^ion is −Z_v/r + τ_c
//! (spec §4.3): bitwise under the highest-channel local, and under the polynomial local up to the identity's
//! quadrature residual (≈ 1e-12 Ha: a ΔQ/r tail from the Hartree cumulative's step across r_c, common to all
//! channels — measured δV·r constant beyond max r_c, Be −7.7e-13, Al −1.3e-12, Ar −1.7e-12). [`kb_form`]
//! finds r_β as the first radius from which every V_l^ion is −Z_v/r + τ_c to `v_ion_tail`/100 for good
//! (Task 2's identity bound; with the LFC core of Task 4 it moves out to r_nlc by itself), asserts that the
//! δV_l it discards are below that bound (a violation is a construction bug, not a refusal), sets δV_l and
//! β_l to 0 from r_β on and recomputes E_KB and GSS. `Projector::support` is r_β, the radius Tasks 5 and 6
//! size the filter and the sphere by. A 1e-12-relative rule would read the residual instead (polynomial
//! local, Al p at f = 1.6: 5.28 Bohr against r_β = 4.25).
//!
//! # Method-convergence literals (review S2 practice; none moves a converged result beyond its tolerance)
//! β's support: the last point with |β| > 1e-12·max|β|; r_m three points beyond. Root bisection to
//! 1e-13·max(1, |ε|). The SCF's lowest level: bisection on the exact count, its bracket narrowed to
//! ±0.1·max(|ε_prev|, 0.1) Ha around the previous ε where the count confirms it. `kb_state` takes the local
//! state when |W| < 1e-6 of its scale. r_β's residual bound: `v_ion_tail`/100 (Task 2's identity bound). A KB state's sign: positive at its outermost maximum above
//! `LEDGER.r_peak_floor`·max|u| (the `numerov` convention).
use super::atom::{hartree_potential, Diis, RadialError, SpinMode};
use super::grid::LogGrid;
use super::ledger::{LocalCandidate, LEDGER};
use super::numerov::{bound_state, eigenfunction, g_coeff, outward, outward_sourced, regular_start, shooting_end};
use super::partition::Partition;
use super::tm::{TmChannel, Unscreened};
use crate::grid::neumaier;
use crate::xc::exc_point;
use std::f64::consts::PI;

/// One KB projector of channel l: E_KB = 1/∫u δV u dr, GSS = ∫β²/∫uβ, β = δV·u (u-form), δV = V_l^ion −
/// V_loc, u = u_l^PP. β and δV are exactly 0 from `i_beta` on: r_β = r[i_beta] is the projector's support
/// (spec §4.6), derived in [`kb_form`] (module doc, "The projector support").
#[derive(Clone, Debug, PartialEq)]
pub struct Projector { pub l: u8, pub e_kb: f64, pub gss: f64, pub beta: Vec<f64>, pub dv: Vec<f64>, pub u: Vec<f64>, pub i_beta: usize }

/// The separable potential: V_loc + Σ projectors; `local_l` is the channel used as V_loc (None for the
/// polynomial local or a test's hand-made local); `rho_pc` the partial core in E_xc; r_loc = min r_c;
/// r_cmp from the unscreening.
#[derive(Clone, Debug, PartialEq)]
pub struct KbPotential {
    pub z: u8, pub z_v: f64, pub local: LocalCandidate, pub local_l: Option<u8>, pub v_loc: Vec<f64>,
    pub projectors: Vec<Projector>, pub rho_pc: Vec<f64>, pub r_loc: f64, pub r_cmp: f64,
}

/// What the KB atom needs: (l, E_KB, β u-form) per projector.
#[derive(Clone, Debug, PartialEq)]
pub struct KbInput { pub z: u8, pub z_v: f64, pub v_loc: Vec<f64>, pub projectors: Vec<(u8, f64, Vec<f64>)>, pub rho_pc: Vec<f64> }

/// Valence occupations, index = l, (↑, ↓).
#[derive(Clone, Debug, PartialEq)]
pub struct ValenceConfig { pub occ: Vec<[f64; 2]> }

/// The converged KB atom: ε and u per (l, spin) (NaN/empty where unoccupied), spin densities, the
/// potentials of the output density, the total valence energy (with E_xc[ρ_v + ρ_pc]).
#[derive(Clone, Debug, PartialEq)]
pub struct KbAtom { pub eps: Vec<[f64; 2]>, pub u: Vec<[Vec<f64>; 2]>, pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub energy: f64 }

#[derive(Clone, Debug, PartialEq)]
pub enum GssVerdict { GhostFree, Ghost { e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64> } }

/// G1 per projector; G2's worst eigenvalue error and the located roots per l ≤ l_max; `g2_missing`:
/// (l, AE energies, KB roots) below the G2 window top where the pseudo-atom binds fewer states than the
/// all-electron atom — not a ghost, recorded for T5 and the generation record (Task 3 review S1).
#[derive(Clone, Debug, PartialEq)]
pub struct GhostReport { pub g1: Vec<(u8, GssVerdict)>, pub g2_max_eig_error: f64, pub g2_roots: Vec<(u8, Vec<f64>)>, pub g2_missing: Vec<(u8, Vec<f64>, Vec<f64>)> }

/// τ_c(r) = V_H[ρ_c](r) − N_c/r (spec §4.3): the frozen core's Hartree tail, ≤ 0, ≡ 0 without a core.
/// `rho_c` must be the *full* core density (ρ_pc never enters V_H) and N_c = Z − Z_v.
pub fn core_tail(grid: &LogGrid, rho_c: &[f64], n_core: f64) -> Vec<f64> {
    let vh = hartree_potential(grid, rho_c);
    grid.r.iter().zip(&vh).map(|(r, v)| v - n_core / r).collect()
}

/// The polynomial local (module doc): T = −Z_v/r + τ_c from r_loc outwards, the even quartic matched C² to T
/// inside. r_loc must be a grid point (the generator's min r_c is one); an off-grid caller is told, not moved.
pub fn polynomial_local(grid: &LogGrid, z_v: f64, r_loc: f64, tau_c: &[f64]) -> Vec<f64> {
    assert_eq!(tau_c.len(), grid.len(), "τ_c is a grid table");
    let i = grid.nearest(r_loc);
    let r = grid.r[i];
    assert!((r - r_loc).abs() < 1e-12 * r_loc, "r_loc = {r_loc} is not a grid point (nearest {r})");
    let (dt, d2t) = { let d = grid.derivative(tau_c); let d2 = grid.derivative(&d); (d[i], d2[i]) };
    let t0 = -z_v / r + tau_c[i];
    let t1 = z_v / (r * r) + dt;
    let t2 = -2.0 * z_v / r.powi(3) + d2t;
    let a4 = (t2 - t1 / r) / (8.0 * r * r);
    let a2 = (t1 - 4.0 * a4 * r.powi(3)) / (2.0 * r);
    let a0 = t0 - a2 * r * r - a4 * r.powi(4);
    grid.r.iter().zip(tau_c).enumerate()
        .map(|(k, (&x, &t))| if k >= i { -z_v / x + t } else { a0 + x * x * (a2 + x * x * a4) }).collect()
}

/// The KB potential with the local candidate `local` (spec §4.4): the highest generated channel, or the
/// polynomial local with the core tail `tau_c` ([`core_tail`] of the full core).
#[allow(clippy::too_many_arguments)]
pub fn kb_form(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, local: LocalCandidate, rho_pc: Vec<f64>, tau_c: &[f64]) -> KbPotential {
    let top = channels.len() - 1;
    let (v_loc, local_l) = match local {
        LocalCandidate::HighestChannel => (un.v_ion[top].clone(), Some(channels[top].l)),
        LocalCandidate::Polynomial => (polynomial_local(grid, z_v, r_loc(channels), tau_c), None),
    };
    let mut kb = kb_form_with_local(grid, z, z_v, channels, un, v_loc, local_l, rho_pc);
    kb.local = local;
    // review ruling (b) (Task 3 report): the derived support r_β, the first radius from which every V_l^ion is
    // −Z_v/r + τ_c to v_ion_tail/100 for good (module doc)
    let n = grid.len();
    let tol = LEDGER.v_ion_tail / 100.0;
    let i_beta = (0..n).rev()
        .find(|&i| un.v_ion.iter().any(|v| (v[i] + z_v / grid.r[i] - tau_c[i]).abs() >= tol))
        .map_or(0, |i| i + 1).min(n - 1);
    for p in &mut kb.projectors {
        let worst = (i_beta..n).map(|i| p.dv[i].abs()).fold(0.0, f64::max);
        assert!(worst < tol, "Z = {z}, l = {}: δV = {worst:.2e} beyond r_β = {:.4} — the local does not carry the ionic tail (a construction bug)", p.l, grid.r[i_beta]);
        for i in i_beta..n { p.dv[i] = 0.0; p.beta[i] = 0.0; }
        let denom = grid.integrate(&(0..n).map(|i| p.u[i] * p.beta[i]).collect::<Vec<_>>());
        let num = grid.integrate(&p.beta.iter().map(|b| b * b).collect::<Vec<_>>());
        p.e_kb = 1.0 / denom;
        p.gss = num / denom;
        p.i_beta = i_beta;
    }
    kb
}

fn r_loc(channels: &[TmChannel]) -> f64 { channels.iter().map(|c| c.r_c).fold(f64::INFINITY, f64::min) }

/// Every channel other than `local_l` becomes a projector δV_l = V_l^ion − V_loc. `local` is set to
/// `HighestChannel` here; [`kb_form`] overwrites it with the candidate it built and derives r_β. A
/// hand-made local (the ghost tests) has no derived support: `i_beta` is then one point past the last
/// |β| > 1e-12·max|β| (a method fallback, never a physical radius).
#[allow(clippy::too_many_arguments)]
pub fn kb_form_with_local(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, v_loc: Vec<f64>, local_l: Option<u8>, rho_pc: Vec<f64>) -> KbPotential {
    let n = grid.len();
    let projectors = channels.iter().enumerate().filter(|(_, c)| Some(c.l) != local_l).map(|(k, c)| {
        let dv: Vec<f64> = (0..n).map(|i| un.v_ion[k][i] - v_loc[i]).collect();
        let beta: Vec<f64> = (0..n).map(|i| dv[i] * c.u[i]).collect();
        let denom = grid.integrate(&(0..n).map(|i| c.u[i] * beta[i]).collect::<Vec<_>>());
        let num = grid.integrate(&beta.iter().map(|b| b * b).collect::<Vec<_>>());
        let i_beta = (support_index(&beta) + 1).min(n - 1);
        Projector { l: c.l, e_kb: 1.0 / denom, gss: num / denom, beta, dv, u: c.u.clone(), i_beta }
    }).collect();
    KbPotential { z, z_v, local: LocalCandidate::HighestChannel, local_l, v_loc, projectors, rho_pc, r_loc: r_loc(channels), r_cmp: un.r_cmp }
}

impl KbPotential {
    pub fn input(&self) -> KbInput {
        KbInput { z: self.z, z_v: self.z_v, v_loc: self.v_loc.clone(),
                  projectors: self.projectors.iter().map(|p| (p.l, p.e_kb, p.beta.clone())).collect(), rho_pc: self.rho_pc.clone() }
    }
}

impl Projector {
    /// r_β, the projector's support (spec §4.6): β ≡ 0 from here on. Derived for [`kb_form`] tables —
    /// max(max_l r_c,l, r_nlc) to the grid — and what Tasks 5 and 6 size the filter and the sphere by.
    pub fn support(&self, grid: &LogGrid) -> f64 { grid.r[self.i_beta] }
}

fn support_index(beta: &[f64]) -> usize {
    let m = beta.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    beta.iter().rposition(|b| b.abs() > 1e-12 * m).unwrap_or(0)
}

/// β's support plus 3 points: the ε-independent matching index (an ε-dependent one would make det M jump).
fn matching_index(beta: &[f64]) -> usize { support_index(beta) + 3 }

/// The pieces of the Green-function form (module doc, "Stability") at energy e: u_h on [0, r_{i_m+2}], u_i
/// on [r_0, end] (inward from `shooting_end`, 0 beyond), the r-Wronskian W = W_r(u_h, u_i) taken where the
/// solutions are smoothest, its natural scale |u_h u_i′| + |u_h′ u_i| there, and C_h(r) = ∫₀^r β u_h dr′.
// deviation 1 (Task 3 report): det M through the Green kernel — the brief's u_h/u_p form cancels e^{2κr_c} at depth
struct Green { uh: Vec<f64>, ui: Vec<f64>, w: f64, w_scale: f64, ch: Vec<f64> }

fn green(grid: &LogGrid, v: &[f64], beta: &[f64], l: u8, e: f64, i_m: usize) -> Green {
    let n = grid.len();
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    let (yh, _) = outward(&g, y0, y1, i_m + 2, grid.delta);
    let ie = shooting_end(&g, grid.delta).max(i_m + 4).min(n - 1);
    let yi = inward_dirichlet(&g, ie, grid.delta);
    let sr = |i: usize| grid.r[i].sqrt();
    let uh: Vec<f64> = (0..=i_m + 2).map(|i| yh[i] * sr(i)).collect();
    let ui: Vec<f64> = (0..n).map(|i| if i <= ie { yi[i] * sr(i) } else { 0.0 }).collect();
    let dx = |f: &[f64], i: usize| (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) / (12.0 * grid.delta);
    // W is constant: take it where the solutions are smoothest in x, the point of least |g| (the turning
    // point when there is one). At r_m they change by e^{±√g δ} per step and the 5-point derivative's
    // relative error (√g δ)⁴/30 reaches 3e-3 (the oscillator oracle at r_m = 8.4) — harmless inside M, whose
    // row 1 carries it in both entries, but not in s
    let iw = (2..=i_m - 2).min_by(|&a, &b| g[a].abs().partial_cmp(&g[b].abs()).unwrap()).unwrap();
    let (a, b) = (uh[iw] * dx(&ui, iw), dx(&uh, iw) * ui[iw]);
    let ch = grid.cumulative(&(0..n).map(|i| if i <= i_m { beta[i] * uh[i] } else { 0.0 }).collect::<Vec<_>>());
    Green { uh, ui, w: (a - b) / grid.r[iw], w_scale: (a.abs() + b.abs()) / grid.r[iw], ch }
}

/// The solution with y(x_{i_start}) = 0, run inward (Numerov) to the first grid point: the Dirichlet
/// condition at r_ie that n_loc's Sturm count (nodes of u_h on (0, r_ie)) assumes. `numerov::inward`
/// starts instead from the WKB decay ratio, which is not that solution once the decay length exceeds the
/// box (ε → 0): the count then fell from 2 to 1 between ε = −1e-3 and −1e-4 Ha (polynomial-local Al s at
/// f = 1.0, Task 3 round 1) — impossible for a count. Deep in the gap both starts give the same, decaying,
/// solution. Rescaled by 1e-150 past |y| > 1e150 (scale-free).
// round 1 (Task 3 report): u_i vanishes at r_ie, the Dirichlet problem n_loc's Sturm count assumes
fn inward_dirichlet(g: &[f64], i_start: usize, delta: f64) -> Vec<f64> {
    let k = delta * delta / 12.0;
    let mut y = vec![0.0; i_start + 1];
    y[i_start - 1] = 1e-20;
    for i in (1..i_start).rev() {
        y[i - 1] = (2.0 * y[i] * (1.0 + 5.0 * k * g[i]) - y[i + 1] * (1.0 - k * g[i + 1])) / (1.0 - k * g[i - 1]);
        if y[i - 1].abs() > 1e150 { for v in &mut y[i - 1..] { *v *= 1e-150; } }
    }
    y
}

/// E_KB·W·s(e) = −W + 4E_KB∫₀^{r_m} β u_i C_h dr (= det M/r_m, evaluated stably) and W.
fn kb_green(grid: &LogGrid, v: &[f64], e_kb: f64, beta: &[f64], l: u8, e: f64, i_m: usize) -> (f64, f64) {
    let n = grid.len();
    let gr = green(grid, v, beta, l, e, i_m);
    let j = grid.partial(&(0..n).map(|i| if i <= i_m { beta[i] * gr.ui[i] * gr.ch[i] } else { 0.0 }).collect::<Vec<_>>(), i_m);
    (-gr.w + 4.0 * e_kb * j, gr.w)
}

/// min_r(V + l(l+1)/2r²) + min(0, E_KB∫β²): a lower bound of channel l's KB spectrum (the kinetic
/// operator is ≥ 0 and ⟨u|β⟩² ≤ ∫β² for ∫u² = 1).
pub fn kb_lower_bound(grid: &LogGrid, v: &[f64], l: u8, proj: Option<(f64, &[f64])>) -> f64 {
    let lf = l as f64;
    let floor = grid.r.iter().zip(v).map(|(r, vi)| vi + lf * (lf + 1.0) / (2.0 * r * r)).fold(f64::INFINITY, f64::min);
    floor + proj.map_or(0.0, |(e, b)| (e * grid.integrate(&b.iter().map(|x| x * x).collect::<Vec<_>>())).min(0.0))
}

/// The number of local-channel eigenvalues below e (Sturm: the sign changes of the regular solution out
/// to `shooting_end`, as `bound_state` counts them).
fn local_count(grid: &LogGrid, v: &[f64], l: u8, e: f64) -> usize {
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    outward(&g, y0, y1, shooting_end(&g, grid.delta), grid.delta).1
}

/// The exact number of KB eigenvalues of channel l below e (plan review Q5; module doc):
/// n_KB(e) = n_loc(e) + [s(e) < 0] − [E_KB > 0], s = det M/(E_KB·W(u_h,u_i)) in its stable form
/// (`kb_green`). Without a projector: node counting.
pub fn kb_count_below(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> usize {
    let n_loc = local_count(grid, v, l, e);
    let Some((e_kb, beta)) = proj else { return n_loc };
    let (det, w) = kb_green(grid, v, e_kb, beta, l, e, matching_index(beta));
    let s = det / (e_kb * w);
    debug_assert!(n_loc + usize::from(s < 0.0) >= usize::from(e_kb > 0.0), "inertia count below zero: a W/node inconsistency (n_loc {n_loc}, s {s}, E_KB {e_kb})");
    (n_loc + usize::from(s < 0.0)).saturating_sub(usize::from(e_kb > 0.0))
}

/// Every eigenvalue of channel l in [e_lo, e_hi], ascending: without a projector by node counting
/// (`bound_state`); with one, by the sign changes of det M (its stable form, `kb_green`) on a mesh, each
/// bisected to 1e-13·max(1, |e|).
/// The mesh only *locates* roots — [`kb_count_below`] is the exact count G2 gates on (Spec question 5).
pub fn kb_eigenvalues(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e_lo: f64, e_hi: f64, mesh: f64) -> Vec<f64> {
    match proj {
        None => (0..).map_while(|k| bound_state(grid, v, l, k, 0.0, None)).map(|b| b.e)
            .take_while(|&e| e < e_hi).filter(|&e| e >= e_lo).collect(),
        Some((e_kb, beta)) => {
            let i_m = matching_index(beta);
            let det = |e: f64| kb_green(grid, v, e_kb, beta, l, e, i_m).0;
            let steps = ((e_hi - e_lo) / mesh).ceil().max(1.0) as usize;
            let (mut a, mut fa) = (e_lo, det(e_lo));
            let mut roots = Vec::new();
            for k in 1..=steps {
                let b = (e_lo + k as f64 * mesh).min(e_hi);
                let fb = det(b);
                if fa * fb < 0.0 {
                    let (mut lo, mut hi, mut flo) = (a, b, fa);
                    while hi - lo > 1e-13 * lo.abs().max(1.0) {
                        let mid = 0.5 * (lo + hi);
                        let fm = det(mid);
                        if flo * fm <= 0.0 { hi = mid } else { lo = mid; flo = fm; }
                    }
                    roots.push(0.5 * (lo + hi));
                }
                a = b;
                fa = fb;
            }
            roots
        }
    }
}

/// The normalised eigenfunction at an eigenvalue e (positive at its outermost maximum). With a projector,
/// u ∝ Gβ (H_KB u = εu ⇔ (H_loc − ε)u = −E_KB β⟨β|u⟩): u ∝ u_i(r)C_h(r) + u_h(r)D_i(r), D_i(r) = ∫_r^∞ u_i β
/// — the Green-function form `kb_eigenvalues` locates the roots with (module doc, "Stability"), so the
/// state is self-consistent at exactly that root, bounded at any depth, and needs no division by W. The
/// exception is ⟨β|u⟩ = 0: e is then a local eigenvalue with its state orthogonal to β (W = 0 and Gβ
/// vanishes), and the state is the local one — taken when |W| < 1e-6 of its scale (a method literal: a
/// measure-zero coincidence for a real potential, exercised by the oscillator oracle's untouched levels).
// deviation 2 (Task 3 report): u ∝ Gβ, the root-finder's own kernel — M's null vector cancels growing parts
pub fn kb_state(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> Vec<f64> {
    let Some((_, beta)) = proj else { return eigenfunction(grid, v, l, e, 0.0).u };
    let n = grid.len();
    let i_m = matching_index(beta);
    let gr = green(grid, v, beta, l, e, i_m);
    if gr.w.abs() < 1e-6 * gr.w_scale { return eigenfunction(grid, v, l, e, 0.0).u; }
    let di = grid.cumulative(&(0..n).map(|i| if i <= i_m { gr.ui[i] * beta[i] } else { 0.0 }).collect::<Vec<_>>());
    let d_tot = di[i_m];
    let u: Vec<f64> = (0..n).map(|i| if i <= i_m { gr.ui[i] * gr.ch[i] + gr.uh[i] * (d_tot - di[i]) } else { gr.ui[i] * gr.ch[i_m] }).collect();
    let norm = grid.integrate(&u.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
    let umax = u.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let ip = (1..n - 1).rev()
        .find(|&i| u[i].abs() >= u[i - 1].abs() && u[i].abs() >= u[i + 1].abs() && u[i].abs() > LEDGER.r_peak_floor * umax)
        .unwrap_or(0);
    let sgn = if u[ip] < 0.0 { -1.0 } else { 1.0 };
    u.iter().map(|x| sgn * x / norm).collect()
}

/// The regular solution at energy e out to i_end (unnormalised): T5's log derivatives (Task 4). With a
/// projector: u = (1 − E_KB⟨β|u_p⟩)·u_h + E_KB⟨β|u_h⟩·u_p, ⟨β|·⟩ over β's support. Then E_KB⟨β|u⟩ =
/// E_KB⟨β|u_h⟩ is the coefficient of u_p — the self-consistent c, scaled by (1 − E_KB⟨β|u_p⟩) instead of
/// divided by it, so there is no pole where that factor vanishes (plan review Q12). Homogeneous of degree
/// one in u_h, so its overflow rescaling is harmless.
pub fn kb_regular(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64, i_end: usize) -> Vec<f64> {
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    let reach = proj.map_or(i_end, |(_, b)| matching_index(b).max(i_end));
    let (yh, _) = outward(&g, y0, y1, reach, grid.delta);
    let sr = |i: usize| grid.r[i].sqrt();
    let uh: Vec<f64> = (0..=reach).map(|i| yh[i] * sr(i)).collect();
    let Some((e_kb, beta)) = proj else { return uh[..=i_end].to_vec() };
    let s: Vec<f64> = (0..=reach).map(|i| 2.0 * grid.r[i].powf(1.5) * beta[i]).collect();
    let up: Vec<f64> = outward_sourced(&g, &s, reach, grid.delta).iter().enumerate().map(|(i, y)| y * sr(i)).collect();
    let n = grid.len();
    let dot = |u: &[f64]| grid.partial(&(0..n).map(|i| if i <= reach { beta[i] * u[i] } else { 0.0 }).collect::<Vec<_>>(), reach);
    let (a, c) = (1.0 - e_kb * dot(&up), e_kb * dot(&uh));
    (0..=i_end).map(|i| a * uh[i] + c * up[i]).collect()
}

/// The reference configuration: each occupied channel's occupation, half per spin; index l = 0..=l_max.
pub fn reference_config(part: &Partition) -> ValenceConfig {
    ValenceConfig { occ: (0..=part.l_max).map(|l| {
        let o = part.channels.iter().find(|c| c.l == l && c.occupied).map_or(0.0, |c| c.occ);
        [0.5 * o, 0.5 * o]
    }).collect() }
}

/// V_σ = V_loc + V_H[ρ↑ + ρ↓] + v_xc,σ[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc].
fn kb_potential(grid: &LogGrid, input: &KbInput, rho: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
    let n = grid.len();
    let total: Vec<f64> = (0..n).map(|i| rho[0][i] + rho[1][i]).collect();
    let v_h = hartree_potential(grid, &total);
    let mut v = [vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let c = 0.5 * input.rho_pc[i];
        let (_, vu, vd) = exc_point(rho[0][i] + c, rho[1][i] + c);
        v[0][i] = input.v_loc[i] + v_h[i] + vu;
        v[1][i] = input.v_loc[i] + v_h[i] + vd;
    }
    v
}

/// The lowest eigenvalue of channel l, by bisection on the exact count (`kb_count_below`): the infimum of
/// {e : n_KB(e) ≥ 1} over [lower bound, 0], to 1e-13·max(1, |e|); without a projector `bound_state` with
/// no node (the same bisection on the node count). `prev` (the last SCF step's ε) only narrows the
/// bracket, ±0.1·max(|prev|, 0.1) Ha, and only where the count confirms it (as `bound_state` uses a
/// guess). Deviation from the brief, whose search returned the first root in a ±0.1 Ha window around
/// `prev`: when the level moved by more than that between early SCF steps the window held the next state,
/// and H's KB atom converged to a self-consistent 2s configuration (ε = −0.0712 vs −0.2335 Ha, measured).
// deviation 3 (Task 3 report): bisection on the exact count — the brief's ±0.1 Ha window put H in a 2s state
fn lowest(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, prev: f64) -> Option<f64> {
    let guess = prev.is_finite().then_some(prev);
    if proj.is_none() { return bound_state(grid, v, l, 0, 0.0, guess).map(|b| b.e); }
    let count = |e: f64| kb_count_below(grid, v, proj, l, e);
    let mut hi = 0.0;
    if count(hi) == 0 { return None; }
    let mut lo = kb_lower_bound(grid, v, l, proj);
    if let Some(gs) = guess {
        let w = 0.1 * gs.abs().max(0.1);
        if gs - w > lo && count(gs - w) == 0 { lo = gs - w; }
        if gs + w < hi && count(gs + w) >= 1 { hi = gs + w; }
    }
    while hi - lo > 1e-13 * lo.abs().max(1.0) {
        let mid = 0.5 * (lo + hi);
        if count(mid) >= 1 { hi = mid } else { lo = mid }
    }
    Some(0.5 * (lo + hi))
}

/// The self-consistent radial KB atom (module doc). `Restricted`: both spins in the ↑ potential.
pub fn solve_kb_atom(grid: &LogGrid, input: &KbInput, config: &ValenceConfig, spin: SpinMode) -> Result<KbAtom, RadialError> {
    let n = grid.len();
    let num = &LEDGER.numerics;
    let w: Vec<f64> = grid.r.iter().map(|r| r * r * r * grid.delta).collect();
    let nl = config.occ.len();
    let mut eps = vec![[f64::NAN; 2]; nl];
    let mut us: Vec<[Vec<f64>; 2]> = vec![[Vec::new(), Vec::new()]; nl];
    let occupy = |v: &[Vec<f64>; 2], eps: &mut Vec<[f64; 2]>, us: &mut Vec<[Vec<f64>; 2]>| -> Result<[Vec<f64>; 2], RadialError> {
        let mut rho = [vec![0.0; n], vec![0.0; n]];
        for l in 0..nl {
            let proj = input.projectors.iter().find(|p| p.0 as usize == l).map(|p| (p.1, p.2.as_slice()));
            for s in 0..2 {
                if config.occ[l][s] <= 0.0 { continue; }
                if spin == SpinMode::Restricted && s == 1 && config.occ[l][0] > 0.0 {
                    // the same potential: the ↑ state
                    eps[l][1] = eps[l][0];
                    us[l][1] = us[l][0].clone();
                } else {
                    let src = if spin == SpinMode::Restricted { 0 } else { s };
                    let e = lowest(grid, &v[src], proj, l as u8, eps[l][s])
                        .ok_or(RadialError::Unbound { z: input.z, n: 0, l: l as u8, after_backtracking: false })?;
                    us[l][s] = kb_state(grid, &v[src], proj, l as u8, e);
                    eps[l][s] = e;
                }
                for i in 0..n { rho[s][i] += config.occ[l][s] * us[l][s][i] * us[l][s][i] / (4.0 * PI * grid.r[i] * grid.r[i]); }
            }
        }
        Ok(rho)
    };
    // deviation 4 (Task 3 report): ρ_in = 0 (the bare pseudo-ion), as `atom` starts (its deviation 1): the bare-ion density is too
    // compact, and taken whole as the first input it over-screens and unbinds the p shell (C–Ne, measured);
    // from ρ = 0 the first mixing step feeds in only α of it
    let mut rho_in = [vec![0.0; n], vec![0.0; n]];
    let mut diis = Diis::new(num.diis_history);
    let mut delta = f64::INFINITY;
    for _ in 0..num.scf_max_iter {
        let v = kb_potential(grid, input, &rho_in);
        let rho_out = occupy(&v, &mut eps, &mut us)?;
        delta = (0..2).map(|s| (0..n).map(|i| (rho_out[s][i] - rho_in[s][i]).abs() * w[i]).sum::<f64>()).sum();
        if delta < num.scf_tol {
            let shell = |f: &[f64]| grid.integrate(&f.iter().zip(&grid.r).map(|(a, r)| 4.0 * PI * r * r * a).collect::<Vec<_>>());
            let band = neumaier((0..nl).flat_map(|l| (0..2).filter(move |&s| config.occ[l][s] > 0.0).map(move |s| (l, s))).map(|(l, s)| config.occ[l][s] * eps[l][s]));
            let pot = neumaier((0..2).map(|s| shell(&(0..n).map(|i| rho_out[s][i] * v[s][i]).collect::<Vec<_>>())));
            let total: Vec<f64> = (0..n).map(|i| rho_out[0][i] + rho_out[1][i]).collect();
            let v_h = hartree_potential(grid, &total);
            let e_loc = shell(&(0..n).map(|i| total[i] * input.v_loc[i]).collect::<Vec<_>>());
            let e_h = 0.5 * shell(&(0..n).map(|i| total[i] * v_h[i]).collect::<Vec<_>>());
            let e_xc = shell(&(0..n).map(|i| { let c = 0.5 * input.rho_pc[i]; exc_point(rho_out[0][i] + c, rho_out[1][i] + c).0 }).collect::<Vec<_>>());
            let v_out = kb_potential(grid, input, &rho_out);
            let energy = neumaier([band, -pot, e_loc, e_h, e_xc].into_iter());
            return Ok(KbAtom { eps, u: us, rho: rho_out, v: v_out, energy });
        }
        rho_in = diis.mix(rho_in, rho_out, &w, num.mixing);
    }
    Err(RadialError::NotConverged { z: input.z, iterations: num.scf_max_iter, delta })
}

/// GSS (module doc): ghost-free iff (E_KB < 0 ∧ ε_l < ε₀^loc) ∨ (E_KB > 0 ∧ ε₀^loc < ε_l < ε₁^loc); an
/// unbound local level sits at the continuum edge, 0.
pub fn gss_verdict(e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64>) -> GssVerdict {
    let (e0, e1) = (eps0_loc.unwrap_or(0.0), eps1_loc.unwrap_or(0.0));
    let ok = if e_kb < 0.0 { eps_ref < e0 } else { e0 < eps_ref && eps_ref < e1 };
    if ok { GssVerdict::GhostFree } else { GssVerdict::Ghost { e_kb, eps_ref, eps0_loc, eps1_loc } }
}

/// G1 (GSS, every projector channel, on the reference screening `screen`), then G2 (spec §4.6): the SCF KB atom at
/// the reference configuration reproduces the valence eigenvalues to LEDGER.gates.g2_eig, and for every l ≤ l_max
/// the exact inertia count below top = min(0, ε_v,max + g2_window) does not exceed the all-electron reference's
/// bound states of l above its core below top (`Partition::spectrum`). More is a ghost (refused); fewer is an AE
/// state the pseudo-atom does not bind, not a ghost (recorded in `g2_missing` for T5 and the record). The mesh roots
/// must number the count, else the mesh missed a pair (a numerical miss, not a ghost verdict).
pub fn ghost_gates(grid: &LogGrid, kb: &KbPotential, screen: &[f64], part: &Partition) -> Result<GhostReport, String> {
    let v_ref: Vec<f64> = kb.v_loc.iter().zip(screen).map(|(a, b)| a + b).collect();
    let mut g1 = Vec::new();
    for p in &kb.projectors {
        let ch = part.channels.iter().find(|c| c.l == p.l).expect("a projector has a channel");
        let e0 = bound_state(grid, &v_ref, p.l, 0, 0.0, None).map(|b| b.e);
        let e1 = bound_state(grid, &v_ref, p.l, 1, 0.0, None).map(|b| b.e);
        g1.push((p.l, gss_verdict(p.e_kb, ch.eps, e0, e1)));
    }
    if let Some((l, v)) = g1.iter().find(|(_, v)| *v != GssVerdict::GhostFree) {
        return Err(format!("G1: ghost in l = {l}: {v:?}"));
    }
    let atom = solve_kb_atom(grid, &kb.input(), &reference_config(part), SpinMode::Restricted).map_err(|e| format!("G2: {e}"))?;
    let g2_max_eig_error = part.channels.iter().filter(|c| c.occupied)
        .map(|c| (atom.eps[c.l as usize][0] - c.eps).abs()).fold(0.0, f64::max);
    if g2_max_eig_error >= LEDGER.gates.g2_eig {
        return Err(format!("G2: KB eigenvalues miss the reference by {:.3} mHa", g2_max_eig_error * 1e3));
    }
    let eps_v_max = part.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max);
    let top = (eps_v_max + LEDGER.gates.g2_window).min(0.0);
    let mut g2_roots = Vec::new();
    let mut g2_missing = Vec::new();
    for l in 0..=part.l_max {
        let proj = kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
        let count = kb_count_below(grid, &atom.v[0], proj, l, top);
        let roots = kb_eigenvalues(grid, &atom.v[0], proj, l, kb_lower_bound(grid, &atom.v[0], l, proj), top, LEDGER.numerics.g2_mesh);
        if roots.len() != count {
            return Err(format!("G2: l = {l}: the g2_mesh located {} roots ({roots:?}) but the inertia count is {count} — report (a numerical miss, not a ghost verdict)", roots.len()));
        }
        // deviation 5 + review S1 (Task 3 report): count against the AE spectrum, refuse only an excess.
        // The all-electron atom's own states of l above its core (Partition::spectrum): the neutral LSDA atom
        // binds one (n+1)s just below 0, which the KB potential reproduces — not a ghost. More than the AE
        // atom is a ghost; fewer is an AE state the pseudo-atom does not bind (T5's, recorded)
        let ae: Vec<f64> = part.spectrum[l as usize].iter().copied().filter(|&e| e < top).collect();
        if count > ae.len() {
            return Err(format!("G2: l = {l} has {count} bound states below {top:.3} Ha (inertia count), the all-electron atom {}: an extra state (ghost); KB {roots:?}, AE {ae:?}", ae.len()));
        }
        if count < ae.len() { g2_missing.push((l, ae, roots.clone())); }
        g2_roots.push((l, roots));
    }
    Ok(GhostReport { g1, g2_max_eig_error, g2_roots, g2_missing })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::numerov::{bound_state, inward};
    use crate::radial::partition::{core_density, partition};
    use crate::radial::recipe::generate;
    use crate::radial::tm::{tm_channels, unscreen};
    use crate::radial::{ground_state, SpinMode, LEDGER};

    /// Spec §8 row 3, first failing test: the radial KB atom (integro-differential, the 2×2 determinant)
    /// reproduces the semilocal TM atom's valence eigenvalues — the AE reference ones — to < 1 mHa.
    #[test]
    fn kb_atom_reproduces_semilocal_eigenvalues() {
        for z in [6u8, 7, 8, 9, 10] {
            let t = generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
            let atom = solve_kb_atom(&t.grid, &t.kb.input(), &reference_config(&t.partition), SpinMode::Restricted).expect("KB atom converges");
            for c in t.partition.channels.iter().filter(|c| c.occupied) {
                let e = atom.eps[c.l as usize][0];
                eprintln!("Z = {z}, l = {}: KB atom {e:.10} vs reference {:.10}: Δ {:+.2e}", c.l, c.eps, e - c.eps);
                assert!((e - c.eps).abs() < 1e-3, "Z = {z}, l = {}: KB {e:.6} vs semilocal/AE {:.6}", c.l, c.eps);
            }
        }
    }

    /// Spec §8 row 3, second first-failing test, as Spec question 4 rules (plan review: claim confirmed,
    /// ruling accepted with two test amendments): the GSS criterion and G2 must catch both constructible
    /// ghost cases on O's s channel:
    /// (A) E_KB > 0 with a local so deep that ε₁^loc < ε_s (the textbook deep-local ghost);
    /// (B) E_KB < 0 with ε₀^loc < ε_s (a local repulsive where φ_s lives, with a narrow deep well at the origin).
    /// (A local with ε₀^loc above ε_s forces E_KB < 0 by the variational principle, which GSS calls ghost-free.)
    /// G2 is checked on the fixed reference screening (v_loc + t.screen), unconditionally, in both cases
    /// (review Q4(b)): a ghost can make the KB SCF fail, so a check behind `if let Ok(atom)` is vacuous
    /// exactly when it matters. By interlacing, each case puts a KB level below ε_s besides ε_s itself
    /// (A: ε₀^KB ≤ ε₁^loc < ε_s; B: ε₀^KB ≤ ε₀^loc < ε_s), so the exact count at ε_s + g2_window is ≥ 2
    /// where the s channel expects 1.
    #[test]
    fn deliberate_bad_local_is_caught_as_ghost() {
        let t = generate(8).unwrap();
        let g = &t.grid;
        let un = crate::radial::tm::unscreen(g, t.z_v, &t.channels, &t.partition.channels.iter().map(|c| c.occ).collect::<Vec<_>>(), &t.kb.rho_pc);
        let eps_s = t.partition.channels[0].eps;
        let rc = t.channels[0].r_c;
        let top = eps_s + LEDGER.gates.g2_window;
        let screened = |v: &[f64]| v.iter().zip(&t.screen).map(|(a, b)| a + b).collect::<Vec<f64>>();
        let s_levels = |v: &[f64]| (bound_state(g, &screened(v), 0, 0, 0.0, None).map(|b| b.e), bound_state(g, &screened(v), 0, 1, 0.0, None).map(|b| b.e));
        let bump = |w: f64| g.r.iter().map(|r| (-(r / w).powi(2)).exp()).collect::<Vec<f64>>();
        // G2 on the reference screening: the exact inertia count below `top`, and the roots the mesh
        // locates there (the mesh sized for at most 2·10⁴ determinant evaluations, so the deep case B
        // stays a unit test; the count, not the mesh, is the proof)
        let g2 = |kb: &KbPotential| {
            let s = kb.projectors.iter().find(|p| p.l == 0).unwrap();
            let v = screened(&kb.v_loc);
            let proj = Some((s.e_kb, s.beta.as_slice()));
            let count = kb_count_below(g, &v, proj, 0, top);
            let lo = kb_lower_bound(g, &v, 0, proj);
            let roots = kb_eigenvalues(g, &v, proj, 0, lo, top, ((top - lo) / 2e4).max(LEDGER.numerics.g2_mesh));
            (count, roots)
        };
        // (A) V_loc = V_p^ion − A·e^{−(r/r_c)²}, A raised until two local s levels lie below ε_s
        let p_ion = &un.v_ion[1];
        let wide = bump(rc);
        let a = (1..400).map(|k| k as f64 * 0.5).find(|&a| {
            let v: Vec<f64> = p_ion.iter().zip(&wide).map(|(p, b)| p - a * b).collect();
            matches!(s_levels(&v), (_, Some(e1)) if e1 < eps_s)
        }).expect("a deep enough local exists");
        let v_a: Vec<f64> = p_ion.iter().zip(&wide).map(|(p, b)| p - a * b).collect();
        let kb_a = kb_form_with_local(g, 8, t.z_v, &t.channels, &un, v_a, None, t.kb.rho_pc.clone());
        let s = kb_a.projectors.iter().find(|p| p.l == 0).unwrap();
        assert!(s.e_kb > 0.0, "case A must have E_KB > 0, got {}", s.e_kb);
        let err = ghost_gates(g, &kb_a, &t.screen, &t.partition).expect_err("case A is a ghost");
        assert!(err.contains("G1") && err.contains("l = 0"), "{err}");
        let (e0, e1) = s_levels(&kb_a.v_loc);
        assert!(matches!(gss_verdict(s.e_kb, eps_s, e0, e1), GssVerdict::Ghost { .. }));
        let (count, roots) = g2(&kb_a);
        eprintln!("case A: A = {a} Ha, E_KB(s) = {:.6}, ε₀^loc {e0:?}, ε₁^loc {e1:?}, ε_s {eps_s:.6}; G2 count below {top:.4} = {count}, roots {roots:?}; gates: {err}", s.e_kb);
        assert!(count >= 2, "case A: G2's inertia count below {top:.3} Ha is {count}; a ghost besides ε_s makes it ≥ 2");
        assert_eq!(roots.len(), count, "case A: mesh roots {roots:?} vs the inertia count {count}");
        // (B) V_loc = V_s^ion + 1 Ha·e^{−(r/r_c)²} − D·e^{−(r/w)²}, w = 0.05 Bohr, D scanned geometrically
        // over [1, 1e4] Ha (review Q4(a)): binding a deep level needs D·w² ≳ 1–10, while the narrow well's
        // share of ⟨φ|δV|φ⟩ grows only as D·w³·u′(0)² ∝ w, so at w = 0.15 the window D_bind < D < D_flip
        // (E_KB turning positive at D_flip) can be empty.
        let s_ion = &un.v_ion[0];
        let narrow = bump(0.05);
        let mut found = None;
        for k in 0..=80 {
            let d = 10f64.powf(k as f64 / 20.0);
            let v: Vec<f64> = (0..g.len()).map(|i| s_ion[i] + wide[i] - d * narrow[i]).collect();
            if !matches!(s_levels(&v), (Some(e0), _) if e0 < eps_s) { continue; }
            let kb_b = kb_form_with_local(g, 8, t.z_v, &t.channels, &un, v, None, t.kb.rho_pc.clone());
            let e_kb = kb_b.projectors.iter().find(|p| p.l == 0).unwrap().e_kb;
            eprintln!("case B scan: first D with ε₀^loc < ε_s: D = {d:.2} Ha, E_KB(s) = {e_kb:.6}");
            if e_kb < 0.0 { found = Some((d, kb_b)); }
            break; // ⟨φ|δV|φ⟩ grows with D: past D_bind, E_KB < 0 here or never
        }
        let (d, kb_b) = found.expect("case B's window D_bind < D < D_flip was not found for D ≤ 1e4 Ha at w = 0.05 Bohr — report the scan (review Q4(a))");
        let sb = kb_b.projectors.iter().find(|p| p.l == 0).unwrap();
        let err = ghost_gates(g, &kb_b, &t.screen, &t.partition).expect_err("case B is a ghost");
        assert!(err.contains("G1"), "{err}");
        let (e0, e1) = s_levels(&kb_b.v_loc);
        assert!(matches!(gss_verdict(sb.e_kb, eps_s, e0, e1), GssVerdict::Ghost { .. }));
        let (count, roots) = g2(&kb_b);
        eprintln!("case B: D = {d:.2} Ha, E_KB(s) = {:.6}, ε₀^loc {e0:?}, ε₁^loc {e1:?}, ε_s {eps_s:.6}; G2 count below {top:.4} = {count}, roots {roots:?}; gates: {err}", sb.e_kb);
        assert!(count >= 2, "case B (D = {d:.1} Ha): G2's inertia count below {top:.3} Ha is {count}; a ghost besides ε_s makes it ≥ 2");
        assert_eq!(roots.len(), count, "case B: mesh roots {roots:?} vs the inertia count {count}");
    }

    /// Plan review Q5: the inertia count n_KB(E) = n_loc(E) + [s(E) < 0] − [E_KB > 0] equals the number
    /// of KB roots below E, for every projector channel of C–Ne at several energies, and is the plain
    /// node count without a projector.
    #[test]
    fn inertia_count_matches_the_roots() {
        for z in [6u8, 8, 10] {
            let t = generate(z).unwrap();
            let v: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
            for p in &t.kb.projectors {
                let proj = Some((p.e_kb, p.beta.as_slice()));
                let lo = kb_lower_bound(&t.grid, &v, p.l, proj);
                for e in [-2.0, -1.0, -0.5, -0.1, 0.0] {
                    let roots = kb_eigenvalues(&t.grid, &v, proj, p.l, lo, e, LEDGER.numerics.g2_mesh);
                    assert_eq!(kb_count_below(&t.grid, &v, proj, p.l, e), roots.len(), "Z = {z}, l = {}, E = {e}: roots {roots:?}", p.l);
                }
            }
            let lo = kb_lower_bound(&t.grid, &v, 1, None);
            assert_eq!(kb_count_below(&t.grid, &v, None, 1, -0.05), kb_eigenvalues(&t.grid, &v, None, 1, lo, -0.05, LEDGER.numerics.g2_mesh).len());
        }
    }

    #[test]
    fn gss_criterion_cases() {
        assert!(matches!(gss_verdict(-1.0, -1.0, Some(-0.5), Some(0.0)), GssVerdict::GhostFree));
        assert!(matches!(gss_verdict(-1.0, -1.0, Some(-2.0), Some(-0.5)), GssVerdict::Ghost { .. }));
        assert!(matches!(gss_verdict(1.0, -1.0, Some(-2.0), Some(-0.5)), GssVerdict::GhostFree));
        assert!(matches!(gss_verdict(1.0, -1.0, Some(-3.0), Some(-2.0)), GssVerdict::Ghost { .. }));
        assert!(matches!(gss_verdict(1.0, -1.0, Some(-2.0), None), GssVerdict::GhostFree)); // ε₁ unbound: the continuum edge
    }

    /// Recovers (a₀, a₂, a₄) from three inner points of a polynomial-local table (exact for an even
    /// quartic up to rounding) and asserts that every inner point is that quartic.
    fn quartic_of(g: &LogGrid, v: &[f64], r_loc: f64) -> [f64; 3] {
        let i_loc = g.nearest(r_loc);
        let pts = [g.nearest(0.3 * r_loc), g.nearest(0.6 * r_loc), g.nearest(0.9 * r_loc)];
        let m = nalgebra::Matrix3::from_fn(|k, j| g.r[pts[k]].powi(2 * j as i32));
        let a = m.lu().solve(&nalgebra::Vector3::from_fn(|k, _| v[pts[k]])).expect("3×3");
        let worst = (0..i_loc).map(|i| (a[0] + g.r[i].powi(2) * (a[1] + g.r[i].powi(2) * a[2]) - v[i]).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-12 * a[0].abs(), "the inside is not one even quartic: {worst:.2e}");
        [a[0], a[1], a[2]]
    }

    /// The polynomial local candidate (spec §4.4) without a core (τ_c ≡ 0): −Z_v/r beyond r_loc, matched
    /// C² there, even inside; V(0) = −15Z_v/(8r_loc). r_loc is snapped to the grid (deviation from the
    /// brief's literal 1.2: the generator's r_loc = min r_c is always a grid point, and τ_c and its
    /// derivatives are grid tables).
    /// The construction is checked exactly: the coefficients recovered from the table are the closed forms
    /// a₄ = −3Z_v/(8r⁵), a₂ = 5Z_v/(4r³), a₀ = −15Z_v/(8r) to 1e-12 relative, so value, slope and curvature
    /// match −Z_v/r at r_loc identically. The brief's grid-derivative checks are kept for V and V′; for V″
    /// its 1e-2 was not derived and fails (7.7e-2): the twice-applied 5-point stencil straddles the C³ jump
    /// [V‴] = T‴ − 24a₄r = 15Z_v/r⁴ at r_loc, and for f = J·x₊³/6 (x = ln r) it returns D(D f)(0) = (5/27)Jδ
    /// instead of 0. Hence measured V″ − T″ = (5/27)·r³[V‴]·δ/r² = (25/9)Z_vδ/r³ (0.07699 here) plus
    /// O(δ²) relative (2.2e-4 at δ = 0.008, falling 4× per halving — replicated off-line); asserted to 1e-3
    /// relative, which also bounds a genuine C² mismatch to ≈ 1.5e-4 (a jump K in V″ enters the twice-applied
    /// stencil at half weight: D(D(K·x₊²/2))(0) = K/2).
    #[test]
    fn polynomial_local_is_c2_at_r_loc() {
        let g = crate::radial::LogGrid::universal();
        let (z_v, r_loc) = (6.0, g.r[g.nearest(1.2)]);
        let v = polynomial_local(&g, z_v, r_loc, &vec![0.0; g.len()]);
        assert!((v[0] + 15.0 * z_v / (8.0 * r_loc)).abs() < 1e-9);
        let a = quartic_of(&g, &v, r_loc);
        let want = [-15.0 * z_v / (8.0 * r_loc), 5.0 * z_v / (4.0 * r_loc.powi(3)), -3.0 * z_v / (8.0 * r_loc.powi(5))];
        for k in 0..3 { assert!((a[k] - want[k]).abs() < 1e-12 * want[k].abs(), "a{} = {} vs {}", 2 * k, a[k], want[k]); }
        let (dv, d2v) = (g.derivative(&v), g.derivative(&g.derivative(&v)));
        let i = g.nearest(r_loc);
        let r = g.r[i];
        let stencil = 25.0 / 9.0 * z_v * g.delta / r.powi(3);
        eprintln!("r_loc {r:.6}: V + Z_v/r {:.2e}, V′ − Z_v/r² {:.2e}, V″ + 2Z_v/r³ {:.5e} (stencil term {stencil:.5e})", v[i] + z_v / r, dv[i] - z_v / (r * r), d2v[i] + 2.0 * z_v / r.powi(3));
        assert!((v[i] + z_v / r).abs() < 1e-6 && (dv[i] - z_v / (r * r)).abs() < 1e-4);
        assert!((d2v[i] + 2.0 * z_v / r.powi(3) - stencil).abs() < 1e-3 * stencil, "V″ at r_loc: {:.5e} vs the stencil term {stencil:.5e}", d2v[i] + 2.0 * z_v / r.powi(3));
    }

    /// Task 2 review S3 (spec §4.4 item 2): beyond r_loc the polynomial local is −Z_v/r + τ_c, the same
    /// tail every V_l^ion has beyond max r_c (spec §4.3), so δV_l = V_l^ion − V_loc vanishes there up to
    /// the unscreening identity's residual (Task 2: ≤ 2.4e-12, bound v_ion_tail/100), and β_l is as
    /// compact as the highest-channel local's: O at f = 1.6 has its 1e-12-relative β support within the
    /// first grid point beyond max r_c (≤ 1.41 Bohr; measured 1.39 for s and p), where a local without τ_c
    /// gives 2.14 (the review probe: 1.97; asserted here as "beyond", so the test is seen to discriminate). Al (the review's second probe, 4.25 → 5.58)
    /// likewise: the derived support r_β (`kb_form` zeroes β from the first point where every V_l^ion is
    /// −Z_v/r + τ_c to v_ion_tail/100) lies within the first point beyond max r_c for every projector —
    /// the 1e-12-relative rule this replaced read the ~1e-12/r identity residual for Al p (5.28 Bohr).
    /// Also: the quartic recovered from the table matches T = −Z_v/r + τ_c at r_loc in value, slope and
    /// curvature to 1e-8, with T′ and T″ from Gauss's law (independent of the construction's grid derivatives).
    #[test]
    fn polynomial_local_carries_the_core_tail() {
        for z in [8u8, 13] { polynomial_local_core_tail_case(z); }
    }

    fn polynomial_local_core_tail_case(z: u8) {
        let a = ground_state(z, SpinMode::Restricted).unwrap();
        let part = partition(&a).unwrap();
        let g = &a.grid;
        let n = g.len();
        let f = LEDGER.f_ladder[0];
        let chans = tm_channels(&a, &part, f).unwrap();
        let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
        let rho_c = core_density(&a, &part);
        let un = unscreen(g, part.z_v, &chans, &occ, &rho_c);
        let tau = core_tail(g, &rho_c, part.n_core as f64);
        let i_from = chans.iter().map(|c| c.i_c).max().unwrap() + 1;
        let kb = kb_form(g, z, part.z_v, &chans, &un, LocalCandidate::Polynomial, rho_c.clone(), &tau);
        assert_eq!(kb.local_l, None);
        assert_eq!(kb.projectors.len(), 2, "both s and p are projectors under the polynomial local");
        // the identity residual the derived support discards (Task 2's bound v_ion_tail/100)
        let resid = un.v_ion.iter().map(|v| (i_from..n).map(|i| (v[i] + part.z_v / g.r[i] - tau[i]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        eprintln!("Z = {z}: max |V_ion + Z_v/r − τ_c| beyond max r_c {resid:.2e}");
        assert!(resid < LEDGER.v_ion_tail / 100.0);
        for p in &kb.projectors {
            let worst = (i_from..n).map(|i| p.dv[i].abs()).fold(0.0, f64::max);
            let sup = p.support(g);
            // β beyond max r_c is the identity residual and nothing else: |β| ≤ (v_ion_tail/100)·|u| there
            assert!((i_from..n).all(|i| p.beta[i].abs() <= LEDGER.v_ion_tail / 100.0 * p.u[i].abs()), "Z = {z}, l = {}: β beyond max r_c exceeds the identity residual", p.l);
            eprintln!("Z = {z}, f = {f}, polynomial local, l = {}: max |δV| beyond max r_c ({:.4}) {worst:.2e}; β support {sup:.4} (first point beyond max r_c {:.4}); E_KB {:.6}, GSS {:.6}",
                      p.l, g.r[i_from - 1], g.r[i_from], p.e_kb, p.gss);
            assert!(worst < LEDGER.v_ion_tail / 100.0, "Z = {z}, l = {}: δV beyond max r_c = {worst:.2e}", p.l);
            // the derived support r_β (Task 3 review, ruling (b)): within the first point beyond max r_c while
            // ρ_pc = ρ_c, for O and Al alike (the 1e-12-relative rule read Al p's residual: 5.28 Bohr)
            assert!(sup <= g.r[i_from], "Z = {z}, l = {}: β support {sup:.4} beyond {:.4}", p.l, g.r[i_from]);
            assert!(p.beta[p.i_beta..].iter().all(|&b| b == 0.0) && p.dv[p.i_beta..].iter().all(|&d| d == 0.0));
        }
        // the S3 defect is visible to this test: without τ_c the s projector's support runs past max r_c
        let bare = kb_form(g, z, part.z_v, &chans, &un, LocalCandidate::Polynomial, rho_c.clone(), &vec![0.0; n]);
        let sup_bare = bare.projectors[0].support(g);
        eprintln!("Z = {z}: without τ_c: s β support {sup_bare:.4}");
        assert!(sup_bare > g.r[i_from + 10], "the τ_c-free local should leave a β tail (S3 probe: 1.97 Bohr), got {sup_bare:.4}");
        // C² at r_loc against T = −Z_v/r + τ_c, exactly: the quartic's analytic slope and curvature against
        // T′ = Z_v/r² + (N_c − Q_c)/r², T″ = −2Z_v/r³ − 2(N_c − Q_c)/r³ − 4πρ_c (Gauss's law, independent of
        // the grid derivatives of τ_c the construction uses; Q_c's cumulative error ≲ 1e-9 and the 5-point
        // truncation on the smooth τ_c bound the difference far below 1e-8)
        let i = g.nearest(kb.r_loc);
        let r = g.r[i];
        let a = quartic_of(g, &kb.v_loc, kb.r_loc);
        let q_c = g.cumulative(&(0..n).map(|k| 4.0 * PI * g.r[k] * g.r[k] * rho_c[k]).collect::<Vec<_>>())[i];
        let nc = part.n_core as f64;
        let t = [-part.z_v / r + tau[i], part.z_v / (r * r) + (nc - q_c) / (r * r), -2.0 * part.z_v / r.powi(3) - 2.0 * (nc - q_c) / r.powi(3) - 4.0 * PI * rho_c[i]];
        let p = [a[0] + a[1] * r * r + a[2] * r.powi(4), 2.0 * a[1] * r + 4.0 * a[2] * r.powi(3), 2.0 * a[1] + 12.0 * a[2] * r * r];
        eprintln!("Z = {z}, r_loc {r:.4}: quartic − T: value {:.2e}, slope {:.2e}, curvature {:.2e}; τ_c(r_loc) {:.3e}; V_loc(0) {:.6}", p[0] - t[0], p[1] - t[1], p[2] - t[2], tau[i], kb.v_loc[0]);
        assert_eq!(kb.v_loc[i], t[0]);
        for k in 0..3 { assert!((p[k] - t[k]).abs() < 1e-8 * t[k].abs().max(1.0), "derivative {k}: quartic {} vs T {}", p[k], t[k]); }
    }

    /// The radial 3-D harmonic oscillator, l = 0, shifted: V = ½r² − 10, ε_n = 2n + 3/2 − 10, with the
    /// analytic eigenfunctions u₀ = (4/√π)^{1/2} r e^{−r²/2} and u₁ = (8/(3√π))^{1/2} r(3/2 − r²)e^{−r²/2}
    /// (∫r²e^{−r²} = √π/4, ∫r²(3/2 − r²)²e^{−r²} = 3√π/8).
    fn oscillator(g: &LogGrid) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sp = PI.sqrt();
        let v = g.r.iter().map(|r| 0.5 * r * r - 10.0).collect();
        let u0 = g.r.iter().map(|r| (4.0 / sp).sqrt() * r * (-0.5 * r * r).exp()).collect();
        let u1 = g.r.iter().map(|r| (8.0 / (3.0 * sp)).sqrt() * r * (1.5 - r * r) * (-0.5 * r * r).exp()).collect();
        (v, u0, u1)
    }

    /// The exact KB spectrum for β = c₀u₀ + c₁u₁: in span{u₀, u₁} the 2×2 matrix diag(ε₀, ε₁) + E_KB ccᵀ,
    /// every other oscillator level unchanged — sorted, below 0.
    fn oscillator_oracle(e_kb: f64, c: [f64; 2]) -> Vec<f64> {
        let (a, b, d) = (-8.5 + e_kb * c[0] * c[0], e_kb * c[0] * c[1], -6.5 + e_kb * c[1] * c[1]);
        let (m, h) = (0.5 * (a + d), (0.25 * (a - d) * (a - d) + b * b).sqrt());
        let mut e = vec![m - h, m + h, -4.5, -2.5, -0.5];
        e.sort_by(|x, y| x.partial_cmp(y).unwrap());
        e
    }

    /// Analytic oracle for the integro-differential solver (`kb_eigenvalues`, `kb_state`) and for the
    /// inertia count, independent of the pseudopotential pipeline: on the oscillator with β = 0.6u₀ + 0.8u₁
    /// and E_KB = ±3, every eigenvalue below 0 matches the 2×2 oracle (E_KB = +3 gives a level ordering that
    /// is not the local one, lifting ε₁ above the untouched −4.5; −3 keeps the local order), the two coupled
    /// eigenfunctions match the oracle's x₀u₀ + x₁u₁ ((x₀, x₁) ∝ (b, λ − a) for the 2×2 [[a, b], [b, d]]),
    /// the inertia count equals the oracle's count on an energy sweep, and both errors are the grid's:
    /// they fall as δ⁴ (ratio per halving in [14, 18], Task 1's window), δ = 0.032 → 0.016 → 0.008. The
    /// eigenvalue ratio is driven by the untouched −0.5 level (plain Numerov; the coupled levels' errors are
    /// ~1e-8 at δ = 0.008); the eigenfunction ratios, over the coupled states only, certify the projector
    /// part's order.
    #[test]
    fn kb_spectrum_matches_the_oscillator_oracle() {
        let c = [0.6, 0.8];
        for e_kb in [3.0, -3.0] {
            let oracle = oscillator_oracle(e_kb, c);
            let (a2, b2) = (-8.5 + e_kb * c[0] * c[0], e_kb * c[0] * c[1]);
            let coupled: Vec<f64> = oracle.iter().copied().filter(|e| ![-4.5, -2.5, -0.5].contains(e)).collect();
            let mut errs = Vec::new();
            let mut uerrs = Vec::new();
            for m in [4.0, 2.0, 1.0] {
                let g = LogGrid::new(LEDGER.r0, LEDGER.r_max, m * LEDGER.delta);
                let (v, u0, u1) = oscillator(&g);
                let beta: Vec<f64> = u0.iter().zip(&u1).map(|(a, b)| c[0] * a + c[1] * b).collect();
                let proj = Some((e_kb, beta.as_slice()));
                let lo = kb_lower_bound(&g, &v, 0, proj);
                let roots = kb_eigenvalues(&g, &v, proj, 0, lo, 0.0, LEDGER.numerics.g2_mesh);
                assert_eq!(roots.len(), oracle.len(), "E_KB = {e_kb}, δ = {}: roots {roots:?} vs oracle {oracle:?}", g.delta);
                let err = roots.iter().zip(&oracle).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                eprintln!("E_KB = {e_kb:+}, δ = {:.3}: roots {roots:.10?}; max |Δ| vs oracle {err:.3e}", g.delta);
                errs.push(err);
                // the coupled eigenfunctions against x₀u₀ + x₁u₁ (sign by overlap), at the solver's own roots
                let mut uerr = 0.0f64;
                for &lam in &coupled {
                    let root = *roots.iter().min_by(|x, y| (*x - lam).abs().partial_cmp(&(*y - lam).abs()).unwrap()).unwrap();
                    let (x0, x1) = (b2, lam - a2);
                    let nx = (x0 * x0 + x1 * x1).sqrt();
                    let exact: Vec<f64> = u0.iter().zip(&u1).map(|(p, q)| (x0 * p + x1 * q) / nx).collect();
                    let u = kb_state(&g, &v, proj, 0, root);
                    let sgn = g.integrate(&u.iter().zip(&exact).map(|(a, b)| a * b).collect::<Vec<_>>()).signum();
                    uerr = uerr.max(u.iter().zip(&exact).map(|(a, b)| (sgn * a - b).abs()).fold(0.0, f64::max));
                }
                eprintln!("E_KB = {e_kb:+}, δ = {:.3}: coupled eigenfunctions max |Δu| {uerr:.3e}", g.delta);
                uerrs.push(uerr);
                for k in 0..=48 {
                    let e = -12.0 + 0.25 * k as f64 + 0.0123;
                    let want = oracle.iter().filter(|&&x| x < e).count();
                    assert_eq!(kb_count_below(&g, &v, proj, 0, e), want, "E_KB = {e_kb}, δ = {}: count below {e}", g.delta);
                }
            }
            let (q1, q2) = (errs[0] / errs[1], errs[1] / errs[2]);
            let (p1, p2) = (uerrs[0] / uerrs[1], uerrs[1] / uerrs[2]);
            eprintln!("E_KB = {e_kb:+}: eigenvalue error ratios {q1:.2}, {q2:.2}; eigenfunction {p1:.2}, {p2:.2}");
            assert!((14.0..=18.0).contains(&q1) && (14.0..=18.0).contains(&q2), "E_KB = {e_kb}: ratios {q1:.2}, {q2:.2}");
            assert!((14.0..=18.0).contains(&p1) && (14.0..=18.0).contains(&p2), "E_KB = {e_kb}: eigenfunction ratios {p1:.2}, {p2:.2}");
        }
    }

    /// G2's refusal branch with G1 passing (Task 3 review N10, S1's "extra" direction): Al with its p local
    /// plus a shallow spherical-shell well at 20 Bohr (−A·e^{−(r−20)²}), A raised until the shell's s level
    /// lies below −0.03 Ha. The shell binds one extra s (and p) state between ε_v,max and 0, far from the
    /// valence (u_3p(20) ~ 1e-4), so G1 stays ghost-free (ε₀^loc < ε_s < ε₁^loc = the shell level) and the
    /// valence eigenvalues stay within g2_eig, while the s count below 0 exceeds the AE atom's 2: refused
    /// as an extra state.
    #[test]
    fn g2_refuses_an_extra_state_with_g1_passing() {
        let t = generate(13).unwrap();
        let g = &t.grid;
        let un = unscreen(g, t.z_v, &t.channels, &t.partition.channels.iter().map(|c| c.occ).collect::<Vec<_>>(), &t.kb.rho_pc);
        let shell: Vec<f64> = g.r.iter().map(|r| (-(r - 20.0).powi(2)).exp()).collect();
        let shell_s = |v: &[f64]| {
            let vs: Vec<f64> = v.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
            (0..6).filter_map(|k| bound_state(g, &vs, 0, k, 0.0, None)).find(|b| (17.0..23.0).contains(&b.r_peak)).map(|b| b.e)
        };
        let (a, v) = (1..=200).map(|k| 0.01 * k as f64).map(|a| (a, t.kb.v_loc.iter().zip(&shell).map(|(v, b)| v - a * b).collect::<Vec<f64>>()))
            .find(|(_, v)| matches!(shell_s(v), Some(e) if e < -0.03)).expect("the shell binds an s level below −0.03 Ha");
        let e_shell = shell_s(&v).unwrap();
        eprintln!("Al + shell well A = {a:.2} Ha: shell s level {e_shell:.5} Ha (ε_p {:.5})", t.partition.channels[1].eps);
        assert!(e_shell > t.partition.channels[1].eps, "the shell level must stay above the valence p");
        let kb = kb_form_with_local(g, 13, t.z_v, &t.channels, &un, v, Some(1), t.kb.rho_pc.clone());
        let err = ghost_gates(g, &kb, &t.screen, &t.partition).expect_err("an extra bound state is a ghost");
        eprintln!("{err}");
        assert!(err.starts_with("G2: l = 0 has") && err.contains("an extra state (ghost)"), "{err}");
    }

    /// G2's other direction (Task 3 review S1): fewer bound states than the all-electron atom is not a
    /// ghost. Al with its p local plus a weak repulsive shell in the AE 4s's tail (+H·e^{−((r−14)/3)²}, H
    /// raised until the s count below 0 on the reference screening drops from 2 to 1): the 4s (−0.0121 Ha,
    /// decay length 6.4 Bohr) is pushed into the continuum while the valence (3p decay length 2.2 Bohr; a
    /// shell at 8 Bohr moved it by 5.8 mHa) moves by less than g2_eig. G1 stays ghost-free, G2 passes, and `g2_missing` records (0, [AE 3s, 4s], [KB 3s]).
    #[test]
    fn g2_records_a_missing_ae_state() {
        let t = generate(13).unwrap();
        let g = &t.grid;
        let un = unscreen(g, t.z_v, &t.channels, &t.partition.channels.iter().map(|c| c.occ).collect::<Vec<_>>(), &t.kb.rho_pc);
        let bump: Vec<f64> = g.r.iter().map(|r| (-((r - 14.0) / 3.0).powi(2)).exp()).collect();
        let s_count = |kb: &KbPotential| {
            let v: Vec<f64> = kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
            let p = kb.projectors.iter().find(|p| p.l == 0).unwrap();
            kb_count_below(g, &v, Some((p.e_kb, &p.beta)), 0, 0.0)
        };
        let (h, kb) = (1..=200).map(|k| 0.005 * k as f64)
            .map(|h| (h, kb_form_with_local(g, 13, t.z_v, &t.channels, &un, t.kb.v_loc.iter().zip(&bump).map(|(v, b)| v + h * b).collect(), Some(1), t.kb.rho_pc.clone())))
            .find(|(_, kb)| s_count(kb) == 1).expect("a weak shell unbinds the 4s");
        let rep = ghost_gates(g, &kb, &t.screen, &t.partition).expect("not a ghost: G1 and G2 pass");
        eprintln!("Al + repulsive shell H = {h:.3} Ha: G2 max |Δε| {:.2e}; g2_missing {:?}", rep.g2_max_eig_error, rep.g2_missing);
        assert_eq!(rep.g2_missing.len(), 1, "{:?}", rep.g2_missing);
        let (l, ae, roots) = &rep.g2_missing[0];
        assert_eq!((*l, ae.as_slice(), roots.len()), (0, t.partition.spectrum[0].as_slice(), 1), "(l, every AE level below top, every KB root)");
        assert!(rep.g1.iter().all(|(_, v)| *v == GssVerdict::GhostFree));
    }

    /// The matching matrix of the spec's derivation, M = [[W_x(u_h,u_i), W_x(u_p,u_i)], [E_KB⟨β|u_h⟩,
    /// E_KB⟨β|u_p⟩ − 1]] at r_m (W in x), evaluated directly (test-only: the brief's form, with `numerov::inward`'s
    /// decaying start); returns (det M, M₀₀ = W_x(u_h,u_i)), so s = det M/(E_KB·M₀₀) needs no other run's W.
    fn matching_det(g: &LogGrid, v: &[f64], e_kb: f64, beta: &[f64], l: u8, e: f64) -> (f64, f64) {
        let n = g.len();
        let i_m = matching_index(beta);
        let gc = g_coeff(g, v, l, e);
        let (y0, y1) = regular_start(g, l, 0.0);
        let (yh, _) = outward(&gc, y0, y1, i_m + 2, g.delta);
        let src: Vec<f64> = (0..=i_m + 2).map(|i| 2.0 * g.r[i].powf(1.5) * beta[i]).collect();
        let yp = outward_sourced(&gc, &src, i_m + 2, g.delta);
        let ie = shooting_end(&gc, g.delta).max(i_m + 4).min(n - 1);
        let yi = inward(&gc, i_m - 2, ie, g.delta);
        let uh: Vec<f64> = (0..=i_m + 2).map(|i| yh[i] * g.r[i].sqrt()).collect();
        let up: Vec<f64> = (0..=i_m + 2).map(|i| yp[i] * g.r[i].sqrt()).collect();
        let ui: Vec<f64> = (0..=4).map(|j| yi[j] * g.r[i_m - 2 + j].sqrt()).collect();
        let d5 = |f: [f64; 5]| (f[0] - 8.0 * f[1] + 8.0 * f[3] - f[4]) / (12.0 * g.delta);
        let around = |a: &[f64]| [a[i_m - 2], a[i_m - 1], a[i_m], a[i_m + 1], a[i_m + 2]];
        let ui5 = [ui[0], ui[1], ui[2], ui[3], ui[4]];
        let w = |a: &[f64]| a[i_m] * d5(ui5) - d5(around(a)) * ui[2];
        let dot = |u: &[f64]| g.partial(&(0..n).map(|i| if i <= i_m { beta[i] * u[i] } else { 0.0 }).collect::<Vec<_>>(), i_m);
        (w(&uh) * (e_kb * dot(&up) - 1.0) - w(&up) * e_kb * dot(&uh), w(&uh))
    }

    /// The stable evaluation is the spec's 2×2 determinant: E_KB·W_r·s = det M/r_m (module doc) — compared
    /// through s, which is free of either run's normalisation (`kb_green`'s u_i has the Dirichlet start,
    /// M's here the decaying one; away from ε = 0 they are the same solution) — checked
    /// where M is well conditioned (O's s projector, the reference screening, ε ∈ [−2, 0], f = 1.6). The
    /// two are different discretisations of one function, so the bound is a-posteriori: each one's own
    /// change between δ and 2δ (same tables subsampled), summed.
    #[test]
    fn green_form_is_the_matching_determinant() {
        let t = generate(8).unwrap();
        let g = &t.grid;
        let gc = LogGrid::new(LEDGER.r0, LEDGER.r_max, 2.0 * LEDGER.delta);
        let sub = |v: &[f64]| (0..gc.len()).map(|i| v[2 * i]).collect::<Vec<f64>>();
        let v: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
        let p = &t.kb.projectors[0];
        let (vc, bc) = (sub(&v), sub(&p.beta));
        for e in [-2.0, -1.5, -1.0, -0.5, -0.2, -0.05] {
            let green_s = |g: &LogGrid, v: &[f64], b: &[f64]| { let (d, w) = kb_green(g, v, p.e_kb, b, p.l, e, matching_index(b)); d / (p.e_kb * w) };
            let m_s = |g: &LogGrid, v: &[f64], b: &[f64]| { let (d, m00) = matching_det(g, v, p.e_kb, b, p.l, e); d / (p.e_kb * m00) };
            let (sg, sm) = (green_s(g, &v, &p.beta), m_s(g, &v, &p.beta));
            let bound = (sg - green_s(&gc, &vc, &bc)).abs() + (sm - m_s(&gc, &vc, &bc)).abs();
            eprintln!("O s, ε = {e}: s (Green) {sg:.12}, s (M) {sm:.12}, Δ {:.2e} (bound {bound:.2e})", sg - sm);
            assert!((sg - sm).abs() <= bound, "ε = {e}: Green {sg} vs M {sm} (bound {bound:.2e})");
        }
    }

    /// `kb_regular` (Task 4's T5 input: the regular self-consistent solution, M's second row, no division)
    /// at the KB root of O's s projector on the reference screening is the semilocal state u^PP on
    /// [0, 1.2 r_c] (the T5 radius), scaled at u^PP's maximum there; a-posteriori bound as in
    /// `kb_reproduces_the_semilocal_channel_on_the_reference_screening` (each function's δ → 2δ change).
    /// Finite at every energy of a sweep through [−3, 0] Ha, including the local channel's eigenvalues.
    #[test]
    fn kb_regular_is_the_semilocal_state_at_the_root() {
        let t = generate(8).unwrap();
        let g = &t.grid;
        let gc = LogGrid::new(LEDGER.r0, LEDGER.r_max, 2.0 * LEDGER.delta);
        let sub = |v: &[f64]| (0..gc.len()).map(|i| v[2 * i]).collect::<Vec<f64>>();
        let v: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
        let p = &t.kb.projectors[0];
        let ch = t.channels.iter().find(|c| c.l == p.l).unwrap();
        let r_d = 1.2 * ch.r_c;
        let scaled = |g: &LogGrid, v: &[f64], b: &[f64], vs: &[f64]| {
            let e = kb_eigenvalues(g, v, Some((p.e_kb, b)), p.l, ch.eps - 0.01, ch.eps + 0.01, 1e-4)[0];
            let ie = g.nearest(r_d);
            let reg = kb_regular(g, v, Some((p.e_kb, b)), p.l, e, ie);
            let sl = bound_state(g, vs, p.l, 0, 0.0, Some(ch.eps)).unwrap().u;
            let ip = (0..=ie).max_by(|&a, &b| sl[a].abs().partial_cmp(&sl[b].abs()).unwrap()).unwrap();
            let lam = sl[ip] / reg[ip];
            ((0..=ie).map(|i| lam * reg[i]).collect::<Vec<f64>>(), sl[..=ie].to_vec())
        };
        let (reg, sl) = scaled(g, &v, &p.beta, &ch.v_scr);
        let (reg2, sl2) = scaled(&gc, &sub(&v), &sub(&p.beta), &sub(&ch.v_scr));
        let k = reg2.len().min(reg.len() / 2 + 1);
        let bound = (0..k).map(|i| (reg[2 * i] - reg2[i]).abs()).fold(0.0, f64::max) + (0..k).map(|i| (sl[2 * i] - sl2[i]).abs()).fold(0.0, f64::max);
        let du = reg.iter().zip(&sl).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("O s: max |λ·kb_regular − u_sl| on [0, {r_d:.3}] {du:.2e} (bound {bound:.2e})");
        assert!(du <= bound, "{du:.2e} vs bound {bound:.2e}");
        let ie = g.nearest(r_d);
        for k in 0..=300 {
            let e = -3.0 + 0.01 * k as f64;
            assert!(kb_regular(g, &v, Some((p.e_kb, &p.beta)), p.l, e, ie).iter().all(|x| x.is_finite()), "E = {e}");
        }
    }

    /// On the reference screening, each projector channel's KB eigenvalue and state reproduce the
    /// semilocal channel (V_l^scr's own nodeless state, i.e. the TM u^PP): O at f = 1.6. The bound is the
    /// a-posteriori one of Task 2's `screened_potential_reproduces_the_reference_state`: each solve's
    /// discretisation error is at most its change between δ and 2δ (the 2δ solve on every other point of
    /// the same tables, β included), so |ε_KB − ε_sl| ≤ |Δ_KB| + |Δ_sl|.
    #[test]
    fn kb_reproduces_the_semilocal_channel_on_the_reference_screening() {
        let t = generate(8).unwrap();
        let g = &t.grid;
        let gc = LogGrid::new(LEDGER.r0, LEDGER.r_max, 2.0 * LEDGER.delta);
        let sub = |v: &[f64]| (0..gc.len()).map(|i| v[2 * i]).collect::<Vec<f64>>();
        let v: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
        for p in &t.kb.projectors {
            let ch = t.channels.iter().find(|c| c.l == p.l).unwrap();
            let kb_root = |g: &LogGrid, v: &[f64], b: &[f64]| {
                let r = kb_eigenvalues(g, v, Some((p.e_kb, b)), p.l, ch.eps - 0.01, ch.eps + 0.01, 1e-4);
                assert_eq!(r.len(), 1, "l = {}: roots near ε {r:?}", p.l);
                r[0]
            };
            let e_kb = kb_root(g, &v, &p.beta);
            let e_kb2 = kb_root(&gc, &sub(&v), &sub(&p.beta));
            let sl = bound_state(g, &ch.v_scr, p.l, 0, 0.0, Some(ch.eps)).unwrap();
            let sl2 = bound_state(&gc, &sub(&ch.v_scr), p.l, 0, 0.0, Some(ch.eps)).unwrap();
            let bound = (e_kb - e_kb2).abs() + (sl.e - sl2.e).abs();
            let u = kb_state(g, &v, Some((p.e_kb, &p.beta)), p.l, e_kb);
            let du = u.iter().zip(&sl.u).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let u2 = kb_state(&gc, &sub(&v), Some((p.e_kb, &sub(&p.beta))), p.l, e_kb2);
            let bound_u = (0..gc.len()).map(|i| (u[2 * i] - u2[i]).abs()).fold(0.0, f64::max)
                + (0..gc.len()).map(|i| (sl.u[2 * i] - sl2.u[i]).abs()).fold(0.0, f64::max);
            eprintln!("O l = {}: ε_KB {e_kb:.12} ε_sl {:.12} Δ {:+.2e} (bound {bound:.2e}); max|u_KB − u_sl| {du:.2e} (bound {bound_u:.2e})", p.l, sl.e, e_kb - sl.e);
            assert!((e_kb - sl.e).abs() <= bound, "l = {}: ε_KB {e_kb} vs ε_sl {} (bound {bound:.2e})", p.l, sl.e);
            assert!(du <= bound_u, "l = {}: max |u_KB − u_sl| = {du:.2e} (bound {bound_u:.2e})", p.l);
        }
    }
}
