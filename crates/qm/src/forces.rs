//! Hellmann-Feynman forces on Gaussian-smeared nuclei. Atomic units (Hartree/Bohr).
//!
//! # What is differentiated
//! The SCF of `crate::scf` minimises the *discrete* functional
//!   E_h[{ψ}; R] = T_s[ψ] + h³ Σ_p ρ_p V_nuc(r_p; R) + E_H[ρ] + E_xc[ρ] + E_nn(R),
//!   V_nuc(r; R) = Σ_I V_I(r − R_I),   V_I(d) = −Z_I g(d),   g(d) = erf(d/(σ√2))/d,
//! over orbitals ψ that live on a grid which does not move with the nuclei. Only E_ext and
//! E_nn depend on R explicitly. At the self-consistent point E_h is stationary with respect
//! to every orthonormality-preserving variation of ψ, and ∂ψ/∂R_I is such a variation inside
//! the *same* fixed basis, so
//!   dE_h/dR_I = ∂E_h/∂R_I |_ψ + (∂E_h/∂ψ)(∂ψ/∂R_I) = ∂E_h/∂R_I |_ψ.
//! There is no Pulay term (a basis that moved with R_I — atom-centred functions — would add
//! one). The Hellmann-Feynman force is therefore the *exact* gradient of the discrete energy
//! the SCF reports, including its egg-box ripple, up to the non-self-consistency of the state
//! (first order in Δρ and in the eigensolver residual, both ≤ 1e-8 here).
//!
//! # Sign chain
//! V_I depends on r and R_I only through u = r − R_I, so ∂V_I/∂R_I = −∇_u V_I = −∇_r V_I and
//!   F_I = −∂E_h/∂R_I = −h³ Σ_p ρ_p ∂V_I(r_p − R_I)/∂R_I − ∂E_nn/∂R_I
//!       = +∫ ρ ∇_r V_I d³r + F_I^{nn}.
//! Physical check: V_I = −Z_I g(d) with g decreasing, so ∇_r V_I = −Z_I g′(d)(r − R_I)/d points
//! *away* from the nucleus (g′ < 0) and F_I pulls the nucleus toward the electron density —
//! the attractive direction, as it must be.
//!
//! # g′(d) and its d → 0 limit
//! With x = d/a, a = σ√2:  g(d) = erf(x)/d,
//!   g′(d) = [ d · (2/√π) e^{−x²}/a − erf(x) ]/d² = [ (2/√π) x e^{−x²} − erf(x) ]/d².
//! Series: erf(x) = (2/√π)(x − x³/3 + x⁵/10 − …), x e^{−x²} = x − x³ + x⁵/2 − …, so the bracket
//! is (2/√π)(−2x³/3 + 2x⁵/5 − …) and
//!   g′(d) = −(4/(3√π)) d/a³ + O(d³)  →  g′(0) = 0  (g is even in d: erf is odd, d is odd).
//! Hence ∇_r V_I = −Z_I g′(d)(r − R_I)/d → +Z_I (4/(3√π a³)) (r − R_I) → 0 linearly: the
//! integrand is smooth through the nucleus and a nucleus sitting exactly on a grid point
//! contributes nothing from that point. The code skips d < 1e-8 (the `nuclear.rs` coincidence
//! threshold): the exact contribution of such a point is below
//! Z ρ(R) h³ (4/(3√π a³)) d ≈ 17 Z ρ h³ d ≈ 1e-9 Ha/Bohr at σ = 0.25, and evaluating the
//! two-term bracket there would divide rounding noise ε(2/√π)x by d², i.e. an absolute error
//! ≈ ε (2/√π)/(a d) in g′ — the same 1e-9 Ha/Bohr size after the ρ h³ weight. Both are far
//! below any force tolerance; the skip is exact to that level and avoids the 0/0.
//!
//! # Nuclear–nuclear part
//! E_nn = Σ_{I<J} Z_I Z_J/|R_I − R_J| (point charges: the nuclei never see each other smeared),
//!   F_I^{nn} = −∂E_nn/∂R_I = Σ_{J≠I} Z_I Z_J (R_I − R_J)/|R_I − R_J|³,
//! repulsive (points from J to I). Coincident nuclei (d < 1e-8) are the same error as in
//! `nuclear_repulsion`.
//!
//! # Expected size of the egg-box force and of HF vs finite differences
//! Task 9 measured E(h/2) − E(0) = +3.3 µHa for H at σ = h = 0.25 in a 12 Bohr box and read it as
//! the egg-box ripple. It is not (M7 Task 6 review, ruling C): E(h) − E(0) = 4.00·[E(h/2) − E(0)]
//! there, a parabola, where a lattice-periodic ripple would give E(h) = E(0). The 3.3 µHa (3.3329e-6
//! re-measured) is the restoring force of the Dirichlet walls on the off-centre atom, smooth and
//! falling 19× in a 16 Bohr box. The periodic egg-box is the remainder, 2.2–2.5e-8 Ha at h/2 in both
//! boxes, i.e. A ≈ 1.2e-8 and |F_egg| ≈ A·2π/h ≈ 3e-7 per axis, ≈ 5e-7 Ha/Bohr at the (±h/4)³ probe
//! (what the frozen-atom estimate gives). The ≤ 4e-5 Ha/Bohr quoted here before was the wall term.
//! Both are errors against the *continuum*, and HF and a finite difference of E_h contain them
//! consistently, so they enter HF − FD only through the FD truncation, ≲ δ² |F‴| ≈ 1e-7 at δ = 0.01. The HF − FD discrepancy is instead bounded
//! by (i) central-difference truncation δ²|E‴|/6 on the bond curve — for H₂ a Morse estimate
//! E‴ ≈ −6 D a³ with D ≈ 0.17 Ha, a ≈ 1.0 Bohr⁻¹ gives |E‴| ≈ 1.1 Ha/Bohr³, i.e. ≈ 2e-5 at
//! δ = 0.01 (×4 at δ = 0.02); (ii) energy noise over 2δ — the SCF energy is second order in the
//! density and orbital residuals (Δρ < 1e-9, LOBPCG residual < 1e-8 ⇒ ≪ 1e-12 Ha) and first
//! order in the Poisson residual (1e-10 relative ⇒ ~1e-10 Ha), so ≲ 1e-8 Ha/Bohr; (iii) the
//! first-order non-stationarity of the KS fixed point for E_h, if the discrete Hartree map
//! ρ ↦ V_H (Dirichlet Laplacian + multipole boundary) is not exactly symmetric — of the order
//! of the Poisson boundary error (3.5e-5 in V_H, Task 5) times ∫|∂ρ/∂R| ≈ 1, so ≲ 1e-5. The
//! test's 2e-3 Ha/Bohr bound is thus ~100× above the expected floor; the measured value and
//! its δ-scaling are recorded in the Task 10 report and commit.
//!
//! # The forces do not sum to zero
//! On a grid Σ_I F_I ≠ 0 in general: the egg-box effect breaks translational invariance (moving
//! every nucleus by the same δ changes E_h, so the total force is not −dE/dδ = 0). For the
//! symmetric H₂ tests the sum is ~1e-13 only because the geometry is mirror-symmetric about a
//! grid plane; for a general geometry it carries the egg-box force (≈ 5e-7 Ha/Bohr per smeared H at
//! σ = h = 0.25, above) and, in a box, the wall's restoring force on the whole system (≈ 3e-5 at
//! 0.06 Bohr off-centre in a 12 Bohr box for H). Dynamics (M3/M4) must not assert total-momentum
//! conservation to better than that.
//!
//! # Precision
//! The force integrands are formed pointwise and reduced through `Grid::integrate` (Neumaier);
//! the nuclear sum is a short f64 loop.
//!
//! # Pseudised ions (M7; spec §5.4, §5.6, §6.6; plan Task 7)
//! `forces` dispatches on `state.ionic`: `None` is everything above, the same code bit for bit
//! (`all_electron_forces`; `scf::none_policy_is_bitwise_today` pins the bits). Under an
//! [`IonicSetup`] the SCF minimises (`scf` module doc, "Ionic species (M7)")
//!   E_h = T_s + h³Σ_p ρ_p V_loc(r_p; R) + E_NL(R) + E_H[ρ] + E_xc[ρ↑ + ½ρ_pc(R), ρ↓ + ½ρ_pc(R)] + E_ii(R),
//! ρ_σ = Σ_i f_iσ ψ_iσ², E_NL = Σ_iσ f_iσ Σ_{I,lm} E_KB c_lm[ψ_iσ]², c_lm = h³Σ_p β_lm(r_p − R_I) ψ(r_p).
//! The no-Pulay argument above is unchanged: the grid does not move with R, E_h is stationary in ψ
//! at fixed occupations (Janak: ∂E/∂f_i = ε_i, so with the f frozen the weighted functional is
//! stationary, `scf` "Occupations"), and T_s, E_H do not depend on R. So
//!   F_I = −dE_h/dR_I = −∂E_h/∂R_I |_{ψ, f},
//! and every R-dependence is a function of u = r − R_I (∂/∂R_I = −∇_u). Five explicit terms, each
//! the negative R-derivative of one *frozen* energy (ρ, ρ_σ, ψ and f held fixed) — which is why each
//! is a separate `pub(crate)` function, checked against central differences of its own energy
//! (`force_terms_are_exact_derivatives_of_frozen_energies`, plan review S2: HF-vs-FD on a converged
//! state is blind below its bound, a missing or sign-flipped term is O(1e-3–1) there):
//! - **Local, long range** (`local_lr_forces`): V_LR,I(d) = −q_I erf(d/a)/d, a = σ√2, with
//!   (q, σ) = (Z_v, σ_LR) for a pseudised ion and today's (Z, σ_AE) for an all-electron species in the
//!   same setup. F^LR_I = −h³Σ_p ρ_p ∂V_LR,I/∂R_I = +h³Σ_p ρ_p V_LR′(d_p)(r_p − R_I)/d_p, V_LR′ analytic
//!   (`FilteredTable::dv_lr`; `g_prime` for the all-electron centre: V′ = −Z g′). Today's integrand
//!   with Z → Z_v, σ → σ_LR; attractive (V_LR′ > 0 pulls R toward the density).
//! - **Local, short range** (`local_sr_forces`): F^SR_I = +h³Σ_{p ∈ sphere} ρ_p V_SR′(d_p)(r_p − R_I)/d_p,
//!   V_SR′ = `dv_sr`, the exact derivative of the same uniform-table interpolant `v_sr` samples (plan
//!   review Q2), over the points within `sr_radius` (the table is exactly 0 beyond: no tail branch).
//!   F^LR + F^SR is the force of the one local potential V_loc = V_LR + V_SR; the split is exact
//!   (§5.1), so only the sum is physical (7c re-review: a lone O's F^LR and F^SR are not separately
//!   meaningful egg-box terms).
//! - **Nonlocal** (`nonlocal_forces`): ∂c_lm/∂R_I = −h³Σ_p ∇_r β_lm ψ = −⟨∇β_lm|ψ⟩, so
//!   F^NL_I = −Σ_iσ f_iσ Σ_lm E_KB 2c_lm ∂c_lm/∂R_I = +2 Σ_iσ f_iσ Σ_lm E_KB c_lm ⟨∇β_lm|ψ_iσ⟩,
//!   ∇β_lm = t′(d)N_l R_lm d̂ + t(d)N_l ∇R_lm (`NonlocalOp::build_with_gradients`, t′ = `dbeta_radial`).
//! - **Partial core** (`partial_core_forces`): with n_σ = ρ_σ + ½ρ_pc, ∂E_xc/∂ρ_pc = ½(v_xc↑ + v_xc↓) = v̄
//!   on n_σ — the same v̄ as the SCF and T8(e) (NLCC ruling, Ruling 6) — so
//!   F^pc_I = +h³Σ_{p ∈ sphere} v̄_p ρ_pc′(d_p)(r_p − R_I)/d_p, ρ_pc′ = `drho_pc_at`. For a lone spherical
//!   atom this is `radial::filter::pc_lattice_force`, the same lattice sum
//!   (`partial_core_force_is_the_radial_lattice_sum`).
//! - **Ion–ion** (`ion_ion_forces`): E_ii = Σ_{I<J} q_I q_J/R_IJ (valence charges), F_I = Σ_{J≠I} q_I q_J
//!   (R_I − R_J)/R_IJ³, as above.
//! Every radial derivative is the exact derivative of the interpolant the energy samples, so the HF
//! force is the gradient of the sampled E_h to rounding and the SCF residual (on-site points d < 1e-8
//! are skipped: every radial integrand is odd and smooth through d = 0, as g′ above). The force is
//! formed from the occupations f_iσ (`KsState.occ_*`, spec §5.7): ρ in the local and partial-core terms
//! is the weighted output density, and the nonlocal sum carries f_iσ.
//! Expected HF − FD for the molecules here (OH, plan Task 7): the central difference's truncation
//! δ²|E‴|/6 with a Morse estimate (D ≈ 0.17 Ha, a ≈ 1.2 Bohr⁻¹: |E‴| ≈ 6Da³ ≈ 1.8 Ha/Bohr³) is ≈ 3e-5 at
//! δ = 0.01 (×4 at 0.02), the HF − FD(0.01) floor of ≈ 2–3e-5 the plan quotes. Richardson
//! (4D(δ) − D(2δ))/3 removes the δ² term (the δ⁴ remainder is ~1e-8), leaving the energy noise
//! (≲ 1e-8 Ha at tol 1e-9, i.e. ≲ 1e-6 Ha/Bohr after the combination) and the Hartree-map asymmetry
//! (≲ 1e-5, above). The 1e-4 bound on |HF − Richardson| is 5× the 2e-5 floor (spec §6.6 revision,
//! plan review S2).

use crate::error::QmError;
use crate::grid::{neumaier, Grid};
use crate::ionic::{Ionic, IonicSetup};
use crate::nonlocal::NonlocalOp;
use crate::nuclear::{partial_core_field, points_within, Nucleus};
use crate::scf::{xc_with_core, KsState};
use std::f64::consts::PI;

/// Below this distance (Bohr) a grid point is treated as on the nucleus (module doc: the
/// exact contribution is < 1e-9 Ha/Bohr and the two-term g′ would be rounding noise). Same
/// value as the coincidence threshold of `nuclear::nuclear_repulsion`.
const ON_SITE: f64 = 1e-8;

/// g′(d) for g(d) = erf(d/(σ√2))/d (module doc); exactly 0 for d < `ON_SITE`.
fn g_prime(d: f64, sigma: f64) -> f64 {
    if d < ON_SITE { return 0.0; }
    let u = d / (sigma * 2f64.sqrt());
    ((2.0 / PI.sqrt()) * u * (-u * u).exp() - libm::erf(u)) / (d * d)
}

/// Force on each nucleus (Hartree/Bohr) from the converged state: the Hellmann-Feynman
/// electronic part ∫ρ ∇_r V_I plus the point nuclear–nuclear repulsion (module doc); under an
/// ionic setup (`state.ionic`) the pseudised terms of "Pseudised ions (M7)".
/// `grid` and `sigma` must be the grid and smearing width the state was converged with
/// (kept as explicit parameters; panics if they differ from `state.grid`/`state.sigma`).
/// Errors on coincident nuclei (the repulsion is undefined), with the lower index first.
pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError> {
    assert_eq!(&state.grid, grid, "KsState was converged on a different grid");
    assert!((state.sigma - sigma).abs() < 1e-15, "KsState was converged with sigma = {}, forces called with {}", state.sigma, sigma);
    debug_assert!(sigma > 0.0, "sigma must be positive, got {sigma}");
    match state.ionic.as_deref() {
        None => all_electron_forces(grid, state, nuclei, sigma),
        Some(setup) => pseudo_forces(grid, state, nuclei, setup),
    }
}

/// Today's `forces` body (before M7), moved without change: the `None` path, bit for bit.
fn all_electron_forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError> {
    let rho = state.rho();
    let mut out = vec![[0.0f64; 3]; nuclei.len()];
    for (i, nuc) in nuclei.iter().enumerate() {
        let z = nuc.z as f64;
        // electronic part: F = ∫ ρ ∇_r V_I,  ∇_r V_I = −Z g′(d) (r − R)/d
        let mut comp = [vec![0.0; grid.len()], vec![0.0; grid.len()], vec![0.0; grid.len()]];
        for p in 0..grid.len() {
            let r = grid.point(p);
            let dv = [r[0] - nuc.pos[0], r[1] - nuc.pos[1], r[2] - nuc.pos[2]];
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            if d < ON_SITE { continue; }
            let gp = -z * g_prime(d, sigma) / d;
            for a in 0..3 { comp[a][p] = rho[p] * gp * dv[a]; }
        }
        for a in 0..3 { out[i][a] = grid.integrate(&comp[a]); }
        // nuclear–nuclear part: Σ_{J≠I} Z_I Z_J (R_I − R_J)/|R_I − R_J|³
        for (j, other) in nuclei.iter().enumerate() {
            if i == j { continue; }
            let dv = [nuc.pos[0] - other.pos[0], nuc.pos[1] - other.pos[1], nuc.pos[2] - other.pos[2]];
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            if d < ON_SITE { return Err(QmError::CoincidentNuclei(i.min(j), i.max(j))); }
            let s = z * other.z as f64 / (d * d * d);
            for a in 0..3 { out[i][a] += s * dv[a]; }
        }
    }
    Ok(out)
}

/// Σ_p w_p · V′(d_p)(r_p − R)/d_p × h³ over `points` (every grid point when None), one Neumaier sum per
/// component; points with d < `ON_SITE` contribute 0 (every radial integrand here is odd and smooth through
/// d = 0, module doc).
fn radial_force(grid: &Grid, points: Option<&[usize]>, centre: [f64; 3], w: &[f64], dvdd: &dyn Fn(f64) -> f64) -> [f64; 3] {
    let all: Vec<usize>;
    let pts = match points { Some(p) => p, None => { all = (0..grid.len()).collect(); &all } };
    let mut comp: [Vec<f64>; 3] = [Vec::with_capacity(pts.len()), Vec::with_capacity(pts.len()), Vec::with_capacity(pts.len())];
    for &p in pts {
        let r = grid.point(p);
        let d = [r[0] - centre[0], r[1] - centre[1], r[2] - centre[2]];
        let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if dd < ON_SITE { continue; }
        let s = w[p] * dvdd(dd) / dd;
        for a in 0..3 { comp[a].push(s * d[a]); }
    }
    std::array::from_fn(|a| neumaier(comp[a].iter().copied()) * grid.dv())
}

/// The pseudised forces (module doc, "Pseudised ions (M7)"): ion–ion + local LR + local SR + nonlocal + partial
/// core, with the state's occupations.
fn pseudo_forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], setup: &IonicSetup) -> Result<Vec<[f64; 3]>, QmError> {
    let parts = pseudo_terms(grid, state, nuclei, setup)?;
    Ok((0..nuclei.len()).map(|i| std::array::from_fn(|a| neumaier(parts.iter().map(|p| p[i][a])))).collect())
}

/// The five pseudised force terms of a state converged under an ionic setup, each per nucleus, in the order
/// [ion–ion, local LR, local SR, nonlocal, partial core] (module doc, "Pseudised ions (M7)"); `forces` is their
/// per-component Neumaier sum in this order. The per-term reference of the GPU force kernels (M7 Task 8b: `qm_gpu`
/// `GpuSolver::force_terms` on the same state). Panics if the state has no ionic setup (the all-electron force is one
/// term); the same grid/σ checks and the coincident-nuclei error as `forces`.
pub fn pseudo_force_terms(grid: &Grid, state: &KsState, nuclei: &[Nucleus]) -> Result<[Vec<[f64; 3]>; 5], QmError> {
    assert_eq!(&state.grid, grid, "KsState was converged on a different grid");
    let setup = state.ionic.as_deref().expect("pseudo_force_terms: the state has no ionic setup");
    pseudo_terms(grid, state, nuclei, setup)
}

fn pseudo_terms(grid: &Grid, state: &KsState, nuclei: &[Nucleus], setup: &IonicSetup) -> Result<[Vec<[f64; 3]>; 5], QmError> {
    let ii = ion_ion_forces(nuclei, setup)?;
    let rho = state.rho();
    let orbitals: Vec<&[f64]> = state.orbitals_up.iter().chain(&state.orbitals_dn).map(|v| v.as_slice()).collect();
    let occ: Vec<f64> = state.occ_up.iter().chain(&state.occ_dn).copied().collect();
    Ok([
        ii,
        local_lr_forces(grid, &rho, nuclei, setup),
        local_sr_forces(grid, &rho, nuclei, setup),
        nonlocal_forces(grid, &orbitals, &occ, nuclei, setup),
        partial_core_forces(grid, &state.rho_up, &state.rho_dn, nuclei, setup),
    ])
}

/// −∂/∂R_I h³Σ_p ρ_p V_LR(r_p; R) (module doc): every centre's smeared erf — (Z_v, σ_LR) for a pseudised ion,
/// today's (Z, σ_AE) for an all-electron species in the same setup.
pub fn local_lr_forces(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { sigma } => { let (z, s) = (n.z as f64, *sigma); radial_force(grid, None, n.pos, rho, &|d| -z * g_prime(d, s)) }
        Ionic::Pseudo(t) => radial_force(grid, None, n.pos, rho, &|d| t.dv_lr(d)),
    }).collect()
}

/// −∂/∂R_I h³Σ_p ρ_p V_SR(|r_p − R_I|) (module doc): the one V_SR table's interpolant derivative over its sphere;
/// 0 for an all-electron species.
pub fn local_sr_forces(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { .. } => [0.0; 3],
        Ionic::Pseudo(t) => radial_force(grid, Some(&points_within(grid, n.pos, t.sr_radius())), n.pos, rho, &|d| t.dv_sr(d)),
    }).collect()
}

/// −∂/∂R_I Σ_i f_i Σ_lm E_KB c_lm[ψ_i]² = +2 Σ_i f_i Σ_lm E_KB c_lm ⟨∇β_lm|ψ_i⟩ (module doc); `orbitals[i]`
/// carries the occupation `occ[i]` (spin orbitals; f = 0 orbitals are skipped).
pub fn nonlocal_forces(grid: &Grid, orbitals: &[&[f64]], occ: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    assert_eq!(orbitals.len(), occ.len(), "one occupation per orbital");
    let nl = NonlocalOp::build_with_gradients(grid, nuclei, setup);
    let mut terms: Vec<[Vec<f64>; 3]> = vec![[Vec::new(), Vec::new(), Vec::new()]; nuclei.len()];
    for s in &nl.spheres {
        for (psi, &f) in orbitals.iter().zip(occ) {
            if f == 0.0 { continue; }
            for (b, gb) in s.beta.iter().zip(&s.grad) {
                let c = neumaier(s.points.iter().zip(b).map(|(&p, &bv)| bv * psi[p])) * nl.dv;
                for a in 0..3 {
                    let gpsi = neumaier(s.points.iter().zip(gb).map(|(&p, gv)| gv[a] * psi[p])) * nl.dv;
                    terms[s.ion][a].push(2.0 * f * s.e_kb * c * gpsi);
                }
            }
        }
    }
    terms.into_iter().map(|t| std::array::from_fn(|a| neumaier(t[a].iter().copied()))).collect()
}

/// −∂/∂R_I E_xc[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc] at fixed ρ_σ = +h³Σ_p v̄_p ρ_pc′(d_p)(r_p − R_I)/d_p, v̄ = ½(v_xc↑ + v_xc↓)
/// on ρ_σ + ½ρ_pc (the full ρ_pc of every ion; module doc). 0 for an all-electron species and the no-core branch.
pub fn partial_core_forces(grid: &Grid, rho_up: &[f64], rho_dn: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    let cored = |n: &Nucleus| matches!(setup.ionic(n.z), Ionic::Pseudo(t) if t.pc_radius > 0.0);
    if !nuclei.iter().any(cored) { return vec![[0.0; 3]; nuclei.len()]; }
    let pc = partial_core_field(grid, nuclei, setup);
    // the SCF's own XC call (`scf::xc_with_core`): the same v̄ by construction, not by copy
    let xc = xc_with_core(rho_up, rho_dn, Some(&pc));
    let v_avg: Vec<f64> = xc.v_up.iter().zip(&xc.v_dn).map(|(a, b)| 0.5 * (a + b)).collect();
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::Pseudo(t) if t.pc_radius > 0.0 => radial_force(grid, Some(&points_within(grid, n.pos, t.pc_radius)), n.pos, &v_avg, &|d| t.drho_pc_at(d)),
        _ => [0.0; 3],
    }).collect()
}

/// −∂/∂R_I Σ_{I<J} q_I q_J/R_IJ with valence charges (module doc); coincident nuclei are an error, lower index
/// first (as `all_electron_forces`).
pub fn ion_ion_forces(nuclei: &[Nucleus], setup: &IonicSetup) -> Result<Vec<[f64; 3]>, QmError> {
    let mut out = vec![[0.0f64; 3]; nuclei.len()];
    for i in 0..nuclei.len() {
        for j in 0..nuclei.len() {
            if i == j { continue; }
            let dv = [nuclei[i].pos[0] - nuclei[j].pos[0], nuclei[i].pos[1] - nuclei[j].pos[1], nuclei[i].pos[2] - nuclei[j].pos[2]];
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            if d < ON_SITE { return Err(QmError::CoincidentNuclei(i.min(j), i.max(j))); }
            let s = setup.valence_charge(nuclei[i].z) * setup.valence_charge(nuclei[j].z) / (d * d * d);
            for a in 0..3 { out[i][a] += s * dv[a]; }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::nuclear::Nucleus;
    use crate::scf::{scf, ScfSettings, System};

    fn h2(r: f64) -> System {
        System::new(vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }], 0, None)
    }

    fn energy_and_force(g: &Grid, r: f64) -> (f64, Vec<[f64; 3]>) {
        let mut s = ScfSettings::for_grid(g);
        s.tol = 1e-9; s.eig_tol = 1e-8;
        let sys = h2(r);
        let (state, stats) = scf(g, &sys, &s, None).unwrap();
        (stats.energy.total, forces(g, &state, &sys.nuclei, s.sigma).unwrap())
    }

    #[test]
    fn h2_forces_are_equal_and_opposite() {
        let g = Grid::cube(49, 0.25);
        let (_, f) = energy_and_force(&g, 1.4);
        for a in 0..3 { assert!((f[0][a] + f[1][a]).abs() < 1e-6, "{f:?}"); }
        assert!(f[0][1].abs() < 1e-6 && f[0][2].abs() < 1e-6, "off-axis force {f:?}");
    }

    #[test]
    fn h2_force_matches_finite_difference_energy() {
        // E depends on the nuclei only through R = x₀ − x₁, so −∂E/∂x₀ = −dE/dR exactly.
        // Keep nucleus 1 fixed and move nucleus 0 by ±d so the finite difference is a true ∂E/∂x₀.
        let g = Grid::cube(49, 0.25);
        let d = 0.01;
        let mut s = ScfSettings::for_grid(&g);
        s.tol = 1e-9; s.eig_tol = 1e-8;
        let energy_at = |x0: f64| {
            let sys = System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
            scf(&g, &sys, &s, None).unwrap().1.energy.total
        };
        let f_fd = -(energy_at(0.7 + d) - energy_at(0.7 - d)) / (2.0 * d);
        let (_, f) = energy_and_force(&g, 1.4);
        assert!((f[0][0] - f_fd).abs() < 2e-3, "HF {} vs FD {}", f[0][0], f_fd);
    }

    #[test]
    fn force_sign_pushes_toward_equilibrium() {
        let g = Grid::cube(49, 0.25);
        let (_, f_short) = energy_and_force(&g, 1.0); // compressed: nuclei repel ⇒ F_x0 > 0
        let (_, f_long) = energy_and_force(&g, 2.5);  // stretched: attract ⇒ F_x0 < 0
        assert!(f_short[0][0] > 0.0, "{f_short:?}");
        assert!(f_long[0][0] < 0.0, "{f_long:?}");
    }

    #[test]
    #[should_panic(expected = "KsState was converged with sigma")]
    fn forces_reject_a_state_converged_with_another_sigma() {
        let g = Grid::cube(17, 0.5);
        let mut s = ScfSettings::for_grid(&g);
        s.tol = 1e-4; s.eig_tol = 1e-4;
        let sys = System::new(vec![Nucleus { pos: [0.0; 3], z: 1 }], 0, None);
        let (state, _) = scf(&g, &sys, &s, None).unwrap();
        let _ = forces(&g, &state, &sys.nuclei, s.sigma + 0.1);
    }

    // ---- M7 Task 7: the pseudised terms ----

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 { ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() }

    /// Σ_p |w_p V′(d_p)(r_p − R)_a/d_p|·h³ per component: the gross of a radial force term.
    fn gross_radial(g: &Grid, pts: Option<&[usize]>, c: [f64; 3], w: &[f64], dvdd: &dyn Fn(f64) -> f64) -> [f64; 3] {
        let all: Vec<usize> = (0..g.len()).collect();
        let mut out = [0.0f64; 3];
        for &p in pts.unwrap_or(&all) {
            let r = g.point(p);
            let d = [r[0] - c[0], r[1] - c[1], r[2] - c[2]];
            let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if dd < ON_SITE { continue; }
            let s = (w[p] * dvdd(dd) / dd).abs();
            for a in 0..3 { out[a] += s * d[a].abs(); }
        }
        out.map(|x| x * g.dv())
    }

    /// The gross sum G[term][ion][axis] of each pseudised force term (Task 7 review M3), in the order LR, SR, NL,
    /// ρ_pc, ion–ion: Σ of the absolute values of the pointwise contributions whose signed sum is the force. For NL
    /// the contributions are 2f|E_KB c_lm|·h³|∂_aβ_lm ψ| per orbital, sphere, m and point.
    #[allow(clippy::too_many_arguments)]
    fn frozen_gross(g: &Grid, rho: &[f64], up: &[f64], dn: &[f64], orbitals: &[&[f64]], occ: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> [Vec<[f64; 3]>; 5] {
        use crate::ionic::Ionic;
        let lr = nuclei.iter().map(|n| match setup.ionic(n.z) {
            Ionic::AllElectron { sigma } => { let (z, s) = (n.z as f64, *sigma); gross_radial(g, None, n.pos, rho, &|d| -z * g_prime(d, s)) }
            Ionic::Pseudo(t) => gross_radial(g, None, n.pos, rho, &|d| t.dv_lr(d)),
        }).collect();
        let sr = nuclei.iter().map(|n| match setup.ionic(n.z) {
            Ionic::Pseudo(t) => gross_radial(g, Some(&points_within(g, n.pos, t.sr_radius())), n.pos, rho, &|d| t.dv_sr(d)),
            _ => [0.0; 3],
        }).collect();
        let nl_op = NonlocalOp::build_with_gradients(g, nuclei, setup);
        let mut nl = vec![[0.0f64; 3]; nuclei.len()];
        for s in &nl_op.spheres {
            for (psi, &f) in orbitals.iter().zip(occ) {
                for (b, gb) in s.beta.iter().zip(&s.grad) {
                    let c = neumaier(s.points.iter().zip(b).map(|(&p, &bv)| bv * psi[p])) * nl_op.dv;
                    for a in 0..3 {
                        let gpsi: f64 = s.points.iter().zip(gb).map(|(&p, gv)| (gv[a] * psi[p]).abs()).sum::<f64>() * nl_op.dv;
                        nl[s.ion][a] += 2.0 * f * (s.e_kb * c).abs() * gpsi;
                    }
                }
            }
        }
        let pc_field = partial_core_field(g, nuclei, setup);
        let xc = xc_with_core(up, dn, Some(&pc_field));
        let v_avg: Vec<f64> = xc.v_up.iter().zip(&xc.v_dn).map(|(a, b)| 0.5 * (a + b)).collect();
        let pc = nuclei.iter().map(|n| match setup.ionic(n.z) {
            Ionic::Pseudo(t) if t.pc_radius > 0.0 => gross_radial(g, Some(&points_within(g, n.pos, t.pc_radius)), n.pos, &v_avg, &|d| t.drho_pc_at(d)),
            _ => [0.0; 3],
        }).collect();
        let ii = (0..nuclei.len()).map(|i| {
            let mut out = [0.0f64; 3];
            for j in (0..nuclei.len()).filter(|&j| j != i) {
                let d: [f64; 3] = std::array::from_fn(|a| nuclei[i].pos[a] - nuclei[j].pos[a]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                for a in 0..3 { out[a] += (setup.valence_charge(nuclei[i].z) * setup.valence_charge(nuclei[j].z) * d[a] / (r * r * r)).abs(); }
            }
            out
        }).collect();
        [lr, sr, nl, pc, ii]
    }

    /// Plan review S2: every pseudised force term is the exact negative R-derivative of its own frozen energy — ρ,
    /// the spin densities, the orbitals and their occupations held fixed — checked against Richardson-extrapolated
    /// central differences (δ = 1e-4, 2e-4) to 1e-8 + 1e-9·G, G the term's gross sum per component
    /// (`frozen_gross`). The uniform tables' 4-point interpolants are C⁰ at their nodes (Δ = h/256): a δ-step that
    /// straddles a node sees the derivative jump J = Δ³f⁗/6, so the FD error scales with the gross, not the net,
    /// and as √δ (Task 7 review M3, measured on NL ion 0: 1.64e-8 / 5.68e-9 / 2.10e-9 at δ = 1e-3 / 1e-4 / 1e-5,
    /// 2.9× per decade against √10; 1.5e-10 of its gross 37.5 per component, against a net of 6e-7). The plan's
    /// 1e-8 + 1e-6·|F_net| scaled with the net, which a parity-cancelling component makes arbitrary. Independent
    /// of the SCF: a missing or sign-flipped term fails here at O(1e-3–1), while HF-vs-FD on a converged state is
    /// blind below its bound. Two pseudo-O ions off the grid points (ion–ion, overlapping spheres, projector
    /// spheres clipped by the 6.4 Bohr box); a mixed setup's all-electron H (σ = h) beside them exercises
    /// `local_lr_forces`' all-electron branch. The frozen state is synthetic: a density blob per ion, an s-like
    /// and a p-like orbital with occupations 1 and ⅔ (spec §5.7), spin 0.6/0.4.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn force_terms_are_exact_derivatives_of_frozen_energies() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::nonlocal::NonlocalOp;
        use crate::nuclear::{centres, ionic_repulsion, smeared_potential};
        use crate::radial::test_cache::filtered_at;
        let h = 0.2;
        let g = Grid::cube(33, h);
        let t = filtered_at(8, h);
        let setup = IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(t.clone())), (1, Ionic::AllElectron { sigma: h })]));
        let base = vec![Nucleus { pos: [-1.13, 0.07, 0.03], z: 8 }, Nucleus { pos: [1.16, -0.04, 0.05], z: 8 }, Nucleus { pos: [0.11, 1.62, -0.09], z: 1 }];
        let blob = |c: [f64; 3], w: f64| g.sample(|r| (-dist(r, c).powi(2) / w).exp());
        let rho: Vec<f64> = blob(base[0].pos, 1.0).iter().zip(blob(base[1].pos, 1.0)).map(|(a, b)| a + b).collect();
        let up: Vec<f64> = rho.iter().map(|x| 0.6 * x).collect();
        let dn: Vec<f64> = rho.iter().map(|x| 0.4 * x).collect();
        let s_orb = blob(base[0].pos, 1.5);
        let p_orb: Vec<f64> = g.sample(|r| r[0] - base[0].pos[0]).iter().zip(&s_orb).map(|(x, s)| x * s).collect();
        let orbitals: [&[f64]; 2] = [&s_orb, &p_orb];
        let occ = [1.0, 2.0 / 3.0];
        // the frozen energies as functions of the geometry
        let e_lr = |n: &[Nucleus]| g.dot(&rho, &smeared_potential(&g, &centres(n, &setup)));
        let e_sr = |n: &[Nucleus]| n.iter().filter(|ion| ion.z == 8).map(|ion| crate::grid::neumaier(points_within(&g, ion.pos, t.sr_radius()).into_iter()
            .map(|p| rho[p] * t.v_sr(dist(g.point(p), ion.pos)))) * g.dv()).sum::<f64>();
        let e_nl = |n: &[Nucleus]| NonlocalOp::build(&g, n, &setup).energy_weighted(&[s_orb.clone(), p_orb.clone()], &occ);
        let e_pc = |n: &[Nucleus]| g.integrate(&xc_with_core(&up, &dn, Some(&partial_core_field(&g, n, &setup))).energy_density);
        let e_ii = |n: &[Nucleus]| ionic_repulsion(n, &setup).unwrap();
        let terms: [(&str, Vec<[f64; 3]>, &dyn Fn(&[Nucleus]) -> f64); 5] = [
            ("LR", local_lr_forces(&g, &rho, &base, &setup), &e_lr),
            ("SR", local_sr_forces(&g, &rho, &base, &setup), &e_sr),
            ("NL", nonlocal_forces(&g, &orbitals, &occ, &base, &setup), &e_nl),
            ("ρ_pc", partial_core_forces(&g, &up, &dn, &base, &setup), &e_pc),
            ("ion–ion", ion_ion_forces(&base, &setup).unwrap(), &e_ii),
        ];
        let gross = frozen_gross(&g, &rho, &up, &dn, &orbitals, &occ, &base, &setup);
        for (k, (name, f, e)) in terms.into_iter().enumerate() {
            let mut worst = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..base.len() {
                for a in 0..3 {
                    let fd = |d: f64| { let (mut p, mut m) = (base.clone(), base.clone()); p[i].pos[a] += d; m[i].pos[a] -= d; -(e(&p) - e(&m)) / (2.0 * d) };
                    let rich = (4.0 * fd(1e-4) - fd(2e-4)) / 3.0;
                    let (err, gr) = ((f[i][a] - rich).abs(), gross[k][i][a]);
                    assert!(gr >= f[i][a].abs() * (1.0 - 1e-12), "{name}: the gross {gr:.3e} is below the net {:.3e}", f[i][a].abs());
                    if err > worst.0 { worst = (err, f[i][a].abs(), gr); }
                    assert!(err <= 1e-8 + 1e-9 * gr, "{name}: ion {i}, axis {a}: F {:.10e} vs −dE/dR {rich:.10e} (gross {gr:.3e})", f[i][a]);
                }
            }
            eprintln!("frozen-state {name}: worst |F − (−dE/dR)| = {:.2e} (at |F| = {:.3e}, gross {:.3e}: {:.1e} of it; bound {:.2e}); F = {:?}", worst.0, worst.1, worst.2,
                      worst.0 / worst.2.max(f64::MIN_POSITIVE), 1e-8 + 1e-9 * worst.2, f.iter().map(|v| v.map(|x| format!("{x:+.6e}"))).collect::<Vec<_>>());
        }
        // the H is all-electron: no SR, NL or partial-core force on it
        assert_eq!(local_sr_forces(&g, &rho, &base, &setup)[2], [0.0; 3]);
        assert_eq!(nonlocal_forces(&g, &orbitals, &occ, &base, &setup)[2], [0.0; 3]);
        assert_eq!(partial_core_forces(&g, &up, &dn, &base, &setup)[2], [0.0; 3]);
    }

    /// NLCC ruling, Ruling 6: the 3-D partial-core force of a frozen spherical filtered atom, fed the sampled radial
    /// valence spin densities, equals `radial::filter::pc_lattice_force` (T8(e)'s lattice sum) to 1e-10 relative: it
    /// is the same sum, v̄ on ρ_v,σ + ½ρ_pc pointwise. O's Hund atom (spin-polarised, so ½Σ_σ v_xc,σ is exercised) on
    /// its own filtered tables, at R = (h/4)(1, 1, 1) and a low-symmetry offset. The spacings are dyadic, h = 3/16 and
    /// 5/32 (between O's 0.20 and 0.15), so the grid's points origin + i·h are exactly the lattice's i·h and only the
    /// summation order differs (Neumaier: ~1e-16 of Σ|t|). At h = 0.20, the production h_req, the coordinates differ
    /// by ≈ 1 ulp of 4.4 (|δx| ≈ 9e-16); with the integrand's log-derivative κ ≲ s + π/h ≈ 30 Bohr⁻¹ the worst coherent
    /// case is |ΔF| ≤ κ|δx|Σ_p|t_p| ≈ 3e-14·Σ|t|, so every spacing asserts |F₃ − F_lat|∞ ≤ 1e-13·Σ_p|t_p| (Task 7
    /// review M1; Σ|t| = h³Σ_p|v̄ ρ_pc′|, ≥ each component's gross). Measured at 0.20: 4.9e-9 and 6.8e-9 relative to
    /// |F|, with Σ|t|/|F| ≈ 2e6, i.e. 2.5e-15 of the gross: rounding of the inputs, not a different sum — the
    /// 1e-10·|F| form is ill-conditioned there by construction and is asserted only at the dyadic spacings.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn partial_core_force_is_the_radial_lattice_sum() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        use crate::radial::filter::pc_lattice_force;
        use crate::radial::test_cache::{filtered_at, table};
        use crate::radial::{ground_state, solve_kb_atom, transfer_configs, SpinMode};
        let t = table(8);
        let hund_ae = ground_state(8, SpinMode::Polarised).expect("the Hund atom");
        let hund = transfer_configs(t, &hund_ae).into_iter().find(|c| c.name == "Hund").expect("the Hund configuration");
        for h in [0.1875, 0.15625, 0.20] {
            let dyadic = h != 0.20; // 1e-10·|F| only where the coordinates are exact (doc); the gross bound everywhere
            let ft = filtered_at(8, h);
            assert!(ft.pc_radius > 0.0, "O has a partial core");
            let atom = solve_kb_atom(&ft.grid, &ft.input(), &hund.pp, hund.spin).expect("the filtered Hund KB atom");
            let rho_v = |d: f64| [ft.grid.interp(&atom.rho[0], d), ft.grid.interp(&atom.rho[1], d)];
            let rho_pc = |d: f64| (ft.rho_pc_at(d), ft.drho_pc_at(d));
            let setup = IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(ft.clone()))]));
            let n = 2 * ((ft.pc_radius / h).ceil() as usize + 3) + 1;
            let g = Grid::cube(n, h);
            for r in [[0.25 * h, 0.25 * h, 0.25 * h], [0.37 * h, -0.11 * h, 0.23 * h]] {
                let up = g.sample(|p| rho_v(dist(p, r))[0]);
                let dn = g.sample(|p| rho_v(dist(p, r))[1]);
                let f3 = partial_core_forces(&g, &up, &dn, &[Nucleus { pos: r, z: 8 }], &setup)[0];
                let fl = pc_lattice_force(h, ft.pc_radius, r, &rho_v, &rho_pc);
                let scale = (fl[0] * fl[0] + fl[1] * fl[1] + fl[2] * fl[2]).sqrt();
                let rel = (0..3).map(|a| (f3[a] - fl[a]).abs()).fold(0.0, f64::max) / scale;
                // the gross sum h³Σ_p|v̄ ρ_pc′| the odd integrand cancels down to |F|
                let gross = g.dv() * (0..g.len()).map(|p| {
                    let d = dist(g.point(p), r);
                    if d > ft.pc_radius || d == 0.0 { return 0.0; }
                    let ([u, w], (pc, dpc)) = (rho_v(d), rho_pc(d));
                    let (_, vu, vd) = crate::xc::exc_point(u + 0.5 * pc, w + 0.5 * pc);
                    (0.5 * (vu + vd) * dpc).abs()
                }).sum::<f64>();
                eprintln!("O Hund atom, h = {h}, {n}³, R = {r:.4?}: 3-D F_pc ({:+.6e}, {:+.6e}, {:+.6e}) vs lattice ({:+.6e}, {:+.6e}, {:+.6e}); rel {rel:.1e}; Σ|t|/|F| = {:.1e}; |ΔF|∞/Σ|t| = {:.1e}{}",
                          f3[0], f3[1], f3[2], fl[0], fl[1], fl[2], gross / scale, (0..3).map(|a| (f3[a] - fl[a]).abs()).fold(0.0, f64::max) / gross,
                          if dyadic { "" } else { " (grid coordinates one ulp off the lattice's)" });
                assert!(scale > 0.0, "a nonzero lattice force");
                if dyadic { assert!(rel < 1e-10, "h = {h}, R = {r:?}: 3-D {f3:?} vs lattice {fl:?} (rel {rel:.2e})"); }
                // every spacing (the production h_req 0.20 included): the input-rounding bound on the gross (doc)
                let diff = (0..3).map(|a| (f3[a] - fl[a]).abs()).fold(0.0, f64::max);
                assert!(diff <= 1e-13 * gross, "h = {h}, R = {r:?}: |F₃ − F_lat|∞ {diff:.2e} > 1e-13·Σ|t| = {:.2e}", 1e-13 * gross);
            }
        }
    }

    /// Spec §5.2: an all-electron species inside a setup is today's smeared nucleus, so the dispatch's pseudised path on
    /// a setup whose every species is all-electron returns today's forces on the same state (the sums are formed in
    /// another order: 1e-12 relative); and the pseudised path keeps the coincident-nuclei error.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn all_electron_setup_forces_are_todays() {
        use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
        let g = Grid::cube(25, 0.3);
        let mut s = ScfSettings::for_grid(&g);
        s.tol = 1e-5; s.eig_tol = 1e-5;
        let nuc = vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.05], z: 1 }];
        let (mut in_setup, _) = scf(&g, &System::new(nuc.clone(), 0, None), &s, None).unwrap();
        let today = forces(&g, &in_setup, &nuc, s.sigma).unwrap();
        in_setup.ionic = Some(std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::None, g.h, s.sigma,
            std::collections::BTreeMap::from([(3, Ionic::AllElectron { sigma: s.sigma }), (1, Ionic::AllElectron { sigma: s.sigma })]))));
        let via = forces(&g, &in_setup, &nuc, s.sigma).unwrap();
        let scale = today.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        for i in 0..2 { for a in 0..3 {
            assert!((today[i][a] - via[i][a]).abs() <= 1e-12 * scale, "nucleus {i}, axis {a}: {} vs {} (scale {scale:.3e})", today[i][a], via[i][a]);
        }}
        let coincident = [nuc[0], nuc[1], nuc[1]];
        assert!(matches!(forces(&g, &in_setup, &coincident, s.sigma), Err(QmError::CoincidentNuclei(1, 2))));
    }

    /// Diagnostic, not a test (ignored by default): prints the H₂ forces at R = 1.0 … 2.5
    /// (locating the force zero at σ = h) with SCF step counts and wall times, and the Hellmann-Feynman vs central-difference
    /// comparison at δ = 0.01 and 0.02 so Richardson extrapolation separates the O(δ²) FD
    /// truncation from any force error. Run with
    /// `cargo test -p qm forces --release -- --ignored --nocapture report_numbers`.
    #[test]
    #[ignore]
    fn report_numbers() {
        use std::time::Instant;
        let g = Grid::cube(49, 0.25);
        let mut s = ScfSettings::for_grid(&g);
        s.tol = 1e-9; s.eig_tol = 1e-8;
        let solve = |nuclei: Vec<Nucleus>| {
            let sys = System::new(nuclei, 0, None);
            let t0 = Instant::now();
            let (state, stats) = scf(&g, &sys, &s, None).unwrap();
            let f = forces(&g, &state, &sys.nuclei, s.sigma).unwrap();
            (stats, f, t0.elapsed().as_secs_f64())
        };
        for r in [1.0, 1.4, 1.6, 1.8, 2.0, 2.5] {
            let (st, f, t) = solve(vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }]);
            eprintln!("H2 R {r:.2}: E {:+.9}  F0 {:?}  F1 {:?}  sum {:+.2e}  iters {}  dRho {:.2e}  face {:.2e}  {t:.2}s",
                st.energy.total, f[0], f[1], f[0][0] + f[1][0], st.iterations, st.delta_rho, st.boundary_density);
        }
        let x1 = -0.7;
        let (st0, f0, _) = solve(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [x1, 0.0, 0.0], z: 1 }]);
        let hf = f0[0][0];
        let mut fd = [0.0; 2];
        for (k, d) in [0.01, 0.02].into_iter().enumerate() {
            let t0 = Instant::now();
            let (sp, _, _) = solve(vec![Nucleus { pos: [0.7 + d, 0.0, 0.0], z: 1 }, Nucleus { pos: [x1, 0.0, 0.0], z: 1 }]);
            let (sm, _, _) = solve(vec![Nucleus { pos: [0.7 - d, 0.0, 0.0], z: 1 }, Nucleus { pos: [x1, 0.0, 0.0], z: 1 }]);
            fd[k] = -(sp.energy.total - sm.energy.total) / (2.0 * d);
            eprintln!("delta {d:.2}: E+ {:+.10} ({} it)  E- {:+.10} ({} it)  FD {:+.8}  HF {hf:+.8}  HF-FD {:+.3e}  {:.2}s",
                sp.energy.total, sp.iterations, sm.energy.total, sm.iterations, fd[k], hf - fd[k], t0.elapsed().as_secs_f64());
        }
        // Richardson: FD(δ) = F + c δ² ⇒ F = (4 FD(0.01) − FD(0.02))/3
        let rich = (4.0 * fd[0] - fd[1]) / 3.0;
        eprintln!("E(1.4) {:+.10}  Richardson FD {rich:+.8}  HF-Richardson {:+.3e}  (HF-FD ratio 0.02/0.01 = {:.2})",
            st0.energy.total, hf - rich, (hf - fd[1]) / (hf - fd[0]));
    }
}
