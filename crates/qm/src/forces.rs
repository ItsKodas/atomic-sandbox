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
//! Task 9 measured the egg-box *energy* ripple for H at σ = h = 0.25: E(h/2) − E(0) = +3.3 µHa.
//! E_h is periodic in a nuclear coordinate with period h and even about x = 0 and x = h/2, so
//! those are its extrema and the leading Fourier term is E ≈ E₀ − A cos(2πx/h) with
//! 2A = 3.3 µHa, A ≈ 1.7 µHa. The spurious egg-box force is its derivative,
//!   |F_egg| ≤ A · 2π/h ≈ 1.7e-6 × 25 ≈ 4e-5 Ha/Bohr per nucleus (H-like density, σ = h).
//! That is the force error against the *continuum*; HF and a finite difference of E_h both
//! contain it consistently, so it does not enter HF − FD except through the FD truncation of
//! the ripple, δ² A (2π/h)³/6 ≈ 4e-7 at δ = 0.01. The HF − FD discrepancy is instead bounded
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
//! grid plane; for a general geometry it is bounded by the egg-box force, ~4e-5 Ha/Bohr per
//! nucleus at σ = h = 0.25 (above). Dynamics (M3/M4) must not assert total-momentum
//! conservation to better than that.
//!
//! # Precision
//! The force integrands are formed pointwise and reduced through `Grid::integrate` (Neumaier);
//! the nuclear sum is a short f64 loop.

use crate::error::QmError;
use crate::grid::Grid;
use crate::nuclear::Nucleus;
use crate::scf::KsState;
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
/// electronic part ∫ρ ∇_r V_I plus the point nuclear–nuclear repulsion (module doc).
/// `grid` and `sigma` must be the grid and smearing width the state was converged with
/// (kept as explicit parameters; panics if they differ from `state.grid`/`state.sigma`).
/// Errors on coincident nuclei (the repulsion is undefined), with the lower index first.
pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError> {
    assert_eq!(&state.grid, grid, "KsState was converged on a different grid");
    assert!((state.sigma - sigma).abs() < 1e-15, "KsState was converged with sigma = {}, forces called with {}", state.sigma, sigma);
    debug_assert!(sigma > 0.0, "sigma must be positive, got {sigma}");
    // the integrand below is the all-electron one (charge Z at σ): applied to a pseudised state it would return a
    // plausible wrong number, so it refuses until M7 Task 7 dispatches on the setup
    assert!(state.ionic.is_none(), "forces on an ionic-setup state arrive in M7 Task 7 (the V_SR, V_NL and ρ_pc terms)");
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
