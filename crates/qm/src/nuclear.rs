//! Nuclei as Gaussian charges of width σ: charge deposition, analytic external potential,
//! point nuclear repulsion. Atomic units: lengths in Bohr, charges in e, energies in Hartree.
//!
//! # Why smear
//! A point nucleus has a 1/r singularity the grid cannot represent (and the 8th-order
//! Laplacian of `crate::stencil` would turn its cusp into O(1) noise). Each nucleus is
//! therefore represented, *for the electrons only*, by a Gaussian charge of width σ. The
//! external potential is then smooth and bounded, at the cost of an O(σ²) energy error that
//! validation (Task 9) removes by extrapolating σ → 0. Nuclei never see each other smeared:
//! the nucleus–nucleus term is the exact point-charge Coulomb sum.
//!
//! # Charge density and normalisation
//!   ρ_nuc(r) = Σ_I Z_I g_σ(r − R_I),   g_σ(r) = (2πσ²)^{−3/2} exp(−|r|²/2σ²).
//! g_σ is a product of three 1D normal densities, each with ∫ exp(−x²/2σ²) dx = σ√(2π), so
//! ∫ g_σ d³r = (2πσ²)^{3/2}/(2πσ²)^{3/2} = 1 and ∫ ρ_nuc = Σ_I Z_I. Per-axis variance is σ²,
//! so ⟨|r|²⟩ = 3σ².
//!
//! # Potential of one Gaussian charge (sign convention of `crate::poisson`)
//! φ solves ∇²φ = −4πρ, φ(r) = ∫ ρ(r′)/|r − r′| d³r′, so a positive charge gives φ > 0.
//! For a spherically symmetric ρ(r), Gauss's law gives φ(d) = Q(d)/d + ∫_d^∞ 4πr ρ(r) dr,
//! where Q(d) is the charge inside radius d. For Z g_σ:
//!   Q(d)            = Z [erf(d/σ√2) − √(2/π) (d/σ) exp(−d²/2σ²)],
//!   ∫_d^∞ 4πr ρ dr  = Z (2πσ²)^{−3/2} 4πσ² exp(−d²/2σ²) = Z √(2/π) exp(−d²/2σ²)/σ,
//! and the two exponential terms cancel exactly:
//!   φ_I(d) = Z erf(d/(σ√2))/d.
//! Check d → ∞: erf → 1, φ → Z/d (point charge). Check d → 0: erf(x) = (2/√π)(x − x³/3 + …),
//! so erf(d/σ√2)/d → (2/√π)/(σ√2) = √(2/π)/σ, finite: φ_I(0) = Z√(2/π)/σ.
//!
//! # The electron's potential energy: sign
//! An electron has charge −1, so its potential energy in φ is V_nuc = −φ:
//!   V_nuc(r) = −Σ_I Z_I erf(|r − R_I|/(σ√2))/|r − R_I|,   V_nuc(R_I) → −Z_I √(2/π)/σ.
//! Attractive (negative) everywhere, bounded, and −Z/d far from every nucleus.
//! Poisson cross-check: solving `PoissonSolver` for the deposited ρ_nuc must return φ with
//! V_nuc + φ = 0 to discretisation accuracy (the test
//! `analytic_potential_matches_poisson_of_deposit` pins this; the GPU path in M2 obtains
//! V_nuc that way).
//!
//! # Leading-order smearing error in the energy
//! The electron–nucleus energy with the smeared nucleus is E^σ = ∫ ρ_e V_nuc^σ, with the
//! point nucleus E⁰ = −Z ∫ ρ_e/d. Their difference,
//!   ΔE = E^σ − E⁰ = −Z ∫ ρ_e(r) [erf(d/σ√2) − 1]/d d³r = Z ∫ ρ_e(r) erfc(d/σ√2)/d d³r,
//! has an integrand confined to d ≲ few σ around R. For small σ expand ρ_e about R:
//! ρ_e(r) = ρ_e(R) + (r − R)·∇ρ_e + …; the linear term vanishes by symmetry, so
//!   ΔE ≈ Z ρ_e(R) ∫ erfc(d/σ√2)/d · 4πd² dd = 4π Z ρ_e(R) ∫_0^∞ d erfc(d/a) dd,  a = σ√2.
//! With ∫_0^∞ u erfc(u) du = (1/√π)∫_0^∞ u² e^{−u²} du = 1/4 (integrate by parts; the
//! boundary term vanishes), ∫_0^∞ d erfc(d/a) dd = a²/4 = σ²/2, hence
//!   ΔE ≈ 2π Z ρ_e(R) σ²  (> 0: smearing *raises* the energy).
//! Independent check: E^σ = −Z ⟨φ_e⟩_{g_σ} (the smeared electron potential at R), and
//! ⟨φ_e⟩ = φ_e(R) + ½⟨(r−R)_i(r−R)_j⟩ ∂_i∂_j φ_e = φ_e(R) + (σ²/2)∇²φ_e(R)
//! = φ_e(R) − 2πσ² ρ_e(R), giving the same ΔE = 2π Z ρ_e(R) σ².
//! By the variational principle this first-order estimate is also the leading change of the
//! self-consistent total energy. The next term is O(σ³), not O(σ⁴): the point-nucleus density
//! has a cusp, ρ_e(r) ≈ ρ_e(R)(1 − 2Z d), and ∫ d² erfc(d/a) dd = a³/(3√π) ≠ 0. For hydrogen
//! (ρ_e(0) = 1/π) the exact ΔE(σ) = 4∫ d e^{−2d} erfc(d/σ√2) dd gives ΔE/σ² = 1.34, 1.63,
//! 1.80, 1.96 at σ = 0.2, 0.1, 0.05, 0.01 → 2 = 2π/π, confirming the coefficient and the
//! size of the σ³ correction. The σ → 0 extrapolation therefore fits the four-point model
//! E = a + bσ² + cσ³ + dσ⁴ (`scf::extrapolate_sigma`, whose module doc covers its bias).
//!
//! # Resolution of σ by the grid (measured in the tests)
//! - Deposited charge: uniform-grid quadrature of a Gaussian is exact up to aliasing
//!   (Poisson summation): per axis, h Σ_n g(nh − x₀) = ∫g · [1 + 2Σ_{k≥1} cos(2πk x₀/h)
//!   exp(−2π²k²σ²/h²)]. At σ = h the k = 1 term is 2e^{−2π²} = 5.35e-9 per axis — the total
//!   charge is reproduced to ~1e-8 relative, not to rounding; at σ = 2h it is 2e^{−8π²} ≈ 1e-34.
//! - Poisson path: the disagreement between `nuclear_potential` and the Poisson solve of
//!   `deposit_charge` is the 8th-order stencil's truncation error on a feature of width σ.
//!   It peaks at the nucleus, is linear in Z, and falls as (h/σ)^8: at h = 0.25, Z = 2 the
//!   max |V_nuc + φ| is 4.07e-3, 4.50e-4, 7.67e-5, 1.74e-5, 1.49e-6 for σ = 0.3, 0.4, 0.5,
//!   0.6, 0.8 (log-slope −8.1). So the Poisson route to V_nuc needs σ ≳ 2h for 1e-4 accuracy;
//!   the analytic erf route has no such floor (it is exact pointwise). M2's GPU path, which
//!   takes V_nuc from the Poisson solver, must respect this.
//!
//! # Precision
//! `deposit_charge` and `nuclear_potential` are pointwise (a short sum over nuclei per grid
//! point, no accumulation). `nuclear_repulsion` sums O(N²) positive terms in f64; callers
//! integrate grid quantities through `Grid::integrate`/`Grid::dot` (compensated).

use crate::error::QmError;
use crate::grid::Grid;
use std::f64::consts::PI;

/// A nucleus: position in Bohr and atomic number. Mass is not needed here (the integrator
/// owns it); nothing else per element is allowed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nucleus {
    pub pos: [f64; 3],
    pub z: u8,
}

/// Two nuclei closer than this (Bohr) are reported as coincident.
const COINCIDENCE: f64 = 1e-8;
/// Below this distance (Bohr) the erf/d form is replaced by its exact d → 0 limit.
const ON_SITE: f64 = 1e-10;

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Positive charge density ρ_nuc(r) = Σ_I Z_I (2πσ²)^{−3/2} exp(−|r − R_I|²/2σ²) sampled on
/// the grid. Integrates to Σ Z_I up to the grid's aliasing floor (see module doc).
pub fn deposit_charge(grid: &Grid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64> {
    debug_assert!(sigma > 0.0, "sigma must be positive, got {sigma}");
    let norm = 1.0 / (2.0 * PI * sigma * sigma).powf(1.5);
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    grid.sample(|r| {
        nuclei
            .iter()
            .map(|n| n.z as f64 * norm * (-dist2(r, n.pos) * inv_2s2).exp())
            .sum()
    })
}

/// Potential energy of an electron in the field of the smeared nuclei:
/// V_nuc(r) = −Σ_I Z_I erf(|r − R_I|/(σ√2))/|r − R_I|, with the exact limit −Z_I √(2/π)/σ at
/// r = R_I. Equals −φ where φ is the Poisson potential of `deposit_charge`.
pub fn nuclear_potential(grid: &Grid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64> {
    debug_assert!(sigma > 0.0, "sigma must be positive, got {sigma}");
    let s2 = sigma * 2f64.sqrt();
    let on_site = (2.0 / PI).sqrt() / sigma;
    grid.sample(|r| {
        nuclei
            .iter()
            .map(|n| {
                let d = dist2(r, n.pos).sqrt();
                let z = n.z as f64;
                if d < ON_SITE { -z * on_site } else { -z * libm::erf(d / s2) / d }
            })
            .sum()
    })
}

/// Σ_{I<J} Z_I Z_J / |R_I − R_J| for point nuclei (nuclei never see each other smeared).
/// Errors with the first coincident pair (indices in input order) if any two nuclei are
/// closer than 1e-8 Bohr.
pub fn nuclear_repulsion(nuclei: &[Nucleus]) -> Result<f64, QmError> {
    let mut e = 0.0;
    for i in 0..nuclei.len() {
        for j in (i + 1)..nuclei.len() {
            let d = dist2(nuclei[i].pos, nuclei[j].pos).sqrt();
            if d < COINCIDENCE {
                return Err(QmError::CoincidentNuclei(i, j));
            }
            e += nuclei[i].z as f64 * nuclei[j].z as f64 / d;
        }
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::poisson::PoissonSolver;
    use approx::assert_relative_eq;

    /// Max |V_nuc + φ| over the grid and the grid point where it occurs.
    fn poisson_mismatch(g: &Grid, solver: &mut PoissonSolver, nuc: &[Nucleus], sigma: f64) -> (f64, [f64; 3]) {
        let rho = deposit_charge(g, nuc, sigma);
        let mut phi = vec![0.0; g.len()];
        let stats = solver.solve(g, &rho, &mut phi, 1e-10, 3000);
        assert!(stats.residual < 1e-10, "Poisson did not converge: {stats:?}");
        let v = nuclear_potential(g, nuc, sigma);
        let (mut err, mut at) = (0.0, 0);
        for (i, (a, b)) in v.iter().zip(&phi).enumerate() {
            if (a + b).abs() > err { err = (a + b).abs(); at = i; }
        }
        (err, g.point(at))
    }

    #[test]
    fn deposited_charge_integrates_to_total_z() {
        // Uniform-grid quadrature of a Gaussian is exact up to aliasing (module doc). With
        // y = z = 0 on grid points and x₀/h = ±2.8 (cos(2π·2.8) = cos(1.6π) = 0.309), each
        // nucleus's charge comes out high by (1 + 0.618ε)(1 + 2ε)² − 1 with ε = e^{−2π²} at
        // σ = h: 1.236e-8 relative, 4.942e-8 absolute for Z_tot = 4 (measured 4.9418e-8).
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 3 }];
        let rho = deposit_charge(&g, &nuc, 0.25);
        let eps = (-2.0 * PI * PI).exp();
        let x0_over_h = 0.7 / 0.25;
        let predicted = 4.0 * ((1.0 + 2.0 * eps * (2.0 * PI * x0_over_h).cos()) * (1.0 + 2.0 * eps).powi(2) - 1.0);
        assert_relative_eq!(g.integrate(&rho) - 4.0, predicted, max_relative = 1e-2);
        // At σ = 2h the aliasing term is 2e^{−8π²} ≈ 1e-34: rounding only.
        let rho = deposit_charge(&g, &nuc, 0.5);
        assert_relative_eq!(g.integrate(&rho), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn analytic_potential_matches_poisson_of_deposit() {
        // σ = 0.6 = 2.4h: the stencil resolves the Gaussian and the remaining mismatch is the
        // discretisation floor measured in `poisson_mismatch_is_stencil_truncation` (1.74e-5
        // here; Task 5 measured 3.78e-5 for Z = 1, σ = 0.5). Two off-centre nuclei exercise the
        // multipole boundary; any sign or normalisation error would show up as O(1).
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.7, 0.3, -0.2], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 2 }];
        let (err, _) = poisson_mismatch(&g, &mut PoissonSolver::new(&g), &nuc, 0.6);
        assert!(err < 1e-4, "max |V_nuc + φ| = {err:.3e}");
    }

    #[test]
    fn poisson_mismatch_is_stencil_truncation() {
        // Measured at h = 0.25: 4.07e-3, 4.50e-4, 7.67e-5, 1.74e-5, 1.49e-6 for
        // σ = 0.3, 0.4, 0.5, 0.6, 0.8 (log-slope −8.1, the stencil order), always at the grid
        // point nearest the Z = 2 nucleus. Doubling σ/h from 1.2 to 2.4 gives a ratio of 234
        // (8th order asymptotically 256; Task 5 saw 106 for the same resolution doubling).
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.7, 0.3, -0.2], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 2 }];
        let mut solver = PoissonSolver::new(&g);
        let (coarse, at_coarse) = poisson_mismatch(&g, &mut solver, &nuc, 0.3);
        let (fine, at_fine) = poisson_mismatch(&g, &mut solver, &nuc, 0.6);
        for at in [at_coarse, at_fine] {
            let d = dist2(at, nuc[1].pos).sqrt();
            assert!(d < g.h, "max mismatch at {at:?}, {d:.3} Bohr from the Z = 2 nucleus");
        }
        assert!(coarse / fine > 100.0, "σ = 0.3: {coarse:.3e}, σ = 0.6: {fine:.3e}, ratio {}", coarse / fine);
    }

    #[test]
    fn potential_is_coulombic_far_away() {
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.0; 3], z: 6 }];
        let v = nuclear_potential(&g, &nuc, 0.2);
        let far = g.idx(44, 24, 24); // x = 5 Bohr
        assert_relative_eq!(v[far], -6.0 / 5.0, epsilon = 1e-12);
        let on = g.idx(24, 24, 24);
        assert_relative_eq!(v[on], -6.0 * (2.0 / std::f64::consts::PI).sqrt() / 0.2, epsilon = 1e-12);
    }

    #[test]
    fn repulsion_h2() {
        let nuc = [Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }];
        assert_relative_eq!(nuclear_repulsion(&nuc).unwrap(), 1.0 / 1.4, epsilon = 1e-15);
    }

    #[test]
    fn coincident_nuclei_is_an_error() {
        let nuc = [Nucleus { pos: [0.0; 3], z: 1 }, Nucleus { pos: [0.0; 3], z: 1 }];
        assert!(matches!(nuclear_repulsion(&nuc), Err(QmError::CoincidentNuclei(0, 1))));
    }
}
