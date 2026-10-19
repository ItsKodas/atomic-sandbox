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
//!
//! # Ionic species (M7; spec §5.1, §5.2; Spec questions 2 and 6 as the plan review ruled)
//! Under an [`IonicSetup`] every nucleus I is a *centre* (q_I, σ_I): q = Z, σ = σ_AE for an all-electron
//! species (exactly the smeared nucleus above), q = Z_v, σ = σ_LR = `LEDGER.sigma_lr_over_h`·h for a pseudised
//! one. The electron's local potential energy is
//!   V_loc(r) = Σ_I [−q_I erf(d_I/(σ_I√2))/d_I]  +  Σ_{I pseudised} V_SR,I(d_I),   d_I = |r − R_I|,
//! with the d → 0 limit −q√(2/π)/σ of each erf term, as `nuclear_potential`. For a pseudised species the split
//! V_loc^I = V_LR + V_SR is *exact* (filter.rs "The local split"): V_LR = −Z_v erf(d/a)/d (a = σ_LR√2) is the
//! analytic erf, exact pointwise, never tabulated (Spec question 6); V_SR = V_loc^I + Z_v erf(d/a)/d is the ONE
//! filtered table (Spec question 2, plan review M2), smooth and finite at 0, sampled once per geometry at the
//! points within its table radius γ_r R_SR + 8h (beyond it the table is exactly 0: the band completion's outer
//! step). The partial core ρ_pc,I (pseudised species with a core only; the no-core branch adds nothing) is
//! sampled the same way within its table radius. Both read only the uniform Δ = h/256 tables through
//! `FilteredTable::{v_sr, rho_pc_at}` (spec §5.1 "Sampling", Task 5 physics review M3), never the δ = 0.008 log
//! grid: its interpolation alone exceeds T8(d)'s whole budget.
//! Signs: V_LR < 0 (attractive); V_SR has the table's sign; ρ_pc ≥ 0 up to the filter's Gibbs lobes (allowed:
//! T8(c) certifies the LSDA domain n_σ = ρ_v,σ + ½ρ_pc ≥ 0 at every certified h).
//! Ion–ion: E_ii = Σ_{I<J} q_I q_J/R_IJ, point charges (q = Z_v for pseudised species, Z for all-electron
//! ones), with the same coincidence error as `nuclear_repulsion`.
//! The cold-start blob (`deposit_centres`) is `deposit_charge` with per-centre charges q_I: the right valence
//! charge in roughly the right places.

use crate::error::QmError;
use crate::grid::Grid;
use crate::ionic::{Ionic, IonicSetup};
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

/// One smeared centre of the local potential (module doc, "Ionic species"): position (Bohr), charge q (e) and
/// width σ (Bohr).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Centre { pub pos: [f64; 3], pub charge: f64, pub sigma: f64 }

/// −Σ q erf(d/(σ√2))/d with the d → 0 limit −q√(2/π)/σ: `nuclear_potential` with per-centre q and σ.
pub fn smeared_potential(grid: &Grid, centres: &[Centre]) -> Vec<f64> {
    grid.sample(|r| centres.iter().map(|c| {
        let d = dist2(r, c.pos).sqrt();
        if d < ON_SITE { -c.charge * (2.0 / PI).sqrt() / c.sigma } else { -c.charge * libm::erf(d / (c.sigma * 2f64.sqrt())) / d }
    }).sum())
}

/// Σ q (2πσ²)^{−3/2} e^{−d²/2σ²}: `deposit_charge` with per-centre q and σ (the cold guess's blobs).
pub fn deposit_centres(grid: &Grid, centres: &[Centre]) -> Vec<f64> {
    grid.sample(|r| centres.iter().map(|c| c.charge / (2.0 * PI * c.sigma * c.sigma).powf(1.5) * (-dist2(r, c.pos) / (2.0 * c.sigma * c.sigma)).exp()).sum())
}

/// The centres of `nuclei` under `setup`: (Z, σ_AE) for all-electron species, (Z_v, σ_LR) for pseudised ones.
pub fn centres(nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<Centre> {
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { sigma } => Centre { pos: n.pos, charge: n.z as f64, sigma: *sigma },
        Ionic::Pseudo(t) => Centre { pos: n.pos, charge: t.z_v, sigma: t.sigma_lr },
    }).collect()
}

/// The grid points within `radius` of `centre` (|r_p − centre| ≤ radius), ascending: a bounding box of indices
/// clipped to the box, then the distance test. Points outside the box do not exist (ψ = 0 there under the
/// Dirichlet wall), so a sphere cut by a face is the exact restriction of the operator it carries.
pub fn points_within(grid: &Grid, centre: [f64; 3], radius: f64) -> Vec<usize> {
    let mut range = [(0usize, 0usize); 3];
    for (a, rg) in range.iter_mut().enumerate() {
        let lo = ((centre[a] - radius - grid.origin[a]) / grid.h).floor().max(0.0);
        let hi = ((centre[a] + radius - grid.origin[a]) / grid.h).ceil().min((grid.n[a] - 1) as f64);
        if !(hi >= lo) { return Vec::new(); } // the sphere misses the box on this axis (or a NaN coordinate)
        *rg = (lo as usize, hi as usize);
    }
    let mut out = Vec::new();
    for k in range[2].0..=range[2].1 { for j in range[1].0..=range[1].1 { for i in range[0].0..=range[0].1 {
        let p = grid.idx(i, j, k);
        if dist2(grid.point(p), centre) <= radius * radius { out.push(p); }
    }}}
    out
}

/// V_loc = Σ_I V_LR,I + Σ_{pseudised} V_SR,I (module doc): the smeared erf of every centre plus the one V_SR
/// table of each pseudised ion, sampled within its table radius from the uniform table.
pub fn ionic_local_potential(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<f64> {
    let mut v = smeared_potential(grid, &centres(nuclei, setup));
    for n in nuclei {
        if let Ionic::Pseudo(t) = setup.ionic(n.z) {
            for p in points_within(grid, n.pos, t.sr_radius()) { v[p] += t.v_sr(dist2(grid.point(p), n.pos).sqrt()); }
        }
    }
    v
}

/// Σ_I ρ_pc,I over the pseudised species that carry a partial core (module doc); identically 0 when none does
/// (the no-core branch: H and He under `All`, or every species all-electron).
pub fn partial_core_field(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<f64> {
    let mut rho = vec![0.0; grid.len()];
    for n in nuclei {
        if let Ionic::Pseudo(t) = setup.ionic(n.z) {
            if t.pc_radius <= 0.0 { continue; } // the no-core branch: no partial core
            for p in points_within(grid, n.pos, t.pc_radius) { rho[p] += t.rho_pc_at(dist2(grid.point(p), n.pos).sqrt()); }
        }
    }
    rho
}

/// Whether any nucleus is a pseudised species with a partial core (otherwise ρ_pc ≡ 0: the no-core branch).
pub fn has_partial_core(nuclei: &[Nucleus], setup: &IonicSetup) -> bool {
    nuclei.iter().any(|n| matches!(setup.ionic(n.z), Ionic::Pseudo(t) if t.pc_radius > 0.0))
}

/// Σ_{I<J} q_I q_J / R_IJ with valence charges for pseudised species (module doc); coincident nuclei are the
/// same error as `nuclear_repulsion`.
pub fn ionic_repulsion(nuclei: &[Nucleus], setup: &IonicSetup) -> Result<f64, QmError> {
    let mut e = 0.0;
    for i in 0..nuclei.len() {
        for j in (i + 1)..nuclei.len() {
            let d = dist2(nuclei[i].pos, nuclei[j].pos).sqrt();
            if d < COINCIDENCE { return Err(QmError::CoincidentNuclei(i, j)); }
            e += setup.valence_charge(nuclei[i].z) * setup.valence_charge(nuclei[j].z) / d;
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

    // ---- M7 Task 6: ionic species ----

    /// The sphere lists are exactly the brute-force ball: inside, cut by faces, centred outside the box, and
    /// missing the box entirely.
    #[test]
    fn points_within_is_the_brute_force_ball() {
        let g = Grid { n: [13, 9, 11], h: 0.3, origin: [-1.7, 0.2, -1.1] };
        for (c, r) in [([0.1, 1.4, 0.3], 0.95), ([-1.6, 0.25, 1.9], 1.3), ([-2.5, 1.0, 0.0], 1.0), ([2.0, 3.0, -2.0], 2.2),
                       ([9.0, 9.0, 9.0], 1.0), ([0.1, 1.4, 0.3], 0.0), ([-1.7, 0.2, -1.1], 0.0)] {
            let brute: Vec<usize> = (0..g.len()).filter(|&p| dist2(g.point(p), c) <= r * r).collect();
            assert_eq!(points_within(&g, c, r), brute, "centre {c:?}, radius {r}");
        }
    }

    /// An all-electron centre (Z, σ) is `nuclear_potential` / `deposit_charge` to rounding, and the deposited
    /// centres carry their charges.
    #[test]
    fn all_electron_centres_are_todays_nucleus() {
        let g = Grid::cube(25, 0.25);
        let nuc = [Nucleus { pos: [0.3, -0.1, 0.0], z: 3 }, Nucleus { pos: [-1.0, 0.0, 0.25], z: 1 }];
        let c: Vec<Centre> = nuc.iter().map(|n| Centre { pos: n.pos, charge: n.z as f64, sigma: 0.25 }).collect();
        for (a, b) in smeared_potential(&g, &c).iter().zip(nuclear_potential(&g, &nuc, 0.25)) { assert!((a - b).abs() <= 1e-14 * b.abs(), "{a} vs {b}"); }
        for (a, b) in deposit_centres(&g, &c).iter().zip(deposit_charge(&g, &nuc, 0.25)) { assert!((a - b).abs() <= 1e-14 * b.abs().max(1e-300), "{a} vs {b}"); }
        // σ = 2h: aliasing ≈ 2e^{−8π²} (module doc); the 12 Bohr box holds the blobs to e^{−(5.7)²/0.5} ≈ 1e-28
        let g = Grid::cube(49, 0.25);
        let half: Vec<Centre> = c.iter().map(|x| Centre { charge: 0.5 * x.charge, sigma: 0.5, ..*x }).collect();
        assert_relative_eq!(g.integrate(&deposit_centres(&g, &half)), 2.0, epsilon = 1e-12);
    }

    /// The exact split on the grid (spec §5.1): with the identity filter (no band limit, the uniform table
    /// interpolating the log grid), V_LR + V_SR sampled on the grid is the radial V_loc at every point, and
    /// beyond the V_SR table it is the analytic erf exactly (−Z_v/d far out). The residual is the two
    /// interpolations' disagreement, bounded from the table itself:
    /// - The log-grid interpolant P (4-point Lagrange, spacing s = δ·d) is cubic on each interval; at a node it
    ///   is an interior node of both adjacent stencils, where |P′ − V′| ≤ |V⁗|·2s³/24, so P′ jumps there by
    ///   κ ≤ s³|V⁗|/6. Within three intervals of the local channel's TM junction r_c, V is only C² (its third
    ///   derivative jumps by J3): P′ then errs by at most the derivative of (J3/6)(d − r_c)³₊ across a stencil,
    ///   κ ≤ (J3/2)(3s)².
    /// - The uniform 4-point Lagrange U (Δ = h/256) is exact on each cubic piece of P. A stencil that straddles
    ///   one kink κ errs by ≤ (1 + Λ)·E*, where E* = κ·(1.5Δ)/4 is the best linear approximation's error of a
    ///   kink over the stencil's width 3Δ and Λ ≤ 2.2 the 4-point Lebesgue constant: |U − P| ≤ 1.2·Δ·κ.
    /// V⁗ is the 5-point divided difference on the log grid, J3 the difference of the one-sided 4-point third
    /// divided differences at r_c. Measured (O, h = 0.2): 1.0e-7 Ha away from r_c (|V⁗| reaches ~10³ in the TM
    /// polynomial region), 1.1e-6 within three intervals of r_c,p (J3 ≈ 214 Ha/Bohr³). The filtered tables the
    /// solver uses are band-limited and filled exactly, so neither error applies to them; what this test pins is
    /// the split itself (Z_v, σ_LR and the sign of V_LR agree with the table's), whose errors would be O(0.1 Ha).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn local_potential_is_the_exact_split() {
        use crate::radial::filter::FilteredTable;
        use crate::radial::test_cache::table;
        let t = table(8);
        let h = 0.2;
        let ft = std::sync::Arc::new(FilteredTable::unfiltered(t, h));
        let setup = IonicSetup::from_tables(crate::ionic::IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(ft.clone()))]));
        let g = Grid::cube(61, h);
        let pos = [0.031, -0.047, 0.012];
        let v = ionic_local_potential(&g, &[Nucleus { pos, z: 8 }], &setup);
        // the bound from the table (doc above)
        assert_eq!(t.record.local, "p-local", "O is p-local at this recipe version (pinned in the generation records)");
        let (lg, f) = (&t.grid, &t.kb.v_loc);
        let rc = t.record.r_c.iter().find(|x| x.0 == 1).expect("the local channel's r_c").1;
        let ic = lg.nearest(rc);
        let divided = |idx: &[usize]| -> f64 { // n! · f[x_0, …, x_n]
            let mut c: Vec<f64> = idx.iter().map(|&k| f[k]).collect();
            for order in 1..idx.len() { for j in (order..idx.len()).rev() { c[j] = (c[j] - c[j - 1]) / (lg.r[idx[j]] - lg.r[idx[j - order]]); } }
            c[idx.len() - 1] * (1..idx.len()).product::<usize>() as f64
        };
        let j3 = divided(&[ic + 2, ic + 4, ic + 6, ic + 8]) - divided(&[ic - 8, ic - 6, ic - 4, ic - 2]);
        let kappa_junction = 4.5 * j3.abs() * (lg.delta * rc).powi(2);
        let k_max = (0..lg.len()).take_while(|&k| lg.r[k] <= ft.sr_radius() + 0.1).last().unwrap();
        let kappa_smooth = (3..k_max).filter(|&k| k + 6 < ic || k > ic + 6)
            .map(|k| (lg.delta * lg.r[k]).powi(3) * divided(&[k - 2, k - 1, k, k + 1, k + 2]).abs() / 6.0).fold(0.0, f64::max);
        let (bound_smooth, bound_junction) = (1.2 * ft.du * kappa_smooth, 1.2 * ft.du * kappa_junction);
        let (mut smooth, mut junction, mut far) = (0.0f64, 0.0f64, 0.0f64);
        for (p, &vp) in v.iter().enumerate() {
            let d = dist2(g.point(p), pos).sqrt();
            if d <= ft.sr_radius() {
                let e = (vp - lg.interp(f, d)).abs();
                if (d - rc).abs() <= 3.0 * lg.delta * rc { junction = junction.max(e) } else { smooth = smooth.max(e) }
            } else {
                let lr = -6.0 * libm::erf(d / (ft.sigma_lr * 2f64.sqrt())) / d;
                assert!((vp - lr).abs() <= 1e-15 * lr.abs(), "beyond the V_SR table, d = {d}: {vp} vs {lr}");
                if d > 6.0 { far = far.max((vp + 6.0 / d).abs()); }
            }
        }
        eprintln!("O, identity filter, h = {h}: max |V_grid − V_loc(radial)| = {smooth:.2e} Ha away from r_c,p = {rc:.4} (bound {bound_smooth:.2e}), {junction:.2e} within three log intervals of it (J3 = {j3:.1} Ha/Bohr³, bound {bound_junction:.2e}); R_SR = {:.3}; beyond 6 Bohr max |V + Z_v/d| = {far:.2e}", ft.sr_radius());
        assert!(smooth <= bound_smooth, "{smooth:.3e} vs {bound_smooth:.3e}");
        assert!(junction <= bound_junction, "{junction:.3e} vs {bound_junction:.3e}");
        assert!(far < 1e-12, "{far:.3e}");
    }

    /// The partial core on the grid carries the radial model core's charge 4π∫ρ_pc r²dr (filtered table, off a
    /// grid point): the lattice sum of a band-limited function is its integral up to its leakage beyond π/h —
    /// Task 5 measured the weight-1 spread of O's ρ_pc at ≤ 2.3e-6 relative at h = 0.25 (≤ 7.7e-6 through the
    /// log-grid interpolation the uniform table removes), so 1e-5 relative. The no-core branch (H) adds nothing.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn partial_core_field_carries_the_model_core() {
        use crate::radial::test_cache::filtered_at;
        let h = 0.2;
        let (o, hy) = (filtered_at(8, h), filtered_at(1, h));
        let setup = IonicSetup::from_tables(crate::ionic::IonicPolicy::All, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(o.clone())), (1, Ionic::Pseudo(hy))]));
        let g = Grid::cube(61, h);
        let nuc = [Nucleus { pos: [0.037, -0.061, 0.023], z: 8 }, Nucleus { pos: [1.85, 0.0, 0.0], z: 1 }];
        let pc = partial_core_field(&g, &nuc, &setup);
        let radial = o.grid.integrate(&o.rho_pc.iter().zip(&o.grid.r).map(|(f, r)| 4.0 * PI * r * r * f).collect::<Vec<_>>());
        let grid = g.integrate(&pc);
        eprintln!("O ρ_pc at h = {h}: grid {grid:.10} vs radial {radial:.10} (relative {:.2e}); table radius {:.3}", (grid - radial) / radial, o.pc_radius);
        assert!(((grid - radial) / radial).abs() < 1e-5);
        assert!(has_partial_core(&nuc, &setup) && !has_partial_core(&nuc[1..], &setup));
        assert!(partial_core_field(&g, &nuc[1..], &setup).iter().all(|&x| x == 0.0), "the no-core branch");
    }

    /// E_ii with valence charges for pseudised species and Z for all-electron ones; the coincidence error.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn ionic_repulsion_uses_valence_charges() {
        use crate::radial::test_cache::filtered_at;
        let h = 0.2;
        let setup = IonicSetup::from_tables(crate::ionic::IonicPolicy::CoresOnly, h, h,
            std::collections::BTreeMap::from([(8, Ionic::Pseudo(filtered_at(8, h))), (1, Ionic::AllElectron { sigma: h })]));
        let nuc = [Nucleus { pos: [0.0; 3], z: 8 }, Nucleus { pos: [2.28, 0.0, 0.0], z: 8 }, Nucleus { pos: [0.0, 1.8, 0.0], z: 1 }];
        let d12 = (2.28f64 * 2.28 + 1.8 * 1.8).sqrt();
        assert_relative_eq!(ionic_repulsion(&nuc, &setup).unwrap(), 36.0 / 2.28 + 6.0 / 1.8 + 6.0 / d12, epsilon = 1e-13);
        assert!(matches!(ionic_repulsion(&[nuc[0], nuc[0]], &setup), Err(QmError::CoincidentNuclei(0, 1))));
        let c = centres(&nuc, &setup);
        assert_eq!((c[0].charge, c[0].sigma, c[2].charge, c[2].sigma), (6.0, 2.0 * h, 1.0, h));
    }
}
