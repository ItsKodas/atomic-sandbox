//! Headless physics-oracle suite (design spec §9, layer 2). Every reference number used by a
//! test is cited in `references.md`; the latest baseline is written by hand to `last-run.md`
//! (gitignored — the numbers go in the commit message). Run with
//! `cargo test -p validate --release` (minutes; see the per-test wall times in `references.md`).
//!
//! # Why every oracle extrapolates σ → 0
//! The electrons see each nucleus as a Gaussian charge of width σ (`qm::nuclear`), which raises
//! the energy by ΔE(σ) = 2π Z ρ(R) σ² + c₃σ³ + c₄σ⁴ + … (the σ³ term comes from the density
//! cusp). For a hydrogenic cusp ρ(d) ≈ ρ(0) e^{−2Zd} the whole first-order series is a function
//! of s = Zσ alone (substitute u = Zd in ΔE₁ = Z ∫ ρ erfc(d/σ√2)/d d³r):
//!   ΔE₁(σ) = (4π ρ(0)/Z) · F(Zσ),   4F(s) = 1 − 2a/√π + (2a² − 1) e^{a²} erfc(a),   a = s√2,
//! which is the closed form of `qm::scf`'s module doc (H: Z = 1, ρ(0) = 1/π, prefactor 4),
//! an alternating series in s with ratio ≈ 2s. `qm::extrapolate_sigma` fits
//! E = a + bσ² + cσ³ + dσ⁴ through four points and returns a; the dropped s⁵ term leaks into a.
//! Measured on H (Task 9, σ ∈ [0.25, 0.40]): +1.73 mHa on the first-order curve, +4.67 mHa on
//! the exact relaxed energies (radial Numerov). The bias therefore scales as
//! (4πρ(0)/Z) × s_max⁵ ∝ ρ(0) Z⁴ σ_max⁵, and σ_max is pinned to the grid: σ ≥ h is needed for
//! the 8th-order stencil to resolve the smeared orbital (`qm::nuclear`: truncation ∝ (h/σ)⁸),
//! so the ladder σ = h·(1, 1.2, 1.4, 1.6) gives s_max = 1.6 Z h. The bias is controlled by Z·h.
//!
//! Model predictions for the four-point fit on the first-order curve (`first_order_shift`),
//! with the H relaxed/first-order ratio 2.7 in brackets as the expected size on real energies:
//!   H  (ρ(0) = 1/π,  Z = 1)  h = 0.15: +0.21 mHa [0.6]                       bound 4 mHa
//!   He (ρ(0) ≈ 3.6,  Z = 2)  h = 0.15: +20 mHa [53];   h = 0.10: +4.0 mHa [11]   bound 15 mHa
//!   Li (ρ(0) ≈ 13.8, Z = 3)  h = 0.15: +0.21 Ha [0.56]; h = 0.10: +50 mHa [136];
//!                            h = 0.075: +17 mHa [45]                           bound 40 mHa
//! (prefactors 4, 22.6, 57.8; the cusp Z is used because the smearing integrand lives within
//! ~2σ of the nucleus where the cusp, not the orbital's mean exponent, sets the slope). So H is
//! safe at h = 0.15, He at h = 0.15 is predicted to miss and h = 0.10 to be marginal, and Li
//! cannot meet 40 mHa with this model at any grid the CPU reference can afford (h = 0.075 in a
//! 16 Bohr box is 214³ ≈ 9.8 M points). The tests report the raw E(σ) table, the fit, and
//! E(σ) − a against the first-order curve so a miss is attributable (grid error vs extrapolation
//! bias vs solver); no bound is loosened here.
//!
//! Measured (Task 11, release): H h = 0.15 → a = −0.477710, +0.96 mHa; He h = 0.15 →
//! −2.788945, +45.9 mHa (fails 15), h = 0.10 → −2.822561, +12.3 mHa; Li h = 0.15 → −7.004270,
//! +340 mHa against NIST LSD −7.343957 (fails 40). The misses are extrapolation bias, not grid error: the quartic fitted to
//! the He h = 0.10 ladder reproduces the h = 0.15 ladder's E(0.15) to 0.3 mHa (and the H h = 0.15
//! fit reproduces Task 9's h = 0.25 E(0.25) to 0.1 mHa) while the intercepts differ by 34 mHa.
//! E(σ) − a is 0.77 → 0.71 (H), 0.73 → 0.67 (He, h = 0.10), 0.50 → 0.48 (Li) of the first-order
//! curve — density relaxation against the smeared potential, decreasing smoothly with σ, as
//! Task 9 found for H against an exact radial reference. Rulings (2026-10-13): the He oracle
//! runs at h = 0.10 (`Grid::cube(101, 0.10)`, 10 Bohr, bound unchanged); the Li oracle keeps
//! its design and bound but is `#[ignore]`d as a documented limitation of the Gaussian-smeared
//! all-electron rung for Z ≥ 3 (its doc comment has the numbers and the planned remedies:
//! self-generated pseudopotentials, spec §11 item 1, or h ≲ 0.07 on the GPU solver).
//!
//! # Oracle designs
//! - **Atoms** (`h_atom_lsda`, `he_atom_lsda`, `li_atom_lsda`): `Grid::cube(n, h)` with the box
//!   chosen so the face density stays below the 1e-5 warning (H 12 Bohr at h = 0.15; He 10 Bohr
//!   at h = 0.10, face 1.6e-7; Li 16 Bohr at h = 0.15, the warning fired at 12 Bohr in Task 9),
//!   σ ladder h·(1, 1.2, 1.4, 1.6), SCF tol 1e-7 and
//!   eig_tol 1e-7 (energy is second order in both residuals, so E(σ) is held to ≪ 1e-6 Ha —
//!   the fit's intercept weights sum to |w|₁ ≈ 88, so 1e-6 noise is < 1e-4 in a). Each σ is
//!   warm-started from the previous one: the fixed point does not depend on the start (same
//!   geometry, same tolerance), and it saves most of the SCF steps.
//! - **H₂ bond length** (`h2_bond_length`): at σ = h = 0.25 the smeared minimum sits at
//!   R ≈ 1.79 Bohr (Task 10), because the smearing term 2πσ² Σ_I ρ(R_I) is repulsive in R, so
//!   R_e itself must be extrapolated. By the implicit-function theorem the root of
//!   F(R; σ) = F₀(R) + σ²F₂(R) + σ³F₃(R) + … has the same expansion R_e(σ) = R₀ + r₂σ² + r₃σ³ + …,
//!   so `extrapolate_sigma` applies unchanged. R_e(σ) is located from the Hellmann-Feynman
//!   force on nucleus 0 (the exact derivative of the SCF energy to 6e-8 Ha/Bohr, Task 10) rather
//!   than from a parabola through E(R), whose vertex is biased by E‴Δ²/(6E″) ≈ 0.016 Bohr at
//!   Δ = 0.15. The scan R ∈ {1.45, 1.60, 1.75, 1.90} brackets the σ-shifted minimum for every
//!   σ in the ladder (expected R_e(σ) ≈ 1.57–1.78 from the σ² scaling of the +0.35 Bohr shift
//!   measured at σ = 0.25); the root of the Lagrange cubic through the four F values is taken in
//!   the interval where F changes sign (F > 0 compressed, < 0 stretched, so it is unique), and a
//!   missing sign change fails the test loudly. Egg-box noise in F (≤ 4e-5 Ha/Bohr) moves the
//!   root by ≤ 1e-4 Bohr, i.e. ≤ 1e-2 Bohr after the ×88 extrapolation weights — below the
//!   0.05 Bohr bound. The same amplification is why the interpolant must use the *same* nodes
//!   at every σ: a 3-point parabola whose bracketing triple changes with σ differs from the
//!   cubic by only ~1e-3 Bohr per σ, but the error alternates in sign along the ladder and
//!   extrapolates to 1.539 Bohr (measured) instead of 1.451.
//!   Measured (Task 11): R_e(σ) = 1.6008 / 1.6530 / 1.7092 / 1.7685 → 1.4511 Bohr (+0.005);
//!   F_x0(1.446; σ) = +0.0516 / +0.0690 / +0.0874 / +0.1066 → +0.0010 Ha/Bohr.
//! - **H₂ force at equilibrium** (`h2_force_vanishes_at_equilibrium`): F_x on nucleus 0 at
//!   R = 1.446 for the four σ (tol 1e-9, eig_tol 1e-8 as in Task 10), extrapolated to σ → 0;
//!   at finite σ the force is positive (≈ +0.12 at σ = 0.25) and the extrapolated value must
//!   vanish to 1e-2 Ha/Bohr.
//!
//! All energies in Hartree, lengths in Bohr, forces in Hartree/Bohr.

pub use qm::extrapolate_sigma;
use qm::{scf, Grid, KsState, ScfSettings, ScfStats, System};

/// The σ ladder used by every oracle: σ = h·(1, 1.2, 1.4, 1.6) (module doc).
pub fn sigma_ladder(h: f64) -> [f64; 4] { [1.0, 1.2, 1.4, 1.6].map(|f| f * h) }

/// One converged SCF at `sigma` with the given tolerances, warm-started from `warm` if given.
/// Panics (with the error) if the SCF does not converge: an oracle must not silently skip.
pub fn converged(grid: &Grid, sys: &System, sigma: f64, tol: f64, eig_tol: f64, warm: Option<&KsState>) -> (KsState, ScfStats) {
    let mut s = ScfSettings::for_grid(grid);
    s.sigma = sigma; s.tol = tol; s.eig_tol = eig_tol;
    scf(grid, sys, &s, warm).unwrap_or_else(|e| panic!("SCF at σ = {sigma} failed: {e}"))
}

/// E(σ) over `sigmas`, each σ warm-started from the previous one (module doc). Returns the
/// energies and the per-σ stats (iterations, face density, warnings) for the report.
pub fn energy_ladder(grid: &Grid, sys: &System, sigmas: &[f64; 4], tol: f64, eig_tol: f64) -> ([f64; 4], Vec<ScfStats>) {
    let mut e = [0.0; 4];
    let mut stats = Vec::with_capacity(4);
    let mut prev: Option<KsState> = None;
    for (k, &sigma) in sigmas.iter().enumerate() {
        let (state, st) = converged(grid, sys, sigma, tol, eig_tol, prev.as_ref());
        e[k] = st.energy.total;
        stats.push(st);
        prev = Some(state);
    }
    (e, stats)
}

/// Exact first-order smearing shift of a hydrogenic cusp density ρ(d) = ρ₀ e^{−2Zd}
/// (module doc): (4πρ₀/Z)·F(Zσ) with 4F(s) = 1 − 2a/√π + (2a² − 1)e^{a²}erfc(a), a = s√2.
/// For H (Z = 1, ρ₀ = 1/π) this is the closed form pinned in `qm::scf`'s tests.
pub fn first_order_shift(sigma: f64, z: f64, rho0: f64) -> f64 {
    let a = z * sigma * 2f64.sqrt();
    let f4 = 1.0 - 2.0 * a / std::f64::consts::PI.sqrt() + (2.0 * a * a - 1.0) * (a * a).exp() * libm::erfc(a);
    std::f64::consts::PI * rho0 / z * f4
}

/// Value at `x` of the Lagrange polynomial through the points (xs[i], ys[i]).
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..xs.len() {
        let mut l = ys[i];
        for j in 0..xs.len() { if j != i { l *= (x - xs[j]) / (xs[i] - xs[j]); } }
        sum += l;
    }
    sum
}

/// Root of the Lagrange polynomial through (xs, ys) inside [lo, hi], which must bracket a sign
/// change of the interpolant; bisection to 1e-10 (the interpolant is cheap, so no Newton).
pub fn interpolated_root(xs: &[f64], ys: &[f64], lo: f64, hi: f64) -> f64 {
    let (mut a, mut b) = (lo, hi);
    let (mut fa, fb) = (lagrange(xs, ys, a), lagrange(xs, ys, b));
    assert!(fa * fb <= 0.0, "no sign change of the interpolant in [{lo}, {hi}]: {fa} / {fb}");
    while b - a > 1e-10 {
        let m = 0.5 * (a + b);
        let fm = lagrange(xs, ys, m);
        if fa * fm <= 0.0 { b = m; } else { a = m; fa = fm; }
    }
    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn first_order_shift_reduces_to_the_hydrogen_closed_form() {
        // qm::scf pins ΔE₁(σ) for H at 0.0766 / 0.1010 / 0.1262 Ha (σ = 0.25 / 0.30 / 0.35).
        assert_relative_eq!(first_order_shift(0.25, 1.0, 1.0 / std::f64::consts::PI), 0.0766, epsilon = 5e-5);
        assert_relative_eq!(first_order_shift(0.30, 1.0, 1.0 / std::f64::consts::PI), 0.1010, epsilon = 5e-5);
        assert_relative_eq!(first_order_shift(0.35, 1.0, 1.0 / std::f64::consts::PI), 0.1262, epsilon = 5e-5);
        // small-σ limit 2π Z ρ₀ σ²: at Zσ = 0.01 the series is 2s² − 4.26s³ ⇒ ratio 0.979
        let z = 2.0; let rho0 = 3.6; let s = 0.005;
        assert_relative_eq!(first_order_shift(s, z, rho0) / (2.0 * std::f64::consts::PI * z * rho0 * s * s), 0.979, epsilon = 2e-3);
    }

    #[test]
    fn interpolated_root_of_a_cubic_is_exact() {
        // f(R) = −0.37 (R − 1.6) + 0.8 (R − 1.6)³ sampled on the oracle's scan has its root at 1.6.
        let xs = [1.45, 1.60, 1.75, 1.90];
        let f = |r: f64| -0.37 * (r - 1.6) + 0.8 * (r - 1.6f64).powi(3);
        let ys = xs.map(f);
        assert_relative_eq!(interpolated_root(&xs, &ys, 1.45, 1.75), 1.6, epsilon = 1e-9);
        assert_relative_eq!(lagrange(&xs, &ys, 1.7), f(1.7), epsilon = 1e-14);
    }

    #[test]
    fn sigma_ladder_is_h_times_one_to_1_6() {
        let s = sigma_ladder(0.15);
        for (a, b) in s.iter().zip([0.15, 0.18, 0.21, 0.24]) { assert_relative_eq!(*a, b, epsilon = 1e-15); }
    }
}
