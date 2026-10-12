//! LOBPCG (Knyazev 2001) for the lowest k eigenpairs of a grid Hamiltonian, with full
//! subspace orthonormalisation. Inner product = `Grid::dot` (L², h³-weighted) throughout.
//!
//! # Problem
//! H ψ = E ψ with H = −½∇² + V(r) (atomic units, ħ = mₑ = 1; `stencil::kinetic` is −½∇²
//! with f = 0 outside the box). H is symmetric in the L² inner product ⟨a, b⟩ = h³ Σ aᵢbᵢ:
//! the stencil is symmetric (`poisson` proves L_int = L_intᵀ) and V is diagonal. We want the
//! k lowest eigenpairs. For symmetric H the lowest k eigenvalues are the minimum of
//!   tr(XᵀHX) over X with XᵀX = I (k columns),
//! which is what LOBPCG minimises iteratively.
//!
//! # Rayleigh–Ritz in an orthonormalised subspace
//! Given any subspace spanned by the columns of S = [s₁ … s_m], the best approximations to
//! eigenpairs within it are the stationary points of the Rayleigh quotient
//!   ρ(Sc) = cᵀ(SᵀHS)c / cᵀ(SᵀMS)c,      M = L² mass matrix (h³·I),
//! i.e. the generalised symmetric problem (SᵀHS)c = θ(SᵀMS)c. If S is first made
//! M-orthonormal (SᵀMS = I) by modified Gram–Schmidt in `grid.dot`, this collapses to the
//! *standard* symmetric problem A c = θ c with A = SᵀHS (`nalgebra::SymmetricEigen`), and the
//! Ritz vectors S C are orthonormal for free because the eigenvector matrix C of a symmetric
//! matrix is orthogonal. By Cauchy interlacing the Ritz values θ₁ ≤ … ≤ θ_m are upper bounds
//! on the corresponding eigenvalues of H, so taking the lowest k is the right choice.
//! `SymmetricEigen` returns eigenvalues in no particular order; we sort them explicitly.
//!
//! # The iteration
//! Each iteration, with X (k orthonormal columns) and Λ = diag of the Rayleigh quotients
//! λᵢ = ⟨xᵢ, Hxᵢ⟩:
//!   R = HX − XΛ                     residuals (the gradient of the Rayleigh quotient),
//!   W = T R                         preconditioned residuals (T = identity if none),
//!   S = orthonormalise([X, W, P])   m ≤ 3k columns,
//!   A = SᵀHS,  A C = C Θ,           keep the k lowest θ,
//!   X_new = S C_k,
//!   P_new = X_new − X (XᵀX_new) = S_{W,P part} C_k.
//! The P update: textbook LOBPCG writes X_new = X C_X + W C_W + P C_P and sets
//! P_new = W C_W + P C_P, "the direction X moved along". Because MGS leaves an already
//! orthonormal prefix unchanged (up to rounding), the first k columns of S *are* X, so the
//! W,P part of S spans the W and P directions projected orthogonally to X, and P_new is the
//! Ritz vector minus its X-component. This is the orthogonalised-P variant (Hetmaniuk &
//! Lehoucq 2006) that keeps S well conditioned. With S = [X, W] alone the method is
//! preconditioned block steepest descent; the P block upgrades it to a conjugate-gradient
//! three-term recurrence and the CG convergence rate. On the first iteration P is empty.
//!
//! `orthonormalise` drops vectors of norm < 1e-12 (an exactly converged residual gives W = 0,
//! and a Ritz vector with no component outside X gives P = 0), so m may be < 3k. The P-part
//! test "column index ≥ k" uses positions *after* drops and is correct only because X's k
//! vectors are never dropped: they are orthonormal on entry (initial MGS + assert) and the
//! Ritz vectors are orthonormal up to rounding, so their re-orthonormalisation at the end of
//! each iteration can never fall below the drop threshold; an assert enforces this.
//!
//! # Convergence criterion
//! λᵢ = ⟨xᵢ, Hxᵢ⟩ is the μ minimising ‖Hxᵢ − μxᵢ‖, and for symmetric H there is an
//! eigenvalue within ‖rᵢ‖ of λᵢ (Kato–Temple gives the sharper ‖rᵢ‖²/gap). We stop when
//! max_i ‖rᵢ‖_{L²} < tol, which bounds the eigenvalue error of the *discrete* H; the
//! discretisation error against the continuum (h⁸ for the stencil, plus the Dirichlet
//! box) is separate and is what the oscillator test's 1e-4 tolerance covers.
//!
//! # Preconditioner: multi-sweep damped Jacobi on (H − λ) w = r
//! The diagonal. The 8th-order stencil contributes its centre coefficient c₀ = −205/72 once
//! per axis at every point, including points at the faces (ghost terms only remove
//! off-diagonal entries), so diag(−½∇²) = −½ · 3c₀/h² = −1.5 c₀/h² (> 0 since c₀ < 0;
//! 68.3 Ha at h = 0.25) and D := max(diag(H) − λ, 0.1) with diag(H) = −1.5c₀/h² + V(r).
//! The floor 0.1 keeps D positive where V − λ < −1.5c₀/h².
//!
//! Why one Jacobi sweep (w = D⁻¹ r) is useless here. LOBPCG is exactly invariant to a scalar
//! preconditioner (span[X, αR, P] = span[X, R, P]), and on a uniform grid the kinetic
//! diagonal is a constant that dominates: for the oscillator test at h = 0.25, D is 66.8 at
//! the origin and 71.3 at r = 3 (±3% where the eigenfunctions live), reaching 90.8 only at
//! the box corners. Measured on the 33³ test, k = 2, tol 1e-6: plain 136 iterations, a pure
//! scalar r/68 also 136, one-sweep Jacobi 138. This is the same observation `poisson.rs`
//! makes for Jacobi-PCG on the constant-diagonal Laplacian.
//!
//! What does work is to use the diagonal as a *smoother*: `JACOBI_SWEEPS` sweeps of damped
//! Jacobi on (H − λ) w = r,
//!   w₀ = D⁻¹ r,    w_{k+1} = w_k + ω D⁻¹ (r − (H − λ) w_k),   k = 0 … s − 2,
//! where w₀ counts as sweep 1 and is followed by s − 1 damped updates (s = `JACOBI_SWEEPS`),
//! so T = p(D⁻¹(H − λ)) D⁻¹ with p a polynomial that approximates the inverse on the
//! high-frequency part of the spectrum — the part LOBPCG's low-dimensional Rayleigh–Ritz
//! cannot resolve by itself. One sweep is exactly the diagonal scaling above.
//!
//! Damping factor. The error propagator of a sweep is I − ω D⁻¹(H − λ). Per axis the stencil
//! symbol runs from 0 (θ = 0) to |c₀ + 2 Σ_m (−1)^m c_m| = 6.50 at θ = π against the diagonal
//! |c₀| = 2.85, so the eigenvalues μ of D⁻¹(H − λ) lie in [≈0, ρ] with
//!   ρ = (c₀ + 2 Σ_m (−1)^m c_m) / c₀ = 2.284
//! (the 3D Kronecker sum has the same ratio; V(r) adds equally to numerator and denominator
//! and only pulls the ratio toward 1, so ρ is the bound). Stability needs ω < 2/ρ = 0.876.
//! A smoother should damp the upper half μ ∈ [ρ/2, ρ]; minimising max |1 − ωμ| there gives
//!   ω = 4/(3ρ) = 0.584,    smoothing factor 1/3 per sweep,
//! the same construction that yields the textbook ω = 2/3 for the 2nd-order stencil (ρ = 2).
//! ω is computed from `stencil::COEFF` so it tracks the stencil. Modes with μ < 0 (where
//! λ > V, the low modes) are amplified by at most (1 + ω|μ|)^{s−1} ≈ 1 + 3·0.584·0.02 at s = 4,
//! negligible,
//! and LOBPCG re-optimises those directions anyway.
//!
//! Positivity. LOBPCG needs T symmetric positive definite. T = D^{-1/2} p(N) D^{-1/2} with
//! N = D^{-1/2}(H − λ)D^{-1/2} symmetric, so T is SPD iff p(ν) > 0 on N's spectrum, where
//! after s sweeps (w₀ plus s − 1 damped updates) p(ν) = 1/ν + (1 − ων)^{s−1} (1 − 1/ν). A
//! sharper bound including V − λ < 0 at the centre is ν ≤ (ρ D_T − 1.5)/(D_T − 1.5) = 2.31 on
//! the oscillator test. For s = 2, p(ν) = 1 + ω(1 − ν) changes sign at ν = 1 + 1/ω, so
//! ω < 1/(ν_max − 1) = 0.76 is required; the derived ω = 0.584 gives p(ν_max) = 0.23. For
//! s = 4 (the default), p(ν_max) > 0 needs (1 − ων_max)³ > −1/(ν_max − 1), i.e.
//! ω < (1 + (ν_max − 1)^{−1/3})/ν_max ≈ 0.83. Measured: ω = 0.85 (stable as an iteration,
//! |1 − ων| < 1, but T indefinite at both s = 2 and s = 4) does not converge in 500 iterations, while
//! ω = 0.584 and ω = 2/3 behave identically to within three iterations at every s.
//!
//! Each sweep after the first costs one `op.apply`; measured on the 33³ test (k = 2,
//! tol 1e-6, ω = 0.584): s = 1/2/4/8 gives 138/73/53/37 iterations against 136 plain, and
//! wall time 0.75/0.43/0.37/0.34 s against 0.72 s. Returns diminish past s = 4 (8% more
//! wall-time saving for twice the applications per residual), so `JACOBI_SWEEPS` = 4.
//!
//! # Cost
//! Orthonormalising all of S and recomputing H S each iteration costs more than textbook
//! LOBPCG (which reuses HX, HW, HP), but is numerically robust against the ill-conditioning
//! of [X, W, P] near convergence; the GPU plan may revisit.

use crate::grid::Grid;
use crate::stencil::{kinetic, COEFF};
use nalgebra::{DMatrix, SymmetricEigen};

/// A symmetric linear operator on grid functions: y = op(x).
pub trait Operator { fn apply(&self, x: &[f64], y: &mut [f64]); }

/// H = −½∇² + V(r) for a local potential `v` on the grid.
pub struct LocalHamiltonian<'a> { pub grid: &'a Grid, pub v: &'a [f64] }

impl Operator for LocalHamiltonian<'_> {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        kinetic(self.grid, x, y);
        for i in 0..y.len() { y[i] += self.v[i] * x[i]; }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EigenStats {
    /// Number of Rayleigh–Ritz steps taken (0 = the initial guess already met `tol`).
    pub iterations: usize,
    /// max_i ‖H xᵢ − λᵢ xᵢ‖_{L²} at exit.
    pub max_residual: f64,
}

/// Number of damped-Jacobi sweeps in `jacobi_precond` (1 = plain diagonal scaling).
pub const JACOBI_SWEEPS: usize = 4;

/// Damping factor ω = 4/(3ρ), ρ = max eigenvalue of D⁻¹(−∇²) = symbol(π)/c₀ (see module doc).
fn jacobi_omega() -> f64 {
    let symbol_pi = COEFF[0] + 2.0 * (-COEFF[1] + COEFF[2] - COEFF[3] + COEFF[4]);
    let rho = symbol_pi / COEFF[0];
    4.0 / (3.0 * rho)
}

/// Preconditioner for `LocalHamiltonian { grid, v }`: `JACOBI_SWEEPS` sweeps of damped Jacobi
/// on (H − λ) w = r starting from w = r / max(diag(H) − λ, 0.1), diag(H) = −1.5·c₀/h² + V(r)
/// (see module doc).
pub fn jacobi_precond<'a>(grid: &'a Grid, v: &'a [f64], lambda: f64) -> impl Fn(&[f64], &mut [f64]) + 'a {
    let diag_t = -1.5 * COEFF[0] / (grid.h * grid.h); // diagonal of −½∇²
    let inv_d: Vec<f64> = v.iter().map(|&vi| 1.0 / (diag_t + vi - lambda).max(0.1)).collect();
    let omega = jacobi_omega();
    let op = LocalHamiltonian { grid, v };
    move |r: &[f64], w: &mut [f64]| {
        for i in 0..r.len() { w[i] = r[i] * inv_d[i]; }
        let mut hw = vec![0.0; r.len()];
        for _ in 1..JACOBI_SWEEPS {
            op.apply(w, &mut hw);
            for i in 0..r.len() { w[i] += omega * inv_d[i] * (r[i] - (hw[i] - lambda * w[i])); }
        }
    }
}

/// Modified Gram–Schmidt in the grid inner product; drops vectors with norm < 1e-12.
fn orthonormalise(grid: &Grid, s: &mut Vec<Vec<f64>>) {
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(s.len());
    for mut v in s.drain(..) {
        for q in &kept {
            let c = grid.dot(q, &v);
            for i in 0..v.len() { v[i] -= c * q[i]; }
        }
        let nrm = grid.dot(&v, &v).sqrt();
        if nrm > 1e-12 { for x in v.iter_mut() { *x /= nrm; } kept.push(v); }
    }
    *s = kept;
}

/// Lowest `x.len()` eigenpairs of `op` with respect to the L² inner product `grid.dot`.
/// `x` holds the initial guess (any linearly independent vectors) and the result, orthonormal;
/// `evals[i]` receives the Rayleigh quotient ⟨xᵢ, H xᵢ⟩ of the returned xᵢ, ascending.
/// `precond` maps a residual to an approximate correction (None = identity).
/// Stops when max_i ‖H xᵢ − λᵢ xᵢ‖ < `tol` or after `max_iter` Rayleigh–Ritz steps.
pub fn lobpcg(grid: &Grid, op: &dyn Operator, x: &mut [Vec<f64>], evals: &mut [f64],
              precond: Option<&dyn Fn(&[f64], &mut [f64])>, tol: f64, max_iter: usize) -> EigenStats {
    let k = x.len();
    let n = grid.len();
    assert_eq!(evals.len(), k, "evals must have one slot per eigenvector");
    let mut xs: Vec<Vec<f64>> = x.to_vec();
    orthonormalise(grid, &mut xs);
    assert_eq!(xs.len(), k, "initial guess vectors are linearly dependent");
    let mut p: Vec<Vec<f64>> = Vec::new();
    let mut hx: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    let mut stats = EigenStats { iterations: 0, max_residual: f64::INFINITY };

    loop {
        // Rayleigh quotients and residuals
        for i in 0..k { op.apply(&xs[i], &mut hx[i]); evals[i] = grid.dot(&xs[i], &hx[i]); }
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut max_res = 0.0f64;
        for i in 0..k {
            let r: Vec<f64> = (0..n).map(|j| hx[i][j] - evals[i] * xs[i][j]).collect();
            max_res = max_res.max(grid.dot(&r, &r).sqrt());
            let mut wi = vec![0.0; n];
            match precond { Some(t) => t(&r, &mut wi), None => wi.copy_from_slice(&r) }
            w.push(wi);
        }
        stats.max_residual = max_res;
        if max_res < tol || stats.iterations >= max_iter { break; }
        stats.iterations += 1;

        // Subspace S = [X, W, P], orthonormalised. X's k vectors are orthonormal and come
        // first, so MGS keeps them at positions 0..k; only W/P vectors can be dropped.
        let mut s: Vec<Vec<f64>> = Vec::with_capacity(3 * k);
        s.extend(xs.iter().cloned());
        s.extend(w.into_iter());
        s.extend(p.iter().cloned());
        orthonormalise(grid, &mut s);
        let m = s.len();

        // Rayleigh–Ritz: (SᵀHS) c = θ c
        let mut hs: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        for i in 0..m { op.apply(&s[i], &mut hs[i]); }
        let mut a = DMatrix::<f64>::zeros(m, m);
        for i in 0..m { for j in 0..=i {
            let v = grid.dot(&s[i], &hs[j]);
            a[(i, j)] = v; a[(j, i)] = v;
        }}
        let eig = SymmetricEigen::new(a);
        // SymmetricEigen does not sort; take the k lowest Ritz values explicitly.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));

        // New X and P from the lowest k Ritz vectors; P excludes the X-part of S (first k columns)
        let mut new_x = vec![vec![0.0; n]; k];
        let mut new_p = vec![vec![0.0; n]; k];
        for (col, &idx) in order.iter().take(k).enumerate() {
            let c = eig.eigenvectors.column(idx);
            for (si, sv) in s.iter().enumerate() {
                let coef = c[si];
                if coef == 0.0 { continue; }
                for j in 0..n { new_x[col][j] += coef * sv[j]; }
                if si >= k { for j in 0..n { new_p[col][j] += coef * sv[j]; } }
            }
        }
        xs = new_x;
        // Ritz vectors are orthonormal up to rounding; this only cleans up and never drops.
        orthonormalise(grid, &mut xs);
        assert_eq!(xs.len(), k, "Ritz vectors lost rank; the Rayleigh–Ritz step is broken");
        p = new_p;
    }
    for i in 0..k { x[i].copy_from_slice(&xs[i]); }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;

    /// 3D isotropic harmonic oscillator: H = −½∇² + ½r², E = n + 3/2 ⇒ 1.5, 2.5 (×3), 3.5 (×6)
    fn oscillator(n: usize, h: f64) -> (Grid, Vec<f64>) {
        let g = Grid::cube(n, h);
        let v = g.sample(|[x, y, z]| 0.5 * (x * x + y * y + z * z));
        (g, v)
    }

    fn random_guess(g: &Grid, k: usize) -> Vec<Vec<f64>> {
        // deterministic pseudo-random so the test is reproducible
        let mut seed = 12345u64;
        (0..k).map(|_| g.sample(|[x, y, z]| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            (u - 0.5) * (-(x * x + y * y + z * z) / 8.0).exp()
        })).collect()
    }

    #[test]
    fn oscillator_spectrum() {
        let (g, v) = oscillator(49, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let mut x = random_guess(&g, 4);
        let mut e = vec![0.0; 4];
        let stats = lobpcg(&g, &op, &mut x, &mut e, None, 1e-6, 500);
        assert!(stats.max_residual < 1e-6, "{stats:?}");
        assert_relative_eq!(e[0], 1.5, epsilon = 1e-4);
        for i in 1..4 { assert_relative_eq!(e[i], 2.5, epsilon = 1e-4); }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let (g, v) = oscillator(33, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let mut x = random_guess(&g, 3);
        let mut e = vec![0.0; 3];
        lobpcg(&g, &op, &mut x, &mut e, None, 1e-6, 500);
        for i in 0..3 { for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!((g.dot(&x[i], &x[j]) - expect).abs() < 1e-8);
        }}
    }

    #[test]
    fn preconditioner_reduces_iterations() {
        let (g, v) = oscillator(33, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let (mut x1, mut e1) = (random_guess(&g, 2), vec![0.0; 2]);
        let plain = lobpcg(&g, &op, &mut x1, &mut e1, None, 1e-6, 500);
        let (mut x2, mut e2) = (random_guess(&g, 2), vec![0.0; 2]);
        let pc = jacobi_precond(&g, &v, 1.5);
        let pre = lobpcg(&g, &op, &mut x2, &mut e2, Some(&pc), 1e-6, 500);
        assert!(pre.iterations < plain.iterations, "plain {} pre {}", plain.iterations, pre.iterations);
    }

    #[test]
    fn warm_start_converges_immediately() {
        let (g, v) = oscillator(33, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let (mut x, mut e) = (random_guess(&g, 2), vec![0.0; 2]);
        lobpcg(&g, &op, &mut x, &mut e, None, 1e-8, 500);
        let again = lobpcg(&g, &op, &mut x, &mut e, None, 1e-6, 500);
        assert!(again.iterations <= 1, "{again:?}");
    }
}
