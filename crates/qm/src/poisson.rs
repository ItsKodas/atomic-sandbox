//! Poisson solver: ∇²V = −4πρ (atomic units), isolated (free-space) boundary conditions.
//!
//! # Equation and sign
//! A charge density ρ (e/Bohr³) produces V(r) = ∫ ρ(r′)/|r − r′| d³r′ (Hartree). Since
//! ∇²(1/|r|) = −4πδ(r), this V satisfies ∇²V = −4πρ. For ρ > 0 the potential is positive
//! (Q/r far away), so this is the potential of the charge itself; the energy of an electron
//! (charge −1) in it is −V.
//!
//! # Discretisation and the boundary split
//! The 8th-order stencil L of `crate::stencil` needs four neighbours per side. For a point
//! near a face some neighbours fall outside the box ("ghost points"). Write
//!   L V = L_int V + L_bc,
//! where L_int V is the stencil with ghosts taken as 0 (exactly `stencil::laplacian`) and
//!   (L_bc)_i = (1/h²) Σ_{ghost neighbours g of i} c_m V_bc(r_g)
//! collects the dropped terms with the *known* ghost values V_bc. L_bc does not depend on
//! the unknown V, so the discrete equation L V = −4πρ becomes
//!   A V = b,   A = −L_int,   b = 4πρ + L_bc.
//!
//! # A is symmetric positive-definite
//! Symmetry: c_{−m} = c_m, and if j is i's neighbour at offset +m then i is j's at −m with
//! the same coefficient; dropping out-of-box neighbours is mutual, so L_int = L_intᵀ.
//! Positivity: on one axis d²/dx² = ((2/h) asinh(hδ/2))² exactly (δ = centred difference),
//! and the 2p-order Fornberg stencil is the truncation of that series to p terms, whose
//! Fourier symbol is −(4/h²) Σ_{k=1}^{p} a_k (s/4)^k with s = 4 sin²(θ/2) and a_k > 0.
//! Hence −L₁ has symbol ≥ 0, vanishing only at θ = 0. The Dirichlet matrix −L₁,int is a
//! finite section of that Toeplitz operator, so its eigenvalues lie strictly inside the
//! range of the symbol, i.e. are > 0. The 3D operator is the Kronecker sum
//! −L₁⊕(−L₁)⊕(−L₁), whose eigenvalues are sums of positive numbers. So A is SPD and
//! conjugate gradient converges, monotonically in the A-norm.
//!
//! # Jacobi preconditioner
//! The centre coefficient c₀ (< 0) enters once per axis, so diag(A) = −3c₀/h² for every
//! point, including points at the faces (ghost terms only ever remove off-diagonal
//! entries). With a uniform h the diagonal is a constant, so Jacobi-PCG is mathematically
//! identical to plain CG here (CG is invariant to a scalar preconditioner); the
//! preconditioned form is kept so the structure matches the general (non-constant
//! diagonal) case and the GPU path.
//!
//! # Ghost values: multipole expansion in real solid harmonics
//! For |r′| < |r| the Laplace expansion 1/|r − r′| = Σ_l r′^l P_l(cos γ)/r^{l+1} together
//! with the addition theorem P_l(cos γ) = Σ_m S_lm(r̂′) S_lm(r̂) (S_lm real spherical
//! harmonics in Racah normalisation, S_lm = √(4π/(2l+1)) × orthonormal Y_lm) gives
//!   1/|r − r′| = Σ_{l,m} R_lm(r′) R_lm(r) / |r|^{2l+1},   R_lm(r) = |r|^l S_lm(r̂),
//! where R_lm are the *regular solid harmonics* — homogeneous harmonic polynomials:
//! R_00 = 1, R_10 = z, R_11 = x, R_1,−1 = y, R_20 = (3z² − r²)/2, R_21 = √3 xz, … So
//!   V_bc(r) = Σ_{l ≤ L_MAX} Σ_m Q_lm R_lm(r) / |r|^{2l+1},   Q_lm = ∫ ρ R_lm,
//! which at l ≤ 1 is exactly Q/|r| + p·r/|r|³ with Q = ∫ρ and p = ∫ρ r.
//! Truncation error at a ghost point for charge within radius d of the origin is bounded by
//! the Legendre tail Σ_{l>L} d^l/r^{l+1} = (d/r)^{L+1}/(r − d). Boundary errors propagate
//! essentially undiminished to interior points next to the faces (discrete maximum
//! principle), so this floor is visible in V. With L_MAX = 1 (monopole + dipole) a charge
//! 1.5 Bohr off centre in a 12 Bohr box gives ~1e-2; L_MAX = 6 brings it to ~1e-5, below
//! the 8th-order discretisation error at h = 0.25.
//!
//! R_lm is evaluated by the standard recursion on the complex Racah-normalised harmonics
//! C_l^m = (x + iy)-ladder on the diagonal, three-term in l at fixed m:
//!   C_l^l = √((2l−1)/(2l)) (x + iy) C_{l−1}^{l−1},
//!   C_l^m = [(2l−1) z C_{l−1}^m − √((l−1)² − m²) r² C_{l−2}^m] / √(l² − m²),
//! (the diagonal step omits the Condon–Shortley (−1)^l so that R_11 = +x), then
//! R_l0 = C_l^0, R_lm = √2 Re C_l^m, R_l,−m = √2 Im C_l^m for m > 0.
//! Why R_lm is harmonic matters for testing: by the mean-value property, every moment of a
//! spherically symmetric charge q centred at d is exactly q R_lm(d).
//!
//! # Precision
//! Every reduction (moments, norms, dot products) goes through `Grid::integrate`/`Grid::dot`
//! (Neumaier-compensated). Vector updates and the 49-term ghost sum are pointwise and need
//! no compensation. The moment pass stores all R_lm on the grid in a `[moment][point]`
//! buffer (49 × N f64, 46 MB at 49³) so each Q_lm is one `Grid::dot`.

use crate::grid::Grid;
use crate::stencil::COEFF;
use std::f64::consts::PI;

/// Highest multipole order kept in the boundary expansion.
const L_MAX: usize = 6;
/// Number of real solid harmonics with l ≤ L_MAX.
const N_MOMENTS: usize = (L_MAX + 1) * (L_MAX + 1);

/// Position of R_lm in a moment array: l² + l + m, m ∈ [−l, l].
#[inline]
fn lm_index(l: usize, m: i32) -> usize {
    debug_assert!(m.unsigned_abs() as usize <= l);
    l * l + (m + l as i32) as usize
}

/// Real regular solid harmonics R_lm(r) for l ≤ L_MAX (Racah normalisation, see module doc),
/// written to `out[lm_index(l, m)]`.
fn solid_harmonics(r: [f64; 3], out: &mut [f64; N_MOMENTS]) {
    let [x, y, z] = r;
    let r2 = x * x + y * y + z * z;
    // complex C_l^m for m ≥ 0 as (re, im); row l, column m
    let mut c = [[(0.0f64, 0.0f64); L_MAX + 1]; L_MAX + 1];
    c[0][0] = (1.0, 0.0);
    for l in 1..=L_MAX {
        let lf = l as f64;
        // diagonal ladder
        let f = ((2.0 * lf - 1.0) / (2.0 * lf)).sqrt();
        let (pr, pi) = c[l - 1][l - 1];
        c[l][l] = (f * (x * pr - y * pi), f * (x * pi + y * pr));
        // three-term recursion in l at fixed m < l
        for m in 0..l {
            let mf = m as f64;
            let a = (2.0 * lf - 1.0) * z;
            let (r1, i1) = c[l - 1][m];
            let (r0, i0) = if l >= 2 && m <= l - 2 { c[l - 2][m] } else { (0.0, 0.0) };
            let b = (((lf - 1.0) * (lf - 1.0) - mf * mf).max(0.0)).sqrt() * r2;
            let d = (lf * lf - mf * mf).sqrt();
            c[l][m] = ((a * r1 - b * r0) / d, (a * i1 - b * i0) / d);
        }
    }
    let s2 = 2f64.sqrt();
    for l in 0..=L_MAX {
        out[lm_index(l, 0)] = c[l][0].0;
        for m in 1..=l {
            out[lm_index(l, m as i32)] = s2 * c[l][m].0;
            out[lm_index(l, -(m as i32))] = s2 * c[l][m].1;
        }
    }
}

/// Jacobi-preconditioned conjugate-gradient Poisson solver with preallocated work vectors.
pub struct PoissonSolver {
    r: Vec<f64>, z: Vec<f64>, p: Vec<f64>, ap: Vec<f64>, b: Vec<f64>,
    /// R_lm sampled on the grid, layout `[moment][point]` (length N_MOMENTS × grid.len()).
    harm: Vec<f64>,
    /// The grid `harm` and the work vectors were built for; `solve` must be called with an equal grid.
    grid: Grid,
}

#[derive(Debug, Clone, Copy)]
pub struct PoissonStats { pub iterations: usize, pub residual: f64 }

impl PoissonSolver {
    pub fn new(grid: &Grid) -> PoissonSolver {
        let n = grid.len();
        let mut harm = vec![0.0; N_MOMENTS * n];
        let mut h = [0.0; N_MOMENTS];
        for i in 0..n {
            solid_harmonics(grid.point(i), &mut h);
            for (lm, value) in h.iter().enumerate() { harm[lm * n + i] = *value; }
        }
        PoissonSolver {
            r: vec![0.0; n], z: vec![0.0; n], p: vec![0.0; n], ap: vec![0.0; n], b: vec![0.0; n],
            harm, grid: grid.clone(),
        }
    }

    /// The sampled harmonics and work vectors are only valid for the grid given to `new`.
    fn check_grid(&self, grid: &Grid) {
        assert!(
            *grid == self.grid,
            "PoissonSolver was built for a different grid: built for {:?}, called with {:?}", self.grid, grid
        );
    }

    /// Q_lm = ∫ ρ R_lm for all l ≤ L_MAX, each a compensated `Grid::dot`.
    fn multipole_moments(&self, grid: &Grid, rho: &[f64]) -> [f64; N_MOMENTS] {
        self.check_grid(grid);
        let n = grid.len();
        debug_assert_eq!(rho.len(), n);
        let mut q = [0.0; N_MOMENTS];
        for lm in 0..N_MOMENTS { q[lm] = grid.dot(rho, &self.harm[lm * n..(lm + 1) * n]); }
        q
    }

    /// y = A x = −L_int x  (ghost cells treated as zero)
    fn apply_a(grid: &Grid, x: &[f64], y: &mut [f64]) {
        crate::stencil::laplacian(grid, x, y);
        for v in y.iter_mut() { *v = -*v; }
    }

    /// Adds L_bc (contribution of ghost values) to `out`. Ghost value at point r is bc(r).
    fn add_boundary_term(grid: &Grid, bc: &dyn Fn([f64; 3]) -> f64, out: &mut [f64]) {
        let [n0, n1, n2] = grid.n;
        let inv_h2 = 1.0 / (grid.h * grid.h);
        for k in 0..n2 { for j in 0..n1 { for i in 0..n0 {
            let pos = [i as isize, j as isize, k as isize];
            let mut acc = 0.0;
            for axis in 0..3 {
                for m in 1..=4isize {
                    for sign in [-1isize, 1] {
                        let mut q = pos; q[axis] += sign * m;
                        if q[axis] < 0 || q[axis] >= grid.n[axis] as isize {
                            let mut r = grid.point(grid.idx(i, j, k));
                            r[axis] += (sign * m) as f64 * grid.h;
                            acc += COEFF[m as usize] * bc(r);
                        }
                    }
                }
            }
            out[grid.idx(i, j, k)] += acc * inv_h2;
        }}}
    }

    /// Solve ∇²V = −4π ρ on the grid with V(r) outside the box given by the real solid-harmonic
    /// multipole expansion of ρ to l_max = 6: V_bc(r) = Σ_{l≤6,m} Q_lm R_lm(r)/|r|^{2l+1}.
    /// `v` is both the initial guess and the output. Converges to ‖r‖/‖b‖ < tol.
    pub fn solve(&mut self, grid: &Grid, rho: &[f64], v: &mut [f64], tol: f64, max_iter: usize) -> PoissonStats {
        let q = self.multipole_moments(grid, rho);
        let bc = move |r: [f64; 3]| {
            let d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let mut h = [0.0; N_MOMENTS];
            solid_harmonics(r, &mut h);
            // Σ_l (Σ_m Q_lm R_lm(r)) / |r|^{2l+1}; terms fall off geometrically, plain sum is fine
            let mut acc = 0.0;
            let mut inv_pow = 1.0 / d2.sqrt(); // |r|^{-(2l+1)}
            for l in 0..=L_MAX {
                let mut s = 0.0;
                for m in -(l as i32)..=(l as i32) { let i = lm_index(l, m); s += q[i] * h[i]; }
                acc += s * inv_pow;
                inv_pow /= d2;
            }
            acc
        };
        self.solve_with_boundary(grid, rho, v, &bc, tol, max_iter)
    }

    /// Solve −L_int V = 4πρ + L_bc with the ghost values given by `bc`.
    fn solve_with_boundary(&mut self, grid: &Grid, rho: &[f64], v: &mut [f64], bc: &dyn Fn([f64; 3]) -> f64,
                           tol: f64, max_iter: usize) -> PoissonStats {
        self.check_grid(grid);
        let n = grid.len();
        debug_assert_eq!(rho.len(), n);
        debug_assert_eq!(v.len(), n);
        // b = 4πρ + L_bc
        for i in 0..n { self.b[i] = 4.0 * PI * rho[i]; }
        Self::add_boundary_term(grid, bc, &mut self.b);
        // r = b − A v
        Self::apply_a(grid, v, &mut self.ap);
        for i in 0..n { self.r[i] = self.b[i] - self.ap[i]; }
        let inv_diag = -(grid.h * grid.h) / (3.0 * COEFF[0]);
        let b_norm = grid.dot(&self.b, &self.b).sqrt().max(1e-300);
        for i in 0..n { self.z[i] = inv_diag * self.r[i]; }
        self.p.copy_from_slice(&self.z);
        let mut rz = grid.dot(&self.r, &self.z);
        let mut iterations = 0;
        let mut residual = grid.dot(&self.r, &self.r).sqrt() / b_norm;
        while residual > tol && iterations < max_iter {
            Self::apply_a(grid, &self.p, &mut self.ap);
            let pap = grid.dot(&self.p, &self.ap);
            debug_assert!(pap > 0.0, "A lost positive-definiteness: p·Ap = {pap}");
            let alpha = rz / pap;
            for i in 0..n { v[i] += alpha * self.p[i]; self.r[i] -= alpha * self.ap[i]; }
            for i in 0..n { self.z[i] = inv_diag * self.r[i]; }
            let rz_new = grid.dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..n { self.p[i] = self.z[i] + beta * self.p[i]; }
            residual = grid.dot(&self.r, &self.r).sqrt() / b_norm;
            iterations += 1;
        }
        PoissonStats { iterations, residual }
    }
}

/// Hartree energy ½ ∫ ρ V (Hartree).
pub fn hartree_energy(grid: &Grid, rho: &[f64], v: &[f64]) -> f64 { 0.5 * grid.dot(rho, v) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// Normalised Gaussian charge q, width σ: ρ = q (2πσ²)^{-3/2} exp(−r²/2σ²); V = q erf(r/(σ√2))/r.
    fn gaussian_case(n: usize, h: f64, q: f64, sigma: f64, centre: [f64; 3]) -> (Grid, Vec<f64>, Vec<f64>) {
        let g = Grid::cube(n, h);
        let norm = q / (2.0 * PI * sigma * sigma).powf(1.5);
        let rho = g.sample(|[x, y, z]| {
            let r2 = (x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2);
            norm * (-r2 / (2.0 * sigma * sigma)).exp()
        });
        let exact = g.sample(|[x, y, z]| {
            let r = ((x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2)).sqrt();
            if r < 1e-12 { q * (2.0 / PI).sqrt() / sigma } else { q * libm::erf(r / (sigma * 2f64.sqrt())) / r }
        });
        (g, rho, exact)
    }

    fn max_err(n: usize, h: f64, centre: [f64; 3]) -> f64 {
        let (g, rho, exact) = gaussian_case(n, h, 1.0, 0.5, centre);
        let mut v = vec![0.0; g.len()];
        let stats = PoissonSolver::new(&g).solve(&g, &rho, &mut v, 1e-10, 2000);
        assert!(stats.residual < 1e-10, "did not converge: {:?}", stats.residual);
        v.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn gaussian_potential_centred() {
        assert!(max_err(49, 0.25, [0.0; 3]) < 1e-4);
    }

    #[test]
    fn gaussian_potential_off_centre_uses_dipole_boundary() {
        assert!(max_err(49, 0.25, [1.3, -0.7, 0.4]) < 1e-4);
    }

    #[test]
    fn converges_with_h() {
        let coarse = max_err(25, 0.5, [0.0; 3]);
        let fine = max_err(49, 0.25, [0.0; 3]);
        assert!(coarse / fine > 30.0, "ratio = {}", coarse / fine);
    }

    #[test]
    fn hartree_energy_of_gaussian() {
        // E_H = q² / (2 σ √π)
        let (g, rho, _) = gaussian_case(49, 0.25, 1.0, 0.5, [0.0; 3]);
        let mut v = vec![0.0; g.len()];
        PoissonSolver::new(&g).solve(&g, &rho, &mut v, 1e-10, 2000);
        assert_relative_eq!(hartree_energy(&g, &rho, &v), 1.0 / (2.0 * 0.5 * PI.sqrt()), epsilon = 1e-4);
    }

    // ---- boundary expansion: real solid harmonics and multipole moments ----

    /// Closed forms, Racah normalisation (R_00 = 1, R_1m = z, x, y; R_2m = (3z²−r²)/2, √3 xz, √3 yz, √3(x²−y²)/2, √3 xy).
    #[test]
    fn solid_harmonics_match_closed_forms_up_to_l2() {
        let [x, y, z] = [0.7, -0.4, 1.1];
        let r2 = x * x + y * y + z * z;
        let mut h = [0.0; N_MOMENTS];
        solid_harmonics([x, y, z], &mut h);
        let s3 = 3f64.sqrt();
        let expect = [
            (0, 0, 1.0), (1, 0, z), (1, 1, x), (1, -1, y),
            (2, 0, (3.0 * z * z - r2) / 2.0), (2, 1, s3 * x * z), (2, -1, s3 * y * z),
            (2, 2, s3 * (x * x - y * y) / 2.0), (2, -2, s3 * x * y),
        ];
        for (l, m, v) in expect {
            assert_relative_eq!(h[lm_index(l, m)], v, epsilon = 1e-14, max_relative = 1e-14);
        }
    }

    /// Addition theorem: for |a| < |b|, Σ_{l≤L} Σ_m R_lm(a) R_lm(b)/|b|^{2l+1} → 1/|a−b|
    /// with Legendre tail Σ_{l>L} |a|^l/|b|^{l+1}. Pins the normalisation of every (l, m) ≤ 6.
    #[test]
    fn solid_harmonics_satisfy_addition_theorem() {
        let a = [0.3, -0.2, 0.5];
        let b = [2.0, 1.5, -1.0];
        let (mut ha, mut hb) = ([0.0; N_MOMENTS], [0.0; N_MOMENTS]);
        solid_harmonics(a, &mut ha);
        solid_harmonics(b, &mut hb);
        let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let exact = 1.0 / ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        let mut partial = 0.0;
        let mut prev_err = f64::INFINITY;
        for l in 0..=L_MAX {
            for m in -(l as i32)..=(l as i32) {
                let i = lm_index(l, m);
                partial += ha[i] * hb[i] / nb.powi(2 * l as i32 + 1);
            }
            let err = (partial - exact).abs();
            let tail = (na / nb).powi(l as i32 + 1) / (nb - na); // Σ_{l'>l} |a|^l'/|b|^{l'+1}
            assert!(err <= tail, "l = {l}: err {err:.3e} exceeds Legendre tail {tail:.3e}");
            assert!(err < prev_err, "l = {l}: error did not decrease ({prev_err:.3e} → {err:.3e})");
            prev_err = err;
        }
    }

    /// Moments of a centred Gaussian: only Q_00 = q; moments of an off-centre one at d are
    /// Q_00 = q, Q_10 = q d_z, Q_11 = q d_x, Q_1,-1 = q d_y, i.e. the charge and dipole p = ∫ρ r.
    #[test]
    fn multipole_moments_reduce_to_charge_and_dipole() {
        let (g, rho, _) = gaussian_case(49, 0.25, 2.0, 0.5, [0.0; 3]);
        let solver = PoissonSolver::new(&g);
        let q = solver.multipole_moments(&g, &rho);
        assert_relative_eq!(q[lm_index(0, 0)], 2.0, epsilon = 1e-10);
        for i in 1..N_MOMENTS { assert!(q[i].abs() < 1e-10, "centred Gaussian has Q[{i}] = {:.3e}", q[i]); }

        let d = [1.3, -0.7, 0.4];
        let (g, rho, _) = gaussian_case(49, 0.25, 2.0, 0.5, d);
        let q = solver.multipole_moments(&g, &rho);
        let p: Vec<f64> = (0..3).map(|axis| {
            let w: Vec<f64> = (0..g.len()).map(|i| rho[i] * g.point(i)[axis]).collect();
            g.integrate(&w)
        }).collect();
        assert_relative_eq!(q[lm_index(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(q[lm_index(1, 1)], p[0], epsilon = 1e-12);
        assert_relative_eq!(q[lm_index(1, -1)], p[1], epsilon = 1e-12);
        assert_relative_eq!(q[lm_index(1, 0)], p[2], epsilon = 1e-12);
        assert_relative_eq!(p[0], 2.0 * d[0], epsilon = 1e-10);
    }

    /// R_lm is harmonic, so by the mean-value property every moment of a spherically symmetric
    /// charge q about d equals q R_lm(d) — for all l, not just l ≤ 1.
    #[test]
    fn multipole_moments_of_spherical_charge_are_harmonics_at_its_centre() {
        let d = [1.3, -0.7, 0.4];
        let (g, rho, _) = gaussian_case(49, 0.25, 1.0, 0.5, d);
        let q = PoissonSolver::new(&g).multipole_moments(&g, &rho);
        let mut h = [0.0; N_MOMENTS];
        solid_harmonics(d, &mut h);
        for i in 0..N_MOMENTS {
            assert_relative_eq!(q[i], h[i], epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    /// With exact ghost values the off-centre case has only discretisation error (observed 3.48e-5,
    /// same as the centred case). Separates the CG core from the multipole truncation.
    #[test]
    fn exact_ghost_values_leave_only_discretisation_error() {
        let c = [1.3, -0.7, 0.4];
        let (g, rho, exact) = gaussian_case(49, 0.25, 1.0, 0.5, c);
        let exact_bc = |r: [f64; 3]| {
            let d = ((r[0] - c[0]).powi(2) + (r[1] - c[1]).powi(2) + (r[2] - c[2]).powi(2)).sqrt();
            libm::erf(d / (0.5 * 2f64.sqrt())) / d
        };
        let mut v = vec![0.0; g.len()];
        let stats = PoissonSolver::new(&g).solve_with_boundary(&g, &rho, &mut v, &exact_bc, 1e-10, 2000);
        assert!(stats.residual < 1e-10);
        let err = v.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 5e-5, "max error with exact ghosts = {err:.3e}");
    }

    /// The harmonics are sampled on the grid given to `new`; a same-size grid with a different
    /// spacing or origin must be rejected loudly, not produce wrong moments.
    #[test]
    #[should_panic(expected = "PoissonSolver was built for a different grid")]
    fn solve_on_a_different_grid_panics() {
        let built_for = Grid::cube(9, 0.5);
        let other = Grid::cube(9, 0.25);
        let rho = vec![0.0; other.len()];
        let mut v = vec![0.0; other.len()];
        PoissonSolver::new(&built_for).solve(&other, &rho, &mut v, 1e-10, 10);
    }

    #[test]
    fn warm_start_takes_fewer_iterations() {
        let (g, rho, _) = gaussian_case(33, 0.25, 1.0, 0.5, [0.0; 3]);
        let mut solver = PoissonSolver::new(&g);
        let mut v = vec![0.0; g.len()];
        let cold = solver.solve(&g, &rho, &mut v, 1e-10, 2000);
        let warm = solver.solve(&g, &rho, &mut v, 1e-10, 2000);
        assert!(warm.iterations < cold.iterations / 4, "cold {} warm {}", cold.iterations, warm.iterations);
    }

}
