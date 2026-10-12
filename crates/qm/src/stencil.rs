//! 8th-order central finite-difference Laplacian on a uniform grid, Dirichlet (f = 0) outside the box.
//!
//! One axis: d²f/dx² ≈ (1/h²) Σ_{m=-4}^{4} c_{|m|} f_{i+m} with
//!   c = [−205/72, 8/5, −1/5, 8/315, −1/560]   (Fornberg 1988).
//!
//! The stencil is the unique symmetric 9-point rule that satisfies the moment conditions
//!   c₀ + 2 Σ_{m=1}^{4} c_m = 0            (annihilates constants),
//!   2 Σ m² c_m = 2                         (reproduces f'' exactly),
//!   Σ m⁴ c_m = Σ m⁶ c_m = Σ m⁸ c_m = 0     (kills the h², h⁴, h⁶ error terms),
//! so the leading truncation error is from the 10th moment:
//!   (2 Σ m¹⁰ c_m / 10!) h⁸ f⁽¹⁰⁾ = −(1152/3628800) h⁸ f⁽¹⁰⁾ ≈ −3.2×10⁻⁴ h⁸ f⁽¹⁰⁾,
//! i.e. the error scales as h⁸.
//!
//! ∇² = Σ over the three axes, so the centre coefficient c₀ appears three times.
//! Dirichlet boundary: any f_{i+m} that falls outside the box is taken as 0, i.e. that term is dropped.
//! Kinetic energy in atomic units (ħ = mₑ = 1): T f = −½ ∇² f.

use crate::grid::Grid;

/// Central 8th-order second-derivative coefficients c₀..c₄ (c_{-m} = c_m).
pub const COEFF: [f64; 5] = [-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0];

/// out = ∇²f with f = 0 outside the box (Dirichlet). out.len() == f.len() == grid.len().
pub fn laplacian(grid: &Grid, f: &[f64], out: &mut [f64]) {
    debug_assert_eq!(f.len(), grid.len());
    debug_assert_eq!(out.len(), grid.len());
    let [n0, n1, n2] = grid.n;
    let inv_h2 = 1.0 / (grid.h * grid.h);
    let strides = [1usize, n0, n0 * n1];
    for k in 0..n2 {
        for j in 0..n1 {
            for i in 0..n0 {
                let id = grid.idx(i, j, k);
                let mut acc = 3.0 * COEFF[0] * f[id];
                let pos = [i, j, k];
                for axis in 0..3 {
                    for m in 1..=4usize {
                        let c = COEFF[m];
                        if pos[axis] + m < grid.n[axis] { acc += c * f[id + m * strides[axis]]; }
                        if pos[axis] >= m { acc += c * f[id - m * strides[axis]]; }
                    }
                }
                out[id] = acc * inv_h2;
            }
        }
    }
}

/// out = -½ ∇² f  (kinetic-energy operator in atomic units)
pub fn kinetic(grid: &Grid, f: &[f64], out: &mut [f64]) {
    laplacian(grid, f, out);
    for v in out.iter_mut() { *v *= -0.5; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    fn max_interior_err(n: usize, h: f64) -> f64 {
        // f = exp(-r²/2)  ⇒  ∇²f = (r² − 3) f
        let g = Grid::cube(n, h);
        let f = g.sample(|[x, y, z]| (-(x * x + y * y + z * z) / 2.0).exp());
        let exact = g.sample(|[x, y, z]| { let r2 = x * x + y * y + z * z; (r2 - 3.0) * (-r2 / 2.0).exp() });
        let mut out = vec![0.0; g.len()];
        laplacian(&g, &f, &mut out);
        // skip 4 cells at each face where the Dirichlet ghost (0) differs from the true Gaussian tail
        let mut err: f64 = 0.0;
        for k in 4..n - 4 { for j in 4..n - 4 { for i in 4..n - 4 {
            let id = g.idx(i, j, k);
            err = err.max((out[id] - exact[id]).abs());
        }}}
        err
    }

    #[test]
    fn coefficients_sum_to_zero() {
        // A constant field has zero Laplacian: c0 + 2(c1+c2+c3+c4) = 0
        let s = COEFF[0] + 2.0 * (COEFF[1] + COEFF[2] + COEFF[3] + COEFF[4]);
        assert!(s.abs() < 1e-15, "sum = {s}");
    }

    #[test]
    fn laplacian_of_gaussian() {
        assert!(max_interior_err(49, 0.25) < 1e-4);
    }

    #[test]
    fn converges_at_eighth_order() {
        let coarse = max_interior_err(25, 0.5);
        let fine = max_interior_err(49, 0.25);
        // 8th order predicts 256×; demand at least 64× to allow for the error constant
        assert!(coarse / fine > 64.0, "ratio = {}", coarse / fine);
    }

    #[test]
    fn kinetic_is_minus_half_laplacian() {
        let g = Grid::cube(9, 0.5);
        let f = g.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let (mut l, mut t) = (vec![0.0; g.len()], vec![0.0; g.len()]);
        laplacian(&g, &f, &mut l);
        kinetic(&g, &f, &mut t);
        for i in 0..g.len() { assert!((t[i] + 0.5 * l[i]).abs() < 1e-15); }
    }
}
