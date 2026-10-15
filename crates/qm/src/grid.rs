//! Uniform real-space grid. Layout: idx = i + n0*(j + n1*k). Lengths in Bohr.

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub n: [usize; 3],
    pub h: f64,
    pub origin: [f64; 3],
}

impl Grid {
    pub fn cube(n: usize, h: f64) -> Grid {
        let half = (n as f64 - 1.0) * h / 2.0;
        Grid { n: [n; 3], h, origin: [-half; 3] }
    }
    pub fn len(&self) -> usize { self.n[0] * self.n[1] * self.n[2] }
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize { i + self.n[0] * (j + self.n[1] * k) }
    pub fn point(&self, idx: usize) -> [f64; 3] {
        let i = idx % self.n[0];
        let j = (idx / self.n[0]) % self.n[1];
        let k = idx / (self.n[0] * self.n[1]);
        [self.origin[0] + i as f64 * self.h,
         self.origin[1] + j as f64 * self.h,
         self.origin[2] + k as f64 * self.h]
    }
    pub fn dv(&self) -> f64 { self.h * self.h * self.h }

    /// ∫ f d³r: Neumaier-compensated sum times the volume element (via `neumaier`).
    pub fn integrate(&self, f: &[f64]) -> f64 {
        debug_assert_eq!(f.len(), self.len());
        neumaier(f.iter().copied()) * self.dv()
    }

    /// ∫ a b d³r: Neumaier-compensated sum of pointwise products times the volume element (via `neumaier`).
    pub fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        neumaier(a.iter().zip(b).map(|(p, q)| p * q)) * self.dv()
    }

    pub fn sample(&self, mut f: impl FnMut([f64; 3]) -> f64) -> Vec<f64> {
        (0..self.len()).map(|i| f(self.point(i))).collect()
    }

    /// The trilinear cell of `p` (Bohr): the lower corner (i₀, j₀, k₀) = ⌊x⌋ and the fractions
    /// t = x − i₀ ∈ [0, 1) per axis, x = (p − origin)/h in grid-index coordinates. On an upper
    /// face i₀ = n − 1 and t = 0: the corner's +1 neighbours lie beyond the grid and carry
    /// weight 0 (they read as 0 — [`trilinear`](Self::trilinear) skips them, the GPU's padded
    /// field has zero ghosts there). t is never 1, so a node is read with t = 0 on every axis,
    /// which returns the stored value exactly however the blend a·(1 − t) + b·t is rounded (a
    /// GPU compiler may evaluate it as a + t·(b − a), inexact at t = 1). `None` outside
    /// [origin, origin + (n−1)h]³ (a NaN coordinate included). The one definition of the cell and
    /// weights: [`trilinear`](Self::trilinear) and the GPU sampler
    /// (`qm_gpu::GpuSolver::sample_density`) both use it.
    pub fn trilinear_cell(&self, p: [f64; 3]) -> Option<([usize; 3], [f64; 3])> {
        let mut cell = [0usize; 3];
        let mut t = [0.0f64; 3];
        for a in 0..3 {
            let x = (p[a] - self.origin[a]) / self.h;
            if !(x >= 0.0 && x <= (self.n[a] - 1) as f64) {
                return None;
            }
            let i0 = x.floor();
            cell[a] = i0 as usize;
            t[a] = x - i0;
        }
        Some((cell, t))
    }

    /// ρ at `p` (Bohr) by trilinear interpolation on this grid's points (x fastest); 0 outside
    /// [origin, origin + (n−1)h]³. Cell and fractions from [`trilinear_cell`](Self::trilinear_cell)
    /// (a neighbour beyond an upper face reads as 0 with weight 0); the eight values are blended
    /// as a·(1 − t) + b·t along x, then y, then z — the GPU sampler's order — and a node returns
    /// its stored value bitwise.
    pub fn trilinear(&self, f: &[f64], p: [f64; 3]) -> f64 {
        debug_assert_eq!(f.len(), self.len());
        let Some(([i, j, k], [tx, ty, tz])) = self.trilinear_cell(p) else { return 0.0 };
        let [n0, n1, n2] = self.n;
        let at = |di: usize, dj: usize, dk: usize| {
            let (a, b, c) = (i + di, j + dj, k + dk);
            if a < n0 && b < n1 && c < n2 { f[self.idx(a, b, c)] } else { 0.0 }
        };
        let lerp = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;
        let x = |dj, dk| lerp(at(0, dj, dk), at(1, dj, dk), tx);
        let y = |dk| lerp(x(0, dk), x(1, dk), ty);
        lerp(y(0), y(1), tz)
    }
}

/// Neumaier-compensated summation. The only place grid data is summed;
/// `Grid::integrate` and `Grid::dot` both go through it.
fn neumaier(iter: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for x in iter {
        let t = sum + x;
        if sum.abs() >= x.abs() { comp += (sum - t) + x; } else { comp += (x - t) + sum; }
        sum = t;
    }
    sum + comp
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn cube_is_centred() {
        let g = Grid::cube(5, 0.5);
        assert_eq!(g.len(), 125);
        let c = g.point(g.idx(2, 2, 2));
        assert_relative_eq!(c[0], 0.0); assert_relative_eq!(c[1], 0.0); assert_relative_eq!(c[2], 0.0);
        let p = g.point(g.idx(0, 0, 0));
        assert_relative_eq!(p[0], -1.0);
    }

    #[test]
    fn integrates_gaussian() {
        // ∫ exp(-r²) d³r = π^{3/2}; uniform grids are spectrally accurate for smooth decaying f.
        let g = Grid::cube(49, 0.25); // 12 Bohr box
        let f = g.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        assert_relative_eq!(g.integrate(&f), PI.powf(1.5), epsilon = 1e-9);
    }

    #[test]
    fn integrate_is_compensated() {
        let g = Grid { n: [3, 1, 1], h: 1.0, origin: [0.0; 3] };
        assert_eq!(g.integrate(&[1e16, 1.0, -1e16]), 1.0);
    }

    /// A deterministic xorshift64* stream of uniforms in [0, 1) (no `rand` in `qm`).
    fn uniforms(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Trilinear interpolation is exact for a field linear in x, y, z; it returns the stored value
    /// at a grid point (bitwise, the upper faces included) and 0 outside [origin, origin + (n−1)h]³.
    #[test]
    fn trilinear_reproduces_a_linear_field_exactly_and_is_zero_outside() {
        let g = Grid::cube(9, 0.5); // [-2, 2]³
        let lin = |[x, y, z]: [f64; 3]| 1.0 + 2.0 * x - y + 3.0 * z;
        let f = g.sample(lin);
        let mut u = uniforms(0x5eed_0001);
        let mut worst = 0.0f64;
        for _ in 0..50 {
            let p = [-2.0 + 4.0 * u(), -2.0 + 4.0 * u(), -2.0 + 4.0 * u()];
            let d = (g.trilinear(&f, p) - lin(p)).abs();
            worst = worst.max(d);
            assert!(d <= 1e-13, "p = {p:?}: trilinear {} vs f {} (|Δ| {d:.2e})", g.trilinear(&f, p), lin(p));
        }
        println!("linear field, 50 random points: max |Δ| = {worst:.2e}");
        // grid points, the 8 corners (upper faces: t = 0 against the zero neighbours beyond) and a face centre: the stored value, bitwise
        for idx in [0, g.idx(3, 5, 7), g.idx(8, 0, 0), g.idx(0, 8, 0), g.idx(0, 0, 8), g.idx(8, 8, 0), g.idx(8, 0, 8), g.idx(0, 8, 8),
                    g.idx(8, 8, 8), g.idx(8, 4, 4), g.idx(4, 4, 4)] {
            assert_eq!(g.trilinear(&f, g.point(idx)).to_bits(), f[idx].to_bits(), "grid point {idx} at {:?}", g.point(idx));
        }
        // outside, by a little and by a lot, on each axis and side
        for p in [[2.0 + 1e-12, 0.0, 0.0], [-2.0 - 1e-12, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, -7.0, 0.0], [0.0, 0.0, 2.000001],
                  [0.0, 0.0, -2.1], [3.0, 3.0, 3.0], [f64::NAN, 0.0, 0.0]] {
            assert_eq!(g.trilinear(&f, p), 0.0, "outside point {p:?}");
        }
        // a non-cubic grid with an offset origin: still exact, x fastest
        let g2 = Grid { n: [5, 7, 4], h: 0.3, origin: [0.1, -0.4, 1.0] };
        let f2 = g2.sample(lin);
        for _ in 0..20 {
            let p = [0.1 + 1.2 * u(), -0.4 + 1.8 * u(), 1.0 + 0.9 * u()];
            assert!((g2.trilinear(&f2, p) - lin(p)).abs() <= 1e-13, "non-cubic p = {p:?}");
        }
        assert_eq!(g2.trilinear(&f2, [1.31, 0.0, 1.5]), 0.0, "beyond x = 0.1 + 4·0.3 = 1.3");
    }

    /// The interpolation error of a smooth field is O(h²) with the leading term the spec's
    /// cell-midpoint bias (§3.1): f = exp(−r²) on 33³, 65³ and 129³ over the same 8 Bohr box
    /// (h = 0.25, 0.125, 0.0625). Measured as the max error over **every cell centre** with
    /// |x|, |y|, |z| ≤ 3 — at t = ½ on all axes the odd h³ term vanishes and the in-cell position
    /// is the same on every grid; it falls by ≥ 3.6× per halving (measured 3.71, 3.92; 4×
    /// asymptotically), and at each grid it is (h²/8)∇²f to ≤ 5 % of the max (measured 2.3 %,
    /// 0.6 %, 0.1 %).
    ///
    /// The leading term: linear interpolation between x₀ and x₀ + h at x = x₀ + th, expanded
    /// about x, errs by I f − f = (h²/2)·t(1−t)·f″ + (h³/6)·t(1−t)(1−2t)·f‴ + O(h⁴); at t = ½ that
    /// is (h²/8)·f″ with no h³ term. Trilinear is the tensor product of three such operators,
    /// I = I_x I_y I_z, and I_a = 1 + E_a with E_a = O(h²), so I − 1 = E_x + E_y + E_z + O(h⁴)
    /// (the cross terms E_a E_b are products of two O(h²) errors): at a cell centre the error is
    /// (h²/8)(∂²_x + ∂²_y + ∂²_z) f = (h²/8)∇²f + O(h⁴) — for f = exp(−r²), ∇²f = (4r² − 6)e^{−r²}.
    ///
    /// Why not the brief's 17³/33³/65³ and 200 random points: the error at a fixed point depends
    /// on its fractional position in the cell (factor t(1−t)), which changes from grid to grid,
    /// so a max over 200 points swings from 2.7× to 8× per halving with the seed; and h = 0.5 is
    /// pre-asymptotic for exp(−r²) — the max over all cell centres falls only 2.96× from h = 0.5
    /// to 0.25 (printed below, not asserted), so no correct interpolator meets 3.6× there.
    #[test]
    fn trilinear_of_a_gaussian_converges_second_order() {
        let gauss = |[x, y, z]: [f64; 3]| (-(x * x + y * y + z * z)).exp();
        let lap = |[x, y, z]: [f64; 3]| { let r2 = x * x + y * y + z * z; (4.0 * r2 - 6.0) * (-r2).exp() };
        let measure = |n: usize| -> (f64, f64) {
            let g = Grid::cube(n, 8.0 / (n - 1) as f64);
            let f = g.sample(gauss);
            let centres: Vec<f64> = (0..n - 1).map(|i| g.origin[0] + (i as f64 + 0.5) * g.h).filter(|c| c.abs() <= 3.0).collect();
            let (mut worst, mut beyond_leading) = (0.0f64, 0.0f64);
            for &x in &centres { for &y in &centres { for &z in &centres {
                let p = [x, y, z];
                let err = g.trilinear(&f, p) - gauss(p);
                worst = worst.max(err.abs());
                beyond_leading = beyond_leading.max((err - g.h * g.h / 8.0 * lap(p)).abs());
            }}}
            (worst, beyond_leading)
        };
        let coarse = measure(17).0;
        let m: Vec<(f64, f64)> = [33usize, 65, 129].into_iter().map(measure).collect();
        let ratios = [m[0].0 / m[1].0, m[1].0 / m[2].0];
        println!("exp(−r²), max error over the cell centres in [-3, 3]³: h 0.5 {coarse:.3e} (pre-asymptotic, ratio to h 0.25 {:.2}), h 0.25 {:.3e}, h 0.125 {:.3e}, h 0.0625 {:.3e}; ratios {:.2} {:.2}",
                 coarse / m[0].0, m[0].0, m[1].0, m[2].0, ratios[0], ratios[1]);
        for r in ratios { assert!(r >= 3.6, "error ratio {r:.2} per halving of h ({m:?})"); }
        for (n, (worst, beyond)) in [33, 65, 129].into_iter().zip(&m) {
            println!("  {n}³: max |err − (h²/8)∇²f| = {beyond:.2e} ({:.1} % of the max error)", 100.0 * beyond / worst);
            assert!(*beyond <= 0.05 * worst, "{n}³: the error is not the leading (h²/8)∇²f term: {beyond:.2e} of {worst:.2e}");
        }
    }

    #[test]
    fn dot_matches_integrate() {
        let g = Grid::cube(9, 0.5);
        let a = g.sample(|[x, _, _]| x);
        let b = g.sample(|[x, _, _]| x);
        let prod: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p * q).collect();
        assert_relative_eq!(g.dot(&a, &b), g.integrate(&prod), epsilon = 1e-14);
    }
}
