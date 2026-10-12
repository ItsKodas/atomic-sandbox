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

    /// Neumaier-compensated sum times the volume element. The only place grid data is summed.
    pub fn integrate(&self, f: &[f64]) -> f64 {
        debug_assert_eq!(f.len(), self.len());
        let (mut sum, mut comp) = (0.0f64, 0.0f64);
        for &x in f {
            let t = sum + x;
            if sum.abs() >= x.abs() { comp += (sum - t) + x; } else { comp += (x - t) + sum; }
            sum = t;
        }
        (sum + comp) * self.dv()
    }

    pub fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let (mut sum, mut comp) = (0.0f64, 0.0f64);
        for (&p, &q) in a.iter().zip(b) {
            let x = p * q;
            let t = sum + x;
            if sum.abs() >= x.abs() { comp += (sum - t) + x; } else { comp += (x - t) + sum; }
            sum = t;
        }
        (sum + comp) * self.dv()
    }

    pub fn sample(&self, f: impl Fn([f64; 3]) -> f64) -> Vec<f64> {
        (0..self.len()).map(|i| f(self.point(i))).collect()
    }
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

    #[test]
    fn dot_matches_integrate() {
        let g = Grid::cube(9, 0.5);
        let a = g.sample(|[x, _, _]| x);
        let b = g.sample(|[x, _, _]| x);
        let prod: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p * q).collect();
        assert_relative_eq!(g.dot(&a, &b), g.integrate(&prod), epsilon = 1e-14);
    }
}
