//! The universal logarithmic radial grid r_i = r0·e^{iδ} (spec §4.1; `LEDGER.r0/r_max/delta`).
//! Integrals over r become integrals over x = ln r with dr = r dx. `integrate` is the trapezoid in x:
//! for integrands f·r that vanish smoothly at both ends (every bound-state quantity here) the
//! Euler–Maclaurin corrections vanish and the rule is exponentially accurate. `cumulative` is
//! 4th order (cubic Lagrange through four neighbours). Both include the sphere inside r0 (`inner`):
//! f ≈ a·r^p there, with p read off the first two points, so ∫₀^{r0} f dr = f(r0)·r0/(p + 1). It is
//! negligible for density-like integrands (∝ r², ≈ ρ(0)r0³ ≈ 1e-12 for Ne) but not for Coulomb-weighted
//! ones (∝ r): ∫₀^{r0} 4πr²ρ·Z/r dr = 2πZρ(0)r0² ≈ 4e-6 Ha for Ne, growing as Z⁴ — E_en and the
//! potential integral in T_s each miss it otherwise (they cancel in the total, not in the components).
use super::ledger::LEDGER;
use crate::grid::neumaier;

#[derive(Clone, Debug, PartialEq)]
pub struct LogGrid { pub r: Vec<f64>, pub x0: f64, pub delta: f64 }

impl LogGrid {
    pub fn new(r0: f64, r_max: f64, delta: f64) -> LogGrid {
        let n = ((r_max / r0).ln() / delta).floor() as usize + 1;
        LogGrid { r: (0..n).map(|i| r0 * (i as f64 * delta).exp()).collect(), x0: r0.ln(), delta }
    }
    pub fn universal() -> LogGrid { LogGrid::new(LEDGER.r0, LEDGER.r_max, LEDGER.delta) }
    pub fn len(&self) -> usize { self.r.len() }
    pub fn is_empty(&self) -> bool { self.r.is_empty() }

    /// ∫₀^{r_max} f dr (trapezoid in x, Neumaier).
    pub fn integrate(&self, f: &[f64]) -> f64 {
        debug_assert_eq!(f.len(), self.len());
        let n = self.len();
        let sum = neumaier(f.iter().zip(&self.r).map(|(a, r)| a * r));
        self.inner(f) + (sum - 0.5 * (f[0] * self.r[0] + f[n - 1] * self.r[n - 1])) * self.delta
    }

    /// ∫₀^{r0} f dr for f ≈ a·r^p inside the first point, p = ln(f₁/f₀)/δ. Zero when f₀ = 0, when f
    /// changes sign between the first two points, or when p ≤ −1 (not integrable: the caller's
    /// integrand is wrong, and the grid's own start is all that can be returned).
    fn inner(&self, f: &[f64]) -> f64 {
        let (f0, f1) = (f[0], f[1]);
        if f0 == 0.0 || f0 * f1 <= 0.0 { return 0.0; }
        let p = (f1 / f0).ln() / self.delta;
        if p > -1.0 { f0 * self.r[0] / (p + 1.0) } else { 0.0 }
    }

    /// c_i = ∫₀^{r_i} f dr for every i (4th order in x).
    pub fn cumulative(&self, f: &[f64]) -> Vec<f64> {
        let n = self.len();
        let g: Vec<f64> = f.iter().zip(&self.r).map(|(a, r)| a * r).collect();
        let w = self.delta / 24.0;
        let mut c = vec![0.0; n];
        c[0] = self.inner(f);
        c[1] = c[0] + w * (9.0 * g[0] + 19.0 * g[1] - 5.0 * g[2] + g[3]);
        for i in 1..n - 2 {
            c[i + 1] = c[i] + w * (-g[i - 1] + 13.0 * g[i] + 13.0 * g[i + 1] - g[i + 2]);
        }
        c[n - 1] = c[n - 2] + w * (g[n - 4] - 5.0 * g[n - 3] + 19.0 * g[n - 2] + 9.0 * g[n - 1]);
        c
    }

    /// ∫₀^{r_{i_end}} f dr.
    pub fn partial(&self, f: &[f64], i_end: usize) -> f64 { self.cumulative(f)[i_end] }

    /// The grid index nearest to r (in x).
    pub fn nearest(&self, r: f64) -> usize {
        (((r.ln() - self.x0) / self.delta).round().max(0.0) as usize).min(self.len() - 1)
    }

    /// df/dr = (df/dx)/r, 5-point centred in x, one-sided 5-point at both ends.
    pub fn derivative(&self, f: &[f64]) -> Vec<f64> {
        let n = self.len();
        let d = 12.0 * self.delta;
        (0..n).map(|i| {
            let fx = if i >= 2 && i + 2 < n {
                (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) / d
            } else if i < 2 {
                let o = 0;
                if i == 0 { (-25.0 * f[o] + 48.0 * f[o + 1] - 36.0 * f[o + 2] + 16.0 * f[o + 3] - 3.0 * f[o + 4]) / d }
                else { (-3.0 * f[o] - 10.0 * f[o + 1] + 18.0 * f[o + 2] - 6.0 * f[o + 3] + f[o + 4]) / d }
            } else {
                let o = n - 5;
                if i == n - 1 { (3.0 * f[o] - 16.0 * f[o + 1] + 36.0 * f[o + 2] - 48.0 * f[o + 3] + 25.0 * f[o + 4]) / d }
                else { (-f[o] + 6.0 * f[o + 1] - 18.0 * f[o + 2] + 10.0 * f[o + 3] + 3.0 * f[o + 4]) / d }
            };
            fx / self.r[i]
        }).collect()
    }

    /// 4-point Lagrange interpolation in x; f[0] below r0, f[n−1] beyond r_max (callers bound the
    /// support themselves). `interp_deriv` is d/dr of the same interpolant (so a force taken with it
    /// is the exact derivative of an energy sampled with `interp`).
    pub fn interp(&self, f: &[f64], r: f64) -> f64 { self.lagrange(f, r).0 }
    pub fn interp_deriv(&self, f: &[f64], r: f64) -> f64 { self.lagrange(f, r).1 }

    fn lagrange(&self, f: &[f64], r: f64) -> (f64, f64) {
        let n = self.len();
        if r <= self.r[0] { return (f[0], 0.0); }
        if r >= self.r[n - 1] { return (f[n - 1], 0.0); }
        let s = (r.ln() - self.x0) / self.delta;
        let i0 = (s.floor() as usize).saturating_sub(1).min(n - 4);
        let t = s - i0 as f64;
        let (a, b, c, d) = (f[i0], f[i0 + 1], f[i0 + 2], f[i0 + 3]);
        let l = [-(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0, t * (t - 2.0) * (t - 3.0) / 2.0,
                 -t * (t - 1.0) * (t - 3.0) / 2.0, t * (t - 1.0) * (t - 2.0) / 6.0];
        let dl = [-((t - 2.0) * (t - 3.0) + (t - 1.0) * (t - 3.0) + (t - 1.0) * (t - 2.0)) / 6.0,
                  ((t - 2.0) * (t - 3.0) + t * (t - 3.0) + t * (t - 2.0)) / 2.0,
                  -((t - 1.0) * (t - 3.0) + t * (t - 3.0) + t * (t - 1.0)) / 2.0,
                  ((t - 1.0) * (t - 2.0) + t * (t - 2.0) + t * (t - 1.0)) / 6.0];
        let v = a * l[0] + b * l[1] + c * l[2] + d * l[3];
        let dv_ds = a * dl[0] + b * dl[1] + c * dl[2] + d * dl[3];
        (v, dv_ds / (self.delta * r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hydrogenic(g: &LogGrid) -> Vec<f64> { g.r.iter().map(|r| 4.0 * r * r * (-2.0 * r).exp()).collect() }

    #[test]
    fn integrals_of_a_hydrogenic_density() {
        // ∫ 4 r² e^{−2r} dr = 1; the cumulative to r: 1 − e^{−2r}(1 + 2r + 2r²)
        let g = LogGrid::universal();
        let f = hydrogenic(&g);
        assert!((g.integrate(&f) - 1.0).abs() < 1e-12);
    }

    /// Ruling (a) of the Task 1 review (2026-10-15): the cumulative rule's error is its own leading
    /// truncation term, asserted to 1 % (a wrong weight changes the coefficient, not only the size).
    /// Derivation (D = r d/dr = d/dx, g = f·r): the interior step δ/24·(−g_{i−1} + 13g_i + 13g_{i+1} −
    /// g_{i+2}) integrates a cubic exactly and has local error rule − exact = −(11/720)δ⁵g⁗(ξ) (one-signed
    /// Peano kernel); summed over the steps, c_i − exact = −(11/720)δ⁴[g‴(x_i) − g‴(x_0)] + O(δ⁶). For
    /// f = 4r²e^{−2r}: g = 4r³e^{−2r}, g‴ = D³g = −4r³(8r³ − 48r² + 74r − 27)e^{−2r}. At r_i = nearest(1.3)
    /// (1.300924) the prediction is +2.3070e-10, measured +2.3069e-10 — the brief's 1e-11 was not derived.
    #[test]
    fn cumulative_matches_its_truncation_term() {
        let g = LogGrid::universal();
        let c = g.cumulative(&hydrogenic(&g));
        let i = g.nearest(1.3);
        let r = g.r[i];
        let exact = 1.0 - (-2.0 * r).exp() * (1.0 + 2.0 * r + 2.0 * r * r);
        let g3 = |r: f64| -4.0 * r.powi(3) * (8.0 * r.powi(3) - 48.0 * r * r + 74.0 * r - 27.0) * (-2.0 * r).exp();
        let pred = -11.0 / 720.0 * g.delta.powi(4) * (g3(r) - g3(g.r[0]));
        let err = c[i] - exact;
        eprintln!("cumulative at r = {r:.6}: error {err:.5e}, truncation term {pred:.5e}");
        assert!((err - pred).abs() <= 0.01 * pred.abs(), "error {err:.5e} vs predicted {pred:.5e}");
    }

    /// ⟨Z/r⟩-type integrands are ∝ r near the origin, so the sphere inside r0 is not negligible:
    /// ∫₀^{r0} 4πr²ρ·(1/r) dr = 2πρ(0)r0² (≈ 4e-6 Ha·Z for Ne's ρ(0)). Hydrogenic density of charge
    /// Z = 10: ∫ 4πrρ dr = ⟨1/r⟩ = Z exactly; without the inner sphere the grid would miss 2Z³r0² = 2e-7.
    #[test]
    fn coulomb_weighted_integral_includes_the_inner_sphere() {
        let g = LogGrid::universal();
        let z = 10.0f64;
        let f: Vec<f64> = g.r.iter().map(|r| 4.0 * r * z.powi(3) * (-2.0 * z * r).exp()).collect();
        let i = g.integrate(&f);
        assert!((i - z).abs() < 1e-10, "⟨1/r⟩ = {i:.13} vs {z} (Δ {:.2e})", i - z);
        let c = g.cumulative(&f);
        assert!((c[c.len() - 1] - z).abs() < 1e-10, "cumulative end {:.13}", c[c.len() - 1]);
    }

    /// Convergence is a test: the cumulative integral's error falls as δ⁴ (ratio 16 per halving).
    #[test]
    fn cumulative_is_fourth_order() {
        let err = |delta: f64| {
            let g = LogGrid::new(1e-5, 60.0, delta);
            let c = g.cumulative(&hydrogenic(&g));
            let i = g.nearest(1.3);
            let r = g.r[i];
            (c[i] - (1.0 - (-2.0 * r).exp() * (1.0 + 2.0 * r + 2.0 * r * r))).abs()
        };
        let (e1, e2, e3) = (err(0.128), err(0.064), err(0.032));
        let (p1, p2) = ((e1 / e2).log2(), (e2 / e3).log2());
        eprintln!("cumulative error δ = 0.128/0.064/0.032: {e1:.2e} {e2:.2e} {e3:.2e}; orders {p1:.2} {p2:.2}");
        assert!((3.7..4.3).contains(&p1) && (3.7..4.3).contains(&p2), "orders {p1:.2}, {p2:.2}");
    }

    #[test]
    fn derivative_and_interpolation_of_a_smooth_function() {
        let g = LogGrid::universal();
        let f: Vec<f64> = g.r.iter().map(|r| (-r).exp() * r).collect();
        let d = g.derivative(&f);
        for &r in &[0.01, 0.5, 2.0, 7.0] {
            let i = g.nearest(r);
            let ri = g.r[i];
            assert!((d[i] - (-ri).exp() * (1.0 - ri)).abs() < 1e-9);
            let x = r * 1.0037;
            assert!((g.interp_deriv(&f, x) - (-x).exp() * (1.0 - x)).abs() < 1e-7);
        }
    }

    /// Ruling (a) of the Task 1 review: the 4-point Lagrange interpolant's error is its Cauchy
    /// remainder. With nodes at s = 0…3 and t = s − i0 ∈ [1, 2] (as in `lagrange`),
    /// f − p = δ⁴·D⁴f(ξ)·t(t−1)(t−2)(t−3)/24, so p − f ≈ −[t(t−1)(t−2)(t−3)/24]δ⁴D⁴f(x), with
    /// D⁴(re^{−r}) = r(r⁴ − 10r³ + 25r² − 15r + 1)e^{−r}. Asserted to 5 % + 1e-15 (the next order is
    /// O(δ·D⁵f/D⁴f); the review measured residuals 0.15–2.3 %). At r ≈ 2: predicted −1.1997e-10,
    /// measured −1.1955e-10 — the brief's 1e-10 was not derived.
    #[test]
    fn interpolation_matches_its_truncation_term() {
        let g = LogGrid::universal();
        let n = g.len();
        let f: Vec<f64> = g.r.iter().map(|r| (-r).exp() * r).collect();
        for &r in &[0.01f64, 0.5, 2.0, 7.0] {
            let x = r * 1.0037;
            let s = (x.ln() - g.x0) / g.delta;
            let t = s - ((s.floor() as usize).saturating_sub(1).min(n - 4)) as f64;
            let d4 = x * (x.powi(4) - 10.0 * x.powi(3) + 25.0 * x * x - 15.0 * x + 1.0) * (-x).exp();
            let pred = -t * (t - 1.0) * (t - 2.0) * (t - 3.0) / 24.0 * g.delta.powi(4) * d4;
            let err = g.interp(&f, x) - x * (-x).exp();
            eprintln!("interp at r = {x:.5}: error {err:.5e}, truncation term {pred:.5e}");
            assert!((err - pred).abs() <= 0.05 * pred.abs() + 1e-15, "r = {x}: error {err:.5e} vs predicted {pred:.5e}");
        }
    }
}
