//! The radial Schrödinger equation −½u″ + [l(l+1)/2r² + V(r)]u = εu on the log grid, by Numerov's
//! rule in x = ln r (spec §4.1; Hartree 1957, Herman & Skillman 1963, Fuchs & Scheffler 1999).
//! Atomic units; V is the electron's potential energy (Hartree), ε < 0 for a bound state.
//!
//! # The transformed equation (derivation)
//! With u(r) = e^{x/2} y(x): du/dr = e^{−x/2}(y′ + y/2), d²u/dr² = e^{−3x/2}(y″ − y/4). Substituting,
//!   y″ = g(x) y,   g = (l + ½)² + 2r²(V(r) − ε)
//! — no first derivative, so Numerov applies: y_{i+1}(1 − k g_{i+1}) = 2y_i(1 + 5k g_i) − y_{i−1}(1 − k g_{i−1}),
//! k = δ²/12, local error O(δ⁶), eigenvalue error O(δ⁴). With a source, y″ = g y + S,
//! −½u″ + (V_eff − ε)u = −s(r) gives S = 2r^{3/2} s, added as k(S_{i+1} + 10S_i + S_{i−1}).
//! ∫u² dr = ∫ r² y² dx.
//!
//! # Regular start
//! u ∝ r^{l+1}(1 − Z r/(l+1) + …) at a Coulomb origin of charge Z (`z_origin`; 0 for a finite
//! pseudopotential), so y = r^{l+½}(1 − Z r/(l+1)) at the first two points (error O(r0²) relative).
//!
//! # Shooting (`bound_state`)
//! The outward solution at energy e has k(e) sign changes up to `shooting_end` (the point beyond the
//! last turning point where the WKB decay ∫√g dx exceeds 40, or k·g > 0.1 where Numerov's denominator
//! would degrade). k(e) is non-decreasing; the state with `nodes` radial nodes is the boundary between
//! k = nodes and k = nodes + 1 (the growing tail flips sign there). Bisection to 1e-13·max(1, |e|).
//! Unbound (`None`) when even e = 0 gives k ≤ nodes (r_max = 60 Bohr is the box). The eigenfunction:
//! outward to the outermost turning point, inward from `shooting_end`, scaled to match, normalised;
//! sign: positive at its outermost maximum.

use super::grid::LogGrid;

#[derive(Clone, Debug, PartialEq)]
pub struct BoundState { pub e: f64, pub u: Vec<f64>, pub r_peak: f64, pub r_node: Option<f64> }

pub fn g_coeff(grid: &LogGrid, v: &[f64], l: u8, e: f64) -> Vec<f64> {
    let lh = l as f64 + 0.5;
    grid.r.iter().zip(v).map(|(&r, &vi)| lh * lh + 2.0 * r * r * (vi - e)).collect()
}

pub fn regular_start(grid: &LogGrid, l: u8, z_origin: f64) -> (f64, f64) {
    let f = |r: f64| r.powf(l as f64 + 0.5) * (1.0 - z_origin * r / (l as f64 + 1.0));
    (f(grid.r[0]), f(grid.r[1]))
}

/// Homogeneous outward run over 0..=i_end; returns y and its sign changes. Rescales the run by
/// 1e-150 whenever |y| > 1e150 (signs and nodes unchanged).
pub fn outward(g: &[f64], y0: f64, y1: f64, i_end: usize, delta: f64) -> (Vec<f64>, usize) {
    let k = delta * delta / 12.0;
    let mut y = vec![0.0; i_end + 1];
    y[0] = y0;
    if i_end >= 1 { y[1] = y1; }
    let mut nodes = 0;
    for i in 1..i_end {
        y[i + 1] = (2.0 * y[i] * (1.0 + 5.0 * k * g[i]) - y[i - 1] * (1.0 - k * g[i - 1])) / (1.0 - k * g[i + 1]);
        if y[i] * y[i + 1] < 0.0 { nodes += 1; }
        if y[i + 1].abs() > 1e150 { for v in &mut y[..=i + 1] { *v *= 1e-150; } }
    }
    (y, nodes)
}

/// The regular particular solution of y″ = g y + s (y = 0 at the first two points: the source is
/// O(r^{l+1}) in u-form, so the particular solution is O(r^{l+3})).
pub fn outward_sourced(g: &[f64], s: &[f64], i_end: usize, delta: f64) -> Vec<f64> {
    let k = delta * delta / 12.0;
    let mut y = vec![0.0; i_end + 1];
    for i in 1..i_end {
        y[i + 1] = (2.0 * y[i] * (1.0 + 5.0 * k * g[i]) - y[i - 1] * (1.0 - k * g[i - 1])
            + k * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / (1.0 - k * g[i + 1]);
    }
    y
}

/// Decaying inward run from i_start down to i_match; element j ↔ grid index i_match + j.
pub fn inward(g: &[f64], i_match: usize, i_start: usize, delta: f64) -> Vec<f64> {
    let k = delta * delta / 12.0;
    let m = i_start - i_match;
    let mut y = vec![0.0; m + 1];
    y[m] = 1e-20;
    y[m - 1] = 1e-20 * (g[i_start].max(0.0).sqrt() * delta).exp();
    for j in (1..m).rev() {
        let i = i_match + j;
        y[j - 1] = (2.0 * y[j] * (1.0 + 5.0 * k * g[i]) - y[j + 1] * (1.0 - k * g[i + 1])) / (1.0 - k * g[i - 1]);
        if y[j - 1].abs() > 1e150 { for v in &mut y[j - 1..] { *v *= 1e-150; } }
    }
    y
}

pub fn shooting_end(g: &[f64], delta: f64) -> usize {
    let n = g.len();
    let it = (0..n).rev().find(|&i| g[i] < 0.0).unwrap_or(0);
    let k = delta * delta / 12.0;
    let mut acc = 0.0;
    for (i, &gi) in g.iter().enumerate().skip(it) {
        if k * gi > 0.1 { return i; }
        acc += gi.max(0.0).sqrt() * delta;
        if acc > 40.0 { return i; }
    }
    n - 1
}

pub fn bound_state(grid: &LogGrid, v: &[f64], l: u8, nodes: usize, z_origin: f64, guess: Option<f64>) -> Option<BoundState> {
    let n = grid.len();
    let (y0, y1) = regular_start(grid, l, z_origin);
    let count = |e: f64| {
        let g = g_coeff(grid, v, l, e);
        outward(&g, y0, y1, shooting_end(&g, grid.delta), grid.delta).1
    };
    let mut e_hi = 0.0;
    if count(e_hi) <= nodes { return None; }
    let lf = l as f64;
    let mut e_lo = (0..n).map(|i| v[i] + lf * (lf + 1.0) / (2.0 * grid.r[i] * grid.r[i])).fold(f64::INFINITY, f64::min);
    if let Some(gs) = guess {
        let w = 0.1 * gs.abs().max(0.1);
        if count(gs - w) <= nodes { e_lo = gs - w; }
        if gs + w < 0.0 && count(gs + w) > nodes { e_hi = gs + w; }
    }
    while e_hi - e_lo > 1e-13 * e_lo.abs().max(1.0) {
        let mid = 0.5 * (e_lo + e_hi);
        if count(mid) > nodes { e_hi = mid } else { e_lo = mid }
    }
    Some(eigenfunction(grid, v, l, 0.5 * (e_lo + e_hi), z_origin))
}

pub fn eigenfunction(grid: &LogGrid, v: &[f64], l: u8, e: f64, z_origin: f64) -> BoundState {
    let n = grid.len();
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, z_origin);
    let it = (0..n).rev().find(|&i| g[i] < 0.0).unwrap_or(n / 2).clamp(2, n - 3);
    let ie = shooting_end(&g, grid.delta).max(it + 2).min(n - 1);
    let (yo, _) = outward(&g, y0, y1, it, grid.delta);
    let yi = inward(&g, it, ie, grid.delta);
    let scale = yo[it] / yi[0];
    let mut u = vec![0.0; n];
    for i in 0..=it { u[i] = yo[i] * grid.r[i].sqrt(); }
    for j in 1..yi.len() { u[it + j] = scale * yi[j] * grid.r[it + j].sqrt(); }
    let norm = grid.integrate(&u.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
    for x in &mut u { *x /= norm; }
    let umax = u.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let ip = (1..n - 1).rev()
        .find(|&i| u[i].abs() >= u[i - 1].abs() && u[i].abs() >= u[i + 1].abs() && u[i].abs() > 1e-3 * umax)
        .unwrap_or(0);
    if u[ip] < 0.0 { for x in &mut u { *x = -*x; } }
    let r_node = (1..=ip).rev().find(|&i| u[i - 1] * u[i] < 0.0).map(|i| {
        let t = u[i - 1] / (u[i - 1] - u[i]);
        grid.r[i - 1] + t * (grid.r[i] - grid.r[i - 1])
    });
    BoundState { e, u, r_peak: grid.r[ip], r_node }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coulomb_error(delta: f64, z: f64, n: u8, l: u8) -> f64 {
        let g = LogGrid::new(1e-5, 60.0, delta);
        let v: Vec<f64> = g.r.iter().map(|r| -z / r).collect();
        let b = bound_state(&g, &v, l, (n - l - 1) as usize, z, None).expect("bound");
        b.e - (-z * z / (2.0 * (n as f64).powi(2)))
    }

    /// Convergence is a test: Numerov's eigenvalue error falls as δ⁴ (spec §4.1; the plan's open risk
    /// on whether δ = 0.008 meets the 1e-10 hydrogenic bound). Measured on the coarse end of the
    /// ladder, where the truncation error dominates roundoff and the bisection tolerance.
    #[test]
    fn numerov_eigenvalue_error_is_fourth_order() {
        for &(z, n, l) in &[(1.0, 1u8, 0u8), (1.0, 2, 1), (8.0, 1, 0), (8.0, 3, 2)] {
            let deltas = [0.064, 0.032, 0.016, 0.008, 0.004];
            let errs: Vec<f64> = deltas.iter().map(|&d| coulomb_error(d, z, n, l)).collect();
            let abs: Vec<String> = errs.iter().map(|e| format!("{e:+.2e}")).collect();
            let rel: Vec<String> = errs.iter().map(|e| format!("{:+.2e}", e / (z * z))).collect();
            eprintln!("Z = {z}, n = {n}, l = {l}: Δε at δ = {deltas:?}: {abs:?}; Δε/Z² {rel:?}");
            let p1 = (errs[0] / errs[1]).abs().log2();
            let p2 = (errs[1] / errs[2]).abs().log2();
            assert!((3.7..4.3).contains(&p1) && (3.7..4.3).contains(&p2), "Z = {z} n = {n} l = {l}: orders {p1:.2}, {p2:.2}");
        }
    }

    /// The node count of the eigenfunction and its normalisation (∫u² dr = 1), the peak sign.
    #[test]
    fn eigenfunction_shape() {
        let g = LogGrid::universal();
        let v: Vec<f64> = g.r.iter().map(|r| -1.0 / r).collect();
        let b = bound_state(&g, &v, 0, 1, 1.0, None).expect("2s bound");
        let norm = g.integrate(&b.u.iter().map(|x| x * x).collect::<Vec<_>>());
        assert!((norm - 1.0).abs() < 1e-12);
        // hydrogen 2s: u ∝ r(2 − r)e^{−r/2}: node at r = 2, outermost maximum at r = 3 + √5
        let node = b.r_node.expect("one node");
        assert!((node - 2.0).abs() < 1e-3, "node {node}");
        assert!((b.r_peak - (3.0 + 5f64.sqrt())).abs() < 0.03, "peak {}", b.r_peak);
        assert!(b.u[g.nearest(b.r_peak)] > 0.0);
    }
}

