//! Troullier–Martins norm-conserving channels and unscreening (spec §4.3, §4.4 r_c rule; Troullier &
//! Martins, Phys. Rev. B 43, 1993 (1991), eqs. 4–13; review M1, S1, S5). CPU, f64, atomic units:
//! V is the electron's potential energy (Hartree), u = rR, ∫u² dr = 1, ε < 0.
//!
//! # The pseudo-wavefunction
//! u^PP(r) = u^AE(r) for r ≥ r_c; u^PP(r) = r^{l+1} e^{p(r)}, p = c₀ + c₂r² + c₄r⁴ + … + c₁₂r¹², for
//! r < r_c (`c = [c₀, c₂, c₄, c₆, c₈, c₁₀, c₁₂]`; the degree is `LEDGER.tm_degree` = 12). Nodeless by
//! construction, so r_c must lie beyond the outermost node of u^AE (u^AE > 0 there with the
//! eigenfunction sign convention of `numerov`: positive at the outermost maximum).
//!
//! # Inversion (the screened potential)
//! u′/u = (l+1)/r + p′ and u″/u = (u′/u)′ + (u′/u)² = l(l+1)/r² + 2(l+1)p′/r + p′² + p″. The radial
//! equation −½u″ + [l(l+1)/2r² + V]u = εu gives u″/u = l(l+1)/r² + 2(V − ε), hence
//!   V_scr(r) = ε + (l+1)p′/r + ½(p′² + p″)                                      (r < r_c),
//! evaluated with p′/r = 2c₂ + 4c₄r² + … + 12c₁₂r¹⁰ expanded (no division at small r).
//!
//! # The origin (review M1)
//! p′/r = 2c₂ + 4c₄r² + O(r⁴), p′² = 4c₂²r² + O(r⁴), p″ = 2c₂ + 12c₄r² + O(r⁴) give
//!   V_scr(r) = ε + (2l+3)c₂ + 2[c₂² + (2l+5)c₄]r² + O(r⁴),
//! so zero curvature at the origin, V_scr″(0) = 0, is **c₂² + (2l+5)c₄ = 0** (TM eq. 12), and
//! V_scr(0) = ε + (2l+3)c₂.
//!
//! # The matching targets (TM eqs. 8–11), from the AE solution at r_c
//! p = ln(u/r^{l+1}), p′ = u′/u − (l+1)/r, and from the V_scr relation solved for p″ and differentiated
//! (V = V_AE, V′, V″ its grid derivatives):
//!   p″ = 2(V − ε) − 2(l+1)p′/r − p′²,
//!   p‴ = 2V′ + 2(l+1)p′/r² − 2(l+1)p″/r − 2p′p″,
//!   p⁗ = 2V″ − 4(l+1)p′/r³ + 4(l+1)p″/r² − 2(l+1)p‴/r − 2p″² − 2p′p‴.
//! Continuity of p, p′, …, p⁗ at r_c is continuity of u, u′ and of V_scr, V_scr′, V_scr″ (S1: the
//! tests check the latter independently of these formulas).
//!
//! # Solving (TM's own scheme; `tm_solve`, given the targets)
//! For a trial c₂: c₄ = −c₂²/(2l+5); the five derivative conditions are linear in (c₀, c₆, c₈, c₁₀, c₁₂)
//! — a 5×5 LU solve. Norm conservation ∫₀^{r_c}(u^PP)² dr = ∫₀^{r_c}(u^AE)² dr is then one equation in
//! c₂, solved by Newton from c₂ = 0 on F(c₂) = ln N_PP − ln N_AE. r_c is snapped to the log grid
//! (`LogGrid::nearest`) and both norms use the same cumulative quadrature (`LogGrid::partial`) on the
//! complete functions — u^PP is u^AE beyond r_c, so the 4-point step into r_c sees the same g_{i_c+1}
//! for both (deviation from the plan brief, which zeroed u^PP beyond r_c: its residual then solved a
//! different equation from the one the norm test checks) — so norm conservation is exact on that
//! quadrature.
//!
//! # r_c (spec §4.4, review S5)
//! r_c,l = f·r_peak,l for occupied channels; for an unoccupied channel min(f·r_peak,l, max over the
//! occupied channels of r_c) — both snapped to the grid.
//!
//! # Unscreening (spec §4.3)
//!   V_l^ion = V_l^scr − V_H[ρ_v^PP] − v_xc[ρ_v^PP + ρ_pc],  ρ_v^PP = Σ_l f_l (u_l^PP)²/(4πr²),
//! spin-restricted (each spin ½(ρ_v^PP + ρ_pc), the reference's own convention), V_H by Gauss's law
//! (`hartree_potential`, the routine the reference atom used). Beyond max(r_c) and with ρ_pc = ρ_c,
//! V_l^ion = −Z/r + V_H[ρ_c] = −Z_v/r + (the core's Hartree tail), since ρ_v^PP = ρ_v^AE there and the
//! norm-conserving difference ρ_v^AE − ρ_v^PP carries no charge outside. The core tail is
//! V_H[ρ_c] − N_c/r = −∫_r^∞ 4πr′²ρ_c(r′)(1/r − 1/r′) dr′ ≤ 0 — the core charge still outside r, and
//! independent of f (measured on O: −4.46e-8 at r = 1.409, just beyond max r_c at f = 1.6). r_cmp is the
//! smallest grid radius from which |V_ion + Z_v/r| < `LEDGER.v_ion_tail` for every channel (computed, not
//! extended); whenever the core tail outlasts max r_c, it is the core tail that sets r_cmp.
//!
//! # Method-convergence literals (review S2 practice; none moves a converged result beyond its tolerance)
//! Newton on c₂: stop at |F| < 1e-14, at most 100 steps, accept |F| < 1e-12; central-difference
//! step 1e-6·max(1, |c₂|); each step clamped to ±1. Probe radius 1e-3 Bohr in the origin test.
use super::atom::{hartree_potential, AtomResult};
use super::grid::LogGrid;
use super::ledger::LEDGER;
use super::partition::Partition;
use crate::xc::exc_point;
use nalgebra::{Matrix5, Vector5};

#[derive(Clone, Debug, PartialEq)]
pub struct TmChannel {
    pub l: u8, pub eps: f64,
    /// r_c snapped to the grid, r_c = r[i_c].
    pub r_c: f64, pub i_c: usize,
    /// [c₀, c₂, c₄, c₆, c₈, c₁₀, c₁₂].
    pub c: [f64; 7],
    /// u^PP (u^AE beyond i_c) and V_l^scr (V_AE beyond i_c) on the whole grid.
    pub u: Vec<f64>, pub v_scr: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Unscreened {
    /// V_l^ion per channel, in the order of the input channels (Hartree).
    pub v_ion: Vec<Vec<f64>>,
    /// ρ_v^PP (e/Bohr³).
    pub rho_v: Vec<f64>,
    /// The smallest grid radius from which |V_ion + Z_v/r| < `LEDGER.v_ion_tail` for every channel — a
    /// 1e-8 Ha potential criterion (set by the core's Hartree tail where it outlasts max r_c), the support
    /// §5.1 uses; not a core radius (§5.6 uses r_core; review N7).
    pub r_cmp: f64,
}

/// p = ln(u/r^{l+1}) and its first four derivatives at r_{ic} from the AE solution (TM eqs. 8–11):
/// p′ = u′/u − (l+1)/r; p″ = 2(V − ε) − 2(l+1)p′/r − p′²;
/// p‴ = 2V′ + 2(l+1)p′/r² − 2(l+1)p″/r − 2p′p″;
/// p⁗ = 2V″ − 4(l+1)p′/r³ + 4(l+1)p″/r² − 2(l+1)p‴/r − 2p″² − 2p′p‴.
pub(crate) fn tm_targets(grid: &LogGrid, l: u8, eps: f64, u_ae: &[f64], v_ae: &[f64], ic: usize) -> [f64; 5] {
    let lp = l as f64 + 1.0;
    let rc = grid.r[ic];
    let du = grid.derivative(u_ae);
    let dv = grid.derivative(v_ae);
    let d2v = grid.derivative(&dv);
    let p0 = (u_ae[ic] / rc.powf(lp)).ln();
    let p1 = du[ic] / u_ae[ic] - lp / rc;
    let p2 = 2.0 * (v_ae[ic] - eps) - 2.0 * lp * p1 / rc - p1 * p1;
    let p3 = 2.0 * dv[ic] + 2.0 * lp * p1 / (rc * rc) - 2.0 * lp * p2 / rc - 2.0 * p1 * p2;
    let p4 = 2.0 * d2v[ic] - 4.0 * lp * p1 / rc.powi(3) + 4.0 * lp * p2 / (rc * rc) - 2.0 * lp * p3 / rc - 2.0 * p2 * p2 - 2.0 * p1 * p3;
    [p0, p1, p2, p3, p4]
}

/// d^k/dr^k r^m = m!/(m−k)! r^{m−k} (0 when k > m).
fn dpow(m: i32, k: usize, r: f64) -> f64 {
    let mut f = 1.0;
    for q in 0..k as i32 { f *= (m - q) as f64; }
    if f == 0.0 { 0.0 } else { f * r.powi(m - k as i32) }
}

/// d^k/dr^k of p(r) = Σ_j c_j r^{2j}, j = 0..6.
pub(crate) fn poly_derivative(c: &[f64; 7], k: usize, r: f64) -> f64 {
    (0..7).map(|j| c[j] * dpow(2 * j as i32, k, r)).sum()
}

/// The TM channel at r_c (snapped to the grid) from the AE state (ε, u_AE) in the potential v_AE.
/// Errors: r_c inside (or at) the outermost node of u_AE, or no solution of the norm equation.
pub fn tm_channel(grid: &LogGrid, l: u8, eps: f64, u_ae: &[f64], v_ae: &[f64], r_c: f64) -> Result<TmChannel, String> {
    let n = grid.len();
    let ic = grid.nearest(r_c);
    let rc = grid.r[ic];
    if ic < 2 || ic + 3 > n { return Err(format!("l = {l}: r_c = {rc:.3e} outside the grid's interior")); }
    // nodeless pseudo-wavefunction: u_AE must keep its (positive) sign from r_c outwards
    if u_ae[ic] <= 0.0 || (ic..n - 1).any(|i| u_ae[i] * u_ae[i + 1] < 0.0) {
        return Err(format!("l = {l}: r_c = {rc:.3} is inside the outermost node of u_AE (u_AE(r_c) = {:.3e})", u_ae[ic]));
    }
    let lf = l as f64;
    let target = tm_targets(grid, l, eps, u_ae, v_ae, ic);
    let c = tm_solve(grid, l, u_ae, ic, &target)?;
    let u_of = |c: &[f64; 7], r: f64| r.powf(lf + 1.0) * poly_derivative(c, 0, r).exp();
    let mut u = u_ae.to_vec();
    let mut v_scr = v_ae.to_vec();
    for i in 0..=ic {
        let r = grid.r[i];
        u[i] = u_of(&c, r);
        // V = ε + (l+1)p′/r + ½(p′² + p″), with p′/r expanded (no division at small r)
        let r2 = r * r;
        let p1_over_r = 2.0 * c[1] + r2 * (4.0 * c[2] + r2 * (6.0 * c[3] + r2 * (8.0 * c[4] + r2 * (10.0 * c[5] + r2 * 12.0 * c[6]))));
        let p1 = r * p1_over_r;
        let p2 = poly_derivative(&c, 2, r);
        v_scr[i] = eps + (lf + 1.0) * p1_over_r + 0.5 * (p1 * p1 + p2);
    }
    Ok(TmChannel { l, eps, r_c: rc, i_c: ic, c, u, v_scr })
}

/// The TM coefficients for matching data `target` = (p, p′, p″, p‴, p⁗) at r_{ic} and the norm of
/// u_AE inside r_{ic}: c₄ = −c₂²/(2l+5), the 5×5 solve for (c₀, c₆, c₈, c₁₀, c₁₂), Newton on c₂ from 0.
pub(crate) fn tm_solve(grid: &LogGrid, l: u8, u_ae: &[f64], ic: usize, target: &[f64; 5]) -> Result<[f64; 7], String> {
    let n = grid.len();
    let rc = grid.r[ic];
    let lf = l as f64;
    let n_ae = grid.partial(&u_ae.iter().map(|u| u * u).collect::<Vec<_>>(), ic);
    let ms = [0, 6, 8, 10, 12];
    let lu = Matrix5::from_fn(|k, j| dpow(ms[j], k, rc)).lu();
    // for a given c2: c4 = −c2²/(2l+5) (V_scr″(0) = 0), the 5×5 linear system for c0, c6, c8, c10, c12
    let coeffs = |c2: f64| -> Option<[f64; 7]> {
        let c4 = -c2 * c2 / (2.0 * lf + 5.0);
        let b = Vector5::from_fn(|k, _| target[k] - c2 * dpow(2, k, rc) - c4 * dpow(4, k, rc));
        let x = lu.solve(&b)?;
        let c = [x[0], c2, c4, x[1], x[2], x[3], x[4]];
        c.iter().all(|v| v.is_finite()).then_some(c)
    };
    let u_of = |c: &[f64; 7], r: f64| r.powf(lf + 1.0) * poly_derivative(c, 0, r).exp();
    // the complete u_PP² (u_AE beyond i_c), so both norms see the same quadrature points
    let resid = |c2: f64| -> Option<f64> {
        let c = coeffs(c2)?;
        let f: Vec<f64> = (0..n).map(|i| if i <= ic { u_of(&c, grid.r[i]).powi(2) } else { u_ae[i] * u_ae[i] }).collect();
        let r = grid.partial(&f, ic).ln() - n_ae.ln();
        r.is_finite().then_some(r)
    };
    let singular = || format!("l = {l}, r_c = {rc:.3}: TM matching system singular or overflowing");
    // Newton on c2 from 0 (TM), central-difference derivative, steps clamped to ±1
    let mut c2 = 0.0;
    let mut f = resid(c2).ok_or_else(singular)?;
    for _ in 0..100 {
        if f.abs() < 1e-14 { break; }
        let h = 1e-6 * c2.abs().max(1.0);
        let df = (resid(c2 + h).ok_or_else(singular)? - resid(c2 - h).ok_or_else(singular)?) / (2.0 * h);
        if !df.is_finite() || df == 0.0 { return Err(format!("l = {l}: TM norm equation has a flat derivative at c2 = {c2}")); }
        c2 += (-f / df).clamp(-1.0, 1.0);
        f = resid(c2).ok_or_else(singular)?;
    }
    if !(f.abs() < 1e-12) { return Err(format!("l = {l}, r_c = {rc:.3}: norm conservation has no TM solution (residual {f:.2e})")); }
    coeffs(c2).ok_or_else(singular)
}

/// Spec §4.4 r_c rule: r_c,l = f·r_peak,l (occupied channels); unoccupied: min(f·r_peak,l, max
/// occupied r_c) (review S5). Same order as `part.channels` (ascending l).
pub fn tm_channels(reference: &AtomResult, part: &Partition, f: f64) -> Result<Vec<TmChannel>, String> {
    let g = &reference.grid;
    let snap = |r: f64| g.r[g.nearest(r)];
    let cap = part.channels.iter().filter(|c| c.occupied).map(|c| snap(f * c.r_peak)).fold(0.0, f64::max);
    part.channels.iter().map(|c| {
        let r_c = if c.occupied { f * c.r_peak } else { (f * c.r_peak).min(cap) };
        tm_channel(g, c.l, c.eps, &c.u, &reference.v[0], r_c).map_err(|e| format!("Z = {}, f = {f}: {e}", part.z))
    }).collect()
}

/// V_l^ion = V_l^scr − V_H[ρ_v^PP] − v_xc[ρ_v^PP + ρ_pc] (spin-restricted: each spin ½(ρ_v + ρ_pc));
/// `occ[k]` is channel k's occupation (both spins); r_cmp as in [`Unscreened`]. v_xc is
/// `crate::xc::exc_point` (PW92, the generator's functional): the channels must come from a PW92
/// reference (`ground_state`), not from `ground_state_with` another `XcFn` — `AtomResult` does not
/// record its functional, and a mismatch would leave an O(1e-3) Ha error in V_ion (review N1).
pub fn unscreen(grid: &LogGrid, z_v: f64, channels: &[TmChannel], occ: &[f64], rho_pc: &[f64]) -> Unscreened {
    assert_eq!(channels.len(), occ.len(), "one occupation per channel");
    let n = grid.len();
    let rho_v: Vec<f64> = (0..n).map(|i| {
        let r = grid.r[i];
        channels.iter().zip(occ).map(|(c, f)| f * c.u[i] * c.u[i]).sum::<f64>() / (4.0 * std::f64::consts::PI * r * r)
    }).collect();
    let v_h = hartree_potential(grid, &rho_v);
    let vxc: Vec<f64> = (0..n).map(|i| { let half = 0.5 * (rho_v[i] + rho_pc[i]); exc_point(half, half).1 }).collect();
    let v_ion: Vec<Vec<f64>> = channels.iter().map(|c| (0..n).map(|i| c.v_scr[i] - v_h[i] - vxc[i]).collect()).collect();
    let bad = (0..n).rev().find(|&i| v_ion.iter().any(|v| (v[i] + z_v / grid.r[i]).abs() >= LEDGER.v_ion_tail));
    let r_cmp = bad.map_or(grid.r[0], |i| grid.r[(i + 1).min(n - 1)]);
    Unscreened { v_ion, rho_v, r_cmp }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::numerov::bound_state;
    use crate::radial::partition::{core_density, partition};
    use crate::radial::{ground_state, SpinMode, LEDGER};

    fn sq(u: &[f64]) -> Vec<f64> { u.iter().map(|x| x * x).collect() }

    /// Spec §8 row 2 (review M1 and the NOTE on Task 2's first test): the seven TM conditions on O 2p.
    #[test]
    fn tm_oxygen_2p_satisfies_the_seven_conditions() {
        let a = ground_state(8, SpinMode::Restricted).expect("O converges");
        let part = partition(&a).expect("O is accepted by the partition");
        let ch = part.channels.iter().find(|c| c.l == 1).expect("a p channel");
        let f = LEDGER.f_ladder[0];
        let tm = tm_channel(&a.grid, 1, ch.eps, &ch.u, &a.v[0], f * ch.r_peak).expect("TM solves");
        let g = &a.grid;
        let (ic, rc) = (tm.i_c, g.r[tm.i_c]);
        eprintln!("O 2p: ε {:.8}, r_peak {:.4}, r_c {rc:.4} (i_c {ic}), c = {:?}", ch.eps, ch.r_peak, tm.c);
        // (1) norm conservation, ∫₀^{r_c}|u^PP|² = ∫₀^{r_c}|u^AE|², to 1e-10
        let (n_ae, n_pp) = (g.partial(&sq(&ch.u), ic), g.partial(&sq(&tm.u), ic));
        eprintln!("(1) norm AE {n_ae:.14} PP {n_pp:.14} Δ {:.2e}", n_pp - n_ae);
        assert!((n_ae - n_pp).abs() < 1e-10, "norm: AE {n_ae:.12} PP {n_pp:.12}");
        // (2)–(6) p, p′, p″, p‴, p⁗ at r_c equal the AE values (TM eqs. 8–11), 1e-8 relative
        let target = tm_targets(g, 1, ch.eps, &ch.u, &a.v[0], ic);
        for k in 0..5 {
            let got = poly_derivative(&tm.c, k, rc);
            eprintln!("({}) p^({k})(r_c) {got:.12e} vs {:.12e}: rel {:.2e}", k + 2, target[k], (got - target[k]).abs() / target[k].abs().max(1.0));
            assert!((got - target[k]).abs() <= 1e-8 * target[k].abs().max(1.0), "p^({k})(r_c): {got:.12e} vs {:.12e}", target[k]);
        }
        // (2′)–(6′) the physical content of TM's C⁴ matching, independent of `tm_targets` (plan review S1:
        // the loop above compares the polynomial with the very right-hand side the 5×5 solve consumed,
        // so a sign error in the p‴ or p⁗ formula would pass it). From the coefficients alone, with
        // V = ε + (l+1)p′/r + ½(p′² + p″):
        //   V′ = (l+1)(p″/r − p′/r²) + p′p″ + ½p‴,
        //   V″ = (l+1)(p‴/r − 2p″/r² + 2p′/r³) + p″² + p′p‴ + ½p⁗,
        // compared with the AE potential's own grid derivatives; and u^PP′(r_c⁻) = u^PP·((l+1)/r + p′)
        // with the AE u′.
        let lp = 2.0; // l + 1 for the p channel
        let p = |k: usize| poly_derivative(&tm.c, k, rc);
        let v_pp = ch.eps + lp * p(1) / rc + 0.5 * (p(1) * p(1) + p(2));
        let dv_pp = lp * (p(2) / rc - p(1) / (rc * rc)) + p(1) * p(2) + 0.5 * p(3);
        let d2v_pp = lp * (p(3) / rc - 2.0 * p(2) / (rc * rc) + 2.0 * p(1) / rc.powi(3)) + p(2) * p(2) + p(1) * p(3) + 0.5 * p(4);
        let dv_ae = g.derivative(&a.v[0]);
        let d2v_ae = g.derivative(&dv_ae);
        let du_pp = rc.powf(lp) * p(0).exp() * (lp / rc + p(1));
        let du_ae = g.derivative(&ch.u)[ic];
        eprintln!("(S1) V {v_pp:.12} vs {:.12} (Δ {:.2e}); V′ rel {:.2e}; V″ rel {:.2e}; u′ rel {:.2e}", a.v[0][ic], v_pp - a.v[0][ic],
                  (dv_pp - dv_ae[ic]).abs() / dv_ae[ic].abs(), (d2v_pp - d2v_ae[ic]).abs() / d2v_ae[ic].abs(), (du_pp - du_ae).abs() / du_ae.abs());
        assert!((v_pp - a.v[0][ic]).abs() < 1e-8, "V_scr(r_c⁻) {v_pp:.12} vs V_AE(r_c) {:.12}", a.v[0][ic]);
        assert!((dv_pp - dv_ae[ic]).abs() <= 1e-6 * dv_ae[ic].abs(), "V′: PP {dv_pp:.10e} vs AE {:.10e}", dv_ae[ic]);
        assert!((d2v_pp - d2v_ae[ic]).abs() <= 1e-6 * d2v_ae[ic].abs(), "V″: PP {d2v_pp:.10e} vs AE {:.10e}", d2v_ae[ic]);
        assert!((du_pp - du_ae).abs() <= 1e-8 * du_ae.abs(), "u′(r_c): PP {du_pp:.12e} vs AE {du_ae:.12e}");
        // (7) V_scr″(0) = 0 ⇔ c₂² + (2l+5)c₄ = 0, with V_scr(0) = ε + (2l+3)c₂ (spec §4.3, review M1)
        let (c2, c4) = (tm.c[1], tm.c[2]);
        eprintln!("(7) c2² + 7c4 = {:.3e} (c2 {c2:.10}); V_scr(r0) − (ε + 5c2) = {:.3e}", c2 * c2 + 7.0 * c4, tm.v_scr[0] - (ch.eps + 5.0 * c2));
        assert!((c2 * c2 + 7.0 * c4).abs() <= 1e-12 * (c2 * c2).max(1e-300), "c2² + 7c4 = {:.3e}", c2 * c2 + 7.0 * c4);
        let v0 = ch.eps + 5.0 * c2;
        assert!((tm.v_scr[0] - v0).abs() < 1e-8, "V_scr(r0) = {:.12} vs ε + 5c₂ = {v0:.12}", tm.v_scr[0]);
        let i = g.nearest(1e-3);
        assert!(((tm.v_scr[i] - v0) / (g.r[i] * g.r[i])).abs() < 1e-4, "the r² coefficient of V_scr does not vanish");
        // Unscreening (spec §4.3): "each channel's ionic potential equals −Z_v/r only beyond
        // max(r_c, r_nlc) and the core tail". At Task 2 ρ_pc = ρ_c (the r_nlc → 0 limit; Task 4 re-runs
        // this with the LFC partial core); the module doc's derivation then makes V_ion + Z_v/r beyond
        // max r_c exactly the core's Hartree tail V_H[ρ_c] − N_c/r (the 1s density beyond r). Asserted to
        // v_ion_tail/100 = 1e-10 — tight enough that r_cmp is fixed by the core tail alone — then r_cmp is
        // verified: beyond it every channel is −Z_v/r to v_ion_tail, it is minimal, and it is the radius
        // where the core tail alone drops below v_ion_tail. Every element and f:
        // `v_ion_tail_is_the_core_hartree_tail_across_the_ladder`.
        let chans = tm_channels(&a, &part, f).expect("every channel solves");
        let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
        let rho_c = core_density(&a, &part);
        let un = unscreen(g, part.z_v, &chans, &occ, &rho_c);
        let vhc = hartree_potential(g, &rho_c);
        let core_tail = |i: usize| vhc[i] - part.n_core as f64 / g.r[i];
        let n = g.len();
        let i_from = chans.iter().map(|c| c.i_c).max().unwrap() + 1;
        for (k, v) in un.v_ion.iter().enumerate() {
            let worst = (i_from..n).map(|i| (v[i] + part.z_v / g.r[i] - core_tail(i)).abs()).fold(0.0, f64::max);
            eprintln!("channel l = {}: r_c {:.4}; max |V_ion + Z_v/r − (V_H[ρ_c] − N_c/r)| beyond max r_c ({:.4}) = {worst:.2e}; V_ion(r0) {:.6}",
                      chans[k].l, chans[k].r_c, g.r[i_from], v[0]);
            assert!(worst < LEDGER.v_ion_tail / 100.0, "channel {k}: V_ion + Z_v/r is not the core tail ({worst:.2e})");
        }
        for i in (i_from..n).step_by(6).take(8) {
            eprintln!("  r {:.4}: V_ion,s + Z_v/r {:+.4e}, core tail {:+.4e}", g.r[i], un.v_ion[0][i] + part.z_v / g.r[i], core_tail(i));
        }
        let i_cmp = g.nearest(un.r_cmp);
        assert!((i_cmp..n).all(|i| un.v_ion.iter().all(|v| (v[i] + part.z_v / g.r[i]).abs() < LEDGER.v_ion_tail)), "r_cmp does not bound the tail");
        assert!(i_cmp > 0 && un.v_ion.iter().any(|v| (v[i_cmp - 1] + part.z_v / g.r[i_cmp - 1]).abs() >= LEDGER.v_ion_tail), "r_cmp is not minimal");
        let i_core = (0..n).rev().find(|&i| core_tail(i).abs() >= LEDGER.v_ion_tail).unwrap() + 1;
        eprintln!("r_cmp = {:.4}; the core tail alone falls below v_ion_tail at {:.4}", un.r_cmp, g.r[i_core]);
        assert!(i_core > i_from && i_cmp == i_core, "r_cmp {} vs the core tail's radius {}", un.r_cmp, g.r[i_core]);
    }

    /// Spec §4.3 (Task 2 review, ruling 1 and M1): beyond max r_c (ρ_pc = ρ_c here), every channel's
    /// V_ion + Z_v/r is the frozen core's Hartree tail τ_c = V_H[ρ_c] − N_c/r ≤ 0, not zero, for every
    /// element with a core and every f. The plan's earlier literal bound "|V_ion + Z_v/r| < 1e-8 from the
    /// first point beyond max r_c" was removed: it is false wherever τ_c outlasts max r_c (O at f = 1.6:
    /// τ_c = −4.46e-8 at r = 1.409). Asserted:
    /// (a) |V_ion + Z_v/r − τ_c| < v_ion_tail/100 for every channel from i_from = max i_c + 1 — the
    ///     identity is exact up to the one-step quadrature offset of the Hartree cumulative across r_c
    ///     (measured 1.0e-12 on O);
    /// (b) r_cmp bounds the tail (|V_ion + Z_v/r| < v_ion_tail from r_cmp on) and is minimal;
    /// (c) with i_tail the first index from which |τ_c| < v_ion_tail for good: if i_tail > i_from,
    ///     |i_cmp − i_tail| ≤ 1 (the identity residual can move the crossing by one point only when τ_c is
    ///     within 1e-10 of 1e-8 there); otherwise i_cmp ≤ i_from — the case where the plan's original
    ///     `r_cmp <= r[i_from]` holds (Be and Al at f = 1.6).
    #[test]
    fn v_ion_tail_is_the_core_hartree_tail_across_the_ladder() {
        for z in (4u8..=10).chain(13..=18) {
            let a = ground_state(z, SpinMode::Restricted).unwrap();
            let part = partition(&a).unwrap();
            let g = &a.grid;
            let n = g.len();
            let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
            let rho_c = core_density(&a, &part);
            let vhc = hartree_potential(g, &rho_c);
            let tau = |i: usize| vhc[i] - part.n_core as f64 / g.r[i];
            let i_tail = (0..n).rev().find(|&i| tau(i).abs() >= LEDGER.v_ion_tail).map_or(0, |i| i + 1);
            for f in LEDGER.f_ladder {
                let chans = tm_channels(&a, &part, f).unwrap();
                let un = unscreen(g, part.z_v, &chans, &occ, &rho_c);
                let i_from = chans.iter().map(|c| c.i_c).max().unwrap() + 1;
                let dev = |v: &[f64], i: usize| (v[i] + part.z_v / g.r[i]).abs();
                let worst = un.v_ion.iter().map(|v| (i_from..n).map(|i| (v[i] + part.z_v / g.r[i] - tau(i)).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
                let i_cmp = g.nearest(un.r_cmp);
                eprintln!("Z = {z:2}, f = {f}: max r_c {:.4}, τ_c there {:+.2e}; identity residual {worst:.1e}; r_cmp {:.4}, τ_c 1e-8 radius {:.4}",
                          g.r[i_from - 1], tau(i_from), un.r_cmp, g.r[i_tail]);
                assert!(worst < LEDGER.v_ion_tail / 100.0, "Z = {z}, f = {f}: V_ion + Z_v/r − τ_c = {worst:.2e}");
                assert!((i_cmp..n).all(|i| un.v_ion.iter().all(|v| dev(v, i) < LEDGER.v_ion_tail)), "Z = {z}, f = {f}: r_cmp does not bound the tail");
                assert!(i_cmp > 0 && un.v_ion.iter().any(|v| dev(v, i_cmp - 1) >= LEDGER.v_ion_tail), "Z = {z}, f = {f}: r_cmp is not minimal");
                if i_tail > i_from {
                    assert!(i_cmp.abs_diff(i_tail) <= 1, "Z = {z}, f = {f}: r_cmp {} vs the core tail's radius {}", un.r_cmp, g.r[i_tail]);
                } else {
                    assert!(i_cmp <= i_from, "Z = {z}, f = {f}: r_cmp {} beyond the first point past max r_c {}", un.r_cmp, g.r[i_from]);
                }
            }
        }
    }

    /// Analytic oracle for the solve itself: when u^AE is of TM form — u = r^{l+1}e^{p}, p = c₀ + c₂r² +
    /// c₄r⁴ with c₄ = −c₂²/(2l+5) — and the matching data are p's exact derivatives at r_c, the solve must
    /// return that polynomial (c₆…c₁₂ = 0): it is a root of the norm equation (u^PP = u^AE, so the two
    /// quadratures agree exactly), and Newton from c₂ = 0 must find it rather than another root. Bound:
    /// Newton stops at |F| < 1e-14 with dF/dc₂ = O(1), so δc₂ ~ 1e-14, propagated through the 5×5 solve
    /// (r_c^12 ≲ 4e3) — 1e-11 leaves two orders of margin. l = 0, 1, 2 check the (2l+5) factor.
    #[test]
    fn tm_recovers_an_exact_tm_polynomial() {
        let g = LogGrid::universal();
        for l in 0u8..=2 {
            let (c0, c2) = (0.2, -0.3);
            let exact = [c0, c2, -c2 * c2 / (2.0 * l as f64 + 5.0), 0.0, 0.0, 0.0, 0.0];
            let u: Vec<f64> = g.r.iter().map(|&r| r.powi(l as i32 + 1) * poly_derivative(&exact, 0, r).exp()).collect();
            for r_c in [1.0, 1.5, 2.0] {
                let ic = g.nearest(r_c);
                let rc = g.r[ic];
                let target: [f64; 5] = std::array::from_fn(|k| poly_derivative(&exact, k, rc));
                let c = tm_solve(&g, l, &u, ic, &target).expect("TM solves");
                let worst = (0..=ic).map(|i| (poly_derivative(&c, 0, g.r[i]) - poly_derivative(&exact, 0, g.r[i])).abs()).fold(0.0, f64::max);
                eprintln!("l = {l}, r_c {rc:.4}: Δc2 {:.2e}, max|Δp| on [0, r_c] {worst:.2e}, c6..c12 {:?}", c[1] - c2, &c[3..]);
                assert!(worst < 1e-11 && (c[1] - c2).abs() < 1e-11, "l = {l}, r_c {rc}: max |Δp| {worst:.2e}, Δc2 {:.2e}", c[1] - c2);
            }
        }
    }

    /// Convergence is a test: the full `tm_channel` (grid derivatives of u_AE and V_AE in `tm_targets`)
    /// on the analytic case above, l = 1, with V_AE the exact inversion V = ε + 2p′/r + ½(p′² + p″).
    /// The only error source is the 5-point derivatives' O(δ⁴) truncation (the norm quadratures agree
    /// exactly at the exact c), so at a fixed r_c common to the grids δ = 0.032, 0.016, 0.008 the error
    /// of p must fall by 2⁴ per halving: ratios in [14, 18] (Task 1's order-check window, review M1).
    #[test]
    fn tm_channel_converges_to_the_exact_polynomial_at_fourth_order() {
        let (eps, c0, c2) = (-0.4, 0.2, -0.3);
        let exact = [c0, c2, -c2 * c2 / 7.0, 0.0, 0.0, 0.0, 0.0];
        let p = |k: usize, r: f64| poly_derivative(&exact, k, r);
        let coarse = LogGrid::new(LEDGER.r0, LEDGER.r_max, 4.0 * LEDGER.delta);
        let r_c = coarse.r[coarse.nearest(1.5)];
        let err = |delta: f64| {
            let g = LogGrid::new(LEDGER.r0, LEDGER.r_max, delta);
            let u: Vec<f64> = g.r.iter().map(|&r| r * r * p(0, r).exp()).collect();
            let v: Vec<f64> = g.r.iter().map(|&r| eps + 2.0 * (2.0 * exact[1] + 4.0 * exact[2] * r * r) + 0.5 * (p(1, r).powi(2) + p(2, r))).collect();
            let tm = tm_channel(&g, 1, eps, &u, &v, r_c).expect("TM solves");
            assert!((tm.r_c - r_c).abs() < 1e-12 * r_c, "r_c is a point of every grid");
            (0..=tm.i_c).map(|i| (poly_derivative(&tm.c, 0, g.r[i]) - p(0, g.r[i])).abs()).fold(0.0, f64::max)
        };
        let e: Vec<f64> = [4.0, 2.0, 1.0].iter().map(|m| err(m * LEDGER.delta)).collect();
        let (q1, q2) = (e[0] / e[1], e[1] / e[2]);
        eprintln!("r_c {r_c:.4}: max|Δp| at δ = 0.032/0.016/0.008: {:.3e} {:.3e} {:.3e}; ratios {q1:.2} {q2:.2}", e[0], e[1], e[2]);
        assert!((14.0..=18.0).contains(&q1) && (14.0..=18.0).contains(&q2), "ratios {q1:.2}, {q2:.2}");
    }

    /// The inversion is consistent: the screened potential's own nodeless bound state is u^PP at the AE
    /// eigenvalue. Like for like, the AE eigenvalue is re-solved in the same stored potential a.v (the
    /// reference's ε comes from the last SCF input potential, ≈ 1e-10 away). The bound is a-posteriori:
    /// each Numerov solve's discretisation error is at most its change between δ and 2δ (the 2δ solve on
    /// every other point of the same tabulated potential; error ∝ δ^p with p ≥ 1 gives err(δ) =
    /// Δ/(2^p − 1) ≤ Δ — V_scr is only C² at r_c, so the clean δ⁴ of a smooth potential is not assumed), so
    /// |ε_PP − ε_AE| ≤ |Δ_PP| + |Δ_AE|, and likewise for the eigenfunctions on the common points.
    #[test]
    fn screened_potential_reproduces_the_reference_state() {
        let a = ground_state(8, SpinMode::Restricted).unwrap();
        let part = partition(&a).unwrap();
        let g = &a.grid;
        let gc = LogGrid::new(LEDGER.r0, LEDGER.r_max, 2.0 * LEDGER.delta);
        let sub = |v: &[f64]| (0..gc.len()).map(|i| v[2 * i]).collect::<Vec<f64>>();
        let du_max = |fine: &[f64], coarse: &[f64]| (0..gc.len()).map(|i| (fine[2 * i] - coarse[i]).abs()).fold(0.0, f64::max);
        for f in LEDGER.f_ladder {
            let chans = tm_channels(&a, &part, f).expect("O channels solve");
            for (c, t) in part.channels.iter().zip(&chans) {
                let nodes = (c.n - c.l - 1) as usize;
                let ae = bound_state(g, &a.v[0], c.l, nodes, a.z as f64, Some(c.eps)).expect("AE bound");
                let ae2 = bound_state(&gc, &sub(&a.v[0]), c.l, nodes, a.z as f64, Some(c.eps)).expect("AE bound at 2δ");
                let pp = bound_state(g, &t.v_scr, c.l, 0, 0.0, Some(c.eps)).expect("PP bound");
                let pp2 = bound_state(&gc, &sub(&t.v_scr), c.l, 0, 0.0, Some(c.eps)).expect("PP bound at 2δ");
                let bound_e = (pp.e - pp2.e).abs() + (ae.e - ae2.e).abs();
                let du = pp.u.iter().zip(&t.u).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
                let bound_u = du_max(&pp.u, &pp2.u) + du_max(&ae.u, &ae2.u);
                eprintln!("O f = {f}, l = {}: r_c {:.4}; ε_AE(a.v) {:.12} ε_PP {:.12} Δ {:+.2e} (bound {bound_e:.2e}; SCF lag ε − ε_AE(a.v) {:+.2e}); max|Δu| {du:.2e} (bound {bound_u:.2e})",
                          c.l, t.r_c, ae.e, pp.e, pp.e - ae.e, c.eps - ae.e);
                assert!((pp.e - ae.e).abs() <= bound_e, "f = {f}, l = {}: ε_PP {} vs ε_AE {} (bound {bound_e:.2e})", c.l, pp.e, ae.e);
                assert!(du <= bound_u, "f = {f}, l = {}: max |u_PP − u_TM| = {du:.2e} (bound {bound_u:.2e})", c.l);
            }
        }
    }

    /// The seven coefficients c0, c2, ..., c12 are the ledger's TM degree (spec §4.3, LEDGER.tm_degree).
    #[test]
    fn coefficient_count_is_the_ledger_degree() {
        assert_eq!(2 * (TmChannel { l: 0, eps: 0.0, r_c: 0.0, i_c: 0, c: [0.0; 7], u: vec![], v_scr: vec![] }.c.len() - 1), LEDGER.tm_degree as usize);
    }

    /// r_c inside the outermost node of u^AE (O 2s at half its node radius) is refused, not pseudised.
    #[test]
    fn r_c_inside_the_outermost_node_is_refused() {
        let a = ground_state(8, SpinMode::Restricted).unwrap();
        let s = a.orbitals.iter().find(|o| o.n == 2 && o.l == 0).unwrap();
        let node = s.r_node.expect("2s has a node");
        let e = tm_channel(&a.grid, 0, s.eps[0], &s.u[0], &a.v[0], 0.5 * node).expect_err("must refuse");
        eprintln!("{e}");
        assert!(e.contains("node"), "{e}");
    }

    /// Every first-row element with a core, and Al–Ar, constructs at every f of the ladder: norm
    /// conservation to 1e-10 per channel, the zero-curvature condition, and a finite V_ion with
    /// r_cmp reported (ρ_pc = ρ_c at Task 2).
    #[test]
    fn channels_construct_across_the_ladder() {
        for z in (4u8..=10).chain(13..=18) {
            let a = ground_state(z, SpinMode::Restricted).unwrap();
            let part = partition(&a).unwrap();
            let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
            let rho_c = core_density(&a, &part);
            for f in LEDGER.f_ladder {
                let chans = tm_channels(&a, &part, f).unwrap_or_else(|e| panic!("Z = {z}, f = {f}: {e}"));
                let mut line = format!("Z = {z:2}, f = {f}:");
                for (c, t) in part.channels.iter().zip(&chans) {
                    let dn = a.grid.partial(&sq(&t.u), t.i_c) - a.grid.partial(&sq(&c.u), t.i_c);
                    let lf = c.l as f64;
                    let curv = t.c[1] * t.c[1] + (2.0 * lf + 5.0) * t.c[2];
                    line += &format!(" l={}{} r_c {:.4} c2 {:+.5} Δnorm {dn:+.1e} curv {curv:+.1e};", c.l, if c.occupied { "" } else { "*" }, t.r_c, t.c[1]);
                    assert!(dn.abs() < 1e-10, "Z = {z}, f = {f}, l = {}: Δnorm {dn:.2e}", c.l);
                    assert!(curv.abs() <= 1e-12 * (t.c[1] * t.c[1]).max(1e-300), "Z = {z}, f = {f}, l = {}: {curv:.2e}", c.l);
                }
                let un = unscreen(&a.grid, part.z_v, &chans, &occ, &rho_c);
                let i_from = chans.iter().map(|c| c.i_c).max().unwrap() + 1;
                let tail = un.v_ion.iter().map(|v| (i_from..a.grid.len()).map(|i| (v[i] + part.z_v / a.grid.r[i]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
                line += &format!(" max|V_ion + Z_v/r| beyond max r_c {tail:.1e}, r_cmp {:.4}", un.r_cmp);
                eprintln!("{line}");
                assert!(un.v_ion.iter().flatten().all(|x| x.is_finite()));
            }
        }
    }
}

