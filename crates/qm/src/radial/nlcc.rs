//! The nonlinear core correction (spec §4.5) and the core radius of the overlap warning (spec §5.6).
//! CPU, f64, atomic units: densities in e/Bohr³, radii in Bohr.
//!
//! # Spec §4.5 (verbatim)
//! LSDA is nonlinear in the density, so E_xc[ρ_v] ≠ E_xc[ρ_v + ρ_c] − E_xc[ρ_c], and the error is largest
//! exactly where it hurts a spin-polarised calculation (the open-shell O triplet). Louie, Froyen & Cohen
//! (PRB 26, 1738 (1982)) keep a *partial* core density in the XC functional: ρ_pc(r) = ρ_c(r) for
//! r ≥ r_nlc, A sin(Br)/r for r < r_nlc with A, B fixed by value and slope continuity at r_nlc. Universal
//! rule for r_nlc: the radius where **ρ_c(r) = ρ_v(r)** (LFC's own criterion, factor 1; the crossover is a
//! property of the atom, not a number entered per element), taken as the outermost radius with ρ_c ≥ ρ_v
//! and with ρ_v the *all-electron* atom's valence density, which does not depend on f. In the molecule,
//! E_xc is evaluated on ρ_v,σ + ½ρ_pc per spin (the core is spin-unpolarised); ρ_pc never enters the
//! Hartree term, the electron count, or the displayed valence density (it is a labelled option in the
//! renderer, §6.5). It must be smooth enough for the grid, which the Fourier filter of §5.1 guarantees.
//!
//! # A and B (derivation)
//! Let R = r_nlc (a grid point: the outermost index with ρ_c ≥ `LEDGER.nlcc_crossover`·ρ_v and ρ_c > 0 —
//! a point where both densities have underflowed to 0 is not core-dominated), m(r) = A sin(Br)/r.
//! - Value: m(R) = ρ_c(R) ⇒ A = ρ_c(R)·R / sin(BR).
//! - Slope: m′/m = B cot(Br) − 1/r, so m′(R)/m(R) = ρ_c′(R)/ρ_c(R) ⇔ x cot x = κ, with x = BR and
//!   κ = 1 + R ρ_c′(R)/ρ_c(R).
//! - ρ_c falls monotonically outside its 1s maximum region, so ρ_c′(R) < 0 and κ < 1. On (0, π), x cot x
//!   falls monotonically from 1 (x → 0⁺: x cot x = 1 − x²/3 − …) to −∞ (x → π⁻), since
//!   d/dx(x cot x) = (sin x cos x − x)/sin²x < 0 there. So exactly one root x ∈ (0, π) exists for every
//!   κ < 1, found by bisection (200 halvings of (0, π): far below f64 resolution). sin x > 0 there, so A > 0
//!   and m > 0 on (0, R]: ρ_pc is positive, finite at the origin (m(0) = AB) and even in r (sin(Br)/r has
//!   only even powers), i.e. smooth at r = 0 — what the filter of §5.1 needs.
//! - ρ_c′ is the grid's 5-point derivative (`LogGrid::derivative`, O(δ⁴)); the tests check value and slope
//!   continuity against that same derivative.
//! - No core (H, He: ρ_c ≡ 0) or `NlccMode::Off` (the QMC switch, spec §3): ρ_pc ≡ 0, r_nlc = None (the
//!   explicit no-core branch, spec §4.2 item 4).
//!
//! # r_core (spec §5.6)
//! The smallest grid radius r_i with ∫_{r_i}^∞ 4πr²ρ_c dr ≤ `LEDGER.eps_core`·N_c, N_c = ∫₀^∞ 4πr²ρ_c dr
//! (both from one `LogGrid::cumulative`, so "all but ε_core of N_c inside" holds on that quadrature exactly);
//! 0 without a core. It is the frozen-core overlap radius — never r_cmp (a 1e-8 Ha potential criterion,
//! Task 2 review N7) and never r_nlc.
//!
//! # Method-convergence literals (review S2 practice)
//! The bisection bracket (1e-12, π − 1e-12) and its 200 halvings.
use super::grid::LogGrid;
use super::ledger::{NlccMode, LEDGER};
use std::f64::consts::PI;

/// The LFC partial core: ρ_pc on the grid, r_nlc (None without a core), the inner model's A and B, plus
/// the core radius r_core (§5.6) and the core charge N_c of the AE core density it was built from.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialCore { pub rho_pc: Vec<f64>, pub r_nlc: Option<f64>, pub a: f64, pub b: f64, pub r_core: f64, pub n_core: f64 }

fn shell(grid: &LogGrid, rho: &[f64]) -> Vec<f64> { grid.r.iter().zip(rho).map(|(r, x)| 4.0 * PI * r * r * x).collect() }

/// §5.6: the smallest grid r with ∫_r^∞ 4πr²ρ_c ≤ LEDGER.eps_core·N_c; 0 without a core.
pub fn core_radius(grid: &LogGrid, rho_c: &[f64]) -> f64 {
    let c = grid.cumulative(&shell(grid, rho_c));
    let n_c = c[grid.len() - 1];
    if n_c <= 0.0 { return 0.0; }
    let i = (0..grid.len()).find(|&i| n_c - c[i] <= LEDGER.eps_core * n_c).expect("the cumulative reaches N_c");
    grid.r[i]
}

/// LFC (module doc): ρ_pc = ρ_c for r ≥ r_nlc, A sin(Br)/r inside, A and B from value and slope continuity
/// (x cot x = κ). `rho_v` is the all-electron reference's valence density (f-independent). No core or
/// `NlccMode::Off`: ρ_pc ≡ 0 and r_nlc = None. An error (κ ≥ 1, no root: ρ_c not falling at r_nlc) is
/// refused by the recipe (`Refusal::PartialCore`, Task 4 review N2), not a panic.
pub fn partial_core(grid: &LogGrid, rho_c: &[f64], rho_v: &[f64]) -> Result<PartialCore, String> {
    let n = grid.len();
    let n_core = grid.integrate(&shell(grid, rho_c));
    let r_core = core_radius(grid, rho_c);
    if n_core <= 0.0 || LEDGER.nlcc == NlccMode::Off {
        return Ok(PartialCore { rho_pc: vec![0.0; n], r_nlc: None, a: 0.0, b: 0.0, r_core, n_core });
    }
    let i = (0..n).rev().find(|&i| rho_c[i] > 0.0 && rho_c[i] >= LEDGER.nlcc_crossover * rho_v[i])
        .ok_or("the core never dominates the valence density")?;
    let r = grid.r[i];
    let kappa = 1.0 + r * grid.derivative(rho_c)[i] / rho_c[i];
    if kappa >= 1.0 { return Err(format!("ρ_c is not falling at r_nlc = {r} (κ = {kappa}): x cot x = κ has no root in (0, π)")); }
    let (mut lo, mut hi) = (1e-12, PI - 1e-12);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if m / m.tan() > kappa { lo = m } else { hi = m }
    }
    let x = 0.5 * (lo + hi);
    let (b, a) = (x / r, rho_c[i] * r / x.sin());
    let rho_pc = (0..n).map(|j| if j < i { a * (b * grid.r[j]).sin() / grid.r[j] } else { rho_c[j] }).collect();
    Ok(PartialCore { rho_pc, r_nlc: Some(r), a, b, r_core, n_core })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::{core_density, ground_state, partition, valence_density, SpinMode, LEDGER};

    #[test]
    fn partial_core_is_continuous_with_slope_at_r_nlc() {
        let a = ground_state(8, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let (rc, rv) = (core_density(&a, &p), valence_density(&a, &p));
        let pc = partial_core(&a.grid, &rc, &rv).unwrap();
        let r = pc.r_nlc.expect("O has a core");
        let model = |x: f64| pc.a * (pc.b * x).sin() / x;
        let i = a.grid.nearest(r);
        assert!((model(r) - rc[i]).abs() < 1e-10 * rc[i]);
        let d = a.grid.derivative(&rc)[i];
        let dm = pc.a * (pc.b * (pc.b * r).cos() / r - (pc.b * r).sin() / (r * r));
        assert!((dm - d).abs() < 1e-8 * d.abs(), "slope {dm} vs {d}");
        // r_nlc is the definition — the outermost grid point with ρ_c ≥ ρ_v — checked as such, not as a
        // 5 % closeness (plan review NOTE: |d ln(ρ_c/ρ_v)/d ln r| ≈ 5 near 0.45 Bohr is ≈ 4 % per log-grid step).
        // Where both densities are exactly 0 (beyond the eigenfunctions' shooting end: O, the last 21 points,
        // r ≥ 50.76 Bohr, measured) "ρ_c ≥ ρ_v" holds as 0 ≥ 0 and says nothing; the brief's literal
        // `all(ρ_c < ρ_v)` fails there. The definition is read on points with core density (module doc), and
        // those trailing points must be joint zeros.
        assert!(rc[i] >= LEDGER.nlcc_crossover * rv[i], "ρ_c < ρ_v at r_nlc = {r}");
        assert!((i + 1..a.grid.len()).all(|j| rc[j] < LEDGER.nlcc_crossover * rv[j] || (rc[j] == 0.0 && rv[j] == 0.0)),
                "a later grid point has ρ_c ≥ ρ_v with density present");
        eprintln!("O: r_nlc = {r:.4} Bohr, A = {:.6}, B = {:.6} (x = BR = {:.6}), ρ_pc(0) = AB = {:.4} vs ρ_c(0) = {:.1}; N_c = {:.10}",
                  pc.a, pc.b, pc.b * r, pc.a * pc.b, rc[0], pc.n_core);
    }

    /// The explicit no-core branch: H has ρ_pc ≡ 0, r_nlc undefined, r_core = 0 (spec §4.2 item 4, §5.6).
    #[test]
    fn no_core_means_no_partial_core() {
        let a = ground_state(1, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let pc = partial_core(&a.grid, &core_density(&a, &p), &valence_density(&a, &p)).unwrap();
        assert!(pc.rho_pc.iter().all(|&x| x == 0.0) && pc.r_nlc.is_none() && pc.r_core == 0.0);
    }

    /// r_core is its definition (§5.6): all but ε_core of N_c inside it, not so one grid point earlier.
    #[test]
    fn core_radius_is_its_definition() {
        let a = ground_state(8, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let rc = core_density(&a, &p);
        let g = &a.grid;
        let r_core = core_radius(g, &rc);
        let shell: Vec<f64> = g.r.iter().zip(&rc).map(|(r, x)| 4.0 * std::f64::consts::PI * r * r * x).collect();
        let c = g.cumulative(&shell);
        let n_c = c[g.len() - 1];
        let i = g.nearest(r_core);
        assert!(n_c - c[i] <= LEDGER.eps_core * n_c && n_c - c[i - 1] > LEDGER.eps_core * n_c);
        eprintln!("O: r_core = {r_core:.4} Bohr (spec §5.6 hydrogenic estimate ≈ 0.7)");
    }

    /// Task 2's tail identity re-run with the LFC partial core: beyond max(r_c, r_nlc), ρ_pc = ρ_c, so V_ion + Z_v/r is
    /// the full core's Hartree tail (spec §4.3).
    #[test]
    fn v_ion_tail_is_the_core_hartree_tail_with_lfc() {
        let t = crate::radial::generate(8).unwrap();
        let g = &t.grid;
        let n = g.len();
        let i_from = t.channels.iter().map(|c| c.i_c).max().unwrap().max(g.nearest(t.partial_core.r_nlc.unwrap())) + 1;
        let occ: Vec<f64> = t.partition.channels.iter().map(|c| c.occ).collect();
        let un = crate::radial::tm::unscreen(g, t.z_v, &t.channels, &occ, &t.partial_core.rho_pc);
        let rho_c = crate::radial::core_density(&t.reference, &t.partition);
        let vhc = crate::radial::hartree_potential(g, &rho_c);
        let tail = |i: usize| vhc[i] - t.n_core as f64 / g.r[i];
        for (k, v) in un.v_ion.iter().enumerate() {
            for i in i_from..n {
                let d = v[i] + t.z_v / g.r[i] - tail(i);
                assert!(d.abs() < crate::radial::LEDGER.v_ion_tail / 100.0, "channel {k}, r = {:.4}: V_ion + Z_v/r − τ_c = {d:.3e}", g.r[i]);
            }
        }
        // Task 2's r_cmp checks: it bounds the tail, it is minimal, and it is the core tail's 1e-8 radius (±1 point)
        // when that lies beyond i_from
        let tol = crate::radial::LEDGER.v_ion_tail;
        let dev = |v: &[f64], i: usize| (v[i] + t.z_v / g.r[i]).abs();
        let i_cmp = g.nearest(un.r_cmp);
        assert!((i_cmp..n).all(|i| un.v_ion.iter().all(|v| dev(v, i) < tol)), "r_cmp does not bound the tail");
        assert!(i_cmp > 0 && un.v_ion.iter().any(|v| dev(v, i_cmp - 1) >= tol), "r_cmp is not minimal");
        let i_tail = (0..n).rev().find(|&i| tail(i).abs() >= tol).map_or(0, |i| i + 1);
        if i_tail > i_from {
            assert!(i_cmp.abs_diff(i_tail) <= 1, "r_cmp {} vs the core tail's radius {}", un.r_cmp, g.r[i_tail]);
        } else {
            assert!(i_cmp <= i_from, "r_cmp {} beyond the first point past max(r_c, r_nlc)", un.r_cmp);
        }
        eprintln!("O with LFC: r_nlc {:.4}, first point past max(r_c, r_nlc) {:.4}, r_cmp {:.4}, core tail's 1e-8 radius {:.4}",
                  t.partial_core.r_nlc.unwrap(), g.r[i_from], un.r_cmp, g.r[i_tail]);
    }
}
