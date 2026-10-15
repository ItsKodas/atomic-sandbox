//! The spherical-Bessel filter (spec §5.1, §4.8 T8; plan review M2/Q2, M3/Q3, M4, Q11): for a grid of spacing h,
//! the tables an ion contributes — the one short-range local table V_SR, the KB projectors β_l and the partial core
//! ρ_pc — band-limited for that grid and sampled from uniform 3-D tables, T8 at every ladder h (the filtered T1/T3
//! re-run, the LSDA domain, the table egg-box force) and h_req. CPU, f64, atomic units (Hartree, Bohr, Ha/Bohr); V is
//! the electron's potential energy (V_LR < 0). Task 5 physics review (2026-10-15):
//! `.superpowers/sdd/2026-10-15-m7-pseudopotentials/task-5-ruling.md`.
//!
//! # The local split (spec §5.1; Spec question 2 as the plan review ruled, M2)
//! a = σ_LR√2, σ_LR = `LEDGER.sigma_lr_over_h`·h. Exactly, V_loc(d) = V_LR(d) + V_SR(d) with
//!   V_LR(d) = −Z_v erf(d/a)/d   (analytic per centre on the grid, never tabulated; −2Z_v/(√π a) at d = 0),
//!   V_SR(d) = V_loc(d) + Z_v erf(d/a)/d   (ONE table).
//! With ĝ(q) = ∫g e^{−iq·r} d³r: FT[−Z_v erf(d/a)/d] = −4πZ_v e^{−q²a²/4}/q² (the Gaussian-smeared Coulomb). The earlier
//! split V_cmp = V_loc + Z_v/d, V_tail = −Z_v erfc(d/a)/d was singular piece by piece — V_loc is finite at 0, so V_cmp
//! carries +Z_v/d and V_tail −Z_v/d — and V_tail was not band-limited by σ_LR: FT[−Z_v erfc(d/a)/d] =
//! −4πZ_v(1 − e^{−q²a²/4})/q² is the *complement* of the Gaussian part and decays only as q⁻². Their sum V_SR is smooth
//! and finite at 0 (V_loc even and finite, erf(d/a)/d → 2/(√π a)). Beyond r_cmp, V_loc = −Z_v/d + τ_c to
//! `v_ion_tail` (spec §4.3; |τ_c| < v_ion_tail there), so V_SR = −Z_v erfc(d/a)/d to v_ion_tail. Its support is
//! R_SR = max(r_cmp, r_floor), r_floor the outermost grid radius with Z_v erfc(r/a)/r ≥ `v_tail_floor` (erfc(x)/x falls
//! monotonically): beyond R_SR, |V_SR| < v_tail_floor + v_ion_tail, so truncating there discards nothing of the exact
//! split. Derived from two ledger fields; no new constant.
//! V_LR's weight above q_c is e^{−(q_c a/2)²}, q_c a/2 = (0.7π/h)(2√2h)/2 = 0.7π√2: e^{−9.67} ≈ 6.3e-5. It reaches the
//! re-run because `FilteredTable::input` sums v_lr + v_sr. Its alias at 2π/h is ∝ e^{−(πa/h)²} = e^{−8π²}.
//!
//! # The transform pair
//! For g(r) = F(r)Y_lm(r̂): ĝ(q) = (−i)^l Y_lm(q̂)F̂(q) with F̂(q) = 4π∫₀^∞F(r)j_l(qr)r²dr and F(r) =
//! (1/2π²)∫₀^∞F̂(q)j_l(qr)q²dq (from e^{iq·r} = 4πΣ_lm i^l j_l(qr)Y*_lm(q̂)Y_lm(r̂)). So the 3-D support of ĝ is F̂'s
//! radial support, for every l, and a finite sum Σ_k c_k j_l(q_k r) with every q_k ≤ q_max is band-limited to q_max
//! *exactly*, whatever the quadrature weights: the quadrature decides how faithfully the kept part represents F (the
//! re-run measures that), never whether the result is band-limited.
//! The u-form ↔ 3-D convention: F_l = β_u/r (β_u = δV·u, the radial KB projector of `kb`), t_l = F_l/r^l, and the grid
//! function β_lm(r) = t_l(d)·√((2l+1)/4π)·R_lm(d) = F_l(d)Y_lm (Racah R_lm = √(4π/(2l+1)) r^l Y_lm, `poisson`), so
//! ⟨β_lm|ψ⟩ = ∫F_l (u/r) r² dr = ∫β_u u dr for ψ = (u/r)Y_lm: the same E_KB. `input` inverts it, β_u = r^{l+1}·t_l.
//!
//! # Poisson summation and the shifts (spec §8 row 5; plan review M4)
//! h³Σ_{p ∈ ℤ³} g(hp − R) = Σ_{G ∈ (2π/h)ℤ³} ĝ(G)e^{−iG·R}. If ĝ(G) = 0 for every G ≠ 0 the sum is ĝ(0) for every R:
//! translation invariant. For an inversion-even g it is Σ_G ĝ(G)cos(G·R). The first shell (G = ±(2π/h)ê_a)
//! contributes 2ĝ(G₁)Σ_a cos(2πR_a/h), exactly 0 at every R_a = ±h/4, so those 8 shifts see only the √2 and √3 shells
//! — and for a spherical g all 8 are images of one another under the cubic group, so their spread is exactly 0.
//! Between R = 0 and R = (h/2, h/2, h/2) (cos = ±1) the first shell moves the sum by 12ĝ(G₁), the √2 shell by 0 and
//! the √3 shell by 16ĝ(G√3): the {0, h/2}³ spread is ≈ 12|ĝ(G₁)| + 16|ĝ(G√3)| relative to ĝ(0), i.e. 12–30× a
//! per-G ratio. A product of two functions band-limited to q_max (β², the projector norm) is band-limited to
//! 2q_max = 2π/h, the first shell's radius, where the convolution's overlap volume vanishes.
//! The weight-1 sum h³Σ_p g is not the judge (the plan's "1e-8 lattice-sum standard" was never derived and is
//! withdrawn, Task 5 physics review): it weighs the truncation zone, where the fields a table multiplies are 10⁻³–10⁻⁴ of
//! their central values, as heavily as the centre, so it is neither the energy nor the force. The judge is T8(d) below.
//!
//! # King-Smith, Payne & Lin's band completion (PRB 44, 13063 (1991); Spec question 3 as the plan review ruled, M3)
//! For each table g ∈ {V_SR, β_l, ρ_pc} with support R (R_SR; r_β = max(r_c, r_nlc) for every β_l, spec §4.6, never a
//! 1e-12 rule; for ρ_pc the XC-weight radius: the smallest r with 4π∫_r^∞|v̄_xc|ρ_pc r′²dr′ ≤ `v_tail_floor`, v̄_xc =
//! ½(v_xc↑ + v_xc↓) of the unfiltered reference KB atom on ρ_v + ρ_pc — the first-order XC energy of the discarded
//! tail, the energy floor V_SR's support already uses; h-independent like r_β; review S1):
//! 1. F̂_<(q) = 4π∫F j_l(qr) r² dr on [0, q_c], q_c = `q_c_over_nyquist`·π/h (`LogGrid::integrate`). Kept.
//! 2. On (q_c, q_max], q_max = π/h (the grid Nyquist, derived): F̂_b(q) = Σ_{c<m} x_c T_c(s), s = 2(q − q_c)/(q_max −
//!    q_c) − 1, m = `numerics.kspl_terms`. x minimises J(x) = ∫_{r_cut}^{r_max}|F_< + F_b|² r² dr, r_cut = γ_r R: rows
//!    r_j = r_cut + jΔr with weight √(r_j²Δr), Δr = π/(4q_max) = h/4 (the band's shortest wavelength, 2π/q_max = 2h,
//!    sampled 8 times) — the linear least-squares problem ‖Ax − b‖, b = −F_< on the rows.
//! 3. The result is truncated by the ψ-step over [r_cut, r_cut + w], w = `numerics.kspl_step_over_h`·h = 8h (below);
//!    the table's radius is γ_r R + 8h.
//!
//! The solve — deviation from the brief's SVD floor 1e-12·σ_max (measured): the discrete Picard truncation. With
//! A = UΣVᵀ, direction k enters x only if |u_kᵀb| exceeds b's rounding level ‖δb‖ and σ_k exceeds the numerical-rank
//! floor ε·max(rows, m)·σ_max, where δb_j = e_F·√(r_j²Δr)/(2π²)·Σ_k w_k q_k²|j_l(q_k r_j)| and e_F = ε·4π∫|F|r²dr bounds
//! a transform's rounding (|j_l| ≤ 1). Why: when every band function is concentrated inside r_cut (the Gaussian
//! round trip of the tests, r_cut = 50), b is pure rounding (‖b‖ = 7.2e-16), the 1e-12 floor kept σ down to 2e-12, and
//! x (|x| = 1.1e-5) moved F(0) by 1.4e-3 without lowering J. A direction on which b's projection is at its rounding
//! level carries no information about the tail. On the real tables every direction passes (|u_kᵀb| ≫ ‖δb‖ ≤ 6e-14),
//! so there the result is the plain least-squares solution.
//!
//! # The outer truncation (placement confirmed; width 8h — Task 5 physics review, items 6 and M2)
//! The ψ-step acts on [γ_r R, γ_r R + w], outside γ_r R: the least squares minimises the tail from γ_r R on, so the
//! step acts where the optimised tail is small (review Q3: "keep the ψ-step only as the outer truncation"). A step over
//! [γ_r R − h, γ_r R] acts where nothing was minimised; measured 2–7× worse on every table. The *width* sets the
//! leakage: the step's spectrum beyond π/h is a function of w·π/h only, so w is a pure number times h, and
//! `numerics.kspl_step_over_h` = 8 is the smallest of {1, 2, 4, 8} that brings every accepted element under T8(d) at
//! its h_req (C and N at 0.30 need 8). Measured here: O's F_NL at h = 0.20 is 8.0e-5 Ha/Bohr with w = h
//! (`table_eggbox_probe_sees_a_one_spacing_step`) and 9.3e-7 with 8h.
//! The step is ψ(1 − t)/(ψ(t) + ψ(1 − t)), ψ(t) = e^{−1/t}, t = (r − r_in)/(r_out − r_in): C^∞, no parameter.
//! F̂_b(q_max) = 0 is not imposed: a jump inside the band leaves the function band-limited, and the review measured
//! no gain from it.
//!
//! # The q mesh — deviation from the brief: dq ≤ π/(2r_max), not π/r_max
//! Simpson, uniform, an even number of intervals, on [0, q_c] and [q_c, q_max] separately (F̂ jumps at q_c). A
//! uniform-q trapezoid sum makes rF(r) a sine series of period 2π/dq, so rF recurs (an odd image) at r ≈ 2π/dq; Simpson
//! = (4T_dq − T_2dq)/3, and T_2dq's period is π/dq: an image of weight −1/3 at r ≈ π/dq. The brief's dq ≤ π/r_max
//! (whose "alias-free out to 2r_max" is the trapezoid's statement) put that image at r_max, inside the least-squares
//! window, where r² weighs it ~900× the near tail: measured 2.2e-3 at r = 59.5 for a unit Gaussian (exact 0), and the
//! solution then moved F(0) by 0.76. With dq ≤ π/(2r_max) the image sits at ≥ 2r_max, beyond every row and the whole
//! grid (`simpson_images_lie_beyond_the_grid`). Derived from the ledger's r_max; no new constant.
//!
//! # 3-D sampling (spec §5.1 "Sampling"; Task 5 physics review, MUST M3)
//! The grid samples every table (V_SR, t_l, ρ_pc and their radial derivatives) only through the uniform tables
//! (`*_u`): spacing Δ = h/`numerics.table_points_per_h` = h/256, filled with the band-limited function evaluated exactly
//! (`Completion::value_at`, the finite Bessel sum times the step), read by 4-point Lagrange; the force accessors return
//! the derivative of the same interpolant. Each table is even in d, so index −1 mirrors index 1 and the stencil is
//! centred down to d = 0. Bound: on the central interval |g − p₃| ≤ (9/16)(Δ⁴/4!)·max|g⁗|; restricted to a ray a 3-D
//! function band-limited to |q| ≤ π/h is a 1-D function band-limited to π/h, so Bernstein gives |∂⁴g| ≤ (π/h)⁴max|g|,
//! hence ≤ (9/16)(π/256)⁴/24 = 5.3e-10 of max|g| (measured ≤ 1.8e-11, `uniform_tables_match_the_band_limited_function`).
//! The log grid (δ = 0.008) is never interpolated in 3-D: its 4-point interpolation alone gives F_SR = 1.4–4.7e-6
//! Ha/Bohr, above T8(d)'s whole budget. The log-grid arrays feed only the radial re-run, at their exact node values.
//!
//! # T8: the filtered re-run (S2), the LSDA domain (S3) and the table egg-box force (spec §4.8)
//! At every ladder h, on the filtered tables (`FilterRow`, `FilterRow::passes`):
//! - (a) T1: `de` = E_filt − E_unfilt at the reference configuration and `max_eig` = max over the occupied channels of
//!   |ε_filt − ε_unfilt|, each < `h_req_tol` (T1's 1 mHa: the filter must not degrade the potential below its own
//!   acceptance standard).
//! - (b) T3 (S2): `t3_max` = max over the gated T3 configurations of |ΔE_PP^filt − ΔE_AE| < `t3_delta_e`. ΔE_AE is the
//!   gate's own number (`GateReport::t3`; the all-electron atom is deterministic and not re-solved). The unfiltered KB
//!   atom and the Hund multiplet are solved once per table (`Rerun`), not once per h.
//! - (c) The LSDA domain (S3): ρ_pc is a model density entering only E_xc, through n_σ = ρ_v,σ + ½ρ_pc; PW92 and Dirac
//!   are defined on n_σ ≥ 0. A band-limited ρ_pc rings (interior Gibbs lobes from cutting the core tail's spectrum at
//!   q_c, up to −1.5 % (O) and −4 % (Ne) of its maximum), and that is allowed; what is not is n_σ < 0, where
//!   `qm::xc::exc_point`'s clamp engages and E_xc stops being differentiable in ρ_pc (F^pc would no longer be −∂E/∂R).
//!   `clamped` counts the (configuration, spin, r) points with ρ_pc < 0 and n_σ < 0 over the reference and every T3
//!   configuration ([`xc_domain_violations`]); it must be 0. A positivity-preserving window (a nonnegative kernel must
//!   damp the band below q_c the core needs) was measured at 19–270 mHa and rejected (review Table D).
//! - (d) The table egg-box force ([`table_eggbox`]). Lattice Parseval: h³Σ_p w_p g(r_p − R) = ∫w̃(r)g(r − R)d³r exactly
//!   when g is band-limited inside the zone (w̃ the zone-limited interpolant of w_p), so sampling adds nothing; the
//!   table's leakage g_> beyond π/h does, to first order h³Σ_p w(r_p − R)g_>(r_p − R) with w the field the table
//!   multiplies in the energy (spec §5.2): ρ_v for V_SR; the channel orbital φ_l for β_l through E_NL =
//!   Σ_m (f_l/(2l+1))E_KB c_lm², c_lm = h³Σ_p β_lm φ_lm (an unoccupied channel: its reference pseudo-wavefunction, f = 1);
//!   v̄_xc = ½(v_xc↑ + v_xc↓) for ρ_pc. The probe takes w from the filtered KB atom at the reference configuration, hard
//!   cut at |q| = π/h (so w_< g_< is band-limited to 2π/h and exactly invariant: what remains is exactly the leakage),
//!   reads g through the uniform tables, and returns F_g = max over R ∈ {±h/4}³ (the §6.6 probe: sin(2πR_a/h) = ±1) of
//!   |∇_R h³Σ_p w g|. Budget: §6.6 gates |Σ_I F_I| < 3e-5 Ha/Bohr; one third to the tables (the rest is the valence
//!   floor, which no table controls), shared over the ≤ 5 atoms of M7's molecules: F_SR + F_NL + F_pc <
//!   `gates.h_req_force` = 2e-6 Ha/Bohr.
//! T8 (Spec question 11): h_req = the coarsest ladder h such that it and every finer ladder h pass (a)–(d); None when
//! even the finest fails — an outcome the record reports (a resolution limit, not a refusal of the potential); a scene
//! finer than the ladder is certified by (a)–(d) at its own h.
//!
//! # Method literals (review S2 practice; none moves a result beyond its own tolerance)
//! The identity filter's 1e-12-of-maximum cut (its V_SR and ρ_pc); d < 1e-10 for V_LR's on-site value; the Bessel
//! series' 10 terms for |x| < 1 (the next term is below 1e-23); the Picard and rank floors above (ε = f64::EPSILON); the
//! egg-box probe's central-difference step 2e-4 Bohr (the review's; the gradient of the sampled interpolant, its
//! O(step²) error relative ≈ (2e-4·2π/h)² ≤ 7e-5 of a force that is itself compared with a budget).
use super::atom::{ground_state, AtomResult, SpinMode};
use super::gates::{transfer_configs, TransferConfig};
use super::grid::LogGrid;
use super::kb::{reference_config, solve_kb_atom, KbAtom, KbInput};
use super::ledger::LEDGER;
use super::record::{FilterRow, GenerationRecord};
use super::recipe::PseudoTable;
use crate::grid::neumaier;
use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
use crate::xc::exc_point;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

/// j_l(x)/x^l, finite at 0: the series Σ_k (−x²/2)^k/(k!(2l+2k+1)!!) for |x| < 1, else `sph_bessel`/x^l.
pub fn sph_bessel_scaled(l: u8, x: f64) -> f64 {
    if x.abs() < 1.0 {
        let mut df = 1.0;
        for k in 1..=l as usize { df *= (2 * k + 1) as f64; }
        let (mut term, mut sum) = (1.0 / df, 1.0 / df);
        for k in 1..10usize {
            term *= -x * x / (2.0 * k as f64 * (2 * l as usize + 2 * k + 1) as f64);
            sum += term;
        }
        return sum;
    }
    sph_bessel(l, x) / x.powi(l as i32)
}

/// The spherical Bessel function j_l(x): closed forms for l ≤ 3 at |x| ≥ 1 (upward recurrence beyond), the
/// series below.
pub fn sph_bessel(l: u8, x: f64) -> f64 {
    if x.abs() < 1.0 { return sph_bessel_scaled(l, x) * x.powi(l as i32); }
    let (s, c) = (x.sin(), x.cos());
    match l {
        0 => s / x,
        1 => s / (x * x) - c / x,
        2 => (3.0 / (x * x) - 1.0) * s / x - 3.0 * c / (x * x),
        3 => (15.0 / x.powi(3) - 6.0 / x) * s / x - (15.0 / (x * x) - 1.0) * c / x,
        _ => {
            let (mut a, mut b) = (sph_bessel(2, x), sph_bessel(3, x));
            for k in 3..l { let n = (2 * k + 1) as f64 / x * b - a; a = b; b = n; }
            b
        }
    }
}

fn psi(t: f64) -> f64 { if t > 0.0 { (-1.0 / t).exp() } else { 0.0 } }

/// The C^∞ step ψ(1−t)/(ψ(t) + ψ(1−t)), t = (r − r_in)/(r_out − r_in), ψ(t) = e^{−1/t}: 1 on r ≤ r_in, 0 on
/// r ≥ r_out, no parameter.
pub fn mask(r: f64, r_in: f64, r_out: f64) -> f64 {
    if r <= r_in { return 1.0; }
    if r >= r_out { return 0.0; }
    let t = (r - r_in) / (r_out - r_in);
    psi(1.0 - t) / (psi(t) + psi(1.0 - t))
}

/// A uniform Simpson mesh on [a, b] with an even number of intervals and dq ≤ π/(2r_max) (module doc, "The q mesh"):
/// (nodes, weights).
fn q_mesh(a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 2 * (((b - a) * LEDGER.r_max / PI).ceil() as usize).max(1);
    let dq = (b - a) / n as f64;
    let q = (0..=n).map(|k| a + k as f64 * dq).collect();
    let w = (0..=n).map(|k| dq / 3.0 * if k == 0 || k == n { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 }).collect();
    (q, w)
}

/// Chebyshev T_0..T_{m−1} at s ∈ [−1, 1].
fn chebyshev(s: f64, m: usize) -> Vec<f64> {
    let mut t = vec![1.0; m];
    if m > 1 { t[1] = s; }
    for k in 2..m { t[k] = 2.0 * s * t[k - 1] - t[k - 2]; }
    t
}

/// King-Smith, Payne & Lin's band completion (module doc): F̂ kept on [0, q_c], the `kspl_terms`-term Chebyshev band on
/// (q_c, q_max] minimising ∫_{r_cut}^{r_max}|F|²r²dr (discrete Picard truncation), the ψ-step over [r_cut, r_cut + width].
/// Returns F^filt/r^l on the grid (finite at 0). Transform pair: F̂(q) = 4π∫F j_l(qr) r² dr, F(r) = (1/2π²)∫F̂ j_l(qr) q² dq.
pub fn filter_radial(grid: &LogGrid, f: &[f64], l: u8, q_c: f64, q_max: f64, r_cut: f64, width: f64) -> Vec<f64> {
    let c = complete(grid, f, l, q_c, q_max, r_cut, width);
    grid.r.iter().map(|&r| c.value_at(r)).collect()
}

/// The completed function as a finite Bessel sum: F^filt(r)/r^l = (1/2π²)·Σ_k c_k q_k^{l+2} j_l(q_k r)/(q_k r)^l times the
/// outer step (c_k = quadrature weight × F̂(q_k)). `value_at` is exact at any r; the grid table interpolates it.
struct Completion { l: u8, q: Vec<f64>, c: Vec<f64>, r_cut: f64, width: f64 }

impl Completion {
    fn value_at(&self, r: f64) -> f64 {
        let m = mask(r, self.r_cut, self.r_cut + self.width);
        if m == 0.0 { return 0.0; }
        let li = self.l as i32;
        neumaier(self.q.iter().zip(&self.c).map(|(&q, &c)| c * q.powi(li + 2) * sph_bessel_scaled(self.l, q * r))) / (2.0 * PI * PI) * m
    }
}

fn complete(grid: &LogGrid, f: &[f64], l: u8, q_c: f64, q_max: f64, r_cut: f64, width: f64) -> Completion {
    let n = grid.len();
    let (qk, wk) = q_mesh(0.0, q_c);
    let (qb, wb) = q_mesh(q_c, q_max);
    let fk: Vec<f64> = qk.iter().map(|&q| 4.0 * PI * grid.integrate(&(0..n).map(|i| f[i] * sph_bessel(l, q * grid.r[i]) * grid.r[i] * grid.r[i]).collect::<Vec<_>>())).collect();
    let m = LEDGER.numerics.kspl_terms;
    let cheb: Vec<Vec<f64>> = qb.iter().map(|&q| chebyshev(2.0 * (q - q_c) / (q_max - q_c) - 1.0, m)).collect();
    // least squares over the tail: rows r_j ∈ [r_cut, r_max], spacing π/(4q_max), weight √(r_j²Δr)
    let dr = PI / (4.0 * q_max);
    let rs: Vec<f64> = (0..).map(|j| r_cut + j as f64 * dr).take_while(|&r| r <= LEDGER.r_max).collect();
    // a transform's rounding: |δF̂(q)| ≤ ε·4π∫|F|r² dr (|j_l| ≤ 1), propagated to each row of b
    let e_f = f64::EPSILON * 4.0 * PI * grid.integrate(&(0..n).map(|i| f[i].abs() * grid.r[i] * grid.r[i]).collect::<Vec<_>>());
    let mut a = DMatrix::<f64>::zeros(rs.len(), m);
    let mut b = DVector::<f64>::zeros(rs.len());
    let mut noise2 = 0.0;
    for (j, &r) in rs.iter().enumerate() {
        let sw = (r * r * dr).sqrt() / (2.0 * PI * PI);
        b[j] = -sw * neumaier((0..qk.len()).map(|k| wk[k] * fk[k] * qk[k] * qk[k] * sph_bessel(l, qk[k] * r)));
        noise2 += (e_f * sw * (0..qk.len()).map(|k| wk[k] * qk[k] * qk[k] * sph_bessel(l, qk[k] * r).abs()).sum::<f64>()).powi(2);
        for k in 0..qb.len() {
            let jb = sw * wb[k] * qb[k] * qb[k] * sph_bessel(l, qb[k] * r);
            for c in 0..m { a[(j, c)] += jb * cheb[k][c]; }
        }
    }
    // the discrete Picard truncation (module doc, deviation from the brief's 1e-12·σ_max floor)
    let svd = a.svd(true, true);
    let (u, vt) = (svd.u.as_ref().expect("U computed"), svd.v_t.as_ref().expect("Vᵀ computed"));
    let rank_floor = f64::EPSILON * rs.len().max(m) as f64 * svd.singular_values.max();
    let noise = noise2.sqrt();
    let mut x = DVector::<f64>::zeros(m);
    for k in 0..svd.singular_values.len() {
        let (sk, ub) = (svd.singular_values[k], u.column(k).dot(&b));
        if sk > rank_floor && ub.abs() > noise { x += vt.row(k).transpose() * (ub / sk); }
    }
    let fb: Vec<f64> = cheb.iter().map(|t| t.iter().zip(x.iter()).map(|(tc, xc)| tc * xc).sum::<f64>()).collect();
    let q = qk.iter().chain(&qb).copied().collect();
    let c = (0..qk.len()).map(|k| wk[k] * fk[k]).chain((0..qb.len()).map(|k| wb[k] * fb[k])).collect();
    Completion { l, q, c, r_cut, width }
}

/// A 3-D sampling table (spec §5.1 "Sampling"): f at d = k·du, k = 0, 1, …, out to `radius` plus the 4-point stencil.
fn uniform_fill(du: f64, radius: f64, f: impl Fn(f64) -> f64) -> Vec<f64> {
    let n = (radius / du).ceil() as usize + 4;
    (0..n).map(|k| f(k as f64 * du)).collect()
}

/// 4-point Lagrange on a uniform table of an even function (index −1 mirrors index 1, so the stencil is centred down
/// to d = 0 and the central-interval bound holds everywhere): (value, d/dd of the same interpolant). Zero beyond the table.
fn uniform_lagrange(f: &[f64], du: f64, d: f64) -> (f64, f64) {
    let s = d / du;
    let i0 = s.floor() as i64 - 1;
    if i0 + 3 >= f.len() as i64 { return (0.0, 0.0); }
    let at = |i: i64| f[i.unsigned_abs() as usize];
    let t = s - i0 as f64;
    let (a, b, c, e) = (at(i0), at(i0 + 1), at(i0 + 2), at(i0 + 3));
    let l = [-(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0, t * (t - 2.0) * (t - 3.0) / 2.0,
             -t * (t - 1.0) * (t - 3.0) / 2.0, t * (t - 1.0) * (t - 2.0) / 6.0];
    let dl = [-((t - 2.0) * (t - 3.0) + (t - 1.0) * (t - 3.0) + (t - 1.0) * (t - 2.0)) / 6.0,
              ((t - 2.0) * (t - 3.0) + t * (t - 3.0) + t * (t - 2.0)) / 2.0,
              -((t - 1.0) * (t - 3.0) + t * (t - 3.0) + t * (t - 1.0)) / 2.0,
              ((t - 1.0) * (t - 2.0) + t * (t - 2.0) + t * (t - 1.0)) / 6.0];
    (a * l[0] + b * l[1] + c * l[2] + e * l[3], (a * dl[0] + b * dl[1] + c * dl[2] + e * dl[3]) / du)
}

/// One filtered projector: `t` = F_l^filt/r^l on the log grid (the radial re-run), `t_u` the same on the uniform 3-D
/// table, E_KB unchanged, `radius` the table's extent γ_r·r_β + 8h (r_β for the identity filter).
#[derive(Clone, Debug)]
pub struct FilteredProjector { pub l: u8, pub e_kb: f64, pub t: Vec<f64>, pub t_u: Vec<f64>, pub radius: f64 }

/// The tables the grid samples for one element at one h (module doc). The log-grid arrays (`v_sr_table`, `t`, `rho_pc`)
/// feed the radial re-run at their exact node values; the 3-D grid reads only the uniform tables (`*_u`, spacing
/// `du` = h/`table_points_per_h`, index 0 at d = 0) through the accessors. `check` is T8 at this h; `record` the
/// element's generation record.
#[derive(Clone, Debug)]
pub struct FilteredTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub h: f64, pub sigma_lr: f64,
    pub v_sr_table: Vec<f64>, pub r_sr: f64,
    pub projectors: Vec<FilteredProjector>, pub rho_pc: Vec<f64>, pub pc_radius: f64, pub r_core: f64,
    pub du: f64, pub v_sr_u: Vec<f64>, pub rho_pc_u: Vec<f64>,
    pub check: FilterRow, pub record: GenerationRecord, pub grid: LogGrid,
}

impl FilteredTable {
    fn a(&self) -> f64 { self.sigma_lr * 2f64.sqrt() }
    /// V_LR(d) = −Z_v erf(d/a)/d, a = σ_LR√2 (analytic; −2Z_v/(√π a) at d = 0).
    pub fn v_lr(&self, d: f64) -> f64 {
        let a = self.a();
        if d < 1e-10 { -self.z_v * 2.0 / (PI.sqrt() * a) } else { -self.z_v * libm::erf(d / a) / d }
    }
    /// dV_LR/dd = −Z_v[(2/√π)(d/a)e^{−(d/a)²} − erf(d/a)]/d² (0 at d = 0, V_LR being even).
    pub fn dv_lr(&self, d: f64) -> f64 {
        if d < 1e-10 { return 0.0; }
        let x = d / self.a();
        -self.z_v * ((2.0 / PI.sqrt()) * x * (-x * x).exp() - libm::erf(x)) / (d * d)
    }
    fn read(&self, f: &[f64], radius: f64, d: f64) -> (f64, f64) { if d > radius { (0.0, 0.0) } else { uniform_lagrange(f, self.du, d) } }
    /// The one short-range table (plan review Q2) read from its uniform table, and the exact derivative of the same
    /// interpolant (so a force taken with `dv_sr` is the derivative of the energy sampled with `v_sr`).
    pub fn v_sr(&self, d: f64) -> f64 { self.read(&self.v_sr_u, self.r_sr, d).0 }
    pub fn dv_sr(&self, d: f64) -> f64 { self.read(&self.v_sr_u, self.r_sr, d).1 }
    pub fn sr_radius(&self) -> f64 { self.r_sr }
    /// t_k(d) = F_l^filt(d)/d^l of projector k; the grid function is β_lm = t_k·√((2l+1)/4π)·R_lm (spec §4.6 S1).
    pub fn beta_radial(&self, k: usize, d: f64) -> f64 { let p = &self.projectors[k]; self.read(&p.t_u, p.radius, d).0 }
    pub fn dbeta_radial(&self, k: usize, d: f64) -> f64 { let p = &self.projectors[k]; self.read(&p.t_u, p.radius, d).1 }
    pub fn rho_pc_at(&self, d: f64) -> f64 { self.read(&self.rho_pc_u, self.pc_radius, d).0 }
    pub fn drho_pc_at(&self, d: f64) -> f64 { self.read(&self.rho_pc_u, self.pc_radius, d).1 }

    /// The radial KB atom's input on these tables, at the exact log-node values (not the uniform tables): V_loc = V_LR +
    /// V_SR, β_u = r^{l+1}·t_l (the u-form ↔ 3-D convention F_l = β_u/r, t_l = F_l/r^l, inverted), ρ_pc.
    pub fn input(&self) -> KbInput {
        let r = &self.grid.r;
        KbInput {
            z: self.z, z_v: self.z_v,
            v_loc: r.iter().zip(&self.v_sr_table).map(|(&d, &v)| self.v_lr(d) + v).collect(),
            projectors: self.projectors.iter()
                .map(|p| (p.l, p.e_kb, r.iter().zip(&p.t).map(|(&d, &t)| d.powi(p.l as i32 + 1) * t).collect())).collect(),
            rho_pc: self.rho_pc.clone(),
        }
    }

    /// The identity filter: the same split and conventions, no band limit (Task 6's S1 test). Its uniform tables
    /// interpolate the log grid (an unfiltered table is not band-limited, so there is nothing exact to sample).
    pub fn unfiltered(table: &PseudoTable, h: f64) -> FilteredTable { build(table, h, false, &Rerun::of(table), LEDGER.numerics.kspl_step_over_h) }
}

/// The tables filtered for a grid of spacing h, with T8 at that h (module doc).
pub fn filtered(table: &PseudoTable, h: f64) -> FilteredTable { build(table, h, true, &Rerun::of(table), LEDGER.numerics.kspl_step_over_h) }

/// R_SR = max(r_cmp, r_floor) (module doc, "The local split").
fn sr_support(t: &PseudoTable, a: f64) -> f64 {
    let g = &t.grid;
    let r_floor = g.r.iter().copied().rev().find(|&d| t.z_v * libm::erfc(d / a) / d >= LEDGER.v_tail_floor).unwrap_or(g.r[0]);
    t.kb.r_cmp.max(r_floor)
}

/// The last grid radius where |f| exceeds 1e-12 of its maximum (the identity filter's cut).
fn identity_support(g: &LogGrid, f: &[f64]) -> f64 {
    let m = f.iter().fold(0.0f64, |x, y| x.max(y.abs()));
    g.r[f.iter().rposition(|y| y.abs() > 1e-12 * m).unwrap_or(0)]
}

/// ½(v_xc↑ + v_xc↓) on ρ_σ + ½ρ_pc (the field ρ_pc multiplies in E_xc, to first order).
fn mean_vxc(rho: &[Vec<f64>; 2], rho_pc: &[f64]) -> Vec<f64> {
    (0..rho_pc.len()).map(|i| { let c = 0.5 * rho_pc[i]; let (_, vu, vd) = exc_point(rho[0][i] + c, rho[1][i] + c); 0.5 * (vu + vd) }).collect()
}

/// ρ_pc's support (module doc, Task 5 physics review S1): the smallest grid r with 4π∫_r^∞ |v̄_xc| ρ_pc r′² dr′ ≤
/// `v_tail_floor`, v̄_xc of the unfiltered reference KB atom — the first-order XC energy of the discarded tail. None
/// without a core.
fn pc_support(t: &PseudoTable, un: &KbAtom) -> Option<f64> {
    let (g, rho) = (&t.grid, &t.partial_core.rho_pc);
    if rho.iter().all(|&x| x == 0.0) { return None; }
    let v = mean_vxc(&un.rho, rho);
    let c = g.cumulative(&(0..g.len()).map(|i| 4.0 * PI * g.r[i] * g.r[i] * v[i].abs() * rho[i]).collect::<Vec<_>>());
    let total = c[g.len() - 1];
    Some(g.r[(0..g.len()).find(|&i| total - c[i] <= LEDGER.v_tail_floor).expect("the cumulative reaches its total")])
}

/// The hard cut at q_max (no completion, no step): the band-limited weight fields of `table_eggbox`.
fn hard_cut(grid: &LogGrid, f: &[f64], l: u8, q_max: f64) -> Completion {
    let n = grid.len();
    let (q, w) = q_mesh(0.0, q_max);
    let c = q.iter().zip(&w).map(|(&q, &w)| w * 4.0 * PI * grid.integrate(&(0..n).map(|i| f[i] * sph_bessel(l, q * grid.r[i]) * grid.r[i] * grid.r[i]).collect::<Vec<_>>())).collect();
    Completion { l, q, c, r_cut: f64::INFINITY, width: 1.0 }
}

/// The completions of one table set at h (outer step `step_over_h`·h): V_SR, every β_l, ρ_pc (None without a core).
struct Completions { v_sr: (Completion, f64), betas: Vec<(Completion, f64)>, pc: Option<(Completion, f64)> }

fn completions(t: &PseudoTable, h: f64, r_pc: Option<f64>, step_over_h: f64) -> Completions {
    let g = &t.grid;
    let n = g.len();
    let (gamma, width) = (LEDGER.gamma_r, step_over_h * h);
    let a = LEDGER.sigma_lr_over_h * h * 2f64.sqrt();
    let (q_c, q_max) = (LEDGER.q_c_over_nyquist * PI / h, PI / h);
    let band = |f: &[f64], l: u8, r: f64| (complete(g, f, l, q_c, q_max, gamma * r, width), gamma * r + width);
    // the one short-range table (plan review Q2): V_SR = V_loc + Z_v erf(d/a)/d
    let v_sr: Vec<f64> = g.r.iter().zip(&t.kb.v_loc).map(|(&d, &v)| v + t.z_v * libm::erf(d / a) / d).collect();
    // F_l = δV·u/r = β_u/r, the 3-D radial part; R = r_β (spec §4.6), derived
    let betas = t.kb.projectors.iter().map(|p| band(&(0..n).map(|i| p.beta[i] / g.r[i]).collect::<Vec<_>>(), p.l, p.support(g))).collect();
    Completions { v_sr: band(&v_sr, 0, sr_support(t, a)), betas, pc: r_pc.map(|r| band(&t.partial_core.rho_pc, 0, r)) }
}

fn build(t: &PseudoTable, h: f64, filter: bool, rerun: &Rerun, step_over_h: f64) -> FilteredTable {
    let g = &t.grid;
    let n = g.len();
    let du = h / LEDGER.numerics.table_points_per_h as f64;
    let sigma_lr = LEDGER.sigma_lr_over_h * h;
    let a = sigma_lr * 2f64.sqrt();
    // (log-grid values, uniform table, radius)
    type Tab = (Vec<f64>, Vec<f64>, f64);
    let exact = |(c, r): &(Completion, f64)| -> Tab { (g.r.iter().map(|&x| c.value_at(x)).collect(), uniform_fill(du, *r, |d| c.value_at(d)), *r) };
    // the identity filter: the function itself, zero beyond its radius, the uniform table interpolating the log grid
    let identity = |f: Vec<f64>, r: f64| -> Tab {
        let f: Vec<f64> = f.iter().zip(&g.r).map(|(&v, &x)| if x > r { 0.0 } else { v }).collect();
        let u = uniform_fill(du, r, |d| if d > r { 0.0 } else { g.interp(&f, d) });
        (f, u, r)
    };
    let (v_sr, betas, pc): (Tab, Vec<Tab>, Option<Tab>) = if filter {
        let c = completions(t, h, rerun.r_pc, step_over_h);
        (exact(&c.v_sr), c.betas.iter().map(exact).collect(), c.pc.as_ref().map(exact))
    } else {
        let v: Vec<f64> = g.r.iter().zip(&t.kb.v_loc).map(|(&d, &v)| v + t.z_v * libm::erf(d / a) / d).collect();
        let r_v = identity_support(g, &v);
        let betas = t.kb.projectors.iter().map(|p| identity((0..n).map(|i| p.beta[i] / g.r[i].powi(p.l as i32 + 1)).collect(), p.support(g))).collect();
        let pc = rerun.r_pc.map(|_| identity(t.partial_core.rho_pc.clone(), identity_support(g, &t.partial_core.rho_pc)));
        (identity(v, r_v), betas, pc)
    };
    let projectors = t.kb.projectors.iter().zip(betas)
        .map(|(p, (t_log, t_u, radius))| FilteredProjector { l: p.l, e_kb: p.e_kb, t: t_log, t_u, radius }).collect();
    let (rho_pc, rho_pc_u, pc_radius) = pc.unwrap_or((vec![0.0; n], vec![0.0; 4], 0.0));
    let mut ft = FilteredTable {
        z: t.z, z_v: t.z_v, n_core: t.n_core, h, sigma_lr, v_sr_table: v_sr.0, r_sr: v_sr.2,
        projectors, rho_pc, pc_radius, r_core: t.partial_core.r_core, du, v_sr_u: v_sr.1, rho_pc_u,
        check: FilterRow { h, ..FilterRow::default() }, record: t.record.clone(), grid: g.clone(),
    };
    ft.check = rerun.run(t, &ft);
    ft
}

/// T8(c) (spec §4.8, §4.5): the number of (spin, r) points where ρ_pc < 0 and ρ_σ + ½ρ_pc < 0 — where
/// `qm::xc::exc_point`'s clamp would engage.
pub fn xc_domain_violations(rho: &[Vec<f64>; 2], rho_pc: &[f64]) -> usize {
    (0..rho_pc.len()).filter(|&i| rho_pc[i] < 0.0)
        .map(|i| (0..2).filter(|&s| rho[s][i] + 0.5 * rho_pc[i] < 0.0).count()).sum()
}

/// max over R ∈ {±h/4}³ of |∇_R E(R)| (central differences, step 2e-4 Bohr; module doc).
fn probe(h: f64, e: impl Fn([f64; 3]) -> f64) -> f64 {
    let (q, step) = (h / 4.0, 2e-4);
    let mut worst = 0.0f64;
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let r = [sx, sy, sz];
        let grad: Vec<f64> = (0..3).map(|a| {
            let (mut p, mut m) = (r, r);
            p[a] += step;
            m[a] -= step;
            (e(p) - e(m)) / (2.0 * step)
        }).collect();
        worst = worst.max(grad.iter().map(|x| x * x).sum::<f64>().sqrt());
    }}}
    worst
}

/// The lattice points r_p − R (spacing h) within `radius` of R, and h³.
fn lattice(h: f64, radius: f64, r: [f64; 3]) -> impl Iterator<Item = [f64; 3]> {
    let k = (radius / h).ceil() as i64 + 1;
    (-k..=k).flat_map(move |i| (-k..=k).flat_map(move |j| (-k..=k).map(move |l| [i as f64 * h - r[0], j as f64 * h - r[1], l as f64 * h - r[2]])))
        .filter(move |d| d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= radius * radius)
}

fn norm3(d: [f64; 3]) -> f64 { (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() }

/// T8(d) (spec §5.1 "Translation invariance"): [F_SR, F_NL, F_pc] in Ha/Bohr, each the max over R ∈ {±h/4}³ of
/// |∇_R h³Σ_p w(r_p − R) g(r_p − R)|, g read through the uniform tables and w from `atom` (the filtered KB atom at the
/// reference configuration) hard-cut at |q| = π/h: ρ_v for V_SR; φ_l through Σ_m (f_l/(2l+1)) E_KB c_lm² for β_l (an
/// unoccupied channel: its reference pseudo-wavefunction, f = 1); ½(v_xc↑ + v_xc↓)[ρ_v + ρ_pc^filt] for ρ_pc.
pub fn table_eggbox(t: &PseudoTable, ft: &FilteredTable, atom: &KbAtom) -> [f64; 3] {
    let (g, h, du) = (&ft.grid, ft.h, ft.du);
    let n = g.len();
    let q_max = PI / h;
    let h3 = h * h * h;
    let weight = |f: &[f64], l: u8, radius: f64| { let c = hard_cut(g, f, l, q_max); uniform_fill(du, radius, |d| c.value_at(d)) };
    // a spherical table g with a spherical weight w
    let spherical = |w: &[f64], radius: f64, gf: &dyn Fn(f64) -> f64| probe(h, |r| {
        h3 * neumaier(lattice(h, radius, r).map(|d| { let x = norm3(d); uniform_lagrange(w, du, x).0 * gf(x) }))
    });
    let rho_v: Vec<f64> = (0..n).map(|i| atom.rho[0][i] + atom.rho[1][i]).collect();
    let f_sr = spherical(&weight(&rho_v, 0, ft.r_sr), ft.r_sr, &|x| ft.v_sr(x));
    let f_pc = if ft.pc_radius > 0.0 {
        spherical(&weight(&mean_vxc(&atom.rho, &ft.rho_pc), 0, ft.pc_radius), ft.pc_radius, &|x| ft.rho_pc_at(x))
    } else { 0.0 };
    // E_NL = Σ_k Σ_m (f_l/(2l+1)) E_KB c_km², c_km = h³Σ_p β_klm φ_lm, φ_lm = (φ_l/d^l)·√((2l+1)/4π)·R_lm
    let base = reference_config(&t.partition);
    let projs: Vec<(usize, u8, f64, f64, Vec<f64>)> = ft.projectors.iter().enumerate().map(|(k, p)| {
        let li = p.l as usize;
        let occ = base.occ[li][0] + base.occ[li][1];
        let (u, f) = if occ > 0.0 { (&atom.u[li][0], occ) } else { (&t.channels.iter().find(|c| c.l == p.l).expect("a channel per projector").u, 1.0) };
        (k, p.l, f, p.e_kb, weight(&(0..n).map(|i| u[i] / g.r[i]).collect::<Vec<_>>(), p.l, p.radius))
    }).collect();
    let f_nl = if projs.is_empty() { 0.0 } else {
        probe(h, |r| neumaier(projs.iter().flat_map(|(k, l, f, e_kb, phi)| {
            let (radius, li) = (ft.projectors[*k].radius, *l as usize);
            let nrm2 = (2 * li + 1) as f64 / (4.0 * PI);
            let mut c = vec![Vec::new(); 2 * li + 1];
            let mut sh = [0.0; N_MOMENTS];
            for d in lattice(h, radius, r) {
                let x = norm3(d);
                let tp = ft.beta_radial(*k, x) * uniform_lagrange(phi, du, x).0 * nrm2;
                if li > 0 { solid_harmonics(d, &mut sh); }
                for (m, cm) in c.iter_mut().enumerate() {
                    let rlm = if li == 0 { 1.0 } else { sh[lm_index(li, m as i32 - li as i32)] };
                    cm.push(tp * rlm * rlm);
                }
            }
            let a = f / (2 * li + 1) as f64 * e_kb;
            c.into_iter().map(move |cm| { let s = h3 * neumaier(cm.into_iter()); a * s * s }).collect::<Vec<_>>()
        })))
    };
    [f_sr, f_nl, f_pc]
}

/// What the filtered re-run compares with, computed once per table (not per h): the unfiltered KB atom at the
/// reference configuration, the T3 configurations the gates measured with their all-electron ΔE, and ρ_pc's support.
struct Rerun { un: KbAtom, t3: Vec<(TransferConfig, f64)>, r_pc: Option<f64> }

impl Rerun {
    fn of(t: &PseudoTable) -> Rerun {
        let un = solve_kb_atom(&t.grid, &t.kb.input(), &reference_config(&t.partition), SpinMode::Restricted)
            .expect("the unfiltered reference KB atom converged in generate");
        let t3 = if t.gates.t3.is_empty() { Vec::new() } else {
            let hund: AtomResult = ground_state(t.z, SpinMode::Polarised).expect("the Hund atom converged in generate");
            transfer_configs(t, &hund).into_iter()
                .filter_map(|c| t.gates.t3.iter().find(|r| r.name == c.name).map(|r| (c, r.de_ae))).collect()
        };
        let r_pc = pc_support(t, &un);
        Rerun { un, t3, r_pc }
    }

    /// T8 on the filtered tables: (a) the reference energy and occupied eigenvalues against the unfiltered KB atom;
    /// (b) every gated transfer ΔE against the all-electron one (S2); (c) the LSDA domain over the reference and every
    /// T3 configuration; (d) the table egg-box force.
    fn run(&self, t: &PseudoTable, ft: &FilteredTable) -> FilterRow {
        let input = ft.input();
        let Ok(fa) = solve_kb_atom(&ft.grid, &input, &reference_config(&t.partition), SpinMode::Restricted) else {
            return FilterRow { h: ft.h, de: f64::INFINITY, max_eig: f64::INFINITY, t3_max: f64::INFINITY, clamped: 0, f_table: [f64::INFINITY; 3] };
        };
        let max_eig = t.partition.channels.iter().filter(|c| c.occupied)
            .map(|c| (fa.eps[c.l as usize][0] - self.un.eps[c.l as usize][0]).abs()).fold(0.0, f64::max);
        let mut clamped = xc_domain_violations(&fa.rho, &ft.rho_pc);
        let mut t3_max = 0.0f64;
        for (cfg, de_ae) in &self.t3 {
            t3_max = t3_max.max(match solve_kb_atom(&ft.grid, &input, &cfg.pp, cfg.spin) {
                Ok(pp) => { clamped += xc_domain_violations(&pp.rho, &ft.rho_pc); ((pp.energy - fa.energy) - de_ae).abs() }
                Err(_) => f64::INFINITY,
            });
        }
        FilterRow { h: ft.h, de: fa.energy - self.un.energy, max_eig, t3_max, clamped, f_table: table_eggbox(t, ft, &fa) }
    }
}

/// T8 (spec §4.8; Spec question 11): h_req = the coarsest ladder h such that it and every finer ladder h pass all four
/// criteria (`FilterRow::passes`). The rows (one per ladder h, coarsest first) are the generation record's filter ladder.
/// None when even the finest fails: an outcome, not a refusal of the potential.
pub fn h_req(table: &PseudoTable) -> (Option<f64>, Vec<FilterRow>) {
    let rerun = Rerun::of(table);
    let rows: Vec<FilterRow> = LEDGER.h_ladder.iter().map(|&h| build(table, h, true, &rerun, LEDGER.numerics.kspl_step_over_h).check).collect();
    let first = (0..rows.len()).find(|&k| rows[k..].iter().all(FilterRow::passes));
    (first.map(|k| rows[k].h), rows)
}

/// The same build with another outer-step width (`table_eggbox_probe_sees_a_one_spacing_step`).
#[cfg(test)]
fn build_with_step(t: &PseudoTable, h: f64, step_over_h: f64) -> FilteredTable { build(t, h, true, &Rerun::of(t), step_over_h) }

#[cfg(test)]
mod tests {
    use super::*;
    /// Each element generated once per test process (generate runs the whole T8 ladder), shared with the
    /// other modules' tests.
    use crate::radial::test_cache::table;

    /// h³ Σ_p g over a lattice of spacing h around the origin, shifted by `s`.
    fn lattice_sum(h: f64, radius: f64, s: [f64; 3], g: impl Fn([f64; 3]) -> f64) -> f64 {
        let k = (radius / h).ceil() as i64 + 2;
        let mut terms = Vec::new();
        for i in -k..=k { for j in -k..=k { for l in -k..=k {
            terms.push(g([i as f64 * h - s[0], j as f64 * h - s[1], l as f64 * h - s[2]]));
        }}}
        crate::grid::neumaier(terms.into_iter()) * h * h * h
    }

    /// The rigid shifts {0, h/2}³ (cos(2πR_a/h) = ±1: the first shell at full weight) and {±h/4}³ (identical by the
    /// axis reflections for every reflection-even integrand: reported, carries no information; ruling N1).
    fn shifts(h: f64) -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for set in [[0.0, h / 2.0], [-h / 4.0, h / 4.0]] {
            for a in set { for b in set { for c in set { v.push([a, b, c]); } } }
        }
        v
    }

    /// (max − min)/scale.
    fn spread(vals: &[f64], scale: f64) -> f64 {
        (vals.iter().cloned().fold(f64::MIN, f64::max) - vals.iter().cloned().fold(f64::MAX, f64::min)) / scale
    }

    fn mean_abs(v: &[f64]) -> f64 { v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64 }

    /// Report for references.md (no bound; spec §5.1: the weight-1 sum is neither the energy nor the force): the
    /// {0, h/2}³ lattice spreads of every filtered table at h = 0.25 for C and O, the ±h/4 set (identically 0 by
    /// cubic symmetry for every reflection-even integrand), and the T8(d) forces beside them.
    #[test]
    fn lattice_spreads_report() {
        let h = 0.25;
        for z in [6u8, 8] {
            let ft = filtered(table(z), h);
            let report = |what: &str, radius: f64, g: &dyn Fn([f64; 3]) -> f64| {
                let vals: Vec<f64> = shifts(h).into_iter().map(|s| lattice_sum(h, radius, s, g)).collect();
                let scale = mean_abs(&vals);
                eprintln!("Z = {z} {what}: weight-1 spread {{0, h/2}}³ {:.2e}, ±h/4 {:.2e} (mean |S| {scale:.6e}, radius {radius:.3})",
                          spread(&vals[..8], scale), spread(&vals[8..], scale));
            };
            report("V_SR", ft.sr_radius(), &|d| ft.v_sr(norm3(d)));
            report("ρ_pc", ft.pc_radius, &|d| ft.rho_pc_at(norm3(d)));
            for (k, p) in ft.projectors.iter().enumerate() {
                let nrm = ((2 * p.l + 1) as f64 / (4.0 * PI)).sqrt();
                for m in -(p.l as i32)..=(p.l as i32) {
                    let beta = |d: [f64; 3]| { let mut sh = [0.0; N_MOMENTS]; solid_harmonics(d, &mut sh); ft.beta_radial(k, norm3(d)) * nrm * sh[lm_index(p.l as usize, m)] };
                    if p.l == 0 { report("β sum", p.radius, &beta); }
                    report(&format!("β norm l = {} m = {m}", p.l), p.radius, &|d| beta(d).powi(2));
                }
            }
            eprintln!("Z = {z} h = {h}: T8(d) F_SR {:.2e}, F_NL {:.2e}, F_pc {:.2e} Ha/Bohr", ft.check.f_table[0], ft.check.f_table[1], ft.check.f_table[2]);
        }
    }

    /// Spec §5.1 "Sampling": every uniform 3-D table reproduces the band-limited function it samples to ≤ 1e-9 of its
    /// maximum (derived bound (9/16)(π/256)⁴/24 = 5.3e-10, Bernstein) at 10⁴ quasi-random d in [0, radius].
    #[test]
    fn uniform_tables_match_the_band_limited_function() {
        for (z, h) in [(8u8, 0.20), (6, 0.30), (9, 0.15)] {
            let t = table(z);
            let rerun = Rerun::of(t);
            let step = LEDGER.numerics.kspl_step_over_h;
            let c = completions(t, h, rerun.r_pc, step);
            let ft = build(t, h, true, &rerun, step);
            let ft = &ft;
            let mut tabs: Vec<(String, &Completion, Box<dyn Fn(f64) -> f64 + '_>, f64)> = vec![("V_SR".into(), &c.v_sr.0, Box::new(|d| ft.v_sr(d)), ft.r_sr)];
            for (k, p) in ft.projectors.iter().enumerate() { tabs.push((format!("β_{}", p.l), &c.betas[k].0, Box::new(move |d| ft.beta_radial(k, d)), p.radius)); }
            if let Some((pc, _)) = &c.pc { tabs.push(("ρ_pc".into(), pc, Box::new(|d| ft.rho_pc_at(d)), ft.pc_radius)); }
            for (name, comp, read, radius) in tabs {
                let (mut worst, mut gmax) = (0.0f64, 0.0f64);
                for k in 0..10_000 {
                    let d = radius * ((k as f64 * 0.618_033_988_749_894_9) % 1.0);
                    let exact = comp.value_at(d);
                    gmax = gmax.max(exact.abs());
                    worst = worst.max((read(d) - exact).abs());
                }
                eprintln!("Z = {z} h = {h} {name}: max |uniform − exact| / max|g| = {:.2e}", worst / gmax);
                assert!(worst <= 1e-9 * gmax, "Z = {z} h = {h} {name}: {:.3e} of max|g|", worst / gmax);
            }
        }
    }

    /// T8(d)'s probe is first-order exact in the leakage (spec §5.1): a projector that is band-limited *and* compact
    /// gives no egg-box force. Test function t = e^{−d²/s} on O's s-projector sphere (radius ρ = γ_r r_β + 8h): its
    /// content at π/h is e^{−(π/h)²s/4} and its value where the table ends is e^{−ρ²/s}; both ≤ ε = 1e-13 needs
    /// 4 ln(1/ε)(h/π)² ≤ s ≤ ρ²/ln(1/ε), feasible at h = 0.15 (s ∈ [0.274, 0.307]; s = their geometric mean). Measured
    /// F_NL 7.6e-13 (review: ≤ 4e-13). Deviation from the plan text, whose e^{−d²/0.5} has 2.7e-9 of its content at π/h
    /// for h = 0.25 (not ≤ 1e-9) and is cut at 1e-8 by the h = 0.15 sphere: there the probe measures that leakage,
    /// 7.8e-12 and 2.5e-9 (printed below for h = 0.25).
    #[test]
    fn table_eggbox_probe_is_blind_to_band_limited_tables() {
        let t = table(8);
        let eps: f64 = 1e-13;
        let with_gaussian = |h: f64, s: f64| {
            let mut ft = filtered(t, h);
            let atom = solve_kb_atom(&ft.grid, &ft.input(), &reference_config(&t.partition), SpinMode::Restricted).unwrap();
            let du = ft.du;
            for (k, v) in ft.projectors[0].t_u.iter_mut().enumerate() { *v = (-(k as f64 * du).powi(2) / s).exp(); }
            (table_eggbox(t, &ft, &atom)[1], ft.projectors[0].radius)
        };
        let (plan, _) = with_gaussian(0.25, 0.5);
        eprintln!("O, h = 0.25, e^(−d²/0.5) (content at π/h 2.7e-9): F_NL {plan:.2e} — the test function's own leakage");
        let h = 0.15;
        let rho = LEDGER.gamma_r * table(8).kb.projectors[0].support(&t.grid) + LEDGER.numerics.kspl_step_over_h * h;
        let (s_min, s_max) = (4.0 * (1.0 / eps).ln() * (h / PI).powi(2), rho * rho / (1.0 / eps).ln());
        assert!(s_min <= s_max, "no band-limited compact Gaussian at h = {h}: s ∈ [{s_min:.3}, {s_max:.3}]");
        let s = (s_min * s_max).sqrt();
        let (f, radius) = with_gaussian(h, s);
        assert!((radius - rho).abs() < 1e-12);
        eprintln!("O, h = {h}, e^(−d²/{s:.3}) (content and end value ≤ {eps:.0e}): F_NL {f:.2e}");
        assert!(f < 1e-12, "F_NL {f:.3e}");
    }

    /// The probe sees what it must (Task 5 physics review, Table A): O at h = 0.20 with the outer step one spacing wide
    /// (the brief's construction) has F_NL = 8.0e-5 Ha/Bohr. Assert it is > 10 × h_req_force, so a blind probe fails
    /// here. With the ruled 8h it is 9.2e-7.
    #[test]
    fn table_eggbox_probe_sees_a_one_spacing_step() {
        let narrow = build_with_step(table(8), 0.20, 1.0);
        let f = narrow.check.f_table[1];
        eprintln!("O, h = 0.20, step width h: F_NL {f:.2e}");
        assert!(f > 10.0 * LEDGER.gates.h_req_force, "F_NL {f:.3e}");
    }

    /// T8(c)'s counter on a synthetic case: a minority spin of 1e-4 against ½ρ_pc = −2e-4 at one point counts once; the
    /// same lobe under a majority density of 1e-1 does not count.
    #[test]
    fn xc_domain_counter_counts_minority_spin_violations() {
        let rho = [vec![1e-1, 1e-1], vec![1e-1, 1e-4]];
        assert_eq!(xc_domain_violations(&rho, &[-4e-4, -4e-4]), 1);
        assert_eq!(xc_domain_violations(&rho, &[-4e-4, 1.0]), 0);
    }

    fn print_ladder(z: u8, t: &PseudoTable) {
        for r in &t.record.filter {
            eprintln!("Z = {z} h = {:.2}: ΔE {:+.3e}, Δε {:.3e}, T3 {:.3e}, clamped {}, F {:.2e}/{:.2e}/{:.2e} → {}",
                      r.h, r.de, r.max_eig, r.t3_max, r.clamped, r.f_table[0], r.f_table[1], r.f_table[2], r.passes());
        }
        eprintln!("Z = {z}: h_req {:?}", t.record.h_req);
    }

    /// The rows at or below h_req (empty without h_req).
    fn certified(t: &PseudoTable) -> Vec<&FilterRow> {
        t.record.h_req.map_or(Vec::new(), |h| t.record.filter.iter().filter(|r| r.h <= h + 1e-12).collect())
    }

    /// Spec §4.8 T8, §6.4: the species of M7's molecules (H₂O, NH₃, CH₄, O₂) have an h_req, and every ladder row at or
    /// below it passes all four T8 criteria (FilterRow::passes); every row is printed (ΔE, Δε, T3, clamped, F_SR/F_NL/F_pc).
    /// Ne's h_req (None at -3: ρ_pc carries 0.97 of its 1.21 mHa at 0.15) is pinned by the records table, not here.
    #[test]
    fn m7_species_have_h_req_and_every_ladder_row_below_passes() {
        let mut fails = Vec::new();
        for z in [1u8, 6, 7, 8] {
            let t = table(z);
            print_ladder(z, t);
            let Some(h) = t.record.h_req else { fails.push(format!("Z = {z}: no h_req")); continue };
            for r in t.record.filter.iter().filter(|r| r.h <= h + 1e-12) { if !r.passes() { fails.push(format!("Z = {z} h = {}: fails T8", r.h)); } }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }

    /// Spec §8 row 5 / §4.8 T8(a, b) (review S2): at h_req the filtered radial KB atom reproduces the unfiltered total
    /// energy and every valence eigenvalue within h_req_tol (= T1), and the filtered T3 stays within the T3 gate — H and
    /// C–F (Ne's None is an outcome, pinned by the records table).
    #[test]
    fn filtered_kb_atom_within_1mha_at_h_req() {
        let g = &LEDGER.gates;
        for z in [1u8, 6, 7, 8, 9] {
            let t = table(z);
            let h = t.record.h_req.unwrap_or_else(|| panic!("Z = {z}: no h on the ladder passes"));
            assert!(LEDGER.h_ladder.contains(&h));
            for r in certified(t) {
                assert!(r.de.abs() < g.h_req_tol && r.max_eig < g.h_req_tol, "Z = {z} h = {}: ΔE {:.3e}, max Δε {:.3e}", r.h, r.de, r.max_eig);
                assert!(r.t3_max < g.t3_delta_e, "Z = {z} h = {}: filtered T3 {:.3e}", r.h, r.t3_max);
            }
        }
    }

    /// Spec §8 row 5 / §4.8 T8(d): at every certified h the table egg-box force F_SR + F_NL + F_pc is below
    /// h_req_force (2e-6 Ha/Bohr, derived from §6.6), each table with its own energy weight at the §6.6 probe.
    #[test]
    fn table_eggbox_force_within_budget_at_h_req() {
        for z in [1u8, 6, 7, 8, 9] {
            let t = table(z);
            for r in certified(t) {
                let f: f64 = r.f_table.iter().sum();
                eprintln!("Z = {z} h = {:.2}: F_SR {:.2e}, F_NL {:.2e}, F_pc {:.2e}, F_atom {f:.2e} Ha/Bohr", r.h, r.f_table[0], r.f_table[1], r.f_table[2]);
                assert!(f < LEDGER.gates.h_req_force, "Z = {z} h = {}: F_atom {f:.3e}", r.h);
            }
        }
    }

    /// Spec §4.5, §4.8 T8(c) (S3 as the LSDA domain): at every certified h, n_σ = ρ_σ + ½ρ_pc^filt ≥ 0 in the reference
    /// and every T3 configuration — the XC clamp never engages; min ρ_pc^filt is printed (its interior lobes are allowed).
    #[test]
    fn filtered_xc_domain_is_respected() {
        for z in [6u8, 7, 8, 9, 14] {
            let t = table(z);
            for r in certified(t) {
                let ft = filtered(t, r.h);
                let (k, min) = ft.rho_pc.iter().copied().enumerate().fold((0, f64::INFINITY), |m, (i, x)| if x < m.1 { (i, x) } else { m });
                eprintln!("Z = {z} h = {:.2}: clamped {}, min ρ_pc^filt {min:.3e} at r = {:.3} (pc_radius {:.3})", r.h, ft.check.clamped, ft.grid.r[k], ft.pc_radius);
                assert_eq!(ft.check.clamped, 0, "Z = {z} h = {}", r.h);
            }
        }
    }

    #[test]
    fn bessel_functions_and_transform_pair() {
        // closed forms vs series across x = 1
        for l in 0..=3u8 { for &x in &[0.999, 1.001, 0.3, 5.0] {
            let s = sph_bessel_scaled(l, x) * x.powi(l as i32);
            assert!((s - sph_bessel(l, x)).abs() < 1e-12, "l = {l}, x = {x}");
        }}
        // a Gaussian is band-limited enough that a cut far beyond its spectrum returns it (the band
        // completion then has nothing to cancel: its least-squares solution is ≈ 0)
        let g = LogGrid::universal();
        let f: Vec<f64> = g.r.iter().map(|r| (-r * r).exp()).collect();
        let back = filter_radial(&g, &f, 0, 40.0, 45.0, 50.0, 5.0);
        for &r in &[0.0f64, 0.5, 1.5] { let i = g.nearest(r.max(1e-5)); assert!((back[i] - f[i]).abs() < 1e-9, "r = {r}: {:.3e}", back[i] - f[i]); }
    }

    /// Module doc, "The q mesh": Simpson's coarse trapezoid recurs at π/dq, so dq ≤ π/(2r_max) keeps the image of the
    /// kept part beyond the grid. The analytic transform of a unit Gaussian (π^{3/2}e^{−q²/4}) summed on the [0, 40] mesh
    /// is ≤ 1e-15 everywhere on [r_max/2, r_max] (with the brief's dq ≤ π/r_max: 2.2e-3 at r = 59.5).
    #[test]
    fn simpson_images_lie_beyond_the_grid() {
        let (q, w) = q_mesh(0.0, 40.0);
        let worst = (0..=60).map(|k| LEDGER.r_max * (0.5 + k as f64 / 120.0))
            .map(|r| (0..q.len()).map(|k| w[k] * PI.powf(1.5) * (-q[k] * q[k] / 4.0).exp() * q[k] * q[k] * sph_bessel(0, q[k] * r)).sum::<f64>().abs() / (2.0 * PI * PI))
            .fold(0.0, f64::max);
        eprintln!("worst kept-part image on [r_max/2, r_max]: {worst:.2e}");
        assert!(worst <= 1e-15, "{worst:.2e}");
    }

    /// Plan review Q3 diagnostic (report): on the TM-class model f = −3(1 − (r/r_c)²)³, the ruled completion (outer step
    /// 8h) lies below the hard cut at the first and √2 shells (the hard cut's 1.574e-6 / 5.557e-7, reproduced in
    /// `hard_cut_reproduces_the_plan_review_table`). The judge is T8(d); the old 1e-7 was not derived.
    #[test]
    fn band_completion_suppresses_the_first_shell() {
        let g = LogGrid::universal();
        let (h, rc) = (0.25, 1.4);
        let f: Vec<f64> = g.r.iter().map(|&r| if r < rc { -3.0 * (1.0 - (r / rc).powi(2)).powi(3) } else { 0.0 }).collect();
        let (q_c, q_max) = (LEDGER.q_c_over_nyquist * PI / h, PI / h);
        let out = filter_radial(&g, &f, 0, q_c, q_max, LEDGER.gamma_r * rc, LEDGER.numerics.kspl_step_over_h * h);
        let ft = |q: f64| 4.0 * PI * g.integrate(&(0..g.len()).map(|i| out[i] * sph_bessel(0, q * g.r[i]) * g.r[i] * g.r[i]).collect::<Vec<_>>());
        let (f0, f1, f2) = (ft(0.0), ft(2.0 * PI / h), ft(2f64.sqrt() * 2.0 * PI / h));
        eprintln!("band completion (8h): |F(2π/h)|/|F(0)| {:.2e}, |F(√2·2π/h)|/|F(0)| {:.2e}", (f1 / f0).abs(), (f2 / f0).abs());
        assert!((f1 / f0).abs() < 1.574e-6 && (f2 / f0).abs() < 5.557e-7);
    }

    /// The measurement chain against the plan review's own table (Q3): the hard cut at q_c followed by the ψ-step over
    /// [r_c, γ_r r_c] on f = −3(1 − (r/r_c)²)³ reproduces the review's |F(G₁)|/|F(0)| and |F(G√2)|/|F(0)| to their printed
    /// two digits; the ruled KSPL construction (8h) is printed beside it.
    #[test]
    fn hard_cut_reproduces_the_plan_review_table() {
        let g = LogGrid::universal();
        let review = [((0.25, 1.4), "1.6e-6", "5.6e-7"), ((0.25, 1.9), "9.7e-8", "2.5e-9"), ((0.35, 1.4), "2.2e-5", "3.6e-6"), ((0.35, 1.9), "1.3e-6", "6.6e-7")];
        for ((h, rc), g1, g2) in review {
            let f: Vec<f64> = g.r.iter().map(|&r| if r < rc { -3.0 * (1.0 - (r / rc).powi(2)).powi(3) } else { 0.0 }).collect();
            let (q_c, q_max) = (LEDGER.q_c_over_nyquist * PI / h, PI / h);
            let kspl = complete(&g, &f, 0, q_c, q_max, LEDGER.gamma_r * rc, LEDGER.numerics.kspl_step_over_h * h);
            let nk = q_mesh(0.0, q_c).0.len();
            let hard = Completion { l: 0, q: kspl.q[..nk].to_vec(), c: kspl.c[..nk].to_vec(), r_cut: rc, width: (LEDGER.gamma_r - 1.0) * rc };
            let ft = |c: &Completion, q: f64| 4.0 * PI * g.integrate(&(0..g.len()).map(|i| c.value_at(g.r[i]) * sph_bessel(0, q * g.r[i]) * g.r[i] * g.r[i]).collect::<Vec<_>>());
            let ratios = |c: &Completion| { let f0 = ft(c, 0.0); ((ft(c, 2.0 * PI / h) / f0).abs(), (ft(c, 2f64.sqrt() * 2.0 * PI / h) / f0).abs()) };
            let ((h1, h2), (k1, k2)) = (ratios(&hard), ratios(&kspl));
            eprintln!("h = {h}, r_c = {rc}: hard cut {h1:.3e} / {h2:.3e} (review {g1} / {g2}); KSPL (8h) {k1:.2e} / {k2:.2e}");
            assert_eq!((format!("{h1:.1e}"), format!("{h2:.1e}")), (g1.to_string(), g2.to_string()), "h = {h}, r_c = {rc}");
        }
    }

    /// Report: where the weight-1 {0, h/2}³ lattice spread of each table at h = 0.25 (C, O) comes from — the radial
    /// ĝ(G)/ĝ(0) at the first three shells, Poisson's 12|ĝ(G₁)| + 16|ĝ(G√3)| (module doc), the spread with exact
    /// evaluation at every lattice point, with the uniform tables, and the tables' own share.
    #[test]
    fn lattice_spread_sources_report() {
        let h = 0.25;
        for z in [6u8, 8] {
            let t = table(z);
            let g = &t.grid;
            let n = g.len();
            let rerun = Rerun::of(t);
            let ft = build(t, h, true, &rerun, LEDGER.numerics.kspl_step_over_h);
            let c = completions(t, h, rerun.r_pc, LEDGER.numerics.kspl_step_over_h);
            let pc = c.pc.as_ref().expect("a core");
            let tables: [(&str, &Completion, f64, &dyn Fn(f64) -> f64); 3] = [("V_SR", &c.v_sr.0, c.v_sr.1, &|d| ft.v_sr(d)),
                ("ρ_pc", &pc.0, pc.1, &|d| ft.rho_pc_at(d)), ("β_s", &c.betas[0].0, c.betas[0].1, &|d| ft.beta_radial(0, d))];
            for (name, comp, radius, read) in tables {
                let log: Vec<f64> = g.r.iter().map(|&x| comp.value_at(x)).collect();
                let ghat = |q: f64| 4.0 * PI * g.integrate(&(0..n).map(|i| log[i] * sph_bessel(0, q * g.r[i]) * g.r[i] * g.r[i]).collect::<Vec<_>>());
                let g0 = ghat(0.0);
                let s: Vec<f64> = [1.0f64, 2.0, 3.0].iter().map(|k| ghat(k.sqrt() * 2.0 * PI / h) / g0).collect();
                let set = |gf: &dyn Fn([f64; 3]) -> f64| shifts(h).into_iter().take(8).map(|s| lattice_sum(h, radius, s, gf)).collect::<Vec<f64>>();
                let exact = set(&|d| comp.value_at(norm3(d)));
                let uniform = set(&|d| read(norm3(d)));
                let diff: Vec<f64> = exact.iter().zip(&uniform).map(|(e, u)| u - e).collect();
                let scale = mean_abs(&exact);
                eprintln!("Z = {z} {name}: ĝ/ĝ(0) at G₁, G√2, G√3 {:.2e} {:.2e} {:.2e}; Poisson {:.2e}; spread exact {:.2e}, uniform tables {:.2e}, tables alone {:.2e}",
                          s[0], s[1], s[2], 12.0 * s[0].abs() + 16.0 * s[2].abs(), spread(&exact, scale), spread(&uniform, scale), spread(&diff, scale));
            }
        }
    }

    /// The u-form ↔ 3-D convention round trip (module doc): the identity filter's input is the KB input.
    #[test]
    fn unfiltered_input_reproduces_the_kb_input() {
        let t = table(8);
        let (a, b) = (t.kb.input(), FilteredTable::unfiltered(t, 0.2).input());
        for (pa, pb) in a.projectors.iter().zip(&b.projectors) {
            for i in (0..t.grid.len()).step_by(37) { assert!((pa.2[i] - pb.2[i]).abs() <= 1e-10 * (1.0 + pa.2[i].abs()), "β at r = {}", t.grid.r[i]); }
        }
        for i in (0..t.grid.len()).step_by(37) { assert!((a.v_loc[i] - b.v_loc[i]).abs() < 1e-9, "V_loc at r = {}", t.grid.r[i]); }
    }
}
