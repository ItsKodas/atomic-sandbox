//! Fermi–Dirac occupations at a fixed electronic temperature, per spin channel (host f64; the one rule both
//! solvers call: the CPU loop in `scf` and `qm-gpu`). Spec `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`
//! §4.1–4.3, §5, §5.1. Atomic units: ε, μ, kT in Hartree; k_B = 1, so S is dimensionless and T_el = kT.
//!
//! # The rule (one channel σ with N_σ = n > 0 electrons and k ≥ n + 1 solved levels ε_i)
//! ```text
//! f_i = 1 / (1 + e^{x_i}),   x_i = (ε_i − μ) / kT,   Σ_i f_i = n
//! ```
//! f is the occupation of spin orbital i, 0 ≤ f ≤ 1; it decreases with x, so Σf(μ) increases with μ
//! (dΣf/dμ = Σ f(1 − f)/kT > 0). It is evaluated overflow-free: f = e^{−x}/(1 + e^{−x}) for x ≥ 0 and
//! 1/(1 + e^{x}) for x < 0, so neither branch forms an e^{|x|} > 1; x = ±∞ gives exactly 0 or 1, and nothing is
//! clipped (a level above μ keeps f ≈ e^{−x} down to the subnormals).
//!
//! # μ by bisection to the plateau midpoint [MUST-1]
//! - **Bracket** [ε_min − 50 kT, ε_max + 50 kT] (`MU_BRACKET_KT`). At the lower end every x ≥ 50, f ≤ e^{−50},
//!   so Σf ≤ k·e^{−50} < n; at the upper end every f ≥ 1 − e^{−50} (= 1.0 in f64), so Σf = k > n. Both ends are
//!   evaluated once in f64 before bisecting; they fail only if |ε|/kT ≳ 1e14 absorbs the 50 kT margin.
//! - **Two bisections**, each halving [lo, hi] at mid = ½lo + ½hi (cannot overflow) and stopping when mid equals
//!   an endpoint in f64, i.e. when lo and hi are adjacent floats:
//!   - (≥) mid → hi when Σf(mid) − n ≥ 0, else → lo. Invariant Σf(lo) < n ≤ Σf(hi), so hi converges to a crossing
//!     μ₋ (the least f64 μ with Σf ≥ n, up to rounding: the f64 Σf need not be monotone in its last ulp);
//!   - (≤) mid → lo when Σf(mid) − n ≤ 0, else → hi. Invariant Σf(lo) ≤ n < Σf(hi), so lo converges to a crossing
//!     μ₊ (the greatest f64 μ with Σf ≤ n, up to rounding).
//!   - μ = ½μ₋ + ½μ₊. In a gapped channel Σf == n *exactly* in f64 over a plateau of μ (1 − f_HOMO below ½ ulp(1),
//!     f_LUMO below ½ ulp(n)), from about ε_HOMO + 37 kT to ε_LUMO − 36 kT once the gap exceeds ≈ 74 kT, and a
//!     single tie rule would put μ on one edge. μ₋ and μ₊ are the two edges, so μ is the plateau midpoint (near the
//!     FD root at mid-gap); every f is the same to ≤ e^{−36} anywhere on the plateau. Without a plateau μ₋ and μ₊
//!     coincide or are adjacent floats around the root, μ is one of them, and
//!     |Σf − n| ≤ (dΣf/dμ)·ulp(μ) ≤ k·ulp(μ)/(4kT).
//! - **Termination** [NOTE-1]. A bisection exhausts its bracket after ≈ 60 halvings at |μ| ~ 1; ≲ 120 near μ = 0 at
//!   kT = 5 mHa (measured 113–116: within |x| < 1.1e-16 e^x rounds to 1, so Σf is flat over ≈ ±1.1e-16·kT around a
//!   root at 0 and the bisection stops there); ≈ 1 100 only for a subnormal kT, whose flat zone is itself below the
//!   subnormal spacing, so the bisection runs through the subnormals. `MAX_MU_HALVINGS` = 2 048 is a backstop that never binds, so termination never rests on the stop rule
//!   alone. `halvings` reports the larger of the two counts.
//! - **Inputs refused before bisecting** (`QmError::Numerical`) [SHOULD-1]: k < n + 1 (then Σf ≤ k = n at every μ,
//!   no bracket), a non-finite eigenvalue (a NaN never equals an endpoint, so the stop would never fire), a kT
//!   that is not finite and > 0, and a bracket that fails its one check (an end not finite, or not
//!   Σf(lo) < n < Σf(hi)). n = 0 is an unsolved channel: every f = 0, μ = None, S = 0, no level needed.
//! - No tolerance parameter; the result is a deterministic function of the f64 inputs (the same bits on every run
//!   and in both solvers).
//!
//! # Entropy (spec §4.3)
//! S = Σ_i s(f_i), s(f) = −[f ln f + (1 − f) ln(1 − f)] ≥ 0, evaluated from x, not from f:
//! ```text
//! s(x) = ln(1 + e^{−|x|}) + |x| / (1 + e^{|x|})        (= ln_1p(e^{−|x|}) + |x|·min(f, 1 − f))
//! ```
//! Derivation: with a = |x|, f_< = 1/(1 + e^{a}) = min(f, 1 − f) and f_> = 1 − f_< = e^{a}/(1 + e^{a}),
//! −f_< ln f_< − f_> ln f_> = f_< ln(1 + e^{a}) + f_>[ln(1 + e^{a}) − a] = ln(1 + e^{a}) − a f_>
//! = ln(1 + e^{−a}) + a(1 − f_>) = ln(1 + e^{−a}) + a f_<. Both terms are ≥ 0 and → 0 as a → ∞, so the
//! 0·ln 0 = 0 limit is built in and nothing cancels at f → 0 or 1; s(±∞) is defined as 0 (∞/∞ otherwise). A
//! channel's S is a Neumaier sum (rule 4); the solver's free energy is F = E − kT·Σ_σ S_σ.
//!
//! # Witness and partial levels (spec §4.2, §5)
//! - The highest solved level ε_max must satisfy ε_max ≥ ε_(n) + `WITNESS_GAP` (ε_(n) the n-th lowest) and
//!   ε_max − μ ≥ `WITNESS_TAIL_KT`·kT; at 36 kT the first unsolved level would hold f < 2.4e-16.
//! - A level is partial when min(f, 1 − f) ≥ `PARTIAL_OCCUPATION_MIN` = 1e-6, i.e. |ε − μ| ≲ 13.8 kT; the GPU
//!   stop is max(tol_eff, `GPU_PARTIAL_TOL`) on a step with one (spec §5.1).

use crate::error::QmError;
use crate::grid::neumaier;
use crate::scf::ascending;

/// The electronic temperature kT (Ha): 5 mHa ≈ 1 579 K, the user's ruling of 2026-11-14 (spec §2 item 3). One
/// solver constant for every scene; not per element.
pub const ELECTRON_KT: f64 = 5e-3;
/// Witness arm 1 (Ha): the highest solved level lies at least this far above the n-th lowest (spec §4.2; the
/// value of the former `CANDIDATE_TAU`).
pub const WITNESS_GAP: f64 = 0.1;
/// Witness arm 2: the highest solved level lies at least this many kT above μ (f < 2.4e-16 beyond it).
pub const WITNESS_TAIL_KT: f64 = 36.0;
/// A level with min(f, 1 − f) at or above this is partly occupied (|ε − μ| ≲ 13.8 kT).
pub const PARTIAL_OCCUPATION_MIN: f64 = 1e-6;
/// The GPU SCF's convergence stop on a step whose occupations have a partial level (spec §5.1; the user's
/// 2026-11-14 value). The one definition (preflight F11): `qm-gpu`, the tags and the dimer ledger read it.
pub const GPU_PARTIAL_TOL: f64 = 1e-5;
/// The μ bracket's margin beyond the extreme levels, in kT.
pub const MU_BRACKET_KT: f64 = 50.0;
/// Backstop on each μ bisection; never binds for finite inputs (≲ 120 halvings at kT = 5 mHa, ≈ 1 100 only for a
/// subnormal kT; module doc, "Termination").
pub const MAX_MU_HALVINGS: usize = 2048;

/// One channel's Fermi–Dirac occupations.
#[derive(Clone, Debug, PartialEq)]
pub struct FdOccupations {
    /// f_i in the order of the `evals` passed in.
    pub f: Vec<f64>,
    /// μ (Ha), the plateau midpoint; `None` for an empty channel. A physical diagnostic only where a level is
    /// partial (spec §4.1).
    pub mu: Option<f64>,
    /// S = Σ s(f_i), dimensionless (k_B = 1); the free-energy term is kT·S.
    pub entropy: f64,
    /// The larger halving count of the two bisections (< `MAX_MU_HALVINGS` for finite inputs).
    pub halvings: usize,
}

/// f at x = (ε − μ)/kT, overflow-free; never clipped.
fn occupation(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Σf at μ, compensated.
fn total(evals: &[f64], mu: f64, kt: f64) -> f64 { neumaier(evals.iter().map(|&e| occupation((e - mu) / kt))) }

/// One tie-directed bisection on [lo, hi]. `ge`: Σf − n ≥ 0 moves hi, and hi (μ₋) is returned; otherwise Σf − n ≤ 0
/// moves lo, and lo (μ₊) is returned. Also returns the halving count.
fn bisect(evals: &[f64], n: f64, kt: f64, mut lo: f64, mut hi: f64, ge: bool) -> (f64, usize) {
    let mut halvings = 0;
    while halvings < MAX_MU_HALVINGS {
        let mid = 0.5 * lo + 0.5 * hi;
        if mid <= lo || mid >= hi {
            break;
        }
        let d = total(evals, mid, kt) - n;
        let to_hi = if ge { d >= 0.0 } else { d > 0.0 };
        if to_hi {
            hi = mid
        } else {
            lo = mid
        }
        halvings += 1;
    }
    (if ge { hi } else { lo }, halvings)
}

/// Fermi–Dirac occupations of one channel with `n` electrons over `evals` (any order) at temperature `kt` (Ha):
/// f in `evals`' order, μ the plateau midpoint of the two tie-directed bisections, S with Neumaier compensation.
/// The module doc has the rule and the inputs it refuses.
pub fn fermi_dirac(evals: &[f64], n: usize, kt: f64) -> Result<FdOccupations, QmError> {
    if n == 0 {
        return Ok(FdOccupations { f: vec![0.0; evals.len()], mu: None, entropy: 0.0, halvings: 0 });
    }
    if evals.len() < n + 1 {
        return Err(QmError::Numerical(format!("Fermi–Dirac needs at least n + 1 = {} levels, got {}", n + 1, evals.len())));
    }
    if let Some(e) = evals.iter().find(|e| !e.is_finite()) {
        return Err(QmError::Numerical(format!("non-finite eigenvalue {e} in the Fermi–Dirac occupations")));
    }
    if !(kt.is_finite() && kt > 0.0) {
        return Err(QmError::Numerical(format!("Fermi–Dirac kT = {kt} is not finite and > 0")));
    }
    let nf = n as f64;
    let lo = evals.iter().copied().fold(f64::INFINITY, f64::min) - MU_BRACKET_KT * kt;
    let hi = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max) + MU_BRACKET_KT * kt;
    if !(lo.is_finite() && hi.is_finite() && total(evals, lo, kt) < nf && total(evals, hi, kt) > nf) {
        return Err(QmError::Numerical(format!("the Fermi–Dirac μ bracket [{lo:e}, {hi:e}] does not bracket n = {n} at kT = {kt:e}")));
    }
    let (mu_minus, h_ge) = bisect(evals, nf, kt, lo, hi, true);
    let (mu_plus, h_le) = bisect(evals, nf, kt, lo, hi, false);
    let mu = 0.5 * mu_minus + 0.5 * mu_plus;
    let f = evals.iter().map(|&e| occupation((e - mu) / kt)).collect();
    let entropy = neumaier(evals.iter().map(|&e| fd_entropy((e - mu) / kt)));
    Ok(FdOccupations { f, mu: Some(mu), entropy, halvings: h_ge.max(h_le) })
}

/// The entropy s ≥ 0 of one level at x = (ε − μ)/kT: ln_1p(e^{−|x|}) + |x|/(1 + e^{|x|}) (module doc); s(±∞) = 0.
pub fn fd_entropy(x: f64) -> f64 {
    let a = x.abs();
    if a.is_infinite() {
        return 0.0;
    }
    (-a).exp().ln_1p() + a / (1.0 + a.exp())
}

/// Whether a channel's highest solved level is a witness for `n` electrons at μ = `mu` (spec §4.2): ε_max ≥ ε_(n) +
/// `WITNESS_GAP` (ε_(n) the n-th lowest) and ε_max − μ ≥ `WITNESS_TAIL_KT`·kT. An empty channel (n = 0) needs none;
/// a channel with no level above its n-th never has one.
pub fn witness_reached(evals: &[f64], n: usize, mu: f64, kt: f64) -> bool {
    if n == 0 {
        return true;
    }
    if evals.len() <= n {
        return false;
    }
    let order = ascending(evals);
    let (eps_n, eps_max) = (evals[order[n - 1]], evals[order[order.len() - 1]]);
    eps_max >= eps_n + WITNESS_GAP && eps_max - mu >= WITNESS_TAIL_KT * kt
}

/// One partly occupied level: min(f, 1 − f) ≥ `PARTIAL_OCCUPATION_MIN`.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialLevel {
    /// 0 = ↑, 1 = ↓.
    pub spin: u8,
    /// The level's rank in ascending order of ε within its channel (equal ε keep their input order).
    pub index: usize,
    /// ε (Ha).
    pub eps: f64,
    /// Its occupation f.
    pub f: f64,
}

/// The channel's partly occupied levels, in ascending order of ε.
pub fn partial_levels(evals: &[f64], f: &[f64], spin: u8) -> Vec<PartialLevel> {
    assert_eq!(evals.len(), f.len(), "partial_levels: one f per level");
    ascending(evals)
        .into_iter()
        .enumerate()
        .filter(|&(_, i)| f[i].min(1.0 - f[i]) >= PARTIAL_OCCUPATION_MIN)
        .map(|(rank, i)| PartialLevel { spin, index: rank, eps: evals[i], f: f[i] })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::QmError;
    use crate::grid::neumaier;
    use std::f64::consts::LN_2;

    /// Σf with the rule's own compensated sum.
    fn sum(f: &[f64]) -> f64 { neumaier(f.iter().copied()) }

    /// splitmix64: a deterministic stream for the randomised tests (no `rand` in `qm`).
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [0, 1).
        fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    }

    #[test]
    fn fd_degenerate_pair_half_filled() {
        let o = fermi_dirac(&[-0.3, -0.3], 1, 1e-3).unwrap();
        assert_eq!(o.f, vec![0.5, 0.5]);
        let mu = o.mu.unwrap();
        assert!((mu + 0.3).abs() <= 1e-15, "μ = {mu:e}");
        assert!((o.entropy - 2.0 * LN_2).abs() <= 1e-14, "S = {}", o.entropy);
    }

    #[test]
    fn fd_gap_symmetric_mu() {
        let kt = ELECTRON_KT;
        let o = fermi_dirac(&[-0.3, -0.3 + 5.0 * kt], 1, kt).unwrap();
        let (f0, f1) = (1.0 / (1.0 + (-2.5f64).exp()), 1.0 / (1.0 + 2.5f64.exp()));
        assert!((o.f[0] - f0).abs() <= 1e-14, "f0 = {} vs {f0}", o.f[0]);
        assert!((o.f[1] - f1).abs() <= 1e-14, "f1 = {} vs {f1}", o.f[1]);
        let mu = o.mu.unwrap();
        assert!((mu - (-0.3 + 2.5 * kt)).abs() <= 1e-12, "μ = {mu}");
    }

    /// `f` follows `evals`' order, whatever that order is.
    #[test]
    fn fd_order_follows_evals() {
        let kt = ELECTRON_KT;
        let a = fermi_dirac(&[-0.4, -0.31, -0.3, 0.2], 2, kt).unwrap();
        let b = fermi_dirac(&[0.2, -0.3, -0.4, -0.31], 2, kt).unwrap();
        assert_eq!(b.f, vec![a.f[3], a.f[2], a.f[0], a.f[1]]);
        assert_eq!(a.mu, b.mu);
    }

    /// [MUST-1, SHOULD-5] A gap of 0.6 Ha = 120 kT: Σf == 2 in f64 over a plateau of μ, and μ is its midpoint.
    #[test]
    fn fd_gapped_plateau_midpoint() {
        let kt = ELECTRON_KT;
        let o = fermi_dirac(&[-0.5, -0.5, 0.1], 2, kt).unwrap();
        assert_eq!(o.f[0], 1.0);
        assert_eq!(o.f[1], 1.0);
        assert!(o.f[2] > 0.0 && o.f[2] <= (-36.0f64).exp(), "f_LUMO = {:e}", o.f[2]);
        assert!(o.entropy <= 1e-14, "S = {:e}", o.entropy);
        let mu = o.mu.unwrap();
        assert!((-0.5 + 30.0 * kt..=0.1 - 30.0 * kt).contains(&mu), "μ = {mu}");
        assert!((mu - (-0.2 + 0.5 * kt)).abs() <= 2.0 * kt, "μ = {mu}");
        eprintln!("plateau midpoint μ = {mu:.6} (−0.2 + 0.5 kT = {:.6}), halvings {}", -0.2 + 0.5 * kt, o.halvings);
    }

    #[test]
    fn fd_empty_channel() {
        let o = fermi_dirac(&[-0.3, 0.1], 0, ELECTRON_KT).unwrap();
        assert_eq!(o.f, vec![0.0, 0.0]);
        assert_eq!(o.mu, None);
        assert_eq!(o.entropy, 0.0);
        assert_eq!(o.halvings, 0);
        // n = 0 needs no extra level.
        assert_eq!(fermi_dirac(&[], 0, ELECTRON_KT).unwrap().f, Vec::<f64>::new());
    }

    /// 200 spectra shaped like a channel's levels: k ≤ 20 levels in [−spread, 0], spread log-uniform in
    /// [1e-6, 20] Ha (both ends included), 0 < n < k, in random order. After the bisection
    /// |Σf − n| ≲ (slope at μ)·ulp(μ) ≤ k·ulp(μ)/(4kT); levels are sparse where |μ| is large, so ≈ 1e-13.
    #[test]
    fn fd_sums_to_n() {
        let kt = ELECTRON_KT;
        let mut rng = Mix(0x5EED_F00D);
        let (mut worst, mut max_halvings) = (0.0f64, 0usize);
        for t in 0..200 {
            let k = 2 + (rng.next() % 19) as usize;
            let n = 1 + (rng.next() % (k as u64 - 1)) as usize;
            let spread = match t { 0 => 1e-6, 1 => 20.0, _ => 1e-6 * (2e7f64).powf(rng.unit()) };
            let evals: Vec<f64> = (0..k).map(|_| -spread * rng.unit()).collect();
            let o = fermi_dirac(&evals, n, kt).unwrap();
            let dev = (sum(&o.f) - n as f64).abs();
            assert!(dev <= 1e-12, "set {t}: k {k}, n {n}, spread {spread:e}: |Σf − n| = {dev:e}");
            assert!(o.halvings < MAX_MU_HALVINGS, "set {t}: {} halvings", o.halvings);
            assert!(o.f.iter().all(|&f| (0.0..=1.0).contains(&f)));
            worst = worst.max(dev);
            max_halvings = max_halvings.max(o.halvings);
        }
        eprintln!("fd_sums_to_n: max |Σf − n| = {worst:e}, max halvings {max_halvings}");
    }

    /// [SHOULD-1] A NaN never equals a bisection endpoint, so it must be refused before the bisection.
    #[test]
    fn fd_non_finite_is_numerical() {
        for evals in [[-0.3, f64::NAN], [-0.3, f64::INFINITY], [f64::NEG_INFINITY, 0.1]] {
            assert!(matches!(fermi_dirac(&evals, 1, ELECTRON_KT), Err(QmError::Numerical(_))), "{evals:?}");
        }
    }

    /// Too few levels (k < n + 1) and a kT that is not finite and > 0 are refused too.
    #[test]
    fn fd_bad_count_or_kt_is_numerical() {
        assert!(matches!(fermi_dirac(&[-0.3, 0.1], 2, ELECTRON_KT), Err(QmError::Numerical(_))));
        assert!(matches!(fermi_dirac(&[-0.3], 1, ELECTRON_KT), Err(QmError::Numerical(_))));
        for kt in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            assert!(matches!(fermi_dirac(&[-0.3, 0.1], 1, kt), Err(QmError::Numerical(_))), "kT = {kt}");
        }
    }

    /// [NOTE-1] μ at 0, where the f64 spacing reaches subnormals: still far under the backstop.
    #[test]
    fn fd_halvings_near_zero_mu() {
        let o = fermi_dirac(&[-1e-300, 1e-300], 1, ELECTRON_KT).unwrap();
        assert!(o.halvings < MAX_MU_HALVINGS, "{} halvings", o.halvings);
        assert!((sum(&o.f) - 1.0).abs() <= 1e-15);
        eprintln!("μ ≈ 0: μ = {:e}, halvings {}", o.mu.unwrap(), o.halvings);
    }

    #[test]
    fn fd_is_deterministic() {
        let evals = [-0.71, -0.30, -0.297, -0.2955, 0.05, -0.30 + 1e-9];
        let (a, b) = (fermi_dirac(&evals, 3, ELECTRON_KT).unwrap(), fermi_dirac(&evals, 3, ELECTRON_KT).unwrap());
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.f), bits(&b.f));
        assert_eq!(a.mu.unwrap().to_bits(), b.mu.unwrap().to_bits());
        assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
        assert_eq!(a.halvings, b.halvings);
    }

    /// s(x) from x against −f ln f − (1−f) ln(1−f) from f = 1/(1 + eˣ), over f in [1e-12, 1 − 1e-12]: 500 points
    /// log-spaced in [1e-12, ½] and their mirrors 1 − f (the tails are where a cancellation would show).
    #[test]
    fn fd_entropy_matches_direct_formula() {
        let mut worst = 0.0f64;
        for j in 0..500 {
            let low = 10f64.powf(-12.0 + (12.0 - LN_2 / std::f64::consts::LN_10) * j as f64 / 499.0);
            for f in [low, 1.0 - low] {
                let x = ((1.0 - f) / f).ln();
                let direct = -f * f.ln() - (1.0 - f) * (1.0 - f).ln();
                let d = (fd_entropy(x) - direct).abs();
                assert!(d <= 1e-14, "f = {f:e}, x = {x}: |Δs| = {d:e}");
                worst = worst.max(d);
            }
        }
        eprintln!("fd_entropy vs direct: max |Δs| = {worst:e}");
        for x in [800.0, -800.0] {
            let s = fd_entropy(x);
            assert!(s.is_finite() && s <= 1e-300, "s({x}) = {s:e}");
        }
        assert!((fd_entropy(2f64.ln()) - 0.636514).abs() <= 1e-6, "s(⅓) = {}", fd_entropy(2f64.ln()));
        assert!((fd_entropy(0.0) - LN_2).abs() <= 1e-15);
    }

    #[test]
    fn witness_both_arms() {
        // ε_(2) = −0.3, the levels in any order.
        let lvl = |top: f64| vec![-0.3, top, -0.5];
        // Arm 1 fails (ε_max = ε_(n) + 0.099), arm 2 holds (0.199 Ha ≥ 36 kT = 0.18 Ha).
        assert!(!witness_reached(&lvl(-0.3 + 0.099), 2, -0.4, 5e-3));
        // Arm 1 met (ε_max = ε_(n) + 0.2), arm 2 fails at kT = 3e-3: ε_max − μ = 0.105 < 36 kT = 0.108.
        assert!(!witness_reached(&lvl(-0.1), 2, -0.1 - 0.105, 3e-3));
        // Both met.
        assert!(witness_reached(&lvl(-0.1), 2, -0.1 - 0.2, 3e-3));
        // No level above the n-th: never reached.
        assert!(!witness_reached(&[-0.5, -0.3], 2, -0.9, 3e-3));
        // An empty channel needs no witness.
        assert!(witness_reached(&[], 0, 0.0, 3e-3));
    }

    #[test]
    fn partial_levels_threshold() {
        let evals = [0.2, -0.1, 0.05, -0.4, 0.3];
        let f = [1e-6, 0.5, 9.9e-7, 1.0, 1.0 - 9.9e-7];
        let p = partial_levels(&evals, &f, 1);
        assert_eq!(
            p,
            vec![
                PartialLevel { spin: 1, index: 1, eps: -0.1, f: 0.5 },
                PartialLevel { spin: 1, index: 3, eps: 0.2, f: 1e-6 },
            ]
        );
        assert!(partial_levels(&[-0.5, 0.2], &[1.0, 0.0], 0).is_empty());
    }
}
