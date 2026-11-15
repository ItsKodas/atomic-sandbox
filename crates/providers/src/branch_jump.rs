//! The `branch_jump` work-consistency guard (spec 2026-11-13 §6.3, V7/V8; plan Task 8): it flags a step whose change of
//! the free energy F disagrees with the work the forces did beyond what noise and the trapezoid's own error allow — the
//! SCF moved to another stationary state (a branch jump, spec §4.7(g)) — and the surrogate never trains on that step
//! (`sim::EvalStats::untrainable`). It replaces the `DegenerateShells` occupation-pattern watch (spec §8 item 8).
//!
//! # The check (atomic units: Ha, Bohr, Ha/Bohr)
//! Two consecutive evaluations of the same provider, at geometries R₀ and R₁ = R₀ + ΔR with energies E₀, E₁ (the
//! provider's reported total: F under Fermi–Dirac) and forces F_I = −∂E/∂R_I at each:
//!
//! ```text
//! ΔF = E₁ − E₀                                    the energy change
//! W  = −½ Σ_I (F_I,0 + F_I,1)·ΔR_I                the trapezoid estimate of ΔF along the straight path
//! m  = |ΔF − W|                                   the mismatch
//! n  = 3·(√2·σ_E + ‖ΔR‖·σ_F/√2)                   the noise term
//! b  = n + 3·κ̂·‖ΔR‖³,   κ̂ = min(median(window), κ_max)     the bound
//! flagged ⇔ m > b
//! ```
//!
//! **Sign.** Along the path R(s) = R₀ + sΔR, dE/ds = Σ_I ∂E/∂R_I·ΔR_I = −Σ_I F_I(s)·ΔR_I, so
//! ΔF = −∫₀¹ Σ_I F_I(s)·ΔR_I ds; the trapezoid rule over the two ends gives W. On a smooth surface m is the trapezoid
//! error alone, |g″(ξ)|/12 for g(s) = E(R(s)), g″ = Σ ΔR·∇³E·ΔRΔR ≤ |E‴|·‖ΔR‖³ along the step: m ≤ (1/12)|E‴|‖ΔR‖³.
//! κ̂ stands in for (1/12)|E‴| (re-review R-MUST-A).
//!
//! **Noise.** Each solve carries a path scatter σ_E in E and σ_F per force component (the warm/cold difference at the
//! tolerance it was solved at, measured, `references.md` "The branch_jump guard"). ΔF is a difference of two solves:
//! √2·σ_E. W's noise is ½Σ(δF₀ + δF₁)·ΔR: per component (δF₀ + δF₁)/2 has std √2σ_F/2 = σ_F/√2, and its projection
//! on ΔR has std ‖ΔR‖·σ_F/√2. The two are added linearly (≥ their quadrature sum) and the factor 3 makes the noise
//! term a 3σ bound. The constants are taken from `references.md`, never fitted per scene.
//!
//! **κ̂, the window.** κ_med is the median of m/‖ΔR‖³ over the last [`BRANCH_JUMP_WINDOW`] = 8 judged, **unflagged**
//! steps whose third-order term would dominate their noise term (3κ_max‖ΔR‖³ ≥ n). Steps where noise dominates are
//! judged but never enter it — a run of 1e-3 Bohr relax steps, where m is pure noise and m/‖ΔR‖³ ~ 1e3–1e4, cannot
//! inflate κ̂ and blind the guard — and a flagged step never enters it either, so a trajectory that switches branches
//! repeatedly cannot calibrate its own jumps away. Until the window holds 8 entries κ̂ = [`KAPPA_MAX`] alone, so after
//! a reset a 0.1 Bohr drag of a stiff bond is not falsely flagged.
//!
//! **κ_max = 1 Ha/Bohr³** (a global constant, not per element): (1/12)·6Da³ = 0.34 for a Morse bond with D = 0.2 Ha,
//! a = 1.5 Bohr⁻¹ — E‴(R_e) = −6Da³ — rounded up; the "M7 forces" OH gives (1/12)|E‴| ≈ 0.18. With κ̂ ≤ κ_max the bound at
//! a 0.1 Bohr step is ≤ 3e-3 Ha + n, below every jump spec §4.7(e) documents (12–24 mHa). Plan Task 8 step 6's physics
//! review checks it against the measured E‴ of the stiffest app bonds (O–H, N–H at short range).
//!
//! # When a step is judged
//! - Only when z (the species in order, hence the atom count), the cell (n, h) and the grid origin all equal the
//!   previous evaluation's. Any difference **restarts** the history: the window is emptied and the step is not judged
//!   (a box change alters the discrete surface by egg-box and wall terms, re-review R-MUST-A; an added, removed or
//!   swapped atom is another system). The Kohn–Sham adapters also restart it on `reset` (every edit) and on a grid change.
//! - Only when the largest single-atom displacement max_I |ΔR_I| ≤ [`BRANCH_JUMP_REACH`] = 0.1 Bohr: consecutive MD,
//!   relax or drag steps, consecutive teacher calls of a fallback streak. A longer step is not judged (and the window
//!   is kept: the surface is the same).
//! - **Not covered** (rule 7, re-review R-NOTE-b): sampling-pass solves, which are of unrelated perturbed geometries
//!   (> 0.1 Bohr from the previous solve), and the surrogate's sparse background audits, whose teacher calls are usually
//!   more than 0.1 Bohr apart. Under the surrogate only fallback streaks are judged; every other label relies on the
//!   leave-one-out machinery.
//!
//! # Tiers (ruling R8)
//! [`noise_tier`] picks σ_E, σ_F by surface and by the tolerance the solve stopped at: the sample tier when
//! `scf_tol_used ≤` [`SAMPLE_TOL`], the app tier otherwise. Every GPU solve with a partly occupied level stops at
//! `qm::GPU_PARTIAL_TOL` = 1e-5 (spec §5.1), so it is judged on the app tier even when the provider asked for the
//! sample tolerance [kT-5]. The CPU f64 path has a far smaller scatter than the GPU's; it is judged on the same
//! (GPU-measured) tiers, which is the conservative side.
//!
//! # The spin measure
//! With N↑ = N↓ each evaluation can carry ∫|ρ↑ − ρ↓| (electrons, the adapters' spin reduction); a judged step then
//! reports |Δ∫|ρ↑ − ρ↓||, where a restricted ↔ broken-symmetry switch shows. Reported with the verdict, never decided on.
use std::collections::VecDeque;

/// The factor on both the noise term and the third-order term (a 3σ bound; spec §6.3).
pub const BRANCH_JUMP_SIGMAS: f64 = 3.0;
/// The number of judged, unflagged, third-order-dominated steps the running median of m/‖ΔR‖³ is taken over.
pub const BRANCH_JUMP_WINDOW: usize = 8;
/// Bohr: the largest single-atom displacement between two evaluations that is judged.
pub const BRANCH_JUMP_REACH: f64 = 0.1;
/// Ha/Bohr³: the cap on κ̂, and κ̂ itself until the window is full (module doc).
pub const KAPPA_MAX: f64 = 1.0;
/// The sampling teacher's SCF tolerance (`app::worker::SAMPLE_TOL`, private to the app binary): a solve that stopped at
/// or below it is judged on the sample tier.
pub const SAMPLE_TOL: f64 = 1e-6;

/// A surface's per-solve path scatter at one tolerance tier: σ_E (Ha) and σ_F (Ha/Bohr per force component).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseTier {
    pub sigma_e: f64,
    pub sigma_f: f64,
}

/// Pseudo surface, sample tolerance 1e-6: H₂O (`All`) warm vs cold 1.87e-6 Ha (references.md "GPU vs CPU at fixed σ",
/// pseudo path scatter) and `PSEUDO_FORCE_NOISE` (the GPU pseudo path's measured force scatter at 1e-6).
pub const PSEUDO_SAMPLE_TIER: NoiseTier = NoiseTier { sigma_e: 1.87e-6, sigma_f: crate::surrogate::PSEUDO_FORCE_NOISE };
/// All-electron surface, sample tolerance 1e-6: H₂O's GPU path scatter 5.4e-6 Ha (references.md, M3 Task 8) and the
/// force scatter measured by [`tests::tier_scatter_measurement`] (all-electron H₂O at 95³ × 0.15 under FD at 1e-6: the
/// largest of 5 warm-vs-cold repeats 2.35e-5, rounded up; σ_E there 3.6e-6, below the 5.4e-6 kept; references.md
/// "The branch_jump guard").
pub const ALL_ELECTRON_SAMPLE_TIER: NoiseTier = NoiseTier { sigma_e: 5.4e-6, sigma_f: 2.4e-5 };
/// Pseudo surface, app tier (interactive 1e-5, and every partial-level GPU stop at 1e-5): measured by
/// [`tests::tier_scatter_measurement`] at 95³ × 0.15 under FD, the larger of H₂O at `effective_gpu_tol(1e-5)` (σ_E
/// 1.56e-6, σ_F 3.93e-5) and linear OH pinned at 1e-5 by its partial π level (σ_E 5.35e-7, σ_F 8.95e-5), each the
/// largest of 5 warm-vs-cold repeats, rounded up to two digits (references.md "The branch_jump guard").
pub const PSEUDO_APP_TIER: NoiseTier = NoiseTier { sigma_e: 1.6e-6, sigma_f: 9.0e-5 };
/// All-electron surface, app tier: measured by [`tests::tier_scatter_measurement`] — all-electron H₂O at 95³ × 0.15 under
/// FD at 1e-5 (gapped: no partial stop), σ_E 4.66e-6, σ_F 1.35e-5, the largest of 5 warm-vs-cold repeats, rounded up to
/// two digits (references.md "The branch_jump guard"; cold 94 s, warm 38–60 s, no witness cap).
pub const ALL_ELECTRON_APP_TIER: NoiseTier = NoiseTier { sigma_e: 4.7e-6, sigma_f: 1.4e-5 };

/// The scatter tier a solve is judged on (module doc, "Tiers"): by surface, and by the tolerance it stopped at.
pub fn noise_tier(all_electron: bool, scf_tol_used: f64) -> NoiseTier {
    match (all_electron, scf_tol_used <= SAMPLE_TOL) {
        (false, true) => PSEUDO_SAMPLE_TIER,
        (true, true) => ALL_ELECTRON_SAMPLE_TIER,
        (false, false) => PSEUDO_APP_TIER,
        (true, false) => ALL_ELECTRON_APP_TIER,
    }
}

/// One evaluation as the guard sees it: species in order, positions (Bohr), forces (Ha/Bohr), the reported energy
/// (Ha; F under Fermi–Dirac), the cell and the grid origin, ∫|ρ↑ − ρ↓| when N↑ = N↓ (else `None`) and its tier.
pub struct Evaluation<'a> {
    pub z: &'a [u8],
    pub positions: &'a [[f64; 3]],
    pub forces: &'a [[f64; 3]],
    pub energy: f64,
    pub cell: sim::Cell,
    pub origin: [f64; 3],
    pub spin_abs: Option<f64>,
    pub tier: NoiseTier,
}

/// The verdict on one evaluation against the previous one. `judged == false`: nothing compared (`mismatch`, `bound`
/// 0, `flagged` false). `spin_change`: |Δ∫|ρ↑ − ρ↓|| on a judged step where both evaluations carry it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Judgement {
    pub judged: bool,
    pub mismatch: f64,
    pub bound: f64,
    pub flagged: bool,
    pub spin_change: Option<f64>,
}

impl Judgement {
    fn not_judged() -> Judgement {
        Judgement { judged: false, mismatch: 0.0, bound: 0.0, flagged: false, spin_change: None }
    }

    /// `Some(m)` when flagged: what `EvalStats::branch_jump` carries.
    pub fn branch_jump(&self) -> Option<f64> {
        self.flagged.then_some(self.mismatch)
    }
}

/// What the guard keeps of the previous evaluation.
struct Previous {
    z: Vec<u8>,
    positions: Vec<[f64; 3]>,
    forces: Vec<[f64; 3]>,
    energy: f64,
    cell: sim::Cell,
    origin: [f64; 3],
    spin_abs: Option<f64>,
}

/// The guard (module doc): the previous evaluation and the κ window.
#[derive(Default)]
pub struct BranchJumpGuard {
    previous: Option<Previous>,
    window: VecDeque<f64>,
}

/// Neumaier-compensated sum (CLAUDE.md rule 4: energy reductions are compensated).
fn neumaier(terms: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut c) = (0.0f64, 0.0f64);
    for x in terms {
        let t = sum + x;
        c += if sum.abs() >= x.abs() { (sum - t) + x } else { (x - t) + sum };
        sum = t;
    }
    sum + c
}

impl BranchJumpGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forgets the previous evaluation and empties the window: the next evaluation is not judged.
    pub fn reset(&mut self) {
        self.previous = None;
        self.window.clear();
    }

    /// κ̂: κ_max until the window holds [`BRANCH_JUMP_WINDOW`] entries, then min(median, κ_max).
    fn kappa(&self) -> f64 {
        if self.window.len() < BRANCH_JUMP_WINDOW {
            return KAPPA_MAX;
        }
        let mut v: Vec<f64> = self.window.iter().copied().collect();
        v.sort_by(f64::total_cmp);
        let k = v.len();
        let median = if k % 2 == 0 { 0.5 * (v[k / 2 - 1] + v[k / 2]) } else { v[k / 2] };
        median.min(KAPPA_MAX)
    }

    /// Judges `e` against the previous evaluation (module doc) and records `e` as the new previous one.
    pub fn judge(&mut self, e: &Evaluation) -> Judgement {
        let previous = self.previous.replace(Previous {
            z: e.z.to_vec(),
            positions: e.positions.to_vec(),
            forces: e.forces.to_vec(),
            energy: e.energy,
            cell: e.cell.clone(),
            origin: e.origin,
            spin_abs: e.spin_abs,
        });
        let Some(p) = previous else {
            self.window.clear();
            return Judgement::not_judged();
        };
        let same = p.z == e.z && p.positions.len() == e.positions.len() && e.forces.len() == e.positions.len() && p.forces.len() == p.positions.len()
            && p.cell == e.cell && p.origin == e.origin;
        if !same {
            self.window.clear();
            return Judgement::not_judged();
        }
        let dr: Vec<[f64; 3]> = e.positions.iter().zip(&p.positions).map(|(a, b)| std::array::from_fn(|k| a[k] - b[k])).collect();
        let reach = dr.iter().map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()).fold(0.0f64, f64::max);
        if !(reach <= BRANCH_JUMP_REACH) {
            return Judgement::not_judged();
        }
        let norm = neumaier(dr.iter().flatten().map(|x| x * x)).sqrt();
        let delta_f = e.energy - p.energy;
        let work = -0.5 * neumaier(dr.iter().zip(p.forces.iter().zip(e.forces)).flat_map(|(d, (f0, f1))| (0..3).map(move |k| (f0[k] + f1[k]) * d[k])));
        let mismatch = (delta_f - work).abs();
        let noise = BRANCH_JUMP_SIGMAS * (std::f64::consts::SQRT_2 * e.tier.sigma_e + norm * e.tier.sigma_f / std::f64::consts::SQRT_2);
        let cubic = norm * norm * norm;
        let bound = noise + BRANCH_JUMP_SIGMAS * self.kappa() * cubic;
        // a NaN mismatch (non-finite input) is flagged: it is no evidence of a smooth surface
        let flagged = !(mismatch <= bound);
        if !flagged && BRANCH_JUMP_SIGMAS * KAPPA_MAX * cubic >= noise && cubic > 0.0 {
            self.window.push_back(mismatch / cubic);
            if self.window.len() > BRANCH_JUMP_WINDOW {
                self.window.pop_front();
            }
        }
        let spin_change = match (p.spin_abs, e.spin_abs) {
            (Some(a), Some(b)) => Some((b - a).abs()),
            _ => None,
        };
        Judgement { judged: true, mismatch, bound, flagged, spin_change }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The analytic Morse dimer of the guard's tests (plan Task 8 step 2): D = 0.2 Ha, a = 1.5 Bohr⁻¹, R_e = 1.8 Bohr
    /// — the bond κ_max is derived from, (1/12)·6Da³ = 0.34 Ha/Bohr³ at R_e.
    const D: f64 = 0.2;
    const A: f64 = 1.5;
    const R_E: f64 = 1.8;

    /// E and forces of the dimer: atom 0 at `p0`, atom 1 at `p1`.
    fn morse(p0: [f64; 3], p1: [f64; 3]) -> (f64, Vec<[f64; 3]>) {
        let d: [f64; 3] = std::array::from_fn(|k| p1[k] - p0[k]);
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let x = (-A * (r - R_E)).exp();
        let e = D * ((1.0 - x) * (1.0 - x) - 1.0);
        let de = 2.0 * D * A * (1.0 - x) * x;
        let u = d.map(|c| c / r);
        (e, vec![u.map(|c| de * c), u.map(|c| -de * c)])
    }

    /// (1/12)|E‴(r)| of the dimer along its bond.
    fn third(r: f64) -> f64 {
        let x = (-A * (r - R_E)).exp();
        (2.0 * D * A * A * A * (x - 4.0 * x * x)).abs() / 12.0
    }

    fn cell() -> sim::Cell {
        sim::Cell { n: 81, h: 0.15 }
    }

    const ORIGIN: [f64; 3] = [-6.0; 3];
    const Z2: [u8; 2] = [8, 1];
    /// The bond's direction: off the grid axes (the analytic surface has no grid; any direction is the same).
    const DIR: [f64; 3] = [0.6, 0.48, 0.64];

    /// Atom 0 fixed at the origin, atom 1 at `r` along `DIR`; `shift` is added to E (a synthetic branch jump).
    fn judge_at(g: &mut BranchJumpGuard, r: f64, shift: f64) -> Judgement {
        let p = [[0.0; 3], DIR.map(|c| c * r)];
        let (e, f) = morse(p[0], p[1]);
        g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e + shift, cell: cell(), origin: ORIGIN, spin_abs: None, tier: PSEUDO_SAMPLE_TIER })
    }

    #[test]
    fn the_constants_are_the_pre_registered_ones() {
        assert_eq!((BRANCH_JUMP_SIGMAS, BRANCH_JUMP_WINDOW, BRANCH_JUMP_REACH, KAPPA_MAX), (3.0, 8, 0.1, 1.0));
        assert!((DIR.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-15);
        // κ_max against the Morse bond it is derived from: (1/12)·6Da³ at R_e
        assert!((third(R_E) - 6.0 * D * A * A * A / 12.0).abs() < 1e-15 && third(R_E) < KAPPA_MAX);
        assert_eq!(noise_tier(false, 1e-6), PSEUDO_SAMPLE_TIER);
        assert_eq!(noise_tier(true, 1e-6), ALL_ELECTRON_SAMPLE_TIER);
        assert_eq!(noise_tier(false, 1e-5), PSEUDO_APP_TIER);
        assert_eq!(noise_tier(true, qm::GPU_PARTIAL_TOL), ALL_ELECTRON_APP_TIER);
        assert_eq!(noise_tier(false, 1.2e-6), PSEUDO_APP_TIER, "a floor-raised stop above the sample tolerance is the app tier");
    }

    /// The trapezoid W has the sign of ΔF: on a smooth path m is third order, so halving the step divides it by 8.
    #[test]
    fn the_mismatch_is_the_trapezoid_error() {
        for r0 in [1.5, 1.8, 2.4] {
            let mut ms = Vec::new();
            for h in [0.02, 0.01] {
                let mut g = BranchJumpGuard::new();
                judge_at(&mut g, r0, 0.0);
                let j = judge_at(&mut g, r0 + h, 0.0);
                assert!(j.judged && !j.flagged);
                // the trapezoid error of g(s) = E(r0 + s h): h³ E‴/12 at the midpoint, to O(h⁴)
                let expect = third(r0 + 0.5 * h) * h * h * h;
                assert!((j.mismatch - expect).abs() <= 0.05 * expect + 1e-15, "r0 {r0} h {h}: m {:e} vs {:e}", j.mismatch, expect);
                ms.push(j.mismatch);
            }
            assert!((ms[0] / ms[1] - 8.0).abs() < 0.5, "third order: {ms:?}");
        }
    }

    /// From step 10 on, E is shifted by +12 mHa (0.02 Bohr steps): flagged at step 10 only.
    #[test]
    fn synthetic_jump_is_flagged() {
        let mut g = BranchJumpGuard::new();
        for i in 0..30 {
            let j = judge_at(&mut g, 1.6 + 0.02 * i as f64, if i >= 10 { 0.012 } else { 0.0 });
            assert_eq!(j.judged, i > 0, "step {i}");
            assert_eq!(j.flagged, i == 10, "step {i}: m {:e} b {:e}", j.mismatch, j.bound);
            if i == 10 {
                println!("synthetic jump at a 0.02 Bohr step: m {:.4e}, b {:.3e}", j.mismatch, j.bound);
                assert!((j.mismatch - 0.012).abs() < 1e-4, "m ≈ the jump: {:e}", j.mismatch);
                assert_eq!(j.branch_jump(), Some(j.mismatch));
            }
        }
    }

    /// Steps of 0.005, 0.02 and 0.05 Bohr (‖ΔR‖) across R ∈ [0.8, 2.0]·R_e, out and back, with one atom moving (a
    /// heavy–light bond: the bond length changes by ‖ΔR‖) and with both moving symmetrically (equal masses: the bond
    /// changes by √2‖ΔR‖, so m/‖ΔR‖³ is 2√2 times (1/12)|E‴|), plus one 0.1 Bohr step with an empty window: zero
    /// flags. Prints the tightest m/b, which the physics review reads (the symmetric 0.05 Bohr path at the compressed
    /// end is the tight case: (1/12)|E‴| reaches 1.1 there, and ×2√2 = 3.2 against the bound's 3κ_max).
    #[test]
    fn smooth_morse_never_flagged() {
        let (lo, hi) = (0.8 * R_E, 2.0 * R_E);
        for symmetric in [false, true] {
            for step in [0.005, 0.02, 0.05] {
                let bond_step = if symmetric { std::f64::consts::SQRT_2 * step } else { step };
                let mut g = BranchJumpGuard::new();
                let k = ((hi - lo) / bond_step).round() as usize;
                let path: Vec<f64> = (0..=k).map(|i| lo + bond_step * i as f64).chain((0..k).rev().map(|i| lo + bond_step * i as f64)).collect();
                let (mut worst, mut at) = (0.0f64, 0.0);
                for (i, &r) in path.iter().enumerate() {
                    let p = if symmetric { [DIR.map(|c| -0.5 * c * r), DIR.map(|c| 0.5 * c * r)] } else { [[0.0; 3], DIR.map(|c| c * r)] };
                    let (e, f) = morse(p[0], p[1]);
                    let j = g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e, cell: cell(), origin: ORIGIN, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
                    assert!(!j.flagged, "{} {step}: flagged at r = {r} (point {i}): m {:e} b {:e}", if symmetric { "symmetric" } else { "one atom" }, j.mismatch, j.bound);
                    if j.judged && j.mismatch / j.bound > worst {
                        (worst, at) = (j.mismatch / j.bound, r);
                    }
                }
                println!("smooth Morse, {}, ‖ΔR‖ = {step} Bohr: worst m/b {worst:.3} at R = {at:.3}", if symmetric { "both atoms" } else { "one atom" });
            }
        }
        // a 0.1 Bohr step right after a reset (κ̂ = κ_max), at the compressed end where |E‴| is largest
        for r in [0.8 * R_E, R_E, 2.0 * R_E - 0.1] {
            let mut g = BranchJumpGuard::new();
            judge_at(&mut g, r, 0.0);
            // 0.1 Bohr to 1e-12, so the rounding of r + 0.1 cannot put it past the reach
            let j = judge_at(&mut g, r + 0.1 * (1.0 - 1e-12), 0.0);
            assert!(j.judged && !j.flagged, "0.1 Bohr from {r}: m {:e} b {:e}", j.mismatch, j.bound);
            assert!(j.bound <= 3e-3 + 3.0 * (std::f64::consts::SQRT_2 * 1.87e-6 + 0.1 * 5e-5 / std::f64::consts::SQRT_2) + 1e-15, "b ≤ 3 mHa + noise at 0.1 Bohr");
        }
    }

    /// One atom moving 0.11 Bohr is not judged (and the window is kept: the surface is the same).
    #[test]
    fn over_reach_not_judged() {
        let mut g = BranchJumpGuard::new();
        for i in 0..12 {
            judge_at(&mut g, 1.6 + 0.02 * i as f64, 0.0);
        }
        let kept = g.window.len();
        assert_eq!(kept, BRANCH_JUMP_WINDOW);
        let j = judge_at(&mut g, 1.82 + 0.11, 0.012);
        assert!(!j.judged && !j.flagged && j.branch_jump().is_none());
        assert_eq!(g.window.len(), kept);
        // just inside the reach it is judged
        let j = judge_at(&mut g, 1.82 + 0.11 + 0.099, 0.0);
        assert!(j.judged);
    }

    /// 50 steps of 1e-3 Bohr (noise-dominated: the window stays empty), then a 0.02 Bohr step carrying +12 mHa: flagged.
    #[test]
    fn relax_then_jump_flagged() {
        let mut g = BranchJumpGuard::new();
        for i in 0..=50 {
            let j = judge_at(&mut g, 1.7 + 1e-3 * i as f64, 0.0);
            assert!(!j.flagged);
        }
        assert!(g.window.is_empty(), "noise-dominated steps never enter the window: {:?}", g.window);
        let j = judge_at(&mut g, 1.75 + 0.02, 0.012);
        assert!(j.judged && j.flagged, "m {:e} b {:e}", j.mismatch, j.bound);
    }

    /// ±12 mHa on every other 0.02 Bohr step for 40 steps: every jump is flagged, every step between is not, and no
    /// flagged step ever enters the window.
    #[test]
    fn alternating_jumps_stay_flagged() {
        let mut g = BranchJumpGuard::new();
        let mut shift = 0.0;
        for i in 0..40 {
            let jump = i > 0 && i % 2 == 0;
            if jump {
                shift = if shift == 0.0 { 0.012 } else { 0.0 };
            }
            let j = judge_at(&mut g, 1.6 + 0.02 * i as f64, shift);
            assert_eq!(j.flagged, jump, "step {i}: m {:e} b {:e}", j.mismatch, j.bound);
            assert!(g.window.iter().all(|&k| k < KAPPA_MAX), "step {i}: a jump entered the window {:?}", g.window);
        }
        assert_eq!(g.window.len(), BRANCH_JUMP_WINDOW, "the smooth steps between fill it");
    }

    /// The cell's n, its h or the origin changes: not judged, the window is emptied.
    #[test]
    fn cell_or_origin_change_restarts() {
        type Edit = fn(&mut sim::Cell, &mut [f64; 3]);
        let edits: [(&str, Edit); 3] = [("n", |c, _| c.n += 2), ("h", |c, _| c.h = 0.25), ("origin", |_, o| o[1] += 0.15)];
        for (what, edit) in edits {
            let mut g = BranchJumpGuard::new();
            for i in 0..12 {
                judge_at(&mut g, 1.6 + 0.02 * i as f64, 0.0);
            }
            assert_eq!(g.window.len(), BRANCH_JUMP_WINDOW);
            let p = [[0.0; 3], DIR.map(|c| c * 1.84)];
            let (e, f) = morse(p[0], p[1]);
            let (mut c, mut o) = (cell(), ORIGIN);
            edit(&mut c, &mut o);
            // a box change moves the discrete surface: here by a "jump" that must not be flagged
            let j = g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e + 0.012, cell: c.clone(), origin: o, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
            assert!(!j.judged && !j.flagged, "{what}");
            assert!(g.window.is_empty(), "{what}: the window restarts");
            // the next step on the new box is judged against the new box's evaluation
            let p = [[0.0; 3], DIR.map(|c| c * 1.86)];
            let (e2, f2) = morse(p[0], p[1]);
            let j = g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f2, energy: e2 + 0.012, cell: c, origin: o, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
            assert!(j.judged && !j.flagged, "{what}: m {:e}", j.mismatch);
        }
    }

    /// Review Focus 2: an atom added, one removed, two species swapped — not judged, no panic, the window restarts.
    #[test]
    fn atom_set_change_restarts() {
        let warm = |g: &mut BranchJumpGuard| {
            for i in 0..12 {
                judge_at(g, 1.6 + 0.02 * i as f64, 0.0);
            }
            assert_eq!(g.window.len(), BRANCH_JUMP_WINDOW);
        };
        let p2 = [[0.0; 3], DIR.map(|c| c * 1.84)];
        let (e2, f2) = morse(p2[0], p2[1]);
        let p3 = [p2[0], p2[1], [3.0, 0.0, 0.0]];
        let f3 = [f2[0], f2[1], [0.0; 3]];
        let p1 = [p2[0]];
        let f1 = [[0.0; 3]];
        let cases: [(&str, &[u8], &[[f64; 3]], &[[f64; 3]]); 3] = [("added", &[8, 1, 1], &p3, &f3), ("removed", &[8], &p1, &f1), ("swapped", &[1, 8], &p2, &f2)];
        for (what, z, p, f) in cases {
            let mut g = BranchJumpGuard::new();
            warm(&mut g);
            let j = g.judge(&Evaluation { z, positions: p, forces: f, energy: e2, cell: cell(), origin: ORIGIN, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
            assert!(!j.judged && !j.flagged, "{what}");
            assert!(g.window.is_empty(), "{what}");
            // and back to the dimer: another restart, never a panic on the shorter or longer previous
            let j = judge_at(&mut g, 1.84, 0.0);
            assert!(!j.judged, "{what}: back");
        }
        // forces of another length than the positions are no evidence either (a malformed answer): not judged
        let mut g = BranchJumpGuard::new();
        warm(&mut g);
        let j = g.judge(&Evaluation { z: &Z2, positions: &p2, forces: &f1, energy: e2, cell: cell(), origin: ORIGIN, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
        assert!(!j.judged);
    }

    /// The spin measure: reported on a judged step where both evaluations carry ∫|ρ↑ − ρ↓|; never on a restart.
    #[test]
    fn spin_change_is_reported_on_judged_steps() {
        let mut g = BranchJumpGuard::new();
        let at = |g: &mut BranchJumpGuard, r: f64, s: Option<f64>| {
            let p = [[0.0; 3], DIR.map(|c| c * r)];
            let (e, f) = morse(p[0], p[1]);
            g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e, cell: cell(), origin: ORIGIN, spin_abs: s, tier: PSEUDO_SAMPLE_TIER })
        };
        assert_eq!(at(&mut g, 1.8, Some(1e-5)).spin_change, None, "first evaluation");
        let j = at(&mut g, 1.82, Some(0.75));
        assert!(j.judged && (j.spin_change.unwrap() - (0.75 - 1e-5)).abs() < 1e-15);
        assert_eq!(at(&mut g, 1.84, None).spin_change, None, "N↑ ≠ N↓");
        g.reset();
        assert_eq!(at(&mut g, 1.86, Some(0.7)).spin_change, None, "after a reset");
    }

    /// `reset` forgets the previous evaluation and the window.
    #[test]
    fn reset_restarts() {
        let mut g = BranchJumpGuard::new();
        for i in 0..12 {
            judge_at(&mut g, 1.6 + 0.02 * i as f64, 0.0);
        }
        g.reset();
        assert!(g.window.is_empty());
        let j = judge_at(&mut g, 1.84, 0.012);
        assert!(!j.judged);
    }

    /// Plan Task 8 step 1 [kT-5]: the per-solve path scatter behind the tiers (references.md "The branch_jump guard").
    /// The providers still run the old occupation rule, so `GpuSolver` is driven directly with explicit
    /// `FermiDirac { kt: ELECTRON_KT }` on the scene `ks::problem` builds, at the app's Accurate grid for the H₂O preset
    /// (its 14 Bohr box: 95³ × 0.15), density extrapolation off. One cold solve at R₀; then 5 repeats of: a warm solve at
    /// a neighbour R₀ + δ (every atom moved 0.02 Bohr in a seeded random direction), a warm solve back at R₀. Per
    /// repeat |E_warm − E_cold| and the largest per-component |F_warm − F_cold|; σ_E and σ_F are the largest over the 5
    /// (each a difference of two solves: conservative by up to √2 against a per-solve σ, as the sample tier's
    /// 1.87e-6 is). Cases (`BRANCH_JUMP_TIER_CASE`): `pseudo_h2o` (`Accepted`, `effective_gpu_tol(1e-5)`), `oh` (linear
    /// OH along x, R = 1.85, `Accepted`, the sample tolerance 1e-6: its partial π level stops it at
    /// `GPU_PARTIAL_TOL`), `ae_h2o` (`IonicPolicy::None`, 1e-5) and `ae_h2o_sample` (`None`, 1e-6: the all-electron
    /// sample tier's σ_F). A solve that takes over 5 min, or a case over 14 min (the gentle-GPU batch), is abandoned and reported as not
    /// measured.
    #[test]
    #[ignore = "measurement (GPU, minutes): BRANCH_JUMP_TIER_CASE=<case> cargo test -p providers --release tier_scatter_measurement -- --ignored --nocapture"]
    fn tier_scatter_measurement() {
        use rand::{Rng, SeedableRng};
        let Some(ctx) = gpu::GpuContext::new_headless() else {
            eprintln!("tier_scatter_measurement: no GPU adapter — skipped");
            return;
        };
        crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx.clone()));
        let case = std::env::var("BRANCH_JUMP_TIER_CASE").unwrap_or_else(|_| "pseudo_h2o".into());
        let h2o = vec![sim::Nucleus::new(8, [0.0, 0.0, 0.0]), sim::Nucleus::new(1, [1.43, 1.11, 0.0]), sim::Nucleus::new(1, [-1.43, 1.11, 0.0])];
        let oh = vec![sim::Nucleus::new(8, [-0.6, 0.0, 0.0]), sim::Nucleus::new(1, [1.25, 0.0, 0.0])];
        let (nuclei, ionic, tol) = match case.as_str() {
            "pseudo_h2o" => (h2o, sim::IonicPolicy::Accepted, 1e-5),
            "oh" => (oh, sim::IonicPolicy::Accepted, SAMPLE_TOL),
            "ae_h2o" => (h2o, sim::IonicPolicy::None, 1e-5),
            "ae_h2o_sample" => (h2o, sim::IonicPolicy::None, SAMPLE_TOL),
            other => panic!("unknown case {other}"),
        };
        let scene = sim::Scene { nuclei, cell: sim::Cell { n: 95, h: 0.15 }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic, train_mask: None };
        let mut notes = Vec::new();
        let (grid, system, mut settings) = crate::ks::problem(&scene, tol, &mut notes).expect("a valid scene");
        settings.occupations = qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT };
        if system.ionic.is_some() {
            settings.tol = crate::pseudo::effective_gpu_tol(tol, (system.n_up + system.n_dn) as i64);
        }
        println!("case {case}: {}³ × {}, {:?}, N↑ {} N↓ {}, tol {:.1e} (asked {tol:.0e}), FD kT {}, notes {notes:?}",
                 grid.n[0], grid.h, scene.ionic, system.n_up, system.n_dn, settings.tol, qm::ELECTRON_KT);
        let mut solver = qm_gpu::GpuSolver::new(ctx, &grid);
        solver.set_density_extrapolation(false);
        // the gentle-GPU batch limit: the whole case stops at 14 min, whatever is left unmeasured
        let start = std::time::Instant::now();
        let solve = |solver: &mut qm_gpu::GpuSolver, nuclei: &[qm::Nucleus], warm: bool, what: &str| -> Option<(f64, Vec<[f64; 3]>)> {
            let sys = qm::System { nuclei: nuclei.to_vec(), n_up: system.n_up, n_dn: system.n_dn, ionic: system.ionic.clone() };
            let t = std::time::Instant::now();
            let r = solver.solve_with(&sys, &settings, warm, &mut || t.elapsed().as_secs() >= 300 || start.elapsed().as_secs() >= 840).map(|r| r.map(|(_, st)| st));
            let st = match r {
                Ok(Some(v)) => v,
                Ok(None) => {
                    println!("  {what}: abandoned after {:.0} s (5 min per solve, 14 min per case) — not measured", t.elapsed().as_secs_f64());
                    return None;
                }
                Err(e) => {
                    println!("  {what}: {e} — not measured");
                    return None;
                }
            };
            let state = solver.state().expect("a converged solve exposes its state");
            let f = solver.forces(state, &sys.nuclei, settings.sigma).expect("forces");
            println!("  {what}: {} it, {:.1} s, tol_used {:.1e}, F {:.9}, TS {:.2e}, partial {:?}, μ {:?}, eig res {:.1e}, notes {:?}",
                     st.iterations, t.elapsed().as_secs_f64(), st.tol_used, st.energy.total, st.energy.ts,
                     st.partial.iter().map(|p| (p.spin, p.index, (p.f * 1e3).round() / 1e3)).collect::<Vec<_>>(), st.mu, st.max_eigen_residual, solver.profile().notes);
            Some((st.energy.total, f))
        };
        let r0: Vec<qm::Nucleus> = system.nuclei.clone();
        let Some((e_cold, f_cold)) = solve(&mut solver, &r0, false, "cold R₀") else { return };
        let mut rng = rand::rngs::StdRng::seed_from_u64(8);
        let (mut sig_e, mut sig_f) = (0.0f64, 0.0f64);
        let mut rows = Vec::new();
        for k in 0..5 {
            let moved: Vec<qm::Nucleus> = r0.iter().map(|n| {
                let (u, phi): (f64, f64) = (rng.gen_range(-1.0..1.0), rng.gen_range(0.0..std::f64::consts::TAU));
                let s = (1.0 - u * u).sqrt();
                let d = [s * phi.cos(), s * phi.sin(), u].map(|c| 0.02 * c);
                qm::Nucleus { pos: std::array::from_fn(|a| n.pos[a] + d[a]), z: n.z }
            }).collect();
            if solve(&mut solver, &moved, true, &format!("repeat {k}: warm neighbour")).is_none() { break }
            let Some((e, f)) = solve(&mut solver, &r0, true, &format!("repeat {k}: warm R₀")) else { break };
            let de = (e - e_cold).abs();
            let df = f.iter().zip(&f_cold).flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs())).fold(0.0f64, f64::max);
            (sig_e, sig_f) = (sig_e.max(de), sig_f.max(df));
            rows.push((de, df));
            println!("  repeat {k}: |E_warm − E_cold| {de:.2e}, max |ΔF| {df:.2e}");
        }
        if rows.len() < 5 {
            println!("case {case}: only {} of 5 repeats measured", rows.len());
            if rows.is_empty() {
                return;
            }
        }
        println!("case {case}: |E_warm − E_cold| {:?}", rows.iter().map(|r| format!("{:.2e}", r.0)).collect::<Vec<_>>());
        println!("case {case}: max |ΔF| per repeat {:?}", rows.iter().map(|r| format!("{:.2e}", r.1)).collect::<Vec<_>>());
        let rms = |v: Vec<f64>| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        println!("case {case}: σ_E (max) {sig_e:.3e} (rms {:.3e}), σ_F (max) {sig_f:.3e} (rms of the per-repeat max {:.3e})",
                 rms(rows.iter().map(|r| r.0).collect()), rms(rows.iter().map(|r| r.1).collect()));
    }
}
