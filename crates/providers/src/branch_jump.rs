//! The `branch_jump` work-consistency guard (spec 2026-11-13 §6.3, V7/V8; plan Task 8): it flags a step whose change of
//! the free energy F disagrees with the work the forces did beyond what noise and the trapezoid's own error allow — the
//! SCF moved to another stationary state (a branch jump, spec §4.7(g)) — and the surrogate never trains on that step
//! (`sim::EvalStats::untrainable`). It replaces the `DegenerateShells` occupation-pattern watch (spec §8 item 8).
//!
//! # The check (atomic units: Ha, Bohr, Ha/Bohr)
//! Two consecutive evaluations of the same provider, at geometries R₀ and R₁ = R₀ + ΔR with energies E₀, E₁ (the
//! provider's reported total: F under Fermi–Dirac), forces F_I = −∂E/∂R_I and noise tiers (σ_E,0, σ_F,0), (σ_E,1,
//! σ_F,1) at each:
//!
//! ```text
//! ΔF = E₁ − E₀                                          the energy change
//! W  = −½ Σ_I (F_I,0 + F_I,1)·ΔR_I                      the trapezoid estimate of ΔF along the straight path
//! m  = |ΔF − W|                                         the mismatch
//! n  = 3·(√(σ_E,0² + σ_E,1²) + ½‖ΔR‖·√(σ_F,0² + σ_F,1²))  the noise term
//! b  = n + 3·κ_max·‖ΔR‖³                                the bound
//! flagged ⇔ m > b
//! ```
//!
//! **Sign.** Along the path R(s) = R₀ + sΔR, g(s) = E(R(s)), g′(s) = Σ_I ∂E/∂R_I·ΔR_I = −Σ_I F_I(s)·ΔR_I, so
//! ΔF = ∫₀¹ g′ ds and the trapezoid rule on g′ over the two ends gives W. On a smooth surface m is the trapezoid error
//! alone, |g‴(ξ)|/12 for some ξ in the step, with g‴ = ∂³E/∂s³ = ‖ΔR‖³·∂³E/∂u³ along the unit 3N direction
//! u = ΔR/‖ΔR‖: m = (1/12)|∂³E/∂u³|·‖ΔR‖³. κ_max bounds (1/12)|∂³E/∂u³| (re-review R-MUST-A).
//!
//! **Noise.** Each solve carries a path scatter σ_E in E and σ_F per force component (the warm/cold difference at the
//! tolerance it was solved at, measured, `references.md` "The branch_jump guard"). ΔF is a difference of two solves:
//! std √(σ_E,0² + σ_E,1²). W's noise is ½Σ(δF₀ + δF₁)·ΔR: per component ½(δF₀ + δF₁) has std ½√(σ_F,0² + σ_F,1²), and
//! its projection on ΔR has std ½‖ΔR‖√(σ_F,0² + σ_F,1²). Each evaluation brings its own tier (review M4: a step whose
//! stop moved between 1e-6 and 1e-5 is judged on both). With equal tiers this is √2σ_E and ‖ΔR‖σ_F/√2. The two terms
//! are added linearly (≥ their quadrature sum) and the factor 3 makes the noise term a 3σ bound. The constants are taken
//! from `references.md`, never fitted per scene.
//!
//! **κ_max = 1 Ha/Bohr³, always** (a global constant, not per element). It is (1/12)·6Da³ = 0.34 for a Morse bond with
//! D = 0.2 Ha, a = 1.5 Bohr⁻¹ — E‴(R_e) = −6Da³ — rounded up; the "M7 forces" OH gives (1/12)|E‴| ≈ 0.18. Along one bond
//! of length r, ∂³E/∂u³ is the bond's V‴(r) times (dr/ds)³/‖ΔR‖³: 1 when one atom moves, **2√2** when two equal masses
//! move oppositely (r changes by √2‖ΔR‖), ≈ 1.19 for O–H in MD (momentum conservation: H moves 16× O). The measured
//! (1/12)|V‴| at 0.8 R_e, 0.40 (O–H in H₂O) and 0.35 (N–H in NH₃), give 1.13 and 0.99 with the 2√2 — under 3κ_max = 3,
//! which the factor 3 covers (review M2).
//!
//! **Why no running median** (fix round 1, review C1; spec §6.3 amended). The first build let κ̂ = min(median of
//! m/‖ΔR‖³ over the last 8 judged, unflagged steps, κ_max). The running median is biased low on large-amplitude bond
//! motion and locks in, because flagged steps never enter it: Morse V‴ ∝ x − 4x² vanishes at R_e + ln 4/a, steps there
//! fill the window with m/‖ΔR‖³ ≈ 0.001–0.005, the third derivative then rises 100–300× within a few steps into the well,
//! and those steps are flagged and never admitted. A replica flagged 655 of 2999 steps of H₂ forming from rest at 4 Bohr
//! (Task 8 review C1). The guard flags on the derived bound alone.
//!
//! **What it can see (rule 7; review I2).** The reach is per atom, the bound uses the 3N norm: with N atoms each moving
//! δ, ‖ΔR‖ = √N·δ and b ≈ 3κ_max·N^{3/2}·δ³ + n. At the reach (δ = 0.1 Bohr) the third-order term is 3.0, 8.5, 15.6,
//! 33.5 and 94.9 mHa for N = 1, 2, 3, 5 and 10. **A 12 mHa jump (the smallest spec §4.7(e) documents, 12–24 mHa) is not
//! detectable on a step that moves three or more atoms near the reach**; it is detectable on MD steps, where an H moves
//! about 0.013, 0.024 and 0.041 Bohr per 0.25 fs step at 300, 1000 and 3000 K (N = 10 at 0.041 Bohr: 6.5 mHa + n).
//!
//! # When a step is judged
//! - Only when z (the species in order, hence the atom count), the cell (n, h) and the grid origin all equal the
//!   previous evaluation's. Any difference **restarts** the history: the step is not judged (a box change alters the
//!   discrete surface by egg-box and wall terms, re-review R-MUST-A; an added, removed or swapped atom is another
//!   system). The Kohn–Sham adapters also restart it on `reset` (every edit) and on a grid change.
//! - Only when the largest single-atom displacement max_I |ΔR_I| ≤ [`BRANCH_JUMP_REACH`] = 0.1 Bohr: consecutive MD,
//!   relax or drag steps, consecutive teacher calls of a fallback streak. A longer step is not judged.
//! - **Not covered** (rule 7, re-review R-NOTE-b): sampling-pass solves, whose perturbed geometries are usually more
//!   than 0.1 Bohr from the previous solve, and the surrogate's sparse background audits, whose teacher calls are
//!   usually more than 0.1 Bohr apart. Dense scans and teach path points closer than 0.1 Bohr are judged (review M8).
//!   Under the surrogate otherwise only fallback streaks are judged; every other label relies on the leave-one-out
//!   machinery.
//!
//! # Tiers (ruling R8)
//! [`noise_tier`] picks σ_E, σ_F by surface and by the tolerance the solve stopped at: the sample tier when
//! `scf_tol_used ≤` [`SAMPLE_TOL`], the app tier otherwise. Every GPU solve with a partly occupied level stops at
//! `qm::GPU_PARTIAL_TOL` = 1e-5 (spec §5.1), so it is judged on the app tier even when the provider asked for the
//! sample tolerance [kT-5]. The CPU f64 path has a far smaller scatter than the GPU's; it is judged on the same
//! (GPU-measured) tiers, which is the conservative side. σ_E is raised to c·|E| where that is larger (review I1:
//! the f32 path scatter tracks |E|, c = 9e-8 pseudo, 1.0e-7 all-electron from the H₂O measurements; inert on every
//! preset, which all carry ≤ 8 valence electrons, and protecting larger scenes); σ_F has no such term.
//!
//! # The spin measure
//! With N↑ = N↓ each evaluation can carry ∫|ρ↑ − ρ↓| (electrons, the adapters' spin reduction); a judged step then
//! reports |Δ∫|ρ↑ − ρ↓||, where a restricted ↔ broken-symmetry switch shows. Reported with the verdict, never decided on.

/// The factor on both the noise term and the third-order term (a 3σ bound; spec §6.3).
pub const BRANCH_JUMP_SIGMAS: f64 = 3.0;
/// Bohr: the largest single-atom displacement between two evaluations that is judged.
pub const BRANCH_JUMP_REACH: f64 = 0.1;
/// Ha/Bohr³: the bound on (1/12)|∂³E/∂u³| along the step's unit 3N direction (module doc).
pub const KAPPA_MAX: f64 = 1.0;
/// The sampling teacher's SCF tolerance (`app::worker::SAMPLE_TOL`, pinned equal there): a solve that stopped at or
/// below it is judged on the sample tier.
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

/// The relative energy scatter of the pseudo surface (fix round 1, review I1): σ_E ≥ this × |E|. From the H₂O
/// measurement at 95³ × 0.15, 1e-5: 1.56e-6 / 17.477 Ha = 8.9e-8, rounded up (references.md "The branch_jump guard").
pub const PSEUDO_RELATIVE_SIGMA_E: f64 = 9e-8;
/// The relative energy scatter of the all-electron surface: all-electron H₂O at 95³ × 0.15, the larger of its two
/// tolerances, 4.66e-6 / 46.558 Ha = 1.0e-7.
pub const ALL_ELECTRON_RELATIVE_SIGMA_E: f64 = 1.0e-7;

/// The scatter tier a solve is judged on (module doc, "Tiers"): by surface, and by the tolerance it stopped at, with
/// σ_E = max(the tier's σ_E, c·|energy|) — one global rule per surface (fix round 1, review I1: the f32 path scatter
/// tracks |E|, 8.9e-8·|E| pseudo and 1.0e-7·|E| all-electron on H₂O, so a larger system than the H₂O the tiers were
/// measured on gets a proportionally larger σ_E). σ_F has no such term: the measurements show no growth.
pub fn noise_tier(all_electron: bool, scf_tol_used: f64, energy: f64) -> NoiseTier {
    let (tier, c) = match (all_electron, scf_tol_used <= SAMPLE_TOL) {
        (false, true) => (PSEUDO_SAMPLE_TIER, PSEUDO_RELATIVE_SIGMA_E),
        (true, true) => (ALL_ELECTRON_SAMPLE_TIER, ALL_ELECTRON_RELATIVE_SIGMA_E),
        (false, false) => (PSEUDO_APP_TIER, PSEUDO_RELATIVE_SIGMA_E),
        (true, false) => (ALL_ELECTRON_APP_TIER, ALL_ELECTRON_RELATIVE_SIGMA_E),
    };
    // a non-finite energy leaves the tier as it is (the mismatch is then NaN and flagged)
    let scaled = c * energy.abs();
    NoiseTier { sigma_e: if scaled.is_finite() { tier.sigma_e.max(scaled) } else { tier.sigma_e }, ..tier }
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
    tier: NoiseTier,
}

/// The guard (module doc): the previous evaluation.
#[derive(Default)]
pub struct BranchJumpGuard {
    previous: Option<Previous>,
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

    /// Forgets the previous evaluation: the next evaluation is not judged.
    pub fn reset(&mut self) {
        self.previous = None;
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
            tier: e.tier,
        });
        let Some(p) = previous else {
            return Judgement::not_judged();
        };
        let same = p.z == e.z && p.positions.len() == e.positions.len() && e.forces.len() == e.positions.len() && p.forces.len() == p.positions.len()
            && p.cell == e.cell && p.origin == e.origin;
        if !same {
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
        // each evaluation's own tier (review M4)
        let sigma_e = p.tier.sigma_e.hypot(e.tier.sigma_e);
        let sigma_f = p.tier.sigma_f.hypot(e.tier.sigma_f);
        let noise = BRANCH_JUMP_SIGMAS * (sigma_e + 0.5 * norm * sigma_f);
        let bound = noise + BRANCH_JUMP_SIGMAS * KAPPA_MAX * norm * norm * norm;
        // a NaN mismatch (non-finite input) is flagged: it is no evidence of a smooth surface
        let flagged = !(mismatch <= bound);
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
        // fix round 1 (review C1): the window constant is gone with the running median; 3, 0.1 and κ_max = 1 stay
        assert_eq!((BRANCH_JUMP_SIGMAS, BRANCH_JUMP_REACH, KAPPA_MAX), (3.0, 0.1, 1.0));
        assert!((DIR.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-15);
        // κ_max against the Morse bond it is derived from: (1/12)·6Da³ at R_e
        assert!((third(R_E) - 6.0 * D * A * A * A / 12.0).abs() < 1e-15 && third(R_E) < KAPPA_MAX);
        // the tiers at the H₂O energies they were measured on: the |E| term does not bind there
        assert_eq!(noise_tier(false, 1e-6, -17.477), PSEUDO_SAMPLE_TIER);
        assert_eq!(noise_tier(true, 1e-6, -46.558), ALL_ELECTRON_SAMPLE_TIER);
        assert_eq!(noise_tier(false, 1e-5, -17.477), PSEUDO_APP_TIER);
        assert_eq!(noise_tier(true, qm::GPU_PARTIAL_TOL, -46.558), ALL_ELECTRON_APP_TIER);
        assert_eq!(noise_tier(false, 1.2e-6, -17.477), PSEUDO_APP_TIER, "a floor-raised stop above the sample tolerance is the app tier");
        assert_eq!((PSEUDO_RELATIVE_SIGMA_E, ALL_ELECTRON_RELATIVE_SIGMA_E), (9e-8, 1.0e-7));
    }

    /// Review I1's rule: σ_E = max(tier, c·|E|), per surface; σ_F untouched; a NaN energy keeps the tier.
    #[test]
    fn the_energy_scatter_scales_with_the_energy() {
        // the largest preset (SiH₄, pseudo |E| 6.45 Ha): the tier binds; its measured scatter 5.2e-7 is under both
        assert_eq!(noise_tier(false, 1e-5, -6.454), PSEUDO_APP_TIER);
        assert!(PSEUDO_RELATIVE_SIGMA_E * 6.454 >= 5.2e-7);
        // a system ten times H₂O's |E|: σ_E grows with it, σ_F does not
        let big = noise_tier(false, 1e-5, -174.77);
        assert!((big.sigma_e - 9e-8 * 174.77).abs() < 1e-18 && big.sigma_f == PSEUDO_APP_TIER.sigma_f, "{big:?}");
        let big = noise_tier(true, 1e-6, -465.58);
        assert!((big.sigma_e - 1.0e-7 * 465.58).abs() < 1e-18 && big.sigma_f == ALL_ELECTRON_SAMPLE_TIER.sigma_f, "{big:?}");
        assert_eq!(noise_tier(true, 1e-5, f64::NAN), ALL_ELECTRON_APP_TIER);
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
    /// changes by √2‖ΔR‖, so m/‖ΔR‖³ is 2√2 times (1/12)|E‴|), plus single 0.1 Bohr steps: zero flags. Prints the
    /// tightest m/b, which the physics review reads (the symmetric 0.05 Bohr path at the compressed end is the tight
    /// case: (1/12)|E‴| reaches 1.1 there, and ×2√2 = 3.2 against the bound's 3κ_max = 3, covered by the third
    /// derivative falling over the step and the noise term).
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
        // a 0.1 Bohr step, at the compressed end where |E‴| is largest
        for r in [0.8 * R_E, R_E, 2.0 * R_E - 0.1] {
            let mut g = BranchJumpGuard::new();
            judge_at(&mut g, r, 0.0);
            // 0.1 Bohr to 1e-12, so the rounding of r + 0.1 cannot put it past the reach
            let j = judge_at(&mut g, r + 0.1 * (1.0 - 1e-12), 0.0);
            assert!(j.judged && !j.flagged, "0.1 Bohr from {r}: m {:e} b {:e}", j.mismatch, j.bound);
            assert!(j.bound <= 3e-3 + 3.0 * (std::f64::consts::SQRT_2 * 1.87e-6 + 0.1 * 5e-5 / std::f64::consts::SQRT_2) + 1e-15, "b ≤ 3 mHa + noise at 0.1 Bohr");
        }
    }

    /// Velocity Verlet on a Morse dimer of masses `m` (amu) from separation `r0` at relative speed `v_rel` (Bohr per
    /// atomic time unit, approaching), dt 0.25 fs, `steps` steps along `DIR`, every step judged by the guard on the
    /// pseudo app tier (the tightest σ_E). Returns (flags, judged steps, worst m/b, the range of r visited).
    fn md_flags(d: f64, a: f64, r_e: f64, m: [f64; 2], r0: f64, v_rel: f64, steps: usize) -> (usize, usize, f64, f64, f64) {
        let morse_at = |p: &[[f64; 3]; 2]| -> (f64, Vec<[f64; 3]>) {
            let dv: [f64; 3] = std::array::from_fn(|k| p[1][k] - p[0][k]);
            let r = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            let x = (-a * (r - r_e)).exp();
            let de = 2.0 * d * a * (1.0 - x) * x;
            let u = dv.map(|c| c / r);
            (d * ((1.0 - x) * (1.0 - x) - 1.0), vec![u.map(|c| de * c), u.map(|c| -de * c)])
        };
        let m = m.map(units::amu_to_au_mass);
        let dt = units::fs_to_au_time(0.25);
        let mtot = m[0] + m[1];
        // centre of mass at rest at the origin; atom 1 moves toward atom 0 at v_rel
        let mut p = [DIR.map(|c| -m[1] / mtot * r0 * c), DIR.map(|c| m[0] / mtot * r0 * c)];
        let mut v = [DIR.map(|c| m[1] / mtot * v_rel * c), DIR.map(|c| -m[0] / mtot * v_rel * c)];
        let (mut g, mut flags, mut judged, mut worst) = (BranchJumpGuard::new(), 0, 0, 0.0f64);
        let (mut r_min, mut r_max) = (f64::INFINITY, 0.0f64);
        let (mut e, mut f) = morse_at(&p);
        for _ in 0..steps {
            let j = g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e, cell: cell(), origin: ORIGIN, spin_abs: None, tier: PSEUDO_APP_TIER });
            if j.judged {
                judged += 1;
                worst = worst.max(j.mismatch / j.bound);
            }
            flags += j.flagged as usize;
            for i in 0..2 {
                for k in 0..3 {
                    v[i][k] += 0.5 * dt * f[i][k] / m[i];
                    p[i][k] += dt * v[i][k];
                }
            }
            (e, f) = morse_at(&p);
            for i in 0..2 {
                for k in 0..3 {
                    v[i][k] += 0.5 * dt * f[i][k] / m[i];
                }
            }
            let r = (0..3).map(|k| (p[1][k] - p[0][k]).powi(2)).sum::<f64>().sqrt();
            (r_min, r_max) = (r_min.min(r), r_max.max(r));
        }
        (flags, judged, worst, r_min, r_max)
    }

    /// Review C1's case: two H from rest at 4 Bohr on an LSDA-like H₂ Morse (D 0.17 Ha, a 1.03 Bohr⁻¹, R_e 1.45 Bohr),
    /// velocity Verlet at dt 0.25 fs, through several passes of the well (bond formation from rest: the running median
    /// flagged 655 of 2999 such steps in the review's replica). Zero flags.
    #[test]
    fn h2_forming_from_rest_is_never_flagged() {
        let h = units::isotopes::isotope_mass_amu(1).expect("H");
        let (flags, judged, worst, r_min, r_max) = md_flags(0.17, 1.03, 1.45, [h, h], 4.0, 0.0, 3000);
        println!("H₂ from rest at 4 Bohr, 3000 Verlet steps of 0.25 fs: {flags} flags of {judged} judged, worst m/b {worst:.3}, r ∈ [{r_min:.3}, {r_max:.3}]");
        assert!(r_min < 1.0 && r_max > 3.9, "several passes of the well: r ∈ [{r_min}, {r_max}]");
        assert_eq!(judged, 2999);
        assert_eq!(flags, 0, "worst m/b {worst}");
    }

    /// Review C1's collisions: an H atom onto an O at relative kinetic energies E_rel = 0, 0.005, 0.01 and 0.02 Ha, on an
    /// O–H Morse fitted to the measured pseudo-surface bond (references.md "Fermi–Dirac migration": R_e 1.85 Bohr,
    /// E‴ −1.80 at R_e and −4.77 at 0.8 R_e; D 0.17 Ha gives a = (1.80/(6D))^⅓ = 1.21 Bohr⁻¹ and E‴(0.8 R_e) = −4.9),
    /// from 5 Bohr, dt 0.25 fs, through the collision and several vibrations. Zero flags.
    #[test]
    fn o_h_collision_is_never_flagged() {
        let (mo, mh) = (units::isotopes::isotope_mass_amu(8).expect("O"), units::isotopes::isotope_mass_amu(1).expect("H"));
        let mu = units::amu_to_au_mass(mo * mh / (mo + mh));
        for e_rel in [0.0, 0.005, 0.01, 0.02] {
            let v_rel = (2.0 * e_rel / mu).sqrt();
            let (flags, judged, worst, r_min, _) = md_flags(0.17, 1.21, 1.85, [mo, mh], 5.0, v_rel, 3000);
            println!("O–H collision, E_rel {e_rel} Ha: {flags} flags of {judged} judged, worst m/b {worst:.3}, r_min {r_min:.3}");
            assert!(r_min < 1.6, "E_rel {e_rel}: the collision reaches the repulsive wall ({r_min})");
            assert_eq!(flags, 0, "E_rel {e_rel}: worst m/b {worst}");
        }
    }

    /// Review M4: each evaluation brings its own tier, combined in quadrature — a step from a sample-tier solve to an
    /// app-tier one is judged on both.
    #[test]
    fn each_evaluation_brings_its_own_tier() {
        let mut g = BranchJumpGuard::new();
        let at = |g: &mut BranchJumpGuard, r: f64, tier: NoiseTier| {
            let p = [[0.0; 3], DIR.map(|c| c * r)];
            let (e, f) = morse(p[0], p[1]);
            g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e, cell: cell(), origin: ORIGIN, spin_abs: None, tier })
        };
        let (a, b) = (PSEUDO_SAMPLE_TIER, ALL_ELECTRON_APP_TIER);
        at(&mut g, 1.8, a);
        let j = at(&mut g, 1.82, b);
        let n = 3.0 * (a.sigma_e.hypot(b.sigma_e) + 0.5 * 0.02 * a.sigma_f.hypot(b.sigma_f));
        assert!((j.bound - (n + 3.0 * KAPPA_MAX * 0.02f64.powi(3))).abs() < 1e-15, "{} vs {}", j.bound, n);
        // equal tiers reduce to √2σ_E and ‖ΔR‖σ_F/√2
        let j = at(&mut g, 1.84, b);
        let n = 3.0 * (std::f64::consts::SQRT_2 * b.sigma_e + 0.02 * b.sigma_f / std::f64::consts::SQRT_2);
        assert!((j.bound - (n + 3.0 * KAPPA_MAX * 0.02f64.powi(3))).abs() < 1e-15);
    }

    /// One atom moving 0.11 Bohr is not judged; the next step within the reach is judged again (the surface is the same).
    #[test]
    fn over_reach_not_judged() {
        let mut g = BranchJumpGuard::new();
        for i in 0..12 {
            judge_at(&mut g, 1.6 + 0.02 * i as f64, 0.0);
        }
        let j = judge_at(&mut g, 1.82 + 0.11, 0.012);
        assert!(!j.judged && !j.flagged && j.branch_jump().is_none());
        // just inside the reach it is judged (on the same branch as the unjudged step: not flagged)
        let j = judge_at(&mut g, 1.82 + 0.11 + 0.099, 0.012);
        assert!(j.judged && !j.flagged);
    }

    /// 50 steps of 1e-3 Bohr (noise-dominated), then a 0.02 Bohr step carrying +12 mHa: flagged.
    #[test]
    fn relax_then_jump_flagged() {
        let mut g = BranchJumpGuard::new();
        for i in 0..=50 {
            let j = judge_at(&mut g, 1.7 + 1e-3 * i as f64, 0.0);
            assert!(!j.flagged);
        }
        let j = judge_at(&mut g, 1.75 + 0.02, 0.012);
        assert!(j.judged && j.flagged, "m {:e} b {:e}", j.mismatch, j.bound);
    }

    /// ±12 mHa on every other 0.02 Bohr step for 40 steps: every jump is flagged, every step between is not.
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
        }
    }

    /// The cell's n, its h or the origin changes: not judged; the next step on the new box is judged against it.
    #[test]
    fn cell_or_origin_change_restarts() {
        type Edit = fn(&mut sim::Cell, &mut [f64; 3]);
        let edits: [(&str, Edit); 3] = [("n", |c, _| c.n += 2), ("h", |c, _| c.h = 0.25), ("origin", |_, o| o[1] += 0.15)];
        for (what, edit) in edits {
            let mut g = BranchJumpGuard::new();
            for i in 0..12 {
                judge_at(&mut g, 1.6 + 0.02 * i as f64, 0.0);
            }
            let p = [[0.0; 3], DIR.map(|c| c * 1.84)];
            let (e, f) = morse(p[0], p[1]);
            let (mut c, mut o) = (cell(), ORIGIN);
            edit(&mut c, &mut o);
            // a box change moves the discrete surface: here by a "jump" that must not be flagged
            let j = g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f, energy: e + 0.012, cell: c.clone(), origin: o, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
            assert!(!j.judged && !j.flagged, "{what}");
            // the next step on the new box is judged against the new box's evaluation
            let p = [[0.0; 3], DIR.map(|c| c * 1.86)];
            let (e2, f2) = morse(p[0], p[1]);
            let j = g.judge(&Evaluation { z: &Z2, positions: &p, forces: &f2, energy: e2 + 0.012, cell: c, origin: o, spin_abs: None, tier: PSEUDO_SAMPLE_TIER });
            assert!(j.judged && !j.flagged, "{what}: m {:e}", j.mismatch);
        }
    }

    /// Review Focus 2: an atom added, one removed, two species swapped — not judged, no panic.
    #[test]
    fn atom_set_change_restarts() {
        let warm = |g: &mut BranchJumpGuard| {
            for i in 0..12 {
                judge_at(g, 1.6 + 0.02 * i as f64, 0.0);
            }
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

    /// `reset` forgets the previous evaluation.
    #[test]
    fn reset_restarts() {
        let mut g = BranchJumpGuard::new();
        for i in 0..12 {
            judge_at(&mut g, 1.6 + 0.02 * i as f64, 0.0);
        }
        g.reset();
        let j = judge_at(&mut g, 1.84, 0.012);
        assert!(!j.judged);
    }

    /// Plan Task 8 step 1 [kT-5] and fix round 1 (review I1): the per-solve path scatter behind the tiers
    /// (references.md "The branch_jump guard"). The providers still run the old occupation rule, so `GpuSolver` is
    /// driven directly with explicit `FermiDirac { kt: ELECTRON_KT }` on the scene `ks::problem` builds, σ = h, in the
    /// presets' 14 Bohr box. One cold solve at R₀, then 5 repeats; each repeat moves every atom by δ = 0.02 Bohr in a
    /// seeded random direction and
    /// - extrapolation off (Step 1): a warm solve at R₀ + δ, then a warm solve back at R₀;
    /// - extrapolation on (fix round 1, the MD case): the history dropped, then warm solves at R₀ − 2δ, R₀ − δ and R₀,
    ///   the last two from the extrapolated start 2ρ_t − ρ_{t−1} (equal steps, as in MD).
    ///
    /// Per repeat |E_warm − E_cold| and the largest per-component |F_warm − F_cold|; σ_E and σ_F are the largest over
    /// the 5 (each a difference of two solves: conservative by up to √2 against a per-solve σ, as the sample tier's
    /// 1.87e-6 is). `BRANCH_JUMP_TIER_CASE` is a comma-separated list, run in order in one process:
    /// - `pseudo_h2o` (`Accepted`, `effective_gpu_tol(1e-5)`), `oh` (linear OH along x, R = 1.85, `Accepted`, asked 1e-6:
    ///   its partial π level stops it at `GPU_PARTIAL_TOL`), `ae_h2o` (`IonicPolicy::None`, 1e-5) and `ae_h2o_sample`
    ///   (`None`, 1e-6): 95³ × 0.15, extrapolation off (Step 1);
    /// - `h2o_x`, `sih4_x` (the largest preset, 5 atoms; `Accepted`, 1e-5) at Accurate 95³ × 0.15 and `sih4_bal_x` at
    ///   Balanced 57³ × 0.25, extrapolation on (fix round 1).
    ///
    /// A solve that takes over 5 min, or a run over 14 min (the gentle-GPU batch), is abandoned and reported.
    #[test]
    #[ignore = "measurement (GPU, minutes): BRANCH_JUMP_TIER_CASE=<case,…> cargo test -p providers --release tier_scatter_measurement -- --ignored --nocapture"]
    fn tier_scatter_measurement() {
        use rand::{Rng, SeedableRng};
        let Some(ctx) = gpu::GpuContext::new_headless() else {
            eprintln!("tier_scatter_measurement: no GPU adapter — skipped");
            return;
        };
        crate::pseudo::set_dimer_backend(crate::pseudo::DimerBackend::Gpu(ctx.clone()));
        let cases = std::env::var("BRANCH_JUMP_TIER_CASE").unwrap_or_else(|_| "pseudo_h2o".into());
        // the gentle-GPU batch limit: the whole run stops at 14 min, whatever is left unmeasured
        let start = std::time::Instant::now();
        for case in cases.split(',').map(str::trim) {
            let h2o = vec![sim::Nucleus::new(8, [0.0, 0.0, 0.0]), sim::Nucleus::new(1, [1.43, 1.11, 0.0]), sim::Nucleus::new(1, [-1.43, 1.11, 0.0])];
            let oh = vec![sim::Nucleus::new(8, [-0.6, 0.0, 0.0]), sim::Nucleus::new(1, [1.25, 0.0, 0.0])];
            let a = 1.6147;
            let sih4 = vec![sim::Nucleus::new(14, [0.0; 3]), sim::Nucleus::new(1, [a, a, a]), sim::Nucleus::new(1, [a, -a, -a]), sim::Nucleus::new(1, [-a, a, -a]), sim::Nucleus::new(1, [-a, -a, a])];
            let accurate = sim::Cell { n: 95, h: 0.15 };
            let balanced = sim::Cell { n: 57, h: 0.25 };
            use sim::IonicPolicy::{Accepted, None as AllElectron};
            let (nuclei, ionic, tol, cell, extrapolate) = match case {
                "pseudo_h2o" => (h2o, Accepted, 1e-5, accurate, false),
                "oh" => (oh, Accepted, SAMPLE_TOL, accurate, false),
                "ae_h2o" => (h2o, AllElectron, 1e-5, accurate, false),
                "ae_h2o_sample" => (h2o, AllElectron, SAMPLE_TOL, accurate, false),
                "h2o_x" => (h2o, Accepted, 1e-5, accurate, true),
                "sih4_x" => (sih4, Accepted, 1e-5, accurate, true),
                "sih4_bal_x" => (sih4, Accepted, 1e-5, balanced, true),
                other => panic!("unknown case {other}"),
            };
            let scene = sim::Scene { nuclei, cell, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic, train_mask: None };
            let mut notes = Vec::new();
            let (grid, system, mut settings) = match crate::ks::problem(&scene, tol, &mut notes) {
                Ok(v) => v,
                Err(e) => {
                    println!("case {case}: not posed: {e}");
                    continue;
                }
            };
            settings.occupations = qm::Occupations::FermiDirac { kt: qm::ELECTRON_KT };
            if system.ionic.is_some() {
                settings.tol = crate::pseudo::effective_gpu_tol(tol, (system.n_up + system.n_dn) as i64);
            }
            println!("case {case}: {}³ × {}, {:?}, N↑ {} N↓ {}, tol {:.1e} (asked {tol:.0e}), FD kT {}, extrapolation {extrapolate}, notes {notes:?}",
                     grid.n[0], grid.h, scene.ionic, system.n_up, system.n_dn, settings.tol, qm::ELECTRON_KT);
            let mut solver = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
            solver.set_density_extrapolation(false);
            let solve = |solver: &mut qm_gpu::GpuSolver, nuclei: &[qm::Nucleus], warm: bool, what: &str| -> Option<(f64, Vec<[f64; 3]>)> {
                let sys = qm::System { nuclei: nuclei.to_vec(), n_up: system.n_up, n_dn: system.n_dn, ionic: system.ionic.clone() };
                let t = std::time::Instant::now();
                let r = solver.solve_with(&sys, &settings, warm, &mut || t.elapsed().as_secs() >= 300 || start.elapsed().as_secs() >= 840).map(|r| r.map(|(_, st)| st));
                let st = match r {
                    Ok(Some(v)) => v,
                    Ok(None) => {
                        println!("  {what}: abandoned after {:.0} s (5 min per solve, 14 min per run) — not measured", t.elapsed().as_secs_f64());
                        return None;
                    }
                    Err(e) => {
                        println!("  {what}: {e} — not measured");
                        return None;
                    }
                };
                let state = solver.state().expect("a converged solve exposes its state");
                let f = solver.forces(state, &sys.nuclei, settings.sigma).expect("forces");
                println!("  {what}: {} it, {:.1} s, tol_used {:.1e}, F {:.9}, TS {:.2e}, partial {:?}, extrapolated {}, eig res {:.1e}",
                         st.iterations, t.elapsed().as_secs_f64(), st.tol_used, st.energy.total, st.energy.ts,
                         st.partial.iter().map(|p| (p.spin, p.index, (p.f * 1e3).round() / 1e3)).collect::<Vec<_>>(), solver.profile().extrapolated, st.max_eigen_residual);
                Some((st.energy.total, f))
            };
            let r0: Vec<qm::Nucleus> = system.nuclei.clone();
            let Some((e_cold, f_cold)) = solve(&mut solver, &r0, false, "cold R₀") else { continue };
            let mut rng = rand::rngs::StdRng::seed_from_u64(8);
            let (mut sig_e, mut sig_f) = (0.0f64, 0.0f64);
            let mut rows = Vec::new();
            for k in 0..5 {
                let dirs: Vec<[f64; 3]> = r0.iter().map(|_| {
                    let (u, phi): (f64, f64) = (rng.gen_range(-1.0..1.0), rng.gen_range(0.0..std::f64::consts::TAU));
                    let s = (1.0 - u * u).sqrt();
                    [s * phi.cos(), s * phi.sin(), u].map(|c| 0.02 * c)
                }).collect();
                let at = |c: f64| -> Vec<qm::Nucleus> { r0.iter().zip(&dirs).map(|(n, d)| qm::Nucleus { pos: std::array::from_fn(|a| n.pos[a] + c * d[a]), z: n.z }).collect() };
                let warm = if extrapolate {
                    // a fresh history, then two equal steps into R₀: the last two starts are extrapolated
                    solver.set_density_extrapolation(false);
                    solver.set_density_extrapolation(true);
                    if solve(&mut solver, &at(-2.0), true, &format!("repeat {k}: warm R₀ − 2δ")).is_none() { break }
                    if solve(&mut solver, &at(-1.0), true, &format!("repeat {k}: warm R₀ − δ")).is_none() { break }
                    solve(&mut solver, &r0, true, &format!("repeat {k}: warm R₀"))
                } else {
                    if solve(&mut solver, &at(1.0), true, &format!("repeat {k}: warm neighbour")).is_none() { break }
                    solve(&mut solver, &r0, true, &format!("repeat {k}: warm R₀"))
                };
                let Some((e, f)) = warm else { break };
                let de = (e - e_cold).abs();
                let df = f.iter().zip(&f_cold).flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs())).fold(0.0f64, f64::max);
                (sig_e, sig_f) = (sig_e.max(de), sig_f.max(df));
                rows.push((de, df));
                println!("  repeat {k}: |E_warm − E_cold| {de:.2e}, max |ΔF| {df:.2e}");
            }
            solver.set_density_extrapolation(false);
            if rows.len() < 5 {
                println!("case {case}: only {} of 5 repeats measured", rows.len());
                if rows.is_empty() {
                    continue;
                }
            }
            println!("case {case}: |E_warm − E_cold| {:?}", rows.iter().map(|r| format!("{:.2e}", r.0)).collect::<Vec<_>>());
            println!("case {case}: max |ΔF| per repeat {:?}", rows.iter().map(|r| format!("{:.2e}", r.1)).collect::<Vec<_>>());
            let rms = |v: Vec<f64>| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
            println!("case {case}: |E_cold| {:.6} Ha, σ_E (max) {sig_e:.3e} (rms {:.3e}; σ_E/|E| {:.2e}), σ_F (max) {sig_f:.3e} (rms of the per-repeat max {:.3e})",
                     e_cold.abs(), rms(rows.iter().map(|r| r.0).collect()), sig_e / e_cold.abs(), rms(rows.iter().map(|r| r.1).collect()));
        }
    }
}
