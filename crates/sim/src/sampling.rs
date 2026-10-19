//! The sampling pass's data types (spec §4 (b)): the [`SamplingPlan`] a pass is made from, its
//! [`SamplingProgress`], the reference-scan rule [`allowed_multiplicities`] and the reference
//! cache's alias. The pass itself — `Sampler`, which drives a teacher over the plan's geometries
//! and feeds a surrogate — lives with the providers (`providers::sampling`, whose module doc
//! describes what a pass does); the driver reaches it only through [`Trainable`](crate::Trainable).
//! The plan's solve count, trust floor and scaling need the surrogate's basis and are
//! `providers::sampling::PlanSolves`.

pub use crate::references::AtomReferences;

/// What a sampling pass generates (module doc). `Default` is the spec's plan.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingPlan {
    /// Random displacements of the scene (60).
    pub perturbations: usize,
    /// Width of each coordinate's Gaussian displacement, Bohr (0.15).
    pub perturbation_sigma: f64,
    /// Teacher evaluations of the hot burst (200).
    pub hot_burst_steps: usize,
    /// The burst's temperature, K (2000).
    pub hot_burst_t: f64,
    /// Points per pair scan (12).
    pub scan_points_per_pair: usize,
    /// Scan range as fractions of the pair's CURRENT separation (0.7, 1.8): from the scene,
    /// never from a radius table (rule 1).
    pub scan_range: (f64, f64),
    pub seed: u64,
}

impl Default for SamplingPlan {
    fn default() -> SamplingPlan {
        SamplingPlan {
            perturbations: 60,
            perturbation_sigma: 0.15,
            hot_burst_steps: 200,
            hot_burst_t: 2000.0,
            scan_points_per_pair: 12,
            scan_range: (0.7, 1.8),
            seed: 0,
        }
    }
}

/// Why a pass over an empty scene is refused (`providers::sampling::Sampler::new`, and the
/// driver's `StartSampling`): the app shows it as is.
pub const EMPTY_SCENE: &str = "the scene is empty: place atoms to sample";

/// Most jobs one pass may plan (each is a teacher solve; the jobs are built up front).
pub const MAX_JOBS: usize = 100_000;

/// The multiplicities 2S + 1 an isolated neutral atom of `electrons` electrons is solved at
/// (ruling S9): the parity of the electron count, from the lowest up to 5, while 2S ≤ the
/// electron count. The same rule for every Z — nothing per element.
pub fn allowed_multiplicities(electrons: u32) -> Vec<u32> {
    (1 + electrons % 2..=5).step_by(2).filter(|&m| m - 1 <= electrons).collect()
}

impl SamplingPlan {
    /// Finite non-negative σ and T, 0 < scan_range.0 ≤ scan_range.1, both finite, and no count
    /// above [`MAX_JOBS`].
    pub fn validate(&self) -> Result<(), String> {
        for (what, n) in [("perturbations", self.perturbations), ("hot burst steps", self.hot_burst_steps), ("scan points per pair", self.scan_points_per_pair)] {
            if n > MAX_JOBS {
                return Err(format!("{n} {what}: at most {MAX_JOBS} per pass"));
            }
        }
        if !(self.perturbation_sigma.is_finite() && self.perturbation_sigma >= 0.0) {
            return Err(format!("perturbation σ {} Bohr: must be finite and ≥ 0", self.perturbation_sigma));
        }
        if !(self.hot_burst_t.is_finite() && self.hot_burst_t >= 0.0) {
            return Err(format!("hot burst T {} K: must be finite and ≥ 0", self.hot_burst_t));
        }
        let (lo, hi) = self.scan_range;
        if !(lo.is_finite() && hi.is_finite() && lo > 0.0 && lo <= hi) {
            return Err(format!("scan range ({lo}, {hi}): need 0 < lo ≤ hi, both finite"));
        }
        Ok(())
    }
}

/// Where a pass stands. `done` counts the jobs accounted for — solved, skipped by the cell
/// guard, failed, or a reference taken from the cache — out of `total` (references + perturbations
/// + scan points + burst evaluations, fixed when the pass starts).
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingProgress {
    pub done: usize,
    pub total: usize,
    /// "references", "perturbations", "scans", "hot burst", "teaching" (a teach-around pass),
    /// "done" or "cancelled".
    pub stage: &'static str,
    /// A teach-around pass: the store index of the blocker whose region is being solved
    /// ("teaching around #345"); `None` otherwise.
    pub target: Option<usize>,
    /// The last teacher energy (Hartree); NaN before the first solve.
    pub last_energy: f64,
    pub cancelled: bool,
    /// Geometries the cell guard kept from the teacher (module doc).
    pub skipped: usize,
    /// Teacher evaluations that failed (or whose sample the store refused).
    pub failed: usize,
    /// No job left (complete, cancelled, or ended by a failed reference).
    pub finished: bool,
}

/// One nucleus of the pass's ghost: Z and position (Bohr).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GhostAtom {
    pub z: u8,
    pub pos: [f64; 3],
}

/// What the job being solved is (the ghost's label; display only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GhostJob {
    /// The isolated neutral atom of species `z`, at the cell centre, as the pass solves it.
    Reference { z: u8 },
    /// Random nudge number `index` (1-based) of `of`.
    Perturbation { index: usize, of: usize },
    /// Point `k` (1-based) of `of` of the scan of the pair (`i`, `j`) of the ghost's atoms, at
    /// `ratio` = r/r₀ and r = `r` Bohr.
    Scan { i: usize, j: usize, k: usize, of: usize, ratio: f64, r: f64 },
    /// The shell check of the scan of the pair (`i`, `j`): its first point beyond 5/6 r_cut, at
    /// r = `r` Bohr — solved as a check of the model's cutoff, never fitted (final review M9).
    ShellCheck { i: usize, j: usize, r: f64 },
    /// Evaluation `step` (1-based) of `of` of the hot burst.
    HotBurst { step: usize, of: usize },
    /// Solve `index` (1-based) of `of` of a teach-around pass, near the blocking sample `blocker`
    /// (a store index); `holdout` = a held-out check (solved, never fitted).
    Teach { blocker: usize, index: usize, of: usize, holdout: bool },
}

/// How the last finished job of the pass ended, in the solve log's words
/// (`kept` / `failed` or the SCF end `stalled`, `cap`, `not finite` / `skipped`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GhostOutcome {
    /// Solved and kept as a sample (a reference: solved).
    Kept,
    /// The solve failed (with how its SCF ended, when the error says) or the store refused it.
    Failed(Option<crate::activity::ScfEnd>),
    /// Not solved (the cell guard, a scan point beyond a failed one) or solved and not kept
    /// (overlapping cores, a changed shell pattern).
    Skipped,
}

impl GhostOutcome {
    /// The solve log's word for it.
    pub fn word(self) -> &'static str {
        match self {
            GhostOutcome::Kept => "kept",
            GhostOutcome::Failed(Some(end)) if end != crate::activity::ScfEnd::Converged => end.word(),
            GhostOutcome::Failed(_) => "failed",
            GhostOutcome::Skipped => "skipped",
        }
    }
}

/// The last job the pass finished: a counter (`serial` grows with every job accounted for, so a
/// viewer that sees a new one starts a flash), how it ended, and the geometry that was solved.
#[derive(Clone, Debug, PartialEq)]
pub struct GhostFinished {
    pub serial: u64,
    pub outcome: GhostOutcome,
    pub atoms: Vec<GhostAtom>,
}

/// What a running sampling pass is solving, for the viewer's "show what training is solving"
/// (display only: a few dozen numbers, no density, nothing read back from the GPU). Built by the
/// provider (`Trainable::pass_ghost`) from the job at the head of the pass's queue — the one the
/// worker is solving, or about to, while this snapshot is on screen.
#[derive(Clone, Debug, PartialEq)]
pub struct PassGhost {
    /// "references", "perturbations", "scans", "hot burst" or "teaching" (`SamplingProgress::stage`).
    pub stage: &'static str,
    pub job: GhostJob,
    /// The geometry being solved.
    pub atoms: Vec<GhostAtom>,
    /// The pass's base positions (the scene as the pass started, same order as `atoms`); empty
    /// for a reference, whose atom is the only one.
    pub base: Vec<[f64; 3]>,
    /// A hot burst's last few solved geometries' positions, oldest first (empty otherwise).
    pub trail: Vec<Vec<[f64; 3]>>,
    /// The last finished job, `None` before the first.
    pub finished: Option<GhostFinished>,
}

/// The isolated-atom reference cache (module doc of `providers::sampling`, "References"): an
/// alias over [`AtomReferences`] until Task 3 makes the driver its owner.
pub type ReferenceCache = AtomReferences;

#[cfg(test)]
mod tests {
    use super::*;

    /// `qm::radial` (M7 Task 1) scans the same multiplicities as ruling S9: its rule reads the cap
    /// from `qm::radial::LEDGER.multiplicity_max`, this one is written out; they must agree.
    #[test]
    fn radial_multiplicities_mirror_ruling_s9() {
        for n in 0u32..=40 { assert_eq!(qm::radial::allowed_multiplicities(n), allowed_multiplicities(n), "N = {n}"); }
    }
}
