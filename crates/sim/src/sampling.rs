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
    /// "references", "perturbations", "scans", "hot burst", "done" or "cancelled".
    pub stage: &'static str,
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

/// The isolated-atom reference cache (module doc of `providers::sampling`, "References"): an
/// alias over [`AtomReferences`] until Task 3 makes the driver its owner.
pub type ReferenceCache = AtomReferences;
