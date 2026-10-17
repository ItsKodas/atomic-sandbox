//! The sampling pass (spec §4 (b)): drives the DFT teacher over geometries derived from the
//! current scene and feeds every answer to a [`SurrogateKs`] as a training sample. Atomic units
//! throughout (Bohr, Hartree); the plan's temperature is in kelvin and the burst's τ and dt in
//! fs only as constants at this edge.
//!
//! # What a pass does, in order
//!
//! 1. **References** (spec §3, "Reference energies"; ruling S9): for each species present, the
//!    isolated neutral atom — one nucleus at the centre of the scene's own grid (the origin:
//!    `qm::Grid::cube` is centred there), the scene's cell and σ, charge 0 — is solved by the
//!    teacher at every allowed multiplicity ([`allowed_multiplicities`]: 2S + 1 of the parity of
//!    the Z electrons, from the lowest up to 5, stopping where 2S exceeds the electron count —
//!    H {2}, He {1, 3}, Li {2, 4}, C/O {1, 3, 5}), each solve cold (`reset()` before it: another
//!    spin state's orbitals are a poor start and could steer the SCF into the wrong state), and
//!    the **lowest energy** is the reference. Hund's rule is not put in; it is meant to emerge:
//!    the same scan runs for every Z (rule 1), and the note records the outcome ("reference O
//!    (Z = 8): multiplicity 3 (E = …) chosen over 1 (…), 5 (…)"). Measured (rule 7): at the
//!    project's h = 0.25, σ = h the teacher puts the O singlet lowest (33³: −36.565 vs triplet
//!    −36.338 Ha) and its cold triplet SCF does not converge at 57³ — an unresolved, smeared
//!    open-shell atom, a solver question (test `gpu_oxygen_reference_multiplicity`, ignored with
//!    the numbers); the scan then takes the singlet, consistently with the teacher's surface. A multiplicity whose solve fails is
//!    noted and left out; only if none succeeds does the pass end. The energy is set with
//!    `SurrogateKs::set_reference_energy` — never a table — and cached with its multiplicity
//!    in a [`ReferenceCache`] keyed by the atom's settings (grid, h, σ, teacher, tol) and Z, so a
//!    later pass at the same settings skips the scan. The sidecar records the rule
//!    ([`surrogate::sidecar::REFERENCE_RULE`]); references from a file without it are not
//!    cached and are re-solved by the next pass (`Dynamics::LoadModel`). References are not
//!    store samples: an isolated atom's charge and multiplicity differ from a charged or
//!    spin-fixed scene's, so it would carry another provenance. **Charged scenes keep neutral
//!    references** (documented approximation): the model's dissociation limit is then Σ of the
//!    neutral atoms, not the physical one with the charge on a fragment; inside r_cut the fit
//!    absorbs the offset, beyond it the model cannot represent it.
//! 2. **Perturbations**: `perturbations` copies of the scene with every coordinate displaced by
//!    an independent Gaussian of width `perturbation_sigma` (0.15 Bohr), seeded by `seed`.
//! 3. **Pair scans**: for every pair closer than the model's cutoff (the surrogate's basis
//!    r_cut when it has one, else the default `Basis::new(..).r_cut` = 6 Bohr) — coincident pairs have no axis and are skipped with a note — the separation is
//!    scanned over `scan_points_per_pair` equally spaced values from `scan_range.0` to
//!    `scan_range.1` times the pair's CURRENT separation r₀ (0.7 … 1.8 r₀): the ranges come from
//!    the scene, never from an element radius table (rule 1). The lighter nucleus of the pair
//!    (isotope mass; the higher index on a tie) moves along the pair axis, every other nucleus
//!    stays where it is.
//! 4. **Hot burst**: `hot_burst_steps` teacher evaluations along a velocity-Verlet trajectory
//!    (dt [`HOT_BURST_DT_FS`] = 0.25 fs) from the scene's positions with Maxwell–Boltzmann
//!    velocities at `hot_burst_t` (2000 K), CSVR-thermostatted at that temperature with
//!    τ [`HOT_BURST_TAU_FS`] = 5 fs: the first evaluation is the start geometry (the burst needs
//!    its forces), each later one a Verlet step. Every evaluation is a `HotBurst` sample.
//!
//! **Cell guard**: a generated geometry with a nucleus closer than [`CELL_MARGIN_H`]·h to a
//! cell face (the providers' own R12 margin) is never sent to the teacher; it is counted in
//! `SamplingProgress::skipped`. A burst that leaves the cell ends there (its remaining steps are
//! skipped). A teacher error on a sample geometry is counted in `failed` with a note and the
//! pass goes on; a failed reference solve ends the pass with the error (no fit is possible
//! without it).
//!
//! **Box-affected samples** (final review I1): every sample goes to the surrogate with its
//! teacher's face density (`EvalStats::face_density`, `SurrogateKs::add_sample_with_face`), so
//! the box rule (`crate::surrogate::box_affected_face_density`) applies to the pass as it does to
//! live answers: on a small box a scan to 1.8 r₀ or a 2000 K burst reaches geometries whose
//! labels carry the box's confinement; they are kept (the model is still being built), never
//! quarantine candidates, and leave the fit once the trust floor is met without them. The pass
//! notes how many there were.
//!
//! **Activity events** (activity spec A3): every teacher solve of the pass leaves one
//! `TeachEvent` (origin Pass) on the surrogate's ledger (`SurrogateKs::push_event`) — a reference
//! multiplicity (`Reference` or `Failed`), a sample (`Kept`, `KeptBoxAffected` or `StoreRefused`;
//! a hot-burst one with the solved geometry's temperature before the thermostat), a failed
//! sample solve (`Failed`) — and every geometry the cell guard skips leaves a `Cell` /
//! `NotSolved` row (the burst's skipped tail as one). A cache hit and a cancelled solve leave
//! none. The events record what the pass did; nothing in the pass reads them.
//!
//! The teacher's warm state is managed the way M4 learned (Task 4 report): `reset` whenever the
//! nucleus count changes (reference ↔ scene), `discontinuity` before every geometry unrelated to
//! the previous one (each perturbation, the first point of each pair scan, the burst start) —
//! warm orbitals kept, no density extrapolation across the jump — and nothing inside a scan or
//! the burst, whose consecutive points are equally spaced along one path. The caller should
//! call `discontinuity` once more after the pass before the live run continues (`Dynamics`
//! does).
//!
//! # The teacher
//!
//! Samples and reference-cache keys carry the provenance of the surrogate's OWN teacher
//! (`SurrogateKs::provenance_for`: its name and recorded tolerance). [`Sampler::step`] with a
//! separate teacher handle therefore refuses one whose `name()` differs from the surrogate's
//! teacher (its tolerance is not visible through the boundary, so equal names are the check the
//! boundary allows: the caller vouches for the tolerance, as `SurrogateKs::with_teacher_tol`
//! already does); [`Sampler::step_own_teacher`] has no second handle at all and is what
//! `Dynamics` uses.
//!
//! # Resumable
//!
//! [`Sampler`] is a state machine: [`Sampler::step`] performs at most one teacher solve and
//! returns, so the dynamics worker can advance it one solve per tick slice and stay responsive.
//! A solve the teacher abandons (`ProviderError::Cancelled`) leaves the job queued: the next
//! `step` retries it. [`run_sampling`] is the convenience loop over it. Neither refits — the
//! caller does (a partial pass refits like a full one).

use sim::activity::{PassJob, TeachEvent, TeachOrigin, TeachOutcome, TeachRule};
use sim::integrator::Integrator;
use sim::references::{AtomScan, ReferenceKey};
use sim::sampling::{allowed_multiplicities, SamplingPlan, SamplingProgress, EMPTY_SCENE, MAX_JOBS};
use sim::thermostat::{standard_normal, Csvr};
use sim::{ForceProvider, ForceResult, ProviderError, ReferenceCache, Scene, CELL_MARGIN_H};
use crate::surrogate::{box_affected_face_density, SurrogateKs, MIN_RESIDUAL_DOF};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::time::Instant;
use surrogate::{Basis, Provenance, Sample, SampleTag};
use units::isotopes::symbol;

/// The hot burst's CSVR coupling time (fs).
pub const HOT_BURST_TAU_FS: f64 = 5.0;
/// The hot burst's time step (fs).
pub const HOT_BURST_DT_FS: f64 = 0.25;

/// The plan's solve count, trust floor and scaling (they need the surrogate's basis and the
/// `Sampler`, so they live here; `use providers::PlanSolves` brings `plan.solves(scene)` and
/// friends into scope).
pub trait PlanSolves: Sized {
    /// Teacher solves a pass with this plan over `scene` makes (references included, each
    /// multiplicity of the scan counted; the pass's `SamplingProgress::total`). Err for an
    /// invalid plan.
    fn solves(&self, scene: &Scene) -> Result<usize, String>;

    /// The fewest non-reference samples a pass must add for the first fit to be able to earn
    /// trust on `scene`'s species: Q′ (one geometry per independent model column) plus the
    /// trust rule's 30 residual energy degrees of freedom (`crate::surrogate::MIN_RESIDUAL_DOF`;
    /// module doc there, "Trust rule": n_samples ≥ Q′ and n − γ_E ≥ 30 with γ_E ≤ Q′).
    fn min_samples_for_trust(scene: &Scene) -> usize;

    /// This plan with its perturbations, scan points per pair and hot-burst steps scaled down by
    /// one common factor so that a pass makes at most `max_solves` teacher solves — but never
    /// below [`min_samples_for_trust`](Self::min_samples_for_trust) non-reference solves (a
    /// shorter pass could not earn trust: then the plan that just reaches it). Unchanged when it
    /// already fits. Used to keep the viewer's automatic first pass near its time target (Task
    /// 9); σ, T, the scan range and the seed are kept.
    fn scaled_to(&self, scene: &Scene, max_solves: usize) -> SamplingPlan;
}

impl PlanSolves for SamplingPlan {
    fn solves(&self, scene: &Scene) -> Result<usize, String> {
        Sampler::new(scene, self).map(|s| s.progress().total)
    }

    fn min_samples_for_trust(scene: &Scene) -> usize {
        let mut species: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        species.sort_unstable();
        species.dedup();
        Basis::new(&species).column_classes().len() + MIN_RESIDUAL_DOF as usize
    }

    fn scaled_to(&self, scene: &Scene, max_solves: usize) -> SamplingPlan {
        let Ok(full) = self.solves(scene) else { return self.clone() };
        if full <= max_solves {
            return self.clone();
        }
        let at = |f: f64| SamplingPlan {
            perturbations: (self.perturbations as f64 * f).ceil() as usize,
            scan_points_per_pair: (self.scan_points_per_pair as f64 * f).ceil() as usize,
            hot_burst_steps: (self.hot_burst_steps as f64 * f).ceil() as usize,
            ..self.clone()
        };
        let refs = at(0.0).solves(scene).unwrap_or(0);
        let floor = Self::min_samples_for_trust(scene);
        let ok = |p: &SamplingPlan| p.solves(scene).unwrap_or(usize::MAX);
        // the largest factor that fits, by bisection (the count is monotone in f)
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if ok(&at(mid)) <= max_solves { lo = mid } else { hi = mid }
        }
        let fit = at(lo);
        if ok(&fit).saturating_sub(refs) >= floor {
            return fit;
        }
        // too short to earn trust: the smallest factor that reaches the floor
        let (mut lo, mut hi) = (lo, 1.0f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if ok(&at(mid)).saturating_sub(refs) >= floor { hi = mid } else { lo = mid }
        }
        at(hi)
    }
}

/// The cache key of an isolated atom computed at `scene_provenance`'s settings, species `z`:
/// the same grid, σ, tolerance and teacher, neutral, and the lowest-energy multiplicity
/// (ruling S9; `ReferenceCache::atom_provenance` before M6).
pub fn atom_key(scene_provenance: &Provenance, z: u8) -> ReferenceKey {
    let p = scene_provenance;
    ReferenceKey { grid_n: p.grid_n, h: p.h, sigma: p.sigma, tol: p.tol, teacher: p.teacher.clone(), z }
}

/// One planned pair scan (for the progress readout and the tests).
#[derive(Clone, Debug, PartialEq)]
pub struct PairScan {
    pub i: usize,
    pub j: usize,
    /// The nucleus that moves (the lighter of the two; `j` on a tie).
    pub moved: usize,
    /// The pair's separation in the scene, Bohr.
    pub r0: f64,
    /// The scanned separations, Bohr.
    pub r: Vec<f64>,
}

enum Job {
    /// The multiplicity scan of one species (ruling S9; `sim::references::AtomScan`, one solve
    /// per step, resumable): the energies and the shell-binned density of the winner.
    Reference(AtomScan),
    Perturbation(Scene),
    /// A scan point: the geometry, its pair index and point index.
    Scan(Scene, usize, usize),
    /// A geometry the cell guard rejected, with the stage it belonged to (and its pair and
    /// point index for a scan point: [`Sampler::rescale`] thins scans by index).
    Skip(&'static str, Option<(usize, usize)>),
}

/// The hot burst's trajectory state.
struct Burst {
    /// Evaluations still to do.
    left: usize,
    scene: Scene,
    /// Forces at `scene`'s positions (`None` before the start geometry was evaluated).
    forces: Option<Vec<[f64; 3]>>,
    csvr: Csvr,
    dt: f64,
}

/// The resumable pass (module doc).
pub struct Sampler {
    scene: Scene,
    /// The plan's counts as they stand: the plan the pass was built with until a
    /// [`rescale`](Self::rescale) lowers them.
    plan: SamplingPlan,
    /// Reference solves the pass plans (every multiplicity of every species' scan).
    refs_total: usize,
    queue: VecDeque<Job>,
    scans: Vec<PairScan>,
    /// Points each pair scan plans (its `r.len()` until a rescale thins it).
    scan_planned: Vec<usize>,
    burst: Burst,
    refs: ReferenceCache,
    progress: SamplingProgress,
    notes: Vec<String>,
    added: usize,
    /// Samples added whose teacher answer was box-affected (module doc, "Box-affected samples").
    boxed: usize,
    /// Nucleus count of the last scene sent to the teacher.
    last_n: Option<usize>,
    /// (pair, point) of the last scan point sent to the teacher.
    last_scan: Option<(usize, usize)>,
    /// The teacher solved something else since our last solve (`teacher_used_elsewhere`).
    interrupted: bool,
}

/// Every nucleus at least [`CELL_MARGIN_H`]·h inside every face, finite (the providers' R12
/// test, `provider::check_inside_cell`).
fn inside_cell(scene: &Scene) -> bool {
    let max = scene.cell.max_coord();
    scene.nuclei.iter().all(|n| n.pos.iter().all(|x| x.is_finite() && x.abs() <= max))
}

/// `n` of the sorted indices `q`, evenly spaced over them with both ends kept (the middle one
/// for `n` = 1; all of them for `n` ≥ `q.len()`).
fn spread(q: &[usize], n: usize) -> Vec<usize> {
    let len = q.len();
    match n {
        0 => Vec::new(),
        _ if n >= len => q.to_vec(),
        1 => vec![q[(len - 1) / 2]],
        _ => (0..n).map(|i| q[(i * (len - 1) + (n - 1) / 2) / (n - 1)]).collect(),
    }
}

impl Sampler {
    /// Plans a pass over `scene` with the default model cutoff (velocities and the external
    /// field are not used: the pass samples the potential-energy surface). Err for an invalid
    /// plan, and for an empty scene (no species: nothing to sample — [`EMPTY_SCENE`]).
    pub fn new(scene: &Scene, plan: &SamplingPlan) -> Result<Sampler, String> {
        Self::with_cutoff(scene, plan, Basis::new(&[]).r_cut)
    }

    /// [`new`](Self::new) with the pair-scan cutoff `r_cut` (Bohr): the surrogate's own basis
    /// cutoff when it has one (`Dynamics` and [`run_sampling`] pass it).
    pub fn with_cutoff(scene: &Scene, plan: &SamplingPlan, r_cut: f64) -> Result<Sampler, String> {
        plan.validate()?;
        if scene.nuclei.is_empty() {
            return Err(EMPTY_SCENE.into());
        }
        let mut base = scene.clone();
        base.external_field = [0.0; 3];
        for n in &mut base.nuclei {
            n.vel = [0.0; 3];
        }
        let mut queue = VecDeque::new();
        let mut notes = Vec::new();

        // 1. references, one per species present
        let mut species: Vec<u8> = base.nuclei.iter().map(|n| n.z).collect();
        species.sort_unstable();
        species.dedup();
        for &z in &species {
            let (a, mass) = base.nuclei.iter().find(|n| n.z == z).map(|n| (n.a, n.mass)).expect("a species of the scene");
            queue.push_back(Job::Reference(AtomScan::new(&base.cell, base.sigma, z, a, mass)));
        }

        // 2. perturbations
        let mut rng = StdRng::seed_from_u64(plan.seed);
        for _ in 0..plan.perturbations {
            let mut sc = base.clone();
            for n in &mut sc.nuclei {
                for x in &mut n.pos {
                    *x += plan.perturbation_sigma * standard_normal(&mut rng);
                }
            }
            queue.push_back(if inside_cell(&sc) { Job::Perturbation(sc) } else { Job::Skip("perturbations", None) });
        }

        // 3. pair scans, ranges from the scene's own separations
        let (lo, hi) = plan.scan_range;
        let m = plan.scan_points_per_pair;
        let mut scans = Vec::new();
        let nuc = &base.nuclei;
        for i in 0..nuc.len() {
            for j in i + 1..nuc.len() {
                let d: [f64; 3] = std::array::from_fn(|a| nuc[j].pos[a] - nuc[i].pos[a]);
                let r0 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                if r0 >= r_cut || m == 0 {
                    continue;
                }
                if r0 == 0.0 {
                    notes.push(format!("sampling: nuclei {i} and {j} coincide — no axis to scan, pair skipped"));
                    continue;
                }
                let (moved, fixed) = if nuc[i].mass < nuc[j].mass { (i, j) } else { (j, i) };
                let u: [f64; 3] = std::array::from_fn(|a| (nuc[moved].pos[a] - nuc[fixed].pos[a]) / r0);
                let r: Vec<f64> = (0..m)
                    .map(|k| {
                        let f = if m == 1 { 0.5 * (lo + hi) } else { lo + (hi - lo) * k as f64 / (m - 1) as f64 };
                        r0 * f
                    })
                    .collect();
                let pair = scans.len();
                for (k, &rk) in r.iter().enumerate() {
                    let mut sc = base.clone();
                    let p_fixed = sc.nuclei[fixed].pos;
                    sc.nuclei[moved].pos = std::array::from_fn(|a| p_fixed[a] + rk * u[a]);
                    queue.push_back(if inside_cell(&sc) { Job::Scan(sc, pair, k) } else { Job::Skip("scans", Some((pair, k))) });
                }
                scans.push(PairScan { i, j, moved, r0, r });
            }
        }

        // 4. the hot burst: Maxwell–Boltzmann start at T, CSVR at T
        let burst_seed = plan.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut csvr = Csvr::new(plan.hot_burst_t, HOT_BURST_TAU_FS, burst_seed);
        let mut burst_scene = base.clone();
        csvr.maxwell_boltzmann(&mut burst_scene);
        let burst = Burst {
            left: plan.hot_burst_steps,
            scene: burst_scene,
            forces: None,
            csvr,
            dt: units::fs_to_au_time(HOT_BURST_DT_FS),
        };

        let refs_total = species.iter().map(|&z| allowed_multiplicities(z as u32).len()).sum::<usize>();
        let total = Self::jobs_in(&queue) + plan.hot_burst_steps;
        if total > MAX_JOBS {
            return Err(format!("{total} jobs planned (pairs × scan points included): at most {MAX_JOBS} per pass"));
        }
        let stage = Self::stage_of(queue.front(), burst.left);
        Ok(Sampler {
            scene: base,
            plan: plan.clone(),
            refs_total,
            queue,
            scan_planned: scans.iter().map(|s| s.r.len()).collect(),
            scans,
            burst,
            refs: ReferenceCache::default(),
            progress: SamplingProgress {
                done: 0,
                total,
                stage,
                last_energy: f64::NAN,
                cancelled: false,
                skipped: 0,
                failed: 0,
                finished: total == 0,
            },
            notes,
            added: 0,
            boxed: 0,
            last_n: None,
            last_scan: None,
            interrupted: false,
        })
    }

    /// Starts from an existing reference cache (references found there cost no solve).
    pub fn with_references(mut self, cache: ReferenceCache) -> Sampler {
        self.refs = cache;
        self
    }

    /// The reference cache, with whatever this pass solved added (the caller keeps it for the
    /// next pass).
    pub fn take_references(&mut self) -> ReferenceCache {
        std::mem::take(&mut self.refs)
    }

    pub fn progress(&self) -> SamplingProgress {
        self.progress.clone()
    }

    /// The planned pair scans (as built: after a [`rescale`](Self::rescale) an evenly spaced
    /// subset of each one's points is solved).
    pub fn scans(&self) -> &[PairScan] {
        &self.scans
    }

    /// Samples handed to the surrogate so far.
    pub fn samples_added(&self) -> usize {
        self.added
    }

    /// Lines for the user (reference energies and their multiplicity, skipped pairs, failures);
    /// drained by the caller.
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    /// Solves the queued jobs stand for (a reference job counts its multiplicities left).
    fn jobs_in(queue: &VecDeque<Job>) -> usize {
        queue
            .iter()
            .map(|j| match j {
                Job::Reference(s) => s.remaining().len(),
                _ => 1,
            })
            .sum()
    }

    /// Shrinks the running pass to `plan`'s counts — its perturbations, scan points per pair and
    /// hot-burst steps; nothing else of `plan` is read (M6 Task 9: the viewer's automatic first
    /// pass, re-sized from the median of its first solves with [`PlanSolves::scaled_to`]). Only
    /// jobs not yet run are dropped: the perturbations from the tail of the seeded sequence, each
    /// pair scan thinned to an evenly spaced subset of its queued points (both ends kept), the
    /// burst shortened. **The references are always kept**, and the non-reference jobs the pass
    /// plans (run + queued) never drop below [`PlanSolves::min_samples_for_trust`] (or the whole
    /// plan, when that is less): a `plan` under the floor is raised to it by scaling the current
    /// counts with one common factor, then trimmed to the floor exactly: the burst's last steps
    /// first, then the last queued perturbations, and only when neither is left the queue's tail
    /// (which may be a scan end). Never grows a pass (each count is capped by the current one);
    /// `SamplingProgress::total` follows. No-op on a finished pass.
    pub fn rescale(&mut self, plan: &SamplingPlan) {
        if self.progress.finished {
            return;
        }
        let pairs = self.scans.len();
        let cur = (self.plan.perturbations, self.plan.scan_points_per_pair, self.plan.hot_burst_steps);
        // the non-reference jobs a set of counts plans (a thinned pair counted at its own size)
        let planned = self.scan_planned.clone();
        let nonref = |(p, m, b): (usize, usize, usize)| p + planned.iter().map(|&n| n.min(m)).sum::<usize>() + b;
        let mut want = (plan.perturbations.min(cur.0), plan.scan_points_per_pair.min(cur.1), plan.hot_burst_steps.min(cur.2));
        let floor = <SamplingPlan as PlanSolves>::min_samples_for_trust(&self.scene).min(nonref(cur));
        // the exact count to end at: the plan's, or the floor (before the raise below rounds up)
        let target = nonref(want).max(floor);
        if nonref(want) < floor {
            // the current proportions, by the smallest common factor that reaches the floor
            let at = |f: f64| ((cur.0 as f64 * f).ceil() as usize, (cur.1 as f64 * f).ceil() as usize, (cur.2 as f64 * f).ceil() as usize);
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if nonref(at(mid)) >= floor {
                    hi = mid
                } else {
                    lo = mid
                }
            }
            want = at(hi);
        }
        let (p, m, b) = want;

        // perturbations: the head of the seeded sequence
        let is_pert = |j: &Job| matches!(j, Job::Perturbation(_) | Job::Skip("perturbations", _));
        let ran_p = cur.0 - self.queue.iter().filter(|j| is_pert(j)).count();
        let mut keep_p = p.saturating_sub(ran_p);
        // scans: per pair, an evenly spaced subset of the queued points
        let scan_at = |j: &Job| match j {
            Job::Scan(_, pair, k) => Some((*pair, *k)),
            Job::Skip("scans", at) => *at,
            _ => None,
        };
        let mut queued: Vec<Vec<usize>> = vec![Vec::new(); pairs];
        for (pair, k) in self.queue.iter().filter_map(scan_at) {
            queued[pair].push(k);
        }
        let ran_k: Vec<usize> = (0..pairs).map(|pair| planned[pair] - queued[pair].len()).collect();
        let keep_k: Vec<Vec<usize>> = (0..pairs).map(|pair| spread(&queued[pair], m.min(planned[pair]).saturating_sub(ran_k[pair]))).collect();
        self.queue.retain(|j| {
            if is_pert(j) {
                let keep = keep_p > 0;
                keep_p = keep_p.saturating_sub(1);
                return keep;
            }
            match scan_at(j) {
                Some((pair, k)) => keep_k[pair].contains(&k),
                None => true,
            }
        });
        // the burst: shortened
        let ran_b = cur.2 - self.burst.left;
        self.burst.left = self.burst.left.min(b.saturating_sub(ran_b));

        // the ceilings' rounding: trimmed to the target exactly — the burst's last steps, then
        // the last queued perturbations, and only then the queue's tail (a scan end, review M4)
        let planned_nonref = (self.progress.done + Self::jobs_in(&self.queue) + self.burst.left).saturating_sub(self.refs_total);
        let mut excess = planned_nonref.saturating_sub(target);
        let cut = excess.min(self.burst.left);
        self.burst.left -= cut;
        excess -= cut;
        while excess > 0 {
            let Some(i) = self.queue.iter().rposition(|j| is_pert(j)) else { break };
            self.queue.remove(i);
            excess -= 1;
        }
        while excess > 0 && self.queue.back().is_some_and(|j| !matches!(j, Job::Reference { .. })) {
            self.queue.pop_back();
            excess -= 1;
        }

        // the counts as they now stand (a later rescale works from them)
        let mut now = ran_k;
        for (pair, _) in self.queue.iter().filter_map(scan_at) {
            now[pair] += 1;
        }
        self.plan.scan_points_per_pair = now.iter().copied().max().unwrap_or(0);
        self.scan_planned = now;
        self.plan.perturbations = ran_p + self.queue.iter().filter(|j| is_pert(j)).count();
        self.plan.hot_burst_steps = ran_b + self.burst.left;
        self.progress.total = self.progress.done + Self::jobs_in(&self.queue) + self.burst.left;
        self.progress.stage = Self::stage_of(self.queue.front(), self.burst.left);
    }

    /// Ends the pass early: what was sampled stays in the surrogate.
    pub fn cancel(&mut self) {
        if !self.progress.finished {
            self.progress.cancelled = true;
            self.progress.finished = true;
            self.progress.stage = "cancelled";
        }
    }

    /// At most one solve by `teacher`, its sample pushed into `surrogate` (module doc).
    /// `Ok(Some(progress))` after a solve; `Ok(None)` when nothing is left (the pass is
    /// finished); `Err(Cancelled)` when the teacher abandoned the solve (the job stays queued);
    /// any other `Err` only from a reference solve, which ends the pass.
    /// `Err(Invalid)` for a `teacher` that is not the surrogate's own (module doc, "The
    /// teacher"): its answers would be filed under the wrong provenance.
    pub fn step(&mut self, teacher: &mut dyn ForceProvider, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        let own = surrogate.teacher_mut().name();
        if teacher.name() != own {
            return Err(ProviderError::Invalid(format!(
                "sampling: teacher '{}' is not the surrogate's teacher '{own}' — its samples would be filed under the wrong provenance",
                teacher.name()
            )));
        }
        self.step_impl(Some(teacher), surrogate, should_abort)
    }

    /// The teacher solved a geometry outside this pass (a live evaluation while the pass runs,
    /// review m2): the next solve drops the density extrapolation (`discontinuity`).
    pub fn teacher_used_elsewhere(&mut self) {
        self.interrupted = true;
        self.last_scan = None;
    }

    /// [`step`](Self::step) with the surrogate's own teacher (the live case: `Dynamics` owns the
    /// surrogate, which owns the teacher).
    pub fn step_own_teacher(&mut self, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        self.step_impl(None, surrogate, should_abort)
    }

    fn stage_of(job: Option<&Job>, burst_left: usize) -> &'static str {
        match job {
            Some(Job::Reference(_)) => "references",
            Some(Job::Perturbation(_)) | Some(Job::Skip("perturbations", _)) => "perturbations",
            Some(Job::Scan(..)) | Some(Job::Skip("scans", _)) => "scans",
            Some(Job::Skip(other, _)) => other,
            None if burst_left > 0 => "hot burst",
            None => "done",
        }
    }

    fn step_impl(&mut self, mut teacher: Option<&mut dyn ForceProvider>, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        loop {
            if self.progress.finished {
                return Ok(None);
            }
            let Some(job) = self.queue.pop_front() else {
                if self.burst.left == 0 {
                    self.finish();
                    return Ok(None);
                }
                self.progress.stage = "hot burst";
                if self.burst_step(&mut teacher, surrogate, should_abort)? {
                    return Ok(Some(self.progress()));
                }
                continue; // the burst left the cell: no solve, look again
            };
            self.progress.stage = Self::stage_of(Some(&job), self.burst.left);
            match job {
                Job::Skip(stage, _) => {
                    self.progress.done += 1;
                    self.progress.skipped += 1;
                    let mut ev = TeachEvent::new(TeachOrigin::Pass, TeachRule::Cell, TeachOutcome::NotSolved);
                    ev.detail = format!("sampling: a generated geometry of the {stage} is closer than {CELL_MARGIN_H}h to a cell face — not solved");
                    ev.n_atoms = self.scene.nuclei.len();
                    surrogate.push_event(ev);
                }
                Job::Reference(mut scan) => {
                    let z = scan.z();
                    let prov = surrogate.provenance_for(&self.scene);
                    let key = atom_key(&prov, z);
                    if !scan.started() {
                        // the cache (the driver's references, an earlier pass, a loaded model):
                        // the key carries the tolerance and the teacher, so a hit is this
                        // surface's energy
                        if let Some(r) = self.refs.get(&key) {
                            let (e, m) = (r.energy, r.multiplicity);
                            surrogate.set_reference_energy(z, e);
                            self.progress.done += scan.remaining().len();
                            let m = m.map_or("from a loaded model".to_string(), |m| format!("multiplicity {m}"));
                            self.notes.push(format!("sampling: reference {} (Z = {z}) from the cache: E = {e:.6} Ha, {m}", symbol(z)));
                            continue;
                        }
                    }
                    let m = *scan.remaining().first().expect("a reference job is consumed when its last multiplicity is solved");
                    let t: &mut dyn ForceProvider = match &mut teacher {
                        Some(t) => &mut **t,
                        None => surrogate.teacher_mut(),
                    };
                    // each multiplicity cold (the scan resets before it: another spin state's
                    // orbitals must not steer the SCF); the molecule that follows resets too
                    self.last_n = None;
                    self.interrupted = false;
                    // the time includes the scan's density binning after the solve (small, but
                    // not pure teacher time); `AtomScan::step` reports no SCF steps (None)
                    let started = Instant::now();
                    let outcome = scan.step(t, should_abort);
                    let solve_ms = started.elapsed().as_secs_f64() * 1e3;
                    if let Err(ProviderError::Cancelled) = outcome {
                        self.queue.push_front(Job::Reference(scan));
                        return Err(ProviderError::Cancelled);
                    }
                    self.progress.done += 1;
                    let rule = TeachRule::Pass(PassJob::Reference { z, multiplicity: m });
                    let failure = match outcome {
                        Ok(solved) => {
                            // `Ok(None)` (nothing left to solve) cannot happen here — `m` was taken
                            // from `remaining()` above — and would be no solve: no event (review M6)
                            if let Some((_, e)) = solved {
                                let mut ev = TeachEvent::new(TeachOrigin::Pass, rule, TeachOutcome::Reference);
                                ev.detail = format!("sampling: reference {} (Z = {z}) at multiplicity {m}: E = {e:.6} Ha", symbol(z));
                                ev.n_atoms = 1;
                                ev.solve_ms = solve_ms;
                                surrogate.push_event(ev);
                            }
                            None
                        }
                        Err(e) => {
                            self.progress.failed += 1;
                            let note = format!("sampling: reference {} (Z = {z}) at multiplicity {m} failed ({e}) — left out of the scan", symbol(z));
                            let mut ev = TeachEvent::new(TeachOrigin::Pass, rule, TeachOutcome::Failed);
                            ev.detail = note.clone();
                            ev.n_atoms = 1;
                            ev.solve_ms = solve_ms;
                            surrogate.push_event(ev);
                            self.notes.push(note);
                            Some(e)
                        }
                    };
                    if !scan.is_done() {
                        self.queue.push_front(Job::Reference(scan));
                        return Ok(Some(self.progress()));
                    }
                    // the scan is complete: the lowest energy wins (ruling S9)
                    let tried: Vec<(u32, f64)> = scan.tried().collect();
                    let Some(r) = scan.finish(prov.tol, &prov.teacher) else {
                        self.notes.push(format!("sampling: no multiplicity of {} (Z = {z}) could be solved — the pass ends (no fit without a reference)", symbol(z)));
                        self.finish();
                        return Err(failure.expect("no success means the last solve failed"));
                    };
                    let (m_best, e_best) = (r.multiplicity.expect("a solved scan has its multiplicity"), r.energy);
                    self.refs.insert(key, r);
                    surrogate.set_reference_energy(z, e_best);
                    let others: Vec<String> = tried.iter().filter(|t| t.0 != m_best).map(|(m, e)| format!("{m} ({e:.6})")).collect();
                    let c = &self.scene.cell;
                    self.notes.push(format!(
                        "sampling: reference {} (Z = {z}): multiplicity {m_best} (E = {e_best:.6} Ha){} — isolated neutral atom at the cell centre, {}³ h {} σ {}, each multiplicity solved cold",
                        symbol(z),
                        if others.is_empty() { " — the only one solved".to_string() } else { format!(" chosen over {}", others.join(", ")) },
                        c.n,
                        c.h,
                        self.scene.sigma.unwrap_or(c.h),
                    ));
                    return Ok(Some(self.progress()));
                }
                Job::Perturbation(sc) => match self.sample(&mut teacher, surrogate, &sc, SampleTag::Perturbation, true, should_abort) {
                    Err(e) => {
                        self.queue.push_front(Job::Perturbation(sc));
                        return Err(e);
                    }
                    Ok(_) => return Ok(Some(self.progress())),
                },
                Job::Scan(sc, pair, k) => {
                    let fresh = k == 0 || self.last_scan != Some((pair, k - 1));
                    match self.sample(&mut teacher, surrogate, &sc, SampleTag::Scan, fresh, should_abort) {
                        Err(e) => {
                            self.queue.push_front(Job::Scan(sc, pair, k));
                            return Err(e);
                        }
                        Ok(solved) => {
                            self.last_scan = solved.then_some((pair, k));
                            return Ok(Some(self.progress()));
                        }
                    }
                }
            }
        }
    }

    /// The teacher's answer at `scene`, after the warm-state hint (module doc): `reset` when
    /// the nucleus count changed, `discontinuity` for a `fresh` (unrelated) geometry. A result
    /// with the wrong shape or a non-finite number is an error (it must never reach the store).
    fn solve(
        &mut self,
        teacher: &mut Option<&mut dyn ForceProvider>,
        surrogate: &mut SurrogateKs,
        scene: &Scene,
        fresh: bool,
        should_abort: &mut dyn FnMut() -> bool,
    ) -> Result<ForceResult, ProviderError> {
        let t: &mut dyn ForceProvider = match teacher {
            Some(t) => &mut **t,
            None => surrogate.teacher_mut(),
        };
        let n = scene.nuclei.len();
        if self.last_n != Some(n) {
            t.reset();
        } else if fresh || self.interrupted {
            t.discontinuity();
        }
        self.last_n = Some(n);
        self.interrupted = false;
        let r = t.evaluate(scene, should_abort)?;
        if r.forces.len() != n {
            return Err(ProviderError::Invalid(format!("teacher '{}' returned {} forces for {n} nuclei", t.name(), r.forces.len())));
        }
        if !r.energy.is_finite() || r.forces.iter().flatten().any(|x| !x.is_finite()) {
            return Err(ProviderError::Invalid(format!("teacher '{}' returned a non-finite energy or force", t.name())));
        }
        self.progress.last_energy = r.energy;
        Ok(r)
    }

    /// A sample job: solve, keep. `Ok(true)` solved and kept (or refused by the store),
    /// `Ok(false)` the teacher failed (counted, noted; the pass goes on), `Err` only for
    /// `Cancelled` (the caller re-queues the job).
    fn sample(
        &mut self,
        teacher: &mut Option<&mut dyn ForceProvider>,
        surrogate: &mut SurrogateKs,
        scene: &Scene,
        tag: SampleTag,
        fresh: bool,
        should_abort: &mut dyn FnMut() -> bool,
    ) -> Result<bool, ProviderError> {
        let started = Instant::now();
        match self.solve(teacher, surrogate, scene, fresh, should_abort) {
            Ok(r) => {
                self.record(surrogate, scene, r, tag, None);
                Ok(true)
            }
            Err(ProviderError::Cancelled) => Err(ProviderError::Cancelled),
            Err(e) => {
                self.progress.done += 1;
                self.progress.failed += 1;
                let note = format!("sampling: a {} geometry failed ({e}) — skipped", self.progress.stage);
                let mut ev = TeachEvent::new(TeachOrigin::Pass, TeachRule::Pass(pass_job(tag)), TeachOutcome::Failed);
                ev.detail = note.clone();
                ev.n_atoms = scene.nuclei.len();
                ev.solve_ms = started.elapsed().as_secs_f64() * 1e3;
                surrogate.push_event(ev);
                self.notes.push(note);
                Ok(false)
            }
        }
    }

    /// Hands a solved geometry to the surrogate as a training sample, with the teacher's face
    /// density (module doc, "Box-affected samples"), and leaves its activity event (`burst_t`: a
    /// hot-burst geometry's temperature, module doc "Activity events").
    fn record(&mut self, surrogate: &mut SurrogateKs, scene: &Scene, r: ForceResult, tag: SampleTag, burst_t: Option<f64>) {
        let face = r.stats.face_density;
        let (solve_ms, scf) = (r.stats.solve_ms as f64, r.stats.scf_steps);
        let detail = format!("sampling: {} {}/{}", self.progress.stage, self.progress.done + 1, self.progress.total);
        let sample = Sample {
            pos: scene.nuclei.iter().map(|n| n.pos).collect(),
            z: scene.nuclei.iter().map(|n| n.z).collect(),
            energy: r.energy,
            forces: r.forces,
            tag,
            weight: 1.0,
        };
        let boxed = face.is_some_and(|rho| rho > box_affected_face_density(scene.cell.h));
        let outcome = match surrogate.add_sample_with_face(sample, scene, face) {
            Ok(()) => {
                self.added += 1;
                if boxed {
                    self.boxed += 1;
                }
                if boxed { TeachOutcome::KeptBoxAffected } else { TeachOutcome::Kept }
            }
            Err(e) => {
                self.progress.failed += 1;
                self.notes.push(format!("sampling: the store refused a {} sample ({e})", self.progress.stage));
                TeachOutcome::StoreRefused
            }
        };
        self.progress.done += 1;
        let mut ev = TeachEvent::new(TeachOrigin::Pass, TeachRule::Pass(pass_job(tag)), outcome);
        ev.face_density = face;
        ev.box_affected = boxed;
        ev.n_atoms = scene.nuclei.len();
        ev.solve_ms = solve_ms;
        ev.scf_steps = Some(scf);
        ev.burst_t = burst_t;
        ev.detail = detail;
        surrogate.push_event(ev);
    }

    /// One burst evaluation (module doc): the start geometry, or a velocity-Verlet step with
    /// the thermostat after it. `Ok(false)` when the step would leave the cell (the burst ends,
    /// its remaining evaluations are skipped, no solve happened).
    fn burst_step(&mut self, teacher: &mut Option<&mut dyn ForceProvider>, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<bool, ProviderError> {
        let dt = self.burst.dt;
        let mut work = self.burst.scene.clone();
        let fresh = self.burst.forces.is_none();
        if let Some(f) = &self.burst.forces {
            Integrator::half_kick(&mut work, f, dt);
            Integrator::drift(&mut work, dt);
        }
        if !inside_cell(&work) {
            let left = std::mem::take(&mut self.burst.left);
            self.progress.done += left;
            self.progress.skipped += left;
            let note = format!("sampling: the hot burst reached the cell margin; its last {left} evaluations were skipped");
            let mut ev = TeachEvent::new(TeachOrigin::Pass, TeachRule::Cell, TeachOutcome::NotSolved);
            ev.detail = note.clone();
            ev.n_atoms = work.nuclei.len();
            surrogate.push_event(ev);
            self.notes.push(note);
            return Ok(false);
        }
        let started = Instant::now();
        match self.solve(teacher, surrogate, &work, fresh, should_abort) {
            Ok(r) => {
                let f_new = r.forces.clone();
                if !fresh {
                    Integrator::half_kick(&mut work, &f_new, dt);
                }
                // the instantaneous kinetic T (3N − 3 dof, `Scene::temperature`) of the solved
                // geometry: the Verlet velocities v(t + dt), before CSVR rescales them (plan
                // review S6); at the burst's start the Maxwell–Boltzmann draw's. A read only:
                // the thermostat's draws are untouched
                let burst_t = work.temperature();
                if !fresh {
                    self.burst.csvr.apply(&mut work, dt);
                }
                self.record(surrogate, &work, r, SampleTag::HotBurst, Some(burst_t));
                self.burst.scene = work;
                self.burst.forces = Some(f_new);
                self.burst.left -= 1;
                Ok(true)
            }
            Err(ProviderError::Cancelled) => Err(ProviderError::Cancelled),
            Err(e) => {
                // the trajectory has no forces to continue with: the burst ends here
                let rest = std::mem::take(&mut self.burst.left) - 1;
                self.progress.done += 1 + rest;
                self.progress.failed += 1;
                self.progress.skipped += rest;
                let note = format!("sampling: a hot-burst step failed ({e}); the burst ends, {rest} evaluations skipped");
                let mut ev = TeachEvent::new(TeachOrigin::Pass, TeachRule::Pass(PassJob::HotBurst), TeachOutcome::Failed);
                ev.detail = note.clone();
                ev.n_atoms = work.nuclei.len();
                ev.solve_ms = started.elapsed().as_secs_f64() * 1e3;
                surrogate.push_event(ev);
                self.notes.push(note);
                Ok(true)
            }
        }
    }

    fn finish(&mut self) {
        self.progress.finished = true;
        if !self.progress.cancelled {
            self.progress.stage = "done";
        }
        if self.progress.skipped > 0 {
            self.notes.push(format!(
                "sampling: {} generated geometries were closer than {CELL_MARGIN_H}h to a cell face and were not solved",
                self.progress.skipped
            ));
        }
        if self.boxed > 0 {
            self.notes.push(format!(
                "sampling: {} of {} samples are box-affected (the teacher's density on the outermost grid plane above {:.1e}) — kept while the model is built, never quarantined, left out of the fit once enough others earn trust; a bigger box teaches those geometries",
                self.boxed,
                self.added,
                box_affected_face_density(self.scene.cell.h)
            ));
        }
    }
}

/// The activity code of a pass sample's job (`TeachRule::Pass`).
fn pass_job(tag: SampleTag) -> PassJob {
    match tag {
        SampleTag::Perturbation => PassJob::Perturbation,
        SampleTag::Scan => PassJob::Scan,
        _ => PassJob::HotBurst,
    }
}

/// Drives a whole pass (module doc) with `teacher`, which must be the surrogate's own (by
/// name; module doc, "The teacher"): `progress` after every solve and once at the end,
/// `should_abort` polled between solves and handed to the teacher (an abort cancels the pass;
/// the samples so far stay). Returns the number of samples added. Does not refit.
pub fn run_sampling(
    scene: &Scene,
    plan: &SamplingPlan,
    teacher: &mut dyn ForceProvider,
    surrogate: &mut SurrogateKs,
    progress: &mut dyn FnMut(SamplingProgress),
    should_abort: &mut dyn FnMut() -> bool,
) -> Result<usize, ProviderError> {
    let r_cut = surrogate.basis().map_or(Basis::new(&[]).r_cut, |b| b.r_cut);
    let mut s = Sampler::with_cutoff(scene, plan, r_cut).map_err(ProviderError::Invalid)?;
    loop {
        if should_abort() {
            s.cancel();
            break;
        }
        match s.step(teacher, surrogate, should_abort) {
            Ok(Some(p)) => progress(p),
            Ok(None) => break,
            Err(ProviderError::Cancelled) => {
                s.cancel();
                break;
            }
            Err(e) => return Err(e),
        }
    }
    progress(s.progress());
    Ok(s.samples_added())
}

#[cfg(test)]
mod tests;
