//! The sampling pass (spec §4 (b)): drives the DFT teacher over geometries derived from the
//! current scene and feeds every answer to a [`SurrogateKs`] as a training sample. Atomic units
//! throughout (Bohr, Hartree); the plan's temperature is in kelvin and the burst's τ and dt in
//! fs only as constants at this edge.
//!
//! # What a pass does, in order
//!
//! 1. **References** (spec §3, "Reference energies"): one isolated-atom solve per species
//!    present, by the teacher, with the single nucleus at the centre of the scene's own grid
//!    (the origin: `qm::Grid::cube` is centred there), the scene's cell and σ, charge 0 and
//!    `multiplicity: None` — the solver's lowest multiplicity for Z electrons (1 + Z mod 2).
//!    A high-spin atomic ground state (Hund's rule: C, N, O are triplet/quartet/triplet) is not
//!    searched, so for those elements the reference sits above the true LSDA atom by the spin
//!    polarisation energy; the model's binding energies are then relative to that reference
//!    (the gap is documented in each reference note, rule 7). The energy is set with
//!    `SurrogateKs::set_reference_energy` — never a table (rule 1) — and cached in a
//!    [`ReferenceCache`] keyed by the atom's settings (grid, h, σ, teacher, tol) and Z, so a
//!    later pass at the same settings skips the solve. References are not store samples: an
//!    isolated atom's charge and multiplicity differ from a charged or spin-fixed scene's, so it
//!    would carry another provenance.
//! 2. **Perturbations**: `perturbations` copies of the scene with every coordinate displaced by
//!    an independent Gaussian of width `perturbation_sigma` (0.15 Bohr), seeded by `seed`.
//! 3. **Pair scans**: for every pair closer than the model's cutoff (`Basis::new(..).r_cut`,
//!    6 Bohr) — coincident pairs have no axis and are skipped with a note — the separation is
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
//! The teacher's warm state is managed the way M4 learned (Task 4 report): `reset` whenever the
//! nucleus count changes (reference ↔ scene), `discontinuity` before every geometry unrelated to
//! the previous one (each perturbation, the first point of each pair scan, the burst start) —
//! warm orbitals kept, no density extrapolation across the jump — and nothing inside a scan or
//! the burst, whose consecutive points are equally spaced along one path. The caller should
//! call `discontinuity` once more after the pass before the live run continues (`Dynamics`
//! does).
//!
//! # Resumable
//!
//! [`Sampler`] is a state machine: [`Sampler::step`] performs at most one teacher solve and
//! returns, so the dynamics worker can advance it one solve per tick slice and stay responsive.
//! A solve the teacher abandons (`ProviderError::Cancelled`) leaves the job queued: the next
//! `step` retries it. [`run_sampling`] is the convenience loop over it. Neither refits — the
//! caller does (a partial pass refits like a full one).

use crate::integrator::Integrator;
use crate::provider::{symbol, ForceProvider, ForceResult, ProviderError};
use crate::scene::{Nucleus, Scene, CELL_MARGIN_H};
use crate::surrogate_ks::SurrogateKs;
use crate::thermostat::Csvr;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;
use surrogate::{Basis, Provenance, Sample, SampleTag};

/// The hot burst's CSVR coupling time (fs).
pub const HOT_BURST_TAU_FS: f64 = 5.0;
/// The hot burst's time step (fs).
pub const HOT_BURST_DT_FS: f64 = 0.25;

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

impl SamplingPlan {
    /// Finite non-negative σ and T, and 0 < scan_range.0 ≤ scan_range.1, both finite.
    pub fn validate(&self) -> Result<(), String> {
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

/// Isolated-atom energies already solved, keyed by the atom's solve settings and Z (module doc).
#[derive(Clone, Debug, Default)]
pub struct ReferenceCache {
    entries: Vec<(Provenance, u8, f64)>,
}

impl ReferenceCache {
    /// The settings an isolated atom of `scene_provenance`'s solve is computed at: the same
    /// grid, σ, tolerance and teacher, neutral, lowest multiplicity.
    pub fn atom_provenance(scene_provenance: &Provenance) -> Provenance {
        Provenance { charge: 0, multiplicity: None, ..scene_provenance.clone() }
    }

    pub fn get(&self, atom: &Provenance, z: u8) -> Option<f64> {
        self.entries.iter().find(|(p, zz, _)| p == atom && *zz == z).map(|e| e.2)
    }

    pub fn insert(&mut self, atom: Provenance, z: u8, e: f64) {
        match self.entries.iter_mut().find(|(p, zz, _)| *p == atom && *zz == z) {
            Some(entry) => entry.2 = e,
            None => self.entries.push((atom, z, e)),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
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
    Reference(u8),
    Perturbation(Scene),
    /// A scan point: the geometry, its pair index and point index.
    Scan(Scene, usize, usize),
    /// A geometry the cell guard rejected, with the stage it belonged to.
    Skip(&'static str),
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
    queue: VecDeque<Job>,
    scans: Vec<PairScan>,
    burst: Burst,
    refs: ReferenceCache,
    progress: SamplingProgress,
    notes: Vec<String>,
    added: usize,
    /// Nucleus count of the last scene sent to the teacher.
    last_n: Option<usize>,
    /// (pair, point) of the last scan point sent to the teacher.
    last_scan: Option<(usize, usize)>,
}

/// Every nucleus at least [`CELL_MARGIN_H`]·h inside every face, finite (the providers' R12
/// test, `provider::check_inside_cell`).
fn inside_cell(scene: &Scene) -> bool {
    let max = scene.cell.max_coord();
    scene.nuclei.iter().all(|n| n.pos.iter().all(|x| x.is_finite() && x.abs() <= max))
}

/// A standard normal (Marsaglia's polar method, as `Csvr` draws them).
fn gaussian(rng: &mut StdRng) -> f64 {
    loop {
        let u = 2.0 * rng.gen::<f64>() - 1.0;
        let v = 2.0 * rng.gen::<f64>() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

impl Sampler {
    /// Plans a pass over `scene` (velocities and the external field are not used: the pass
    /// samples the potential-energy surface). Err for an invalid plan.
    pub fn new(scene: &Scene, plan: &SamplingPlan) -> Result<Sampler, String> {
        plan.validate()?;
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
        queue.extend(species.iter().map(|&z| Job::Reference(z)));

        // 2. perturbations
        let mut rng = StdRng::seed_from_u64(plan.seed);
        for _ in 0..plan.perturbations {
            let mut sc = base.clone();
            for n in &mut sc.nuclei {
                for x in &mut n.pos {
                    *x += plan.perturbation_sigma * gaussian(&mut rng);
                }
            }
            queue.push_back(if inside_cell(&sc) { Job::Perturbation(sc) } else { Job::Skip("perturbations") });
        }

        // 3. pair scans, ranges from the scene's own separations
        let r_cut = Basis::new(&[]).r_cut;
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
                    queue.push_back(if inside_cell(&sc) { Job::Scan(sc, pair, k) } else { Job::Skip("scans") });
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

        let total = queue.len() + plan.hot_burst_steps;
        let stage = Self::stage_of(queue.front(), burst.left);
        Ok(Sampler {
            scene: base,
            queue,
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
            last_n: None,
            last_scan: None,
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

    /// The planned pair scans.
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
    pub fn step(&mut self, teacher: &mut dyn ForceProvider, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        self.step_impl(Some(teacher), surrogate, should_abort)
    }

    /// [`step`](Self::step) with the surrogate's own teacher (the live case: `Dynamics` owns the
    /// surrogate, which owns the teacher).
    pub fn step_own_teacher(&mut self, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        self.step_impl(None, surrogate, should_abort)
    }

    fn stage_of(job: Option<&Job>, burst_left: usize) -> &'static str {
        match job {
            Some(Job::Reference(_)) => "references",
            Some(Job::Perturbation(_)) | Some(Job::Skip("perturbations")) => "perturbations",
            Some(Job::Scan(..)) | Some(Job::Skip(_)) => "scans",
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
                Job::Skip(_) => {
                    self.progress.done += 1;
                    self.progress.skipped += 1;
                }
                Job::Reference(z) => {
                    let key = ReferenceCache::atom_provenance(&surrogate.provenance_for(&self.scene));
                    if let Some(e) = self.refs.get(&key, z) {
                        surrogate.set_reference_energy(z, e);
                        self.progress.done += 1;
                        self.notes.push(format!("sampling: reference {} (Z = {z}) from the cache: E = {e:.6} Ha", symbol(z)));
                        continue;
                    }
                    let atom = self.atom_scene(z);
                    match self.solve(&mut teacher, surrogate, &atom, true, should_abort) {
                        Ok(r) => {
                            self.refs.insert(key, z, r.energy);
                            surrogate.set_reference_energy(z, r.energy);
                            self.progress.done += 1;
                            let c = &self.scene.cell;
                            self.notes.push(format!(
                                "sampling: reference {} (Z = {z}): E = {:.6} Ha — one atom at the cell centre, {}³ h {} σ {}, neutral, multiplicity {} (the solver's lowest for {z} electron{}; a higher-spin atomic ground state is not searched, so the reference may lie above it)",
                                symbol(z),
                                r.energy,
                                c.n,
                                c.h,
                                self.scene.sigma.unwrap_or(c.h),
                                1 + z % 2,
                                if z == 1 { "" } else { "s" }
                            ));
                            return Ok(Some(self.progress()));
                        }
                        Err(ProviderError::Cancelled) => {
                            self.queue.push_front(Job::Reference(z));
                            return Err(ProviderError::Cancelled);
                        }
                        Err(e) => {
                            self.progress.done += 1;
                            self.progress.failed += 1;
                            self.notes.push(format!("sampling: the reference solve for {} (Z = {z}) failed ({e}) — the pass ends (no fit without it)", symbol(z)));
                            self.finish();
                            return Err(e);
                        }
                    }
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

    /// One isolated atom of species `z` at the cell centre (module doc, "References"): the
    /// scene's cell and σ, neutral, the solver's lowest multiplicity.
    fn atom_scene(&self, z: u8) -> Scene {
        let mass = self.scene.nuclei.iter().find(|n| n.z == z).map(|n| n.mass).expect("a species of the scene");
        Scene {
            nuclei: vec![Nucleus { pos: [0.0; 3], vel: [0.0; 3], z, mass }],
            cell: self.scene.cell.clone(),
            charge: 0,
            multiplicity: None,
            sigma: self.scene.sigma,
            external_field: [0.0; 3],
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
        } else if fresh {
            t.discontinuity();
        }
        self.last_n = Some(n);
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
        match self.solve(teacher, surrogate, scene, fresh, should_abort) {
            Ok(r) => {
                self.record(surrogate, scene, r, tag);
                Ok(true)
            }
            Err(ProviderError::Cancelled) => Err(ProviderError::Cancelled),
            Err(e) => {
                self.progress.done += 1;
                self.progress.failed += 1;
                self.notes.push(format!("sampling: a {} geometry failed ({e}) — skipped", self.progress.stage));
                Ok(false)
            }
        }
    }

    /// Hands a solved geometry to the surrogate as a training sample.
    fn record(&mut self, surrogate: &mut SurrogateKs, scene: &Scene, r: ForceResult, tag: SampleTag) {
        let sample = Sample {
            pos: scene.nuclei.iter().map(|n| n.pos).collect(),
            z: scene.nuclei.iter().map(|n| n.z).collect(),
            energy: r.energy,
            forces: r.forces,
            tag,
            weight: 1.0,
        };
        match surrogate.add_sample(sample, scene) {
            Ok(()) => self.added += 1,
            Err(e) => {
                self.progress.failed += 1;
                self.notes.push(format!("sampling: the store refused a {} sample ({e})", self.progress.stage));
            }
        }
        self.progress.done += 1;
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
            self.notes.push(format!("sampling: the hot burst reached the cell margin; its last {left} evaluations were skipped"));
            return Ok(false);
        }
        match self.solve(teacher, surrogate, &work, fresh, should_abort) {
            Ok(r) => {
                let f_new = r.forces.clone();
                if !fresh {
                    Integrator::half_kick(&mut work, &f_new, dt);
                    self.burst.csvr.apply(&mut work, dt);
                }
                self.record(surrogate, &work, r, SampleTag::HotBurst);
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
                self.notes.push(format!("sampling: a hot-burst step failed ({e}); the burst ends, {rest} evaluations skipped"));
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
    }
}

/// Drives a whole pass (module doc): `progress` after every solve and once at the end,
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
    let mut s = Sampler::new(scene, plan).map_err(ProviderError::Invalid)?;
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
