//! Plain records of what the providers are doing, for the app's Activity view (spec
//! `2026-10-15-activity-tab-design.md`): the live solve state ([`ActivityProbe`], atomics shared
//! between the solver threads and the render thread), one [`TeachEvent`] per teaching solve with
//! the rule that caused it and what happened to the answer, lifetime [`SolveCounts`], the trust
//! rule's conditions ([`TrustCheck`]), a stuck model's reason ([`StuckInfo`]) and the training
//! views' data ([`TrainingViews`]). Atomic units (Hartree, Bohr), kelvin, milliseconds. Nothing
//! here decides anything: the providers fill these from decisions they already made.
use crate::AuditVerdict;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Which solver a probe slot follows: the stepping teacher, or the background auditor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Main,
    Side,
}

#[derive(Debug, Default)]
struct SolveSlot {
    running: AtomicBool,
    scf_step: AtomicU32,
    started_us: AtomicU64,
    solves: AtomicU64,
    /// The running (or last) solve's per-iteration record (`ActivityProbe::scf_iteration`): written
    /// once per SCF iteration, read by the render thread at most every 100 ms — a lock per
    /// iteration against an iteration of milliseconds.
    live: Mutex<Option<LiveSlot>>,
}

#[derive(Debug, Clone, Copy)]
struct LiveSlot {
    live: ScfLive,
    /// When the current attempt started (µs since the probe's origin).
    attempt_us: u64,
    /// When the last iteration was reported.
    last_us: u64,
}

/// Points of the Δρ sparkline ([`ScfLive::spark`]).
pub const SPARK_POINTS: usize = 64;

/// One SCF iteration as the solver reports it to the probe (`qm::ScfProgress` without the
/// dependency): Δρ and its target, the Harris–Foulkes energy of the iteration's input density
/// (Ha; NaN when the solver does not compute it), and the stall stop's countdown — iterations
/// since the best Δρ last improved 2×, of `window` (0 = no stall stop) — with the best Δρ and the
/// stall level the stop also needs it at or above (`qm::SCF_STALL_LEVEL_PER_ELECTRON`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfIteration {
    pub iteration: usize,
    pub max_iter: usize,
    pub delta_rho: f64,
    pub tol: f64,
    pub energy: f64,
    pub since_gain: usize,
    pub window: usize,
    pub best_delta_rho: f64,
    pub stall_level: f64,
}

/// The running solve's iterations for the Activity view (activity spec §3.3.1; the stall stop's
/// countdown, `qm::scf` "The stall stop"). `attempt` is 2 on the providers' cold retry of a warm
/// solve that failed (R15): its iterations count from 1 again. `ms_per_iteration` is this
/// attempt's mean wall time per iteration; `d_energy` the change of the Harris–Foulkes energy
/// since the previous iteration (NaN at the first). `spark` holds log10 Δρ of every
/// `spark_stride`-th iteration (iterations `spark_stride`, 2·`spark_stride`, …): when it fills it
/// is decimated 2:1 and the stride doubles, so it always spans the whole attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfLive {
    pub attempt: u32,
    pub iteration: u32,
    pub max_iter: u32,
    pub delta_rho: f64,
    pub tol: f64,
    pub energy: f64,
    pub d_energy: f64,
    pub ms_per_iteration: f64,
    pub since_gain: u32,
    pub window: u32,
    /// The best Δρ of this attempt and the stall level: the stop counts down only while best ≥ level.
    pub best_delta_rho: f64,
    pub stall_level: f64,
    pub spark: [f32; SPARK_POINTS],
    pub spark_len: u8,
    pub spark_stride: u32,
}

impl ScfLive {
    /// The sparkline's points in order.
    pub fn spark(&self) -> &[f32] {
        &self.spark[..self.spark_len as usize]
    }

    /// Iterations left before the stall stop ends the solve: `None` without one, and while the best Δρ is
    /// below the stall level (the stop does not apply there: the solve converges or runs to the cap).
    pub fn stall_left(&self) -> Option<u32> {
        (self.window > 0 && self.best_delta_rho >= self.stall_level).then(|| self.window.saturating_sub(self.since_gain))
    }

    fn push_spark(&mut self, iteration: u32, delta_rho: f64) {
        if iteration % self.spark_stride != 0 {
            return;
        }
        let v = if delta_rho > 0.0 && delta_rho.is_finite() { delta_rho.log10() as f32 } else { f32::NAN };
        if self.spark_len as usize == SPARK_POINTS {
            // keep the points at even multiples of the stride (indices 1, 3, 5, …)
            for j in 0..SPARK_POINTS / 2 {
                self.spark[j] = self.spark[2 * j + 1];
            }
            self.spark_len = (SPARK_POINTS / 2) as u8;
            self.spark_stride *= 2;
            if iteration % self.spark_stride != 0 {
                return;
            }
        }
        self.spark[self.spark_len as usize] = v;
        self.spark_len += 1;
    }
}

/// Live solve state (spec §4.3 A1). Written with relaxed atomics by the solver threads, read by
/// the render thread; the numbers are for display, so no ordering between fields is promised.
#[derive(Debug)]
pub struct ActivityProbe {
    origin: Instant,
    main: SolveSlot,
    side: SolveSlot,
    /// Start of the refit in flight (µs since `origin`), `u64::MAX` when none.
    refit_started_us: AtomicU64,
    refit_samples: AtomicU32,
    /// Refits ended so far (`end_refit`).
    refits: AtomicU64,
}

/// Marks a slot running until dropped (an error, a cancel or a panic clears it too).
#[must_use]
pub struct Busy {
    probe: Arc<ActivityProbe>,
    slot: Slot,
}

impl Drop for Busy {
    fn drop(&mut self) {
        let s = self.probe.slot(self.slot);
        s.running.store(false, Relaxed);
        s.solves.fetch_add(1, Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SlotView {
    pub running: bool,
    /// SCF step being solved (running) or the last solve's count (idle), counted at the solver's
    /// per-step abort poll over every attempt; 0 for the CPU solver (it has no abort poll).
    pub scf_step: u32,
    /// Seconds since the running solve started (0 when idle).
    pub elapsed_s: f64,
    pub solves: u64,
    /// The running (or last) solve's latest iteration, from a solver that reports them
    /// ([`ActivityProbe::scf_iteration`]); `None` before its first iteration.
    pub live: Option<ScfLive>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProbeView {
    pub main: SlotView,
    pub side: SlotView,
    /// (seconds since the refit started, samples it fits) while one runs.
    pub refit: Option<(f64, u32)>,
    /// Refits ended so far (successful or not).
    pub refits: u64,
}

impl ActivityProbe {
    pub fn new() -> Arc<ActivityProbe> {
        Arc::new(ActivityProbe {
            origin: Instant::now(),
            main: SolveSlot::default(),
            side: SolveSlot::default(),
            refit_started_us: AtomicU64::new(u64::MAX),
            refit_samples: AtomicU32::new(0),
            refits: AtomicU64::new(0),
        })
    }

    fn slot(&self, s: Slot) -> &SolveSlot {
        match s {
            Slot::Main => &self.main,
            Slot::Side => &self.side,
        }
    }

    fn now_us(&self) -> u64 {
        self.origin.elapsed().as_micros() as u64
    }

    /// A solve starts on `slot`; it ends when the guard drops.
    pub fn busy(self: &Arc<Self>, slot: Slot) -> Busy {
        let s = self.slot(slot);
        s.scf_step.store(0, Relaxed);
        s.started_us.store(self.now_us(), Relaxed);
        *s.live.lock().unwrap_or_else(|e| e.into_inner()) = None;
        s.running.store(true, Relaxed);
        Busy { probe: self.clone(), slot }
    }

    /// One SCF iteration finished on `slot` (the solver's progress observer). An iteration number
    /// not above the previous one starts a new attempt (the providers' cold retry).
    pub fn scf_iteration(&self, slot: Slot, it: &ScfIteration) {
        let now = self.now_us();
        let s = self.slot(slot);
        let mut g = s.live.lock().unwrap_or_else(|e| e.into_inner());
        let iteration = it.iteration.min(u32::MAX as usize) as u32;
        let fresh = |attempt: u32, attempt_us: u64| LiveSlot {
            live: ScfLive {
                attempt,
                iteration: 0,
                max_iter: 0,
                delta_rho: f64::NAN,
                tol: f64::NAN,
                energy: f64::NAN,
                d_energy: f64::NAN,
                ms_per_iteration: f64::NAN,
                since_gain: 0,
                window: 0,
                best_delta_rho: f64::NAN,
                stall_level: f64::NAN,
                spark: [f32::NAN; SPARK_POINTS],
                spark_len: 0,
                spark_stride: 1,
            },
            attempt_us,
            last_us: attempt_us,
        };
        let mut l = match *g {
            Some(l) if iteration > l.live.iteration => l,
            Some(l) => fresh(l.live.attempt + 1, l.last_us),
            None => fresh(1, s.started_us.load(Relaxed)),
        };
        let prev_energy = l.live.energy;
        l.live.iteration = iteration;
        l.live.max_iter = it.max_iter.min(u32::MAX as usize) as u32;
        l.live.delta_rho = it.delta_rho;
        l.live.tol = it.tol;
        l.live.energy = it.energy;
        l.live.d_energy = it.energy - prev_energy;
        l.live.ms_per_iteration = now.saturating_sub(l.attempt_us) as f64 * 1e-3 / iteration.max(1) as f64;
        l.live.since_gain = it.since_gain.min(u32::MAX as usize) as u32;
        l.live.window = it.window.min(u32::MAX as usize) as u32;
        l.live.best_delta_rho = it.best_delta_rho;
        l.live.stall_level = it.stall_level;
        l.live.push_spark(iteration, it.delta_rho);
        l.last_us = now;
        *g = Some(l);
    }

    /// One SCF step starts (called from the solver's per-step abort poll).
    pub fn scf_step(&self, slot: Slot) {
        self.slot(slot).scf_step.fetch_add(1, Relaxed);
    }

    pub fn begin_refit(&self, samples: usize) {
        self.refit_samples.store(samples.min(u32::MAX as usize) as u32, Relaxed);
        self.refit_started_us.store(self.now_us(), Relaxed);
    }

    /// The refit in flight ended (successfully or not): the phase clears and `refits` counts it.
    pub fn end_refit(&self) {
        self.refit_started_us.store(u64::MAX, Relaxed);
        self.refits.fetch_add(1, Relaxed);
    }

    pub fn view(&self) -> ProbeView {
        let now = self.now_us();
        let slot = |s: &SolveSlot| {
            let running = s.running.load(Relaxed);
            SlotView {
                running,
                scf_step: s.scf_step.load(Relaxed),
                elapsed_s: if running { now.saturating_sub(s.started_us.load(Relaxed)) as f64 * 1e-6 } else { 0.0 },
                solves: s.solves.load(Relaxed),
                live: s.live.lock().unwrap_or_else(|e| e.into_inner()).map(|l| l.live),
            }
        };
        let r = self.refit_started_us.load(Relaxed);
        ProbeView {
            main: slot(&self.main),
            side: slot(&self.side),
            refit: (r != u64::MAX).then(|| (now.saturating_sub(r) as f64 * 1e-6, self.refit_samples.load(Relaxed))),
            refits: self.refits.load(Relaxed),
        }
    }
}

/// Who asked the teacher (spec §3.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeachOrigin {
    Pass,
    Fallback,
    Audit,
    /// The relax verification (`Trainable::force_teacher_next`).
    Verification,
}

/// A sampling-pass job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassJob {
    Reference { z: u8, multiplicity: u32 },
    Perturbation,
    Scan,
    HotBurst,
}

/// The class-wise trust rule's classes (`providers::surrogate::ENV_CLASSES` order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvClass {
    Pair,
    ManyBody,
}

/// The trust rule's conditions, in the order `trust_checks` lists them and `untrusted_check`
/// tests them (`providers::surrogate`). The audit condition is `Audit` (the per-atom energy gap,
/// or no failed audit since the refit) or `AuditForces` (the failed audit's worst atom force
/// residual): plan review M1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustKey {
    Samples,
    ResidualDofE,
    ResidualDofF,
    EnergyResidual,
    ForceResidual,
    ForceNoise,
    Audit,
    AuditForces,
}

/// Why the teacher answered (the fallback policy's decision list, `providers::surrogate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeachRule {
    /// 0: the cell check refused the geometry (a pass skip, or the run's refusal).
    Cell,
    NoModel,
    NewSpecies(u8),
    OtherSettings,
    Untrusted(TrustKey),
    Class(EnvClass),
    IsolatedAtom,
    Coordination,
    Spread,
    NoGrade,
    /// 5d: the geometry reaches directions no training row spans (the null-space novelty).
    NewEnvironment,
    /// 5e at γ = ∞: an atom's descriptor has a component along a direction no training row of
    /// its species spans (5d is the null-space novelty test; plan review S7).
    UnspannedAtom,
    Grade,
    ForceUncertainty,
    EnergyUncertainty,
    /// The model's prediction (energy, a force, or a std when evaluated) is not finite: never an
    /// answer, draft mode included (activity Task 3b).
    NonFiniteAnswer,
    NearQuarantine,
    Audit,
    RelaxVerification,
    Pass(PassJob),
}

/// What happened to the teacher's answer (`providers::surrogate::admit` and the pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeachOutcome {
    Kept,
    /// Kept although box-affected (the model is still being built; it leaves the fit later).
    KeptBoxAffected,
    KeptByVote,
    KeptReleased { released: u32, superseded: u32 },
    Thinned,
    BoxNotKept,
    /// A teacher answer that is used for its step and never trained on: its cores overlap (the frozen-core
    /// approximation fails, spec §5.6) or its shared-shell occupation pattern changed since the previous step
    /// (E(R) jumps there, Task 6 review A.6). M7.
    UnfitNotKept,
    FenceDisagrees,
    WaitingForPass,
    NotOffered,
    /// A background audit's answer arrived after the model changed settings or species: not
    /// judged, not kept (plan review S11).
    Discarded,
    StoreRefused,
    Reference,
    Failed,
    NotSolved,
}

impl TeachOutcome {
    pub fn is_kept(self) -> bool {
        matches!(self, TeachOutcome::Kept | TeachOutcome::KeptBoxAffected | TeachOutcome::KeptByVote | TeachOutcome::KeptReleased { .. })
    }
}

/// How a teacher's SCF ended (the solve log's iteration cell): converged, the stall stop
/// (`qm::scf` "The stall stop"), the iteration cap, or a non-finite Δρ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfEnd {
    Converged,
    Stalled,
    Cap,
    NonFinite,
}

impl ScfEnd {
    pub fn word(self) -> &'static str {
        match self {
            ScfEnd::Converged => "converged",
            ScfEnd::Stalled => "stalled",
            ScfEnd::Cap => "cap",
            ScfEnd::NonFinite => "not finite",
        }
    }
}

/// One teaching solve (spec §4.3 A3). Energies in Hartree, the face density in e/Bohr³.
#[derive(Clone, Debug, PartialEq)]
pub struct TeachEvent {
    pub at: Instant,
    /// The driver step whose evaluation produced this event, stamped by `sim::Dynamics` right
    /// after that `evaluate` (plan review M6); `None` for a sampling-pass event. A background
    /// audit carries the step it was judged at; `audit_lag` says how many answered steps after
    /// its dispatch that was.
    pub step: Option<u64>,
    pub origin: TeachOrigin,
    pub rule: TeachRule,
    /// The atom the rule names, when it names one.
    pub atom: Option<usize>,
    /// Every note the provider wrote for this solve, one per line (the technical text, shown on
    /// hover; plan review S12).
    pub detail: String,
    pub outcome: TeachOutcome,
    pub face_density: Option<f64>,
    pub box_affected: bool,
    /// E_teacher − E_model where the model was consulted.
    pub gap: Option<f64>,
    pub n_atoms: usize,
    /// Wall time of the teacher's solve (ms); 0 for a row with no solve (`NotSolved`).
    pub solve_ms: f64,
    /// The teacher's SCF steps for this solve; `None` when the solver did not report them (a
    /// failed solve, a reference multiplicity scan, a row with no solve) — never a stand-in 0
    /// (Task 2 review I1: an analytic teacher really reports 0). A failed solve whose error names its
    /// iterations (`SolverError::scf_end`) carries them, with `scf_end`.
    pub scf_steps: Option<usize>,
    /// How the SCF ended, where known (`None`: no SCF, or a provider that does not say).
    pub scf_end: Option<ScfEnd>,
    pub audit: Option<AuditVerdict>,
    /// A background audit: answered steps between its dispatch and its verdict.
    pub audit_lag: Option<usize>,
    /// A hot-burst geometry's instantaneous kinetic temperature (K, `Scene::temperature`, 3N − 3
    /// dof): the Verlet velocities v(t + dt) at the solved geometry, before the CSVR thermostat
    /// rescales them (plan review S6).
    pub burst_t: Option<f64>,
}

impl TeachEvent {
    pub fn new(origin: TeachOrigin, rule: TeachRule, outcome: TeachOutcome) -> TeachEvent {
        TeachEvent {
            at: Instant::now(),
            step: None,
            origin,
            rule,
            atom: None,
            detail: String::new(),
            outcome,
            face_density: None,
            box_affected: false,
            gap: None,
            n_atoms: 0,
            solve_ms: 0.0,
            scf_steps: None,
            scf_end: None,
            audit: None,
            audit_lag: None,
            burst_t: None,
        }
    }

    /// A teacher evaluation happened (a cell-guard skip did not).
    pub fn is_solve(&self) -> bool {
        self.outcome != TeachOutcome::NotSolved
    }

    /// Counted in the model's lifetime solves (spec A7: not a rule-3 solve).
    pub fn counts_for_model(&self) -> bool {
        self.is_solve() && self.rule != TeachRule::OtherSettings
    }
}

/// Lifetime teaching solves of one model (spec §3.3.2, A2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveCounts {
    pub pass: u64,
    pub fallback: u64,
    pub audit: u64,
    pub verification: u64,
    pub kept: u64,
    /// The model came from a file without the record: solves before it are unrecorded.
    pub unrecorded_before: bool,
}

impl SolveCounts {
    pub fn total(&self) -> u64 {
        self.pass + self.fallback + self.audit + self.verification
    }

    pub fn not_kept(&self) -> u64 {
        self.total().saturating_sub(self.kept)
    }

    pub fn count(&mut self, e: &TeachEvent) {
        if !e.counts_for_model() {
            return;
        }
        match e.origin {
            TeachOrigin::Pass => self.pass += 1,
            TeachOrigin::Fallback => self.fallback += 1,
            TeachOrigin::Audit => self.audit += 1,
            TeachOrigin::Verification => self.verification += 1,
        }
        if e.outcome.is_kept() {
            self.kept += 1;
        }
    }
}

/// Teaching events the driver holds between two drains by the worker (plan review M6).
pub const DRIVER_EVENTS: usize = 2048;

/// The driver's teaching events since the worker's last drain (plan review M6). `sim::Dynamics`
/// moves the provider's events here right after each `evaluate`, stamped with the step whose
/// evaluation produced them, so a tick of n steps gives n stamps; a pass event keeps `None`, and
/// a stamp an event already has is kept. Bounded: the oldest is dropped and counted.
#[derive(Debug, Default)]
pub struct TeachBuffer {
    events: std::collections::VecDeque<TeachEvent>,
    dropped: u64,
}

impl TeachBuffer {
    pub fn absorb(&mut self, events: Vec<TeachEvent>, step: u64) {
        for mut e in events {
            if e.origin != TeachOrigin::Pass && e.step.is_none() {
                e.step = Some(step);
            }
            if self.events.len() == DRIVER_EVENTS {
                self.events.pop_front();
                self.dropped += 1;
            }
            self.events.push_back(e);
        }
    }

    pub fn take(&mut self) -> Vec<TeachEvent> {
        self.events.drain(..).collect()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// One condition of the trust rule (spec §3.3.4, A4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrustCheck {
    pub key: TrustKey,
    pub met: bool,
    pub value: f64,
    pub limit: f64,
}

/// A trusted model that keeps falling back (spec A5; `SurrogateKs::stuck` in structured form).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StuckInfo {
    /// The rule of the streak's latest fallback; `None` when the provider stored no reason
    /// (plan review S10: never a guessed rule).
    pub rule: Option<TeachRule>,
    pub atom: Option<usize>,
    pub steps: usize,
    pub box_streak: bool,
    pub gap: Option<f64>,
    /// The largest face density over the trailing box-affected run (e/Bohr³); meaningful only
    /// when `box_streak` (`stuck` prints it only then).
    pub face_density: f64,
}

/// A stored sample's origin by its store tag (audits are `Fallback` until sidecar v4, spec A6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    Pass,
    Fallback,
    Live,
}

/// Whether a stored sample is in the current fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleState {
    Fitted,
    Unfitted,
    Quarantined,
    Superseded,
    BoxExcluded,
    /// A draft-mode audit's answer held out of the strict fit.
    HeldOut,
}

/// One sample of the energy parity plot: (E − Σ E_ref)/N for the teacher and the model (Ha/atom).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParityPoint {
    pub dft: f64,
    pub model: f64,
    pub kind: SampleKind,
    pub state: SampleState,
    pub index: usize,
}

/// A species pair's model curve (spec §3.3.5, §9). Hartree and Bohr.
///
/// **A two-body slice, not a pair potential** (spec §9.4, plan review S3): the model's energy of
/// an *isolated* a–b dimer. The model is a sum of per-atom terms whose descriptors couple every
/// neighbour, so this curve is not the model's prediction for an a–b pair inside a molecule —
/// there the model has no per-pair energy at all. `sigma`, `force_ratio` and `refused` are
/// likewise the isolated dimer's.
#[derive(Clone, Debug, PartialEq)]
pub struct PairCurve {
    pub z: (u8, u8),
    pub r: Vec<f64>,
    /// E_model(dimer at r) − E_ref(a) − E_ref(b), with the fit's own references.
    pub energy: Vec<f64>,
    /// The epistemic energy std √(energy_std² − f²), f = `TrainingViews::dimer_floor`: the band
    /// ±σ — **±1σ model (coefficient) uncertainty, what rule 7 judges; not an error bar against
    /// DFT** (it holds no basis misfit, so the model's error can exceed it; Task 4 review I1).
    /// Rule 7 compares σ/2 with `energy_std_per_atom` (the lane's σ/2 ÷ tier against 1). Exactly
    /// 0 where σ_E² − f² rounds to ≤ 0 (a log lane clamps it). Empty without a posterior.
    pub sigma: Vec<f64>,
    /// Rule 6's ratio per r: max over the dimer's atoms of σ_F,epi/(√3 · bound), the quantity
    /// rule 6 compares with 1 (plan review M5); empty without a posterior.
    pub force_ratio: Vec<f64>,
    /// Per r: `None` when the model would answer this dimer, else the rule that refuses it —
    /// `SurrogateKs::would_answer`'s verdict (rules 1–7 and the fence, a fresh uncertainty check;
    /// rule 0, draft mode, audits and the check's reuse between steps aside), never derived from
    /// `sigma`, asked under the model's own provenance (rule 3 never fires here). The dominant
    /// encoding of the plot (plan review M5).
    pub refused: Vec<Option<TeachRule>>,
    /// Stored two-atom samples of this pair: (r, E − E_ref(a) − E_ref(b), state).
    pub dots: Vec<(f64, f64, SampleState)>,
    pub r_cut: f64,
}

/// Pair distances < r_cut over the fitted samples, bins of `bin` Bohr from 0.
#[derive(Clone, Debug, PartialEq)]
pub struct PairHistogram {
    pub z: (u8, u8),
    pub bin: f64,
    pub counts: Vec<u32>,
}

/// The training views' data (spec §3.3.5, A6).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingViews {
    /// The surface epoch (`ForceProvider::surface_epoch`): it does not move on a refit that
    /// leaves the model untrusted — key staleness on `model_gen` (Task 4 review I2).
    pub epoch: u64,
    /// The model generation the views describe (`SurrogateKs::model_generation`): changes at
    /// every installed, restored or forgotten model and every policy change.
    pub model_gen: u64,
    pub samples: usize,
    pub parity_energy: Vec<ParityPoint>,
    /// (teacher, model) per force component of fitted samples (Ha/Bohr), thinned to ≤ 20 000.
    pub parity_force: Vec<[f32; 2]>,
    pub force_points_total: usize,
    /// Stored samples the parity plot omits (a species without a model reference, a failed
    /// prediction, or a label or prediction that is not finite; plan review S18, review m5).
    pub parity_omitted: usize,
    pub curves: Vec<PairCurve>,
    /// Species pairs whose curve was skipped because a prediction on its grid failed.
    pub curves_skipped: usize,
    pub pairs: Vec<PairHistogram>,
    /// The floor `predict` adds in quadrature to a dimer's energy std (Ha): 2·s_E for a model
    /// fitted per atom, s_E for one fitted per sample. **The fit's residual scatter: teacher
    /// noise plus what this basis cannot fit** — more samples of the same kind do not remove it
    /// (Task 4 review I1, m6; not "the teacher's scatter").
    pub dimer_floor: f64,
    /// The rule-7 per-atom tier (Ha/atom).
    pub energy_tier: f64,
    /// Wall time of this build (ms), the curve cache's rebuild included when the model
    /// generation moved: the worker's backstop interval reads it (max(2 s, 50 × build_ms), Task 5).
    pub build_ms: f64,
}

/// Max force-parity points (spec §3.3.5).
pub const FORCE_PARITY_CAP: usize = 20_000;
/// Points per pair curve and its inner end (Bohr).
pub const CURVE_POINTS: usize = 96;
pub const CURVE_R_MIN: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_counts_scf_steps_and_clears_at_the_end() {
        let p = ActivityProbe::new();
        assert!(!p.view().main.running);
        {
            let _b = p.busy(Slot::Main);
            p.scf_step(Slot::Main);
            p.scf_step(Slot::Main);
            let v = p.view();
            assert!(v.main.running && !v.side.running);
            assert_eq!(v.main.scf_step, 2);
        }
        let v = p.view();
        assert!(!v.main.running);
        assert_eq!(v.main.solves, 1);
        assert_eq!(v.main.scf_step, 2, "the last solve's count stays readable");
    }

    /// The per-iteration record: ΔE between iterations, the countdown, a new attempt when the
    /// iteration number restarts, the sparkline decimated to span the whole attempt, and a new
    /// solve clears it.
    #[test]
    fn the_probe_keeps_the_running_solves_iterations() {
        let p = ActivityProbe::new();
        let it = |i: usize, d: f64, e: f64, k: usize| ScfIteration { iteration: i, max_iter: 200, delta_rho: d, tol: 1e-6, energy: e, since_gain: k, window: 50, best_delta_rho: d, stall_level: 2e-3 };
        let b = p.busy(Slot::Main);
        assert_eq!(p.view().main.live, None);
        p.scf_iteration(Slot::Main, &it(1, 1e-1, -17.0, 0));
        p.scf_iteration(Slot::Main, &it(2, 1e-2, -17.25, 0));
        let l = p.view().main.live.unwrap();
        assert_eq!((l.attempt, l.iteration, l.max_iter, l.since_gain, l.window), (1, 2, 200, 0, 50));
        assert!((l.d_energy + 0.25).abs() < 1e-12 && l.ms_per_iteration >= 0.0);
        assert_eq!(l.spark(), &[-1.0, -2.0]);
        assert_eq!(l.stall_left(), Some(50));
        // 200 iterations: the sparkline keeps every 4th (stride 2 at iteration 65, 4 at 130)
        for i in 3..=200 {
            p.scf_iteration(Slot::Main, &it(i, 10f64.powf(-(i as f64) / 100.0), -17.0, 31));
        }
        let l = p.view().main.live.unwrap();
        assert_eq!((l.spark_stride, l.spark_len), (4, 50));
        assert!((l.spark()[0] - (-0.04)).abs() < 1e-6 && (l.spark()[49] - (-2.0)).abs() < 1e-6, "{:?}", l.spark());
        assert_eq!(l.stall_left(), Some(19), "best Δρ 1e-2 is above the level 2e-3: the countdown runs");
        // below the level the stop does not apply: no countdown (ruling (a))
        p.scf_iteration(Slot::Main, &it(201, 1e-4, -17.0, 40));
        assert_eq!(p.view().main.live.unwrap().stall_left(), None);
        // the cold retry: iterations from 1 again
        p.scf_iteration(Slot::Main, &it(1, 0.3, -16.9, 0));
        let l = p.view().main.live.unwrap();
        assert_eq!((l.attempt, l.iteration, l.spark_len), (2, 1, 1));
        assert!(l.d_energy.is_nan(), "no previous iteration in this attempt");
        drop(b);
        assert!(p.view().main.live.is_some(), "the last solve's record stays readable");
        let _c = p.busy(Slot::Main);
        assert_eq!(p.view().main.live, None, "a new solve starts empty");
    }

    #[test]
    fn a_new_solve_restarts_the_step_count() {
        let p = ActivityProbe::new();
        drop(p.busy(Slot::Side));
        p.scf_step(Slot::Side);
        let _b = p.busy(Slot::Side);
        assert_eq!(p.view().side.scf_step, 0);
    }

    #[test]
    fn the_refit_phase_is_reported_with_its_sample_count() {
        let p = ActivityProbe::new();
        assert_eq!((p.view().refit, p.view().refits), (None, 0));
        p.begin_refit(250);
        assert!(matches!(p.view().refit, Some((t, 250)) if t >= 0.0));
        assert_eq!(p.view().refits, 0, "counted when it ends");
        p.end_refit();
        assert_eq!((p.view().refit, p.view().refits), (None, 1));
    }

    #[test]
    fn solve_counts_total_and_not_kept() {
        let c = SolveCounts { pass: 273, fallback: 70, audit: 18, verification: 1, kept: 343, unrecorded_before: false };
        assert_eq!(c.total(), 362);
        assert_eq!(c.not_kept(), 19);
    }

    #[test]
    fn kept_outcomes_and_counted_events() {
        assert!(TeachOutcome::KeptReleased { released: 1, superseded: 0 }.is_kept());
        assert!(!TeachOutcome::Thinned.is_kept() && !TeachOutcome::Reference.is_kept());
        let mut e = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Grade, TeachOutcome::Kept);
        assert!(e.is_solve() && e.counts_for_model());
        e.rule = TeachRule::OtherSettings;
        assert!(e.is_solve() && !e.counts_for_model(), "spec A7: a rule-3 solve is not spent on the model");
        e.outcome = TeachOutcome::NotSolved;
        assert!(!e.is_solve());
    }

    /// Plan review M6: the driver stamps an event with the step whose evaluation produced it,
    /// leaves pass events unstamped (the log shows "pass"), and never overwrites a stamp.
    #[test]
    fn the_driver_stamps_each_event_with_its_step_and_leaves_pass_events_unstamped() {
        let ev = |o| TeachEvent::new(o, TeachRule::Grade, TeachOutcome::Kept);
        let mut b = TeachBuffer::default();
        b.absorb(vec![ev(TeachOrigin::Fallback)], 7);
        b.absorb(vec![ev(TeachOrigin::Pass), ev(TeachOrigin::Audit)], 8);
        let mut pre = ev(TeachOrigin::Fallback);
        pre.step = Some(3);
        b.absorb(vec![pre], 9);
        let steps: Vec<Option<u64>> = b.take().into_iter().map(|e| e.step).collect();
        assert_eq!(steps, vec![Some(7), None, Some(8), Some(3)]);
        assert!(b.take().is_empty(), "a take drains the buffer");
    }

    /// The driver's buffer is bounded like the provider's queue: the oldest go first and are
    /// counted.
    #[test]
    fn the_driver_buffer_drops_the_oldest_and_counts_them() {
        let mut b = TeachBuffer::default();
        let events = (0..DRIVER_EVENTS + 5)
            .map(|i| {
                let mut e = TeachEvent::new(TeachOrigin::Fallback, TeachRule::Grade, TeachOutcome::Kept);
                e.n_atoms = i;
                e
            })
            .collect();
        b.absorb(events, 1);
        assert_eq!(b.dropped(), 5);
        let e = b.take();
        assert_eq!(e.len(), DRIVER_EVENTS);
        assert_eq!(e[0].n_atoms, 5, "the oldest went first");
    }

    /// Plan review S11: a background audit discarded because the model changed is a solve spent
    /// on the model (counted), not a kept sample.
    #[test]
    fn a_discarded_audit_is_counted_and_not_kept() {
        let e = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Discarded);
        assert!(e.is_solve() && e.counts_for_model() && !e.outcome.is_kept());
        let mut c = SolveCounts::default();
        c.count(&e);
        assert_eq!((c.audit, c.kept, c.not_kept()), (1, 0, 1));
    }
}
