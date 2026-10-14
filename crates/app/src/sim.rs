//! Simulation worker: owns a [`sim::Dynamics`] (the M4 driver) off the render thread, feeds it the
//! viewer's commands and publishes one immutable [`Snapshot`] per tick.
//!
//! The render thread talks to the worker through two unbounded channels: [`Command`]s in,
//! [`Event`]s out. The loop is: take the queued commands (blocking for the first one only when
//! there is nothing to do — paused, no queued steps, forces fresh), apply each to the driver,
//! `tick` once (≈ 16 ms of stepping when playing; the queued `Step`s when paused; the stale-force
//! re-evaluation after an edit), publish the tick's snapshot. A paused burst is collected for
//! [`COALESCE_WINDOW`] after its first command, so dragging a nucleus while paused costs one
//! evaluation per burst, as in M2; consecutive drag moves (and consecutive teleports of the same
//! nucleus) collapse to the last one.
//!
//! # Backends
//! [`Backend::Gpu`] gives the driver a `sim::GpuKs` on a clone of the viewer's `GpuContext` (one
//! device and one queue, submitted to from two threads — legal in wgpu); its density arrives as
//! `sim::Display::Gpu`, a texture from the provider's pool that the viewer binds as is.
//! [`Backend::Cpu`] gives it `sim::CpuKs` (`qm::scf`, the f64 reference); its density arrives as
//! values the viewer uploads. [`Backend::Estimated`] (M5) gives it a `sim::SurrogateKs` with the
//! default `FallbackPolicy` around a teacher — `GpuKs` when a context exists, else `CpuKs` — at
//! the sample tolerance [`SAMPLE_TOL`] = 1e-6 rather than the DFT backends' interactive 1e-5
//! (ruling S18: its answers are permanent training labels); a surrogate step has no density (`display: None`, the viewer
//! keeps the last DFT cloud, faded and labelled). A backend switch spawns a new worker
//! (`Viewer::start`).
//!
//! # The estimated solver (M5)
//! - **Teaching.** [`Command::EnsureModel`] (sent by the viewer right after spawning an estimated
//!   worker) installs the session's autosaved model if there is one, and otherwise starts a
//!   sampling pass (`DynCommand::StartSampling`) when the surrogate has no model — the panel's
//!   plan if it fits [`PASS_TARGET_S`] at the expected per-solve cost, else that plan scaled down
//!   (never below what trust needs), else nothing and a [`PassOffer`] for the panel's "Start
//!   anyway" (Task 9: [`first_pass`]). While a pass runs, [`ModelStatus::pass_eta_s`] is its time
//!   left at its measured rate. A refused autoload (another provenance) starts nothing: the
//!   panel offers to forget it. Playing while no model exists (an offer waiting, a refused
//!   autoload) runs DFT steps whose answers are kept without a refit — up to 4 Q′ of them, then
//!   one in ten with a "waiting for a sampling pass" note (`sim::surrogate_ks`, "No model yet";
//!   final review B2).
//! - **Audits.** With the GPU teacher the surrogate gets a second `GpuKs` on the same context as
//!   its background auditor (ruling S14, `SurrogateKs::with_auditor`): an audit no longer stops
//!   the stepping. The CPU teacher audits in the step.
//!   While a pass runs the dynamics stay paused and the worker keeps ticking (each tick spends its
//!   16 ms budget on the pass); edits are accepted.
//! - **Refits.** After every tick in which the surrogate kept a sample (`pending_refit`: a
//!   fallback step — active learning), a model exists (the first fit is the sampling pass's,
//!   which also solves the reference energies a fit needs) and no pass is running, the worker
//!   refits it, between ticks as `sim::surrogate_ks` requires — while playing at most once per
//!   [`REFIT_INTERVAL`] unless [`REFIT_SAMPLES`] new samples arrived first (a fallback burst does
//!   not refit every step); paused, at once. The driver notices the new surface epoch and books
//!   the jump on its next tick. Every outcome goes to the snapshot notes ("refit #k: rmse …,
//!   trusted: yes/no — reason", the recovery's quarantine note, or "refit failed: …") and to
//!   [`ModelStatus::last_refit`]; a failed refit is retried at the next opportunity, never fatal.
//!   An untrusted refit triggers the surrogate's trust recovery (ruling S15: a degree-cap retry,
//!   then quarantine of the samples the model cannot represent — `sim::surrogate_ks` module doc).
//! - **The cloud** (M5b, "Density in estimated mode"). A model step has no electrons, so the
//!   background auditor's real densities are published: every audit's (its teacher has display
//!   on), and — [`Command::SetCloud`]'s `refresh_s` — a display-only solve at the current
//!   geometry every N s while playing, asked of the idle auditor between ticks, never on the step
//!   path (`SurrogateKs::request_display`). They reach the snapshot as
//!   `DynSnapshot::side_display`, stamped with their geometry's time. With `superposition` on
//!   (the default) the worker also asks the idle auditor for each species' isolated-atom density
//!   on the scene's cell — once per species and grid — and publishes their radial profiles
//!   ([`Snapshot::atom_profiles`], `crate::cloud`): the viewer's atomic superposition between
//!   real densities. With the CPU teacher there is no auditor: neither exists (its audits run in
//!   the step and show their own density).
//! - **Readouts.** Every snapshot carries a [`ModelStatus`] read off the surrogate (trust and the
//!   reason when untrusted, samples, epoch, last fit, the tightest force bound at the current
//!   forces, a provenance mismatch).
//! - **Autosave.** An estimated worker spawned with an `autosave` path writes its model there
//!   when it shuts down (or removes the file when it has none) and then sends [`Event::Retired`]:
//!   leaving the estimated solver and coming back, or a panic recovery, keeps the model. It does
//!   so **only once it owns the file** (fix round 1, I1): after it applied `EnsureModel` and the
//!   autoload succeeded or there was nothing to load, or after the user started a pass, loaded a
//!   model or forgot it. A worker shut down before its `EnsureModel` ran (a quick radio toggle
//!   during the cold first evaluation), or whose autoload was refused (another provenance), leaves
//!   the file as it found it — so a timed-out, detached old worker can never delete a model the
//!   session still has.
//!
//! # The empty scene (the `empty` preset)
//! A scene with no nuclei has nothing to solve: the driver answers it without its provider
//! (`sim::dynamics`, "The empty scene") — E = 0, no forces, no density — on every backend. Play is
//! accepted but time does not advance, and the worker does not tick while it waits ([`has_work`]
//! is false for an empty scene that is merely playing: no busy loop, no snapshot flood); placing
//! the first atom is an ordinary edit, solved at once (and, while playing, the run starts). In
//! estimated mode `EnsureModel` on an empty scene still restores the session's autosaved model,
//! but starts no sampling pass and makes no offer (no species, no model needed; the Model tab
//! says "place atoms to sample", and a pass sent anyway is refused with that reason); the
//! surrogate is never asked about the empty scene, so there is no fallback, sample or audit.
//! Once atoms are placed, **Start sampling** teaches the model as on any other scene (the first
//! pass is not started automatically then: one placed atom is rarely the scene to sample).
//!
//! # What aborts a running step (ruling R11)
//! The driver polls `should_abort` before every step and the GPU provider before every SCF
//! step (latency ≤ one SCF step; the CPU provider polls once, before its solve). The worker
//! raises it **only** while a command that makes the running step pointless is queued
//! ([`Command::aborts`]): Pause, the scene edits (teleport, add, remove, a replaced scene,
//! charge / multiplicity / grid / σ — those become `ReplaceScene`), the model commands that end or
//! replace what a sampling solve is for (StartSampling, CancelSampling, ForgetModel, LoadModel,
//! EnsureModel; not SaveModel, a file write) and Shutdown. A Pause during a pass abandons the
//! sampling solve in flight; the job stays queued and the pass resumes on the next tick. Drag moves,
//! the drag pull (the wheel), speed, pins, rewind, thermostat, field, dt and the COM toggle never abort: they only change
//! the state the next step starts from, and a drag move arrives every frame — aborting on it
//! would cancel every DFT step while the user drags. The construction of the driver
//! (`Dynamics::new`'s first, cold evaluation) is not cancellable: a backend switch or shutdown
//! issued during it waits for that one solve.
//!
//! # Pacing (fix round 1, user request)
//! A playing tick that ran all its `speed` steps in less than [`TICK_BUDGET_MS`] waits out the
//! rest of the budget (on the inbox: a command still wakes the worker at once), so "steps per
//! tick" means steps per ≈ 16 ms of wall and a low speed is a slow, watchable run — before, a
//! surrogate tick of a few steps was followed at once by the next, and even speed 1 ran at
//! thousands of fs/s. A DFT step fills its tick by itself and is never paced. The top of the
//! slider, `MAX_SPEED`, stays unpaced ("as fast as possible"), so the surrogate's full rate is
//! still available. A command arriving before the deadline (a DragMove per mouse event, a slider
//! drag) is applied and published at once **without a tick** (fix round 2): only the deadline
//! starts the next tick, so a command flood neither speeds the run up nor pushes the schedule
//! ahead of the clock (which used to freeze the run for seconds after a drag ended); each
//! deadline is also capped at one budget after its tick started.
//!
//! # Errors
//! Provider failures (coincident nuclei, no SCF convergence, an impossible charge) pause the
//! run and arrive in `DynSnapshot::last_error`; refused commands in `last_rejection`. Worker
//! errors (an element without an isotope mass, a panic) arrive as [`Event::Error`]; after a
//! panic the driver is rebuilt from the last snapshot (time restarts, the settings are kept).
use crate::scene::Scene;
use gpu::GpuContext;
use sim::{CpuKs, DynCommand, DynSnapshot, Dynamics, FallbackPolicy, ForceProvider, GpuKs, SamplingPlan, SurrogateKs};
use crate::cloud::AtomProfile;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The time step a new run starts with (spec §7: resolves the ~10 fs H–X stretch; the driver
/// warns above `sim::dynamics::DT_WARN_FS`).
pub const DEFAULT_DT_FS: f64 = 0.25;

/// Wall budget of one playing tick (ms): one 60 Hz frame. A step that costs more (any DFT step
/// at 49³) runs alone in its tick; the snapshot rate is then the step rate.
const TICK_BUDGET_MS: f64 = 16.0;

/// Seeds every thermostat the driver creates (one seed reproduces a command sequence).
const SEED: u64 = 0x5EED_0A70_B1C5_A4D0;

/// Interactive SCF tolerance on ∫|Δρ| (both backends): looser than the oracles' 1e-7 (and
/// `ScfSettings::for_grid`'s 1e-6). CPU (9800X3D): a 49³ H₂ solve takes 5–7 s. GPU (RX 7900 XTX,
/// re-measured in M3 Task 9 by `diag_interactive_tol`): the wall time is flat in tol (the late
/// SCF steps are cheap) — H₂ 49³ cold 0.5 s / warm after a 0.1 Bohr nudge 0.41–0.47 s, H₂O 57³
/// cold 8.1–8.2 s / warm 3.9–4.1 s for tol 1e-5 … 1e-6 — and at every tol the warm density and
/// energy sit at the GPU's own f32 path scatter against a tol-1e-7 reference (max|Δρ|/ρmax
/// 3.5e-6 for H₂, 4e-7 for H₂O; |ΔE| ≤ 3e-6 Ha). Nothing is gained by tightening, so 1e-5
/// stays (fewest SCF steps, i.e. the least GPU queue time competing with the render thread).
/// Interactive snapshots are for looking at, not for reference numbers; the dynamics validation
/// suite (M4 Task 7) sets its own tolerance.
const INTERACTIVE_TOL: f64 = 1e-5;

/// The estimated solver's own teacher — its sampling pass, its fallback steps and its background
/// auditor — solves at this tolerance (ruling S18; spec §4, review MUST C2(b)): its answers are
/// training labels, kept for good, and at the interactive 1e-5 the SCF lag (~2e-4 Ha/Bohr in the
/// forces, the spec's estimate) would set the evidence's noise floor instead of the grid. The
/// auditor is included because an audit's answer is admitted as a training sample too (the
/// audit *gap* alone would tolerate 1e-5: the energy lag there is ≤ 3e-6 Ha against a
/// threshold of N × 1e-3 Ha). Cost: none measurable per solve on the GPU (the wall time is flat
/// in tol, above); the H₂ 49³ pass, `references.md` "Surrogate (M5)". The provenance records it,
/// so a model taught at 1e-5 is refused with both tolerances named.
const SAMPLE_TOL: f64 = 1e-6;

/// After the first command of a paused burst, keep collecting commands for this long before
/// ticking. A sender pushing a burst races the worker's wake-up; without the window the worker
/// could evaluate for the first command of the burst alone and only then for the rest.
/// Negligible next to a solve (≥ tens of ms) and shorter than a 60 Hz frame.
const COALESCE_WINDOW: Duration = Duration::from_millis(5);

#[derive(Clone, Debug)]
pub enum Command {
    /// Replaces everything (preset, file load); the driver pauses, time restarts, the external
    /// field is kept.
    SetScene(Scene),
    /// Teleport (paused only; while playing, drag with `Dyn(DragStart/Move/End)`).
    MoveNucleus {
        index: usize,
        pos: [f64; 3],
    },
    /// Placed at rest; allowed while playing.
    AddNucleus {
        z: u8,
        pos: [f64; 3],
    },
    /// Allowed while playing.
    RemoveNucleus(usize),
    /// These four pause the run and replace the scene with the driver's current one (positions
    /// and velocities kept) with the setting changed: time, pins and drag restart.
    SetCharge(i32),
    SetMultiplicity(Option<u32>),
    SetGrid {
        n: usize,
        h: f64,
    },
    SetSigma(Option<f64>),
    /// Time control, thermostat, field, COM correction, pins and drag springs, passed to the
    /// driver as is. Scene edits go through the variants above (the viewer mirrors those);
    /// `Dyn(ReplaceScene | MoveNucleusDiscontinuous | AddNucleus | RemoveNucleus)` is never sent.
    Dyn(DynCommand),
    /// Estimated solver (module doc): install the model at `autoload` if the file exists — a
    /// refused load starts nothing (the panel offers to forget it) — else, when the surrogate has
    /// no model and no pass runs, start a sampling pass with `plan`. Ignored by the other
    /// backends.
    EnsureModel {
        autoload: Option<PathBuf>,
        plan: SamplingPlan,
    },
    /// Estimated solver: the surrogate's audit cadence (`FallbackPolicy::audit_every`, ruling
    /// S7: a DFT check every n answered steps; 2000 by default, ruling S14's interim; 0 = never).
    /// The rest of the current policy is kept. Ignored by the other backends.
    SetAudit(usize),
    /// Estimated solver: draft mode (`FallbackPolicy::never_fallback`, `sim::surrogate_ks`,
    /// "Draft mode"): the model answers every step it can be asked at all, extrapolating; the
    /// rest of the policy is kept. Per session, off by default. Ignored by the other backends.
    SetDraft(bool),
    /// Estimated solver, the cloud (module doc): fetch the isolated-atom profiles for the
    /// viewer's atomic superposition, and ask for a display-only density every `refresh_s`
    /// seconds while playing (`None`: never). Ignored by the other backends.
    SetCloud {
        superposition: bool,
        refresh_s: Option<f64>,
    },
    Shutdown,
}

impl Command {
    /// Applies the command to the viewer's mirror of the scene: teleports, additions, removals
    /// and settings show at once; the driver's commands (time control, drag springs, pins…)
    /// leave the mirror alone — in particular a drag never jumps the nucleus to the cursor: the
    /// spring moves it, and the mirror follows the snapshots.
    pub fn apply(&self, s: &mut Scene) {
        match self {
            Command::SetScene(new) => *s = new.clone(),
            Command::MoveNucleus { index, pos } => {
                if let Some(n) = s.nuclei.get_mut(*index) {
                    n.pos = *pos;
                    n.vel = [0.0; 3];
                }
            }
            Command::AddNucleus { z, pos } => s.nuclei.push(crate::scene::SceneNucleus {
                z: *z,
                pos: *pos,
                vel: [0.0; 3],
            }),
            Command::RemoveNucleus(i) => {
                if *i < s.nuclei.len() {
                    s.nuclei.remove(*i);
                }
            }
            Command::SetCharge(c) => s.charge = *c,
            Command::SetMultiplicity(m) => s.multiplicity = *m,
            Command::SetGrid { n, h } => {
                s.grid_n = *n;
                s.grid_h = *h;
            }
            Command::SetSigma(x) => s.sigma = *x,
            Command::Dyn(_) | Command::EnsureModel { .. } | Command::SetAudit(_) | Command::SetDraft(_) | Command::SetCloud { .. } | Command::Shutdown => {}
        }
    }

    /// Commands the viewer mirrors (scene edits, and the play state: Play / Pause / Step and the
    /// pausing model commands, StartSampling and EnsureModel). The
    /// worker counts them in `Snapshot::edits`; the viewer adopts a snapshot's scene and play
    /// state only once every such command it sent is in it, so a snapshot taken before an edit
    /// never undoes the edit on screen.
    pub fn is_edit(&self) -> bool {
        match self {
            Command::Dyn(c) => matches!(
                c,
                DynCommand::Play | DynCommand::Pause | DynCommand::Step(_) | DynCommand::StartSampling(_)
            ),
            Command::Shutdown | Command::SetAudit(_) | Command::SetDraft(_) | Command::SetCloud { .. } => false,
            _ => true,
        }
    }

    /// Whether the driver ends up paused by it (Pause, Step, the scene replacements, and a
    /// sampling pass starting).
    pub fn pauses(&self) -> bool {
        matches!(
            self,
            Command::SetScene(_)
                | Command::SetCharge(_)
                | Command::SetMultiplicity(_)
                | Command::SetGrid { .. }
                | Command::SetSigma(_)
                | Command::EnsureModel { .. }
                | Command::Dyn(DynCommand::Pause | DynCommand::Step(_) | DynCommand::StartSampling(_))
        )
    }

    /// Whether the command is handed to the driver (which may refuse it: `last_rejection`); the
    /// worker's own settings (audit cadence, draft mode, the cloud, the model's first teaching) and
    /// Shutdown are not.
    fn reaches_driver(&self) -> bool {
        !matches!(
            self,
            Command::EnsureModel { .. } | Command::SetAudit(_) | Command::SetDraft(_) | Command::SetCloud { .. } | Command::Shutdown
        )
    }

    /// Whether a queued copy of this command abandons the step (and solve) in flight (module
    /// doc, R11).
    pub fn aborts(&self) -> bool {
        match self {
            Command::Dyn(c) => matches!(
                c,
                DynCommand::Pause
                    | DynCommand::ReplaceScene(_)
                    | DynCommand::MoveNucleusDiscontinuous { .. }
                    | DynCommand::AddNucleus(_)
                    | DynCommand::RemoveNucleus(_)
                    | DynCommand::StartSampling(_)
                    | DynCommand::CancelSampling
                    | DynCommand::ForgetModel
                    | DynCommand::LoadModel(_)
            ),
            // a policy change applies from the next evaluation on, the cloud is display only:
            // nothing to abandon
            Command::SetAudit(_) | Command::SetDraft(_) | Command::SetCloud { .. } => false,
            _ => true,
        }
    }

    /// The next command makes this one redundant: consecutive drag moves, consecutive teleports
    /// of the same nucleus, consecutive drag pulls (a touchpad's wheel events) — the next one
    /// valid, so a refusal never drops an accepted change; the bookings at fixed positions add
    /// up to the last one's, U(F₂) − U(F₁) + U(F₁) − U(F₀) = U(F₂) − U(F₀).
    fn superseded_by(&self, next: Option<&Command>) -> bool {
        match (self, next) {
            (
                Command::Dyn(DynCommand::DragMove { .. }),
                Some(Command::Dyn(DynCommand::DragMove { .. })),
            ) => true,
            (
                Command::Dyn(DynCommand::SetDragForce { .. }),
                Some(Command::Dyn(DynCommand::SetDragForce { f_max })),
            ) => (sim::dynamics::F_DRAG_LO..=sim::dynamics::F_DRAG_HI).contains(f_max),
            (
                Command::MoveNucleus { index: a, .. },
                Some(Command::MoveNucleus { index: b, .. }),
            ) => a == b,
            _ => false,
        }
    }
}

/// Which solver the worker's driver runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `sim::CpuKs`: `qm::scf` on the CPU (f64; the reference).
    Cpu,
    /// `sim::GpuKs`: `qm_gpu::GpuSolver` on the viewer's device (f32 grid quantities).
    Gpu,
    /// `sim::SurrogateKs` around a Kohn-Sham teacher: `GpuKs` when a context exists, else
    /// `CpuKs` (module doc, "The estimated solver").
    Estimated,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Gpu => "GPU",
            Backend::Estimated => "estimated",
        }
    }
}

/// The last refit the worker ran (module doc, "Refits").
#[derive(Clone, Debug, PartialEq)]
pub struct RefitLog {
    /// Wall time of the refit (ms).
    pub ms: f64,
    /// The driver's step count when it ran.
    pub at_step: u64,
    /// `Ok(samples fitted)` or the fit error.
    pub result: Result<usize, String>,
}

/// The surrogate's state as the panel shows it (module doc, "Readouts"). Atomic units.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStatus {
    /// The teacher's name (`ForceProvider::name`).
    pub teacher: &'static str,
    /// A model has been fitted (it may still be untrusted).
    pub fitted: bool,
    /// The trust rule passes (`SurrogateKs::trusted`).
    pub trusted: bool,
    /// Why not (the first failing condition of the trust rule), when fitted and untrusted.
    pub untrusted_reason: Option<String>,
    /// Training samples held, and whether some arrived since the last refit.
    pub samples: usize,
    pub pending_refit: bool,
    /// The surrogate's surface epoch (bumped when a refit moves the surface).
    pub epoch: u64,
    /// The last fit: samples, Q′, rmse E (Ha/atom), rmse F (Ha/Bohr per component).
    pub fit: Option<(usize, usize, f64, f64)>,
    /// The worst training residuals (Ha/atom; Ha/Bohr per component, and that atom's bound).
    pub residuals: Option<(f64, f64, f64)>,
    /// The tightest per-atom force-uncertainty bound at the current forces (Ha/Bohr per
    /// component): `FallbackPolicy::force_bound(min_j |F_j|)` — the surrogate answers only if
    /// every atom's uncertainty is inside its own bound, so this is the strictest one.
    pub force_bound: f64,
    /// The model's provenance differs from the scene's: every step falls back (rule 3).
    pub provenance_mismatch: Option<String>,
    /// The worker's last refit.
    pub last_refit: Option<RefitLog>,
    /// Refits the worker has run (the sampling pass's own refits not counted).
    pub refits: u64,
    /// The autoload of [`Command::EnsureModel`] was refused (the driver's message).
    pub autoload_refused: Option<String>,
    /// The policy's audit cadence ([`Command::SetAudit`]).
    pub audit_every: usize,
    /// Draft mode is on ([`Command::SetDraft`]).
    pub draft: bool,
    /// The step the last audit's geometry was dispatched at: the snapshot step its verdict
    /// arrived with, less its audit lag (`AuditRecord::unaudited_steps`).
    pub last_audit_step: Option<u64>,
    /// The most recent teacher gap (E_DFT − E_model on a fallback booked on the model's
    /// surface, Ha) and the step it was seen at — latched, unlike `DynSnapshot::teacher_gap`.
    pub last_gap: Option<(f64, u64)>,
    /// Samples held in quarantine (ruling S15) and the last refit's recovery note.
    pub quarantined: usize,
    pub recovery: Option<String>,
    /// The audits run on a background teacher (ruling S14), one is in flight, and the last one's
    /// outcome.
    pub audits_background: bool,
    pub audit_in_flight: bool,
    pub last_audit: Option<sim::surrogate_ks::AuditRecord>,
    /// A trusted model that keeps falling back where the molecule is: why
    /// (`SurrogateKs::stuck`).
    pub stuck: Option<String>,
    /// The expected teacher cost of one pass solve on this scene (s): the running pass's (or
    /// the last pass's) measured rate, else [`PASS_SOLVE_PER_COLD`] × the last teacher
    /// evaluation's wall time. The panel's ETA for a pass is solves × this.
    pub solve_s: Option<f64>,
    /// The running pass's remaining time (s) at its measured rate (after ≥ 2 solves).
    pub pass_eta_s: Option<f64>,
    /// The automatic first pass is waiting for "Start anyway" ([`PassOffer`]).
    pub pass_offer: Option<PassOffer>,
}

/// One tick of the driver. Immutable once published; the density (`dynamics.display`) and the
/// history are shared, not copied, by consumers.
pub struct Snapshot {
    pub dynamics: DynSnapshot,
    /// `dynamics.scene` in the file format (what the viewer's mirror and the panel use).
    pub scene: Scene,
    pub backend: Backend,
    /// The Kohn-Sham engine behind the run: `backend` itself, or the estimated solver's teacher
    /// (`Cpu` or `Gpu`).
    pub engine: Backend,
    /// The surrogate's state (estimated solver only).
    pub model: Option<ModelStatus>,
    /// The isolated-atom radial profiles for the viewer's atomic superposition (estimated solver
    /// with a background auditor; module doc, "The cloud"), for the scene's grid and σ.
    pub atom_profiles: Arc<Vec<AtomProfile>>,
    /// Mirrored commands ([`Command::is_edit`]) applied before this snapshot.
    pub edits: u64,
    /// Every command handed to the driver before this snapshot (edits or not; superseded drag
    /// moves skipped): a repeated refusal of the same command (`DynSnapshot::last_rejection`
    /// unchanged) shows here as a new count, so the viewer can give it fresh feedback.
    pub driver_commands: u64,
    /// The worker has more to do without a new command: playing, queued steps, or a stale-force
    /// evaluation to run. Equal to the latest `Event::Busy`.
    pub working: bool,
}

// One snapshot per tick: boxed so the channel's other events stay small.
pub enum Event {
    Snapshot(Box<Snapshot>),
    /// A worker-level failure (not a provider failure: those are in `DynSnapshot::last_error`).
    Error(String),
    /// Playing or solving (true) / idle until the next command (false).
    Busy(bool),
    /// The worker has shut down and written its autosave (module doc, "Autosave"): its last
    /// event.
    Retired,
}

pub struct SimWorker {
    tx: Sender<Command>,
    rx: Receiver<Event>,
    handle: Option<JoinHandle<()>>,
}

impl SimWorker {
    /// Spawns the worker thread; it builds the driver on `initial` (paused, dt =
    /// [`DEFAULT_DT_FS`]) — one cold evaluation — and publishes the first snapshot. `Backend::Gpu`
    /// needs `gpu`, the context whose device the viewer renders with (the density textures are
    /// created on it); `Backend::Cpu` ignores it.
    #[cfg(test)]
    pub fn spawn(initial: Scene, backend: Backend, gpu: Option<GpuContext>) -> SimWorker {
        SimWorker::spawn_with(initial, backend, gpu, Vec::new(), None)
    }

    /// [`SimWorker::spawn`], with `settings` (run settings: dt, speed, thermostat, field, COM
    /// correction) applied to the driver before its first snapshot is published, and (estimated
    /// solver) the `autosave` path its model is written to when it shuts down (module doc).
    /// `Backend::Estimated` without a context teaches with the CPU solver.
    pub fn spawn_with(
        initial: Scene,
        backend: Backend,
        gpu: Option<GpuContext>,
        settings: Vec<DynCommand>,
        autosave: Option<PathBuf>,
    ) -> SimWorker {
        assert!(
            backend != Backend::Gpu || gpu.is_some(),
            "Backend::Gpu needs a GpuContext"
        );
        let (ctx, crx) = channel::<Command>();
        let (etx, erx) = channel::<Event>();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, settings, crx, etx, backend, gpu, autosave, None))
            .expect("spawn sim thread");
        SimWorker {
            tx: ctx,
            rx: erx,
            handle: Some(handle),
        }
    }

    /// A worker on a provider of the test's choosing (labelled `Backend::Cpu`): the pacing and
    /// command-flood tests use an analytic one, so they need no GPU and measure no solver.
    #[cfg(test)]
    pub fn spawn_test(initial: Scene, make: MakeProvider) -> SimWorker {
        let (ctx, crx) = channel::<Command>();
        let (etx, erx) = channel::<Event>();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, Vec::new(), crx, etx, Backend::Cpu, None, None, Some(make)))
            .expect("spawn sim thread");
        SimWorker {
            tx: ctx,
            rx: erx,
            handle: Some(handle),
        }
    }

    pub fn send(&self, cmd: Command) {
        let _ = self.tx.send(cmd);
    }

    /// Non-blocking, for the render loop.
    pub fn try_recv(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    pub fn recv_timeout(&self, d: Duration) -> Option<Event> {
        self.rx.recv_timeout(d).ok()
    }

    /// Stops the worker without waiting for it (switching backends): it exits at its next
    /// command check — at once when idle, at the next SCF step on the GPU, after the solve in
    /// flight on the CPU or in the driver's construction. Its remaining events are dropped with
    /// the channel.
    pub fn detach(mut self) {
        let _ = self.tx.send(Command::Shutdown);
        self.handle = None; // a dropped JoinHandle detaches the thread; `Drop` then has nothing to join
    }

    /// Shuts the worker down and waits up to `wait` for its [`Event::Retired`] (its autosave is
    /// then on disk), then detaches it. False if it did not retire in time (a CPU solve in
    /// flight cannot be interrupted): the autosave may then be missing or older.
    pub fn retire(self, wait: Duration) -> bool {
        let _ = self.tx.send(Command::Shutdown);
        let deadline = Instant::now() + wait;
        let mut retired = false;
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match self.rx.recv_timeout(left) {
                Ok(Event::Retired) => {
                    retired = true;
                    break;
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        self.detach();
        retired
    }
}

impl Drop for SimWorker {
    /// Asks the worker to stop and joins it. A GPU solve in flight is abandoned at the next SCF
    /// step (`Shutdown` aborts); a CPU solve, or the driver's construction, runs to completion
    /// first. If the thread already exited, the send fails and the join returns at once.
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The command receiver plus the commands the abort check took off the channel, handed out
/// first (in order) by the next [`Inbox::take`].
struct Inbox {
    rx: Receiver<Command>,
    held: VecDeque<Command>,
    /// The viewer is gone: whatever runs is for nobody.
    closed: bool,
}

impl Inbox {
    fn pull(&mut self) {
        while !self.closed {
            match self.rx.try_recv() {
                Ok(c) => self.held.push_back(c),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => self.closed = true,
            }
        }
    }

    /// `should_abort` (module doc, R11): an aborting command is queued, or the viewer is gone.
    fn abort_pending(&mut self) -> bool {
        self.pull();
        self.closed || self.held.iter().any(Command::aborts)
    }

    /// Waits for a command until `deadline` (the tick pacing, module doc); returns at once when
    /// one is already held.
    fn wait_until(&mut self, deadline: Instant) {
        self.pull();
        if self.closed || !self.held.is_empty() {
            return;
        }
        match self.rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(c) => self.held.push_back(c),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => self.closed = true,
        }
    }

    /// The next batch of commands; with `block`, waits for the first one and collects the burst
    /// for [`COALESCE_WINDOW`]. `None`: shut down (Shutdown queued or the viewer gone).
    fn take(&mut self, block: bool) -> Option<Vec<Command>> {
        if block && self.held.is_empty() && !self.closed {
            match self.rx.recv() {
                Ok(c) => self.held.push_back(c),
                Err(_) => self.closed = true,
            }
            let deadline = Instant::now() + COALESCE_WINDOW;
            while !self.closed {
                match self.rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                    Ok(c) => self.held.push_back(c),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => self.closed = true,
                }
            }
        }
        self.pull();
        if self.closed || self.held.iter().any(|c| matches!(c, Command::Shutdown)) {
            return None;
        }
        Some(self.held.drain(..).collect())
    }
}

/// Whether the driver has something to do without a new command (`Snapshot::working`): playing
/// (with nuclei to move: an empty scene that plays has nothing to tick — module doc, "The empty
/// scene"), queued steps, a stale-force evaluation, or a sampling pass in progress. A stale
/// evaluation that failed is not retried until Play, Step or an edit (they clear `last_error`).
fn has_work(s: &DynSnapshot) -> bool {
    (!s.paused && !s.scene.nuclei.is_empty())
        || s.pending_steps > 0
        || (s.forces_stale && s.last_error.is_none())
        || sampling_active(s)
}

/// A sampling pass is running (not finished, not cancelled).
pub fn sampling_active(s: &DynSnapshot) -> bool {
    s.sampling.as_ref().is_some_and(|p| !p.finished)
}

/// The surrogate behind the driver's provider, if the provider is one (the worker's view behind
/// the boundary: refits and readouts, `sim::surrogate_ks` module doc).
fn surrogate(dy: &mut Dynamics) -> Option<&mut SurrogateKs> {
    dy.provider_mut().as_any_mut()?.downcast_mut::<SurrogateKs>()
}

type MakeProvider = Box<dyn Fn() -> Box<dyn ForceProvider> + Send>;

struct Worker {
    dy: Dynamics,
    make: MakeProvider,
    tx: Sender<Event>,
    backend: Backend,
    engine: Backend,
    edits: u64,
    /// Commands applied to the driver ([`Snapshot::driver_commands`]).
    driver_commands: u64,
    busy: bool,
    /// The last published snapshot without its display (holding the texture would keep it out
    /// of the provider's pool): the idle test and the panic recovery read it.
    last: DynSnapshot,
    /// Estimated solver: where the model goes at shutdown (module doc, "Autosave").
    autosave: Option<PathBuf>,
    last_refit: Option<RefitLog>,
    /// The store size at the last refit (the throttle's sample count).
    samples_at_refit: usize,
    /// When the last refit ended (the refit throttle, [`REFIT_INTERVAL`]).
    last_refit_at: Option<Instant>,
    refits: u64,
    autoload_refused: Option<String>,
    /// This worker may write or remove the autosave (module doc, "Autosave").
    owns_autosave: bool,
    /// The audit cadence set by [`Command::SetAudit`] and draft mode by [`Command::SetDraft`]
    /// (re-applied after a panic recovery).
    audit_every: Option<usize>,
    draft: Option<bool>,
    /// The last audit record seen and the step its geometry was dispatched at (latched).
    last_audit_seen: Option<sim::surrogate_ks::AuditRecord>,
    last_audit_step: Option<u64>,
    /// The latched teacher gap ([`ModelStatus::last_gap`]).
    last_gap: Option<(f64, u64)>,
    /// The end of the current tick's budget, when it is to be waited out (module doc, "Pacing").
    pace_until: Option<Instant>,
    /// The pacing schedule's last deadline.
    pace_anchor: Option<Instant>,
    /// Measured or estimated wall cost of one pass solve (s; [`ModelStatus::solve_s`]).
    solve_s: Option<f64>,
    /// The running pass: when its current timing segment started, its `done` count then, and
    /// whether that segment is the reference stage (timed apart: `track_cost`).
    pass_clock: Option<(Instant, usize, bool)>,
    /// The pass's remaining time (s), from its measured rate.
    pass_eta_s: Option<f64>,
    pass_offer: Option<PassOffer>,
    /// The cloud (module doc): profiles wanted, the refresh interval and the last request, the
    /// profiles held and the atom job in flight (Z, n, h, σ).
    cloud_superposition: bool,
    cloud_refresh: Option<Duration>,
    last_refresh: Option<Instant>,
    profiles: Arc<Vec<AtomProfile>>,
    atom_pending: Option<(u8, usize, f64, Option<f64>)>,
    /// The device a GPU atom density is read back from.
    gpu: Option<GpuContext>,
}

/// The automatic first sampling pass (`Command::EnsureModel`) aims to finish within this many
/// seconds (Task 9 item 4): the panel's plan is scaled down to it, never below what trust needs;
/// a pass that still takes longer waits for the user's "Start anyway" ([`PassOffer`]).
pub const PASS_TARGET_S: f64 = 120.0;

/// A pass solve costs about this multiple of the scene's first, cold evaluation (the estimate
/// before any pass solve was timed; once a pass runs, its own rate is used). Measured
/// (`diag_pass_rate`, RX 7900 XTX): H₂ 41³/49³ — first evaluation 0.20/0.23 s, perturbations
/// 0.17/0.21 s, scans 0.16/0.19 s, hot burst 0.08/0.10 s (the whole pass ≈ 0.55×); H₂O 41³/57³ —
/// first evaluation 0.81/1.11 s, perturbations 1.28/1.87 s (1.6–1.7×), and the O-atom reference
/// scan 13–15 s a multiplicity (open-shell, cold: not modelled — the live ETA shows it). 1.6:
/// right for the molecules whose passes are long, and still under the 2-minute target for H₂'s
/// default pass (273 × 0.37 s ≈ 100 s at 49³).
pub const PASS_SOLVE_PER_COLD: f64 = 1.6;

/// A first pass that would take longer than [`PASS_TARGET_S`] even at the fewest solves that can
/// earn trust: not started until the user confirms (the panel's "Start anyway").
#[derive(Clone, Debug, PartialEq)]
pub struct PassOffer {
    /// The plan "Start anyway" starts (the panel's, scaled down as far as trust allows).
    pub plan: SamplingPlan,
    pub solves: usize,
    /// Estimated wall time (s) at `per_solve_s`.
    pub eta_s: f64,
    pub per_solve_s: f64,
    /// The panel's plan unscaled: solves and estimated wall time (s).
    pub full_solves: usize,
    pub full_eta_s: f64,
}

/// What the automatic first pass does ([`first_pass`]).
#[derive(Clone, Debug, PartialEq)]
pub enum FirstPass {
    /// The panel's plan fits the target (or nothing is known about the cost).
    Start(SamplingPlan),
    /// Scaled down to fit, with the note saying so.
    Scaled(SamplingPlan, String),
    /// Even the fewest solves that can earn trust exceed the target: ask.
    Offer(PassOffer),
}

/// The first-pass decision (Task 9 item 4): `plan` on `scene` at `per_solve_s` seconds a solve
/// against [`PASS_TARGET_S`] (5 % slack for the scaled plan's rounding).
pub fn first_pass(scene: &sim::Scene, plan: SamplingPlan, per_solve_s: Option<f64>) -> FirstPass {
    let Some(per) = per_solve_s.filter(|c| *c > 0.0) else { return FirstPass::Start(plan) };
    let full = plan.solves(scene).unwrap_or(0);
    if full as f64 * per <= PASS_TARGET_S {
        return FirstPass::Start(plan);
    }
    let scaled = plan.scaled_to(scene, (PASS_TARGET_S / per).floor() as usize);
    let n = scaled.solves(scene).unwrap_or(full);
    if n as f64 * per <= PASS_TARGET_S * 1.05 {
        let note = format!(
            "sampling plan scaled to {n} solves (≈ {:.0} s at ≈ {per:.2} s per solve here; the panel's plan, {full} solves, would take ≈ {:.0} min) — Sample more adds to it",
            n as f64 * per,
            full as f64 * per / 60.0
        );
        FirstPass::Scaled(scaled, note)
    } else {
        FirstPass::Offer(PassOffer { plan: scaled, solves: n, eta_s: n as f64 * per, per_solve_s: per, full_solves: full, full_eta_s: full as f64 * per })
    }
}

/// While playing, a pending sample waits for a refit at most this long after the previous one
/// (module doc, "Refits") …
const REFIT_INTERVAL: Duration = Duration::from_millis(500);
/// … or until this many samples arrived since it, whichever comes first.
const REFIT_SAMPLES: usize = 25;

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".into())
}

/// The estimated solver's background auditor: a second `GpuKs` on the same context (its own
/// solver, its own buffers — ≈ the first one's memory again, a few MB at 49³), its density copied
/// into a texture of its own pool: the audits' and refreshes' densities are the cloud between DFT
/// steps (module doc, "The cloud").
fn audit_teacher(ctx: &GpuContext) -> GpuKs {
    GpuKs::new(ctx.clone(), SAMPLE_TOL).with_vram_budget(crate::quality::vram_budget_bytes() / 2)
}

/// The Kohn-Sham provider for `engine` (`Gpu` needs `gpu`; anything else is the CPU solver) at
/// SCF tolerance `tol` ([`INTERACTIVE_TOL`] for a DFT backend, [`SAMPLE_TOL`] for the
/// estimated solver's teacher).
/// A GPU solver gets the GPU memory budget (`quality::vram_budget_bytes`; half of it when it is
/// the estimated solver's teacher (`solvers` = 2), whose auditor takes the other half — `sim::provider`, "GPU
/// memory"), so a scene over it is refused before anything is allocated.
fn kohn_sham(engine: Backend, gpu: &Option<GpuContext>, tol: f64, solvers: u64) -> Box<dyn ForceProvider> {
    let budget = crate::quality::vram_budget_bytes() / solvers.max(1);
    match gpu {
        Some(ctx) if engine == Backend::Gpu => Box::new(GpuKs::new(ctx.clone(), tol).with_vram_budget(budget)),
        _ => Box::new(CpuKs::new(tol)),
    }
}

fn worker(
    mut initial: Scene,
    settings: Vec<DynCommand>,
    rx: Receiver<Command>,
    tx: Sender<Event>,
    backend: Backend,
    gpu: Option<GpuContext>,
    autosave: Option<PathBuf>,
    provider: Option<MakeProvider>,
) {
    let gpu_readback = gpu.clone();
    // the Kohn-Sham engine: the backend itself, or the estimated solver's teacher
    let engine = match backend {
        Backend::Estimated if gpu.is_some() => Backend::Gpu,
        Backend::Estimated => Backend::Cpu,
        b => b,
    };
    let make: MakeProvider = provider.unwrap_or_else(|| {
        Box::new(move || {
            if backend == Backend::Estimated {
                let ks = kohn_sham(engine, &gpu, SAMPLE_TOL, 2);
                let s = SurrogateKs::new(ks, FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL);
                Box::new(match gpu.as_ref().filter(|_| engine == Backend::Gpu) {
                    // the audits on a second GPU teacher, off the stepping path (ruling S14)
                    Some(ctx) => s.with_auditor(Box::new(audit_teacher(ctx))),
                    // the CPU teacher audits in the step (module doc, "The estimated solver")
                    None => s,
                })
            } else {
                kohn_sham(engine, &gpu, INTERACTIVE_TOL, 1)
            }
        })
    });
    let _ = tx.send(Event::Busy(true));
    // the viewer validates its scenes; a nucleus without a mass would make `to_sim` fail
    let before = initial.nuclei.len();
    initial
        .nuclei
        .retain(|n| units::isotope_mass_amu(n.z).is_some());
    if initial.nuclei.len() != before {
        let _ = tx.send(Event::Error(format!(
            "dropped {} nuclei without an isotope mass (Z outside 1..=10)",
            before - initial.nuclei.len()
        )));
    }
    let scene = initial.to_sim().expect("every nucleus has a mass");
    let dy = match catch_unwind(AssertUnwindSafe(|| {
        Dynamics::new(scene, make(), DEFAULT_DT_FS, SEED)
    })) {
        Ok(d) => d,
        Err(p) => {
            let _ = tx.send(Event::Error(format!("solver panicked: {}", panic_message(p))));
            let _ = tx.send(Event::Busy(false));
            let _ = tx.send(Event::Retired);
            return;
        }
    };
    let mut w = Worker {
        dy,
        make,
        tx,
        backend,
        engine,
        edits: 0,
        driver_commands: 0,
        busy: true,
        last: DynSnapshot::default(),
        autosave,
        last_refit: None,
        last_refit_at: None,
        samples_at_refit: 0,
        refits: 0,
        autoload_refused: None,
        owns_autosave: false,
        audit_every: None,
        draft: None,
        last_audit_seen: None,
        last_audit_step: None,
        last_gap: None,
        pace_until: None,
        pace_anchor: None,
        solve_s: None,
        pass_clock: None,
        pass_eta_s: None,
        pass_offer: None,
        cloud_superposition: false,
        cloud_refresh: None,
        last_refresh: None,
        profiles: Arc::new(Vec::new()),
        atom_pending: None,
        gpu: gpu_readback,
    };
    for c in settings {
        w.dy.apply(c);
    }
    let first = w.dy.snapshot();
    w.publish(first);
    let mut inbox = Inbox {
        rx,
        held: VecDeque::new(),
        closed: false,
    };
    loop {
        if let Some(d) = w.pace_until {
            inbox.wait_until(d);
        }
        let Some(cmds) = inbox.take(!has_work(&w.last)) else {
            w.retire();
            return;
        };
        // a command before the paced deadline: applied and published, no tick (module doc)
        let early = w.pace_until.is_some_and(|d| Instant::now() < d);
        let r = catch_unwind(AssertUnwindSafe(|| {
            if early {
                w.apply_batch(cmds);
                let s = w.dy.snapshot();
                w.publish(s);
            } else {
                w.pace_until = None;
                w.round(cmds, &mut inbox);
            }
        }));
        if let Err(p) = r {
            if !w.recover(panic_message(p)) {
                let _ = w.tx.send(Event::Retired);
                return;
            }
        }
    }
}

impl Worker {
    /// Applies a batch (mirrored edits counted, superseded drag moves and teleports skipped);
    /// whether it held any command.
    fn apply_batch(&mut self, cmds: Vec<Command>) -> bool {
        let applied = !cmds.is_empty();
        let mut it = cmds.into_iter().peekable();
        while let Some(c) = it.next() {
            if c.is_edit() {
                self.edits += 1;
            }
            if !c.superseded_by(it.peek()) {
                if c.reaches_driver() {
                    self.driver_commands += 1;
                }
                self.apply(c);
            }
        }
        applied
    }

    /// Applies a batch, ticks once, refits the surrogate if it kept a sample, publishes.
    fn round(&mut self, cmds: Vec<Command>, inbox: &mut Inbox) {
        let applied = self.apply_batch(cmds);
        let will_work = if applied {
            has_work(&self.dy.snapshot())
        } else {
            has_work(&self.last)
        };
        self.set_busy(will_work);
        let t0 = Instant::now();
        let mut snap = self
            .dy
            .tick(TICK_BUDGET_MS, &mut || inbox.abort_pending());
        // pacing (module doc): a playing tick that ran all its steps early waits out the budget.
        // The deadlines follow a fixed 16 ms schedule rather than "start + 16 ms": a sleep that
        // overshoots (Windows timers tick at ~15.6 ms) is made up by not waiting next time, so the
        // mean tick stays 16 ms (measured: 8 fs/s instead of 15.6 at speed 1 without it).
        let budget = Duration::from_secs_f64(TICK_BUDGET_MS * 1e-3);
        let paced = !snap.paused && snap.speed < sim::dynamics::MAX_SPEED && snap.steps_last_tick >= snap.speed as usize;
        self.pace_until = if paced {
            let next = match self.pace_anchor {
                // never more than one budget ahead of this tick's start (fix round 2)
                Some(a) if a + budget + budget > t0 => (a + budget).min(t0 + budget),
                _ => t0 + budget, // the first paced tick, or far behind: restart the schedule
            };
            self.pace_anchor = Some(next);
            Some(next)
        } else {
            self.pace_anchor = None;
            None
        };
        if self.refit_if_pending(&snap) {
            // the refit's note is in the driver now: publish it with this snapshot
            let d = self.dy.snapshot();
            (snap.notes, snap.notes_total) = (d.notes, d.notes_total);
        }
        self.cloud_jobs(&snap);
        self.publish(snap);
    }

    /// The cloud's background jobs, between ticks (module doc, "The cloud"): arrived atom
    /// densities become profiles; a missing profile for the scene's species, grid and σ is asked
    /// of the auditor, one species at a time; while playing, a display refresh every
    /// `cloud_refresh` (queued ahead of the next audit when the auditor is busy). Never on the
    /// step path; nothing without a background auditor.
    fn cloud_jobs(&mut self, snap: &DynSnapshot) {
        let gpu = self.gpu.clone();
        let Some(s) = surrogate(&mut self.dy) else { return };
        if !s.audits_asynchronous() {
            return;
        }
        let cell = &snap.scene.cell;
        let key = (cell.n, cell.h, snap.scene.sigma);
        for (z, d) in s.take_atom_densities() {
            let pending = self.atom_pending.take();
            let (values, n, h, origin) = match &d {
                sim::Display::Cpu { rho, cell, origin, .. } => (rho.to_vec(), cell.n, cell.h, *origin),
                sim::Display::Gpu { volume, cell, origin, .. } => match &gpu {
                    Some(ctx) => (volume.read_back(ctx), cell.n, cell.h, *origin),
                    None => continue,
                },
            };
            // asked for on another grid (the scene changed since): dropped, asked again below
            let Some((pz, pn, ph, psigma)) = pending else { continue };
            if pz != z || (pn, ph, psigma) != key || (n, h) != (pn, ph) {
                continue;
            }
            let profile = crate::cloud::radial_profile(z, &values, pn, ph, origin, psigma);
            let mut all: Vec<AtomProfile> = self.profiles.iter().filter(|p| p.z != z).cloned().collect();
            all.push(profile);
            self.profiles = Arc::new(all);
        }
        // profiles of another grid or σ are dropped (and asked for again below)
        if self.profiles.iter().any(|p| !p.fits(key.0, key.1, key.2)) {
            self.profiles = Arc::new(self.profiles.iter().filter(|p| p.fits(key.0, key.1, key.2)).cloned().collect());
        }
        if self.cloud_superposition && self.atom_pending.is_none() {
            let mut species: Vec<u8> = snap.scene.nuclei.iter().map(|n| n.z).collect();
            species.sort_unstable();
            species.dedup();
            if let Some(&z) = species.iter().find(|&&z| !self.profiles.iter().any(|p| p.z == z)) {
                if s.request_atom_density(z, &snap.scene) {
                    self.atom_pending = Some((z, key.0, key.1, key.2));
                }
            }
        }
        if let Some(every) = self.cloud_refresh {
            let due = self.last_refresh.is_none_or(|t| t.elapsed() >= every);
            if !snap.paused && due && s.request_display(&snap.scene) {
                self.last_refresh = Some(Instant::now());
            }
        }
    }

    fn apply(&mut self, c: Command) {
        match c {
            Command::SetScene(s) => match s.to_sim() {
                Ok(mut new) => {
                    new.external_field = self.dy.snapshot().scene.external_field;
                    self.dy.apply(DynCommand::Pause);
                    self.dy.apply(DynCommand::ReplaceScene(new));
                }
                Err(e) => self.error(e),
            },
            Command::MoveNucleus { index, pos } => self
                .dy
                .apply(DynCommand::MoveNucleusDiscontinuous { index, pos }),
            Command::AddNucleus { z, pos } => match units::isotope_mass_amu(z) {
                Some(_) => self
                    .dy
                    .apply(DynCommand::AddNucleus(sim::Nucleus::new(z, pos))),
                None => self.error(format!("no isotope mass for Z = {z}")),
            },
            Command::RemoveNucleus(i) => self.dy.apply(DynCommand::RemoveNucleus(i)),
            Command::SetCharge(c) => self.replace(|s| s.charge = c),
            Command::SetMultiplicity(m) => self.replace(|s| s.multiplicity = m),
            Command::SetGrid { n, h } => self.replace(|s| s.cell = sim::Cell { n, h }),
            Command::SetSigma(x) => self.replace(|s| s.sigma = x),
            Command::Dyn(c) => {
                // the user's own model actions make the session's model this worker's
                let takes_over = matches!(c, DynCommand::ForgetModel | DynCommand::StartSampling(_) | DynCommand::LoadModel(_));
                let forget = matches!(c, DynCommand::ForgetModel);
                if takes_over {
                    self.pass_offer = None;
                }
                self.dy.apply(c);
                if takes_over && self.dy.snapshot().last_rejection.is_none() && surrogate(&mut self.dy).is_some() {
                    self.autoload_refused = None;
                    self.owns_autosave = true;
                    if forget {
                        self.last_gap = None;
                    }
                }
            }
            Command::EnsureModel { autoload, plan } => self.ensure_model(autoload, plan),
            Command::SetAudit(n) => {
                self.audit_every = Some(n);
                self.apply_audit();
            }
            Command::SetDraft(on) => {
                self.draft = Some(on);
                self.apply_audit();
            }
            Command::SetCloud { superposition, refresh_s } => {
                self.cloud_superposition = superposition;
                self.cloud_refresh = refresh_s.filter(|x| x.is_finite() && *x > 0.0).map(Duration::from_secs_f64);
            }
            Command::Shutdown => {} // `Inbox::take` never hands it out
        }
    }

    /// The [`Command::SetAudit`] cadence and [`Command::SetDraft`] onto the surrogate's current
    /// policy.
    fn apply_audit(&mut self) {
        let (audit, draft) = (self.audit_every, self.draft);
        if let Some(s) = surrogate(&mut self.dy) {
            let mut p = *s.policy();
            if let Some(n) = audit {
                p.audit_every = n;
            }
            if let Some(on) = draft {
                p.never_fallback = on;
            }
            if p != *s.policy() {
                s.set_policy(p);
            }
        }
    }

    /// [`Command::EnsureModel`] (module doc, "Teaching"; "Autosave" for the ownership).
    fn ensure_model(&mut self, autoload: Option<PathBuf>, plan: SamplingPlan) {
        if surrogate(&mut self.dy).is_none() {
            return;
        }
        if let Some(path) = autoload.filter(|p| p.exists()) {
            self.dy.apply(DynCommand::LoadModel(path));
            if let Some(r) = self.dy.snapshot().last_rejection {
                // not ours: the file stays for a later worker on the settings it fits
                self.autoload_refused = Some(r);
                return;
            }
        }
        self.owns_autosave = true;
        // an empty scene needs no model (module doc, "The empty scene"): no pass, no offer
        if self.dy.snapshot().scene.nuclei.is_empty() {
            return;
        }
        let fitted = surrogate(&mut self.dy).is_some_and(|s| s.model().is_some());
        if !fitted && !sampling_active(&self.dy.snapshot()) {
            self.start_first_pass(plan);
        }
    }

    /// The automatic first pass (Task 9 item 4): the panel's plan if it fits
    /// [`PASS_TARGET_S`] at the expected per-solve cost; else that plan scaled down to fit, with
    /// a note; else — when even the fewest solves that can earn trust take longer — nothing, and
    /// a [`PassOffer`] for the panel's "Start anyway".
    fn start_first_pass(&mut self, plan: SamplingPlan) {
        let scene = self.dy.snapshot().scene;
        match first_pass(&scene, plan, self.solve_s) {
            FirstPass::Start(plan) => self.dy.apply(DynCommand::StartSampling(plan)),
            FirstPass::Scaled(plan, note) => {
                self.dy.add_note(note);
                self.dy.apply(DynCommand::StartSampling(plan));
            }
            FirstPass::Offer(o) => self.pass_offer = Some(o),
        }
    }

    /// Active learning (module doc, "Refits"): the surrogate kept a fallback sample and no pass
    /// is running (a pass refits at its end).
    /// Whether a refit ran.
    fn refit_if_pending(&mut self, snap: &DynSnapshot) -> bool {
        if sampling_active(snap) {
            return false;
        }
        let Some(s) = surrogate(&mut self.dy) else { return false };
        // before the first fit there are no reference energies (the pass computes them and
        // makes the first fit at its end): rule-1 fallback samples wait for it
        if !s.pending_refit() || s.model().is_none() {
            return false;
        }
        if let (false, Some(at)) = (snap.paused, self.last_refit_at) {
            if at.elapsed() < REFIT_INTERVAL && s.samples() < self.samples_at_refit + REFIT_SAMPLES {
                return false;
            }
        }
        let t = Instant::now();
        let fit = s.refit();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.refits += 1;
        let k = self.refits;
        let note = match &fit {
            Ok(r) => format!(
                "refit #{k}: {} samples, rmse_E {:.1e} Ha/atom, rmse_F {:.1e} Ha/Bohr, {ms:.0} ms — trusted: {}{}",
                r.n_samples,
                r.rmse_energy_per_atom,
                r.rmse_force,
                match s.untrusted_reason() {
                    None => "yes".to_string(),
                    Some(why) => format!("no — {why}"),
                },
                s.recovery_note().map_or(String::new(), |n| format!("; {n}"))
            ),
            Err(e) => format!("refit #{k} failed: {e} (retried at the next opportunity)"),
        };
        let samples = s.samples();
        let result = fit.map(|r| r.n_samples).map_err(|e| e.to_string());
        self.samples_at_refit = samples;
        self.last_refit = Some(RefitLog { ms, at_step: snap.step, result });
        self.last_refit_at = Some(Instant::now());
        self.dy.add_note(note);
        true
    }

    /// The surrogate's readouts (estimated solver only); `forces` are the snapshot's.
    fn model_status(&mut self, scene: &sim::Scene, forces: &[[f64; 3]], step: u64) -> Option<ModelStatus> {
        let (last_refit, refits, autoload_refused, last_gap) =
            (self.last_refit.clone(), self.refits, self.autoload_refused.clone(), self.last_gap);
        let last_seen = self.last_audit_seen.clone();
        let (solve_s, pass_eta_s, pass_offer) = (self.solve_s, self.pass_eta_s, self.pass_offer.clone());
        let s = surrogate(&mut self.dy)?;
        let policy = *s.policy();
        // latch the step of a new audit record (its verdict's step less its lag)
        let audit = s.last_audit().cloned();
        let new_audit = audit.is_some() && audit != last_seen;
        let fmin = forces
            .iter()
            .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
            .fold(f64::INFINITY, f64::min);
        let want = s.provenance_for(scene);
        let provenance_mismatch = s.model().filter(|m| m.provenance != want).map(|m| {
            let p = &m.provenance;
            format!(
                "the model was trained at {}³ h {} σ {}, charge {}, multiplicity {} — the scene is {}³ h {} σ {}, charge {}, multiplicity {}",
                p.grid_n,
                p.h,
                p.sigma,
                p.charge,
                p.multiplicity.map_or("lowest".into(), |m| m.to_string()),
                want.grid_n,
                want.h,
                want.sigma,
                want.charge,
                want.multiplicity.map_or("lowest".into(), |m| m.to_string()),
            )
        });
        let status = ModelStatus {
            teacher: s.teacher_mut().name(),
            fitted: s.model().is_some(),
            trusted: s.trusted(),
            untrusted_reason: s.model().and_then(|_| s.untrusted_reason()),
            samples: s.samples(),
            pending_refit: s.pending_refit(),
            epoch: s.epoch(),
            fit: s.report().map(|r| (r.n_samples, r.q_prime, r.rmse_energy_per_atom, r.rmse_force)),
            residuals: s.training_residuals().map(|t| (t.energy_per_atom, t.force, t.force_bound)),
            force_bound: policy.force_bound(if fmin.is_finite() { fmin } else { 0.0 }),
            provenance_mismatch,
            last_refit,
            refits,
            autoload_refused,
            audit_every: policy.audit_every,
            draft: policy.never_fallback,
            last_audit_step: None, // filled below
            last_gap,
            quarantined: s.quarantined(),
            recovery: s.recovery_note().map(str::to_string),
            audits_background: s.audits_asynchronous(),
            audit_in_flight: s.audit_in_flight(),
            last_audit: s.last_audit().cloned(),
            stuck: s.stuck(),
            solve_s,
            pass_eta_s,
            pass_offer,
        };
        if new_audit {
            let lag = audit.as_ref().map_or(0, |a| a.unaudited_steps as u64);
            self.last_audit_step = Some(step.saturating_sub(lag));
            self.last_audit_seen = audit;
        }
        Some(ModelStatus { last_audit_step: self.last_audit_step, ..status })
    }

    /// Pause, then replace the scene by the driver's current one with `f` applied.
    fn replace(&mut self, f: impl FnOnce(&mut sim::Scene)) {
        self.dy.apply(DynCommand::Pause);
        let mut s = self.dy.snapshot().scene;
        f(&mut s);
        self.dy.apply(DynCommand::ReplaceScene(s));
    }

    fn error(&self, message: String) {
        let _ = self.tx.send(Event::Error(message));
    }

    fn set_busy(&mut self, b: bool) {
        if b != self.busy {
            self.busy = b;
            let _ = self.tx.send(Event::Busy(b));
        }
    }

    fn publish(&mut self, snap: DynSnapshot) {
        self.track_cost(&snap);
        let working = has_work(&snap);
        if let Some(g) = snap.teacher_gap {
            self.last_gap = Some((g, snap.step));
        }
        let model = self.model_status(&snap.scene, &snap.forces, snap.step);
        self.last = DynSnapshot {
            display: None,
            side_display: None,
            ..snap.clone()
        };
        // the receiver may be gone (SimWorker dropped mid-tick); the next take sees that
        let _ = self.tx.send(Event::Snapshot(Box::new(Snapshot {
            scene: Scene::from_sim(&snap.scene),
            dynamics: snap,
            backend: self.backend,
            engine: self.engine,
            model,
            atom_profiles: self.profiles.clone(),
            edits: self.edits,
            driver_commands: self.driver_commands,
            working,
        })));
        self.set_busy(working);
    }

    /// The per-solve cost estimate and the pass ETA ([`ModelStatus::solve_s`], `pass_eta_s`).
    fn track_cost(&mut self, snap: &DynSnapshot) {
        let teacher_step = snap.source != sim::Source::Surrogate && !snap.forces_stale && snap.eval.solve_ms > 0;
        match snap.sampling.as_ref().filter(|p| !p.finished) {
            Some(p) => {
                // the isolated-atom reference scans (open-shell and cold: 13–15 s each for O)
                // are cached for later passes, so they are timed apart: the rate a later "Sample
                // more" is estimated with is the geometry solves' alone (Task 9 review M2)
                let refs = p.stage == "references";
                if self.pass_clock.is_some_and(|c| c.2 != refs) {
                    self.pass_clock = None;
                }
                let (t0, d0, _) = *self.pass_clock.get_or_insert((Instant::now(), p.done, refs));
                let solved = p.done.saturating_sub(d0);
                if solved >= 2 {
                    let per = t0.elapsed().as_secs_f64() / solved as f64;
                    if !refs {
                        self.solve_s = Some(per);
                    }
                    self.pass_eta_s = Some(per * p.total.saturating_sub(p.done) as f64);
                }
            }
            None => {
                if self.pass_clock.take().is_none() && teacher_step && self.solve_s.is_none() {
                    // no pass timed yet: the scene's own teacher evaluation, scaled
                    self.solve_s = Some(PASS_SOLVE_PER_COLD * snap.eval.solve_ms as f64 * 1e-3);
                }
                self.pass_eta_s = None;
            }
        }
    }

    /// Writes the surrogate's model to `path` (a fitted model), or removes `path` (none); the
    /// error message on failure.
    fn save_model_to(&mut self, path: &std::path::Path) -> Result<bool, String> {
        let Some(s) = surrogate(&mut self.dy) else { return Ok(false) };
        if s.model().is_some() {
            s.save_sidecar(path).map_err(|e| format!("autosave {}: {e}", path.display()))?;
            Ok(true)
        } else {
            match std::fs::remove_file(path) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("autosave {}: {e}", path.display())),
                _ => Ok(false),
            }
        }
    }

    /// Shutdown (module doc, "Autosave"): the model to the autosave path — only if this worker
    /// owns it — then `Retired`.
    fn retire(&mut self) {
        if let Some(path) = self.autosave.clone().filter(|_| self.owns_autosave) {
            if let Err(e) = self.save_model_to(&path) {
                self.error(e);
            }
        }
        let _ = self.tx.send(Event::Retired);
    }

    /// After a panic the driver may be mid-step: rebuild it from the last snapshot (a fresh
    /// provider, time restarts) with the run's settings re-applied, paused; an estimated run's
    /// model is carried over through the autosave path when it can still be written. False if
    /// that panics too (the worker then exits).
    fn recover(&mut self, message: String) -> bool {
        self.error(format!(
            "solver panicked: {message} — the run was rebuilt from the last snapshot (paused, time restarts)"
        ));
        let carried = match self.autosave.clone().filter(|_| self.owns_autosave) {
            Some(path) => match catch_unwind(AssertUnwindSafe(|| self.save_model_to(&path))) {
                Ok(Ok(true)) => Some(path),
                _ => None,
            },
            None => None,
        };
        let last = self.last.clone();
        let dt = if last.dt > 0.0 { last.dt_fs() } else { DEFAULT_DT_FS };
        let make = &self.make;
        match catch_unwind(AssertUnwindSafe(|| {
            let mut d = Dynamics::new(last.scene.clone(), make(), dt, SEED);
            d.apply(DynCommand::SetSpeed(last.speed));
            d.apply(DynCommand::SetComCorrection(last.com_correction));
            d.apply(DynCommand::SetThermostat(last.thermostat));
            if last.drag_force_max != sim::dynamics::F_DRAG_MAX {
                d.apply(DynCommand::SetDragForce { f_max: last.drag_force_max });
            }
            for (i, &p) in last.pins.iter().enumerate() {
                if p {
                    d.apply(DynCommand::Pin(i, true));
                }
            }
            if let Some(path) = carried {
                d.apply(DynCommand::LoadModel(path));
            }
            d
        })) {
            Ok(d) => {
                self.dy = d;
                self.apply_audit();
                let s = self.dy.snapshot();
                self.publish(s);
                true
            }
            Err(p) => {
                self.error(format!(
                    "solver panicked again while recovering: {} — the worker stopped; switch the solver to restart it",
                    panic_message(p)
                ));
                self.set_busy(false);
                false
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::scene::Scene;
    use gpu::Volume3D;
    use sim::Display;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(120);

    /// Serialises the tests that run GPU solves (here and in `viewer`): run concurrently they
    /// share one adapter, and the timing-sensitive ones (steps per second under a drag-move
    /// flood, abort latency) would measure each other. Poisoning (a failed test) is harmless.
    static GPU_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(crate) fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        GPU_TESTS.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The next snapshot (worker errors fail the test).
    fn next_snapshot(w: &SimWorker) -> Snapshot {
        let deadline = Instant::now() + WAIT;
        while Instant::now() < deadline {
            match w.recv_timeout(Duration::from_secs(1)) {
                Some(Event::Snapshot(s)) => return *s,
                Some(Event::Error(message)) => panic!("{message}"),
                _ => {}
            }
        }
        panic!("no snapshot within {WAIT:?}");
    }

    /// Snapshots until one satisfies `f`.
    fn snapshot_until(w: &SimWorker, what: &str, f: impl Fn(&Snapshot) -> bool) -> Snapshot {
        let deadline = Instant::now() + WAIT;
        loop {
            assert!(Instant::now() < deadline, "no snapshot with {what} within {WAIT:?}");
            let s = next_snapshot(w);
            if f(&s) {
                return s;
            }
        }
    }

    /// Snapshots until the worker is idle (paused, nothing queued, forces fresh or failed).
    fn idle(w: &SimWorker) -> Snapshot {
        snapshot_until(w, "an idle worker", |s| !s.working)
    }

    /// The headless context for the GPU backend tests (`None` without an adapter: they skip).
    pub(crate) fn gpu_ctx() -> Option<GpuContext> {
        let c = GpuContext::new_headless();
        if c.is_none() {
            eprintln!("no GPU adapter: skipping the GPU backend");
        }
        c
    }

    /// Both backends when a GPU exists, else CPU only.
    fn backends() -> Vec<(Backend, Option<GpuContext>)> {
        let mut v = vec![(Backend::Cpu, None)];
        if let Some(c) = gpu_ctx() {
            v.push((Backend::Gpu, Some(c)));
        }
        v
    }

    /// The `h2` preset (R = 1.6 Bohr) on a 33³ grid (box 8 Bohr): the dynamics tests' system.
    fn h2_33() -> Scene {
        let mut s = crate::scene::h2_r16();
        s.grid_n = 33;
        s
    }

    fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
    }

    fn display_len(s: &Snapshot) -> usize {
        match &s.dynamics.display {
            Some(Display::Cpu { rho, .. }) => rho.len(),
            Some(Display::Gpu { volume, .. }) => volume.len(),
            None => 0,
        }
    }

    // ---- dynamics in the worker (M4 Task 6) ----

    /// Play on GPU H₂ 33³: snapshots stream (≥ 5 in 3 s), time increases, T is finite, the drift
    /// is `None` until the history holds 100 points and `Some` after.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn playing_advances_sim_time_and_publishes_snapshots_with_temperature() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
        let s0 = idle(&w);
        assert!(s0.dynamics.paused && s0.dynamics.step == 0 && s0.dynamics.drift_per_ps.is_none());
        assert_eq!(display_len(&s0), 33 * 33 * 33, "the first evaluation's density");
        w.send(Command::Dyn(DynCommand::Play));
        let t0 = Instant::now();
        let mut snaps = Vec::new();
        while t0.elapsed() < Duration::from_secs(3) {
            snaps.push(next_snapshot(&w));
        }
        assert!(snaps.len() >= 5, "{} snapshots in 3 s", snaps.len());
        let playing: Vec<&Snapshot> = snaps.iter().filter(|s| !s.dynamics.paused).collect();
        assert!(playing.iter().all(|s| s.working));
        assert!(playing.windows(2).all(|p| p[1].dynamics.sim_time >= p[0].dynamics.sim_time));
        assert!(playing.iter().all(|s| s.dynamics.temperature.is_finite()));
        let first = playing.first().expect("a playing snapshot");
        assert!(first.dynamics.drift_per_ps.is_none(), "the drift needs 100 points");
        let last = playing.last().unwrap();
        assert!(last.dynamics.sim_time > first.dynamics.sim_time);
        let rate = last.dynamics.step as f64 / 3.0;
        eprintln!(
            "GPU H₂ 33³ playing: {} snapshots in 3 s, {} steps ({rate:.0} steps/s, {:.1} ms/step EMA), t = {:.2} fs, T = {:.0} K, {:.1} fs/s",
            snaps.len(),
            last.dynamics.step,
            last.dynamics.ms_per_step,
            last.dynamics.sim_time_fs(),
            last.dynamics.temperature,
            last.dynamics.wall_ratio
        );
        let s = snapshot_until(&w, "a drift estimate", |s| s.dynamics.drift_per_ps.is_some());
        assert!(s.dynamics.step >= sim::dynamics::DRIFT_MIN_POINTS as u64);
        assert_eq!(display_len(&s), 33 * 33 * 33, "each step carries its density");
        eprintln!(
            "drift after {} steps: {:+.2e} Ha/ps",
            s.dynamics.step,
            s.dynamics.drift_per_ps.unwrap()
        );
    }

    /// Pause stops time (two later snapshots agree), Step(1) advances exactly one dt.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn pause_freezes_and_step_advances_one_dt() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
        idle(&w);
        w.send(Command::Dyn(DynCommand::Play));
        snapshot_until(&w, "5 steps", |s| s.dynamics.step >= 5);
        w.send(Command::Dyn(DynCommand::Pause));
        let a = idle(&w);
        assert!(a.dynamics.paused);
        std::thread::sleep(Duration::from_millis(300));
        // a harmless command for a fresh snapshot
        w.send(Command::Dyn(DynCommand::SetSpeed(1)));
        let b = next_snapshot(&w);
        assert_eq!(b.dynamics.sim_time, a.dynamics.sim_time, "paused time moved");
        assert_eq!(b.dynamics.step, a.dynamics.step);
        assert_eq!(b.dynamics.scene, a.dynamics.scene);
        w.send(Command::Dyn(DynCommand::Step(1)));
        let c = idle(&w);
        assert!(c.dynamics.paused);
        assert_eq!(c.dynamics.step, a.dynamics.step + 1);
        let dt = units::fs_to_au_time(DEFAULT_DT_FS);
        assert!(
            (c.dynamics.sim_time - a.dynamics.sim_time - dt).abs() < 1e-9 * dt,
            "Δt = {} a.u., dt = {dt}",
            c.dynamics.sim_time - a.dynamics.sim_time
        );
        assert_ne!(c.dynamics.scene, a.dynamics.scene, "the step moved the nuclei");
    }

    /// While playing, a drag is the spring: the nucleus moves a little per snapshot (never more
    /// than 0.1 Bohr — a teleport would jump 0.3 at once) and ends up nearer the target.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn drag_while_playing_uses_the_spring_not_teleport() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
        idle(&w);
        w.send(Command::Dyn(DynCommand::Play));
        let s = snapshot_until(&w, "a step", |s| s.dynamics.step >= 1);
        let start = s.dynamics.scene.nuclei[0].pos;
        let target = [start[0] + 0.3, start[1], start[2]];
        w.send(Command::Dyn(DynCommand::DragStart { index: 0, target }));
        let mut prev = start;
        let mut max_step = 0.0f64;
        let mut n = 0;
        let mut x_max = start[0];
        while n < 60 {
            let s = next_snapshot(&w);
            if s.dynamics.drag.is_none() {
                continue; // a tick from before the DragStart
            }
            assert!(!s.dynamics.paused, "{:?}", s.dynamics.last_error);
            let p = s.dynamics.scene.nuclei[0].pos;
            let d = dist(p, prev);
            max_step = max_step.max(d);
            assert!(d < 0.1, "nucleus 0 jumped {d:.3} Bohr in one snapshot");
            x_max = x_max.max(p[0]);
            prev = p;
            n += 1;
        }
        eprintln!(
            "drag while playing: 60 snapshots, max |Δr| per snapshot {max_step:.4} Bohr, x from {:.3} to max {x_max:.3} (target {:.3})",
            start[0], target[0]
        );
        assert!(x_max - start[0] > 0.1, "the spring did not pull: x max {x_max}");
        w.send(Command::Dyn(DynCommand::DragEnd));
        let s = snapshot_until(&w, "the drag ended", |s| s.dynamics.drag.is_none());
        assert!(!s.dynamics.paused);
    }

    /// Coincident nuclei after a teleport: the provider fails, the run is paused with a message
    /// naming the cause, Play retries and pauses again, and moving the nucleus away recovers.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn provider_failure_pauses_with_message() {
        let _gpu = gpu_lock();
        for (backend, ctx) in backends() {
            let w = SimWorker::spawn(h2_33(), backend, ctx);
            let s = idle(&w);
            let other = s.dynamics.scene.nuclei[1].pos;
            w.send(Command::MoveNucleus { index: 0, pos: other });
            let s = idle(&w);
            let e = s.dynamics.last_error.clone().expect("a provider error");
            assert!(e.contains("coincide"), "{backend:?}: {e}");
            assert!(s.dynamics.paused && s.dynamics.forces_stale);
            w.send(Command::Dyn(DynCommand::Play));
            let s = idle(&w);
            assert!(s.dynamics.paused, "{backend:?}: a failing Play must pause again");
            assert!(s.dynamics.last_error.as_deref().is_some_and(|e| e.contains("coincide")));
            assert_eq!(s.dynamics.step, 0);
            w.send(Command::MoveNucleus {
                index: 0,
                pos: [0.8, 0.0, 0.0],
            });
            let s = idle(&w);
            assert_eq!(s.dynamics.last_error, None, "{backend:?}");
            assert!(!s.dynamics.forces_stale);
            eprintln!("{backend:?}: provider failure → paused with \"{e}\"; recovered");
        }
    }

    /// R11: a drag move every frame (60/s) never aborts a step — every playing tick runs at least
    /// one step (an aborted tick would publish `steps_last_tick == 0`): that is the
    /// discriminating witness (it fails when drag moves abort). The `steps >= 10` in 3 s bound is
    /// a hardware-dependent tripwire (R3; ~49 steps measured here, H₂ 33³ at ~60 ms per step),
    /// not a performance claim.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn drag_move_flood_does_not_stall_progress() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
        idle(&w);
        w.send(Command::Dyn(DynCommand::Play));
        let s = snapshot_until(&w, "a step", |s| s.dynamics.step >= 1);
        let (step0, p0) = (s.dynamics.step, s.dynamics.scene.nuclei[0].pos);
        w.send(Command::Dyn(DynCommand::DragStart { index: 0, target: p0 }));
        let t0 = Instant::now();
        let (mut moves, mut ticks, mut last) = (0u32, 0u32, s);
        while t0.elapsed() < Duration::from_secs(3) {
            let k = moves as f64;
            let target = [p0[0] + 0.1 * (k * 0.1).sin(), p0[1] + 0.05 * (k * 0.07).cos(), p0[2]];
            w.send(Command::Dyn(DynCommand::DragMove { target }));
            moves += 1;
            let frame = Instant::now() + Duration::from_millis(16);
            while let Some(ev) = w.recv_timeout(frame.saturating_duration_since(Instant::now())) {
                match ev {
                    Event::Snapshot(s) => {
                        assert!(!s.dynamics.paused, "{:?}", s.dynamics.last_error);
                        assert!(
                            s.dynamics.steps_last_tick >= 1,
                            "a playing tick ran no step (aborted by a drag move?)"
                        );
                        ticks += 1;
                        last = *s;
                    }
                    Event::Error(e) => panic!("{e}"),
                    Event::Busy(_) | Event::Retired => {}
                }
            }
        }
        let steps = last.dynamics.step - step0;
        eprintln!(
            "drag-move flood: {moves} moves in 3 s, {ticks} ticks, {steps} steps, {:.1} ms/step",
            last.dynamics.ms_per_step
        );
        assert!(moves >= 100, "the flood was only {moves} moves");
        assert!(steps >= 10, "{steps} steps in 3 s under the flood");
    }

    /// R11: Pause abandons a running solve within about one SCF step. The worker is busy with a
    /// cold 97³ H₂O evaluation (≈ 3.6 s); Pause 300 ms in must produce a snapshot with the forces
    /// still stale (the evaluation was abandoned, not finished) — the discriminating witness.
    /// The 1.5 s latency bound is a hardware-dependent tripwire (R3; 190 ms measured). This
    /// aborts the stale re-evaluation after `SetScene`, not a playing step; the predicate is the
    /// same closure, and the step path is covered in `sim` (`cancelled_evaluation_ends_the_tick…`).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn pause_aborts_a_running_solve_promptly() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
        idle(&w);
        let mut water = Scene::preset("h2o").unwrap();
        water.grid_n = 97;
        w.send(Command::SetScene(water));
        std::thread::sleep(Duration::from_millis(300));
        let t = Instant::now();
        w.send(Command::Dyn(DynCommand::Pause));
        // the snapshot of the aborted tick (forces stale), before any snapshot of a finished one
        let s = next_snapshot(&w);
        let latency = t.elapsed();
        eprintln!(
            "Pause during a cold 97³ H₂O evaluation: snapshot {:.0} ms after the command (forces stale: {})",
            latency.as_secs_f64() * 1e3,
            s.dynamics.forces_stale
        );
        assert_eq!(s.dynamics.scene.nuclei.len(), 3);
        assert!(s.dynamics.forces_stale, "the evaluation finished instead of being abandoned");
        assert!(latency < Duration::from_millis(1500), "Pause took {latency:?}");
    }

    /// R13: a paused teleport warm-starts. Nudging one H of the `h2` preset (49³) by 0.02 Bohr
    /// takes fewer readbacks than a cold solve of the displaced geometry on a fresh worker
    /// (`warm < 0.85·cold`, the M3/M4 margin rule: one extra warm SCF step is ≈ +55 readbacks),
    /// no more SCF steps, and lands on the same energy. This is the app-level witness that
    /// `MoveNucleusDiscontinuous` reaches `GpuKs::discontinuity` (warm state kept) rather than
    /// `reset`; readback counts are deterministic per driver.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn moving_a_nucleus_warm_starts() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(crate::scene::h2_r16(), Backend::Gpu, Some(ctx.clone()));
        let cold0 = idle(&w);
        let pos = [0.82, 0.0, 0.0];
        w.send(Command::MoveNucleus { index: 0, pos });
        let warm = idle(&w);
        assert_eq!(warm.dynamics.scene.nuclei[0].pos, pos);
        assert_eq!(warm.dynamics.last_error, None);
        let mut displaced = crate::scene::h2_r16();
        displaced.nuclei[0].pos = pos;
        let w2 = SimWorker::spawn(displaced, Backend::Gpu, Some(ctx));
        let cold = idle(&w2);
        assert_eq!(cold.scene, warm.scene);
        let (wr, cr) = (
            warm.dynamics.eval.readbacks.expect("GPU readbacks"),
            cold.dynamics.eval.readbacks.expect("GPU readbacks"),
        );
        let (ws, cs) = (warm.dynamics.eval.scf_steps, cold.dynamics.eval.scf_steps);
        eprintln!(
            "GPU h2 49³, 0.02 Bohr teleport: warm {wr} readbacks ({ws} steps) vs cold {cr} readbacks ({cs} steps; first cold {} readbacks); E warm {:+.9} cold {:+.9}",
            cold0.dynamics.eval.readbacks.unwrap_or(0),
            warm.dynamics.energy_ks,
            cold.dynamics.energy_ks
        );
        assert!((wr as f64) < 0.85 * cr as f64, "warm {wr} readbacks vs cold {cr}: the teleport did not warm-start");
        assert!(ws <= cs, "warm {ws} steps vs cold {cs}: the orbital lag is back");
        assert!((warm.dynamics.energy_ks - cold.dynamics.energy_ks).abs() < 1e-5);
    }

    /// R12: a teleport that clears the viewer's clamp but not the providers' 3h margin pauses with
    /// the "left the simulation cell" error and keeps the scene (the move itself is accepted).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn a_nucleus_at_the_cell_face_pauses_with_a_message() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
        idle(&w);
        w.send(Command::MoveNucleus { index: 0, pos: [3.6, 0.0, 0.0] });
        let s = idle(&w);
        let e = s.dynamics.last_error.clone().expect("a provider error");
        assert!(e.contains("left the simulation cell"), "{e}");
        assert!(s.dynamics.paused && s.dynamics.forces_stale);
    }

    /// The viewer's mirror: teleports and additions show at once, a drag never moves the mirror
    /// (the spring does, through the snapshots), and the abort/edit classification matches R11.
    #[test]
    fn mirror_moves_for_teleports_but_not_for_drags() {
        let mut s = crate::scene::h2_r16();
        let before = s.clone();
        for c in [
            Command::Dyn(DynCommand::DragStart {
                index: 0,
                target: [3.0, 0.0, 0.0],
            }),
            Command::Dyn(DynCommand::DragMove {
                target: [4.0, 0.0, 0.0],
            }),
            Command::Dyn(DynCommand::DragEnd),
            Command::Dyn(DynCommand::SetDragForce { f_max: 0.5 }),
            Command::Dyn(DynCommand::Play),
            Command::Dyn(DynCommand::Pin(0, true)),
        ] {
            c.apply(&mut s);
            assert!(!c.aborts(), "{c:?} must not abort a step");
        }
        assert_eq!(s, before);
        let mv = Command::MoveNucleus {
            index: 0,
            pos: [1.0, 0.0, 0.0],
        };
        mv.apply(&mut s);
        assert_eq!(s.nuclei[0].pos, [1.0, 0.0, 0.0]);
        for c in [
            mv,
            Command::Dyn(DynCommand::Pause),
            Command::AddNucleus {
                z: 1,
                pos: [0.0; 3],
            },
            Command::RemoveNucleus(0),
            Command::SetCharge(1),
            Command::SetGrid { n: 33, h: 0.3 },
            Command::SetScene(before.clone()),
            Command::Dyn(DynCommand::StartSampling(SamplingPlan::default())),
            Command::Dyn(DynCommand::CancelSampling),
            Command::Dyn(DynCommand::ForgetModel),
            Command::Dyn(DynCommand::LoadModel("x.model".into())),
            Command::EnsureModel { autoload: None, plan: SamplingPlan::default() },
            Command::Shutdown,
        ] {
            assert!(c.aborts(), "{c:?} must abort a step");
        }
        // a sampling pass starting pauses the run (mirrored at once); writing a file does not
        // abort or pause anything
        assert!(Command::Dyn(DynCommand::StartSampling(SamplingPlan::default())).pauses());
        assert!(Command::Dyn(DynCommand::StartSampling(SamplingPlan::default())).is_edit());
        let save = Command::Dyn(DynCommand::SaveModel("x.model".into()));
        assert!(!save.aborts() && !save.pauses() && !save.is_edit());
        // consecutive drag pulls coalesce into the last valid one (review M6)
        let pull = |f| Command::Dyn(DynCommand::SetDragForce { f_max: f });
        assert!(pull(0.2).superseded_by(Some(&pull(0.3))));
        assert!(!pull(0.2).superseded_by(Some(&pull(3.0))), "an invalid next one never drops an accepted change");
        assert!(!pull(0.2).superseded_by(Some(&Command::Dyn(DynCommand::DragEnd))));
        for c in [
            Command::Dyn(DynCommand::SetSpeed(10)),
            Command::Dyn(DynCommand::SetDt(0.3)),
            Command::Dyn(DynCommand::Rewind),
            Command::Dyn(DynCommand::SetThermostat(Some((300.0, 10.0)))),
            Command::Dyn(DynCommand::SetExternalField([0.0, -1e-22, 0.0])),
            Command::Dyn(DynCommand::SetComCorrection(true)),
            Command::Dyn(DynCommand::SetDragForce { f_max: 0.5 }),
        ] {
            assert!(!c.aborts() && !c.is_edit() && !c.pauses(), "{c:?}");
        }
    }

    // ---- the estimated solver (M5 Task 8) ----

    /// A shorter plan than the default (1 reference + 30 perturbations + 12 scan points + a
    /// 40-step burst = 83 solves, a few seconds at 33³ on the GPU, against the default's 273):
    /// enough samples for H₂'s Q′ (29) with ≥ 30 residual degrees of freedom in both kinds.
    fn test_plan() -> SamplingPlan {
        SamplingPlan {
            perturbations: 30,
            hot_burst_steps: 40,
            scan_points_per_pair: 12,
            ..SamplingPlan::default()
        }
    }

    /// The `h2` preset on a 41³ grid (box 10 Bohr) — the estimated tests' system. Not 33³: in
    /// the 8 Bohr box the teacher's own scatter (confinement of the stretched and hot samples,
    /// Task 4's finding) leaves a training energy residual of 2.7e-3 Ha/atom > 1e-3 after this
    /// plan, and the trust rule rightly refuses the model (measured here, 83 samples).
    fn h2_41() -> Scene {
        let mut s = crate::scene::h2_r16();
        s.grid_n = 41;
        s
    }

    fn model(s: &Snapshot) -> &ModelStatus {
        s.model.as_ref().expect("an estimated snapshot carries the model's status")
    }

    /// An estimated worker on `scene` whose sampling pass (the viewer's `EnsureModel`) has
    /// finished with a trusted model; returns it with the last snapshot and the pass's wall time.
    fn taught(scene: Scene, ctx: GpuContext, autosave: Option<PathBuf>) -> (SimWorker, Snapshot, Duration) {
        let w = SimWorker::spawn_with(scene, Backend::Estimated, Some(ctx), Vec::new(), autosave);
        idle(&w);
        let t = Instant::now();
        w.send(Command::EnsureModel { autoload: None, plan: test_plan() });
        snapshot_until(&w, "a running pass", |s| sampling_active(&s.dynamics));
        let s = snapshot_until(&w, "the end of the pass", |s| {
            s.dynamics.sampling.as_ref().is_some_and(|p| p.finished)
        });
        let wall = t.elapsed();
        let m = model(&s);
        assert!(m.fitted, "no model after the pass: {:?}", s.dynamics.notes);
        assert!(m.trusted, "untrusted after the pass: {:?} ({:?})", m.untrusted_reason, m.fit);
        (w, s, wall)
    }

    /// Plays (the worker must already be playing) for `wall` and returns the simulated fs per
    /// wall second over the whole window — Δsim_time/Δwall between the first and the last
    /// playing snapshot, what the viewer's "fs/s simulated (last 2 s)" shows, audits, refits
    /// and publishing included (fix round 1, I3: not one tick's own figure) — with the
    /// snapshots received.
    fn play_rate(w: &SimWorker, wall: Duration) -> (f64, Vec<Snapshot>) {
        let t0 = Instant::now();
        let (mut first, mut last, mut snaps) = (None, None, Vec::new());
        while t0.elapsed() < wall {
            let s = next_snapshot(w);
            assert_eq!(s.dynamics.last_error, None);
            if !s.dynamics.paused {
                let at = (Instant::now(), s.dynamics.sim_time_fs());
                first.get_or_insert(at);
                last = Some(at);
            }
            snaps.push(s);
        }
        let ((w0, t0), (w1, t1)) = (first.expect("a playing snapshot"), last.unwrap());
        let dt = (w1 - w0).as_secs_f64();
        assert!(dt > 0.5 * wall.as_secs_f64(), "playing snapshots spanned only {dt:.2} s");
        ((t1 - t0) / dt, snaps)
    }

    /// Spec §6 / brief test 1. The estimated solver with no model: `EnsureModel` (what the viewer
    /// sends when the radio is switched) starts a pass — the dynamics paused, Play refused, the
    /// progress counting up — that ends in a trusted model at surface epoch ≥ 1; then Play at
    /// 1000 steps per tick: over the first 200 steps more than 80 % are answered by the model,
    /// and the simulated fs per wall second exceed 50× the GPU DFT run's measured in this test.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn enabling_estimated_without_a_model_starts_a_sampling_pass() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        // the DFT reference rate: the same scene and speed on the GPU solver
        let dft = SimWorker::spawn(h2_41(), Backend::Gpu, Some(ctx.clone()));
        idle(&dft);
        dft.send(Command::Dyn(DynCommand::SetSpeed(1000)));
        dft.send(Command::Dyn(DynCommand::Play));
        let (dft_rate, _) = play_rate(&dft, Duration::from_secs(3));
        drop(dft);

        let w = SimWorker::spawn(h2_41(), Backend::Estimated, Some(ctx));
        let s0 = idle(&w);
        assert_eq!((s0.backend, s0.engine), (Backend::Estimated, Backend::Gpu));
        let m0 = model(&s0);
        // the first evaluation was the teacher's (rule 1) and is kept; no refit without a model
        assert!(!m0.fitted && m0.samples == 1 && m0.last_refit.is_none() && s0.dynamics.sampling.is_none(), "{m0:?}");
        assert_eq!(s0.dynamics.source, sim::Source::Fallback, "no model yet: the teacher answers");
        let t = Instant::now();
        w.send(Command::EnsureModel { autoload: None, plan: test_plan() });
        let first = snapshot_until(&w, "a running pass", |s| sampling_active(&s.dynamics));
        assert!(first.dynamics.paused && first.working);
        w.send(Command::Dyn(DynCommand::Play));
        let refused = snapshot_until(&w, "the refused Play", |s| s.dynamics.last_rejection.is_some());
        assert!(refused.dynamics.last_rejection.as_deref().unwrap().contains("sampling pass is running"));
        assert!(refused.dynamics.paused);
        // progress counts up, monotonically, to the end of the pass
        let mut done = vec![first.dynamics.sampling.as_ref().unwrap().done];
        let end = loop {
            let s = next_snapshot(&w);
            let p = s.dynamics.sampling.as_ref().expect("the pass's progress");
            assert!(p.done >= *done.last().unwrap(), "{} after {:?}", p.done, done);
            if p.done != *done.last().unwrap() {
                done.push(p.done);
            }
            if p.finished {
                break s;
            }
        };
        assert!(done.len() >= 3, "progress seen at {done:?} only");
        let pass = t.elapsed();
        let p = end.dynamics.sampling.clone().unwrap();
        assert!(!p.cancelled && p.done == p.total, "{p:?}");
        let m = model(&end).clone();
        assert!(m.fitted && m.trusted, "{m:?}");
        assert!(m.epoch >= 1 && end.dynamics.model_epoch >= 1, "epoch {}", m.epoch);
        eprintln!(
            "pass: {}/{} jobs in {:.1} s (skipped {}, failed {}), {} samples, fit {:?}, residuals {:?}, epoch {}",
            p.done,
            p.total,
            pass.as_secs_f64(),
            p.skipped,
            p.failed,
            m.samples,
            m.fit,
            m.residuals,
            m.epoch
        );

        w.send(Command::Dyn(DynCommand::SetSpeed(1000)));
        w.send(Command::Dyn(DynCommand::Play));
        let start_step = end.dynamics.step;
        // the default audit cadence (FallbackPolicy::default, 2000 — S14's interim value)
        let (rate, played) = play_rate(&w, Duration::from_secs(3));
        let at200 = played
            .iter()
            .find(|s| s.dynamics.step >= start_step + 200)
            .expect("200 steps in 3 s");
        let src = &at200.dynamics.history_source;
        assert_eq!(src.len(), at200.dynamics.history.len());
        let first200 = &src[src.len() - (at200.dynamics.step - start_step) as usize..][..200];
        let surrogate = first200.iter().filter(|&&x| x == sim::Source::Surrogate).count();
        let m = model(played.last().unwrap());
        // the model alone, for the record (audits off: unsafe, not the default)
        w.send(Command::SetAudit(0));
        let (rate_off, _) = play_rate(&w, Duration::from_secs(2));
        // pacing (fix round 1): one step per 16 ms tick is 0.25 fs / 16 ms ≈ 15.6 fs/s — a
        // watchable run — not one step per tick back to back (thousands of fs/s)
        w.send(Command::Dyn(DynCommand::SetSpeed(1)));
        let (rate_1, _) = play_rate(&w, Duration::from_secs(2));
        let paced = units::au_time_to_fs(units::fs_to_au_time(DEFAULT_DT_FS)) / (TICK_BUDGET_MS * 1e-3);
        eprintln!("speed 1 (paced): {rate_1:.1} fs/s, the pacing's {paced:.1}");
        assert!(rate_1 < 1.2 * paced && rate_1 > 0.8 * paced, "speed 1: {rate_1:.1} fs/s vs paced {paced:.1}");
        eprintln!(
            "estimated H₂ 41³: {surrogate}/200 surrogate steps (fallbacks {}); Δsim/Δwall {rate:.0} fs/s at audit {} vs DFT {dft_rate:.2} fs/s (×{:.0}); audits off {rate_off:.0} fs/s (×{:.0})",
            first200.iter().filter(|&&x| x == sim::Source::Fallback).count(),
            m.audit_every,
            rate / dft_rate,
            rate_off / dft_rate
        );
        assert_eq!(m.audit_every, FallbackPolicy::default().audit_every);
        assert!(surrogate * 10 > 200 * 8, "{surrogate}/200 steps answered by the model");
        assert!(rate > 50.0 * dft_rate, "{rate:.0} fs/s vs DFT {dft_rate:.2}");
    }

    /// Brief test 2: a geometry the model was not taught is answered by the teacher and the
    /// snapshot says so — teleport the trained H₂ to R = 0.8 Bohr (compressed below the pass's
    /// 1.12 Bohr scan start; not R = 4, which in the 10 Bohr box puts 1e-4 e/Bohr³ on the faces
    /// and is box-affected since Task 9): the next evaluation is a `Fallback` with the model
    /// consulted (uncertainty and teacher gap present, the note naming the reason), the sample
    /// is kept and the worker refits; the following evaluation at the same geometry is
    /// reported (the model learning, not asserted).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn fallbacks_are_visible_in_snapshots() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let (w, trained, _) = taught(h2_41(), ctx, None);
        let n0 = model(&trained).samples;
        w.send(Command::MoveNucleus { index: 0, pos: [0.4, 0.0, 0.0] });
        w.send(Command::MoveNucleus { index: 1, pos: [-0.4, 0.0, 0.0] });
        let s = snapshot_until(&w, "the evaluation at R = 0.8", |s| {
            !s.dynamics.forces_stale && s.dynamics.scene.nuclei[0].pos[0] == 0.4 && s.dynamics.scene.nuclei[1].pos[0] == -0.4
        });
        let d = &s.dynamics;
        let m = model(&s);
        let note = d.eval.notes.iter().find(|n| n.starts_with("surrogate")).cloned().unwrap_or_default();
        eprintln!(
            "R = 0.8: {:?}, uncertainty {:?} (bound {:.1e}), teacher gap {:?}, note \"{note}\"; samples {n0} → {}, last refit {:?}",
            d.source, d.uncertainty, m.force_bound, d.teacher_gap, m.samples, m.last_refit
        );
        assert_eq!(d.source, sim::Source::Fallback);
        assert!(d.teacher_gap.is_some(), "a trusted model's fallback is booked on its surface");
        assert!(d.uncertainty.is_some(), "the model was consulted");
        assert!(note.contains("teacher answered"), "{:?}", d.eval.notes);
        assert_eq!(m.samples, n0 + 1, "the fallback's answer is a new sample");
        assert!(
            m.last_refit.as_ref().is_some_and(|r| r.result.is_ok()),
            "the worker refits after a kept sample: {:?}",
            m.last_refit
        );
        assert!(!m.pending_refit);
        // the next tick re-evaluates at the same positions on the refitted surface
        w.send(Command::Dyn(DynCommand::SetSpeed(1)));
        let after = idle(&w);
        eprintln!(
            "R = 0.8 after the refit: {:?}, uncertainty {:?}, teacher gap {:?}, epoch {} → {}",
            after.dynamics.source,
            after.dynamics.uncertainty,
            after.dynamics.teacher_gap,
            m.epoch,
            model(&after).epoch
        );
    }

    /// The cloud in estimated mode (M5b): a background audit publishes its density as the
    /// snapshot's side density, labelled Audit and stamped with the time of the geometry it
    /// solved (≤ the snapshot's); the worker fetches the H atom's profile on the scene's grid
    /// for the superposition — a real isolated-atom solve whose shells hold ≈ 1 electron — and
    /// with audits off a "refresh every 1 s" brings display-only densities (never a sample: the
    /// `sim` side-density test).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn an_audit_and_a_refresh_publish_their_densities_and_the_atoms_their_profiles() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let (w, trained, _) = taught(h2_41(), ctx, None);
        assert!(model(&trained).audits_background, "the GPU teacher audits in the background");
        w.send(Command::SetCloud { superposition: true, refresh_s: None });
        w.send(Command::SetAudit(50));
        w.send(Command::Dyn(DynCommand::SetSpeed(20)));
        w.send(Command::Dyn(DynCommand::Play));
        let s = snapshot_until(&w, "an audit density and the H profile", |s| {
            s.dynamics.side_display.as_ref().is_some_and(|d| d.kind == sim::SideKind::Audit) && !s.atom_profiles.is_empty()
        });
        let sd = s.dynamics.side_display.clone().unwrap();
        let p = &s.atom_profiles[0];
        eprintln!(
            "audit density solved at t = {:.2} fs, snapshot at {:.2} fs ({:?}); H profile: {} shells to {:.2} Bohr, ρ(0) {:.4}, electrons {:.4}",
            sd.t_fs(), s.dynamics.sim_time_fs(), sd.display, p.r.len(), p.reach(), p.rho[0], p.electrons
        );
        assert!(sd.t <= s.dynamics.sim_time && sd.t > 0.0 && sd.positions.len() == 2);
        assert!(matches!(&sd.display, sim::Display::Gpu { cell, .. } if cell.n == 41));
        assert_eq!((p.z, p.n, p.h), (1, 41, 0.25));
        assert!((p.electrons - 1.0).abs() < 2e-2, "the H atom's shells hold {} electrons", p.electrons);
        // refreshes, audits off: display only
        w.send(Command::SetAudit(0));
        w.send(Command::SetCloud { superposition: true, refresh_s: Some(1.0) });
        let r = snapshot_until(&w, "a refresh density", |s| {
            s.dynamics.side_display.as_ref().is_some_and(|d| d.kind == sim::SideKind::Refresh)
        });
        let rd = r.dynamics.side_display.clone().unwrap();
        eprintln!("refresh density solved at t = {:.2} fs, snapshot at {:.2} fs", rd.t_fs(), r.dynamics.sim_time_fs());
        assert!(rd.t <= r.dynamics.sim_time && rd.t >= sd.t);
        w.send(Command::Dyn(DynCommand::Pause));
    }

    /// Brief test 3: Save model writes the sidecar; a new estimated worker on the same scene
    /// restores it through `EnsureModel`'s autoload — samples and fit equal, trusted, no pass
    /// started, the first step answered by the model; a worker on another grid refuses it (and
    /// starts no pass: the panel offers to forget it). Retiring an estimated worker writes the
    /// autosave the viewer carries across solver switches.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn model_sidecar_saves_and_loads_with_the_scene() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let dir = std::env::temp_dir().join(format!("atomic-sandbox-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (saved, autosave) = (dir.join("h2.model"), dir.join("autosave.model"));
        let _ = std::fs::remove_file(&autosave);
        let (a, trained, _) = taught(h2_41(), ctx.clone(), Some(autosave.clone()));
        let ma = model(&trained).clone();
        a.send(Command::Dyn(DynCommand::SaveModel(saved.clone())));
        let s = snapshot_until(&a, "the save", |s| {
            s.dynamics.last_rejection.is_some() || s.dynamics.notes.iter().any(|n| n.contains("model saved to"))
        });
        assert_eq!(s.dynamics.last_rejection, None);
        eprintln!("saved: {:?}", s.dynamics.notes.last());
        assert!(a.retire(Duration::from_secs(10)), "the worker did not retire");
        assert!(saved.exists() && autosave.exists());

        for path in [&saved, &autosave] {
            let b = SimWorker::spawn(h2_41(), Backend::Estimated, Some(ctx.clone()));
            idle(&b);
            b.send(Command::EnsureModel { autoload: Some(path.clone()), plan: test_plan() });
            let s = snapshot_until(&b, "the loaded model", |s| model(s).fitted);
            let mb = model(&s);
            assert!(s.dynamics.sampling.is_none(), "a pass started despite the model");
            assert_eq!((mb.samples, mb.fit, mb.trusted), (ma.samples, ma.fit, true));
            assert!(mb.epoch >= 1);
            assert!(s.dynamics.notes.iter().any(|n| n.contains("model loaded from")));
            b.send(Command::Dyn(DynCommand::Step(1)));
            let s = snapshot_until(&b, "one step", |s| s.dynamics.step == 1 && !s.working);
            assert_eq!(s.dynamics.source, sim::Source::Surrogate, "{:?}", s.dynamics.eval.notes);
            eprintln!(
                "loaded {}: {} samples, fit {:?}, epoch {}; first step {:?} in {:.3} ms",
                path.display(),
                mb.samples,
                mb.fit,
                mb.epoch,
                s.dynamics.source,
                s.dynamics.ms_per_step
            );
        }

        let mut other = h2_41();
        other.grid_n = 33;
        // the refusing worker is given the file as ITS autosave too: leaving it must not delete
        // the other-provenance model (fix round 1, I1 scenario 2)
        let bytes = std::fs::read(&saved).unwrap();
        let c = SimWorker::spawn_with(other, Backend::Estimated, Some(ctx), Vec::new(), Some(saved.clone()));
        idle(&c);
        c.send(Command::EnsureModel { autoload: Some(saved.clone()), plan: test_plan() });
        let s = snapshot_until(&c, "the refused load", |s| model(s).autoload_refused.is_some());
        let why = model(&s).autoload_refused.clone().unwrap();
        assert!(why.contains("41³") && why.contains("33³"), "{why}");
        assert!(!model(&s).fitted && s.dynamics.sampling.is_none(), "a refused autoload starts nothing");
        assert!(c.retire(Duration::from_secs(10)));
        assert_eq!(std::fs::read(&saved).unwrap(), bytes, "a worker that refused the autosave rewrote or removed it");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// R11 for the sampling pass: a Pause queued while a pass solves abandons the solve in
    /// flight (the job stays queued: the pass is not cancelled and carries on), and Cancel ends
    /// the pass, both within a tripwire bound — on H₂O at 97³, where one reference solve (the O
    /// atom, cold, three multiplicities) takes seconds, so a prompt snapshot is the abort.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn pause_and_cancel_abort_a_sampling_solve_promptly() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let mut water = Scene::preset("h2o").unwrap();
        water.grid_n = 97;
        let w = SimWorker::spawn(water, Backend::Estimated, Some(ctx));
        let s0 = idle(&w);
        // Task 9 item 4: the automatic first pass would take far longer than its 2-minute target
        // here, so it waits for "Start anyway" (an offer, no pass) …
        w.send(Command::EnsureModel { autoload: None, plan: SamplingPlan::default() });
        let offered = snapshot_until(&w, "the pass offer", |s| model(s).pass_offer.is_some());
        let o = model(&offered).pass_offer.clone().unwrap();
        eprintln!(
            "H₂O 97³: first evaluation {} ms; offer {} solves ≈ {:.0} s at ≈ {:.2} s a solve (the default plan: {} solves ≈ {:.0} s)",
            s0.dynamics.eval.solve_ms, o.solves, o.eta_s, o.per_solve_s, o.full_solves, o.full_eta_s
        );
        assert!(offered.dynamics.sampling.is_none() && o.eta_s > PASS_TARGET_S, "{o:?}");
        // … which the panel's button sends as a plain StartSampling
        w.send(Command::Dyn(DynCommand::StartSampling(o.plan.clone())));
        let s = snapshot_until(&w, "a running pass", |s| sampling_active(&s.dynamics));
        assert!(model(&s).pass_offer.is_none());
        let done0 = s.dynamics.sampling.as_ref().unwrap().done;
        std::thread::sleep(Duration::from_millis(400));
        // the snapshot after the Pause is then the aborted tick's (the round that applies the
        // Pause resumes the pass with a whole solve before it publishes)
        while w.try_recv().is_some() {}
        let t = Instant::now();
        w.send(Command::Dyn(DynCommand::Pause));
        let s = next_snapshot(&w);
        let pause_latency = t.elapsed();
        let p = s.dynamics.sampling.clone().unwrap();
        assert!(!p.cancelled && !p.finished, "Pause must not end the pass: {p:?}");
        // and the pass carries on: the abandoned job is re-queued and solved
        let resumed = snapshot_until(&w, "the pass resuming after the Pause", |s| {
            s.dynamics.sampling.as_ref().is_some_and(|q| q.done > p.done)
        });
        let resumed_done = resumed.dynamics.sampling.as_ref().unwrap().done;
        std::thread::sleep(Duration::from_millis(400));
        let t = Instant::now();
        w.send(Command::Dyn(DynCommand::CancelSampling));
        let s = snapshot_until(&w, "the cancelled pass", |s| s.dynamics.sampling.as_ref().is_some_and(|p| p.finished));
        let cancel_latency = t.elapsed();
        let p = s.dynamics.sampling.clone().unwrap();
        eprintln!(
            "H₂O 97³ pass: Pause → snapshot in {:.0} ms (done {done0}), resumed to {resumed_done}, Cancel → cancelled pass in {:.0} ms ({}/{} jobs, stage {})",
            pause_latency.as_secs_f64() * 1e3,
            cancel_latency.as_secs_f64() * 1e3,
            p.done,
            p.total,
            p.stage
        );
        assert!(p.cancelled);
        assert!(pause_latency < Duration::from_millis(1500), "Pause took {pause_latency:?}");
        assert!(cancel_latency < Duration::from_millis(1500), "Cancel took {cancel_latency:?}");
    }

    /// Teleports the (paused) H₂ to separation `r` along x and returns the idle snapshot.
    fn teleport_h2(w: &SimWorker, r: f64) -> Snapshot {
        w.send(Command::MoveNucleus { index: 0, pos: [r / 2.0, 0.0, 0.0] });
        w.send(Command::MoveNucleus { index: 1, pos: [-r / 2.0, 0.0, 0.0] });
        snapshot_until(&w, &format!("the evaluation at R = {r}"), |s| {
            !s.working && !s.dynamics.forces_stale && (s.dynamics.scene.nuclei[0].pos[0] - r / 2.0).abs() < 1e-12
        })
    }

    /// Fix round 3 (the user's "once it falls back it never returns"), the regression, as the
    /// app sees it. Root cause: one sample the model cannot fit — H₂ pulled to R = 8 Bohr, whose
    /// teacher energy is +0.3 Ha above two atoms because at 41³ its nuclei sit 1 Bohr from the
    /// box faces (confinement; fix round 4 corrected the round-3 attribution) — failed the
    /// training-residual trust rule at every later refit, every step after it was a rule-4
    /// fallback, and every fallback kept its sample (the user's 3540). Reproduced before the fix:
    /// 200 steps all Fallback, refits Ok, trusted never. Now (fix round 4) that answer carries
    /// the teacher's "box too small" warning and is not trained on at all (quarantine, ruling
    /// S15, is the general remedy: the `sim` tests for a geometry genuinely unrepresentable);
    /// at R = 3.3 the model answers again within ≤ 20 fallback steps and ≤ 2 refits.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn after_a_far_drag_the_surrogate_answers_again() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let (w, trained, _) = taught(h2_41(), ctx, None);
        let n0 = model(&trained).samples;
        let far = teleport_h2(&w, 8.0);
        let m = model(&far);
        eprintln!("R = 8: {:?}, trusted {}, samples {n0} → {}, notes {:?}", far.dynamics.source, m.trusted, m.samples, far.dynamics.eval.notes);
        assert_eq!(far.dynamics.source, sim::Source::Fallback);
        assert!(far.dynamics.eval.notes.iter().any(|n| n.contains("box-affected")), "{:?}", far.dynamics.eval.notes);
        assert!(m.trusted && m.samples == n0, "the box-affected answer must not be trained on: {m:?}");
        let s = teleport_h2(&w, 3.3);
        let refits0 = model(&s).refits;
        w.send(Command::Dyn(DynCommand::Play));
        let (mut fallbacks, mut seen) = (usize::from(s.dynamics.source == sim::Source::Fallback), 0u64);
        let back = loop {
            let s = next_snapshot(&w);
            if s.dynamics.step > seen {
                let src = &s.dynamics.history_source;
                let new = (s.dynamics.step - seen) as usize;
                fallbacks += src[src.len().saturating_sub(new)..].iter().filter(|&&x| x == sim::Source::Fallback).count();
                seen = s.dynamics.step;
            }
            let m = model(&s);
            if s.dynamics.source == sim::Source::Surrogate && m.trusted {
                break s;
            }
            assert!(fallbacks <= 20 && m.refits - refits0 <= 2, "still falling back: {fallbacks} fallback steps, {} refits, trusted {}, reason {:?}", m.refits - refits0, m.trusted, m.untrusted_reason);
        };
        let m = model(&back);
        eprintln!(
            "R = 3.3 after the far drag: Surrogate again at step {} after {fallbacks} fallback steps and {} refits; {} samples",
            back.dynamics.step,
            m.refits - refits0,
            m.samples
        );
    }

    /// Fix round 4: a far geometry in a small box (R = 8 at 41³: the density reaches the faces)
    /// is answered by DFT and never trained on — the model stays trusted, R = 1.6 stays the
    /// model's, and R = 8 again is a fallback, the store unchanged.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn a_far_geometry_in_a_small_box_falls_back_and_is_not_trained_on() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let (w, trained, _) = taught(h2_41(), ctx, None);
        let n0 = model(&trained).samples;
        let far = teleport_h2(&w, 8.0);
        let gap = far.dynamics.teacher_gap.unwrap_or(f64::NAN);
        let warned = far.dynamics.eval.warnings.clone();
        let near = teleport_h2(&w, 1.6);
        assert_eq!(near.dynamics.source, sim::Source::Surrogate, "{:?}", near.dynamics.eval.notes);
        let again = teleport_h2(&w, 8.0);
        let m = model(&again);
        eprintln!("R = 8: teacher − model {gap:+.3e} Ha, teacher warnings {warned:?}; R = 1.6 {:?}; R = 8 again {:?}; samples {n0} → {}, quarantined {}", near.dynamics.source, again.dynamics.source, m.samples, m.quarantined);
        assert_eq!(again.dynamics.source, sim::Source::Fallback);
        assert!(warned.iter().any(|w| w.contains("box too small")), "{warned:?}");
        assert!(m.trusted && m.samples == n0 && m.quarantined == 0, "{m:?}");
    }

    /// Task 9 item 4: the automatic first pass keeps to its 2-minute target — the panel's
    /// (default) plan when it fits, a scaled-down plan when that fits, and an offer to confirm
    /// when even the fewest solves that can earn trust take longer (the user's 12-minute H₂O
    /// pass at 57³: 2.4 s a solve).
    #[test]
    fn the_first_pass_keeps_to_its_time_target_or_asks() {
        let h2 = h2_41().to_sim().unwrap();
        let plan = SamplingPlan::default();
        assert_eq!(first_pass(&h2, plan.clone(), None), FirstPass::Start(plan.clone()), "no cost known: the plan as is");
        assert_eq!(first_pass(&h2, plan.clone(), Some(0.12)), FirstPass::Start(plan.clone()), "273 × 0.12 s = 33 s");
        match first_pass(&h2, plan.clone(), Some(0.9)) {
            FirstPass::Scaled(p, note) => {
                let n = p.solves(&h2).unwrap();
                eprintln!("H₂ at 0.9 s a solve: {n} solves ({note})");
                assert!(n as f64 * 0.9 <= PASS_TARGET_S * 1.05 && n >= SamplingPlan::min_samples_for_trust(&h2), "{n}");
                assert!(note.contains("scaled to"));
            }
            other => panic!("{other:?}"),
        }
        let mut water = Scene::preset("h2o").unwrap();
        water.grid_n = 57;
        let water = water.to_sim().unwrap();
        match first_pass(&water, plan.clone(), Some(2.4)) {
            FirstPass::Offer(o) => {
                eprintln!("H₂O 57³ at 2.4 s a solve: offer {} solves ≈ {:.0} s (the plan: {} solves ≈ {:.0} s)", o.solves, o.eta_s, o.full_solves, o.full_eta_s);
                assert!(o.eta_s > PASS_TARGET_S && o.full_eta_s > o.eta_s);
                assert!(o.solves >= SamplingPlan::min_samples_for_trust(&water));
            }
            other => panic!("{other:?}"),
        }
    }

    /// Diagnostic (Task 9 item 4): the per-solve cost of a sampling pass against the scene's cold
    /// first evaluation, H₂O at `DIAG_GRID` (57): stage by stage for 90 s.
    #[test]
    #[ignore = "diagnostic: cargo test -p app --release diag_pass_rate -- --ignored --nocapture"]
    fn diag_pass_rate() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let mut water = Scene::preset(&std::env::var("DIAG_PRESET").unwrap_or("h2o".into())).unwrap();
        water.grid_n = std::env::var("DIAG_GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(57);
        let w = SimWorker::spawn(water, Backend::Estimated, Some(ctx));
        let s0 = idle(&w);
        eprintln!("first (cold) evaluation: {} ms, {} SCF steps", s0.dynamics.eval.solve_ms, s0.dynamics.eval.scf_steps);
        w.send(Command::Dyn(DynCommand::StartSampling(SamplingPlan::default())));
        let t0 = Instant::now();
        let (mut stage, mut at, mut done_at) = ("", t0, 0usize);
        let mut last = None;
        while t0.elapsed() < Duration::from_secs(90) {
            let s = next_snapshot(&w);
            let Some(p) = s.dynamics.sampling.clone() else { continue };
            last = Some(s);
            if p.stage != stage {
                if !stage.is_empty() {
                    let n = p.done - done_at;
                    eprintln!("  {stage}: {n} jobs in {:.1} s ({:.2} s each)", at.elapsed().as_secs_f64(), at.elapsed().as_secs_f64() / n.max(1) as f64);
                }
                (stage, at, done_at) = (p.stage, Instant::now(), p.done);
            }
            if p.finished { break; }
        }
        let s = last.expect("a pass snapshot");
        let p = s.dynamics.sampling.clone().unwrap();
        eprintln!("  {stage} (partial): {} jobs in {:.1} s; overall {}/{} in {:.1} s; worker estimate {:?} s a solve, ETA {:?} s",
                  p.done - done_at, at.elapsed().as_secs_f64(), p.done, p.total, t0.elapsed().as_secs_f64(), model(&s).solve_s, model(&s).pass_eta_s);
    }

    /// Fix round 2: a DragMove flood while playing paced (an analytic Morse H₂, speed 1: one
    /// 0.25 fs step per 16 ms tick ≈ 15.6 fs/s) neither speeds the run up — every command used to
    /// run a full tick — nor pushes the pacing schedule ahead of the clock, which froze the run for
    /// seconds after the flood ended. Commands every ~4 ms for 2 s; during the flood sim time
    /// advances at the paced rate (within 0.7–1.3×), and after the last command the next step
    /// arrives within 50 ms (a tripwire bound: one budget plus scheduling).
    #[test]
    fn a_command_flood_neither_speeds_up_nor_stalls_a_paced_run() {
        let w = SimWorker::spawn_test(h2_33(), Box::new(|| Box::new(sim::MorseDimer::h2())));
        idle(&w);
        w.send(Command::Dyn(DynCommand::SetSpeed(1)));
        w.send(Command::Dyn(DynCommand::Play));
        let s = snapshot_until(&w, "a step", |s| s.dynamics.step >= 1);
        let p0 = s.dynamics.scene.nuclei[0].pos;
        w.send(Command::Dyn(DynCommand::DragStart { index: 0, target: p0 }));
        let paced = DEFAULT_DT_FS / (TICK_BUDGET_MS * 1e-3);
        let (t0, sim0) = (Instant::now(), s.dynamics.sim_time_fs());
        let (mut moves, mut last_sim) = (0u32, sim0);
        while t0.elapsed() < Duration::from_secs(2) {
            w.send(Command::Dyn(DynCommand::DragMove { target: p0 }));
            moves += 1;
            std::thread::sleep(Duration::from_millis(4));
            while let Some(ev) = w.try_recv() {
                match ev {
                    Event::Snapshot(s) => last_sim = s.dynamics.sim_time_fs(),
                    Event::Error(e) => panic!("{e}"),
                    _ => {}
                }
            }
        }
        let flood_wall = t0.elapsed().as_secs_f64();
        let flood_rate = (last_sim - sim0) / flood_wall;
        // the last command, then the first step after it
        w.send(Command::Dyn(DynCommand::DragEnd));
        let t_end = Instant::now();
        let ended = snapshot_until(&w, "the DragEnd", |s| s.dynamics.drag.is_none()).dynamics.sim_time_fs();
        let s = snapshot_until(&w, "a step after the flood", |s| s.dynamics.sim_time_fs() > ended);
        let resume = t_end.elapsed();
        eprintln!(
            "command flood: {moves} DragMoves in {flood_wall:.2} s, sim rate {flood_rate:.1} fs/s (paced {paced:.1}); next step {:.1} ms after the last command (t = {:.2} fs)",
            resume.as_secs_f64() * 1e3,
            s.dynamics.sim_time_fs()
        );
        assert!(moves >= 200, "only {moves} commands");
        assert!(flood_rate < 1.3 * paced && flood_rate > 0.7 * paced, "flood rate {flood_rate:.1} fs/s vs paced {paced:.1}");
        assert!(resume < Duration::from_millis(50), "the run resumed {resume:?} after the flood");
    }

    /// The wheel's pull reaches the driver while playing (analytic Morse H₂): the snapshot
    /// reports it, steps keep coming (`SetDragForce` is not an aborting command), it outlives the
    /// drag, and an out-of-range value is refused without changing it.
    #[test]
    fn the_drag_pull_reaches_the_driver_while_playing() {
        let w = SimWorker::spawn_test(h2_33(), Box::new(|| Box::new(sim::MorseDimer::h2())));
        idle(&w);
        w.send(Command::Dyn(DynCommand::Play));
        let s = snapshot_until(&w, "a step", |s| s.dynamics.step >= 1);
        let p0 = s.dynamics.scene.nuclei[0].pos;
        w.send(Command::Dyn(DynCommand::DragStart { index: 0, target: [p0[0] - 2.0, p0[1], p0[2]] }));
        w.send(Command::Dyn(DynCommand::SetDragForce { f_max: 0.5 }));
        let s = snapshot_until(&w, "the pull", |s| s.dynamics.drag_force_max == 0.5);
        let step = s.dynamics.step;
        let s = snapshot_until(&w, "steps after the pull", |s| s.dynamics.step > step + 3);
        assert!(s.dynamics.last_rejection.is_none() && s.dynamics.last_error.is_none() && !s.dynamics.paused);
        w.send(Command::Dyn(DynCommand::DragEnd));
        let s = snapshot_until(&w, "the drag's end", |s| s.dynamics.drag.is_none());
        assert_eq!(s.dynamics.drag_force_max, 0.5, "the pull outlives the drag");
        w.send(Command::Dyn(DynCommand::SetDragForce { f_max: 3.0 }));
        let s = snapshot_until(&w, "the refusal", |s| s.dynamics.last_rejection.is_some());
        assert_eq!(s.dynamics.drag_force_max, 0.5);
    }

    /// Fix round 1, I1 scenario 1: an estimated worker shut down before its `EnsureModel` ran
    /// (a quick radio toggle during the uncancellable first evaluation: `Inbox::take` sees
    /// `[EnsureModel, Shutdown]` and stops) does not own the session autosave and must leave it
    /// as it found it. Before the fix it found no model and removed the file.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn a_worker_shut_down_before_ensure_model_leaves_the_autosave_alone() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let path = std::env::temp_dir().join(format!("atomic-sandbox-test-{}-own.model", std::process::id()));
        std::fs::write(&path, b"the session's model").unwrap();
        let w = SimWorker::spawn_with(h2_41(), Backend::Estimated, Some(ctx), Vec::new(), Some(path.clone()));
        // both queued while the worker is still in `Dynamics::new`'s cold evaluation
        w.send(Command::EnsureModel { autoload: Some(path.clone()), plan: test_plan() });
        assert!(w.retire(Duration::from_secs(30)), "the worker did not retire");
        assert_eq!(std::fs::read(&path).unwrap(), b"the session's model", "the autosave was rewritten or removed");
        let _ = std::fs::remove_file(&path);
    }

    /// Plays (speed 1: one step per snapshot) until `steps` more steps have run; the snapshots.
    fn play_steps(w: &SimWorker, steps: u64) -> Vec<Snapshot> {
        let s0 = snapshot_until(w, "a playing step", |s| !s.dynamics.paused && s.dynamics.step > 0);
        let end = s0.dynamics.step + steps;
        let mut seen = vec![s0];
        loop {
            let s = next_snapshot(w);
            let done = s.dynamics.step >= end || s.dynamics.last_error.is_some();
            seen.push(s);
            if done {
                return seen;
            }
        }
    }

    /// Back at the centre at the equilibrium bond (R = 1.8 Bohr, at rest) and playing: the
    /// model answers within ≤ 50 steps, and the first audit dispatched after the return (the
    /// test's cadence: every 100 answered steps) passes with |teacher − model| < 1e-4 Ha. Returns
    /// (steps to the first surrogate answer, that audit's gap).
    fn back_at_equilibrium(w: &SimWorker) -> (u64, f64) {
        w.send(Command::Dyn(DynCommand::Pause));
        idle(w);
        let start = teleport_h2(w, 1.8).dynamics.step;
        w.send(Command::Dyn(DynCommand::Play));
        let first = snapshot_until(w, "a surrogate answer at equilibrium", |s| {
            assert!(s.dynamics.step <= start + 50, "no surrogate answer within 50 steps of the return: {:?}, {:?}", model(s).untrusted_reason, s.dynamics.eval.notes);
            s.dynamics.step > start && s.dynamics.source == sim::Source::Surrogate && model(s).trusted
        });
        snapshot_until(w, "an audit dispatched after the return", |s| s.dynamics.eval.notes.iter().any(|n| n.contains("surrogate audit dispatched")));
        let judged = snapshot_until(w, "its verdict", |s| !model(s).audit_in_flight && model(s).last_audit.is_some());
        let a = model(&judged).last_audit.clone().unwrap();
        eprintln!("back at equilibrium: Surrogate after {} steps; next audit {a:?}", first.dynamics.step - start);
        assert_eq!(a.verdict, sim::surrogate_ks::AuditVerdict::Passed, "{a:?}");
        assert!(a.gap.abs() < 1e-4, "the audit at equilibrium: teacher − model {:+.2e} Ha", a.gap);
        (first.dynamics.step - start, a.gap)
    }

    /// Task 9 item 1, the reproduction of the user's report as the brief poses it: the taught
    /// H₂ (41³) driven hot — 12 random teleports (bond 0.9–5 Bohr, anywhere in the cell, atoms
    /// ≥ 1 Bohr from the faces), 12 steps each — then back near equilibrium. On 6c393c3 the far
    /// answers near the faces (box confinement ≈ 50 Ha per e/Bohr³ of face density: 1e-2–3e-2 Ha
    /// at the 3e-4–9e-4 e/Bohr³ they had, all under the old 1e-3 box rule) were trained on, the
    /// refits quarantined them one by one up to the 10 % cap, and the model ended untrusted
    /// ("max training energy residual 2.0e-2 > 1.0e-3 Ha/atom"): 500 steps at equilibrium, 100 %
    /// DFT. Now (box rule 2e-5) they are used for their steps and not trained on, and at
    /// equilibrium the model answers again within 50 steps with the next audit < 1e-4 Ha.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn after_a_hot_excursion_the_model_answers_at_equilibrium_again() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let (w, trained, _) = taught(h2_41(), ctx, None);
        let n0 = model(&trained).samples;
        w.send(Command::SetAudit(100));
        w.send(Command::Dyn(DynCommand::SetSpeed(1)));
        let half = 0.5 * (41 - 1) as f64 * 0.25;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut fallbacks, mut steps) = (0usize, 0usize);
        for _ in 0..12 {
            let r = 0.9 + 4.1 * rnd();
            let (ct, ph) = (2.0 * rnd() - 1.0, std::f64::consts::TAU * rnd());
            let st = (1.0 - ct * ct).sqrt();
            let u = [st * ph.cos(), st * ph.sin(), ct];
            let span = half - 2.5;
            let c = [span * (2.0 * rnd() - 1.0), span * (2.0 * rnd() - 1.0), span * (2.0 * rnd() - 1.0)];
            let a: [f64; 3] = std::array::from_fn(|k| (c[k] + 0.5 * r * u[k]).clamp(1.0 - half, half - 1.0));
            let b: [f64; 3] = std::array::from_fn(|k| (c[k] - 0.5 * r * u[k]).clamp(1.0 - half, half - 1.0));
            w.send(Command::Dyn(DynCommand::Pause));
            idle(&w);
            w.send(Command::MoveNucleus { index: 0, pos: a });
            w.send(Command::MoveNucleus { index: 1, pos: b });
            let s = snapshot_until(&w, "the teleport", |s| !s.working && !s.dynamics.forces_stale && s.dynamics.scene.nuclei[0].pos == a && s.dynamics.scene.nuclei[1].pos == b);
            if s.dynamics.last_error.is_some() {
                continue;
            }
            w.send(Command::Dyn(DynCommand::Play));
            for s in play_steps(&w, 12) {
                steps += 1;
                fallbacks += usize::from(s.dynamics.source == sim::Source::Fallback);
            }
        }
        let hot = next_snapshot(&w);
        let m = model(&hot);
        eprintln!(
            "hot excursion: {fallbacks}/{steps} fallback snapshots; samples {n0} → {}, quarantined {}, trusted {} ({:?})",
            m.samples, m.quarantined, m.trusted, m.untrusted_reason
        );
        assert!(fallbacks > 0, "test premise: the excursion left what the model was taught");
        let (back, gap) = back_at_equilibrium(&w);
        eprintln!("after the hot excursion: Surrogate {back} steps after the return, audit gap {gap:+.2e} Ha");
    }

    /// Task 9 item 1, the user's exact signature ("trusted, Fallback, 100 % of 100, teacher gap
    /// +2.9e-3 Ha" at the equilibrium bond): the taught H₂ parked at its equilibrium bond with
    /// its centre 1.5 Bohr off-centre (the far atom 2.6 Bohr from a face, face density ≈ 4e-5),
    /// audits every 100 answered steps. On 6c393c3 the first audit there failed on the box's
    /// confinement (+6.4e-3 Ha), the refit quarantined that sample, and because the quarantine
    /// fence lives in descriptor space (translation-invariant) it fenced the equilibrium bond
    /// everywhere in the cell: every later step was a fallback whose answer, confined too,
    /// disagreed with the model — never kept, never releasing (64 steps of 100 % DFT until the
    /// box's own push moved the molecule inward). Now the audit is recorded as box-affected and
    /// not judged, nothing is kept or quarantined, the model answers throughout; back at the
    /// centre it answers and audits clean.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn a_molecule_parked_near_a_face_is_not_fenced_by_the_box() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let (w, trained, _) = taught(h2_41(), ctx, None);
        let n0 = model(&trained).samples;
        w.send(Command::SetAudit(100));
        w.send(Command::Dyn(DynCommand::SetSpeed(1)));
        let x = 1.5;
        w.send(Command::MoveNucleus { index: 0, pos: [x + 0.9, 0.0, 0.0] });
        w.send(Command::MoveNucleus { index: 1, pos: [x - 0.9, 0.0, 0.0] });
        snapshot_until(&w, "the parked molecule", |s| !s.working && !s.dynamics.forces_stale && s.dynamics.scene.nuclei[1].pos[0] == x - 0.9);
        w.send(Command::Dyn(DynCommand::Play));
        let seen = play_steps(&w, 250);
        let fallbacks = seen.iter().filter(|s| s.dynamics.source == sim::Source::Fallback).count();
        let boxed = seen.iter().filter_map(|s| model(s).last_audit.clone()).filter(|a| a.verdict == sim::surrogate_ks::AuditVerdict::BoxAffected).count();
        let last = seen.last().unwrap();
        let m = model(last);
        eprintln!(
            "parked 1.5 Bohr off-centre: {fallbacks}/{} fallback snapshots, last audit {:?}; samples {n0} → {}, quarantined {}, trusted {}",
            seen.len(), m.last_audit, m.samples, m.quarantined, m.trusted
        );
        assert!(boxed > 0, "test premise: an audit near the face was box-affected: {:?}", m.last_audit);
        assert!(m.last_audit.as_ref().is_some_and(|a| a.gap.abs() > 2e-3), "test premise: the confinement gap exceeds the per-atom threshold: {:?}", m.last_audit);
        assert!(m.trusted && m.quarantined == 0 && m.samples == n0, "{m:?}");
        assert_eq!(fallbacks, 0, "the model must keep answering near the face");
        assert!(m.stuck.is_none());
        back_at_equilibrium(&w);
    }

    // ---- the empty scene (the `empty` preset; module doc, "The empty scene") ----

    /// No snapshot with fresh forces arrives within `d` (the worker does not tick an empty scene
    /// that merely plays); returns how many snapshots did.
    fn quiet_for(w: &SimWorker, d: Duration) -> usize {
        let t0 = Instant::now();
        let mut n = 0;
        while let Some(ev) = w.recv_timeout(d.saturating_sub(t0.elapsed())) {
            match ev {
                Event::Snapshot(_) => n += 1,
                Event::Error(m) => panic!("{m}"),
                _ => {}
            }
            if t0.elapsed() >= d {
                break;
            }
        }
        n
    }

    fn assert_empty(s: &Snapshot) {
        let d = &s.dynamics;
        assert!(d.scene.nuclei.is_empty() && s.scene.nuclei.is_empty());
        assert!(!d.forces_stale && d.forces.is_empty() && d.display.is_none(), "{:?}", d.last_error);
        assert_eq!((d.energy_ks, d.total, d.temperature, d.eval.scf_steps), (0.0, 0.0, 0.0, 0));
        assert!(d.last_error.is_none() && !s.working, "{:?}", d.last_error);
    }

    /// The `empty` preset through the worker, on each Kohn-Sham backend (GPU skipped without an
    /// adapter): nothing is solved at start-up; Play is accepted, time stands still and the
    /// worker goes quiet (no tick loop); placing H (paused) is one solve that converges to the
    /// hydrogen atom's energy with its density; removing it is back to the empty answer.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn the_empty_preset_solves_nothing_until_an_atom_is_placed() {
        let _gpu = gpu_lock();
        for (backend, ctx) in backends() {
            let w = SimWorker::spawn(Scene::preset("empty").unwrap(), backend, ctx);
            let s0 = idle(&w);
            assert_empty(&s0);
            assert_eq!((s0.dynamics.scene.cell.n, s0.dynamics.scene.cell.h), (49, 0.25));

            w.send(Command::Dyn(DynCommand::Play));
            let s = snapshot_until(&w, "the play state", |s| !s.dynamics.paused);
            assert_empty(&s);
            assert_eq!((s.dynamics.step, s.dynamics.sim_time), (0, 0.0));
            let n = quiet_for(&w, Duration::from_millis(500));
            assert!(n <= 1, "{backend:?}: {n} snapshots in 0.5 s from an empty scene that plays (a tick loop)");
            w.send(Command::Dyn(DynCommand::Pause));
            assert_empty(&snapshot_until(&w, "the pause", |s| s.dynamics.paused));

            let t = Instant::now();
            w.send(Command::AddNucleus { z: 1, pos: [0.0; 3] });
            let s = snapshot_until(&w, "the placed atom solved", |s| s.dynamics.scene.nuclei.len() == 1 && !s.dynamics.forces_stale);
            let d = &s.dynamics;
            eprintln!("{backend:?}: empty → H: E = {:+.6} Ha in {} SCF steps, {:.2} s", d.energy_ks, d.eval.scf_steps, t.elapsed().as_secs_f64());
            assert!(d.last_error.is_none(), "{:?}", d.last_error);
            assert!(d.energy_ks < -0.40 && d.energy_ks > -0.50, "{backend:?}: E(H) = {}", d.energy_ks);
            assert!(d.eval.scf_steps > 0 && d.forces.len() == 1 && display_len(&s) == 49usize.pow(3));
            assert_eq!(d.step, 0, "paused: placing an atom solves, it does not step");

            w.send(Command::RemoveNucleus(0));
            let s = snapshot_until(&w, "back to empty", |s| s.dynamics.scene.nuclei.is_empty() && !s.dynamics.forces_stale);
            assert_empty(&s);
        }
    }

    /// The estimated solver on the `empty` preset: switching to it is allowed and needs no model —
    /// `EnsureModel` starts no pass and makes no offer, a pass sent anyway is refused with "place
    /// atoms to sample", and nothing is solved, kept or audited. No GPU needed (CPU teacher; it is
    /// never asked).
    #[test]
    fn the_estimated_solver_on_an_empty_scene_needs_no_model() {
        let w = SimWorker::spawn(Scene::preset("empty").unwrap(), Backend::Estimated, None);
        assert_empty(&idle(&w));
        w.send(Command::EnsureModel { autoload: None, plan: test_plan() });
        let s = idle(&w);
        assert_empty(&s);
        let m = model(&s);
        assert!(!m.fitted && m.samples == 0 && m.pass_offer.is_none() && s.dynamics.sampling.is_none(), "{m:?}");
        assert!(s.dynamics.last_rejection.is_none());
        w.send(Command::Dyn(DynCommand::StartSampling(test_plan())));
        let s = snapshot_until(&w, "the refusal", |s| s.dynamics.last_rejection.is_some());
        assert_eq!(s.dynamics.last_rejection.as_deref(), Some(sim::sampling::EMPTY_SCENE));
        assert!(s.dynamics.sampling.is_none() && !s.working);
        w.send(Command::Dyn(DynCommand::Play));
        let s = snapshot_until(&w, "the play state", |s| !s.dynamics.paused);
        assert_empty(&s);
        assert_eq!(model(&s).samples, 0, "nothing solved, nothing kept");
    }

    // ---- the M2/M3 worker behaviour, on the driver ----

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn hydrogen_snapshot_has_unit_charge() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap(), Backend::Cpu, None);
        let s = idle(&w);
        let Some(Display::Cpu { cell, rho, max, .. }) = &s.dynamics.display else {
            panic!("CPU backend publishes values")
        };
        let dv = (cell.h as f32).powi(3);
        let q: f32 = rho.iter().sum::<f32>() * dv;
        assert!((q - 1.0).abs() < 1e-3, "charge {q}");
        assert!(*max > 0.1 && *max < 1.0);
        assert_eq!(rho.len(), cell.n.pow(3));
        assert!(s.dynamics.energy_ks < -0.3);
        assert_eq!(s.dynamics.forces.len(), 1);
    }

    /// A paused burst of teleports is one evaluation (as in M2), and the last position wins.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn queued_commands_are_coalesced_into_one_solve() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap(), Backend::Cpu, None);
        let first = idle(&w);
        for i in 0..20 {
            w.send(Command::MoveNucleus {
                index: 0,
                pos: [0.01 * i as f64, 0.0, 0.0],
            });
        }
        let s = idle(&w);
        assert_eq!(s.dynamics.scene.nuclei[0].pos, [0.19, 0.0, 0.0]);
        assert_eq!(s.edits, 20, "every teleport counts for the viewer's mirror");
        // count the evaluations that finished after the burst: idle snapshots with fresh forces
        let mut evaluations = 1;
        while let Some(ev) = w.recv_timeout(Duration::from_secs(2)) {
            if let Event::Snapshot(s) = ev {
                if !s.dynamics.forces_stale {
                    evaluations += 1;
                }
            }
        }
        eprintln!(
            "coalescing: first seq {}, burst seq {}, {evaluations} evaluations",
            first.dynamics.seq, s.dynamics.seq
        );
        assert!(evaluations <= 2, "{evaluations} evaluations for one burst");
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn grid_change_resizes_the_density() {
        let _gpu = gpu_lock();
        for (backend, ctx) in backends() {
            let w = SimWorker::spawn(Scene::preset("h").unwrap(), backend, ctx);
            let a = idle(&w);
            w.send(Command::SetGrid { n: 33, h: 0.3 });
            let b = idle(&w);
            assert_eq!(a.scene.grid_n, 49);
            assert_eq!(b.scene.grid_n, 33);
            assert_eq!(display_len(&b), 33 * 33 * 33, "{backend:?}");
            assert_eq!(b.backend, backend);
        }
    }

    /// The same H atom through both backends: one physics, two engines. Bounds are the M3 plan's
    /// GPU-vs-CPU energy bound (1e-4 Ha) and the same absolute bound on ρ max (≈ 0.3 e/Bohr³);
    /// both solves stop at `INTERACTIVE_TOL`, so this is looser than the fixed-σ suite.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn gpu_backend_hydrogen_snapshot_matches_cpu_backend() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let scene = Scene::preset("h").unwrap();
        let gw = SimWorker::spawn(scene.clone(), Backend::Gpu, Some(ctx));
        let cw = SimWorker::spawn(scene, Backend::Cpu, None);
        let (g, c) = (idle(&gw), idle(&cw));
        let (Some(Display::Gpu { origin: go, max: gmax, .. }), Some(Display::Cpu { origin: co, max: cmax, .. })) =
            (&g.dynamics.display, &c.dynamics.display)
        else {
            panic!("GPU publishes a texture, CPU values")
        };
        eprintln!(
            "H 49³: GPU E {:+.8} ρmax {gmax:.6} ({} steps) vs CPU E {:+.8} ρmax {cmax:.6} ({} steps)",
            g.dynamics.energy_ks, g.dynamics.eval.scf_steps, c.dynamics.energy_ks, c.dynamics.eval.scf_steps
        );
        assert_eq!((g.backend, c.backend), (Backend::Gpu, Backend::Cpu));
        assert!((g.dynamics.energy_ks - c.dynamics.energy_ks).abs() < 1e-4);
        assert!((gmax - cmax).abs() < 1e-4);
        assert_eq!(go, co);
    }

    /// The display texture is the unpadded n³ interior, x fastest — the centred H atom peaks at
    /// the centre voxel, the box corner is empty, and the charge is 1.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn gpu_density_texture_is_centred_and_unpadded() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(Scene::preset("h").unwrap(), Backend::Gpu, Some(ctx.clone()));
        let s = idle(&w);
        let Some(Display::Gpu { cell, volume, max, .. }) = &s.dynamics.display else {
            panic!("GPU backend must publish a texture")
        };
        let n = cell.n;
        assert_eq!(volume.dims, [n as u32; 3]);
        let rho = volume.read_back(&ctx);
        assert_eq!(rho.len(), n * n * n);
        let (argmax, &peak) = rho
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let c = n / 2;
        assert_eq!(
            argmax,
            c + n * (c + n * c),
            "peak at voxel {argmax}, expected the centre ({c}, {c}, {c})"
        );
        assert_eq!(peak, *max, "the display max is the texture's maximum");
        assert!(rho[0] >= 0.0 && rho[0] < 1e-4, "corner voxel {}", rho[0]);
        let q: f64 = rho.iter().map(|&x| x as f64).sum::<f64>() * cell.h.powi(3);
        assert!((q - 1.0).abs() < 1e-3, "charge {q}");
    }

    /// Diagnostic (ignored; `cargo test -p app --release -- --ignored --nocapture diag_`): the
    /// measurement behind `INTERACTIVE_TOL` on the GPU. For H₂ 49³ and H₂O 57³: a cold solve
    /// and a warm solve after a 0.1 Bohr nudge at tol 1e-5, 3e-6 and 1e-6, each against a
    /// tol-1e-7 reference of the nudged geometry — steps, wall, ΔE and the worst density
    /// difference relative to ρ max (what the picture can show). Uses the solver directly (the
    /// providers hide the tolerance sweep); the inputs are built as `sim::provider` builds them.
    #[test]
    #[ignore]
    fn diag_interactive_tol() {
        let _gpu = gpu_lock();
        use qm::{Grid, ScfSettings, System};
        use qm_gpu::GpuSolver;
        let Some(ctx) = gpu_ctx() else { return };
        let problem = |s: &Scene| {
            let grid = Grid::cube(s.grid_n, s.grid_h);
            let nuclei = s.nuclei.iter().map(|n| qm::Nucleus { pos: n.pos, z: n.z }).collect();
            let system = System::new(nuclei, s.charge, s.multiplicity);
            let settings = ScfSettings::for_grid(&grid);
            (grid, system, settings)
        };
        for (name, n) in [("h2", 49usize), ("h2o", 57)] {
            let mut scene = Scene::preset(name).unwrap();
            scene.grid_n = n;
            let mut nudged = scene.clone();
            nudged.nuclei[1].pos[0] += 0.1;
            let (grid, sys0, _) = problem(&scene);
            let (_, sys1, mut reference) = problem(&nudged);
            let mut s = GpuSolver::new(ctx.clone(), &grid);
            reference.tol = 1e-7;
            let (_, exact) = s.solve(&sys1, &reference, false).unwrap();
            let rho_ref = s.density_f32(s.state().unwrap());
            let peak = rho_ref.iter().copied().fold(0.0f32, f32::max);
            for tol in [1e-5, 3e-6, 1e-6] {
                let (_, _, mut st) = problem(&scene);
                st.tol = tol;
                let t = Instant::now();
                let (_, cold) = s.solve(&sys0, &st, false).unwrap();
                let cold_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let (_, warm) = s.solve(&sys1, &st, true).unwrap();
                let warm_ms = t.elapsed().as_secs_f64() * 1e3;
                let rho = s.density_f32(s.state().unwrap());
                let worst = rho
                    .iter()
                    .zip(&rho_ref)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "{name} {n}³ tol {tol:.0e}: cold {} steps {cold_ms:.0} ms; warm (0.1 Bohr nudge) {} steps {warm_ms:.0} ms, ΔE vs 1e-7 {:+.1e} Ha, max|Δρ|/ρmax {:.1e}",
                    cold.iterations,
                    warm.iterations,
                    warm.energy.total - exact.energy.total,
                    worst / peak
                );
            }
        }
    }

    /// Review Focus 5 (M3): the render thread and the GPU worker submit to one queue. Frames
    /// rendered (spheres + a 97³ raymarch at 640×400) while a 97³ H₂O solve runs (here the
    /// driver's first, cold evaluation) must keep completing — each may wait for the solver's
    /// submission in flight, never for the solve. A headless frame includes a blocking readback,
    /// so it measures the frame's whole GPU latency behind the solver's work: measured (RX 7900
    /// XTX, M3 Task 9) p50 1.0 ms, p95 2.5 ms, max 7–8 ms. The bound asserted is generous
    /// (hardware-dependent); it catches a solver that submits seconds of work at once.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn frames_keep_flowing_while_a_gpu_solve_runs() {
        let _gpu = gpu_lock();
        use glam::Vec3;
        use render::camera::OrbitCamera;
        use render::headless::{Headless, HEADLESS_FORMAT};
        use render::volume::VolumeUniform;
        use render::Renderer;
        let Some(hl) = Headless::new(640, 400) else {
            return;
        };
        let mut water = Scene::preset("h2o").unwrap();
        water.grid_n = 97;
        let w = SimWorker::spawn(water, Backend::Gpu, Some(hl.ctx.clone()));
        let n = 97usize;
        let h = 0.25f32;
        let lo = -h * (n as f32 - 1.0) / 2.0;
        let mut rho = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = [i, j, k].map(|c| lo + c as f32 * h);
                    rho.push((-(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) / 4.0).exp());
                }
            }
        }
        let vol = Volume3D::new(&hl.ctx, [n as u32; 3]);
        vol.upload(&hl.ctx, &rho);
        let mut r = Renderer::new(&hl.ctx, HEADLESS_FORMAT);
        r.volume.bind(&hl.ctx, &vol, &hl.depth);
        r.volume_params = VolumeUniform {
            box_min: [lo; 3],
            box_max: [-lo; 3],
            scale: 1.0,
            step: 0.1,
            ..Default::default()
        };
        r.sync_volume_params(&hl.ctx);
        r.update_camera(&hl.ctx, &OrbitCamera::new(Vec3::ZERO, 20.0), 640.0, 400.0);
        assert!(matches!(
            w.recv_timeout(Duration::from_secs(60)),
            Some(Event::Busy(true))
        ));
        let t0 = Instant::now();
        let mut ms = Vec::new();
        let mut ms_after_start = Vec::new();
        let ended = loop {
            match w.try_recv() {
                Some(Event::Snapshot(s)) => break Some(s),
                Some(Event::Error(message)) => panic!("the 97³ solve failed: {message}"),
                Some(Event::Busy(b)) => panic!("unexpected Busy({b}) before the snapshot"),
                Some(Event::Retired) | None => {}
            }
            if t0.elapsed() > Duration::from_secs(60) {
                break None;
            }
            let t = Instant::now();
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            r.render(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK);
            hl.ctx.queue.submit([enc.finish()]);
            let _ = hl.read_rgba();
            let frame = t.elapsed().as_secs_f64() * 1e3;
            ms.push(frame);
            if t0.elapsed() > Duration::from_secs(1) {
                ms_after_start.push(frame);
            }
        };
        let solve_s = t0.elapsed().as_secs_f64();
        let snap = ended.unwrap_or_else(|| panic!("the 97³ solve did not finish within 60 s ({} frames rendered)", ms.len()));
        assert_eq!(snap.scene.grid_n, 97);
        assert_eq!(snap.dynamics.last_error, None);
        assert!(solve_s >= 1.0, "the 97³ solve ended after {solve_s:.2} s");
        ms.sort_by(|a, b| a.total_cmp(b));
        let q = |v: &[f64], f: f64| v[((v.len() - 1) as f64 * f) as usize];
        ms_after_start.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "frames during a 97³ H₂O GPU solve (render + readback): {} in {solve_s:.2} s, p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms; after the first second: {} frames, p95 {:.1} ms, max {:.1} ms",
            ms.len(),
            q(&ms, 0.5),
            q(&ms, 0.95),
            q(&ms, 1.0),
            ms_after_start.len(),
            if ms_after_start.is_empty() { f64::NAN } else { q(&ms_after_start, 0.95) },
            if ms_after_start.is_empty() { f64::NAN } else { q(&ms_after_start, 1.0) },
        );
        assert!(ms.len() >= 20, "only {} frames during the solve", ms.len());
        assert!(q(&ms, 1.0) < 1000.0, "a frame waited {:.0} ms", q(&ms, 1.0));
    }

    /// A newer scene abandons the evaluation in flight: a cold 97³ H₂O evaluation (the stale
    /// re-evaluation after `SetScene`) is replaced 100 ms in by an H₂ scene; no H₂O snapshot
    /// with fresh forces is ever published, and the first fresh one is the H₂ scene.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn newer_command_cancels_in_flight_solve() {
        let _gpu = gpu_lock();
        let Some(ctx) = gpu_ctx() else { return };
        let h2 = h2_33();
        let w = SimWorker::spawn(h2.clone(), Backend::Gpu, Some(ctx));
        idle(&w);
        let mut water = Scene::preset("h2o").unwrap();
        water.grid_n = 97;
        w.send(Command::SetScene(water));
        std::thread::sleep(Duration::from_millis(100));
        let t = Instant::now();
        w.send(Command::SetScene(h2.clone()));
        let s = snapshot_until(&w, "fresh forces", |s| {
            assert!(
                s.dynamics.forces_stale || s.dynamics.scene.nuclei.len() == 2,
                "the abandoned H₂O evaluation published fresh forces"
            );
            !s.dynamics.forces_stale
        });
        eprintln!(
            "cancel: H₂ evaluated {:.0} ms after replacing the 97³ H₂O scene",
            t.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(s.scene, h2, "the first fresh snapshot must be the newer scene's");
        assert_eq!(display_len(&s), 33 * 33 * 33);
    }
}
