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
//! The files (split out of `app/src/sim.rs` in M6 Task 1; `crate::sim` re-exports this module):
//! `commands.rs` (what the viewer sends and how a command is applied), `model.rs` (the estimated
//! solver's readouts, first pass and refits, through `sim::Trainable`), `autosave.rs` (the session
//! autosave), `tests.rs`; this file keeps the worker handle, the loop, publishing and recovery.
//!
//! # Backends
//! [`Backend::Gpu`] gives the driver a `providers::GpuKs` on a clone of the viewer's `GpuContext` (one
//! device and one queue, submitted to from two threads — legal in wgpu); its density arrives as
//! `sim::Display::Gpu`, a texture from the provider's pool that the viewer binds as is.
//! [`Backend::Cpu`] gives it `providers::CpuKs` (`qm::scf`, the f64 reference); its density arrives as
//! values the viewer uploads. [`Backend::Estimated`] (M5) gives it a `providers::SurrogateKs` with the
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
//!   plan, re-sized once after its first [`PASS_SIZING_SOLVES`] solves to [`PASS_TARGET_S`] at
//!   their median wall time (never below what trust needs; M6 Task 9: [`pass_resize`],
//!   `Trainable::rescale_pass`), or — when even that floor exceeds the target at the scene's cold
//!   evaluation — nothing and a [`PassOffer`] for the panel's "Start anyway" ([`first_pass`]).
//!   While a pass runs, [`ModelStatus::pass_eta_s`] is its time
//!   left at its measured rate. A refused autoload (another provenance) starts nothing: the
//!   panel offers to forget it. Playing while no model exists (an offer waiting, a refused
//!   autoload) runs DFT steps whose answers are kept without a refit — up to 4 Q′ of them, then
//!   one in ten with a "waiting for a sampling pass" note (`providers::surrogate`, "No model yet";
//!   final review B2).
//! - **Audits.** With the GPU teacher the surrogate gets a second `GpuKs` on the same context as
//!   its background auditor (ruling S14, `SurrogateKs::with_auditor`): an audit no longer stops
//!   the stepping. The CPU teacher audits in the step.
//!   While a pass runs the dynamics stay paused and the worker keeps ticking (each tick spends its
//!   16 ms budget on the pass); edits are accepted.
//! - **Refits.** After every tick in which the surrogate kept a sample (`pending_refit`: a
//!   fallback step — active learning), a model exists (the first fit is the sampling pass's,
//!   which also solves the reference energies a fit needs) and no pass is running, the worker
//!   refits it (through `sim::Trainable`, the only way past the boundary — nothing downcasts the
//!   provider), between ticks as `providers::surrogate` requires — while playing at most once per
//!   [`REFIT_INTERVAL`] unless [`REFIT_SAMPLES`] new samples arrived first (a fallback burst does
//!   not refit every step); paused, at once. The driver notices the new surface epoch and books
//!   the jump on its next tick. Every outcome goes to the snapshot notes ("refit #k: rmse …,
//!   trusted: yes/no — reason", the recovery's quarantine note, or "refit failed: …") and to
//!   [`ModelStatus::last_refit`]; a failed refit is retried at the next opportunity, never fatal.
//!   An untrusted refit triggers the surrogate's trust recovery (ruling S15: a degree-cap retry,
//!   then quarantine of the samples the model cannot represent — `providers::surrogate` module doc).
//!   A pass that ends — finished or cancelled — with fewer usable samples than its basis's Q′
//!   is not fitted (M6 Task 3 fix round 2: below Q′ no fit can earn trust) and leaves **no
//!   model**: its samples are kept, and since this refit needs a fitted model they are not
//!   fitted here but at the end of the next pass, together with that pass's own.
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
//! # The orbital view (orbital-viewer spec §4.3, §7; plan Task 3)
//! [`Command::OrbitalView`] sets what the orbital view shows. It is display only: never an edit,
//! never a reason to abort a step, consecutive ones collapse to the last. The driver
//! (`sim::dynamics`, "Orbital display") forwards it to its provider — with `force_every_step`
//! while paused or single-stepping, the provider's rate bound (`sim::ORBITAL_MIN_INTERVAL`)
//! while playing — and, paused, shows a changed view at once without a solve
//! (`ForceProvider::orbital_display_now`). The newest payload is `DynSnapshot::orbitals`
//! (textures from the provider's pools on the GPU, values on the CPU), stamped with its geometry's
//! time. In estimated mode the one producer is the surrogate's background teacher
//! (`providers::surrogate`, "Side densities"): its audits and display refreshes carry the
//! orbitals, a changed view asks it for a refresh at the current geometry, and "refresh cloud
//! every N s" is how the orbitals follow the run; with the CPU teacher (no background teacher)
//! the estimated solver shows none. The view survives a panic recovery (re-applied); a backend
//! switch spawns a new worker, which gets it through its start-up settings
//! (`DynCommand::SetOrbitalView`).
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
//! # The display interrupt (orbital viewer Task 4 review I2)
//! Display-only work — the orbital view's unoccupied-orbital eigensolve, seconds at the larger
//! grids — yields to the user on **any** command, not only R11's: it writes nothing a command
//! could depend on. `SimWorker::send` counts every command before sending it, `Inbox::take`
//! uncounts what it hands out, and `apply_batch` counts the batch's commands not yet applied; the
//! interrupt (`sim::DisplayInterrupt`, installed on every driver the worker builds) is true while
//! either count is non-zero. The driver hands it to its provider and to `orbital_display_now`; an
//! interrupted eigensolve keeps the previous unoccupied rows (stamped not current) and never
//! touches the evaluation's energy or forces.
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
//! deadline is also capped at one budget after its tick started. An idle worker with a bond
//! reference solving off the stepping path (M6 Task 3; the driver's `poll_references`) polls it
//! the same way — one tick per budget, never a spin — publishing only when its progress changes
//! or it lands; that work is not `working` (no busy state, `--screenshot` does not wait). A
//! worker retired while such a job runs on a **CPU** side solver (a backend switch, shutdown)
//! drops its driver, which raises the job's abort flag; the CPU solve polls it only between
//! multiplicities, so the orphaned thread finishes the multiplicity it is solving and then ends.
//! It holds nothing the next worker needs — no lock, no channel anyone waits on (its answer is
//! dropped with the channel) — so it blocks nothing; it only spends that core until then.
//!
//! # Errors
//! Provider failures (coincident nuclei, no SCF convergence, an impossible charge) pause the
//! run and arrive in `DynSnapshot::last_error`; refused commands in `last_rejection`. Worker
//! errors (an element without an isotope mass, a panic) arrive as [`Event::Error`]; after a
//! panic the driver is rebuilt from the last snapshot (time restarts, the settings are kept).
mod autosave;
mod commands;
mod model;

pub use commands::*;
pub use model::*;

use crate::names::Name;
use crate::scene::Scene;
use gpu::GpuContext;
use providers::{CpuKs, FallbackPolicy, GpuKs, PlanSolves, SurrogateKs};
use sim::{DynCommand, DynSnapshot, Dynamics, ForceProvider, SamplingPlan, Trainable};
use crate::cloud::AtomProfile;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Every command taken off the channel before this snapshot — superseded drag moves and
    /// worker settings included, so it equals the viewer's count of commands sent
    /// (`Viewer::commands_sent`) once the worker has seen them all: the input replayer's
    /// `wait idle` (`crate::replay::idle`) needs a snapshot taken after its last command.
    pub commands: u64,
    /// The worker has more to do without a new command: playing, queued steps, a stale-force
    /// evaluation to run, a relax running or a sampling pass (`has_work`). Equal to the latest
    /// `Event::Busy`.
    pub working: bool,
    /// One label per molecule of `dynamics.bonds` (its `components`, in order): formula and name
    /// (`names::label_components`, M6 spec §4), computed here on the worker once per bond graph
    /// (and charge), never on the render thread. Empty without a graph, or when the graph's
    /// nucleus count is not the scene's.
    pub labels: Arc<Vec<Name>>,
    /// The scene's net charge when it cannot be put on one molecule (several molecules and a
    /// nonzero charge, ruling P1): "net charge +1 — not assignable …".
    pub charge_note: Option<String>,
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
    /// Commands sent and not yet taken by the worker (`Inbox::take`): half of the display
    /// interrupt (module doc, "The display interrupt").
    pending: Arc<AtomicUsize>,
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
        let pending = Arc::new(AtomicUsize::new(0));
        let p = pending.clone();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, settings, crx, etx, backend, gpu, autosave, None, p))
            .expect("spawn sim thread");
        SimWorker {
            tx: ctx,
            rx: erx,
            handle: Some(handle),
            pending,
        }
    }

    /// A worker on a provider of the test's choosing (labelled `Backend::Cpu`): the pacing and
    /// command-flood tests use an analytic one, so they need no GPU and measure no solver.
    #[cfg(test)]
    pub fn spawn_test(initial: Scene, make: MakeProvider) -> SimWorker {
        let (ctx, crx) = channel::<Command>();
        let (etx, erx) = channel::<Event>();
        let pending = Arc::new(AtomicUsize::new(0));
        let p = pending.clone();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, Vec::new(), crx, etx, Backend::Cpu, None, None, Some(make), p))
            .expect("spawn sim thread");
        SimWorker {
            tx: ctx,
            rx: erx,
            handle: Some(handle),
            pending,
        }
    }

    /// Counted before it is sent (the display interrupt, module doc), so the worker never takes a
    /// command the count does not hold yet.
    pub fn send(&self, cmd: Command) {
        self.pending.fetch_add(1, Ordering::SeqCst);
        if self.tx.send(cmd).is_err() {
            self.pending.fetch_sub(1, Ordering::SeqCst);
        }
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
    /// `SimWorker::pending`: decremented by what `take` hands out.
    pending: Arc<AtomicUsize>,
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
        let batch: Vec<Command> = self.held.drain(..).collect();
        let n = batch.len();
        // exact: every command was counted before it was sent (`SimWorker::send`)
        self.pending.fetch_sub(n, Ordering::SeqCst);
        Some(batch)
    }
}

/// Whether the driver has something to do without a new command (`Snapshot::working`): playing
/// (with nuclei to move: an empty scene that plays has nothing to tick — module doc, "The empty
/// scene"), queued steps, a stale-force evaluation, a relax running (its FIRE steps advance one tick at a
/// time like play's; M6 Task 8, found by the `relax_h2` replay), or a sampling pass in progress.
/// A stale evaluation that failed is not retried until Play, Step or an edit (they clear
/// `last_error`). A bond
/// reference solving off the stepping path is **not** work (M6 Task 3 fix round 2): it is
/// background, reported by `DynSnapshot::bond_reference`, never busy, and `--screenshot` does
/// not wait for it; an otherwise idle worker polls it (`polls_references_only`).
fn has_work(s: &DynSnapshot) -> bool {
    (!s.paused && !s.scene.nuclei.is_empty())
        || s.pending_steps > 0
        || (s.forces_stale && s.last_error.is_none())
        || s.relax.as_ref().is_some_and(|r| r.converged.is_none())
        || sampling_active(s)
}

/// Idle but for a bond reference to poll (`bonds_pending`): the worker ticks once per
/// [`TICK_BUDGET_MS`] (paced, never a spin, never a block) and publishes only when the
/// reference's progress changes or it lands (`reference_key`); a command still lands at once
/// (the paced wait applies and publishes it).
fn polls_references_only(s: &DynSnapshot) -> bool {
    !has_work(s) && !s.bonds_pending.is_empty()
}

/// What a polling tick must publish for (`polls_references_only`): the queue, the running
/// job's species and progress, and the driver's change counter (a landing, a note, a graph).
fn reference_key(s: &DynSnapshot) -> (Vec<u8>, Option<(u8, usize)>, u64) {
    (s.bonds_pending.clone(), s.bond_reference.map(|r| (r.z, r.done)), s.seq)
}

pub(crate) type MakeProvider = Box<dyn Fn() -> Box<dyn ForceProvider> + Send>;

struct Worker {
    dy: Dynamics,
    make: MakeProvider,
    tx: Sender<Event>,
    backend: Backend,
    engine: Backend,
    edits: u64,
    /// Commands applied to the driver ([`Snapshot::driver_commands`]).
    driver_commands: u64,
    /// Commands received ([`Snapshot::commands`]).
    commands: u64,
    busy: bool,
    /// The last published snapshot without its display and orbital payload (holding their
    /// textures would keep them out of the provider's pools): the idle test and the panic
    /// recovery read it.
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
    last_audit_seen: Option<sim::AuditRecord>,
    last_audit_step: Option<u64>,
    /// The latched teacher gap ([`ModelStatus::last_gap`]).
    last_gap: Option<(f64, u64)>,
    /// The end of the current tick's budget, when it is to be waited out (module doc, "Pacing").
    pace_until: Option<Instant>,
    /// The pacing schedule's last deadline.
    pace_anchor: Option<Instant>,
    /// Measured or estimated wall cost of one pass solve (s; [`ModelStatus::solve_s`]) and
    /// whether it was measured on a pass.
    solve_s: Option<f64>,
    solve_s_measured: bool,
    /// The running pass: when its current timing segment started, its `done` count then, and
    /// whether that segment is the reference stage (timed apart: `track_cost`).
    pass_clock: Option<(Instant, usize, bool)>,
    /// The pass's remaining time (s), from its measured rate.
    pass_eta_s: Option<f64>,
    pass_offer: Option<PassOffer>,
    /// The automatic first pass being timed for its one re-sizing (`Worker::size_pass`).
    sizing: Option<PassSizing>,
    /// The cloud (module doc): profiles wanted, the refresh interval and the last request, the
    /// profiles held and the atom job in flight (Z, n, h, σ).
    cloud_superposition: bool,
    cloud_refresh: Option<Duration>,
    last_refresh: Option<Instant>,
    profiles: Arc<Vec<AtomProfile>>,
    atom_pending: Option<(u8, usize, f64, Option<f64>)>,
    /// The device a GPU atom density is read back from.
    gpu: Option<GpuContext>,
    /// The batch being applied, its commands not yet applied, and the display interrupt built on
    /// it and `SimWorker::pending` (module doc, "The display interrupt").
    batch: Arc<AtomicUsize>,
    interrupt: sim::DisplayInterrupt,
    /// The molecule labels of the last bond graph and charge they were computed for
    /// ([`Snapshot::labels`]): one computation per graph, not per snapshot.
    labels: Option<(Arc<sim::BondGraph>, i32, Arc<Vec<Name>>, Option<String>)>,
}

/// The plain-data view `names` reads (`names::MoleculeGraph`) of the detector's graph: its bonds
/// with their multi-centre flag (the trihydrogen cation's name needs the ring) and its components.
pub fn molecule_graph(g: &sim::BondGraph) -> crate::names::MoleculeGraph {
    crate::names::MoleculeGraph {
        bonds: g.bonds.iter().map(|b| crate::names::GraphBond { i: b.i, j: b.j, multi_centre: b.multi_centre }).collect(),
        components: g.components.clone(),
    }
}

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
/// the estimated solver's teacher (`solvers` = 2), whose auditor takes the other half —
/// `providers::ks`, "GPU memory"), so a scene over it is refused before anything is allocated.
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
    pending: Arc<AtomicUsize>,
) {
    let gpu_readback = gpu.clone();
    // the display interrupt (module doc): a command sent and not yet taken, or one of the batch
    // being applied not yet applied
    let batch = Arc::new(AtomicUsize::new(0));
    let interrupt: sim::DisplayInterrupt = {
        let (p, b) = (pending.clone(), batch.clone());
        Arc::new(move || p.load(Ordering::SeqCst) > 0 || b.load(Ordering::SeqCst) > 0)
    };
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
        let mut d = Dynamics::new(scene, make(), DEFAULT_DT_FS, SEED);
        d.set_display_interrupt(interrupt.clone());
        d
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
        commands: 0,
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
        solve_s_measured: false,
        pass_clock: None,
        pass_eta_s: None,
        pass_offer: None,
        sizing: None,
        cloud_superposition: false,
        cloud_refresh: None,
        last_refresh: None,
        profiles: Arc::new(Vec::new()),
        labels: None,
        atom_pending: None,
        gpu: gpu_readback,
        batch,
        interrupt,
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
        pending,
    };
    loop {
        if let Some(d) = w.pace_until {
            inbox.wait_until(d);
        }
        // (idle with a bond reference to poll: the round's pacing waits, the take must not block)
        let Some(cmds) = inbox.take(!has_work(&w.last) && w.last.bonds_pending.is_empty()) else {
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
        // the batch's commands still to apply count for the display interrupt (module doc)
        self.batch.store(cmds.len(), Ordering::SeqCst);
        let mut it = cmds.into_iter().peekable();
        while let Some(c) = it.next() {
            self.batch.fetch_sub(1, Ordering::SeqCst);
            self.commands += 1;
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
        let fresh = applied.then(|| self.dy.snapshot());
        let pre = fresh.as_ref().unwrap_or(&self.last);
        let will_work = has_work(pre);
        // the automatic pass's sizing times ticks that can do nothing but the pass
        let pass_only = pre.paused && pre.pending_steps == 0 && !pre.forces_stale;
        let done_before = pre.sampling.as_ref().map(|p| (p.done, p.skipped));
        drop(fresh);
        self.set_busy(will_work);
        let t0 = Instant::now();
        let mut snap = self
            .dy
            .tick(TICK_BUDGET_MS, &mut || inbox.abort_pending());
        let tick_s = t0.elapsed().as_secs_f64();
        self.size_pass(&mut snap, done_before, tick_s, pass_only);
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
            // paused with a reference solving off the stepping path: poll it once a budget
            polls_references_only(&snap).then_some(t0 + budget)
        };
        if self.refit_if_pending(&snap) {
            // the refit's note is in the driver now: publish it with this snapshot
            let d = self.dy.snapshot();
            (snap.notes, snap.notes_total) = (d.notes, d.notes_total);
        }
        self.cloud_jobs(&snap);
        // a polling tick that changed nothing publishes nothing (`polls_references_only`)
        let quiet = !applied && polls_references_only(&self.last) && reference_key(&snap) == reference_key(&self.last);
        if !quiet {
            self.publish(snap);
        }
    }

    /// The cloud's background jobs, between ticks (module doc, "The cloud"): arrived atom
    /// densities become profiles; a missing profile for the scene's species, grid and σ is asked
    /// of the auditor, one species at a time; while playing, a display refresh every
    /// `cloud_refresh` (queued ahead of the next audit when the auditor is busy). Never on the
    /// step path; nothing without a background auditor.
    fn cloud_jobs(&mut self, snap: &DynSnapshot) {
        let gpu = self.gpu.clone();
        let Some(s) = trainable(&mut self.dy) else { return };
        if !s.status(&snap.scene, &snap.forces).audits_background {
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
        let (labels, charge_note) = self.labels_for(&snap);
        self.last = DynSnapshot {
            display: None,
            side_display: None,
            orbitals: None,
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
            commands: self.commands,
            working,
            labels,
            charge_note,
        })));
        self.set_busy(working);
    }

    /// The molecule labels for `snap`'s bond graph ([`Snapshot::labels`]), from the cache while
    /// the graph (by identity) and the charge are unchanged.
    fn labels_for(&mut self, snap: &DynSnapshot) -> (Arc<Vec<Name>>, Option<String>) {
        let Some(g) = snap.bonds.as_ref().filter(|g| g.n_nuclei == snap.scene.nuclei.len()) else {
            return (Arc::default(), None);
        };
        let charge = snap.scene.charge;
        if let Some((cg, cq, l, n)) = &self.labels {
            if Arc::ptr_eq(cg, g) && *cq == charge {
                return (l.clone(), n.clone());
            }
        }
        let z: Vec<u8> = snap.scene.nuclei.iter().map(|n| n.z).collect();
        let (names, note) = crate::names::label_components(&z, &molecule_graph(g), charge);
        let names = Arc::new(names);
        self.labels = Some((g.clone(), charge, names.clone(), note.clone()));
        (names, note)
    }

    /// After a panic the driver may be mid-step: rebuild it from the last snapshot (a fresh
    /// provider, time restarts) with the run's settings re-applied, paused; an estimated run's
    /// model is carried over through the autosave path when it can still be written. False if
    /// that panics too (the worker then exits).
    fn recover(&mut self, message: String) -> bool {
        // the batch in flight is gone with the panic
        self.batch.store(0, Ordering::SeqCst);
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
        let interrupt = self.interrupt.clone();
        match catch_unwind(AssertUnwindSafe(|| {
            let mut d = Dynamics::new(last.scene.clone(), make(), dt, SEED);
            d.set_display_interrupt(interrupt.clone());
            d.apply(DynCommand::SetSpeed(last.speed));
            d.apply(DynCommand::SetComCorrection(last.com_correction));
            d.apply(DynCommand::SetThermostat(last.thermostat));
            // the container too (on by default since 2026-10-14; final review app I4: a
            // recovered run used to go unconfined while the panel still showed the walls)
            if let Some(w) = last.walls {
                d.apply(DynCommand::SetWalls(Some(w)));
            }
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
            // the orbital view too (orbital-viewer plan Task 3)
            if !last.orbital_view.what.is_off() {
                d.apply(DynCommand::SetOrbitalView(last.orbital_view.clone()));
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
pub(crate) mod tests;
