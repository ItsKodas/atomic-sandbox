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
//! values the viewer uploads. A backend switch spawns a new worker (`Viewer::start`).
//!
//! # What aborts a running step (ruling R11)
//! The driver polls `should_abort` before every step and the GPU provider before every SCF
//! step (latency ≤ one SCF step; the CPU provider polls once, before its solve). The worker
//! raises it **only** while a command that makes the running step pointless is queued
//! ([`Command::aborts`]): Pause, the scene edits (teleport, add, remove, a replaced scene,
//! charge / multiplicity / grid / σ — those become `ReplaceScene`) and Shutdown. Drag moves,
//! speed, pins, rewind, thermostat, field, dt and the COM toggle never abort: they only change
//! the state the next step starts from, and a drag move arrives every frame — aborting on it
//! would cancel every DFT step while the user drags. The construction of the driver
//! (`Dynamics::new`'s first, cold evaluation) is not cancellable: a backend switch or shutdown
//! issued during it waits for that one solve.
//!
//! # Errors
//! Provider failures (coincident nuclei, no SCF convergence, an impossible charge) pause the
//! run and arrive in `DynSnapshot::last_error`; refused commands in `last_rejection`. Worker
//! errors (an element without an isotope mass, a panic) arrive as [`Event::Error`]; after a
//! panic the driver is rebuilt from the last snapshot (time restarts, the settings are kept).
use crate::scene::Scene;
use gpu::GpuContext;
use sim::{CpuKs, DynCommand, DynSnapshot, Dynamics, ForceProvider, GpuKs};
use std::collections::VecDeque;
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
            Command::Dyn(_) | Command::Shutdown => {}
        }
    }

    /// Commands the viewer mirrors (scene edits, and the play state: Play / Pause / Step). The
    /// worker counts them in `Snapshot::edits`; the viewer adopts a snapshot's scene and play
    /// state only once every such command it sent is in it, so a snapshot taken before an edit
    /// never undoes the edit on screen.
    pub fn is_edit(&self) -> bool {
        match self {
            Command::Dyn(c) => matches!(c, DynCommand::Play | DynCommand::Pause | DynCommand::Step(_)),
            Command::Shutdown => false,
            _ => true,
        }
    }

    /// Whether the driver ends up paused by it (Pause, Step, and the scene replacements).
    pub fn pauses(&self) -> bool {
        matches!(
            self,
            Command::SetScene(_)
                | Command::SetCharge(_)
                | Command::SetMultiplicity(_)
                | Command::SetGrid { .. }
                | Command::SetSigma(_)
                | Command::Dyn(DynCommand::Pause | DynCommand::Step(_))
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
            ),
            _ => true,
        }
    }

    /// The next command makes this one redundant: consecutive drag moves, consecutive teleports
    /// of the same nucleus.
    fn superseded_by(&self, next: Option<&Command>) -> bool {
        match (self, next) {
            (
                Command::Dyn(DynCommand::DragMove { .. }),
                Some(Command::Dyn(DynCommand::DragMove { .. })),
            ) => true,
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
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Gpu => "GPU",
        }
    }
}

/// One tick of the driver. Immutable once published; the density (`dynamics.display`) and the
/// history are shared, not copied, by consumers.
pub struct Snapshot {
    pub dynamics: DynSnapshot,
    /// `dynamics.scene` in the file format (what the viewer's mirror and the panel use).
    pub scene: Scene,
    pub backend: Backend,
    /// Mirrored commands ([`Command::is_edit`]) applied before this snapshot.
    pub edits: u64,
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
        SimWorker::spawn_with(initial, backend, gpu, Vec::new())
    }

    /// [`SimWorker::spawn`], with `settings` (run settings: dt, speed, thermostat, field, COM
    /// correction) applied to the driver before its first snapshot is published.
    pub fn spawn_with(
        initial: Scene,
        backend: Backend,
        gpu: Option<GpuContext>,
        settings: Vec<DynCommand>,
    ) -> SimWorker {
        assert!(
            backend == Backend::Cpu || gpu.is_some(),
            "Backend::Gpu needs a GpuContext"
        );
        let (ctx, crx) = channel::<Command>();
        let (etx, erx) = channel::<Event>();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, settings, crx, etx, backend, gpu))
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

/// Whether the driver has something to do without a new command (`Snapshot::working`). A stale
/// evaluation that failed is not retried until Play, Step or an edit (they clear `last_error`).
fn has_work(s: &DynSnapshot) -> bool {
    !s.paused || s.pending_steps > 0 || (s.forces_stale && s.last_error.is_none())
}

type MakeProvider = Box<dyn Fn() -> Box<dyn ForceProvider>>;

struct Worker {
    dy: Dynamics,
    make: MakeProvider,
    tx: Sender<Event>,
    backend: Backend,
    edits: u64,
    busy: bool,
    /// The last published snapshot without its display (holding the texture would keep it out
    /// of the provider's pool): the idle test and the panic recovery read it.
    last: DynSnapshot,
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".into())
}

fn worker(
    mut initial: Scene,
    settings: Vec<DynCommand>,
    rx: Receiver<Command>,
    tx: Sender<Event>,
    backend: Backend,
    gpu: Option<GpuContext>,
) {
    let make: MakeProvider = Box::new(move || match &gpu {
        Some(ctx) if backend == Backend::Gpu => Box::new(GpuKs::new(ctx.clone(), INTERACTIVE_TOL)),
        _ => Box::new(CpuKs::new(INTERACTIVE_TOL)),
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
            return;
        }
    };
    let mut w = Worker {
        dy,
        make,
        tx,
        backend,
        edits: 0,
        busy: true,
        last: DynSnapshot::default(),
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
        let Some(cmds) = inbox.take(!has_work(&w.last)) else {
            return;
        };
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| w.round(cmds, &mut inbox))) {
            if !w.recover(panic_message(p)) {
                return;
            }
        }
    }
}

impl Worker {
    /// Applies a batch, ticks once, publishes.
    fn round(&mut self, cmds: Vec<Command>, inbox: &mut Inbox) {
        let applied = !cmds.is_empty();
        let mut it = cmds.into_iter().peekable();
        while let Some(c) = it.next() {
            if c.is_edit() {
                self.edits += 1;
            }
            if !c.superseded_by(it.peek()) {
                self.apply(c);
            }
        }
        let will_work = if applied {
            has_work(&self.dy.snapshot())
        } else {
            has_work(&self.last)
        };
        self.set_busy(will_work);
        let snap = self
            .dy
            .tick(TICK_BUDGET_MS, &mut || inbox.abort_pending());
        self.publish(snap);
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
            Command::Dyn(c) => self.dy.apply(c),
            Command::Shutdown => {} // `Inbox::take` never hands it out
        }
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
        let working = has_work(&snap);
        self.last = DynSnapshot {
            display: None,
            ..snap.clone()
        };
        // the receiver may be gone (SimWorker dropped mid-tick); the next take sees that
        let _ = self.tx.send(Event::Snapshot(Box::new(Snapshot {
            scene: Scene::from_sim(&snap.scene),
            dynamics: snap,
            backend: self.backend,
            edits: self.edits,
            working,
        })));
        self.set_busy(working);
    }

    /// After a panic the driver may be mid-step: rebuild it from the last snapshot (a fresh
    /// provider, time restarts) with the run's settings re-applied, paused. False if that panics
    /// too (the worker then exits).
    fn recover(&mut self, message: String) -> bool {
        self.error(format!(
            "solver panicked: {message} — the run was rebuilt from the last snapshot (paused, time restarts)"
        ));
        let last = self.last.clone();
        let dt = if last.dt > 0.0 { last.dt_fs() } else { DEFAULT_DT_FS };
        let make = &self.make;
        match catch_unwind(AssertUnwindSafe(|| {
            let mut d = Dynamics::new(last.scene.clone(), make(), dt, SEED);
            d.apply(DynCommand::SetSpeed(last.speed));
            d.apply(DynCommand::SetComCorrection(last.com_correction));
            d.apply(DynCommand::SetThermostat(last.thermostat));
            for (i, &p) in last.pins.iter().enumerate() {
                if p {
                    d.apply(DynCommand::Pin(i, true));
                }
            }
            d
        })) {
            Ok(d) => {
                self.dy = d;
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
        let mut s = Scene::preset("h2").unwrap();
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
    /// one step (an aborted tick would publish `steps_last_tick == 0`) and time keeps advancing.
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
                    Event::Busy(_) => {}
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
    /// cold 97³ H₂O evaluation (≈ 3.6 s); Pause 300 ms in must produce a snapshot well before it
    /// would end, with the forces still stale (the evaluation was abandoned, not finished).
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

    /// The viewer's mirror: teleports and additions show at once, a drag never moves the mirror
    /// (the spring does, through the snapshots), and the abort/edit classification matches R11.
    #[test]
    fn mirror_moves_for_teleports_but_not_for_drags() {
        let mut s = Scene::preset("h2").unwrap();
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
            Command::Shutdown,
        ] {
            assert!(c.aborts(), "{c:?} must abort a step");
        }
        for c in [
            Command::Dyn(DynCommand::SetSpeed(10)),
            Command::Dyn(DynCommand::SetDt(0.3)),
            Command::Dyn(DynCommand::Rewind),
            Command::Dyn(DynCommand::SetThermostat(Some((300.0, 10.0)))),
            Command::Dyn(DynCommand::SetExternalField([0.0, -1e-22, 0.0])),
            Command::Dyn(DynCommand::SetComCorrection(true)),
        ] {
            assert!(!c.aborts() && !c.is_edit(), "{c:?}");
        }
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
                None => {}
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
