//! Simulation worker: owns the qm state, runs SCF off the render thread, publishes immutable snapshots.
//!
//! The render thread talks to the worker through two unbounded channels: [`Command`]s in,
//! [`Event`]s out. The worker blocks until a command arrives, then coalesces the whole burst
//! (every queued command is applied to its local [`Scene`]) into a single solve, so dragging a
//! nucleus produces one SCF per pause rather than one per mouse event. Each solve warm-starts
//! from the previous state when the grid and the orbital counts are unchanged (the solvers also
//! ignore a warm state that does not match). Solver errors *and* panics become
//! [`Event::Error`]; the worker keeps running either way.
//!
//! # Backends
//! [`Backend::Cpu`] runs `qm::scf` (f64, the reference) and publishes the density as values the
//! viewer uploads. [`Backend::Gpu`] runs `qm_gpu::GpuSolver` on a clone of the viewer's
//! `GpuContext` — one device and one queue, submitted to from two threads (legal in wgpu; the
//! GPU executes the submissions in order) — and publishes the density as a texture the viewer
//! binds as is: the converged density is copied on the GPU (`GpuSolver::copy_density_to_volume`,
//! unpadded n³) into one of the worker's `Volume3D`s and never visits the CPU. A texture is
//! reused only once nothing else holds it ([`TexturePool`]), so a displayed texture is never
//! overwritten.
//!
//! # Cancellation (GPU backend)
//! Before every SCF step the GPU solve checks whether a newer command is queued; if one is, the
//! solve is abandoned — its `Busy(true)` … `Busy(false)` bracket closes with neither a snapshot
//! nor an error, the command is collected as usual, and the next solve warm-starts from the
//! abandoned solve's in-progress state (`GpuSolver::solve_with`). A drag therefore never waits
//! for a stale geometry, and `Drop` (whose `Shutdown` is a command) returns within one SCF step.
//! `qm::scf` has no cancellation point (and `qm` is the untouched reference), so a CPU solve
//! always runs to completion.
use crate::scene::{Scene, SceneNucleus};
use gpu::{GpuContext, Volume3D};
use qm::{scf, Energy, Grid, KsState, Nucleus, ScfSettings, ScfStats, System};
use qm_gpu::GpuSolver;
use std::sync::mpsc::{channel, Receiver, RecvError, RecvTimeoutError, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum Command {
    /// Replaces everything; the grid may change.
    SetScene(Scene),
    MoveNucleus {
        index: usize,
        pos: [f64; 3],
    },
    AddNucleus {
        z: u8,
        pos: [f64; 3],
    },
    RemoveNucleus(usize),
    SetCharge(i32),
    SetMultiplicity(Option<u32>),
    SetGrid {
        n: usize,
        h: f64,
    },
    SetSigma(Option<f64>),
    Shutdown,
}

/// Which solver the worker runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// `qm::scf` on the CPU (f64; the reference).
    Cpu,
    /// `qm_gpu::GpuSolver` on the viewer's device (f32 grid quantities).
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

/// A snapshot's density ρ↑ + ρ↓ (e/Bohr³) as f32 on the unpadded n³ grid, x fastest.
#[derive(Clone)]
pub enum Density {
    /// Values in grid layout (CPU backend); the viewer uploads them.
    Cpu(Arc<Vec<f32>>),
    /// A texture on the shared device (GPU backend), already holding the density; the viewer
    /// binds it as is. The worker writes it again only after every `Arc` outside its pool is gone.
    Gpu(Arc<Volume3D>),
}

impl Density {
    /// Number of grid values (n³ for a well-formed snapshot).
    pub fn len(&self) -> usize {
        match self {
            Density::Cpu(v) => v.len(),
            Density::Gpu(v) => v.len(),
        }
    }
}

/// One converged solve. Immutable once published; `density` is shared, not copied, by consumers.
pub struct Snapshot {
    /// The scene this density belongs to.
    pub scene: Scene,
    pub grid_n: usize,
    pub grid_h: f64,
    pub origin: [f64; 3],
    pub density: Density,
    /// Maximum of `density` (the GPU backend reduces it on the GPU).
    pub density_max: f32,
    pub energy: Energy,
    pub iterations: usize,
    pub max_eigen_residual: f64,
    pub poisson_residual: f64,
    pub warnings: Vec<String>,
    pub solve_ms: u64,
    /// Monotonically increasing, one per solve (errors and abandoned solves consume a number too).
    pub seq: u64,
    /// The backend that produced it.
    pub backend: Backend,
}

// Three events per solve: the ~250-byte variant costs nothing, and boxing it would change the
// `Event::Snapshot(Snapshot)` shape the viewer matches on.
#[allow(clippy::large_enum_variant)]
pub enum Event {
    Snapshot(Snapshot),
    Error { seq: u64, message: String },
    Busy(bool),
}

pub struct SimWorker {
    tx: Sender<Command>,
    rx: Receiver<Event>,
    handle: Option<JoinHandle<()>>,
}

/// Interactive SCF tolerance on ∫|Δρ| (both backends): looser than the oracles' 1e-7 (and
/// `ScfSettings::for_grid`'s 1e-6). CPU (9800X3D): a 49³ H₂ solve takes 5–7 s. GPU (RX 7900 XTX,
/// re-measured in M3 Task 9 by `diag_interactive_tol`): the wall time is flat in tol (the late
/// SCF steps are cheap) — H₂ 49³ cold 0.5 s / warm after a 0.1 Bohr nudge 0.41–0.47 s, H₂O 57³
/// cold 8.1–8.2 s / warm 3.9–4.1 s for tol 1e-5 … 1e-6 — and at every tol the warm density and
/// energy sit at the GPU's own f32 path scatter against a tol-1e-7 reference (max|Δρ|/ρmax
/// 3.5e-6 for H₂, 4e-7 for H₂O; |ΔE| ≤ 3e-6 Ha). Nothing is gained by tightening, so 1e-5
/// stays (fewest SCF steps, i.e. the least GPU queue time competing with the render thread).
/// Interactive snapshots are for looking at, not for reference numbers.
const INTERACTIVE_TOL: f64 = 1e-5;

/// After the first command of a burst, keep collecting commands for this long before solving.
/// A sender pushing a burst races the worker's wake-up; without the window the worker could
/// solve for the first command of the burst alone and only then for the rest. Negligible next
/// to a solve (≥ tens of ms) and shorter than a 60 Hz frame, so a drag still streams.
const COALESCE_WINDOW: Duration = Duration::from_millis(5);

impl SimWorker {
    /// Spawns the worker thread; it immediately solves `initial`. `Backend::Gpu` needs `gpu`, the
    /// context whose device the viewer renders with (the density textures are created on it);
    /// `Backend::Cpu` ignores it.
    pub fn spawn(initial: Scene, backend: Backend, gpu: Option<GpuContext>) -> SimWorker {
        let engine = match backend {
            Backend::Cpu => Engine::Cpu { warm: None },
            Backend::Gpu => Engine::Gpu(GpuEngine {
                ctx: gpu.expect("Backend::Gpu needs a GpuContext"),
                solver: None,
                textures: TexturePool::default(),
            }),
        };
        let (ctx, crx) = channel::<Command>();
        let (etx, erx) = channel::<Event>();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, crx, etx, engine))
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
    /// flight on the CPU. Its remaining events are dropped with the channel.
    pub fn detach(mut self) {
        let _ = self.tx.send(Command::Shutdown);
        self.handle = None; // a dropped JoinHandle detaches the thread; `Drop` then has nothing to join
    }
}

impl Drop for SimWorker {
    /// Asks the worker to stop and joins it. A GPU solve in flight is abandoned at the next SCF
    /// step (`Shutdown` is a command); a CPU solve runs to completion first (no cancellation
    /// point). If the thread already exited, the send fails and the join returns at once —
    /// neither is an error here.
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Local {
    scene: Scene,
    /// A command since the last solve invalidated the warm start (grid or orbital counts).
    reset: bool,
    seq: u64,
}

impl Command {
    /// Applies the command to `s`; returns true if a warm start from the previous state must be
    /// discarded (grid or orbital counts may have changed). The worker and the viewer's local
    /// mirror of the scene both go through this, so the two stay in lockstep.
    pub fn apply(self, s: &mut Scene) -> bool {
        match self {
            Command::SetScene(new) => {
                let reset = new.grid_n != s.grid_n
                    || new.grid_h != s.grid_h
                    || new.nuclei.len() != s.nuclei.len()
                    || new.charge != s.charge
                    || new.multiplicity != s.multiplicity;
                *s = new;
                reset
            }
            Command::MoveNucleus { index, pos } => {
                if let Some(n) = s.nuclei.get_mut(index) {
                    n.pos = pos;
                }
                false
            }
            Command::AddNucleus { z, pos } => {
                s.nuclei.push(SceneNucleus {
                    z,
                    pos,
                    vel: [0.0; 3],
                });
                true
            }
            Command::RemoveNucleus(i) => {
                if i < s.nuclei.len() {
                    s.nuclei.remove(i);
                }
                true
            }
            Command::SetCharge(c) => {
                s.charge = c;
                true
            }
            Command::SetMultiplicity(m) => {
                s.multiplicity = m;
                true
            }
            Command::SetGrid { n, h } => {
                s.grid_n = n;
                s.grid_h = h;
                true
            }
            // A different σ is a legitimate warm start (the σ ladder relies on it).
            Command::SetSigma(x) => {
                s.sigma = x;
                false
            }
            Command::Shutdown => false,
        }
    }
}

/// The command receiver plus at most one command the cancellation check took off the channel
/// (`has_news`), handed out first by the next receive so nothing is lost or reordered.
struct Inbox {
    rx: Receiver<Command>,
    held: Option<Command>,
}

impl Inbox {
    fn recv(&mut self) -> Result<Command, RecvError> {
        match self.held.take() {
            Some(c) => Ok(c),
            None => self.rx.recv(),
        }
    }

    fn recv_timeout(&mut self, d: Duration) -> Result<Command, RecvTimeoutError> {
        match self.held.take() {
            Some(c) => Ok(c),
            None => self.rx.recv_timeout(d),
        }
    }

    fn try_recv(&mut self) -> Result<Command, TryRecvError> {
        match self.held.take() {
            Some(c) => Ok(c),
            None => self.rx.try_recv(),
        }
    }

    /// Whether a command is waiting (it stays queued). A disconnected channel counts too: the
    /// viewer is gone and the solve in flight is for nobody.
    fn has_news(&mut self) -> bool {
        if self.held.is_some() {
            return true;
        }
        match self.rx.try_recv() {
            Ok(c) => {
                self.held = Some(c);
                true
            }
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => true,
        }
    }
}

/// What the receive phase decided.
enum Next {
    Solve,
    Exit,
}

/// Blocks for the first command (unless a solve is already pending), then coalesces the burst.
fn collect(local: &mut Local, inbox: &mut Inbox, pending: bool) -> Next {
    fn take(local: &mut Local, c: Command) {
        if c.apply(&mut local.scene) {
            local.reset = true;
        }
    }
    if !pending {
        match inbox.recv() {
            Ok(Command::Shutdown) | Err(_) => return Next::Exit,
            Ok(c) => take(local, c),
        }
        let deadline = Instant::now() + COALESCE_WINDOW;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match inbox.recv_timeout(left) {
                Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => return Next::Exit,
                Ok(c) => take(local, c),
                Err(RecvTimeoutError::Timeout) => break,
            }
        }
    }
    // drain whatever else is queued
    loop {
        match inbox.try_recv() {
            Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => return Next::Exit,
            Ok(c) => take(local, c),
            Err(TryRecvError::Empty) => return Next::Solve,
        }
    }
}

fn worker(initial: Scene, rx: Receiver<Command>, tx: Sender<Event>, mut engine: Engine) {
    let mut inbox = Inbox { rx, held: None };
    let mut local = Local {
        scene: initial,
        reset: true,
        seq: 0,
    };
    let mut pending = true; // solve the initial scene
    loop {
        if let Next::Exit = collect(&mut local, &mut inbox, pending) {
            return;
        }
        pending = false;
        local.seq += 1;
        let seq = local.seq;
        let _ = tx.send(Event::Busy(true));
        let t0 = Instant::now();
        let scene = local.scene.clone();
        let reset = std::mem::take(&mut local.reset);
        let mut abort = || inbox.has_news();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.solve(&scene, reset, &mut abort)
        }));
        let event = match result {
            Ok(Ok(Some(snap))) => Some(Event::Snapshot(Snapshot {
                solve_ms: t0.elapsed().as_millis() as u64,
                seq,
                ..snap
            })),
            // abandoned for a newer command: no event, the command is collected next
            Ok(Ok(None)) => None,
            Ok(Err(message)) => Some(Event::Error { seq, message }),
            Err(p) => {
                engine.forget();
                let message = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "solver panicked".into());
                Some(Event::Error {
                    seq,
                    message: format!("solver panicked: {message}"),
                })
            }
        };
        // The receiver may be gone (SimWorker dropped mid-solve); the next recv sees Shutdown.
        if let Some(e) = event {
            let _ = tx.send(e);
        }
        let _ = tx.send(Event::Busy(false));
    }
}

/// The solver a worker runs, with its warm state. One per worker thread, never moved after the
/// spawn: the variants' size difference costs nothing.
#[allow(clippy::large_enum_variant)]
enum Engine {
    Cpu {
        /// The last converged state. Taken, not borrowed, by a solve: a failed solve leaves no
        /// warm state, so the next one runs cold.
        warm: Option<KsState>,
    },
    Gpu(GpuEngine),
}

/// The GPU backend: the solver for the current grid (rebuilt on a grid change; it keeps its own
/// warm state) and the density textures.
struct GpuEngine {
    ctx: GpuContext,
    solver: Option<GpuSolver>,
    textures: TexturePool,
}

impl Engine {
    /// Runs one solve. `reset`: the warm start is invalid (grid or orbital counts changed).
    /// `Ok(None)`: abandoned because `abort` returned true (GPU only).
    fn solve(
        &mut self,
        scene: &Scene,
        reset: bool,
        abort: &mut dyn FnMut() -> bool,
    ) -> Result<Option<Snapshot>, String> {
        let (grid, system, settings) = problem(scene)?;
        match self {
            Engine::Cpu { warm } => {
                let prev = warm.take().filter(|_| !reset);
                let (state, stats) =
                    scf(&grid, &system, &settings, prev.as_ref()).map_err(|e| e.to_string())?;
                let density: Vec<f32> = state.rho().iter().map(|&x| x as f32).collect();
                let density_max = density.iter().copied().fold(0.0f32, f32::max);
                *warm = Some(state);
                let density = Density::Cpu(Arc::new(density));
                Ok(Some(snapshot(
                    scene,
                    &grid,
                    density,
                    density_max,
                    stats,
                    Backend::Cpu,
                )))
            }
            Engine::Gpu(e) => e.solve(scene, &grid, &system, &settings, !reset, abort),
        }
    }

    /// After a panic the solver may be mid-solve: drop its state (the GPU solver entirely).
    fn forget(&mut self) {
        match self {
            Engine::Cpu { warm } => *warm = None,
            Engine::Gpu(e) => e.solver = None,
        }
    }
}

impl GpuEngine {
    fn solve(
        &mut self,
        scene: &Scene,
        grid: &Grid,
        system: &System,
        settings: &ScfSettings,
        warm: bool,
        abort: &mut dyn FnMut() -> bool,
    ) -> Result<Option<Snapshot>, String> {
        if self.solver.as_ref().is_none_or(|s| s.grid() != grid) {
            self.solver = None; // free the old grid's buffers before allocating the new ones
            self.solver = Some(GpuSolver::new(self.ctx.clone(), grid));
        }
        let solver = self.solver.as_mut().expect("created above");
        let Some((_, stats)) = solver
            .solve_with(system, settings, warm, abort)
            .map_err(|e| e.to_string())?
        else {
            return Ok(None);
        };
        let state = solver.state().expect("a converged solve exposes its state");
        let n = scene.grid_n as u32;
        let volume = self.textures.take(&self.ctx, [n; 3]);
        // Queue order makes the copy visible to any later render submission; the max reduction
        // below also waits for the GPU.
        solver.copy_density_to_volume(state, &volume);
        let density_max = solver.density_max(state);
        let density = Density::Gpu(volume);
        Ok(Some(snapshot(
            scene,
            grid,
            density,
            density_max,
            stats,
            Backend::Gpu,
        )))
    }
}

/// The GPU worker's density textures. A texture leaves the pool inside a snapshot as an `Arc`
/// and is written again only when the pool holds the only reference: no snapshot in the
/// channel, in the viewer (displayed or bound) or anywhere else still points at it, so a
/// texture is never overwritten while it can be on screen. The check is race-free: other
/// holders can only drop references, never create one from nothing. With a viewer that keeps
/// one snapshot this settles at two textures (ping-pong: the displayed one and the one being
/// written); a grid change drops the old size.
#[derive(Default)]
struct TexturePool {
    all: Vec<Arc<Volume3D>>,
}

impl TexturePool {
    fn take(&mut self, ctx: &GpuContext, dims: [u32; 3]) -> Arc<Volume3D> {
        self.all.retain(|t| t.dims == dims);
        if let Some(t) = self.all.iter().find(|t| Arc::strong_count(t) == 1) {
            return t.clone();
        }
        let t = Arc::new(Volume3D::new(ctx, dims));
        self.all.push(t.clone());
        t
    }
}

/// Pre-validates what `System::new` would panic on and builds the solve's inputs.
fn problem(scene: &Scene) -> Result<(Grid, System, ScfSettings), String> {
    let grid = Grid::cube(scene.grid_n, scene.grid_h);
    let nuclei: Vec<Nucleus> = scene
        .nuclei
        .iter()
        .map(|n| Nucleus { pos: n.pos, z: n.z })
        .collect();
    let protons: i32 = nuclei.iter().map(|n| n.z as i32).sum();
    let total = protons - scene.charge;
    if total < 0 {
        return Err(format!(
            "charge {} exceeds the {protons} protons",
            scene.charge
        ));
    }
    if let Some(m) = scene.multiplicity {
        let n = total as usize;
        if m < 1 || (m as usize - 1) > n || !(n - (m as usize - 1)).is_multiple_of(2) {
            return Err(format!("multiplicity {m} impossible for {n} electrons"));
        }
    }
    let system = System::new(nuclei, scene.charge, scene.multiplicity);
    let mut settings = ScfSettings::for_grid(&grid);
    settings.tol = INTERACTIVE_TOL;
    if let Some(s) = scene.sigma {
        settings.sigma = s;
    }
    Ok((grid, system, settings))
}

/// Packages a converged solve (`solve_ms`/`seq` are filled in by the worker loop).
fn snapshot(
    scene: &Scene,
    grid: &Grid,
    density: Density,
    density_max: f32,
    stats: ScfStats,
    backend: Backend,
) -> Snapshot {
    Snapshot {
        scene: scene.clone(),
        grid_n: scene.grid_n,
        grid_h: scene.grid_h,
        origin: grid.origin,
        density,
        density_max,
        energy: stats.energy,
        iterations: stats.iterations,
        max_eigen_residual: stats.max_eigen_residual,
        poisson_residual: stats.poisson_residual,
        warnings: stats.warnings,
        solve_ms: 0,
        seq: 0,
        backend,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use std::time::Duration;

    fn next_snapshot(w: &SimWorker) -> Snapshot {
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        while std::time::Instant::now() < deadline {
            match w.recv_timeout(Duration::from_secs(1)) {
                Some(Event::Snapshot(s)) => return s,
                Some(Event::Error { message, .. }) => panic!("{message}"),
                _ => {}
            }
        }
        panic!("no snapshot within 120 s");
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn hydrogen_snapshot_has_unit_charge() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap(), Backend::Cpu, None);
        let s = next_snapshot(&w);
        let Density::Cpu(rho) = &s.density else {
            panic!("CPU backend publishes values")
        };
        let dv = (s.grid_h as f32).powi(3);
        let q: f32 = rho.iter().sum::<f32>() * dv;
        assert!((q - 1.0).abs() < 1e-3, "charge {q}");
        assert!(s.density_max > 0.1 && s.density_max < 1.0);
        assert_eq!(s.density.len(), s.grid_n.pow(3));
        assert!(s.energy.total < -0.3);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn moving_a_nucleus_warm_starts() {
        for (backend, ctx) in backends() {
            let w = SimWorker::spawn(Scene::preset("h2").unwrap(), backend, ctx);
            let cold = next_snapshot(&w);
            w.send(Command::MoveNucleus {
                index: 0,
                pos: [0.72, 0.0, 0.0],
            });
            let warm = next_snapshot(&w);
            eprintln!(
                "{backend:?} h2 49^3: cold {} iters ({} ms, E {:+.6}), warm {} iters ({} ms, E {:+.6})",
                cold.iterations,
                cold.solve_ms,
                cold.energy.total,
                warm.iterations,
                warm.solve_ms,
                warm.energy.total
            );
            assert!(warm.seq > cold.seq);
            assert!(
                warm.iterations <= cold.iterations,
                "{backend:?}: warm {} cold {}",
                warm.iterations,
                cold.iterations
            );
            assert_eq!(warm.scene.nuclei[0].pos, [0.72, 0.0, 0.0]);
        }
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn queued_commands_are_coalesced_into_one_solve() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap(), Backend::Cpu, None);
        let first = next_snapshot(&w);
        for i in 0..20 {
            w.send(Command::MoveNucleus {
                index: 0,
                pos: [0.01 * i as f64, 0.0, 0.0],
            });
        }
        let s = next_snapshot(&w);
        assert_eq!(s.scene.nuclei[0].pos, [0.19, 0.0, 0.0]);
        // at most a couple of solves should have happened, not 20. Busy events are skipped: one
        // always follows a snapshot, and stopping at it would leave the count unchecked. 2 s of
        // silence while idle suffices: a further solve announces itself with Busy(true) at once
        // (before Busy(false) of the previous one is even read), and while one is running (≥ 1 s)
        // the loop keeps waiting for its snapshot instead of stopping at the silence.
        let (mut extra, mut busy) = (0, false);
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        while std::time::Instant::now() < deadline {
            match w.recv_timeout(Duration::from_secs(2)) {
                Some(Event::Snapshot(_)) => extra += 1,
                Some(Event::Busy(b)) => busy = b,
                Some(Event::Error { .. }) => {}
                None if !busy => break,
                None => {}
            }
        }
        eprintln!(
            "coalescing: first seq {}, burst snapshot seq {}, {extra} extra",
            first.seq, s.seq
        );
        assert!(extra <= 2, "{extra} extra solves after seq {}", first.seq);
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn coincident_nuclei_report_an_error_not_a_crash() {
        for (backend, ctx) in backends() {
            let w = SimWorker::spawn(Scene::preset("h2").unwrap(), backend, ctx);
            let _ = next_snapshot(&w);
            w.send(Command::MoveNucleus {
                index: 0,
                pos: [-0.7, 0.0, 0.0],
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            loop {
                match w.recv_timeout(Duration::from_secs(1)) {
                    Some(Event::Error { message, .. }) => {
                        assert!(message.contains("coincide"), "{backend:?}: {message}");
                        break;
                    }
                    Some(Event::Snapshot(_)) => {
                        panic!("{backend:?}: coincident nuclei must not produce a snapshot")
                    }
                    _ => assert!(std::time::Instant::now() < deadline, "no error event"),
                }
            }
            // worker is still alive
            w.send(Command::MoveNucleus {
                index: 0,
                pos: [0.7, 0.0, 0.0],
            });
            let _ = next_snapshot(&w);
        }
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn grid_change_resizes_the_density() {
        for (backend, ctx) in backends() {
            let w = SimWorker::spawn(Scene::preset("h").unwrap(), backend, ctx);
            let a = next_snapshot(&w);
            w.send(Command::SetGrid { n: 33, h: 0.3 });
            let b = next_snapshot(&w);
            assert_eq!(a.grid_n, 49);
            assert_eq!(b.grid_n, 33);
            assert_eq!(b.density.len(), 33 * 33 * 33, "{backend:?}");
            assert_eq!(b.backend, backend);
        }
    }

    /// Diagnostic (ignored; `cargo test -p app --release -- --ignored --nocapture diag_`): the
    /// measurement behind `INTERACTIVE_TOL` on the GPU. For H₂ 49³ and H₂O 57³: a cold solve
    /// and a warm solve after a 0.1 Bohr nudge (what one drag pause costs) at tol 1e-5, 3e-6 and
    /// 1e-6, each against a tol-1e-7 reference of the nudged geometry — steps, wall, ΔE and the
    /// worst density difference relative to ρ max (what the picture can show).
    #[test]
    #[ignore]
    fn diag_interactive_tol() {
        let Some(ctx) = gpu_ctx() else { return };
        for (name, n) in [("h2", 49usize), ("h2o", 57)] {
            let mut scene = Scene::preset(name).unwrap();
            scene.grid_n = n;
            let mut nudged = scene.clone();
            nudged.nuclei[1].pos[0] += 0.1;
            let (grid, sys0, _) = problem(&scene).unwrap();
            let (_, sys1, mut reference) = problem(&nudged).unwrap();
            let mut s = GpuSolver::new(ctx.clone(), &grid);
            reference.tol = 1e-7;
            let (_, exact) = s.solve(&sys1, &reference, false).unwrap();
            let rho_ref = s.density_f32(s.state().unwrap());
            let peak = rho_ref.iter().copied().fold(0.0f32, f32::max);
            for tol in [1e-5, 3e-6, 1e-6] {
                let (_, _, mut st) = problem(&scene).unwrap();
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

    /// Review Focus 5: the render thread and the GPU worker submit to one queue. Frames rendered
    /// (spheres + a 97³ raymarch at 640×400) while a 97³ H₂O solve runs must keep completing —
    /// each may wait for the solver's submission in flight, never for the solve. A headless
    /// frame here includes a blocking readback (`read_rgba` polls until the queue is idle), so
    /// it measures the frame's whole GPU latency behind the solver's work: measured (RX 7900
    /// XTX, Task 9) p50 1.0 ms, p95 2.5 ms, max 7–8 ms; in a window at 240 Hz during a 97³ solve
    /// the frame interval was p50 4.1 ms, p95 6–7 ms, max 20–28 ms. The bound asserted is
    /// generous (hardware-dependent); it catches a solver that submits seconds of work at once.
    /// The frames are rendered for the whole solve (Task 10 brought it from 10–14 s to ~3 s).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn frames_keep_flowing_while_a_gpu_solve_runs() {
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
        // Render frames for as long as the worker is busy (M3 Task 10: the cold 97³ solve
        // takes ~3 s now, so the window is the solve itself rather than a fixed 4 s inside a
        // ≥ 10 s solve); frames in the first second overlap the CPU-side cold start (solver
        // tables, the f64 guess) and are reported separately.
        let t0 = Instant::now();
        let mut ms = Vec::new();
        let mut ms_after_start = Vec::new();
        let ended = loop {
            if let Some(ev) = w.try_recv() {
                break Some(ev);
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
        assert!(
            ended.is_some(),
            "the 97³ solve did not finish within 60 s ({} frames rendered)",
            ms.len()
        );
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
        // the solve must have run long enough for the frames to measure something
        assert!(ms.len() >= 20, "only {} frames during the solve", ms.len());
        assert!(q(&ms, 1.0) < 1000.0, "a frame waited {:.0} ms", q(&ms, 1.0));
    }

    /// The headless context for the GPU backend tests (`None` without an adapter: they skip).
    fn gpu_ctx() -> Option<GpuContext> {
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

    /// The same H atom through both backends: one physics, two engines. Bounds are the M3 plan's
    /// GPU-vs-CPU energy bound (1e-4 Ha) and the same absolute bound on ρ max (≈ 0.3 e/Bohr³);
    /// both solves stop at `INTERACTIVE_TOL`, so this is looser than the fixed-σ suite.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn gpu_backend_hydrogen_snapshot_matches_cpu_backend() {
        let Some(ctx) = gpu_ctx() else { return };
        let scene = Scene::preset("h").unwrap();
        let gw = SimWorker::spawn(scene.clone(), Backend::Gpu, Some(ctx));
        let cw = SimWorker::spawn(scene, Backend::Cpu, None);
        let (g, c) = (next_snapshot(&gw), next_snapshot(&cw));
        eprintln!(
            "H 49³: GPU E {:+.8} ρmax {:.6} ({} steps, {} ms) vs CPU E {:+.8} ρmax {:.6} ({} steps, {} ms)",
            g.energy.total,
            g.density_max,
            g.iterations,
            g.solve_ms,
            c.energy.total,
            c.density_max,
            c.iterations,
            c.solve_ms
        );
        assert_eq!((g.backend, c.backend), (Backend::Gpu, Backend::Cpu));
        assert!(matches!(g.density, Density::Gpu(_)));
        assert!(matches!(c.density, Density::Cpu(_)));
        assert!((g.energy.total - c.energy.total).abs() < 1e-4);
        assert!((g.density_max - c.density_max).abs() < 1e-4);
        assert_eq!(
            (g.grid_n, g.grid_h, g.origin),
            (c.grid_n, c.grid_h, c.origin)
        );
    }

    /// Review Focus 2: the snapshot's texture is the unpadded n³ interior, x fastest — the
    /// centred H atom peaks at the centre voxel, the box corner is empty, and the charge is 1.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn gpu_density_texture_is_centred_and_unpadded() {
        let Some(ctx) = gpu_ctx() else { return };
        let w = SimWorker::spawn(Scene::preset("h").unwrap(), Backend::Gpu, Some(ctx.clone()));
        let s = next_snapshot(&w);
        let Density::Gpu(vol) = &s.density else {
            panic!("GPU backend must publish a texture")
        };
        let n = s.grid_n;
        assert_eq!(vol.dims, [n as u32; 3]);
        let rho = vol.read_back(&ctx);
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
        assert_eq!(peak, s.density_max, "density_max is the texture's maximum");
        assert!(rho[0] >= 0.0 && rho[0] < 1e-4, "corner voxel {}", rho[0]);
        let q: f64 = rho.iter().map(|&x| x as f64).sum::<f64>() * s.grid_h.powi(3);
        assert!((q - 1.0).abs() < 1e-3, "charge {q}");
    }

    /// A newer command abandons the solve in flight at the next SCF step boundary: a 97³ H₂O
    /// solve (≈ 14 s on the RX 7900 XTX) is replaced 100 ms after it started by an H₂ scene; the
    /// H₂O snapshot never arrives, the abandoned solve's `Busy` bracket closes without an
    /// event, and the H₂ snapshot is the first one published.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn newer_command_cancels_in_flight_solve() {
        let Some(ctx) = gpu_ctx() else { return };
        let mut water = Scene::preset("h2o").unwrap();
        water.grid_n = 97;
        let h2 = Scene::preset("h2").unwrap();
        let w = SimWorker::spawn(water.clone(), Backend::Gpu, Some(ctx));
        // the next event must be a Busy edge
        let busy = |w: &SimWorker| match w.recv_timeout(Duration::from_secs(60)) {
            Some(Event::Busy(b)) => b,
            Some(Event::Snapshot(s)) => {
                panic!("unexpected snapshot of {} nuclei", s.scene.nuclei.len())
            }
            Some(Event::Error { message, .. }) => panic!("{message}"),
            None => panic!("worker silent for 60 s"),
        };
        assert!(busy(&w), "the initial solve starts at once");
        std::thread::sleep(Duration::from_millis(100));
        let t = Instant::now();
        w.send(Command::SetScene(h2.clone()));
        assert!(
            !busy(&w),
            "the abandoned solve closes its Busy bracket with no snapshot or error"
        );
        let abandoned = t.elapsed();
        assert!(busy(&w), "the H₂ solve starts");
        let s = next_snapshot(&w);
        eprintln!(
            "cancel: H₂O 97³ abandoned {:.0} ms after the command; H₂ snapshot after {:.0} ms ({} ms solve)",
            abandoned.as_secs_f64() * 1e3,
            t.elapsed().as_secs_f64() * 1e3,
            s.solve_ms
        );
        assert_eq!(s.scene, h2, "the first snapshot must be the newer scene's");
        assert_eq!(s.grid_n, 49);
    }
}
