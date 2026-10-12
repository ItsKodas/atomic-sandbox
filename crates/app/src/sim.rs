//! Simulation worker: owns the qm state, runs SCF off the render thread, publishes immutable snapshots.
//!
//! The render thread talks to the worker through two unbounded channels: [`Command`]s in,
//! [`Event`]s out. The worker blocks until a command arrives, then coalesces the whole burst
//! (every queued command is applied to its local [`Scene`]) into a single solve, so dragging a
//! nucleus produces one SCF per pause rather than one per mouse event. Each solve warm-starts
//! from the previous converged state when the grid and the orbital counts are unchanged (`scf`
//! itself also ignores a warm state that does not match). Solver errors *and* panics become
//! [`Event::Error`]; the worker keeps running either way.
use crate::scene::{Scene, SceneNucleus};
use qm::{scf, Energy, Grid, KsState, Nucleus, ScfSettings, System};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender, TryRecvError};
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

/// One converged solve. Immutable once published; `density` is shared, not copied, by consumers.
pub struct Snapshot {
    /// The scene this density belongs to.
    pub scene: Scene,
    pub grid_n: usize,
    pub grid_h: f64,
    pub origin: [f64; 3],
    /// ρ↑ + ρ↓ (e/Bohr³) as f32, len = n³, grid layout.
    pub density: Arc<Vec<f32>>,
    pub density_max: f32,
    pub energy: Energy,
    pub iterations: usize,
    pub max_eigen_residual: f64,
    pub poisson_residual: f64,
    pub warnings: Vec<String>,
    pub solve_ms: u64,
    /// Monotonically increasing, one per solve (errors consume a number too).
    pub seq: u64,
}

// Three events per multi-second solve: the 240-byte variant costs nothing, and boxing it would
// change the `Event::Snapshot(Snapshot)` shape the viewer matches on.
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

/// Interactive SCF tolerance: looser than the oracles' 1e-7 (and `ScfSettings::for_grid`'s
/// 1e-6) so a 49³ H₂ solve takes about a second. Interactive snapshots are for looking at, not
/// for reference numbers.
const INTERACTIVE_TOL: f64 = 1e-5;

/// After the first command of a burst, keep collecting commands for this long before solving.
/// A sender pushing a burst races the worker's wake-up; without the window the worker could
/// solve for the first command of the burst alone and only then for the rest. Negligible next
/// to a solve (≥ hundreds of ms) and shorter than a 60 Hz frame, so a drag still streams.
const COALESCE_WINDOW: Duration = Duration::from_millis(5);

impl SimWorker {
    /// Spawns the worker thread; it immediately solves `initial`.
    pub fn spawn(initial: Scene) -> SimWorker {
        let (ctx, crx) = channel::<Command>();
        let (etx, erx) = channel::<Event>();
        let handle = std::thread::Builder::new()
            .name("sim".into())
            .spawn(move || worker(initial, crx, etx))
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
}

impl Drop for SimWorker {
    /// Asks the worker to stop and joins it. If a solve is in flight, this waits for it to finish
    /// (the solver has no cancellation point). If the thread already exited, the send fails and
    /// the join returns at once — neither is an error here.
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

struct Local {
    scene: Scene,
    warm: Option<KsState>,
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

/// What the receive phase decided.
enum Next {
    Solve,
    Exit,
}

/// Blocks for the first command (unless a solve is already pending), then coalesces the burst.
fn collect(local: &mut Local, rx: &Receiver<Command>, pending: bool) -> Next {
    fn take(local: &mut Local, c: Command) {
        if c.apply(&mut local.scene) {
            local.warm = None;
        }
    }
    if !pending {
        match rx.recv() {
            Ok(Command::Shutdown) | Err(_) => return Next::Exit,
            Ok(c) => take(local, c),
        }
        let deadline = Instant::now() + COALESCE_WINDOW;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => return Next::Exit,
                Ok(c) => take(local, c),
                Err(RecvTimeoutError::Timeout) => break,
            }
        }
    }
    // drain whatever else is queued
    loop {
        match rx.try_recv() {
            Ok(Command::Shutdown) | Err(TryRecvError::Disconnected) => return Next::Exit,
            Ok(c) => take(local, c),
            Err(TryRecvError::Empty) => return Next::Solve,
        }
    }
}

fn worker(initial: Scene, rx: Receiver<Command>, tx: Sender<Event>) {
    let mut local = Local {
        scene: initial,
        warm: None,
        seq: 0,
    };
    let mut pending = true; // solve the initial scene
    loop {
        if let Next::Exit = collect(&mut local, &rx, pending) {
            return;
        }
        pending = false;
        local.seq += 1;
        let seq = local.seq;
        let _ = tx.send(Event::Busy(true));
        let t0 = Instant::now();
        let scene = local.scene.clone();
        // Taken, not borrowed: a failed solve leaves no warm state, so the next one runs cold.
        let warm = local.warm.take();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve(&scene, warm.as_ref())
        }));
        let event = match result {
            Ok(Ok((state, snap))) => {
                local.warm = Some(state);
                Event::Snapshot(Snapshot {
                    solve_ms: t0.elapsed().as_millis() as u64,
                    seq,
                    ..snap
                })
            }
            Ok(Err(message)) => Event::Error { seq, message },
            Err(p) => {
                let message = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "solver panicked".into());
                Event::Error {
                    seq,
                    message: format!("solver panicked: {message}"),
                }
            }
        };
        // The receiver may be gone (SimWorker dropped mid-solve); the next recv sees Shutdown.
        let _ = tx.send(event);
        let _ = tx.send(Event::Busy(false));
    }
}

/// Pre-validates what `System::new` would panic on, then runs the SCF and packages a snapshot
/// (`solve_ms`/`seq` are filled in by the caller).
fn solve(scene: &Scene, warm: Option<&KsState>) -> Result<(KsState, Snapshot), String> {
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
    let (state, stats) = scf(&grid, &system, &settings, warm).map_err(|e| e.to_string())?;
    let density: Vec<f32> = state.rho().iter().map(|&x| x as f32).collect();
    let density_max = density.iter().copied().fold(0.0f32, f32::max);
    let snap = Snapshot {
        scene: scene.clone(),
        grid_n: scene.grid_n,
        grid_h: scene.grid_h,
        origin: grid.origin,
        density: Arc::new(density),
        density_max,
        energy: stats.energy,
        iterations: stats.iterations,
        max_eigen_residual: stats.max_eigen_residual,
        poisson_residual: stats.poisson_residual,
        warnings: stats.warnings,
        solve_ms: 0,
        seq: 0,
    };
    Ok((state, snap))
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
    fn hydrogen_snapshot_has_unit_charge() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap());
        let s = next_snapshot(&w);
        let dv = (s.grid_h as f32).powi(3);
        let q: f32 = s.density.iter().sum::<f32>() * dv;
        assert!((q - 1.0).abs() < 1e-3, "charge {q}");
        assert!(s.density_max > 0.1 && s.density_max < 1.0);
        assert_eq!(s.density.len(), s.grid_n.pow(3));
        assert!(s.energy.total < -0.3);
    }

    #[test]
    fn moving_a_nucleus_warm_starts() {
        let w = SimWorker::spawn(Scene::preset("h2").unwrap());
        let cold = next_snapshot(&w);
        w.send(Command::MoveNucleus {
            index: 0,
            pos: [0.72, 0.0, 0.0],
        });
        let warm = next_snapshot(&w);
        eprintln!(
            "h2 49^3: cold {} iters ({} ms, E {:+.6}), warm {} iters ({} ms, E {:+.6})",
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
            "warm {} cold {}",
            warm.iterations,
            cold.iterations
        );
        assert_eq!(warm.scene.nuclei[0].pos, [0.72, 0.0, 0.0]);
    }

    #[test]
    fn queued_commands_are_coalesced_into_one_solve() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap());
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
        // always follows a snapshot, and stopping at it would leave the count unchecked.
        let mut extra = 0;
        while let Some(ev) = w.recv_timeout(Duration::from_secs(10)) {
            if let Event::Snapshot(_) = ev {
                extra += 1;
            }
        }
        eprintln!(
            "coalescing: first seq {}, burst snapshot seq {}, {extra} extra",
            first.seq, s.seq
        );
        assert!(extra <= 2, "{extra} extra solves after seq {}", first.seq);
    }

    #[test]
    fn coincident_nuclei_report_an_error_not_a_crash() {
        let w = SimWorker::spawn(Scene::preset("h2").unwrap());
        let _ = next_snapshot(&w);
        w.send(Command::MoveNucleus {
            index: 0,
            pos: [-0.7, 0.0, 0.0],
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        loop {
            match w.recv_timeout(Duration::from_secs(1)) {
                Some(Event::Error { message, .. }) => {
                    assert!(message.contains("coincide"), "{message}");
                    break;
                }
                Some(Event::Snapshot(_)) => {
                    panic!("coincident nuclei must not produce a snapshot")
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

    #[test]
    fn grid_change_resizes_the_density() {
        let w = SimWorker::spawn(Scene::preset("h").unwrap());
        let a = next_snapshot(&w);
        w.send(Command::SetGrid { n: 33, h: 0.3 });
        let b = next_snapshot(&w);
        assert_eq!(a.grid_n, 49);
        assert_eq!(b.grid_n, 33);
        assert_eq!(b.density.len(), 33 * 33 * 33);
    }
}
