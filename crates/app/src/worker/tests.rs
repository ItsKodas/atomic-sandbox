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

/// Idle **and** every bond reference in (`bonds_pending` empty; M6 Task 3 fix round 2: they are
/// background work, not `working`). A test that sends a command and then waits for `idle` starts
/// from here, so no reference's progress or landing snapshot can be taken for the command's.
fn settled(w: &SimWorker) -> Snapshot {
    snapshot_until(w, "an idle worker with its bond references in", |s| !s.working && s.dynamics.bonds_pending.is_empty())
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
        let s = settled(&w);
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
    settled(&w);
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
    let w = SimWorker::spawn(Scene::preset("h2").unwrap(), Backend::Gpu, Some(ctx.clone()));
    let cold0 = settled(&w);
    let pos = [0.82, 0.0, 0.0];
    w.send(Command::MoveNucleus { index: 0, pos });
    let warm = idle(&w);
    assert_eq!(warm.dynamics.scene.nuclei[0].pos, pos);
    assert_eq!(warm.dynamics.last_error, None);
    let mut displaced = Scene::preset("h2").unwrap();
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
    settled(&w);
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
            a: None,
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
    // a relax pauses the run (mirrored at once); Stop abandons a FIRE step in flight, like Pause
    assert!(Command::Dyn(DynCommand::Relax).pauses() && Command::Dyn(DynCommand::Relax).is_edit());
    assert!(Command::Dyn(DynCommand::StopRelax).aborts());
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
    let mut s = Scene::preset("h2").unwrap();
    s.grid_n = 41;
    s
}

/// The `h2` preset on a 49³ grid (box 12 Bohr): the far-drag regression's system (M6 Task 9),
/// where the stretched scan samples of the pass stay clear of the box.
fn h2_49() -> Scene {
    let mut s = Scene::preset("h2").unwrap();
    s.grid_n = 49;
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
/// the pass, both within a tripwire bound — on H₂O at 97³, where one solve takes seconds, so a
/// prompt snapshot is the abort.
///
/// Re-pinned in M6 Task 3 fix round 2 (bounds unchanged): the driver's bond references now
/// seed the pass, so once they are in (the O scan on the background auditor: 96–130 s at 97³,
/// background work — `bond_reference`, not `working` — waited for here explicitly) the pass has
/// no reference stage and the Pause and the Cancel land in its geometry stage, with a few
/// samples kept. Below Q′ the pass's end no longer refits (the refit and its trust recovery
/// took 45–50 s for 3 samples inside the Cancel); the note says the samples were kept.
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
    // the bond references in first (they seed the pass): a longer wait than `settled`'s
    let t = Instant::now();
    while !s0.dynamics.bonds_pending.is_empty() {
        assert!(t.elapsed() < Duration::from_secs(400), "bond references not in within 400 s");
        if next_snapshot(&w).dynamics.bonds_pending.is_empty() {
            break;
        }
    }
    eprintln!("H₂O 97³: bond references in {:.0} s (background)", t.elapsed().as_secs_f64());
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
    let stage = s.dynamics.sampling.as_ref().unwrap().stage;
    assert_ne!(stage, "references", "the driver's references seed the pass: no reference stage");
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
    let kept = s.dynamics.notes.iter().find(|n| n.contains("sampling pass cancelled")).cloned();
    eprintln!("{kept:?}");
    assert!(kept.as_deref().is_some_and(|n| n.contains("not enough to fit")), "{:?}", s.dynamics.notes);
}

/// M6 Task 3 fix round 2: Cancel never blocks the worker for seconds. A pass cancelled in its
/// geometry stage with fewer samples than Q′ keeps them and skips the refit and its trust
/// recovery (the model cannot earn trust below Q′ — the trust rule needs n ≥ Q′; on H₂O the
/// skipped refit took 45–50 s for 3 samples inside the Cancel). H₂ at 41³, estimated: the bond
/// reference in first (it seeds the pass, so there is no reference stage), a few geometry
/// samples, then Cancel within the R11 bound (1.5 s), the note "N samples kept — not enough to
/// fit (Q′ = M)", no model fitted.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs real SCF solves; cargo test -p app --release"
)]
fn cancel_below_q_prime_keeps_the_samples_without_a_refit() {
    let _gpu = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let w = SimWorker::spawn(h2_41(), Backend::Estimated, Some(ctx));
    settled(&w);
    w.send(Command::Dyn(DynCommand::StartSampling(test_plan())));
    let s = snapshot_until(&w, "a few geometry samples", |s| sampling_active(&s.dynamics) && model(s).samples >= 3);
    let stage = s.dynamics.sampling.as_ref().unwrap().stage;
    assert_ne!(stage, "references", "the driver's reference seeds the pass");
    while w.try_recv().is_some() {}
    let t = Instant::now();
    w.send(Command::Dyn(DynCommand::CancelSampling));
    let c = snapshot_until(&w, "the cancelled pass", |s| s.dynamics.sampling.as_ref().is_some_and(|p| p.finished));
    let cancel = t.elapsed();
    let note = c.dynamics.notes.iter().find(|n| n.contains("sampling pass cancelled")).cloned().unwrap_or_default();
    eprintln!(
        "H₂ 41³ estimated: Cancel in the {stage} stage with {} samples → cancelled pass in {:.0} ms; {note}",
        model(&c).samples,
        cancel.as_secs_f64() * 1e3
    );
    assert!(cancel < Duration::from_millis(1500), "Cancel took {cancel:?}");
    assert!(note.contains("samples kept — not enough to fit (Q′ = "), "{:?}", c.dynamics.notes);
    assert!(!model(&c).fitted, "no refit below Q′");
    assert!(model(&c).samples >= 3, "the samples are kept");
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
/// at R = 3.3 the model answers again within ≤ 2 fallback steps and ≤ 2 refits.
///
/// M6 Task 9 re-pinned it at 49³ (12 Bohr box; was 41³ with a bound of 20). At 41³ the pass's
/// stretched scan end is box-affected (left out of the fit since the M5 final review, I1), so
/// R = 3.3 started outside the trained envelope and its box-affected answers were not kept: 15
/// fallback steps and 0 refits (deterministic; re-measured in M6 Task 9, 83 samples throughout).
/// At 49³ the scan stays clear of the box: **1 fallback step** — the R = 3.3 teleport's own
/// evaluation, its sample kept (83 → 84) — then the model at step 1, **0 refits** after Play,
/// four runs out of four. The refit that does the work is the teleport round's: paused, the
/// worker refits a kept sample at once (`refit_if_pending` throttles only while playing), before
/// `refits0` is read — so the count does not depend on wall time (review M2; the play-phase
/// throttle would matter only if step 1 fell back too, and a slower GPU would then need fewer
/// steps, not more). Step 1's worst atom uncertainty is 0.015 of its bound (printed). The bound
/// is the measured count × 2 (M6 spec §9); the R = 8 answer is still box-affected at 49³ (face
/// density 6.3e-4 > 2e-5) and not trained on.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs real SCF solves; cargo test -p app --release"
)]
fn after_a_far_drag_the_surrogate_answers_again() {
    let _gpu = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let (w, trained, _) = taught(h2_49(), ctx, None);
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
        assert!(fallbacks <= 2 && m.refits - refits0 <= 2, "still falling back: {fallbacks} fallback steps, {} refits, trusted {}, reason {:?}", m.refits - refits0, m.trusted, m.untrusted_reason);
    };
    let m = model(&back);
    // the margin under the tight bound (review M2): the answering step's worst atom uncertainty
    // as a fraction of its bound (≥ 1 would have fallen back)
    let margin = back.dynamics.eval.atom_uncertainty.as_ref().map(|u| u.iter().copied().fold(0.0f64, f64::max));
    eprintln!(
        "R = 3.3 after the far drag: Surrogate again at step {} after {fallbacks} fallback steps and {} refits (refits before Play: {refits0}); {} samples; worst atom uncertainty / bound {margin:?}",
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

/// Task 9 item 4, re-pinned in M6 Task 9: the automatic first pass keeps to its 2-minute
/// target — it starts with the panel's (default) plan whenever the fewest solves that can earn
/// trust fit (the size is set later, from the median of its first three solves:
/// [`pass_resize`]), and offers to confirm when even those take longer (the user's 12-minute
/// H₂O pass at 57³: 2.4 s a solve). `pass_resize` scales to the target at the median — one
/// slow solve among three moves nothing — never below the trust floor, and leaves a plan that
/// fits.
#[test]
fn the_first_pass_keeps_to_its_time_target_or_asks() {
    let h2 = h2_41().to_sim().unwrap();
    let plan = SamplingPlan::default();
    let full = plan.solves(&h2).unwrap();
    let floor = SamplingPlan::min_samples_for_trust(&h2);
    assert_eq!(first_pass(&h2, plan.clone(), None), FirstPass::Start(plan.clone()), "no cost known: the plan as is");
    assert_eq!(first_pass(&h2, plan.clone(), Some(0.12)), FirstPass::Start(plan.clone()), "273 × 0.12 s = 33 s");
    assert_eq!(first_pass(&h2, plan.clone(), Some(0.9)), FirstPass::Start(plan.clone()), "the floor fits: started whole, sized from its own solves");
    // the re-sizing at the median of the first three solves
    assert_eq!(median(&[0.9, 0.1, 0.1]), Some(0.1));
    assert_eq!(median(&[]), None);
    assert_eq!(pass_resize(&h2, &plan, &[0.9, 0.1, 0.1]), None, "{full} × 0.1 s fits: unchanged");
    let (p, per) = pass_resize(&h2, &plan, &[1.0, 1.0, 1.0]).expect("273 s: scaled");
    let n = p.solves(&h2).unwrap();
    eprintln!("H₂ at a median of 1.0 s: {full} → {n} solves (floor {floor} + 1 reference)");
    assert_eq!(per, 1.0);
    assert!((110..=120).contains(&n) && n > floor, "{n}");
    let (p, _) = pass_resize(&h2, &plan, &[0.1, 0.9, 0.9]).expect("the median is 0.9 s");
    assert_eq!(p, plan.scaled_to(&h2, 133));
    let (p, _) = pass_resize(&h2, &plan, &[30.0, 30.0, 0.1]).expect("30 s a solve");
    let n = p.solves(&h2).unwrap() - 1;
    assert!(n >= floor && n <= floor + 2, "never below the trust floor (+ the rounding Sampler::rescale trims): {n}");
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

/// A scripted teacher for the pass-sizing test: H₂'s Morse well ([`sim::MorseDimer::h2`]; one
/// nucleus → its `e_atom`, no force), each solve first sleeping the next wall time of a shared
/// script ([`SCRIPT_REST_S`] once it is empty, so a pass outlasts the test's checks), and
/// [`SLOW_EDIT_S`] for the geometry with nucleus 0 at [`EDIT_MARK`] (the test's teleport: a live
/// evaluation longer than the tick budget, so its tick runs no pass step).
struct ScriptedMorse {
    morse: sim::MorseDimer,
    script: Arc<std::sync::Mutex<VecDeque<f64>>>,
}

/// [`ScriptedMorse`]'s solve time once its script is empty (s): longer than the 16 ms tick.
const SCRIPT_REST_S: f64 = 0.02;
/// Where the pass-sizing test teleports nucleus 0 (Bohr), and how long the teacher takes there.
const EDIT_MARK: [f64; 3] = [0.75, 0.0625, 0.0];
const SLOW_EDIT_S: f64 = 0.06;

impl ForceProvider for ScriptedMorse {
    fn evaluate(&mut self, scene: &sim::Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<sim::ForceResult, sim::ProviderError> {
        let scripted = self.script.lock().unwrap().pop_front();
        let marked = scene.nuclei.len() == 2 && scene.nuclei[0].pos == EDIT_MARK;
        let sleep = if marked { SLOW_EDIT_S } else { scripted.unwrap_or(SCRIPT_REST_S) };
        std::thread::sleep(Duration::from_secs_f64(sleep));
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (energy, forces) = match pos.len() {
            1 => (self.morse.e_atom, vec![[0.0; 3]]),
            _ => self.morse.energy_and_forces(&pos)?,
        };
        let mut stats = sim::EvalStats::default();
        stats.energy.total = energy;
        Ok(sim::ForceResult { energy, forces, stats, display: None })
    }

    fn name(&self) -> &'static str {
        "scripted morse"
    }

    fn reset(&mut self) {}
}

/// M6 Task 9 (M5 handoff: "the automatic first pass is sized from one cold solve × 1.6 and
/// varies under GPU sharing (72–115 solves seen)"): the pass is sized from the median of its
/// first three solves. A surrogate over a scripted teacher whose pass solves take 0.9 s, 0.1 s,
/// 0.1 s (one slow outlier, like the cold reference) and a plan of 3213 jobs: after the third
/// solve the pass is rescaled to the 2-minute target at ≈ 0.1 s a solve (1000–1200 jobs; at
/// 0.9 s it would be 133); with 1.0 s, 1.0 s, 1.0 s the default plan (273) is scaled to ≈ 120.
/// Either way `progress.total` changes exactly once, and the note says why — also across a
/// teleport queued during the third solve, applied in the round right after the re-sizing,
/// whose evaluation outlasts the tick so that tick runs no pass step (review I2: the driver's
/// published progress must not revert to the old total).
#[test]
fn the_first_pass_is_sized_from_the_median_of_its_first_three_solves() {
    let big = SamplingPlan { perturbations: 3000, ..SamplingPlan::default() };
    let cases = [([0.9, 0.1, 0.1], big, 1000..=1200usize), ([1.0, 1.0, 1.0], SamplingPlan::default(), 110..=120)];
    for (times, plan, want) in cases {
        let script = Arc::new(std::sync::Mutex::new(VecDeque::new()));
        let sc = script.clone();
        let w = SimWorker::spawn_test(
            h2_33(),
            Box::new(move || {
                let teacher = ScriptedMorse { morse: sim::MorseDimer::h2(), script: sc.clone() };
                Box::new(SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(SAMPLE_TOL))
            }),
        );
        let s0 = idle(&w);
        let full = plan.solves(&s0.dynamics.scene).unwrap();
        script.lock().unwrap().extend(times);
        w.send(Command::EnsureModel { autoload: None, plan: plan.clone() });
        // follow the pass until a few snapshots after the edit's evaluation; the edit (a slow
        // live evaluation: a tick without a pass step) is sent a third into the third solve
        let mut totals: Vec<usize> = Vec::new();
        let mut notes = Vec::new();
        let deadline = Instant::now() + WAIT;
        let (mut sent, mut edited, mut after) = (false, false, 0);
        while after < 4 {
            assert!(Instant::now() < deadline, "the pass and the edit did not run: totals {totals:?}");
            let s = next_snapshot(&w);
            notes.extend(s.dynamics.notes.iter().filter(|n| n.contains("median")).cloned());
            if let Some(p) = &s.dynamics.sampling {
                if totals.last() != Some(&p.total) {
                    totals.push(p.total);
                }
                if p.done == 2 && !sent {
                    std::thread::sleep(Duration::from_secs_f64(times[2] / 3.0));
                    w.send(Command::MoveNucleus { index: 0, pos: EDIT_MARK });
                    sent = true;
                }
            }
            edited |= s.dynamics.scene.nuclei[0].pos == EDIT_MARK && !s.dynamics.forces_stale;
            after += usize::from(edited);
        }
        w.send(Command::Dyn(DynCommand::CancelSampling));
        notes.dedup();
        eprintln!("solves {times:?}: totals {totals:?} (the plan: {full}); {notes:?}");
        assert_eq!(totals.len(), 2, "progress.total must change exactly once: {totals:?}");
        assert_eq!(totals[0], full);
        assert!(want.contains(&totals[1]), "rescaled to {} jobs, want {want:?}", totals[1]);
        assert!(notes.iter().any(|n| n.contains(&format!("{full} → {} jobs", totals[1]))), "{notes:?}");
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
    assert_eq!(a.verdict, sim::AuditVerdict::Passed, "{a:?}");
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
    let boxed = seen.iter().filter_map(|s| model(s).last_audit.clone()).filter(|a| a.verdict == sim::AuditVerdict::BoxAffected).count();
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
    let first = settled(&w);
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
        let a = settled(&w);
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

/// M6 Task 3 fix round 1 (review I2): the isolated-atom references are side jobs, solved off
/// the stepping path, so a queued or running reference job neither costs the stepping
/// provider its warm start nor delays a command. H₂O at 57³ (the O reference: three cold
/// solves — measured ≈ 2 s for the singlet and 20 s + 50 s for the triplet and quintet, which
/// stop unconverged at the 200-iteration cap on the GPU, so the scan keeps the singlet; then
/// H's, 0.2 s):
/// - GPU backend (a second `GpuKs` on the driver's own thread): a Pause sent while the run
///   plays and the references solve returns a snapshot within the R11 bound (1.5 s) with the
///   references still pending; the paused worker keeps polling until they are in (not counted
///   as `working`) and publishes the graph — the Pause abandoned a step, so the provider re-solves
///   the density once at the step's start (both O–H pairs bonded, one component, nothing
///   stepped); a 0.02 Bohr teleport of an H then
///   warm-starts (readbacks < 0.85 × a cold solve of the displaced geometry on a fresh worker —
///   `moving_a_nucleus_warm_starts`' rule) although the references were solved in between.
/// - Estimated backend (the references in the background auditor's job slot): a pass started
///   at once finds the references queued (no job starts while a pass runs, unless the first
///   tick already handed O to the auditor); Cancel ends the pass within the same 1.5 s bound — the first draft started the O scan on
///   the stepping teacher the moment the pass stopped and held Cancel for its whole length
///   (18.5 s at 97³) — and the references are then left to the auditor (the providers test
///   `the_references_are_the_background_auditors` pins that path; waiting here for a 70 s O
///   scan would add nothing).
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs real SCF solves; cargo test -p app --release"
)]
fn a_reference_job_neither_resets_the_warm_start_nor_delays_a_command() {
    let _gpu = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let mut water = Scene::preset("h2o").unwrap();
    water.grid_n = 57;
    let w = SimWorker::spawn(water.clone(), Backend::Gpu, Some(ctx.clone()));
    let first = next_snapshot(&w);
    assert!(
        first.dynamics.bonds_pending == vec![8, 1],
        "references queued by the first evaluation: {:?}",
        first.dynamics.bonds_pending
    );
    w.send(Command::Dyn(DynCommand::Play));
    std::thread::sleep(Duration::from_millis(300));
    while w.try_recv().is_some() {}
    let t = Instant::now();
    w.send(Command::Dyn(DynCommand::Pause));
    let s = snapshot_until(&w, "the paused snapshot", |s| s.dynamics.paused);
    let latency = t.elapsed();
    eprintln!(
        "H₂O 57³ GPU: Pause while playing with the references solving → paused snapshot in {:.0} ms (pending {:?}, step {})",
        latency.as_secs_f64() * 1e3,
        s.dynamics.bonds_pending,
        s.dynamics.step
    );
    assert!(latency < Duration::from_millis(1500), "Pause took {latency:?}");
    assert!(!s.dynamics.bonds_pending.is_empty(), "the references were still solving when the Pause landed");
    let t = Instant::now();
    let done = loop {
        // (the O scan alone is ≈ 70 s: a longer wait than `snapshot_until`'s)
        assert!(t.elapsed() < Duration::from_secs(400), "no bond graph within 400 s");
        let s = next_snapshot(&w);
        if s.dynamics.bonds_pending.is_empty() && s.dynamics.bonds.is_some() {
            break s;
        }
    };
    let g = done.dynamics.bonds.clone().unwrap();
    let bonds: Vec<(usize, usize)> = g.bonds.iter().map(|b| (b.i, b.j)).collect();
    eprintln!(
        "references in {:.1} s after the Pause; bonds {bonds:?} (routes {:?}), components {:?}; reports {:?}",
        t.elapsed().as_secs_f64(),
        g.bonds.iter().map(|b| b.route).collect::<Vec<_>>(),
        g.components,
        done.dynamics.pair_reports
    );
    assert!(done.dynamics.paused && done.dynamics.step == s.dynamics.step, "nothing stepped while paused");
    // (only the O–H pairs are asserted: the H–H verdict on the real surface is Task 7's — this
    // run reports it Bonded on Route D near an H end, a finding recorded in the fix-round report)
    assert!(bonds.contains(&(0, 1)) && bonds.contains(&(0, 2)), "both O–H pairs bonded: {bonds:?}");
    assert_eq!(g.components, vec![vec![0, 1, 2]]);
    // M6 Task 10: the chrome's view of it — two bonds drawn between the nuclei as they are, and the
    // molecule's label, computed on the worker
    let pos: Vec<[f64; 3]> = done.scene.nuclei.iter().map(|n| n.pos).collect();
    assert_eq!(crate::overlay::bond_lines(&pos, &g, done.dynamics.bonds_stale).len(), 2 * 2 * crate::overlay::BOND_STRANDS, "two bonds drawn");
    assert_eq!(done.labels.iter().map(|n| n.label()).collect::<Vec<_>>(), vec!["H₂O — water"]);
    assert_eq!(done.charge_note, None);
    assert!(
        done.dynamics.notes.iter().any(|n| n.contains("atom reference O (Z = 8)") && n.contains("off the stepping path")),
        "{:?}",
        done.dynamics.notes
    );
    // (the graph's snapshot is usually the idle one already: no further snapshot comes)
    let idle0 = if done.working { idle(&w) } else { done };
    assert!(idle0.dynamics.last_error.is_none());
    // the stepping provider's warm state survived the references: a teleport warm-starts
    let pos = [1.45, 1.11, 0.0];
    w.send(Command::MoveNucleus { index: 1, pos });
    let warm = idle(&w);
    assert_eq!(warm.dynamics.last_error, None);
    drop(w);
    let mut displaced = water.clone();
    displaced.nuclei[1].pos = pos;
    let w2 = SimWorker::spawn(displaced, Backend::Gpu, Some(ctx.clone()));
    // (its first snapshot is the cold evaluation; idle would also wait for its own O scan)
    let cold = next_snapshot(&w2);
    drop(w2);
    let (wr, cr) = (
        warm.dynamics.eval.readbacks.expect("GPU readbacks"),
        cold.dynamics.eval.readbacks.expect("GPU readbacks"),
    );
    eprintln!(
        "H₂O 57³ teleport after the references: warm {wr} readbacks ({} steps) vs cold {cr} ({} steps)",
        warm.dynamics.eval.scf_steps,
        cold.dynamics.eval.scf_steps
    );
    assert!((wr as f64) < 0.85 * cr as f64, "warm {wr} vs cold {cr}: the references cost the warm start");
    // estimated: the references wait behind a pass, and Cancel does not wait for them
    let e = SimWorker::spawn(water, Backend::Estimated, Some(ctx));
    e.send(Command::Dyn(DynCommand::StartSampling(test_plan())));
    let running = snapshot_until(&e, "a running pass", |s| sampling_active(&s.dynamics));
    assert!(!running.dynamics.bonds_pending.is_empty(), "the references are queued behind the pass");
    std::thread::sleep(Duration::from_millis(400));
    let queued = snapshot_until(&e, "a pass snapshot", |_| true);
    let t = Instant::now();
    e.send(Command::Dyn(DynCommand::CancelSampling));
    let c = snapshot_until(&e, "the cancelled pass", |s| s.dynamics.sampling.as_ref().is_some_and(|p| p.finished));
    let cancel = t.elapsed();
    eprintln!(
        "H₂O 57³ estimated: Cancel with the references queued ({:?}) → cancelled pass in {:.0} ms (pending then {:?})",
        queued.dynamics.bonds_pending,
        cancel.as_secs_f64() * 1e3,
        c.dynamics.bonds_pending
    );
    assert!(c.dynamics.sampling.as_ref().unwrap().cancelled);
    assert!(cancel < Duration::from_millis(1500), "Cancel took {cancel:?}");
    assert!(!queued.dynamics.bonds_pending.is_empty(), "the references were still queued when Cancel landed");
    // and a Pause of the run, with the O scan now in the auditor's slot, is as prompt
    e.send(Command::Dyn(DynCommand::Play));
    snapshot_until(&e, "playing", |s| !s.dynamics.paused);
    std::thread::sleep(Duration::from_millis(300));
    while e.try_recv().is_some() {}
    let t = Instant::now();
    e.send(Command::Dyn(DynCommand::Pause));
    let p = snapshot_until(&e, "the paused snapshot", |s| s.dynamics.paused);
    let pause = t.elapsed();
    eprintln!("estimated: Pause with the O reference in the auditor's slot → snapshot in {:.0} ms (pending {:?})", pause.as_secs_f64() * 1e3, p.dynamics.bonds_pending);
    assert!(pause < Duration::from_millis(1500), "Pause took {pause:?}");
}


// ---- M6 Task 10: the chrome's data from the worker ----

/// The bond graph's labels are computed once per graph on the worker (`Snapshot::labels`): H₂ is
/// "dihydrogen" with one bond drawn; He₂ at 3 Bohr (a closed-shell contact, `validate --test
/// tools`) has no bond — nothing drawn — and two one-atom molecules.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs real SCF solves; cargo test -p app --release"
)]
fn labels_are_computed_once_per_snapshot_on_the_worker() {
    let _gpu = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let w = SimWorker::spawn(Scene::preset("h2").unwrap(), Backend::Gpu, Some(ctx.clone()));
    let s = snapshot_until(&w, "the H₂ bond graph", |s| s.dynamics.bonds.is_some() && s.dynamics.bonds_pending.is_empty());
    let g = s.dynamics.bonds.clone().unwrap();
    eprintln!("H₂ preset: bonds {:?}, reports {:?}", g.bonds, s.dynamics.pair_reports);
    assert_eq!(
        *s.labels,
        vec![crate::names::Name {
            formula: "H₂".into(),
            text: Some("dihydrogen".into()),
            kind: crate::names::NameKind::Trivial,
            ring: false,
            stoichiometric: Some("dihydrogen".into()),
            hydride: None
        }]
    );
    let pos: Vec<[f64; 3]> = s.scene.nuclei.iter().map(|n| n.pos).collect();
    assert_eq!(crate::overlay::bond_lines(&pos, &g, s.dynamics.bonds_stale).len(), 2 * crate::overlay::BOND_STRANDS);
    // the next snapshot of the same graph shares the labels (no recomputation)
    w.send(Command::Dyn(DynCommand::SetSpeed(2)));
    let s2 = next_snapshot(&w);
    if s2.dynamics.bonds.as_ref().is_some_and(|g2| Arc::ptr_eq(g2, &g)) {
        assert!(Arc::ptr_eq(&s.labels, &s2.labels), "labels recomputed for the same graph");
    }
    drop(w);
    let mut he2 = Scene::preset("h2").unwrap();
    he2.nuclei[0].z = 2;
    he2.nuclei[1].z = 2;
    he2.nuclei[0].pos = [-1.5, 0.0, 0.0];
    he2.nuclei[1].pos = [1.5, 0.0, 0.0];
    let w = SimWorker::spawn(he2, Backend::Gpu, Some(ctx));
    let s = snapshot_until(&w, "the He₂ bond graph", |s| s.dynamics.bonds.is_some() && s.dynamics.bonds_pending.is_empty());
    let g = s.dynamics.bonds.clone().unwrap();
    eprintln!("He₂ at 3 Bohr: reports {:?}", s.dynamics.pair_reports);
    assert!(g.bonds.is_empty(), "{:?}", s.dynamics.pair_reports);
    let pos: Vec<[f64; 3]> = s.scene.nuclei.iter().map(|n| n.pos).collect();
    assert!(crate::overlay::bond_lines(&pos, &g, false).is_empty(), "no bond drawn for He₂");
    assert_eq!(s.labels.iter().map(|n| n.label()).collect::<Vec<_>>(), vec!["He", "He"]);
}

/// The Selection tab's isotope choice round trip: `Action::SetIsotope` → `DynCommand::SetIsotope`
/// → the viewer's mirror shows ²H at once → the driver's snapshot carries A = 2 and the deuteron's
/// mass; the atom card follows. (Analytic provider: no SCF.)
#[test]
fn the_isotope_choice_round_trips_through_the_worker() {
    let w = SimWorker::spawn_test(h2_33(), Box::new(|| Box::new(sim::MorseDimer::h2())));
    let s0 = idle(&w);
    let cam = render::camera::OrbitCamera::new(glam::Vec3::ZERO, 14.0);
    let mut st = crate::ui::UiState::new("h2", &s0.scene);
    let inp = crate::ui::PanelInput {
        scene: &s0.scene,
        snapshot: Some(&s0),
        busy: false,
        playing: false,
        last_error: None,
        selected: Some(1),
        measured_fs_per_s: None,
        camera: &cam,
        density_label: None,
        draft_banner: None,
        bonds_label: None,
    };
    let mut out = crate::ui::PanelOutput::default();
    crate::ui::actions::apply(&mut st, &inp, crate::ui::Action::SetIsotope { index: 1, a: 2 }, &mut out);
    assert_eq!(out.commands.len(), 1);
    let cmd = out.commands.pop().unwrap();
    let mut mirror = s0.scene.clone();
    cmd.apply(&mut mirror);
    assert_eq!((mirror.nuclei[0].a, mirror.nuclei[1].a), (None, Some(2)), "the mirror shows it at once");
    assert_eq!(crate::atom_card::title(&mirror.nuclei[1]), "²H hydrogen");
    w.send(cmd);
    let s = snapshot_until(&w, "the deuteron", |s| s.dynamics.scene.nuclei[1].a == 2);
    assert_eq!(s.scene.nuclei[1].a, Some(2));
    assert_eq!(s.dynamics.scene.nuclei[1].mass, units::amu_to_au_mass(units::isotope_mass_amu_a(1, 2).unwrap()));
    assert_eq!(s.dynamics.scene.nuclei[0].a, 1);
    assert_eq!(crate::badge::badge_text(1, s.scene.nuclei[1].mass_number().unwrap(), None), "²H · 1 p · 1 n");
}

/// A relax runs to its end on its own: a running relax is work for the worker, like playing, so
/// it keeps ticking FIRE steps without another command (found by the `relax_h2` replay: the worker
/// went idle after the first FIRE step and waited for a command that never came). Analytic Morse
/// H₂ (r_e = 1.4 Bohr) from R = 1.6, each evaluation 20 ms — longer than a tick's budget, as a DFT
/// solve is, so a tick runs one FIRE step.
#[test]
fn a_relax_runs_to_its_end_without_another_command() {
    let script = Arc::new(std::sync::Mutex::new(VecDeque::new()));
    let w = SimWorker::spawn_test(h2_33(), Box::new(move || Box::new(ScriptedMorse { morse: sim::MorseDimer::h2(), script: script.clone() })));
    idle(&w);
    w.send(Command::Dyn(DynCommand::Relax));
    let s = snapshot_until(&w, "the relax's end", |s| s.dynamics.relax.as_ref().is_some_and(|r| r.converged.is_some()));
    assert_eq!(s.dynamics.relax.as_ref().unwrap().converged, Some(true), "{:?}", s.dynamics.relax);
    let r = dist(s.dynamics.scene.nuclei[0].pos, s.dynamics.scene.nuclei[1].pos);
    assert!((r - 1.4).abs() < 0.01, "R = {r}");
    assert!(!s.working, "idle once it ended");
}

// ---- input replay (M6 Task 8) ----

/// Every checklist script in `crates/app/replays` passes headlessly on the GPU (`replay::run`:
/// the screenshot harness's frame, the `h2` preset to start from; each script names its own
/// scene). Each script's wall time is printed. Skipped without an adapter.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs real SCF solves; cargo test -p app --release"
)]
fn replays_pass() {
    let _gpu = gpu_lock();
    if gpu_ctx().is_none() {
        return;
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("replays");
    let mut scripts: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "replay"))
        .collect();
    scripts.sort();
    assert!(scripts.len() >= 10, "{} scripts in {}", scripts.len(), dir.display());
    let mut failed = Vec::new();
    for s in &scripts {
        let t0 = Instant::now();
        let r = crate::replay::run(s, "h2", Backend::Gpu, None);
        let name = s.file_name().unwrap().to_string_lossy();
        eprintln!("replay {name}: {} in {:.1} s", if r.is_ok() { "passed" } else { "FAILED" }, t0.elapsed().as_secs_f64());
        if let Err(e) = r {
            failed.push(e);
        }
    }
    assert!(failed.is_empty(), "{} of {} scripts failed:\n{}", failed.len(), scripts.len(), failed.join("\n"));
}
