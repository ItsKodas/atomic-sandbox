use super::*;
use crate::tools;
use crate::ui::ToolMode;
use std::cell::Cell;
use std::rc::Rc;

/// The test host: the worker's events straight into the viewer, no renderer (the analytic
/// provider's snapshots carry no density).
struct Bare;

impl Host for Bare {
    fn pump(&mut self, v: &mut Viewer, timeout: Duration) -> Option<bool> {
        let ev = v.sim.as_ref().and_then(|w| w.recv_timeout(timeout))?;
        let snapshot = matches!(ev, crate::sim::Event::Snapshot(_));
        v.on_event_bare(ev);
        Some(snapshot)
    }
}

/// The `h2` preset on an analytic Morse H₂ worker (no SCF, no GPU), idle after its first
/// evaluation, with a frame run so the viewport is the chrome's at 1400 × 900.
/// H₂ on the analytic Morse dimer, pinned to what these tests were written for (the merge of main,
/// 2026-10-14): R = 1.6 Bohr (`scene::h2_r16`) on 33³ × 0.25, walls off — not the app's defaults
/// (the h2 preset at 1.45 Bohr, Accurate 81³ × 0.15, walls on).
fn morse_h2() -> (Viewer, Bare) {
    let mut v = Viewer::new("h2").unwrap();
    let mut s = crate::scene::h2_r16();
    s.grid_n = 33;
    v.ui.adopt(&s);
    v.ui.walls_on = false;
    v.ui.ionic = sim::IonicPolicy::None; // all-electron, like the file it is pinned to (M7: the app starts under cores only)
    v.scene = s;
    v.start_test(Box::new(|| Box::new(sim::MorseDimer::h2())));
    let mut host = Bare;
    let mut p = Player::new(&mut host, (1400, 900));
    p.frame(&mut v);
    p.wait_idle(&mut v).unwrap();
    drop(p); // the player frees its unpainted texture updates (`Drop`) and releases the host
    (v, host)
}

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("atomic-sandbox-replay-{}-{name}", std::process::id()))
}

/// Nucleus i's place on screen in view coordinates (the frame's viewport, the default camera).
fn at(v: &Viewer, i: usize) -> (f32, f32) {
    let px = tools::project(&v.camera, tools::v3(v.scene.nuclei[i].pos), v.viewport).unwrap();
    to_view(px, v.viewport)
}

const EVERY_LINE: &str = "\
# atomic sandbox replay v1
view 1280 800
action smooth-motion false
t 120 cursor 0.25 -0.125
t 130 button left down
wait ms 270
t 400 cursor 0.5 0.0625   # a trailing comment
t 410 button left up
t 420 button middle down
t 421 button right up
t 430 wheel -1.5
t 440 key space down
t 441 key delete up
t 450 resize 1000 700
t 460 cursor-left
t 470 focus-lost
action tool select
action element 8
action play
wait steps 20
wait fs 2.5
wait idle
expect nuclei 3
expect paused false
expect selected 1
expect selected none
expect dist 0 1 1.5 2.5
expect bonds 2
expect molecule 0 H2O
expect no-error
expect isotope 0 2
expect relaxed true
expect source surrogate
scene ../my scenes/water.scene
# not replayable: apply-grid (its settings live in the panel, not in a script): ApplyGrid
mark
wait source fallback
expect max-dist 0 1 3
expect fallbacks 1
expect new-samples 2
expect new-epochs 1
expect fallback-fraction 0.1
expect pull 0.3 0.31
expect bonds-stale false
expect solve-log 3
expect counters
";

/// Spec §8.2: every directive parses, `write` is `parse`'s inverse, and an unknown word or a
/// missing argument is refused naming the line.
#[test]
fn the_replay_format_round_trips() {
    let lines = parse(EVERY_LINE).unwrap();
    assert_eq!(lines.len(), 45);
    assert_eq!(lines[0], Line::View(1280, 800));
    assert_eq!(lines[1], Line::Action(Action::SmoothMotion(false)));
    assert_eq!(lines[2], Line::At(120, Event::Cursor { x: 0.25, y: -0.125 }));
    assert_eq!(lines[4], Line::WaitMs(270));
    assert_eq!(lines[9], Line::At(430, Event::Wheel { lines: -1.5 }));
    assert_eq!(lines[12], Line::At(450, Event::Resize { w: 1000, h: 700 }));
    assert_eq!(lines[18], Line::WaitSteps(20));
    assert_eq!(lines[19], Line::WaitFs(2.5));
    assert_eq!(lines[24], Line::Expect(Expect::Selected(None)));
    assert_eq!(lines[25], Line::Expect(Expect::Dist(0, 1, 1.5, 2.5)));
    assert_eq!(lines[27], Line::Expect(Expect::Molecule(0, "H2O".into())));
    assert_eq!(lines[31], Line::Expect(Expect::Source("surrogate".into())));
    assert_eq!(lines[32], Line::Scene("../my scenes/water.scene".into()), "a scene path keeps its spaces");
    assert_eq!(lines[33], Line::NotReplayable("apply-grid".into()));
    assert_eq!(lines[34], Line::Mark);
    assert_eq!(lines[35], Line::WaitSource("fallback".into()));
    assert_eq!(lines[36], Line::Expect(Expect::MaxDist(0, 1, 3.0)));
    assert_eq!(lines[41], Line::Expect(Expect::Pull(0.3, 0.31)));
    assert_eq!(lines[42], Line::Expect(Expect::BondsStale(false)));
    assert_eq!(lines[43], Line::Expect(Expect::SolveLog(3)));
    assert_eq!(lines[44], Line::Expect(Expect::Counters));
    let text = write(&lines);
    assert!(text.starts_with("# atomic sandbox replay v1\n"), "{text}");
    assert_eq!(parse(&text).unwrap(), lines);
    // a recorded cursor's f32 survives the text exactly
    let odd = vec![Line::At(7, Event::Cursor { x: 0.123_456_79, y: -1.0e-7 })];
    assert_eq!(parse(&write(&odd)).unwrap(), odd);
    // errors name the line
    let bad = "view 1280 800\n\n# c\naction play\nwait idle\nexpect nuclei 2\nexpct nuclei 3\n";
    assert_eq!(parse(bad).unwrap_err(), "line 7: unknown directive 'expct'");
    let e = parse("view 1 1\nexpect dist 0 1 1.5\n").unwrap_err();
    assert!(e.starts_with("line 2: ") && e.contains("expect dist"), "{e}");
    let e = parse("action warp 9\n").unwrap_err();
    assert!(e.starts_with("line 1: ") && e.contains("unknown action 'warp'"), "{e}");
    assert!(parse("wait source teacher\n").unwrap_err().starts_with("line 1: "));
    let e = parse("t 5 button left sideways\n").unwrap_err();
    assert!(e.starts_with("line 1: "), "{e}");
}

/// `Action::describe` and `Action::parse` are inverses for every action a script can name; the
/// ones whose meaning is panel state a script does not carry describe as `None`.
#[test]
fn actions_parse_and_describe_symmetrically() {
    use crate::names::NameStyle;
    let all = [
        Action::TogglePlay,
        Action::Play,
        Action::Pause,
        Action::Step,
        Action::Rewind,
        Action::SetSpeed(250),
        Action::SetDt(0.125),
        Action::Tool(ToolMode::Select),
        Action::Tool(ToolMode::Place),
        Action::Tool(ToolMode::Orbit),
        Action::Element(8),
        Action::ElementIsotope(1, 2),
        Action::SetIsotope { index: 1, a: 2 },
        Action::Deselect,
        Action::Delete(3),
        Action::Focus,
        Action::Pin(1, true),
        Action::Pin(0, false),
        Action::Backend(Backend::Estimated),
        Action::Backend(Backend::Cpu),
        Action::Preset("h2o".into()),
        Action::Thermostat { on: true, t: 300.0, tau: 5.0 },
        Action::Thermostat { on: false, t: 300.0, tau: 10.0 },
        Action::Gravity(true),
        Action::ComCorrection(false),
        Action::SetCharge(-1),
        Action::SetMultiplicity(Some(3)),
        Action::SetMultiplicity(None),
        Action::SetSigma(Some(0.3)),
        Action::SetSigma(None),
        Action::Forget,
        Action::CancelSampling,
        Action::SetAudit(0),
        Action::VolumeMode(1),
        Action::Walls { on: true, half_width: 3.5 },
        Action::Cloud { superposition: true, refresh: false, refresh_s: 2.0 },
        Action::Draft(true),
        Action::DragForce(0.16),
        Action::DragForceWheel(-2.0),
        Action::ShowBonds(false),
        Action::ShowLabels(true),
        Action::ShowBadge(false),
        Action::NameStyle(NameStyle::Hydride),
        Action::NameStyle(NameStyle::FormulaOnly),
        Action::Relax,
        Action::StopRelax,
        Action::SmoothMotion(false),
        Action::Show(crate::ui::ShowMode::Spin),
        Action::PickOrbital { spin: sim::Spin::Down, track: 4 },
        Action::OrbitalPhase(false),
        Action::Unoccupied(6),
        Action::Include { track: 2, on: false },
        Action::IncludeAllOccupied,
        Action::ColourBlend { additive: false },
        Action::ColourScale { physical: true },
    ];
    for a in all {
        let text = a.describe().unwrap_or_else(|| panic!("{a:?} has no script form"));
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(Action::parse(&words).unwrap(), a, "{text}");
    }
    for a in [Action::ApplyGrid, Action::SaveScene, Action::LoadScene { sidecar: true }, Action::SaveModel, Action::LoadModel, Action::Sample { long: false }, Action::SampleAfterRefusal] {
        assert_eq!(a.describe(), None, "{a:?}");
    }
    assert!(Action::parse(&["pin", "1", "maybe"]).unwrap_err().contains("true/false"));
    assert!(Action::parse(&["speed"]).unwrap_err().contains("1 argument"));
}

/// The recorder (spec §8.2): the header, the view and `action smooth-motion false` first; a gap of
/// more than 50 ms since the previous line becomes `wait ms <gap>` before the next line (a
/// recorded drag replays at its pace), a shorter one nothing; `wait idle` once per idle gap.
#[test]
fn the_recorder_emits_wait_idle_once_per_idle_gap_and_wait_ms_for_long_gaps() {
    let path = temp("recorder.replay");
    let now = Rc::new(Cell::new(0u64));
    let clock = now.clone();
    let mut r = Recorder::with_clock(&path, (1280, 800), Box::new(move || Duration::from_millis(clock.get()))).unwrap();
    let cursor = |x: f32| Event::Cursor { x, y: 0.0 };
    r.event(&cursor(0.1)); // t 0
    r.idle();
    r.idle(); // the same idle gap: nothing
    now.set(400);
    r.event(&cursor(0.2)); // 400 ms after the `wait idle` line
    now.set(420);
    r.event(&cursor(0.3)); // 20 ms: no wait
    now.set(430);
    r.action(&Action::Play);
    now.set(1430);
    r.action(&Action::Pause); // a second later
    r.idle();
    r.action(&Action::SaveScene); // not replayable: a comment
    drop(r);
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let body: Vec<&str> = text.lines().collect();
    assert_eq!(
        body,
        [
            "# atomic sandbox replay v1",
            "view 1280 800",
            "action smooth-motion false",
            "t 0 cursor 0.1 0",
            "wait idle",
            "wait ms 400",
            "t 400 cursor 0.2 0",
            "t 420 cursor 0.3 0",
            "action play",
            "wait ms 1000",
            "action pause",
            "wait idle",
            "# not replayable: save-scene (its settings live in the panel, not in a script): SaveScene",
        ],
        "{text}"
    );
    let lines = parse(&text).unwrap();
    assert_eq!(lines.len(), 12);
    assert_eq!(lines[11], Line::NotReplayable("save-scene".into()), "the replayer sees the dropped action");
}

/// `wait steps N` / `wait fs X` block on the snapshot's step counter and simulated time (counted
/// from where they stood when the wait began), not on wall time; a wait for steps while the run is
/// paused and idle fails at once, naming the step.
#[test]
fn wait_steps_and_wait_fs_block_on_the_snapshot_counters() {
    let (mut v, mut host) = morse_h2();
    let mut p = Player::new(&mut host, (1400, 900));
    let obs = p.play(&mut v, "action play\nwait steps 7\nwait fs 1.0\naction pause\nwait idle\nexpect paused true\n").unwrap();
    assert_eq!(obs.len(), 3);
    let (line, first) = &obs[0];
    assert_eq!(*line, 2);
    assert!((7..=9).contains(&first.step), "returned at step {}", first.step);
    let second = &obs[1].1;
    assert!(second.sim_time_fs >= first.sim_time_fs + 1.0 - 1e-9, "{} → {}", first.sim_time_fs, second.sim_time_fs);
    assert!(second.sim_time_fs <= first.sim_time_fs + 1.0 + 3.0 * 0.25 + 1e-9, "{} → {}", first.sim_time_fs, second.sim_time_fs);
    assert!(obs[2].1.paused);
    let e = p.play(&mut v, "wait steps 1000\n").unwrap_err();
    assert!(e.starts_with("line 1: wait steps 1000: ") && e.contains("paused at step"), "{e}");
}

/// `wait idle` (spec §8.2): the worker idle, no relax running and no bond reference pending.
#[test]
fn wait_idle_waits_for_a_running_relax_and_the_bond_references() {
    let snap = |d: sim::DynSnapshot, working: bool| Snapshot {
        scene: crate::scene::Scene::preset("h2").unwrap(),
        dynamics: d,
        backend: Backend::Gpu,
        engine: Backend::Gpu,
        model: None,
        atom_profiles: Default::default(),
        edits: 0,
        driver_commands: 0,
        commands: 0,
        working,
        labels: Default::default(),
        charge_note: None,
        activity: Default::default(),
    };
    let relax = |converged| sim::RelaxProgress {
        steps: 3,
        max_force: 1e-2,
        max_force_augmented: 1e-2,
        energy: -1.0,
        energy_start: -0.99,
        dt_fs: 0.1,
        alpha: 0.1,
        converged,
        verified_max_force: None,
        note: None,
    };
    assert!(snapshot_idle(&snap(sim::DynSnapshot::default(), false)));
    assert!(!snapshot_idle(&snap(sim::DynSnapshot::default(), true)));
    let mut d = sim::DynSnapshot::default();
    d.relax = Some(relax(None));
    assert!(!snapshot_idle(&snap(d.clone(), false)), "a relax running");
    d.relax = Some(relax(Some(true)));
    assert!(snapshot_idle(&snap(d.clone(), false)), "a relax that ended");
    d.bonds_pending = vec![8];
    assert!(!snapshot_idle(&snap(d, false)), "a bond reference pending");
}

/// What the replayer saw, without the line numbers (a live session has none).
fn trace(obs: &[(usize, Observation)]) -> Vec<Observation> {
    obs.iter().map(|(_, o)| o.clone()).collect()
}

/// A live session: the recorder and its clock (10 ms per line: no `wait ms`).
struct Live {
    rec: Recorder,
    now: Rc<Cell<u64>>,
}

impl Live {
    fn act(&mut self, v: &mut Viewer, p: &mut Player<Bare>, a: Action) {
        self.now.set(self.now.get() + 10);
        self.rec.action(&a);
        v.apply(a);
        p.frame(v);
    }

    fn input(&mut self, v: &mut Viewer, p: &mut Player<Bare>, e: Event) {
        self.now.set(self.now.get() + 10);
        self.rec.event(&e);
        v.input(&e, false);
        p.frame(v);
    }

    /// Waits for the worker as the window's idle detection would, and notes it.
    fn idle(&mut self, v: &mut Viewer, p: &mut Player<Bare>) -> Observation {
        p.wait_idle(v).unwrap();
        self.rec.idle();
        observation(v.snapshot.as_ref().unwrap())
    }
}

/// A session driven live through `Viewer::input` / `Viewer::apply` and recorded, then replayed
/// from its file on a fresh viewer and worker, passes through the same snapshots at every
/// `wait idle` (analytic Morse H₂: paused edits and steps are deterministic).
#[test]
fn a_recorded_session_replays_to_the_same_snapshot_sequence() {
    let path = temp("session.replay");
    let now = Rc::new(Cell::new(0u64));
    let clock = now.clone();
    let rec = Recorder::with_clock(&path, (1400, 900), Box::new(move || Duration::from_millis(clock.get()))).unwrap();
    let mut s = Live { rec, now };
    let (mut v, mut host) = morse_h2();
    let mut live = Vec::new();
    {
        let mut p = Player::new(&mut host, (1400, 900));
        v.apply(Action::SmoothMotion(false)); // the recorder's first line
        s.act(&mut v, &mut p, Action::Tool(ToolMode::Select));
        let (x, y) = at(&v, 0);
        s.input(&mut v, &mut p, Event::Cursor { x, y });
        s.input(&mut v, &mut p, Event::Button { which: Button::Left, down: true });
        s.input(&mut v, &mut p, Event::Cursor { x: x + 0.1, y: y + 0.02 });
        s.input(&mut v, &mut p, Event::Cursor { x: x + 0.25, y: y + 0.05 });
        s.input(&mut v, &mut p, Event::Button { which: Button::Left, down: false });
        live.push(s.idle(&mut v, &mut p));
        for a in [Action::Step, Action::Step, Action::Pin(1, true), Action::Step, Action::Rewind, Action::Step] {
            s.act(&mut v, &mut p, a);
            live.push(s.idle(&mut v, &mut p));
        }
        s.input(&mut v, &mut p, Event::Key { key: KeyName::Delete, down: true });
        s.input(&mut v, &mut p, Event::Key { key: KeyName::Delete, down: false });
        live.push(s.idle(&mut v, &mut p));
    }
    drop(s);
    assert_eq!(live.len(), 8);
    assert!(live[0].positions[0] != [0.8, 0.0, 0.0], "the drag moved nucleus 0: {:?}", live[0].positions);
    assert_eq!(live[1].step, 1);
    assert_eq!(live.last().unwrap().positions.len(), 1, "Delete removed the selected nucleus");
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let (mut w, mut host2) = morse_h2();
    let replayed = Player::new(&mut host2, (1400, 900)).play(&mut w, &text).unwrap();
    assert_eq!(trace(&replayed), live, "{text}");
}

/// View coordinates (offsets from the 3D view's centre in half its height) hit the same nucleus
/// and land a placement on the same point at another window size — a different viewport rect
/// and aspect (the compact chrome below 1240 pt).
#[test]
fn view_coordinates_survive_a_window_size_change() {
    let mut v = Viewer::new("h2").unwrap();
    let mut host = Bare;
    let sizes = [(1400, 900), (1000, 640)];
    let mut viewports = Vec::new();
    let mut placed = Vec::new();
    let (n1, q) = {
        let mut p = Player::new(&mut host, sizes[0]);
        p.frame(&mut v);
        p.frame(&mut v);
        (at(&v, 1), (0.3f32, -0.2f32))
    };
    for size in sizes {
        let mut p = Player::new(&mut host, size);
        p.frame(&mut v);
        p.frame(&mut v);
        viewports.push(v.viewport);
        v.apply(Action::Tool(ToolMode::Select));
        v.input(&Event::Cursor { x: n1.0, y: n1.1 }, false);
        v.input(&Event::Button { which: Button::Left, down: true }, false);
        v.input(&Event::Button { which: Button::Left, down: false }, false);
        assert_eq!(v.tool.selected, Some(1), "{size:?}: {:?}", v.viewport);
        v.apply(Action::Element(8)); // and the Place tool
        v.input(&Event::Cursor { x: q.0, y: q.1 }, false);
        v.input(&Event::Button { which: Button::Left, down: true }, false);
        v.input(&Event::Button { which: Button::Left, down: false }, false);
        let o = v.scene.nuclei.pop().unwrap();
        assert_eq!(o.z, 8);
        placed.push(o.pos);
    }
    assert_ne!(viewports[0], viewports[1]);
    assert!((viewports[0].aspect() - viewports[1].aspect()).abs() > 0.05, "{viewports:?}");
    let d = (0..3).map(|k| (placed[0][k] - placed[1][k]).powi(2)).sum::<f64>().sqrt();
    // f32 unprojection (near 0.05, far 500 Bohr) carries ~1e-4 Bohr at 14 Bohr from the eye; the
    // pixel itself differs (the rects differ), so this is the coordinate frame's invariance to
    // that precision (measured 1.5e-4) — against a grid step of 0.25 Bohr
    assert!(d < 1e-3, "the same view point lands {d} Bohr apart: {placed:?}");
    // the conversion itself
    let vp = Viewport { origin: [0.0, 40.0], size: [1000.0, 700.0] };
    assert_eq!(to_view((500.0, 390.0), vp), (0.0, 0.0));
    assert_eq!(to_view((850.0, 40.0), vp), (1.0, -1.0));
    assert_eq!(to_pixels((1.0, -1.0), vp), (850.0, 40.0));
}

/// A failed expectation names its line, repeats it, and says what was observed.
#[test]
fn a_failing_expectation_fails_with_a_readable_message() {
    let (mut v, mut host) = morse_h2();
    let mut p = Player::new(&mut host, (1400, 900));
    assert!(p.play(&mut v, "expect nuclei 2\nexpect dist 0 1 1.59 1.61\nexpect no-error\n").is_ok());
    let e = p.play(&mut v, "# water?\nexpect nuclei 2\nexpect nuclei 3\n").unwrap_err();
    assert_eq!(e, "line 3: expect nuclei 3: the scene has 2 nuclei");
    let e = p.play(&mut v, "expect dist 0 1 2.0 2.5\n").unwrap_err();
    assert_eq!(e, "line 1: expect dist 0 1 2 2.5: nuclei 0 and 1 are 1.6000 Bohr apart");
    let e = p.play(&mut v, "expect selected 0\n").unwrap_err();
    assert_eq!(e, "line 1: expect selected 0: nothing is selected");
    let e = p.play(&mut v, "expect molecule 0 H2\n").unwrap_err();
    assert!(e.starts_with("line 1: expect molecule 0 H2: "), "{e}");
}

/// The window's winit events become the same `Event`s a script holds (the cursor in view
/// coordinates), and a Select press through `Viewer::input` — the only path — picks the nucleus.
/// "Only path" is not structural: nothing in the types stops code from calling `tools::pick` or
/// `Viewer::begin_drag` on its own. It holds today because `main.rs` has no input handling left —
/// every winit mouse and key event goes through `replay::from_winit` into `Viewer::input`, and the
/// chrome's controls through `ui::actions::apply` — and this test pins the conversion and the pick
/// that path relies on.
#[test]
fn viewer_input_is_the_only_path() {
    use winit::dpi::{PhysicalPosition, PhysicalSize};
    use winit::event::{DeviceId, ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
    use winit::keyboard::{Key, NamedKey};
    let mut v = Viewer::new("h2").unwrap();
    let mut host = Bare;
    Player::new(&mut host, (1400, 900)).frame(&mut v);
    v.apply(Action::Tool(ToolMode::Select));
    let px = tools::project(&v.camera, tools::v3(v.scene.nuclei[0].pos), v.viewport).unwrap();
    let vp = v.viewport;
    let id = DeviceId::dummy();
    let moved = WindowEvent::CursorMoved { device_id: id, position: PhysicalPosition::new(px.0 as f64, px.1 as f64) };
    let e = from_winit(&moved, vp).unwrap();
    assert_eq!(e, Event::Cursor { x: to_view(px, vp).0, y: to_view(px, vp).1 });
    v.input(&e, false);
    let press = WindowEvent::MouseInput { device_id: id, state: ElementState::Pressed, button: MouseButton::Left };
    let e = from_winit(&press, vp).unwrap();
    assert_eq!(e, Event::Button { which: Button::Left, down: true });
    v.input(&e, false);
    assert_eq!(v.tool.selected, Some(0));
    assert!(v.tool.dragging);
    // M4 fix round 1 m1: focus loss, or the cursor leaving the window, mid-drag is a release
    v.input(&Event::FocusLost, false);
    assert!(!v.tool.dragging && !v.mouse.left, "focus loss ends the drag");
    v.input(&Event::Button { which: Button::Left, down: true }, false);
    assert!(v.tool.dragging);
    v.input(&Event::CursorLeft, false);
    assert!(!v.tool.dragging && !v.mouse.left && v.tool.cursor.is_none(), "the cursor leaving ends the drag");
    let release = WindowEvent::MouseInput { device_id: id, state: ElementState::Released, button: MouseButton::Right };
    assert_eq!(from_winit(&release, vp), Some(Event::Button { which: Button::Right, down: false }));
    let back = WindowEvent::MouseInput { device_id: id, state: ElementState::Pressed, button: MouseButton::Back };
    assert_eq!(from_winit(&back, vp), None);
    let wheel = |delta| WindowEvent::MouseWheel { device_id: id, delta, phase: TouchPhase::Moved };
    assert_eq!(from_winit(&wheel(MouseScrollDelta::LineDelta(0.0, 2.0)), vp), Some(Event::Wheel { lines: 2.0 }));
    assert_eq!(from_winit(&wheel(MouseScrollDelta::PixelDelta(PhysicalPosition::new(0.0, -80.0))), vp), Some(Event::Wheel { lines: -2.0 }));
    assert_eq!(from_winit(&WindowEvent::CursorLeft { device_id: id }, vp), Some(Event::CursorLeft));
    assert_eq!(from_winit(&WindowEvent::Focused(false), vp), Some(Event::FocusLost));
    assert_eq!(from_winit(&WindowEvent::Focused(true), vp), None);
    assert_eq!(from_winit(&WindowEvent::Resized(PhysicalSize::new(800, 600)), vp), Some(Event::Resize { w: 800, h: 600 }));
    // keys (winit's `KeyEvent` cannot be built outside winit: the conversion's key half)
    assert_eq!(key_event(&Key::Named(NamedKey::Space), ElementState::Pressed, false), Some(Event::Key { key: KeyName::Space, down: true }));
    assert_eq!(key_event(&Key::Named(NamedKey::Space), ElementState::Pressed, true), None, "auto-repeat");
    assert_eq!(key_event(&Key::Named(NamedKey::Delete), ElementState::Released, false), Some(Event::Key { key: KeyName::Delete, down: false }));
    assert_eq!(key_event(&Key::Character("a".into()), ElementState::Pressed, false), None);
    // what egui consumed is recorded only when the 3D view still acts on it
    assert!(recordable(&Event::Cursor { x: 0.0, y: 0.0 }, true));
    assert!(recordable(&Event::Button { which: Button::Left, down: false }, true));
    assert!(!recordable(&Event::Button { which: Button::Left, down: true }, true));
    assert!(!recordable(&Event::Key { key: KeyName::Space, down: true }, true));
    assert!(!recordable(&Event::Wheel { lines: 1.0 }, true));
    assert!(recordable(&Event::Key { key: KeyName::Space, down: true }, false));
    // the same press through a consumed event does nothing
    let mut w = Viewer::new("h2").unwrap();
    Player::new(&mut host, (1400, 900)).frame(&mut w);
    w.apply(Action::Tool(ToolMode::Select));
    w.input(&Event::Cursor { x: to_view(px, vp).0, y: to_view(px, vp).1 }, true);
    w.input(&Event::Button { which: Button::Left, down: true }, true);
    assert_eq!(w.tool.selected, None);
}

/// A recorded action a script cannot carry: a save is skipped with a warning (it does not change
/// the run), anything else is refused with a message saying why and what to do.
#[test]
fn unreplayable_recorded_actions_are_skipped_or_refused() {
    let (mut v, mut host) = morse_h2();
    let mut p = Player::new(&mut host, (1400, 900));
    assert!(p.play(&mut v, "# not replayable: save-scene (…): SaveScene\nexpect nuclei 2\n").is_ok());
    let e = p.play(&mut v, "expect nuclei 2\n# not replayable: load-scene (…): LoadScene\n").unwrap_err();
    assert!(e.starts_with("line 2: # not replayable: load-scene: a recorded load-scene cannot be replayed") && e.contains("remove the line"), "{e}");
    for a in [Action::ApplyGrid, Action::LoadModel, Action::SampleAfterRefusal, Action::Sample { long: true }] {
        let kind = unreplayable_kind(&a).unwrap();
        assert!(!harmless(kind), "{kind}");
    }
    assert_eq!(unreplayable_kind(&Action::Play), None);
}

/// `mark` and what is counted from it: the largest distance seen, the fallbacks (none on an analytic
/// provider), the model's samples (none); a `scene` line loads a file relative to the script.
#[test]
fn mark_relative_expectations_and_scene_lines() {
    let (mut v, mut host) = morse_h2();
    let dir = temp("scene-dir");
    std::fs::create_dir_all(&dir).unwrap();
    let mut scene = crate::scene::h2_r16();
    scene.grid_n = 33;
    scene.nuclei[1].pos = [-1.2, 0.0, 0.0];
    std::fs::write(dir.join("wide h2.scene"), scene.to_text()).unwrap();
    let mut p = Player::new(&mut host, (1400, 900));
    p.base = Some(dir.clone());
    let obs = p.play(&mut v, "scene wide h2.scene\nwait idle\nexpect dist 0 1 1.99 2.01\nmark\naction play\nwait steps 30\naction pause\nwait idle\nexpect max-dist 0 1 1.999\nexpect fallbacks 0\nexpect new-samples 0\nexpect fallback-fraction 0\nexpect pull 0.1 0.1\n");
    let _ = std::fs::remove_dir_all(&dir);
    obs.unwrap();
    let e = p.play(&mut v, "mark\nexpect max-dist 0 1 5\n").unwrap_err();
    assert!(e.starts_with("line 2: expect max-dist 0 1 5: nuclei 0 and 1 were at most "), "{e}");
    let e = p.play(&mut v, "expect fallbacks 1\n").unwrap_err();
    assert_eq!(e, "line 1: expect fallbacks 1: 0 snapshots since the mark were answered by a fallback");
    let e = p.play(&mut v, "scene no-such.scene\n").unwrap_err();
    assert!(e.starts_with("line 1: scene no-such.scene: "), "{e}");
}

/// A resize keeps the viewport's margins until the next frame lays the chrome out (fixed-width
/// panels: the estimate is the chrome's own rect), so a cursor between the two maps through the new
/// rect: the same view point picks the same nucleus before and after that frame.
#[test]
fn a_cursor_after_a_resize_maps_through_the_new_viewport() {
    let mut v = Viewer::new("h2").unwrap();
    let mut host = Bare;
    {
        let mut p = Player::new(&mut host, (1400, 900));
        p.frame(&mut v);
        p.frame(&mut v);
    }
    let old = v.viewport;
    v.input(&Event::Resize { w: 1700, h: 1000 }, false);
    let estimate = v.viewport;
    assert_ne!(estimate, old);
    {
        let mut p = Player::new(&mut host, (1700, 1000));
        p.frame(&mut v);
        p.frame(&mut v);
    }
    let real = v.viewport;
    for k in 0..2 {
        assert!((estimate.origin[k] - real.origin[k]).abs() <= 1.0 && (estimate.size[k] - real.size[k]).abs() <= 1.0, "estimate {estimate:?} vs the chrome's {real:?}");
    }
}

/// Activity final review I3: `expect solve-log` and `expect counters` fail on bad input — no rows,
/// only pass rows, an unstamped fallback, a ledger that stopped ticking or ticks for rows the log
/// does not have, a store that did not grow, no model — and pass on a good one.
#[test]
fn the_activity_expectations_can_fail() {
    use crate::ui::activity::{ActivityState, LogRow};
    use sim::activity::{PassJob, TeachOrigin as O, TeachOutcome as K, TeachRule as R};
    let row = |origin, rule, outcome, step| {
        let mut e = sim::TeachEvent::new(origin, rule, outcome);
        e.step = step;
        LogRow { at: Duration::ZERO, event: e, near_face: false }
    };
    let push = |a: &mut ActivityState, r: LogRow| {
        a.rows.push_back(r);
        a.rows_seen += 1;
    };
    let mut a = ActivityState::new();
    let since = Since::default();
    assert!(solve_log_check(&a, 1, &since).is_err(), "no rows");
    push(&mut a, row(O::Pass, R::Pass(PassJob::Scan), K::Kept, None));
    assert!(solve_log_check(&a, 1, &since).is_err(), "a pass row is not a fallback");
    push(&mut a, row(O::Fallback, R::Grade, K::Kept, None));
    assert!(solve_log_check(&a, 1, &since).is_err(), "a fallback with no driver step");
    push(&mut a, row(O::Fallback, R::Grade, K::Kept, Some(3)));
    assert_eq!(solve_log_check(&a, 1, &since), Ok(()));
    assert!(solve_log_check(&a, 4, &since).is_err(), "fewer rows than asked");
    let later = Since { rows_seen: a.rows_seen, ..Default::default() };
    assert!(solve_log_check(&a, 1, &later).is_err(), "no rows since a later mark");
    // counters: three rows that count for the model, all kept
    let mut m = crate::ui::tests::model();
    m.samples = 10;
    m.solves = sim::SolveCounts { pass: 1, fallback: 2, kept: 3, ..Default::default() };
    assert_eq!(counters_check(&a, Some(&m), &since), Ok(()));
    let mut stopped = m.clone();
    stopped.solves = sim::SolveCounts::default();
    assert!(counters_check(&a, Some(&stopped), &since).is_err(), "the ledger stopped ticking");
    let mut extra = m.clone();
    extra.solves = sim::SolveCounts { pass: 4, fallback: 2, kept: 3, ..Default::default() };
    assert!(counters_check(&a, Some(&extra), &since).is_err(), "the ledger ticked for rows the log lacks");
    let mut small = m.clone();
    small.samples = 2;
    assert!(counters_check(&a, Some(&small), &since).is_err(), "three kept answers, a store that grew by two");
    assert!(counters_check(&a, None, &since).is_err(), "no model");
    // a mark taken after some solves counts only the growth
    let marked = Since { rows_seen: 1, model_solves: 1, samples: 8, ..Default::default() };
    assert_eq!(counters_check(&a, Some(&m), &marked), Ok(()));
}
