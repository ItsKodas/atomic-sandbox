use super::*;
use crate::tools;
use crate::ui::ToolMode;
use std::cell::Cell;
use std::rc::Rc;

/// The test host: the worker's events straight into the viewer, no renderer (the analytic
/// provider's snapshots carry no density).
struct Bare;

impl Host for Bare {
    fn pump(&mut self, v: &mut Viewer, timeout: Duration) -> bool {
        let Some(ev) = v.sim.as_ref().and_then(|w| w.recv_timeout(timeout)) else { return false };
        v.on_event_bare(ev);
        true
    }
}

/// The `h2` preset on an analytic Morse H₂ worker (no SCF, no GPU), idle after its first
/// evaluation, with a frame run so the viewport is the chrome's at 1400 × 900.
fn morse_h2() -> (Viewer, Bare) {
    let mut v = Viewer::new("h2").unwrap();
    v.scene.grid_n = 33;
    v.start_test(Box::new(|| Box::new(sim::MorseDimer::h2())));
    let mut host = Bare;
    let mut p = Player::new(&mut host, (1400, 900));
    p.frame(&mut v);
    p.wait_idle(&mut v).unwrap();
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
";

/// Spec §8.2: every directive parses, `write` is `parse`'s inverse, and an unknown word or a
/// missing argument is refused naming the line.
#[test]
fn the_replay_format_round_trips() {
    let lines = parse(EVERY_LINE).unwrap();
    assert_eq!(lines.len(), 32);
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
    ];
    for a in all {
        let text = a.describe().unwrap_or_else(|| panic!("{a:?} has no script form"));
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(Action::parse(&words).unwrap(), a, "{text}");
    }
    for a in [Action::ApplyGrid, Action::SaveScene, Action::LoadScene, Action::SaveModel, Action::LoadModel, Action::Sample { long: false }, Action::SampleAfterRefusal] {
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
            "# not replayable (its settings live in the panel, not in a script): SaveScene",
        ],
        "{text}"
    );
    assert_eq!(parse(&text).unwrap().len(), 11);
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
