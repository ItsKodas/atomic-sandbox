//! Input replay (M6 spec §8): the viewer's manual checklist as scripts that run headlessly.
//!
//! **One input path.** The 3D view's mouse and keys are device-independent [`Event`]s; the window
//! converts its winit events ([`from_winit`]) and feeds them to `Viewer::input`, which is also
//! what the replayer feeds. The chrome's controls are `ui::Action`s — the type every toolbar,
//! menu and dock widget already emits — run by `Viewer::apply` through the same
//! `ui::actions::apply` as a click; the egui layout itself is never replayed (a script names the
//! action, not the pixel of its button). Picks, placements and drags inside `Viewer::input` use
//! the snapshot mirror, never the interpolated display positions (S5).
//!
//! **Coordinates.** A cursor position is stored in *view coordinates*: its offset from the centre
//! of the 3D view's rect (`Viewer::viewport`, the space the chrome leaves) in units of half that
//! rect's height, +x right, +y down ([`to_view`]; the top and bottom edges are y = ∓1, the side
//! edges x = ∓aspect). The camera's field of view is vertical, so a view point is the same line
//! of sight at any window size, dock state or layout: a replay at another size picks the same
//! nucleus and places on the same point (`view_coordinates_survive_a_window_size_change`). Pan is
//! per view height and so invariant too; orbit is per pixel, so a replay at another size orbits by
//! a different angle — the replayer therefore renders at the script's `view` size unless told
//! otherwise.
//!
//! **Format** (text, one entry per line, `#` comments, like the scene file):
//! ```text
//! # atomic sandbox replay v1
//! view 1280 800                 # the window when recorded (physical pixels)
//! action smooth-motion false    # every script starts with it (S5)
//! scene ../my scenes/w.scene    # a scene file (relative to the script; the rest of the line)
//! t 120 cursor 0.25 -0.125      # t = recorded ms (informational), then the event
//! t 130 button left down        # left|right|middle down|up
//! wait ms 270                   # wall time: a recorded gap > 50 ms (a drag keeps its pace)
//! t 400 wheel -1                # lines, up positive
//! t 410 key space down          # space|delete down|up; also resize W H, cursor-left, focus-lost
//! action tool select            # any `Action::describe` form
//! wait idle                     # the worker idle, no relax running, no bond reference pending
//! wait steps 20                 # until the step counter advanced by 20 from where it stood
//! wait fs 2.5                   # until the simulated time advanced by 2.5 fs
//! wait source surrogate         # until the latest step was answered by it (polled)
//! mark                          # the baseline of max-dist, fallbacks, new-samples, new-epochs
//! expect nuclei 3               # also: paused B, selected I|none, dist I J MIN MAX (Bohr),
//! expect molecule 0 H2O         #   bonds N, bonds-stale B, molecule I FORMULA (ASCII digits),
//!                               #   no-error, isotope I A, relaxed B, source dft|surrogate|fallback,
//!                               #   pull MIN MAX (Ha/Bohr), fallback-fraction MAX (last 100 steps);
//!                               #   since the mark: max-dist I J MIN, fallbacks N, new-samples N,
//!                               #   new-epochs N, solve-log N (Activity tab rows); and counters
//!                               #   (the tab's samples counter is the model's, its session count
//!                               #   covers the solve rows since the mark)
//! # not replayable: apply-grid  # a recorded action without a script form (below)
//! ```
//!
//! **Actions a script cannot carry.** Apply grid, save / load scene or model, Sample, the offered
//! pass and "sample after refusal" mean panel state a script does not hold (the pending grid
//! fields, the file path, the sampling plan). The recorder writes them as `# not replayable:
//! <kind> …` and says so on stderr; the replayer skips a save with a warning (it does not change
//! the run) and refuses the others with a message — the replay would diverge from the session —
//! so a recorded script holding one is edited by hand (delete the line, or replace it with script
//! lines) before it replays.
//!
//! **Determinism.** A replay never waits on wall time except at `wait ms` (a recorded gap), and
//! asserts only on snapshot quantities with the tolerance written in the line. Dynamics advance by
//! `wait steps` / `wait fs` on the snapshot's counters; the thermostat is seeded (the worker's
//! fixed seed); the driver runs a fixed 16 ms tick schedule at `--speed` steps per tick (rule 8:
//! never dt). Paused edits, steps and relaxations are deterministic; input interleaved with a
//! *playing* run lands at a step that depends on the machine, which the expectations' ranges
//! must allow for (say why in the script).
//!
//! **Authoring by hand.** To aim at a world point p from the default camera (every preset starts
//! with it): `to_view(tools::project(&OrbitCamera::new(Vec3::ZERO, 14.0), p, vp).unwrap(), vp)` for
//! any viewport `vp` (the result does not depend on it); a placement lands on the camera's focus
//! plane through the origin, a drag on the plane through the nucleus parallel to it. The scripts in
//! `crates/app/replays` were written this way (each header names its world points).
//!
//! **Data.** A replay touches only its script, the optional `--screenshot` PNG and the estimated
//! solver's per-session autosave in the OS temp directory (deleted at the end). When the
//! user-data milestone gives the app a data root, `--record` and `--replay` must run with a
//! temporary one (`--data-dir <tmp>`), so a replay neither reads nor writes the user's profile,
//! models or scenes.
use crate::screenshot::Shot;
use crate::sim::{Backend, Snapshot};
use crate::ui::Action;
use crate::viewer::Viewer;
use render::camera::Viewport;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

/// The first line of every script.
pub const HEADER: &str = "# atomic sandbox replay v1";

/// The frame when a script names none (the window's default size).
pub const DEFAULT_VIEW: (u32, u32) = (1400, 900);

/// A gap between two recorded lines longer than this is written as `wait ms <gap>`.
const GAP: Duration = Duration::from_millis(50);

/// How long any one wait may take (a sampling pass ≈ 40 s, an O bond reference ≈ 70 s).
///
/// **Left at 600 s on purpose** (activity Tasks 6–9, the Task 5 carry): the far-drag replay's
/// `wait idle` takes ≈ 65 s alone, and under heavy load on the machine (a second build or suite
/// running) it can pass this wall-clock limit and fail the replay. That is a loaded machine
/// showing, not a replay to widen: run it alone (`cargo test -p app --release replays_pass`).
const WAIT_LIMIT: Duration = Duration::from_secs(600);

/// How often the replayer runs a chrome frame while it waits (the window's frame rate, about).
const FRAME_EVERY: Duration = Duration::from_millis(16);

#[derive(Clone, Debug, PartialEq)]
pub enum Button {
    Left,
    Right,
    Middle,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyName {
    Space,
    Delete,
}

/// One input event of the 3D view. `Cursor` is in view coordinates (module doc), `Resize` in
/// physical pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Cursor { x: f32, y: f32 },
    Button { which: Button, down: bool },
    Wheel { lines: f32 },
    Key { key: KeyName, down: bool },
    Resize { w: u32, h: u32 },
    CursorLeft,
    FocusLost,
}

/// One script line (module doc).
#[derive(Clone, Debug, PartialEq)]
pub enum Line {
    View(u32, u32),
    /// `scene <path>`: load a scene file (relative to the script's directory) as File ▸ Load
    /// does — the recorder's first line for a session started with `--scene`.
    Scene(String),
    At(u64, Event),
    Action(Action),
    /// `# not replayable: <kind> …`: a recorded chrome action whose meaning is panel state a
    /// script does not carry ([`unreplayable_kind`]). The replayer warns and skips a save (it does
    /// not change the run) and refuses the rest (the replay would diverge from the session).
    NotReplayable(String),
    /// `mark`: the baseline of the `max-dist`, `fallbacks`, `new-samples` and `new-epochs`
    /// expectations (the script's start until the first mark).
    Mark,
    WaitIdle,
    WaitMs(u64),
    WaitSteps(u64),
    WaitFs(f64),
    /// `wait source S`: until the latest step was answered by S (polled each snapshot).
    WaitSource(String),
    Expect(Expect),
}

/// What an `expect` line asserts, on the viewer after the waits before it.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    /// The scene's nucleus count (the mirror).
    Nuclei(usize),
    /// The snapshot's run state.
    Paused(bool),
    /// The selection.
    Selected(Option<usize>),
    /// Nuclei i and j are between min and max Bohr apart (the snapshot's positions).
    Dist(usize, usize, f64, f64),
    /// The bonds of the snapshot's graph (0 without one, or when it is not this scene's).
    Bonds(usize),
    /// Molecule i's formula, typed in ASCII (`H2O` for H₂O).
    Molecule(usize, String),
    /// Neither the viewer nor the run has an error.
    NoError,
    /// Nucleus i's mass number.
    Isotope(usize, u16),
    /// The relax readout ended converged (true) or not (false: none, running or stopped).
    Relaxed(bool),
    /// Who answered the latest step: `dft`, `surrogate` or `fallback`.
    Source(String),
    /// The largest distance between nuclei i and j in any snapshot since the mark is at least min
    /// Bohr (a drag really stretched the bond).
    MaxDist(usize, usize, f64),
    /// At least n snapshots since the mark had their latest step answered by a DFT fallback.
    Fallbacks(usize),
    /// The estimated solver's model gained at least n samples / refits (epochs) since the mark.
    NewSamples(usize),
    NewEpochs(u64),
    /// At most this fraction of the last ≤ 100 steps fell back (`fallback_fraction_100`): a ratio
    /// that does not flicker with the single latest step while playing.
    FallbackFraction(f64),
    /// The driver's drag pull (`drag_force_max`, Ha/Bohr) is between lo and hi.
    Pull(f64, f64),
    /// The bond graph is (true) or is not (false) an earlier DFT step's (`bonds_stale`).
    BondsStale(bool),
    /// At least n Activity-tab solve-log rows arrived since the mark, each with a rule label and
    /// an outcome label (activity spec §8).
    SolveLog(usize),
    /// The Activity tab's samples counter equals `ModelStatus::samples`, and its teaching-solves
    /// count has grown by at least the solve rows (`TeachEvent::is_solve`) since the mark.
    Counters,
}

// ---- coordinates ----

/// Window physical pixels → view coordinates (module doc).
pub fn to_view((px, py): (f32, f32), vp: Viewport) -> (f32, f32) {
    let half = vp.size[1] / 2.0;
    ((px - (vp.origin[0] + vp.size[0] / 2.0)) / half, (py - (vp.origin[1] + half)) / half)
}

/// View coordinates → window physical pixels, the inverse of [`to_view`].
pub fn to_pixels((x, y): (f32, f32), vp: Viewport) -> (f32, f32) {
    let half = vp.size[1] / 2.0;
    (vp.origin[0] + vp.size[0] / 2.0 + x * half, vp.origin[1] + half + y * half)
}

// ---- text ----

fn button_word(b: &Button) -> &'static str {
    match b {
        Button::Left => "left",
        Button::Right => "right",
        Button::Middle => "middle",
    }
}

fn updown(down: bool) -> &'static str {
    if down { "down" } else { "up" }
}

impl Event {
    /// The event's words in a `t` line.
    pub fn describe(&self) -> String {
        match self {
            Event::Cursor { x, y } => format!("cursor {x} {y}"),
            Event::Button { which, down } => format!("button {} {}", button_word(which), updown(*down)),
            Event::Wheel { lines } => format!("wheel {lines}"),
            Event::Key { key, down } => format!("key {} {}", if *key == KeyName::Space { "space" } else { "delete" }, updown(*down)),
            Event::Resize { w, h } => format!("resize {w} {h}"),
            Event::CursorLeft => "cursor-left".into(),
            Event::FocusLost => "focus-lost".into(),
        }
    }

    fn parse(w: &[&str]) -> Result<Event, String> {
        let down = |s: &str| match s {
            "down" => Ok(true),
            "up" => Ok(false),
            _ => Err(format!("expected down or up, got '{s}'")),
        };
        Ok(match w {
            ["cursor", x, y] => Event::Cursor { x: number(x)?, y: number(y)? },
            ["button", b, d] => Event::Button {
                which: match *b {
                    "left" => Button::Left,
                    "right" => Button::Right,
                    "middle" => Button::Middle,
                    _ => return Err(format!("button left, right or middle, got '{b}'")),
                },
                down: down(d)?,
            },
            ["wheel", l] => Event::Wheel { lines: number(l)? },
            ["key", k, d] => Event::Key {
                key: match *k {
                    "space" => KeyName::Space,
                    "delete" => KeyName::Delete,
                    _ => return Err(format!("key space or delete, got '{k}'")),
                },
                down: down(d)?,
            },
            ["resize", w, h] => Event::Resize { w: number(w)?, h: number(h)? },
            ["cursor-left"] => Event::CursorLeft,
            ["focus-lost"] => Event::FocusLost,
            _ => return Err(format!("unknown event '{}'", w.join(" "))),
        })
    }
}

impl Expect {
    fn describe(&self) -> String {
        match self {
            Expect::Nuclei(n) => format!("nuclei {n}"),
            Expect::Paused(b) => format!("paused {b}"),
            Expect::Selected(i) => format!("selected {}", i.map_or("none".into(), |i| i.to_string())),
            Expect::Dist(i, j, lo, hi) => format!("dist {i} {j} {lo} {hi}"),
            Expect::Bonds(n) => format!("bonds {n}"),
            Expect::Molecule(i, f) => format!("molecule {i} {f}"),
            Expect::NoError => "no-error".into(),
            Expect::Isotope(i, a) => format!("isotope {i} {a}"),
            Expect::Relaxed(b) => format!("relaxed {b}"),
            Expect::Source(s) => format!("source {s}"),
            Expect::MaxDist(i, j, min) => format!("max-dist {i} {j} {min}"),
            Expect::Fallbacks(n) => format!("fallbacks {n}"),
            Expect::NewSamples(n) => format!("new-samples {n}"),
            Expect::NewEpochs(n) => format!("new-epochs {n}"),
            Expect::FallbackFraction(x) => format!("fallback-fraction {x}"),
            Expect::Pull(lo, hi) => format!("pull {lo} {hi}"),
            Expect::BondsStale(b) => format!("bonds-stale {b}"),
            Expect::SolveLog(n) => format!("solve-log {n}"),
            Expect::Counters => "counters".into(),
        }
    }

    fn parse(w: &[&str]) -> Result<Expect, String> {
        let flag = |s: &str| match s {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(format!("expected true or false, got '{s}'")),
        };
        Ok(match w {
            ["nuclei", n] => Expect::Nuclei(number(n)?),
            ["paused", b] => Expect::Paused(flag(b)?),
            ["selected", "none"] => Expect::Selected(None),
            ["selected", i] => Expect::Selected(Some(number(i)?)),
            ["dist", i, j, lo, hi] => Expect::Dist(number(i)?, number(j)?, number(lo)?, number(hi)?),
            ["bonds", n] => Expect::Bonds(number(n)?),
            ["molecule", i, f] => Expect::Molecule(number(i)?, f.to_string()),
            ["no-error"] => Expect::NoError,
            ["isotope", i, a] => Expect::Isotope(number(i)?, number(a)?),
            ["relaxed", b] => Expect::Relaxed(flag(b)?),
            ["source", s] => Expect::Source(source_word(s)?),
            ["max-dist", i, j, min] => Expect::MaxDist(number(i)?, number(j)?, number(min)?),
            ["fallbacks", n] => Expect::Fallbacks(number(n)?),
            ["new-samples", n] => Expect::NewSamples(number(n)?),
            ["new-epochs", n] => Expect::NewEpochs(number(n)?),
            ["fallback-fraction", x] => Expect::FallbackFraction(number(x)?),
            ["pull", lo, hi] => Expect::Pull(number(lo)?, number(hi)?),
            ["bonds-stale", b] => Expect::BondsStale(flag(b)?),
            ["solve-log", n] => Expect::SolveLog(number(n)?),
            ["counters"] => Expect::Counters,
            ["dist", ..] => return Err("expect dist needs i j min max".into()),
            _ => return Err(format!("unknown or incomplete expectation 'expect {}'", w.join(" "))),
        })
    }
}

fn number<T: std::str::FromStr>(w: &str) -> Result<T, String> {
    w.parse().map_err(|_| format!("bad number '{w}'"))
}

fn source_word(s: &str) -> Result<String, String> {
    match s {
        "dft" | "surrogate" | "fallback" => Ok(s.to_string()),
        _ => Err(format!("source dft, surrogate or fallback, got '{s}'")),
    }
}

fn source_name(s: sim::Source) -> &'static str {
    match s {
        sim::Source::Dft => "dft",
        sim::Source::Surrogate => "surrogate",
        sim::Source::Fallback => "fallback",
    }
}

/// The comment prefix of a recorded action a script cannot replay.
const NOT_REPLAYABLE: &str = "# not replayable:";

/// The kind word of an action without a script form (`Action::describe` is `None`), written after
/// [`NOT_REPLAYABLE`]; `None` for a replayable action.
pub fn unreplayable_kind(a: &Action) -> Option<&'static str> {
    if a.describe().is_some() {
        return None;
    }
    Some(match a {
        Action::ApplyGrid => "apply-grid",
        Action::SaveScene => "save-scene",
        Action::LoadScene { .. } => "load-scene",
        Action::SaveModel => "save-model",
        Action::LoadModel => "load-model",
        Action::Sample { .. } => "sample",
        Action::StartOffered(_) => "start-offered",
        Action::SampleAfterRefusal => "sample-after-refusal",
        _ => "action",
    })
}

/// A skipped recorded save changes nothing the replay checks; any other unreplayable action would
/// make the replay diverge from the session.
fn harmless(kind: &str) -> bool {
    matches!(kind, "save-scene" | "save-model")
}

/// One line's text (the inverse of the parser; an action without a script form becomes a comment).
pub fn write_line(l: &Line) -> String {
    match l {
        Line::View(w, h) => format!("view {w} {h}"),
        Line::Scene(path) => format!("scene {path}"),
        Line::At(t, e) => format!("t {t} {}", e.describe()),
        Line::Action(a) => match (a.describe(), unreplayable_kind(a)) {
            (Some(words), _) => format!("action {words}"),
            (None, kind) => format!("{NOT_REPLAYABLE} {} (its settings live in the panel, not in a script): {a:?}", kind.unwrap_or("action")),
        },
        Line::NotReplayable(kind) => format!("{NOT_REPLAYABLE} {kind}"),
        Line::Mark => "mark".into(),
        Line::WaitSource(src) => format!("wait source {src}"),
        Line::WaitIdle => "wait idle".into(),
        Line::WaitMs(n) => format!("wait ms {n}"),
        Line::WaitSteps(n) => format!("wait steps {n}"),
        Line::WaitFs(x) => format!("wait fs {x}"),
        Line::Expect(e) => format!("expect {}", e.describe()),
    }
}

fn parse_line(words: &[&str]) -> Result<Line, String> {
    Ok(match words {
        ["view", w, h] => Line::View(number(w)?, number(h)?),
        ["view", ..] => return Err("view needs a width and a height".into()),
        ["t", t, rest @ ..] => Line::At(number(t)?, Event::parse(rest)?),
        ["action", rest @ ..] => Line::Action(Action::parse(rest)?),
        ["wait", "idle"] => Line::WaitIdle,
        ["wait", "ms", n] => Line::WaitMs(number(n)?),
        ["wait", "steps", n] => Line::WaitSteps(number(n)?),
        ["wait", "fs", x] => Line::WaitFs(number(x)?),
        ["wait", "source", src] => Line::WaitSource(source_word(src)?),
        ["wait", ..] => return Err(format!("wait idle, ms N, steps N, fs X or source S, got '{}'", words.join(" "))),
        ["mark"] => Line::Mark,
        ["expect", rest @ ..] => Line::Expect(Expect::parse(rest)?),
        [other, ..] => return Err(format!("unknown directive '{other}'")),
        [] => unreachable!("blank lines are skipped"),
    })
}

/// The script's lines with their 1-based line numbers; an error names the line.
pub fn parse_numbered(text: &str) -> Result<Vec<(usize, Line)>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        if let Some(rest) = raw.trim_start().strip_prefix(NOT_REPLAYABLE) {
            let kind = rest.split_whitespace().next().unwrap_or("action");
            out.push((i + 1, Line::NotReplayable(kind.to_string())));
            continue;
        }
        let body = raw.split('#').next().unwrap_or("").trim();
        // a scene path may hold spaces: the rest of the line
        if let Some(path) = body.strip_prefix("scene ").map(str::trim).filter(|p| !p.is_empty()) {
            out.push((i + 1, Line::Scene(path.to_string())));
            continue;
        }
        let words: Vec<&str> = body.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        out.push((i + 1, parse_line(&words).map_err(|e| format!("line {}: {e}", i + 1))?));
    }
    Ok(out)
}

/// The script's lines (`"line N: …"` errors).
#[cfg_attr(not(test), allow(dead_code))] // the format's API; the app itself runs `parse_numbered`
pub fn parse(text: &str) -> Result<Vec<Line>, String> {
    Ok(parse_numbered(text)?.into_iter().map(|(_, l)| l).collect())
}

/// A script: the header, then one line per entry; `parse(&write(l)) == l`.
#[cfg_attr(not(test), allow(dead_code))] // the format's API; the recorder writes line by line
pub fn write(lines: &[Line]) -> String {
    let mut s = format!("{HEADER}\n");
    for l in lines {
        s += &write_line(l);
        s.push('\n');
    }
    s
}

// ---- the window's events ----

/// A winit window event as the 3D view's [`Event`] (the cursor through the frame's viewport into
/// view coordinates); `None` for what the view does not use.
pub fn from_winit(e: &winit::event::WindowEvent, vp: Viewport) -> Option<Event> {
    use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
    Some(match e {
        WindowEvent::CursorMoved { position, .. } => {
            let (x, y) = to_view((position.x as f32, position.y as f32), vp);
            Event::Cursor { x, y }
        }
        WindowEvent::CursorLeft { .. } => Event::CursorLeft,
        WindowEvent::Focused(false) => Event::FocusLost,
        WindowEvent::MouseInput { state, button, .. } => Event::Button {
            which: match button {
                MouseButton::Left => Button::Left,
                MouseButton::Right => Button::Right,
                MouseButton::Middle => Button::Middle,
                _ => return None,
            },
            down: state.is_pressed(),
        },
        // a pixel wheel's 40 px count as a line
        WindowEvent::MouseWheel { delta, .. } => Event::Wheel {
            lines: match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
            },
        },
        WindowEvent::KeyboardInput { event, .. } => return key_event(&event.logical_key, event.state, event.repeat),
        WindowEvent::Resized(s) => Event::Resize { w: s.width, h: s.height },
        _ => return None,
    })
}

/// The key half of [`from_winit`]: Space and Delete, presses and releases; auto-repeats are
/// dropped (a held Space would toggle play at the repeat rate).
pub fn key_event(key: &winit::keyboard::Key, state: winit::event::ElementState, repeat: bool) -> Option<Event> {
    use winit::keyboard::{Key, NamedKey};
    if repeat {
        return None;
    }
    let key = match key {
        Key::Named(NamedKey::Space) => KeyName::Space,
        Key::Named(NamedKey::Delete) => KeyName::Delete,
        _ => return None,
    };
    Some(Event::Key { key, down: state.is_pressed() })
}

/// Whether the recorder keeps an event egui `consumed` (the pointer over the chrome, a text field
/// focused): only what `Viewer::input` still acts on — motion, releases, the cursor leaving, focus
/// loss, a resize. A consumed press, wheel step or key never reaches the 3D view, so a replay
/// (which feeds nothing to egui) must not see it either.
pub fn recordable(e: &Event, consumed: bool) -> bool {
    !consumed
        || matches!(
            e,
            Event::Cursor { .. } | Event::Button { down: false, .. } | Event::CursorLeft | Event::FocusLost | Event::Resize { .. }
        )
}

// ---- the recorder ----

/// `--record <path>`: writes the window's session as a script while it runs — the view size, every
/// input event the 3D view acts on, every chrome action, `wait ms` for a gap of more than 50 ms
/// between two lines, and `wait idle` once each time the worker went idle after something was
/// sent. Each line is flushed as it is written (the app exits without unwinding).
pub struct Recorder {
    out: Box<dyn Write>,
    clock: Box<dyn FnMut() -> Duration>,
    /// The time of the last line written.
    last: Duration,
    /// Something was written since the last `wait idle`.
    pending: bool,
    /// The first write error (reported once; the session goes on unrecorded).
    failed: bool,
}

impl Recorder {
    /// A recorder on `path` timed by the wall clock from now.
    pub fn new(path: &Path, view: (u32, u32)) -> std::io::Result<Recorder> {
        let t0 = Instant::now();
        Recorder::with_clock(path, view, Box::new(move || t0.elapsed()))
    }

    /// A recorder with its own clock (the tests'). Writes the header, `view` and
    /// `action smooth-motion false`.
    pub fn with_clock(path: &Path, view: (u32, u32), mut clock: Box<dyn FnMut() -> Duration>) -> std::io::Result<Recorder> {
        let mut out: Box<dyn Write> = Box::new(std::io::BufWriter::new(std::fs::File::create(path)?));
        writeln!(out, "{HEADER}")?;
        writeln!(out, "{}", write_line(&Line::View(view.0, view.1)))?;
        writeln!(out, "{}", write_line(&Line::Action(Action::SmoothMotion(false))))?;
        out.flush()?;
        let last = clock();
        Ok(Recorder { out, clock, last, pending: false, failed: false })
    }

    fn line(&mut self, text: &str) {
        if self.failed {
            return;
        }
        if let Err(e) = writeln!(self.out, "{text}").and_then(|_| self.out.flush()) {
            eprintln!("--record: write failed ({e}); the rest of the session is not recorded");
            self.failed = true;
        }
    }

    /// The time now (ms since the recorder started), after a `wait ms` for a long gap.
    fn stamp(&mut self) -> u64 {
        let now = (self.clock)();
        let gap = now.saturating_sub(self.last);
        if gap > GAP {
            self.line(&write_line(&Line::WaitMs(gap.as_millis() as u64)));
        }
        self.last = now;
        now.as_millis() as u64
    }

    /// One input event the 3D view acted on (see [`recordable`]).
    pub fn event(&mut self, e: &Event) {
        let t = self.stamp();
        self.line(&write_line(&Line::At(t, e.clone())));
        self.pending = true;
    }

    /// One chrome action (`Viewer::chrome_actions`). One without a script form is written as a
    /// `# not replayable: <kind>` comment and reported on stderr (the replay will warn about it, or
    /// refuse it when it changes the run).
    pub fn action(&mut self, a: &Action) {
        self.stamp();
        if let Some(kind) = unreplayable_kind(a) {
            eprintln!(
                "--record: {kind} cannot be replayed (its settings live in the panel, not in a script); written as a comment — {}",
                if harmless(kind) { "a replay skips it with a warning" } else { "a replay of this script refuses it" }
            );
        }
        self.line(&write_line(&Line::Action(a.clone())));
        self.pending = true;
    }

    /// `scene <path>`: the session started from a scene file (`--scene`).
    pub fn scene(&mut self, path: &str) {
        self.stamp();
        self.line(&write_line(&Line::Scene(path.to_string())));
        self.pending = true;
    }

    /// The worker went idle ([`idle`]): `wait idle`, once per idle gap.
    pub fn idle(&mut self) {
        if self.pending {
            self.line(&write_line(&Line::WaitIdle));
            self.pending = false;
            self.last = (self.clock)();
        }
    }
}

// ---- waiting and checking ----

/// The snapshot's half of `wait idle`: the worker has nothing to do, no relax is running and no
/// bond reference is pending (they solve in the background: `Snapshot::working` does not count
/// them).
pub fn snapshot_idle(s: &Snapshot) -> bool {
    !s.working && !crate::ui::relaxing(&s.dynamics) && s.dynamics.bonds_pending.is_empty()
}

/// `wait idle`: the viewer's worker is settled (no retiring estimated worker, its `EnsureModel`
/// sent), reported Busy(false), and its latest snapshot has seen every command sent
/// (`Snapshot::commands`) and is [`snapshot_idle`].
pub fn idle(v: &Viewer) -> bool {
    v.settled() && !v.busy && v.snapshot.as_ref().is_some_and(|s| s.commands == v.commands_sent() && snapshot_idle(s))
}

/// The snapshot's latest command count matches the viewer's: it reflects everything sent.
fn synced(v: &Viewer) -> bool {
    v.snapshot.as_ref().is_some_and(|s| s.commands == v.commands_sent())
}

/// What the replayer saw after a wait.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    pub step: u64,
    pub sim_time_fs: f64,
    pub paused: bool,
    pub positions: Vec<[f64; 3]>,
}

pub(crate) fn observation(s: &Snapshot) -> Observation {
    Observation {
        step: s.dynamics.step,
        sim_time_fs: s.dynamics.sim_time_fs(),
        paused: s.dynamics.paused,
        positions: s.scene.nuclei.iter().map(|n| n.pos).collect(),
    }
}

/// A formula as typed in a script: subscript and superscript digits and charges in ASCII.
fn ascii_formula(f: &str) -> String {
    f.chars()
        .map(|c| match c {
            '₀'..='₉' => char::from(b'0' + (c as u32 - '₀' as u32) as u8),
            '⁰' => '0',
            '¹' => '1',
            '²' => '2',
            '³' => '3',
            '⁴'..='⁹' => char::from(b'4' + (c as u32 - '⁴' as u32) as u8),
            '⁺' => '+',
            '⁻' => '-',
            c => c,
        })
        .collect()
}

/// Checks one expectation; `Err` says what was observed.
fn check(v: &Viewer, e: &Expect, since: &Since) -> Result<(), String> {
    let snap = || v.snapshot.as_ref().ok_or_else(|| "no snapshot yet".to_string());
    let plural = |n: usize, one: &str| format!("{n} {one}{}", if n == 1 { "" } else { "s" });
    match e {
        Expect::Nuclei(n) => {
            let got = v.scene.nuclei.len();
            if got != *n {
                return Err(format!("the scene has {got} nuclei"));
            }
        }
        Expect::Paused(b) => {
            let paused = snap()?.dynamics.paused;
            if paused != *b {
                return Err(format!("the run is {}", if paused { "paused" } else { "playing" }));
            }
        }
        Expect::Selected(i) => {
            if v.tool.selected != *i {
                return Err(match v.tool.selected {
                    Some(s) => format!("nucleus {s} is selected"),
                    None => "nothing is selected".into(),
                });
            }
        }
        Expect::Dist(i, j, lo, hi) => {
            let s = snap()?;
            let n = &s.scene.nuclei;
            let (Some(a), Some(b)) = (n.get(*i), n.get(*j)) else {
                return Err(format!("the snapshot has {} nuclei", n.len()));
            };
            let d = (0..3).map(|k| (a.pos[k] - b.pos[k]).powi(2)).sum::<f64>().sqrt();
            if !(*lo..=*hi).contains(&d) {
                return Err(format!("nuclei {i} and {j} are {d:.4} Bohr apart"));
            }
        }
        Expect::Bonds(n) => {
            let s = snap()?;
            let got = match &s.dynamics.bonds {
                Some(g) if g.n_nuclei == s.scene.nuclei.len() => g.bonds.len(),
                Some(g) => return Err(format!("the bond graph is for {} nuclei, the scene has {}", g.n_nuclei, s.scene.nuclei.len())),
                None => return Err(format!("no bond graph yet (references pending: {:?})", s.dynamics.bonds_pending)),
            };
            if got != *n {
                return Err(format!("the graph has {}", plural(got, "bond")));
            }
        }
        Expect::Molecule(i, f) => {
            let s = snap()?;
            let Some(name) = s.labels.get(*i) else {
                return Err(format!("{} named", plural(s.labels.len(), "molecule")));
            };
            if ascii_formula(&name.formula) != *f {
                return Err(format!("molecule {i} is {} ({:?})", name.formula, name.text));
            }
        }
        Expect::NoError => {
            if let Some(e) = &v.last_error {
                return Err(format!("the viewer shows: {e}"));
            }
            if let Some(e) = &snap()?.dynamics.last_error {
                return Err(format!("the run reports: {e}"));
            }
        }
        Expect::Isotope(i, a) => {
            let s = snap()?;
            let Some(n) = s.scene.nuclei.get(*i) else { return Err(format!("no nucleus {i}")) };
            if n.mass_number() != Some(*a) {
                return Err(format!("nucleus {i} has A = {:?}", n.mass_number()));
            }
        }
        Expect::Relaxed(b) => {
            let r = snap()?.dynamics.relax.clone();
            let converged = r.as_ref().and_then(|r| r.converged) == Some(true);
            if converged != *b {
                return Err(match r {
                    None => "no relax has run".into(),
                    Some(r) => crate::ui::relax_lines(&r).join(" / "),
                });
            }
        }
        Expect::Source(want) => {
            let got = source_name(snap()?.dynamics.source);
            if got != want {
                return Err(format!("the latest step was answered by {got}"));
            }
        }
        Expect::MaxDist(i, j, min) => {
            let d = since.max_d.get(&(*i.min(j), *i.max(j))).copied();
            if d.is_none_or(|d| d < *min) {
                return Err(match d {
                    Some(d) => format!("nuclei {i} and {j} were at most {d:.4} Bohr apart since the mark"),
                    None => format!("no snapshot since the mark had nuclei {i} and {j}"),
                });
            }
            // the margin over the bound, for the log (a near miss shows before it becomes a flake)
            eprintln!("replay: max-dist {i} {j} = {:.4} Bohr since the mark (bound {min})", d.unwrap_or_default());
        }
        Expect::Fallbacks(n) => {
            if since.fallbacks < *n {
                return Err(format!("{} snapshots since the mark were answered by a fallback", since.fallbacks));
            }
        }
        Expect::NewSamples(n) => {
            let now = snap()?.model.as_ref().map_or(0, |m| m.samples);
            if now < since.samples + n {
                return Err(format!("the model has {now} samples, {} at the mark", since.samples));
            }
        }
        Expect::NewEpochs(n) => {
            let now = snap()?.model.as_ref().map_or(0, |m| m.epoch);
            if now < since.epoch + n {
                return Err(format!("the model is at epoch {now}, {} at the mark", since.epoch));
            }
        }
        Expect::FallbackFraction(max) => {
            let f = snap()?.dynamics.fallback_fraction_100;
            if f > *max {
                return Err(format!("{:.0} % of the last 100 steps fell back", 100.0 * f));
            }
        }
        Expect::Pull(lo, hi) => {
            let f = snap()?.dynamics.drag_force_max;
            if !(*lo..=*hi).contains(&f) {
                return Err(format!("the drag pull is {f:.4} Ha/Bohr"));
            }
        }
        Expect::SolveLog(n) => {
            let a = &v.ui.activity;
            let new = (a.rows_seen - since.rows_seen) as usize;
            if new < *n {
                return Err(format!("{new} solve-log rows since the mark"));
            }
            for r in a.rows.iter().rev().take(new) {
                let (rule, outcome) = (crate::ui::activity::words::rule_label(r.event.rule, r.event.burst_t), crate::ui::activity::words::outcome_label(r.event.outcome));
                if rule.is_empty() || outcome.is_empty() {
                    return Err(format!("a solve-log row has no rule or outcome label: {:?}", r.event.rule));
                }
            }
        }
        Expect::Counters => {
            let a = &v.ui.activity;
            let model = snap()?.model.as_ref();
            let values = crate::ui::activity::log::counter_values(a, model);
            if values.samples != model.map(|m| m.samples) {
                return Err(format!("the samples counter shows {:?}, the model holds {:?}", values.samples, model.map(|m| m.samples)));
            }
            let new = (a.rows_seen - since.rows_seen) as usize;
            let solves = a.rows.iter().rev().take(new).filter(|r| r.event.is_solve()).count() as u64;
            let grew = a.session_solves - since.solves;
            if grew < solves {
                return Err(format!("the session counter grew by {grew}, the log has {solves} solve rows since the mark"));
            }
        }
        Expect::BondsStale(b) => {
            let s = snap()?;
            if s.dynamics.bonds.is_none() {
                return Err("no bond graph yet".into());
            }
            if s.dynamics.bonds_stale != *b {
                return Err(format!("the bonds are {}", if s.dynamics.bonds_stale { "an earlier DFT step's (stale)" } else { "this step's (fresh)" }));
            }
        }
    }
    Ok(())
}

/// What the `max-dist`, `fallbacks`, `new-samples` and `new-epochs` expectations count from: the
/// model's samples and epoch at the mark, then every snapshot the replayer received since.
#[derive(Default)]
pub(crate) struct Since {
    samples: usize,
    epoch: u64,
    /// The Activity tab's events and teaching solves seen at the mark.
    rows_seen: u64,
    solves: u64,
    /// Snapshots whose new step was answered by a fallback.
    fallbacks: usize,
    /// The step of the last snapshot counted (a paused run republishes its last step's source).
    last_step: Option<u64>,
    /// The largest distance of each pair (i < j) seen.
    max_d: std::collections::HashMap<(usize, usize), f64>,
}

impl Since {
    fn mark(v: &Viewer) -> Since {
        let model = v.snapshot.as_ref().and_then(|s| s.model.as_ref());
        let mut since = Since {
            samples: model.map_or(0, |m| m.samples),
            epoch: model.map_or(0, |m| m.epoch),
            rows_seen: v.ui.activity.rows_seen,
            solves: v.ui.activity.session_solves,
            ..Default::default()
        };
        if let Some(s) = &v.snapshot {
            since.last_step = Some(s.dynamics.step);
            since.note(s);
        }
        since
    }

    fn note(&mut self, s: &Snapshot) {
        let d = &s.dynamics;
        if self.last_step != Some(d.step) {
            self.last_step = Some(d.step);
            self.fallbacks += usize::from(d.source == sim::Source::Fallback);
        }
        let n = &s.scene.nuclei;
        for i in 0..n.len() {
            for j in i + 1..n.len() {
                let r = (0..3).map(|k| (n[i].pos[k] - n[j].pos[k]).powi(2)).sum::<f64>().sqrt();
                let m = self.max_d.entry((i, j)).or_insert(r);
                *m = m.max(r);
            }
        }
    }
}

// ---- the replayer ----

/// Where the worker's events go: the viewer with a renderer (the GPU run) or without one (the
/// tests' analytic worker).
pub(crate) trait Host {
    /// Hands the viewer the worker's next event, waiting up to `timeout`: `None` if none came,
    /// else whether it was a snapshot.
    fn pump(&mut self, v: &mut Viewer, timeout: Duration) -> Option<bool>;
}

/// The headless frame's device and painter receive the densities, as the window's do.
struct ShotHost<'a> {
    shot: &'a mut Shot,
}

impl Host for ShotHost<'_> {
    fn pump(&mut self, v: &mut Viewer, timeout: Duration) -> Option<bool> {
        let ev = v.sim.as_ref().and_then(|w| w.recv_timeout(timeout))?;
        let snapshot = matches!(ev, crate::sim::Event::Snapshot(_));
        v.on_event(ev, &self.shot.hl.ctx, &mut self.shot.painter, &self.shot.hl.depth);
        Some(snapshot)
    }
}

/// Runs script lines against a viewer: events through `Viewer::input`, actions through
/// `Viewer::apply`, a chrome frame (`run_ui`, not painted) after each and every ~16 ms while
/// waiting — so the viewport, queued actions and the estimated solver's hand-over behave as in
/// the window.
pub(crate) struct Player<'h, H: Host> {
    host: &'h mut H,
    /// The window size the chrome is laid out for (a script's `view` and `resize` lines).
    pub screen: (u32, u32),
    /// The size was given from outside (`--size`): `view` lines do not change it.
    pub fixed: bool,
    /// egui texture updates of the unpainted frames (the screenshot applies them).
    pub textures: egui::TexturesDelta,
    /// Where a `scene` line's relative path starts (the script's directory).
    pub base: Option<std::path::PathBuf>,
    /// The `mark` baseline and what was seen since.
    since: Since,
}

/// The unpainted frames' texture updates are freed with the player: epaint's `TexturesDelta`
/// debug-asserts on dropping unapplied deltas (M6 final review app I2 — every debug replay test
/// tripped it), and a player whose frames are never painted (a test, a replay without
/// `--screenshot`, an early error) has nobody else to apply them. A screenshot takes them first.
impl<H: Host> Drop for Player<'_, H> {
    fn drop(&mut self) {
        self.textures.clear();
    }
}

impl<'h, H: Host> Player<'h, H> {
    pub fn new(host: &'h mut H, screen: (u32, u32)) -> Self {
        Player { host, screen, fixed: false, textures: Default::default(), base: None, since: Since::default() }
    }

    /// One worker event (false if none came within `timeout`); a snapshot is noted for the
    /// `mark`-relative expectations.
    fn pump(&mut self, v: &mut Viewer, timeout: Duration) -> bool {
        match self.host.pump(v, timeout) {
            None => false,
            Some(snapshot) => {
                if let Some(s) = v.snapshot.as_ref().filter(|_| snapshot) {
                    self.since.note(s);
                }
                true
            }
        }
    }

    /// One chrome frame at the current screen size.
    pub fn frame(&mut self, v: &mut Viewer) {
        let (w, h) = self.screen;
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))),
            ..Default::default()
        };
        let mut full = v.run_ui(raw);
        self.textures.append(std::mem::take(&mut full.textures_delta));
    }

    /// Every event already waiting.
    fn drain(&mut self, v: &mut Viewer) {
        while self.pump(v, Duration::ZERO) {}
    }

    /// Runs frames and takes the worker's events until `done` says so (`Err` ends the wait).
    fn wait(&mut self, v: &mut Viewer, mut done: impl FnMut(&Viewer) -> Result<bool, String>) -> Result<(), String> {
        let t0 = Instant::now();
        let mut last_frame = Instant::now();
        loop {
            self.drain(v);
            if last_frame.elapsed() >= FRAME_EVERY {
                self.frame(v);
                last_frame = Instant::now();
            }
            if done(v)? {
                return Ok(());
            }
            if t0.elapsed() > WAIT_LIMIT {
                return Err(format!("timed out after {WAIT_LIMIT:?} ({})", state(v)));
            }
            if !self.pump(v, FRAME_EVERY) && v.sim.is_none() {
                std::thread::sleep(FRAME_EVERY);
            }
        }
    }

    /// `wait idle`.
    pub fn wait_idle(&mut self, v: &mut Viewer) -> Result<(), String> {
        self.wait(v, |v| Ok(idle(v)))
    }

    /// Until `counter` (the snapshot's step or simulated time) advanced by `by` from its value in
    /// the first snapshot that has seen every command sent (so a preset, rewind or solver switch
    /// just before does not leave the base on the old run's counter); fails at once when the run
    /// is paused and idle short of it.
    fn wait_advance(&mut self, v: &mut Viewer, by: f64, counter: fn(&Snapshot) -> f64) -> Result<(), String> {
        self.wait(v, |v| Ok(synced(v) && v.settled()))?;
        let base = v.snapshot.as_ref().map_or(0.0, counter);
        self.wait(v, |v| {
            let Some(s) = v.snapshot.as_ref() else { return Ok(false) };
            if counter(s) >= base + by - 1e-9 {
                return Ok(true);
            }
            if synced(v) && s.dynamics.paused && !s.working {
                return Err(format!("the run is paused at step {} (t = {:.2} fs)", s.dynamics.step, s.dynamics.sim_time_fs()));
            }
            Ok(false)
        })
    }

    /// Runs a script's text; returns what it saw after each wait, with the wait's line number.
    #[cfg_attr(not(test), allow(dead_code))] // the tests' entry; `run_with` plays parsed lines
    pub fn play(&mut self, v: &mut Viewer, text: &str) -> Result<Vec<(usize, Observation)>, String> {
        let lines = parse_numbered(text)?;
        self.play_lines(v, &lines)
    }

    pub fn play_lines(&mut self, v: &mut Viewer, lines: &[(usize, Line)]) -> Result<Vec<(usize, Observation)>, String> {
        let mut seen = Vec::new();
        self.frame(v);
        self.since = Since::mark(v);
        for (n, line) in lines {
            let fail = |e: String| format!("line {n}: {}: {e}", write_line(line));
            self.drain(v);
            match line {
                Line::Scene(path) => {
                    let p = std::path::Path::new(path);
                    let full = match &self.base {
                        Some(b) if p.is_relative() => b.join(p),
                        _ => p.to_path_buf(),
                    };
                    let text = std::fs::read_to_string(&full).map_err(|e| fail(format!("{}: {e}", full.display())))?;
                    let scene = crate::scene::Scene::parse(&text).map_err(|e| fail(format!("{}: {e}", full.display())))?;
                    scene.validate().map_err(|e| fail(format!("{}: {e}", full.display())))?;
                    v.ui.scene_path = full.display().to_string();
                    // the file only: a model sidecar next to it is never read (module doc, "Data")
                    v.apply(Action::LoadScene { sidecar: false });
                }
                Line::NotReplayable(kind) => {
                    if harmless(kind) {
                        eprintln!("replay: line {n}: skipping a recorded {kind} (not replayable; it does not change the run)");
                    } else {
                        return Err(fail(format!(
                            "a recorded {kind} cannot be replayed (its settings live in the panel, not in a script), and the replay would diverge from the session: remove the line to replay without it, or replace it with script lines"
                        )));
                    }
                }
                Line::Mark => self.since = Since::mark(v),
                Line::WaitSource(src) => {
                    let src = src.clone();
                    self.wait(v, |v| Ok(v.snapshot.as_ref().is_some_and(|s| source_name(s.dynamics.source) == src))).map_err(fail)?;
                }
                Line::View(w, h) => {
                    if !self.fixed {
                        self.screen = (*w, *h);
                    }
                }
                Line::At(_, e) => {
                    if let Event::Resize { w, h } = e {
                        if !self.fixed && *w > 0 && *h > 0 {
                            self.screen = (*w, *h);
                        }
                    }
                    v.input(e, false);
                }
                Line::Action(a) => v.apply(a.clone()),
                Line::WaitIdle => self.wait_idle(v).map_err(fail)?,
                Line::WaitMs(ms) => {
                    let end = Instant::now() + Duration::from_millis(*ms);
                    self.wait(v, |_| Ok(Instant::now() >= end)).map_err(fail)?;
                }
                Line::WaitSteps(k) => self.wait_advance(v, *k as f64, |s| s.dynamics.step as f64).map_err(fail)?,
                Line::WaitFs(x) => self.wait_advance(v, *x, |s| s.dynamics.sim_time_fs()).map_err(fail)?,
                Line::Expect(e) => check(v, e, &self.since).map_err(fail)?,
            }
            if matches!(line, Line::WaitIdle | Line::WaitMs(_) | Line::WaitSteps(_) | Line::WaitFs(_) | Line::WaitSource(_)) {
                if let Some(s) = &v.snapshot {
                    seen.push((*n, observation(s)));
                }
            } else {
                self.frame(v);
            }
        }
        Ok(seen)
    }
}

/// The viewer's state for a timeout message.
fn state(v: &Viewer) -> String {
    match &v.snapshot {
        None => "no snapshot yet".into(),
        Some(s) => format!(
            "busy {}, working {}, commands {}/{}, step {}, paused {}, relaxing {}, bond references pending {:?}, settled {}",
            v.busy,
            s.working,
            s.commands,
            v.commands_sent(),
            s.dynamics.step,
            s.dynamics.paused,
            crate::ui::relaxing(&s.dynamics),
            s.dynamics.bonds_pending,
            v.settled()
        ),
    }
}

/// `--replay` without the extras: [`run_with`] with no start-up flags (`replays_pass`).
#[cfg_attr(not(test), allow(dead_code))] // main passes its flags through `run_with`
pub fn run(path: &Path, preset: &str, backend: Backend, screenshot: Option<&str>) -> Result<(), String> {
    let o = Options { preset: preset.into(), backend, screenshot: screenshot.map(str::to_string), size: None };
    run_with(path, &o, &|_| Ok(()))
}

/// What `--replay` takes besides the script.
pub struct Options {
    pub preset: String,
    pub backend: Backend,
    /// Write the last frame here (`--screenshot`), passed or failed.
    pub screenshot: Option<String>,
    /// Lay the chrome out at this size instead of the script's `view` (`--size`).
    pub size: Option<(u32, u32)>,
}

/// `--replay <path>`: solves the preset headlessly (the screenshot harness's frame and first-solve
/// wait), runs the script, and returns the first failed line as `Err`. `setup` applies the
/// start-up flags to the viewer before its worker starts.
pub fn run_with(path: &Path, o: &Options, setup: &dyn Fn(&mut Viewer) -> Result<(), String>) -> Result<(), String> {
    let name = path.display().to_string();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{name}: {e}"))?;
    let lines = parse_numbered(&text).map_err(|e| format!("{name}: {e}"))?;
    let view = lines.iter().find_map(|(_, l)| match l {
        Line::View(w, h) => Some((*w, *h)),
        _ => None,
    });
    let screen = o.size.or(view).unwrap_or(DEFAULT_VIEW);
    let mut v = Viewer::new(&o.preset).ok_or_else(|| format!("unknown preset {}", o.preset))?;
    setup(&mut v)?;
    let mut shot = Shot::new(screen.0, screen.1).ok_or("no GPU adapter")?;
    v.start(o.backend, Some(shot.hl.ctx.clone()));
    let t0 = Instant::now();
    crate::screenshot::wait_for_first_solve(&mut v, &shot.hl, &mut shot.painter)?;
    let (result, screen, mut textures) = {
        let mut host = ShotHost { shot: &mut shot };
        let mut p = Player::new(&mut host, screen);
        p.fixed = o.size.is_some();
        p.base = path.parent().map(|d| d.to_path_buf());
        let r = p.play_lines(&mut v, &lines);
        (r, p.screen, std::mem::take(&mut p.textures))
    };
    let _ = std::fs::remove_file(&v.autosave);
    let shot_result = match &o.screenshot {
        Some(png) => {
            shot.resize(screen.0, screen.1);
            shot.render(&mut v, textures);
            shot.write(png)
        }
        // no painter: free the unpainted frames' texture updates (epaint debug-asserts on
        // dropping unapplied ones), before the `?` below can drop them on an error
        None => {
            textures.clear();
            Ok(())
        }
    };
    let seen = result.map_err(|e| format!("{name}: {e}"))?;
    eprintln!("replay {name}: {} lines, {} waits, passed in {:.1} s", lines.len(), seen.len(), t0.elapsed().as_secs_f64());
    shot_result
}

#[cfg(test)]
mod tests;
