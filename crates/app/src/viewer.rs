//! Window-independent viewer state: the worker, the local scene mirror, the latest snapshot, the
//! camera, and the chrome (`ui`). Shared by the interactive window (`main`), the headless
//! screenshot and the input replayer (`crate::replay`): the 3D view's mouse and keys enter through
//! [`Viewer::input`] and the chrome's actions through [`Viewer::apply`] on every one of them — one
//! code path, so a replay drives exactly what a hand does.
use crate::gfx::Painter;
use crate::replay;
use crate::scene::{Scene, SceneNucleus};
use crate::sim::{Backend, Command, Event as SimEvent, SimWorker, Snapshot};
use crate::tools::{self, ToolState};
use crate::ui::{self, UiState};
use glam::Vec3;
use gpu::GpuContext;
use render::camera::{OrbitCamera, Viewport};
use sim::{DynCommand, Source};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The app's zoom floor: the camera's own 0.5 Bohr floor puts the eye inside a Ne-sized sphere.
pub const MIN_DISTANCE: f32 = 2.0;

/// While playing, one stderr line per this interval (the trajectory's vital signs).
const PLAY_LOG_EVERY: Duration = Duration::from_secs(1);

/// The window of the measured fs/s readout (`Viewer::measured_fs_per_s`): a tick's own figure
/// swings between the surrogate's thousands and a DFT step's few.
const RATE_WINDOW: Duration = Duration::from_secs(2);

/// How much of the last DFT density shows while the surrogate answers (it has no electrons):
/// the cloud is drawn at this fraction and labelled with the time it was solved at.
pub const SURROGATE_DENSITY_FADE: f32 = 0.3;

/// The atomic superposition is rebuilt at most this often (≈ 30 Hz: a 49³ build is ~1–3 ms on
/// the render thread).
const SUPERPOSITION_EVERY: Duration = Duration::from_millis(33);

/// How long a retiring estimated worker is waited for — on a reaper thread, never the render
/// thread — before the next estimated worker is taught without its autosave (a GPU solve stops
/// within one SCF step; a CPU solve in flight, or the cold first evaluation, cannot be
/// interrupted).
const RETIRE_WAIT: Duration = Duration::from_secs(60);

/// Who solved the real density on screen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DensitySource {
    /// A step's own evaluation (DFT, or a surrogate's fallback).
    #[default]
    Step,
    /// The surrogate's background audit (`sim::SideKind::Audit`).
    Audit,
    /// A display refresh (`sim::SideKind::Refresh`).
    Refresh,
}

impl DensitySource {
    fn what(self) -> &'static str {
        match self {
            DensitySource::Step => "last DFT step",
            DensitySource::Audit => "audit",
            DensitySource::Refresh => "refresh",
        }
    }
}

/// The density on screen: which DFT evaluation it came from, whether it is shown faded because
/// the latest step was the surrogate's (spec §5: no electrons on a surrogate step), and whether
/// the atomic superposition is shown over it (`crate::cloud`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DensityShown {
    /// Simulated time (fs) of the geometry the real density was solved at, and the step of the
    /// snapshot that carried it.
    pub from: Option<(f64, u64)>,
    /// The latest evaluation was the surrogate's: the density is the last DFT one, faded.
    pub faded: bool,
    pub source: DensitySource,
    /// The atomic superposition is shown, at this weight against the real density (1 = the
    /// superposition alone; `cloud::staleness`).
    pub superposition: Option<f32>,
}

impl DensityShown {
    /// Follows a snapshot: a density marks a DFT evaluation; none on a `Surrogate` step fades
    /// the last one; none otherwise (a failed first evaluation) leaves it as it is.
    pub fn follow(&mut self, s: &Snapshot) {
        let d = &s.dynamics;
        match (&d.display, d.source) {
            (Some(_), _) => {
                *self = DensityShown {
                    from: Some((d.sim_time_fs(), d.step)),
                    faded: false,
                    source: DensitySource::Step,
                    superposition: None,
                }
            }
            (None, Source::Surrogate) => self.faded = self.from.is_some(),
            (None, _) => {}
        }
    }

    /// A side density arrived (`DynSnapshot::side_display`: an audit's or a refresh's), with the
    /// snapshot at `step`: real, but at an earlier geometry than the current one — shown faded.
    pub fn side(&mut self, sd: &sim::SideDensity, step: u64) {
        *self = DensityShown {
            from: Some((sd.t_fs(), step)),
            faded: true,
            source: match sd.kind {
                sim::SideKind::Audit => DensitySource::Audit,
                sim::SideKind::Refresh => DensitySource::Refresh,
            },
            superposition: None,
        };
    }

    /// The volume pass's fade factor (the superposition is drawn at full strength: its label
    /// says what it is).
    pub fn fade(&self) -> f32 {
        if self.faded && self.superposition.is_none() {
            SURROGATE_DENSITY_FADE
        } else {
            1.0
        }
    }

    /// The label over the cloud (`None` for a current DFT density). `step` is the current step:
    /// a smaller one means time restarted since the density was solved (a replaced scene).
    pub fn label(&self, step: u64) -> Option<String> {
        if let Some(w) = self.superposition {
            let base = "atomic superposition — not solved (isolated atoms at the current nuclei: where the atoms are, not their bonds)";
            return Some(match self.from {
                Some((t, from_step)) if w < 1.0 && step >= from_step => {
                    format!("{base}, blended {:.0} % with the DFT density from the {} at t = {t:.2} fs", 100.0 * (1.0 - w), self.source.what())
                }
                _ => base.into(),
            });
        }
        let (t, from_step) = self.from.filter(|_| self.faded)?;
        Some(match self.source {
            _ if step < from_step => {
                format!("density from the {}, before the scene was replaced (the surrogate has no electrons)", self.source.what())
            }
            DensitySource::Step => format!("density from the last DFT step (t = {t:.2} fs) — the surrogate has no electrons"),
            DensitySource::Audit | DensitySource::Refresh => {
                format!("DFT density from the {} at t = {t:.2} fs — the surrogate has no electrons", self.source.what())
            }
        })
    }
}

/// What the orbital view shows when there is nothing to draw (spec §7), as the text over the view.
pub const ORBITALS_NEED_DFT: &str = "needs a DFT step";
pub const ORBITALS_NO_ELECTRONS: &str = "no electrons — nothing to show";
/// Estimated mode with the CPU teacher has no background solver, hence no orbital producer (spec
/// §7: a fallback DFT step never brings orbitals either): say so plainly instead of "needs a DFT
/// step", which would wait for ever.
pub const ORBITALS_CPU_TEACHER: &str = "the estimated solver on the CPU has no background solver, so it brings no orbitals: switch the solver to GPU, or to DFT";

/// The orbitals on screen (orbital-viewer spec §7): mirrors [`DensityShown`]. A payload from a
/// background solve of the estimated solver (`DynSnapshot::orbitals_side`) is real but not the
/// current step's — the estimated solver has no electrons — so it is drawn faded and labelled with
/// the time it was solved at.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrbitalShown {
    /// Simulated time (fs) of the geometry the payload was solved at.
    pub from: Option<f64>,
    /// The payload is a background solve's: shown faded.
    pub faded: bool,
}

impl OrbitalShown {
    /// Follows a snapshot: its newest payload (none: the view is off, or nothing yet).
    pub fn follow(&mut self, d: &sim::DynSnapshot) {
        *self = match &d.orbitals {
            Some(_) => OrbitalShown { from: Some(units::au_time_to_fs(d.orbitals_t)), faded: d.orbitals_side.is_some() },
            None => OrbitalShown::default(),
        };
    }

    /// The volume pass's fade factor: [`SURROGATE_DENSITY_FADE`] for a background solve's orbitals.
    pub fn fade(&self) -> f32 {
        if self.faded { SURROGATE_DENSITY_FADE } else { 1.0 }
    }

    /// The text over the view in an orbital mode (`None`: nothing to say — the current step's own
    /// orbitals are drawn, or the mode is Density). `s` is the latest snapshot.
    pub fn label(&self, ui: &UiState, s: Option<&Snapshot>) -> Option<String> {
        if ui.show == ui::ShowMode::Density {
            return None;
        }
        let Some(s) = s else { return Some(ORBITALS_NEED_DFT.into()) };
        let d = &s.dynamics;
        if d.scene.nuclei.is_empty() {
            return Some(ORBITALS_NO_ELECTRONS.into());
        }
        if s.backend == Backend::Estimated && s.model.as_ref().is_some_and(|m| !m.audits_background) {
            return Some(ORBITALS_CPU_TEACHER.into());
        }
        let Some(o) = d.orbitals.as_ref() else { return Some(ORBITALS_NEED_DFT.into()) };
        if o.spin.n_up + o.spin.n_dn == 0 && o.rows.is_empty() {
            return Some(ORBITALS_NO_ELECTRONS.into());
        }
        if o.field.is_none() && ui.show != ui::ShowMode::Orbitals {
            // refused (the GPU memory guard) or nothing to draw: the payload says why
            return o.notes.first().cloned();
        }
        let t = self.from?;
        if self.faded {
            return Some(format!("orbitals from t = {t:.2} fs (last DFT solve). The estimated solver has no electrons"));
        }
        // paused on a geometry the payload was not solved at (an edit, a failed evaluation)
        let here: Vec<[f64; 3]> = d.scene.nuclei.iter().map(|n| n.pos).collect();
        (d.paused && o.positions != here).then(|| format!("orbitals from t = {t:.2} fs (an earlier geometry)"))
    }
}

/// Smooth motion (fix round 1, user request): the nuclei of an adopted playing snapshot and when
/// it arrived.
#[derive(Clone, Debug)]
pub struct MotionSample {
    pub nuclei: Vec<SceneNucleus>,
    pub at: Instant,
    pub step: u64,
}

/// Longer than this between two snapshots (paused, a long DFT step) and the frame shows the
/// latest one as is.
pub const MOTION_MAX_GAP: Duration = Duration::from_millis(500);

/// The nuclei a frame at `now` draws between two playing snapshots: lerp(prev, cur, α) with
/// α = (now − t_cur)/(t_cur − t_prev) clamped to [0, 1] — one publish interval behind the
/// latest snapshot, so the spheres glide instead of jumping at the worker's tick rate. Snaps to
/// `cur` (no lerp) when the nuclei changed (count or Z: an addition or removal), when time did
/// not advance (a replaced scene restarts it), or when the snapshots are more than
/// [`MOTION_MAX_GAP`] apart.
pub fn interpolate(prev: &MotionSample, cur: &MotionSample, now: Instant) -> Vec<SceneNucleus> {
    let gap = cur.at.saturating_duration_since(prev.at);
    let same = prev.nuclei.len() == cur.nuclei.len() && prev.nuclei.iter().zip(&cur.nuclei).all(|(a, b)| a.z == b.z);
    if !same || cur.step <= prev.step || gap.is_zero() || gap > MOTION_MAX_GAP {
        return cur.nuclei.clone();
    }
    let alpha = (now.saturating_duration_since(cur.at).as_secs_f64() / gap.as_secs_f64()).clamp(0.0, 1.0);
    prev.nuclei
        .iter()
        .zip(&cur.nuclei)
        .map(|(a, b)| SceneNucleus {
            z: b.z,
            a: b.a,
            pos: std::array::from_fn(|k| a.pos[k] + alpha * (b.pos[k] - a.pos[k])),
            vel: b.vel,
        })
        .collect()
}

/// The frame's viewport: the rect the chrome leaves for the 3D view (egui points, `PanelOutput::view`)
/// in physical pixels, snapped to whole pixels so the render pass's viewport and scissor agree.
pub fn viewport_of(view: egui::Rect, pixels_per_point: f32) -> Viewport {
    let (min, max) = ((view.min * pixels_per_point).round(), (view.max * pixels_per_point).round());
    Viewport { origin: [min.x, min.y], size: [(max.x - min.x).max(1.0), (max.y - min.y).max(1.0)] }
}

/// The density crossfade's weight at `now` (user request 2026-10-14, "smooth the density too"):
/// the same α as [`interpolate`] — (now − t_cur)/(t_cur − t_prev) clamped to [0, 1] — between the
/// arrival times of the last two densities shown (`[previous, current]`), 1 (the current density
/// alone: a snap) without a previous one or when they are more than [`MOTION_MAX_GAP`] apart.
/// Display-only: a blend of two solved densities, not a transported one.
pub fn density_alpha(arrivals: [Option<Instant>; 2], now: Instant) -> f32 {
    let [Some(prev), Some(cur)] = arrivals else { return 1.0 };
    let gap = cur.saturating_duration_since(prev);
    if gap.is_zero() || gap > MOTION_MAX_GAP {
        return 1.0;
    }
    (now.saturating_duration_since(cur).as_secs_f64() / gap.as_secs_f64()).clamp(0.0, 1.0) as f32
}

/// A fresh autosave path per viewer (module doc of `sim`, "Autosave"): the process id and a
/// counter, in the OS temp directory.
fn autosave_path() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "atomic-sandbox-{}-{}.model",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// The centroid of the listed nuclei (the Edit tool's handles), `None` for none.
pub fn centroid(set: &[usize], nuclei: &[crate::scene::SceneNucleus]) -> Option<Vec3> {
    let ps: Vec<Vec3> = set.iter().filter_map(|&i| nuclei.get(i)).map(|n| tools::v3(n.pos)).collect();
    (!ps.is_empty()).then(|| ps.iter().copied().sum::<Vec3>() / ps.len() as f32)
}

/// The Edit tool's overlay (screen space, the guides' painter): the box being dragged, the
/// handles at the selection's centroid (X red, Y green, Z blue; the plane squares in the colour of
/// their normal; the hovered or held one yellow) and, while a handle is held, its offset in Å.
fn edit_overlay(painter: &egui::Painter, camera: &OrbitCamera, scene: &Scene, tool: &ToolState, vp: Viewport, ppp: f32) {
    use egui::{Color32, Stroke};
    let to_pos = |(x, y): (f32, f32)| egui::pos2(x / ppp, y / ppp);
    if let (Some(a), Some(b)) = (tool.boxing, tool.cursor) {
        let r = egui::Rect::from_two_pos(to_pos(a), to_pos(b));
        painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(120, 170, 255, 30));
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::from_rgb(120, 170, 255)), egui::StrokeKind::Middle);
    }
    let Some(c) = centroid(&tool.set(scene.nuclei.len()), &scene.nuclei) else { return };
    let (Some(l), Some(c_px)) = (tools::gizmo_length(camera, c, vp), tools::project(camera, c, vp)) else { return };
    let active = tool.gizmo.as_ref().map(|g| g.handle);
    let hover = if active.is_none() { tool.cursor.and_then(|(x, y)| tools::gizmo_hit(camera, vp, c, x, y)) } else { None };
    let lit = |h: tools::Handle| active == Some(h) || hover == Some(h);
    let colours = [Color32::from_rgb(235, 85, 85), Color32::from_rgb(120, 205, 90), Color32::from_rgb(90, 145, 255)];
    let yellow = Color32::from_rgb(255, 215, 90);
    for k in 0..3 {
        let pts: Option<Vec<egui::Pos2>> = tools::plane_quad(c, l, k).iter().map(|p| tools::project(camera, *p, vp).map(to_pos)).collect();
        let Some(pts) = pts else { continue };
        let h = tools::Handle::Plane(k);
        let (col, fill) = if lit(h) { (yellow, 0.45) } else { (colours[k], 0.2) };
        painter.add(egui::Shape::convex_polygon(pts, col.gamma_multiply(fill), Stroke::new(1.0, col)));
    }
    let o = to_pos(c_px);
    for k in 0..3 {
        let Some(tip) = tools::project(camera, c + tools::axis(k) * l, vp) else { continue };
        let h = tools::Handle::Axis(k);
        let (col, w) = if lit(h) { (yellow, 3.5) } else { (colours[k], 2.5) };
        let b = to_pos(tip);
        painter.line_segment([o, b], Stroke::new(w, col));
        let v = b - o;
        if v.length() > 2.0 {
            let dir = v / v.length();
            let perp = egui::vec2(-dir.y, dir.x);
            painter.add(egui::Shape::convex_polygon(vec![b + dir * 11.0, b + perp * 5.0, b - perp * 5.0], col, Stroke::NONE));
            painter.text(b + dir * 21.0, egui::Align2::CENTER_CENTER, ["X", "Y", "Z"][k], egui::FontId::proportional(12.0), col);
        }
    }
    painter.circle_filled(o, 3.0, Color32::WHITE);
    if let (Some(g), Some(at)) = (&tool.gizmo, tool.cursor) {
        let d = g.delta.map(units::bohr_to_angstrom);
        let names = ["x", "y", "z"];
        let mut text = match g.handle {
            tools::Handle::Axis(k) => format!("Δ{} {:+.3} Å", names[k], d[k]),
            tools::Handle::Plane(k) => {
                let (a, b) = (((k + 1) % 3).min((k + 2) % 3), ((k + 1) % 3).max((k + 2) % 3));
                format!("Δ{} {:+.3}  Δ{} {:+.3} Å", names[a], d[a], names[b], d[b])
            }
        };
        if g.resume {
            text += "  · paused while held";
        }
        let galley = painter.layout_no_wrap(text, egui::FontId::proportional(12.0), yellow);
        let rect = egui::Align2::LEFT_TOP.anchor_size(to_pos(at) + egui::vec2(16.0, 14.0), galley.size()).expand(3.0);
        painter.rect_filled(rect, 3.0, Color32::from_black_alpha(170));
        painter.galley(rect.min + egui::vec2(3.0, 3.0), galley, Color32::WHITE);
    }
}

/// The mouse as the 3D view sees it: the cursor (window physical pixels) and the buttons whose
/// press landed in the view (not on the chrome); only these drive the tools.
#[derive(Default)]
pub struct Mouse {
    /// Where the left button went down in Inspect (`ToolMode::Orbit`): a release near it is a click
    /// that selects the nucleus under it (a drag orbits).
    pub press_at: Option<(f32, f32)>,
    pub pos: (f32, f32),
    /// Shift is held (the Edit tool adds to its selection).
    pub shift: bool,
    pub left: bool,
    pub right: bool,
    pub middle: bool,
}

pub struct Viewer {
    /// The bond-length and angle labels' smoothing (display only; [`LabelSmoother`]).
    pub label_smooth: std::cell::RefCell<LabelSmoother>,
    /// The side of its bond each bond-length label sits on (+1/−1), kept until the other side is
    /// clearly better: a bond pointing at the molecule's centre flipped it every frame.
    pub label_side: std::cell::RefCell<std::collections::HashMap<(u8, usize, usize, usize), f32>>,
    /// `None` until [`Viewer::start`] (the window has no GPU context before `resumed`).
    pub sim: Option<SimWorker>,
    /// The backend `sim` runs.
    pub backend: Backend,
    /// The render context; the GPU backend runs on a clone of it (same device).
    gpu: Option<GpuContext>,
    /// The scene on screen. Edits apply to it at once (`Command::apply`: teleports, additions,
    /// removals, settings) so spheres move without waiting for the worker; otherwise it follows
    /// the snapshots — every snapshot that already contains all the edits sent so far replaces
    /// it, so while playing the spheres move with the dynamics. Drags never move it directly
    /// (the spring moves the nucleus; the target is drawn as a marker).
    pub scene: Scene,
    pub snapshot: Option<Snapshot>,
    pub camera: OrbitCamera,
    pub ui: UiState,
    pub tool: ToolState,
    /// The worker is playing or solving (`Event::Busy`).
    pub busy: bool,
    /// UI-side and worker-level failures (provider failures are in the snapshot); cleared by the
    /// next edit.
    pub last_error: Option<String>,
    pub egui_ctx: egui::Context,
    /// The play state as the user intends it: set by Play / Pause / Step (and the pausing edits)
    /// at once, then confirmed — or corrected: a provider failure pauses — by synced snapshots.
    pub playing: bool,
    /// The active drag's latest target (the marker), as sent.
    pub drag_target: Option<[f64; 3]>,
    /// What the density on screen is (faded on surrogate steps).
    pub density: DensityShown,
    /// What the orbitals on screen are (faded when a background solve's).
    pub orbitals: OrbitalShown,
    /// The last orbital payload adopted (`DynSnapshot::orbitals_seq`), and when the last two
    /// arrived (the crossfade, like `density_at`).
    orbital_seq: Option<u64>,
    /// The orbital view last sent, and whether a request is out whose payload has not arrived (the
    /// display eigensolve may be running: `PanelInput::orbital_waiting`).
    orbital_sent: Option<sim::OrbitalViewRequest>,
    orbital_waiting: bool,
    orbital_at: [Option<Instant>; 2],
    /// Where the 3D scene is drawn this frame (physical pixels): the rect the chrome leaves, from the
    /// last `run_ui`. The renderer and every mouse ↔ world path use it.
    pub viewport: Viewport,
    /// Where an estimated worker keeps the model across solver switches (deleted at exit).
    pub autosave: PathBuf,
    /// Estimated mode with no model starts a sampling pass by itself (`Command::EnsureModel`):
    /// the screenshot and replay harness; the window sets it false (a pass from Start sampling only).
    pub auto_pass: bool,
    /// (wall, simulated fs) of recent playing snapshots: the measured rate over [`RATE_WINDOW`].
    rate: std::collections::VecDeque<(Instant, f64)>,
    /// The last two adopted playing snapshots (smooth motion, [`interpolate`]).
    motion: [Option<MotionSample>; 2],
    /// When the last two densities shown arrived (the crossfade, [`density_alpha`]); the first
    /// is `None` after a snap (a new grid).
    density_at: [Option<Instant>; 2],
    /// The positions the last real density was solved at (the superposition's blend weight,
    /// `cloud::staleness`), the last side density adopted (`SideDensity::seq`), and when and at
    /// which positions the superposition was last built.
    real_at: Option<Vec<[f64; 3]>>,
    side_seq: Option<u64>,
    superposition_built: Option<(Instant, Vec<[f64; 3]>)>,
    /// Estimated workers shutting down on reaper threads: each sends whether it retired (wrote
    /// its autosave) within [`RETIRE_WAIT`].
    retiring: Vec<std::sync::mpsc::Receiver<bool>>,
    /// The current estimated worker's `EnsureModel` (and audit cadence), held until every
    /// retiring worker has written the autosave it is to load.
    pending_teach: Option<Vec<Command>>,
    /// Mirrored commands (`Command::is_edit`) sent to the current worker.
    edits_sent: u64,
    /// Every command sent to the current worker (`Snapshot::commands` echoes it back): the
    /// replayer's `wait idle` waits for a snapshot that has seen them all.
    commands_sent: u64,
    /// Atom trails (`crate::trails`, View ▸ Display aids ▸ Trails): the per-atom position history
    /// of the snapshots received while they are on; empty (nothing collected) while off.
    pub trails: crate::trails::Trails,
    /// The 3D view's mouse ([`Viewer::input`]).
    pub mouse: Mouse,
    /// The chrome's own actions of the last `run_ui` (`ui::PanelOutput::actions`), for `--record`.
    pub chrome_actions: Vec<ui::Action>,
    /// The window size of the last `run_ui` (physical pixels): a resize's viewport estimate
    /// ([`Viewer::input`]) keeps the margins to it.
    screen: Option<(f32, f32)>,
    last_play_log: Option<Instant>,
    /// The flash of the last sample the pass finished (`crate::ghost`).
    ghost: crate::ghost::Tracker,
}

impl Viewer {
    /// The preset's scene, no worker yet: call [`Viewer::start`].
    /// A viewer on `preset` at the default quality (`quality::DEFAULT_H`, Accurate: the preset's
    /// box at h = σ = 0.15 — 81³ for the 12 Bohr presets).
    pub fn new(preset: &str) -> Option<Viewer> {
        let file = Scene::preset(preset)?;
        let mut scene = crate::quality::preset(preset, crate::quality::DEFAULT_H)?;
        // the app's presets start under the default ionic policy (the preset file itself is all-electron)
        scene.ionic = ui::DEFAULT_IONIC;
        let mut ui = UiState::new(preset, &scene);
        ui.box_len = crate::quality::box_len(&file);
        let egui_ctx = egui::Context::default();
        ui::apply_theme(&egui_ctx);
        Some(Viewer {
            label_smooth: Default::default(),
            label_side: Default::default(),
            sim: None,
            backend: Backend::Cpu,
            gpu: None,
            ui,
            scene,
            snapshot: None,
            camera: OrbitCamera::new(Vec3::ZERO, 14.0),
            tool: ToolState::default(),
            busy: false,
            last_error: None,
            egui_ctx,
            playing: false,
            drag_target: None,
            density: DensityShown::default(),
            orbitals: OrbitalShown::default(),
            orbital_seq: None,
            orbital_sent: None,
            orbital_waiting: false,
            orbital_at: [None, None],
            viewport: Viewport::full(1400.0, 900.0),
            autosave: autosave_path(),
            auto_pass: true,
            rate: Default::default(),
            retiring: Vec::new(),
            pending_teach: None,
            motion: [None, None],
            density_at: [None, None],
            real_at: None,
            side_seq: None,
            superposition_built: None,
            edits_sent: 0,
            commands_sent: 0,
            trails: Default::default(),
            mouse: Mouse::default(),
            chrome_actions: Vec::new(),
            screen: None,
            last_play_log: None,
            ghost: Default::default(),
        })
    }

    /// (Re)starts the worker on `backend` with the current scene (velocities included), paused,
    /// and re-sends the panel's run settings (dt, speed, thermostat, field, COM correction); pins,
    /// the time and the drift history start afresh. `Gpu` without a context falls back to `Cpu`
    /// (`Estimated` then teaches with the CPU). A previous worker is detached, not joined: a GPU
    /// solve stops at its next SCF step, a CPU solve in flight finishes in the background and is
    /// discarded. An estimated one is retired on a reaper thread (fix round 1, I2: `start` never
    /// waits): it writes its model to the autosave path, if it owns it (`sim` module doc). A new
    /// estimated worker is sent [`Command::EnsureModel`] — the autosaved model if there is one,
    /// else a sampling pass with the panel's plan — once every retiring worker has finished
    /// ([`Viewer::poll_retiring`], called from the event and frame paths). The displayed density
    /// stays until the new worker publishes. With a device, the providers' dimer gate (M7 Task 11: pseudo-H and
    /// pseudo-He are generated with a 3-D dimer check, ≈ 1 min on the GPU) runs on it — whichever solver backend is
    /// chosen; without one it is the CPU fallback (≈ 40 min per element, said by `ui::pseudo::IONIC_TIP`).
    pub fn start(&mut self, backend: Backend, gpu: Option<GpuContext>) {
        self.begin_loading(self.scene.grid_n);
        if let Some(ctx) = &gpu {
            providers::pseudo::set_dimer_backend(providers::pseudo::DimerBackend::Gpu(ctx.clone()));
            // the GPU memory budget follows the card when it reports its memory (`quality::vram_budget`)
            crate::quality::set_adapter_memory(ctx.device_local_memory_bytes());
        }
        self.pending_teach = None; // a worker that never got its EnsureModel owns nothing
        if let Some(old) = self.sim.take() {
            if self.backend == Backend::Estimated {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::Builder::new()
                    .name("sim-retire".into())
                    .spawn(move || {
                        let _ = tx.send(old.retire(RETIRE_WAIT));
                    })
                    .expect("spawn the retiring thread");
                self.retiring.push(rx);
            } else {
                old.detach();
            }
        }
        let backend = if gpu.is_none() && backend == Backend::Gpu { Backend::Cpu } else { backend };
        let estimated = backend == Backend::Estimated;
        self.reset_orbital_session();
        self.trails.clear(); // a new run: its clock restarts, and a trail never crosses a solver switch
        // the new worker has built no training views and was told nothing (activity Task 5 review
        // I2: the old worker's model is not the new one's); its first snapshot starts a new run
        self.ui.activity.new_worker();
        self.ui.activity.views_sent = false;
        self.ui.activity.views = None;
        self.ui.activity.views_at = None;
        self.sim = Some(SimWorker::spawn_with(
            self.scene.clone(),
            backend,
            gpu.clone(),
            self.ui.run_settings(),
            estimated.then(|| self.autosave.clone()),
        ));
        self.gpu = gpu;
        self.backend = backend;
        self.ui.backend = backend;
        self.ui.gpu_available = self.gpu.is_some();
        self.busy = true; // the worker evaluates the scene at once
        self.edits_sent = 0;
        self.commands_sent = 0;
        self.playing = false;
        self.drag_target = None;
        self.tool.dragging = false;
        self.side_seq = None; // the new driver numbers its side densities afresh
        if estimated {
            let mut teach = vec![
                Command::EnsureModel {
                    autoload: None, // decided when it is sent: the autosave may still be written
                    plan: self.ui.plan.clone(),
                    auto_pass: self.auto_pass,
                },
                self.ui.cloud_command(),
            ];
            if self.ui.audit_every != providers::FallbackPolicy::default().audit_every {
                teach.push(Command::SetAudit(self.ui.audit_every));
            }
            if self.ui.draft {
                teach.push(Command::SetDraft(true));
            }
            self.pending_teach = Some(teach);
            self.poll_retiring();
        }
    }

    /// A new worker's orbital session (a solver switch, the start): its provider numbers its
    /// orbital tracks afresh, from 0 in energy order, so a pick or an include set of the old
    /// worker's tracks would silently name other orbitals (final review I5) — the selection goes
    /// back to the HOMO placeholder (adopted again from the first payload) and every occupied
    /// orbital. The view it starts with (`run_settings`, sent after this) asks for no unoccupied
    /// orbitals until its first payload is in (`drive_orbital_view`): set before it is built.
    fn reset_orbital_session(&mut self) {
        self.orbital_seq = None;
        self.orbital_waiting = false;
        self.orbital_sent = None;
        self.ui.orbital_pick = None;
        self.ui.orbital_pick_adopted = false;
        self.ui.included = None;
        self.ui.unocc_deferred = self.ui.show != ui::ShowMode::Density;
    }

    /// Estimated workers still writing their autosave on a reaper thread.
    #[cfg(test)]
    pub fn retirements_pending(&self) -> usize {
        self.retiring.len()
    }

    /// No estimated worker is retiring and the current one has had its `EnsureModel` (the
    /// replayer's `wait idle`).
    pub fn settled(&self) -> bool {
        self.retiring.is_empty() && self.pending_teach.is_none()
    }

    /// Every command sent to the current worker ([`Viewer::send`]).
    pub fn commands_sent(&self) -> u64 {
        self.commands_sent
    }

    /// A worker on a provider of the test's choosing (`SimWorker::spawn_test`, labelled
    /// `Backend::Cpu`), as [`Viewer::start`] would start one: the replayer's tests run on an
    /// analytic provider, without a GPU.
    #[cfg(test)]
    pub fn start_test(&mut self, make: crate::sim::MakeProvider) {
        self.sim = Some(SimWorker::spawn_test(self.scene.clone(), make));
        self.backend = Backend::Cpu;
        self.ui.backend = Backend::Cpu;
        self.busy = true;
        self.edits_sent = 0;
        self.commands_sent = 0;
        self.playing = false;
    }

    /// Drops the retirements that finished; once none is left, sends the held `EnsureModel`
    /// (with the autoload decided now, after the last write).
    pub fn poll_retiring(&mut self) {
        let mut timed_out = false;
        self.retiring.retain(|rx| match rx.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            Ok(retired) => {
                timed_out |= !retired;
                false
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => false,
        });
        if timed_out {
            self.ui.set_status(format!(
                "the previous estimated solver did not shut down within {RETIRE_WAIT:?}: its model may not have been carried over"
            ));
        }
        if !self.retiring.is_empty() {
            return;
        }
        let Some(teach) = self.pending_teach.take() else { return };
        for mut c in teach {
            if let Command::EnsureModel { autoload, .. } = &mut c {
                *autoload = self.autosave.exists().then(|| self.autosave.clone());
            }
            self.send(c);
        }
    }

    /// Sends a command and applies it to the local mirror. Index-shifting or whole-scene
    /// commands invalidate the selection.
    pub fn send(&mut self, cmd: Command) {
        self.trails.on_command(&cmd);
        // a new scene to solve: the old picture hides until its first solve lands
        let new_n = match &cmd {
            Command::SetScene(s) => Some(s.grid_n),
            Command::SetGrid { n, .. } => Some(*n),
            Command::SetIonicPolicy(_) => Some(self.scene.grid_n),
            _ => None,
        };
        if let Some(n) = new_n {
            self.begin_loading(n);
        }
        if matches!(cmd, Command::SetScene(_) | Command::RemoveNucleus(_)) {
            self.tool = ToolState::default();
            self.drag_target = None;
        }
        cmd.apply(&mut self.scene);
        if cmd.is_edit() {
            self.edits_sent += 1;
            self.last_error = None;
        }
        let replaces = matches!(
            cmd,
            Command::SetScene(_)
                | Command::SetCharge(_)
                | Command::SetMultiplicity(_)
                | Command::SetGrid { .. }
                | Command::SetSigma(_)
                | Command::SetIonicPolicy(_)
        );
        if replaces && self.playing {
            self.ui.set_status("paused: this edit replaces the scene (time, pins and the drift history restart) — press Play to continue");
        }
        match &cmd {
            // a changed orbital view, or a paused step with the view on, brings a payload (and
            // maybe a seconds-long eigensolve): the tab says so until it arrives
            Command::OrbitalView(r) => {
                if self.orbital_sent.as_ref().is_none_or(|l| !l.same_view(r)) {
                    self.orbital_waiting = !r.what.is_off();
                }
                self.orbital_sent = Some(r.clone());
            }
            Command::Dyn(DynCommand::Step(_)) if self.ui.show != ui::ShowMode::Density => self.orbital_waiting = true,
            _ => {}
        }
        match &cmd {
            Command::Dyn(DynCommand::Play) => {
                self.playing = true;
                self.ui.status.clear(); // "paused: this edit…" is answered
            }
            c if c.pauses() => self.playing = false,
            _ => {}
        }
        match &cmd {
            Command::Dyn(DynCommand::DragStart { target, .. } | DynCommand::DragMove { target }) => {
                self.drag_target = Some(*target)
            }
            Command::Dyn(DynCommand::DragEnd) => self.drag_target = None,
            _ => {}
        }
        if let Some(sim) = &self.sim {
            self.commands_sent += 1;
            sim.send(cmd);
        }
    }

    /// One input event of the 3D view (`replay::Event`: the window's winit events converted, or a
    /// replay script's lines) — the only way mouse and keys reach the tools. `consumed`: egui took
    /// it (the pointer is over the chrome, a text field has the keys): presses, wheel steps and
    /// keys are then ignored here; motion and releases are always tracked so a drag that ends over
    /// the chrome ends. Picks, placements and drags map through the frame's viewport against the
    /// snapshot mirror `self.scene`, never the interpolated `shown_scene` (S5: display only).
    pub fn input(&mut self, e: &replay::Event, consumed: bool) {
        use replay::{Button, Event, KeyName};
        // the molecule builder's window has the mouse and the keys (Space, Delete, Esc are its own)
        let consumed = consumed || self.ui.builder.open;
        let vp = self.viewport;
        // the Edit tool's group is its own: another tool selects one nucleus
        if self.ui.mode != ui::ToolMode::Edit {
            self.tool.group.clear();
        }
        match *e {
            Event::Cursor { x, y } => {
                let p = replay::to_pixels((x, y), vp);
                let d = (p.0 - self.mouse.pos.0, p.1 - self.mouse.pos.1);
                self.mouse.pos = p;
                // distance guides follow the cursor only while it is over the 3D view
                self.tool.cursor = (!consumed && vp.contains(p.0, p.1)).then_some(p);
                self.on_motion(d);
            }
            // A button held while the cursor leaves or the window loses focus (alt-tab, a UAC
            // prompt) may never report its release: treat both as a release.
            Event::CursorLeft => {
                self.tool.cursor = None;
                self.release_all();
            }
            Event::FocusLost => {
                self.mouse.shift = false;
                self.release_all();
            }
            Event::Button { ref which, down } => {
                // a press over the chrome (egui's) or outside the 3D view does nothing here
                if down && (consumed || !vp.contains(self.mouse.pos.0, self.mouse.pos.1)) {
                    return;
                }
                match which {
                    Button::Left => {
                        self.mouse.left = down;
                        if down {
                            self.on_left_press();
                        } else {
                            self.inspect_click();
                            self.end_drag();
                            self.end_edit();
                        }
                    }
                    Button::Right => self.mouse.right = down,
                    Button::Middle => self.mouse.middle = down,
                }
            }
            // zoom, or during a Select-drag the drag's pull (`Viewer::wheel`)
            Event::Wheel { lines } if !consumed => self.wheel(lines),
            Event::Key { key: KeyName::Delete, down: true } if !consumed => {
                // every selected nucleus, the highest index first (a removal shifts the later ones)
                let mut set = self.tool.set(self.scene.nuclei.len());
                set.sort_unstable();
                for i in set.into_iter().rev() {
                    self.apply(ui::Action::Delete(i));
                }
            }
            Event::Key { key: KeyName::Shift, down } => self.mouse.shift = down,
            Event::Key { key: KeyName::Escape, down: true } if !consumed => self.tool.clear_selection(),
            Event::Key { key: KeyName::Space, down: true } if !consumed => self.apply(ui::Action::TogglePlay),
            Event::Resize { w, h } => self.resize_viewport(w as f32, h as f32),
            _ => {}
        }
    }

    /// The window was resized to `w × h` physical pixels: until the next frame lays the chrome out,
    /// the viewport keeps its margins to the window's edges (the panels have fixed widths), so a
    /// cursor event before that frame maps through the new rect, not the old one. The next
    /// `run_ui` replaces the estimate with the chrome's own rect (a layout switch to compact).
    fn resize_viewport(&mut self, w: f32, h: f32) {
        let Some((sw, sh)) = self.screen.filter(|_| w > 0.0 && h > 0.0) else { return };
        let vp = self.viewport;
        let right = sw - (vp.origin[0] + vp.size[0]);
        let bottom = sh - (vp.origin[1] + vp.size[1]);
        self.viewport = Viewport {
            origin: vp.origin,
            size: [(w - vp.origin[0] - right).max(1.0), (h - vp.origin[1] - bottom).max(1.0)],
        };
        self.screen = Some((w, h));
    }

    fn on_left_press(&mut self) {
        let vp = self.viewport;
        let (px, py) = self.mouse.pos;
        match self.ui.mode {
            ui::ToolMode::Orbit => self.mouse.press_at = Some((px, py)),
            ui::ToolMode::Place => {
                if let Some(pos) = tools::place_point(&self.camera, px, py, vp) {
                    let pos = tools::clamp_into_cell(pos, &self.scene);
                    self.send(Command::AddNucleus {
                        z: self.ui.element,
                        a: self.ui.isotope,
                        pos,
                    });
                }
            }
            ui::ToolMode::Edit => self.edit_press(px, py),
            ui::ToolMode::Select => {
                let hit = tools::pick(&self.camera, px, py, vp, &self.scene.nuclei);
                self.tool.selected = hit;
                if let Some(i) = hit {
                    let plane = (tools::v3(self.scene.nuclei[i].pos), self.camera.focus_plane().1);
                    self.begin_drag(i, plane);
                }
            }
        }
    }

    /// Every button up: ends a Select-drag ([`Viewer::end_drag`]) and stops orbit / pan.
    fn release_all(&mut self) {
        self.mouse.press_at = None;
        self.mouse.left = false;
        self.mouse.right = false;
        self.mouse.middle = false;
        self.end_drag();
        self.end_edit();
    }

    /// Inspect: the left button came up — a click (press and release within a few pixels) selects
    /// the nucleus under the cursor, or clears the selection on empty space; nothing moves.
    fn inspect_click(&mut self) {
        let Some((x0, y0)) = self.mouse.press_at.take() else { return };
        let (px, py) = self.mouse.pos;
        if self.ui.mode == ui::ToolMode::Orbit && (px - x0).hypot(py - y0) < 5.0 {
            self.tool.selected = tools::pick(&self.camera, px, py, self.viewport, &self.scene.nuclei);
        }
    }

    // ---- The Edit tool (user request 2026-10-20) ----
    //
    // Select one nucleus (click), add or remove one (Shift-click) or a group (a box dragged on
    // empty space; Shift adds), clear (Esc, or a click on empty space). The handles sit at the
    // selection's centroid: an arrow moves it along its axis, a square in that plane. Holding a
    // handle pauses a running simulation and releasing it resumes Play; each motion sends one
    // `Command::MoveNuclei` with the group's grab-time positions plus the offset (a rigid move,
    // velocities kept, clamped as a whole into the cell). After a Pause the grab-time positions
    // are read once the worker's snapshot has seen it, so the group starts from where the run
    // actually stopped.

    fn edit_press(&mut self, px: f32, py: f32) {
        let vp = self.viewport;
        let set = self.tool.set(self.scene.nuclei.len());
        let handle = centroid(&set, &self.scene.nuclei).and_then(|c| tools::gizmo_hit(&self.camera, vp, c, px, py).map(|h| (h, c)));
        if let Some((h, c)) = handle {
            let Some(start) = tools::handle_point(&self.camera, vp, px, py, h, c) else { return };
            let resume = self.playing;
            if resume {
                self.send(Command::Dyn(DynCommand::Pause));
            }
            // paused already: the mirror is the driver's state; playing: read after the Pause lands
            let base = if resume { Vec::new() } else { set.iter().map(|&i| (i, self.scene.nuclei[i].pos)).collect() };
            self.tool.gizmo = Some(tools::GizmoDrag { handle: h, base, centre: c, start, resume, delta: [0.0; 3] });
        } else if let Some(i) = tools::pick(&self.camera, px, py, vp, &self.scene.nuclei) {
            if self.mouse.shift {
                let mut g = set;
                match g.iter().position(|&j| j == i) {
                    Some(k) => {
                        g.remove(k);
                    }
                    None => g.push(i),
                }
                self.tool.selected = g.last().copied();
                self.tool.group = g;
            } else {
                self.tool.group = vec![i];
                self.tool.selected = Some(i);
            }
        } else {
            self.tool.boxing = Some((px, py));
        }
    }

    /// One motion while an Edit handle is held: the group to its grab-time positions plus the
    /// handle's offset, clamped as a whole into the cell.
    fn edit_motion(&mut self) {
        let Some(handle) = self.tool.gizmo.as_ref().map(|g| g.handle) else { return };
        if self.tool.gizmo.as_ref().is_some_and(|g| g.base.is_empty()) {
            if !self.snapshot.as_ref().is_some_and(|s| s.edits == self.edits_sent) {
                return; // the Pause has not landed yet
            }
            let base: Vec<(usize, [f64; 3])> = self.tool.set(self.scene.nuclei.len()).into_iter().map(|i| (i, self.scene.nuclei[i].pos)).collect();
            match self.tool.gizmo.as_mut() {
                Some(g) if !base.is_empty() => g.base = base,
                _ => return,
            }
        }
        let (px, py) = self.mouse.pos;
        let Some(g) = self.tool.gizmo.as_ref() else { return };
        let Some(now) = tools::handle_point(&self.camera, self.viewport, px, py, handle, g.centre) else { return };
        let d = tools::clamp_delta(&g.base, tools::handle_delta(handle, g.start, now), &self.scene);
        if d == g.delta {
            return;
        }
        let moves = g.base.iter().map(|&(i, p)| (i, [p[0] + d[0], p[1] + d[1], p[2] + d[2]])).collect();
        if let Some(g) = self.tool.gizmo.as_mut() {
            g.delta = d;
        }
        self.send(Command::MoveNuclei { moves });
    }

    /// The left button is up: a held handle lets go (Play resumes if the press paused it), a box
    /// selects what it covers (Shift adds), a click on empty space clears the selection.
    fn end_edit(&mut self) {
        if let Some(g) = self.tool.gizmo.take() {
            if g.resume && !self.playing {
                self.send(Command::Dyn(DynCommand::Play));
            }
        }
        if let Some(a) = self.tool.boxing.take() {
            let b = self.mouse.pos;
            if (b.0 - a.0).hypot(b.1 - a.1) < 5.0 {
                if !self.mouse.shift {
                    self.tool.clear_selection();
                }
            } else {
                let hits = tools::box_select(&self.camera, self.viewport, &self.scene.nuclei, a, b);
                let mut g = if self.mouse.shift { self.tool.set(self.scene.nuclei.len()) } else { Vec::new() };
                for i in hits {
                    if !g.contains(&i) {
                        g.push(i);
                    }
                }
                self.tool.selected = g.last().copied();
                self.tool.group = g;
            }
        }
    }


    fn on_motion(&mut self, (dx, dy): (f32, f32)) {
        let vp = self.viewport;
        let (px, py) = self.mouse.pos;
        if self.mouse.left && self.ui.mode == ui::ToolMode::Edit {
            // a held handle moves the group; a box follows the cursor (drawn by the overlay)
            self.edit_motion();
        } else if self.mouse.left {
            match (self.tool.selected, self.tool.drag_plane) {
                (Some(_), Some(plane)) if self.tool.dragging => {
                    if let Some(pos) = tools::drag_point(&self.camera, px, py, vp, plane) {
                        self.drag_to(pos);
                    }
                }
                // Select mode on empty space orbits too; Place mode only places
                _ if self.ui.mode != ui::ToolMode::Place => self.camera.orbit(dx, dy),
                _ => {}
            }
        } else if self.mouse.middle {
            // Middle-drag orbits in every tool mode (also while Place is active)
            self.camera.orbit(dx, dy);
        } else if self.mouse.right {
            self.camera.pan(dx, dy, vp.size[1]);
        }
    }

    /// One chrome action outside a frame — a replay script's `action` line, the 3D view's keys —
    /// through the same `ui::actions::apply` the widgets' clicks go through, then acted on as
    /// `run_ui` acts on a frame's output.
    pub fn apply(&mut self, a: ui::Action) {
        let measured_fs_per_s = self.measured_fs_per_s();
        let mut out = ui::PanelOutput::default();
        let input = ui::PanelInput {
            scene: &self.scene,
            snapshot: self.snapshot.as_ref(),
            busy: self.busy,
            playing: self.playing,
            last_error: self.last_error.as_deref(),
            selected: self.tool.selected,
            measured_fs_per_s,
            camera: &self.camera,
            density_label: None,
            orbital_label: None,
            orbital_waiting: false,
            pseudo_generating: providers::pseudo::generating(),
            draft_banner: None,
            bonds_label: None,
            probe: self.sim.as_ref().map(|w| w.probe().view()),
        };
        ui::actions::apply(&mut self.ui, &input, a, &mut out);
        self.act_on(out);
    }

    /// What a frame's (or one action's) chrome output implies: the error line, the selection, the
    /// camera focus, play / pause, the commands, and a worker re-spawn when the solver changed.
    fn act_on(&mut self, out: ui::PanelOutput) {
        if let Some(e) = out.error {
            self.last_error = Some(e);
        }
        if out.deselect && !self.tool.dragging {
            self.tool.clear_selection();
        }
        if self.ui.mode != ui::ToolMode::Edit {
            self.tool.group.clear();
        }
        if out.focus_selected {
            if let Some(n) = self.tool.selected.and_then(|i| self.scene.nuclei.get(i)) {
                self.camera.focus(tools::v3(n.pos));
            }
        }
        if out.toggle_play {
            self.toggle_play();
        }
        for cmd in out.commands {
            self.send(cmd);
        }
        if self.ui.backend != self.backend {
            self.start(self.ui.backend, self.gpu.clone());
            self.ui.set_status(format!(
                "solver switched to {}: the run restarts paused at t = 0 with the panel's settings",
                self.backend.label()
            ));
        }
    }

    /// Starts a Select-drag of nucleus `index` (the button went down on it). Nothing is sent
    /// yet: the first motion decides spring or teleport (`tools::drag_commands`), so a plain
    /// select-click never tethers a moving nucleus to the cursor.
    pub fn begin_drag(&mut self, index: usize, plane: (Vec3, Vec3)) {
        self.tool.selected = Some(index);
        self.tool.dragging = true;
        self.tool.spring = false;
        self.tool.drag_plane = Some(plane);
    }

    /// One motion event of the current Select-drag: the position is clamped into the cell
    /// (review I2) and sent as a spring target or a teleport per the run's state now (I1).
    pub fn drag_to(&mut self, pos: [f64; 3]) {
        let Some(index) = self.tool.selected.filter(|_| self.tool.dragging) else {
            return;
        };
        let pos = tools::clamp_into_cell(pos, &self.scene);
        for c in tools::drag_commands(&mut self.tool.spring, self.playing, index, pos) {
            self.send(c);
        }
    }

    /// Ends the Select-drag: button release, focus loss or the cursor leaving the window
    /// (`DragEnd` if the spring is active).
    pub fn end_drag(&mut self) {
        if self.tool.dragging && self.tool.spring {
            self.send(Command::Dyn(DynCommand::DragEnd));
        }
        self.tool.dragging = false;
        self.tool.spring = false;
        self.tool.drag_plane = None;
    }

    /// Play ↔ Pause (the panel button and the Space key).
    pub fn toggle_play(&mut self) {
        if !self.playing && self.estimated_without_model() {
            self.ui.set_status(NO_MODEL_PLAY.to_string());
            return;
        }
        let c = if self.playing {
            DynCommand::Pause
        } else {
            DynCommand::Play
        };
        self.send(Command::Dyn(c));
    }

    /// A new scene is loading: its ETA from the last solve's time scaled by the grid's points (n³).
    fn begin_loading(&mut self, n: usize) {
        if self.scene.nuclei.is_empty() && n == self.scene.grid_n {
            return;
        }
        // the past completion of this very load (the user's, else the repository's), else the
        // last solve scaled by the grid's points (n³)
        let key = format!("load {} {}^3 {} {}", self.ui.source.label().replace(' ', "_"), n, self.scene.ionic.word(), self.ui.backend.label().replace(' ', "_"));
        let eta = self.ui.timings.eta(&key).unwrap_or_else(|| self.ui.last_solve.map_or(2.0, |(n0, t0)| t0 * (n as f64 / n0.max(1) as f64).powi(3)) + 0.3);
        self.ui.loading = Some((Instant::now(), eta, key));
    }

    /// The loading ends with the first solve of the new scene (the snapshot has seen every edit and
    /// its forces are fresh), or after two minutes.
    fn finish_loading(&mut self) {
        let Some((t, _, key)) = self.ui.loading.clone() else { return };
        let done = self.snapshot.as_ref().is_some_and(|s| s.edits == self.edits_sent && !s.dynamics.forces_stale);
        if done {
            self.ui.timings.record(&key, t.elapsed().as_secs_f64());
        }
        if done || t.elapsed().as_secs() > 120 {
            self.ui.loading = None;
        }
    }

    /// The floating dots are on screen (Dots mode, floating on, the cloud shown): the window keeps
    /// drawing so they drift (`main`, only while focused).
    pub fn dots_floating(&self) -> bool {
        self.ui.volume_mode == ui::VOLUME_DOTS && self.ui.dots_float && !self.ui.hide_cloud
    }

    /// The estimated solver with no fitted model (and no pass running that will make one).
    fn estimated_without_model(&self) -> bool {
        let Some(s) = self.snapshot.as_ref().filter(|s| s.backend == crate::sim::Backend::Estimated) else { return false };
        !s.model.as_ref().is_some_and(|m| m.fitted) && !crate::sim::sampling_active(&s.dynamics)
    }

    /// Estimated play stops itself (user request 2026-10-19): with no model (it was forgotten or
    /// refused), or when every one of the last 100 steps fell back to DFT (the model answers
    /// nothing here: playing on is a slow DFT run).
    fn guard_estimated(&mut self) {
        if !self.playing {
            return;
        }
        let why = if self.estimated_without_model() {
            Some("paused: the estimated solver has no model — load one (File ▸ Open) or Start sampling in the Model tab")
        } else {
            self.snapshot
                .as_ref()
                .filter(|s| s.backend == crate::sim::Backend::Estimated && s.dynamics.step >= 100 && s.dynamics.fallback_fraction_100 >= 0.999)
                .map(|_| "paused: every one of the last 100 steps fell back to DFT — the model answers nothing here (Model tab: teach around the blockers, or sample more)")
        };
        if let Some(why) = why {
            self.send(Command::Dyn(DynCommand::Pause));
            self.ui.set_status(why.to_string());
        }
    }

    /// The wheel over the 3D view (`lines` up positive; a pixel wheel's 40 px count as a line):
    /// during a Select-drag it sets the drag pull (`ui::Action::DragForceWheel`, applied with the
    /// next frame's chrome actions; the camera stays put), otherwise it zooms.
    pub fn wheel(&mut self, lines: f32) {
        if self.tool.dragging {
            self.ui.queued.push(ui::Action::DragForceWheel(lines));
        } else {
            self.zoom(lines);
        }
    }

    pub fn zoom(&mut self, lines: f32) {
        self.camera.zoom(lines);
        self.camera.distance = self.camera.distance.max(MIN_DISTANCE);
    }

    /// Drains every pending worker event (non-blocking); returns whether any arrived.
    pub fn drain_sim(
        &mut self,
        ctx: &GpuContext,
        painter: &mut Painter,
        depth: &wgpu::TextureView,
    ) -> bool {
        self.poll_retiring();
        let mut any = false;
        while let Some(ev) = self.sim.as_ref().and_then(SimWorker::try_recv) {
            self.on_event(ev, ctx, painter, depth);
            any = true;
        }
        any
    }

    pub fn on_event(
        &mut self,
        ev: SimEvent,
        ctx: &GpuContext,
        painter: &mut Painter,
        depth: &wgpu::TextureView,
    ) {
        self.on_event_with(ev, Some((ctx, painter, depth)));
    }

    /// [`Viewer::on_event`] without a renderer (the replayer's tests on an analytic provider,
    /// whose snapshots carry no density): everything but the density uploads.
    #[cfg(test)]
    pub fn on_event_bare(&mut self, ev: SimEvent) {
        self.on_event_with(ev, None);
    }

    fn on_event_with(&mut self, ev: SimEvent, gfx: Option<(&GpuContext, &mut Painter, &wgpu::TextureView)>) {
        self.poll_retiring();
        match ev {
            SimEvent::Snapshot(s) => {
                self.log(&s);
                self.ui.activity.absorb(&s);
                if let Some(w) = &self.sim {
                    self.ui.activity.absorb_display(w.probe().take_display_solves());
                }
                if s.dynamics.scene.nuclei.is_empty() && !s.dynamics.forces_stale {
                    // the empty scene (`sim::dynamics`, "The empty scene"): no electrons, so no
                    // cloud — not the last molecule's, faded or not
                    if let Some((_, painter, _)) = gfx {
                        painter.clear_density();
                    }
                    self.density = DensityShown::default();
                    self.orbitals = OrbitalShown::default();
                    self.real_at = None;
                    self.superposition_built = None;
                } else {
                    let step_density = s.dynamics.display.is_some();
                    self.density.follow(&s);
                    self.orbitals.follow(&s.dynamics);
                    let new_orbitals = self.orbital_seq != Some(s.dynamics.orbitals_seq);
                    self.orbital_seq = Some(s.dynamics.orbitals_seq);
                    if new_orbitals {
                        self.orbital_waiting = false;
                    }
                    if step_density {
                        self.real_at = Some(s.dynamics.scene.nuclei.iter().map(|n| n.pos).collect());
                    }
                    if let Some((ctx, painter, depth)) = gfx {
                        match painter.show_snapshot(ctx, &s, depth) {
                            Ok(shown) => self.density_arrived(shown),
                            Err(e) => self.last_error = Some(e),
                        }
                        self.show_side_density(&s, ctx, painter, depth, step_density);
                        self.show_superposition(&s, ctx, painter, depth, step_density);
                        if new_orbitals {
                            match &s.dynamics.orbitals {
                                Some(o) => match painter.show_orbitals(ctx, o, depth) {
                                    Ok(shown) => self.orbital_arrived(shown),
                                    Err(e) => self.last_error = Some(e),
                                },
                                None => painter.clear_orbitals(),
                            }
                        }
                    }
                    self.drive_orbital_view(&s.dynamics);
                }
                self.track_rate(&s);
                if s.edits == self.edits_sent {
                    self.scene = s.scene.clone();
                    self.playing = !s.dynamics.paused;
                    if s.dynamics.paused {
                        self.motion = [None, None];
                    } else {
                        let sample = MotionSample { nuclei: s.scene.nuclei.clone(), at: Instant::now(), step: s.dynamics.step };
                        let cur = self.motion[1].take();
                        self.motion = [cur, Some(sample)];
                    }
                }
                if self
                    .tool
                    .selected
                    .is_some_and(|i| i >= self.scene.nuclei.len())
                {
                    self.tool = ToolState::default();
                }
                if s.dynamics.drag.is_none() && !self.tool.dragging {
                    self.drag_target = None;
                }
                let ghost = s.dynamics.ghost.as_ref().filter(|_| crate::sim::sampling_active(&s.dynamics));
                self.ghost.follow(ghost, Instant::now());
                self.snapshot = Some(*s);
                self.record_trails();
            }
            SimEvent::Error(message) => {
                eprintln!("worker error: {message}");
                self.last_error = Some(message);
            }
            SimEvent::Busy(b) => self.busy = b,
            SimEvent::Retired => {}
        }
    }

    /// Atom trails: adds the latest snapshot's positions at its simulated time to the history when
    /// View ▸ Display aids ▸ Trails is on and the snapshot holds every edit sent (one taken before a
    /// teleport must not extend the trail it broke); with Trails off, forgets the history (nothing
    /// is collected, and turning it on starts from now).
    /// Motion blur (View ▸ Display aids, `UiState::motion_blur`; user request 2026-10-19): while
    /// playing, a translucent copy of each atom at the positions the last tick's steps passed
    /// through (`DynSnapshot::tick_path`), fainter the older. A tick of one step (DFT play) has no
    /// path, so this shows when many steps run per frame (the estimated solver). Display only.
    pub fn blur_ghosts(&self, shown: &Scene) -> Vec<(usize, [f64; 3], f32)> {
        let s = self.ui.motion_blur;
        if s <= 0.0 || !self.playing {
            return Vec::new();
        }
        let Some(snap) = self.snapshot.as_ref() else { return Vec::new() };
        let path = &snap.dynamics.tick_path;
        let n = shown.nuclei.len();
        if path.len() < 2 || path.iter().any(|p| p.len() != n) {
            return Vec::new();
        }
        // BLUR_FILL copies between each two samples (straight lines between them): a smooth smear
        const BLUR_FILL: usize = 4;
        let total = ((path.len() - 1) * BLUR_FILL + 1) as f32;
        let mut out = Vec::with_capacity(n * total as usize);
        for (j, w) in path.windows(2).enumerate() {
            for f in 0..BLUR_FILL {
                let t = f as f64 / BLUR_FILL as f64;
                let idx = (j * BLUR_FILL + f) as f32;
                let a = s * 0.6 * (idx + 1.0) / total;
                for i in 0..n {
                    let pos = std::array::from_fn(|c| w[0][i][c] + t * (w[1][i][c] - w[0][i][c]));
                    out.push((i, pos, a));
                }
            }
        }
        let last = path.last().expect("two or more samples");
        out.extend(last.iter().enumerate().map(|(i, p)| (i, *p, s * 0.6)));
        out
    }

    /// The Camera view (View ▸ Camera, `UiState::camera_view`): the positions at each instant of the
    /// shutter — the last `shutter` share of the steps the last tick passed through
    /// (`DynSnapshot::tick_path`) and the drawn positions — for the glow's time-average. Paused, or
    /// one step per tick, it is the drawn positions alone (sharp).
    pub fn camera_frames(&self, shown: &Scene) -> Vec<Vec<[f64; 3]>> {
        let now: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
        let n = now.len();
        let path = self.snapshot.as_ref().filter(|_| self.playing).map(|s| &s.dynamics.tick_path);
        let Some(path) = path.filter(|p| p.len() >= 2 && p.iter().all(|f| f.len() == n)) else { return vec![now] };
        let keep = ((path.len() as f32 * self.ui.shutter).ceil() as usize).clamp(1, path.len());
        // between two samples, straight lines: 3 instants each, for a smooth streak
        let tail = &path[path.len() - keep..];
        let mut frames = Vec::with_capacity(keep * 3 + 1);
        for w in tail.windows(2) {
            for f in 0..3 {
                let t = f as f64 / 3.0;
                frames.push((0..n).map(|i| std::array::from_fn(|c| w[0][i][c] + t * (w[1][i][c] - w[0][i][c]))).collect());
            }
        }
        frames.push(tail[tail.len() - 1].clone());
        frames
    }

    fn record_trails(&mut self) {
        if !self.ui.show_trails {
            self.trails.clear();
            return;
        }
        let Some(s) = self.snapshot.as_ref().filter(|s| s.edits == self.edits_sent) else { return };
        let pos: Vec<[f64; 3]> = s.scene.nuclei.iter().map(|n| n.pos).collect();
        self.trails.record(s.dynamics.sim_time_fs(), &pos, self.ui.trail_fs);
    }

    /// The scene a frame at `now` draws (smooth motion, [`interpolate`]): the mirror with its
    /// nuclei between the last two playing snapshots — or the mirror itself when smooth motion
    /// is off, paused, or the mirror holds an edit the snapshots have not confirmed yet.
    /// Model ▸ Audits ▸ auto (user request 2026-10-18): sets the audit cadence from the measured
    /// play speed ([`ui::auto_audit_every`]), only when audits are on and the new cadence differs
    /// from the current one by more than 20 % (no command flood while the speed jitters).
    pub fn auto_audit(&mut self, fs_per_s: Option<f64>) {
        if !self.ui.audit_auto || self.ui.audit_every == 0 {
            return;
        }
        let Some(n) = ui::auto_audit_every(fs_per_s, self.ui.dt_fs) else { return };
        let cur = self.ui.audit_every as f64;
        if (n as f64 - cur).abs() > 0.2 * cur {
            self.apply(ui::Action::SetAudit(n));
        }
    }

    /// Dock ▸ Selection ▸ follow (user request 2026-10-18): centres the camera on the selected
    /// nucleus as drawn this frame. No selection, or follow off: nothing.
    pub fn follow_selected(&mut self, shown: &Scene) {
        if !self.ui.follow_selected {
            return;
        }
        if let Some(n) = self.tool.selected.and_then(|i| shown.nuclei.get(i)) {
            self.camera.focus(tools::v3(n.pos));
        }
    }

    pub fn shown_scene(&self, now: Instant) -> Scene {
        match &self.motion {
            [Some(prev), Some(cur)] if self.ui.smooth_motion && self.scene.nuclei == cur.nuclei => Scene {
                nuclei: interpolate(prev, cur, now),
                ..self.scene.clone()
            },
            _ => self.scene.clone(),
        }
    }

    /// A new side density (an audit's or a refresh's, `DynSnapshot::side_display`) is shown
    /// as the real density — unless the snapshot carries the step's own, which is newer.
    fn show_side_density(&mut self, s: &Snapshot, ctx: &GpuContext, painter: &mut Painter, depth: &wgpu::TextureView, step_density: bool) {
        let Some(sd) = s.dynamics.side_display.as_ref() else { return };
        if self.side_seq == Some(sd.seq) {
            return;
        }
        self.side_seq = Some(sd.seq);
        if step_density {
            return;
        }
        match painter.show_display(ctx, &sd.display, depth) {
            Ok(shown) => {
                self.density_arrived(shown);
                self.density.side(sd, s.dynamics.step);
                self.real_at = Some(sd.positions.clone());
            }
            Err(e) => self.last_error = Some(e),
        }
    }

    /// The atomic superposition between real densities (`crate::cloud`; the estimated solver with
    /// the panel's toggle on and the worker's profiles for every species on this grid): built at
    /// the snapshot's positions — at most every [`SUPERPOSITION_EVERY`], and only when they moved
    /// — and blended with the last real density by how far the atoms moved from it. A real
    /// density at the current positions (weight 0) is shown as is.
    fn show_superposition(&mut self, s: &Snapshot, ctx: &GpuContext, painter: &mut Painter, depth: &wgpu::TextureView, step_density: bool) {
        let wanted = s.backend == Backend::Estimated && self.ui.cloud_superposition;
        if !wanted {
            if self.density.superposition.take().is_some() {
                painter.show_real(ctx, depth);
            }
            self.superposition_built = None;
            return;
        }
        if step_density {
            self.superposition_built = None;
            return;
        }
        let scene = &s.dynamics.scene;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let w = self.real_at.as_deref().map_or(1.0, |r| crate::cloud::staleness(r, &pos));
        if w == 0.0 && self.density.superposition.is_none() {
            return;
        }
        if let Some((at, built)) = &self.superposition_built {
            if *built == pos || at.elapsed() < SUPERPOSITION_EVERY {
                return;
            }
        }
        let (n, h, sigma) = (scene.cell.n, scene.cell.h, scene.sigma);
        if !s.atom_profiles.iter().all(|p| p.fits(n, h, sigma, scene.ionic)) {
            return;
        }
        let nuclei: Vec<(u8, [f64; 3])> = scene.nuclei.iter().map(|n| (n.z, n.pos)).collect();
        let origin = [-(n as f64 - 1.0) * h / 2.0; 3];
        let Some(rho) = crate::cloud::superpose(&s.atom_profiles, &nuclei, n, h, origin) else { return };
        let max = rho.iter().cloned().fold(0.0f32, f32::max);
        let display = sim::Display::Cpu { cell: scene.cell.clone(), origin, rho: std::sync::Arc::new(rho), max };
        match painter.show_superposition(ctx, &display, depth) {
            Ok(()) => {
                self.density.superposition = Some(w);
                self.superposition_built = Some((Instant::now(), pos));
            }
            Err(e) => self.last_error = Some(e),
        }
    }

    /// Draft mode's per-nucleus risk for the sphere tint (`gfx::risk_tint`): the larger of each
    /// atom's force uncertainty ÷ its bound and its envelope distance ÷ the limit, from the
    /// latest evaluation — empty (no tint) outside draft mode or when the count does not match.
    pub fn atom_risk(&self) -> Vec<f32> {
        let Some(s) = self.snapshot.as_ref().filter(|s| s.backend == Backend::Estimated) else { return Vec::new() };
        if !s.model.as_ref().is_some_and(|m| m.draft) {
            return Vec::new();
        }
        let ev = &s.dynamics.eval;
        let n = self.scene.nuclei.len();
        let unc = ev.atom_uncertainty.as_deref().filter(|u| u.len() == n);
        let env = ev.extrapolation.as_ref().map(|x| &x.atom_envelope[..]).filter(|e| e.len() == n);
        if unc.is_none() && env.is_none() {
            return Vec::new();
        }
        (0..n)
            .map(|i| unc.map_or(0.0, |u| u[i]).max(env.map_or(0.0, |e| e[i])) as f32)
            .collect()
    }

    /// Draft mode's red banner (`providers::surrogate`, "Draft mode"): while the model answers
    /// beyond what it has learned.
    pub fn draft_banner(&self) -> Option<String> {
        let s = self.snapshot.as_ref().filter(|s| s.backend == Backend::Estimated)?;
        let x = s.dynamics.eval.extrapolation.as_ref().filter(|_| s.dynamics.source == Source::Surrogate)?;
        let mut text = if x.atoms_outside > 0 {
            format!(
                "extrapolating: {} atom{} beyond what the model has learned — forces here are an educated guess",
                x.atoms_outside,
                if x.atoms_outside == 1 { "" } else { "s" }
            )
        } else {
            "extrapolating: the model would not answer here on its own — forces here are an educated guess".to_string()
        };
        if let Some(why) = &x.untrusted {
            text.push_str(&format!(" ({why})"));
        }
        Some(text)
    }

    /// Records a density the painter just bound ([`density_alpha`]'s arrival times).
    pub fn density_arrived(&mut self, shown: crate::gfx::Shown) {
        use crate::gfx::Shown;
        match shown {
            Shown::Same => {}
            Shown::Blended => self.density_at = [self.density_at[1], Some(Instant::now())],
            Shown::Snapped => self.density_at = [None, Some(Instant::now())],
        }
    }

    /// The volume pass's crossfade weight for a frame at `now` (`Painter::set_view`'s `mix`):
    /// [`density_alpha`] while the nuclei glide (smooth motion on, playing, the mirror confirmed —
    /// the conditions under which [`Viewer::shown_scene`] interpolates), else 1 (the current
    /// density alone).
    pub fn density_mix(&self, now: Instant) -> f32 {
        // the superposition's blend weight (`show_superposition`) is the mix while it is shown
        if let Some(w) = self.density.superposition {
            return w;
        }
        let gliding = match &self.motion {
            [Some(_), Some(cur)] => self.ui.smooth_motion && self.playing && self.scene.nuclei == cur.nuclei,
            _ => false,
        };
        if gliding { density_alpha(self.density_at, now) } else { 1.0 }
    }

    /// Records an orbital field the painter just bound ([`density_alpha`]'s arrival times).
    pub fn orbital_arrived(&mut self, shown: crate::gfx::Shown) {
        use crate::gfx::Shown;
        match shown {
            Shown::Same => {}
            Shown::Blended => self.orbital_at = [self.orbital_at[1], Some(Instant::now())],
            Shown::Snapped => self.orbital_at = [None, Some(Instant::now())],
        }
    }

    /// The orbital layer's crossfade weight at `now`: [`density_alpha`] of the payload arrivals
    /// while the nuclei glide (the conditions of [`Viewer::density_mix`]), else 1.
    pub fn orbital_mix(&self, now: Instant) -> f32 {
        let gliding = match &self.motion {
            [Some(_), Some(cur)] => self.ui.smooth_motion && self.playing && self.scene.nuclei == cur.nuclei,
            _ => false,
        };
        if gliding { density_alpha(self.orbital_at, now) } else { 1.0 }
    }

    /// The fade and the crossfade weight of the layer the painter draws (`Painter::drawn`).
    pub fn look(&self, drawn: crate::gfx::Drawn, now: Instant) -> (f32, f32) {
        match drawn {
            crate::gfx::Drawn::Orbital => (self.orbitals.fade(), self.orbital_mix(now)),
            _ => (self.density.fade(), self.density_mix(now)),
        }
    }

    /// Sets the painter's look for this frame (`Painter::set_view`) from the viewer's state.
    pub fn set_view(&self, painter: &mut Painter, ctx: &GpuContext) {
        let (fade, mix) = self.look(painter.drawn(&self.ui), Instant::now());
        // the ghost replaces the scene: the scene's cloud would hang in the air beside it
        let fade = if self.ghost_now().is_some() { 0.0 } else { fade };
        painter.set_view(ctx, &self.ui, fade, mix);
    }

    /// The scale the nuclei's spheres are drawn at now, from the view-mode state `Painter::set_view`
    /// reads (`gfx::nucleus_scale`): the orbital scale while an orbital mode is on and the newest
    /// payload is that mode's (the painter draws its orbital layer only then: a payload of another
    /// request, left over from before a mode switch, draws nothing and the spheres stay full size),
    /// else the `--ui` override or 1. Computed from the viewer's own state, so the arcs (in
    /// `overlay_lines`) and the labels (in `run_ui`, before `set_view`) agree on the same frame.
    fn sphere_scale(&self) -> f32 {
        let orbital_layer = self.ui.show != ui::ShowMode::Density
            && self.snapshot.as_ref().and_then(|s| s.dynamics.orbitals.as_ref()).is_some_and(|o| crate::gfx::mode_of_request(&o.what) == self.ui.show);
        crate::gfx::nucleus_scale(&self.ui, orbital_layer)
    }

    /// The drawn radius of nucleus `i`'s sphere (Bohr): its element's radius × [`Viewer::sphere_scale`].
    fn drawn_radius(&self, shown: &Scene, i: usize) -> f32 {
        shown.nuclei.get(i).map_or(0.0, |n| render::elements::radius(n.z) * self.sphere_scale())
    }

    /// Keeps the worker's orbital view in step with what the panel means (`UiState::orbital_request`)
    /// as payloads arrive, display only:
    /// - the orbital the producer resolved a placeholder or a vanished selection to (the HOMO
    ///   before anything was picked) becomes the pick, and is sent once, so the producer stops
    ///   falling back;
    /// - the unoccupied orbitals, held back while a freshly switched-on view waited for its first
    ///   payload (the eigensolve takes seconds; the list should not), are asked for now.
    fn drive_orbital_view(&mut self, d: &sim::DynSnapshot) {
        let Some(o) = d.orbitals.as_ref().filter(|_| self.ui.show != ui::ShowMode::Density) else { return };
        let mut resend = false;
        if let (ui::ShowMode::Orbitals, sim::OrbitalRequest::Orbital { spin, track, .. }) = (self.ui.show, &o.what) {
            let wanted = self.ui.orbital_pick.unwrap_or((sim::Spin::Up, ui::HOMO_TRACK));
            let exists = o.rows.iter().any(|r| (r.spin, r.track) == wanted);
            if !exists && self.ui.orbital_pick != Some((*spin, *track)) {
                // adopting the HOMO the producer showed for the placeholder — or again after its
                // tracks restarted (an isolated-atom job of the estimated solver renumbers them) —
                // is no selection of the user's; replacing a pick the user made is, and keeps its note
                self.ui.orbital_pick_adopted = self.ui.orbital_pick.is_none() || self.ui.orbital_pick_adopted;
                self.ui.orbital_pick = Some((*spin, *track));
                resend = true;
            }
        }
        if self.ui.unocc_deferred {
            self.ui.unocc_deferred = false;
            resend = true;
        }
        if resend {
            self.send(Command::OrbitalView(self.ui.orbital_request()));
        }
    }

    /// Simulated fs per wall second over the last [`RATE_WINDOW`] of playing snapshots (`None`
    /// before two of them, or paused).
    pub fn measured_fs_per_s(&self) -> Option<f64> {
        let (a, b) = (self.rate.front()?, self.rate.back()?);
        let wall = (b.0 - a.0).as_secs_f64();
        (wall > 0.2).then(|| (b.1 - a.1) / wall)
    }

    fn track_rate(&mut self, s: &Snapshot) {
        let d = &s.dynamics;
        if d.paused {
            self.rate.clear();
            return;
        }
        let now = Instant::now();
        if self.rate.back().is_some_and(|b| d.sim_time_fs() < b.1) {
            self.rate.clear(); // time restarted
        }
        self.rate.push_back((now, d.sim_time_fs()));
        while self.rate.front().is_some_and(|f| now - f.0 > RATE_WINDOW) {
            self.rate.pop_front();
        }
    }

    /// stderr: every snapshot that ends a burst of work (idle again), and while playing or
    /// sampling one line per [`PLAY_LOG_EVERY`] with the trajectory's vital signs (the estimated
    /// solver adds who answered, the fallback fraction and the model's samples and epoch) or the
    /// pass's progress.
    fn log(&mut self, s: &Snapshot) {
        let d = &s.dynamics;
        let model = s.model.as_ref().map_or(String::new(), |m| {
            format!(
                ", {:?}, fallbacks {:.0} % of 100, model {} samples epoch {} {}, strict here: {}, draft {}{}{}",
                d.source,
                100.0 * d.fallback_fraction_100,
                m.samples,
                m.epoch,
                if m.trusted { "trusted" } else { "untrusted" },
                m.readiness.as_ref().map_or("—".to_string(), |r| {
                    let verdict = if r.answers { "answers".to_string() } else { format!("refuses ({:?}: {})", r.rule.expect("a refusal names its rule"), r.text) };
                    match &r.unlearned {
                        Some(u) => format!("{verdict}; {u}"),
                        None => verdict,
                    }
                }),
                if m.draft { "on" } else { "off" },
                d.teacher_gap.map_or(String::new(), |g| format!(", teacher gap {g:+.2e} Ha")),
                // the rule of the step itself when the teacher answered it
                (d.source != sim::Source::Surrogate)
                    .then(|| d.eval.notes.first().map(|n| format!(", step's note: {n}")))
                    .flatten()
                    .unwrap_or_default()
            )
        });
        if crate::sim::sampling_active(d) {
            if self.last_play_log.is_none_or(|t| t.elapsed() >= PLAY_LOG_EVERY) {
                self.last_play_log = Some(Instant::now());
                let p = d.sampling.as_ref().expect("active");
                eprintln!(
                    "sampling ({}): {} {}/{}, last E {:+.6} Ha, skipped {}, failed {}",
                    s.engine.label(),
                    p.stage,
                    p.done,
                    p.total,
                    p.last_energy,
                    p.skipped,
                    p.failed
                );
            }
            return;
        }
        let r01 = match d.scene.nuclei.as_slice() {
            [a, b, ..] => format!(
                ", d(0,1) {:.4} Bohr",
                (0..3)
                    .map(|k| (a.pos[k] - b.pos[k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            ),
            _ => String::new(),
        };
        if !s.working {
            self.last_play_log = None;
            eprintln!(
                "snapshot ({}, idle): {} nuclei, {}³ × {} Bohr, E_KS {:+.6} Ha, t {:.2} fs, step {}, T {:.1} K, {} SCF steps, {} ms{r01}{model}{}",
                s.backend.label(),
                d.scene.nuclei.len(),
                d.scene.cell.n,
                d.scene.cell.h,
                d.energy_ks,
                d.sim_time_fs(),
                d.step,
                d.temperature,
                d.eval.scf_steps,
                d.eval.solve_ms,
                d.last_error
                    .as_ref()
                    .map_or(String::new(), |e| format!(" — error: {e}"))
            );
        } else if !d.paused
            && self
                .last_play_log
                .is_none_or(|t| t.elapsed() >= PLAY_LOG_EVERY)
        {
            self.last_play_log = Some(Instant::now());
            let drift = d
                .drift_per_ps
                .map_or("—".to_string(), |x| format!("{x:+.2e} Ha/ps"));
            let measured = self
                .measured_fs_per_s()
                .map_or(String::new(), |r| format!(" ({r:.0} fs/s over {RATE_WINDOW:?})"));
            eprintln!(
                "playing ({}): t {:.2} fs, step {}, T {:.1} K, E_total {:+.6} Ha, E_cons {:+.6} Ha, drift {drift}, {:.2} fs/s this tick{measured}, {:.3} ms/step, {} SCF steps{r01}{model}",
                s.backend.label(),
                d.sim_time_fs(),
                d.step,
                d.temperature,
                d.total,
                d.energy_conserved,
                d.wall_ratio,
                d.ms_per_step,
                d.eval.scf_steps
            );
        }
    }

    /// Faint distance lines (Bohr and Angstrom) from the nucleus being dragged, or in Place mode
    /// from the point under the cursor, to the nuclei within `tools::GUIDE_RADIUS`, so a geometry
    /// can be built to a number by hand; plus the drag spring's target (a ring tied to its
    /// nucleus by a line) and a small square on each pinned nucleus. Screen-space, drawn with
    /// egui's foreground painter.
    fn distance_overlay(
        root: &egui::Context,
        camera: &OrbitCamera,
        scene: &Scene,
        tool: &ToolState,
        mode: ui::ToolMode,
        drag_target: Option<[f64; 3]>,
        pins: &[bool],
        vp: Viewport,
        pull: Option<String>,
    ) {
        if vp.size[0] <= 1.0 || vp.size[1] <= 1.0 {
            return;
        }
        let ppp = root.pixels_per_point();
        let to_pos = |(x, y): (f32, f32)| egui::pos2(x / ppp, y / ppp);
        // clipped to the 3D view: a guide to a nucleus behind the dock stops at its edge
        let clip = egui::Rect::from_min_size(to_pos((vp.origin[0], vp.origin[1])), egui::vec2(vp.size[0] / ppp, vp.size[1] / ppp));
        fixed_layer(root, "distance-guides", clip);
        let painter = root
            .layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("distance-guides")))
            .with_clip_rect(clip);
        let line = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(90));
        // pins: a small square around each pinned nucleus
        let pin = egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 200, 255));
        for (i, _) in pins.iter().enumerate().filter(|(_, &p)| p) {
            if let Some(px) = scene
                .nuclei
                .get(i)
                .and_then(|n| tools::project(camera, tools::v3(n.pos), vp))
            {
                let r = egui::Rect::from_center_size(to_pos(px), egui::vec2(14.0, 14.0));
                painter.rect_stroke(r, 0.0, pin, egui::StrokeKind::Middle);
            }
        }
        // the drag spring: target ring and a line to its nucleus
        if let (Some(t), Some(i)) = (drag_target, tool.selected) {
            let spring = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 190, 90));
            if let Some(t_px) = tools::project(camera, tools::v3(t), vp) {
                painter.circle_stroke(to_pos(t_px), 5.0, spring);
                if let Some(n_px) = scene
                    .nuclei
                    .get(i)
                    .and_then(|n| tools::project(camera, tools::v3(n.pos), vp))
                {
                    painter.line_segment([to_pos(n_px), to_pos(t_px)], spring);
                }
            }
        }
        // the drag's pull (the wheel sets it) next to the cursor, else by the dragged nucleus
        if let Some(text) = pull {
            let at = tool.cursor.map(to_pos).or_else(|| {
                tool.selected
                    .and_then(|i| scene.nuclei.get(i))
                    .and_then(|n| tools::project(camera, tools::v3(n.pos), vp))
                    .map(to_pos)
            });
            if let Some(at) = at {
                let galley = painter.layout_no_wrap(text, egui::FontId::proportional(12.0), egui::Color32::from_rgb(255, 190, 90));
                let rect = egui::Align2::LEFT_TOP.anchor_size(at + egui::vec2(16.0, 14.0), galley.size()).expand(3.0);
                painter.rect_filled(rect, 3.0, egui::Color32::from_black_alpha(170));
                painter.galley(rect.min + egui::vec2(3.0, 3.0), galley, egui::Color32::WHITE);
            }
        }
        if mode == ui::ToolMode::Edit {
            edit_overlay(&painter, camera, scene, tool, vp, ppp);
        }
        let (origin, exclude) = match (tool.dragging, tool.selected, mode, tool.cursor) {
            (true, Some(i), _, _) if i < scene.nuclei.len() => (scene.nuclei[i].pos, Some(i)),
            (false, _, ui::ToolMode::Place, Some((px, py))) => {
                match tools::place_point(camera, px, py, vp) {
                    Some(p) => (p, None),
                    None => return,
                }
            }
            _ => return,
        };
        let guides = tools::distance_guides(origin, &scene.nuclei, exclude);
        let Some(o_px) = tools::project(camera, tools::v3(origin), vp) else {
            return;
        };
        let font = egui::FontId::proportional(12.0);
        if exclude.is_none() {
            // placement marker: a small cross where the nucleus would land
            let c = to_pos(o_px);
            for (dx, dy) in [(6.0, 0.0), (0.0, 6.0)] {
                let d = egui::vec2(dx, dy);
                painter.line_segment([c - d, c + d], line);
            }
        }
        for g in guides {
            let Some(t_px) = tools::project(camera, tools::v3(scene.nuclei[g.index].pos), vp)
            else {
                continue;
            };
            let (a, b) = (to_pos(o_px), to_pos(t_px));
            painter.line_segment([a, b], line);
            let label = format!(
                "{:.2} Bohr  {:.3} Å",
                g.distance,
                units::bohr_to_angstrom(g.distance)
            );
            let mid = a + (b - a) * 0.5;
            let galley = painter.layout_no_wrap(label, font.clone(), egui::Color32::WHITE);
            let rect = egui::Align2::CENTER_CENTER
                .anchor_size(mid, galley.size())
                .expand(3.0);
            painter.rect_filled(rect, 3.0, egui::Color32::from_black_alpha(160));
            painter.galley(rect.min + egui::vec2(3.0, 3.0), galley, egui::Color32::WHITE);
        }
    }

    /// The 3D view's lines for a frame drawing `shown`: the reference geometry (`overlay::build`),
    /// the atom trails with View ▸ Display aids ▸ Trails on and, with View ▸ Bonds on, the latest bond graph between the nuclei as drawn now (faded
    /// when it is an earlier DFT step's: spec §3.4; nothing when the counts differ).
    pub fn overlay_lines(&self, shown: &Scene) -> Vec<render::lines::LineVertex> {
        let mut lines = crate::overlay::build(&self.ui, &self.scene, &self.camera, self.ui.mode);
        // while a sampling pass runs the ghost of its job replaces the scene: the cell and the
        // walls stay, the scene's bonds do not
        if let Some(g) = self.ghost_now() {
            lines.extend(crate::ghost::build(g, &crate::ghost::CamBasis::of(&self.camera), self.ghost.flash(Instant::now()), &self.ghost.history()));
            return lines;
        }
        if self.ui.show_trails {
            lines.extend(self.trails.lines(&shown.nuclei, self.ui.trail_fs, self.ui.trail_opacity));
        }
        if let Some(d) = self.snapshot.as_ref().map(|s| &s.dynamics).filter(|_| self.ui.show_bonds && self.ui.loading.is_none()) {
            if let Some(g) = &d.bonds {
                let pos: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
                lines.extend(crate::overlay::bond_lines(&pos, g, d.bonds_stale));
            }
        }
        // M7: pairs whose frozen cores overlap (the latest solve's; drawn between the nuclei as shown now,
        // only while the counts agree) — always, not a bond display option: the approximation is failing there
        if let Some(d) = self.snapshot.as_ref().map(|s| &s.dynamics).filter(|d| !d.eval.core_overlaps.is_empty() && d.scene.nuclei.len() == shown.nuclei.len()) {
            let pos: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
            lines.extend(crate::overlay::core_overlap_lines(&pos, &d.eval.core_overlaps));
        }
        if let Some((marks, stale)) = self.angle_marks(shown) {
            let pos: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
            for m in &marks {
                lines.extend(crate::overlay::angle_arc_lines(&pos, m, self.drawn_radius(shown, m.vertex), stale));
            }
        }
        lines
    }

    /// What the running sampling pass is solving, to be drawn instead of the scene (View ▸ Display
    /// aids ▸ show what training is solving; `crate::ghost`): `None` when the toggle is off, no pass
    /// runs (it ended, was cancelled or failed) or the snapshot carries none.
    pub fn ghost_now(&self) -> Option<&sim::PassGhost> {
        let s = self.snapshot.as_ref().filter(|s| self.ui.show_ghost && crate::sim::sampling_active(&s.dynamics))?;
        s.dynamics.ghost.as_ref()
    }

    /// Where the shown field was solved (the display warp, `render::warp`; View ▸ fields follow the
    /// atoms): the density's geometry (`real_at`) — not while the atomic superposition is blended
    /// in, which is built at the current positions — or the orbital payload's; `None` with the
    /// toggle off or while the training ghost replaces the scene.
    pub fn warp_source(&self) -> Option<Vec<[f64; 3]>> {
        if !self.ui.warp_fields || self.ghost_now().is_some() {
            return None;
        }
        match self.ui.show {
            ui::ShowMode::Density => self.real_at.clone().filter(|_| self.density.superposition.is_none()),
            _ => self.snapshot.as_ref()?.dynamics.orbitals.as_ref().map(|o| o.positions.clone()),
        }
    }

    /// The nuclei the sphere pass draws: the scene's, or none while the ghost replaces them.
    pub fn drawn_nuclei<'a>(&self, shown: &'a Scene) -> &'a [crate::scene::SceneNucleus] {
        if self.ghost_now().is_some() { &[] } else { &shown.nuclei }
    }

    /// The nucleus under the cursor with the Select tool (the badge's and the angles' hover).
    fn hovered_nucleus(&self, shown: &Scene) -> Option<usize> {
        match (self.ui.mode, self.tool.cursor) {
            (ui::ToolMode::Select | ui::ToolMode::Orbit | ui::ToolMode::Edit, Some((x, y))) => tools::pick(&self.camera, x, y, self.viewport, &shown.nuclei),
            _ => None,
        }
    }

    /// The bond angles to draw between the nuclei as drawn now (M7 Task 13), with the stale flag of
    /// the graph (an earlier DFT step's: faded): every angle at an atom with at most
    /// `ANGLE_DECLUTTER_BONDS` bonds, those of a busier atom only while it is selected or hovered
    /// (any tool but Place, [`Self::hovered_nucleus`]). `None` when View ▸ bonds or ▸ bond angles is off, there is no graph, or it
    /// belongs to another nucleus count.
    fn angle_marks(&self, shown: &Scene) -> Option<(Vec<crate::overlay::AngleMark>, bool)> {
        if !(self.ui.show_bonds && self.ui.show_angles) {
            return None;
        }
        let d = &self.snapshot.as_ref()?.dynamics;
        let g = d.bonds.as_ref().filter(|g| g.n_nuclei == shown.nuclei.len())?;
        let pos: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
        let all = crate::overlay::bond_angles(&pos, g);
        let selected = self.tool.selected.filter(|&i| i < pos.len());
        let marks = crate::overlay::shown_angles(&all, g, selected, self.hovered_nucleus(shown)).into_iter().cloned().collect();
        Some((marks, d.bonds_stale))
    }

    /// The angle labels of this frame (M7 Task 13): the degree text, its anchor in the world and the
    /// pill's text and background alpha — those of the molecule labels, fading with the bonds when
    /// the graph is an earlier step's. Empty whenever [`Viewer::angle_marks`] is `None`.
    pub fn angle_labels(&self, shown: &Scene) -> Vec<AngleLabel> {
        let Some((marks, stale)) = self.angle_marks(shown) else { return Vec::new() };
        let (alpha, bg) = if stale { (110, 70) } else { (235, 150) };
        let pos: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
        marks
            .iter()
            .map(|m| AngleLabel {
                text: crate::overlay::angle_label(m),
                anchor: crate::overlay::angle_label_anchor(&pos, m, self.drawn_radius(shown, m.vertex)),
                alpha,
                bg,
                key: (0, m.vertex, m.a, m.b),
                value: m.radians,
                ends: None,
            })
            .collect()
    }

    /// The partial-charge labels of this frame (`sim::charges`; View ▸ Bonds and molecules ▸ partial
    /// charges): (nucleus, its charge, alpha, background alpha) for the charges that can be drawn
    /// on `shown` (`ui::shown_charges`), faded like stale bonds when they were measured at another
    /// geometry. With more than [`CHARGE_DECLUTTER`] atoms only the selected or hovered atom and its
    /// bonded neighbours.
    pub fn charge_labels(&self, shown: &Scene) -> Vec<(usize, f64, u8, u8)> {
        if !self.ui.show_charges {
            return Vec::new();
        }
        let Some(d) = self.snapshot.as_ref().map(|s| &s.dynamics) else { return Vec::new() };
        let Some((c, current)) = ui::shown_charges(d).filter(|(c, _)| c.q.len() == shown.nuclei.len()) else { return Vec::new() };
        let (alpha, bg) = if current { (235, 150) } else { (110, 70) };
        let mut keep: Vec<usize> = (0..c.q.len()).collect();
        if c.q.len() > CHARGE_DECLUTTER {
            let focus: Vec<usize> = self.tool.selected.filter(|&i| i < shown.nuclei.len()).into_iter().chain(self.hovered_nucleus(shown)).collect();
            let near = |i: usize| {
                focus.contains(&i)
                    || d.bonds.as_ref().filter(|g| g.n_nuclei == shown.nuclei.len()).is_some_and(|g| g.bonds.iter().any(|b| (b.i == i && focus.contains(&b.j)) || (b.j == i && focus.contains(&b.i))))
            };
            keep.retain(|&i| near(i));
        }
        keep.into_iter().map(|i| (i, c.q[i], alpha, bg)).collect()
    }

    /// The bond-length labels of this frame (user request 2026-10-18): each drawn bond's length in Å at
    /// its midpoint, from the nuclei as drawn now, faded with the bonds when the graph is an earlier
    /// step's. With more than [`BOND_LENGTH_DECLUTTER`] bonds only the selected or hovered atom's.
    pub fn bond_length_labels(&self, shown: &Scene) -> Vec<AngleLabel> {
        if !(self.ui.show_bonds && self.ui.show_bond_lengths) {
            return Vec::new();
        }
        let Some(d) = self.snapshot.as_ref().map(|s| &s.dynamics) else { return Vec::new() };
        let Some(g) = d.bonds.as_ref().filter(|g| g.n_nuclei == shown.nuclei.len()) else { return Vec::new() };
        let (alpha, bg) = if d.bonds_stale { (110, 70) } else { (235, 150) };
        let focus: Vec<usize> = self.tool.selected.filter(|&i| i < shown.nuclei.len()).into_iter().chain(self.hovered_nucleus(shown)).collect();
        let all = g.bonds.len() <= BOND_LENGTH_DECLUTTER;
        g.bonds
            .iter()
            .filter(|b| all || focus.contains(&b.i) || focus.contains(&b.j))
            .filter(|b| b.i < shown.nuclei.len() && b.j < shown.nuclei.len())
            .map(|b| {
                let (p, q) = (shown.nuclei[b.i].pos, shown.nuclei[b.j].pos);
                let r = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                AngleLabel { text: bond_length_text(r), anchor: std::array::from_fn(|k| 0.5 * (p[k] + q[k])), alpha, bg, key: (1, b.i, b.j, 0), value: r, ends: Some((p, q)) }
            })
            .collect()
    }

    /// The faded bonds' label for the overlays (View ▸ Bonds on, the graph an earlier DFT
    /// step's), and under it the faded charges' ("charges from t = … fs", `ui::charges_label`).
    pub fn bonds_label(&self) -> Option<String> {
        let s = self.snapshot.as_ref()?;
        let bonds = self.ui.show_bonds.then(|| ui::bonds_label(&s.dynamics)).flatten();
        // the faded charges' time (estimated mode between DFT densities), under the bonds'
        let charges = self.ui.show_charges.then(|| ui::charges_label(&s.dynamics)).flatten();
        match (bonds, charges) {
            (Some(b), Some(c)) => Some(format!("{b}\n{c}")),
            (b, c) => b.or(c),
        }
    }

    /// The composition badge (spec §7) and where it goes (physical pixels): the nucleus under the
    /// cursor with the Select tool — at the cursor —, else the selected one — beside it. Its
    /// isotope from its own mass number; the raw provider force when the snapshot's forces are
    /// this scene's and fresh.
    pub fn badge(&self, shown: &Scene) -> Option<(String, (f32, f32))> {
        if !self.ui.show_badge || self.tool.dragging {
            return None;
        }
        let vp = self.viewport;
        let hovered = match (self.hovered_nucleus(shown), self.tool.cursor) {
            (Some(i), Some(at)) => Some((i, at)),
            _ => None,
        };
        let (i, at) = match hovered {
            Some(h) => h,
            None => {
                let i = self.tool.selected.filter(|&i| i < shown.nuclei.len())?;
                let (x, y) = tools::project(&self.camera, tools::v3(shown.nuclei[i].pos), vp)?;
                (i, (x, y))
            }
        };
        let n = &shown.nuclei[i];
        let force = self
            .snapshot
            .as_ref()
            .map(|s| &s.dynamics)
            .filter(|d| !d.forces_stale && d.forces.len() == shown.nuclei.len())
            .map(|d| d.forces[i]);
        let mut text = crate::badge::badge_text(n.z, n.mass_number()?, force);
        // a pseudised species says so: its valence charge, its radius and the grid it needs (M7; the latest solve's record)
        if let Some(p) = self.snapshot.as_ref().and_then(|s| ui::pseudo::of_species(&s.dynamics.eval.pseudised, n.z)) {
            text.push_str(&format!(" · {}", ui::pseudo::badge_tail(p)));
        }
        // its partial charge (View ▸ Bonds and molecules ▸ partial charges; `sim::charges`)
        if let Some((c, current)) = self.snapshot.as_ref().filter(|_| self.ui.show_charges).and_then(|s| ui::shown_charges(&s.dynamics)) {
            if let Some(&q) = c.q.get(i) {
                text.push_str(&format!(" · q {} e{}", ui::charge_text(q), if current { "" } else { " (earlier)" }));
            }
        }
        Some((text, at))
    }

    /// The molecule labels (formula — name, spec §4.4) above each molecule of two or more nuclei,
    /// at its centroid as drawn now, faded with the bonds when the graph is an earlier step's;
    /// and the composition badge. Screen-space, egui's foreground painter, clipped to the view.
    fn label_overlay(&self, root: &egui::Context, shown: &Scene, vp: Viewport) {
        // nor over the File window (they are drawn on the foreground layer)
        if vp.size[0] <= 1.0 || vp.size[1] <= 1.0 || self.ui.files.open || self.ui.builder.open || self.ui.loading.is_some() {
            return;
        }
        let ppp = root.pixels_per_point();
        let to_pos = |(x, y): (f32, f32)| egui::pos2(x / ppp, y / ppp);
        let clip = egui::Rect::from_min_size(to_pos((vp.origin[0], vp.origin[1])), egui::vec2(vp.size[0] / ppp, vp.size[1] / ppp));
        // under every chrome element (the background order: the overlays, menus and windows draw
        // over them), clipped to the 3D view, at View ▸ Bonds and molecules ▸ label opacity
        // registered as an area so egui keeps its place in the layer order (a bare painter layer's
        // place among the background layers changes frame to frame: the labels flickered)
        fixed_layer(root, "molecule-labels", clip);
        let mut painter = root
            .layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("molecule-labels")))
            .with_clip_rect(clip);
        painter.set_opacity(self.ui.label_opacity);
        let pill = |at: egui::Pos2, anchor: egui::Align2, text: &str, colour: egui::Color32, bg: u8| {
            let galley = painter.layout_job(ui::rich(text, 12.5, colour));
            let rect = anchor.anchor_size(at, galley.size()).expand(3.0);
            painter.rect_filled(rect, 4.0, egui::Color32::from_black_alpha(bg));
            painter.galley(rect.min + egui::vec2(3.0, 3.0), galley, colour);
        };
        if let Some(g) = self.ghost_now() {
            // the ghost's label, top centre of the view under the status chip; the scene's molecule labels and badge hide
            let at = to_pos((vp.origin[0] + 0.5 * vp.size[0], vp.origin[1])) + egui::vec2(0.0, 58.0);
            // while a finished job's flash fades its outcome is named, in the solve log's word
            let last = self.ghost.flash(Instant::now()).map(|(f, _)| format!(" · last sample: {}", f.outcome.word())).unwrap_or_default();
            let galley = painter.layout_no_wrap(format!("training is solving · {}{last}", crate::ghost::stage_line(g)), egui::FontId::proportional(11.5), egui::Color32::from_white_alpha(170));
            let main = painter.layout_no_wrap(crate::ghost::label(g), egui::FontId::proportional(14.5), egui::Color32::WHITE);
            let size = egui::vec2(galley.size().x.max(main.size().x), galley.size().y + main.size().y + 2.0);
            let mut rect = egui::Align2::CENTER_TOP.anchor_size(at, size).expand(6.0);
            // below the status chip when they would overlap (a narrow view)
            if let Some(chip) = self.ui.chip_rect.filter(|c| c.intersects(rect.expand(4.0))) {
                rect = rect.translate(egui::vec2(0.0, chip.bottom() + 10.0 - rect.top()));
            }
            painter.rect_filled(rect, 5.0, egui::Color32::from_black_alpha(175));
            let cx = rect.center().x;
            painter.galley(egui::pos2(cx - galley.size().x / 2.0, rect.top() + 6.0), galley.clone(), egui::Color32::WHITE);
            painter.galley(egui::pos2(cx - main.size().x / 2.0, rect.top() + 6.0 + galley.size().y + 2.0), main, egui::Color32::WHITE);
            return;
        }
        if let Some(s) = self.snapshot.as_ref().filter(|_| self.ui.show_labels) {
            if let Some(g) = s.dynamics.bonds.as_ref().filter(|g| g.n_nuclei == shown.nuclei.len() && s.labels.len() == g.components.len()) {
                let (alpha, bg) = if s.dynamics.bonds_stale { (110, 70) } else { (235, 150) };
                // the camera's up in world space: a sphere's top edge on screen is its centre + r·up
                let up = self.camera.view().inverse().y_axis.truncate();
                let mut smooth = self.label_smooth.borrow_mut();
                smooth.begin_once(root.input(|i| i.time));
                for (name, members) in s.labels.iter().zip(&g.components).filter(|(_, m)| m.len() >= 2) {
                    // anchored on the molecule's centre of mass (user request 2026-10-18: the label
                    // must not ride the vibration — an isolated molecule's centre of mass does not
                    // vibrate at all), raised by the molecule's extent (its farthest sphere edge from
                    // the centre, averaged over the last second), and smoothed like the bond labels
                    let mass = |m: usize| {
                        let n = &shown.nuclei[m];
                        n.a.or_else(|| units::most_abundant_a(n.z)).unwrap_or(2 * n.z as u16).max(1) as f64
                    };
                    let total: f64 = members.iter().map(|&m| mass(m)).sum();
                    let com: [f64; 3] = std::array::from_fn(|k| members.iter().map(|&m| mass(m) * shown.nuclei[m].pos[k]).sum::<f64>() / total);
                    let extent = members
                        .iter()
                        .map(|&m| {
                            let n = &shown.nuclei[m];
                            (0..3).map(|k| (n.pos[k] - com[k]).powi(2)).sum::<f64>().sqrt() + render::elements::radius(n.z) as f64
                        })
                        .fold(0.0, f64::max);
                    let key = (2, members[0], members.len(), 0);
                    let (anchor, r) = smooth.update(key, com, extent);
                    let lifted = tools::v3(anchor) + up * r as f32;
                    if let Some(px) = tools::project(&self.camera, lifted, vp) {
                        let at = to_pos(px) - egui::vec2(0.0, 6.0);
                        pill(at, egui::Align2::CENTER_BOTTOM, &name.label_in(self.ui.name_style), egui::Color32::from_white_alpha(alpha), bg);
                    }
                }
                drop(smooth);
            }
        }
        // the bond lengths and angles (Task 13; user request 2026-10-18): smoothed anchors and
        // one-second averaged values, so the labels are readable while the atoms vibrate
        let mut smooth = self.label_smooth.borrow_mut();
        smooth.begin_once(root.input(|i| i.time));
        let centre = {
            let n = shown.nuclei.len().max(1) as f64;
            let c: [f64; 3] = std::array::from_fn(|k| shown.nuclei.iter().map(|x| x.pos[k]).sum::<f64>() / n);
            tools::project(&self.camera, tools::v3(c), vp)
        };
        for l in self.bond_length_labels(shown) {
            let (anchor, r) = smooth.update(l.key, l.anchor, l.value);
            if let Some(px) = tools::project(&self.camera, tools::v3(anchor), vp) {
                // beside the bond, on the side away from the molecule's centre (clear of the arcs)
                let mut at = to_pos(px);
                if let Some((p, q)) = l.ends {
                    if let (Some(a), Some(b)) = (tools::project(&self.camera, tools::v3(p), vp), tools::project(&self.camera, tools::v3(q), vp)) {
                        let mut n = egui::vec2(-(b.1 - a.1), b.0 - a.0);
                        if n.length() > 1e-3 {
                            n = n.normalized();
                            // away from the centre, with hysteresis: the side changes only when the
                            // other one is clearly better (|cos| > 0.35), else the label keeps its side
                            let mut sides = self.label_side.borrow_mut();
                            let side = sides.entry(l.key).or_insert(1.0);
                            if let Some(c) = centre {
                                let out = egui::vec2(px.0 - c.0, px.1 - c.1);
                                let cos = n.dot(out) / out.length().max(1e-3);
                                if cos > 0.35 {
                                    *side = 1.0;
                                } else if cos < -0.35 {
                                    *side = -1.0;
                                }
                            }
                            at += n * (14.0 * *side);
                        }
                    }
                }
                pill(at, egui::Align2::CENTER_CENTER, &bond_length_text(r), egui::Color32::from_rgba_unmultiplied(205, 225, 245, l.alpha), l.bg);
            }
        }
        for l in self.angle_labels(shown) {
            let (anchor, rad) = smooth.update(l.key, l.anchor, l.value);
            if let Some(px) = tools::project(&self.camera, tools::v3(anchor), vp) {
                pill(to_pos(px), egui::Align2::CENTER_CENTER, &format!("{:.1}°", rad.to_degrees()), egui::Color32::from_rgba_unmultiplied(235, 225, 190, l.alpha), l.bg);
            }
        }
        // the partial charges, just below each atom's sphere (the camera's down), the value averaged
        // over the last second like the bond lengths; the label follows the atom
        let up = self.camera.view().inverse().y_axis.truncate();
        for (i, q, alpha, bg) in self.charge_labels(shown) {
            let n = &shown.nuclei[i];
            let (_, q) = smooth.update((3, i, 0, 0), n.pos, q);
            let below = tools::v3(n.pos) - up * render::elements::radius(n.z);
            if let Some(px) = tools::project(&self.camera, below, vp) {
                pill(to_pos(px) + egui::vec2(0.0, 4.0), egui::Align2::CENTER_TOP, &ui::charge_text(q), ui::charge_colour(q, alpha), bg);
            }
        }
        drop(smooth);
        if let Some((text, at)) = self.badge(shown) {
            pill(to_pos(at) + egui::vec2(14.0, 14.0), egui::Align2::LEFT_TOP, &text, egui::Color32::from_rgb(200, 230, 255), 190);
        }
    }

    /// Runs the panel for one frame and acts on its output.
    pub fn run_ui(&mut self, raw: egui::RawInput) -> egui::FullOutput {
        let mut outs = Vec::new();
        self.poll_retiring();
        // Trails switched off forget their history; switched on while paused (no snapshot comes)
        // they start from the current one
        if !self.ui.show_trails || self.trails.is_empty() {
            self.record_trails();
        }
        // the Camera view's field of view (the diagram keeps 50°)
        self.camera.fov_y = if self.ui.camera_view { self.ui.fov_deg.clamp(10.0, 120.0).to_radians() } else { 50f32.to_radians() };
        let ctx = self.egui_ctx.clone();
        let measured_fs_per_s = self.measured_fs_per_s();
        self.auto_audit(measured_fs_per_s);
        self.guard_estimated();
        self.finish_loading();
        let shown = self.shown_scene(Instant::now());
        self.follow_selected(&shown);
        let step = self.snapshot.as_ref().map_or(0, |s| s.dynamics.step);
        let density_label = self.density.label(step);
        let orbital_label = self.orbitals.label(&self.ui, self.snapshot.as_ref());
        let orbital_waiting = self.orbital_waiting && self.ui.show != ui::ShowMode::Density;
        // the ghost replaces the scene in the view: no distance guides, pins or drag marker of the
        // hidden nuclei
        let hidden_by_ghost = self.ghost_now().map(|_| Scene { nuclei: Vec::new(), ..shown.clone() });
        if self.ghost.animating(Instant::now()) {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        let draft_banner = self.draft_banner();
        let bonds_label = self.bonds_label();
        self.ui.edit_group = if self.ui.mode == ui::ToolMode::Edit { self.tool.set(self.scene.nuclei.len()) } else { Vec::new() };
        // the live solve state, read per frame: a long solve publishes no snapshot
        let probe = self.sim.as_ref().map(|w| w.probe().view());
        let mut viewport = self.viewport;
        let screen_points = raw.screen_rect.map(|r| r.size());
        let full = ctx.run_ui(raw, |root| {
            let out = ui::show(
                root,
                &mut self.ui,
                &ui::PanelInput {
                    scene: &self.scene,
                    snapshot: self.snapshot.as_ref(),
                    busy: self.busy,
                    playing: self.playing,
                    last_error: self.last_error.as_deref(),
                    selected: self.tool.selected,
                    measured_fs_per_s,
                    camera: &self.camera,
                    density_label: density_label.clone(),
                    orbital_label: orbital_label.clone(),
                    orbital_waiting,
                    pseudo_generating: providers::pseudo::generating(),
                    draft_banner: draft_banner.clone(),
                    bonds_label: bonds_label.clone(),
                    probe,
                },
            );
            if let Some(r) = out.view {
                viewport = viewport_of(r, root.pixels_per_point());
            }
            outs.push(out);
            let pins = self.snapshot.as_ref().map_or(&[][..], |s| &s.dynamics.pins[..]);
            // the guides, not over the File window (they share its foreground layer)
            let guide_vp = if self.ui.files.open || self.ui.builder.open { Viewport { size: [0.0, 0.0], ..viewport } } else { viewport };
            Self::distance_overlay(
                root,
                &self.camera,
                hidden_by_ghost.as_ref().unwrap_or(&shown),
                &self.tool,
                self.ui.mode,
                self.drag_target,
                pins,
                guide_vp,
                self.tool.dragging.then(|| {
                    let mut l = ui::pull_label(self.ui.drag_force);
                    if !self.playing {
                        l += " (pulls while playing)";
                    }
                    l
                }),
            );
            self.label_overlay(root, &shown, viewport);
        });
        self.viewport = viewport;
        let ppp = ctx.pixels_per_point();
        if let Some(s) = screen_points {
            self.screen = Some((s.x * ppp, s.y * ppp));
        }
        // egui may run the closure more than once in a frame (discarded passes); input is
        // consumed in the first, so keeping every pass's output loses nothing and repeats nothing.
        self.chrome_actions.clear();
        for mut out in outs {
            self.chrome_actions.append(&mut out.actions);
            self.act_on(out);
        }
        full
    }
}

/// More drawn bonds than this: only the selected or hovered atom's bonds get a length label.
pub const BOND_LENGTH_DECLUTTER: usize = 12;

/// More atoms than this: only the selected or hovered atom and its bonded neighbours get a
/// partial-charge label.
pub const CHARGE_DECLUTTER: usize = 12;

/// "0.97 Å" for a length in Bohr (display; `units` converts).
pub fn bond_length_text(r_bohr: f64) -> String {
    format!("{:.2} Å", units::bohr_to_angstrom(r_bohr))
}

/// One bond-angle label of a frame (`Viewer::angle_labels`): the degree text, where it anchors in
/// the world, and the pill's text and background alpha.
#[derive(Clone, Debug, PartialEq)]
pub struct AngleLabel {
    pub text: String,
    pub anchor: [f64; 3],
    pub alpha: u8,
    pub bg: u8,
    /// Which measurement this is ((0, vertex, a, b) for an angle, (1, i, j, 0) for a bond): the
    /// smoothing's key.
    pub key: (u8, usize, usize, usize),
    /// The instantaneous value (radians or Bohr), averaged for display by [`LabelSmoother`].
    pub value: f64,
    /// A bond's two ends (to set its label beside the bond, away from the molecule).
    pub ends: Option<([f64; 3], [f64; 3])>,
}

/// Wall-clock time constant of the labels' anchor smoothing (s).
pub const LABEL_SMOOTH_S: f64 = 0.35;
/// The window of the labels' value average (s), as the temperature readout's.
pub const LABEL_AVG_S: f64 = 1.0;

/// Readable measurement labels (user request 2026-10-18: bond-length and angle labels bouncing
/// with the vibrating atoms were hard to read). Each label's world anchor is low-pass filtered with
/// [`LABEL_SMOOTH_S`] (in world space, so a camera move does not lag) and its value is the mean over
/// the last [`LABEL_AVG_S`] of wall time. Labels not drawn in a frame are forgotten. Display only.
#[derive(Default)]
pub struct LabelSmoother {
    entries: std::collections::HashMap<(u8, usize, usize, usize), SmoothEntry>,
    frame: f64,
}

struct SmoothEntry {
    anchor: [f64; 3],
    at: f64,
    values: std::collections::VecDeque<(f64, f64)>,
    seen: f64,
}

impl LabelSmoother {
    /// Starts a frame at wall time `now` (s); entries not updated in the previous frame are dropped.
    pub fn begin(&mut self, now: f64) {
        let last = self.frame;
        self.entries.retain(|_, e| e.seen >= last);
        self.frame = now;
    }

    /// [`LabelSmoother::begin`] unless this frame (same `now`) has already begun — several label
    /// groups share one smoother per frame.
    pub fn begin_once(&mut self, now: f64) {
        if now != self.frame {
            self.begin(now);
        }
    }

    /// The smoothed anchor and averaged value of `key` this frame.
    pub fn update(&mut self, key: (u8, usize, usize, usize), anchor: [f64; 3], value: f64) -> ([f64; 3], f64) {
        let now = self.frame;
        let e = self.entries.entry(key).or_insert_with(|| SmoothEntry { anchor, at: now, values: Default::default(), seen: now });
        let a = 1.0 - (-(now - e.at).max(0.0) / LABEL_SMOOTH_S).exp();
        e.anchor = std::array::from_fn(|k| e.anchor[k] + a * (anchor[k] - e.anchor[k]));
        e.at = now;
        e.seen = now;
        e.values.push_back((now, value));
        while e.values.len() > 1 && e.values.front().is_some_and(|&(t, _)| now - t > LABEL_AVG_S) {
            e.values.pop_front();
        }
        let mean = e.values.iter().map(|v| v.1).sum::<f64>() / e.values.len() as f64;
        (e.anchor, mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use render::headless::{Headless, HEADLESS_FORMAT};
    use std::time::Duration;

    /// Spec §7 / plan Task 10 step 1: with the Select tool and the cursor over nucleus 0 the badge
    /// is that nucleus's — its isotope (²H after an isotope swap), protons, neutrons and the raw
    /// force — at the cursor; off the nuclei it is the selection's, beside it; View ▸ badge off
    /// shows none.
    #[test]
    fn the_badge_follows_the_picked_nucleus() {
        let mut v = Viewer::new("h2").unwrap();
        v.scene.nuclei[1].pos = [0.0, 3.0, 0.0]; // apart on screen from the default camera
        let forces = vec![[0.0, 0.012, 0.0], [0.003, 0.004, 0.0]];
        let mut d = sim::DynSnapshot::default();
        d.forces = forces.clone();
        d.forces_stale = false;
        v.snapshot = Some(Snapshot {
            scene: v.scene.clone(),
            dynamics: d,
            backend: Backend::Gpu,
            engine: Backend::Gpu,
            model: None,
            atom_profiles: Default::default(),
            edits: 0,
            driver_commands: 0,
            commands: 0,
            working: false,
            labels: Default::default(),
            charge_note: None,
            activity: Default::default(),
        });
        v.ui.mode = ui::ToolMode::Select;
        let shown = v.scene.clone();
        let at = |v: &Viewer, i: usize| tools::project(&v.camera, tools::v3(v.scene.nuclei[i].pos), v.viewport).unwrap();
        let p0 = at(&v, 0);
        v.tool.cursor = Some(p0);
        assert_eq!(v.badge(&shown), Some((crate::badge::badge_text(1, 1, Some(forces[0])), p0)));
        // an isotope: the mirror's own A
        let mut deut = shown.clone();
        deut.nuclei[0].a = Some(2);
        assert_eq!(v.badge(&deut).unwrap().0, "²H · 1 p · 1 n · |F| 0.012 Ha/Bohr");
        // off the nuclei: the selection's, beside it; nothing without a selection
        v.tool.cursor = Some((5.0, 5.0));
        assert_eq!(v.badge(&shown), None);
        v.tool.selected = Some(1);
        assert_eq!(v.badge(&shown), Some(("¹H · 1 p · 0 n · |F| 0.005 Ha/Bohr".to_string(), at(&v, 1))));
        v.ui.show_badge = false;
        assert_eq!(v.badge(&shown), None);
    }

    /// Atom trails (user request 2026-10-17): nothing is collected while Trails is off, a snapshot
    /// taken before an unconfirmed edit is not recorded, the history is the confirmed snapshots'
    /// positions, a teleport breaks that atom's trail (a drag does not), and the lines reach the
    /// overlay only while on.
    #[test]
    fn trails_collect_only_while_on_and_break_on_a_teleport() {
        fn feed(v: &mut Viewer, t_fs: f64, x: f64, edits: u64) {
            let mut scene = v.scene.clone();
            scene.nuclei[0].pos = [x, 0.0, 0.0];
            scene.nuclei[1].pos = [0.0, x, 0.0];
            let mut d = sim::DynSnapshot::default();
            d.sim_time = t_fs / units::au_time_to_fs(1.0);
            v.snapshot = Some(Snapshot {
                scene,
                dynamics: d,
                backend: Backend::Gpu,
                engine: Backend::Gpu,
                model: None,
                atom_profiles: Default::default(),
                edits,
                driver_commands: 0,
                commands: 0,
                working: false,
                labels: Default::default(),
                charge_note: None,
                activity: Default::default(),
            });
            v.record_trails();
        }
        let mut v = Viewer::new("h2").unwrap();
        assert!(!v.ui.show_trails && v.trails.is_empty());
        // off: no history is collected
        feed(&mut v, 0.0, 0.0, 0);
        feed(&mut v, 1.0, 0.1, 0);
        assert!(v.trails.is_empty());
        let shown = v.scene.clone();
        assert!(v.overlay_lines(&shown).len() == crate::overlay::build(&v.ui, &v.scene, &v.camera, v.ui.mode).len());
        // on: from now
        v.ui.show_trails = true;
        feed(&mut v, 2.0, 0.2, 0);
        feed(&mut v, 3.0, 0.3, 0);
        feed(&mut v, 4.0, 0.4, 0);
        assert_eq!((v.trails.points(0), v.trails.points(1)), (3, 3));
        // the overlay carries the trail between the oldest point and the drawn position
        let mut shown = v.scene.clone();
        shown.nuclei[0].pos = [0.4, 0.0, 0.0];
        shown.nuclei[1].pos = [0.0, 0.4, 0.0];
        let base = crate::overlay::build(&v.ui, &v.scene, &v.camera, v.ui.mode).len();
        let with = v.overlay_lines(&shown);
        assert_eq!(with.len(), base + 2 * 2 * 2, "two segments per atom");
        v.ui.show_trails = false;
        assert_eq!(v.overlay_lines(&shown).len(), base);
        v.ui.show_trails = true;
        // a snapshot that does not hold every edit sent is not recorded; a teleport breaks its atom's trail
        v.send(Command::MoveNucleus { index: 0, pos: [2.0, 2.0, 2.0] });
        assert_eq!((v.trails.points(0), v.trails.points(1)), (0, 3));
        feed(&mut v, 5.0, 0.5, 0);
        assert_eq!((v.trails.points(0), v.trails.points(1)), (0, 3), "taken before the edit");
        let sent = v.edits_sent;
        feed(&mut v, 5.0, 0.5, sent);
        assert_eq!((v.trails.points(0), v.trails.points(1)), (1, 4));
        // a drag keeps the trail
        v.send(Command::Dyn(DynCommand::DragEnd));
        assert_eq!((v.trails.points(0), v.trails.points(1)), (1, 4));
        // adding a nucleus clears every trail
        v.send(Command::AddNucleus { z: 1, a: None, pos: [3.0, 0.0, 0.0] });
        assert!(v.trails.is_empty());
        // turning Trails off forgets the history
        v.ui.show_trails = false;
        v.record_trails();
        assert!(v.trails.is_empty());
    }

    /// A snapshot carrying only a bond graph (for the bond-angle overlay's wiring).
    fn snapshot_with_bonds(scene: &Scene, g: sim::BondGraph, stale: bool) -> Snapshot {
        let mut d = sim::DynSnapshot::default();
        d.bonds = Some(std::sync::Arc::new(g));
        d.bonds_stale = stale;
        Snapshot {
            scene: scene.clone(),
            dynamics: d,
            backend: Backend::Gpu,
            engine: Backend::Gpu,
            model: None,
            atom_profiles: Default::default(),
            edits: 0,
            driver_commands: 0,
            commands: 0,
            working: false,
            labels: Default::default(),
            charge_note: None,
            activity: Default::default(),
        }
    }

    /// Review R1: the spheres are drawn at the orbital scale only while the newest payload is the one
    /// the current show mode draws — a payload of another request (a mode switch not yet answered)
    /// draws nothing, so the spheres, and the arcs clearing them, stay at full size.
    #[test]
    fn sphere_scale_follows_the_payload_the_painter_draws() {
        use sim::{OrbitalDisplay, OrbitalRequest, Spin};
        let payload = |what| OrbitalDisplay {
            cell: sim::Cell { n: 33, h: 0.25 }, origin: [0.0; 3], positions: Vec::new(), rows: Vec::new(), field: None,
            spin: sim::SpinSummary::default(), unoccupied: None, notes: Vec::new(), what, lone_atom: sim::LoneAtomLabels::NotLoneAtom,
        };
        let empty = sim::BondGraph { bonds: Vec::new(), components: Vec::new(), step: 0, sim_time: 0.0, n_nuclei: 0 };
        let mut v = Viewer::new("h2").unwrap();
        let mut snap = snapshot_with_bonds(&v.scene, empty, false);
        snap.dynamics.orbitals = Some(payload(OrbitalRequest::Orbital { spin: Spin::Up, track: 1, squared: false }));
        v.snapshot = Some(snap);
        assert_eq!(v.sphere_scale(), 1.0, "Density mode ignores a payload");
        v.ui.show = ui::ShowMode::Orbitals;
        assert_eq!(v.sphere_scale(), 0.3, "an orbital payload in Orbitals mode");
        v.ui.show = ui::ShowMode::Spin;
        assert_eq!(v.sphere_scale(), 1.0, "Spin selected, the old orbital payload is not drawn");
        v.snapshot.as_mut().unwrap().dynamics.orbitals = Some(payload(OrbitalRequest::SpinDensity));
        assert_eq!(v.sphere_scale(), 0.3, "the spin payload arrived");
        v.snapshot.as_mut().unwrap().dynamics.orbitals = None;
        assert_eq!(v.sphere_scale(), 1.0, "no payload yet");
        v.snapshot.as_mut().unwrap().dynamics.orbitals = Some(payload(OrbitalRequest::SpinDensity));
        v.ui.nucleus_scale = Some(0.5);
        assert_eq!(v.sphere_scale(), 0.5, "the --ui nuclei= override wins");
    }

    /// User request 2026-10-18: with follow on, the camera's target is the selected nucleus as
    /// drawn; off, or nothing selected, the camera is left alone.
    #[test]
    fn follow_keeps_the_camera_on_the_selected_atom() {
        let mut v = Viewer::new("h2").unwrap();
        let shown = Scene::parse("grid 49 0.25\ncharge 0\nnucleus 1 1 2 3\nnucleus 1 2 2 3\n").unwrap();
        let before = v.camera.target;
        v.follow_selected(&shown);
        assert_eq!(v.camera.target, before, "follow off");
        v.ui.follow_selected = true;
        v.follow_selected(&shown);
        assert_eq!(v.camera.target, before, "nothing selected");
        v.tool.selected = Some(1);
        v.follow_selected(&shown);
        assert_eq!(v.camera.target, tools::v3([2.0, 2.0, 3.0]));
    }

    /// User request 2026-10-18: the labels are smoothed — the anchor follows with LABEL_SMOOTH_S and
    /// the value is the mean over LABEL_AVG_S; a label not drawn for a frame starts afresh.
    #[test]
    fn label_smoother_averages_and_follows() {
        let mut s = LabelSmoother::default();
        s.begin(0.0);
        assert_eq!(s.update((1, 0, 1, 0), [0.0; 3], 1.0), ([0.0; 3], 1.0));
        s.begin(0.35);
        let (a, v) = s.update((1, 0, 1, 0), [1.0, 0.0, 0.0], 3.0);
        assert!((a[0] - (1.0 - (-1.0f64).exp())).abs() < 1e-12, "{a:?}");
        assert_eq!(v, 2.0);
        s.begin(1.5);
        let (_, v) = s.update((1, 0, 1, 0), [1.0, 0.0, 0.0], 5.0);
        assert_eq!(v, 5.0, "older values left the window");
        s.begin(1.6);
        s.begin(1.7);
        let (a, _) = s.update((1, 0, 1, 0), [9.0, 0.0, 0.0], 0.0);
        assert_eq!(a, [9.0, 0.0, 0.0], "a label not drawn for a frame starts afresh");
    }

    /// User request 2026-10-18: each drawn bond's length in Å at its midpoint; off with View ▸ bond
    /// lengths or bonds; a graph of another nucleus count draws nothing; stale graphs fade.
    #[test]
    fn bond_lengths_are_labelled_at_the_midpoints() {
        let mut v = Viewer::new("h2").unwrap();
        let shown = Scene::parse("grid 49 0.25
charge 0
nucleus 1 0 0 0
nucleus 1 1.4 0 0
").unwrap();
        let bond = sim::Bond { i: 0, j: 1, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: sim::bonds::Route::Saddle, multi_centre: false, laplacian_bcp: -1.0, laplacian_noise: 0.1 };
        let g = |n| sim::BondGraph { bonds: vec![bond.clone()], components: vec![vec![0, 1]], step: 1, sim_time: 0.0, n_nuclei: n };
        v.snapshot = Some(snapshot_with_bonds(&shown, g(2), false));
        let l = v.bond_length_labels(&shown);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].text, bond_length_text(1.4));
        assert_eq!(l[0].text, "0.74 Å");
        assert_eq!(l[0].anchor, [0.7, 0.0, 0.0]);
        assert_eq!((l[0].alpha, l[0].bg), (235, 150));
        v.ui.show_bond_lengths = false;
        assert!(v.bond_length_labels(&shown).is_empty());
        v.ui.show_bond_lengths = true;
        v.ui.show_bonds = false;
        assert!(v.bond_length_labels(&shown).is_empty());
        v.ui.show_bonds = true;
        v.snapshot = Some(snapshot_with_bonds(&shown, g(3), false));
        assert!(v.bond_length_labels(&shown).is_empty(), "another nucleus count");
        v.snapshot = Some(snapshot_with_bonds(&shown, g(2), true));
        assert_eq!((v.bond_length_labels(&shown)[0].alpha, v.bond_length_labels(&shown)[0].bg), (110, 70), "stale fades");
    }

    /// M7 Task 13 (review I1): the viewer's wiring of the bond-angle overlay — a methane carbon with
    /// four bonds shows no angle until it is selected or hovered (hover is the Select tool's pick, not
    /// the Orbit tool's), an H with one bond never does, View ▸ bond angles / ▸ bonds off and a graph
    /// of another nucleus count draw nothing, and a stale graph fades the arcs and the labels.
    #[test]
    fn bond_angles_follow_the_declutter_gating_and_the_stale_fade() {
        use crate::overlay;
        let mut v = Viewer::new("h2").unwrap();
        v.ui.show_cell = false;
        v.ui.show_floor = false;
        v.ui.show_axes = false;
        v.ui.show_place_grid = false;
        v.ui.walls_on = false;
        let a = 2.06 / 3f64.sqrt();
        let shown = Scene::parse(&format!(
            "grid 49 0.25
charge 0
nucleus 6 0 0 0
nucleus 1 {a} {a} {a}
nucleus 1 {} {} {a}
nucleus 1 {} {a} {}
nucleus 1 {a} {} {}
",
            -a, -a, -a, -a, -a, -a
        ))
        .unwrap();
        let pos: Vec<[f64; 3]> = shown.nuclei.iter().map(|n| n.pos).collect();
        let graph = |n: usize, pairs: &[(usize, usize)]| sim::BondGraph {
            bonds: pairs.iter().map(|&(i, j)| sim::Bond { i, j, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: sim::Route::Saddle, multi_centre: false, laplacian_bcp: -1.0, laplacian_noise: 0.1 }).collect(),
            components: vec![(0..n).collect()],
            step: 1,
            sim_time: 0.0,
            n_nuclei: n,
        };
        let ch4 = graph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)]);
        let bond_len = overlay::bond_lines(&pos, &ch4, false).len();
        v.snapshot = Some(snapshot_with_bonds(&shown, ch4.clone(), false));
        v.camera = render::camera::OrbitCamera::new(Vec3::ZERO, 10.0);
        let arcs = |v: &Viewer| v.overlay_lines(&shown).split_off(bond_len);

        // nothing selected or hovered: four bonds at C, so no angle
        assert_eq!(v.overlay_lines(&shown).len(), bond_len);
        assert!(v.angle_labels(&shown).is_empty());
        // selected C: its six angles, fresh alpha; an H (one bond) has none
        v.tool.selected = Some(0);
        let fresh = arcs(&v);
        assert!(!fresh.is_empty() && fresh.iter().all(|p| p.colour[3] == overlay::BOND_ALPHA));
        let labels = v.angle_labels(&shown);
        assert_eq!(labels.len(), 6);
        assert!(labels.iter().all(|l| (l.alpha, l.bg) == (235, 150) && l.text == "109.5°"), "{labels:?}");
        v.tool.selected = Some(1);
        assert_eq!(v.overlay_lines(&shown).len(), bond_len);
        assert!(v.angle_labels(&shown).is_empty());
        // an out-of-range selection (a stale index after a delete) is no selection
        v.tool.selected = Some(99);
        assert!(v.angle_labels(&shown).is_empty());
        // hovered C: only with a tool that hovers (Inspect, Grab, Edit — Inspect since 15bf237, not
        // Place) and the cursor over it
        v.tool.selected = None;
        let at_c = tools::project(&v.camera, Vec3::ZERO, v.viewport).unwrap();
        assert_eq!(tools::pick(&v.camera, at_c.0, at_c.1, v.viewport, &shown.nuclei), Some(0), "the test's cursor is over C");
        v.tool.cursor = Some(at_c);
        v.ui.mode = ui::ToolMode::Place;
        assert!(v.angle_labels(&shown).is_empty(), "the Place tool does not hover");
        for mode in [ui::ToolMode::Orbit, ui::ToolMode::Edit, ui::ToolMode::Select] {
            v.ui.mode = mode;
            assert_eq!(v.angle_labels(&shown).len(), 6, "{mode:?}'s pick of C");
        }
        v.tool.cursor = Some((5.0, 5.0));
        assert!(v.angle_labels(&shown).is_empty(), "off the nuclei");
        v.tool.cursor = None;
        // the toggles and a graph of another nucleus count
        v.tool.selected = Some(0);
        v.ui.show_angles = false;
        assert_eq!(v.overlay_lines(&shown).len(), bond_len);
        assert!(v.angle_labels(&shown).is_empty());
        v.ui.show_angles = true;
        v.ui.show_bonds = false;
        assert!(v.overlay_lines(&shown).is_empty() && v.angle_labels(&shown).is_empty(), "bonds off: no bonds, no angles");
        v.ui.show_bonds = true;
        let mut fewer = shown.clone();
        fewer.nuclei.pop();
        assert!(v.angle_labels(&fewer).is_empty(), "a graph of another nucleus count");
        // a stale graph: faded arcs and labels
        v.snapshot = Some(snapshot_with_bonds(&shown, ch4, true));
        let stale = arcs(&v);
        assert_eq!(stale.len(), fresh.len());
        assert!(stale.iter().all(|p| p.colour[3] == overlay::STALE_BOND_ALPHA));
        assert!(v.angle_labels(&shown).iter().all(|l| (l.alpha, l.bg) == (110, 70)));
    }

    /// A viewer on `preset` pinned to the preset file's own grid (49³ × 0.25 for the 12 Bohr
    /// presets, 57³ for h2o) instead of the app's default quality (Accurate, 81³) — and for h2 to
    /// the R = 1.6 Bohr geometry (`scene::h2_r16`): these tests measure the viewer and the worker,
    /// not the grid or the preset, and were written and timed at that grid and geometry.
    fn pinned(preset: &str) -> Viewer {
        let mut v = Viewer::new(preset).unwrap();
        // all-electron, the physics they were written for (the app starts its presets under accepted species, M7 Task 11)
        v.ui.ionic = sim::IonicPolicy::None;
        // h2: the R = 1.6 Bohr geometry these tests were written for (the preset is 1.45 now)
        let file = if preset == "h2" { crate::scene::h2_r16() } else { Scene::preset(preset).unwrap() };
        v.ui.adopt(&file);
        v.ui.wall_half_width = crate::ui::default_wall_half_width(&file);
        v.scene = file;
        v
    }

    /// A fresh viewer is at the default quality, Accurate: h = σ = 0.15 with the preset's box kept
    /// — 81³ for the 12 Bohr presets (empty included: its worker publishes that cell without a
    /// solve), 95³ for the 14 Bohr boxes (lih, h2o, h2s, hcl, ph3, sih4).
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn a_fresh_viewer_runs_at_accurate_quality() {
        for (name, n) in [("empty", 81), ("h", 81), ("h2", 81), ("he", 81), ("lih", 95), ("h2o", 95), ("h2s", 95), ("hcl", 95), ("ph3", 95), ("sih4", 95)] {
            let v = Viewer::new(name).unwrap();
            assert_eq!((v.scene.grid_n, v.scene.grid_h, v.scene.sigma), (n, 0.15, None), "{name}");
            assert_eq!(v.ui.quality_h, 0.15);
            assert_eq!(v.ui.box_len, crate::quality::box_len(&Scene::preset(name).unwrap()));
        }
        let mut v = Viewer::new("empty").unwrap();
        v.start(Backend::Cpu, None);
        let s = first_snapshot(&v);
        assert_eq!((s.dynamics.scene.cell.n, s.dynamics.scene.cell.h), (81, 0.15));
    }

    /// The first snapshot `v`'s worker publishes.
    fn first_snapshot(v: &Viewer) -> crate::sim::Snapshot {
        let w = v.sim.as_ref().expect("a worker");
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(120) {
            match w.recv_timeout(Duration::from_secs(1)) {
                Some(SimEvent::Snapshot(s)) => return *s,
                Some(SimEvent::Error(m)) => panic!("{m}"),
                _ => {}
            }
        }
        panic!("no snapshot within 120 s");
    }

    /// Container walls are on by default (user request 2026-10-14): a fresh viewer's worker runs
    /// with them at the panel's default half-width, 1.5 Bohr inside the cell's face — 4.5 Bohr in
    /// the `empty` preset's 12 Bohr cell (81³ × 0.15 at the default quality; CPU: nothing is
    /// solved) — and on the estimated solver too (h2 pinned at 49³; GPU teacher, skipped without an
    /// adapter). The driver's own default stays off.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn a_fresh_worker_runs_with_the_panels_walls() {
        let mut v = Viewer::new("empty").unwrap();
        assert!(v.ui.walls_on);
        v.start(Backend::Cpu, None);
        let s = first_snapshot(&v);
        assert_eq!(s.dynamics.walls.map(|w| w.half_width), Some(4.5), "empty preset");
        assert_eq!(s.dynamics.walls.map(|w| w.k), Some(sim::Walls::new(4.5).k));
        // `--no-walls`'s setting: none
        let mut off = Viewer::new("empty").unwrap();
        off.ui.walls_on = false;
        off.start(Backend::Cpu, None);
        assert_eq!(first_snapshot(&off).dynamics.walls, None);
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(ctx) = crate::sim::tests::gpu_ctx() else { return };
        let mut v = pinned("h2");
        v.start(Backend::Estimated, Some(ctx));
        let s = first_snapshot(&v);
        assert_eq!(s.backend, Backend::Estimated);
        assert_eq!(s.dynamics.walls.map(|w| w.half_width), Some(4.5), "h2, estimated");
    }

    /// Draft mode's banner: shown while the model answers beyond what it learned (the count of
    /// atoms outside the envelope, the trust rule's reason), never on a teacher's step.
    #[test]
    fn the_draft_banner_counts_the_atoms_beyond_the_envelope() {
        let mut v = Viewer::new("h2").unwrap();
        let snap = |source, x: Option<sim::Extrapolation>| {
            let mut d = sim::DynSnapshot::default();
            d.source = source;
            d.eval.extrapolation = x;
            Snapshot {
                scene: v.scene.clone(),
                dynamics: d,
                backend: Backend::Estimated,
                engine: Backend::Gpu,
                model: None,
                atom_profiles: Default::default(),
                edits: 0,
                driver_commands: 0,
                commands: 0,
                working: false,
                labels: Default::default(),
                charge_note: None,
                activity: Default::default(),
            }
        };
        let x = |n, untrusted: Option<&str>| sim::Extrapolation { atoms_outside: n, untrusted: untrusted.map(str::to_string), reason: None, atom_envelope: vec![] };
        v.snapshot = Some(snap(Source::Surrogate, None));
        assert_eq!(v.draft_banner(), None);
        v.snapshot = Some(snap(Source::Surrogate, Some(x(2, None))));
        assert_eq!(v.draft_banner().as_deref(), Some("extrapolating: 2 atoms beyond what the model has learned — forces here are an educated guess"));
        v.snapshot = Some(snap(Source::Surrogate, Some(x(0, Some("surrogate untrusted: n_samples 10 < q_prime 29")))));
        assert!(v.draft_banner().unwrap().ends_with("(surrogate untrusted: n_samples 10 < q_prime 29)"));
        v.snapshot = Some(snap(Source::Fallback, Some(x(1, None))));
        assert_eq!(v.draft_banner(), None, "a teacher's step guesses nothing");
    }

    /// The cloud's labels (M5b): a side density says who solved it and when, faded; the
    /// superposition says it is not solved and how much of which real density is blended in.
    #[test]
    fn side_and_superposition_densities_are_labelled() {
        let mut d = DensityShown::default();
        let sd = |kind| sim::SideDensity {
            display: sim::Display::Cpu { cell: sim::Cell { n: 1, h: 0.25 }, origin: [0.0; 3], rho: std::sync::Arc::new(vec![0.0]), max: 0.0 },
            kind,
            t: units::fs_to_au_time(12.5),
            positions: vec![[0.0; 3]],
            seq: 1,
        };
        d.side(&sd(sim::SideKind::Audit), 60);
        assert_eq!(d.fade(), SURROGATE_DENSITY_FADE);
        let l = d.label(61).unwrap();
        assert!(l.starts_with("DFT density from the audit at t = 12.50 fs"), "{l}");
        d.side(&sd(sim::SideKind::Refresh), 60);
        assert!(d.label(61).unwrap().starts_with("DFT density from the refresh at t = 12.50 fs"));
        d.superposition = Some(0.25);
        assert_eq!(d.fade(), 1.0, "the superposition is drawn at full strength");
        let l = d.label(61).unwrap();
        assert!(l.starts_with("atomic superposition — not solved") && l.contains("blended 75 % with the DFT density from the refresh at t = 12.50 fs"), "{l}");
        d.superposition = Some(1.0);
        assert!(!d.label(61).unwrap().contains("blended"));
    }

    /// The density crossfade's α: the nuclei's formula on the densities' arrival times, a snap
    /// without a previous density or after a long gap.
    #[test]
    fn density_alpha_follows_the_arrivals_and_snaps() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(16);
        assert_eq!(density_alpha([None, Some(t1)], t1), 1.0, "nothing to fade from");
        assert_eq!(density_alpha([Some(t0), Some(t1)], t1), 0.0, "α = 0 at the latest arrival");
        assert!((density_alpha([Some(t0), Some(t1)], t1 + Duration::from_millis(8)) - 0.5).abs() < 1e-6);
        assert_eq!(density_alpha([Some(t0), Some(t1)], t1 + Duration::from_secs(1)), 1.0, "clamped at 1");
        let late = t0 + MOTION_MAX_GAP + Duration::from_millis(1);
        assert_eq!(density_alpha([Some(t0), Some(late)], late), 1.0, "a long gap snaps");
        // the nuclei's α at the same instants (one arrival each): identical
        let n = |x: f64| SceneNucleus { z: 1, a: None, pos: [x, 0.0, 0.0], vel: [0.0; 3] };
        let prev = MotionSample { nuclei: vec![n(0.0)], at: t0, step: 1 };
        let cur = MotionSample { nuclei: vec![n(1.0)], at: t1, step: 2 };
        for ms in [0u64, 4, 8, 12, 16, 30] {
            let now = t1 + Duration::from_millis(ms);
            let a = interpolate(&prev, &cur, now)[0].pos[0];
            assert!((density_alpha([Some(t0), Some(t1)], now) as f64 - a).abs() < 1e-6, "{ms} ms");
        }
    }

    /// Smooth motion: α runs from 0 (one publish interval behind: the previous snapshot) to 1
    /// (the latest) and the positions lerp monotonically; a change of the nuclei, time not
    /// advancing, or snapshots more than `MOTION_MAX_GAP` apart give the current scene as is.
    #[test]
    fn interpolation_is_monotone_and_snaps_on_count_change() {
        let n = |x: f64| SceneNucleus { z: 1, a: None, pos: [x, 0.0, 0.0], vel: [0.0; 3] };
        let t0 = Instant::now();
        let prev = MotionSample { nuclei: vec![n(0.0), n(2.0)], at: t0, step: 10 };
        let cur = MotionSample { nuclei: vec![n(1.0), n(3.0)], at: t0 + Duration::from_millis(16), step: 20 };
        let mut last = -1.0;
        for ms in [0u64, 4, 8, 12, 16, 40] {
            let p = interpolate(&prev, &cur, cur.at + Duration::from_millis(ms));
            let x = p[0].pos[0];
            assert!((0.0..=1.0).contains(&x) && x >= last, "x = {x} at +{ms} ms");
            assert!((p[1].pos[0] - p[0].pos[0] - 2.0).abs() < 1e-12, "both nuclei share α");
            last = x;
        }
        assert_eq!(interpolate(&prev, &cur, cur.at)[0].pos[0], 0.0, "α = 0 at the latest arrival");
        assert!((interpolate(&prev, &cur, cur.at + Duration::from_millis(8))[0].pos[0] - 0.5).abs() < 1e-12);
        assert_eq!(interpolate(&prev, &cur, cur.at + Duration::from_secs(1))[0].pos[0], 1.0, "clamped at 1");
        // an addition: the current scene, no lerp across indices
        let added = MotionSample { nuclei: vec![n(1.0), n(3.0), n(5.0)], ..cur.clone() };
        assert_eq!(interpolate(&prev, &added, added.at), added.nuclei);
        // a changed element at the same index snaps too
        let other = MotionSample { nuclei: vec![SceneNucleus { z: 2, ..n(1.0) }, n(3.0)], ..cur.clone() };
        assert_eq!(interpolate(&prev, &other, other.at), other.nuclei);
        // a stale window (paused, a long DFT step) and a restarted time: the current scene
        let late = MotionSample { at: t0 + Duration::from_millis(600), ..cur.clone() };
        assert_eq!(interpolate(&prev, &late, late.at), late.nuclei);
        let restarted = MotionSample { step: 0, ..cur.clone() };
        assert_eq!(interpolate(&prev, &restarted, restarted.at), restarted.nuclei);
    }

    /// Drops a frame's output the way the window would have consumed it: its texture deltas are
    /// marked handled (epaint asserts in debug builds that none is dropped unapplied).
    fn discard(mut full: egui::FullOutput) {
        full.textures_delta.clear();
    }

    /// The wheel over the 3D view: without a drag it zooms; during a Select-drag it leaves the
    /// camera alone and queues the pull for the next frame's chrome, which sends `SetDragForce`
    /// (×1.25 per line); the pull outlives the drag (the user's words: "scroll wheel change how
    /// much force you're applying").
    #[test]
    fn the_wheel_sets_the_pull_while_dragging_and_zooms_otherwise() {
        let mut v = Viewer::new("h2").unwrap();
        let d0 = v.camera.distance;
        v.wheel(1.0);
        assert_ne!(v.camera.distance, d0, "no drag: the wheel zooms");
        assert!(v.ui.queued.is_empty());
        let d1 = v.camera.distance;
        v.ui.mode = ui::ToolMode::Select;
        v.begin_drag(0, (Vec3::ZERO, Vec3::Z));
        v.wheel(2.0);
        assert_eq!(v.camera.distance, d1, "a drag's wheel never zooms");
        assert_eq!(v.ui.queued, vec![ui::Action::DragForceWheel(2.0)]);
        discard(v.run_ui(egui::RawInput::default()));
        assert!(v.ui.queued.is_empty());
        assert!((v.ui.drag_force - 0.1 * 1.25 * 1.25).abs() < 1e-15, "{}", v.ui.drag_force);
        v.end_drag();
        v.wheel(-1.0);
        assert!(v.ui.queued.is_empty() && v.camera.distance != d1, "after the drag the wheel zooms again");
        assert!((v.ui.drag_force - 0.15625).abs() < 1e-15, "the pull stays for the next drag");
    }

    /// The frame's viewport is the rect the chrome leaves (top bar above, toolbar below, dock and
    /// strip on the right), in physical pixels, snapped to whole pixels at a fractional scale.
    #[test]
    fn the_viewport_is_the_rect_the_chrome_leaves() {
        let mut v = Viewer::new("h2").unwrap();
        let screen = |w: f32, h: f32| egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h))),
            ..Default::default()
        };
        for _ in 0..2 {
            discard(v.run_ui(screen(1400.0, 900.0)));
        }
        let frames = |v: &mut Viewer| {
            for _ in 0..2 {
                discard(v.run_ui(screen(1400.0, 900.0)));
            }
            v.viewport
        };
        // the default (2026-10-19): both docks open — the left on Activity, the right on Model —
        // each 336 pt wide (`UiState::dock_w`) beside its icon strip
        let both = v.viewport;
        eprintln!("viewport at 1400×900: {both:?}");
        assert!((360.0..400.0).contains(&both.origin[0]), "the left dock and its strip are on the left: {both:?}");
        assert!((30.0..60.0).contains(&both.origin[1]), "the top bar is above: {both:?}");
        let bottom = both.origin[1] + both.size[1];
        assert!((1000.0..1040.0).contains(&(both.origin[0] + both.size[0])), "the right dock and its strip are on the right: {both:?}");
        assert!((780.0..880.0).contains(&bottom), "the toolbar is below: {both:?}");
        // folding the right dock gives its width back, up to its strip
        v.ui.dock_open = false;
        let left_only = frames(&mut v);
        assert_eq!(left_only.origin, both.origin, "the left dock did not move");
        assert!((1330.0..1360.0).contains(&(left_only.origin[0] + left_only.size[0])), "only the right dock's strip is on the right: {left_only:?}");
        // opening it again (another tab) takes the width back from the right
        v.ui.open_tab(crate::ui::DockTab::Selection);
        assert_eq!(frames(&mut v), both, "the right dock keeps its width across tabs");
        // folding the left dock widens the view on the left, to its strip
        v.ui.left_open = false;
        let right_only = frames(&mut v);
        assert!((30.0..60.0).contains(&right_only.origin[0]), "{right_only:?}");
        assert!(right_only.size[0] > both.size[0] + 250.0, "{right_only:?}");
        // both folded: the view spans the window between the two strips
        v.ui.dock_open = false;
        let none = frames(&mut v);
        assert!(none.size[0] > right_only.size[0] + 250.0, "{none:?}");
        assert!((1340.0..1370.0).contains(&(none.origin[0] + none.size[0])) && none.origin[0] < 60.0, "{none:?}");
        // points → physical pixels, whole pixels
        let r = viewport_of(egui::Rect::from_min_max(egui::pos2(0.0, 38.3), egui::pos2(971.6, 816.0)), 1.5);
        assert_eq!(r, Viewport { origin: [0.0, 57.0], size: [1457.0, 1167.0] });
    }

    /// Feeds worker events to the viewer until a snapshot from `backend` is shown and the worker
    /// is idle; returns the centre pixel's brightness of a frame rendered with it.
    fn settle(v: &mut Viewer, hl: &Headless, painter: &mut Painter, backend: Backend) -> u32 {
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "no {backend:?} snapshot"
            );
            let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) else {
                continue;
            };
            v.on_event(ev, &hl.ctx, painter, &hl.depth);
            assert_eq!(v.last_error, None);
            if !v.busy && v.snapshot.as_ref().is_some_and(|s| s.backend == backend) {
                break;
            }
        }
        let (w, h) = (hl.width, hl.height);
        painter
            .renderer
            .update_camera(&hl.ctx, &v.camera, w as f32, h as f32);
        let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
        let clear = wgpu::Color::BLACK;
        painter
            .renderer
            .render(&mut enc, &hl.colour_view, &hl.depth, clear);
        hl.ctx.queue.submit([enc.finish()]);
        let p = hl.read_rgba().expect("readback")[(h / 2 * w + w / 2) as usize];
        p[0] as u32 + p[1] as u32 + p[2] as u32
    }

    /// The panel's backend radio re-spawns the worker with the *current* scene (including an
    /// edit made before the switch) and the painter shows each backend's density: GPU (texture
    /// bound as is) → CPU (uploaded into the painter's own texture) → GPU again.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn switching_backend_respawns_the_worker_with_the_current_scene() {
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(hl) = Headless::new(96, 96) else {
            return;
        };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = pinned("h");
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        let gpu_lit = settle(&mut v, &hl, &mut painter, Backend::Gpu);
        v.send(Command::MoveNucleus {
            index: 0,
            pos: [0.25, 0.0, 0.0],
        });
        v.ui.backend = Backend::Cpu;
        discard(v.run_ui(egui::RawInput::default()));
        assert_eq!(v.backend, Backend::Cpu);
        let cpu_lit = settle(&mut v, &hl, &mut painter, Backend::Cpu);
        let s = v.snapshot.as_ref().unwrap();
        assert_eq!(
            s.scene, v.scene,
            "the new worker must solve the current scene"
        );
        assert_eq!(s.scene.nuclei[0].pos, [0.25, 0.0, 0.0]);
        assert!(s.dynamics.com_correction, "the panel's settings are re-sent to a new worker");
        v.ui.backend = Backend::Gpu;
        discard(v.run_ui(egui::RawInput::default()));
        let gpu_again = settle(&mut v, &hl, &mut painter, Backend::Gpu);
        eprintln!("centre brightness: GPU {gpu_lit}, CPU {cpu_lit}, GPU again {gpu_again}");
        for b in [gpu_lit, cpu_lit, gpu_again] {
            assert!(b > 150, "density not drawn: centre brightness {b}");
        }
    }

    /// While playing, the mirror follows the dynamics (the spheres move) — and a snapshot that
    /// predates an edit never undoes the edit on screen.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn mirror_follows_the_dynamics_and_keeps_unconfirmed_edits() {
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(hl) = Headless::new(64, 64) else {
            return;
        };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = pinned("h2");
        v.scene.grid_n = 33;
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        let start = v.scene.nuclei[0].pos;
        v.toggle_play();
        assert!(v.playing, "Play shows at once");
        let deadline = Instant::now() + Duration::from_secs(60);
        while v.snapshot.as_ref().is_none_or(|s| s.dynamics.step < 10) {
            assert!(Instant::now() < deadline, "no 10 steps");
            if let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) {
                v.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            }
        }
        assert_ne!(v.scene.nuclei[0].pos, start, "the spheres must move while playing");
        assert_eq!(v.scene, v.snapshot.as_ref().unwrap().scene);
        // an addition shows at once and survives the snapshots taken before the worker saw it
        v.send(Command::AddNucleus {
            z: 1,
            a: None,
            pos: [0.0, 2.5, 0.0],
        });
        assert_eq!(v.scene.nuclei.len(), 3);
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            assert!(Instant::now() < deadline, "the addition never arrived");
            let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) else {
                continue;
            };
            v.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            assert_eq!(v.scene.nuclei.len(), 3, "a stale snapshot undid the addition");
            if v.snapshot.as_ref().is_some_and(|s| s.scene.nuclei.len() == 3) {
                break;
            }
        }
        v.toggle_play();
        assert!(!v.playing);
    }

    /// Feeds events until a snapshot satisfies `f` (or 60 s pass).
    fn until(v: &mut Viewer, hl: &Headless, painter: &mut Painter, what: &str, f: impl Fn(&Snapshot) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            assert!(Instant::now() < deadline, "no snapshot with {what}");
            if let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) {
                v.on_event(ev, &hl.ctx, painter, &hl.depth);
                if v.snapshot.as_ref().is_some_and(&f) {
                    return;
                }
            }
        }
    }

    /// The panel's dynamics controls, through the viewer as the panel and the mouse send them
    /// (the window's checklist items that posted input cannot reach): a drag shows its target
    /// marker without moving the mirror; a pin holds a nucleus still while playing; Rewind flips
    /// every velocity; the thermostat switched on at rest starts from Maxwell–Boltzmann
    /// velocities with the driver's note; the gravity preset reaches the driver.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn panel_controls_reach_the_driver() {
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(hl) = Headless::new(64, 64) else {
            return;
        };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = pinned("h2");
        v.scene.grid_n = 33;
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        v.toggle_play();
        until(&mut v, &hl, &mut painter, "3 steps", |s| s.dynamics.step >= 3);
        // drag: marker set, mirror not jumped
        v.tool.selected = Some(0);
        let before = v.scene.nuclei[0].pos;
        let target = [before[0] + 0.5, before[1], before[2]];
        v.send(Command::Dyn(DynCommand::DragStart { index: 0, target }));
        assert_eq!(v.drag_target, Some(target));
        assert_eq!(v.scene.nuclei[0].pos, before, "the mirror must not jump to the target");
        v.send(Command::Dyn(DynCommand::DragEnd));
        assert_eq!(v.drag_target, None);
        // pin nucleus 1 while playing: it stays put over the following steps
        v.send(Command::Dyn(DynCommand::Pin(1, true)));
        until(&mut v, &hl, &mut painter, "the pin", |s| s.dynamics.pins == [false, true]);
        let (held, step) = {
            let s = v.snapshot.as_ref().unwrap();
            (s.dynamics.scene.nuclei[1].pos, s.dynamics.step)
        };
        until(&mut v, &hl, &mut painter, "5 more steps", |s| s.dynamics.step >= step + 5);
        let s = v.snapshot.as_ref().unwrap();
        assert_eq!(s.dynamics.scene.nuclei[1].pos, held, "a pinned nucleus moved");
        assert_eq!(s.dynamics.scene.nuclei[1].vel, [0.0; 3]);
        v.send(Command::Dyn(DynCommand::Pin(1, false)));
        // rewind while paused: velocities flip exactly
        v.toggle_play();
        until(&mut v, &hl, &mut painter, "a pause", |s| !s.working && s.dynamics.paused);
        let vel: Vec<[f64; 3]> = v.snapshot.as_ref().unwrap().dynamics.scene.nuclei.iter().map(|n| n.vel).collect();
        assert!(vel.iter().any(|x| x[0] != 0.0), "nothing moves: {vel:?}");
        v.send(Command::Dyn(DynCommand::Rewind));
        let flipped: Vec<[f64; 3]> = vel.iter().map(|x| x.map(|c| -c)).collect();
        until(&mut v, &hl, &mut painter, "flipped velocities", |s| {
            s.dynamics.scene.nuclei.iter().map(|n| n.vel).collect::<Vec<_>>() == flipped
        });
        // thermostat at rest: a replaced scene is at rest; switching on draws MB velocities
        let mut rest = crate::scene::h2_r16();
        rest.grid_n = 33;
        v.send(Command::SetScene(rest));
        until(&mut v, &hl, &mut painter, "the scene at rest", |s| s.dynamics.kinetic == 0.0 && !s.working);
        v.ui.thermostat_on = true;
        v.send(Command::Dyn(DynCommand::SetThermostat(Some((300.0, 10.0)))));
        until(&mut v, &hl, &mut painter, "the thermostat", |s| s.dynamics.thermostat == Some((300.0, 10.0)));
        let s = v.snapshot.as_ref().unwrap();
        let t_mb = s.dynamics.temperature;
        assert!(t_mb > 0.0, "no Maxwell–Boltzmann start");
        assert!(
            s.dynamics.notes.last().is_some_and(|n| n.contains("Maxwell–Boltzmann")),
            "{:?}",
            s.dynamics.notes
        );
        // the gravity preset reaches the driver as a −y acceleration of g in a.u.
        v.ui.gravity = true;
        v.send(Command::Dyn(DynCommand::SetExternalField(v.ui.field())));
        until(&mut v, &hl, &mut painter, "the field", |s| s.dynamics.scene.external_field[1] < 0.0);
        let g = -v.snapshot.as_ref().unwrap().dynamics.scene.external_field[1];
        assert_eq!(g, ui::gravity_au());
        eprintln!(
            "panel controls: pin held nucleus 1 at {held:?}; rewind flipped {vel:?}; thermostat MB start T = {t_mb:.0} K; g = {g:.4e} a.u."
        );
    }

    /// The estimated radio end to end, through the viewer as the panel drives it: switching to
    /// it starts a sampling pass (the panel's plan, `EnsureModel`); after it, playing shows —
    /// with the atomic superposition off — the last DFT density faded to
    /// `SURROGATE_DENSITY_FADE` with its label while the surrogate answers, and with it on (M5b)
    /// the superposition at full strength, labelled not solved, once the worker has the H atom's
    /// profile; switching to GPU and back restores the model from the session autosave without a
    /// new pass; the autosave is per viewer.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn estimated_radio_samples_fades_the_density_and_keeps_the_model_across_a_switch() {
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(hl) = Headless::new(64, 64) else {
            return;
        };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = pinned("h2");
        assert_ne!(v.autosave, Viewer::new("h2").unwrap().autosave, "one autosave per viewer");
        v.scene.grid_n = 41;
        v.ui.plan = sim::SamplingPlan { perturbations: 30, hot_burst_steps: 40, ..Default::default() };
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        assert_eq!(v.density.fade(), 1.0);
        v.ui.cloud_superposition = false;
        v.ui.backend = Backend::Estimated;
        discard(v.run_ui(egui::RawInput::default()));
        assert_eq!(v.backend, Backend::Estimated);
        until(&mut v, &hl, &mut painter, "a running pass", |s| crate::sim::sampling_active(&s.dynamics));
        assert!(!v.playing);
        let deadline = Instant::now() + Duration::from_secs(120);
        while !v.snapshot.as_ref().is_some_and(|s| s.dynamics.sampling.as_ref().is_some_and(|p| p.finished)) {
            assert!(Instant::now() < deadline, "the pass did not end");
            if let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) {
                v.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            }
        }
        let m = v.snapshot.as_ref().unwrap().model.clone().unwrap();
        assert!(m.trusted, "{m:?}");
        v.ui.speed = 1000;
        v.send(Command::Dyn(DynCommand::SetSpeed(1000)));
        v.toggle_play();
        until(&mut v, &hl, &mut painter, "a surrogate step", |s| s.dynamics.source == Source::Surrogate && s.dynamics.step > 10);
        assert!(v.density.faded && v.density.fade() == SURROGATE_DENSITY_FADE);
        let label = v.density.label(v.snapshot.as_ref().unwrap().dynamics.step).expect("a label");
        assert!(label.contains("density from the last DFT step (t = "), "{label}");
        // the atomic superposition (M5b): the panel's toggle on, the worker fetches the H atom
        v.ui.cloud_superposition = true;
        let c = v.ui.cloud_command();
        v.send(c);
        let deadline = Instant::now() + Duration::from_secs(60);
        while v.density.superposition.is_none() {
            assert!(Instant::now() < deadline, "no superposition shown");
            if let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) {
                v.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            }
        }
        let label = v.density.label(v.snapshot.as_ref().unwrap().dynamics.step).expect("a label");
        eprintln!("superposition shown: weight {:?}, \"{label}\"", v.density.superposition);
        assert!(label.starts_with("atomic superposition — not solved"), "{label}");
        assert_eq!(v.density.fade(), 1.0);
        v.toggle_play();
        until(&mut v, &hl, &mut painter, "a pause", |s| s.dynamics.paused && !s.working);
        // the model as it is now (the run's fallbacks, if any, were learned from)
        let m = v.snapshot.as_ref().unwrap().model.clone().unwrap();
        // away and back: the model comes back from the autosave, no new pass
        v.ui.backend = Backend::Gpu;
        discard(v.run_ui(egui::RawInput::default()));
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        // the old worker retires on its own thread (fix round 1, I2)
        let deadline = Instant::now() + Duration::from_secs(30);
        while v.retirements_pending() > 0 {
            assert!(Instant::now() < deadline, "the estimated worker never retired");
            std::thread::sleep(Duration::from_millis(20));
            v.poll_retiring();
        }
        assert!(v.autosave.exists(), "leaving the estimated solver writes the autosave");
        assert!(!v.density.faded, "a DFT step shows its own density");
        v.ui.backend = Backend::Estimated;
        discard(v.run_ui(egui::RawInput::default()));
        until(&mut v, &hl, &mut painter, "the restored model", |s| s.model.as_ref().is_some_and(|m| m.fitted));
        let s = v.snapshot.as_ref().unwrap();
        let back = s.model.as_ref().unwrap();
        assert!(s.dynamics.sampling.is_none(), "a pass started although the model was restored");
        assert_eq!((back.samples, back.fit, back.trusted), (m.samples, m.fit, true));
        eprintln!("estimated radio: {} samples, fit {:?}; faded label \"{label}\"; restored after GPU → estimated", m.samples, m.fit);
        let _ = std::fs::remove_file(&v.autosave);
    }

    /// Fix round 1, I2 + I1: leaving the estimated solver never blocks the render thread, and the
    /// toggles of a hurried user never lose the session autosave. The estimated worker is on a
    /// cold 97³ H₂O first evaluation (uncancellable, seconds) when the radio goes to GPU:
    /// `run_ui` (which calls `start`) returns in < 50 ms — a tripwire bound, hardware-dependent;
    /// before the fix it waited up to 3 s for the old worker. Then estimated → GPU again while
    /// the first is still retiring (the second estimated worker's `EnsureModel` is held back
    /// and never sent). After every retirement the autosave is byte-for-byte what it was.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "release-only: runs real SCF solves; cargo test -p app --release"
    )]
    fn leaving_the_estimated_solver_neither_blocks_nor_loses_the_autosave() {
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(hl) = Headless::new(64, 64) else {
            return;
        };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = pinned("h2o");
        v.scene.grid_n = 97;
        std::fs::write(&v.autosave, b"the session's model").unwrap();
        v.start(Backend::Estimated, Some(hl.ctx.clone()));
        std::thread::sleep(Duration::from_millis(300)); // well inside the cold 97³ evaluation
        let mut worst = Duration::ZERO;
        for b in [Backend::Gpu, Backend::Estimated, Backend::Gpu] {
            v.ui.backend = b;
            let t = Instant::now();
            discard(v.run_ui(egui::RawInput::default()));
            worst = worst.max(t.elapsed());
            assert_eq!(v.backend, b);
        }
        let pending = v.retirements_pending();
        let deadline = Instant::now() + Duration::from_secs(120);
        while v.retirements_pending() > 0 {
            assert!(Instant::now() < deadline, "the retiring workers never finished");
            if let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_millis(200)) {
                v.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            }
            v.poll_retiring();
        }
        eprintln!("backend switches during a cold 97³ estimated evaluation: worst run_ui {:.1} ms; {pending} retirements pending right after", worst.as_secs_f64() * 1e3);
        assert!(worst < Duration::from_millis(50), "a backend switch blocked the render thread for {worst:?}");
        assert_eq!(std::fs::read(&v.autosave).unwrap(), b"the session's model", "the autosave was rewritten or removed");
        let _ = std::fs::remove_file(&v.autosave);
    }

    /// `drive_orbital_view` (no worker needed): the orbital the producer resolved the placeholder
    /// to becomes the pick, once; the unoccupied orbitals held back for a freshly switched-on view
    /// are released with the first payload; the density mode and a stale pick are left alone.
    #[test]
    fn the_viewer_adopts_the_resolved_orbital_and_releases_the_unoccupied_rows() {
        use sim::{OrbitalDisplay, OrbitalInfo, OrbitalLabel, OrbitalRequest, Spin};
        let row = |spin, track| OrbitalInfo {
            spin, index: 0, track, slot: 0, energy: -0.5, eps_range: (-0.5, -0.5), occupation: 1.0, label: OrbitalLabel::Index(1),
            cluster: None, rotated: false, grid_split: None, box_state: false, smeared_1s: false, residual: None, not_converged: false,
        };
        let payload = |what, rows| OrbitalDisplay {
            cell: sim::Cell { n: 33, h: 0.25 }, origin: [0.0; 3], positions: Vec::new(), rows, field: None,
            spin: sim::SpinSummary::default(), unoccupied: None, notes: Vec::new(), what, lone_atom: sim::LoneAtomLabels::NotLoneAtom,
        };
        let mut v = pinned("h2");
        let mut d = sim::DynSnapshot::default();
        // Density mode ignores payloads
        d.orbitals = Some(payload(OrbitalRequest::Orbital { spin: Spin::Up, track: 4, squared: false }, vec![row(Spin::Up, 4)]));
        v.drive_orbital_view(&d);
        assert_eq!(v.ui.orbital_pick, None);
        // switched on: the placeholder (HOMO_TRACK) is not among the rows, so the resolved orbital is adopted
        v.apply(ui::Action::Show(ui::ShowMode::Orbitals));
        assert!(v.ui.unocc_deferred && v.ui.orbital_pick.is_none());
        v.drive_orbital_view(&d);
        assert_eq!(v.ui.orbital_pick, Some((Spin::Up, 4)), "the HOMO the producer showed");
        assert!(v.ui.orbital_pick_adopted, "adopted from the placeholder, not the user's pick");
        assert!(!v.ui.unocc_deferred, "the first payload releases the unoccupied rows");
        // the producer restarts its tracks (an isolated-atom job of the estimated solver): the adopted
        // HOMO is adopted again — still no pick of the user's, so its "no longer exists" note is untrue
        let restarted = payload(OrbitalRequest::Orbital { spin: Spin::Up, track: 20, squared: false }, vec![row(Spin::Up, 20)]);
        let mut d2 = d.clone();
        d2.orbitals = Some(OrbitalDisplay { notes: vec!["selected orbital no longer exists — showing #1 ↑".into()], ..restarted });
        v.drive_orbital_view(&d2);
        assert_eq!(v.ui.orbital_pick, Some((Spin::Up, 20)), "re-adopted");
        assert!(v.ui.orbital_pick_adopted, "still an adopted pick");
        assert!(crate::ui::orbital_tab::visible_notes(d2.orbitals.as_ref().unwrap(), v.ui.orbital_pick_adopted).is_empty(), "the untrue note is not shown");
        // a pick that is among the rows is the user's, whatever the payload's `what`
        v.ui.orbital_pick = Some((Spin::Up, 4));
        v.ui.orbital_pick_adopted = false;
        d.orbitals = Some(payload(OrbitalRequest::Orbital { spin: Spin::Up, track: 9, squared: false }, vec![row(Spin::Up, 4), row(Spin::Up, 9)]));
        v.drive_orbital_view(&d);
        assert_eq!(v.ui.orbital_pick, Some((Spin::Up, 4)));
        // a pick that vanished falls back to what the producer showed
        v.ui.orbital_pick = Some((Spin::Down, 77));
        v.drive_orbital_view(&d);
        assert_eq!(v.ui.orbital_pick, Some((Spin::Up, 9)));
        assert!(!v.ui.orbital_pick_adopted, "a vanished pick of the user's stays the user's: its note is true and is shown");
        d2.orbitals.as_mut().unwrap().what = OrbitalRequest::Orbital { spin: Spin::Up, track: 9, squared: false };
        assert_eq!(crate::ui::orbital_tab::visible_notes(d2.orbitals.as_ref().unwrap(), v.ui.orbital_pick_adopted).len(), 1);
    }

    /// A solver switch starts a new worker whose provider numbers its tracks afresh: the pick and
    /// the include set must not carry over (final review I5) and its first view holds back the
    /// unoccupied orbitals (minor 1: `run_settings` is built after the deferral is set). Through
    /// the real `start`, on the empty scene (answered without a solve).
    #[test]
    fn a_new_worker_starts_with_the_homo_placeholder_and_no_unoccupied_rows() {
        use sim::{OrbitalRequest, Spin};
        let mut v = Viewer::new("empty").unwrap();
        v.ui.show = ui::ShowMode::Orbitals;
        v.ui.unoccupied = 4;
        v.ui.orbital_pick = Some((Spin::Down, 4));
        v.ui.included = Some([1u32, 2].into_iter().collect());
        v.start(Backend::Cpu, None);
        assert_eq!((v.ui.orbital_pick, v.ui.included.clone()), (None, None));
        assert!(v.ui.unocc_deferred);
        let deadline = Instant::now() + Duration::from_secs(30);
        let first = loop {
            assert!(Instant::now() < deadline, "no snapshot from the new worker");
            if let Some(crate::sim::Event::Snapshot(s)) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) {
                break *s;
            }
        };
        let req = &first.dynamics.orbital_view;
        assert_eq!(req.what, OrbitalRequest::Orbital { spin: Spin::Up, track: sim::TRACK_HOMO, squared: false }, "the HOMO, not the old worker's track 4");
        assert_eq!(req.unoccupied, 0, "no unoccupied rows asked for until the first payload");
        // an include set of All orbitals coloured starts over too
        v.ui.show = ui::ShowMode::Coloured;
        v.ui.included = Some([7u32].into_iter().collect());
        v.start(Backend::Cpu, None);
        assert!(v.ui.included.is_none());
        let _ = v.sim.take();
    }

    /// `orbital_waiting`: a changed view, or a paused step with the view on, is a request out until
    /// a payload arrives; the same view again is not.
    #[test]
    fn a_request_out_is_waiting_until_its_payload_arrives() {
        let mut v = pinned("h2");
        assert!(!v.orbital_waiting);
        v.apply(ui::Action::Show(ui::ShowMode::Orbitals));
        assert!(v.orbital_waiting, "a changed view");
        v.orbital_waiting = false; // its payload arrived
        v.send(Command::OrbitalView(v.ui.orbital_request()));
        assert!(!v.orbital_waiting, "the same view again asks for nothing");
        v.apply(ui::Action::Step);
        assert!(v.orbital_waiting, "a paused step brings a payload");
        v.orbital_waiting = false;
        v.apply(ui::Action::Show(ui::ShowMode::Density));
        assert!(!v.orbital_waiting, "switching off waits for nothing");
    }

    /// The orbital view through the real worker: switched on while paused, the first payload brings
    /// the list at once and the unoccupied rows follow (the eigensolve is held back, then
    /// released), the picked orbital is drawn as a ± surface, All orbitals coloured draws the
    /// colour volume, Density brings the density back untouched, and a solver switch (a new worker)
    /// keeps the view — the CPU backend's payload (values the painter uploads) draws too.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only: runs real SCF solves; cargo test -p app --release")]
    fn the_orbital_view_runs_end_to_end_and_survives_a_solver_switch() {
        use crate::gfx::Drawn;
        let _gpu = crate::sim::tests::gpu_lock();
        let Some(hl) = Headless::new(96, 96) else { return };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = pinned("h2");
        v.ui.unoccupied = 2;
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        assert_eq!(painter.drawn(&v.ui), Drawn::Density);
        // the window's frame path (`run_ui`) ends a load before every render; this test renders
        // without it, so it ends the load itself — else `set_view` keeps the cloud hidden (since
        // 1285c6c a new worker's scene loads until its first solve)
        let frame = |v: &mut Viewer, painter: &mut Painter| -> (usize, usize, usize) {
            v.finish_loading();
            assert!(v.ui.loading.is_none(), "test premise: the new scene's first solve is in");
            v.set_view(painter, &hl.ctx);
            painter.renderer.update_camera(&hl.ctx, &v.camera, 96.0, 96.0);
            let mut enc = hl.ctx.device.create_command_encoder(&Default::default());
            painter.renderer.render(&mut enc, &hl.colour_view, &hl.depth, wgpu::Color::BLACK);
            hl.ctx.queue.submit([enc.finish()]);
            let px = hl.read_rgba().expect("readback");
            let lit = |f: &dyn Fn([u8; 4]) -> bool| px.iter().filter(|&&p| f(p)).count();
            (lit(&|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60), lit(&|p| p[0] as i32 - p[2] as i32 > 60), lit(&|p| p[2] as i32 - p[0] as i32 > 60))
        };
        let (density_lit, _, _) = frame(&mut v, &mut painter);
        assert!(density_lit > 100, "test premise: the density is drawn");
        // Orbitals: the first payload (no eigensolve yet), then the unoccupied rows
        v.apply(ui::Action::Show(ui::ShowMode::Orbitals));
        until(&mut v, &hl, &mut painter, "the first orbital payload", |s| s.dynamics.orbitals.as_ref().is_some_and(|o| o.field.is_some()));
        until(&mut v, &hl, &mut painter, "the unoccupied rows", |s| {
            s.dynamics.orbitals.as_ref().and_then(|o| o.unoccupied.as_ref()).is_some_and(|u| u.requested == 2 && u.current)
        });
        assert!(!v.ui.unocc_deferred);
        let o = v.snapshot.as_ref().unwrap().dynamics.orbitals.clone().unwrap();
        assert!(o.rows.iter().any(|r| r.occupation == 0.0), "the σ_u (or a box state) is listed");
        assert!(v.ui.orbital_pick.is_some(), "the HOMO was adopted");
        assert_eq!(painter.drawn(&v.ui), Drawn::Orbital);
        v.ui.volume_mode = ui::VOLUME_ISO;
        let (lit, red, blue) = frame(&mut v, &mut painter);
        eprintln!("σ_g as a ± surface: {lit} lit pixels, {red} red, {blue} blue");
        assert!(lit > 100 && red > 50, "the all-positive σ_g is a red surface ({lit} lit, {red} red)");
        assert_eq!(blue, 0, "no negative lobe in σ_g");
        assert_eq!(v.orbitals.label(&v.ui, v.snapshot.as_ref()), None, "the step's own orbitals carry no label");
        // All orbitals coloured
        v.apply(ui::Action::Show(ui::ShowMode::Coloured));
        until(&mut v, &hl, &mut painter, "the colour volume", |s| {
            s.dynamics.orbitals.as_ref().is_some_and(|o| matches!(o.field, Some(sim::OrbitalField::GpuColour { .. })))
        });
        v.ui.volume_mode = ui::VOLUME_VOLUME;
        assert!(frame(&mut v, &mut painter).0 > 100, "the colour volume is drawn");
        // Density again: the density layer was never touched
        v.apply(ui::Action::Show(ui::ShowMode::Density));
        assert_eq!(painter.drawn(&v.ui), Drawn::Density);
        assert_eq!(frame(&mut v, &mut painter).0, density_lit, "the density is back as it was");
        // the solver switch keeps the view: the new worker starts with it
        v.apply(ui::Action::Show(ui::ShowMode::Orbitals));
        v.ui.unoccupied = 0;
        v.ui.backend = Backend::Cpu;
        discard(v.run_ui(egui::RawInput::default()));
        assert_eq!(v.backend, Backend::Cpu);
        assert!(v.ui.show == ui::ShowMode::Orbitals);
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            assert!(Instant::now() < deadline, "no orbital payload from the CPU worker");
            if let Some(ev) = v.sim.as_ref().unwrap().recv_timeout(Duration::from_secs(1)) {
                v.on_event(ev, &hl.ctx, &mut painter, &hl.depth);
            }
            let cpu = v.snapshot.as_ref().filter(|s| s.backend == Backend::Cpu).and_then(|s| s.dynamics.orbitals.as_ref());
            if cpu.is_some_and(|o| matches!(o.field, Some(sim::OrbitalField::Cpu { .. }))) {
                break;
            }
        }
        assert_eq!(painter.drawn(&v.ui), Drawn::Orbital);
        let (lit, red, _) = frame(&mut v, &mut painter);
        assert!(lit > 100 && red > 50, "the CPU orbital is drawn too ({lit} lit, {red} red)");
    }

    // ---- show what training is solving (`crate::ghost`) ----

    fn ghost_of(job: sim::GhostJob, atoms: Vec<sim::GhostAtom>, serial: Option<(u64, sim::GhostOutcome)>) -> sim::PassGhost {
        let stage = match job {
            sim::GhostJob::Reference { .. } => "references",
            sim::GhostJob::Perturbation { .. } => "perturbations",
            sim::GhostJob::Scan { .. } | sim::GhostJob::ShellCheck { .. } => "scans",
            sim::GhostJob::HotBurst { .. } => "hot burst",
            sim::GhostJob::Teach { .. } => "teaching",
        };
        sim::PassGhost {
            stage,
            job,
            base: atoms.iter().map(|a| a.pos).collect(),
            finished: serial.map(|(serial, outcome)| sim::GhostFinished { serial, outcome, atoms: atoms.clone() }),
            atoms,
            trail: Vec::new(),
        }
    }

    fn pass_progress(done: usize, finished: bool, cancelled: bool) -> sim::SamplingProgress {
        sim::SamplingProgress { done, total: 20, stage: "scans", last_energy: -1.1, cancelled, skipped: 0, failed: 0, finished, target: None }
    }

    fn ghost_snapshot(v: &Viewer, sampling: Option<sim::SamplingProgress>, ghost: Option<sim::PassGhost>) -> Snapshot {
        let mut d = sim::DynSnapshot::default();
        d.scene = v.scene.to_sim().expect("the preset is a valid scene");
        d.sampling = sampling;
        d.ghost = ghost;
        Snapshot {
            scene: v.scene.clone(),
            dynamics: d,
            backend: Backend::Estimated,
            engine: Backend::Gpu,
            model: None,
            atom_profiles: Default::default(),
            edits: 0,
            driver_commands: 0,
            commands: 0,
            working: false,
            labels: Default::default(),
            charge_note: None,
            activity: Default::default(),
        }
    }

    fn ghost_lines_in(v: &Viewer) -> usize {
        // vertices in the element colour of the ghost's oxygen: only the ghost draws them (the
        // scene is H₂)
        let o = render::elements::colour(8);
        v.overlay_lines(&v.scene).iter().filter(|p| p.colour[..3] == o[..]).count()
    }

    /// While a pass runs the view draws the job's ghost instead of the scene — no spheres, no bonds,
    /// the cell kept — and the scene returns when the pass ends, is cancelled or fails (the driver
    /// then ships no ghost), and when the toggle is off.
    #[test]
    fn the_ghost_replaces_the_scene_while_a_pass_runs_and_the_scene_returns() {
        let mut v = Viewer::new("h2").unwrap();
        v.ui.show_cell = true;
        let shown = v.scene.clone();
        assert_eq!(v.drawn_nuclei(&shown).len(), 2, "no snapshot: the scene");
        let job = sim::GhostJob::Reference { z: 8 };
        let atoms = vec![sim::GhostAtom { z: 8, pos: [0.0; 3] }];
        let running = |v: &Viewer, done| ghost_snapshot(v, Some(pass_progress(done, false, false)), Some(ghost_of(job, atoms.clone(), None)));
        let s = running(&v, 3);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(v.ghost_now().is_some());
        assert!(v.drawn_nuclei(&shown).is_empty(), "the scene's spheres are not drawn");
        assert!(ghost_lines_in(&v) > 0, "the ghost's outline is");
        let cell = v.overlay_lines(&shown).iter().filter(|p| p.colour == [0.95, 0.80, 0.35, 0.55]).count();
        assert_eq!(cell, 24, "the cell box stays");
        assert_eq!(v.ghost_now().map(crate::ghost::label).as_deref(), Some("reference: isolated O"));

        // the toggle: off, the scene shows as before
        v.ui.show_ghost = false;
        assert!(v.ghost_now().is_none());
        assert_eq!(v.drawn_nuclei(&shown).len(), 2);
        assert_eq!(ghost_lines_in(&v), 0);
        v.ui.show_ghost = true;
        assert!(v.ghost_now().is_some());

        // the pass ends: no ghost, the scene is back
        let s = ghost_snapshot(&v, Some(pass_progress(20, true, false)), None);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(v.ghost_now().is_none());
        assert_eq!(v.drawn_nuclei(&shown).len(), 2);
        assert_eq!(ghost_lines_in(&v), 0);

        // cancelled: the driver's progress says finished + cancelled, and ships no ghost
        let s = running(&v, 5);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(v.ghost_now().is_some());
        let s = ghost_snapshot(&v, Some(pass_progress(5, true, true)), None);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(v.ghost_now().is_none() && v.drawn_nuclei(&shown).len() == 2, "cancelled");

        // failed: the pass is over (finished) with an error; even a stale ghost in that snapshot is not drawn
        let s = running(&v, 6);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(v.ghost_now().is_some());
        let mut s = ghost_snapshot(&v, Some(pass_progress(6, true, false)), Some(ghost_of(job, atoms.clone(), None)));
        s.dynamics.last_error = Some("sampling: the teacher failed".into());
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(v.ghost_now().is_none() && v.drawn_nuclei(&shown).len() == 2, "failed");
    }

    /// A finished job's flash: a new serial in a snapshot starts it (in the outcome's colour, on the
    /// finished geometry), it fades out within the flash time, the same serial does not restart it,
    /// and it is gone with the ghost. The labels come from the job.
    #[test]
    fn a_finished_sample_flashes_the_ghost_in_its_outcome_colour() {
        use sim::GhostOutcome as O;
        let mut v = Viewer::new("h2").unwrap();
        let atoms = vec![sim::GhostAtom { z: 8, pos: [0.0; 3] }, sim::GhostAtom { z: 1, pos: [2.0, 0.0, 0.0] }];
        let scan = |k, outcome| ghost_of(sim::GhostJob::Scan { i: 0, j: 1, k, of: 12, ratio: 1.1, r: 2.0 }, atoms.clone(), Some((k as u64, outcome)));
        let colour_count = |v: &Viewer, c: [f32; 3]| {
            let shown = v.scene.clone();
            v.overlay_lines(&shown).iter().filter(|p| p.colour[..3] == c[..]).count()
        };
        let s = ghost_snapshot(&v, Some(pass_progress(2, false, false)), Some(scan(2, O::Kept)));
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(colour_count(&v, crate::ghost::KEPT) > 0, "green for a kept sample");
        assert_eq!(colour_count(&v, crate::ghost::FAILED), 0);
        assert_eq!(v.ghost_now().map(crate::ghost::label).as_deref(), Some("O–H · 1.10 × r₀ · 2.00 Bohr"));
        // red for a stalled one, grey for a skipped one
        let s = ghost_snapshot(&v, Some(pass_progress(3, false, false)), Some(scan(3, O::Failed(Some(sim::activity::ScfEnd::Stalled)))));
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(colour_count(&v, crate::ghost::FAILED) > 0);
        // the kept sample stays behind it, fainter (the last three stay on screen: user request 2026-10-18)
        assert!(colour_count(&v, crate::ghost::KEPT) > 0);
        let s = ghost_snapshot(&v, Some(pass_progress(4, false, false)), Some(scan(4, O::Skipped)));
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert!(colour_count(&v, crate::ghost::SKIPPED) > 0);
        // expiry: the tracker's flash is gone after the flash time
        let later = Instant::now() + Duration::from_millis(600);
        assert!(v.ghost.flash(later).is_none());
        // ... and a repeat snapshot of the same serial does not bring it back
        let s = ghost_snapshot(&v, Some(pass_progress(4, false, false)), Some(scan(4, O::Skipped)));
        let before = v.ghost.flash(Instant::now()).map(|(f, _)| f.at);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert_eq!(v.ghost.flash(Instant::now()).map(|(f, _)| f.at), before);
        // the pass ends: no ghost, no flash
        let s = ghost_snapshot(&v, Some(pass_progress(20, true, false)), None);
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        assert_eq!(colour_count(&v, crate::ghost::SKIPPED), 0);
        assert!(v.ghost.flash(Instant::now()).is_none());
    }

    /// The density cloud hides while the ghost replaces the scene: the painter is told a fade of 0.
    #[test]
    fn the_ghost_hides_the_cloud() {
        let Some(hl) = Headless::new(64, 64) else { return };
        let mut painter = Painter::new(&hl.ctx, HEADLESS_FORMAT);
        let mut v = Viewer::new("h2").unwrap();
        v.set_view(&mut painter, &hl.ctx);
        let visible = painter.renderer.volume_params.fade;
        assert!(visible > 0.0, "the cloud is drawn when no pass runs");
        let atoms = vec![sim::GhostAtom { z: 1, pos: [0.0; 3] }];
        let s = ghost_snapshot(&v, Some(pass_progress(1, false, false)), Some(ghost_of(sim::GhostJob::Reference { z: 1 }, atoms, None)));
        v.on_event_bare(SimEvent::Snapshot(Box::new(s)));
        v.set_view(&mut painter, &hl.ctx);
        assert_eq!(painter.renderer.volume_params.fade, 0.0);
        v.ui.show_ghost = false;
        v.set_view(&mut painter, &hl.ctx);
        assert_eq!(painter.renderer.volume_params.fade, visible);
    }
}

/// An empty, non-interactive background area with this id over `rect`: its layer then has a fixed
/// place in egui's layer order (under the chrome's areas, menus and windows), so what a
/// `layer_painter` of the same id draws does not flicker.
fn fixed_layer(ctx: &egui::Context, id: &str, rect: egui::Rect) {
    egui::Area::new(egui::Id::new(id))
        .order(egui::Order::Background)
        .fixed_pos(rect.min)
        .interactable(false)
        .show(ctx, |ui| {
            ui.allocate_space(rect.size());
        });
}

/// Play refused in estimated mode without a model (the status line).
const NO_MODEL_PLAY: &str = "the estimated solver has no model to play on: load one (File ▸ Open) or press Start sampling in the Model tab";
