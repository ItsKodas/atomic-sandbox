//! The viewer's chrome (M6 UI redesign; the spec is the mockup,
//! `docs/superpowers/notes/2026-10-14-ui-mockup.html`, and its Appendix A maps every control of
//! the old side panel to its place here):
//!
//! - a **top bar** (`menus`): File / Scene / View / Physics / Help menus for what is set once,
//!   the scene chip (preset picker and Reset), and the Ctrl K control search (`search`);
//! - a **bottom toolbar** (`toolbar`) for what the hands do: time, speed with the measured fs/s,
//!   dt, the tool, the element to place, the selection's actions, the solver with its trust light;
//! - a **right dock** (`dock`) for what is read: Readouts, Graph, Model, Selection and Log tabs,
//!   collapsible to an icon strip whose dots keep drift, trust and errors visible;
//! - **overlays** in the 3D view (`overlays`): the status chip, the atom card, the faded-density
//!   label, toasts, the sampling and first-pass cards, the Volume / Isosurface switch and the axes
//!   gizmo. The view is the space the panels leave (no `CentralPanel`: egui would then count the
//!   pointer over the 3D view as over egui, and the mouse tools would stop working).
//!
//! The chrome never edits the scene directly: widgets emit [`Action`]s, and [`actions::apply`]
//! turns them into the same [`Command`]s the old side panel sent (the viewer sends them to the
//! worker and applies them to its mirror). Only view-only settings and the run settings last sent
//! (tool mode, element, volume look, pending grid edits, dt, speed, thermostat, field, file path)
//! live in [`UiState`]. Units: the driver speaks atomic units; the chrome converts at its edge
//! through `units` (fs, K, eV, m/s²) — CLAUDE.md rule 3. Speed changes steps per tick, never dt
//! (rule 8).
pub mod actions;
mod dock;
mod icons;
pub mod logbook;
mod menus;
mod overlays;
pub mod ptable;
mod sampling_presets;
pub mod search;
pub mod theme;
mod toolbar;

pub use actions::Action;
pub use theme::apply as apply_theme;

use crate::scene::Scene;
use crate::sim::{Backend, Command, ModelStatus, Snapshot, DEFAULT_DT_FS};
use egui::Color32;
use logbook::{LogBook, LogFilter};
use render::camera::OrbitCamera;
use sim::dynamics::{F_DRAG_HI, F_DRAG_LO, F_DRAG_MAX, MAX_SPEED};
use sim::{DynCommand, SamplingPlan, Source, Walls};
use std::time::{Duration, Instant};

pub const RED: Color32 = Color32::from_rgb(255, 90, 90);
pub const AMBER: Color32 = Color32::from_rgb(240, 200, 60);
pub const GREEN: Color32 = Color32::from_rgb(110, 210, 120);
/// Sparkline colours: E_total (also the DFT source colour) and T.
pub const SPARK_E: Color32 = Color32::from_rgb(110, 180, 255);
pub const SPARK_T: Color32 = Color32::from_rgb(255, 150, 80);
/// How long the first click on "Forget model" stays armed for the confirming second click.
const FORGET_CONFIRM: Duration = Duration::from_secs(4);
/// Below this width (points) the toolbar and the top bar switch to their compact layouts.
const COMPACT_WIDTH: f32 = 1240.0;

/// Who answered a step, as a colour: surrogate green, fallback amber, DFT blue (the sparkline's
/// E colour, so a Kohn-Sham run looks as before).
pub fn source_colour(s: Source) -> Color32 {
    match s {
        Source::Surrogate => GREEN,
        Source::Fallback => AMBER,
        Source::Dft => SPARK_E,
    }
}

pub fn source_label(s: Source) -> &'static str {
    match s {
        Source::Surrogate => "surrogate",
        Source::Fallback => "fallback (DFT)",
        Source::Dft => "DFT",
    }
}

/// The status chip's state on a scene with no nuclei (`sim::dynamics`, "The empty scene": Play is
/// accepted, time does not advance).
pub const EMPTY_STATUS: &str = "empty scene — place atoms to start";

/// The Model tab's sampling row on an empty scene (`sim::sampling::EMPTY_SCENE` refuses a pass).
pub const EMPTY_SAMPLE: &str = "place atoms to sample";

/// The kinetic temperature with `decimals` places, or "—" with no nuclei (undefined: nothing
/// moves; the driver's 0 K is only a finite placeholder).
pub fn temperature_text(d: &sim::DynSnapshot, decimals: usize, units: TempUnits) -> String {
    if d.scene.nuclei.is_empty() {
        "—".into()
    } else {
        units.show(d.temperature, decimals)
    }
}

/// The source tooltip (the old "this step" readout's).
const SOURCE_TIP: &str = "surrogate (green): the model answered; fallback (amber): the model was unsure or untrusted and the DFT teacher answered — its answer becomes a training sample; DFT (blue): a plain Kohn-Sham step";

/// The sparkline shows the newest this many history points (100 fs at dt = 0.25 fs: about ten
/// H–H periods stay legible; the whole ≤ 2000-point history is what the drift fit uses).
const SPARK_WINDOW: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    Orbit,
    Place,
    Select,
}

/// The dock's tabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockTab {
    Readouts,
    Graph,
    Model,
    Selection,
    Log,
}

impl DockTab {
    pub const ALL: [DockTab; 5] = [DockTab::Readouts, DockTab::Graph, DockTab::Model, DockTab::Selection, DockTab::Log];

    pub fn title(self) -> &'static str {
        match self {
            DockTab::Readouts => "Readouts",
            DockTab::Graph => "Graph",
            DockTab::Model => "Model · estimated solver",
            DockTab::Selection => "Selection",
            DockTab::Log => "Log",
        }
    }
}

/// Energies in the Readouts tab: Hartree, eV, or both (the old panel showed both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyUnits {
    Ha,
    Ev,
    Both,
}

/// Temperatures on screen: kelvin, Celsius or Fahrenheit (user request 2026-10-14). Display only
/// (rule 3): the driver, the thermostat and the sampling plan stay in kelvin; values convert at
/// this edge through `units`, and typed ones are clamped at absolute zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TempUnits {
    K,
    C,
    F,
}

impl TempUnits {
    /// A kelvin value in this scale.
    pub fn from_k(self, t: f64) -> f64 {
        match self {
            TempUnits::K => t,
            TempUnits::C => units::kelvin_to_celsius(t),
            TempUnits::F => units::kelvin_to_fahrenheit(t),
        }
    }

    /// A value in this scale, in kelvin, clamped at absolute zero.
    pub fn to_k(self, x: f64) -> f64 {
        let t = match self {
            TempUnits::K => x,
            TempUnits::C => units::celsius_to_kelvin(x),
            TempUnits::F => units::fahrenheit_to_kelvin(x),
        };
        t.max(0.0)
    }

    /// " K", " °C", " °F".
    pub fn suffix(self) -> &'static str {
        match self {
            TempUnits::K => " K",
            TempUnits::C => " °C",
            TempUnits::F => " °F",
        }
    }

    /// `t` (K) shown with `decimals` places and the unit.
    pub fn show(self, t: f64, decimals: usize) -> String {
        format!("{:.*}{}", decimals, self.from_k(t), self.suffix())
    }

    /// The edit range in this scale for a kelvin range (`lo` ≥ 0: absolute zero at most).
    pub fn range(self, lo_k: f64, hi_k: f64) -> std::ops::RangeInclusive<f64> {
        self.from_k(lo_k.max(0.0))..=self.from_k(hi_k)
    }
}

/// The Graph tab's curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphSeries {
    Both,
    Energy,
    Temperature,
}

/// Help windows: the controls, and the "Physics, honestly" topics (long-form text).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HelpTopic {
    Controls,
    Model,
    Estimated,
    Gravity,
    AtomCard,
    FastPlay,
}

/// Standard gravity g = 9.8 m/s² as an acceleration in atomic units (Bohr per atomic time unit²):
/// 9.8 m/s² = 9.8e10 Å/s² → Bohr/s² (`units::angstrom_to_bohr`), times the square of one atomic
/// time unit in seconds (`units::au_time_to_fs(1) · 1e-15 s`) ≈ 1.08e-22 Bohr/a.u.². Over a
/// picosecond that moves a nucleus by ½ g t² ≈ 9e-12 Bohr — invisible at atomic scale, which is
/// the honest answer (spec: gravity exists as a hook, not as a visible effect).
pub fn gravity_au() -> f64 {
    let au_time_s = units::au_time_to_fs(1.0) * 1e-15;
    units::angstrom_to_bohr(9.8 * 1e10) * au_time_s * au_time_s
}

pub struct UiState {
    /// Z placed by the Place tool.
    pub element: u8,
    /// Its isotope's mass number (`None` = the most abundant one; the periodic table's isotope row).
    pub isotope: Option<u16>,
    /// View ▸ Bonds and molecules (M6): bond lines, molecule labels in the view, the composition
    /// badge on hover. Display only; on by default.
    pub show_bonds: bool,
    pub show_labels: bool,
    pub show_badge: bool,
    /// How molecules are named (View ▸ Bonds and molecules ▸ name style; the formula always shows).
    /// A viewer preference for the session (a profile setting once user data exists).
    pub name_style: crate::names::NameStyle,
    /// The periodic table's search text.
    pub ptable_query: String,
    pub mode: ToolMode,
    pub preset: String,
    /// Pending grid edit, sent on "Apply grid".
    pub grid_n: usize,
    pub grid_h: f64,
    /// The quality the presets are loaded at and the Custom slider's value: h = σ (Bohr,
    /// `crate::quality`; Accurate 0.15 by default). A scene file keeps its own grid.
    pub quality_h: f64,
    /// The Custom slider's value while it is dragged: sent as `Action::Quality` on release, and
    /// set back to [`quality_h`](Self::quality_h) when that is refused (the GPU memory guard).
    pub quality_slider: f64,
    /// The box length a quality change keeps (Bohr): the preset's or the loaded file's, not the
    /// rounded-up box of the current grid (so repeated changes never let it creep).
    pub box_len: f64,
    /// The latest DFT solve seen: (n, seconds) — the quality levels' cost estimates scale it.
    pub last_solve: Option<(usize, f64)>,
    /// 0 = volume, 1 = isosurface (the `VolumeUniform::mode` values).
    pub volume_mode: u32,
    pub gamma: f32,
    pub opacity: f32,
    /// Isosurface threshold as a fraction of the density maximum.
    pub iso: f32,
    pub scene_path: String,
    /// Neutral feedback (saved, loaded, paused for an edit); errors go to `PanelOutput::error`.
    /// Shown in the status chip until replaced or cleared, as a toast, and kept in the log. Set it
    /// with [`UiState::set_status`] so a repeated message gets fresh feedback.
    pub status: String,
    /// Bumped by every [`UiState::set_status`] (the log and the toast key on it, not on the text).
    pub status_seq: u64,
    /// The backend the user picked; the viewer re-spawns the worker when it differs from the
    /// running one (`Viewer::run_ui`).
    pub backend: Backend,
    /// A GPU context exists (else the GPU choice is disabled).
    pub gpu_available: bool,
    /// Display aids (`overlay::build`): the simulation cell, a floor grid in the xz plane, xyz
    /// axes, and (Place mode) the plane the next click lands on; grid spacing 1 Bohr or 1 Å.
    pub show_cell: bool,
    pub show_floor: bool,
    pub show_axes: bool,
    pub show_place_grid: bool,
    pub grid_in_angstrom: bool,
    /// Run settings as last sent (re-sent to a new worker, `run_settings`): time step (fs),
    /// steps per tick, thermostat on / target (K) / τ (fs), gravity preset, COM correction.
    pub dt_fs: f64,
    pub speed: u32,
    pub thermostat_on: bool,
    pub thermostat_t: f64,
    pub thermostat_tau: f64,
    pub gravity: bool,
    pub com_correction: bool,
    /// Container walls (`sim::Walls`, `DynCommand::SetWalls`): on, and their half-width (Bohr).
    /// **On by default in the app** (user request 2026-10-14) at the slider's default
    /// half-width, [`default_wall_half_width`] (1.5 Bohr inside the cell's face: 4.5 Bohr for the
    /// presets' 49³ × 0.25 cell), so every fresh launch, preset and backend switch starts confined
    /// — the walls act in the integrator's tool forces, on every solver including the estimated
    /// one. The driver's own default stays off (`sim::Dynamics`: the validate suites test
    /// unconfined physics); `--no-walls` turns them off at start-up.
    pub walls_on: bool,
    pub wall_half_width: f64,
    /// The estimated solver's next sampling pass (the seed advances after each Start, so
    /// "sample more" draws new geometries).
    pub plan: SamplingPlan,
    /// The first click on "Forget model" (a second within [`FORGET_CONFIRM`] forgets).
    pub forget_armed: Option<Instant>,
    /// The first click on a sampling button whose pass would take longer than
    /// `sim::PASS_TARGET_S` (a second within [`FORGET_CONFIRM`] starts it: "Start anyway").
    pub start_armed: Option<Instant>,
    /// Draw the nuclei between the last two snapshots (`viewer::interpolate`).
    pub smooth_motion: bool,
    /// The estimated solver's cloud (`Command::SetCloud`, `crate::cloud`): the atomic
    /// superposition between DFT densities (default on), and a display refresh every N s.
    pub cloud_superposition: bool,
    pub cloud_refresh: bool,
    pub cloud_refresh_s: f64,
    /// The surrogate's DFT audit cadence (`FallbackPolicy::default`: 2000, ruling S14's interim;
    /// `Command::SetAudit`).
    pub audit_every: usize,
    /// Draft mode (`Command::SetDraft`): per session, off by default.
    pub draft: bool,
    /// The drag spring's force cap ("pull", Ha/Bohr; `DynCommand::SetDragForce`): the wheel while
    /// dragging scales it by [`DRAG_WHEEL_STEP`] per line within the driver's range. Per session;
    /// it outlives the drag (the next one uses it).
    pub drag_force: f64,
    /// Actions from outside the chrome (the 3D view's wheel during a drag, `Viewer::wheel`),
    /// applied at the start of the next frame like a click's.
    pub queued: Vec<Action>,
    // ---- the chrome's own view state (M6 redesign) ----
    /// The dock's tab, and whether the dock is open (else only its icon strip shows).
    pub dock_tab: DockTab,
    pub dock_open: bool,
    /// Energies in Ha, eV or both (Readouts).
    pub energy_units: EnergyUnits,
    /// Temperatures on screen (Physics ▸ Thermostat's K / °C / °F; display only).
    pub temp_units: TempUnits,
    /// The atom card in the 3D view (Selection tab toggle).
    pub show_atom_card: bool,
    /// The Graph tab's curves and window (the newest [`SPARK_WINDOW`] points, or all ≤ 2000).
    pub graph_series: GraphSeries,
    pub graph_all: bool,
    /// The session log (Log tab, toasts, the strip's counter) and its filter.
    pub log: LogBook,
    pub log_filter: LogFilter,
    /// The Ctrl K control search, and the control it last revealed.
    pub search: search::SearchState,
    /// The open Help window.
    pub help: Option<HelpTopic>,
}

impl UiState {
    pub fn new(preset: &str, scene: &Scene) -> UiState {
        UiState {
            backend: Backend::Cpu,
            gpu_available: false,
            element: 1,
            isotope: None,
            show_bonds: true,
            show_labels: true,
            show_badge: true,
            name_style: Default::default(),
            ptable_query: String::new(),
            mode: ToolMode::Orbit,
            preset: preset.to_string(),
            grid_n: scene.grid_n,
            grid_h: scene.grid_h,
            quality_h: crate::quality::DEFAULT_H,
            quality_slider: crate::quality::DEFAULT_H,
            box_len: crate::quality::box_len(scene),
            last_solve: None,
            volume_mode: 0,
            gamma: 0.5,
            opacity: 2.0,
            iso: 0.1,
            scene_path: "my.scene".into(),
            status: String::new(),
            status_seq: 0,
            show_cell: true,
            show_floor: true,
            show_axes: true,
            show_place_grid: true,
            grid_in_angstrom: false,
            dt_fs: DEFAULT_DT_FS,
            speed: 1,
            thermostat_on: false,
            thermostat_t: 300.0,
            thermostat_tau: 10.0,
            gravity: false,
            // the grid's egg-box net force (~3e-4 Ha/Bohr at h = 0.25) would otherwise make
            // molecules drift (M3 handoff)
            com_correction: true,
            walls_on: true,
            wall_half_width: default_wall_half_width(scene),
            plan: SamplingPlan::default(),
            forget_armed: None,
            start_armed: None,
            audit_every: providers::FallbackPolicy::default().audit_every,
            draft: false,
            drag_force: F_DRAG_MAX,
            queued: Vec::new(),
            smooth_motion: true,
            cloud_superposition: true,
            cloud_refresh: false,
            cloud_refresh_s: 2.0,
            dock_tab: DockTab::Readouts,
            dock_open: true,
            energy_units: EnergyUnits::Both,
            temp_units: TempUnits::K,
            show_atom_card: true,
            graph_series: GraphSeries::Both,
            graph_all: false,
            log: LogBook::default(),
            log_filter: LogFilter::All,
            search: search::SearchState::default(),
            help: None,
        }
    }

    /// The cloud settings as the worker command (`sim` module doc, "The cloud").
    pub fn cloud_command(&self) -> Command {
        Command::SetCloud {
            superposition: self.cloud_superposition,
            refresh_s: self.cloud_refresh.then_some(self.cloud_refresh_s),
        }
    }

    /// Sets the status line; a repeat of the same text still counts as a new message (toast, log).
    pub fn set_status(&mut self, s: impl Into<String>) {
        self.status = s.into();
        self.status_seq += 1;
    }

    /// Resets the pending edits after the whole scene was replaced (preset or file load).
    /// Takes a new scene's grid as the pending grid and its box as the one quality changes keep.
    pub(crate) fn adopt(&mut self, scene: &Scene) {
        self.grid_n = scene.grid_n;
        self.grid_h = scene.grid_h;
        self.box_len = crate::quality::box_len(scene);
    }

    /// The uniform acceleration the field controls stand for (a.u.): g along −y, or none.
    pub fn field(&self) -> [f64; 3] {
        if self.gravity {
            [0.0, -gravity_au(), 0.0]
        } else {
            [0.0; 3]
        }
    }

    fn thermostat(&self) -> Option<(f64, f64)> {
        self.thermostat_on
            .then_some((self.thermostat_t, self.thermostat_tau))
    }

    /// The commands that put a fresh worker's driver into the panel's run settings
    /// (`SimWorker::spawn_with` applies them before the first snapshot).
    pub fn run_settings(&self) -> Vec<DynCommand> {
        let mut v = vec![
            DynCommand::SetSpeed(self.speed),
            DynCommand::SetComCorrection(self.com_correction),
        ];
        if self.dt_fs != DEFAULT_DT_FS {
            v.push(DynCommand::SetDt(self.dt_fs));
        }
        if self.gravity {
            v.push(DynCommand::SetExternalField(self.field()));
        }
        if let Some(t) = self.thermostat() {
            v.push(DynCommand::SetThermostat(Some(t)));
        }
        if let Some(w) = self.walls() {
            v.push(DynCommand::SetWalls(Some(w)));
        }
        if self.drag_force != F_DRAG_MAX {
            v.push(DynCommand::SetDragForce { f_max: self.drag_force });
        }
        v
    }

    /// The walls the panel stands for (`None` when off).
    pub fn walls(&self) -> Option<Walls> {
        self.walls_on.then(|| Walls::new(self.wall_half_width))
    }
}

/// The largest container the scene's cell allows (`sim::Walls::max_half_width`, Bohr).
pub fn max_wall_half_width(scene: &Scene) -> f64 {
    Walls::max_half_width(&sim::Cell { n: scene.grid_n, h: scene.grid_h })
}

/// The container's default half-width (walls are on by default: [`UiState::walls_on`]):
/// [`BOX_MARGIN_WARN`] inside the cell's face (so a nucleus at the wall raises no "density cut off"
/// warning), within the allowed range — 4.5 Bohr for a 49³ × 0.25 Bohr cell (half-width 6, the
/// slider's maximum 5.125).
pub fn default_wall_half_width(scene: &Scene) -> f64 {
    let half = (scene.grid_n as f64 - 1.0) * scene.grid_h / 2.0;
    (half - BOX_MARGIN_WARN).clamp(MIN_WALL_HALF_WIDTH.min(max_wall_half_width(scene)), max_wall_half_width(scene).max(MIN_WALL_HALF_WIDTH))
}

/// The wheel's step for the drag pull: × this per wheel line up, ÷ per line down (a UI choice:
/// 10.3 lines from the default 0.1 to either end of 0.01–1 Ha/Bohr; `validate/references.md`,
/// "Drag force (wheel)").
pub const DRAG_WHEEL_STEP: f64 = 1.25;

/// The pull after `lines` wheel lines (geometric, clamped to the driver's range).
pub fn wheel_pull(f: f64, lines: f32) -> f64 {
    (f * DRAG_WHEEL_STEP.powf(lines as f64)).clamp(F_DRAG_LO, F_DRAG_HI)
}

/// "pull 0.160 Ha/Bohr · 8.23 eV/Å" (the cursor label and the Physics menu's line).
pub fn pull_label(f: f64) -> String {
    format!("pull {} Ha/Bohr · {} eV/Å", sig3(f), sig3(units::hartree_to_ev(f) / units::bohr_to_angstrom(1.0)))
}

/// Three significant figures (the wheel's ×1.25 steps stay distinct at the bottom of the range:
/// 0.0100, 0.0125, 0.0156 — `{:.2}` showed 0.01, 0.01, 0.02; review M5).
pub fn sig3(x: f64) -> String {
    if !(x.is_finite() && x > 0.0) {
        return format!("{x}");
    }
    let decimals = (2 - x.log10().floor() as i32).max(0) as usize;
    format!("{x:.decimals$}")
}

/// The panel's smallest container (Bohr): the walls' soft layer is 1 Bohr thick, so below this
/// the box is mostly wall (`sim::dynamics::WALL_LAYER`).
pub const MIN_WALL_HALF_WIDTH: f64 = 1.5;

/// What the chrome reads besides its own state.
pub struct PanelInput<'a> {
    pub scene: &'a Scene,
    pub snapshot: Option<&'a Snapshot>,
    pub busy: bool,
    pub playing: bool,
    pub last_error: Option<&'a str>,
    pub selected: Option<usize>,
    /// Simulated fs per wall second over the last couple of seconds of play (the viewer's).
    pub measured_fs_per_s: Option<f64>,
    /// The camera (the axes gizmo).
    pub camera: &'a OrbitCamera,
    /// The label over a faded density (`viewer::DensityShown::label`).
    pub density_label: Option<String>,
    /// Draft mode's red "extrapolating" banner (`viewer::Viewer::draft_banner`), drawn over the
    /// 3D view while some atom is beyond what the model has learned.
    pub draft_banner: Option<String>,
    /// The faded bonds' label (`viewer::bonds_label`: "bonds from the last DFT step (t = … fs) —
    /// the surrogate has no electrons"), under the density's.
    pub bonds_label: Option<String>,
}

impl PanelInput<'_> {
    /// The estimated solver's model state, when the snapshot is the estimated solver's.
    fn model(&self) -> Option<&ModelStatus> {
        self.snapshot.filter(|s| s.backend == Backend::Estimated).and_then(|s| s.model.as_ref())
    }

    /// A sampling pass is running.
    fn sampling(&self) -> bool {
        self.snapshot.is_some_and(|s| crate::sim::sampling_active(&s.dynamics))
    }

    /// A relax is running (the driver's readout has no outcome yet).
    fn relaxing(&self) -> bool {
        self.snapshot.is_some_and(|s| relaxing(&s.dynamics))
    }

    /// The selected nucleus, if the index is in the scene.
    fn selection(&self) -> Option<(usize, &crate::scene::SceneNucleus)> {
        self.selected.and_then(|i| self.scene.nuclei.get(i).map(|n| (i, n)))
    }

    /// Whether nucleus `i` is pinned (`None` before the worker reports it).
    fn pinned(&self, i: usize) -> Option<bool> {
        self.snapshot.and_then(|s| s.dynamics.pins.get(i).copied())
    }
}

/// What the user's edits in one frame imply; the viewer acts on it.
#[derive(Default)]
pub struct PanelOutput {
    pub commands: Vec<Command>,
    pub focus_selected: bool,
    pub toggle_play: bool,
    /// Esc: drop the selection.
    pub deselect: bool,
    /// A UI-side failure (invalid grid, unreadable file) to show in red.
    pub error: Option<String>,
    /// The 3D view: the rect (points) the top bar, toolbar and dock leave. The viewer turns it into
    /// the frame's `render::camera::Viewport`.
    pub view: Option<egui::Rect>,
    /// The actions the chrome's own widgets and keys emitted this frame, in order (not the ones
    /// queued from the 3D view, which the recorder already has as input events): what
    /// `--record` writes as `action` lines (`crate::replay::Recorder`).
    pub actions: Vec<Action>,
}

/// Draws the whole chrome into the root `ui` (egui 0.36 panels attach to a `Ui`) and returns
/// what the frame's input implies. Replaces the old `side_panel`.
pub fn show(root: &mut egui::Ui, st: &mut UiState, input: &PanelInput<'_>) -> PanelOutput {
    let mut out = PanelOutput::default();
    // the 3D view's actions first (they happened before this frame's widgets)
    let mut acts: Vec<Action> = std::mem::take(&mut st.queued);
    let queued = acts.len();
    st.log.record(&st.status, st.status_seq, input);
    // the latest DFT solve's cost (the quality levels' estimates)
    if let Some(d) = input.snapshot.map(|s| &s.dynamics).filter(|d| !d.forces_stale && d.eval.scf_steps > 0 && d.eval.solve_ms > 0) {
        st.last_solve = Some((d.scene.cell.n, d.eval.solve_ms as f64 / 1e3));
    }
    keys(root.ctx(), st, &mut acts);
    let compact = root.available_width() < COMPACT_WIDTH;
    menus::top_bar(root, st, input, compact, &mut acts);
    toolbar::show(root, st, input, compact, &mut acts);
    dock::show(root, st, input, &mut acts);
    let view = root.available_rect_before_wrap();
    out.view = Some(view);
    overlays::show(root.ctx(), view, st, input, &mut acts);
    search::show(root.ctx(), st, &mut acts);
    menus::help_window(root.ctx(), st);
    out.actions = acts[queued..].to_vec();
    for a in acts {
        actions::apply(st, input, a, &mut out);
    }
    out
}

#[cfg(test)]
thread_local! {
    /// Test hook: the rect of each probed widget in the last frame (`probe`), so the headless click
    /// tests can press the real widget.
    pub(crate) static PROBES: std::cell::RefCell<std::collections::HashMap<String, egui::Rect>> = Default::default();
}

/// Records a widget's rect under `name` for the click tests (a no-op outside tests).
#[inline]
fn probe(name: &str, r: &egui::Response) {
    #[cfg(test)]
    PROBES.with(|p| {
        p.borrow_mut().insert(name.to_string(), r.rect);
    });
    #[cfg(not(test))]
    let _ = (name, r);
}

/// [`probe`] in expression position: records and passes the response on.
#[inline]
fn probed(name: &str, r: egui::Response) -> egui::Response {
    probe(name, &r);
    r
}

/// The status line when Relax is pressed during a sampling pass (the driver refuses it too).
pub const RELAX_REFUSED_SAMPLING: &str = "relax: a sampling pass is running — let it finish (or Cancel it) first";

/// The Relax button's tooltip.
pub const RELAX_TIP: &str = "relax the geometry: FIRE (Bitzek et al. 2006) moves the nuclei downhill on this solver's surface until every atom's force is below 1e-3 Ha/Bohr; the dynamics pause while it runs, and its energy drop is booked as work done by tools. Under the estimated solver it relaxes on the model, then checks the result with one DFT teacher solve. Pins hold; a drag holds its nucleus against the spring.";

/// A relax is running: the driver's readout has no outcome yet.
pub fn relaxing(d: &sim::DynSnapshot) -> bool {
    d.relax.as_ref().is_some_and(|r| r.converged.is_none())
}

/// The relax readout (Physics menu, the relax card): the progress or result line, the teacher's
/// verification under the estimated solver, and the driver's note (why it ended early).
/// "relaxing: step 42 · max|F| 3.1e-3 Ha/Bohr (stop < 1e-3) · ΔE −2.10e-3 Ha · Δt 0.60 fs".
pub fn relax_lines(r: &sim::RelaxProgress) -> Vec<String> {
    let de = r.energy - r.energy_start;
    let forces = if (r.max_force_augmented - r.max_force).abs() > 1e-12 {
        format!("max|F| {:.1e} Ha/Bohr (with tools {:.1e})", r.max_force, r.max_force_augmented)
    } else {
        format!("max|F| {:.1e} Ha/Bohr", r.max_force)
    };
    let mut v = vec![match r.converged {
        None => format!("relaxing: step {} · {forces} (stop < {:.0e}) · ΔE {de:+.2e} Ha · Δt {:.2} fs", r.steps, sim::F_TOL, r.dt_fs),
        Some(true) => format!("relaxed in {} steps: {forces} (< {:.0e}) · ΔE {de:+.2e} Ha", r.steps, sim::F_TOL),
        Some(false) => format!("relax ended after {} steps: {forces} · ΔE {de:+.2e} Ha", r.steps),
    }];
    if let Some(vf) = r.verified_max_force {
        v.push(format!("verified by one teacher solve: DFT max|F| {vf:.1e} Ha/Bohr"));
    }
    if let Some(n) = &r.note {
        v.push(n.clone());
    }
    v
}

/// `text` (a formula, a name, a badge) as a layout job whose superscripts — mass numbers and ion
/// charges, `¹⁶O`, `H₃⁺`, `O²⁻` — are drawn as small raised text: egui's default proportional font
/// has the superscript digits but not the signs ⁺ ⁻ (U+207A/B), which would draw as a missing-glyph
/// box, and a raised run keeps "²⁺" in one style. The names keep the proper characters
/// (`names::formula`); only what is drawn goes through this. Subscripts are in the font and stay.
pub fn rich(text: &str, size: f32, colour: Color32) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let normal = TextFormat { font_id: egui::FontId::proportional(size), color: colour, ..Default::default() };
    let raised = TextFormat { font_id: egui::FontId::proportional(size * 0.68), color: colour, valign: egui::Align::TOP, ..Default::default() };
    let sup = |c: char| match c {
        '⁰' => Some('0'),
        '¹' => Some('1'),
        '²' => Some('2'),
        '³' => Some('3'),
        '⁴'..='⁹' => char::from_digit(c as u32 - '⁴' as u32 + 4, 10),
        '⁺' => Some('+'),
        '⁻' => Some('−'),
        _ => None,
    };
    let mut job = LayoutJob::default();
    let mut run = String::new();
    let mut run_raised = false;
    for c in text.chars() {
        let (ch, up) = match sup(c) {
            Some(d) => (d, true),
            None => (c, false),
        };
        if up != run_raised && !run.is_empty() {
            job.append(&run, 0.0, if run_raised { raised.clone() } else { normal.clone() });
            run.clear();
        }
        run_raised = up;
        run.push(ch);
    }
    if !run.is_empty() {
        job.append(&run, 0.0, if run_raised { raised } else { normal });
    }
    job
}

/// The tooltip of Stop (the toolbar, Physics ▸ Relax geometry, the relax card).
pub const STOP_RELAX_TIP: &str = "end the relax now: the nuclei stay where FIRE has brought them, at rest";

/// Relax, or Stop while a relax runs (the toolbar's Time group, Physics ▸ Relax geometry): disabled
/// during a sampling pass. `probes` names the two buttons for the click tests.
fn relax_toggle(ui: &mut egui::Ui, input: &PanelInput<'_>, size: egui::Vec2, (relax, stop): (&str, &str), acts: &mut Vec<Action>) {
    if input.relaxing() {
        if probed(stop, ui.add(egui::Button::new("Stop").min_size(size))).on_hover_text(STOP_RELAX_TIP).clicked() {
            acts.push(Action::StopRelax);
        }
    } else if probed(relax, ui.add_enabled(!input.sampling(), egui::Button::new("Relax").min_size(size)))
        .on_hover_text(RELAX_TIP)
        .on_disabled_hover_text(RELAX_REFUSED_SAMPLING)
        .clicked()
    {
        acts.push(Action::Relax);
    }
}

/// The listed isotopes of Z as selectable buttons with their masses ("²H 2.014102 u"), `current`
/// highlighted; returns the mass number clicked (the periodic table's isotope row, the Selection
/// tab's isotope choice).
fn isotope_buttons(ui: &mut egui::Ui, z: u8, current: Option<u16>, probe_prefix: &str) -> Option<u16> {
    let default = units::most_abundant_a(z);
    let mut picked = None;
    for &a in units::isotopes_of(z) {
        let mass = units::isotope_mass_amu_a(z, a).unwrap_or(f64::NAN);
        let label = rich(&format!("{} {mass:.6} u", units::isotopes::isotope_label(z, a)), 13.0, theme::INK);
        let mut tip = format!("A = {a}: {z} protons, {} neutrons; atomic mass (AME2020)", a.saturating_sub(z as u16));
        if Some(a) == default {
            tip += "; the most abundant isotope (the default)";
        }
        tip += ". Only the mass changes: the forces (the Born-Oppenheimer surface) are the same for every isotope";
        let r = ui.add(egui::Button::selectable(current == Some(a), label)).on_hover_text(tip);
        probe(&format!("{probe_prefix}-{a}"), &r);
        if r.clicked() {
            picked = Some(a);
        }
    }
    picked
}

/// One sampled pair's verdict with its numbers (spec §4.4: Dock ▸ Selection and its tooltips):
/// "O–H: bonded — ρ_b 0.180 e/Bohr³, 1.30× the free atoms' density there, deformation route", or
/// why not.
pub fn pair_line(scene: &Scene, r: &sim::PairReport) -> String {
    use sim::{Route, Verdict};
    let sym = |i: usize| scene.nuclei.get(i).map_or("?", |n| render::elements::symbol(n.z));
    let who = format!("{}–{} (#{}–#{})", sym(r.i), sym(r.j), r.i, r.j);
    let why = match r.verdict {
        Verdict::Bonded { multi_centre } => {
            let route = match r.route {
                Some(Route::Saddle) => "saddle route",
                Some(Route::Deformation) => "deformation route",
                None => "?",
            };
            format!(
                "bonded{} — ρ_b {} e/Bohr³, {}× the free atoms' density there, {route}",
                if multi_centre { " (multi-centre: a ring)" } else { "" },
                r.rho_min.map_or("?".into(), |x| format!("{x:.3}")),
                r.enhancement.map_or("?".into(), |x| format!("{x:.2}"))
            )
        }
        Verdict::TooFar => format!("not a bond — farther apart than {} Bohr", sim::bonds::R_MAX),
        Verdict::NoInteriorMinimum => "not a bond — no density saddle and no deformation maximum between them".into(),
        Verdict::NotATransverseMaximum => "not a bond — the density between them keeps rising sideways, toward a neighbour".into(),
        Verdict::BelowFloor => format!("not a bond — the density between them is below {} e/Bohr³", sim::bonds::RHO_FLOOR),
        Verdict::NotAccumulated => format!(
            "not a bond — no more density between them than two free atoms would put there (×{} < {:.2})",
            r.enhancement.map_or("?".into(), |x| format!("{x:.2}")),
            1.0 + sim::bonds::DELTA
        ),
        Verdict::NeighbourRegion { atom } => format!("not a bond — the feature between them lies in {} #{atom}'s share of space: another bond seen from the side", sym(atom)),
    };
    format!("{who}: {why}")
}

/// The faded bonds' label (spec §3.4) when the graph on screen is an earlier DFT step's.
pub fn bonds_label(d: &sim::DynSnapshot) -> Option<String> {
    let g = d.bonds.as_ref().filter(|_| d.bonds_stale)?;
    Some(format!("bonds from the last DFT step (t = {:.2} fs) — the surrogate has no electrons", units::au_time_to_fs(g.sim_time)))
}

/// Opens the periodic table (the toolbar's element group; the screenshot flag's `ptable=open`).
pub fn open_ptable(ctx: &egui::Context) {
    egui::Popup::open_id(ctx, ptable::popup_id());
}

/// Opens a top-bar menu by its name ("file", "scene", "view", "physics", "help"); `None` for an
/// unknown name (the screenshot flag's `menu=`).
pub fn open_menu(ctx: &egui::Context, name: &str) -> Option<()> {
    let m = menus::MenuId::ALL.into_iter().find(|m| m.title().eq_ignore_ascii_case(name))?;
    egui::Popup::open_id(ctx, menus::popup_id(m));
    Some(())
}

/// Ctrl K opens the control search; Esc deselects when nothing else wants it (a text field, an
/// open menu or the search).
fn keys(ctx: &egui::Context, st: &mut UiState, acts: &mut Vec<Action>) {
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::K)) {
        st.search.toggle();
    }
    if !st.search.open
        && !ctx.egui_wants_keyboard_input()
        && !egui::Popup::is_any_open(ctx)
        && ctx.input(|i| i.key_pressed(egui::Key::Escape))
    {
        acts.push(Action::Deselect);
    }
}

/// "45 s", "3.2 min", "1.4 h".
fn duration_label(s: f64) -> String {
    if s < 90.0 {
        format!("{s:.0} s")
    } else if s < 5400.0 {
        format!("{:.1} min", s / 60.0)
    } else {
        format!("{:.1} h", s / 3600.0)
    }
}

/// One H–X stretch period (≈ 10 fs) shown in about this much wall time by the "watchable" speed.
const WATCHABLE_FS_PER_S: f64 = 10.0 / 0.5;

/// The steps per tick for [`WATCHABLE_FS_PER_S`] (clamped to 1 .. MAX_SPEED − 1: the top of
/// the slider is unpaced) and the fs/s it gives: a paced tick lasts max(16 ms, n · ms per step).
fn watchable_speed(dt_fs: f64, ms_per_step: f64) -> (u32, f64) {
    let tick_s: f64 = 0.016;
    let rate = |n: u32| n as f64 * dt_fs / tick_s.max(n as f64 * ms_per_step * 1e-3);
    let n = ((WATCHABLE_FS_PER_S * tick_s / dt_fs).round() as u32).clamp(1, MAX_SPEED - 1);
    (n, rate(n))
}

/// Drift colour (spec §8): amber above 1e-4 Ha/ps, red above 1e-3; `None` below.
fn drift_colour(drift: f64) -> Option<Color32> {
    match drift.abs() {
        x if x > 1e-3 => Some(RED),
        x if x > 1e-4 => Some(AMBER),
        _ => None,
    }
}

/// Distance (Bohr) from a nucleus to the box face below which the chrome warns: the grid ends
/// there, so the nucleus's density is cut off (a display heuristic — an H 1s density at 1.5 Bohr
/// is e⁻³ ≈ 5 % of its peak — not a physics parameter).
const BOX_MARGIN_WARN: f64 = 1.5;

/// The nucleus nearest to (or beyond) the box face, when it is within [`BOX_MARGIN_WARN`]: the
/// solver's world ends at the box, so forces there are unreliable and a nucleus outside it is
/// invisible to the electrons (nothing stops a molecule with momentum from leaving it). Returns the
/// colour, the message and the nucleus's index.
fn box_warning(scene: &Scene) -> Option<(Color32, String, usize)> {
    let half = (scene.grid_n as f64 - 1.0) * scene.grid_h / 2.0;
    let (i, margin) = scene
        .nuclei
        .iter()
        .enumerate()
        .map(|(i, n)| (i, half - n.pos.iter().fold(0.0f64, |m, x| m.max(x.abs()))))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    if margin < 0.0 {
        Some((RED, format!("nucleus #{i} is outside the box ({:.1} Bohr beyond its face): the electrons cannot reach it, its forces are meaningless", -margin), i))
    } else if margin < BOX_MARGIN_WARN {
        Some((AMBER, format!("nucleus #{i} is {margin:.1} Bohr from the box face: its density is cut off there, forces and energies are unreliable"), i))
    } else {
        None
    }
}

/// Which of the trust readout's five variants applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrustKind {
    NoModel,
    /// Trusted, but it has not answered at this geometry for a while.
    Stuck,
    /// Trusted, with quarantined samples.
    Quarantined,
    Trusted,
    Untrusted,
}

impl TrustKind {
    /// The toolbar's short word.
    fn short(self) -> &'static str {
        match self {
            TrustKind::NoModel => "no model yet",
            TrustKind::Trusted => "trusted",
            TrustKind::Stuck | TrustKind::Quarantined => "trusted, see Model",
            TrustKind::Untrusted => "untrusted, see Model",
        }
    }
}

/// The trust readout (row 13 of the mapping: all five variants keep their text).
struct TrustView {
    kind: TrustKind,
    colour: Color32,
    text: String,
    /// The variant's tooltip.
    tip: Option<&'static str>,
}

fn trust(m: Option<&ModelStatus>) -> TrustView {
    let v = |kind, colour, text: String, tip| TrustView { kind, colour, text, tip };
    let Some(m) = m.filter(|m| m.fitted) else {
        return v(TrustKind::NoModel, Color32::GRAY, "no model yet".into(), None);
    };
    if let Some(why) = m.stuck.as_deref().filter(|_| m.trusted) {
        v(TrustKind::Stuck, AMBER, format!("yes, but {why}"), Some("the model is trusted where it was taught, but it has not answered at this geometry for a while: every step here is DFT. Move on, sample this region, or — when every answer here touches the box wall (box-affected answers are never trained on, so nothing is being learned) — enlarge the grid or turn container walls on."))
    } else if m.trusted && m.quarantined > 0 {
        v(TrustKind::Quarantined, GREEN, format!("yes — {} samples quarantined (the model cannot represent them)", m.quarantined), Some("kept in the store and the model file but excluded from the fit; geometries near them fall back to DFT (ruling S15)"))
    } else if m.trusted {
        v(TrustKind::Trusted, GREEN, "yes".into(), None)
    } else {
        v(TrustKind::Untrusted, AMBER, format!("no — {}", m.untrusted_reason.as_deref().unwrap_or("?")), Some("an untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust (enough samples for its Q′ columns, ≥ 30 residual degrees of freedom, and its own training data reproduced to the accuracy it claims — one stray sample can untrust it)"))
    }
}

/// An energy's two grid cells in the chosen units (Ha and eV; the unused one blank).
fn energy_cells(ui: &mut egui::Ui, x: f64, units: EnergyUnits, strong: bool) {
    let t = |s: String| {
        let r = egui::RichText::new(s).monospace();
        if strong {
            r.strong()
        } else {
            r
        }
    };
    let ev = units::hartree_to_ev(x);
    match units {
        EnergyUnits::Ha => {
            ui.label(t(format!("{x:+.6} Ha")));
            ui.label("");
        }
        EnergyUnits::Ev => {
            ui.label(t(format!("{ev:+.3} eV")));
            ui.label("");
        }
        EnergyUnits::Both => {
            ui.label(t(format!("{x:+.6} Ha")));
            ui.label(t(format!("{ev:+.3} eV")));
        }
    }
}

fn load(path: &str) -> Result<Scene, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("load {path}: {e}"))?;
    let scene = Scene::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    scene.validate().map_err(|e| format!("{path}: {e}"))?;
    Ok(scene)
}

#[cfg(test)]
mod tests;
