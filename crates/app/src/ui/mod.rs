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
pub mod activity;
mod dock;
mod icons;
pub mod logbook;
mod menus;
pub mod model_offer;
mod overlays;
pub mod pseudo;
pub mod ptable;
mod sampling_presets;
pub mod search;
pub mod theme;
mod toolbar;

pub use actions::Action;
#[cfg(test)]
pub use dock::orbitals as orbital_tab;
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

/// The temperature readouts' one-second average (user request 2026-10-18: a few nuclei's
/// instantaneous kinetic T swings with every bond stretch and is unreadable). Display only: the
/// samples are the snapshots' T, one per new step, kept for [`TempAverage::WINDOW_S`] of wall time;
/// a change of the nucleus count or a step going backwards (a reset, a new scene) starts over.
#[derive(Clone, Debug, Default)]
pub struct TempAverage {
    samples: std::collections::VecDeque<(f64, u64, f64)>,
    n_nuclei: usize,
}

impl TempAverage {
    /// The averaging window, wall-clock seconds.
    pub const WINDOW_S: f64 = 1.0;

    /// Records the snapshot's T at wall time `now_s` (one sample per step) and drops samples older
    /// than the window, keeping the newest.
    pub fn observe(&mut self, now_s: f64, d: &sim::DynSnapshot) {
        let n = d.scene.nuclei.len();
        if n != self.n_nuclei || self.samples.back().is_some_and(|&(_, step, _)| d.step < step) {
            self.samples.clear();
            self.n_nuclei = n;
        }
        if self.samples.back().is_none_or(|&(_, step, _)| step != d.step) {
            self.samples.push_back((now_s, d.step, d.temperature));
        }
        while self.samples.len() > 1 && self.samples.front().is_some_and(|&(t, _, _)| now_s - t > Self::WINDOW_S) {
            self.samples.pop_front();
        }
    }

    /// The mean T over the window, if any sample is held.
    pub fn mean(&self) -> Option<f64> {
        (!self.samples.is_empty()).then(|| self.samples.iter().map(|s| s.2).sum::<f64>() / self.samples.len() as f64)
    }
}

/// The readouts' temperature: the one-second average ([`TempAverage`]) when one is held, else the
/// snapshot's instantaneous T; "—" with no nuclei.
pub fn averaged_temperature_text(d: &sim::DynSnapshot, avg: &TempAverage, decimals: usize, units: TempUnits) -> String {
    if d.scene.nuclei.is_empty() {
        "—".into()
    } else {
        units.show(avg.mean().unwrap_or(d.temperature), decimals)
    }
}

/// Wall-clock seconds between audits under Model ▸ Audits ▸ auto.
pub const AUTO_AUDIT_WALL_S: f64 = 10.0;
/// The auto cadence's bounds (answered steps): never more often than every 200 steps, never
/// rarer than every 1 000 000.
pub const AUTO_AUDIT_MIN: usize = 200;
pub const AUTO_AUDIT_MAX: usize = 1_000_000;

/// The auto audit cadence for a measured play speed: the steps that pass in
/// [`AUTO_AUDIT_WALL_S`] of wall time, rounded to two significant figures and clamped. `None`
/// while the speed is not measured (paused, just started).
pub fn auto_audit_every(fs_per_s: Option<f64>, dt_fs: f64) -> Option<usize> {
    let steps_per_s = fs_per_s.filter(|v| v.is_finite() && *v > 0.0)? / dt_fs.max(1e-9);
    let raw = (steps_per_s * AUTO_AUDIT_WALL_S).max(1.0);
    let mag = 10f64.powf(raw.log10().floor() - 1.0);
    Some((((raw / mag).round() * mag) as usize).clamp(AUTO_AUDIT_MIN, AUTO_AUDIT_MAX))
}

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

/// Where the current scene came from (the scene chip, the preset picker, Reset; M6 final review
/// app I1): a preset by name, or a scene file by path (`--scene`, File ▸ Load, a replay's `scene`
/// line — all through [`UiState::adopt_file`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneSource {
    Preset(String),
    File(String),
}

impl SceneSource {
    /// The preset's name; `None` for a file.
    pub fn preset(&self) -> Option<&str> {
        match self {
            SceneSource::Preset(p) => Some(p),
            SceneSource::File(_) => None,
        }
    }

    /// The scene chip's text: the preset's name, or the file's name (`foo.scene`).
    pub fn label(&self) -> String {
        match self {
            SceneSource::Preset(p) => p.clone(),
            SceneSource::File(f) => std::path::Path::new(f).file_name().map_or_else(|| f.clone(), |n| n.to_string_lossy().into_owned()),
        }
    }
}

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
    /// What runs, what the teacher was asked and why, and how the model learns (activity spec §3.3).
    Activity,
    /// The Kohn–Sham orbitals (orbital-viewer spec §3): the `show` selector, the list, the controls.
    Orbitals,
    Selection,
    Log,
}

impl DockTab {
    pub const ALL: [DockTab; 7] = [DockTab::Readouts, DockTab::Graph, DockTab::Model, DockTab::Activity, DockTab::Orbitals, DockTab::Selection, DockTab::Log];

    pub fn title(self) -> &'static str {
        match self {
            DockTab::Readouts => "Readouts",
            DockTab::Graph => "Graph",
            DockTab::Model => "Model · estimated solver",
            DockTab::Activity => "Activity",
            DockTab::Orbitals => "Orbitals",
            DockTab::Selection => "Selection",
            DockTab::Log => "Log",
        }
    }
}

/// `UiState::volume_mode`: the density (or an orbital's |ψ|²) as a glowing volume.
pub const VOLUME_VOLUME: u32 = 0;
/// `UiState::volume_mode`: as an isosurface (for a signed field: the ± isosurface).
pub const VOLUME_ISO: u32 = 1;
/// `UiState::volume_mode`: as probability dots.
pub const VOLUME_DOTS: u32 = 2;

/// View ▸ Density ▸ show (orbital-viewer spec §3): what the 3D view shows. `Density` is today's
/// view and the default; the other three need Kohn–Sham orbitals, so a DFT step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShowMode {
    #[default]
    Density,
    /// One orbital by track, as ± phase or |ψ|², plus the list of occupied and unoccupied ones.
    Orbitals,
    /// Every included orbital in its own colour.
    Coloured,
    /// ρ↑ − ρ↓.
    Spin,
}

impl ShowMode {
    pub const ALL: [ShowMode; 4] = [ShowMode::Density, ShowMode::Orbitals, ShowMode::Coloured, ShowMode::Spin];

    /// The selector's and the search's name for it.
    pub fn title(self) -> &'static str {
        match self {
            ShowMode::Density => "Density",
            ShowMode::Orbitals => "Orbitals",
            ShowMode::Coloured => "All orbitals coloured",
            ShowMode::Spin => "Spin density",
        }
    }

    /// The word a replay script and `--ui show=` use.
    pub fn word(self) -> &'static str {
        match self {
            ShowMode::Density => "density",
            ShowMode::Orbitals => "orbitals",
            ShowMode::Coloured => "colour",
            ShowMode::Spin => "spin",
        }
    }

    pub fn from_word(w: &str) -> Option<ShowMode> {
        ShowMode::ALL.into_iter().find(|m| m.word() == w)
    }
}

/// `UiState::orbital_pick` before the user has picked a row: a track no producer has issued, so
/// the provider's selection fallback shows the HOMO (and the payload says which).
pub const HOMO_TRACK: u32 = sim::TRACK_HOMO;

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

/// The ionic policy the app starts its presets with (M7, spec §4.7 "default at first"): pseudopotentials for
/// every species with a core, H and He all-electron. `Scene` literals, scene files without an `ionic` key and the
/// validate suites stay `None`; `--ionic` overrides this and a `--scene` file keeps its own key.
pub const DEFAULT_IONIC: sim::IonicPolicy = sim::IonicPolicy::CoresOnly;

pub struct UiState {
    /// The ionic policy the scene in force was set with (Physics ▸ Ionic potential; presets loaded
    /// next are loaded under it). M7.
    pub ionic: sim::IonicPolicy,
    /// View ▸ Density ▸ with partial core: the density shown includes ρ_pc (display only). M7.
    pub show_partial_core: bool,
    /// Z placed by the Place tool.
    pub element: u8,
    /// Its isotope's mass number (`None` = the most abundant one; the periodic table's isotope row).
    pub isotope: Option<u16>,
    /// View ▸ Bonds and molecules (M6): bond lines, molecule labels in the view, the composition
    /// badge on hover. Display only; on by default.
    pub show_bonds: bool,
    pub show_labels: bool,
    pub show_badge: bool,
    /// View ▸ Display aids ▸ "show what training is solving" (`crate::ghost`): while a sampling pass
    /// runs, the 3D view shows the geometry the teacher is solving instead of the scene. Display
    /// only; on by default.
    pub show_ghost: bool,
    /// View ▸ Bonds and molecules ▸ bond angles (M7 Task 13): an arc and a degree label between each
    /// pair of drawn bonds at an atom (`overlay::bond_angles`). Needs "bonds" on; display only; on by
    /// default.
    pub show_angles: bool,
    /// View ▸ Bonds and molecules ▸ bond lengths: each drawn bond's length (Å) at its midpoint
    /// (user request 2026-10-18). Display only, not recorded.
    pub show_bond_lengths: bool,
    /// Dock ▸ Selection ▸ follow: the camera stays centred on the selected nucleus as it moves
    /// (user request 2026-10-18). Display only, not recorded; nothing selected → no effect.
    pub follow_selected: bool,
    /// The audit cadence restored when Model ▸ Audits ▸ "audits on" is ticked again (user request
    /// 2026-10-18: a switch to turn audits off for smoother playback). Display state only.
    pub audit_every_on: usize,
    /// Model ▸ Audits ▸ "auto" (on by default since 2026-10-19, user request): the cadence follows the measured play speed so audits arrive
    /// about every [`AUTO_AUDIT_WALL_S`] of wall time at any speed (user request 2026-10-18).
    pub audit_auto: bool,
    /// How molecules are named (View ▸ Bonds and molecules ▸ name style; the formula always shows).
    /// A viewer preference for the session (a profile setting once user data exists).
    pub name_style: crate::names::NameStyle,
    /// The periodic table's search text.
    pub ptable_query: String,
    pub mode: ToolMode,
    /// The preset or file the scene was loaded from ([`SceneSource`]).
    pub source: SceneSource,
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
    /// [`VOLUME_VOLUME`], [`VOLUME_ISO`] or [`VOLUME_DOTS`].
    pub volume_mode: u32,
    pub gamma: f32,
    pub opacity: f32,
    /// Isosurface threshold as a fraction of the density maximum.
    pub iso: f32,
    /// Probability dots: resample every frame.
    pub shimmer: bool,
    /// With "All orbitals coloured", colour the dots by orbital (else by the density ramp).
    pub dots_by_orbital: bool,
    /// View ▸ Density ▸ show (orbital-viewer spec §3). Everything below is the orbital viewer's
    /// view state; [`UiState::orbital_request`] turns it into what the worker is asked for.
    pub show: ShowMode,
    /// The orbital picked in the list, by channel and track (`None`: not picked yet, the HOMO is
    /// shown and adopted when the payload says which it is).
    pub orbital_pick: Option<(sim::Spin, u32)>,
    /// The pick was adopted from the HOMO placeholder (`Viewer::drive_orbital_view`), not made by
    /// the user: a producer restart that renumbers its tracks is then no vanished selection (no
    /// note; the HOMO is adopted again). Cleared by picking a row.
    pub orbital_pick_adopted: bool,
    /// Orbitals mode: the signed ψ̃ (± phase colours) instead of ψ̃².
    pub orbital_phase: bool,
    /// "show unoccupied": m per channel (0–8, [`UNOCCUPIED_DEFAULT`]).
    pub unoccupied: u8,
    /// The request goes out with no unoccupied orbitals until the first payload of a freshly
    /// switched-on view has arrived (the display eigensolve takes seconds; the list should not
    /// wait for it): the viewer clears this and re-sends (`Viewer::drive_orbital_view`).
    pub unocc_deferred: bool,
    /// Iso level of the ± isosurface (× the field's scale): phase and spin.
    pub phase_iso: f32,
    pub spin_iso: f32,
    /// All orbitals coloured: additive (emission only) or weighted blend; each orbital to its own
    /// peak or to the physical ρ max.
    pub colour_additive: bool,
    pub colour_physical: bool,
    /// All orbitals coloured: the tracks included (`None` = every occupied orbital).
    pub included: Option<std::collections::BTreeSet<u32>>,
    /// Screenshots: the spheres' size factor instead of the automatic one (the automatic one
    /// shrinks them while an orbital product is drawn, so they do not hide it).
    pub nucleus_scale: Option<f32>,
    pub scene_path: String,
    /// The model file chosen in the last Save/Load model dialog (`None`: next to the scene file,
    /// `scene::model_path`). Display/plumbing only.
    pub model_file: Option<std::path::PathBuf>,
    /// A picked model file whose settings differ from the scene's (`model_offer`): the card over the 3D view offers to
    /// apply them and load, or to leave everything as it is. Display state, not a command: nothing reaches the worker until
    /// the user presses Apply.
    pub model_offer: Option<model_offer::ModelOffer>,
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
    /// Atom trails (`crate::trails`, user request 2026-10-17; display only, not recorded): off by
    /// default; how long a point stays, in simulated fs ([`trails::LENGTH_MIN_FS`]…
    /// [`trails::LENGTH_MAX_FS`], logarithmic slider), and the alpha of the newest end (0–1).
    pub show_trails: bool,
    pub trail_fs: f64,
    pub trail_opacity: f32,
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
    /// The temperature readouts' one-second average (display only).
    pub temp_avg: TempAverage,
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
    /// The Activity tab's state: the solve log, counters, fit history, training views, timeline.
    pub activity: activity::ActivityState,
}

impl UiState {
    pub fn new(preset: &str, scene: &Scene) -> UiState {
        UiState {
            ionic: DEFAULT_IONIC,
            show_partial_core: false,
            backend: Backend::Cpu,
            gpu_available: false,
            element: 1,
            isotope: None,
            show_bonds: true,
            show_labels: true,
            show_badge: true,
            show_ghost: true,
            show_angles: true,
            show_bond_lengths: true,
            follow_selected: false,
            audit_every_on: 2000,
            audit_auto: true,
            name_style: Default::default(),
            ptable_query: String::new(),
            mode: ToolMode::Orbit,
            source: SceneSource::Preset(preset.to_string()),
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
            shimmer: true,
            dots_by_orbital: true,
            show: ShowMode::Density,
            orbital_pick: None,
            orbital_pick_adopted: false,
            orbital_phase: true,
            unoccupied: UNOCCUPIED_DEFAULT,
            unocc_deferred: false,
            phase_iso: PHASE_ISO_DEFAULT,
            spin_iso: SPIN_ISO_DEFAULT,
            colour_additive: true,
            colour_physical: false,
            included: None,
            nucleus_scale: None,
            scene_path: "my.scene".into(),
            model_file: None,
            model_offer: None,
            status: String::new(),
            status_seq: 0,
            show_cell: true,
            show_floor: true,
            show_axes: true,
            show_place_grid: true,
            grid_in_angstrom: false,
            show_trails: false,
            trail_fs: crate::trails::LENGTH_DEFAULT_FS,
            trail_opacity: crate::trails::OPACITY_DEFAULT,
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
            temp_avg: TempAverage::default(),
            show_atom_card: true,
            graph_series: GraphSeries::Both,
            graph_all: false,
            log: LogBook::default(),
            log_filter: LogFilter::All,
            search: search::SearchState::default(),
            help: None,
            activity: activity::ActivityState::new(),
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

    /// A scene file was loaded from `path` — the one place `--scene`, File ▸ Load and a replay's
    /// `scene` line record it (M6 final review app I1/I3): the source (the chip names the file,
    /// Reset reloads it), the file path (Save / Load and the model sidecar point at it), the
    /// file's grid and box, the walls' default for its cell, and the file's spacing as the
    /// quality later presets load at (so a recorded `--scene` session and its replay agree).
    pub(crate) fn adopt_file(&mut self, path: &str, scene: &Scene) {
        self.source = SceneSource::File(path.to_string());
        self.scene_path = path.to_string();
        // a file keeps its own policy (no key: all-electron, the physics it was written for)
        self.ionic = scene.ionic;
        self.adopt(scene);
        self.wall_half_width = default_wall_half_width(scene);
        self.quality_h = scene.grid_h;
        self.quality_slider = scene.grid_h;
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
        // the orbital view survives a solver switch (a new worker): without it the view would be
        // "on" in the panel and off in the new driver
        if self.show != ShowMode::Density {
            v.push(DynCommand::SetOrbitalView(self.orbital_request()));
        }
        // likewise the partial-core display
        if self.show_partial_core {
            v.push(DynCommand::SetDisplayPartialCore(true));
        }
        v
    }

    /// The walls the panel stands for (`None` when off).
    pub fn walls(&self) -> Option<Walls> {
        self.walls_on.then(|| Walls::new(self.wall_half_width))
    }
}

/// The sampling preset `i` of the Model tab's row (the screenshot flag's `plan=`).
pub fn sampling_preset(i: usize) -> sampling_presets::SamplingPreset {
    sampling_presets::PRESETS[i]
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

/// "show unoccupied" default (spec §3): m = 4 per channel.
pub const UNOCCUPIED_DEFAULT: u8 = 4;
/// The largest m the slider offers (spec §3; the provider caps it too).
pub const UNOCCUPIED_MAX: u8 = 8;
/// Iso defaults of the ± isosurface, × the field's scale (spec §3).
pub const PHASE_ISO_DEFAULT: f32 = 0.15;
pub const SPIN_ISO_DEFAULT: f32 = 0.1;

impl UiState {
    /// What the worker is asked to show (`Command::OrbitalView`): the panel's `show` mode and its
    /// settings. In dots mode an orbital is drawn as ψ̃² (dots are samples of |ψ|², smooth through
    /// the nodes), not as the signed field.
    pub fn orbital_request(&self) -> sim::OrbitalViewRequest {
        use sim::OrbitalRequest as R;
        let (what, unoccupied) = match self.show {
            ShowMode::Density => (R::Off, 0),
            ShowMode::Orbitals => {
                let (spin, track) = self.orbital_pick.unwrap_or((sim::Spin::Up, HOMO_TRACK));
                (R::Orbital { spin, track, squared: !self.orbital_phase || self.volume_mode == VOLUME_DOTS }, self.unoccupied)
            }
            ShowMode::Coloured => (
                // (the blend is a render setting, `Painter::set_view`, not part of the request: a toggle
                // would otherwise be a view change, and in estimated mode a background solve)
                R::AllColoured { include: self.included.as_ref().map(|s| s.iter().copied().collect()), physical: self.colour_physical },
                self.unoccupied,
            ),
            ShowMode::Spin => (R::SpinDensity, 0),
        };
        sim::OrbitalViewRequest { what, unoccupied: if self.unocc_deferred { 0 } else { unoccupied }, force_every_step: false }
    }
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
    /// The label over the orbitals when they are not the current step's (`viewer::OrbitalShown`:
    /// the estimated solver has no electrons, so they are the last DFT solve's, faded), and the
    /// note when there is nothing to draw ("needs a DFT step", "no electrons — nothing to show",
    /// the CPU teacher's missing producer). Shown over the view while an orbital mode is on.
    pub orbital_label: Option<String>,
    /// An orbital request is out and its payload has not arrived: the display eigensolve may be
    /// running ("computing unoccupied orbitals…", `Viewer::orbital_waiting`).
    pub orbital_waiting: bool,
    /// Draft mode's red "extrapolating" banner (`viewer::Viewer::draft_banner`), drawn over the
    /// 3D view while some atom is beyond what the model has learned.
    pub draft_banner: Option<String>,
    /// The faded bonds' label (`viewer::bonds_label`: "bonds from the last DFT step (t = … fs) —
    /// the surrogate has no electrons"), under the density's.
    pub bonds_label: Option<String>,
    /// The live solve state of the worker (`SimWorker::probe`): the SCF step being solved, a
    /// refit in flight, the background auditor. Read per frame, so a long solve that publishes no
    /// snapshot still moves on screen.
    pub probe: Option<sim::ProbeView>,
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
    // the activity headline (the chip, the bar, the strip's peek): rebuilt at most every 100 ms
    let reference = st.activity.current_reference();
    st.activity.headline.get(Instant::now(), || activity::headline::display(input, reference));
    let compact = root.available_width() < COMPACT_WIDTH;
    menus::top_bar(root, st, input, compact, &mut acts);
    let bar_rect = toolbar::show(root, st, input, compact, &mut acts);
    activity::bar(root, st, input, bar_rect);
    dock::show(root, st, input, &mut acts);
    let view = root.available_rect_before_wrap();
    out.view = Some(view);
    // the training views are built only while shown (activity spec §6)
    let want = st.dock_open && st.dock_tab == DockTab::Activity && st.activity.training_open;
    if want != st.activity.views_sent {
        st.activity.views_sent = want;
        out.commands.push(Command::SetActivityViews(want));
    }
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

/// The caveat shown with the bond strength (user request, M6; `sim::bonds` module doc, "LSDA").
pub const LSDA_STRETCHED_NOTE: &str = "LSDA overbinds stretched contacts (delocalisation error): a weak contact here may be weaker or absent in reality";

/// Spec §5 and plan review Q10: under pseudopotentials the Laplacian of the density at a bond point
/// is evaluated on the valence (+ρ_pc) density, which is smooth by construction inside each r_c — so
/// a bond point there has a kind that is not the all-electron one. (Words, not ∇²: the UI font has
/// no ∇.)
pub const VALENCE_KIND_NOTE: &str = "valence-density kind: this bond point lies inside a pseudopotential's core radius r_c, where the valence density is smooth by construction — the sign of the Laplacian of the density here is not the all-electron one";

/// Whether the bond point of i–j (pos_i + s·(pos_j − pos_i)) lies within r_c,max of a pseudised end
/// atom. `r_c_max_of(z)` is that species' largest core radius (Bohr), `None` for an all-electron
/// species (so a scene with nothing pseudised never has a valence-density kind).
// The Selection tab passes `|z| pseudo::of_species(&pseudised, z).map(|p| p.r_c_max)` over the scene's
// pseudised species (`sim::PseudoSummary`, `pseudo::pseudised`).
pub fn valence_density_kind(pos: &[[f64; 3]], zs: &[u8], i: usize, j: usize, s: f64, r_c_max_of: impl Fn(u8) -> Option<f64>) -> bool {
    let (Some(pi), Some(pj)) = (pos.get(i), pos.get(j)) else { return false };
    let p: [f64; 3] = std::array::from_fn(|k| pi[k] + s * (pj[k] - pi[k]));
    [i, j].into_iter().any(|a| {
        let (Some(&z), Some(c)) = (zs.get(a), pos.get(a)) else { return false };
        r_c_max_of(z).is_some_and(|r_c| (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>().sqrt() < r_c)
    })
}

/// Which surface a bond angle is measured on (Dock ▸ Selection): the angle of a triple whose three
/// species are all pseudised sits on the pseudopotential surface; otherwise some nucleus is still the
/// smeared all-electron one (σ = h) and the angle carries the smearing.
pub fn angle_surface_note(pseudised: &[u8], zs: [u8; 3]) -> &'static str {
    if zs.iter().all(|z| pseudised.contains(z)) {
        "pseudopotential surface"
    } else {
        "includes smeared all-electron nuclei (σ = h): the angle carries the smearing"
    }
}

/// A bond point's QTAIM kind in plain words (`sim::bonds` module doc, "Bond strength").
pub fn kind_words(kind: sim::BondKind) -> &'static str {
    match kind {
        sim::BondKind::Shared => "covalent (shared)",
        sim::BondKind::ClosedShell => "weak / stretched contact (closed-shell)",
        sim::BondKind::Indeterminate => "not resolved on this grid (the Laplacian is within its noise)",
    }
}

/// The strength clause of a pair with a Laplacian: "Laplacian -1.198 e/Bohr⁵: covalent (shared)"
/// (in words: the UI font has no ∇ or arrow glyphs).
pub fn strength_clause(r: &sim::PairReport) -> Option<String> {
    Some(format!("Laplacian {:+.3} e/Bohr⁵: {}", r.laplacian?, kind_words(r.kind()?)))
}

/// One sampled pair's verdict with its numbers (spec §4.4: Dock ▸ Selection and its tooltips):
/// "O–H: bonded — ρ_b 0.180 e/Bohr³, 1.30× the free atoms' density there, deformation route", or
/// why not; then, for a pair with a bond point worth classifying, its strength clause.
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
    match strength_clause(r) {
        Some(c) => format!("{who}: {why}; {c}"),
        None => format!("{who}: {why}"),
    }
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
    /// Trusted overall, but the strict policy refuses at the current geometry (`readiness`).
    NotHere,
    /// Trusted overall, but an environment class among the scene's atoms is untrusted everywhere
    /// (rule 4b; round 1, SHOULD-5).
    ClassUntrusted,
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
            TrustKind::NotHere => "trusted, not here",
            TrustKind::ClassUntrusted => "a class untrusted, see Model",
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
    /// The provider's own technical text for what the plain line says (the old refusal or stuck
    /// string): shown on hover after the tip.
    detail: Option<String>,
}

/// The trust readout in plain words (activity spec §5.3): the first unmet line of the checklist
/// for a refusal, the stuck sentence for a trusted model that keeps falling back; the provider's
/// own text stays on hover (`detail`).
fn trust(m: Option<&ModelStatus>, scene: &Scene, st: &UiState) -> TrustView {
    let v = |kind, colour, text: String, tip| TrustView { kind, colour, text, tip, detail: None };
    let Some(m) = m.filter(|m| m.fitted) else {
        return v(TrustKind::NoModel, Color32::GRAY, "no model yet".into(), None);
    };
    if let Some(why) = m.stuck.as_deref().filter(|_| m.trusted) {
        // the structured form in plain words; the old string when the scene cannot be converted
        let plain = m.stuck_info.zip(scene.to_sim().ok()).map(|(i, s)| activity::words::stuck_sentence(&i, &s));
        let text = match &plain {
            Some(p) => format!("yes, {}", p.strip_prefix("trusted, ").unwrap_or(p)),
            None => format!("yes, but {why}"),
        };
        let view = v(TrustKind::Stuck, AMBER, text, Some("the model is trusted where it was taught, but it has not answered at this geometry for a while: every step here is DFT. Move on, sample this region, or — when every answer here touches the box wall (box-affected answers are never trained on, so nothing is being learned) — enlarge the grid or turn container walls on."));
        TrustView { detail: plain.is_some().then(|| why.to_string()), ..view }
    } else if let Some(r) = m.readiness.as_ref().filter(|r| m.trusted && (!r.answers || r.unlearned.is_some())) {
        // trusted overall, refused here (fix (a), 2026-10-18): the badge must not say "yes" where
        // strict mode falls back on every step — nor for a scene whose own pass is not in the fit (e)
        let class = matches!(r.rule, Some(sim::activity::TeachRule::Class(_)));
        let lead = if class { "yes overall, but" } else { "not here —" };
        let text = match (&r.unlearned, r.answers) {
            (Some(u), true) => format!("not this scene — {u}"),
            (Some(u), false) => format!("{lead} {}; {u}", r.text),
            (None, _) => format!("{lead} {}", r.text),
        };
        let kind = if class && !r.answers { TrustKind::ClassUntrusted } else { TrustKind::NotHere };
        let view = v(kind, AMBER, text, Some(if class { CLASS_UNTRUSTED_TIP } else { NOT_HERE_TIP }));
        TrustView { detail: m.untrusted_reason.clone(), ..view }
    } else if m.trusted && m.quarantined > 0 {
        v(TrustKind::Quarantined, GREEN, format!("yes — {} samples quarantined (the model cannot represent them)", m.quarantined), Some("kept in the store and the model file but excluded from the fit; geometries near them fall back to DFT (ruling S15)"))
    } else if m.trusted {
        v(TrustKind::Trusted, GREEN, "yes".into(), None)
    } else {
        // the first unmet line of the checklist in plain words — the box line first when it is
        // the box that blocks trust (diag-3h) — with the fit's own text on hover
        let tip = if crate::sim::box_blocks_trust(m).is_some() { BOX_TRUST_TIP } else { UNTRUSTED_TIP };
        let reason = m.untrusted_reason.clone();
        match activity::trust::lines(m, scene, st).into_iter().find(|l| l.why) {
            Some(l) => TrustView { detail: reason, ..v(TrustKind::Untrusted, AMBER, format!("no — {}", l.text), Some(tip)) },
            None => v(TrustKind::Untrusted, AMBER, format!("no — {}", reason.as_deref().unwrap_or("?")), Some(tip)),
        }
    }
}

/// The tooltip of the "trusted, not here" variant (`ModelStatus::readiness`).
const NOT_HERE_TIP: &str = "the model passes its global trust rule, but at this geometry strict mode refuses it — every step here is DFT. The reason names the rule. Move on, run a sampling pass here, or switch on draft mode to guess (unsafe).";

/// The tooltip of the class variant (rule 4b; round 1, SHOULD-5).
const CLASS_UNTRUSTED_TIP: &str = "an environment class (two-atom, or three-or-more-atom) among this scene's atoms is not learned well enough: some training sample of that class cannot be predicted by the rest (leave-one-out), so strict mode refuses that class everywhere — every step with such atoms is DFT. The class's verdict changes only when samples near the named one exist (a sampling pass around it); steps elsewhere do not change it.";

/// The tooltip of the plain untrusted variant.
const UNTRUSTED_TIP: &str = "an untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust (enough samples for its Q′ columns, ≥ 30 residual degrees of freedom, and its own training data reproduced to the accuracy it claims — one stray sample can untrust it)";

/// The tooltip of the box variant of "trusted: no" (`crate::sim::box_blocks_trust`).
const BOX_TRUST_TIP: &str = "a box-affected sample is a teacher answer whose density reaches the cell's faces: the box's confinement then depends on where the molecule sits and how it is turned, which a model of the internal geometry cannot learn. Such samples leave the fit once enough others are held (the trust floor, Q′ + 30); until then they are fitted and the model stays untrusted — every step is DFT. A bigger cell is the cure.";

/// The advice under the box variant of "trusted: no" (diag-3h): a cell that puts every nucleus
/// at least d_box from a face, d_box = ln(1 / (π ρ_rule(h))) / 2 — where the hydrogen 1s density
/// e^(−2r)/π falls to the box rule's face density ρ_rule(h)
/// (`providers::box_affected_face_density`; 5.35 Bohr at h = 0.15, 4.84 at 0.25). The H 1s tail
/// is a stated estimate, not a per-element value (LSDA tails and other atoms decay at other
/// rates); the next pass's box-affected count says whether it was enough. For the diag-3h scene
/// (H₂ + H out to x = 3.725 at h = 0.15) it gives 123³ (18.3 Bohr), next to the 121³ (18 Bohr)
/// measured trusted. Not the walls' 1.5 Bohr face margin: that keeps the density warning quiet,
/// not the confinement out of the labels (it would suggest a 10.5 Bohr cell there, smaller than
/// the 12 Bohr one that failed). When the current geometry already fits that box (the samples
/// reached farther than the scene now does), the cell grows by d_box on every side. The GPU
/// memory guard's verdict on the suggestion is part of the advice.
pub fn box_advice(scene: &Scene, st: &UiState) -> String {
    box_advice_with(scene, crate::quality::gpu_solvers(st.backend, st.gpu_available), crate::quality::vram_budget_bytes())
}

/// [`box_advice`] for `solvers` GPU solvers under a memory budget of `budget` bytes.
fn box_advice_with(scene: &Scene, solvers: u64, budget: u64) -> String {
    let h = scene.grid_h;
    // The H 1s tail e^(−2r)/π for every species: it undersizes the cell for diffuse atoms — Li's
    // valence tail decays as e^(−2√(2I) r) with I ≈ 0.198 Ha, which puts d_box near 8.5 Bohr, not
    // 5.35 (M6 final re-review). Follow-up: size d_box per species from the solver's own free-atom
    // radial tables (`sim::references::RadialDensity`, already solved at this grid for the bond
    // detector) — the r where ρ̄_Z(r) falls below the box rule's face density.
    let d_box = activity::words::d_box(h);
    let extent = scene.nuclei.iter().flat_map(|n| n.pos.iter().map(|x| x.abs())).fold(0.0f64, f64::max);
    let now = crate::quality::box_len(scene);
    let want = 2.0 * (extent + d_box);
    let want = if want > now { want } else { now + 2.0 * d_box };
    let s = Scene { grid_n: crate::quality::grid_n(want, h), ..scene.clone() };
    let mut text = format!(
        "Enlarge the cell (Scene ▸ Grid: {}³ at {h} Bohr, a {:.1} Bohr box — every nucleus ≥ {d_box:.1} Bohr from a face, where a hydrogen-like density tail falls below the box rule) and sample again.",
        s.grid_n,
        crate::quality::box_len(&s)
    );
    if s.grid_n > crate::scene::MAX_GRID_N {
        text += &format!(" That is beyond the largest grid ({}³): a coarser spacing at that box.", crate::scene::MAX_GRID_N);
    } else if let Err(e) = crate::quality::vram_check(&s, solvers, budget) {
        text += &format!(" The GPU memory guard would refuse that cell ({e}): a coarser spacing at that box, or fewer atoms.");
    }
    text
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
pub(crate) mod tests;

