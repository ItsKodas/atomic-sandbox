//! The egui side panel: solver choice and the estimated solver's model (sampling pass, trust,
//! uncertainty, model files), time control, dynamics readouts, thermostat and field, scene, tools,
//! electrons, grid, view controls, energy readouts, file I/O.
//!
//! The panel never edits the scene directly: edits become [`Command`]s that the viewer sends to
//! the worker (and applies to its local mirror of the scene). Only view-only settings and the
//! run settings the panel last sent (tool mode, element, volume look, pending grid edits, dt,
//! speed, thermostat, field, file path) live in [`UiState`]. Units: the driver speaks atomic
//! units; this panel converts at its edge through `units` (fs, K, eV, m/s²) — CLAUDE.md rule 3.
use crate::scene::{model_path, validate_grid, Scene};
use crate::sim::{sampling_active, Backend, Command, ModelStatus, Snapshot, DEFAULT_DT_FS};
use egui::{Color32, RichText};
use render::elements;
use sim::dynamics::{DRIFT_MIN_POINTS, DT_WARN_FS, F_DRAG_MAX, MAX_SPEED};
use sim::{DynCommand, SamplingPlan, Source};
use std::time::{Duration, Instant};

const RED: Color32 = Color32::from_rgb(255, 90, 90);
const AMBER: Color32 = Color32::from_rgb(240, 200, 60);
const GREEN: Color32 = Color32::from_rgb(110, 210, 120);
/// Sparkline colours: E_total (also the DFT source colour) and T.
const SPARK_E: Color32 = Color32::from_rgb(110, 180, 255);
const SPARK_T: Color32 = Color32::from_rgb(255, 150, 80);
/// How long the first click on "Forget model" stays armed for the confirming second click.
const FORGET_CONFIRM: Duration = Duration::from_secs(4);

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
/// The sparkline shows the newest this many history points (100 fs at dt = 0.25 fs: about ten
/// H–H periods stay legible; the whole ≤ 2000-point history is what the drift fit uses).
const SPARK_WINDOW: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolMode {
    Orbit,
    Place,
    Select,
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
    pub mode: ToolMode,
    pub preset: String,
    /// Pending grid edit, sent on "Apply grid".
    pub grid_n: usize,
    pub grid_h: f64,
    /// 0 = volume, 1 = isosurface (the `VolumeUniform::mode` values).
    pub volume_mode: u32,
    pub gamma: f32,
    pub opacity: f32,
    /// Isosurface threshold as a fraction of the density maximum.
    pub iso: f32,
    pub scene_path: String,
    /// Neutral feedback (saved, loaded, paused for an edit); errors go to `PanelOutput::error`.
    pub status: String,
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
    /// The surrogate's DFT audit cadence (`FallbackPolicy::default`: 2000, ruling S14's interim;
    /// `Command::SetAudit`).
    pub audit_every: usize,
}

impl UiState {
    pub fn new(preset: &str, scene: &Scene) -> UiState {
        UiState {
            backend: Backend::Cpu,
            gpu_available: false,
            element: 1,
            mode: ToolMode::Orbit,
            preset: preset.to_string(),
            grid_n: scene.grid_n,
            grid_h: scene.grid_h,
            volume_mode: 0,
            gamma: 0.5,
            opacity: 2.0,
            iso: 0.1,
            scene_path: "my.scene".into(),
            status: String::new(),
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
            plan: SamplingPlan::default(),
            forget_armed: None,
            start_armed: None,
            audit_every: sim::FallbackPolicy::default().audit_every,
            smooth_motion: true,
        }
    }

    /// Resets the pending edits after the whole scene was replaced (preset or file load).
    fn adopt(&mut self, scene: &Scene) {
        self.grid_n = scene.grid_n;
        self.grid_h = scene.grid_h;
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
        v
    }
}

/// What the panel reads besides its own state.
pub struct PanelInput<'a> {
    pub scene: &'a Scene,
    pub snapshot: Option<&'a Snapshot>,
    pub busy: bool,
    pub playing: bool,
    pub last_error: Option<&'a str>,
    pub selected: Option<usize>,
    /// Simulated fs per wall second over the last couple of seconds of play (the viewer's).
    pub measured_fs_per_s: Option<f64>,
}

/// What the user's edits in one frame imply; the viewer acts on it.
#[derive(Default)]
pub struct PanelOutput {
    pub commands: Vec<Command>,
    pub focus_selected: bool,
    pub toggle_play: bool,
    /// A UI-side failure (invalid grid, unreadable file) to show in red.
    pub error: Option<String>,
}

/// Draws the side panel into the root `ui` (egui 0.36 panels attach to a `Ui`, not the context).
pub fn side_panel(root: &mut egui::Ui, st: &mut UiState, input: &PanelInput<'_>) -> PanelOutput {
    let mut out = PanelOutput::default();
    egui::Panel::left("side_panel")
        .resizable(true)
        .default_size(330.0)
        .show(root, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Atomic Sandbox");
                status_line(ui, st, input);
                ui.separator();
                backend_row(ui, st);
                if st.backend == Backend::Estimated {
                    estimated_section(ui, st, input, &mut out);
                }
                time_section(ui, st, input, &mut out);
                dynamics_readouts(ui, input.snapshot);
                thermostat_section(ui, st, input.snapshot, &mut out);
                scene_section(ui, st, &mut out);
                tool_section(ui, st, input, &mut out);
                electrons_section(ui, st, input.scene, &mut out);
                view_section(ui, st);
                display_section(ui, st);
                energy_section(ui, input.snapshot);
                let sampling = input.snapshot.is_some_and(|s| sampling_active(&s.dynamics));
                file_section(ui, st, input.scene, sampling, &mut out);
            });
        });
    out
}

fn status_line(ui: &mut egui::Ui, st: &UiState, input: &PanelInput<'_>) {
    let snap = input.snapshot;
    ui.horizontal(|ui| {
        let sampling = snap.and_then(|s| s.dynamics.sampling.as_ref().filter(|_| sampling_active(&s.dynamics)));
        if let Some(p) = sampling {
            ui.spinner();
            ui.label(format!("sampling the teacher: {} {}/{}", p.stage, p.done, p.total));
        } else if input.playing {
            ui.spinner();
            ui.label("playing");
        } else if input.busy {
            ui.spinner();
            ui.label("solving…");
        } else if let Some(s) = snap {
            if s.dynamics.forces_stale {
                ui.label("paused — not evaluated (see the error)");
            } else if s.scene == *input.scene {
                ui.label("paused — converged");
            } else {
                ui.label("paused — density is from an earlier geometry");
            }
        } else {
            ui.label("waiting for the first solve");
        }
    });
    if let Some(e) = input.last_error {
        ui.colored_label(RED, e);
    }
    if let Some(s) = snap {
        if let Some(e) = &s.dynamics.last_error {
            ui.colored_label(RED, format!("paused by a solver failure: {e}"));
        }
        if let Some(r) = &s.dynamics.last_rejection {
            ui.colored_label(AMBER, format!("refused: {r}"));
        }
    }
    if !st.status.is_empty() {
        ui.weak(&st.status);
    }
}

fn backend_row(ui: &mut egui::Ui, st: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("solver");
        ui.radio_value(&mut st.backend, Backend::Cpu, "CPU")
            .on_hover_text("qm::scf, f64 reference; a solve cannot be interrupted (one MD step = one SCF solve: slow)");
        ui.add_enabled_ui(st.gpu_available, |ui| {
            ui.radio_value(&mut st.backend, Backend::Gpu, "GPU")
                .on_hover_text("wgpu compute, f32 grid; Pause and edits abandon the solve in flight")
                .on_disabled_hover_text("no GPU adapter");
        });
        let teacher = if st.gpu_available { "GPU" } else { "CPU" };
        ui.radio_value(&mut st.backend, Backend::Estimated, "estimated")
            .on_hover_text(format!(
                "a surrogate model fitted to this scene's DFT (teacher: the {teacher} solver), with a Bayesian uncertainty: it answers a step in ~10 µs when it is sure and hands the step to DFT when it is not. Without a model a sampling pass runs first."
            ));
    });
}

/// The estimated solver (spec §6): the honesty line, the model's state per snapshot, the
/// sampling pass (plan, progress, Start / Cancel, sample more), and the model files.
fn estimated_section(ui: &mut egui::Ui, st: &mut UiState, input: &PanelInput<'_>, out: &mut PanelOutput) {
    ui.weak("estimated = a model fitted to this scene's DFT; valid only where it has been taught — it falls back to DFT when unsure");
    let snap = input.snapshot.filter(|s| s.backend == Backend::Estimated);
    let d = snap.map(|s| &s.dynamics);
    let m: Option<&ModelStatus> = snap.and_then(|s| s.model.as_ref());
    let sampling = d.is_some_and(sampling_active);
    if let Some(m) = m {
        model_readouts(ui, d.expect("a snapshot"), m);
        if let Some(why) = &m.provenance_mismatch {
            ui.colored_label(AMBER, format!("{why}: every step falls back to DFT and teaches nothing — restore those settings, or forget the model and sample this scene"));
        }
        if let Some(r) = &m.autoload_refused {
            ui.colored_label(AMBER, format!("this session's model was not restored: {r}"));
            if ui.button("Forget it and sample this scene").clicked() {
                out.commands.push(Command::Dyn(DynCommand::StartSampling(st.plan.clone())));
                st.plan.seed += 1;
            }
        }
    }
    // the sampling pass
    if let Some(p) = d.and_then(|d| d.sampling.as_ref()) {
        if sampling {
            let frac = if p.total > 0 { p.done as f32 / p.total as f32 } else { 0.0 };
            let eta = m.and_then(|m| m.pass_eta_s).map_or(String::new(), |t| format!(" — about {} left", duration_label(t)));
            ui.add(egui::ProgressBar::new(frac).show_percentage().text(format!("{} — {}/{}{eta}", p.stage, p.done, p.total)));
            let e = if p.last_energy.is_finite() { format!("{:+.6} Ha", p.last_energy) } else { "—".into() };
            ui.weak(format!("last teacher energy {e} · skipped {} · failed {} — the dynamics are paused; edits are allowed", p.skipped, p.failed));
        } else {
            ui.weak(format!(
                "last pass {}: {}/{} jobs, skipped {}, failed {}",
                if p.cancelled { "cancelled" } else { "finished" },
                p.done,
                p.total,
                p.skipped,
                p.failed
            ));
        }
    }
    if let Some(o) = m.and_then(|m| m.pass_offer.as_ref()).filter(|_| !sampling) {
        ui.colored_label(
            AMBER,
            format!(
                "a first sampling pass here takes about {} ({} solves × ≈ {:.1} s; the panel's plan, {} solves, ≈ {}) — that is already the fewest solves that can earn trust on this scene. A smaller grid is faster.",
                duration_label(o.eta_s),
                o.solves,
                o.per_solve_s,
                o.full_solves,
                duration_label(o.full_eta_s)
            ),
        );
        if ui.button(format!("Start anyway (≈ {})", duration_label(o.eta_s))).clicked() {
            out.commands.push(Command::Dyn(DynCommand::StartSampling(o.plan.clone())));
        }
    }
    let fitted = m.is_some_and(|m| m.fitted);
    ui.horizontal(|ui| {
        ui.label("DFT audit every");
        let mut n = st.audit_every;
        if ui
            .add(egui::DragValue::new(&mut n).range(0..=1_000_000).speed(10.0))
            .on_hover_text("every n-th step the model answers, the DFT teacher checks it at that geometry and the gap is recorded (a failed audit untrusts the model until a refit). 2000 by default. With a GPU teacher the audit runs on a second teacher in the background while the model keeps stepping (ruling S14), so the cadence costs no speed: the check simply arrives a few hundred steps later. With the CPU teacher it runs in the step and sets the speed. 0 = never (unsafe: nothing checks the model while it answers).")
            .changed()
        {
            st.audit_every = n;
            out.commands.push(Command::SetAudit(n));
        }
        ui.label("answered steps");
        // what the worker runs (after a panic recovery it re-applies this value)
        if let Some(m) = m.filter(|m| m.audit_every != st.audit_every) {
            ui.colored_label(AMBER, format!("(solver: {})", m.audit_every));
        }
    });
    if st.audit_every == 0 {
        ui.colored_label(AMBER, "audits off: nothing checks the model against DFT while it answers (unsafe)");
    }
    egui::CollapsingHeader::new("sampling plan")
        .default_open(!fitted)
        .show(ui, |ui| plan_fields(ui, &mut st.plan));
    // the plan's cost on this scene: solves × the expected teacher cost per solve
    let solves = d.and_then(|d| st.plan.solves(&d.scene).ok());
    let eta = solves.zip(m.and_then(|m| m.solve_s)).map(|(n, c)| n as f64 * c);
    ui.horizontal_wrapped(|ui| {
        let label = if fitted || m.is_some_and(|m| m.samples > 0) { "Sample more" } else { "Start sampling" };
        let long = eta.is_some_and(|t| t > crate::sim::PASS_TARGET_S);
        let armed = long && st.start_armed.is_some_and(|t| t.elapsed() < FORGET_CONFIRM);
        let button = if armed {
            egui::Button::new(RichText::new(format!("Start anyway (≈ {})", duration_label(eta.unwrap_or(0.0)))).color(AMBER))
        } else {
            egui::Button::new(label)
        };
        if ui
            .add_enabled(!sampling, button)
            .on_hover_text("one pass over the current scene with the plan above: isolated-atom references, random perturbations, pair scans, a hot MD burst; the samples are added to the model's and it is refitted. Pauses the dynamics. A pass longer than 2 min asks for a second click.")
            .clicked()
        {
            if long && !armed {
                st.start_armed = Some(Instant::now());
            } else {
                st.start_armed = None;
                out.commands.push(Command::Dyn(DynCommand::StartSampling(st.plan.clone())));
                st.plan.seed += 1;
            }
        } else if st.start_armed.is_some() && !armed {
            st.start_armed = None;
        }
        if let (Some(n), false) = (solves, sampling) {
            match eta {
                Some(t) => ui.weak(format!("{n} solves, ≈ {}", duration_label(t))),
                None => ui.weak(format!("{n} solves")),
            };
        }
        if ui
            .add_enabled(sampling, egui::Button::new("Cancel"))
            .on_hover_text("end the pass now: what it sampled is kept and refitted")
            .clicked()
        {
            out.commands.push(Command::Dyn(DynCommand::CancelSampling));
        }
        let armed = st.forget_armed.is_some_and(|t| t.elapsed() < FORGET_CONFIRM);
        let samples = m.map_or(0, |m| m.samples);
        let forget = if armed {
            egui::Button::new(RichText::new(format!("Click again to forget {samples} samples")).color(RED))
        } else {
            egui::Button::new("Forget model")
        };
        if ui
            .add_enabled(m.is_some_and(|m| m.samples > 0 || m.fitted) || sampling, forget)
            .on_hover_text("drop the model, its samples and references (two clicks within 4 s); the teacher answers every step until the next pass")
            .clicked()
        {
            if armed {
                st.forget_armed = None;
                out.commands.push(Command::Dyn(DynCommand::ForgetModel));
            } else {
                st.forget_armed = Some(Instant::now());
            }
        } else if st.forget_armed.is_some() && !armed {
            st.forget_armed = None;
        }
    });
    let path = model_path(&st.scene_path);
    ui.horizontal_wrapped(|ui| {
        if ui
            .add_enabled(fitted, egui::Button::new("Save model"))
            .on_hover_text(format!("write the model, its fit and its samples to {} (next to the scene file)", path.display()))
            .clicked()
        {
            out.commands.push(Command::Dyn(DynCommand::SaveModel(path.clone())));
            st.status = format!("saving the model to {}", path.display());
        }
        if ui
            .add_enabled(!sampling, egui::Button::new("Load model"))
            .on_hover_text(format!("read {} — refused unless it was trained on this scene's grid, σ, charge, multiplicity and teacher", path.display()))
            .clicked()
        {
            if path.exists() {
                out.commands.push(Command::Dyn(DynCommand::LoadModel(path.clone())));
            } else {
                out.error = Some(format!("no model file {}", path.display()));
            }
        }
        ui.weak(format!("{}", path.display()));
    });
    ui.separator();
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

/// The plan's fields (`sim::SamplingPlan`; the scan range stays at its default).
fn plan_fields(ui: &mut egui::Ui, plan: &mut SamplingPlan) {
    egui::Grid::new("plan").num_columns(2).show(ui, |ui| {
        ui.label("perturbations");
        ui.add(egui::DragValue::new(&mut plan.perturbations).range(0..=2000).speed(1.0))
            .on_hover_text("random displacements of the scene, one teacher solve each");
        ui.end_row();
        ui.label("σ_r");
        ui.add(egui::DragValue::new(&mut plan.perturbation_sigma).range(0.0..=0.5).speed(0.002).fixed_decimals(3).suffix(" Bohr"))
            .on_hover_text("width of each coordinate's Gaussian displacement");
        ui.end_row();
        ui.label("hot burst");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut plan.hot_burst_steps).range(0..=5000).speed(1.0).suffix(" steps"));
            ui.add(egui::DragValue::new(&mut plan.hot_burst_t).range(0.0..=10000.0).speed(10.0).suffix(" K"));
        })
        .response
        .on_hover_text("a short thermostatted DFT-MD run from the scene (dt 0.25 fs, CSVR τ 5 fs); every step is a sample");
        ui.end_row();
        ui.label("scan points / pair");
        ui.add(egui::DragValue::new(&mut plan.scan_points_per_pair).range(0..=200).speed(0.2))
            .on_hover_text(format!(
                "each pair closer than the model's cutoff is scanned from {}× to {}× its current separation",
                plan.scan_range.0, plan.scan_range.1
            ));
        ui.end_row();
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut plan.seed).speed(0.2))
            .on_hover_text("advances after each pass, so a second pass draws new geometries");
        ui.end_row();
    });
}

/// The model's state: who answered this step, its uncertainty against the threshold, trust (with
/// the reason when not), samples, epoch, the last fit, fallbacks over the last 100 steps, the
/// teacher gap and the booked force-mismatch work.
fn model_readouts(ui: &mut egui::Ui, d: &sim::DynSnapshot, m: &ModelStatus) {
    egui::Grid::new("model").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("this step");
        ui.colored_label(source_colour(d.source), RichText::new(source_label(d.source)).strong())
            .on_hover_text("surrogate (green): the model answered; fallback (amber): the model was unsure or untrusted and the DFT teacher answered — its answer becomes a training sample; DFT (blue): a plain Kohn-Sham step");
        ui.end_row();
        ui.label("uncertainty");
        match d.uncertainty {
            Some(u) => {
                let c = if u <= m.force_bound { GREEN } else { AMBER };
                ui.colored_label(c, RichText::new(format!("{u:.1e} / {:.1e} Ha/Bohr", m.force_bound)).monospace())
            }
            None => ui.colored_label(Color32::GRAY, "— (model not consulted)"),
        }
        .on_hover_text("the model's force std per component (max over atoms) against the tightest per-atom bound min(max(2e-3, 10 % |F|), 1e-2) Ha/Bohr. A MODEL uncertainty: the surrogate inherits every DFT approximation (smearing, LSDA) on top of it.");
        ui.end_row();
        ui.label("trusted");
        if !m.fitted {
            ui.colored_label(Color32::GRAY, "no model yet");
        } else if let Some(why) = m.stuck.as_deref().filter(|_| m.trusted) {
            ui.colored_label(AMBER, format!("yes, but {why}"))
                .on_hover_text("the model is trusted where it was taught, but it has not answered at this geometry for a while: every step here is DFT. Move on, sample this region, or (box-affected answers) enlarge the grid.");
        } else if m.trusted && m.quarantined > 0 {
            ui.colored_label(GREEN, format!("yes — {} samples quarantined (the model cannot represent them)", m.quarantined))
                .on_hover_text("kept in the store and the model file but excluded from the fit; geometries near them fall back to DFT (ruling S15)");
        } else if m.trusted {
            ui.colored_label(GREEN, "yes");
        } else {
            ui.colored_label(AMBER, format!("no — {}", m.untrusted_reason.as_deref().unwrap_or("?")))
                .on_hover_text("an untrusted model never answers: every step is DFT and is kept as a sample until a refit earns trust (enough samples for its Q′ columns, ≥ 30 residual degrees of freedom, and its own training data reproduced to the accuracy it claims — one stray sample can untrust it)");
        }
        ui.end_row();
        ui.label("samples");
        ui.monospace(format!("{}{}", m.samples, if m.pending_refit { " (refit pending)" } else { "" }));
        ui.end_row();
        ui.label("model epoch");
        ui.monospace(m.epoch.to_string()).on_hover_text("bumped whenever a refit moves the energy surface (the jump is booked into E conserved)");
        ui.end_row();
        ui.label("last fit");
        match m.fit {
            Some((n, q, e, f)) => ui.monospace(format!("{n} samples, Q′ {q}: rmse {e:.1e} Ha/atom, {f:.1e} Ha/Bohr")),
            None => ui.colored_label(Color32::GRAY, "—"),
        };
        ui.end_row();
        if let Some(r) = &m.last_refit {
            ui.label("last refit");
            match &r.result {
                Ok(n) => ui.monospace(format!("{n} samples, {:.0} ms, at step {} ({} this run)", r.ms, r.at_step, m.refits)),
                Err(e) => ui.colored_label(AMBER, format!("failed at step {}: {e}", r.at_step)),
            };
            ui.end_row();
        }
        ui.label("fallbacks");
        ui.monospace(format!("{:.0} % of the last 100 steps", 100.0 * d.fallback_fraction_100));
        ui.end_row();
        ui.label("teacher gap");
        match (d.teacher_gap, m.last_gap) {
            (Some(g), _) => ui.monospace(format!("{g:+.2e} Ha (this step)")),
            (None, Some((g, step))) => ui.monospace(format!("{g:+.2e} Ha at step {step}")),
            (None, None) => ui.colored_label(Color32::GRAY, "— (no fallback on the model's surface yet)"),
        }
        .on_hover_text("E_DFT − E_model on the most recent fallback or audit booked on the model's surface (the energy stays on the model's surface; the gap is the model's error there)");
        ui.end_row();
        ui.label("last audit");
        match (&m.last_audit, m.audit_in_flight) {
            (_, true) => ui.colored_label(Color32::GRAY, "in flight on the background teacher"),
            (Some(a), false) => {
                let (c, verdict) = match a.verdict {
                    sim::surrogate_ks::AuditVerdict::Passed => (GREEN, "passed"),
                    sim::surrogate_ks::AuditVerdict::Failed => (AMBER, "failed"),
                    sim::surrogate_ks::AuditVerdict::BoxAffected => (AMBER, "not judged (box-affected)"),
                };
                let when = if a.asynchronous { format!(", judged {} steps after dispatch", a.unaudited_steps) } else { String::new() };
                ui.colored_label(c, format!("{verdict}: {:+.2e} Ha ({:.0} ms DFT{when})", a.gap, a.teacher_ms))
            }
            (None, false) => ui.colored_label(Color32::GRAY, "— (none yet)"),
        }
        .on_hover_text(if m.audits_background {
            "E_DFT − E_model at the last audited geometry. The audit runs on a second GPU teacher in the background while the model keeps answering (ruling S14); a failed one untrusts the model when it arrives (the steps since were answered unaudited). A box-affected teacher answer (density on the box faces > 2e-5 e/Bohr³) carries the box's confinement and is not judged."
        } else {
            "E_DFT − E_model at the last audited geometry. With the CPU teacher the audit runs in the step (a second CPU solver would compete for the same cores). A box-affected teacher answer (density on the box faces > 2e-5 e/Bohr³) carries the box's confinement and is not judged."
        });
        ui.end_row();
        ui.label("mismatch work");
        ui.monospace(format!("{:+.2e} Ha", d.work_mismatch))
            .on_hover_text("work of (F_DFT − F_model) on fallback steps, booked into E conserved so the drift measures the integrator, not the model");
        ui.end_row();
    });
}

fn time_section(
    ui: &mut egui::Ui,
    st: &mut UiState,
    input: &PanelInput<'_>,
    out: &mut PanelOutput,
) {
    ui.separator();
    let d = input.snapshot.map(|s| &s.dynamics);
    let sampling = d.is_some_and(sampling_active);
    ui.horizontal(|ui| {
        let label = if input.playing { "Pause" } else { "Play" };
        if ui
            .add_enabled_ui(!sampling, |ui| {
                ui.add_sized([64.0, 0.0], egui::Button::new(RichText::new(label).strong()))
            })
            .inner
            .on_hover_text("Space")
            .on_disabled_hover_text("a sampling pass is running: the dynamics stay paused until it ends (or Cancel it)")
            .clicked()
        {
            out.toggle_play = true;
        }
        if ui
            .add_enabled(!sampling, egui::Button::new("Step"))
            .on_hover_text("pause and advance one time step")
            .clicked()
        {
            out.commands.push(Command::Dyn(DynCommand::Step(1)));
        }
        if ui
            .button("Rewind")
            .on_hover_text("reverse every velocity (v → −v): without thermostat or drag the trajectory retraces itself")
            .clicked()
        {
            out.commands.push(Command::Dyn(DynCommand::Rewind));
        }
    });
    let mut speed = st.speed;
    if ui
        .add(
            egui::Slider::new(&mut speed, 1..=MAX_SPEED)
                .logarithmic(true)
                .text("speed (steps/tick)"),
        )
        .on_hover_text("time steps attempted per ≈16 ms tick; never changes dt. One DFT step usually fills a tick by itself, so the measured fs/s is what counts; with the estimated solver answering (~10 µs a step) a tick runs all its steps and then waits out its 16 ms, except at 1000 (as fast as possible). Above ~30 fs/s a 10 fs bond vibration is faster than the frame rate — you will see a shimmer, not a swing; use the watchable preset to see motion.")
        .changed()
    {
        st.speed = speed;
        out.commands.push(Command::Dyn(DynCommand::SetSpeed(speed)));
    }
    if let Some(d) = d {
        let (n, rate) = watchable_speed(st.dt_fs, d.ms_per_step);
        if ui
            .button(format!("watchable (≈ {rate:.0} fs/s)"))
            .on_hover_text("steps per tick for one ~10 fs H–X vibration in ≈ 0.5 s of wall time, at the current time step and measured cost per step")
            .clicked()
        {
            st.speed = n;
            out.commands.push(Command::Dyn(DynCommand::SetSpeed(n)));
        }
    }
    if let Some(d) = d {
        let measured = input
            .measured_fs_per_s
            .map_or(String::new(), |r| format!("{r:.1} fs/s simulated (last 2 s) · "));
        ui.weak(format!(
            "{measured}{:.2} fs/s this tick · {} step{} last tick · {:.3} ms/step",
            d.wall_ratio,
            d.steps_last_tick,
            if d.steps_last_tick == 1 { "" } else { "s" },
            d.ms_per_step
        ))
        .on_hover_text("simulated femtoseconds per wall second: over the last 2 s of play (what you see), and over the last tick alone (a DFT step and a surrogate tick differ by 10⁴)");
    }
    ui.horizontal(|ui| {
        ui.label("dt");
        let mut dt = st.dt_fs;
        if ui
            .add(
                egui::DragValue::new(&mut dt)
                    .range(0.01..=2.0)
                    .speed(0.005)
                    .fixed_decimals(3)
                    .suffix(" fs"),
            )
            .on_hover_text(format!(
                "time step; the default {DEFAULT_DT_FS} fs resolves the ~10 fs H–X stretch, above {DT_WARN_FS} fs it is under-resolved. Changing dt restarts the drift history."
            ))
            .changed()
        {
            st.dt_fs = dt;
            out.commands.push(Command::Dyn(DynCommand::SetDt(dt)));
        }
        if let Some(d) = d {
            ui.weak(format!("t = {:.2} fs, step {}", d.sim_time_fs(), d.step));
        }
    });
    if let Some(w) = d.and_then(|d| d.dt_warning.as_ref()) {
        ui.colored_label(AMBER, w);
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

/// Drift colour (spec §8): amber above 1e-4 Ha/ps, red above 1e-3.
fn drift_colour(ui: &egui::Ui, drift: f64) -> Color32 {
    match drift.abs() {
        x if x > 1e-3 => RED,
        x if x > 1e-4 => AMBER,
        _ => ui.visuals().text_color(),
    }
}

fn dynamics_readouts(ui: &mut egui::Ui, snapshot: Option<&Snapshot>) {
    ui.separator();
    let Some(s) = snapshot else {
        ui.weak("no solution yet");
        return;
    };
    let d = &s.dynamics;
    let ev = units::hartree_to_ev;
    egui::Grid::new("dynamics")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            ui.label("T");
            ui.monospace(format!("{:.1} K", d.temperature))
                .on_hover_text("kinetic temperature by equipartition over 3N − 3 degrees of freedom (COM motion removed; rotation counted; a single nucleus: 3, lab frame)");
            ui.label("");
            ui.end_row();
            let mut row = |name: &str, x: f64, strong: bool, tip: &str| {
                let t = |s: String| {
                    if strong {
                        RichText::new(s).strong()
                    } else {
                        RichText::new(s)
                    }
                };
                ui.label(t(name.to_string())).on_hover_text(tip);
                ui.monospace(t(format!("{x:+.6} Ha")));
                ui.monospace(t(format!("{:+.3} eV", ev(x))));
                ui.end_row();
            };
            row("KE (nuclei)", d.kinetic, false, "Σ ½ m v² of the nuclei (lab frame)");
            row("E_KS", d.energy_ks, false, "Kohn-Sham energy at the current positions (electrons + nuclear repulsion)");
            row("E total", d.total, true, "KE + E_KS: conserved in NVE without tools");
            row(
                "E conserved",
                d.energy_conserved,
                false,
                "E total + drag spring + field potential − the work of the thermostat, pins, drag damping and COM correction: the invariant the drift is fitted on",
            );
            let estimated = s.backend == Backend::Estimated;
            let model_work = if estimated { d.work_refit + d.work_mismatch } else { 0.0 };
            row(
                "work done by tools",
                d.total - d.energy_conserved - model_work,
                false,
                &format!(
                    "the energy the drag, field, thermostat, pins and COM correction have put into the nuclei and electrons (net of what the spring and the field still hold). \
                     The drag spring pulls with at most {F_DRAG_MAX} Ha/Bohr: drag slowly and the molecule follows with little work; a fast drag mostly heats it (and every tug shows up here). \
                     With the estimated solver the model's own bookings are on the rows below; the three add up to E total − E conserved."
                ),
            );
            if estimated {
                row(
                    "model refit",
                    d.work_refit,
                    false,
                    "the jumps of the model's energy surface under the atoms (a refit, a failed audit's switch to DFT), booked so E conserved stays continuous: a model learning a new bonding situation can book a large one. Not work anybody did — the surface the energy is measured on moved.",
                );
                row(
                    "model force mismatch",
                    d.work_mismatch,
                    false,
                    "work of (F_DFT − F_model) on fallback steps: the atoms move on DFT forces while the energy is the model's, so the difference is booked like an external force",
                );
            }
            ui.label("drift");
            match d.drift_per_ps {
                Some(x) => ui.colored_label(drift_colour(ui, x), RichText::new(format!("{x:+.2e} Ha/ps")).monospace()),
                None => ui.colored_label(Color32::GRAY, "—"),
            }
            .on_hover_text(format!(
                "least-squares slope of E conserved over the last ≤ 2000 steps (shown after {DRIFT_MIN_POINTS}); amber above 1e-4, red above 1e-3 Ha/ps"
            ));
            ui.label("");
            ui.end_row();
            ui.label("per step");
            if d.source == Source::Surrogate {
                ui.monospace("model, no SCF");
                ui.monospace(format!("{:.3} ms", d.ms_per_step_model)).on_hover_text(format!(
                    "moving mean over the model's own steps; DFT steps here (fallbacks; audits in the step with the CPU teacher) average {:.0} ms",
                    d.ms_per_step_dft
                ));
            } else {
                let cost = match d.eval.readbacks {
                    Some(r) => format!("{} SCF steps, {r} readbacks", d.eval.scf_steps),
                    None => format!("{} SCF steps", d.eval.scf_steps),
                };
                ui.monospace(cost);
                ui.monospace(format!("{} ms", d.eval.solve_ms));
            }
            ui.end_row();
        });
    sparkline(ui, &d.history, &d.history_source, s.backend == Backend::Estimated);
    if let Some(w) = box_warning(&s.scene) {
        ui.colored_label(w.0, w.1);
    }
    for n in d.notes.iter().rev().take(3) {
        ui.weak(n);
    }
}

/// Distance (Bohr) from a nucleus to the box face below which the panel warns: the grid ends
/// there, so the nucleus's density is cut off (a display heuristic — an H 1s density at 1.5 Bohr
/// is e⁻³ ≈ 5 % of its peak — not a physics parameter).
const BOX_MARGIN_WARN: f64 = 1.5;

/// The nucleus nearest to (or beyond) the box face, when it is within [`BOX_MARGIN_WARN`]: the
/// solver's world ends at the box, so forces there are unreliable and a nucleus outside it is
/// invisible to the electrons (nothing stops a molecule with momentum from leaving it).
fn box_warning(scene: &Scene) -> Option<(Color32, String)> {
    let half = (scene.grid_n as f64 - 1.0) * scene.grid_h / 2.0;
    let (i, margin) = scene
        .nuclei
        .iter()
        .enumerate()
        .map(|(i, n)| (i, half - n.pos.iter().fold(0.0f64, |m, x| m.max(x.abs()))))
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    if margin < 0.0 {
        Some((RED, format!("nucleus #{i} is outside the box ({:.1} Bohr beyond its face): the electrons cannot reach it, its forces are meaningless", -margin)))
    } else if margin < BOX_MARGIN_WARN {
        Some((AMBER, format!("nucleus #{i} is {margin:.1} Bohr from the box face: its density is cut off there, forces and energies are unreliable")))
    } else {
        None
    }
}

/// E_total and T over the history window, each scaled to its own range: `Painter::line_segment`
/// polylines, no plot crate. Each E_total segment takes the colour of the source of the step it
/// ends on (`source_colour`: surrogate green, fallback amber, DFT blue), and fallback steps get a
/// dot, so a single DFT step among a thousand surrogate ones is visible; `legend` adds the key.
fn sparkline(ui: &mut egui::Ui, history: &[(f64, f64, f64)], sources: &[Source], legend: bool) {
    let width = ui.available_width().max(120.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 56.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_black_alpha(70));
    if history.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "E total and T appear here once time runs",
            egui::FontId::proportional(11.0),
            ui.visuals().weak_text_color(),
        );
        return;
    }
    let start = history.len().saturating_sub(SPARK_WINDOW);
    let pts: Vec<&(f64, f64, f64)> = history[start..].iter().collect();
    // parallel to `history` (a snapshot without sources draws everything as DFT)
    let src = |i: usize| sources.get(start + i).copied().unwrap_or(Source::Dft);
    let (t0, t1) = (pts[0].0, pts[pts.len() - 1].0);
    let range = |f: &dyn Fn(&(f64, f64, f64)) -> f64| {
        pts.iter()
            .map(|p| f(p))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)))
    };
    let inner = rect.shrink(4.0);
    let series = |f: &dyn Fn(&(f64, f64, f64)) -> f64, colour: Option<Color32>| -> (f64, f64) {
        let (lo, hi) = range(f);
        let span = if hi > lo { hi - lo } else { 1.0 };
        let to = |p: &(f64, f64, f64)| {
            let x = if t1 > t0 { (p.0 - t0) / (t1 - t0) } else { 0.0 };
            let y = if hi > lo { (f(p) - lo) / span } else { 0.5 };
            egui::pos2(
                inner.left() + x as f32 * inner.width(),
                inner.bottom() - y as f32 * inner.height(),
            )
        };
        for (i, w) in pts.windows(2).enumerate() {
            let c = colour.unwrap_or_else(|| source_colour(src(i + 1)));
            painter.line_segment([to(w[0]), to(w[1])], egui::Stroke::new(1.2, c));
        }
        if colour.is_none() {
            for (i, p) in pts.iter().enumerate() {
                if src(i) == Source::Fallback {
                    painter.circle_filled(to(p), 2.2, AMBER);
                }
            }
        }
        (lo, hi)
    };
    let (elo, ehi) = series(&|p| p.1, None);
    let (tlo, thi) = series(&|p| p.2, Some(SPARK_T));
    response.on_hover_text(
        "each curve is scaled to its own range over the window shown; E total = KE + E_KS (it moves when tools or the thermostat act — the drift readout uses E conserved)",
    );
    let span_fs = units::au_time_to_fs(t1 - t0);
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(SPARK_E, format!("E total {:+.5} … {:+.5} Ha", elo, ehi));
        ui.colored_label(SPARK_T, format!("T {tlo:.0} … {thi:.0} K"));
        ui.weak(format!("last {span_fs:.1} fs"));
    });
    if legend {
        ui.horizontal_wrapped(|ui| {
            ui.weak("E total by step:");
            for s in [Source::Surrogate, Source::Fallback, Source::Dft] {
                ui.colored_label(source_colour(s), source_label(s));
            }
        });
    }
}

fn thermostat_section(
    ui: &mut egui::Ui,
    st: &mut UiState,
    snapshot: Option<&Snapshot>,
    out: &mut PanelOutput,
) {
    ui.separator();
    ui.horizontal(|ui| {
        let toggled = ui
            .checkbox(&mut st.thermostat_on, "thermostat")
            .on_hover_text("Bussi CSVR (canonical velocity rescaling). Switching it on with the nuclei at rest draws Maxwell–Boltzmann velocities at T first.")
            .changed();
        let t = ui
            .add(
                egui::DragValue::new(&mut st.thermostat_t)
                    .range(0.0..=5000.0)
                    .speed(1.0)
                    .suffix(" K"),
            )
            .on_hover_text("target temperature")
            .changed();
        let tau = ui
            .add(
                egui::DragValue::new(&mut st.thermostat_tau)
                    .range(1.0..=1000.0)
                    .speed(0.2)
                    .prefix("τ ")
                    .suffix(" fs"),
            )
            .on_hover_text("coupling time: smaller = tighter control, more disturbed dynamics")
            .changed();
        if toggled || (st.thermostat_on && (t || tau)) {
            out.commands
                .push(Command::Dyn(DynCommand::SetThermostat(st.thermostat())));
        }
    });
    ui.horizontal(|ui| {
        ui.label("uniform acceleration (gravity hook)")
            .on_hover_text("invisible at atomic scale, by design");
        if ui
            .selectable_label(st.gravity, "g = 9.8 m/s² (−y)")
            .on_hover_text(format!(
                "{:.3e} Bohr per a.u.² — invisible at atomic scale, by design (½ g t² over 1 ps ≈ 1e-11 Bohr)",
                gravity_au()
            ))
            .clicked()
        {
            st.gravity = true;
            out.commands
                .push(Command::Dyn(DynCommand::SetExternalField(st.field())));
        }
        if ui.selectable_label(!st.gravity, "off").clicked() {
            st.gravity = false;
            out.commands
                .push(Command::Dyn(DynCommand::SetExternalField(st.field())));
        }
    });
    if ui
        .checkbox(&mut st.com_correction, "COM correction")
        .on_hover_text("subtract the mean provider force: the grid's egg-box net force (~3e-4 Ha/Bohr at h = 0.25) would otherwise push a free molecule across the box")
        .changed()
    {
        out.commands.push(Command::Dyn(DynCommand::SetComCorrection(
            st.com_correction,
        )));
    }
    if let Some(s) = snapshot {
        let a = s.dynamics.scene.external_field;
        if a != [0.0; 3] {
            ui.weak(format!(
                "field |a| = {:.3e} Bohr/a.u.² (invisible at atomic scale, by design)",
                (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
            ));
        }
    }
}

fn scene_section(ui: &mut egui::Ui, st: &mut UiState, out: &mut PanelOutput) {
    ui.separator();
    ui.horizontal(|ui| {
        let before = st.preset.clone();
        egui::ComboBox::from_label("preset")
            .selected_text(&st.preset)
            .show_ui(ui, |ui| {
                for name in Scene::PRESETS {
                    ui.selectable_value(&mut st.preset, name.to_string(), *name);
                }
            });
        let reload = ui
            .button("Reset")
            .on_hover_text("reload the preset (pauses; time restarts)")
            .clicked();
        if st.preset != before || reload {
            if let Some(s) = Scene::preset(&st.preset) {
                st.adopt(&s);
                out.commands.push(Command::SetScene(s));
            }
        }
    });
}

fn tool_section(
    ui: &mut egui::Ui,
    st: &mut UiState,
    input: &PanelInput<'_>,
    out: &mut PanelOutput,
) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("tool");
        ui.radio_value(&mut st.mode, ToolMode::Orbit, "Orbit");
        ui.radio_value(&mut st.mode, ToolMode::Place, "Place")
            .on_hover_text("click to drop a nucleus at rest (also while playing)");
        ui.radio_value(&mut st.mode, ToolMode::Select, "Select")
            .on_hover_text("click a nucleus; drag it: paused = move it there, playing = pull it with a spring");
    });
    let name = |z: u8| format!("{} (Z = {z})", elements::symbol(z));
    egui::ComboBox::from_label("element to place")
        .selected_text(name(st.element))
        .show_ui(ui, |ui| {
            for z in 1..=10u8 {
                ui.selectable_value(&mut st.element, z, name(z));
            }
        });
    let Some((i, nuc)) = input
        .selected
        .and_then(|i| input.scene.nuclei.get(i).map(|n| (i, n)))
    else {
        ui.weak("no selection (Select tool, click a nucleus)");
        return;
    };
    ui.label(format!(
        "selected: {} #{i} at ({:.2}, {:.2}, {:.2}) Bohr",
        elements::symbol(nuc.z),
        nuc.pos[0],
        nuc.pos[1],
        nuc.pos[2]
    ));
    ui.horizontal(|ui| {
        if ui.button("Delete selected").clicked() {
            out.commands.push(Command::RemoveNucleus(i));
        }
        if ui.button("Focus selected").clicked() {
            out.focus_selected = true;
        }
        let pins = input.snapshot.map(|s| &s.dynamics.pins);
        if let Some(&was) = pins.and_then(|p| p.get(i)) {
            let mut pinned = was;
            if ui
                .checkbox(&mut pinned, "pinned")
                .on_hover_text("hold this nucleus still (velocity zeroed every step)")
                .changed()
            {
                out.commands.push(Command::Dyn(DynCommand::Pin(i, pinned)));
            }
        }
    });
}

fn electrons_section(ui: &mut egui::Ui, st: &mut UiState, scene: &Scene, out: &mut PanelOutput) {
    ui.separator();
    let protons: i32 = scene.nuclei.iter().map(|n| n.z as i32).sum();
    let electrons = (protons - scene.charge).max(0);
    ui.weak("charge, multiplicity, σ and grid edits pause the run and restart its time");
    egui::Grid::new("electrons").num_columns(2).show(ui, |ui| {
        ui.label("charge");
        let mut c = scene.charge;
        if ui
            // at least one electron must remain; anions down to −4
            .add(
                egui::DragValue::new(&mut c)
                    .range(-4..=(protons - 1).max(0))
                    .speed(0.05),
            )
            .changed()
            && c != scene.charge
        {
            out.commands.push(Command::SetCharge(c));
        }
        ui.end_row();
        ui.label("multiplicity");
        ui.horizontal(|ui| {
            let mut auto = scene.multiplicity.is_none();
            let mut m = scene.multiplicity.unwrap_or(1 + (electrons as u32 % 2));
            let toggled = ui.checkbox(&mut auto, "auto").changed();
            let edited = ui
                .add_enabled(
                    !auto,
                    egui::DragValue::new(&mut m).range(1..=11).speed(0.05),
                )
                .changed();
            let want = if auto { None } else { Some(m) };
            if (toggled || edited) && want != scene.multiplicity {
                out.commands.push(Command::SetMultiplicity(want));
            }
        });
        ui.end_row();
        ui.label("σ (nuclear smearing)");
        ui.horizontal(|ui| {
            let mut auto = scene.sigma.is_none();
            let mut s = scene.sigma.unwrap_or(scene.grid_h);
            let toggled = ui.checkbox(&mut auto, "auto = h").changed();
            let edited = ui
                .add_enabled(
                    !auto,
                    egui::DragValue::new(&mut s)
                        .range(0.05..=1.0)
                        .speed(0.005)
                        .suffix(" Bohr"),
                )
                .changed();
            let want = if auto { None } else { Some(s) };
            if (toggled || edited) && want != scene.sigma {
                out.commands.push(Command::SetSigma(want));
            }
        });
        ui.end_row();
    });
    grid_controls(ui, st, scene, electrons, out);
}

fn grid_controls(
    ui: &mut egui::Ui,
    st: &mut UiState,
    scene: &Scene,
    electrons: i32,
    out: &mut PanelOutput,
) {
    ui.horizontal(|ui| {
        ui.label("grid n");
        // n must be odd: edit k with n = 2k + 1, shown and typed as n
        let mut k = (st.grid_n.max(1) - 1) / 2;
        ui.add(
            egui::DragValue::new(&mut k)
                .range(8..=48)
                .speed(0.1)
                .custom_formatter(|v, _| format!("{}", 2 * v as usize + 1))
                .custom_parser(|s| {
                    s.trim()
                        .parse::<f64>()
                        .ok()
                        .map(|n| ((n - 1.0) / 2.0).round())
                }),
        );
        st.grid_n = 2 * k + 1;
        ui.label("h");
        ui.add(
            egui::DragValue::new(&mut st.grid_h)
                .range(0.1..=0.5)
                .speed(0.002)
                .fixed_decimals(3)
                .suffix(" Bohr"),
        );
    });
    let points = (st.grid_n as f64).powi(3);
    ui.weak(format!(
        "cost {}³ × {electrons} electrons ≈ {:.2} M points (box {:.1} Bohr; now {}³ at {})",
        st.grid_n,
        points * electrons.max(1) as f64 / 1e6,
        (st.grid_n - 1) as f64 * st.grid_h,
        scene.grid_n,
        scene.grid_h
    ));
    let pending = st.grid_n != scene.grid_n || st.grid_h != scene.grid_h;
    if ui
        .add_enabled(pending, egui::Button::new("Apply grid"))
        .clicked()
    {
        match validate_grid(st.grid_n, st.grid_h) {
            Ok(()) => out.commands.push(Command::SetGrid {
                n: st.grid_n,
                h: st.grid_h,
            }),
            Err(e) => out.error = Some(e),
        }
    }
}

fn view_section(ui: &mut egui::Ui, st: &mut UiState) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("density");
        ui.radio_value(&mut st.volume_mode, 0, "volume");
        ui.radio_value(&mut st.volume_mode, 1, "isosurface");
    });
    let volume = st.volume_mode == 0;
    ui.add_enabled(
        volume,
        egui::Slider::new(&mut st.gamma, 0.2..=1.0).text("gamma"),
    );
    ui.add_enabled(
        volume,
        egui::Slider::new(&mut st.opacity, 0.1..=20.0)
            .logarithmic(true)
            .text("opacity"),
    );
    ui.add_enabled(
        !volume,
        egui::Slider::new(&mut st.iso, 0.01..=0.9)
            .logarithmic(true)
            .text("iso (× ρ max)"),
    );
}

fn display_section(ui: &mut egui::Ui, st: &mut UiState) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("display");
        ui.checkbox(&mut st.show_cell, "cell")
            .on_hover_text("the simulation box: nuclei must stay inside it");
        ui.checkbox(&mut st.show_floor, "floor grid");
        ui.checkbox(&mut st.show_axes, "axes");
        ui.checkbox(&mut st.smooth_motion, "smooth motion")
            .on_hover_text("interpolates between simulation snapshots (one publish interval of latency)");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut st.show_place_grid, "placement plane")
            .on_hover_text("Place tool: the plane the next click lands on");
        ui.label("spacing");
        ui.radio_value(&mut st.grid_in_angstrom, false, "1 Bohr");
        ui.radio_value(&mut st.grid_in_angstrom, true, "1 Å");
    });
}

/// The latest evaluation's energy breakdown and solver diagnostics.
fn energy_section(ui: &mut egui::Ui, snapshot: Option<&Snapshot>) {
    ui.separator();
    let Some(s) = snapshot else {
        return;
    };
    let d = &s.dynamics;
    let e = &d.eval.energy;
    let ev = units::hartree_to_ev;
    egui::CollapsingHeader::new("energy breakdown and solver")
        .default_open(false)
        .show(ui, |ui| {
            egui::Grid::new("energy")
                .num_columns(3)
                .striped(true)
                .show(ui, |ui| {
                    let mut row = |name: &str, x: f64| {
                        ui.label(name);
                        ui.monospace(format!("{x:+.6} Ha"));
                        ui.monospace(format!("{:+.3} eV", ev(x)));
                        ui.end_row();
                    };
                    row("E_KS", e.total);
                    row("kinetic (electrons)", e.kinetic);
                    row("external", e.external);
                    row("Hartree", e.hartree);
                    row("xc", e.xc);
                    row("nuclear", e.nuclear);
                });
            let sigma = s.scene.sigma.unwrap_or(s.scene.grid_h);
            ui.weak(format!(
                "smeared nuclei (σ = {sigma} Bohr): absolute energies depend on σ and h"
            ));
            if d.source == Source::Surrogate {
                ui.weak("surrogate step: only the total is the model's (it has no electrons); the parts above are 0");
            } else if d.source == Source::Fallback {
                ui.weak("fallback step: the DFT teacher's breakdown (with a trusted model E_KS above is the model's: total − teacher gap)");
            }
            if s.engine == Backend::Gpu {
                ui.weak("GPU (f32 grid): energies scatter by up to ~1e-5 Ha between solve paths");
            }
            egui::Grid::new("scf").num_columns(2).show(ui, |ui| {
                let mut row = |name: &str, v: String| {
                    ui.label(name);
                    ui.monospace(v);
                    ui.end_row();
                };
                row("SCF steps", d.eval.scf_steps.to_string());
                // the worst LOBPCG residual among the solves the converged state rests on (the
                // GPU's deliberately loosened early SCF steps excluded)
                row(
                    "eigen residual (final)",
                    format!("{:.1e}", d.eval.max_eigen_residual),
                );
                row("Poisson residual", format!("{:.1e}", d.eval.poisson_residual));
                row(
                    "solve",
                    match d.eval.readbacks {
                        Some(r) => {
                            format!("{} ms, {r} readbacks ({})", d.eval.solve_ms, s.engine.label())
                        }
                        None => format!("{} ms ({})", d.eval.solve_ms, s.engine.label()),
                    },
                );
                row("grid", format!("{}³ × {} Bohr", s.scene.grid_n, s.scene.grid_h));
                if let Some(max) = d.display.as_ref().map(|x| match x {
                    sim::Display::Cpu { max, .. } | sim::Display::Gpu { max, .. } => *max,
                }) {
                    row("ρ max", format!("{max:.3} e/Bohr³"));
                }
            });
            for n in &d.eval.notes {
                ui.weak(n);
            }
        });
    for w in &d.eval.warnings {
        ui.colored_label(AMBER, w);
    }
}

fn file_section(ui: &mut egui::Ui, st: &mut UiState, scene: &Scene, sampling: bool, out: &mut PanelOutput) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("file");
        ui.text_edit_singleline(&mut st.scene_path);
    });
    ui.horizontal(|ui| {
        if ui
            .button("Save")
            .on_hover_text("positions and velocities as they are now")
            .clicked()
        {
            match std::fs::write(&st.scene_path, scene.to_text()) {
                Ok(()) => {
                    st.status = format!("saved {} nuclei to {}", scene.nuclei.len(), st.scene_path)
                }
                Err(e) => out.error = Some(format!("save {}: {e}", st.scene_path)),
            }
        }
        if ui
            .button("Load")
            .on_hover_text("estimated solver: the model sidecar next to the file (same name, .model) is loaded with it when it exists")
            .clicked()
        {
            match load(&st.scene_path) {
                Ok(s) => {
                    st.status = format!("loaded {}", st.scene_path);
                    st.adopt(&s);
                    out.commands.push(Command::SetScene(s));
                    let sidecar = model_path(&st.scene_path);
                    if st.backend == Backend::Estimated && sidecar.exists() {
                        if sampling {
                            // the new scene cancels the pass, but a load would be refused while
                            // it runs (and abort its solve for nothing)
                            st.status += &format!(" — its model {} was not loaded (a sampling pass was running): press Load model", sidecar.display());
                        } else {
                            st.status += &format!(" and its model {}", sidecar.display());
                            out.commands.push(Command::Dyn(DynCommand::LoadModel(sidecar)));
                        }
                    }
                }
                Err(e) => out.error = Some(e),
            }
        }
    });
}

fn load(path: &str) -> Result<Scene, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("load {path}: {e}"))?;
    let scene = Scene::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    scene.validate().map_err(|e| format!("{path}: {e}"))?;
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// g = 9.8 m/s² in atomic units: a₀ / t_au² = 0.529177e-10 m / (2.418884e-17 s)²
    /// = 9.0442e22 m/s², so g = 1.0836e-22 Bohr/a.u.².
    #[test]
    fn gravity_in_atomic_units() {
        let g = gravity_au();
        assert!((g - 1.0836e-22).abs() < 2e-26, "g = {g:e}");
    }

    /// The box warning names the nucleus nearest the face: none for the preset (6 Bohr half-box,
    /// nuclei at ±0.8), amber within 1.5 Bohr, red outside.
    #[test]
    fn box_warning_flags_nuclei_near_or_beyond_the_face() {
        let mut s = Scene::preset("h2").unwrap();
        assert_eq!(box_warning(&s), None);
        s.nuclei[1].pos = [0.0, -5.0, 0.0];
        let (c, m) = box_warning(&s).unwrap();
        assert_eq!(c, AMBER);
        assert!(m.contains("#1") && m.contains("1.0 Bohr"), "{m}");
        s.nuclei[0].pos = [0.0, 0.0, 7.5];
        let (c, m) = box_warning(&s).unwrap();
        assert_eq!(c, RED);
        assert!(m.contains("#0") && m.contains("outside"), "{m}");
    }

    /// A new worker gets the panel's run settings; defaults send only speed and COM correction.
    #[test]
    fn run_settings_mirror_the_panel() {
        let mut st = UiState::new("h2", &Scene::preset("h2").unwrap());
        assert!(st.com_correction, "COM correction defaults on in the app");
        assert_eq!(st.run_settings().len(), 2);
        st.thermostat_on = true;
        st.gravity = true;
        st.dt_fs = 0.4;
        let v = st.run_settings();
        assert!(v.iter().any(|c| matches!(c, DynCommand::SetThermostat(Some((t, tau))) if *t == 300.0 && *tau == 10.0)));
        assert!(v.iter().any(|c| matches!(c, DynCommand::SetDt(x) if *x == 0.4)));
        assert!(v.iter().any(|c| matches!(c, DynCommand::SetExternalField(a) if a[1] < 0.0)));
    }

    /// "watchable": one ~10 fs vibration in ≈ 0.5 s — at dt 0.25 fs on the surrogate (≈ 0.01 ms
    /// a step) 1 step per paced 16 ms tick, 15.6 fs/s; a DFT step (100 ms) is slower than that
    /// whatever the speed (1 step, 2.5 fs/s); a small dt asks for more steps per tick.
    #[test]
    fn the_watchable_speed_shows_a_vibration_in_about_half_a_second() {
        let (n, r) = watchable_speed(0.25, 0.01);
        assert_eq!(n, 1);
        assert!((r - 15.625).abs() < 1e-9, "{r}");
        let (n, r) = watchable_speed(0.25, 100.0);
        assert_eq!(n, 1);
        assert!((r - 2.5).abs() < 1e-9, "{r}");
        let (n, r) = watchable_speed(0.01, 0.01);
        assert_eq!(n, 32);
        assert!((r - 20.0).abs() < 1e-9, "{r}");
    }
}
