//! Window-independent viewer state: the worker, the local scene mirror, the latest snapshot, the
//! camera, and the panel. Shared by the interactive window (`main`) and the headless screenshot.
use crate::gfx::Painter;
use crate::scene::Scene;
use crate::sim::{Backend, Command, Event as SimEvent, SimWorker, Snapshot};
use crate::tools::{self, ToolState};
use crate::ui::{self, UiState};
use glam::Vec3;
use gpu::GpuContext;
use render::camera::OrbitCamera;
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

/// How long leaving the estimated solver waits for its worker to write the autosave (a GPU
/// solve stops within one SCF step; a CPU solve in flight cannot be interrupted).
const RETIRE_WAIT: Duration = Duration::from_secs(3);

/// The density on screen: which DFT evaluation it came from, and whether it is shown faded
/// because the latest step was the surrogate's (spec §5: no electrons on a surrogate step).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DensityShown {
    /// Simulated time (fs) and step of the snapshot whose evaluation carried the density.
    pub from: Option<(f64, u64)>,
    /// The latest evaluation was the surrogate's: the density is the last DFT one, faded.
    pub faded: bool,
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
                }
            }
            (None, Source::Surrogate) => self.faded = self.from.is_some(),
            (None, _) => {}
        }
    }

    /// The volume pass's fade factor.
    pub fn fade(&self) -> f32 {
        if self.faded {
            SURROGATE_DENSITY_FADE
        } else {
            1.0
        }
    }

    /// The label over the faded cloud (`None` when not faded). `step` is the current step: a
    /// smaller one means time restarted since the density was solved (a replaced scene).
    pub fn label(&self, step: u64) -> Option<String> {
        let (t, from_step) = self.from.filter(|_| self.faded)?;
        Some(if step < from_step {
            "density from the last DFT step, before the scene was replaced (the surrogate has no electrons)".into()
        } else {
            format!("density from the last DFT step (t = {t:.2} fs) — the surrogate has no electrons")
        })
    }
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

pub struct Viewer {
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
    /// Where an estimated worker keeps the model across solver switches (deleted at exit).
    pub autosave: PathBuf,
    /// (wall, simulated fs) of recent playing snapshots: the measured rate over [`RATE_WINDOW`].
    rate: std::collections::VecDeque<(Instant, f64)>,
    /// Mirrored commands (`Command::is_edit`) sent to the current worker.
    edits_sent: u64,
    last_play_log: Option<Instant>,
}

impl Viewer {
    /// The preset's scene, no worker yet: call [`Viewer::start`].
    pub fn new(preset: &str) -> Option<Viewer> {
        let scene = Scene::preset(preset)?;
        Some(Viewer {
            sim: None,
            backend: Backend::Cpu,
            gpu: None,
            ui: UiState::new(preset, &scene),
            scene,
            snapshot: None,
            camera: OrbitCamera::new(Vec3::ZERO, 14.0),
            tool: ToolState::default(),
            busy: false,
            last_error: None,
            egui_ctx: egui::Context::default(),
            playing: false,
            drag_target: None,
            density: DensityShown::default(),
            autosave: autosave_path(),
            rate: Default::default(),
            edits_sent: 0,
            last_play_log: None,
        })
    }

    /// (Re)starts the worker on `backend` with the current scene (velocities included), paused,
    /// and re-sends the panel's run settings (dt, speed, thermostat, field, COM correction); pins,
    /// the time and the drift history start afresh. `Gpu` without a context falls back to `Cpu`
    /// (`Estimated` then teaches with the CPU). A previous worker is detached, not joined: a GPU
    /// solve stops at its next SCF step, a CPU solve in flight finishes in the background and is
    /// discarded — except an estimated one, which is given [`RETIRE_WAIT`] to write its model to
    /// the autosave path first. A new estimated worker is sent [`Command::EnsureModel`]: the
    /// autosaved model if there is one, else a sampling pass with the panel's plan. The displayed
    /// density stays until the new worker publishes.
    pub fn start(&mut self, backend: Backend, gpu: Option<GpuContext>) {
        if let Some(old) = self.sim.take() {
            if self.backend == Backend::Estimated {
                if !old.retire(RETIRE_WAIT) {
                    self.last_error = Some(format!(
                        "the estimated solver did not shut down within {RETIRE_WAIT:?}: its model may not have been carried over"
                    ));
                }
            } else {
                old.detach();
            }
        }
        let backend = if gpu.is_none() && backend == Backend::Gpu { Backend::Cpu } else { backend };
        let estimated = backend == Backend::Estimated;
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
        self.playing = false;
        self.drag_target = None;
        self.tool.dragging = false;
        if estimated {
            let autoload = self.autosave.exists().then(|| self.autosave.clone());
            self.send(Command::EnsureModel {
                autoload,
                plan: self.ui.plan.clone(),
            });
            if self.ui.audit_every != sim::FallbackPolicy::default().audit_every {
                self.send(Command::SetAudit(self.ui.audit_every));
            }
        }
    }

    /// Sends a command and applies it to the local mirror. Index-shifting or whole-scene
    /// commands invalidate the selection.
    pub fn send(&mut self, cmd: Command) {
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
        );
        if replaces && self.playing {
            self.ui.status = "paused: this edit replaces the scene (time, pins and the drift history restart) — press Play to continue".into();
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
            sim.send(cmd);
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
        let c = if self.playing {
            DynCommand::Pause
        } else {
            DynCommand::Play
        };
        self.send(Command::Dyn(c));
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
        match ev {
            SimEvent::Snapshot(s) => {
                self.log(&s);
                if let Err(e) = painter.show_snapshot(ctx, &s, depth) {
                    self.last_error = Some(e);
                }
                self.density.follow(&s);
                self.track_rate(&s);
                if s.edits == self.edits_sent {
                    self.scene = s.scene.clone();
                    self.playing = !s.dynamics.paused;
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
                self.snapshot = Some(*s);
            }
            SimEvent::Error(message) => {
                eprintln!("worker error: {message}");
                self.last_error = Some(message);
            }
            SimEvent::Busy(b) => self.busy = b,
            SimEvent::Retired => {}
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
                ", {:?}, fallbacks {:.0} % of 100, model {} samples epoch {} {}{}",
                d.source,
                100.0 * d.fallback_fraction_100,
                m.samples,
                m.epoch,
                if m.trusted { "trusted" } else { "untrusted" },
                d.teacher_gap.map_or(String::new(), |g| format!(", teacher gap {g:+.2e} Ha"))
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
    ) {
        let (w, h) = tool.view_px;
        if w <= 1.0 || h <= 1.0 {
            return;
        }
        let ppp = root.pixels_per_point();
        let to_pos = |(x, y): (f32, f32)| egui::pos2(x / ppp, y / ppp);
        let painter = root.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("distance-guides"),
        ));
        let line = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(90));
        // pins: a small square around each pinned nucleus
        let pin = egui::Stroke::new(1.5, egui::Color32::from_rgb(120, 200, 255));
        for (i, _) in pins.iter().enumerate().filter(|(_, &p)| p) {
            if let Some(px) = scene
                .nuclei
                .get(i)
                .and_then(|n| tools::project(camera, tools::v3(n.pos), w, h))
            {
                let r = egui::Rect::from_center_size(to_pos(px), egui::vec2(14.0, 14.0));
                painter.rect_stroke(r, 0.0, pin, egui::StrokeKind::Middle);
            }
        }
        // the drag spring: target ring and a line to its nucleus
        if let (Some(t), Some(i)) = (drag_target, tool.selected) {
            let spring = egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 190, 90));
            if let Some(t_px) = tools::project(camera, tools::v3(t), w, h) {
                painter.circle_stroke(to_pos(t_px), 5.0, spring);
                if let Some(n_px) = scene
                    .nuclei
                    .get(i)
                    .and_then(|n| tools::project(camera, tools::v3(n.pos), w, h))
                {
                    painter.line_segment([to_pos(n_px), to_pos(t_px)], spring);
                }
            }
        }
        let (origin, exclude) = match (tool.dragging, tool.selected, mode, tool.cursor) {
            (true, Some(i), _, _) if i < scene.nuclei.len() => (scene.nuclei[i].pos, Some(i)),
            (false, _, ui::ToolMode::Place, Some((px, py))) => {
                match tools::place_point(camera, px, py, w, h) {
                    Some(p) => (p, None),
                    None => return,
                }
            }
            _ => return,
        };
        let guides = tools::distance_guides(origin, &scene.nuclei, exclude);
        let Some(o_px) = tools::project(camera, tools::v3(origin), w, h) else {
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
            let Some(t_px) = tools::project(camera, tools::v3(scene.nuclei[g.index].pos), w, h)
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

    /// Runs the panel for one frame and acts on its output.
    pub fn run_ui(&mut self, raw: egui::RawInput) -> egui::FullOutput {
        let mut outs = Vec::new();
        let ctx = self.egui_ctx.clone();
        let measured_fs_per_s = self.measured_fs_per_s();
        let full = ctx.run_ui(raw, |root| {
            outs.push(ui::side_panel(
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
                },
            ));
            if let Some((i, n)) = self.tool.selected.and_then(|i| self.scene.nuclei.get(i).map(|n| (i, n))) {
                crate::atom_card::show(root, n, i);
            }
            let step = self.snapshot.as_ref().map_or(0, |s| s.dynamics.step);
            if let Some(label) = self.density.label(step) {
                // bottom centre of the 3D view (the side panel is ≈ 330 points wide)
                egui::Area::new(egui::Id::new("density-label"))
                    .anchor(egui::Align2::CENTER_BOTTOM, [165.0, -14.0])
                    .interactable(false)
                    .show(root, |ui| {
                        egui::Frame::new()
                            .fill(egui::Color32::from_black_alpha(150))
                            .corner_radius(4.0)
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(label).color(egui::Color32::from_gray(210)));
                            });
                    });
            }
            let pins = self.snapshot.as_ref().map_or(&[][..], |s| &s.dynamics.pins[..]);
            Self::distance_overlay(
                root,
                &self.camera,
                &self.scene,
                &self.tool,
                self.ui.mode,
                self.drag_target,
                pins,
            );
        });
        // egui may run the closure more than once in a frame (discarded passes); input is
        // consumed in the first, so keeping every pass's output loses nothing and repeats nothing.
        for out in outs {
            if let Some(e) = out.error {
                self.last_error = Some(e);
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
        }
        if self.ui.backend != self.backend {
            self.start(self.ui.backend, self.gpu.clone());
            self.ui.status = format!(
                "solver switched to {}: the run restarts paused at t = 0 with the panel's settings",
                self.backend.label()
            );
        }
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use render::headless::{Headless, HEADLESS_FORMAT};
    use std::time::Duration;

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
        let p = hl.read_rgba()[(h / 2 * w + w / 2) as usize];
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
        let mut v = Viewer::new("h").unwrap();
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        let gpu_lit = settle(&mut v, &hl, &mut painter, Backend::Gpu);
        v.send(Command::MoveNucleus {
            index: 0,
            pos: [0.25, 0.0, 0.0],
        });
        v.ui.backend = Backend::Cpu;
        let _ = v.run_ui(egui::RawInput::default());
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
        let _ = v.run_ui(egui::RawInput::default());
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
        let mut v = Viewer::new("h2").unwrap();
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
        let mut v = Viewer::new("h2").unwrap();
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
        let mut rest = Scene::preset("h2").unwrap();
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
    /// it starts a sampling pass (the panel's plan, `EnsureModel`); after it, playing shows the
    /// last DFT density faded to `SURROGATE_DENSITY_FADE` with its label while the surrogate
    /// answers; switching to GPU and back restores the model from the session autosave without a
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
        let mut v = Viewer::new("h2").unwrap();
        assert_ne!(v.autosave, Viewer::new("h2").unwrap().autosave, "one autosave per viewer");
        v.scene.grid_n = 41;
        v.ui.plan = sim::SamplingPlan { perturbations: 30, hot_burst_steps: 40, ..Default::default() };
        v.start(Backend::Gpu, Some(hl.ctx.clone()));
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        assert_eq!(v.density.fade(), 1.0);
        v.ui.backend = Backend::Estimated;
        let _ = v.run_ui(egui::RawInput::default());
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
        v.toggle_play();
        until(&mut v, &hl, &mut painter, "a pause", |s| s.dynamics.paused && !s.working);
        // the model as it is now (the run's fallbacks, if any, were learned from)
        let m = v.snapshot.as_ref().unwrap().model.clone().unwrap();
        // away and back: the model comes back from the autosave, no new pass
        v.ui.backend = Backend::Gpu;
        let _ = v.run_ui(egui::RawInput::default());
        assert!(v.autosave.exists(), "leaving the estimated solver writes the autosave");
        settle(&mut v, &hl, &mut painter, Backend::Gpu);
        assert!(!v.density.faded, "a DFT step shows its own density");
        v.ui.backend = Backend::Estimated;
        let _ = v.run_ui(egui::RawInput::default());
        until(&mut v, &hl, &mut painter, "the restored model", |s| s.model.as_ref().is_some_and(|m| m.fitted));
        let s = v.snapshot.as_ref().unwrap();
        let back = s.model.as_ref().unwrap();
        assert!(s.dynamics.sampling.is_none(), "a pass started although the model was restored");
        assert_eq!((back.samples, back.fit, back.trusted), (m.samples, m.fit, true));
        eprintln!("estimated radio: {} samples, fit {:?}; faded label \"{label}\"; restored after GPU → estimated", m.samples, m.fit);
        let _ = std::fs::remove_file(&v.autosave);
    }
}
