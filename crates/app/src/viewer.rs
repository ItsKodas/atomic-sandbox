//! Window-independent viewer state: the worker, the local scene mirror, the latest snapshot, the
//! camera, and the chrome (`ui`). Shared by the interactive window (`main`) and the headless
//! screenshot.
use crate::gfx::Painter;
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
    /// Where the 3D scene is drawn this frame (physical pixels): the rect the chrome leaves, from the
    /// last `run_ui`. The renderer and every mouse ↔ world path use it.
    pub viewport: Viewport,
    /// Where an estimated worker keeps the model across solver switches (deleted at exit).
    pub autosave: PathBuf,
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
    last_play_log: Option<Instant>,
}

impl Viewer {
    /// The preset's scene, no worker yet: call [`Viewer::start`].
    pub fn new(preset: &str) -> Option<Viewer> {
        let scene = Scene::preset(preset)?;
        let egui_ctx = egui::Context::default();
        ui::apply_theme(&egui_ctx);
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
            egui_ctx,
            playing: false,
            drag_target: None,
            density: DensityShown::default(),
            viewport: Viewport::full(1400.0, 900.0),
            autosave: autosave_path(),
            rate: Default::default(),
            retiring: Vec::new(),
            pending_teach: None,
            motion: [None, None],
            density_at: [None, None],
            real_at: None,
            side_seq: None,
            superposition_built: None,
            edits_sent: 0,
            last_play_log: None,
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
    /// stays until the new worker publishes.
    pub fn start(&mut self, backend: Backend, gpu: Option<GpuContext>) {
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
        self.side_seq = None; // the new driver numbers its side densities afresh
        if estimated {
            let mut teach = vec![
                Command::EnsureModel {
                    autoload: None, // decided when it is sent: the autosave may still be written
                    plan: self.ui.plan.clone(),
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

    /// Estimated workers still writing their autosave on a reaper thread.
    #[cfg(test)]
    pub fn retirements_pending(&self) -> usize {
        self.retiring.len()
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
            self.ui.set_status("paused: this edit replaces the scene (time, pins and the drift history restart) — press Play to continue");
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
        self.poll_retiring();
        match ev {
            SimEvent::Snapshot(s) => {
                self.log(&s);
                match painter.show_snapshot(ctx, &s, depth) {
                    Ok(shown) => self.density_arrived(shown),
                    Err(e) => self.last_error = Some(e),
                }
                self.density.follow(&s);
                let step_density = s.dynamics.display.is_some();
                if step_density {
                    self.real_at = Some(s.dynamics.scene.nuclei.iter().map(|n| n.pos).collect());
                }
                self.show_side_density(&s, ctx, painter, depth, step_density);
                self.show_superposition(&s, ctx, painter, depth, step_density);
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

    /// The scene a frame at `now` draws (smooth motion, [`interpolate`]): the mirror with its
    /// nuclei between the last two playing snapshots — or the mirror itself when smooth motion
    /// is off, paused, or the mirror holds an edit the snapshots have not confirmed yet.
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
        if !s.atom_profiles.iter().all(|p| p.fits(n, h, sigma)) {
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
        let painter = root
            .layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("distance-guides")))
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

    /// Runs the panel for one frame and acts on its output.
    pub fn run_ui(&mut self, raw: egui::RawInput) -> egui::FullOutput {
        let mut outs = Vec::new();
        self.poll_retiring();
        let ctx = self.egui_ctx.clone();
        let measured_fs_per_s = self.measured_fs_per_s();
        let shown = self.shown_scene(Instant::now());
        let step = self.snapshot.as_ref().map_or(0, |s| s.dynamics.step);
        let density_label = self.density.label(step);
        let draft_banner = self.draft_banner();
        let mut viewport = self.viewport;
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
                    draft_banner: draft_banner.clone(),
                },
            );
            if let Some(r) = out.view {
                viewport = viewport_of(r, root.pixels_per_point());
            }
            outs.push(out);
            let pins = self.snapshot.as_ref().map_or(&[][..], |s| &s.dynamics.pins[..]);
            Self::distance_overlay(
                root,
                &self.camera,
                &shown,
                &self.tool,
                self.ui.mode,
                self.drag_target,
                pins,
                viewport,
                self.tool.dragging.then(|| {
                    let mut l = ui::pull_label(self.ui.drag_force);
                    if !self.playing {
                        l += " (pulls while playing)";
                    }
                    l
                }),
            );
        });
        self.viewport = viewport;
        // egui may run the closure more than once in a frame (discarded passes); input is
        // consumed in the first, so keeping every pass's output loses nothing and repeats nothing.
        for out in outs {
            if let Some(e) = out.error {
                self.last_error = Some(e);
            }
            if out.deselect && !self.tool.dragging {
                self.tool.selected = None;
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
            self.ui.set_status(format!(
                "solver switched to {}: the run restarts paused at t = 0 with the panel's settings",
                self.backend.label()
            ));
        }
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use render::headless::{Headless, HEADLESS_FORMAT};
    use std::time::Duration;

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
                working: false,
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
        let vp = v.viewport;
        eprintln!("viewport at 1400×900: {vp:?}");
        assert_eq!(vp.origin[0], 0.0, "nothing on the left");
        assert!((30.0..60.0).contains(&vp.origin[1]), "the top bar is above: {vp:?}");
        let right = vp.origin[0] + vp.size[0];
        let bottom = vp.origin[1] + vp.size[1];
        assert!((900.0..1100.0).contains(&right), "the dock and strip are on the right: {vp:?}");
        assert!((780.0..880.0).contains(&bottom), "the toolbar is below: {vp:?}");
        // folding the dock widens the view to the strip
        v.ui.dock_open = false;
        for _ in 0..2 {
            discard(v.run_ui(screen(1400.0, 900.0)));
        }
        assert!(v.viewport.size[0] > vp.size[0] + 250.0, "{:?}", v.viewport);
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
        let mut v = Viewer::new("h2").unwrap();
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
        let mut v = Viewer::new("h2o").unwrap();
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
}
