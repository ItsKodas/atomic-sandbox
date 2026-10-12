//! Window-independent viewer state: the worker, the local scene mirror, the latest snapshot, the
//! camera, and the panel. Shared by the interactive window (`main`) and the headless screenshot.
use crate::gfx::Painter;
use crate::scene::Scene;
use crate::sim::{Command, Event as SimEvent, SimWorker, Snapshot};
use crate::tools::{self, ToolState};
use crate::ui::{self, UiState};
use glam::Vec3;
use gpu::GpuContext;
use render::camera::OrbitCamera;

/// The app's zoom floor: the camera's own 0.5 Bohr floor puts the eye inside a Ne-sized sphere.
pub const MIN_DISTANCE: f32 = 2.0;

pub struct Viewer {
    pub sim: SimWorker,
    /// The scene as the user intends it: every command sent to the worker is applied here too
    /// (through the same `Command::apply`), so spheres move at once while the density catches up.
    /// The snapshot carries the scene its density belongs to.
    pub scene: Scene,
    pub snapshot: Option<Snapshot>,
    pub camera: OrbitCamera,
    pub ui: UiState,
    pub tool: ToolState,
    pub busy: bool,
    pub last_error: Option<String>,
    pub egui_ctx: egui::Context,
}

impl Viewer {
    pub fn new(preset: &str) -> Option<Viewer> {
        let scene = Scene::preset(preset)?;
        Some(Viewer {
            sim: SimWorker::spawn(scene.clone()),
            ui: UiState::new(preset, &scene),
            scene,
            snapshot: None,
            camera: OrbitCamera::new(Vec3::ZERO, 14.0),
            tool: ToolState::default(),
            busy: true, // the worker solves the initial scene at once
            last_error: None,
            egui_ctx: egui::Context::default(),
        })
    }

    /// Sends a command and applies it to the local mirror. Index-shifting or whole-scene
    /// commands invalidate the selection.
    pub fn send(&mut self, cmd: Command) {
        if matches!(cmd, Command::SetScene(_) | Command::RemoveNucleus(_)) {
            self.tool = ToolState::default();
        }
        cmd.clone().apply(&mut self.scene);
        self.sim.send(cmd);
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
        while let Some(ev) = self.sim.try_recv() {
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
                eprintln!(
                    "snapshot #{}: {} nuclei, {}³ × {} Bohr, {} values, ρmax {:.4}, E {:+.6} Ha, {} SCF steps, {} ms",
                    s.seq, s.scene.nuclei.len(), s.grid_n, s.grid_h, s.density.len(), s.density_max, s.energy.total, s.iterations, s.solve_ms
                );
                match painter.upload_snapshot(ctx, &s, depth) {
                    Ok(()) => {
                        self.last_error = None;
                        self.snapshot = Some(s);
                    }
                    Err(e) => self.last_error = Some(e),
                }
            }
            SimEvent::Error { seq, message } => {
                eprintln!("solve #{seq} failed: {message}");
                self.last_error = Some(message);
            }
            SimEvent::Busy(b) => self.busy = b,
        }
    }

    /// Runs the panel for one frame and acts on its output.
    pub fn run_ui(&mut self, raw: egui::RawInput) -> egui::FullOutput {
        let mut outs = Vec::new();
        let full = self.egui_ctx.run_ui(raw, |root| {
            outs.push(ui::side_panel(
                root,
                &mut self.ui,
                &self.scene,
                self.snapshot.as_ref(),
                self.busy,
                self.last_error.as_deref(),
                self.tool.selected,
            ));
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
            for cmd in out.commands {
                self.send(cmd);
            }
        }
        full
    }
}
