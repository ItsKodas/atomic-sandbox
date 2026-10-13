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

/// The app's zoom floor: the camera's own 0.5 Bohr floor puts the eye inside a Ne-sized sphere.
pub const MIN_DISTANCE: f32 = 2.0;

pub struct Viewer {
    /// `None` until [`Viewer::start`] (the window has no GPU context before `resumed`).
    pub sim: Option<SimWorker>,
    /// The backend `sim` runs.
    pub backend: Backend,
    /// The render context; the GPU backend runs on a clone of it (same device).
    gpu: Option<GpuContext>,
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
        })
    }

    /// (Re)starts the worker on `backend` with the current scene; `Gpu` without a context falls
    /// back to `Cpu`. A previous worker is detached, not joined: a GPU solve stops at its next
    /// SCF step, a CPU solve in flight finishes in the background and is discarded. The
    /// displayed density stays until the new worker publishes.
    pub fn start(&mut self, backend: Backend, gpu: Option<GpuContext>) {
        if let Some(old) = self.sim.take() {
            old.detach();
        }
        let backend = if gpu.is_none() { Backend::Cpu } else { backend };
        self.sim = Some(SimWorker::spawn(self.scene.clone(), backend, gpu.clone()));
        self.gpu = gpu;
        self.backend = backend;
        self.ui.backend = backend;
        self.ui.gpu_available = self.gpu.is_some();
        self.busy = true; // the worker solves the scene at once
    }

    /// Sends a command and applies it to the local mirror. Index-shifting or whole-scene
    /// commands invalidate the selection.
    pub fn send(&mut self, cmd: Command) {
        if matches!(cmd, Command::SetScene(_) | Command::RemoveNucleus(_)) {
            self.tool = ToolState::default();
        }
        cmd.clone().apply(&mut self.scene);
        if let Some(sim) = &self.sim {
            sim.send(cmd);
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
                eprintln!(
                    "snapshot #{} ({}): {} nuclei, {}³ × {} Bohr, {} values, ρmax {:.4}, E {:+.6} Ha, {} SCF steps, {} ms",
                    s.seq, s.backend.label(), s.scene.nuclei.len(), s.grid_n, s.grid_h, s.density.len(), s.density_max, s.energy.total, s.iterations, s.solve_ms
                );
                match painter.show_snapshot(ctx, &s, depth) {
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
        if self.ui.backend != self.backend {
            self.start(self.ui.backend, self.gpu.clone());
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
        v.ui.backend = Backend::Gpu;
        let _ = v.run_ui(egui::RawInput::default());
        let gpu_again = settle(&mut v, &hl, &mut painter, Backend::Gpu);
        eprintln!("centre brightness: GPU {gpu_lit}, CPU {cpu_lit}, GPU again {gpu_again}");
        for b in [gpu_lit, cpu_lit, gpu_again] {
            assert!(b > 150, "density not drawn: centre brightness {b}");
        }
    }
}
