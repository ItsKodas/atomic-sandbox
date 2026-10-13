//! Atomic Sandbox viewer: a winit window with an egui side panel over the sphere + density
//! renderer, driven by the simulation worker thread.
mod gfx;
mod overlay;
mod scene;
mod screenshot;
mod sim;
mod tools;
mod ui;
mod viewer;

use gfx::{Painter, Target, World};
use gpu::{GpuContext, SurfaceState};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ui::ToolMode;
use viewer::Viewer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// How often an idle window polls the worker (it has no way to wake the event loop).
const IDLE_POLL: Duration = Duration::from_millis(100);

struct Gfx {
    ctx: GpuContext,
    surface: SurfaceState,
    painter: Painter,
}

#[derive(Default)]
struct Mouse {
    pos: (f32, f32),
    /// Buttons whose press landed in the 3D view (not on the panel); only these drive tools.
    left: bool,
    right: bool,
    middle: bool,
}

struct App {
    // Never dropped in the windowed app: `main` exits the process after the event loop (see there).
    viewer: Viewer,
    /// Backend to start the worker with once the window's GPU context exists.
    backend: sim::Backend,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    egui_state: Option<egui_winit::State>,
    mouse: Mouse,
    /// egui asked for another frame (animation, tooltip).
    egui_repaint: bool,
    /// Presented-frame intervals while playing, for a stderr line every [`FRAME_LOG_EVERY`].
    frames: FrameStats,
}

/// How often the frame-interval line is printed while playing.
const FRAME_LOG_EVERY: Duration = Duration::from_secs(5);

/// Intervals between presented frames (ms) since the last report.
#[derive(Default)]
struct FrameStats {
    last: Option<Instant>,
    since: Option<Instant>,
    ms: Vec<f64>,
}

impl FrameStats {
    /// Records a presented frame; while `playing`, prints p50 / p95 / max every
    /// [`FRAME_LOG_EVERY`] (the render thread must stay smooth while the worker steps).
    fn frame(&mut self, playing: bool) {
        let now = Instant::now();
        if !playing {
            *self = FrameStats::default();
            return;
        }
        if let Some(t) = self.last {
            self.ms.push((now - t).as_secs_f64() * 1e3);
        }
        self.last = Some(now);
        let since = *self.since.get_or_insert(now);
        if now - since >= FRAME_LOG_EVERY && !self.ms.is_empty() {
            self.ms.sort_by(|a, b| a.total_cmp(b));
            let q = |f: f64| self.ms[((self.ms.len() - 1) as f64 * f) as usize];
            eprintln!(
                "frames while playing: {} in {:.1} s, interval p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms",
                self.ms.len(),
                (now - since).as_secs_f64(),
                q(0.5),
                q(0.95),
                q(1.0)
            );
            self.ms.clear();
            self.since = Some(now);
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut preset, mut shot, mut backend, mut play_ms) =
        ("h2".to_string(), None, sim::Backend::Gpu, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--preset" => preset = args.next().expect("--preset needs a name"),
            "--screenshot" => shot = Some(args.next().expect("--screenshot needs a path")),
            "--play-ms" => {
                play_ms = Some(
                    args.next()
                        .and_then(|v| v.parse::<u64>().ok())
                        .expect("--play-ms needs a number of milliseconds"),
                )
            }
            "--backend" => {
                backend = match args.next().as_deref() {
                    Some("gpu") => sim::Backend::Gpu,
                    Some("cpu") => sim::Backend::Cpu,
                    other => panic!("--backend needs gpu or cpu, got {other:?}"),
                }
            }
            other => eprintln!("ignoring unknown argument {other}"),
        }
    }
    if let Some(path) = shot {
        if let Err(e) = screenshot::run(&path, &preset, backend, play_ms) {
            eprintln!("screenshot failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    let viewer = Viewer::new(&preset)
        .unwrap_or_else(|| panic!("unknown preset {preset}; have {:?}", scene::Scene::PRESETS));
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        viewer,
        backend,
        window: None,
        gfx: None,
        egui_state: None,
        mouse: Mouse::default(),
        egui_repaint: true,
        frames: FrameStats::default(),
    };
    event_loop.run_app(&mut app).expect("run");
    // Exit without dropping `app`: `SimWorker::drop` would join the sim thread, which on the CPU
    // backend may be in the middle of a solve (minutes for H₂O) that has no cancellation point
    // (a GPU solve would stop within one SCF step). Nothing needs flushing — scenes are saved
    // explicitly from the panel — so let the OS reclaim everything.
    std::process::exit(0);
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Atomic Sandbox")
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 900.0));
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        let (ctx, surface) = GpuContext::new_for_surface(wgpu::SurfaceTarget::from(window.clone()))
            .expect("no GPU adapter");
        let size = window.inner_size();
        let surface = SurfaceState::new(&ctx, surface, size.width, size.height);
        let painter = Painter::new(&ctx, surface.config.format);
        let max_side = ctx.device.limits().max_texture_dimension_2d as usize;
        self.egui_state = Some(egui_winit::State::new(
            self.viewer.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_side),
        ));
        eprintln!(
            "window {}×{}, surface format {:?}, adapter {}, float32-filterable {}, solver {}",
            size.width,
            size.height,
            surface.config.format,
            ctx.adapter.get_info().name,
            ctx.float32_filterable,
            self.backend.label()
        );
        // The worker shares this device: the GPU backend's density textures are bound as is.
        self.viewer.start(self.backend, Some(ctx.clone()));
        self.gfx = Some(Gfx {
            ctx,
            surface,
            painter,
        });
        self.window = Some(window);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let consumed = match &mut self.egui_state {
            Some(s) => s.on_window_event(&window, &event).consumed,
            None => false,
        };
        match event {
            WindowEvent::CloseRequested => {
                // Disappear at once; `main` then exits without waiting for the solver.
                window.set_title("Atomic Sandbox — closing…");
                window.set_visible(false);
                el.exit();
            }
            WindowEvent::Resized(sz) => {
                if let Some(g) = &mut self.gfx {
                    g.surface.resize(&g.ctx, sz.width, sz.height); // no-op when minimised (0×0)
                    g.painter.rebind(&g.ctx, &g.surface.depth);
                }
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(&window),
            e => {
                self.handle_input(&e, consumed);
                window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let got = self.drain_sim();
        let animate = self.viewer.busy || self.viewer.tool.dragging || self.egui_repaint || got;
        // Not while minimised: winit posts WM_PAINT for a redraw request whether or not the
        // window is visible, and `redraw` returns at once at 0×0 without presenting, so nothing
        // (no vsync) would pace the loop and a solve would spin a core. Poll instead.
        let visible = self.window.as_ref().is_some_and(|w| {
            let s = w.inner_size();
            s.width > 0 && s.height > 0
        });
        if animate && visible {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
            el.set_control_flow(ControlFlow::Wait);
        } else {
            el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + IDLE_POLL));
        }
    }
}

impl App {
    fn drain_sim(&mut self) -> bool {
        let Some(g) = &mut self.gfx else { return false };
        self.viewer
            .drain_sim(&g.ctx, &mut g.painter, &g.surface.depth)
    }

    /// Size of the render target in physical pixels (the frame `pixel_ray` works in).
    fn view_size(&self) -> (f32, f32) {
        self.gfx.as_ref().map_or((1.0, 1.0), |g| {
            (
                g.surface.config.width as f32,
                g.surface.config.height as f32,
            )
        })
    }

    /// Mouse and keyboard for the 3D view. Presses and wheel steps egui consumed (over the panel)
    /// are ignored; releases and motion are always tracked so a drag that ends over the panel ends.
    fn handle_input(&mut self, event: &WindowEvent, consumed: bool) {
        match *event {
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x as f32, position.y as f32);
                let d = (p.0 - self.mouse.pos.0, p.1 - self.mouse.pos.1);
                self.mouse.pos = p;
                // distance guides follow the cursor only while it is over the 3D view
                self.viewer.tool.view_px = self.view_size();
                self.viewer.tool.cursor = (!consumed).then_some(p);
                self.on_motion(d);
            }
            // A button held while the cursor leaves or the window loses focus (alt-tab, a UAC
            // prompt) may never report its release: treat both as a release.
            WindowEvent::CursorLeft { .. } => {
                self.viewer.tool.cursor = None;
                self.release_all();
            }
            WindowEvent::Focused(false) => self.release_all(),
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                if down && consumed {
                    return;
                }
                match button {
                    MouseButton::Left => {
                        self.mouse.left = down;
                        if down {
                            self.on_left_press();
                        } else {
                            self.viewer.end_drag();
                        }
                    }
                    MouseButton::Right => self.mouse.right = down,
                    MouseButton::Middle => self.mouse.middle = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !consumed && !self.viewer.tool.dragging => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.viewer.zoom(lines);
            }
            WindowEvent::KeyboardInput { ref event, .. }
                if !consumed
                    && event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Delete) =>
            {
                if let Some(i) = self.viewer.tool.selected {
                    self.viewer.send(sim::Command::RemoveNucleus(i));
                }
            }
            WindowEvent::KeyboardInput { ref event, .. }
                if !consumed
                    && event.state == ElementState::Pressed
                    && !event.repeat
                    && event.logical_key == Key::Named(NamedKey::Space) =>
            {
                self.viewer.toggle_play();
            }
            _ => {}
        }
    }

    fn on_left_press(&mut self) {
        let (w, h) = self.view_size();
        let (px, py) = self.mouse.pos;
        let v = &mut self.viewer;
        match v.ui.mode {
            ToolMode::Orbit => {}
            ToolMode::Place => {
                if let Some(pos) = tools::place_point(&v.camera, px, py, w, h) {
                    let pos = tools::clamp_into_cell(pos, &v.scene);
                    v.send(sim::Command::AddNucleus {
                        z: v.ui.element,
                        pos,
                    });
                }
            }
            ToolMode::Select => {
                let hit = tools::pick(&v.camera, px, py, w, h, &v.scene.nuclei);
                v.tool.selected = hit;
                if let Some(i) = hit {
                    let plane = (tools::v3(v.scene.nuclei[i].pos), v.camera.focus_plane().1);
                    v.begin_drag(i, plane);
                }
            }
        }
    }

    /// Every button up: ends a Select-drag (`Viewer::end_drag`) and stops orbit / pan.
    fn release_all(&mut self) {
        self.mouse.left = false;
        self.mouse.right = false;
        self.mouse.middle = false;
        self.viewer.end_drag();
    }

    fn on_motion(&mut self, (dx, dy): (f32, f32)) {
        let (w, h) = self.view_size();
        let (px, py) = self.mouse.pos;
        let v = &mut self.viewer;
        if self.mouse.left {
            match (v.tool.selected, v.tool.drag_plane) {
                (Some(_), Some(plane)) if v.tool.dragging => {
                    if let Some(pos) = tools::drag_point(&v.camera, px, py, w, h, plane) {
                        v.drag_to(pos);
                    }
                }
                // Select mode on empty space orbits too; Place mode only places
                _ if v.ui.mode != ToolMode::Place => v.camera.orbit(dx, dy),
                _ => {}
            }
        } else if self.mouse.middle {
            // Middle-drag orbits in every tool mode (also while Place is active)
            v.camera.orbit(dx, dy);
        } else if self.mouse.right {
            v.camera.pan(dx, dy, h);
        }
    }

    fn redraw(&mut self, window: &Window) {
        let (Some(g), Some(egui_state)) = (&mut self.gfx, &mut self.egui_state) else {
            return;
        };
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return; // minimised
        }
        let frame = match g.surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                g.surface.resize(&g.ctx, size.width, size.height);
                g.painter.rebind(&g.ctx, &g.surface.depth);
                window.request_redraw(); // retry the skipped frame on the new surface
                return;
            }
            _ => {
                window.request_redraw(); // timeout / occluded: retry the skipped frame
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let raw = egui_state.take_egui_input(window);
        let mut out = self.viewer.run_ui(raw);
        egui_state.handle_platform_output(window, std::mem::take(&mut out.platform_output));
        self.egui_repaint = out
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|v| v.repaint_delay.is_zero());
        g.painter.set_view(&g.ctx, &self.viewer.ui);
        let v = &self.viewer;
        let target = Target {
            colour: &view,
            depth: &g.surface.depth,
            size: [g.surface.config.width, g.surface.config.height],
        };
        let lines = overlay::build(&v.ui, &v.scene, &v.camera, v.ui.mode);
        let world = World {
            camera: &v.camera,
            nuclei: &v.scene.nuclei,
            selected: v.tool.selected,
            lines: &lines,
        };
        g.painter.paint(&g.ctx, target, world, &v.egui_ctx, out);
        window.pre_present_notify();
        g.ctx.queue.present(frame);
        self.frames.frame(self.viewer.playing);
    }
}
