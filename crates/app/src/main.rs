//! Atomic Sandbox viewer: a winit window with an egui side panel over the sphere + density
//! renderer, driven by the simulation worker thread.
mod gfx;
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
    // Dropped first: `SimWorker::drop` waits for an in-flight solve while the window (titled
    // "closing…") is still up.
    viewer: Viewer,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    egui_state: Option<egui_winit::State>,
    mouse: Mouse,
    /// egui asked for another frame (animation, tooltip).
    egui_repaint: bool,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut preset, mut shot) = ("h2".to_string(), None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--preset" => preset = args.next().expect("--preset needs a name"),
            "--screenshot" => shot = Some(args.next().expect("--screenshot needs a path")),
            other => eprintln!("ignoring unknown argument {other}"),
        }
    }
    if let Some(path) = shot {
        if let Err(e) = screenshot::run(&path, &preset) {
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
        window: None,
        gfx: None,
        egui_state: None,
        mouse: Mouse::default(),
        egui_repaint: true,
    };
    event_loop.run_app(&mut app).expect("run");
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
            "window {}×{}, surface format {:?}, adapter {}",
            size.width,
            size.height,
            surface.config.format,
            ctx.adapter.get_info().name
        );
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
                window.set_title("Atomic Sandbox — closing…");
                el.exit();
            }
            WindowEvent::Resized(sz) => {
                eprintln!("resized to {}×{}", sz.width, sz.height);
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
        if animate {
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
                self.on_motion(d);
            }
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
                            self.viewer.tool.dragging = false;
                            self.viewer.tool.drag_plane = None;
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
                    v.tool.dragging = true;
                    v.tool.drag_plane =
                        Some((tools::v3(v.scene.nuclei[i].pos), v.camera.focus_plane().1));
                }
            }
        }
    }

    fn on_motion(&mut self, (dx, dy): (f32, f32)) {
        let (w, h) = self.view_size();
        let (px, py) = self.mouse.pos;
        let v = &mut self.viewer;
        if self.mouse.left {
            match (v.tool.selected, v.tool.drag_plane) {
                (Some(index), Some(plane)) if v.tool.dragging => {
                    if let Some(pos) = tools::drag_point(&v.camera, px, py, w, h, plane) {
                        v.send(sim::Command::MoveNucleus { index, pos });
                    }
                }
                // Select mode on empty space orbits too; Place mode only places
                _ if v.ui.mode != ToolMode::Place => v.camera.orbit(dx, dy),
                _ => {}
            }
        } else if self.mouse.right || self.mouse.middle {
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
                return;
            }
            _ => return, // timeout / occluded: try again next frame
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
        let world = World {
            camera: &v.camera,
            nuclei: &v.scene.nuclei,
            selected: v.tool.selected,
        };
        g.painter.paint(&g.ctx, target, world, &v.egui_ctx, out);
        window.pre_present_notify();
        g.ctx.queue.present(frame);
    }
}
