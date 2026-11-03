//! Detached windows, the window side (docking spec D-12, D3; user decision 6: "our own OS windows
//! for detaching, on the capture pop-out's pattern" — `capture/popout.rs`; user request: panels
//! detachable to another monitor, the second one holding Activity and Graph). Each is a real winit
//! window on the main window's GPU device with its own surface, its own egui `Context`, egui-winit
//! `State` and egui-wgpu `Renderer`; what it shows — its `egui_dock` tree of panels, drawn from the
//! same `UiState` and `PanelInput` as the main window — is the UI side's
//! (`ui::workspace::detached`, `ui::show_window`, `Viewer::run_ui_with`). `main.rs` opens, closes
//! and focuses them on the UI's `WindowRequest`s, routes their events here by `WindowId`, and
//! draws them each time the main window draws, after it. Display state only (spec D-13).
//!
//! - **Placement:** at its saved rect (outer position and inner size, physical pixels); a new one
//!   where the OS puts it, its leaf's size; the Two monitors layout's maximised on the second
//!   monitor (placed inside it first, so Windows maximises it there: its work area, not full
//!   screen). Where it is goes back to the UI as it moves (`Workspace::set_window_rect`), which the
//!   profile saves.
//! - **Frames:** its input is taken when the main window's is; a frame is drawn only while the
//!   window has a size (minimised: skipped, its texture uploads still applied). Its present mode is
//!   Mailbox when the GPU has it, so its vsync never holds the main window's frames (as the
//!   pop-out's).
use crate::ui::{theme, WinRect};
use gpu::{GpuContext, SurfaceState};
use std::sync::Arc;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

/// A new window's size when its leaf's is not known (points).
const DEFAULT_SIZE: [f32; 2] = [420.0, 620.0];

pub struct DetachedWindow {
    /// The UI side's id ("window N").
    pub id: u32,
    pub window: Arc<Window>,
    surface: SurfaceState,
    /// The surface still needs a configure that wgpu refused (a submit on the shared device raced it).
    unconfigured: bool,
    pub egui: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    title: String,
    /// The window has the keyboard focus.
    pub focused: bool,
}

impl DetachedWindow {
    /// Opens window `id` on `ctx`'s device: at `rect` (or `size` points, where the OS puts it), or
    /// maximised on the monitor `rect` names when `fill`.
    pub fn open(el: &ActiveEventLoop, ctx: &GpuContext, id: u32, title: &str, rect: Option<WinRect>, size: Option<[f32; 2]>, fill: bool) -> Result<DetachedWindow, String> {
        let mut attrs = Window::default_attributes()
            .with_title(title)
            .with_min_inner_size(winit::dpi::LogicalSize::new(240.0, 160.0))
            .with_window_icon(winit::window::Icon::from_rgba(crate::icon::rgba(64), 64, 64).ok());
        attrs = match (rect, fill) {
            (Some(m), true) => attrs
                .with_position(winit::dpi::PhysicalPosition::new(m.x + m.w as i32 / 8, m.y + m.h as i32 / 8))
                .with_inner_size(winit::dpi::PhysicalSize::new(m.w * 3 / 4, m.h * 3 / 4))
                .with_maximized(true),
            (Some(r), false) => attrs.with_position(winit::dpi::PhysicalPosition::new(r.x, r.y)).with_inner_size(winit::dpi::PhysicalSize::new(r.w, r.h)),
            (None, _) => {
                let [w, h] = size.map_or(DEFAULT_SIZE, |[w, h]| [w.max(240.0), h.max(160.0)]);
                attrs.with_inner_size(winit::dpi::LogicalSize::new(w as f64, h as f64))
            }
        };
        let window = Arc::new(el.create_window(attrs).map_err(|e| format!("the window: {e}"))?);
        #[cfg(windows)]
        crate::frame::exe_icon(&window);
        let surface = ctx.instance.create_surface(wgpu::SurfaceTarget::from(window.clone())).map_err(|e| format!("the window's surface: {e}"))?;
        let caps = surface.get_capabilities(&ctx.adapter);
        if caps.formats.is_empty() {
            return Err("the GPU cannot present to this window".into());
        }
        let px = window.inner_size();
        // the solver may be submitting on the shared device: a refused configure is retried at the next frame
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut surface = SurfaceState::new(ctx, surface, px.width.max(1), px.height.max(1));
        let mut unconfigured = pollster::block_on(scope.pop()).is_some();
        // several windows presented per frame: this one's vsync must not hold the main window's frames
        if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            surface.config.present_mode = wgpu::PresentMode::Mailbox;
            unconfigured = true;
        }
        let egui = egui::Context::default();
        crate::ui::apply_theme(&egui);
        let max_side = ctx.device.limits().max_texture_dimension_2d as usize;
        let state = egui_winit::State::new(egui.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, Some(max_side));
        let renderer = egui_wgpu::Renderer::new(&ctx.device, surface.config.format, egui_wgpu::RendererOptions { msaa_samples: 1, ..Default::default() });
        Ok(DetachedWindow { id, window, surface, unconfigured, egui, state, renderer, title: title.to_string(), focused: true })
    }

    /// Where the window is: its outer position and inner size (physical pixels).
    pub fn rect(&self) -> Option<WinRect> {
        let p = self.window.outer_position().ok()?;
        let s = self.window.inner_size();
        (s.width > 0 && s.height > 0).then_some(WinRect { x: p.x, y: p.y, w: s.width, h: s.height })
    }

    /// Feeds one of its window events to its egui state.
    pub fn on_event(&mut self, event: &WindowEvent) {
        let _ = self.state.on_window_event(&self.window, event);
    }

    /// Brings it to the front (a panel in it was asked for: Ctrl K, the Window menu).
    pub fn focus(&self) {
        if self.window.is_minimized() == Some(true) {
            self.window.set_minimized(false);
        }
        self.window.focus_window();
    }

    /// This frame's input for its egui context.
    pub fn take_input(&mut self) -> egui::RawInput {
        self.state.take_egui_input(&self.window)
    }

    /// Sets its title when it changed (its panels' names).
    pub fn set_title(&mut self, title: String) {
        if title != self.title {
            self.window.set_title(&title);
            self.title = title;
        }
    }

    /// Paints and presents its egui output (after the main window's frame was presented). The
    /// texture uploads are applied even when nothing can be drawn (minimised, a surface not ready),
    /// so no upload is lost.
    pub fn paint(&mut self, ctx: &GpuContext, out: egui::FullOutput) {
        self.state.handle_platform_output(&self.window, out.platform_output);
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(&ctx.device, &ctx.queue, *id, delta);
            }
        }
        self.draw(ctx, out.shapes, out.pixels_per_point);
        for id in &out.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    fn draw(&mut self, ctx: &GpuContext, shapes: Vec<egui::epaint::ClippedShape>, ppp: f32) {
        let px = self.window.inner_size();
        if px.width == 0 || px.height == 0 {
            return; // minimised
        }
        if self.unconfigured || (self.surface.config.width, self.surface.config.height) != (px.width, px.height) {
            if self.surface.resize(ctx, px.width, px.height).is_err() {
                return; // refused (the worker raced it): the next frame retries
            }
            self.unconfigured = false;
        }
        let frame = match self.surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.unconfigured = true;
                return;
            }
            _ => return,
        };
        let jobs = self.egui.tessellate(shapes, ppp);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [px.width, px.height], pixels_per_point: ppp };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("detached window") });
        let extra = self.renderer.update_buffers(&ctx.device, &ctx.queue, &mut enc, &jobs, &screen);
        {
            // the app's background (sRGB → linear for the clear) under the panels
            let [r, g, b, _] = egui::Rgba::from(theme::APP).to_array();
            let clear = wgpu::Color { r: r as f64, g: g as f64, b: b as f64, a: 1.0 };
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("detached window"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &jobs, &screen);
        }
        ctx.queue.submit(extra.into_iter().chain([enc.finish()]));
        self.window.pre_present_notify();
        ctx.queue.present(frame);
    }
}

/// The connected monitors' rects (physical pixels), the one holding `main` first (the UI's
/// "second monitor" is then the next).
pub fn monitors(el: &ActiveEventLoop, main: &Window) -> Vec<WinRect> {
    let rect = |m: &winit::monitor::MonitorHandle| {
        let (p, s) = (m.position(), m.size());
        WinRect { x: p.x, y: p.y, w: s.width, h: s.height }
    };
    let mut list: Vec<WinRect> = el.available_monitors().map(|m| rect(&m)).filter(|r| r.w > 0 && r.h > 0).collect();
    if let Some(at) = main.current_monitor().map(|m| rect(&m)).and_then(|r| list.iter().position(|x| *x == r)) {
        let m = list.remove(at);
        list.insert(0, m);
    }
    list
}
