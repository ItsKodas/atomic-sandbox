//! The pop-out view (user request 2026-10-20): a second, real OS window — a winit window on the
//! main window's device with its own surface — showing the 3D view exactly as Screenshot / Record
//! would capture it now. It is opened and closed by `Settings::popout` (the "Pop out view" button
//! in the capture settings, or Ctrl K); closing the window clears the flag and changes nothing else.
//!
//! What it shows is made by the capture's own path, so it cannot drift from the files:
//! - **the UI hidden** (or a simulated-time recording): [`Viewer::render_capture`] into the
//!   pop-out's own offscreen frame at the capture's size ([`frame`]: the resolution and aspect, or a
//!   recording's fixed size), from the capture camera, with the capture's overlay and look (Camera
//!   view, bloom, motion blur, dots — the painter's passes; the pop-out has its own bloom, glow and
//!   overlay egui renderer, [`crate::gfx::Fx::Popout`], and its own label smoothing);
//! - **the UI shown**: the window's presented frame, copied as the capture copies it.
//!
//! The frame is drawn scaled to fit the window (letterboxed) with a strip under it naming the
//! size, the mode and the scale shown, and — while recording — the red dot and the frame count
//! (also in the title). One extra offscreen render per window frame while it is open (display
//! only); it is drawn when the main window draws, so it follows the main camera live. A
//! simulated-time recording's frames are interpolated instants; the pop-out shows the live state.
//! F11 toggles borderless full screen (the strip hidden), Esc leaves it.
use super::{camera, readback::Tag, Job, Offscreen, RecordMode, RecordingView, Settings, SurfaceFrame, View};
use crate::ui::theme;
use egui::{pos2, vec2, Align2, Color32, FontId, Rect, Stroke};
use gpu::{GpuContext, SurfaceState};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

/// The strip under the frame (points).
pub const STRIP: f32 = 24.0;

/// What a capture would take now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The 3D view alone, rendered offscreen (else the window as presented).
    pub hidden: bool,
    /// Pixels.
    pub size: (u32, u32),
}

/// The frame a capture takes now, as `Capture::plan` would: a recording's own (its hide-UI choice
/// and its size, fixed at its first frame), else the settings' — the 3D view at the resolution and
/// aspect when the UI is hidden (or the window cannot be copied), else the window at its size.
pub fn frame(settings: &Settings, cv: &View, view_px: (u32, u32), window_px: (u32, u32), max_side: u32) -> Frame {
    let hidden = match &cv.recording {
        Some(r) => r.hide_ui,
        None => settings.hide_ui || !cv.surface_copyable,
    };
    let size = match &cv.recording {
        Some(r) if r.size.0 > 0 && r.size.1 > 0 => {
            if hidden {
                camera::fit(r.size, max_side)
            } else {
                r.size
            }
        }
        _ if hidden => settings.frame_size(view_px, max_side),
        _ => window_px,
    };
    Frame { hidden, size }
}

/// Where the frame is drawn in a window of `win` points: the largest rect of its aspect above the
/// strip (or in the whole window without it).
pub fn layout(win: egui::Vec2, size: (u32, u32), strip: bool) -> Rect {
    let h = (win.y - if strip { STRIP } else { 0.0 }).max(1.0);
    let ratio = size.0.max(1) as f32 / size.1.max(1) as f32;
    let (x, y, w, h) = camera::inscribed((0.0, 0.0, win.x.max(1.0), h), ratio);
    Rect::from_min_size(pos2(x, y), vec2(w, h))
}

/// The strip's left text: "1920 × 1080 · 1080p 16:9 · 3D view only · orbit · shown at 50 %".
pub fn size_words(f: &Frame, s: &Settings, recording: bool, shown_px: f32) -> String {
    let mut parts = vec![format!("{} × {}", f.size.0, f.size.1)];
    if f.hidden {
        if !recording {
            parts.push(match (s.resolution, s.aspect) {
                (super::Resolution::View, camera::Aspect::View) => "the view's size".to_string(),
                (r, a) => format!("{} {}", r.label(), a.label()),
            });
        }
        parts.push("3D view only".into());
        match s.camera.mode {
            camera::CameraMode::View => {}
            camera::CameraMode::Fixed => parts.push("fixed camera".into()),
            camera::CameraMode::Orbit => parts.push("orbiting camera".into()),
        }
    } else {
        parts.push("the whole window".into());
    }
    let pct = 100.0 * shown_px / f.size.0.max(1) as f32;
    parts.push(format!("shown at {pct:.0} %"));
    parts.join(" · ")
}

/// The recording's words: "REC · 360 frames · 12.0 s video".
pub fn rec_words(r: &RecordingView) -> String {
    let len = match r.limit_s {
        Some(l) => format!("{:.1} / {l:.1} s video", r.playback_s),
        None => format!("{:.1} s video", r.playback_s),
    };
    match r.mode {
        RecordMode::SimTime => format!("REC · {} · {} frames · {len}", super::fs_words(r.sim_fs), r.frames),
        RecordMode::AsSeen => format!("REC · {} frames · {len}", r.frames),
    }
}

/// The strip's job line (user request 2026-10-20: the Activity tab's job card, in the strip): what
/// runs, its time left, and its bar — the scene loading (the Jobs card's words and bar), else the
/// status chip's headline (solving, sampling, a relax, an audit…), else playing / paused.
#[derive(Clone, Debug, PartialEq)]
pub struct JobLine {
    pub title: String,
    pub detail: Option<String>,
    pub bar: Option<crate::ui::activity::headline::BarMode>,
}

pub fn job_line(ui: &crate::ui::UiState, playing: bool, now: std::time::Instant) -> JobLine {
    use crate::ui::activity::headline::BarMode;
    if let Some((t0, eta, _)) = &ui.loading {
        let el = now.duration_since(*t0).as_secs_f64();
        let left = eta - el;
        return JobLine {
            title: "Loading the new scene".into(),
            detail: Some(if left > 0.5 { format!("≈ {left:.0} s left") } else { "almost done".into() }),
            bar: Some(BarMode::Fraction((el / eta.max(0.1)).min(0.97) as f32)),
        };
    }
    if let Some(h) = ui.activity.headline.peek().as_ref().filter(|h| !h.text.is_empty()) {
        return JobLine { title: h.text.clone(), detail: h.audit.clone(), bar: Some(h.bar) };
    }
    JobLine { title: if playing { "playing".into() } else { "paused".into() }, detail: None, bar: None }
}

/// The window's title.
pub fn title(f: &Frame, rec: Option<&RecordingView>) -> String {
    let base = format!("Capture view {} × {} — Atomic Sandbox", f.size.0, f.size.1);
    match rec {
        Some(r) => format!("● REC {} frames — {base}", r.frames),
        None => base,
    }
}

/// What the last prepared frame was (for the strip and the title).
struct Shown {
    frame: Frame,
    recording: Option<RecordingView>,
    settings: Settings,
    /// The app's job line ([`job_line`]).
    job: JobLine,
}

pub struct Popout {
    pub window: Arc<Window>,
    surface: SurfaceState,
    /// The surface still needs a configure that wgpu refused (a submit on the shared device raced it).
    unconfigured: bool,
    egui: egui::Context,
    renderer: egui_wgpu::Renderer,
    /// The frame (the pop-out's own offscreen target) and its egui texture.
    target: Option<Offscreen>,
    texture: Option<egui::TextureId>,
    /// The target was recreated since it was registered with egui.
    stale_texture: bool,
    /// The overlay's egui context and label smoothing (its own: the capture's run on the video clock).
    labels: super::overlay::LabelState,
    shown: Option<Shown>,
    title: String,
    titled: Option<Instant>,
    started: Instant,
    /// The window has the keyboard focus (the floating dots animate then too).
    pub focused: bool,
}

impl Popout {
    /// Opens the window (sized to the frame's aspect) and its surface on `ctx`'s device.
    pub fn open(el: &ActiveEventLoop, ctx: &GpuContext, frame: Option<Frame>) -> Result<Popout, String> {
        let ratio = frame.map_or(16.0 / 9.0, |f| f.size.0.max(1) as f32 / f.size.1.max(1) as f32);
        let (w, h) = if ratio >= 1.0 { (960.0, 960.0 / ratio) } else { (540.0 * ratio, 540.0) };
        let attrs = Window::default_attributes()
            .with_title("Capture view — Atomic Sandbox")
            .with_inner_size(winit::dpi::LogicalSize::new(w as f64, (h + STRIP) as f64))
            .with_min_inner_size(winit::dpi::LogicalSize::new(200.0, 140.0))
            .with_window_icon(winit::window::Icon::from_rgba(crate::icon::rgba(64), 64, 64).ok());
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
        let mut surface = SurfaceState::new(ctx, surface, px.width, px.height);
        let mut unconfigured = pollster::block_on(scope.pop()).is_some();
        // two windows presented per frame: this one's vsync must not hold the main window's frames
        if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            surface.config.present_mode = wgpu::PresentMode::Mailbox;
            unconfigured = true;
        }
        let egui = egui::Context::default();
        crate::ui::apply_theme(&egui);
        // dithering off: the frame's pixels are shown as they are
        let options = egui_wgpu::RendererOptions { msaa_samples: 1, dithering: false, ..Default::default() };
        let renderer = egui_wgpu::Renderer::new(&ctx.device, surface.config.format, options);
        Ok(Popout {
            window,
            surface,
            unconfigured,
            egui,
            renderer,
            target: None,
            texture: None,
            stale_texture: false,
            labels: Default::default(),
            shown: None,
            title: String::new(),
            titled: None,
            started: Instant::now(),
            focused: true,
        })
    }

    /// The frame of this window frame, after the main window's frame was painted and before it is
    /// presented: the capture's offscreen render (UI hidden) or a copy of the window's frame.
    pub fn prepare(&mut self, v: &crate::viewer::Viewer, ctx: &GpuContext, painter: &mut crate::gfx::Painter, window_depth: &wgpu::TextureView, surface: Option<SurfaceFrame<'_>>) {
        if self.minimised() {
            return;
        }
        let view_px = (v.viewport.size[0].round() as u32, v.viewport.size[1].round() as u32);
        let window_px = surface.as_ref().map_or(view_px, |s| s.size);
        let mut f = frame(&v.ui.capture, &v.ui.capture_view, view_px, window_px, v.capture_max_side(ctx));
        let copy = surface.filter(|s| !f.hidden && s.copyable && s.format == painter.format && s.size == f.size);
        if !f.hidden && copy.is_none() {
            // the window cannot be copied as it is (the capture then takes the 3D view alone)
            f = Frame { hidden: true, size: v.ui.capture.frame_size(view_px, v.capture_max_side(ctx)) };
        }
        if f.size.0 < 2 || f.size.1 < 2 {
            return;
        }
        if self.target.as_ref().is_none_or(|t| t.size != f.size || t.format != painter.format) {
            self.target = Some(Offscreen::new(ctx, f.size, painter.format));
            self.stale_texture = true;
        }
        let target = self.target.as_ref().expect("made above");
        match copy {
            Some(s) => {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pop-out copy") });
                enc.copy_texture_to_texture(
                    s.texture.as_image_copy(),
                    target.colour.as_image_copy(),
                    wgpu::Extent3d { width: f.size.0, height: f.size.1, depth_or_array_layers: 1 },
                );
                ctx.queue.submit([enc.finish()]);
            }
            None => v.render_popout(ctx, painter, target, &mut self.labels, window_depth),
        }
        let job = job_line(&v.ui, v.playing, std::time::Instant::now());
        self.shown = Some(Shown { frame: f, recording: v.ui.capture_view.recording.clone(), settings: v.ui.capture.clone(), job });
    }

    fn minimised(&self) -> bool {
        let s = self.window.inner_size();
        s.width == 0 || s.height == 0
    }

    fn fullscreen(&self) -> bool {
        self.window.fullscreen().is_some()
    }

    /// F11: borderless full screen on the window's monitor, and back.
    pub fn toggle_fullscreen(&self) {
        let to = if self.fullscreen() { None } else { Some(winit::window::Fullscreen::Borderless(None)) };
        self.window.set_fullscreen(to);
    }

    pub fn leave_fullscreen(&self) {
        if self.fullscreen() {
            self.window.set_fullscreen(None);
        }
    }

    /// Draws the prepared frame letterboxed, with its strip, and presents the window.
    pub fn present(&mut self, ctx: &GpuContext) {
        let px = self.window.inner_size();
        let Some(shown) = self.shown.as_ref() else { return };
        let Some(target) = self.target.as_ref() else { return };
        if px.width == 0 || px.height == 0 {
            return;
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
        match self.texture {
            None => self.texture = Some(self.renderer.register_native_texture(&ctx.device, &target.raw, wgpu::FilterMode::Linear)),
            Some(id) if self.stale_texture => self.renderer.update_egui_texture_from_wgpu_texture(&ctx.device, &target.raw, wgpu::FilterMode::Linear, id),
            Some(_) => {}
        }
        self.stale_texture = false;
        let tex = self.texture.expect("registered above");
        let ppp = self.window.scale_factor() as f32;
        let win = vec2(px.width as f32 / ppp, px.height as f32 / ppp);
        let mut raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), win)),
            time: Some(self.started.elapsed().as_secs_f64()),
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        let strip = !self.fullscreen();
        let rect = layout(win, shown.frame.size, strip);
        let words = size_words(&shown.frame, &shown.settings, shown.recording.is_some(), rect.width() * ppp);
        let rec = shown.recording.as_ref().map(rec_words);
        let job = shown.job.clone();
        let t = self.started.elapsed().as_secs_f64();
        let out = self.egui.run_ui(raw, |root| {
            let p = root.ctx().layer_painter(egui::LayerId::background());
            p.image(tex, rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            p.rect_stroke(rect.expand(1.0), 0.0, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Outside);
            if strip {
                let y = win.y - STRIP / 2.0;
                let left = p.text(pos2(10.0, y), Align2::LEFT_CENTER, &words, FontId::proportional(12.0), theme::INK_DIM);
                // right to left: the recording, then the app's live state (loading, solving, …)
                let mut right = win.x - 10.0;
                if let Some(r) = &rec {
                    let g = p.text(pos2(right, y), Align2::RIGHT_CENTER, r, FontId::monospace(12.0), theme::INK);
                    let on = t.fract() < 0.6;
                    let red = crate::ui::RED;
                    p.circle_filled(pos2(g.left() - 10.0, y), 5.0, if on { red } else { red.gamma_multiply(0.35) });
                    right = g.left() - 28.0;
                }
                // the job: its title (strong) and time left (dim), only when it fits beside the size
                // words (a narrow window keeps the left text whole); its bar along the strip's top edge
                let detail = job.detail.as_deref().map(|d| p.layout_no_wrap(format!("  {d}"), FontId::proportional(11.5), theme::INK_DIM));
                let title = p.layout_no_wrap(job.title.clone(), FontId::proportional(12.5), theme::INK);
                let w = title.size().x + detail.as_ref().map_or(0.0, |d| d.size().x);
                if right - w > left.right() + 24.0 {
                    let x0 = right - w;
                    let tw = title.size().x;
                    p.galley(pos2(x0, y - title.size().y / 2.0), title, theme::INK);
                    if let Some(d) = detail {
                        p.galley(pos2(x0 + tw, y - d.size().y / 2.0), d, theme::INK_DIM);
                    }
                }
                let top = win.y - STRIP;
                match job.bar {
                    Some(crate::ui::activity::headline::BarMode::Fraction(f)) => {
                        p.rect_filled(Rect::from_min_max(pos2(0.0, top), pos2(win.x, top + 3.0)), 0.0, theme::LINE);
                        p.rect_filled(Rect::from_min_max(pos2(0.0, top), pos2(win.x * f.clamp(0.0, 1.0), top + 3.0)), 0.0, theme::ACCENT);
                    }
                    Some(crate::ui::activity::headline::BarMode::Sweep) => {
                        let e = crate::ui::activity::headline::sweep_edge(crate::ui::activity::headline::sweep_phase(t));
                        let (a, b) = ((e.max(0.0)) * win.x, ((e + 0.2).min(1.0)) * win.x);
                        p.rect_filled(Rect::from_min_max(pos2(a, top), pos2(b, top + 3.0)), 0.0, theme::ACCENT);
                    }
                    None => {}
                }
            } else if rec.is_some() {
                // full screen: the dot alone, in the top-right corner
                let on = t.fract() < 0.6;
                let red = crate::ui::RED;
                p.circle_filled(pos2(win.x - 14.0, 14.0), 6.0, if on { red } else { red.gamma_multiply(0.35) });
            }
        });
        let jobs = self.egui.tessellate(out.shapes, out.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [px.width, px.height], pixels_per_point: out.pixels_per_point };
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(&ctx.device, &ctx.queue, *id, delta);
            }
        }
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pop-out") });
        let extra = self.renderer.update_buffers(&ctx.device, &ctx.queue, &mut enc, &jobs, &screen);
        {
            // the bars: the app's background (sRGB → linear for the clear)
            let [r, g, b, _] = egui::Rgba::from(theme::APP).to_array();
            let clear = wgpu::Color { r: r as f64, g: g as f64, b: b as f64, a: 1.0 };
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("pop-out"),
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
        for id in &out.textures_delta.free {
            self.renderer.free_texture(id);
        }
        self.window.pre_present_notify();
        ctx.queue.present(frame);
        // the title: a recording's frame count, a few times a second at most
        let title = title(&shown.frame, shown.recording.as_ref());
        if title != self.title && self.titled.is_none_or(|t| t.elapsed() >= Duration::from_millis(250)) {
            self.window.set_title(&title);
            self.title = title;
            self.titled = Some(Instant::now());
        }
    }
}

impl crate::viewer::Viewer {
    /// The pop-out's frame: [`Self::render_capture`] as a screenshot would take it now — or, while
    /// recording, at the recording's video clock (the orbit's phase, the time readout's rate).
    pub fn render_popout(&self, ctx: &GpuContext, painter: &mut crate::gfx::Painter, t: &Offscreen, labels: &mut super::overlay::LabelState, window_depth: &wgpu::TextureView) {
        let rec = self.ui.capture_view.recording.as_ref();
        let job = Job {
            tag: if rec.is_some() { Tag::Frame { session: 0, index: 0 } } else { Tag::Screenshot(Default::default()) },
            size: t.size,
            at: None,
            // as the view's gizmo previews it: the wall clock, or the recording's video time
            video_t: rec.map_or_else(|| self.egui_ctx.input(|i| i.time), |r| r.playback_s),
            video_fs_per_s: rec.filter(|r| r.mode == RecordMode::SimTime).map(|r| r.fs_per_frame * super::clock::FPS),
        };
        self.render_capture(ctx, painter, crate::gfx::Fx::Popout, t, &job, labels, window_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::Resolution;

    fn rec(size: (u32, u32), hide_ui: bool) -> RecordingView {
        RecordingView {
            mode: RecordMode::SimTime,
            frames: 360,
            sim_fs: 180.0,
            playback_s: 12.0,
            fs_per_frame: 0.5,
            size,
            hide_ui,
            notes: vec![],
            started: Instant::now(),
            limit_s: None,
        }
    }

    #[test]
    fn the_frame_is_what_a_capture_takes_now() {
        let view = View { surface_copyable: true, ..Default::default() };
        let s = Settings { resolution: Resolution::P1080, aspect: camera::Aspect::Wide, ..Settings::default() };
        // the UI hidden (the default): the resolution and aspect
        assert_eq!(frame(&s, &view, (800, 600), (1400, 900), 8192), Frame { hidden: true, size: (1920, 1080) });
        // within the GPU's limit, as the capture fits it
        assert_eq!(frame(&s, &view, (800, 600), (1400, 900), 1024).size, camera::fit((1920, 1080), 1024));
        // the UI shown: the window at its size
        let shown = Settings { hide_ui: false, ..s.clone() };
        assert_eq!(frame(&shown, &view, (800, 600), (1400, 900), 8192), Frame { hidden: false, size: (1400, 900) });
        // …unless the window cannot be copied
        let nocopy = View { surface_copyable: false, ..Default::default() };
        assert!(frame(&shown, &nocopy, (800, 600), (1400, 900), 8192).hidden);
        // a recording keeps its own choice and size, whatever the settings say now
        let recording = View { recording: Some(rec((1280, 720), true)), ..view.clone() };
        assert_eq!(frame(&shown, &recording, (800, 600), (1400, 900), 8192), Frame { hidden: true, size: (1280, 720) });
        // before its first frame its size is not fixed yet: the settings'
        let starting = View { recording: Some(rec((0, 0), true)), ..view };
        assert_eq!(frame(&shown, &starting, (800, 600), (1400, 900), 8192), Frame { hidden: true, size: (1920, 1080) });
    }

    #[test]
    fn the_frame_is_letterboxed_above_the_strip() {
        // a wide frame in a tall window: full width, centred vertically above the strip
        let r = layout(vec2(800.0, 800.0 + STRIP), (1920, 1080), true);
        assert!((r.width() - 800.0).abs() < 1e-3 && (r.height() - 450.0).abs() < 1e-3);
        assert!((r.center().y - 400.0).abs() < 1e-3, "{r:?}");
        // a tall frame in a wide window: full height, centred horizontally
        let r = layout(vec2(1000.0, 500.0), (1080, 1920), false);
        assert!((r.height() - 500.0).abs() < 1e-3 && (r.center().x - 500.0).abs() < 1e-3, "{r:?}");
        assert!((r.width() / r.height() - 1080.0 / 1920.0).abs() < 1e-4);
    }

    #[test]
    fn the_strip_and_the_title_name_the_frame() {
        let s = Settings { resolution: Resolution::P1080, aspect: camera::Aspect::Wide, ..Settings::default() };
        let f = Frame { hidden: true, size: (1920, 1080) };
        assert_eq!(size_words(&f, &s, false, 960.0), "1920 × 1080 · 1080p 16:9 · 3D view only · shown at 50 %");
        let w = Frame { hidden: false, size: (1400, 900) };
        assert_eq!(size_words(&w, &s, false, 1400.0), "1400 × 900 · the whole window · shown at 100 %");
        let mut orbit = s.clone();
        orbit.camera.mode = camera::CameraMode::Orbit;
        assert!(size_words(&f, &orbit, true, 480.0).contains("orbiting camera"));
        assert!(!size_words(&f, &orbit, true, 480.0).contains("1080p"), "a recording's size is its own");
        let r = rec((1920, 1080), true);
        assert_eq!(rec_words(&r), "REC · 180.0 fs · 360 frames · 12.0 s video");
        assert_eq!(title(&f, Some(&r)), "● REC 360 frames — Capture view 1920 × 1080 — Atomic Sandbox");
        assert_eq!(title(&f, None), "Capture view 1920 × 1080 — Atomic Sandbox");
    }
}
