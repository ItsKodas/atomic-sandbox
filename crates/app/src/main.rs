//! Atomic Sandbox viewer: a winit window with the egui chrome (top menus, bottom toolbar, right
//! dock: `ui`) around the sphere + density renderer, driven by the simulation worker thread.
mod atom_card;
mod badge;
mod cloud;
mod gfx;
mod names;
mod overlay;
mod replay;
mod scene;
mod screenshot;
mod sim;
mod tools;
mod ui;
mod viewer;
mod worker;

use gfx::{Painter, Target, World};
use gpu::{GpuContext, SurfaceState};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use viewer::Viewer;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// How often an idle window polls the worker (it has no way to wake the event loop).
const IDLE_POLL: Duration = Duration::from_millis(100);

struct Gfx {
    ctx: GpuContext,
    surface: SurfaceState,
    painter: Painter,
}

impl Gfx {
    /// Reconfigures the surface to `w × h` and rebinds the painter to the new depth view.
    /// Returns false when wgpu refused the reconfigure (the sim worker submitted on the shared
    /// device while `configure` waited for it to go idle; see `SurfaceState::resize`): the old
    /// surface stays, nothing is drawn this frame, and `redraw` retries next frame because
    /// `surface.config` then differs from the window size.
    fn resize(&mut self, w: u32, h: u32) -> bool {
        match self.surface.resize(&self.ctx, w, h) {
            Ok(()) => {
                self.painter.rebind(&self.ctx, &self.surface.depth);
                true
            }
            Err(e) => {
                eprintln!("surface: reconfigure to {w}×{h} deferred to the next frame: {e}");
                false
            }
        }
    }
}

struct App {
    // Never dropped in the windowed app: `main` exits the process after the event loop (see there).
    viewer: Viewer,
    /// Backend to start the worker with once the window's GPU context exists.
    backend: sim::Backend,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    egui_state: Option<egui_winit::State>,
    /// `--record <path>`: the session's script (`replay::Recorder`), opened when the window exists
    /// (its first line is the window's size); the worker's idle edges write `wait idle`.
    record: Option<Record>,
    /// egui asked for another frame (animation, tooltip).
    egui_repaint: bool,
    /// Presented-frame intervals while playing, for a stderr line every [`FRAME_LOG_EVERY`].
    frames: FrameStats,
}

/// `--record`: the path until the window opens, then the recorder, and whether the worker was idle
/// at the last check (`replay::idle`; an idle edge writes `wait idle`).
struct Record {
    path: PathBuf,
    /// What the window started from, written as the script's first lines (the replayer starts
    /// from `--preset`/`--backend`, h2 on the GPU by default): `--scene` as a `scene` line (an
    /// absolute path), the preset otherwise, then the solver, speed, audit cadence, walls, draft
    /// mode and cloud refresh when not the defaults.
    start: Vec<ui::Action>,
    scene: Option<String>,
    recorder: Option<replay::Recorder>,
    was_idle: bool,
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

/// Hidden start-up settings for checking the M5b viewer extras without clicking (window and
/// `--screenshot` alike): `--scene <file>` (a scene file instead of the preset), `--walls <Bohr>`
/// (container walls on at that half-width, clamped to the cell's maximum), `--draft` (the
/// estimated solver's draft mode), `--cloud-refresh <s>` (a display refresh every s seconds).
#[derive(Default)]
struct StartFlags {
    scene: Option<String>,
    walls: Option<f64>,
    draft: bool,
    cloud_refresh: Option<f64>,
}

impl StartFlags {
    /// Applies the flags to a viewer before its worker starts.
    fn apply(&self, v: &mut Viewer) -> Result<(), String> {
        if let Some(path) = &self.scene {
            let text = std::fs::read_to_string(path).map_err(|e| format!("--scene {path}: {e}"))?;
            let s = scene::Scene::parse(&text).map_err(|e| format!("--scene {path}: {e}"))?;
            s.validate().map_err(|e| format!("--scene {path}: {e}"))?;
            v.ui.grid_n = s.grid_n;
            v.ui.grid_h = s.grid_h;
            v.scene = s;
        }
        if let Some(l) = self.walls {
            v.ui.walls_on = true;
            v.ui.wall_half_width = l.min(ui::max_wall_half_width(&v.scene));
        }
        v.ui.draft = self.draft;
        if let Some(t) = self.cloud_refresh {
            v.ui.cloud_refresh = true;
            v.ui.cloud_refresh_s = t;
        }
        Ok(())
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut preset, mut shot, mut backend, mut play_ms, mut sample, mut audit, mut speed) =
        ("h2".to_string(), None, sim::Backend::Gpu, None, false, None, None);
    // screenshot only: the frame size and the chrome's state (`screenshot::ShotUi`)
    let (mut size, mut shot_ui, mut shot_ui_given) = (None, screenshot::ShotUi::default(), false);
    let mut flags = StartFlags::default();
    // input replay (M6 spec §8): record the window's session, or run a script headlessly
    let (mut record, mut replay_path) = (None, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--preset" => preset = args.next().expect("--preset needs a name"),
            "--record" => record = Some(PathBuf::from(args.next().expect("--record needs a path"))),
            "--replay" => replay_path = Some(PathBuf::from(args.next().expect("--replay needs a path"))),
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
                    Some("estimated") => sim::Backend::Estimated,
                    other => panic!("--backend needs gpu, cpu or estimated, got {other:?}"),
                }
            }
            // the estimated solver (M5): a surrogate taught by the GPU solver
            "--estimated" => backend = sim::Backend::Estimated,
            // screenshot only: wait for the estimated solver's sampling pass before playing
            "--sample" => sample = true,
            // the steps-per-tick slider's starting value (window)
            "--speed" => {
                speed = Some(
                    args.next()
                        .and_then(|v| v.parse::<u32>().ok())
                        .expect("--speed needs a number of steps per tick"),
                )
            }
            // the estimated solver's DFT audit cadence (default 2000, ruling S14; 0 = never)
            "--audit" => {
                audit = Some(
                    args.next()
                        .and_then(|v| v.parse::<usize>().ok())
                        .expect("--audit needs a number of answered steps"),
                )
            }
            "--size" => {
                size = args
                    .next()
                    .and_then(|v| v.split_once('x').and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))));
                assert!(size.is_some(), "--size needs <width>x<height>");
            }
            "--ui" => {
                shot_ui_given = true;
                shot_ui = screenshot::ShotUi::parse(&args.next().expect("--ui needs \"key=value;…\""))
                    .unwrap_or_else(|e| panic!("{e}"))
            }
            "--scene" => flags.scene = Some(args.next().expect("--scene needs a file")),
            "--walls" => {
                flags.walls = Some(
                    args.next()
                        .and_then(|v| v.parse::<f64>().ok())
                        .expect("--walls needs a half-width in Bohr"),
                )
            }
            "--draft" => flags.draft = true,
            "--cloud-refresh" => {
                flags.cloud_refresh = Some(
                    args.next()
                        .and_then(|v| v.parse::<f64>().ok())
                        .expect("--cloud-refresh needs seconds"),
                )
            }
            other => eprintln!("ignoring unknown argument {other}"),
        }
    }
    if let Some(path) = replay_path {
        // headless; `--screenshot` then names the last frame's PNG
        for (given, flag) in [(sample, "--sample"), (play_ms.is_some(), "--play-ms"), (shot_ui_given, "--ui")] {
            if given {
                eprintln!("--replay ignores {flag} (a screenshot-only flag): the script decides what runs");
            }
        }
        let options = replay::Options { preset, backend, screenshot: shot, size };
        let setup = |v: &mut Viewer| {
            flags.apply(v)?;
            if let Some(n) = audit {
                v.ui.audit_every = n;
            }
            if let Some(n) = speed {
                v.ui.speed = n.clamp(1, ::sim::dynamics::MAX_SPEED);
            }
            Ok(())
        };
        if let Err(e) = replay::run_with(&path, &options, &setup) {
            eprintln!("replay failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(path) = shot {
        let options = screenshot::ShotOptions { preset, backend, play_ms, sample, audit, size, ui: shot_ui };
        if let Err(e) = screenshot::run(&path, &options, &|v| flags.apply(v)) {
            eprintln!("screenshot failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    let mut viewer = Viewer::new(&preset)
        .unwrap_or_else(|| panic!("unknown preset {preset}; have {:?}", scene::Scene::PRESETS));
    if let Err(e) = flags.apply(&mut viewer) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    if let Some(n) = audit {
        viewer.ui.audit_every = n;
    }
    if let Some(n) = speed {
        viewer.ui.speed = n.clamp(1, ::sim::dynamics::MAX_SPEED);
    }
    let record = record.map(|path| {
        let u = &viewer.ui;
        // the scene file is not a preset: its own `scene` line (written first, below)
        let mut start = if flags.scene.is_some() { vec![] } else { vec![ui::Action::Preset(u.preset.clone())] };
        if backend != sim::Backend::Gpu {
            start.push(ui::Action::Backend(backend));
        }
        if u.speed != 1 {
            start.push(ui::Action::SetSpeed(u.speed));
        }
        if u.audit_every != providers::FallbackPolicy::default().audit_every {
            start.push(ui::Action::SetAudit(u.audit_every));
        }
        if u.walls_on {
            start.push(ui::Action::Walls { on: true, half_width: u.wall_half_width });
        }
        if u.draft {
            start.push(ui::Action::Draft(true));
        }
        if u.cloud_refresh {
            start.push(ui::Action::Cloud { superposition: u.cloud_superposition, refresh: true, refresh_s: u.cloud_refresh_s });
        }
        let scene = flags.scene.as_ref().map(|f| std::path::absolute(f).map_or(f.clone(), |p| p.display().to_string()));
        Record { path, start, scene, recorder: None, was_idle: false }
    });
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        viewer,
        backend,
        window: None,
        gfx: None,
        egui_state: None,
        record,
        egui_repaint: true,
        frames: FrameStats::default(),
    };
    event_loop.run_app(&mut app).expect("run");
    // the estimated solver's session autosave is for this session only
    let _ = std::fs::remove_file(&app.viewer.autosave);
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
        if let Some(r) = &mut self.record {
            match replay::Recorder::new(&r.path, (size.width, size.height)) {
                Ok(mut rec) => {
                    eprintln!("recording input to {}", r.path.display());
                    if let Some(scene) = &r.scene {
                        rec.scene(scene);
                    }
                    for a in &r.start {
                        rec.action(a);
                    }
                    r.recorder = Some(rec);
                }
                Err(e) => eprintln!("--record {}: {e}; not recording", r.path.display()),
            }
        }
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
                    g.resize(sz.width, sz.height); // no-op when minimised (0×0); a refusal retries in redraw
                }
                // the viewer re-estimates its viewport, so a cursor before the next frame maps right
                let ev = replay::Event::Resize { w: sz.width, h: sz.height };
                self.record_event(&ev, consumed);
                self.viewer.input(&ev, consumed);
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(&window),
            e => {
                // the 3D view's mouse and keys: one path, `Viewer::input` (the replayer's too)
                if let Some(ev) = replay::from_winit(&e, self.viewer.viewport) {
                    self.record_event(&ev, consumed);
                    self.viewer.input(&ev, consumed);
                }
                window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let got = self.drain_sim();
        // `--record`: a `wait idle` each time the worker goes idle after something was sent
        if let Some(r) = &mut self.record {
            let idle = replay::idle(&self.viewer);
            if idle && !r.was_idle {
                if let Some(rec) = &mut r.recorder {
                    rec.idle();
                }
            }
            r.was_idle = idle;
        }
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

    /// `--record`: the window event (converted once, `replay::from_winit`), when the 3D view acts
    /// on it (`replay::recordable`).
    fn record_event(&mut self, ev: &replay::Event, consumed: bool) {
        let Some(rec) = self.record.as_mut().and_then(|r| r.recorder.as_mut()) else { return };
        if replay::recordable(ev, consumed) {
            rec.event(ev);
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
        // A Resized whose reconfigure was refused (worker race) left `config` at the old size.
        if (g.surface.config.width, g.surface.config.height) != (size.width, size.height)
            && !g.resize(size.width, size.height)
        {
            window.request_redraw(); // still refused: skip this frame, try again
            return;
        }
        let frame = match g.surface.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                g.resize(size.width, size.height); // a refusal is retried on the next redraw too
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
        if let Some(rec) = self.record.as_mut().and_then(|r| r.recorder.as_mut()) {
            for a in &self.viewer.chrome_actions {
                rec.action(a);
            }
        }
        egui_state.handle_platform_output(window, std::mem::take(&mut out.platform_output));
        self.egui_repaint = out
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|v| v.repaint_delay.is_zero());
        g.painter.set_view(&g.ctx, &self.viewer.ui, self.viewer.density.fade(), self.viewer.density_mix(Instant::now()));
        let v = &self.viewer;
        let target = Target {
            colour: &view,
            depth: &g.surface.depth,
            size: [g.surface.config.width, g.surface.config.height],
        };
        let shown = v.shown_scene(Instant::now());
        let lines = v.overlay_lines(&shown);
        let risk = v.atom_risk();
        let world = World {
            camera: &v.camera,
            nuclei: &shown.nuclei,
            selected: v.tool.selected,
            lines: &lines,
            viewport: v.viewport,
            risk: &risk,
        };
        g.painter.paint(&g.ctx, target, world, &v.egui_ctx, out);
        window.pre_present_notify();
        g.ctx.queue.present(frame);
        self.frames.frame(self.viewer.playing);
    }
}
