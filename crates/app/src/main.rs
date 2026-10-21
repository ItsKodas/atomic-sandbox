//! Atomic Sandbox viewer: a winit window with the egui chrome (top menus, bottom toolbar, right
//! dock: `ui`) around the sphere + density renderer, driven by the simulation worker thread.
// A windowed program on Windows: no console window beside it, and the taskbar shows its icon (a
// console program's window got the generic one). Started from a terminal it attaches to that
// terminal's console (`attach_console`), so the stderr lines and `--screenshot` messages still show.
#![cfg_attr(not(test), windows_subsystem = "windows")]
mod atom_card;
mod badge;
mod builder;
mod capture;
mod cloud;
mod frame;
mod icon;
mod timings;
mod gfx;
mod ghost;
mod names;
mod overlay;
mod progress;
mod quality;
mod reference;
mod replay;
mod scene;
mod screenshot;
mod sim;
mod tools;
mod trails;
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
    /// egui asked for a frame later (a tooltip's hover delay, a blinking cursor): when.
    egui_repaint_at: Option<Instant>,
    /// The custom frame's close button was pressed: exit at the next `about_to_wait`.
    closing: bool,
    /// The window has the keyboard focus: the floating dots animate only then (a background window
    /// draws nothing new).
    focused: bool,
    /// Presented-frame intervals while playing, for a stderr line every [`FRAME_LOG_EVERY`].
    frames: FrameStats,
    /// The pop-out view's window (`capture::popout`), while `ui.capture.popout` asks for it.
    popout: Option<capture::popout::Popout>,
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
/// (the container walls' half-width, clamped to the cell's maximum — they are on by default at
/// `ui::default_wall_half_width`), `--no-walls` (start with the walls off), `--quality
/// fast|balanced|accurate|precise` (the preset's grid spacing, box kept — `quality`; a `--scene`
/// file keeps its own grid; `--backend cpu` without either loads Balanced), `--draft` (the
/// estimated solver's draft mode), `--cloud-refresh <s>` (a display refresh every s seconds),
/// `--ionic none|cores|accepted|all` (M7: the ionic policy the presets start with; the app's default is
/// `ui::DEFAULT_IONIC` = accepted species; a `--scene` file keeps its own `ionic` key).
#[derive(Default)]
struct StartFlags {
    ionic: Option<::sim::IonicPolicy>,
    scene: Option<String>,
    walls: Option<f64>,
    no_walls: bool,
    quality: Option<f64>,
    draft: bool,
    cloud_refresh: Option<f64>,
}

impl StartFlags {
    /// `--backend cpu` without `--quality` or `--scene`: Balanced (an 81³ CPU step is ≈ 40 s and
    /// cannot be cancelled). Other backends keep the default (Accurate).
    fn cpu_default_quality(&mut self, backend: sim::Backend) {
        if backend == sim::Backend::Cpu && self.quality.is_none() && self.scene.is_none() {
            self.quality = quality::parse("balanced");
        }
    }

    /// Applies the flags to a viewer before its worker starts.
    fn apply(&self, v: &mut Viewer) -> Result<(), String> {
        // the policy first: the preset below is built under it (a scene file replaces it with its own)
        if let Some(p) = self.ionic {
            v.ui.ionic = p;
            v.scene.ionic = p;
            if self.scene.is_some() {
                eprintln!("--ionic {} is ignored for the --scene file, which keeps its own `ionic` line (none when it has none)", p.word());
            }
        }
        if let Some(path) = &self.scene {
            let text = std::fs::read_to_string(path).map_err(|e| format!("--scene {path}: {e}"))?;
            let s = scene::Scene::parse(&text).map_err(|e| format!("--scene {path}: {e}"))?;
            s.validate().map_err(|e| format!("--scene {path}: {e}"))?;
            // the chip names the file, Reset and Save / Load point at it, and the Custom slider
            // shows the file's spacing (its grid beats any --quality) — as File ▸ Load does
            v.ui.adopt_file(path, &s);
            v.scene = s;
        }
        if let Some(h) = self.quality.filter(|_| self.scene.is_none()) {
            // the quality applies to the preset; an explicit grid (a scene file) beats it
            v.ui.quality_h = h;
            v.ui.quality_slider = h;
            let name = v.ui.source.label();
            let mut s = quality::preset(&name, h).ok_or_else(|| format!("--quality: unknown preset {name}"))?;
            s.ionic = v.ui.ionic;
            let box_len = v.ui.box_len;
            v.ui.adopt(&s);
            v.ui.box_len = box_len;
            v.scene = s;
        }
        if self.no_walls {
            v.ui.walls_on = false;
        } else if let Some(l) = self.walls {
            v.ui.walls_on = true;
            v.ui.wall_half_width = l.min(ui::max_wall_half_width(&v.scene));
        } else if self.scene.is_some() || self.quality.is_some() {
            // on by default, at the default half-width of the cell actually loaded
            v.ui.wall_half_width = ui::default_wall_half_width(&v.scene);
        }
        v.ui.draft = self.draft;
        if let Some(t) = self.cloud_refresh {
            v.ui.cloud_refresh = true;
            v.ui.cloud_refresh_s = t;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--scene` loads the file the way File ▸ Load does (final review app I1/I3): the chip
    /// names it, Save / Load point at it, its spacing is the quality, the walls follow its cell.
    #[test]
    fn the_scene_flag_adopts_the_file() {
        let dir = std::env::temp_dir().join(format!("atomic-scene-flag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bar.scene").display().to_string();
        let file = scene::Scene::preset("h2o").unwrap(); // 57³ × 0.25
        std::fs::write(&path, file.to_text()).unwrap();
        let mut v = Viewer::new("h2").unwrap();
        StartFlags { scene: Some(path.clone()), ..Default::default() }.apply(&mut v).unwrap();
        assert_eq!(v.scene, file);
        assert_eq!((v.ui.source.label().as_str(), v.ui.source.preset(), v.ui.scene_path.as_str()), ("bar.scene", None, path.as_str()));
        assert_eq!((v.ui.quality_h, v.ui.quality_slider), (0.25, 0.25));
        assert_eq!(v.ui.wall_half_width, ui::default_wall_half_width(&file));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `--quality` reloads the preset at that spacing with its box kept (and the walls' default
    /// for that cell); the level names parse.
    #[test]
    fn the_quality_flag_regrids_the_preset() {
        assert_eq!(quality::parse("precise"), Some(0.10));
        let mut v = Viewer::new("h2").unwrap();
        StartFlags { quality: quality::parse("fast"), ..Default::default() }.apply(&mut v).unwrap();
        assert_eq!((v.scene.grid_n, v.scene.grid_h, v.ui.grid_n, v.ui.quality_h), (41, 0.30, 41, 0.30));
        assert_eq!((v.ui.box_len, v.ui.wall_half_width), (12.0, 4.5));
        let mut w = Viewer::new("h2o").unwrap();
        StartFlags { quality: Some(0.25), ..Default::default() }.apply(&mut w).unwrap();
        assert_eq!((w.scene.grid_n, w.scene.grid_h), (57, 0.25), "h2o's own 14 Bohr box");
        assert!((w.ui.wall_half_width - 5.5).abs() < 1e-12, "the walls follow the cell");
    }

    /// M7 Task 9: the app starts its presets under `ui::DEFAULT_IONIC` (accepted species since Task 11); `--ionic` overrides it for the
    /// preset (and a later `--quality` regrid keeps it); a `--scene` file keeps its own `ionic` line.
    #[test]
    fn the_ionic_flag_overrides_the_default_policy_and_a_scene_file_keeps_its_own() {
        let v = Viewer::new("h2o").unwrap();
        assert_eq!((v.scene.ionic, v.ui.ionic), (ui::DEFAULT_IONIC, ui::DEFAULT_IONIC));
        assert_eq!(ui::DEFAULT_IONIC, ::sim::IonicPolicy::Accepted, "M7 Task 11 fix round: accepted species by default");
        for p in ::sim::IonicPolicy::ALL {
            let mut w = Viewer::new("h2o").unwrap();
            StartFlags { ionic: Some(p), quality: Some(0.25), ..Default::default() }.apply(&mut w).unwrap();
            assert_eq!((w.scene.ionic, w.ui.ionic), (p, p), "--ionic {}", p.word());
            assert_eq!(w.scene.grid_h, 0.25, "the regrid kept the policy");
        }
        let dir = std::env::temp_dir().join(format!("ionic-flag-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.scene");
        std::fs::write(&path, "grid 33 0.25
ionic all
nucleus 1 0.7 0 0
nucleus 1 -0.7 0 0
").unwrap();
        let mut f = Viewer::new("h2").unwrap();
        StartFlags { ionic: Some(::sim::IonicPolicy::None), scene: Some(path.display().to_string()), ..Default::default() }.apply(&mut f).unwrap();
        assert_eq!((f.scene.ionic, f.ui.ionic), (::sim::IonicPolicy::All, ::sim::IonicPolicy::All), "the file's key wins");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `--backend cpu` alone loads Balanced; with `--quality` or a scene file, or on the GPU, the
    /// choice stands. A `--scene` file keeps its grid (over `--quality`) and the Custom slider
    /// shows its spacing.
    #[test]
    fn the_cpu_backend_starts_at_balanced_and_a_scene_file_keeps_its_grid() {
        let mut f = StartFlags::default();
        f.cpu_default_quality(sim::Backend::Cpu);
        assert_eq!(f.quality, Some(0.25));
        let mut f = StartFlags { quality: Some(0.10), ..Default::default() };
        f.cpu_default_quality(sim::Backend::Cpu);
        assert_eq!(f.quality, Some(0.10));
        let mut f = StartFlags::default();
        f.cpu_default_quality(sim::Backend::Gpu);
        assert_eq!(f.quality, None);
        let path = std::env::temp_dir().join(format!("atomic-flags-{}.scene", std::process::id()));
        std::fs::write(&path, "grid 61 0.2
charge 0
nucleus 1 0 0 0
").unwrap();
        let mut f = StartFlags { scene: Some(path.display().to_string()), quality: Some(0.15), ..Default::default() };
        f.cpu_default_quality(sim::Backend::Cpu);
        let mut v = Viewer::new("h2").unwrap();
        f.apply(&mut v).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!((v.scene.grid_n, v.scene.grid_h, v.ui.quality_h), (61, 0.2, 0.2));
    }
}

/// Windows: print into the terminal that started us, when there is one and stderr is not already
/// redirected (a pipe or file stays as it is).
#[cfg(windows)]
fn attach_console() {
    use windows_sys::Win32::System::Console::{AttachConsole, GetStdHandle, ATTACH_PARENT_PROCESS, STD_ERROR_HANDLE};
    unsafe {
        let h = GetStdHandle(STD_ERROR_HANDLE);
        if h.is_null() || h == windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE {
            AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

fn main() {
    #[cfg(windows)]
    attach_console();
    // our own taskbar identity: Windows then takes the button's icon from the window (`frame::exe_icon`)
    // instead of resolving one for the exe path (it showed the generic icon)
    #[cfg(windows)]
    unsafe {
        let id: Vec<u16> = "ItsKodas.AtomicSandbox".encode_utf16().chain(std::iter::once(0)).collect();
        windows_sys::Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID(id.as_ptr());
    }
    // the dimer gate's derived verdicts are kept per species identity in the cache directory, so pseudo-H/He cost their
    // minute once per recipe, not per launch (M7 Task 11 fix round; `providers::pseudo` "The disk cache")
    providers::pseudo::set_disk_cache(providers::pseudo::default_disk_cache_dir());
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
            "--no-walls" => flags.no_walls = true,
            "--quality" => {
                let q = args.next().expect("--quality needs fast, balanced, accurate or precise");
                flags.quality = Some(quality::parse(&q).unwrap_or_else(|| panic!("--quality needs fast, balanced, accurate or precise, got {q:?}")));
            }
            "--draft" => flags.draft = true,
            "--ionic" => {
                let w = args.next().expect("--ionic needs none, cores, accepted or all");
                flags.ionic = Some(::sim::IonicPolicy::parse(&w).unwrap_or_else(|| panic!("--ionic needs none, cores, accepted or all, got {w:?}")));
            }
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
    // the CPU reference at 81³ takes ≈ 40 s per H₂ MD step (estimated) and cannot be cancelled:
    // started on the CPU without a chosen quality or scene file, the preset loads at Balanced
    // (49³). Switching the solver later keeps the grid (Scene ▸ Quality changes it).
    flags.cpu_default_quality(backend);
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
        // `--speed` as for the window and `--replay` (final review app M2: it was documented for
        // the screenshot but ignored there): set before the worker starts, so its run settings
        // carry it
        let setup = |v: &mut Viewer| {
            flags.apply(v)?;
            if let Some(n) = speed {
                v.ui.speed = n.clamp(1, ::sim::dynamics::MAX_SPEED);
            }
            Ok(())
        };
        if let Err(e) = screenshot::run(&path, &options, &setup) {
            eprintln!("screenshot failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    let mut viewer = Viewer::new(&preset)
        .unwrap_or_else(|| panic!("unknown preset {preset}; have {:?}", scene::Scene::PRESETS));
    // the window never starts a sampling pass by itself: Start sampling does (user request 2026-10-19)
    viewer.auto_pass = false;
    // the estimated solver learns into the shared collection of the scene's solver surface (M8a Task 4b)
    viewer.collections_dir = scene::collections_dir();
    if viewer.collections_dir.is_none() {
        eprintln!("note: no models folder beside the exe or in the working directory: the estimated solver keeps this session's model only (no shared collection)");
    }
    // the profile in force (the layout and look, `ui::profile`): kept per user, the app opens as it was left
    if let Some(base) = std::env::var_os("LOCALAPPDATA") {
        viewer.ui.profiles.current_file = Some(std::path::PathBuf::from(&base).join("atomic-sandbox").join("current.profile"));
        ui::profile::load_current(&mut viewer.ui);
        viewer.ui.timings = timings::Timings::load(Some(std::path::PathBuf::from(&base).join("atomic-sandbox").join("timings.txt")));
        // a running sampling pass is kept resumable there (`worker::checkpoint`), and offered at the next start
        viewer.training_dir = Some(std::path::PathBuf::from(base).join("atomic-sandbox").join("training"));
    }
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
        // the scene file is not a preset: its own `scene` line (written first, below), with its own
        // grid; a preset is recorded with the quality it was loaded at (the quality first, so the
        // preset loads at it), so a replay does not depend on the app's default quality
        let mut start = match u.source.preset() {
            Some(p) if flags.scene.is_none() => vec![ui::Action::Quality(u.quality_h), ui::Action::Preset(p.to_string())],
            _ => vec![],
        };
        // the ionic policy first (M7): a preset loads under the policy in force, so the pin precedes it — and
        // a replay never depends on the app's default policy
        start.insert(0, ui::Action::Ionic(u.ionic));
        if backend != sim::Backend::Gpu {
            start.push(ui::Action::Backend(backend));
        }
        if u.speed != 1 {
            start.push(ui::Action::SetSpeed(u.speed));
        }
        if u.audit_every != providers::FallbackPolicy::default().audit_every {
            start.push(ui::Action::SetAudit(u.audit_every));
        }
        // on or off: the app's default (on since 2026-10-14) may change again
        start.push(ui::Action::Walls { on: u.walls_on, half_width: u.wall_half_width });
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
        egui_repaint_at: None,
        closing: false,
        focused: true,
        frames: FrameStats::default(),
        popout: None,
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
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 900.0))
            // shown after the first frame (end of `resumed`), so it never flashes white while loading
            .with_visible(false)
            .with_window_icon(winit::window::Icon::from_rgba(icon::rgba(64), 64, 64).ok());
        // the custom frame (`frame.rs`): the top bar is the title bar
        #[cfg(windows)]
        let attrs = if frame::native_requested() {
            attrs
        } else {
            use winit::platform::windows::{CornerPreference, WindowAttributesExtWindows};
            attrs
                .with_decorations(false)
                .with_undecorated_shadow(true)
                .with_corner_preference(CornerPreference::Round)
        };
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        // the exe's embedded icon for the taskbar and the window (Windows loads it at its sizes);
        // winit's RGBA icon (`with_window_icon`) stays when the exe has none
        #[cfg(windows)]
        frame::exe_icon(&window);
        if !frame::native_requested() && !frame::install(&window) {
            window.set_decorations(true); // no subclass: the native caption, so it can still be moved
        }
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
        self.window = Some(window.clone());
        self.redraw(&window);
        window.set_visible(true);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if self.popout.as_ref().is_some_and(|p| p.window.id() == id) {
            self.popout_event(event);
            window.request_redraw(); // it is drawn with the main window's frame
            return;
        }
        if id != window.id() {
            return; // a window already closed
        }
        let consumed = match &mut self.egui_state {
            Some(s) => s.on_window_event(&window, &event).consumed,
            None => false,
        };
        match event {
            WindowEvent::CloseRequested => {
                ui::profile::flush(&mut self.viewer.ui); // the layout and look as left
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
            WindowEvent::Focused(f) => {
                self.focused = f;
                window.request_redraw();
            }
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
        if self.closing {
            el.exit();
            return;
        }
        self.sync_popout(el);
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
        // a frame egui asked for later (tooltips appear after a hover delay with the mouse still)
        let due = self.egui_repaint_at.is_some_and(|t| Instant::now() >= t);
        if due {
            self.egui_repaint_at = None;
        }
        // the floating dots drift by themselves: keep drawing while they are shown and the window has focus
        let floating = (self.focused || self.popout.as_ref().is_some_and(|p| p.focused)) && self.viewer.dots_floating();
        // a capture in progress keeps the frames coming (its readbacks land a frame or two later)
        let capturing = self.viewer.capture.active();
        // the pop-out shows an orbiting capture camera turning (as the view's gizmo previews it)
        let orbiting = self.popout.is_some() && self.viewer.ui.capture.camera.mode == capture::camera::CameraMode::Orbit;
        let animate = self.viewer.busy || self.viewer.tool.dragging || self.egui_repaint || got || due || floating || capturing || orbiting;
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
            let poll = Instant::now() + IDLE_POLL;
            el.set_control_flow(ControlFlow::WaitUntil(self.egui_repaint_at.map_or(poll, |t| t.min(poll))));
        }
    }
}

impl App {
    /// Opens or closes the pop-out view's window to match `ui.capture.popout`.
    fn sync_popout(&mut self, el: &ActiveEventLoop) {
        let want = self.viewer.ui.capture.popout;
        if !want {
            self.popout = None;
            return;
        }
        let (Some(g), None) = (&self.gfx, &self.popout) else { return };
        let v = &self.viewer;
        let view_px = (v.viewport.size[0].round() as u32, v.viewport.size[1].round() as u32);
        let window_px = (g.surface.config.width, g.surface.config.height);
        let frame = capture::popout::frame(&v.ui.capture, &v.ui.capture_view, view_px, window_px, v.capture_max_side(&g.ctx));
        match capture::popout::Popout::open(el, &g.ctx, Some(frame)) {
            Ok(p) => {
                eprintln!("pop-out view opened ({} × {} frame)", frame.size.0, frame.size.1);
                self.popout = Some(p);
                self.viewer.ui.set_status("pop-out view: the capture's frame in its own window (F11: full screen)".to_string());
            }
            Err(e) => {
                self.viewer.ui.capture.popout = false;
                self.viewer.last_error = Some(format!("pop-out view: {e}"));
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// The pop-out window's events: closing it clears the flag; F11 toggles full screen, Esc
    /// leaves it. Anything else only asks for a frame (it is drawn with the main window's).
    fn popout_event(&mut self, event: WindowEvent) {
        let Some(p) = &mut self.popout else { return };
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                self.popout = None;
                self.viewer.ui.capture.popout = false;
            }
            WindowEvent::Focused(f) => p.focused = f,
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() && !event.repeat => {
                use winit::keyboard::{Key, NamedKey};
                match event.logical_key {
                    Key::Named(NamedKey::F11) => p.toggle_fullscreen(),
                    Key::Named(NamedKey::Escape) => p.leave_fullscreen(),
                    _ => {}
                }
            }
            _ => {}
        }
    }

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
        frame::set_maximised(window.is_maximized());
        let mut out = self.viewer.run_ui(raw);
        match frame::take_cmd() {
            Some(frame::Cmd::Minimise) => window.set_minimized(true),
            Some(frame::Cmd::ToggleMaximise) => window.set_maximized(!window.is_maximized()),
            Some(frame::Cmd::Close) => {
                ui::profile::flush(&mut self.viewer.ui);
                // the same path as the native close: `CloseRequested` is not sent for our button
                window.set_title("Atomic Sandbox — closing…");
                window.set_visible(false);
                self.closing = true;
            }
            None => {}
        }
        if let Some(rec) = self.record.as_mut().and_then(|r| r.recorder.as_mut()) {
            for a in &self.viewer.chrome_actions {
                rec.action(a);
            }
        }
        egui_state.handle_platform_output(window, std::mem::take(&mut out.platform_output));
        let delay = out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay);
        self.egui_repaint = delay.is_some_and(|d| d.is_zero());
        // a later frame (egui says Duration::MAX for none): the idle loop wakes for it
        self.egui_repaint_at = delay.filter(|d| !d.is_zero() && *d < Duration::from_secs(60)).map(|d| Instant::now() + d);
        self.viewer.set_view(&mut g.painter, &g.ctx);
        let v = &self.viewer;
        let target = Target {
            colour: &view,
            depth: &g.surface.depth,
            size: [g.surface.config.width, g.surface.config.height],
            sampleable: g.surface.sampleable,
        };
        let shown = v.shown_scene(Instant::now());
        let lines = v.overlay_lines(&shown);
        let risk = v.atom_risk();
        let blur = if v.ui.camera_view { Vec::new() } else { v.blur_ghosts(&shown) };
        let frames = if v.ui.camera_view { v.camera_frames(&shown) } else { Vec::new() };
        let selected = v.tool.set(shown.nuclei.len());
        let warp_from = v.warp_source();
        let world = World {
            camera: &v.camera,
            nuclei: v.drawn_nuclei(&shown),
            selected: &selected,
            lines: &lines,
            viewport: v.viewport,
            risk: &risk,
            blur: &blur,
            glow: v.ui.camera_view.then_some((&frames[..], v.ui.exposure_ev, v.ui.camera_solidity, v.ui.camera_size, v.ui.camera_smooth)),
            warp_from: warp_from.as_deref(),
        };
        g.painter.paint(&g.ctx, target, world, &v.egui_ctx, out);
        // screenshots and recordings: the window as painted, or the 3D view offscreen (display only)
        let surface = capture::SurfaceFrame {
            texture: &frame.texture,
            size: (g.surface.config.width, g.surface.config.height),
            format: g.surface.config.format,
            copyable: g.surface.copyable,
        };
        self.viewer.capture_after_paint(&g.ctx, &mut g.painter, &g.surface.depth, Some(surface));
        // the pop-out view: its frame through the capture's path (or a copy of this one), before this one is presented
        if let Some(p) = &mut self.popout {
            p.prepare(&self.viewer, &g.ctx, &mut g.painter, &g.surface.depth, Some(surface));
        }
        window.pre_present_notify();
        g.ctx.queue.present(frame);
        if let Some(p) = &mut self.popout {
            p.present(&g.ctx);
        }
        self.frames.frame(self.viewer.playing);
    }
}
