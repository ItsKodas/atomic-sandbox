//! Viewer capture (user request 2026-10-19/20): the Screenshot and Record buttons at the top right
//! of the 3D view (`ui::capture`). Display only — capture never changes the simulation.
//!
//! - **Screenshot**: one PNG in `captures/` (beside the exe in a dist folder, else in the working
//!   directory: the repository's, gitignored), named `<date>_<time>_<scene>.png`.
//! - **Record**: a PNG sequence written by [`writer::Writer`] threads, encoded to MP4 with `ffmpeg`
//!   when it is on PATH (H.264, yuv420p, 30 fps) and the PNGs deleted after a successful encode;
//!   otherwise the PNGs stay and the note says how to encode them. Frames are taken by **simulated
//!   time** (default; [`clock::SimClock`]: one frame per `fs_per_frame` of simulated time,
//!   positions interpolated between snapshots, nothing while paused) or **as seen**
//!   ([`clock::WallClock`]: 30 frames per wall second, for recording interaction).
//! - **Hide UI**: the frame is the 3D view alone, rendered offscreen at the chosen resolution (the
//!   view's size, 1080p, 1440p or 4K) through the window's own painter
//!   ([`crate::gfx::Painter::paint_offscreen`]); with the UI shown the frame is the window as
//!   presented (its size; the surface must allow copies). Simulated-time frames are always the 3D
//!   view alone: the chrome shows wall-clock state, not the frame's instant.
//!
//! Nothing here waits on the GPU or the disk: copies are read back by [`readback::Pool`] a frame or
//! two later, PNGs are written on writer threads behind a bounded queue, and the finish (gap
//! filling, encoding) runs on its own thread. When the recording falls behind, simulated-time mode
//! doubles its fs per frame and as-seen mode repeats frames; either says so by the button.
pub mod camera;
pub mod clock;
pub mod overlay;
pub mod readback;
pub mod writer;

use clock::{Sample, SimClock, SimFrame, WallClock, FPS};
use readback::{Pool, Tag};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use writer::{Image, Note, Writer};

/// The offscreen frame's size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The 3D view's size in the window (physical pixels).
    View,
    P1080,
    P1440,
    P2160,
}

impl Resolution {
    pub const ALL: [Resolution; 4] = [Resolution::View, Resolution::P1080, Resolution::P1440, Resolution::P2160];

    pub fn label(self) -> &'static str {
        match self {
            Resolution::View => "view's size",
            Resolution::P1080 => "1080p",
            Resolution::P1440 => "1440p",
            Resolution::P2160 => "4K",
        }
    }

    /// The frame's short side (`None`: the view's height; [`camera::frame_size`]).
    pub fn short_side(self) -> Option<f32> {
        match self {
            Resolution::View => None,
            Resolution::P1080 => Some(1080.0),
            Resolution::P1440 => Some(1440.0),
            Resolution::P2160 => Some(2160.0),
        }
    }
}

/// What a capture with the UI hidden may still draw over the 3D view (a separate switch): the
/// labels as in the window (bond lengths, angles with their arcs, charges, molecule names), a
/// title in the top-left corner listing the molecules (instead of their names over them), the
/// time readout and the distance scale.
#[derive(Clone, Debug, PartialEq)]
pub struct Overlay {
    pub on: bool,
    pub labels: bool,
    pub title: bool,
    pub time: bool,
    pub scale: bool,
}

impl Default for Overlay {
    fn default() -> Self {
        Overlay { on: false, labels: true, title: true, time: true, scale: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordMode {
    /// One frame per slice of simulated time (a time-lapse of a slow run).
    SimTime,
    /// 30 frames per wall second, as on screen.
    AsSeen,
}

/// The quick settings (the popover by the buttons; display state, not recorded).
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub hide_ui: bool,
    pub resolution: Resolution,
    pub aspect: camera::Aspect,
    /// Labels, title, time and scale over a hidden-UI capture.
    pub overlay: Overlay,
    /// Where a hidden-UI capture is drawn from (live: a change applies to the next frame).
    pub camera: camera::CameraSettings,
    pub mode: RecordMode,
    pub fs_per_frame: f64,
    /// The popover is open.
    pub open: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            hide_ui: true,
            resolution: Resolution::View,
            aspect: camera::Aspect::View,
            overlay: Overlay::default(),
            camera: camera::CameraSettings::default(),
            mode: RecordMode::SimTime,
            fs_per_frame: clock::DEFAULT_FS_PER_FRAME,
            open: false,
        }
    }
}

impl Settings {
    /// The frame size for a 3D view of `view` pixels.
    pub fn frame_size(&self, view: (u32, u32)) -> (u32, u32) {
        camera::frame_size(self.resolution, self.aspect, view)
    }
}

/// What the chrome asks of the capture (`ui::Action::Screenshot` / `Record`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    Screenshot,
    StartRecording,
    StopRecording,
}

/// What the buttons show (refreshed by the viewer before each frame's chrome).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct View {
    pub recording: Option<RecordingView>,
    /// A recording being finished (writing, encoding).
    pub finishing: Option<String>,
    /// The last thing the capture did (saved, finished, an error; short: file names, not paths) and when.
    pub last: Option<(Instant, String, bool)>,
    /// The window's surface can be copied (a capture with the UI shown).
    pub surface_copyable: bool,
    /// `ffmpeg` is on PATH (`None` until the probe answered).
    pub ffmpeg: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecordingView {
    pub mode: RecordMode,
    pub frames: u64,
    /// Simulated time covered (fs; simulated-time mode).
    pub sim_fs: f64,
    /// The video's length at 30 fps (s).
    pub playback_s: f64,
    /// The current fs per frame (simulated-time mode; doubles when behind).
    pub fs_per_frame: f64,
    pub size: (u32, u32),
    pub hide_ui: bool,
    /// Why the video is not what the settings asked for (fell behind, coarse snapshots).
    pub notes: Vec<String>,
    pub started: Instant,
}

/// A recording in progress.
struct Session {
    id: u64,
    mode: RecordMode,
    hide_ui: bool,
    resolution: Resolution,
    aspect: camera::Aspect,
    dir: PathBuf,
    video: PathBuf,
    writer: Writer,
    sim: SimClock,
    wall: WallClock,
    /// The frame size, fixed at the first frame (an encode needs one size).
    size: Option<(u32, u32)>,
    /// Frames the readback pool or the writers refused (filled with the previous frame).
    dropped: u64,
    started: Instant,
    /// The playing state the clock last saw (a pause breaks the interpolation).
    playing: bool,
}

/// A screenshot waiting for the next frame.
struct Shot {
    path: PathBuf,
    hide_ui: bool,
    resolution: Resolution,
    aspect: camera::Aspect,
}

/// A frame to render offscreen this frame.
pub struct Job {
    pub tag: Tag,
    pub size: (u32, u32),
    /// Simulated-time frames: the positions at the frame's instant and the crossfade weight.
    pub at: Option<SimFrame>,
    /// The frame's time in the video (s): the orbit's phase, the labels' smoothing clock.
    pub video_t: f64,
    /// Simulated-time recordings: simulated fs per second of video (the time readout).
    pub video_fs_per_s: Option<f64>,
}

/// What [`Capture::plan`] wants of this frame.
#[derive(Default)]
pub struct Plan {
    /// Copy the presented window frame (the UI shown).
    pub surface: Option<Tag>,
    pub offscreen: Vec<Job>,
}

impl Plan {
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.surface.is_none() && self.offscreen.is_empty()
    }
}

#[derive(Clone)]
/// The offscreen target (colour in the painter's format, readable and sampleable for bloom, and
/// its depth), kept between frames of one size.
pub struct Offscreen {
    pub colour: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
}

impl Offscreen {
    fn new(ctx: &gpu::GpuContext, size: (u32, u32), format: wgpu::TextureFormat) -> Offscreen {
        let extent = wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 };
        let colour = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture colour"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = colour.create_view(&Default::default());
        let depth = ctx
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("capture depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gpu::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Offscreen { colour, view, depth, size, format }
    }
}

/// Offscreen frames rendered per window frame at most (the rest stay due).
const PER_FRAME: usize = 3;

pub struct Capture {
    pool: Pool,
    notes_tx: Sender<Note>,
    notes_rx: Receiver<Note>,
    session: Option<Session>,
    /// A stopped recording whose last frames are still being read back.
    closing: Option<Session>,
    sessions: u64,
    shot: Option<Shot>,
    target: Option<Offscreen>,
    view: View,
    /// The ffmpeg probe was started (once, on a thread).
    probed: bool,
    /// The overlay's egui context and the labels' smoothing (hidden-UI captures with the overlay on).
    labels: overlay::LabelState,
}

impl Default for Capture {
    fn default() -> Self {
        let (notes_tx, notes_rx) = writer::notes();
        Capture { pool: Pool::default(), notes_tx, notes_rx, session: None, closing: None, sessions: 0, shot: None, target: None, view: View::default(), probed: false, labels: Default::default() }
    }
}

/// The `captures` folder: beside the executable in a dist folder (one with `models/` beside the
/// exe), else in the working directory (the repository root under `start.bat`; gitignored).
pub fn captures_dir() -> PathBuf {
    let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
    match exe_dir.filter(|d| d.join("models").is_dir()) {
        Some(d) => d.join("captures"),
        None => std::env::current_dir().unwrap_or_default().join("captures"),
    }
}

/// `2026-10-20_14-03-22_h2o` (local time; the scene's name made file-safe).
pub fn stem(scene: &str) -> String {
    let (y, mo, d, h, mi, s) = local_time();
    let scene: String = scene
        .trim_end_matches(".scene")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    format!("{y:04}-{mo:02}-{d:02}_{h:02}-{mi:02}-{s:02}_{scene}")
}

#[cfg(windows)]
fn local_time() -> (u16, u16, u16, u16, u16, u16) {
    // SAFETY: GetLocalTime only writes the struct it is given.
    let t = unsafe {
        let mut t = std::mem::zeroed::<windows_sys::Win32::Foundation::SYSTEMTIME>();
        windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut t);
        t
    };
    (t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond)
}

/// UTC elsewhere (civil date from the Unix day count).
#[cfg(not(windows))]
fn local_time() -> (u16, u16, u16, u16, u16, u16) {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y as u16, m as u16, d as u16, (rem / 3600) as u16, (rem % 3600 / 60) as u16, (rem % 60) as u16)
}

/// The time readout's "12.34 fs", "1.234 ps".
pub fn time_words(fs: f64) -> String {
    if fs.abs() >= 1000.0 {
        format!("{:.3} ps", fs / 1000.0)
    } else {
        format!("{fs:.2} fs")
    }
}

/// "12.5 fs", "1.25 ps".
pub fn fs_words(fs: f64) -> String {
    if fs >= 1000.0 {
        format!("{:.2} ps", fs / 1000.0)
    } else {
        format!("{fs:.1} fs")
    }
}

impl Capture {
    /// What the buttons show now.
    pub fn view(&self) -> &View {
        &self.view
    }

    /// The chrome's request. `scene` names the files.
    pub fn request(&mut self, r: Request, settings: &Settings, scene: &str) -> Result<String, String> {
        let dir = captures_dir();
        match r {
            Request::Screenshot => {
                std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let mut path = dir.join(format!("{}.png", stem(scene)));
                // two in one second: a suffix
                let mut k = 2;
                while path.exists() || self.shot.as_ref().is_some_and(|s| s.path == path) {
                    path = dir.join(format!("{}-{k}.png", stem(scene)));
                    k += 1;
                }
                self.shot = Some(Shot { path, hide_ui: settings.hide_ui || !self.view.surface_copyable, resolution: settings.resolution, aspect: settings.aspect });
                Ok("screenshot: taking the next frame".into())
            }
            Request::StartRecording => {
                if self.session.is_some() {
                    return Ok("already recording".into());
                }
                let stem = stem(scene);
                let frames = dir.join(format!("{stem}_frames"));
                std::fs::create_dir_all(&frames).map_err(|e| format!("{}: {e}", frames.display()))?;
                self.sessions += 1;
                let hide_ui = settings.hide_ui || settings.mode == RecordMode::SimTime || !self.view.surface_copyable;
                self.session = Some(Session {
                    id: self.sessions,
                    mode: settings.mode,
                    hide_ui,
                    resolution: settings.resolution,
                    aspect: settings.aspect,
                    video: dir.join(format!("{stem}.mp4")),
                    writer: Writer::new(frames.clone()),
                    dir: frames,
                    sim: SimClock::new(settings.fs_per_frame),
                    wall: WallClock::default(),
                    size: None,
                    dropped: 0,
                    started: Instant::now(),
                    playing: false,
                });
                Ok(match settings.mode {
                    RecordMode::SimTime => format!("recording by simulated time: one frame per {} — nothing is recorded while paused", fs_words(settings.fs_per_frame)),
                    RecordMode::AsSeen => "recording as seen (30 frames per second of wall time)".into(),
                })
            }
            Request::StopRecording => {
                let Some(s) = self.session.take() else { return Ok("not recording".into()) };
                // the frames still on the GPU land first (`tick`), then the finish starts
                self.finish_closing();
                self.closing = Some(s);
                self.view.finishing = Some("waiting for the last frames…".into());
                Ok("recording stopped: finishing the video…".into())
            }
        }
    }

    /// Hands the stopped recording to its finish thread.
    fn finish_closing(&mut self) {
        let Some(s) = self.closing.take() else { return };
        let frames = match s.mode {
            RecordMode::SimTime => s.sim.frames(),
            RecordMode::AsSeen => s.wall.frames(),
        };
        let summary = summary(&s, frames);
        writer::finish(writer::Finish { writer: s.writer, dir: s.dir, video: s.video, frames, summary }, self.notes_tx.clone());
    }

    /// An adopted snapshot (the viewer's mirror took it): simulated-time recording follows the run.
    pub fn on_snapshot(&mut self, t_fs: f64, step: u64, pos: Vec<[f64; 3]>, playing: bool) {
        let Some(s) = self.session.as_mut().filter(|s| s.mode == RecordMode::SimTime) else { return };
        if !playing {
            if s.playing {
                s.sim.pause();
            }
            s.playing = false;
            return;
        }
        s.playing = true;
        s.sim.push(Sample { t_fs, step, pos });
    }

    /// The scene was edited (a teleport, an addition): never interpolate across it.
    pub fn on_edit(&mut self) {
        if let Some(s) = self.session.as_mut() {
            s.sim.pause();
        }
    }

    /// This frame's captures. `view_px`: the 3D view's size; `window_px`: the window's.
    pub fn plan(&mut self, now: Instant, view_px: (u32, u32), window_px: (u32, u32)) -> Plan {
        let mut plan = Plan::default();
        if let Some(shot) = self.shot.take() {
            let tag = Tag::Screenshot(shot.path);
            if shot.hide_ui {
                plan.offscreen.push(Job { tag, size: camera::frame_size(shot.resolution, shot.aspect, view_px), at: None, video_t: 0.0, video_fs_per_s: None });
            } else {
                plan.surface = Some(tag);
            }
        }
        let room = readback::IN_FLIGHT.saturating_sub(self.pool.pending() + plan.offscreen.len() + usize::from(plan.surface.is_some()));
        let Some(s) = self.session.as_mut() else { return plan };
        let size = *s.size.get_or_insert(if s.hide_ui { camera::frame_size(s.resolution, s.aspect, view_px) } else { window_px });
        match s.mode {
            RecordMode::SimTime => {
                if s.sim.backlog() {
                    s.sim.fell_behind();
                }
                for (k, f) in s.sim.due(PER_FRAME).into_iter().enumerate() {
                    if k >= room {
                        s.dropped += 1; // its slot is filled with the previous frame at the finish
                        continue;
                    }
                    let (video_t, rate) = (f.index as f64 / FPS, s.sim.slice * FPS);
                    plan.offscreen.push(Job { tag: Tag::Frame { session: s.id, index: f.index }, size, at: Some(f), video_t, video_fs_per_s: Some(rate) });
                }
            }
            RecordMode::AsSeen => {
                if let Some(slot) = s.wall.due(now) {
                    let tag = Tag::Frame { session: s.id, index: slot };
                    if room == 0 {
                        s.dropped += 1;
                    } else if s.hide_ui {
                        plan.offscreen.push(Job { tag, size, at: None, video_t: slot as f64 / FPS, video_fs_per_s: None });
                    } else if window_px == size {
                        plan.surface = plan.surface.take().or(Some(tag));
                    } else {
                        s.dropped += 1; // the window was resized: the video keeps its first size
                    }
                }
            }
        }
        plan
    }

    /// The offscreen target at `size` in `format` (recreated when either changes).
    pub fn target(&mut self, ctx: &gpu::GpuContext, size: (u32, u32), format: wgpu::TextureFormat) -> &Offscreen {
        if self.target.as_ref().is_none_or(|t| t.size != size || t.format != format) {
            self.target = Some(Offscreen::new(ctx, size, format));
        }
        self.target.as_ref().expect("made above")
    }

    /// Records the copy of the offscreen target into `enc` (after its frame was rendered).
    pub fn copy_target(&mut self, ctx: &gpu::GpuContext, enc: &mut wgpu::CommandEncoder, tag: Tag) -> bool {
        let Some(t) = &self.target else { return false };
        self.pool.copy(ctx, enc, &t.colour, t.size.0, t.size.1, t.format, tag)
    }

    /// Records the copy of the window's frame into `enc`.
    pub fn copy_surface(&mut self, ctx: &gpu::GpuContext, enc: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, size: (u32, u32), format: wgpu::TextureFormat, tag: Tag) -> bool {
        self.pool.copy(ctx, enc, texture, size.0, size.1, format, tag)
    }

    /// A copy the pool refused: a screenshot says so, a recording's frame is counted as dropped.
    pub fn refused(&mut self, tag: Tag) {
        match tag {
            Tag::Screenshot(p) => self.note(Note::Error(format!("screenshot {}: no room to read the frame back; try again", p.display()))),
            Tag::Frame { session, .. } => {
                if let Some(s) = self.session.as_mut().filter(|s| s.id == session) {
                    s.dropped += 1;
                }
            }
        }
    }

    /// The copies of this frame were submitted.
    pub fn submitted(&mut self) {
        self.pool.submitted();
    }

    /// Something is in progress (a screenshot or frames on their way, a recording, a finish): the
    /// window keeps drawing so they complete.
    pub fn active(&self) -> bool {
        self.shot.is_some() || self.session.is_some() || self.closing.is_some() || self.pool.pending() > 0 || self.view.finishing.is_some()
    }

    /// Hands arrived frames to the disk, reads the threads' notes and refreshes the buttons' view.
    /// Returns the notes for the status line.
    pub fn tick(&mut self, ctx: Option<&gpu::GpuContext>, surface_copyable: bool) -> Vec<(String, bool)> {
        self.view.surface_copyable = surface_copyable;
        if !self.probed {
            self.probed = true;
            let tx = self.notes_tx.clone();
            std::thread::Builder::new().name("capture-ffmpeg-probe".into()).spawn(move || tx.send(Note::Ffmpeg(writer::ffmpeg_available())).ok()).ok();
        }
        if let Some(ctx) = ctx {
            for (tag, img) in self.pool.collect(ctx) {
                self.deliver(tag, img);
            }
        }
        // a stopped recording is finished once its last frames are off the GPU
        if self.closing.as_ref().is_some_and(|s| !self.pool.pending_for(s.id)) {
            self.finish_closing();
        }
        // (status line text, its short form under the buttons, is an error)
        let mut lines: Vec<(String, String, bool)> = Vec::new();
        while let Ok(n) = self.notes_rx.try_recv() {
            match n {
                Note::Saved(p) => lines.push((format!("screenshot saved: {}", p.display()), format!("saved {}", writer::file_name(&p)), false)),
                Note::Finishing(t) => {
                    self.view.finishing = Some(t);
                    continue;
                }
                Note::Finished { text, short } => {
                    self.view.finishing = None;
                    lines.push((text, short, false));
                }
                Note::Error(e) => lines.push((e.clone(), e, true)),
                Note::Ffmpeg(found) => {
                    self.view.ffmpeg = Some(found);
                    continue;
                }
            }
        }
        if let Some((_, short, err)) = lines.last() {
            self.view.last = Some((Instant::now(), short.clone(), *err));
        }
        let lines = lines.into_iter().map(|(text, _, err)| (text, err)).collect();
        self.view.recording = self.session.as_ref().map(|s| recording_view(s));
        lines
    }

    fn note(&self, n: Note) {
        self.notes_tx.send(n).ok();
    }

    fn deliver(&mut self, tag: Tag, img: Result<Image, String>) {
        match (tag, img) {
            (Tag::Screenshot(path), Ok(img)) => {
                let tx = self.notes_tx.clone();
                std::thread::Builder::new()
                    .name("capture-screenshot".into())
                    .spawn(move || {
                        let n = match writer::write_png(&path, &img) {
                            Ok(()) => Note::Saved(path),
                            Err(e) => Note::Error(format!("screenshot: {e}")),
                        };
                        tx.send(n).ok();
                    })
                    .ok();
            }
            (Tag::Screenshot(path), Err(e)) => self.note(Note::Error(format!("screenshot {}: {e}", path.display()))),
            (Tag::Frame { session, index }, img) => {
                let Some(s) = self.session.as_mut().filter(|s| s.id == session).or(self.closing.as_mut().filter(|s| s.id == session)) else { return };
                let sent = img.is_ok_and(|img| s.writer.try_send(index, img));
                if !sent {
                    s.dropped += 1;
                    // the writers are behind: fewer frames from now on (sim time) — as seen repeats
                    if s.mode == RecordMode::SimTime && s.dropped % 4 == 1 {
                        s.sim.fell_behind();
                    }
                }
            }
        }
    }
}

/// The window's frame as painted, before it is presented.
pub struct SurfaceFrame<'a> {
    pub texture: &'a wgpu::Texture,
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
    /// The surface was configured with `COPY_SRC`.
    pub copyable: bool,
}

impl crate::viewer::Viewer {
    /// The capture's part of a window frame, after the frame is painted and before it is
    /// presented: copies the window (a capture with the UI), renders and copies the offscreen
    /// frames due, hands arrived frames to the disk and puts the notes in the status line.
    pub fn capture_after_paint(&mut self, ctx: &gpu::GpuContext, painter: &mut crate::gfx::Painter, window_depth: &wgpu::TextureView, surface: Option<SurfaceFrame<'_>>) {
        let copyable = surface.as_ref().is_some_and(|s| s.copyable && readback::readable(s.format).is_some());
        let mut cap = std::mem::take(&mut self.capture);
        let view_px = (self.viewport.size[0].round() as u32, self.viewport.size[1].round() as u32);
        let window_px = surface.as_ref().map_or(view_px, |s| s.size);
        let plan = cap.plan(Instant::now(), view_px, window_px);
        if let Some(tag) = plan.surface {
            match surface.as_ref().filter(|_| copyable) {
                Some(s) => {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture window") });
                    let ok = cap.copy_surface(ctx, &mut enc, s.texture, s.size, s.format, tag.clone());
                    ctx.queue.submit([enc.finish()]);
                    if !ok {
                        cap.refused(tag);
                    }
                }
                None => cap.refused(tag),
            }
        }
        for job in plan.offscreen {
            let target = cap.target(ctx, job.size, painter.format).clone();
            self.render_capture(ctx, painter, &target, &job, &mut cap.labels, window_depth);
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture copy") });
            let ok = cap.copy_target(ctx, &mut enc, job.tag.clone());
            ctx.queue.submit([enc.finish()]);
            if !ok {
                cap.refused(job.tag);
            }
        }
        cap.submitted();
        let notes = cap.tick(Some(ctx), copyable);
        self.capture = cap;
        for (text, err) in notes {
            if err {
                self.last_error = Some(format!("capture: {text}"));
            } else {
                self.ui.set_status(text);
            }
        }
    }

    /// The capture camera for a frame of aspect `frame_ratio` at video time `t_video`
    /// ([`camera::capture_camera`] of the settings and the window's camera).
    pub fn capture_camera(&self, frame_ratio: f32, t_video: f64) -> render::camera::OrbitCamera {
        camera::capture_camera(&self.camera, self.viewport.aspect(), frame_ratio, &self.ui.capture.camera, t_video)
    }

    /// The capture camera drawn in the editor view while it is not the view's own (a fixed or
    /// orbiting camera) and the settings are open or a recording runs; an idle orbit spins on the
    /// wall clock as a preview. Never in a capture (it is editor geometry).
    pub fn capture_gizmo(&self) -> Vec<render::lines::LineVertex> {
        let s = &self.ui.capture;
        let recording = self.ui.capture_view.recording.as_ref();
        let hidden = s.hide_ui || recording.is_some_and(|r| r.hide_ui);
        if s.camera.mode == camera::CameraMode::View || !hidden || !(s.open || recording.is_some()) {
            return Vec::new();
        }
        let vp = self.viewport;
        let (w, h) = s.frame_size((vp.size[0] as u32, vp.size[1] as u32));
        let ratio = w as f32 / h as f32;
        let t = recording.map_or_else(|| self.egui_ctx.input(|i| i.time), |r| r.playback_s);
        camera::gizmo(&self.capture_camera(ratio, t), ratio, s.camera.mode == camera::CameraMode::Orbit)
    }

    /// The molecules as the view labels them (two or more atoms, in the name style), for the title.
    fn molecule_names(&self) -> Vec<String> {
        let Some(s) = self.snapshot.as_ref() else { return Vec::new() };
        let Some(g) = s.dynamics.bonds.as_ref().filter(|g| g.n_nuclei == self.scene.nuclei.len() && s.labels.len() == g.components.len()) else { return Vec::new() };
        s.labels.iter().zip(&g.components).filter(|(_, m)| m.len() >= 2).map(|(n, _)| n.label_in(self.ui.name_style)).collect()
    }

    /// One hidden-UI frame into `t`: the scene as the window draws it — or, for a simulated-time
    /// frame, its nuclei at the frame's instant and the density crossfade at its weight — from the
    /// capture camera, without the chrome, the editor geometry or the selection highlight; with the
    /// overlay (labels, title, time, scale) when it is on. Motion blur and the Camera view's
    /// shutter are wall-clock effects: a simulated-time frame is one instant.
    fn render_capture(&self, ctx: &gpu::GpuContext, painter: &mut crate::gfx::Painter, t: &Offscreen, job: &Job, labels: &mut overlay::LabelState, window_depth: &wgpu::TextureView) {
        let at = job.at.as_ref();
        let now = Instant::now();
        let mut shown = self.shown_scene(now);
        if let Some(f) = at.filter(|f| f.pos.len() == shown.nuclei.len()) {
            for (n, p) in shown.nuclei.iter_mut().zip(&f.pos) {
                n.pos = *p;
            }
        }
        let (fade, mix) = self.look(painter.drawn(&self.ui), now);
        let fade = if self.ghost_now().is_some() { 0.0 } else { fade };
        let mix = match at {
            Some(f) if self.density.superposition.is_none() => f.alpha as f32,
            _ => mix,
        };
        painter.set_view(ctx, &self.ui, fade, mix);
        let o = &self.ui.capture.overlay;
        let show_labels = o.on && o.labels;
        // no editor geometry (cell box, walls, floor, axes, placement grid); the angle arcs with the labels
        let lines = self.scene_lines(&shown, false, show_labels);
        let risk = self.atom_risk();
        let blur = if self.ui.camera_view || at.is_some() { Vec::new() } else { self.blur_ghosts(&shown) };
        let frames = match (self.ui.camera_view, at) {
            (false, _) => Vec::new(),
            (true, Some(_)) => vec![shown.nuclei.iter().map(|n| n.pos).collect()],
            (true, None) => self.camera_frames(&shown),
        };
        let warp_from = self.warp_source();
        let (w, h) = t.size;
        let cam = self.capture_camera(w as f32 / h as f32, job.video_t);
        let world = crate::gfx::World {
            camera: &cam,
            nuclei: self.drawn_nuclei(&shown),
            selected: &[],
            lines: &lines,
            viewport: render::camera::Viewport::full(w as f32, h as f32),
            risk: &risk,
            blur: &blur,
            glow: self.ui.camera_view.then_some((&frames[..], self.ui.exposure_ev, self.ui.camera_solidity, self.ui.camera_size, self.ui.camera_smooth)),
            warp_from: warp_from.as_deref(),
        };
        let target = crate::gfx::Target { colour: &t.view, depth: &t.depth, size: [w, h], sampleable: true };
        if !o.on {
            painter.paint_offscreen(ctx, target, world, (&self.egui_ctx, None), window_depth);
            return;
        }
        // the overlay at the size it has in the window: the frame's height over what it shows of the view
        let view_h = match self.ui.capture.camera.mode {
            camera::CameraMode::View => {
                let vp = self.viewport;
                camera::inscribed((0.0, 0.0, vp.size[0], vp.size[1]), w as f32 / h as f32).3
            }
            _ => self.viewport.size[1],
        };
        let ppp = (self.egui_ctx.pixels_per_point() * h as f32 / view_h.max(1.0)).max(0.5);
        let hud = self.capture_hud(job);
        let scale = camera::scale_bar(&cam, h as f32 / ppp, 0.22 * w as f32 / ppp);
        // a recording's labels are smoothed on the video's clock, a screenshot's on the wall's
        let clock = if job.video_fs_per_s.is_some() || matches!(job.tag, readback::Tag::Frame { .. }) { job.video_t } else { self.egui_ctx.input(|i| i.time) };
        let names = !(o.title && !hud.title.is_empty());
        let (octx, out) = labels.run((w, h), ppp, clock, |c, smooth, sides| {
            if show_labels {
                let lv = crate::viewer::LabelView { camera: &cam, smooth, sides, capture: true, names };
                self.label_overlay(c, &shown, render::camera::Viewport::full(w as f32, h as f32), &lv);
            }
            overlay::draw_hud(c, &hud, Some(scale));
        });
        painter.paint_offscreen(ctx, target, world, (&octx, Some(out)), window_depth);
    }

    /// The overlay's corner text for `job`: the molecules (title), the scene and solver, the time
    /// and the rate.
    fn capture_hud(&self, job: &Job) -> overlay::Hud {
        let o = &self.ui.capture.overlay;
        let mut hud = overlay::Hud { scale: o.scale, ..Default::default() };
        if o.title {
            hud.title = overlay::title_lines(&self.molecule_names());
            hud.subtitle = Some(format!("{} · {}", self.ui.source.label().trim_end_matches(".scene"), self.backend.label()));
        }
        if o.time {
            let d = self.snapshot.as_ref().map(|s| &s.dynamics);
            let (t_fs, step) = match &job.at {
                Some(f) => (f.t_fs, f.step),
                None => (d.map_or(0.0, |d| d.sim_time_fs()), d.map_or(0, |d| d.step)),
            };
            hud.time.push(format!("t = {} · step {step}", time_words(t_fs)));
            hud.time.push(match (job.video_fs_per_s, self.measured_fs_per_s()) {
                (Some(v), Some(run)) => format!("1 s of video = {} · the run: {:.1} fs/s", time_words(v), run),
                (Some(v), None) => format!("1 s of video = {}", time_words(v)),
                (None, Some(run)) if self.playing => format!("running at {run:.1} fs/s"),
                _ if self.playing => "running".to_string(),
                _ => "paused".to_string(),
            });
        }
        hud
    }
}

fn recording_view(s: &Session) -> RecordingView {
    let frames = match s.mode {
        RecordMode::SimTime => s.sim.frames(),
        RecordMode::AsSeen => s.wall.frames(),
    };
    let mut notes = Vec::new();
    if s.mode == RecordMode::SimTime {
        if s.sim.coarsened > 0 {
            notes.push(format!("fell behind: now one frame per {}", fs_words(s.sim.slice)));
        }
        if s.sim.coarse > 0 {
            notes.push(format!("{} frames at snapshots too far apart to interpolate — lower the speed for smooth motion", s.sim.coarse));
        }
    }
    let repeated = s.dropped + if s.mode == RecordMode::AsSeen { s.wall.repeated } else { 0 };
    if repeated > 0 {
        notes.push(format!("{repeated} frames repeated (the capture fell behind)"));
    }
    RecordingView {
        mode: s.mode,
        frames,
        sim_fs: s.sim.covered_fs(),
        playback_s: frames as f64 / FPS,
        fs_per_frame: s.sim.slice,
        size: s.size.unwrap_or((0, 0)),
        hide_ui: s.hide_ui,
        notes,
        started: s.started,
    }
}

fn summary(s: &Session, frames: u64) -> String {
    let len = format!("{frames} frames, {:.1} s at 30 fps", frames as f64 / FPS);
    match s.mode {
        RecordMode::SimTime => format!("recorded {} of simulated time ({len}, {} per frame)", fs_words(s.sim.covered_fs()), fs_words(s.sim.slice)),
        RecordMode::AsSeen => format!("recorded {:.1} s as seen ({len})", s.started.elapsed().as_secs_f64()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolutions_and_names() {
        let s = stem("my scene.scene");
        assert!(s.ends_with("_my-scene") && s.len() == "2026-10-20_14-03-22_my-scene".len(), "{s}");
        assert_eq!(fs_words(12.5), "12.5 fs");
        assert_eq!(fs_words(1250.0), "1.25 ps");
    }

    #[test]
    fn a_screenshot_with_the_ui_copies_the_window_and_without_renders_offscreen() {
        let mut c = Capture::default();
        c.view.surface_copyable = true;
        c.shot = Some(Shot { path: "a.png".into(), hide_ui: false, resolution: Resolution::P1080, aspect: camera::Aspect::Wide });
        let p = c.plan(Instant::now(), (800, 600), (1000, 700));
        assert_eq!(p.surface, Some(Tag::Screenshot("a.png".into())));
        c.shot = Some(Shot { path: "b.png".into(), hide_ui: true, resolution: Resolution::P1080, aspect: camera::Aspect::Wide });
        let p = c.plan(Instant::now(), (800, 600), (1000, 700));
        assert!(p.surface.is_none() && p.offscreen.len() == 1 && p.offscreen[0].size == (1920, 1080));
        assert!(c.plan(Instant::now(), (800, 600), (1000, 700)).is_empty(), "taken once");
    }
}
