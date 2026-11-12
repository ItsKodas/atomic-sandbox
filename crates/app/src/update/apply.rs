//! Installing a release from the app (installer spec, "Updates during a session"; plan Task 3):
//! the download job, the session saved before the restart, and the hand-off to the silent installer.
//!
//! # The download ([`Download`])
//! On its own thread, into `<user root>\updates`:
//! 1. `SHA256SUMS.txt` and its `.minisig` (small, held in memory);
//! 2. the installer, streamed to the job's own `<asset>.<pid>-<n>.part` and hashed as it arrives,
//!    with `(bytes, total)` progress over a channel;
//! 3. [`verify`] (fails closed: the signature under the compiled-in key, then the asset's SHA-256);
//! 4. only then renamed to `<asset>` (an atomic replace: the name only ever holds a whole verified
//!    file), which [`Download::verified`] hands to [`hand_off`] as a [`Verified`].
//!
//! Refused up front: an asset whose URL isn't exactly its release's `<tag>/<name>`, an installer
//! listed at 0 bytes (the listed size bounds what is written), no release key. Any later failure —
//! a fetch refused, a download that stops part-way or sends more than the release lists, a full
//! disk, a verification refusal, a cancel, a panic — removes the job's `.part` (a drop guard) and
//! reports why. A `.part` left by a killed process is Task 4's startup cleanup. The bytes come
//! from a [`Source`] ([`super::net::Http`] in the app), so the tests feed them from memory.
//!
//! # The session ([`Session`])
//! What a restart after an update brings back (spec, "Builder-check rulings"): the scene as it
//! stands (positions, velocities, charge, spin — the multiplicity — grid, σ, the ionic policy),
//! the MD time, and the backend, quality, ionic policy and walls; always paused. The profile (dock
//! layout, display settings) is the per-user `current.profile`, loaded as at any start.
//!
//! **The file** is `<user root>\updates\session.scene`. Its body is exactly what Save scene writes
//! ([`Scene::to_text`]); the extra settings ride in comment lines, `#session <key> <value>`. The
//! scene parser refuses unknown directives, so a new directive would break every older build that
//! opened the file — but every build ignores comments: the session file is a plain v1 scene file
//! to all of them (File ▸ Load, `--scene`), while `--resume-session` reads the `#session` lines as
//! well. Normal scene files are unchanged (no new directive, `SCENE_FORMAT` stays). A `#session`
//! key this build doesn't know is ignored (a newer build's extra), a missing one is an error.
//! The file is written whole (a temporary file renamed over it) and kept after the resume: the next
//! update overwrites it.
//!
//! # The hand-off ([`hand_off`])
//! Starts the verified installer detached ([`installer_command`]: `/VERYSILENT /SUPPRESSMSGBOXES
//! /NORESTART /WAITPID=<pid> /RELAUNCH="<exe> --resume-session <session>"`), then hands back [`CloseForUpdate`]:
//! the caller closes the app through its own clean-exit path (the profile flushed, the window gone)
//! — this module never exits the process. Installed builds only: a portable copy offers the zip
//! instead, a dev build has updates off.

use super::release::{installable, Asset, Release};
use super::verify::{has_key, verify, NO_RELEASE_KEY, RELEASE_KEY};
use crate::paths::{self, DataMode};
use crate::scene::Scene;
use crate::sim::Backend;
use crate::ui::SceneSource;
use crate::viewer::Viewer;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

/// The folder downloads and the session file go in: `<user root>\updates`.
pub fn updates_dir() -> Option<PathBuf> {
    paths::user_root().map(|r| r.join("updates"))
}

/// The session file's name in [`updates_dir`].
pub const SESSION_FILE: &str = "session.scene";

/// `<user root>\updates\session.scene`.
pub fn session_path() -> Option<PathBuf> {
    updates_dir().map(|d| d.join(SESSION_FILE))
}

/// A `.part` this old is no live download's: a running job writes to its own at least every read
/// timeout (30 s) or fails. Younger ones may be another window's download, and are left.
pub const STALE_PART: std::time::Duration = std::time::Duration::from_secs(120);

/// The startup cleanup of an installed build (plan Task 4 ruling): in `dir` (`<user root>\updates`)
/// only, and only plain files directly in it, removes
/// - every `*.part` older than [`STALE_PART`] (a download a killed or closed app left behind), and
/// - every installer (`AtomicSandbox-Setup-*.exe`) but this version's own — the one that just
///   installed it, which may still be running (it started this app) and goes at a later start.
///
/// The session file and anything else are left; links are never followed. Returns what was removed.
pub fn clean_updates(dir: &Path, current: &semver::Version) -> Vec<PathBuf> {
    clean_updates_at(dir, current, std::time::SystemTime::now())
}

fn clean_updates_at(dir: &Path, current: &semver::Version, now: std::time::SystemTime) -> Vec<PathBuf> {
    let keep = super::release::installer_name(current);
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut removed = Vec::new();
    for e in rd.flatten() {
        // `DirEntry::file_type` does not follow links: a link is neither a file nor removed
        if !e.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        // the time through a handle (`fs::metadata`), not the directory entry's (`DirEntry::metadata`,
        // which NTFS may update late while another window's download still writes the file)
        let stale_part = name.ends_with(".part")
            && std::fs::metadata(e.path()).and_then(|m| m.modified()).is_ok_and(|t| now.duration_since(t).is_ok_and(|age| age >= STALE_PART));
        let old_installer = name.starts_with("AtomicSandbox-Setup-") && name.ends_with(".exe") && name != keep;
        if (stale_part || old_installer) && std::fs::remove_file(e.path()).is_ok() {
            removed.push(e.path());
        }
    }
    removed.sort();
    removed
}

// ---- the download ----

/// Where a download's bytes come from: the reader for an asset URL, or why not. The app's is
/// [`super::net::Http`]; a closure `FnMut(&str) -> Result<Box<dyn Read + Send>, String>` is one too.
pub trait Source: Send {
    fn open(&mut self, url: &str) -> Result<Box<dyn Read + Send>, String>;
}

impl<F> Source for F
where
    F: FnMut(&str) -> Result<Box<dyn Read + Send>, String> + Send,
{
    fn open(&mut self, url: &str) -> Result<Box<dyn Read + Send>, String> {
        self(url)
    }
}

/// A download's state as [`Download::poll`] reports it.
#[derive(Clone, Debug, PartialEq)]
pub enum DownloadState {
    /// The installer's bytes so far and the size the release lists (0 before it starts streaming).
    Running(u64, u64),
    /// Verified and renamed: the installer ([`Download::verified`] hands it to [`hand_off`]).
    Ready(PathBuf),
    /// Nothing usable was kept (the `.part` is removed); the reason.
    Failed(String),
}

/// An installer that passed [`verify`] and was renamed into place: only [`Download::verified`]
/// makes one (from [`DownloadState::Ready`]), and [`hand_off`] takes nothing else.
#[derive(Debug, PartialEq, Eq)]
pub struct Verified(PathBuf);

impl Verified {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

/// [`DownloadState::Failed`]'s reason after [`Download::cancel`].
pub const CANCELLED: &str = "download cancelled";

/// The largest `SHA256SUMS.txt` or `.minisig` taken (they are a few hundred bytes).
const SMALL_MAX: u64 = 1 << 20;

/// A release's installer downloading on its own thread (module doc). Dropping it cancels it.
pub struct Download {
    tag: String,
    rx: Receiver<DownloadState>,
    state: Mutex<DownloadState>,
    cancel: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Download {
    /// Starts downloading `release`'s installer into [`updates_dir`] over HTTPS, to be checked
    /// against the compiled-in release key. Refused at once — nothing fetched — for a release that
    /// can't be installed from the app ([`installable`]), a build without a release key, or no
    /// per-user folder.
    pub fn start(release: &Release) -> Result<Download, String> {
        let dir = updates_dir().ok_or("there is no per-user folder to download into (LOCALAPPDATA isn't set)")?;
        let source = Box::new(super::net::Http::new());
        #[cfg(not(test))]
        return Download::start_with(release, dir, source);
        #[cfg(test)]
        return Download::start_with(release, dir, source, RELEASE_KEY);
    }

    /// [`Download::start`] into `dir`, from `source`. Outside the tests the key is always the
    /// compiled-in [`RELEASE_KEY`]; the tests pass theirs (the TEST-ONLY fixture key).
    ///
    /// Refused before anything is fetched: [`installable`]'s refusals, no key, an installer listed
    /// at 0 bytes (the size bounds what is written to disk), and any of the three assets whose URL
    /// isn't exactly its release's `<tag>/<name>` ([`super::net::check_asset`]).
    pub(crate) fn start_with(release: &Release, dir: PathBuf, source: Box<dyn Source>, #[cfg(test)] pubkey: &'static str) -> Result<Download, String> {
        #[cfg(not(test))]
        let pubkey = RELEASE_KEY;
        let (setup, sums, sig) = installable(release)?;
        if !has_key(pubkey) {
            return Err(NO_RELEASE_KEY.into());
        }
        for a in [setup, sums, sig] {
            super::net::check_asset(&release.tag, a)?;
        }
        if setup.size == 0 {
            return Err(format!("the release lists {} as 0 bytes: it can't be installed from the app", setup.name));
        }
        let part = part_path(&dir, &setup.name);
        let job = Job { setup: setup.clone(), sums: sums.clone(), sig: sig.clone(), dir, part, source, pubkey };
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let thread = std::thread::Builder::new()
            .name("update-download".into())
            .spawn(move || job.run(&tx, &flag))
            .map_err(|e| format!("couldn't start the download: {e}"))?;
        Ok(Download { tag: release.tag.clone(), rx, state: Mutex::new(DownloadState::Running(0, setup.size)), cancel, thread: Some(thread) })
    }

    /// The release's tag (`v0.2.0`).
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// The latest state (never blocks).
    pub fn poll(&self) -> DownloadState {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match self.rx.try_recv() {
                Ok(s) => *state = s,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // the thread ended: it always sends its outcome first, unless it panicked
                    if matches!(*state, DownloadState::Running(..)) {
                        *state = DownloadState::Failed("the download stopped unexpectedly".into());
                    }
                    break;
                }
            }
        }
        state.clone()
    }

    /// The verified installer, once the state is [`DownloadState::Ready`]: what [`hand_off`] takes.
    pub fn verified(&self) -> Option<Verified> {
        match self.poll() {
            DownloadState::Ready(p) => Some(Verified(p)),
            _ => None,
        }
    }

    /// Stops the download: the thread removes the `.part` and reports [`CANCELLED`] after its
    /// current read (at most the 30 s read timeout on a stalled connection). Never blocks.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Waits for the thread (a panic in it included) and returns the final state (tests).
    #[cfg(test)]
    fn finish(mut self) -> DownloadState {
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.poll()
    }
}

impl Drop for Download {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Each job's own `.part` (M3: a cancelled job still finishing a read never shares a file with the
/// next one): `<name>.<pid>-<n>.part`.
fn part_path(dir: &Path, name: &str) -> PathBuf {
    static JOBS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = JOBS.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("{name}.{}-{n}.part", std::process::id()))
}

/// Removes the job's `.part` when the job ends however it ends — a refusal, a cancel, a panic
/// (unwinding) — and harmlessly after a successful rename, when it is gone already.
struct PartGuard<'a>(&'a Path);

impl Drop for PartGuard<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

struct Job {
    setup: Asset,
    sums: Asset,
    sig: Asset,
    dir: PathBuf,
    part: PathBuf,
    source: Box<dyn Source>,
    pubkey: &'static str,
}

impl Job {
    fn run(mut self, tx: &Sender<DownloadState>, cancel: &AtomicBool) {
        let part = self.part.clone();
        let _guard = PartGuard(&part);
        let outcome = self.fetch(&part, tx, cancel);
        let _ = tx.send(match outcome {
            Ok(path) => DownloadState::Ready(path),
            Err(e) => DownloadState::Failed(e),
        });
    }

    fn fetch(&mut self, part: &Path, tx: &Sender<DownloadState>, cancel: &AtomicBool) -> Result<PathBuf, String> {
        let sums = self.small(&self.sums.clone())?;
        let sig = self.small(&self.sig.clone())?;
        if cancel.load(Ordering::Relaxed) {
            return Err(CANCELLED.into());
        }
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("couldn't create {}: {e}", self.dir.display()))?;
        let digest = self.stream(part, tx, cancel)?;
        verify(&sums, &sig, &self.setup.name, &digest, self.pubkey)?;
        let done = self.dir.join(&self.setup.name);
        // the rename replaces atomically (Windows: replace-existing semantics), so the installer's
        // name only ever holds one whole verified file — whichever job renamed last, never a part
        std::fs::rename(part, &done).map_err(|e| format!("couldn't rename {} to {}: {e}", part.display(), done.display()))?;
        Ok(done)
    }

    /// A small asset whole, in memory.
    fn small(&mut self, a: &Asset) -> Result<Vec<u8>, String> {
        let r = self.source.open(&a.url).map_err(|e| format!("couldn't download {}: {e}", a.name))?;
        let mut bytes = Vec::new();
        r.take(SMALL_MAX + 1).read_to_end(&mut bytes).map_err(|e| format!("the download of {} stopped part-way: {e}", a.name))?;
        if bytes.len() as u64 > SMALL_MAX {
            return Err(format!("{} is larger than {SMALL_MAX} bytes: not a checksum or signature file", a.name));
        }
        Ok(bytes)
    }

    /// The installer to `part`, hashed as it arrives: its SHA-256.
    fn stream(&mut self, part: &Path, tx: &Sender<DownloadState>, cancel: &AtomicBool) -> Result<[u8; 32], String> {
        let (name, total) = (self.setup.name.clone(), self.setup.size);
        let mut r = self.source.open(&self.setup.url).map_err(|e| format!("couldn't download {name}: {e}"))?;
        let mut f = std::fs::File::create(part).map_err(|e| format!("couldn't write {}: {e}", part.display()))?;
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        let mut got = 0u64;
        let _ = tx.send(DownloadState::Running(0, total));
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            let n = match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("the download of {name} stopped part-way ({got} of {total} bytes): {e}")),
            };
            got += n as u64;
            if got > total {
                return Err(format!("the download of {name} sent more than the {total} bytes the release lists"));
            }
            hash.update(&buf[..n]);
            f.write_all(&buf[..n]).map_err(|e| format!("couldn't write {}: {e}", part.display()))?;
            let _ = tx.send(DownloadState::Running(got, total));
        }
        if got < total {
            return Err(format!("the download of {name} stopped part-way ({got} of {total} bytes)"));
        }
        // a full disk may only show here
        f.sync_all().map_err(|e| format!("couldn't write {}: {e}", part.display()))?;
        Ok(hash.finalize().into())
    }
}

// ---- the session ----

/// What a restart after an update brings back (module doc, "The session").
#[derive(Clone, Debug, PartialEq)]
pub struct Session {
    /// Positions, velocities, charge, multiplicity, grid, σ, the scene's ionic policy.
    pub scene: Scene,
    /// The MD time, atomic time units.
    pub sim_time: f64,
    pub backend: Backend,
    /// The quality: the spacing presets load at and the Custom slider's.
    pub quality_h: f64,
    pub quality_slider: f64,
    /// The box later quality changes keep (Bohr).
    pub box_len: f64,
    /// The policy the panel shows and presets load under (`UiState::ionic`).
    pub ionic: ::sim::IonicPolicy,
    pub walls_on: bool,
    pub wall_half_width: f64,
    /// What the chip names, and where Save scene writes (the user's own file, not this one).
    pub source: SceneSource,
    pub scene_path: String,
}

fn backend_word(b: Backend) -> &'static str {
    match b {
        Backend::Cpu => "cpu",
        Backend::Gpu => "gpu",
        Backend::Estimated => "estimated",
    }
}

fn backend_from(w: &str) -> Option<Backend> {
    [Backend::Cpu, Backend::Gpu, Backend::Estimated].into_iter().find(|&b| backend_word(b) == w)
}

impl Session {
    /// The viewer's session now: its scene as it stands (it follows the worker's snapshots) and
    /// the snapshot's MD time.
    pub fn of(v: &Viewer) -> Session {
        let u = &v.ui;
        Session {
            scene: v.scene.clone(),
            sim_time: v.snapshot.as_ref().map_or(0.0, |s| s.dynamics.sim_time),
            backend: v.backend,
            quality_h: u.quality_h,
            quality_slider: u.quality_slider,
            box_len: u.box_len,
            ionic: u.ionic,
            walls_on: u.walls_on,
            wall_half_width: u.wall_half_width,
            source: u.source.clone(),
            scene_path: u.scene_path.clone(),
        }
    }

    /// The file's text (module doc): the `#session` lines, then the scene as Save scene writes it.
    pub fn to_text(&self) -> String {
        let source = match &self.source {
            SceneSource::Preset(p) => format!("preset {p}"),
            SceneSource::File(f) => format!("file {f}"),
            SceneSource::Built(f) => format!("built {f}"),
            SceneSource::Segment(f) => format!("segment {f}"),
        };
        let mut s = String::from(
            "# atomic sandbox session: the scene as it stood before an update, which --resume-session\n\
             # reopens (paused). Its #session lines are the run's settings; any build opens the rest as a\n\
             # plain scene file.\n",
        );
        s += &format!("#session backend {}\n", backend_word(self.backend));
        s += &format!("#session time {}\n", self.sim_time);
        s += &format!("#session quality {} {}\n", self.quality_h, self.quality_slider);
        s += &format!("#session box {}\n", self.box_len);
        s += &format!("#session ionic {}\n", self.ionic.word());
        s += &format!("#session walls {} {}\n", if self.walls_on { "on" } else { "off" }, self.wall_half_width);
        s += &format!("#session source {source}\n");
        s += &format!("#session path {}\n", self.scene_path);
        s + &self.scene.to_text()
    }

    /// Reads [`Session::to_text`]'s form; the scene must pass [`Scene::validate`].
    pub fn parse(text: &str) -> Result<Session, String> {
        let scene = Scene::parse(text)?;
        scene.validate()?;
        let mut keys: std::collections::HashMap<&str, &str> = Default::default();
        for line in text.lines() {
            let Some(rest) = line.trim_end_matches('\r').strip_prefix("#session ") else { continue };
            let (k, v) = rest.split_once(' ').unwrap_or((rest, ""));
            keys.insert(k, v);
        }
        let get = |k: &str| keys.get(k).copied().ok_or_else(|| format!("no '#session {k}' line"));
        let num = |k: &str, w: &str| match w.trim().parse::<f64>() {
            Ok(x) if x.is_finite() => Ok(x),
            _ => Err(format!("'#session {k}': bad number '{w}'")),
        };
        let backend = backend_from(get("backend")?.trim()).ok_or("'#session backend' takes cpu, gpu or estimated")?;
        let sim_time = num("time", get("time")?)?;
        if sim_time < 0.0 {
            return Err(format!("'#session time' {sim_time} is negative"));
        }
        let (q, slider) = get("quality")?.trim().split_once(' ').ok_or("'#session quality' takes two spacings")?;
        let (quality_h, quality_slider) = (num("quality", q)?, num("quality", slider)?);
        let box_len = num("box", get("box")?)?;
        let ionic = ::sim::IonicPolicy::parse(get("ionic")?.trim()).ok_or("'#session ionic' takes none, cores, accepted or all")?;
        let (on, hw) = get("walls")?.trim().split_once(' ').ok_or("'#session walls' takes on|off and a half-width")?;
        let walls_on = match on {
            "on" => true,
            "off" => false,
            _ => return Err(format!("'#session walls' takes on or off, got '{on}'")),
        };
        let wall_half_width = num("walls", hw)?;
        let (kind, what) = get("source")?.split_once(' ').ok_or("'#session source' takes a kind and a name")?;
        let source = match kind {
            "preset" => SceneSource::Preset(what.to_string()),
            "file" => SceneSource::File(what.to_string()),
            "built" => SceneSource::Built(what.to_string()),
            "segment" => SceneSource::Segment(what.to_string()),
            _ => return Err(format!("'#session source' kind '{kind}' isn't preset, file, built or segment")),
        };
        let scene_path = get("path")?.to_string();
        Ok(Session { scene, sim_time, backend, quality_h, quality_slider, box_len, ionic, walls_on, wall_half_width, source, scene_path })
    }

    /// Puts the session into a viewer whose worker hasn't started (the backend is the caller's:
    /// `main` starts the window on [`Session::backend`]). The run starts paused, as every start does;
    /// the clock goes to the first driver ([`Viewer::resume_clock`]).
    pub fn apply(self, v: &mut Viewer) {
        let mut scene = self.scene;
        // a segment's mask is not loaded while the feature is deferred (as File ▸ Load)
        scene.drop_deferred_train_mask(crate::segment::SEGMENT_TRAINING);
        let u = &mut v.ui;
        u.adopt(&scene); // the pending grid
        u.source = self.source;
        u.scene_path = self.scene_path;
        u.ionic = self.ionic;
        u.quality_h = self.quality_h;
        u.quality_slider = self.quality_slider;
        u.box_len = self.box_len;
        u.walls_on = self.walls_on;
        u.wall_half_width = self.wall_half_width.min(crate::ui::max_wall_half_width(&scene));
        v.scene = scene;
        v.playing = false;
        v.resume_clock = (self.sim_time > 0.0).then_some(self.sim_time);
    }
}

/// Writes the viewer's [`Session`] to `path` whole: a temporary file beside it, renamed over it.
pub fn save_session(v: &Viewer, path: &Path) -> Result<(), String> {
    write_whole(path, &Session::of(v).to_text())
}

fn write_whole(path: &Path, text: &str) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("couldn't save the session to {}: {e}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(fail)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let written = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written.map_err(fail)
}

/// The session in `path` (`--resume-session`). The file is kept.
pub fn load_session(path: &Path) -> Result<Session, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("couldn't read {}: {e}", path.display()))?;
    Session::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

// ---- the hand-off ----

/// The installer's command line: `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /WAITPID=<wait_pid>`,
/// then [`relaunch_arg`]. The installer's own path is quoted by `Command` (spaces, non-ASCII
/// letters).
///
/// `/WAITPID=<pid>` (for the Task 5 script, review M6): the process id of the app handing off. The
/// script waits for that process to exit (e.g. `OpenProcess(SYNCHRONIZE)` + `WaitForSingleObject`;
/// a pid that is already gone means go on) before it replaces any file, so the installer never
/// races the app's own exit; `CloseApplications` still deals with any other copy.
pub fn installer_command(installer: &Path, exe_after: &Path, session: &Path, wait_pid: u32) -> Command {
    let mut c = Command::new(installer);
    c.args(["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]);
    c.arg(format!("/WAITPID={wait_pid}"));
    let relaunch = relaunch_arg(exe_after, session);
    // verbatim: Rust's own quoting would escape the quotes as \", which Inno Setup doesn't read
    #[cfg(windows)]
    std::os::windows::process::CommandExt::raw_arg(&mut c, relaunch);
    #[cfg(not(windows))]
    c.arg(relaunch);
    c
}

/// `/RELAUNCH="<exe_after> --resume-session <session>"`: the whole value in one pair of quotes,
/// so paths with spaces stay inside one parameter. The paths are not quoted again inside it: Inno
/// Setup's parameter reader drops quotes and has no escape for one inside a quoted value. The
/// installer script (Task 5) reads `{param:RELAUNCH}` = `<exe_after> --resume-session <session>`,
/// splits it at the first ` --resume-session `, and runs the first half with
/// `--resume-session "<second half>"`. A Windows path never holds a `"`.
pub fn relaunch_arg(exe_after: &Path, session: &Path) -> OsString {
    let mut s = OsString::from("/RELAUNCH=\"");
    s.push(exe_after.as_os_str());
    s.push(" --resume-session ");
    s.push(session.as_os_str());
    s.push("\"");
    s
}

/// The installer is running: close the app now, through its own clean-exit path (the profile
/// flushed, the window gone; `main`'s close). The installer waits for this process to exit
/// (`/WAITPID`); its `CloseApplications` deals with another copy still holding the exe.
#[must_use = "the app must now close through its own exit path: the installer is waiting to replace it"]
#[derive(Debug, PartialEq, Eq)]
pub struct CloseForUpdate;

/// `DETACHED_PROCESS` and `CREATE_NEW_PROCESS_GROUP` (Win32 process creation flags): the installer
/// has no console of ours and no share in our Ctrl+C, so it outlives the app's exit.
#[cfg(windows)]
const DETACHED: u32 = 0x0000_0008 | 0x0000_0200;

/// Starts the verified `installer` detached, to wait for this process to exit, install silently
/// and relaunch this exe with `--resume-session <session>` ([`installer_command`]), then asks the
/// caller to close ([`CloseForUpdate`]). It takes only a [`Verified`] installer (from
/// [`Download::verified`]). An installed build only: a portable copy offers the zip instead and a
/// dev build has updates off, so both are refused here too.
pub fn hand_off(installer: &Verified, session: &Path) -> Result<CloseForUpdate, String> {
    let installer = installer.path();
    match paths::mode() {
        DataMode::Installed => {}
        DataMode::Portable => return Err("a portable copy isn't updated in place: download the zip instead".into()),
        DataMode::Dev => return Err("running from source: updates are off".into()),
    }
    if !installer.is_file() {
        return Err(format!("the installer {} isn't there", installer.display()));
    }
    let exe = std::env::current_exe().map_err(|e| format!("couldn't find this program's path to relaunch it: {e}"))?;
    let mut c = installer_command(installer, &exe, session, std::process::id());
    c.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut c, DETACHED);
    c.spawn().map_err(|e| format!("couldn't start the installer {}: {e}", installer.display()))?;
    Ok(CloseForUpdate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::parse_releases;

    const RELEASES: &str = include_str!("fixtures/releases.json");
    const SUMS: &[u8] = include_bytes!("fixtures/SHA256SUMS.txt");
    const SIG: &[u8] = include_bytes!("fixtures/SHA256SUMS.txt.minisig");
    const TEST_KEY: &str = include_str!("fixtures/TEST-ONLY.pub");
    /// The bytes the fixture `SHA256SUMS.txt` lists as the v0.2.0 installer (`verify`'s tests
    /// signed the sums from them; no real installer is committed).
    const FIXTURE_SETUP: &[u8] = b"AtomicSandbox-Setup-0.2.0.exe: a stand-in for the installer, test only\n";
    const SETUP: &str = "AtomicSandbox-Setup-0.2.0.exe";

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("atomic-update-apply-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// The fixture's v0.2.0, its installer's listed size the stand-in's.
    fn release() -> Release {
        let mut r = parse_releases(RELEASES).unwrap().into_iter().find(|r| r.tag == "v0.2.0").unwrap();
        r.assets.iter_mut().find(|a| a.name == SETUP).unwrap().size = FIXTURE_SETUP.len() as u64;
        r
    }

    /// A source serving the fixture sums and signature, and `setup` as the installer; the URLs it
    /// was asked for go to the returned list.
    fn source(setup: impl Fn() -> Box<dyn Read + Send> + Send + 'static) -> (Box<dyn Source>, Arc<Mutex<Vec<String>>>) {
        let asked = Arc::new(Mutex::new(Vec::new()));
        let log = asked.clone();
        let f = move |url: &str| -> Result<Box<dyn Read + Send>, String> {
            log.lock().unwrap().push(url.to_string());
            let name = url.rsplit('/').next().unwrap();
            match name {
                "SHA256SUMS.txt" => Ok(Box::new(SUMS)),
                "SHA256SUMS.txt.minisig" => Ok(Box::new(SIG)),
                SETUP => Ok(setup()),
                _ => Err(format!("no {name}")),
            }
        };
        (Box::new(f), asked)
    }

    /// A reader that yields `bytes` 16 at a time, then ends — or, with `drops`, fails (a
    /// connection that drops).
    struct Trickle {
        bytes: &'static [u8],
        drops: bool,
    }
    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.bytes.is_empty() {
                return if self.drops { Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "connection reset")) } else { Ok(0) };
            }
            let n = self.bytes.len().min(buf.len()).min(16);
            buf[..n].copy_from_slice(&self.bytes[..n]);
            self.bytes = &self.bytes[n..];
            Ok(n)
        }
    }

    fn files_in(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).map_or_else(|_| Vec::new(), |rd| rd.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect());
        v.sort();
        v
    }

    /// The signed sums and the signature first, then the installer: verified, renamed from
    /// `.part`, the progress reaching the listed size.
    #[test]
    fn a_verified_download_is_renamed_from_part() {
        let dir = temp("ok");
        let (src, asked) = source(|| Box::new(FIXTURE_SETUP));
        let d = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap();
        assert_eq!(d.tag(), "v0.2.0");
        while matches!(d.poll(), DownloadState::Running(..)) {
            std::thread::yield_now();
        }
        assert_eq!(d.verified().as_ref().map(Verified::path), Some(dir.join(SETUP).as_path()), "Ready makes a Verified");
        let state = d.finish();
        assert_eq!(state, DownloadState::Ready(dir.join(SETUP)));
        assert_eq!(std::fs::read(dir.join(SETUP)).unwrap(), FIXTURE_SETUP);
        assert_eq!(files_in(&dir), [SETUP], "no .part left");
        let asked = asked.lock().unwrap().clone();
        let names: Vec<&str> = asked.iter().map(|u| u.rsplit('/').next().unwrap()).collect();
        assert_eq!(names, ["SHA256SUMS.txt", "SHA256SUMS.txt.minisig", SETUP], "the small files first");
        assert!(asked.iter().all(|u| crate::update::net::is_release_asset(u)), "{asked:?}");
        // a retry over a finished one: the same outcome
        let (src, _) = source(|| Box::new(FIXTURE_SETUP));
        assert_eq!(Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap().finish(), DownloadState::Ready(dir.join(SETUP)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Progress is `(bytes, total)` and only grows; it ends at the listed size.
    #[test]
    fn progress_counts_up_to_the_listed_size() {
        let dir = temp("progress");
        let (src, _) = source(|| Box::new(Trickle { bytes: FIXTURE_SETUP, drops: false }));
        let d = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap();
        let mut seen = Vec::new();
        loop {
            match d.poll() {
                DownloadState::Running(b, t) => {
                    assert_eq!(t, FIXTURE_SETUP.len() as u64);
                    seen.push(b);
                    std::thread::yield_now();
                }
                DownloadState::Ready(_) => break,
                DownloadState::Failed(e) => panic!("{e}"),
            }
        }
        assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
        assert!(d.finish() == DownloadState::Ready(dir.join(SETUP)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Review Focus 2: a download that stops part-way is never used. A connection that drops, a
    /// stream that ends short of the listed size, and one longer than listed are each refused, and
    /// nothing is left behind — no installer, no `.part`; an installer listed at 0 bytes is refused
    /// before anything is fetched. A retry then starts clean (its own `.part`) and succeeds.
    #[test]
    fn a_truncated_download_is_rejected_and_removed() {
        let dir = temp("truncated");
        let half = &FIXTURE_SETUP[..FIXTURE_SETUP.len() / 2];
        // the connection drops part-way
        let (src, _) = source(move || Box::new(Trickle { bytes: half, drops: true }));
        let DownloadState::Failed(e) = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap().finish() else { panic!() };
        assert!(e.contains(&format!("stopped part-way ({} of {} bytes)", half.len(), FIXTURE_SETUP.len())) && e.contains("connection reset"), "{e}");
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        // the stream ends early, cleanly
        let (src, _) = source(move || Box::new(half));
        let DownloadState::Failed(e) = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap().finish() else { panic!() };
        assert!(e.contains(&format!("stopped part-way ({} of {} bytes)", half.len(), FIXTURE_SETUP.len())), "{e}");
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        // listed at 0 bytes: refused up front, nothing fetched (the size bounds what is written)
        let mut unsized_release = release();
        unsized_release.assets.iter_mut().find(|a| a.name == SETUP).unwrap().size = 0;
        let (src, asked) = source(move || Box::new(half));
        let e = Download::start_with(&unsized_release, dir.clone(), src, TEST_KEY).err().unwrap();
        assert_eq!(e, format!("the release lists {SETUP} as 0 bytes: it can't be installed from the app"));
        assert!(asked.lock().unwrap().is_empty());
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        // more than the release lists
        let mut longer = FIXTURE_SETUP.to_vec();
        longer.extend_from_slice(b"tail");
        let longer: &'static [u8] = Box::leak(longer.into_boxed_slice());
        let (src, _) = source(move || Box::new(longer));
        let DownloadState::Failed(e) = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap().finish() else { panic!() };
        assert!(e.contains("sent more than"), "{e}");
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        // a stale .part from a killed run is never reused: the retry writes its own (the stale file
        // is Task 4's startup cleanup, left untouched here)
        let stale = format!("{SETUP}.1-0.part");
        std::fs::write(dir.join(&stale), b"stale bytes from an earlier attempt").unwrap();
        let (src, _) = source(|| Box::new(FIXTURE_SETUP));
        assert_eq!(Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap().finish(), DownloadState::Ready(dir.join(SETUP)));
        assert_eq!(files_in(&dir), [SETUP.to_string(), stale]);
        assert_eq!(std::fs::read(dir.join(SETUP)).unwrap(), FIXTURE_SETUP);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A flipped byte fails verification: removed, with `verify`'s reason.
    #[test]
    fn a_corrupted_download_fails_verification_and_is_removed() {
        let dir = temp("corrupt");
        let mut bad = FIXTURE_SETUP.to_vec();
        bad[3] ^= 1;
        let bad: &'static [u8] = Box::leak(bad.into_boxed_slice());
        let (src, _) = source(move || Box::new(bad));
        let DownloadState::Failed(e) = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap().finish() else { panic!() };
        assert!(e.contains("doesn't match SHA256SUMS.txt"), "{e}");
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A job that panics mid-stream leaves no `.part` (the drop guard runs as it unwinds) and is
    /// reported, not stuck at Running.
    #[test]
    fn a_panicking_download_leaves_no_part() {
        let dir = temp("panic");
        struct Bomb(usize);
        impl Read for Bomb {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.0 += 1;
                assert!(self.0 < 3, "a reader that panics (test)");
                buf[0] = b'x';
                Ok(1)
            }
        }
        let (src, _) = source(|| Box::new(Bomb(0)));
        let d = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap();
        assert_eq!(d.finish(), DownloadState::Failed("the download stopped unexpectedly".into()));
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two jobs of the same asset at once write separate `.part` files; both verify, and the
    /// installer's name ends up holding one whole verified copy, nothing else left.
    #[test]
    fn concurrent_downloads_never_share_a_part() {
        let dir = temp("concurrent");
        let jobs: Vec<Download> = (0..2)
            .map(|_| {
                let (src, _) = source(|| Box::new(Trickle { bytes: FIXTURE_SETUP, drops: false }));
                Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap()
            })
            .collect();
        for d in jobs {
            assert_eq!(d.finish(), DownloadState::Ready(dir.join(SETUP)));
        }
        assert_eq!(files_in(&dir), [SETUP]);
        assert_eq!(std::fs::read(dir.join(SETUP)).unwrap(), FIXTURE_SETUP);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Each asset's URL is tied to its own release: the sums of another tag, an asset on another
    /// path, or an encoded name are refused before anything is fetched.
    #[test]
    fn asset_urls_are_tied_to_their_release() {
        let dir = temp("bound");
        let with = |name: &str, url: &str| {
            let mut r = release();
            r.assets.iter_mut().find(|a| a.name == name).unwrap().url = url.to_string();
            r
        };
        for r in [
            with("SHA256SUMS.txt", "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.1.0/SHA256SUMS.txt"),
            with("SHA256SUMS.txt.minisig", "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/other.minisig"),
            with(SETUP, "https://github.com/ItsKodas/atomic-sandbox/releases/download/v0.2.0/%2e%2e/AtomicSandbox-Setup-0.2.0.exe"),
            with(SETUP, "https://objects.githubusercontent.com/AtomicSandbox-Setup-0.2.0.exe"),
        ] {
            let (src, asked) = source(|| Box::new(FIXTURE_SETUP));
            assert!(Download::start_with(&r, dir.clone(), src, TEST_KEY).err().unwrap().contains("isn't this release's"));
            assert!(asked.lock().unwrap().is_empty());
        }
        let mut odd = release();
        odd.tag = "v0.2.0%2f..".into();
        let (src, _) = source(|| Box::new(FIXTURE_SETUP));
        assert!(Download::start_with(&odd, dir.clone(), src, TEST_KEY).err().unwrap().contains("isn't a plain file name"));
        assert!(files_in(&dir).is_empty());
    }

    /// Refused before anything is fetched: the placeholder key (this build's), an unsigned
    /// release; a refused fetch of the sums is a failure that leaves nothing.
    #[test]
    fn nothing_is_fetched_without_a_key_or_a_signature() {
        let dir = temp("refused");
        let fetched = Arc::new(Mutex::new(0usize));
        let count = |n: Arc<Mutex<usize>>| -> Box<dyn Source> {
            Box::new(move |_: &str| -> Result<Box<dyn Read + Send>, String> {
                *n.lock().unwrap() += 1;
                Err("offline".into())
            })
        };
        let placeholder = "# no release key yet\n";
        assert_eq!(Download::start_with(&release(), dir.clone(), count(fetched.clone()), placeholder).err().unwrap(), NO_RELEASE_KEY);
        if !has_key(RELEASE_KEY) {
            assert_eq!(Download::start(&release()).err().unwrap(), NO_RELEASE_KEY, "this build has no key yet");
        }
        let unsigned = parse_releases(RELEASES).unwrap().into_iter().find(|r| r.tag == "v0.1.0").unwrap();
        assert!(Download::start_with(&unsigned, dir.clone(), count(fetched.clone()), TEST_KEY).err().unwrap().contains("no signature"));
        assert_eq!(*fetched.lock().unwrap(), 0);
        let DownloadState::Failed(e) = Download::start_with(&release(), dir.clone(), count(fetched.clone()), TEST_KEY).unwrap().finish() else { panic!() };
        assert_eq!(e, "couldn't download SHA256SUMS.txt: offline");
        assert!(files_in(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cancelling stops the thread mid-stream and removes the `.part`.
    #[test]
    fn a_cancelled_download_is_removed() {
        let dir = temp("cancel");
        let (gate_tx, gate_rx) = std::sync::mpsc::sync_channel::<()>(0);
        let gate = Mutex::new(Some(gate_rx));
        /// A reader that yields a byte per read, each read waiting for the test's go-ahead.
        struct Gated(std::sync::mpsc::Receiver<()>);
        impl Read for Gated {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let _ = self.0.recv();
                buf[0] = b'x';
                Ok(1)
            }
        }
        let (src, _) = source(move || Box::new(Gated(gate.lock().unwrap().take().unwrap())));
        let d = Download::start_with(&release(), dir.clone(), src, TEST_KEY).unwrap();
        gate_tx.send(()).unwrap(); // one byte in: the .part exists
        while d.poll() != DownloadState::Running(1, FIXTURE_SETUP.len() as u64) {
            std::thread::yield_now();
        }
        let parts = files_in(&dir);
        assert!(parts.len() == 1 && parts[0].starts_with(&format!("{SETUP}.{}-", std::process::id())) && parts[0].ends_with(".part"), "{parts:?}");
        assert_eq!(d.verified(), None, "nothing verified while it runs");
        d.cancel();
        drop(gate_tx); // the blocked read returns
        assert_eq!(d.finish(), DownloadState::Failed(CANCELLED.into()));
        assert!(files_in(&dir).is_empty(), "{:?}", files_in(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Review Focus 3: paths with spaces and non-ASCII letters stay whole. The installer runs
    /// silently with the relaunch value in one pair of quotes, which Inno Setup's reader (quotes
    /// dropped, no escape) hands back as one parameter whose halves are the two paths.
    #[test]
    fn installer_command_line_quotes_paths() {
        let installer = Path::new(r"C:\Users\Zoë Ångström\AppData\Local\atomic-sandbox\updates\AtomicSandbox-Setup-0.2.0.exe");
        let exe = Path::new(r"C:\Users\Zoë Ångström\AppData\Local\Programs\Atomic Sandbox\AtomicSandbox.exe");
        let session = Path::new(r"C:\Users\Zoë Ångström\AppData\Local\atomic-sandbox\updates\session.scene");
        let c = installer_command(installer, exe, session, 4242);
        assert_eq!(c.get_program(), installer.as_os_str());
        let args: Vec<String> = c.get_args().map(|a| a.to_str().unwrap().to_string()).collect();
        let relaunch = format!("/RELAUNCH=\"{} --resume-session {}\"", exe.display(), session.display());
        assert_eq!(args, ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/WAITPID=4242", relaunch.as_str()]);
        // Inno Setup's (Delphi's) parameter reader over the tail as passed (only the raw
        // argument holds spaces, inside its quotes): whitespace splits outside quotes, quotes dropped
        let params = delphi_params(&args.join(" "));
        assert_eq!(params.len(), 5, "{params:?}");
        assert_eq!(params[3], "/WAITPID=4242");
        let value = params[4].strip_prefix("/RELAUNCH=").unwrap();
        let (exe_back, session_back) = value.split_once(" --resume-session ").unwrap();
        assert_eq!((Path::new(exe_back), Path::new(session_back)), (exe, session));
    }

    /// Delphi's `ParamStr` split (what Inno Setup's `{param:…}` reads): runs of non-space
    /// characters, a quote toggling whether spaces split, quotes themselves dropped.
    fn delphi_params(s: &str) -> Vec<String> {
        let (mut out, mut cur, mut quoted, mut any) = (Vec::new(), String::new(), false, false);
        for ch in s.chars() {
            match ch {
                '"' => {
                    quoted = !quoted;
                    any = true;
                }
                ' ' if !quoted => {
                    if any {
                        out.push(std::mem::take(&mut cur));
                    }
                    any = false;
                }
                c => {
                    cur.push(c);
                    any = true;
                }
            }
        }
        if any {
            out.push(cur);
        }
        out
    }

    /// The viewer's session written and loaded back: the same positions, velocities, charge,
    /// spin, grid and settings, the MD time for the first driver, paused; the file is a plain scene
    /// file to the scene parser (older builds included), and is kept after loading.
    #[test]
    fn session_scene_round_trips() {
        let dir = temp("session");
        let path = dir.join("upd ates").join(SESSION_FILE);
        let mut v = Viewer::new("h2o").unwrap();
        v.scene.nuclei[0].pos = [0.123456789012345, -1.0e-3, 2.5];
        v.scene.nuclei[1].vel = [1.5e-4, -2.25e-5, 3.0e-6];
        v.scene.nuclei[2].a = Some(2); // a deuterium (nucleus 2 is an H)
        v.scene.charge = 1;
        v.scene.multiplicity = Some(2);
        v.backend = Backend::Estimated;
        v.ui.quality_h = 0.2;
        v.ui.quality_slider = 0.2;
        v.ui.walls_on = false;
        v.ui.wall_half_width = 3.75;
        v.ui.ionic = ::sim::IonicPolicy::CoresOnly;
        v.ui.source = SceneSource::File(r"C:\Users\Zoë Ångström\scenes\my water #2.scene".into());
        v.ui.scene_path = r"C:\Users\Zoë Ångström\scenes\my water #2.scene".into();
        v.playing = true;
        let mut want = Session::of(&v);
        assert_eq!(want.sim_time, 0.0, "no snapshot yet");
        want.sim_time = 413.413_733_352_143_6; // 10 fs
        write_whole(&path, &want.to_text()).unwrap();
        let back = load_session(&path).unwrap();
        assert_eq!(back, want);
        assert!(path.exists(), "the file is kept");
        assert_eq!(files_in(path.parent().unwrap()), [SESSION_FILE], "no temporary file left");
        // any build's scene parser reads the body: the #session lines are comments to it
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(Scene::parse(&text).unwrap(), v.scene);
        assert!(text.lines().filter(|l| !l.trim().is_empty()).all(|l| l.starts_with('#') || v.scene.to_text().lines().any(|s| s == l)), "only comments are added:\n{text}");
        // a fresh viewer resumes it, paused, the clock waiting for its first driver
        let mut w = Viewer::new("h2").unwrap();
        w.playing = true;
        back.apply(&mut w);
        assert_eq!(w.scene, v.scene);
        assert!(!w.playing && w.sim.is_none(), "paused, nothing started");
        assert_eq!(w.resume_clock, Some(want.sim_time));
        let u = &w.ui;
        assert_eq!((u.quality_h, u.quality_slider, u.box_len, u.walls_on, u.wall_half_width, u.ionic), (0.2, 0.2, v.ui.box_len, false, 3.75, ::sim::IonicPolicy::CoresOnly));
        assert_eq!((&u.source, u.scene_path.as_str(), u.grid_n, u.grid_h), (&v.ui.source, v.ui.scene_path.as_str(), v.scene.grid_n, v.scene.grid_h));
        assert_eq!(want.backend, Backend::Estimated, "main starts the window on the session's backend");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The session reader refuses a damaged file and ignores a newer build's extra keys.
    #[test]
    fn session_parse_is_strict_but_takes_unknown_keys() {
        let mut s = Session::of(&Viewer::new("h2").unwrap());
        s.sim_time = 12.5;
        s.backend = Backend::Gpu;
        s.walls_on = true;
        let text = s.to_text();
        assert_eq!(Session::parse(&text.replace("#session time 12.5\n", "#session time 12.5\n#session thermostat 300 100\n")).unwrap(), s);
        for (from, to) in [
            ("#session time 12.5\n", ""),
            ("#session time 12.5", "#session time -1"),
            ("#session time 12.5", "#session time NaN"),
            ("#session backend gpu", "#session backend tpu"),
            ("#session walls on", "#session walls maybe"),
            ("#session source preset h2", "#session source cloud h2"),
            ("grid ", "grod "),
        ] {
            assert!(text.contains(from), "{from}");
            assert!(Session::parse(&text.replace(from, to)).is_err(), "{from} → {to}");
        }
    }

    /// Task 4's startup cleanup: stale `.part`s and every installer but this version's go; a
    /// fresh `.part` (another window's download), the session file, other files and a folder stay;
    /// nothing outside the folder is touched; a missing folder is no error.
    #[test]
    fn startup_cleanup_removes_stale_parts_and_old_installers() {
        let root = temp("cleanup");
        let dir = root.join("updates");
        std::fs::create_dir_all(dir.join("AtomicSandbox-Setup-0.0.9.exe")).unwrap(); // a folder of that name
        for f in [
            "AtomicSandbox-Setup-0.2.0.exe.4242-0.part",
            "AtomicSandbox-Setup-0.1.0.exe",
            "AtomicSandbox-Setup-0.2.0.exe",
            "AtomicSandbox-Setup-0.3.0-beta.1.exe",
            SESSION_FILE,
            "notes.txt",
        ] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::write(root.join("AtomicSandbox-Setup-0.1.0.exe"), b"outside").unwrap();
        let now = std::time::SystemTime::now();
        let current = semver::Version::parse("0.2.0").unwrap();
        // the .part was just written: it may be a live download, so it stays this time
        let removed = clean_updates_at(&dir, &current, now);
        assert_eq!(removed, [dir.join("AtomicSandbox-Setup-0.1.0.exe"), dir.join("AtomicSandbox-Setup-0.3.0-beta.1.exe")]);
        // two minutes on it is stale
        let removed = clean_updates_at(&dir, &current, now + STALE_PART);
        assert_eq!(removed, [dir.join("AtomicSandbox-Setup-0.2.0.exe.4242-0.part")]);
        assert_eq!(files_in(&dir), ["AtomicSandbox-Setup-0.0.9.exe", "AtomicSandbox-Setup-0.2.0.exe", "notes.txt", SESSION_FILE]);
        assert!(root.join("AtomicSandbox-Setup-0.1.0.exe").is_file(), "outside updates\\: untouched");
        assert!(clean_updates(&root.join("nowhere"), &current).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
