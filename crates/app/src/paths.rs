//! Where the app's data lives (installer spec 2026-11-12, "Where data lives" and its builder-check
//! rulings): one module decides, and every folder lookup goes through [`dir`].
//!
//! **Three modes**, from a marker file beside the executable ([`mode_for`]):
//! - **Portable** (`portable.txt`, the release zip) and **Dev** (no marker: `cargo run`,
//!   `start.bat`, the tests and the replay scripts) keep exactly the lookups the app always had —
//!   `models/`, `scenes/`, `profiles/` and `timings/` beside the exe, else in the working directory;
//!   `captures/` beside the exe when a `models/` is there, else in the working directory.
//! - **Installed** (`installed.txt`, written by the installer) keeps the user's `models\` (with
//!   `models\collections`), `profiles\`, `scenes\` and `captures\` under the per-user root
//!   [`user_root`], `%LOCALAPPDATA%\atomic-sandbox`, created on first use. The shipped, read-only
//!   content sits in the install folder's `bundled\{models,scenes}` ([`bundled_dir`]) and is copied
//!   into the root once ([`seed_installed`]).
//!
//! Every mode already shares four things under the root: `current.profile`, `timings.txt`,
//! `training\` and `cache\pseudo` (`providers::pseudo::default_disk_cache_dir`). In Installed mode
//! [`DataDir::Timings`] is the root itself, so the "shipped" timings are the user's own
//! `timings.txt` there (an installed build ships none) and no extra `timings\` folder is made.
//!
//! **Nothing here moves, deletes or converts user data.** [`copy_tree_new`] only adds files that
//! are missing, through a `*.part` file renamed into place, so nothing is ever half-written or
//! overwritten; the first-run seed and File ▸ Import data from a folder… both use it.
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The marker beside the exe of the portable zip (its data stays beside the exe).
pub const PORTABLE_MARKER: &str = "portable.txt";
/// The marker beside the exe the installer writes.
pub const INSTALLED_MARKER: &str = "installed.txt";
/// Under the root: the first installed launch copied the bundled models and scenes in.
pub const SEEDED_MARKER: &str = "installed_seeded.txt";
/// Under the root: the folders File ▸ Import data from a folder… copied from, one per line.
pub const MIGRATED_FROM: &str = "migrated_from.txt";
/// The disabled Import item's tooltip, and the action's refusal outside Installed mode.
pub const IMPORT_ONLY_INSTALLED: &str = "only the installed app keeps its data in %LOCALAPPDATA%";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataMode {
    /// No marker: the repository checkout or a `build.bat` dist folder, the lookups as always.
    Dev,
    /// `portable.txt` beside the exe: the lookups as always (the data stays beside the exe).
    Portable,
    /// `installed.txt` beside the exe: the user's folders under [`user_root`].
    Installed,
}

/// The folders the app reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataDir {
    Models,
    /// The shared collections: `collections` inside [`DataDir::Models`].
    Collections,
    Scenes,
    Profiles,
    /// Completion times shipped with the repository (`timings/timings.txt`; `crate::timings`).
    Timings,
    Captures,
}

/// The mode of an exe in `exe_dir`: `portable.txt` beside it → Portable (it wins over
/// `installed.txt`), else `installed.txt` → Installed, else Dev. The working directory is not
/// consulted: a marker there (the repository root under `start.bat`) never changes the mode.
pub fn mode_for(exe_dir: &Path, _cwd: &Path) -> DataMode {
    if exe_dir.join(PORTABLE_MARKER).is_file() {
        DataMode::Portable
    } else if exe_dir.join(INSTALLED_MARKER).is_file() {
        DataMode::Installed
    } else {
        DataMode::Dev
    }
}

/// The running process: its mode, the exe's folder and the working directory, read once.
struct Process {
    mode: DataMode,
    exe_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
}

fn process() -> &'static Process {
    static P: OnceLock<Process> = OnceLock::new();
    P.get_or_init(|| {
        let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
        let cwd = std::env::current_dir().ok();
        let mode = match &exe_dir {
            Some(d) => mode_for(d, cwd.as_deref().unwrap_or(Path::new(""))),
            None => DataMode::Dev,
        };
        Process { mode, exe_dir, cwd }
    })
}

/// This process's mode (cached: the exe's folder and the working directory are read once).
pub fn mode() -> DataMode {
    process().mode
}

/// The per-user root `%LOCALAPPDATA%\atomic-sandbox` (every mode keeps `current.profile`,
/// `timings.txt`, `training\` and `cache\pseudo` there); `None` without `LOCALAPPDATA`.
pub fn user_root() -> Option<PathBuf> {
    root_from(std::env::var_os("LOCALAPPDATA"))
}

/// The root under a `LOCALAPPDATA` value (an empty one counts as unset, as the pseudo cache's
/// lookup does). Paths stay `PathBuf`: spaces and non-ASCII letters pass through untouched.
fn root_from(local_app_data: Option<OsString>) -> Option<PathBuf> {
    local_app_data.filter(|v| !v.is_empty()).map(|v| PathBuf::from(v).join("atomic-sandbox"))
}

/// What a lookup depends on: passed in by the tests, [`here`] for the running process.
#[derive(Clone, Debug)]
struct Ctx {
    mode: DataMode,
    exe_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
    root: Option<PathBuf>,
}

fn here() -> Ctx {
    let p = process();
    Ctx { mode: p.mode, exe_dir: p.exe_dir.clone(), cwd: p.cwd.clone(), root: user_root() }
}

impl Ctx {
    /// Installed (with a root): under the root, created on first use. Otherwise the lookups the
    /// app always had ([`legacy`]).
    fn dir(&self, kind: DataDir) -> Option<PathBuf> {
        match (&self.mode, &self.root) {
            (DataMode::Installed, Some(root)) => {
                let d = match kind {
                    DataDir::Models => root.join("models"),
                    DataDir::Collections => root.join("models").join("collections"),
                    DataDir::Scenes => root.join("scenes"),
                    DataDir::Profiles => root.join("profiles"),
                    DataDir::Timings => root.clone(),
                    DataDir::Captures => root.join("captures"),
                };
                if let Err(e) = std::fs::create_dir_all(&d) {
                    eprintln!("data folder {}: could not create it: {e}", d.display());
                }
                Some(d)
            }
            _ => legacy(kind, self.exe_dir.as_deref(), self.cwd.as_deref()),
        }
    }

    /// Installed: the install folder's read-only `bundled\models` and `bundled\scenes`.
    fn bundled(&self, kind: DataDir) -> Option<PathBuf> {
        if self.mode != DataMode::Installed {
            return None;
        }
        let name = match kind {
            DataDir::Models => "models",
            DataDir::Scenes => "scenes",
            _ => return None,
        };
        Some(self.exe_dir.as_ref()?.join("bundled").join(name))
    }
}

/// The Dev and Portable lookups, exactly as the app had them before this module (`scene.rs`'s
/// `*_dir` and `capture::captures_dir`): the folder beside the exe, else the one in the working
/// directory, whichever exists; captures beside the exe only when a `models/` is there.
fn legacy(kind: DataDir, exe_dir: Option<&Path>, cwd: Option<&Path>) -> Option<PathBuf> {
    let find = |name: &str| [exe_dir.map(|d| d.join(name)), cwd.map(|d| d.join(name))].into_iter().flatten().find(|d| d.is_dir());
    match kind {
        DataDir::Models => find("models"),
        DataDir::Collections => Some(find("models")?.join("collections")),
        DataDir::Scenes => find("scenes"),
        DataDir::Profiles => find("profiles"),
        DataDir::Timings => find("timings"),
        DataDir::Captures => Some(match exe_dir.filter(|d| d.join("models").is_dir()) {
            Some(d) => d.join("captures"),
            None => cwd.map(Path::to_path_buf).unwrap_or_default().join("captures"),
        }),
    }
}

/// The folder for `kind` in this process's mode: Installed → under [`user_root`] (created on
/// first use); Dev and Portable → beside the exe, else in the working directory, as always.
pub fn dir(kind: DataDir) -> Option<PathBuf> {
    here().dir(kind)
}

/// The install folder's shipped, read-only content (`exe_dir\bundled\{models,scenes}`) in
/// Installed mode; `None` otherwise and for the other kinds.
pub fn bundled_dir(kind: DataDir) -> Option<PathBuf> {
    here().bundled(kind)
}

/// What [`copy_tree_new`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CopyReport {
    pub copied: usize,
    /// Files already in the destination (never overwritten).
    pub skipped_existing: usize,
    /// The files (or folders) that could not be copied, with why; nothing of them is left behind.
    pub failed: Vec<(PathBuf, String)>,
}

impl CopyReport {
    fn add(&mut self, other: CopyReport) {
        self.copied += other.copied;
        self.skipped_existing += other.skipped_existing;
        self.failed.extend(other.failed);
    }
}

/// Copies the files under `src` that are not already in `dst` (recursively; `dst` and its
/// folders created as needed). Never overwrites, moves or deletes anything: each file is copied
/// into `<name>.part` beside its target and renamed into place, so a failed copy leaves nothing
/// half-written (its `.part` is removed) and the next run starts clean. `Err` only when `src` is
/// not a readable folder or `dst` cannot be created; a file's failure is in
/// [`CopyReport::failed`] and the rest go on.
pub fn copy_tree_new(src: &Path, dst: &Path) -> Result<CopyReport, String> {
    copy_tree_with(src, dst, &|from, to| std::fs::copy(from, to).map(|_| ()))
}

/// [`copy_tree_new`] with the file copy passed in (the tests make one fail part-way).
fn copy_tree_with(src: &Path, dst: &Path, copy: &dyn Fn(&Path, &Path) -> std::io::Result<()>) -> Result<CopyReport, String> {
    if !src.is_dir() {
        return Err(format!("{} is not a folder", src.display()));
    }
    std::fs::create_dir_all(dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    let entries = std::fs::read_dir(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let mut report = CopyReport::default();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                report.failed.push((src.to_path_buf(), e.to_string()));
                continue;
            }
        };
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            match copy_tree_with(&from, &to, copy) {
                Ok(r) => report.add(r),
                Err(e) => report.failed.push((from, e)),
            }
        } else if std::fs::symlink_metadata(&to).is_ok() {
            report.skipped_existing += 1;
        } else {
            match copy_one(&from, &to, copy) {
                Ok(true) => report.copied += 1,
                Ok(false) => report.skipped_existing += 1,
                Err(e) => report.failed.push((from, e)),
            }
        }
    }
    Ok(report)
}

/// `to` with `.part` appended to its file name (`a.model` → `a.model.part`).
fn part_path(to: &Path) -> PathBuf {
    let mut name = to.file_name().map(OsString::from).unwrap_or_default();
    name.push(".part");
    to.with_file_name(name)
}

/// One file through its `.part`: `Ok(false)` when `to` appeared meanwhile (kept; ours dropped).
fn copy_one(from: &Path, to: &Path, copy: &dyn Fn(&Path, &Path) -> std::io::Result<()>) -> Result<bool, String> {
    let part = part_path(to);
    if let Err(e) = copy(from, &part) {
        let _ = std::fs::remove_file(&part);
        return Err(e.to_string());
    }
    if std::fs::symlink_metadata(to).is_ok() {
        let _ = std::fs::remove_file(&part);
        return Ok(false);
    }
    std::fs::rename(&part, to).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        e.to_string()
    })?;
    Ok(true)
}

/// The first launch of an installed build: the install folder's `bundled\models` and
/// `bundled\scenes` copied into the root's `models\` and `scenes\` ([`copy_tree_new`]: what the
/// user already has is kept), then `installed_seeded.txt` written so it runs once. Nothing
/// outside Installed mode. A failure is printed and leaves the marker unwritten, so the next
/// launch tries again (only the missing files).
pub fn seed_installed() {
    if mode() != DataMode::Installed {
        return;
    }
    let Some(root) = user_root() else {
        eprintln!("first run: no LOCALAPPDATA, so the bundled models and scenes were not copied in");
        return;
    };
    let (models, scenes) = (bundled_dir(DataDir::Models), bundled_dir(DataDir::Scenes));
    if let Some(r) = seed_from(models.as_deref(), scenes.as_deref(), &root) {
        if !r.failed.is_empty() {
            eprintln!("first run: {} of the bundled files could not be copied into {}:", r.failed.len(), root.display());
            for (p, why) in &r.failed {
                eprintln!("  {}: {why}", p.display());
            }
        }
    }
}

/// [`seed_installed`]'s work with the paths passed in: `None` when the root's marker says it
/// already ran, else what was copied. A missing bundled folder is skipped.
fn seed_from(models: Option<&Path>, scenes: Option<&Path>, root: &Path) -> Option<CopyReport> {
    let marker = root.join(SEEDED_MARKER);
    if marker.exists() {
        return None;
    }
    let mut report = CopyReport::default();
    for (src, name) in [(models, "models"), (scenes, "scenes")] {
        let Some(src) = src.filter(|s| s.is_dir()) else { continue };
        match copy_tree_new(src, &root.join(name)) {
            Ok(r) => report.add(r),
            Err(e) => report.failed.push((src.to_path_buf(), e)),
        }
    }
    if report.failed.is_empty() {
        let text = format!("bundled models and scenes copied in by Atomic Sandbox v{}\n", env!("CARGO_PKG_VERSION"));
        if let Err(e) = std::fs::write(&marker, text) {
            report.failed.push((marker, e.to_string()));
        }
    }
    Some(report)
}

/// The folders File ▸ Import data from a folder… copies, each into the same name under the root.
pub const IMPORTED: [&str; 3] = ["models", "profiles", "scenes"];

/// What an import copied ([`import_into`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportReport {
    /// The [`IMPORTED`] folders the source had.
    pub found: Vec<&'static str>,
    pub copy: CopyReport,
}

impl ImportReport {
    /// The status line: the counts, and the first failures with why.
    pub fn line(&self, src: &Path) -> String {
        if self.found.is_empty() {
            return format!("nothing imported: {} has no models, profiles or scenes folder", src.display());
        }
        let c = &self.copy;
        let mut s = format!(
            "imported {} from {}: {} file{} copied, {} already there (kept)",
            self.found.join(", "),
            src.display(),
            c.copied,
            if c.copied == 1 { "" } else { "s" },
            c.skipped_existing
        );
        if !c.failed.is_empty() {
            let first: Vec<String> = c.failed.iter().take(3).map(|(p, why)| format!("{} ({why})", p.display())).collect();
            s += &format!("; {} failed: {}{}", c.failed.len(), first.join(", "), if c.failed.len() > 3 { ", …" } else { "" });
        }
        s
    }
}

/// `Action::ImportData`'s work: `<src>/{models,profiles,scenes}` copied into the same folders
/// under `root` ([`copy_tree_new`]: nothing overwritten, nothing moved), and `src` appended to
/// the root's `migrated_from.txt` when it had any of them.
pub fn import_into(src: &Path, root: &Path) -> ImportReport {
    let mut report = ImportReport::default();
    for name in IMPORTED {
        let from = src.join(name);
        if !from.is_dir() {
            continue;
        }
        report.found.push(name);
        match copy_tree_new(&from, &root.join(name)) {
            Ok(r) => report.copy.add(r),
            Err(e) => report.copy.failed.push((from, e)),
        }
    }
    if !report.found.is_empty() {
        use std::io::Write;
        let line = format!("{}\n", src.display());
        let wrote = std::fs::OpenOptions::new().create(true).append(true).open(root.join(MIGRATED_FROM)).and_then(|mut f| f.write_all(line.as_bytes()));
        if let Err(e) = wrote {
            report.copy.failed.push((root.join(MIGRATED_FROM), e.to_string()));
        }
    }
    report
}

/// [`import_into`] on its own thread (a models folder with collections can be large): the status
/// line arrives on the receiver.
pub fn spawn_import(src: PathBuf, root: PathBuf) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = import_into(&src, &root);
        for (p, why) in &r.copy.failed {
            eprintln!("import {}: {why}", p.display());
        }
        let _ = tx.send(r.line(&src));
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty temp folder for one test (removed again by [`Tmp`]'s drop).
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let d = std::env::temp_dir().join(format!("atomic-paths-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Tmp(d)
        }
        fn mk(&self, rel: &str) -> PathBuf {
            let d = self.0.join(rel);
            std::fs::create_dir_all(&d).unwrap();
            d
        }
        fn file(&self, rel: &str, text: &str) -> PathBuf {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, text).unwrap();
            p
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn ctx(mode: DataMode, exe: &Path, cwd: &Path, root: Option<&Path>) -> Ctx {
        Ctx { mode, exe_dir: Some(exe.to_path_buf()), cwd: Some(cwd.to_path_buf()), root: root.map(Path::to_path_buf) }
    }

    /// Every `.part` file under `d`.
    fn parts(d: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(parts(&p));
            } else if p.extension().is_some_and(|x| x == "part") {
                out.push(p);
            }
        }
        out
    }

    #[test]
    fn dev_mode_when_no_marker() {
        let t = Tmp::new("dev");
        let (exe, cwd) = (t.mk("exe"), t.mk("cwd"));
        assert_eq!(mode_for(&exe, &cwd), DataMode::Dev);
        // a marker in the working directory (the repo root under start.bat) changes nothing
        t.file("cwd/installed.txt", "x");
        t.file("cwd/portable.txt", "x");
        assert_eq!(mode_for(&exe, &cwd), DataMode::Dev);
        // nor does a folder of that name beside the exe
        t.mk("exe/installed.txt");
        assert_eq!(mode_for(&exe, &cwd), DataMode::Dev);
        // the test binary itself runs in Dev mode (no marker in target/)
        assert_eq!(mode(), DataMode::Dev);
        assert_eq!(bundled_dir(DataDir::Models), None);
    }

    #[test]
    fn portable_marker_wins() {
        let t = Tmp::new("portable");
        let (exe, cwd) = (t.mk("exe"), t.mk("cwd"));
        t.file("exe/portable.txt", "");
        assert_eq!(mode_for(&exe, &cwd), DataMode::Portable);
        t.file("exe/installed.txt", "installed by AtomicSandbox-Setup-0.2.0");
        assert_eq!(mode_for(&exe, &cwd), DataMode::Portable, "portable.txt wins over installed.txt");
        // portable data stays beside the exe, as always, even with a root at hand
        let root = t.mk("root");
        let models = t.mk("exe/models");
        let c = ctx(DataMode::Portable, &exe, &cwd, Some(&root));
        assert_eq!(c.dir(DataDir::Models), Some(models.clone()));
        assert_eq!(c.dir(DataDir::Collections), Some(models.join("collections")));
        assert_eq!(c.dir(DataDir::Captures), Some(exe.join("captures")));
        assert_eq!(c.bundled(DataDir::Models), None);
        assert!(std::fs::read_dir(&root).unwrap().next().is_none(), "nothing created under the root");
    }

    #[test]
    fn installed_marker_uses_user_root() {
        let t = Tmp::new("installed");
        let (exe, cwd) = (t.mk("exe"), t.mk("cwd"));
        t.file("exe/installed.txt", "installed by AtomicSandbox-Setup-0.2.0");
        assert_eq!(mode_for(&exe, &cwd), DataMode::Installed);
        // folders beside the exe or in the working directory are not the installed build's
        for d in ["exe/models", "exe/scenes", "cwd/models", "cwd/profiles", "cwd/timings"] {
            t.mk(d);
        }
        let root = t.0.join("root");
        let c = ctx(DataMode::Installed, &exe, &cwd, Some(&root));
        let want = [
            (DataDir::Models, root.join("models")),
            (DataDir::Collections, root.join("models").join("collections")),
            (DataDir::Scenes, root.join("scenes")),
            (DataDir::Profiles, root.join("profiles")),
            (DataDir::Timings, root.clone()),
            (DataDir::Captures, root.join("captures")),
        ];
        for (kind, path) in want {
            assert_eq!(c.dir(kind), Some(path.clone()), "{kind:?}");
            assert!(path.is_dir(), "{kind:?} created on first use");
        }
        assert_eq!(c.bundled(DataDir::Models), Some(exe.join("bundled").join("models")));
        assert_eq!(c.bundled(DataDir::Scenes), Some(exe.join("bundled").join("scenes")));
        assert_eq!(c.bundled(DataDir::Profiles), None);
        // without LOCALAPPDATA the lookups fall back to the old ones rather than to nothing
        let c = ctx(DataMode::Installed, &exe, &cwd, None);
        assert_eq!(c.dir(DataDir::Models), Some(exe.join("models")));
    }

    #[test]
    fn dev_lookups_unchanged() {
        let t = Tmp::new("lookups");
        let (exe, cwd) = (t.mk("exe"), t.mk("cwd"));
        let c = ctx(DataMode::Dev, &exe, &cwd, Some(&t.0.join("root")));
        // nothing anywhere: no folder (captures: the working directory's)
        for kind in [DataDir::Models, DataDir::Collections, DataDir::Scenes, DataDir::Profiles, DataDir::Timings] {
            assert_eq!(c.dir(kind), None, "{kind:?}");
        }
        assert_eq!(c.dir(DataDir::Captures), Some(cwd.join("captures")));
        // the working directory's
        for name in ["models", "scenes", "profiles", "timings"] {
            t.mk(&format!("cwd/{name}"));
        }
        assert_eq!(c.dir(DataDir::Models), Some(cwd.join("models")));
        assert_eq!(c.dir(DataDir::Collections), Some(cwd.join("models").join("collections")));
        assert_eq!(c.dir(DataDir::Scenes), Some(cwd.join("scenes")));
        assert_eq!(c.dir(DataDir::Profiles), Some(cwd.join("profiles")));
        assert_eq!(c.dir(DataDir::Timings), Some(cwd.join("timings")));
        assert_eq!(c.dir(DataDir::Captures), Some(cwd.join("captures")));
        // the beside-exe folders are preferred, each on its own
        t.mk("exe/models");
        t.mk("exe/profiles");
        assert_eq!(c.dir(DataDir::Models), Some(exe.join("models")));
        assert_eq!(c.dir(DataDir::Collections), Some(exe.join("models").join("collections")));
        assert_eq!(c.dir(DataDir::Profiles), Some(exe.join("profiles")));
        assert_eq!(c.dir(DataDir::Scenes), Some(cwd.join("scenes")));
        assert_eq!(c.dir(DataDir::Captures), Some(exe.join("captures")), "a dist folder (models/ beside the exe) keeps its captures");
        assert!(!t.0.join("root").exists(), "dev mode creates nothing under the root");
        assert!(!exe.join("models").join("collections").exists() && !exe.join("captures").exists(), "nor any folder it only names");
        // this process: the module agrees with the lookups scene.rs and capture/mod.rs had
        let old = |name: &str| {
            let beside_exe = std::env::current_exe().ok().and_then(|e| Some(e.parent()?.join(name)));
            let in_cwd = std::env::current_dir().ok().map(|d| d.join(name));
            [beside_exe, in_cwd].into_iter().flatten().find(|d| d.is_dir())
        };
        assert_eq!(dir(DataDir::Models), old("models"));
        assert_eq!(dir(DataDir::Collections), old("models").map(|m| m.join("collections")));
        assert_eq!(dir(DataDir::Scenes), old("scenes"));
        assert_eq!(dir(DataDir::Profiles), old("profiles"));
        assert_eq!(dir(DataDir::Timings), old("timings"));
        let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
        let old_captures = match exe_dir.filter(|d| d.join("models").is_dir()) {
            Some(d) => d.join("captures"),
            None => std::env::current_dir().unwrap_or_default().join("captures"),
        };
        assert_eq!(dir(DataDir::Captures), Some(old_captures));
    }

    #[test]
    fn data_root_handles_spaces_and_non_ascii() {
        let local = r"C:\Users\Zoë Q\AppData\Local";
        assert_eq!(root_from(Some(local.into())), Some(PathBuf::from(local).join("atomic-sandbox")));
        assert_eq!(root_from(Some(local.into())).unwrap().to_str(), Some(r"C:\Users\Zoë Q\AppData\Local\atomic-sandbox"));
        assert_eq!(root_from(Some("".into())), None, "an empty LOCALAPPDATA is unset");
        assert_eq!(root_from(None), None);
        // and on disk: a root under such a folder seeds, resolves and imports like any other
        let t = Tmp::new("zoë q");
        let local = t.mk("Users/Zoë Q/AppData/Local");
        let root = root_from(Some(local.clone().into_os_string())).unwrap();
        t.file("Install Dir/bundled/models/hydrogen.model", "h");
        t.file("Install Dir/bundled/scenes/h2o ø.scene", "s");
        let exe = t.0.join("Install Dir");
        let c = ctx(DataMode::Installed, &exe, &exe, Some(&root));
        let r = seed_from(c.bundled(DataDir::Models).as_deref(), c.bundled(DataDir::Scenes).as_deref(), &root).unwrap();
        assert_eq!((r.copied, r.failed.len()), (2, 0), "{r:?}");
        assert_eq!(c.dir(DataDir::Scenes), Some(local.join("atomic-sandbox").join("scenes")));
        assert_eq!(std::fs::read_to_string(root.join("scenes").join("h2o ø.scene")).unwrap(), "s");
        t.file("Old Copy é/profiles/mine.profile", "p");
        let src = t.0.join("Old Copy é");
        let r = import_into(&src, &root);
        assert_eq!((r.found.clone(), r.copy.copied), (vec!["profiles"], 1), "{r:?}");
        assert_eq!(std::fs::read_to_string(root.join(MIGRATED_FROM)).unwrap(), format!("{}\n", src.display()));
    }

    #[test]
    fn copy_tree_new_never_overwrites() {
        let t = Tmp::new("copy");
        t.file("src/a.model", "new a");
        t.file("src/b.model", "new b");
        t.file("src/collections/h2/samples.bin", "new samples");
        t.file("dst/a.model", "the user's a");
        t.file("dst/collections/h2/samples.bin", "the user's samples");
        t.file("dst/mine.model", "only the user's");
        let (src, dst) = (t.0.join("src"), t.0.join("dst"));
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 2, failed: vec![] });
        assert_eq!(std::fs::read_to_string(dst.join("a.model")).unwrap(), "the user's a");
        assert_eq!(std::fs::read_to_string(dst.join("b.model")).unwrap(), "new b");
        assert_eq!(std::fs::read_to_string(dst.join("collections/h2/samples.bin")).unwrap(), "the user's samples");
        assert_eq!(std::fs::read_to_string(dst.join("mine.model")).unwrap(), "only the user's");
        // the source is untouched (a copy, never a move)
        assert_eq!(std::fs::read_to_string(src.join("a.model")).unwrap(), "new a");
        // again: everything is there now
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!(r, CopyReport { copied: 0, skipped_existing: 3, failed: vec![] });
        assert!(parts(&dst).is_empty());
        // into a destination that does not exist yet
        let fresh = t.0.join("fresh").join("models");
        let r = copy_tree_new(&src, &fresh).unwrap();
        assert_eq!((r.copied, r.skipped_existing), (3, 0));
        assert!(copy_tree_new(&t.0.join("missing"), &dst).is_err(), "a missing source is an error");
    }

    #[test]
    fn copy_tree_new_reports_a_failure_and_leaves_no_part_file() {
        let t = Tmp::new("copyfail");
        t.file("src/good.model", "good");
        t.file("src/bad.model", "bad");
        t.file("src/sub/also.scene", "also");
        let (src, dst) = (t.0.join("src"), t.0.join("dst"));
        // a copy that writes half of bad.model, then fails (a full disk, a pulled drive)
        let flaky = |from: &Path, to: &Path| -> std::io::Result<()> {
            if from.file_name().is_some_and(|n| n == "bad.model") {
                std::fs::write(to, "ba")?;
                return Err(std::io::Error::other("disk full"));
            }
            std::fs::copy(from, to).map(|_| ())
        };
        let r = copy_tree_with(&src, &dst, &flaky).unwrap();
        assert_eq!((r.copied, r.skipped_existing), (2, 0), "{r:?}");
        assert_eq!(r.failed.len(), 1);
        assert_eq!(r.failed[0].0, src.join("bad.model"));
        assert!(r.failed[0].1.contains("disk full"), "{:?}", r.failed);
        assert!(!dst.join("bad.model").exists(), "no half-written file");
        assert!(parts(&dst).is_empty(), "no .part left: {:?}", parts(&dst));
        assert_eq!(std::fs::read_to_string(dst.join("sub/also.scene")).unwrap(), "also");
        // a retry starts clean and completes
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 2, failed: vec![] });
        assert_eq!(std::fs::read_to_string(dst.join("bad.model")).unwrap(), "bad");
        // a real failure: a folder where a file's target should go is kept, the file is reported
        t.file("src2/x.model", "x");
        t.mk("dst2/x.model.part");
        let r = copy_tree_new(&t.0.join("src2"), &t.0.join("dst2")).unwrap();
        assert_eq!((r.copied, r.failed.len()), (0, 1), "{r:?}");
        assert!(!t.0.join("dst2/x.model").exists());
    }

    #[test]
    fn seed_runs_once() {
        let t = Tmp::new("seed");
        t.file("exe/installed.txt", "installed by AtomicSandbox-Setup-0.2.0");
        t.file("exe/bundled/models/hydrogen.model", "h");
        t.file("exe/bundled/scenes/h2o.scene", "w");
        t.file("root/scenes/h2o.scene", "the user's own h2o");
        let (exe, root) = (t.0.join("exe"), t.0.join("root"));
        let c = ctx(DataMode::Installed, &exe, &exe, Some(&root));
        let (m, s) = (c.bundled(DataDir::Models), c.bundled(DataDir::Scenes));
        let r = seed_from(m.as_deref(), s.as_deref(), &root).expect("the first launch seeds");
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 1, failed: vec![] });
        assert_eq!(std::fs::read_to_string(root.join("models/hydrogen.model")).unwrap(), "h");
        assert_eq!(std::fs::read_to_string(root.join("scenes/h2o.scene")).unwrap(), "the user's own h2o", "never overwritten");
        assert!(root.join(SEEDED_MARKER).is_file());
        // the user deletes a seeded model: the next launch does not bring it back
        std::fs::remove_file(root.join("models/hydrogen.model")).unwrap();
        assert_eq!(seed_from(m.as_deref(), s.as_deref(), &root), None);
        assert!(!root.join("models/hydrogen.model").exists());
        // a failed seed leaves the marker unwritten, so the next launch tries again
        let root2 = t.0.join("root2");
        t.mk("root2/models/hydrogen.model.part");
        let r = seed_from(m.as_deref(), s.as_deref(), &root2).unwrap();
        assert_eq!((r.copied, r.failed.len()), (1, 1), "{r:?}");
        assert!(!root2.join(SEEDED_MARKER).exists());
        std::fs::remove_dir(root2.join("models/hydrogen.model.part")).unwrap();
        let r = seed_from(m.as_deref(), s.as_deref(), &root2).unwrap();
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 1, failed: vec![] });
        assert!(root2.join(SEEDED_MARKER).is_file());
        // no bundled folder (a broken install): nothing copied, marker written
        let root3 = t.0.join("root3");
        std::fs::create_dir_all(&root3).unwrap();
        assert_eq!(seed_from(None, Some(&t.0.join("nowhere")), &root3), Some(CopyReport::default()));
        // and outside Installed mode seeding does nothing at all
        assert_eq!(ctx(DataMode::Dev, &exe, &exe, Some(&root)).bundled(DataDir::Models), None);
    }

    #[test]
    fn import_copies_the_three_folders_and_records_the_source() {
        let t = Tmp::new("import");
        t.file("old/models/h2o.model", "m");
        t.file("old/models/collections/h2o/s.bin", "c");
        t.file("old/profiles/mine.profile", "p");
        t.file("old/scenes/w.scene", "s");
        t.file("old/captures/shot.png", "not imported");
        t.file("root/scenes/w.scene", "the installed build's own");
        let (src, root) = (t.0.join("old"), t.0.join("root"));
        let r = import_into(&src, &root);
        assert_eq!(r.found, vec!["models", "profiles", "scenes"]);
        assert_eq!(r.copy, CopyReport { copied: 3, skipped_existing: 1, failed: vec![] });
        assert_eq!(std::fs::read_to_string(root.join("models/collections/h2o/s.bin")).unwrap(), "c");
        assert_eq!(std::fs::read_to_string(root.join("scenes/w.scene")).unwrap(), "the installed build's own");
        assert!(!root.join("captures").exists());
        assert!(src.join("models/h2o.model").is_file(), "copied, not moved");
        let line = r.line(&src);
        assert!(line.contains("3 files copied") && line.contains("1 already there"), "{line}");
        // a second import appends its source
        let r = import_into(&src, &root);
        assert_eq!(r.copy.copied, 0);
        assert_eq!(std::fs::read_to_string(root.join(MIGRATED_FROM)).unwrap(), format!("{0}\n{0}\n", src.display()));
        // a folder without any of the three: nothing copied, nothing recorded
        let empty = t.mk("empty");
        let root2 = t.mk("root2");
        let r = import_into(&empty, &root2);
        assert!(r.found.is_empty() && !root2.join(MIGRATED_FROM).exists());
        assert!(r.line(&empty).starts_with("nothing imported"), "{}", r.line(&empty));
        // the threaded form sends the same line
        let rx = spawn_import(src.clone(), t.0.join("root3"));
        let line = rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        assert!(line.contains("4 files copied"), "{line}");
    }

    /// The pseudopotential cache (`providers::pseudo::default_disk_cache_dir`, unchanged) sits
    /// under the same root, `cache\pseudo`, unless `ATOMIC_SANDBOX_CACHE_DIR` overrides it.
    #[test]
    fn pseudo_cache_sits_under_user_root() {
        let set = |k: &str| std::env::var_os(k).is_some_and(|v| !v.is_empty());
        if set("ATOMIC_SANDBOX_CACHE_DIR") || !set("LOCALAPPDATA") {
            return;
        }
        assert_eq!(providers::pseudo::default_disk_cache_dir(), user_root().map(|r| r.join("cache").join("pseudo")));
    }
}
