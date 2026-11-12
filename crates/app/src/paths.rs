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
//!   content sits in the install folder's `bundled\{models,scenes,timings}` ([`bundled_dir`]); the
//!   models and scenes are copied into the root at the first launch of each version
//!   ([`seed_installed`]: missing files only), the shipped completion times are read from there
//!   ([`DataDir::Timings`]).
//!
//! Every mode already shares four things under the root: `current.profile`, `timings.txt` (the
//! user's own completion times), `training\` and `cache\pseudo`
//! (`providers::pseudo::default_disk_cache_dir`).
//!
//! **Nothing here moves, deletes or converts user data.** [`copy_tree_new`] only adds files that
//! are missing, each through a fresh part file linked into place (never a replace), copies a
//! shared collection whole or not at all, never follows links and never copies a `.lock`; the
//! first-run seed and File ▸ Import data from a folder… both use it.
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The marker beside the exe of the portable zip (its data stays beside the exe).
pub const PORTABLE_MARKER: &str = "portable.txt";
/// The marker beside the exe the installer writes.
pub const INSTALLED_MARKER: &str = "installed.txt";
/// Under the root: the bundled models and scenes were copied in, by the version on its
/// `version=` line ([`seed_installed`]).
pub const SEEDED_MARKER: &str = "installed_seeded.txt";
/// Under the root: the folders File ▸ Import data from a folder… copied from, one per line.
pub const MIGRATED_FROM: &str = "migrated_from.txt";
/// The disabled Import item's tooltip, and the action's refusal outside Installed mode.
pub const IMPORT_ONLY_INSTALLED: &str = "only the installed app keeps its data in %LOCALAPPDATA%";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataMode {
    /// No marker: the repository checkout (`cargo run`, `start.bat`, the tests, the replays), the
    /// lookups as always.
    Dev,
    /// `portable.txt` beside the exe (the release zip, and a `build.bat` dist folder, which
    /// `scripts/package.ps1` gives the marker): the lookups as always (the data stays beside the
    /// exe); it checks GitHub at launch and offers "Download zip".
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
    /// Installed (with a root): under the root, created on first use — except the shipped
    /// completion times, read-only in the install folder ([`Ctx::bundled`]; the user's own
    /// `timings.txt` stays at the root). Otherwise the lookups the app always had ([`legacy`]).
    fn dir(&self, kind: DataDir) -> Option<PathBuf> {
        match (&self.mode, &self.root) {
            (DataMode::Installed, Some(root)) => {
                let d = match kind {
                    DataDir::Models => root.join("models"),
                    DataDir::Collections => root.join("models").join("collections"),
                    DataDir::Scenes => root.join("scenes"),
                    DataDir::Profiles => root.join("profiles"),
                    DataDir::Timings => return self.bundled(kind),
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

    /// Installed: the install folder's read-only `bundled\models`, `bundled\scenes` and
    /// `bundled\timings`.
    fn bundled(&self, kind: DataDir) -> Option<PathBuf> {
        if self.mode != DataMode::Installed {
            return None;
        }
        let name = match kind {
            DataDir::Models => "models",
            DataDir::Scenes => "scenes",
            DataDir::Timings => "timings",
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
/// first use; the shipped timings: [`bundled_dir`]); Dev and Portable → beside the exe, else in
/// the working directory, as always.
pub fn dir(kind: DataDir) -> Option<PathBuf> {
    here().dir(kind)
}

/// The install folder's shipped, read-only content (`exe_dir\bundled\{models,scenes,timings}`)
/// in Installed mode; `None` otherwise and for the other kinds.
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
    /// Symlinks and junctions in the source: never followed, never copied.
    pub skipped_links: Vec<PathBuf>,
    /// Collections the destination already had (by folder name): kept whole, the source's not
    /// merged into them ([`copy_tree_new`]).
    pub kept_collections: Vec<String>,
}

impl CopyReport {
    fn add(&mut self, other: CopyReport) {
        self.copied += other.copied;
        self.skipped_existing += other.skipped_existing;
        self.failed.extend(other.failed);
        self.skipped_links.extend(other.skipped_links);
        self.kept_collections.extend(other.kept_collections);
    }
}

/// The file a collection's owner holds while it has it open (`surrogate::collection`): never
/// copied, so a copy never looks held by another process.
const LOCK_FILE: &str = ".lock";

/// Copies the files under `src` that are not already in `dst` (recursively; `dst` and its
/// folders created as needed). Never overwrites, moves or deletes anything:
/// - each file is written to a fresh `<name>.<pid>.<n>.part` beside its target (`create_new`),
///   then hard-linked into place (or, where hard links are not supported, renamed without
///   replacing), which fails rather than replace a target that appeared meanwhile; the part is
///   removed after, and a failed copy leaves nothing behind;
/// - symlinks and junctions are not followed ([`CopyReport::skipped_links`]);
/// - a **collection** — a folder inside a `collections` folder, or any folder holding `key.txt`
///   and `samples.bin` — is copied whole (into a part folder renamed into place) only when the
///   destination has no folder of that name, else kept as it is ([`CopyReport::kept_collections`]):
///   two collections are never merged file by file;
/// - `.lock` files are never copied.
///
/// `Err` only when `src` is not a folder (or is a link) or `dst` cannot be created; a file's
/// failure is in [`CopyReport::failed`] and the rest go on.
pub fn copy_tree_new(src: &Path, dst: &Path) -> Result<CopyReport, String> {
    copy_tree_with(src, dst, &copy_contents)
}

/// The file copy [`copy_tree_new`] uses: `from`'s bytes into the freshly created part file, its
/// modification time kept.
fn copy_contents(from: &Path, out: &mut std::fs::File) -> std::io::Result<()> {
    let mut f = std::fs::File::open(from)?;
    std::io::copy(&mut f, out)?;
    if let Ok(t) = f.metadata().and_then(|m| m.modified()) {
        let _ = out.set_modified(t);
    }
    Ok(())
}

/// The file copy seam: `from`'s contents into the open part file (the tests make one fail part-way).
type CopyFn<'a> = &'a dyn Fn(&Path, &mut std::fs::File) -> std::io::Result<()>;

/// [`copy_tree_new`] with the file copy passed in.
fn copy_tree_with(src: &Path, dst: &Path, copy: CopyFn) -> Result<CopyReport, String> {
    let meta = std::fs::symlink_metadata(src).map_err(|e| format!("{}: {e}", src.display()))?;
    if meta.file_type().is_symlink() {
        return Err(format!("{} is a link (not followed)", src.display()));
    }
    if !meta.is_dir() {
        return Err(format!("{} is not a folder", src.display()));
    }
    std::fs::create_dir_all(dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    let entries = std::fs::read_dir(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let in_collections = src.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.eq_ignore_ascii_case("collections"));
    let mut report = CopyReport::default();
    for entry in entries {
        let (entry, kind) = match entry.and_then(|e| Ok((e.file_type()?, e))) {
            Ok((t, e)) => (e, t),
            Err(e) => {
                report.failed.push((src.to_path_buf(), e.to_string()));
                continue;
            }
        };
        let name = entry.file_name();
        let from = entry.path();
        let to = dst.join(&name);
        // `DirEntry::file_type` does not follow links; on Windows `is_symlink` is true for
        // symlinks and junctions alike
        if kind.is_symlink() {
            report.skipped_links.push(from);
        } else if kind.is_dir() {
            if in_collections || is_collection(&from) {
                copy_collection(&from, &to, &name.to_string_lossy(), copy, &mut report);
            } else {
                match copy_tree_with(&from, &to, copy) {
                    Ok(r) => report.add(r),
                    Err(e) => report.failed.push((from, e)),
                }
            }
        } else if name == LOCK_FILE {
            // a lock names its owner's process: never copied
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

/// A folder that is a shared collection (`surrogate::collection`'s layout): `key.txt` and
/// `samples.bin` in it.
fn is_collection(d: &Path) -> bool {
    // follows links on purpose: a collection whose files are links is still routed to
    // `copy_collection`, which then refuses it whole rather than copying it in part
    d.join("key.txt").is_file() && d.join("samples.bin").is_file()
}

/// One collection, whole or not at all: kept when `to` exists; else copied into a fresh part
/// folder, renamed into place only when every file copied (and `to` is still absent), the part
/// folder removed otherwise.
fn copy_collection(from: &Path, to: &Path, key: &str, copy: CopyFn, report: &mut CopyReport) {
    if std::fs::symlink_metadata(to).is_ok() {
        report.kept_collections.push(key.to_string());
        return;
    }
    let part = unique_part(to);
    if let Err(e) = std::fs::create_dir(&part) {
        report.failed.push((from.to_path_buf(), e.to_string()));
        return;
    }
    let r = match copy_tree_with(from, &part, copy) {
        Ok(r) if r.failed.is_empty() && r.skipped_links.is_empty() => r,
        // a skipped link inside a collection would leave it partly copied: whole or not at all
        Ok(r) => {
            let _ = std::fs::remove_dir_all(&part);
            if r.failed.is_empty() {
                report.failed.push((from.to_path_buf(), "a link inside the collection (links are not followed): not copied".to_string()));
            }
            report.skipped_links.extend(r.skipped_links);
            report.failed.extend(r.failed);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&part);
            report.failed.push((from.to_path_buf(), e));
            return;
        }
    };
    match rename_no_replace(&part, to) {
        Ok(()) => report.add(r),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&part);
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                report.kept_collections.push(key.to_string());
            } else {
                report.failed.push((from.to_path_buf(), e.to_string()));
            }
        }
    }
}

/// A name beside `to` that no one else uses: `<name>.<pid>.<n>.part`.
fn unique_part(to: &Path) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut name = to.file_name().map(OsString::from).unwrap_or_default();
    name.push(format!(".{}.{n}.part", std::process::id()));
    to.with_file_name(name)
}

/// One file through a fresh part file: `Ok(false)` when `to` exists by the time it is placed
/// (kept; ours dropped).
fn copy_one(from: &Path, to: &Path, copy: CopyFn) -> Result<bool, String> {
    let part = unique_part(to);
    // `create_new`: the part is ours alone, so removing it never removes anything else
    let mut out = std::fs::OpenOptions::new().write(true).create_new(true).open(&part).map_err(|e| e.to_string())?;
    let wrote = copy(from, &mut out);
    drop(out);
    if let Err(e) = wrote {
        let _ = std::fs::remove_file(&part);
        return Err(e.to_string());
    }
    let placed = match std::fs::hard_link(&part, to) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        // no hard links here (FAT, some network shares): a rename that refuses an existing target
        Err(_) => match rename_no_replace(&part, to) {
            Ok(()) => return Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(e) => Err(e.to_string()),
        },
    };
    let _ = std::fs::remove_file(&part);
    placed
}

/// Renames `from` to `to`, failing (`AlreadyExists`) when `to` exists: never a replace.
#[cfg(windows)]
fn rename_no_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    let wide = |p: &Path| p.as_os_str().encode_wide().chain(Some(0)).collect::<Vec<u16>>();
    let (a, b) = (wide(from), wide(to));
    // flags 0: without MOVEFILE_REPLACE_EXISTING an existing target is an error
    if unsafe { windows_sys::Win32::Storage::FileSystem::MoveFileExW(a.as_ptr(), b.as_ptr(), 0) } != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Renames `from` to `to`, failing when `to` exists (best effort off Windows, which ships no build).
#[cfg(not(windows))]
fn rename_no_replace(from: &Path, to: &Path) -> std::io::Result<()> {
    if std::fs::symlink_metadata(to).is_ok() {
        return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
    }
    std::fs::rename(from, to)
}

/// The first launch of each installed version: the install folder's `bundled\models` and
/// `bundled\scenes` copied into the root's `models\` and `scenes\` ([`copy_tree_new`]: only the
/// missing files, so what the user already has — edited or not — is kept), then
/// `installed_seeded.txt` written with this version, so it runs once per version (final review
/// I2: a model or library scene an update adds reaches an installed user). Nothing outside
/// Installed mode. A failure is printed and leaves the marker as it was, so the next launch tries
/// again (only the missing files).
///
/// Accepted trade-off: a bundled file the user deleted comes back at the first launch after an
/// update (never at a later launch of the same version).
pub fn seed_installed() {
    if mode() != DataMode::Installed {
        return;
    }
    let Some(root) = user_root() else {
        eprintln!("bundled content: no LOCALAPPDATA, so the bundled models and scenes were not copied in");
        return;
    };
    let (models, scenes) = (bundled_dir(DataDir::Models), bundled_dir(DataDir::Scenes));
    if let Some(r) = seed_from(models.as_deref(), scenes.as_deref(), &root) {
        if !r.failed.is_empty() {
            eprintln!("bundled content: not everything could be copied into {} (the next launch tries again):", root.display());
            for (p, why) in &r.failed {
                eprintln!("  {}: {why}", p.display());
            }
        }
    }
}

/// [`seed_installed`]'s work with the paths passed in: `None` when the root's marker says this
/// version already seeded, else what was copied. A marker without a `version=` line (the format
/// before final review I2) or with another version's seeds again — missing files only, never an
/// overwrite or a delete — and is rewritten. A missing bundled folder is skipped; the root is
/// created for the marker.
///
/// A root with neither `models\` nor `scenes\` is seeded again whatever the marker says (installer
/// Task 5 fix round 1, I2): the uninstaller's "Also remove my models, profiles, scenes and
/// captures" takes those folders but leaves the marker, and a reinstall must not open with no
/// bundled models or library scenes. It still never overwrites ([`copy_tree_new`]).
fn seed_from(models: Option<&Path>, scenes: Option<&Path>, root: &Path) -> Option<CopyReport> {
    let marker = root.join(SEEDED_MARKER);
    let version = env!("CARGO_PKG_VERSION");
    let emptied = !root.join("models").exists() && !root.join("scenes").exists();
    if seeded_version(&marker).as_deref() == Some(version) && !emptied {
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
        let text = format!("version={version}\nbundled models and scenes copied in (missing files only) by Atomic Sandbox v{version}; another version copies again what is missing\n");
        if let Err(e) = std::fs::create_dir_all(root).and_then(|()| std::fs::write(&marker, text)) {
            report.failed.push((marker, e.to_string()));
        }
    }
    Some(report)
}

/// The version on the seed marker's `version=` line; `None` when there is no marker, it can't be
/// read, or it has no such line (the first format).
fn seeded_version(marker: &Path) -> Option<String> {
    let text = std::fs::read_to_string(marker).ok()?;
    text.lines().find_map(|l| l.trim().strip_prefix("version=")).map(|v| v.trim().to_string())
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
    /// The status line: the counts, the collections kept, the links skipped, and the first
    /// failures with why.
    pub fn line(&self, src: &Path) -> String {
        let c = &self.copy;
        if self.found.is_empty() {
            let links = if c.skipped_links.is_empty() { String::new() } else { format!(" ({} of them a link, not followed)", c.skipped_links.len()) };
            return format!("nothing imported: {} has no models, profiles or scenes folder{links}", src.display());
        }
        let mut s = format!(
            "imported {} from {}: {} file{} copied, {} already there (kept)",
            self.found.join(", "),
            src.display(),
            c.copied,
            if c.copied == 1 { "" } else { "s" },
            c.skipped_existing
        );
        if !c.kept_collections.is_empty() {
            s += &format!("; kept yours: {} (the older copy's collection not merged in)", c.kept_collections.join(", "));
        }
        if !c.skipped_links.is_empty() {
            s += &format!("; {} link{} skipped (not followed)", c.skipped_links.len(), if c.skipped_links.len() == 1 { "" } else { "s" });
        }
        if !c.failed.is_empty() {
            let first: Vec<String> = c.failed.iter().take(3).map(|(p, why)| format!("{} ({why})", p.display())).collect();
            s += &format!("; {} failed: {}{}", c.failed.len(), first.join(", "), if c.failed.len() > 3 { ", …" } else { "" });
        }
        s
    }
}

/// `Action::ImportData`'s work: `<src>/{models,profiles,scenes}` copied into the same folders
/// under `root` ([`copy_tree_new`]: nothing overwritten, nothing moved, collections whole), and
/// `src` appended to the root's `migrated_from.txt` when it had any of them. A folder of the
/// three that is itself a link is skipped. [`check_import`] refuses a source that overlaps the
/// root first.
pub fn import_into(src: &Path, root: &Path) -> ImportReport {
    let mut report = ImportReport::default();
    for name in IMPORTED {
        let from = src.join(name);
        match std::fs::symlink_metadata(&from) {
            Ok(m) if m.file_type().is_symlink() => {
                report.copy.skipped_links.push(from);
                continue;
            }
            Ok(m) if m.is_dir() => {}
            _ => continue,
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
        let wrote = std::fs::create_dir_all(root)
            .and_then(|()| std::fs::OpenOptions::new().create(true).append(true).open(root.join(MIGRATED_FROM)))
            .and_then(|mut f| f.write_all(line.as_bytes()));
        if let Err(e) = wrote {
            report.copy.failed.push((root.join(MIGRATED_FROM), e.to_string()));
        }
    }
    report
}

/// `p` resolved as far as it exists (links and `..` resolved by the OS), the rest appended: so a
/// root that does not exist yet still compares by its real parent.
fn resolved(p: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(p) {
        return c;
    }
    match (p.parent(), p.file_name()) {
        (Some(parent), Some(name)) if parent != p => resolved(parent).join(name),
        _ => p.to_path_buf(),
    }
}

/// Whether File ▸ Import data from a folder… may copy `src` into the data root, given the mode and
/// the root (passed in, so the tests reach the Installed branch): the root to import into, or the
/// refusal for the status line. Refused outside Installed mode, without a root, and when `src` is
/// the root, inside it or holds it (an import of the root into itself).
pub fn check_import(mode: DataMode, root: Option<&Path>, src: &Path) -> Result<PathBuf, String> {
    if mode != DataMode::Installed {
        return Err(format!("import data: {IMPORT_ONLY_INSTALLED}"));
    }
    let root = root.ok_or("import data: no LOCALAPPDATA folder to import into")?;
    let (s, r) = (resolved(src), resolved(root));
    if s.starts_with(&r) || r.starts_with(&s) {
        return Err(format!(
            "import data: {} {} this build's own data folder ({}); pick an older copy's folder",
            src.display(),
            if s == r { "is" } else if s.starts_with(&r) { "is inside" } else { "holds" },
            root.display()
        ));
    }
    Ok(root.to_path_buf())
}

/// [`check_import`], then [`import_into`] on its own thread (a models folder with collections can
/// be large): the status line arrives on the receiver.
pub fn start_import(mode: DataMode, root: Option<&Path>, src: PathBuf) -> Result<std::sync::mpsc::Receiver<String>, String> {
    let root = check_import(mode, root, &src)?;
    Ok(spawn_import(src, root))
}

/// [`import_into`] on its own thread: the status line arrives on the receiver.
fn spawn_import(src: PathBuf, root: PathBuf) -> std::sync::mpsc::Receiver<String> {
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
        fn read(&self, rel: &str) -> String {
            std::fs::read_to_string(self.0.join(rel)).unwrap()
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

    /// Every `.part` file or folder under `d`.
    fn parts(d: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "part") {
                out.push(p);
            } else if e.file_type().unwrap().is_dir() {
                out.extend(parts(&p));
            }
        }
        out
    }

    /// A directory link `link` → `target`: a symlink where that is permitted, else (Windows, no
    /// developer mode) a junction, which needs no privilege; `false` when neither can be made.
    fn dir_link(target: &Path, link: &Path) -> bool {
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(target, link).is_ok() {
                return true;
            }
            std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(link)
                .arg(target)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        }
        #[cfg(not(windows))]
        {
            std::os::unix::fs::symlink(target, link).is_ok()
        }
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
        assert_eq!(bundled_dir(DataDir::Timings), None);
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
        let timings = t.mk("exe/timings");
        let c = ctx(DataMode::Portable, &exe, &cwd, Some(&root));
        assert_eq!(c.dir(DataDir::Models), Some(models.clone()));
        assert_eq!(c.dir(DataDir::Collections), Some(models.join("collections")));
        assert_eq!(c.dir(DataDir::Timings), Some(timings));
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
            (DataDir::Captures, root.join("captures")),
        ];
        for (kind, path) in want {
            assert_eq!(c.dir(kind), Some(path.clone()), "{kind:?}");
            assert!(path.is_dir(), "{kind:?} created on first use");
        }
        // the shipped completion times: read-only in the install folder, never created; the
        // user's own timings.txt stays at the root (`main`)
        let shipped = exe.join("bundled").join("timings");
        assert_eq!(c.dir(DataDir::Timings), Some(shipped.clone()));
        assert!(!shipped.exists());
        assert_eq!(c.bundled(DataDir::Timings), Some(shipped));
        assert_eq!(c.bundled(DataDir::Models), Some(exe.join("bundled").join("models")));
        assert_eq!(c.bundled(DataDir::Scenes), Some(exe.join("bundled").join("scenes")));
        assert_eq!(c.bundled(DataDir::Profiles), None);
        assert_eq!(c.bundled(DataDir::Captures), None);
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
        assert_eq!(check_import(DataMode::Installed, Some(&root), &src), Ok(root.clone()));
        let r = import_into(&src, &root);
        assert_eq!((r.found.clone(), r.copy.copied), (vec!["profiles"], 1), "{r:?}");
        assert_eq!(std::fs::read_to_string(root.join(MIGRATED_FROM)).unwrap(), format!("{}\n", src.display()));
    }

    #[test]
    fn copy_tree_new_never_overwrites() {
        let t = Tmp::new("copy");
        t.file("src/a.model", "new a");
        t.file("src/b.model", "new b");
        t.file("src/sub/c.scene", "new c");
        t.file("dst/a.model", "the user's a");
        t.file("dst/sub/c.scene", "the user's c");
        t.file("dst/mine.model", "only the user's");
        let (src, dst) = (t.0.join("src"), t.0.join("dst"));
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 2, ..Default::default() });
        assert_eq!(t.read("dst/a.model"), "the user's a");
        assert_eq!(t.read("dst/b.model"), "new b");
        assert_eq!(t.read("dst/sub/c.scene"), "the user's c");
        assert_eq!(t.read("dst/mine.model"), "only the user's");
        // the source is untouched (a copy, never a move)
        assert_eq!(t.read("src/a.model"), "new a");
        // again: everything is there now
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!(r, CopyReport { copied: 0, skipped_existing: 3, ..Default::default() });
        assert!(parts(&dst).is_empty());
        // into a destination that does not exist yet
        let fresh = t.0.join("fresh").join("models");
        let r = copy_tree_new(&src, &fresh).unwrap();
        assert_eq!((r.copied, r.skipped_existing), (3, 0));
        assert!(copy_tree_new(&t.0.join("missing"), &dst).is_err(), "a missing source is an error");
        // a source holding both a.model and a.model.part: two files, neither lost (the part
        // files are unique, `create_new`)
        t.file("src2/a.model", "a");
        t.file("src2/a.model.part", "the user's part-named file");
        let r = copy_tree_new(&t.0.join("src2"), &t.0.join("dst2")).unwrap();
        assert_eq!(r, CopyReport { copied: 2, ..Default::default() });
        assert_eq!(t.read("dst2/a.model"), "a");
        assert_eq!(t.read("dst2/a.model.part"), "the user's part-named file");
        assert_eq!(std::fs::read_dir(t.0.join("dst2")).unwrap().count(), 2, "no part file left");
    }

    /// M1: a target that appears while its part is being written (a model autosave of the same
    /// name, a second copy importing) is never replaced: the copy counts it as already there.
    #[test]
    fn a_target_that_appears_during_the_copy_is_kept() {
        let t = Tmp::new("race");
        t.file("src/a.model", "ours");
        let dst = t.mk("dst");
        let target = dst.join("a.model");
        let racing = |from: &Path, out: &mut std::fs::File| -> std::io::Result<()> {
            std::fs::write(&target, "theirs")?;
            copy_contents(from, out)
        };
        let r = copy_tree_with(&t.0.join("src"), &dst, &racing).unwrap();
        assert_eq!(r, CopyReport { skipped_existing: 1, ..Default::default() });
        assert_eq!(t.read("dst/a.model"), "theirs");
        assert!(parts(&dst).is_empty(), "{:?}", parts(&dst));
        // the rename fallback refuses an existing target too
        let (a, b) = (t.file("x", "x"), t.file("y", "y"));
        assert_eq!(rename_no_replace(&a, &b).unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!((t.read("x"), t.read("y")), ("x".to_string(), "y".to_string()));
        rename_no_replace(&a, &t.0.join("z")).unwrap();
        assert_eq!(t.read("z"), "x");
    }

    #[test]
    fn copy_tree_new_reports_a_failure_and_leaves_no_part_file() {
        let t = Tmp::new("copyfail");
        t.file("src/good.model", "good");
        t.file("src/bad.model", "bad");
        t.file("src/sub/also.scene", "also");
        let (src, dst) = (t.0.join("src"), t.0.join("dst"));
        // a copy that writes half of bad.model, then fails (a full disk, a pulled drive)
        let flaky = |from: &Path, out: &mut std::fs::File| -> std::io::Result<()> {
            if from.file_name().is_some_and(|n| n == "bad.model") {
                use std::io::Write;
                out.write_all(b"ba")?;
                return Err(std::io::Error::other("disk full"));
            }
            copy_contents(from, out)
        };
        let r = copy_tree_with(&src, &dst, &flaky).unwrap();
        assert_eq!((r.copied, r.skipped_existing), (2, 0), "{r:?}");
        assert_eq!(r.failed.len(), 1);
        assert_eq!(r.failed[0].0, src.join("bad.model"));
        assert!(r.failed[0].1.contains("disk full"), "{:?}", r.failed);
        assert!(!dst.join("bad.model").exists(), "no half-written file");
        assert!(parts(&dst).is_empty(), "no .part left: {:?}", parts(&dst));
        assert_eq!(t.read("dst/sub/also.scene"), "also");
        // a retry starts clean and completes
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 2, ..Default::default() });
        assert_eq!(t.read("dst/bad.model"), "bad");
        // a real failure: a source file another process holds open exclusively
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let held = t.file("src3/held.model", "held");
            t.file("src3/free.model", "free");
            let _lock = std::fs::OpenOptions::new().read(true).share_mode(0).open(&held).unwrap();
            let r = copy_tree_new(&t.0.join("src3"), &t.0.join("dst3")).unwrap();
            assert_eq!((r.copied, r.failed.len()), (1, 1), "{r:?}");
            assert_eq!(r.failed[0].0, held);
            assert!(!t.0.join("dst3/held.model").exists());
            assert!(parts(&t.0.join("dst3")).is_empty());
        }
    }

    /// I1: links (symlinks, junctions) are never followed: a junction back to an ancestor does not
    /// recurse, and the link is counted.
    #[test]
    fn copy_tree_new_does_not_follow_links() {
        let t = Tmp::new("links");
        t.file("src/models/a.model", "a");
        let looped = t.0.join("src/models/loop");
        if !dir_link(&t.0.join("src"), &looped) {
            eprintln!("copy_tree_new_does_not_follow_links: no directory link could be made here; skipped");
            return;
        }
        let dst = t.0.join("dst");
        let r = copy_tree_new(&t.0.join("src/models"), &dst).unwrap();
        assert_eq!((r.copied, r.failed.len()), (1, 0), "{r:?}");
        assert_eq!(r.skipped_links, vec![looped.clone()]);
        assert!(!dst.join("loop").exists());
        // a link as the folder itself is refused, and an import skips a linked models folder
        assert!(copy_tree_new(&looped, &t.0.join("dst2")).unwrap_err().contains("link"));
        let old = t.mk("old");
        assert!(dir_link(&t.0.join("src/models"), &old.join("models")));
        let r = import_into(&old, &t.0.join("root"));
        assert!(r.found.is_empty() && r.copy.skipped_links == vec![old.join("models")], "{r:?}");
        assert!(r.line(&old).contains("a link, not followed"), "{}", r.line(&old));
        // remove the links before the temp folder (`remove_dir_all` does not follow them either)
        let _ = std::fs::remove_dir(&looped);
        let _ = std::fs::remove_dir(old.join("models"));
    }

    /// I2: a collection is copied whole or not at all, never merged into the destination's, and no
    /// `.lock` is ever copied.
    #[test]
    fn collections_are_copied_whole_or_kept() {
        let t = Tmp::new("collections");
        // the installed build already learnt into an h2 collection
        t.file("root/models/collections/h2/key.txt", "h2 key");
        t.file("root/models/collections/h2/samples.bin", "installed samples");
        // the older copy: a large h2 collection, a water one, and a collection kept elsewhere
        for (k, f, s) in [
            ("h2", "key.txt", "h2 key"),
            ("h2", "samples.bin", "old samples"),
            ("h2", "history.bin", "old history"),
            ("h2", "references.txt", "old refs"),
            ("h2", ".lock", "pid 42"),
            ("h2o", "key.txt", "h2o key"),
            ("h2o", "samples.bin", "h2o samples"),
            ("h2o", ".lock", "pid 42"),
        ] {
            t.file(&format!("old/models/collections/{k}/{f}"), s);
        }
        t.file("old/models/mine/key.txt", "k");
        t.file("old/models/mine/samples.bin", "s");
        t.file("old/models/h2o.model", "m");
        t.file("old/scenes/.lock", "not a collection's, still never copied");
        t.file("old/scenes/w.scene", "w");
        t.file("root/models/mine/samples.bin", "the installed build's");
        let (src, root) = (t.0.join("old"), t.0.join("root"));
        let r = import_into(&src, &root);
        let mut kept = r.copy.kept_collections.clone();
        kept.sort();
        assert_eq!(kept, vec!["h2".to_string(), "mine".to_string()], "{r:?}");
        assert!(r.copy.failed.is_empty(), "{r:?}");
        // h2: the installed build's, untouched, with nothing of the old one beside it
        assert_eq!(t.read("root/models/collections/h2/samples.bin"), "installed samples");
        for f in ["history.bin", "references.txt", ".lock"] {
            assert!(!root.join("models/collections/h2").join(f).exists(), "{f}");
        }
        // h2o: whole, without its lock
        assert_eq!(t.read("root/models/collections/h2o/samples.bin"), "h2o samples");
        assert_eq!(t.read("root/models/collections/h2o/key.txt"), "h2o key");
        assert!(!root.join("models/collections/h2o/.lock").exists());
        assert!(!root.join("models/mine/key.txt").exists(), "a collection outside collections/ is a unit too");
        assert!(!root.join("scenes/.lock").exists());
        assert_eq!(r.copy.copied, 4, "h2o's two, h2o.model, w.scene: {r:?}");
        let line = r.line(&src);
        assert!(line.contains("kept yours: ") && line.contains("h2") && line.contains("mine"), "{line}");
        assert!(parts(&root).is_empty(), "{:?}", parts(&root));
    }

    /// I2: a collection that fails part-way leaves nothing: no half collection, no part folder.
    #[test]
    fn a_collection_that_fails_part_way_leaves_nothing() {
        let t = Tmp::new("collfail");
        t.file("src/collections/h2o/key.txt", "k");
        t.file("src/collections/h2o/samples.bin", "s");
        t.file("src/collections/h2o/history.bin", "h");
        t.file("src/plain.model", "m");
        let (src, dst) = (t.0.join("src"), t.0.join("dst"));
        let flaky = |from: &Path, out: &mut std::fs::File| -> std::io::Result<()> {
            if from.file_name().is_some_and(|n| n == "history.bin") {
                return Err(std::io::Error::other("disk full"));
            }
            copy_contents(from, out)
        };
        let r = copy_tree_with(&src, &dst, &flaky).unwrap();
        assert_eq!(r.copied, 1, "only plain.model: {r:?}");
        assert_eq!(r.failed.len(), 1);
        assert!(!dst.join("collections/h2o").exists());
        assert!(parts(&dst).is_empty(), "{:?}", parts(&dst));
        // the retry copies it whole
        let r = copy_tree_new(&src, &dst).unwrap();
        assert_eq!((r.copied, r.skipped_existing, r.failed.len()), (3, 1, 0), "{r:?}");
        assert_eq!(t.read("dst/collections/h2o/history.bin"), "h");
    }

    /// Re-review N1: a link inside a collection is not followed, so the collection would arrive
    /// part-copied; it is refused whole instead (reported as failed, no part folder left).
    #[test]
    fn a_collection_holding_a_link_is_refused_whole() {
        let t = Tmp::new("colllink");
        t.file("src/collections/h2o/key.txt", "k");
        t.file("src/collections/h2o/samples.bin", "s");
        t.file("elsewhere/extra.bin", "x");
        let link = t.0.join("src/collections/h2o/extra");
        if !dir_link(&t.0.join("elsewhere"), &link) {
            eprintln!("a_collection_holding_a_link_is_refused_whole: no directory link could be made here; skipped");
            return;
        }
        let dst = t.0.join("dst");
        let r = copy_tree_new(&t.0.join("src"), &dst).unwrap();
        assert!(!dst.join("collections/h2o").exists(), "{r:?}");
        assert_eq!(r.failed.len(), 1, "{r:?}");
        assert!(r.failed[0].1.contains("link"), "{r:?}");
        assert_eq!(r.skipped_links, vec![link.clone()]);
        assert!(parts(&dst).is_empty(), "{:?}", parts(&dst));
        let _ = std::fs::remove_dir(&link);
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
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 1, ..Default::default() });
        assert_eq!(t.read("root/models/hydrogen.model"), "h");
        assert_eq!(t.read("root/scenes/h2o.scene"), "the user's own h2o", "never overwritten");
        assert!(root.join(SEEDED_MARKER).is_file());
        assert_eq!(seeded_version(&root.join(SEEDED_MARKER)).as_deref(), Some(env!("CARGO_PKG_VERSION")));
        // the user deletes a seeded model: the next launch of this version does not bring it back
        std::fs::remove_file(root.join("models/hydrogen.model")).unwrap();
        assert_eq!(seed_from(m.as_deref(), s.as_deref(), &root), None);
        assert!(!root.join("models/hydrogen.model").exists());
        // a failed seed (a file where models\ should be) leaves the marker unwritten, so the next
        // launch tries again
        let root2 = t.0.join("root2");
        t.file("root2/models", "not a folder");
        let r = seed_from(m.as_deref(), s.as_deref(), &root2).unwrap();
        assert_eq!((r.copied, r.failed.len()), (1, 1), "{r:?}");
        assert!(!root2.join(SEEDED_MARKER).exists());
        std::fs::remove_file(root2.join("models")).unwrap();
        let r = seed_from(m.as_deref(), s.as_deref(), &root2).unwrap();
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 1, ..Default::default() });
        assert!(root2.join(SEEDED_MARKER).is_file());
        // M3: no bundled folder (a broken install) and no root yet: nothing copied, the root made
        // for the marker, so the next launch is quiet
        let root3 = t.0.join("not yet").join("root3");
        assert_eq!(seed_from(None, Some(&t.0.join("nowhere")), &root3), Some(CopyReport::default()));
        assert!(root3.join(SEEDED_MARKER).is_file());
        // (it has neither models\ nor scenes\, so a later launch looks again — and, with nothing
        // bundled, still copies nothing and says nothing)
        assert_eq!(seed_from(None, None, &root3), Some(CopyReport::default()));
        // and outside Installed mode there is nothing bundled to seed from
        assert_eq!(ctx(DataMode::Dev, &exe, &exe, Some(&root)).bundled(DataDir::Models), None);
    }

    /// Installer Task 5 fix round 1, I2: the uninstaller's "Also remove my data" takes models\ and
    /// scenes\ (and profiles\, captures\) but leaves the marker; a reinstall seeds them again. One
    /// of the two still there means the user's own: no re-seed.
    #[test]
    fn seed_runs_again_when_models_and_scenes_are_both_gone() {
        let t = Tmp::new("reseed");
        t.file("exe/bundled/models/hydrogen.model", "h");
        t.file("exe/bundled/scenes/h2o.scene", "w");
        let (m, s, root) = (t.0.join("exe/bundled/models"), t.0.join("exe/bundled/scenes"), t.0.join("root"));
        let r = seed_from(Some(&m), Some(&s), &root).expect("the first launch seeds");
        assert_eq!(r.copied, 2);
        // only scenes\ removed: models\ is there, the marker holds
        std::fs::remove_dir_all(root.join("scenes")).unwrap();
        assert_eq!(seed_from(Some(&m), Some(&s), &root), None);
        assert!(!root.join("scenes").exists());
        // both removed (the uninstaller's box): seeded again, the marker left as it was
        std::fs::remove_dir_all(root.join("models")).unwrap();
        assert!(root.join(SEEDED_MARKER).is_file(), "the uninstaller leaves the marker");
        let r = seed_from(Some(&m), Some(&s), &root).expect("re-seeded");
        assert_eq!(r, CopyReport { copied: 2, ..Default::default() });
        assert_eq!((t.read("root/models/hydrogen.model").as_str(), t.read("root/scenes/h2o.scene").as_str()), ("h", "w"));
        // and the next launch is quiet again
        assert_eq!(seed_from(Some(&m), Some(&s), &root), None);
    }

    /// Final review I2: the first launch of another version copies in what its bundle adds —
    /// missing files only, a file the user edited untouched — and rewrites the marker; a marker
    /// of the first format (no `version=` line) counts as another version; the same version
    /// copies nothing.
    #[test]
    fn seed_runs_again_for_another_version() {
        let t = Tmp::new("seed-version");
        t.file("exe/bundled/models/hydrogen.model", "h");
        t.file("exe/bundled/scenes/h2o.scene", "w");
        let (m, s, root) = (t.0.join("exe/bundled/models"), t.0.join("exe/bundled/scenes"), t.0.join("root"));
        let marker = root.join(SEEDED_MARKER);
        let this = env!("CARGO_PKG_VERSION");
        // the first format's marker (an earlier build), the user's edit of a seeded scene, and a
        // model the new bundle adds
        t.file("root/models/hydrogen.model", "h");
        t.file("root/scenes/h2o.scene", "the user's edit");
        t.file("root/installed_seeded.txt", "bundled models and scenes copied in by Atomic Sandbox v0.1.0\n");
        t.file("exe/bundled/models/helium.model", "he");
        assert_eq!(seeded_version(&marker), None, "no version= line");
        let r = seed_from(Some(&m), Some(&s), &root).expect("a marker without a version seeds again");
        assert_eq!(r, CopyReport { copied: 1, skipped_existing: 2, ..Default::default() });
        assert_eq!(t.read("root/models/helium.model"), "he", "the new bundled file is added");
        assert_eq!(t.read("root/scenes/h2o.scene"), "the user's edit", "never overwritten");
        assert_eq!(seeded_version(&marker).as_deref(), Some(this), "the marker rewritten with this version");
        // the same version: nothing copied, even a deleted bundled file stays deleted
        std::fs::remove_file(root.join("models/helium.model")).unwrap();
        t.file("exe/bundled/scenes/benzene.scene", "b");
        assert_eq!(seed_from(Some(&m), Some(&s), &root), None);
        assert!(!root.join("models/helium.model").exists() && !root.join("scenes/benzene.scene").exists());
        // another version (an update): the new file added, the existing ones untouched, the marker
        // rewritten; the deleted bundled file comes back (the accepted trade-off)
        t.file("root/installed_seeded.txt", "version=0.0.1-old\nbundled models and scenes copied in\n");
        assert_eq!(seeded_version(&marker).as_deref(), Some("0.0.1-old"));
        let r = seed_from(Some(&m), Some(&s), &root).expect("another version seeds again");
        assert_eq!(r, CopyReport { copied: 2, skipped_existing: 2, ..Default::default() });
        assert_eq!(t.read("root/scenes/benzene.scene"), "b");
        assert_eq!(t.read("root/models/helium.model"), "he", "a deleted bundled file returns after an update");
        assert_eq!((t.read("root/models/hydrogen.model").as_str(), t.read("root/scenes/h2o.scene").as_str()), ("h", "the user's edit"));
        assert_eq!(seeded_version(&marker).as_deref(), Some(this));
        assert_eq!(seed_from(Some(&m), Some(&s), &root), None, "and quiet again");
    }

    #[test]
    fn import_copies_the_three_folders_and_records_the_source() {
        let t = Tmp::new("import");
        t.file("old/models/h2o.model", "m");
        t.file("old/models/collections/h2o/key.txt", "k");
        t.file("old/models/collections/h2o/samples.bin", "c");
        t.file("old/profiles/mine.profile", "p");
        t.file("old/scenes/w.scene", "s");
        t.file("old/captures/shot.png", "not imported");
        t.file("root/scenes/w.scene", "the installed build's own");
        let (src, root) = (t.0.join("old"), t.0.join("root"));
        let r = import_into(&src, &root);
        assert_eq!(r.found, vec!["models", "profiles", "scenes"]);
        assert_eq!(r.copy, CopyReport { copied: 4, skipped_existing: 1, ..Default::default() });
        assert_eq!(t.read("root/models/collections/h2o/samples.bin"), "c");
        assert_eq!(t.read("root/scenes/w.scene"), "the installed build's own");
        assert!(!root.join("captures").exists());
        assert!(src.join("models/h2o.model").is_file(), "copied, not moved");
        let line = r.line(&src);
        assert!(line.contains("4 files copied") && line.contains("1 already there"), "{line}");
        // a second import appends its source (the collection now kept whole)
        let r = import_into(&src, &root);
        assert_eq!((r.copy.copied, r.copy.kept_collections.clone()), (0, vec!["h2o".to_string()]));
        assert_eq!(std::fs::read_to_string(root.join(MIGRATED_FROM)).unwrap(), format!("{0}\n{0}\n", src.display()));
        // a folder without any of the three: nothing copied, nothing recorded
        let empty = t.mk("empty");
        let root2 = t.mk("root2");
        let r = import_into(&empty, &root2);
        assert!(r.found.is_empty() && !root2.join(MIGRATED_FROM).exists());
        assert!(r.line(&empty).starts_with("nothing imported"), "{}", r.line(&empty));
        // the threaded form sends the same line
        let rx = start_import(DataMode::Installed, Some(&t.0.join("root3")), src.clone()).unwrap();
        let line = rx.recv_timeout(std::time::Duration::from_secs(30)).unwrap();
        assert!(line.contains("5 files copied"), "{line}");
    }

    /// I1 / M5: the import's checks, with the mode and the root passed in.
    #[test]
    fn check_import_refuses_outside_installed_and_the_root_itself() {
        let t = Tmp::new("check");
        let local = t.mk("Local");
        let root = local.join("atomic-sandbox");
        let old = t.mk("old copy");
        let only = format!("import data: {IMPORT_ONLY_INSTALLED}");
        assert_eq!(check_import(DataMode::Dev, Some(&root), &old), Err(only.clone()));
        assert_eq!(check_import(DataMode::Portable, Some(&root), &old), Err(only));
        assert!(check_import(DataMode::Installed, None, &old).unwrap_err().contains("no LOCALAPPDATA"));
        // the root not made yet: still compared by its real parent
        assert_eq!(check_import(DataMode::Installed, Some(&root), &old), Ok(root.clone()));
        let e = check_import(DataMode::Installed, Some(&root), &local).unwrap_err();
        assert!(e.contains("holds this build's own data folder"), "{e}");
        std::fs::create_dir_all(root.join("models")).unwrap();
        let e = check_import(DataMode::Installed, Some(&root), &root).unwrap_err();
        assert!(e.contains(" is this build's own data folder"), "{e}");
        let e = check_import(DataMode::Installed, Some(&root), &root.join("models")).unwrap_err();
        assert!(e.contains("is inside this build's own data folder"), "{e}");
        // spelled another way (a `..`): the same folder
        let e = check_import(DataMode::Installed, Some(&root), &root.join("models").join("..")).unwrap_err();
        assert!(e.contains(" is this build's own data folder"), "{e}");
        // a sibling whose name starts like the root's is not inside it
        let sibling = t.mk("Local/atomic-sandbox-old");
        assert_eq!(check_import(DataMode::Installed, Some(&root), &sibling), Ok(root.clone()));
        assert!(start_import(DataMode::Dev, Some(&root), old).is_err());
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
