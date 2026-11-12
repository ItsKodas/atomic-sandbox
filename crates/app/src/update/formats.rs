//! The data formats this build reads and writes, and the downgrade warnings drawn from them
//! (installer spec, "The manifest", "Downgrades" and the builder-check ruling on scenes and
//! profiles).
//!
//! Every number is read from the code's own version constant, never typed here:
//!
//! | key | constant | on disk |
//! |---|---|---|
//! | `collection` | `surrogate::collection::RECORD_VERSION` | each record of `models/collections/<key>/samples.bin` |
//! | `sidecar` | the newest model-file magic, `surrogate::sidecar::MAGIC_V8` | the magic of `models/*.model`, `training/pass.model` |
//! | `history` | `surrogate::history::MAGIC` (`ASAUDIT1`) | the magic of `models/collections/<key>/history.bin` |
//! | `replay` | `crate::replay::HEADER` | a replay script's first line (wherever the user keeps it) |
//! | `pseudo_cache` | `providers::pseudo::DISK_MAGIC` | `cache/pseudo` (a cache: rebuilt, never lost) |
//! | `checkpoint` | `providers::sampling::CHECKPOINT_MAGIC` | the first line of `training/pass.ckpt` |
//! | `scene` | `crate::scene::SCENE_FORMAT` (code-only) | none: scene files carry no version |
//! | `profile` | `crate::ui::profile::PROFILE_FORMAT` (code-only) | none |
//!
//! A downgrade never changes a file. [`downgrade_warnings`] only reads the headers of the files
//! under the per-user root to say which of them the older version won't open.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Format name → version (a manifest's `formats`).
pub type Formats = BTreeMap<String, u32>;

pub const COLLECTION: &str = "collection";
pub const SIDECAR: &str = "sidecar";
pub const HISTORY: &str = "history";
pub const REPLAY: &str = "replay";
pub const PSEUDO_CACHE: &str = "pseudo_cache";
pub const CHECKPOINT: &str = "checkpoint";
pub const SCENE: &str = "scene";
pub const PROFILE: &str = "profile";

/// The version number a magic or header string ends with (`ASAUDIT1` → 1, `… replay v1` → 1).
/// Panics without one: a constant that lost its number is a bug the tests catch.
fn trailing_number(s: &[u8]) -> u32 {
    let digits: Vec<u8> = s.iter().rev().take_while(|b| b.is_ascii_digit()).copied().collect();
    assert!(!digits.is_empty(), "{:?} ends without a version number", String::from_utf8_lossy(s));
    digits.iter().rev().fold(0u32, |n, d| n * 10 + u32::from(d - b'0'))
}

/// This build's data formats, read from the version constants (module doc).
pub fn current_formats() -> Formats {
    [
        (COLLECTION, u32::from(surrogate::collection::RECORD_VERSION)),
        (SIDECAR, trailing_number(surrogate::sidecar::MAGIC_V8)),
        (HISTORY, trailing_number(surrogate::history::MAGIC)),
        (REPLAY, trailing_number(crate::replay::HEADER.as_bytes())),
        (PSEUDO_CACHE, trailing_number(providers::pseudo::DISK_MAGIC.as_bytes())),
        (CHECKPOINT, trailing_number(providers::sampling::CHECKPOINT_MAGIC.as_bytes())),
        (SCENE, crate::scene::SCENE_FORMAT),
        (PROFILE, crate::ui::profile::PROFILE_FORMAT),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// The highest record version in a `samples.bin` (module doc of `surrogate::collection`: magic,
/// then `u32 len · u32 crc · payload` records whose payload starts `kind: u8, version: u8`),
/// walking the length prefixes without reading the payloads. `None` for a file that is not a
/// collection or holds no record. A damaged tail ends the walk (the app cuts it on open anyway).
fn collection_version(samples: &Path) -> Option<u32> {
    let mut f = std::io::BufReader::new(std::fs::File::open(samples).ok()?);
    let total = std::fs::metadata(samples).ok()?.len();
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != surrogate::collection::MAGIC {
        return None;
    }
    let (mut at, mut best) = (8u64, None::<u32>);
    loop {
        let mut head = [0u8; 10];
        if at + 10 > total || f.read_exact(&mut head).is_err() {
            return best;
        }
        let len = u64::from(u32::from_le_bytes(head[..4].try_into().expect("4 bytes")));
        if len < 2 || at + 8 + len > total {
            return best;
        }
        best = Some(best.map_or(u32::from(head[9]), |b| b.max(u32::from(head[9]))));
        at += 8 + len;
        f.seek(SeekFrom::Start(at)).ok()?;
    }
}

/// The version digit of an 8-byte magic `<prefix><digit>` (`ASMODEL8`, `ASAUDIT1`).
fn magic_version(path: &Path, prefix: &[u8; 7]) -> Option<u32> {
    let mut magic = [0u8; 8];
    std::fs::File::open(path).ok()?.read_exact(&mut magic).ok()?;
    (&magic[..7] == prefix && magic[7].is_ascii_digit()).then(|| u32::from(magic[7] - b'0'))
}

/// The version a checkpoint's first line ends with, when it is a checkpoint.
fn checkpoint_version(path: &Path) -> Option<u32> {
    let mut head = Vec::new();
    std::fs::File::open(path).ok()?.take(256).read_to_end(&mut head).ok()?;
    let first = head.split(|&b| b == b'\n').next()?;
    let first = first.strip_suffix(b"\r").unwrap_or(first);
    let prefix = providers::sampling::CHECKPOINT_MAGIC.trim_end_matches(|c: char| c.is_ascii_digit());
    (first.starts_with(prefix.as_bytes()) && first.len() > prefix.len() && first[prefix.len()..].iter().all(u8::is_ascii_digit))
        .then(|| trailing_number(first))
}

/// The entries of a folder, sorted by name (none when it can't be read).
fn entries(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
    v.sort();
    v
}

fn name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// "1 of your collections (it was …): a. It stays …" / "2 of … (they were …): a, b. They stay …"
fn files_line(tv: &str, what: &str, names: &[String], verb: &str) -> String {
    let n = names.len();
    let (were, stay, obj) = if n == 1 { ("it was", "It stays", "it") } else { ("they were", "They stay", "them") };
    format!(
        "{tv} can't {verb} {n} of your {what} ({were} written in a newer format): {}. {stay} untouched; that version just won't load {obj}.",
        names.join(", ")
    )
}

/// What a move to an older release would leave unreadable, as lines for the Version panel (empty
/// when nothing would). `target_version` is the release chosen, `target` its manifest's formats
/// (`None`: it has no manifest, e.g. `v0.1.0`), `current` this build's ([`current_formats`]) and
/// `root` the per-user data root (`paths::user_root`; tests pass a temp folder).
///
/// - No manifest: one generic line.
/// - A format whose target number is lower (a key the target lacks counts as lower):
///   - collections, saved models, an interrupted training pass — the files under `root` whose
///     headers carry a newer version are named, by folder or file;
///   - scenes, profiles, replays — one line by type (they carry no version, or live anywhere);
///   - the pseudopotential cache — one line when `root/cache/pseudo` exists (it is rebuilt).
///
/// Nothing is changed, moved or converted: the lines say so.
pub fn downgrade_warnings(target_version: &semver::Version, target: Option<&Formats>, current: &Formats, root: &Path) -> Vec<String> {
    let tv = format!("v{target_version}");
    let Some(target) = target else {
        return vec![format!("{tv} records no data formats: collections, models or profiles saved by this version may not open in it")];
    };
    let this = format!("v{}", env!("CARGO_PKG_VERSION"));
    let theirs = |k: &str| target.get(k).copied().unwrap_or(0);
    let lower = |k: &str| current.get(k).is_some_and(|&c| theirs(k) < c);
    let newer_than = |v: Option<u32>, k: &str| lower(k) && v.is_some_and(|v| v > theirs(k));
    let mut out = Vec::new();

    // collections: their records, and their audit history
    let models = root.join("models");
    if lower(COLLECTION) || lower(HISTORY) {
        let names: Vec<String> = entries(&models.join("collections"))
            .into_iter()
            .filter(|d| d.is_dir())
            .filter(|d| newer_than(collection_version(&d.join("samples.bin")), COLLECTION) || newer_than(magic_version(&d.join("history.bin"), b"ASAUDIT"), HISTORY))
            .map(|d| Path::new("models").join("collections").join(name(&d)).display().to_string())
            .collect();
        if !names.is_empty() {
            out.push(files_line(&tv, "collections", &names, "open"));
        }
    }
    // saved models
    if lower(SIDECAR) {
        let names: Vec<String> = entries(&models)
            .into_iter()
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("model")))
            .filter(|p| newer_than(magic_version(p, b"ASMODEL"), SIDECAR))
            .map(|p| Path::new("models").join(name(&p)).display().to_string())
            .collect();
        if !names.is_empty() {
            out.push(files_line(&tv, "saved models", &names, "open"));
        }
    }
    // an interrupted training pass (`worker::checkpoint`: pass.ckpt and, per scene, pass.model)
    let training = root.join("training");
    let ckpt = training.join(crate::worker::checkpoint::CHECKPOINT_FILE);
    let pass_model = training.join(crate::worker::checkpoint::MODEL_FILE);
    let stale: Vec<String> = [
        (newer_than(checkpoint_version(&ckpt), CHECKPOINT), &ckpt),
        (newer_than(magic_version(&pass_model, b"ASMODEL"), SIDECAR), &pass_model),
    ]
    .into_iter()
    .filter(|(newer, _)| *newer)
    .map(|(_, p)| Path::new("training").join(name(p)).display().to_string())
    .collect();
    if !stale.is_empty() {
        out.push(format!(
            "{tv} can't resume your interrupted training pass ({} written in a newer format). It stays untouched; that version just won't offer to resume it.",
            if stale.len() == 1 { format!("{} was", stale[0]) } else { format!("{} were", stale.join(" and ")) }
        ));
    }
    // by type
    if lower(SCENE) {
        out.push(format!("scenes saved by this version ({this}) may not open in {tv}"));
    }
    if lower(PROFILE) {
        out.push(format!("profiles saved by this version ({this}) may not open in {tv}"));
    }
    if lower(REPLAY) {
        out.push(format!("replays recorded by this version ({this}) may not play in {tv}"));
    }
    if lower(PSEUDO_CACHE) && root.join("cache").join("pseudo").is_dir() {
        out.push(format!(
            "{tv} won't reuse the pseudopotential checks this version cached ({}): it derives its own, about a minute per species the first time",
            Path::new("cache").join("pseudo").display()
        ));
    }
    // a format this table doesn't know yet (a later build added it): said by name
    let known = [COLLECTION, SIDECAR, HISTORY, REPLAY, PSEUDO_CACHE, CHECKPOINT, SCENE, PROFILE];
    for k in current.keys().filter(|k| !known.contains(&k.as_str()) && lower(k)) {
        out.push(format!("{k} data saved by this version ({this}) may not open in {tv}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> semver::Version {
        semver::Version::parse(s).unwrap()
    }

    /// A temp root, removed when dropped.
    struct Root(PathBuf);
    impl Root {
        fn new(tag: &str) -> Root {
            let d = std::env::temp_dir().join(format!("atomic-update-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Root(d)
        }
    }
    impl Drop for Root {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A `samples.bin`: the magic, then one record per version given (kind 0 first, then samples).
    fn samples_bin(versions: &[u8]) -> Vec<u8> {
        let mut b = surrogate::collection::MAGIC.to_vec();
        for (i, &ver) in versions.iter().enumerate() {
            let payload = [u8::from(i > 0), ver, 0xAA, 0xBB, 0xCC];
            b.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // the CRC is not read here
            b.extend_from_slice(&payload);
        }
        b
    }

    fn collection(root: &Path, slug: &str, versions: &[u8]) {
        let d = root.join("models").join("collections").join(slug);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("samples.bin"), samples_bin(versions)).unwrap();
        std::fs::write(d.join("history.bin"), surrogate::history::MAGIC).unwrap();
    }

    #[test]
    fn current_formats_reads_the_constants() {
        let f = current_formats();
        let want: Formats = [
            ("checkpoint", 1),
            ("collection", 1),
            ("history", 1),
            ("profile", 1),
            ("pseudo_cache", 1),
            ("replay", 1),
            ("scene", 1),
            ("sidecar", 8),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(f, want);
        assert_eq!(trailing_number(b"format v12"), 12);
        // the sidecar number is the newest magic this build reads: a v9 needs this module updated
        let newest = format!("ASMODEL{}", f["sidecar"]);
        assert_eq!(newest.as_bytes(), surrogate::sidecar::MAGIC_V8);
    }

    #[test]
    fn downgrade_to_no_manifest_warns_generically() {
        let root = Root::new("generic");
        let w = downgrade_warnings(&v("0.1.0"), None, &current_formats(), &root.0);
        assert_eq!(w, ["v0.1.0 records no data formats: collections, models or profiles saved by this version may not open in it"]);
    }

    #[test]
    fn downgrade_names_newer_collections() {
        let root = Root::new("newer-coll");
        let mut current = current_formats();
        current.insert(COLLECTION.into(), 2);
        let target = current_formats(); // collection 1
        collection(&root.0, "h-new", &[2, 2]);
        collection(&root.0, "h-old", &[1, 1, 1]);
        collection(&root.0, "h-appended", &[1, 1, 2]); // an older collection a newer build appended to
        std::fs::write(root.0.join("models").join("collections").join("README.txt"), "not a folder").unwrap();
        let w = downgrade_warnings(&v("0.1.0"), Some(&target), &current, &root.0);
        let a = Path::new("models").join("collections").join("h-appended").display().to_string();
        let n = Path::new("models").join("collections").join("h-new").display().to_string();
        assert_eq!(
            w,
            [format!("v0.1.0 can't open 2 of your collections (they were written in a newer format): {a}, {n}. They stay untouched; that version just won't load them.")]
        );
        // one: singular
        let root1 = Root::new("newer-coll-1");
        collection(&root1.0, "h-new", &[2]);
        let w1 = downgrade_warnings(&v("0.1.0"), Some(&target), &current, &root1.0);
        assert_eq!(w1.len(), 1);
        assert!(w1[0].contains("can't open 1 of your collections (it was written in a newer format)") && w1[0].contains("h-new") && w1[0].contains("It stays untouched"), "{}", w1[0]);
        // the files are untouched
        assert_eq!(std::fs::read(root1.0.join("models/collections/h-new/samples.bin")).unwrap(), samples_bin(&[2]));
    }

    #[test]
    fn downgrade_names_newer_models_history_and_training() {
        let root = Root::new("newer-files");
        let mut current = current_formats();
        current.insert(HISTORY.into(), 2);
        current.insert(CHECKPOINT.into(), 2);
        let mut target = current_formats();
        target.insert(SIDECAR.into(), 7);
        let models = root.0.join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("b.model"), b"ASMODEL8rest").unwrap();
        std::fs::write(models.join("a.model"), b"ASMODEL7rest").unwrap();
        collection(&root.0, "h-hist", &[1]);
        std::fs::write(models.join("collections").join("h-hist").join("history.bin"), b"ASAUDIT2").unwrap();
        let training = root.0.join("training");
        std::fs::create_dir_all(&training).unwrap();
        std::fs::write(training.join("pass.ckpt"), "atomic-sandbox pass checkpoint 2\r\ndone 3\n").unwrap();
        let w = downgrade_warnings(&v("0.2.0"), Some(&target), &current, &root.0);
        assert_eq!(w.len(), 3, "{w:#?}");
        assert!(w[0].contains("1 of your collections") && w[0].contains("h-hist"), "{}", w[0]);
        assert!(w[1].contains("1 of your saved models") && w[1].contains("b.model") && !w[1].contains("a.model"), "{}", w[1]);
        assert!(w[2].contains("interrupted training pass") && w[2].contains("pass.ckpt was"), "{}", w[2]);
    }

    #[test]
    fn downgrade_across_a_scene_or_profile_bump_warns_by_type() {
        let root = Root::new("by-type");
        let mut current = current_formats();
        current.insert(SCENE.into(), 2);
        current.insert(PROFILE.into(), 3);
        current.insert("lattice".into(), 1); // a format a later build added
        let target = current_formats();
        let w = downgrade_warnings(&v("0.2.0"), Some(&target), &current, &root.0);
        let this = env!("CARGO_PKG_VERSION");
        assert_eq!(
            w,
            [
                format!("scenes saved by this version (v{this}) may not open in v0.2.0"),
                format!("profiles saved by this version (v{this}) may not open in v0.2.0"),
                format!("lattice data saved by this version (v{this}) may not open in v0.2.0"),
            ]
        );
    }

    #[test]
    fn same_formats_warn_nothing() {
        let root = Root::new("same");
        collection(&root.0, "h-old", &[1, 1]);
        let f = current_formats();
        assert!(downgrade_warnings(&v("0.1.5"), Some(&f), &f, &root.0).is_empty());
        // an upgrade (the target reads newer formats) warns nothing either
        let mut later = f.clone();
        later.insert(COLLECTION.into(), 5);
        assert!(downgrade_warnings(&v("0.9.0"), Some(&later), &f, &root.0).is_empty());
    }
}
