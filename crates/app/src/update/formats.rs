//! The data formats this build reads and writes, and the downgrade warnings drawn from them
//! (installer spec, "The manifest", "Downgrades" and the builder-check ruling on scenes and
//! profiles).
//!
//! Every number is read from the code's own version constant, never typed here:
//!
//! | key | constant | on disk |
//! |---|---|---|
//! | `collection` | `surrogate::collection::RECORD_VERSION` | each record of `models/collections/<key>/samples.bin` |
//! | `sidecar` | the newest model-file version the reader takes: `surrogate::sidecar::MAGICS.len()` | the magic of `models/*.model`, `training/pass.model` |
//! | `history` | `surrogate::history::MAGIC` (`ASAUDIT1`) | the magic of `models/collections/<key>/history.bin` |
//! | `replay` | `crate::replay::HEADER` | a replay script's first line (wherever the user keeps it) |
//! | `pseudo_cache` | `providers::pseudo::DISK_MAGIC` | `cache/pseudo` (a cache: rebuilt, never lost) |
//! | `checkpoint` | `providers::sampling::CHECKPOINT_MAGIC` | the first line of `training/pass.ckpt` |
//! | `scene` | `crate::scene::SCENE_FORMAT` (code-only) | none: scene files carry no version |
//! | `profile` | `crate::ui::profile::PROFILE_FORMAT` (code-only) | none |
//!
//! Choosing a downgrade never changes a file: [`downgrade_warnings`] only reads the headers of the
//! files under the per-user root to say which of them the older version won't open, and where the
//! older version itself would overwrite newer data once running (a collection's audit history, an
//! interrupted training pass), it says that too.

use std::collections::BTreeMap;
use std::io::Read;
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

/// The version number a magic or header line ends with (`ASAUDIT1` → 1, `… replay v1` → 1):
/// `None` without one, or when it overflows a `u32` (a damaged file is unreadable, not a version).
fn trailing_number_checked(s: &[u8]) -> Option<u32> {
    let start = s.len() - s.iter().rev().take_while(|b| b.is_ascii_digit()).count();
    std::str::from_utf8(&s[start..]).ok().filter(|d| !d.is_empty())?.parse::<u32>().ok()
}

/// [`trailing_number_checked`] of one of the code's own constants. Panics without a number: a
/// constant that lost it is a bug the tests catch.
fn trailing_number(s: &[u8]) -> u32 {
    trailing_number_checked(s).unwrap_or_else(|| panic!("{:?} ends without a version number", String::from_utf8_lossy(s)))
}

/// This build's data formats, read from the version constants (module doc).
pub fn current_formats() -> Formats {
    [
        (COLLECTION, u32::from(surrogate::collection::RECORD_VERSION)),
        // the newest model-file version this build reads: the length of the reader's magic list
        (SIDECAR, u32::try_from(surrogate::sidecar::MAGICS.len()).expect("a handful of versions")),
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
/// then `u32 len · u32 crc · payload` records whose payload starts `kind: u8, version: u8`). Each
/// record is held to the reader's own tests — the length within
/// `surrogate::collection::MAX_RECORD` and the file, the CRC-32 matching — and the walk stops at
/// the first that fails, keeping the best version seen before it: a damaged tail (which the app
/// cuts on open) never reports a version. `None` for a file that is not a collection or holds no
/// whole record. One sequential pass through a reused buffer.
fn collection_version(samples: &Path) -> Option<u32> {
    let mut f = std::io::BufReader::new(std::fs::File::open(samples).ok()?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != surrogate::collection::MAGIC {
        return None;
    }
    let (mut best, mut payload) = (None::<u32>, Vec::new());
    loop {
        let mut head = [0u8; 8];
        if f.read_exact(&mut head).is_err() {
            return best;
        }
        let len = u32::from_le_bytes(head[..4].try_into().expect("4 bytes")) as usize;
        let crc = u32::from_le_bytes(head[4..].try_into().expect("4 bytes"));
        if !(2..=surrogate::collection::MAX_RECORD).contains(&len) {
            return best;
        }
        payload.resize(len, 0);
        if f.read_exact(&mut payload).is_err() || surrogate::collection::crc32(&payload) != crc {
            return best;
        }
        let version = u32::from(payload[1]);
        best = Some(best.map_or(version, |b| b.max(version)));
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
    let rest = first.strip_prefix(prefix.as_bytes())?;
    // digits only, and within a u32 (an overflowing number is a damaged file: no warning from it)
    (!rest.is_empty() && rest.iter().all(u8::is_ascii_digit)).then(|| trailing_number_checked(rest)).flatten()
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
///     headers carry a newer version are named, by folder or file. A collection with a newer
///     `samples.bin` won't open (untouched); one whose only newer file is `history.bin` opens with
///     an empty audit history that overwrites it at the next save (its own line says so); a newer
///     pass can't be resumed, and a new pass there replaces it;
///   - scenes, profiles, replays — one line by type (they carry no version, or live anywhere);
///   - the pseudopotential cache — one line when `root/cache/pseudo` exists (it is rebuilt).
///
/// Nothing is changed, moved or converted here.
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

    // collections: a newer `samples.bin` refuses the open (`Collection::open`, untouched); one whose
    // only newer file is `history.bin` opens, starts an empty audit history and overwrites
    // `history.bin` at its next save — said on its own line, as the loss it is
    let models = root.join("models");
    if lower(COLLECTION) || lower(HISTORY) {
        let (mut refused, mut history_lost) = (Vec::new(), Vec::new());
        for d in entries(&models.join("collections")).into_iter().filter(|d| d.is_dir()) {
            let shown = Path::new("models").join("collections").join(name(&d)).display().to_string();
            if newer_than(collection_version(&d.join("samples.bin")), COLLECTION) {
                refused.push(shown);
            } else if newer_than(magic_version(&d.join("history.bin"), b"ASAUDIT"), HISTORY) {
                history_lost.push(shown);
            }
        }
        if !refused.is_empty() {
            out.push(files_line(&tv, "collections", &refused, "open"));
        }
        if !history_lost.is_empty() {
            let n = history_lost.len();
            let (their, its) = if n == 1 { ("its", "it was") } else { ("their", "they were") };
            out.push(format!(
                "{tv} would open {n} of your collections but not {their} audit history ({its} written in a newer format): {}. It starts an empty audit history and overwrites history.bin on its next save there, so the audit history in {} would be lost.",
                history_lost.join(", "),
                if n == 1 { "that collection" } else { "those collections" }
            ));
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
    // a newer pass.ckpt is not offered there; a newer pass.model alone is offered and its resume
    // fails; either way a new pass started there replaces both files ("one folder, one pass")
    if !stale.is_empty() {
        out.push(format!(
            "{tv} can't resume your interrupted training pass ({} written in a newer format): it won't offer the resume, or the resume fails. Starting a new training pass in {tv} replaces those files, and the interrupted pass is lost.",
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
            b.extend_from_slice(&record(&payload));
        }
        b
    }

    /// One whole record: length, CRC-32, payload.
    fn record(payload: &[u8]) -> Vec<u8> {
        let mut r = (payload.len() as u32).to_le_bytes().to_vec();
        r.extend_from_slice(&surrogate::collection::crc32(payload).to_le_bytes());
        r.extend_from_slice(payload);
        r
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
        assert_eq!(trailing_number_checked(b"checkpoint 99999999999"), None, "an overflow is unreadable, not a version");
        assert_eq!(trailing_number_checked(b"no number"), None);
        // the magic list is in version order
        for (i, m) in surrogate::sidecar::MAGICS.iter().enumerate() {
            assert_eq!(trailing_number(&m[..]), i as u32 + 1, "{:?}", String::from_utf8_lossy(&m[..]));
        }
    }

    /// The tripwire for the model-file number (review I2): the sidecar reader itself must refuse
    /// the version after the manifest's as "newer than this build", and must not refuse the
    /// manifest's own so. A V9 reader added to `surrogate::sidecar` without `MAGICS` (and so the
    /// manifest) moving on fails here.
    #[test]
    fn the_sidecar_number_is_the_newest_the_reader_takes() {
        let root = Root::new("sidecar-tripwire");
        let n = current_formats()[SIDECAR];
        assert!(n < 9, "a version {} magic no longer fits `ASMODEL<digit>`: this test needs the new layout", n + 1);
        let probe = |ver: u32| {
            let p = root.0.join(format!("probe-v{ver}.model"));
            let mut bytes = format!("ASMODEL{ver}").into_bytes();
            bytes.extend_from_slice(&2u64.to_le_bytes());
            bytes.extend_from_slice(b"{}");
            std::fs::write(&p, bytes).unwrap();
            surrogate::sidecar::read_provenance(&p).expect_err("a header without provenance never reads").to_string()
        };
        let next = probe(n + 1);
        assert!(next.contains("newer than this build"), "the reader takes version {}: {next}", n + 1);
        let this = probe(n);
        assert!(!this.contains("newer than this build"), "the reader refuses its own newest version {n}: {this}");
    }

    /// The collection walk holds each record to the reader's tests (review M4): a bad CRC, a length
    /// over `MAX_RECORD` or past the end stops it with the best version before; a short tail too.
    #[test]
    fn the_collection_walk_stops_at_a_damaged_record() {
        let root = Root::new("walk");
        let p = root.0.join("samples.bin");
        let walk = |bytes: &[u8]| {
            std::fs::write(&p, bytes).unwrap();
            collection_version(&p)
        };
        assert_eq!(walk(&samples_bin(&[1, 1])), Some(1));
        assert_eq!(walk(&samples_bin(&[1, 3, 2])), Some(3));
        assert_eq!(walk(b"ASCOLL1\0"), None, "no record");
        assert_eq!(walk(b"NOTCOLL!rest"), None, "not a collection");
        // a version-9 record with a wrong CRC is damage, not a newer format
        let mut bad_crc = samples_bin(&[1]);
        let mut r = record(&[1, 9, 0, 0]);
        r[4] ^= 0xff;
        bad_crc.extend_from_slice(&r);
        bad_crc.extend_from_slice(&record(&[1, 9, 0, 0])); // never reached
        assert_eq!(walk(&bad_crc), Some(1));
        // a length over the cap, or past the end of the file
        let mut huge = samples_bin(&[1]);
        huge.extend_from_slice(&((surrogate::collection::MAX_RECORD as u32) + 1).to_le_bytes());
        huge.extend_from_slice(&[0; 12]);
        assert_eq!(walk(&huge), Some(1));
        let mut torn = samples_bin(&[1]);
        torn.extend_from_slice(&record(&[1, 9, 0, 0, 0, 0])[..9]);
        assert_eq!(walk(&torn), Some(1));
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
        assert!(w[0].contains("h-hist") && w[0].contains("audit history") && !w[0].contains("can't open"), "{}", w[0]);
        assert!(w[1].contains("1 of your saved models") && w[1].contains("b.model") && !w[1].contains("a.model"), "{}", w[1]);
        let pass = Path::new("training").join("pass.ckpt").display().to_string();
        assert_eq!(
            w[2],
            format!("v0.2.0 can't resume your interrupted training pass ({pass} was written in a newer format): it won't offer the resume, or the resume fails. Starting a new training pass in v0.2.0 replaces those files, and the interrupted pass is lost.")
        );
        // a checkpoint whose number overflows is unreadable: no warning from it
        std::fs::write(training.join("pass.ckpt"), "atomic-sandbox pass checkpoint 99999999999\n").unwrap();
        let w = downgrade_warnings(&v("0.2.0"), Some(&target), &current, &root.0);
        assert!(!w.iter().any(|l| l.contains("training pass")), "{w:#?}");
    }

    /// Review I1: a collection whose only newer file is `history.bin` opens in the older build,
    /// which starts an empty audit history and overwrites `history.bin` at its next save. Its line
    /// says that loss, apart from the collections that won't open at all.
    #[test]
    fn a_newer_audit_history_is_warned_as_overwritten() {
        let root = Root::new("history-lost");
        let mut current = current_formats();
        current.insert(COLLECTION.into(), 2);
        current.insert(HISTORY.into(), 2);
        let target = current_formats(); // collection 1, history 1
        let hist = |slug: &str| std::fs::write(root.0.join("models").join("collections").join(slug).join("history.bin"), b"ASAUDIT2").unwrap();
        collection(&root.0, "h-both", &[2]); // won't open: named as refused only
        hist("h-both");
        collection(&root.0, "h-hist-a", &[1]);
        hist("h-hist-a");
        collection(&root.0, "h-hist-b", &[1]);
        hist("h-hist-b");
        collection(&root.0, "h-fine", &[1]);
        let w = downgrade_warnings(&v("0.1.0"), Some(&target), &current, &root.0);
        let p = |s: &str| Path::new("models").join("collections").join(s).display().to_string();
        assert_eq!(w.len(), 2, "{w:#?}");
        assert!(w[0].starts_with("v0.1.0 can't open 1 of your collections") && w[0].contains(&p("h-both")), "{}", w[0]);
        assert_eq!(
            w[1],
            format!(
                "v0.1.0 would open 2 of your collections but not their audit history (they were written in a newer format): {}, {}. It starts an empty audit history and overwrites history.bin on its next save there, so the audit history in those collections would be lost.",
                p("h-hist-a"),
                p("h-hist-b")
            )
        );
        // one: singular
        let root1 = Root::new("history-lost-1");
        collection(&root1.0, "h-hist", &[1]);
        std::fs::write(root1.0.join("models/collections/h-hist/history.bin"), b"ASAUDIT2").unwrap();
        let w1 = downgrade_warnings(&v("0.1.0"), Some(&target), &current, &root1.0);
        assert_eq!(w1.len(), 1);
        assert!(w1[0].contains("would open 1 of your collections but not its audit history (it was") && w1[0].contains("overwrites history.bin") && w1[0].contains("in that collection would be lost"), "{}", w1[0]);
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
