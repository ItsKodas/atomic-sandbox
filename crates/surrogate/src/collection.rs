//! Collections (M8a Task 3; spec `docs/superpowers/specs/2026-10-20-m8-shared-model-and-sandbox-design.md`
//! §4.2, plan `docs/superpowers/plans/2026-10-21-m8a-shared-model.md` Task 3): one append-only
//! store of teacher answers per solver surface ([`SurfaceKey`]), shared by every scene solved on
//! that surface whatever its box. This module is storage and the admission rules; the provider
//! (`providers::surrogate`, Task 4) fits from it. Atomic units throughout (Hartree, Bohr).
//!
//! # Layout
//!
//! `<root>/<slug>/` with `<root>` = `models/collections` in the app and `<slug>` =
//! [`SurfaceKey::slug`]:
//!
//! | file | content |
//! |---|---|
//! | `key.txt` | the key, human-readable ([`SurfaceKey::to_text`]); checked on every open, so a slug collision is refused, never mixed |
//! | `samples.bin` | the samples: magic [`MAGIC`], then append-only records (below) |
//! | `references.txt` | isolated-atom energies, one `Z E` line each (Hartree, shortest round-trip decimal) |
//! | `history.bin` | the audit history ([`AuditHistory::encode`]) |
//! | `README.txt` | what it knows, for people: species, pairs, counts, scenes ([`Collection::write_readme`]) |
//! | `.lock` | the owning process id while a writer has it open |
//! | `samples.bin.dropped-<n>` | the bytes a damaged tail held, set aside when it was cut (never deleted) |
//!
//! `references.txt` and `history.bin` are rewritten whole through a temporary file renamed into
//! place; `samples.bin` is only ever appended to (or cut back to its last whole record after a
//! crash, below).
//!
//! # `samples.bin` records
//!
//! After the 8-byte magic, each record is `u32 LE payload length · u32 LE CRC-32 (IEEE) of the
//! payload · payload`, written with one `write_all` and flushed and synced per record. A payload
//! starts with `kind: u8, version: u8`:
//!
//! * kind 0, **header** (first record, once): `u16 LE length · the key's text` — must equal this
//!   key's [`SurfaceKey::to_text`].
//! * kind 1, **sample**: `flags: u16 LE` (0 today: room for Task 10's per-atom force mask and
//!   use-energy flag — a set bit adds its fields after the ones below, under the same magic), `tag:
//!   u8` (the sidecar's tag code), `multiplicity_forced: u8`, `grid_n: u32`, `weight: f64`,
//!   `energy: f64`, `n: u32`, `z: n × u8`, `pos: 3n × f64`, `forces: 3n × f64`, `scene: u16 length ·
//!   UTF-8` (all little-endian; f64 bit patterns kept exactly).
//! * kind 2, **status**: `sample index: u64, code: u8` — a later quarantine change of the sample
//!   with that index ([`crate::quarantine`] codes); the latest record per sample wins on load.
//!   A sample enters with [`Q_NONE`] (the admission rules refuse every other code).
//!
//! Every record kind is version 1. A record whose checksum holds but whose kind, version or flags
//! this build does not know was written by a newer build: the open is refused with that reason and
//! the file is left untouched. A record whose checksum does not hold, or a tail too short for its
//! length prefix, is **damage** (a crash mid-append): it and everything after it are dropped
//! ([`OpenReport::dropped_tail`] counts the records as far as their length prefixes can be
//! followed), the dropped bytes are copied to `samples.bin.dropped-<n>`, and a writer cuts the file
//! back to its last whole record so the next append follows it. A collection is never refused for
//! a damaged tail.
//!
//! # Admission (spec §4.2, controller rulings P3/P5 of Task 3)
//!
//! [`Collection::admit`], in this order: a charged scene's sample is refused (collections are
//! neutral until M9); a sample whose quarantine code is [`Q_BOX`] (box-affected: finite-box error
//! must not enter the shared fit or its calibration) or any other non-zero code is refused; a
//! sample of a scene that forced its multiplicity (`Provenance::multiplicity` is `Some`, the
//! code's only encoding of "forced") is refused — an automatic multiplicity is the lowest-energy
//! state the teacher found and may share; a sample of another surface is refused naming the
//! differing field. Box size (`grid_n`) is not a reason: it is recorded per sample.
//! [`Collection::append`] re-checks the shape and the forced flag and refuses an exact duplicate.
//!
//! # The lock
//!
//! `.lock` holds the process id of the writer. An open finds it absent (creates it, atomically),
//! held by a live process (this one included — a second handle in the same process — the open is
//! read-only: [`Collection::read_only`], [`OpenReport::locked_by_other`]), or held by a process
//! that no longer runs (a stale lock: taken over, with a note). Liveness is asked of the OS
//! directly (`OpenProcess` + `WaitForSingleObject` on Windows, `kill(pid, 0)` on Unix; no crate).
//! A pid the OS has since reused for another live process keeps the collection read-only until
//! that process exits — the safe side. The writer removes the lock when the collection is dropped.

use crate::fit::{Sample, SampleTag};
use crate::history::AuditHistory;
use crate::model::{fnv1a64, Provenance, SurfaceKey};
use crate::quarantine::{Q_BOX, Q_DRAFT, Q_NONE};
use crate::sidecar;
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// The magic of `samples.bin` (module doc).
pub const MAGIC: &[u8; 8] = b"ASCOLL1\0";

/// The index of an [`ImportReport::refused`] entry that refuses the whole file (its charge, a
/// forced multiplicity, another surface, or no samples at all) rather than one sample.
pub const WHOLE_FILE: usize = usize::MAX;

const REC_HEADER: u8 = 0;
const REC_SAMPLE: u8 = 1;
const REC_STATUS: u8 = 2;
/// The version every record kind is written at (module doc).
const RECORD_VERSION: u8 = 1;
/// The sample flags this build reads (none yet; module doc).
const SAMPLE_FLAGS_KNOWN: u16 = 0;
/// The largest record accepted (a hostile or damaged length prefix is not followed past this):
/// 64 MiB ≈ a 1.3-million-atom sample.
const MAX_RECORD: usize = 1 << 26;

/// One stored teacher answer and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionSample {
    pub sample: Sample,
    /// The box it was solved in (grid points per side): recorded, not a gate (spec §4.2).
    pub grid_n: usize,
    /// The scene forced its multiplicity (`Provenance::multiplicity.is_some()`, ruling P3). Never
    /// true in a collection today — admission refuses such samples — but kept per record so a
    /// later physics ruling can admit them without a new format.
    pub multiplicity_forced: bool,
    /// The scene's name, for the knowledge map and the README.
    pub scene: String,
}

/// An open collection (module doc).
#[derive(Debug)]
pub struct Collection {
    pub key: SurfaceKey,
    /// Every stored sample, in the order appended.
    pub samples: Vec<CollectionSample>,
    /// Per sample, its current quarantine code ([`crate::quarantine`]; the latest status record).
    pub quarantine: Vec<u8>,
    /// Isolated-atom energies by Z, Hartree.
    pub references: BTreeMap<u8, f64>,
    /// The audit history ([`save_history`](Self::save_history) writes it).
    pub history: AuditHistory,
    /// Another live handle holds the lock: everything reads, nothing writes.
    pub read_only: bool,
    folder: PathBuf,
    file: Option<File>,
    _lock: Option<Lock>,
    locked_by: Option<u32>,
    seen: HashSet<u64>,
}

/// What [`Collection::open`] found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenReport {
    /// Samples loaded.
    pub samples: usize,
    /// Records dropped from a damaged tail (module doc).
    pub dropped_tail: usize,
    /// The lock is held by another live handle; the collection opened read-only.
    pub locked_by_other: bool,
    /// Anything else worth saying (a stale lock taken over, a damaged tail set aside, an
    /// unreadable history or reference line ignored).
    pub notes: Vec<String>,
}

/// What [`Collection::import_sidecar`] did. Nothing is imported silently: every sample of the
/// file is either counted in `imported` or listed in `refused` with its reason.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportReport {
    pub imported: usize,
    /// (sample index in the file, or [`WHOLE_FILE`], the reason).
    pub refused: Vec<(usize, String)>,
    /// References taken or kept, and the like.
    pub notes: Vec<String>,
}

// ---- checksum ----

const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-32 (IEEE 802.3, reflected, the zip/PNG one) of `bytes`.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

// ---- the lock ----

/// The owner's lock file; removed on drop if it still names this process.
#[derive(Debug)]
struct Lock {
    path: PathBuf,
}

impl Drop for Lock {
    fn drop(&mut self) {
        if std::fs::read_to_string(&self.path).is_ok_and(|t| t.trim() == std::process::id().to_string()) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

enum LockState {
    Owned(Lock, Option<String>),
    HeldBy(u32),
}

fn acquire(path: &Path) -> io::Result<LockState> {
    let me = std::process::id();
    let mut note = None;
    for _ in 0..8 {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut f) => {
                f.write_all(format!("{me}\n").as_bytes())?;
                f.flush()?;
                return Ok(LockState::Owned(Lock { path: path.to_path_buf() }, note));
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                let mut text = match std::fs::read_to_string(path) {
                    Ok(t) => t,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(e),
                };
                if text.trim().is_empty() {
                    // another opener between create and write: give it a moment
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    text = std::fs::read_to_string(path).unwrap_or_default();
                }
                match text.trim().parse::<u32>() {
                    Ok(pid) if pid_alive(pid) => return Ok(LockState::HeldBy(pid)),
                    parsed => {
                        note = Some(match parsed {
                            Ok(pid) => format!("a stale lock of process {pid} (no longer running) was taken over"),
                            Err(_) => format!("a stale, unreadable lock ('{}') was taken over", text.trim()),
                        });
                        match std::fs::remove_file(path) {
                            Ok(()) => {}
                            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                            Err(e) => return Err(e),
                        }
                    }
                }
            }
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::other("the lock changed hands repeatedly while opening"))
}

/// Whether process `pid` is running (module doc, "The lock").
#[cfg(windows)]
fn pid_alive(pid: u32) -> bool {
    type Handle = *mut core::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn WaitForSingleObject(h: Handle, ms: u32) -> u32;
        fn CloseHandle(h: Handle) -> i32;
        fn GetLastError() -> u32;
    }
    const SYNCHRONIZE: u32 = 0x0010_0000;
    const WAIT_TIMEOUT: u32 = 0x0000_0102;
    const ERROR_ACCESS_DENIED: u32 = 5;
    if pid == 0 {
        return false;
    }
    // SAFETY: plain Win32 calls on a handle this function opens and closes; no memory is shared.
    unsafe {
        let h = OpenProcess(SYNCHRONIZE, 0, pid);
        if h.is_null() {
            // no such process (ERROR_INVALID_PARAMETER), or one we may not open — which exists
            return GetLastError() == ERROR_ACCESS_DENIED;
        }
        let r = WaitForSingleObject(h, 0);
        CloseHandle(h);
        r == WAIT_TIMEOUT
    }
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    let Ok(p) = i32::try_from(pid) else { return false };
    if p <= 0 {
        return false;
    }
    // SAFETY: signal 0 only checks for the process; nothing is sent.
    let r = unsafe { kill(p, 0) };
    r == 0 || io::Error::last_os_error().raw_os_error() == Some(1 /* EPERM: exists, not ours */)
}

#[cfg(not(any(windows, unix)))]
fn pid_alive(_pid: u32) -> bool {
    true // unknown platform: treat every lock as live (read-only, the safe side)
}

// ---- encoding ----

fn put_f64(out: &mut Vec<u8>, x: f64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let mut rec = Vec::with_capacity(8 + payload.len());
    rec.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    rec.extend_from_slice(&crc32(payload).to_le_bytes());
    rec.extend_from_slice(payload);
    rec
}

fn header_record(key: &SurfaceKey) -> Vec<u8> {
    let text = key.to_text();
    let mut p = vec![REC_HEADER, RECORD_VERSION];
    p.extend_from_slice(&(text.len() as u16).to_le_bytes());
    p.extend_from_slice(text.as_bytes());
    frame(&p)
}

fn sample_record(s: &CollectionSample) -> Vec<u8> {
    let x = &s.sample;
    let n = x.z.len();
    let mut p = Vec::with_capacity(40 + n * 49 + s.scene.len());
    p.extend_from_slice(&[REC_SAMPLE, RECORD_VERSION]);
    p.extend_from_slice(&0u16.to_le_bytes());
    p.push(sidecar::tag_index(x.tag));
    p.push(s.multiplicity_forced as u8);
    p.extend_from_slice(&(s.grid_n as u32).to_le_bytes());
    put_f64(&mut p, x.weight);
    put_f64(&mut p, x.energy);
    p.extend_from_slice(&(n as u32).to_le_bytes());
    p.extend_from_slice(&x.z);
    for v in x.pos.iter().chain(&x.forces) {
        for &c in v {
            put_f64(&mut p, c);
        }
    }
    p.extend_from_slice(&(s.scene.len() as u16).to_le_bytes());
    p.extend_from_slice(s.scene.as_bytes());
    frame(&p)
}

fn status_record(index: usize, code: u8) -> Vec<u8> {
    let mut p = vec![REC_STATUS, RECORD_VERSION];
    p.extend_from_slice(&(index as u64).to_le_bytes());
    p.push(code);
    frame(&p)
}

/// A cursor over one payload; every read names what ran short.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| format!("the record ends inside {what}"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self, what: &str) -> Result<u8, String> {
        Ok(self.take(1, what)?[0])
    }
    fn u16(&mut self, what: &str) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().expect("2 bytes")))
    }
    fn u32(&mut self, what: &str) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self, what: &str) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }
    fn f64(&mut self, what: &str) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }
    fn triples(&mut self, n: usize, what: &str) -> Result<Vec<[f64; 3]>, String> {
        (0..n).map(|_| Ok([self.f64(what)?, self.f64(what)?, self.f64(what)?])).collect()
    }
    fn done(&self) -> Result<(), String> {
        if self.at == self.b.len() {
            Ok(())
        } else {
            Err(format!("{} bytes after the record's last field", self.b.len() - self.at))
        }
    }
}

fn decode_sample(c: &mut Cursor) -> Result<CollectionSample, String> {
    let flags = c.u16("flags")?;
    if flags & !SAMPLE_FLAGS_KNOWN != 0 {
        return Err(format!("newer build: sample flags {flags:#06x}"));
    }
    let tag_code = c.u8("tag")?;
    let tag = sidecar::tag_from_index(tag_code).ok_or_else(|| format!("newer build: tag code {tag_code}"))?;
    let forced = match c.u8("multiplicity flag")? {
        0 => false,
        1 => true,
        k => return Err(format!("multiplicity flag {k} is neither 0 nor 1")),
    };
    let grid_n = c.u32("grid_n")? as usize;
    let weight = c.f64("weight")?;
    let energy = c.f64("energy")?;
    let n = c.u32("atom count")? as usize;
    if n.saturating_mul(49) > c.b.len() {
        return Err(format!("{n} atoms do not fit the record"));
    }
    let z = c.take(n, "z")?.to_vec();
    let pos = c.triples(n, "positions")?;
    let forces = c.triples(n, "forces")?;
    let len = c.u16("scene name length")? as usize;
    let scene = String::from_utf8(c.take(len, "scene name")?.to_vec()).map_err(|_| "the scene name is not UTF-8".to_string())?;
    c.done()?;
    Ok(CollectionSample { sample: Sample { pos, z, energy, forces, tag, weight }, grid_n, multiplicity_forced: forced, scene })
}

/// What `samples.bin` held.
struct Parsed {
    samples: Vec<CollectionSample>,
    quarantine: Vec<u8>,
    /// Bytes of whole, valid records (magic included); a writer cuts the file here.
    good_len: usize,
    /// A header record was read (else the file is new or its header was torn).
    has_header: bool,
    dropped: usize,
}

/// Reads `samples.bin` (module doc): `Err` only for a file that is not a collection, of another
/// surface, or written by a newer build.
fn parse(bytes: &[u8], key: &SurfaceKey) -> Result<Parsed, String> {
    let mut out = Parsed { samples: Vec::new(), quarantine: Vec::new(), good_len: 0, has_header: false, dropped: 0 };
    if bytes.len() < MAGIC.len() {
        // nothing, or a magic torn mid-write
        if !MAGIC.starts_with(bytes) {
            return Err("samples.bin is not a collection (bad magic)".into());
        }
        out.dropped = usize::from(!bytes.is_empty());
        return Ok(out);
    }
    if &bytes[..8] != MAGIC {
        return Err(format!("samples.bin is not a collection (magic {:?}, expected ASCOLL1)", String::from_utf8_lossy(&bytes[..8])));
    }
    let mut at = 8usize;
    out.good_len = at;
    let mut index = 0usize;
    while at < bytes.len() {
        let rest = &bytes[at..];
        let whole = rest.len() >= 8 && {
            let len = u32::from_le_bytes(rest[..4].try_into().expect("4 bytes")) as usize;
            len <= MAX_RECORD && 8 + len <= rest.len() && crc32(&rest[8..8 + len]) == u32::from_le_bytes(rest[4..8].try_into().expect("4 bytes"))
        };
        if !whole {
            // damage: count the records the length prefixes still lead through
            let mut k = at;
            while k < bytes.len() {
                out.dropped += 1;
                let r = &bytes[k..];
                if r.len() < 8 {
                    break;
                }
                let len = u32::from_le_bytes(r[..4].try_into().expect("4 bytes")) as usize;
                if len > MAX_RECORD || 8 + len > r.len() {
                    break;
                }
                k += 8 + len;
            }
            return Ok(out);
        }
        let len = u32::from_le_bytes(rest[..4].try_into().expect("4 bytes")) as usize;
        let payload = &rest[8..8 + len];
        let mut c = Cursor { b: payload, at: 0 };
        let what = |e: String| {
            if e.starts_with("newer build") {
                format!("samples.bin record {index} was written by a newer build ({}); this build does not read it", e.trim_start_matches("newer build: "))
            } else {
                format!("samples.bin record {index}: {e}")
            }
        };
        let kind = c.u8("kind").map_err(what)?;
        let version = c.u8("version").map_err(what)?;
        if kind > REC_STATUS || version != RECORD_VERSION {
            return Err(what(format!("newer build: record kind {kind} version {version}")));
        }
        if (kind == REC_HEADER) != (index == 0) {
            return Err(what(if kind == REC_HEADER { "a second header".into() } else { "the first record is not the header".into() }));
        }
        match kind {
            REC_HEADER => {
                let n = c.u16("key length").map_err(what)? as usize;
                let text = std::str::from_utf8(c.take(n, "key").map_err(what)?).map_err(|_| what("the key is not UTF-8".into()))?;
                c.done().map_err(what)?;
                let theirs = SurfaceKey::from_text(text).map_err(|e| what(format!("key: {e}")))?;
                if let Some(d) = key.difference(&theirs) {
                    return Err(format!("samples.bin belongs to another surface: {d}"));
                }
                out.has_header = true;
            }
            REC_SAMPLE => {
                let s = decode_sample(&mut c).map_err(what)?;
                out.samples.push(s);
                out.quarantine.push(Q_NONE);
            }
            _ => {
                let i = c.u64("sample index").map_err(what)?;
                let code = c.u8("code").map_err(what)?;
                c.done().map_err(what)?;
                let slot = usize::try_from(i).ok().and_then(|i| out.quarantine.get_mut(i)).ok_or_else(|| what(format!("a status for sample {i} of {}", out.samples.len())))?;
                *slot = code;
            }
        }
        at += 8 + len;
        out.good_len = at;
        index += 1;
    }
    Ok(out)
}

/// The identity of a sample for duplicate detection: Z and the exact bits of positions, energy
/// and forces.
fn fingerprint(s: &Sample) -> u64 {
    let mut b = Vec::with_capacity(s.z.len() * 49 + 8);
    b.extend_from_slice(&s.z);
    for v in s.pos.iter().chain(&s.forces) {
        for &c in v {
            b.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    b.extend_from_slice(&s.energy.to_bits().to_le_bytes());
    fnv1a64(&b)
}

/// The shape, finiteness and weight of a sample (what a fit needs of it).
fn check_shape(s: &Sample) -> Result<(), String> {
    let n = s.z.len();
    if n == 0 || s.pos.len() != n || s.forces.len() != n {
        return Err(format!("sample has {} positions, {n} charges and {} forces", s.pos.len(), s.forces.len()));
    }
    if s.z.contains(&0) {
        return Err("sample has an atom of Z 0".into());
    }
    if !s.energy.is_finite() || s.pos.iter().chain(&s.forces).flatten().any(|x| !x.is_finite()) {
        return Err("sample has a non-finite energy, position or force".into());
    }
    if !(s.weight > 0.0 && s.weight.is_finite()) {
        return Err(format!("sample weight {} is not positive", s.weight));
    }
    Ok(())
}

/// `bytes` to `path` through `<name>.tmp` renamed into place.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

fn element(z: u8) -> String {
    let s = units::elements::symbol(z);
    if s.is_empty() || s == "?" {
        format!("Z{z}")
    } else {
        s.to_string()
    }
}

impl Collection {
    /// Opens (creating when absent) the collection of `key` under `root` — the folder
    /// `root/<key.slug()>` (module doc, "Layout"). `Err` for an unusable folder, a folder of
    /// another surface (`key.txt` or the samples' header differ), a `samples.bin` that is not a
    /// collection or holds a record of a newer build; never for a damaged tail.
    pub fn open(root: &Path, key: &SurfaceKey) -> Result<(Collection, OpenReport), String> {
        let folder = root.join(key.slug());
        let at = |e: io::Error| format!("{}: {e}", folder.display());
        std::fs::create_dir_all(&folder).map_err(at)?;
        let mut report = OpenReport::default();
        let (lock, locked_by) = match acquire(&folder.join(".lock")).map_err(|e| format!("{}: the lock: {e}", folder.display()))? {
            LockState::Owned(l, note) => {
                report.notes.extend(note);
                (Some(l), None)
            }
            LockState::HeldBy(pid) => {
                report.locked_by_other = true;
                report.notes.push(format!("opened read-only: process {pid} holds the lock"));
                (None, Some(pid))
            }
        };
        let read_only = lock.is_none();

        // key.txt
        let key_path = folder.join("key.txt");
        match std::fs::read_to_string(&key_path) {
            Ok(text) => {
                let theirs = SurfaceKey::from_text(&text).map_err(|e| format!("{}: {e}", key_path.display()))?;
                if let Some(d) = key.difference(&theirs) {
                    return Err(format!("{} holds another surface: {d}", folder.display()));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                if !read_only {
                    write_atomic(&key_path, key.to_text().as_bytes()).map_err(at)?;
                }
            }
            Err(e) => return Err(at(e)),
        }

        // samples.bin
        let path = folder.join("samples.bin");
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(at(e)),
        };
        let parsed = parse(&bytes, key).map_err(|e| format!("{}: {e}", path.display()))?;
        report.dropped_tail = parsed.dropped;
        let mut file = None;
        if parsed.good_len < bytes.len() {
            let n = bytes.len() - parsed.good_len;
            if read_only {
                report.notes.push(format!("{n} bytes ({} records) of a damaged tail were not read (left for the writer)", parsed.dropped));
            } else {
                let mut k = 0;
                let aside = loop {
                    let p = folder.join(format!("samples.bin.dropped-{k}"));
                    if !p.exists() {
                        break p;
                    }
                    k += 1;
                };
                std::fs::write(&aside, &bytes[parsed.good_len..]).map_err(at)?;
                report.notes.push(format!("dropped a damaged tail of {n} bytes ({} records; a crash mid-append), kept aside in {}", parsed.dropped, aside.display()));
            }
        }
        if !read_only {
            let mut f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path).map_err(at)?;
            if !parsed.has_header {
                // a new file, or one whose magic or header was torn: start it afresh
                f.set_len(0).map_err(at)?;
                let mut head = MAGIC.to_vec();
                head.extend_from_slice(&header_record(key));
                f.write_all(&head).map_err(at)?;
                f.sync_data().map_err(at)?;
            } else if parsed.good_len < bytes.len() {
                f.set_len(parsed.good_len as u64).map_err(at)?;
                f.sync_data().map_err(at)?;
            }
            drop(f);
            file = Some(OpenOptions::new().append(true).open(&path).map_err(at)?);
        }

        // references.txt
        let mut references = BTreeMap::new();
        let ref_path = folder.join("references.txt");
        match std::fs::read_to_string(&ref_path) {
            Ok(text) => {
                for (k, line) in text.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    let parsed = line.split_once(' ').and_then(|(z, e)| Some((z.trim().parse::<u8>().ok()?, e.trim().parse::<f64>().ok()?)));
                    match parsed {
                        Some((z, e)) if z > 0 && e.is_finite() => {
                            references.insert(z, e);
                        }
                        _ => report.notes.push(format!("references.txt line {}: '{line}' is not 'Z energy' — ignored (that reference will be solved again)", k + 1)),
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(at(e)),
        }

        // history.bin
        let history = match std::fs::read(folder.join("history.bin")) {
            Ok(b) => AuditHistory::decode(&b).unwrap_or_else(|e| {
                report.notes.push(format!("history.bin unreadable ({e}): starting an empty audit history"));
                AuditHistory::default()
            }),
            Err(e) if e.kind() == io::ErrorKind::NotFound => AuditHistory::default(),
            Err(e) => return Err(at(e)),
        };

        report.samples = parsed.samples.len();
        let seen = parsed.samples.iter().map(|s| fingerprint(&s.sample)).collect();
        let c = Collection { key: key.clone(), samples: parsed.samples, quarantine: parsed.quarantine, references, history, read_only, folder, file, _lock: lock, locked_by, seen };
        Ok((c, report))
    }

    /// The collection's folder (`root/<slug>`).
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// Would this sample — produced at `prov`, with the provider's quarantine code `quarantine` —
    /// be admitted (module doc, "Admission")? The caller passes [`Q_BOX`] for a teacher answer
    /// whose face density is above the box rule even when the provider kept it in its own fit
    /// (the per-scene store may need it; the shared one never does). `Ok(())` or the reason.
    pub fn admit(&self, s: &Sample, prov: &Provenance, quarantine: u8) -> Result<(), String> {
        if prov.charge != 0 {
            return Err(format!("charge {}: collections are neutral until M9", prov.charge));
        }
        if quarantine == Q_BOX {
            return Err("box-affected sample".into());
        }
        if quarantine != Q_NONE {
            return Err(format!("quarantined sample (code {quarantine})"));
        }
        if let Some(m) = prov.multiplicity {
            return Err(format!("the scene forced multiplicity {m}"));
        }
        if let Some(d) = self.key.difference(&prov.surface_key()) {
            return Err(format!("another solver surface: {d}"));
        }
        check_shape(s)
    }

    fn writable(&self) -> Result<(), String> {
        if self.read_only {
            return Err(format!("the collection is read-only: process {} holds its lock", self.locked_by.map_or("?".into(), |p| p.to_string())));
        }
        Ok(())
    }

    fn write_record(&mut self, rec: &[u8]) -> Result<(), String> {
        let f = self.file.as_mut().ok_or("the collection has no open sample file")?;
        f.write_all(rec).and_then(|_| f.flush()).and_then(|_| f.sync_data()).map_err(|e| format!("{}: {e}", self.folder.join("samples.bin").display()))
    }

    /// Appends a sample (flushed and synced before it returns; module doc). The caller has
    /// [`admit`](Self::admit)ted it; this re-checks the shape and the forced flag and refuses an
    /// exact duplicate of a stored sample.
    pub fn append(&mut self, s: CollectionSample) -> Result<(), String> {
        self.writable()?;
        check_shape(&s.sample)?;
        if s.multiplicity_forced {
            return Err("a sample of a scene that forced its multiplicity is never shared".into());
        }
        if s.scene.len() > u16::MAX as usize || s.grid_n > u32::MAX as usize || s.sample.z.len() > u32::MAX as usize {
            return Err("sample too large for a record".into());
        }
        let fp = fingerprint(&s.sample);
        if self.seen.contains(&fp) && self.samples.iter().any(|t| t.sample == s.sample) {
            return Err("an identical sample is already in the collection".into());
        }
        let rec = sample_record(&s);
        if rec.len() - 8 > MAX_RECORD {
            return Err("sample too large for a record".into());
        }
        self.write_record(&rec)?;
        self.seen.insert(fp);
        self.samples.push(s);
        self.quarantine.push(Q_NONE);
        Ok(())
    }

    /// Records a later quarantine change of sample `index` (a status record; module doc).
    pub fn set_status(&mut self, index: usize, code: u8) -> Result<(), String> {
        self.writable()?;
        if index >= self.samples.len() {
            return Err(format!("no sample {index} (the collection holds {})", self.samples.len()));
        }
        self.write_record(&status_record(index, code))?;
        self.quarantine[index] = code;
        Ok(())
    }

    /// Sets the isolated-atom energy of `z` (Hartree) and rewrites `references.txt`.
    pub fn set_reference(&mut self, z: u8, energy: f64) -> Result<(), String> {
        self.writable()?;
        if z == 0 || !energy.is_finite() {
            return Err(format!("reference Z {z}, E {energy} is not a reference"));
        }
        let mut next = self.references.clone();
        next.insert(z, energy);
        let mut text = String::from("# Atomic Sandbox collection: isolated-atom reference energies, one 'Z E' per line (Hartree)\n");
        for (z, e) in &next {
            text.push_str(&format!("{z} {e:?}\n"));
        }
        let path = self.folder.join("references.txt");
        write_atomic(&path, text.as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
        self.references = next;
        Ok(())
    }

    /// Writes [`history`](Self::history) to `history.bin`.
    pub fn save_history(&self) -> Result<(), String> {
        self.writable()?;
        let path = self.folder.join("history.bin");
        write_atomic(&path, &self.history.encode()).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Imports the samples of a `.model` sidecar (any version: fitted or samples-only) — never
    /// silently: whole-file refusals (a charged scene, a forced multiplicity, another surface, no
    /// samples) and per-sample ones (quarantined, box-affected — its code, or its recorded face
    /// density above `face_limit`, the box rule at this surface's h, `providers::box_affected_face_density`
    /// — a draft audit, a duplicate) are all in the report. A reference the collection lacks is
    /// taken from the file when the file says it was made under the reference rule
    /// ([`sidecar::REFERENCE_RULE`]); one it has is kept (a note when the file's differs). The
    /// file's audit history is not imported (it judged that file's model, not this collection's).
    /// `Err` only for an unreadable file or a read-only collection.
    pub fn import_sidecar(&mut self, path: &Path, face_limit: f64) -> Result<ImportReport, String> {
        self.writable()?;
        let contents = sidecar::read_any(path).map_err(|e| e.to_string())?;
        let (prov, samples, quarantine, face, references, rule) = match contents {
            sidecar::Contents::Fitted(s) => {
                let s = *s;
                let refs: Vec<(u8, f64)> = s.model.basis.species.iter().copied().zip(s.model.ref_energy.iter().copied()).collect();
                (s.model.provenance, s.samples, s.quarantine, s.face_density, refs, s.reference_rule)
            }
            sidecar::Contents::Samples(s) => (s.provenance, s.samples, s.quarantine, s.face_density, s.references, s.reference_rule),
        };
        let mut report = ImportReport::default();
        // the file-level rules first, in admit's order, so a whole file is refused once
        let whole = if prov.charge != 0 {
            Some(format!("charge {}: collections are neutral until M9", prov.charge))
        } else if let Some(m) = prov.multiplicity {
            Some(format!("the scene forced multiplicity {m}"))
        } else if let Some(d) = self.key.difference(&prov.surface_key()) {
            Some(format!("another solver surface: {d}"))
        } else if samples.is_empty() {
            Some("the file holds no training samples (a model-only file)".into())
        } else {
            None
        };
        if let Some(why) = whole {
            report.refused.push((WHOLE_FILE, why));
            return Ok(report);
        }
        let scene = path.file_stem().map_or_else(|| "imported".to_string(), |s| s.to_string_lossy().into_owned());
        for (i, s) in samples.into_iter().enumerate() {
            let code = if s.tag == SampleTag::DraftAudit { Q_DRAFT } else { quarantine.get(i).copied().unwrap_or(Q_NONE) };
            let rho = face.get(i).copied().unwrap_or(f64::NAN);
            let verdict = if code == Q_NONE && rho.is_finite() && rho > face_limit {
                Err(format!("box-affected sample (face density {rho:.2e} e/Bohr³ above the box rule's {face_limit:.2e})"))
            } else {
                self.admit(&s, &prov, code)
            };
            if let Err(why) = verdict {
                report.refused.push((i, why));
                continue;
            }
            match self.append(CollectionSample { sample: s, grid_n: prov.grid_n, multiplicity_forced: false, scene: scene.clone() }) {
                Ok(()) => report.imported += 1,
                Err(why) => report.refused.push((i, why)),
            }
        }
        for (z, e) in references {
            let name = element(z);
            match self.references.get(&z) {
                Some(&have) if have.to_bits() == e.to_bits() => {}
                Some(&have) => report.notes.push(format!("the file's {name} reference {e:?} Ha differs from the collection's {have:?} Ha by {:.1e}; the collection's is kept", e - have)),
                None if rule.as_deref() == Some(sidecar::REFERENCE_RULE) => {
                    self.set_reference(z, e)?;
                    report.notes.push(format!("{name} reference {e:?} Ha taken from the file"));
                }
                None => report.notes.push(format!("the file's {name} reference was not made under the reference rule ('{}'); not taken", sidecar::REFERENCE_RULE)),
            }
        }
        Ok(report)
    }

    /// Writes `README.txt`: the key, the sample counts (by status, by origin), the species (atoms,
    /// samples, references), the species pairs seen within the default basis's r_cut, and the scenes.
    pub fn write_readme(&self) -> Result<(), String> {
        self.writable()?;
        let path = self.folder.join("README.txt");
        write_atomic(&path, self.readme().as_bytes()).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn readme(&self) -> String {
        let r_cut = crate::descriptor::Basis::new(&[1]).r_cut;
        let mut t = String::new();
        t.push_str("Atomic Sandbox — shared model collection\n\n");
        t.push_str(&format!("surface: {}\nfolder:  {}\n\n", self.key.describe(), self.key.slug()));
        let count = |code: u8| self.quarantine.iter().filter(|&&q| q == code).count();
        let n = self.samples.len();
        t.push_str(&format!(
            "{n} samples: {} trained on, {} held, {} superseded, {} box-affected, {} draft\n",
            count(Q_NONE),
            count(crate::quarantine::Q_HELD),
            count(crate::quarantine::Q_SUPERSEDED),
            count(Q_BOX),
            count(Q_DRAFT)
        ));
        let mut tags: BTreeMap<String, usize> = BTreeMap::new();
        for s in &self.samples {
            *tags.entry(format!("{:?}", s.sample.tag)).or_default() += 1;
        }
        t.push_str(&format!("by origin: {}\n\n", tags.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ")));
        let mut species: BTreeMap<u8, (usize, usize)> = BTreeMap::new();
        let mut pairs: BTreeMap<(u8, u8), (usize, usize, f64)> = BTreeMap::new();
        for s in &self.samples {
            let x = &s.sample;
            let mut zs: Vec<u8> = x.z.clone();
            zs.sort_unstable();
            zs.dedup();
            for &z in &x.z {
                species.entry(z).or_default().0 += 1;
            }
            for z in zs {
                species.entry(z).or_default().1 += 1;
            }
            let mut here: BTreeMap<(u8, u8), usize> = BTreeMap::new();
            let mut nearest: BTreeMap<(u8, u8), f64> = BTreeMap::new();
            for i in 0..x.z.len() {
                for j in i + 1..x.z.len() {
                    let d = (0..3).map(|a| (x.pos[i][a] - x.pos[j][a]).powi(2)).sum::<f64>().sqrt();
                    if d < r_cut {
                        let p = (x.z[i].min(x.z[j]), x.z[i].max(x.z[j]));
                        *here.entry(p).or_default() += 1;
                        let m = nearest.entry(p).or_insert(f64::INFINITY);
                        *m = m.min(d);
                    }
                }
            }
            for (p, k) in here {
                let e = pairs.entry(p).or_insert((0, 0, f64::INFINITY));
                e.0 += 1;
                e.1 += k;
                e.2 = e.2.min(nearest[&p]);
            }
        }
        t.push_str("species:\n");
        for (z, (atoms, samples)) in &species {
            let r = self.references.get(z).map_or("no reference yet".to_string(), |e| format!("reference {e:.6} Ha"));
            t.push_str(&format!("  {} (Z {z}): {atoms} atoms in {samples} samples, {r}\n", element(*z)));
        }
        t.push_str(&format!("\npairs within {r_cut} Bohr (the default basis's cutoff):\n"));
        for ((a, b), (samples, atom_pairs, nearest)) in &pairs {
            t.push_str(&format!("  {}–{}: {samples} samples, {atom_pairs} atom pairs, nearest {nearest:.3} Bohr\n", element(*a), element(*b)));
        }
        let mut scenes: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
        for s in &self.samples {
            let e = scenes.entry(s.scene.as_str()).or_insert((0, usize::MAX, 0));
            e.0 += 1;
            e.1 = e.1.min(s.grid_n);
            e.2 = e.2.max(s.grid_n);
        }
        t.push_str("\nscenes:\n");
        for (name, (k, lo, hi)) in &scenes {
            let grids = if lo == hi { format!("{lo}³") } else { format!("{lo}³ … {hi}³") };
            t.push_str(&format!("  {name}: {k} samples, grid {grids}\n"));
        }
        t.push_str(&format!("\naudit history: {} points ({} recorded)\n", self.history.len(), self.history.recorded()));
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::SampleTag;
    use crate::history::{AuditPoint, PointKind};
    use crate::quarantine::{Q_BOX, Q_HELD, Q_NONE, Q_SUPERSEDED};
    use crate::sidecar::{self, SampleSet, REFERENCE_RULE};
    use std::path::PathBuf;

    /// A fresh, empty directory of its own for test `name`.
    fn root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ascoll-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn prov() -> Provenance {
        Provenance { grid_n: 49, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "Kohn-Sham (GPU)".into(), masses: vec![(1, 1)], ionic: "none".into() }
    }

    /// An H₂ geometry at 1.4 + 0.01·i Bohr with awkward values (−0.0, a 17-digit energy).
    fn h2(i: usize) -> Sample {
        let d = 1.4 + 0.01 * i as f64;
        Sample {
            pos: vec![[0.1, -0.0, 0.0], [0.1 + d, 0.0, 1e-17]],
            z: vec![1, 1],
            energy: -1.123_456_789_012_345_6 - 1e-3 * i as f64,
            forces: vec![[0.01 * i as f64, -0.0, 1e-300], [-0.01 * i as f64, 0.0, -1e-300]],
            tag: [SampleTag::LiveStep, SampleTag::Scan, SampleTag::Audit][i % 3],
            weight: if i.is_multiple_of(4) { 2.5 } else { 1.0 },
        }
    }

    fn cs(i: usize) -> CollectionSample {
        CollectionSample { sample: h2(i), grid_n: 49 + 8 * (i % 2), multiplicity_forced: false, scene: format!("h2-{}", i % 2) }
    }

    fn samples_path(dir: &Path, key: &SurfaceKey) -> PathBuf {
        dir.join(key.slug()).join("samples.bin")
    }

    #[test]
    fn surface_key_ignores_grid_n_charge_and_multiplicity() {
        let a = prov();
        let mut b = prov();
        b.grid_n = 97;
        b.charge = 1;
        b.multiplicity = Some(3);
        b.masses = vec![(1, 2)];
        assert_ne!(a, b, "the per-scene provenance still tells them apart");
        assert_eq!(a.surface_key(), b.surface_key());
        assert_eq!(a.surface_key().functional, "lsda");
        type Change = Box<dyn Fn(&mut Provenance)>;
        let changes: Vec<(&str, Change, &str)> = vec![
            ("h", Box::new(|p| p.h = 0.15), "h 0.15"),
            ("sigma", Box::new(|p| p.sigma = 0.2), "σ 0.2"),
            ("tol", Box::new(|p| p.tol = 1e-5), "tol 1e-5"),
            ("teacher", Box::new(|p| p.teacher = "Kohn-Sham (CPU)".into()), "teacher 'Kohn-Sham (CPU)'"),
            ("ionic", Box::new(|p| p.ionic = "all/0123".into()), "ionic all/0123"),
        ];
        for (what, change, field) in changes {
            let mut c = prov();
            change(&mut c);
            assert_ne!(a.surface_key(), c.surface_key(), "{what}");
            let d = a.surface_key().difference(&c.surface_key()).expect(what);
            assert!(d.starts_with(field), "{what}: {d}");
            assert_ne!(a.surface_key().slug(), c.surface_key().slug(), "{what}: the slug tells them apart");
        }
        assert_eq!(a.surface_key().difference(&b.surface_key()), None);
    }

    /// The slug is pinned (stable across runs and platforms) and filesystem-safe; a long ionic tag
    /// is hashed; the key reads back bit-exactly from its text.
    #[test]
    fn the_collection_slug_is_stable_and_safe() {
        let k = prov().surface_key();
        assert_eq!(k.slug(), "lsda_h0.25_s0.25_tol1e-6_kohn-sham-gpu_none");
        let mut p = prov();
        p.h = 0.15;
        p.sigma = 0.15;
        p.ionic = "all/0123456789abcdef/shared0.5/0.25/stop2x5e-8".into();
        let k2 = p.surface_key();
        let s = k2.slug();
        let head = "lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-";
        assert!(s.starts_with(head), "{s}");
        assert_eq!(s.len(), head.len() + 12, "{s}");
        assert!(s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._+-".contains(c)), "{s}");
        assert_eq!(s, k2.slug(), "deterministic");
        for key in [k, k2] {
            assert_eq!(SurfaceKey::from_text(&key.to_text()).unwrap(), key);
        }
    }

    #[test]
    fn admit_refuses_charged_box_and_forced_spin_samples() {
        let dir = root("admit");
        let (mut c, _) = Collection::open(&dir, &prov().surface_key()).unwrap();
        let s = h2(1);
        assert_eq!(c.admit(&s, &prov(), Q_NONE), Ok(()));
        let mut big_box = prov();
        big_box.grid_n = 81;
        assert_eq!(c.admit(&s, &big_box, Q_NONE), Ok(()), "another box is the same surface");
        let mut charged = prov();
        charged.charge = 1;
        assert_eq!(c.admit(&s, &charged, Q_NONE), Err("charge 1: collections are neutral until M9".to_string()));
        assert_eq!(c.admit(&s, &prov(), Q_BOX), Err("box-affected sample".to_string()));
        assert_eq!(c.admit(&s, &prov(), Q_HELD), Err("quarantined sample (code 1)".to_string()));
        assert_eq!(c.admit(&s, &prov(), Q_SUPERSEDED), Err("quarantined sample (code 2)".to_string()));
        let mut forced = prov();
        forced.multiplicity = Some(3);
        assert_eq!(c.admit(&s, &forced, Q_NONE), Err("the scene forced multiplicity 3".to_string()));
        let mut other = prov();
        other.h = 0.15;
        let e = c.admit(&s, &other, Q_NONE).unwrap_err();
        assert!(e.contains("h 0.15 (the collection's 0.25)"), "{e}");
        let mut bad = h2(1);
        bad.forces.pop();
        assert!(c.admit(&bad, &prov(), Q_NONE).is_err(), "shape");
        // append refuses a forced-spin sample too (defence in depth)
        let mut f = cs(1);
        f.multiplicity_forced = true;
        let e = c.append(f).unwrap_err();
        assert!(e.contains("forced"), "{e}");
        assert!(c.samples.is_empty());
    }

    /// Appends, status updates, references and the history survive a reopen bit-exactly.
    #[test]
    fn a_collection_round_trips_samples_statuses_references_and_history() {
        let dir = root("roundtrip");
        let key = prov().surface_key();
        {
            let (mut c, r) = Collection::open(&dir, &key).unwrap();
            assert_eq!((r.samples, r.dropped_tail, r.locked_by_other, c.read_only), (0, 0, false, false));
            for i in 0..5 {
                c.append(cs(i)).unwrap();
            }
            c.set_status(3, Q_HELD).unwrap();
            c.set_status(1, Q_SUPERSEDED).unwrap();
            c.set_status(3, Q_NONE).unwrap();
            assert!(c.set_status(5, Q_HELD).is_err(), "no sample 5");
            c.set_reference(1, -0.478_123_456_789_012_3).unwrap();
            c.history.push(AuditPoint { kind: PointKind::Audit, classes: vec![1, 1], pred_force_std: vec![1e-3; 2], pred_force_std_epistemic: vec![5e-4; 2], pred_energy_std: 1e-4, err_force: vec![2e-4; 2], teacher_force: vec![0.01; 2], err_energy: -3e-5, n_atoms: 2, model_gen: 7 });
            c.save_history().unwrap();
            let e = c.append(cs(2)).unwrap_err();
            assert!(e.contains("already"), "an identical sample is refused: {e}");
        }
        let (c, r) = Collection::open(&dir, &key).unwrap();
        assert_eq!((r.samples, r.dropped_tail), (5, 0));
        assert_eq!(c.samples, (0..5).map(cs).collect::<Vec<_>>());
        for (a, b) in c.samples.iter().zip((0..5).map(cs)) {
            assert_eq!(a.sample.energy.to_bits(), b.sample.energy.to_bits());
            assert_eq!(a.sample.forces[0][1].to_bits(), (-0.0f64).to_bits());
        }
        assert_eq!(c.quarantine, vec![Q_NONE, Q_SUPERSEDED, Q_NONE, Q_NONE, Q_NONE]);
        assert_eq!(c.references.get(&1).map(|e| e.to_bits()), Some((-0.478_123_456_789_012_3f64).to_bits()));
        assert_eq!(c.history.len(), 1);
        let folder = dir.join(key.slug());
        assert_eq!(SurfaceKey::from_text(&std::fs::read_to_string(folder.join("key.txt")).unwrap()).unwrap(), key);
        assert_eq!(&std::fs::read(folder.join("samples.bin")).unwrap()[..8], MAGIC);
    }

    #[test]
    fn append_survives_a_torn_tail() {
        let dir = root("torn");
        let key = prov().surface_key();
        {
            let (mut c, _) = Collection::open(&dir, &key).unwrap();
            for i in 0..10 {
                c.append(cs(i)).unwrap();
            }
        }
        let path = samples_path(&dir, &key);
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 7).unwrap(); // a crash in the middle of the tenth record
        drop(f);
        {
            let (mut c, r) = Collection::open(&dir, &key).unwrap();
            assert_eq!((r.samples, r.dropped_tail), (9, 1), "{r:?}");
            assert_eq!(c.samples, (0..9).map(cs).collect::<Vec<_>>());
            assert!(r.notes.iter().any(|n| n.contains("dropped")), "{:?}", r.notes);
            let kept = std::fs::read_dir(dir.join(key.slug())).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("samples.bin.dropped")).count();
            assert_eq!(kept, 1, "the dropped bytes are kept aside, never deleted");
            // the file was cut back to the last whole record: a new append lands after it
            c.append(cs(9)).unwrap();
        }
        let (c, r) = Collection::open(&dir, &key).unwrap();
        assert_eq!((r.samples, r.dropped_tail), (10, 0), "{r:?}");
        assert_eq!(c.samples, (0..10).map(cs).collect::<Vec<_>>());
        drop(c);
        // a flipped bit inside the last record is a checksum failure: that record goes
        let mut bytes = std::fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 20] ^= 0x10;
        std::fs::write(&path, &bytes).unwrap();
        let (c, r) = Collection::open(&dir, &key).unwrap();
        assert_eq!((r.samples, r.dropped_tail), (9, 1), "{r:?}");
        drop(c);
    }

    /// A record written by a newer build (an unknown kind with a valid checksum) refuses the open —
    /// it is never truncated as if torn.
    #[test]
    fn a_record_from_a_newer_build_is_refused_not_dropped() {
        let dir = root("newer");
        let key = prov().surface_key();
        {
            let (mut c, _) = Collection::open(&dir, &key).unwrap();
            c.append(cs(0)).unwrap();
        }
        let path = samples_path(&dir, &key);
        let mut bytes = std::fs::read(&path).unwrap();
        let payload = [9u8, 1, 0, 0];
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&crc32(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(&path, &bytes).unwrap();
        let e = Collection::open(&dir, &key).unwrap_err();
        assert!(e.contains("newer build"), "{e}");
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "untouched");
    }

    #[test]
    fn a_folder_of_another_surface_is_refused() {
        let dir = root("other");
        let key = prov().surface_key();
        drop(Collection::open(&dir, &key).unwrap());
        let mut other = key.clone();
        other.teacher = "Kohn-Sham (CPU)".into();
        // a slug collision (forced here by writing the other key into this folder) is caught by key.txt
        std::fs::write(dir.join(key.slug()).join("key.txt"), other.to_text()).unwrap();
        let e = Collection::open(&dir, &key).unwrap_err();
        assert!(e.contains("teacher"), "{e}");
    }

    #[test]
    fn a_second_open_is_read_only() {
        let dir = root("lock");
        let key = prov().surface_key();
        let (mut a, ra) = Collection::open(&dir, &key).unwrap();
        assert!(!ra.locked_by_other && !a.read_only);
        a.append(cs(0)).unwrap();
        let lock = dir.join(key.slug()).join(".lock");
        assert_eq!(std::fs::read_to_string(&lock).unwrap().trim(), std::process::id().to_string());
        let (mut b, rb) = Collection::open(&dir, &key).unwrap();
        assert!(rb.locked_by_other && b.read_only, "{rb:?}");
        assert_eq!(b.samples, a.samples, "a read-only open still reads");
        let e = b.append(cs(1)).unwrap_err();
        assert!(e.contains("read-only"), "{e}");
        assert!(b.set_reference(1, -0.5).is_err());
        assert!(b.write_readme().is_err());
        drop(b);
        assert!(lock.exists(), "a read-only handle never removes the owner's lock");
        drop(a);
        assert!(!lock.exists(), "released on drop");
        let (c, rc) = Collection::open(&dir, &key).unwrap();
        assert!(!rc.locked_by_other && !c.read_only);
        drop(c);
        // a stale lock (its process has exited) is taken over
        #[cfg(windows)]
        let mut child = std::process::Command::new("cmd").args(["/C", "exit 0"]).spawn().unwrap();
        #[cfg(not(windows))]
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let dead = child.id();
        child.wait().unwrap();
        drop(child);
        std::fs::write(&lock, format!("{dead}\n")).unwrap();
        let (d, rd) = Collection::open(&dir, &key).unwrap();
        assert!(!rd.locked_by_other && !d.read_only, "{rd:?}");
        assert!(rd.notes.iter().any(|n| n.contains("stale")), "{:?}", rd.notes);
        assert_eq!(std::fs::read_to_string(&lock).unwrap().trim(), std::process::id().to_string());
    }

    /// Writes `set` as a version-6 file.
    fn samples_file(dir: &Path, name: &str, set: &SampleSet) -> PathBuf {
        let p = dir.join(name);
        sidecar::write_samples(&p, set).unwrap();
        p
    }

    fn set(prov: Provenance, samples: Vec<Sample>, quarantine: Vec<u8>, face: Vec<f64>) -> SampleSet {
        SampleSet { provenance: prov, samples, references: vec![(1, -0.478)], reference_rule: Some(REFERENCE_RULE.into()), quarantine, face_density: face, teacher_solves: None, teacher_solves_note: None, created_by: sidecar::CREATED_BY.into() }
    }

    #[test]
    fn import_reports_what_it_refused() {
        let dir = root("import");
        let key = prov().surface_key();
        let (mut c, _) = Collection::open(&dir, &key).unwrap();
        // a fitted version-5 model of a charged scene (this build's version-7 output, header surgery)
        let mut fitted = crate::sidecar::tests::random_sidecar(41, false);
        fitted.model.provenance = prov();
        fitted.model.provenance.charge = 1;
        let v7 = dir.join("nh4+.v7.model");
        sidecar::write(&v7, &fitted.model, &fitted.accumulator, fitted.report.as_ref(), &fitted.samples, None).unwrap();
        let bytes = std::fs::read(&v7).unwrap();
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[16..16 + hl]).unwrap().replacen("ASMODEL7", "ASMODEL5", 1);
        let mut v5 = sidecar::MAGIC_V5.to_vec();
        v5.extend_from_slice(&(header.len() as u64).to_le_bytes());
        v5.extend_from_slice(header.as_bytes());
        v5.extend_from_slice(&bytes[16 + hl..]);
        let charged = dir.join("nh4+.model");
        std::fs::write(&charged, v5).unwrap();
        assert!(matches!(sidecar::read_any(&charged).unwrap(), sidecar::Contents::Fitted(_)));
        let r = c.import_sidecar(&charged, 1e-4).unwrap();
        assert_eq!(r.imported, 0);
        assert_eq!(r.refused, vec![(WHOLE_FILE, "charge 1: collections are neutral until M9".to_string())]);
        assert!(c.samples.is_empty());

        // a forced multiplicity and another surface: refused as a whole, named
        let mut forced = prov();
        forced.multiplicity = Some(3);
        let f = samples_file(&dir, "triplet.model", &set(forced, vec![h2(0)], vec![], vec![]));
        assert_eq!(c.import_sidecar(&f, 1e-4).unwrap().refused, vec![(WHOLE_FILE, "the scene forced multiplicity 3".to_string())]);
        let mut other = prov();
        other.h = 0.2;
        let f = samples_file(&dir, "coarse.model", &set(other, vec![h2(0)], vec![], vec![]));
        let r = c.import_sidecar(&f, 1e-4).unwrap();
        assert_eq!((r.imported, r.refused.len()), (0, 1));
        assert!(r.refused[0].1.contains("h 0.2 (the collection's 0.25)"), "{:?}", r.refused);
        assert!(c.samples.is_empty());

        // per sample: held, box-flagged, box by face density, a draft audit, a duplicate
        let mut draft = h2(4);
        draft.tag = SampleTag::DraftAudit;
        let mut big_box = prov();
        big_box.grid_n = 81;
        let samples = vec![h2(0), h2(1), h2(2), h2(3), draft, h2(5), h2(0)];
        let f = samples_file(&dir, "h2.model", &set(big_box, samples, vec![0, 1, 3, 0, 0, 0, 0], vec![1e-9, 1e-9, 1e-9, 1e-3, 1e-9, f64::NAN, 1e-9]));
        let r = c.import_sidecar(&f, 1e-4).unwrap();
        assert_eq!(r.imported, 2, "{r:?}");
        let why: Vec<(usize, &str)> = r.refused.iter().map(|(i, s)| (*i, s.as_str())).collect();
        assert_eq!(why.len(), 5, "{why:?}");
        assert_eq!(why[0], (1, "quarantined sample (code 1)"));
        assert_eq!(why[1], (2, "box-affected sample"));
        assert_eq!(why[2].0, 3);
        assert!(why[2].1.starts_with("box-affected sample (face density"), "{why:?}");
        assert_eq!(why[3], (4, "quarantined sample (code 4)"));
        assert_eq!(why[4].0, 6);
        assert!(why[4].1.contains("already"), "{why:?}");
        let got: Vec<_> = c.samples.iter().map(|s| (s.sample.clone(), s.grid_n, s.scene.clone(), s.multiplicity_forced)).collect();
        assert_eq!(got, vec![(h2(0), 81, "h2".to_string(), false), (h2(5), 81, "h2".to_string(), false)]);
        assert_eq!(c.references.get(&1), Some(&-0.478), "a missing reference is taken from the file");
        // importing the same file again adds nothing
        let r = c.import_sidecar(&f, 1e-4).unwrap();
        assert_eq!(r.imported, 0);
        assert_eq!(c.samples.len(), 2);
    }

    #[test]
    fn the_readme_says_what_the_collection_knows() {
        let dir = root("readme");
        let key = prov().surface_key();
        let (mut c, _) = Collection::open(&dir, &key).unwrap();
        for i in 0..3 {
            c.append(cs(i)).unwrap();
        }
        let water = Sample { pos: vec![[0.0; 3], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]], z: vec![8, 1, 1], energy: -17.0, forces: vec![[0.0; 3]; 3], tag: SampleTag::Perturbation, weight: 1.0 };
        c.append(CollectionSample { sample: water, grid_n: 57, multiplicity_forced: false, scene: "h2o".into() }).unwrap();
        c.set_status(2, Q_HELD).unwrap();
        c.set_reference(1, -0.478).unwrap();
        c.write_readme().unwrap();
        let text = std::fs::read_to_string(dir.join(key.slug()).join("README.txt")).unwrap();
        println!("{text}");
        for want in ["lsda", "h 0.25", "4 samples", "H (Z 1)", "O (Z 8)", "H–H", "H–O", "h2o", "1 held"] {
            assert!(text.contains(want), "README lacks '{want}':\n{text}");
        }
    }

    #[test]
    fn the_record_checksum_is_crc32() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }
}
