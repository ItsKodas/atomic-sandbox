//! The `.model` sidecar (spec §6): a fitted [`Model`], the [`Accumulator`] it was solved from
//! (so a loaded model keeps its posterior, its sufficient statistics and its out-of-distribution
//! envelope and can keep learning), the last [`FitReport`] and, optionally, the training
//! [`Sample`]s themselves. Hand-written serialisation (no serde): the format is small and fixed.
//!
//! # Format (versions 1 to 7; this build writes a fitted model as version 7 — for a body-2 or a
//! body-3 basis alike — and samples without a fit as version 6, and reads every version)
//!
//! ```text
//! "ASMODEL1" … "ASMODEL7"          8 bytes, magic (the version)
//! header_len                       u64 little-endian
//! header                           header_len bytes of UTF-8 JSON (below)
//! array₁ … arrayₖ                  in the order the header's "arrays" lists them, each
//!                                  u64 LE length n, then n f64 LE values
//! (end of file — trailing bytes are an error)
//! ```
//!
//! The header is one JSON object with the keys `format` (the magic as a string), `created_by`
//! (string), `provenance` {grid_n, h, sigma, tol, charge, multiplicity (int or null), teacher;
//! from version 3 also `masses`: [[Z, A], …], the recorded isotope set, sorted unique — a record,
//! not a gate, `Provenance` doc; from version 4 also `ionic`, the ionic-surface tag, a gate}, `basis`
//! {n_max, l_max, r_cut, degree_cap, body3, species}, `species` (the basis's, repeated for
//! readers that only want the header), `n_samples` (the model's), `model` {sigma_e, sigma_f,
//! energy_per_atom, lambda, has_covariance}, `fit` (every [`FitReport`] field, or null),
//! `accumulator` {q_prime, yy_e, n_e, sum_log_w_e, yy_f, n_f, sum_log_w_f, n_samples,
//! energy_per_atom (bool or null), envelope_count [per species]}, `reference_rule` (how the
//! reference energies were made — [`REFERENCE_RULE`] for references solved under ruling S9, or
//! null; a missing key reads as null), `samples` (the number of training samples stored) and
//! `arrays` ([[name, length], …]). Numbers: integers as JSON
//! integers; f64 scalars as Rust's shortest round-trip decimal (`{:?}`, parsed back exactly — the
//! round trip is bit-exact for every finite value and ±0), non-finite ones as the strings
//! "NaN", "Infinity", "-Infinity" (a NaN's payload is not kept).
//!
//! The arrays, with S species, P = n_max·S features, Q′ independent columns
//! (`Basis::n_independent_columns`), M stored samples and A = Σ atoms over them:
//!
//! | name | length | content |
//! |---|---|---|
//! | coeffs | S·P | `Model::coeffs`, row-major (centre species, feature) |
//! | mean, scale | P each | standardisation |
//! | ref_energy | S | isolated-atom energies (Hartree) |
//! | sigma_inv_chol | Q′² or 0 | the posterior factor (0 ⇔ `has_covariance` false) |
//! | gram_e, gram_f | Q′² each | `Accumulator` Gram matrices |
//! | rhs_e, rhs_f | Q′ each | right-hand sides |
//! | sumsq | P | per-feature Σφ² |
//! | env_min, env_max, env_sum, env_sumsq | S·P each | the `Envelope` |
//! | sample_atoms, sample_tag, sample_weight, sample_energy | M each | per sample (counts and tag codes as exact integral f64) |
//! | sample_z | A | nuclear charges (integral f64) |
//! | sample_pos, sample_forces | 3A each | Bohr, Hartree/Bohr |
//!
//! Tag codes: LiveStep 0, Perturbation 1, HotBurst 2, Scan 3, Fallback 4, Reference 5.
//!
//! **Version 2** (was written for a basis with `body3: true` while version 1 was written for a
//! body-2 basis; both still read): identical, with P = n_max·S + P₃ and Q′ as
//! `Basis::{n_features, n_independent_columns}` give them for the body-3 basis, and ONE array
//! appended after `sample_forces`:
//!
//! | name | length | content |
//! |---|---|---|
//! | body3_terms | 5·P₃ | (t₁, n₁, t₂, n₂, l) of every body-3 feature in feature order, as integral f64 |
//!
//! **Optional, both versions** (ruling S15, quarantine): when any stored sample is quarantined
//! (kept, but excluded from the fit — `providers::SurrogateKs`), one more array follows last:
//!
//! | name | length | content |
//! |---|---|---|
//! | sample_quarantine | M | 0 trained on, 1 quarantined, 2 superseded, 3 box-affected and excluded (exact integral f64) |
//!
//! A file without it reads as "nothing quarantined", so every earlier file still reads, and a
//! file with nothing quarantined is written exactly as before.
//!
//! **Optional, both versions** (Task 9 fix round 1, box-affected samples): when the writer marks
//! any stored sample's teacher answer as box-affected, one more array follows (after
//! `sample_quarantine` when both are present):
//!
//! | name | length | content |
//! |---|---|---|
//! | sample_face_density | M | the teacher's density on the outermost grid plane, e/Bohr³ (NaN = not known) |
//!
//! A file without it reads as "not known" for every sample.
//!
//! **Optional, every version** (activity spec A2; plan review M4, S17): the header key
//! `teacher_solves` {pass, fallback, audit, verification, kept (JSON integers), complete (bool)}
//! — the model's lifetime teaching solves by origin and how many were kept; `complete: false`
//! when the model was once loaded from a file without the record, so its earlier solves are
//! unrecorded. Written only when the writer has it ([`write_full`]), so a file without it is
//! byte-identical to the older writer's; older readers ignore it (keys are looked up by name)
//! and nothing gates on it. Unreadable → ignored with a note ([`Sidecar::teacher_solves_note`]),
//! never a refused file.
//!
//! **Optional, every version** (2026-10-18 trust review, round 1, SHOULD-2): the header key
//! `failed_audits`, an array of the provider's failed audits not yet re-earned — each {index (the
//! stored sample, or null when its answer was not kept), energy, ionic, rejudges, fenced (bool), z,
//! pos, forces (flat number arrays), optionally origin ("teach-check", "held-out-guess",
//! "shell-check"; absent = a strict-mode audit), and — required with origin "shell-check", absent
//! otherwise — pair ([Z_a, Z_b], the species pair the shell check measured, Z_a ≤ Z_b: the pair its
//! fence refuses in the cutoff shell, carried explicitly rather than re-derived from the case's
//! geometry — final review I2)} — so a model saved while a failed audit untrusts it does not
//! load trusted. Written only when non-empty; older readers ignore it. Unlike `teacher_solves` an
//! unreadable record refuses the file: dropping it would loosen trust — and so does a shell-check
//! case without its pair (a file written before version 5 by a build of the shell fence's first
//! form; no compatibility shim: such models are to be re-sampled).
//!
//! The reader regenerates the terms from the basis (`Basis::body3_terms`) and refuses a file
//! whose list differs: the body-3 layout (unordered pairs once, the degree cap, the l order) is
//! thereby pinned in the file itself, not only in this build. A version-1 file that declares
//! `body3: true` is refused (it has no terms array); a version-2 file always has the terms
//! array; every file's `format` string must equal its magic.
//!
//! **Version 3** (M6 isotopes; from here on one version is written for a body-2 or a body-3
//! basis alike): identical to version 2 except that the `body3_terms` array is present exactly
//! when the basis header says `body3: true`, and `provenance` carries the required `masses`
//! array ([[Z, A], …] as JSON integers; Z in 1..=255, A in 1..=65535, strictly ascending pairs;
//! `[]` when the writer recorded none). Versions 1 and 2 are still read, with an empty mass
//! record.
//!
//! **Version 4** (M7 pseudopotentials): identical to version 3
//! except that `provenance` carries the required `ionic` string (`Provenance::ionic`: `"none"` for the
//! all-electron surface, else `"<policy>/<recipe hash>"`, at most 256 characters). Unlike `masses`,
//! `ionic` **is compared** — a model taught on one ionic surface is never consulted on another.
//! Versions 1 to 3 are still read, with `ionic = "none"`: every model before M7 was all-electron.
//!
//! **Version 5** (the shell fence's species pair, final review I2): identical to version 4
//! except that a `failed_audits` record of origin "shell-check"
//! carries the required `pair` key (above). The bump marks the files whose shell fences are
//! complete; versions 1 to 4 are still read as before, and a version-4 file that holds a
//! shell-check case is refused by the `pair` rule (it cannot have one).
//!
//! **Version 6** (samples without a fit, 2026-10-20: a PH₃ session whose first fit never formed —
//! its P reference did not solve — could not be saved, and the session autosave deleted the
//! minutes of DFT it had taught): written only by [`write_samples`], only when there is no fitted
//! model; a fitted model was still written as version 5, byte for byte as before. The header keeps
//! `format`, `created_by`, `provenance` (as version 5), `reference_rule`, `samples`, `arrays` and the
//! optional `teacher_solves`, and adds `references` — [[Z, E], …], the isolated-atom energies the
//! writer had (Hartree, strictly ascending Z; a species without one is absent: what a failed
//! reference leaves). There is no `basis`, `model`, `fit`, `accumulator` or `failed_audits` key
//! (nothing was fitted, so nothing was audited). The arrays are the sample ones only, in the same
//! order and with the same optional tail:
//! `sample_atoms, sample_tag, sample_weight, sample_energy, sample_z, sample_pos, sample_forces`
//! [, `sample_quarantine`] [, `sample_face_density`]. [`read`] refuses a version-6 file (it has no
//! model); [`read_any`] reads every version; [`read_provenance`] reads its header like any other.
//! A build older than version 6 refuses it as "newer than this build" — never misreads it.
//!
//! **Version 7** (M8a Task 1, the audit history; what this build writes for a fitted model):
//! version 5 plus the provider's [`AuditHistory`] (`crate::history`: per judged audit the model's
//! predictive stds against the teacher's errors, the evidence of a calibration). When the writer's
//! history is not blank (a point recorded or an `Unrecorded` counter set) the header carries the
//! key `audit_history` {points, recorded, no_variance, box_affected, draft, unfit (JSON integers)}
//! right after `arrays`, and the ten history arrays follow the sample tail, last, in the order of
//! [`HistoryArrays::NAMES`](crate::history::HistoryArrays::NAMES) — `audit_atoms`,
//! `audit_model_gen`, `audit_kind`, `audit_energy_std`, `audit_err_energy` (`points` each) and
//! `audit_classes`, `audit_force_std`, `audit_force_std_epistemic`, `audit_err_force`,
//! `audit_teacher_force` (Σ atoms each; module doc of `crate::history` for the contents and the
//! validation; `audit_kind` and the `unfit` counter were added in Task 1's fix round and
//! `audit_teacher_force` in Task 2's, while version 7 was unreleased — no earlier form of
//! version 7 exists in a file). With a blank history nothing is added:
//! the file is version 5's bytes under the version-7 magic. The key and the arrays are read for
//! version 7 only; versions 1 to 5 read with an empty history, version 6 has none (nothing was
//! fitted, so nothing was audited). The bump keeps an older build from misreading the arrays:
//! it refuses the file as "newer than this build".
//!
//! # Validation
//!
//! [`read`] never panics on a corrupt or hostile file: every length is checked against the
//! basis and against the bytes actually present before anything is allocated, the basis must
//! be one this build can evaluate (sorted unique species, 1 ≤ n_max ≤ 64, finite positive
//! r_cut; with body-3 also l_max ≤ `basis::L_MAX_SUPPORTED` and at most 32 species, so the
//! implied array lengths stay representable), integral fields must be integral and in range, and sample weights
//! positive. Any failure is an `io::Error` of kind `InvalidData` naming the problem. Whether the
//! model belongs to the current scene (its [`Provenance`]) is the loader's decision, not the
//! reader's (`sim::Dynamics` refuses a mismatch). [`read_provenance`] reads the header alone (no arrays, no samples) so a
//! viewer can compare a picked file with the scene before loading it.

use crate::descriptor::Basis;
use crate::fit::{Accumulator, FitReport, Sample, SampleTag};
use crate::history::{AuditHistory, HistoryArrays, Unrecorded};
use crate::model::{Model, Provenance};
use std::io;
use std::path::Path;
use std::sync::OnceLock;

/// The file's first eight bytes for a version-1 (body-2) file.
pub const MAGIC: &[u8; 8] = b"ASMODEL1";
/// The first eight bytes of a version-2 file (a body-3 basis; module doc).
pub const MAGIC_V2: &[u8; 8] = b"ASMODEL2";
/// The first eight bytes of a version-3 file (the mass record; still read).
pub const MAGIC_V3: &[u8; 8] = b"ASMODEL3";
/// The first eight bytes of a version-4 file (the ionic tag; still read).
pub const MAGIC_V4: &[u8; 8] = b"ASMODEL4";
/// The first eight bytes of a version-5 file (the shell fence's species pair; still read).
pub const MAGIC_V5: &[u8; 8] = b"ASMODEL5";
/// The first eight bytes of a version-6 file: training samples without a fit ([`write_samples`];
/// module doc, "Version 6").
pub const MAGIC_V6: &[u8; 8] = b"ASMODEL6";
/// The first eight bytes of a version-7 file (the audit history; what this build writes for a
/// fitted model — module doc, "Version 7").
pub const MAGIC_V7: &[u8; 8] = b"ASMODEL7";
/// Species a body-3 file may declare (a hostile-header guard: P₃ grows with (S·n_max)²).
const MAX_SPECIES_V2: usize = 32;
/// The `reference_rule` of references solved under ruling S9: each isolated atom at its
/// lowest-energy multiplicity (`sim::sampling`). A file without it predates the rule.
pub const REFERENCE_RULE: &str = "lowest-energy-multiplicity";
/// The `created_by` string this build writes.
pub const CREATED_BY: &str = concat!("atomic-sandbox surrogate ", env!("CARGO_PKG_VERSION"));

/// Everything a sidecar holds (module doc).
#[derive(Clone, Debug)]
pub struct Sidecar {
    pub model: Model,
    pub accumulator: Accumulator,
    /// The fit that produced `model`, when the writer had it.
    pub report: Option<FitReport>,
    /// The training samples (the accumulated ones first, in the order they were added); empty
    /// for a model-only file ([`Model::write_sidecar`]).
    pub samples: Vec<Sample>,
    pub created_by: String,
    /// How the reference energies were made (header `reference_rule`; module doc).
    pub reference_rule: Option<String>,
    /// Per stored sample: 0 trained on, 1 quarantined, 2 superseded, 3 box-affected (all kept,
    /// 1–3 excluded from the fit; module doc). Empty when the file has no `sample_quarantine` array.
    pub quarantine: Vec<u8>,
    /// Per stored sample: the teacher's face density (e/Bohr³; NaN = not known). Empty when the
    /// file has no `sample_face_density` array.
    pub face_density: Vec<f64>,
    /// The model's lifetime teaching solves (header `teacher_solves`, optional; module doc).
    pub teacher_solves: Option<TeacherSolves>,
    /// Why a present `teacher_solves` record was not read (plan review S17: an unreadable
    /// optional record is reported, never a refused file); `None` when it was read or absent.
    pub teacher_solves_note: Option<String>,
    /// The failed audits the writer still held (header key "failed_audits"; empty when absent).
    pub failed_audits: Vec<AuditCase>,
    /// The writer's audit history (version 7; module doc). Empty for a file of an earlier version
    /// or one written with a blank history.
    pub audit_history: AuditHistory,
}

/// What a version-6 file holds: training samples taught before any fit formed (module doc,
/// "Version 6").
#[derive(Clone, Debug)]
pub struct SampleSet {
    /// The settings the samples were produced at (the store's).
    pub provenance: Provenance,
    pub samples: Vec<Sample>,
    /// Isolated-atom energies the writer had, (Z, Hartree), strictly ascending Z; a species
    /// without one (a reference that did not solve) is absent.
    pub references: Vec<(u8, f64)>,
    /// How `references` were made ([`REFERENCE_RULE`] or `None`).
    pub reference_rule: Option<String>,
    /// As [`Sidecar::quarantine`].
    pub quarantine: Vec<u8>,
    /// As [`Sidecar::face_density`].
    pub face_density: Vec<f64>,
    pub teacher_solves: Option<TeacherSolves>,
    pub teacher_solves_note: Option<String>,
    pub created_by: String,
}

/// Any sidecar: a fitted model (versions 1 to 5 and 7) or samples without a fit (version 6).
#[derive(Clone, Debug)]
pub enum Contents {
    Fitted(Box<Sidecar>),
    Samples(SampleSet),
}

/// One failed audit carried by the file (header key "failed_audits"; module doc): the audited
/// geometry and the teacher's labels there, the stored sample's index (if the answer was kept), the
/// surface's ionic tag, how many refits re-judged it and found it still failed, and whether it was
/// turned into a local fence. The provider (`providers::surrogate`) owns its meaning.
#[derive(Clone, Debug, PartialEq)]
pub struct AuditCase {
    pub pos: Vec<[f64; 3]>,
    pub z: Vec<u8>,
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub index: Option<usize>,
    pub ionic: String,
    pub rejudges: u32,
    pub fenced: bool,
    /// Where it came from ("audit", "teach-check", "held-out-guess", "shell-check"; optional key
    /// "origin", written only when not "audit", so older files stay byte-identical; absent → "audit").
    pub origin: String,
    /// The species pair (Z_a ≤ Z_b) a shell check measured — the pair its fence refuses in the
    /// cutoff shell (key "pair"; required with origin "shell-check", `None` and absent otherwise).
    pub pair: Option<(u8, u8)>,
}

/// Lifetime teaching solves of the model (header key "teacher_solves"; optional, activity spec
/// A2): pass, fallback, audit, verification and how many were kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeacherSolves {
    pub pass: u64,
    pub fallback: u64,
    pub audit: u64,
    pub verification: u64,
    pub kept: u64,
    /// `false`: the model was once loaded from a file without this record, so the counts start
    /// there and earlier solves are unrecorded (plan review M4: the flag must survive the
    /// session autosave, which round-trips every model through the file).
    pub complete: bool,
}

impl Model {
    /// A model-only sidecar: the model and its accumulator, no report, no samples, no
    /// reference rule (module doc). `providers::SurrogateKs` writes the full form through [`write`].
    pub fn write_sidecar(&self, acc: &Accumulator, path: &Path) -> io::Result<()> {
        write(path, self, acc, None, &[], None)
    }

    /// Reads any sidecar's model and accumulator (module doc, "Validation").
    pub fn read_sidecar(path: &Path) -> io::Result<(Model, Accumulator)> {
        let s = read(path)?;
        Ok((s.model, s.accumulator))
    }
}

/// Writes a sidecar (module doc) to `path`, through a temporary file renamed into place so a
/// failed write never leaves a truncated model behind. `reference_rule` is what the writer
/// vouches for about `model.ref_energy` ([`REFERENCE_RULE`] or `None`).
pub fn write(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>) -> io::Result<()> {
    write_with_quarantine(path, model, acc, report, samples, reference_rule, &[])
}

/// [`write`] with the samples' quarantine flags (`quarantine` empty or one per sample; the
/// optional `sample_quarantine` array is written only when one is set — module doc).
pub fn write_with_quarantine(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8]) -> io::Result<()> {
    write_with_flags(path, model, acc, report, samples, reference_rule, quarantine, &[])
}

/// [`write_with_quarantine`] with the samples' face densities (`face_density` empty — no array —
/// or one per sample, NaN for "not known"; module doc).
#[allow(clippy::too_many_arguments)]
pub fn write_with_flags(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64]) -> io::Result<()> {
    write_full(path, model, acc, report, samples, reference_rule, quarantine, face_density, None)
}

/// [`write_with_flags`] with the model's lifetime teaching solves (the optional header key
/// `teacher_solves`, written only when `solves` is `Some`; module doc).
#[allow(clippy::too_many_arguments)]
pub fn write_full(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64], solves: Option<&TeacherSolves>) -> io::Result<()> {
    write_full_with_audits(path, model, acc, report, samples, reference_rule, quarantine, face_density, solves, &[])
}

/// [`write_full`] with the provider's failed audits (header key "failed_audits", written only when
/// `audits` is not empty; module doc).
#[allow(clippy::too_many_arguments)]
pub fn write_full_with_audits(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64], solves: Option<&TeacherSolves>, audits: &[AuditCase]) -> io::Result<()> {
    write_full_with_history(path, model, acc, report, samples, reference_rule, quarantine, face_density, solves, audits, &AuditHistory::default())
}

/// [`write_full_with_audits`] with the provider's audit history (version 7; the key and arrays are
/// written only when `history` is not blank — module doc, "Version 7").
#[allow(clippy::too_many_arguments)]
pub fn write_full_with_history(path: &Path, model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64], solves: Option<&TeacherSolves>, audits: &[AuditCase], history: &AuditHistory) -> io::Result<()> {
    if !quarantine.is_empty() && quarantine.len() != samples.len() {
        return Err(invalid_input(format!("sidecar: {} quarantine flags for {} samples", quarantine.len(), samples.len())));
    }
    if !face_density.is_empty() && face_density.len() != samples.len() {
        return Err(invalid_input(format!("sidecar: {} face densities for {} samples", face_density.len(), samples.len())));
    }
    let bytes = encode(model, acc, report, samples, reference_rule, quarantine, face_density, solves, audits, history)?;
    write_atomic(path, &bytes)
}

/// Writes a version-6 file (module doc, "Version 6"): training samples without a fit, through a
/// temporary file renamed into place like [`write_full_with_audits`]. `InvalidInput` for an empty
/// set, flags of the wrong length, or references that are not finite and strictly ascending in Z.
pub fn write_samples(path: &Path, set: &SampleSet) -> io::Result<()> {
    let bytes = encode_samples(set)?;
    write_atomic(path, &bytes)
}

/// Reads any sidecar (module doc): a fitted model, or a version-6 set of samples without a fit.
pub fn read_any(path: &Path) -> io::Result<Contents> {
    let bytes = std::fs::read(path)?;
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display()));
    let version = parse_header(&bytes).map_err(invalid)?.0;
    if version == 6 {
        decode_samples_file(&bytes).map(Contents::Samples).map_err(invalid)
    } else {
        decode(&bytes).map(|s| Contents::Fitted(Box::new(s))).map_err(invalid)
    }
}

/// `bytes` to `path` through `<name>.tmp` renamed into place (a failed write never leaves a
/// truncated file behind).
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path.file_name().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "sidecar path has no file name"))?;
    let mut tmp_name = name.to_os_string();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Reads only the provenance of a sidecar (magic, header length, the JSON header): no fit, no
/// samples, no arrays are decoded — and only `16 + header_len` bytes of the file are read, so it is
/// cheap on a multi-megabyte model. The header is validated as far as the provenance needs it (the
/// magic, the header length against the file, the `format` string, the provenance fields and their
/// ranges); the body is not looked at, so a file whose body is corrupt still yields its provenance
/// here and is refused by [`read`]. Used by the viewer to compare a model with the scene before
/// loading it (user request 2026-10-18).
pub fn read_provenance(path: &Path) -> io::Result<Provenance> {
    use std::io::Read;
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display()));
    let mut f = std::fs::File::open(path)?;
    let file_len = f.metadata()?.len();
    let mut head = [0u8; 16];
    f.read_exact(&mut head).map_err(|_| invalid(format!("{file_len} bytes is too short for a sidecar")))?;
    let header_len = u64::from_le_bytes(head[8..16].try_into().expect("8 bytes"));
    // the same bounds `decode` applies (before anything is allocated)
    let len = usize::try_from(header_len)
        .ok()
        .filter(|&l| l <= MAX_HEADER && (l as u64).saturating_add(16) <= file_len)
        .ok_or_else(|| invalid(format!("header length {header_len} does not fit the {file_len}-byte file")))?;
    let mut bytes = Vec::with_capacity(16 + len);
    bytes.extend_from_slice(&head);
    bytes.resize(16 + len, 0);
    f.read_exact(&mut bytes[16..]).map_err(|e| invalid(format!("header unreadable: {e}")))?;
    let (version, h) = parse_header(&bytes).map_err(invalid)?;
    decode_provenance(h.get("provenance").map_err(invalid)?, version).map_err(invalid)
}

/// Reads and validates a sidecar (module doc).
pub fn read(path: &Path) -> io::Result<Sidecar> {
    let bytes = std::fs::read(path)?;
    decode(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display())))
}

// ---- encoding ----

/// The tag codes (appended only: a file written before a tag existed reads unchanged; a reader
/// older than a tag refuses a file carrying it with "tag code … is unknown").
const TAGS: [SampleTag; 9] = [
    SampleTag::LiveStep,
    SampleTag::Perturbation,
    SampleTag::HotBurst,
    SampleTag::Scan,
    SampleTag::Fallback,
    SampleTag::Reference,
    SampleTag::Audit,
    SampleTag::DraftAudit,
    SampleTag::Targeted,
];

fn tag_code(t: SampleTag) -> f64 {
    TAGS.iter().position(|&x| x == t).expect("every tag is listed") as f64
}

/// Largest n_max a file may declare (the default is 8; the bound only keeps a hostile header
/// from asking for an absurd basis).
const MAX_N_MAX: usize = 64;
/// Header bytes accepted (a real header is a few kB).
const MAX_HEADER: usize = 1 << 24;
/// The longest `provenance.ionic` tag accepted (`"cores/"` + 16 hex digits is 22).
const MAX_IONIC_TAG: usize = 256;
/// JSON nesting accepted (the header nests three deep).
const MAX_DEPTH: usize = 16;

fn invalid_input(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

#[allow(clippy::too_many_arguments)]
fn encode(model: &Model, acc: &Accumulator, report: Option<&FitReport>, samples: &[Sample], reference_rule: Option<&str>, quarantine: &[u8], face_density: &[f64], solves: Option<&TeacherSolves>, audits: &[AuditCase], history: &AuditHistory) -> io::Result<Vec<u8>> {
    let basis = &model.basis;
    let body3 = basis.body3;
    let magic: &[u8; 8] = MAGIC_V7;
    if body3 && basis.l_max > crate::basis::L_MAX_SUPPORTED {
        return Err(invalid_input(format!("sidecar: body-3 l_max {} exceeds the supported {}", basis.l_max, crate::basis::L_MAX_SUPPORTED)));
    }
    if acc.basis != *basis {
        return Err(invalid_input("sidecar: the accumulator's basis differs from the model's".into()));
    }
    let p = basis.n_features();
    let s = basis.n_species();
    let q = acc.q_prime;
    let env = &acc.envelope;
    let mut shapes = vec![
        ("coeffs (rows)", model.coeffs.len(), s),
        ("mean", model.mean.len(), p),
        ("scale", model.scale.len(), p),
        ("ref_energy", model.ref_energy.len(), s),
        ("gram_e", acc.gram_e.len(), q * q),
        ("gram_f", acc.gram_f.len(), q * q),
        ("rhs_e", acc.rhs_e.len(), q),
        ("rhs_f", acc.rhs_f.len(), q),
        ("sumsq", acc.sumsq.len(), p),
        ("envelope.min", env.min.len(), s * p),
        ("envelope.max", env.max.len(), s * p),
        ("envelope.sum", env.sum.len(), s * p),
        ("envelope.sumsq", env.sumsq.len(), s * p),
        ("envelope.count", env.count.len(), s),
    ];
    for row in &model.coeffs {
        shapes.push(("coeffs (row length)", row.len(), p));
    }
    if let Some(l) = &model.sigma_inv_chol {
        shapes.push(("sigma_inv_chol", l.len(), q * q));
    }
    for (name, len, want) in shapes {
        if len != want {
            return Err(invalid_input(format!("sidecar: {name} has {len} entries, expected {want}")));
        }
    }
    let [s_atoms, s_tag, s_weight, s_energy, s_z, s_pos, s_forces] = sample_arrays(samples)?;

    // arrays, in format order
    let coeffs: Vec<f64> = model.coeffs.iter().flatten().copied().collect();
    let chol: &[f64] = model.sigma_inv_chol.as_deref().unwrap_or(&[]);
    let terms: Vec<f64> = basis.body3_terms().iter().flat_map(|t| [t.t1, t.n1, t.t2, t.n2, t.l].map(|v| v as f64)).collect();
    // the compensated totals (`Accumulator::sums`; read back with a zero compensation)
    let sums = acc.sums();
    let mut arrays: Vec<(&str, &[f64])> = vec![
        ("coeffs", &coeffs),
        ("mean", &model.mean),
        ("scale", &model.scale),
        ("ref_energy", &model.ref_energy),
        ("sigma_inv_chol", chol),
        ("gram_e", &sums.gram_e),
        ("rhs_e", &sums.rhs_e),
        ("gram_f", &sums.gram_f),
        ("rhs_f", &sums.rhs_f),
        ("sumsq", &acc.sumsq),
        ("env_min", &env.min),
        ("env_max", &env.max),
        ("env_sum", &env.sum),
        ("env_sumsq", &env.sumsq),
        ("sample_atoms", &s_atoms),
        ("sample_tag", &s_tag),
        ("sample_weight", &s_weight),
        ("sample_energy", &s_energy),
        ("sample_z", &s_z),
        ("sample_pos", &s_pos),
        ("sample_forces", &s_forces),
    ];
    if body3 {
        arrays.push(("body3_terms", &terms));
    }
    let q_flags = quarantine_values(quarantine)?;
    push_sample_tail(&mut arrays, &q_flags, face_density);
    // the audit history (version 7), last: nothing when blank, so such a file is version 5's bytes
    let with_history = *history != AuditHistory::default();
    let hist_arrays = with_history.then(|| history.to_arrays());
    if let Some(a) = &hist_arrays {
        arrays.extend(a.as_slices());
    }

    let provenance = provenance_json(&model.provenance);
    let species = format!("[{}]", basis.species.iter().map(|z| z.to_string()).collect::<Vec<_>>().join(","));
    let basis_json = obj(&[
        ("n_max", basis.n_max.to_string()),
        ("l_max", basis.l_max.to_string()),
        ("r_cut", num(basis.r_cut)),
        ("degree_cap", basis.degree_cap.to_string()),
        ("body3", basis.body3.to_string()),
        ("species", species.clone()),
    ]);
    let model_json = obj(&[
        ("sigma_e", num(model.sigma_e)),
        ("sigma_f", num(model.sigma_f)),
        ("energy_per_atom", model.energy_per_atom.to_string()),
        ("lambda", num(model.lambda)),
        ("has_covariance", model.sigma_inv_chol.is_some().to_string()),
    ]);
    let fit = match report {
        None => "null".to_string(),
        Some(r) => obj(&[
            ("rmse_energy_per_atom", num(r.rmse_energy_per_atom)),
            ("rmse_force", num(r.rmse_force)),
            ("sigma_e", num(r.sigma_e)),
            ("sigma_f", num(r.sigma_f)),
            ("lambda", num(r.lambda)),
            ("log_evidence", num(r.log_evidence)),
            ("dof_eff", num(r.dof_eff)),
            ("gamma_e", num(r.gamma_e)),
            ("gamma_f", num(r.gamma_f)),
            ("residual_dof_e", num(r.residual_dof_e)),
            ("residual_dof_f", num(r.residual_dof_f)),
            ("n_rows", r.n_rows.to_string()),
            ("n_samples", r.n_samples.to_string()),
            ("q_prime", r.q_prime.to_string()),
            ("condition_estimate", num(r.condition_estimate)),
            ("hyper_iterations", r.hyper_iterations.to_string()),
            ("converged", r.converged.to_string()),
        ]),
    };
    let acc_json = obj(&[
        ("q_prime", acc.q_prime.to_string()),
        ("yy_e", num(sums.yy_e)),
        ("n_e", acc.n_e.to_string()),
        ("sum_log_w_e", num(acc.sum_log_w_e)),
        ("yy_f", num(sums.yy_f)),
        ("n_f", acc.n_f.to_string()),
        ("sum_log_w_f", num(acc.sum_log_w_f)),
        ("n_samples", acc.n_samples.to_string()),
        ("energy_per_atom", acc.energy_per_atom.map_or("null".into(), |b| b.to_string())),
        ("envelope_count", format!("[{}]", env.count.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","))),
    ]);
    let arrays_json = arrays_json(&arrays);
    let mut pairs: Vec<(&str, String)> = vec![
        ("format", string(std::str::from_utf8(magic).expect("ASCII"))),
        ("created_by", string(CREATED_BY)),
        ("provenance", provenance),
        ("basis", basis_json),
        ("species", species),
        ("n_samples", model.n_samples.to_string()),
        ("model", model_json),
        ("fit", fit),
        ("accumulator", acc_json),
        ("reference_rule", reference_rule.map_or("null".to_string(), string)),
        ("samples", samples.len().to_string()),
        ("arrays", arrays_json),
    ];
    if with_history {
        let u = history.unrecorded;
        pairs.push((
            "audit_history",
            obj(&[
                ("points", history.len().to_string()),
                ("recorded", history.recorded().to_string()),
                ("no_variance", u.no_variance.to_string()),
                ("box_affected", u.box_affected.to_string()),
                ("draft", u.draft.to_string()),
                ("unfit", u.unfit.to_string()),
            ]),
        ));
    }
    // optional (module doc): absent, the header is byte-identical to the older writer's
    if let Some(t) = solves {
        pairs.push(("teacher_solves", solves_json(t)));
    }
    if !audits.is_empty() {
        for a in audits {
            if !(a.energy.is_finite() && a.pos.iter().chain(&a.forces).flatten().all(|x| x.is_finite())) || a.pos.len() != a.z.len() || a.forces.len() != a.z.len() {
                return Err(invalid_input("sidecar: a failed audit with a non-finite or misshapen record".into()));
            }
            if (a.origin == "shell-check") != a.pair.is_some() || a.pair.is_some_and(|(p, q)| p > q) {
                return Err(invalid_input("sidecar: a shell-check case carries its sorted species pair, and only a shell-check case does".into()));
            }
        }
        let num = |x: f64| format!("{x:?}");
        let flat = |v: &[[f64; 3]]| format!("[{}]", v.iter().flatten().map(|x| num(*x)).collect::<Vec<_>>().join(","));
        let list: Vec<String> = audits
            .iter()
            .map(|a| {
                let mut kv = vec![
                    ("index", a.index.map_or("null".to_string(), |i| i.to_string())),
                    ("energy", num(a.energy)),
                    ("ionic", string(&a.ionic)),
                    ("rejudges", a.rejudges.to_string()),
                    ("fenced", a.fenced.to_string()),
                    ("z", format!("[{}]", a.z.iter().map(|z| z.to_string()).collect::<Vec<_>>().join(","))),
                    ("pos", flat(&a.pos)),
                    ("forces", flat(&a.forces)),
                ];
                if a.origin != "audit" && !a.origin.is_empty() {
                    kv.push(("origin", string(&a.origin)));
                }
                if let Some((p, q)) = a.pair {
                    kv.push(("pair", format!("[{p},{q}]")));
                }
                obj(&kv)
            })
            .collect();
        pairs.push(("failed_audits", format!("[{}]", list.join(","))));
    }
    Ok(assemble(magic, &obj(&pairs), &arrays))
}

/// The file's bytes: magic, header length, header, then each array as its length and values.
fn assemble(magic: &[u8; 8], header: &str, arrays: &[(&str, &[f64])]) -> Vec<u8> {
    let body: usize = arrays.iter().map(|(_, a)| 8 + 8 * a.len()).sum();
    let mut out = Vec::with_capacity(16 + header.len() + body);
    out.extend_from_slice(magic);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, a) in arrays {
        out.extend_from_slice(&(a.len() as u64).to_le_bytes());
        for x in *a {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// The header's `arrays` list, [[name, length], …].
fn arrays_json(arrays: &[(&str, &[f64])]) -> String {
    format!("[{}]", arrays.iter().map(|(n, a)| format!("[{},{}]", string(n), a.len())).collect::<Vec<_>>().join(","))
}

/// The seven per-sample arrays in format order (atoms, tag, weight, energy, z, pos, forces);
/// `InvalidInput` for a sample whose shapes disagree or that has no atoms.
fn sample_arrays(samples: &[Sample]) -> io::Result<[Vec<f64>; 7]> {
    for (i, smp) in samples.iter().enumerate() {
        if smp.pos.len() != smp.z.len() || smp.pos.len() != smp.forces.len() || smp.pos.is_empty() {
            return Err(invalid_input(format!("sidecar: sample {i} has inconsistent shapes")));
        }
    }
    Ok([
        samples.iter().map(|x| x.pos.len() as f64).collect(),
        samples.iter().map(|x| tag_code(x.tag)).collect(),
        samples.iter().map(|x| x.weight).collect(),
        samples.iter().map(|x| x.energy).collect(),
        samples.iter().flat_map(|x| x.z.iter().map(|&z| z as f64)).collect(),
        samples.iter().flat_map(|x| x.pos.iter().flatten().copied()).collect(),
        samples.iter().flat_map(|x| x.forces.iter().flatten().copied()).collect(),
    ])
}

/// The quarantine codes as array values (checked 0 to 3).
fn quarantine_values(quarantine: &[u8]) -> io::Result<Vec<f64>> {
    if let Some(&q) = quarantine.iter().find(|&&q| q > 3) {
        return Err(invalid_input(format!("sidecar: quarantine code {q} (0, 1, 2 or 3)")));
    }
    Ok(quarantine.iter().map(|&q| q as f64).collect())
}

/// The optional sample tail (module doc): the quarantine flags when one is set, then the face
/// densities when given.
fn push_sample_tail<'a>(arrays: &mut Vec<(&'a str, &'a [f64])>, q_flags: &'a [f64], face_density: &'a [f64]) {
    if q_flags.iter().any(|&q| q != 0.0) {
        arrays.push(("sample_quarantine", q_flags));
    }
    if !face_density.is_empty() {
        arrays.push(("sample_face_density", face_density));
    }
}

/// The header's `provenance` object.
fn provenance_json(pr: &Provenance) -> String {
    obj(&[
        ("grid_n", pr.grid_n.to_string()),
        ("h", num(pr.h)),
        ("sigma", num(pr.sigma)),
        ("tol", num(pr.tol)),
        ("charge", pr.charge.to_string()),
        ("multiplicity", pr.multiplicity.map_or("null".into(), |m| m.to_string())),
        ("teacher", string(&pr.teacher)),
        ("ionic", string(&pr.ionic)),
        ("masses", format!("[{}]", pr.masses.iter().map(|(z, a)| format!("[{z},{a}]")).collect::<Vec<_>>().join(","))),
    ])
}

/// The optional `teacher_solves` record.
fn solves_json(t: &TeacherSolves) -> String {
    obj(&[
        ("pass", t.pass.to_string()),
        ("fallback", t.fallback.to_string()),
        ("audit", t.audit.to_string()),
        ("verification", t.verification.to_string()),
        ("kept", t.kept.to_string()),
        ("complete", t.complete.to_string()),
    ])
}

/// A version-6 file's bytes (module doc, "Version 6").
fn encode_samples(set: &SampleSet) -> io::Result<Vec<u8>> {
    let n = set.samples.len();
    if n == 0 {
        return Err(invalid_input("sidecar: a samples-only file needs at least one sample".into()));
    }
    if !set.quarantine.is_empty() && set.quarantine.len() != n {
        return Err(invalid_input(format!("sidecar: {} quarantine flags for {n} samples", set.quarantine.len())));
    }
    if !set.face_density.is_empty() && set.face_density.len() != n {
        return Err(invalid_input(format!("sidecar: {} face densities for {n} samples", set.face_density.len())));
    }
    if set.references.windows(2).any(|w| w[0].0 >= w[1].0) || set.references.iter().any(|&(z, e)| z == 0 || !e.is_finite()) {
        return Err(invalid_input("sidecar: references must be finite energies in strictly ascending Z ≥ 1".into()));
    }
    let [s_atoms, s_tag, s_weight, s_energy, s_z, s_pos, s_forces] = sample_arrays(&set.samples)?;
    let mut arrays: Vec<(&str, &[f64])> = vec![
        ("sample_atoms", &s_atoms),
        ("sample_tag", &s_tag),
        ("sample_weight", &s_weight),
        ("sample_energy", &s_energy),
        ("sample_z", &s_z),
        ("sample_pos", &s_pos),
        ("sample_forces", &s_forces),
    ];
    let q_flags = quarantine_values(&set.quarantine)?;
    push_sample_tail(&mut arrays, &q_flags, &set.face_density);
    let references = format!("[{}]", set.references.iter().map(|&(z, e)| format!("[{z},{}]", num(e))).collect::<Vec<_>>().join(","));
    let mut pairs: Vec<(&str, String)> = vec![
        ("format", string(std::str::from_utf8(MAGIC_V6).expect("ASCII"))),
        ("created_by", string(CREATED_BY)),
        ("provenance", provenance_json(&set.provenance)),
        ("references", references),
        ("reference_rule", set.reference_rule.as_deref().map_or("null".to_string(), string)),
        ("samples", n.to_string()),
        ("arrays", arrays_json(&arrays)),
    ];
    if let Some(t) = &set.teacher_solves {
        pairs.push(("teacher_solves", solves_json(t)));
    }
    Ok(assemble(MAGIC_V6, &obj(&pairs), &arrays))
}

/// `{"k":v,…}` from already-encoded values.
fn obj(pairs: &[(&str, String)]) -> String {
    let body: Vec<String> = pairs.iter().map(|(k, v)| format!("{}:{v}", string(k))).collect();
    format!("{{{}}}", body.join(","))
}

/// An f64 as JSON: the shortest round-trip decimal, or a string for a non-finite value.
fn num(x: f64) -> String {
    if x.is_finite() {
        format!("{x:?}")
    } else if x.is_nan() {
        "\"NaN\"".into()
    } else if x > 0.0 {
        "\"Infinity\"".into()
    } else {
        "\"-Infinity\"".into()
    }
}

/// A JSON string literal (quotes, backslash and control characters escaped; other characters
/// as UTF-8).
fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---- decoding ----

/// The magic's version, the header JSON (`bytes` holds at least the 16 fixed bytes and the whole
/// header) and the `format` string checked against the magic: the front of [`decode`], shared with
/// [`read_provenance`].
fn parse_header(bytes: &[u8]) -> Result<(u8, Json), String> {
    if bytes.len() < 16 {
        return Err(format!("{} bytes is too short for a sidecar", bytes.len()));
    }
    let version = match &bytes[..8] {
        m if m == MAGIC => 1,
        m if m == MAGIC_V2 => 2,
        m if m == MAGIC_V3 => 3,
        m if m == MAGIC_V4 => 4,
        m if m == MAGIC_V5 => 5,
        m if m == MAGIC_V6 => 6,
        m if m == MAGIC_V7 => 7,
        m if &m[..7] == b"ASMODEL" && m[7].is_ascii_digit() => {
            return Err(format!("format version {} is newer than this build (it reads versions 1 to 7)", m[7] as char));
        }
        _ => return Err("not a model sidecar (magic is not ASMODEL1 … ASMODEL7)".into()),
    };
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
    let header_end = usize::try_from(header_len)
        .ok()
        .filter(|&l| l <= MAX_HEADER)
        .and_then(|l| l.checked_add(16))
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| format!("header length {header_len} does not fit the {}-byte file", bytes.len()))?;
    let header = std::str::from_utf8(&bytes[16..header_end]).map_err(|e| format!("header is not UTF-8: {e}"))?;
    let h = Parser::parse(header)?;
    let format = h.get("format")?.str()?;
    let magic_of_version: &[u8; 8] = match version {
        1 => MAGIC,
        2 => MAGIC_V2,
        3 => MAGIC_V3,
        4 => MAGIC_V4,
        5 => MAGIC_V5,
        6 => MAGIC_V6,
        _ => MAGIC_V7,
    };
    if format.as_bytes() != magic_of_version {
        return Err(format!("header format {format:?} does not match the file's version {version}"));
    }
    Ok((version, h))
}

fn decode(bytes: &[u8]) -> Result<Sidecar, String> {
    let (version, h) = parse_header(bytes)?;
    if version == 6 {
        return Err("the file holds training samples without a fitted model (format version 6): there is no model to read".into());
    }
    let created_by = h.get("created_by")?.str()?.to_string();
    let reference_rule = decode_reference_rule(&h)?;
    let (teacher_solves, teacher_solves_note) = decode_solves_lenient(&h);

    // optional, strict (module doc): a failed audit dropped would loosen trust
    let failed_audits = match h.get("failed_audits") {
        Err(_) | Ok(Json::Null) => Vec::new(),
        Ok(list) => list.arr()?.iter().map(decode_audit_case).collect::<Result<Vec<_>, String>>().map_err(|e| format!("failed_audits: {e}"))?,
    };

    // basis
    let b = h.get("basis")?;
    let n_max = b.get("n_max")?.usize()?;
    if !(1..=MAX_N_MAX).contains(&n_max) {
        return Err(format!("basis n_max {n_max} outside 1..={MAX_N_MAX}"));
    }
    let body3 = b.get("body3")?.bool()?;
    if body3 && version == 1 {
        return Err("a body-3 basis needs format version 2 (this file is version 1)".into());
    }
    let l_max = b.get("l_max")?.usize()?;
    if body3 && l_max > crate::basis::L_MAX_SUPPORTED {
        return Err(format!("basis l_max {l_max} exceeds the supported {}", crate::basis::L_MAX_SUPPORTED));
    }
    let r_cut = b.get("r_cut")?.f64()?;
    if !(r_cut.is_finite() && r_cut > 0.0) {
        return Err(format!("basis r_cut {r_cut} is not a positive finite length"));
    }
    let species = species_list(b.get("species")?)?;
    if species.is_empty() || species.windows(2).any(|w| w[0] >= w[1]) || species[0] == 0 {
        return Err(format!("basis species {species:?} must be non-empty, sorted, unique and ≥ 1"));
    }
    if species_list(h.get("species")?)? != species {
        return Err("header species differ from the basis's".into());
    }
    if body3 && species.len() > MAX_SPECIES_V2 {
        return Err(format!("body-3 basis with {} species (at most {MAX_SPECIES_V2})", species.len()));
    }
    let basis = Basis {
        n_max,
        l_max,
        r_cut,
        degree_cap: b.get("degree_cap")?.usize()?,
        body3,
        species,
    };
    let s = basis.n_species();
    let p = basis.n_features();
    let q = basis.n_independent_columns();

    let provenance = decode_provenance(h.get("provenance")?, version)?;

    let m = h.get("model")?;
    let has_cov = m.get("has_covariance")?.bool()?;
    let (sigma_e, sigma_f, lambda) = (m.get("sigma_e")?.f64()?, m.get("sigma_f")?.f64()?, m.get("lambda")?.f64()?);
    if !(sigma_e.is_finite() && sigma_f.is_finite() && lambda.is_finite()) {
        return Err("model sigma_e, sigma_f and lambda must be finite".into());
    }
    let report = match h.get("fit")? {
        Json::Null => None,
        r => Some(FitReport {
            rmse_energy_per_atom: r.get("rmse_energy_per_atom")?.f64()?,
            rmse_force: r.get("rmse_force")?.f64()?,
            sigma_e: r.get("sigma_e")?.f64()?,
            sigma_f: r.get("sigma_f")?.f64()?,
            lambda: r.get("lambda")?.f64()?,
            log_evidence: r.get("log_evidence")?.f64()?,
            dof_eff: r.get("dof_eff")?.f64()?,
            gamma_e: r.get("gamma_e")?.f64()?,
            gamma_f: r.get("gamma_f")?.f64()?,
            residual_dof_e: r.get("residual_dof_e")?.f64()?,
            residual_dof_f: r.get("residual_dof_f")?.f64()?,
            n_rows: r.get("n_rows")?.usize()?,
            n_samples: r.get("n_samples")?.usize()?,
            q_prime: r.get("q_prime")?.usize()?,
            condition_estimate: r.get("condition_estimate")?.f64()?,
            hyper_iterations: r.get("hyper_iterations")?.usize()?,
            converged: r.get("converged")?.bool()?,
        }),
    };
    let a = h.get("accumulator")?;
    if a.get("q_prime")?.usize()? != q {
        return Err(format!("accumulator q_prime differs from the basis's {q}"));
    }
    let env_count: Vec<usize> = a.get("envelope_count")?.arr()?.iter().map(|c| c.usize()).collect::<Result<_, _>>()?;
    if env_count.len() != s {
        return Err(format!("envelope_count has {} entries for {s} species", env_count.len()));
    }
    let n_stored = h.get("samples")?.usize()?;

    // arrays: the header's list must be the format's, with the lengths the basis implies
    let declared = declared_arrays(&h)?;
    let sp = s * p;
    let mut fixed: Vec<(&str, Option<usize>)> = vec![
        ("coeffs", Some(sp)),
        ("mean", Some(p)),
        ("scale", Some(p)),
        ("ref_energy", Some(s)),
        ("sigma_inv_chol", Some(if has_cov { q * q } else { 0 })),
        ("gram_e", Some(q * q)),
        ("rhs_e", Some(q)),
        ("gram_f", Some(q * q)),
        ("rhs_f", Some(q)),
        ("sumsq", Some(p)),
        ("env_min", Some(sp)),
        ("env_max", Some(sp)),
        ("env_sum", Some(sp)),
        ("env_sumsq", Some(sp)),
        ("sample_atoms", Some(n_stored)),
        ("sample_tag", Some(n_stored)),
        ("sample_weight", Some(n_stored)),
        ("sample_energy", Some(n_stored)),
        ("sample_z", None),
        ("sample_pos", None),
        ("sample_forces", None),
    ];
    let has_terms = version == 2 || (version >= 3 && body3);
    if has_terms {
        fixed.push(("body3_terms", Some(5 * basis.n_body3())));
    }
    // the optional quarantine flags and face densities, in that order (module doc)
    let (has_quarantine, has_face) = sample_tail(&declared, &mut fixed, n_stored);
    // the audit history (version 7, module doc): the key announces the ten arrays, last
    let history_counts = if version >= 7 { decode_history_key(&h)? } else { None };
    if let Some((points, _, _)) = &history_counts {
        for (i, name) in HistoryArrays::NAMES.iter().enumerate() {
            fixed.push((*name, (i < HistoryArrays::PER_POINT).then_some(*points)));
        }
    }
    let arrays = read_arrays(bytes, &declared, &fixed, version)?;
    let mut it = arrays.into_iter();
    let mut next = || it.next().expect("every declared array read");
    let finite = |name: &str, v: &[f64]| -> Result<(), String> {
        match v.iter().position(|x| !x.is_finite()) {
            Some(i) => Err(format!("array {name}[{i}] is not finite")),
            None => Ok(()),
        }
    };
    let no_nan = |name: &str, v: &[f64]| -> Result<(), String> {
        match v.iter().position(|x| x.is_nan()) {
            Some(i) => Err(format!("array {name}[{i}] is NaN")),
            None => Ok(()),
        }
    };

    let coeffs = next();
    let (mean, scale, ref_energy, chol) = (next(), next(), next(), next());
    for (n, v) in [("coeffs", &coeffs), ("mean", &mean), ("scale", &scale), ("ref_energy", &ref_energy), ("sigma_inv_chol", &chol)] {
        finite(n, v)?;
    }
    let model = Model {
        basis: basis.clone(),
        coeffs: coeffs.chunks_exact(p).map(|c| c.to_vec()).collect(),
        mean,
        scale,
        ref_energy,
        sigma_e,
        sigma_f,
        energy_per_atom: m.get("energy_per_atom")?.bool()?,
        lambda,
        sigma_inv_chol: has_cov.then_some(chol),
        n_samples: h.get("n_samples")?.usize()?,
        provenance,
        representatives: OnceLock::new(),
    };

    let mut acc = Accumulator::new(&basis);
    acc.gram_e = next();
    acc.rhs_e = next();
    acc.gram_f = next();
    acc.rhs_f = next();
    acc.sumsq = next();
    for (n, v) in [("gram_e", &acc.gram_e), ("rhs_e", &acc.rhs_e), ("gram_f", &acc.gram_f), ("rhs_f", &acc.rhs_f), ("sumsq", &acc.sumsq)] {
        finite(n, v)?;
    }
    acc.yy_e = a.get("yy_e")?.f64()?;
    acc.n_e = a.get("n_e")?.usize()?;
    acc.sum_log_w_e = a.get("sum_log_w_e")?.f64()?;
    acc.yy_f = a.get("yy_f")?.f64()?;
    acc.n_f = a.get("n_f")?.usize()?;
    acc.sum_log_w_f = a.get("sum_log_w_f")?.f64()?;
    acc.n_samples = a.get("n_samples")?.usize()?;
    acc.energy_per_atom = match a.get("energy_per_atom")? {
        Json::Null => None,
        v => Some(v.bool()?),
    };
    let (min, max, sum, sumsq) = (next(), next(), next(), next());
    no_nan("env_min", &min)?;
    no_nan("env_max", &max)?;
    finite("env_sum", &sum)?;
    finite("env_sumsq", &sumsq)?;
    acc.envelope.count = env_count;
    acc.envelope.min = min;
    acc.envelope.max = max;
    acc.envelope.sum = sum;
    acc.envelope.sumsq = sumsq;

    let (atoms, tags, weights, energies) = (next(), next(), next(), next());
    let (zs, pos, forces) = (next(), next(), next());
    let samples = decode_sample_list(&atoms, &tags, &weights, &energies, &zs, &pos, &forces)?;
    if has_terms {
        let stored = next();
        let expect: Vec<f64> = basis.body3_terms().iter().flat_map(|t| [t.t1, t.n1, t.t2, t.n2, t.l].map(|v| v as f64)).collect();
        if stored != expect {
            let at = stored.iter().zip(&expect).position(|(a, b)| a != b).unwrap_or(0);
            return Err(format!("body3_terms differ from this build's layout for the basis (first at entry {at}: term {}, field {})", at / 5, at % 5));
        }
    }
    let quarantine = if has_quarantine { decode_quarantine(&next())? } else { Vec::new() };
    let face_density = if has_face { next() } else { Vec::new() };
    let audit_history = match history_counts {
        None => AuditHistory::default(),
        Some((_, recorded, unrecorded)) => {
            let a = HistoryArrays::from_vecs([next(), next(), next(), next(), next(), next(), next(), next(), next(), next()]);
            AuditHistory::from_arrays(&a, recorded, unrecorded).map_err(|e| format!("audit_history: {e}"))?
        }
    };
    Ok(Sidecar { model, accumulator: acc, report, samples, created_by, reference_rule, quarantine, face_density, teacher_solves, teacher_solves_note, failed_audits, audit_history })
}

/// The version-7 header key `audit_history` (module doc): (points, recorded, the unrecorded
/// counters), `None` when absent or null. Every field is required inside it, and a present key is
/// strict — a history dropped silently would hide what a calibration has to see.
fn decode_history_key(h: &Json) -> Result<Option<(usize, u64, Unrecorded)>, String> {
    let k = match h.get("audit_history") {
        Err(_) | Ok(Json::Null) => return Ok(None),
        Ok(k) => k,
    };
    let count = |name: &str| -> Result<u64, String> { k.get(name)?.usize().map(|v| v as u64).map_err(|e| format!("audit_history.{name}: {e}")) };
    let points = k.get("points")?.usize().map_err(|e| format!("audit_history.points: {e}"))?;
    if points > crate::history::HISTORY_CAP {
        return Err(format!("audit_history.points {points} exceeds the ring's {}", crate::history::HISTORY_CAP));
    }
    let recorded = count("recorded")?;
    let unrecorded = Unrecorded { no_variance: count("no_variance")?, box_affected: count("box_affected")?, draft: count("draft")?, unfit: count("unfit")? };
    Ok(Some((points, recorded, unrecorded)))
}

/// A version-6 file (module doc, "Version 6").
fn decode_samples_file(bytes: &[u8]) -> Result<SampleSet, String> {
    let (version, h) = parse_header(bytes)?;
    if version != 6 {
        return Err(format!("format version {version} is not a samples-only file"));
    }
    let created_by = h.get("created_by")?.str()?.to_string();
    let reference_rule = decode_reference_rule(&h)?;
    let (teacher_solves, teacher_solves_note) = decode_solves_lenient(&h);
    let provenance = decode_provenance(h.get("provenance")?, version)?;
    let mut references: Vec<(u8, f64)> = Vec::new();
    for r in h.get("references")?.arr()? {
        let (z, e) = match r.arr()? {
            [z, e] => (z.u64()?, e.f64()?),
            _ => return Err("references entries are [Z, E]".into()),
        };
        let z = u8::try_from(z).ok().filter(|&z| z >= 1).ok_or_else(|| format!("reference Z {z} outside 1..=255"))?;
        if !e.is_finite() || references.last().is_some_and(|&(p, _)| p >= z) {
            return Err(format!("reference of Z = {z}: energies must be finite, in strictly ascending Z"));
        }
        references.push((z, e));
    }
    let n_stored = h.get("samples")?.usize()?;
    if n_stored == 0 {
        return Err("a samples-only file without samples".into());
    }
    let declared = declared_arrays(&h)?;
    let mut fixed: Vec<(&str, Option<usize>)> = vec![
        ("sample_atoms", Some(n_stored)),
        ("sample_tag", Some(n_stored)),
        ("sample_weight", Some(n_stored)),
        ("sample_energy", Some(n_stored)),
        ("sample_z", None),
        ("sample_pos", None),
        ("sample_forces", None),
    ];
    let (has_quarantine, has_face) = sample_tail(&declared, &mut fixed, n_stored);
    let mut it = read_arrays(bytes, &declared, &fixed, version)?.into_iter();
    let mut next = || it.next().expect("every declared array read");
    let (atoms, tags, weights, energies) = (next(), next(), next(), next());
    let (zs, pos, forces) = (next(), next(), next());
    let samples = decode_sample_list(&atoms, &tags, &weights, &energies, &zs, &pos, &forces)?;
    let quarantine = if has_quarantine { decode_quarantine(&next())? } else { Vec::new() };
    let face_density = if has_face { next() } else { Vec::new() };
    Ok(SampleSet { provenance, samples, references, reference_rule, quarantine, face_density, teacher_solves, teacher_solves_note, created_by })
}

/// The header's `reference_rule` (a missing key reads as null).
fn decode_reference_rule(h: &Json) -> Result<Option<String>, String> {
    Ok(match h.get("reference_rule") {
        Err(_) | Ok(Json::Null) => None,
        Ok(r) => Some(r.str()?.to_string()),
    })
}

/// The optional `teacher_solves` record, lenient (plan review S17): an unreadable record is a
/// note, never a refused file.
fn decode_solves_lenient(h: &Json) -> (Option<TeacherSolves>, Option<String>) {
    match h.get("teacher_solves") {
        Err(_) | Ok(Json::Null) => (None, None),
        Ok(t) => match decode_teacher_solves(t) {
            Ok(s) => (Some(s), None),
            Err(e) => (None, Some(format!("the model file's \"teacher_solves\" record is unreadable ({e}); its earlier solves count as unrecorded"))),
        },
    }
}

/// The header's `arrays` list as (name, length).
fn declared_arrays(h: &Json) -> Result<Vec<(String, usize)>, String> {
    h.get("arrays")?
        .arr()?
        .iter()
        .map(|e| match e.arr()? {
            [n, l] => Ok((n.str()?.to_string(), l.usize()?)),
            _ => Err("arrays entries are [name, length]".to_string()),
        })
        .collect()
}

/// The optional quarantine flags and face densities after the `fixed` arrays, in that order
/// (module doc): appended to `fixed` when declared; (quarantine present, face densities present).
fn sample_tail(declared: &[(String, usize)], fixed: &mut Vec<(&str, Option<usize>)>, n_stored: usize) -> (bool, bool) {
    let extra: Vec<&str> = declared.iter().skip(fixed.len()).map(|(n, _)| n.as_str()).collect();
    let has_quarantine = extra.first() == Some(&"sample_quarantine");
    let has_face = extra.get(usize::from(has_quarantine)) == Some(&"sample_face_density");
    if has_quarantine {
        fixed.push(("sample_quarantine", Some(n_stored)));
    }
    if has_face {
        fixed.push(("sample_face_density", Some(n_stored)));
    }
    (has_quarantine, has_face)
}

/// The arrays after the header: the declared list must be `fixed` (names in order, lengths where
/// fixed), each length prefix must match, and nothing may trail the last one.
fn read_arrays(bytes: &[u8], declared: &[(String, usize)], fixed: &[(&str, Option<usize>)], version: u8) -> Result<Vec<Vec<f64>>, String> {
    if declared.len() != fixed.len() {
        return Err(format!("{} arrays declared, format version {version} has {}", declared.len(), fixed.len()));
    }
    // `parse_header` bounded the length by the file
    let header_end = 16 + u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
    let mut at = header_end;
    let mut arrays: Vec<Vec<f64>> = Vec::with_capacity(fixed.len());
    for ((name, len), (want_name, want_len)) in declared.iter().zip(fixed) {
        let (want_name, want_len) = (*want_name, *want_len);
        if name != want_name {
            return Err(format!("array {name:?} where the format has {want_name:?}"));
        }
        if let Some(w) = want_len {
            if *len != w {
                return Err(format!("array {name} has {len} entries, the basis implies {w}"));
            }
        }
        let prefix = bytes.get(at..at + 8).ok_or_else(|| format!("file ends before array {name}"))?;
        let n = u64::from_le_bytes(prefix.try_into().expect("8 bytes"));
        if n != *len as u64 {
            return Err(format!("array {name}: length prefix {n} differs from the header's {len}"));
        }
        at += 8;
        let end = len
            .checked_mul(8)
            .and_then(|b| b.checked_add(at))
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| format!("file ends inside array {name}"))?;
        arrays.push(bytes[at..end].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect());
        at = end;
    }
    if at != bytes.len() {
        return Err(format!("{} trailing bytes after the last array", bytes.len() - at));
    }
    Ok(arrays)
}

/// The `sample_quarantine` array's codes (each 0, 1, 2 or 3).
fn decode_quarantine(v: &[f64]) -> Result<Vec<u8>, String> {
    v.iter()
        .enumerate()
        .map(|(i, &q)| if q == 0.0 || q == 1.0 || q == 2.0 || q == 3.0 { Ok(q as u8) } else { Err(format!("sample_quarantine[{i}] = {q} is not 0, 1, 2 or 3")) })
        .collect()
}

/// The samples from their seven arrays (module doc): every count, code, charge and weight checked.
fn decode_sample_list(atoms: &[f64], tags: &[f64], weights: &[f64], energies: &[f64], zs: &[f64], pos: &[f64], forces: &[f64]) -> Result<Vec<Sample>, String> {
    let finite = |name: &str, v: &[f64]| -> Result<(), String> {
        match v.iter().position(|x| !x.is_finite()) {
            Some(i) => Err(format!("array {name}[{i}] is not finite")),
            None => Ok(()),
        }
    };
    let integral = |x: f64, lo: f64, hi: f64| x.fract() == 0.0 && x >= lo && x <= hi;
    let mut total = 0usize;
    for (i, &n) in atoms.iter().enumerate() {
        if !integral(n, 1.0, 1e6) {
            return Err(format!("sample {i}: atom count {n} is not an integer in 1..=1e6"));
        }
        total += n as usize;
    }
    if zs.len() != total || pos.len() != 3 * total || forces.len() != 3 * total {
        return Err(format!(
            "sample arrays hold {} charges, {} position and {} force components for {total} atoms",
            zs.len(),
            pos.len(),
            forces.len()
        ));
    }
    finite("sample_energy", energies)?;
    finite("sample_pos", pos)?;
    finite("sample_forces", forces)?;
    let mut samples = Vec::with_capacity(atoms.len());
    let mut k = 0usize;
    for i in 0..atoms.len() {
        let n = atoms[i] as usize;
        let tag = tags[i];
        if !integral(tag, 0.0, (TAGS.len() - 1) as f64) {
            return Err(format!("sample {i}: tag code {tag} is unknown"));
        }
        let w = weights[i];
        if !(w.is_finite() && w > 0.0) {
            return Err(format!("sample {i}: weight {w} is not positive"));
        }
        let mut z = Vec::with_capacity(n);
        for &zz in &zs[k..k + n] {
            if !integral(zz, 1.0, 255.0) {
                return Err(format!("sample {i}: nuclear charge {zz} is not an integer in 1..=255"));
            }
            z.push(zz as u8);
        }
        let triples = |v: &[f64]| -> Vec<[f64; 3]> { v[3 * k..3 * (k + n)].chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() };
        samples.push(Sample { pos: triples(pos), z, energy: energies[i], forces: triples(forces), tag: TAGS[tag as usize], weight: w });
        k += n;
    }
    Ok(samples)
}

/// One "failed_audits" record (module doc); every field is required inside it.
/// The header's `provenance` object (`version` decides the defaults of the fields older files lack).
fn decode_provenance(pr: &Json, version: u8) -> Result<Provenance, String> {
    let multiplicity = match pr.get("multiplicity")? {
        Json::Null => None,
        m => Some(u32::try_from(m.u64()?).map_err(|_| "multiplicity out of range".to_string())?),
    };
    Ok(Provenance {
        grid_n: pr.get("grid_n")?.usize()?,
        h: pr.get("h")?.f64()?,
        sigma: pr.get("sigma")?.f64()?,
        tol: pr.get("tol")?.f64()?,
        charge: i32::try_from(pr.get("charge")?.i64()?).map_err(|_| "charge out of range".to_string())?,
        multiplicity,
        teacher: pr.get("teacher")?.str()?.to_string(),
        masses: match pr.get("masses") {
            Err(_) if version < 3 => Vec::new(),
            Err(e) => return Err(e),
            Ok(m) => mass_list(m)?,
        },
        ionic: match pr.get("ionic") {
            Err(_) if version < 4 => "none".to_string(),
            Err(e) => return Err(e),
            Ok(t) => {
                let t = t.str()?;
                if t.is_empty() || t.chars().count() > MAX_IONIC_TAG || t.chars().any(char::is_control) {
                    return Err(format!("provenance ionic tag of {} characters (1 to {MAX_IONIC_TAG}, no control characters)", t.chars().count()));
                }
                t.to_string()
            }
        },
    })
}

fn decode_audit_case(a: &Json) -> Result<AuditCase, String> {
    let z: Vec<u8> = a.get("z")?.arr()?.iter().map(|v| v.u64().and_then(|z| u8::try_from(z).map_err(|_| "Z out of range".to_string()))).collect::<Result<_, _>>()?;
    let triples = |key: &str| -> Result<Vec<[f64; 3]>, String> {
        let v: Vec<f64> = a.get(key)?.arr()?.iter().map(|x| x.f64()).collect::<Result<_, _>>()?;
        if v.len() != 3 * z.len() || !v.iter().all(|x| x.is_finite()) {
            return Err(format!("{key}: {} finite components expected", 3 * z.len()));
        }
        Ok(v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
    };
    let energy = a.get("energy")?.f64()?;
    if !energy.is_finite() || z.is_empty() {
        return Err("a failed audit without a finite energy or atoms".into());
    }
    let origin = match a.get("origin") {
        Ok(v) => v.str()?.to_string(),
        Err(_) => "audit".to_string(),
    };
    // the shell fence's species pair (version 5): required with its origin, absent otherwise
    let pair = match a.get("pair") {
        Ok(v) => {
            let p: Vec<u8> = v.arr()?.iter().map(|x| x.u64().and_then(|z| u8::try_from(z).map_err(|_| "pair Z out of range".to_string()))).collect::<Result<_, _>>()?;
            match p[..] {
                [p, q] if p >= 1 && p <= q => Some((p, q)),
                _ => return Err("pair: two species [Z_a, Z_b] with 1 ≤ Z_a ≤ Z_b expected".into()),
            }
        }
        Err(_) => None,
    };
    if (origin == "shell-check") != pair.is_some() {
        return Err(if pair.is_none() {
            "a shell-check case without its species pair (written before format version 5 — the model is to be re-sampled)".into()
        } else {
            format!("a {origin} case with a species pair (only a shell-check case carries one)")
        });
    }
    Ok(AuditCase {
        pos: triples("pos")?,
        forces: triples("forces")?,
        index: match a.get("index")? {
            Json::Null => None,
            v => Some(v.usize()?),
        },
        ionic: a.get("ionic")?.str()?.to_string(),
        rejudges: u32::try_from(a.get("rejudges")?.u64()?).map_err(|_| "rejudges out of range".to_string())?,
        fenced: a.get("fenced")?.bool()?,
        origin,
        pair,
        energy,
        z,
    })
}

/// The optional "teacher_solves" record (activity spec A2); every field is required inside it.
fn decode_teacher_solves(t: &Json) -> Result<TeacherSolves, String> {
    Ok(TeacherSolves {
        pass: t.get("pass")?.u64()?,
        fallback: t.get("fallback")?.u64()?,
        audit: t.get("audit")?.u64()?,
        verification: t.get("verification")?.u64()?,
        kept: t.get("kept")?.u64()?,
        complete: t.get("complete")?.bool()?,
    })
}

/// `[[Z, A], …]`: integral pairs in range, strictly ascending (sorted unique).
fn mass_list(v: &Json) -> Result<Vec<(u8, u16)>, String> {
    let list = v
        .arr()?
        .iter()
        .map(|e| match e.arr()? {
            [z, a] => {
                let z = u8::try_from(z.u64()?).map_err(|_| "masses: Z out of range".to_string())?;
                let a = u16::try_from(a.u64()?).map_err(|_| "masses: A out of range".to_string())?;
                if z == 0 || a == 0 {
                    return Err(format!("masses: ({z}, {a}) is not an isotope"));
                }
                Ok((z, a))
            }
            _ => Err("masses entries are [Z, A]".to_string()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    if list.windows(2).any(|w| w[0] >= w[1]) {
        return Err(format!("masses {list:?} must be sorted and unique"));
    }
    Ok(list)
}

fn species_list(v: &Json) -> Result<Vec<u8>, String> {
    v.arr()?
        .iter()
        .map(|z| u8::try_from(z.u64()?).map_err(|_| "species Z out of range".to_string()))
        .collect()
}

// ---- the JSON subset of the header ----

/// A parsed JSON value; numbers keep their text so integers are read exactly.
#[derive(Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Result<&Json, String> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v).ok_or_else(|| format!("header key {key:?} is missing")),
            _ => Err(format!("expected an object holding {key:?}")),
        }
    }

    fn str(&self) -> Result<&str, String> {
        match self {
            Json::Str(s) => Ok(s),
            other => Err(format!("expected a string, found {other:?}")),
        }
    }

    fn bool(&self) -> Result<bool, String> {
        match self {
            Json::Bool(b) => Ok(*b),
            other => Err(format!("expected true or false, found {other:?}")),
        }
    }

    fn arr(&self) -> Result<&[Json], String> {
        match self {
            Json::Arr(a) => Ok(a),
            other => Err(format!("expected an array, found {other:?}")),
        }
    }

    fn f64(&self) -> Result<f64, String> {
        match self {
            Json::Num(t) => t.parse().map_err(|_| format!("{t:?} is not a number")),
            Json::Str(s) if s == "NaN" => Ok(f64::NAN),
            Json::Str(s) if s == "Infinity" => Ok(f64::INFINITY),
            Json::Str(s) if s == "-Infinity" => Ok(f64::NEG_INFINITY),
            other => Err(format!("expected a number, found {other:?}")),
        }
    }

    fn u64(&self) -> Result<u64, String> {
        match self {
            Json::Num(t) => t.parse().map_err(|_| format!("{t:?} is not a non-negative integer")),
            other => Err(format!("expected an integer, found {other:?}")),
        }
    }

    fn i64(&self) -> Result<i64, String> {
        match self {
            Json::Num(t) => t.parse().map_err(|_| format!("{t:?} is not an integer")),
            other => Err(format!("expected an integer, found {other:?}")),
        }
    }

    fn usize(&self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|_| "integer out of range".to_string())
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("header JSON: trailing text at byte {}", p.i));
        }
        match v {
            Json::Obj(_) => Ok(v),
            _ => Err("header JSON is not an object".into()),
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("header JSON: {what} at byte {}", self.i))
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        self.ws();
        match self.b.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                if self.eat(b'}') {
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    if !self.eat(b':') {
                        return self.err("expected ':'");
                    }
                    kv.push((k, self.value(depth + 1)?));
                    if self.eat(b',') {
                        continue;
                    }
                    if self.eat(b'}') {
                        return Ok(Json::Obj(kv));
                    }
                    return self.err("expected ',' or '}'");
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                if self.eat(b']') {
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    if self.eat(b',') {
                        continue;
                    }
                    if self.eat(b']') {
                        return Ok(Json::Arr(a));
                    }
                    return self.err("expected ',' or ']'");
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.b[start..self.i]).expect("ASCII");
                if t.parse::<f64>().is_err() {
                    return self.err("malformed number");
                }
                Ok(Json::Num(t.to_string()))
            }
            _ => self.err("unexpected character"),
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            self.err("unknown literal")
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return self.err("expected a string");
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.b.get(self.i) else { return self.err("unterminated string") };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else { return self.err("unterminated escape") };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                if !(self.b.get(self.i) == Some(&b'\\') && self.b.get(self.i + 1) == Some(&b'u')) {
                                    return self.err("lone surrogate");
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.err("bad surrogate pair");
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            match char::from_u32(code) {
                                Some(ch) => ch,
                                None => return self.err("invalid \\u escape"),
                            }
                        }
                        _ => return self.err("unknown escape"),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c if c < 0x20 => return self.err("control character in a string"),
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| "header JSON: string is not UTF-8".to_string())
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self.b.get(self.i..self.i + 4).ok_or_else(|| "header JSON: short \\u escape".to_string())?;
        let t = std::str::from_utf8(digits).map_err(|_| "header JSON: bad \\u escape".to_string())?;
        let v = u32::from_str_radix(t, 16).map_err(|_| "header JSON: bad \\u escape".to_string())?;
        self.i += 4;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::Envelope;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("asmodel-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn random_vec(rng: &mut StdRng, n: usize) -> Vec<f64> {
        (0..n).map(|_| rng.gen_range(-1.0..1.0) * 10f64.powi(rng.gen_range(-12..4))).collect()
    }

    /// A random model, accumulator, report and samples with awkward values (subnormal-free but
    /// spread over 16 decades, −0.0, a non-finite condition estimate).
    fn random_sidecar(seed: u64, body3: bool) -> Sidecar {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut basis = Basis::new(&[1, 6, 8]);
        basis.body3 = body3;
        if body3 {
            // C = 9 channels, 45 unordered pairs × 3 l (D = 10 never binds at n_max 3, l_max 2):
            // P = 9 + 135, Q′ = 18 + 3·135
            basis.n_max = 3;
            basis.l_max = 2;
            basis.degree_cap = 10;
        }
        let p = basis.n_features();
        let s = basis.n_species();
        let q = basis.n_independent_columns();
        let mut acc = Accumulator::new(&basis);
        acc.gram_e = random_vec(&mut rng, q * q);
        acc.rhs_e = random_vec(&mut rng, q);
        acc.yy_e = 0.123_456_789_012_345_6;
        acc.n_e = 41;
        acc.sum_log_w_e = -0.0;
        acc.gram_f = random_vec(&mut rng, q * q);
        acc.rhs_f = random_vec(&mut rng, q);
        acc.yy_f = 7.0e-300;
        acc.n_f = 41 * 9;
        acc.sum_log_w_f = 1.0 / 3.0;
        acc.sumsq = random_vec(&mut rng, p);
        acc.envelope = Envelope {
            species: basis.species.clone(),
            n_features: p,
            count: vec![41, 0, 82],
            min: random_vec(&mut rng, s * p),
            max: random_vec(&mut rng, s * p),
            sum: random_vec(&mut rng, s * p),
            sumsq: random_vec(&mut rng, s * p),
        };
        // an untouched species keeps the ±∞ of `Envelope::new`
        acc.envelope.min[p] = f64::INFINITY;
        acc.envelope.max[p] = f64::NEG_INFINITY;
        acc.n_samples = 41;
        acc.energy_per_atom = Some(true);
        let model = Model {
            basis: basis.clone(),
            coeffs: (0..s).map(|_| random_vec(&mut rng, p)).collect(),
            mean: vec![0.0; p],
            scale: random_vec(&mut rng, p),
            ref_energy: vec![-0.478_123_456_789, -37.1, -74.999_999_999_999_9],
            sigma_e: 1.3e-5,
            sigma_f: 0.1 + 0.2,
            energy_per_atom: true,
            lambda: 0.831_234_5,
            sigma_inv_chol: Some(random_vec(&mut rng, q * q)),
            n_samples: 41,
            provenance: Provenance {
                grid_n: 41,
                h: 0.25,
                sigma: 0.25,
                tol: 1e-6,
                charge: -1,
                multiplicity: Some(2),
                teacher: "Kohn-Sham (GPU) \"quoted\" \\ π".into(),
                masses: vec![(1, 1), (1, 2), (6, 12), (8, 16)],
                ionic: "none".into(),
            },
            representatives: OnceLock::new(),
        };
        let report = FitReport {
            rmse_energy_per_atom: 1.26e-5,
            rmse_force: 1.43e-4,
            sigma_e: 1.3e-5,
            sigma_f: 1.4e-4,
            lambda: 1.52,
            log_evidence: 12_345.678_9,
            dof_eff: 7.9,
            gamma_e: 1.1,
            gamma_f: 6.8,
            residual_dof_e: 38.9,
            residual_dof_f: 238.1,
            n_rows: 410,
            n_samples: 41,
            q_prime: q,
            condition_estimate: f64::INFINITY,
            hyper_iterations: 7,
            converged: true,
        };
        let tags = [SampleTag::LiveStep, SampleTag::Perturbation, SampleTag::HotBurst, SampleTag::Scan, SampleTag::Fallback, SampleTag::Reference, SampleTag::Audit, SampleTag::DraftAudit, SampleTag::Targeted];
        let samples = (0..43)
            .map(|i| {
                let n = 1 + i % 4;
                Sample {
                    pos: (0..n).map(|_| [rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0)]).collect(),
                    z: (0..n).map(|k| [1u8, 6, 8][(i + k) % 3]).collect(),
                    energy: rng.gen_range(-80.0..-1.0),
                    forces: (0..n).map(|_| [rng.gen_range(-0.1..0.1), -0.0, rng.gen_range(-0.1..0.1)]).collect(),
                    tag: tags[i % tags.len()],
                    weight: if i % 5 == 0 { 2.5 } else { 1.0 },
                }
            })
            .collect();
        Sidecar { model, accumulator: acc, report: Some(report), samples, created_by: CREATED_BY.into(), reference_rule: Some(REFERENCE_RULE.into()), quarantine: Vec::new(), face_density: Vec::new(), teacher_solves: None, teacher_solves_note: None, failed_audits: Vec::new(), audit_history: AuditHistory::default() }
    }

    /// A body-2 fixture and a directory of its own for test `name`.
    fn fixture_sidecar(name: &str) -> (Sidecar, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("sidecar-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        (random_sidecar(31, false), dir)
    }

    /// The optional "teacher_solves" key (activity spec A2): written only when given, read back
    /// exactly with its "complete" flag (plan review M4), absent → None; a file without it is
    /// byte-identical to the old writer's.
    #[test]
    fn teacher_solves_is_an_optional_header_key() {
        let (orig, dir) = fixture_sidecar("solves");
        let a = dir.join("a.model");
        let b = dir.join("b.model");
        write_with_flags(&a, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[]).unwrap();
        write_full(&b, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], None).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
        for complete in [true, false] {
            let t = TeacherSolves { pass: 273, fallback: 70, audit: 18, verification: 1, kept: 343, complete };
            write_full(&b, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], Some(&t)).unwrap();
            let back = read(&b).unwrap();
            assert_eq!((back.teacher_solves, back.teacher_solves_note), (Some(t), None));
        }
        assert_eq!(read(&a).unwrap().teacher_solves, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan review S17: a malformed optional record is an error of the record only — `read`
    /// reports it as `teacher_solves_note` and never refuses the file through it.
    #[test]
    fn a_malformed_teacher_solves_record_is_reported_not_fatal() {
        let ok = Parser::parse(r#"{"pass": 1, "fallback": 2, "audit": 3, "verification": 0, "kept": 4, "complete": false}"#).unwrap();
        assert_eq!(decode_teacher_solves(&ok), Ok(TeacherSolves { pass: 1, fallback: 2, audit: 3, verification: 0, kept: 4, complete: false }));
        // (the header parser takes objects only: each record is the value of a key)
        for bad in [r#"{"pass": 1}"#, r#"{"pass": -1, "fallback": 2, "audit": 3, "verification": 0, "kept": 4, "complete": true}"#, r#"[1, 2]"#] {
            let h = Parser::parse(&format!(r#"{{"t": {bad}}}"#)).unwrap();
            assert!(decode_teacher_solves(h.get("t").unwrap()).is_err(), "{bad}");
        }
        // and a whole file with such a record reads, with the note
        let (orig, dir) = fixture_sidecar("malformed");
        let path = dir.join("malformed.model");
        let t = TeacherSolves { pass: 1, fallback: 2, audit: 3, verification: 0, kept: 4, complete: true };
        write_full(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], Some(&t)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[16..16 + hl]).unwrap();
        let bad = header.replacen("\"kept\":4", "\"kept\":-4", 1);
        assert_ne!(bad, header);
        let mut v = MAGIC_V7.to_vec();
        v.extend_from_slice(&(bad.len() as u64).to_le_bytes());
        v.extend_from_slice(bad.as_bytes());
        v.extend_from_slice(&bytes[16 + hl..]);
        std::fs::write(&path, v).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(back.teacher_solves, None);
        assert!(back.teacher_solves_note.as_deref().is_some_and(|n| n.contains("unreadable")), "{:?}", back.teacher_solves_note);
        assert_eq!(back.model, orig.model);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// A deterministic audit point of `n` atoms for the history tests (M8a Task 1).
    fn history_point(i: u64, n: usize) -> crate::history::AuditPoint {
        let k = i as f64;
        crate::history::AuditPoint {
            kind: if i % 3 == 0 { crate::history::PointKind::Audit } else { crate::history::PointKind::Fallback { rule: (i % 23) as u8 } },
            classes: (0..n).map(|j| ((i as usize + j) % 3) as u8).collect(),
            pred_force_std: (0..n).map(|j| 1e-3 * (1.0 + k + j as f64)).collect(),
            pred_force_std_epistemic: (0..n).map(|j| 5e-4 * (1.0 + k + j as f64)).collect(),
            pred_energy_std: 2e-5 * (1.0 + k),
            err_force: (0..n).map(|j| 7e-4 * (1.0 + j as f64) * if i % 2 == 0 { 1.0 } else { 1.5 }).collect(),
            teacher_force: (0..n).map(|j| 2e-2 * (1.0 + k) + 1e-3 * j as f64).collect(),
            err_energy: if i % 3 == 0 { -1.3e-5 * (1.0 + k) } else { 2.1e-6 * k },
            n_atoms: n,
            model_gen: 3 + i / 7,
        }
    }

    /// M8a Task 1, Step 1: the ring keeps the newest `HISTORY_CAP` points — after HISTORY_CAP + 10
    /// pushes `len()` is the cap and the oldest ten are gone; `recorded` counts every push.
    #[test]
    fn history_ring_keeps_the_newest() {
        use crate::history::{AuditHistory, HISTORY_CAP};
        let mut h = AuditHistory::default();
        assert!(h.is_empty() && h.len() == 0 && h.recorded() == 0);
        let extra = 10u64;
        for i in 0..HISTORY_CAP as u64 + extra {
            h.push(history_point(i, 1 + (i % 3) as usize));
        }
        assert_eq!(h.len(), HISTORY_CAP);
        assert_eq!(h.recorded(), HISTORY_CAP as u64 + extra);
        let first = h.iter().next().expect("not empty");
        assert_eq!(first, &history_point(extra, 1 + (extra % 3) as usize), "the oldest ten were evicted");
        let last = h.iter().last().expect("not empty");
        let i_last = HISTORY_CAP as u64 + extra - 1;
        assert_eq!(last, &history_point(i_last, 1 + (i_last % 3) as usize));
        assert_eq!(h.iter().count(), HISTORY_CAP);
    }

    /// M8a Task 1, Step 1: a fitted model is written as version 7 with its audit history and reads
    /// back equal (the unrecorded counters included); a version-5 file reads with an empty history;
    /// the standalone `encode`/`decode` pair (Task 3's `history.bin`) round-trips the same ring,
    /// bit for bit; a history with a class code beyond 2 or misshapen arrays is refused.
    #[test]
    fn sidecar_round_trips_the_history() {
        use crate::history::{AuditHistory, HISTORY_CAP};
        let orig = random_sidecar(47, false);
        let mut hist = AuditHistory::default();
        for i in 0..57u64 {
            hist.push(history_point(i, 1 + (i % 4) as usize));
        }
        hist.unrecorded.no_variance = 3;
        hist.unrecorded.box_affected = 2;
        hist.unrecorded.draft = 1;
        hist.unrecorded.unfit = 4;
        let path = tmp("history-v7.model");
        write_full_with_history(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], None, &[], &hist).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC_V7);
        let back = read(&path).unwrap();
        assert_eq!(back.audit_history, hist);
        assert_eq!(back.audit_history.recorded(), 57);
        assert_eq!(back.model, orig.model);
        // a file without a history: the same header and arrays as version 5 under the new magic
        let plain = tmp("history-none.model");
        write(&plain, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None).unwrap();
        let plain_bytes = std::fs::read(&plain).unwrap();
        assert_eq!(&plain_bytes[..8], MAGIC_V7);
        assert_eq!(read(&plain).unwrap().audit_history, AuditHistory::default());
        let hl = u64::from_le_bytes(plain_bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&plain_bytes[16..16 + hl]).unwrap();
        assert!(!header.contains("audit_history"), "{header}");
        // the same bytes as a version-5 file: an empty history
        let h5 = header.replacen("ASMODEL7", "ASMODEL5", 1);
        let mut v5 = MAGIC_V5.to_vec();
        v5.extend_from_slice(&(h5.len() as u64).to_le_bytes());
        v5.extend_from_slice(h5.as_bytes());
        v5.extend_from_slice(&plain_bytes[16 + hl..]);
        let p5 = tmp("history-v5.model");
        std::fs::write(&p5, v5).unwrap();
        let old = read(&p5).unwrap();
        assert_eq!(old.audit_history, AuditHistory::default());
        assert_eq!(old.model, orig.model);
        // the standalone encoding
        let enc = hist.encode();
        assert_eq!(&enc[..8], crate::history::MAGIC);
        let dec = AuditHistory::decode(&enc).unwrap();
        assert_eq!(dec, hist);
        for (a, b) in dec.iter().zip(hist.iter()) {
            assert_eq!(bits(&a.err_force), bits(&b.err_force));
            assert_eq!(bits(&a.pred_force_std), bits(&b.pred_force_std));
            assert_eq!(a.err_energy.to_bits(), b.err_energy.to_bits());
        }
        assert!(AuditHistory::decode(&enc[..enc.len() - 1]).is_err(), "truncated");
        assert!(AuditHistory::decode(&[enc.as_slice(), &[0u8]].concat()).is_err(), "trailing byte");
        assert!(AuditHistory::decode(b"ASAUDIT9").is_err(), "bad magic");
        // a full ring round-trips through the encoding at exactly the cap
        let mut full = AuditHistory::default();
        for i in 0..HISTORY_CAP as u64 + 5 {
            full.push(history_point(i, 2));
        }
        assert_eq!(AuditHistory::decode(&full.encode()).unwrap(), full);
        // refused: a class code of 3, an atom count that no longer matches the per-atom arrays
        let hl7 = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header7 = std::str::from_utf8(&bytes[16..16 + hl7]).unwrap();
        assert!(header7.contains("\"audit_history\":{\"points\":57,\"recorded\":57,\"no_variance\":3,\"box_affected\":2,\"draft\":1,\"unfit\":4}"), "{header7}");
        let n_pts = 57usize;
        let a_sum = hist.to_arrays().classes.len();
        // the body walk: the history arrays are the last ten (five of `points`, five of Σ atoms)
        let per_point = crate::history::HistoryArrays::PER_POINT;
        let hist_bytes = 10 * 8 + 8 * (per_point * n_pts + 5 * a_sum);
        let hist_at = bytes.len() - hist_bytes;
        let classes_at = hist_at + per_point * (8 + 8 * n_pts) + 8;
        let mut bad_class = bytes.clone();
        bad_class[classes_at..classes_at + 8].copy_from_slice(&3.0f64.to_le_bytes());
        let p = tmp("history-bad-class.model");
        std::fs::write(&p, &bad_class).unwrap();
        let e = read(&p).unwrap_err();
        assert!(e.to_string().contains("audit_classes"), "{e}");
        let mut bad_atoms = bytes.clone();
        bad_atoms[hist_at + 8..hist_at + 16].copy_from_slice(&(history_point(0, 1).n_atoms as f64 + 1.0).to_le_bytes());
        std::fs::write(&p, &bad_atoms).unwrap();
        let e = read(&p).unwrap_err();
        assert!(e.to_string().contains("audit_atoms"), "{e}");
        for f in [path, plain, p5, p] {
            let _ = std::fs::remove_file(f);
        }
    }

    fn assert_accumulators_bit_equal(a: &Accumulator, b: &Accumulator) {
        assert_eq!(a.basis, b.basis);
        assert_eq!(a.q_prime, b.q_prime);
        // the compensated totals are what a sidecar stores (`Accumulator::sums`)
        let (sa, sb) = (a.sums(), b.sums());
        for (name, x, y) in [
            ("gram_e", &sa.gram_e, &sb.gram_e),
            ("rhs_e", &sa.rhs_e, &sb.rhs_e),
            ("gram_f", &sa.gram_f, &sb.gram_f),
            ("rhs_f", &sa.rhs_f, &sb.rhs_f),
            ("sumsq", &a.sumsq, &b.sumsq),
            ("env.min", &a.envelope.min, &b.envelope.min),
            ("env.max", &a.envelope.max, &b.envelope.max),
            ("env.sum", &a.envelope.sum, &b.envelope.sum),
            ("env.sumsq", &a.envelope.sumsq, &b.envelope.sumsq),
        ] {
            assert_eq!(bits(x), bits(y), "{name}");
        }
        for (name, x, y) in [("yy_e", sa.yy_e, sb.yy_e), ("sum_log_w_e", a.sum_log_w_e, b.sum_log_w_e), ("yy_f", sa.yy_f, sb.yy_f), ("sum_log_w_f", a.sum_log_w_f, b.sum_log_w_f)] {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}");
        }
        assert_eq!((a.n_e, a.n_f, a.n_samples, a.energy_per_atom), (b.n_e, b.n_f, b.n_samples, b.energy_per_atom));
        assert_eq!(a.envelope.count, b.envelope.count);
        assert_eq!(a.envelope.species, b.envelope.species);
        assert_eq!(a.envelope.n_features, b.envelope.n_features);
    }

    #[test]
    fn sidecar_round_trips_bit_exact() {
        round_trip(random_sidecar(11, false), "round-trip.model", "bare.model", MAGIC_V7);
    }

    /// Version 2: a body-3 basis round-trips bit-exactly through the ASMODEL2 form, and the
    /// loaded model evaluates the body-3 hot path bitwise like the original.
    #[test]
    fn sidecar_v2_round_trips_a_body3_model_bit_exact() {
        let orig = random_sidecar(17, true);
        assert_eq!(orig.model.basis.n_features(), 144);
        assert_eq!(orig.accumulator.q_prime, 423);
        round_trip(orig, "round-trip-v2.model", "bare-v2.model", MAGIC_V7);
    }

    /// Task 9 fix round 1 (S1): the optional `sample_face_density` array round-trips with and
    /// without `sample_quarantine` (NaN = not known survives); a file without it reads as
    /// empty ("not known"); a count that does not match is refused on write.
    #[test]
    fn face_densities_round_trip_and_stay_optional() {
        let dir = std::env::temp_dir().join(format!("sidecar-face-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let orig = random_sidecar(5, true);
        let n = orig.samples.len();
        let face: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { f64::NAN } else { 1e-6 * i as f64 }).collect();
        let mut flags = vec![0u8; n];
        flags[1] = 1;
        for (name, q) in [("both", flags.clone()), ("face-only", Vec::new())] {
            let path = dir.join(format!("{name}.model"));
            write_with_flags(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref(), &q, &face).unwrap();
            let back = read(&path).unwrap();
            assert_eq!(back.face_density.len(), n);
            for (a, b) in back.face_density.iter().zip(&face) {
                assert!(a == b || (a.is_nan() && b.is_nan()), "{a} vs {b}");
            }
            assert_eq!(back.quarantine.is_empty(), q.is_empty());
        }
        let plain = dir.join("plain.model");
        write(&plain, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref()).unwrap();
        assert!(read(&plain).unwrap().face_density.is_empty(), "an older file reads as not known");
        assert!(write_with_flags(&plain, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &face[..1]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Ruling S15: the optional `sample_quarantine` array round-trips (both versions; codes 0–3,
    /// 3 = box-affected since Task 9 fix round 1); a file with nothing quarantined is
    /// byte-identical to the plain writer's and reads as empty flags; a flag other than 0–3 is
    /// refused; a flag count that does not match is refused on write.
    #[test]
    fn quarantine_flags_round_trip_and_stay_optional() {
        for (seed, body3) in [(23u64, false), (29, true)] {
            let orig = random_sidecar(seed, body3);
            let flags: Vec<u8> = (0..orig.samples.len()).map(|i| [0, 0, 0, 1, 0, 2, 3][i % 7]).collect();
            let (with, plain, none) = (tmp(&format!("q-{seed}.model")), tmp(&format!("p-{seed}.model")), tmp(&format!("n-{seed}.model")));
            write_with_quarantine(&with, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref(), &flags).unwrap();
            assert_eq!(read(&with).unwrap().quarantine, flags);
            write(&plain, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref()).unwrap();
            let all_false = vec![0u8; orig.samples.len()];
            write_with_quarantine(&none, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref(), &all_false).unwrap();
            assert_eq!(std::fs::read(&none).unwrap(), std::fs::read(&plain).unwrap(), "nothing quarantined: the same file");
            assert!(read(&plain).unwrap().quarantine.is_empty());
            // a corrupt flag: the last array's first entry is right after its length prefix
            let mut bytes = std::fs::read(&with).unwrap();
            let n = orig.samples.len();
            let at = bytes.len() - 8 * n;
            bytes[at..at + 8].copy_from_slice(&0.5f64.to_le_bytes());
            std::fs::write(&with, &bytes).unwrap();
            let e = read(&with).unwrap_err();
            assert!(e.to_string().contains("sample_quarantine"), "{e}");
            let short = vec![1u8; n - 1];
            assert!(write_with_quarantine(&with, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &short).is_err());
        }
    }

    /// Version 6 (2026-10-20, the PH₃ data loss): samples without a fit round-trip bit-exactly with
    /// their flags, partial references and solve record; `read` refuses the file (no model),
    /// `read_provenance` and `read_any` read it, and a fitted file still reads as `Fitted`.
    #[test]
    fn samples_without_a_fit_round_trip_as_version_6() {
        let orig = random_sidecar(31, false);
        let n = orig.samples.len();
        let set = SampleSet {
            provenance: orig.model.provenance.clone(),
            samples: orig.samples.clone(),
            // H solved, P did not: only H's energy is known
            references: vec![(1, -0.4581234567891)],
            reference_rule: Some(REFERENCE_RULE.into()),
            quarantine: (0..n).map(|i| [0, 1, 0, 3][i % 4]).collect(),
            face_density: (0..n).map(|i| if i % 3 == 0 { f64::NAN } else { 1e-6 * i as f64 }).collect(),
            teacher_solves: Some(TeacherSolves { pass: 40, fallback: 3, audit: 0, verification: 0, kept: 43, complete: true }),
            teacher_solves_note: None,
            created_by: CREATED_BY.into(),
        };
        let p = tmp("unfitted-v6.model");
        write_samples(&p, &set).unwrap();
        assert_eq!(&std::fs::read(&p).unwrap()[..8], MAGIC_V6);
        let Contents::Samples(back) = read_any(&p).unwrap() else { panic!("a samples-only file reads as Samples") };
        assert_eq!(back.provenance, set.provenance);
        assert_eq!(back.references.len(), 1);
        assert_eq!((back.references[0].0, back.references[0].1.to_bits()), (1, set.references[0].1.to_bits()));
        assert_eq!(back.reference_rule.as_deref(), Some(REFERENCE_RULE));
        assert_eq!(back.quarantine, set.quarantine);
        assert_eq!(bits(&back.face_density), bits(&set.face_density));
        assert_eq!(back.teacher_solves, set.teacher_solves);
        assert_eq!(back.samples.len(), n);
        for (a, b) in back.samples.iter().zip(&set.samples) {
            assert_eq!((a.z.clone(), a.tag, a.weight.to_bits(), a.energy.to_bits()), (b.z.clone(), b.tag, b.weight.to_bits(), b.energy.to_bits()));
            assert_eq!(bits(&a.pos.concat()), bits(&b.pos.concat()));
            assert_eq!(bits(&a.forces.concat()), bits(&b.forces.concat()));
        }
        assert_eq!(read_provenance(&p).unwrap(), set.provenance);
        let e = read(&p).unwrap_err();
        assert!(e.to_string().contains("without a fitted model"), "{e}");
        // nothing optional: no flags, no densities, no references
        let bare = SampleSet { quarantine: Vec::new(), face_density: Vec::new(), references: Vec::new(), teacher_solves: None, reference_rule: None, ..set.clone() };
        write_samples(&p, &bare).unwrap();
        let Contents::Samples(back) = read_any(&p).unwrap() else { panic!() };
        assert!(back.quarantine.is_empty() && back.face_density.is_empty() && back.references.is_empty() && back.teacher_solves.is_none());
        // refused on write: no samples, misshapen flags, unsorted references
        assert!(write_samples(&p, &SampleSet { samples: Vec::new(), quarantine: Vec::new(), face_density: Vec::new(), ..set.clone() }).is_err());
        assert!(write_samples(&p, &SampleSet { quarantine: vec![0; n - 1], ..set.clone() }).is_err());
        assert!(write_samples(&p, &SampleSet { references: vec![(15, -6.4), (1, -0.45)], ..set.clone() }).is_err());
        // a truncated samples-only file is an error, never a panic
        write_samples(&p, &set).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        std::fs::write(&p, &bytes[..bytes.len() - 5]).unwrap();
        assert!(read_any(&p).is_err());
        // a fitted file through `read_any`
        let q = tmp("fitted-any.model");
        write(&q, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref()).unwrap();
        assert!(matches!(read_any(&q).unwrap(), Contents::Fitted(_)));
    }

    fn round_trip(orig: Sidecar, name: &str, bare_name: &str, magic: &[u8; 8]) {
        let path = tmp(name);
        write(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, orig.reference_rule.as_deref()).unwrap();
        assert_eq!(&std::fs::read(&path).unwrap()[..8], magic);
        let back = read(&path).unwrap();
        // Model: PartialEq is bitwise for the finite values here except ±0 (== treats them equal),
        // so compare the bits of every float array as well
        assert_eq!(back.model, orig.model);
        for (a, b) in back.model.coeffs.iter().zip(&orig.model.coeffs) {
            assert_eq!(bits(a), bits(b));
        }
        assert_eq!(bits(&back.model.mean), bits(&orig.model.mean));
        assert_eq!(bits(&back.model.scale), bits(&orig.model.scale));
        assert_eq!(bits(&back.model.ref_energy), bits(&orig.model.ref_energy));
        assert_eq!(bits(back.model.sigma_inv_chol.as_ref().unwrap()), bits(orig.model.sigma_inv_chol.as_ref().unwrap()));
        for (x, y) in [(back.model.sigma_e, orig.model.sigma_e), (back.model.sigma_f, orig.model.sigma_f), (back.model.lambda, orig.model.lambda), (back.model.provenance.h, orig.model.provenance.h), (back.model.provenance.tol, orig.model.provenance.tol)] {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_accumulators_bit_equal(&back.accumulator, &orig.accumulator);
        // the header: report (∞ condition estimate included), provenance, created-by
        let (r, o) = (back.report.as_ref().unwrap(), orig.report.as_ref().unwrap());
        assert_eq!(r, o);
        assert_eq!(r.condition_estimate, f64::INFINITY);
        assert_eq!(back.created_by, CREATED_BY);
        assert_eq!(back.reference_rule.as_deref(), Some(REFERENCE_RULE));
        assert_eq!(back.model.provenance, orig.model.provenance);
        assert_eq!(back.model.provenance.masses, orig.model.provenance.masses, "the mass record (not compared by ==)");
        // samples, −0.0 force components included
        assert_eq!(back.samples.len(), orig.samples.len());
        for (a, b) in back.samples.iter().zip(&orig.samples) {
            assert_eq!(a, b);
            let fa: Vec<f64> = a.forces.iter().flatten().copied().collect();
            let fb: Vec<f64> = b.forces.iter().flatten().copied().collect();
            assert_eq!(bits(&fa), bits(&fb));
            assert_eq!(a.energy.to_bits(), b.energy.to_bits());
        }
        // a loaded model evaluates bitwise like the original
        let pos = [[0.0, 0.0, 0.0], [1.1, 0.2, -0.3], [-0.9, 0.4, 0.5]];
        let z = [8, 1, 1];
        let (e0, f0) = crate::model::energy_and_forces(&orig.model, &pos, &z).unwrap();
        let (e1, f1) = crate::model::energy_and_forces(&back.model, &pos, &z).unwrap();
        assert_eq!(e0.to_bits(), e1.to_bits());
        assert_eq!(f0, f1);

        // the model-only form: no report, no samples, no covariance
        let mut bare = orig.model.clone();
        bare.sigma_inv_chol = None;
        let path2 = tmp(bare_name);
        bare.write_sidecar(&orig.accumulator, &path2).unwrap();
        let (m, a) = Model::read_sidecar(&path2).unwrap();
        assert_eq!(m, bare);
        assert_accumulators_bit_equal(&a, &orig.accumulator);
        let full = read(&path2).unwrap();
        assert!(full.report.is_none() && full.samples.is_empty() && full.reference_rule.is_none());
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path2);
    }

    /// Spec §5.3 / P2: the version-3 header records the scene's mass set in `provenance.masses`
    /// and reads it back; a version-1 or -2 file (made here from the same fixtures by header
    /// surgery, since this build writes only version 3) reads an empty list; two provenances
    /// differing only in masses compare EQUAL (the gate does not look at masses) and
    /// `describe` ends with the mass labels.
    #[test]
    fn sidecar_v3_records_the_mass_set_and_v2_reads_empty() {
        let mut orig = random_sidecar(23, true);
        orig.model.provenance.masses = vec![(1, 1), (1, 2)];
        let path = tmp("masses-v3.model");
        write(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC_V7);
        let back = read(&path).unwrap();
        assert_eq!(back.model.provenance.masses, vec![(1, 1), (1, 2)]);
        assert_eq!(back.model.provenance, orig.model.provenance);
        assert!(back.model.provenance.describe_recorded().ends_with("masses ¹H ²H (recorded, not compared)"), "{}", back.model.provenance.describe_recorded());
        assert_eq!(back.model.provenance.masses_label(), "¹H ²H");
        assert!(!back.model.provenance.describe().contains("masses"), "the gate's description names compared fields only: {}", back.model.provenance.describe());
        // P2: masses are recorded, not gated on
        let mut other = orig.model.provenance.clone();
        other.masses = vec![(1, 2), (8, 16)];
        assert_eq!(other, orig.model.provenance);
        other.charge += 1;
        assert_ne!(other, orig.model.provenance);
        // an empty record describes itself as such
        let mut none = orig.model.provenance.clone();
        none.masses.clear();
        assert_eq!(none.masses_label(), "unrecorded");
        assert!(none.describe_recorded().ends_with("masses unrecorded (recorded, not compared)"), "{}", none.describe_recorded());

        // the same fixture as a version-2 file: v2 magic and format string, no masses key
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[16..16 + header_len]).unwrap();
        let masses_key = ",\"masses\":[[1,1],[1,2]]";
        assert!(header.contains(masses_key), "{header}");
        let as_version = |magic: &[u8; 8]| {
            let h = header.replacen("ASMODEL7", std::str::from_utf8(magic).unwrap(), 1).replacen(masses_key, "", 1).replacen(",\"ionic\":\"none\"", "", 1);
            let mut v = magic.to_vec();
            v.extend_from_slice(&(h.len() as u64).to_le_bytes());
            v.extend_from_slice(h.as_bytes());
            v.extend_from_slice(&bytes[16 + header_len..]);
            v
        };
        let p2 = tmp("masses-v2.model");
        std::fs::write(&p2, as_version(MAGIC_V2)).unwrap();
        let v2 = read(&p2).unwrap();
        assert_eq!(v2.model.provenance.masses, Vec::<(u8, u16)>::new());
        assert_eq!(v2.model, orig.model, "everything else reads as before");
        assert!(v2.model.basis.body3);
        // and a body-2 fixture as version 1
        let mut b2 = random_sidecar(29, false);
        b2.model.provenance.masses = vec![(6, 12), (8, 16)];
        let p3 = tmp("masses-v3-body2.model");
        write(&p3, &b2.model, &b2.accumulator, None, &[], None).unwrap();
        let bytes2 = std::fs::read(&p3).unwrap();
        assert_eq!(&bytes2[..8], MAGIC_V7, "version 7 is written for a body-2 basis too");
        let hl = u64::from_le_bytes(bytes2[8..16].try_into().unwrap()) as usize;
        let h1 = std::str::from_utf8(&bytes2[16..16 + hl]).unwrap().replacen("ASMODEL7","ASMODEL1", 1).replacen(",\"masses\":[[6,12],[8,16]]", "", 1).replacen(",\"ionic\":\"none\"", "", 1);
        let mut v1 = MAGIC.to_vec();
        v1.extend_from_slice(&(h1.len() as u64).to_le_bytes());
        v1.extend_from_slice(h1.as_bytes());
        v1.extend_from_slice(&bytes2[16 + hl..]);
        let p1 = tmp("masses-v1.model");
        std::fs::write(&p1, v1).unwrap();
        let r1 = read(&p1).unwrap();
        assert!(r1.model.provenance.masses.is_empty());
        assert_eq!(r1.model, b2.model);
        // version 3 requires the key and validates it
        for (what, bad) in [
            ("missing masses", header.replacen(masses_key, "", 1)),
            ("unsorted masses", header.replacen(masses_key, ",\"masses\":[[1,2],[1,1]]", 1)),
            ("duplicate masses", header.replacen(masses_key, ",\"masses\":[[1,1],[1,1]]", 1)),
            ("Z = 0", header.replacen(masses_key, ",\"masses\":[[0,1]]", 1)),
            ("A out of range", header.replacen(masses_key, ",\"masses\":[[1,70000]]", 1)),
            ("not a pair", header.replacen(masses_key, ",\"masses\":[[1,1,1]]", 1)),
        ] {
            let mut v = MAGIC_V7.to_vec();
            v.extend_from_slice(&(bad.len() as u64).to_le_bytes());
            v.extend_from_slice(bad.as_bytes());
            v.extend_from_slice(&bytes[16 + header_len..]);
            let p = tmp("masses-bad.model");
            std::fs::write(&p, v).unwrap();
            let e = read(&p).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{what}: {e}");
            println!("{what}: {e}");
        }
        for p in [path, p2, p3, p1, tmp("masses-bad.model")] {
            let _ = std::fs::remove_file(p);
        }
    }

    /// M7 Task 9 (spec §5.5): the version-4 header carries `provenance.ionic`, round-trips it, and the tag is
    /// **compared** (unlike the masses): a model taught on one ionic surface is not the model of another.
    #[test]
    fn version_4_round_trips_the_ionic_tag() {
        let mut orig = random_sidecar(31, true);
        orig.model.provenance.ionic = "cores/0123456789abcdef".into();
        let path = tmp("ionic-v4.model");
        write(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"ASMODEL7");
        let back = read(&path).unwrap();
        assert_eq!(back.model.provenance.ionic, "cores/0123456789abcdef");
        assert_eq!(back.model, orig.model);
        assert!(back.model.provenance.describe().ends_with(", ionic cores/0123456789abcdef"), "{}", back.model.provenance.describe());
        let mut other = orig.model.provenance.clone();
        other.ionic = "all/0123456789abcdef".into();
        assert_ne!(other, orig.model.provenance, "the ionic tag is a gate");
        // version 4 requires the key and validates it
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[16..16 + hl]).unwrap();
        let key = ",\"ionic\":\"cores/0123456789abcdef\"";
        assert!(header.contains(key), "{header}");
        for (what, bad) in [
            ("missing ionic", header.replacen(key, "", 1)),
            ("empty ionic", header.replacen(key, ",\"ionic\":\"\"", 1)),
            ("control character", header.replacen(key, ",\"ionic\":\"a\\u0001b\"", 1)),
            ("not a string", header.replacen(key, ",\"ionic\":7", 1)),
            ("too long", header.replacen(key, &format!(",\"ionic\":\"{}\"", "x".repeat(MAX_IONIC_TAG + 1)), 1)),
        ] {
            let mut v = MAGIC_V7.to_vec();
            v.extend_from_slice(&(bad.len() as u64).to_le_bytes());
            v.extend_from_slice(bad.as_bytes());
            v.extend_from_slice(&bytes[16 + hl..]);
            let p = tmp("ionic-bad.model");
            std::fs::write(&p, v).unwrap();
            let e = read(&p).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{what}: {e}");
            let _ = std::fs::remove_file(&p);
        }
        let _ = std::fs::remove_file(path);
    }

    /// User request 2026-10-18: `read_provenance` returns exactly the provenance `read` does (masses and ionic
    /// tag included, every version), from the header alone — a file whose body is cut off still yields it, and
    /// `read` refuses the same file — and it refuses what is not a sidecar with an `InvalidData` error.
    #[test]
    fn the_header_only_reader_returns_the_provenance_of_every_version() {
        let mut orig = random_sidecar(41, true);
        orig.model.provenance.ionic = "cores/0123456789abcdef".into();
        let path = tmp("prov-only.model");
        write(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None).unwrap();
        let p = read_provenance(&path).unwrap();
        assert_eq!(p, orig.model.provenance);
        assert_eq!(p.masses, orig.model.provenance.masses, "the mass record (not compared by ==)");
        assert_eq!(p.teacher, orig.model.provenance.teacher);
        assert_eq!((p.grid_n, p.charge, p.multiplicity), (41, -1, Some(2)));
        // the body is never looked at: cut it off and the provenance still reads, the full reader refuses
        let bytes = std::fs::read(&path).unwrap();
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let cut = tmp("prov-only-cut.model");
        std::fs::write(&cut, &bytes[..16 + hl]).unwrap();
        assert_eq!(read_provenance(&cut).unwrap(), orig.model.provenance);
        assert_eq!(read(&cut).unwrap_err().kind(), io::ErrorKind::InvalidData);
        // an older version: header surgery as in `old_versions_read_as_all_electron` gives no mass record and "none"
        let header = std::str::from_utf8(&bytes[16..16 + hl]).unwrap();
        let v2 = header.replacen("ASMODEL7","ASMODEL2", 1).replacen(",\"ionic\":\"cores/0123456789abcdef\"", "", 1);
        let masses = format!(",\"masses\":[{}]", orig.model.provenance.masses.iter().map(|(z, a)| format!("[{z},{a}]")).collect::<Vec<_>>().join(","));
        let v2 = v2.replacen(&masses, "", 1);
        let mut v = MAGIC_V2.to_vec();
        v.extend_from_slice(&(v2.len() as u64).to_le_bytes());
        v.extend_from_slice(v2.as_bytes());
        let old = tmp("prov-only-v2.model");
        std::fs::write(&old, v).unwrap();
        let p2 = read_provenance(&old).unwrap();
        assert_eq!((p2.ionic.as_str(), p2.masses.len()), ("none", 0));
        // not a sidecar, too short, a header length past the end, a missing file
        for (what, data) in [
            ("bad magic", [b"ASMODELX".as_slice(), &bytes[8..]].concat()),
            ("empty", Vec::new()),
            ("short", bytes[..12].to_vec()),
            ("header length past the end", [&bytes[..8], &u64::MAX.to_le_bytes()[..], &bytes[16..]].concat()),
            ("header cut", bytes[..16 + hl / 2].to_vec()),
        ] {
            let bad = tmp("prov-only-bad.model");
            std::fs::write(&bad, data).unwrap();
            let e = read_provenance(&bad).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{what}: {e}");
            let _ = std::fs::remove_file(&bad);
        }
        assert_eq!(read_provenance(&tmp("prov-only-missing.model")).unwrap_err().kind(), io::ErrorKind::NotFound);
        for f in [path, cut, old] {
            let _ = std::fs::remove_file(f);
        }
    }

    /// Version 5 (final review I2): a "shell-check" audit case round-trips its species pair; a
    /// shell-check case without the pair (what a build before version 5 wrote) is refused, and so
    /// is any other case carrying one — the reader never guesses a fence's pair.
    #[test]
    fn version_5_carries_the_shell_fence_pair_and_refuses_a_fence_without_it() {
        let orig = random_sidecar(41, false);
        let case = |origin: &str, pair: Option<(u8, u8)>| AuditCase {
            pos: vec![[0.0; 3], [5.2, 0.0, 0.0]],
            z: vec![1, 1],
            energy: -1.0,
            forces: vec![[1e-3, 0.0, 0.0], [-1e-3, 0.0, 0.0]],
            index: None,
            ionic: "none".into(),
            rejudges: 3,
            fenced: true,
            origin: origin.into(),
            pair,
        };
        let audits = [case("shell-check", Some((1, 1))), case("audit", None)];
        let path = tmp("shell-pair-v5.model");
        write_full_with_audits(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None, &[], &[], None, &audits).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], MAGIC_V7);
        let back = read(&path).unwrap();
        assert_eq!(back.failed_audits, audits.to_vec());
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[16..16 + hl]).unwrap();
        assert!(header.contains(",\"origin\":\"shell-check\",\"pair\":[1,1]}"), "{header}");
        for (what, bad, says) in [
            ("a shell-check case without its pair", header.replacen(",\"pair\":[1,1]", "", 1), "without its species pair"),
            ("an audit case with a pair", header.replacen("\"origin\":\"shell-check\",\"pair\":[1,1]}", "\"origin\":\"shell-check\",\"pair\":[1,1]},{\"index\":null,\"energy\":-1.0,\"ionic\":\"none\",\"rejudges\":3,\"fenced\":true,\"z\":[1,1],\"pos\":[0.0,0.0,0.0,5.2,0.0,0.0],\"forces\":[0.001,0.0,0.0,-0.001,0.0,0.0],\"pair\":[1,1]}", 1), "only a shell-check case carries one"),
            ("an unsorted pair", header.replacen(",\"pair\":[1,1]", ",\"pair\":[2,1]", 1), "Z_a ≤ Z_b"),
        ] {
            assert_ne!(bad, header, "{what}");
            let mut v = MAGIC_V7.to_vec();
            v.extend_from_slice(&(bad.len() as u64).to_le_bytes());
            v.extend_from_slice(bad.as_bytes());
            v.extend_from_slice(&bytes[16 + hl..]);
            let p = tmp("shell-pair-bad.model");
            std::fs::write(&p, v).unwrap();
            let e = read(&p).unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{what}: {e}");
            assert!(e.to_string().contains(says), "{what}: {e}");
            println!("{what}: {e}");
            let _ = std::fs::remove_file(&p);
        }
        // the writer refuses the same mismatch before a file exists
        let e = write_full_with_audits(&path, &orig.model, &orig.accumulator, None, &[], None, &[], &[], None, &[case("shell-check", None)]).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput, "{e}");
        let _ = std::fs::remove_file(path);
    }

    /// M7 Task 9: a version-1, -2, -3 or -4 file (made by header surgery from this build's own v5 output) reads
    /// `ionic == "none"` — every model before M7 was all-electron, and version 4 carries the tag — and equals
    /// the same model with that tag.
    #[test]
    fn old_versions_read_as_all_electron() {
        let orig = random_sidecar(37, true);
        assert_eq!(orig.model.provenance.ionic, "none");
        let path = tmp("ionic-old-src.model");
        write(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[16..16 + hl]).unwrap();
        let masses = format!(",\"masses\":[{}]", orig.model.provenance.masses.iter().map(|(z, a)| format!("[{z},{a}]")).collect::<Vec<_>>().join(","));
        for (name, magic, strip_masses) in [("v4", MAGIC_V4, false), ("v3", MAGIC_V3, false), ("v2", MAGIC_V2, true)] {
            let mut h = header.replacen("ASMODEL7", std::str::from_utf8(magic).unwrap(), 1);
            if magic != MAGIC_V4 {
                h = h.replacen(",\"ionic\":\"none\"", "", 1);
            }
            if strip_masses {
                h = h.replacen(&masses, "", 1);
            }
            let mut v = magic.to_vec();
            v.extend_from_slice(&(h.len() as u64).to_le_bytes());
            v.extend_from_slice(h.as_bytes());
            v.extend_from_slice(&bytes[16 + hl..]);
            let p = tmp(&format!("ionic-old-{name}.model"));
            std::fs::write(&p, v).unwrap();
            let back = read(&p).unwrap();
            assert_eq!(back.model.provenance.ionic, "none", "{name}");
            assert_eq!(back.model.provenance, orig.model.provenance, "{name}");
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_corrupt_sidecar_is_an_io_error_never_a_panic() {
        let orig = random_sidecar(5, false);
        let path = tmp("corrupt-src.model");
        write(&path, &orig.model, &orig.accumulator, orig.report.as_ref(), &orig.samples, None).unwrap();
        let good = std::fs::read(&path).unwrap();
        let header_len = u64::from_le_bytes(good[8..16].try_into().unwrap()) as usize;
        let header = String::from_utf8(good[16..16 + header_len].to_vec()).unwrap();
        let body = &good[16 + header_len..];
        let with_header = |h: &str| {
            let mut v = MAGIC_V7.to_vec();
            v.extend_from_slice(&(h.len() as u64).to_le_bytes());
            v.extend_from_slice(h.as_bytes());
            v.extend_from_slice(body);
            v
        };
        let mut cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("v2 magic on a v3 file", [MAGIC_V2.as_slice(), &good[8..]].concat()),
            ("v1 magic on a v3 file", [MAGIC.as_slice(), &good[8..]].concat()),
            ("bad magic", [b"ASMODELX".as_slice(), &good[8..]].concat()),
            ("truncated header length", good[..12].to_vec()),
            ("header length past the end", [&good[..8], &(u64::MAX).to_le_bytes()[..], &good[16..]].concat()),
            ("truncated in the arrays", good[..good.len() - 9].to_vec()),
            ("trailing bytes", [good.as_slice(), &[0u8; 3]].concat()),
            ("not JSON", with_header(&header.replacen('{', "[", 1))),
            ("wrong species count", with_header(&header.replacen("\"species\":[1,6,8]", "\"species\":[1,6]", 1))),
            ("unsorted species", with_header(&header.replace("[1,6,8]", "[6,1,8]"))),
            ("body-3 basis", with_header(&header.replacen("\"body3\":false", "\"body3\":true", 1))),
            ("absurd n_max", with_header(&header.replacen("\"n_max\":8", "\"n_max\":4000000000", 1))),
            ("fractional integer", with_header(&header.replacen("\"grid_n\":41", "\"grid_n\":41.5", 1))),
            ("missing key", with_header(&header.replacen("\"lambda\"", "\"lambada\"", 1))),
        ];
        // an array length prefix that disagrees with the header
        let mut bad_len = good.clone();
        bad_len[16 + header_len] ^= 1;
        cases.push(("array length prefix", bad_len));
        // a sample weight of zero (the first sample's weight is the first entry of sample_weight)
        let mut zero_w = good.clone();
        let names: Vec<(String, usize)> = {
            let start = header.find("\"arrays\":").unwrap();
            let list = &header[start + 9..];
            list.split("],[").map(|e| {
                let e = e.trim_matches(|c| c == '[' || c == ']' || c == '}');
                let mut it = e.split(',');
                (it.next().unwrap().trim_matches('"').to_string(), it.next().unwrap().trim().parse().unwrap())
            }).collect()
        };
        let mut off = 16 + header_len;
        for (name, n) in &names {
            if name == "sample_weight" {
                zero_w[off + 8..off + 16].copy_from_slice(&0.0f64.to_le_bytes());
            }
            off += 8 + 8 * n;
        }
        assert_eq!(off, good.len(), "the test's own array walk");
        cases.push(("zero sample weight", zero_w));
        // a newer format version says so instead of "not a model sidecar"
        let p = tmp("newer.model");
        std::fs::write(&p, [b"ASMODEL8".as_slice(), &good[8..]].concat()).unwrap();
        let e = read(&p).unwrap_err();
        assert!(e.to_string().contains("format version 8 is newer than this build"), "{e}");
        // body-3 corruptions: a v1 format string under the v3 magic, an unsupported l_max, a
        // body-2 header over a body-3 body, a terms list that does not match the layout (two
        // swapped)
        let v2 = random_sidecar(6, true);
        let path_v2 = tmp("corrupt-src-v2.model");
        write(&path_v2, &v2.model, &v2.accumulator, None, &[], None).unwrap();
        let good2 = std::fs::read(&path_v2).unwrap();
        let hl2 = u64::from_le_bytes(good2[8..16].try_into().unwrap()) as usize;
        let header2 = String::from_utf8(good2[16..16 + hl2].to_vec()).unwrap();
        let body2 = good2[16 + hl2..].to_vec();
        let with_header2 = |h: &str| {
            let mut v = MAGIC_V7.to_vec();
            v.extend_from_slice(&(h.len() as u64).to_le_bytes());
            v.extend_from_slice(h.as_bytes());
            v.extend_from_slice(&body2);
            v
        };
        assert!(header2.contains("\"format\":\"ASMODEL7\"") && header2.contains("\"body3\":true"));
        cases.push(("v3 file with a v1 format string", with_header2(&header2.replacen("ASMODEL7","ASMODEL1", 1))));
        cases.push(("v2 l_max 5", with_header2(&header2.replacen("\"l_max\":2", "\"l_max\":5", 1))));
        cases.push(("v2 body3 false but a terms array", with_header2(&header2.replacen("\"body3\":true", "\"body3\":false", 1))));
        let mut swapped = good2.clone();
        let n_terms = 5 * v2.model.basis.n_body3();
        let terms_at = good2.len() - 8 * n_terms;
        let (a, b) = (terms_at, terms_at + 5 * 8); // term 0 and term 1
        let (ta, tb) = (good2[a..a + 40].to_vec(), good2[b..b + 40].to_vec());
        swapped[a..a + 40].copy_from_slice(&tb);
        swapped[b..b + 40].copy_from_slice(&ta);
        cases.push(("v2 terms out of order", swapped));
        assert!(read(&path_v2).unwrap().model.basis.body3);
        let _ = std::fs::remove_file(path_v2);
        for (what, bytes) in cases {
            let p = tmp("corrupt.model");
            std::fs::write(&p, &bytes).unwrap();
            let r = std::panic::catch_unwind(|| read(&p));
            match r {
                Ok(Err(e)) => {
                    assert_eq!(e.kind(), io::ErrorKind::InvalidData, "{what}: {e}");
                    println!("{what}: {e}");
                }
                Ok(Ok(_)) => panic!("{what}: a corrupt file was accepted"),
                Err(_) => panic!("{what}: read panicked"),
            }
        }
        assert!(read(&tmp("does-not-exist.model")).is_err());
        let _ = std::fs::remove_file(path);
    }
}
