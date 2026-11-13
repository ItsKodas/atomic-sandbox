//! The quarantine codes of a stored sample — one source of truth for the sidecar's
//! `sample_quarantine` array ([`crate::sidecar`]), the provider's per-sample flags
//! (`providers::surrogate`, whose `Q_*` are aliases of these) and a collection's status records
//! ([`crate::collection`]). Codes 1–4 and 7 are kept but excluded from the fit; what each means, and when
//! the provider sets it, is the provider's (`providers::surrogate` module doc, "Trust recovery and
//! quarantine"). M8a Task 3 moved them here from `providers::surrogate::quarantine` (controller
//! ruling P4), values unchanged.

/// Trained on (the default; a sidecar without a `sample_quarantine` array reads as all zero).
pub const Q_NONE: u8 = 0;

/// Quarantined: a sample the model could not fit, held out pending release or supersede.
pub const Q_HELD: u8 = 1;

/// Superseded by a later answer at (nearly) the same geometry.
pub const Q_SUPERSEDED: u8 = 2;

/// A box-affected sample (its face density above the box rule) taken while the model was still
/// being built, excluded from the fit once the trust floor is met without it (Task 9 fix round
/// 1, S1): never a quarantine candidate, never fencing. Never admitted to a shared collection
/// (spec §4.2: finite-box error must not enter the shared fit).
pub const Q_BOX: u8 = 3;

/// A draft-mode audit's answer at a geometry the strict policy refused (fix (b), 2026-10-18;
/// `SampleTag::DraftAudit`): held out of the strict fit — not fitted, not in the envelope, the
/// grade or the leave-one-out, never fencing. Not written to the sidecar's quarantine array (the
/// tag carries it; the provider's `install_sidecar` restores the code from the tag).
pub const Q_DRAFT: u8 = 4;

/// **Forgotten by the user** (spec `2026-11-12-sample-recovery-design.md` R2: the Model tab's
/// "Worst-fitting samples" › Forget): kept, but held out of the fit, the envelope, the grade and
/// the leave-one-out like a held sample ([`Q_HELD`]), and it **fences** like one (geometries near it
/// fall back to DFT: the safe side, the user said the model should not learn from it) — until the
/// teacher agrees with the model there three times in a row ([`Q_FORGOTTEN_UNFENCED`], 2026-11-14). Unlike a
/// held sample it is **never released** by a refit that reproduces it and never superseded by a
/// fresh answer near it: only the user's Restore clears it. It counts with the held samples against
/// the trust recovery's 10 % cap (forgetting is no way round it). Written as a collection status
/// record and in the sidecar's `sample_quarantine` array (codes 5 and 6 are the provider's own,
/// never written; 7 is the next free persisted code).
///
/// **An older build meeting code 7** (checked against the code before 2026-11-12, `b6bc3ab`):
/// - **a model file** (`sample_quarantine` with a 7): refused as a whole with
///   "sample_quarantine[i] = 7 is not 0, 1, 2 or 3" (its reader accepts 0–3 only) — a clear
///   error, nothing installed; the file is untouched. The same holds for a **session-only file**
///   (the session autosave of collection mode, the same sidecar format) holding a forgotten
///   session-only sample: the older build's restore of it is refused whole.
/// - **a shared collection** (a status record with code 7): read as is (the collection reader
///   keeps any code). The provider then treats the sample as **excluded** (`excluded` is "code ≠
///   0") and **not fencing** (its fence reads code 1 only); it never releases or supersedes it
///   (those touch code 1 only) and never rewrites its status (the code it loaded is the code it
///   holds, so no status record differs). Its trust recovery's 10 % cap does **not** count it (that
///   build counts code 1 only). The Activity view shows it as "superseded" (the catch-all for a
///   code it does not know) and its README does not count it. Its "Save a specialised model…"
///   leaves it out (that writer keeps code-0 samples only and writes no quarantine array). No
///   path of that build writes a 7. So an older build never fits a forgotten sample and never
///   brings it back; it only lets the model answer near it (no fence).
/// - A newer build reads every older file and collection as before: code 7 never appears in them.
pub const Q_FORGOTTEN: u8 = 7;

/// **Forgotten, its fence lifted** (2026-11-14, user-approved trust change): a [`Q_FORGOTTEN`]
/// sample whose neighbourhood earned the model back — three mutually consistent fresh
/// teacher answers near it agreed with the model (the provider's fence voting, `providers::surrogate`
/// "Lifting a forgotten sample's fence"). Still forgotten: out of the fit, the envelope, the grade
/// and the leave-one-out, never released or superseded, listed for Restore (which returns it to
/// [`Q_NONE`], clearing the lift) and purged by a purge — but it **no longer fences**. Persisted like
/// code 7 (a collection status record, the sidecar's `sample_quarantine`); a file or collection
/// written before it holds no 8, so every older file reads as "not lifted".
///
/// **An older build meeting code 8**: a model file holding one is refused whole by every earlier
/// build ("sample_quarantine[i] = 8 is not 0, 1, 2, 3 or 7"; before 2026-11-12, "… 0, 1, 2 or 3");
/// a shared collection's status record 8 is read as is (the reader keeps any code): that build holds
/// the sample out of its fit (code ≠ 0), lets the model answer near it (its fence reads 1 and 7),
/// never releases, supersedes or rewrites it — but does not list it as forgotten (it shows under its
/// catch-all), so its Restore is this build's.
pub const Q_FORGOTTEN_UNFENCED: u8 = 8;

/// Whether `code` marks a sample the user forgot ([`Q_FORGOTTEN`], or [`Q_FORGOTTEN_UNFENCED`] once
/// its fence was lifted): what Restore lists and a purge deletes (`crate::collection`, "Purging
/// forgotten samples").
pub fn is_forgotten_code(code: u8) -> bool {
    code == Q_FORGOTTEN || code == Q_FORGOTTEN_UNFENCED
}

/// Whether `code` is one a file or a collection may hold ([`Q_NONE`]…[`Q_BOX`], [`Q_FORGOTTEN`] and
/// [`Q_FORGOTTEN_UNFENCED`]; [`Q_DRAFT`] is carried by the sample's tag, 5 and 6 are the provider's
/// own).
pub fn is_persisted_code(code: u8) -> bool {
    matches!(code, Q_NONE | Q_HELD | Q_SUPERSEDED | Q_BOX | Q_FORGOTTEN | Q_FORGOTTEN_UNFENCED)
}
