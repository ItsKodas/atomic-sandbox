//! The quarantine codes of a stored sample — one source of truth for the sidecar's
//! `sample_quarantine` array ([`crate::sidecar`]), the provider's per-sample flags
//! (`providers::surrogate`, whose `Q_*` are aliases of these) and a collection's status records
//! ([`crate::collection`]). Codes 1–4 are kept but excluded from the fit; what each means, and when
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
