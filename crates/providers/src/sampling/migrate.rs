//! The collection migration's re-solve pass (Fermi–Dirac spec
//! `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md` §7.3–7.5; plan 2026-11-13 Task 14): every sample of
//! the **predecessor** folder (the same solver surface under the previous solver's ionic tag, `predecessor_of`) is
//! solved again, cold, by the new teacher, compared with its old label, classified, and appended to the attached
//! collection. A pass kind of the existing machinery ([`Sampler::migration`]): one solve per `step`, so it shares the GPU
//! with the live scene, pauses, resumes and cancels like any pass. Atomic units throughout (Hartree, Bohr, Ha/Bohr).
//!
//! # What is compared (spec §7.4)
//!
//! The old label is (E_old, F_old): E_old the energy the previous solver stored — its total energy, with integer or
//! equal-share ("degenerate shells") occupations, so no entropy term — and F_old_{Ia} = −∂E_old/∂R_{Ia} its forces. The
//! new cold solve reports the Mermin free energy F = E − TS (`ForceResult::energy` = `EvalStats::energy.total`), its
//! internal energy E (`EvalStats::energy.internal`; TS = E − F ≥ 0) and its forces −∂F/∂R. With ε_E (Ha per sample)
//! and ε_F (Ha/Bohr per component) from the teacher's noise model ([`migration_eps`], "The thresholds" below) and
//! ΔF_{Ia} = F_new,{Ia} − F_old,{Ia}:
//!
//! - **U, unchanged:** |F − E_old| ≤ ε_E and max_{Ia} |ΔF_{Ia}| ≤ ε_F. A gapped state: TS ≈ 0, the same label within
//!   solver noise. The old record is kept bit for bit.
//! - **T, entropy only:** not U, |E − E_old| ≤ ε_E (the **internal** energy) and max |ΔF_{Ia}| ≤ ε_F. The same
//!   electronic state (an equal-share ensemble before, the same occupations now); only the reported quantity moved, by
//!   −TS (−2 kT ln 2 = −6.93 mHa for linear OH's ↓ π pair at kT = 5 mHa). The new F and forces are the label.
//! - **S, state changed:** anything else — the old rule had a different state at this geometry (a released integer
//!   pattern where Fermi–Dirac pins the frontier at μ, or another SCF branch). The new label is appended at code 9
//!   ([`migrated_code`]) until the review (spec §7.6). It may replace a **lower-F** label: the near-square H₄ of the V7
//!   fixture (aspect 1.005) had E_DS = −2.028985 Ha in the broken-symmetry integer state the shared-shell rule released
//!   into, while the cold Fermi–Dirac solve reaches the symmetric ensemble, F = −2.014486 Ha (14.5 mHa above; spec
//!   §4.7(e), CLAUDE.md rule 7): the new label is the state the new teacher converges to, not the lowest one.
//!
//! Every comparison is `|x| ≤ ε` and fails for NaN, so a non-finite number is class S, never a carried label. Masked
//! samples (capped segments) are compared on **all** atoms (conservative); their mask and energy-not-fitted flag are
//! carried. A sample's class never reads its old code.
//!
//! # The thresholds
//!
//! Per sample, from the per-solve path scatter the `branch_jump` guard already uses (`crate::branch_jump::noise_tier`:
//! by surface — pseudo or all-electron —, by the tolerance the new solve stopped at, `EvalStats::scf_tol_used`, and
//! with σ_E = max(the tier's σ_E, c·|E_old|), c = 9e-8 pseudo, 1.0e-7 all-electron; Task 14 fix round 1, physics
//! review I2):
//!
//! - ε_E = max([`MIGRATION_EPS_E`] = 1e-5 Ha, [`MIGRATION_EPS_E_SIGMAS`] = 5 × σ_E). The floor is 5× the pseudo sample
//!   tier's 1.87e-6 Ha (pseudo-H₂O warm vs cold, `validate/references.md` "SCF, GPU against CPU"), the value spec §7.4
//!   chose for small molecules. Pseudo (sample tier or partial stop): 1e-5 up to |E| ≈ 22 Ha, then 5c|E| —
//!   35 Ha → 1.58e-5, 70 Ha → 3.15e-5. All-electron: the tier's own σ_E is already above the floor — 2.7e-5 at the
//!   sample tier up to |E| = 54 Ha (2.35e-5 at the app tier up to 47 Ha), then 5c|E| — 70 Ha → 3.5e-5.
//! - ε_F = [`MIGRATION_EPS_F_SIGMAS`] = 3 × σ_F: pseudo sample tier 1.5e-4 Ha/Bohr ([`MIGRATION_EPS_F`], 3 ×
//!   `PSEUDO_FORCE_NOISE`), pseudo partial stop (every GPU solve with a partly occupied level stops at
//!   `qm::GPU_PARTIAL_TOL` = 1e-5: the app tier) 2.7e-4; all-electron 7.2e-5 at either tier — the larger σ_F of its two
//!   tiers, because the all-electron app tier's 1.4e-5 was measured on gapped H₂O only and a partial-level stop on that
//!   surface is unmeasured (fix round 2).
//! - Derived from the documented noise model, not chosen to make a case pass. A fixed 1e-5 was 1.6σ_E at 70 Ha (about a
//!   quarter of gapped samples falsely S) and below 2σ_F for a partial-stop solve.
//! - Both errors are safe: a false S costs a review (and, for a misfit hold or a lifted fence, re-judges it); a false U
//!   keeps a label within ε_E of the new one. The classes decide review only: a T or S label is always the new solve's.
//!
//! # Per sample, in file order (spec §7.3)
//!
//! 1. Checks that need no solve: a status code a collection holds, a shared (unforced) multiplicity, Z in the isotope
//!    table, a neighbour within the model's cutoff (a lone-atom sample teaches nothing; an import refuses it too).
//! 2. The **cold** solve (`reset` before it): the attached key's h, σ and the teacher's tolerance, the sample's own box
//!    (`grid_n`) and its stored positions bit for bit, multiplicity auto, the key's ionic policy; the scene's surface key
//!    must be the attached collection's. A cold GPU solve is bit-reproducible (`sim::references`), so the outcome does
//!    not depend on the order or on a resume.
//! 3. The solve must have converged (a stall or the cap is an error), carry no solver warning and nothing
//!    untrainable (overlapping cores, a branch jump), report a face density within the box rule
//!    (`box_affected_face_density(h)`), and reproduce the recorded multiplicity when one is recorded — a consistency
//!    check of the settings only: under multiplicity auto the electron count's parity fixes it, so it says nothing about
//!    the electronic state (physics review M1; the class does). A sample that fails is **not migrated**: listed in `migration.txt` with its reason, never appended — its old label belongs to
//!    the old surface. A device error ("GPU busy") is not the sample's: the cursor stays and the provider retries.
//! 4. Classify, then append (spec §7.5): U the old record as it was, plus a status record when its code is not 0; T the
//!    new label with the old metadata and the old code; S the new label with [`migrated_code`]. An append refused as a
//!    duplicate (`REFUSE_DUPLICATE`: the same answer is stored — a crash between the append and the cursor write) is
//!    done: its status is written if it is missing.
//!    **The status window** (fix round 1, I1): a status record can only follow its sample record, so a crash between
//!    the two would leave a sample with a non-zero code (any S, a U or T of a held, forgotten or superseded sample) at
//!    0 — in the fit. So before such an append the log records `pending <index> <code>` with the sample's identity (its
//!    energy's bits and `surrogate::migration::answer_print`; through the temporary file); the next attach applies it,
//!    to that sample only, before anything is fitted (`SurrogateKs::find_migration`), and the cursor write clears it.
//!    An append that fails before writing anything drops it at once (fix round 2): it names no record.
//! 5. The cursor is rewritten (`surrogate::write_migration`, through a temporary file).
//!
//! **Spin** (M1): the cold solve's ∫|ρ↑ − ρ↓| is not on its `ForceResult` (`EvalStats::spin_change` is a difference
//! between two judged evaluations, and a cold solve after `reset` is not judged), so it is not logged for S samples.
//!
//! **Superseded samples** (M4): a migrated code-2 sample stays 2. The collection does not record which answer
//! superseded it; the pass takes the nearest later source sample of the same atoms (rms displacement) as its superseder,
//! and when that one was not migrated — or there is none — lists the superseded sample in the log (`orphaned`), so
//! that geometry is seen to have no label in the fit.
//!
//! **References** come first: every (species, box) of the old `references.txt` is solved by the multiplicity scan
//! (lowest F; `sim::references::AtomScan`) and offered through `Collection::offer_reference`; the log keeps the old and
//! the new energy. A scan with a multiplicity that did not solve in the end is not offered (its lowest solved F need not
//! be the lowest; M2): logged without a new energy, and retried by a resume, which skips only references logged with one.
//!
//! **The source fingerprint (F3).** The pass records the source `samples.bin`'s length and hash when it starts. A resume
//! whose source differs (purged, or appended to by an older build since) starts again from sample 0 with the counts
//! reset; what was appended before is refused as a duplicate and counted as done, so the counts are an uninterrupted
//! run's on the changed source.

use super::*;
use crate::surrogate::PSEUDO_FORCE_NOISE;
use sim::IonicPolicy;
use std::path::{Path, PathBuf};
use surrogate::collection::CollectionReference;
use surrogate::migration::MIGRATION_FILE;
use surrogate::quarantine::{is_collection_code, Q_FORGOTTEN, Q_FORGOTTEN_UNFENCED, Q_RESOLVED, Q_SUPERSEDED};
use surrogate::{Collection, CollectionSample, MigrationLog, MigrationState, NotMigrated, SurfaceKey};

/// ε_E's floor (module doc, "The thresholds"): Hartree per sample.
pub const MIGRATION_EPS_E: f64 = 1e-5;
/// ε_E = max([`MIGRATION_EPS_E`], this × σ_E) (module doc, "The thresholds").
pub const MIGRATION_EPS_E_SIGMAS: f64 = 5.0;
/// ε_F = this × σ_F (module doc, "The thresholds").
pub const MIGRATION_EPS_F_SIGMAS: f64 = 3.0;
/// ε_F on the pseudo sample tier: 3 × `PSEUDO_FORCE_NOISE` = 1.5e-4 Ha/Bohr (module doc, "The thresholds").
pub const MIGRATION_EPS_F: f64 = MIGRATION_EPS_F_SIGMAS * PSEUDO_FORCE_NOISE;

/// A sample's thresholds (module doc, "The thresholds"): ε_E (Ha) and ε_F (Ha/Bohr per component).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MigrationEps {
    pub e: f64,
    pub f: f64,
}

/// The thresholds of a sample whose old energy is `e_old` (Ha), re-solved on the all-electron surface or not, by a solve
/// that stopped at `scf_tol_used` (module doc, "The thresholds"; `crate::branch_jump::noise_tier`).
pub fn migration_eps(all_electron: bool, scf_tol_used: f64, e_old: f64) -> MigrationEps {
    use crate::branch_jump::{ALL_ELECTRON_APP_TIER, ALL_ELECTRON_SAMPLE_TIER};
    let t = crate::branch_jump::noise_tier(all_electron, scf_tol_used, e_old);
    // the all-electron app tier's σ_F was measured on gapped H₂O only; a partial-level stop there is unmeasured, so the
    // all-electron surface takes the larger σ_F of its two tiers (fix round 2)
    let sigma_f = if all_electron { ALL_ELECTRON_SAMPLE_TIER.sigma_f.max(ALL_ELECTRON_APP_TIER.sigma_f) } else { t.sigma_f };
    MigrationEps { e: MIGRATION_EPS_E.max(MIGRATION_EPS_E_SIGMAS * t.sigma_e), f: MIGRATION_EPS_F_SIGMAS * sigma_f }
}
/// The time estimate uses the mean of the pass's own sample solves, per [`prior_solve_s`]'s grid unit, once it has made
/// this many (spec §7.3, "Cost").
pub const MIGRATION_ETA_SOLVES: usize = 20;
/// The a-priori cost of one cold sample solve (s) per [`MIGRATION_PRIOR_GRID_N`]³ of box on the RX 7900 XTX, scaled by
/// (n/57)³ for another box: **4.1 s**, the mean of V11's 480 cold sample re-solves of a real Accurate collection (plan
/// Task 17, 2026-11-20: 17 915 s for Σ(n/57)³ = 4 350 at 109–127³, ≈ 41 s per sample at 123³; `validate/references.md`
/// "V11"). It was 1.1 s — one gapped H₂O solve at 57³ — which put a 1 657-sample collection at 4.5 h against ≈ 17 h
/// at the measured rate: stretched, hot and open-shell samples cost far more than one equilibrium molecule (the measured per-unit cost
/// ranged 0.28–14 s over blocks of 100). An estimate for the offer, replaced by the pass's own measured mean after
/// [`MIGRATION_ETA_SOLVES`] solves.
pub const MIGRATION_PRIOR_SOLVE_S: f64 = 4.1;
/// The box of [`MIGRATION_PRIOR_SOLVE_S`] (grid points per side).
pub const MIGRATION_PRIOR_GRID_N: usize = 57;
/// The pass's progress stage while it re-solves samples (`SamplingProgress::stage`).
pub const MIGRATION_STAGE: &str = "re-solving";

/// A migrated sample's class (module doc, "What is compared").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationClass {
    /// U: the same label within solver noise.
    Unchanged,
    /// T: the same state; the label moved by −TS only.
    EntropyOnly,
    /// S: another state.
    StateChanged,
}

impl MigrationClass {
    /// Its slot in `MigrationLog::counts` (U, T, S; 3 is "not migrated").
    pub fn index(self) -> usize {
        match self {
            MigrationClass::Unchanged => 0,
            MigrationClass::EntropyOnly => 1,
            MigrationClass::StateChanged => 2,
        }
    }
}

/// The largest |ΔF_{Ia}| between two force sets (Ha/Bohr); NaN — never within a bound — when the shapes differ or a
/// component is not finite.
fn max_force_diff(old_f: &[[f64; 3]], new_f: &[[f64; 3]]) -> f64 {
    if old_f.len() != new_f.len() {
        return f64::NAN;
    }
    let mut m = 0.0f64;
    for (a, b) in old_f.iter().zip(new_f) {
        for k in 0..3 {
            let d = (b[k] - a[k]).abs();
            if !d.is_finite() {
                return f64::NAN;
            }
            m = m.max(d);
        }
    }
    m
}

/// The class of a re-solved sample (module doc, "What is compared"): `old_e`, `old_f` the stored label (Ha, Ha/Bohr),
/// `new_free` the new solve's F, `new_internal` its E, `new_f` its forces −∂F/∂R, `eps` its thresholds ([`migration_eps`]).
pub fn classify(old_e: f64, old_f: &[[f64; 3]], new_free: f64, new_internal: f64, new_f: &[[f64; 3]], eps: MigrationEps) -> MigrationClass {
    let forces_within = max_force_diff(old_f, new_f) <= eps.f;
    if (new_free - old_e).abs() <= eps.e && forces_within {
        MigrationClass::Unchanged
    } else if (new_internal - old_e).abs() <= eps.e && forces_within {
        MigrationClass::EntropyOnly
    } else {
        MigrationClass::StateChanged
    }
}

/// The status code a migrated sample takes in the new collection (spec §7.5): U and T keep `old` (it judged the same
/// label: a forgotten-unfenced 8 stays 8); S takes [`Q_RESOLVED`] (9, awaiting review), except a forgotten sample, which
/// stays forgotten — 7 stays 7 and 8 returns to 7 (the user's Forget survives, the fence returns: the lift judged the old
/// label) — and a superseded one, which stays 2 (what superseded it is migrated too). A held sample (1) goes to review.
pub fn migrated_code(old: u8, class: MigrationClass) -> u8 {
    match class {
        MigrationClass::Unchanged | MigrationClass::EntropyOnly => old,
        MigrationClass::StateChanged => match old {
            Q_FORGOTTEN | Q_FORGOTTEN_UNFENCED => Q_FORGOTTEN,
            Q_SUPERSEDED => Q_SUPERSEDED,
            _ => Q_RESOLVED,
        },
    }
}

/// What the provider did with a migrated sample (`SurrogateKs::take_migrated`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigratedAppend {
    /// Appended at this index of the new collection.
    Appended(usize),
    /// The same answer was stored already, at this index (module doc, 4.): counted as done.
    Duplicate(usize),
}

/// The a-priori time of one cold solve on an n³ grid (s; [`MIGRATION_PRIOR_SOLVE_S`]).
pub fn prior_solve_s(n: usize) -> f64 {
    MIGRATION_PRIOR_SOLVE_S * (n as f64 / MIGRATION_PRIOR_GRID_N as f64).powi(3)
}

/// Today's date, YYYY-MM-DD (UTC), from the system clock (days since the epoch → the civil date; Howard Hinnant's
/// algorithm, no date crate).
fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// One reference of the source to re-solve: its (Z, n), the old energy, and the scan once started.
struct RefJob {
    z: u8,
    n: usize,
    old: f64,
    scan: Option<AtomScan>,
}

/// The pass (module doc): the source as read when it started, the log it keeps, and its progress.
pub struct Migrator {
    /// The new folder (where `migration.txt` is written).
    dir: PathBuf,
    source_samples: Vec<CollectionSample>,
    source_codes: Vec<u8>,
    refs: VecDeque<RefJob>,
    log: MigrationLog,
    /// The attached key's settings and the ionic policy its tag is.
    h: f64,
    sigma: f64,
    policy: IonicPolicy,
    /// The key's tolerance (the stop a solve that reports none is taken to have used) and its surface.
    tol: f64,
    all_electron: bool,
    /// The model's cutoff (Bohr): a sample without a neighbour within it is not migrated (module doc, 1.).
    r_cut: f64,
    /// This run's sample solves' wall times (s), each per [`prior_solve_s`]'s grid unit (the estimate's mean).
    walls: Vec<f64>,
    progress: SamplingProgress,
    notes: Vec<String>,
    /// Cold solves refused (`sim::answer_refusal`) since the last accepted one: the pass ends at
    /// `super::PASS_REFUSALS_IN_A_ROW` (gpu-fault review A-I2), before marking that sample.
    refused_in_a_row: usize,
    /// Test hook (Review Focus 4, fix round 1 I1): sample `k`'s append stops with an error right after its sample
    /// record, before its status record and the cursor are written — a crash in the status window.
    #[cfg(test)]
    pub(crate) crash_in_window: Option<usize>,
    /// Test hook (fix round 2): sample `k`'s append fails before anything is written.
    #[cfg(test)]
    pub(crate) fail_append: Option<usize>,
}

impl Migrator {
    /// Starts the pass from `source` into the collection folder `dir` of `key`, whose ionic tag is the policy `policy`;
    /// `existing` is the folder's `migration.txt` when it has one (a resume). `Err` with the reason, nothing written:
    /// the source cannot be read (a live writer holds it — spec §7.1 —, a newer format, not a collection), or the log
    /// says the migration is complete. A fresh start (or a source whose fingerprint moved, F3) writes the log at once
    /// with its fingerprint, and copies `bond_experience.txt` when the new folder has none (spec §7.5).
    pub fn start(source: &Path, dir: &Path, key: &SurfaceKey, policy: IonicPolicy, r_cut: f64, existing: Option<MigrationLog>) -> Result<Migrator, String> {
        let ro = Collection::open_read_only(source).map_err(|e| e.to_string())?;
        let slug = source.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let mut notes = Vec::new();
        if let Some(why) = &ro.damaged_tail {
            notes.push(format!("re-solving {slug}: {why}"));
        }
        let fresh = || MigrationLog {
            source_slug: slug.clone(),
            source_key: ro.key.to_text(),
            build: env!("CARGO_PKG_VERSION").to_string(),
            date: today(),
            source_len: ro.source_len,
            source_hash: ro.source_hash,
            cursor: 0,
            state: MigrationState::Running,
            counts: [0; 4],
            not_migrated: Vec::new(),
            references: Vec::new(),
            class_u_max: (0.0, 0.0),
            review: Vec::new(),
            waiting: Vec::new(),
            pending: None,
            orphaned: Vec::new(),
        };
        let (log, new_start) = match existing {
            Some(l) if l.state != MigrationState::Running => return Err(format!("the collection was already re-solved from {} ({})", l.source_slug, if l.state == MigrationState::Complete { "complete" } else { "reviewed" })),
            Some(l) if (l.source_len, l.source_hash) != (ro.source_len, ro.source_hash) => {
                notes.push(format!(
                    "re-solving {slug}: its samples file changed since the pass started ({} → {} bytes; purged, or appended to by an older build) — the pass starts again from its first sample, and the samples already re-solved are counted as done",
                    l.source_len, ro.source_len
                ));
                (fresh(), true)
            }
            Some(l) => (l, false),
            None => (fresh(), true),
        };
        ::surrogate::write_migration(dir, &log)?;
        if new_start {
            let (from, to) = (source.join(BOND_EXPERIENCE_FILE), dir.join(BOND_EXPERIENCE_FILE));
            if from.exists() && !to.exists() {
                if let Err(e) = std::fs::copy(&from, &to) {
                    notes.push(format!("re-solving {slug}: {BOND_EXPERIENCE_FILE} was not copied ({e}) — it is display only"));
                }
            }
        }
        // a reference logged without a new energy did not solve: a resume tries it again (M2)
        let done: Vec<(u8, usize)> = log.references.iter().filter(|r| r.3.is_some()).map(|r| (r.0, r.1)).collect();
        let refs: VecDeque<RefJob> = ro.references.iter().filter(|(z, n, _)| !done.contains(&(*z, *n))).map(|&(z, n, r): &(u8, usize, CollectionReference)| RefJob { z, n, old: r.energy, scan: None }).collect();
        let ref_solves: usize = refs.iter().map(|r| allowed_multiplicities(r.z as u32).len()).sum();
        let total = ref_solves + ro.samples.len();
        let done = log.cursor.min(ro.samples.len());
        let stage = if !refs.is_empty() { "references" } else { MIGRATION_STAGE };
        let progress = SamplingProgress { done, total, stage, target: None, last_energy: f64::NAN, cancelled: false, skipped: 0, failed: log.counts[3], finished: false, paused: false };
        Ok(Migrator {
            dir: dir.to_path_buf(),
            source_samples: ro.samples,
            source_codes: ro.codes,
            refs,
            log,
            h: key.h,
            sigma: key.sigma,
            policy,
            tol: key.tol,
            all_electron: crate::pseudo::is_all_electron_tag(&key.ionic),
            r_cut,
            walls: Vec::new(),
            progress,
            notes,
            refused_in_a_row: 0,
            #[cfg(test)]
            crash_in_window: None,
            #[cfg(test)]
            fail_append: None,
        })
    }

    /// The log as it stands (written after every sample).
    pub fn log(&self) -> &MigrationLog {
        &self.log
    }

    /// Samples in the source.
    pub fn source_samples(&self) -> usize {
        self.source_samples.len()
    }

    pub fn progress(&self) -> SamplingProgress {
        self.progress.clone()
    }

    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    /// Ends the pass early (Cancel): the log stays at Running with its cursor, so a later start resumes there.
    pub fn cancel(&mut self) {
        if !self.progress.finished {
            self.progress.cancelled = true;
            self.progress.finished = true;
            self.progress.stage = "cancelled";
        }
    }

    /// The time left (s; spec §7.3 "Cost"): the a-priori [`prior_solve_s`] of every solve left, or — once the pass has
    /// made [`MIGRATION_ETA_SOLVES`] sample solves — the same sum with the pass's own **mean** cost per grid unit in place
    /// of [`MIGRATION_PRIOR_SOLVE_S`]. The mean, not the median (plan Task 17, V11): the time left is a sum, and the solve
    /// times are heavy-tailed — V11's median was 2.1 s while its mean was 37 s per sample, so a median put 1 257 samples
    /// at 45 min that were taking ≈ 13 h.
    pub fn eta_s(&self) -> f64 {
        let left = &self.source_samples[self.log.cursor.min(self.source_samples.len())..];
        let ref_solves = |r: &RefJob| r.scan.as_ref().map_or(allowed_multiplicities(r.z as u32).len(), |s| s.remaining().len());
        let per_unit = if self.walls.len() >= MIGRATION_ETA_SOLVES { self.walls.iter().sum::<f64>() / self.walls.len() as f64 / MIGRATION_PRIOR_SOLVE_S } else { 1.0 };
        per_unit * (left.iter().map(|s| prior_solve_s(s.grid_n)).sum::<f64>() + self.refs.iter().map(|r| ref_solves(r) as f64 * prior_solve_s(r.n)).sum::<f64>())
    }

    /// The isolated neutral atom's scene parts of a reference job, and the cold scan.
    fn scan_for(&self, z: u8, n: usize) -> Option<AtomScan> {
        let a = units::most_abundant_a(z)?;
        let mass = sim::Nucleus::isotope(z, a, [0.0; 3]).ok()?.mass;
        Some(AtomScan::new(&sim::Cell { n, h: self.h }, Some(self.sigma), self.policy, z, a, mass))
    }

    /// The cold solve's scene of a source sample (module doc, 2.): `Err` with the reason when a Z has no isotope.
    fn scene_of(&self, s: &CollectionSample) -> Result<Scene, String> {
        let mut nuclei = Vec::with_capacity(s.sample.z.len());
        for (&z, &pos) in s.sample.z.iter().zip(&s.sample.pos) {
            let a = units::most_abundant_a(z).ok_or_else(|| format!("Z = {z} has no isotope in the table"))?;
            nuclei.push(sim::Nucleus::isotope(z, a, pos)?);
        }
        Ok(Scene { nuclei, cell: sim::Cell { n: s.grid_n, h: self.h }, charge: 0, multiplicity: None, sigma: Some(self.sigma), external_field: [0.0; 3], ionic: self.policy, train_mask: None })
    }

    fn write_log(&self) -> Result<(), ProviderError> {
        ::surrogate::write_migration(&self.dir, &self.log).map_err(|e| ProviderError::Invalid(format!("re-solving: {MIGRATION_FILE} could not be written ({e}) — the pass stops; it resumes from sample {}", self.log.cursor)))
    }

    /// Sample `i` is not migrated, for the reason `code` in the words `why` (module doc, 3.): listed, counted, the cursor
    /// moved on.
    fn not_migrated(&mut self, i: usize, code: NotMigrated, why: String) -> Result<Option<SamplingProgress>, ProviderError> {
        self.notes.push(format!("re-solving {}: sample {i} not migrated ({why}) — it stays in the old collection", self.log.source_slug));
        self.log.not_migrated.push((i, code, why));
        self.log.counts[3] += 1;
        self.log.cursor = i + 1;
        self.write_log()?;
        self.progress.done += 1;
        self.progress.failed += 1;
        Ok(Some(self.progress()))
    }

    /// One solve of the reference at the queue's head (module doc, "References"): when its scan is done, the lowest F is
    /// offered to the collection (`SurrogateKs::set_reference_solved`) and logged with the old energy.
    fn reference_step(&mut self, mut teacher: Option<&mut dyn ForceProvider>, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        let (z, n) = self.refs.front().map(|j| (j.z, j.n)).expect("a reference job");
        if self.refs.front().is_some_and(|j| j.scan.is_none()) {
            match self.scan_for(z, n) {
                Some(scan) => self.refs.front_mut().expect("a reference job").scan = Some(scan),
                None => {
                    let job = self.refs.pop_front().expect("a reference job");
                    self.log.references.retain(|r| (r.0, r.1) != (z, n));
                    self.log.references.push((z, n, Some(job.old), None));
                    self.notes.push(format!("re-solving: the reference of Z = {z} has no isotope in the table — not re-solved"));
                    self.write_log()?;
                    return Ok(Some(self.progress()));
                }
            }
        }
        let scan = self.refs.front_mut().and_then(|j| j.scan.as_mut()).expect("a started scan");
        let outcome = {
            let t: &mut dyn ForceProvider = match &mut teacher {
                Some(t) => &mut **t,
                None => surrogate.teacher_mut(),
            };
            scan.step(t, should_abort)
        };
        if let Err(ProviderError::Cancelled) = outcome {
            return Err(ProviderError::Cancelled);
        }
        self.progress.done += 1;
        if !scan.is_done() {
            return Ok(Some(self.progress()));
        }
        let job = self.refs.pop_front().expect("a reference job");
        let scan = job.scan.expect("a started scan");
        // the multiplicities that did not solve in the end (a device loss retried successfully is solved)
        let solved: Vec<u32> = scan.tried().map(|t| t.0).collect();
        let failed: Vec<String> = scan.failed().iter().filter(|(m, _)| !solved.contains(m)).map(|(m, e)| format!("M = {m}: {e}")).collect();
        let tol = surrogate.collection().map_or(f64::NAN, |c| c.key.tol);
        let name = surrogate.teacher_mut().name().to_string();
        self.log.references.retain(|r| (r.0, r.1) != (z, n));
        // a scan with an unsolved multiplicity is not offered: its lowest solved F need not be the lowest (M2)
        let best = if failed.is_empty() { scan.finish(tol, &name) } else { None };
        match best {
            Some(r) => {
                surrogate.set_reference_solved(z, r.energy, n, r.face_density);
                self.log.references.push((z, n, Some(job.old), Some(r.energy)));
                self.notes.push(format!("re-solving: reference {} (Z = {z}) of the {n}³ box, {:.6} Ha before, {:.6} Ha now (multiplicity {})", symbol(z), job.old, r.energy, r.multiplicity.unwrap_or(0)));
            }
            None => {
                self.log.references.push((z, n, Some(job.old), None));
                self.notes.push(format!("re-solving: reference {} (Z = {z}) of the {n}³ box was not taken ({}) — not offered: the lowest of the solved multiplicities need not be the lowest; a resume tries it again, and its samples wait for a reference", symbol(z), if failed.is_empty() { "nothing solved".to_string() } else { failed.join("; ") }));
            }
        }
        self.write_log()?;
        Ok(Some(self.progress()))
    }

    /// At most one solve (module doc): `Ok(Some(progress))` after it, `Ok(None)` when the pass is over (complete or
    /// cancelled), `Err(Cancelled)` when the solve was abandoned (the job stays), any other `Err` ends the pass with the
    /// log at its cursor (a device error the provider retries first).
    pub fn step(&mut self, mut teacher: Option<&mut dyn ForceProvider>, surrogate: &mut SurrogateKs, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<SamplingProgress>, ProviderError> {
        if self.progress.finished {
            return Ok(None);
        }
        // 0. the references (module doc)
        if !self.refs.is_empty() {
            self.progress.stage = "references";
            return self.reference_step(teacher, surrogate, should_abort);
        }
        self.progress.stage = MIGRATION_STAGE;
        // the end
        let i = self.log.cursor;
        if i >= self.source_samples.len() {
            self.log.orphaned = self.orphaned_superseded();
            for &k in &self.log.orphaned {
                self.notes.push(format!("re-solving {}: superseded sample {k} keeps code 2, but what superseded it was not migrated — that geometry has no label in the fit", self.log.source_slug));
            }
            self.log.state = MigrationState::Complete;
            self.write_log()?;
            self.progress.finished = true;
            self.progress.stage = "done";
            return Ok(None);
        }
        let old = self.source_samples[i].clone();
        let code = self.source_codes.get(i).copied().unwrap_or(0);
        // 1. no solve needed to refuse
        if !is_collection_code(code) {
            return self.not_migrated(i, NotMigrated::StatusCode, format!("its status code {code} is not one a collection holds"));
        }
        if old.multiplicity_forced {
            return self.not_migrated(i, NotMigrated::ForcedMultiplicity, "its scene forced the multiplicity (never shared)".into());
        }
        if no_neighbours(&old.sample.pos, self.r_cut) {
            return self.not_migrated(i, NotMigrated::Lone, ::surrogate::collection::REFUSE_LONE.into());
        }
        let scene = match self.scene_of(&old) {
            Ok(s) => s,
            Err(why) => return self.not_migrated(i, NotMigrated::NoScene, why),
        };
        let want = surrogate.collection().map(|c| c.key.clone());
        let got = surrogate.provenance_for(&scene).surface_key();
        if want.as_ref() != Some(&got) {
            let d = want.as_ref().and_then(|w| w.difference(&got)).unwrap_or_else(|| "no collection is attached".into());
            return self.not_migrated(i, NotMigrated::OtherSurface, format!("its scene is not on the collection's surface ({d})"));
        }
        // 2. the cold solve
        let started = Instant::now();
        let r = {
            let t: &mut dyn ForceProvider = match &mut teacher {
                Some(t) => &mut **t,
                None => surrogate.teacher_mut(),
            };
            t.reset();
            t.evaluate(&scene, should_abort)
        };
        let r = match r {
            Err(ProviderError::Cancelled) => return Err(ProviderError::Cancelled),
            Err(e @ ProviderError::Solver(sim::SolverError::Device(_))) => return Err(e),
            // the session's GPU failure ends the migration (resumable): marking every later sample "not migrated" on
            // a dead device would be a fast loop through the collection (the gpu-fault fix)
            Err(e @ ProviderError::Solver(sim::SolverError::GpuFailed(_))) => {
                surrogate.note_teacher_error(&e);
                return Err(e);
            }
            Err(e) => {
                let code = match &e {
                    ProviderError::Solver(sim::SolverError::ScfStalled { .. }) => NotMigrated::Stalled,
                    ProviderError::Solver(sim::SolverError::ScfNotConverged(..)) => NotMigrated::NotConverged,
                    _ => NotMigrated::SolveFailed,
                };
                return self.not_migrated(i, code, format!("the cold solve failed: {e}"));
            }
            Ok(r) => r,
        };
        // per grid unit: a measured solve of a 123³ box predicts a 109³ one by the same (n/57)³ as the prior (`eta_s`)
        self.walls.push(started.elapsed().as_secs_f64() * MIGRATION_PRIOR_SOLVE_S / prior_solve_s(old.grid_n));
        self.progress.last_energy = r.energy;
        // 3. the checks; refused answers in a row end the pass before the last is marked (module doc of `sampling`,
        // "Refused answers"): a device returning bad data would otherwise mark the rest of the collection not migrated,
        // and the resume re-solves the sample at the cursor
        if let Some(why) = sim::answer_refusal(&r) {
            self.refused_in_a_row += 1;
            if self.refused_in_a_row >= super::PASS_REFUSALS_IN_A_ROW {
                self.refused_in_a_row = 0;
                return Err(ProviderError::Invalid(format!(
                    "re-solving stopped after {} cold solves in a row failed the answer check (the last: {why}) — it resumes from sample {i}. The GPU may be near its memory limit: lower the quality or GPU memory (Scene ▸ Quality) before resuming; if it happens again, restart the app",
                    super::PASS_REFUSALS_IN_A_ROW
                )));
            }
        } else {
            self.refused_in_a_row = 0;
        }
        if let Err((code, why)) = check_solve(&r, old.sample.z.len(), self.h, old.multiplicity) {
            return self.not_migrated(i, code, why);
        }
        // 4. classify, append
        let eps = migration_eps(self.all_electron, r.stats.scf_tol_used.unwrap_or(self.tol), old.sample.energy);
        let class = classify(old.sample.energy, &old.sample.forces, r.energy, r.stats.energy.internal, &r.forces, eps);
        let mut cs = old.clone();
        if class != MigrationClass::Unchanged {
            cs.sample.energy = r.energy;
            cs.sample.forces = r.forces.clone();
        }
        let new_code = migrated_code(code, class);
        // the status window (module doc, 4.): the code — and the identity of the sample it is for — is on disk before
        // the sample record that needs it
        let len = surrogate.collection().map_or(0, |c| c.samples.len());
        if new_code != 0 {
            self.log.pending = Some(::surrogate::PendingStatus::of(len, new_code, &cs.sample));
            self.write_log()?;
        }
        #[cfg(test)]
        {
            if self.crash_in_window == Some(i) {
                surrogate.arm_status_window_crash();
            }
            if self.fail_append == Some(i) {
                surrogate.arm_migrated_append_failure();
            }
        }
        if let Err(e) = surrogate.take_migrated(cs, new_code) {
            // nothing appended: the pending status names no record and is dropped, in memory and on disk (fix round 2);
            // a record appended without its status keeps it — the next attach applies it, to that record only
            if self.log.pending.is_some() && surrogate.collection().is_some_and(|c| c.samples.len() == len) {
                self.log.pending = None;
                let _ = self.write_log();
            }
            return Err(ProviderError::Invalid(format!("re-solving: the new collection could not be written ({e}) — the pass stops; it resumes from sample {i}")));
        }
        if class == MigrationClass::Unchanged {
            let (e, f) = self.log.class_u_max;
            self.log.class_u_max = ((r.energy - old.sample.energy).abs().max(e), max_force_diff(&old.sample.forces, &r.forces).max(f));
        }
        self.log.counts[class.index()] += 1;
        self.log.cursor = i + 1;
        self.log.pending = None;
        self.write_log()?;
        self.progress.done += 1;
        Ok(Some(self.progress()))
    }
}

impl Migrator {
    /// The migrated superseded samples whose likely superseder was not migrated (module doc, "Superseded samples"):
    /// for each source sample at code 2 that is not in the not-migrated list, the nearest later source sample of the same
    /// atoms (rms displacement) is taken as its superseder; listed when that one was not migrated or there is none.
    fn orphaned_superseded(&self) -> Vec<usize> {
        let not: Vec<usize> = self.log.not_migrated.iter().map(|x| x.0).collect();
        let rms = |a: &[[f64; 3]], b: &[[f64; 3]]| (a.iter().zip(b).map(|(p, q)| (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>()).sum::<f64>() / a.len().max(1) as f64).sqrt();
        let mut out = Vec::new();
        for (i, s) in self.source_samples.iter().enumerate() {
            if self.source_codes.get(i) != Some(&Q_SUPERSEDED) || not.contains(&i) {
                continue;
            }
            let superseder = (i + 1..self.source_samples.len())
                .filter(|&j| self.source_samples[j].sample.z == s.sample.z)
                .min_by(|&a, &b| rms(&self.source_samples[a].sample.pos, &s.sample.pos).total_cmp(&rms(&self.source_samples[b].sample.pos, &s.sample.pos)));
            if superseder.is_none_or(|j| not.contains(&j)) {
                out.push(i);
            }
        }
        out
    }
}

/// The display-only bond verdicts a collection folder may hold (`app::bond_experience::FILE_NAME`; spec §7.5: copied).
pub const BOND_EXPERIENCE_FILE: &str = "bond_experience.txt";

/// The checks of a cold solve (module doc, 3.): `Err` with the reason a sample is not migrated.
fn check_solve(r: &ForceResult, n: usize, h: f64, recorded: Option<u32>) -> Result<(), (NotMigrated, String)> {
    if r.forces.len() != n {
        return Err((NotMigrated::SolveFailed, format!("the solve returned {} forces for {n} nuclei", r.forces.len())));
    }
    if !r.energy.is_finite() || !r.stats.energy.internal.is_finite() || r.forces.iter().flatten().any(|x| !x.is_finite()) {
        return Err((NotMigrated::SolveFailed, "the solve returned a non-finite energy or force".into()));
    }
    // converged, at least one SCF step, the scene's electrons (the gpu-fault fix: `sim::answer_refusal`)
    if let Some(why) = sim::answer_refusal(r) {
        return Err((NotMigrated::SolveFailed, format!("the solve's answer was refused ({why})")));
    }
    if let Some(why) = r.stats.untrainable() {
        return Err((NotMigrated::Untrainable, format!("the solve is not trained on ({why})")));
    }
    if !r.stats.warnings.is_empty() {
        return Err((NotMigrated::SolverWarning, format!("the solve warned: {}", r.stats.warnings.join("; "))));
    }
    let limit = box_affected_face_density(h);
    match r.stats.face_density {
        None => return Err((NotMigrated::BoxUnknown, "box status unknown (the solve reported no face density)".into())),
        Some(f) if !(f <= limit) => return Err((NotMigrated::BoxAffected, format!("box-affected: face density {f:.2e} e/Bohr³ above the box rule's {limit:.2e} at h = {h}"))),
        Some(_) => {}
    }
    if let Some(m) = recorded {
        if r.stats.multiplicity != Some(m) {
            let got = r.stats.multiplicity.map_or("an unreported multiplicity".to_string(), |g| format!("multiplicity {g}"));
            return Err((NotMigrated::Multiplicity, format!("solved at {got}, recorded {m}")));
        }
    }
    Ok(())
}
