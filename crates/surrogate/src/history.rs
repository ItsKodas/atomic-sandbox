//! The audit history (M8a Task 1; plan `docs/superpowers/plans/2026-10-21-m8a-shared-model.md`):
//! per judged audit, what the model predicted against what the teacher measured, kept in a ring
//! of [`HISTORY_CAP`] points, persisted in the `.model` sidecar (format version 7,
//! [`crate::sidecar`]) and on its own ([`AuditHistory::encode`] / [`AuditHistory::decode`], the
//! `history.bin` of a shared collection). Nothing here decides anything: the history is the
//! evidence a calibration (Task 2) reads. Bookkeeping only, in atomic units (Hartree, Bohr).
//!
//! # What a point records
//!
//! For an audited geometry of N atoms, the model's prediction (E_m, F_m and, from its posterior
//! factor, the predictive standard deviations) and the teacher's answer (E_t, F_t), per atom j:
//!
//! ```text
//! err_force[j]                = ‖F_t[j] − F_m[j]‖                     Ha/Bohr, the Euclidean norm
//! pred_force_std[j]           = Prediction::force_std[j]
//!                             = √( Σ_a (σ_f² + g_{ja}ᵀ Σ g_{ja}) )     the same norm: for a calibrated
//!                                                                     posterior with label noise σ_f per
//!                                                                     component, E‖F_t[j] − F_m[j]‖² = pred_force_std[j]²
//! pred_force_std_epistemic[j] = Prediction::force_std_epistemic[j]
//!                             = √( Σ_a g_{ja}ᵀ Σ g_{ja} )              pred_force_std² = 3σ_f² + epistemic²
//! err_energy                  = E_t − E_m                              Ha, signed (teacher − model)
//! pred_energy_std             = Prediction::energy_std                  the total-energy std, aleatoric floor included
//! ```
//!
//! so for a calibrated posterior z_j = err_force[j] / pred_force_std[j] has E[z_j²] = 1 and
//! z_E = err_energy / pred_energy_std is a standard normal — what a calibration measures. Only the
//! second moment is shape-free: z_j² is χ²₃/3 when the three component errors are independent with
//! equal stds, but for a dimer under a pair model the model's force is exactly radial and the
//! teacher's transverse force is egg-box noise, so ΔF has one effective component and z_j² ≈ χ²₁
//! (calibrated median 0.68, p90 1.65, p99 2.58, against χ²₃/3's 0.89, 1.44, 1.95 — the physics
//! review's check). A calibration should fit the rms per class, not quantiles that assume a shape.
//! Nor are a point's atoms independent draws: the model's ΣF = 0 exactly and the teacher's ΣF is at
//! the egg-box level, so a dimer's two atoms are one draw (ΔF₂ = −ΔF₁ to a few e-6 Ha/Bohr and, by
//! the symmetry of their descriptors, equal stds); N atoms carry N − 1 independent error vectors.
//! Count evidence per point, or weight a point's atoms by (N − 1)/N. The epistemic part is kept beside the total
//! because rule 6 (`providers::surrogate`) judges the *epistemic* std against its bound, while the
//! aleatoric floor 3σ_f² is a property of the fit, not of the geometry: a calibration that inflates
//! the epistemic part alone needs the two apart, and σ_f at the time of the audit is recoverable
//! from them (σ_f² = (pred_force_std² − epistemic²)/3). Non-finite values are kept as they were
//! judged (a NaN gap is a failed audit, `providers::surrogate::audit`); a calibration filters them.
//!
//! # What kind of point
//!
//! `AuditPoint::kind` says where the comparison came from ([`PointKind`]): an **audit** (the
//! model answered the step and the teacher checked it: evidence inside the region today's rules
//! accept), or a **fallback** (the step went to the teacher because a rule refused the model's
//! answer, and the model's prediction with its variance existed for the geometry — rules 4b, 5,
//! 5b, 5d, 5e, 6, 7, the quarantine and audit fences, the relax verification, the first step after
//! an edit: evidence at the edge of that region, where a calibrated verdict decides; fix round 1,
//! the physics review's I3). A fallback point carries the refusing rule as a `u8` code whose table
//! the provider owns (`providers::surrogate::history_rule_code`); the history itself never reads
//! it. What is still censored: steps the gate refused before consulting the model (rules 1–4: no
//! prediction exists), steps inside a fenced cutoff shell (nothing is judged there — audits
//! included), non-finite predictions, box-affected teacher answers, unfit teacher answers (cores
//! overlap, an occupation change) and draft-mode guesses — the last four counted in
//! [`Unrecorded`].
//!
//! # Class numbering
//!
//! `AuditPoint::classes[j]` is the atom's environment class by its neighbour count within the
//! model's r_cut, in the plan's `u8` code: **0 isolated, 1 pair, 2 many-body** —
//! `providers::surrogate::atom_class(count).map_or(0, |k| k as u8 + 1)`. The provider's own
//! class tables (`ENV_CLASSES`, `class_loo`, and Task 2's `Calibration.per_class`) are indexed
//! like `atom_class`: **0 pair, 1 many-body**; an isolated atom (code 0 here) has no entry there.
//! So code c ≥ 1 of a point is row c − 1 of a per-class table.
//!
//! # Species (M8a user-test fix 2: class trust per environment species)
//!
//! `AuditPoint::species[j]` = (Z_j, the sorted distinct species of the atom's neighbours within
//! r_cut): with the class code it is the atom's [`EnvKey`], the key the calibrated class verdict
//! keeps its evidence under (`providers::surrogate::calibrate`). Z is a key only (rule 1). Empty
//! for a point recorded before species were kept: such a point counts for **every** key of its
//! atoms' classes (the conservative reading; the controller's ruling). Serialised as the optional
//! array `audit_species` after the ten ([`HistoryArrays::species`]): per point m = 0 (unknown) or
//! m = n_atoms followed, per atom, by Z, k and the k neighbour species ascending. A file without the
//! array reads with every point unknown.
//!
//! # What is not recorded
//!
//! A judged audit, and a fallback with the model's prediction, always leaves a point when the
//! model carried a variance factor; the [`Unrecorded`] counters say how many left none and why:
//! the model had no variance factor (`Prediction::force_std` was `None`), the teacher's answer was
//! box-affected (an audit's verdict `BoxAffected`: its gap is the box's confinement as much as the
//! model's error, and a calibration must not learn the box), the teacher's answer was unfit (cores
//! overlap or an occupation change: its labels are not the surface the model learns), or the
//! audit was a draft-mode guess at a geometry strict mode refuses (not a verdict on the strict
//! model). Audits that are not judged at all (discarded after a settings change, not judged inside
//! a fenced shell band) are not counted here: they have their activity events.
//!
//! # Trust-tightening fields (plan `docs/superpowers/plans/2026-11-19-trust-tightening.md` Task 1)
//!
//! Three per-point fields the trust-tightening calibration reads (spec
//! `2026-11-18-trust-tightening-design.md` §3, §4); the history records them and decides nothing:
//!
//! - `AuditPoint::fit_id`: the fit that was judged, a counter bumped only by a refit that solved new
//!   coefficients (never by a policy change, restore, forget, install or attach). `None` for a point
//!   recorded before fit ids — every point saved before this field — which does not count toward
//!   the evidence floor's fits.
//! - `AuditPoint::flags` ([`PointFlags`]): bit 1 a calibration tick, bit 2 fenced (inside a
//!   quarantine, misfit or audit fence of its key), bit 4 cadence-fresh (by the never-reset
//!   counter). 0 when absent. Bits above 7 are refused.
//! - `AuditPoint::energy_std_epistemic`: Ha, the total-energy epistemic std (`pred_energy_std`
//!   holds the total, aleatoric floor included). `None` when not recorded: such a point is not in
//!   λ_E's sample.
//!
//! Serialised as three optional per-point arrays after the species array: `audit_fit` (−1 for an
//! absent id, else an integer below 2⁵³), `audit_flags` (an integer 0–7), `audit_energy_std_epistemic`
//! (−1 for absent, else finite and ≥ 0); any other value — another negative, a NaN, an infinity, a
//! fraction — is refused, never read. The three are written together, for every point, only when
//! some point carries one of the fields; otherwise none is written and the file is version 1, bit
//! for bit. [`AuditHistory::push`] panics on a point whose fields could not be written (an id of 2⁵³
//! or more, a flags word above 7, a non-finite or negative std): a programming error at the
//! recording site, like a class code beyond 2.
//!
//! # Standalone encoding ([`AuditHistory::encode`])
//!
//! ```text
//! "ASAUDIT1" or "ASAUDIT2"        8 bytes, magic (the version)
//! recorded                        u64 LE, every point ever pushed (len() + the evicted)
//! no_variance, box_affected, draft, unfit   u64 LE each, the Unrecorded counters
//! array₁ … array₁₀                each u64 LE length n, then n f64 LE values — the ten
//!                                 arrays of [`HistoryArrays::NAMES`] in that order
//! array₁₁                         `audit_species` ("Species" above; optional in version 1,
//!                                 absent in its older files; required in version 2)
//! array₁₂ … array₁₄ (version 2)   `audit_fit`, `audit_flags`, `audit_energy_std_epistemic`
//!                                 ("Trust-tightening fields" above)
//! (end — trailing bytes are an error)
//! ```
//!
//! **Version 2** ([`MAGIC`], `ASAUDIT2`) is written when the history carries the trust-tightening
//! arrays, **version 1** ([`MAGIC_V1`]) otherwise — bit for bit the writer before version 2. The
//! bump is real: a version-1 reader calls the three arrays trailing bytes. A build before version 2
//! starts an empty history over a version-2 file and overwrites it at its next save (the app's
//! downgrade warning says so, `app::update::formats`); this build and every later one refuse a
//! newer build's file by its magic ([`written_by_newer_build`]: `ASAUD…` that is not one of
//! [`MAGICS`]) and a collection never overwrites it (`crate::collection`, "history.bin").
//!
//! The arrays, with M points and A = Σ atoms over them: `audit_atoms`, `audit_model_gen`,
//! `audit_kind` (0 an audit, 1 + rule code a fallback), `audit_energy_std`, `audit_err_energy`
//! (M each; counts, generations and kinds as exact integral f64), `audit_classes`,
//! `audit_force_std`, `audit_force_std_epistemic`, `audit_err_force`, `audit_teacher_force` (A
//! each; class codes 0, 1, 2 as integral f64; `audit_teacher_force` = |F_teacher| per atom, added
//! in M8a Task 2's fix round while version 7 was unreleased — the per-atom bound's input). The
//! sidecar embeds the same ten arrays after its sample tail under a header
//! key `audit_history` that carries the six counters ([`crate::sidecar`], "Version 7"; the species
//! and the three trust-tightening arrays follow, "Version 10"), so both
//! forms share [`AuditHistory::to_arrays`] / [`AuditHistory::from_arrays`]. Every length is checked
//! against the bytes present before anything is allocated; a decode never panics on a corrupt
//! input.

use std::collections::VecDeque;
use std::io;

/// Points the ring keeps (the plan's figure): at one audit per 50 answered steps and ≈ 3 fs/s of
/// DFT-paced audits this is hours of audits; at the surrogate's own rate it is the last few
/// thousand.
pub const HISTORY_CAP: usize = 4096;

/// The first eight bytes of a version-2 standalone history file (module doc, "Standalone
/// encoding"): the newest version, written when a point carries a trust-tightening field.
pub const MAGIC: &[u8; 8] = b"ASAUDIT2";
/// The first eight bytes of a version-1 file: the ten arrays and the optional species array only.
/// Still written for a history none of whose points carries a trust-tightening field, bit for bit
/// as before, so an older build keeps reading it.
pub const MAGIC_V1: &[u8; 8] = b"ASAUDIT1";
/// Every magic this build reads, oldest first (version = index + 1). [`MAGIC`] is the last; its
/// number is the `history` format a release manifest records (`app::update::formats`).
pub const MAGICS: [&[u8; 8]; 2] = [MAGIC_V1, MAGIC];

/// Largest atom count a point may declare (a hostile-input guard, far above any scene).
const MAX_ATOMS: f64 = (1u64 << 24) as f64;
/// Largest model generation accepted (exact in f64).
const MAX_GEN: f64 = (1u64 << 53) as f64;
/// Fit ids are below 2⁵³, so every id is exact in its f64 array.
const FIT_LIMIT: u64 = 1u64 << 53;
/// An absent fit id or epistemic energy std in its array (module doc, "Trust-tightening fields").
const ABSENT: f64 = -1.0;

/// Whether `bytes` start with an audit-history magic of a newer build: `ASAUD…` that is none of
/// [`MAGICS`] (a later `ASAUDIT<d>`, or any other form a later layout might take). Such a file is
/// refused by [`AuditHistory::decode`] as a newer build's — never as damage — and a collection
/// keeps it untouched (`crate::collection`, "history.bin").
pub fn written_by_newer_build(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes.starts_with(b"ASAUD") && !MAGICS.iter().any(|m| &bytes[..8] == *m)
}

/// A point's flags word (module doc, "Trust-tightening fields"): which sample the point belongs to,
/// set by the provider when it records the point. Bits beyond [`ALL`](Self::ALL) are refused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PointFlags(pub u8);

impl PointFlags {
    /// The point was recorded at a calibration tick.
    pub const TICK: u8 = 1;
    /// The geometry sits inside a quarantine, misfit or audit fence of its key: excluded from P_A.
    pub const FENCED: u8 = 2;
    /// The step was fresh by the never-reset cadence counter (a multiple of `variance_cadence`).
    pub const CADENCE_FRESH: u8 = 4;
    /// Every bit this build defines.
    pub const ALL: u8 = Self::TICK | Self::FENCED | Self::CADENCE_FRESH;

    /// Every bit of `bit` is set (one bit: that bit is set).
    pub fn has(self, bit: u8) -> bool {
        self.0 & bit == bit
    }
}

/// Where a point's comparison came from (module doc, "What kind of point").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointKind {
    /// A judged audit: the model answered, the teacher checked.
    Audit,
    /// A step the teacher answered because `rule` refused the model's answer (the code table is
    /// the provider's, `providers::surrogate::history_rule_code`); the model's prediction with its
    /// variance existed for the geometry.
    Fallback { rule: u8 },
}

impl PointKind {
    /// The serialised code: 0 an audit, 1 + rule a fallback.
    pub fn code(self) -> u16 {
        match self {
            PointKind::Audit => 0,
            PointKind::Fallback { rule } => 1 + rule as u16,
        }
    }

    /// From the serialised code (`None` beyond 256).
    pub fn from_code(c: u16) -> Option<PointKind> {
        match c {
            0 => Some(PointKind::Audit),
            1..=256 => Some(PointKind::Fallback { rule: (c - 1) as u8 }),
            _ => None,
        }
    }

    pub fn is_audit(self) -> bool {
        self == PointKind::Audit
    }
}

/// One judged comparison — an audit or a fallback step — of the model's prediction against the
/// teacher's answer (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct AuditPoint {
    pub kind: PointKind,
    /// Per atom: 0 isolated, 1 pair, 2 many-body (module doc, "Class numbering").
    pub classes: Vec<u8>,
    /// Per atom, Ha/Bohr: the total predictive force std (`Prediction::force_std`).
    pub pred_force_std: Vec<f64>,
    /// Per atom, Ha/Bohr: its epistemic part (`Prediction::force_std_epistemic`).
    pub pred_force_std_epistemic: Vec<f64>,
    /// Ha: the total-energy predictive std (`Prediction::energy_std`).
    pub pred_energy_std: f64,
    /// Per atom, Ha/Bohr: ‖F_teacher − F_model‖.
    pub err_force: Vec<f64>,
    /// Per atom, Ha/Bohr: |F_teacher| — what the policy's per-atom force bound
    /// max(2e-3, 10 % |F|) ≤ 1e-2 is formed from, so a calibration can judge the error in the
    /// units the policy claims (M8a Task 2 fix round 1, the physics review's A.1).
    pub teacher_force: Vec<f64>,
    /// Ha: E_teacher − E_model (signed).
    pub err_energy: f64,
    pub n_atoms: usize,
    /// The provider's model generation that was judged (`SurrogateKs::model_gen`).
    pub model_gen: u64,
    /// Per atom: (Z, its neighbours' distinct species within r_cut, ascending) — module doc,
    /// "Species"; empty when unknown (a point recorded before species were kept).
    pub species: Vec<(u8, Vec<u8>)>,
    /// The fit that was judged: a counter bumped only by a refit that solved new coefficients
    /// (module doc, "Trust-tightening fields"). `None` = absent (a point recorded before fit ids):
    /// it does not count toward the evidence floor's fits.
    pub fit_id: Option<u64>,
    /// Tick / fenced / cadence-fresh ([`PointFlags`]); 0 when absent — such a point is not in P_A.
    pub flags: PointFlags,
    /// Ha: the total-energy **epistemic** std (`pred_energy_std` without the aleatoric floor);
    /// `None` = not recorded — such a point is not in λ_E's sample.
    pub energy_std_epistemic: Option<f64>,
}

/// The key the calibrated class verdict keeps its evidence under (module doc, "Species"): an
/// atom's class code (1 pair, 2 many-body), its Z and its neighbours' distinct species within
/// r_cut, ascending. Z is a key only — nothing per element is read from it (rule 1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EnvKey {
    pub class: u8,
    pub z: u8,
    pub neighbours: Vec<u8>,
}

impl AuditPoint {
    /// Atom j's [`EnvKey`], `None` for an isolated atom (class 0) or unknown species.
    pub fn key(&self, j: usize) -> Option<EnvKey> {
        let (z, nb) = self.species.get(j)?;
        (self.classes[j] != 0).then(|| EnvKey { class: self.classes[j], z: *z, neighbours: nb.clone() })
    }
}

impl AuditPoint {
    /// The shapes agree: every per-atom array has `n_atoms` entries and every class code is 0–2.
    fn well_formed(&self) -> Result<(), String> {
        let n = self.n_atoms;
        if n == 0 {
            return Err("an audit point of no atoms".into());
        }
        for (name, len) in [("classes", self.classes.len()), ("pred_force_std", self.pred_force_std.len()), ("pred_force_std_epistemic", self.pred_force_std_epistemic.len()), ("err_force", self.err_force.len()), ("teacher_force", self.teacher_force.len())] {
            if len != n {
                return Err(format!("audit point: {name} has {len} entries for {n} atoms"));
            }
        }
        if let Some(c) = self.classes.iter().find(|&&c| c > 2) {
            return Err(format!("audit point: class code {c} (0 isolated, 1 pair, 2 many-body)"));
        }
        if !(self.species.is_empty() || self.species.len() == n) {
            return Err(format!("audit point: species has {} entries for {n} atoms", self.species.len()));
        }
        if self.species.iter().any(|(z, nb)| *z == 0 || nb.contains(&0) || nb.windows(2).any(|w| w[0] >= w[1])) {
            return Err("audit point: a species entry is 0 or its neighbours are not ascending and distinct".into());
        }
        if let Some(id) = self.fit_id.filter(|&id| id >= FIT_LIMIT) {
            return Err(format!("audit point: fit id {id} is 2^53 or more"));
        }
        if self.flags.0 > PointFlags::ALL {
            return Err(format!("audit point: flags word {} (bits 1 tick, 2 fenced, 4 cadence-fresh)", self.flags.0));
        }
        if let Some(s) = self.energy_std_epistemic.filter(|s| !(s.is_finite() && *s >= 0.0)) {
            return Err(format!("audit point: epistemic energy std {s} is not finite and non-negative"));
        }
        Ok(())
    }

    /// The point carries a trust-tightening field (a fit id, a flag or an epistemic energy std).
    fn has_new_fields(&self) -> bool {
        self.fit_id.is_some() || self.flags.0 != 0 || self.energy_std_epistemic.is_some()
    }
}

/// Judged audits and consulted-model fallbacks that left no point, by reason (module doc, "What
/// is not recorded").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unrecorded {
    /// The model had no variance factor when the comparison was made.
    pub no_variance: u64,
    /// The teacher's answer was box-affected (an audit's verdict `BoxAffected`).
    pub box_affected: u64,
    /// A draft-mode guess at a geometry strict mode refuses.
    pub draft: u64,
    /// The teacher's answer was unfit (cores overlap, an occupation change: used, never fitted).
    pub unfit: u64,
}

/// The ring of the newest [`HISTORY_CAP`] audit points, oldest first (module doc).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AuditHistory {
    ring: VecDeque<AuditPoint>,
    recorded: u64,
    pub unrecorded: Unrecorded,
}

/// The ten arrays of the history's serialised form (module doc, "Standalone encoding").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HistoryArrays {
    pub atoms: Vec<f64>,
    pub model_gen: Vec<f64>,
    pub kind: Vec<f64>,
    pub energy_std: Vec<f64>,
    pub err_energy: Vec<f64>,
    pub classes: Vec<f64>,
    pub force_std: Vec<f64>,
    pub force_std_epistemic: Vec<f64>,
    pub err_force: Vec<f64>,
    pub teacher_force: Vec<f64>,
    /// The optional `audit_species` array (module doc, "Species"); empty = every point unknown.
    pub species: Vec<f64>,
    /// The three optional per-point arrays of version 2 (module doc, "Trust-tightening fields"):
    /// all three empty = every point without them; otherwise one entry per point each.
    pub fit: Vec<f64>,
    pub flags: Vec<f64>,
    pub energy_std_epistemic: Vec<f64>,
}

impl HistoryArrays {
    /// The optional eleventh array's name (module doc, "Species").
    pub const SPECIES: &'static str = "audit_species";
    /// The version-2 per-point arrays' names, in serialised order after the species array (module
    /// doc, "Trust-tightening fields").
    pub const FIT: &'static str = "audit_fit";
    pub const FLAGS: &'static str = "audit_flags";
    pub const ENERGY_STD_EPI: &'static str = "audit_energy_std_epistemic";

    /// The three version-2 arrays in serialised order, when present (all three empty: `None`).
    pub fn trust_fields(&self) -> Option<[(&'static str, &[f64]); 3]> {
        let present = !(self.fit.is_empty() && self.flags.is_empty() && self.energy_std_epistemic.is_empty());
        present.then_some([(Self::FIT, &self.fit[..]), (Self::FLAGS, &self.flags[..]), (Self::ENERGY_STD_EPI, &self.energy_std_epistemic[..])])
    }

    /// The arrays' names in serialised order (the first [`PER_POINT`](Self::PER_POINT) are per
    /// point, the rest per atom).
    pub const NAMES: [&'static str; 10] = ["audit_atoms", "audit_model_gen", "audit_kind", "audit_energy_std", "audit_err_energy", "audit_classes", "audit_force_std", "audit_force_std_epistemic", "audit_err_force", "audit_teacher_force"];
    /// How many of [`NAMES`](Self::NAMES) are per point.
    pub const PER_POINT: usize = 5;

    /// The arrays in [`NAMES`](Self::NAMES) order.
    pub fn as_slices(&self) -> [(&'static str, &[f64]); 10] {
        [
            (Self::NAMES[0], &self.atoms),
            (Self::NAMES[1], &self.model_gen),
            (Self::NAMES[2], &self.kind),
            (Self::NAMES[3], &self.energy_std),
            (Self::NAMES[4], &self.err_energy),
            (Self::NAMES[5], &self.classes),
            (Self::NAMES[6], &self.force_std),
            (Self::NAMES[7], &self.force_std_epistemic),
            (Self::NAMES[8], &self.err_force),
            (Self::NAMES[9], &self.teacher_force),
        ]
    }

    /// The arrays from ten vectors in [`NAMES`](Self::NAMES) order.
    pub fn from_vecs(v: [Vec<f64>; 10]) -> HistoryArrays {
        let [atoms, model_gen, kind, energy_std, err_energy, classes, force_std, force_std_epistemic, err_force, teacher_force] = v;
        HistoryArrays { atoms, model_gen, kind, energy_std, err_energy, classes, force_std, force_std_epistemic, err_force, teacher_force, ..HistoryArrays::default() }
    }
}

impl AuditHistory {
    /// Records a judged audit; when the ring is full the oldest point goes.
    ///
    /// # Panics
    /// If the point's per-atom arrays do not all have `n_atoms` entries or a class code is beyond 2
    /// (a programming error at the recording site, not a data condition).
    pub fn push(&mut self, p: AuditPoint) {
        if let Err(e) = p.well_formed() {
            panic!("AuditHistory::push: {e}");
        }
        if self.ring.len() >= HISTORY_CAP {
            self.ring.pop_front();
        }
        self.ring.push_back(p);
        self.recorded += 1;
    }

    /// The points held, oldest first.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &AuditPoint> + DoubleEndedIterator {
        self.ring.iter()
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Every point ever pushed (`len()` plus the evicted).
    pub fn recorded(&self) -> u64 {
        self.recorded
    }

    /// The arrays of the serialised form (module doc): the ten, the species array and — only when
    /// some point carries a trust-tightening field — the three version-2 arrays, for every point.
    pub fn to_arrays(&self) -> HistoryArrays {
        let mut a = HistoryArrays::default();
        if self.ring.iter().any(AuditPoint::has_new_fields) {
            a.fit = self.ring.iter().map(|p| p.fit_id.map_or(ABSENT, |id| id as f64)).collect();
            a.flags = self.ring.iter().map(|p| p.flags.0 as f64).collect();
            a.energy_std_epistemic = self.ring.iter().map(|p| p.energy_std_epistemic.unwrap_or(ABSENT)).collect();
        }
        for p in &self.ring {
            a.atoms.push(p.n_atoms as f64);
            a.model_gen.push(p.model_gen as f64);
            a.kind.push(p.kind.code() as f64);
            a.energy_std.push(p.pred_energy_std);
            a.err_energy.push(p.err_energy);
            a.classes.extend(p.classes.iter().map(|&c| c as f64));
            a.force_std.extend_from_slice(&p.pred_force_std);
            a.force_std_epistemic.extend_from_slice(&p.pred_force_std_epistemic);
            a.err_force.extend_from_slice(&p.err_force);
            a.teacher_force.extend_from_slice(&p.teacher_force);
            a.species.push(p.species.len() as f64);
            for (z, nb) in &p.species {
                a.species.push(*z as f64);
                a.species.push(nb.len() as f64);
                a.species.extend(nb.iter().map(|&x| x as f64));
            }
        }
        a
    }

    /// The history from its arrays and counters (module doc): the per-point arrays must agree in
    /// length, the counts be integral and sum to the per-atom arrays' length, the generations
    /// integral, the kinds a [`PointKind`] code, the class codes 0, 1 or 2; the version-2 arrays all
    /// absent or all one per point, each fit id −1 (absent) or an integer in 0 .. 2⁵³, each flags
    /// word an integer in 0 ..= 7, each epistemic energy std −1 (absent) or finite and ≥ 0. More
    /// points than [`HISTORY_CAP`] keep the newest.
    pub fn from_arrays(a: &HistoryArrays, recorded: u64, unrecorded: Unrecorded) -> Result<AuditHistory, String> {
        let m = a.atoms.len();
        for (name, len) in [("audit_model_gen", a.model_gen.len()), ("audit_kind", a.kind.len()), ("audit_energy_std", a.energy_std.len()), ("audit_err_energy", a.err_energy.len())] {
            if len != m {
                return Err(format!("{name} has {len} entries for {m} audit points (audit_atoms)"));
            }
        }
        let trust = decode_trust_fields(a)?;
        let mut total = 0usize;
        for (i, &n) in a.atoms.iter().enumerate() {
            if !(n.fract() == 0.0 && n >= 1.0 && n <= MAX_ATOMS) {
                return Err(format!("audit_atoms[{i}] = {n} is not a positive atom count"));
            }
            total = total.checked_add(n as usize).ok_or("audit_atoms overflow")?;
        }
        for (name, len) in [("audit_classes", a.classes.len()), ("audit_force_std", a.force_std.len()), ("audit_force_std_epistemic", a.force_std_epistemic.len()), ("audit_err_force", a.err_force.len()), ("audit_teacher_force", a.teacher_force.len())] {
            if len != total {
                return Err(format!("{name} has {len} entries, audit_atoms sums to {total}"));
            }
        }
        for (i, &g) in a.model_gen.iter().enumerate() {
            if !(g.fract() == 0.0 && g >= 0.0 && g <= MAX_GEN) {
                return Err(format!("audit_model_gen[{i}] = {g} is not a generation"));
            }
        }
        let mut kinds = Vec::with_capacity(m);
        for (i, &k) in a.kind.iter().enumerate() {
            match (k.fract() == 0.0 && (0.0..=256.0).contains(&k)).then(|| PointKind::from_code(k as u16)).flatten() {
                Some(kind) => kinds.push(kind),
                None => return Err(format!("audit_kind[{i}] = {k} is not a point kind (0 an audit, 1 + rule a fallback)")),
            }
        }
        if let Some((i, &c)) = a.classes.iter().enumerate().find(|(_, &c)| !(c == 0.0 || c == 1.0 || c == 2.0)) {
            return Err(format!("audit_classes[{i}] = {c} is not 0, 1 or 2"));
        }
        let species = decode_species(&a.species, &a.atoms)?;
        let mut h = AuditHistory { ring: VecDeque::with_capacity(m.min(HISTORY_CAP)), recorded: 0, unrecorded };
        let mut at = 0usize;
        for (i, species) in species.into_iter().enumerate() {
            let n = a.atoms[i] as usize;
            let r = at..at + n;
            let (fit_id, flags, energy_std_epistemic) = trust.as_ref().map_or((None, PointFlags::default(), None), |t| t[i]);
            h.push(AuditPoint {
                kind: kinds[i],
                classes: a.classes[r.clone()].iter().map(|&c| c as u8).collect(),
                pred_force_std: a.force_std[r.clone()].to_vec(),
                pred_force_std_epistemic: a.force_std_epistemic[r.clone()].to_vec(),
                pred_energy_std: a.energy_std[i],
                err_force: a.err_force[r.clone()].to_vec(),
                teacher_force: a.teacher_force[r].to_vec(),
                err_energy: a.err_energy[i],
                n_atoms: n,
                model_gen: a.model_gen[i] as u64,
                species,
                fit_id,
                flags,
                energy_std_epistemic,
            });
            at += n;
        }
        if recorded < h.ring.len() as u64 {
            return Err(format!("recorded {recorded} is below the {} points present", h.ring.len()));
        }
        h.recorded = recorded;
        Ok(h)
    }

    /// The standalone form (module doc, "Standalone encoding"): version 2 when some point carries a
    /// trust-tightening field, else version 1, bit for bit as the writer before it.
    pub fn encode(&self) -> Vec<u8> {
        encode_arrays(&self.to_arrays(), self.recorded, self.unrecorded)
    }

    /// Reads the standalone form, either version; `InvalidData` names what is wrong (never a
    /// panic). A newer build's file ([`written_by_newer_build`]) is refused as such.
    pub fn decode(bytes: &[u8]) -> io::Result<AuditHistory> {
        Self::decode_checked(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("audit history: {e}")))
    }

    fn decode_checked(bytes: &[u8]) -> Result<AuditHistory, String> {
        if written_by_newer_build(bytes) {
            return Err(format!("written by a newer build (magic {}; this build reads ASAUDIT1 and ASAUDIT2)", String::from_utf8_lossy(&bytes[..8])));
        }
        if bytes.len() < 8 + 5 * 8 {
            return Err(format!("{} bytes is too short for a history", bytes.len()));
        }
        let v2 = match &bytes[..8] {
            m if m == MAGIC_V1 => false,
            m if m == MAGIC => true,
            _ => return Err("not an audit history (magic is not ASAUDIT1 or ASAUDIT2)".into()),
        };
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
        let (recorded, no_variance, box_affected, draft, unfit) = (u64_at(8), u64_at(16), u64_at(24), u64_at(32), u64_at(40));
        let mut at = 48usize;
        let mut arrays: Vec<Vec<f64>> = Vec::with_capacity(HistoryArrays::NAMES.len() + 4);
        // the ten, then the species array (module doc, "Species": optional in version 1, absent in
        // older files), then in version 2 the three trust-tightening arrays (required there)
        let trust_names = [HistoryArrays::FIT, HistoryArrays::FLAGS, HistoryArrays::ENERGY_STD_EPI];
        let names = HistoryArrays::NAMES.iter().copied().chain([HistoryArrays::SPECIES]).chain(trust_names.into_iter().filter(|_| v2));
        for (i, name) in names.enumerate() {
            if !v2 && i == HistoryArrays::NAMES.len() && at == bytes.len() {
                break;
            }
            let prefix = bytes.get(at..at + 8).ok_or_else(|| format!("file ends before array {name}"))?;
            let n = u64::from_le_bytes(prefix.try_into().expect("8 bytes"));
            at += 8;
            let len = usize::try_from(n).ok().and_then(|n| n.checked_mul(8)).and_then(|b| b.checked_add(at)).filter(|&e| e <= bytes.len()).ok_or_else(|| format!("file ends inside array {name}"))?;
            arrays.push(bytes[at..len].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect());
            at = len;
        }
        if at != bytes.len() {
            return Err(format!("{} trailing bytes after the last array", bytes.len() - at));
        }
        let (fit, flags, energy_std_epistemic) = if v2 {
            let e = arrays.pop().expect("fourteen");
            let f = arrays.pop().expect("fourteen");
            (arrays.pop().expect("fourteen"), f, e)
        } else {
            Default::default()
        };
        let species = if arrays.len() > HistoryArrays::NAMES.len() { arrays.pop().expect("eleven") } else { Vec::new() };
        let a = HistoryArrays { species, fit, flags, energy_std_epistemic, ..HistoryArrays::from_vecs(arrays.try_into().expect("ten arrays")) };
        Self::from_arrays(&a, recorded, Unrecorded { no_variance, box_affected, draft, unfit })
    }
}

/// The standalone bytes of `a` (module doc, "Standalone encoding"): version 2 with the three
/// trust-tightening arrays when any of them is non-empty, else version 1.
fn encode_arrays(a: &HistoryArrays, recorded: u64, unrecorded: Unrecorded) -> Vec<u8> {
    let trust = a.trust_fields();
    let slices = a.as_slices();
    let arrays: Vec<&[f64]> = slices.iter().map(|(_, v)| *v).chain([&a.species[..]]).chain(trust.iter().flatten().map(|(_, v)| *v)).collect();
    let body: usize = arrays.iter().map(|v| 8 + 8 * v.len()).sum();
    let mut out = Vec::with_capacity(8 + 5 * 8 + body);
    out.extend_from_slice(if trust.is_some() { MAGIC } else { MAGIC_V1 });
    for c in [recorded, unrecorded.no_variance, unrecorded.box_affected, unrecorded.draft, unrecorded.unfit] {
        out.extend_from_slice(&c.to_le_bytes());
    }
    // the ten, the species array (module doc, "Species"), then the three of version 2
    for v in arrays {
        out.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// The three version-2 arrays (module doc, "Trust-tightening fields") into per-point (fit id,
/// flags, epistemic energy std); `None` when all three are absent. Strict: one entry per point
/// each, −1 or an integer below 2⁵³ for an id, an integer 0 ..= 7 for a flags word, −1 or a finite
/// non-negative value for a std.
#[allow(clippy::type_complexity)]
fn decode_trust_fields(a: &HistoryArrays) -> Result<Option<Vec<(Option<u64>, PointFlags, Option<f64>)>>, String> {
    let Some(fields) = a.trust_fields() else { return Ok(None) };
    let m = a.atoms.len();
    for (name, v) in fields {
        if v.len() != m {
            return Err(format!("{name} has {} entries for {m} audit points (audit_atoms)", v.len()));
        }
    }
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let (f, g, s) = (a.fit[i], a.flags[i], a.energy_std_epistemic[i]);
        let fit_id = if f == ABSENT {
            None
        } else if f.fract() == 0.0 && f >= 0.0 && f < FIT_LIMIT as f64 {
            Some(f as u64)
        } else {
            return Err(format!("{}[{i}] = {f} is not a fit id (an integer below 2^53, or -1 when absent)", HistoryArrays::FIT));
        };
        if !(g.fract() == 0.0 && (0.0..=PointFlags::ALL as f64).contains(&g)) {
            return Err(format!("{}[{i}] = {g} is not a flags word (an integer 0 to 7)", HistoryArrays::FLAGS));
        }
        let std = if s == ABSENT {
            None
        } else if s.is_finite() && s >= 0.0 {
            Some(s)
        } else {
            return Err(format!("{}[{i}] = {s} is not an energy std (finite and non-negative, or -1 when absent)", HistoryArrays::ENERGY_STD_EPI));
        };
        out.push((fit_id, PointFlags(g as u8), std));
    }
    Ok(Some(out))
}

/// The `audit_species` array (module doc, "Species") into per-point species lists; empty → every
/// point unknown. Strict: integral codes, m ∈ {0, n_atoms}, Z in 1..=255, neighbours ascending and
/// distinct, every entry consumed.
fn decode_species(v: &[f64], atoms: &[f64]) -> Result<Vec<Vec<(u8, Vec<u8>)>>, String> {
    if v.is_empty() {
        return Ok(vec![Vec::new(); atoms.len()]);
    }
    let mut at = 0usize;
    let mut next = |what: &str, max: f64| -> Result<usize, String> {
        let x = *v.get(at).ok_or_else(|| format!("audit_species ends inside {what}"))?;
        if !(x.fract() == 0.0 && (0.0..=max).contains(&x)) {
            return Err(format!("audit_species[{at}] = {x} is not a valid {what}"));
        }
        at += 1;
        Ok(x as usize)
    };
    let mut out = Vec::with_capacity(atoms.len());
    for &n in atoms {
        let n = n as usize;
        let m = next("atom count", MAX_ATOMS)?;
        if m != 0 && m != n {
            return Err(format!("audit_species: {m} atoms for a point of {n}"));
        }
        let mut p = Vec::with_capacity(m);
        for _ in 0..m {
            let z = next("species", 255.0)? as u8;
            let k = next("neighbour count", 255.0)?;
            let nb = (0..k).map(|_| next("species", 255.0).map(|x| x as u8)).collect::<Result<Vec<u8>, String>>()?;
            if z == 0 || nb.contains(&0) || nb.windows(2).any(|w| w[0] >= w[1]) {
                return Err("audit_species: a species is 0 or the neighbours are not ascending and distinct".into());
            }
            p.push((z, nb));
        }
        out.push(p);
    }
    if at != v.len() {
        return Err(format!("audit_species: {} trailing entries", v.len() - at));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(n: usize, gen: u64) -> AuditPoint {
        AuditPoint {
            kind: if gen % 2 == 0 { PointKind::Audit } else { PointKind::Fallback { rule: (gen % 7) as u8 } },
            classes: (0..n).map(|j| (j % 3) as u8).collect(),
            pred_force_std: vec![1e-3; n],
            pred_force_std_epistemic: vec![5e-4; n],
            pred_energy_std: 1e-5,
            err_force: (0..n).map(|j| 1e-4 * j as f64).collect(),
            teacher_force: (0..n).map(|j| 3e-2 * (1 + j) as f64).collect(),
            err_energy: -2e-6,
            n_atoms: n,
            model_gen: gen,
            species: if gen % 3 == 0 { Vec::new() } else { (0..n).map(|j| (1 + (j % 2) as u8, if j == 0 { vec![1, 8] } else { vec![8] })).collect() },
            fit_id: None,
            flags: PointFlags::default(),
            energy_std_epistemic: None,
        }
    }

    /// M8a user-test fix 2: the species ride along as the optional eleventh array; a file written
    /// without it (the ten arrays only) reads with every point's species unknown, the rest intact;
    /// a malformed species array is refused.
    #[test]
    fn species_round_trip_and_old_files_read_unknown() {
        let mut h = AuditHistory::default();
        for g in 0..7 {
            h.push(point(3, g));
        }
        assert!(h.iter().any(|p| !p.species.is_empty()) && h.iter().any(|p| p.species.is_empty()));
        assert_eq!(AuditHistory::decode(&h.encode()).unwrap(), h);
        assert_eq!(h.iter().nth(1).unwrap().key(1), Some(EnvKey { class: 1, z: 2, neighbours: vec![8] }));
        assert_eq!(h.iter().nth(1).unwrap().key(0), None, "an isolated atom has no key");
        // the ten arrays only: drop the species array (its length prefix and values, last)
        let bytes = h.encode();
        let n = h.to_arrays().species.len();
        let old = &bytes[..bytes.len() - 8 - 8 * n];
        let back = AuditHistory::decode(old).unwrap();
        assert!(back.iter().all(|p| p.species.is_empty()));
        assert!(back.iter().zip(h.iter()).all(|(a, b)| AuditPoint { species: Vec::new(), ..b.clone() } == *a));
        // malformed: a species count that is not the point's atom count
        let mut a = h.to_arrays();
        a.species[0] = 2.0;
        assert!(AuditHistory::from_arrays(&a, h.recorded(), h.unrecorded).is_err());
    }

    /// The arrays form and back is the identity, counters included; non-finite values survive
    /// bit for bit (a NaN gap is a real failed audit).
    #[test]
    fn arrays_round_trip_with_non_finite_values() {
        let mut h = AuditHistory::default();
        let mut p = point(3, 9);
        p.err_energy = f64::NAN;
        p.err_force[1] = f64::INFINITY;
        h.push(p);
        h.push(point(1, 10));
        h.push(point(2, 255 + 256));
        h.unrecorded.draft = 4;
        h.unrecorded.unfit = 2;
        let back = AuditHistory::from_arrays(&h.to_arrays(), h.recorded(), h.unrecorded).unwrap();
        assert_eq!(back.len(), 3);
        let (a, b) = (back.iter().next().unwrap(), h.iter().next().unwrap());
        assert!(a.err_energy.is_nan() && a.err_force[1] == f64::INFINITY);
        assert_eq!((a.classes.clone(), a.n_atoms, a.model_gen, a.kind), (b.classes.clone(), b.n_atoms, b.model_gen, b.kind));
        assert_eq!(back.iter().map(|p| p.kind).collect::<Vec<_>>(), vec![PointKind::Fallback { rule: 2 }, PointKind::Audit, PointKind::Fallback { rule: 0 }]);
        assert_eq!(PointKind::from_code(256), Some(PointKind::Fallback { rule: 255 }));
        assert_eq!(PointKind::from_code(257), None);
        assert_eq!(back.unrecorded, h.unrecorded);
        assert_eq!(back.recorded(), 3);
        let dec = AuditHistory::decode(&h.encode()).unwrap();
        assert!(dec.iter().next().unwrap().err_energy.is_nan());
    }

    /// Misshapen input is refused with the array named; `recorded` below the points present too.
    #[test]
    fn misshapen_arrays_are_refused() {
        let mut h = AuditHistory::default();
        h.push(point(2, 1));
        let good = h.to_arrays();
        let mut a = good.clone();
        a.model_gen.pop();
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_model_gen"));
        let mut a = good.clone();
        a.atoms[0] = 1.5;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_atoms"));
        let mut a = good.clone();
        a.err_force.push(0.0);
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_err_force"));
        let mut a = good.clone();
        a.teacher_force.pop();
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_teacher_force"));
        let mut a = good.clone();
        a.classes[1] = 3.0;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_classes"));
        let mut a = good.clone();
        a.model_gen[0] = -1.0;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_model_gen"));
        let mut a = good.clone();
        a.kind[0] = 257.0;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_kind"));
        let mut a = good.clone();
        a.kind[0] = 0.5;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_kind"));
        assert!(AuditHistory::from_arrays(&good, 0, Unrecorded::default()).unwrap_err().contains("recorded"));
        assert!(AuditHistory::from_arrays(&good, 1, Unrecorded::default()).is_ok());
    }

    /// A [`point`] carrying the three trust-tightening fields (plan 2026-11-19 Task 1).
    fn point_ext(n: usize, gen: u64, fit_id: Option<u64>, flags: u8, epi: Option<f64>) -> AuditPoint {
        AuditPoint { fit_id, flags: PointFlags(flags), energy_std_epistemic: epi, ..point(n, gen) }
    }

    /// The standalone bytes of a history's arrays (the writer's own path, for hand-altered arrays).
    fn bytes_of(a: &HistoryArrays, recorded: u64) -> Vec<u8> {
        encode_arrays(a, recorded, Unrecorded::default())
    }

    /// Trust tightening Task 1: the fit id, the flags word (every bit, alone and together) and the
    /// epistemic energy std round-trip through the arrays and the standalone encoding, Some and None
    /// alike, bit for bit; a history that carries any of them is written under the version-2 magic
    /// with the three arrays for every point (−1 for an absent id or std), one that carries none is
    /// written as version 1, as before.
    #[test]
    fn the_new_fields_round_trip() {
        assert_eq!(MAGIC, b"ASAUDIT2");
        assert_eq!(MAGICS, [b"ASAUDIT1", b"ASAUDIT2"]);
        assert_eq!((HistoryArrays::FIT, HistoryArrays::FLAGS, HistoryArrays::ENERGY_STD_EPI), ("audit_fit", "audit_flags", "audit_energy_std_epistemic"));
        let mut h = AuditHistory::default();
        for f in 0..=PointFlags::ALL {
            let fit = (f % 2 == 0).then_some(1000 * f as u64);
            let epi = (f % 3 != 0).then_some(1.25e-4 * f as f64);
            h.push(point_ext(2 + f as usize % 2, f as u64, fit, f, epi));
        }
        // fit 0 and a zero std are values, not absent; the largest id; the smallest positive std
        h.push(point_ext(1, 9, Some(0), 0, Some(0.0)));
        h.push(point_ext(1, 10, Some((1u64 << 53) - 1), PointFlags::ALL, Some(f64::MIN_POSITIVE)));
        let a = h.to_arrays();
        assert_eq!((a.fit.len(), a.flags.len(), a.energy_std_epistemic.len()), (h.len(), h.len(), h.len()));
        assert_eq!(&a.fit[..3], &[0.0, -1.0, 2000.0]);
        assert_eq!(&a.energy_std_epistemic[..2], &[-1.0, 1.25e-4]);
        assert_eq!(AuditHistory::from_arrays(&a, h.recorded(), h.unrecorded).unwrap(), h);
        let enc = h.encode();
        assert_eq!(&enc[..8], MAGIC);
        let back = AuditHistory::decode(&enc).unwrap();
        assert_eq!(back, h);
        for (p, q) in back.iter().zip(h.iter()) {
            assert_eq!(p.energy_std_epistemic.map(f64::to_bits), q.energy_std_epistemic.map(f64::to_bits));
            assert_eq!((p.fit_id, p.flags), (q.fit_id, q.flags));
        }
        assert_eq!(back.iter().map(|p| p.flags.0).take(8).collect::<Vec<_>>(), (0..=7).collect::<Vec<u8>>());
        // the bits
        let f = PointFlags(PointFlags::TICK | PointFlags::CADENCE_FRESH);
        assert!(f.has(PointFlags::TICK) && f.has(PointFlags::CADENCE_FRESH) && !f.has(PointFlags::FENCED));
        assert_eq!((PointFlags::TICK, PointFlags::FENCED, PointFlags::CADENCE_FRESH, PointFlags::ALL), (1, 2, 4, 7));
        assert_eq!(PointFlags::default(), PointFlags(0));
        // none of the three anywhere: version 1, the three arrays absent
        let mut old = AuditHistory::default();
        old.push(point(2, 1));
        old.push(point(3, 2));
        assert!(old.to_arrays().fit.is_empty() && old.to_arrays().flags.is_empty() && old.to_arrays().energy_std_epistemic.is_empty());
        assert_eq!(&old.encode()[..8], MAGIC_V1);
        assert_eq!(AuditHistory::decode(&old.encode()).unwrap(), old);
        // one field anywhere puts the three arrays in, for every point
        for extra in [point_ext(1, 3, Some(7), 0, None), point_ext(1, 3, None, PointFlags::FENCED, None), point_ext(1, 3, None, 0, Some(3e-6))] {
            let mut mixed = old.clone();
            mixed.push(extra.clone());
            let a = mixed.to_arrays();
            assert_eq!(a.fit, vec![-1.0, -1.0, extra.fit_id.map_or(-1.0, |i| i as f64)]);
            assert_eq!(a.flags, vec![0.0, 0.0, extra.flags.0 as f64]);
            assert_eq!(a.energy_std_epistemic, vec![-1.0, -1.0, extra.energy_std_epistemic.unwrap_or(-1.0)]);
            let enc = mixed.encode();
            assert_eq!(&enc[..8], MAGIC);
            assert_eq!(AuditHistory::decode(&enc).unwrap(), mixed);
        }
        // an empty history is version 1 (nothing to carry)
        assert_eq!(&AuditHistory::default().encode()[..8], MAGIC_V1);
    }

    /// Trust tightening Task 1: a version-1 file — made by hand here, not by this writer — reads with
    /// every point's fit id absent, flags 0 and no epistemic energy std, its other fields intact,
    /// with or without its species array; rewritten it is the same bytes (nothing new to carry). A
    /// version-1 file with the three arrays after its species is refused: version 1 ends there.
    #[test]
    fn a_legacy_history_reads_with_absent_fields() {
        let arr = |b: &mut Vec<u8>, v: &[f64]| {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
        };
        let mut ten = b"ASAUDIT1".to_vec();
        for c in [3u64, 1, 0, 0, 2] {
            ten.extend_from_slice(&c.to_le_bytes());
        }
        for v in [&[2.0][..], &[5.0], &[6.0], &[1e-5], &[-2e-6], &[1.0, 2.0], &[1e-3, 2e-3], &[5e-4, 6e-4], &[2e-4, 3e-4], &[0.01, 0.02]] {
            arr(&mut ten, v);
        }
        let mut b = ten.clone();
        arr(&mut b, &[2.0, 1.0, 1.0, 8.0, 8.0, 2.0, 1.0, 6.0]); // m = 2: H{O}, O{C, H}
        for bytes in [&b, &ten] {
            let h = AuditHistory::decode(bytes).unwrap();
            assert_eq!((h.len(), h.recorded(), h.unrecorded), (1, 3, Unrecorded { no_variance: 1, box_affected: 0, draft: 0, unfit: 2 }));
            let p = h.iter().next().unwrap();
            assert_eq!((p.fit_id, p.flags, p.energy_std_epistemic), (None, PointFlags(0), None));
            assert_eq!((p.kind, p.n_atoms, p.model_gen, p.classes.clone()), (PointKind::Fallback { rule: 5 }, 2, 5, vec![1, 2]));
            assert_eq!((p.pred_energy_std, p.err_energy, p.teacher_force.clone()), (1e-5, -2e-6, vec![0.01, 0.02]));
            assert!(h.to_arrays().fit.is_empty() && &h.encode()[..8] == MAGIC_V1, "nothing new to carry");
        }
        assert_eq!(AuditHistory::decode(&b).unwrap().iter().next().unwrap().species, vec![(1, vec![8]), (8, vec![1, 6])]);
        assert_eq!(AuditHistory::decode(&b).unwrap().encode(), b, "rewritten bit for bit as version 1");
        let mut ext = b.clone();
        for v in [[-1.0], [0.0], [-1.0]] {
            arr(&mut ext, &v);
        }
        assert!(AuditHistory::decode(&ext).unwrap_err().to_string().contains("trailing bytes"), "version 1 ends after the species");
    }

    /// Trust tightening Task 1: hostile values in the three new arrays are refused, through the
    /// arrays and through the standalone bytes, naming the array — never a panic: a wrong length, a
    /// non-integral fit id or flags word, an id of 2⁵³ or more, a negative id other than −1, a flags
    /// word above 7 or negative, a negative (other than −1), NaN or infinite energy std; a version-2
    /// file that ends before them, or whose length prefix runs past the end.
    #[test]
    fn hostile_new_arrays_are_rejected() {
        let mut h = AuditHistory::default();
        h.push(point_ext(2, 1, Some(4), 5, Some(2e-5)));
        h.push(point_ext(1, 2, None, 0, None));
        let good = h.to_arrays();
        type Bad = (&'static str, fn(&mut HistoryArrays));
        let cases: [Bad; 22] = [
            ("audit_fit", |a| {
                a.fit.pop();
            }),
            ("audit_flags", |a| a.flags.push(0.0)),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic.clear()),
            ("audit_fit", |a| a.fit.clear()),
            ("audit_fit", |a| a.fit[0] = 1.5),
            ("audit_fit", |a| a.fit[0] = (1u64 << 53) as f64),
            ("audit_fit", |a| a.fit[0] = 1e300),
            ("audit_fit", |a| a.fit[0] = -2.0),
            ("audit_fit", |a| a.fit[0] = -0.5),
            ("audit_fit", |a| a.fit[0] = f64::NAN),
            ("audit_fit", |a| a.fit[0] = f64::INFINITY),
            ("audit_flags", |a| a.flags[0] = 8.0),
            ("audit_flags", |a| a.flags[0] = 255.0),
            ("audit_flags", |a| a.flags[0] = 0.5),
            ("audit_flags", |a| a.flags[0] = -1.0),
            ("audit_flags", |a| a.flags[0] = f64::NAN),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic[0] = -0.5),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic[0] = -2.0),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic[0] = f64::NAN),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic[1] = f64::NAN),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic[0] = f64::INFINITY),
            ("audit_energy_std_epistemic", |a| a.energy_std_epistemic[0] = f64::NEG_INFINITY),
        ];
        for (i, (name, bad)) in cases.iter().enumerate() {
            let mut a = good.clone();
            bad(&mut a);
            let e = AuditHistory::from_arrays(&a, 2, Unrecorded::default()).expect_err(&format!("case {i}"));
            assert!(e.contains(name), "case {i}: {e}");
            let e = AuditHistory::decode(&bytes_of(&a, 2)).expect_err(&format!("case {i} (bytes)")).to_string();
            assert!(e.contains(name), "case {i} (bytes): {e}");
        }
        assert!(AuditHistory::from_arrays(&good, 2, Unrecorded::default()).is_ok());
        // the bytes: a version-2 file that ends after the species, or after one of the three
        let enc = h.encode();
        let tail = |k: usize| 8 * k + 8 * 2 * k; // k of the three arrays, 2 points each
        for k in 1..=3 {
            let e = AuditHistory::decode(&enc[..enc.len() - tail(k)]).unwrap_err().to_string();
            assert!(e.contains("ends before array"), "{k}: {e}");
        }
        // a length prefix past the end (checked before anything is allocated)
        let mut huge = enc.clone();
        let at = enc.len() - (8 + 8 * 2);
        huge[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(AuditHistory::decode(&huge).unwrap_err().to_string().contains("audit_energy_std_epistemic"));
        // and the version-2 magic is held to its own layout: the species array is not optional
        let mut v2_ten = h.encode();
        v2_ten.truncate(enc.len() - tail(3) - (8 + 8 * h.to_arrays().species.len()));
        assert!(AuditHistory::decode(&v2_ten).is_err());
    }

    /// Review Focus 1, the history's half: a file whose magic is a later version (or any other
    /// `ASAUD…` magic this build does not read) is refused as a newer build's, never as damage, so
    /// the collection keeps it (`collection::tests::a_newer_history_is_never_overwritten_by_this_reader`).
    #[test]
    fn a_newer_history_is_refused_as_newer() {
        let next = format!("ASAUDIT{}", MAGICS.len() + 1);
        let mut bytes = next.clone().into_bytes();
        bytes.extend_from_slice(&[0u8; 64]);
        assert!(written_by_newer_build(&bytes));
        assert!(written_by_newer_build(b"ASAUD010 and more"));
        let e = AuditHistory::decode(&bytes).unwrap_err().to_string();
        assert!(e.contains("newer build") && e.contains(&next), "{e}");
        for own in MAGICS {
            let mut b = own.to_vec();
            b.extend_from_slice(&[0u8; 64]);
            assert!(!written_by_newer_build(&b), "{}", String::from_utf8_lossy(own));
        }
        assert!(!written_by_newer_build(b"ASMODEL9 a model"));
        assert!(!written_by_newer_build(b"ASAUDIT"), "short");
        assert!(!written_by_newer_build(b"garbage of some length"));
        assert!(!AuditHistory::decode(b"ASAUDIT2 torn").unwrap_err().to_string().contains("newer"), "this build's own magic, damaged");
    }

    #[test]
    #[should_panic(expected = "flags word 8")]
    fn a_flags_word_above_7_is_a_programming_error_at_the_push() {
        AuditHistory::default().push(point_ext(1, 0, None, 8, None));
    }

    #[test]
    #[should_panic(expected = "epistemic energy std")]
    fn a_nan_epistemic_energy_std_is_a_programming_error_at_the_push() {
        AuditHistory::default().push(point_ext(1, 0, None, 0, Some(f64::NAN)));
    }

    #[test]
    #[should_panic(expected = "class code 3")]
    fn a_bad_class_code_is_a_programming_error_at_the_push() {
        let mut h = AuditHistory::default();
        let mut p = point(1, 0);
        p.classes[0] = 3;
        h.push(p);
    }
}
