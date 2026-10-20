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
//! so for a calibrated posterior z_j = err_force[j] / pred_force_std[j] has E[z_j²] = 1 (z_j² is
//! χ²₃/3 when the three component stds are equal) and z_E = err_energy / pred_energy_std is a
//! standard normal — what a calibration measures. The epistemic part is kept beside the total
//! because rule 6 (`providers::surrogate`) judges the *epistemic* std against its bound, while the
//! aleatoric floor 3σ_f² is a property of the fit, not of the geometry: a calibration that inflates
//! the epistemic part alone needs the two apart, and σ_f at the time of the audit is recoverable
//! from them (σ_f² = (pred_force_std² − epistemic²)/3). Non-finite values are kept as they were
//! judged (a NaN gap is a failed audit, `providers::surrogate::audit`); a calibration filters them.
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
//! # What is not recorded
//!
//! A judged audit always leaves a point when the model carried a variance factor; the
//! [`Unrecorded`] counters say how many judged audits left none and why: the model had no
//! variance factor (`Prediction::force_std` was `None`), the teacher's answer was box-affected
//! (the verdict `BoxAffected`: its gap is the box's confinement as much as the model's error, and
//! a calibration must not learn the box), or the audit was a draft-mode guess at a geometry strict
//! mode refuses (not a verdict on the strict model). Audits that are not judged at all (discarded
//! after a settings change, not judged inside a fenced shell band) are not counted here: they
//! have their activity events.
//!
//! # Standalone encoding ([`AuditHistory::encode`])
//!
//! ```text
//! "ASAUDIT1"                      8 bytes, magic (the version)
//! recorded                        u64 LE, every point ever pushed (len() + the evicted)
//! no_variance, box_affected, draft   u64 LE each, the Unrecorded counters
//! array₁ … array₈                 each u64 LE length n, then n f64 LE values — the eight
//!                                 arrays of [`HistoryArrays::NAMES`] in that order
//! (end — trailing bytes are an error)
//! ```
//!
//! The arrays, with M points and A = Σ atoms over them: `audit_atoms`, `audit_model_gen`,
//! `audit_energy_std`, `audit_err_energy` (M each; counts and generations as exact integral f64),
//! `audit_classes`, `audit_force_std`, `audit_force_std_epistemic`, `audit_err_force` (A each;
//! class codes 0, 1, 2 as integral f64). The sidecar embeds the same eight arrays after its
//! sample tail under a header key `audit_history` that carries the five counters
//! ([`crate::sidecar`], "Version 7"), so both forms share [`AuditHistory::to_arrays`] /
//! [`AuditHistory::from_arrays`]. Every length is checked against the bytes present before
//! anything is allocated; a decode never panics on a corrupt input.

use std::collections::VecDeque;
use std::io;

/// Points the ring keeps (the plan's figure): at one audit per 50 answered steps and ≈ 3 fs/s of
/// DFT-paced audits this is hours of audits; at the surrogate's own rate it is the last few
/// thousand.
pub const HISTORY_CAP: usize = 4096;

/// The first eight bytes of a standalone history file.
pub const MAGIC: &[u8; 8] = b"ASAUDIT1";

/// Largest atom count a point may declare (a hostile-input guard, far above any scene).
const MAX_ATOMS: f64 = (1u64 << 24) as f64;
/// Largest model generation accepted (exact in f64).
const MAX_GEN: f64 = (1u64 << 53) as f64;

/// One judged audit: the model's prediction against the teacher's answer (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct AuditPoint {
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
    /// Ha: E_teacher − E_model (signed).
    pub err_energy: f64,
    pub n_atoms: usize,
    /// The provider's model generation that was judged (`SurrogateKs::model_gen`).
    pub model_gen: u64,
}

impl AuditPoint {
    /// The shapes agree: every per-atom array has `n_atoms` entries and every class code is 0–2.
    fn well_formed(&self) -> Result<(), String> {
        let n = self.n_atoms;
        if n == 0 {
            return Err("an audit point of no atoms".into());
        }
        for (name, len) in [("classes", self.classes.len()), ("pred_force_std", self.pred_force_std.len()), ("pred_force_std_epistemic", self.pred_force_std_epistemic.len()), ("err_force", self.err_force.len())] {
            if len != n {
                return Err(format!("audit point: {name} has {len} entries for {n} atoms"));
            }
        }
        if let Some(c) = self.classes.iter().find(|&&c| c > 2) {
            return Err(format!("audit point: class code {c} (0 isolated, 1 pair, 2 many-body)"));
        }
        Ok(())
    }
}

/// Judged audits that left no point, by reason (module doc, "What is not recorded").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unrecorded {
    /// The model had no variance factor when the audit was judged.
    pub no_variance: u64,
    /// The teacher's answer was box-affected (verdict `BoxAffected`).
    pub box_affected: u64,
    /// A draft-mode guess at a geometry strict mode refuses.
    pub draft: u64,
}

/// The ring of the newest [`HISTORY_CAP`] audit points, oldest first (module doc).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AuditHistory {
    ring: VecDeque<AuditPoint>,
    recorded: u64,
    pub unrecorded: Unrecorded,
}

/// The eight arrays of the history's serialised form (module doc, "Standalone encoding").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HistoryArrays {
    pub atoms: Vec<f64>,
    pub model_gen: Vec<f64>,
    pub energy_std: Vec<f64>,
    pub err_energy: Vec<f64>,
    pub classes: Vec<f64>,
    pub force_std: Vec<f64>,
    pub force_std_epistemic: Vec<f64>,
    pub err_force: Vec<f64>,
}

impl HistoryArrays {
    /// The arrays' names in serialised order (the first four are per point, the last four per atom).
    pub const NAMES: [&'static str; 8] = ["audit_atoms", "audit_model_gen", "audit_energy_std", "audit_err_energy", "audit_classes", "audit_force_std", "audit_force_std_epistemic", "audit_err_force"];

    /// The arrays in [`NAMES`](Self::NAMES) order.
    pub fn as_slices(&self) -> [(&'static str, &[f64]); 8] {
        [
            (Self::NAMES[0], &self.atoms),
            (Self::NAMES[1], &self.model_gen),
            (Self::NAMES[2], &self.energy_std),
            (Self::NAMES[3], &self.err_energy),
            (Self::NAMES[4], &self.classes),
            (Self::NAMES[5], &self.force_std),
            (Self::NAMES[6], &self.force_std_epistemic),
            (Self::NAMES[7], &self.err_force),
        ]
    }

    /// The arrays from eight vectors in [`NAMES`](Self::NAMES) order.
    pub fn from_vecs(v: [Vec<f64>; 8]) -> HistoryArrays {
        let [atoms, model_gen, energy_std, err_energy, classes, force_std, force_std_epistemic, err_force] = v;
        HistoryArrays { atoms, model_gen, energy_std, err_energy, classes, force_std, force_std_epistemic, err_force }
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

    /// The eight arrays of the serialised form (module doc).
    pub fn to_arrays(&self) -> HistoryArrays {
        let mut a = HistoryArrays::default();
        for p in &self.ring {
            a.atoms.push(p.n_atoms as f64);
            a.model_gen.push(p.model_gen as f64);
            a.energy_std.push(p.pred_energy_std);
            a.err_energy.push(p.err_energy);
            a.classes.extend(p.classes.iter().map(|&c| c as f64));
            a.force_std.extend_from_slice(&p.pred_force_std);
            a.force_std_epistemic.extend_from_slice(&p.pred_force_std_epistemic);
            a.err_force.extend_from_slice(&p.err_force);
        }
        a
    }

    /// The history from its arrays and counters (module doc): the per-point arrays must agree in
    /// length, the counts be integral and sum to the per-atom arrays' length, the generations
    /// integral, the class codes 0, 1 or 2. More points than [`HISTORY_CAP`] keep the newest.
    pub fn from_arrays(a: &HistoryArrays, recorded: u64, unrecorded: Unrecorded) -> Result<AuditHistory, String> {
        let m = a.atoms.len();
        for (name, len) in [("audit_model_gen", a.model_gen.len()), ("audit_energy_std", a.energy_std.len()), ("audit_err_energy", a.err_energy.len())] {
            if len != m {
                return Err(format!("{name} has {len} entries for {m} audit points (audit_atoms)"));
            }
        }
        let mut total = 0usize;
        for (i, &n) in a.atoms.iter().enumerate() {
            if !(n.fract() == 0.0 && n >= 1.0 && n <= MAX_ATOMS) {
                return Err(format!("audit_atoms[{i}] = {n} is not a positive atom count"));
            }
            total = total.checked_add(n as usize).ok_or("audit_atoms overflow")?;
        }
        for (name, len) in [("audit_classes", a.classes.len()), ("audit_force_std", a.force_std.len()), ("audit_force_std_epistemic", a.force_std_epistemic.len()), ("audit_err_force", a.err_force.len())] {
            if len != total {
                return Err(format!("{name} has {len} entries, audit_atoms sums to {total}"));
            }
        }
        for (i, &g) in a.model_gen.iter().enumerate() {
            if !(g.fract() == 0.0 && g >= 0.0 && g <= MAX_GEN) {
                return Err(format!("audit_model_gen[{i}] = {g} is not a generation"));
            }
        }
        if let Some((i, &c)) = a.classes.iter().enumerate().find(|(_, &c)| !(c == 0.0 || c == 1.0 || c == 2.0)) {
            return Err(format!("audit_classes[{i}] = {c} is not 0, 1 or 2"));
        }
        let mut h = AuditHistory { ring: VecDeque::with_capacity(m.min(HISTORY_CAP)), recorded: 0, unrecorded };
        let mut at = 0usize;
        for i in 0..m {
            let n = a.atoms[i] as usize;
            let r = at..at + n;
            h.push(AuditPoint {
                classes: a.classes[r.clone()].iter().map(|&c| c as u8).collect(),
                pred_force_std: a.force_std[r.clone()].to_vec(),
                pred_force_std_epistemic: a.force_std_epistemic[r.clone()].to_vec(),
                pred_energy_std: a.energy_std[i],
                err_force: a.err_force[r].to_vec(),
                err_energy: a.err_energy[i],
                n_atoms: n,
                model_gen: a.model_gen[i] as u64,
            });
            at += n;
        }
        if recorded < h.ring.len() as u64 {
            return Err(format!("recorded {recorded} is below the {} points present", h.ring.len()));
        }
        h.recorded = recorded;
        Ok(h)
    }

    /// The standalone form (module doc, "Standalone encoding").
    pub fn encode(&self) -> Vec<u8> {
        let a = self.to_arrays();
        let slices = a.as_slices();
        let body: usize = slices.iter().map(|(_, v)| 8 + 8 * v.len()).sum();
        let mut out = Vec::with_capacity(8 + 4 * 8 + body);
        out.extend_from_slice(MAGIC);
        for c in [self.recorded, self.unrecorded.no_variance, self.unrecorded.box_affected, self.unrecorded.draft] {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for (_, v) in slices {
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Reads the standalone form; `InvalidData` names what is wrong (never a panic).
    pub fn decode(bytes: &[u8]) -> io::Result<AuditHistory> {
        Self::decode_checked(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("audit history: {e}")))
    }

    fn decode_checked(bytes: &[u8]) -> Result<AuditHistory, String> {
        if bytes.len() < 8 + 4 * 8 {
            return Err(format!("{} bytes is too short for a history", bytes.len()));
        }
        if &bytes[..8] != MAGIC {
            return Err("not an audit history (magic is not ASAUDIT1)".into());
        }
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
        let (recorded, no_variance, box_affected, draft) = (u64_at(8), u64_at(16), u64_at(24), u64_at(32));
        let mut at = 40usize;
        let mut arrays: Vec<Vec<f64>> = Vec::with_capacity(8);
        for name in HistoryArrays::NAMES {
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
        let a = HistoryArrays::from_vecs(arrays.try_into().expect("eight arrays"));
        Self::from_arrays(&a, recorded, Unrecorded { no_variance, box_affected, draft })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(n: usize, gen: u64) -> AuditPoint {
        AuditPoint {
            classes: (0..n).map(|j| (j % 3) as u8).collect(),
            pred_force_std: vec![1e-3; n],
            pred_force_std_epistemic: vec![5e-4; n],
            pred_energy_std: 1e-5,
            err_force: (0..n).map(|j| 1e-4 * j as f64).collect(),
            err_energy: -2e-6,
            n_atoms: n,
            model_gen: gen,
        }
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
        h.unrecorded.draft = 4;
        let back = AuditHistory::from_arrays(&h.to_arrays(), h.recorded(), h.unrecorded).unwrap();
        assert_eq!(back.len(), 2);
        let (a, b) = (back.iter().next().unwrap(), h.iter().next().unwrap());
        assert!(a.err_energy.is_nan() && a.err_force[1] == f64::INFINITY);
        assert_eq!((a.classes.clone(), a.n_atoms, a.model_gen), (b.classes.clone(), b.n_atoms, b.model_gen));
        assert_eq!(back.unrecorded, h.unrecorded);
        assert_eq!(back.recorded(), 2);
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
        a.classes[1] = 3.0;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_classes"));
        let mut a = good.clone();
        a.model_gen[0] = -1.0;
        assert!(AuditHistory::from_arrays(&a, 1, Unrecorded::default()).unwrap_err().contains("audit_model_gen"));
        assert!(AuditHistory::from_arrays(&good, 0, Unrecorded::default()).unwrap_err().contains("recorded"));
        assert!(AuditHistory::from_arrays(&good, 1, Unrecorded::default()).is_ok());
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
