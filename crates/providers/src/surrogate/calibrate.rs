//! The calibrated class verdict (M8a Task 2; spec §4.1): rule 4b's verdict per environment class
//! from the error the teacher **measured** at the geometries the class verdict decides, with the
//! fit's leave-one-out error as a floor and hysteresis. It replaces the max rule of M6 Task 11
//! round 6 (one sample alone decided its class) wherever evidence exists; without evidence that
//! rule stands unchanged. Pure functions of the audit history, the policy's bounds and the
//! per-class leave-one-out statistics; the hysteresis state is the standing verdict the caller
//! passes in (`SurrogateKs::class_memory`). Atomic units throughout (Hartree, Bohr); every number
//! below is a ratio to the policy's own bounds.
//!
//! # What the verdict asks
//!
//! Rule 4b asks, per environment class c (pair, many-body; [`super::trust::atom_class`]): may the
//! model answer a geometry whose atoms are in c? The claim behind "yes" is the one rule 6 makes
//! per step — the expected error is within the policy's bound. Three measurements bear on it:
//!
//! 1. The **audit history**'s points in this verdict's *jurisdiction* (`surrogate::history`):
//!    audits (every rule passed and the model answered) and the fallbacks rule 4b itself refused
//!    (every other rule passed; only the class verdict stood between the model and an answer) —
//!    the geometries this verdict decides, with the teacher's answer beside the model's. Per atom
//!    the error ‖F_t,j − F_m,j‖ and |F_t,j|, from which the policy's own per-atom bound
//!    b_j = force_bound(|F_t,j|) = min(max(2e-3, 10 % |F|), 1e-2) is formed; per point the energy
//!    gap E_t − E_m.
//! 2. The fit's **leave-one-out residuals** at its training samples (`surrogate::loo`): per fitted
//!    atom of class c the force error of the fit without its sample, ‖r_F,j‖/√3 ÷ b_j, and per
//!    sample the energy error |r_E|/N over the tier. Exact, measured, in-domain.
//! 3. The model's **predictive std** at the audited geometries against the error there: z_j =
//!    ‖ΔF_j‖ / pred_force_std_j, E[z²] = 1 for a calibrated posterior — the spec's "predicted
//!    uncertainty vs measured error", reported as the calibration factor κ_c.
//!
//! # The statistics
//!
//! Per class c, over its jurisdiction points p (weights w_p below; n_c = the number of points):
//!
//! ```text
//! W_c    = √( Σ_p w_p w̄_p² / Σ_p w_p ),   w̄_p² = mean over the point's atoms of class c of (‖ΔF_j‖/√3 / b_j)²     the measured force error, in bound units
//! W_E,c  = √( Σ_p w_p (|ΔE_p|/N_p / tier)² / Σ_p w_p )                                                        the measured energy error, in tier units
//! κ_c    = max(1, ẑ_c) · (1 + 1/√(2 n_c)),   ẑ_c² = Σ_p w_p z̄_p² / Σ_p w_p,  z̄_p² = mean of z_j² over the class atoms   the calibration factor (a readout)
//! R_c    = RMS over the class's fitted atoms of ‖r_F,j‖/√3 / b_j      (ClassLoo::force_rms)       the LOO force error
//! R_E,c  = RMS over the class's samples of |r_E|/N / tier               (ClassLoo::energy_rms)      the LOO energy error
//! V_c    = max( R_c,  W_c · (1 + 1/√(2 n_c)) )                          (ClassLoo::measured for the second term)   the verdict's force value
//! V_E,c  = W_E,c · (1 + 1/√(2 n_c))                                     (ClassLoo::measured_energy)                the verdict's energy value
//! ```
//!
//! **Why the measured error and not κ·R** (the fix round's A.1, the physics review's I1/I2). The
//! first version decided on κ_c · R_c. That product equals the error at visited geometries only if
//! the posterior's std there matches the leave-one-out error at a training point — false for a
//! sparse class, where removing a sample is a high-leverage removal (large R) while a visited point
//! between samples has a small std: V overstated by the leverage ratio (measured: V 1.8–2.6
//! claimed where the test's own checker found 0.25–0.58× the bound). And z has no floor on its
//! denominator: one confident comparison with a modest error (z = 62 among 266 points) set a
//! class's κ for the ring's lifetime — the max rule's knife-edge relocated. The teacher's error in
//! bound units has neither defect: it is the quantity the policy claims, bounded per atom by the
//! error itself over a fixed scale. κ_c is still computed and reported (the Model tab, a later
//! variance inflation) but decides nothing.
//!
//! **One draw per point, not per atom**: the model's ΣF = 0 makes a dimer's two atoms one draw
//! (the Task 1 physics review's I1); w̄_p² is the mean over the point's class atoms. **Weights**:
//! an audit point weighs 1; a class's rule-4b fallback points (both class codes together) weigh
//! together at most max(n_audits, [`MIN_EVIDENCE`]) — spread evenly — so one class refusal
//! repeated over a heating run (3313 near-identical pair points in the baseline) is worth one
//! class's evidence and never outweighs the audits (controller ruling), yet a refused class with
//! no audits can still be calibrated by its own fallbacks. Points refused by rules 5/5b/5d/5e/6/7
//! or the fences are not evidence: those rules refuse whatever the class verdict says, and their
//! errors (z rms 12–54 in the baseline: the posterior extrapolating) argue for *those* rules, not
//! about the model inside their reach — including a class refusal the variance verdict would have
//! refused as well, which `evaluate` codes apart (`super::HISTORY_RULE_CLASS_AND_VARIANCE`). The
//! **inflation** (1 + 1/√(2n)) is one sigma of the estimate's own scatter — the rms of n draws of a
//! χ²₁-like variable (a dimer's error has one effective component; Var = 2) scatters by ≈ 1/√(2n)
//! relative: ×1.41 at 3 points, ×1.14 at 26, ×1.04 at 336. It is applied with n = the number of
//! points (the scatter is a property of the draws), while the weights settle influence (review
//! M1). The energy is judged by its measured error, never by z_E (biased: +1.7 σ_E median at n 26,
//! Task 1 review M5). A comparison that is not a number (NaN or ∞) makes W and κ infinite: the
//! worst evidence, never dropped (the activity Task 3b ruling). R_c stays as the floor — the model
//! is never trusted beyond its own leave-one-out; robust, being an RMS over hundreds of atoms.
//!
//! # The verdict, with hysteresis
//!
//! A class with fewer than Q′ samples is untrusted (`ClassKey::Samples`, as before). Otherwise:
//!
//! - **calibrated** (n_c ≥ [`MIN_EVIDENCE`] = 3 points): untrusted → trusted when V_c ≤ [`ENTER`]
//!   = 1/√2 and V_E,c ≤ ENTER; trusted → stays while V_c ≤ [`LEAVE`] = 1 and V_E,c ≤ LEAVE. The
//!   class is never trusted with its measured rms error beyond the bound — the tolerance rules 6
//!   and 8 use — and enters only once that error is clearly inside: the band is a factor √2 in
//!   error (2 in variance), equal to the estimate's one-sigma scatter at n = 3 and wider from n = 4
//!   (review M3), so refit-to-refit jitter of W and R does not flip the verdict.
//! - **uncalibrated** (fewer than 3 points): the rule of before, unchanged — LOO force max ≤ 1 and
//!   rms ≤ 1, LOO energy max ≤ 1 (`super::trust::class_refusal`) — **plus one guard the evidence
//!   gives even when thin** (review I4): a class is refused (does not enter, and leaves) while any
//!   jurisdiction point has an atom's error beyond [`WORST_ATOM`] = 3× its bound — for a
//!   calibrated posterior at std = bound, P(χ²₃/3 > 9) = 5.9e-6 per atom, so such an error is a
//!   posterior that is wrong, not unlucky (the repro test's own derivation). Without evidence
//!   nothing is loosened; the fallbacks a refused class produces are themselves evidence (rule-4b
//!   points), so a class the max rule refuses for one outlier earns the calibrated verdict after
//!   three teacher steps of its own and a refit. A class whose only recent comparisons were
//!   box-affected has no evidence from them (they leave no point): unverified, never passing by
//!   calibration.
//!
//! Three is the smallest count from which an rms can be told from a single draw — the count the
//! audit burst already uses after a cleared failure ([`super::AUDIT_BURST`]). The hysteresis state
//! is the verdict that stood when the refit, install or policy change began (`class_memory`,
//! committed once it is done) and the generation it was entered at (`ClassLoo::trusted_since`);
//! it is reset by a forget or an installed model (another model's verdict is no memory for this
//! one — so a saved model with a verdict inside the band re-enters on load: review M2, documented,
//! conservative). Numbers of record: `validate/references.md`, "M8a calibrated verdict".

use sim::activity::{EnvClass, TeachRule};
use sim::ClassKey;
use surrogate::history::{AuditHistory, PointKind};

use super::FallbackPolicy;

/// An untrusted class becomes trusted when its verdict values (force and energy) are at most
/// this: 1/√2 — half the bound's variance.
pub const ENTER: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// A trusted class stays trusted while its verdict values are at most this: the bound itself.
pub const LEAVE: f64 = 1.0;
/// Evidence points (audits plus the capped rule-4b fallback weight) a class needs before the
/// calibrated verdict judges it; below it the rule of before stands, with the [`WORST_ATOM`] guard.
pub const MIN_EVIDENCE: usize = 3;
/// With fewer than [`MIN_EVIDENCE`] points, one jurisdiction point with an atom's measured force
/// error beyond this many times its bound refuses the class (review I4): P(χ²₃/3 > 9) = 5.9e-6 per
/// atom for a calibrated posterior at std = bound.
pub const WORST_ATOM: f64 = 3.0;

const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// One class's calibration from the history (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassCalibration {
    /// κ_c: measured over predicted, ≥ 1 (1 without any evidence; ∞ when a comparison in the
    /// evidence was not a number). A readout; it decides nothing.
    pub factor: f64,
    /// Audit points of the class in the evidence.
    pub audits: usize,
    /// Rule-4b fallback points of the class in the evidence (before the weight cap).
    pub fallbacks: usize,
    /// n_c = Σw: the audits plus the capped fallback weight, in points.
    pub evidence: f64,
    /// The jurisdiction points of the class (audits + fallbacks): what the inflation counts.
    pub points: usize,
    /// ẑ_c before the floor and the inflation (NaN without evidence).
    pub z_rms: f64,
    /// W_c · (1 + 1/√(2n)): the measured force error in bound units, inflated (NaN without
    /// evidence) — the verdict's force value beside the LOO floor.
    pub measured: f64,
    /// W_E,c · (1 + 1/√(2n)): the measured energy error in tier units, inflated (NaN without
    /// evidence).
    pub measured_energy: f64,
    /// The largest per-atom measured force error over the jurisdiction points, in bound units
    /// (the [`WORST_ATOM`] guard's input; 0 without evidence).
    pub worst_atom: f64,
}

impl ClassCalibration {
    /// Enough evidence for the calibrated verdict ([`MIN_EVIDENCE`]).
    pub fn calibrated(&self) -> bool {
        self.evidence >= MIN_EVIDENCE as f64
    }
}

/// The calibration of both classes (indexed like `atom_class` / `ENV_CLASSES`: 0 pair,
/// 1 many-body).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibration {
    pub per_class: [ClassCalibration; 2],
}

/// The per-class leave-one-out statistics the verdict reads (what
/// `SurrogateKs::compute_class_loo_full` measures): ratios to the policy's bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassStats {
    pub samples: usize,
    pub force_rms: f64,
    pub force_max: f64,
    pub energy_rms: f64,
    pub energy_max: f64,
}

/// The verdict for one class.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassVerdict {
    Trusted,
    /// The first failing condition (its key, value and limit — `ClassLoo::refusal`) and the reason
    /// in the refusal's words (the part after "isn't learned well enough yet (<class>: ").
    Untrusted { key: ClassKey, value: f64, limit: f64, reason: String },
}

/// A history point in the verdict's jurisdiction: an audit, or a fallback refused by rule 4b.
fn in_jurisdiction(kind: PointKind) -> bool {
    match kind {
        PointKind::Audit => true,
        PointKind::Fallback { rule } => {
            rule == super::history_rule_code(TeachRule::Class(EnvClass::Pair)) || rule == super::history_rule_code(TeachRule::Class(EnvClass::ManyBody))
        }
    }
}

/// A non-finite square sum: the worst evidence, never dropped (Task 3b).
fn sq(x: f64) -> f64 {
    if x.is_finite() { x * x } else { f64::INFINITY }
}

/// Per point and class: (w̄², z̄², the energy term, the worst atom ratio).
struct PointEvidence {
    w2: f64,
    z2: f64,
    e2: f64,
    worst: f64,
}

/// The calibration per class from the history and the policy's bounds (module doc).
pub fn calibrate(history: &AuditHistory, policy: &FallbackPolicy) -> Calibration {
    let mut audits: [Vec<PointEvidence>; 2] = [Vec::new(), Vec::new()];
    let mut fallbacks: [Vec<PointEvidence>; 2] = [Vec::new(), Vec::new()];
    let tier = policy.energy_std_per_atom;
    for p in history.iter() {
        if !in_jurisdiction(p.kind) {
            continue;
        }
        for c in 0..2 {
            let code = c as u8 + 1;
            let (mut w2, mut z2, mut worst, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for j in 0..p.n_atoms {
                if p.classes[j] != code {
                    continue;
                }
                let bound = policy.force_bound(p.teacher_force[j]);
                let w = p.err_force[j] / SQRT_3 / bound;
                w2 += sq(w);
                z2 += sq(p.err_force[j] / p.pred_force_std[j]);
                worst = if w.is_finite() { worst.max(w) } else { f64::INFINITY };
                n += 1;
            }
            if n == 0 {
                continue;
            }
            let ev = PointEvidence { w2: w2 / n as f64, z2: z2 / n as f64, e2: sq(p.err_energy.abs() / p.n_atoms.max(1) as f64 / tier), worst };
            match p.kind {
                PointKind::Audit => audits[c].push(ev),
                PointKind::Fallback { .. } => fallbacks[c].push(ev),
            }
        }
    }
    let per_class = std::array::from_fn(|c| {
        let (n_a, n_f) = (audits[c].len(), fallbacks[c].len());
        let cap = n_a.max(MIN_EVIDENCE) as f64;
        // the fallbacks of the class together weigh at most the audits (never fewer than three)
        let w_f = if n_f == 0 { 0.0 } else { (n_f as f64).min(cap) / n_f as f64 };
        let den = n_a as f64 + w_f * n_f as f64;
        if den == 0.0 {
            return ClassCalibration { factor: 1.0, audits: 0, fallbacks: 0, evidence: 0.0, points: 0, z_rms: f64::NAN, measured: f64::NAN, measured_energy: f64::NAN, worst_atom: 0.0 };
        }
        let sum = |f: &dyn Fn(&PointEvidence) -> f64| audits[c].iter().map(f).sum::<f64>() + w_f * fallbacks[c].iter().map(f).sum::<f64>();
        let points = n_a + n_f;
        let inflation = 1.0 + 1.0 / (2.0 * points as f64).sqrt();
        let z_rms = (sum(&|e| e.z2) / den).sqrt();
        let measured = (sum(&|e| e.w2) / den).sqrt() * inflation;
        let measured_energy = (sum(&|e| e.e2) / den).sqrt() * inflation;
        let worst_atom = audits[c].iter().chain(&fallbacks[c]).map(|e| e.worst).fold(0.0f64, |m, w| if w.is_nan() || m.is_nan() { f64::NAN } else { m.max(w) });
        ClassCalibration { factor: z_rms.max(1.0) * inflation, audits: n_a, fallbacks: n_f, evidence: den, points, z_rms, measured, measured_energy, worst_atom }
    });
    Calibration { per_class }
}

impl Calibration {
    /// The verdict for `class` (0 pair, 1 many-body) from its leave-one-out statistics, the fit's
    /// Q′ and the standing verdict `was_trusted` (the hysteresis; module doc, "The verdict").
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
    pub fn class_verdict(&self, class: usize, st: &ClassStats, q_prime: usize, was_trusted: bool) -> ClassVerdict {
        if st.samples < q_prime {
            return ClassVerdict::Untrusted { key: ClassKey::Samples, value: st.samples as f64, limit: q_prime as f64, reason: format!("{} samples < Q′ {q_prime}", st.samples) };
        }
        let cal = self.per_class[class];
        if !cal.calibrated() {
            // the rule of before, unchanged — no evidence loosens nothing — plus the thin-evidence
            // guard: an atom measured beyond 3× its bound is a posterior that is wrong (review I4)
            if !(cal.worst_atom <= WORST_ATOM) {
                return ClassVerdict::Untrusted {
                    key: ClassKey::Calibrated,
                    value: cal.worst_atom,
                    limit: WORST_ATOM,
                    reason: format!("a DFT check found an atom's force {:.1}× its bound (limit {WORST_ATOM:.0}× while unverified: {} of {MIN_EVIDENCE} comparisons), {} samples", cal.worst_atom, cal.points, st.samples),
                };
            }
            return match super::trust::class_refusal(st.samples, q_prime, st.force_rms, st.force_max, st.energy_max) {
                None => ClassVerdict::Trusted,
                Some((ClassKey::Force, value, limit)) => {
                    let energy = if st.energy_max > 1.0 { format!(", energy {:.1}× the tier", st.energy_max) } else { String::new() };
                    ClassVerdict::Untrusted { key: ClassKey::Force, value, limit, reason: format!("LOO force error {:.1}× the bound, RMS {:.2}×{energy}, {} samples; unverified — {} of {MIN_EVIDENCE} comparisons", st.force_max, st.force_rms, st.samples, cal.points) }
                }
                Some((key, value, limit)) => ClassVerdict::Untrusted { key, value, limit, reason: format!("LOO energy error {:.1}× the tier, {} samples; unverified — {} of {MIN_EVIDENCE} comparisons", st.energy_max, st.samples, cal.points) },
            };
        }
        let limit = if was_trusted { LEAVE } else { ENTER };
        let gate = if was_trusted { "stays while ≤" } else { "enters at ≤" };
        let evidence = format!("{} audits and {} class fallbacks", cal.audits, cal.fallbacks);
        // the force: the measured error with the leave-one-out as the floor (NaN fails either way)
        let v = if cal.measured.is_nan() || st.force_rms.is_nan() { f64::NAN } else { cal.measured.max(st.force_rms) };
        if !(v <= limit) {
            return ClassVerdict::Untrusted {
                key: ClassKey::Calibrated,
                value: v,
                limit,
                reason: format!(
                    "measured error {v:.2}× the bound over {evidence} (DFT rms {:.2}×, leave-one-out RMS {:.2}×; the posterior {:.1}× too confident; {gate} {limit:.2}), {} samples",
                    cal.measured, st.force_rms, cal.factor, st.samples
                ),
            };
        }
        if !(cal.measured_energy <= limit) {
            return ClassVerdict::Untrusted {
                key: ClassKey::Energy,
                value: cal.measured_energy,
                limit,
                reason: format!("measured energy error {:.2}× the tier over {evidence} ({gate} {limit:.2}), {} samples", cal.measured_energy, st.samples),
            };
        }
        ClassVerdict::Trusted
    }
}
