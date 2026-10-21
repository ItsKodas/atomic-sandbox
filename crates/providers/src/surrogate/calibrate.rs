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
//! # The evidence window (fix round 2: evidence ages by model generation)
//!
//! A history point is a measurement of the model that judged it (`AuditPoint::model_gen`). Once a
//! refit has taught the model at the geometries a point measured — every fallback and audit answer
//! is a training sample — that point describes a model that no longer answers, and the ring would
//! otherwise keep it for 4096 points: a class refused on its audits receives only rule-4b
//! fallbacks afterwards, capped at the audits' weight, so perfect fallbacks of every later model
//! could never bring W below the refusing audits' share (the task review's I4: 8 audits at 1.46×
//! leave W ≥ 1.03 > LEAVE), and one 3× atom would block the entry guard below for the ring's
//! lifetime. So per class the evidence is a **window of whole generations**: walking the class's
//! jurisdiction points from the newest generation back, generations are taken until they hold at
//! least [`WINDOW_POINTS`] = 300 points; older points are aged out (`ClassCalibration::aged`).
//! With fewer than 300 points in the ring nothing ages. A generation is never split: the points one
//! model produced are judged together.
//!
//! What this does, both ways: evidence produced by a model **does not age while that model
//! stands** — more fallbacks under the refused model are added to the same window as the refusing
//! audits, capped, and cannot outvote them; after a refit (which has the failing geometries in its
//! training set) the newer models' own comparisons accumulate, and once they reach 300 points the
//! verdict computed at the following refit reads them alone. Released after a teaching refit and
//! the newer models' own evidence, blocked until then (`evidence_ages_by_model_generation_both_ways`,
//! `a_class_refused_on_its_evidence_reenters_after_a_teaching_refit`) — and never after more than
//! 300 newer points, where the whole ring kept the refusing points for 4096.
//!
//! Why 300: the window feeds two statistics with two requirements, and the stricter one sets it.
//! (1) The rms must not be noisier than the hysteresis band tolerates: the rms of n χ²₁-like draws
//! scatters by 1/√(2n) relative (below), and two sigma of that inside the band √2 − 1 = 0.414 needs
//! n ≥ 11.7. (2) The worst-atom guard must be able to **see a tail**: a window free of 3× atoms has
//! to make a 3× tail at the rate the audited runs measured unlikely — fix round 1 measured 1 judged
//! answer of 67 beyond 3× the bound (1.5 %) on a many-body class whose rms sat inside the bound;
//! for a tail rate p = 1 % a window of n points shows none with probability 0.99ⁿ, ≤ 5 % from
//! n = 298 → 300. Consecutive MD points are correlated, so the effective n is smaller than the
//! count (the window errs on the side of seeing too little); that is what the hysteresis and the
//! per-step rules 6/8 are for. The first fix-round-2 draft used 12 (requirement 1 alone) and the
//! gated repro measured why that is wrong: the many-body class's bad first rule-4b points (NVE run,
//! per-step rms up to 2.1× the bound) aged out behind 12 benign newer ones, the class entered on 3
//! points, and — the run's audits off — no comparison of its answers ever reached the window, so it
//! answered the hot run (673 steps; 27 judged answers beyond 3× the force bound or the energy tier,
//! up to 5.4× and 3.4e-3 Ha/atom). Generation numbers are monotone within a provider and an install
//! moves the provider's generation past every generation in the installed history
//! (`install_sidecar`), so "newer" is a comparison of numbers.
//!
//! # The statistics
//!
//! Per class c, over its jurisdiction points p in the window (weights w_p below; n_c = the number
//! of points):
//!
//! ```text
//! W_c    = √( Σ_p w_p w̄_p² / Σ_p w_p ),   w̄_p² = mean over the point's atoms of class c of (‖ΔF_j‖/√3 / b_j)²     the measured force error, in bound units
//! W_E,c  = √( Σ_p w_p (|ΔE_p|/N_p / tier)² / Σ_p w_p )                                                        the measured energy error, in tier units
//! κ_c    = max(1, ẑ_c) · (1 + 1/√(2 n_c)),   ẑ_c² = Σ_p w_p z̄_p² / Σ_p w_p,  z̄_p² = mean of z_j² over the class atoms   the calibration factor (a readout)
//! A_c    = max over the window's points and their class atoms of ‖ΔF_j‖/√3 / b_j                       (ClassLoo::worst_atom)      the worst atom
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
//! no audits can still be calibrated by its own fallbacks. The evidence count n_a + min(n_f, cap)
//! is formed in integers (fix round 2, the task review's I1: as n_a + (min(n_f, cap)/n_f)·n_f it
//! came out 2.9999999999999996 at n_f = 47, 94, 147, … and silently dropped the class to the max
//! rule — a refit-time knife-edge from rounding). Points refused by rules 5/5b/5d/5e/6/7 or the
//! fences are not evidence: those rules refuse whatever the class verdict says, and their errors
//! (z rms 12–54 in the baseline: the posterior extrapolating) argue for *those* rules, not about
//! the model inside their reach — including a class refusal the variance verdict would have
//! refused as well, which `evaluate` codes apart (`super::HISTORY_RULE_CLASS_AND_VARIANCE`). The
//! **inflation** (1 + 1/√(2n)) is one sigma of the estimate's own scatter — the rms of n draws of a
//! χ²₁-like variable (a dimer's error has one effective component; Var = 2) scatters by ≈ 1/√(2n)
//! relative: ×1.41 at 3 points, ×1.14 at 26, ×1.04 at 336. It is applied with n = the number of
//! points (the scatter is a property of the draws), while the weights settle influence (review
//! M1). The energy is judged by its measured error, never by z_E (biased: +1.7 σ_E median at n 26,
//! Task 1 review M5). A comparison that is not a number (NaN or ∞) makes W, A and κ infinite: the
//! worst evidence, never dropped (the activity Task 3b ruling). R_c stays as the floor — the model
//! is never trusted beyond its own leave-one-out; robust, being an RMS over hundreds of atoms.
//!
//! # The variance inflation (fix round 3; the controller's ruling on C1)
//!
//! The class verdict judges a class's **rms** error; rules 6/7 judge each step by the posterior's
//! epistemic std. Where the posterior is overconfident (κ 2–12 for the hot many-body class) a
//! trusted class's heavy tail passes rule 6 — answers to 5.4× the bound in fix rounds 1–2. So the
//! posterior's std is recalibrated per class by the error the teacher measured where that std
//! decides (the standard post-hoc variance recalibration: one scale factor on the predictive
//! variance, fitted on held-out comparisons — Levi, Gispan, Giladi & Fetaya, Sensors 22, 5540
//! (2022), "Evaluating and calibrating uncertainty prediction in regression tasks" (not verified);
//! Busk et al., Mach. Learn.: Sci. Technol. 3, 015012 (2022), for molecular models (not verified) —
//! both cited from memory without access to the papers; the method stands on the measurement here,
//! not on them):
//!
//! ```text
//! λ_c = max(1, √( Σ_p max(0, w̄_p² − ā_p²) / Σ_p s̄_p² )) · (1 + 1/√(2 n_c)),   s̄_p² = mean over the point's class atoms of (σ_epi,j/√3 / b_j)²
//!       ā_p² = mean over the point's class atoms of (ā_c / b_j)²,   ā_c = min(σ_F, the class's training force-residual rms per component)
//! rule 6:  λ_c(j) · σ_epi,j/√3 ≤ b_j             for every atom j (its own class's λ; 1 for an isolated atom)
//! rule 7:  max_j λ_c(j) · σ_E,epi/N ≤ tier        (one scale of the posterior covariance Σ: both stds are linear in √Σ)
//! ```
//!
//! **The aleatoric part is removed from the numerator** (M8a Task 5 fix round 1, the review's I2).
//! At a fresh geometry the error the teacher measures has the expectation E[(‖ΔF‖/√3)²] =
//! σ_epi²/3 + σ_F² — the epistemic part *and* the fit's misfit floor σ_F, which no amount of data
//! at that geometry removes and which rule 6 (the epistemic std only) is not about. Without the
//! subtraction even a calibrated posterior got λ² ≈ 1 + σ_F²/s̄² — 24.9 / 18.9 (223 a day earlier)
//! on the user's collection — and λ·s̄ was the total rms error: rule 6 then refused any atom whose
//! epistemic std exceeded a few times the class's typical one, and every fallback it added shrank
//! s̄ and grew λ. Per point the subtraction is clamped at 0 (a point whose error is below the floor
//! is no evidence of overconfidence, not negative evidence). ā_c is min(σ_F, the class's own
//! training residual rms): σ_F is the fit's estimate over all rows, a class that the fit
//! represents better than average must not have more subtracted than its own misfit. Dividing by
//! the total std instead was rejected: it under-inflates a truly overconfident posterior (σ_F
//! dominates the denominator exactly where the epistemic std is too small).
//!
//! over the class's points in the **posterior's jurisdiction** — the verdict's (audits, rule-4b
//! fallbacks) plus the fallbacks rules 6/7 refused alone (codes 14, 15) or with the class (21): the
//! geometries where the std decides. Points of rules 5/5a–5e and the fences are not: those refuse
//! whatever the std says. The same window of generations as the verdict (≥ [`WINDOW_POINTS`]),
//! one draw per point, no weight cap (every comparison is equally a draw of the posterior; the cap
//! protected the verdict's audits from heating floods). The guards the ruling asked for:
//!
//! - **The ratio of sums, not a mean of z²** (the physics review's I2: z = ‖ΔF‖/σ has no floor
//!   on its denominator, and one confident comparison at z = 62 among 266 set κ ≥ 5.4 for the
//!   ring's lifetime). Both sums are in the bound's units, so a point with a tiny std *and* a small
//!   error adds little to either: the same 266 + 1 points give λ_raw = 1.0000 where the mean of z²
//!   gives 3.9 (`the_variance_inflation_is_a_ratio_of_sums_over_the_posteriors_jurisdiction`).
//!   The sums are dominated by the comparisons whose error or std is a sizeable fraction of the
//!   bound — the ones rule 6 decides.
//! - **Only with ≥ [`MIN_EVIDENCE`] points**; below, λ = 1 exactly (the uninflated rule of before:
//!   nothing loosened, nothing tightened without evidence).
//! - **Floored at 1**: never deflated — a posterior measured underconfident keeps its own std
//!   (rule 6 is not loosened by calibration).
//! - **× (1 + 1/√(2n))**: the conservative side of the estimate's one-sigma scatter, as for W.
//! - **No cap**, and why that is safe from a lockout: a large λ only sends steps to the teacher,
//!   and those steps are rule-6/7 points — the factor's own evidence — so λ is re-measured from the
//!   refused geometries themselves as the window moves; a class whose model improves comes back.
//!   A comparison that is not a number makes λ = ∞ (every atom of the class refused) until it ages.
//!
//! λ moves at refits, installs and policy changes, with the verdict (`refresh_class_loo`, which
//! also drops a cached variance verdict). Rule 6's reported uncertainty is the inflated std; the
//! history records the model's own (uninflated) stds, which λ is measured from.
//!
//! # The verdict, with hysteresis
//!
//! A class with fewer than Q′ samples is untrusted (`ClassKey::Samples`, as before). Otherwise:
//!
//! - **The worst-atom guard** ([`WORST_ATOM`] = 3; `ClassKey::WorstAtom`): an untrusted class does
//!   not **enter** while any point in its window has an atom measured beyond 3× its bound — at
//!   every evidence count (fix round 2, the controller's ruling on the entry question); while the
//!   class has fewer than [`MIN_EVIDENCE`] points such an atom also makes a trusted class leave.
//!   For a calibrated posterior at std = bound, P(χ²₃/3 > 9) = 5.9e-6 per atom: such an error is a
//!   posterior that is wrong somewhere the class would answer, not an unlucky draw (the repro
//!   test's own derivation). Why at entry and at every n: the many-body error is heavy-tailed
//!   (refused-step max/rms 4–6 against ≈ 1.9 for a calibrated Gaussian), so an rms over the
//!   benign early points of a refused class — at any n — admitted it before the compact
//!   collisions that followed reached 3–5× (the fix-round-1 measurement). A trusted class with
//!   evidence keeps being judged by its rms (rule 8's audits fail a trusted model's 3× answer per
//!   step and the window then holds the failing audit). The window ages the guard's points too.
//! - **calibrated** (n_c ≥ [`MIN_EVIDENCE`] = 3 points): untrusted → trusted when V_c ≤ [`ENTER`]
//!   = 1/√2 and V_E,c ≤ ENTER; trusted → stays while V_c ≤ [`LEAVE`] = 1 and V_E,c ≤ LEAVE. The
//!   class is never trusted with its measured rms error beyond the bound — the tolerance rules 6
//!   and 8 use — and enters only once that error is clearly inside: the band is a factor √2 in
//!   error (2 in variance), equal to the estimate's one-sigma scatter at n = 3 and wider from n = 4
//!   (review M3), so refit-to-refit jitter of W and R does not flip the verdict. The refusal names
//!   which term decided: the measured error (`ClassKey::Calibrated`), the leave-one-out floor
//!   (`ClassKey::LooFloor`) or the measured energy (`ClassKey::MeasuredEnergy`).
//! - **uncalibrated** (fewer than 3 points): the rule of before, unchanged — LOO force max ≤ 1 and
//!   rms ≤ 1, LOO energy max ≤ 1 (`super::trust::class_refusal`) — plus the worst-atom guard both
//!   ways. Without evidence nothing is loosened; the fallbacks a refused class produces are
//!   themselves evidence (rule-4b points), so a class the max rule refuses for one outlier earns
//!   the calibrated verdict after three teacher steps of its own and a refit. A class whose only
//!   recent comparisons were box-affected has no evidence from them (they leave no point):
//!   unverified, never passing by calibration.
//!
//! Three is the smallest count from which an rms can be told from a single draw — the count the
//! audit burst already uses after a cleared failure ([`super::AUDIT_BURST`]). The hysteresis state
//! is the verdict that stood when the refit, install or policy change began (`class_memory`,
//! committed once it is done) and the generation it was entered at (`ClassLoo::trusted_since`);
//! it is reset by a forget, and **saved with the model** (the sidecar's `class_memory` header key,
//! fix round 2, the task review's M4): a reload, the worker's re-spawn with its carried model and a
//! resumed training keep a verdict that sat inside the band, instead of re-entering. The refusal's
//! words are [`refusal_words`], one function for the verdict and the provider's Log note (fix
//! round 2, the task review's M2). Numbers of record: `validate/references.md`, "M8a calibrated
//! verdict".

use sim::activity::{EnvClass, TeachRule};
use sim::ClassKey;
use std::collections::BTreeMap;
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
/// One jurisdiction point in the window with an atom's measured force error beyond this many
/// times its bound keeps an untrusted class from entering (at every evidence count) and, with
/// fewer than [`MIN_EVIDENCE`] points, makes a trusted one leave: P(χ²₃/3 > 9) = 5.9e-6 per atom
/// for a calibrated posterior at std = bound.
pub const WORST_ATOM: f64 = 3.0;
/// The evidence window (module doc, "The evidence window"): the newest whole generations of a
/// class's jurisdiction points holding at least this many points — enough that a window with no
/// atom beyond [`WORST_ATOM`] makes a 3× tail at 1 % unlikely (0.99ⁿ ≤ 5 %: n ≥ 298), which also
/// keeps two sigma of the rms's own scatter, 2/√(2n), far inside the hysteresis band √2 − 1.
pub const WINDOW_POINTS: usize = 300;

const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// One class's calibration from the history (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassCalibration {
    /// κ_c: measured over predicted, ≥ 1 (1 without any evidence; ∞ when a comparison in the
    /// evidence was not a number). A readout; it decides nothing.
    pub factor: f64,
    /// Audit points of the class in the window.
    pub audits: usize,
    /// Rule-4b fallback points of the class in the window (before the weight cap).
    pub fallbacks: usize,
    /// n_c = Σw: the audits plus the capped fallback weight, in points — an integer, formed
    /// exactly (module doc).
    pub evidence: usize,
    /// The jurisdiction points of the class in the window (audits + fallbacks): what the
    /// inflation counts.
    pub points: usize,
    /// The class's jurisdiction points older than the window (aged out; module doc).
    pub aged: usize,
    /// The oldest model generation in the window, `None` when the window is the whole ring
    /// (fewer than [`WINDOW_POINTS`] points of the class held) or there is no evidence.
    pub window_from: Option<u64>,
    /// ẑ_c before the floor and the inflation (NaN without evidence).
    pub z_rms: f64,
    /// W_c · (1 + 1/√(2n)): the measured force error in bound units, inflated (NaN without
    /// evidence) — the verdict's force value beside the LOO floor.
    pub measured: f64,
    /// W_E,c · (1 + 1/√(2n)): the measured energy error in tier units, inflated (NaN without
    /// evidence).
    pub measured_energy: f64,
    /// A_c: the largest per-atom measured force error over the window's points, in bound units
    /// (the [`WORST_ATOM`] guard's input; 0 without evidence, ∞ for a comparison that is not a
    /// number).
    pub worst_atom: f64,
    /// λ_c, the variance inflation rules 6 and 7 apply to the class's atoms (fix round 3; module
    /// doc, "The variance inflation"): ≥ 1, exactly 1 with fewer than [`MIN_EVIDENCE`] points in
    /// its window, ∞ when a comparison there was not a number.
    pub inflation: f64,
    /// The points behind `inflation` (its own window over the posterior's jurisdiction).
    pub inflation_points: usize,
    /// √(Σ w̄² / Σ s̄²) before the floor at 1 and the scatter factor (NaN with fewer than
    /// [`MIN_EVIDENCE`] points): the measured error over the epistemic std, both in bound units.
    pub inflation_raw: f64,
}

impl ClassCalibration {
    /// Enough evidence for the calibrated verdict ([`MIN_EVIDENCE`]).
    pub fn calibrated(&self) -> bool {
        self.evidence >= MIN_EVIDENCE
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
    /// in the refusal's words ([`refusal_words`]: the part after "isn't learned well enough yet
    /// (<class>: ").
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

/// Per point and class: (w̄², z̄², the energy term, the worst atom ratio, s̄² the epistemic force
/// std in bound units), its generation and kind.
struct PointEvidence {
    gen: u64,
    audit: bool,
    w2: f64,
    z2: f64,
    e2: f64,
    worst: f64,
    s2: f64,
    /// ā̄² = mean over the point's class atoms of (ā_c/b_j)²: the aleatoric part of w̄² (fix round 1)
    a2: f64,
}

/// A history point in the **posterior's** jurisdiction (fix round 3, the variance inflation): the
/// geometries where the model's predictive std decided or would have decided — the class
/// verdict's jurisdiction (audits, rule-4b fallbacks) plus the fallbacks rules 6/7 refused alone
/// (codes 14, 15) and those the class and rules 6/7 refused together (code 21). Not rules
/// 5/5a–5e or the fences: those refuse whatever the std says (the extrapolating posterior's z rms
/// 12–54 there argues for those rules, not for the std inside their reach).
fn in_posterior_jurisdiction(kind: PointKind) -> bool {
    match kind {
        PointKind::Audit => true,
        PointKind::Fallback { rule } => {
            in_jurisdiction(kind)
                || rule == super::history_rule_code(TeachRule::ForceUncertainty)
                || rule == super::history_rule_code(TeachRule::EnergyUncertainty)
                || rule == super::HISTORY_RULE_CLASS_AND_VARIANCE
        }
    }
}

/// The oldest generation of the window (module doc, "The evidence window"): from the newest
/// generation back, whole generations until they hold [`WINDOW_POINTS`]; `None` = every point.
fn window_from(points: &[PointEvidence]) -> Option<u64> {
    let mut per_gen: BTreeMap<u64, usize> = BTreeMap::new();
    for p in points {
        *per_gen.entry(p.gen).or_default() += 1;
    }
    let mut held = 0usize;
    for (&gen, &n) in per_gen.iter().rev() {
        held += n;
        if held >= WINDOW_POINTS {
            return Some(gen);
        }
    }
    None
}

/// The calibration per class from the history and the policy's bounds (module doc), with no
/// aleatoric level removed from the inflation's numerator ([`calibrate_with`] with ā = 0).
pub fn calibrate(history: &AuditHistory, policy: &FallbackPolicy) -> Calibration {
    calibrate_with(history, policy, [0.0; 2])
}

/// [`calibrate`] with each class's aleatoric force level ā_c (Ha/Bohr per component; module doc,
/// "The variance inflation": min(σ_F, the class's training force-residual rms)) removed from the
/// inflation's measured errors.
pub fn calibrate_with(history: &AuditHistory, policy: &FallbackPolicy, aleatoric: [f64; 2]) -> Calibration {
    let mut classes: [Vec<PointEvidence>; 2] = [Vec::new(), Vec::new()];
    let mut posterior: [Vec<PointEvidence>; 2] = [Vec::new(), Vec::new()];
    let tier = policy.energy_std_per_atom;
    for p in history.iter() {
        if !in_posterior_jurisdiction(p.kind) {
            continue;
        }
        let verdict = in_jurisdiction(p.kind);
        for c in 0..2 {
            let code = c as u8 + 1;
            let (mut w2, mut z2, mut worst, mut s2, mut a2, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
            for j in 0..p.n_atoms {
                if p.classes[j] != code {
                    continue;
                }
                let bound = policy.force_bound(p.teacher_force[j]);
                let w = p.err_force[j] / SQRT_3 / bound;
                w2 += sq(w);
                z2 += sq(p.err_force[j] / p.pred_force_std[j]);
                s2 += sq(p.pred_force_std_epistemic[j] / SQRT_3 / bound);
                a2 += sq(aleatoric[c] / bound);
                worst = if w.is_finite() { worst.max(w) } else { f64::INFINITY };
                n += 1;
            }
            if n == 0 {
                continue;
            }
            let ev = || PointEvidence {
                gen: p.model_gen,
                audit: p.kind.is_audit(),
                w2: w2 / n as f64,
                z2: z2 / n as f64,
                e2: sq(p.err_energy.abs() / p.n_atoms.max(1) as f64 / tier),
                worst,
                s2: s2 / n as f64,
                a2: a2 / n as f64,
            };
            if verdict {
                classes[c].push(ev());
            }
            posterior[c].push(ev());
        }
    }
    let per_class = std::array::from_fn(|c| {
        let (inflation_var, inflation_points, inflation_raw) = variance_inflation(&posterior[c]);
        let all = &classes[c];
        let from = window_from(all);
        let inside = |e: &&PointEvidence| from.is_none_or(|g| e.gen >= g);
        let audits: Vec<&PointEvidence> = all.iter().filter(inside).filter(|e| e.audit).collect();
        let fallbacks: Vec<&PointEvidence> = all.iter().filter(inside).filter(|e| !e.audit).collect();
        let (n_a, n_f) = (audits.len(), fallbacks.len());
        let points = n_a + n_f;
        let aged = all.len() - points;
        if points == 0 {
            return ClassCalibration { factor: 1.0, audits: 0, fallbacks: 0, evidence: 0, points: 0, aged, window_from: None, z_rms: f64::NAN, measured: f64::NAN, measured_energy: f64::NAN, worst_atom: 0.0, inflation: inflation_var, inflation_points, inflation_raw };
        }
        // the fallbacks of the class together weigh at most the audits (never fewer than three);
        // the count in integers (review I1), the per-point weight only for the sums
        let cap = n_a.max(MIN_EVIDENCE);
        let evidence = n_a + n_f.min(cap);
        let w_f = if n_f == 0 { 0.0 } else { n_f.min(cap) as f64 / n_f as f64 };
        let den = evidence as f64;
        let sum = |f: &dyn Fn(&PointEvidence) -> f64| audits.iter().map(|e| f(e)).sum::<f64>() + w_f * fallbacks.iter().map(|e| f(e)).sum::<f64>();
        let inflation = 1.0 + 1.0 / (2.0 * points as f64).sqrt();
        let z_rms = (sum(&|e| e.z2) / den).sqrt();
        let measured = (sum(&|e| e.w2) / den).sqrt() * inflation;
        let measured_energy = (sum(&|e| e.e2) / den).sqrt() * inflation;
        let worst_atom = audits.iter().chain(&fallbacks).map(|e| e.worst).fold(0.0f64, |m, w| if w.is_nan() || m.is_nan() { f64::NAN } else { m.max(w) });
        ClassCalibration { factor: z_rms.max(1.0) * inflation, audits: n_a, fallbacks: n_f, evidence, points, aged, window_from: from, z_rms, measured, measured_energy, worst_atom, inflation: inflation_var, inflation_points, inflation_raw }
    });
    Calibration { per_class }
}

/// λ_c (module doc, "The variance inflation"): over the class's points in the posterior's
/// jurisdiction, in their own window of the newest generations ([`WINDOW_POINTS`]), one draw per
/// point, λ = max(1, √(Σ w̄² / Σ s̄²)) · (1 + 1/√(2n)) — the measured force error over the
/// epistemic std, both in the policy's bound units, as a ratio of sums (not a mean of ratios: a
/// confident point with a small error adds little to either sum, so no point's tiny std can set
/// the class's factor — the physics review's I2). 1 with fewer than [`MIN_EVIDENCE`] points; 1
/// when every std is exactly 0 (rule 6 then compares 0 whatever the factor); ∞ when a
/// comparison was not a number. Returns (λ, n, the raw ratio).
fn variance_inflation(points: &[PointEvidence]) -> (f64, usize, f64) {
    let from = window_from(points);
    let inside: Vec<&PointEvidence> = points.iter().filter(|e| from.is_none_or(|g| e.gen >= g)).collect();
    let n = inside.len();
    if n < MIN_EVIDENCE {
        return (1.0, n, f64::NAN);
    }
    // the measured error less its aleatoric part, per point and never below 0 (fix round 1, the
    // review's I2): E[w̄²] = s̄² + ā̄² for a calibrated posterior at a fresh geometry
    let sw: f64 = inside.iter().map(|e| (e.w2 - e.a2).max(0.0)).sum();
    let ss: f64 = inside.iter().map(|e| e.s2).sum();
    if !(sw.is_finite() && ss.is_finite()) {
        return (f64::INFINITY, n, f64::INFINITY);
    }
    if ss == 0.0 {
        return (1.0, n, f64::NAN);
    }
    let raw = (sw / ss).sqrt();
    (raw.max(1.0) * (1.0 + 1.0 / (2.0 * n as f64).sqrt()), n, raw)
}

/// What the refusal's words read (a `ClassLoo` holds all of it; `class_verdict` passes its own).
pub struct WordsInput {
    pub samples: usize,
    pub force_rms: f64,
    pub force_max: f64,
    pub energy_max: f64,
    /// The measured force / energy errors (inflated), `None` without evidence.
    pub measured: Option<f64>,
    pub measured_energy: Option<f64>,
    /// The evidence count and the calibration factor κ.
    pub evidence: usize,
    pub calibration: f64,
}

/// The words of a class refusal (key, value, limit): the part after "isn't learned well enough yet
/// (<class>: " — the verdict's `reason` and the provider's Log note alike (one wording, review M2).
/// The uncalibrated keys keep their words of before (pinned byte for byte by the refusal-string
/// tests).
pub fn refusal_words(key: ClassKey, value: f64, limit: f64, w: &WordsInput) -> String {
    let n = w.samples;
    // "the model's own uncertainty was κ× too small" only when it says something (κ is floored at
    // 1 and ∞ when a comparison failed: review I3)
    let posterior = if w.calibration.is_finite() && w.calibration > 1.05 { format!("; the model's own uncertainty was {:.1}× too small there", w.calibration) } else { String::new() };
    let over = format!("{} DFT comparison{}", w.evidence, if w.evidence == 1 { "" } else { "s" });
    match key {
        ClassKey::Samples => format!("{n} samples < Q′ {limit:.0}"),
        ClassKey::Force => {
            let energy = if w.energy_max > 1.0 { format!(", energy {:.1}× the tier", w.energy_max) } else { String::new() };
            format!("LOO force error {:.1}× the bound, RMS {:.2}×{energy}, {n} samples", w.force_max, w.force_rms)
        }
        ClassKey::Energy => format!("LOO energy error {:.1}× the tier, {n} samples", w.energy_max),
        ClassKey::WorstAtom if w.evidence < MIN_EVIDENCE => {
            format!("a DFT check found an atom's force {value:.1}× its bound, limit {limit:.0}× while fewer than {MIN_EVIDENCE} comparisons judge the class; {n} samples")
        }
        ClassKey::WorstAtom => format!("a DFT check found an atom's force {value:.1}× its bound — the class does not enter while its recent evidence holds an atom beyond {limit:.0}×; {n} samples"),
        ClassKey::Calibrated => format!("DFT-measured force error {value:.2}× the bound over {over}, limit {limit:.2} (leave-one-out RMS {:.2}×){posterior}; {n} samples", w.force_rms),
        ClassKey::LooFloor => format!(
            "leave-one-out force RMS {value:.2}× the bound, limit {limit:.2} (DFT-measured {} over {over}); {n} samples",
            w.measured.map_or("—".to_string(), |m| format!("{m:.2}×"))
        ),
        ClassKey::MeasuredEnergy => format!("DFT-measured energy error {value:.2}× the tier over {over}, limit {limit:.2}; {n} samples"),
    }
}

impl Calibration {
    /// The verdict for `class` (0 pair, 1 many-body) from its leave-one-out statistics, the fit's
    /// Q′ and the standing verdict `was_trusted` (the hysteresis; module doc, "The verdict").
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
    pub fn class_verdict(&self, class: usize, st: &ClassStats, q_prime: usize, was_trusted: bool) -> ClassVerdict {
        let cal = self.per_class[class];
        let finite = |x: f64| (!x.is_nan()).then_some(x);
        let words = WordsInput {
            samples: st.samples,
            force_rms: st.force_rms,
            force_max: st.force_max,
            energy_max: st.energy_max,
            measured: if cal.points > 0 { finite(cal.measured) } else { None },
            measured_energy: if cal.points > 0 { finite(cal.measured_energy) } else { None },
            evidence: cal.evidence,
            calibration: cal.factor,
        };
        let untrusted = |key: ClassKey, value: f64, limit: f64| ClassVerdict::Untrusted { key, value, limit, reason: refusal_words(key, value, limit, &words) };
        if st.samples < q_prime {
            return untrusted(ClassKey::Samples, st.samples as f64, q_prime as f64);
        }
        // the worst-atom guard: on entry at every evidence count, both ways while unverified
        if (!was_trusted || !cal.calibrated()) && !(cal.worst_atom <= WORST_ATOM) {
            return untrusted(ClassKey::WorstAtom, cal.worst_atom, WORST_ATOM);
        }
        if !cal.calibrated() {
            // the rule of before, unchanged — no evidence loosens nothing
            return match super::trust::class_refusal(st.samples, q_prime, st.force_rms, st.force_max, st.energy_max) {
                None => ClassVerdict::Trusted,
                Some((key, value, limit)) => untrusted(key, value, limit),
            };
        }
        let limit = if was_trusted { LEAVE } else { ENTER };
        // the force: the measured error with the leave-one-out as the floor (NaN fails either way);
        // the key names the term that decided
        let (w, r) = (cal.measured, st.force_rms);
        if !(w <= limit) && !(w < r) {
            return untrusted(ClassKey::Calibrated, w, limit);
        }
        if !(r <= limit) {
            return untrusted(ClassKey::LooFloor, r, limit);
        }
        if !(cal.measured_energy <= limit) {
            return untrusted(ClassKey::MeasuredEnergy, cal.measured_energy, limit);
        }
        ClassVerdict::Trusted
    }
}
