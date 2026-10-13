//! [`SurrogateKs`]: the surrogate force provider (spec §5). It answers from the fitted linear
//! model (`surrogate::Model`) when the model is trustworthy *and* confident at the current
//! geometry, and otherwise hands the step to its DFT teacher — any [`ForceProvider`] — keeping
//! the teacher's answer as a new training sample. Atomic units throughout; nothing here knows
//! about rendering or the UI (CLAUDE.md rule 5), and nothing is per element beyond Z as a
//! species label (rule 1: the reference energies are inputs computed by the teacher).
//!
//! # The fallback policy, as a decision list
//!
//! `evaluate(scene)` walks this list; the first rule that applies decides. Every rule that hands
//! the step to the teacher writes its reason into `EvalStats::notes` ("surrogate: …").
//!
//! 1. **No model** (nothing fitted yet) → teacher; the sample is kept; `Source::Fallback`.
//! 2. **Unknown species** (a Z the model's `Basis` has no channel for) → teacher,
//!    unconditionally; the sample is kept and the model is marked for a rebuild with the
//!    enlarged species set at the next [`refit`](SurrogateKs::refit). Never a guess.
//! 3. **Other settings** — the scene's (grid n, h, σ, charge, multiplicity) differ from the
//!    model's `Provenance` → teacher, unconditionally; the sample is NOT kept (it belongs to a
//!    different Kohn-Sham surface than the store's).
//! 4. **Untrusted fit** (trust rule below) → teacher; the sample is kept so the fit can earn
//!    trust. The note says which condition failed, e.g. "surrogate untrusted: n_samples 12 <
//!    q_prime 16".
//! 5. **Out of the training envelope**: any atom whose descriptor lies farther than
//!    `max_descriptor_distance` (3.0) training spreads outside the per-feature range the fit
//!    has seen (`surrogate::Envelope`) → teacher; sample kept. This guards the directions the
//!    variance cannot: a linear model's predictive variance is a leverage in feature space and
//!    is small wherever the features themselves are small — all body-2 features vanish as a
//!    pair separates, so a dissociating dimer would otherwise be answered "confidently" with
//!    Σ ref by a model that never saw an empty neighbourhood (envelope module doc for the unit).
//! 6. **Force uncertainty**: with `force_std[j]` the model's expected ‖δF_j‖ (Task 3's binding
//!    unit), the per-component equivalent `force_std[j]/√3` must satisfy
//!    `force_std[j]/√3 ≤ min(max(force_std_abs, force_std_rel·|F_j|), 1e-2)` for every atom
//!    (2e-3 Ha/Bohr absolute, 10 % relative, the relative bound capped at
//!    [`FORCE_STD_BOUND_MAX`] = 1e-2 Ha/Bohr so a large force never buys an unbounded error).
//!    Otherwise → teacher; sample kept.
//! 7. **Energy uncertainty, per atom and epistemic only**: `energy_std` is the TOTAL energy's
//!    std with the aleatoric floor (N·s_E)² (Task 3), so the criterion is
//!    `√(energy_std² − (N s_E)²) / N ≤ energy_std_per_atom` (1e-3 Ha). The floor is excluded
//!    because it is the teacher's own scatter (egg-box, SCF lag) — the part of the residual no
//!    number of samples removes, so a fallback triggered by it would teach the model nothing;
//!    the epistemic part is what new samples shrink, so it is the part a fallback can act on.
//!    (Per atom the floor contributes s_E ≈ 1e-5 in quadrature, far below 1e-3: including it
//!    would be inert in practice, not wrong; it is excluded for the reason, not the size.)
//!    Otherwise → teacher; sample kept.
//! 8. **Audit** (ruling S7): every `audit_every`-th step the model would have answered (2000 by
//!    default — ruling S14's interim value: at 200 one ~0.1 s DFT audit per 200 ~10 µs steps set
//!    the viewer's pace to ≈ 300–400 fs/s; 0 disables) the teacher is called anyway, the sample is kept and the gap is
//!    recorded. A passing audit is an ordinary model-surface Fallback (`teacher_gap`). If
//!    |gap|/N > `energy_std_per_atom` the audit **fails**: the model is untrusted until the next
//!    refit, and that step is already reported on the teacher surface — `energy` = teacher's,
//!    `teacher_gap` None, `refit_jump = Some(gap)` (the change of the reported energy at these
//!    positions, exactly as after a refit) — so the driver books the Model → Teacher flip into
//!    W continuously instead of restarting the drift history.
//! 9. Else the model answers: `Source::Surrogate`, no density (`display: None`: the surrogate
//!    has no electrons).
//!
//! `uncertainty` = max_j force_std[j]/√3, the **per-component** equivalent (S1's "max per-atom
//! force std" is √3 times it); it is reported whenever the model was consulted (rules 5–9), so
//! the readouts can show how far off the model was on a fallback. `variance_cadence` = k
//! evaluates the O(Q′²) variance (rules 6–7) only on every k-th consultation and reuses that
//! verdict in between (k = 1 here; Task 9 tunes); the envelope (rule 5) is evaluated on every
//! consultation — it is cheap, and a stored envelope rejection must never be replayed at a
//! geometry that is inside.
//!
//! # Trust rule (from the Task 3 fit, binding)
//!
//! The model may answer only when the last `FitReport` has `n_samples ≥ q_prime` (distinct
//! geometries, not rows: the 3N force rows of one sample probe one geometry) AND
//! `residual_dof_e ≥ 30` AND `residual_dof_f ≥ 30` (the two noise levels are then known to
//! ≈ 25 %, so the variance means something). Below that the fit is a curve through too few
//! points with a noise estimate collapsing towards its floor — overconfident inside its span.
//!
//! **Training residuals (ruling S8).** The posterior cannot see basis misspecification: with
//! n_max = 8 and a teacher whose pair energy is not ≈ 0 at r_cut, one training point outside
//! the window made the two-noise evidence absorb a 1e-2 Ha energy offset as "noise" (s_E
//! 8e-4/atom, a 6σ outlier) while the forces and the epistemic stds stayed tiny — no
//! Gaussian-noise criterion fires on that. So after every refit one pass over the store
//! evaluates the model at its own training data, and trust additionally requires
//! max_i |E_model − E_i|/N_i ≤ `energy_std_per_atom` and, over every atom,
//! ‖F_model − F‖/√3 ≤ the policy's force bound for that atom (the same numbers rules 6–7
//! use). "The fit reproduces its own data to the accuracy it claims" is what a model must show
//! before it may answer; [`SurrogateKs::training_residuals`] exposes both maxima and the note
//! says which failed ("surrogate untrusted: max training energy residual 5.0e-3 > 1.0e-3
//! Ha/atom"). A failed audit (rule 8) is the fourth condition, until the next refit.
//! [`SurrogateKs::trusted`] exposes the verdict; every evaluate of an untrusted model is a
//! Fallback (rule 4) and the samples accumulate until a refit earns trust.
//!
//! # Trust recovery and quarantine (ruling S15, Task 8 fix round 3)
//!
//! The training-residual condition is a maximum over the store, so ONE sample the basis cannot
//! represent — a dimer pulled apart beyond the cutoff, where every feature vanishes and the
//! model can only say Σ E_ref while the teacher's restricted-spin dissociation lies far above it;
//! a stray hot-burst geometry — untrusts the whole model at the next refit, and before this
//! round nothing ever gave the trust back: every later step was a rule-4 fallback whose sample
//! (kept on every step) could not repair the maximum. So when a refit leaves the training
//! residuals above the policy's thresholds, [`refit`](SurrogateKs::refit):
//! 1. retries the fit once with `degree_cap = 6` on a body-3 basis (the documented setting for
//!    bent molecules with wide bond excursions) and keeps it if the residuals then pass;
//! 2. otherwise **quarantines** the worst offenders — the samples whose per-atom energy or worst
//!    force residual, relative to its threshold, is within a factor 2 of the worst — and refits
//!    without them, up to [`QUARANTINE_ROUNDS`] rounds, never more than
//!    [`QUARANTINE_MAX_FRACTION`] of the store. A quarantined sample stays in the store (and in
//!    the sidecar, `sample_quarantine`); it is excluded from the fit, the envelope and the
//!    residuals; a geometry within [`QUARANTINE_FRACTION`] · `max_descriptor_distance` training
//!    spreads of one falls back ("near a quarantined geometry") and that answer is not kept (it would
//!    only be quarantined again); `ForgetModel` clears it. The trust rule
//!    is unchanged — what is excluded is said: `quarantined()` and the recovery note.
//!
//! **Fallback thinning.** A fallback answer at a geometry like one already held teaches nothing
//! new, and one per step for thousands of steps (a molecule parked where the model cannot
//! answer) bloats the store and every refit. A fallback sample is therefore kept only if its
//! nearest stored descriptor (every stored sample, quarantined included; L∞ over features in the
//! envelope's training-spread units, the worst atom) is farther than
//! [`THIN_FRACTION`] · `max_descriptor_distance`, or if [`THIN_EVERY`] teacher steps have
//! passed since the last kept one. Exempt (always kept): audits; fallbacks with no model yet,
//! with a new species, or while the trust rule still needs more samples (n_samples < Q′ or too
//! few residual degrees of freedom) — those samples are what earns trust.
//!
//! # One-surface energy accounting (spec §5)
//!
//! The driver's `E_conserved` must be computed on ONE potential-energy surface, or every hand-
//! off between model and teacher would look like a drift. So on a Fallback step with a trusted
//! model the result is `ForceResult { energy: E_model(r), forces: F_teacher(r) }` and
//! `teacher_gap = E_teacher − E_model` is reported separately: the atoms move on the teacher's
//! forces (accuracy), the invariant stays on the model surface (continuity), and the gap is a
//! readout of the model's error rather than a corruption of the drift. Relative to the model
//! Hamiltonian, ΔF = F_teacher − F_model is then an **external force** — the same category as
//! the drag spring and the thermostat — and the driver books its work: the result also carries
//! `EvalStats::model_forces` on such a step and `Dynamics` adds ½(ΔF_old + ΔF_new)·Δr per step
//! to W (trapezoid, O(dt²)-exact like the drag booking; `DynSnapshot::work_mismatch` publishes
//! the running sum). Unbooked, `E_conserved` would move by −Δ(teacher_gap) on every fallback
//! step and the drift readout would mix integrator error with the model's force error — which
//! is exactly what the two readouts are meant to separate. When no trusted model
//! exists (rules 1–4) the step is plain DFT: `energy` is the teacher's and `teacher_gap` is
//! `None` — [`EvalStats::energy_surface`] then says `Surface::Teacher`. A change of surface
//! between consecutive evaluations is a discontinuity of the invariant; the driver books it when
//! the provider announces it (`refit_jump`, below) and otherwise restarts the `E_conserved`
//! history, the same path it uses for a discontinuous edit.
//!
//! A refit moves the model surface under the atoms. [`refit`](SurrogateKs::refit) is called by
//! the worker between ticks, never inside `evaluate`; it evaluates the new model at the last
//! evaluated positions and reports, once, on the first evaluate after it,
//! `refit_jump = E_new(r_last) − E_reported(r_last)`, where `E_reported` is whatever energy the
//! last evaluate returned (the old model's on the model surface, the teacher's before the first
//! trusted fit — so the first transition onto the model surface is booked too). The driver adds
//! it to its work term W exactly like thermostat work, and because it notices the refit through
//! [`ForceProvider::surface_epoch`] it re-evaluates at the unchanged positions first: the jump is
//! then exact (no half-kick on mixed forces) and `E_conserved` is continuous to rounding
//! (test: 1e-12). If the new fit is not trusted the surface does not change and nothing is
//! booked; if a rebuild makes a trusted model untrusted (a new species raised Q′) the next
//! evaluate is on the teacher surface with no jump, and the driver's restart path applies.
//!
//! `reset()` / `discontinuity()` pass to the teacher; the model persists (it is a function of
//! positions, not of solver state). The teacher's errors propagate unchanged; a sample is never
//! kept from a failed teacher call.
//!
//! # Species, references, provenance
//!
//! The species list of the `Basis` is the sorted, deduplicated set of Z over the samples held
//! when the first refit runs (the first sample's set, in the brief's words, plus anything added
//! before that refit); a later sample with a new species marks the basis dirty and the next
//! refit rebuilds the accumulator from the store with the enlarged set — which needs a
//! reference energy for the new species ([`set_reference_energy`](SurrogateKs::set_reference_energy);
//! Task 5 computes them with the teacher on an isolated atom) and otherwise fails with
//! `FitError::MissingReference`, keeping the old model. Provenance (grid, h, σ, charge,
//! multiplicity, teacher name, teacher tolerance) is derived from the scene a sample came from
//! and pinned by the first sample; the teacher's SCF tolerance is not visible through the
//! boundary, so it is set with [`with_teacher_tol`](SurrogateKs::with_teacher_tol) (0 = not
//! recorded) for the sidecar's benefit — the provider itself never compares it.

use crate::provider::{EvalStats, ForceProvider, ForceResult, ProviderError, Source};
use crate::scene::Scene;
use std::collections::BTreeMap;
use std::time::Instant;
use surrogate::{
    describe, Accumulator, Basis, Envelope, FitError, FitOptions, FitReport, Model, Provenance,
    Sample, SampleStore, SampleTag, StoreError,
};

/// The relative force criterion's bound never exceeds this (Ha/Bohr).
pub const FORCE_STD_BOUND_MAX: f64 = 1e-2;
/// Trust rule: both residual degrees of freedom of the fit must reach this.
pub const MIN_RESIDUAL_DOF: f64 = 30.0;
/// Fallback thinning and the quarantine neighbourhood, as a fraction of
/// `FallbackPolicy::max_descriptor_distance` (module doc, "Trust recovery").
pub const THIN_FRACTION: f64 = 0.25;
/// A fallback sample is kept at least every this many teacher steps (module doc).
pub const THIN_EVERY: usize = 10;
/// A geometry within this fraction of `max_descriptor_distance` (training-spread units) of a
/// quarantined sample falls back (module doc): small — the quarantined geometry itself and its
/// immediate neighbourhood, not the region around it the model still represents.
pub const QUARANTINE_FRACTION: f64 = 0.1;
/// The body-3 degree cap the first recovery step retries with (the bent-molecule setting).
pub const RECOVERY_DEGREE_CAP: usize = 6;
/// Quarantine rounds per refit (module doc).
pub const QUARANTINE_ROUNDS: usize = 5;
/// Never quarantine more than this fraction of the store.
pub const QUARANTINE_MAX_FRACTION: f64 = 0.1;
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// When the model may answer (module doc, "The fallback policy"). Defaults are the spec's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FallbackPolicy {
    /// Absolute bound on the per-component force std, Ha/Bohr (2e-3).
    pub force_std_abs: f64,
    /// Relative bound, fraction of |F_j| (0.10); the resulting bound is capped at
    /// [`FORCE_STD_BOUND_MAX`].
    pub force_std_rel: f64,
    /// Bound on the epistemic energy std per atom, Ha (1e-3).
    pub energy_std_per_atom: f64,
    /// Out-of-distribution envelope: training spreads outside the per-feature training range
    /// (3.0).
    pub max_descriptor_distance: f64,
    /// Evaluate the O(Q′²) variance on every k-th consultation (1 here; Task 9 tunes).
    pub variance_cadence: usize,
    /// Audit (ruling S7): every `audit_every`-th step the model would have answered, the
    /// teacher is called instead and the gap recorded; 0 disables (2000, ruling S14 interim;
    /// S7 had 200).
    pub audit_every: usize,
}

impl Default for FallbackPolicy {
    fn default() -> FallbackPolicy {
        FallbackPolicy {
            force_std_abs: 2e-3,
            force_std_rel: 0.10,
            energy_std_per_atom: 1e-3,
            max_descriptor_distance: 3.0,
            variance_cadence: 1,
            audit_every: 2000,
        }
    }
}

impl FallbackPolicy {
    /// The per-component force bound for an atom with force magnitude `fmag`:
    /// min(max(abs, rel·|F|), [`FORCE_STD_BOUND_MAX`]).
    pub fn force_bound(&self, fmag: f64) -> f64 {
        self.force_std_abs.max(self.force_std_rel * fmag).min(FORCE_STD_BOUND_MAX)
    }
}

/// What the last evaluate reported, for the refit jump.
struct LastEval {
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    energy: f64,
    on_model: bool,
    provenance: Provenance,
}

/// The variance verdict (rules 6–7) of the last consultation that evaluated it.
#[derive(Clone)]
struct VarianceVerdict {
    accept: bool,
    uncertainty: f64,
    reason: Option<String>,
}

/// Why the model is not consulted (rules 1–4), whether the teacher's answer is kept, and
/// whether that sample is subject to thinning (module doc, "Trust recovery").
struct Gate {
    reason: String,
    keep_sample: bool,
    thin: bool,
}

/// The model's answer at the geometry of a teacher step (rules 5–7 and audits).
struct ModelAnswer {
    energy: f64,
    forces: Vec<[f64; 3]>,
    uncertainty: Option<f64>,
}

/// One teacher step: why, with what from the model, and whether it is an audit.
struct Handoff {
    prov: Provenance,
    pos: Vec<[f64; 3]>,
    z: Vec<u8>,
    reason: String,
    keep_sample: bool,
    /// The kept sample is subject to thinning (module doc).
    thin: bool,
    model: Option<ModelAnswer>,
    audit: bool,
}

/// Worst training residuals of the current model over the store (trust rule).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingResiduals {
    /// max over samples of |E_model − E_sample| / N (Ha per atom).
    pub energy_per_atom: f64,
    /// max over atoms of ‖F_model − F_sample‖/√3 (Ha/Bohr, per component equivalent) …
    pub force: f64,
    /// … and the policy's force bound for that atom (its teacher force magnitude).
    pub force_bound: f64,
}

/// The surrogate force provider (module doc).
pub struct SurrogateKs {
    teacher: Box<dyn ForceProvider>,
    name: &'static str,
    teacher_tol: f64,
    policy: FallbackPolicy,
    store: Option<SampleStore>,
    accumulator: Option<Accumulator>,
    /// Store samples already in the accumulator (a prefix).
    accumulated: usize,
    refs: BTreeMap<u8, f64>,
    /// Species whose reference came from a sidecar without the S9 reference rule (re-solved by
    /// the next sampling pass; `save_sidecar` writes no rule while any remain).
    legacy_refs: std::collections::BTreeSet<u8>,
    /// A reference changed since the accumulator was built → rebuild at the next refit.
    refs_dirty: bool,
    /// The store holds a species the accumulator's basis lacks → rebuild at the next refit.
    species_dirty: bool,
    model: Option<Model>,
    report: Option<FitReport>,
    envelope: Option<Envelope>,
    training_residuals: Option<TrainingResiduals>,
    /// A failed audit's teacher − model gap: untrusted until the next refit.
    audit_failed: Option<f64>,
    epoch: u64,
    pending_refit: bool,
    pending_jump: Option<f64>,
    last: Option<LastEval>,
    consults: usize,
    last_variance: Option<VarianceVerdict>,
    /// Steps the model answered or would have answered (the audit counter).
    answered: usize,
    /// Per stored sample: quarantined (module doc, "Trust recovery"); shorter than the store
    /// means "not quarantined" for the rest.
    quarantine: Vec<bool>,
    /// The body-3 degree cap a recovery settled on (applied to every rebuilt basis).
    degree_cap: Option<usize>,
    /// Teacher steps since the last kept fallback sample (thinning).
    since_kept: usize,
    /// Per stored sample: its atoms' (species index, descriptor) under `desc_basis`.
    desc_cache: Vec<Vec<(usize, Vec<f64>)>>,
    desc_basis: Option<Basis>,
    /// What the last refit's recovery did, for the readouts.
    recovery: Option<String>,
}

/// The fitted state a recovery attempt may have to restore.
struct FitState {
    accumulator: Option<Accumulator>,
    accumulated: usize,
    model: Option<Model>,
    report: Option<FitReport>,
    envelope: Option<Envelope>,
    training_residuals: Option<TrainingResiduals>,
    degree_cap: Option<usize>,
}

impl SurrogateKs {
    pub fn new(teacher: Box<dyn ForceProvider>, policy: FallbackPolicy) -> SurrogateKs {
        // `name()` returns `&'static str`: a few bytes leaked once per provider built (the
        // worker builds one per backend switch), not per evaluation
        let name: &'static str = Box::leak(format!("surrogate({})", teacher.name()).into_boxed_str());
        SurrogateKs {
            teacher,
            name,
            teacher_tol: 0.0,
            policy,
            store: None,
            accumulator: None,
            accumulated: 0,
            refs: BTreeMap::new(),
            legacy_refs: std::collections::BTreeSet::new(),
            refs_dirty: false,
            species_dirty: false,
            model: None,
            report: None,
            envelope: None,
            training_residuals: None,
            audit_failed: None,
            epoch: 0,
            pending_refit: false,
            pending_jump: None,
            last: None,
            consults: 0,
            last_variance: None,
            answered: 0,
            quarantine: Vec::new(),
            degree_cap: None,
            since_kept: THIN_EVERY,
            desc_cache: Vec::new(),
            desc_basis: None,
            recovery: None,
        }
    }

    /// Samples held in quarantine (module doc, "Trust recovery").
    pub fn quarantined(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q).count()
    }

    /// Whether stored sample `i` is quarantined.
    pub fn is_quarantined(&self, i: usize) -> bool {
        self.quarantine.get(i).copied().unwrap_or(false)
    }

    /// What the last refit's trust recovery did ("degree_cap 6 …", "N samples quarantined …"),
    /// `None` when it was not needed.
    pub fn recovery_note(&self) -> Option<&str> {
        self.recovery.as_deref()
    }

    /// Records the teacher's SCF tolerance in the samples' provenance (module doc).
    pub fn with_teacher_tol(mut self, tol: f64) -> SurrogateKs {
        self.teacher_tol = tol;
        self
    }

    /// The accumulator's basis (`None` before the first refit).
    pub fn basis(&self) -> Option<&Basis> {
        self.accumulator.as_ref().map(|a| &a.basis)
    }

    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    /// The last fit's report (`None` before the first refit).
    pub fn report(&self) -> Option<&FitReport> {
        self.report.as_ref()
    }

    /// The current model's worst training residuals (`None` before the first refit).
    pub fn training_residuals(&self) -> Option<TrainingResiduals> {
        self.training_residuals
    }

    /// The teacher − model gap of the audit that made the model untrusted, until the next refit.
    pub fn audit_failed(&self) -> Option<f64> {
        self.audit_failed
    }

    /// Samples held (fallback steps and `add_sample`).
    pub fn samples(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.len())
    }

    /// Surface counter: changes when a refit moves the energy surface (module doc); the same
    /// number as [`ForceProvider::surface_epoch`].
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Whether a fallback step has been kept since the last refit.
    pub fn pending_refit(&self) -> bool {
        self.pending_refit
    }

    pub fn policy(&self) -> &FallbackPolicy {
        &self.policy
    }

    /// Replaces the policy; the variance cadence restarts and the training residuals are
    /// re-judged against the new thresholds.
    pub fn set_policy(&mut self, p: FallbackPolicy) {
        self.policy = p;
        self.consults = 0;
        self.last_variance = None;
        self.training_residuals = self.compute_training_residuals();
    }

    /// Teach from an existing sample (a sampling pass, a loaded sidecar). `scene` supplies the
    /// settings the sample was produced at (its provenance); the first sample pins the store's.
    pub fn add_sample(&mut self, s: Sample, scene: &Scene) -> Result<(), StoreError> {
        let prov = self.scene_provenance(scene);
        self.add_sample_with_provenance(s, &prov)
    }

    /// The primitive behind [`add_sample`](Self::add_sample): a sample with the provenance it
    /// was produced at (what a sidecar stores).
    pub fn add_sample_with_provenance(&mut self, s: Sample, prov: &Provenance) -> Result<(), StoreError> {
        self.push_sample(s, prov)
    }

    /// Isolated-atom energy of species `z` (Hartree), the model's reference (Z-only: computed
    /// by the teacher, never a table). A change after a fit rebuilds the accumulator at the next
    /// refit. The caller vouches that `e` follows the reference rule (ruling S9: the atom's
    /// lowest-energy multiplicity — what `crate::sampling` computes); the sidecar records it.
    pub fn set_reference_energy(&mut self, z: u8, e: f64) {
        self.legacy_refs.remove(&z);
        if self.refs.insert(z, e) != Some(e) {
            self.refs_dirty = true;
        }
    }

    /// The trust rule (module doc): a model exists and its report, its training residuals and
    /// the last audit pass.
    pub fn trusted(&self) -> bool {
        self.model.is_some() && self.untrusted_reason().is_none()
    }

    /// Why the model may not answer (the trust rule's first failing condition), `None` when it
    /// may (or when nothing was fitted: see [`trusted`](Self::trusted)).
    pub fn untrusted_reason(&self) -> Option<String> {
        let r = self.report.as_ref()?;
        if r.n_samples < r.q_prime {
            return Some(format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime));
        }
        if r.residual_dof_e < MIN_RESIDUAL_DOF {
            return Some(format!("residual_dof_e {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_e));
        }
        if r.residual_dof_f < MIN_RESIDUAL_DOF {
            return Some(format!("residual_dof_f {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_f));
        }
        if let Some(t) = &self.training_residuals {
            if t.energy_per_atom > self.policy.energy_std_per_atom {
                return Some(format!(
                    "max training energy residual {:.1e} > {:.1e} Ha/atom",
                    t.energy_per_atom, self.policy.energy_std_per_atom
                ));
            }
            if t.force > t.force_bound {
                return Some(format!(
                    "max training force residual {:.1e} > {:.1e} Ha/Bohr",
                    t.force, t.force_bound
                ));
            }
        }
        if let Some(gap) = self.audit_failed {
            return Some(format!("audit found teacher − model {gap:+.1e} Ha; refit to recover"));
        }
        None
    }

    /// One pass over the samples the current model was fitted on (the accumulated prefix of
    /// the store — samples kept since the last refit are not training data yet; module doc,
    /// "Trust rule").
    fn compute_training_residuals(&self) -> Option<TrainingResiduals> {
        let (model, store) = (self.model.as_ref()?, self.store.as_ref()?);
        let mut worst = TrainingResiduals { energy_per_atom: 0.0, force: 0.0, force_bound: f64::INFINITY };
        let mut worst_ratio = 0.0;
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.is_quarantined(i) {
                continue;
            }
            let Ok(p) = model.predict(&s.pos, &s.z, false) else { continue };
            let n = s.pos.len().max(1) as f64;
            worst.energy_per_atom = worst.energy_per_atom.max((p.energy - s.energy).abs() / n);
            for (fm, fs) in p.forces.iter().zip(&s.forces) {
                let d = ((fm[0] - fs[0]).powi(2) + (fm[1] - fs[1]).powi(2) + (fm[2] - fs[2]).powi(2)).sqrt() / SQRT_3;
                let bound = self.policy.force_bound((fs[0] * fs[0] + fs[1] * fs[1] + fs[2] * fs[2]).sqrt());
                if d / bound > worst_ratio {
                    worst_ratio = d / bound;
                    worst.force = d;
                    worst.force_bound = bound;
                }
            }
        }
        Some(worst)
    }

    /// Rebuild the model from the samples (module doc): the accumulator takes the samples added
    /// since the last refit, or is rebuilt from the store after a species or reference change;
    /// then one hyperparameter search and solve, and one pass over the store for the training
    /// residuals. Clears a failed audit. Computes the refit jump for the next evaluate and bumps
    /// the surface epoch when the surface moved. On an error the previous model, report and
    /// envelope stay in place.
    pub fn refit(&mut self) -> Result<FitReport, FitError> {
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        let mut report = self.fit_store()?;
        self.recovery = None;
        if self.residuals_fail() {
            report = self.recover_trust(report);
        }
        self.surface_moved(was_trusted, was_model).map_err(FitError::Describe)?;
        Ok(report)
    }

    /// The training residuals exceed the policy's thresholds (the trust rule's S8 part).
    fn residuals_fail(&self) -> bool {
        self.training_residuals
            .is_some_and(|t| t.energy_per_atom > self.policy.energy_std_per_atom || t.force > t.force_bound)
    }

    fn save_state(&self) -> FitState {
        FitState {
            accumulator: self.accumulator.clone(),
            accumulated: self.accumulated,
            model: self.model.clone(),
            report: self.report.clone(),
            envelope: self.envelope.clone(),
            training_residuals: self.training_residuals,
            degree_cap: self.degree_cap,
        }
    }

    fn restore_state(&mut self, s: FitState) {
        self.accumulator = s.accumulator;
        self.accumulated = s.accumulated;
        self.model = s.model;
        self.report = s.report;
        self.envelope = s.envelope;
        self.training_residuals = s.training_residuals;
        self.degree_cap = s.degree_cap;
    }

    /// Ruling S15 (module doc, "Trust recovery"): the degree-cap retry, then quarantine rounds.
    /// Returns the report of the fit that stands; a failed attempt restores the previous fit.
    fn recover_trust(&mut self, report: FitReport) -> FitReport {
        // 1. the bent-molecule setting
        let cap = self.basis().map(|b| (b.body3, b.degree_cap));
        if let Some((true, d)) = cap.filter(|_| self.degree_cap.is_none()) {
            if d < RECOVERY_DEGREE_CAP {
                let saved = self.save_state();
                self.degree_cap = Some(RECOVERY_DEGREE_CAP);
                self.species_dirty = true;
                match self.fit_store() {
                    Ok(r) if !self.residuals_fail() => {
                        self.recovery = Some(format!("refit with degree_cap {RECOVERY_DEGREE_CAP} (the bent-molecule setting) restored the training residuals"));
                        return r;
                    }
                    _ => self.restore_state(saved),
                }
            }
        }
        // 2. quarantine the worst offenders, round by round
        let mut report = report;
        let n_store = self.samples();
        let before = self.quarantined();
        for _ in 0..QUARANTINE_ROUNDS {
            if !self.residuals_fail() {
                break;
            }
            let worst = self.worst_offenders();
            let total = self.quarantined() + worst.len();
            if worst.is_empty() || total as f64 > QUARANTINE_MAX_FRACTION * n_store as f64 {
                break;
            }
            let saved = self.save_state();
            let flags = self.quarantine.clone();
            self.quarantine.resize(n_store, false);
            for &i in &worst {
                self.quarantine[i] = true;
            }
            self.species_dirty = true; // rebuild without them
            match self.fit_store() {
                Ok(r) => report = r,
                Err(_) => {
                    self.restore_state(saved);
                    self.quarantine = flags;
                    break;
                }
            }
        }
        let added = self.quarantined() - before;
        if added > 0 {
            self.recovery = Some(format!(
                "{} samples quarantined — the model cannot represent them ({added} this refit; kept, excluded from the fit; geometries near them fall back)",
                self.quarantined()
            ));
        }
        report
    }

    /// The training samples (not quarantined) whose residual ratio (per-atom energy residual
    /// over its threshold, or worst atom force residual over its bound) exceeds 1 and is within a
    /// factor 2 of the worst ratio.
    fn worst_offenders(&self) -> Vec<usize> {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else { return Vec::new() };
        let p = &self.policy;
        let ratios: Vec<(usize, f64)> = store.samples[..self.accumulated.min(store.len())]
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.is_quarantined(*i))
            .filter_map(|(i, s)| {
                let pr = model.predict(&s.pos, &s.z, false).ok()?;
                let n = s.pos.len().max(1) as f64;
                let mut r = (pr.energy - s.energy).abs() / n / p.energy_std_per_atom;
                for (fm, fs) in pr.forces.iter().zip(&s.forces) {
                    let d = ((fm[0] - fs[0]).powi(2) + (fm[1] - fs[1]).powi(2) + (fm[2] - fs[2]).powi(2)).sqrt() / SQRT_3;
                    r = r.max(d / p.force_bound((fs[0] * fs[0] + fs[1] * fs[1] + fs[2] * fs[2]).sqrt()));
                }
                Some((i, r))
            })
            .collect();
        let worst = ratios.iter().map(|x| x.1).fold(0.0, f64::max);
        if worst <= 1.0 {
            return Vec::new();
        }
        ratios.into_iter().filter(|&(_, r)| r > 1.0 && r >= 0.5 * worst).map(|(i, _)| i).collect()
    }

    /// The fit itself (module doc): incremental, or a rebuild from the store skipping the
    /// quarantined samples; no surface bookkeeping (`refit` does it once).
    fn fit_store(&mut self) -> Result<FitReport, FitError> {
        let Some(store) = self.store.as_ref() else { return Err(FitError::NoSamples) };
        if store.is_empty() {
            return Err(FitError::NoSamples);
        }
        let provenance = store.provenance.clone();
        let opts = FitOptions::default();
        // (Re)build the accumulator: from scratch after a species or reference change, else
        // only the samples added since the last refit.
        let rebuild = self.accumulator.is_none() || self.species_dirty || self.refs_dirty;
        let basis = if rebuild {
            let mut species: Vec<u8> = store.samples.iter().flat_map(|s| s.z.iter().copied()).collect();
            species.sort_unstable();
            species.dedup();
            let mut b = Basis::new(&species);
            if let Some(d) = self.degree_cap.filter(|_| b.body3) {
                b.degree_cap = d;
            }
            b
        } else {
            self.accumulator.as_ref().expect("checked").basis.clone()
        };
        // the references first: an error here changes nothing
        let refs = self.reference_vector(&basis)?;
        let (mut acc, mut accumulated) = if rebuild {
            (Accumulator::new(&basis), 0)
        } else {
            (self.accumulator.take().expect("checked"), self.accumulated)
        };
        let store = self.store.as_ref().expect("checked");
        for (i, s) in store.samples.iter().enumerate().skip(accumulated) {
            if self.quarantine.get(i).copied().unwrap_or(false) {
                accumulated += 1; // held, not fitted (module doc, "Trust recovery")
                continue;
            }
            if let Err(e) = acc.add(s, &basis, &refs, &opts) {
                // cannot happen after a rebuild (the basis covers the store); keep the old
                // accumulator state consistent on the incremental path by rebuilding next time
                self.species_dirty = true;
                if !rebuild {
                    self.accumulator = Some(acc);
                }
                return Err(e.into());
            }
            accumulated += 1;
        }
        let solved = acc.solve(&opts, &provenance, &refs);
        let (model, report) = match solved {
            Ok(x) => x,
            Err(e) => {
                if !rebuild {
                    self.accumulator = Some(acc);
                    self.accumulated = accumulated;
                }
                return Err(e);
            }
        };
        self.envelope = Some(acc.envelope.clone());
        self.accumulator = Some(acc);
        self.accumulated = accumulated;
        self.species_dirty = false;
        self.refs_dirty = false;
        self.model = Some(model);
        self.report = Some(report.clone());
        self.audit_failed = None;
        self.training_residuals = self.compute_training_residuals();
        self.pending_refit = false;
        self.consults = 0;
        self.last_variance = None;
        Ok(report)
    }

    /// Surface bookkeeping after `self.model` was replaced (a refit or an installed sidecar;
    /// module doc): the jump for the next evaluate, and the epoch. `was_trusted` / `was_model`
    /// describe the state before the replacement.
    fn surface_moved(&mut self, was_trusted: bool, was_model: bool) -> Result<(), surrogate::DescribeError> {
        let model = self.model.as_ref().expect("a model was just installed");
        let next_is_model = self.trusted()
            && match &self.last {
                None => true,
                Some(l) => l.provenance == model.provenance && model.basis.species_indices(&l.z).is_ok(),
            };
        self.pending_jump = match (&self.last, next_is_model) {
            (Some(l), true) => {
                let e_new = model.predict(&l.pos, &l.z, false)?.energy;
                Some(e_new - l.energy)
            }
            _ => None,
        };
        if next_is_model || was_model || (was_trusted != self.trusted()) {
            self.epoch += 1;
        }
        Ok(())
    }

    /// The teacher, for work the policy must not see: the sampling pass (`crate::sampling`)
    /// solves its geometries with it directly — they are training data, not steps.
    pub fn teacher_mut(&mut self) -> &mut dyn ForceProvider {
        self.teacher.as_mut()
    }

    /// The provenance a sample produced at this scene's settings by this provider's teacher
    /// carries (what the store and a loaded model are compared with).
    pub fn provenance_for(&self, scene: &Scene) -> Provenance {
        self.scene_provenance(scene)
    }

    /// The isolated-atom energy set for `z`, if any (Hartree).
    pub fn reference_energy(&self, z: u8) -> Option<f64> {
        self.refs.get(&z).copied()
    }

    /// The training samples held (`None` before the first one).
    pub fn store(&self) -> Option<&SampleStore> {
        self.store.as_ref()
    }

    /// The sufficient statistics of the current fit (`None` before the first refit).
    pub fn accumulator(&self) -> Option<&Accumulator> {
        self.accumulator.as_ref()
    }

    /// Drops the model, its samples, statistics and reference energies: the provider is a
    /// pass-through to its teacher again (rule 1 of the policy). If the model was the reported
    /// surface the epoch moves, so the driver re-evaluates at once; that evaluation is on the
    /// teacher surface without a jump and the driver restarts its conserved-energy history
    /// (the honest outcome: the surface the invariant lived on is gone).
    pub fn forget(&mut self) {
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        if self.trusted() || was_model {
            self.epoch += 1;
        }
        self.store = None;
        self.accumulator = None;
        self.accumulated = 0;
        self.refs.clear();
        self.legacy_refs.clear();
        self.refs_dirty = false;
        self.species_dirty = false;
        self.model = None;
        self.report = None;
        self.envelope = None;
        self.training_residuals = None;
        self.audit_failed = None;
        self.pending_refit = false;
        self.pending_jump = None;
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        self.quarantine.clear();
        self.degree_cap = None;
        self.since_kept = THIN_EVERY;
        self.desc_cache.clear();
        self.desc_basis = None;
        self.recovery = None;
    }

    /// Writes the full `.model` sidecar (`surrogate::sidecar`): model, accumulator, last fit
    /// report and every sample held — including any taken since the last refit, which a load
    /// restores as pending. `InvalidInput` if nothing has been fitted yet.
    pub fn save_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        let (Some(model), Some(acc)) = (&self.model, &self.accumulator) else {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "no fitted model to save (sample and refit first)"));
        };
        let samples = self.store.as_ref().map_or(&[][..], |s| &s.samples[..]);
        let rule = self.legacy_refs.is_empty().then_some(surrogate::sidecar::REFERENCE_RULE);
        let flags: Vec<bool> = (0..samples.len()).map(|i| self.is_quarantined(i)).collect();
        surrogate::sidecar::write_with_quarantine(path, model, acc, self.report.as_ref(), samples, rule, &flags)
    }

    /// Species whose reference energy predates ruling S9 (from a sidecar without the rule).
    pub fn legacy_references(&self) -> Vec<u8> {
        self.legacy_refs.iter().copied().collect()
    }

    /// Replaces everything learned with a sidecar's contents (the caller has checked its
    /// provenance against the scene — `Dynamics` refuses a mismatch). The file must hold the
    /// fit report and at least the samples its accumulator was built from (a model-only file
    /// from `Model::write_sidecar` cannot be judged by the trust rule's training residuals and
    /// is refused). The training residuals are recomputed from the samples; the surface
    /// bookkeeping is a refit's (epoch, jump at the last evaluated positions).
    pub fn install_sidecar(&mut self, sc: surrogate::Sidecar) -> Result<(), String> {
        let surrogate::Sidecar { model, accumulator, report, samples, reference_rule, quarantine, .. } = sc;
        let Some(report) = report else {
            return Err("the file has no fit report (a model-only sidecar): it cannot be judged by the trust rule".into());
        };
        if accumulator.basis != model.basis {
            return Err("the file's accumulator and model disagree on the basis".into());
        }
        if samples.len() < accumulator.n_samples {
            return Err(format!(
                "the file holds {} samples but its fit used {}: without them the training residuals cannot be checked (a model-only sidecar)",
                samples.len(),
                accumulator.n_samples
            ));
        }
        if model.ref_energy.len() != model.basis.n_species() {
            return Err("the file's reference energies do not match its species".into());
        }
        let mut store = SampleStore::new(model.provenance.clone());
        for (i, smp) in samples.into_iter().enumerate() {
            store.push(smp).map_err(|e| format!("sample {i}: {e}"))?;
        }
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        self.refs = model.basis.species.iter().copied().zip(model.ref_energy.iter().copied()).collect();
        self.legacy_refs = match reference_rule.as_deref() {
            Some(surrogate::sidecar::REFERENCE_RULE) => Default::default(),
            _ => model.basis.species.iter().copied().collect(),
        };
        self.refs_dirty = false;
        self.species_dirty = store.samples.iter().flat_map(|s| s.z.iter()).any(|&z| model.basis.species_index(z).is_none());
        // the accumulated store prefix: the samples the fit used plus the quarantined ones among
        // them (quarantine only ever marks samples inside the prefix)
        let q = |i: usize| quarantine.get(i).copied().unwrap_or(false);
        let mut prefix = 0usize;
        let mut fitted = 0usize;
        while prefix < store.len() && (fitted < accumulator.n_samples || q(prefix)) {
            if !q(prefix) {
                fitted += 1;
            }
            prefix += 1;
        }
        self.accumulated = prefix;
        self.pending_refit = store.len() > self.accumulated;
        self.envelope = Some(accumulator.envelope.clone());
        self.accumulator = Some(accumulator);
        self.store = Some(store);
        self.quarantine = quarantine;
        self.degree_cap = (model.basis.body3 && model.basis.degree_cap != Basis::new(&[]).degree_cap).then_some(model.basis.degree_cap);
        self.desc_cache.clear();
        self.desc_basis = None;
        self.since_kept = THIN_EVERY;
        self.recovery = None;
        self.model = Some(model);
        self.report = Some(report);
        self.audit_failed = None;
        self.training_residuals = self.compute_training_residuals();
        self.consults = 0;
        self.last_variance = None;
        self.answered = 0;
        // Cannot fail here: `surface_moved` predicts only at a geometry whose species the new
        // basis covers (it checks `species_indices` first), and `describe` has no other error;
        // the state above is therefore never left half-installed (review m7).
        self.surface_moved(was_trusted, was_model).map_err(|e| e.to_string())
    }

    /// Provenance of a sample produced at this scene's settings by this teacher.
    fn scene_provenance(&self, scene: &Scene) -> Provenance {
        Provenance {
            grid_n: scene.cell.n,
            h: scene.cell.h,
            sigma: scene.sigma.unwrap_or(scene.cell.h),
            tol: self.teacher_tol,
            charge: scene.charge,
            multiplicity: scene.multiplicity,
            teacher: self.teacher.name().to_string(),
        }
    }

    /// Into the store (created with `prov` if absent); marks the basis dirty for a new species.
    fn push_sample(&mut self, s: Sample, prov: &Provenance) -> Result<(), StoreError> {
        let z = s.z.clone();
        if !self.quarantine.is_empty() {
            let n = self.store.as_ref().map_or(0, |st| st.len());
            self.quarantine.resize(n, false);
        }
        match &mut self.store {
            Some(store) => store.push_checked(s, prov)?,
            None => {
                let mut store = SampleStore::new(prov.clone());
                store.push(s)?;
                self.store = Some(store);
            }
        }
        if let Some(acc) = &self.accumulator {
            if z.iter().any(|&zi| acc.basis.species_index(zi).is_none()) {
                self.species_dirty = true;
            }
        }
        Ok(())
    }

    /// Reference energies in `basis.species` order.
    fn reference_vector(&self, basis: &Basis) -> Result<Vec<f64>, FitError> {
        basis
            .species
            .iter()
            .map(|&z| match self.refs.get(&z) {
                Some(&e) if e.is_finite() => Ok(e),
                _ => Err(FitError::MissingReference(z)),
            })
            .collect()
    }

    fn describe_provenance(p: &Provenance) -> String {
        format!(
            "charge {}, multiplicity {}, {}³ h {} σ {}",
            p.charge,
            p.multiplicity.map_or("lowest".to_string(), |m| m.to_string()),
            p.grid_n,
            p.h,
            p.sigma
        )
    }

    /// Rules 1–4: whether the model may be consulted for this scene.
    fn gate(&self, z: &[u8], prov: &Provenance) -> Result<(), Gate> {
        let Some(model) = &self.model else {
            return Err(Gate { reason: "surrogate: no model yet".into(), keep_sample: true, thin: false });
        };
        if model.provenance != *prov {
            return Err(Gate {
                reason: format!(
                    "surrogate: scene ({}) differs from the model's ({}) — sample not kept",
                    Self::describe_provenance(prov),
                    Self::describe_provenance(&model.provenance)
                ),
                keep_sample: false,
                thin: false,
            });
        }
        if let Some(&zu) = z.iter().find(|&&zi| model.basis.species_index(zi).is_none()) {
            let known: Vec<String> = model.basis.species.iter().map(|z| crate::provider::symbol(*z).to_string()).collect();
            return Err(Gate {
                reason: format!(
                    "surrogate: species {} is not in the model ({}) — the model is rebuilt with it at the next refit",
                    crate::provider::symbol(zu),
                    known.join(", ")
                ),
                keep_sample: true,
                thin: false,
            });
        }
        if let Some(why) = self.untrusted_reason() {
            return Err(Gate { reason: format!("surrogate untrusted: {why}"), keep_sample: true, thin: !self.trust_needs_samples() });
        }
        Ok(())
    }

    /// The trust rule fails for want of data (n_samples < Q′ or too few residual degrees of
    /// freedom): every teacher answer is then worth keeping (no thinning).
    fn trust_needs_samples(&self) -> bool {
        self.report.as_ref().is_none_or(|r| r.n_samples < r.q_prime || r.residual_dof_e < MIN_RESIDUAL_DOF || r.residual_dof_f < MIN_RESIDUAL_DOF)
    }

    /// Fills the per-sample descriptor cache under the current model's basis.
    fn ensure_desc_cache(&mut self) {
        let Some(basis) = self.model.as_ref().map(|m| m.basis.clone()) else { return };
        if self.desc_basis.as_ref() != Some(&basis) {
            self.desc_cache.clear();
            self.desc_basis = Some(basis.clone());
        }
        let Some(store) = self.store.as_ref() else { return };
        for s in &store.samples[self.desc_cache.len()..] {
            let atoms = match (describe(&basis, &s.pos, &s.z), basis.species_indices(&s.z)) {
                (Ok(d), Ok(sp)) => sp.into_iter().zip(d).map(|(si, di)| (si, di.b)).collect(),
                _ => Vec::new(), // a species the basis lacks: never "near" anything
            };
            self.desc_cache.push(atoms);
        }
    }

    /// The distance (training-spread units, L∞ over features; the worst query atom's nearest
    /// stored atom of its species) from the geometry to the nearest stored sample — only the
    /// quarantined ones with `quarantined_only`. ∞ without a model, an envelope or a candidate.
    fn nearest_stored(&mut self, pos: &[[f64; 3]], z: &[u8], quarantined_only: bool) -> f64 {
        self.ensure_desc_cache();
        let (Some(basis), Some(env)) = (self.desc_basis.as_ref(), self.envelope.as_ref()) else { return f64::INFINITY };
        let (Ok(desc), Ok(sp)) = (describe(basis, pos, z), basis.species_indices(z)) else { return f64::INFINITY };
        let dist = |s: usize, a: &[f64], b: &[f64]| -> f64 {
            let mut worst = 0.0f64;
            for (k, (x, y)) in a.iter().zip(b).enumerate() {
                let gap = (x - y).abs();
                if gap == 0.0 {
                    continue;
                }
                let u = env.unit(s, k);
                let d = if u > 0.0 { gap / u } else { f64::INFINITY };
                worst = worst.max(d);
            }
            worst
        };
        let mut worst_atom = 0.0f64;
        for (qs, qd) in sp.iter().zip(&desc) {
            let mut best = f64::INFINITY;
            for (i, atoms) in self.desc_cache.iter().enumerate() {
                if quarantined_only && !self.quarantine.get(i).copied().unwrap_or(false) {
                    continue;
                }
                for (s, b) in atoms {
                    if s == qs {
                        best = best.min(dist(*s, &qd.b, b));
                    }
                }
            }
            worst_atom = worst_atom.max(best);
        }
        worst_atom
    }

    /// Rule 5: the reason an atom is outside the training envelope, if one is.
    fn envelope_check(&self, model: &Model, pos: &[[f64; 3]], z: &[u8]) -> Option<String> {
        let env = self.envelope.as_ref()?;
        let (Ok(desc), Ok(s)) = (describe(&model.basis, pos, z), model.basis.species_indices(z)) else { return None };
        let mut worst = (0.0f64, 0usize, 0usize);
        for (i, d) in desc.iter().enumerate() {
            let (dist, k) = env.distance(s[i], &d.b);
            if dist > worst.0 {
                worst = (dist, i, k);
            }
        }
        (worst.0 > self.policy.max_descriptor_distance).then(|| {
            format!(
                "surrogate: atom {} is {:.1} training spreads outside the descriptor envelope (feature {}, limit {})",
                worst.1,
                worst.0,
                model.basis.feature_names()[worst.2],
                self.policy.max_descriptor_distance
            )
        })
    }

    /// Rules 6–7 on a prediction that carries stds.
    fn variance_check(&self, model: &Model, pred: &surrogate::Prediction, e_std: f64, f_std: &[f64]) -> VarianceVerdict {
        let n = pred.forces.len();
        let p = &self.policy;
        // 6. forces
        let mut uncertainty = 0.0f64;
        let mut worst: Option<(usize, f64, f64)> = None;
        for (j, (&sd, f)) in f_std.iter().zip(&pred.forces).enumerate() {
            let fs = sd / SQRT_3;
            uncertainty = uncertainty.max(fs);
            let bound = p.force_bound((f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt());
            if fs > bound && worst.is_none_or(|(_, s, b)| fs / bound > s / b) {
                worst = Some((j, fs, bound));
            }
        }
        if let Some((j, fs, bound)) = worst {
            return VarianceVerdict {
                accept: false,
                uncertainty,
                reason: Some(format!("surrogate: force std {fs:.2e} > bound {bound:.2e} Ha/Bohr on atom {j}")),
            };
        }
        // 7. energy, per atom, epistemic only
        let floor = if model.energy_per_atom { n as f64 * model.sigma_e } else { model.sigma_e };
        let epistemic_per_atom = (e_std * e_std - floor * floor).max(0.0).sqrt() / n as f64;
        if epistemic_per_atom > p.energy_std_per_atom {
            return VarianceVerdict {
                accept: false,
                uncertainty,
                reason: Some(format!(
                    "surrogate: epistemic energy std {epistemic_per_atom:.2e} > {:.2e} Ha/atom",
                    p.energy_std_per_atom
                )),
            };
        }
        VarianceVerdict { accept: true, uncertainty, reason: None }
    }

    /// The teacher's step: evaluate, keep the sample, build the result on the right surface
    /// (module doc, "One-surface accounting" and "Audit").
    fn teach(&mut self, scene: &Scene, h: Handoff, t0: Instant, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let r = self.teacher.evaluate(scene, should_abort)?;
        let mut notes = Vec::new();
        // thinning (module doc, "Trust recovery"): audits and the samples trust needs are kept
        let keep = h.keep_sample && {
            if h.thin && !h.audit && self.since_kept + 1 < THIN_EVERY {
                let d = self.nearest_stored(&h.pos, &h.z, false);
                let limit = THIN_FRACTION * self.policy.max_descriptor_distance;
                if d > limit {
                    true
                } else {
                    self.since_kept += 1;
                    notes.push(format!(
                        "surrogate: sample not kept — {d:.2} training spreads from a stored one (≤ {limit:.2}; one in {THIN_EVERY} such is kept)"
                    ));
                    false
                }
            } else {
                true
            }
        };
        if keep {
            self.since_kept = 0;
            let sample = Sample {
                pos: h.pos.clone(),
                z: h.z.clone(),
                energy: r.energy,
                forces: r.forces.clone(),
                tag: SampleTag::Fallback,
                weight: 1.0,
            };
            match self.push_sample(sample, &h.prov) {
                Ok(()) => self.pending_refit = true,
                Err(e) => notes.push(format!("surrogate: sample not kept ({e})")),
            }
        }
        let n = h.pos.len().max(1) as f64;
        let mut stats = r.stats;
        stats.source = Source::Fallback;
        stats.uncertainty = h.model.as_ref().and_then(|m| m.uncertainty);
        let (energy, on_model) = match h.model {
            Some(m) => {
                let gap = r.energy - m.energy;
                if h.audit && gap.abs() / n > self.policy.energy_std_per_atom {
                    // failed audit: this step is already on the teacher surface, and the jump
                    // of the reported energy at these positions is the gap itself
                    self.audit_failed = Some(gap);
                    // on the refresh right after a refit the refit's own jump (E_new − E_reported
                    // at these positions) is still pending: the reported energy moves by both
                    stats.refit_jump = Some(self.pending_jump.take().unwrap_or(0.0) + gap);
                    notes.insert(0, format!(
                        "surrogate audit failed: teacher − model {gap:+.2e} Ha ({:.1e} > {:.1e} Ha/atom) — untrusted until the next refit",
                        gap.abs() / n, self.policy.energy_std_per_atom
                    ));
                    (r.energy, false)
                } else {
                    stats.teacher_gap = Some(gap);
                    stats.model_forces = Some(m.forces);
                    stats.refit_jump = self.pending_jump.take();
                    notes.insert(0, format!("{} — teacher answered (teacher − model {gap:+.2e} Ha)", h.reason));
                    (m.energy, true)
                }
            }
            None => {
                self.pending_jump = None;
                notes.insert(0, format!("{} — teacher answered", h.reason));
                (r.energy, false)
            }
        };
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        stats.notes.extend(notes);
        self.last = Some(LastEval { pos: h.pos, z: h.z, energy, on_model, provenance: h.prov });
        Ok(ForceResult { energy, forces: r.forces, stats, display: r.display })
    }
}

impl ForceProvider for SurrogateKs {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        let t0 = Instant::now();
        let prov = self.scene_provenance(scene);
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // rules 1–4
        if let Err(Gate { reason, keep_sample, thin }) = self.gate(&z, &prov) {
            let h = Handoff { prov, pos, z, reason, keep_sample, thin, model: None, audit: false };
            return self.teach(scene, h, t0, should_abort);
        }
        // rules 5–8: consult the model; the variance on the cadence, the envelope every time
        let with_variance = {
            let k = self.policy.variance_cadence.max(1);
            let fresh = self.consults % k == 0 || self.last_variance.is_none();
            self.consults += 1;
            fresh
        };
        let model = self.model.as_ref().expect("gate passed");
        let pred = model
            .predict(&pos, &z, with_variance)
            .map_err(|e| ProviderError::Invalid(format!("surrogate: {e}")))?;
        let variance = match (pred.energy_std, pred.force_std.as_ref()) {
            (Some(e), Some(f)) => {
                let v = self.variance_check(model, &pred, e, f);
                self.last_variance = Some(v.clone());
                v
            }
            _ => self.last_variance.clone().expect("a variance verdict exists when none is evaluated"),
        };
        let uncertainty = Some(variance.uncertainty);
        let envelope = self.envelope_check(model, &pos, &z);
        let mut reason = envelope.or_else(|| if variance.accept { None } else { variance.reason.clone() });
        // a quarantined geometry is outside what the model may answer (ruling S15); its answer
        // is not kept: the model is known not to represent it, a new sample would only be
        // quarantined again
        let mut near_quarantine = false;
        if reason.is_none() && self.quarantined() > 0 {
            let d = self.nearest_stored(&pos, &z, true);
            let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
            if d <= limit {
                near_quarantine = true;
                reason = Some(format!("surrogate: {d:.2} training spreads from a quarantined geometry the model cannot represent (≤ {limit:.2}) — sample not kept"));
            }
        }
        let audit = reason.is_none() && {
            self.answered += 1;
            self.policy.audit_every > 0 && self.answered % self.policy.audit_every == 0
        };
        if reason.is_some() || audit {
            let reason = reason.unwrap_or_else(|| format!("surrogate audit (every {} answered steps)", self.policy.audit_every));
            let h = Handoff {
                prov,
                pos,
                z,
                reason,
                keep_sample: !near_quarantine,
                thin: true,
                model: Some(ModelAnswer { energy: pred.energy, forces: pred.forces, uncertainty }),
                audit,
            };
            return self.teach(scene, h, t0, should_abort);
        }
        let mut stats = EvalStats::default();
        stats.source = Source::Surrogate;
        stats.uncertainty = uncertainty;
        stats.refit_jump = self.pending_jump.take();
        stats.energy.total = pred.energy;
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        self.last = Some(LastEval { pos, z, energy: pred.energy, on_model: true, provenance: prov });
        Ok(ForceResult { energy: pred.energy, forces: pred.forces, stats, display: None })
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn reset(&mut self) {
        self.teacher.reset();
        self.last = None;
    }

    fn discontinuity(&mut self) {
        self.teacher.discontinuity();
        self.last = None;
    }

    fn surface_epoch(&self) -> u64 {
        self.epoch
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{DynCommand, Dynamics};
    use crate::provider::Surface;
    use crate::scene::{Cell, Nucleus};
    use crate::test_providers::{MorseDimer, Shared};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::sync::{Arc, Mutex};

    fn never() -> impl FnMut() -> bool {
        || false
    }

    fn cell() -> Cell {
        Cell { n: 33, h: 0.25 }
    }

    /// Two nuclei (Z `z0`, `z1`) `r` apart along `dir`, centred on `origin` (so a dimer sits
    /// near the middle of the 8 Bohr cell: the GPU test's energies must not see the faces).
    fn dimer(r: f64, dir: [f64; 3], origin: [f64; 3], z0: u8, z1: u8) -> Scene {
        let p0: [f64; 3] = std::array::from_fn(|a| origin[a] - 0.5 * r * dir[a]);
        let p1: [f64; 3] = std::array::from_fn(|a| origin[a] + 0.5 * r * dir[a]);
        Scene {
            nuclei: vec![Nucleus::new(z0, p0), Nucleus::new(z1, p1)],
            cell: cell(),
            charge: 0,
            multiplicity: None,
            sigma: None,
            external_field: [0.0; 3],
        }
    }

    fn h2(r: f64) -> Scene {
        dimer(r, [1.0, 0.0, 0.0], [0.0; 3], 1, 1)
    }

    fn unit_vector(rng: &mut StdRng) -> [f64; 3] {
        loop {
            let v: [f64; 3] = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if (1e-3..1.0).contains(&n) {
                return v.map(|x| x / n);
            }
        }
    }

    /// A Morse-teacher sample of the scene (the teacher's exact answer).
    fn morse_sample(teacher: &MorseDimer, scene: &Scene) -> Sample {
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (energy, forces) = teacher.energy_and_forces(&pos).unwrap();
        Sample { pos, z: scene.nuclei.iter().map(|n| n.z).collect(), energy, forces, tag: SampleTag::Scan, weight: 1.0 }
    }

    /// A surrogate over a shared Morse teacher with `n` scan samples on r ∈ [lo, hi] (random
    /// orientations and origins), the H reference set, refitted when `n > 0`.
    fn trained(lo: f64, hi: f64, n: usize, policy: FallbackPolicy) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
        let teacher = Shared::new(MorseDimer::h2());
        let handle = teacher.handle();
        let mut s = SurrogateKs::new(Box::new(teacher), policy).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let mut rng = StdRng::seed_from_u64(4);
        for i in 0..n {
            let r = lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64;
            let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
            let sc = dimer(r, unit_vector(&mut rng), origin, 1, 1);
            s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
        }
        if n > 0 {
            let rep = s.refit().expect("refit");
            println!("trained on {n} samples r ∈ [{lo}, {hi}]: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e} λ {:.3e}; dof_E {:.1} dof_F {:.1} q′ {}; converged {}",
                     rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.residual_dof_e, rep.residual_dof_f, rep.q_prime, rep.converged);
        }
        (s, handle)
    }

    fn calls(h: &Arc<Mutex<MorseDimer>>) -> usize {
        h.lock().unwrap().calls
    }

    fn max_force_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
        a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f64::max)
    }

    /// The Verlet swing to compare against: E_conserved's spread and largest per-step change
    /// over `steps` of a plain run on `teacher` from `scene` (same dt, no surrogate).
    fn verlet_reference(scene: &Scene, teacher: MorseDimer, steps: usize) -> (f64, f64) {
        let mut d = Dynamics::new(scene.clone(), Box::new(teacher), 0.25, 1);
        let mut prev = d.snapshot().energy_conserved;
        let (mut lo, mut hi, mut max_step) = (prev, prev, 0.0f64);
        for _ in 0..steps {
            d.apply(DynCommand::Step(1));
            let e = d.tick(f64::INFINITY, &mut never()).energy_conserved;
            lo = lo.min(e);
            hi = hi.max(e);
            max_step = max_step.max((e - prev).abs());
            prev = e;
        }
        (hi - lo, max_step)
    }

    fn surrogate_of(d: &mut Dynamics) -> &mut SurrogateKs {
        d.provider_mut().as_any_mut().and_then(|a| a.downcast_mut::<SurrogateKs>()).expect("a SurrogateKs")
    }

    /// Rule 1: nothing fitted → the teacher answers, the sample is kept, the result is on the
    /// teacher surface.
    #[test]
    fn no_model_means_fallback_and_a_sample() {
        let (mut s, h) = trained(0.0, 0.0, 0, FallbackPolicy::default());
        assert_eq!(s.name(), "surrogate(morse dimer)");
        assert!(!s.trusted() && s.model().is_none() && s.basis().is_none());
        let scene = h2(1.5);
        let r = s.evaluate(&scene, &mut never()).unwrap();
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 1);
        assert!(s.pending_refit());
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(r.stats.teacher_gap, None);
        assert_eq!(r.stats.refit_jump, None);
        assert_eq!(r.stats.uncertainty, None);
        let (e, f) = MorseDimer::h2().energy_and_forces(&[[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]).unwrap();
        assert_eq!(r.energy, e);
        assert_eq!(r.forces, f);
        assert!(r.stats.notes.iter().any(|n| n.contains("no model yet")), "{:?}", r.stats.notes);
        assert_eq!(s.epoch(), 0);
    }

    /// Rules 6–8 and the learning loop: a window-trained dimer answers inside the window to
    /// the teacher's accuracy, falls back outside it (keeping the sample), and answers there
    /// after the refit.
    #[test]
    fn window_trained_model_answers_inside_and_falls_back_outside() {
        let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        assert!(s.trusted(), "{:?}", s.report());
        assert_eq!(s.basis().unwrap().species, vec![1]);
        let teacher = MorseDimer::h2();

        let inside = h2(1.6);
        let r = s.evaluate(&inside, &mut never()).unwrap();
        let (e, f) = teacher.energy_and_forces(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]]).unwrap();
        println!("r = 1.6: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}", r.stats.source, (r.energy - e).abs(), max_force_diff(&r.forces, &f), r.stats.uncertainty);
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
        assert_eq!(calls(&h), 0, "the teacher was not called");
        assert!((r.energy - e).abs() < 1e-5);
        assert!(max_force_diff(&r.forces, &f) < 1e-4);
        assert!(r.stats.uncertainty.is_some_and(|u| u < 2e-3));
        assert_eq!(r.stats.teacher_gap, None);
        assert!(r.display.is_none());
        assert_eq!(r.stats.scf_steps, 0);
        assert_eq!(r.stats.energy.total, r.energy);

        // outside the window: the teacher answers on the model surface and the sample is kept
        let near = h2(3.0);
        let r = s.evaluate(&near, &mut never()).unwrap();
        println!("r = 3.0: {:?}, notes {:?}", r.stats.source, r.stats.notes);
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 41);
        assert!(s.pending_refit());
        let (e3, f3) = teacher.energy_and_forces(&[[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]).unwrap();
        assert_eq!(r.forces, f3, "the teacher's forces");
        let gap = r.stats.teacher_gap.expect("on the model surface");
        assert!((r.energy + gap - e3).abs() < 1e-12, "energy is the model's: {} + {gap} vs {e3}", r.energy);
        assert_eq!(r.stats.energy_surface(), Surface::Model);
        assert!(r.stats.energy.total == e3, "the teacher's breakdown is kept");
        assert!(r.stats.model_forces.is_some(), "the model forces travel with a model-surface fallback");

        let rep = s.refit().unwrap();
        let tr = s.training_residuals().unwrap();
        println!("after the refit with the r = 3.0 sample: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; training residuals E {:.1e}/atom, F {:.1e} (bound {:.1e})",
                 rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, tr.energy_per_atom, tr.force, tr.force_bound);
        assert_eq!(rep.n_samples, 41);
        assert!(!s.pending_refit());
        assert!(s.trusted(), "{tr:?}");
        assert_eq!(s.epoch(), 2, "the surface moved twice (first fit, this refit)");
        let r = s.evaluate(&near, &mut never()).unwrap();
        println!("r = 3.0 after refit: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}, notes {:?}", r.stats.source, (r.energy - e3).abs(), max_force_diff(&r.forces, &f3), r.stats.uncertainty, r.stats.notes);
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
        assert_eq!(calls(&h), 1);
        assert!((r.energy - e3).abs() < 1e-4);
        assert!(max_force_diff(&r.forces, &f3) < 1e-4);
        assert!(r.stats.refit_jump.is_some(), "the first evaluate after a refit reports the jump");
        assert_eq!(s.evaluate(&near, &mut never()).unwrap().stats.refit_jump, None, "once");

    }

    /// A surrogate over a shared Morse teacher trained on `n` window samples of the scripted
    /// teacher Morse(r) + a·sin(k r), k = 2π·`periods`/(hi − lo) — the sinusoid rides on the
    /// Morse energy with its consistent force −a k cos(k r) along the bond. With a = 0 this is
    /// `trained`.
    fn trained_oscillatory(lo: f64, hi: f64, n: usize, a: f64, periods: f64) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
        let teacher = Shared::new(MorseDimer::h2());
        let handle = teacher.handle();
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.set_reference_energy(1, MorseDimer::h2().e_atom);
        let k = 2.0 * std::f64::consts::PI * periods / (hi - lo);
        let mut rng = StdRng::seed_from_u64(4);
        for i in 0..n {
            let r = lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64;
            let origin = [rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)];
            let dir = unit_vector(&mut rng);
            let sc = dimer(r, dir, origin, 1, 1);
            let mut smp = morse_sample(&MorseDimer::h2(), &sc);
            smp.energy += a * (k * r).sin();
            let de = a * k * (k * r).cos(); // dE/dr; nucleus 1 sits at +r/2·dir
            for c in 0..3 {
                smp.forces[1][c] -= de * dir[c];
                smp.forces[0][c] += de * dir[c];
            }
            s.add_sample(smp, &sc).unwrap();
        }
        s.refit().expect("refit");
        (s, handle)
    }

    /// Ruling S8 (review C): basis misspecification is invisible to the posterior and caught by
    /// the training residuals. The fixture must be something NO basis the provider can build
    /// represents at the training points. The original one (40 Morse points on [1.2, 2.2] plus
    /// one at r = 4.0, whose −3.4e-3 Ha tail the f_c-damped body-2 n_max = 8 basis could not
    /// carry: 1e-2 Ha offset at a training point) was a *body-2* misspecification: the body-3
    /// default basis adds the radial shapes T_{n₁}T_{n₂}f_c²(r/r_cut)^{2l} and fits window + r =
    /// 4.0 (and + r = 5.0) to ~1e-8 Ha/atom, correctly earning trust (Task 6 fix round). The
    /// fixture now is Morse plus a sinusoid of 6 periods across the window, amplitude 1e-2 Ha:
    /// every feature of the default basis is a polynomial in r of degree ≤ 7 (body-2) or ≤ 4
    /// (body-3 at D = 4) times the smooth f_c, so no fit can follow 12 extrema on 1 Bohr — the
    /// two-noise evidence settles on "the data are noisy" (s_E ≈ 4e-3) while the max training
    /// energy residual per atom sits at ≈ a/2 ≥ 3 × `energy_std_per_atom`; the refit must leave
    /// the model untrusted with the note naming that maximum, and the fallback teacher answers.
    /// The same construction with a = 0 is the Morse window and is trusted (the control).
    #[test]
    fn misspecified_fit_is_untrusted_by_its_training_residuals() {
        let (mut s, h) = trained_oscillatory(1.2, 2.2, 40, 1e-2, 6.0);
        let rep = s.report().unwrap().clone();
        let tr = s.training_residuals().unwrap();
        println!("Morse + 1e-2·sin(6 periods): rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; γ_E {:.2} γ_F {:.2}; training residuals E {:.1e}/atom, F {:.1e} (bound {:.1e}); trusted {}; reason {:?}",
                 rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.gamma_e, rep.gamma_f, tr.energy_per_atom, tr.force, tr.force_bound, s.trusted(), s.untrusted_reason());
        assert!(rep.n_samples >= rep.q_prime && rep.residual_dof_e >= MIN_RESIDUAL_DOF && rep.residual_dof_f >= MIN_RESIDUAL_DOF, "{rep:?}: the sample-count conditions pass; only the residuals can refuse");
        assert!(!s.trusted(), "the misspecified fit must not be trusted");
        assert!(tr.energy_per_atom > 3e-3, "{tr:?}: max training energy residual ≥ 3× energy_std_per_atom");
        assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"), "{:?}", s.untrusted_reason());
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert!(r.stats.notes.iter().any(|n| n.contains("surrogate untrusted: max training energy residual")), "{:?}", r.stats.notes);
        assert_eq!(calls(&h), 1);
        // the epistemic energy criterion alone (the floor N·s_E excluded, as the provider does)
        // would not have caught it at an in-window geometry: the point of the rule
        let model = s.model().unwrap();
        let pred = model.predict(&[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], &[1, 1], true).unwrap();
        let std_e = pred.energy_std.unwrap();
        let epistemic_per_atom = (std_e * std_e - (2.0 * model.sigma_e).powi(2)).max(0.0).sqrt() / 2.0;
        println!("at r = 1.6 the model's epistemic energy std is {epistemic_per_atom:.1e} Ha/atom (total {std_e:.1e}, floor 2·s_E {:.1e}) against a {:.1e} Ha/atom training residual", 2.0 * model.sigma_e, tr.energy_per_atom);
        assert!(epistemic_per_atom < tr.energy_per_atom / 3.0, "the training residual is what refuses, not the posterior");

        // control: the same window without the sinusoid is the Morse fit and is trusted
        let (s0, _) = trained_oscillatory(1.2, 2.2, 40, 0.0, 6.0);
        let tr0 = s0.training_residuals().unwrap();
        println!("a = 0 control: training residuals E {:.1e}/atom, F {:.1e}; trusted {}", tr0.energy_per_atom, tr0.force, s0.trusted());
        assert!(s0.trusted(), "{:?}", s0.untrusted_reason());
        assert!(tr0.energy_per_atom < 1e-6);
    }

    /// Rule 4: a fit with too few samples is never consulted, and the note says why.
    #[test]
    fn untrusted_model_falls_back_with_the_reason() {
        let (mut s, h) = trained(1.2, 2.2, 5, FallbackPolicy::default());
        assert!(!s.trusted());
        let rep = s.report().unwrap().clone();
        assert!(rep.n_samples < rep.q_prime, "{rep:?}");
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 6);
        let want = format!("surrogate untrusted: n_samples {} < q_prime {}", rep.n_samples, rep.q_prime);
        assert!(r.stats.notes.iter().any(|n| n.contains(&want)), "{:?}", r.stats.notes);

        // enough samples for n_samples ≥ q′ (29 with the body-3 default basis, S = 1) but not
        // for 30 residual energy degrees of freedom (n_E − γ_E < 30 at 30 samples)
        let (mut s, _) = trained(1.2, 2.2, 30, FallbackPolicy::default());
        let rep = s.report().unwrap().clone();
        assert!(rep.n_samples >= rep.q_prime && rep.residual_dof_e < MIN_RESIDUAL_DOF, "{rep:?}");
        assert!(!s.trusted());
        let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
        assert!(r.stats.notes.iter().any(|n| n.contains("untrusted: residual_dof_e")), "{:?}", r.stats.notes);
    }

    /// Rule 3: a scene at other settings than the model's is never answered, even where the
    /// variance would pass; the teacher's answer is not kept.
    #[test]
    fn unseen_species_or_charge_falls_back_unconditionally() {
        let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        let mut cation = h2(1.6);
        cation.charge = 1;
        let r = s.evaluate(&cation, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 40, "a sample from another surface is not kept");
        assert_eq!(r.stats.teacher_gap, None);
        assert_eq!(r.stats.uncertainty, None);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert!(r.stats.notes.iter().any(|n| n.contains("charge 1") && n.contains("differs from the model's")), "{:?}", r.stats.notes);
        // a different σ, grid or multiplicity likewise
        let mut other = h2(1.6);
        other.sigma = Some(0.3);
        assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
        let mut other = h2(1.6);
        other.cell = Cell { n: 49, h: 0.25 };
        assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
        let mut other = h2(1.6);
        other.multiplicity = Some(3);
        assert_eq!(s.evaluate(&other, &mut never()).unwrap().stats.source, Source::Fallback);
        assert_eq!(s.samples(), 40);
        // an unknown species at a foreign provenance: one note (rule 3 precedes rule 2), the
        // sample not kept, no rebuild scheduled
        let mut h_he_cation = dimer(1.6, [0.0, 1.0, 0.0], [0.0; 3], 1, 2);
        h_he_cation.charge = 1;
        let r = s.evaluate(&h_he_cation, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        let surrogate_notes: Vec<&String> = r.stats.notes.iter().filter(|n| n.starts_with("surrogate")).collect();
        assert_eq!(surrogate_notes.len(), 1, "{:?}", r.stats.notes);
        assert!(surrogate_notes[0].contains("differs from the model's") && !surrogate_notes[0].contains("species"), "{:?}", r.stats.notes);
        assert_eq!(s.samples(), 40);
        assert!(!s.species_dirty);
        // the matching scene is still answered
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    }

    /// Rule 2 and the rebuild: a species the model does not know falls back unconditionally and
    /// is learned at the next refit — once its reference energy exists.
    #[test]
    fn new_species_falls_back_and_is_learned_at_the_next_refit() {
        // 180 samples: the rebuilt two-species body-3 basis has Q′ = 176 (ruling S11/S12), and
        // the trust rule needs n_samples ≥ Q′ after the rebuild
        let (mut s, h) = trained(1.2, 2.2, 180, FallbackPolicy::default());
        let h_he = dimer(1.6, [0.0, 1.0, 0.0], [0.1, -0.1, 0.0], 1, 2);
        let r = s.evaluate(&h_he, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 181, "kept: the rebuild needs it");
        assert!(r.stats.notes.iter().any(|n| n.contains("species He is not in the model (H)")), "{:?}", r.stats.notes);
        // no reference for He yet: the refit fails and the old model stands
        let old_epoch = s.epoch();
        assert_eq!(s.refit(), Err(FitError::MissingReference(2)));
        assert_eq!(s.basis().unwrap().species, vec![1]);
        assert!(s.trusted() && s.epoch() == old_epoch);
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate, "still answers H₂");
        // with the reference the basis is rebuilt with both species
        s.set_reference_energy(2, -2.83);
        let rep = s.refit().unwrap();
        assert_eq!(s.basis().unwrap().species, vec![1, 2]);
        assert_eq!(rep.q_prime, 8 * 3 + 2 * 76, "n_max·S(S+1)/2 body-2 classes plus S·P₃ body-3 columns (D = 4) with S = 2");
        assert_eq!(rep.n_samples, 181);
        let r = s.evaluate(&h_he, &mut never()).unwrap();
        assert!(!r.stats.notes.iter().any(|n| n.contains("species")), "{:?}", r.stats.notes);
        println!("H–He after the rebuild: {:?} {:?}", r.stats.source, r.stats.notes);
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate, "H₂ still answered after the rebuild");
    }

    /// Rule 5: near r_cut every body-2 feature vanishes, the variance is at its floor and would
    /// pass, and only the envelope knows the model has never seen an empty neighbourhood.
    #[test]
    fn envelope_catches_the_vanishing_feature_direction_the_variance_cannot() {
        let policy = FallbackPolicy::default();
        let (mut s, h) = trained(1.2, 2.2, 40, policy);
        // Beyond r_cut = 6.0 every body-2 feature AND its gradient is exactly zero: the leverage
        // vanishes, the stds sit on their floors, and the model would answer "Σ ref, no force"
        // where the Morse still binds by 2e-3 Ha with a 2e-3 Ha/Bohr force. (Just inside r_cut,
        // r = 5.9, the force variance still fires: the Chebyshev gradients do not vanish there.)
        let far = h2(6.5);
        let model = s.model().unwrap();
        let pred = model.predict(&[[-3.25, 0.0, 0.0], [3.25, 0.0, 0.0]], &[1, 1], true).unwrap();
        let fs = pred.force_std.as_ref().unwrap().iter().fold(0.0f64, |m, &x| m.max(x)) / SQRT_3;
        let floor = 2.0 * model.sigma_e;
        let e_epi = (pred.energy_std.unwrap().powi(2) - floor * floor).max(0.0).sqrt() / 2.0;
        let (e_t, f_t) = MorseDimer::h2().energy_and_forces(&[[-3.25, 0.0, 0.0], [3.25, 0.0, 0.0]]).unwrap();
        println!("r = 6.5: per-component force std {fs:.1e} (bound {:.0e}), epistemic energy std/atom {e_epi:.1e} (bound {:.0e}); model E − teacher {:+.1e} Ha, model F {:?} vs teacher {:+.1e}",
                 policy.force_std_abs, policy.energy_std_per_atom, pred.energy - e_t, pred.forces[0], f_t[0][0]);
        assert_eq!(pred.forces, vec![[0.0; 3]; 2], "beyond r_cut the model has no force");
        assert!(fs < 0.1 * policy.force_std_abs && e_epi < 0.1 * policy.energy_std_per_atom, "test premise: the variance passes");
        let r = s.evaluate(&far, &mut never()).unwrap();
        println!("notes {:?}", r.stats.notes);
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert!(r.stats.notes.iter().any(|n| n.contains("outside the descriptor envelope")), "{:?}", r.stats.notes);
        assert!(r.stats.uncertainty.is_some_and(|u| u < policy.force_std_abs), "the variance is still reported");
        assert_eq!(r.stats.energy_surface(), Surface::Model);
        // inside the window the envelope is quiet
        let r = s.evaluate(&h2(1.7), &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate);
        // the envelope's limit is the policy's
        s.set_policy(FallbackPolicy { max_descriptor_distance: f64::INFINITY, ..policy });
        let r = s.evaluate(&far, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    }

    /// One-surface accounting through the driver: with the policy forcing fallbacks, the atoms
    /// move on the teacher's forces while `energy_ks` is the model's and `teacher_gap` the known
    /// difference between the two surfaces.
    #[test]
    fn fallback_step_uses_dft_forces_but_books_the_model_energy() {
        // trained on D = 0.17; the live teacher then has D′ = 1.25 D (energies and forces differ).
        // Fallbacks are forced through the envelope (a negative limit rejects every geometry):
        // zero force thresholds would also fail the training-residual trust condition, which
        // uses the same numbers, and an untrusted model is a teacher-surface fallback.
        let always_fallback = FallbackPolicy { max_descriptor_distance: -1.0, audit_every: 0, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, always_fallback);
        let model = s.model().unwrap().clone();
        let mut live = MorseDimer::h2();
        live.d *= 1.25;
        h.lock().unwrap().d = live.d;

        let v0 = [3e-4, -1e-4, 2e-4];
        let mut scene = h2(1.55);
        scene.nuclei[1].vel = v0;
        let (m0, m1) = (scene.nuclei[0].mass, scene.nuclei[1].mass);
        let dt = units::fs_to_au_time(0.25);
        let mut d = Dynamics::new(scene.clone(), Box::new(s), 0.25, 1);
        let snap0 = d.snapshot();
        assert_eq!(snap0.source, Source::Fallback);
        let p0: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let (e_t0, f_t0) = live.energy_and_forces(&p0).unwrap();
        let e_m0 = model.predict(&p0, &[1, 1], false).unwrap().energy;
        assert_eq!(snap0.energy_ks, e_m0, "the model's energy");
        assert_eq!(snap0.forces, f_t0, "the teacher's forces");
        assert!((snap0.teacher_gap.unwrap() - (e_t0 - e_m0)).abs() < 1e-14);

        d.apply(DynCommand::Step(1));
        let snap1 = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap1.step, 1);
        // Verlet drift with the teacher's forces
        let r1: Vec<[f64; 3]> = [(p0[0], [0.0; 3], m0), (p0[1], v0, m1)]
            .iter()
            .zip(&f_t0)
            .map(|((r, v, m), f)| std::array::from_fn(|a| r[a] + dt * v[a] + 0.5 * dt * dt * f[a] / m))
            .collect();
        for (n, r) in snap1.scene.nuclei.iter().zip(&r1) {
            for a in 0..3 {
                assert!((n.pos[a] - r[a]).abs() < 1e-15, "{:?} vs {r:?}", n.pos);
            }
        }
        let p1: Vec<[f64; 3]> = snap1.scene.nuclei.iter().map(|n| n.pos).collect();
        let (e_t1, f_t1) = live.energy_and_forces(&p1).unwrap();
        let e_m1 = model.predict(&p1, &[1, 1], false).unwrap().energy;
        assert_eq!(snap1.source, Source::Fallback);
        assert_eq!(snap1.energy_ks, e_m1);
        assert_eq!(snap1.forces, f_t1);
        let gap = snap1.teacher_gap.unwrap();
        assert!((gap - (e_t1 - e_m1)).abs() < 1e-14, "gap {gap} vs {}", e_t1 - e_m1);
        // the known offset between the surfaces: 0.25 D [(1 − x)² − 1] at r₁
        let r = ((p1[1][0] - p1[0][0]).powi(2) + (p1[1][1] - p1[0][1]).powi(2) + (p1[1][2] - p1[0][2]).powi(2)).sqrt();
        let x = (-(r - 1.4)).exp();
        let offset = 0.25 * 0.17 * ((1.0 - x) * (1.0 - x) - 1.0);
        println!("gap {gap:+.6e} Ha, analytic offset {offset:+.6e} Ha, model error {:+.1e}", gap - offset);
        assert!((gap - offset).abs() < 1e-5, "the gap is the teacher change plus the model's fit error");
        assert_eq!(snap1.eval.energy.total, e_t1, "the teacher's breakdown");
        assert_eq!(snap1.fallback_fraction_100, 1.0);
        assert_eq!(calls(&h), 2);
    }

    /// The accounting across a refit: `E_conserved` is continuous to rounding when the model
    /// first becomes trusted (teacher surface → model surface), across the re-evaluation the
    /// driver does when the epoch changes, and `teacher_gap` is `None` on Surrogate steps.
    #[test]
    fn energy_conserved_is_continuous_across_a_forced_refit() {
        let (s, _h) = trained(0.0, 0.0, 0, FallbackPolicy::default());
        let mut scene = h2(1.55);
        scene.nuclei[1].vel = [4e-4, 1e-4, 0.0];
        let (_, ref_step) = verlet_reference(&scene, MorseDimer::h2(), 100);
        let mut d = Dynamics::new(scene.clone(), Box::new(s), 0.25, 1);
        let mut e_c = Vec::new();
        let mut sources = Vec::new();
        for _ in 0..50 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            sources.push(snap.source);
        }
        assert!(sources.iter().all(|&s| s == Source::Fallback));
        let before = d.snapshot();
        assert_eq!(before.model_epoch, 0);
        assert_eq!(before.eval.energy_surface(), Surface::Teacher);
        {
            let s = surrogate_of(&mut d);
            assert_eq!(s.samples(), 51);
            // 20 scan samples from a slightly different well (D 0.5 % deeper): the fit then
            // compromises between them and the 51 live samples, so the model surface differs
            // visibly from the teacher's at r_50 and the jump is a real number, not rounding —
            // while the training residuals (~2e-4/atom) stay inside the trust condition
            let mut deeper = MorseDimer::h2();
            deeper.d *= 1.005;
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..20 {
                let r = 1.2 + i as f64 / 19.0;
                let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
                s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
            }
            let rep = s.refit().unwrap();
            println!("mid-run refit: {} samples, dof_E {:.1}, dof_F {:.1}, s_F {:.1e}, training residuals {:?}", rep.n_samples, rep.residual_dof_e, rep.residual_dof_f, rep.sigma_f, s.training_residuals());
            assert!(s.trusted(), "{:?}", s.training_residuals());
            assert_eq!(s.epoch(), 1);
        }
        // a paused tick with nothing queued re-evaluates at the unchanged positions (the epoch
        // changed) and books the jump: nothing else moves, so E_conserved must not move either
        let refreshed = d.tick(f64::INFINITY, &mut never());
        assert_eq!(refreshed.step, 50);
        assert_eq!(refreshed.model_epoch, 1);
        assert_eq!(refreshed.eval.energy_surface(), Surface::Model);
        assert_eq!(refreshed.scene, before.scene, "nothing moved");
        let jump = refreshed.eval.refit_jump.expect("the first evaluate after the refit reports the jump");
        let delta = refreshed.energy_conserved - before.energy_conserved;
        println!("E_c before {:.12}, after the refresh {:.12} (Δ {:+.2e}); refit jump {jump:+.3e} Ha (E_ks {:.9} → {:.9}); source {:?}",
                 before.energy_conserved, refreshed.energy_conserved, delta, before.energy_ks, refreshed.energy_ks, refreshed.source);
        assert!((refreshed.energy_ks - before.energy_ks - jump).abs() < 1e-12, "the jump is the change of the reported energy");
        assert!(jump.abs() > 1e-5, "test premise: the surfaces differ visibly");
        assert!(delta.abs() < 1e-12, "E_conserved moved by {delta:+.3e} across the refit");
        assert!(refreshed.history.len() == before.history.len(), "no history restart");
        for _ in 0..50 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            if snap.source == Source::Surrogate {
                assert_eq!(snap.teacher_gap, None);
            }
            e_c.push(snap.energy_conserved);
            sources.push(snap.source);
        }
        let n_sur = sources.iter().filter(|&&s| s == Source::Surrogate).count();
        let n_fb = sources.iter().filter(|&&s| s == Source::Fallback).count();
        let fluct = e_c.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        println!("100 steps: {n_sur} surrogate, {n_fb} fallback; max |ΔE_c| per step {fluct:.2e} Ha; fallback fraction (last 100) {:.2}", d.snapshot().fallback_fraction_100);
        assert!(n_sur > 0, "the trained model answers somewhere on the path");
        // no step jumps: the fallback steps' force-mismatch work is booked, so only the Verlet
        // swing of this (0.15 Bohr off r_e) trajectory remains, the same as on the teacher alone
        println!("force-mismatch work booked over the run: {:+.3e} Ha; plain Verlet max per-step change {ref_step:.2e}", d.snapshot().work_mismatch);
        assert!(fluct < 1.5 * ref_step, "a jump: {fluct:.2e} vs the Verlet swing {ref_step:.2e}");
        assert!((d.snapshot().fallback_fraction_100 - n_fb as f64 / 100.0).abs() < 1e-12);
        // the history records each step's source (the viewer's sparkline colours by it): its
        // last 100 entries are exactly the 100 steps' sources, in order
        let snap = d.snapshot();
        assert_eq!(snap.history_source.len(), snap.history.len());
        assert_eq!(&snap.history_source[snap.history_source.len() - 100..], &sources[sources.len() - 100..]);
    }

    /// Review (d), ruling S8: on model-surface fallback steps the teacher's force moves the atoms
    /// while the invariant lives on the model surface; ΔF·Δr is booked into W, so E_conserved is
    /// flat to the Verlet swing, while the unbooked quantity (E_c + work_mismatch) oscillates with
    /// the difference of the two surfaces along the path.
    #[test]
    fn fallback_force_mismatch_work_is_booked() {
        // forced through the envelope (see `fallback_step_uses_dft_forces_but_books_the_model_energy`)
        let always_fallback = FallbackPolicy { max_descriptor_distance: -1.0, audit_every: 0, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, always_fallback);
        h.lock().unwrap().d *= 1.25; // the live teacher's well is 25 % deeper than the model's
        let mut scene = h2(1.55);
        scene.nuclei[0].vel = [-3e-4, 0.0, 0.0];
        scene.nuclei[1].vel = [3e-4, 0.0, 0.0];
        // the same trajectory on the teacher alone: what Verlet itself does to E_conserved
        // (the dimer starts 0.15 Bohr off r_e, 4e-3 Ha of oscillation energy, so the swing is
        // tens of µHa — far above the 1e-7 of a near-equilibrium start)
        let mut live = MorseDimer::h2();
        live.d *= 1.25;
        let (ref_spread, _) = verlet_reference(&scene, live, 200);
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        let e0 = d.snapshot().energy_conserved;
        let (mut lo_b, mut hi_b, mut lo_u, mut hi_u) = (e0, e0, e0, e0);
        for _ in 0..200 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            assert_eq!(snap.source, Source::Fallback);
            assert!(snap.eval.model_forces.is_some());
            let b = snap.energy_conserved;
            let u = snap.energy_conserved + snap.work_mismatch;
            lo_b = lo_b.min(b);
            hi_b = hi_b.max(b);
            lo_u = lo_u.min(u);
            hi_u = hi_u.max(u);
        }
        let snap = d.snapshot();
        println!("200 forced-fallback steps: E_conserved spread {:.2e} Ha booked, {:.2e} Ha unbooked; plain Verlet on the teacher {:.2e}; Σ mismatch work {:+.3e} Ha; last teacher − model gap {:+.2e}",
                 hi_b - lo_b, hi_u - lo_u, ref_spread, snap.work_mismatch, snap.teacher_gap.unwrap());
        // booked: the Verlet swing and nothing else (a wrong sign would double the unbooked spread)
        assert!(hi_b - lo_b < 1.5 * ref_spread, "booked {:.2e} vs the Verlet swing {ref_spread:.2e}", hi_b - lo_b);
        assert!(hi_u - lo_u > 10.0 * (hi_b - lo_b), "unbooked must visibly move: {:.2e} vs {:.2e}", hi_u - lo_u, hi_b - lo_b);
        assert!(snap.work_mismatch.abs() > 1e-6);
        assert_eq!(snap.fallback_fraction_100, 1.0);
    }

    /// Ruling S7: every `audit_every`-th answered step calls the teacher anyway; a passing audit
    /// is an ordinary model-surface fallback, a failed one (the teacher drifted away) marks the
    /// model untrusted and flips to the teacher surface with the gap booked as `refit_jump`, so
    /// E_conserved stays continuous and the next refit clears the flag.
    #[test]
    fn audit_calls_the_teacher_and_a_failed_audit_flips_the_surface_continuously() {
        let policy = FallbackPolicy { audit_every: 5, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, policy);
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1); // answered step 1
        assert_eq!(d.snapshot().source, Source::Surrogate);
        let mut e_c = vec![d.snapshot().energy_conserved];
        for i in 0..4 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            if i == 3 {
                // the 5th answered step is audited: teacher called, gap tiny, still trusted
                assert_eq!(snap.source, Source::Fallback, "{:?}", snap.eval.notes);
                assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit (every 5 answered steps)")), "{:?}", snap.eval.notes);
                let gap = snap.teacher_gap.expect("a passing audit is a model-surface fallback");
                println!("passing audit: teacher − model {gap:+.2e} Ha");
                assert!(gap.abs() < 1e-5);
                assert_eq!(snap.eval.refit_jump, None);
            } else {
                assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
            }
        }
        assert_eq!(calls(&h), 1);
        assert!(surrogate_of(&mut d).trusted());
        assert_eq!(surrogate_of(&mut d).samples(), 41);
        // the teacher drifts away from the model
        h.lock().unwrap().d *= 1.1;
        for i in 0..5 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            e_c.push(snap.energy_conserved);
            if i == 4 {
                let jump = snap.eval.refit_jump.expect("the failed audit books the flip");
                let delta = e_c[e_c.len() - 1] - e_c[e_c.len() - 2];
                println!("failed audit: jump {jump:+.3e} Ha, ΔE_c across the flip {delta:+.2e} Ha, notes {:?}", snap.eval.notes);
                assert_eq!(snap.source, Source::Fallback);
                assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate audit failed")), "{:?}", snap.eval.notes);
                assert_eq!(snap.teacher_gap, None);
                assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
                assert!(jump.abs() / 2.0 > 1e-3, "test premise: the drift exceeds the per-atom bound");
                // The jump itself is booked exactly (same positions). What remains is the
                // mixed-force half-kick of this one step: its first half-kick used the model's
                // forces at r_prev (a Surrogate step), its second the teacher's at r_new, and
                // relative to the teacher surface the first was an external force
                // F_model − F_teacher whose work ½ dt v·ΔF (≈ 5·10·2e-4·5e-3 ≈ 5e-6 here,
                // measured 1e-6) cannot be booked without F_teacher(r_prev). A refit avoids
                // this by refreshing at unchanged positions; an audit flips mid-trajectory,
                // once. Bound it at 1e-5 against the 1e-2 Ha jump it replaced.
                assert!(delta.abs() < 1e-5, "E_conserved must not see the flip beyond the half-kick remainder: {delta:+.2e}");
                assert!(snap.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", snap.notes);
            } else {
                assert_eq!(snap.source, Source::Surrogate, "{:?}", snap.eval.notes);
            }
        }
        assert_eq!(calls(&h), 2);
        assert!(!surrogate_of(&mut d).trusted());
        assert!(surrogate_of(&mut d).audit_failed().is_some());
        // every step until a refit is a teacher-surface fallback with the reason
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.source, Source::Fallback);
        assert!(snap.eval.notes.iter().any(|n| n.contains("surrogate untrusted: audit found teacher − model")), "{:?}", snap.eval.notes);
        assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
        let fluct = e_c.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
        println!("max |ΔE_c| per step over the run incl. the flip: {fluct:.2e} Ha");
        assert!(fluct < 1e-5);
        // a refit clears the audit flag (whatever the residuals then say)
        let s = surrogate_of(&mut d);
        s.refit().unwrap();
        assert!(s.audit_failed().is_none());
        println!("after the refit over the drifted samples: trusted {} ({:?})", s.trusted(), s.training_residuals());
    }

    /// Task 5 carry-over (Task 4 re-review): a failed audit on the refresh evaluation right after
    /// a refit must book the refit's pending jump AND the audit's gap — the reported energy at the
    /// unchanged positions moves from the old model's to the teacher's, which is both.
    #[test]
    fn a_failed_audit_on_the_refresh_after_a_refit_books_the_pending_jump_too() {
        let policy = FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() };
        let (s, h) = trained(1.2, 2.2, 40, policy);
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        let mut d = Dynamics::new(scene, Box::new(s), 0.25, 1);
        for _ in 0..5 {
            d.apply(DynCommand::Step(1));
            assert_eq!(d.tick(f64::INFINITY, &mut never()).source, Source::Surrogate);
        }
        let before = d.snapshot();
        {
            let s = surrogate_of(&mut d);
            // a refit that moves the model surface a little (scan samples from a 0.5 % deeper well)
            let mut deeper = MorseDimer::h2();
            deeper.d *= 1.005;
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..20 {
                let r = 1.2 + i as f64 / 19.0;
                let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
                s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
            }
            s.refit().unwrap();
            assert!(s.trusted(), "{:?}", s.untrusted_reason());
            let pos: Vec<[f64; 3]> = before.scene.nuclei.iter().map(|n| n.pos).collect();
            let pending = s.model().unwrap().predict(&pos, &[1, 1], false).unwrap().energy - before.energy_ks;
            println!("the refit's pending jump at r_last: {pending:+.3e} Ha");
            assert!(pending.abs() > 1e-6, "test premise: the refit moved the surface visibly");
            // every answered step is audited from now on, and the teacher has drifted 10 %:
            // the refresh evaluation is a failing audit
            s.set_policy(FallbackPolicy { audit_every: 1, ..FallbackPolicy::default() });
        }
        h.lock().unwrap().d *= 1.1;
        let refreshed = d.tick(f64::INFINITY, &mut never());
        assert_eq!(refreshed.scene, before.scene, "nothing moved");
        assert!(refreshed.eval.notes.iter().any(|n| n.contains("surrogate audit failed")), "{:?}", refreshed.eval.notes);
        assert_eq!(refreshed.eval.energy_surface(), Surface::Teacher);
        let jump = refreshed.eval.refit_jump.expect("the flip is booked");
        let delta = refreshed.energy_conserved - before.energy_conserved;
        let reported = refreshed.energy_ks - before.energy_ks;
        println!("refresh = failed audit: booked jump {jump:+.6e} Ha, reported energy change {reported:+.6e} Ha, ΔE_c {delta:+.2e}");
        assert!((jump - reported).abs() < 1e-12, "the booked jump is the whole change of the reported energy");
        assert!(delta.abs() < 1e-12, "E_conserved moved by {delta:+.3e} across the refresh");
        assert!(refreshed.notes.iter().all(|n| !n.contains("history restarts")), "{:?}", refreshed.notes);
    }

    /// Ruling S15, fallback thinning (module doc, "Trust recovery"): fallbacks at one geometry
    /// while the model is untrusted for its residuals (not for want of data) keep one sample in
    /// [`THIN_EVERY`]; a geometry far from every stored one is kept at once; while the trust rule
    /// still needs samples (n_samples < Q′) every answer is kept.
    #[test]
    fn thinning_keeps_one_in_ten_near_identical_fallbacks_and_every_far_one() {
        let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        assert!(s.trusted());
        // untrusted by its residuals (an impossible energy threshold), not by its sample count
        s.set_policy(FallbackPolicy { energy_std_per_atom: 1e-12, ..FallbackPolicy::default() });
        assert!(s.untrusted_reason().unwrap().starts_with("max training energy residual"));
        let n0 = s.samples();
        let mut thinned_notes = 0;
        for _ in 0..25 {
            let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
            assert_eq!(r.stats.source, Source::Fallback);
            thinned_notes += r.stats.notes.iter().filter(|n| n.contains("sample not kept")).count();
        }
        println!("25 fallbacks at r = 1.6: {} kept, {thinned_notes} thinned", s.samples() - n0);
        assert_eq!(s.samples() - n0, 3, "the 1st, 11th and 21st are kept");
        assert_eq!(thinned_notes, 22);
        // far from every stored geometry: kept right away (4 thinned since the last kept one)
        let far = s.nearest_stored(&h2(4.0).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1], false);
        s.evaluate(&h2(4.0), &mut never()).unwrap();
        println!("r = 4.0 is {far:.1} training spreads from the nearest stored sample: kept");
        assert!(far > THIN_FRACTION * FallbackPolicy::default().max_descriptor_distance);
        assert_eq!(s.samples() - n0, 4);
        // a model that still needs data keeps every answer
        let (mut few, _) = trained(1.2, 2.2, 5, FallbackPolicy::default());
        assert!(few.untrusted_reason().unwrap().starts_with("n_samples"));
        let m0 = few.samples();
        for _ in 0..5 {
            few.evaluate(&h2(1.6), &mut never()).unwrap();
        }
        assert_eq!(few.samples() - m0, 5);
    }

    /// Ruling S15, quarantine: one stray sample (the teacher's energy at an in-window geometry
    /// 0.1 Ha off — the model cannot fit it and the rest of the window together) untrusted the
    /// whole model before this round, for good. Now the refit quarantines it: trusted again,
    /// the sample held (store, sidecar) but excluded, its geometry a fallback whose answer is not
    /// kept, the window answered by the model; Forget clears the quarantine; the flags survive a
    /// sidecar round trip.
    #[test]
    fn a_stray_sample_is_quarantined_and_trust_returns() {
        let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
        let stray = dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1);
        let mut bad = morse_sample(&MorseDimer::h2(), &stray);
        bad.energy += 0.1;
        s.add_sample(bad, &stray).unwrap();
        let rep = s.refit().unwrap();
        println!("after the stray sample: {} samples, rmse E/atom {:.1e}; trusted {}; quarantined {}; note {:?}",
                 rep.n_samples, rep.rmse_energy_per_atom, s.trusted(), s.quarantined(), s.recovery_note());
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        assert_eq!(s.quarantined(), 1);
        assert!(s.is_quarantined(40), "the stray sample (index 40) is the one held back");
        assert!(s.recovery_note().unwrap().contains("1 samples quarantined"));
        assert_eq!(s.samples(), 41, "never silently dropped");
        assert_eq!(rep.n_samples, 40, "excluded from the fit");
        // its geometry falls back (and teaches nothing new); the window is the model's
        let calls0 = calls(&h);
        let r = s.evaluate(&stray, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert!(r.stats.notes.iter().any(|n| n.contains("quarantined geometry")), "{:?}", r.stats.notes);
        assert_eq!((s.samples(), calls(&h)), (41, calls0 + 1));
        let near = |s: &mut SurrogateKs, r: f64| s.nearest_stored(&h2(r).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1], true);
        let (d18, d20) = (near(&mut s, 1.8), near(&mut s, 2.0));
        let r18 = s.evaluate(&h2(1.8), &mut never()).unwrap();
        println!("from the quarantined geometry: r = 1.8 at {d18:.2}, r = 2.0 at {d20:.2} training spreads; r = 1.8 answered by {:?} {:?}", r18.stats.source, r18.stats.notes);
        assert_eq!(r18.stats.source, Source::Surrogate, "the window around the stray sample is still the model's");
        // the flags in the sidecar
        let path = std::env::temp_dir().join(format!("quarantine-{}.model", std::process::id()));
        s.save_sidecar(&path).unwrap();
        let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
        back.install_sidecar(surrogate::sidecar::read(&path).unwrap()).unwrap();
        assert_eq!((back.quarantined(), back.is_quarantined(40), back.trusted()), (1, true, true));
        assert_eq!(back.evaluate(&stray, &mut never()).unwrap().stats.source, Source::Fallback);
        let _ = std::fs::remove_file(&path);
        // Forget clears it
        s.forget();
        assert_eq!(s.quarantined(), 0);
    }

    /// The driver's safety net: a change of energy surface between consecutive evaluations that
    /// carries no `refit_jump` restarts the conserved-energy history with a note.
    #[test]
    fn a_surface_change_without_a_jump_restarts_the_history() {
        /// A Morse teacher that starts labelling its answers `Source::Surrogate` after `flip_at`
        /// evaluations (the energy stays the teacher's: a bare class flip, no jump reported).
        struct Flipper {
            inner: MorseDimer,
            flip_at: usize,
        }
        impl ForceProvider for Flipper {
            fn evaluate(&mut self, scene: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
                let mut r = self.inner.evaluate(scene, a)?;
                if self.inner.calls > self.flip_at {
                    r.stats.source = Source::Surrogate;
                }
                Ok(r)
            }
            fn name(&self) -> &'static str {
                "flipper"
            }
            fn reset(&mut self) {}
        }
        let mut scene = h2(1.5);
        scene.nuclei[1].vel = [2e-4, 0.0, 0.0];
        // evaluate 1 = Dynamics::new; evaluates 2–4 = steps 1–3 (Teacher); evaluate 5 = step 4 (Model)
        let mut d = Dynamics::new(scene, Box::new(Flipper { inner: MorseDimer::h2(), flip_at: 4 }), 0.25, 1);
        for _ in 0..3 {
            d.apply(DynCommand::Step(1));
            let snap = d.tick(f64::INFINITY, &mut never());
            assert_eq!(snap.eval.energy_surface(), Surface::Teacher);
        }
        assert_eq!(d.snapshot().history.len(), 3);
        d.apply(DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert_eq!(snap.source, Source::Surrogate);
        assert_eq!(snap.eval.energy_surface(), Surface::Model);
        assert_eq!(snap.history.len(), 1, "the history restarted at this step");
        assert!(snap.notes.last().is_some_and(|n| n.contains("energy surface changed (Teacher → Model) without a booked jump")), "{:?}", snap.notes);
        assert_eq!(snap.work_mismatch, 0.0);
    }

    /// GPU H₂ round trip: the teacher's own samples (40 centred scan geometries at random
    /// orientations, r ∈ [1.1, 2.0]) train a model that reproduces the teacher at R_e (body-2
    /// only; Task 7 pins the tiered targets). The brief's bounds, 1e-4 Ha / 2e-3 Ha/Bohr, are
    /// pinned at 41³ (10 Bohr box), where the teacher's own orientation scatter is at the
    /// egg-box level (s_E 1.3e-5/atom, s_F 1.4e-4). At 33³ (8 Bohr) the dimer's far atom is
    /// 2.8 Bohr from a face and the energy depends on the orientation at 7e-5/atom (the H atom
    /// itself is 8 mHa higher than at 41³: confinement, the provider's own "face density"
    /// warning) — a bound below the teacher's reproducibility would measure the box, so the 33³
    /// run asserts the force bound and a calibration bound (|ΔE| within 3× the fit's total
    /// energy scatter) and records the numbers.
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn gpu_teacher_h2_round_trip() {
        let Some(ctx) = gpu::GpuContext::new_headless() else {
            eprintln!("sim: no GPU adapter — skipped");
            return;
        };
        let (de, df, s_e) = gpu_round_trip(ctx.clone(), 41);
        assert!(de < 1e-4, "41³: |ΔE| = {de:.2e}");
        assert!(df < 2e-3, "41³: max |ΔF| = {df:.2e}");
        assert!(s_e < 3e-5, "41³: the teacher's energy scatter is at the egg-box level, s_E = {s_e:.1e}");
        let (de, df, s_e) = gpu_round_trip(ctx, 33);
        assert!(df < 2e-3, "33³: max |ΔF| = {df:.2e}");
        assert!(de < 3.0 * 2.0 * s_e, "33³: |ΔE| = {de:.2e} against the fit's total energy scatter 2·s_E = {:.2e}", 2.0 * s_e);
    }

    /// One round trip on an n³ grid: (|ΔE| at R_e, max |ΔF| at R_e, the fit's s_E per atom).
    fn gpu_round_trip(ctx: gpu::GpuContext, n_grid: usize) -> (f64, f64, f64) {
        let cell = Cell { n: n_grid, h: 0.25 };
        let with_cell = |mut sc: Scene| {
            sc.cell = cell.clone();
            sc
        };
        let t0 = Instant::now();
        let mut teacher = crate::provider::GpuKs::new(ctx.clone(), 1e-6);
        teacher.set_display(false);
        let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
        // the H reference from the teacher itself (an isolated atom at the same settings)
        let atom = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell: cell.clone(), charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] };
        let e_h = s.evaluate(&atom, &mut never()).unwrap().energy;
        s.set_reference_energy(1, e_h);
        // 40 scan geometries on r ∈ [1.1, 2.0]: every fallback keeps its sample
        let mut rng = StdRng::seed_from_u64(7);
        let n = 40;
        for i in 0..n {
            let r = 1.1 + 0.9 * i as f64 / (n - 1) as f64;
            let origin = [rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2), rng.gen_range(-0.2..0.2)];
            let sc = with_cell(dimer(r, unit_vector(&mut rng), origin, 1, 1));
            // unrelated geometries: keep the warm state, drop the density extrapolation (R13)
            s.discontinuity();
            let r = s.evaluate(&sc, &mut never()).expect("GPU teacher");
            assert_eq!(r.stats.source, Source::Fallback);
        }
        assert_eq!(s.samples(), n + 1);
        let t_scan = t0.elapsed().as_secs_f64();
        let rep = s.refit().expect("refit");
        println!("GPU H₂ {n_grid}³: E_H = {e_h:+.6} Ha; {} samples in {t_scan:.1} s; fit rmse E/atom {:.2e} Ha, F {:.2e} Ha/Bohr; s_E {:.1e} s_F {:.1e} λ {:.3e}; dof_E {:.1} dof_F {:.1}; trusted {}",
                 rep.n_samples, rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.lambda, rep.residual_dof_e, rep.residual_dof_f, s.trusted());
        assert!(s.trusted(), "{rep:?}");
        let re = with_cell(h2(1.45));
        s.discontinuity();
        let t1 = Instant::now();
        let r = s.evaluate(&re, &mut never()).unwrap();
        let t_sur = t1.elapsed().as_secs_f64() * 1e3;
        println!("surrogate at R = 1.45: {:?} E {:+.6} Ha, F_x(0) {:+.4e}, uncertainty {:?}, {t_sur:.2} ms, notes {:?}", r.stats.source, r.energy, r.forces[0][0], r.stats.uncertainty, r.stats.notes);
        assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
        let mut direct = crate::provider::GpuKs::new(ctx, 1e-6);
        direct.set_display(false);
        let d = direct.evaluate(&re, &mut never()).unwrap();
        let de = (r.energy - d.energy).abs();
        let df = max_force_diff(&r.forces, &d.forces);
        println!("direct GPU: E {:+.6} Ha, F_x(0) {:+.4e}; |ΔE| {de:.2e} Ha, max|ΔF| {df:.2e} Ha/Bohr", d.energy, d.forces[0][0]);
        (de, df, rep.sigma_e)
    }
}
