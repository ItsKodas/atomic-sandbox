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
//!    because it is the teacher's own scatter — the part of the residual no number of samples
//!    removes (egg-box, SCF lag): a criterion that included it would fall back permanently once
//!    N·s_E exceeds the threshold (N ≳ 100 at s_E = 1e-5) while learning nothing from the
//!    extra teacher calls; the epistemic part is what new samples shrink, so it is the part a
//!    fallback can act on. Otherwise → teacher; sample kept.
//! 8. Else the model answers: `Source::Surrogate`, `uncertainty` = max_j force_std[j]/√3, no
//!    density (`display: None`: the surrogate has no electrons).
//!
//! `uncertainty` is also reported on policy fallbacks (rules 5–7), so the readouts can show how
//! far off the model was. `variance_cadence` = k evaluates the O(Q′²) variance only on every
//! k-th consultation and reuses the previous verdict in between (k = 1 here; Task 9 tunes);
//! the envelope (rule 5) is checked on every consultation, it is cheap.
//!
//! # Trust rule (from the Task 3 fit, binding)
//!
//! The model may answer only when the last `FitReport` has `n_samples ≥ q_prime` (distinct
//! geometries, not rows: the 3N force rows of one sample probe one geometry) AND
//! `residual_dof_e ≥ 30` AND `residual_dof_f ≥ 30` (the two noise levels are then known to
//! ≈ 25 %, so the variance means something). Below that the fit is a curve through too few
//! points with a noise estimate collapsing towards its floor — overconfident inside its span.
//! [`SurrogateKs::trusted`] exposes the verdict; every evaluate of an untrusted model is a
//! Fallback (rule 4) and the samples accumulate until a refit earns trust.
//!
//! # One-surface energy accounting (spec §5)
//!
//! The driver's `E_conserved` must be computed on ONE potential-energy surface, or every hand-
//! off between model and teacher would look like a drift. So on a Fallback step with a trusted
//! model the result is `ForceResult { energy: E_model(r), forces: F_teacher(r) }` and
//! `teacher_gap = E_teacher − E_model` is reported separately: the atoms move on the teacher's
//! forces (accuracy), the invariant stays on the model surface (continuity), and the gap is a
//! readout of the model's error rather than a corruption of the drift. When no trusted model
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
}

impl Default for FallbackPolicy {
    fn default() -> FallbackPolicy {
        FallbackPolicy {
            force_std_abs: 2e-3,
            force_std_rel: 0.10,
            energy_std_per_atom: 1e-3,
            max_descriptor_distance: 3.0,
            variance_cadence: 1,
        }
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

/// The variance verdict of the last consultation (reused between variance evaluations).
#[derive(Clone)]
struct Verdict {
    accept: bool,
    uncertainty: Option<f64>,
    reason: Option<String>,
}

/// Why the model is not consulted (rules 1–4), and whether the teacher's answer is kept.
struct Gate {
    reason: String,
    keep_sample: bool,
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
    /// A reference changed since the accumulator was built → rebuild at the next refit.
    refs_dirty: bool,
    /// The store holds a species the accumulator's basis lacks → rebuild at the next refit.
    species_dirty: bool,
    model: Option<Model>,
    report: Option<FitReport>,
    envelope: Option<Envelope>,
    epoch: u64,
    pending_refit: bool,
    pending_jump: Option<f64>,
    last: Option<LastEval>,
    consults: usize,
    last_verdict: Option<Verdict>,
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
            refs_dirty: false,
            species_dirty: false,
            model: None,
            report: None,
            envelope: None,
            epoch: 0,
            pending_refit: false,
            pending_jump: None,
            last: None,
            consults: 0,
            last_verdict: None,
        }
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

    pub fn set_policy(&mut self, p: FallbackPolicy) {
        self.policy = p;
        self.consults = 0;
        self.last_verdict = None;
    }

    /// Teach from an existing sample (a sampling pass, a loaded sidecar). `scene` supplies the
    /// settings the sample was produced at (its provenance); the first sample pins the store's.
    pub fn add_sample(&mut self, s: Sample, scene: &Scene) -> Result<(), StoreError> {
        let prov = self.scene_provenance(scene);
        self.push_sample(s, &prov)
    }

    /// Isolated-atom energy of species `z` (Hartree), the model's reference (Z-only: computed
    /// by the teacher, never a table). A change after a fit rebuilds the accumulator at the next
    /// refit.
    pub fn set_reference_energy(&mut self, z: u8, e: f64) {
        if self.refs.insert(z, e) != Some(e) {
            self.refs_dirty = true;
        }
    }

    /// The trust rule (module doc): a model exists and its report passes.
    pub fn trusted(&self) -> bool {
        self.model.is_some() && self.untrusted_reason().is_none()
    }

    fn untrusted_reason(&self) -> Option<String> {
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
        None
    }

    /// Rebuild the model from the samples (module doc): the accumulator takes the samples added
    /// since the last refit, or is rebuilt from the store after a species or reference change;
    /// then one hyperparameter search and solve. Computes the refit jump for the next evaluate
    /// and bumps the surface epoch when the surface moved. On an error the previous model,
    /// report and envelope stay in place.
    pub fn refit(&mut self) -> Result<FitReport, FitError> {
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
            Basis::new(&species)
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
        for s in &store.samples[accumulated..] {
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
        self.pending_refit = false;
        self.consults = 0;
        self.last_verdict = None;

        // Surface bookkeeping (module doc): the jump for the next evaluate, and the epoch.
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        let model = self.model.as_ref().expect("set above");
        let next_is_model = self.trusted()
            && match &self.last {
                None => true,
                Some(l) => l.provenance == provenance && model.basis.species_indices(&l.z).is_ok(),
            };
        self.pending_jump = match (&self.last, next_is_model) {
            (Some(l), true) => {
                let e_new = model.predict(&l.pos, &l.z, false).map_err(FitError::Describe)?.energy;
                Some(e_new - l.energy)
            }
            _ => None,
        };
        if next_is_model || was_model {
            self.epoch += 1;
        }
        Ok(report)
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
            return Err(Gate { reason: "surrogate: no model yet".into(), keep_sample: true });
        };
        if let Some(&zu) = z.iter().find(|&&zi| model.basis.species_index(zi).is_none()) {
            let known: Vec<String> = model.basis.species.iter().map(|z| crate::provider::symbol(*z).to_string()).collect();
            return Err(Gate {
                reason: format!(
                    "surrogate: species {} is not in the model ({}) — the model is rebuilt with it at the next refit",
                    crate::provider::symbol(zu),
                    known.join(", ")
                ),
                keep_sample: true,
            });
        }
        if model.provenance != *prov {
            return Err(Gate {
                reason: format!(
                    "surrogate: scene ({}) differs from the model's ({}) — sample not kept",
                    Self::describe_provenance(prov),
                    Self::describe_provenance(&model.provenance)
                ),
                keep_sample: false,
            });
        }
        if let Some(why) = self.untrusted_reason() {
            return Err(Gate { reason: format!("surrogate untrusted: {why}"), keep_sample: true });
        }
        Ok(())
    }

    /// Rules 5–7 on a prediction. `with_variance` says whether `pred` carries fresh stds; if it
    /// does not, the previous variance verdict is reused.
    fn judge(&self, model: &Model, pred: &surrogate::Prediction, pos: &[[f64; 3]], z: &[u8]) -> Verdict {
        let n = pos.len();
        let p = &self.policy;
        // 5. envelope
        if let Some(env) = &self.envelope {
            if let (Ok(desc), Ok(s)) = (describe(&model.basis, pos, z), model.basis.species_indices(z)) {
                let mut worst = (0.0f64, 0usize, 0usize);
                for (i, d) in desc.iter().enumerate() {
                    let (dist, k) = env.distance(s[i], &d.b);
                    if dist > worst.0 {
                        worst = (dist, i, k);
                    }
                }
                if worst.0 > p.max_descriptor_distance {
                    let names = model.basis.feature_names();
                    let uncertainty = pred.force_std.as_ref().map(|f| f.iter().fold(0.0f64, |m, &x| m.max(x)) / SQRT_3);
                    return Verdict {
                        accept: false,
                        uncertainty,
                        reason: Some(format!(
                            "surrogate: atom {} is {:.1} training spreads outside the descriptor envelope (feature {}, limit {})",
                            worst.1, worst.0, names[worst.2], p.max_descriptor_distance
                        )),
                    };
                }
            }
        }
        let (Some(e_std), Some(f_std)) = (pred.energy_std, pred.force_std.as_ref()) else {
            return match &self.last_verdict {
                Some(v) => v.clone(),
                None => Verdict { accept: false, uncertainty: None, reason: Some("surrogate: no variance available".into()) },
            };
        };
        // 6. forces
        let mut uncertainty = 0.0f64;
        let mut worst: Option<(usize, f64, f64)> = None;
        for (j, (&sd, f)) in f_std.iter().zip(&pred.forces).enumerate() {
            let fs = sd / SQRT_3;
            uncertainty = uncertainty.max(fs);
            let fmag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            let bound = p.force_std_abs.max(p.force_std_rel * fmag).min(FORCE_STD_BOUND_MAX);
            if fs > bound && worst.is_none_or(|(_, s, b)| fs / bound > s / b) {
                worst = Some((j, fs, bound));
            }
        }
        if let Some((j, fs, bound)) = worst {
            return Verdict {
                accept: false,
                uncertainty: Some(uncertainty),
                reason: Some(format!("surrogate: force std {fs:.2e} > bound {bound:.2e} Ha/Bohr on atom {j}")),
            };
        }
        // 7. energy, per atom, epistemic only
        let floor = if model.energy_per_atom { n as f64 * model.sigma_e } else { model.sigma_e };
        let epistemic_per_atom = (e_std * e_std - floor * floor).max(0.0).sqrt() / n as f64;
        if epistemic_per_atom > p.energy_std_per_atom {
            return Verdict {
                accept: false,
                uncertainty: Some(uncertainty),
                reason: Some(format!(
                    "surrogate: epistemic energy std {epistemic_per_atom:.2e} > {:.2e} Ha/atom",
                    p.energy_std_per_atom
                )),
            };
        }
        Verdict { accept: true, uncertainty: Some(uncertainty), reason: None }
    }

    /// The teacher's step: evaluate, keep the sample, build the result on the right surface.
    #[allow(clippy::too_many_arguments)]
    fn teach(
        &mut self,
        scene: &Scene,
        prov: Provenance,
        pos: Vec<[f64; 3]>,
        z: Vec<u8>,
        model_energy: Option<f64>,
        uncertainty: Option<f64>,
        reason: String,
        keep_sample: bool,
        t0: Instant,
        should_abort: &mut dyn FnMut() -> bool,
    ) -> Result<ForceResult, ProviderError> {
        let r = self.teacher.evaluate(scene, should_abort)?;
        let mut notes = Vec::new();
        if keep_sample {
            let sample = Sample {
                pos: pos.clone(),
                z: z.clone(),
                energy: r.energy,
                forces: r.forces.clone(),
                tag: SampleTag::Fallback,
                weight: 1.0,
            };
            match self.push_sample(sample, &prov) {
                Ok(()) => self.pending_refit = true,
                Err(e) => notes.push(format!("surrogate: sample not kept ({e})")),
            }
        }
        let (energy, teacher_gap, refit_jump) = match model_energy {
            Some(m) => (m, Some(r.energy - m), self.pending_jump.take()),
            None => {
                self.pending_jump = None;
                (r.energy, None, None)
            }
        };
        let mut stats = r.stats;
        stats.source = Source::Fallback;
        stats.uncertainty = uncertainty;
        stats.teacher_gap = teacher_gap;
        stats.refit_jump = refit_jump;
        stats.solve_ms = t0.elapsed().as_millis() as u64;
        let gap = teacher_gap.map_or(String::new(), |g| format!(" (teacher − model {g:+.2e} Ha)"));
        notes.insert(0, format!("{reason} — teacher answered{gap}"));
        stats.notes.extend(notes);
        self.last = Some(LastEval { pos, z, energy, on_model: model_energy.is_some(), provenance: prov });
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
        if let Err(Gate { reason, keep_sample }) = self.gate(&z, &prov) {
            return self.teach(scene, prov, pos, z, None, None, reason, keep_sample, t0, should_abort);
        }
        // rules 5–8: consult the model
        let with_variance = {
            let k = self.policy.variance_cadence.max(1);
            let fresh = self.consults % k == 0 || self.last_verdict.is_none();
            self.consults += 1;
            fresh
        };
        let model = self.model.as_ref().expect("gate passed");
        let pred = model
            .predict(&pos, &z, with_variance)
            .map_err(|e| ProviderError::Invalid(format!("surrogate: {e}")))?;
        let verdict = self.judge(model, &pred, &pos, &z);
        if with_variance {
            self.last_verdict = Some(verdict.clone());
        }
        if !verdict.accept {
            let reason = verdict.reason.unwrap_or_else(|| "surrogate: policy".into());
            return self.teach(scene, prov, pos, z, Some(pred.energy), verdict.uncertainty, reason, true, t0, should_abort);
        }
        let mut stats = EvalStats::default();
        stats.source = Source::Surrogate;
        stats.uncertainty = verdict.uncertainty;
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

        let outside = h2(4.0);
        let r = s.evaluate(&outside, &mut never()).unwrap();
        println!("r = 4.0: {:?}, notes {:?}", r.stats.source, r.stats.notes);
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 41);
        assert!(s.pending_refit());
        let (e4, f4) = teacher.energy_and_forces(&[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]).unwrap();
        assert_eq!(r.forces, f4, "the teacher's forces");
        let gap = r.stats.teacher_gap.expect("on the model surface");
        assert!((r.energy + gap - e4).abs() < 1e-12, "energy is the model's: {} + {gap} vs {e4}", r.energy);
        assert_eq!(r.stats.energy_surface(), Surface::Model);
        assert!(r.stats.energy.total == e4, "the teacher's breakdown is kept");

        let rep = s.refit().unwrap();
        println!("after the refit with the r = 4.0 sample: rmse E/atom {:.1e}, F {:.1e}; s_E {:.1e} s_F {:.1e}; γ_E {:.2} γ_F {:.2}", rep.rmse_energy_per_atom, rep.rmse_force, rep.sigma_e, rep.sigma_f, rep.gamma_e, rep.gamma_f);
        assert_eq!(rep.n_samples, 41);
        assert!(!s.pending_refit());
        assert_eq!(s.epoch(), 2, "the surface moved twice (first fit, this refit)");
        let r = s.evaluate(&outside, &mut never()).unwrap();
        println!("r = 4.0 after refit: {:?}, |ΔE| {:.1e}, max|ΔF| {:.1e}, uncertainty {:?}, notes {:?}", r.stats.source, (r.energy - e4).abs(), max_force_diff(&r.forces, &f4), r.stats.uncertainty, r.stats.notes);
        assert_eq!(r.stats.source, Source::Surrogate);
        assert_eq!(calls(&h), 1);
        assert!(max_force_diff(&r.forces, &f4) < 1e-4);
        // The energy at r = 4.0 is NOT reproduced to 1e-5 (≈ 1e-2 off, with s_E ≈ 8e-4): the
        // Morse tail at r_cut = 6 is −3.4e-3 Ha, which an f_c-damped n_max = 8 basis cannot
        // carry once a point outside the window pins the energy difference, and the two-noise
        // evidence then settles on "forces exact, energies noisy" (γ_F ≈ 7 of 8). The policy's
        // energy criterion is epistemic-only by ruling, so this aleatoric-classified offset is
        // accepted; the fallback step's teacher_gap readout is where it shows. Task 7 (tiered
        // targets, n_max / body-3) owns the representability; the report records it.
        println!("energy at r = 4.0 after the refit: model − teacher {:+.2e} Ha (model energy_std {:.1e})", r.energy - e4, s.model().unwrap().predict(&[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &[1, 1], true).unwrap().energy_std.unwrap());
        assert!((r.energy - e4).abs() < 3e-2);
        assert!(r.stats.refit_jump.is_some(), "the first evaluate after a refit reports the jump");
        assert_eq!(s.evaluate(&outside, &mut never()).unwrap().stats.refit_jump, None, "once");
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

        // enough samples for n_samples ≥ q′ but not for 30 residual degrees of freedom
        let (mut s, _) = trained(1.2, 2.2, 12, FallbackPolicy::default());
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
        // the matching scene is still answered
        assert_eq!(s.evaluate(&h2(1.6), &mut never()).unwrap().stats.source, Source::Surrogate);
    }

    /// Rule 2 and the rebuild: a species the model does not know falls back unconditionally and
    /// is learned at the next refit — once its reference energy exists.
    #[test]
    fn new_species_falls_back_and_is_learned_at_the_next_refit() {
        let (mut s, h) = trained(1.2, 2.2, 60, FallbackPolicy::default());
        let h_he = dimer(1.6, [0.0, 1.0, 0.0], [0.1, -0.1, 0.0], 1, 2);
        let r = s.evaluate(&h_he, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback);
        assert_eq!(r.stats.energy_surface(), Surface::Teacher);
        assert_eq!(calls(&h), 1);
        assert_eq!(s.samples(), 61, "kept: the rebuild needs it");
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
        assert_eq!(rep.q_prime, 8 * 3, "n_max·S(S+1)/2 with S = 2");
        assert_eq!(rep.n_samples, 61);
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
        // trained on D = 0.17; the live teacher then has D′ = 1.25 D (energies and forces differ)
        let always_fallback = FallbackPolicy { force_std_abs: 0.0, force_std_rel: 0.0, ..FallbackPolicy::default() };
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
            // 20 scan samples from a slightly different well (D 5 % deeper): the fit then
            // compromises between them and the 51 live samples, so the model surface differs
            // visibly from the teacher's at r_50 and the jump is a real number, not rounding
            let mut deeper = MorseDimer::h2();
            deeper.d *= 1.05;
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..20 {
                let r = 1.2 + i as f64 / 19.0;
                let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
                s.add_sample(morse_sample(&deeper, &sc), &sc).unwrap();
            }
            let rep = s.refit().unwrap();
            println!("mid-run refit: {} samples, dof_E {:.1}, dof_F {:.1}, s_F {:.1e}", rep.n_samples, rep.residual_dof_e, rep.residual_dof_f, rep.sigma_f);
            assert!(s.trusted());
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
        // No step jumps. The per-step change is not only the Verlet fluctuation: on a fallback
        // step booked on the model surface the atoms move on the teacher's forces, so E_c
        // changes by (F_teacher − F_model)·Δr — the model's force error integrated along the
        // path, which is what one-surface accounting makes visible (≈ 5 % of 0.05 Ha/Bohr
        // over 4e-3 Bohr here, 1e-5 Ha). A trusted model keeps |ΔF| ≲ 2e-3 per component.
        assert!(fluct < 1e-4, "a jump, not the force-mismatch work: {fluct:.2e}");
        assert!((d.snapshot().fallback_fraction_100 - n_fb as f64 / 100.0).abs() < 1e-12);
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
