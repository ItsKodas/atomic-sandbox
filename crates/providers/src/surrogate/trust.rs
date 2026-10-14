//! The trust rule and the fit (the `surrogate` module doc, "Trust rule" and "One-surface energy
//! accounting": the refit and its surface bookkeeping): [`TrainingResiduals`], `trusted`,
//! `untrusted_reason`, `refit` and the store fit behind it, the "stuck" readout. Split out of
//! `surrogate_ks.rs` in M6 Task 1; the narrative stays in the parent module doc.

use super::*;

/// Trust rule: both residual degrees of freedom of the fit must reach this.
pub const MIN_RESIDUAL_DOF: f64 = 30.0;

/// A trusted model that has fallen back on this many consecutive consultations is reported as
/// "not answering here" with the rule that fires ([`SurrogateKs::stuck`]; module doc, "Stuck").
pub const STUCK_STEPS: usize = 50;

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

/// The environment classes of the class-wise trust rule (module doc, "Class-wise trust"), by
/// [`env_class`] index.
pub const ENV_CLASSES: [&str; 2] = ["pair", "many-body"];

/// The environment class of a geometry from its per-atom neighbour counts within r_cut
/// (`neighbour_counts`): the largest count over its atoms — 1 → "pair" (index 0), ≥ 2 →
/// "many-body" (index 1); 0 (every atom isolated) → `None`, the isolated class, which the class
/// rule exempts (module doc). Universal: no constant beyond the model's own r_cut, nothing per
/// element.
pub fn env_class(counts: &[usize]) -> Option<usize> {
    match counts.iter().copied().max().unwrap_or(0) {
        0 => None,
        1 => Some(0),
        _ => Some(1),
    }
}

/// The fitted state a recovery attempt may have to restore.
pub(super) struct FitState {
    pub(super) accumulator: Option<Accumulator>,
    pub(super) accumulated: usize,
    pub(super) model: Option<Model>,
    pub(super) report: Option<FitReport>,
    pub(super) envelope: Option<Envelope>,
    pub(super) null_space: Option<NullSpace>,
    pub(super) active_set: Option<ActiveSet>,
    pub(super) training_residuals: Option<TrainingResiduals>,
    pub(super) class_loo: Vec<ClassLoo>,
    pub(super) degree_cap: Option<usize>,
}

impl SurrogateKs {
    /// The current model's worst training residuals (`None` before the first refit).
    pub fn training_residuals(&self) -> Option<TrainingResiduals> {
        self.training_residuals
    }

    /// The teacher − model gap of the audit that made the model untrusted, until the next refit.
    pub fn audit_failed(&self) -> Option<f64> {
        self.audit_failed
    }

    /// The trust rule (`surrogate` module doc): a model exists and its report, its training residuals and
    /// the last audit pass.
    pub fn trusted(&self) -> bool {
        self.model.is_some() && self.untrusted_reason().is_none()
    }

    /// Why the model may not answer (the trust rule's first failing condition), `None` when it
    /// may (or when nothing was fitted: see [`trusted`](Self::trusted)).
    pub fn untrusted_reason(&self) -> Option<String> {
        let r = self.report.as_ref()?;
        // the count conditions: the app's `COUNT_REFUSALS` matches their wording ("n_samples ",
        // "residual_dof_") so the Model tab does not blame the box for them — keep the prefixes
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
        if self.noise_fails() {
            return Some(format!(
                "force noise σ_f {:.1e} > {:.1e} Ha/Bohr — the fit's own force labels scatter beyond the force tier",
                r.sigma_f, self.policy.force_std_abs
            ));
        }
        if let Some(gap) = self.audit_failed {
            return Some(match self.audit_force {
                Some((d, b)) => format!("audit found forces off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.1e} Ha); refit to recover"),
                None => format!("audit found teacher − model {gap:+.1e} Ha; refit to recover"),
            });
        }
        None
    }

    /// One pass over the samples the current model was fitted on (the accumulated prefix of
    /// the store — samples kept since the last refit are not training data yet; module doc,
    /// "Trust rule").
    pub(super) fn compute_training_residuals(&self) -> Option<TrainingResiduals> {
        let (model, store) = (self.model.as_ref()?, self.store.as_ref()?);
        let mut worst = TrainingResiduals { energy_per_atom: 0.0, force: 0.0, force_bound: f64::INFINITY };
        let mut worst_ratio = 0.0;
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            let Ok(p) = model.predict(&s.pos, &s.z, false) else { continue };
            let n = s.pos.len().max(1) as f64;
            worst.energy_per_atom = worst.energy_per_atom.max((p.energy - s.energy).abs() / n);
            for (fm, fs) in p.forces.iter().zip(&s.forces) {
                let (d, bound) = self.policy.force_residual(fm, fs);
                if d / bound > worst_ratio {
                    worst_ratio = d / bound;
                    worst.force = d;
                    worst.force_bound = bound;
                }
            }
        }
        Some(worst)
    }

    /// The class-wise trust numbers (module doc, "Class-wise trust"): for every fitted sample (the
    /// accumulated prefix, excluded samples skipped, as the training residuals) its environment
    /// class and exact leave-one-out residuals (`surrogate::loo`); per class the LOO per-atom
    /// force error ‖r_F,j‖/√3 as a ratio to the atom's bound `force_bound(|F_j|)` — RMS and max
    /// over the class's atoms — and the largest LOO per-atom energy error as a ratio to
    /// `energy_std_per_atom`. Trusted iff the class holds ≥ Q′ samples and all three ratios are
    /// ≤ 1. A sample whose LOO is undefined (I − H_ss singular: it alone determines a direction)
    /// counts as an infinite error. Empty without a model.
    pub(super) fn compute_class_loo(&self) -> Vec<ClassLoo> {
        let (Some(model), Some(store), Some(report)) = (self.model.as_ref(), self.store.as_ref(), self.report.as_ref()) else {
            return Vec::new();
        };
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        // per class: samples, Σ ratio², atoms, max force ratio, max energy ratio
        let mut acc = [(0usize, 0.0f64, 0usize, 0.0f64, 0.0f64); 2];
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            let Some(c) = env_class(&neighbour_counts(&s.pos, rc2)) else { continue };
            let a = &mut acc[c];
            a.0 += 1;
            let Some(l) = surrogate::loo::leave_one_out(model, s) else {
                a.3 = f64::INFINITY;
                a.4 = f64::INFINITY;
                continue;
            };
            let n = s.pos.len().max(1) as f64;
            let e = if model.energy_per_atom { l.energy.abs() } else { l.energy.abs() / n };
            a.4 = a.4.max(e / self.policy.energy_std_per_atom);
            for (r, f) in l.forces.iter().zip(&s.forces) {
                let (d, b) = self.policy.force_residual(&[f[0] - r[0], f[1] - r[1], f[2] - r[2]], f);
                let ratio = d / b;
                a.1 += ratio * ratio;
                a.2 += 1;
                a.3 = a.3.max(ratio);
            }
        }
        acc.iter()
            .zip(ENV_CLASSES)
            .map(|(&(samples, ss, atoms, force_max, energy_max), class)| {
                let force_rms = if atoms > 0 { (ss / atoms as f64).sqrt() } else { 0.0 };
                let force_rms = if force_max.is_finite() { force_rms } else { f64::INFINITY };
                let trusted = samples >= report.q_prime && force_rms <= 1.0 && force_max <= 1.0 && energy_max <= 1.0;
                ClassLoo { class, samples, force_rms, force_max, energy_max, trusted }
            })
            .collect()
    }

    /// The per-class leave-one-out numbers of the current fit (module doc, "Class-wise trust").
    pub fn class_loo(&self) -> &[ClassLoo] {
        &self.class_loo
    }

    /// The class rule (module doc, "Class-wise trust"; rule 4b) for a query with per-atom
    /// neighbour counts `counts`: `None` when its class is trusted (or isolated: exempt), else
    /// (the reason, whether the class is short of samples — such fallbacks are never thinned:
    /// they are what earns the class its trust).
    pub(super) fn class_untrusted(&self, counts: &[usize]) -> Option<(String, bool)> {
        let c = env_class(counts)?;
        let q = self.report.as_ref()?.q_prime;
        let Some(l) = self.class_loo.get(c) else {
            return Some((format!("surrogate: this kind of environment isn't learned well enough yet ({}: no leave-one-out numbers)", ENV_CLASSES[c]), true));
        };
        if l.trusted {
            return None;
        }
        Some(if l.samples < q {
            (format!("surrogate: this kind of environment isn't learned well enough yet ({}: {} samples < Q′ {q})", l.class, l.samples), true)
        } else if l.force_max > 1.0 || l.force_rms > 1.0 {
            let energy = if l.energy_max > 1.0 { format!(", energy {:.1}× the tier", l.energy_max) } else { String::new() };
            (format!(
                "surrogate: this kind of environment isn't learned well enough yet ({}: LOO force error {:.1}× the bound, RMS {:.2}×{energy}, {} samples)",
                l.class, l.force_max, l.force_rms, l.samples
            ), false)
        } else {
            (format!(
                "surrogate: this kind of environment isn't learned well enough yet ({}: LOO energy error {:.1}× the tier, {} samples)",
                l.class, l.energy_max, l.samples
            ), false)
        })
    }

    /// Rebuild the model from the samples (`surrogate` module doc): the accumulator takes the samples added
    /// since the last refit, or is rebuilt from the store after a species or reference change;
    /// then one hyperparameter search and solve, and one pass over the store for the training
    /// residuals. Clears a failed audit. Computes the refit jump for the next evaluate and bumps
    /// the surface epoch when the surface moved. On an error the previous model, report and
    /// envelope stay in place. With a probe attached ([`with_probe`](SurrogateKs::with_probe)) the
    /// refit's phase is reported on it for its duration (spec A1), cleared on every exit.
    pub fn refit(&mut self) -> Result<FitReport, FitError> {
        /// Ends the probe's refit phase when dropped (an error or a panic clears it too).
        struct RefitPhase(Arc<sim::ActivityProbe>);
        impl Drop for RefitPhase {
            fn drop(&mut self) {
                self.0.end_refit();
            }
        }
        let _phase = self.probe.clone().map(|p| {
            p.begin_refit(self.samples());
            RefitPhase(p)
        });
        self.refit_inner()
    }

    /// [`refit`](Self::refit)'s body.
    fn refit_inner(&mut self) -> Result<FitReport, FitError> {
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        let mut report = self.fit_store()?;
        self.recovery = None;
        // fence votes (`surrogate` module doc, "Fence voting"): a voted sample the new model cannot
        // reproduce is quarantined itself and its region takes no further vote
        let rejected = self.judge_votes();
        if rejected > 0 {
            self.species_dirty = true;
            report = self.fit_store()?;
            self.recovery = Some(format!(
                "{rejected} voted samples quarantined: the model cannot learn the fresh teacher answers next to a quarantined geometry — that region stays fenced"
            ));
        }
        // release what the new model reproduces (`surrogate` module doc, "Release")
        let released = self.release_reproduced();
        if released > 0 {
            self.species_dirty = true;
            report = self.fit_store()?;
            self.recovery = Some(format!("{released} quarantined samples released: the model now reproduces them"));
        }
        if self.recoverable_fail() {
            report = self.recover_trust(report);
        }
        self.surface_moved(was_trusted, was_model).map_err(FitError::Describe)?;
        Ok(report)
    }

    /// The training residuals exceed the policy's thresholds (the trust rule's S8 part).
    pub(super) fn residuals_fail(&self) -> bool {
        self.training_residuals
            .is_some_and(|t| t.energy_per_atom > self.policy.energy_std_per_atom || t.force > t.force_bound)
    }

    /// The fit's force-label noise σ_f exceeds the absolute force tier `force_std_abs` (the trust
    /// rule's noise condition, M6 Task 11; module doc, "Trust rule").
    pub(super) fn noise_fails(&self) -> bool {
        self.report.as_ref().is_some_and(|r| r.sigma_f > self.policy.force_std_abs)
    }

    /// A trust failure the recovery (degree cap, quarantine) can act on: the training residuals
    /// or the force noise.
    pub(super) fn recoverable_fail(&self) -> bool {
        self.residuals_fail() || self.noise_fails()
    }

    pub(super) fn save_state(&self) -> FitState {
        FitState {
            accumulator: self.accumulator.clone(),
            accumulated: self.accumulated,
            model: self.model.clone(),
            report: self.report.clone(),
            envelope: self.envelope.clone(),
            null_space: self.null_space.clone(),
            active_set: self.active_set.clone(),
            training_residuals: self.training_residuals,
            class_loo: self.class_loo.clone(),
            degree_cap: self.degree_cap,
        }
    }

    pub(super) fn restore_state(&mut self, s: FitState) {
        self.accumulator = s.accumulator;
        self.accumulated = s.accumulated;
        self.model = s.model;
        self.report = s.report;
        self.envelope = s.envelope;
        self.null_space = s.null_space;
        self.active_set = s.active_set;
        self.training_residuals = s.training_residuals;
        self.class_loo = s.class_loo;
        self.degree_cap = s.degree_cap;
    }

    /// The fit itself (`surrogate` module doc): incremental, or a rebuild from the store skipping the
    /// quarantined samples; no surface bookkeeping (`refit` does it once).
    pub(super) fn fit_store(&mut self) -> Result<FitReport, FitError> {
        let Some(store) = self.store.as_ref() else { return Err(FitError::NoSamples) };
        if store.is_empty() {
            return Err(FitError::NoSamples);
        }
        self.exclude_box_samples();
        let store = self.store.as_ref().expect("checked");
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
            if self.quarantine.get(i).is_some_and(|&q| q != Q_NONE) {
                accumulated += 1; // held, not fitted (`surrogate` module doc, "Trust recovery")
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
        self.null_space = Some(NullSpace::of(&acc, &model.scale));
        self.accumulator = Some(acc);
        self.accumulated = accumulated;
        self.species_dirty = false;
        self.refs_dirty = false;
        self.model = Some(model);
        self.report = Some(report.clone());
        self.audit_failed = None;
        self.audit_force = None;
        self.training_residuals = self.compute_training_residuals();
        self.class_loo = self.compute_class_loo();
        self.rebuild_active_set();
        self.pending_refit = false;
        self.consults = 0;
        self.last_variance = None;
        // the votes disagreed with the previous model
        self.fence_votes.clear();
        Ok(report)
    }

    /// Surface bookkeeping after `self.model` was replaced (a refit or an installed sidecar;
    /// module doc): the jump for the next evaluate, and the epoch. `was_trusted` / `was_model`
    /// describe the state before the replacement. The next answer is the model's when it is
    /// trusted — or, in draft mode, whenever rules 1–3 pass (the `match`): an untrusted refit
    /// under draft mode moves the surface the atoms are on, and its jump must be booked too.
    pub(super) fn surface_moved(&mut self, was_trusted: bool, was_model: bool) -> Result<(), surrogate::DescribeError> {
        let model = self.model.as_ref().expect("a model was just installed");
        let next_is_model = (self.trusted() || self.policy.never_fallback)
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

    /// The trust rule fails for want of data (n_samples < Q′ or too few residual degrees of
    /// freedom): every teacher answer is then worth keeping (no thinning).
    pub(super) fn trust_needs_samples(&self) -> bool {
        self.report.as_ref().is_none_or(|r| r.n_samples < r.q_prime || r.residual_dof_e < MIN_RESIDUAL_DOF || r.residual_dof_f < MIN_RESIDUAL_DOF)
    }

    /// Reference energies in `basis.species` order.
    pub(super) fn reference_vector(&self, basis: &Basis) -> Result<Vec<f64>, FitError> {
        basis
            .species
            .iter()
            .map(|&z| match self.refs.get(&z) {
                Some(&e) if e.is_finite() => Ok(e),
                _ => Err(FitError::MissingReference(z)),
            })
            .collect()
    }

    /// A trusted model that has fallen back on the last ≥ [`STUCK_STEPS`] consultations: why (the
    /// rule that fires, the latest teacher − model gap) — trusted elsewhere, not answering here
    /// (`surrogate` module doc, "Stuck").
    ///
    /// When every one of the last [`STUCK_STEPS`] fallbacks was box-affected (M6 Task 11) it says
    /// that instead: those answers are never trained on (module doc, "Box-affected answers"), so
    /// nothing is being learned here and only a bigger box (or walls that keep the atoms off the
    /// faces) can change that.
    pub fn stuck(&self) -> Option<String> {
        (self.streak >= STUCK_STEPS && self.trusted()).then(|| {
            let reason = self.streak_reason.as_deref().unwrap_or("surrogate: fallback");
            let reason = reason.strip_prefix("surrogate: ").unwrap_or(reason);
            if self.streak_box >= STUCK_STEPS {
                let rule = box_affected_face_density(self.store.as_ref().map_or(0.25, |s| s.provenance.h));
                return format!(
                    "not learning here: every answer here touches the box wall — enlarge the grid (or turn container walls on). The last {} steps it was asked fell back ({reason}); their teacher answers are box-affected (face density up to {:.1e} > {rule:.1e} e/Bohr³) and never trained on",
                    self.streak, self.streak_box_face
                );
            }
            match self.streak_gap {
                Some(g) => format!("not answering here: the last {} steps it was asked fell back — {reason} (teacher − model {g:+.1e} Ha)", self.streak),
                None => format!("not answering here: the last {} steps it was asked fell back — {reason}", self.streak),
            }
        })
    }
}
