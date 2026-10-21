//! The trust rule and the fit (the `surrogate` module doc, "Trust rule" and "One-surface energy
//! accounting": the refit and its surface bookkeeping): [`TrainingResiduals`], `trusted`,
//! `untrusted_reason`, `trust_checks`, `refit` and the store fit behind it, the class rule's
//! verdict (`class_refusal`), the "stuck" readout and its structured form. Split out of
//! `surrogate_ks.rs` in M6 Task 1; the narrative stays in the parent module doc.
//!
//! The trust rule, in the order it is tested (activity spec A4; nothing here is new: Activity
//! Task 3 only gave each condition one predicate, plan review M2). With the fit report, the
//! training residuals and the policy's tiers, the model may answer iff no condition fails:
//!
//! | key | fails when | value / limit |
//! |---|---|---|
//! | `Samples` | n_samples < Q′ | n_samples / Q′ |
//! | `ResidualDofE` | not ν_E ≥ 30 (`MIN_RESIDUAL_DOF`) | ν_E / 30 |
//! | `ResidualDofF` | not ν_F ≥ 30 | ν_F / 30 |
//! | `EnergyResidual` | max over samples of \|E_model − E_s\|/N_s > `energy_std_per_atom` | that max / the tier (Ha/atom) |
//! | `ForceResidual` | ‖ΔF‖/√3 > its bound at the atom with the largest ratio | that residual / its bound (Ha/Bohr) |
//! | `ForceNoise` | σ_f > `force_std_abs` | σ_f / the tier (Ha/Bohr) |
//! | `Audit` | an audit failed since the refit, on the energy (\|gap\|/N > the tier) | \|gap\|/N / the tier (Ha/atom) |
//! | `AuditForces` | an audit failed since the refit, on the forces (`audit_force` set) | the worst atom's residual / its bound (Ha/Bohr) |
//!
//! The two residual rows exist only when training residuals were computed. The audit row is met
//! at value 0 when no audit failed since the refit, which says nothing about whether one ran
//! (plan review M1). Every comparison is its predicate's own, so a NaN or a value at the limit
//! lands on the same side for the checklist, the refusal and `trusted`. A value at its limit
//! passes; a NaN fails every condition it is read by (activity Task 3b: "not within the limit"),
//! and the accumulators behind the residual rows and the class rule carry a NaN prediction
//! through (`max_nan`) instead of dropping it.

use super::*;
use sim::activity::{StuckInfo, TeachRule, TrustCheck, TrustKey};
use sim::ClassKey;

/// Trust rule: both residual degrees of freedom of the fit must reach this.
pub const MIN_RESIDUAL_DOF: f64 = 30.0;

// The trust rule's conditions, one predicate each (plan review M2). `trust_checks` builds the
// checklist from them as `met = !fails`; `untrusted_check` is that list's first unmet entry;
// `residuals_fail`, `noise_fails` and `trust_needs_samples` call them. No condition is written
// twice, so the checklist and the refusal agree at equality and on NaN by construction. Each is
// written as "not within the limit" (`!(x <= limit)`, `!(x >= limit)`), so a NaN fails it: a
// number the rule cannot compare is never a pass (activity Task 3b, the physics review's ruling
// (b)); at the limit itself a condition is met, as before.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
mod predicates {
    use super::*;
    pub(super) fn samples_short(r: &FitReport) -> bool {
        r.n_samples < r.q_prime
    }
    pub(super) fn dof_short(dof: f64) -> bool {
        !(dof >= MIN_RESIDUAL_DOF)
    }
    pub(super) fn energy_residual_fails(t: &TrainingResiduals, p: &FallbackPolicy) -> bool {
        !(t.energy_per_atom <= p.energy_std_per_atom)
    }
    pub(super) fn force_residual_fails(t: &TrainingResiduals) -> bool {
        !(t.force <= t.force_bound)
    }
    pub(super) fn force_noise_fails(r: &FitReport, p: &FallbackPolicy) -> bool {
        !(r.sigma_f <= p.force_std_abs)
    }
}
use predicates::*;

/// `f64::max` that keeps a NaN (`f64::max` returns the other operand): the accumulators below
/// must carry a NaN prediction to the trust rule, never report it as 0 (activity Task 3b).
pub(super) fn max_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}
fn audit_fails(audit_failed: Option<f64>) -> bool {
    audit_failed.is_some()
}

/// The uncalibrated class rule for one class (module doc, "Class-wise trust"; since M8a Task 2
/// the rule `calibrate::class_verdict` applies while a class has fewer than three DFT
/// comparisons): the first failing condition (too few samples, the LOO force error, the LOO
/// energy error) as (key, value, limit), `None` when the class passes (plan review M3). The force and energy tests are written as `!(x <= 1)`, so a NaN ratio
/// fails, exactly as the old `trusted`'s `<=` made it fail. The force value is the larger ratio,
/// or NaN when either is NaN (`f64::max` would drop the NaN and show a failing check with a
/// value inside its limit).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: NaN fails
pub(super) fn class_refusal(samples: usize, q_prime: usize, force_rms: f64, force_max: f64, energy_max: f64) -> Option<(ClassKey, f64, f64)> {
    if samples < q_prime {
        return Some((ClassKey::Samples, samples as f64, q_prime as f64));
    }
    if !(force_max <= 1.0 && force_rms <= 1.0) {
        let value = if force_max.is_nan() || force_rms.is_nan() { f64::NAN } else { force_max.max(force_rms) };
        return Some((ClassKey::Force, value, 1.0));
    }
    if !(energy_max <= 1.0) {
        return Some((ClassKey::Energy, energy_max, 1.0));
    }
    None
}

/// A failed audit re-judged and still failed at this many refits becomes an audit fence (round 1,
/// SHOULD-3): it no longer untrusts the model everywhere. Chosen: three refits are three batches of
/// samples the worker kept near the trajectory; a failure that survives them is a local problem.
pub const AUDIT_REJUDGE_LIMIT: u32 = 3;

/// After a cleared failed audit the next this many answered steps are audited (round 1, SHOULD-1):
/// the cost is that many teacher solves, never a wrong answer.
pub const AUDIT_BURST: usize = 3;

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
/// One atom's environment class from its neighbour count within r_cut (since 2026-10-18 the class
/// rule classes atoms, not geometries: module doc, "Class-wise trust"): 1 → pair (0), ≥ 2 →
/// many-body (1), 0 → `None` (isolated, exempt).
pub fn atom_class(count: usize) -> Option<usize> {
    match count {
        0 => None,
        1 => Some(0),
        _ => Some(1),
    }
}

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
    pub(super) class_memory: Vec<(bool, Option<u64>)>,
    pub(super) class_worst: Vec<Option<(usize, f64)>>,
    pub(super) class_blockers: Vec<Vec<(usize, f64)>>,
    pub(super) degree_cap: Option<usize>,
    pub(super) fit_members: (usize, Vec<u8>),
}

impl SurrogateKs {
    /// The current model's worst training residuals (`None` before the first refit).
    pub fn training_residuals(&self) -> Option<TrainingResiduals> {
        self.training_residuals
    }

    /// The teacher − model gap of the audit that made the model untrusted, until a refit passes it
    /// (module doc, rule 8, "A failed audit stays failed").
    pub fn audit_failed(&self) -> Option<f64> {
        self.audit_failed
    }

    /// The trust rule (`surrogate` module doc): a model exists and its report, its training residuals and
    /// the last audit pass. The same verdict as `untrusted_reason().is_none()`, without formatting a
    /// string or allocating (an empty list, no report yet, is all met, as `untrusted_reason`'s
    /// `None` was).
    pub fn trusted(&self) -> bool {
        self.model.is_some() && self.checks().all(|c| c.met)
    }

    /// Why the model may not answer (the trust rule's first failing condition), `None` when it
    /// may (or when nothing was fitted: see [`trusted`](Self::trusted)).
    pub fn untrusted_reason(&self) -> Option<String> {
        self.untrusted_check().map(|(_, s)| s)
    }

    /// The trust rule's conditions with their numbers, in the order the rule tests them
    /// (activity spec A4); empty before the first fit. Each entry is `met = !(its predicate)`;
    /// the first unmet one is the refusal ([`untrusted_check`](Self::untrusted_check)), by
    /// construction. The audit entry is per atom: |gap|/N against `energy_std_per_atom`, or, when
    /// the forces failed the audit, the worst atom's residual against its bound (`AuditForces`);
    /// with no failed audit since the refit it is met with value 0, whether or not an audit ever
    /// ran, so the UI words that line from `last_audit` and `last_audit_this_fit` (none yet /
    /// none since the refit / passed / box-affected), never as "passed" (plan review M1, Task 3
    /// review I1).
    pub fn trust_checks(&self) -> Vec<TrustCheck> {
        self.checks().collect()
    }

    /// [`trust_checks`](Self::trust_checks) without the allocation: the one sequence that
    /// `trusted`, `untrusted_check` and the checklist read.
    fn checks(&self) -> impl Iterator<Item = TrustCheck> + use<> {
        let p = &self.policy;
        let check = |key, fails: bool, value: f64, limit: f64| TrustCheck { key, met: !fails, value, limit };
        let r = self.report.as_ref();
        let t = r.and(self.training_residuals.as_ref());
        let fails = audit_fails(self.audit_failed);
        let audit = match (self.audit_failed, self.audit_force) {
            (Some(_), Some((d, b))) => check(TrustKey::AuditForces, fails, d, b),
            (Some(gap), None) => check(TrustKey::Audit, fails, gap.abs() / self.audit_atoms.max(1) as f64, p.energy_std_per_atom),
            (None, _) => check(TrustKey::Audit, fails, 0.0, p.energy_std_per_atom),
        };
        [
            r.map(|r| check(TrustKey::Samples, samples_short(r), r.n_samples as f64, r.q_prime as f64)),
            r.map(|r| check(TrustKey::ResidualDofE, dof_short(r.residual_dof_e), r.residual_dof_e, MIN_RESIDUAL_DOF)),
            r.map(|r| check(TrustKey::ResidualDofF, dof_short(r.residual_dof_f), r.residual_dof_f, MIN_RESIDUAL_DOF)),
            t.map(|t| check(TrustKey::EnergyResidual, energy_residual_fails(t, p), t.energy_per_atom, p.energy_std_per_atom)),
            t.map(|t| check(TrustKey::ForceResidual, force_residual_fails(t), t.force, t.force_bound)),
            r.map(|r| check(TrustKey::ForceNoise, force_noise_fails(r, p), r.sigma_f, p.force_std_abs)),
            r.map(|_| audit),
        ]
        .into_iter()
        .flatten()
    }

    /// [`untrusted_reason`](Self::untrusted_reason) with the failing condition's key (activity
    /// spec A3: the rule-4 code): the checklist's first unmet entry, worded with the refusal
    /// strings of before the Activity tab (byte-identical; pinned by
    /// `the_refusal_strings_are_unchanged` and the pre-Task-3 oracle sweep in the tests).
    pub fn untrusted_check(&self) -> Option<(TrustKey, String)> {
        let r = self.report.as_ref()?;
        let c = self.checks().find(|c| !c.met)?;
        let gap = self.audit_failed.unwrap_or(0.0);
        // the count conditions: the app's `COUNT_REFUSALS` matches their wording ("n_samples ",
        // "residual_dof_") so the Model tab does not blame the box for them: keep the prefixes
        let text = match c.key {
            TrustKey::Samples => format!("n_samples {} < q_prime {}", r.n_samples, r.q_prime),
            TrustKey::ResidualDofE => format!("residual_dof_e {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_e),
            TrustKey::ResidualDofF => format!("residual_dof_f {:.1} < {MIN_RESIDUAL_DOF:.0}", r.residual_dof_f),
            TrustKey::EnergyResidual => format!("max training energy residual {:.1e} > {:.1e} Ha/atom", c.value, c.limit),
            TrustKey::ForceResidual => format!("max training force residual {:.1e} > {:.1e} Ha/Bohr", c.value, c.limit),
            TrustKey::ForceNoise => format!(
                "force noise σ_f {:.1e} > {:.1e} Ha/Bohr — the fit's own force labels scatter beyond the force tier",
                c.value, c.limit
            ),
            // a teach-around check says so (review SHOULD-T5); the audit strings stay byte-identical
            TrustKey::AuditForces if self.audit_origin == AuditOrigin::TeachCheck => format!(
                "a teaching check failed (forces {:.2}× their bound at a taught geometry: {:.1e} > {:.1e} Ha/Bohr); untrusted until a refit predicts that geometry",
                c.value / c.limit,
                c.value,
                c.limit
            ),
            TrustKey::Audit if self.audit_origin == AuditOrigin::TeachCheck => {
                format!("a teaching check failed (teacher − model {gap:+.1e} Ha at a taught geometry); untrusted until a refit predicts that geometry")
            }
            TrustKey::AuditForces if self.audit_origin == AuditOrigin::ShellCheck => format!(
                "a shell check failed: a pair still interacts beyond 5/6 of the cutoff (forces {:.2}× their bound: {:.1e} > {:.1e} Ha/Bohr) — the cutoff truncates it; untrusted until it becomes a local fence",
                c.value / c.limit,
                c.value,
                c.limit
            ),
            TrustKey::Audit if self.audit_origin == AuditOrigin::ShellCheck => {
                format!("a shell check failed: a pair still interacts beyond 5/6 of the cutoff (teacher − model {gap:+.1e} Ha) — the cutoff truncates it; untrusted until it becomes a local fence")
            }
            TrustKey::AuditForces => {
                format!("audit found forces off by {:.1e} > {:.1e} Ha/Bohr (teacher − model {gap:+.1e} Ha); untrusted until a refit predicts that geometry without the audited sample", c.value, c.limit)
            }
            TrustKey::Audit => format!("audit found teacher − model {gap:+.1e} Ha; untrusted until a refit predicts that geometry without the audited sample"),
        };
        Some((c.key, text))
    }

    /// One pass over the samples the current model was fitted on (the accumulated prefix of
    /// the store — samples kept since the last refit are not training data yet; module doc,
    /// "Trust rule").
    pub(super) fn compute_training_residuals(&self) -> Option<TrainingResiduals> {
        let (model, store) = (self.model.as_ref()?, self.store.as_ref()?);
        let mut worst = TrainingResiduals { energy_per_atom: 0.0, force: 0.0, force_bound: f64::INFINITY };
        let mut worst_ratio = 0.0f64;
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            let Ok(p) = model.predict(&s.pos, &s.z, false) else { continue };
            let n = s.pos.len().max(1) as f64;
            worst.energy_per_atom = max_nan(worst.energy_per_atom, (p.energy - s.energy).abs() / n);
            for (fm, fs) in p.forces.iter().zip(&s.forces) {
                let (d, bound) = self.policy.force_residual(fm, fs);
                // a NaN ratio is the worst and stays so (activity Task 3b)
                let ratio = d / bound;
                if !worst_ratio.is_nan() && (ratio.is_nan() || ratio > worst_ratio) {
                    worst_ratio = ratio;
                    worst.force = d;
                    worst.force_bound = bound;
                }
            }
        }
        Some(worst)
    }

    /// The class rule's numbers and the readiness readout's worst sample per class (module doc,
    /// "Class-wise trust", "Readiness"): one pass sets `class_loo` and `class_worst`.
    pub(super) fn refresh_class_loo(&mut self) {
        let (loo, worst, blockers) = self.compute_class_loo_full();
        self.class_loo = loo;
        // the class verdicts carry rule 6/7's variance inflation (M8a Task 2 fix round 3): a cached
        // variance verdict judged with the old factors is not reused
        self.last_variance = None;
        self.class_worst = worst;
        self.class_blockers = blockers;
        // a class that is trusted again restarts its teach-around round count (SHOULD-T6)
        let trusted: Vec<usize> = self.class_loo.iter().enumerate().filter(|(_, c)| c.trusted).map(|(k, _)| k).collect();
        for c in trusted {
            self.teach_rounds.remove(&c);
        }
    }

    /// The class-wise trust numbers (module doc, "Class-wise trust"): for every fitted sample (the
    /// accumulated prefix, excluded samples skipped, as the training residuals) its exact
    /// leave-one-out residuals (`surrogate::loo`) and, **per atom**, the atom's environment class
    /// ([`atom_class`]: one neighbour within r_cut → pair, two or more → many-body, none →
    /// exempt). Per class: the samples with at least one atom of that class; the LOO per-atom
    /// force error ‖r_F,j‖/√3 against each atom's bound `force_bound(|F_j|)` over the class's
    /// atoms (RMS and max), and the LOO per-atom energy error against `energy_std_per_atom` over
    /// the class's samples (RMS and max; the energy is not per atom: it counts for every class the
    /// sample holds — conservative). The verdict is the **calibrated, hysteretic** one of
    /// [`super::calibrate`] (M8a Task 2): max(the LOO force RMS, the DFT-measured force error over
    /// the class's evidence window) and the DFT-measured energy error against `ENTER` / `LEAVE`,
    /// with the worst-atom guard on entry, and the standing verdict (`self.class_memory`) as the
    /// hysteresis memory; with fewer than `MIN_EVIDENCE` points of evidence it is the rule of
    /// before — ≥ Q′ samples and all three ratios ≤ 1 (`class_refusal`) — plus the worst-atom
    /// guard both ways. A sample whose LOO is undefined (I − H_ss singular: it alone determines
    /// a direction) counts as infinitely off. Per class also the sample of the largest ratio (force
    /// or energy) — the readiness readout's worst sample, deciding nothing. Held-out draft guesses
    /// (`Q_DRAFT`) are excluded like every other excluded sample (`surrogate` module doc, "Draft
    /// mode"). Empty without a model.
    #[allow(clippy::type_complexity)]
    pub(super) fn compute_class_loo_full(&self) -> (Vec<ClassLoo>, Vec<Option<(usize, f64)>>, Vec<Vec<(usize, f64)>>) {
        let (Some(model), Some(store), Some(report)) = (self.model.as_ref(), self.store.as_ref(), self.report.as_ref()) else {
            return (Vec::new(), Vec::new(), Vec::new());
        };
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        // per class: samples, Σ force ratio², atoms, max force ratio, max energy ratio, Σ energy ratio²
        let mut acc = [(0usize, 0.0f64, 0usize, 0.0f64, 0.0f64, 0.0f64); 2];
        let mut worst: Vec<Option<(usize, f64)>> = vec![None; 2];
        // per class: the samples beyond 1× (the blockers a teach-around pass may target)
        let mut blockers: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 2];
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            let classes: Vec<Option<usize>> = neighbour_counts(&s.pos, rc2).into_iter().map(atom_class).collect();
            let l = surrogate::loo::leave_one_out(model, s);
            let n = s.pos.len().max(1) as f64;
            let e_ratio = l.as_ref().map_or(f64::INFINITY, |l| (if model.energy_per_atom { l.energy.abs() } else { l.energy.abs() / n }) / self.policy.energy_std_per_atom);
            for (c, a) in acc.iter_mut().enumerate() {
                if !classes.contains(&Some(c)) {
                    continue;
                }
                a.0 += 1;
                a.4 = max_nan(a.4, e_ratio);
                a.5 += e_ratio * e_ratio;
                let mut f_worst = 0.0f64;
                match l.as_ref() {
                    None => {
                        a.3 = f64::INFINITY;
                        f_worst = f64::INFINITY;
                    }
                    Some(l) => {
                        for ((r, f), cj) in l.forces.iter().zip(&s.forces).zip(&classes) {
                            if *cj != Some(c) {
                                continue;
                            }
                            let (d, b) = self.policy.force_residual(&[f[0] - r[0], f[1] - r[1], f[2] - r[2]], f);
                            let ratio = d / b;
                            a.1 += ratio * ratio;
                            a.2 += 1;
                            a.3 = max_nan(a.3, ratio);
                            f_worst = max_nan(f_worst, ratio);
                        }
                    }
                }
                let r = max_nan(f_worst, e_ratio);
                if worst[c].is_none_or(|(_, w)| !(r <= w)) {
                    worst[c] = Some((i, r));
                }
                if !(r <= 1.0) {
                    blockers[c].push((i, r));
                }
            }
        }
        // the audit calibration per class (M8a Task 2) — the verdict's evidence
        let cal = super::calibrate::calibrate(&self.audit_history, &self.policy);
        let loo = acc
            .iter()
            .zip(ENV_CLASSES)
            .enumerate()
            .map(|(c, (&(samples, ss, atoms, force_max, energy_max, se), class))| {
                let force_rms = if atoms > 0 { (ss / atoms as f64).sqrt() } else { 0.0 };
                // an undefined LOO (∞) makes the RMS ∞; a NaN stays NaN and fails (Task 3b)
                let force_rms = if force_max == f64::INFINITY { f64::INFINITY } else { force_rms };
                let energy_rms = if samples > 0 { (se / samples as f64).sqrt() } else { 0.0 };
                let energy_rms = if energy_max == f64::INFINITY { f64::INFINITY } else { energy_rms };
                // the hysteresis memory is the verdict that stood when this refit, install or
                // policy change began (`class_memory`; reset by a forget or an install)
                let (was_trusted, since) = self.class_memory.get(c).copied().unwrap_or((false, None));
                let st = super::calibrate::ClassStats { samples, force_rms, force_max, energy_rms, energy_max };
                let refusal = match cal.class_verdict(c, &st, report.q_prime, was_trusted) {
                    super::calibrate::ClassVerdict::Trusted => None,
                    super::calibrate::ClassVerdict::Untrusted { key, value, limit, .. } => Some((key, value, limit)),
                };
                let trusted = refusal.is_none();
                let trusted_since = if trusted { since.or(Some(self.model_gen)) } else { None };
                let k = cal.per_class[c];
                let some = |x: f64| (k.points > 0).then_some(x);
                ClassLoo {
                    class,
                    samples,
                    force_rms,
                    force_max,
                    energy_rms,
                    energy_max,
                    trusted,
                    refusal,
                    calibration: k.factor,
                    measured: some(k.measured),
                    measured_energy: some(k.measured_energy),
                    worst_atom: some(k.worst_atom),
                    evidence: k.evidence,
                    trusted_since,
                    inflation: k.inflation,
                }
            })
            .collect();
        for b in &mut blockers {
            b.sort_by(|x, y| y.1.total_cmp(&x.1));
        }
        (loo, worst, blockers)
    }

    /// The per-class leave-one-out numbers of the current fit (module doc, "Class-wise trust").
    pub fn class_loo(&self) -> &[ClassLoo] {
        &self.class_loo
    }

    /// Makes the standing class verdicts the calibrated verdict's hysteresis memory
    /// (`class_memory`; M8a Task 2): called once a refit, an install or a policy change is done,
    /// never by the intermediate fits inside one of them.
    pub(super) fn commit_class_memory(&mut self) {
        self.class_memory = self.class_loo.iter().map(|l| (l.trusted, l.trusted_since)).collect();
    }

    /// Per atom of every fitted sample (the accumulated prefix, excluded samples skipped — the same
    /// members as `class_loo`): its class code in the history's numbering (0 isolated, 1 pair,
    /// 2 many-body) and the leave-one-out force error ‖F_LOO − F_sample‖ (Ha/Bohr; ∞ when the
    /// sample's LOO is undefined). The in-domain error distribution a measurement compares audit
    /// errors with (M8a Task 1's trust baseline). A readout; it decides nothing. Empty without a
    /// model.
    pub fn loo_force_errors(&self) -> Vec<(u8, f64)> {
        let (Some(model), Some(store)) = (self.model.as_ref(), self.store.as_ref()) else {
            return Vec::new();
        };
        let rc2 = model.basis.r_cut * model.basis.r_cut;
        let mut out = Vec::new();
        for (i, s) in store.samples[..self.accumulated.min(store.len())].iter().enumerate() {
            if self.excluded(i) {
                continue;
            }
            let classes = neighbour_counts(&s.pos, rc2).into_iter().map(|c| atom_class(c).map_or(0, |k| k as u8 + 1));
            match surrogate::loo::leave_one_out(model, s) {
                None => out.extend(classes.map(|c| (c, f64::INFINITY))),
                Some(l) => out.extend(classes.zip(&l.forces).map(|(c, r)| (c, (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()))),
            }
        }
        out
    }

    /// The class rule (module doc, "Class-wise trust"; rule 4b) for a query with per-atom
    /// neighbour counts `counts`: `None` when every class among its atoms ([`atom_class`]) is
    /// trusted (isolated atoms are exempt), else the first untrusted class's refusal (pair before
    /// many-body) and whether that class is short of samples.
    pub(super) fn class_untrusted(&self, counts: &[usize]) -> Option<(Refusal, bool)> {
        let present: Vec<usize> = (0..ENV_CLASSES.len()).filter(|&c| counts.iter().any(|&k| atom_class(k) == Some(c))).collect();
        present.into_iter().find_map(|c| self.one_class_untrusted(c))
    }

    /// The class rule's refusal for class `c` (index into [`ENV_CLASSES`]), `None` when trusted.
    fn one_class_untrusted(&self, c: usize) -> Option<(Refusal, bool)> {
        let q = self.report.as_ref()?.q_prime;
        let refusal = |text: String| Refusal::new(text, TeachRule::Class(env_class_code(c)), None);
        let Some(l) = self.class_loo.get(c) else {
            return Some((refusal(format!("surrogate: this kind of environment isn't learned well enough yet ({}: no leave-one-out numbers)", ENV_CLASSES[c])), true));
        };
        if l.trusted {
            return None;
        }
        // the class rule's own verdict (`calibrate::class_verdict`, plan review M3), never a re-test
        // of the numbers, in the verdict's own words (`calibrate::refusal_words`: one wording for
        // the verdict and this note, the M8a Task 2 review's M2); `None` cannot occur here
        // (`trusted == refusal.is_none()`) and is worded as the uncalibrated force rule
        let (key, value, limit) = l.refusal.unwrap_or((ClassKey::Force, l.force_max.max(l.force_rms), 1.0));
        let words = super::calibrate::WordsInput {
            samples: l.samples,
            force_rms: l.force_rms,
            force_max: l.force_max,
            energy_max: l.energy_max,
            measured: l.measured,
            measured_energy: l.measured_energy,
            evidence: l.evidence,
            calibration: l.calibration,
        };
        let limit = if key == ClassKey::Samples { q as f64 } else { limit };
        let text = format!("surrogate: this kind of environment isn't learned well enough yet ({}: {})", l.class, super::calibrate::refusal_words(key, value, limit, &words));
        Some((refusal(text), key == ClassKey::Samples))
    }

    /// Rebuild the model from the samples (`surrogate` module doc): the accumulator takes the samples added
    /// since the last refit, or is rebuilt from the store after a species or reference change;
    /// then one hyperparameter search and solve, and one pass over the store for the training
    /// residuals. Judges the held-out draft guesses (`judge_held_out`) and re-judges the failed audits
    /// (`rejudge_failed_audits`). Computes the refit jump for the next evaluate and bumps
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
        // the held-out draft guesses strict mode's domain now reaches: judged, promoted, refitted
        if self.judge_held_out() > 0 {
            report = self.fit_store()?;
            if self.recoverable_fail() {
                report = self.recover_trust(report);
            }
        }
        self.rejudge_failed_audits(true);
        // the verdicts that stand now are the next refit's hysteresis memory (M8a Task 2)
        self.commit_class_memory();
        self.surface_moved(was_trusted, was_model).map_err(FitError::Describe)?;
        Ok(report)
    }

    /// What the fit that stands predicts at a failed audit's geometry **without** the audited
    /// answer — the exact leave-one-out prediction when the answer is in the fit (`surrogate::loo`),
    /// the plain prediction when it is not — judged by the audit's own criteria: (teacher − model
    /// gap, the force failure if any, how it was predicted). An undefined leave-one-out fails (NaN).
    fn judge_case(&self, case: &FailedAudit) -> Option<(f64, Option<(f64, f64)>, &'static str)> {
        let model = self.model.as_ref()?;
        let n = case.pos.len().max(1) as f64;
        let in_fit = case.index.filter(|&i| i < self.accumulated && !self.excluded(i));
        let (e_pred, f_pred, how) = match in_fit.and_then(|i| self.store.as_ref().map(|st| &st.samples[i])) {
            Some(smp) => match surrogate::loo::leave_one_out(model, smp) {
                Some(l) => {
                    let per = if model.energy_per_atom { n } else { 1.0 };
                    let f: Vec<[f64; 3]> = smp.forces.iter().zip(&l.forces).map(|(f, r)| [f[0] - r[0], f[1] - r[1], f[2] - r[2]]).collect();
                    (smp.energy - l.energy * per, f, "the refit's model without the audited sample (leave-one-out)")
                }
                None => (f64::NAN, vec![[f64::NAN; 3]; case.pos.len()], "the leave-one-out prediction is undefined (the sample alone spans a direction)"),
            },
            None => match model.predict(&case.pos, &case.z, false) {
                Ok(p) => (p.energy, p.forces, "the refit's model (the audited answer is not in the fit)"),
                Err(_) => (f64::NAN, vec![[f64::NAN; 3]; case.pos.len()], "the refit's model cannot describe the geometry"),
            },
        };
        let gap = case.energy - e_pred;
        let force_fail = worst_force_residual(&self.policy, &f_pred, &case.forces, audit_force_floor(&case.ionic)).filter(|(d, b)| !(d <= b));
        Some((gap, force_fail, how))
    }

    /// The trust rule's audit numbers from the active failed audits (the first one's), or none.
    pub(super) fn sync_audit_failed(&mut self) {
        match self.failed_audits.first() {
            None => {
                // without a case (an older state, or an in-step failure before a case existed) a
                // refit clears, as before
                self.audit_failed = None;
                self.audit_force = None;
            }
            Some(c) => {
                self.audit_origin = c.origin;
                if let Some((gap, force, _)) = self.judge_case(c) {
                    self.audit_failed = Some(gap);
                    self.audit_force = force;
                } else {
                    self.audit_failed = self.audit_failed.or(Some(f64::NAN));
                }
                self.audit_atoms = c.pos.len();
            }
        }
    }

    /// The failed audits after a refit or an install (module doc, rule 8, "A failed audit stays
    /// failed"): each active case judged again at its own geometry by [`judge_case`]
    /// (Self::judge_case); passed → cleared, and the next [`AUDIT_BURST`] answered steps are audited
    /// (round 1, SHOULD-1: a twin of the audited geometry can make its leave-one-out pass without
    /// testing the neighbourhood the failure called into question); quarantined or superseded by
    /// the recovery → cleared, the quarantine fence takes over; failed → still untrusted, and after
    /// [`AUDIT_REJUDGE_LIMIT`] failed re-judges (`count`: a refit, not an install) the case becomes
    /// an **audit fence** (round 1, SHOULD-3): the global untrust turns local — geometries within
    /// the quarantine radius of it fall back, everything else is judged by the rest of the policy
    /// (the class rule still sees the sample's leave-one-out when it is fitted). Fenced cases are
    /// judged too and lifted when they pass — **except a shell fence** (`AuditOrigin::ShellCheck`;
    /// final review C1): it is a measured structural limitation of the cutoff, not a model error
    /// a refit can cure, and a one-point re-judgement at the check's geometry says nothing about
    /// the band it fences (the fit can lean on a nearby row and pass there while f_c → 0 beyond
    /// it); it stands until a later pass's shell check for the pair measures the shell again.
    pub(super) fn rejudge_failed_audits(&mut self, count: bool) {
        if self.audit_failed.is_some() && self.failed_audits.is_empty() {
            self.audit_failed = None;
            self.audit_force = None;
        }
        let quarantined = |s: &SurrogateKs, c: &FailedAudit| c.index.is_some_and(|i| matches!(s.quarantine.get(i), Some(&Q_HELD) | Some(&Q_SUPERSEDED)));
        let cases = std::mem::take(&mut self.failed_audits);
        for mut case in cases {
            if quarantined(self, &case) {
                self.user_notes.push("surrogate: the failed audit's sample was quarantined by the refit — its neighbourhood falls back (the fence) instead of the whole model".into());
                continue;
            }
            let Some((gap, force_fail, how)) = self.judge_case(&case) else {
                self.failed_audits.push(case);
                continue;
            };
            let n = case.pos.len().max(1) as f64;
            let tier = self.policy.energy_std_per_atom;
            if (gap.abs() / n <= tier) && force_fail.is_none() {
                self.audit_burst = AUDIT_BURST;
                self.user_notes.push(format!(
                    "surrogate: the failed audit is passed by {how}: teacher − model {gap:+.2e} Ha — trust restored; the next {AUDIT_BURST} answered steps are audited"
                ));
                continue;
            }
            if count {
                case.rejudges += 1;
            }
            if case.rejudges >= AUDIT_REJUDGE_LIMIT {
                self.user_notes.push(format!(
                    "surrogate: the failed audit was not re-earned in {AUDIT_REJUDGE_LIMIT} refits ({how} still off, teacher − model {gap:+.2e} Ha) — it no longer untrusts the whole model: geometries near it fall back (an audit fence)"
                ));
                self.audit_fences.push(case);
                continue;
            }
            self.user_notes.push(match force_fail {
                Some((d, b)) => format!("surrogate: still untrusted — at the failed audit's geometry {how} is off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.2e} Ha)"),
                None => format!("surrogate: still untrusted — at the failed audit's geometry {how} is off by {gap:+.2e} Ha ({:.1e} > {tier:.1e} Ha/atom)", gap.abs() / n),
            });
            self.failed_audits.push(case);
        }
        // fenced cases: lifted when they pass — never a shell fence (structural, C1)
        let fences = std::mem::take(&mut self.audit_fences);
        for case in fences {
            if case.origin == AuditOrigin::ShellCheck {
                self.audit_fences.push(case);
                continue;
            }
            match self.judge_case(&case) {
                Some((gap, None, how)) if gap.abs() / case.pos.len().max(1) as f64 <= self.policy.energy_std_per_atom => {
                    self.user_notes.push(format!("surrogate: an audit fence is lifted — {how} now passes there (teacher − model {gap:+.2e} Ha)"));
                }
                _ if quarantined(self, &case) => {}
                _ => self.audit_fences.push(case),
            }
        }
        self.sync_audit_failed();
    }

    /// Round 1, SHOULD-3: an active failed audit whose composition the scene no longer holds (an atom
    /// deleted, another molecule) can never be re-judged by what the scene visits — it becomes an
    /// audit fence instead of untrusting the model everywhere for good; and (R1-4) a fence made that
    /// way — not one past `AUDIT_REJUDGE_LIMIT` — becomes an active failed audit again as soon as the
    /// scene holds its composition (an atom added and removed again must not turn a fresh global
    /// untrust into a local fence without a single re-judge). Whether anything changed.
    pub(super) fn fence_failed_audits_off_composition(&mut self, z: &[u8]) -> bool {
        let mut want = z.to_vec();
        want.sort_unstable();
        let holds = |c: &FailedAudit| {
            let mut have = c.z.clone();
            have.sort_unstable();
            have == want
        };
        let (keep, off): (Vec<FailedAudit>, Vec<FailedAudit>) = std::mem::take(&mut self.failed_audits).into_iter().partition(|c| holds(c));
        // a shell fence is never an active failed audit (it never untrusts globally, C1)
        let (back, fenced): (Vec<FailedAudit>, Vec<FailedAudit>) =
            std::mem::take(&mut self.audit_fences).into_iter().partition(|c| c.origin != AuditOrigin::ShellCheck && c.rejudges < AUDIT_REJUDGE_LIMIT && holds(c));
        self.failed_audits = keep;
        self.audit_fences = fenced;
        if off.is_empty() && back.is_empty() {
            return false;
        }
        if !off.is_empty() {
            self.pending_notes.push(format!(
                "surrogate: {} failed audit(s) of another composition no longer untrust the whole model — their neighbourhoods fall back (audit fences)",
                off.len()
            ));
        }
        if !back.is_empty() {
            self.pending_notes.push(format!(
                "surrogate: {} failed audit(s) of this composition untrust the model again — not yet re-earned",
                back.len()
            ));
        }
        self.audit_fences.extend(off);
        self.failed_audits.extend(back);
        self.sync_audit_failed();
        true
    }

    /// Ruling (shell → fence), 2026-10-19: a failed shell check is a measured structural limitation
    /// of the cutoff, fenced at once (`teach::judge_shell_checks`). Strict mode refuses a geometry
    /// in which a pair of the fenced species lies in the **shell band** [`SCAN_CUTOFF_FRACTION`]·r_cut
    /// ≤ r < r_cut — what the cutoff truncates there was measured, not assumed — and names the
    /// limitation; the fence's own geometry also refuses its quarantine-radius neighbourhood
    /// (`near_audit_fence`, as every audit fence). The fenced species pair is the check's own
    /// (`FailedAudit::pair`, saved with the fence; final review I2): the band is refused for that
    /// pair wherever the check point itself lay — a short user plan can put the first point beyond
    /// 5/6 r_cut at or past r_cut, where a measured interaction is a stronger truncation still —
    /// and a bystander pair of the check's geometry that happened to lie in the band is never
    /// fenced by it.
    pub(super) fn shell_fence_refusal(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<Refusal> {
        let rc = self.model.as_ref()?.basis.r_cut;
        let lo = crate::sampling::SCAN_CUTOFF_FRACTION * rc;
        let dist = |p: &[[f64; 3]], a: usize, b: usize| (0..3).map(|k| (p[a][k] - p[b][k]).powi(2)).sum::<f64>().sqrt();
        let key = |a: u8, b: u8| if a <= b { (a, b) } else { (b, a) };
        let fenced: Vec<(u8, u8)> = self.audit_fences.iter().filter(|c| c.origin == AuditOrigin::ShellCheck).filter_map(|c| c.pair).map(|(a, b)| key(a, b)).collect();
        if fenced.is_empty() {
            return None;
        }
        let n = pos.len();
        for a in 0..n {
            for b in a + 1..n {
                let r = dist(pos, a, b);
                if (lo..rc).contains(&r) && fenced.contains(&key(z[a], z[b])) {
                    return Some(Refusal::new(
                        format!(
                            "surrogate: {}–{} at {r:.2} Bohr is in the cutoff shell ({lo:.2}–{rc:.2} Bohr), where a shell check measured an interaction the model's cutoff truncates (a structural limitation, fenced) — DFT answers here",
                            units::elements::symbol(z[a]),
                            units::elements::symbol(z[b])
                        ),
                        TeachRule::NearQuarantine,
                        Some(b),
                    ));
                }
            }
        }
        None
    }

    /// The audit fence a geometry is inside (round 1, SHOULD-3): within the quarantine radius
    /// (`QUARANTINE_FRACTION` · `max_descriptor_distance`, training-spread units; the worst query
    /// atom's nearest atom of its species in the fenced geometry) of a fenced failed audit.
    pub(super) fn near_audit_fence(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<usize> {
        if self.audit_fences.is_empty() {
            return None;
        }
        let (model, env) = (self.model.as_ref()?, self.envelope.as_ref()?);
        let basis = &model.basis;
        let (Ok(desc), Ok(sp)) = (describe_values(basis, pos, z), basis.species_indices(z)) else { return None };
        let limit = QUARANTINE_FRACTION * self.policy.max_descriptor_distance;
        self.audit_fences.iter().position(|c| {
            let atoms = describe_sample(basis, &Sample { pos: c.pos.clone(), z: c.z.clone(), energy: c.energy, forces: c.forces.clone(), tag: SampleTag::Audit, weight: 1.0 });
            let worst = sp.iter().zip(&desc).fold(0.0f64, |worst, (qs, qd)| {
                let best = atoms.iter().filter(|(s, _)| s == qs).map(|(s, b)| spread_distance(env, *s, qd, b)).fold(f64::INFINITY, f64::min);
                worst.max(best)
            });
            worst <= limit
        })
    }

    /// The refusal of a geometry inside an audit fence (`near_audit_fence`).
    pub(super) fn audit_fence_refusal(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<Refusal> {
        if let Some(r) = self.shell_fence_refusal(pos, z) {
            return Some(r);
        }
        self.near_audit_fence(pos, z).map(|_| {
            Refusal::new(
                format!(
                    "surrogate: within {:.2} training spreads of a failed audit's geometry the model has not re-earned (an audit fence)",
                    QUARANTINE_FRACTION * self.policy.max_descriptor_distance
                ),
                TeachRule::NearQuarantine,
                None,
            )
        })
    }

    /// The geometry-specific refusals at (`pos`, `z`) whatever the fit's global state (round 1,
    /// MUST-1): a non-finite answer, the envelope (5a–5c, 5e — not "no grade", which is global),
    /// the novelty (5d), a quarantine or audit fence. `None`: strict mode refuses there only for
    /// reasons that lift with trust (4, 4b, 6, 7), or answers.
    pub(super) fn geometry_refusal(&self, pos: &[[f64; 3]], z: &[u8]) -> Option<Refusal> {
        let model = self.model.as_ref()?;
        let pred = match model.predict(pos, z, true) {
            Ok(p) => p,
            Err(e) => return Some(Refusal::new(format!("surrogate: {e}"), TeachRule::NonFiniteAnswer, None)),
        };
        if let Some(r) = non_finite_answer(&pred) {
            return Some(r);
        }
        if let (_, Some(r)) = self.envelope_outside(model, pos, z) {
            if r.rule != TeachRule::NoGrade {
                return Some(r);
            }
        }
        if pred.energy_std.is_some() {
            if let Some((nu, at)) = self.novelty(pos, z).filter(|(nu, _)| *nu > 1.0) {
                return Some(novelty_refusal(nu, at));
            }
        }
        if !self.near_quarantined_cached(pos, z).is_empty() {
            return Some(Refusal::new("surrogate: near a quarantined geometry", TeachRule::NearQuarantine, None));
        }
        self.audit_fence_refusal(pos, z)
    }

    /// The held-out draft guesses as a test set (round 1, MUST-1): at a refit, every `Q_DRAFT`
    /// sample whose geometry strict mode no longer refuses for a geometry-specific reason
    /// (`geometry_refusal`) is judged by the plain prediction of the fit that never saw it, by the
    /// audit's own criteria, and promoted into the fit (`Q_NONE`); a failure becomes a failed audit
    /// (rule 8: it untrusts until re-earned). How many were promoted (the caller refits then).
    pub(super) fn judge_held_out(&mut self) -> usize {
        let held: Vec<usize> = (0..self.quarantine.len()).filter(|&i| self.quarantine[i] == Q_DRAFT).collect();
        if held.is_empty() || self.model.is_none() {
            return 0;
        }
        let ionic = self.store.as_ref().map_or("none".to_string(), |s| s.provenance.ionic.clone());
        let mut promoted = 0;
        for i in held {
            let s = self.store.as_ref().expect("held samples are stored").samples[i].clone();
            if self.geometry_refusal(&s.pos, &s.z).is_some() {
                continue;
            }
            let case = FailedAudit { pos: s.pos.clone(), z: s.z.clone(), energy: s.energy, forces: s.forces.clone(), index: Some(i), ionic: ionic.clone(), rejudges: 0, origin: AuditOrigin::HeldOutGuess, pair: None };
            let Some((gap, force_fail, _)) = self.judge_case(&case) else { continue };
            self.quarantine[i] = Q_NONE;
            // judged as an audit now: its tag says so (a file keeps it fitted, never re-held)
            if let Some(st) = self.store.as_mut() {
                st.samples[i].tag = SampleTag::Audit;
            }
            promoted += 1;
            let n = s.pos.len().max(1) as f64;
            if !(gap.abs() / n <= self.policy.energy_std_per_atom) || force_fail.is_some() {
                self.user_notes.push(format!(
                    "surrogate: a held-out draft-mode audit (#{i}) is now in strict mode's domain and the model is off there (teacher − model {gap:+.2e} Ha{}) — a failed audit: untrusted until re-earned",
                    force_fail.map_or(String::new(), |(d, b)| format!(", force {d:.1e} > {b:.1e} Ha/Bohr"))
                ));
                self.failed_audits.push(case);
            }
        }
        if promoted > 0 {
            self.species_dirty = true; // rebuild with them
            self.user_notes.push(format!("surrogate: {promoted} held-out draft-mode audit(s) promoted into the fit — strict mode's domain now reaches them"));
        }
        promoted
    }

    /// The training residuals exceed the policy's thresholds (the trust rule's S8 part).
    pub(super) fn residuals_fail(&self) -> bool {
        self.training_residuals.as_ref().is_some_and(|t| energy_residual_fails(t, &self.policy) || force_residual_fails(t))
    }

    /// The fit's force-label noise σ_f exceeds the absolute force tier `force_std_abs` (the trust
    /// rule's noise condition, M6 Task 11; module doc, "Trust rule").
    pub(super) fn noise_fails(&self) -> bool {
        self.report.as_ref().is_some_and(|r| force_noise_fails(r, &self.policy))
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
            class_memory: self.class_memory.clone(),
            class_worst: self.class_worst.clone(),
            class_blockers: self.class_blockers.clone(),
            degree_cap: self.degree_cap,
            fit_members: self.fit_members.clone(),
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
        self.class_memory = s.class_memory;
        self.class_worst = s.class_worst;
        self.class_blockers = s.class_blockers;
        self.degree_cap = s.degree_cap;
        self.fit_members = s.fit_members;
        self.model_gen += 1; // a view cache key: the model moved back
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
        // M7: the labels of a pseudopotential surface carry the GPU path's scatter (policy.rs `PSEUDO_FORCE_NOISE`)
        let opts = FitOptions { sigma_f_min: force_noise_prior(&provenance.ionic), ..FitOptions::default() };
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
        self.fits += 1;
        // readouts (activity Task 4 review): the views' cache key and their `Fitted`
        self.model_gen += 1;
        self.fit_members = (accumulated, self.quarantine.clone());
        // a failed audit is not cleared by the fit itself: `refit` re-judges it once the fit that
        // stands is settled (module doc, rule 8, "A failed audit stays failed")
        self.training_residuals = self.compute_training_residuals();
        self.refresh_class_loo();
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
        self.report.as_ref().is_none_or(|r| samples_short(r) || dof_short(r.residual_dof_e) || dof_short(r.residual_dof_f))
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
            let reason = self.streak_reason.as_ref().map_or("surrogate: fallback", |r| r.text.as_str());
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

    /// [`stuck`](Self::stuck) in structured form (activity spec A5): the same streak and state,
    /// the rule and atom of the streak's latest fallback; the rule is `None` when no reason was
    /// stored, never a guessed one (plan review S10).
    pub fn stuck_info(&self) -> Option<StuckInfo> {
        (self.streak >= STUCK_STEPS && self.trusted()).then(|| {
            let r = self.streak_reason.as_ref();
            StuckInfo {
                rule: r.map(|r| r.rule),
                atom: r.and_then(|r| r.atom),
                steps: self.streak,
                box_streak: self.streak_box >= STUCK_STEPS,
                gap: self.streak_gap,
                face_density: self.streak_box_face,
            }
        })
    }
}
