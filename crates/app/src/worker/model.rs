//! The estimated solver's side of the worker (the parent module doc, "The estimated solver"): the
//! readouts ([`ModelStatus`], [`RefitLog`]), the automatic first pass ([`first_pass`], [`PassOffer`],
//! and its one re-sizing from the median of its first solves, [`pass_resize`]), the
//! active-learning refits and the per-solve cost estimate. The provider is reached only through
//! `sim::Trainable`. Split out of `app/src/sim.rs` in M6 Task 1.

use super::*;

/// The last refit the worker ran (`worker` module doc, "Refits").
#[derive(Clone, Debug, PartialEq)]
pub struct RefitLog {
    /// Wall time of the refit (ms).
    pub ms: f64,
    /// The driver's step count when it ran.
    pub at_step: u64,
    /// `Ok(samples fitted)` or the fit error.
    pub result: Result<usize, String>,
}

/// The surrogate's state as the panel shows it (`worker` module doc, "Readouts"). Atomic units.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStatus {
    /// The teacher's name (`ForceProvider::name`).
    pub teacher: &'static str,
    /// A model has been fitted (it may still be untrusted).
    pub fitted: bool,
    /// The trust rule passes (`SurrogateKs::trusted`).
    pub trusted: bool,
    /// Why not (the first failing condition of the trust rule), when fitted and untrusted.
    pub untrusted_reason: Option<String>,
    /// Training samples held, and whether some arrived since the last refit.
    pub samples: usize,
    pub pending_refit: bool,
    /// The surrogate's surface epoch (bumped when a refit moves the surface).
    pub epoch: u64,
    /// The last fit: samples, Q′, rmse E (Ha/atom), rmse F (Ha/Bohr per component).
    pub fit: Option<(usize, usize, f64, f64)>,
    /// The worst training residuals (Ha/atom; Ha/Bohr per component, and that atom's bound).
    pub residuals: Option<(f64, f64, f64)>,
    /// The tightest per-atom force-uncertainty bound at the current forces (Ha/Bohr per
    /// component): `FallbackPolicy::force_bound(min_j |F_j|)` — the surrogate answers only if
    /// every atom's uncertainty is inside its own bound, so this is the strictest one.
    pub force_bound: f64,
    /// The model's provenance differs from the scene's: every step falls back (rule 3).
    pub provenance_mismatch: Option<String>,
    /// The fitted model's provenance with its recorded mass set
    /// (`surrogate::Provenance::describe_recorded`: the masses are recorded, not compared — P2).
    pub provenance: Option<String>,
    /// The worker's last refit.
    pub last_refit: Option<RefitLog>,
    /// Refits the worker has run (the sampling pass's own refits not counted).
    pub refits: u64,
    /// The autoload of [`Command::EnsureModel`] was refused (the driver's message).
    pub autoload_refused: Option<String>,
    /// The policy's audit cadence ([`Command::SetAudit`]).
    pub audit_every: usize,
    /// Draft mode is on ([`Command::SetDraft`]).
    pub draft: bool,
    /// The driver step the last audit was judged at: the stamp `sim::Dynamics` put on its
    /// teaching event (activity Task 5, plan review M6(a)). For an audit in the step that is the
    /// audited step itself; for a background audit it is the step its verdict arrived at,
    /// `last_audit.unaudited_steps` **answered** steps after its dispatch — fallback steps in
    /// between are not counted in that lag, so no dispatch step is computed from it (it used to
    /// be "the snapshot step less the lag", which mixed the two kinds of step and, at several
    /// steps a tick, named the tick's last step). `None` without an audit record, and from the
    /// start of a new run (a scene replacement, a recovery: the step would be the old run's) until
    /// the next audit.
    pub last_audit_step: Option<u64>,
    /// The most recent teacher gap (E_DFT − E_model on a fallback booked on the model's
    /// surface, Ha) and the step it was seen at — latched, unlike `DynSnapshot::teacher_gap`.
    pub last_gap: Option<(f64, u64)>,
    /// Samples held in quarantine (ruling S15) and the last refit's recovery note.
    pub quarantined: usize,
    pub recovery: Option<String>,
    /// Superseded quarantined samples, box-affected samples excluded from the fit, and samples
    /// kept since the last refit: with `quarantined`, what the store holds beyond the fit
    /// (`samples` = fit + quarantined + superseded + box_excluded + unfitted right after a refit;
    /// the panel's "fit vs held", M6 Task 11).
    pub superseded: usize,
    pub box_excluded: usize,
    pub unfitted: usize,
    /// Stored box-affected samples (excluded yet or not) and the trust floor their exclusion waits
    /// for (`sim::TrainableStatus`): the Model tab names the box when it keeps the model
    /// untrusted ([`box_blocks_trust`]).
    pub box_affected: usize,
    pub trust_floor: Option<usize>,
    /// The panel's "fit vs held" line, final text ([`fit_vs_held`]); `None` before a fit.
    pub fit_vs_held: Option<String>,
    /// The audits run on a background teacher (ruling S14), one is in flight, and the last one's
    /// outcome.
    pub audits_background: bool,
    pub audit_in_flight: bool,
    pub last_audit: Option<sim::AuditRecord>,
    /// `last_audit` judged the current fit (`sim::TrainableStatus::last_audit_this_fit`): with
    /// `last_audit`, never audited / audited only before the last refit / audited this fit.
    pub last_audit_this_fit: bool,
    /// A trusted model that keeps falling back where the molecule is: why
    /// (`SurrogateKs::stuck`).
    pub stuck: Option<String>,
    /// Per environment class (pair, many-body): the fit's leave-one-out errors and whether the
    /// class may be answered (`providers::surrogate`, "Class-wise trust"; for the Model tab).
    pub class_loo: Vec<sim::ClassLoo>,
    /// The model's lifetime teaching solves by origin and how many were kept (activity spec A2;
    /// since it was built, forgotten, or as loaded from its file).
    pub solves: sim::SolveCounts,
    /// The trust rule's conditions in the order it tests them, each with its value, limit and
    /// whether it is met (activity spec A4); the first unmet one is `untrusted_reason`. Empty
    /// before the first fit.
    pub trust_checks: Vec<sim::activity::TrustCheck>,
    /// `stuck` in structured form (activity spec A5): the rule, atom, streak and box flag the
    /// Activity view words.
    pub stuck_info: Option<sim::activity::StuckInfo>,
    /// The model generation (`sim::TrainableStatus::model_gen`): what the training views'
    /// rebuilds are keyed on (`worker::activity`, `ViewKey`).
    pub model_gen: u64,
    /// The expected teacher cost of one pass solve on this scene (s): the running pass's (or
    /// the last pass's) measured rate, else the last teacher evaluation's wall time (the scene's
    /// cold first evaluation, as measured — no multiplier, M6 Task 9; not a bound either way: a
    /// pass's own solves differ — cold references, warm starts). The panel's ETA for a pass is
    /// solves × this.
    pub solve_s: Option<f64>,
    /// `solve_s` is a pass's measured rate (else the scene's first solve): the panel says which.
    pub solve_s_measured: bool,
    /// The running pass's remaining time (s) at its measured rate (after ≥ 2 solves).
    pub pass_eta_s: Option<f64>,
    /// The automatic first pass is waiting for "Start anyway" ([`PassOffer`]).
    pub pass_offer: Option<PassOffer>,
}

/// The box keeps the model untrusted (diag-3h, a user's H₂ + H at the Accurate grid's 12 Bohr cell:
/// 130 of 145 samples box-affected, refused as "max training energy residual 2.2e-3" — the same
/// scene in an 18 Bohr cell is trusted at 7.6e-5 Ha/atom): `Some((in the fit, held))` when the
/// model is fitted and untrusted, some box-affected samples are still in the fit, and the others
/// alone are below the trust floor. **Why this threshold and not a share:** box-affected samples
/// leave the fit as soon as the clean ones reach the floor (`providers::surrogate`,
/// "Box-affected answers"); below it they are necessarily fitted, and their labels depend on
/// where the molecule sits, which no translation-invariant model can represent — so whatever
/// the refusal's residual or accuracy condition says, the box is in the fit it judged, and growing
/// the clean set (a bigger cell) is the way out. No tuned fraction.
///
/// **Only for a condition of the fit the box samples are in** (activity plan review S15): the
/// first unmet [`TrustKey`](sim::activity::TrustKey) must be the training energies, the training
/// forces or the force noise. It is **not** blamed for a count (`Samples`, `ResidualDofE/F`:
/// Sample more cures it, whatever the box does, M6 final re-review), nor for a failed audit
/// (`Audit`, `AuditForces`: judged against a fresh teacher answer that is not box-affected).
/// Keyed on the check list the provider exports, no longer on the refusal's wording.
pub fn box_blocks_trust(m: &ModelStatus) -> Option<(usize, usize)> {
    use sim::activity::TrustKey;
    let in_fit = m.box_affected.saturating_sub(m.box_excluded);
    let floor = m.trust_floor?;
    let clean = m.samples.saturating_sub(m.box_affected + m.quarantined + m.superseded);
    let fit_condition = m
        .trust_checks
        .iter()
        .find(|c| !c.met)
        .is_some_and(|c| matches!(c.key, TrustKey::EnergyResidual | TrustKey::ForceResidual | TrustKey::ForceNoise));
    (m.fitted && !m.trusted && fit_condition && in_fit > 0 && clean < floor).then_some((in_fit, m.samples))
}

/// The hover text of the "fit vs held" line.
pub const FIT_VS_HELD_HELP: &str = "why the fit uses fewer samples than the model holds: box-affected samples (the teacher's density on the outermost grid plane above the box rule) are excluded once enough others exist — they carry the box's confinement, which no invariant model can learn; quarantined ones are those the model cannot represent; superseded ones were contradicted by a fresh answer; new ones arrived after the last refit";

/// "fit N of S held: B box-excluded, Q quarantined, X superseded, P newer than the fit" — every
/// sample the store holds beyond the fit, by reason (exact right after a refit, M6 Task 11:
/// the user's unexplained "230 vs 250").
pub fn fit_vs_held(fit: usize, held: usize, box_excluded: usize, quarantined: usize, superseded: usize, unfitted: usize) -> String {
    let parts: Vec<String> = [(box_excluded, "box-excluded"), (quarantined, "quarantined"), (superseded, "superseded"), (unfitted, "newer than the fit")]
        .into_iter()
        .filter(|&(k, _)| k > 0)
        .map(|(k, what)| format!("{k} {what}"))
        .collect();
    if parts.is_empty() {
        format!("{fit} of {held} held")
    } else {
        format!("{fit} of {held} held: {}", parts.join(", "))
    }
}

/// A sampling pass is running (not finished, not cancelled).
pub fn sampling_active(s: &DynSnapshot) -> bool {
    s.sampling.as_ref().is_some_and(|p| !p.finished)
}

/// The driver's provider as a [`Trainable`], if it can be taught (the estimated solver): the
/// worker's refits, readouts and density requests (`providers::surrogate` module doc). Never a
/// downcast.
pub(super) fn trainable(dy: &mut Dynamics) -> Option<&mut dyn Trainable> {
    dy.provider_mut().as_trainable()
}

/// The automatic first sampling pass (`Command::EnsureModel`) aims to finish within this many
/// seconds (Task 9 item 4): it starts with the panel's plan and is scaled down to the target
/// after its first [`PASS_SIZING_SOLVES`] solves ([`pass_resize`]), never below what trust
/// needs; a pass whose fewest trust-earning solves already exceed the target at the scene's
/// cold evaluation waits for the user's "Start anyway" ([`PassOffer`]).
pub const PASS_TARGET_S: f64 = 120.0;

/// The automatic first pass is re-sized once, from the median wall time of this many of its
/// first solves (M6 Task 9; M5 handoff: "size from a median of the reference solves"). The
/// median, not one cold solve × 1.6 (M5's estimate, which saw 72–115 solves for one scene under
/// GPU sharing): one slow solve — the cold isolated-atom reference that runs first, a solve
/// slowed by the viewer's own GPU work — moves it not at all. Three because a pass's first
/// three solves are the references and its first perturbations, which run anyway; the timing
/// is the worker's (one tick's wall time over the solves it made — its jobs less the
/// cell-guard skips; a reference found in the cache still counts its multiplicities, which the
/// median absorbs — one solve per tick for any real solve, which outlasts the 16 ms tick
/// budget — counted only for ticks that did nothing but the pass: paused, no queued steps,
/// fresh forces).
///
/// Measured (`diag_pass_rate`, RX 7900 XTX), for what a median of the first three sees: H₂
/// 41³/49³ — first evaluation 0.20/0.23 s, H reference ≈ the first evaluation, perturbations
/// 0.17/0.21 s, scans 0.16/0.19 s, hot burst 0.08/0.10 s; H₂O 41³/57³ — first evaluation
/// 0.81/1.11 s, perturbations 1.28/1.87 s, and the O-atom reference scan 13–15 s a multiplicity
/// (open-shell, cold). H₂O's first three solves are its H reference and two O multiplicities,
/// so its median is an O reference's and sizes the pass to its trust floor (a later "Sample
/// more" finds the references cached and is estimated from geometry solves: `track_cost`).
pub const PASS_SIZING_SOLVES: usize = 3;

/// A first pass that would take longer than [`PASS_TARGET_S`] even at the fewest solves that can
/// earn trust: not started until the user confirms (the panel's "Start anyway").
#[derive(Clone, Debug, PartialEq)]
pub struct PassOffer {
    /// The plan "Start anyway" starts (the panel's, scaled down as far as trust allows).
    pub plan: SamplingPlan,
    pub solves: usize,
    /// Estimated wall time (s) at `per_solve_s`, the scene's first solve — an estimate with
    /// its basis, not a bound (the panel says so; a pass's own solves differ).
    pub eta_s: f64,
    pub per_solve_s: f64,
    /// The panel's plan unscaled: solves and estimated wall time (s).
    pub full_solves: usize,
    pub full_eta_s: f64,
}

/// What the automatic first pass does ([`first_pass`]).
#[derive(Clone, Debug, PartialEq)]
pub enum FirstPass {
    /// Start the panel's plan; it is re-sized after its first solves ([`pass_resize`]).
    Start(SamplingPlan),
    /// Even the fewest solves that can earn trust exceed the target: ask.
    Offer(PassOffer),
}

/// The first-pass decision (Task 9 item 4; the offer rule): `plan` on `scene` at `per_solve_s`
/// seconds a solve (the scene's cold first evaluation, as measured: [`ModelStatus::solve_s`])
/// against [`PASS_TARGET_S`] (5 % slack for the scaled plan's rounding). The plan always starts
/// whole — the size is set from the pass's own solves ([`pass_resize`]) — unless even its
/// fewest trust-earning solves would take longer: then an offer, with that plan.
///
/// Since M6 Task 9 the cold solve is taken at 1× (M5: 1.6×), so the threshold moved with it: a
/// scene whose trust floor costs 79–126 s at the cold rate now starts without asking, and a
/// water-like scene then runs ≈ 1.7× that plus its references (references.md, "M6 Task 9").
pub fn first_pass(scene: &sim::Scene, plan: SamplingPlan, per_solve_s: Option<f64>) -> FirstPass {
    let Some(per) = per_solve_s.filter(|c| *c > 0.0) else { return FirstPass::Start(plan) };
    // the fewest solves that can earn trust within the target (the plan itself when it fits)
    let floor = plan.scaled_to(scene, (PASS_TARGET_S / per).floor() as usize);
    let n = floor.solves(scene).unwrap_or(0);
    if n as f64 * per <= PASS_TARGET_S * 1.05 {
        return FirstPass::Start(plan);
    }
    let full = plan.solves(scene).unwrap_or(n);
    FirstPass::Offer(PassOffer { plan: floor, solves: n, eta_s: n as f64 * per, per_solve_s: per, full_solves: full, full_eta_s: full as f64 * per })
}

/// The median of `xs` (the mean of the middle two for an even count); `None` when empty.
pub fn median(xs: &[f64]) -> Option<f64> {
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(v[n / 2]),
        _ => Some(0.5 * (v[n / 2 - 1] + v[n / 2])),
    }
}

/// The automatic first pass's one re-sizing (M6 Task 9, [`PASS_SIZING_SOLVES`]): the pass's
/// `plan` on `scene` at the median of `per_solve_s` (s, its first solves) against
/// [`PASS_TARGET_S`]. `Some((plan scaled to the target, the median))` when the plan would take
/// longer — `PlanSolves::scaled_to`, so never below the solves trust needs — and `None` when it
/// fits (or nothing was timed). The running pass is shrunk to it by `Trainable::rescale_pass`.
pub fn pass_resize(scene: &sim::Scene, plan: &SamplingPlan, per_solve_s: &[f64]) -> Option<(SamplingPlan, f64)> {
    let per = median(per_solve_s).filter(|m| m.is_finite() && *m > 0.0)?;
    let max = (PASS_TARGET_S / per).floor() as usize;
    let full = plan.solves(scene).ok()?;
    (full > max).then(|| (plan.scaled_to(scene, max), per))
}

/// The automatic first pass being timed for its re-sizing ([`pass_resize`]).
#[derive(Clone, Debug)]
pub(super) struct PassSizing {
    /// The scene and plan the pass was started with.
    pub scene: sim::Scene,
    pub plan: SamplingPlan,
    /// Wall time per job of each pass-only tick so far (s).
    pub per_solve_s: Vec<f64>,
}

/// While playing, a pending sample waits for a refit at most this long after the previous one
/// (`worker` module doc, "Refits") …
pub(super) const REFIT_INTERVAL: Duration = Duration::from_millis(500);

/// … or until this many samples arrived since it, whichever comes first.
pub(super) const REFIT_SAMPLES: usize = 25;

impl Worker {
    /// [`Command::EnsureModel`] (`worker` module doc, "Teaching"; "Autosave" for the ownership).
    pub(super) fn ensure_model(&mut self, autoload: Option<PathBuf>, plan: SamplingPlan) {
        if trainable(&mut self.dy).is_none() {
            return;
        }
        if let Some(path) = autoload.filter(|p| p.exists()) {
            self.dy.apply(DynCommand::LoadModel(path));
            if let Some(r) = self.dy.snapshot().last_rejection {
                // not ours: the file stays for a later worker on the settings it fits
                self.autoload_refused = Some(r);
                return;
            }
        }
        self.owns_autosave = true;
        // an empty scene needs no model (module doc, "The empty scene"): no pass, no offer
        if self.dy.snapshot().scene.nuclei.is_empty() {
            return;
        }
        let fitted = trainable(&mut self.dy).is_some_and(|s| s.fitted());
        if !fitted && !sampling_active(&self.dy.snapshot()) {
            self.start_first_pass(plan);
        }
    }

    /// The automatic first pass (Task 9 item 4): the panel's plan, re-sized to [`PASS_TARGET_S`]
    /// after its first [`PASS_SIZING_SOLVES`] solves ([`size_pass`](Self::size_pass)); or —
    /// when even the fewest solves that can earn trust take longer at the scene's cold
    /// evaluation — nothing, and a [`PassOffer`] for the panel's "Start anyway" (a plain
    /// `StartSampling`, never re-sized).
    pub(super) fn start_first_pass(&mut self, plan: SamplingPlan) {
        let scene = self.dy.snapshot().scene;
        match first_pass(&scene, plan, self.solve_s) {
            FirstPass::Start(plan) => {
                self.dy.apply(DynCommand::StartSampling(plan.clone()));
                if sampling_active(&self.dy.snapshot()) {
                    self.sizing = Some(PassSizing { scene, plan, per_solve_s: Vec::new() });
                }
            }
            FirstPass::Offer(o) => self.pass_offer = Some(o),
        }
    }

    /// The automatic pass's re-sizing (`round`, after each tick; [`PASS_SIZING_SOLVES`]):
    /// `before` is the pass's (`done`, `skipped`) before the tick, `tick_s` the tick's wall time
    /// and `pass_only` whether the tick could do nothing but the pass. After the third timed
    /// tick the pass is shrunk to [`pass_resize`]'s plan through the driver
    /// (`Dynamics::rescale_sampling`, which publishes the new total from then on) — once — with a
    /// note when the total changed, and `snap` carries the new total. Dropped when the pass ends
    /// first.
    pub(super) fn size_pass(&mut self, snap: &mut DynSnapshot, before: Option<(usize, usize)>, tick_s: f64, pass_only: bool) {
        let Some(sizing) = self.sizing.as_mut() else { return };
        let Some(p) = snap.sampling.as_ref().filter(|p| !p.finished) else {
            self.sizing = None;
            return;
        };
        // the solves the tick made: its jobs less the cell-guard skips (review M3)
        let solved = before.map_or(0, |(done, skipped)| p.done.saturating_sub(done).saturating_sub(p.skipped.saturating_sub(skipped)));
        if !pass_only || solved == 0 {
            return;
        }
        sizing.per_solve_s.push(tick_s / solved as f64);
        if sizing.per_solve_s.len() < PASS_SIZING_SOLVES {
            return;
        }
        let before = p.total;
        let Some(sizing) = self.sizing.take() else { return };
        let Some((plan, per)) = pass_resize(&sizing.scene, &sizing.plan, &sizing.per_solve_s) else { return };
        self.dy.rescale_sampling(&plan);
        let Some(now) = self.dy.snapshot().sampling.filter(|now| now.total != before) else { return };
        let full = sizing.plan.solves(&sizing.scene).unwrap_or(before);
        let times: Vec<String> = sizing.per_solve_s.iter().map(|x| format!("{x:.2}")).collect();
        self.dy.add_note(format!(
            "sampling pass sized from the median of its first {PASS_SIZING_SOLVES} solves ({} s → ≈ {per:.2} s a solve): {before} → {} jobs, ≈ {:.0} s{} (the panel's plan, {full} jobs, would take ≈ {:.0} min) — Sample more adds to it",
            times.join(", "),
            now.total,
            now.total as f64 * per,
            if now.total as f64 * per > PASS_TARGET_S * 1.05 { ", the fewest that can earn trust" } else { "" },
            full as f64 * per / 60.0
        ));
        let d = self.dy.snapshot();
        (snap.sampling, snap.notes, snap.notes_total) = (d.sampling, d.notes, d.notes_total);
    }

    /// Active learning (`worker` module doc, "Refits"): the surrogate kept a fallback sample and no pass
    /// is running (a pass refits at its end).
    /// Whether a refit ran.
    pub(super) fn refit_if_pending(&mut self, snap: &DynSnapshot) -> bool {
        if sampling_active(snap) {
            return false;
        }
        let Some(s) = trainable(&mut self.dy) else { return false };
        // before the first fit there are no reference energies (the pass computes them and
        // makes the first fit at its end): rule-1 fallback samples wait for it
        if !s.pending_refit() || !s.fitted() {
            return false;
        }
        if let (false, Some(at)) = (snap.paused, self.last_refit_at) {
            if at.elapsed() < REFIT_INTERVAL && s.samples() < self.samples_at_refit + REFIT_SAMPLES {
                return false;
            }
        }
        let t = Instant::now();
        let fit = s.refit();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.refits += 1;
        let k = self.refits;
        let st = s.status(&snap.scene, &snap.forces);
        let note = match &fit {
            Ok(summary) => format!(
                "refit #{k}: {summary}, {ms:.0} ms — trusted: {}{}",
                match &st.untrusted_reason {
                    None => "yes".to_string(),
                    Some(why) => format!("no — {why}"),
                },
                st.recovery.as_ref().map_or(String::new(), |n| format!("; {n}"))
            ),
            Err(e) => format!("refit #{k} failed: {e} (retried at the next opportunity)"),
        };
        let samples = st.samples;
        // the fit's sample count: the report the refit just made
        let result = fit.map(|_| st.fit.map_or(0, |f| f.0));
        self.samples_at_refit = samples;
        self.last_refit = Some(RefitLog { ms, at_step: snap.step, result });
        self.last_refit_at = Some(Instant::now());
        self.dy.add_note(note);
        true
    }

    /// The surrogate's readouts (estimated solver only); `forces` are the snapshot's. The
    /// teaching events must have been drained first (`publish`): the last audit's step is their
    /// stamp.
    pub(super) fn model_status(&mut self, scene: &sim::Scene, forces: &[[f64; 3]]) -> Option<ModelStatus> {
        let (last_refit, refits, autoload_refused, last_gap) =
            (self.last_refit.clone(), self.refits, self.autoload_refused.clone(), self.last_gap);
        let (solve_s, solve_s_measured, pass_eta_s, pass_offer) = (self.solve_s, self.solve_s_measured, self.pass_eta_s, self.pass_offer.clone());
        let s = trainable(&mut self.dy)?;
        let st = s.status(scene, forces);
        // no record (never audited, forgotten, a model loaded): no step either
        if st.last_audit.is_none() {
            self.activity.last_audit_step = None;
        }
        Some(ModelStatus {
            teacher: st.teacher,
            fitted: st.fitted,
            trusted: st.trusted,
            untrusted_reason: st.untrusted_reason,
            samples: st.samples,
            pending_refit: st.pending_refit,
            epoch: st.epoch,
            fit: st.fit,
            residuals: st.residuals,
            force_bound: st.force_bound,
            provenance_mismatch: st.provenance_mismatch,
            provenance: st.provenance,
            last_refit,
            refits,
            autoload_refused,
            audit_every: st.audit_every,
            draft: st.draft,
            last_audit_step: self.activity.last_audit_step,
            last_gap,
            quarantined: st.quarantined,
            recovery: st.recovery,
            superseded: st.superseded,
            box_excluded: st.box_excluded,
            unfitted: st.unfitted,
            box_affected: st.box_affected,
            trust_floor: st.trust_floor,
            fit_vs_held: st.fit.map(|f| fit_vs_held(f.0, st.samples, st.box_excluded, st.quarantined, st.superseded, st.unfitted)),
            audits_background: st.audits_background,
            audit_in_flight: st.audit_in_flight,
            last_audit: st.last_audit,
            last_audit_this_fit: st.last_audit_this_fit,
            stuck: st.stuck,
            class_loo: st.class_loo,
            solves: st.solves,
            trust_checks: st.trust_checks,
            stuck_info: st.stuck_info,
            model_gen: st.model_gen,
            solve_s,
            solve_s_measured,
            pass_eta_s,
            pass_offer,
        })
    }

    /// The per-solve cost estimate and the pass ETA ([`ModelStatus::solve_s`], `pass_eta_s`).
    pub(super) fn track_cost(&mut self, snap: &DynSnapshot) {
        let teacher_step = snap.source != sim::Source::Surrogate && !snap.forces_stale && snap.eval.solve_ms > 0;
        match snap.sampling.as_ref().filter(|p| !p.finished) {
            Some(p) => {
                // the isolated-atom reference scans (open-shell and cold: 13–15 s each for O)
                // are cached for later passes, so they are timed apart: the rate a later "Sample
                // more" is estimated with is the geometry solves' alone (Task 9 review M2)
                let refs = p.stage == "references";
                if self.pass_clock.is_some_and(|c| c.2 != refs) {
                    self.pass_clock = None;
                }
                let (t0, d0, _) = *self.pass_clock.get_or_insert((Instant::now(), p.done, refs));
                let solved = p.done.saturating_sub(d0);
                if solved >= 2 {
                    let per = t0.elapsed().as_secs_f64() / solved as f64;
                    if !refs {
                        self.solve_s = Some(per);
                        self.solve_s_measured = true;
                    }
                    self.pass_eta_s = Some(per * p.total.saturating_sub(p.done) as f64);
                }
            }
            None => {
                if self.pass_clock.take().is_none() && teacher_step && self.solve_s.is_none() {
                    // no pass timed yet: the scene's own teacher evaluation, as measured
                    self.solve_s = Some(snap.eval.solve_ms as f64 * 1e-3);
                }
                self.pass_eta_s = None;
            }
        }
    }
}

#[cfg(test)]
mod fit_vs_held_tests {
    use super::fit_vs_held;

    /// M6 Task 11: the user's "230 vs 250" — the panel names every sample held beyond the fit.
    #[test]
    fn the_fit_vs_held_line_names_what_the_fit_left_out() {
        assert_eq!(fit_vs_held(250, 250, 0, 0, 0, 0), "250 of 250 held");
        assert_eq!(fit_vs_held(230, 250, 20, 0, 0, 0), "230 of 250 held: 20 box-excluded");
        assert_eq!(fit_vs_held(70, 76, 3, 1, 1, 1), "70 of 76 held: 3 box-excluded, 1 quarantined, 1 superseded, 1 newer than the fit");
    }
}
