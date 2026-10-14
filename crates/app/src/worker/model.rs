//! The estimated solver's side of the worker (the parent module doc, "The estimated solver"): the
//! readouts ([`ModelStatus`], [`RefitLog`]), the automatic first pass ([`first_pass`], [`PassOffer`]),
//! the active-learning refits and the per-solve cost estimate. The provider is reached only through
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
    /// The step the last audit's geometry was dispatched at: the snapshot step its verdict
    /// arrived with, less its audit lag (`AuditRecord::unaudited_steps`).
    pub last_audit_step: Option<u64>,
    /// The most recent teacher gap (E_DFT − E_model on a fallback booked on the model's
    /// surface, Ha) and the step it was seen at — latched, unlike `DynSnapshot::teacher_gap`.
    pub last_gap: Option<(f64, u64)>,
    /// Samples held in quarantine (ruling S15) and the last refit's recovery note.
    pub quarantined: usize,
    pub recovery: Option<String>,
    /// The audits run on a background teacher (ruling S14), one is in flight, and the last one's
    /// outcome.
    pub audits_background: bool,
    pub audit_in_flight: bool,
    pub last_audit: Option<sim::AuditRecord>,
    /// A trusted model that keeps falling back where the molecule is: why
    /// (`SurrogateKs::stuck`).
    pub stuck: Option<String>,
    /// The expected teacher cost of one pass solve on this scene (s): the running pass's (or
    /// the last pass's) measured rate, else [`PASS_SOLVE_PER_COLD`] × the last teacher
    /// evaluation's wall time. The panel's ETA for a pass is solves × this.
    pub solve_s: Option<f64>,
    /// The running pass's remaining time (s) at its measured rate (after ≥ 2 solves).
    pub pass_eta_s: Option<f64>,
    /// The automatic first pass is waiting for "Start anyway" ([`PassOffer`]).
    pub pass_offer: Option<PassOffer>,
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
/// seconds (Task 9 item 4): the panel's plan is scaled down to it, never below what trust needs;
/// a pass that still takes longer waits for the user's "Start anyway" ([`PassOffer`]).
pub const PASS_TARGET_S: f64 = 120.0;

/// A pass solve costs about this multiple of the scene's first, cold evaluation (the estimate
/// before any pass solve was timed; once a pass runs, its own rate is used). Measured
/// (`diag_pass_rate`, RX 7900 XTX): H₂ 41³/49³ — first evaluation 0.20/0.23 s, perturbations
/// 0.17/0.21 s, scans 0.16/0.19 s, hot burst 0.08/0.10 s (the whole pass ≈ 0.55×); H₂O 41³/57³ —
/// first evaluation 0.81/1.11 s, perturbations 1.28/1.87 s (1.6–1.7×), and the O-atom reference
/// scan 13–15 s a multiplicity (open-shell, cold: not modelled — the live ETA shows it). 1.6:
/// right for the molecules whose passes are long, and still under the 2-minute target for H₂'s
/// default pass (273 × 0.37 s ≈ 100 s at 49³).
pub const PASS_SOLVE_PER_COLD: f64 = 1.6;

/// A first pass that would take longer than [`PASS_TARGET_S`] even at the fewest solves that can
/// earn trust: not started until the user confirms (the panel's "Start anyway").
#[derive(Clone, Debug, PartialEq)]
pub struct PassOffer {
    /// The plan "Start anyway" starts (the panel's, scaled down as far as trust allows).
    pub plan: SamplingPlan,
    pub solves: usize,
    /// Estimated wall time (s) at `per_solve_s`.
    pub eta_s: f64,
    pub per_solve_s: f64,
    /// The panel's plan unscaled: solves and estimated wall time (s).
    pub full_solves: usize,
    pub full_eta_s: f64,
}

/// What the automatic first pass does ([`first_pass`]).
#[derive(Clone, Debug, PartialEq)]
pub enum FirstPass {
    /// The panel's plan fits the target (or nothing is known about the cost).
    Start(SamplingPlan),
    /// Scaled down to fit, with the note saying so.
    Scaled(SamplingPlan, String),
    /// Even the fewest solves that can earn trust exceed the target: ask.
    Offer(PassOffer),
}

/// The first-pass decision (Task 9 item 4): `plan` on `scene` at `per_solve_s` seconds a solve
/// against [`PASS_TARGET_S`] (5 % slack for the scaled plan's rounding).
pub fn first_pass(scene: &sim::Scene, plan: SamplingPlan, per_solve_s: Option<f64>) -> FirstPass {
    let Some(per) = per_solve_s.filter(|c| *c > 0.0) else { return FirstPass::Start(plan) };
    let full = plan.solves(scene).unwrap_or(0);
    if full as f64 * per <= PASS_TARGET_S {
        return FirstPass::Start(plan);
    }
    let scaled = plan.scaled_to(scene, (PASS_TARGET_S / per).floor() as usize);
    let n = scaled.solves(scene).unwrap_or(full);
    if n as f64 * per <= PASS_TARGET_S * 1.05 {
        let note = format!(
            "sampling plan scaled to {n} solves (≈ {:.0} s at ≈ {per:.2} s per solve here; the panel's plan, {full} solves, would take ≈ {:.0} min) — Sample more adds to it",
            n as f64 * per,
            full as f64 * per / 60.0
        );
        FirstPass::Scaled(scaled, note)
    } else {
        FirstPass::Offer(PassOffer { plan: scaled, solves: n, eta_s: n as f64 * per, per_solve_s: per, full_solves: full, full_eta_s: full as f64 * per })
    }
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
        let scene = self.dy.snapshot().scene;
        let fitted = trainable(&mut self.dy).is_some_and(|s| s.status(&scene, &[]).fitted);
        if !fitted && !sampling_active(&self.dy.snapshot()) {
            self.start_first_pass(plan);
        }
    }

    /// The automatic first pass (Task 9 item 4): the panel's plan if it fits
    /// [`PASS_TARGET_S`] at the expected per-solve cost; else that plan scaled down to fit, with
    /// a note; else — when even the fewest solves that can earn trust take longer — nothing, and
    /// a [`PassOffer`] for the panel's "Start anyway".
    pub(super) fn start_first_pass(&mut self, plan: SamplingPlan) {
        let scene = self.dy.snapshot().scene;
        match first_pass(&scene, plan, self.solve_s) {
            FirstPass::Start(plan) => self.dy.apply(DynCommand::StartSampling(plan)),
            FirstPass::Scaled(plan, note) => {
                self.dy.add_note(note);
                self.dy.apply(DynCommand::StartSampling(plan));
            }
            FirstPass::Offer(o) => self.pass_offer = Some(o),
        }
    }

    /// Active learning (`worker` module doc, "Refits"): the surrogate kept a fallback sample and no pass
    /// is running (a pass refits at its end).
    /// Whether a refit ran.
    pub(super) fn refit_if_pending(&mut self, snap: &DynSnapshot) -> bool {
        if sampling_active(snap) {
            return false;
        }
        let Some(s) = trainable(&mut self.dy) else { return false };
        let st = s.status(&snap.scene, &snap.forces);
        // before the first fit there are no reference energies (the pass computes them and
        // makes the first fit at its end): rule-1 fallback samples wait for it
        if !st.pending_refit || !st.fitted {
            return false;
        }
        if let (false, Some(at)) = (snap.paused, self.last_refit_at) {
            if at.elapsed() < REFIT_INTERVAL && st.samples < self.samples_at_refit + REFIT_SAMPLES {
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

    /// The surrogate's readouts (estimated solver only); `forces` are the snapshot's.
    pub(super) fn model_status(&mut self, scene: &sim::Scene, forces: &[[f64; 3]], step: u64) -> Option<ModelStatus> {
        let (last_refit, refits, autoload_refused, last_gap) =
            (self.last_refit.clone(), self.refits, self.autoload_refused.clone(), self.last_gap);
        let last_seen = self.last_audit_seen.clone();
        let (solve_s, pass_eta_s, pass_offer) = (self.solve_s, self.pass_eta_s, self.pass_offer.clone());
        let s = trainable(&mut self.dy)?;
        let st = s.status(scene, forces);
        // latch the step of a new audit record (its verdict's step less its lag)
        let audit = st.last_audit.clone();
        let new_audit = audit.is_some() && audit != last_seen;
        let status = ModelStatus {
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
            last_refit,
            refits,
            autoload_refused,
            audit_every: st.audit_every,
            draft: st.draft,
            last_audit_step: None, // filled below
            last_gap,
            quarantined: st.quarantined,
            recovery: st.recovery,
            audits_background: st.audits_background,
            audit_in_flight: st.audit_in_flight,
            last_audit: st.last_audit,
            stuck: st.stuck,
            solve_s,
            pass_eta_s,
            pass_offer,
        };
        if new_audit {
            let lag = audit.as_ref().map_or(0, |a| a.unaudited_steps as u64);
            self.last_audit_step = Some(step.saturating_sub(lag));
            self.last_audit_seen = audit;
        }
        Some(ModelStatus { last_audit_step: self.last_audit_step, ..status })
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
                    }
                    self.pass_eta_s = Some(per * p.total.saturating_sub(p.done) as f64);
                }
            }
            None => {
                if self.pass_clock.take().is_none() && teacher_step && self.solve_s.is_none() {
                    // no pass timed yet: the scene's own teacher evaluation, scaled
                    self.solve_s = Some(PASS_SOLVE_PER_COLD * snap.eval.solve_ms as f64 * 1e-3);
                }
                self.pass_eta_s = None;
            }
        }
    }
}
