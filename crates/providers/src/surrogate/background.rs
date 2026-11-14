//! Background refits (2026-11-12, `references.md` "Refit cost at scale", fix 4): while the
//! dynamics play, a refit runs on a copy of the provider on its own thread and the model that
//! stands — with its own verdicts — keeps answering; the result is installed between ticks.
//!
//! # The copy
//!
//! [`SurrogateKs::fork_for_refit`]: everything a refit reads or writes — the store, the
//! accumulator, the fit and its trust state, the quarantine, the failed audits and fences, the
//! fence votes, the audit history, the hysteresis memory, the teach-around counts, the descriptor
//! cache — with a teacher that is never asked ([`NoTeacher`]: a refit solves nothing), no auditor,
//! and the collection as a detached read-only view (`Collection::detached_view`: the basis, the
//! box references and boxes a refit reads; it writes nothing — the copy's `refit` skips the status
//! records and the history flush, which the live provider does after the install, as a synchronous
//! `refit` does at its end).
//!
//! # The install
//!
//! [`SurrogateKs::poll_background_refit`], once the thread is done. The copy refitted the store's
//! first P samples (P = the store length at the start); meanwhile the live provider kept
//! evaluating with the model that stood. What it did since is merged only when it **appended**:
//! - samples after P (their codes, face densities and collection links stay the live provider's;
//!   they are the next refit's, `pending_refit`);
//! - audit-history points (the class verdicts are re-judged against the whole history after the
//!   install — `rejudge_classes`, as when new evidence arrives after a synchronous refit: between
//!   refits a key can leave trust on it, never enter);
//! - failed audits and audit fences after the ones the copy judged (they stay failed: rule 8);
//! - fence votes kept after the ones the copy judged (the next refit judges them);
//! - notes.
//!
//! The hysteresis memory is the copy's, except that a key the live provider's between-refit
//! evidence made **leave** while the copy ran stays out (the re-judge after the install must
//! re-admit it on ENTER, as after a synchronous refit followed by that evidence — the review's
//! finding 2). Of a stored sample (< P) the install takes only the tag a refit may change.
//!
//! Anything else is a **conflict** and the result is discarded, never merged: a model, store or
//! policy that changed (`model_gen`: a forget, a load, an install, an attach, a policy change, a
//! pass's or teaching round's own fit), another reference set — canonical or per box: a box
//! reference's arrival rebases that box's stored energies (the review's finding 1) — or degree cap
//! or body-4 degree (spec 2026-11-13 §4.1: the degree a fit chose is provider state), a quarantine code of
//! a sample < P (a reference arrival's rebase, a fence vote's release or supersession), the failed
//! audits or fences the copy judged, the teach-around counts. The refit is then simply due again
//! (`pending_refit`) — the model that stood keeps standing until a refit installs. So the state
//! after an install is the state a synchronous refit at the start would have left, followed by
//! the live provider's appended events: `a_background_refit_installs_the_synchronous_state` and
//! `appends_during_a_background_refit_survive_the_install` compare them line by line.
//!
//! # The user's codes
//!
//! A **Forget** or **Restore** of a stored sample (spec R2, `SurrogateKs::forget_sample`) while the
//! copy runs is the one quarantine change of a sample < P that does not discard it: the live
//! provider writes the status at once (the collection's record) and records the code on the
//! running refit (`note_user_code`); the conflict test skips those samples, and the install
//! **re-applies** the user's codes over the copy's — so the refit that lands never brings a
//! forgotten sample back (nor re-forgets a restored one), the status record it then syncs is the
//! user's, and a rebuild is due (`pending_refit`): the installed fit still holds what it was
//! started with, as a fit does between a forget and the next refit. Pinned by
//! `a_forget_during_a_background_refit_survives_its_install`.
//!
//! The misfit set M (`trust` module doc) is the copy's too: a sample forgotten while the copy ran
//! and in the copy's M stays in the installed M — its `NearMisfit` refusal, its count against M's
//! limit and its share of the class rule's RMS terms — until the next refit, which `pending_refit`
//! makes due; that refit fits without it and leaves it out of M (a forgotten sample is excluded).
//! Conservative meanwhile: it can only refuse more (its geometry is fenced as forgotten anyway),
//! never answer more. (Final whole-branch review M3, 2026-11-14; documented, not changed.)
//!
//! What a background refit changes is *when* the new model stands (a refit's wall later, as the
//! refit throttle already delays it by up to `REFIT_INTERVAL`), never what any verdict is: the
//! model that answers meanwhile is judged by its own verdicts, exactly as between two refits.

use super::*;

/// The copy's teacher: never asked (a refit solves nothing); named as the live teacher, whose
/// name the provenance checks read.
struct NoTeacher(&'static str);

impl ForceProvider for NoTeacher {
    fn evaluate(&mut self, _scene: &Scene, _should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        Err(ProviderError::Invalid("a background refit's copy has no teacher".into()))
    }
    fn name(&self) -> &'static str {
        self.0
    }
    fn reset(&mut self) {}
}

/// A refit running on its own thread, and what the live provider looked like when it started.
/// Dropped — installed, discarded, or with the provider (a backend switch) — it asks the thread to
/// stop at its next fit (`refit_cancel`) and ends the probe's refit phase.
pub(super) struct BackgroundRefit {
    handle: Option<std::thread::JoinHandle<(Box<SurrogateKs>, Result<FitReport, FitError>)>>,
    snap: Snapshot,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    probe: Option<Arc<sim::ActivityProbe>>,
    /// The user's Forget / Restore codes set while it runs (module doc, "The user's codes"): the
    /// last one per sample, re-applied at the install.
    user_codes: BTreeMap<usize, u8>,
}

impl BackgroundRefit {
    /// Asks the thread to stop at its next fit (tests; `Drop` does it).
    #[cfg(test)]
    pub(super) fn cancel_now(&self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for BackgroundRefit {
    fn drop(&mut self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(p) = &self.probe {
            p.end_refit();
        }
    }
}

/// The live state a background refit started from, as far as the install's conflict test needs it.
pub(super) struct Snapshot {
    /// The store length: the copy refits these samples.
    p: usize,
    model_gen: u64,
    quarantine: Vec<u8>,
    refs: BTreeMap<u8, f64>,
    degree_cap: Option<usize>,
    /// The body-4 degree the standing fit chose (`SurrogateKs::body4_degree`, spec 2026-11-13 §4.1).
    pub(super) body4_degree: usize,
    failed_audits: Vec<FailedAudit>,
    audit_fences: Vec<FailedAudit>,
    vote_pending: Vec<(usize, Vec<usize>)>,
    teach_rounds: BTreeMap<usize, u32>,
    audit_burst: usize,
    /// The audit history's stamp (`shared::history_stamp`).
    history: (u64, u64),
    /// The collection's references per (Z, box): a box reference's arrival rebases the stored
    /// energies of that box (`rebase_collection`) without moving the canonical `refs`.
    box_refs: Option<BTreeMap<(u8, usize), f64>>,
    /// The hysteresis memory (a key the live provider's between-refit evidence made leave stays
    /// out at the install).
    class_memory: BTreeMap<surrogate::EnvKey, (bool, Option<u64>)>,
}

/// What [`SurrogateKs::poll_background_refit`] found.
pub enum BackgroundPoll {
    /// No background refit.
    Idle,
    /// Still running.
    Running,
    /// Installed: the refit's own outcome (an error leaves what a synchronous refit's error leaves).
    Installed(Result<FitReport, FitError>),
    /// Discarded on a conflict (module doc), with the reason; the refit is due again.
    Discarded(String),
}

impl SurrogateKs {
    /// A copy of the provider for a refit on another thread (module doc, "The copy").
    pub(super) fn fork_for_refit(&self) -> SurrogateKs {
        let mut f = SurrogateKs::new(Box::new(NoTeacher(self.teacher.name())), self.policy.clone());
        f.name = self.name;
        f.teacher_tol = self.teacher_tol;
        f.store = self.store.clone();
        f.accumulator = self.accumulator.clone();
        f.accumulated = self.accumulated;
        f.refs = self.refs.clone();
        f.legacy_refs = self.legacy_refs.clone();
        f.refs_dirty = self.refs_dirty;
        f.species_dirty = self.species_dirty;
        f.model = self.model.clone();
        f.report = self.report.clone();
        f.envelope = self.envelope.clone();
        f.null_space = self.null_space.clone();
        f.active_set = self.active_set.clone();
        f.training_residuals = self.training_residuals;
        f.misfit = self.misfit.clone();
        f.class_loo = self.class_loo.clone();
        f.class_memory = self.class_memory.clone();
        f.class_judge = self.class_judge.clone();
        f.deferred_trust = self.deferred_trust;
        f.loo_ratios = self.loo_ratios.clone();
        f.audit_failed = self.audit_failed;
        f.audit_force = self.audit_force;
        f.failed_audits = self.failed_audits.clone();
        f.audit_fences = self.audit_fences.clone();
        f.audit_burst = self.audit_burst;
        f.audit_atoms = self.audit_atoms;
        f.audit_origin = self.audit_origin;
        f.epoch = self.epoch;
        f.pending_refit = self.pending_refit;
        f.quarantine = self.quarantine.clone();
        f.lone_codes = self.lone_codes.clone();
        f.degree_cap = self.degree_cap;
        f.body4_degree = self.body4_degree;
        // the descriptor cache is not copied: the copy's active set fills its own, off the step
        // path, and the install takes it
        f.recovery = self.recovery.clone();
        f.audit_history = self.audit_history.clone();
        f.fits = self.fits;
        f.model_gen = self.model_gen;
        f.fit_members = self.fit_members.clone();
        f.fence_votes = self.fence_votes.clone();
        f.fence_kept = self.fence_kept.clone();
        f.vote_pending = self.vote_pending.clone();
        f.class_worst = self.class_worst.clone();
        f.class_blockers = self.class_blockers.clone();
        f.teach_rounds = self.teach_rounds.clone();
        f.face = self.face.clone();
        f.last_fit_error = self.last_fit_error.clone();
        f.collection_r_cut = self.collection_r_cut;
        f.shared = self.shared.as_ref().map(|sh| sh.detached());
        // the probe's refit phase is the live provider's to begin and end (`BackgroundRefit`)
        f.refit_fork = true;
        #[cfg(test)]
        {
            f.body4_off = self.body4_off;
            f.body4_force = self.body4_force;
            f.eager_trust_passes = self.eager_trust_passes;
            f.skip_recovery = self.skip_recovery;
        }
        f
    }

    /// Starts a refit on a copy on its own thread (module doc); `false` when one is running already,
    /// a sampling pass runs, or there is nothing to fit. The live provider keeps answering with the
    /// model that stands until [`poll_background_refit`](Self::poll_background_refit) installs it.
    pub fn begin_background_refit(&mut self) -> bool {
        if self.background.is_some() || self.pass.is_some() || self.model.is_none() || self.store.as_ref().is_none_or(|s| s.is_empty()) {
            return false;
        }
        let snap = self.snapshot();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut fork = self.fork_for_refit();
        fork.refit_cancel = Some(cancel.clone());
        let handle = std::thread::Builder::new()
            .name("surrogate-refit".into())
            .spawn(move || {
                let r = SurrogateKs::refit(&mut fork);
                (Box::new(fork), r)
            })
            .expect("spawn the background refit");
        let probe = self.probe.clone();
        if let Some(p) = &probe {
            p.begin_refit(self.samples());
        }
        self.background = Some(BackgroundRefit { handle: Some(handle), snap, cancel, probe, user_codes: BTreeMap::new() });
        true
    }

    /// The live state a background refit starting now would start from (its [`Snapshot`]).
    pub(super) fn snapshot(&self) -> Snapshot {
        Snapshot {
            p: self.samples(),
            model_gen: self.model_gen,
            quarantine: self.quarantine.clone(),
            refs: self.refs.clone(),
            degree_cap: self.degree_cap,
            body4_degree: self.body4_degree,
            failed_audits: self.failed_audits.clone(),
            audit_fences: self.audit_fences.clone(),
            vote_pending: self.vote_pending.clone(),
            teach_rounds: self.teach_rounds.clone(),
            audit_burst: self.audit_burst,
            history: shared::history_stamp(&self.audit_history),
            box_refs: self.shared.as_ref().map(|sh| sh.box_refs.clone()),
            class_memory: self.class_memory.clone(),
        }
    }

    /// Whether a background refit is running.
    pub fn background_refit_running(&self) -> bool {
        self.background.is_some()
    }

    /// The user's code `code` of sample `i` (Forget, Restore), for a running background refit's
    /// install (module doc, "The user's codes"); a no-op without one.
    pub(super) fn note_user_code(&mut self, i: usize, code: u8) {
        if let Some(b) = self.background.as_mut() {
            b.user_codes.insert(i, code);
        }
    }

    /// Installs a finished background refit, or discards it on a conflict (module doc); `Running`
    /// while it runs, `Idle` without one.
    pub fn poll_background_refit(&mut self) -> BackgroundPoll {
        match self.background.as_ref() {
            None => return BackgroundPoll::Idle,
            Some(b) if !b.handle.as_ref().is_some_and(|h| h.is_finished()) => return BackgroundPoll::Running,
            Some(_) => {}
        }
        let mut done = self.background.take().expect("checked");
        let handle = done.handle.take().expect("a running refit has its thread");
        let snap = std::mem::replace(&mut done.snap, Snapshot::empty());
        let user = std::mem::take(&mut done.user_codes);
        // `done` drops at the end: the probe's refit phase ends whatever the outcome
        let (fork, result) = match handle.join() {
            Ok(x) => x,
            Err(_) => {
                self.pending_refit = true;
                return BackgroundPoll::Discarded("the background refit panicked; it is due again".into());
            }
        };
        if let Some(why) = self.background_conflict(&snap, &user) {
            self.pending_refit = true;
            return BackgroundPoll::Discarded(why);
        }
        self.install_fork(*fork, &snap, result.is_ok(), &user);
        BackgroundPoll::Installed(result)
    }

    /// Waits for a running background refit and installs or discards it (tests; an operation that
    /// must see the refit's outcome).
    pub fn finish_background_refit(&mut self) -> BackgroundPoll {
        if let Some(h) = self.background.as_ref().and_then(|b| b.handle.as_ref()) {
            while !h.is_finished() {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        self.poll_background_refit()
    }

    /// Why the live provider's changes since `snap` are not appends (module doc), `None` when they
    /// are. The samples whose code the user set while it ran (`user`) are not a conflict (module
    /// doc, "The user's codes").
    pub(super) fn background_conflict(&self, snap: &Snapshot, user: &BTreeMap<usize, u8>) -> Option<String> {
        let why = |s: &str| Some(format!("background refit discarded: {s} changed while it ran — it is due again"));
        if self.model_gen != snap.model_gen {
            return why("the model, its samples or the policy");
        }
        if self.samples() < snap.p {
            return why("the samples");
        }
        if self.refs != snap.refs || self.shared.as_ref().map(|sh| &sh.box_refs) != snap.box_refs.as_ref() {
            return why("the isolated-atom references");
        }
        if self.degree_cap != snap.degree_cap {
            return why("the degree cap");
        }
        if self.body4_degree != snap.body4_degree {
            return why("the body-4 degree");
        }
        let code = |q: &[u8], i: usize| q.get(i).copied().unwrap_or(Q_NONE);
        if (0..snap.p).any(|i| !user.contains_key(&i) && code(&self.quarantine, i) != code(&snap.quarantine, i)) {
            return why("a stored sample's quarantine code");
        }
        if !self.failed_audits.starts_with(&snap.failed_audits) || !self.audit_fences.starts_with(&snap.audit_fences) {
            return why("the failed audits");
        }
        if !self.vote_pending.starts_with(&snap.vote_pending) {
            return why("the fence votes");
        }
        if self.teach_rounds != snap.teach_rounds {
            return why("the teach-around rounds");
        }
        None
    }

    /// The install (module doc): the copy's fit and trust state for the first P samples, the live
    /// provider's appends after them; then what a synchronous refit does at its end — the jump
    /// booked at the current geometry, the collection's status records and history — and the class
    /// verdicts re-judged against the whole history.
    fn install_fork(&mut self, mut f: SurrogateKs, snap: &Snapshot, ok: bool, user: &BTreeMap<usize, u8>) {
        let was_trusted = self.trusted();
        let was_model = self.last.as_ref().is_some_and(|l| l.on_model);
        let p = snap.p;
        // the store: what a refit changes in a stored sample is its tag (a promoted held-out guess
        // becomes `Audit`, `judge_held_out`); everything else stays the live provider's
        if let (Some(live), Some(fs)) = (self.store.as_mut(), f.store.as_ref()) {
            for (l, c) in live.samples.iter_mut().zip(&fs.samples).take(p) {
                l.tag = c.tag;
            }
        }
        // the quarantine: the copy's codes for the first P, the live ones after
        let mut q = std::mem::take(&mut f.quarantine);
        if self.quarantine.len() > p {
            q.resize(p, Q_NONE);
            q.extend(self.quarantine.iter().skip(p).copied());
        }
        self.quarantine = q;
        // the user's Forget / Restore while it ran, over the copy's codes (module doc, "The user's
        // codes"): the copy fitted with what it started from; the next refit takes the user's
        let user_moved = user.iter().any(|(&i, &c)| i < p && self.quarantine.get(i).copied().unwrap_or(Q_NONE) != c);
        for (&i, &c) in user.iter().filter(|(&i, _)| i < p) {
            self.quarantine.resize(self.quarantine.len().max(i + 1), Q_NONE);
            self.quarantine[i] = c;
        }
        // the fit and its trust state
        self.accumulator = f.accumulator.take();
        self.accumulated = f.accumulated;
        self.refs_dirty = f.refs_dirty;
        self.species_dirty = f.species_dirty || user_moved || self.store.as_ref().is_some_and(|st| {
            let basis = self.accumulator.as_ref().map(|a| &a.basis);
            st.samples[p.min(st.len())..].iter().any(|s| basis.is_none_or(|b| b.species_indices(&s.z).is_err()))
        });
        self.model = f.model.take();
        self.report = f.report.take();
        self.envelope = f.envelope.take();
        self.null_space = f.null_space.take();
        self.active_set = f.active_set.take();
        self.training_residuals = f.training_residuals;
        self.misfit = f.misfit.take();
        self.class_loo = std::mem::take(&mut f.class_loo);
        // the hysteresis memory: the copy's, except that a key the live provider's between-refit
        // evidence made leave since the start stays out (a synchronous refit followed by that
        // evidence would have it out; the install's re-judge below must not re-admit it on LEAVE)
        let left: Vec<surrogate::EnvKey> = snap.class_memory.iter().filter(|(k, &(was, _))| was && self.class_memory.get(*k).is_none_or(|&(now, _)| !now)).map(|(k, _)| k.clone()).collect();
        self.class_memory = std::mem::take(&mut f.class_memory);
        for k in &left {
            if let Some(m) = self.class_memory.get_mut(k) {
                *m = (false, None);
            }
        }
        self.class_judge = f.class_judge.take();
        self.class_worst = std::mem::take(&mut f.class_worst);
        self.class_blockers = std::mem::take(&mut f.class_blockers);
        self.deferred_trust = f.deferred_trust;
        // the copy's leave-one-out ratios, keyed to its fit (`model_gen` below is the copy's: the
        // conflict test saw no live change of it), re-checked against the members at every read
        self.loo_ratios = f.loo_ratios.take();
        self.degree_cap = f.degree_cap;
        self.body4_degree = f.body4_degree;
        self.fit_members = std::mem::take(&mut f.fit_members);
        self.fits = f.fits;
        self.model_gen = self.model_gen.max(f.model_gen);
        self.recovery = f.recovery.take();
        self.last_fit_error = f.last_fit_error.take();
        self.teach_rounds = std::mem::take(&mut f.teach_rounds);
        // the copy's descriptor cache when it filled one under the new basis (its active set did),
        // else the live one (`ensure_desc_cache` re-checks the basis and extends it)
        if f.desc_basis.is_some() && f.desc_cache.len() >= p.min(self.desc_cache.len()) {
            self.desc_cache = std::mem::take(&mut f.desc_cache);
            self.desc_basis = f.desc_basis.take();
        }
        self.curve_cache = None;
        self.fit_floor_cache.lock().unwrap().clear();
        // the failed audits and fences the copy judged, then the live ones added since
        let added_audits = self.failed_audits.split_off(snap.failed_audits.len());
        let added_fences = self.audit_fences.split_off(snap.audit_fences.len());
        self.failed_audits = std::mem::take(&mut f.failed_audits);
        self.failed_audits.extend(added_audits);
        self.audit_fences = std::mem::take(&mut f.audit_fences);
        self.audit_fences.extend(added_fences);
        if f.audit_burst != snap.audit_burst {
            self.audit_burst = f.audit_burst;
        }
        // the votes the copy judged are gone, the live ones kept since wait for the next refit
        let added_votes = self.vote_pending.split_off(snap.vote_pending.len());
        self.vote_pending = added_votes;
        for (k, v) in std::mem::take(&mut f.fence_kept) {
            let e = self.fence_kept.entry(k).or_insert(v);
            *e = (*e).max(v);
        }
        self.fence_votes.clear();
        self.lift_votes.clear();
        // the refit's notes after the live ones
        self.user_notes.extend(std::mem::take(&mut f.user_notes));
        // what `fit_store` and `refit` do once the fit stands
        self.consults = 0;
        self.last_variance = None;
        self.pending_refit = self.samples() > p || f.pending_refit || user_moved;
        // a refit that ended in an error re-judges nothing and moves no surface (`refit_inner`): the
        // failed audits stay as they stood
        if ok {
            self.sync_audit_failed();
            if let Err(e) = self.surface_moved(was_trusted, was_model) {
                self.user_notes.push(format!("surrogate: the refit jump could not be booked ({e})"));
            }
        }
        self.sync_collection_status();
        self.flush_collection();
        // evidence that arrived while the copy ran is read as it would have been after a
        // synchronous refit: between refits (`rejudge_classes` — it can make a key leave, never
        // enter, until the next refit's own judgement)
        if shared::history_stamp(&self.audit_history) != snap.history || !left.is_empty() {
            self.rejudge_classes();
        }
    }
}

impl Snapshot {
    /// A placeholder for a snapshot moved out of a finished refit.
    fn empty() -> Snapshot {
        Snapshot {
            p: 0,
            model_gen: 0,
            quarantine: Vec::new(),
            refs: BTreeMap::new(),
            degree_cap: None,
            body4_degree: surrogate::BODY4_DEGREE_START,
            failed_audits: Vec::new(),
            audit_fences: Vec::new(),
            vote_pending: Vec::new(),
            teach_rounds: BTreeMap::new(),
            audit_burst: 0,
            history: (0, 0),
            box_refs: None,
            class_memory: BTreeMap::new(),
        }
    }
}
