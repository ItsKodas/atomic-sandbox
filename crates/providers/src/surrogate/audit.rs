//! The audits (the `surrogate` module doc, rule 8 "Audit") and the background auditor (ruling S14)
//! with its side densities (the module doc, "Side densities"): `AuditVerdict`/`AuditRecord` (defined
//! in `sim` since M6, for `TrainableStatus`), the auditor thread, dispatch and judgement, the
//! display and isolated-atom requests. Split out of `surrogate_ks.rs` in M6 Task 1; the narrative
//! stays in the parent module doc.

use super::*;

pub use sim::{AuditRecord, AuditVerdict};

/// What a background job is for (`surrogate` module doc, "Audit" and "Side densities").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum JobKind {
    Audit,
    /// A display-only solve at the current geometry.
    Refresh,
    /// The isolated atom of this Z on the scene's cell.
    Atom(u8),
    /// The isolated-atom reference of this Z: `sim::references::solve_atom`'s whole multiplicity
    /// scan on the scene's cell and σ (M6 Task 3 review I2: never on the stepping teacher).
    Reference(u8),
}

/// One asynchronous job (ruling S14): the geometry, its provenance, the auditor generation it
/// belongs to (a Forget or an installed sidecar starts a new one), `answered` at dispatch, what it
/// is for, its side-job number (0 for an atom) and the orbital view its teacher shows (the
/// surrogate's, with `force_every_step` set only for a refresh the driver forces — paused — never
/// for an audit: the module doc, "Orbitals", Task 4 review I1), and the driver's display interrupt
/// its teacher's display work polls (Task 4 re-review N1).
pub(super) struct AuditJob {
    pub(super) scene: Scene,
    pub(super) prov: Provenance,
    pub(super) generation: u64,
    pub(super) answered: usize,
    pub(super) kind: JobKind,
    pub(super) id: u64,
    pub(super) view: OrbitalViewRequest,
    pub(super) interrupt: Option<sim::DisplayInterrupt>,
    /// The display includes the partial core ρ_pc (M7; `SurrogateKs::display_core`).
    pub(super) core: bool,
}

/// What the auditor's teacher returned for an audit or a display job.
pub(super) struct Solved {
    pub(super) energy: f64,
    pub(super) forces: Vec<[f64; 3]>,
    pub(super) face: Option<f64>,
    pub(super) display: Option<Display>,
    pub(super) orbitals: Option<OrbitalDisplay>,
    /// Why this answer must not be trained on (`EvalStats::untrainable`), if it must not.
    pub(super) unfit: Option<&'static str>,
}

/// The auditor's answer: what the teacher solved, or its error; for a `Reference` job the scan's
/// outcome is in `reference` (and `result` is unused). `scf_steps`: the solve's SCF steps as its
/// teacher reported them (`None` for an error or a reference scan).
pub(super) struct AuditAnswer {
    pub(super) job: AuditJob,
    pub(super) result: Result<Solved, String>,
    pub(super) reference: Option<Result<AtomReference, ProviderError>>,
    pub(super) ms: f64,
    pub(super) scf_steps: Option<usize>,
}

/// The background auditor (ruling S14, module doc "Audit"): a second teacher instance on its own
/// thread; at most one job in flight. A job of an older generation is abandoned at the teacher's
/// next abort check (GPU: the next SCF step) and its answer discarded.
pub(super) struct Auditor {
    pub(super) jobs: Option<mpsc::Sender<AuditJob>>,
    pub(super) answers: mpsc::Receiver<AuditAnswer>,
    pub(super) generation: Arc<AtomicU64>,
    pub(super) handle: Option<std::thread::JoinHandle<()>>,
    /// The job in flight.
    pub(super) busy: Option<JobKind>,
    /// Multiplicities done by the `Reference` job in flight (reset when one starts).
    pub(super) ref_done: Arc<std::sync::atomic::AtomicUsize>,
}

impl Auditor {
    fn spawn(mut teacher: Box<dyn ForceProvider>) -> Auditor {
        let (jtx, jrx) = mpsc::channel::<AuditJob>();
        let (atx, arx) = mpsc::channel::<AuditAnswer>();
        let generation = Arc::new(AtomicU64::new(0));
        let current = generation.clone();
        let ref_done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let done_count = ref_done.clone();
        let handle = std::thread::Builder::new()
            .name("surrogate-audit".into())
            .spawn(move || {
                let mut last_atom = false;
                // the orbital view the teacher has (module doc, "Side densities")
                let mut view = OrbitalViewRequest::default();
                let mut core = false;
                while let Ok(job) = jrx.recv() {
                    // the audits are thousands of steps apart: no extrapolation through the
                    // previous one's geometry (the density is kept as a warm start) — but an
                    // isolated atom's density is no warm start for a molecule, nor the reverse
                    let atom = matches!(job.kind, JobKind::Atom(_) | JobKind::Reference(_));
                    if atom || last_atom {
                        teacher.reset();
                    } else {
                        teacher.discontinuity();
                    }
                    last_atom = atom;
                    // the one orbital producer: on for audits and refreshes, off for an atom
                    let want = if atom { OrbitalViewRequest::default() } else { job.view.clone() };
                    if want != view {
                        teacher.set_orbital_view(want.clone());
                        view = want;
                    }
                    if job.core != core {
                        teacher.set_display_partial_core(job.core);
                        core = job.core;
                    }
                    // the driver's display interrupt (re-review N1): a forced refresh's unoccupied
                    // eigensolve yields this job slot to the user's next command
                    if let Some(i) = &job.interrupt {
                        teacher.set_display_interrupt(i.clone());
                    }
                    let t = Instant::now();
                    let g = job.generation;
                    let mut scf_steps = None;
                    let (result, reference) = match job.kind {
                        // the multiplicity scan (each solve cold, the teacher reset after it)
                        JobKind::Reference(z) => {
                            let mass = job.scene.nuclei.first().map_or(0.0, |n| n.mass);
                            done_count.store(0, Ordering::Relaxed);
                            let r = sim::references::solve_atom_reporting(teacher.as_mut(), &job.scene.cell, job.scene.sigma, job.scene.ionic, z, mass, &mut || current.load(Ordering::Relaxed) != g, &mut |k| done_count.store(k, Ordering::Relaxed));
                            (Err(String::new()), Some(r))
                        }
                        _ => (
                            teacher
                                .evaluate(&job.scene, &mut || current.load(Ordering::Relaxed) != g)
                                .map(|r| {
                                    scf_steps = Some(r.stats.scf_steps);
                                    Solved { energy: r.energy, forces: r.forces, face: r.stats.face_density, unfit: r.stats.untrainable(), display: r.display, orbitals: r.orbitals }
                                })
                                .map_err(|e| e.to_string()),
                            None,
                        ),
                    };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    if atx.send(AuditAnswer { job, result, reference, ms, scf_steps }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the surrogate audit thread");
        Auditor { jobs: Some(jtx), answers: arx, generation, handle: Some(handle), busy: None, ref_done }
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Abandons the job in flight (its answer is discarded on arrival).
    fn new_generation(&mut self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for Auditor {
    /// Abandons the job in flight and joins the thread (≤ one SCF step on the GPU).
    fn drop(&mut self) {
        self.jobs = None;
        self.generation.store(u64::MAX, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl SurrogateKs {
    /// Runs the audits (rule 8) on `teacher`, a second instance of this provider's teacher, on a
    /// background thread (ruling S14; module doc, "Audit"): the model keeps answering while the
    /// audit solves. Without one the audits are synchronous — the CPU teacher's case: a second
    /// `CpuKs` would compete with the first for the same cores, and a CPU solve cannot be
    /// abandoned. Panics if `teacher` is not the same kind of teacher (its `name`).
    pub fn with_auditor(mut self, teacher: Box<dyn ForceProvider>) -> SurrogateKs {
        assert_eq!(teacher.name(), self.teacher.name(), "the auditor must be a second instance of the teacher");
        self.auditor = Some(Auditor::spawn(teacher));
        self
    }

    /// The audits run on a background auditor ([`with_auditor`](Self::with_auditor)).
    pub fn audits_asynchronous(&self) -> bool {
        self.auditor.is_some()
    }

    /// An asynchronous audit is in flight (or its answer arrived and waits for the next evaluate).
    pub fn audit_in_flight(&self) -> bool {
        self.held_audit.is_some() || self.auditor.as_ref().is_some_and(|a| a.busy == Some(JobKind::Audit))
    }

    /// The background auditor exists and has no job in flight (`surrogate` module doc, "Side densities").
    pub fn auditor_idle(&self) -> bool {
        self.auditor.as_ref().is_some_and(|a| a.busy.is_none())
    }

    /// Asks the background auditor for a display-only Kohn-Sham solve at `scene` (`surrogate` module doc,
    /// "Side densities"); its density arrives as a `SideKind::Refresh` [`SideDisplay`]. Sent at
    /// once when the auditor is idle, else queued ahead of the next audit (a newer request
    /// replaces a queued one). False without an auditor. Never judged, never a sample.
    pub fn request_display(&mut self, scene: &Scene) -> bool {
        self.poll_side_jobs();
        // an empty scene has no density to show (`sim::provider`, "Nothing to solve")
        if self.auditor.is_none() || scene.nuclei.is_empty() {
            return false;
        }
        let prov = self.scene_provenance(scene);
        self.side_jobs += 1;
        self.side_queue.retain(|j| j.2 != JobKind::Refresh);
        self.side_queue.push_back((scene.clone(), prov, JobKind::Refresh, self.side_jobs));
        self.pump_side_queue();
        true
    }

    /// Asks the background auditor for the isolated atom Z on `scene`'s cell (σ kept, neutral,
    /// lowest multiplicity, at the origin), queued like a refresh; its display arrives through
    /// [`take_atom_densities`](Self::take_atom_densities). False without an auditor.
    pub fn request_atom_density(&mut self, z: u8, scene: &Scene) -> bool {
        self.poll_side_jobs();
        if self.auditor.is_none() {
            return false;
        }
        let atom = Scene {
            nuclei: vec![sim::Nucleus::new(z, [0.0; 3])],
            cell: scene.cell.clone(),
            charge: 0,
            multiplicity: None,
            sigma: scene.sigma,
            external_field: [0.0; 3],
            ionic: sim::IonicPolicy::None,
        };
        let prov = self.scene_provenance(&atom);
        self.side_queue.retain(|j| j.2 != JobKind::Atom(z));
        self.side_queue.push_back((atom, prov, JobKind::Atom(z), 0));
        self.pump_side_queue();
        true
    }

    /// Asks the background auditor for species `z`'s isolated-atom reference
    /// (`sim::references::solve_atom` on `scene`'s cell and σ, the atom of mass `mass`; M6 Task 3
    /// review I2), queued like a refresh; its outcome arrives through
    /// [`take_atom_references`](Self::take_atom_references) — exactly once per accepted request
    /// (a repeated request while one is queued or running is the same request). False without an
    /// auditor.
    pub fn request_atom_reference(&mut self, z: u8, mass: f64, scene: &Scene) -> bool {
        self.poll_side_jobs();
        if self.auditor.is_none() {
            return false;
        }
        let pending = self.side_queue.iter().any(|j| j.2 == JobKind::Reference(z)) || self.auditor.as_ref().is_some_and(|a| a.busy == Some(JobKind::Reference(z)));
        if pending {
            return true;
        }
        let mut nucleus = sim::Nucleus::new(z, [0.0; 3]);
        nucleus.mass = mass;
        let atom = Scene { nuclei: vec![nucleus], cell: scene.cell.clone(), charge: 0, multiplicity: None, sigma: scene.sigma, external_field: [0.0; 3], ionic: scene.ionic };
        let prov = self.scene_provenance(&atom);
        self.side_queue.push_back((atom, prov, JobKind::Reference(z), 0));
        self.pump_side_queue();
        true
    }

    /// Multiplicities done by the reference scan in flight on the auditor (`None` when none is).
    pub fn atom_reference_done(&self) -> Option<usize> {
        let a = self.auditor.as_ref()?;
        matches!(a.busy, Some(JobKind::Reference(_))).then(|| a.ref_done.load(Ordering::Relaxed))
    }

    /// The isolated-atom references that came back since the last call: (Z, the scan's outcome).
    pub fn take_atom_references(&mut self) -> Vec<(u8, Result<AtomReference, ProviderError>)> {
        self.poll_side_jobs();
        std::mem::take(&mut self.atom_references)
    }

    /// Answers every reference request still queued — and `busy`'s, when it is one — with `why`
    /// (the auditor's thread is gone): each accepted request is answered exactly once.
    pub(super) fn fail_references(&mut self, busy: Option<JobKind>, why: &str) {
        let mut zs: Vec<u8> = self.side_queue.iter().filter_map(|j| if let JobKind::Reference(z) = j.2 { Some(z) } else { None }).collect();
        if let Some(JobKind::Reference(z)) = busy {
            zs.insert(0, z);
        }
        self.side_queue.retain(|j| !matches!(j.2, JobKind::Reference(_)));
        for z in zs {
            self.atom_references.push((z, Err(ProviderError::Invalid(why.to_string()))));
        }
    }

    /// Sends the next queued side job if the auditor is idle (`surrogate` module doc, "Side densities": a
    /// queued request goes ahead of an audit that fell due meanwhile).
    pub(super) fn pump_side_queue(&mut self) {
        if !self.auditor_idle() {
            return;
        }
        if let Some((scene, prov, kind, id)) = self.side_queue.pop_front() {
            if !self.send_job(scene, prov, kind, id) {
                // the thread is gone: the references asked for are answered, the rest dropped
                if let JobKind::Reference(z) = kind {
                    self.atom_references.push((z, Err(ProviderError::Invalid("the background teacher is gone".into()))));
                }
                self.fail_references(None, "the background teacher is gone");
                self.side_queue.clear();
            }
        }
    }

    /// The isolated-atom densities that arrived since the last call: (Z, its display).
    pub fn take_atom_densities(&mut self) -> Vec<(u8, Display)> {
        self.poll_side_jobs();
        std::mem::take(&mut self.atom_densities)
    }

    /// Outside `evaluate`: takes whatever answer arrived — a refresh's or an atom's is handled at
    /// once; an audit's is **held** unjudged (its verdict is taken at the first evaluate after it
    /// arrives, by the model current then: module doc, "Audit") — and hands the freed auditor the
    /// next queued side job, so a request made while paused is not stuck behind an audit
    /// (review M7).
    pub(super) fn poll_side_jobs(&mut self) {
        // (one job is in flight at a time, and no audit is dispatched outside `evaluate`, so a
        // held audit answer is never followed by another before it is judged)
        if let Some(ans) = self.receive_answer() {
            if ans.job.kind == JobKind::Audit {
                self.held_audit = Some(ans);
            } else {
                self.handle_answer(ans);
            }
        }
        self.pump_side_queue();
    }

    /// Sends a job to the idle auditor; false when there is none, it is busy, or its thread is
    /// gone (then the audits run in the step from now on).
    pub(super) fn send_job(&mut self, scene: Scene, prov: Provenance, kind: JobKind, id: u64) -> bool {
        let answered = self.answered;
        let Some(a) = self.auditor.as_mut() else { return false };
        if a.busy.is_some() {
            return false;
        }
        // the rate flag (module doc, "Orbitals"; Task 4 review I1): a refresh is forced only when
        // the driver forces the view (paused), an audit never, so the auditor teacher's
        // `UnoccupiedCadence` budgets the unoccupied eigensolve against its own SCF wall
        let force = kind == JobKind::Refresh && self.orbital_view.force_every_step;
        let view = OrbitalViewRequest { force_every_step: force, ..self.orbital_view.clone() };
        let job = AuditJob { scene, prov, generation: a.generation(), answered, kind, id, view, interrupt: self.display_interrupt.clone(), core: self.display_core };
        match a.jobs.as_ref().map(|tx| tx.send(job)) {
            Some(Ok(())) => {
                a.busy = Some(kind);
                true
            }
            _ => {
                self.auditor = None;
                self.audit_due = false;
                false
            }
        }
    }

    /// The last audit's outcome.
    pub fn last_audit(&self) -> Option<&AuditRecord> {
        self.last_audit.as_ref()
    }

    /// The last audit judged the current fit: `false` when no audit ran since the last refit
    /// (`last_audit` is then an earlier fit's, or `None` when the model was never audited).
    /// Activity Task 3 review I1: the checklist's audit line must not read an earlier fit's
    /// "passed" as a check of this one. A readout only.
    pub fn last_audit_this_fit(&self) -> bool {
        self.last_audit.is_some() && self.last_audit_fit == self.fits
    }

    /// A model forgotten or replaced: the audit in flight is abandoned, nothing is due or booked.
    pub(super) fn reset_audits(&mut self) {
        if let Some(a) = self.auditor.as_mut() {
            a.new_generation();
        }
        self.audit_due = false;
        self.audit_flip = false;
        self.last_audit = None;
        self.streak = 0;
        self.streak_reason = None;
        self.streak_gap = None;
        self.streak_box = 0;
        self.streak_box_face = 0.0;
    }

    /// Hands an audit that fell due to the background auditor (`surrogate` module doc, "Audit"); `false`
    /// when there is none (the audit then runs in the step). A busy auditor defers it to the
    /// next answered step.
    pub(super) fn dispatch_audit(&mut self, scene: &Scene, prov: &Provenance) -> bool {
        let every = self.policy.audit_every;
        let Some(a) = self.auditor.as_ref() else { return false };
        if a.busy.is_some() {
            self.audit_due = true;
            return true;
        }
        self.side_jobs += 1;
        if self.send_job(scene.clone(), prov.clone(), JobKind::Audit, self.side_jobs) {
            self.audit_due = false;
            self.pending_notes.push(format!("surrogate audit dispatched to the background teacher (every {every} answered steps; the model answers meanwhile)"));
            true
        } else {
            // the thread is gone (`send_job` dropped it): audit in the step from now on
            self.side_jobs -= 1;
            false
        }
    }

    /// Inside `evaluate`: handles a held audit answer, else the one that arrived
    /// ([`handle_answer`](Self::handle_answer)), then hands the freed auditor the next queued side
    /// job.
    pub(super) fn poll_audit(&mut self) {
        if let Some(ans) = self.held_audit.take().or_else(|| self.receive_answer()) {
            self.handle_answer(ans);
        }
        self.pump_side_queue();
    }

    /// The background auditor's answer, if one arrived (the auditor is then free).
    pub(super) fn receive_answer(&mut self) -> Option<AuditAnswer> {
        let a = self.auditor.as_mut()?;
        match a.answers.try_recv() {
            Ok(ans) => {
                a.busy = None;
                Some(ans)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                let busy = a.busy;
                self.auditor = None;
                self.audit_due = false;
                self.fail_references(busy, "the background teacher stopped");
                self.pending_notes.push("surrogate: the background audit thread stopped — audits run in the step from now on".into());
                None
            }
        }
    }

    /// One answer of the background auditor: a side density is kept for the app, an audit is
    /// judged with the current model (`surrogate` module doc, "Audit"): the sample is admitted like any
    /// teacher answer, the gap recorded; a failed audit untrusts the model and books the flip at
    /// the next evaluation. An answer of an older generation is discarded (a cancelled solve: no
    /// activity event, not counted); a failed audit solve leaves a `Failed` event (plan review
    /// S11).
    pub(super) fn handle_answer(&mut self, mut ans: AuditAnswer) {
        // a reference is answered whatever the generation (its requester waits for it); one
        // abandoned by a Forget or an installed model comes back `Cancelled`
        if let (JobKind::Reference(z), Some(r)) = (ans.job.kind, ans.reference.take()) {
            return self.atom_references.push((z, r));
        }
        let Some(a) = self.auditor.as_ref() else { return };
        if ans.job.generation != a.generation() {
            return; // the model it was for was forgotten or replaced
        }
        let positions = || ans.job.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<[f64; 3]>>();
        match (ans.job.kind, ans.result) {
            (JobKind::Atom(z), Ok(Solved { display: Some(d), .. })) => return self.atom_densities.push((z, d)),
            (JobKind::Refresh, Ok(Solved { display: Some(d), orbitals, .. })) => {
                self.side_display = Some(SideDisplay { display: d, orbitals, kind: SideKind::Refresh, job: ans.job.id, positions: positions() });
                return;
            }
            (JobKind::Reference(z), _) => {
                return self.atom_references.push((z, Err(ProviderError::Invalid("the background teacher returned no reference".into()))));
            }
            (JobKind::Atom(_) | JobKind::Refresh, Ok(_)) => {
                return self.pending_notes.push("surrogate: the background teacher returned no density (its display is off)".into());
            }
            (JobKind::Atom(z), Err(e)) => {
                return self.pending_notes.push(format!("surrogate: the isolated-atom density of Z = {z} failed on the background teacher ({e})"));
            }
            (JobKind::Refresh, Err(e)) => {
                return self.pending_notes.push(format!("surrogate: the density refresh failed on the background teacher ({e})"));
            }
            (JobKind::Audit, Err(e)) => {
                let note = format!("surrogate audit: the background teacher failed ({e}) — not judged");
                let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Failed);
                ev.detail = note.clone();
                ev.n_atoms = ans.job.scene.nuclei.len();
                ev.solve_ms = ans.ms;
                ev.audit_lag = Some(self.answered.saturating_sub(ans.job.answered));
                self.push_event(ev);
                return self.pending_notes.push(note);
            }
            (JobKind::Audit, Ok(Solved { energy, forces, face, display, orbitals, unfit })) => {
                if let Some(d) = display {
                    self.side_display = Some(SideDisplay { display: d, orbitals, kind: SideKind::Audit, job: ans.job.id, positions: positions() });
                }
                self.judge_audit(ans.job, energy, forces, face, unfit, ans.ms, ans.scf_steps);
            }
        }
    }

    /// A background audit's verdict (`surrogate` module doc, "Audit"): the sample is admitted like any
    /// teacher answer, the gap recorded; a failed audit untrusts the model and books the flip at
    /// the next evaluation. One activity event either way: judged (the verdict, the admission's
    /// outcome) or discarded (plan review S11).
    pub(super) fn judge_audit(&mut self, job: AuditJob, energy: f64, forces: Vec<[f64; 3]>, face: Option<f64>, unfit: Option<&str>, ms: f64, scf_steps: Option<usize>) {
        let unaudited = self.answered.saturating_sub(job.answered);
        let AuditJob { scene, prov, .. } = job;
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // the model current now is the one to judge (a refit since the dispatch is what counts)
        let pred = self
            .model
            .as_ref()
            .filter(|m| m.provenance == prov && m.basis.species_indices(&z).is_ok())
            .and_then(|m| m.predict(&pos, &z, false).ok());
        let Some(pred) = pred else {
            let note = "surrogate audit: discarded — the model changed settings or species since its dispatch";
            let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::Discarded);
            ev.detail = note.to_string();
            ev.n_atoms = pos.len();
            ev.solve_ms = ms;
            ev.scf_steps = scf_steps;
            ev.scf_end = scf_steps.filter(|&n| n > 0).map(|_| sim::ScfEnd::Converged);
            ev.audit_lag = Some(unaudited);
            self.push_event(ev);
            self.pending_notes.push(note.into());
            return;
        };
        let n = pos.len().max(1) as f64;
        let gap = energy - pred.energy;
        let near = self.near_quarantined(&pos, &z);
        let mut notes = Vec::new();
        let rule = box_affected_face_density(prov.h);
        let outcome = self.admit_and_keep(&pos, &z, energy, &forces, face, Some((pred.energy, &pred.forces)), &near, true, false, false, &prov, unfit, &mut notes);
        let box_rho = face.filter(|&rho| rho > rule);
        let thr = self.policy.energy_std_per_atom;
        // forces are judged too (M6 Task 11): the worst atom against the variance rule's bound
        // "not within": a NaN gap or force residual fails the audit (activity Task 3b)
        let force_fail = worst_force_residual(&self.policy, &pred.forces, &forces, audit_force_floor(&prov.ionic)).filter(|(d, b)| !(d <= b));
        let verdict = if box_rho.is_some() {
            AuditVerdict::BoxAffected
        } else if !(gap.abs() / n <= thr) || force_fail.is_some() {
            AuditVerdict::Failed
        } else {
            AuditVerdict::Passed
        };
        notes.insert(0, match verdict {
            AuditVerdict::Passed => format!("surrogate audit passed (background teacher, judged {unaudited} answered steps after its dispatch): teacher − model {gap:+.2e} Ha"),
            AuditVerdict::BoxAffected => format!(
                "surrogate audit not judged: the teacher's answer is box-affected (face density {:.1e} > {rule:.1e}) — its teacher − model {gap:+.2e} Ha is the box's confinement as much as the model's error",
                box_rho.unwrap_or(0.0)
            ),
            AuditVerdict::Failed => {
                self.audit_failed = Some(gap);
                self.audit_force = force_fail;
                self.audit_atoms = n as usize;
                self.audit_flip = self.last.as_ref().is_some_and(|l| l.on_model);
                match force_fail {
                    Some((d, b)) if gap.abs() / n <= thr => format!(
                        "surrogate audit failed on forces (background teacher): worst atom off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.2e} Ha) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until the next refit"
                    ),
                    _ => format!(
                        "surrogate audit failed (background teacher): teacher − model {gap:+.2e} Ha ({:.1e} > {thr:.1e} Ha/atom) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until the next refit",
                        gap.abs() / n
                    ),
                }
            }
        });
        self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: unaudited, asynchronous: true, teacher_ms: ms });
        // the prediction judged above is the current model's, so the stamp is this fit's
        self.last_audit_fit = self.fits;
        let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, outcome);
        ev.detail = notes.join("\n");
        ev.face_density = face;
        ev.box_affected = box_rho.is_some();
        ev.gap = Some(gap);
        ev.n_atoms = pos.len();
        ev.solve_ms = ms;
        ev.scf_steps = scf_steps;
        ev.scf_end = scf_steps.filter(|&n| n > 0).map(|_| sim::ScfEnd::Converged);
        ev.audit = Some(verdict);
        ev.audit_lag = Some(unaudited);
        self.push_event(ev);
        self.pending_notes.extend(notes);
    }
}
