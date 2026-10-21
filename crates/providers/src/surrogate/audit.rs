//! The audits (the `surrogate` module doc, rule 8 "Audit") and the background auditor (ruling S14)
//! with its side densities (the module doc, "Side densities"): `AuditVerdict`/`AuditRecord` (defined
//! in `sim` since M6, for `TrainableStatus`), the auditor thread, dispatch and judgement, the
//! display and isolated-atom requests. Split out of `surrogate_ks.rs` in M6 Task 1; the narrative
//! stays in the parent module doc.

use super::*;
use surrogate::history::PointKind;

pub use sim::{AuditRecord, AuditVerdict};

/// The `u8` code a fallback point of the audit history carries for its refusing rule
/// (`surrogate::history::PointKind::Fallback`; M8a Task 1 fix round 1). The table is the
/// provider's: the `surrogate` crate never reads it. Appended only — a code, once given, keeps
/// its meaning in every history file.
///
/// | code | rule |
/// |---|---|
/// | 0 | `Cell` (the cell check) |
/// | 1 | `NoModel` (rule 1) |
/// | 2 | `NewSpecies` (rule 2) |
/// | 3 | `OtherSettings` (rule 3) |
/// | 4 | `Untrusted` (rule 4, the trust rule) |
/// | 5 | `Class(Pair)` (rule 4b) |
/// | 6 | `Class(ManyBody)` (rule 4b) |
/// | 7 | `IsolatedAtom` (rule 5a) |
/// | 8 | `Coordination` (rule 5b) |
/// | 9 | `Spread` (rule 5, the envelope) |
/// | 10 | `NoGrade` (rule 5e, no active set) |
/// | 11 | `NewEnvironment` (rule 5d) |
/// | 12 | `UnspannedAtom` (rule 5e at γ = ∞) |
/// | 13 | `Grade` (rule 5e) |
/// | 14 | `ForceUncertainty` (rule 6) |
/// | 15 | `EnergyUncertainty` (rule 7) |
/// | 16 | `NonFiniteAnswer` (never recorded: no finite prediction) |
/// | 17 | `NearQuarantine` (the quarantine, audit and shell fences) |
/// | 18 | `Audit` (an audit's own kind is `PointKind::Audit`; the code is for completeness) |
/// | 19 | `RelaxVerification` |
/// | 20 | `Pass` (a sampling-pass job; never a step) |
/// | 21 | `Class(_)` refused **and** the variance verdict (rule 6/7) refused too — the class's note
///       is shown, but the step is not the class verdict's evidence ([`HISTORY_RULE_CLASS_AND_VARIANCE`];
///       M8a Task 2: `calibrate` reads codes 5 and 6 only) |
pub fn history_rule_code(rule: TeachRule) -> u8 {
    use sim::activity::EnvClass;
    match rule {
        TeachRule::Cell => 0,
        TeachRule::NoModel => 1,
        TeachRule::NewSpecies(_) => 2,
        TeachRule::OtherSettings => 3,
        TeachRule::Untrusted(_) => 4,
        TeachRule::Class(EnvClass::Pair) => 5,
        TeachRule::Class(EnvClass::ManyBody) => 6,
        TeachRule::IsolatedAtom => 7,
        TeachRule::Coordination => 8,
        TeachRule::Spread => 9,
        TeachRule::NoGrade => 10,
        TeachRule::NewEnvironment => 11,
        TeachRule::UnspannedAtom => 12,
        TeachRule::Grade => 13,
        TeachRule::ForceUncertainty => 14,
        TeachRule::EnergyUncertainty => 15,
        TeachRule::NonFiniteAnswer => 16,
        TeachRule::NearQuarantine => 17,
        TeachRule::Audit => 18,
        TeachRule::RelaxVerification => 19,
        TeachRule::Pass(_) => 20,
    }
}

/// The history code of a step the class rule (4b) refused while the variance verdict (rule 6/7)
/// refused it as well (M8a Task 2): `evaluate` shows the class's note (rules 6–7 yield to the
/// class's), but the model would not have answered the step had the class been trusted, so the
/// comparison is not evidence for the class verdict — `calibrate::in_jurisdiction` leaves it out.
pub const HISTORY_RULE_CLASS_AND_VARIANCE: u8 = 21;

/// The rule a [`history_rule_code`] names, in the decision list's words ("?" for an unknown code).
pub fn history_rule_name(code: u8) -> &'static str {
    match code {
        0 => "cell check",
        1 => "rule 1 (no model)",
        2 => "rule 2 (new species)",
        3 => "rule 3 (other settings)",
        4 => "rule 4 (untrusted)",
        5 => "rule 4b (pair class)",
        6 => "rule 4b (many-body class)",
        7 => "rule 5a (isolated atom)",
        8 => "rule 5b (coordination)",
        9 => "rule 5 (envelope)",
        10 => "rule 5e (no grade)",
        11 => "rule 5d (new environment)",
        12 => "rule 5e (unspanned atom)",
        13 => "rule 5e (grade)",
        14 => "rule 6 (force uncertainty)",
        15 => "rule 7 (energy uncertainty)",
        16 => "non-finite answer",
        17 => "fence (quarantine, audit or shell)",
        18 => "audit",
        19 => "relax verification",
        20 => "pass",
        21 => "rule 4b with rule 6/7 failing too",
        _ => "?",
    }
}

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

/// Where a failed audit came from (the teach-around review, SHOULD-T5: the chip says which).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum AuditOrigin {
    /// A strict-mode audit (`audit_every`), or one of before the origin was recorded.
    #[default]
    Audit,
    /// A teach-around pass's held-out check (`teach`).
    TeachCheck,
    /// A held-out draft guess judged once strict mode reached it (`judge_held_out`).
    HeldOutGuess,
    /// A pass's shell check (`sampling` module doc, 3.): the interaction left beyond the scan limit
    /// that the cutoff truncates.
    ShellCheck,
}

impl AuditOrigin {
    /// The sidecar's word ("audit" is written as no key at all: older files stay byte-identical).
    pub(super) fn word(self) -> &'static str {
        match self {
            AuditOrigin::Audit => "audit",
            AuditOrigin::TeachCheck => "teach-check",
            AuditOrigin::HeldOutGuess => "held-out-guess",
            AuditOrigin::ShellCheck => "shell-check",
        }
    }

    pub(super) fn from_word(w: &str) -> AuditOrigin {
        match w {
            "teach-check" => AuditOrigin::TeachCheck,
            "held-out-guess" => AuditOrigin::HeldOutGuess,
            "shell-check" => AuditOrigin::ShellCheck,
            _ => AuditOrigin::Audit,
        }
    }
}

/// A failed audit's case (module doc, rule 8, "A failed audit stays failed"): the audited geometry,
/// the teacher's labels there, the store index of the kept answer (`None`: not kept — an unfit or
/// refused answer) and the surface's ionic tag (the force audit's floor).
#[derive(Clone, Debug)]
pub(super) struct FailedAudit {
    pub(super) pos: Vec<[f64; 3]>,
    pub(super) z: Vec<u8>,
    pub(super) energy: f64,
    pub(super) forces: Vec<[f64; 3]>,
    pub(super) index: Option<usize>,
    pub(super) ionic: String,
    /// Refits that re-judged it and found it still failed (`AUDIT_REJUDGE_LIMIT`).
    pub(super) rejudges: u32,
    pub(super) origin: AuditOrigin,
    /// The species pair (Z_a ≤ Z_b) a shell check measured: the pair its fence refuses in the
    /// cutoff shell (`trust::shell_fence_refusal`; final review I2 — carried, never re-derived
    /// from the case's geometry). `Some` exactly for `AuditOrigin::ShellCheck`.
    pub(super) pair: Option<(u8, u8)>,
}

/// The isolated neutral atom of species `z` at the centre of `scene`'s cell, solved under the scene's own ionic policy
/// (the display's atomic superposition and the references both compare it with the scene's solved density: under
/// `CoresOnly` an oxygen atom of 6 valence electrons, never the 8-electron all-electron one — Task 9 review I1 /
/// physics SHOULD-1). `mass`: the nucleus's mass when the caller has one (a reference), else the table's.
pub(super) fn isolated_atom(z: u8, mass: Option<f64>, scene: &Scene) -> Scene {
    let mut nucleus = sim::Nucleus::new(z, [0.0; 3]);
    if let Some(m) = mass {
        nucleus.mass = m;
    }
    Scene { nuclei: vec![nucleus], cell: scene.cell.clone(), charge: 0, multiplicity: None, sigma: scene.sigma, external_field: [0.0; 3], ionic: scene.ionic }
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
    /// An audit dispatched in draft mode whose geometry the strict policy would not have answered
    /// (fix (b), 2026-10-18): its answer is a `DraftAudit` sample, held out of the strict fit.
    pub(super) draft_guess: bool,
}

/// What the auditor's teacher returned for an audit or a display job.
pub(super) struct Solved {
    pub(super) energy: f64,
    pub(super) forces: Vec<[f64; 3]>,
    pub(super) face: Option<f64>,
    /// The multiplicity the teacher solved at (`EvalStats::multiplicity`).
    pub(super) multiplicity: Option<u32>,
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
                                    Solved { energy: r.energy, forces: r.forces, face: r.stats.face_density, multiplicity: r.stats.multiplicity, unfit: r.stats.untrainable(), display: r.display, orbitals: r.orbitals }
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
        let atom = isolated_atom(z, None, scene);
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
        // the driver wants this one too: its answer is handed on (`shared`, "References per box")
        let n = scene.cell.n;
        self.own_ref_requests.retain(|&o| o != (z, n));
        self.request_atom_reference_job(z, mass, scene)
    }

    /// [`request_atom_reference`](Self::request_atom_reference)'s job, for the driver or for the
    /// provider itself (`shared::request_box_references`).
    pub(super) fn request_atom_reference_job(&mut self, z: u8, mass: f64, scene: &Scene) -> bool {
        self.poll_side_jobs();
        if self.auditor.is_none() {
            return false;
        }
        let pending = self.side_queue.iter().any(|j| j.2 == JobKind::Reference(z)) || self.auditor.as_ref().is_some_and(|a| a.busy == Some(JobKind::Reference(z)));
        if pending {
            return true;
        }
        let atom = isolated_atom(z, Some(mass), scene);
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
        let draft_guess = std::mem::take(&mut self.dispatch_draft_guess);
        let job = AuditJob { scene, prov, generation: a.generation(), answered, kind, id, view, interrupt: self.display_interrupt.clone(), core: self.display_core, draft_guess };
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

    /// Every judged comparison of the model's prediction with the teacher's answer — audits and
    /// consulted-model fallbacks, by `PointKind` (M8a Task 1, `surrogate::history`): the newest
    /// `surrogate::HISTORY_CAP` points, oldest first, and the counts of those that left none. A
    /// record; nothing in the policy reads it.
    pub fn audit_history(&self) -> &surrogate::AuditHistory {
        &self.audit_history
    }

    /// Records a judged comparison in the history (`surrogate::history`, M8a Task 1) — an audit, or
    /// (fix round 1, the physics review's I3) a step the teacher answered because a rule refused
    /// the model's answer, `kind` saying which: the model's predictive stds at the geometry —
    /// `stds` = (total-energy std, per-atom force std, its epistemic part) — against the teacher's
    /// errors, ‖F_teacher − F_model‖ per atom and the signed gap E_teacher − E_model, with each
    /// atom's class code (0 isolated, 1 pair, 2 many-body by its neighbour count within the model's
    /// r_cut) and the model generation judged. No point, one count, for a box-affected answer (the
    /// box's confinement is in the error), a draft-mode guess at a geometry strict mode refuses (no
    /// verdict on the strict model), an unfit teacher answer (cores overlap, an occupation change:
    /// not the surface the model learns), or a model without a variance factor (`stds` None).
    ///
    /// # Panics
    /// If the shapes disagree with the geometry (a programming error at the call site: every
    /// judged comparison leaves a point or a count, so nothing is dropped silently).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_history_point(&mut self, kind: PointKind, pos: &[[f64; 3]], stds: Option<(f64, &[f64], &[f64])>, model_energy: f64, model_forces: &[[f64; 3]], energy: f64, forces: &[[f64; 3]], box_affected: bool, draft: bool, unfit: bool) {
        if box_affected {
            self.audit_history.unrecorded.box_affected += 1;
            return;
        }
        if draft {
            self.audit_history.unrecorded.draft += 1;
            return;
        }
        if unfit {
            self.audit_history.unrecorded.unfit += 1;
            return;
        }
        let Some((energy_std, force_std, force_std_epistemic)) = stds else {
            self.audit_history.unrecorded.no_variance += 1;
            return;
        };
        let n = pos.len();
        assert!(model_forces.len() == n && forces.len() == n, "a comparison's forces agree with its geometry");
        let rc2 = self.model.as_ref().map_or(0.0, |m| m.basis.r_cut * m.basis.r_cut);
        let classes = neighbour_counts(pos, rc2).into_iter().map(|c| atom_class(c).map_or(0, |k| k as u8 + 1)).collect();
        let err_force = forces.iter().zip(model_forces).map(|(t, m)| ((t[0] - m[0]).powi(2) + (t[1] - m[1]).powi(2) + (t[2] - m[2]).powi(2)).sqrt()).collect();
        // |F_teacher| per atom: the per-atom bound's input (M8a Task 2 fix round 1, review A.1)
        let teacher_force = forces.iter().map(|t| (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()).collect();
        // (`push` checks the per-atom lengths and the class codes and panics on a mismatch)
        self.audit_history.push(surrogate::AuditPoint {
            kind,
            classes,
            pred_force_std: force_std.to_vec(),
            pred_force_std_epistemic: force_std_epistemic.to_vec(),
            pred_energy_std: energy_std,
            err_force,
            teacher_force,
            err_energy: energy - model_energy,
            n_atoms: n,
            model_gen: self.model_gen,
        });
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
        self.box_audits_unjudged = 0;
        self.audit_flip = false;
        self.last_audit = None;
        self.failed_audits.clear();
        self.audit_fences.clear();
        self.audit_burst = 0;
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
            // in collection mode every solved reference is offered for its box (`shared` module
            // doc, "References per box"); one the provider asked for itself is not handed on
            let n = ans.job.scene.cell.n;
            if let Ok(a) = &r {
                if self.shared.is_some() {
                    self.set_reference_solved(z, a.energy, n, a.face_density);
                }
            }
            if let Some(k) = self.own_ref_requests.iter().position(|&o| o == (z, n)) {
                self.own_ref_requests.remove(k);
                if let Err(e) = &r {
                    self.user_notes.push(format!("shared collection: the {} reference of the {n}³ box did not solve ({e}) — its answers keep waiting (a sampling pass solves it too)", symbol(z)));
                }
                return;
            }
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
            (JobKind::Audit, Ok(Solved { energy, forces, face, multiplicity, display, orbitals, unfit })) => {
                if let Some(d) = display {
                    self.side_display = Some(SideDisplay { display: d, orbitals, kind: SideKind::Audit, job: ans.job.id, positions: positions() });
                }
                self.judge_audit(ans.job, energy, forces, face, multiplicity, unfit, ans.ms, ans.scf_steps);
            }
        }
    }

    /// A background audit's verdict (`surrogate` module doc, "Audit"): the sample is admitted like any
    /// teacher answer, the gap recorded; a failed audit untrusts the model and books the flip at
    /// the next evaluation. One activity event either way: judged (the verdict, the admission's
    /// outcome) or discarded (plan review S11).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn judge_audit(&mut self, job: AuditJob, energy: f64, forces: Vec<[f64; 3]>, face: Option<f64>, multiplicity: Option<u32>, unfit: Option<&str>, ms: f64, scf_steps: Option<usize>) {
        let unaudited = self.answered.saturating_sub(job.answered);
        let AuditJob { scene, prov, draft_guess, .. } = job;
        let tag = if draft_guess { SampleTag::DraftAudit } else { SampleTag::Audit };
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // the canonical surface in collection mode (`shared` module doc; the identity per scene)
        let energy = self.to_canonical(&z, prov.grid_n, energy);
        // the model current now is the one to judge (a refit since the dispatch is what counts);
        // with the variance: the history's point needs the model's stds here (M8a Task 1 — one
        // variance evaluation per audit, within what the cadence budgets)
        let pred = self
            .model
            .as_ref()
            .filter(|m| self.same_surface(&m.provenance, &prov) && m.basis.species_indices(&z).is_ok())
            .and_then(|m| m.predict(&pos, &z, true).ok());
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
        // inside a fenced shell band (the shell fence; final review I1, the coordinator's ruling):
        // strict mode refuses the geometry, so the answer is no judgement of the model — not
        // judged, not kept (a draft guess's included: a band row is never fitted, now or after a
        // later shell check lifts the fence)
        if let Some(why) = self.shell_fence_refusal(&pos, &z) {
            let note = format!("surrogate audit not judged: {} — not a judgement of the model; the answer is not fitted", why.text.trim_start_matches("surrogate: "));
            let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::ShellNotKept);
            ev.detail = note.clone();
            ev.n_atoms = pos.len();
            ev.solve_ms = ms;
            ev.scf_steps = scf_steps;
            ev.scf_end = scf_steps.filter(|&n| n > 0).map(|_| sim::ScfEnd::Converged);
            ev.audit_lag = Some(unaudited);
            self.push_event(ev);
            self.pending_notes.push(note);
            return;
        }
        let n = pos.len().max(1) as f64;
        let gap = energy - pred.energy;
        let near = self.near_quarantined(&pos, &z);
        let mut notes = Vec::new();
        let rule = box_affected_face_density(prov.h);
        let outcome = self.admit_and_keep(&pos, &z, energy, &forces, face, multiplicity, Some((pred.energy, &pred.forces)), &near, true, false, false, false, &prov, unfit, tag, &mut notes);
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
                "surrogate audit not judged: the teacher's answer is box-affected (face density {:.1e} > {rule:.1e}) — its teacher − model {gap:+.2e} Ha is the box's confinement as much as the model's error; the next answered step is audited again",
                box_rho.unwrap_or(0.0)
            ),
            // a draft guess's audit (a geometry strict mode refuses): recorded, held out of the
            // strict fit, and not a verdict on the strict model (module doc, "Draft mode")
            AuditVerdict::Failed if draft_guess => format!(
                "surrogate draft-mode audit failed (background teacher): teacher − model {gap:+.2e} Ha ({:.1e} Ha/atom{}) at a geometry strict mode refuses — the guess was off; the strict model's trust is unchanged and the answer is held out of its fit",
                gap.abs() / n,
                force_fail.map_or(String::new(), |(d, b)| format!(", worst atom's force off by {d:.1e} > {b:.1e} Ha/Bohr"))
            ),
            AuditVerdict::Failed => {
                self.audit_failed = Some(gap);
                self.audit_force = force_fail;
                self.audit_atoms = n as usize;
                let index = outcome.is_kept().then(|| self.samples().saturating_sub(1));
                self.failed_audits.push(FailedAudit { pos: pos.clone(), z: z.clone(), energy, forces: forces.clone(), index, ionic: prov.ionic.clone(), rejudges: 0, origin: AuditOrigin::Audit, pair: None });
                self.audit_flip = self.last.as_ref().is_some_and(|l| l.on_model);
                match force_fail {
                    Some((d, b)) if gap.abs() / n <= thr => format!(
                        "surrogate audit failed on forces (background teacher): worst atom off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.2e} Ha) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until a refit predicts that geometry without the audited sample"
                    ),
                    _ => format!(
                        "surrogate audit failed (background teacher): teacher − model {gap:+.2e} Ha ({:.1e} > {thr:.1e} Ha/atom) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until a refit predicts that geometry without the audited sample",
                        gap.abs() / n
                    ),
                }
            }
        });
        self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: unaudited, asynchronous: true, teacher_ms: ms });
        // the prediction judged above is the current model's, so the stamp is this fit's
        self.last_audit_fit = self.fits;
        // a box-affected audit judged nothing (M8a Task 2, the Task 1 physics review's I4): it
        // does not satisfy the cadence — the next answered step dispatches another — and the
        // checklist counts it; the model's surface and trust are untouched (the gap is the box's)
        if verdict == AuditVerdict::BoxAffected {
            self.audit_due = true;
            self.box_audits_unjudged += 1;
        } else {
            self.box_audits_unjudged = 0;
        }
        // the history's point (M8a Task 1)
        let stds = match (pred.energy_std, pred.force_std.as_ref(), pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f), Some(fe)) => Some((e, &f[..], &fe[..])),
            _ => None,
        };
        self.record_history_point(PointKind::Audit, &pos, stds, pred.energy, &pred.forces, energy, &forces, box_rho.is_some(), draft_guess, unfit.is_some());
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

#[cfg(test)]
mod isolated_atom_tests {
    use super::*;

    /// Task 9 review I1: the isolated atom of the display's superposition and of the references is solved under the
    /// scene's own ionic policy (never all-electron by default), on its cell and σ, neutral, with the caller's mass.
    #[test]
    fn the_isolated_atom_follows_the_scenes_policy() {
        let scene = Scene {
            nuclei: vec![sim::Nucleus::new(8, [0.0; 3]), sim::Nucleus::new(1, [1.8, 0.0, 0.0])],
            cell: sim::Cell { n: 61, h: 0.2 },
            charge: 1,
            multiplicity: Some(2),
            sigma: Some(0.3),
            external_field: [0.0; 3],
            ionic: sim::IonicPolicy::CoresOnly,
        };
        for p in sim::IonicPolicy::ALL {
            let atom = isolated_atom(8, None, &Scene { ionic: p, ..scene.clone() });
            assert_eq!((atom.ionic, atom.charge, atom.multiplicity, atom.sigma, atom.cell.clone()), (p, 0, None, Some(0.3), scene.cell.clone()));
            assert_eq!((atom.nuclei.len(), atom.nuclei[0].z, atom.nuclei[0].pos), (1, 8, [0.0; 3]));
        }
        assert_eq!(isolated_atom(8, Some(12345.0), &scene).nuclei[0].mass, 12345.0);
        assert_eq!(isolated_atom(8, None, &scene).nuclei[0].mass, sim::Nucleus::new(8, [0.0; 3]).mass);
    }
}
