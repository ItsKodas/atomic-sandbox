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
/// | 22 | `Segment` (a capped segment's scene: teacher only, never recorded — no model answer; M8a Task 10) |
/// | 23 | `NearMisfit` (an atom next to a training sample the model cannot reproduce; spec 2026-11-13 §4.3) |
/// | 21 | `Class(_)` refused **and** the variance verdict (rule 6/7) refused too — the class's note
///       is shown, but the step is not the class verdict's evidence ([`HISTORY_RULE_CLASS_AND_VARIANCE`];
///       M8a Task 2: `calibrate` reads codes 5 and 6 only) |
/// | 24 | `ForceUncertainty` / `EnergyUncertainty` refused **only** by the tail-aware factor — rule 6/7
///       at λ_F = max(λ_sum, λ_T) or λ_E refused, at λ_sum alone it passes ([`HISTORY_RULE_TAIL_INFLATION`];
///       trust tightening, spec 2026-11-18 §4: never in the answer sample P_A, since λ selected it) |
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
        TeachRule::Segment => 22,
        TeachRule::NearMisfit => 23,
    }
}

/// The history code of a step the class rule (4b) refused while the variance verdict (rule 6/7)
/// refused it as well (M8a Task 2): `evaluate` shows the class's note (rules 6–7 yield to the
/// class's), but the model would not have answered the step had the class been trusted, so the
/// comparison is not evidence for the class verdict — `calibrate::in_jurisdiction` leaves it out.
pub const HISTORY_RULE_CLASS_AND_VARIANCE: u8 = 21;

/// The history code of a step rule 6 or 7 refused only at the tail-aware inflations (trust
/// tightening, spec 2026-11-18 §4; `calibrate`, "The tail-aware inflations"): rule 6 at
/// λ_F(j) = max(λ_sum(k), λ_T(k)) or rule 7 at max(max_j λ_F(j), max_k λ_E(k)) refused, and at
/// λ_sum alone both pass. Recorded for the Log and the measurement; `calibrate::in_answer_sample`
/// leaves it out (λ selected it), while λ_sum's posterior jurisdiction reads it as it reads 14/15.
pub const HISTORY_RULE_TAIL_INFLATION: u8 = 24;

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
        22 => "capped segment (teacher only)",
        23 => "near a misfit",
        24 => "rule 6/7 at the tail-aware inflation (passes at λ_sum)",
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
    /// A Harris display solve at the job's geometry (`ForceProvider::harris_display`; live-orbitals
    /// spec §4.2): orbitals only, display only — never a fallback, an audit, a teacher solve, a
    /// sample or a teach event. At most one is queued (`SurrogateKs::harris_queued`).
    HarrisDisplay,
    /// A composition check's bonded part alone (`stray`, spec 2026-11-18 §4.3): the scene's
    /// bonded nuclei in its cell, charge 0, automatic multiplicity, the scene's ionic policy; its
    /// answer goes to the check in flight (`SurrogateKs::take_stray_answer`), never a sample.
    Fragment,
    /// A composition check's stray atom of this Z alone at its scene position, at its reference
    /// multiplicity, in the scene's cell — like [`Fragment`](JobKind::Fragment) never a sample.
    AtomAt(u8),
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
    /// A failed audit at a scene with stray atoms that its attribution check blamed on the model
    /// (`stray` module doc, "Audits of a cleared answer"): its label is the bonded part's teacher answer
    /// plus each stray atom's E_ref — valid only while those atoms are stray at the model's cutoff
    /// (`rejudge_failed_audits` fences it otherwise; Task 5 fix round 1, M7).
    StrayAttributed,
}

impl AuditOrigin {
    /// The sidecar's word ("audit" is written as no key at all: older files stay byte-identical).
    pub(super) fn word(self) -> &'static str {
        match self {
            AuditOrigin::Audit => "audit",
            AuditOrigin::TeachCheck => "teach-check",
            AuditOrigin::HeldOutGuess => "held-out-guess",
            AuditOrigin::ShellCheck => "shell-check",
            AuditOrigin::StrayAttributed => "stray-attributed",
        }
    }

    pub(super) fn from_word(w: &str) -> AuditOrigin {
        match w {
            "teach-check" => AuditOrigin::TeachCheck,
            "held-out-guess" => AuditOrigin::HeldOutGuess,
            "shell-check" => AuditOrigin::ShellCheck,
            "stray-attributed" => AuditOrigin::StrayAttributed,
            _ => AuditOrigin::Audit,
        }
    }
}

/// A failed audit's case (module doc, rule 8, "A failed audit stays failed"): the audited geometry,
/// the teacher's labels there, the store index of the kept answer (`None`: not kept — an unfit or
/// refused answer) and the surface's ionic tag (the force audit's floor).
#[derive(Clone, Debug, PartialEq)]
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

/// The epistemic energy std a history point may carry (`record_history_point`): a finite,
/// non-negative value as itself (−0 as +0), anything else — NaN, ±∞, a negative number — as absent.
/// `AuditHistory::push` panics on what this refuses; a radicand below 0 is clamped to 0 before the
/// root (`epistemic_energy_std`), so a cancelled epistemic part arrives as 0, never as NaN.
pub(super) fn recordable_energy_std(s: Option<f64>) -> Option<f64> {
    s.filter(|v| v.is_finite() && *v >= 0.0).map(|v| if v == 0.0 { 0.0 } else { v })
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
    Scene { nuclei: vec![nucleus], cell: scene.cell.clone(), charge: 0, multiplicity: None, sigma: scene.sigma, external_field: [0.0; 3], ionic: scene.ionic, train_mask: None }
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
    /// The orbital shares' free-atom tables (`SurrogateKs::share_tables`), handed to the teacher
    /// before it solves; it weights them at this job's positions (module doc, "Orbitals").
    pub(super) shares: Option<Arc<ShareTables>>,
    /// An audit's history flags from the step that dispatched it (`SurrogateKs::dispatch_flags`;
    /// module doc, "The history's flags"): its cadence-fresh and fenced bits. Empty for every other
    /// kind. `TICK` on a calibration tick's audit (`SurrogateKs` module doc, "Calibration ticks").
    pub(super) flags: surrogate::history::PointFlags,
    /// A calibration tick alone, not rule 8's audit (`SurrogateKs::dispatch_tick_only`): judged as
    /// an audit, but a box-affected answer does not make rule 8 audit again. False for every other
    /// kind and for a rule-8 audit that is a tick as well.
    pub(super) tick_only: bool,
    /// The isolated-atom profiles a [`JobKind::HarrisDisplay`] job superposes (`None` for every
    /// other kind).
    pub(super) harris: Option<Arc<sim::HarrisAtoms>>,
}

/// A Harris job's error string when its solve was cancelled (a newer generation, or the display
/// interrupt: the user's next command): `handle_answer` leaves no note for it.
const HARRIS_CANCELLED: &str = "";

/// What the auditor's teacher returned for an audit or a display job. A Harris job's answer has
/// only `orbitals` (no energy: `energy` is NaN, `forces` empty, never read).
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
    /// The solve's per-spin frontier levels and partial levels (`EvalStats::frontier`,
    /// `EvalStats::partial`) and whether it warned (`EvalStats::warnings`): a composition check's
    /// (`stray::PartAnswer`).
    pub(super) frontier: [sim::Frontier; 2],
    pub(super) partial: Vec<sim::PartialLevel>,
    pub(super) warned: bool,
}

/// The auditor's answer: what the teacher solved, or its error; for a `Reference` job the scan's
/// outcome is in `reference` (and `result` is unused). `scf_steps`: the solve's SCF steps as its
/// teacher reported them (`None` for an error or a reference scan); `ts` its electronic entropy term T_el·S
/// (`EvalStats::energy.ts`, Ha; `None` likewise — the Activity row's "−TS", Fermi–Dirac spec 2026-11-13 §6.6).
pub(super) struct AuditAnswer {
    pub(super) job: AuditJob,
    pub(super) result: Result<Solved, String>,
    pub(super) reference: Option<Result<AtomReference, ProviderError>>,
    pub(super) ms: f64,
    pub(super) scf_steps: Option<usize>,
    pub(super) ts: Option<f64>,
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
                    let atom = matches!(job.kind, JobKind::Atom(_) | JobKind::Reference(_) | JobKind::AtomAt(_));
                    // a Harris display solve touches neither the SCF warm state nor its
                    // extrapolation (it writes its own `HarrisState`; spec §4.2): no reset, no
                    // discontinuity, and `last_atom` stays the last SCF job's
                    if job.kind != JobKind::HarrisDisplay {
                        // the audits are thousands of steps apart: no extrapolation through the
                        // previous one's geometry (the density is kept as a warm start) — but an
                        // isolated atom's density is no warm start for a molecule, nor the reverse. A
                        // composition check's parts start cold too (review M1): one failing check refuses
                        // its signature for the session, so its solves must not depend on the job before
                        // (a reproducible cold solve, as the migration's)
                        let cold = atom || job.kind == JobKind::Fragment;
                        if cold || last_atom {
                            teacher.reset();
                        } else {
                            teacher.discontinuity();
                        }
                        last_atom = atom;
                    }
                    // the one orbital producer: on for audits, refreshes and Harris, off for an atom — except a composition
                    // check's parts, which ask for one unoccupied level per spin channel (`stray` module doc, "The
                    // levels"; controller ruling 10b): an empty channel stores no orbital, so without it a stray H's empty
                    // 1s↓ — a real acceptor — would have no level and the check would be void for good
                    let part = matches!(job.kind, JobKind::Fragment | JobKind::AtomAt(_));
                    let want = if part {
                        super::stray::part_view()
                    } else if atom {
                        OrbitalViewRequest::default()
                    } else {
                        job.view.clone()
                    };
                    if want != view {
                        teacher.set_orbital_view(want.clone());
                        view = want;
                    }
                    // every isolated-atom profile is the valence density, without ρ_pc (spec
                    // §4.1: the superposition cloud and the Harris input are built from them)
                    let want_core = job.core && !atom;
                    if want_core != core {
                        teacher.set_display_partial_core(want_core);
                        core = want_core;
                    }
                    // the driver's display interrupt (re-review N1): a forced refresh's unoccupied
                    // eigensolve yields this job slot to the user's next command
                    if let Some(i) = &job.interrupt {
                        teacher.set_display_interrupt(i.clone());
                    }
                    // the share tables (orbital-atom-shares): weighted at this job's positions
                    teacher.set_share_tables(job.shares.clone());
                    let t = Instant::now();
                    let g = job.generation;
                    let (mut scf_steps, mut ts) = (None, None);
                    let (result, reference) = match job.kind {
                        // the multiplicity scan (each solve cold, the teacher reset after it)
                        JobKind::Reference(z) => {
                            let mass = job.scene.nuclei.first().map_or(0.0, |n| n.mass);
                            done_count.store(0, Ordering::Relaxed);
                            let r = sim::references::solve_atom_reporting(teacher.as_mut(), &job.scene.cell, job.scene.sigma, job.scene.ionic, z, mass, &mut || current.load(Ordering::Relaxed) != g, &mut |k| done_count.store(k, Ordering::Relaxed));
                            (Err(String::new()), Some(r))
                        }
                        // display only: abandoned with its generation, and yields to the user's
                        // next command like any display work (the driver's display interrupt)
                        JobKind::HarrisDisplay => {
                            let atoms = job.harris.clone().unwrap_or_else(|| Arc::new(sim::HarrisAtoms { profiles: Vec::new() }));
                            let interrupt = job.interrupt.clone();
                            let mut abort = || current.load(Ordering::Relaxed) != g || interrupt.as_ref().is_some_and(|i| i());
                            let result = match teacher.harris_display(&job.scene, &atoms, &mut abort) {
                                Some(sim::HarrisAnswer::Solved(Ok(o))) => Ok(Solved { energy: f64::NAN, forces: Vec::new(), face: None, multiplicity: None, display: None, orbitals: Some(o), unfit: None, frontier: Default::default(), partial: Vec::new(), warned: false }),
                                // superseded or yielded to the user: silent (review minor 3)
                                Some(sim::HarrisAnswer::Solved(Err(ProviderError::Cancelled))) => Err(HARRIS_CANCELLED.into()),
                                Some(sim::HarrisAnswer::Solved(Err(e))) => Err(e.to_string()),
                                Some(sim::HarrisAnswer::Queued { .. }) => Err("the background teacher queued it instead of solving".into()),
                                None => Err("the background teacher has no Harris display".into()),
                            };
                            (result, None)
                        }
                        _ => (
                            teacher
                                .evaluate(&job.scene, &mut || current.load(Ordering::Relaxed) != g)
                                .map(|r| {
                                    (scf_steps, ts) = (Some(r.stats.scf_steps), Some(r.stats.energy.ts));
                                    let frontier = if part { super::stray::with_unoccupied(r.stats.frontier, r.orbitals.as_ref()) } else { r.stats.frontier };
                                    Solved {
                                        energy: r.energy,
                                        forces: r.forces,
                                        face: r.stats.face_density,
                                        multiplicity: r.stats.multiplicity,
                                        unfit: r.stats.untrainable(),
                                        display: r.display,
                                        orbitals: r.orbitals,
                                        frontier,
                                        warned: !r.stats.warnings.is_empty(),
                                        partial: r.stats.partial,
                                    }
                                })
                                .map_err(|e| e.to_string()),
                            None,
                        ),
                    };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    if atx.send(AuditAnswer { job, result, reference, ms, scf_steps, ts }).is_err() {
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

    /// Queues a Harris display job at `scene` with the profiles `atoms` (`ForceProvider::harris_display`;
    /// `surrogate` module doc, "Side densities"): the one queued Harris job is replaced (a newer
    /// geometry wins), sent when the auditor is free and the side queue is empty — while playing
    /// also only with no audit due ([`pump_side_queue`](Self::pump_side_queue)). Its side-job
    /// number, or `None` without an auditor, for an empty scene, or while a sampling pass runs (a
    /// pass's solves come first, and starting one drops the queued job; the caller asks again).
    pub(super) fn queue_harris(&mut self, scene: &Scene, atoms: &Arc<sim::HarrisAtoms>) -> Option<u64> {
        self.poll_side_jobs();
        if self.auditor.is_none() || scene.nuclei.is_empty() || self.pass.is_some() {
            return None;
        }
        self.side_jobs += 1;
        self.harris_queued = Some((scene.clone(), atoms.clone(), self.side_jobs));
        self.pump_side_queue();
        Some(self.side_jobs)
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
        self.request_atom_reference_job(z, mass, scene, true)
    }

    /// [`request_atom_reference`](Self::request_atom_reference)'s job, for the driver (`driver`:
    /// its answer is handed on, always) or for the provider itself (`shared::request_box_references`).
    /// One job per (Z, box n): a repeat — by either — joins the one queued or running (fix round 3:
    /// a job of the same Z in another box is another job).
    pub(super) fn request_atom_reference_job(&mut self, z: u8, mass: f64, scene: &Scene, driver: bool) -> bool {
        self.poll_side_jobs();
        if self.auditor.is_none() {
            return false;
        }
        let n = scene.cell.n;
        if let Some(j) = self.ref_jobs.iter_mut().find(|j| j.z == z && j.n == n) {
            j.driver |= driver;
            return true;
        }
        // the collection the requesting scene shares, if any: only then may its answer become one
        // of the collection's references (fix round 3, review 1)
        let collection = self.shared.as_ref().filter(|sh| sh.coll.admit_provenance(&self.scene_provenance(scene)).is_ok()).map(|sh| sh.coll.key.slug());
        let atom = isolated_atom(z, Some(mass), scene);
        let prov = self.scene_provenance(&atom);
        self.ref_jobs.push(super::shared::RefJob { z, n, driver, collection });
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
    pub(super) fn fail_references(&mut self, _busy: Option<JobKind>, why: &str) {
        // every job queued or running is in `ref_jobs`; the driver's are answered (fix round 3)
        self.side_queue.retain(|j| !matches!(j.2, JobKind::Reference(_)));
        for j in std::mem::take(&mut self.ref_jobs) {
            if j.driver {
                self.atom_references.push((j.z, Err(ProviderError::Invalid(why.to_string()))));
            }
        }
    }

    /// Sends the next queued side job if the auditor is idle (`surrogate` module doc, "Side densities": a
    /// queued request goes ahead of an audit that fell due meanwhile). The queued Harris job goes
    /// last: only with the side queue empty (refreshes, and the atoms and references it needs) and
    /// no sampling pass running; while playing also only with no audit due (a safety check; spec
    /// §4.2, "Scheduling"). Paused (the driver's `force_every_step`) it does not wait for a due
    /// audit: only an answered step dispatches that, and paused there may be none for a long time
    /// (spec G1: a paused change is shown) — the audit goes at the next evaluate.
    pub(super) fn pump_side_queue(&mut self) {
        if !self.auditor_idle() {
            return;
        }
        let sent = if let Some((scene, prov, kind, id)) = self.side_queue.pop_front() {
            self.send_job(scene, prov, kind, id, None)
        } else if !(self.audit_due && !self.orbital_view.force_every_step) && self.pass.is_none() && self.harris_queued.is_some() {
            let (scene, atoms, id) = self.harris_queued.take().expect("checked");
            let prov = self.scene_provenance(&scene);
            self.send_job(scene, prov, JobKind::HarrisDisplay, id, Some(atoms))
        } else {
            true
        };
        if !sent {
            // the thread is gone: the references asked for are answered (the popped one is still
            // in `ref_jobs`), the rest dropped
            self.fail_references(None, "the background teacher is gone");
            self.side_queue.clear();
            self.harris_queued = None;
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
    /// gone (then the audits run in the step from now on). `harris`: a Harris job's profiles.
    pub(super) fn send_job(&mut self, scene: Scene, prov: Provenance, kind: JobKind, id: u64, harris: Option<Arc<sim::HarrisAtoms>>) -> bool {
        let answered = self.answered;
        let Some(a) = self.auditor.as_mut() else { return false };
        if a.busy.is_some() {
            return false;
        }
        // the rate flag (module doc, "Orbitals"; Task 4 review I1): a refresh — or a Harris job, the
        // same request — is forced only when the driver forces the view (paused), an audit never,
        // so the auditor teacher's `UnoccupiedCadence` budgets the unoccupied eigensolve against
        // its own SCF wall
        let force = matches!(kind, JobKind::Refresh | JobKind::HarrisDisplay) && self.orbital_view.force_every_step;
        let view = OrbitalViewRequest { force_every_step: force, ..self.orbital_view.clone() };
        let draft_guess = std::mem::take(&mut self.dispatch_draft_guess);
        let flags = std::mem::take(&mut self.dispatch_flags);
        let tick_only = std::mem::take(&mut self.dispatch_tick_only);
        let job = AuditJob { scene, prov, generation: a.generation(), answered, kind, id, view, interrupt: self.display_interrupt.clone(), core: self.display_core, draft_guess, shares: self.share_tables.clone(), harris, flags, tick_only };
        match a.jobs.as_ref().map(|tx| tx.send(job)) {
            Some(Ok(())) => {
                a.busy = Some(kind);
                true
            }
            _ => {
                // the thread is gone: nothing queued can run (review minor 4: else
                // `side_job_in_flight` would hold forever); the references asked for are answered
                self.auditor = None;
                self.audit_due = false;
                self.skip_tick_in_flight();
                self.fail_references(None, "the background teacher is gone");
                self.side_queue.clear();
                self.harris_queued = None;
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
    /// Since trust tightening Task 3 the point carries `flags` (the step's cadence-fresh, fenced
    /// and tick bits) and `energy_std_epistemic`, the total-energy epistemic std
    /// (`epistemic_energy_std`; module doc, "The history's flags") — recorded only when finite and
    /// non-negative (−0 as +0), else absent: the history refuses anything else with a panic, and
    /// a panic here would take the provider down with its worker thread.
    ///
    /// # Panics
    /// If the shapes disagree with the geometry (a programming error at the call site: every
    /// judged comparison leaves a point or a count, so nothing is dropped silently).
    ///
    /// Returns whether a point was pushed (a calibration tick's count reads it).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_history_point(&mut self, kind: PointKind, pos: &[[f64; 3]], z: &[u8], stds: Option<(f64, &[f64], &[f64])>, model_energy: f64, model_forces: &[[f64; 3]], energy: f64, forces: &[[f64; 3]], box_affected: bool, draft: bool, unfit: bool, flags: surrogate::history::PointFlags, energy_std_epistemic: Option<f64>) -> bool {
        if box_affected {
            self.audit_history.unrecorded.box_affected += 1;
            return false;
        }
        if draft {
            self.audit_history.unrecorded.draft += 1;
            return false;
        }
        if unfit {
            self.audit_history.unrecorded.unfit += 1;
            return false;
        }
        let Some((energy_std, force_std, force_std_epistemic)) = stds else {
            self.audit_history.unrecorded.no_variance += 1;
            return false;
        };
        // a scene with stray atoms is recorded over its bonded part (Task 5 fix round 1, I3): calibration's energy
        // evidence is per atom over the point's atoms (`calibrate`, e2 = (err_E / n_atoms / tier)²), and the stray
        // atoms are no evidence (no feature, exactly 0 force std); the energy std is the bonded part's bit for bit
        // (`Model::predict`'s floor per energy-row atom, ruling 9a) and the energy error the scene's
        if let Some(sp) = self.model.as_ref().and_then(|m| super::stray::split(pos, m.basis.r_cut)).filter(|_| z.len() == pos.len()) {
            let keep = |v: &[f64]| sp.bonded.iter().map(|&i| v[i]).collect::<Vec<f64>>();
            let keep3 = |v: &[[f64; 3]]| sp.bonded.iter().map(|&i| v[i]).collect::<Vec<[f64; 3]>>();
            let (bp, bz) = sp.bonded_part(pos, z);
            let (fs, fe) = (keep(force_std), keep(force_std_epistemic));
            return self.record_history_point(kind, &bp, &bz, Some((energy_std, &fs, &fe)), model_energy, &keep3(model_forces), energy, &keep3(forces), false, false, false, flags, energy_std_epistemic);
        }
        let n = pos.len();
        assert!(model_forces.len() == n && forces.len() == n, "a comparison's forces agree with its geometry");
        let rc2 = self.model.as_ref().map_or(0.0, |m| m.basis.r_cut * m.basis.r_cut);
        let classes = neighbour_counts(pos, rc2).into_iter().map(|c| atom_class(c).map_or(0, |k| k as u8 + 1)).collect();
        // per atom (Z, its neighbours' species): the class verdict's key (user-test fix 2)
        let species = if z.len() == n { neighbour_species(pos, z, rc2) } else { Vec::new() };
        let err_force = forces.iter().zip(model_forces).map(|(t, m)| ((t[0] - m[0]).powi(2) + (t[1] - m[1]).powi(2) + (t[2] - m[2]).powi(2)).sqrt()).collect();
        // |F_teacher| per atom: the per-atom bound's input (M8a Task 2 fix round 1, review A.1)
        let teacher_force = forces.iter().map(|t| (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt()).collect();
        // |F_model| per atom: the input of rule 6's bound, the same expression as `variance_check`'s,
        // so λ_T admits exactly the atoms rule 6 would (Task 5 fix round 1, review I1); recorded only
        // when every magnitude is a finite number (`push` refuses another), else absent
        let model_force: Vec<f64> = model_forces.iter().map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()).collect();
        let model_force = if model_force.iter().all(|f| f.is_finite()) { model_force } else { Vec::new() };
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
            species,
            // the fit that judged it (trust tightening Task 2; `fit_id`: a point without one does
            // not count toward the floor); the step's flags and the epistemic energy std (Task 3)
            fit_id: self.model.is_some().then_some(self.fit_id),
            flags,
            energy_std_epistemic: recordable_energy_std(energy_std_epistemic),
            model_force,
        });
        // new verdict evidence re-judges the class verdicts now, not at the next refit (user-test
        // fix 2, the controller's ruling 3: a parked scene whose fallbacks are all "similar" never
        // refits, and its evidence would otherwise never be read)
        if super::calibrate::in_jurisdiction(kind) {
            self.rejudge_classes();
        }
        true
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
        // a composition check in flight is the old model's: its answers come back of an older
        // generation (discarded), and its queued parts are dropped (`stray`) — an attribution
        // check's failed audit too, without counting it (nothing is booked; Task 5 fix round 1, I1)
        self.discard_stray_check();
        self.audit_due = false;
        self.box_audits_unjudged = 0;
        self.audit_flip = false;
        self.last_audit = None;
        self.failed_audits.clear();
        self.audit_fences.clear();
        self.audit_burst = 0;
        // an abandoned tick counts as skipped (module doc, "Calibration ticks")
        self.skip_tick_in_flight();
        self.streak = 0;
        self.streak_reason = None;
        self.streak_gap = None;
        self.streak_box = 0;
        self.streak_box_face = 0.0;
    }

    /// Hands an audit that fell due to the background auditor (`surrogate` module doc, "Audit"); `false`
    /// when there is none (the audit then runs in the step). A busy auditor defers it to the
    /// next answered step. A calibration tick at the step (`dispatch_flags` carries `TICK`; module
    /// doc, "Calibration ticks") rides on the same job, or is the job alone (`dispatch_tick_only`):
    /// it never sets or clears rule 8's `audit_due`, and is never deferred — a busy auditor skips it
    /// (which `evaluate` has already decided before the variance: it dispatches a tick only to a
    /// free auditor).
    pub(super) fn dispatch_audit(&mut self, scene: &Scene, prov: &Provenance) -> bool {
        let every = self.policy.audit_every;
        let tick = self.dispatch_flags.has(surrogate::history::PointFlags::TICK);
        let tick_only = self.dispatch_tick_only;
        let Some(a) = self.auditor.as_ref() else { return false };
        if a.busy.is_some() {
            if !tick_only {
                self.audit_due = true;
            }
            if tick {
                self.tick_outcome(super::TickOutcome::SkippedBusy);
            }
            return true;
        }
        self.side_jobs += 1;
        if self.send_job(scene.clone(), prov.clone(), JobKind::Audit, self.side_jobs, None) {
            if !tick_only {
                self.audit_due = false;
                self.pending_notes.push(format!("surrogate audit dispatched to the background teacher (every {every} answered steps; the model answers meanwhile)"));
            }
            if tick {
                self.tick_in_flight = true;
                self.pending_notes.push(format!("surrogate calibration tick dispatched to the background teacher (every {} in-envelope steps; the model answers meanwhile)", self.tick_every));
            }
            true
        } else {
            // the thread is gone (`send_job` dropped it): audit in the step from now on
            self.side_jobs -= 1;
            false
        }
    }

    /// A calibration tick's job on the auditor will not be judged — abandoned by a Forget or an
    /// installed model, or its thread gone: lost (module doc, "Calibration ticks"), and the next
    /// tick's count starts here.
    pub(super) fn skip_tick_in_flight(&mut self) {
        if std::mem::take(&mut self.tick_in_flight) {
            self.tick_outcome(super::TickOutcome::Lost);
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
                // every job's wall time, whatever its kind and outcome: the ticks' dead time
                self.time_auditor_job(ans.ms);
                Some(ans)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                let busy = a.busy;
                self.auditor = None;
                self.audit_due = false;
                self.skip_tick_in_flight();
                self.fail_references(busy, "the background teacher stopped");
                // nothing queued can run any more (review minor 4)
                self.side_queue.clear();
                self.harris_queued = None;
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
            // the job's record: who asked and which collection its scene shared (fix round 3); an
            // answer without one (it cannot happen) is handed on and offered nowhere
            let n = ans.job.scene.cell.n;
            let job = self.ref_jobs.iter().position(|j| j.z == z && j.n == n).map(|k| self.ref_jobs.remove(k));
            let (driver, collection) = job.map_or((true, None), |j| (j.driver, j.collection));
            // offered to the collection only when the requesting scene shared it — and it is still
            // the one attached (`shared` module doc, "References per box"; review 1: never another
            // surface's atom)
            let ours = collection.is_some() && self.shared.as_ref().map(|s| s.coll.key.slug()) == collection;
            if let (Ok(a), true) = (&r, ours) {
                self.set_reference_solved(z, a.energy, n, a.face_density);
            }
            if driver {
                return self.atom_references.push((z, r));
            }
            if let Err(e) = &r {
                self.user_notes.push(format!("shared collection: the {} reference of the {n}³ box did not solve ({e}) — its answers keep waiting (a sampling pass solves it too)", symbol(z)));
            }
            return;
        }
        let Some(a) = self.auditor.as_ref() else { return };
        if ans.job.generation != a.generation() {
            return; // the model it was for was forgotten or replaced
        }
        // a composition check's part (`stray`): to the check in flight, never a sample
        if matches!(ans.job.kind, JobKind::Fragment | JobKind::AtomAt(_)) {
            return self.take_stray_answer(ans);
        }
        let positions = || ans.job.scene.nuclei.iter().map(|n| n.pos).collect::<Vec<[f64; 3]>>();
        // a calibration tick's answer, of this generation (an older one's was counted lost when it
        // was abandoned): taken once judged — recorded or not — else lost (module doc, "Calibration
        // ticks"); the next tick's count starts here
        let tick = ans.job.kind == JobKind::Audit && ans.job.flags.has(surrogate::history::PointFlags::TICK);
        if tick {
            self.tick_in_flight = false;
        }
        match (ans.job.kind, ans.result) {
            (JobKind::Atom(z), Ok(Solved { display: Some(d), .. })) => return self.atom_densities.push((z, d)),
            (JobKind::Refresh, Ok(Solved { display: Some(d), orbitals, .. })) => {
                self.side_display = Some(SideDisplay { display: Some(d), orbitals, kind: SideKind::Refresh, job: ans.job.id, positions: positions() });
                return;
            }
            (JobKind::Reference(z), _) => {
                return self.atom_references.push((z, Err(ProviderError::Invalid("the background teacher returned no reference".into()))));
            }
            (JobKind::Fragment | JobKind::AtomAt(_), _) => unreachable!("routed to take_stray_answer above"),
            // display only (G4): a payload, else a note — never a teach event or a counter
            (JobKind::HarrisDisplay, Ok(Solved { orbitals: Some(o), .. })) => {
                self.side_harris = Some(SideDisplay { display: None, orbitals: Some(o), kind: SideKind::Harris, job: ans.job.id, positions: positions() });
                return;
            }
            (JobKind::HarrisDisplay, Ok(_)) => {
                return self.pending_notes.push("surrogate: the orbital display solve on the background teacher returned no orbitals".into());
            }
            (JobKind::HarrisDisplay, Err(e)) if e == HARRIS_CANCELLED => return,
            (JobKind::HarrisDisplay, Err(e)) => {
                return self.pending_notes.push(format!("surrogate: the orbital display solve (not self-consistent) on the background teacher did not finish ({e})"));
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
                if tick {
                    self.tick_outcome(super::TickOutcome::Lost);
                }
                return self.pending_notes.push(note);
            }
            (JobKind::Audit, Ok(Solved { energy, forces, face, multiplicity, display, orbitals, unfit, .. })) => {
                if let Some(d) = display {
                    self.side_display = Some(SideDisplay { display: Some(d), orbitals, kind: SideKind::Audit, job: ans.job.id, positions: positions() });
                }
                let judged = self.judge_audit(ans.job, energy, forces, face, multiplicity, unfit, ans.ms, ans.scf_steps, ans.ts);
                if tick {
                    self.tick_outcome(match judged {
                        None => super::TickOutcome::Lost,
                        Some(true) => super::TickOutcome::Recorded,
                        Some(false) => super::TickOutcome::Unrecorded,
                    });
                }
            }
        }
    }

    /// A background audit's verdict (`surrogate` module doc, "Audit"): the sample is admitted like any
    /// teacher answer, the gap recorded; a failed audit untrusts the model and books the flip at
    /// the next evaluation. One activity event either way: judged (the verdict, the admission's
    /// outcome) or discarded (plan review S11). Returns `None` when the answer was not judged
    /// (discarded: the model changed settings or species; inside a fenced shell band), else whether
    /// its history point was recorded (`record_history_point`: not for a box-affected, unfit or
    /// variance-less answer) — what a calibration tick's count reads.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn judge_audit(&mut self, job: AuditJob, energy: f64, forces: Vec<[f64; 3]>, face: Option<f64>, multiplicity: Option<u32>, unfit: Option<&str>, ms: f64, scf_steps: Option<usize>, ts: Option<f64>) -> Option<bool> {
        let unaudited = self.answered.saturating_sub(job.answered);
        let AuditJob { scene, prov, draft_guess, flags: dispatched, tick_only, .. } = job;
        let tag = if draft_guess { SampleTag::DraftAudit } else { SampleTag::Audit };
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        // the canonical surface in collection mode (`shared` module doc; the identity per scene)
        let raw_energy = energy;
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
            ev.n_atoms = self.audit_n(&pos);
            ev.solve_ms = ms;
            ev.scf_steps = scf_steps;
            ev.ts = ts;
            ev.scf_end = scf_steps.filter(|&n| n > 0).map(|_| sim::ScfEnd::Converged);
            ev.audit_lag = Some(unaudited);
            self.push_event(ev);
            self.pending_notes.push(note.into());
            return None;
        };
        // at a cleared stray-atom scene the fences judge the bonded part alone, as `evaluate`'s do
        // (`policy::judged_part`; the smearing branch's final review, M1): the FENCED bit and the
        // keep decision of a background tick are the in-step ones
        let cleared = self.stray_cleared(&pos, &z, &prov);
        let part = cleared.as_ref().map(|c| &c.0);
        // inside a fenced shell band (the shell fence; final review I1, the coordinator's ruling):
        // strict mode refuses the geometry, so the answer is no judgement of the model — not
        // judged, not kept (a draft guess's included: a band row is never fitted, now or after a
        // later shell check lifts the fence)
        if let Some(why) = self.shell_fence_refusal_judged(&pos, &z, part) {
            let note = format!("surrogate audit not judged: {} — not a judgement of the model; the answer is not fitted", why.text.trim_start_matches("surrogate: "));
            let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, TeachOutcome::ShellNotKept);
            ev.detail = note.clone();
            ev.n_atoms = self.audit_n(&pos);
            ev.solve_ms = ms;
            ev.scf_steps = scf_steps;
            ev.ts = ts;
            ev.scf_end = scf_steps.filter(|&n| n > 0).map(|_| sim::ScfEnd::Converged);
            ev.audit_lag = Some(unaudited);
            self.push_event(ev);
            self.pending_notes.push(note);
            return None;
        }
        // per bonded atom at a scene with stray atoms (`stray` module doc, "Audits of a cleared answer"), else per atom
        let n = self.audit_n(&pos) as f64;
        let unit = self.per_atom_unit(&pos);
        let gap = energy - pred.energy;
        let near = super::policy::judged_part(&pos, &z, part, |p, q| self.near_quarantined(p, q));
        // next to a misfit the answer is kept only if it agrees with the model (spec 2026-11-13 §4.3)
        let near_misfit = super::policy::judged_part(&pos, &z, part, |p, q| self.near_misfit(p, q)).contains(&true);
        // the history's flags (module doc, "The history's flags"): the dispatching step's, fenced
        // as well when a fence applies at the judgement (read-only; the shell fence returned above)
        let fenced = !near.is_empty() || near_misfit || self.audit_fence_refusal_judged(&pos, &z, part).is_some();
        let flags = surrogate::history::PointFlags(dispatched.0 | if fenced { surrogate::history::PointFlags::FENCED } else { 0 });
        let energy_std_epistemic = pred.energy_std.zip(self.model.as_ref()).map(|(e, m)| epistemic_energy_std(m, surrogate::energy_row_atoms(&m.basis, &pos), e));
        let mut notes = Vec::new();
        let rule = box_affected_face_density(prov.h);
        // (never a capped segment's: a masked scene is teacher-only, nothing is audited there)
        let outcome = self.admit_and_keep(&pos, &z, energy, raw_energy, &forces, face, multiplicity, Some((pred.energy, &pred.forces)), &near, near_misfit, true, false, false, false, &prov, unfit, tag, None, &mut notes);
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
        let mut stray_note = None;
        // an audit a check attributes leaves its history point to the check (ruling 9b)
        let mut deferred = false;
        notes.insert(0, match verdict {
            AuditVerdict::Passed => format!("surrogate audit passed (background teacher, judged {unaudited} answered steps after its dispatch): teacher − model {gap:+.2e} Ha"),
            AuditVerdict::BoxAffected => format!(
                "surrogate audit not judged: the teacher's answer is box-affected (face density {:.1e} > {rule:.1e}) — its teacher − model {gap:+.2e} Ha is the box's confinement as much as the model's error{}",
                box_rho.unwrap_or(0.0),
                if tick_only { " (a calibration tick: rule 8's cadence is unchanged)" } else { "; the next answered step is audited again" }
            ),
            // a draft guess's audit (a geometry strict mode refuses): recorded, held out of the
            // strict fit, and not a verdict on the strict model (module doc, "Draft mode")
            AuditVerdict::Failed if draft_guess => format!(
                "surrogate draft-mode audit failed (background teacher): teacher − model {gap:+.2e} Ha ({:.1e} {unit}{}) at a geometry strict mode refuses — the guess was off; the strict model's trust is unchanged and the answer is held out of its fit",
                gap.abs() / n,
                force_fail.map_or(String::new(), |(d, b)| format!(", worst atom's force off by {d:.1e} > {b:.1e} Ha/Bohr"))
            ),
            AuditVerdict::Failed => {
                let index = outcome.is_kept().then(|| self.samples().saturating_sub(1));
                let case = FailedAudit { pos: pos.clone(), z: z.clone(), energy, forces: forces.clone(), index, ionic: prov.ionic.clone(), rejudges: 0, origin: AuditOrigin::Audit, pair: None };
                // at a scene with stray atoms: the signature refused, its samples held, and a check attributes the
                // failure (`stray` module doc, "Audits of a cleared answer")
                let why = match force_fail {
                    Some((d, b)) if gap.abs() / n <= thr => format!("worst atom's force off by {d:.1e} > {b:.1e} Ha/Bohr"),
                    _ => format!("teacher − model {gap:+.2e} Ha, {:.1e} > {thr:.1e} Ha per bonded atom", gap.abs() / n),
                };
                let stray = self.stray_audit_failed(&scene, raw_energy, &forces, &case, gap, why, flags);
                let attributing = matches!(stray, stray::StrayAudit::Attributing(_));
                deferred = attributing;
                if let stray::StrayAudit::Attributing(note) | stray::StrayAudit::CountsNow(note) = stray {
                    stray_note = Some(note);
                }
                if !attributing {
                    self.audit_failed = Some(gap);
                    self.audit_force = force_fail;
                    self.audit_atoms = n as usize;
                    self.failed_audits.push(case);
                    self.audit_flip = self.last.as_ref().is_some_and(|l| l.on_model);
                }
                match force_fail {
                    _ if attributing => format!(
                        "surrogate audit failed (background teacher) at a scene with stray atoms: teacher − model {gap:+.2e} Ha ({:.1e} Ha per bonded atom{}) — a check of the bonded part alone says whether the model or the composition failed",
                        gap.abs() / n,
                        force_fail.map_or(String::new(), |(d, b)| format!(", worst atom's force off by {d:.1e} > {b:.1e} Ha/Bohr"))
                    ),
                    Some((d, b)) if gap.abs() / n <= thr => format!(
                        "surrogate audit failed on forces (background teacher): worst atom off by {d:.1e} > {b:.1e} Ha/Bohr (teacher − model {gap:+.2e} Ha) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until a refit predicts that geometry without the audited sample"
                    ),
                    _ => format!(
                        "surrogate audit failed (background teacher): teacher − model {gap:+.2e} Ha ({:.1e} > {thr:.1e} {unit}) — the {unaudited} answered steps since its dispatch were answered by an unaudited model; untrusted until a refit predicts that geometry without the audited sample",
                        gap.abs() / n
                    ),
                }
            }
        });
        notes.extend(stray_note);
        self.last_audit = Some(AuditRecord { gap, verdict, face_density: face, unaudited_steps: unaudited, asynchronous: true, teacher_ms: ms });
        // the prediction judged above is the current model's, so the stamp is this fit's
        self.last_audit_fit = self.fits;
        // a box-affected audit judged nothing (M8a Task 2, the Task 1 physics review's I4): it
        // does not satisfy the cadence — the next answered step dispatches another — and the
        // checklist counts it; the model's surface and trust are untouched (the gap is the box's).
        // A calibration tick alone is not rule 8's audit: neither satisfied nor missed, its cadence
        // and the checklist's count stay as they were (module doc, "Calibration ticks")
        match (tick_only, verdict == AuditVerdict::BoxAffected) {
            (true, _) => {}
            (false, true) => {
                self.audit_due = true;
                self.box_audits_unjudged += 1;
            }
            (false, false) => self.box_audits_unjudged = 0,
        }
        // the history's point (M8a Task 1)
        let stds = match (pred.energy_std, pred.force_std.as_ref(), pred.force_std_epistemic.as_ref()) {
            (Some(e), Some(f), Some(fe)) => Some((e, &f[..], &fe[..])),
            _ => None,
        };
        // a failed audit at a stray-atom scene that a check attributes (`deferred`) records its point when the check
        // answers (ruling 9b; `stray::wiring`, with this audit's flags carried in its `Attribution`): no point now, so
        // a calibration tick on it reads "taken, unrecorded"
        let recorded = !deferred && self.record_history_point(PointKind::Audit, &pos, &z, stds, pred.energy, &pred.forces, energy, &forces, box_rho.is_some(), draft_guess, unfit.is_some(), flags, energy_std_epistemic);
        let mut ev = TeachEvent::new(TeachOrigin::Audit, TeachRule::Audit, outcome);
        ev.detail = notes.join("\n");
        ev.face_density = face;
        ev.box_affected = box_rho.is_some();
        ev.gap = Some(gap);
        // the atoms the verdict's per-atom gap divides by (N_e: `audit_n`), the Activity hover's count
        ev.n_atoms = n as usize;
        ev.solve_ms = ms;
        ev.scf_steps = scf_steps;
        ev.ts = ts;
        ev.scf_end = scf_steps.filter(|&n| n > 0).map(|_| sim::ScfEnd::Converged);
        ev.audit = Some(verdict);
        ev.audit_lag = Some(unaudited);
        self.push_event(ev);
        self.pending_notes.extend(notes);
        Some(recorded)
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
            train_mask: None,
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
