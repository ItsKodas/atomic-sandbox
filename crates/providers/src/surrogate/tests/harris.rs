//! Live-orbitals Task 5 (spec §4.2): the background auditor's Harris display job —
//! `JobKind::HarrisDisplay`, at most one queued (the newest wins), due audits and the side queue
//! first, the auditor's teacher never reset or warm-state-touched by it, never a fallback, audit,
//! teacher solve, sample or `TeachEvent` (G4); and the isolated-atom jobs solved without the
//! display's partial core (spec §4.1).
use super::*;
use sim::{HarrisAnswer, HarrisAtoms, OrbitalKind, OrbitalRequest, OrbitalViewRequest};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

/// Serialises this module's GPU tests among themselves (poisoning is harmless here).
static GPU: Mutex<()> = Mutex::new(());

/// The bond length of a two-atom scene (the mock teacher's log key).
fn bond(scene: &Scene) -> f64 {
    let (a, b) = (scene.nuclei[0].pos, scene.nuclei[1].pos);
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A gated Morse auditor teacher that also answers `harris_display` (a stand-in payload of kind
/// `Harris` at the job's positions), logging each job — ("solve" or "harris", its bond length) —
/// in the order it ran. It answers only once `open` is set (or gives up when told to abort).
struct HarrisGated {
    inner: Arc<Mutex<MorseDimer>>,
    open: Arc<AtomicBool>,
    log: Arc<Mutex<Vec<(&'static str, f64)>>>,
}

impl HarrisGated {
    fn wait(&self, should_abort: &mut dyn FnMut() -> bool) -> bool {
        while !self.open.load(SeqCst) {
            if should_abort() {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        true
    }
}

impl ForceProvider for HarrisGated {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        if !self.wait(should_abort) {
            return Err(ProviderError::Cancelled);
        }
        self.log.lock().unwrap().push(("solve", bond(scene)));
        let mut r = self.inner.lock().unwrap().evaluate(scene, should_abort)?;
        r.display = Some(Display::Cpu { cell: scene.cell.clone(), origin: [0.0; 3], rho: Arc::new(vec![1.0]), max: 1.0 });
        Ok(r)
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
    fn harris_display(&mut self, scene: &Scene, _atoms: &Arc<HarrisAtoms>, should_abort: &mut dyn FnMut() -> bool) -> Option<HarrisAnswer> {
        if !self.wait(should_abort) {
            return Some(HarrisAnswer::Solved(Err(ProviderError::Cancelled)));
        }
        self.log.lock().unwrap().push(("harris", bond(scene)));
        Some(HarrisAnswer::Solved(Ok(sim::OrbitalDisplay {
            cell: scene.cell.clone(), origin: [0.0; 3], positions: scene.nuclei.iter().map(|n| n.pos).collect(), rows: Vec::new(),
            field: None, spin: sim::SpinSummary::default(), unoccupied: None, notes: Vec::new(), what: OrbitalRequest::List,
            lone_atom: sim::LoneAtomLabels::NotLoneAtom, share_z: None, kind: OrbitalKind::Harris { experience: None, rescale: 1.0, wall_s: 0.0 },
        })))
    }
}

#[allow(clippy::type_complexity)]
fn harris_gated(s: SurrogateKs, inner: Arc<Mutex<MorseDimer>>) -> (SurrogateKs, Arc<AtomicBool>, Arc<Mutex<Vec<(&'static str, f64)>>>) {
    let open = Arc::new(AtomicBool::new(false));
    let log = Arc::new(Mutex::new(Vec::new()));
    let auditor = HarrisGated { inner, open: open.clone(), log: log.clone() };
    (s.with_auditor(Box::new(auditor)), open, log)
}

/// No profiles: the mock never reads them.
fn no_atoms() -> Arc<HarrisAtoms> {
    Arc::new(HarrisAtoms { profiles: Vec::new() })
}

/// `harris_display`'s side-job number (the surrogate always queues).
fn queued(s: &mut SurrogateKs, scene: &Scene, atoms: &Arc<HarrisAtoms>) -> u64 {
    match s.harris_display(scene, atoms, &mut never()) {
        Some(HarrisAnswer::Queued { job }) => job,
        other => panic!("the surrogate queues a Harris job: {other:?}"),
    }
}

/// Takes side displays until `f` accepts one (≤ `secs`), returning everything taken.
fn take_until(s: &mut SurrogateKs, secs: f64, f: impl Fn(&SideDisplay) -> bool) -> Vec<SideDisplay> {
    let t0 = Instant::now();
    let mut seen = Vec::new();
    loop {
        while let Some(d) = s.take_side_display() {
            let done = f(&d);
            seen.push(d);
            if done {
                return seen;
            }
        }
        assert!(t0.elapsed().as_secs_f64() < secs, "not within {secs} s: {:?}", seen.iter().map(|d| (d.kind, d.job)).collect::<Vec<_>>());
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Spec §4.2 "Scheduling": one Harris job is queued at most and a newer geometry replaces the
/// queued one. The auditor is busy with a refresh; three requests at R = 1.40, 1.45, 1.50 → the
/// teacher solves exactly one Harris, at 1.50, and exactly one Harris payload arrives, numbered
/// as the last request.
#[test]
fn a_newer_harris_request_replaces_the_queued_one() {
    let (s, h) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let (mut s, open, log) = harris_gated(s, h);
    assert!(s.request_display(&h2(1.6)), "the refresh occupies the auditor");
    assert!(!s.auditor_idle());
    let atoms = no_atoms();
    let jobs: Vec<u64> = [1.40, 1.45, 1.50].iter().map(|&r| queued(&mut s, &h2(r), &atoms)).collect();
    assert!(jobs.windows(2).all(|w| w[1] > w[0]), "each request is a new side job: {jobs:?}");
    assert!(Trainable::side_job_in_flight(&s), "queued");
    open.store(true, SeqCst);
    let seen = take_until(&mut s, 2.0, |d| d.kind == SideKind::Harris);
    // anything later would be a second Harris answer
    std::thread::sleep(std::time::Duration::from_millis(200));
    let mut seen = seen;
    while let Some(d) = s.take_side_display() {
        seen.push(d);
    }
    let harris: Vec<&SideDisplay> = seen.iter().filter(|d| d.kind == SideKind::Harris).collect();
    println!("taken {:?}; the teacher ran {:?}", seen.iter().map(|d| (d.kind, d.job)).collect::<Vec<_>>(), log.lock().unwrap());
    assert_eq!(harris.len(), 1, "exactly one Harris answer");
    let h = harris[0];
    assert_eq!(h.job, jobs[2], "the newest request's");
    assert!((bond(&Scene { nuclei: h.positions.iter().map(|&p| Nucleus::new(1, p)).collect(), ..h2(1.5) }) - 1.50).abs() < 1e-12);
    assert!(h.display.is_none() && h.orbitals.as_ref().is_some_and(|o| matches!(o.kind, OrbitalKind::Harris { .. })));
    assert_eq!(*log.lock().unwrap(), vec![("solve", 1.6), ("harris", 1.5)], "one refresh, then one Harris at the newest geometry");
    assert!(!Trainable::side_job_in_flight(&s), "nothing left");
}

/// Spec §4.2 "Scheduling": an audit that fell due while the auditor was busy goes before a queued
/// Harris job — a refresh in flight, an answered step whose audit falls due (deferred: `audit_due`),
/// a Harris queued; when the refresh frees the auditor the Harris is held, the next answered step
/// dispatches the audit, and only then does the Harris run.
#[test]
fn a_due_audit_goes_before_harris() {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 1, ..FallbackPolicy::default() });
    let (mut s, open, log) = harris_gated(s, h);
    assert!(s.request_display(&h2(1.6)));
    let r = s.evaluate(&h2(1.45), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert!(s.audit_due, "the audit fell due while the auditor was busy");
    let job = queued(&mut s, &h2(1.40), &no_atoms());
    open.store(true, SeqCst);
    take_until(&mut s, 2.0, |d| d.kind == SideKind::Refresh);
    assert!(s.auditor_idle(), "the refresh is done");
    std::thread::sleep(std::time::Duration::from_millis(50));
    s.poll_side_jobs();
    assert!(s.auditor_idle() && s.harris_queued.is_some(), "the Harris waits for the due audit");
    assert_eq!(*log.lock().unwrap(), vec![("solve", 1.6)]);
    // the next answered step dispatches the due audit; the Harris follows it
    let r = s.evaluate(&h2(1.47), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(!s.audit_due && s.audit_in_flight(), "the audit was dispatched");
    let seen = take_until(&mut s, 2.0, |d| d.kind == SideKind::Harris);
    assert_eq!(seen.last().map(|d| d.job), Some(job));
    let ran = log.lock().unwrap().clone();
    println!("the teacher ran {ran:?}");
    assert_eq!(ran.len(), 3);
    assert_eq!((ran[1].0, ran[2]), ("solve", ("harris", 1.40)), "the audit first, then the Harris: {ran:?}");
    assert!((ran[1].1 - 1.47).abs() < 1e-12, "the audit is the step's geometry: {ran:?}");
}

/// Review fix round 1, Important 1 (spec G1 over the brief's literal rule): **paused** (the
/// driver's `force_every_step`) a queued Harris job does not wait for a due audit — only an
/// answered step dispatches that, and paused there may be none — so the Harris runs when the
/// auditor frees and the audit goes at the next evaluate. (Playing, the audit goes first:
/// [`a_due_audit_goes_before_harris`].)
#[test]
fn a_paused_harris_does_not_wait_for_a_due_audit() {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 1, ..FallbackPolicy::default() });
    let (mut s, open, log) = harris_gated(s, h);
    s.set_orbital_view(OrbitalViewRequest { force_every_step: true, ..OrbitalViewRequest::new(OrbitalRequest::List) });
    assert!(s.request_display(&h2(1.6)));
    let r = s.evaluate(&h2(1.45), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    assert!(s.audit_due, "the audit fell due while the auditor was busy");
    let job = queued(&mut s, &h2(1.40), &no_atoms());
    open.store(true, SeqCst);
    let seen = take_until(&mut s, 2.0, |d| d.kind == SideKind::Harris);
    assert_eq!(seen.last().map(|d| d.job), Some(job));
    assert_eq!(*log.lock().unwrap(), vec![("solve", 1.6), ("harris", 1.40)], "paused: the Harris after the refresh");
    assert!(s.audit_due, "the audit still waits for an answered step");
    let r = s.evaluate(&h2(1.47), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate);
    assert!(!s.audit_due && s.audit_in_flight(), "the next answered step dispatched it");
}

/// Review fix round 1, minor 3: a Harris solve cancelled by the display interrupt (the user's
/// next command; or a newer generation) leaves no note and no payload — a superseded solve is not
/// news.
#[test]
fn a_cancelled_harris_leaves_no_note() {
    let (s, h) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let (mut s, _open, log) = harris_gated(s, h);
    s.set_display_interrupt(Arc::new(|| true));
    queued(&mut s, &h2(1.40), &no_atoms());
    let t0 = Instant::now();
    while Trainable::side_job_in_flight(&s) {
        assert!(s.take_side_display().is_none(), "no payload");
        assert!(t0.elapsed().as_secs_f64() < 2.0, "the cancelled job did not come back");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(log.lock().unwrap().is_empty(), "nothing was solved");
    assert!(s.pending_notes.is_empty(), "no note: {:?}", s.pending_notes);
}

/// Final review M2: a Harris payload polled in (the worker's per-round `take_atom_densities`
/// polls the auditor) but not yet taken by the driver still counts as in flight — the auditor is
/// idle by then, and the worker's lost-job check would otherwise re-send the same geometry.
#[test]
fn a_polled_harris_payload_is_in_flight_until_taken() {
    let (s, h) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let (mut s, open, _log) = harris_gated(s, h);
    let job = queued(&mut s, &h2(1.40), &no_atoms());
    open.store(true, SeqCst);
    let t0 = Instant::now();
    while s.side_harris.is_none() {
        assert!(s.take_atom_densities().is_empty(), "no atom job was asked for");
        assert!(t0.elapsed().as_secs_f64() < 2.0, "the Harris payload did not arrive");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(s.auditor_idle() && s.harris_queued.is_none() && s.side_queue.is_empty(), "nothing else in flight");
    assert!(Trainable::side_job_in_flight(&s), "the payload is not in the driver yet");
    let d = s.take_side_display().expect("the Harris payload");
    assert_eq!((d.kind, d.job), (SideKind::Harris, job));
    assert!(!Trainable::side_job_in_flight(&s), "taken");
}

/// A gated auditor teacher whose thread dies once the gate opens: its job then fails, and the next job's preparation
/// (`discontinuity`, outside the job) panics. (A panic *inside* a job no longer ends the thread: the auditor catches it
/// and answers the job with an error — the gpu-fault fix.)
struct Dies {
    open: Arc<AtomicBool>,
}

impl ForceProvider for Dies {
    fn evaluate(&mut self, _scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        while !self.open.load(SeqCst) {
            if should_abort() {
                return Err(ProviderError::Cancelled);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        Err(ProviderError::Invalid("the test's auditor teacher is dying".into()))
    }
    fn name(&self) -> &'static str {
        "morse dimer"
    }
    fn reset(&mut self) {}
    fn discontinuity(&mut self) {
        if self.open.load(SeqCst) {
            panic!("the test's auditor teacher dies (expected)");
        }
    }
}

/// Review fix round 1, minor 4: when the auditor's thread is gone, the queued Harris job (and the
/// side queue) go with it, so `side_job_in_flight` does not hold forever (the worker would poll
/// for ever).
#[test]
fn a_gone_auditor_drops_the_queued_harris() {
    let (s, _h) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let open = Arc::new(AtomicBool::new(false));
    let mut s = s.with_auditor(Box::new(Dies { open: open.clone() }));
    assert!(s.request_display(&h2(1.6)));
    queued(&mut s, &h2(1.40), &no_atoms());
    assert!(s.request_display(&h2(1.5)), "a refresh queued behind the running one");
    assert!(s.harris_queued.is_some() && !s.side_queue.is_empty());
    open.store(true, SeqCst);
    let t0 = Instant::now();
    while s.audits_asynchronous() {
        s.take_side_display();
        assert!(t0.elapsed().as_secs_f64() < 2.0, "the dead thread was not noticed");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(s.harris_queued.is_none() && s.side_queue.is_empty(), "nothing left to run");
    assert!(!Trainable::side_job_in_flight(&s));
}

/// Review fix round 1, minor 5: starting a sampling pass drops the queued Harris job (the worker
/// asks again after the pass), and no new one is accepted during it.
#[test]
fn a_pass_drops_the_queued_harris() {
    let (s, h) = trained(1.2, 2.2, 0, FallbackPolicy::default());
    let (mut s, _open, _log) = harris_gated(s, h);
    assert!(s.request_display(&h2(1.6)));
    queued(&mut s, &h2(1.40), &no_atoms());
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &h2(1.45), &sim::SamplingPlan::default(), &mut refs).expect("a pass");
    assert!(s.harris_queued.is_none(), "dropped at the pass's start");
    assert!(s.harris_display(&h2(1.45), &no_atoms(), &mut never()).is_none(), "none during a pass");
}

/// Review fix round 2: every pass entry point goes through `begin_pass`, which drops the queued
/// Harris job — here the teach-around pass (`start_teach`, the gap fixture's blocked pair class),
/// with a Harris job planted in the queue (the fixture has no auditor to queue one through).
#[test]
fn a_teach_around_pass_drops_the_queued_harris() {
    let (mut s, _h) = gap_blocked(2.0, 3.6);
    s.harris_queued = Some((h2(1.4), no_atoms(), 99));
    s.start_teach(&in_big_cell(h2(1.6))).expect("a teach-around pass");
    assert!(s.pass.is_some(), "premise: a pass runs");
    assert!(s.harris_queued.is_none(), "dropped at the pass's start");
}

/// The analytic H 1s profile e^{−2r}/π on `s`'s surface (the ks tests' stand-in for the
/// auditor's own isolated-atom profile).
fn h_atoms(s: &Scene) -> Arc<HarrisAtoms> {
    let (n, h) = (s.cell.n, s.cell.h);
    let grid = qm::Grid::cube(n, h);
    let rho: Vec<f32> = grid.sample(|p| (-2.0 * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).exp() / std::f64::consts::PI)
        .into_iter().map(|v| v as f32).collect();
    Arc::new(HarrisAtoms { profiles: vec![sim::radial_profile(1, &rho, n, h, grid.origin, s.sigma, s.ionic)] })
}

/// An estimated-mode surrogate on GPU teachers (stepping and auditor), the orbital list on.
fn gpu_surrogate(ctx: &gpu::GpuContext) -> SurrogateKs {
    let teacher = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
    let auditor = crate::ks::GpuKs::new(ctx.clone(), 1e-6);
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6).with_auditor(Box::new(auditor));
    s.set_orbital_view(OrbitalViewRequest::new(OrbitalRequest::List));
    s
}

/// G4 / spec §4.3: a Harris job is never a fallback, an audit, a teacher solve, a sample or a
/// `TeachEvent`. Estimated H₂ on GPU teachers: one fallback first (so the counts are not all 0),
/// then `harris_display` → `Queued`; its payload arrives as a `SideKind::Harris` side display
/// (no density, orbitals of kind `Harris`, the job's number) and the teach events, the audit
/// history, the solve counts and the samples are as before.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn harris_job_is_never_a_fallback_or_audit() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { eprintln!("no GPU adapter — skipped"); return };
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut s = gpu_surrogate(&ctx);
    let r = s.evaluate(&h2(1.4), &mut never()).expect("the GPU teacher");
    assert_eq!(r.stats.source, Source::Fallback, "no model: the teacher answers");
    Trainable::take_teach_events(&mut s);
    let (counts, history, samples) = (s.solve_counts(), s.audit_history().clone(), s.samples());
    assert!(counts.fallback == 1 && samples == 1, "{counts:?}, {samples} samples");
    let scene = h2(1.45);
    let job = queued(&mut s, &scene, &h_atoms(&scene));
    let seen = take_until(&mut s, 60.0, |d| d.kind == SideKind::Harris);
    let d = seen.last().unwrap();
    let o = d.orbitals.as_ref().expect("the Harris payload");
    println!("Harris side display: job {} (asked {job}), kind {:?}, rows {:?}, notes {:?}", d.job, o.kind, o.rows.iter().map(|r| (r.spin, r.energy, r.occupation)).collect::<Vec<_>>(), o.notes);
    assert_eq!((d.kind, d.job), (SideKind::Harris, job));
    assert!(d.display.is_none(), "a Harris job solves no density");
    assert!(matches!(o.kind, OrbitalKind::Harris { .. }) && !o.rows.is_empty(), "{:?}", o.kind);
    assert_eq!(d.positions, scene.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>());
    assert!(Trainable::take_teach_events(&mut s).is_empty(), "no TeachEvent");
    assert_eq!(s.solve_counts(), counts, "no counter changes");
    assert_eq!(*s.audit_history(), history, "no audit point");
    assert_eq!(s.samples(), samples, "no sample");
    assert!(s.last_audit().is_none());
}

/// Sends a display refresh at `scene` and waits for its answer, returning its SCF steps (the
/// answer is then handled as usual and its side display taken).
fn refresh_steps(s: &mut SurrogateKs, scene: &Scene) -> usize {
    assert!(s.request_display(scene));
    let t0 = Instant::now();
    let ans = loop {
        if let Some(a) = s.receive_answer() {
            break a;
        }
        assert!(t0.elapsed().as_secs_f64() < 60.0, "no refresh within 60 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    assert_eq!(ans.job.kind, JobKind::Refresh);
    let steps = ans.scf_steps.expect("a solved refresh");
    s.handle_answer(ans);
    assert_eq!(s.take_side_display().map(|d| d.kind), Some(SideKind::Refresh));
    steps
}

/// Spec §4.2 "No warm-start contamination": the Harris job never resets the auditor's teacher and
/// never touches its SCF warm state. Refreshes at R = 1.4: cold, then warm (n_w steps); a Harris
/// at R + 0.05; a refresh at R again takes exactly n_w steps — the warm state survived (a reset
/// would make it the cold count).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn harris_never_resets_the_auditor_teacher() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { eprintln!("no GPU adapter — skipped"); return };
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut s = gpu_surrogate(&ctx);
    let r = 1.4;
    let cold = refresh_steps(&mut s, &h2(r));
    let warm = refresh_steps(&mut s, &h2(r));
    let scene = h2(r + 0.05);
    let job = queued(&mut s, &scene, &h_atoms(&scene));
    let seen = take_until(&mut s, 60.0, |d| d.kind == SideKind::Harris);
    let o = seen.last().and_then(|d| d.orbitals.as_ref()).expect("the Harris payload");
    assert_eq!(seen.last().map(|d| d.job), Some(job));
    assert!(matches!(o.kind, OrbitalKind::Harris { .. }) && !o.rows.is_empty(), "a Harris solve ran: {:?}", o.notes);
    let after = refresh_steps(&mut s, &h2(r));
    println!("refresh SCF steps at R = {r}: cold {cold}, warm {warm}, after the Harris at {:.2}: {after}", r + 0.05);
    assert!(cold > warm, "the warm count discriminates a reset: cold {cold}, warm {warm}");
    assert_eq!(after, warm, "the Harris left the auditor's warm state alone");
}

/// Spec §4.1: every isolated-atom profile is the valence density without ρ_pc, so the auditor
/// solves `Atom` jobs with the display's partial core off whatever the provider's setting. A
/// pseudo-O (Accepted, 61³ × 0.2) asked for with "show partial core" on integrates to Z_v = 6
/// (± the grid's 1e-2), not 6 + ∫ρ_pc.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
fn atom_jobs_solve_without_the_partial_core() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { eprintln!("no GPU adapter — skipped"); return };
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut s = gpu_surrogate(&ctx);
    s.set_display_partial_core(true);
    let scene = Scene { nuclei: vec![Nucleus::new(8, [0.0; 3])], cell: Cell { n: 61, h: 0.2 }, ionic: sim::IonicPolicy::Accepted, ..h2(1.4) };
    assert!(s.request_atom_density(8, &scene));
    let t0 = Instant::now();
    let d = loop {
        if let Some((z, d)) = s.take_atom_densities().pop() {
            assert_eq!(z, 8);
            break d;
        }
        assert!(t0.elapsed().as_secs_f64() < 300.0, "no O atom within 300 s");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    let (values, cell) = match &d {
        Display::Cpu { rho, cell, .. } => (rho.to_vec(), cell.clone()),
        Display::Gpu { volume, cell, .. } => (volume.read_back(&ctx).expect("the read-back"), cell.clone()),
    };
    let dv = cell.h.powi(3);
    let n_el = values.iter().map(|&v| v as f64).sum::<f64>() * dv;
    // ∫ρ_pc on the same grid: what a profile with the core would carry on top
    let atom = isolated_atom(8, None, &scene);
    let (grid, system, _) = crate::ks::problem(&atom, 1e-6, &mut Vec::new()).expect("posed");
    let setup = system.ionic.as_deref().expect("O is pseudised under Accepted");
    let core = qm::nuclear::partial_core_field(&grid, &system.nuclei, setup).iter().sum::<f64>() * dv;
    println!("pseudo-O atom density with \"show partial core\" on: ∫ρ = {n_el:.6} (Z_v = 6; ∫ρ_pc = {core:.4})");
    assert!(core > 5e-2, "the test discriminates: ∫ρ_pc = {core}");
    assert!((n_el - 6.0).abs() < 1e-2, "valence only: ∫ρ = {n_el}");
}
