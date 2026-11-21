//! The gpu-fault fix (2026-11-21; `surrogate` module doc, "A failed GPU"): a teacher whose GPU failed — a panic inside
//! the solve, `SolverError::GpuFailed`, or an answer with no data — never leaves a sample, never answers a step, and
//! never keeps a pass, an audit or the run going in a fast loop on a dead device. Fault injection through
//! `sim::Faulty`; no GPU is touched.
use super::*;
use sim::activity::{TeachOrigin, TeachOutcome};
use sim::{Fault, Faulty};
use std::sync::atomic::Ordering;

/// H₂ at `r` with a tiny offset per call so no two geometries are the same.
fn h2_at_step(i: usize) -> Scene {
    h2(1.3 + 0.002 * i as f64)
}

/// No model: every step is the teacher's. Two answers are kept, then the device fails: the failed step is an error,
/// nothing more is kept, and every later evaluation is refused at once — no teacher call, no model answer — with the
/// user's message; the provider reports the failure (`device_failure`), which is what stops the driver.
#[test]
fn a_teacher_whose_gpu_failed_keeps_nothing_and_answers_nothing_more() {
    let teacher = Faulty::new(AtomMorse(MorseDimer::h2()), 3, Fault::GpuFailed);
    let calls = teacher.calls.clone();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    for i in 0..2 {
        s.evaluate(&h2_at_step(i), &mut never()).expect("a healthy teacher's step");
    }
    assert_eq!(s.samples(), 2);
    let e = s.evaluate(&h2_at_step(2), &mut never()).err().expect("the failed solve is an error");
    assert!(e.gpu_failure().is_some(), "{e:?}");
    assert!(e.to_string().starts_with(sim::GPU_FAILED_MESSAGE), "{e}");
    let at_failure = calls.load(Ordering::SeqCst);
    for i in 3..200 {
        let e = s.evaluate(&h2_at_step(i), &mut never()).err().expect("refused");
        assert!(e.gpu_failure().is_some(), "{e:?}");
    }
    assert_eq!(calls.load(Ordering::SeqCst), at_failure, "no solve is asked of a failed device");
    assert_eq!(s.samples(), 2, "nothing was kept after the failure");
    assert!(ForceProvider::device_failure(&s).is_some());
}

/// The answers a failed GPU could leave — an empty density (E = 0, F = 0, 0 electrons), an SCF without an iteration,
/// one that did not converge — are refused: never kept, never used for the step (an error, so the driver pauses),
/// each counted and logged with its reason. 50 of them in a row keep 0 samples.
#[test]
fn physically_empty_teacher_answers_are_never_kept() {
    for (fault, why) in [(Fault::EmptyDensity, "electrons"), (Fault::NoIterations, "no iteration"), (Fault::NotConverged, "convergence")] {
        let mut s = SurrogateKs::new(Box::new(Faulty::new(AtomMorse(MorseDimer::h2()), 1, fault)), FallbackPolicy::default()).with_teacher_tol(1e-6);
        for i in 0..50 {
            let e = s.evaluate(&h2_at_step(i), &mut never()).err().unwrap_or_else(|| panic!("{fault:?}: an empty answer was used"));
            assert!(e.to_string().contains("refused") && e.to_string().contains(why), "{fault:?}: {e}");
        }
        assert_eq!(s.samples(), 0, "{fault:?}: an empty answer was kept");
        assert_eq!(s.refused_answers(), 50, "{fault:?}");
        let events = drain(&mut s);
        let refused: Vec<_> = events.iter().filter(|e| e.outcome == TeachOutcome::Failed && e.detail.contains("refused") && e.detail.contains(why)).collect();
        assert_eq!(refused.len(), 50, "{fault:?}: one Failed event with the reason per refusal: {:?}", events.iter().map(|e| &e.detail).collect::<Vec<_>>());
        assert!(ForceProvider::device_failure(&s).is_none(), "{fault:?}: an empty answer alone is no device failure");
    }
}

/// The pass on a GPU that fails at its 8th solve: before the fix every later job failed in microseconds and the pass
/// went on through all of them (hundreds of "failed" rows, no data); now the pass ends at the failure with the
/// error, no job after it is solved, and nothing after it is kept.
#[test]
fn a_pass_ends_at_the_gpu_failure() {
    let plan = SamplingPlan { perturbations: 20, scan_points_per_pair: 20, hot_burst_steps: 40, ..SamplingPlan::default() };
    let teacher = Faulty::new(AtomMorse(MorseDimer::h2()), 8, Fault::GpuFailed);
    let calls = teacher.calls.clone();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &h2(1.45), &plan, &mut refs).unwrap();
    let mut steps = 0;
    let end = loop {
        steps += 1;
        assert!(steps < 500, "the pass did not end");
        match Trainable::pass_step(&mut s, &mut never()) {
            Ok(Some(_)) => {}
            Ok(None) => break None,
            Err(e) => break Some(e),
        }
    };
    let e = end.expect("the pass ends with the GPU failure");
    assert!(e.gpu_failure().is_some(), "{e:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 8, "no job was solved after the failure");
    assert!(!Trainable::pass_active(&mut s));
    assert!(s.samples() < 8, "{} samples", s.samples());
    assert!(s.evaluate(&h2(1.45), &mut never()).err().is_some_and(|e| e.gpu_failure().is_some()));
}

/// A pass whose teacher answers with empty densities from its 4th solve on (review A-I2): every such answer is refused
/// (counted, a Failed event with the reason), none kept, and after `PASS_REFUSALS_IN_A_ROW` (3) of them in a row the
/// pass ends with an error that says why — no 4th refused solve, where before it walked its whole queue (24 jobs here)
/// one refused row after another. The answers before are kept.
#[test]
fn a_pass_refuses_empty_answers_and_ends_after_three_in_a_row() {
    assert_eq!(crate::sampling::PASS_REFUSALS_IN_A_ROW, 3);
    let plan = SamplingPlan { perturbations: 6, scan_points_per_pair: 6, hot_burst_steps: 6, ..SamplingPlan::default() };
    let teacher = Faulty::new(AtomMorse(MorseDimer::h2()), 4, Fault::EmptyDensity);
    let calls = teacher.calls.clone();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &h2(1.45), &plan, &mut refs).unwrap();
    let total = Trainable::progress(&s).expect("a pass").total;
    let mut steps = 0;
    let end = loop {
        steps += 1;
        assert!(steps < 500, "the pass did not end");
        match Trainable::pass_step(&mut s, &mut never()) {
            Ok(Some(_)) => {}
            Ok(None) => break None,
            Err(e) => break Some(e),
        }
    };
    let e = end.expect("the pass ends with an error on the refusals");
    let m = e.to_string();
    println!("{m} (the pass had {total} jobs)");
    assert!(m.contains("3 teacher answers in a row failed the answer check") && m.contains("electrons"), "{m}");
    assert!(m.contains("GPU memory") && m.contains("restart the app"), "{m}");
    assert!(e.gpu_failure().is_none(), "refusals alone are no device failure");
    assert_eq!(calls.load(Ordering::SeqCst), 6, "the reference, two kept samples, three refused — and no further solve");
    assert!(total > 6, "the pass had more jobs than it solved ({total})");
    assert!(!Trainable::pass_active(&mut s));
    assert_eq!(s.samples(), 2, "the two answers before the faults are kept, nothing after");
    assert_eq!(s.refused_answers(), 3);
    let events = drain(&mut s);
    let refused = events.iter().filter(|e| e.origin == TeachOrigin::Pass && e.outcome == TeachOutcome::Failed && e.detail.contains("refused") && e.detail.contains("electrons")).count();
    assert_eq!(refused, 3, "one Failed event per refused answer");
}

/// The count is of refusals **in a row**: an accepted answer between them starts it again, so a pass whose every
/// other answer is refused runs to its end (and keeps the accepted ones).
#[test]
fn an_accepted_answer_resets_the_refusal_count() {
    let plan = SamplingPlan { perturbations: 6, scan_points_per_pair: 6, hot_burst_steps: 6, ..SamplingPlan::default() };
    let mut s = SurrogateKs::new(Box::new(Alternating { inner: Faulty::new(AtomMorse(MorseDimer::h2()), 1, Fault::EmptyDensity), healthy: AtomMorse(MorseDimer::h2()), calls: 0 }), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &h2(1.45), &plan, &mut refs).unwrap();
    let mut steps = 0;
    while Trainable::pass_step(&mut s, &mut never()).expect("refusals that are not in a row do not end the pass").is_some() {
        steps += 1;
        assert!(steps < 500);
    }
    assert!(s.refused_answers() >= 3, "{} refused", s.refused_answers());
    assert!(s.samples() > 0);
}

/// Odd calls healthy (with a plausible SCF check), even calls an empty density.
struct Alternating {
    inner: Faulty,
    healthy: AtomMorse,
    calls: usize,
}

impl ForceProvider for Alternating {
    fn evaluate(&mut self, scene: &Scene, should_abort: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
        self.calls += 1;
        if self.calls % 2 == 0 {
            return self.inner.evaluate(scene, should_abort);
        }
        let mut r = self.healthy.evaluate(scene, should_abort)?;
        let n = scene.nuclei.iter().map(|n| n.z as f64).sum::<f64>() - scene.charge as f64;
        r.stats.scf_steps = r.stats.scf_steps.max(1);
        r.stats.scf_check = Some(sim::ScfCheck { converged: true, electrons: n, expected: n });
        Ok(r)
    }
    fn name(&self) -> &'static str {
        self.healthy.name()
    }
    fn reset(&mut self) {
        self.healthy.reset();
    }
    fn scf_tol(&self) -> Option<f64> {
        self.healthy.scf_tol()
    }
}

/// The background auditor's teacher panics inside every solve (a GPU failure before the fix's handler): each audit
/// comes back as a failed, unjudged audit — nothing kept — and the auditor's thread survives (it answered every job).
#[test]
fn a_panicking_auditor_keeps_nothing_and_survives() {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
    let mut s = s.with_auditor(Box::new(Faulty::new(Shared(h), 1, Fault::Panic)));
    let n0 = s.samples();
    let t0 = Instant::now();
    let mut failed = 0;
    let mut i = 0;
    while failed < 3 {
        assert!(t0.elapsed().as_secs_f64() < 10.0, "no failed audit came back");
        let _ = s.evaluate(&h2(1.5 + 0.001 * (i % 50) as f64), &mut never());
        i += 1;
        failed += drain(&mut s).iter().filter(|e| e.origin == TeachOrigin::Audit && e.outcome == TeachOutcome::Failed && e.detail.contains("panicked")).count();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(s.audits_asynchronous(), "the auditor's thread is still there");
    assert_eq!(s.samples(), n0, "a panicked audit kept something");
    assert!(ForceProvider::device_failure(&s).is_none());
}

/// The background auditor's device fails: its answer carries the failure, and from then on the surrogate answers
/// nothing — not even with its trusted model — and reports the failure, so the driver stops the run.
#[test]
fn an_auditor_whose_gpu_failed_stops_the_surrogate() {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
    let mut s = s.with_auditor(Box::new(Faulty::new(Shared(h), 1, Fault::GpuFailed)));
    let n0 = s.samples();
    let t0 = Instant::now();
    let mut i = 0;
    let e = loop {
        assert!(t0.elapsed().as_secs_f64() < 10.0, "the auditor's failure never reached the surrogate");
        match s.evaluate(&h2(1.5 + 0.001 * (i % 50) as f64), &mut never()) {
            Ok(r) => assert_ne!(r.stats.source, Source::Fallback, "the stepping teacher is healthy here; the model answers"),
            Err(e) => break e,
        }
        i += 1;
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    assert!(e.gpu_failure().is_some(), "{e:?}");
    assert!(ForceProvider::device_failure(&s).is_some());
    assert!(s.evaluate(&h2(1.5), &mut never()).is_err(), "the model answers no more steps");
    assert_eq!(s.samples(), n0);
}

/// The background auditor's answers are empty densities: each audit is refused (counted, its event says why), never
/// judged and never kept.
#[test]
fn empty_audit_answers_are_refused() {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
    let mut s = s.with_auditor(Box::new(Faulty::new(Shared(h), 1, Fault::EmptyDensity)));
    let n0 = s.samples();
    let t0 = Instant::now();
    let mut i = 0;
    while s.refused_answers() < 3 {
        assert!(t0.elapsed().as_secs_f64() < 10.0, "no refused audit came back");
        let _ = s.evaluate(&h2(1.5 + 0.001 * (i % 50) as f64), &mut never());
        i += 1;
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(s.samples(), n0, "an empty audit answer was kept");
    assert!(s.last_audit().is_none_or(|a| a.verdict != sim::AuditVerdict::Passed), "an empty answer judged the model");
}

/// The estimated mode's loop as the user met it, through the driver: a surrogate without a model plays, its teacher
/// answering empty densities. Before the fix every step kept one such answer (hundreds in seconds); now the first
/// step's answer is refused, the run pauses with the reason, and Play again keeps nothing.
#[test]
fn playing_on_empty_answers_keeps_nothing() {
    let s = SurrogateKs::new(Box::new(Faulty::new(AtomMorse(MorseDimer::h2()), 2, Fault::EmptyDensity)), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let (p, handle) = driven(s);
    let mut d = Dynamics::new(h2(1.45), p, 0.25, 1); // the first evaluation: healthy, kept
    d.apply(DynCommand::SetSpeed(50));
    for _ in 0..5 {
        d.apply(DynCommand::Play);
        let t = d.tick(f64::INFINITY, &mut never());
        assert!(t.paused, "the refused answer pauses the run");
        assert_eq!(t.step, 0, "no step was taken on an empty answer");
        assert!(t.last_error.as_deref().is_some_and(|e| e.contains("refused")), "{:?}", t.last_error);
    }
    let s = handle.lock().unwrap();
    assert_eq!(s.samples(), 1, "only the healthy answer was kept");
}

/// Review A-I1: once the auditor's GPU failure is latched, nothing more goes to the auditor — a display refresh, a
/// Harris job, an atom density, an atom reference are each refused — nothing is reported in flight (the worker goes
/// idle instead of polling), and the auditor's teacher is asked for no solve.
#[test]
fn after_a_gpu_failure_no_job_is_sent_to_the_auditor() {
    let (s, h) = trained(1.2, 2.2, 40, FallbackPolicy { audit_every: 2, ..FallbackPolicy::default() });
    let auditor = Faulty::new(Shared(h), 1, Fault::GpuFailed);
    let auditor_calls = auditor.calls.clone();
    let mut s = s.with_auditor(Box::new(auditor));
    let t0 = Instant::now();
    let mut i = 0;
    while s.evaluate(&h2(1.5 + 0.001 * (i % 50) as f64), &mut never()).is_ok() {
        assert!(t0.elapsed().as_secs_f64() < 10.0, "the auditor's failure never reached the surrogate");
        i += 1;
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(ForceProvider::device_failure(&s).is_some());
    let at_failure = auditor_calls.load(Ordering::SeqCst);
    let scene = h2(1.45);
    let atoms = Arc::new(sim::HarrisAtoms { profiles: Vec::new() });
    for _ in 0..20 {
        assert!(!s.request_display(&scene), "a display refresh was accepted after the failure");
        assert!(s.queue_harris(&scene, &atoms).is_none(), "a Harris job was queued after the failure");
        assert!(!s.request_atom_density(1, &scene), "an atom density was accepted after the failure");
        assert!(!s.request_atom_reference(1, 1837.0, &scene), "an atom reference was accepted after the failure");
        assert!(!Trainable::side_job_in_flight(&s), "a side job is reported in flight after the failure");
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert_eq!(auditor_calls.load(Ordering::SeqCst), at_failure, "the auditor was asked for a solve after the failure");
    assert!(!Trainable::side_job_in_flight(&s));
}

/// Review A-I1, the stepping teacher's side: its device failed and nothing was latched yet (no answer came back
/// since) — the requests are refused all the same and nothing is reported in flight.
#[test]
fn a_failed_stepping_teacher_stops_the_side_jobs_before_any_answer() {
    let h = Arc::new(Mutex::new(MorseDimer::h2()));
    let teacher = Faulty::new(Shared(h.clone()), 2, Fault::GpuFailed);
    let teacher_calls = teacher.calls.clone();
    let auditor = Faulty::new(Shared(h), usize::MAX, Fault::GpuFailed);
    let auditor_calls = auditor.calls.clone();
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6).with_auditor(Box::new(auditor));
    let scene = h2(1.45);
    assert!(ForceProvider::device_failure(&s).is_none());
    assert!(s.request_display(&scene), "a healthy device takes the request");
    // the auditor solves it (a healthy job, before the failure)
    let t0 = Instant::now();
    while auditor_calls.load(Ordering::SeqCst) < 1 {
        assert!(t0.elapsed().as_secs_f64() < 10.0, "the healthy refresh was never solved");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // the stepping teacher's device fails (its next call would be its 2nd): nothing has been latched
    teacher_calls.store(2, Ordering::SeqCst);
    assert!(ForceProvider::device_failure(&s).is_some(), "the teacher's device reports the failure");
    let t0 = Instant::now();
    while Trainable::side_job_in_flight(&s) {
        assert!(t0.elapsed().as_secs_f64() < 1.0, "a side job is still reported in flight after the failure");
    }
    assert!(!s.request_display(&scene));
    assert!(!s.request_atom_density(1, &scene));
    assert!(!s.request_atom_reference(1, 1837.0, &scene));
    let before = auditor_calls.load(Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(!s.request_display(&scene));
    assert_eq!(auditor_calls.load(Ordering::SeqCst), before, "a job went to the auditor after the failure");
    assert_eq!(before, 1, "only the refresh asked before the failure was solved");
}

/// Review A-M5: an atom reference asked of an auditor whose GPU fails is answered exactly once, with the failure — no
/// reference is made of it — and the failure is latched.
#[test]
fn an_atom_reference_on_a_failed_auditor_is_answered_with_the_failure() {
    let h = Arc::new(Mutex::new(MorseDimer::h2()));
    let teacher = Shared(h.clone());
    let auditor = Faulty::new(Shared(h), 1, Fault::GpuFailed);
    let mut s = SurrogateKs::new(Box::new(teacher), FallbackPolicy::default()).with_teacher_tol(1e-6).with_auditor(Box::new(auditor));
    assert!(s.request_atom_reference(1, 1837.0, &h2(1.45)), "a healthy session takes the request");
    let t0 = Instant::now();
    let answers = loop {
        let a = s.take_atom_references();
        if !a.is_empty() {
            break a;
        }
        assert!(t0.elapsed().as_secs_f64() < 10.0, "the reference was never answered");
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    assert_eq!(answers.len(), 1, "answered exactly once");
    let (z, r) = &answers[0];
    assert_eq!(*z, 1);
    let e = r.as_ref().err().expect("no reference from a failed GPU");
    assert!(e.to_string().contains("GPU"), "{e}");
    assert!(ForceProvider::device_failure(&s).is_some(), "the failure is latched");
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(s.take_atom_references().is_empty(), "answered once only");
}
