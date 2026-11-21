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

/// A pass whose teacher answers with empty densities from its 4th solve on: every such answer is refused (counted,
/// a Failed event with the reason), none kept — the pass runs to its end on its own failures, as for any failed
/// solve, and keeps only the answers before.
#[test]
fn a_pass_refuses_empty_answers() {
    let plan = SamplingPlan { perturbations: 6, scan_points_per_pair: 6, hot_burst_steps: 6, ..SamplingPlan::default() };
    let mut s = SurrogateKs::new(Box::new(Faulty::new(AtomMorse(MorseDimer::h2()), 4, Fault::EmptyDensity)), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let mut refs = sim::AtomReferences::default();
    Trainable::start_pass(&mut s, &h2(1.45), &plan, &mut refs).unwrap();
    let mut steps = 0;
    while Trainable::pass_step(&mut s, &mut never()).unwrap().is_some() {
        steps += 1;
        assert!(steps < 500);
    }
    // the reference (one H multiplicity) and two samples came before the faults
    assert!(s.samples() <= 2, "{} samples kept", s.samples());
    assert!(s.refused_answers() > 0);
    let events = drain(&mut s);
    assert!(events.iter().any(|e| e.origin == TeachOrigin::Pass && e.outcome == TeachOutcome::Failed && e.detail.contains("refused") && e.detail.contains("electrons")));
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
