//! 2026-11-14 (user-approved trust change): a forgotten sample's fence can lift (`surrogate` module
//! doc, "Lifting a forgotten sample's fence"). `FENCE_VOTES` consecutive, mutually consistent fresh
//! teacher answers near only-forgotten fencing samples that agree with the model lift the fence of
//! those samples (code `Q_FORGOTTEN_UNFENCED`): the last answer is kept (`FenceLifted`), the model
//! answers there again, the samples stay forgotten (Restore still brings them back, and clears the
//! lift), the state persists, a disagreeing answer breaks the run, a quarantined neighbour behaves as
//! before, and a purge drops the votes of purged samples. Atomic units throughout.

use super::*;

fn forget(s: &mut SurrogateKs, i: usize) {
    let id = s.sample_id(i).unwrap();
    s.forget_sample(id).unwrap();
}

/// Stored sample `k`'s geometry, its bond stretched by `stretch` (Bohr) about its midpoint.
fn near_sample(s: &SurrogateKs, k: usize, stretch: f64) -> Scene {
    let smp = &s.store.as_ref().unwrap().samples[k];
    let mut sc = dimer(1.0, [1.0, 0.0, 0.0], [0.0; 3], smp.z[0], smp.z[1]);
    sc.cell = Cell { n: 33, h: 0.25 };
    for (n, p) in sc.nuclei.iter_mut().zip(&smp.pos) {
        n.pos = *p;
    }
    let (a, b) = (sc.nuclei[0].pos, sc.nuclei[1].pos);
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    for c in 0..3 {
        sc.nuclei[0].pos[c] -= 0.5 * stretch * d[c] / r;
        sc.nuclei[1].pos[c] += 0.5 * stretch * d[c] / r;
    }
    sc
}

/// One evaluation at `sc` after an edit: its source and its activity row's outcome.
fn visit(s: &mut SurrogateKs, sc: &Scene) -> (Source, Option<TeachOutcome>, Vec<String>) {
    s.discontinuity();
    let r = s.evaluate(sc, &mut never()).unwrap();
    let outcome = s.take_events().into_iter().last().map(|e| e.outcome);
    (r.stats.source, outcome, r.stats.notes)
}

/// The window of exact H₂ Morse samples with sample `k` forgotten and refitted without it.
fn window_without(k: usize) -> (SurrogateKs, Arc<Mutex<MorseDimer>>) {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    forget(&mut s, k);
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    (s, h)
}

fn fence_at(s: &mut SurrogateKs, sc: &Scene) -> Vec<usize> {
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    s.near_quarantined(&pos, &z)
}

/// Three agreeing answers next to a forgotten sample lift its fence: the first two are the
/// ordinary agreeing answers of before (kept or thinned, the fence stands), the third is kept with
/// `FenceLifted`; the sample is still forgotten (out of the fit, listed) but fences nothing, and
/// the model answers there again. Restore then clears the lift.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn three_agreeing_answers_lift_a_forgotten_fence() {
    let k = 17;
    let (mut s, _) = window_without(k);
    let stretches = [0.0, 2e-3, 4e-3];
    for (j, &x) in stretches.iter().enumerate() {
        let sc = near_sample(&s, k, x);
        assert_eq!(fence_at(&mut s, &sc), vec![k], "the forgotten sample fences (visit {j})");
        let before = s.samples();
        let (src, out, notes) = visit(&mut s, &sc);
        assert_eq!(src, Source::Fallback, "{notes:?}");
        if j + 1 < stretches.len() {
            assert!(notes.iter().any(|n| n.contains(&format!("agreeing answer {} of {FENCE_VOTES} toward lifting their fence", j + 1))), "{notes:?}");
            assert_eq!(s.quarantine[k], Q_FORGOTTEN);
        } else {
            assert_eq!(out, Some(TeachOutcome::FenceLifted { lifted: 1 }), "{notes:?}");
            assert!(notes.iter().any(|n| n.contains("the fence of 1 forgotten sample is lifted")), "{notes:?}");
            assert_eq!(s.samples(), before + 1, "the last answer is kept");
        }
    }
    assert_eq!(s.quarantine[k], Q_FORGOTTEN_UNFENCED);
    assert_eq!((s.forgotten(), s.forgotten_unfenced(), s.is_forgotten(k)), (1, 1, true), "still forgotten");
    assert_eq!(s.forgotten_samples().len(), 1, "still listed for Restore");
    let sc = near_sample(&s, k, 1e-3);
    assert!(!s.fences(k) && fence_at(&mut s, &sc).is_empty(), "no longer fencing");
    // the model answers there again (a step after the edit's first)
    s.discontinuity();
    s.evaluate(&sc, &mut never()).unwrap();
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Surrogate, "{:?}", r.stats.notes);
    // the lift stays out of the fit: a refit does not fit it
    s.refit().unwrap();
    assert_eq!(s.quarantine[k], Q_FORGOTTEN_UNFENCED, "never released by a refit");
    assert_eq!(s.forgotten_in_fit(), 0);
    // Restore clears the lift with the forget
    let id = s.sample_id(k).unwrap();
    s.restore_sample(id).unwrap();
    assert_eq!((s.quarantine[k], s.forgotten()), (Q_NONE, 0));
}

/// A disagreeing answer breaks the run: two agreeing, one disagreeing (the teacher moved), then two
/// agreeing again do not lift; a third does. Disagreeing answers alone never lift (they vote as
/// before, `KeptByVote`), and the forgotten sample stays fencing meanwhile.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_disagreeing_answer_breaks_the_run() {
    let k = 17;
    let (mut s, teacher) = window_without(k);
    let stretch = |j: usize| 1e-3 * j as f64;
    let mut j = 0;
    let mut step = |s: &mut SurrogateKs| {
        j += 1;
        let sc = near_sample(s, k, stretch(j));
        visit(s, &sc)
    };
    step(&mut s);
    step(&mut s);
    // the teacher disagrees (its well 5 mHa deeper): the run is broken
    teacher.lock().unwrap().d += 5e-3;
    let (_, out, notes) = step(&mut s);
    assert_eq!(out, Some(TeachOutcome::FenceDisagrees), "{notes:?}");
    teacher.lock().unwrap().d -= 5e-3;
    let (_, out, _) = step(&mut s);
    assert_ne!(out, Some(TeachOutcome::FenceLifted { lifted: 1 }));
    let (_, out, _) = step(&mut s);
    assert_ne!(out, Some(TeachOutcome::FenceLifted { lifted: 1 }), "two agreeing since the break");
    assert_eq!(s.quarantine[k], Q_FORGOTTEN, "still fencing");
    let (_, out, notes) = step(&mut s);
    assert_eq!(out, Some(TeachOutcome::FenceLifted { lifted: 1 }), "{notes:?}");
    // disagreeing answers alone vote as before and never lift
    let (mut d, teacher) = window_without(k);
    teacher.lock().unwrap().d += 5e-3;
    let mut outs = Vec::new();
    for j in 0..3 {
        let sc = near_sample(&d, k, 1e-3 * j as f64);
        outs.push(visit(&mut d, &sc).1);
    }
    assert!(!outs.contains(&Some(TeachOutcome::FenceLifted { lifted: 1 })), "{outs:?}");
    assert_eq!(d.quarantine[k], Q_FORGOTTEN);
}

/// A forgotten and a quarantined sample fencing together: an agreeing answer releases (or
/// supersedes) the quarantined one as before and casts no lift vote; the forgotten one keeps
/// fencing.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_quarantined_neighbour_behaves_as_before() {
    let k = 17;
    let (mut s, _) = window_without(k);
    s.quarantine[k + 1] = Q_HELD; // held since the fit: fences
    let sc = near_sample(&s, k, 0.0);
    let near = fence_at(&mut s, &sc);
    assert!(near.contains(&k) && near.contains(&(k + 1)), "{near:?}");
    let (_, out, notes) = visit(&mut s, &sc);
    assert!(matches!(out, Some(TeachOutcome::KeptReleased { .. })), "{out:?} {notes:?}");
    assert_ne!(s.quarantine[k + 1], Q_HELD, "released or superseded as before");
    assert_eq!(s.quarantine[k], Q_FORGOTTEN, "the forgotten one is not lifted by it");
    assert!(s.lift_votes.is_empty(), "no lift vote from a mixed fence");
}

/// The lift persists: a sidecar round trip and a collection reopen read code 8 back (not
/// fencing, still forgotten); a file without the code reads as before (not lifted).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_lifted_fence_survives_a_reload() {
    let k = 17;
    let (mut s, _) = window_without(k);
    for x in [0.0, 2e-3, 4e-3] {
        let sc = near_sample(&s, k, x);
        visit(&mut s, &sc);
    }
    assert_eq!(s.quarantine[k], Q_FORGOTTEN_UNFENCED);
    let path = std::env::temp_dir().join(format!("fence-lift-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let file = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(file.quarantine[k], Q_FORGOTTEN_UNFENCED);
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    back.install_sidecar(file).unwrap();
    assert!(back.is_forgotten(k) && !back.fences(k), "forgotten, not fencing, after the reload");
    // the shared collection: the lift is a status record, written at once
    let dir = collection_root("fence-lift");
    let (mut c, key) = super::forget::small_collection(&dir);
    let k = 20;
    forget(&mut c, k);
    c.refit().unwrap();
    let mut lifted = false;
    for x in [0.0, 2e-3, 4e-3] {
        let sc = near_sample(&c, k, x);
        if fence_at(&mut c, &sc).is_empty() {
            break;
        }
        lifted |= visit(&mut c, &sc).1 == Some(TeachOutcome::FenceLifted { lifted: 1 });
    }
    assert!(lifted, "the collection-mode run lifts too");
    assert_eq!(c.collection().unwrap().quarantine[k], Q_FORGOTTEN_UNFENCED, "written at once");
    drop(c);
    let (coll, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(coll.quarantine[k], Q_FORGOTTEN_UNFENCED);
    let (mut again, _) = pair_student();
    again.attach(coll).unwrap();
    assert!(again.is_forgotten(k) && !again.fences(k));
}

/// A purge removes the fence state of the purged samples: lift votes and fence votes next to them
/// are dropped, and nothing fences there any more.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_purge_drops_the_fence_state_of_purged_samples() {
    let k = 17;
    let (mut s, _) = window_without(k);
    for x in [0.0, 2e-3] {
        let sc = near_sample(&s, k, x);
        visit(&mut s, &sc);
    }
    assert_eq!(s.lift_votes.len(), 2, "two agreeing votes toward the lift");
    let sc = near_sample(&s, k, 1e-3);
    s.purge_forgotten().unwrap();
    assert!(s.lift_votes.is_empty() && s.fence_votes.is_empty());
    assert!(fence_at(&mut s, &sc).is_empty(), "a purged sample fences nothing");
}

/// The user's scenario (2026-11-14): a band of forgotten samples fences ordinary H₂ geometries and
/// every agreeing DFT answer there was thinned, forever. Now a slow trajectory through the band
/// falls back for three steps, the third lifts the fence, and the model answers from then on.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forgotten_band_stops_falling_back_after_three_steps() {
    let (mut s, _) = trained(1.2, 2.2, 80, FallbackPolicy::default());
    for i in [30, 32, 34] {
        forget(&mut s, i);
    }
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    let k = 32;
    let mut sources = Vec::new();
    let mut outcomes = Vec::new();
    let sc0 = near_sample(&s, k, 0.0);
    assert!(!fence_at(&mut s, &sc0).is_empty(), "the band fences");
    s.discontinuity();
    s.evaluate(&sc0, &mut never()).unwrap();
    s.take_events();
    for j in 1..=8 {
        let sc = near_sample(&s, k, 5e-4 * j as f64);
        let r = s.evaluate(&sc, &mut never()).unwrap();
        sources.push(r.stats.source);
        outcomes.push(s.take_events().into_iter().last().map(|e| e.outcome));
    }
    let lifted_at = outcomes.iter().position(|o| matches!(o, Some(TeachOutcome::FenceLifted { .. }))).expect("the fence lifts");
    assert!(lifted_at <= 2, "at the third fallback at the latest: {outcomes:?}");
    assert!(sources[lifted_at + 1..].iter().all(|&x| x == Source::Surrogate), "the model answers after the lift: {sources:?} {outcomes:?}");
}

/// The Model tab's words (2026-11-14): the readiness at a geometry fenced by forgotten samples
/// alone says so (`Readiness::forgotten_fence`, the refusal text names how it lifts); with a
/// quarantined sample among the fence it does not.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_readiness_tells_a_forgotten_fence_apart() {
    let k = 17;
    let (mut s, _) = window_without(k);
    let sc = near_sample(&s, k, 0.0);
    s.ensure_desc_cache();
    let r = s.readiness(&sc).unwrap();
    assert_eq!((r.answers, r.rule, r.forgotten_fence), (false, Some(TeachRule::NearQuarantine), true), "{r:?}");
    assert!(r.text.contains("samples you forgot") && r.text.contains("3 fresh teacher answers in a row agree"), "{}", r.text);
    s.quarantine[k + 1] = Q_HELD;
    let r = s.readiness(&sc).unwrap();
    assert_eq!((r.rule, r.forgotten_fence), (Some(TeachRule::NearQuarantine), false), "{r:?}");
    assert!(r.text.contains("held out of the fit"), "{}", r.text);
}
