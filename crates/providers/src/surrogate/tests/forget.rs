//! Spec `2026-11-12-sample-recovery-design.md` R2: forgetting and restoring one sample
//! ([`SurrogateKs::forget_sample`], [`SurrogateKs::restore_sample`], code [`Q_FORGOTTEN`]) and the
//! Model tab's "Worst-fitting samples" ([`SurrogateKs::sample_suspects`]). A forgotten sample is
//! not fitted, never released or superseded, fences like a held one, counts against the 10 %
//! recovery cap, survives a collection reopen and a sidecar round trip, and a background refit
//! that lands after the forget never brings it back. Fix round 1: only a fitted sample (code 0)
//! can be forgotten, the buttons carry a checked sample identity (`sim::SampleId`), and a forget
//! never lifts rule 8's global untrust. Atomic units throughout.

use super::*;

/// The window of exact H₂ Morse samples (`trained`): 40 dimers 1.2–2.2 Bohr, fitted and trusted.
fn window() -> SurrogateKs {
    let (s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    s
}

/// The fit's members by their code at the fit (`fit_members`): whether sample `i` was fitted.
fn fitted(s: &SurrogateKs, i: usize) -> bool {
    i < s.fit_members.0 && s.fit_members.1.get(i).is_none_or(|&q| q == Q_NONE)
}

/// Forget stored sample `i` by the id the Model tab would send (an unchecked one past the store).
fn forget(s: &mut SurrogateKs, i: usize) -> Result<(), String> {
    let id = s.sample_id(i).unwrap_or(sim::SampleId { index: i, key: 0 });
    s.forget_sample(id)
}

/// Restore stored sample `i` by its id.
fn restore(s: &mut SurrogateKs, i: usize) -> Result<(), String> {
    let id = s.sample_id(i).unwrap_or(sim::SampleId { index: i, key: 0 });
    s.restore_sample(id)
}

/// Forget: written at once (the store's code), a refit due, not fitted by it, and never released
/// by a refit whose model reproduces it — where a held sample at the same place is (the control).
/// It fences like a held sample: its geometry falls back; a fresh answer there that agrees with
/// the model is kept, and the forgotten sample stays forgotten (never released or superseded);
/// the note calls it forgotten, not quarantined (fix round 1, M4).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forgotten_sample_is_not_fitted_and_never_released() {
    let mut s = window();
    let k = 17;
    assert!(fitted(&s, k));
    forget(&mut s, k).unwrap();
    assert_eq!((s.quarantine[k], s.forgotten(), s.quarantined()), (Q_FORGOTTEN, 1, 0));
    assert_eq!(s.forgotten_in_fit(), 1, "the standing fit still holds it until the refit");
    assert!(s.pending_refit(), "a refit is due");
    assert!(s.user_notes.iter().any(|n| n.contains("sample #17 forgotten") && n.contains("saved with the model file")), "{:?}", s.user_notes);
    assert!(forget(&mut s, k).is_err(), "forgotten already");
    let n = s.samples();
    assert!(forget(&mut s, n).is_err(), "no such sample");
    let rep = s.refit().unwrap();
    assert_eq!(rep.n_samples, 39, "the refit fits without it");
    assert!(!fitted(&s, k));
    assert_eq!(s.forgotten_in_fit(), 0);
    assert!(s.reproduces_sample(&s.store.as_ref().unwrap().samples[k].clone()), "the premise: the model reproduces it");
    assert_eq!(s.quarantine[k], Q_FORGOTTEN, "and does not release it");
    s.refit().unwrap();
    assert_eq!(s.quarantine[k], Q_FORGOTTEN, "nor the next");
    // the control: a held sample the model reproduces is released by the refit
    let mut c = window();
    c.quarantine.resize(c.samples(), Q_NONE);
    c.quarantine[k] = Q_HELD;
    c.species_dirty = true;
    c.refit().unwrap();
    assert_eq!(c.quarantine[k], Q_NONE, "the control: a held sample it reproduces is released");
    // the fence: the forgotten geometry falls back; the teacher (exact Morse) agrees with the
    // model there — kept, and the forgotten sample stays forgotten
    let sc = {
        let smp = &s.store.as_ref().unwrap().samples[k];
        let mut sc = dimer(1.0, [1.0, 0.0, 0.0], [0.0; 3], 1, 1);
        for (n, p) in sc.nuclei.iter_mut().zip(&smp.pos) {
            n.pos = *p;
        }
        sc
    };
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    assert_eq!(s.near_quarantined(&pos, &[1, 1]), vec![k], "it fences");
    let before = s.samples();
    s.discontinuity();
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
    assert!(r.stats.notes.iter().any(|n| n.contains("near forgotten samples agrees with the model") && n.contains("forgotten kept")), "{:?}", r.stats.notes);
    assert!(!r.stats.notes.iter().any(|n| n.contains("near quarantined samples")), "{:?}", r.stats.notes);
    assert_eq!((s.quarantine[k], s.superseded(), s.samples()), (Q_FORGOTTEN, 0, before + 1));
}

/// Final review I1: a forgotten sample is never released or superseded, and each agreeing teacher
/// answer next to it is an ordinary one: the first is kept (as pinned above), a second at the same
/// geometry is a copy of it, as anywhere else; before, every visit added a sample. Since
/// 2026-11-14 (the fence lift, `fence_lift.rs`) the fence is no longer permanent: the third
/// consecutive agreeing answer — which was thinned before — lifts it and is kept.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn agreeing_answers_next_to_a_forgotten_sample_are_thinned() {
    let mut s = window();
    let k = 17;
    forget(&mut s, k).unwrap();
    s.refit().unwrap();
    let at = |s: &SurrogateKs, stretch: f64| {
        let smp = &s.store.as_ref().unwrap().samples[k];
        let mut sc = dimer(1.0, [1.0, 0.0, 0.0], [0.0; 3], 1, 1);
        for (n, p) in sc.nuclei.iter_mut().zip(&smp.pos) {
            n.pos = *p;
        }
        // stretch the bond along itself, about its midpoint
        let (a, b) = (sc.nuclei[0].pos, sc.nuclei[1].pos);
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        for c in 0..3 {
            sc.nuclei[0].pos[c] -= 0.5 * stretch * d[c] / r;
            sc.nuclei[1].pos[c] += 0.5 * stretch * d[c] / r;
        }
        sc
    };
    let visit = |s: &mut SurrogateKs, stretch: f64| {
        let sc = at(s, stretch);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        assert_eq!(s.near_quarantined(&pos, &[1, 1]), vec![k], "the forgotten sample fences (stretch {stretch})");
        s.discontinuity();
        let r = s.evaluate(&sc, &mut never()).unwrap();
        assert_eq!(r.stats.source, Source::Fallback, "{:?}", r.stats.notes);
        assert!(r.stats.notes.iter().any(|n| n.contains("near forgotten samples agrees with the model") && n.contains("forgotten kept")), "{:?}", r.stats.notes);
        assert_eq!((s.quarantine[k], s.superseded()), (Q_FORGOTTEN, 0), "never released or superseded");
        r.stats.notes
    };
    let before = s.samples();
    visit(&mut s, 0.0);
    assert_eq!(s.samples(), before + 1, "the first agreeing answer is kept");
    let notes = visit(&mut s, 0.0);
    assert_eq!(s.samples(), before + 1, "the second, the same answer, is not: {notes:?}");
    assert!(notes.iter().any(|n| n.contains("a copy of stored sample")), "{notes:?}");
    // the third a hair away: before 2026-11-14 thinned with the fence standing; now it lifts it
    let sc = at(&s, 2e-3);
    s.discontinuity();
    let r = s.evaluate(&sc, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("the fence of 1 forgotten sample is lifted")), "{:?}", r.stats.notes);
    assert_eq!((s.quarantine[k], s.forgotten(), s.samples()), (Q_FORGOTTEN_UNFENCED, 1, before + 2), "still forgotten, the answer kept");
    assert!(s.near_quarantined(&at(&s, 0.0).nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), &[1, 1]).is_empty(), "the fence is lifted");
}

/// Restore: back in the fit at the refit (code 0, what it had); a sample that is not forgotten is
/// refused.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_restored_sample_is_fitted_again() {
    let mut s = window();
    assert!(restore(&mut s, 3).is_err(), "not forgotten");
    forget(&mut s, 3).unwrap();
    s.refit().unwrap();
    assert!(!fitted(&s, 3));
    restore(&mut s, 3).unwrap();
    assert_eq!((s.quarantine[3], s.forgotten()), (Q_NONE, 0));
    assert!(s.pending_refit());
    let rep = s.refit().unwrap();
    assert!(fitted(&s, 3));
    assert_eq!(rep.n_samples, 40);
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
}

/// Fix round 1, M3: only a sample in the fit (code 0) can be forgotten — a quarantined,
/// superseded, box-excluded, draft, waiting or lone one is refused by name, nothing changed, so
/// Restore (back to 0) always restores what it was.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn only_a_fitted_sample_can_be_forgotten() {
    let mut s = window();
    s.quarantine.resize(s.samples(), Q_NONE);
    for (i, q, words) in [
        (1, Q_HELD, "quarantined by the trust recovery"),
        (2, Q_SUPERSEDED, "superseded by a fresh answer"),
        (3, Q_BOX, "box-excluded"),
        (4, Q_DRAFT, "a draft guess held out of the strict fit"),
        (5, Q_REF_WAIT, "waiting for its box's isolated-atom reference"),
        (6, Q_LONE, "lone atoms"),
    ] {
        s.quarantine[i] = q;
        let e = forget(&mut s, i).unwrap_err();
        assert!(e.contains(&format!("sample #{i} is not in the fit")) && e.contains(words), "{e}");
        assert_eq!(s.quarantine[i], q, "nothing changed");
    }
    assert_eq!(s.forgotten(), 0);
}

/// Fix round 1, M8: the buttons carry the sample's identity; an id whose index now holds another
/// sample (the store changed between the list and the click) or no sample is refused with "the
/// sample list changed — pick it again", nothing changed — for Forget and Restore alike.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_stale_sample_id_is_refused() {
    let mut s = window();
    let id = s.sample_id(5).unwrap();
    assert_eq!(id.index, 5);
    assert_ne!(id.key, s.sample_id(6).unwrap().key, "two samples, two keys");
    let stale = sim::SampleId { index: 5, key: s.sample_id(6).unwrap().key };
    let e = s.forget_sample(stale).unwrap_err();
    assert!(e.contains("the sample list changed — pick it again"), "{e}");
    let e = s.forget_sample(sim::SampleId { index: 999, key: id.key }).unwrap_err();
    assert!(e.contains("the sample list changed — pick it again"), "{e}");
    assert_eq!(s.forgotten(), 0, "nothing changed");
    s.forget_sample(id).unwrap();
    let e = s.restore_sample(stale).unwrap_err();
    assert!(e.contains("the sample list changed — pick it again"), "{e}");
    assert!(s.is_forgotten(5));
    s.restore_sample(id).unwrap();
    assert_eq!(s.forgotten(), 0);
    // the list's rows carry the same id
    s.ensure_loo_ratios();
    assert!(s.sample_suspects().iter().all(|x| s.sample_id(x.id.index) == Some(x.id)));
}

/// Fix round 1, I1: forgetting the sample a failed audit kept does not lift rule 8 — the case is
/// re-judged at the refit as before (here it still fails: the fit without the audited sample
/// extrapolates wrongly there), so the model stays untrusted with the failed-audit reason. The
/// fixture is `a_failed_audit_stays_failed_until_a_refit_predicts_it_without_its_sample`'s.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn forgetting_a_failed_audits_sample_keeps_rule_8() {
    let (mut s, h) = trained_with_auditor(1.2, 1.6, 40, FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() });
    let teacher = |sc: &Scene| h.lock().unwrap().clone().evaluate(sc, &mut never()).unwrap();
    let sc = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        .iter()
        .map(|&r| dimer(r, [0.6, 0.8, 0.0], [0.0; 3], 1, 1))
        .find(|sc| {
            let (pos, z) = pos_z(sc);
            let p = s.model().unwrap().predict(&pos, &z, false).unwrap();
            (teacher(sc).energy - p.energy).abs() / 2.0 > 1e-3
        })
        .expect("an extrapolation beyond the energy tier");
    let t = teacher(&sc);
    let prov = s.provenance_for(&sc);
    s.handle_answer(audit_answer(&s, &sc, prov, Ok((t.energy, t.forces.clone(), Some(0.0), None))));
    assert_eq!(s.last_audit().map(|a| a.verdict), Some(AuditVerdict::Failed));
    s.refit().unwrap();
    assert!(!s.trusted() && s.audit_failed().is_some(), "the premise: rule 8 holds after the refit");
    let k = s.failed_audits.iter().find_map(|c| c.index).expect("the failed audit kept its sample");
    assert_eq!(s.quarantine.get(k).copied().unwrap_or(Q_NONE), Q_NONE, "the premise: the audited sample is fitted");
    forget(&mut s, k).unwrap();
    s.refit().unwrap();
    let notes = Trainable::take_notes(&mut s);
    println!("after forgetting the audited sample #{k}: trusted {}, {:?}, notes {notes:?}", s.trusted(), s.untrusted_reason());
    assert!(!fitted(&s, k) && s.is_forgotten(k));
    assert!(!s.trusted() && s.audit_failed().is_some(), "forgetting never lifts rule 8: {:?}", s.untrusted_reason());
    assert!(s.failed_audits.iter().any(|c| c.index == Some(k)), "the case is kept and re-judged");
    assert!(!notes.iter().any(|n| n.contains("quarantined by the refit")), "{notes:?}");
}

/// A model file keeps the code: the sidecar writes 7 and a load restores it (forgotten, excluded,
/// fencing); a forget between the fit and the save is caught by the load's membership check.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forgotten_sample_survives_a_sidecar_round_trip() {
    let mut s = window();
    forget(&mut s, 5).unwrap();
    s.refit().unwrap();
    forget(&mut s, 9).unwrap(); // forgotten since the fit: the file's prefix is not the fit's
    let path = std::env::temp_dir().join(format!("forget-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let file = surrogate::sidecar::read(&path).unwrap();
    assert_eq!((file.quarantine[5], file.quarantine[9]), (Q_FORGOTTEN, Q_FORGOTTEN));
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    back.install_sidecar(file).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!((back.forgotten(), back.is_forgotten(5), back.is_forgotten(9)), (2, true, true));
    back.refit().unwrap();
    assert!(!fitted(&back, 5) && !fitted(&back, 9));
    assert_eq!(back.report().unwrap().n_samples, 38);
}

/// The shared collection: Forget writes the status record **at once** (before any refit), a reopen
/// and a new attach read it back (excluded, counted as forgotten), and Restore writes it back.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forgotten_sample_survives_a_collection_reopen() {
    let dir = collection_root("forget-reopen");
    let (mut s, key) = small_collection(&dir);
    forget(&mut s, 4).unwrap();
    assert_eq!(s.collection().unwrap().quarantine[4], Q_FORGOTTEN, "written at once");
    assert!(s.user_notes.iter().any(|n| n.contains("written to the shared collection")), "{:?}", s.user_notes);
    drop(s);
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(c.quarantine[4], Q_FORGOTTEN, "the forget survived the reopen");
    let (mut again, _) = pair_student();
    again.attach(c).unwrap();
    assert_eq!((again.forgotten(), again.is_forgotten(4)), (1, true));
    assert!(again.model.is_some(), "the attach refitted");
    assert!(!fitted(&again, 4), "not fitted after the reopen");
    restore(&mut again, 4).unwrap();
    assert_eq!(again.collection().unwrap().quarantine[4], Q_NONE, "the restore is written at once");
    drop(again);
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(c.quarantine[4], Q_NONE);
}

/// Fix round 1, M2: on a read-only collection (another process holds its lock) Forget and the
/// Restore after it are this session's only, and both notes say so — the Restore is not reported
/// as written because the collection still reads its old code.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn forget_and_restore_on_a_read_only_collection_say_this_session_only() {
    let dir = collection_root("forget-read-only");
    let (mut s, key) = small_collection(&dir);
    // a second handle while the first holds the lock is read-only
    let (ro, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert!(ro.read_only, "the premise: the second open is read-only");
    let (mut other, _) = pair_student();
    other.attach(ro).unwrap();
    s.user_notes.clear();
    forget(&mut other, 4).unwrap();
    restore(&mut other, 4).unwrap();
    let notes = Trainable::take_notes(&mut other);
    let forgot = notes.iter().find(|n| n.contains("sample #4 forgotten")).expect("the forget's note");
    let restored = notes.iter().find(|n| n.contains("sample #4 restored")).expect("the restore's note");
    assert!(forgot.contains("this session only"), "{forgot}");
    assert!(restored.contains("this session only") && !restored.contains("(written to the shared collection)"), "{restored}");
    assert_eq!(s.collection().unwrap().quarantine[4], Q_NONE, "the writable handle's collection is untouched");
}

/// A model over a fresh shared collection in `dir`: 40 H₂ samples and the H reference, fitted.
pub(super) fn small_collection(dir: &std::path::Path) -> (SurrogateKs, surrogate::SurfaceKey) {
    let (mut s, _) = pair_student();
    s.attach_collection(dir, &h2(1.4)).unwrap();
    s.set_reference_solved(1, SpeciesMorse::atom(1), 33, Some(0.0));
    let mut rng = StdRng::seed_from_u64(11);
    for i in 0..40 {
        let sc = h2_at(1.1 + 1.2 * i as f64 / 39.0, &mut rng);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
        s.add_sample_answer(Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }, &sc, Some(0.0), Some(1)).unwrap();
    }
    s.refit().unwrap();
    assert_eq!(s.collection().unwrap().samples.len(), 40);
    let key = s.provenance_for(&h2(1.4)).surface_key();
    (s, key)
}

/// The plan's review focus 3: a Forget while a background refit runs is written to the collection
/// at once, does **not** discard the refit (the user's codes are not a conflict), and the install
/// re-applies it over the copy's code — the refit that lands never brings the sample back, the
/// collection keeps 7, and a rebuild is due; the next refit fits without it. A Restore in flight
/// lands the same way. Fix round 1, M5: while the installed fit still holds the sample it is
/// counted in the fit (`forgotten_in_fit`), not twice.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forget_during_a_background_refit_survives_its_install() {
    let dir = collection_root("forget-in-flight");
    let (mut s, key) = small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(12);
    let sc = h2_at(1.45, &mut rng);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
    s.add_sample_answer(Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Fallback, weight: 1.0, force_mask: None, use_energy: true }, &sc, Some(0.0), Some(1)).unwrap();
    assert!(s.begin_background_refit());
    forget(&mut s, 7).unwrap();
    assert_eq!(s.collection().unwrap().quarantine[7], Q_FORGOTTEN, "written while the refit runs");
    match s.finish_background_refit() {
        BackgroundPoll::Installed(r) => {
            r.unwrap();
        }
        BackgroundPoll::Discarded(why) => panic!("a forget is not a conflict: {why}"),
        _ => panic!("installed"),
    }
    assert_eq!(s.quarantine[7], Q_FORGOTTEN, "the install re-applies the user's code");
    assert_eq!(s.collection().unwrap().quarantine[7], Q_FORGOTTEN, "and syncs it, not the copy's");
    assert!(fitted(&s, 7), "the installed fit is the copy's, started before the forget");
    assert_eq!((s.forgotten(), s.forgotten_in_fit()), (1, 1), "counted in the fit, not beside it");
    assert!(s.pending_refit(), "a rebuild is due");
    assert!(!s.sample_suspects().iter().any(|x| x.id.index == 7), "not listed among the fitted samples");
    s.refit().unwrap();
    assert!(!fitted(&s, 7), "the next refit fits without it");
    assert_eq!((s.quarantine[7], s.forgotten_in_fit()), (Q_FORGOTTEN, 0));
    // a Restore in flight lands the same way
    assert!(s.begin_background_refit());
    restore(&mut s, 7).unwrap();
    assert_eq!(s.collection().unwrap().quarantine[7], Q_NONE);
    assert!(matches!(s.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
    assert_eq!(s.quarantine[7], Q_NONE, "the copy's 7 is not put back");
    assert_eq!(s.collection().unwrap().quarantine[7], Q_NONE);
    assert!(s.pending_refit());
    // another code change of a sample < P is still a conflict (unchanged)
    assert!(s.begin_background_refit());
    s.quarantine[3] = Q_SUPERSEDED;
    assert!(matches!(s.finish_background_refit(), BackgroundPoll::Discarded(_)));
    drop(s);
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(c.quarantine[7], Q_NONE);
}

/// The 10 % recovery cap counts forgotten samples with the held ones (spec R2: forgetting is no
/// way round it). The high-leverage outlier store (131 samples, cap 13.1): with 12 others forgotten
/// the recovery still holds the outlier (12 + 1 ≤ 13.1); with 13 forgotten it holds nothing
/// (13 + 1 > 13.1) and the trust rule judges what stands.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_recovery_cap_counts_forgotten_samples() {
    for (forgotten, held) in [(12usize, 1usize), (13, 0)] {
        let mut s = super::recovery_loo::outlier_store();
        for i in 0..forgotten {
            forget(&mut s, i).unwrap();
        }
        s.fit_store().unwrap();
        assert!(s.residuals_fail(), "the premise: the outlier still breaks the fit");
        let report = s.report().unwrap().clone();
        let _ = s.quarantine_rounds(report);
        assert_eq!(s.quarantined(), held, "{forgotten} forgotten");
        assert_eq!(s.forgotten(), forgotten, "never released by the rounds");
        if held == 1 {
            assert!(s.is_quarantined(super::recovery_loo::OUTLIER));
        }
    }
}

/// The Model tab's list: the fit's members by their leave-one-out ratio, worst first, at most 8,
/// formula and origin from the sample; a sample forgotten since the fit leaves the list at once
/// (no refit) and is listed among the forgotten ones with the ratio its fit gave it; no model, no
/// list.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_worst_fitting_samples_are_the_fits_own_ranking() {
    let mut s = super::recovery_loo::outlier_store();
    assert!(s.sample_suspects().is_empty(), "no model: no list");
    s.fit_store().unwrap();
    s.ensure_loo_ratios();
    let list = s.sample_suspects();
    assert_eq!(list.len(), SUSPECTS_SHOWN);
    let out = super::recovery_loo::OUTLIER;
    assert_eq!((list[0].id, list[0].formula.as_str(), list[0].origin.as_str()), (s.sample_id(out).unwrap(), "H2", "pass scan"));
    let ratios: Vec<f64> = list.iter().map(|x| x.loo_ratio.unwrap()).collect();
    assert!(ratios.windows(2).all(|w| w[0] >= w[1]), "worst first: {ratios:?}");
    let all = s.loo_ratios().unwrap().to_vec();
    assert_eq!(ratios[0], all.iter().map(|x| x.1).fold(0.0, f64::max), "the worst of the fit");
    forget(&mut s, out).unwrap();
    let after = s.sample_suspects();
    assert!(!after.iter().any(|x| x.id.index == out), "forgotten since the fit: not listed");
    assert_eq!(after[0], list[1], "the rest move up, the same numbers (no recompute)");
    let gone = s.forgotten_samples();
    assert_eq!(gone.len(), 1);
    assert_eq!((gone[0].id.index, gone[0].loo_ratio), (out, Some(ratios[0])));
    s.refit().unwrap();
    assert_eq!(s.forgotten_samples()[0].loo_ratio, None, "not in the new fit");
}

/// The plan's review focus 4: forgetting the only sample of an environment (here the one He
/// sample among H₂) must not panic; the refit stands or fails honestly, and the forget's note says
/// why the trust rule may refuse.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn forgetting_the_only_sample_of_a_species_is_honest() {
    let (mut s, _) = pair_student();
    s.set_reference_energy(1, SpeciesMorse::atom(1));
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let mut rng = StdRng::seed_from_u64(21);
    for i in 0..60 {
        let sc = h2_at(1.1 + 1.2 * i as f64 / 59.0, &mut rng);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
        s.add_sample(Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }, &sc).unwrap();
    }
    // H₂ with a He atom 2.5 Bohr from its centre, in the same box as the H₂ samples
    let mut sc = h2_at(1.4, &mut rng);
    sc.nuclei.push(Nucleus::new(2, [0.0, 2.5, 0.0]));
    let (pos, z): (Vec<[f64; 3]>, Vec<u8>) = (sc.nuclei.iter().map(|n| n.pos).collect(), sc.nuclei.iter().map(|n| n.z).collect());
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    s.add_sample(Sample { pos, z, energy: e, forces: f, tag: SampleTag::Fallback, weight: 1.0, force_mask: None, use_energy: true }, &sc).unwrap();
    let he = s.samples() - 1;
    let _ = s.refit();
    forget(&mut s, he).unwrap();
    assert!(s.user_notes.iter().any(|n| n.contains("the only sample of its kind") && n.contains("sample counts may refuse")), "{:?}", s.user_notes);
    let r = s.refit();
    println!("after forgetting the only He sample: {:?}; trusted {} {:?}", r.as_ref().map(|r| r.n_samples), s.trusted(), s.untrusted_reason());
    assert!(s.model.is_none() || !fitted(&s, he));
    assert!(s.trusted() || s.untrusted_reason().is_some_and(|w| !w.is_empty()), "an untrusted model says why");
    assert!(s.is_forgotten(he));
}

/// 2026-11-13 (display only): the two training-residual checks carry where their worst value
/// sits ([`TrustCheck::worst`](sim::activity::TrustCheck)) — the true argmax over the fitted
/// samples and their trained atoms, found here by a brute-force pass of its own over the fit's
/// members — with the formula and the atom's Z of that stored sample (H₂ and H₂·He samples: two
/// species). The location never moves a number: value and limit are the residual pass's. A sample
/// held since reads not forgettable; a changed sample at that index drops the location (the same
/// identity check Forget makes).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_residual_checks_name_their_true_argmax() {
    use sim::activity::TrustKey as K;
    let (mut s, _) = pair_student();
    s.set_reference_energy(1, SpeciesMorse::atom(1));
    s.set_reference_energy(2, SpeciesMorse::atom(2));
    let mut rng = StdRng::seed_from_u64(31);
    let add = |s: &mut SurrogateKs, sc: &Scene, tag| {
        let (pos, z): (Vec<[f64; 3]>, Vec<u8>) = (sc.nuclei.iter().map(|n| n.pos).collect(), sc.nuclei.iter().map(|n| n.z).collect());
        let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
        s.add_sample(Sample { pos, z, energy: e, forces: f, tag, weight: 1.0, force_mask: None, use_energy: true }, sc).unwrap();
    };
    for i in 0..60 {
        let sc = h2_at(1.1 + 1.2 * i as f64 / 59.0, &mut rng);
        add(&mut s, &sc, SampleTag::Scan);
    }
    for i in 0..8 {
        let mut sc = h2_at(1.3 + 0.05 * i as f64, &mut rng);
        sc.nuclei.push(Nucleus::new(2, [0.0, 2.2 + 0.1 * i as f64, 0.0]));
        add(&mut s, &sc, SampleTag::Fallback);
    }
    s.refit().unwrap();
    let t = s.training_residuals().expect("a fit with residuals");
    // the brute-force argmaxes over the fit's members: the largest |ΔE|/N, and the largest
    // force-residual ratio (first wins a tie, as the residual pass's strict `>` does)
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let (mut e_best, mut f_best): (Option<(f64, usize)>, Option<(f64, usize, usize, f64)>) = (None, None);
    for (i, smp) in store.samples[..s.accumulated].iter().enumerate().filter(|(i, _)| !s.excluded(*i)) {
        let p = model.predict(&smp.pos, &smp.z, false).unwrap();
        let e = (p.energy - smp.energy).abs() / smp.pos.len() as f64;
        if smp.use_energy && e_best.is_none_or(|(b, _)| e > b) {
            e_best = Some((e, i));
        }
        for (j, (fm, fs)) in p.forces.iter().zip(&smp.forces).enumerate().filter(|(j, _)| smp.trains_atom(*j)) {
            let (d, bound) = s.policy.force_residual(fm, fs);
            if f_best.is_none_or(|(b, ..)| d / bound > b) {
                f_best = Some((d / bound, i, j, d));
            }
        }
    }
    let ((e_max, e_i), (_, f_i, f_j, f_d)) = (e_best.unwrap(), f_best.unwrap());
    println!("worst energy residual {e_max:.3e} at #{e_i}, worst force residual {f_d:.3e} at #{f_i} atom {f_j}");
    assert!(e_max > 0.0 && f_d > 0.0, "the premise: nonzero residuals");
    assert_eq!((t.energy_per_atom, t.force), (e_max, f_d), "the location rides with the same numbers");
    let checks = s.trust_checks();
    let get = |k| checks.iter().find(|c| c.key == k).unwrap().clone();
    let (ce, cf) = (get(K::EnergyResidual), get(K::ForceResidual));
    let we = ce.worst.clone().expect("the energy check names its sample");
    let wf = cf.worst.clone().expect("the force check names its sample and atom");
    assert_eq!(we.id, s.sample_id(e_i).unwrap());
    assert_eq!((wf.id, wf.atom.map(|a| a.0)), (s.sample_id(f_i).unwrap(), Some(f_j)));
    assert_eq!(we.atom, None, "an energy is per sample");
    for (w, i) in [(&we, e_i), (&wf, f_i)] {
        let smp = &store.samples[i];
        assert_eq!(w.formula, shared::formula(&smp.z));
        assert_eq!(w.origin, super::super::quarantine::origin_word(smp.tag));
        assert!(w.forgettable, "a fitted sample: code 0");
    }
    assert_eq!(wf.atom.unwrap().1, store.samples[f_i].z[f_j]);
    // display only: the rest of every check is what the rule's own sequence gives
    for (c, plain) in checks.iter().zip(s.checks()) {
        assert_eq!((c.key, c.met, c.value.to_bits(), c.limit.to_bits()), (plain.key, plain.met, plain.value.to_bits(), plain.limit.to_bits()));
        assert!(plain.worst.is_none());
        assert!(c.worst.is_none() || matches!(c.key, K::EnergyResidual | K::ForceResidual));
    }
    // held since the fit: named, not forgettable (Forget would refuse a non-zero code)
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[f_i] = Q_HELD;
    let held = s.trust_checks().into_iter().find(|c| c.key == K::ForceResidual).unwrap().worst.unwrap();
    assert!(!held.forgettable);
    s.quarantine[f_i] = Q_NONE;
    // another sample at that index (a model switch, attach or load): no location, today's line
    s.store.as_mut().unwrap().samples[f_i].energy += 1e-9;
    assert_eq!(s.trust_checks().into_iter().find(|c| c.key == K::ForceResidual).unwrap().worst, None);
}
