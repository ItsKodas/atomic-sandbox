//! Spec `2026-11-12-sample-recovery-design.md` R2: forgetting and restoring one sample
//! ([`SurrogateKs::forget_sample`], [`SurrogateKs::restore_sample`], code [`Q_FORGOTTEN`]) and the
//! Model tab's "Worst-fitting samples" ([`SurrogateKs::sample_suspects`]). A forgotten sample is
//! not fitted, never released or superseded, fences like a held one, counts against the 10 %
//! recovery cap, survives a collection reopen and a sidecar round trip, and a background refit
//! that lands after the forget never brings it back. Atomic units throughout.

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

/// Forget: written at once (the store's code), a refit due, not fitted by it, and never released
/// by a refit whose model reproduces it — where a held sample at the same place is (the control).
/// It fences like a held sample: its geometry falls back; a fresh answer there that agrees with
/// the model is kept, and the forgotten sample stays forgotten (never released or superseded).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forgotten_sample_is_not_fitted_and_never_released() {
    let mut s = window();
    let k = 17;
    assert!(fitted(&s, k));
    s.forget_sample(k).unwrap();
    assert_eq!((s.quarantine[k], s.forgotten(), s.quarantined()), (Q_FORGOTTEN, 1, 0));
    assert!(s.pending_refit(), "a refit is due");
    assert!(s.user_notes.iter().any(|n| n.contains("sample #17 forgotten") && n.contains("saved with the model file")), "{:?}", s.user_notes);
    assert!(s.forget_sample(k).is_err(), "forgotten already");
    assert!(s.forget_sample(s.samples()).is_err(), "no such sample");
    let rep = s.refit().unwrap();
    assert_eq!(rep.n_samples, 39, "the refit fits without it");
    assert!(!fitted(&s, k));
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
    assert!(r.stats.notes.iter().any(|n| n.contains("forgotten kept")), "{:?}", r.stats.notes);
    assert_eq!((s.quarantine[k], s.superseded(), s.samples()), (Q_FORGOTTEN, 0, before + 1));
}

/// Restore: back in the fit at the refit; a sample that is not forgotten is refused.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_restored_sample_is_fitted_again() {
    let mut s = window();
    assert!(s.restore_sample(3).is_err(), "not forgotten");
    s.forget_sample(3).unwrap();
    s.refit().unwrap();
    assert!(!fitted(&s, 3));
    s.restore_sample(3).unwrap();
    assert_eq!((s.quarantine[3], s.forgotten()), (Q_NONE, 0));
    assert!(s.pending_refit());
    let rep = s.refit().unwrap();
    assert!(fitted(&s, 3));
    assert_eq!(rep.n_samples, 40);
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
}

/// A model file keeps the code: the sidecar writes 7 and a load restores it (forgotten, excluded,
/// fencing); a forget between the fit and the save is caught by the load's membership check.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_forgotten_sample_survives_a_sidecar_round_trip() {
    let mut s = window();
    s.forget_sample(5).unwrap();
    s.refit().unwrap();
    s.forget_sample(9).unwrap(); // forgotten since the fit: the file's prefix is not the fit's
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
    s.forget_sample(4).unwrap();
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
    again.restore_sample(4).unwrap();
    assert_eq!(again.collection().unwrap().quarantine[4], Q_NONE, "the restore is written at once");
    drop(again);
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!(c.quarantine[4], Q_NONE);
}

/// A model over a fresh shared collection in `dir`: 40 H₂ samples and the H reference, fitted.
fn small_collection(dir: &std::path::Path) -> (SurrogateKs, surrogate::SurfaceKey) {
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
/// lands the same way.
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
    s.forget_sample(7).unwrap();
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
    assert!(s.pending_refit(), "a rebuild is due");
    assert!(!s.sample_suspects().iter().any(|x| x.index == 7), "not listed among the fitted samples");
    s.refit().unwrap();
    assert!(!fitted(&s, 7), "the next refit fits without it");
    assert_eq!(s.quarantine[7], Q_FORGOTTEN);
    // a Restore in flight lands the same way
    assert!(s.begin_background_refit());
    s.restore_sample(7).unwrap();
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
            s.forget_sample(i).unwrap();
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
    assert_eq!((list[0].index, list[0].formula.as_str(), list[0].origin.as_str()), (out, "H2", "pass scan"));
    let ratios: Vec<f64> = list.iter().map(|x| x.loo_ratio.unwrap()).collect();
    assert!(ratios.windows(2).all(|w| w[0] >= w[1]), "worst first: {ratios:?}");
    let all = s.loo_ratios().unwrap().to_vec();
    assert_eq!(ratios[0], all.iter().map(|x| x.1).fold(0.0, f64::max), "the worst of the fit");
    s.forget_sample(out).unwrap();
    let after = s.sample_suspects();
    assert!(!after.iter().any(|x| x.index == out), "forgotten since the fit: not listed");
    assert_eq!(after[0], list[1], "the rest move up, the same numbers (no recompute)");
    let gone = s.forgotten_samples();
    assert_eq!(gone.len(), 1);
    assert_eq!((gone[0].index, gone[0].loo_ratio), (out, Some(ratios[0])));
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
    s.forget_sample(he).unwrap();
    assert!(s.user_notes.iter().any(|n| n.contains("the only sample of its kind") && n.contains("sample counts may refuse")), "{:?}", s.user_notes);
    let r = s.refit();
    println!("after forgetting the only He sample: {:?}; trusted {} {:?}", r.as_ref().map(|r| r.n_samples), s.trusted(), s.untrusted_reason());
    assert!(s.model.is_none() || !fitted(&s, he));
    assert!(s.trusted() || s.untrusted_reason().is_some_and(|w| !w.is_empty()), "an untrusted model says why");
    assert!(s.is_forgotten(he));
}
