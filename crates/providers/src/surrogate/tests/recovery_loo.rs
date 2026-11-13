//! Spec `2026-11-12-sample-recovery-design.md` R1: the trust recovery ranks its offenders by their
//! **leave-one-out ratio** (`quarantine::loo_ratio`) first, and — fix round 1, option A — falls back
//! to the training-residual ranking from the same entry state when those rounds do not recover;
//! the ratios are the class pass's, cached per fit (ruling R1b); an undefined one ranks first (R1a).
//!
//! The fixture (`outlier_store`): [`SpeciesMorse`]'s exact, pair-additive H surface — 100 H₃
//! triangles (every pair 1.0–3.0 Bohr) and 30 H₂ (1.2–2.6 Bohr) — and one **high-leverage outlier**,
//! H₂ compressed to 0.9 Bohr with forces 2.5× the surface's (|F| 0.83 Ha/Bohr: its bound is the
//! 1e-2 cap). It mirrors the user's #1804 (a lone OH squeezed to 1.31 Bohr, 0.72 Ha/Bohr): the
//! fit reproduces the outlier inside its own wide bound by bending the shared H–H pair term, and
//! the victims — samples with small forces and the 2e-3–6e-3 bounds — fail.
//!
//! **The outlier's energy row is left out** (`use_energy: false`, as a capped segment's): #1804's
//! energy was fitted by directions of its own (in-sample within the tier, leave-one-out 962×),
//! and this H-only basis (Q′ 29) has no such direction for one dimer's energy — with its energy
//! (the surface's, consistent) the outlier fails its own 1e-3 Ha/atom tier in-sample, and both
//! rankings name it (measured: 1.4e-2 Ha/atom at 0.95 Bohr × 2; the exploration's other variants —
//! dimers only, a lone dimer among H₃, a compressed H₃ on another Morse — likewise, or else span
//! their own directions and untrust nothing). The force-only outlier is the closest honest variant
//! of the mechanism: a wide bound that hides the sample from its own residual.
use super::*;

/// [`SpeciesMorse`]'s H surface at the scene (`SpeciesMorse::energy_and_forces`): the teacher's
/// exact answer.
fn h_sample(sc: &Scene) -> Sample {
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let z = vec![1; pos.len()];
    let (energy, forces) = SpeciesMorse::energy_and_forces(&pos, &z);
    Sample { pos, z, energy, forces, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }
}

/// H₃: a random triangle with every pair in [lo, hi] (65³ cell).
fn h3(rng: &mut StdRng, lo: f64, hi: f64) -> Scene {
    loop {
        let p: Vec<[f64; 3]> = (0..3).map(|_| [rng.gen_range(-1.6..1.6), rng.gen_range(-1.6..1.6), rng.gen_range(-1.6..1.6)]).collect();
        let d = |a: usize, b: usize| (0..3).map(|k| (p[a][k] - p[b][k]).powi(2)).sum::<f64>().sqrt();
        if [(0, 1), (0, 2), (1, 2)].iter().all(|&(a, b)| (lo..=hi).contains(&d(a, b))) {
            let mut sc = in_cell_65(h2(1.4));
            sc.nuclei = p.into_iter().map(|x| Nucleus::new(1, x)).collect();
            return sc;
        }
    }
}

/// The outlier's store index in [`outlier_store`].
pub(super) const OUTLIER: usize = 130;

/// The module doc's fixture: 100 H₃, 30 H₂, then the outlier (index [`OUTLIER`]); not fitted.
pub(super) fn outlier_store() -> SurrogateKs {
    let (mut s, _) = pair_student();
    s.set_reference_energy(1, SpeciesMorse::atom(1));
    let mut rng = StdRng::seed_from_u64(1804);
    for _ in 0..100 {
        let sc = h3(&mut rng, 1.0, 3.0);
        s.add_sample(h_sample(&sc), &sc).unwrap();
    }
    for i in 0..30 {
        let r = 1.2 + 1.4 * i as f64 / 29.0;
        let sc = in_cell_65(dimer(r, unit_vector(&mut rng), [rng.gen_range(-0.3..0.3), 0.0, 0.0], 1, 1));
        s.add_sample(h_sample(&sc), &sc).unwrap();
    }
    let sc = in_cell_65(dimer(0.9, unit_vector(&mut rng), [0.0; 3], 1, 1));
    let mut out = h_sample(&sc);
    out.forces.iter_mut().for_each(|f| f.iter_mut().for_each(|x| *x *= 2.5));
    out.use_energy = false;
    s.add_sample(out, &sc).unwrap();
    assert_eq!(s.samples(), OUTLIER + 1);
    s
}

/// The sample's own in-sample force residual over its bound (the training ranking's number).
fn own_training_ratio(s: &SurrogateKs, k: usize) -> f64 {
    let smp = &s.store.as_ref().unwrap().samples[k];
    let pr = s.model.as_ref().unwrap().predict(&smp.pos, &smp.z, false).unwrap();
    pr.forces
        .iter()
        .zip(&smp.forces)
        .map(|(m, f)| {
            let (d, b) = s.policy.force_residual(m, f);
            d / b
        })
        .fold(0.0, f64::max)
}

/// Spec R1's pinned behaviour: the training-residual ranking would hold the victims; the
/// leave-one-out ranking holds the outlier alone and the model is trusted after the refit.
///
/// Measured (seed 1804): the entry fit is untrusted by a victim (max training force residual
/// 2.6e-2 > 5.8e-3 Ha/Bohr, the H₂ at 1.2 Bohr) while the outlier's own residual is 0.86× its
/// bound; the training ranking names 2 victims (the H₂ at 1.2 and 1.25 Bohr), and its five rounds
/// hold 7 and end untrusted (6.0e-3 > 5.4e-3 Ha/Bohr) — the user's failure — the outlier still
/// fitted; the leave-one-out ranking names the outlier alone at 28.8× (the next sample 6.0×), one
/// round, trusted. Nearby variants behave alike (0.85–1.0 Bohr, forces × 1.6–2.5, seeds 1804, 1,
/// 2, with the outlier's own ratio 0.57–0.98): the training ranking holds 5–8 victims, never the
/// outlier, and the leave-one-out ranking the outlier alone; with a gentler outlier (× ≤ 1.8 at
/// 0.85–0.95 Bohr) step 1's degree-cap retry recovers first, under either ranking. (That sweep
/// was an exploration and is not pinned; `references.md` says so.)
#[test]
fn a_high_leverage_outlier_is_held_alone_and_trust_returns() {
    // the premise, on the entry fit
    let mut s = outlier_store();
    s.fit_store().unwrap();
    let t = s.training_residuals().unwrap();
    println!("entry fit: {:?}; worst training force {:.2e} / {:.2e}; outlier's own training ratio {:.2}", s.untrusted_reason(), t.force, t.force_bound, own_training_ratio(&s, OUTLIER));
    assert!(!s.trusted() && s.residuals_fail(), "the premise: the outlier untrusts the fit: {:?}", s.untrusted_reason());
    assert!(own_training_ratio(&s, OUTLIER) <= 1.0, "the premise: the outlier is inside its own bound");
    let (old, _) = s.training_offenders();
    assert!(!old.is_empty() && !old.contains(&OUTLIER), "the premise: the training ranking names victims, not the outlier: {old:?}");
    let (new, worst) = s.worst_offenders();
    println!("training ranking {old:?}; leave-one-out ranking {new:?} (worst {worst:.2})");
    assert_eq!(new, vec![OUTLIER], "the leave-one-out ranking names the outlier alone");
    let ratios = s.loo_ratios().unwrap().to_vec();
    let next = ratios.iter().filter(|r| r.0 != OUTLIER).map(|r| r.1).fold(0.0, f64::max);
    println!("leave-one-out: outlier {worst:.2}, next {next:.2}");
    assert!(worst > 2.0 * next, "every other sample below half the outlier's ratio");

    // the rounds the training ranking would have run: victims held, the outlier fitted, the
    // residuals still failing (the rounds alone, before the refit's no-ratchet restore)
    let mut rounds = outlier_store();
    rounds.training_ranking = true;
    rounds.fit_store().unwrap();
    let report = rounds.report().unwrap().clone();
    let _ = rounds.quarantine_rounds(report);
    println!("training ranking's rounds: {} held, outlier held {}, residuals fail {}", rounds.quarantined(), rounds.is_quarantined(OUTLIER), rounds.recoverable_fail());
    assert!(rounds.quarantined() >= 5 && !rounds.is_quarantined(OUTLIER), "the training ranking holds victims");
    assert!(rounds.recoverable_fail(), "and does not restore the residuals");
    // the refit by that ranking: since the open-shell merge (spec 2026-11-13 §4.2, no ratchet) a
    // recovery that does not restore the residuals gives its holds back — adapted from main's
    // "victims held" on the refit itself
    let mut old_way = outlier_store();
    old_way.training_ranking = true;
    old_way.refit().unwrap();
    println!(
        "training ranking's refit: trusted {}, {} held ({:?}), outlier held {}; {:?}",
        old_way.trusted(),
        old_way.quarantined(),
        (0..old_way.samples()).filter(|&i| old_way.is_quarantined(i)).collect::<Vec<_>>(),
        old_way.is_quarantined(OUTLIER),
        old_way.untrusted_reason()
    );
    assert_eq!(old_way.quarantined(), 0, "the failed recovery holds nothing (no ratchet)");
    assert!(!old_way.trusted(), "and ends untrusted: {:?}", old_way.untrusted_reason());

    // the refit as it is: the outlier alone, trusted
    let mut s = outlier_store();
    s.refit().unwrap();
    let held: Vec<usize> = (0..s.samples()).filter(|&i| s.is_quarantined(i)).collect();
    println!("leave-one-out ranking's refit: trusted {}, held {held:?}; note {:?}", s.trusted(), s.recovery_note());
    assert_eq!(held, vec![OUTLIER], "the outlier is held, no victim is");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!(s.degree_cap, None, "step 2 recovered it on the default basis (step 1's cap retry did not)");
    assert_eq!(
        s.recovery_note(),
        Some("1 samples quarantined this refit (1 held in all) — the fit without each of them predicts it at up to 29× its bound, and the fit without them recovers (kept, excluded from the fit; geometries near them fall back)")
    );
    assert_eq!(s.samples(), OUTLIER + 1, "never dropped");
}

/// [`loo_ratio`] on hand-made residuals (atomic units): ρ_E = |r_E|/N/tier for a whole-energy fit,
/// |r_E|/tier per atom; d_j = ‖r_j‖/√3 against the atom's own bound max(2e-3, 10 %|F_j|) capped at
/// 1e-2 (ρ) and against `force_std_abs` (ρ_σ); a masked-out atom (`None`) is skipped; an undefined
/// leave-one-out and a NaN are ∞.
#[test]
fn the_leave_one_out_ratio_is_the_class_rules_reading() {
    use surrogate::loo::SampleLoo;
    let p = FallbackPolicy::default();
    let s = Sample {
        pos: vec![[0.0; 3], [1.4, 0.0, 0.0], [0.0, 3.0, 0.0]],
        z: vec![1, 1, 1],
        energy: -1.5,
        // |F| 0.5 (bound capped at 1e-2), 0.01 (bound the 2e-3 floor), 0.05 (masked out below)
        forces: vec![[0.5, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.05]],
        tag: SampleTag::Scan,
        weight: 1.0,
        force_mask: None,
        use_energy: true,
    };
    let l = SampleLoo { energy: Some(-6e-3), forces: vec![Some([0.0, 0.03 * 3f64.sqrt(), 0.0]), Some([0.0, 0.0, 0.004 * 3f64.sqrt()]), None] };
    // energy: 6e-3 / 3 atoms / 1e-3 = 2 (whole-energy fit), 6 per atom;
    // atom 0: d 0.03 / 1e-2 = 3 (ρ), / 2e-3 = 15 (ρ_σ); atom 1: d 0.004 / 2e-3 = 2 both
    // (ρ, ρ_σ, the channel attaining ρ_σ — re-review N1)
    let close = |a: (f64, f64, Channel), b: (f64, f64, Channel)| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12 && a.2 == b.2;
    assert!(close(loo_ratio(&p, false, &s, Some(&l)), (3.0, 15.0, Channel::Force)), "{:?}", loo_ratio(&p, false, &s, Some(&l)));
    assert!(close(loo_ratio(&p, true, &s, Some(&l)), (6.0, 15.0, Channel::Force)), "{:?}", loo_ratio(&p, true, &s, Some(&l)));
    let no_energy = Sample { use_energy: false, ..s.clone() };
    let l2 = SampleLoo { energy: None, ..l.clone() };
    assert!(close(loo_ratio(&p, false, &no_energy, Some(&l2)), (3.0, 15.0, Channel::Force)));
    let small = SampleLoo { energy: Some(1e-4), forces: vec![Some([0.0; 3]), Some([0.0; 3]), None] };
    assert!(close(loo_ratio(&p, true, &s, Some(&small)), (0.1, 0.1, Channel::Energy)), "the energy channel attains ρ_σ");
    // an energy outlier whose forces the rest predicts: ρ_σ is its energy ratio (#1804's shape)
    let energy_outlier = SampleLoo { energy: Some(0.05), forces: vec![Some([0.0, 0.003, 0.0]), Some([0.0; 3]), None] };
    let (own, tier, ch) = loo_ratio(&p, true, &s, Some(&energy_outlier));
    assert!((own - 50.0).abs() < 1e-9 && (tier - 50.0).abs() < 1e-9 && ch == Channel::Energy, "{own} {tier} {ch:?}");
    assert_eq!(loo_ratio(&p, false, &s, None), (f64::INFINITY, f64::INFINITY, Channel::Force), "undefined (ruling R1a)");
    let nan = SampleLoo { energy: Some(f64::NAN), ..l.clone() };
    let (own, tier, _) = loo_ratio(&p, false, &s, Some(&nan));
    assert_eq!((own, tier), (f64::INFINITY, f64::INFINITY), "not a number: the worst");
}

/// Re-review N1 and N2, Task 2 fix round 1 (M6 and the ruling on concern 1): the note's words. In
/// the noise-only mode it names the tier of the channel that attained a sample's ratio (an energy
/// outlier's against its 1e-3 Ha/atom energy tier, not the force tier); rounds held under
/// different bounds state each bound's own maximum (no understatement); the last clause is the
/// measured fact only — ", and the fit without them recovers" (residuals, noise and counts),
/// "… passes" (residuals and noise, not the counts), nothing when it still fails; it never claims
/// the held samples broke their neighbours (a ratio above 1 says only that the rest of the data
/// could not predict the sample).
#[test]
fn the_recovery_note_names_the_channel_and_claims_no_repair_it_did_not_see() {
    let held = |pairs: &[(f64, Bound)]| {
        let mut h = Held::NONE;
        for &(r, b) in pairs {
            h.add(r, b);
        }
        h
    };
    assert_eq!(
        quarantine_words(1, 1, held(&[(962.0, Bound::EnergyTier(1e-3))]), Ranking::LeaveOneOut, Outcome::Recovers),
        "1 samples quarantined this refit (1 held in all) — the fit without each of them predicts it at up to 962× its 1e-3 Ha/atom energy tier, and the fit without them recovers (kept, excluded from the fit; geometries near them fall back)"
    );
    assert_eq!(
        quarantine_words(3, 2, held(&[(4.24, Bound::ForceTier(2e-3)), (2.1, Bound::ForceTier(2e-3))]), Ranking::LeaveOneOut, Outcome::Fails),
        "2 samples quarantined this refit (3 held in all) — the fit without each of them predicts it at up to 4.2× the 2e-3 Ha/Bohr force-noise tier (kept, excluded from the fit; geometries near them fall back)"
    );
    assert_eq!(
        quarantine_words(1, 1, held(&[(f64::INFINITY, Bound::Own)]), Ranking::LeaveOneOut, Outcome::Fails),
        "1 samples quarantined this refit (1 held in all) — the fit without one of them cannot predict it at all (it alone spans a direction of the fit) (kept, excluded from the fit; geometries near them fall back)"
    );
    // M6: an own-bound round's 29× and a later noise-only round's 6× and 3× (energy): each bound
    // with its own maximum, never "up to 6× the tier" for the 29× sample
    assert_eq!(
        quarantine_words(4, 3, held(&[(29.0, Bound::Own), (6.0, Bound::ForceTier(2e-3)), (3.0, Bound::EnergyTier(1e-3))]), Ranking::LeaveOneOut, Outcome::Passes),
        "3 samples quarantined this refit (4 held in all) — the fit without each of them predicts it at up to 29× its bound and 6.0× the 2e-3 Ha/Bohr force-noise tier and 3.0× its 1e-3 Ha/atom energy tier, and the fit without them passes (kept, excluded from the fit; geometries near them fall back)"
    );
    assert_eq!(held(&[(29.0, Bound::Own), (31.0, Bound::ForceTier(2e-3))]).worst(), 31.0);
    for o in [Outcome::Fails, Outcome::Passes, Outcome::Recovers] {
        assert!(!quarantine_words(128, 23, held(&[(2.2, Bound::Own)]), Ranking::LeaveOneOut, o).contains("neighbours"), "no claim about the neighbours ({o:?})");
    }
    assert_eq!((ratio_words(38.4), ratio_words(9.96), ratio_words(f64::INFINITY)), ("38".to_string(), "10.0".to_string(), "∞".to_string()));
}

/// Ruling R1b and the plan's review focus 2: the class pass leaves its ratios cached for its own
/// fit — the numbers a fresh computation gives, bit for bit. A code moved between fits (a hold, a
/// release) does not change the fit, and the ratios stay that fit's: a sample held since is still
/// listed, one released since never is (the model never saw it). A new fit makes the cache stale;
/// it is recomputed over the new fit's members, never read; a restored fit takes back its own; a
/// forget drops it.
#[test]
fn the_leave_one_out_ratios_are_cached_for_their_own_fit_only() {
    let mut s = outlier_store();
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[7] = Q_HELD; // held at the fit
    s.fit_store().unwrap(); // with the class pass
    let cached = s.fresh_loo_ratios().cloned().expect("the class pass left this fit's ratios");
    assert_eq!(cached.gen, s.model_gen);
    assert_eq!(cached.rows.len(), OUTLIER, "every fitted sample (#7 is held)");
    assert!(cached.rows.iter().all(|r| r.0 != 7));
    s.loo_ratios = None;
    let fresh = s.ensure_loo_ratios().cloned().unwrap();
    assert_eq!(fresh, cached, "the class pass's numbers are a fresh computation's, bit for bit");
    // between fits: the outlier held, #7 released — the fit that stands is unchanged
    let saved = s.save_state();
    s.quarantine[OUTLIER] = Q_HELD;
    s.quarantine[7] = Q_NONE;
    assert_eq!(s.fresh_loo_ratios(), Some(&cached), "still the standing fit's");
    s.loo_ratios = None;
    assert_eq!(s.loo_ratios().unwrap(), &cached.rows[..], "recomputed over the fit's own members: the released #7 is no member");
    // the next fit (without the outlier, with #7): a new generation, its own ratios
    s.species_dirty = true;
    s.fit_store_deferred().unwrap();
    assert!(s.fresh_loo_ratios().is_none(), "a deferred fit has no class pass: the cache is an earlier fit's");
    let refit = s.loo_ratios().unwrap().to_vec();
    assert_eq!(s.loo_ratios.as_ref().unwrap().gen, s.model_gen);
    assert!(refit.iter().any(|r| r.0 == 7) && refit.iter().all(|r| r.0 != OUTLIER), "the new fit's members");
    // the restored fit takes back its own (the caller restores the codes)
    s.restore_state(saved);
    s.quarantine[OUTLIER] = Q_NONE;
    s.quarantine[7] = Q_HELD;
    let back = s.fresh_loo_ratios().cloned().expect("the saved fit's ratios, re-keyed to the restored generation");
    assert_eq!((back.rows, back.tier), (cached.rows.clone(), cached.tier.clone()));
    assert_eq!(back.gen, s.model_gen);
    // a forget: nothing
    s.forget();
    assert!(s.loo_ratios.is_none() && s.loo_ratios().is_none());
}

/// Ruling R1a and the plan's review focus 1: a sample whose leave-one-out is undefined ranks
/// first — while one exists only those are taken — and when more of them than the 10 % cap
/// allows are named, the round holds none of them (the cap is checked before holding); the trust
/// rule then judges the fit that stands (untrusted, honestly).
#[test]
fn undefined_leave_one_out_ranks_first_and_never_crosses_the_cap() {
    let mut s = outlier_store();
    s.fit_store().unwrap();
    assert!(s.residuals_fail());
    let gen = s.model_gen;
    let rows = s.loo_ratios().unwrap().to_vec();
    // as if these samples alone spanned a direction of the fit
    let inject = |s: &mut SurrogateKs, undefined: &[usize]| {
        let rows: Vec<(usize, f64)> = rows.iter().map(|&(i, r)| (i, if undefined.contains(&i) { f64::INFINITY } else { r })).collect();
        let tier = rows.iter().map(|r| r.1).collect();
        let tier_channel = vec![Channel::Force; rows.len()];
        s.loo_ratios = Some(LooRatios { gen, rows, tier, tier_channel });
    };
    inject(&mut s, &[3, 40]);
    assert_eq!(s.worst_offenders(), (vec![3, 40], f64::INFINITY), "only the undefined ones, ahead of the outlier");
    // more undefined than the cap allows (10 % of 131 = 13.1): nothing is held
    let many: Vec<usize> = (0..20).collect();
    inject(&mut s, &many);
    let report = s.report().unwrap().clone();
    let (_, held, ranking) = s.quarantine_rounds(report);
    let worst = held.worst();
    println!("20 undefined: held {}, trusted {} {:?}; {ranking:?}", s.quarantined(), s.trusted(), s.untrusted_reason());
    // the leave-one-out rounds hold none; the training ranking's fallback (7 victims, untrusted on
    // this fixture) does not recover either, so the leave-one-out result stands
    assert_eq!((s.quarantined(), worst, ranking), (0, 0.0, Ranking::LeaveOneOut), "the cap is checked before holding");
    assert!(!s.trusted(), "the trust rule judges what stands");
}

/// A background refit's install takes the copy's ratios with its fit: fresh for the installed
/// model, the numbers of a fresh computation.
#[test]
fn a_background_refit_installs_its_leave_one_out_ratios() {
    let mut s = outlier_store();
    s.refit().unwrap();
    let sc = in_cell_65(dimer(1.5, [0.0, 1.0, 0.0], [0.0; 3], 1, 1));
    s.add_sample(h_sample(&sc), &sc).unwrap();
    assert!(s.begin_background_refit());
    assert!(matches!(s.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
    let installed = s.fresh_loo_ratios().cloned().expect("the copy's ratios, fresh for the installed fit");
    assert_eq!(installed.rows.len(), OUTLIER + 1, "the members: all but the held outlier, and the new sample");
    s.loo_ratios = None;
    assert_eq!(s.ensure_loo_ratios().cloned().unwrap(), installed);
}

/// A fresh copy of the collection folder `SURROGATE_RECOVERY_SNAPSHOT` names, opened (the copy's
/// temp root is `tag`'s).
fn snapshot_copy(tag: &str) -> surrogate::Collection {
    let src = std::path::PathBuf::from(std::env::var("SURROGATE_RECOVERY_SNAPSHOT").expect("SURROGATE_RECOVERY_SNAPSHOT names a collection folder copy"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(src.join("key.txt")).unwrap()).unwrap();
    let root = collection_root(tag);
    let dst = root.join(key.slug());
    std::fs::create_dir_all(&dst).unwrap();
    for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
        if src.join(f).exists() {
            std::fs::copy(src.join(f), dst.join(f)).unwrap();
        }
    }
    surrogate::Collection::open(&root, &key).unwrap().0
}

/// The worst training ratio of the fit that stands and the samples above `floor`, worst first.
fn training_ratios(s: &SurrogateKs, floor: f64) -> (f64, Vec<(usize, f64)>) {
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let mut out: Vec<(usize, f64)> = Vec::new();
    for (i, smp) in store.samples[..s.accumulated.min(store.len())].iter().enumerate() {
        if s.excluded(i) {
            continue;
        }
        let Ok(pr) = model.predict(&smp.pos, &smp.z, false) else { continue };
        let n = smp.pos.len().max(1) as f64;
        let mut r = if smp.use_energy { (pr.energy - smp.energy).abs() / n / s.policy.energy_std_per_atom } else { 0.0 };
        for (j, (fm, fs)) in pr.forces.iter().zip(&smp.forces).enumerate() {
            if smp.trains_atom(j) {
                let (d, b) = s.policy.force_residual(fm, fs);
                r = r.max(d / b);
            }
        }
        out.push((i, r));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    let worst = out.first().map_or(0.0, |x| x.1);
    (worst, out.into_iter().filter(|x| x.1 > floor).collect())
}

/// The steps behind [`recovery_on_a_collection_snapshot`]'s result, one fit at a time (a
/// diagnostic; same copy rule): the entry fit from the file's codes, the release, the
/// recovery's rounds as `quarantine_rounds` takes them — and, for comparison, #1804 held before
/// the entry fit (the collection debug's "#1804 only" measurement, which has no release step).
///
/// `SURROGATE_RECOVERY_SNAPSHOT=<copy>/<slug> cargo test -p providers --release --lib recovery_snapshot_steps -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic on a copy of a collection snapshot (SURROGATE_RECOVERY_SNAPSHOT)"]
fn recovery_snapshot_steps() {
    let c = snapshot_copy("recovery-steps");
    let disk = c.quarantine.clone();
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.attach(c).unwrap();
    let lone: Vec<usize> = (0..s.quarantine.len()).filter(|&i| s.quarantine[i] == Q_LONE).collect();
    let set_codes = |s: &mut SurrogateKs, codes: &[u8]| {
        s.quarantine = codes.to_vec();
        s.quarantine.resize(s.samples(), Q_NONE);
        for &i in &lone {
            s.quarantine[i] = Q_LONE;
        }
        s.degree_cap = None;
        s.species_dirty = true;
        s.fit_store_deferred().unwrap();
    };
    let show = |s: &mut SurrogateKs, label: &str| {
        let (worst, over) = training_ratios(s, 1.0);
        let (loo, lw) = s.worst_offenders();
        println!(
            "{label}: n {}, held {}, worst training ratio {worst:.4}, {} samples > 1 {:?}; residuals pass {}; LOO offenders {} (worst {lw:.2}) {:?}",
            s.report().unwrap().n_samples,
            s.quarantined(),
            over.len(),
            &over[..over.len().min(6)],
            !s.recoverable_fail(),
            loo.len(),
            &loo[..loo.len().min(12)]
        );
    };
    // (a) the collection debug's measurement: #1804 out before the entry fit, no release
    let mut codes = disk.clone();
    codes[1804] = Q_HELD;
    set_codes(&mut s, &codes);
    show(&mut s, "(a) #1804 held before the entry fit, nothing released");
    // (b) the refit's own order: the entry fit, the release, then the rounds
    set_codes(&mut s, &disk);
    show(&mut s, "(b) entry fit (the file's codes)");
    let before: Vec<u8> = s.quarantine.clone();
    let released = s.release_reproduced();
    let which: Vec<usize> = (0..before.len()).filter(|&i| before[i] == Q_HELD && s.quarantine[i] != Q_HELD).collect();
    s.species_dirty = true;
    s.fit_store_deferred().unwrap();
    show(&mut s, &format!("    released {released} {which:?} (by the fit with #1804), refitted"));
    let n_store = s.samples();
    for round in 1..=QUARANTINE_ROUNDS {
        if !s.recoverable_fail() {
            break;
        }
        let (worst, _) = s.worst_offenders();
        if round == 1 && n_store == 1805 {
            assert_eq!(worst, vec![1804], "round 1 holds #1804 alone");
        }
        if worst.is_empty() || (s.quarantined() + worst.len()) as f64 > QUARANTINE_MAX_FRACTION * n_store as f64 {
            println!("    round {round}: stops ({} offenders)", worst.len());
            break;
        }
        s.quarantine.resize(n_store, Q_NONE);
        for &i in &worst {
            s.quarantine[i] = Q_HELD;
        }
        s.species_dirty = true;
        s.fit_store_deferred().unwrap();
        show(&mut s, &format!("    round {round}: held {} ({:?})", worst.len(), &worst[..worst.len().min(12)]));
    }
    // (c) #1804 held and the released sample(s) back in the fit, nothing else
    let mut codes = disk.clone();
    codes[1804] = Q_HELD;
    for &i in &which {
        codes[i] = Q_NONE;
    }
    set_codes(&mut s, &codes);
    show(&mut s, &format!("(c) #1804 held and {which:?} released"));
}

/// One line of the measurement: the standing fit's held set, #1804, the worst training ratios, the
/// verdict.
fn report_line(s: &SurrogateKs, label: &str, wall: f64) {
    let t = s.training_residuals().unwrap();
    let r = s.report().unwrap();
    println!(
        "{label}: {wall:.1} s, {} fits so far; Q′ {}, n {}; held {} (cap {:.1}), #1804 held {}; worst training E/atom {:.2e} (ratio {:.3}), F {:.2e} / {:.2e} (ratio {:.4}); trusted {} {:?}\n    recovery: {:?}",
        s.fits,
        r.q_prime,
        r.n_samples,
        s.quarantined(),
        QUARANTINE_MAX_FRACTION * s.samples() as f64,
        s.is_quarantined(1804),
        t.energy_per_atom,
        t.energy_per_atom / s.policy.energy_std_per_atom,
        t.force,
        t.force_bound,
        t.force / t.force_bound,
        s.trusted(),
        s.untrusted_reason(),
        s.recovery_note()
    );
}

/// The spec's measurement (R1, "Pinned behaviour"; `references.md` "Leave-one-out recovery
/// ranking"): a **copy** of a collection snapshot is attached and refitted as the app does, then
/// refitted once more (the app's next refit, nothing new). The folder (a collection directory:
/// `key.txt`, `samples.bin`, …) comes from `SURROGATE_RECOVERY_SNAPSHOT` and is copied again into
/// a fresh temp root here, so neither it nor any user folder is written. Prints the held set's
/// size, whether #1804 is held, the worst training ratios, the trust verdict and the wall time.
///
/// On the user's pre-1805 snapshot it asserts what the ranking itself decides: #1804 is held.
/// Measured 2026-11-12 (fix round 1, pinned in `references.md`): the first refit, 13 fits (the 5
/// fallback rounds do not recover either), ends untrusted at 1.0185× the force bound with 128
/// held; the next trusted at 0.9950× with 122. The spec's "trusted, fewer than 110 held" was wrong
/// ([`recovery_snapshot_steps`]: the refit's release step, which runs before the recovery against
/// the fit #1804 still bends, puts #1455 back, and the fit without #1804 but with #1455 is at
/// 1.0070×, against 0.9991× without both — the collection debug measured the latter).
///
/// `SURROGATE_RECOVERY_SNAPSHOT=<copy>/<slug> cargo test -p providers --release --lib recovery_on_a_collection_snapshot -- --ignored --nocapture`
#[test]
#[ignore = "measurement on a copy of a collection snapshot (SURROGATE_RECOVERY_SNAPSHOT)"]
fn recovery_on_a_collection_snapshot() {
    let c = snapshot_copy("recovery-snapshot");
    let disk_held = c.quarantine.iter().filter(|&&q| q == Q_HELD).count();
    println!("snapshot: {} samples, {disk_held} held on disk", c.samples.len());
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let t0 = std::time::Instant::now();
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
    }
    report_line(&s, "attach + refit", t0.elapsed().as_secs_f64());
    let mut top = s.loo_ratios().unwrap().to_vec();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("    leave-one-out ranking of the fit that stands, top 5: {:?}", &top[..top.len().min(5)]);
    let held_1804 = s.is_quarantined(1804);
    let (fits, t1) = (s.fits, std::time::Instant::now());
    SurrogateKs::refit(&mut s).unwrap();
    report_line(&s, &format!("the next refit ({} fits)", s.fits - fits), t1.elapsed().as_secs_f64());
    if s.samples() == 1805 {
        assert!(held_1804, "#1804 is held");
    }
}

/// Fix round 2, M4: the **energy channel** of the ratio on a real fit. A dimer whose energy label
/// is 0.1 Ha off (`a_stray_sample_is_quarantined_then_superseded_and_trust_returns`'s glitch) among
/// 40 exact ones: its ratio is the energy channel |r_E|/N/`energy_std_per_atom` of its own
/// leave-one-out (the fit's energy per atom or not, as `loo_ratio` reads it), far above its force
/// channel, and the leave-one-out ranking names it alone.
#[test]
fn the_energy_channel_of_the_ratio_names_an_energy_glitch() {
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    let stray = dimer(1.63, [0.0, 1.0, 0.0], [0.1, 0.0, 0.0], 1, 1);
    let mut bad = morse_sample(&MorseDimer::h2(), &stray);
    bad.energy += 0.1;
    s.add_sample(bad, &stray).unwrap();
    s.fit_store().unwrap();
    let (model, smp) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap().samples[40].clone());
    let l = surrogate::loo::leave_one_out(model, &smp).expect("defined");
    let e = l.energy.unwrap().abs() / if model.energy_per_atom { 1.0 } else { 2.0 } / s.policy.energy_std_per_atom;
    let f = l
        .forces
        .iter()
        .zip(&smp.forces)
        .filter_map(|(r, f)| r.map(|r| {
            let (d, b) = s.policy.force_residual(&[f[0] - r[0], f[1] - r[1], f[2] - r[2]], f);
            d / b
        }))
        .fold(0.0, f64::max);
    let rho = s.loo_ratios().unwrap().iter().find(|x| x.0 == 40).unwrap().1;
    println!("energy glitch: ρ {rho:.2} = energy channel {e:.2} (force channel {f:.2e})");
    assert_eq!(rho, e, "the ratio is the energy channel");
    assert!(e > 10.0 * f.max(1.0), "the energy channel decides");
    let c = s.fresh_loo_ratios().unwrap();
    let k = c.rows.iter().position(|x| x.0 == 40).unwrap();
    assert_eq!(c.tier_channel[k], Channel::Energy, "and the cache says so, for the noise-only note (re-review N1)");
    assert_eq!(s.worst_offenders().0, vec![40], "and names the glitch alone");
}

/// Fix round 2, I1 (the coordinator's ruling): step 3 — the default-basis retry when an earlier
/// recovery set `degree_cap` — runs whenever step 2's result is **not recovered**, also when that
/// result passes its residuals but is short of samples (Q′ grows with the cap). The store:
/// `dimers_and_a_tight_triangle(49)` (49 H₂ and one tight H₃, Q′ 51 at cap 6, 29 at cap 4) plus one
/// dimer whose energy label is 0.05 Ha off, entered at cap 6 as an earlier recovery would leave it.
/// Step 1b (the default basis alone) fails on both outliers; step 2 (cap 6) holds the glitch and
/// the triangle (#50, #49): residuals pass, but 49 samples < Q′ 51 — not recovered, by either
/// ranking, so the leave-one-out result stands with `recoverable_fail()` false. Measured with the
/// old gate ("residuals or noise fail"): step 3 was skipped and the refit ended untrusted, "n_samples
/// 49 < 51", cap 6 — this test failed. Gated on "not recovered", step 3 runs from the entry codes
/// on the default basis (Q′ 29) and recovers: the same two held, trusted. (The twin pins the
/// premise with the glitch alone held: residuals pass, short of samples.)
#[test]
fn step_3_runs_when_step_2_passes_its_residuals_but_is_short_of_samples() {
    let make = || {
        let mut s = dimers_and_a_tight_triangle(49);
        let glitch = in_cell_65(dimer(1.7, [0.0, 0.6, 0.8], [0.0; 3], 1, 1));
        let mut bad = morse_sample(&MorseDimer::h2(), &glitch);
        bad.energy += 0.05;
        s.add_sample(bad, &glitch).unwrap();
        s.degree_cap = Some(RECOVERY_DEGREE_CAP);
        s
    };
    let glitch = 50;
    // the premise, on a twin: at cap 6 with the glitch held the residuals pass, the samples do not
    let mut twin = make();
    twin.fit_store().unwrap();
    assert!(twin.recoverable_fail(), "the premise: the entry fit fails its residuals: {:?}", twin.untrusted_reason());
    twin.quarantine.resize(twin.samples(), Q_NONE);
    twin.quarantine[glitch] = Q_HELD;
    twin.species_dirty = true;
    twin.fit_store().unwrap();
    println!("twin, cap 6, glitch held: residuals fail {}, short of samples {} ({:?})", twin.recoverable_fail(), twin.trust_needs_samples(), twin.untrusted_reason());
    assert!(!twin.recoverable_fail() && twin.trust_needs_samples(), "the premise: step 2's result passes its residuals but is short of samples");
    // the refit: step 3 runs and recovers on the default basis
    let mut s = make();
    s.refit().unwrap();
    let held: Vec<usize> = (0..s.samples()).filter(|&i| s.is_quarantined(i)).collect();
    println!("after the refit: cap {:?}, held {held:?}, trusted {} {:?}; {:?}", s.degree_cap, s.trusted(), s.untrusted_reason(), s.recovery_note());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!(s.degree_cap, None, "step 3: the default basis");
    assert!(s.is_quarantined(glitch));
    assert_eq!(held, vec![49, 50], "the triangle and the glitch, as the doc says (re-review N3)");
    assert!(s.recovery_note().is_some_and(|n| n.starts_with("back to the default basis")), "{:?}", s.recovery_note());
}

/// Fix round 1 (option A): where the leave-one-out rounds do not recover, the training-residual
/// rounds from the same entry state do, and their result stands — the user's stuck-H collection
/// (`a_recovery_cap_that_spreads_the_misfit_does_not_leave_the_collection_stuck`, the same live
/// state; it passes unchanged through this path). Step 3's leave-one-out rounds end at 47 held,
/// 1.60× (`stuck_collection_rounds_under_both_rankings`); the fallback's at 38 held, trusted, and the
/// note names it.
#[test]
fn the_stuck_collection_recovers_through_the_training_fallback() {
    const SLUG: &str = "lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5";
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/stuck-h-collection").join(SLUG);
    let dir = collection_root("stuck-fallback");
    std::fs::create_dir_all(dir.join(SLUG)).unwrap();
    for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
        std::fs::copy(src.join(f), dir.join(SLUG).join(f)).unwrap();
    }
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(src.join("key.txt")).unwrap()).unwrap();
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    let live = c.quarantine.clone();
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.body4_off = true;
    s.attach(c).unwrap();
    s.quarantine = live;
    s.degree_cap = Some(RECOVERY_DEGREE_CAP);
    s.species_dirty = true;
    s.fit_store().unwrap();
    assert!(!s.trusted());
    s.refit().unwrap();
    let note = s.recovery_note().unwrap_or_default().to_string();
    println!("after the refit: cap {:?}, held {}, trusted {} {:?}; {note}", s.degree_cap, s.quarantined(), s.trusted(), s.untrusted_reason());
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!((s.degree_cap, s.quarantined()), (None, 38), "step 3 on the default basis, the training ranking's 38");
    assert!(s.is_quarantined(575));
    assert!(note.starts_with("back to the default basis") && note.contains("ranked by their own training residual (the leave-one-out ranking did not recover the fit)"), "{note}");
}

/// A diagnostic (no assertion; in-repo data, < 1 s): the stuck-H collection's recovery
/// (`a_recovery_cap_that_spreads_the_misfit_does_not_leave_the_collection_stuck`; `references.md`
/// "M8a stuck collection") round by round under both rankings, on the capped basis (step 2) and on
/// the default basis from the entry codes (step 3) — the rounds as `quarantine_rounds` takes them.
/// Measured 2026-11-12: the training ranking's step 3 holds #575 first (in-sample 21.8×) and ends
/// trusted at 38 held (0.961); the leave-one-out ranking's holds #561 and #575 (22.4×), then #560,
/// then high-leverage samples whose in-sample residuals pass (26 in round 4, worst 3.3×), and ends
/// untrusted at 47 held (1.597) — why fix round 1 falls back to the training ranking
/// (`the_stuck_collection_recovers_through_the_training_fallback`).
///
/// `cargo test -p providers --release --lib stuck_collection_rounds_under_both_rankings -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic: prints the stuck collection's rounds under both rankings"]
fn stuck_collection_rounds_under_both_rankings() {
    const SLUG: &str = "lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5";
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/stuck-h-collection").join(SLUG);
    for training in [true, false] {
        let dir = collection_root(if training { "stuck-rounds-t" } else { "stuck-rounds-l" });
        std::fs::create_dir_all(dir.join(SLUG)).unwrap();
        for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
            std::fs::copy(src.join(f), dir.join(SLUG).join(f)).unwrap();
        }
        let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(src.join("key.txt")).unwrap()).unwrap();
        let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
        let live = c.quarantine.clone();
        let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.body4_off = true;
        s.attach(c).unwrap();
        for cap in [Some(RECOVERY_DEGREE_CAP), None] {
            s.quarantine = live.clone();
            s.degree_cap = cap;
            s.species_dirty = true;
            s.fit_store_deferred().unwrap();
            println!("== {} ranking, cap {cap:?}", if training { "training" } else { "loo" });
            for round in 1..=QUARANTINE_ROUNDS {
                let (w, ratio) = if training { s.training_offenders() } else { s.worst_offenders() };
                let (tw, tover) = training_ratios(&s, 1.0);
                println!("  round {round}: worst training {tw:.2} ({} > 1), held {}, offenders {} (worst {ratio:.2}) {:?}", tover.len(), s.quarantined(), w.len(), &w[..w.len().min(10)]);
                if !s.recoverable_fail() || w.is_empty() || (s.quarantined() + w.len()) as f64 > 0.1 * 578.0 {
                    println!("  stop: recoverable_fail {}", s.recoverable_fail());
                    break;
                }
                for &i in &w {
                    s.quarantine[i] = Q_HELD;
                }
                s.species_dirty = true;
                s.fit_store_deferred().unwrap();
            }
            let (tw, _) = training_ratios(&s, 1.0);
            println!("  end: worst training {tw:.3}, held {}, trusted-by-residuals {}", s.quarantined(), !s.recoverable_fail());
            if !training {
                let mut r = s.loo_ratios().unwrap().to_vec();
                r.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("  LOO top 8: {:?}", &r[..8]);
                println!("  #575 LOO: {:?}", r.iter().find(|x| x.0 == 575));
            }
        }
    }
}
