//! Spec `2026-11-17-recovery-warm-start-design.md` (option B): a trust recovery's trial fits are
//! searched from the previous fit's hyperparameters (`fit_trial`, `surrogate::Accumulator::refit_from`),
//! and a trial that would stand is searched again from the data (`confirm_cold`) and judged on
//! that fit. Every fit that stands is the cold search of its store state — what a reopened
//! collection computes — and a trial that fails cold is a failed trial.
//!
//! The fixture (`glitch_collection`): a shared collection (so the basis has the body-4 block and
//! the spectral search runs; H alone keeps Q′ small) of [`SpeciesMorse`]'s H₂ and H₃ answers with
//! a DFT-like scatter, and one H₂ whose energy label is 0.05 Ha off — the training residuals fail,
//! step 1's degree-cap retry cannot fit the glitch, and the quarantine rounds hold it. Measured: the
//! recovery's searches (from a start?, rounds) are the entry fit (no, 3), the cap trial (yes, 2),
//! the leave-one-out round (yes, 5) and its confirmation (no, 5); with the first confirmation
//! failed, the training ranking's round (yes, 5) and its confirmation (no, 5) follow.
use super::*;

/// The glitch's store index in [`glitch_collection`].
const GLITCH: usize = 120;

/// The module doc's fixture in a collection at `dir`: [`GLITCH`] H₂ / H₃ answers (random clusters,
/// `d4_cluster`) labelled with [`SpeciesMorse`]'s surface plus a DFT-like scatter (uniform, ±5e-5 Ha
/// on the energy, ±5e-4 Ha/Bohr per force component: exact labels make the noise levels fall to
/// their floors, where neither search meets its stopping test in 30 rounds), then the glitch; not
/// fitted. Deterministic: two calls build the same store.
fn glitch_collection(dir: &CollectionRoot) -> SurrogateKs {
    let mut s = d4_student(dir);
    let mut rng = StdRng::seed_from_u64(20261117);
    let prov = s.provenance_for(&h2(1.4));
    let mut notes = Vec::new();
    let kinds: [&[u8]; 2] = [&[1, 1], &[1, 1, 1]];
    for i in 0..=GLITCH {
        let z = kinds[i % 2];
        let pos = d4_cluster(z, &mut rng);
        let (mut e, mut f) = SpeciesMorse::energy_and_forces(&pos, z);
        e += rng.gen_range(-5e-5..5e-5);
        for fj in &mut f {
            for c in fj.iter_mut() {
                *c += rng.gen_range(-5e-4..5e-4);
            }
        }
        if i == GLITCH {
            e += 0.05;
        }
        assert!(s.keep_sample(&pos, z, e, &f, Some(0.0), Some(1), &prov, SampleTag::Scan, &mut notes), "{notes:?}");
    }
    assert_eq!(s.samples(), GLITCH + 1);
    s
}

/// The fits a provider made, as their solves (`fit_log`): (from a start, rounds) per spectral search.
fn searches(s: &SurrogateKs) -> Vec<(bool, usize)> {
    s.fit_log
        .iter()
        .filter_map(|k| match *k {
            surrogate::RefitKind::Spectral { rounds, from_start, .. } => Some((from_start, rounds)),
            _ => None,
        })
        .collect()
}

/// The codes of every stored sample.
fn codes(s: &SurrogateKs) -> Vec<u8> {
    (0..s.samples()).map(|i| s.quarantine.get(i).copied().unwrap_or(Q_NONE)).collect()
}

/// The reopen property (spec "Acceptance"): the fit a recovery leaves standing is the cold search
/// of its store state, bit for bit — the hyperparameters, the report, the coefficients and the
/// posterior factor equal those of a fresh provider fitted cold with the same samples, codes and
/// degree cap — and its warm state is the cold search's too: one more sample folded in by a warm
/// refit gives the same fit on both. At least one of the recovery's trials was searched from a
/// start.
#[test]
fn a_recoverys_standing_fit_is_a_cold_search() {
    let dir = collection_root("warm-recovery");
    let mut s = glitch_collection(&dir);
    s.refit().unwrap();
    let held: Vec<usize> = (0..s.samples()).filter(|&i| s.is_quarantined(i)).collect();
    println!(
        "recovery: trusted {} {:?}, held {held:?}, cap {:?}, D₄ {:?}; note {:?}; searches (from a start, rounds) {:?}",
        s.trusted(),
        s.untrusted_reason(),
        s.degree_cap,
        s.body4_degree(),
        s.recovery_note(),
        searches(&s)
    );
    // the premise: a recovery by quarantine on the body-4 (spectral) path
    assert!(s.model().unwrap().basis.body4, "the spectral path");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!(held, vec![GLITCH], "the glitch alone is held");
    assert!(s.recovery_note().is_some_and(|n| n.contains("quarantined this refit")), "{:?}", s.recovery_note());
    assert!(searches(&s).iter().any(|x| x.0), "at least one trial was searched from a start");
    assert!(matches!(s.accumulator.as_ref().unwrap().last_refit(), surrogate::RefitKind::Spectral { from_start: false, .. }), "the standing fit is a search from the data");
    assert!(!s.trial_fit);
    // a fresh provider: the same samples, the recovery's codes and cap, one cold fit
    let dir2 = collection_root("warm-recovery-fresh");
    let mut t = glitch_collection(&dir2);
    t.quarantine = s.quarantine.clone();
    t.degree_cap = s.degree_cap;
    t.species_dirty = true;
    t.fit_store().unwrap();
    let (rs, rt) = (s.report().unwrap(), t.report().unwrap());
    println!("standing λ {:e} σ_E {:e} σ_F {:e}; fresh λ {:e} σ_E {:e} σ_F {:e}", rs.lambda, rs.sigma_e, rs.sigma_f, rt.lambda, rt.sigma_e, rt.sigma_f);
    assert_eq!((rs.lambda.to_bits(), rs.sigma_e.to_bits(), rs.sigma_f.to_bits()), (rt.lambda.to_bits(), rt.sigma_e.to_bits(), rt.sigma_f.to_bits()), "the hyperparameters, bit for bit");
    assert_eq!(rs, rt, "the report");
    let (ms, mt) = (s.model().unwrap(), t.model().unwrap());
    assert_eq!(ms.basis, mt.basis);
    assert_eq!(ms.coeffs, mt.coeffs, "the coefficients, bit for bit");
    assert_eq!(ms.sigma_inv_chol, mt.sigma_inv_chol, "the posterior factor");
    // the warm state: one more sample on both, folded in by the warm refit
    let mut rng = StdRng::seed_from_u64(5);
    let z = [1u8, 1];
    let pos = d4_cluster(&z, &mut rng);
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &z);
    for p in [&mut s, &mut t] {
        let prov = p.provenance_for(&h2(1.4));
        let mut notes = Vec::new();
        assert!(p.keep_sample(&pos, &z, e, &f, Some(0.0), Some(1), &prov, SampleTag::Scan, &mut notes), "{notes:?}");
        p.fit_store().unwrap();
        assert_eq!(p.accumulator.as_ref().unwrap().last_refit(), surrogate::RefitKind::Warm { samples: 1 });
    }
    assert_eq!(s.report().unwrap(), t.report().unwrap(), "the warm refit continues from the cold search on both");
    assert_eq!(s.model().unwrap().coeffs, t.model().unwrap().coeffs);
}

/// Spec "Acceptance" (a trial that fails cold): with the first confirmation forced to fail after
/// its cold refit (`confirm_fail` = `Cold`), the recovery ends exactly as when that trial fails
/// its check outright (`Outright`: no cold refit) — the codes, the degree cap, the note, the fit
/// and its report — so a failed confirmation leaves no half-installed state. Both differ from the
/// recovery without the seam only by the path (here: the training ranking's result stands where the
/// leave-one-out one did).
#[test]
fn a_trial_that_fails_cold_is_a_failed_trial() {
    let run = |fail: Option<ConfirmFail>, tag: &str| {
        let dir = collection_root(tag);
        let mut s = glitch_collection(&dir);
        s.confirm_fail = fail;
        s.refit().unwrap();
        assert!(s.confirm_fail.is_none(), "{fail:?}: the seam fired");
        println!(
            "{fail:?}: trusted {}, held {:?}, cap {:?}, note {:?}; searches {:?}",
            s.trusted(),
            (0..s.samples()).filter(|&i| s.is_quarantined(i)).collect::<Vec<_>>(),
            s.degree_cap,
            s.recovery_note(),
            searches(&s)
        );
        assert!(!s.trial_fit);
        let model = s.model().unwrap().clone();
        (codes(&s), s.degree_cap, s.recovery_note().map(str::to_string), s.report().unwrap().clone(), model.coeffs, model.sigma_inv_chol, s.species_dirty)
    };
    let cold = run(Some(ConfirmFail::Cold), "warm-fail-cold");
    let outright = run(Some(ConfirmFail::Outright), "warm-fail-outright");
    assert_eq!(cold.0, outright.0, "the codes");
    assert_eq!(cold.1, outright.1, "the degree cap");
    assert_eq!(cold.2, outright.2, "the note");
    assert_eq!(cold.3, outright.3, "the report");
    assert_eq!(cold.4, outright.4, "the coefficients");
    assert_eq!(cold.5, outright.5, "the posterior factor");
    assert_eq!(cold.6, outright.6, "species_dirty");
}
