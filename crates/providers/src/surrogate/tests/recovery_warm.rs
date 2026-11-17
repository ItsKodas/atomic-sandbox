//! Spec `2026-11-17-recovery-warm-start-design.md` (option B): a trust recovery's trial fits are
//! searched from the previous fit's hyperparameters (`fit_trial`, `surrogate::Accumulator::refit_from`),
//! and a trial that would stand or end the rounds is searched again from the data (`confirm_cold`) and judged on
//! that fit. Every fit that stands is the cold search of its store state — what a reopened
//! collection computes — and a trial that fails cold is a failed trial.
//!
//! The fixture (`glitch_collection`): a shared collection (so the basis has the body-4 block and
//! the spectral search runs; H alone keeps Q′ small) of [`SpeciesMorse`]'s H₂ and H₃ answers with
//! a DFT-like scatter, and one H₂ whose energy label is 0.05 Ha off — the training residuals fail,
//! step 1's degree-cap retry cannot fit the glitch, and the quarantine rounds hold it. Measured: the
//! recovery's searches (from a start?, rounds) are the entry fit (no, 3), the cap trial (no, 3),
//! the leave-one-out round (yes, 5) and its confirmation at the rounds' stop test (no, 5).
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
        .filter_map(|k| match k.0 {
            surrogate::RefitKind::Spectral { rounds, from_start, .. } => Some((from_start, rounds)),
            _ => None,
        })
        .collect()
}

/// Fix round 1 (the coordinator's ruling: option B never makes a refit slower than main): the cap
/// retry of step 1 is searched from the data, not from a start — a single trial that usually stands
/// would pay a confirmation on top — while the quarantine rounds after it are warm-started. On the
/// fixture step 1's cap-6 fit fails (the glitch) and the leave-one-out round holds the glitch.
#[test]
fn the_cap_trial_is_searched_from_the_data() {
    let dir = collection_root("warm-cap-trial");
    let mut s = glitch_collection(&dir);
    s.refit().unwrap();
    println!("fits (solve, degree cap): {:?}", s.fit_log);
    let capped: Vec<&(surrogate::RefitKind, usize)> = s.fit_log.iter().filter(|f| f.1 == RECOVERY_DEGREE_CAP).collect();
    assert!(!capped.is_empty(), "the premise: step 1 ran its cap retry");
    for f in &capped {
        assert!(matches!(f.0, surrogate::RefitKind::Spectral { from_start: false, .. }), "step 1's fit is a search from the data: {:?}", f.0);
    }
    assert!(searches(&s).iter().any(|x| x.0), "the quarantine round after it was searched from a start");
    assert_eq!(s.degree_cap, None, "step 1's cap did not stand");
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
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

/// A fresh provider's cold fit of the fixture at `codes` and `cap` (the reopen reference).
fn fresh_cold_fit(tag: &str, codes: &[u8], cap: Option<usize>) -> (FitReport, surrogate::Model) {
    let dir = collection_root(tag);
    let mut t = glitch_collection(&dir);
    t.quarantine = codes.to_vec();
    t.degree_cap = cap;
    t.species_dirty = true;
    t.fit_store().unwrap();
    (t.report().unwrap().clone(), t.model().unwrap().clone())
}

/// The fit in place equals the fresh cold fit, bit for bit (report, coefficients, factor).
fn assert_cold(label: &str, report: &FitReport, model: &surrogate::Model, fresh: &(FitReport, surrogate::Model)) {
    assert_eq!(report, &fresh.0, "{label}: the report is the fresh cold fit's");
    assert_eq!(model.coeffs, fresh.1.coeffs, "{label}: the coefficients");
    assert_eq!(model.sigma_inv_chol, fresh.1.sigma_inv_chol, "{label}: the posterior factor");
}

/// The fixture's recovery with the confirmation of index `k` forced to fail (`confirm_fail`):
/// the provider after the refit and what the seam saw. Without a forced failure the recovery makes
/// three confirmations: the leave-one-out rounds' stop test on the round that held the glitch (a
/// warm trial: refitted cold), the rankings' choice and the keep (both on that cold fit).
fn recovery_failing(k: Option<usize>, tag: &str) -> (SurrogateKs, Option<ConfirmFailed>) {
    let dir = collection_root(tag);
    let mut s = glitch_collection(&dir);
    s.confirm_fail = k;
    s.refit().unwrap();
    println!(
        "{tag} (confirmation {k:?} fails): {} confirmations; trusted {}, held {:?}, cap {:?}, note {:?}; searches {:?}",
        s.confirm_calls,
        s.trusted(),
        (0..s.samples()).filter(|&i| s.is_quarantined(i)).collect::<Vec<_>>(),
        s.degree_cap,
        s.recovery_note(),
        searches(&s)
    );
    assert!(!s.trial_fit);
    let seen = s.confirm_failed.take();
    assert_eq!(seen.is_some(), k.is_some(), "the seam fired");
    (s, seen)
}

/// What a failed confirmation leaves (review I2): the codes, the degree cap, the note and
/// `species_dirty` exactly as on entry to it, `trial_fit` false, and in place the cold fit of
/// that state — a fresh provider's fit of the same codes and cap, bit for bit.
fn assert_clean_failure(label: &str, seen: &ConfirmFailed) {
    assert_eq!(seen.after, seen.before, "{label}: codes, cap, note and species_dirty untouched");
    assert!(!seen.trial_fit, "{label}: no trial in place");
    let fresh = fresh_cold_fit(&format!("{label}-fresh"), &seen.after.0, seen.after.1);
    assert_cold(label, seen.report.as_ref().unwrap(), seen.model.as_ref().unwrap(), &fresh);
}

/// Review I1 and I2: the rounds' stop test is judged on the cold fit. The leave-one-out round that
/// holds the glitch passes warm; its confirmation (index 0, a refit of the trial) is forced to fail
/// and that cold fit to fail its checks while it is in place — a trial that passes warm and fails
/// cold. The state right after it is clean (`assert_clean_failure`), and the rounds go on from
/// that cold fit as main's would — measured: its leave-one-out offender #73 is held next (a warm
/// trial of 2 rounds, confirmed in 4), and that result stands; whatever stands is a cold fit of its
/// own state.
#[test]
fn a_trial_that_fails_cold_at_the_rounds_stop_goes_on_from_the_cold_fit() {
    let (plain, _) = recovery_failing(None, "warm-fail-none");
    assert_eq!(plain.confirm_calls, 3, "the premise: stop test, rankings' choice, keep");
    let (s, seen) = recovery_failing(Some(0), "warm-fail-stop");
    let seen = seen.unwrap();
    assert!(seen.refitted, "the stop test's confirmation refitted the warm trial");
    assert_eq!(seen.before.0[GLITCH], Q_HELD, "the round that held the glitch");
    assert_clean_failure("stop", &seen);
    assert!(s.confirm_calls > 1, "the recovery went on after the failed confirmation");
    let codes = codes(&s);
    let fresh = fresh_cold_fit("warm-fail-stop-end", &codes, s.degree_cap);
    assert_cold("standing", s.report().unwrap(), s.model().unwrap(), &fresh);
}

/// Review I2, the keep path (`recover_trust`'s "passes the residuals" keep): its confirmation (the
/// last of the three, on the already cold fit) forced to fail leaves a clean state, and the
/// recovery ends exactly as main's does when the keep's check fails — the entry state restored:
/// no sample held, no cap, the failed recovery's note (settled: the entry fit is trusted with the
/// glitch in its misfit set), the entry fit (a cold fit of the entry codes).
#[test]
fn a_keep_that_fails_cold_restores_the_entry_state() {
    let (plain, _) = recovery_failing(None, "warm-keep-none");
    let keep = plain.confirm_calls - 1;
    let (s, seen) = recovery_failing(Some(keep), "warm-keep-fail");
    let seen = seen.unwrap();
    assert!(!seen.refitted, "the keep judges the cold fit the rounds confirmed");
    assert_clean_failure("keep", &seen);
    let codes = codes(&s);
    assert!(codes.iter().all(|&c| c == Q_NONE), "the holds given back: {codes:?}");
    assert_eq!(s.degree_cap, None);
    // the failed recovery's note, settled by the refit (`settle_recovery_note`): the entry fit is
    // trusted with the glitch as its one unreproducible sample (the misfit set M within its limit)
    assert_eq!(s.recovery_note(), Some("recovery did not restore the training residuals; no samples held; 1 samples the model cannot reproduce stay in the fit"));
    let fresh = fresh_cold_fit("warm-keep-entry", &codes, None);
    assert_cold("entry", s.report().unwrap(), s.model().unwrap(), &fresh);
}
