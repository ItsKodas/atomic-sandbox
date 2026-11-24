//! The remembered fit choice (2026-11-25; `fit_choice` module doc): a cap the trust recovery's
//! step 1 kept is remembered with the collection for the store state it was kept for, and the next
//! launch of that state fits at the cap first.
//!
//! The fixture (`write_collection`): a shared collection (body-4 basis, the spectral search) of
//! 300 H₂ and H₃ answers on [`SpeciesMorse`]'s pair surface plus a three-body term
//! A·Σ w(r_ij) w(r_ik) cos θ_jik with a narrow radial well (w a Gaussian of width 0.5 Bohr at 1.6
//! Bohr, A = 0.02 Ha; forces by central differences, 1e-5 Bohr). Measured: the default basis
//! (degree cap 4, n₁ + n₂ + l ≤ 4) cannot follow the well and fails the training residuals; step
//! 1's cap 6 fits it and is trusted (Q′ 74 of 300 samples) — the launch case of the user's
//! collection, where the default fit fails and step 1 stands. With `glitch`, one more H₂ whose
//! energy is 0.05 Ha off: step 1 fails too, and the quarantine rounds on the default basis hold
//! the glitch with six samples that basis misses (measured: 7 held, trusted).
use super::*;
use surrogate::collection::{fit_choice_to_text, FIT_CHOICE_FILE};
use surrogate::FitChoice;

const SAMPLES: usize = 300;
const AMP: f64 = 0.02;
const WIDTH: f64 = 0.5;

/// The fixture's three-body energy (Hartree): Σ over centres i and pairs j < k of its neighbours.
fn three_body(pos: &[[f64; 3]]) -> f64 {
    let w = |r: f64| (-((r - 1.6) / WIDTH).powi(2)).exp();
    let n = pos.len();
    let mut e = 0.0;
    for i in 0..n {
        for j in (0..n).filter(|&j| j != i) {
            for k in (j + 1..n).filter(|&k| k != i) {
                let a: [f64; 3] = std::array::from_fn(|c| pos[j][c] - pos[i][c]);
                let b: [f64; 3] = std::array::from_fn(|c| pos[k][c] - pos[i][c]);
                let (ra, rb) = ((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt(), (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt());
                e += AMP * w(ra) * w(rb) * (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (ra * rb);
            }
        }
    }
    e
}

/// The fixture's label: [`SpeciesMorse`] plus [`three_body`] (its forces by central differences).
fn label(pos: &[[f64; 3]], z: &[u8]) -> (f64, Vec<[f64; 3]>) {
    let (e2, mut f) = SpeciesMorse::energy_and_forces(pos, z);
    let h = 1e-5;
    for a in 0..pos.len() {
        for c in 0..3 {
            let mut p = pos.to_vec();
            p[a][c] += h;
            let ep = three_body(&p);
            p[a][c] -= 2.0 * h;
            f[a][c] -= (ep - three_body(&p)) / (2.0 * h);
        }
    }
    (e2 + three_body(pos), f)
}

/// The module doc's fixture written to a collection at `dir` by a provider that keeps the answers
/// and fits nothing, then lets the collection go. Deterministic.
fn write_collection(dir: &CollectionRoot, glitch: bool) {
    let mut s = d4_student(dir);
    let mut rng = StdRng::seed_from_u64(20261125);
    let kinds: [&[u8]; 2] = [&[1, 1], &[1, 1, 1]];
    for i in 0..SAMPLES {
        let z = kinds[i % 2];
        keep(&mut s, &d4_cluster(z, &mut rng), z, 0.0);
    }
    if glitch {
        let u = unit_vector(&mut rng);
        keep(&mut s, &[[0.0; 3], [1.7 * u[0], 1.7 * u[1], 1.7 * u[2]]], &[1, 1], 0.05);
    }
    assert!(s.report().is_none(), "nothing fitted");
}

/// Keeps one labelled answer (its energy off by `de`).
fn keep(s: &mut SurrogateKs, pos: &[[f64; 3]], z: &[u8], de: f64) {
    let prov = s.provenance_for(&h2(1.4));
    let (e, f) = label(pos, z);
    let mut notes = Vec::new();
    assert!(s.keep_sample(pos, z, e + de, &f, Some(0.0), Some(1), &prov, SampleTag::Scan, &mut notes), "{notes:?}");
}

/// A launch: a fresh provider attaching the collection at `dir`, not fitted yet.
fn launch(dir: &CollectionRoot) -> SurrogateKs {
    let (mut s, _) = pair_student();
    s.attach_refit_off = true;
    s.attach_collection_at(dir, &h2(1.4)).unwrap();
    assert!(s.report().is_none());
    s
}

/// A launch as the app makes it: the attach leaves its first fit to a background refit.
fn launch_background(dir: &CollectionRoot) -> SurrogateKs {
    let (mut s, _) = pair_student();
    s.fits_in_background = true;
    s.attach_collection_at(dir, &h2(1.4)).unwrap();
    assert!(s.report().is_none() && s.first_fit_ready());
    s
}

/// The collection's `fit_choice.txt` (its one folder under `dir`), `None` when absent.
fn choice_file(dir: &CollectionRoot) -> Option<String> {
    let folder = std::fs::read_dir(&**dir).unwrap().map(|e| e.unwrap().path()).find(|p| p.is_dir()).unwrap();
    std::fs::read_to_string(folder.join(FIT_CHOICE_FILE)).ok()
}

/// The degree cap of every fit a provider made, in order.
fn caps(s: &SurrogateKs) -> Vec<usize> {
    s.fit_log.iter().map(|f| f.1).collect()
}

/// What a refit leaves standing: the fit (report, coefficients, posterior factor, basis), the
/// codes, the cap and D₄, and every verdict (the trust checks, the refusal, the training residuals,
/// the misfit set, the leave-one-out ratios).
#[derive(Debug, PartialEq)]
struct Outcome {
    report: FitReport,
    basis: Basis,
    coeffs: Vec<u64>,
    factor: String,
    codes: Vec<u8>,
    cap: Option<usize>,
    body4_degree: Option<usize>,
    trusted: bool,
    checks: Vec<sim::activity::TrustCheck>,
    refusal: Option<(sim::activity::TrustKey, String)>,
    residuals: Option<TrainingResiduals>,
    misfit: Option<Misfit>,
    loo: Vec<(usize, u64)>,
}

fn outcome(s: &mut SurrogateKs) -> Outcome {
    let m = s.model().unwrap().clone();
    Outcome {
        report: s.report().unwrap().clone(),
        basis: m.basis.clone(),
        coeffs: m.coeffs.iter().flatten().map(|x| x.to_bits()).collect(),
        factor: format!("{:?}", m.sigma_inv_chol),
        codes: (0..s.samples()).map(|i| s.quarantine.get(i).copied().unwrap_or(Q_NONE)).collect(),
        cap: s.degree_cap,
        body4_degree: s.body4_degree(),
        trusted: s.trusted(),
        checks: s.trust_checks(),
        refusal: s.untrusted_check(),
        residuals: s.training_residuals(),
        misfit: s.misfit().cloned(),
        loo: s.loo_ratios().unwrap_or(&[]).iter().map(|&(i, r)| (i, r.to_bits())).collect(),
    }
}

/// The full recovery of a launch, and the next launch of the same store state: the first runs the
/// default-basis fit and step 1's cap retry ([4, 6]), which stands and is written to
/// `fit_choice.txt`; the second reads it, makes one fit at the remembered cap ([6]) and stands where
/// the first stood — the fit bit for bit (report, coefficients, posterior factor) and every verdict.
#[test]
fn a_launch_fits_the_remembered_cap_and_stands_where_the_full_recovery_does() {
    let dir = collection_root("fit-choice-reuse");
    write_collection(&dir, false);
    let (full, choice) = {
        let mut s = launch(&dir);
        assert_eq!(s.fit_choice(), None, "nothing remembered yet");
        let key = s.fit_choice_entry().expect("a launch meets the conditions");
        s.refit().unwrap();
        println!("full recovery: fits (cap) {:?}, note {:?}, trusted {} {:?}", caps(&s), s.recovery_note(), s.trusted(), s.untrusted_reason());
        assert_eq!(caps(&s), vec![4, RECOVERY_DEGREE_CAP], "the premise: the default fit failed, step 1's retry stood");
        assert_eq!(s.recovery_note(), Some(cap_retry_note().as_str()));
        assert!(s.trusted(), "{:?}", s.untrusted_reason());
        let choice = FitChoice { key, degree_cap: RECOVERY_DEGREE_CAP };
        assert_eq!(s.fit_choice(), Some(choice), "remembered");
        assert_eq!(choice_file(&dir), Some(fit_choice_to_text(&choice)), "and written with the collection");
        (outcome(&mut s), choice)
    };
    let mut s = launch(&dir);
    assert_eq!(s.fit_choice(), Some(choice), "read back at attach");
    assert_eq!(s.fit_choice_entry(), Some(choice.key), "the same store state, the same key");
    s.refit().unwrap();
    println!("remembered: fits (cap) {:?}, note {:?}", caps(&s), s.recovery_note());
    assert_eq!(caps(&s), vec![RECOVERY_DEGREE_CAP], "one fit, at the remembered cap");
    let note = s.recovery_note().unwrap();
    assert!(note.starts_with(&cap_retry_note()) && note.contains("remembered"), "{note}");
    assert_eq!(outcome(&mut s), full, "the fit and the verdicts the full recovery reached");
    assert_eq!(s.fit_choice(), Some(choice));
    assert_eq!(choice_file(&dir), Some(fit_choice_to_text(&choice)), "the file as it was");
}

/// The same through a background refit (the app's launch fits on a copy of the provider): the
/// copy's full recovery is remembered at the install and written by the live provider; the next
/// launch's copy fits once at the cap and installs what the first stood at.
#[test]
fn a_background_refit_remembers_and_reuses_the_cap() {
    let dir = collection_root("fit-choice-background");
    write_collection(&dir, false);
    let full = {
        let mut s = launch_background(&dir);
        assert!(s.begin_background_refit());
        assert!(matches!(s.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
        assert_eq!(s.recovery_note(), Some(cap_retry_note().as_str()), "the copy's step 1 stood");
        assert!(s.fit_choice().is_some() && choice_file(&dir).is_some(), "remembered and written at the install");
        outcome(&mut s)
    };
    let mut s = launch_background(&dir);
    assert!(s.begin_background_refit());
    assert!(matches!(s.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
    assert!(s.recovery_note().is_some_and(|n| n.contains("remembered")), "{:?}", s.recovery_note());
    assert_eq!(outcome(&mut s), full);
}

/// A remembered choice whose fit does not pass step 1's check falls back to the full recovery
/// from the default basis, and the memory is dropped. The glitch fixture (step 1 fails; the
/// quarantine holds the glitch) with a memory planted for its own key: the refit tries the cap
/// first, restores the entry state, and then runs exactly the full path — its fits after the
/// first are the full recovery's, and it stands where that stands, bit for bit. (The full
/// recovery runs on a twin collection: its holds are written to its collection, and a launch
/// after it starts from another store state.)
#[test]
fn a_remembered_cap_that_fails_its_check_falls_back_to_the_full_recovery() {
    let (dir, twin) = (collection_root("fit-choice-fallback"), collection_root("fit-choice-fallback-twin"));
    write_collection(&dir, true);
    write_collection(&twin, true);
    let (full, full_caps, full_note) = {
        let mut s = launch(&twin);
        s.refit().unwrap();
        println!("full recovery: fits (cap) {:?}, held {}, note {:?}", caps(&s), s.quarantined(), s.recovery_note());
        assert_eq!(caps(&s)[..2], [4, RECOVERY_DEGREE_CAP], "the premise: the default fit failed and step 1 ran");
        assert_eq!(s.degree_cap, None, "the premise: step 1 did not stand");
        assert!(s.trusted() && s.is_quarantined(SAMPLES), "the premise: the quarantine held the glitch: {:?}", s.untrusted_reason());
        assert_eq!((s.fit_choice(), choice_file(&twin)), (None, None), "nothing remembered");
        (outcome(&mut s), caps(&s), s.recovery_note().map(str::to_string))
    };
    let mut s = launch(&dir);
    let planted = FitChoice { key: s.fit_choice_entry().unwrap(), degree_cap: RECOVERY_DEGREE_CAP };
    {
        let sh = s.shared.as_mut().unwrap();
        sh.coll.save_fit_choice(&planted).unwrap();
        sh.fit_choice = Some(planted);
    }
    s.refit().unwrap();
    println!("planted: fits (cap) {:?}, note {:?}", caps(&s), s.recovery_note());
    let mut want = vec![RECOVERY_DEGREE_CAP];
    want.extend(&full_caps);
    assert_eq!(caps(&s), want, "the remembered cap first, then the full path");
    assert_eq!(outcome(&mut s), full, "where the full recovery stands");
    assert_eq!(s.recovery_note().map(str::to_string), full_note, "with its note");
    assert_eq!((s.fit_choice(), choice_file(&dir)), (None, None), "the memory dropped, the file removed");
}

/// The memory holds only for the store state it was kept for (module doc, "What is remembered"):
/// a new sample, another policy, a changed reference, a sample's code, another reach and another
/// W each give another key — and such a launch runs the full recovery from the default basis
/// (its first fit at cap 4) and remembers the new state's choice; the same state gives the same
/// key at every launch.
#[test]
fn the_remembered_cap_holds_only_for_its_own_store_state() {
    let dir = collection_root("fit-choice-invalidate");
    write_collection(&dir, false);
    let choice = {
        let mut s = launch(&dir);
        s.refit().unwrap();
        s.fit_choice().expect("the premise: step 1 stood and was remembered")
    };
    // the key, unchanged and changed, on one launch
    {
        let mut s = launch(&dir);
        let k0 = s.fit_choice_entry().unwrap();
        assert_eq!(k0, choice.key, "the same state at another launch");
        let differs = |what: &str, s: &mut SurrogateKs, change: &dyn Fn(&mut SurrogateKs), undo: &dyn Fn(&mut SurrogateKs)| {
            change(s);
            assert_ne!(s.fit_choice_key(), Some(k0), "{what} is another key");
            undo(s);
            assert_eq!(s.fit_choice_key(), Some(k0), "{what} undone is the same key");
        };
        let e1 = s.refs[&1];
        differs("a reference", &mut s, &|s| *s.refs.get_mut(&1).unwrap() = f64::from_bits(e1.to_bits() + 1), &|s| *s.refs.get_mut(&1).unwrap() = e1);
        let e0 = s.store.as_ref().unwrap().samples[0].energy;
        differs("a sample's energy", &mut s, &|s| s.store.as_mut().unwrap().samples[0].energy = e0 + 1e-12, &|s| s.store.as_mut().unwrap().samples[0].energy = e0);
        differs("a sample's code", &mut s, &|s| {
            s.quarantine.resize(s.samples(), Q_NONE);
            s.quarantine[3] = Q_HELD;
        }, &|s| s.quarantine[3] = Q_NONE);
        let r = s.collection_r_cut;
        differs("the reach", &mut s, &|s| s.collection_r_cut = r + 0.5, &|s| s.collection_r_cut = r);
        let mix = |w: f64| surrogate::channels::LearntMix {
            k: 1,
            k_requested: 1,
            species: vec![1],
            w: vec![w],
            sigma_w: Vec::new(),
            singular_values: vec![1.0],
            truncation_residual: vec![0.0],
            nonlinearity: 0.0,
            weights: vec![1.0],
            rank: 1,
            low_order: None,
            represented: vec![true],
            n_w_met: vec![true],
            many_body: vec![0],
            history_from: 0,
        };
        s.shared.as_mut().unwrap().mix = Some(mix(1.0));
        let kw = s.fit_choice_key().unwrap();
        assert_ne!(kw, k0, "a W");
        s.shared.as_mut().unwrap().pending_mix = Some((mix(-1.0), "relearnt".into()));
        assert_ne!(s.fit_choice_key(), Some(kw), "a W relearnt (staged) is another key");
        s.shared.as_mut().unwrap().pending_mix = None;
        s.shared.as_mut().unwrap().mix = None;
        let p = s.policy;
        differs("the policy", &mut s, &|s| s.policy.energy_std_per_atom = 0.9e-3, &|s| s.policy = p);
    }
    // a new sample: the full recovery from the default basis, a new choice remembered
    let mut s = launch(&dir);
    let mut rng = StdRng::seed_from_u64(7);
    keep(&mut s, &d4_cluster(&[1, 1, 1], &mut rng), &[1, 1, 1], 0.0);
    let key = s.fit_choice_entry().unwrap();
    assert_ne!(key, choice.key);
    s.refit().unwrap();
    assert_eq!(caps(&s), vec![4, RECOVERY_DEGREE_CAP], "the full recovery: the default basis first");
    assert_eq!(s.fit_choice(), Some(FitChoice { key, degree_cap: RECOVERY_DEGREE_CAP }), "the new state's choice replaces the old");
    // another policy at launch: the full recovery again
    let mut s = launch(&dir);
    let mut p = s.policy;
    p.energy_std_per_atom = 0.95e-3;
    s.set_policy(p);
    assert_ne!(s.fit_choice_entry(), Some(key));
    s.refit().unwrap();
    assert_eq!(caps(&s)[0], 4, "another policy: the default basis first");
}

/// In a session the cap a recovery set stands for the next refit as before (step 1b re-judges
/// it): the remembered choice is neither read nor written while a cap stands, and a refit with
/// the recovery switched off (tests) neither.
#[test]
fn a_standing_cap_and_a_refit_without_recovery_leave_the_memory_alone() {
    let dir = collection_root("fit-choice-session");
    write_collection(&dir, false);
    let mut s = launch(&dir);
    s.refit().unwrap();
    assert_eq!(s.degree_cap, Some(RECOVERY_DEGREE_CAP));
    assert_eq!(s.fit_choice_entry(), None, "a cap stands");
    let before = choice_file(&dir);
    let mut rng = StdRng::seed_from_u64(8);
    keep(&mut s, &d4_cluster(&[1, 1, 1], &mut rng), &[1, 1, 1], 0.0);
    s.refit().unwrap();
    assert_eq!(choice_file(&dir), before, "the standing cap's refit wrote nothing");
    drop(s);
    let mut s = launch(&dir);
    s.skip_recovery = true;
    assert_eq!(s.fit_choice_entry(), None, "no recovery, no memory");
    s.refit().unwrap();
    assert_eq!(caps(&s), vec![4], "the default fit alone");
    assert_eq!(choice_file(&dir), before);
}
