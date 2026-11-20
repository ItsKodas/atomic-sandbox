//! 2026-11-17 (estimated H₂ at 1.4 fs/s instead of ~14): the status every worker round builds
//! asked `teach_offer` → `richer_basis_floor` → `trust::key_column_counts` over every fitted sample
//! in the degree-6 basis, uncached — 60–77 ms per call on the user's 1590-sample collection,
//! twice per round. The words are now cached on a fingerprint of what they read, and the worker's
//! one-bool read (`audits_background`) has an accessor that builds no status. No behaviour
//! change: every cached answer equals a fresh count.

use super::*;
use std::sync::atomic::Ordering::SeqCst;

fn counts(s: &SurrogateKs) -> usize {
    s.richer_floor_counts.load(SeqCst)
}

/// What a fresh provider in the same state would say: the cache emptied, then asked.
fn fresh_floor(s: &SurrogateKs) -> Option<String> {
    let kept = s.richer_floor_cache.lock().unwrap().take();
    let fresh = s.richer_basis_floor();
    *s.richer_floor_cache.lock().unwrap() = kept;
    fresh
}

fn fresh_offer(s: &SurrogateKs) -> Option<sim::TeachOffer> {
    let kept = s.richer_floor_cache.lock().unwrap().take();
    let fresh = s.teach_offer();
    *s.richer_floor_cache.lock().unwrap() = kept;
    fresh
}

/// The cached words and offer equal a fresh provider's; `recount`: the cached read had to count
/// again (what it reads changed), else it answered from the cache.
fn check(s: &SurrogateKs, recount: Option<bool>, what: &str) {
    let n0 = counts(s);
    let floor = s.richer_basis_floor();
    let counted = counts(s) > n0;
    if let Some(r) = recount {
        assert_eq!(counted, r, "{what}: recounted {counted}, expected {r}");
    }
    assert_eq!(floor, fresh_floor(s), "{what}: the cached words are a fresh provider's");
    let n1 = counts(s);
    let offer = s.teach_offer();
    assert_eq!(counts(s), n1, "{what}: the offer reads the cache the floor just filled");
    assert_eq!(offer, fresh_offer(s), "{what}: the cached offer is a fresh provider's");
    println!("{what}: recounted {counted}, floor {floor:?}");
}

/// (a) Two status reads in a row with an offer present count the richer basis's keys once, and
/// the second read's offer is the first's.
#[test]
fn two_status_reads_count_the_richer_keys_once() {
    let (s, _h) = gap_blocked(2.0, 3.6);
    assert!(s.richer_basis().is_some(), "the premise: a body-3 basis below the recovery's degree, no cap");
    let sc = in_big_cell(h2(1.5));
    let forces = vec![[0.0; 3]; 2];
    s.richer_floor_cache.lock().unwrap().take();
    let n0 = counts(&s);
    let a = Trainable::status(&s, &sc, &forces).teach_offer;
    let b = Trainable::status(&s, &sc, &forces).teach_offer;
    assert!(a.as_ref().is_some_and(|o| !o.blockers.is_empty()), "the premise: an offer with blockers: {a:?}");
    assert_eq!(counts(&s) - n0, 1, "one key count for two status reads");
    assert_eq!(a, b);
    assert_eq!(a, fresh_offer(&s));
}

/// (b) Every change of what the words read recounts, and after each event (an append, a refit,
/// a forget and a restore, a policy change, a degree cap, a purge, an installed background
/// refit) the cached answer is a fresh provider's. 30 dimers: the richer basis's pair key is
/// short ("30 of 51"), so the words exist and move with the fitted count.
#[test]
fn the_richer_floor_cache_follows_every_change() {
    let (mut s, _h) = trained(1.2, 2.2, 30, FallbackPolicy::default());
    assert!(s.richer_basis_floor().is_some_and(|w| w.contains("has 30 of")), "the premise: {:?}", s.richer_basis_floor());
    check(&s, None, "fitted");
    check(&s, Some(false), "asked again");

    // an append without a refit: not fitted yet, the words do not move
    let mut rng = StdRng::seed_from_u64(11);
    let sc = dimer(1.45, unit_vector(&mut rng), [0.0; 3], 1, 1);
    s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    check(&s, Some(false), "appended");
    s.refit().unwrap();
    check(&s, Some(true), "refitted over the append");

    // forget and restore one fitted sample (the quarantine codes move)
    let id = s.sample_id(5).unwrap();
    s.forget_sample(id).unwrap();
    check(&s, Some(true), "forgotten");
    s.restore_sample(s.sample_id(5).unwrap()).unwrap();
    check(&s, Some(true), "restored");

    // a policy change: nothing the words read
    s.set_policy(FallbackPolicy { audit_every: 7, ..*s.policy() });
    check(&s, None, "policy changed");

    // a degree cap: no richer basis, no words; lifted, the words are back
    s.degree_cap = Some(RECOVERY_DEGREE_CAP);
    assert!(s.richer_basis().is_none());
    check(&s, Some(false), "capped");
    s.degree_cap = None;
    check(&s, None, "cap lifted");

    // a purge of a forgotten sample: the indices shift
    s.forget_sample(s.sample_id(7).unwrap()).unwrap();
    s.refit().unwrap();
    check(&s, Some(true), "forgotten and refitted");
    s.purge_forgotten().unwrap();
    check(&s, None, "purged");

    // an installed background refit over new samples
    for r in [1.25, 1.65, 1.95] {
        let sc = dimer(r, unit_vector(&mut rng), [0.0; 3], 1, 1);
        s.add_sample(morse_sample(&MorseDimer::h2(), &sc), &sc).unwrap();
    }
    check(&s, Some(false), "appended three");
    assert!(s.begin_background_refit());
    assert!(matches!(s.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
    check(&s, Some(true), "background refit installed");
    assert!(s.richer_basis_floor().is_some_and(|w| w.contains("has 33 of")), "30 + 1 appended − 1 forgotten and purged + 3 appended: {:?}", s.richer_basis_floor());

    // forget everything
    Trainable::forget(&mut s);
    check(&s, None, "forgot all");
    assert_eq!(s.richer_basis_floor(), None);
}

/// (c) The worker's one-bool read: the accessor is the status's field, with and without a
/// background auditor, and builds no status (it asks no key count).
#[test]
fn audits_background_is_the_status_field_without_the_status() {
    let sc = in_big_cell(h2(1.5));
    let forces = vec![[0.0; 3]; 2];
    let (s, _h) = gap_blocked(2.0, 3.6);
    s.richer_floor_cache.lock().unwrap().take();
    let n0 = counts(&s);
    assert!(!Trainable::audits_background(&s));
    assert_eq!(counts(&s), n0, "the accessor builds no status");
    assert_eq!(Trainable::audits_background(&s), Trainable::status(&s, &sc, &forces).audits_background);
    let a = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_auditor(Box::new(MorseDimer::h2()));
    assert!(Trainable::audits_background(&a));
    assert_eq!(Trainable::audits_background(&a), Trainable::status(&a, &sc, &forces).audits_background);
}

/// The measurement on a copy of the user's collection (`crates/validate/references.md`, "Status
/// cache"): one `status()` with the richer-basis words counted afresh (the cache emptied before
/// each call: the cost before 2026-11-17) and from the cache, and the offer's pieces.
/// `COLLECTION_DIR=<copy of a collection's surface dir> PSEUDO_DIR=<the pseudo disk cache> cargo test -p providers --release --lib
/// status_cost_on_a_collection -- --ignored --nocapture`
#[test]
#[ignore = "a measurement on a collection copy (COLLECTION_DIR)"]
fn status_cost_on_a_collection() {
    // the pseudopotential gate's disk cache (the app's `cache\pseudo`, or a copy): without it the
    // first `All`-policy provenance generates the H pseudopotential on the CPU (tens of minutes)
    crate::pseudo::set_disk_cache(std::env::var_os("PSEUDO_DIR").map(std::path::PathBuf::from));
    let dir = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    let (mut c, _) = surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap();
    c.read_only = true;
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let t0 = Instant::now();
    s.attach(c).unwrap();
    println!("attach {:.1} s, report {}", t0.elapsed().as_secs_f64(), s.report().is_some());
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
        println!("refit {:.1} s", t0.elapsed().as_secs_f64());
    }
    let sc = Scene {
        nuclei: vec![Nucleus::new(1, [-0.7, 0.0, 0.0]), Nucleus::new(1, [0.7, 0.0, 0.0])],
        cell: Cell { n: 129, h: 0.15 },
        charge: 0,
        multiplicity: None,
        sigma: Some(0.15),
        external_field: [0.0; 3],
        ionic: sim::IonicPolicy::All,
        train_mask: None,
    };
    let forces = vec![[0.0; 3]; 2];
    // the worker's state: estimated steps (draft on, audits off — the Model-only path) first; the
    // worker's status is at the geometry the last step evaluated (`at`), whose strict verdict the
    // step recorded — at another geometry (`sc`) `readiness` judges afresh
    Trainable::set_draft(&mut s, true);
    Trainable::set_audit_every(&mut s, 0);
    let step = |k: usize| Scene { nuclei: vec![Nucleus::new(1, [-0.7 - 0.005 * k as f64, 0.0, 0.0]), Nucleus::new(1, [0.7, 0.0, 0.0])], ..sc.clone() };
    for k in 0..20 {
        let _ = s.evaluate(&step(k), &mut never());
    }
    let at = step(19);
    println!("20 steps at {:.1} s", t0.elapsed().as_secs_f64());
    println!("samples {}, fitted prefix {}, degree_cap {:?}, richer basis {}", s.samples(), s.accumulated, s.degree_cap, s.richer_basis().is_some());
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let reps = 10;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    for (name, scene) in [("at the last step's geometry (the worker's case)", &at), ("at another geometry", &sc)] {
        let (mut cold, mut warm) = (Vec::new(), Vec::new());
        let mut offers = Vec::new();
        for _ in 0..reps {
            s.richer_floor_cache.lock().unwrap().take();
            let t = Instant::now();
            let st = Trainable::status(&s, scene, &forces);
            cold.push(ms(t));
            offers.push(st.teach_offer);
            let t = Instant::now();
            let st = Trainable::status(&s, scene, &forces);
            warm.push(ms(t));
            offers.push(st.teach_offer);
        }
        assert!(offers.windows(2).all(|w| w[0] == w[1]), "every status's offer is the same");
        println!("{name}:");
        println!("  status(), words counted afresh (before): mean {:.2} ms, max {:.2} ms", mean(&cold), max(&cold));
        println!("  status(), words from the cache (after):  mean {:.2} ms, max {:.2} ms", mean(&warm), max(&warm));
        println!("  offer present {}, notes {}", offers[0].is_some(), offers[0].as_ref().map_or(0, |o| o.notes.len()));
    }
    let time = |name: &str, f: &dyn Fn()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        println!("  {name:<34} {:.3} ms", ms(t) / reps as f64);
    };
    time("readiness (last step's geometry)", &|| {
        std::hint::black_box(s.readiness(&at));
    });
    time("readiness (another geometry)", &|| {
        std::hint::black_box(s.readiness(&sc));
    });
    time("trust_checks", &|| {
        std::hint::black_box(s.trust_checks());
    });
    time("stuck_info", &|| {
        std::hint::black_box(s.stuck_info());
    });
    time("trust_floor", &|| {
        std::hint::black_box(s.trust_floor());
    });
    time("teach_offer (cached words)", &|| {
        std::hint::black_box(s.teach_offer());
    });
    time("  teach_diagnosis", &|| {
        std::hint::black_box(s.teach_diagnosis());
    });
    time("  richer_basis_floor (cache hit)", &|| {
        std::hint::black_box(s.richer_basis_floor());
    });
    time("audits_background (accessor)", &|| {
        std::hint::black_box(Trainable::audits_background(&s));
    });
}

