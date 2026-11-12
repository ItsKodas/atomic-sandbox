//! Measurement (2026-11-12, the user's "what happens at 50 000 samples?"): the wall of one
//! refit-plus-trust cycle on a shared collection, phase by phase, and how each phase scales with
//! the sample count P, the atoms per sample and Q′. Reported, never asserted; nothing here changes
//! what a refit does — each phase is the provider's own function, called again on its own.
//!
//! Reads a **copy** of a collection folder (`COLLECTION_DIR`, never the live one: the subsets and
//! replicas below are built in memory with the copy's writes off, and only the whole-collection
//! run writes — its history and README — to the copy).
//!
//! `COLLECTION_DIR=<copy> REFIT_COST=<runs> cargo test -p providers --release refit_cost -- --ignored --nocapture --test-threads 1`
//! with `<runs>` a comma list of: `full` (the collection as is: the attach's own refit, then the
//! phases, a +25-sample warm cycle, the collection writes), `sub` (the first ¼ and ½ of the
//! samples), `species` (H only, H + O, all), `rep2`, `rep4`, … (the collection with k−1
//! jittered replicas of every sample: P × k).
use super::*;
use super::super::trust::key_column_counts;
use std::time::Instant;
use surrogate::{Collection, CollectionSample};

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn open_copy() -> Collection {
    let dir = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR names a collection folder copy"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    let t = Instant::now();
    let (c, _) = surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap();
    println!("open: {} samples, {:.3} s", c.samples.len(), secs(t));
    c
}

/// A replica of `s` jittered by up to `delta` Bohr per coordinate and relabelled to first order
/// (E' = E − Σ F·δ, F unchanged; the error O(k δ²) in E and O(k δ) in F, ≈ 4e-5 Ha/Bohr at
/// δ = 1e-4 and k ≈ 0.4 Ha/Bohr²: below σ_F, so the trust machinery sees a collection like the
/// real one, P × k samples of it).
fn replica(s: &CollectionSample, rng: &mut StdRng, delta: f64) -> CollectionSample {
    let mut r = s.clone();
    let mut de = 0.0;
    for (p, f) in r.sample.pos.iter_mut().zip(&s.sample.forces) {
        for a in 0..3 {
            let d = rng.gen_range(-delta..delta);
            p[a] += d;
            de -= f[a] * d;
        }
    }
    r.sample.energy += de;
    r
}

/// A provider over the collection `c` (taken as given: its samples and codes), with the attach's
/// own refit — the cold cycle — timed.
fn provider_on(c: Collection) -> (SurrogateKs, f64) {
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let t = Instant::now();
    s.attach(c).unwrap();
    if s.report().is_none() {
        // the attach did not refit (no model wanted yet): the cold cycle is this refit
        SurrogateKs::refit(&mut s).unwrap();
    }
    let cold = secs(t);
    println!("attach: {} fit_store calls, recovery {:?}", s.fits, s.recovery_note());
    (s, cold)
}

/// The phases of a fit-plus-trust cycle, each called again on its own over the provider's
/// current fit (the same members). Prints one table row per phase.
fn phases(s: &mut SurrogateKs, label: &str) {
    let r = s.report().unwrap().clone();
    let store = s.store.as_ref().unwrap();
    let members: Vec<usize> = (0..s.accumulated.min(store.len())).filter(|&i| !s.excluded(i)).collect();
    let atoms: usize = members.iter().map(|&i| store.samples[i].pos.len()).sum();
    println!("\n## {label}: P {} stored, {} fitted ({} atoms, {:.2} per sample), Q′ {}, species {:?}", store.len(), members.len(), atoms, atoms as f64 / members.len().max(1) as f64, r.q_prime, s.model.as_ref().unwrap().basis.species);
    let mut rows: Vec<(String, f64)> = Vec::new();

    // the model side (surrogate::fit): accumulate all members cold, the spectral search, a +25 warm
    let basis = s.accumulator.as_ref().unwrap().basis.clone();
    let refs = s.reference_vector(&basis).unwrap();
    let prov = store.provenance.clone();
    let opts = surrogate::FitOptions { sigma_f_min: force_noise_prior(&prov.ionic), ..surrogate::FitOptions::default() };
    let hold = 25.min(members.len() / 2);
    let (head, tail) = members.split_at(members.len() - hold);
    let mut acc = surrogate::Accumulator::new(&basis);
    let t = Instant::now();
    for &i in head {
        acc.add(&store.samples[i], &basis, &refs, &opts).unwrap();
    }
    let t_acc = secs(t);
    rows.push((format!("accumulate {} samples (cold rebuild)", head.len()), t_acc));
    let t = Instant::now();
    let (_, _) = acc.refit(&opts, &prov, &refs).unwrap();
    rows.push((format!("search ({:?})", acc.last_refit()), secs(t)));
    let t = Instant::now();
    for &i in tail {
        acc.add(&store.samples[i], &basis, &refs, &opts).unwrap();
    }
    rows.push((format!("accumulate +{hold} (incremental)"), secs(t)));
    let t = Instant::now();
    let (model2, _) = acc.refit(&opts, &prov, &refs).unwrap();
    rows.push((format!("refit after +{hold} ({:?})", acc.last_refit()), secs(t)));
    let t = Instant::now();
    let _ns = surrogate::NullSpace::of(&acc, &model2.scale);
    rows.push(("null space (post-fit, Q′ eigen)".into(), secs(t)));

    // the provider side: each pass of `fit_store` after the solve
    let t = Instant::now();
    let _ = s.compute_training_residuals();
    rows.push(("training residuals (predict every member)".into(), secs(t)));
    let model = s.model.as_ref().unwrap();
    let store = s.store.as_ref().unwrap();
    let t = Instant::now();
    let mut loo_by_n: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
    for &i in &members {
        let t1 = Instant::now();
        let _ = surrogate::loo::leave_one_out(model, &store.samples[i]);
        let e = loo_by_n.entry(store.samples[i].pos.len()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += secs(t1);
    }
    rows.push(("  of which: leave-one-out alone".into(), secs(t)));
    let t = Instant::now();
    for &i in &members {
        let s_ = &store.samples[i];
        let _ = surrogate::energy_and_forces(model, &s_.pos, &s_.z);
    }
    rows.push(("  of which: energy_and_forces (train residual per class)".into(), secs(t)));
    let t = Instant::now();
    let _ = key_column_counts(&model.basis, members.iter().map(|&i| &store.samples[i]));
    rows.push(("  of which: key_column_counts".into(), secs(t)));
    let (keys, aleatoric) = {
        let j = s.class_judge.as_ref().unwrap();
        (j.keys.clone(), j.aleatoric)
    };
    let t = Instant::now();
    let _ = super::super::calibrate::calibrate_with(&s.audit_history, &s.policy, aleatoric, &keys);
    rows.push((format!("  of which: calibration ({} history points)", s.audit_history.len()), secs(t)));
    let t = Instant::now();
    let _ = s.compute_class_loo_full();
    let t_loo = secs(t);
    rows.insert(rows.len() - 4, ("class leave-one-out (rule 4b, whole pass)".into(), t_loo));
    let t = Instant::now();
    s.desc_cache.clear();
    s.desc_basis = None;
    s.ensure_desc_cache();
    rows.push(("descriptor cache, cold fill (warm: new samples only)".into(), secs(t)));
    let t = Instant::now();
    s.rebuild_active_set();
    rows.push(("active set (rule 5e: Gram + eigen + MaxVol per species)".into(), secs(t)));

    // one whole provider cycle at the standing fit (nothing new: the warm path, every pass again)
    let t = Instant::now();
    s.fit_store().unwrap();
    rows.push(("fit_store, nothing new (warm solve + all passes)".into(), secs(t)));
    if std::env::var("REFIT_WHOLE").map_or(true, |v| v != "0") {
        let (t, fits) = (Instant::now(), s.fits);
        s.pending_refit = true;
        SurrogateKs::refit(s).unwrap();
        rows.push((format!("refit() whole: {} fit_store calls; recovery {:?}", s.fits - fits, s.recovery_note()), secs(t)));
    }

    for (k, v) in &rows {
        println!("| {k} | {v:.3} s |");
    }
    let per: Vec<String> = loo_by_n.iter().map(|(n, (c, t))| format!("{n} atoms: {c} samples, {:.2} ms each", 1e3 * t / *c as f64)).collect();
    println!("leave-one-out per sample by atom count: {}", per.join("; "));
    println!("accumulate per sample: {:.2} ms", 1e3 * t_acc / head.len().max(1) as f64);
}

/// The collection's own writes after a refit (`flush_collection`, `sync_collection_status`) and
/// one append, on the whole (writable) copy.
fn collection_writes(s: &mut SurrogateKs) {
    let sh = s.shared.as_mut().unwrap();
    if sh.coll.read_only {
        println!("collection writes: read-only copy, skipped");
        return;
    }
    let t = Instant::now();
    sh.coll.save_history().unwrap();
    let t_hist = secs(t);
    let t = Instant::now();
    sh.coll.write_readme().unwrap();
    let t_readme = secs(t);
    let t = Instant::now();
    s.sync_collection_status();
    let t_sync = secs(t);
    println!("| collection: save_history ({} points) | {t_hist:.3} s |\n| collection: write_readme | {t_readme:.3} s |\n| collection: sync_collection_status | {t_sync:.3} s |", s.audit_history.len());
}

fn subset(mut c: Collection, keep: impl Fn(usize, &CollectionSample) -> bool) -> Collection {
    let idx: Vec<usize> = (0..c.samples.len()).filter(|&i| keep(i, &c.samples[i])).collect();
    c.samples = idx.iter().map(|&i| c.samples[i].clone()).collect();
    c.quarantine = idx.iter().map(|&i| c.quarantine[i]).collect();
    c.read_only = true;
    c
}

#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> REFIT_COST=full,sub,species,rep2 cargo test -p providers --release refit_cost -- --ignored --nocapture --test-threads 1"]
fn refit_cost_on_a_collection() {
    let runs = std::env::var("REFIT_COST").unwrap_or_else(|_| "full".into());
    for run in runs.split(',').map(str::trim) {
        match run {
            "full" => {
                let c = open_copy();
                let n = c.samples.len();
                let (mut s, cold) = provider_on(c);
                println!("\n# full collection: attach + cold refit() {cold:.2} s ({n} samples)");
                phases(&mut s, "full collection");
                collection_writes(&mut s);
            }
            "sub" => {
                for frac in [4usize, 2] {
                    let c = open_copy();
                    let n = c.samples.len() / frac;
                    let (mut s, cold) = provider_on(subset(c, |i, _| i < n));
                    println!("\n# first 1/{frac}: attach + cold refit() {cold:.2} s");
                    phases(&mut s, &format!("first 1/{frac} ({n} samples)"));
                }
            }
            "species" => {
                for (name, allowed) in [("H only", vec![1u8]), ("H + O", vec![1, 8])] {
                    let c = open_copy();
                    let (mut s, cold) = provider_on(subset(c, |_, x| x.sample.z.iter().all(|z| allowed.contains(z))));
                    println!("\n# {name}: attach + cold refit() {cold:.2} s");
                    phases(&mut s, name);
                }
            }
            r if r.starts_with("rep") => {
                let k: usize = r[3..].parse().expect("repK");
                let mut c = open_copy();
                let mut rng = StdRng::seed_from_u64(20261112);
                let base = c.samples.clone();
                let codes = c.quarantine.clone();
                for _ in 1..k {
                    for (x, &q) in base.iter().zip(&codes) {
                        c.samples.push(replica(x, &mut rng, 1e-4));
                        c.quarantine.push(q);
                    }
                }
                c.read_only = true;
                let n = c.samples.len();
                let (mut s, cold) = provider_on(c);
                println!("\n# ×{k} jittered replicas: attach + cold refit() {cold:.2} s ({n} samples)");
                phases(&mut s, &format!("×{k} replicas ({n} samples)"));
            }
            other => panic!("unknown run {other}"),
        }
    }
}

/// Everything a refit leaves that a decision or a readout reads: the fit, the trust rule's inputs,
/// the class verdicts and their judgement, the active set, the quarantine, the recovery, the
/// failed audits and fences, the teach-around rounds; the notes without the MaxVol note (said
/// once per fit that computes the set).
fn refit_state(s: &SurrogateKs) -> String {
    let notes: Vec<&String> = s.user_notes.iter().filter(|n| !n.contains("MaxVol")).collect();
    format!(
        "model {:?}\nreport {:?}\ntraining {:?}\nclass_loo {:?}\nclass_judge {:?}\nclass_worst {:?}\nclass_blockers {:?}\nclass_memory {:?}\nactive_set {:?}\nnull_space {:?}\nquarantine {:?}\ndegree_cap {:?}\nrecovery {:?}\ntrusted {} {:?}\nfailed_audits {:?}\nfences {:?}\naudit {:?} {:?}\nteach_rounds {:?}\nfits {} accumulated {} members {:?}\nepoch {} jump {:?}\nnotes {:?}\ndeferred {:?}",
        s.model,
        s.report,
        s.training_residuals,
        s.class_loo,
        s.class_judge,
        s.class_worst,
        s.class_blockers,
        s.class_memory,
        s.active_set,
        s.null_space.as_ref().map(|n| n.dimension()),
        s.quarantine,
        s.degree_cap,
        s.recovery_note(),
        s.trusted(),
        s.untrusted_reason(),
        s.failed_audits.iter().map(|c| (c.index, c.rejudges)).collect::<Vec<_>>(),
        s.audit_fences.iter().map(|c| (c.index, c.rejudges)).collect::<Vec<_>>(),
        s.audit_failed,
        s.audit_force,
        s.teach_rounds,
        s.fits,
        s.accumulated,
        s.fit_members,
        s.epoch,
        s.pending_jump,
        notes,
        s.deferred_trust,
    )
}

/// Asserts two states are the same, naming the first line that differs.
fn assert_same_state(eager: &str, deferred: &str) {
    for (k, (a, b)) in eager.lines().zip(deferred.lines()).enumerate() {
        assert_eq!(a, b, "line {k} differs between the eager and the deferred refit");
    }
    assert_eq!(eager.lines().count(), deferred.lines().count());
}

/// The stuck-H collection (`a_recovery_cap_that_spreads_the_misfit_does_not_leave_the_collection_stuck`)
/// at its live state, refitted eagerly (every fit with its trust passes, as before) and deferred
/// (the passes once, for the fit that stands): the same state, bit for bit, through the cap
/// retry, the quarantine rounds and step 3's restores — on the per-scene body-3 basis and on the
/// collection's body-4 one.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_deferred_refit_leaves_the_eager_refits_state() {
    const SLUG: &str = "lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5";
    let key = surrogate::SurfaceKey {
        functional: "lsda".into(),
        h: 0.15,
        sigma: 0.15,
        tol: 1e-6,
        teacher: "Kohn-Sham (GPU)".into(),
        ionic: "all/30a74d9bda495668/shared1e-3/1e-1/stop2x5e-8".into(),
    };
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/stuck-h-collection").join(SLUG);
    for body4_off in [true, false] {
        let mut states = Vec::new();
        for eager in [true, false] {
            let dir = collection_root(&format!("deferred-{body4_off}-{eager}"));
            std::fs::create_dir_all(dir.join(SLUG)).unwrap();
            for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
                std::fs::copy(src.join(f), dir.join(SLUG).join(f)).unwrap();
            }
            let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
            let live = c.quarantine.clone();
            let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
            s.body4_off = body4_off;
            s.eager_trust_passes = eager;
            s.attach(c).unwrap();
            s.quarantine = live;
            s.degree_cap = Some(RECOVERY_DEGREE_CAP);
            s.species_dirty = true;
            s.fit_store().unwrap();
            let fits = s.fits;
            s.refit().unwrap();
            println!("body4_off {body4_off}, eager {eager}: {} fits in the refit, {} held, recovery {:?}, trusted {}", s.fits - fits, s.quarantined(), s.recovery_note(), s.trusted());
            assert!(s.fits - fits >= 3, "the premise: the refit runs the recovery's fits ({})", s.fits - fits);
            assert_eq!(s.deferred_trust, None, "the passes are the standing fit's");
            // and a second refit from there (the recovery's state carried over)
            s.refit().unwrap();
            states.push(refit_state(&s));
        }
        assert_same_state(&states[0], &states[1]);
    }
}

/// The same comparison on a copy of a real collection (`COLLECTION_DIR`; the subsets and replicas
/// of `refit_cost_on_a_collection` are not needed: the whole collection, read-only), with the
/// walls of both refits.
#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> cargo test -p providers --release --lib deferred_refit_on_a_collection -- --ignored --nocapture"]
fn deferred_refit_on_a_collection() {
    let mut states = Vec::new();
    for eager in [true, false] {
        let mut c = open_copy();
        c.read_only = true;
        let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
        s.eager_trust_passes = eager;
        let t = Instant::now();
        s.attach(c).unwrap();
        if s.report().is_none() {
            SurrogateKs::refit(&mut s).unwrap();
        }
        let attach = secs(t);
        let (t, fits) = (Instant::now(), s.fits);
        s.pending_refit = true;
        SurrogateKs::refit(&mut s).unwrap();
        println!("eager {eager}: attach + cold refit {attach:.2} s ({fits} fits); next refit {:.2} s ({} fits, recovery {:?})", secs(t), s.fits - fits, s.recovery_note());
        states.push(refit_state(&s));
    }
    assert_same_state(&states[0], &states[1]);
    println!("the eager and the deferred refit leave the same state");
}
