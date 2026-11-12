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
    rows.push((format!("accumulate {} samples one at a time (add)", head.len()), t_acc));
    let mut acc_many = surrogate::Accumulator::new(&basis);
    let t = Instant::now();
    acc_many.add_many(&head.iter().map(|&i| &store.samples[i]).collect::<Vec<_>>(), &basis, &refs, &opts).unwrap();
    rows.push((format!("accumulate {} samples batched (add_many: the rebuild)", head.len()), secs(t)));
    assert_eq!(format!("{:?}", acc_many.sums()), format!("{:?}", acc.sums()), "add_many is a loop of add");
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
    let n = s.samples();
    let mut quarantine = s.quarantine.clone();
    quarantine.resize(n.max(quarantine.len()), Q_NONE);
    let store: Vec<(SampleTag, u64, usize)> = s.store.as_ref().map_or(Vec::new(), |st| st.samples.iter().map(|x| (x.tag, x.energy.to_bits(), x.pos.len())).collect());
    let lines = [
        format!("store {store:?}"),
        format!("pending_refit {} votes {:?} kept {:?}", s.pending_refit, s.vote_pending, s.fence_kept),
        format!("model {:?}", s.model),
        format!("report {:?}", s.report),
        format!("training {:?}", s.training_residuals),
        format!("class_loo {:?}", s.class_loo),
        format!("class_judge {:?}", s.class_judge),
        format!("class_worst {:?}", s.class_worst),
        format!("class_blockers {:?}", s.class_blockers),
        format!("class_memory {:?}", s.class_memory),
        format!("active_set {:?}", s.active_set),
        format!("null_space {:?}", s.null_space.as_ref().map(|n| n.dimension())),
        format!("quarantine {quarantine:?}"),
        format!("degree_cap {:?}", s.degree_cap),
        format!("recovery {:?}", s.recovery_note()),
        format!("trusted {} {:?}", s.trusted(), s.untrusted_reason()),
        format!("failed_audits {:?}", s.failed_audits.iter().map(|c| (c.index, c.rejudges)).collect::<Vec<_>>()),
        format!("fences {:?}", s.audit_fences.iter().map(|c| (c.index, c.rejudges)).collect::<Vec<_>>()),
        format!("audit {:?} {:?} burst {}", s.audit_failed, s.audit_force, s.audit_burst),
        format!("teach_rounds {:?}", s.teach_rounds),
        format!("fits {} accumulated {} members {:?}", s.fits, s.accumulated, s.fit_members),
        format!("epoch {} jump {:?}", s.epoch, s.pending_jump),
        format!("notes {notes:?}"),
        format!("deferred {:?}", s.deferred_trust),
    ];
    lines.join("
")
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

/// The stuck-H collection at its live state (`a_recovery_cap_that_spreads_the_misfit_does_not_leave_the_collection_stuck`),
/// in its own folder copy `name`: the provider before its refit.
fn stuck_live(name: &str, body4_off: bool) -> SurrogateKs {
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
    let dir = collection_root(name);
    std::fs::create_dir_all(dir.join(SLUG)).unwrap();
    for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
        std::fs::copy(src.join(f), dir.join(SLUG).join(f)).unwrap();
    }
    let (c, _) = surrogate::Collection::open(&dir, &key).unwrap();
    let live = c.quarantine.clone();
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    s.body4_off = body4_off;
    s.attach(c).unwrap();
    s.quarantine = live;
    s.degree_cap = Some(RECOVERY_DEGREE_CAP);
    s.species_dirty = true;
    s.fit_store().unwrap();
    s.user_notes.clear();
    // `dir` is removed when dropped: the folder must outlive the provider
    std::mem::forget(dir);
    s
}

/// What the dynamics append while a background refit runs (`background` module doc): kept samples
/// (jittered copies of stored ones, links `Out` as a refused collection write leaves them), a
/// failed audit, a kept fence vote.
fn append_events(s: &mut SurrogateKs, k: usize) {
    let mut rng = StdRng::seed_from_u64(7 + k as u64);
    for &i in &[10usize, 120, 300] {
        let mut x = s.store.as_ref().unwrap().samples[i].clone();
        for p in x.pos.iter_mut() {
            for a in 0..3 {
                p[a] += rng.gen_range(-1e-3..1e-3);
            }
        }
        let raw = x.energy;
        s.store.as_mut().unwrap().push(x).unwrap();
        let n = s.samples();
        s.face.resize(n, f64::NAN);
        s.quarantine.resize(n, Q_NONE);
        if let Some(sh) = s.shared.as_mut() {
            sh.link.push(super::super::shared::Link::Out);
            sh.raw.push(raw);
            sh.box_n.push(0);
        }
        s.pending_refit = true;
    }
    let x = s.store.as_ref().unwrap().samples[200].clone();
    s.failed_audits.push(FailedAudit { pos: x.pos, z: x.z, energy: x.energy + 0.05, forces: x.forces, index: None, ionic: "none".into(), rejudges: 0, origin: AuditOrigin::default(), pair: None });
    s.sync_audit_failed();
    let n = s.samples();
    s.vote_pending.push((n - 1, vec![3]));
}

/// A background refit with nothing happening meanwhile installs exactly the state a synchronous
/// refit leaves (line by line), on the stuck-H collection's body-3 recovery (8 fits through step 3's
/// restores) and its body-4 one.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_background_refit_installs_the_synchronous_state() {
    for body4_off in [true, false] {
        let mut sync = stuck_live(&format!("bg-sync-{body4_off}"), body4_off);
        let mut bg = stuck_live(&format!("bg-bg-{body4_off}"), body4_off);
        SurrogateKs::refit(&mut sync).unwrap();
        assert!(bg.begin_background_refit());
        assert!(!bg.begin_background_refit(), "one at a time");
        assert!(bg.background_refit_running());
        match bg.finish_background_refit() {
            BackgroundPoll::Installed(r) => {
                r.unwrap();
            }
            _ => panic!("installed"),
        }
        assert!(!bg.background_refit_running());
        assert!(matches!(bg.poll_background_refit(), BackgroundPoll::Idle));
        assert_same_state(&refit_state(&sync), &refit_state(&bg));
        println!("body4_off {body4_off}: background = synchronous ({} fits, {} held)", bg.fits, bg.quarantined());
    }
}

/// What the dynamics append while a background refit runs — samples, a failed audit, a kept vote —
/// survives the install, and the result is the state of a synchronous refit followed by the same
/// appends; the next (synchronous) refit of both is the same too.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn appends_during_a_background_refit_survive_the_install() {
    let mut sync = stuck_live("bg-app-sync", false);
    let mut bg = stuck_live("bg-app-bg", false);
    SurrogateKs::refit(&mut sync).unwrap();
    append_events(&mut sync, 0);
    assert!(bg.begin_background_refit());
    append_events(&mut bg, 0);
    assert!(matches!(bg.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
    assert!(bg.pending_refit, "the appended samples are the next refit's");
    assert_eq!(bg.vote_pending.len(), 1);
    assert!(bg.failed_audits.iter().any(|c| c.index.is_none() && c.origin == AuditOrigin::default()));
    // the surface epoch is the driver's bookkeeping of what it saw: here the failed audit came
    // before the install (the model was untrusted when it installed: no flip to book), there after
    // a trusted install — each right for its own history; everything else is the same
    let without_epoch = |s: &SurrogateKs| refit_state(s).lines().filter(|l| !l.starts_with("epoch ")).collect::<Vec<_>>().join("
");
    assert_same_state(&without_epoch(&sync), &without_epoch(&bg));
    assert_eq!(bg.pending_jump, None);
    SurrogateKs::refit(&mut sync).unwrap();
    SurrogateKs::refit(&mut bg).unwrap();
    assert_same_state(&without_epoch(&sync), &without_epoch(&bg));
}

/// More than appends while a background refit runs — a policy change (the model generation moves),
/// a stored sample's quarantine code — discards its result: the model that stood still stands, and
/// the refit is due again.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_conflict_discards_a_background_refit() {
    let mut s = stuck_live("bg-conflict-policy", false);
    let model = format!("{:?}", s.model);
    assert!(s.begin_background_refit());
    let p = s.policy.clone();
    s.set_policy(p);
    match s.finish_background_refit() {
        BackgroundPoll::Discarded(why) => assert!(why.contains("the model, its samples or the policy"), "{why}"),
        _ => panic!("discarded"),
    }
    assert!(s.pending_refit);
    assert_eq!(format!("{:?}", s.model), model, "the model that stood still stands");

    let mut s = stuck_live("bg-conflict-code", false);
    let i = (0..s.samples()).find(|&i| !s.excluded(i)).unwrap();
    assert!(s.begin_background_refit());
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[i] = Q_SUPERSEDED;
    match s.finish_background_refit() {
        BackgroundPoll::Discarded(why) => assert!(why.contains("quarantine code"), "{why}"),
        _ => panic!("discarded"),
    }
    assert!(s.pending_refit);
}

/// Review finding 1: a box reference that arrives while a background refit runs rebases that box's
/// stored energies without moving the canonical references — a conflict: the result is discarded
/// (installing it would put the old energies back under the new references).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_box_reference_arriving_mid_refit_discards_it() {
    let mut s = stuck_live("bg-box-ref", false);
    assert!(s.begin_background_refit());
    let sh = s.shared.as_mut().unwrap();
    let k = *sh.box_refs.keys().next().unwrap();
    *sh.box_refs.get_mut(&k).unwrap() += 1e-6;
    match s.finish_background_refit() {
        BackgroundPoll::Discarded(why) => assert!(why.contains("references"), "{why}"),
        _ => panic!("discarded"),
    }
    assert!(s.pending_refit);
}

/// Review finding 2: a key the live provider's between-refit evidence made leave while a background
/// refit ran stays out at the install — the state is a synchronous refit's followed by the same
/// leave and its re-judge, line by line.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_key_that_left_during_a_background_refit_stays_out() {
    let mut sync = stuck_live("bg-left-sync", false);
    let mut bg = stuck_live("bg-left-bg", false);
    // a first refit each, so that keys stand trusted in the memory
    SurrogateKs::refit(&mut sync).unwrap();
    SurrogateKs::refit(&mut bg).unwrap();
    let key = sync.class_memory.iter().find(|(_, &(t, _))| t).map(|(k, _)| k.clone()).expect("a trusted key");
    let leave = |s: &mut SurrogateKs| {
        s.class_memory.insert(key.clone(), (false, None));
    };
    SurrogateKs::refit(&mut sync).unwrap();
    leave(&mut sync);
    sync.rejudge_classes();
    assert!(bg.begin_background_refit());
    leave(&mut bg);
    assert!(matches!(bg.finish_background_refit(), BackgroundPoll::Installed(Ok(_))));
    assert_same_state(&refit_state(&sync), &refit_state(&bg));
    println!("key {key:?}: after the install {:?}", bg.class_memory.get(&key));
}

/// A background refit asked to stop (its provider let it go) stops at its next fit; its result is
/// an error and installs nothing new — the model that stood still stands.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_cancelled_background_refit_installs_nothing_new() {
    let mut s = stuck_live("bg-cancel", false);
    let model = format!("{:?}", s.model);
    assert!(s.begin_background_refit());
    s.background.as_ref().unwrap().cancel_now();
    match s.finish_background_refit() {
        BackgroundPoll::Installed(Err(_)) | BackgroundPoll::Discarded(_) => {}
        _ => panic!("a cancelled refit must not install a new fit"),
    }
    if format!("{:?}", s.model) != model {
        // the thread may have finished its first fit before the flag was read: then the model is that
        // fit's, as a synchronous refit interrupted there would leave it — but never trusted anew
        // without its trust passes
        assert_eq!(s.deferred_trust, None);
    }
}
