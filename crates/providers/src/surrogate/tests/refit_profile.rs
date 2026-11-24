//! Measurement (2026-11-17): where a shared collection's refit spends its wall, stage by stage,
//! with the stage timers of `surrogate::prof` (spans in `surrogate::fit`, `loo`, `grade`,
//! `novelty` and the provider's `trust`, `quarantine`, `shared`, `mod`) — built with
//! `--features profile`, else the stage tables are empty. Reported, never asserted; nothing here
//! changes what a refit does.
//!
//! Reads a **copy** of a collection folder (`COLLECTION_DIR`), opened read-only so every run
//! starts from the same files. `REFIT_PROFILE` is a comma list of:
//! - `cold`: attach (whose own refit is the cold full refit), then a second refit with nothing new;
//! - `append`: the collection less its last `APPEND_N` (20) samples attached, then those samples
//!   appended through the collection-mode store path (`push_shared`) and the refit the worker runs;
//! - `scale`: the LOO pass, the class pass and the training residuals on subsets (first ½, H only,
//!   H + O, all), each attached without the trust recovery (`skip_recovery`).
//!
//! `COLLECTION_DIR=<copy> REFIT_PROFILE=cold,append,scale cargo test -p providers --release --features profile --lib refit_profile -- --ignored --nocapture --test-threads 1`
use super::*;
use std::time::Instant;
use surrogate::Collection;

pub(super) fn open_copy() -> Collection {
    let dir = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").expect("COLLECTION_DIR names a collection folder copy"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dir.join("key.txt")).unwrap()).unwrap();
    let (mut c, _) = surrogate::Collection::open(dir.parent().unwrap(), &key).unwrap();
    c.read_only = true;
    c
}

fn state(s: &SurrogateKs) -> String {
    let r = s.report().unwrap();
    let b = &s.model.as_ref().unwrap().basis;
    let store = s.store.as_ref().unwrap();
    let fitted: Vec<usize> = (0..s.accumulated.min(store.len())).filter(|&i| !s.excluded(i)).collect();
    let atoms: usize = fitted.iter().map(|&i| store.samples[i].pos.len()).sum();
    format!(
        "stored {}, fitted {} ({} atoms, {:.2}/sample), held {}, species {:?}, Q′ {}, body3 {} body4 {} (D₄ {}), degree_cap {}, n_samples {}; trusted {} ({:?}); misfit |M| {}; recovery {:?}; last solve {:?}",
        store.len(),
        fitted.len(),
        atoms,
        atoms as f64 / fitted.len().max(1) as f64,
        s.quarantined(),
        b.species,
        r.q_prime,
        b.body3,
        b.body4,
        b.body4_degree,
        b.degree_cap,
        r.n_samples,
        s.trusted(),
        s.untrusted_reason(),
        s.misfit().map_or(0, |m| m.samples.len()),
        s.recovery_note(),
        s.accumulator.as_ref().map(|a| a.last_refit()),
    )
}

fn dump(label: &str, wall: f64, s: &SurrogateKs) {
    let (rows, counts) = surrogate::prof::take();
    println!("\n## {label}: wall {wall:.2} s\n{}", state(s));
    println!("| stage | calls | s |\n|---|---|---|");
    for (k, n, t) in rows {
        println!("| {k} | {n} | {t:.3} |");
    }
    for (k, n) in counts {
        println!("| count: {k} | {n} | |");
    }
}

pub(super) fn provider() -> SurrogateKs {
    SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6)
}

/// Attach (its own refit is the cold one) and time it; a refit when the attach did not fit.
pub(super) fn attach_timed(s: &mut SurrogateKs, c: Collection) -> f64 {
    let t = Instant::now();
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(s).unwrap();
    }
    t.elapsed().as_secs_f64()
}

fn refit_timed(s: &mut SurrogateKs) -> f64 {
    s.pending_refit = true;
    let t = Instant::now();
    SurrogateKs::refit(s).unwrap();
    t.elapsed().as_secs_f64()
}

/// The provider-side passes on the standing fit, each timed alone (the scaling run).
fn passes(s: &mut SurrogateKs, label: &str) {
    let _ = surrogate::prof::take();
    s.loo_ratios = None;
    let t = Instant::now();
    let _ = s.ensure_loo_ratios();
    let loo = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = s.compute_class_loo_full();
    let class = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = s.compute_residuals();
    let resid = t.elapsed().as_secs_f64();
    let t = Instant::now();
    s.rebuild_active_set();
    let active = t.elapsed().as_secs_f64();
    let (rows, _) = surrogate::prof::take();
    let cpu = |name: &str| rows.iter().find(|r| r.0.contains(name)).map_or(0.0, |r| r.2);
    let store = s.store.as_ref().unwrap();
    let fitted: Vec<usize> = (0..s.accumulated.min(store.len())).filter(|&i| !s.excluded(i)).collect();
    let atoms: usize = fitted.iter().map(|&i| store.samples[i].pos.len()).sum();
    let classes = s.class_judge.as_ref().map_or(0, |j| j.key_stats.len());
    println!(
        "| {label} | {} | {atoms} | {} | {} | {classes} | {loo:.3} | {class:.3} | {resid:.3} | {active:.3} | {:.2} | {:.2} | {:.3} |",
        fitted.len(),
        s.model.as_ref().unwrap().basis.species.len(),
        s.report().unwrap().q_prime,
        cpu("design_rows"),
        cpu("triangular"),
        cpu("serial per-sample aggregation"),
    );
}

#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> REFIT_PROFILE=cold,append,scale cargo test -p providers --release --features profile --lib refit_profile -- --ignored --nocapture --test-threads 1"]
fn refit_profile() {
    println!("worker threads {}", surrogate::threads::worker_threads());
    let runs = std::env::var("REFIT_PROFILE").unwrap_or_else(|_| "cold,append".into());
    for run in runs.split(',').map(str::trim) {
        match run {
            "cold" => {
                let mut s = provider();
                surrogate::prof::reset();
                let wall = attach_timed(&mut s, open_copy());
                dump("(b) cold: attach + full refit", wall, &s);
                let wall = refit_timed(&mut s);
                dump("(b2) second refit, nothing new", wall, &s);
            }
            "append" => {
                let k: usize = std::env::var("APPEND_N").map_or(20, |v| v.parse().unwrap());
                let mut c = open_copy();
                let n = c.samples.len() - k;
                let tail: Vec<_> = c.samples.split_off(n);
                c.quarantine.truncate(n);
                let mut s = provider();
                surrogate::prof::reset();
                let wall = attach_timed(&mut s, c);
                dump(&format!("(a0) attach of the first {n} samples (cold)"), wall, &s);
                let prov0 = s.shared.as_ref().unwrap().prov.clone();
                let mut pushed = 0;
                for cs in tail {
                    let prov = Provenance { grid_n: cs.grid_n, ..prov0.clone() };
                    match s.push_shared(cs.sample, &prov, f64::NAN, cs.multiplicity) {
                        Ok(()) => pushed += 1,
                        Err(e) => println!("push refused: {e:?}"),
                    }
                }
                println!("appended {pushed} of {k} held-back samples through push_shared");
                let wall = refit_timed(&mut s);
                dump(&format!("(a) refit after +{pushed} appended samples"), wall, &s);
            }
            "scale" => {
                println!("\n| subset | fitted | atoms | species | Q′ | env keys | LOO pass s | class pass s | residuals s | active set s | LOO rows+predict cpu-s | LOO solves cpu-s | class serial agg s |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|");
                let subsets: Vec<(&str, Box<dyn Fn(usize, usize, &surrogate::CollectionSample) -> bool>)> = vec![
                    ("first 1/2", Box::new(|i, n, _| i < n / 2)),
                    ("H only", Box::new(|_, _, x| x.sample.z.iter().all(|&z| z == 1))),
                    ("H + O", Box::new(|_, _, x| x.sample.z.iter().all(|&z| z == 1 || z == 8))),
                    ("all", Box::new(|_, _, _| true)),
                ];
                for (name, keep) in subsets {
                    let mut c = open_copy();
                    let n = c.samples.len();
                    let idx: Vec<usize> = (0..n).filter(|&i| keep(i, n, &c.samples[i])).collect();
                    c.samples = idx.iter().map(|&i| c.samples[i].clone()).collect();
                    c.quarantine = idx.iter().map(|&i| c.quarantine[i]).collect();
                    let mut s = provider();
                    s.skip_recovery = true;
                    let _ = attach_timed(&mut s, c);
                    passes(&mut s, name);
                }
            }
            // the misfit-driven recovery: the cold attach with the energy tier at a third, so the
            // training residuals fail and the recovery runs its cap retries and quarantine rounds
            "recovery" => {
                let mut pol = FallbackPolicy::default();
                pol.energy_std_per_atom /= 3.0;
                let mut s = SurrogateKs::new(Box::new(StubKsGpu), pol).with_teacher_tol(1e-6);
                surrogate::prof::reset();
                let wall = attach_timed(&mut s, open_copy());
                dump("(r) cold attach, energy tier / 3 (misfit-driven recovery)", wall, &s);
            }
            // the active set (rule 5e) alone on the standing fit, its sub-stages
            "active" => {
                let mut s = provider();
                s.skip_recovery = true;
                let _ = attach_timed(&mut s, open_copy());
                let _ = surrogate::prof::take();
                let t = Instant::now();
                s.rebuild_active_set();
                dump("active set alone (rule 5e), default basis", t.elapsed().as_secs_f64(), &s);
            }
            // a channel collection's first refit (the W learn, its fit and every trust pass) and a
            // second refit with nothing new (refit visualisation, 2026-11-25: where the time goes);
            // `CHANNELS_K` (default 3)
            "channels" => {
                let k: usize = std::env::var("CHANNELS_K").map_or(3, |v| v.parse().unwrap());
                let mut c = open_copy();
                c.basis = surrogate::BasisKind::Channels { k };
                let mut s = provider();
                s.attach_refit_off = true;
                s.attach(c).unwrap();
                surrogate::prof::reset();
                let wall = refit_timed(&mut s);
                dump(&format!("(c) channels K = {k}: first refit (learn W + fit + trust passes)"), wall, &s);
                let wall = refit_timed(&mut s);
                dump(&format!("(c2) channels K = {k}: second refit, nothing new"), wall, &s);
            }
            // a launch on a channel collection whose channels.txt holds W (2026-11-25, the
            // coordinator's question: does the launch refit relearn W?): a writable temporary copy
            // of COLLECTION_DIR, a first provider learns W and writes it, then a fresh provider
            // attaches the copy as a launch does and refits; the notes and the stage table say
            // whether the W learn ran
            "channels_launch" => {
                let k: usize = std::env::var("CHANNELS_K").map_or(3, |v| v.parse().unwrap());
                let src = std::path::PathBuf::from(std::env::var("COLLECTION_DIR").unwrap());
                let root = std::env::temp_dir().join(format!("refit-launch-{}", std::process::id()));
                let dst = root.join(src.file_name().unwrap());
                let _ = std::fs::remove_dir_all(&root);
                std::fs::create_dir_all(&dst).unwrap();
                for e in std::fs::read_dir(&src).unwrap() {
                    let e = e.unwrap();
                    let name = e.file_name();
                    if e.file_type().unwrap().is_file() && name != ".lock" {
                        std::fs::copy(e.path(), dst.join(&name)).unwrap();
                    }
                }
                let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(dst.join("key.txt")).unwrap()).unwrap();
                let open = || {
                    let (mut c, _) = surrogate::Collection::open(&root, &key).unwrap();
                    c.basis = surrogate::BasisKind::Channels { k };
                    assert!(!c.read_only, "the temporary copy is writable");
                    c
                };
                {
                    let mut s = provider();
                    s.attach_refit_off = true;
                    s.attach(open()).unwrap();
                    surrogate::prof::reset();
                    let wall = refit_timed(&mut s);
                    for n in s.user_notes.drain(..) {
                        println!("  note: {n}");
                    }
                    dump(&format!("(l1) channels K = {k}: the first refit, W learnt and written"), wall, &s);
                }
                assert!(dst.join("channels.txt").exists(), "W written to the copy");
                let mut s = provider();
                s.attach_refit_off = true;
                s.attach(open()).unwrap();
                println!("  attached: W read from channels.txt: {}", s.shared.as_ref().unwrap().mix.is_some());
                surrogate::prof::reset();
                let wall = refit_timed(&mut s);
                for n in s.user_notes.drain(..) {
                    println!("  note: {n}");
                }
                dump(&format!("(l2) channels K = {k}: the launch refit (W on file)"), wall, &s);
                let _ = std::fs::remove_dir_all(&root);
            }
            other => panic!("unknown run {other}"),
        }
    }
}
