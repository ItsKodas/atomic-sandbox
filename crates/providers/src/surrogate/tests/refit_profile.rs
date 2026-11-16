//! TEMPORARY measurement (branch `refit-profile`, 2026-11-17; never merged): where a shared
//! collection's refit spends its wall, stage by stage, with the stage timers of `surrogate::prof`
//! (spans in `surrogate::fit`, `loo`, `novelty` and the provider's `trust`, `quarantine`,
//! `shared`, `mod`). Reported, never asserted; nothing here changes what a refit does.
//!
//! Reads a **copy** of a collection folder (`COLLECTION_DIR`), opened read-only so every run
//! starts from the same files. `REFIT_PROFILE` is a comma list of:
//! - `cold`: attach (whose own refit is the cold full refit), then a second refit with nothing new;
//! - `append`: the collection less its last `APPEND_N` (20) samples attached, then those samples
//!   appended through the collection-mode store path (`push_shared`) and the refit the worker runs;
//! - `scale`: the LOO pass, the class pass and the training residuals on subsets (first ½, H only,
//!   H + O, all), each attached without the trust recovery (`skip_recovery`).
//!
//! `COLLECTION_DIR=<copy> REFIT_PROFILE=cold,append,scale cargo test -p providers --release --lib refit_profile -- --ignored --nocapture --test-threads 1`
use super::*;
use std::time::Instant;
use surrogate::Collection;

fn open_copy() -> Collection {
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

fn provider() -> SurrogateKs {
    SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6)
}

/// Attach (its own refit is the cold one) and time it; a refit when the attach did not fit.
fn attach_timed(s: &mut SurrogateKs, c: Collection) -> f64 {
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
#[ignore = "measurement: COLLECTION_DIR=<copy> REFIT_PROFILE=cold,append,scale cargo test -p providers --release --lib refit_profile -- --ignored --nocapture --test-threads 1"]
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
            other => panic!("unknown run {other}"),
        }
    }
}
