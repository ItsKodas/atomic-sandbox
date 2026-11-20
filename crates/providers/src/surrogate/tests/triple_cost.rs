//! Measurement (2026-11-17, spec 2026-11-17 §6.4): the wall of the three-atom coverage map's build
//! (`SurrogateKs::triple_map`) on a shared collection, against the budget — cold (an empty fitted
//! block cache) ≤ 100 ms, pending-only ≤ 2 ms — beside the knowledge map's own build. Reported,
//! never asserted; nothing here changes what the map counts or when it is built.
//!
//! Reads a **copy** of a collection folder (`COLLECTION_DIR`), opened read-only through
//! `refit_profile`'s helpers. Each timing is the median of [`CALLS`] calls:
//! - (a) `knowledge_map()` on the fit that stands;
//! - (b) `triple_map()` cold: a fresh provider per call (attach, its own refit), so the cache is
//!   empty;
//! - (c) `triple_map()` again with nothing new (the cached fitted block, the pending pass, the
//!   assembly);
//! - (d) `triple_map()` pending-only: `refit_profile`'s `append` recipe — the collection less its
//!   last [`APPEND_N`] samples attached, the map built once, then those samples pushed with
//!   `push_shared` and **no** refit, so they are pending and the fitted block stands.
//!
//! `COLLECTION_DIR=<copy> cargo test -p providers --release --lib triple_cost -- --ignored --nocapture --test-threads 1`
use super::refit_profile::{attach_timed, open_copy, provider};
use super::*;
use std::time::Instant;

/// The calls per timing (a median).
const CALLS: usize = 5;
/// The held-back tail of the `append` recipe (`refit_profile`'s default `APPEND_N`).
const APPEND_N: usize = 20;

fn ms<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let t = Instant::now();
    let out = f();
    (t.elapsed().as_secs_f64() * 1e3, out)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn list(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", ")
}

#[test]
#[ignore = "measurement: COLLECTION_DIR=<copy> cargo test -p providers --release --lib triple_cost -- --ignored --nocapture --test-threads 1"]
fn triple_cost() {
    let (mut know, mut cold, mut warm) = (Vec::new(), Vec::new(), Vec::new());
    let mut shape = None;
    for call in 0..CALLS {
        let mut s = provider();
        let attach = attach_timed(&mut s, open_copy());
        let (t_cold, m) = ms(|| s.triple_map().expect("a fitted model"));
        let (t_warm, m2) = ms(|| s.triple_map().expect("a fitted model"));
        let (t_know, _) = ms(|| s.knowledge_map().expect("a fitted model"));
        assert_eq!(m.types.len(), m2.types.len());
        println!("call {call}: attach {attach:.1} s; triple_map cold {t_cold:.3} ms, again {t_warm:.3} ms; knowledge_map {t_know:.3} ms");
        if call == 0 {
            for t in &m.types {
                println!("  type (a {}, centre {}, b {}): {} fitted triples in {} samples", t.t.a, t.t.centre, t.t.b, t.triples, t.samples);
            }
        }
        cold.push(t_cold);
        warm.push(t_warm);
        know.push(t_know);
        let pending = (0..s.store.as_ref().unwrap().len()).filter(|&i| s.is_pending(i)).count();
        let atoms: usize = s.fitted_indices().iter().map(|&i| s.store.as_ref().unwrap().samples[i].pos.len()).sum();
        shape = Some((
            s.store.as_ref().unwrap().len(),
            s.fitted_indices().len(),
            atoms,
            m.types.len(),
            m.types.iter().map(|t| t.triples).sum::<u64>(),
            pending,
            m.r_cut,
            m.grid.cells(),
        ));
    }

    // (d) the append recipe: the tail held back, attached, the map built, the tail pushed, no refit
    let mut c = open_copy();
    let n = c.samples.len() - APPEND_N;
    let tail: Vec<_> = c.samples.split_off(n);
    c.quarantine.truncate(n);
    let mut s = provider();
    let attach = attach_timed(&mut s, c);
    let (t_first, before) = ms(|| s.triple_map().expect("a fitted model"));
    let prov0 = s.shared.as_ref().unwrap().prov.clone();
    let mut pushed = 0;
    for cs in tail {
        let prov = Provenance { grid_n: cs.grid_n, ..prov0.clone() };
        match s.push_shared(cs.sample, &prov, f64::NAN, cs.multiplicity) {
            Ok(()) => pushed += 1,
            Err(e) => println!("push refused: {e:?}"),
        }
    }
    let mut pend = Vec::new();
    let mut after = None;
    for _ in 0..CALLS {
        let (t, m) = ms(|| s.triple_map().expect("a fitted model"));
        pend.push(t);
        after = Some(m);
    }
    let after = after.unwrap();
    let pending_samples = (0..s.store.as_ref().unwrap().len()).filter(|&i| s.is_pending(i)).count();
    let pending_triples: u64 = after.types.iter().map(|t| t.pending.iter().map(|&k| k as u64).sum::<u64>()).sum();
    let fitted_same = before.types.iter().zip(&after.types).all(|(a, b)| std::sync::Arc::ptr_eq(&a.counts, &b.counts)) && before.types.len() == after.types.len();
    println!(
        "append: attach of the first {n} samples {attach:.1} s, its cold map {t_first:.3} ms; pushed {pushed} of {APPEND_N}; pending samples {pending_samples}, pending triples {pending_triples}; fitted block reused (Arc) {fitted_same}"
    );

    let (stored, fitted, atoms, types, triples, pending, r_cut, cells) = shape.unwrap();
    println!("\nworker threads {}; r_cut {r_cut:.1} Bohr, {cells} cells per type", surrogate::threads::worker_threads());
    println!("stored {stored}, fitted {fitted} ({atoms} atoms), triple types {types}, fitted triples {triples}, pending samples at attach {pending}");
    println!("\n| build | median ms of {CALLS} | budget ms | calls ms |\n|---|---|---|---|");
    println!("| (a) knowledge_map() | {:.3} | — | {} |", median(know.clone()), list(&know));
    println!("| (b) triple_map() cold (empty cache) | {:.3} | 100 | {} |", median(cold.clone()), list(&cold));
    println!("| (c) triple_map() again, nothing new | {:.3} | — | {} |", median(warm.clone()), list(&warm));
    println!("| (d) triple_map() after +{pushed} appended (pending-only) | {:.3} | 2 | {} |", median(pend.clone()), list(&pend));
}
