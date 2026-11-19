//! Fermi–Dirac spec §7.6 and §8 (plan 2026-11-13 Task 15): code 9 (`Q_RESOLVED`) in the trust machinery and its
//! leave-one-out review. A re-solved sample is out of the fit, the envelope, the grade, the leave-one-out and the class
//! statistics; it is not counted by the held cap or `fencing()`; before its review it does not fence. Once the migration
//! pass is complete, a refit reviews every code-9 sample by its leave-one-out ratio in a candidate fit with its siblings:
//! ≤ 1 releases it (9 → 0), > 1 keeps it at 9, fencing (F4) — never code 1, never lifted by purge's votes, moved by one
//! agreeing fresh answer like a held sample (controller ruling B). Its verdict is in `migration.txt`'s review log, read
//! back at attach (R13) and remapped by a purge (F3). Atomic units throughout (Hartree, Bohr).
//!
//! The fixture: the 40 exact pair-Morse H₂ samples of `forget::small_collection` (r ∈ [1.1, 2.3], a shared collection),
//! and re-solved samples appended as the re-solve pass appends them (`take_migrated` at code 9). "Another state" is the
//! analytic surface E_S(r) = E_Morse(r) + δ·(r − R_S)² ([`shifted`]): a whole region that changed together (spec §7.6,
//! "Why this rule") — the fit without any of them cannot reproduce them, the fit with the others predicts each.

use super::super::shared::{predecessor_slug, Link};
use super::*;
use surrogate::{CollectionSample, MigrationLog, MigrationState};

/// The changed state's curvature (Ha/Bohr²) and its origin (Bohr): 2.4 mHa above the Morse curve at r = 2.5 (its force
/// 2.4e-2 Ha/Bohr off), 29 mHa at r = 3.0 — beyond the policy's 1 mHa/atom and its force bounds at every sibling.
const DELTA: f64 = 0.06;
const R_S: f64 = 2.3;
/// The siblings: [`SIBLINGS`] bond lengths evenly on [`SIB_LO`, `SIB_HI`] (Bohr), beyond the stored window.
const SIBLINGS: usize = 10;
const SIB_LO: f64 = 2.5;
const SIB_HI: f64 = 3.0;
/// The inconsistent sample's offset (Ha): 50 mHa, 25× the per-atom energy bound.
const OFFSET: f64 = 0.05;
/// A second change of state beyond the siblings: E_T(r) = E_S(r) + `TAIL`·max(0, r − `R_T`)² (Ha/Bohr², Bohr) — at
/// r = 3.4, 16 mHa above the changed state's continuation and 8e-2 Ha/Bohr off its force.
const TAIL: f64 = 0.1;
const R_T: f64 = 3.0;

/// [`answer`] on the changed state with the second change beyond `R_T` ([`TAIL`]).
fn answer_tail(r: f64, rng: &mut StdRng) -> (Scene, Sample) {
    let (sc, mut smp) = answer(r, 0.0, true, rng);
    let d: [f64; 3] = std::array::from_fn(|a| smp.pos[1][a] - smp.pos[0][a]);
    let rr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let x = (rr - R_T).max(0.0);
    smp.energy += TAIL * x * x;
    for a in 0..3 {
        smp.forces[1][a] -= 2.0 * TAIL * x * d[a] / rr;
        smp.forces[0][a] += 2.0 * TAIL * x * d[a] / rr;
    }
    (sc, smp)
}

/// H₂ at `r` along a random direction (`h2_at`) labelled on the changed state plus `extra` (Ha): E_Morse + δ(r − R_S)²
/// + extra, forces −∂E/∂R. `shift` false: the Morse curve (a fresh answer of the stored surface).
fn answer(r: f64, extra: f64, shift: bool, rng: &mut StdRng) -> (Scene, Sample) {
    let sc = h2_at(r, rng);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let (mut e, mut f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
    if shift {
        let d: [f64; 3] = std::array::from_fn(|a| pos[1][a] - pos[0][a]);
        let rr = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        e += DELTA * (rr - R_S).powi(2);
        let ds = 2.0 * DELTA * (rr - R_S);
        for a in 0..3 {
            f[1][a] -= ds * d[a] / rr;
            f[0][a] += ds * d[a] / rr;
        }
    }
    e += extra;
    (sc, Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Perturbation, weight: 1.0, force_mask: None, use_energy: true })
}

/// Appends `smp` as the re-solve pass appends a class-S sample (`take_migrated` at code 9); its store index.
fn take_resolved(s: &mut SurrogateKs, smp: Sample) -> usize {
    let cs = CollectionSample { sample: smp, grid_n: 33, multiplicity_forced: false, multiplicity: Some(1), scene: "H2".into() };
    s.take_migrated(cs, Q_RESOLVED).unwrap();
    s.samples() - 1
}

/// The siblings on the changed state, appended at code 9; their store indices.
fn siblings(s: &mut SurrogateKs, rng: &mut StdRng) -> Vec<usize> {
    (0..SIBLINGS).map(|i| {
        let r = SIB_LO + (SIB_HI - SIB_LO) * i as f64 / (SIBLINGS - 1) as f64;
        let (_, smp) = answer(r, 0.0, true, rng);
        take_resolved(s, smp)
    }).collect()
}

/// A `migration.txt` at `state` for the attached collection (its source the collection's predecessor slug), read in as
/// an attach reads it (`find_migration`).
fn set_migration(s: &mut SurrogateKs, state: MigrationState) {
    let key = s.collection().unwrap().key.clone();
    let log = MigrationLog {
        source_slug: predecessor_slug(&key).expect("this build's surface"),
        source_key: String::new(),
        build: "test".into(),
        date: "2026-11-20".into(),
        source_len: 0,
        source_hash: 0,
        cursor: 0,
        state,
        counts: [0; 4],
        not_migrated: Vec::new(),
        references: Vec::new(),
        class_u_max: (0.0, 0.0),
        review: Vec::new(),
        waiting: Vec::new(),
        pending: None,
        orphaned: Vec::new(),
    };
    surrogate::write_migration(s.collection().unwrap().folder(), &log).unwrap();
    s.find_migration();
    assert!(s.migration.as_ref().is_some_and(|m| m.log.as_ref().is_some_and(|l| l.state == state)), "the slot holds the log");
}

/// The attached collection's `migration.txt` as on disk.
fn disk_log(s: &SurrogateKs) -> MigrationLog {
    surrogate::read_migration(s.collection().unwrap().folder()).unwrap().expect("a migration.txt")
}

/// Stored sample `i`'s collection index.
fn coll_index(s: &SurrogateKs, i: usize) -> usize {
    match s.shared.as_ref().unwrap().link[i] {
        Link::In(k) => k,
        _ => panic!("sample {i} is in the collection"),
    }
}

/// The review log's verdict for stored sample `i`: (ratio, released).
fn verdict(s: &SurrogateKs, i: usize) -> Option<(f64, bool)> {
    let k = coll_index(s, i);
    disk_log(s).review.iter().find(|e| e.0 == k).map(|e| (e.1, e.2))
}

fn geometry(s: &SurrogateKs, i: usize) -> (Vec<[f64; 3]>, Vec<u8>) {
    let smp = &s.store.as_ref().unwrap().samples[i];
    (smp.pos.clone(), smp.z.clone())
}

/// Stored sample `k`'s geometry as a scene in the 33³ cell, its bond stretched by `stretch` (Bohr) about its midpoint.
fn near_sample(s: &SurrogateKs, k: usize, stretch: f64) -> Scene {
    let (p, z) = geometry(s, k);
    let d: [f64; 3] = std::array::from_fn(|a| p[1][a] - p[0][a]);
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let mut sc = dimer(1.0, [1.0, 0.0, 0.0], [0.0; 3], z[0], z[1]);
    sc.nuclei[0].pos = std::array::from_fn(|a| p[0][a] - 0.5 * stretch * d[a] / r);
    sc.nuclei[1].pos = std::array::from_fn(|a| p[1][a] + 0.5 * stretch * d[a] / r);
    sc
}

/// One evaluation at `sc` after an edit, then the refit a kept answer made due: its source, outcome and notes.
fn step(s: &mut SurrogateKs, sc: &Scene) -> (Source, Option<TeachOutcome>, Vec<String>) {
    s.discontinuity();
    let r = s.evaluate(sc, &mut never()).unwrap();
    let outcome = s.take_events().into_iter().last().map(|e| e.outcome);
    if s.pending_refit() {
        s.refit().unwrap();
    }
    (r.stats.source, outcome, r.stats.notes)
}

/// The model's energy and force bits at a few dimers (the fit, bit for bit).
fn prediction_bits(s: &SurrogateKs) -> Vec<u64> {
    let m = s.model.as_ref().expect("a model");
    let mut bits = Vec::new();
    for r in [1.25, 1.63, 2.15, 2.6, 2.9] {
        let pos = vec![[-r / 2.0, 0.1, 0.0], [r / 2.0, 0.1, 0.0]];
        let p = m.predict(&pos, &[1, 1], true).unwrap();
        bits.push(p.energy.to_bits());
        bits.extend(p.forces.iter().flatten().map(|x| x.to_bits()));
        bits.push(p.energy_std.unwrap().to_bits());
    }
    bits
}

/// §7.6: a code-9 sample is out of the fit, the envelope, the grade (the active set), the leave-one-out and the class
/// statistics like a held sample — the refit with eleven of them (no review: the pass is still running) is the fit
/// without them, bit for bit.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn code9_out_of_fit_envelope_grade_loo_and_stats() {
    let dir = collection_root("code9-out");
    let (mut s, _) = super::forget::small_collection(&dir);
    // the control: the same refit with nothing added (a refit of an unchanged store is the reference, bit for bit)
    s.refit().unwrap();
    s.ensure_loo_ratios();
    let before = (prediction_bits(&s), s.report().unwrap().n_samples, s.envelope.clone(), s.loo_ratios.as_ref().unwrap().rows.clone(), s.class_loo().iter().map(|c| c.samples).collect::<Vec<_>>());
    let active = format!("{:?}", s.active_set);
    let mut rng = StdRng::seed_from_u64(31);
    let mut nine = siblings(&mut s, &mut rng);
    let (_, inside) = answer(1.62, OFFSET, false, &mut rng);
    nine.push(take_resolved(&mut s, inside));
    set_migration(&mut s, MigrationState::Running);
    s.refit().unwrap();
    assert!(nine.iter().all(|&i| s.quarantine[i] == Q_RESOLVED && s.excluded(i) && !s.in_standing_fit(i)), "out of the fit");
    assert_eq!(s.report().unwrap().n_samples, before.1, "the fit's sample count");
    assert_eq!(prediction_bits(&s), before.0, "the fit without them, bit for bit");
    assert_eq!(s.envelope, before.2, "the envelope");
    assert_eq!(format!("{:?}", s.active_set), active, "the grade's active sets");
    s.ensure_loo_ratios();
    assert_eq!(s.loo_ratios.as_ref().unwrap().rows, before.3, "the leave-one-out rows");
    assert_eq!(s.class_loo().iter().map(|c| c.samples).collect::<Vec<_>>(), before.4, "the class statistics");
    assert_eq!(s.reviews_run, 0);
}

/// §7.6, F4: code 9 is not counted against the recovery's 10 % held cap (`quarantined() + forgotten()`) nor by
/// `fencing()` — eleven of 51 samples, 22 % — and a kept one fences without being counted (controller ruling A: the
/// fence's on-switch reads `any_fences`).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn code9_not_counted_in_the_held_cap_or_fencing() {
    let dir = collection_root("code9-cap");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(32);
    let mut nine = siblings(&mut s, &mut rng);
    let (_, inside) = answer(1.62, OFFSET, false, &mut rng);
    let k = take_resolved(&mut s, inside);
    nine.push(k);
    set_migration(&mut s, MigrationState::Running);
    s.refit().unwrap();
    assert_eq!((s.quarantined(), s.forgotten(), s.fencing()), (0, 0, 0), "eleven code-9 samples of {}", s.samples());
    assert!(!s.any_fences() && !s.fences(k), "before a review nothing fences");
    // a kept verdict (what a review that kept it leaves): it fences, and is still not counted
    s.review_kept.insert(k);
    assert!(s.fences(k) && s.any_fences());
    assert_eq!((s.quarantined(), s.forgotten(), s.fencing()), (0, 0, 0), "a kept code 9 is no misfit hold");
    let (pos, z) = geometry(&s, k);
    assert_eq!(s.near_quarantined(&pos, &z), vec![k], "the fence stands without a code-1 or code-7 sample");
    // a Forget's cap warning counts held + forgotten only: 1 of 51
    let id = s.sample_id(3).unwrap();
    s.forget_sample(id).unwrap();
    let note = s.user_notes.last().unwrap();
    assert!(note.contains("sample #3 forgotten") && !note.contains("past the"), "{note}");
}

/// §7.6: before its review a code-9 sample does not fence — the fit simply has no data there; the model answers at its
/// own geometry (inside the stored window, where the label is 50 mHa off the model).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn code9_before_review_does_not_fence() {
    let dir = collection_root("code9-nofence");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(33);
    let (_, inside) = answer(1.62, OFFSET, false, &mut rng);
    let k = take_resolved(&mut s, inside);
    set_migration(&mut s, MigrationState::Running);
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert!(!s.fences(k) && !s.any_fences());
    let (pos, z) = geometry(&s, k);
    assert!(s.near_quarantined(&pos, &z).is_empty());
    s.ensure_desc_cache();
    let prov = s.provenance_for(&near_sample(&s, k, 0.0));
    assert_eq!(s.would_answer(&pos, &z, &prov), Ok(()), "the model answers at the unreviewed sample's geometry");
}

/// §7.6 "Why this rule": ten siblings of one changed region (the analytic changed state, beyond the stored window).
/// The model without them reproduces none of them, so release-by-reproduction would never fire; in the candidate with
/// all of them each one's neighbours predict it (ratio ≤ 1): all are released and the refit proceeds with them as
/// ordinary fitted samples. The verdicts are in `migration.txt` (state reviewed) and the collection's codes.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn review_releases_siblings_its_neighbours_predict() {
    let dir = collection_root("code9-siblings");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(34);
    let nine = siblings(&mut s, &mut rng);
    let reproduced = nine.iter().filter(|&&i| s.reproduces_sample(&s.store.as_ref().unwrap().samples[i].clone())).count();
    assert_eq!(reproduced, 0, "the premise: the fit without them reproduces none");
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 1);
    let log = disk_log(&s);
    assert_eq!(log.state, MigrationState::Reviewed);
    for &i in &nine {
        let (ratio, released) = verdict(&s, i).expect("a verdict per sibling");
        println!("sibling #{i}: ratio {ratio:.3}");
        assert!(released && ratio <= 1.0, "#{i}: ratio {ratio}");
        assert_eq!(s.quarantine.get(i).copied().unwrap_or(Q_NONE), Q_NONE, "released 9 → 0");
        assert_eq!(s.collection().unwrap().quarantine[coll_index(&s, i)], Q_NONE, "the status record");
        assert!(s.in_standing_fit(i), "#{i} is fitted");
    }
    assert_eq!(s.report().unwrap().n_samples, 50, "the refit proceeds with them");
    assert!(s.review_kept.is_empty());
}

/// §7.6 step 4: a sample 50 mHa off its region's state stays at 9 — its verdict is "kept" with its ratio — and fences
/// (`fences()`, F4). Inside the stored window (r = 1.62, the trusted Morse fit) a geometry next to it is refused by the
/// quarantine fence (`would_answer` names `NearQuarantine`), and it is never converted to code 1, by this refit or the
/// next. A later review can still release one: a sample beyond the siblings (r = 3.4, where the model's variance would
/// refuse first, so the fence is seen in `near_quarantined`) in a second change of state ([`answer_tail`]) none of its
/// siblings share is kept; once fresh answers of that state surround it (the store changed), the next refit's review
/// releases it.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn review_keeps_an_inconsistent_sample_and_it_is_refused_near() {
    use sim::activity::TeachRule as R;
    // in the window
    let (_dir, mut s, w) = kept_in_window("code9-keep-in");
    let (ratio, _) = verdict(&s, w).unwrap();
    println!("the inconsistent sample in the window #{w}: ratio {ratio:.3}");
    assert!(ratio > 1.0);
    let near = near_sample(&s, w, 1e-3);
    let (pos, z) = pos_z(&near);
    assert_eq!(s.near_quarantined(&pos, &z), vec![w]);
    s.ensure_desc_cache();
    let prov = s.provenance_for(&near);
    assert_eq!(s.would_answer(&pos, &z, &prov), Err((R::NearQuarantine, None)), "refused by the quarantine fence");
    assert!(s.fence_refusal_text(&[w]).contains("re-solved for the new solver"));
    for _ in 0..2 {
        s.species_dirty = true;
        s.refit().unwrap();
        assert_eq!((s.quarantine[w], s.quarantined()), (Q_RESOLVED, 0), "never code 1");
        assert!(s.fences(w));
    }
    // beyond the siblings, then released by a later review
    let dir = collection_root("code9-keep-far");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(35);
    let nine = siblings(&mut s, &mut rng);
    let (_, odd) = answer_tail(3.4, &mut rng);
    let k = take_resolved(&mut s, odd);
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    let (ratio, released) = verdict(&s, k).expect("a verdict");
    println!("the inconsistent sample beyond the siblings #{k}: ratio {ratio:.3}; siblings {:?}", nine.iter().map(|&i| verdict(&s, i)).collect::<Vec<_>>());
    assert!(!released && ratio > 1.0, "kept: ratio {ratio}");
    assert_eq!(s.quarantine[k], Q_RESOLVED);
    assert!(s.fences(k) && s.review_kept.contains(&k));
    let near = near_sample(&s, k, 1e-3);
    let (pos, z) = pos_z(&near);
    assert!(s.near_quarantined(&pos, &z).contains(&k), "fenced");
    // fresh answers of its own state around it: the store changed, the next review releases it
    for r in [3.1, 3.2, 3.3, 3.36, 3.44, 3.5] {
        let (sc, smp) = answer_tail(r, &mut rng);
        s.add_sample_answer(smp, &sc, Some(0.0), Some(1)).unwrap();
    }
    s.refit().unwrap();
    let (ratio, released) = verdict(&s, k).expect("a verdict");
    println!("after its own state's answers: ratio {ratio:.3}");
    assert!(released && ratio <= 1.0, "released by the later review: ratio {ratio}");
    assert_eq!(s.quarantine.get(k).copied().unwrap_or(Q_NONE), Q_NONE);
    assert!(!s.fences(k) && !s.review_kept.contains(&k));
}

/// A reviewed-and-failed sample at the stored window (50 mHa off the Morse curve) with the teacher (that curve) next to
/// it: `(the failed 9, the provider)`, trusted, `k` kept and fencing.
fn kept_in_window(name: &str) -> (CollectionRoot, SurrogateKs, usize) {
    let dir = collection_root(name);
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(36);
    let (_, odd) = answer(1.62, OFFSET, false, &mut rng);
    let k = take_resolved(&mut s, odd);
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    assert!(s.trusted(), "{:?}", s.untrusted_reason());
    assert_eq!(verdict(&s, k).map(|v| v.1), Some(false), "kept");
    assert!(s.fences(k));
    (dir, s, k)
}

/// Controller ruling B (spec §7.6): a failed code 9 moves like a held sample — one fresh teacher answer next to it that
/// agrees with the model settles it through `admit`'s movable path: superseded here (the model does not reproduce its
/// label), no longer fencing.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_fresh_agreeing_answer_moves_a_failed_code9() {
    let (_dir, mut s, k) = kept_in_window("code9-moved");
    let sc = near_sample(&s, k, 1e-3);
    let (src, out, notes) = step(&mut s, &sc);
    assert_eq!(src, Source::Fallback, "{notes:?}");
    assert_eq!(out, Some(TeachOutcome::KeptReleased { released: 0, superseded: 1 }), "{notes:?}");
    assert_eq!(s.quarantine[k], Q_SUPERSEDED);
    assert!(!s.fences(k) && !s.review_kept.contains(&k));
    assert_eq!(s.collection().unwrap().quarantine[coll_index(&s, k)], Q_SUPERSEDED, "the status record");
}

/// F4: purge's lift votes never lift a failed code 9. Next to it and a forgotten sample, the first agreeing answer
/// moves the code 9 (ruling B, superseded) and casts no lift vote; the agreeing answers after it lift the forgotten
/// sample's fence (code 8) and nothing else: the code 9 is never `Q_FORGOTTEN_UNFENCED`, at any step.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn failed_code9_is_not_lifted_by_purge_votes() {
    let (_dir, mut s, k) = kept_in_window("code9-lift");
    // the stored sample nearest the failed one (r = 1.623), forgotten and refitted without
    let j = 17;
    let id = s.sample_id(j).unwrap();
    s.forget_sample(id).unwrap();
    s.refit().unwrap();
    assert!(s.fences(j) && s.fences(k), "both fence");
    let sc = near_sample(&s, j, 0.0);
    let (pos, z) = pos_z(&sc);
    let mut near = s.near_quarantined(&pos, &z);
    near.sort_unstable();
    assert_eq!(near, vec![j, k], "the premise: one region fenced by both");
    let mut outcomes = Vec::new();
    for (n, x) in [1e-3, 2e-3, 3e-3, 4e-3, 5e-3].into_iter().enumerate() {
        let sc = near_sample(&s, j, x);
        let (_, out, notes) = step(&mut s, &sc);
        println!("answer {n}: {out:?} {notes:?}");
        assert_ne!(s.quarantine[k], Q_FORGOTTEN_UNFENCED, "answer {n}");
        if n == 0 {
            assert_eq!(s.quarantine[k], Q_SUPERSEDED, "ruling B: the first agreeing answer moves it");
            assert!(s.lift_votes.is_empty(), "and casts no lift vote");
        }
        outcomes.push(out);
    }
    assert!(outcomes.iter().any(|o| matches!(o, Some(TeachOutcome::FenceLifted { lifted: 1 }))), "{outcomes:?}");
    assert_eq!((s.quarantine[j], s.quarantine[k]), (Q_FORGOTTEN_UNFENCED, Q_SUPERSEDED), "only the forgotten sample is lifted");
}

/// F3: a purge of forgotten samples that precede the reviewed ones renumbers the new folder; every review-log entry
/// names the same sample at its new index (in memory and in the rewritten `migration.txt`), and the kept one still
/// fences at its new store index.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn purge_after_review_remaps_the_review_log() {
    let dir = collection_root("code9-purge");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(37);
    siblings(&mut s, &mut rng);
    let (_, odd) = answer(3.4, OFFSET, true, &mut rng);
    let k = take_resolved(&mut s, odd);
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    for j in [2, 5] {
        let id = s.sample_id(j).unwrap();
        s.forget_sample(id).unwrap();
    }
    s.refit().unwrap();
    let before = disk_log(&s);
    assert_eq!(before.review.len(), SIBLINGS + 1);
    let named: Vec<(u64, f64, bool)> = before.review.iter().map(|&(c, r, rel)| (s.collection().unwrap().samples[c].sample.energy.to_bits(), r, rel)).collect();
    let kept_energy = s.store.as_ref().unwrap().samples[k].energy.to_bits();
    assert_eq!(s.purge_refusal(), None);
    s.purge_forgotten().unwrap();
    let after = disk_log(&s);
    assert_eq!(after.review.len(), before.review.len());
    for ((e, want), old) in after.review.iter().zip(&named).zip(&before.review) {
        assert_eq!((s.collection().unwrap().samples[e.0].sample.energy.to_bits(), e.1, e.2), *want, "entry {e:?} names the same sample");
        assert_eq!(e.0 + 2, old.0, "moved down by the two purged");
    }
    assert_eq!(after.review, s.migration.as_ref().unwrap().log.as_ref().unwrap().review, "memory and file agree");
    let k2 = k - 2;
    assert_eq!(s.store.as_ref().unwrap().samples[k2].energy.to_bits(), kept_energy);
    assert!(s.fences(k2) && s.review_kept.iter().copied().collect::<Vec<_>>() == vec![k2], "{:?}", s.review_kept);
}

/// Review Focus 3: a refit while the migration pass is still running reviews nothing — every code 9 stays.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn no_review_while_the_pass_is_running() {
    let dir = collection_root("code9-running");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(38);
    let nine = siblings(&mut s, &mut rng);
    set_migration(&mut s, MigrationState::Running);
    s.refit().unwrap();
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 0);
    assert!(nine.iter().all(|&i| s.quarantine[i] == Q_RESOLVED && s.collection().unwrap().quarantine[coll_index(&s, i)] == Q_RESOLVED));
    let log = disk_log(&s);
    assert_eq!((log.state, log.review.len()), (MigrationState::Running, 0));
}

/// §7.6: the review repeats at a later refit only while code-9 samples remain and the store changed since the last one.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn review_repeats_only_when_the_store_changed() {
    let dir = collection_root("code9-repeat");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(39);
    siblings(&mut s, &mut rng);
    let (_, odd) = answer(3.4, OFFSET, true, &mut rng);
    let k = take_resolved(&mut s, odd);
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 1);
    assert_eq!(s.quarantine[k], Q_RESOLVED, "one remains");
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 1, "the store did not change");
    let (sc, smp) = answer(1.5, 0.0, false, &mut rng);
    s.add_sample_answer(smp, &sc, Some(0.0), Some(1)).unwrap();
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 2, "a sample was added");
    // the note only when the verdicts change (Task 15 review M4): the second review kept the same one
    let notes = |s: &SurrogateKs| s.user_notes.iter().filter(|n| n.contains("re-solved for the new solver reviewed")).count();
    assert_eq!(notes(&s), 1, "{:?}", s.user_notes);
    // a policy change re-judges the verdicts (M3): the bounds a ratio is read against moved
    let p = FallbackPolicy { energy_std_per_atom: 1.5e-3, ..s.policy.clone() };
    s.set_policy(p);
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 3, "the policy's bounds changed");
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 3, "and nothing since");
}

/// Task 15 review M1 (spec §7.6, "Not judged"): a code-9 sample with stray atoms is not judged by the review — it waits
/// at 9, not fencing, listed in the log's waiting list, and the log stays "complete" while it remains; the note counts
/// it separately. Its siblings are reviewed as usual.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_stray_code9_waits_and_the_log_stays_complete() {
    let dir = collection_root("code9-stray");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(42);
    let nine = siblings(&mut s, &mut rng);
    // H₂ (the changed state) and an H 8 Bohr from its centre: beyond every cutoff, a stray atom
    let (_, mut smp) = answer(2.7, 0.0, true, &mut rng);
    let mid: [f64; 3] = std::array::from_fn(|a| 0.5 * (smp.pos[0][a] + smp.pos[1][a]));
    smp.pos.push([mid[0], mid[1] + 8.0, mid[2]]);
    smp.z.push(1);
    smp.energy += SpeciesMorse::atom(1);
    smp.forces.push([0.0; 3]);
    let k = take_resolved(&mut s, smp);
    assert!(s.resolved_stray(k), "the premise: stray atoms");
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    assert!(nine.iter().all(|&i| s.quarantine.get(i).copied().unwrap_or(Q_NONE) == Q_NONE), "the siblings are released");
    assert_eq!(s.quarantine[k], Q_RESOLVED, "not judged");
    assert!(!s.fences(k) && !s.review_kept.contains(&k));
    assert_eq!(verdict(&s, k), None, "no verdict");
    let log = disk_log(&s);
    assert_eq!((log.state, log.waiting.clone()), (MigrationState::Complete, vec![coll_index(&s, k)]), "complete while one waits");
    assert!(s.user_notes.iter().any(|n| n.contains("reviewed: 10 rejoined") && n.contains("1 waiting (stray atoms")), "{:?}", s.user_notes);
    // a later refit with nothing changed: no review, no note
    let before = s.user_notes.len();
    s.refit().unwrap();
    assert_eq!(s.reviews_run, 1);
    assert!(!s.user_notes[before..].iter().any(|n| n.contains("re-solved for the new solver reviewed")));
}

/// §7.6: a code-9 sample is not the user's — Forget and Restore refuse it (one or in a batch), nothing changed.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn forget_and_restore_refuse_code9() {
    let dir = collection_root("code9-forget");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(40);
    let (_, inside) = answer(1.62, OFFSET, false, &mut rng);
    let k = take_resolved(&mut s, inside);
    let id = s.sample_id(k).unwrap();
    let e = s.forget_sample(id).unwrap_err();
    assert!(e.contains("re-solved for the new solver") && e.contains("Forget does not apply"), "{e}");
    let e = s.restore_sample(id).unwrap_err();
    assert!(e.contains("re-solved for the new solver") && e.contains("Restore does not apply"), "{e}");
    let other = s.sample_id(4).unwrap();
    let e = s.forget_samples(&[other, id]).unwrap_err();
    assert!(e.contains("Forget does not apply"), "{e}");
    let e = s.restore_samples(&[id, other]).unwrap_err();
    assert!(e.contains("Restore does not apply"), "{e}");
    assert_eq!((s.quarantine[k], s.quarantine.get(4).copied().unwrap_or(Q_NONE)), (Q_RESOLVED, Q_NONE), "nothing changed");
    assert_eq!(s.collection().unwrap().quarantine[coll_index(&s, k)], Q_RESOLVED);
}

/// Ruling R13: a kept verdict is read back from `migration.txt` at attach — the reopened collection's code-9 sample
/// fences again; a released verdict whose sample is back at 0 sets nothing.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_kept_verdict_is_read_back_on_attach() {
    let dir = collection_root("code9-attach");
    let (mut s, _) = super::forget::small_collection(&dir);
    let mut rng = StdRng::seed_from_u64(41);
    let nine = siblings(&mut s, &mut rng);
    let (_, odd) = answer(3.4, OFFSET, true, &mut rng);
    let k = take_resolved(&mut s, odd);
    set_migration(&mut s, MigrationState::Complete);
    s.refit().unwrap();
    assert!(s.fences(k));
    let k_coll = coll_index(&s, k);
    drop(s);
    let (mut t, _) = pair_student();
    // the attach alone (no refit after it, Task 15 review M6): its read-back runs after the codes are loaded
    t.attach_refit_off = true;
    t.attach_collection(&dir, &h2(1.4)).unwrap();
    let k2 = t.shared.as_ref().unwrap().link.iter().position(|l| *l == Link::In(k_coll)).unwrap();
    assert_eq!(t.quarantine[k2], Q_RESOLVED);
    assert_eq!((t.review_kept.iter().copied().collect::<Vec<_>>(), t.reviews_run), (vec![k2], 0), "read back at attach, no review");
    assert!(t.fences(k2));
    // and the read-back again by hand
    t.review_kept.clear();
    t.find_migration();
    assert_eq!(t.review_kept.iter().copied().collect::<Vec<_>>(), vec![k2], "only the kept one, read back");
    assert!(t.fences(k2));
    assert!(nine.iter().all(|&i| !t.fences(i)));
}
