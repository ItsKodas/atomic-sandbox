//! 2026-11-14: purging the forgotten samples (`purge.rs`). Exactly the forgotten samples go — from
//! the store and from the shared collection's file —, every other sample and code stays (renumbered
//! in order), every index the provider holds follows, the fit and its predictions are bitwise
//! unchanged, and a purge is refused (nothing changed) while it would be unsafe. Atomic units.

use super::*;

fn forget(s: &mut SurrogateKs, i: usize) {
    let id = s.sample_id(i).unwrap();
    s.forget_sample(id).unwrap();
}

/// The model's predictions — energy, forces and their stds — at a few dimers, as bit patterns.
fn prediction_bits(s: &SurrogateKs) -> Vec<u64> {
    let m = s.model.as_ref().expect("a model");
    let mut bits = Vec::new();
    for (k, r) in [1.25, 1.4, 1.63, 1.9, 2.15].into_iter().enumerate() {
        let sc = dimer(r, [0.6, 0.8, 0.0], [0.1 * k as f64, -0.05, 0.02], 1, 1);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let p = m.predict(&pos, &[1, 1], true).unwrap();
        bits.push(p.energy.to_bits());
        bits.extend(p.forces.iter().flatten().map(|x| x.to_bits()));
        bits.push(p.energy_std.unwrap().to_bits());
        bits.extend(p.force_std.unwrap().iter().chain(p.force_std_epistemic.as_ref().unwrap()).map(|x| x.to_bits()));
    }
    bits
}

/// Model-file mode: two forgotten samples are purged from the store; the held and superseded codes,
/// the samples and their order stay; the misfit set, the residuals' sample ids and the
/// worst-fitting list follow the renumbering (equal to what a fresh pass over the purged store
/// computes); the predictions are bitwise unchanged and no refit is due; an id taken before the
/// purge is refused; a save writes the file without them. Refused while the fit still holds a
/// forgotten sample, and with nothing forgotten.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_purge_removes_exactly_the_forgotten_samples_and_keeps_the_fit() {
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    assert_eq!(s.purge_refusal().as_deref(), Some("no forgotten samples"));
    forget(&mut s, 5);
    forget(&mut s, 17);
    assert_eq!(s.purge_refusal().as_deref(), Some(super::super::purge::PURGE_WAIT_REFIT), "the fit still holds them");
    assert!(s.purge_forgotten().is_err());
    assert_eq!(s.samples(), 40, "a refused purge changes nothing");
    // a superseded code (the trust system's) must survive, renumbered
    s.quarantine[30] = Q_SUPERSEDED;
    s.refit().unwrap();
    assert_eq!(s.quarantine[30], Q_SUPERSEDED, "the premise: a refit does not touch it");
    s.ensure_loo_ratios();
    assert_eq!(s.purge_refusal(), None);
    let pending = s.pending_refit();
    let before_bits = prediction_bits(&s);
    let before_samples: Vec<Sample> = s.store.as_ref().unwrap().samples.clone();
    let suspects = s.sample_suspects();
    let stale = s.sample_id(20).unwrap();
    let fit_n = s.report().unwrap().n_samples;
    let r = s.purge_forgotten().unwrap();
    assert_eq!((r.purged, r.from_collection, r.backup), (2, 0, None));
    assert!(s.user_notes.iter().any(|n| n.contains("purged 2 forgotten samples from the model") && n.contains("no longer fall back to DFT")), "{:?}", s.user_notes);
    // exactly the forgotten ones went, the rest in order
    let kept: Vec<Sample> = before_samples.iter().enumerate().filter(|(i, _)| ![5, 17].contains(i)).map(|(_, x)| x.clone()).collect();
    assert_eq!(s.store.as_ref().unwrap().samples, kept);
    assert_eq!((s.samples(), s.forgotten(), s.superseded()), (38, 0, 1));
    assert!(s.forgotten_samples().is_empty());
    assert_eq!(s.quarantine.get(28), Some(&Q_SUPERSEDED), "#30 is #28 now, still superseded");
    // the fit stands, bit for bit; nothing refits
    assert_eq!(prediction_bits(&s), before_bits);
    assert_eq!(s.report().unwrap().n_samples, fit_n);
    assert_eq!(s.pending_refit(), pending);
    assert_eq!(s.accumulated, 38);
    // every index followed: a fresh pass over the purged store gives the remapped state
    let (res, mis) = s.compute_residuals().unwrap();
    assert_eq!(s.training_residuals(), Some(res));
    assert_eq!(s.misfit(), Some(&mis));
    let after = s.sample_suspects();
    let shift = |i: usize| i - [5, 17].iter().filter(|&&g| g < i).count();
    assert_eq!(after.len(), suspects.len());
    for (a, b) in after.iter().zip(&suspects) {
        assert_eq!((a.id.index, a.id.key, a.loo_ratio), (shift(b.id.index), b.id.key, b.loo_ratio), "the worst-fitting rows follow their samples");
    }
    assert_eq!(s.desc_cache.len().min(38), s.desc_cache.len());
    // an id taken before the purge no longer names its sample
    assert!(s.forget_sample(stale).unwrap_err().contains("the sample list changed"));
    let fresh = s.sample_id(shift(20)).unwrap();
    assert_eq!(fresh.key, stale.key, "the same sample under its new index");
    // a save writes the file without them, and it loads to the same fit
    let path = std::env::temp_dir().join(format!("purge-{}.model", std::process::id()));
    s.save_sidecar(&path).unwrap();
    let file = surrogate::sidecar::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(file.samples.len(), 38);
    assert!(!file.quarantine.contains(&Q_FORGOTTEN));
    let mut back = SurrogateKs::new(Box::new(MorseDimer::h2()), FallbackPolicy::default()).with_teacher_tol(1e-6);
    back.install_sidecar(file).unwrap();
    assert_eq!(prediction_bits(&back), before_bits);
    assert_eq!(back.purge_refusal().as_deref(), Some("no forgotten samples"));
}

/// Collection mode: the forgotten samples leave the collection's file (backup kept) and the store;
/// the links, the collection's codes and the store's agree after it (and with a reopen from disk);
/// the fit is bitwise unchanged; later samples append after the rewritten file.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_purge_rewrites_the_shared_collection_and_keeps_every_link() {
    let dir = collection_root("purge-shared");
    let (mut s, key) = super::forget::small_collection(&dir);
    s.quarantine.resize(s.samples(), Q_NONE);
    s.quarantine[20] = Q_SUPERSEDED;
    s.sync_collection_status();
    forget(&mut s, 4);
    forget(&mut s, 10);
    s.refit().unwrap();
    assert_eq!(s.collection().unwrap().quarantine[20], Q_SUPERSEDED);
    let before_bits = prediction_bits(&s);
    let coll_before: Vec<surrogate::CollectionSample> = s.collection().unwrap().samples.clone();
    let r = s.purge_forgotten().unwrap();
    assert_eq!((r.purged, r.from_collection), (2, 2));
    let folder = dir.join(key.slug());
    assert_eq!(r.backup.as_deref(), Some(folder.join(surrogate::collection::PURGE_BACKUP).as_path()));
    let c = s.collection().unwrap();
    let kept: Vec<surrogate::CollectionSample> = coll_before.iter().enumerate().filter(|(i, _)| ![4, 10].contains(i)).map(|(_, x)| x.clone()).collect();
    assert_eq!(c.samples, kept);
    assert_eq!(c.quarantine[18], Q_SUPERSEDED, "#20 is #18 now, superseded in the collection too");
    assert!(!c.quarantine.contains(&Q_FORGOTTEN));
    let links = &s.shared.as_ref().unwrap().link;
    assert_eq!(links.len(), 38);
    for (i, l) in links.iter().enumerate() {
        let super::super::shared::Link::In(k) = l else { panic!("sample {i} is in the collection") };
        assert_eq!(c.samples[*k].sample, s.store.as_ref().unwrap().samples[i], "link {i}");
        assert_eq!(c.quarantine[*k], s.quarantine.get(i).copied().unwrap_or(Q_NONE), "code {i}");
    }
    assert_eq!(prediction_bits(&s), before_bits);
    // a new answer appends after the rewritten records
    let mut rng = StdRng::seed_from_u64(99);
    let sc = h2_at(1.47, &mut rng);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let (e, f) = SpeciesMorse::energy_and_forces(&pos, &[1, 1]);
    s.add_sample_answer(Sample { pos, z: vec![1, 1], energy: e, forces: f, tag: SampleTag::Scan, weight: 1.0, force_mask: None, use_energy: true }, &sc, Some(0.0), Some(1)).unwrap();
    assert_eq!(s.collection().unwrap().samples.len(), 39);
    let mem = s.collection().unwrap().samples.clone();
    drop(s);
    let (c, rep) = surrogate::Collection::open(&dir, &key).unwrap();
    assert_eq!((rep.samples, rep.dropped_tail), (39, 0));
    assert_eq!(c.samples, mem, "the disk holds what the handle held");
    assert_eq!(c.quarantine[18], Q_SUPERSEDED);
}

/// Another copy of the app holds the collection's lock (here a second handle, read-only): the
/// purge is refused with the user's words and nothing changes, on disk or in memory.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_purge_on_a_collection_another_copy_holds_is_refused() {
    let dir = collection_root("purge-locked");
    let (_owner, key) = super::forget::small_collection(&dir);
    let path = dir.join(key.slug()).join("samples.bin");
    let (ro, _) = surrogate::Collection::open(&dir, &key).unwrap();
    let (mut other, _) = pair_student();
    other.attach(ro).unwrap();
    // the read-only copy forgets for this session: the collection's record stays 0
    forget(&mut other, 4);
    other.refit().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(other.purge_refusal().as_deref(), Some(surrogate::collection::REFUSE_PURGE_LOCKED));
    assert_eq!(other.purge_forgotten().unwrap_err(), surrogate::collection::REFUSE_PURGE_LOCKED);
    assert_eq!((other.samples(), other.forgotten()), (40, 1), "nothing changed");
    assert_eq!(std::fs::read(&path).unwrap(), bytes, "nothing written");
}

/// A fence or lift vote next to `near`.
fn vote(near: Vec<usize>, gen: u64) -> super::super::quarantine::FenceVote {
    super::super::quarantine::FenceVote { near, n: 2, energy: -1.0, forces: vec![[0.0; 3]; 2], residual: None, gen }
}

fn case(index: Option<usize>) -> super::super::audit::FailedAudit {
    super::super::audit::FailedAudit { pos: vec![[0.0; 3], [1.4, 0.0, 0.0]], z: vec![1, 1], energy: -1.0, forces: vec![[0.0; 3]; 2], index, ionic: "none".into(), rejudges: 0, origin: Default::default(), pair: None }
}

/// Review Important 7: every index the provider holds follows the remap — fence and lift votes,
/// the kept-vote counts, pending votes, failed audits and audit fences, the class rule's worst
/// samples and blockers, the lone codes — and a reference into a purged sample is dropped (a
/// purged sample fences nothing). In collection mode the links follow the collection's own map,
/// session-only samples (`Link::Out`, `Link::Waiting`) interleaved with collection ones.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_remap_follows_every_index() {
    use super::super::shared::Link;
    let (mut s, _) = trained(1.2, 2.2, 10, FallbackPolicy::default());
    // purge 2 and 5 (old → new: 0 0, 1 1, 3 2, 4 3, 6 4, 7 5, 8 6, 9 7)
    let keep: Vec<bool> = (0..10).map(|i| i != 2 && i != 5).collect();
    s.fence_votes = vec![vote(vec![2, 3], 1), vote(vec![5], 1), vote(vec![9], 1)];
    s.lift_votes = vec![vote(vec![2], 1), vote(vec![4, 5, 6], 1)];
    s.fence_kept = [(2, 1), (3, 2), (8, 1)].into_iter().collect();
    s.vote_pending = vec![(4, vec![3, 5]), (5, vec![3]), (7, vec![2])];
    s.failed_audits = vec![case(Some(6)), case(Some(2)), case(None)];
    s.audit_fences = vec![case(Some(9))];
    s.class_worst = vec![Some((7, 3.0)), Some((5, 2.0)), None];
    s.class_blockers = vec![vec![(1, 4.0), (2, 3.0), (8, 2.0)], vec![]];
    s.lone_codes = [(5, Q_NONE), (6, Q_HELD)].into_iter().collect();
    s.remap_after_purge(&keep, None);
    let near = |v: &[super::super::quarantine::FenceVote]| v.iter().map(|v| v.near.clone()).collect::<Vec<_>>();
    assert_eq!(near(&s.fence_votes), vec![vec![2], vec![7]], "the vote only next to a purged sample is dropped");
    assert_eq!(near(&s.lift_votes), vec![vec![3, 4]]);
    assert_eq!(s.fence_kept, [(2, 2), (6, 1)].into_iter().collect());
    assert_eq!(s.vote_pending, vec![(3, vec![2])], "a purged kept sample, or one kept next to purged samples only, goes");
    assert_eq!(s.failed_audits.iter().map(|c| c.index).collect::<Vec<_>>(), vec![Some(4), None, None], "a purged case keeps its labels, not its index");
    assert_eq!(s.audit_fences[0].index, Some(7));
    assert_eq!(s.class_worst, vec![Some((5, 3.0)), None, None]);
    assert_eq!(s.class_blockers, vec![vec![(1, 4.0), (6, 2.0)], vec![]]);
    assert_eq!(s.lone_codes, [(4, Q_HELD)].into_iter().collect());
    assert_eq!(s.samples(), 8);
    // collection mode: session-only samples interleaved with the collection's
    let dir = collection_root("purge-remap-links");
    let (mut c, _) = super::forget::small_collection(&dir);
    {
        let sh = c.shared.as_mut().unwrap();
        // store 0..40 ↔ collection 0..40; make 3 and 7 session-only, 11 a waiting draft guess (a remap test: the
        // collection's own records are not touched)
        sh.link[3] = Link::Out;
        sh.link[7] = Link::Waiting { grid_n: 33, multiplicity: None };
        sh.link[11] = Link::Out;
        sh.pass_start = Some(20);
    }
    // purge store 4 (collection 4) and 7 (session-only): the collection's own map drops only its 4
    let keep: Vec<bool> = (0..40).map(|i| i != 4 && i != 7).collect();
    let cmap: Vec<Option<usize>> = (0..40).map(|k| (k != 4).then(|| if k < 4 { k } else { k - 1 })).collect();
    let raw: Vec<f64> = c.shared.as_ref().unwrap().raw.clone();
    c.remap_after_purge(&keep, Some(&cmap));
    let sh = c.shared.as_ref().unwrap();
    assert_eq!(sh.link.len(), 38);
    assert_eq!(sh.link[3], Link::Out);
    assert_eq!(sh.link[4], Link::In(4), "store 5 (collection 5) is now store 4 → collection 4");
    assert_eq!(sh.link[9], Link::Out, "store 11 is now 9");
    assert_eq!(sh.link[37], Link::In(38), "store 39 (collection 39) is now 37 → 38");
    assert_eq!(sh.raw, raw.iter().enumerate().filter(|(i, _)| *i != 4 && *i != 7).map(|(_, x)| *x).collect::<Vec<_>>());
    assert_eq!(sh.pass_start, Some(19), "the collection's count at the pass start: one of its first 20 went");
}

/// Review Important 3: a collection failure before the final rename (injected) leaves the provider
/// exactly as it was; a failure after it (the reopen) is not an `Err` — the purge stands on disk and
/// in the collection, the provider remaps, and the report carries the warning.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn a_collection_failure_never_leaves_the_provider_half_purged() {
    let dir = collection_root("purge-inject");
    let (mut s, _) = super::forget::small_collection(&dir);
    forget(&mut s, 4);
    forget(&mut s, 10);
    s.refit().unwrap();
    let (samples, codes, links) = (s.store.as_ref().unwrap().samples.clone(), s.quarantine.clone(), s.shared.as_ref().unwrap().link.clone());
    s.shared.as_mut().unwrap().coll.inject_purge_failure(3);
    let e = s.purge_forgotten().unwrap_err();
    assert!(e.contains("injected") && e.contains("unchanged"), "{e}");
    assert_eq!((&s.store.as_ref().unwrap().samples, &s.quarantine, &s.shared.as_ref().unwrap().link), (&samples, &codes, &links), "nothing changed");
    assert_eq!(s.collection().unwrap().samples.len(), 40);
    s.shared.as_mut().unwrap().coll.inject_purge_failure(6);
    let r = s.purge_forgotten().unwrap();
    assert_eq!((r.purged, r.from_collection), (2, 2));
    assert!(r.warning.as_deref().is_some_and(|w| w.contains("read-only until reopened")), "{r:?}");
    assert!(s.collection().unwrap().read_only);
    // disk, collection and provider agree
    let c = s.collection().unwrap();
    assert_eq!(c.samples.len(), 38);
    for (i, l) in s.shared.as_ref().unwrap().link.iter().enumerate() {
        let super::super::shared::Link::In(k) = l else { panic!("sample {i}") };
        assert_eq!(c.samples[*k].sample, s.store.as_ref().unwrap().samples[i], "link {i}");
    }
    assert!(s.user_notes.iter().any(|n| n.contains("read-only until reopened")), "{:?}", s.user_notes);
}

/// The purged samples' fences go with them (the user's decision "purge lifts, warn"): a geometry
/// the forgotten sample fenced is no longer refused by the fence, and the 10 % cap no longer
/// counts it.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn purged_samples_no_longer_fence_or_count() {
    let (mut s, _) = trained(1.2, 2.2, 40, FallbackPolicy::default());
    forget(&mut s, 17);
    s.refit().unwrap();
    let pos = s.store.as_ref().unwrap().samples[17].pos.clone();
    assert_eq!(s.near_quarantined(&pos, &[1, 1]), vec![17]);
    assert_eq!(s.quarantined() + s.forgotten(), 1);
    s.purge_forgotten().unwrap();
    assert!(s.near_quarantined(&pos, &[1, 1]).is_empty(), "no fence any more");
    assert_eq!(s.quarantined() + s.forgotten(), 0, "not counted against the cap");
    assert!(s.strict_verdict.is_none(), "the readiness verdict is recomputed");
}
