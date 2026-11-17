//! The three-atom coverage map's builder (`surrogate::triples`, spec 2026-11-17 §6.1, §6.4, §7
//! B1–B2) on the knowledge map's H₂ + CH₄ fixtures (species-blind Morse: rule 1 is the model's).
use super::knowledge::{h2_and_ch4, random_ch4};
use super::*;
use crate::SCAN_CUTOFF_FRACTION;
use sim::knowledge::{Composition, PairSpan, TripleGrid, TripleType};
use std::collections::{BTreeMap, BTreeSet};

/// The triples of the stored samples `indices`, counted the plain way (its own loops, distances and
/// dot products, independent of `sim::knowledge::each_triple`): for each training centre c, each
/// unordered pair {j, k} of other atoms with 0 < r_cj, r_ck < r_cut, binned with
/// [`TripleGrid::cell`] (r₁ to the species-a neighbour; a folded type in either order). Per type:
/// (counts per cell, the samples holding at least one).
fn brute(s: &SurrogateKs, indices: &[usize]) -> BTreeMap<TripleType, (Vec<u32>, usize)> {
    let r_cut = s.cutoff();
    let grid = TripleGrid::new(r_cut);
    let store = s.store.as_ref().unwrap();
    let mut out: BTreeMap<TripleType, (Vec<u32>, usize)> = BTreeMap::new();
    for &i in indices {
        let smp = &store.samples[i];
        let n = smp.z.len();
        let mut held = BTreeSet::new();
        for c in (0..n).filter(|&c| smp.trains_atom(c)) {
            let v = |j: usize| -> [f64; 3] { [smp.pos[j][0] - smp.pos[c][0], smp.pos[j][1] - smp.pos[c][1], smp.pos[j][2] - smp.pos[c][2]] };
            let len = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for j in (0..n).filter(|&j| j != c) {
                for k in (j + 1..n).filter(|&k| k != c) {
                    let (dj, dk) = (v(j), v(k));
                    let (rj, rk) = (len(dj), len(dk));
                    if !(rj > 0.0 && rj < r_cut && rk > 0.0 && rk < r_cut) {
                        continue;
                    }
                    let t = TripleType::new(smp.z[c], smp.z[j], smp.z[k]);
                    let (r1, r2) = if smp.z[j] == t.a { (rj, rk) } else { (rk, rj) };
                    let u = (dj[0] * dk[0] + dj[1] * dk[1] + dj[2] * dk[2]) / (rj * rk);
                    let cell = grid.cell(t.folded(), r1, r2, u).expect("a distance under r_cut has a bin");
                    out.entry(t).or_insert_with(|| (vec![0; grid.cells()], 0)).0[cell] += 1;
                    held.insert(t);
                }
            }
        }
        for t in held {
            out.get_mut(&t).unwrap().1 += 1;
        }
    }
    out
}

/// Spec B1: the fitted triples are the knowledge map's fitted samples' (a capped segment's masked
/// centre adds none), the pending ones `is_pending`'s, counted apart (a forgotten and a
/// box-affected sample count nowhere); the compositions; the reach's spans are the knowledge
/// map's taught ranges; r_cut, the scan limit and the basis.
#[test]
fn the_triple_map_counts_fitted_triples_and_pending_apart() {
    let mut s = h2_and_ch4(120, 200);
    let fitted: Vec<usize> = (0..s.accumulated).filter(|&i| !s.excluded(i) && !s.in_misfit(i)).collect();
    let ch4_fitted: Vec<usize> = fitted.iter().copied().filter(|&i| s.store.as_ref().unwrap().samples[i].z.len() == 5).collect();
    // one fitted CH₄ whose C does not train (a capped segment's mask), set before the first build
    let masked = *ch4_fitted.last().unwrap();
    assert_eq!(s.store.as_ref().unwrap().samples[masked].z[0], 6, "premise: C first");
    s.store.as_mut().unwrap().samples[masked].force_mask = Some(vec![false, true, true, true, true]);
    // six CH₄ kept since the refit; one forgotten, one box-affected
    let mut rng = StdRng::seed_from_u64(0x3b1);
    for _ in 0..6 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    let n = s.samples();
    s.quarantine.resize(n, Q_NONE);
    s.quarantine[n - 1] = Q_FORGOTTEN;
    s.face.resize(n, f64::NAN);
    s.face[n - 2] = 1.0;
    let m = s.triple_map().expect("a fitted model");
    let pending: Vec<usize> = (0..n).filter(|&i| s.is_pending(i)).collect();
    assert_eq!(pending.len(), 4);
    let (bf, bp) = (brute(&s, &fitted), brute(&s, &pending));
    assert_eq!(m.types.iter().map(|k| k.t).collect::<BTreeSet<_>>(), bf.keys().chain(bp.keys()).copied().collect());
    let zero = || (vec![0u32; m.grid.cells()], 0usize);
    for k in &m.types {
        let (fc, fs) = bf.get(&k.t).cloned().unwrap_or_else(zero);
        let (pc, ps) = bp.get(&k.t).cloned().unwrap_or_else(zero);
        assert_eq!((&k.counts[..], k.samples), (&fc[..], fs), "fitted {:?}", k.t);
        assert_eq!((&k.pending[..], k.pending_samples), (&pc[..], ps), "pending {:?}", k.t);
    }
    let hch = m.get(TripleType::new(6, 1, 1)).unwrap();
    let held = ch4_fitted.len() - 1;
    assert_eq!((hch.samples, hch.triples), (held, 6 * held as u64), "six H–C–H per CH₄, the masked C adds none");
    assert_eq!(hch.compositions, vec![(Composition::of(&[6, 1, 1, 1, 1]), held)]);
    assert_eq!(hch.pending_samples, 4, "the forgotten and the box-affected count nowhere");
    // the reach's spans are the knowledge map's taught ranges, from the same fitted samples
    let km = s.knowledge_map().unwrap();
    for p in km.pairs.iter() {
        assert_eq!(m.span(p.z.0, p.z.1).copied(), PairSpan::of(&p.hist), "{:?}", p.z);
    }
    assert_eq!((m.r_cut, m.reach_limit), (s.cutoff(), SCAN_CUTOFF_FRACTION * s.cutoff()));
    assert_eq!(m.basis.body3, s.basis().unwrap().body3);
}

/// Spec B2: a pending-only rebuild reuses the fitted block (pointer-equal `Arc`) and recounts the
/// tail; a refit recounts; no model, no map.
#[test]
fn a_pending_only_rebuild_reuses_the_fitted_block() {
    let mut s = h2_and_ch4(120, 200);
    let t = TripleType::new(6, 1, 1);
    let m0 = s.triple_map().unwrap();
    assert!(Arc::ptr_eq(&m0.get(t).unwrap().counts, &s.triple_map().unwrap().get(t).unwrap().counts), "nothing changed: cached");
    let mut rng = StdRng::seed_from_u64(0x5e7);
    for _ in 0..4 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    let m1 = s.triple_map().unwrap();
    let (k0, k1) = (m0.get(t).unwrap(), m1.get(t).unwrap());
    assert!(Arc::ptr_eq(&k0.counts, &k1.counts), "pending-only: the fitted block is the cached one");
    assert_eq!((k1.pending_samples, k1.pending.iter().sum::<u32>()), (4, 24));
    s.refit().unwrap();
    let m2 = s.triple_map().unwrap();
    let k2 = m2.get(t).unwrap();
    assert!(!Arc::ptr_eq(&k0.counts, &k2.counts) && k2.triples > k0.triples, "a refit recounts");
    assert!(m2.types.iter().all(|k| k.pending_samples == 0));
    assert!(SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy::default()).triple_map().is_none(), "no model, no map");
}
