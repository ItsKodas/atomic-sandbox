//! The three-atom coverage map's builder (spec 2026-11-17 §6.1, §6.4): [`sim::knowledge::TripleMap`]
//! from the provider's fitted and pending samples. What is counted, the cells, the classes and the
//! reach are `sim::knowledge`'s (`knowledge/triples.rs`); this file only reads the store, the
//! quarantine codes, the misfit set and `is_pending`, and caches the fitted block per fit. Atomic
//! units (Bohr); nothing per element beyond Z (rule 1).
//!
//! **The samples counted** are the knowledge map's, by its own predicates (`knowledge.rs` module
//! doc): fitted = [`SurrogateKs::fitted_indices`] (the accumulated prefix, not excluded, not in the
//! misfit set); pending = [`SurrogateKs::is_pending`] (a forgotten, held, box-excluded or
//! box-affected sample is never pending). The two are tallied apart: pending triples enter no class.
//!
//! **Capped segments**: a triple counts only at a centre that trains (`Sample::trains_atom`), so a
//! cap or an edge atom is never a centre; it may still be a neighbour of an interior centre, whose
//! neighbourhood is the source scene's (as a pair counts when one of its atoms trains).
//!
//! **The reach's spans** are the knowledge map's taught ranges: the fitted pair histograms built by
//! the same [`each_pair`] and bins ([`BIN`] to r_cut), then [`PairSpan::of`]. They are kept for
//! every pair `each_pair` meets, a superset of the pairs the knowledge map lists (it lists only
//! pairs of species with a training atom), equal on those.
//!
//! **The cache** (`TripleCache`): the fitted block (the spans and the per-type blocks) is keyed by
//! (the model generation, r_cut's bits, the fitted store indices). A verdict re-judge that moves no
//! sample, or new pending samples, leave it equal, so a pending-only rebuild recounts the pending
//! tail alone and hands out the cached `Arc<[u32]>` blocks; a refit, a Forget, a Restore or a basis
//! (r_cut) change does not.
//!
//! Cost: Σ over training centres of C(k, 2) (k the neighbours within r_cut) per sample, plus one
//! O(N²) pair pass for the spans, on a cache miss; one triple pass over the pending tail each call.
use super::knowledge::{each_pair, pair_bin, pair_bins};
use super::*;
use crate::SCAN_CUTOFF_FRACTION;
use sim::knowledge::{PairHistogram, PairSpan, TripleBasis, TripleBlock, TripleMap, TripleTally, BIN};

/// The fitted block of the fit that stands (module doc, "The cache").
pub(super) struct TripleCache {
    /// (model generation, r_cut's bits, the fitted store indices).
    key: (u64, u64, Vec<usize>),
    spans: Vec<PairSpan>,
    blocks: Vec<TripleBlock>,
}

impl TripleCache {
    /// Counts the fitted samples `key.2` of `store` at `r_cut` (Bohr): the pair spans and the
    /// triple blocks.
    fn build(store: &SampleStore, key: (u64, u64, Vec<usize>), r_cut: f64) -> TripleCache {
        let (rc2, n_bins) = (r_cut * r_cut, pair_bins(r_cut));
        let mut hists: BTreeMap<(u8, u8), Vec<u32>> = BTreeMap::new();
        let mut tally = TripleTally::new(r_cut);
        for s in key.2.iter().map(|&i| &store.samples[i]) {
            each_pair(s, rc2, |_, _, z, r2| hists.entry(z).or_insert_with(|| vec![0; n_bins])[pair_bin(r2, n_bins)] += 1);
            tally.add(&s.z, &s.pos, |a| s.trains_atom(a));
        }
        let spans = hists.into_iter().filter_map(|(z, counts)| PairSpan::of(&PairHistogram { z, bin: BIN, counts })).collect();
        TripleCache { key, spans, blocks: tally.finish() }
    }
}

impl SurrogateKs {
    /// The three-atom coverage map of the fit that stands and the samples pending since (module
    /// doc); `None` without a model or a store.
    pub fn triple_map(&self) -> Option<TripleMap> {
        let (_, store) = (self.model.as_ref()?, self.store.as_ref()?);
        let r_cut = self.cutoff();
        let mut pending = TripleTally::new(r_cut);
        for (_, s) in store.samples.iter().enumerate().filter(|&(i, _)| self.is_pending(i)) {
            pending.add(&s.z, &s.pos, |a| s.trains_atom(a));
        }
        let pending = pending.finish();
        let basis = self.basis().map_or(TripleBasis { body3: false, body4: false, degree_cap: 0, l_max: 0 }, |b| TripleBasis {
            body3: b.body3,
            body4: b.body4,
            degree_cap: b.degree_cap,
            l_max: b.l_max,
        });
        let key = (self.model_generation(), r_cut.to_bits(), self.fitted_indices());
        let mut cache = self.triple_cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.as_ref().is_none_or(|c| c.key != key) {
            *cache = Some(TripleCache::build(store, key, r_cut));
        }
        let fitted = cache.as_ref().expect("built above");
        Some(TripleMap::assemble(r_cut, SCAN_CUTOFF_FRACTION * r_cut, basis, fitted.spans.clone(), &fitted.blocks, &pending))
    }
}
