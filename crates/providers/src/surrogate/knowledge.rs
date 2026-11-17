//! The knowledge map's builder (M8a Task 9): [`sim::KnowledgeMap`] from the provider's fitted
//! samples and its per-key class verdict. What is counted, the first shell and the confidence
//! rule are `sim::knowledge`'s module doc; this file only reads the store, the quarantine codes
//! and the verdicts. Atomic units (Bohr); nothing per element beyond Z (rule 1).
//!
//! **The samples counted** are the members of the fit that stands, exactly those the class
//! verdict counts (`trust::key_column_counts` over `compute_class_loo_full`'s members): the
//! accumulated prefix of the store without any excluded sample — every quarantine code other than
//! `Q_NONE` (held, superseded, box-excluded, draft, and the provider-local `Q_LONE` and
//! `Q_REF_WAIT`) is held out of the fit and teaches nothing, so it is no knowledge — and without
//! the misfit set M (spec 2026-11-13 §4.3: fitted, but not reproduced, and left out of the class
//! rule's sample counts n_k — it stays in its leave-one-out RMS terms only, the Task 7 physics
//! review's C1).
//!
//! **Capped segments** (M8a Task 10, `surrogate` module doc "Masked scenes"): only a masked
//! sample's interior atoms are knowledge — an atom, its coordination, its angles and its keys are
//! counted when its forces are fitted, a pair when at least one of its two atoms is (that atom's
//! neighbourhood is the source scene's, so the pair is real); a cap or an edge atom adds nothing.
//!
//! **The pending samples** (2026-11-16, the live knowledge graph; `sim::knowledge`, "New samples,
//! waiting for the refit"): the store's tail past the accumulated prefix without a quarantine code
//! and not box-affected ([`SurrogateKs::is_pending`], one predicate for the map, the count and the
//! rows) — a forgotten, held, superseded, box-excluded, draft, lone or reference-waiting sample is
//! never pending, nor is a box-affected one, whose `Q_BOX` code is only set inside the next fit
//! (`exclude_box_samples`): it reads as held. Their atom pairs are binned by the fitted
//! samples' rule ([`each_pair`]) into each pair's `pending`, among the fitted species only, and
//! counted per environment key and class (`pending_keys`, `pending_classes`: the Model tab's
//! Teach next) by the per-key verdict's own rule — a sample once per key a trained atom of it
//! holds (`trust::trained_keys` over `env_keys` at the same r_cut, as `trust::key_column_counts`
//! counts the fitted ones; a key's basis columns are not needed to count it). They enter no other
//! count, so no class, confidence or first shell moves until the refit.
//!
//! **r_cut** is the model's cutoff ([`SurrogateKs::cutoff`]: its basis's — a shared collection's
//! reach or a per-scene basis's), the cutoff the verdict's keys are formed with.
//!
//! **The confidence** reuses the verdict rule 4b applies (`class_untrusted`, per key), behind the
//! model's trust rule (rule 4, [`SurrogateKs::trusted`]): a pair is `Trusted` only when the model
//! may answer at all and every key in which one of its species sees the other is trusted; it
//! recomputes no rule (the controller's ruling on Task 9, which supersedes the plan's "≥ Q′").
//!
//! Cost: two passes over the fitted samples, O(Σ N²) distances and O(Σ N·k²) angles (k the shell
//! neighbours), plus one verdict lookup per key, and one distance pass over the pending samples.
//! No prediction, no teacher.
use super::*;
use sim::knowledge::{angle_bin, first_shell, molecules, Composition, ElementKnowledge, KnowledgeMap, PairHistogram, PairKnowledge, PendingKey, RowStatus, SampleRow, StoredSample, ANGLE_BINS, BIN};
use sim::Confidence;
use std::collections::BTreeSet;

/// Every atom pair (a, b), a < b, of sample `s` that the map counts: closer than r_cut (`rc2` its
/// square) and at least one of the two atoms trained (module doc, "Capped segments"), with its
/// species pair (min Z, max Z) and its squared distance.
fn each_pair(s: &Sample, rc2: f64, mut f: impl FnMut(usize, usize, (u8, u8), f64)) {
    let n = s.pos.len();
    for a in 0..n {
        for b in a + 1..n {
            let r2: f64 = (0..3).map(|k| (s.pos[a][k] - s.pos[b][k]).powi(2)).sum();
            if r2 >= rc2 || !(s.trains_atom(a) || s.trains_atom(b)) {
                continue;
            }
            f(a, b, (s.z[a].min(s.z[b]), s.z[a].max(s.z[b])), r2);
        }
    }
}

impl SurrogateKs {
    /// The knowledge map of the fit that stands (module doc); `None` without a model or a store.
    pub fn knowledge_map(&self) -> Option<KnowledgeMap> {
        let (_, store) = (self.model.as_ref()?, self.store.as_ref()?);
        let r_cut = self.cutoff();
        let rc2 = r_cut * r_cut;
        let n_bins = ((r_cut / BIN).ceil() as usize).max(1);
        let fitted: Vec<&Sample> = store.samples[..self.accumulated.min(store.len())]
            .iter()
            .enumerate()
            .filter(|(i, s)| !self.excluded(*i) && !self.in_misfit(*i) && s.z.len() == s.pos.len())
            .map(|(_, s)| s)
            .collect();
        let d2 = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();

        // pass 1: atoms, coordination, pair samples and distances, and the keys each pair is met in
        let mut atoms: BTreeMap<u8, (usize, Vec<u32>)> = BTreeMap::new();
        let mut pairs: BTreeMap<(u8, u8), (usize, Vec<u32>)> = BTreeMap::new();
        let mut pair_keys: BTreeMap<(u8, u8), BTreeSet<surrogate::EnvKey>> = BTreeMap::new();
        for s in &fitted {
            let n = s.pos.len();
            let mut coord = vec![0usize; n];
            let mut seen: BTreeSet<(u8, u8)> = BTreeSet::new();
            each_pair(s, rc2, |a, b, key, r2| {
                coord[a] += 1;
                coord[b] += 1;
                seen.insert(key);
                let h = &mut pairs.entry(key).or_insert_with(|| (0, vec![0; n_bins])).1;
                h[((r2.sqrt() / BIN) as usize).min(n_bins - 1)] += 1;
            });
            for key in seen {
                pairs.get_mut(&key).expect("a seen pair has a histogram").0 += 1;
            }
            for (a, &c) in coord.iter().enumerate().filter(|(a, _)| s.trains_atom(*a)) {
                let e = atoms.entry(s.z[a]).or_insert_with(|| (0, Vec::new()));
                e.0 += 1;
                if e.1.len() <= c {
                    e.1.resize(c + 1, 0);
                }
                e.1[c] += 1;
            }
            // the keys in which Z sees each neighbour species (the verdict's own keys)
            for k in super::trust::trained_keys(s, env_keys(&s.pos, &s.z, rc2)).into_iter().flatten() {
                for &nb in &k.neighbours {
                    pair_keys.entry((k.z.min(nb), k.z.max(nb))).or_default().insert(k.clone());
                }
            }
        }

        // the first shell of every seen pair: the reference bonded distance the angles use
        let shells: BTreeMap<(u8, u8), f64> =
            pairs.iter().filter_map(|(&z, (_, counts))| first_shell(&PairHistogram { z, bin: BIN, counts: counts.clone() }).map(|r| (z, r))).collect();

        // pass 2: the angles between two shell neighbours of each atom
        let mut angles: BTreeMap<u8, Vec<u32>> = BTreeMap::new();
        for s in &fitted {
            let n = s.pos.len();
            for i in (0..n).filter(|&i| s.trains_atom(i)) {
                let shell: Vec<[f64; 3]> = (0..n)
                    .filter(|&j| j != i)
                    .filter_map(|j| {
                        let r2 = d2(s.pos[i], s.pos[j]);
                        let rs = shells.get(&(s.z[i].min(s.z[j]), s.z[i].max(s.z[j])))?;
                        (r2 < rc2 && r2 <= rs * rs && r2 > 0.0).then(|| {
                            let r = r2.sqrt();
                            std::array::from_fn(|k| (s.pos[j][k] - s.pos[i][k]) / r)
                        })
                    })
                    .collect();
                let h = angles.entry(s.z[i]).or_insert_with(|| vec![0; ANGLE_BINS]);
                for (a, u) in shell.iter().enumerate() {
                    for v in &shell[a + 1..] {
                        let c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
                        h[angle_bin(c.acos().to_degrees())] += 1;
                    }
                }
            }
        }

        // the pending samples: their atom pairs in the same bins, apart from every fitted count
        let pending: Vec<&Sample> = store.samples.iter().enumerate().filter(|&(i, _)| self.is_pending(i)).map(|(_, s)| s).collect();
        let mut pending_pairs: BTreeMap<(u8, u8), (usize, Vec<u32>)> = BTreeMap::new();
        for s in &pending {
            let mut seen: BTreeSet<(u8, u8)> = BTreeSet::new();
            each_pair(s, rc2, |_, _, key, r2| {
                seen.insert(key);
                let h = &mut pending_pairs.entry(key).or_insert_with(|| (0, vec![0; n_bins])).1;
                h[((r2.sqrt() / BIN) as usize).min(n_bins - 1)] += 1;
            });
            for key in seen {
                pending_pairs.get_mut(&key).expect("a seen pair has a histogram").0 += 1;
            }
        }
        // ... and per environment key and class, each sample once (module doc, "The pending samples")
        let mut pending_keys: BTreeMap<surrogate::EnvKey, usize> = BTreeMap::new();
        let mut pending_classes = [0usize; ENV_CLASSES.len()];
        for s in &pending {
            let keys: BTreeSet<surrogate::EnvKey> = super::trust::trained_keys(s, env_keys(&s.pos, &s.z, rc2)).into_iter().flatten().collect();
            let classes: BTreeSet<u8> = keys.iter().map(|k| k.class).collect();
            for c in classes.into_iter().filter_map(|c| (c as usize).checked_sub(1)).filter(|&c| c < ENV_CLASSES.len()) {
                pending_classes[c] += 1;
            }
            for k in keys {
                *pending_keys.entry(k).or_default() += 1;
            }
        }
        let pending_keys: Vec<PendingKey> = pending_keys
            .into_iter()
            .filter_map(|(k, samples)| Some(PendingKey { class: *ENV_CLASSES.get((k.class as usize).checked_sub(1)?)?, z: k.z, neighbours: k.neighbours, samples }))
            .collect();
        let pending_classes: Vec<(&'static str, usize)> = ENV_CLASSES.iter().zip(pending_classes).filter(|(_, n)| *n > 0).map(|(&c, n)| (c, n)).collect();

        // the confidence: the model's trust rule, then every key the pair is met in (module doc)
        let model_trusted = self.trusted();
        let key_trusted = |k: &surrogate::EnvKey| self.class_untrusted(&[Some(k.clone())]).is_none();
        let species: Vec<u8> = atoms.keys().copied().collect();
        let mut pair_list = Vec::new();
        for (ia, &za) in species.iter().enumerate() {
            for &zb in &species[ia..] {
                let z = (za, zb);
                let (samples, counts) = pairs.remove(&z).unwrap_or_else(|| (0, vec![0; n_bins]));
                let confidence = if samples == 0 {
                    Confidence::Unseen
                } else if model_trusted && pair_keys.get(&z).is_some_and(|ks| !ks.is_empty() && ks.iter().all(key_trusted)) {
                    Confidence::Trusted
                } else {
                    Confidence::Thin
                };
                let (pending_samples, pending) = pending_pairs.remove(&z).unwrap_or_else(|| (0, vec![0; n_bins]));
                pair_list.push(PairKnowledge { z, samples, hist: PairHistogram { z, bin: BIN, counts }, confidence, pending, pending_samples });
            }
        }
        // the molecules: every stored sample by composition (`sim::knowledge`, "Molecules")
        let rows = self.sample_rows();
        let stored: Vec<StoredSample> = rows
            .into_iter()
            .zip(&store.samples)
            .filter(|(_, s)| s.z.len() == s.pos.len())
            .map(|(row, s)| StoredSample { row, z: &s.z, pos: &s.pos, mask: s.force_mask.as_deref() })
            .collect();
        let molecules = molecules(&stored, &shells, r_cut);
        let elements = atoms
            .into_iter()
            .map(|(z, (atoms_seen, coordination))| ElementKnowledge { z, atoms_seen, coordination, angles: angles.remove(&z).unwrap_or_else(|| vec![0; ANGLE_BINS]) })
            .collect();
        Some(KnowledgeMap {
            elements,
            pairs: pair_list,
            slug: self.shared.as_ref().map_or_else(String::new, |s| s.coll.key.slug()),
            samples: fitted.len(),
            r_cut,
            molecules,
            pending: pending.len(),
            pending_keys,
            pending_classes,
        })
    }

    /// Whether stored sample `i` is pending (module doc, "The pending samples"): past the fitted
    /// prefix, no quarantine code, not box-affected, and one Z per position.
    pub(super) fn is_pending(&self, i: usize) -> bool {
        let Some(s) = self.store.as_ref().and_then(|st| st.samples.get(i)) else { return false };
        i >= self.accumulated && !self.excluded(i) && !self.is_box(i) && s.z.len() == s.pos.len()
    }

    /// The stored samples kept since the last refit and in the next fit's codes (module doc, "The
    /// pending samples"): of [`unfitted`](Self::unfitted), those [`is_pending`](Self::is_pending).
    pub fn pending_samples(&self) -> usize {
        let n = self.samples();
        (self.accumulated.min(n)..n).filter(|&i| self.is_pending(i)).count()
    }

    /// Every stored sample as the molecules and a Forget rule see it (`sim::knowledge`,
    /// "Molecules"), in store order: its composition, its scene (the shared
    /// collection's record; empty for a session-only sample or a model file's), its status — fitted
    /// as the map counts it (the accumulated prefix in the fit's codes, not in the misfit set),
    /// pending ([`is_pending`](Self::is_pending): kept since the last refit), waiting (in the fit's
    /// codes otherwise: the misfit set), forgotten, or held (a box-affected sample past the prefix
    /// too: the next fit excludes it) — and its leave-one-out ratio from
    /// the standing fit's cache when it is in the fit's codes (as `sample_suspects` reads it).
    pub(super) fn sample_rows(&self) -> Vec<SampleRow> {
        let Some(store) = self.store.as_ref() else { return Vec::new() };
        let cache = self.fresh_loo_ratios();
        let fitted_to = self.accumulated.min(store.len());
        store
            .samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let status = if self.is_forgotten(i) {
                    RowStatus::Forgotten
                } else if self.excluded(i) {
                    RowStatus::Held
                } else if i >= fitted_to {
                    if self.is_pending(i) { RowStatus::Pending } else { RowStatus::Held }
                } else if !self.in_misfit(i) {
                    RowStatus::Fitted
                } else {
                    RowStatus::Waiting
                };
                let ratio = if status.open() { cache.and_then(|c| c.rows.binary_search_by_key(&i, |x| x.0).ok().map(|k| c.rows[k].1)) } else { None };
                let scene = match self.shared.as_ref().and_then(|sh| sh.link.get(i).map(|l| (sh, l))) {
                    Some((sh, super::shared::Link::In(k))) => sh.coll.samples.get(*k).map_or_else(String::new, |c| c.scene.clone()),
                    _ => String::new(),
                };
                SampleRow { index: i, composition: Composition::of(&s.z), scene, status, ratio }
            })
            .collect()
    }
}
