//! The knowledge map's builder (M8a Task 9): [`sim::KnowledgeMap`] from the provider's fitted
//! samples and its per-key class verdict. What is counted, the first shell and the confidence
//! rule are `sim::knowledge`'s module doc; this file only reads the store, the quarantine codes
//! and the verdicts. Atomic units (Bohr); nothing per element beyond Z (rule 1).
//!
//! **The samples counted** are the members of the fit that stands, exactly those the class
//! verdict counts (`trust::key_column_counts` over `compute_class_loo_full`'s members): the
//! accumulated prefix of the store without any excluded sample — every quarantine code other than
//! `Q_NONE` (held, superseded, box-excluded, draft, and the provider-local `Q_LONE` and
//! `Q_REF_WAIT`) is held out of the fit and teaches nothing, so it is no knowledge.
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
//! neighbours), plus one verdict lookup per key. No prediction, no teacher.
use super::*;
use sim::knowledge::{angle_bin, first_shell, ElementKnowledge, KnowledgeMap, PairHistogram, PairKnowledge, ANGLE_BINS, BIN};
use sim::Confidence;
use std::collections::BTreeSet;

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
            .filter(|(i, s)| !self.excluded(*i) && s.z.len() == s.pos.len())
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
            for a in 0..n {
                for b in a + 1..n {
                    let r2 = d2(s.pos[a], s.pos[b]);
                    if r2 >= rc2 {
                        continue;
                    }
                    coord[a] += 1;
                    coord[b] += 1;
                    let key = (s.z[a].min(s.z[b]), s.z[a].max(s.z[b]));
                    seen.insert(key);
                    let h = &mut pairs.entry(key).or_insert_with(|| (0, vec![0; n_bins])).1;
                    h[((r2.sqrt() / BIN) as usize).min(n_bins - 1)] += 1;
                }
            }
            for key in seen {
                pairs.get_mut(&key).expect("a seen pair has a histogram").0 += 1;
            }
            for (a, &c) in coord.iter().enumerate() {
                let e = atoms.entry(s.z[a]).or_insert_with(|| (0, Vec::new()));
                e.0 += 1;
                if e.1.len() <= c {
                    e.1.resize(c + 1, 0);
                }
                e.1[c] += 1;
            }
            // the keys in which Z sees each neighbour species (the verdict's own keys)
            for k in env_keys(&s.pos, &s.z, rc2).into_iter().flatten() {
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
            for i in 0..n {
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
                pair_list.push(PairKnowledge { z, samples, hist: PairHistogram { z, bin: BIN, counts }, confidence });
            }
        }
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
        })
    }
}
