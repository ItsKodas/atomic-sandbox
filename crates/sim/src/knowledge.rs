//! The knowledge map's data (M8a Task 9; spec §4.1, the shared model): what the model's fitted
//! samples cover — per element its atoms, its neighbour counts and its bond angles, per species
//! pair the distances seen and whether the model answers there — and a per-atom coverage of a
//! geometry against it ([`atom_coverage`], the 3D overlay's tint). Plain data: the provider
//! builds it (`providers::surrogate`, `knowledge.rs`), the UI draws it. It decides nothing.
//!
//! Atomic units: distances in Bohr, angles in degrees (a display histogram), counts unitless.
//! Nothing per element beyond Z (rule 1): every distance below is measured from the samples.
//!
//! # What is counted (over the **fitted** samples only: held-out ones teach nothing)
//!
//! - a pair (a, b) is *seen* in a sample when two of its atoms of species a and b are closer than
//!   the model's cutoff r_cut; [`PairKnowledge::samples`] counts the samples, the histogram
//!   ([`PairHistogram`], bins of [`BIN`] Bohr from 0) every such atom pair;
//! - an atom's coordination is its neighbour count within r_cut — the model's own notion of a
//!   neighbour, not a chemical bond ([`ElementKnowledge::coordination`]);
//! - its bond angles are the angles j–i–k between two *shell* neighbours j, k of atom i: those
//!   within [`first_shell`] of the pair's histogram — the first peak's extent, measured, never a
//!   table of bond lengths ([`ElementKnowledge::angles`], [`ANGLE_BIN_DEG`] bins over 0–180°).
//!
//! # The first shell (the "reference bonded distance")
//!
//! The histogram smoothed with the [1, 2, 1]/4 kernel (zero beyond its ends) s_k; from the first
//! bin with s_k > 0, up while s_{k+1} > s_k (the first peak), then on while 0 < s_{k+1} ≤ s_k;
//! the bin m where that stops is the first minimum after the first peak, the conventional end of a
//! radial distribution's first coordination shell, and r_shell = (m + ½)·bin — its centre (when
//! s_m = 0 the raw counts of bins m − 1 … m + 1 are all 0, so any cut in bin m is the same cut). The
//! smoothing keeps one-bin noise inside a sampled peak from splitting it. **A limit, documented:**
//! for a pair that is never bonded (H–H in methane, 3.4 Bohr) the first peak is its nearest
//! non-bonded distance, and those neighbours enter the angle histogram of the central species; the
//! map has no density to tell a bond from a contact (the bond detector's density graph is not in
//! the samples).
//!
//! # Confidence (M8a user-fixes 3; supersedes the plan's "≥ Q′ samples" wording)
//!
//! `Unseen`: no fitted sample holds the pair within r_cut. `Trusted`: it has samples, the model's
//! own trust rule (rule 4) holds, and **every environment key in which a sees b or b sees a** that
//! the fitted samples hold — a key is (class pair/many-body, central Z, the sorted set of its
//! neighbours' species within r_cut; `surrogate::EnvKey`) — is trusted by the provider's
//! calibrated per-key verdict (rule 4b), the one `evaluate` applies, which since user-fix 3 holds
//! each key to its own sample floor (its fitted samples holding an atom of exactly that key,
//! against the basis columns its atoms touch) besides the measured errors. The map reuses that
//! verdict; it recomputes no rule. `Thin`: samples, but not all of that. The plan's "≥ Q′ (the
//! fit's `q_prime`) samples **and** the class verdict" predates user-fix 3, which replaced the
//! global Q′ floor with the per-key one: a pair is now held to the floors of the keys it appears
//! in, not to the whole basis's Q′. A consequence: a pair seen in a short key reads Thin even
//! where other keys holding it are well sampled (H–H with 120 H₃ trimers and 3 NH₃: NH₃'s H{H, N}
//! is short, so H–H is Thin) — the model refuses that environment, and the map says the pair is
//! not known everywhere it was met.
//!
//! # The per-atom coverage
//!
//! [`atom_coverage`]: 0 for a species the map does not hold; 1 for an atom with no neighbour within
//! r_cut of a held species (the isolated-atom reference answers it); otherwise the minimum over
//! its neighbours j of w(confidence of (Z_i, Z_j)) · o_ij, with w = 1 / ½ / 0 for Trusted / Thin /
//! Unseen and o_ij = 1 − exp(−n/[`COVERAGE_COUNT`]), n the atom pairs of the histogram in the bins
//! around d_ij (its own and the two beside it) — 0 at a distance never sampled. The minimum: an
//! atom is as well known as its least-known neighbour. A display mapping, not a probability.
//!
//! # Molecules (2026-11-16: the Knowledge panel's "Molecules trained" and the Model tab's "Forget
//! samples…")
//!
//! A stored sample's **composition** ([`Composition`]) is the multiset of its Z, shown as a
//! formula: an H₂ beside a far H is "H₃" like a bonded H₃ trimer, and a PH₃ beside an H₂ is "PH₅".
//! Splitting a sample into molecules was weighed and not taken: the samples hold no density to tell
//! a bond from a contact, and the measured first shell ([`first_shell`]) is noise-sensitive (a
//! 40-dimer H₂ scan with five samples taken out split its stretched dimers into "H + H") and moves
//! at every refit, so a composition the user confirmed could hold other samples a moment later. The
//! multiset is exact and never moves. [`molecules`] counts every stored sample by composition
//! ([`MoleculeKnowledge`]): fitted, waiting for a refit (in the fit's codes, not yet fitted), held
//! out, forgotten, the scenes they came from, and over the fitted ones the pair-distance ranges and
//! the shell angles per central element (the element card's rule, restricted to the composition).
//! [`ForgetRule`] picks samples to forget by species, composition, scene or leave-one-out ratio;
//! [`select`] resolves it per sample (the worker's side) and [`ForgetPreview::of`] counts it from
//! the molecules (the confirmation's side) — the same rule over the same rows, so the two agree
//! while the map is current. Only a sample in the fit's codes (fitted or waiting) can be forgotten.
//!
//! # New samples, waiting for the refit (2026-11-16: the live knowledge graph)
//!
//! A sample kept since the last refit and in the next fit's codes ([`RowStatus::Pending`]: past
//! the fitted prefix, no quarantine code — so never a forgotten, held, superseded, box-excluded,
//! draft, lone or reference-waiting one) is **pending**: [`KnowledgeMap::pending`] counts them,
//! [`PairKnowledge::pending`] bins their atom pairs exactly as `hist` bins the fitted ones (the
//! same geometry rule: within r_cut, a pair counted when one of its atoms is trained) and
//! [`PairKnowledge::pending_samples`] counts the pending samples holding the pair. Their energies
//! and forces are never read. **Pending samples decide nothing:** a pair's confidence, its sample
//! count, its histogram (and so a bin's taught/thin class), the elements, the first shells and the
//! molecules' geometry statistics stay the fitted samples' alone; the UI draws the pending counts
//! as a separate "new, waiting for refit" segment, which the refit turns into fitted counts. Only
//! the pairs of the fitted species carry them (a species no fitted sample holds has no node or
//! pair yet; its samples show in the molecules' counts until the refit).
//!
//! The Model tab's **Teach next** gaps read them too (2026-11-16, the same design):
//! [`KnowledgeMap::pending_keys`] counts the pending samples holding a trained atom of each
//! environment key (class, central Z, its neighbours' species within r_cut — the key the per-key
//! verdict counts its fitted samples by) and [`KnowledgeMap::pending_classes`] those holding one
//! of each class, each sample once per key or class, as the fitted counts are; no verdict reads
//! them.

/// Pair-distance histogram bin (Bohr): the training views' (`TrainingViews::pairs`).
pub const BIN: f64 = 0.1;

/// Bond-angle histogram bin (degrees): 36 bins over 0–180°.
pub const ANGLE_BIN_DEG: f64 = 5.0;

/// Number of angle bins.
pub const ANGLE_BINS: usize = 36;

/// The atom-pair count at which a distance reads 1 − 1/e ≈ 0.63 known in [`atom_coverage`].
pub const COVERAGE_COUNT: f64 = 4.0;

pub use crate::activity::PairHistogram;
use std::collections::BTreeMap;

/// What the model's fitted samples cover (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeMap {
    /// Every species of the fitted samples, ascending Z.
    pub elements: Vec<ElementKnowledge>,
    /// Every pair of `elements` (z.0 ≤ z.1), seen or not, in (z.0, z.1) order.
    pub pairs: Vec<PairKnowledge>,
    /// The shared collection's slug; empty for a per-scene model.
    pub slug: String,
    /// The fitted samples counted.
    pub samples: usize,
    /// The model's cutoff (Bohr): what "neighbour" means here (an addition to the plan's fields:
    /// [`atom_coverage`] needs it, and the histograms end there).
    pub r_cut: f64,
    /// Every composition of the stored samples (module doc, "Molecules"), most fitted first.
    pub molecules: Vec<MoleculeKnowledge>,
    /// The stored samples kept since the last refit and in the next fit's codes (module doc, "New
    /// samples, waiting for the refit"); not in `samples`.
    pub pending: usize,
    /// Per environment key, the pending samples holding a trained atom of it (module doc), in the
    /// key's order.
    pub pending_keys: Vec<PendingKey>,
    /// Per environment class ("pair", "many-body"), the pending samples holding a trained atom of it.
    pub pending_classes: Vec<(&'static str, usize)>,
}

/// The pending samples of one environment key (module doc, "New samples, waiting for the refit"):
/// its class's name, central Z and neighbour species (ascending, distinct), as
/// `sim::SpeciesVerdict` names a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingKey {
    pub class: &'static str,
    pub z: u8,
    pub neighbours: Vec<u8>,
    pub samples: usize,
}

/// One element's coverage (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct ElementKnowledge {
    pub z: u8,
    /// Its atoms in the fitted samples.
    pub atoms_seen: usize,
    /// Histogram: index = neighbour count within r_cut.
    pub coordination: Vec<u32>,
    /// Histogram of shell-neighbour angles, [`ANGLE_BINS`] bins of [`ANGLE_BIN_DEG`]° from 0.
    pub angles: Vec<u32>,
}

/// One species pair's coverage (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct PairKnowledge {
    /// (min Z, max Z).
    pub z: (u8, u8),
    /// Fitted samples holding the pair within r_cut.
    pub samples: usize,
    /// Its atom-pair distances < r_cut, bins of [`BIN`] Bohr from 0.
    pub hist: PairHistogram,
    pub confidence: Confidence,
    /// The pending samples' atom-pair distances, `hist`'s bins (module doc, "New samples, waiting
    /// for the refit"): never in `hist`, the confidence or a bin's class.
    pub pending: Vec<u32>,
    /// The pending samples holding the pair within r_cut.
    pub pending_samples: usize,
}

/// Whether the model answers a pair's environments (module doc, "Confidence").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    Trusted,
    Thin,
    Unseen,
}

impl KnowledgeMap {
    /// The pair (a, b) in either order.
    pub fn pair(&self, a: u8, b: u8) -> Option<&PairKnowledge> {
        let key = (a.min(b), a.max(b));
        self.pairs.iter().find(|p| p.z == key)
    }

    /// The element `z`.
    pub fn element(&self, z: u8) -> Option<&ElementKnowledge> {
        self.elements.iter().find(|e| e.z == z)
    }

    /// The composition `c`'s entry.
    pub fn molecule(&self, c: &Composition) -> Option<&MoleculeKnowledge> {
        self.molecules.iter().find(|m| &m.composition == c)
    }

    /// The pending samples of the key (class, z, neighbours); 0 for a key none holds.
    pub fn pending_key(&self, class: &str, z: u8, neighbours: &[u8]) -> usize {
        self.pending_keys.iter().find(|k| k.class == class && k.z == z && k.neighbours == neighbours).map_or(0, |k| k.samples)
    }

    /// The pending samples of the class; 0 for a class none holds.
    pub fn pending_class(&self, class: &str) -> usize {
        self.pending_classes.iter().find(|c| c.0 == class).map_or(0, |c| c.1)
    }
}

/// A sample's atoms as the multiset of their Z, ascending (module doc, "Molecules").
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Composition(pub Vec<u8>);

impl Composition {
    /// The composition of atoms `z`, in any order.
    pub fn of(z: &[u8]) -> Composition {
        let mut z = z.to_vec();
        z.sort_unstable();
        Composition(z)
    }

    /// Its species, ascending, each once.
    pub fn species(&self) -> Vec<u8> {
        let mut s = self.0.clone();
        s.dedup();
        s
    }

    /// It holds an atom of `z`.
    pub fn contains(&self, z: u8) -> bool {
        self.0.contains(&z)
    }

    /// Its atoms.
    pub fn atoms(&self) -> usize {
        self.0.len()
    }
}

fn d2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// Where a stored sample stands (module doc, "Molecules").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowStatus {
    /// A member of the fit that stands.
    Fitted,
    /// In the fit's codes (it may be forgotten) but not fitted: in the fit and not reproduced (the
    /// misfit set, out of the class counts).
    Waiting,
    /// In the fit's codes, kept since the last refit: new, waiting for the refit (module doc, "New
    /// samples, waiting for the refit").
    Pending,
    /// Held out of the fit: by the trust recovery, superseded, box-affected, a draft guess, waiting
    /// for its box's reference, lone atoms.
    Held,
    /// Forgotten by the user.
    Forgotten,
}

impl RowStatus {
    /// In the fit's codes: what a Forget can take.
    pub fn open(self) -> bool {
        matches!(self, RowStatus::Fitted | RowStatus::Waiting | RowStatus::Pending)
    }
}

/// One stored sample for [`select`] and [`molecules`]: its index in the store, composition, the
/// scene it came from (empty: none recorded — a session-only sample or a model file's), status, and
/// its leave-one-out ratio in the fit that stands (fitted samples only).
#[derive(Clone, Debug, PartialEq)]
pub struct SampleRow {
    pub index: usize,
    pub composition: Composition,
    pub scene: String,
    pub status: RowStatus,
    pub ratio: Option<f64>,
}

/// A stored sample with its geometry, for [`molecules`]: `mask` is its force mask (a capped
/// segment's: only the atoms whose forces are fitted are knowledge, as the element card counts).
#[derive(Clone, Debug)]
pub struct StoredSample<'a> {
    pub row: SampleRow,
    pub z: &'a [u8],
    pub pos: &'a [[f64; 3]],
    pub mask: Option<&'a [bool]>,
}

/// The samples of one scene within a composition: in the fit's codes (`open`), held, forgotten.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SceneCount {
    /// The scene's name; empty: none recorded.
    pub name: String,
    pub open: usize,
    pub held: usize,
    pub forgotten: usize,
}

/// The distances of one species pair within a composition's fitted samples (< r_cut): the
/// shortest, the longest (Bohr) and how many atom pairs.
#[derive(Clone, Debug, PartialEq)]
pub struct PairRange {
    pub z: (u8, u8),
    pub min: f64,
    pub max: f64,
    pub count: usize,
}

/// One composition of the stored samples (module doc, "Molecules").
#[derive(Clone, Debug, PartialEq)]
pub struct MoleculeKnowledge {
    pub composition: Composition,
    /// Members of the fit that stands.
    pub fitted: usize,
    /// In the fit's codes, not fitted ([`RowStatus::Waiting`] and [`RowStatus::Pending`]).
    pub waiting: usize,
    /// Of `waiting`, the samples kept since the last refit ([`RowStatus::Pending`]): "+k new".
    pub pending: usize,
    /// Held out ([`RowStatus::Held`]).
    pub held: usize,
    pub forgotten: usize,
    /// By scene, by name.
    pub scenes: Vec<SceneCount>,
    /// The leave-one-out ratios in the fit that stands of its samples in the fit's codes that have
    /// one (∞ allowed), ascending: what "Forget all above x×" counts.
    pub ratios: Vec<f64>,
    /// Over its fitted samples: every species pair within r_cut, in (min Z, max Z) order.
    pub pairs: Vec<PairRange>,
    /// Over its fitted samples: per central element (ascending Z), the shell-neighbour angle
    /// histogram ([`ANGLE_BINS`] bins of [`ANGLE_BIN_DEG`]°), the element card's rule.
    pub angles: Vec<(u8, Vec<u32>)>,
}

impl MoleculeKnowledge {
    /// In the fit's codes: what a Forget can take.
    pub fn open(&self) -> usize {
        self.fitted + self.waiting
    }

    /// Every sample of it.
    pub fn total(&self) -> usize {
        self.fitted + self.waiting + self.held + self.forgotten
    }
}

/// Every composition of `samples` (module doc, "Molecules"), most fitted first (then most samples,
/// then the composition): the counts by status and scene over all of them; the leave-one-out
/// ratios of those in the fit's codes; the pair ranges and shell angles over the fitted ones
/// (`shells` the first shells of the map's pairs, `r_cut` the model's).
pub fn molecules(samples: &[StoredSample<'_>], shells: &BTreeMap<(u8, u8), f64>, r_cut: f64) -> Vec<MoleculeKnowledge> {
    let rc2 = r_cut * r_cut;
    let mut by: BTreeMap<Composition, MoleculeKnowledge> = BTreeMap::new();
    let mut ranges: BTreeMap<Composition, BTreeMap<(u8, u8), (f64, f64, usize)>> = BTreeMap::new();
    let mut angles: BTreeMap<Composition, BTreeMap<u8, Vec<u32>>> = BTreeMap::new();
    for s in samples {
        let c = &s.row.composition;
        let m = by.entry(c.clone()).or_insert_with(|| MoleculeKnowledge {
            composition: c.clone(),
            fitted: 0,
            waiting: 0,
            pending: 0,
            held: 0,
            forgotten: 0,
            scenes: Vec::new(),
            ratios: Vec::new(),
            pairs: Vec::new(),
            angles: Vec::new(),
        });
        let k = match m.scenes.binary_search_by(|x| x.name.as_str().cmp(&s.row.scene)) {
            Ok(k) => k,
            Err(k) => {
                m.scenes.insert(k, SceneCount { name: s.row.scene.clone(), ..SceneCount::default() });
                k
            }
        };
        let sc = &mut m.scenes[k];
        match s.row.status {
            RowStatus::Fitted => {
                m.fitted += 1;
                sc.open += 1;
            }
            RowStatus::Waiting => {
                m.waiting += 1;
                sc.open += 1;
            }
            RowStatus::Pending => {
                m.waiting += 1;
                m.pending += 1;
                sc.open += 1;
            }
            RowStatus::Held => {
                m.held += 1;
                sc.held += 1;
            }
            RowStatus::Forgotten => {
                m.forgotten += 1;
                sc.forgotten += 1;
            }
        }
        if let Some(r) = s.row.ratio.filter(|_| s.row.status.open()) {
            m.ratios.push(r);
        }
        if s.row.status != RowStatus::Fitted {
            continue;
        }
        let n = s.z.len().min(s.pos.len());
        let trains = |a: usize| s.mask.is_none_or(|m| m.get(a).copied().unwrap_or(false));
        let rg = ranges.entry(c.clone()).or_default();
        for a in 0..n {
            for b in a + 1..n {
                let r2 = d2(s.pos[a], s.pos[b]);
                if r2 >= rc2 || !(trains(a) || trains(b)) {
                    continue;
                }
                let r = r2.sqrt();
                let e = rg.entry((s.z[a].min(s.z[b]), s.z[a].max(s.z[b]))).or_insert((f64::INFINITY, 0.0, 0));
                e.0 = e.0.min(r);
                e.1 = e.1.max(r);
                e.2 += 1;
            }
        }
        let ag = angles.entry(c.clone()).or_default();
        for i in (0..n).filter(|&i| trains(i)) {
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
            let h = ag.entry(s.z[i]).or_insert_with(|| vec![0; ANGLE_BINS]);
            for (a, u) in shell.iter().enumerate() {
                for v in &shell[a + 1..] {
                    let c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
                    h[angle_bin(c.acos().to_degrees())] += 1;
                }
            }
        }
    }
    let mut out: Vec<MoleculeKnowledge> = by
        .into_iter()
        .map(|(c, mut m)| {
            m.ratios.sort_by(f64::total_cmp);
            m.pairs = ranges.remove(&c).unwrap_or_default().into_iter().map(|(z, (min, max, count))| PairRange { z, min, max, count }).collect();
            m.angles = angles.remove(&c).unwrap_or_default().into_iter().collect();
            m
        })
        .collect();
    out.sort_by(|a, b| b.fitted.cmp(&a.fitted).then_with(|| b.total().cmp(&a.total())).then_with(|| a.composition.cmp(&b.composition)));
    out
}

/// Which stored samples to forget (2026-11-16, the Model tab's "Forget samples…" and "Forget all
/// above x×"): every sample in the fit's codes that it matches.
#[derive(Clone, Debug, PartialEq)]
pub enum ForgetRule {
    /// Holding an atom of this element.
    Element(u8),
    /// Of exactly this composition.
    Composition(Composition),
    /// From the scene of this name.
    Scene(String),
    /// Its leave-one-out ratio in the fit that stands is above this (∞ is above any).
    Above(f64),
}

impl ForgetRule {
    /// Whether a sample of `composition` from `scene` with leave-one-out ratio `ratio` matches.
    pub fn matches(&self, composition: &Composition, scene: &str, ratio: Option<f64>) -> bool {
        match self {
            ForgetRule::Element(z) => composition.contains(*z),
            ForgetRule::Composition(c) => composition == c,
            ForgetRule::Scene(s) => scene == s,
            ForgetRule::Above(x) => ratio.is_some_and(|r| r > *x),
        }
    }
}

/// What a rule picks of the stored samples ([`select`]).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Selection {
    /// The store indices to forget: matched and in the fit's codes, in store order.
    pub chosen: Vec<usize>,
    /// Matched but out of the fit already: held, forgotten.
    pub held: usize,
    pub forgotten: usize,
}

/// The samples of `rows` the rule picks (module doc, "Molecules").
pub fn select(rows: &[SampleRow], rule: &ForgetRule) -> Selection {
    let mut out = Selection::default();
    for r in rows.iter().filter(|r| rule.matches(&r.composition, &r.scene, r.ratio)) {
        match r.status {
            RowStatus::Fitted | RowStatus::Waiting | RowStatus::Pending => out.chosen.push(r.index),
            RowStatus::Held => out.held += 1,
            RowStatus::Forgotten => out.forgotten += 1,
        }
    }
    out
}

/// What a rule would forget, counted from the molecules ([`ForgetPreview::of`]).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ForgetPreview {
    /// The samples it forgets by composition, most first (then by composition).
    pub groups: Vec<(Composition, usize)>,
    pub total: usize,
    /// Matched but out of the fit already.
    pub held: usize,
    pub forgotten: usize,
    /// The species left with no sample in the fit's codes afterwards (they had some), ascending.
    pub emptied: Vec<u8>,
}

impl ForgetPreview {
    /// The rule's preview over `molecules` (module doc, "Molecules").
    pub fn of(molecules: &[MoleculeKnowledge], rule: &ForgetRule) -> ForgetPreview {
        let mut p = ForgetPreview::default();
        let mut taken: Vec<usize> = Vec::with_capacity(molecules.len());
        for m in molecules {
            let (n, held, forgotten) = match rule {
                ForgetRule::Element(_) | ForgetRule::Composition(_) if rule.matches(&m.composition, "", None) => (m.open(), m.held, m.forgotten),
                ForgetRule::Scene(s) => m.scenes.iter().find(|c| &c.name == s).map_or((0, 0, 0), |c| (c.open, c.held, c.forgotten)),
                ForgetRule::Above(_) => (m.ratios.iter().filter(|&&r| rule.matches(&m.composition, "", Some(r))).count(), 0, 0),
                _ => (0, 0, 0),
            };
            taken.push(n);
            p.held += held;
            p.forgotten += forgotten;
            if n > 0 {
                p.groups.push((m.composition.clone(), n));
                p.total += n;
            }
        }
        p.groups.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut species: BTreeMap<u8, (usize, usize)> = BTreeMap::new();
        for (m, &n) in molecules.iter().zip(&taken) {
            for z in m.composition.species() {
                let e = species.entry(z).or_default();
                e.0 += m.open();
                e.1 += n;
            }
        }
        p.emptied = species.into_iter().filter(|&(_, (open, n))| open > 0 && n >= open).map(|(z, _)| z).collect();
        p
    }
}

/// The end of the histogram's first peak (Bohr; module doc, "The first shell"): `None` for an
/// empty histogram.
pub fn first_shell(h: &PairHistogram) -> Option<f64> {
    let n = h.counts.len();
    let c = |k: isize| if k < 0 || k as usize >= n { 0.0 } else { h.counts[k as usize] as f64 };
    let s: Vec<f64> = (0..n as isize).map(|k| 0.25 * c(k - 1) + 0.5 * c(k) + 0.25 * c(k + 1)).collect();
    let mut k = s.iter().position(|&x| x > 0.0)?;
    while k + 1 < n && s[k + 1] > s[k] {
        k += 1;
    }
    while k + 1 < n && s[k + 1] > 0.0 && s[k + 1] <= s[k] {
        k += 1;
    }
    // stopped on a rise (bin k is the minimum) or before an empty smoothed bin (k + 1 is empty)
    let m = if k + 1 < n && s[k + 1] == 0.0 { k + 1 } else { k };
    Some((m as f64 + 0.5) * h.bin)
}

/// The angle bin of θ (degrees, clamped to 0–180).
pub fn angle_bin(theta_deg: f64) -> usize {
    ((theta_deg.clamp(0.0, 180.0) / ANGLE_BIN_DEG) as usize).min(ANGLE_BINS - 1)
}

/// The per-atom coverage of the geometry (`z`, `pos`) against `map`, 0 (unknown) … 1 (well
/// known) (module doc, "The per-atom coverage").
pub fn atom_coverage(map: &KnowledgeMap, z: &[u8], pos: &[[f64; 3]]) -> Vec<f32> {
    assert_eq!(z.len(), pos.len(), "atom_coverage: one Z per position");
    let rc2 = map.r_cut * map.r_cut;
    (0..z.len())
        .map(|i| {
            if map.element(z[i]).is_none() {
                return 0.0;
            }
            let mut cov = 1.0f64;
            for j in (0..z.len()).filter(|&j| j != i) {
                let d2: f64 = (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum();
                if d2 >= rc2 {
                    continue;
                }
                let Some(p) = map.pair(z[i], z[j]) else {
                    return 0.0;
                };
                let w = match p.confidence {
                    Confidence::Trusted => 1.0,
                    Confidence::Thin => 0.5,
                    Confidence::Unseen => 0.0,
                };
                let k = (d2.sqrt() / p.hist.bin) as usize;
                let n: u32 = (k.saturating_sub(1)..=k + 1).filter_map(|b| p.hist.counts.get(b)).sum();
                cov = cov.min(w * (1.0 - (-(n as f64) / COVERAGE_COUNT).exp()));
            }
            cov as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(z: (u8, u8), counts: Vec<u32>) -> PairHistogram {
        PairHistogram { z, bin: BIN, counts }
    }

    /// A sampled peak with one-bin noise, a gap and a second peak: the shell ends in the gap, not
    /// at the noise dip; a peak running into a shallow minimum ends at that minimum's bin.
    #[test]
    fn the_first_shell_ends_after_the_first_peak() {
        let mut c = vec![0u32; 60];
        for (k, v) in [(18, 2), (19, 6), (20, 4), (21, 7), (22, 3), (23, 1), (34, 5), (35, 9), (36, 4)] {
            c[k] = v;
        }
        let r = first_shell(&hist((1, 6), c)).unwrap();
        assert!((2.4..3.4).contains(&r), "the shell ends in the gap: {r}");
        // a dip that does not reach 0: the minimum's bin
        let mut c = vec![0u32; 60];
        for (k, v) in [(10, 4), (11, 10), (12, 4), (13, 2), (14, 2), (15, 6), (16, 12), (17, 6)] {
            c[k] = v;
        }
        let r = first_shell(&hist((1, 1), c)).unwrap();
        assert!((r - 1.35).abs() < 1e-12, "the dip's bin (13): {r}");
        assert_eq!(first_shell(&hist((1, 1), vec![0; 60])), None);
    }

    fn map() -> KnowledgeMap {
        // H–H and C–H seen at their bond lengths (1.4, 2.06) and H–H at methane's 3.36; C–C not
        let mut hh = vec![0u32; 60];
        hh[14] = 50;
        hh[33] = 300;
        let mut ch = vec![0u32; 60];
        ch[20] = 200;
        ch[21] = 200;
        let el = |z| ElementKnowledge { z, atoms_seen: 1, coordination: vec![], angles: vec![0; ANGLE_BINS] };
        KnowledgeMap {
            elements: vec![el(1), el(6)],
            pairs: vec![
                PairKnowledge { z: (1, 1), samples: 100, hist: hist((1, 1), hh), confidence: Confidence::Trusted, pending: vec![0; 60], pending_samples: 0 },
                PairKnowledge { z: (1, 6), samples: 50, hist: hist((1, 6), ch), confidence: Confidence::Trusted, pending: vec![0; 60], pending_samples: 0 },
                PairKnowledge { z: (6, 6), samples: 0, hist: hist((6, 6), vec![0; 60]), confidence: Confidence::Unseen, pending: vec![0; 60], pending_samples: 0 },
            ],
            slug: String::new(),
            samples: 150,
            r_cut: 6.0,
            molecules: Vec::new(),
            pending: 0,
            pending_keys: vec![PendingKey { class: "many-body", z: 1, neighbours: vec![1, 15], samples: 12 }],
            pending_classes: vec![("many-body", 12)],
        }
    }

    #[test]
    fn pending_keys_are_looked_up_by_class_and_key() {
        let m = map();
        assert_eq!(m.pending_key("many-body", 1, &[1, 15]), 12);
        assert_eq!((m.pending_key("pair", 1, &[1, 15]), m.pending_key("many-body", 1, &[1]), m.pending_key("many-body", 15, &[1, 15])), (0, 0, 0));
        assert_eq!((m.pending_class("many-body"), m.pending_class("pair")), (12, 0));
    }

    /// The plan's `atom_coverage_is_low_for_an_unseen_pair`: on a map that knows H–H and C–H but
    /// not C–C, methane's atoms read well known, ethane's carbons (a C–C neighbour) 0, an
    /// element the map lacks 0, a Thin pair at most ½, and a C–H distance never sampled 0.
    #[test]
    fn atom_coverage_is_low_for_an_unseen_pair() {
        let m = map();
        let t = 2.06 / 3f64.sqrt();
        let ch4_z = [6, 1, 1, 1, 1];
        let ch4 = [[0.0; 3], [t, t, t], [t, -t, -t], [-t, t, -t], [-t, -t, t]];
        let c = atom_coverage(&m, &ch4_z, &ch4);
        assert!(c.iter().all(|&x| x > 0.95), "methane is known: {c:?}");
        // ethane: C–C 2.9 Bohr, one H per carbon is enough to show it
        let c2 = atom_coverage(&m, &[6, 6, 1], &[[0.0; 3], [2.9, 0.0, 0.0], [-1.2, 1.68, 0.0]]);
        assert_eq!((c2[0], c2[1]), (0.0, 0.0), "a C–C neighbour is unseen: {c2:?}");
        assert_eq!(atom_coverage(&m, &[8, 1], &[[0.0; 3], [1.8, 0.0, 0.0]]), vec![0.0, 0.0], "O is unknown, and H's neighbour is O");
        assert_eq!(atom_coverage(&m, &[6, 1], &[[0.0; 3], [5.0, 0.0, 0.0]])[0], 0.0, "C–H at 5 Bohr was never sampled");
        assert_eq!(atom_coverage(&m, &[1], &[[0.0; 3]]), vec![1.0], "a lone atom of a known element");
        let mut thin = m.clone();
        thin.pairs[1].confidence = Confidence::Thin;
        let c3 = atom_coverage(&thin, &ch4_z, &ch4);
        assert!(c3.iter().all(|&x| x <= 0.5 && x > 0.45), "a Thin pair caps it at ½: {c3:?}");
    }

    #[test]
    fn angle_bins_cover_0_to_180() {
        assert_eq!(angle_bin(0.0), 0);
        assert_eq!(angle_bin(109.47), 21);
        assert_eq!(angle_bin(180.0), ANGLE_BINS - 1);
        assert_eq!(angle_bin(-3.0), 0);
    }

    fn row(index: usize, z: &[u8], scene: &str, status: RowStatus, ratio: Option<f64>) -> SampleRow {
        SampleRow { index, composition: Composition::of(z), scene: scene.into(), status, ratio }
    }

    #[test]
    fn a_composition_is_the_multiset_of_z() {
        let c = Composition::of(&[15, 1, 1, 1, 1, 1]);
        assert_eq!(c, Composition(vec![1, 1, 1, 1, 1, 15]));
        assert_eq!((c.species(), c.atoms(), c.contains(15), c.contains(8)), (vec![1, 15], 6, true, false));
        assert_eq!(Composition::of(&[8, 1, 1]), Composition::of(&[1, 8, 1]));
    }

    /// The rule per kind: only samples in the fit's codes are chosen (fitted, waiting or pending —
    /// kept since the last refit); the held and forgotten ones it matched are counted, not chosen;
    /// an empty match chooses nothing.
    #[test]
    fn a_rule_selects_only_samples_in_the_fit() {
        let rows = vec![
            row(0, &[15, 1, 1, 1], "ph3", RowStatus::Fitted, Some(1.2)),
            row(1, &[15, 1, 1, 1], "ph3", RowStatus::Waiting, None),
            row(2, &[15, 1, 1, 1, 1, 1], "mix", RowStatus::Fitted, Some(7.5)),
            row(3, &[15, 1, 1, 1], "ph3", RowStatus::Held, None),
            row(4, &[15, 1, 1, 1], "ph3", RowStatus::Forgotten, None),
            row(5, &[1, 1], "h2", RowStatus::Fitted, Some(f64::INFINITY)),
            row(6, &[1, 1], "", RowStatus::Fitted, Some(0.4)),
            row(7, &[15, 1, 1, 1], "ph3", RowStatus::Pending, None),
        ];
        let s = select(&rows, &ForgetRule::Element(15));
        assert_eq!((s.chosen, s.held, s.forgotten), (vec![0, 1, 2, 7], 1, 1));
        let s = select(&rows, &ForgetRule::Composition(Composition::of(&[15, 1, 1, 1])));
        assert_eq!((s.chosen, s.held, s.forgotten), (vec![0, 1, 7], 1, 1), "exact: not the PH₅ sample (PH₃ beside an H₂)");
        let s = select(&rows, &ForgetRule::Scene("ph3".into()));
        assert_eq!((s.chosen, s.held, s.forgotten), (vec![0, 1, 7], 1, 1));
        assert!(RowStatus::Pending.open(), "a pending sample can be forgotten");
        assert_eq!(select(&rows, &ForgetRule::Scene(String::new())).chosen, vec![6], "no scene recorded is a scene of its own");
        let s = select(&rows, &ForgetRule::Above(5.0));
        assert_eq!(s.chosen, vec![2, 5], "7.5 and ∞ are above 5; 1.2 and 0.4 are not; no ratio is never above");
        assert_eq!(select(&rows, &ForgetRule::Above(7.5)).chosen, vec![5], "strictly above");
        assert_eq!(select(&rows, &ForgetRule::Element(8)), Selection::default(), "an empty match does nothing");
    }

    /// The molecules: counted by composition and status (held and forgotten apart from the fitted
    /// count), by scene, most fitted first; the preview agrees with `select` and names a species it
    /// empties.
    #[test]
    fn molecules_count_compositions_and_the_preview_agrees_with_select() {
        let sh = BTreeMap::from([((1, 1), 2.0), ((1, 15), 3.0)]);
        let ph3: Vec<[f64; 3]> = vec![[0.0; 3], [2.7, 0.0, 0.0], [-1.3, 2.3, 0.0], [-1.3, -2.3, 0.0]];
        let h2: Vec<[f64; 3]> = vec![[0.0; 3], [1.4, 0.0, 0.0]];
        let mut mix = ph3.clone();
        mix.extend([[9.0, 0.0, 0.0], [10.4, 0.0, 0.0]]);
        let (zp, zh, zm) = ([15u8, 1, 1, 1], [1u8, 1], [15u8, 1, 1, 1, 1, 1]);
        let one = |i: usize, z: &'static [u8], pos: &'static [[f64; 3]], scene: &str, status: RowStatus, ratio: Option<f64>| StoredSample {
            row: SampleRow { index: i, composition: Composition::of(z), scene: scene.into(), status, ratio },
            z,
            pos,
            mask: None,
        };
        let leak = |v: &Vec<[f64; 3]>| -> &'static [[f64; 3]] { Box::leak(v.clone().into_boxed_slice()) };
        let (zp, zh, zm): (&'static [u8], &'static [u8], &'static [u8]) = (Box::leak(Box::new(zp)), Box::leak(Box::new(zh)), Box::leak(Box::new(zm)));
        let (ph3, h2, mix) = (leak(&ph3), leak(&h2), leak(&mix));
        let mut st: Vec<StoredSample> = (0..3).map(|i| one(i, zh, h2, "h2", RowStatus::Fitted, Some(1.0 + i as f64))).collect();
        st.push(one(3, zp, ph3, "ph3", RowStatus::Fitted, Some(6.0)));
        st.push(one(4, zp, ph3, "ph3", RowStatus::Held, None));
        st.push(one(5, zp, ph3, "", RowStatus::Forgotten, None));
        st.push(one(6, zm, mix, "mix", RowStatus::Waiting, None));
        st.push(one(7, zp, ph3, "ph3", RowStatus::Pending, None));
        st.push(one(8, zp, ph3, "ph3", RowStatus::Pending, None));
        let ms = molecules(&st, &sh, 6.0);
        let names: Vec<(Vec<u8>, usize, usize, usize, usize, usize)> = ms.iter().map(|m| (m.composition.0.clone(), m.fitted, m.waiting, m.pending, m.held, m.forgotten)).collect();
        assert_eq!(
            names,
            vec![(vec![1, 1], 3, 0, 0, 0, 0), (vec![1, 1, 1, 15], 1, 2, 2, 1, 1), (vec![1, 1, 1, 1, 1, 15], 0, 1, 0, 0, 0)],
            "a pending sample is waiting and counted as new; a misfit one only waiting"
        );
        let p = &ms[1];
        assert_eq!(p.scenes, vec![SceneCount { name: String::new(), open: 0, held: 0, forgotten: 1 }, SceneCount { name: "ph3".into(), open: 3, held: 1, forgotten: 0 }]);
        assert!(p.pairs.iter().all(|r| r.count == 3), "the pending PH₃ add nothing to the fitted ranges: {:?}", p.pairs);
        assert_eq!(p.ratios, vec![6.0]);
        assert_eq!(ms[0].ratios, vec![1.0, 2.0, 3.0]);
        assert!(ms[2].pairs.is_empty() && ms[2].angles.is_empty(), "nothing fitted: no geometry statistics");
        let rows: Vec<SampleRow> = st.iter().map(|s| s.row.clone()).collect();
        for rule in [ForgetRule::Element(15), ForgetRule::Element(1), ForgetRule::Composition(p.composition.clone()), ForgetRule::Scene("ph3".into()), ForgetRule::Above(2.5), ForgetRule::Element(8)] {
            let s = select(&rows, &rule);
            let v = ForgetPreview::of(&ms, &rule);
            assert_eq!((v.total, v.held, v.forgotten), (s.chosen.len(), s.held, s.forgotten), "{rule:?}");
        }
        let v = ForgetPreview::of(&ms, &ForgetRule::Element(15));
        assert_eq!(v.groups, vec![(Composition::of(&[15, 1, 1, 1]), 3), (Composition::of(&[15, 1, 1, 1, 1, 1]), 1)], "most first, the pending ones included");
        assert_eq!(v.emptied, vec![15], "every P sample in the fit goes");
        assert!(ForgetPreview::of(&ms, &ForgetRule::Composition(p.composition.clone())).emptied.is_empty(), "the PH₅ sample keeps P");
        let v = ForgetPreview::of(&ms, &ForgetRule::Above(2.5));
        assert_eq!((v.total, v.groups.len()), (2, 2), "3.0 (H₂) and 6.0 (PH₃)");
        assert_eq!(ForgetPreview::of(&ms, &ForgetRule::Element(8)), ForgetPreview::default());
    }

    /// The per-composition angles and pair ranges: a hand-built set of three bent H₂O (104.5°) and
    /// one linear H–O–H (180°), one of the bent ones held, with an H₂ in its own composition; only
    /// the fitted samples count, and each composition holds only its own angles.
    #[test]
    fn per_composition_angles_match_a_hand_built_set() {
        let sh = BTreeMap::from([((1, 1), 2.0), ((1, 8), 2.2)]);
        let bent = |r: f64| -> Vec<[f64; 3]> {
            let t = 104.5f64.to_radians() / 2.0;
            vec![[0.0; 3], [r * t.sin(), r * t.cos(), 0.0], [-r * t.sin(), r * t.cos(), 0.0]]
        };
        let lin = |r: f64| -> Vec<[f64; 3]> { vec![[0.0; 3], [r, 0.0, 0.0], [-r, 0.0, 0.0]] };
        let geoms = [bent(1.81), bent(1.85), lin(1.81), bent(1.83)];
        let h2 = vec![[0.0; 3], [1.4, 0.0, 0.0]];
        let zw = [8u8, 1, 1];
        let zh = [1u8, 1];
        let mut st: Vec<StoredSample> = geoms
            .iter()
            .enumerate()
            .map(|(i, g)| StoredSample {
                row: SampleRow { index: i, composition: Composition::of(&zw), scene: "h2o".into(), status: if i == 3 { RowStatus::Held } else { RowStatus::Fitted }, ratio: None },
                z: &zw,
                pos: g,
                mask: None,
            })
            .collect();
        st.push(StoredSample { row: SampleRow { index: 4, composition: Composition::of(&zh), scene: "h2".into(), status: RowStatus::Fitted, ratio: None }, z: &zh, pos: &h2, mask: None });
        let ms = molecules(&st, &sh, 6.0);
        let w = ms.iter().find(|m| m.composition == Composition::of(&[8, 1, 1])).unwrap();
        assert_eq!((w.fitted, w.held), (3, 1));
        let o = &w.angles.iter().find(|(z, _)| *z == 8).unwrap().1;
        let mut want = vec![0u32; ANGLE_BINS];
        want[angle_bin(104.5)] += 2;
        want[angle_bin(180.0)] += 1;
        assert_eq!(o, &want, "two bent, one linear; the held sample not counted");
        let hh = w.pairs.iter().find(|p| p.z == (1, 1)).unwrap();
        let hh_bent = 2.0 * 1.81 * (104.5f64.to_radians() / 2.0).sin();
        assert!((hh.min - hh_bent).abs() < 1e-9 && (hh.max - 3.62).abs() < 1e-9 && hh.count == 3, "{hh:?}");
        let oh = w.pairs.iter().find(|p| p.z == (1, 8)).unwrap();
        assert!((oh.min - 1.81).abs() < 1e-12 && (oh.max - 1.85).abs() < 1e-12 && oh.count == 6, "{oh:?}");
        // H–H in water is beyond the H–H shell: an H has only O as a shell neighbour, no H-centred angle
        assert!(w.angles.iter().all(|(z, h)| *z == 8 || h.iter().all(|&n| n == 0)), "{:?}", w.angles);
        let h = ms.iter().find(|m| m.composition == Composition::of(&[1, 1])).unwrap();
        assert!(h.angles.iter().all(|(_, a)| a.iter().all(|&n| n == 0)), "a dimer has no angle");
        assert_eq!(h.pairs, vec![PairRange { z: (1, 1), min: 1.4, max: 1.4, count: 1 }]);
    }
}
