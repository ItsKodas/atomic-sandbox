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

/// Pair-distance histogram bin (Bohr): the training views' (`TrainingViews::pairs`).
pub const BIN: f64 = 0.1;

/// Bond-angle histogram bin (degrees): 36 bins over 0–180°.
pub const ANGLE_BIN_DEG: f64 = 5.0;

/// Number of angle bins.
pub const ANGLE_BINS: usize = 36;

/// The atom-pair count at which a distance reads 1 − 1/e ≈ 0.63 known in [`atom_coverage`].
pub const COVERAGE_COUNT: f64 = 4.0;

pub use crate::activity::PairHistogram;

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
                PairKnowledge { z: (1, 1), samples: 100, hist: hist((1, 1), hh), confidence: Confidence::Trusted },
                PairKnowledge { z: (1, 6), samples: 50, hist: hist((1, 6), ch), confidence: Confidence::Trusted },
                PairKnowledge { z: (6, 6), samples: 0, hist: hist((6, 6), vec![0; 60]), confidence: Confidence::Unseen },
            ],
            slug: String::new(),
            samples: 150,
            r_cut: 6.0,
        }
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
}
