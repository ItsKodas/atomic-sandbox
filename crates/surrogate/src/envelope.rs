//! The training-descriptor envelope: per (species, feature) range and spread of the per-atom
//! descriptors the fit has seen, and how far a query atom lies outside it. The out-of-
//! distribution guard the provider applies *beside* the predictive variance.
//!
//! # Why the variance alone is not enough
//!
//! The Bayesian predictive variance of a linear model is a leverage, φ̃ᵀΣφ̃: it is large in
//! feature directions the data do not span and small in directions they do. It is therefore
//! also small wherever the features themselves are small — and the body-2 features R_n f_c all
//! vanish as a pair separates towards r_cut. A dimer trained at r ∈ [1.2, 2.2] Bohr and queried
//! at r = 5.9 has φ̃ ≈ 0, epistemic variance ≈ 0, and the model "confidently" answers Σ ref:
//! the variance cannot see that no training atom ever had an empty neighbourhood. The envelope
//! can: feature n = 0 (f_c itself) was never below f_c(2.2) ≈ 0.70 in training, and the query's
//! 7e-4 is far outside that range.
//!
//! # What is kept and where
//!
//! Per species s and feature k, over every atom of species s in every training sample: the
//! count, the minimum, the maximum, Σ B and Σ B². The statistics live in the [`Accumulator`]
//! (`Accumulator::envelope`), updated by `Accumulator::add` from the descriptors it computes
//! anyway, so they cost nothing extra and travel with the sufficient statistics the sidecar
//! (Task 5) persists — a model restored from statistics keeps its envelope, which a store-side
//! record would lose.
//!
//! # The distance and its unit
//!
//! For a query atom of species s with descriptor b, the gap outside the box per feature is
//! g_k = max(min_k − b_k, b_k − max_k, 0) and the distance is max_k g_k / u_k, with the unit
//!
//!   u_k = max(std_k, 0.1·rms_k),
//!
//! std_k the standard deviation and rms_k the root mean square of feature k over the training
//! atoms of species s. The unit is the training *spread*, not the model's standardisation
//! scale (which is an RMS of an uncentred feature): for a positive, bounded feature such as
//! f_c, g_k/rms_k ≤ 1 in the vanishing direction, so a threshold of 3 in RMS units could never
//! fire — exactly in the case the envelope exists for. In spread units the threshold is a
//! z-score: 3.0 means "three training spreads outside the range ever seen". The 0.1·rms floor
//! keeps a feature with (numerically) no spread from firing on negligible gaps, while still
//! letting the vanishing direction fire (g ≈ rms there, so g/u ≥ 10). A feature that was
//! identically zero in training (a neighbour-species channel never populated) has u_k = 0: any
//! non-zero query value is infinitely far, which is the honest answer. A species with no
//! training atom is infinitely far for every query.
//!
//! # The vanishing direction is judged relatively (M6 Task 11)
//!
//! The premise above — "g ≈ rms in the vanishing direction" — holds only while no training atom
//! ever had a sparse neighbourhood. One kept sample of a stretched pair breaks it: a hot H₂
//! fallback at r = 5.125 Bohr lowered the training minimum of the coordination feature
//! b2[n = 0] = Σ_j f_c(r_ij) from 0.531 to 0.0515 while its spread unit stayed 0.10 (set by the
//! hundreds of near-equilibrium atoms), so an atom with **no** neighbour (b = 0) sat 0.51 spreads
//! outside the box (limit 3) and was answered with E = Σ E_ref, F = 0 — the model's free-atom
//! limit, which the teacher's restricted dissociated pair lies 4.6e-2 Ha above (the user's
//! escape, `docs/superpowers/notes/2026-10-15-stuck-escape-diagnosis.md` §1). A spread is the
//! wrong yardstick there: it measures how much the bulk of the training atoms varied, not how
//! close to zero the fit has evidence for. The coordination features
//!
//!   C_t(i) = A_{(t,0),00}(i) = Σ_{j ∈ t, r_ij < r_cut} f_c(r_ij)   (one per neighbour species t)
//!
//! are non-negative and vanish exactly when atom i has no neighbour of species t inside r_cut —
//! a qualitative boundary (an emptying neighbourhood), and the one direction in which the
//! model's answer is pinned: every feature is a sum over neighbours (ruling S5: no intercept),
//! so E_i → E_ref + O(C) as C → 0 — an interpolation between the data at C ≥ min and a limit the
//! teacher never confirmed in that environment. So for these features the judgement is scale-
//! free: with m_t the training minimum of C_t over the atoms of the centre's species,
//!
//!   ratio_t = m_t / C_t   (∞ at C_t = 0 < m_t; 0 when m_t = 0 — an empty t-neighbourhood was seen),
//!
//! and the atom is outside when max_t ratio_t > [`COORDINATION_RATIO`] = 3: a coordination
//! three times below the smallest the fit has seen. Only the n = 0 channels qualify: the n ≥ 1
//! body-2 features T_n(x) f_c change sign inside the sphere, and the body-3 self-products
//! |A_{c,l}|² (l ≥ 1) vanish by symmetry in full neighbourhoods (the centre of a linear
//! trimer), so a ratio would fire on ordinary geometries there; (C_t)², the l = 0 self-product
//! of the n = 0 channel, carries the same information squared. The spread rule above is kept
//! for every feature; the atom is outside when either fires ([`Envelope::coordination_ratio`]
//! next to [`Envelope::distance`]; the provider applies both and, separately, refuses an atom
//! with no neighbour at all in a scene of more than one atom).
//!
//! **Why 3, and what it changes.** The factor is the spread limit's own "three times beyond
//! what was seen", taken multiplicatively, so no new tuning enters. Where the old premise held
//! the old decision is unchanged: the spread rule fires first whenever its unit is small against
//! the minimum (it fires at C < m − 3u, the ratio at C < m/3; the first is reached first iff
//! u < 2m/9). On the 49³ H₂ pass model (m = 0.531, u = 0.078 = 0.15 m) the spread rule fires at
//! C < 0.297 (r > 3.8 Bohr for one H neighbour) and the ratio would at C < 0.177 (r > 4.34):
//! unchanged. On the repro's model with the r = 5.125 tail sample (m = 0.0515, u = 0.10) the old
//! rule never fires in the vanishing direction (C = 0 is 0.51 spreads out); the new one fires
//! at C < 0.0172 (r > 5.50 Bohr for one neighbour) and always at C = 0. Near r_cut,
//! f_c ≈ (π δ / 2r_cut)² with δ = r_cut − r, so the factor 3 in C is a factor √3 in δ: the
//! query may approach the cutoff to within δ_min/√3 of the training's closest approach δ_min.
//! The window m/3 ≤ C < m is still answered by extrapolation toward the pinned limit; there the
//! variance and the audits are what judge it (a documented gap, not a guarantee).
//!
//! [`Accumulator`]: crate::fit::Accumulator

use crate::descriptor::Basis;

/// Per (species, feature) range and spread of the training descriptors (module doc). All
/// fields public for the sidecar; `add_atom` is the way to fill them.
#[derive(Clone, Debug, PartialEq)]
pub struct Envelope {
    pub species: Vec<u8>,
    pub n_features: usize,
    /// Training atoms per species.
    pub count: Vec<usize>,
    /// Per column s·P + k.
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub sum: Vec<f64>,
    pub sumsq: Vec<f64>,
}

/// Fraction of the RMS that floors the spread unit (module doc).
pub const SPREAD_FLOOR_OF_RMS: f64 = 0.1;

/// A coordination feature more than this factor below its training minimum is outside the
/// envelope (module doc, "The vanishing direction is judged relatively").
pub const COORDINATION_RATIO: f64 = 3.0;

impl Envelope {
    pub fn new(basis: &Basis) -> Envelope {
        let p = basis.n_features();
        let q = basis.n_species() * p;
        Envelope {
            species: basis.species.clone(),
            n_features: p,
            count: vec![0; basis.n_species()],
            min: vec![f64::INFINITY; q],
            max: vec![f64::NEG_INFINITY; q],
            sum: vec![0.0; q],
            sumsq: vec![0.0; q],
        }
    }

    /// Record one training atom of species index `s` with descriptor `b` (length `n_features`).
    pub fn add_atom(&mut self, s: usize, b: &[f64]) {
        assert_eq!(b.len(), self.n_features, "Envelope::add_atom: descriptor length");
        self.count[s] += 1;
        let off = s * self.n_features;
        for (k, &v) in b.iter().enumerate() {
            let q = off + k;
            self.min[q] = self.min[q].min(v);
            self.max[q] = self.max[q].max(v);
            self.sum[q] += v;
            self.sumsq[q] += v * v;
        }
    }

    /// The spread unit u_k = max(std_k, 0.1·rms_k) of feature `k` for species index `s`; 0 for
    /// a feature identically zero in training, NaN for a species without training atoms.
    pub fn unit(&self, s: usize, k: usize) -> f64 {
        let n = self.count[s];
        if n == 0 {
            return f64::NAN;
        }
        let q = s * self.n_features + k;
        let mean = self.sum[q] / n as f64;
        let ms = self.sumsq[q] / n as f64;
        let rms = ms.sqrt();
        let std = (ms - mean * mean).max(0.0).sqrt();
        std.max(SPREAD_FLOOR_OF_RMS * rms)
    }

    /// Distance of descriptor `b` (an atom of species index `s`) outside the training box, in
    /// spread units (module doc), and the feature index that attains it. 0 inside the box;
    /// `f64::INFINITY` for a species never trained or a feature that was identically zero.
    pub fn distance(&self, s: usize, b: &[f64]) -> (f64, usize) {
        assert_eq!(b.len(), self.n_features, "Envelope::distance: descriptor length");
        if self.count[s] == 0 {
            return (f64::INFINITY, 0);
        }
        let off = s * self.n_features;
        let mut worst = (0.0f64, 0usize);
        for (k, &v) in b.iter().enumerate() {
            let q = off + k;
            let gap = (self.min[q] - v).max(v - self.max[q]).max(0.0);
            if gap == 0.0 {
                continue;
            }
            let u = self.unit(s, k);
            let d = if u > 0.0 { gap / u } else { f64::INFINITY };
            if d > worst.0 {
                worst = (d, k);
            }
        }
        worst
    }

    /// The vanishing direction (module doc, "The vanishing direction is judged relatively"): over
    /// the coordination features C_t = b2[(t, n = 0)] of `basis`, the largest ratio of the
    /// training minimum to the query's value — ∞ for C_t = 0 below a positive minimum, 0 for a
    /// channel whose minimum is 0 (an empty neighbourhood was seen) — and that feature's index.
    /// Outside when the ratio exceeds [`COORDINATION_RATIO`]. A species with no training atom
    /// is infinitely far, as in [`distance`](Self::distance).
    pub fn coordination_ratio(&self, basis: &Basis, s: usize, b: &[f64]) -> (f64, usize) {
        assert_eq!(b.len(), self.n_features, "Envelope::coordination_ratio: descriptor length");
        assert_eq!(basis.n_features(), self.n_features, "Envelope::coordination_ratio: basis");
        if self.count[s] == 0 {
            return (f64::INFINITY, 0);
        }
        let off = s * self.n_features;
        let mut worst = (0.0f64, 0usize);
        for t in 0..basis.n_species() {
            let k = basis.body2_index(t, 0);
            let m = self.min[off + k];
            if m <= 0.0 {
                continue;
            }
            let ratio = if b[k] > 0.0 { m / b[k] } else { f64::INFINITY };
            if ratio > worst.0 {
                worst = (ratio, k);
            }
        }
        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_zero_inside_and_in_spread_units_outside() {
        let mut b = Basis::new(&[1]);
        b.body3 = false; // body-2 regression fixture (the default is body-3)
        b.n_max = 2;
        let mut e = Envelope::new(&b);
        // feature 0 ∈ {1, 2, 3} (mean 2, std √(2/3), rms √(14/3)); feature 1 ≡ 0
        for v in [1.0, 2.0, 3.0] {
            e.add_atom(0, &[v, 0.0]);
        }
        assert_eq!(e.count[0], 3);
        assert_eq!(e.distance(0, &[2.5, 0.0]), (0.0, 0));
        let u0 = e.unit(0, 0);
        assert!((u0 - (2.0f64 / 3.0).sqrt()).abs() < 1e-15, "std wins over 0.1·rms here: {u0}");
        let (d, k) = e.distance(0, &[4.0, 0.0]);
        assert_eq!(k, 0);
        assert!((d - 1.0 / u0).abs() < 1e-12, "{d}");
        let (d, k) = e.distance(0, &[-1.0, 0.0]);
        assert_eq!(k, 0);
        assert!((d - 2.0 / u0).abs() < 1e-12, "{d}");
        // a feature identically zero in training: any non-zero value is infinitely far
        assert_eq!(e.distance(0, &[2.0, 1e-9]), (f64::INFINITY, 1));
        assert_eq!(e.unit(0, 1), 0.0);
        // the floor: a feature with no spread but a non-zero value is judged in 0.1·rms units
        let mut f = Envelope::new(&b);
        for _ in 0..4 {
            f.add_atom(0, &[0.5, 0.0]);
        }
        assert!((f.unit(0, 0) - 0.05).abs() < 1e-15);
        assert!((f.distance(0, &[0.0, 0.0]).0 - 10.0).abs() < 1e-12, "the vanishing direction fires");
        assert!(f.distance(0, &[0.5 + 1e-3, 0.0]).0 < 0.03, "a negligible gap does not");
        // a species with no training atom
        let mut b2 = Basis::new(&[1, 8]);
        b2.body3 = false; // body-2 regression fixture (the default is body-3)
        let g = Envelope::new(&b2);
        assert_eq!(g.distance(1, &vec![0.0; b2.n_features()]).0, f64::INFINITY);
        assert!(g.unit(1, 0).is_nan());
    }

    /// M6 Task 11 (the user's escape): one tail sample at a sparse neighbourhood drags the
    /// coordination minimum to 0.05 while the bulk sets a spread of ~0.1, so an empty
    /// neighbourhood is "inside" in spread units; relative to the minimum it is not.
    #[test]
    fn one_tail_sample_does_not_make_an_empty_neighbourhood_inside() {
        let mut b = Basis::new(&[1, 8]);
        b.body3 = false;
        b.n_max = 2;
        // features: C_H = b2[(H,0)], b2[(H,1)], C_O = b2[(O,0)], b2[(O,1)]
        let mut e = Envelope::new(&b);
        for i in 0..200 {
            let c = 0.5 + 0.4 * i as f64 / 199.0; // the bulk: one neighbour at 1.2–2.9 Bohr
            e.add_atom(0, &[c, 0.3 * c - 0.2, 0.0, 0.0]); // never an O neighbour
        }
        e.add_atom(0, &[0.0515, -0.04, 0.0, 0.0]); // the tail sample
        let u = e.unit(0, 0);
        // the spread rule alone: an empty neighbourhood is half a spread out — "inside"
        let (d0, _) = e.distance(0, &[0.0, 0.0, 0.0, 0.0]);
        assert!((d0 - 0.0515 / u).abs() < 1e-12 && d0 < 1.0, "spread distance {d0} (unit {u})");
        // relative to the minimum it is infinitely far; a coordination 2.5× below is inside,
        // 5× below is outside
        assert_eq!(e.coordination_ratio(&b, 0, &[0.0, 0.0, 0.0, 0.0]), (f64::INFINITY, 0));
        let (r, k) = e.coordination_ratio(&b, 0, &[0.0515 / 2.5, 0.0, 0.0, 0.0]);
        assert!((r - 2.5).abs() < 1e-12 && k == 0 && r <= COORDINATION_RATIO);
        assert!(e.coordination_ratio(&b, 0, &[0.0515 / 5.0, 0.0, 0.0, 0.0]).0 > COORDINATION_RATIO);
        // inside the training range the ratio is ≤ 1
        assert!(e.coordination_ratio(&b, 0, &[0.7, 0.01, 0.0, 0.0]).0 < 1.0);
        // a channel whose minimum is 0 (no O neighbour was ever seen... and none here) never fires
        assert_eq!(b.body2_index(1, 0), 2);
        assert!(e.coordination_ratio(&b, 0, &[0.7, 0.0, 0.0, 0.0]).0 < 1.0);
        // a species never trained is infinitely far
        assert_eq!(e.coordination_ratio(&b, 1, &[0.7, 0.0, 0.0, 0.0]).0, f64::INFINITY);
        // without the tail sample the spread rule alone sees the empty neighbourhood (≥ 3
        // spreads: the old premise), and the ratio agrees
        let mut clean = Envelope::new(&b);
        for i in 0..200 {
            let c = 0.5 + 0.4 * i as f64 / 199.0;
            clean.add_atom(0, &[c, 0.3 * c - 0.2, 0.0, 0.0]);
        }
        assert!(clean.distance(0, &[0.0; 4]).0 > 3.0);
        assert_eq!(clean.coordination_ratio(&b, 0, &[0.0; 4]).0, f64::INFINITY);
    }
}
