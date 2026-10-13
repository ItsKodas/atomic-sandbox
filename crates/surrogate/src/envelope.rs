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
}
