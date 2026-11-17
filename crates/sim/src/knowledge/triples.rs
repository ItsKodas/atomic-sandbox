//! The three-atom coverage map's counting core (spec 2026-11-17 §6.1–§6.3; the module doc of
//! [`super`], "Three-atom arrangements").

use super::BIN;

/// The map's distance bin (Bohr): three pair bins, the pair rule's 0.3 Bohr window (spec §6.2).
pub const TRIPLE_BIN: f64 = 3.0 * BIN;

/// The angle's bins: uniform in u = cos θ, Δu = 2/`COS_BINS` = 0.2 (spec §6.2: body-3 sees the
/// angle only through P_l(u), l ≤ 3).
pub const COS_BINS: usize = 10;

/// A triple type a–c–b (spec §6.1): centre species `centre`, neighbour species `a` ≤ `b` — the
/// body-3 term's (centre, t₁, t₂).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TripleType {
    pub centre: u8,
    pub a: u8,
    pub b: u8,
}

impl TripleType {
    /// The type of a centre of species `centre` with neighbours of species `x` and `y`, in either
    /// order.
    pub fn new(centre: u8, x: u8, y: u8) -> TripleType {
        TripleType { centre, a: x.min(y), b: x.max(y) }
    }

    /// The two neighbours are of one species: the type is symmetric under r₁ ↔ r₂ and stored
    /// folded (r₁ = min, r₂ = max, only k₁ ≤ k₂).
    pub fn folded(&self) -> bool {
        self.a == self.b
    }
}

/// The cells of one triple type (spec §6.2): r₁ and r₂ in `n_r` bins of `r_bin` Bohr from 0, u in
/// `cos_bins` bins from u = 1 (θ = 0) down to u = −1 (θ = 180°). Cell index
/// (k₁·n_r + k₂)·cos_bins + q, dense; a folded type uses only k₁ ≤ k₂.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripleGrid {
    pub r_bin: f64,
    pub n_r: usize,
    pub cos_bins: usize,
}

impl TripleGrid {
    /// The grid for cutoff `r_cut` (Bohr): n_r = ⌈r_cut/[`TRIPLE_BIN`]⌉ (20 at 6, 27 at 8, 34 at
    /// 10), so every r < r_cut has a bin; the last may be partial.
    pub fn new(r_cut: f64) -> TripleGrid {
        let r_bin = TRIPLE_BIN;
        let mut n_r = (r_cut / r_bin).ceil().max(1.0) as usize;
        // a rounded quotient must not leave the top edge n_r·r_bin below r_cut (never taken for a
        // decimal multiple of 0.3: TRIPLE_BIN's double lies above 0.3)
        while (n_r as f64) * r_bin < r_cut {
            n_r += 1;
        }
        TripleGrid { r_bin, n_r, cos_bins: COS_BINS }
    }

    /// n_r² · cos_bins.
    pub fn cells(&self) -> usize {
        self.n_r * self.n_r * self.cos_bins
    }

    /// The distance bin of r (Bohr): `None` for a non-finite r, r ≤ 0 or r at or past the grid's
    /// top edge n_r·r_bin (which is ≥ r_cut).
    pub fn r_bin_of(&self, r: f64) -> Option<usize> {
        if !r.is_finite() || r <= 0.0 || r >= self.n_r as f64 * self.r_bin {
            return None;
        }
        // the min: r below the top edge whose quotient rounds up to n_r is in the last bin
        Some(((r / self.r_bin).floor() as usize).min(self.n_r - 1))
    }

    /// The cell index of (k₁, k₂, q); a folded type swaps k₁ and k₂ first when k₁ > k₂.
    pub fn index(&self, folded: bool, k1: usize, k2: usize, q: usize) -> usize {
        let (k1, k2) = if folded && k1 > k2 { (k2, k1) } else { (k1, k2) };
        (k1 * self.n_r + k2) * self.cos_bins + q
    }

    /// The cell of (r₁, r₂, u): `None` for a distance without a bin or a non-finite u (u is clamped
    /// to [−1, 1]).
    pub fn cell(&self, folded: bool, r1: f64, r2: f64, u: f64) -> Option<usize> {
        if !u.is_finite() {
            return None;
        }
        Some(self.index(folded, self.r_bin_of(r1)?, self.r_bin_of(r2)?, cos_bin_of(u, self.cos_bins)))
    }

    /// (k₁, k₂, q) of a cell index.
    pub fn coords(&self, cell: usize) -> (usize, usize, usize) {
        let q = cell % self.cos_bins;
        let k = cell / self.cos_bins;
        (k / self.n_r, k % self.n_r, q)
    }

    /// The cell's centre (r₁, r₂, u): Bohr, Bohr, cos θ.
    pub fn centre(&self, cell: usize) -> (f64, f64, f64) {
        let (k1, k2, q) = self.coords(cell);
        let u = 1.0 - (2 * q + 1) as f64 / self.cos_bins as f64;
        ((k1 as f64 + 0.5) * self.r_bin, (k2 as f64 + 0.5) * self.r_bin, u)
    }

    /// Distance bin k's edges (Bohr): [k·r_bin, (k + 1)·r_bin).
    pub fn r_edges(&self, k: usize) -> (f64, f64) {
        (k as f64 * self.r_bin, (k + 1) as f64 * self.r_bin)
    }
}

/// The cos θ bin of u ([`COS_BINS`] bins of 0.2, θ increasing with q): ⌊(1 − u)·COS_BINS/2⌋,
/// u clamped to [−1, 1] first, capped at COS_BINS − 1 (u = −1 is the last bin's). On an edge u
/// goes to the larger-θ bin (u = 0 → 5). The spec's ⌊(1 − u)/0.2⌋ is written as a multiplication so
/// the exact edges land exactly. A NaN u reads bin 0; [`TripleGrid::cell`] refuses it first.
pub fn cos_bin(u: f64) -> usize {
    cos_bin_of(u, COS_BINS)
}

fn cos_bin_of(u: f64, bins: usize) -> usize {
    (((1.0 - u.clamp(-1.0, 1.0)) * bins as f64 / 2.0).floor() as usize).min(bins - 1)
}

/// Cos bin q's edges as angles (degrees, display only): (θ at u = 1 − 0.2q, θ at u = 1 − 0.2(q + 1)).
/// Rounded: 0, 37, 53, 66, 78, 90, 102, 114, 127, 143, 180.
pub fn cos_edges_deg(q: usize) -> (f64, f64) {
    let theta = |q: usize| (1.0 - 2.0 * q as f64 / COS_BINS as f64).clamp(-1.0, 1.0).acos().to_degrees();
    (theta(q), theta(q + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_spans_r_cut_in_0_3_bohr_bins() {
        assert!((TRIPLE_BIN - 0.3).abs() < 1e-12);
        for (rc, n) in [(6.0, 20), (8.0, 27), (10.0, 34)] {
            let g = TripleGrid::new(rc);
            assert_eq!((g.n_r, g.cos_bins, g.cells()), (n, COS_BINS, n * n * COS_BINS), "r_cut {rc}");
        }
        let g = TripleGrid::new(6.0);
        assert_eq!(g.r_bin_of(1.81), Some(6));
        assert_eq!(g.r_bin_of(5.999_999), Some(19), "a hair under r_cut: the last bin");
        for bad in [0.0, -1.0, 6.3, f64::NAN, f64::INFINITY] {
            assert_eq!(g.r_bin_of(bad), None, "{bad}");
        }
        let (lo, hi) = g.r_edges(6);
        assert!((lo - 1.8).abs() < 1e-12 && (hi - 2.1).abs() < 1e-12);
    }

    /// Review Focus 2: whatever r_cut (a multiple of 0.3 or not), every r in (0, r_cut) — the one
    /// ulp under r_cut too — has a bin, and the grid adds no bin past ⌈r_cut/0.3⌉.
    #[test]
    fn every_distance_under_r_cut_has_a_bin() {
        let cuts = (1..=40).map(|m| m as f64 * 0.3).chain((1..=200).map(|m| m as f64 * 0.05 + 0.01));
        for rc in cuts {
            let g = TripleGrid::new(rc);
            let want = ((rc / 0.3) - 1e-9).ceil() as usize;
            assert_eq!(g.n_r, want, "r_cut {rc}");
            let below = f64::from_bits(rc.to_bits() - 1);
            assert_eq!(g.r_bin_of(below), Some(g.n_r - 1), "one ulp under r_cut {rc}");
            assert!(g.r_bin_of(f64::MIN_POSITIVE) == Some(0));
        }
    }

    /// Spec M2.
    #[test]
    fn cos_bins_are_uniform_in_cos_theta() {
        assert_eq!(cos_bin(1.0), 0, "0°");
        assert_eq!(cos_bin(0.0), 5, "90° on the edge: the larger-θ bin");
        assert_eq!(cos_bin(-1.0), 9, "180°");
        assert_eq!((cos_bin(1.000_000_1), cos_bin(-1.000_000_1)), (0, 9), "clamped");
        assert_eq!(cos_bin(104.5f64.to_radians().cos()), 6);
        let mut edges: Vec<i64> = (0..COS_BINS).map(|q| cos_edges_deg(q).0.round() as i64).collect();
        edges.push(cos_edges_deg(COS_BINS - 1).1.round() as i64);
        assert_eq!(edges, vec![0, 37, 53, 66, 78, 90, 102, 114, 127, 143, 180]);
    }

    /// Spec M3.
    #[test]
    fn a_symmetric_triple_folds() {
        let g = TripleGrid::new(6.0);
        let a = g.cell(true, 1.85, 2.95, -0.25).unwrap();
        assert_eq!(Some(a), g.cell(true, 2.95, 1.85, -0.25), "(r₁, r₂) and (r₂, r₁): one cell");
        assert_eq!(g.coords(a), (6, 9, 6));
        assert_ne!(g.cell(false, 1.85, 2.95, -0.25), g.cell(false, 2.95, 1.85, -0.25), "unfolded keeps the order");
        assert_eq!((g.index(true, 9, 6, 6), g.index(true, 6, 9, 6)), (a, a), "a folded slice at s reads {{min(k₁, s), max(k₁, s)}}");
        let (r1, r2, u) = g.centre(a);
        assert!((r1 - 1.95).abs() < 1e-12 && (r2 - 2.85).abs() < 1e-12 && (u + 0.3).abs() < 1e-12);
        assert_eq!(g.cell(false, 1.8, 1.9, f64::NAN), None);
        assert_eq!(TripleType::new(1, 8, 1), TripleType { centre: 1, a: 1, b: 8 });
        assert!(TripleType::new(8, 1, 1).folded() && !TripleType::new(1, 8, 1).folded());
    }
}
