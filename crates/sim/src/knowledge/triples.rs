//! The three-atom coverage map's counting core (spec 2026-11-17 §6.1–§6.3; the module doc of
//! [`super`], "Three-atom arrangements").

use super::{reads_taught, taught_range, Composition, PairHistogram, BIN};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

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
        assert!(r_cut.is_finite() && r_cut > 0.0, "r_cut {r_cut}");
        let r_bin = TRIPLE_BIN;
        let mut n_r = (r_cut / r_bin).ceil().max(1.0) as usize;
        // a rounded quotient must not leave the top edge n_r·r_bin below r_cut (never taken for a
        // decimal multiple of 0.3: TRIPLE_BIN's double lies above 0.3); one bin always suffices
        if (n_r as f64) * r_bin < r_cut {
            n_r += 1;
        }
        TripleGrid { r_bin, n_r, cos_bins: COS_BINS }
    }

    /// n_r² · cos_bins.
    pub fn cells(&self) -> usize {
        self.n_r * self.n_r * self.cos_bins
    }

    /// The distance bin of r (Bohr): `None` for a non-finite r, r ≤ 0 or r at or past the grid's
    /// top edge n_r·r_bin (which is ≥ r_cut). It does not apply r < r_cut: a partial last bin
    /// takes r in [r_cut, n_r·r_bin) too. Only [`each_triple`] applies the cutoff, so callers must
    /// not rely on this for it. Binning is ⌊r / r_bin⌋, consistent with [`Self::r_edges`] and
    /// [`Self::centre`].
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
/// goes to the larger-θ bin (u = 0 → 5). The spec's ⌊(1 − u)/0.2⌋ is written as a multiplication:
/// u = 0 and ±1 land exactly; other decimal edges may land one bin to the larger-θ side within an
/// ulp. A NaN u reads bin 0; [`TripleGrid::cell`] refuses it first.
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

/// One three-atom arrangement (spec §6.1): centre i = `centre`, neighbours `j` (species `t.a`, at
/// r₁ = |R_j − R_i|) and `k` (species `t.b`, at r₂); for a folded type j is the nearer (a tie: the
/// lower index). u = cos θ_jik, the angle at the centre, in [−1, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triple {
    pub t: TripleType,
    pub centre: usize,
    pub j: usize,
    pub k: usize,
    pub r1: f64,
    pub r2: f64,
    pub u: f64,
}

/// Every triple of a geometry (spec §6.1), the pairs body-3 sums: for each centre i with
/// `trains(i)`, every unordered pair {j, k}, j ≠ k, of its neighbours with 0 < r < `r_cut` (Bohr),
/// whatever the j–k distance. A coincident (r = 0) or non-finite neighbour is skipped. Atoms past
/// min(z.len(), pos.len()) are ignored. The one definition: the builder, the planner and the
/// scene markers all call it.
pub fn each_triple(z: &[u8], pos: &[[f64; 3]], r_cut: f64, trains: impl Fn(usize) -> bool, mut f: impl FnMut(Triple)) {
    let n = z.len().min(pos.len());
    let mut nb: Vec<([f64; 3], f64, usize)> = Vec::new();
    for i in (0..n).filter(|&i| trains(i)) {
        nb.clear();
        for j in (0..n).filter(|&j| j != i) {
            let d: [f64; 3] = std::array::from_fn(|a| pos[j][a] - pos[i][a]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r > 0.0 && r < r_cut {
                nb.push((d.map(|x| x / r), r, j));
            }
        }
        for (p, &(uj, rj, j)) in nb.iter().enumerate() {
            for &(uk, rk, k) in &nb[p + 1..] {
                let t = TripleType::new(z[i], z[j], z[k]);
                // j is listed before k (lower index): it stays first unless k is the species-a
                // neighbour of a mixed pair, or the strictly nearer of a folded one
                let swap = if t.folded() { rk < rj } else { z[j] != t.a };
                let u = (uj[0] * uk[0] + uj[1] * uk[1] + uj[2] * uk[2]).clamp(-1.0, 1.0);
                f(if swap {
                    Triple { t, centre: i, j: k, k: j, r1: rk, r2: rj, u }
                } else {
                    Triple { t, centre: i, j, k, r1: rj, r2: rk, u }
                });
            }
        }
    }
}

/// One triple type's counts over a set of samples ([`TripleTally::finish`]).
#[derive(Clone, Debug, PartialEq)]
pub struct TripleBlock {
    pub t: TripleType,
    /// Per cell of the tally's [`TripleGrid`], the triples counted there.
    pub counts: Arc<[u32]>,
    /// The samples holding at least one triple of the type.
    pub samples: usize,
    /// Its triples.
    pub triples: u64,
    /// The top three compositions holding it, by samples: most first, then by composition.
    pub compositions: Vec<(Composition, usize)>,
}

#[derive(Default)]
struct TallyEntry {
    counts: Vec<u32>,
    samples: usize,
    triples: u64,
    compositions: BTreeMap<Composition, usize>,
}

/// Counts the triples of samples, per type, on the grid of one r_cut (spec §6.1, §6.4).
pub struct TripleTally {
    grid: TripleGrid,
    r_cut: f64,
    types: BTreeMap<TripleType, TallyEntry>,
}

impl TripleTally {
    /// An empty tally for the model's cutoff `r_cut` (Bohr).
    pub fn new(r_cut: f64) -> TripleTally {
        TripleTally { grid: TripleGrid::new(r_cut), r_cut, types: BTreeMap::new() }
    }

    /// Counts one sample: each triple of [`each_triple`] in its cell, the sample once per type it
    /// holds, under its composition.
    pub fn add(&mut self, z: &[u8], pos: &[[f64; 3]], trains: impl Fn(usize) -> bool) {
        let (grid, cells) = (self.grid, self.grid.cells());
        let types = &mut self.types;
        let mut held: BTreeSet<TripleType> = BTreeSet::new();
        each_triple(z, pos, self.r_cut, trains, |t| {
            // 0 < r < r_cut always has a bin (TripleGrid::new) and u is finite: never None
            let Some(c) = grid.cell(t.t.folded(), t.r1, t.r2, t.u) else { return };
            let e = types.entry(t.t).or_default();
            if e.counts.is_empty() {
                e.counts = vec![0; cells];
            }
            e.counts[c] += 1;
            e.triples += 1;
            held.insert(t.t);
        });
        if held.is_empty() {
            return;
        }
        let comp = Composition::of(&z[..z.len().min(pos.len())]);
        for t in held {
            let e = types.get_mut(&t).expect("counted above");
            e.samples += 1;
            *e.compositions.entry(comp.clone()).or_default() += 1;
        }
    }

    /// The blocks, ascending type.
    pub fn finish(self) -> Vec<TripleBlock> {
        self.types
            .into_iter()
            .map(|(t, e)| {
                let mut compositions: Vec<(Composition, usize)> = e.compositions.into_iter().collect();
                compositions.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                compositions.truncate(3);
                TripleBlock { t, counts: e.counts.into(), samples: e.samples, triples: e.triples, compositions }
            })
            .collect()
    }
}

/// A fitted pair's taught distances (Bohr; [`super::taught_range`]): `lo` the lower edge of its
/// first filled bin, the shortest it was taught; `hi` the upper edge of its last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairSpan {
    /// (min Z, max Z).
    pub z: (u8, u8),
    pub lo: f64,
    pub hi: f64,
}

impl PairSpan {
    /// The span of a fitted pair histogram; `None` when it holds no distance.
    pub fn of(h: &PairHistogram) -> Option<PairSpan> {
        let (lo, hi) = taught_range(h)?;
        Some(PairSpan { z: (h.z.0.min(h.z.1), h.z.0.max(h.z.1)), lo, hi })
    }
}

/// What the model's basis holds, for the panel's honest-limits band (spec §6.7): body-3 and
/// body-4 terms, the total-degree cap D and the angular l_max.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripleBasis {
    pub body3: bool,
    pub body4: bool,
    pub degree_cap: usize,
    pub l_max: usize,
}

/// One triple type in the map (spec §6.1): fitted and pending counts on the map's grid.
#[derive(Clone, Debug, PartialEq)]
pub struct TripleKnowledge {
    pub t: TripleType,
    /// The fitted samples holding it.
    pub samples: usize,
    /// Its fitted triples.
    pub triples: u64,
    /// Fitted counts per cell (the fitted block's, shared, not copied): the classes read these alone.
    pub counts: Arc<[u32]>,
    /// Pending counts per cell ("new, waiting for refit"): they enter no class.
    pub pending: Vec<u32>,
    /// The pending samples holding it.
    pub pending_samples: usize,
    /// The top three fitted compositions holding it, most first.
    pub compositions: Vec<(Composition, usize)>,
}

/// The three-atom coverage map (spec §6.1–§6.3): per triple type of the fitted or pending samples,
/// ascending, its counts; the fitted pairs' spans the reach is judged from.
#[derive(Clone, Debug, PartialEq)]
pub struct TripleMap {
    pub grid: TripleGrid,
    /// The model's cutoff (Bohr).
    pub r_cut: f64,
    /// The pair scan's limit (Bohr), 5/6·r_cut (`providers::SCAN_CUTOFF_FRACTION`, supplied by the
    /// builder; `sim` keeps no copy): the reach ends there.
    pub reach_limit: f64,
    pub basis: TripleBasis,
    /// Every fitted pair with a taught distance, z normalised to (min, max).
    pub spans: Vec<PairSpan>,
    pub types: Vec<TripleKnowledge>,
}

impl TripleMap {
    /// The map from the fitted and the pending blocks (each on `TripleGrid::new(r_cut)`): one entry
    /// per type of either, ascending. `counts` is the fitted block's `Arc`, cloned (zeros for a
    /// pending-only type); `pending` the pending block's counts (zeros for none).
    pub fn assemble(r_cut: f64, reach_limit: f64, basis: TripleBasis, spans: Vec<PairSpan>, fitted: &[TripleBlock], pending: &[TripleBlock]) -> TripleMap {
        let grid = TripleGrid::new(r_cut);
        let cells = grid.cells();
        let spans = spans.into_iter().map(|s| PairSpan { z: (s.z.0.min(s.z.1), s.z.0.max(s.z.1)), ..s }).collect();
        let mut by: BTreeMap<TripleType, (Option<&TripleBlock>, Option<&TripleBlock>)> = BTreeMap::new();
        for b in fitted {
            by.entry(b.t).or_default().0 = Some(b);
        }
        for b in pending {
            by.entry(b.t).or_default().1 = Some(b);
        }
        let zeros: Arc<[u32]> = vec![0; cells].into();
        let types = by
            .into_iter()
            .map(|(t, (f, p))| {
                debug_assert!(f.is_none_or(|b| b.counts.len() == cells) && p.is_none_or(|b| b.counts.len() == cells), "a block off the map's grid");
                let mut pend = p.map_or_else(Vec::new, |b| b.counts.to_vec());
                pend.resize(cells, 0);
                TripleKnowledge {
                    t,
                    samples: f.map_or(0, |b| b.samples),
                    triples: f.map_or(0, |b| b.triples),
                    counts: f.map_or_else(|| zeros.clone(), |b| b.counts.clone()),
                    pending: pend,
                    pending_samples: p.map_or(0, |b| b.samples),
                    compositions: f.map_or_else(Vec::new, |b| b.compositions.clone()),
                }
            })
            .collect();
        TripleMap { grid, r_cut, reach_limit, basis, spans, types }
    }

    /// The type `t`'s entry.
    pub fn get(&self, t: TripleType) -> Option<&TripleKnowledge> {
        self.types.iter().find(|k| k.t == t)
    }

    /// The span of the pair (a, b), either order.
    pub fn span(&self, a: u8, b: u8) -> Option<&PairSpan> {
        let z = (a.min(b), a.max(b));
        self.spans.iter().find(|s| s.z == z)
    }
}

/// A cell's class (spec §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellClass {
    /// Fitted count n with [`reads_taught`] (n ≥ `taught_count()` = 10).
    Taught,
    /// 1 ≤ n < 10.
    Thin,
    /// n = 0, in the type's reach.
    Missing,
    /// n = 0 outside the reach (not drawn), a folded type's k₁ > k₂ half, or no cell of the grid.
    OutOfReach,
}

/// The cell is in the type's reach (spec §6.3), judged at its centre (r₁, r₂, u):
/// 1. r₁ in the c–a pair's taught span and r₁ ≤ `reach_limit`;
/// 2. r₂ in the c–b pair's taught span and r₂ ≤ `reach_limit`;
/// 3. r_ab = √(r₁² + r₂² − 2r₁r₂u) (the law of cosines: the a–b side opposite the angle at c) at
///    least the a–b pair's shortest fitted distance, or r_cut when a–b has none within r_cut.
///
/// False for a folded type's k₁ > k₂ half and an index past the grid.
pub fn reach(map: &TripleMap, t: TripleType, cell: usize) -> bool {
    let g = &map.grid;
    if cell >= g.cells() {
        return false;
    }
    let (k1, k2, _) = g.coords(cell);
    if t.folded() && k1 > k2 {
        return false;
    }
    let (r1, r2, u) = g.centre(cell);
    let (Some(ca), Some(cb)) = (map.span(t.centre, t.a), map.span(t.centre, t.b)) else {
        return false;
    };
    let within = |s: &PairSpan, r: f64| s.lo <= r && r <= s.hi && r <= map.reach_limit;
    if !(within(ca, r1) && within(cb, r2)) {
        return false;
    }
    let r_ab = (r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * u).max(0.0).sqrt();
    r_ab >= map.span(t.a, t.b).map_or(map.r_cut, |s| s.lo)
}

/// The cell's class (spec §6.3), from the fitted count alone: pending counts move nothing.
pub fn cell_class(map: &TripleMap, tk: &TripleKnowledge, cell: usize) -> CellClass {
    let g = &map.grid;
    if cell >= g.cells() {
        return CellClass::OutOfReach;
    }
    let (k1, k2, _) = g.coords(cell);
    if tk.t.folded() && k1 > k2 {
        return CellClass::OutOfReach;
    }
    match tk.counts.get(cell).copied().unwrap_or(0) {
        0 if reach(map, tk.t, cell) => CellClass::Missing,
        0 => CellClass::OutOfReach,
        n if reads_taught(n) => CellClass::Taught,
        _ => CellClass::Thin,
    }
}

/// A type's cells by class (spec §6.3): `taught` and `thin` over every filled cell, `missing` over
/// the empty cells in reach; `gaps` (the picker's order) the cells in reach that are not taught,
/// thin or missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TripleSummary {
    pub taught: usize,
    pub thin: usize,
    pub missing: usize,
    pub gaps: usize,
}

/// The type's [`TripleSummary`].
pub fn summary(map: &TripleMap, tk: &TripleKnowledge) -> TripleSummary {
    let mut s = TripleSummary::default();
    for c in 0..map.grid.cells() {
        match cell_class(map, tk, c) {
            CellClass::Taught => s.taught += 1,
            CellClass::Thin => {
                s.thin += 1;
                if reach(map, tk.t, c) {
                    s.gaps += 1;
                }
            }
            CellClass::Missing => {
                s.missing += 1;
                s.gaps += 1;
            }
            CellClass::OutOfReach => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::taught_count;

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

    /// Review Focus 2: a non-finite or non-positive r_cut is refused, never a wrapped n_r.
    #[test]
    fn a_non_finite_r_cut_is_refused() {
        for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
            assert!(std::panic::catch_unwind(|| TripleGrid::new(bad)).is_err(), "r_cut {bad}");
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

    /// Spec M1: three bent H₂O (104.5°) and one linear; in the fourth, O does not train.
    #[test]
    fn each_triple_counts_every_neighbour_pair_of_a_training_centre() {
        let bent = |r: f64| {
            let t = 104.5f64.to_radians() / 2.0;
            vec![[0.0; 3], [r * t.sin(), r * t.cos(), 0.0], [-r * t.sin(), r * t.cos(), 0.0]]
        };
        let lin = |r: f64| vec![[0.0; 3], [r, 0.0, 0.0], [-r, 0.0, 0.0]];
        let z = [8u8, 1, 1];
        let mut tally = TripleTally::new(6.0);
        for g in [bent(1.81), bent(1.85), lin(1.81)] {
            tally.add(&z, &g, |_| true);
        }
        tally.add(&z, &bent(1.83), |a| a != 0);
        let blocks = tally.finish();
        let g = TripleGrid::new(6.0);
        let (hoh, ohh) = (TripleType::new(8, 1, 1), TripleType::new(1, 8, 1));
        assert_eq!(blocks.iter().map(|b| b.t).collect::<Vec<_>>(), vec![ohh, hoh]);
        let w = &blocks[1];
        assert_eq!((w.samples, w.triples), (3, 3), "a non-training centre adds nothing");
        assert_eq!((w.counts[g.index(true, 6, 6, 6)], w.counts[g.index(true, 6, 6, 9)]), (2, 1), "bent at cos −0.25 (bin 6), linear (bin 9)");
        assert_eq!(w.counts.iter().sum::<u32>(), 3);
        let h = &blocks[0];
        assert_eq!((h.samples, h.triples), (4, 8), "two H centres per water");
        assert_eq!(h.counts[g.index(false, 9, 6, 1)], 6, "r₁ to H 2.86–2.93 (bin 9), r₂ to O (bin 6), 37.75° (bin 1)");
        assert_eq!(h.counts[g.index(false, 12, 6, 0)], 2, "linear: r₁ 3.62 (bin 12), 0°");
        assert_eq!(w.compositions, vec![(Composition::of(&z), 3)]);
        // j–k beyond r_cut still counts; j = k never; a dimer has none
        let mut t3 = TripleTally::new(3.0);
        t3.add(&z, &lin(1.81), |_| true);
        t3.add(&[1, 1], &[[0.0; 3], [1.4, 0.0, 0.0]], |_| true);
        let b3 = t3.finish();
        assert_eq!(b3.iter().map(|b| (b.t, b.triples)).collect::<Vec<_>>(), vec![(hoh, 1)], "H–H 3.62 > r_cut: no H-centred triple, H–O–H still counted");
    }

    /// The axes (spec §6.1): r₁ is the species-a neighbour's distance whatever the atom order; a
    /// folded triple puts the nearer neighbour first (a tie: the lower index); u is the angle at the
    /// centre; atoms past min(z.len(), pos.len()) are ignored.
    #[test]
    fn the_axes_follow_the_species_and_the_fold() {
        // C at the origin, H at 2.0 on x, O at 3.0 on y: 90° at C; the O listed before the H
        let (z, pos) = ([6u8, 8, 1], [[0.0; 3], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0]]);
        let mut seen = Vec::new();
        each_triple(&z, &pos, 6.0, |i| i == 0, |t| seen.push(t));
        assert_eq!(seen.len(), 1);
        let t = seen[0];
        assert_eq!((t.t, t.centre, t.j, t.k), (TripleType::new(6, 1, 8), 0, 2, 1));
        assert!((t.r1 - 2.0).abs() < 1e-12 && (t.r2 - 3.0).abs() < 1e-12 && t.u.abs() < 1e-12, "{t:?}");
        // folded: two H at 2.5 (index 1) and 1.5 (index 2) from an O, 60° apart: j is the nearer
        let s = 60f64.to_radians();
        let (z, pos) = ([8u8, 1, 1], [[0.0; 3], [2.5, 0.0, 0.0], [1.5 * s.cos(), 1.5 * s.sin(), 0.0]]);
        let mut seen = Vec::new();
        each_triple(&z, &pos, 6.0, |i| i == 0, |t| seen.push(t));
        assert_eq!((seen[0].j, seen[0].k), (2, 1));
        assert!((seen[0].r1 - 1.5).abs() < 1e-12 && (seen[0].r2 - 2.5).abs() < 1e-12 && (seen[0].u - 0.5).abs() < 1e-12);
        // a tie: the lower index first
        let (z, pos) = ([8u8, 1, 1], [[0.0; 3], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let mut seen = Vec::new();
        each_triple(&z, &pos, 6.0, |i| i == 0, |t| seen.push(t));
        assert_eq!((seen[0].j, seen[0].k), (1, 2));
        // a fourth Z without a position is ignored
        let mut n = 0;
        each_triple(&[8, 1, 1, 1], &pos, 6.0, |_| true, |_| n += 1);
        assert_eq!(n, 3, "one per centre of the three placed atoms");
    }

    /// Review Focus 3.
    #[test]
    fn coincident_atoms_make_no_triple_and_no_nan() {
        let (z, pos) = ([8u8, 1, 1], [[0.0; 3], [0.0; 3], [1.8, 0.0, 0.0]]);
        let mut seen = Vec::new();
        each_triple(&z, &pos, 6.0, |_| true, |t| seen.push(t));
        assert_eq!(seen.len(), 1, "only at the far H, whose two neighbours coincide: {seen:?}");
        assert!(seen.iter().all(|t| t.r1 > 0.0 && t.r2 > 0.0 && (-1.0..=1.0).contains(&t.u)));
        let mut tally = TripleTally::new(6.0);
        tally.add(&z, &pos, |_| true);
        assert_eq!(tally.finish().iter().map(|b| b.triples).sum::<u64>(), 1);
    }

    const BASIS: TripleBasis = TripleBasis { body3: true, body4: false, degree_cap: 4, l_max: 3 };

    fn span(z: (u8, u8), lo: f64, hi: f64) -> PairSpan {
        PairSpan { z, lo, hi }
    }

    /// A block on the r_cut-6 grid with these cell counts, one sample, no compositions.
    fn block(t: TripleType, cells: &[(usize, u32)]) -> TripleBlock {
        let mut counts = vec![0u32; TripleGrid::new(6.0).cells()];
        for &(c, n) in cells {
            counts[c] += n;
        }
        TripleBlock { t, counts: counts.into(), samples: 1, triples: cells.iter().map(|&(_, n)| n as u64).sum(), compositions: Vec::new() }
    }

    /// Spec M4.
    #[test]
    fn a_cell_is_taught_at_taught_count_and_pending_moves_nothing() {
        let g = TripleGrid::new(6.0);
        let (t, only_new) = (TripleType::new(8, 1, 1), TripleType::new(8, 1, 8));
        let (thin, taught, empty) = (g.index(true, 6, 6, 6), g.index(true, 6, 6, 5), g.index(true, 6, 6, 7));
        let spans = vec![span((1, 1), 2.6, 3.2), span((1, 8), 1.6, 2.2)];
        let fitted = block(t, &[(thin, taught_count() - 1), (taught, taught_count())]);
        let m0 = TripleMap::assemble(6.0, 5.0, BASIS, spans.clone(), &[fitted.clone()], &[]);
        let tk = m0.get(t).unwrap();
        assert_eq!([thin, taught, empty].map(|c| cell_class(&m0, tk, c)), [CellClass::Thin, CellClass::Taught, CellClass::Missing]);
        assert_eq!(cell_class(&m0, tk, g.index(false, 7, 6, 6)), CellClass::OutOfReach, "a folded type's k₁ > k₂ half is never a cell");
        let m1 = TripleMap::assemble(6.0, 5.0, BASIS, spans, &[fitted], &[block(t, &[(thin, 50), (empty, 50)]), block(only_new, &[(thin, 3)])]);
        let tk1 = m1.get(t).unwrap();
        assert_eq!([thin, taught, empty].map(|c| cell_class(&m1, tk1, c)), [CellClass::Thin, CellClass::Taught, CellClass::Missing], "pending never moves a class");
        assert!(Arc::ptr_eq(&tk.counts, &tk1.counts) && tk1.pending[empty] == 50);
        let n = m1.get(only_new).unwrap();
        assert_eq!((n.samples, n.triples, n.counts.iter().sum::<u32>(), n.pending_samples), (0, 0, 0, 1), "a type only pending samples hold");
        let s = summary(&m0, tk);
        let brute = (0..g.cells()).filter(|&c| cell_class(&m0, tk, c) == CellClass::Missing).count();
        assert_eq!(s, TripleSummary { taught: 1, thin: 1, missing: brute, gaps: brute + 1 });
    }

    /// Spec M5.
    #[test]
    fn reach_follows_the_three_pair_spans() {
        let g = TripleGrid::new(6.0);
        let hoh = TripleType::new(8, 1, 1);
        let m = TripleMap::assemble(6.0, 5.0, BASIS, vec![span((1, 1), 2.8, 4.0), span((1, 8), 1.5, 2.4)], &[], &[]);
        // 1 and 2: both distances in O–H's taught range, judged at the bin centre (1.65 in, 1.35 and 2.55 out)
        assert!(reach(&m, hoh, g.index(true, 5, 7, 9)));
        assert!(!reach(&m, hoh, g.index(true, 4, 6, 9)) && !reach(&m, hoh, g.index(true, 6, 8, 9)));
        // 3: r_ab ≥ H–H's shortest fitted distance 2.8 — at r₁ = r₂ = 1.95, cos −0.1 gives 2.89 (in), cos 0.1 gives 2.62 (out)
        assert!(reach(&m, hoh, g.index(true, 6, 6, 5)) && !reach(&m, hoh, g.index(true, 6, 6, 4)));
        // a–b never fitted within r_cut: the bound is r_cut, so no water-sized cell is in reach
        let no_hh = TripleMap::assemble(6.0, 5.0, BASIS, vec![span((1, 8), 1.5, 2.4)], &[], &[]);
        assert!((0..g.cells()).all(|c| !reach(&no_hh, hoh, c)));
        // the scan's limit: r ≤ reach_limit even inside the taught range (4.95 in, 5.25 out)
        let far = TripleMap::assemble(6.0, 5.0, BASIS, vec![span((1, 1), 0.5, 6.0), span((1, 8), 1.5, 5.8)], &[], &[]);
        assert!(reach(&far, hoh, g.index(true, 16, 16, 9)) && !reach(&far, hoh, g.index(true, 16, 17, 9)));
        // no c–a span: nothing
        assert!((0..g.cells()).all(|c| !reach(&m, TripleType::new(6, 1, 1), c)));
    }

    /// A mixed type reads its spans by species, c–a for r₁ and c–b for r₂; a span is found in either
    /// order; PairSpan::of is the pair's taught range; a cell index past the grid is out of reach.
    #[test]
    fn a_mixed_type_reads_c_a_for_r1_and_c_b_for_r2() {
        let g = TripleGrid::new(6.0);
        let t = TripleType::new(6, 1, 8); // H–C–O: r₁ to H, r₂ to O
        let spans = vec![span((1, 6), 1.8, 2.4), span((6, 8), 2.4, 3.0), span((1, 8), 2.0, 6.0)];
        let m = TripleMap::assemble(6.0, 5.0, BASIS, spans, &[], &[]);
        assert!(m.span(8, 6).is_some_and(|s| s.z == (6, 8)));
        // r₁ 1.95 (C–H in), r₂ 2.55 (C–O in), cos −0.5: r_ab ≈ 3.91 ≥ 2.0
        assert!(reach(&m, t, g.index(false, 6, 8, 7)));
        assert!(!reach(&m, t, g.index(false, 8, 6, 7)), "the swap: r₁ 2.55 is past C–H's 2.4");
        assert!(!reach(&m, t, g.cells()) && !reach(&m, t, usize::MAX));
        let tk = TripleKnowledge { t, samples: 0, triples: 0, counts: vec![0; g.cells()].into(), pending: vec![0; g.cells()], pending_samples: 0, compositions: Vec::new() };
        assert_eq!(cell_class(&m, &tk, g.cells()), CellClass::OutOfReach);
        let h = PairHistogram { z: (1, 6), bin: BIN, counts: vec![0, 0, 3, 0, 5, 0] };
        let s = PairSpan::of(&h).unwrap();
        assert!(s.z == (1, 6) && (s.lo - 0.2).abs() < 1e-12 && (s.hi - 0.5).abs() < 1e-12);
        assert_eq!(PairSpan::of(&PairHistogram { z: (1, 1), bin: BIN, counts: vec![0; 4] }), None);
    }
}
