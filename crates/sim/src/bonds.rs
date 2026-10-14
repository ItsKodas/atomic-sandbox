//! Bond detection from the density (M6 spec §3.1, §3.3, §3.5): which pairs of nuclei are bonded,
//! read off the converged electron density at a handful of points, with the promolecule of the
//! isolated-atom references (`crate::references`) as the yardstick. Pure f64, no grid: the
//! detector asks for samples at points in rounds and the caller (the driver) fetches them
//! through `ForceProvider::sample_density`. Nothing per element (CLAUDE.md rule 1): the
//! constants below are universal, the references are solved.
//!
//! # The three conditions (spec §3.1), in atomic units (Bohr, e/Bohr³)
//!
//! For a pair (A, B) with R = |R_B − R_A| < [`R_MAX`], the chord p(s) = R_A + s (R_B − R_A) is
//! sampled at s_k = k/32, k = 0 … 32 ([`SEGMENT_SAMPLES`] = 33). Every sample is **bias-corrected**
//! (§3.3, below): ρ̂(p) = ρ(p)·ρ_pro(p)/ρ_pro^tri(p), and the deformation ratio is
//! q(p) = ρ̂(p)/ρ_pro(p) = ρ(p)/ρ_pro^tri(p), where ρ_pro = Σ_C ρ̄_{Z_C}(|p − R_C|) over **all**
//! nuclei of the scene (ruling P5: with the pair's own atoms only, the H–H chord of water
//! compares oxygen's valence density with 2ρ_H and finds a bond).
//!
//! 1. **A bond point.**
//!    - **Route S (saddle):** the interior minimum of ρ̂ (k = 1 … 31) lies below both endpoints
//!      → p* is there. Across the chord, for the unit vectors e₁, e₂ ⊥ u = (R_B − R_A)/R,
//!      ρ̂(p* ± h e_k) ≤ (1 + ε_⊥) ρ̂(p*) ([`EPS_TRANSVERSE`]). A rise above that is followed
//!      before the pair is rejected (the **multi-centre march**, ruling P7 / R-1): a gradient
//!      ascent re-steered at every step. First direction g_⊥ = Σ_k [ρ̂(p* + h e_k) − ρ̂(p* − h e_k)]
//!      e_k from the four probes; each step of h/2 then re-estimates ∇ρ̂ by central differences
//!      from ±h/2 probes on the three lab axes (plus the point itself) and turns along it. It
//!      stops at the first of, in this order: (i) the next point lies within h of any nucleus →
//!      the rise was a neighbour's hill → `NotATransverseMaximum`; (ii) ρ̂ decreased → a
//!      non-nuclear maximum; (iii) |∇ρ̂| < f_g·g₀ ([`MARCH_STATIONARY_FRACTION`] of the gradient
//!      at the first march step) → a non-nuclear stationary point; (iv) 2R_max/h steps →
//!      `NotATransverseMaximum`. Ends (ii) and (iii) keep the pair, flagged `multi_centre`.
//!    - **Route D (deformation, MUST-4):** no interior minimum below both endpoints (at σ = h a
//!      heavy nucleus's smeared tail runs monotonically down to a hydrogen, the saddle degenerates
//!      onto the H nucleus). Then p* is the interior maximum of q along the chord — it must exceed
//!      both endpoint values, else `NoInteriorMinimum` — and the transverse test is applied to q.
//!      No march on this route (the spec defines none).
//!    Route S whenever an interior minimum of ρ̂ exists, Route D only otherwise.
//! 2. **A floor.** ρ̂(p*) ≥ ρ_floor ([`RHO_FLOOR`] = 1e-2): below Bader's closed-shell range
//!    nothing is a bond.
//! 3. **Accumulation.** q(p*) ≥ 1 + δ ([`DELTA`] = 0.10): the atoms put more density between
//!    them than they would merely sitting next to each other (He₂ depletes it).
//!
//! **The transverse basis** e₁, e₂ is not fixed by the spec; it is chosen here by the geometry so
//! that the test is rotation-covariant: e₁ is the component ⊥ u of the direction from the chord
//! to the centroid of the *other* nuclei — where a neighbour's hill or a ring point lies — and
//! e₂ = u × e₁; with no other nucleus, or one on the chord's line, the lab construction
//! (u × the axis least aligned with u). With lab-fixed probes the measured rise of a directional
//! feature varies by up to cos 45° between orientations, which would make `multi_centre` depend
//! on how the molecule is turned.
//!
//! # The same operator on both sides (§3.3, MUST-1)
//!
//! Trilinear interpolation of a tail e^{−κr} on a grid of spacing h errs by +κ²h²/8 at a cell
//! midpoint (3 % for H, 9 % for He at h = 0.25) and depends on where the point sits in its
//! cell. A ratio of two such samples would carry that. So [`Promolecule::trilinear`] applies the
//! provider's own operator — the eight grid nodes around p blended with the weights of
//! `qm::Grid::trilinear`, mirrored here in [`trilinear_cell`] — to the promolecule, and
//! ρ̂ = ρ·exact/trilinear (ρ unchanged where trilinear = 0, beyond every table). The leading
//! bias cancels; what remains is (h²/8)·|∂²ρ/ρ − ∂²ρ_pro/ρ_pro|, second order in the
//! deformation. Measured on the gridded bonded-pair fixture (the test
//! `the_same_operator_cancels_the_interpolation_bias`; `validate/references.md`, "Tools (M6)").
//! For samples that were never interpolated (an analytic provider, the exact fixtures) the
//! promolecule is built without a grid and the operator is the identity.
//!
//! # Constants (every one universal; derivations in `references.md`)
//! M = 33 chord samples (h/2-ish spacing on a 4 Bohr chord); ε_⊥ = 0.07 and δ = 0.11, each ≥ 3×
//! the measured fixture noise (transverse ratio spread 0.018, corrected q spread 3.4 % of its
//! mean — the gridded bonded pair over 24 placements; the spec's provisional δ = 0.10 was raised
//! by that measurement under its own ≥ 3× rule); ρ_floor = 1e-2 (two H 5 Bohr apart:
//! ρ_mid ≈ 4e-3, not bonded; 2.5 Bohr apart: 0.05, a candidate); R_max = 8 Bohr (beyond it no
//! two atoms in range share density above the floor); the march step h/2 with the "within h of
//! a nucleus" rule and f_g = 0.25 (a ring point whose in-plane curvature is nearly flat: the
//! gradient falls to a quarter of its first value within ~0.1 Bohr of the point for a quadratic
//! top starting 0.5 Bohr away).
//!
//! # What this is not
//! Not QTAIM: no gradient-path integration, no basins, no bond order. The chord is a chord of the
//! true bond path; the march is a one-dimensional probe of the ring/cage point, not a critical-
//! point search. Ionic and charged closed-shell bonds are the open question (spec §3.6).
//!
//! # Molecules (§3.5)
//! A molecule is a connected component of the bond graph ([`components`], union–find); isolated
//! nuclei are one-atom components.
use crate::references::{AtomReferences, RadialDensity, ReferenceKey};
use crate::scene::Cell;

/// Chord samples per pair (s = k/32).
pub const SEGMENT_SAMPLES: usize = 33;
/// The transverse rise a bond point may show before it is followed (module doc).
pub const EPS_TRANSVERSE: f64 = 0.07;
/// The density floor at the bond point (e/Bohr³).
pub const RHO_FLOOR: f64 = 1e-2;
/// The accumulation margin: q(p*) ≥ 1 + δ. The spec's provisional 0.10 (≥ 3 × the re-review's
/// 2.8 % noise), raised to 0.11 by this task's measurement on the gridded fixture (q spread
/// 0.0629 on a mean of 1.852: 3 × 3.4 % = 0.102 — the test
/// `the_same_operator_cancels_the_interpolation_bias`; `references.md`, "Tools (M6)").
pub const DELTA: f64 = 0.11;
/// Pairs farther apart than this (Bohr) are not sampled.
pub const R_MAX: f64 = 8.0;
/// The march stops when |∇ρ̂| falls below this fraction of its value at the first march step.
pub const MARCH_STATIONARY_FRACTION: f64 = 0.25;

/// Which rule placed the bond point (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Saddle,
    Deformation,
}

/// What the detector concluded for a pair (module doc; `Bonded` is the only accepting verdict).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Bonded { multi_centre: bool },
    TooFar,
    NoInteriorMinimum,
    NotATransverseMaximum,
    BelowFloor,
    NotAccumulated,
}

/// Why a march stopped (module doc, the four stops).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarchStop {
    Nucleus,
    Maximum,
    Stationary,
    StepCap,
}

/// Every sampled pair with the numbers behind its verdict (MUST-5): `s` the bond point's chord
/// fraction, `rho_min` = ρ̂(p*) (the chord minimum on Route S, the density at the q maximum on
/// Route D), `transverse_ratio` the largest probe/bond-point ratio (of ρ̂ on Route S, of q on
/// Route D), `rho_pro` = ρ_pro(p*), `enhancement` = q(p*); `march_end`/`march_stop` where and why
/// a march stopped. Each is `None` when the pair never reached that stage.
#[derive(Clone, Debug, PartialEq)]
pub struct PairReport {
    pub i: usize,
    pub j: usize,
    pub route: Option<Route>,
    pub s: Option<f64>,
    pub rho_min: Option<f64>,
    pub transverse_ratio: Option<f64>,
    pub rho_pro: Option<f64>,
    pub enhancement: Option<f64>,
    pub march_end: Option<[f64; 3]>,
    pub march_stop: Option<MarchStop>,
    pub verdict: Verdict,
}

/// The accepted subset of the reports.
#[derive(Clone, Debug, PartialEq)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    /// ρ̂ at the bond point (e/Bohr³).
    pub rho_bcp: f64,
    pub enhancement: f64,
    pub s: f64,
    pub route: Route,
    pub multi_centre: bool,
}

/// The bonds and the molecules of one evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct BondGraph {
    pub bonds: Vec<Bond>,
    pub components: Vec<Vec<usize>>,
    pub step: u64,
    pub sim_time: f64,
    pub n_nuclei: usize,
}

// ---- small vector helpers ----

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

// ---- the grid operator (mirror of qm::Grid::trilinear_cell / trilinear) ----

/// The trilinear cell of `p` on `cell`'s grid (`qm::Grid::cube(n, h)`: origin −(n−1)h/2 on each
/// axis): the lower corner ⌊x⌋ and the fractions t = x − ⌊x⌋ ∈ [0, 1) per axis, x = (p −
/// origin)/h; on an upper face the corner is n − 1 with t = 0. `None` outside the grid (NaN
/// included). The same definition as `qm::Grid::trilinear_cell` (asserted equal in the tests).
pub fn trilinear_cell(cell: &Cell, p: [f64; 3]) -> Option<([usize; 3], [f64; 3])> {
    let origin = -cell.half_extent();
    let mut c = [0usize; 3];
    let mut t = [0.0f64; 3];
    for a in 0..3 {
        let x = (p[a] - origin) / cell.h;
        if !(x >= 0.0 && x <= (cell.n - 1) as f64) {
            return None;
        }
        let i0 = x.floor();
        c[a] = i0 as usize;
        t[a] = x - i0;
    }
    Some((c, t))
}

/// The position of grid node (i, j, k) of `cell`.
pub fn node_position(cell: &Cell, ijk: [usize; 3]) -> [f64; 3] {
    let origin = -cell.half_extent();
    ijk.map(|i| origin + i as f64 * cell.h)
}

/// `f` at the eight nodes around `p`, blended a·(1 − t) + b·t along x, then y, then z — the
/// operator `qm::Grid::trilinear` applies to a stored field (a neighbour beyond an upper face has
/// weight 0 and is not read); 0 outside the grid.
pub fn trilinear_blend(cell: &Cell, p: [f64; 3], f: impl Fn([usize; 3]) -> f64) -> f64 {
    let Some(([i, j, k], [tx, ty, tz])) = trilinear_cell(cell, p) else { return 0.0 };
    let n = cell.n;
    let at = |di: usize, dj: usize, dk: usize| {
        let (a, b, c) = (i + di, j + dj, k + dk);
        if a < n && b < n && c < n { f([a, b, c]) } else { 0.0 }
    };
    let lerp = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;
    let x = |dj, dk| if tx == 0.0 { at(0, dj, dk) } else { lerp(at(0, dj, dk), at(1, dj, dk), tx) };
    let y = |dk| if ty == 0.0 { x(0, dk) } else { lerp(x(0, dk), x(1, dk), ty) };
    if tz == 0.0 { y(0) } else { lerp(y(0), y(1), tz) }
}

// ---- the promolecule ----

/// The promolecule of the current scene (module doc, "The same operator"): `exact(p)` = Σ_C
/// ρ̄_{Z_C}(|p − R_C|) from the radial tables; `trilinear(p)` the same quantity at the eight grid
/// nodes around p blended with the provider's weights — or `exact(p)` itself when the samples
/// were not interpolated (`grid` = `None`).
pub struct Promolecule<'a> {
    tables: Vec<&'a RadialDensity>,
    pos: Vec<[f64; 3]>,
    grid: Option<Cell>,
}

impl<'a> Promolecule<'a> {
    /// One table per nucleus, at `pos`; `grid` is the provider's grid (the scene's cell) or
    /// `None` for exact samples.
    pub fn new(tables: Vec<&'a RadialDensity>, pos: &[[f64; 3]], grid: Option<&Cell>) -> Promolecule<'a> {
        assert_eq!(tables.len(), pos.len(), "Promolecule: {} tables for {} nuclei", tables.len(), pos.len());
        Promolecule { tables, pos: pos.to_vec(), grid: grid.cloned() }
    }

    /// From the reference cache: the table of species `z[c]` under `key_for(z[c])` for every
    /// nucleus, on `cell`'s grid. `Err` lists the species without a (non-empty) table, each once,
    /// in scene order.
    pub fn from_refs(refs: &'a AtomReferences, key_for: impl Fn(u8) -> ReferenceKey, cell: &Cell, pos: &[[f64; 3]], z: &[u8]) -> Result<Promolecule<'a>, Vec<u8>> {
        assert_eq!(pos.len(), z.len());
        let mut tables = Vec::with_capacity(z.len());
        let mut missing: Vec<u8> = Vec::new();
        for &zc in z {
            match refs.get(&key_for(zc)).filter(|r| !r.radial.is_empty()) {
                Some(r) => tables.push(&r.radial),
                None => {
                    if !missing.contains(&zc) {
                        missing.push(zc);
                    }
                }
            }
        }
        if !missing.is_empty() {
            return Err(missing);
        }
        Ok(Promolecule { tables, pos: pos.to_vec(), grid: Some(cell.clone()) })
    }

    /// ρ_pro(p) = Σ_C ρ̄_{Z_C}(|p − R_C|).
    pub fn exact(&self, p: [f64; 3]) -> f64 {
        self.tables.iter().zip(&self.pos).map(|(t, &r)| t.at(dist(p, r))).sum()
    }

    /// The promolecule put through the provider's interpolation operator (module doc).
    pub fn trilinear(&self, p: [f64; 3]) -> f64 {
        match &self.grid {
            None => self.exact(p),
            Some(cell) => trilinear_blend(cell, p, |ijk| self.exact(node_position(cell, ijk))),
        }
    }

    /// ρ̂ = ρ·exact/trilinear; ρ unchanged where trilinear(p) = 0.
    pub fn corrected(&self, p: [f64; 3], rho: f64) -> f64 {
        let tri = self.trilinear(p);
        if tri > 0.0 { rho * self.exact(p) / tri } else { rho }
    }

    /// q = ρ/trilinear(p) = ρ̂/exact (0 where the promolecule has no reach: beyond every table).
    pub fn ratio(&self, p: [f64; 3], rho: f64) -> f64 {
        let tri = self.trilinear(p);
        if tri > 0.0 { rho / tri } else { 0.0 }
    }
}

// ---- the detector ----

/// One candidate pair's working state (its indices are in its report).
struct Pair {
    e1: [f64; 3],
    e2: [f64; 3],
    /// The chord points p_k.
    p: Vec<[f64; 3]>,
    /// Filled by round 1.
    p_star: [f64; 3],
    rho_star: f64,
    q_star: f64,
    route: Route,
    /// Index into `Detector::reports`.
    report: usize,
}

/// A march in progress (module doc).
struct March {
    pair: usize,
    /// The last accepted point and ρ̂ there.
    pos: [f64; 3],
    rho: f64,
    /// The point being sampled (pos + h/2·dir).
    next: [f64; 3],
    /// |∇ρ̂| at the first march step.
    g0: Option<f64>,
    steps: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Chord,
    Transverse,
    March,
    Done,
}

/// The detector runs in rounds (module doc): round 1 the chord samples of every pair within
/// [`R_MAX`]; round 2 the four transverse probes of the survivors; then one small round per
/// march step (7 samples per marching pair: the point and ±h/2 on the three axes) until every
/// march has stopped. `points()` is what to sample next; `feed` takes the samples and returns
/// the next set, `None` when done — the reports are complete then.
pub struct Detector {
    pos: Vec<[f64; 3]>,
    pairs: Vec<Pair>,
    reports: Vec<PairReport>,
    points: Vec<[f64; 3]>,
    stage: Stage,
    /// Pairs the current round's points belong to (round 2).
    active: Vec<usize>,
    marches: Vec<March>,
}

/// e₁ ⊥ u toward the centroid of `others` from `mid` (module doc, "The transverse basis"), the
/// lab construction when that is degenerate; e₂ = u × e₁.
fn transverse_basis(u: [f64; 3], mid: [f64; 3], others: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut e1 = None;
    if !others.is_empty() {
        let mut c = [0.0; 3];
        for o in others {
            c = add(c, *o);
        }
        let v = sub(scale(c, 1.0 / others.len() as f64), mid);
        let v_perp = sub(v, scale(u, dot(v, u)));
        let n = norm(v_perp);
        if n > 1e-9 * norm(v).max(1.0) {
            e1 = Some(scale(v_perp, 1.0 / n));
        }
    }
    let e1 = e1.unwrap_or_else(|| {
        let axis = (0..3).min_by(|&a, &b| u[a].abs().total_cmp(&u[b].abs())).expect("three axes");
        let mut a = [0.0; 3];
        a[axis] = 1.0;
        let c = cross(u, a);
        scale(c, 1.0 / norm(c))
    });
    (e1, cross(u, e1))
}

impl Detector {
    /// Round 1: every pair within [`R_MAX`] gets [`SEGMENT_SAMPLES`] chord points (the endpoints
    /// are the nuclei themselves); farther pairs are reported `TooFar` and never sampled.
    pub fn new(pos: &[[f64; 3]]) -> Detector {
        let n = pos.len();
        let mut pairs = Vec::new();
        let mut reports = Vec::new();
        let mut points = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let d = sub(pos[j], pos[i]);
                let r = norm(d);
                let report = PairReport { i, j, route: None, s: None, rho_min: None, transverse_ratio: None, rho_pro: None, enhancement: None, march_end: None, march_stop: None, verdict: Verdict::TooFar };
                if !(r < R_MAX) || r == 0.0 {
                    reports.push(report);
                    continue;
                }
                let u = scale(d, 1.0 / r);
                let mid = add(pos[i], scale(d, 0.5));
                let others: Vec<[f64; 3]> = (0..n).filter(|&c| c != i && c != j).map(|c| pos[c]).collect();
                let (e1, e2) = transverse_basis(u, mid, &others);
                let p: Vec<[f64; 3]> = (0..SEGMENT_SAMPLES).map(|k| add(pos[i], scale(d, k as f64 / (SEGMENT_SAMPLES - 1) as f64))).collect();
                points.extend_from_slice(&p);
                pairs.push(Pair { e1, e2, p, p_star: [0.0; 3], rho_star: 0.0, q_star: 0.0, route: Route::Saddle, report: reports.len() });
                reports.push(report);
            }
        }
        let stage = if pairs.is_empty() { Stage::Done } else { Stage::Chord };
        Detector { pos: pos.to_vec(), pairs, reports, points, stage, active: Vec::new(), marches: Vec::new() }
    }

    /// The points to sample for the current round (empty when done).
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// The reports of every pair, (i < j) order — complete once `feed` has returned `None`.
    pub fn reports(&self) -> &[PairReport] {
        &self.reports
    }

    /// The four transverse probes of a pair: p* + h e₁, p* − h e₁, p* + h e₂, p* − h e₂.
    fn probes(pair: &Pair, h: f64) -> [[f64; 3]; 4] {
        [add(pair.p_star, scale(pair.e1, h)), sub(pair.p_star, scale(pair.e1, h)), add(pair.p_star, scale(pair.e2, h)), sub(pair.p_star, scale(pair.e2, h))]
    }

    /// A march round's seven points at `c`: c, c ± h/2 x̂, c ± h/2 ŷ, c ± h/2 ẑ.
    fn march_points(c: [f64; 3], h: f64) -> [[f64; 3]; 7] {
        let d = 0.5 * h;
        [c, [c[0] + d, c[1], c[2]], [c[0] - d, c[1], c[2]], [c[0], c[1] + d, c[2]], [c[0], c[1] - d, c[2]], [c[0], c[1], c[2] + d], [c[0], c[1], c[2] - d]]
    }

    fn near_nucleus(&self, p: [f64; 3], h: f64) -> bool {
        self.pos.iter().any(|&r| dist(p, r) < h)
    }

    /// Conditions 2 and 3 at the bond point.
    fn finalize(&mut self, idx: usize, multi_centre: bool) {
        let pair = &self.pairs[idx];
        let verdict = if pair.rho_star < RHO_FLOOR {
            Verdict::BelowFloor
        } else if pair.q_star < 1.0 + DELTA {
            Verdict::NotAccumulated
        } else {
            Verdict::Bonded { multi_centre }
        };
        self.reports[pair.report].verdict = verdict;
    }

    /// Feeds the samples of the last `points()` (the provider's ρ at them, in order) and returns
    /// the next point set, or `None` when every pair has its verdict. `pro` is the scene's
    /// promolecule, `h` the grid spacing (probe offset h, march step h/2, the nucleus rule).
    pub fn feed(&mut self, rho: &[f64], pro: &Promolecule<'_>, h: f64) -> Option<&[[f64; 3]]> {
        assert_eq!(rho.len(), self.points.len(), "Detector::feed: {} samples for {} points", rho.len(), self.points.len());
        assert!(h > 0.0 && h.is_finite(), "Detector::feed: h = {h}");
        let cap = (2.0 * R_MAX / h).round() as usize;
        let mut next: Vec<[f64; 3]> = Vec::new();
        match self.stage {
            Stage::Done => return None,
            Stage::Chord => {
                let mut active = Vec::new();
                for (idx, off) in (0..self.pairs.len()).zip((0..).step_by(SEGMENT_SAMPLES)) {
                    let pair = &mut self.pairs[idx];
                    let raw = &rho[off..off + SEGMENT_SAMPLES];
                    let rho_hat: Vec<f64> = pair.p.iter().zip(raw).map(|(&p, &v)| pro.corrected(p, v)).collect();
                    let q: Vec<f64> = pair.p.iter().zip(raw).map(|(&p, &v)| pro.ratio(p, v)).collect();
                    let last = SEGMENT_SAMPLES - 1;
                    let k_min = (1..last).min_by(|&a, &b| rho_hat[a].total_cmp(&rho_hat[b])).expect("interior points");
                    let (route, k_star, interior) = if rho_hat[k_min] < rho_hat[0] && rho_hat[k_min] < rho_hat[last] {
                        (Route::Saddle, k_min, true)
                    } else {
                        let k_max = (1..last).max_by(|&a, &b| q[a].total_cmp(&q[b])).expect("interior points");
                        (Route::Deformation, k_max, q[k_max] > q[0] && q[k_max] > q[last])
                    };
                    pair.route = route;
                    pair.p_star = pair.p[k_star];
                    pair.rho_star = rho_hat[k_star];
                    pair.q_star = q[k_star];
                    let rep = &mut self.reports[pair.report];
                    rep.route = Some(route);
                    rep.s = Some(k_star as f64 / last as f64);
                    rep.rho_min = Some(rho_hat[k_star]);
                    rep.rho_pro = Some(pro.exact(pair.p_star));
                    rep.enhancement = Some(q[k_star]);
                    if !interior {
                        rep.verdict = Verdict::NoInteriorMinimum;
                        continue;
                    }
                    next.extend_from_slice(&Self::probes(pair, h));
                    active.push(idx);
                }
                self.active = active;
                self.stage = Stage::Transverse;
            }
            Stage::Transverse => {
                let mut marches = Vec::new();
                let active = std::mem::take(&mut self.active);
                for (&idx, off) in active.iter().zip((0..).step_by(4)) {
                    let pair = &self.pairs[idx];
                    let probes = Self::probes(pair, h);
                    let raw = &rho[off..off + 4];
                    match pair.route {
                        Route::Deformation => {
                            let qv: Vec<f64> = probes.iter().zip(raw).map(|(&p, &v)| pro.ratio(p, v)).collect();
                            let ratio = qv.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / pair.q_star;
                            self.reports[pair.report].transverse_ratio = Some(ratio);
                            if ratio <= 1.0 + EPS_TRANSVERSE {
                                self.finalize(idx, false);
                            } else {
                                self.reports[pair.report].verdict = Verdict::NotATransverseMaximum;
                            }
                        }
                        Route::Saddle => {
                            let vals: Vec<f64> = probes.iter().zip(raw).map(|(&p, &v)| pro.corrected(p, v)).collect();
                            let ratio = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / pair.rho_star;
                            self.reports[pair.report].transverse_ratio = Some(ratio);
                            if ratio <= 1.0 + EPS_TRANSVERSE {
                                self.finalize(idx, false);
                                continue;
                            }
                            // the march: first direction from the probes' transverse gradient
                            let g = add(scale(pair.e1, vals[0] - vals[1]), scale(pair.e2, vals[2] - vals[3]));
                            let gn = norm(g);
                            let dir = if gn > 0.0 {
                                scale(g, 1.0 / gn)
                            } else {
                                // a symmetric rise on both sides: toward the highest probe
                                let best = (0..4).max_by(|&a, &b| vals[a].total_cmp(&vals[b])).expect("four probes");
                                scale(sub(probes[best], pair.p_star), 1.0 / h)
                            };
                            let next_pos = add(pair.p_star, scale(dir, 0.5 * h));
                            let rep_idx = pair.report;
                            if self.near_nucleus(next_pos, h) {
                                let rep = &mut self.reports[rep_idx];
                                rep.march_end = Some(next_pos);
                                rep.march_stop = Some(MarchStop::Nucleus);
                                rep.verdict = Verdict::NotATransverseMaximum;
                                continue;
                            }
                            next.extend_from_slice(&Self::march_points(next_pos, h));
                            marches.push(March { pair: idx, pos: pair.p_star, rho: pair.rho_star, next: next_pos, g0: None, steps: 0 });
                        }
                    }
                }
                self.marches = marches;
                self.stage = if self.marches.is_empty() { Stage::Done } else { Stage::March };
            }
            Stage::March => {
                let marches = std::mem::take(&mut self.marches);
                let mut keep = Vec::new();
                for (mut m, off) in marches.into_iter().zip((0..).step_by(7)) {
                    let pts = Self::march_points(m.next, h);
                    let v: Vec<f64> = pts.iter().zip(&rho[off..off + 7]).map(|(&p, &x)| pro.corrected(p, x)).collect();
                    let rep_idx = self.pairs[m.pair].report;
                    // (ii) ρ̂ decreased: a non-nuclear maximum lies between pos and next
                    if v[0] < m.rho {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(m.pos);
                        rep.march_stop = Some(MarchStop::Maximum);
                        self.finalize(m.pair, true);
                        continue;
                    }
                    let g = [(v[1] - v[2]) / h, (v[3] - v[4]) / h, (v[5] - v[6]) / h];
                    let gm = norm(g);
                    m.steps += 1;
                    // (iii) the gradient fell below f_g of its first value: a stationary point
                    let stationary = match m.g0 {
                        None => {
                            m.g0 = Some(gm);
                            gm == 0.0
                        }
                        Some(g0) => gm < MARCH_STATIONARY_FRACTION * g0,
                    };
                    if stationary {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(m.next);
                        rep.march_stop = Some(MarchStop::Stationary);
                        self.finalize(m.pair, true);
                        continue;
                    }
                    m.pos = m.next;
                    m.rho = v[0];
                    let next_pos = add(m.pos, scale(g, 0.5 * h / gm));
                    // (i) the next point is a nucleus's: the rise was its hill
                    if self.near_nucleus(next_pos, h) {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(next_pos);
                        rep.march_stop = Some(MarchStop::Nucleus);
                        rep.verdict = Verdict::NotATransverseMaximum;
                        continue;
                    }
                    // (iv) the cap
                    if m.steps >= cap {
                        let rep = &mut self.reports[rep_idx];
                        rep.march_end = Some(m.pos);
                        rep.march_stop = Some(MarchStop::StepCap);
                        rep.verdict = Verdict::NotATransverseMaximum;
                        continue;
                    }
                    m.next = next_pos;
                    next.extend_from_slice(&Self::march_points(next_pos, h));
                    keep.push(m);
                }
                self.marches = keep;
                if self.marches.is_empty() {
                    self.stage = Stage::Done;
                }
            }
        }
        self.points = next;
        if self.points.is_empty() {
            self.stage = Stage::Done;
            return None;
        }
        Some(&self.points)
    }

    /// The bonds (the `Bonded` reports) and the molecules of `n` nuclei at `step`/`sim_time`.
    pub fn graph(&self, n: usize, step: u64, sim_time: f64) -> BondGraph {
        let bonds: Vec<Bond> = self
            .reports
            .iter()
            .filter_map(|r| match r.verdict {
                Verdict::Bonded { multi_centre } => Some(Bond {
                    i: r.i,
                    j: r.j,
                    rho_bcp: r.rho_min.expect("a bonded pair has a bond point"),
                    enhancement: r.enhancement.expect("a bonded pair has its ratio"),
                    s: r.s.expect("a bonded pair has its s"),
                    route: r.route.expect("a bonded pair has its route"),
                    multi_centre,
                }),
                _ => None,
            })
            .collect();
        let components = components(n, &bonds);
        BondGraph { bonds, components, step, sim_time, n_nuclei: n }
    }
}

/// The connected components of `n` nuclei under `bonds` (union–find): each sorted, listed by
/// their smallest member; isolated nuclei are one-atom components.
pub fn components(n: usize, bonds: &[Bond]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for b in bonds {
        if b.i >= n || b.j >= n {
            continue;
        }
        let (a, c) = (find(&mut parent, b.i), find(&mut parent, b.j));
        if a != c {
            parent[a.max(c)] = a.min(c);
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for x in 0..n {
        let root = find(&mut parent, x);
        match groups.iter_mut().find(|g| find(&mut parent.clone(), g[0]) == root) {
            Some(g) => g.push(x),
            None => groups.push(vec![x]),
        }
    }
    groups.sort_by_key(|g| g[0]);
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::references::{shell_nodes, shell_radius};
    use crate::test_providers::{Gaussians, GriddedGaussians, Term};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const H: f64 = 0.25;

    /// A term written as amp·exp(−r²/a) (the fixtures' notation; `Term` stores w with 2w² = a).
    fn g(c: [f64; 3], amp: f64, a: f64) -> Term {
        Term { centre: c, amp, w: (0.5 * a).sqrt() }
    }

    /// An analytic fixture: the density is Σ terms; the first `nuclei.len()` terms are the atoms
    /// (one per nucleus, centred on it) and make the promolecule.
    #[derive(Clone)]
    struct Fixture {
        terms: Vec<Term>,
        nuclei: Vec<[f64; 3]>,
    }

    impl Fixture {
        fn rho(&self, p: [f64; 3]) -> f64 {
            self.terms.iter().map(|t| t.at(p)).sum()
        }

        fn pro(&self, p: [f64; 3]) -> f64 {
            self.terms[..self.nuclei.len()].iter().map(|t| t.at(p)).sum()
        }

        /// Exact radial tables of the atom terms (fine spacing, a function of r only).
        fn tables(&self) -> Vec<RadialDensity> {
            self.terms[..self.nuclei.len()].iter().map(|t| RadialDensity::from_fn(|r| t.amp * (-r * r / (2.0 * t.w * t.w)).exp(), R_MAX + 2.0, 0.005)).collect()
        }

        fn rotated(&self, about: [f64; 3], m: [[f64; 3]; 3]) -> Fixture {
            let rot = |p: [f64; 3]| {
                let d = sub(p, about);
                add(about, [dot(m[0], d), dot(m[1], d), dot(m[2], d)])
            };
            Fixture { terms: self.terms.iter().map(|t| Term { centre: rot(t.centre), ..*t }).collect(), nuclei: self.nuclei.iter().map(|&p| rot(p)).collect() }
        }

        /// The detector on the exact density (no grid, no correction).
        fn detect(&self) -> Detector {
            let tables = self.tables();
            let pro = Promolecule::new(tables.iter().collect(), &self.nuclei, None);
            let mut det = Detector::new(&self.nuclei);
            let mut pts = det.points().to_vec();
            let mut rounds = 0;
            while !pts.is_empty() {
                let rho: Vec<f64> = pts.iter().map(|&p| self.rho(p)).collect();
                rounds += 1;
                assert!(rounds < 200, "the rounds do not end");
                match det.feed(&rho, &pro, H) {
                    Some(next) => pts = next.to_vec(),
                    None => break,
                }
            }
            det
        }
    }

    /// Rodrigues' rotation about the unit `axis` by `angle`.
    fn rotation(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
        let a = scale(axis, 1.0 / norm(axis));
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        [
            [c + a[0] * a[0] * t, a[0] * a[1] * t - a[2] * s, a[0] * a[2] * t + a[1] * s],
            [a[1] * a[0] * t + a[2] * s, c + a[1] * a[1] * t, a[1] * a[2] * t - a[0] * s],
            [a[2] * a[0] * t - a[1] * s, a[2] * a[1] * t + a[0] * s, c + a[2] * a[2] * t],
        ]
    }

    /// The independent fine scan of a pair (MUST-4): 501 chord points, the transverse rise as the
    /// maximum over 72 directions around the chord at radius h, a 0.02 Bohr gradient ascent with
    /// analytic central differences and the four stop rules.
    #[derive(Clone, Debug)]
    struct Scan {
        route: Route,
        verdict: Verdict,
        s: f64,
        rho_min: f64,
        enhancement: f64,
        transverse: f64,
        march_end: Option<[f64; 3]>,
        march_stop: Option<MarchStop>,
    }

    fn grad(f: &dyn Fn([f64; 3]) -> f64, p: [f64; 3]) -> [f64; 3] {
        let d = 1e-5;
        std::array::from_fn(|a| {
            let mut hi = p;
            let mut lo = p;
            hi[a] += d;
            lo[a] -= d;
            (f(hi) - f(lo)) / (2.0 * d)
        })
    }

    /// The scan of pair (i, j) of `fx` with the fixture's own promolecule (its atom terms).
    fn scan_pair(fx: &Fixture, i: usize, j: usize, h: f64) -> Option<Scan> {
        scan_view(&ScanView { rho: &|p| fx.rho(p), pro: &|p| fx.pro(p), nuclei: &fx.nuclei }, i, j, h)
    }

    fn bonded_pair(bump: f64) -> Fixture {
        let (a, b) = ([-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]);
        Fixture { terms: vec![g(a, 0.3, 0.5), g(b, 0.3, 0.5), g([0.0; 3], bump * 0.3, 0.5)], nuclei: vec![a, b] }
    }

    fn report_of<'a>(det: &'a Detector, i: usize, j: usize) -> &'a PairReport {
        det.reports().iter().find(|r| r.i == i && r.j == j).expect("a report per pair")
    }

    #[test]
    fn a_bonded_pair_passes_all_three_conditions() {
        let fx = bonded_pair(0.5);
        let scan = scan_pair(&fx, 0, 1, H).unwrap();
        println!("scan: {scan:?}");
        assert_eq!((scan.route, scan.verdict), (Route::Saddle, Verdict::Bonded { multi_centre: false }));
        let det = fx.detect();
        let r = report_of(&det, 0, 1);
        println!("report: {r:?}");
        assert_eq!(r.verdict, Verdict::Bonded { multi_centre: false });
        assert_eq!(r.route, Some(Route::Saddle));
        assert!((r.s.unwrap() - 0.5).abs() <= 0.03, "s = {:?}", r.s);
        let e = r.enhancement.unwrap();
        assert!((e / scan.enhancement - 1.0).abs() <= 0.02, "enhancement {e} vs the scan's {}", scan.enhancement);
        assert!((scan.enhancement - 1.899).abs() < 0.01, "the fixture's exact ratio (0.3168/0.1668)");
        assert!(r.transverse_ratio.unwrap() < 1.0, "the bump falls off across the chord");
        let graph = det.graph(2, 7, 1.5);
        assert_eq!(graph.bonds.len(), 1);
        assert_eq!((graph.bonds[0].i, graph.bonds[0].j, graph.bonds[0].route, graph.bonds[0].multi_centre), (0, 1, Route::Saddle, false));
        assert_eq!((graph.step, graph.sim_time, graph.n_nuclei), (7, 1.5, 2));
        assert_eq!(graph.components, vec![vec![0, 1]]);
    }

    /// ρ = g_A + g_B − 0.42·g(mid): ρ(mid) ≈ 0.041 > ρ_floor with q ≈ 0.25 — the saddle and the
    /// floor pass, the accumulation does not; every number is in the report.
    #[test]
    fn a_depleted_overlap_is_not_a_bond() {
        let fx = bonded_pair(-0.42);
        let scan = scan_pair(&fx, 0, 1, H).unwrap();
        println!("scan: {scan:?}");
        assert_eq!(scan.verdict, Verdict::NotAccumulated);
        let det = fx.detect();
        let r = report_of(&det, 0, 1);
        println!("report: {r:?}");
        assert_eq!(r.verdict, Verdict::NotAccumulated);
        assert_eq!(r.route, Some(Route::Saddle));
        let (rho_min, e) = (r.rho_min.unwrap(), r.enhancement.unwrap());
        assert!(rho_min > RHO_FLOOR && (rho_min - 0.0408).abs() < 0.002, "ρ̂(p*) = {rho_min}");
        assert!((e - 0.245).abs() < 0.01, "enhancement {e}");
        assert!(r.transverse_ratio.unwrap() < 1.0);
        assert!(det.graph(2, 0, 0.0).bonds.is_empty());
        assert_eq!(det.graph(2, 0, 0.0).components, vec![vec![0], vec![1]]);
    }

    fn water_like() -> Fixture {
        let o = [0.0; 3];
        let h1 = [1.43, 1.11, 0.0];
        let h2 = [-1.43, 1.11, 0.0];
        let bump = |h: [f64; 3]| g(scale(h, 0.55), 0.12, 0.405);
        Fixture {
            terms: vec![g(o, 6.0, 0.18), g(h1, 0.3, 0.5), g(h2, 0.3, 0.5), g(o, 0.5, 1.62), bump(h1), bump(h2)],
            nuclei: vec![o, h1, h2],
        }
    }

    /// The promolecule of the water-like fixture is O's core + valence and the two H's (the O
    /// valence is the fixture's fourth term, so `Fixture::pro` cannot be used): one O table with
    /// both terms, and the H table.
    fn water_tables() -> Vec<RadialDensity> {
        let o_table = RadialDensity::from_fn(|r| 6.0 * (-r * r / 0.18).exp() + 0.5 * (-r * r / 1.62).exp(), R_MAX + 2.0, 0.005);
        let h_table = RadialDensity::from_fn(|r| 0.3 * (-r * r / 0.5).exp(), R_MAX + 2.0, 0.005);
        vec![o_table, h_table.clone(), h_table]
    }

    fn water_pro(fx: &Fixture, p: [f64; 3]) -> f64 {
        fx.terms[..4].iter().map(|t| t.at(p)).sum()
    }

    /// The water-like fixture through the detector with its own promolecule (O core + valence).
    fn detect_water(fx: &Fixture) -> Detector {
        let tables = water_tables();
        let pro = Promolecule::new(tables.iter().collect(), &fx.nuclei, None);
        let mut det = Detector::new(&fx.nuclei);
        let mut pts = det.points().to_vec();
        while !pts.is_empty() {
            let rho: Vec<f64> = pts.iter().map(|&p| fx.rho(p)).collect();
            match det.feed(&rho, &pro, H) {
                Some(next) => pts = next.to_vec(),
                None => break,
            }
        }
        det
    }

    /// The scan of the water-like fixture with the full O promolecule (core + valence).
    fn scan_water(fx: &Fixture, i: usize, j: usize) -> Scan {
        scan_view(&ScanView { rho: &|p| fx.rho(p), pro: &|p| water_pro(fx, p), nuclei: &fx.nuclei }, i, j, H).unwrap()
    }

    /// A density, its promolecule and the nuclei, for the scan.
    struct ScanView<'a> {
        rho: &'a dyn Fn([f64; 3]) -> f64,
        pro: &'a dyn Fn([f64; 3]) -> f64,
        nuclei: &'a [[f64; 3]],
    }

    /// The independent fine scan (struct `Scan`'s doc) of pair (i, j).
    fn scan_view(v: &ScanView<'_>, i: usize, j: usize, h: f64) -> Option<Scan> {
        let (a, b) = (v.nuclei[i], v.nuclei[j]);
        let d = sub(b, a);
        let r = norm(d);
        if !(r < R_MAX) {
            return None;
        }
        let u = scale(d, 1.0 / r);
        let n = 500;
        let pts: Vec<[f64; 3]> = (0..=n).map(|k| add(a, scale(d, k as f64 / n as f64))).collect();
        let rho: Vec<f64> = pts.iter().map(|&p| (v.rho)(p)).collect();
        let q: Vec<f64> = pts.iter().map(|&p| (v.rho)(p) / (v.pro)(p)).collect();
        let (e1, e2) = transverse_basis(u, add(a, scale(d, 0.5)), &[]);
        let circle = |f: &dyn Fn([f64; 3]) -> f64, c: [f64; 3]| -> f64 {
            (0..72).map(|m| {
                let th = 2.0 * std::f64::consts::PI * m as f64 / 72.0;
                f(add(c, scale(add(scale(e1, th.cos()), scale(e2, th.sin())), h)))
            }).fold(f64::NEG_INFINITY, f64::max)
        };
        let k_min = (1..n).min_by(|&x, &y| rho[x].total_cmp(&rho[y])).unwrap();
        let (route, k) = if rho[k_min] < rho[0] && rho[k_min] < rho[n] {
            (Route::Saddle, k_min)
        } else {
            let k_max = (1..n).max_by(|&x, &y| q[x].total_cmp(&q[y])).unwrap();
            if !(q[k_max] > q[0] && q[k_max] > q[n]) {
                return Some(Scan { route: Route::Deformation, verdict: Verdict::NoInteriorMinimum, s: k_max as f64 / n as f64, rho_min: rho[k_max], enhancement: q[k_max], transverse: f64::NAN, march_end: None, march_stop: None });
            }
            (Route::Deformation, k_max)
        };
        let p = pts[k];
        let mut scan = Scan { route, verdict: Verdict::TooFar, s: k as f64 / n as f64, rho_min: rho[k], enhancement: q[k], transverse: f64::NAN, march_end: None, march_stop: None };
        let mut multi = false;
        match route {
            Route::Deformation => {
                let qf = |x: [f64; 3]| (v.rho)(x) / (v.pro)(x);
                scan.transverse = circle(&qf, p) / q[k];
                if scan.transverse > 1.0 + EPS_TRANSVERSE {
                    scan.verdict = Verdict::NotATransverseMaximum;
                    return Some(scan);
                }
            }
            Route::Saddle => {
                let rf = |x: [f64; 3]| (v.rho)(x);
                scan.transverse = circle(&rf, p) / rho[k];
                if scan.transverse > 1.0 + EPS_TRANSVERSE {
                    let step = 0.02;
                    let g = grad(&rf, p);
                    let mut dir = sub(g, scale(u, dot(g, u)));
                    dir = scale(dir, 1.0 / norm(dir));
                    let mut pos = p;
                    let mut val = rho[k];
                    let mut g0: Option<f64> = None;
                    let cap = (2.0 * R_MAX / step) as usize;
                    let mut steps = 0;
                    loop {
                        let next = add(pos, scale(dir, step));
                        if v.nuclei.iter().any(|&nn| dist(next, nn) < h) {
                            scan.march_end = Some(next);
                            scan.march_stop = Some(MarchStop::Nucleus);
                            scan.verdict = Verdict::NotATransverseMaximum;
                            return Some(scan);
                        }
                        let val_next = rf(next);
                        if val_next < val {
                            scan.march_end = Some(pos);
                            scan.march_stop = Some(MarchStop::Maximum);
                            multi = true;
                            break;
                        }
                        let gg = grad(&rf, next);
                        let gm = norm(gg);
                        steps += 1;
                        let stationary = match g0 {
                            None => {
                                g0 = Some(gm);
                                gm == 0.0
                            }
                            Some(g0) => gm < MARCH_STATIONARY_FRACTION * g0,
                        };
                        if stationary {
                            scan.march_end = Some(next);
                            scan.march_stop = Some(MarchStop::Stationary);
                            multi = true;
                            break;
                        }
                        if steps >= cap {
                            scan.march_end = Some(next);
                            scan.march_stop = Some(MarchStop::StepCap);
                            scan.verdict = Verdict::NotATransverseMaximum;
                            return Some(scan);
                        }
                        pos = next;
                        val = val_next;
                        dir = scale(gg, 1.0 / gm);
                    }
                }
            }
        }
        scan.verdict = if rho[k] < RHO_FLOOR {
            Verdict::BelowFloor
        } else if q[k] < 1.0 + DELTA {
            Verdict::NotAccumulated
        } else {
            Verdict::Bonded { multi_centre: multi }
        };
        Some(scan)
    }

    /// Water-like: the O–H chords are monotone (Route D, the bond point at q's interior maximum,
    /// bonded); the H–H chord has an interior minimum whose transverse rise (toward O) is followed
    /// by the march into O's hill → `NotATransverseMaximum`. The expected numbers are the scan's.
    #[test]
    fn a_third_atom_does_not_bond_two_neighbours_and_the_x_h_bonds_take_the_deformation_route() {
        let fx = water_like();
        let oh = scan_water(&fx, 0, 1);
        let hh = scan_water(&fx, 1, 2);
        println!("scan O–H: {oh:?}\nscan H–H: {hh:?}");
        // the brief's pre-computation, re-derived here
        assert_eq!(oh.route, Route::Deformation, "the O–H chord is monotone: ends {:.3}/{:.3}", fx.rho(fx.nuclei[0]), fx.rho(fx.nuclei[1]));
        assert!((fx.rho(fx.nuclei[0]) - 6.52).abs() < 0.02 && (fx.rho(fx.nuclei[1]) - 0.39).abs() < 0.01);
        assert!((oh.enhancement - 1.33).abs() < 0.02 && (oh.s - 0.59).abs() < 0.02, "q max {} at s {}", oh.enhancement, oh.s);
        assert!(oh.transverse < 1.0 + EPS_TRANSVERSE, "{}", oh.transverse);
        assert_eq!(oh.verdict, Verdict::Bonded { multi_centre: false });
        assert_eq!(hh.route, Route::Saddle);
        assert!((hh.rho_min - 0.278).abs() < 0.003 && (hh.s - 0.5).abs() < 1e-9, "H–H minimum {} at {}", hh.rho_min, hh.s);
        assert!((hh.transverse - 1.68).abs() < 0.03, "transverse {}", hh.transverse);
        assert_eq!((hh.verdict, hh.march_stop), (Verdict::NotATransverseMaximum, Some(MarchStop::Nucleus)));
        let end = hh.march_end.unwrap();
        assert!(dist(end, fx.nuclei[0]) < H, "the scan's march ended {} from O", dist(end, fx.nuclei[0]));

        let det = detect_water(&fx);
        for r in det.reports() {
            println!("report: {r:?}");
        }
        let r_oh1 = report_of(&det, 0, 1);
        let r_oh2 = report_of(&det, 0, 2);
        let r_hh = report_of(&det, 1, 2);
        for r in [r_oh1, r_oh2] {
            assert_eq!((r.route, r.verdict), (Some(Route::Deformation), Verdict::Bonded { multi_centre: false }));
            assert!((r.enhancement.unwrap() / oh.enhancement - 1.0).abs() <= 0.02, "{:?}", r.enhancement);
            assert!((r.s.unwrap() - oh.s).abs() <= 0.04);
        }
        assert_eq!(r_hh.verdict, hh.verdict);
        assert_eq!((r_hh.route, r_hh.march_stop), (Some(Route::Saddle), Some(MarchStop::Nucleus)));
        let end = r_hh.march_end.unwrap();
        let d_o = dist(end, fx.nuclei[0]);
        println!("detector H–H march: end {end:?}, {d_o:.3} Bohr from O");
        assert!(d_o < H && (d_o - 0.24).abs() < 0.03, "{d_o}");
        let graph = det.graph(3, 0, 0.0);
        let bonds: Vec<(usize, usize)> = graph.bonds.iter().map(|b| (b.i, b.j)).collect();
        assert_eq!(bonds, vec![(0, 1), (0, 2)]);
        assert!(graph.bonds.iter().all(|b| b.route == Route::Deformation));
        assert_eq!(graph.components, vec![vec![0, 1, 2]]);
    }

    fn ring(bump: f64) -> Fixture {
        let side = 2.117;
        let r = side / 3f64.sqrt();
        let v: Vec<[f64; 3]> = (0..3).map(|k| {
            let th = std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI * k as f64 / 3.0;
            [r * th.cos(), r * th.sin(), 0.0]
        }).collect();
        let mut terms: Vec<Term> = v.iter().map(|&p| g(p, 0.3, 0.72)).collect();
        terms.push(g([0.0; 3], bump, 1.28));
        Fixture { terms, nuclei: v }
    }

    /// H₃⁺-like: three atoms on a triangle of side 2.117 plus a bump at the centroid. At bump 0.25
    /// the +11 % rise toward the centroid exceeds ε_⊥ and the march runs: it stalls on the ring
    /// point (the stationary stop) and every edge is `Bonded { multi_centre: true }`. At a bump
    /// whose rise is inside ε_⊥ the plain saddle route applies (`multi_centre: false`), the
    /// components unchanged. The brief named 0.15 for that run; its rise is +8.0 % (measured
    /// below, printed), above ε_⊥ = 0.07, so the march runs there too — 0.10 (+5.6 %) is the
    /// inside-ε run, and 0.15 is checked against the scan like the others.
    #[test]
    fn a_three_centre_ring_is_kept_by_the_march() {
        let fx = ring(0.25);
        let mid = add(fx.nuclei[0], scale(sub(fx.nuclei[1], fx.nuclei[0]), 0.5));
        let centroid = [0.0; 3];
        println!("ρ(edge mid) {:.4}, ρ(centroid) {:.4}, ratio {:.3}, q(edge mid) {:.3}", fx.rho(mid), fx.rho(centroid), fx.rho(centroid) / fx.rho(mid), fx.rho(mid) / fx.pro(mid));
        assert!((fx.rho(mid) - 0.316).abs() < 0.002 && (fx.rho(centroid) / fx.rho(mid) - 1.148).abs() < 0.005);
        let scan = scan_pair(&fx, 0, 1, H).unwrap();
        println!("scan edge (0,1): {scan:?}");
        assert!((scan.transverse - 1.111).abs() < 0.005, "{}", scan.transverse);
        assert_eq!(scan.verdict, Verdict::Bonded { multi_centre: true });
        assert!((scan.enhancement - 2.44).abs() < 0.02);
        let det = fx.detect();
        let graph = det.graph(3, 0, 0.0);
        for r in det.reports() {
            println!("report: {r:?}");
        }
        assert_eq!(graph.bonds.len(), 3);
        for r in det.reports() {
            assert_eq!(r.verdict, Verdict::Bonded { multi_centre: true }, "{r:?}");
            assert!((r.enhancement.unwrap() / scan.enhancement - 1.0).abs() <= 0.03);
            let end = r.march_end.unwrap();
            let d_c = dist(end, centroid);
            let d_n = fx.nuclei.iter().map(|&p| dist(end, p)).fold(f64::INFINITY, f64::min);
            println!("  march end {end:?}: {d_c:.3} from the centroid, {d_n:.3} from the nearest nucleus ({:?})", r.march_stop);
            assert!(d_c <= 0.2 && d_n >= 1.0, "{d_c} {d_n}");
        }
        assert_eq!(graph.components, vec![vec![0, 1, 2]]);
        // the rise inside ε_⊥: the saddle route alone
        for (bump, expect_march) in [(0.10, false), (0.15, true)] {
            let fx = ring(bump);
            let scan = scan_pair(&fx, 0, 1, H).unwrap();
            let det = fx.detect();
            let graph = det.graph(3, 0, 0.0);
            println!("bump {bump}: rise {:.3} (scan {:.3}), scan {:?}", report_of(&det, 0, 1).transverse_ratio.unwrap(), scan.transverse, scan.verdict);
            assert_eq!(graph.bonds.len(), 3);
            assert_eq!(scan.transverse > 1.0 + EPS_TRANSVERSE, expect_march, "bump {bump}: rise {}", scan.transverse);
            for r in det.reports() {
                assert_eq!(r.verdict, scan.verdict, "bump {bump}: {r:?}");
                assert_eq!(r.verdict, Verdict::Bonded { multi_centre: expect_march });
            }
            assert_eq!(graph.components, vec![vec![0, 1, 2]]);
        }
    }

    /// R-1: the water-like fixture rotated about its H–H axis (45°, 90°) and the ring tilted
    /// 45° about x, plus both under one random rotation: every verdict identical to the
    /// axis-aligned run, `rho_min` and `enhancement` equal to 1e-9 (the chord samples are
    /// rotation-covariant), the march end points equal under the rotation to 0.05 Bohr.
    #[test]
    fn the_march_is_rotation_invariant() {
        let mut rng = StdRng::seed_from_u64(7);
        let random_axis: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
        let random_angle: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let random = rotation(random_axis, random_angle);
        let compare = |name: &str, base: &Detector, rot: &Detector, m: [[f64; 3]; 3], about: [f64; 3]| {
            let map = |p: [f64; 3]| {
                let d = sub(p, about);
                add(about, [dot(m[0], d), dot(m[1], d), dot(m[2], d)])
            };
            for (a, b) in base.reports().iter().zip(rot.reports()) {
                assert_eq!((a.i, a.j, a.verdict, a.route, a.march_stop), (b.i, b.j, b.verdict, b.route, b.march_stop), "{name}: {a:?} vs {b:?}");
                for (x, y) in [(a.rho_min, b.rho_min), (a.enhancement, b.enhancement), (a.s, b.s)] {
                    match (x, y) {
                        (Some(x), Some(y)) => assert!((x - y).abs() <= 1e-9 * x.abs().max(1.0), "{name}: {x} vs {y}"),
                        (None, None) => {}
                        other => panic!("{name}: {other:?}"),
                    }
                }
                if let (Some(ea), Some(eb)) = (a.march_end, b.march_end) {
                    let d = dist(map(ea), eb);
                    println!("{name} pair ({}, {}): march end {ea:?} → {eb:?}, |Δ| = {d:.4}", a.i, a.j);
                    assert!(d <= 0.05, "{name}: the march ends differ by {d}");
                }
            }
        };
        let w = water_like();
        let base = detect_water(&w);
        let hh_axis = sub(w.nuclei[2], w.nuclei[1]);
        for (name, m) in [("water 45° about H–H", rotation(hh_axis, std::f64::consts::FRAC_PI_4)), ("water 90° about H–H", rotation(hh_axis, std::f64::consts::FRAC_PI_2)), ("water random", random)] {
            let r = w.rotated(w.nuclei[1], m);
            let det = detect_water(&r);
            compare(name, &base, &det, m, w.nuclei[1]);
        }
        let ring_fx = ring(0.25);
        let base = ring_fx.detect();
        for (name, m) in [("ring tilted 45° about x", rotation([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_4)), ("ring random", random)] {
            let r = ring_fx.rotated([0.0; 3], m);
            let det = r.detect();
            compare(name, &base, &det, m, [0.0; 3]);
            assert!(det.reports().iter().all(|r| r.verdict == Verdict::Bonded { multi_centre: true }));
        }
    }

    #[test]
    fn below_the_floor_is_not_a_bond_and_far_pairs_are_not_sampled() {
        let near = Fixture { terms: vec![g([-2.5, 0.0, 0.0], 0.3, 0.5), g([2.5, 0.0, 0.0], 0.3, 0.5)], nuclei: vec![[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]] };
        let det = near.detect();
        let r = report_of(&det, 0, 1);
        println!("5 Bohr: {r:?}");
        assert_eq!(r.verdict, Verdict::BelowFloor);
        assert_eq!(r.verdict, scan_pair(&near, 0, 1, H).unwrap().verdict);
        assert!(r.rho_min.is_some_and(|x| x < RHO_FLOOR && x > 0.0));
        let far = Fixture { terms: vec![g([-4.5, 0.0, 0.0], 0.3, 0.5), g([4.5, 0.0, 0.0], 0.3, 0.5)], nuclei: vec![[-4.5, 0.0, 0.0], [4.5, 0.0, 0.0]] };
        let det = Detector::new(&far.nuclei);
        assert!(det.points().is_empty(), "no chord points for a pair beyond R_MAX");
        let det = far.detect();
        let r = report_of(&det, 0, 1);
        assert_eq!(*r, PairReport { i: 0, j: 1, route: None, s: None, rho_min: None, transverse_ratio: None, rho_pro: None, enhancement: None, march_end: None, march_stop: None, verdict: Verdict::TooFar });
        assert!(scan_pair(&far, 0, 1, H).is_none());
        assert_eq!(det.graph(2, 0, 0.0).components, vec![vec![0], vec![1]]);
    }

    /// MUST-1 / R-2: the bonded-pair fixture on a 49³ h = 0.25 grid, sampled trilinearly, with the
    /// promolecule from shell-binned tables of the same gridded atoms, over 24 placements (the
    /// pair shifted by 0, h/4, h/2, 3h/4 along x, in 6 orientations). The spreads are measured
    /// and recorded (`references.md`, "Tools (M6)").
    ///
    /// Ceilings: the corrected q spread must lie within the residual the same-operator
    /// derivation leaves — (h²/8)·|∇²ρ/ρ − ∇²ρ_pro/ρ_pro|·q at the midpoint, the difference
    /// between a cell centre and a node (0.072 for this fixture, computed below from the
    /// Gaussians' Laplacians; the re-review's run measured 0.051 and the brief wrote 0.06 from
    /// it — the measured 0.063 is above that single run and below the derivation's bound);
    /// |mean − exact| ≤ 0.06 with the mean BELOW the exact value (the safe direction); the raw
    /// ratio's spread ≥ 0.10 (what the correction removes); the transverse ratio's spread ≤ 0.03.
    /// Then the constants' rule: 3× the relative q spread ≤ δ and 3× the transverse spread ≤
    /// ε_⊥ — if either fails the constant is raised to 3× the measurement, never lowered (δ was
    /// raised from 0.10 to 0.11 by this measurement: 3 × 0.0629/1.852 = 0.102).
    ///
    /// What the per-placement table (printed) shows: the spread is dominated not by the
    /// residual bias at a fixed point but by the sampled minimum wandering ±1–2 chord samples
    /// (s = 0.4375 … 0.5312) along this fixture's flat bottom, where q itself varies by 2.4 %
    /// over 0.1 Bohr — a property of sampling a flat minimum at 33 points, which a real H₂ chord
    /// shares (Task 2: 0.1853 at the midpoint vs 0.1861 one sample away).
    #[test]
    fn the_same_operator_cancels_the_interpolation_bias() {
        let cell = Cell { n: 49, h: H };
        let (amp, a): (f64, f64) = (0.3, 0.5);
        let g_at = |r2: f64, amp: f64| amp * (-r2 / a).exp();
        let lap = |r2: f64, amp: f64| g_at(r2, amp) * (4.0 * r2 / (a * a) - 6.0 / a);
        let rho_pro_mid = 2.0 * g_at(0.64, amp);
        let rho_mid = rho_pro_mid + 0.5 * amp;
        let exact_q = rho_mid / rho_pro_mid;
        let lap_pro = 2.0 * lap(0.64, amp);
        let lap_rho = lap_pro + lap(0.0, 0.5 * amp);
        let bound_q = H * H / 8.0 * (lap_rho / rho_mid - lap_pro / rho_pro_mid).abs() * exact_q;
        // the atom's table from the gridded isolated atom at the cell centre
        let nodes = shell_nodes(&cell, shell_radius(&cell));
        let atom = g([0.0; 3], amp, a);
        let values: Vec<f64> = nodes.iter().map(|&p| atom.at(p)).collect();
        let table = RadialDensity::from_shells(&nodes, &values, [0.0; 3], 0.5 * H);
        let orientations: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];
        let (mut qc, mut qr, mut tr) = (Vec::new(), Vec::new(), Vec::new());
        for shift in [0.0, 0.25 * H, 0.5 * H, 0.75 * H] {
            for o in &orientations {
                let u = scale(*o, 1.0 / norm(*o));
                let c = [shift, 0.0, 0.0];
                let (ra, rb) = (sub(c, scale(u, 0.8)), add(c, scale(u, 0.8)));
                let terms = vec![g(ra, amp, a), g(rb, amp, a), g(c, 0.5 * amp, a)];
                let gridded = GriddedGaussians::new(terms, cell.clone());
                let pro = Promolecule::new(vec![&table, &table], &[ra, rb], Some(&cell));
                let pro_exact = Promolecule::new(vec![&table, &table], &[ra, rb], None);
                let mut det = Detector::new(&[ra, rb]);
                let mut pts = det.points().to_vec();
                while !pts.is_empty() {
                    let rho = gridded.sample(&pts);
                    match det.feed(&rho, &pro, H) {
                        Some(next) => pts = next.to_vec(),
                        None => break,
                    }
                }
                let r = &det.reports()[0];
                assert_eq!(r.verdict, Verdict::Bonded { multi_centre: false }, "{r:?}");
                let p_star = add(ra, scale(sub(rb, ra), r.s.unwrap()));
                let raw = gridded.sample(&[p_star])[0] / pro_exact.exact(p_star);
                println!("  shift {shift:.4} along {:?}: s {:.4}, q {:.4}, raw {:.4}, transverse {:.4}", o, r.s.unwrap(), r.enhancement.unwrap(), raw, r.transverse_ratio.unwrap());
                qc.push(r.enhancement.unwrap());
                qr.push(raw);
                tr.push(r.transverse_ratio.unwrap());
            }
        }
        let spread = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (sq, sr, st) = (spread(&qc), spread(&qr), spread(&tr));
        let (mq, mr, mt) = (mean(&qc), mean(&qr), mean(&tr));
        println!("24 placements, 49³ h 0.25, bonded pair (exact q = {exact_q:.4}):");
        println!("  corrected q: spread {sq:.4}, mean {mq:.4} (exact − mean = {:.4}, {:.2} %)", exact_q - mq, 100.0 * (exact_q - mq) / exact_q);
        println!("  raw ratio ρ_tri/ρ_pro^exact: spread {sr:.4}, mean {mr:.4}");
        println!("  transverse ratio: spread {st:.4}, mean {mt:.4}");
        println!("  3 × q spread / mean = {:.4} (δ = {DELTA}); 3 × transverse spread = {:.4} (ε_⊥ = {EPS_TRANSVERSE})", 3.0 * sq / mq, 3.0 * st);
        println!("  the derivation's residual at the midpoint: (h²/8)·|∇²ρ/ρ − ∇²ρ_pro/ρ_pro|·q = {bound_q:.4} (re-review's run: 0.051; its ceiling 0.06)");
        assert!(sq <= bound_q, "corrected q spread {sq} beyond the second-order residual {bound_q}");
        assert!((mq - exact_q).abs() <= 0.06 && mq < exact_q, "mean {mq} vs exact {exact_q}");
        assert!(sr >= 0.10, "raw spread {sr}: the correction has nothing to remove?");
        assert!(st <= 0.03, "transverse spread {st}");
        assert!(3.0 * sq / mq <= DELTA, "δ must be raised to {:.3}", 3.0 * sq / mq);
        assert!(3.0 * st <= EPS_TRANSVERSE, "ε_⊥ must be raised to {:.3}", 3.0 * st);
    }

    #[test]
    fn corrected_leaves_rho_alone_beyond_the_tables() {
        let cell = Cell { n: 49, h: H };
        let table = RadialDensity::from_fn(|r| 0.3 * (-r * r / 0.5).exp(), 3.0, 0.01);
        let pro = Promolecule::new(vec![&table], &[[0.0; 3]], Some(&cell));
        // outside the grid (10 Bohr from the atom) and inside the grid beyond the table's reach
        for p in [[10.0, 0.0, 0.0], [5.0, 0.0, 0.0], [3.5, 3.5, 0.0]] {
            assert_eq!(pro.trilinear(p), 0.0, "{p:?}");
            assert_eq!(pro.corrected(p, 0.123), 0.123, "{p:?}");
            assert!(pro.corrected(p, 0.123).is_finite());
            assert_eq!(pro.ratio(p, 0.123), 0.0);
        }
        // inside the table: the correction is the ratio of the two operators, ≈ 1 near a node
        let p = [0.5, 0.25, 0.0];
        assert!((pro.corrected(p, 1.0) - 1.0).abs() < 1e-12, "a node is read exactly");
        let q = [0.625, 0.125, 0.0];
        let c = pro.corrected(q, 1.0);
        assert!(c > 1.0 && c < 1.1, "a cell midpoint of a concave profile: trilinear underestimates, so the correction lifts: {c}");
        // ungridded: the identity
        let id = Promolecule::new(vec![&table], &[[0.0; 3]], None);
        assert_eq!(id.corrected(q, 0.7), 0.7);
        assert!((id.ratio(q, 0.7) - 0.7 / id.exact(q)).abs() < 1e-15);
    }

    #[test]
    fn components_are_connected_components() {
        let b = |i, j| Bond { i, j, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: Route::Saddle, multi_centre: false };
        assert_eq!(components(6, &[b(0, 1), b(1, 2), b(3, 4)]), vec![vec![0, 1, 2], vec![3, 4], vec![5]]);
        assert_eq!(components(3, &[]), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(components(4, &[b(2, 3), b(0, 3), b(1, 2)]), vec![vec![0, 1, 2, 3]]);
        assert_eq!(components(0, &[]), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn round_one_has_33_points_per_pair_with_the_endpoints() {
        let fx = Fixture { terms: vec![g([0.0; 3], 0.3, 0.5), g([1.5, 0.0, 0.0], 0.3, 0.5), g([0.0, 1.5, 0.0], 0.3, 0.5)], nuclei: vec![[0.0; 3], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]] };
        let tables = fx.tables();
        let pro = Promolecule::new(tables.iter().collect(), &fx.nuclei, None);
        let mut det = Detector::new(&fx.nuclei);
        assert_eq!(det.points().len(), 3 * 33);
        for (k, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            let p = &det.points()[33 * k..33 * (k + 1)];
            for a in 0..3 {
                assert!((p[0][a] - fx.nuclei[i][a]).abs() <= 1e-15 && (p[32][a] - fx.nuclei[j][a]).abs() <= 1e-15);
            }
        }
        let rho: Vec<f64> = det.points().iter().map(|&p| fx.rho(p)).collect();
        let next = det.feed(&rho, &pro, H).expect("round 2").to_vec();
        let survivors = det.reports().iter().filter(|r| r.verdict == Verdict::TooFar && r.route.is_some()).count();
        assert_eq!(next.len(), 4 * survivors);
        assert!(survivors >= 1, "{:?}", det.reports());
    }

    /// The grid operator mirrored here is `qm::Grid::trilinear_cell` / `trilinear`: equal cells
    /// and fractions, equal blends of a stored field, on random points, nodes and faces.
    #[test]
    fn the_sim_trilinear_mirror_matches_qm() {
        let cell = Cell { n: 9, h: 0.5 };
        let grid = qm::Grid::cube(9, 0.5);
        let f = grid.sample(|[x, y, z]| (x * 1.3 - y * y + z * 0.7 * x).exp());
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..200 {
            let p: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-2.1..2.1));
            assert_eq!(trilinear_cell(&cell, p), grid.trilinear_cell(p), "{p:?}");
            let mine = trilinear_blend(&cell, p, |ijk| f[grid.idx(ijk[0], ijk[1], ijk[2])]);
            let theirs = grid.trilinear(&f, p);
            assert!((mine - theirs).abs() <= 1e-15 * theirs.abs().max(1.0), "{p:?}: {mine} vs {theirs}");
        }
        for idx in [0, grid.idx(8, 8, 8), grid.idx(8, 0, 4), grid.idx(4, 4, 4), grid.idx(0, 8, 8)] {
            let p = grid.point(idx);
            assert_eq!(node_position(&cell, [idx % 9, (idx / 9) % 9, idx / 81]), p);
            assert_eq!(trilinear_blend(&cell, p, |ijk| f[grid.idx(ijk[0], ijk[1], ijk[2])]).to_bits(), f[idx].to_bits());
        }
        assert_eq!(trilinear_blend(&cell, [2.0000001, 0.0, 0.0], |_| 1.0), 0.0);
    }

    /// `Gaussians::sample_density` is exact and `GriddedGaussians` is the trilinear of the node
    /// values — the test providers the fixtures above rely on.
    #[test]
    fn the_test_providers_sample_as_documented() {
        let terms = vec![g([0.3, -0.2, 0.1], 0.4, 0.6)];
        let mut exact = Gaussians::fixed(terms.clone());
        let cell = Cell { n: 17, h: 0.25 };
        let mut gridded = GriddedGaussians::new(terms.clone(), cell.clone());
        gridded.reset();
        let p = [0.37, 0.11, -0.52];
        assert_eq!(exact.sample(&[p])[0], terms[0].at(p));
        let node = node_position(&cell, [9, 7, 8]);
        assert_eq!(gridded.sample(&[node])[0].to_bits(), terms[0].at(node).to_bits());
        let s = gridded.sample(&[p])[0];
        assert!(s != terms[0].at(p) && (s / terms[0].at(p) - 1.0).abs() < 0.05, "trilinear, close: {s} vs {}", terms[0].at(p));
        assert_eq!(gridded.sample(&[[9.0, 0.0, 0.0]])[0], 0.0);
        // through the trait: the exact provider answers after an evaluation
        use crate::provider::ForceProvider;
        let scene = crate::scene::Scene { nuclei: vec![], cell, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3] };
        exact.evaluate(&scene, &mut || false).unwrap();
        assert_eq!(exact.sample_density(&[p]).unwrap()[0], terms[0].at(p));
    }
}
