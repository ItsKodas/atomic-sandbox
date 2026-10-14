//! Tests for `qm::orbitals` (orbital-viewer spec §5; plan Task 1). Pure f64 on small analytic
//! grids; the one SCF test is release-only. Random numbers come from a local SplitMix64 so the
//! crate needs no `rand` dependency; every seed is fixed.

use super::*;
use crate::grid::Grid;
use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
use nalgebra::DMatrix;

// ───────────────────────────── helpers ─────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn sign(&mut self) -> f64 { if self.next() & 1 == 0 { 1.0 } else { -1.0 } }
}

/// Haar-random element of O(d): QR of a Gaussian matrix with the sign of diag(R) fixed.
fn random_orthogonal(rng: &mut Rng, d: usize) -> DMatrix<f64> {
    let g = DMatrix::from_fn(d, d, |_, _| rng.normal());
    let qr = g.qr();
    let (mut q, r) = (qr.q(), qr.r());
    for j in 0..d {
        if r[(j, j)] < 0.0 { for i in 0..d { q[(i, j)] = -q[(i, j)]; } }
    }
    q
}

/// Uniform random proper rotation of R³ from a unit quaternion.
fn random_rotation3(rng: &mut Rng) -> [[f64; 3]; 3] {
    let mut q = [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in q.iter_mut() { *x /= n; }
    let [w, x, y, z] = q;
    [[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
     [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
     [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]]
}

fn normalise(grid: &Grid, f: &mut [f64]) {
    let n = grid.dot(f, f).sqrt();
    for x in f.iter_mut() { *x /= n; }
}

/// Polynomial part P_nl(ρ) of the hydrogenic radial function R_nl = ρ^l P_nl(ρ) e^{−ρ/n}
/// (unnormalised); ρ^l is carried by the solid harmonic.
fn poly(n: u8, l: u8, rho: f64) -> f64 {
    match (n, l) {
        (1, 0) | (2, 1) | (3, 2) | (4, 3) => 1.0,
        (2, 0) => 2.0 - rho,
        (3, 0) => 27.0 - 18.0 * rho + 2.0 * rho * rho,
        (3, 1) => 6.0 - rho,
        _ => panic!("no hydrogenic polynomial for ({n}, {l})"),
    }
}

/// Hydrogenic-shaped orbital ψ(r) = P_nl(ρ) e^{−ρ/n} S_lm(d), ρ = z|d|, d = Qᵀ(r − centre)
/// (Q = identity unless `rot`), normalised on the grid. `lm` indexes `solid_harmonics`.
fn hydrogenic(grid: &Grid, n: u8, l: u8, lm: usize, z: f64, centre: [f64; 3], rot: Option<[[f64; 3]; 3]>) -> Vec<f64> {
    let mut s = [0.0; N_MOMENTS];
    let mut f = grid.sample(|p| {
        let mut d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
        if let Some(q) = rot {
            d = [q[0][0] * d[0] + q[1][0] * d[1] + q[2][0] * d[2],
                 q[0][1] * d[0] + q[1][1] * d[1] + q[2][1] * d[2],
                 q[0][2] * d[0] + q[1][2] * d[1] + q[2][2] * d[2]];
        }
        let rho = z * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        solid_harmonics(d, &mut s);
        poly(n, l, rho) * (-rho / n as f64).exp() * s[lm]
    });
    normalise(grid, &mut f);
    f
}

/// Solid-harmonic indices of the p set in display order (x, y, z).
const P_LM: [usize; 3] = [3, 1, 2];

fn s_orb(grid: &Grid, n: u8, z: f64, c: [f64; 3]) -> Vec<f64> { hydrogenic(grid, n, 0, 0, z, c, None) }
fn p_set(grid: &Grid, n: u8, z: f64, c: [f64; 3]) -> Vec<Vec<f64>> {
    P_LM.iter().map(|&lm| hydrogenic(grid, n, 1, lm, z, c, None)).collect()
}

/// φ_k = Σ_j ψ_j Q_jk on the grid.
fn mix(psis: &[Vec<f64>], q: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..q.ncols()).map(|k| {
        let mut out = vec![0.0; psis[0].len()];
        for (j, psi) in psis.iter().enumerate() {
            let w = q[(j, k)];
            for (o, v) in out.iter_mut().zip(psi) { *o += w * v; }
        }
        out
    }).collect()
}

fn mix_moments(ms: &[Moments], q: &DMatrix<f64>) -> Vec<Moments> {
    (0..q.ncols()).map(|k| {
        let mut out = Moments { b: [0.0; 16], c: [0.0; 16] };
        for (j, m) in ms.iter().enumerate() {
            for i in 0..16 { out.b[i] += q[(j, k)] * m.b[i]; out.c[i] += q[(j, k)] * m.c[i]; }
        }
        out
    }).collect()
}

/// Moments with a single unit entry (b and c) at `lm`: an ideal pure-harmonic orbital.
fn ideal(lm: usize) -> Moments {
    let mut m = Moments { b: [0.0; 16], c: [0.0; 16] };
    m.b[lm] = 1.0;
    m.c[lm] = 1.0;
    m
}

fn atomic(n: u8, l: u8, m: &'static str) -> Label { Label::Atomic { n, l, m } }

/// One channel's display inputs, owned.
#[derive(Clone, Default)]
struct Chan {
    evals: Vec<f64>, occ: Vec<f64>, n_occ: usize,
    moments: Vec<Moments>, radial: Vec<RadialProfile>,
    first_sign: Vec<[f64; 9]>, max_abs: Vec<f64>, outer: Vec<f64>,
}

impl Chan {
    /// From grid orbitals (moments, profiles and functionals about `centre`; nuclei = [centre]).
    fn from_grid(grid: &Grid, psis: &[Vec<f64>], evals: &[f64], n_occ: usize, centre: [f64; 3]) -> Chan {
        let refs: Vec<&[f64]> = psis.iter().map(|p| p.as_slice()).collect();
        Chan {
            evals: evals.to_vec(),
            occ: (0..evals.len()).map(|i| if i < n_occ { 1.0 } else { 0.0 }).collect(),
            n_occ,
            moments: moments_many(grid, &refs, centre),
            radial: radial_profiles(grid, &refs, centre),
            first_sign: psis.iter().map(|p| first_sign_functionals(grid, p, &[centre])).collect(),
            max_abs: psis.iter().map(|p| p.iter().fold(0.0f64, |a, &v| a.max(v.abs()))).collect(),
            outer: psis.iter().map(|p| outer_weight(grid, p)).collect(),
        }
    }
    /// No grid: only energies; neutral functionals (all zero), max|ψ| = 1, outer weight 0.
    fn synthetic(evals: &[f64], n_occ: usize) -> Chan {
        Chan {
            evals: evals.to_vec(),
            occ: (0..evals.len()).map(|i| if i < n_occ { 1.0 } else { 0.0 }).collect(),
            n_occ,
            moments: Vec::new(), radial: Vec::new(),
            first_sign: vec![[0.0; 9]; evals.len()],
            max_abs: vec![1.0; evals.len()],
            outer: vec![0.0; evals.len()],
        }
    }
    /// Appends synthetic unoccupied rows (no grid data).
    fn with_unoccupied(mut self, evals: &[f64]) -> Chan {
        for &e in evals {
            self.evals.push(e); self.occ.push(0.0);
            self.first_sign.push([0.0; 9]); self.max_abs.push(1.0); self.outer.push(0.0);
        }
        self
    }
    fn truncated(&self, n: usize) -> Chan {
        let mut c = self.clone();
        c.evals.truncate(n); c.occ.truncate(n); c.first_sign.truncate(n); c.max_abs.truncate(n); c.outer.truncate(n);
        c
    }
}

struct Case {
    ch: [Chan; 2],
    prev: [Option<DMatrix<f64>>; 2],
    cross: Option<DMatrix<f64>>,
    lone: bool,
    n_core: [u8; 4],
    z_sigma: Option<f64>,
    tol: f64,
}

impl Case {
    fn new(up: Chan, dn: Chan, lone: bool) -> Case {
        Case { ch: [up, dn], prev: [None, None], cross: None, lone, n_core: [0; 4], z_sigma: None, tol: 1e-5 }
    }
    fn input(&self) -> UpdateInput<'_> {
        let [u, d] = &self.ch;
        UpdateInput {
            evals: [&u.evals, &d.evals],
            occupations: [&u.occ, &d.occ],
            overlaps_prev: [self.prev[0].as_ref(), self.prev[1].as_ref()],
            cross: self.cross.as_ref(),
            moments: if self.lone { Some([&u.moments, &d.moments]) } else { None },
            radial: if self.lone { Some([&u.radial, &d.radial]) } else { None },
            first_sign: [&u.first_sign, &d.first_sign],
            max_abs: [&u.max_abs, &d.max_abs],
            outer_weight: [&u.outer, &d.outer],
            n_occ: [u.n_occ, d.n_occ],
            n_computed: [u.evals.len(), d.evals.len()],
            n_core: self.n_core,
            z_sigma: self.z_sigma,
            tol_scf: self.tol,
        }
    }
}

fn rows(out: &UpdateOutput, spin: Spin) -> Vec<&Row> { out.rows.iter().filter(|r| r.spin == spin).collect() }
fn labels(out: &UpdateOutput, spin: Spin) -> Vec<Label> { rows(out, spin).iter().map(|r| r.label.clone()).collect() }

/// Off-grid nucleus used throughout (the plan's (0.37h, −0.21h, 0.12h) at h = 0.25).
const OFF: [f64; 3] = [0.37 * 0.25, -0.21 * 0.25, 0.12 * 0.25];

/// Ne-like closed channel 1s, 2s, 2p×3 (Z = 2 shapes, compact in a 12 Bohr box), grid 49³ h 0.25.
fn neon_like(grid: &Grid, centre: [f64; 3]) -> Vec<Vec<f64>> {
    let mut v = vec![s_orb(grid, 1, 2.0, centre), s_orb(grid, 2, 2.0, centre)];
    v.extend(p_set(grid, 2, 2.0, centre));
    v
}

// ───────────────────────────── §5.1 clusters ─────────────────────────────

#[test]
fn tau_follows_the_scf_tolerance() {
    assert_eq!(tau(1e-5), 1e-3);
    assert_eq!(tau(1e-6), 1e-4);
    assert_eq!(tau(1e-8), 1e-4);
}

#[test]
fn clusters_join_gaps_up_to_tau() {
    let e = [-1.0, -0.5, -0.5 + 4e-4, -0.5 + 7e-4, -0.2];
    let c = clusters(&e, 1e-3);
    assert_eq!(c, vec![0..1, 1..4, 4..5]);
    // single linkage: the spread of the middle cluster exceeds τ but stays ≤ (d − 1)τ
    let s = spread(&e[1..4]);
    assert!(s > 1e-3 * 0.5 && s <= 2.0 * 1e-3, "spread {s}");
    assert_eq!(clusters(&e, 1e-4), vec![0..1, 1..2, 2..3, 3..4, 4..5]);
    assert!(clusters(&[], 1e-3).is_empty());
}

#[test]
fn combination_residual_is_at_most_half_the_spread() {
    // H = diag(ε) on the span of three exact eigenvectors; ψ = Σ c_j φ_j, ‖c‖ = 1.
    let mut rng = Rng(0xC0FFEE);
    for &s in &[1.41e-3, 0.05] {
        let eps = [-0.5 - s / 2.0, -0.5, -0.5 + s / 2.0];
        let mut worst: f64 = 0.0;
        for _ in 0..500 {
            let mut c = [rng.normal(), rng.normal(), rng.normal()];
            let n = c.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in c.iter_mut() { *x /= n; }
            let rq: f64 = (0..3).map(|j| c[j] * c[j] * eps[j]).sum();
            let res = (0..3).map(|j| (c[j] * (eps[j] - rq)).powi(2)).sum::<f64>().sqrt();
            worst = worst.max(res);
            assert!(res <= s / 2.0 + 1e-14, "residual {res} > s/2 = {}", s / 2.0);
        }
        eprintln!("s = {s}: worst residual over 500 combinations {worst:.3e} (bound s/2 = {:.3e})", s / 2.0);
        // the bound is sharp: equal weight on the two extremes, c = (1/√2, 0, ±1/√2)
        for sg in [1.0, -1.0] {
            let c = [std::f64::consts::FRAC_1_SQRT_2, 0.0, sg * std::f64::consts::FRAC_1_SQRT_2];
            let rq: f64 = (0..3).map(|j| c[j] * c[j] * eps[j]).sum();
            let res = (0..3).map(|j| (c[j] * (eps[j] - rq)).powi(2)).sum::<f64>().sqrt();
            assert!((res - s / 2.0).abs() <= 1e-14, "extremal residual {res} vs s/2 {}", s / 2.0);
        }
    }
}

#[test]
fn clusters_never_cross_an_occupation_change() {
    // An open p shell: p_x occupied, p_y and p_z empty, all at −0.2 (degenerate).
    let up = Chan::synthetic(&[-1.0, -0.2], 2).with_unoccupied(&[-0.2, -0.2, 0.3, 0.4, 0.5]);
    let case = Case::new(up, Chan::synthetic(&[], 0), false);
    let out = Tracker::new().update(&case.input());
    let r = rows(&out, Spin::Up);
    assert_eq!(r[1].cluster, None, "the occupied p_x is alone in its occupation block");
    assert_eq!(r[2].cluster.map(|c| c.0), Some(2));
    assert_eq!(r[3].cluster.map(|c| c.0), Some(2));
}

// ───────────────────────────── §5.2 moments ─────────────────────────────

#[test]
fn moments_of_hydrogenic_functions() {
    // c is linear in ψ, so a test function must fit the box in ∫|ψ|, not only in ∫ψ²: at Z = 1
    // the 2p's ∫r³e^{−r/2} has 29 % beyond the 9.6 Bohr half-box, and the cube's truncation adds
    // l = 3 character (measured W_1 = 0.99745, W_3 = 2.5e-3). Shapes are scaled to Z = 2 (2p) and
    // Z = 3 (3d) so that the function, not the box, is measured.
    let g = Grid::cube(49, 0.4);
    let c = [0.13, -0.21, 0.07];
    let s1 = hydrogenic(&g, 1, 0, 0, 1.0, c, None);
    let px = hydrogenic(&g, 2, 1, lm_index(1, 1), 2.0, c, None);
    let dxy = hydrogenic(&g, 3, 2, lm_index(2, -2), 3.0, c, None);
    let w1s = character(&moments(&g, &s1, c));
    let wp = character(&moments(&g, &px, c));
    let wd = character(&moments(&g, &dxy, c));
    eprintln!("49³ h 0.4: W(1s) {w1s:?}
W(2p_x) {wp:?}
W(3d_xy) {wd:?}");
    assert!(w1s[0] > 0.999);
    assert!(wp[1] > 0.999);
    assert!(wd[2] > 0.999);

    // b of 2p_x: only (1, x) nonzero. b_lm carries r^l, so entries of different l have different
    // units and |b_lm|/|b_(1,x)| has no fixed meaning across l; the scale-free form is the
    // cancellation fraction |⟨ψ|S_lm⟩| / ∫|ψ S_lm| (0 for a forbidden entry). Two error sources
    // compete on 49³ at h = 0.4 (measured, max over the 15 entries): the box truncation, which is
    // asymmetric about the off-grid centre and weighted by r^{l+3} (2.8e-2 at Z = 2, 5.5e-5 at
    // Z = 4), and the lattice aliasing of the e^{−Zr} cusp, f̂ ~ (Z/k)^5 at |G| = 2π/h (b_00:
    // 3.6e-5 at Z = 3, 1.2e-4 at Z = 4; W_1 falls to 0.998 at Z = 6). No Z puts both under 1e-4
    // there, so the sparsity check runs on 97³ at h = 0.2 — the same 9.6 Bohr half-box, the cusp
    // aliasing ~2⁵ smaller — at Z = 4.
    let g = Grid::cube(97, 0.2);
    let px = hydrogenic(&g, 2, 1, lm_index(1, 1), 4.0, c, None);
    let mp = moments(&g, &px, c);
    let mut abs = vec![0.0; 16];
    let mut s = [0.0; N_MOMENTS];
    for p in 0..g.len() {
        let r = g.point(p);
        solid_harmonics([r[0] - c[0], r[1] - c[1], r[2] - c[2]], &mut s);
        for i in 0..16 { abs[i] += (px[p] * s[i]).abs() * g.dv(); }
    }
    let bx = mp.b[lm_index(1, 1)];
    let mut worst: f64 = 0.0;
    for i in 0..16 {
        if i == lm_index(1, 1) { continue; }
        let frac = mp.b[i].abs() / abs[i];
        worst = worst.max(frac);
        eprintln!("97³ h 0.2: b[{i:2}] = {:+.3e}  |b|/|b_(1,x)| {:.2e}  cancellation fraction {frac:.2e}", mp.b[i], mp.b[i].abs() / bx.abs());
        assert!(frac < 1e-4, "b[{i}]: |⟨ψ|S⟩|/∫|ψS| = {frac}");
    }
    eprintln!("b_(1,x) = {bx:.4e}; worst forbidden-entry cancellation fraction {worst:.2e}");
    assert!(character(&mp)[1] > 0.999);
}

#[test]
fn set_character_is_invariant_under_mixing() {
    let g = Grid::cube(41, 0.3);
    let c = [0.05, -0.08, 0.11];
    let mut rng = Rng(0x5E7);
    // Z per set so that each set fits the ±6 Bohr box in ∫|ψ| (the 4f at Z = 2 peaks at r = 6:
    // measured W_3(C) = 0.976 from the box truncation alone; see moments_of_hydrogenic_functions).
    let sets: Vec<(u8, Vec<Vec<f64>>)> = vec![
        (1, (0..3).map(|k| hydrogenic(&g, 2, 1, 1 + k, 2.0, c, None)).collect()),
        (2, (0..5).map(|k| hydrogenic(&g, 3, 2, 4 + k, 4.0, c, None)).collect()),
        (3, (0..7).map(|k| hydrogenic(&g, 4, 3, 9 + k, 8.0, c, None)).collect()),
    ];
    for (l, set) in &sets {
        let refs: Vec<&[f64]> = set.iter().map(|p| p.as_slice()).collect();
        let ms = moments_many(&g, &refs, c);
        let w0 = set_character(&ms)[*l as usize];
        eprintln!("l = {l}: W_l(C) unmixed = {w0:.12}");
        assert!(w0 > 0.999, "an ideal shell has W_l(C) ≈ 1");
        // linearity: the moments of a mixed grid set are the mixed moments
        let q = random_orthogonal(&mut rng, set.len());
        let mixed = mix(set, &q);
        let mrefs: Vec<&[f64]> = mixed.iter().map(|p| p.as_slice()).collect();
        let direct = moments_many(&g, &mrefs, c);
        let via = mix_moments(&ms, &q);
        for (a, b) in direct.iter().zip(&via) {
            for i in 0..16 {
                assert!((a.c[i] - b.c[i]).abs() <= 1e-12 * (1.0 + b.c[i].abs()));
                assert!((a.b[i] - b.b[i]).abs() <= 1e-12 * (1.0 + b.b[i].abs()));
            }
        }
        let mut worst: f64 = 0.0;
        for _ in 0..200 {
            let q = random_orthogonal(&mut rng, set.len());
            let w = set_character(&mix_moments(&ms, &q))[*l as usize];
            worst = worst.max((w - w0).abs());
        }
        eprintln!("l = {l}: max |ΔW_l(C)| over 200 mixings = {worst:.2e} (0 to rounding)");
        assert!(worst <= 1e-12);
    }
}

#[test]
fn radial_profile_sums_to_the_c_moments() {
    // Gaussian-weighted harmonics e^{−r²} S_lm: nothing outside the inscribed sphere (r > 5.9:
    // e^{−34.8}), so the bins must sum to c exactly up to rounding.
    let g = Grid::cube(49, 0.25);
    for (n, l, lm) in [(1u8, 0u8, 0usize), (2, 1, 3), (3, 2, 4)] {
        let mut s = [0.0; N_MOMENTS];
        let mut psi = g.sample(|p| {
            let d = [p[0] - OFF[0], p[1] - OFF[1], p[2] - OFF[2]];
            solid_harmonics(d, &mut s);
            (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).exp() * s[lm]
        });
        normalise(&g, &mut psi);
        let m = moments(&g, &psi, OFF);
        let p = radial_profile(&g, &psi, OFF);
        let mut sum = [0.0; 16];
        for k in 0..p.a.len() { for i in 0..16 { sum[i] += p.a[k][i]; } }
        for i in 0..16 {
            assert!((sum[i] - m.c[i]).abs() <= 1e-12 * m.c[lm].abs(), "({n}{l}) c[{i}] {} vs profile {}", m.c[i], sum[i]);
        }
    }
}

// ───────────────────────────── §5.3 alignment ─────────────────────────────

fn dominance(m: &Moments, l: u8, k: usize) -> f64 {
    let lo = (l as usize) * (l as usize);
    let nrm = (lo..lo + 2 * l as usize + 1).map(|i| m.b[i] * m.b[i]).sum::<f64>().sqrt();
    m.b[display_lm(l, k)].abs() / nrm
}

#[test]
fn procrustes_recovers_a_random_rotation_of_hydrogenic_2p() {
    let g = Grid::cube(41, 0.3);
    let c = [0.0; 3];
    let p = p_set(&g, 2, 2.0, c);
    let r = random_orthogonal(&mut Rng(0x5EED), 3);
    let asym = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| (r[(i, j)] - r[(j, i)]).abs()).fold(0.0, f64::max);
    eprintln!("R = {r}max|R − Rᵀ| = {asym:.3}");
    assert!(asym > 0.3, "R must not be (nearly) symmetric, or the test cannot tell U from Uᵀ");
    // φ_j = Σ_a R_ja p_a  ⇔  φ = mix(p, Rᵀ)
    let phi = mix(&p, &r.transpose());
    let refs: Vec<&[f64]> = phi.iter().map(|x| x.as_slice()).collect();
    let mphi = moments_many(&g, &refs, c);
    let before = (0..3).map(|k| dominance(&mphi[k], 1, k)).fold(1.0, f64::min);
    let (u, rank) = alignment(&mphi, 1);
    let psi = mix(&phi, &u);
    let refs: Vec<&[f64]> = psi.iter().map(|x| x.as_slice()).collect();
    let mpsi = moments_many(&g, &refs, c);
    let after: Vec<f64> = (0..3).map(|k| dominance(&mpsi[k], 1, k)).collect();
    let rtu = r.transpose() * &u - DMatrix::<f64>::identity(3, 3);
    let err = rtu.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    eprintln!("dominance before min {before:.4}; after {after:?}; σ_min/σ_max {rank:.6}; max|RᵀU − I| {err:.2e}");
    assert!(before < 0.9);
    for k in 0..3 {
        assert!(after[k] > 0.9999, "k = {k}: {}", after[k]);
        assert!(mpsi[k].b[display_lm(1, k)] > 0.0, "built-in sign: ⟨ψ_k|x_k⟩ > 0");
    }
    assert!(err <= 1e-8, "RᵀU = I (U = R, review MUST 2)");
}

/// The real l = 2 Wigner matrix of a rotation Q in the solid-harmonic basis (m order):
/// S_j(Qᵀx) = Σ_k S_k(x) D_kj, solved by least squares on 40 random points (exact: the five
/// S_2m span the rotated ones).
fn wigner_l2(q: [[f64; 3]; 3], rng: &mut Rng) -> DMatrix<f64> {
    let n = 40;
    let mut a = DMatrix::zeros(n, 5);
    let mut b = DMatrix::zeros(n, 5);
    let mut s = [0.0; N_MOMENTS];
    for i in 0..n {
        let x = [rng.normal(), rng.normal(), rng.normal()];
        let qtx = [0, 1, 2].map(|c| q[0][c] * x[0] + q[1][c] * x[1] + q[2][c] * x[2]);
        solid_harmonics(x, &mut s);
        for k in 0..5 { a[(i, k)] = s[4 + k]; }
        solid_harmonics(qtx, &mut s);
        for k in 0..5 { b[(i, k)] = s[4 + k]; }
    }
    a.svd(true, true).solve(&b, 1e-14).unwrap()
}

#[test]
fn procrustes_aligns_a_rotated_3d_set() {
    // φ_j(r) = d_j(Qᵀ(r − R)) = Σ_k d_k(r − R) D_kj: the l = 2 real Wigner rep; the alignment must
    // return U = Dᵀ (D U = I), the d analogue of the 2p test's RᵀU = I (there D = Rᵀ).
    // Unlike p (the O_h irrep T1u), d splits into e_g + t_2g on the lattice even on a
    // cube-symmetric site, so ⟨d_k|S_a⟩ = c δ_ka holds only to the lattice error: the shape must be
    // resolved (aliasing of the |r|³ term of r²e^{−Zr/3}) and fit the box in ∫|ψ|r^{2}
    // (truncation; the earlier σ_min/σ_max = 0.809 at Z = 3 in ±6 Bohr was that truncation).
    // Measured max|DU − I| (seed 0xD5E7): 49³/h 0.25: 5.4e-8 at Z = 5, 5.7e-10 at Z = 10;
    // 97³/h 0.2: 1.8e-12 at Z = 10; 129³/h 0.15: 2.7e-11 at Z = 15. Pinned: 49³, Z = 10.
    let g = Grid::cube(49, 0.25);
    let mut rng = Rng(0xD5E7);
    let q = random_rotation3(&mut rng);
    let d = wigner_l2(q, &mut rng);
    let set: Vec<Vec<f64>> = (0..5).map(|k| hydrogenic(&g, 3, 2, 4 + k, 10.0, OFF, Some(q))).collect();
    let refs: Vec<&[f64]> = set.iter().map(|x| x.as_slice()).collect();
    let m = moments_many(&g, &refs, OFF);
    let before = (0..5).map(|k| dominance(&m[k], 2, k)).fold(1.0, f64::min);
    let (u, rank) = alignment(&m, 2);
    let err = (&d * &u - DMatrix::<f64>::identity(5, 5)).iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let psi = mix(&set, &u);
    let refs: Vec<&[f64]> = psi.iter().map(|x| x.as_slice()).collect();
    let mpsi = moments_many(&g, &refs, OFF);
    let after: Vec<f64> = (0..5).map(|k| dominance(&mpsi[k], 2, k)).collect();
    eprintln!("3d: dominance before min {before:.4}; after {after:?}; σ_min/σ_max {rank:.6}; max|DU − I| {err:.2e}");
    assert!(before < 0.9);
    assert!(err <= 1e-8, "U = Dᵀ (the real Wigner matrix)");
    for k in 0..5 {
        assert!(after[k] > 0.999, "k = {k}: {}", after[k]);
        assert!(mpsi[k].b[display_lm(2, k)] > 0.0);
    }
    let names: Vec<&str> = (0..5).map(|k| harmonic_name(2, k)).collect();
    assert_eq!(names, ["xy", "yz", "z²", "xz", "x²−y²"]);
    assert_eq!((0..3).map(|k| harmonic_name(1, k)).collect::<Vec<_>>(), ["x", "y", "z"]);
}

// ───────────────────────────── §5.3 shells and labels ─────────────────────────────

#[test]
fn split_shell_keeps_its_labels() {
    let g = Grid::cube(49, 0.25);
    let base = neon_like(&g, OFF);
    let s_max = 7.05e-4; // Task 0: Ne 57³, h = 0.25, at (h/2, 0, 0)
    for (seed, s) in [(1u64, 2.0 * s_max), (2, 0.05)] {
        let q = random_orthogonal(&mut Rng(seed), 3);
        let mut psis = base[..2].to_vec();
        psis.extend(mix(&base[2..], &q));
        let evals = [-1.0, -0.30, -0.20, -0.20 + s, -0.20 + 2.0 * s];
        let ch = Chan::from_grid(&g, &psis, &evals, 5, OFF);
        let case = Case::new(ch.clone(), ch, true);
        let out = Tracker::new().update(&case.input());
        let want = vec![atomic(1, 0, ""), atomic(2, 0, ""), atomic(2, 1, "x"), atomic(2, 1, "y"), atomic(2, 1, "z")];
        for spin in [Spin::Up, Spin::Down] {
            assert_eq!(labels(&out, spin), want, "S = {s}");
            let r = rows(&out, spin);
            for row in &r[2..5] {
                assert!(row.rotated);
                let split = row.grid_split.expect("aligned shell carries its split");
                assert!((split - 2.0 * s).abs() <= 1e-12, "grid_split {split} vs 2S {}", 2.0 * s);
            }
            assert_eq!(r[0].grid_split, None);
        }
        assert!(matches!(out.lone_atom, LoneAtom::Labelled { shell_order_inverted: false }));
    }
}

#[test]
fn l_character_needs_the_full_shell() {
    let s = ideal(0);
    let [px, py, pz] = P_LM.map(ideal);
    // 3 p → one l = 1 shell
    let ch = [px, py, pz];
    let e = [-0.5, -0.5, -0.5];
    let sh = lone_atom_shells([&ch, &[]], [&e, &[]]).expect("one p shell");
    assert_eq!(sh[0].len(), 1);
    assert_eq!((sh[0][0].l, sh[0][0].range.clone()), (1, 0..3));
    // 6 p (2p + 3p adjacent) → two shells, cut in energy order
    let ch6 = [px, py, pz, px, py, pz];
    let e6 = [-0.8, -0.8, -0.8, -0.3, -0.3, -0.3];
    let sh = lone_atom_shells([&ch6, &[]], [&e6, &[]]).expect("two p shells");
    assert_eq!(sh[0].iter().map(|s| (s.l, s.range.clone())).collect::<Vec<_>>(), vec![(1, 0..3), (1, 3..6)]);
    // 2 p → None (2 ≠ k(2l + 1))
    assert!(lone_atom_shells([&[px, py], &[]], [&[-0.5, -0.5], &[]]).is_none());
    // p_x, p_y and a 2s: the 2s breaks the run
    assert!(lone_atom_shells([&[px, py, s], &[]], [&[-0.5, -0.5, -0.3], &[]]).is_none());
    assert!(lone_atom_shells([&[px, s, py, pz], &[]], [&[-0.5, -0.4, -0.3, -0.3], &[]]).is_none());
    // one channel open makes the whole atom unclassifiable (both channels)
    assert!(lone_atom_shells([&[s, px, py, pz], &[s, px]], [&[-1.0, -0.5, -0.5, -0.5], &[-1.0, -0.5]]).is_none());
}

#[test]
fn sturm_liouville_labels_a_lone_atom() {
    let g = Grid::cube(49, 0.25);
    let mut psis = neon_like(&g, OFF);
    psis.push(s_orb(&g, 3, 2.0, OFF));
    let evals = [-1.0, -0.30, -0.20, -0.20, -0.20, -0.05];
    let ch = Chan::from_grid(&g, &psis, &evals, 6, OFF);
    let nodes: Vec<Option<usize>> = (0..6).map(|i| {
        let l = dominant_l(&ch.moments[i]).unwrap();
        profile_nodes(&ch.radial[i], &ch.moments[i], l)
    }).collect();
    eprintln!("node counts (shell profiles): {nodes:?}");
    assert_eq!(nodes, vec![Some(0), Some(1), Some(0), Some(0), Some(0), Some(2)]);
    let mut case = Case::new(ch, Chan::synthetic(&[], 0), true);
    let out = Tracker::new().update(&case.input());
    assert_eq!(labels(&out, Spin::Up),
               vec![atomic(1, 0, ""), atomic(2, 0, ""), atomic(2, 1, "x"), atomic(2, 1, "y"), atomic(2, 1, "z"), atomic(3, 0, "")]);
    case.n_core = [1, 0, 0, 0];
    let out = Tracker::new().update(&case.input());
    assert_eq!(labels(&out, Spin::Up),
               vec![atomic(2, 0, ""), atomic(3, 0, ""), atomic(2, 1, "x"), atomic(2, 1, "y"), atomic(2, 1, "z"), atomic(4, 0, "")]);
}

#[test]
fn inverted_shell_order_is_labelled_by_nodes_and_flagged() {
    // The Task 0 Ne ordering at h = σ = 0.25: 1s, 2p ×3, 2s (smeared nucleus; spike §1b).
    let g = Grid::cube(49, 0.25);
    let base = neon_like(&g, OFF);
    let psis = vec![base[0].clone(), base[2].clone(), base[3].clone(), base[4].clone(), base[1].clone()];
    let evals = [-5.900786, -0.845152, -0.845152, -0.845152, -0.298213];
    let ch = Chan::from_grid(&g, &psis, &evals, 5, OFF);
    let case = Case::new(ch.clone(), ch, true);
    let out = Tracker::new().update(&case.input());
    for spin in [Spin::Up, Spin::Down] {
        assert_eq!(labels(&out, spin),
                   vec![atomic(1, 0, ""), atomic(2, 1, "x"), atomic(2, 1, "y"), atomic(2, 1, "z"), atomic(2, 0, "")]);
    }
    assert!(matches!(out.lone_atom, LoneAtom::Labelled { shell_order_inverted: true }));
    // the physical order is not flagged
    let normal = Case::new(Chan::from_grid(&g, &base, &[-1.0, -0.3, -0.2, -0.2, -0.2], 5, OFF), Chan::synthetic(&[], 0), true);
    assert!(matches!(Tracker::new().update(&normal.input()).lone_atom, LoneAtom::Labelled { shell_order_inverted: false }));
}

#[test]
fn unresolved_nodes_fall_back_to_index_labels() {
    // Two s shells whose profiles both read 0 nodes: Sturm–Liouville requires 0 and 1 → refuse.
    let g = Grid::cube(49, 0.25);
    let psis = vec![s_orb(&g, 1, 2.0, OFF), s_orb(&g, 1, 0.8, OFF)];
    let ch = Chan::from_grid(&g, &psis, &[-1.0, -0.3], 2, OFF);
    let case = Case::new(ch, Chan::synthetic(&[], 0), true);
    let out = Tracker::new().update(&case.input());
    assert_eq!(labels(&out, Spin::Up), vec![Label::Index(1), Label::Index(2)]);
    assert!(matches!(out.lone_atom, LoneAtom::Unlabelled(Unlabelled::NodesUnresolved)));
}

#[test]
fn radial_nodes_cross_check() {
    let g = Grid::cube(81, 0.2);
    let c = [0.0; 3];
    let s3 = s_orb(&g, 3, 1.0, c);
    let p2 = hydrogenic(&g, 2, 1, lm_index(1, 1), 1.0, c, None);
    let n3 = radial_nodes(&g, &s3, c, [1.0, 0.0, 0.0], NODE_FLOOR);
    let n2 = radial_nodes(&g, &p2, c, [1.0, 0.0, 0.0], NODE_FLOOR);
    eprintln!("3s along +x: {n3} nodes; 2p_x: {n2}");
    assert_eq!(n3, 2);
    assert_eq!(n2, 0);
    // the shell-profile count agrees
    let m = moments(&g, &s3, c);
    assert_eq!(profile_nodes(&radial_profile(&g, &s3, c), &m, 0), Some(2));
}

#[test]
fn open_shell_atom_falls_back_to_index_labels_in_both_channels() {
    // C-like: ↑ {1s, 2s, p, p} (2 of 3 p occupied), ↓ {1s, 2s}
    let s = ideal(0);
    let up_m = [s, s, ideal(3), ideal(1)];
    let dn_m = [s, s];
    let eu = [-10.0, -0.5, -0.2, -0.2];
    let ed = [-10.0, -0.45];
    assert!(lone_atom_shells([&up_m, &dn_m], [&eu, &ed]).is_none());
    let mut up = Chan::synthetic(&eu, 4);
    up.moments = up_m.to_vec();
    let mut dn = Chan::synthetic(&ed, 2);
    dn.moments = dn_m.to_vec();
    let case = Case::new(up, dn, true);
    let out = Tracker::new().update(&case.input());
    assert_eq!(labels(&out, Spin::Up), (1..=4).map(Label::Index).collect::<Vec<_>>());
    assert_eq!(labels(&out, Spin::Down), (1..=2).map(Label::Index).collect::<Vec<_>>());
    assert!(out.rows.iter().all(|r| !r.rotated));
    assert!(matches!(out.lone_atom, LoneAtom::Unlabelled(Unlabelled::OpenShell)));
}

#[test]
fn labels_are_index_only_for_molecules() {
    let up = Chan::synthetic(&[-1.0, -0.5, -0.5, -0.5], 4);
    let case = Case::new(up.clone(), up, false);
    let out = Tracker::new().update(&case.input());
    for spin in [Spin::Up, Spin::Down] {
        assert_eq!(labels(&out, spin), (1..=4).map(Label::Index).collect::<Vec<_>>());
        assert_eq!(rows(&out, spin)[1].cluster.map(|c| c.0), Some(3));
    }
    assert!(matches!(out.lone_atom, LoneAtom::NotLoneAtom));
}

#[test]
fn unoccupied_rows_are_never_named() {
    let g = Grid::cube(49, 0.25);
    let occ = Chan::from_grid(&g, &neon_like(&g, OFF), &[-1.0, -0.3, -0.2, -0.2, -0.2], 5, OFF);
    let full = occ.with_unoccupied(&[-0.02, 0.05, 0.05, 0.05, 0.09, 0.13, 0.2, 0.3]);
    let want = vec![atomic(1, 0, ""), atomic(2, 0, ""), atomic(2, 1, "x"), atomic(2, 1, "y"), atomic(2, 1, "z")];
    for k in 1..=8 {
        let ch = full.truncated(5 + k);
        let case = Case::new(ch.clone(), ch, true);
        let out = Tracker::new().update(&case.input());
        for spin in [Spin::Up, Spin::Down] {
            let l = labels(&out, spin);
            assert_eq!(l[..5], want[..], "n_computed = n_occ + {k}");
            for (i, lab) in l.iter().enumerate().skip(5) {
                assert_eq!(*lab, Label::Index(i + 1), "unoccupied row {i} (n_computed = n_occ + {k})");
            }
        }
    }
}

#[test]
fn block_edge_cluster_is_hidden() {
    // n_computed = n_occ + m + UNOCC_GUARD; the cluster holding the last computed vector is cut.
    let up = Chan::synthetic(&[-0.6], 1).with_unoccupied(&[-0.05, -0.01, 0.10, 0.10, 0.10]);
    let case = Case::new(up, Chan::synthetic(&[], 0), false);
    let out = Tracker::new().update(&case.input());
    assert_eq!(out.hidden_unoccupied, [3, 0]);
    assert_eq!(rows(&out, Spin::Up).len(), 3, "occupied + −0.05 + −0.01");
    let up = Chan::synthetic(&[-0.6], 1).with_unoccupied(&[-0.05, -0.01, 0.10, 0.10, 0.10, 0.25]);
    let case = Case::new(up, Chan::synthetic(&[], 0), false);
    let out = Tracker::new().update(&case.input());
    assert_eq!(out.hidden_unoccupied, [0, 0]);
    let r = rows(&out, Spin::Up);
    assert_eq!(r.len(), 6, "the 3-fold set is shown whole; the guard (0.25) is not");
    assert_eq!(r[3].cluster.map(|c| c.0), Some(3));
    assert!(r[3].box_state, "ε ≥ 0 is a box state");
    assert!(!r[1].box_state);
}

// ───────────────────────────── §5.4 signs and tracks ─────────────────────────────

/// σ_u-shaped (1s_A − 1s_B)/N with exponent 3, so that ψ fits the 33³ box (h = 0.25, ±4 Bohr):
/// the norm outside it is ≲ e^{−6·3.3}·(1 + 19.8 + 196) ≈ 5e-7, so the windowed values of the two
/// boxes may differ only at that level.
fn sigma_u(grid: &Grid, a: [f64; 3], b: [f64; 3]) -> Vec<f64> {
    let mut f = grid.sample(|p| {
        let da = ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2) + (p[2] - a[2]).powi(2)).sqrt();
        let db = ((p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) + (p[2] - b[2]).powi(2)).sqrt();
        (-3.0 * da).exp() - (-3.0 * db).exp()
    });
    normalise(grid, &mut f);
    f
}

/// Runs a fresh tracker on one orbital (molecule mode) and returns the displayed field.
fn display_one(grid: &Grid, psi: &[f64], nuclei: &[[f64; 3]]) -> Vec<f64> {
    let mut ch = Chan::synthetic(&[-0.3], 1);
    ch.first_sign = vec![first_sign_functionals(grid, psi, nuclei)];
    let case = Case::new(ch, Chan::synthetic(&[], 0), false);
    let out = Tracker::new().update(&case.input());
    apply_rotation(&[psi.to_vec()], &out, Spin::Up).remove(0)
}

#[test]
fn first_appearance_sign_is_deterministic_for_sigma_u() {
    let (a, b) = ([0.0, 0.7, 0.0], [0.0, -0.7, 0.0]);
    let mut decided = Vec::new();
    for n in [33, 65] {
        let g = Grid::cube(n, 0.25);
        let psi = sigma_u(&g, a, b);
        let f = first_sign_functionals(&g, &psi, &[a, b]);
        eprintln!("{n}³: functionals {f:?}");
        assert!(f[0].abs() < 1e-12, "the windowed monopole vanishes on σ_u by symmetry");
        let (_, by) = first_appearance_sign(&f);
        decided.push((by, f[2]));
        assert_eq!(by, Some(2), "decided by the windowed y dipole");
        let plus = display_one(&g, &psi, &[a, b]);
        let neg: Vec<f64> = psi.iter().map(|v| -v).collect();
        let minus = display_one(&g, &neg, &[a, b]);
        assert_eq!(plus, minus, "the displayed field does not depend on the solver's sign");
    }
    let (v33, v65) = (decided[0].1, decided[1].1);
    eprintln!("y-dipole value 33³ {v33:.9}, 65³ {v65:.9}");
    assert!((v33 - v65).abs() <= 1e-5 * v65.abs(), "windowed: box-independent (review SHOULD 10)");
}

#[test]
fn no_qualifying_functional_keeps_the_raw_sign() {
    let g = Grid::cube(81, 0.25);
    let far = [8.0, 0.0, 0.0];
    let mut psi = g.sample(|p| (-((p[0] - far[0]).powi(2) + p[1] * p[1] + p[2] * p[2]) / (2.0 * 0.25)).exp());
    normalise(&g, &mut psi);
    let f = first_sign_functionals(&g, &psi, &[[0.0; 3]]);
    eprintln!("functionals of an orbital 8 Bohr out: {f:?}");
    assert!(f.iter().all(|v| v.abs() < SIGN_MOMENT_MIN));
    assert_eq!(first_appearance_sign(&f), (1.0, None));
    // not deterministic: the displayed sign is whatever the solver returned
    assert_eq!(display_one(&g, &psi, &[[0.0; 3]]), psi);
    let neg: Vec<f64> = psi.iter().map(|v| -v).collect();
    assert_eq!(display_one(&g, &neg, &[[0.0; 3]]), neg);
}

#[test]
fn sign_continuity_survives_injected_flips() {
    let g = Grid::cube(33, 0.4);
    let mut rng = Rng(0xF11B);
    let mut dir = [rng.normal(), rng.normal(), rng.normal()];
    let nd = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    for x in dir.iter_mut() { *x /= nd; }
    let evals = [-2.0, -0.5, -0.5, -0.5];
    let mut tracker = Tracker::new();
    let mut prev_disp: Option<Vec<Vec<f64>>> = None;
    let mut prev_tracks: Option<Vec<(u32, u8)>> = None;
    let mut worst: f64 = 1.0;
    for t in 0..100 {
        let c = [0.01 * t as f64 * dir[0], 0.01 * t as f64 * dir[1], 0.01 * t as f64 * dir[2]];
        let mut raw = vec![s_orb(&g, 1, 2.0, c)];
        let q = random_orthogonal(&mut rng, 3);
        raw.extend(mix(&p_set(&g, 2, 2.0, c), &q));
        for psi in raw.iter_mut() {
            let s = rng.sign();
            for v in psi.iter_mut() { *v *= s; }
        }
        let mut ch = Chan::synthetic(&evals, 4);
        ch.first_sign = raw.iter().map(|p| first_sign_functionals(&g, p, &[c])).collect();
        let mut case = Case::new(ch, Chan::synthetic(&[], 0), false);
        if let Some(pd) = &prev_disp {
            case.prev[0] = Some(DMatrix::from_fn(4, pd.len(), |i, j| g.dot(&raw[i], &pd[j])));
        }
        let out = tracker.update(&case.input());
        let disp = apply_rotation(&raw, &out, Spin::Up);
        let tracks: Vec<(u32, u8)> = rows(&out, Spin::Up).iter().map(|r| (r.track, r.slot)).collect();
        if let (Some(pd), Some(pt)) = (&prev_disp, &prev_tracks) {
            assert_eq!(&tracks, pt, "frame {t}: tracks and slots constant");
            for k in 0..4 {
                let o = g.dot(&disp[k], &pd[k]);
                worst = worst.min(o);
                assert!(o >= 0.999, "frame {t}, track {k}: ⟨ψ̃_t|ψ̃_t−1⟩ = {o}");
            }
        }
        prev_disp = Some(disp);
        prev_tracks = Some(tracks);
    }
    eprintln!("worst consecutive displayed overlap over 100 frames: {worst:.6}");
}

#[test]
fn hungarian_recovers_a_permutation_and_retires_tracks() {
    let mut rng = Rng(0x4A11);
    let evals: Vec<f64> = (0..8).map(|i| -1.0 + 0.1 * i as f64).collect();
    let mut tracker = Tracker::new();
    let first = tracker.update(&Case::new(Chan::synthetic(&evals, 8), Chan::synthetic(&[], 0), false).input());
    let t0: Vec<(u32, u8)> = rows(&first, Spin::Up).iter().map(|r| (r.track, r.slot)).collect();
    assert_eq!(t0.iter().map(|t| t.1).collect::<Vec<_>>(), (0..8).collect::<Vec<u8>>(), "lowest free slots in energy order");
    // current i ↔ previous π(i)
    let pi = [3usize, 0, 6, 1, 7, 2, 5, 4];
    let mut case = Case::new(Chan::synthetic(&evals, 8), Chan::synthetic(&[], 0), false);
    case.prev[0] = Some(DMatrix::from_fn(8, 8, |i, j| if pi[i] == j { 0.999 } else { 1e-3 * rng.normal() }));
    let out = tracker.update(&case.input());
    let t1: Vec<(u32, u8)> = rows(&out, Spin::Up).iter().map(|r| (r.track, r.slot)).collect();
    for i in 0..8 { assert_eq!(t1[i], t0[pi[i]], "row {i} follows its previous orbital"); }
    // replace row 5 by an orthogonal newcomer: its old partner (previous row 5 = t1[5]) retires
    let mut case = Case::new(Chan::synthetic(&evals, 8), Chan::synthetic(&[], 0), false);
    case.prev[0] = Some(DMatrix::from_fn(8, 8, |i, j| if i == 5 { 0.1 * rng.normal().clamp(-3.0, 3.0) / 3.0 } else if i == j { -0.999 } else { 1e-3 * rng.normal() }));
    let out = tracker.update(&case.input());
    let t2: Vec<(u32, u8)> = rows(&out, Spin::Up).iter().map(|r| (r.track, r.slot)).collect();
    for i in (0..8).filter(|&i| i != 5) { assert_eq!(t2[i], t1[i]); }
    assert!(t0.iter().all(|t| t.0 != t2[5].0), "a new track id");
    assert_eq!(t2[5].1, t1[5].1, "the retired slot is reused");
    // the sign follows the overlap: the −0.999 rows are flipped
    let rot = &out.rotations[0];
    let sign_of = |i: usize| rot.iter().find(|(r, _)| r.contains(&i)).map(|(r, u)| u[(i - r.start, i - r.start)]).unwrap();
    assert_eq!(sign_of(0), -1.0);
}

#[test]
fn hungarian_is_optimal_against_brute_force() {
    let mut rng = Rng(0xB0F);
    for trial in 0..200 {
        let (n, m) = (1 + trial % 5, 1 + (trial / 5) % 5);
        let w = DMatrix::from_fn(n, m, |_, _| rng.uniform());
        let a = hungarian_max(&w);
        let total: f64 = a.iter().enumerate().filter_map(|(i, j)| j.map(|j| w[(i, j)])).sum();
        // brute force over injective maps rows → cols ∪ {none}
        fn best(w: &DMatrix<f64>, i: usize, used: &mut Vec<bool>) -> f64 {
            if i == w.nrows() { return 0.0; }
            let mut b = best(w, i + 1, used);
            for j in 0..w.ncols() {
                if !used[j] { used[j] = true; b = b.max(w[(i, j)] + best(w, i + 1, used)); used[j] = false; }
            }
            b
        }
        let opt = best(&w, 0, &mut vec![false; m]);
        assert!((total - opt).abs() < 1e-12, "trial {trial}: {total} vs {opt}");
        let mut seen = vec![false; m];
        for j in a.iter().flatten() { assert!(!seen[*j]); seen[*j] = true; }
    }
}

#[test]
fn closed_shell_down_shares_up_colours() {
    let e = [-1.0, -0.5, -0.3];
    let mut case = Case::new(Chan::synthetic(&e, 3), Chan::synthetic(&e, 3), false);
    case.cross = Some(DMatrix::from_fn(3, 3, |i, j| if i == j { 0.999 } else { 0.01 }));
    let out = Tracker::new().update(&case.input());
    let up: Vec<u8> = rows(&out, Spin::Up).iter().map(|r| r.slot).collect();
    let dn: Vec<u8> = rows(&out, Spin::Down).iter().map(|r| r.slot).collect();
    assert_eq!(up, dn);
    let ut: Vec<u32> = rows(&out, Spin::Up).iter().map(|r| r.track).collect();
    let dt: Vec<u32> = rows(&out, Spin::Down).iter().map(|r| r.track).collect();
    assert!(ut.iter().all(|t| !dt.contains(t)), "tracks stay per channel; only the colour is shared");
    // an unmatched ↓ (cross < 0.9) takes a fresh slot
    case.cross = Some(DMatrix::from_fn(3, 3, |i, j| if i == j && i < 2 { 0.999 } else if i == j { 0.6 } else { 0.01 }));
    let out = Tracker::new().update(&case.input());
    let dn: Vec<u8> = rows(&out, Spin::Down).iter().map(|r| r.slot).collect();
    assert_eq!(dn, vec![0, 1, 3]);
}

// ───────────────────────────── §5.5 – §5.7 ─────────────────────────────

#[test]
fn spin_summary_scale_uses_the_floor() {
    let s = SpinSummary { n_up: 1, n_dn: 1, integral: 0.0, abs_integral: 0.0, max_abs: 1e-7, rho_max: 1.0 };
    assert_eq!(s.scale(), 1e-3);
    let s = SpinSummary { max_abs: 0.2, ..s };
    assert_eq!(s.scale(), 0.2);
    // the CPU reference summary of an H-like channel
    let g = Grid::cube(33, 0.25);
    let psi = s_orb(&g, 1, 1.0, [0.0; 3]);
    let up: Vec<f64> = psi.iter().map(|v| v * v).collect();
    let dn = vec![0.0; g.len()];
    let sum = spin_summary(&g, &up, &dn, 1, 0);
    assert!((sum.integral - 1.0).abs() < 1e-12 && (sum.abs_integral - 1.0).abs() < 1e-12);
    assert_eq!(sum.max_abs, sum.rho_max);
}

#[test]
fn colour_volume_reference_composes_weights() {
    let g = Grid::cube(33, 0.25);
    let (a, b) = ([1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]);
    let gauss = |c: [f64; 3]| {
        let mut f = g.sample(|p| (-((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)) / (2.0 * 0.25)).exp());
        normalise(&g, &mut f);
        f
    };
    let (red, green) = (gauss(a), gauss(b));
    let peak = |f: &[f64]| f.iter().fold(0.0f64, |m, v| m.max(v * v));
    let (pr, pg) = (peak(&red), peak(&green));
    let ia = (0..g.len()).find(|&i| { let p = g.point(i); (p[0] - a[0]).abs() < 1e-12 && p[1].abs() < 1e-12 && p[2].abs() < 1e-12 }).unwrap();
    let vol = colour_volume(&g, &[(&red, 1.0, [1.0, 0.0, 0.0], pr), (&green, 1.0, [0.0, 1.0, 0.0], pg)]);
    let eps = green[ia] * green[ia] / pg;
    eprintln!("per-orbital at the red centre: {:?} (ε = {eps:.3e})", vol[ia]);
    assert!((vol[ia][0] - 1.0).abs() < 1e-12);
    assert!((vol[ia][1] - eps).abs() < 1e-15 && vol[ia][2] == 0.0);
    assert!((vol[ia][3] - (1.0 + eps)).abs() < 1e-12);
    // physical scaling: the caller passes P = max ρ for every orbital
    let rho_max = (0..g.len()).map(|i| red[i] * red[i] + green[i] * green[i]).fold(0.0, f64::max);
    let phys = colour_volume(&g, &[(&red, 1.0, [1.0, 0.0, 0.0], rho_max), (&green, 1.0, [0.0, 1.0, 0.0], rho_max)]);
    assert!((phys[ia][3] - (red[ia] * red[ia] + green[ia] * green[ia]) / rho_max).abs() < 1e-15);
    assert!((phys[ia][0] - red[ia] * red[ia] / rho_max).abs() < 1e-15);
}

#[test]
fn box_state_rule_is_box_independent() {
    for n in [33usize, 97] {
        let g = Grid::cube(n, 0.25);
        let l = (n + 1) as f64 * g.h; // Dirichlet: ψ = 0 one step outside the grid
        let mut psi = g.sample(|p| {
            (0..3).map(|a| (std::f64::consts::PI * (p[a] - g.origin[a] + g.h) / l).sin()).product()
        });
        normalise(&g, &mut psi);
        let w = outer_weight(&g, &psi);
        eprintln!("lowest box state, {n}³: outer-L/6 weight {w:.4} (continuum 0.1632)");
        assert!((w - 0.163).abs() <= 0.01);
        assert!(is_box_state(-0.01, w));
    }
    assert!(is_box_state(0.01, 0.0), "ε ≥ 0 is always a box state");
    let g = Grid::cube(79, 0.25); // L = (79 + 1) × 0.25 = 20 Bohr
    let s1 = s_orb(&g, 1, 1.0, [0.0; 3]);
    let w = outer_weight(&g, &s1);
    eprintln!("hydrogenic 1s in a 20 Bohr box: outer weight {w:.3e}");
    assert!(w < 1e-4);
    assert!(!is_box_state(-0.5, w));
}

#[test]
fn smeared_one_s_tag_follows_z_sigma() {
    let g = Grid::cube(33, 0.25);
    let ch = Chan::from_grid(&g, &[s_orb(&g, 1, 1.0, OFF)], &[-0.27], 1, OFF);
    for (zs, tag) in [(0.25, false), (0.5, true), (2.5, true)] {
        let mut case = Case::new(ch.clone(), Chan::synthetic(&[], 0), true);
        case.z_sigma = Some(zs);
        let out = Tracker::new().update(&case.input());
        let r = rows(&out, Spin::Up);
        assert_eq!(r[0].label, atomic(1, 0, ""));
        assert_eq!(r[0].smeared_1s, tag, "Zσ = {zs}");
    }
}

// ───────────────────────────── real eigenproblems (release) ─────────────────────────────

/// One electron in the bare smeared well of Z = 10 at σ = h = 0.25 (Zσ = 2.5, Ne's value):
/// the five lowest states of the discrete operator, by the CPU LOBPCG.
fn smeared_one_electron(centre: [f64; 3]) -> (Grid, Vec<Vec<f64>>, Vec<f64>) {
    use crate::eigen::{jacobi_precond, lobpcg, LocalHamiltonian};
    use crate::nuclear::{nuclear_potential, Nucleus};
    let g = Grid::cube(41, 0.25);
    let nuc = [Nucleus { pos: centre, z: 10 }];
    let v = nuclear_potential(&g, &nuc, 0.25);
    let op = LocalHamiltonian { grid: &g, v: &v };
    let mut x = crate::scf::initial_orbitals(&g, &nuc, 5, 11);
    let mut e = vec![0.0; 5];
    let st = {
        let pc = jacobi_precond(&g, &v, -10.0);
        lobpcg(&g, &op, &mut x, &mut e, Some(&pc), 1e-8, 1000)
    };
    eprintln!("Z = 10, σ = h = 0.25 at {centre:?}: ε = {e:?} ({st:?})");
    assert!(st.max_residual < 1e-8, "{st:?}");
    drop(op);
    (g, x, e)
}

#[cfg_attr(debug_assertions, ignore)]
#[test]
fn smeared_one_electron_atom_is_labelled_by_nodes() {
    for centre in [[0.0; 3], OFF] {
        let (g, x, e) = smeared_one_electron(centre);
        let ch = Chan::from_grid(&g, &x, &e, 5, centre);
        let chars: Vec<[f64; 4]> = ch.moments.iter().map(character).collect();
        let nodes: Vec<Option<usize>> = (0..5).map(|i| profile_nodes(&ch.radial[i], &ch.moments[i], dominant_l(&ch.moments[i]).unwrap())).collect();
        eprintln!("W_l {chars:?}\nnodes {nodes:?}");
        let case = Case::new(ch, Chan::synthetic(&[], 0), true);
        let out = Tracker::new().update(&case.input());
        let l = labels(&out, Spin::Up);
        eprintln!("labels {l:?}  {:?}", out.lone_atom);
        let (s_idx, p0) = if dominant_l(&case.ch[0].moments[1]) == Some(0) { (1, 2) } else { (4, 1) };
        assert_eq!(l[0], atomic(1, 0, ""));
        assert_eq!(l[s_idx], atomic(2, 0, ""));
        assert_eq!(l[p0..p0 + 3].to_vec(), vec![atomic(2, 1, "x"), atomic(2, 1, "y"), atomic(2, 1, "z")]);
        let inverted = e[s_idx] > e[p0];
        assert!(matches!(out.lone_atom, LoneAtom::Labelled { shell_order_inverted } if shell_order_inverted == inverted));
    }
}

#[cfg_attr(debug_assertions, ignore)]
#[test]
fn hydrogen_one_s_from_the_cpu_scf() {
    use crate::nuclear::Nucleus;
    use crate::scf::{scf, ScfSettings, System};
    let g = Grid::cube(33, 0.25);
    let sys = System::new(vec![Nucleus { pos: [0.0; 3], z: 1 }], 0, None);
    let (st, stats) = scf(&g, &sys, &ScfSettings::for_grid(&g), None).unwrap();
    assert!(stats.delta_rho < 1e-6);
    let c = [0.0; 3];
    let up = Chan::from_grid(&g, &st.orbitals_up, &st.evals_up, 1, c);
    let dn = Chan::from_grid(&g, &st.orbitals_dn, &st.evals_dn, 0, c);
    assert!(dn.evals.is_empty());
    let mut case = Case::new(up, dn, true);
    case.z_sigma = Some(1.0 * g.h);
    case.tol = 1e-6;
    let out = Tracker::new().update(&case.input());
    let r = rows(&out, Spin::Up);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].label, atomic(1, 0, ""));
    assert!(!r[0].smeared_1s, "Zσ = 0.25 < 0.5");
    assert!(rows(&out, Spin::Down).is_empty(), "↓: none occupied");
    let disp = apply_rotation(&st.orbitals_up, &out, Spin::Up);
    let norm = g.dot(&disp[0], &disp[0]);
    let f = first_sign_functionals(&g, &disp[0], &[c]);
    eprintln!("H 1s: ε = {:.6}, Σψ̃²h³ = {norm:.15}, windowed monopole {:.6}", st.evals_up[0], f[0]);
    assert!((norm - 1.0).abs() <= 1e-10);
    assert!(f[0] > SIGN_MOMENT_MIN, "sign positive by the windowed monopole");
    assert!(apply_rotation(&st.orbitals_dn, &out, Spin::Down).is_empty());
}
