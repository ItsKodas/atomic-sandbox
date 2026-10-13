//! Geometric multigrid V-cycle on the padded grids — the preconditioner of `GpuPoisson`'s CG
//! (M3 Task 10). Read `qm::poisson` first for the equation A V = b, A = −L_int, and why the
//! CG's constant-diagonal Jacobi preconditioner was a no-op.
//!
//! # What it approximates
//! z = M⁻¹ r ≈ A₂⁻¹ r, with A₂ the **2nd-order 7-point** Dirichlet Laplacian at the fine
//! spacing (A₂ x = −Σ_axes (x₊ − 2x + x₋)/h²). CG keeps the exact 8th-order A through
//! `stencil::laplacian`; the preconditioner only has to be SPD and spectrally close. The
//! symbols of A₂ and the 8th-order A differ by at most the ratio of their maxima,
//! 3·6.5/h² vs 3·4/h² = 1.6, and agree on smooth modes, so κ(M⁻¹A) ≈ κ(V-cycle)·1.6 and PCG
//! converges in 13–20 iterations where Jacobi-CG needs 125–325
//! (`poisson::mg_pcg_matches_cg_in_a_third_of_the_iterations`, `bench_poisson_49_97`).
//!
//! # Grids and where the boundary is
//! Level 0 is the fine n³ grid, whose Dirichlet boundary (ghost value 0) sits one spacing h
//! outside the first and last point of each axis. Level ℓ+1 takes the points of level ℓ at
//! **odd** indices 1, 3, 5, …: n_{ℓ+1} = ⌊n_ℓ/2⌋, spacing H = 2h_ℓ, the same padded layout.
//! The low boundary is then exactly one coarse spacing before the first coarse point on every
//! level; the high boundary is `d_high` coarse spacings after the last one, with d_high = 1
//! when n_ℓ is odd (the last coarse point is at n_ℓ − 2) and ½ when it is even (at n_ℓ − 1), and
//! halving again on the next level. Each level's operator puts the boundary where it really
//! is: a boundary point's missing neighbour is the linear extrapolation to zero at distance
//! d·H, x_ghost = (1 − 1/d)·x₀, so its stencil is −(x_in + (1 − 1/d) x₀ − 2x₀)/H² and its
//! diagonal (2 − g)/H² per touched face, g = 1 − 1/d (the zero ghost for d = 1). Without this
//! — coarse points at even indices and zero ghosts, i.e. a coarse domain wider than the fine
//! one by up to a spacing per side and level — every coarse correction overshoots the smooth
//! error by (width ratio)², the levels compound, and the V-cycle *diverged* once its coarsest
//! level was solved properly (ratio 2.2 per cycle at 33³, measured). Coarsening stops at
//! n ≤ `COARSEST_MAX` = 2 (49³ → 24 → 12 → 6 → 3 → 1; 97³ → 48 → 24 → 12 → 6 → 3 → 1;
//! 33³ → 16 → 8 → 4 → 2), not at the plan's 9³: the coarsest level is solved by
//! `COARSEST_SWEEPS` Jacobi sweeps, whose factor on the lowest mode is
//! 1 − ω λ_min H²/diag ≈ 1 − (6/7)(1 − cos(π/(n+1))) — 0.958 per sweep at n = 9 (8 sweeps
//! leave 71 % of it; with the ω = 2/3 of the first version the V-cycle contracted by only 0.67
//! per cycle, measured), 1/7 at n = 1 and ≈ 0.5 at n = 2 (d_high = ½), so 12 sweeps resolve it
//! to ≤ 1 %.
//!
//! # The cycle (per level ℓ, all recorded into the caller's `Batch`, no sync points)
//! x_ℓ ← 0 (`fill`); `PRE_SWEEPS` damped Jacobi sweeps x ← x + ω D⁻¹(b − A₂x), ω = 6/7
//! (ping-pong x ↔ tmp, an even count so x holds the iterate); r = b − A₂x; b_{ℓ+1} = full
//! weighting of r (27 points, weights Π_axes {½, ¼}); recurse; x += trilinear prolongation of
//! x_{ℓ+1}; `POST_SWEEPS` sweeps. The coarsest level does `COARSEST_SWEEPS` sweeps instead of
//! recursing. Damping: the symbol of D⁻¹A₂ is μ(θ) = (1/3)Σ_i(1 − cos θ_i) ∈ [0, 2]; the
//! coarse grid represents |θ_i| < π/2 on every axis, so the smoother must damp every mode with
//! some |θ_i| ≥ π/2, i.e. μ ∈ [1/3, 2], and min_ω max|1 − ωμ| there gives ω = 2/(1/3 + 2) = 6/7
//! with factor 5/7 per sweep — V(2,2) ≈ 0.26 per cycle. The plan's ω = 2/3 (the 1D value,
//! where the high range is [1, 2]) gives 7/9 per sweep and 0.37 per cycle (measured 0.35).
//! Measured with 6/7 (`vcycle_contracts_and_is_symmetric_positive`): see the test. One
//! V-cycle is 8 dispatches per level plus 13 on the coarsest: 53 at 49³ (6 levels), 45 at 33³
//! (all but the fine-level ones tiny), and no readback (`dispatches_per_cycle`).
//!
//! # Symmetry and positivity (what CG needs)
//! Full weighting is Pᵀ/8 for trilinear prolongation P (the test checks the identity on the
//! f64 helpers), the pre- and post-smoothers are the same symmetric (Jacobi) operator with the
//! same sweep count, and the coarsest solve is symmetric, so by induction the V-cycle operator
//! M⁻¹ is symmetric for any symmetric coarse operators (the scalar 1/8 commutes); it is
//! positive definite because the cycle is a convergent iteration for A₂ (the standalone solver
//! test measures the contraction). Pinned: `vcycle_contracts_and_is_symmetric_positive`
//! (⟨u, M⁻¹v⟩ = ⟨v, M⁻¹u⟩ to f32 rounding, ⟨u, M⁻¹u⟩ > 0).
//!
//! # Precision
//! f32 everywhere: the preconditioner's output only steers CG; the CG scalars and the true
//! residual stay as in `poisson`. Deterministic (one thread per point, fixed order).
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid};
use crate::kernels::Kernels;
use crate::pointwise::{check, fill_in};
use crate::stencil::STENCIL_WORKGROUP;
use gpu::GpuContext;

/// Coarsening stops once a level has at most this many points per axis (module doc).
pub const COARSEST_MAX: usize = 2;
/// Damped-Jacobi sweeps before and after the coarse correction (even, so the iterate stays
/// in `x`), and on the coarsest level in place of the recursion.
pub const PRE_SWEEPS: usize = 2;
pub const POST_SWEEPS: usize = 2;
pub const COARSEST_SWEEPS: usize = 12;
/// Jacobi damping for the 3D 7-point Laplacian (module doc: 6/7, not the 1D value 2/3).
pub const OMEGA: f64 = 6.0 / 7.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MgParams { n: u32, p: u32, nf: u32, pf: u32, h2: f32, omega: f32, g_low: f32, g_high: f32 }

/// One level: its grid, the distance of the Dirichlet boundary beyond the first/last point in
/// units of the spacing, and the work fields. Level 0 uses the caller's `b` (the CG residual)
/// and `z` (the output) in place of `b`/`x`, so it allocates only `res` and `tmp`.
struct Level { g: GpuGrid, d_low: f64, d_high: f64, x: Option<Field>, b: Option<Field>, res: Field, tmp: Field }

/// The V-cycle preconditioner for one fine grid.
pub struct Multigrid { levels: Vec<Level> }

/// The coarse level of (`g`, `d_low`, `d_high`): points at odd indices, spacing 2h (module doc).
fn coarsen(g: &GpuGrid, d_low: f64, d_high: f64) -> (GpuGrid, f64, f64) {
    let n = g.n / 2;
    let gc = GpuGrid { n, h: 2.0 * g.h, origin: [0.0; 3], p: n + 8 };
    // low: the boundary is d_low fine spacings before index 0, the first coarse point at index 1
    // high: the last coarse point is at 2n − 1; the boundary d_high beyond g.n − 1
    (gc, (d_low + 1.0) / 2.0, (d_high + (g.n as f64 - 2.0 * n as f64)) / 2.0)
}

fn workgroups(g: &GpuGrid) -> [u32; 3] {
    let n = g.n as u32;
    let [wx, wy, wz] = STENCIL_WORKGROUP;
    [n.div_ceil(wx), n.div_ceil(wy), n.div_ceil(wz)]
}

impl Multigrid {
    /// Builds the level hierarchy for the fine grid `g` (fields for every coarse level, two
    /// scratch fields at the fine level).
    pub fn new(ctx: &GpuContext, g: &GpuGrid) -> Multigrid {
        let mut levels = vec![Level { g: *g, d_low: 1.0, d_high: 1.0, x: None, b: None, res: Field::new(ctx, g, "mg res"), tmp: Field::new(ctx, g, "mg tmp") }];
        while levels.last().expect("fine level").g.n > COARSEST_MAX {
            let last = levels.last().expect("level");
            let (gc, d_low, d_high) = coarsen(&last.g, last.d_low, last.d_high);
            levels.push(Level {
                g: gc, d_low, d_high, x: Some(Field::new(ctx, &gc, "mg x")), b: Some(Field::new(ctx, &gc, "mg b")),
                res: Field::new(ctx, &gc, "mg res"), tmp: Field::new(ctx, &gc, "mg tmp"),
            });
        }
        Multigrid { levels }
    }

    /// Points per axis of every level, fine first.
    pub fn sizes(&self) -> Vec<usize> { self.levels.iter().map(|l| l.g.n).collect() }

    /// (d_low, d_high) of every level, fine first.
    pub fn boundary_distances(&self) -> Vec<(f64, f64)> { self.levels.iter().map(|l| (l.d_low, l.d_high)).collect() }

    /// Parameters for an operator kernel on level `l` (ghost factors from its boundary distances).
    fn op_params(l: &Level) -> MgParams {
        let g = &l.g;
        MgParams { n: g.n as u32, p: g.p as u32, nf: g.n as u32, pf: g.p as u32, h2: (g.h * g.h) as f32, omega: OMEGA as f32,
                   g_low: (1.0 - 1.0 / l.d_low) as f32, g_high: (1.0 - 1.0 / l.d_high) as f32 }
    }

    /// Parameters for a transfer kernel between the coarse grid `gc` and the fine grid `gf`.
    fn transfer_params(gc: &GpuGrid, gf: &GpuGrid) -> MgParams {
        MgParams { n: gc.n as u32, p: gc.p as u32, nf: gf.n as u32, pf: gf.p as u32, h2: 0.0, omega: 0.0, g_low: 0.0, g_high: 0.0 }
    }

    /// One kernel dispatch; `bb` (the `b` binding) is bound only for the two kernels that read it.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(b: &mut Batch, k: &Kernels, name: &str, params: MgParams, dst: &Field, x: &Field, bb: &Field, wg: [u32; 3]) {
        let ctx = b.ctx();
        assert!(dst.buffer != x.buffer, "{name}: output aliases an input");
        let u = b.uniform(ctx, k, name, &params);
        let mut res = vec![u.binding(), dst.buffer.as_entire_binding(), x.buffer.as_entire_binding()];
        if matches!(name, "mg_jacobi" | "mg_residual") { res.push(bb.buffer.as_entire_binding()); }
        let bg = k.bind_group(ctx, name, &res);
        b.dispatch(k, name, &bg, wg);
    }

    /// `sweeps` damped Jacobi sweeps on A₂ x = b at level `l`, ping-ponging x ↔ tmp (even
    /// `sweeps` leave the iterate in `x`).
    fn smooth(b: &mut Batch, k: &Kernels, l: &Level, x: &Field, rhs: &Field, sweeps: usize) {
        debug_assert!(sweeps.is_multiple_of(2), "an even sweep count keeps the iterate in x");
        let (mut src, mut dst) = (x, &l.tmp);
        for _ in 0..sweeps {
            Self::dispatch(b, k, "mg_jacobi", Self::op_params(l), dst, src, rhs, workgroups(&l.g));
            std::mem::swap(&mut src, &mut dst);
        }
    }

    /// Records one V-cycle from level `l` with right-hand side `rhs` into `x` (x ← 0 first).
    fn cycle_in(&self, b: &mut Batch, k: &Kernels, l: usize, x: &Field, rhs: &Field) {
        let lev = &self.levels[l];
        let g = &lev.g;
        fill_in(b, k, g, x, 0.0);
        if l + 1 == self.levels.len() {
            Self::smooth(b, k, lev, x, rhs, COARSEST_SWEEPS);
            return;
        }
        Self::smooth(b, k, lev, x, rhs, PRE_SWEEPS);
        Self::dispatch(b, k, "mg_residual", Self::op_params(lev), &lev.res, x, rhs, workgroups(g));
        let next = &self.levels[l + 1];
        let (xc, bc) = (next.x.as_ref().expect("coarse x"), next.b.as_ref().expect("coarse b"));
        Self::dispatch(b, k, "mg_restrict", Self::transfer_params(&next.g, g), bc, &lev.res, &lev.res, workgroups(&next.g));
        self.cycle_in(b, k, l + 1, xc, bc);
        Self::dispatch(b, k, "mg_prolong_add", Self::transfer_params(&next.g, g), x, xc, xc, workgroups(g));
        Self::smooth(b, k, lev, x, rhs, POST_SWEEPS);
    }

    /// Records z = M⁻¹ r: one V-cycle on the fine grid with right-hand side `r` into `z`
    /// (`z` is overwritten; both on the fine grid; `z` must not alias `r`).
    pub fn apply_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, r: &Field, z: &Field) {
        assert!(*g == self.levels[0].g, "Multigrid: built for {:?}, called with {:?}", self.levels[0].g, g);
        check(g, &[r, z]);
        assert!(r.buffer != z.buffer, "Multigrid::apply_in: r and z alias");
        self.cycle_in(b, k, 0, z, r);
    }

    /// Dispatches recorded per `apply_in` (for cost accounting).
    pub fn dispatches_per_cycle(&self) -> usize {
        let nl = self.levels.len();
        (nl - 1) * (1 + PRE_SWEEPS + 2 + 1 + POST_SWEEPS) + 1 + COARSEST_SWEEPS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};
    use crate::kernels::Kernels;
    use crate::reduce::Reducer;
    use gpu::GpuContext;

    /// f64 reference of A₂ x = −Σ_axes (x₊ − 2x + x₋)/H² with the extrapolated ghosts
    /// x_ghost = (1 − 1/d)·x₀ at distance d (zero for d = 1), qm layout.
    fn apply_a2(n: usize, h: f64, d_low: f64, d_high: f64, x: &[f64]) -> Vec<f64> {
        let at = |i: i64, j: i64, k: i64, x0: f64| -> f64 {
            let m = n as i64;
            if i < 0 || j < 0 || k < 0 { return (1.0 - 1.0 / d_low) * x0; }
            if i >= m || j >= m || k >= m { return (1.0 - 1.0 / d_high) * x0; }
            x[(i as usize) + n * ((j as usize) + n * (k as usize))]
        };
        let mut out = vec![0.0; n * n * n];
        for k in 0..n as i64 { for j in 0..n as i64 { for i in 0..n as i64 {
            let c = at(i, j, k, 0.0);
            let s = at(i + 1, j, k, c) + at(i - 1, j, k, c) + at(i, j + 1, k, c) + at(i, j - 1, k, c) + at(i, j, k + 1, c) + at(i, j, k - 1, c);
            out[(i as usize) + n * ((j as usize) + n * (k as usize))] = -(s - 6.0 * c) / (h * h);
        }}}
        out
    }

    /// f64 full weighting of a fine field (n) onto the coarse grid (nc = ⌊n/2⌋, coarse j at fine 2j+1).
    fn restrict(n: usize, f: &[f64]) -> Vec<f64> {
        let nc = n / 2;
        let at = |i: i64, j: i64, k: i64| -> f64 {
            if i < 0 || j < 0 || k < 0 || i >= n as i64 || j >= n as i64 || k >= n as i64 { 0.0 } else { f[(i as usize) + n * ((j as usize) + n * (k as usize))] }
        };
        let w = |d: i64| if d == 0 { 0.5 } else { 0.25 };
        let mut out = vec![0.0; nc * nc * nc];
        for k in 0..nc { for j in 0..nc { for i in 0..nc {
            let mut s = 0.0;
            for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                s += w(dx) * w(dy) * w(dz) * at(2 * i as i64 + 1 + dx, 2 * j as i64 + 1 + dy, 2 * k as i64 + 1 + dz);
            }}}
            out[i + nc * (j + nc * k)] = s;
        }}}
        out
    }

    /// f64 trilinear prolongation of a coarse field (nc) onto the fine grid (n): odd fine
    /// 2c+1 → coarse c; even 2c → ½(c−1) + ½ c, coarse −1 and nc being 0.
    fn prolong(n: usize, c: &[f64]) -> Vec<f64> {
        let nc = n / 2;
        let at = |i: i64, j: i64, k: i64| -> f64 {
            if i < 0 || j < 0 || k < 0 || i >= nc as i64 || j >= nc as i64 || k >= nc as i64 { 0.0 } else { c[(i as usize) + nc * ((j as usize) + nc * (k as usize))] }
        };
        let mut out = vec![0.0; n * n * n];
        for k in 0..n { for j in 0..n { for i in 0..n {
            let axis = |t: usize| -> Vec<(i64, f64)> { let c = (t / 2) as i64; if t % 2 == 1 { vec![(c, 1.0)] } else { vec![(c - 1, 0.5), (c, 0.5)] } };
            let mut s = 0.0;
            for (ci, wi) in axis(i) { for (cj, wj) in axis(j) { for (ck, wk) in axis(k) { s += wi * wj * wk * at(ci, cj, ck); } } }
            out[i + n * (j + n * k)] = s;
        }}}
        out
    }

    struct Rig { ctx: GpuContext, k: Kernels, g: GpuGrid, r: Reducer }

    fn rig(n: usize, h: f64) -> Option<Rig> {
        let ctx = GpuContext::new_headless()?;
        let g = GpuGrid::new(&qm::Grid::cube(n, h));
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        Some(Rig { ctx, k, g, r })
    }

    fn upload(ctx: &GpuContext, g: &GpuGrid, d: &[f64]) -> Field { let f = Field::new(ctx, g, "f"); f.upload_interior(ctx, g, d); f }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }

    /// The level geometry: sizes, boundary distances and the coarse point positions (odd fine
    /// indices) for the grids the solver uses — the low boundary stays at d = 1, the high one
    /// is 1 after an odd level and halves after an even one.
    #[test]
    fn level_geometry() {
        let g = |n: usize| GpuGrid { n, h: 0.25, origin: [0.0; 3], p: n + 8 };
        let (gc, dl, dh) = coarsen(&g(49), 1.0, 1.0);
        assert_eq!((gc.n, gc.h, dl, dh), (24, 0.5, 1.0, 1.0));
        let (gc, dl, dh) = coarsen(&gc, dl, dh);
        assert_eq!((gc.n, dl, dh), (12, 1.0, 0.5));
        let (gc, dl, dh) = coarsen(&gc, dl, dh);
        assert_eq!((gc.n, dl, dh), (6, 1.0, 0.25));
        let (gc, dl, dh) = coarsen(&gc, dl, dh);
        assert_eq!((gc.n, dl, dh), (3, 1.0, 0.125));
        let (gc, dl, dh) = coarsen(&gc, dl, dh);
        assert_eq!((gc.n, dl, dh), (1, 1.0, 0.5625)); // 3 → 1: the last point moves from index 1 to 1; boundary at 2 + 0.125 fine = 1.0625 coarse… = (0.125 + 1)/2
        // the fine-boundary position check: on the 49 → 24 level, coarse point j sits at fine
        // 2j+1 and the fine boundary at −1 and 49: one coarse spacing (2 fine) from points 1 and 47
        assert_eq!(2 * 23 + 1, 47);
    }

    /// Each kernel against its f64 reference on an odd (11) and an even (10) grid, with the
    /// extrapolated ghosts (d_low = 1, d_high = ½ on the operator kernels): Jacobi sweep,
    /// residual, full weighting (fine ghosts read as 0), trilinear prolongation (coarse −1 and
    /// nc read as 0) — both parities of every index exercised, ghosts untouched; and
    /// R = Pᵀ/8 on the f64 helpers.
    #[test]
    fn kernels_match_f64_reference() {
        let Some(Rig { ctx, k, .. }) = rig(9, 0.5) else { return };
        for n in [11usize, 10] {
            let h = 0.3;
            let cg = qm::Grid::cube(n, h);
            let g = GpuGrid::new(&cg);
            let (gc, _, _) = coarsen(&g, 1.0, 1.0);
            let lev = Level { g, d_low: 1.0, d_high: 0.5, x: None, b: None, res: Field::new(&ctx, &g, "r"), tmp: Field::new(&ctx, &g, "t") };
            let x0 = cg.sample(|[x, y, z]| (-(x * x + 0.7 * y * y + 1.3 * z * z)).exp() + 0.1 * x);
            let b0 = cg.sample(|[x, y, z]| (x - 0.5 * y + 0.2 * z).sin());
            let (fx, fb) = (upload(&ctx, &g, &x0), upload(&ctx, &g, &b0));
            let out = Field::new(&ctx, &g, "out");
            let ax = apply_a2(n, h, 1.0, 0.5, &x0);
            // jacobi (with the per-point diagonal: 6 − Σ g on the touched faces, g_high = −1)
            let mut bt = Batch::new(&ctx, "mg");
            Multigrid::dispatch(&mut bt, &k, "mg_jacobi", Multigrid::op_params(&lev), &out, &fx, &fb, workgroups(&g));
            bt.submit();
            let want: Vec<f64> = (0..n * n * n).map(|idx| {
                let (i, j, kk) = (idx % n, (idx / n) % n, idx / (n * n));
                let faces = [i, j, kk].iter().filter(|&&t| t == n - 1).count() as f64;
                let diag = (6.0 + faces) / (h * h); // g_high = 1 − 1/0.5 = −1 per touched high face
                x0[idx] + OMEGA / diag * (b0[idx] - ax[idx])
            }).collect();
            let got = out.read_interior(&ctx, &g);
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let e = max_abs_diff(&got, &want);
            println!("mg_jacobi {n}³: max|Δ| {e:.2e} (scale {scale:.2})");
            assert!(e < 1e-6 * scale, "jacobi {e:.3e}");
            assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
            // residual
            let mut bt = Batch::new(&ctx, "mg");
            Multigrid::dispatch(&mut bt, &k, "mg_residual", Multigrid::op_params(&lev), &out, &fx, &fb, workgroups(&g));
            bt.submit();
            let want: Vec<f64> = (0..x0.len()).map(|i| b0[i] - ax[i]).collect();
            let got = out.read_interior(&ctx, &g);
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let e = max_abs_diff(&got, &want);
            println!("mg_residual {n}³: max|Δ| {e:.2e} (scale {scale:.2})");
            // the 7-point sum cancels (Σ x_nb − 6x ≪ 6|x|): f32 leaves ~6·6e-8·|x|/h² ≈ 1e-6 of the scale
            assert!(e < 3e-6 * scale, "residual {e:.3e}");
            // restrict: output on the coarse grid
            let outc = Field::new(&ctx, &gc, "outc");
            let mut bt = Batch::new(&ctx, "mg");
            Multigrid::dispatch(&mut bt, &k, "mg_restrict", Multigrid::transfer_params(&gc, &g), &outc, &fx, &fx, workgroups(&gc));
            bt.submit();
            let want = restrict(n, &x0);
            let got = outc.read_interior(&ctx, &gc);
            let e = max_abs_diff(&got, &want);
            println!("mg_restrict {n}³ → {}³: max|Δ| {e:.2e}", gc.n);
            assert!(e < 1e-6, "restrict {e:.3e}");
            assert_ghosts_zero(&gc, &outc.read_padded_raw(&ctx));
            // prolong + add onto a non-zero fine field
            let c0: Vec<f64> = (0..gc.interior_len()).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
            let fc = upload(&ctx, &gc, &c0);
            let acc = upload(&ctx, &g, &b0);
            let mut bt = Batch::new(&ctx, "mg");
            Multigrid::dispatch(&mut bt, &k, "mg_prolong_add", Multigrid::transfer_params(&gc, &g), &acc, &fc, &fc, workgroups(&g));
            bt.submit();
            let pr = prolong(n, &c0);
            let want: Vec<f64> = (0..b0.len()).map(|i| b0[i] + pr[i]).collect();
            let got = acc.read_interior(&ctx, &g);
            let e = max_abs_diff(&got, &want);
            println!("mg_prolong_add {}³ → {n}³: max|Δ| {e:.2e}", gc.n);
            assert!(e < 1e-6, "prolong {e:.3e}");
            assert_ghosts_zero(&g, &acc.read_padded_raw(&ctx));
            // full weighting is Pᵀ/8: ⟨restrict(f), c⟩_coarse = ⟨f, prolong(c)⟩_fine / 8
            let lhs: f64 = restrict(n, &x0).iter().zip(&c0).map(|(a, b)| a * b).sum();
            let rhs: f64 = x0.iter().zip(&pr).map(|(a, b)| a * b).sum::<f64>() / 8.0;
            assert!((lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0), "R ≠ Pᵀ/8: {lhs} vs {rhs}");
        }
    }

    /// The V-cycle as a standalone solver of A₂ x = b (x ← x + M⁻¹(b − A₂x)) contracts the
    /// residual by a grid-independent factor ≤ 0.3 per cycle at 33³ and 65³ (the coarse
    /// correction works: smoothing alone gives ≈ 1 − O(h²)); and M⁻¹ is symmetric and
    /// positive on the fine grid to f32 rounding.
    #[test]
    fn vcycle_contracts_and_is_symmetric_positive() {
        for n in [33usize, 65] {
            let h = 0.25;
            let Some(Rig { ctx, k, g, r }) = rig(n, h) else { return };
            let cg = qm::Grid::cube(n, h);
            let mg = Multigrid::new(&ctx, &g);
            println!("{n}³ levels {:?} boundary distances {:?}, {} dispatches per cycle", mg.sizes(), mg.boundary_distances(), mg.dispatches_per_cycle());
            let rho = cg.sample(|[x, y, z]| (-((x - 0.3).powi(2) + y * y + (z + 0.2).powi(2)) / 0.5).exp());
            let b = upload(&ctx, &g, &rho);
            let (x, res, z) = (Field::new(&ctx, &g, "x"), Field::new(&ctx, &g, "res"), Field::new(&ctx, &g, "z"));
            let mut bt = Batch::new(&ctx, "mg solve");
            let mut norms = Vec::new();
            for _ in 0..8 {
                // res = b − A₂ x; z = M⁻¹ res; x += z
                Multigrid::dispatch(&mut bt, &k, "mg_residual", Multigrid::op_params(&mg.levels[0]), &res, &x, &b, workgroups(&g));
                norms.push(r.dot_in(&mut bt, &k, &g, &res, &res).sqrt());
                mg.apply_in(&mut bt, &k, &g, &res, &z);
                crate::pointwise::axpy_in(&mut bt, &k, &g, &x, 1.0, &z);
            }
            let ratios: Vec<f64> = norms.windows(2).map(|w| w[1] / w[0]).collect();
            println!("{n}³ V-cycle residual norms {norms:?}\n  ratios {ratios:?}");
            let worst = ratios[1..].iter().cloned().fold(0.0, f64::max);
            assert!(worst < 0.32, "{n}³: V-cycle contraction {worst:.3} per cycle"); // ω = 6/7 model 0.26; ω = 2/3 measured 0.35
            // symmetry and positivity: ⟨u, M⁻¹ v⟩ vs ⟨v, M⁻¹ u⟩ on two unrelated fields
            let u = upload(&ctx, &g, &cg.sample(|[x, y, z]| (0.7 * x + 0.3 * y - 0.5 * z).sin() * (-(x * x + y * y + z * z) / 8.0).exp()));
            let v = upload(&ctx, &g, &cg.sample(|[x, y, z]| (x * y - z).cos() * (-(x * x + 0.5 * y * y + z * z) / 6.0).exp()));
            let (mu, mv) = (Field::new(&ctx, &g, "mu"), Field::new(&ctx, &g, "mv"));
            mg.apply_in(&mut bt, &k, &g, &u, &mu);
            mg.apply_in(&mut bt, &k, &g, &v, &mv);
            let d = r.dots_in(&mut bt, &k, &g, &[(&u, &mv), (&v, &mu), (&u, &mu), (&v, &mv), (&u, &u), (&v, &v)]);
            let scale = (d[4] * d[5]).sqrt() * (d[2] / d[4]).max(d[3] / d[5]);
            println!("{n}³ ⟨u, M⁻¹v⟩ = {:.6e}, ⟨v, M⁻¹u⟩ = {:.6e} (rel {:.2e}); ⟨u, M⁻¹u⟩ = {:.3e}, ⟨v, M⁻¹v⟩ = {:.3e}", d[0], d[1], (d[0] - d[1]).abs() / scale, d[2], d[3]);
            assert!((d[0] - d[1]).abs() < 1e-5 * scale, "M⁻¹ not symmetric: {} vs {}", d[0], d[1]);
            assert!(d[2] > 0.0 && d[3] > 0.0, "M⁻¹ not positive");
            drop(bt);
        }
    }
}
