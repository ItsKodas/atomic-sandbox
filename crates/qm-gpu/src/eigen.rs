//! LOBPCG on the GPU — the port of `qm::eigen::lobpcg` (read its module doc first: the
//! Rayleigh–Ritz derivation, the orthogonalised-P variant, the multi-sweep damped-Jacobi
//! preconditioner with ω = 4/(3ρ) from the stencil symbol, its SPD condition and the 0.1
//! diagonal floor). The algorithm — S = [X, W, P] by MGS, A = SᵀHS, X_new = S·C_k,
//! P_new = S_{W,P}·C_k — is unchanged; this doc covers what the GPU (f32 storage, readbacks as
//! sync points) changes.
//!
//! # What stays f64
//! The ≤ 3k×3k Rayleigh–Ritz (nalgebra), every MGS coefficient, λ, the residual norms and every
//! reduction finish (`Reducer`: f32 workgroup partials → Neumaier f64). A coefficient is cast to
//! f32 only when a `lincomb` consumes it. H x = `stencil::kinetic` + `pointwise::mul_add` (V·x).
//!
//! # Relative MGS drop threshold (plan Review Focus 4)
//! The CPU drops a vector whose norm after projection is < 1e-12 — an absolute threshold that
//! an f32 vector of norm ~1 can never trip (its rounding noise alone is ~1e-7). Here a vector
//! is dropped when ‖v − Σ c_q q‖ < `MGS_DROP`·‖v‖₀ = 1e-6 ‖v‖₀, ‖v‖₀ its norm before the
//! projection. Why 1e-6: the projection coefficients come from f32 dot products (measured
//! relative accuracy ~1e-8 in `reduce`), so one MGS pass leaves a spurious component
//! ~1e-8·‖v‖₀ along each kept direction; a survivor at ratio 1e-6 is therefore orthogonal to
//! the kept set to ~1% (its own rounding noise is ~1e-7·‖v‖₀, i.e. 10% of it), which keeps the
//! Gram matrix of S well conditioned (κ ≲ 1.3); below 1e-6 the leftover would be O(1) and S
//! would lose rank. The CPU's 1e-12 is the same construction with ε_f64 (1e-16 × 10⁴).
//!
//! # MGS with deferred norms (one readback per vector)
//! Vectors are never normalised; kept columns carry their measured ⟨q,q⟩ and the projection
//! coefficient is ⟨q,v⟩/⟨q,q⟩. Processing v needs one readback: ⟨q,v⟩ for every kept q, ⟨v,v⟩
//! (= ‖v‖₀²), and — for the column projected just before — ⟨pend,v⟩ and ⟨pend,pend⟩, which is
//! the first time its post-projection norm is seen; its keep/drop decision is taken then, so
//! every vector costs one `lincomb` and one readback. X (k orthonormal Ritz vectors from the
//! previous step, orthonormal to ~1e-7) is not re-projected; its ⟨x,x⟩ come with the residual
//! readback. The last column's norm arrives with the Gram readback (below).
//!
//! # Rayleigh–Ritz with the measured Gram matrix
//! Because columns are unnormalised and X is only orthonormal to rounding, the projected
//! problem is solved as the generalised one, A c = θ B c with A = SᵀHS and B = SᵀS *both
//! measured* (lower triangles, one readback, symmetrised). After column-normalising with
//! D = diag(B)^{-1/2}, B̃ = D B D = I + O(1e-2) (MGS), so B̃ = U Σ Uᵀ, T = U Σ^{-1/2},
//! Â = TᵀÃT is symmetric-eigen-decomposed, C = D T Y (directions with σ < 1e-10 σ_max would be
//! discarded; MGS guarantees none are). The Ritz vectors are B-orthonormal in f64 exactly, so
//! X_new is orthonormal up to the f32 rounding of the final `lincomb` (~1e-7) whatever MGS
//! achieved — the f32 solver never *assumes* SᵀS = I. The CPU's extra re-orthonormalisation of
//! X_new is therefore not needed and not done.
//!
//! # The iteration (readbacks counted; k = number of eigenpairs, m ≤ 3k)
//! 1. HX = H X (recomputed from the stored X every iteration, so residuals are always true
//!    residuals), R = HX − θX with θ the previous Ritz values (first iteration: measured λ),
//!    then — *before* the convergence test, speculatively — W = T R (per vector: `inv_diag`
//!    with its own λ_i, `mul`, then `JACOBI_SWEEPS` − 1 × (H w, `jacobi_sweep`)), and the dots
//!    for the first MGS vector. **Readback 1**: ⟨x_i,Hx_i⟩, ⟨x_i,x_i⟩, ‖r_i‖², ⟨x_j,w_0⟩, ⟨w_0,w_0⟩.
//!    λ_i = ⟨x_i,Hx_i⟩/⟨x_i,x_i⟩ (the reported eigenvalues; θ and λ agree to ~1e-12 because the
//!    Rayleigh quotient is stationary), residual_i = ‖r_i‖/‖x_i‖. Stop test.
//! 2. MGS of w_1 … w_{k−1}, p_0 … p_{k−1}: **readbacks 2 … 2k**.
//! 3. H S for the W/P columns (HX is reused).
//! 4. **Readback 2k+1**: A and B lower triangles (m(m+1) dots) — the last column's ‖·‖² is B_mm,
//!    which settles its keep/drop.
//! 5. Rayleigh–Ritz in f64; X_new = S·C_k, P_new = S_{W,P}·C_k by `lincomb` (f64 → f32
//!    coefficients); the new fields are swapped in, no copies.
//!
//! 2k + 1 readbacks per iteration (k + 1 on the first, P empty), plus k + 2 once for the guess.
//! The speculative W costs 11k dispatches on the exit iteration only, and saves one readback
//! per iteration. All readbacks go through `Reducer::dots_in`, which reuses one scratch buffer.
//!
//! # Re-seeding a dependent guess
//! The guess is orthogonalised by the same MGS; a guess vector dropped as dependent is replaced
//! by deterministic noise (LCG, seeded by its slot) under a Gaussian envelope and re-projected,
//! so `lobpcg` always returns k vectors. White noise has nonzero overlap with every eigenvector,
//! so the lowest k are still found (`degenerate_guess_does_not_break` checks the eigenvalues,
//! not only the count). X is never dropped later: the Ritz vectors are B-orthonormal by
//! construction. Dropped W/P columns simply shrink m (as on the CPU).
//!
//! # The f32 residual floor
//! Any vector stored in f32 carries rounding noise δx of relative RMS ≈ 2.4e-8 (uniform ±½ ulp
//! over a binade), white on the grid. Its residual (H − λ)δx is white noise filtered by the
//! kinetic symbol, so ‖(H − λ)δx‖ ≈ 2.4e-8·‖T‖_RMS·‖x‖ with ‖T‖²_RMS = mean² + var over the
//! Brillouin zone, mean = 3|c₀|/(2h²), var = 3·(2Σ_{m≥1}c_m²)/(2h²)²: 75 Ha at h = 0.25,
//! 118 Ha at h = 0.2 (V − λ shifts the mean by O(1), negligible). `residual_floor_estimate`
//! gives 1.8e-6 at h = 0.25 and 2.8e-6 at h = 0.2. Each Rayleigh–Ritz step replaces the old
//! noise (the smoother maps it back into W) by fresh noise of the same size, so the *true*
//! residual of the stored X fluctuates around this level and never goes below it — this is not
//! an evaluation artefact of the residual (that part is ~1e-7). It is scale-set by ‖H‖, not by
//! |λ|: relative to the operator norm the floor is ε_f32. The eigenvalues are unaffected at the
//! 1e-6 level: their error is second order (Kato–Temple ‖r‖²/gap ~ 1e-12), as is the density's.
//!
//! Measured (RX 7900 XTX, 33³ h = 0.25, tol 1e-10, `f32_residual_floor_is_documented`): the
//! ground state floors at 1.6–2.0e-6 (model 1.81e-6), stationary from iteration ~55 to 124. The
//! three *degenerate* 2.5 Ha states floor ~2× higher, at 2.7–4.0e-6, because `SymmetricEigen`
//! returns arbitrary directions inside a degenerate cluster: every Rayleigh–Ritz step rotates
//! the cluster with O(1) coefficients, and the f32 `lincomb` of several O(1) terms (plus the
//! chunk boundary at 7 terms) rounds ~√3–2× more than the ground state's single dominant term.
//! k = 2 (one vector in the manifold, nothing to rotate with) floors at 1.04× the model;
//! k = 3 and 4 at ~2×. Aligning cluster eigenvectors with the previous X (Procrustes within
//! numerically degenerate θ) would remove it; not done — eigenvalues and density are second
//! order in the floor. The solver stops on the floor with `Stop::Floor` (below) instead of
//! running to `max_iter`.
//!
//! # Stop reasons
//! `max_residual` is always the true residual of the returned X (recomputed every iteration).
//! `Stop::Converged`: max_i residual_i < tol. `Stop::Floor`: the residual is within
//! `FLOOR_BAND` of `residual_floor_estimate` and its running minimum has not halved for
//! `STALL_WINDOW` iterations — the f32 floor; treat as converged at the reported residual. The
//! band matters: far from the floor the max residual is *not* a progress metric (from a
//! white-noise guess it sits at ~30 Ha for the first 8 iterations while the Ritz values descend
//! 70 → 10 → 2.5; measured, 33³ k = 4), so no stall is diagnosed there and a solve that cannot
//! reach tol for another reason ends as `Stop::MaxIter`, the warning. The window is 12 because
//! the slowest vectors of a block converge slowly near the end — the three degenerate 2.5 Ha
//! oscillator states at 8%/iteration (the smoother is a scalar on low modes, as on the CPU;
//! multigrid is the M3 Task 10 follow-up) — and 12 iterations at 8% is a factor 2.7, so a
//! genuine tail is only mistaken for the floor below 6%/iteration *and* inside the band, where
//! the eigenvalue error is already ≤ (10·floor)²/gap ~ 1e-9 Ha. Callers should key on
//! `max_residual`, not the label (as for `GpuPoisson`), and pass tol ≥ ~2× the estimate when
//! they want `Converged` without the window's extra iterations.
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid};
use crate::kernels::{uniform, workgroups_for, Kernels};
use crate::pointwise::{check, lincomb_in, mul_add_in, mul_in, scale_in, Params};
use crate::reduce::Reducer;
use crate::stencil::kinetic_in;
use gpu::GpuContext;
use nalgebra::{DMatrix, SymmetricEigen};
pub use qm::eigen::JACOBI_SWEEPS;
use qm::eigen::jacobi_omega;
use qm::stencil::COEFF;

/// H = −½∇² + V for a local potential `v` on the grid (a padded field with zero ghosts).
pub struct GpuHamiltonian<'a> { pub v: &'a Field }

impl GpuHamiltonian<'_> {
    /// out = −½∇²x + V·x on the interior (recorded into `b`): `kinetic` then `mul_add`.
    pub fn apply_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, x: &Field, out: &Field) {
        kinetic_in(b, k, g, x, out);
        mul_add_in(b, k, g, out, self.v, x);
    }
}

/// Relative MGS drop threshold: a vector is dropped when its norm after projection is below
/// `MGS_DROP` × its norm before (module doc).
pub const MGS_DROP: f64 = 1e-6;
/// Iterations without the running-minimum max-residual halving, inside the floor band, before
/// the solve stops with `Stop::Floor`.
pub const STALL_WINDOW: usize = 12;
/// The floor band: stalls are diagnosed only at max residual ≤ `FLOOR_BAND` × `residual_floor_estimate`.
pub const FLOOR_BAND: f64 = 10.0;
/// Directions of the column-normalised Gram matrix with eigenvalue below this × the largest are
/// discarded in the Rayleigh–Ritz step (never reached after MGS at `MGS_DROP`).
const RR_DROP: f64 = 1e-10;
/// Relative RMS rounding of an f32 value: uniform ±½ ulp, averaged over a binade (≈ 0.4·2⁻²⁴).
const F32_RMS_ROUNDING: f64 = 2.4e-8;
/// Re-seeding rounds before giving up on an initial guess (noise is independent w.p. 1).
const MAX_RESEED_ROUNDS: usize = 3;

/// Why `lobpcg` returned (module doc). `max_residual` is the true residual in every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// max_i ‖H xᵢ − λᵢ xᵢ‖ < tol.
    Converged,
    /// The residual stopped decreasing at the f32 floor (within `FLOOR_BAND` of the model
    /// estimate). The vectors are as good as f32 storage allows; treat as converged at
    /// `max_residual`.
    Floor,
    /// `max_iter` Rayleigh–Ritz steps taken with the residual still above tol and no floor
    /// signature. A warning (raise `max_iter`, or the solve is not converging).
    MaxIter,
}

#[derive(Debug, Clone, Copy)]
pub struct EigenStats {
    /// Number of Rayleigh–Ritz steps taken (0 = the guess already met `tol`).
    pub iterations: usize,
    /// max_i ‖H xᵢ − λᵢ xᵢ‖_{L²}/‖xᵢ‖ of the returned X, recomputed from the stored vectors.
    pub max_residual: f64,
    pub stop: Stop,
    /// GPU→CPU readbacks (sync points) in this call, guess orthogonalisation included.
    pub readbacks: usize,
}

/// Model of the f32 residual floor ‖(H − λ)δx‖ for ‖x‖ = 1 (module doc): the RMS rounding of
/// the stored vector filtered by the kinetic symbol, 2.4e-8·√(mean² + var).
pub fn residual_floor_estimate(g: &GpuGrid) -> f64 {
    let s = 1.0 / (2.0 * g.h * g.h);
    let mean = 3.0 * COEFF[0].abs() * s;
    let var = 3.0 * 2.0 * COEFF[1..].iter().map(|c| c * c).sum::<f64>() * s * s;
    F32_RMS_ROUNDING * (mean * mean + var).sqrt()
}

/// The stop decision after a residual readback (pure, pinned by `stop_classification`):
/// `max_res` the current max residual, `it` the iterations taken, `best_it` the iteration at
/// which the running-minimum max residual last halved, `floor` = `residual_floor_estimate`.
fn classify(max_res: f64, tol: f64, it: usize, max_iter: usize, best_it: usize, floor: f64) -> Option<Stop> {
    if max_res < tol { return Some(Stop::Converged); }
    if max_res <= FLOOR_BAND * floor && it - best_it >= STALL_WINDOW { return Some(Stop::Floor); }
    if it >= max_iter { return Some(Stop::MaxIter); }
    None
}

/// Preallocated work fields for `lobpcg` with up to `k` eigenpairs on one grid: 8k + 2 fields
/// (HX, R, W, P, H·[W,P], X_new, P_new, D⁻¹, H w).
pub struct LobpcgWork {
    k: usize,
    hx: Vec<Field>, r: Vec<Field>, w: Vec<Field>, p: Vec<Field>, hs: Vec<Field>,
    xn: Vec<Field>, pn: Vec<Field>, d: Field, hw: Field,
    grid: GpuGrid,
}

impl LobpcgWork {
    pub fn new(ctx: &GpuContext, g: &GpuGrid, k: usize) -> LobpcgWork {
        assert!(k > 0, "LobpcgWork::new: k must be positive");
        let fields = |label: &str, count: usize| (0..count).map(|_| Field::new(ctx, g, label)).collect::<Vec<_>>();
        LobpcgWork {
            k, hx: fields("lobpcg hx", k), r: fields("lobpcg r", k), w: fields("lobpcg w", k), p: fields("lobpcg p", k),
            hs: fields("lobpcg hs", 2 * k), xn: fields("lobpcg x_new", k), pn: fields("lobpcg p_new", k),
            d: Field::new(ctx, g, "lobpcg inv_diag"), hw: Field::new(ctx, g, "lobpcg hw"), grid: *g,
        }
    }
}

/// D⁻¹ = 1/max(diag(−½∇²) + V − λ, 0.1) (recorded).
fn inv_diag_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field, v: &Field, diag_t: f64, lambda: f64) {
    let ctx = b.ctx();
    let u = uniform(ctx, "inv_diag", &Params::new(g, diag_t as f32, lambda as f32));
    let bg = k.bind_group(ctx, "inv_diag", &[u.as_entire_binding(), out.buffer.as_entire_binding(), v.buffer.as_entire_binding()]);
    b.dispatch(k, "inv_diag", &bg, [workgroups_for(g.interior_len()), 1, 1]);
}

/// w += ω·D⁻¹·(r − (Hw − λw)) (recorded).
#[allow(clippy::too_many_arguments)]
fn jacobi_sweep_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, w: &Field, d: &Field, r: &Field, hw: &Field, omega: f64, lambda: f64) {
    let ctx = b.ctx();
    let u = uniform(ctx, "jacobi_sweep", &Params::new(g, omega as f32, lambda as f32));
    let bg = k.bind_group(ctx, "jacobi_sweep", &[
        u.as_entire_binding(), w.buffer.as_entire_binding(), d.buffer.as_entire_binding(), r.buffer.as_entire_binding(), hw.buffer.as_entire_binding(),
    ]);
    b.dispatch(k, "jacobi_sweep", &bg, [workgroups_for(g.interior_len()), 1, 1]);
}

/// Deterministic noise under a Gaussian envelope (width = box/4) for re-seeding slot `slot`.
fn reseed(ctx: &GpuContext, g: &GpuGrid, f: &Field, slot: usize, round: usize) {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ ((slot as u64) << 32) ^ (round as u64);
    let n = g.n;
    let sigma2 = ((n - 1) as f64 * g.h / 4.0).powi(2);
    let mut data = Vec::with_capacity(g.interior_len());
    for kk in 0..n { for j in 0..n { for i in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
        let [x, y, z] = [i, j, kk].map(|c| c as f64 * g.h);
        let r2 = (x + g.origin[0]).powi(2) + (y + g.origin[1]).powi(2) + (z + g.origin[2]).powi(2);
        data.push((u - 0.5) * (-r2 / (2.0 * sigma2)).exp());
    }}}
    f.upload_interior(ctx, g, &data);
}

/// A column of S: the field and ⟨s,s⟩ measured on the stored vector.
struct Col<'f> { f: &'f Field, norm2: f64 }

/// MGS with deferred norms (module doc). `kept` are the orthogonal columns (norms known);
/// `pending` was projected but its post-projection norm is measured with the next readback.
struct Mgs<'f> {
    kept: Vec<Col<'f>>,
    pending: Option<(&'f Field, f64 /* ‖v‖₀² before projection */)>,
    dropped: usize,
}

impl<'f> Mgs<'f> {
    fn new(kept: Vec<Col<'f>>) -> Mgs<'f> { Mgs { kept, pending: None, dropped: 0 } }

    /// Dots needed to take `v` in: ⟨q,v⟩ for every kept q, then [⟨pend,v⟩, ⟨pend,pend⟩], then ⟨v,v⟩.
    fn pairs(&self, v: &'f Field) -> Vec<(&'f Field, &'f Field)> {
        let mut pairs: Vec<(&Field, &Field)> = self.kept.iter().map(|q| (q.f, v)).collect();
        if let Some((pf, _)) = self.pending { pairs.push((pf, v)); pairs.push((pf, pf)); }
        pairs.push((v, v));
        pairs
    }

    /// Keep/drop decision for a projected column: norm² `n2` after vs `n0` before the projection.
    fn settle(&mut self, f: &'f Field, n0: f64, n2: f64) -> bool {
        if n2 > MGS_DROP * MGS_DROP * n0 { self.kept.push(Col { f, norm2: n2 }); true } else { self.dropped += 1; false }
    }

    /// Settles the pending column from `dots` (laid out as `pairs`), records
    /// v ← v − Σ_q ⟨q,v⟩/⟨q,q⟩ q over the kept columns, and makes `v` pending.
    fn take(&mut self, b: &mut Batch, k: &Kernels, g: &GpuGrid, v: &'f Field, dots: &[f64]) {
        let nq = self.kept.len();
        let mut terms: Vec<(f32, &Field)> = vec![(1.0, v)];
        for (q, c) in self.kept.iter().zip(dots) { terms.push(((-c / q.norm2) as f32, q.f)); }
        let mut i = nq;
        if let Some((pf, n0)) = self.pending.take() {
            let (pv, pp) = (dots[i], dots[i + 1]);
            i += 2;
            if self.settle(pf, n0, pp) { terms.push(((-pv / pp) as f32, pf)); }
        }
        let vv = dots[i];
        lincomb_in(b, k, g, v, &terms);
        self.pending = Some((v, vv));
    }

    /// The pending column's field (it is part of S whether or not it is later kept).
    fn pending_field(&self) -> Option<&'f Field> { self.pending.map(|(f, _)| f) }

    /// Settles the pending column given its measured norm² `n2`.
    fn finish(&mut self, n2: f64) {
        if let Some((pf, n0)) = self.pending.take() { self.settle(pf, n0, n2); }
    }
}

/// Lower-triangle pair list of ⟨s_i, t_j⟩, j ≤ i, in the order the Gram assembly expects.
fn lower_pairs<'f>(s: &[&'f Field], t: &[&'f Field]) -> Vec<(&'f Field, &'f Field)> {
    (0..s.len()).flat_map(|i| (0..=i).map(move |j| (s[i], t[j]))).collect()
}

/// Symmetric m×m matrix from a lower-triangle dot list (`lower_pairs` order).
fn symmetric_from_lower(m: usize, dots: &[f64]) -> DMatrix<f64> {
    let mut a = DMatrix::<f64>::zeros(m, m);
    let mut idx = 0;
    for i in 0..m { for j in 0..=i { a[(i, j)] = dots[idx]; a[(j, i)] = dots[idx]; idx += 1; } }
    a
}

/// Lowest `nk` Ritz pairs of A c = θ B c (module doc): returns θ ascending and C (m × nk,
/// column i = coefficients of x_i over the columns of S), B-orthonormal.
fn rayleigh_ritz(a: &DMatrix<f64>, b: &DMatrix<f64>, nk: usize) -> (Vec<f64>, DMatrix<f64>) {
    let m = a.nrows();
    let d: Vec<f64> = (0..m).map(|i| { assert!(b[(i, i)] > 0.0, "lobpcg: a column of S has zero norm"); 1.0 / b[(i, i)].sqrt() }).collect();
    let scaled = |mat: &DMatrix<f64>| DMatrix::from_fn(m, m, |i, j| mat[(i, j)] * d[i] * d[j]);
    let (an, bn) = (scaled(a), scaled(b));
    let eb = SymmetricEigen::new(bn);
    let smax = eb.eigenvalues.iter().copied().fold(0.0f64, f64::max);
    let keep: Vec<usize> = (0..m).filter(|&i| eb.eigenvalues[i] > RR_DROP * smax).collect();
    assert!(keep.len() >= nk, "lobpcg: the subspace lost rank ({} of {m} directions) — the Rayleigh–Ritz step is broken", keep.len());
    let t = DMatrix::from_fn(m, keep.len(), |i, c| eb.eigenvectors[(i, keep[c])] / eb.eigenvalues[keep[c]].sqrt());
    let ah = t.transpose() * &an * &t;
    let ah = DMatrix::from_fn(keep.len(), keep.len(), |i, j| 0.5 * (ah[(i, j)] + ah[(j, i)]));
    let ea = SymmetricEigen::new(ah);
    let mut order: Vec<usize> = (0..keep.len()).collect();
    order.sort_by(|&i, &j| ea.eigenvalues[i].total_cmp(&ea.eigenvalues[j]));
    let theta: Vec<f64> = order.iter().take(nk).map(|&i| ea.eigenvalues[i]).collect();
    let ty = &t * DMatrix::from_fn(keep.len(), nk, |i, c| ea.eigenvectors[(i, order[c])]);
    let c = DMatrix::from_fn(m, nk, |i, col| d[i] * ty[(i, col)]);
    (theta, c)
}

/// Lowest `x.len()` eigenpairs of `h` in the L² inner product. `x` holds the initial guess (any
/// vectors; dependent ones are re-seeded) and the result, orthonormal; `evals[i]` receives the
/// Rayleigh quotient ⟨xᵢ, H xᵢ⟩/⟨xᵢ, xᵢ⟩ of the returned xᵢ, ascending. Preconditioner:
/// `JACOBI_SWEEPS` damped-Jacobi sweeps on (H − λᵢ) w = rᵢ with the per-vector λᵢ. Stops when
/// max_i ‖H xᵢ − λᵢ xᵢ‖/‖xᵢ‖ < `tol`, on the f32 floor, or after `max_iter` steps (`Stop`).
/// `work` must come from `LobpcgWork::new` on the same grid with k ≥ `x.len()`.
#[allow(clippy::too_many_arguments)]
pub fn lobpcg(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, r: &Reducer, h: &GpuHamiltonian, x: &mut [Field], evals: &mut [f64],
              work: &mut LobpcgWork, tol: f64, max_iter: usize) -> EigenStats {
    let nk = x.len();
    assert!(nk > 0, "lobpcg: no eigenpairs requested");
    assert!(nk <= work.k, "lobpcg: work was built for k = {}, called with {nk} vectors", work.k);
    assert!(*g == work.grid, "lobpcg: work was built for grid {:?}, called with {:?}", work.grid, g);
    assert_eq!(evals.len(), nk, "lobpcg: evals must have one slot per eigenvector");
    check(g, &[h.v]);
    for f in x.iter() { check(g, &[f]); }
    let LobpcgWork { hx, r: rr, w, p, hs, xn, pn, d, hw, .. } = work;
    let omega = jacobi_omega();
    let diag_t = -1.5 * COEFF[0] / (g.h * g.h);
    let floor = residual_floor_estimate(g);
    let mut b = Batch::new(ctx, "lobpcg");
    let mut readbacks = 0usize;

    // --- Guess: MGS in place, re-seeding dependent vectors; then normalise ---
    let mut xnorm2 = vec![0.0f64; nk];
    {
        let mut mgs = Mgs::new(Vec::new());
        let mut todo: Vec<usize> = (0..nk).collect();
        for round in 0..=MAX_RESEED_ROUNDS {
            assert!(round < MAX_RESEED_ROUNDS, "lobpcg: could not generate {nk} independent guess vectors");
            if round > 0 { for &i in &todo { reseed(ctx, g, &x[i], i, round); } }
            for &i in &todo {
                let pairs = mgs.pairs(&x[i]);
                let dots = r.dots_in(&mut b, k, g, &pairs);
                readbacks += 1;
                mgs.take(&mut b, k, g, &x[i], &dots);
            }
            if let Some(pf) = mgs.pending_field() {
                let n2 = r.dots_in(&mut b, k, g, &[(pf, pf)])[0];
                readbacks += 1;
                mgs.finish(n2);
            }
            todo = (0..nk).filter(|&i| !mgs.kept.iter().any(|q| q.f.buffer == x[i].buffer)).collect();
            if todo.is_empty() { break; }
        }
        for q in &mgs.kept {
            let i = x.iter().position(|f| f.buffer == q.f.buffer).expect("kept column is a guess vector");
            xnorm2[i] = q.norm2;
        }
    }
    for i in 0..nk { scale_in(&mut b, k, g, &x[i], (1.0 / xnorm2[i].sqrt()) as f32); }

    // --- First λ: HX and ⟨x,Hx⟩/⟨x,x⟩ (later iterations use the Ritz values) ---
    for i in 0..nk { h.apply_in(&mut b, k, g, &x[i], &hx[i]); }
    let pairs: Vec<(&Field, &Field)> = (0..nk).flat_map(|i| [(&x[i], &hx[i]), (&x[i], &x[i])]).collect();
    let dots = r.dots_in(&mut b, k, g, &pairs);
    readbacks += 1;
    let mut lambda: Vec<f64> = (0..nk).map(|i| dots[2 * i] / dots[2 * i + 1]).collect();
    let mut np = 0usize; // valid P columns
    let mut iterations = 0usize;
    let (mut best, mut best_it) = (f64::INFINITY, 0usize);
    let stats;

    loop {
        // 1. residuals, speculative W = T R, first MGS dots; one readback
        for i in 0..nk { lincomb_in(&mut b, k, g, &rr[i], &[(1.0, &hx[i]), (-lambda[i] as f32, &x[i])]); }
        for i in 0..nk {
            inv_diag_in(&mut b, k, g, d, h.v, diag_t, lambda[i]);
            mul_in(&mut b, k, g, &w[i], d, &rr[i]);
            for _ in 1..JACOBI_SWEEPS {
                h.apply_in(&mut b, k, g, &w[i], hw);
                jacobi_sweep_in(&mut b, k, g, &w[i], d, &rr[i], hw, omega, lambda[i]);
            }
        }
        let mut pairs: Vec<(&Field, &Field)> = (0..nk).flat_map(|i| [(&x[i], &hx[i]), (&x[i], &x[i]), (&rr[i], &rr[i])]).collect();
        let mgs_start = pairs.len();
        let mut mgs = Mgs::new(Vec::new());
        pairs.extend((0..nk).map(|j| (&x[j], &w[0])));
        pairs.push((&w[0], &w[0]));
        let dots = r.dots_in(&mut b, k, g, &pairs);
        readbacks += 1;
        let mut max_res = 0.0f64;
        for i in 0..nk {
            let (xhx, xx, res2) = (dots[3 * i], dots[3 * i + 1], dots[3 * i + 2]);
            evals[i] = xhx / xx;
            max_res = max_res.max((res2 / xx).sqrt());
            mgs.kept.push(Col { f: &x[i], norm2: xx });
        }
        if max_res < 0.5 * best { best = max_res; best_it = iterations; }
        if let Some(stop) = classify(max_res, tol, iterations, max_iter, best_it, floor) {
            stats = EigenStats { iterations, max_residual: max_res, stop, readbacks };
            break;
        }
        iterations += 1;

        // 2. MGS of W and P against X and each other (one readback per vector; w_0's is above)
        mgs.take(&mut b, k, g, &w[0], &dots[mgs_start..]);
        let rest: Vec<&Field> = w[1..nk].iter().chain(p[..np].iter()).collect();
        for v in rest {
            let pairs = mgs.pairs(v);
            let dots = r.dots_in(&mut b, k, g, &pairs);
            readbacks += 1;
            mgs.take(&mut b, k, g, v, &dots);
        }

        // 3. S = [X, kept W/P, pending]; H S for the non-X columns
        let s: Vec<&Field> = mgs.kept.iter().map(|q| q.f).chain(mgs.pending_field()).collect();
        let m = s.len();
        let mut hs_all: Vec<&Field> = hx[..nk].iter().collect();
        for (j, col) in s[nk..].iter().enumerate() {
            h.apply_in(&mut b, k, g, col, &hs[j]);
            hs_all.push(&hs[j]);
        }

        // 4. A = SᵀHS, B = SᵀS (lower triangles) in one readback; settle the pending column
        let mut pairs = lower_pairs(&s, &hs_all);
        pairs.extend(lower_pairs(&s, &s));
        let dots = r.dots_in(&mut b, k, g, &pairs);
        readbacks += 1;
        let na = m * (m + 1) / 2;
        let (mut a_mat, mut b_mat) = (symmetric_from_lower(m, &dots[..na]), symmetric_from_lower(m, &dots[na..]));
        let had_pending = mgs.pending.is_some();
        mgs.finish(b_mat[(m - 1, m - 1)]);
        let m = if had_pending && mgs.kept.len() < m { // the last column was dropped
            a_mat = a_mat.remove_row(m - 1).remove_column(m - 1);
            b_mat = b_mat.remove_row(m - 1).remove_column(m - 1);
            m - 1
        } else { m };

        // 5. Rayleigh–Ritz (f64); X_new = S C_k, P_new = S_{W,P} C_k
        let (theta, c) = rayleigh_ritz(&a_mat, &b_mat, nk);
        for i in 0..nk {
            let terms: Vec<(f32, &Field)> = (0..m).map(|j| (c[(j, i)] as f32, s[j])).collect();
            lincomb_in(&mut b, k, g, &xn[i], &terms);
            if m > nk { lincomb_in(&mut b, k, g, &pn[i], &terms[nk..]); }
        }
        np = if m > nk { nk } else { 0 };
        lambda = theta;
        drop(s);
        drop(mgs);
        for i in 0..nk {
            std::mem::swap(&mut x[i], &mut xn[i]);
            std::mem::swap(&mut p[i], &mut pn[i]);
        }

        // 6. HX for the new X (true residuals next iteration)
        for i in 0..nk { h.apply_in(&mut b, k, g, &x[i], &hx[i]); }
    }
    drop(b); // empty after the last readback
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};
    use crate::kernels::Kernels;
    use crate::reduce::Reducer;
    use gpu::GpuContext;

    /// 3D isotropic harmonic oscillator: H = −½∇² + ½r², E = n + 3/2 ⇒ 1.5, 2.5 (×3), 3.5 (×6).
    fn oscillator(n: usize, h: f64) -> (qm::Grid, Vec<f64>) {
        let g = qm::Grid::cube(n, h);
        let v = g.sample(|[x, y, z]| 0.5 * (x * x + y * y + z * z));
        (g, v)
    }

    /// The CPU test's deterministic pseudo-random guess (same seed ⇒ same vectors).
    fn random_guess(g: &qm::Grid, k: usize) -> Vec<Vec<f64>> {
        let mut seed = 12345u64;
        (0..k).map(|_| g.sample(|[x, y, z]| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            (u - 0.5) * (-(x * x + y * y + z * z) / 8.0).exp()
        })).collect()
    }

    struct Rig { ctx: GpuContext, k: Kernels, g: GpuGrid, r: Reducer, v: Field }

    fn rig(cg: &qm::Grid, v: &[f64]) -> Option<Rig> {
        let ctx = GpuContext::new_headless()?;
        let g = GpuGrid::new(cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let fv = Field::new(&ctx, &g, "v");
        fv.upload_interior(&ctx, &g, v);
        Some(Rig { ctx, k, g, r, v: fv })
    }

    fn upload(rig: &Rig, data: &[Vec<f64>]) -> Vec<Field> {
        data.iter().map(|d| { let f = Field::new(&rig.ctx, &rig.g, "x"); f.upload_interior(&rig.ctx, &rig.g, d); f }).collect()
    }

    /// CPU reference eigenvalues on the same grid (f64, tol 1e-8, the CPU's Jacobi smoother).
    fn cpu_evals(cg: &qm::Grid, v: &[f64], k: usize) -> Vec<f64> {
        let op = qm::eigen::LocalHamiltonian { grid: cg, v };
        let mut x = random_guess(cg, k);
        let mut e = vec![0.0; k];
        let pc = qm::eigen::jacobi_precond(cg, v, 1.5);
        let stats = qm::eigen::lobpcg(cg, &op, &mut x, &mut e, Some(&pc), 1e-8, 500);
        assert!(stats.max_residual < 1e-8, "cpu reference did not converge: {stats:?}");
        e
    }

    /// Gram matrix of the returned vectors (f64 finish), row-major k×k.
    fn gram(rig: &Rig, x: &[Field]) -> Vec<f64> {
        let xs: Vec<&Field> = x.iter().collect();
        rig.r.gram(&rig.ctx, &rig.k, &rig.g, &xs, &xs)
    }

    #[test]
    fn oscillator_spectrum_matches_cpu() {
        let (cg, v) = oscillator(33, 0.25);
        let k = 4;
        let Some(rig) = rig(&cg, &v) else { return };
        let e_cpu = cpu_evals(&cg, &v, k);
        let mut x = upload(&rig, &random_guess(&cg, k));
        let mut e = vec![0.0; k];
        let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
        let h = GpuHamiltonian { v: &rig.v };
        let t = std::time::Instant::now();
        let stats = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, 1e-6, 500);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let worst = e.iter().zip(&e_cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let floor = residual_floor_estimate(&rig.g);
        println!("33³ h=0.25 k={k}: gpu {stats:?} in {ms:.1} ms; evals gpu {e:?} cpu {e_cpu:?}; worst |Δ| {worst:.2e}; readbacks/iteration {:.1}; floor model {floor:.2e}",
                 stats.readbacks as f64 / stats.iterations.max(1) as f64);
        assert!(matches!(stats.stop, Stop::Converged | Stop::Floor), "{stats:?}");
        assert!(worst < 1e-6, "eigenvalues differ from the CPU by {worst:.3e} Ha: gpu {e:?} cpu {e_cpu:?}");
        // The brief's `residual < 1e-6` is below the f32 floor at h = 0.25 (module doc: model
        // 1.8e-6, measured 1.9e-6 for the ground state and ~2× that for the three degenerate
        // 2.5 Ha states); the solve stops on the floor and the max residual is pinned to ≤ 2.5×
        // the model (measured 2.07×).
        assert!(stats.max_residual <= 2.5 * floor, "residual {:.3e} vs floor model {floor:.3e}", stats.max_residual);
        assert!((e[0] - 1.5).abs() < 1e-4 && e[1..].iter().all(|v| (v - 2.5).abs() < 1e-4), "{e:?}");
    }

    #[test]
    fn eigenvectors_orthonormal_in_f32() {
        let (cg, v) = oscillator(33, 0.25);
        let k = 3;
        let Some(rig) = rig(&cg, &v) else { return };
        let mut x = upload(&rig, &random_guess(&cg, k));
        let mut e = vec![0.0; k];
        let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
        let h = GpuHamiltonian { v: &rig.v };
        let tol = 2.0 * residual_floor_estimate(&rig.g);
        let stats = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, tol, 500);
        let m = gram(&rig, &x);
        let worst = (0..k * k).map(|i| (m[i] - if i / k == i % k { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
        println!("33³ k={k} tol {tol:.2e}: {stats:?}; |XᵀX − I| max {worst:.2e}");
        assert_eq!(stats.stop, Stop::Converged, "{stats:?}");
        assert!(worst < 1e-5, "XᵀX − I = {worst:.3e}: {m:?}"); // achieved 4e-8
        for f in &x { assert_ghosts_zero(&rig.g, &f.read_padded_raw(&rig.ctx)); }
    }

    /// Two identical guess vectors + one random: the relative threshold drops the duplicate,
    /// it is re-seeded with noise, and the solver still returns k converged vectors — the
    /// lowest k, not whatever the noise happened to be near.
    #[test]
    fn degenerate_guess_does_not_break() {
        let (cg, v) = oscillator(25, 0.3);
        let k = 3;
        let Some(rig) = rig(&cg, &v) else { return };
        let e_cpu = cpu_evals(&cg, &v, k);
        let mut guess = random_guess(&cg, 2);
        guess.insert(1, guess[0].clone()); // [g0, g0, g1]
        let mut x = upload(&rig, &guess);
        let mut e = vec![0.0; k];
        let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
        let h = GpuHamiltonian { v: &rig.v };
        let floor = residual_floor_estimate(&rig.g);
        let tol = 2.0 * floor;
        let stats = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, tol, 500);
        let m = gram(&rig, &x);
        let worst_orth = (0..k * k).map(|i| (m[i] - if i / k == i % k { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
        let worst = e.iter().zip(&e_cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("25³ degenerate guess, tol {tol:.2e}: {stats:?}; evals {e:?} cpu {e_cpu:?} (worst |Δ| {worst:.2e}); |XᵀX − I| {worst_orth:.2e}; floor model {floor:.2e}");
        assert_eq!(stats.stop, Stop::Converged, "{stats:?}");
        assert!(stats.max_residual <= tol, "residual {:.3e} vs tol {tol:.3e}", stats.max_residual);
        assert!(worst_orth < 1e-5, "{m:?}");
        assert!(worst < 1e-6, "eigenvalues {e:?} vs cpu {e_cpu:?}");
    }

    #[test]
    fn warm_start_converges_in_one_iteration() {
        let (cg, v) = oscillator(33, 0.25);
        let k = 2;
        let Some(rig) = rig(&cg, &v) else { return };
        let mut x = upload(&rig, &random_guess(&cg, k));
        let mut e = vec![0.0; k];
        let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
        let h = GpuHamiltonian { v: &rig.v };
        // Like the CPU test (1e-8 then 1e-6), with the f32 floor in place of 1e-8: the first
        // solve runs to the floor, the second asks for a tolerance above it (2× the model).
        let tol2 = 2.0 * residual_floor_estimate(&rig.g);
        let first = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, 1e-10, 500);
        let e_first = e.clone();
        let again = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, tol2, 500);
        println!("33³ warm start: first {first:?}, again (tol {tol2:.2e}) {again:?}; evals {e_first:?} / {e:?}");
        assert_eq!(first.stop, Stop::Floor, "{first:?}");
        assert!(again.iterations <= 1, "{again:?}");
        assert_eq!(again.stop, Stop::Converged, "{again:?}");
        assert!(again.max_residual <= tol2, "{again:?}");
        // 0 iterations: only the guess normalisation (f32 scale) and a fresh λ readback happened;
        // the f32 dots put ~3e-9 relative noise on a Rayleigh quotient (measured 7e-9 at 2.5 Ha).
        for (a, b) in e_first.iter().zip(&e) { assert!((a - b).abs() < 1e-7, "warm start moved an eigenvalue: {a} → {b}"); }
    }

    /// Asking for tol = 1e-10 must stop on the f32 floor and report it, not spin to max_iter.
    /// The floor is pinned against the module's model (white rounding noise through the kinetic
    /// symbol): measured 1.96e-6 vs 1.8e-6 at 33³, h = 0.25. The residual reported is the true
    /// one (checked on the CPU in f64 from the returned f32 vectors).
    #[test]
    fn f32_residual_floor_is_documented() {
        let (cg, v) = oscillator(33, 0.25);
        let k = 2;
        let Some(rig) = rig(&cg, &v) else { return };
        let mut x = upload(&rig, &random_guess(&cg, k));
        let mut e = vec![0.0; k];
        let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
        let h = GpuHamiltonian { v: &rig.v };
        let max_iter = 500;
        let stats = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, 1e-10, max_iter);
        let floor = residual_floor_estimate(&rig.g);
        // f64 recomputation of the residual of the returned (f32) vectors with the f64 CPU operator
        let op = qm::eigen::LocalHamiltonian { grid: &cg, v: &v };
        let mut true_res = 0.0f64;
        for (i, f) in x.iter().enumerate() {
            let xi = f.read_interior(&rig.ctx, &rig.g);
            let mut hx = vec![0.0; cg.len()];
            qm::eigen::Operator::apply(&op, &xi, &mut hx);
            let lam = cg.dot(&xi, &hx) / cg.dot(&xi, &xi);
            let r: Vec<f64> = hx.iter().zip(&xi).map(|(a, b)| a - lam * b).collect();
            true_res = true_res.max(cg.dot(&r, &r).sqrt() / cg.dot(&xi, &xi).sqrt());
            // The reported λ is the Rayleigh quotient with the f32 operator; its f32-rounded stencil
            // coefficients bias it by ~2.5e-8 per Ha of kinetic energy (measured 3.8e-8 at λ = 1.5).
            assert!((lam - e[i]).abs() < 1e-7, "reported eval {} vs f64 Rayleigh quotient {lam}", e[i]);
        }
        println!("33³ tol 1e-10: {stats:?}; floor model {floor:.2e}; f64-recomputed true residual {true_res:.2e}; evals {e:?}");
        assert_eq!(stats.stop, Stop::Floor, "{stats:?}");
        assert!(stats.iterations < max_iter, "looped to max_iter");
        assert!(stats.max_residual > 0.5 * floor && stats.max_residual < 1.5 * floor, "floor {:.3e} vs model {floor:.3e}", stats.max_residual);
        assert!((stats.max_residual - true_res).abs() < 0.1 * true_res, "reported {:.3e} vs f64 recomputation {true_res:.3e}", stats.max_residual);
    }

    /// The stop decision (module doc) on measured points: the early phase (residual ~30 Ha for
    /// 8+ iterations while the Ritz values descend) must not stop; a stall is a floor only
    /// inside the band; otherwise max_iter decides.
    #[test]
    fn stop_classification() {
        let floor = 1.8e-6;
        assert_eq!(classify(5e-7, 1e-6, 10, 500, 2, floor), Some(Stop::Converged));
        // 33³ k = 4 iteration 8: max residual 13 Ha, running minimum last halved at 0 → continue
        assert_eq!(classify(13.0, 1e-6, 8, 500, 0, floor), None);
        assert_eq!(classify(1.0, 1e-6, 40, 500, 13, floor), None); // above the band: never a floor
        // the slow 2.5 Ha tail: 5.5e-6 at 63 → 3.9e-6 at 72 (not halved in 9) → continue
        assert_eq!(classify(3.9e-6, 1e-6, 72, 500, 63, floor), None);
        // at the floor: 1.9e-6, no halving for 12 iterations → Floor
        assert_eq!(classify(1.9e-6, 1e-10, 69, 500, 57, floor), Some(Stop::Floor));
        assert_eq!(classify(1.9e-6, 1e-10, 68, 500, 57, floor), None); // 11 iterations: not yet
        assert_eq!(classify(1.0, 1e-6, 500, 500, 490, floor), Some(Stop::MaxIter));
    }

    /// Timing for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture bench_lobpcg`.
    /// Cold solves of the oscillator from the random guess, tol 1e-6 (= the floor at these h,
    /// so each solve ends `Floor` and runs `STALL_WINDOW` extra iterations).
    #[test]
    #[ignore]
    fn bench_lobpcg_49_97() {
        use std::time::Instant;
        let Some(ctx) = GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for (n, h, nk) in [(49, 0.25, 2), (97, 0.2, 5)] {
            let (cg, v) = oscillator(n, h);
            let g = GpuGrid::new(&cg);
            let r = Reducer::new(&ctx, &k, &g);
            let fv = Field::new(&ctx, &g, "v");
            fv.upload_interior(&ctx, &g, &v);
            let ham = GpuHamiltonian { v: &fv };
            let mut work = LobpcgWork::new(&ctx, &g, nk);
            let guess = random_guess(&cg, nk);
            let mut e = vec![0.0; nk];
            for tol in [1e-6, 1e-4] {
                let mut x: Vec<Field> = guess.iter().map(|d| { let f = Field::new(&ctx, &g, "x"); f.upload_interior(&ctx, &g, d); f }).collect();
                let t = Instant::now();
                let s = lobpcg(&ctx, &k, &g, &r, &ham, &mut x, &mut e, &mut work, tol, 500);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let warm = lobpcg(&ctx, &k, &g, &r, &ham, &mut x, &mut e, &mut work, tol, 500);
                let warm_ms = t.elapsed().as_secs_f64() * 1e3;
                println!("{n}³ h={h} k={nk} tol {tol:.0e}: {} it ({:?}) in {ms:.1} ms ({:.2} ms/it, {:.1} readbacks/it), residual {:.2e}, floor model {:.2e}; evals {e:?}; warm again: {} it ({:?}) {warm_ms:.1} ms",
                         s.iterations, s.stop, ms / s.iterations.max(1) as f64, s.readbacks as f64 / s.iterations.max(1) as f64, s.max_residual, residual_floor_estimate(&g),
                         warm.iterations, warm.stop);
            }
        }
    }
}
