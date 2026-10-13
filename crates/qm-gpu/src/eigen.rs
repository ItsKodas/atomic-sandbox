//! LOBPCG on the GPU — the port of `qm::eigen::lobpcg` (read its module doc first: the
//! Rayleigh–Ritz derivation, the orthogonalised-P variant, the multi-sweep damped-Jacobi
//! preconditioner with ω = 4/(3ρ) from the stencil symbol, its SPD condition and the 0.1
//! diagonal floor). The algorithm — S = [X, W, P], A = SᵀHS, X_new = S·C_k, P_new = S_{W,P}·C_k —
//! is unchanged; this doc covers what the GPU (f32 storage, readbacks as sync points) changes.
//!
//! # What stays f64
//! The ≤ 3k×3k Rayleigh–Ritz (nalgebra), λ, the residual norms and every reduction finish
//! (`Reducer`: f32 workgroup partials → Neumaier f64). A coefficient is cast to f32 only when a
//! `lincomb` consumes it. H x is one fused dispatch (`stencil::hamiltonian`: −½∇²x + V·x).
//!
//! # One readback per iteration (M3 Task 10)
//! Task 5's version orthonormalised S by modified Gram–Schmidt on the grid, one readback per
//! vector (2k + 1 per iteration), and built the Gram matrices from one `dot_partials` dispatch
//! per pair (m(m+1) of them). Task 8 measured the k = 5 iteration at ≈ 9 ms grid-independent +
//! 5 ns/point: latency, not bandwidth. Now every iteration is one sync point:
//! 1. R = HX − θX (θ the Ritz values of the previous step; λ on the first), W = T R
//!    (`precond_init` + `JACOBI_SWEEPS` − 1 fused `jacobi_step`s ping-ponging through one
//!    scratch field), HW = H W, HP = H P.
//! 2. **The readback**: every pair of B = SᵀS and A = SᵀHS over the *raw* S = [X, W, P], plus
//!    ‖r_i‖², through `Reducer::block_dots_in` — a handful of fused `gram` dispatches, each
//!    computing all pairs among up to 15 bound fields from one load per field per point.
//!    λ_i = θ_i + ⟨x_i, r_i⟩/⟨x_i, x_i⟩ (= ⟨x_i, Hx_i⟩/⟨x_i, x_i⟩, the reported eigenvalue),
//!    residual_i = ‖r_i‖/‖x_i‖, and the stop test — so on the exit iteration W, HW, HP and
//!    the Gram were computed for nothing (≈ 30 % of one iteration, once per call).
//! 3. Rayleigh–Ritz on the measured A and B (below), X_new = S C_k and
//!    P_new = X_new − X (XᵀX)⁻¹ XᵀX_new by `lincomb`, HX_new = H X_new (recomputed from the
//!    stored vectors, so the next residuals are true residuals). P_new is the orthogonalised-P
//!    variant's step direction with the projection coefficients taken from the measured Gram;
//!    on a raw S the W/P part of C alone (Task 5's P_new = S_{W,P} C_k, correct for an
//!    orthonormal S) is *not* ⊥ X, because W is not — it carries an O(1) component along X
//!    every step, and as its ⊥ part is absorbed into X each iteration P collapses into
//!    span(X) within ~20 iterations (measured on the 25³ k = 3 degenerate case: the three P
//!    columns became identical to f32 rounding and ≈ 0.92·x₂; the method degraded to
//!    preconditioned steepest descent, 97 iterations and a floor exit at 3.9× the model).
//!
//! Dispatches per iteration: k residual lincombs, 4k preconditioner, k HW, k HP, the Gram
//! (1 for k ≤ 2, ≈ 5 for k = 5), 2k Ritz lincombs (3k + 2k at k = 5 through the 7-term chunks),
//! k HX: ≈ 11 for k = 1, ≈ 70 for k = 5 (Task 5: ≈ 45 and ≈ 500). `EigenStats` reports both
//! counts; `bench_lobpcg_49_97` prints them per iteration.
//!
//! # Rayleigh–Ritz on the raw subspace, and the conditioning guard
//! Columns are unnormalised and not orthogonal, so the projected problem is the generalised
//! one, A c = θ B c with A and B *measured* (lower triangles, symmetrised). Column-normalise
//! with D = diag(B)^{-1/2}: B̃ = D B D = U Σ Uᵀ, T = U Σ^{-1/2} over the directions with
//! σ > `RR_DROP`·σ_max, Â = TᵀÃT symmetric-eigen-decomposed, C = D T Y. The Ritz vectors are
//! B-orthonormal in f64 exactly, so X_new is orthonormal up to the f32 rounding of the final
//! `lincomb` whatever S's conditioning — the solver never *assumes* SᵀS = I. `RR_DROP` = 1e-6
//! is tied to the Gram noise: the dots have ~1e-8 relative accuracy (`reduce`), so a direction
//! of B̃ with σ ≲ 1e-6 is known to ~1 % and its 1/√σ ≥ 1e3 amplification would turn that into
//! O(1) error — it is dropped as noise (Task 5's 1e-10, set for an MGS-conditioned S, could
//! never fire).
//!
//! What MGS was really buying: coefficients of size O(1). x_i = S c_i is formed in f32, and
//! its rounding noise is ∝ ‖T y_i‖₂ = (Σ_d y_di²/σ_d)^{1/2} — exactly 1 for an orthonormal S,
//! 1/√σ_d if the Ritz vector leans on a nearly dependent direction d. That noise is white on
//! the grid and sets the residual floor (below), so the Rayleigh–Ritz step returns this
//! amplification α = max_i ‖T y_i‖₂ and the iteration accepts it only when
//! α ≤ max(`ALPHA_MAX`, `ALPHA_RESIDUAL_FRACTION` × max_residual/floor): near convergence at
//! most 2× an orthonormal basis's noise; far from it whatever keeps the injected noise
//! (α × floor, fresh each step — it is not cumulative) below a tenth of the residual.
//! Otherwise — the **fallback** — the W and P columns are orthogonalised against X and each
//! other on the grid by Task 5's MGS (one readback per vector after the first, whose dots are
//! already in B), H is re-applied to them, the Gram is re-measured (one more readback) and the
//! Rayleigh–Ritz step repeated without the guard (MGS at `MGS_DROP` leaves κ(S) ≲ 1.3, so
//! α ≈ 1). `EigenStats::fallbacks` counts the iterations that needed it and `alpha_worst`
//! the largest α seen. Measured (33³ oscillator k = 2 from the random guess, traced with
//! `LOBPCG_TRACE=1`): σ_min(B̃) is 0.1–0.8 and α = 1.00 from iteration 8 (residual 13 Ha)
//! to the floor exit at 77 — the raw [X, W, P] is well conditioned in the convergent phase,
//! because W ⊥ X to the extent the smoother is a scalar on smooth modes and because near the
//! floor the residual is white noise, whose preconditioned image is as independent of X and P
//! as noise gets. Only the random-guess transient (iterations 2–7, residual 30–60 Ha) dips to
//! σ_min 1e-3–1e-5 with α up to 12, where the relaxed bound applies; no measured case
//! (`bench_lobpcg_49_97`, the SCF of H₂/H₂O) needs the fallback.
//!
//! The X columns of A are not measured as ⟨s, Hx⟩ directly: that dot's summands are ~|λ||x||s|
//! while its value (for s ⊥ x) is ~‖r‖‖s‖, so its f32 noise, ~1e-8·|λ|·‖s‖, would perturb the
//! Ritz coefficient on the non-smooth W/P directions and ‖H‖ would amplify the admixture into
//! the residual — measured as a floor growing from 1.9e-6 to 2.8e-6 when the oscillator is
//! shifted by −20 Ha (`floor_independent_of_eigenvalue_shift`). Instead
//! ⟨s_i, H x_j⟩ = ⟨s_i, r_j⟩ + θ_j ⟨s_i, x_j⟩ with the stored residual r_j and the *same*
//! measured B entry, so A − θB is accurate to the noise of ⟨s_i, r_j⟩, whose summands are
//! small. The W/P columns ⟨s, Hw⟩ are Rayleigh-quotient-sized and need nothing.
//!
//! # The guess
//! One readback of XᵀX and ⟨x_i, Hx_i⟩ over the raw guess. A column within `GUESS_DEP`^{1/2} =
//! 1 % of the span of the earlier ones (sequential elimination on the Gram: its Schur
//! complement is below `GUESS_DEP` × its norm²; an exact or f32-near duplicate is far below) is
//! replaced by deterministic noise (LCG, seeded by its slot) under a Gaussian envelope and the
//! Gram re-measured — white noise has nonzero overlap with every eigenvector, so the lowest k
//! are still found (`degenerate_guess_does_not_break` checks the eigenvalues, not only the
//! count). θ_i = ⟨x_i, Hx_i⟩/⟨x_i, x_i⟩ (the residual-minimising shift of x_i), and the guess
//! is **neither rotated nor rescaled**: the generalised Rayleigh–Ritz step handles unnormalised,
//! non-orthogonal columns, and a warm start that already meets tol must return X bit-identical.
//! A first version applied the Rayleigh–Ritz rotation of the guess subspace here; for the SCF
//! that re-mixes the k warm orbitals with O(1) coefficients on every step — fresh f32 rounding
//! of ~√k·2.4e-8 per orbital even on a 0-iteration exit, ~5e-7 in Δρ for H₂O — and H₂O's
//! step count went from 17 to 56 (the density is rotation-invariant; nothing was gained).
//! X is never dropped later: after the first step the Ritz vectors are B-orthonormal by construction.
//!
//! # Relative MGS drop threshold (plan Review Focus 4; the fallback path)
//! The CPU drops a vector whose norm after projection is < 1e-12 — an absolute threshold that
//! an f32 vector of norm ~1 can never trip (its rounding noise alone is ~1e-7). Here a vector
//! is dropped when ‖v − Σ c_q q‖ < `MGS_DROP`·‖v‖₀ = 1e-6 ‖v‖₀, ‖v‖₀ its norm before the
//! projection. Why 1e-6: the projection coefficients come from f32 dot products (measured
//! relative accuracy ~1e-8 in `reduce`), so one MGS pass leaves a spurious component
//! ~1e-8·‖v‖₀ along each kept direction; a survivor at ratio 1e-6 is therefore orthogonal to
//! the kept set to ~1% (its own rounding noise is ~1e-7·‖v‖₀, i.e. 10% of it), which keeps the
//! Gram matrix of S well conditioned (κ ≲ 1.3); below 1e-6 the leftover would be O(1) and S
//! would lose rank. The CPU's 1e-12 is the same construction with ε_f64 (1e-16 × 10⁴). Vectors
//! are never normalised; kept columns carry their measured ⟨q,q⟩ and the projection
//! coefficient is ⟨q,v⟩/⟨q,q⟩; the last column's post-projection norm arrives with the Gram.
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
//! oscillator states at 8%/iteration (the smoother is a scalar on low modes, as on the CPU) —
//! and 12 iterations at 8% is a factor 2.7, so a genuine tail is only mistaken for the floor
//! below 6%/iteration *and* inside the band, where the eigenvalue error is already
//! ≤ (10·floor)²/gap ~ 1e-9 Ha. Callers should key on `max_residual`, not the label (as for
//! `GpuPoisson`), and pass tol ≥ ~2× the estimate when they want `Converged` without the
//! window's extra iterations.
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid};
use crate::kernels::{uniform, workgroups_for, Kernels};
use crate::pointwise::{check, lincomb_in, Params};
use crate::reduce::Reducer;
use crate::stencil::{hamiltonian_in, jacobi_step_in};
use gpu::GpuContext;
use nalgebra::{DMatrix, SymmetricEigen};
pub use qm::eigen::JACOBI_SWEEPS;
use qm::eigen::jacobi_omega;
use qm::stencil::COEFF;

/// H = −½∇² + V for a local potential `v` on the grid (a padded field with zero ghosts).
pub struct GpuHamiltonian<'a> { pub v: &'a Field }

impl GpuHamiltonian<'_> {
    /// out = −½∇²x + V·x on the interior (recorded): one fused dispatch.
    pub fn apply_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, x: &Field, out: &Field) {
        hamiltonian_in(b, k, g, x, self.v, out);
    }
}

/// Relative MGS drop threshold (fallback path): a vector is dropped when its norm after
/// projection is below `MGS_DROP` × its norm before (module doc).
pub const MGS_DROP: f64 = 1e-6;
/// Iterations without the running-minimum max-residual halving, inside the floor band, before
/// the solve stops with `Stop::Floor`.
pub const STALL_WINDOW: usize = 12;
/// The floor band: stalls are diagnosed only at max residual ≤ `FLOOR_BAND` × `residual_floor_estimate`.
pub const FLOOR_BAND: f64 = 10.0;
/// Directions of the column-normalised Gram matrix with eigenvalue below this × the largest are
/// discarded in the Rayleigh–Ritz step: tied to the ~1e-8 relative accuracy of the measured
/// dots (module doc).
pub const RR_DROP: f64 = 1e-6;
/// Largest coefficient amplification α = max_i ‖T y_i‖₂ accepted near convergence from a
/// Rayleigh–Ritz step on the raw subspace (1 for an orthonormal S); above it the W/P columns
/// are orthogonalised on the grid first (the MGS fallback, module doc).
pub const ALPHA_MAX: f64 = 2.0;
/// Far from convergence the guard relaxes to α ≤ this × max_residual/floor: the noise a step
/// injects, α × the floor, only matters once it is comparable to the residual (module doc).
pub const ALPHA_RESIDUAL_FRACTION: f64 = 0.1;
/// A guess column whose Schur complement in XᵀX is below this × its norm² (i.e. within 1 % of
/// the span of the earlier columns) is re-seeded.
const GUESS_DEP: f64 = 1e-4;
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
    /// GPU→CPU readbacks (sync points) in this call, guess included.
    pub readbacks: usize,
    /// Compute dispatches in this call (pointwise, stencil and Gram kernels), guess included.
    pub dispatches: usize,
    /// Iterations that needed the on-grid MGS fallback (module doc).
    pub fallbacks: usize,
    /// Largest coefficient amplification α of a raw-subspace Rayleigh–Ritz step in this call
    /// (before any fallback; 1 for an orthonormal basis).
    pub alpha_worst: f64,
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
/// `guess_tol` applies to the guess only (iteration 0): a warm start within it is accepted
/// without a step (`lobpcg_with_guess_tol`).
fn classify(max_res: f64, tol: f64, it: usize, max_iter: usize, best_it: usize, floor: f64, guess_tol: f64) -> Option<Stop> {
    if max_res < tol || (it == 0 && max_res < guess_tol) { return Some(Stop::Converged); }
    if max_res <= FLOOR_BAND * floor && it - best_it >= STALL_WINDOW { return Some(Stop::Floor); }
    if it >= max_iter { return Some(Stop::MaxIter); }
    None
}

/// Preallocated work fields for `lobpcg` with up to `k` eigenpairs on one grid: 8k + 1 fields
/// (HX, R, W, P, HW, HP, X_new, P_new, one preconditioner scratch).
pub struct LobpcgWork {
    k: usize,
    hx: Vec<Field>, r: Vec<Field>, w: Vec<Field>, p: Vec<Field>, hw: Vec<Field>, hp: Vec<Field>,
    xn: Vec<Field>, pn: Vec<Field>, scratch: Field,
    grid: GpuGrid,
}

impl LobpcgWork {
    pub fn new(ctx: &GpuContext, g: &GpuGrid, k: usize) -> LobpcgWork {
        assert!(k > 0, "LobpcgWork::new: k must be positive");
        let fields = |label: &str, count: usize| (0..count).map(|_| Field::new(ctx, g, label)).collect::<Vec<_>>();
        LobpcgWork {
            k, hx: fields("lobpcg hx", k), r: fields("lobpcg r", k), w: fields("lobpcg w", k), p: fields("lobpcg p", k),
            hw: fields("lobpcg hw", k), hp: fields("lobpcg hp", k), xn: fields("lobpcg x_new", k), pn: fields("lobpcg p_new", k),
            scratch: Field::new(ctx, g, "lobpcg scratch"), grid: *g,
        }
    }
}

/// w₀ = D⁻¹ r, D = max(diag_t + V − λ, 0.1) (recorded).
#[allow(clippy::too_many_arguments)]
fn precond_init_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field, r: &Field, v: &Field, diag_t: f64, lambda: f64) {
    let ctx = b.ctx();
    let u = uniform(ctx, "precond_init", &Params::new(g, diag_t as f32, lambda as f32));
    let bg = k.bind_group(ctx, "precond_init", &[u.as_entire_binding(), out.buffer.as_entire_binding(), r.buffer.as_entire_binding(), v.buffer.as_entire_binding()]);
    b.dispatch(k, "precond_init", &bg, [workgroups_for(g.interior_len()), 1, 1]);
}

/// W = T R for one vector: `JACOBI_SWEEPS` damped-Jacobi sweeps on (H − λ) w = r, the first
/// being w₀ = D⁻¹ r and each later one a fused `jacobi_step`, ping-ponging between `w` and
/// `scratch` so that the result lands in `w` (recorded; `JACOBI_SWEEPS` dispatches).
#[allow(clippy::too_many_arguments)]
fn precondition_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, h: &GpuHamiltonian, w: &Field, scratch: &Field, r: &Field, omega: f64, lambda: f64, diag_t: f64) {
    let steps = JACOBI_SWEEPS - 1;
    let (mut src, mut dst) = if steps % 2 == 1 { (scratch, w) } else { (w, scratch) };
    precond_init_in(b, k, g, src, r, h.v, diag_t, lambda);
    for _ in 0..steps {
        jacobi_step_in(b, k, g, src, h.v, r, dst, omega, lambda, diag_t);
        std::mem::swap(&mut src, &mut dst);
    }
    debug_assert!(src.buffer == w.buffer, "the last sweep must land in w");
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

/// MGS with deferred norms (module doc, the fallback path). `kept` are the orthogonal columns
/// (norms known); `pending` was projected but its post-projection norm is measured with the
/// next readback.
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

/// The one readback of an iteration: every pair of B = SᵀS and A = SᵀT (lower triangles, T
/// the H-images with the X columns replaced by the residuals) and ‖r_i‖² for i < nk.
fn gram_pairs<'f>(s: &[&'f Field], t: &[&'f Field], nk: usize) -> Vec<(&'f Field, &'f Field)> {
    let mut pairs = lower_pairs(s, s);
    pairs.extend(lower_pairs(s, t));
    pairs.extend((0..nk).map(|i| (t[i], t[i])));
    pairs
}

/// Assembles A and B from the `gram_pairs` dots, restoring the X columns of A through the
/// residuals (module doc): A_ij += θ_j B_ij for j < nk. Returns (A, B, ‖r_i‖²).
fn assemble(m: usize, nk: usize, theta: &[f64], dots: &[f64]) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
    let na = m * (m + 1) / 2;
    let b_mat = symmetric_from_lower(m, &dots[..na]);
    let mut a_mat = symmetric_from_lower(m, &dots[na..2 * na]);
    for i in 0..m { for j in 0..=i.min(nk - 1) {
        let v = a_mat[(i, j)] + theta[j] * b_mat[(i, j)];
        a_mat[(i, j)] = v;
        a_mat[(j, i)] = v;
    }}
    assert!(a_mat.iter().chain(b_mat.iter()).all(|v| v.is_finite()), "lobpcg: non-finite Gram entry (NaN in a vector?)");
    (a_mat, b_mat, dots[2 * na..2 * na + nk].to_vec())
}

/// Columns of a Gram matrix within `rel`^{1/2} of the span of the earlier ones: sequential
/// elimination (Cholesky with a pivot test); column i is dependent when its Schur complement
/// d_i ≤ rel·B_ii (a zero or non-positive norm counts as dependent). Pinned by `dependent_columns_detects`.
fn dependent_columns(b: &DMatrix<f64>, rel: f64) -> Vec<usize> {
    let m = b.nrows();
    let mut l = DMatrix::<f64>::zeros(m, m);
    let mut kept: Vec<usize> = Vec::new();
    let mut dependent = Vec::new();
    for i in 0..m {
        let mut d = b[(i, i)];
        for &j in &kept {
            let mut s = b[(i, j)];
            for &q in kept.iter().take_while(|&&q| q < j) { s -= l[(i, q)] * l[(j, q)]; }
            l[(i, j)] = s / l[(j, j)];
            d -= l[(i, j)] * l[(i, j)];
        }
        // NaN compares false on both sides: a non-finite column is dependent too
        let independent = d > rel * b[(i, i)] && b[(i, i)] > 0.0;
        if independent { l[(i, i)] = d.sqrt(); kept.push(i); } else { dependent.push(i); }
    }
    dependent
}

/// Lowest `nk` Ritz pairs of A c = θ B c (module doc): θ ascending, C (m × nk, column i = the
/// coefficients of x_i over the columns of S, B-orthonormal), and the coefficient
/// amplification α = max_i ‖T y_i‖₂ of the step (1 for an orthonormal S).
fn rayleigh_ritz(a: &DMatrix<f64>, b: &DMatrix<f64>, nk: usize) -> (Vec<f64>, DMatrix<f64>, f64) {
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
    let y = DMatrix::from_fn(keep.len(), nk, |i, c| ea.eigenvectors[(i, order[c])]);
    // α_i² = Σ_d y_di² / σ_d over the kept directions (U is orthonormal, so ‖T y‖² = ‖Σ^{-1/2} y‖²)
    let alpha = (0..nk).map(|c| (0..keep.len()).map(|dd| y[(dd, c)] * y[(dd, c)] / eb.eigenvalues[keep[dd]]).sum::<f64>().sqrt()).fold(0.0, f64::max);
    let ty = &t * y;
    let c = DMatrix::from_fn(m, nk, |i, col| d[i] * ty[(i, col)]);
    (theta, c, alpha)
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
    lobpcg_with(ctx, k, g, r, h, x, evals, work, tol, tol, max_iter, ALPHA_MAX, ALPHA_RESIDUAL_FRACTION)
}

/// `lobpcg` that accepts the guess as it is (0 iterations, X untouched) when its residual is
/// already below `guess_tol` ≥ `tol`, and otherwise iterates to `tol`. The SCF's hysteresis
/// band (`qm_gpu::scf`, `EIG_GUESS_FACTOR`): whether a warm orbital set is re-solved or kept
/// frozen is then decided with a margin instead of at the rounding level of a residual that
/// sits just below tol.
#[allow(clippy::too_many_arguments)]
pub fn lobpcg_with_guess_tol(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, r: &Reducer, h: &GpuHamiltonian, x: &mut [Field], evals: &mut [f64],
                             work: &mut LobpcgWork, tol: f64, guess_tol: f64, max_iter: usize) -> EigenStats {
    assert!(guess_tol >= tol, "lobpcg: guess_tol {guess_tol} must not be below tol {tol}");
    lobpcg_with(ctx, k, g, r, h, x, evals, work, tol, guess_tol, max_iter, ALPHA_MAX, ALPHA_RESIDUAL_FRACTION)
}

/// `lobpcg` with the guard's two constants as parameters (tests force the fallback with 0, 0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lobpcg_with(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, r: &Reducer, h: &GpuHamiltonian, x: &mut [Field], evals: &mut [f64],
                          work: &mut LobpcgWork, tol: f64, guess_tol: f64, max_iter: usize, alpha_max: f64, alpha_fraction: f64) -> EigenStats {
    let nk = x.len();
    assert!(nk > 0, "lobpcg: no eigenpairs requested");
    assert!(nk <= work.k, "lobpcg: work was built for k = {}, called with {nk} vectors", work.k);
    assert!(*g == work.grid, "lobpcg: work was built for grid {:?}, called with {:?}", work.grid, g);
    assert_eq!(evals.len(), nk, "lobpcg: evals must have one slot per eigenvector");
    check(g, &[h.v]);
    for f in x.iter() { check(g, &[f]); }
    let LobpcgWork { hx, r: rr, w, p, hw, hp, xn, pn, scratch, .. } = work;
    let omega = jacobi_omega();
    let diag_t = -1.5 * COEFF[0] / (g.h * g.h);
    let floor = residual_floor_estimate(g);
    let mut b = Batch::new(ctx, "lobpcg");
    let (mut readbacks, mut gram_dispatches, mut fallbacks, mut alpha_worst) = (0usize, 0usize, 0usize, 0.0f64);
    let trace = std::env::var_os("LOBPCG_TRACE").is_some();

    // --- Guess: one readback of XᵀX and ⟨x_i, Hx_i⟩, re-seeding dependent vectors (one more
    //     per round). The guess is neither rotated nor rescaled (module doc): a warm start
    //     that already meets tol must leave X bit-identical. ---
    let mut theta: Vec<f64>;
    {
        let mut round = 0;
        loop {
            assert!(round < MAX_RESEED_ROUNDS, "lobpcg: could not generate {nk} independent guess vectors");
            for i in 0..nk { h.apply_in(&mut b, k, g, &x[i], &hx[i]); }
            let xs: Vec<&Field> = x.iter().collect();
            let mut pairs = lower_pairs(&xs, &xs);
            pairs.extend((0..nk).map(|i| (&x[i], &hx[i])));
            let (dots, nd) = r.block_dots_in(&mut b, k, g, &pairs);
            readbacks += 1;
            gram_dispatches += nd;
            let na = nk * (nk + 1) / 2;
            let b_mat = symmetric_from_lower(nk, &dots[..na]);
            let dependent = dependent_columns(&b_mat, GUESS_DEP);
            if dependent.is_empty() {
                theta = (0..nk).map(|i| dots[na + i] / b_mat[(i, i)]).collect();
                break;
            }
            round += 1;
            for &i in &dependent { reseed(ctx, g, &x[i], i, round); }
        }
    }
    let mut lambda = theta.clone();
    let mut np = 0usize; // valid P columns
    let mut iterations = 0usize;
    let (mut best, mut best_it) = (f64::INFINITY, 0usize);
    let stats;

    loop {
        // 1. residuals R = HX − θX, W = T R (speculative), HW, HP
        for i in 0..nk {
            // θ split into an f32 head and tail so the residual carries no coherent (θ − θ_f32)·x
            // component (≤ 2⁻²⁴|θ|‖x‖, ~1e-6 at |θ| ≈ 18 Ha); the same read-only field twice is fine.
            let hi = theta[i] as f32;
            let lo = (theta[i] - hi as f64) as f32;
            lincomb_in(&mut b, k, g, &rr[i], &[(1.0, &hx[i]), (-hi, &x[i]), (-lo, &x[i])]);
        }
        for i in 0..nk { precondition_in(&mut b, k, g, h, &w[i], scratch, &rr[i], omega, lambda[i], diag_t); }
        for i in 0..nk { h.apply_in(&mut b, k, g, &w[i], &hw[i]); }
        for i in 0..np { h.apply_in(&mut b, k, g, &p[i], &hp[i]); }

        // 2. the readback: B = SᵀS, A = SᵀHS (X columns through R), ‖r_i‖²
        let s: Vec<&Field> = x.iter().chain(w[..nk].iter()).chain(p[..np].iter()).collect();
        let t: Vec<&Field> = rr[..nk].iter().chain(hw[..nk].iter()).chain(hp[..np].iter()).collect();
        let m = s.len();
        let (dots, nd) = r.block_dots_in(&mut b, k, g, &gram_pairs(&s, &t, nk));
        readbacks += 1;
        gram_dispatches += nd;
        let (mut a_mat, mut b_mat, res2) = assemble(m, nk, &theta, &dots);
        let mut max_res = 0.0f64;
        for i in 0..nk {
            let xx = b_mat[(i, i)];
            lambda[i] = a_mat[(i, i)] / xx; // = θ_i + ⟨x_i, r_i⟩/⟨x_i, x_i⟩, the Rayleigh quotient of the stored x_i
            evals[i] = lambda[i];
            max_res = max_res.max((res2[i] / xx).sqrt());
        }
        if max_res < 0.5 * best { best = max_res; best_it = iterations; }
        if let Some(stop) = classify(max_res, tol, iterations, max_iter, best_it, floor, guess_tol) {
            stats = EigenStats { iterations, max_residual: max_res, stop, readbacks, dispatches: b.dispatches() + gram_dispatches, fallbacks, alpha_worst };
            break;
        }
        iterations += 1;

        // 3. Rayleigh–Ritz with the conditioning guard; fallback: MGS on the grid, re-measure
        let mut s = s;
        let (th, c) = {
            let (th, c, alpha) = rayleigh_ritz(&a_mat, &b_mat, nk);
            alpha_worst = alpha_worst.max(alpha);
            if trace {
                let bn = DMatrix::from_fn(m, m, |i, j| b_mat[(i, j)] / (b_mat[(i, i)] * b_mat[(j, j)]).sqrt());
                let sig = SymmetricEigen::new(bn.clone()).eigenvalues;
                eprintln!("lobpcg it {iterations}: max_res {max_res:.2e} α {alpha:.2} σ(B̃) min {:.2e} max {:.2e} m {m}", sig.min(), sig.max());
                if sig.min() < 1e-6 && iterations.is_multiple_of(10) { eprintln!("  B̃ = {:.4}", bn); }
            }
            if alpha <= alpha_max.max(alpha_fraction * max_res / floor) || m == nk { (th, c) } else {
                fallbacks += 1;
                let mut mgs = Mgs::new((0..nk).map(|i| Col { f: s[i], norm2: b_mat[(i, i)] }).collect());
                // w_0's projection dots are already in B: ⟨x_j, w_0⟩ and ⟨w_0, w_0⟩
                let d0: Vec<f64> = (0..nk).map(|j| b_mat[(nk, j)]).chain([b_mat[(nk, nk)]]).collect();
                mgs.take(&mut b, k, g, s[nk], &d0);
                for &v in &s[nk + 1..] {
                    let pairs = mgs.pairs(v);
                    let (dv, nd) = r.block_dots_in(&mut b, k, g, &pairs);
                    readbacks += 1;
                    gram_dispatches += nd;
                    mgs.take(&mut b, k, g, v, &dv);
                }
                let s2: Vec<&Field> = mgs.kept.iter().map(|q| q.f).chain(mgs.pending_field()).collect();
                // H of the surviving non-X columns into their own slots (w_i ↔ hw_i, p_i ↔ hp_i)
                let hmap: Vec<(&Field, &Field)> = w.iter().zip(hw.iter()).chain(p.iter().zip(hp.iter())).collect();
                let mut t2: Vec<&Field> = Vec::with_capacity(s2.len());
                for (j, &col) in s2.iter().enumerate() {
                    if j < nk { t2.push(&rr[j]); continue; }
                    let hc = hmap.iter().find(|(c, _)| c.buffer == col.buffer).map(|(_, hc)| *hc).expect("a W/P column");
                    h.apply_in(&mut b, k, g, col, hc);
                    t2.push(hc);
                }
                let m2 = s2.len();
                let (dots, nd) = r.block_dots_in(&mut b, k, g, &gram_pairs(&s2, &t2, nk));
                readbacks += 1;
                gram_dispatches += nd;
                let (a2, b2, _) = assemble(m2, nk, &theta, &dots);
                let had_pending = mgs.pending.is_some();
                mgs.finish(b2[(m2 - 1, m2 - 1)]);
                let (a2, b2, s2) = if had_pending && mgs.kept.len() < m2 {
                    (a2.remove_row(m2 - 1).remove_column(m2 - 1), b2.remove_row(m2 - 1).remove_column(m2 - 1), s2[..m2 - 1].to_vec())
                } else { (a2, b2, s2) };
                a_mat = a2;
                b_mat = b2;
                s = s2;
                let (th, c, _) = rayleigh_ritz(&a_mat, &b_mat, nk);
                (th, c)
            }
        };
        let m = s.len();

        // 4. X_new = S C_k and P_new = X_new − X (XᵀX)⁻¹ XᵀX_new = S C_p with the projection
        //    taken from the measured Gram (module doc: P must be ⊥ X; on a raw S the W/P part
        //    of C alone is not); f64 → f32 coefficients; swap in, no copies
        let c_p = (m > nk).then(|| {
            let bxx = b_mat.view((0, 0), (nk, nk)).into_owned();
            let bxs = b_mat.view((0, 0), (nk, m)).into_owned();
            let proj = bxx.cholesky().expect("lobpcg: XᵀX is SPD (X is orthonormal to f32 rounding)").solve(&(&bxs * &c));
            let mut cp = c.clone();
            for i in 0..nk { for j in 0..nk { cp[(i, j)] -= proj[(i, j)]; } }
            cp
        });
        for i in 0..nk {
            let terms: Vec<(f32, &Field)> = (0..m).map(|j| (c[(j, i)] as f32, s[j])).collect();
            lincomb_in(&mut b, k, g, &xn[i], &terms);
            if let Some(cp) = &c_p {
                let terms: Vec<(f32, &Field)> = (0..m).map(|j| (cp[(j, i)] as f32, s[j])).collect();
                lincomb_in(&mut b, k, g, &pn[i], &terms);
            }
        }
        np = if m > nk { nk } else { 0 };
        theta = th;
        drop(s);
        for i in 0..nk {
            std::mem::swap(&mut x[i], &mut xn[i]);
            std::mem::swap(&mut p[i], &mut pn[i]);
        }

        // 5. HX for the new X (true residuals next iteration)
        for i in 0..nk { h.apply_in(&mut b, k, g, &x[i], &hx[i]); }
    }
    drop(b); // empty after the last readback
    if trace { eprintln!("lobpcg exit: {stats:?} evals {evals:?}"); }
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

    fn per_it(s: &EigenStats) -> String {
        let it = s.iterations.max(1) as f64;
        format!("{:.1} readbacks/it, {:.1} dispatches/it, {} fallbacks, worst α {:.2}", s.readbacks as f64 / it, s.dispatches as f64 / it, s.fallbacks, s.alpha_worst)
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
        println!("33³ h=0.25 k={k}: gpu {stats:?} in {ms:.1} ms ({}); evals gpu {e:?} cpu {e_cpu:?}; worst |Δ| {worst:.2e}; floor model {floor:.2e}", per_it(&stats));
        assert!(matches!(stats.stop, Stop::Converged | Stop::Floor), "{stats:?}");
        assert!(worst < 3e-7, "eigenvalues differ from the CPU by {worst:.3e} Ha: gpu {e:?} cpu {e_cpu:?}"); // plan 1e-6; achieved 6.7e-8
        // The brief's `residual < 1e-6` is below the f32 floor at h = 0.25 (module doc: model
        // 1.8e-6, measured 1.9e-6 for the ground state and ~2× that for the three degenerate
        // 2.5 Ha states); the solve stops on the floor and the max residual is pinned to ≤ 2.5×
        // the model (measured 2.07×).
        assert!(stats.max_residual <= 2.5 * floor, "residual {:.3e} vs floor model {floor:.3e}", stats.max_residual);
        assert!((e[0] - 1.5).abs() < 1e-4 && e[1..].iter().all(|v| (v - 2.5).abs() < 1e-4), "{e:?}");
        assert_eq!(stats.readbacks, stats.iterations + 2, "one readback per iteration, one for the guess, one for the exit test: {stats:?}");
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
        println!("33³ k={k} tol {tol:.2e}: {stats:?} ({}); |XᵀX − I| max {worst:.2e}", per_it(&stats));
        assert_eq!(stats.stop, Stop::Converged, "{stats:?}");
        assert!(worst < 1e-6, "XᵀX − I = {worst:.3e}: {m:?}"); // brief 1e-5; achieved 4.7e-8
        for f in &x { assert_ghosts_zero(&rig.g, &f.read_padded_raw(&rig.ctx)); }
    }

    /// Dependent guess vectors + one random: (a) an exact duplicate (Schur complement exactly
    /// 0 — any threshold drops it) and (b) a NEAR duplicate g₀ + 1e-8·g₁, which in f32 is g₀
    /// with scattered bit flips, Schur complement ratio ~1e-15 against the Gram's ~1e-8 noise:
    /// only a *relative* test on the measured Gram drops it (the CPU's absolute 1e-12 on a
    /// norm-1 vector would keep it and the subspace would be rank-deficient). Both are
    /// re-seeded with noise and the solver still returns the lowest k eigenvalues with
    /// orthonormal vectors.
    #[test]
    fn degenerate_guess_does_not_break() {
        let (cg, v) = oscillator(25, 0.3);
        let k = 3;
        let Some(rig) = rig(&cg, &v) else { return };
        let e_cpu = cpu_evals(&cg, &v, k);
        let base = random_guess(&cg, 2);
        let near: Vec<f64> = base[0].iter().zip(&base[1]).map(|(a, b)| a + 1e-8 * b).collect();
        for (name, dup) in [("exact duplicate", base[0].clone()), ("near duplicate g0 + 1e-8 g1", near)] {
            let guess = vec![base[0].clone(), dup, base[1].clone()];
            let mut x = upload(&rig, &guess);
            // the stored near duplicate really is g0 with bit flips: relative difference ≲ 1e-7
            let (x0, x1) = (x[0].read_interior(&rig.ctx, &rig.g), x[1].read_interior(&rig.ctx, &rig.g));
            let diff: Vec<f64> = x0.iter().zip(&x1).map(|(a, b)| a - b).collect();
            let ratio = cg.dot(&diff, &diff).sqrt() / cg.dot(&x0, &x0).sqrt();
            let mut e = vec![0.0; k];
            let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
            let h = GpuHamiltonian { v: &rig.v };
            let floor = residual_floor_estimate(&rig.g);
            let tol = 2.0 * floor;
            let stats = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, tol, 500);
            let m = gram(&rig, &x);
            let worst_orth = (0..k * k).map(|i| (m[i] - if i / k == i % k { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
            let worst = e.iter().zip(&e_cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            println!("25³ {name} (‖x1 − x0‖/‖x0‖ = {ratio:.2e}), tol {tol:.2e}: {stats:?} ({}); evals {e:?} cpu {e_cpu:?} (worst |Δ| {worst:.2e}); |XᵀX − I| {worst_orth:.2e}; floor model {floor:.2e}", per_it(&stats));
            assert!(ratio < 1e-6, "{name}: the guess pair is not below the drop threshold ({ratio:.2e})");
            assert_eq!(stats.stop, Stop::Converged, "{name}: {stats:?}");
            assert!(stats.max_residual <= tol, "{name}: residual {:.3e} vs tol {tol:.3e}", stats.max_residual);
            assert!(worst_orth < 1e-6, "{name}: {m:?}");
            assert!(worst < 3e-7, "{name}: eigenvalues {e:?} vs cpu {e_cpu:?}");
            assert!(stats.readbacks >= stats.iterations + 3, "a re-seeding round costs one extra readback: {stats:?}");
        }
    }

    /// The guess filter on small Gram matrices: an exact duplicate, an f32-near duplicate
    /// (post-projection ratio 3e-8, i.e. a Schur complement ~1e-15 — below the Gram noise),
    /// a column exactly in the span of two earlier ones, one at 0.5 % of the span and a zero
    /// column are dependent; a column at 5 % of the span and an orthogonal one are kept.
    #[test]
    fn dependent_columns_detects() {
        let gram_of = |cols: &[Vec<f64>]| DMatrix::from_fn(cols.len(), cols.len(), |i, j| cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum());
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];
        let scaled = |v: &[f64], s: f64| -> Vec<f64> { v.iter().map(|x| x * s).collect() };
        let add = |u: &[f64], v: &[f64]| -> Vec<f64> { u.iter().zip(v).map(|(x, y)| x + y).collect() };
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), a.clone(), b.clone()]), GUESS_DEP), vec![1]);
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), add(&a, &scaled(&b, 3e-8)), b.clone()]), GUESS_DEP), vec![1]);
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), b.clone(), add(&a, &scaled(&b, 0.3))]), GUESS_DEP), vec![2]);
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), add(&a, &scaled(&c, 5e-3))]), GUESS_DEP), vec![1]);
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), add(&a, &scaled(&c, 5e-2)), b.clone()]), GUESS_DEP), Vec::<usize>::new());
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), vec![0.0; 3], b.clone()]), GUESS_DEP), vec![1]);
        // the kept set is what later columns are tested against: after dropping the duplicate,
        // b is independent of a and kept, so a third column equal to b is dependent
        assert_eq!(dependent_columns(&gram_of(&[a.clone(), a.clone(), b.clone(), b.clone()]), GUESS_DEP), vec![1, 3]);
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
        println!("33³ warm start: first {first:?} ({}), again (tol {tol2:.2e}) {again:?}; evals {e_first:?} / {e:?}", per_it(&first));
        assert_eq!(first.stop, Stop::Floor, "{first:?}");
        assert!(again.iterations <= 1, "{again:?}");
        assert_eq!(again.stop, Stop::Converged, "{again:?}");
        assert!(again.max_residual <= tol2, "{again:?}");
        // 0 iterations: X is untouched and λ is read back fresh; the f32 dots put ~3e-9
        // relative noise on a Rayleigh quotient (measured 7e-9 at 2.5 Ha).
        for (a, b) in e_first.iter().zip(&e) { assert!((a - b).abs() < 1e-7, "warm start moved an eigenvalue: {a} → {b}"); }
        let after: Vec<Vec<f32>> = x.iter().map(|f| f.read_interior_f32(&rig.ctx, &rig.g)).collect();
        let again2 = lobpcg(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, tol2, 500);
        assert_eq!(again2.iterations, 0);
        for (f, before) in x.iter().zip(&after) { assert_eq!(f.read_interior_f32(&rig.ctx, &rig.g), *before, "a 0-iteration exit must leave X bit-identical"); }
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
            // The reported λ is the Rayleigh quotient with the f32 operator: the f32 coefficient
            // bias is ~1.5e-8 per Ha of kinetic energy, plus ~1e-8 relative dot-evaluation noise
            // (measured offset 3.8e-8 at λ = 1.5, T = 0.75 Ha).
            assert!((lam - e[i]).abs() < 1e-7, "reported eval {} vs f64 Rayleigh quotient {lam}", e[i]);
        }
        println!("33³ tol 1e-10: {stats:?} ({}); floor model {floor:.2e}; f64-recomputed true residual {true_res:.2e}; evals {e:?}", per_it(&stats));
        assert_eq!(stats.stop, Stop::Floor, "{stats:?}");
        assert!(stats.iterations < max_iter, "looped to max_iter");
        assert!(stats.max_residual > 0.5 * floor && stats.max_residual < 1.5 * floor, "floor {:.3e} vs model {floor:.3e}", stats.max_residual);
        assert!((stats.max_residual - true_res).abs() < 0.1 * true_res, "reported {:.3e} vs f64 recomputation {true_res:.3e}", stats.max_residual);
        assert_eq!(stats.fallbacks, 0, "the raw subspace needed the MGS fallback: {stats:?}");
    }

    /// The MGS fallback forced on every iteration (`alpha_max` = 0): the solver still converges
    /// to the same eigenvalues with orthonormal vectors, at Task 5's readback count (2k + 1
    /// per iteration: one Gram, k − 1 + np MGS vectors, one re-measured Gram).
    #[test]
    fn forced_fallback_converges() {
        let (cg, v) = oscillator(25, 0.3);
        let k = 3;
        let Some(rig) = rig(&cg, &v) else { return };
        let e_cpu = cpu_evals(&cg, &v, k);
        let mut x = upload(&rig, &random_guess(&cg, k));
        let mut e = vec![0.0; k];
        let mut work = LobpcgWork::new(&rig.ctx, &rig.g, k);
        let h = GpuHamiltonian { v: &rig.v };
        let tol = 2.0 * residual_floor_estimate(&rig.g);
        let stats = lobpcg_with(&rig.ctx, &rig.k, &rig.g, &rig.r, &h, &mut x, &mut e, &mut work, tol, tol, 500, 0.0, 0.0);
        let m = gram(&rig, &x);
        let worst_orth = (0..k * k).map(|i| (m[i] - if i / k == i % k { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
        let worst = e.iter().zip(&e_cpu).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("25³ k={k} forced fallback: {stats:?} ({}); evals {e:?} cpu {e_cpu:?} (worst |Δ| {worst:.2e}); |XᵀX − I| {worst_orth:.2e}", per_it(&stats));
        assert_eq!(stats.stop, Stop::Converged, "{stats:?}");
        assert_eq!(stats.fallbacks, stats.iterations);
        assert!(worst_orth < 1e-6 && worst < 3e-7);
        // 1 (guess) + 1 (exit) + per iteration: 1 + (2k − 1 − 1 on the first, P empty) + 1
        assert!(stats.readbacks > stats.iterations * (2 * k), "{stats:?}");
    }

    /// A shifted oscillator V = ½r² − 20 (λ ≈ −18.5 Ha, same eigenvectors, same preconditioner
    /// since V − λ is unchanged): the residual floor must not grow with |λ|. With λ cast to a
    /// single f32 the residual would carry a coherent (λ − λ_f32)·x ≈ 1e-6 component; the
    /// head/tail split removes it, leaving only the white evaluation noise ∝ |λ|, a few % in
    /// quadrature (+9% predicted). A single floor exit fluctuates by ±20% (the exit lands on a
    /// random sample of the rounding noise: the unshifted exit alone moved 1.89e-6 → 2.29e-6 from
    /// an ulp-level trajectory change), so each case averages eight exits — the cold solve and
    /// seven warm re-solves at tol 1e-10, each `STALL_WINDOW` iterations of fresh samples
    /// (Task 10: four exits gave means 1.77e-6 / 2.14e-6, +21 % with the one-readback solver,
    /// whose unshifted floor is lower than Task 5's 1.96e-6; eight exits are within the bound).
    /// Means within 20%; eigenvalues shift by exactly −20 up to the f32 rounding of V
    /// (ulp(20) ≈ 2e-6 per point, ≲ 1e-6 on λ).
    #[test]
    fn floor_independent_of_eigenvalue_shift() {
        let (cg, v) = oscillator(33, 0.25);
        let shift = 20.0;
        let vs: Vec<f64> = v.iter().map(|p| p - shift).collect();
        let k = 2;
        let Some(rig_p) = rig(&cg, &v) else { return };
        let Some(rig_s) = rig(&cg, &vs) else { return };
        let run = |rg: &Rig| {
            let mut x = upload(rg, &random_guess(&cg, k));
            let mut e = vec![0.0; k];
            let mut work = LobpcgWork::new(&rg.ctx, &rg.g, k);
            let h = GpuHamiltonian { v: &rg.v };
            let samples: Vec<f64> = (0..8).map(|_| {
                let s = lobpcg(&rg.ctx, &rg.k, &rg.g, &rg.r, &h, &mut x, &mut e, &mut work, 1e-10, 500);
                assert_eq!(s.stop, Stop::Floor, "{s:?}");
                s.max_residual
            }).collect();
            (samples.iter().sum::<f64>() / samples.len() as f64, samples, e)
        };
        let (plain, s_plain, e_plain) = run(&rig_p);
        let (shifted, s_shift, e_shift) = run(&rig_s);
        println!("33³ floor exits: V = ½r² {s_plain:?} mean {plain:.3e} evals {e_plain:?}; V − 20: {s_shift:?} mean {shifted:.3e} evals {e_shift:?}; model {:.3e}", residual_floor_estimate(&rig_p.g));
        let rel = (shifted - plain).abs() / plain;
        assert!(rel < 0.2, "floor changed by {:.0}% under a −20 Ha shift: {shifted:.3e} vs {plain:.3e}", 100.0 * rel);
        for (a, b) in e_plain.iter().zip(&e_shift) { assert!((a - shift - b).abs() < 5e-6, "shifted eval {b} vs {a} − 20"); }
    }

    /// The stop decision (module doc) on measured points: the early phase (residual ~30 Ha for
    /// 8+ iterations while the Ritz values descend) must not stop; a stall is a floor only
    /// inside the band; otherwise max_iter decides.
    #[test]
    fn stop_classification() {
        let floor = 1.8e-6;
        assert_eq!(classify(5e-7, 1e-6, 10, 500, 2, floor, 1e-6), Some(Stop::Converged));
        // 33³ k = 4 iteration 8: max residual 13 Ha, running minimum last halved at 0 → continue
        assert_eq!(classify(13.0, 1e-6, 8, 500, 0, floor, 1e-6), None);
        assert_eq!(classify(1.0, 1e-6, 40, 500, 13, floor, 1e-6), None); // above the band: never a floor
        // the slow 2.5 Ha tail: 5.5e-6 at 63 → 3.9e-6 at 72 (not halved in 9) → continue
        assert_eq!(classify(3.9e-6, 1e-6, 72, 500, 63, floor, 1e-6), None);
        // at the floor: 1.9e-6, no halving for 12 iterations → Floor
        assert_eq!(classify(1.9e-6, 1e-10, 69, 500, 57, floor, 1e-6), Some(Stop::Floor));
        assert_eq!(classify(1.9e-6, 1e-10, 68, 500, 57, floor, 1e-6), None); // 11 iterations: not yet
        assert_eq!(classify(1.0, 1e-6, 500, 500, 490, floor, 1e-6), Some(Stop::MaxIter));
        // the guess band: a warm start at 1.4× tol is accepted at iteration 0 only
        assert_eq!(classify(1.4e-6, 1e-6, 0, 500, 0, floor, 1.5e-6), Some(Stop::Converged));
        assert_eq!(classify(1.4e-6, 1e-6, 1, 500, 0, floor, 1.5e-6), None);
    }

    /// The Rayleigh–Ritz amplification α: 1 for an orthonormal S, 1/√σ_min when the Ritz
    /// vector lies along the small direction of a nearly dependent pair.
    #[test]
    fn rayleigh_ritz_alpha() {
        // S = [s0, s1] with ⟨s0,s1⟩ = ρ: B̃'s directions are (1,±1)/√2 with σ = 1 ± ρ. With
        // A = [[1, a],[a, 1]] the Rayleigh quotients of the two directions are (1 + a)/(1 + ρ)
        // and (1 − a)/(1 − ρ); a = 0.999 makes the difference direction the lowest for every
        // ρ below 0.999, so the Ritz vector lies along the small direction and α = 1/√(1 − ρ).
        for rho in [0.0, 0.5, 0.9, 0.99] {
            let b = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
            let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.999, 0.999, 1.0]);
            let (th, c, alpha) = rayleigh_ritz(&a, &b, 1);
            let want = 1.0 / (1.0 - rho).sqrt();
            println!("ρ = {rho}: θ {th:?} c {:?} α {alpha:.3} (expected {want:.3})", c.as_slice());
            assert!((th[0] - 0.001 / (1.0 - rho)).abs() < 1e-9, "θ {th:?}");
            assert!((alpha - want).abs() < 1e-9, "α {alpha} vs {want}");
        }
        // and the sum direction (a = −0.999): α = 1/√(1 + ρ) < 1
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let a = DMatrix::from_row_slice(2, 2, &[1.0, -0.999, -0.999, 1.0]);
        let (_, _, alpha) = rayleigh_ritz(&a, &b, 1);
        assert!((alpha - 1.0 / 1.5f64.sqrt()).abs() < 1e-9, "α {alpha}");
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
                println!("{n}³ h={h} k={nk} tol {tol:.0e}: {} it ({:?}) in {ms:.1} ms ({:.2} ms/it, {}), residual {:.2e}, floor model {:.2e}; evals {e:?}; warm again: {} it ({:?}) {warm_ms:.1} ms",
                         s.iterations, s.stop, ms / s.iterations.max(1) as f64, per_it(&s), s.max_residual, residual_floor_estimate(&g),
                         warm.iterations, warm.stop);
            }
        }
    }
}
