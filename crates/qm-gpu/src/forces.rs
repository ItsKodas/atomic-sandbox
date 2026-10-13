//! Hellmann-Feynman forces on the GPU — the port of `qm::forces` (read its module doc first:
//! what is differentiated, why there is no Pulay term, the sign chain, the d → 0 limit of g′,
//! the expected size of the egg-box force and of HF − FD). Atomic units (Hartree/Bohr).
//!
//! # The same force, in the form the kernel evaluates
//! With a = σ√2, d = |r − R_I|, x = d/a and E(x) = erf(x)/x (the function `nuclear.wgsl`
//! evaluates, finite at 0):
//!   V_I(d) = −Z_I g(d),   g(d) = erf(d/a)/d = E(x)/a,   g′(d) = E′(x)/a².
//! The electronic force is F_I^el = +∫ρ ∇_r V_I d³r with ∇_r V_I = −Z_I g′(d)(r − R_I)/d
//! (`qm::forces`), and since d = a x,
//!   ∇_r V_I = −(Z_I/a³) · G(x) · (r − R_I),   G(x) ≡ E′(x)/x.
//! G is even and finite at the nucleus: from E(x) = (2/√π) Σ_k c_k x^{2k}, c_k = (−1)^k/(k!(2k+1)),
//!   G(x) = (2/√π) Σ_{k≥1} 2k c_k x^{2k−2} = (2/√π)(−2/3 + 2x²/5 − x⁴/7 + …),   G(0) = −(2/√π)(2/3),
//! which is the CPU doc's g′(d) = −(4/(3√π)) d/a³ + O(d³) (g′(0) = 0 exactly, and the integrand
//! → +Z (4/(3√π a³))(r − R_I) → 0 linearly). On the grid r_p − R_I = h·dc_p with
//! dc_p = (i − cell) − frac (exact integer minus f32 fraction, the `nuclear.wgsl` layout), so
//!   F_I^el ≈ dv · Σ_p ρ_p · c · G(x_p) · dc_p,   c = −Z_I h/a³,   x_p = |dc_p|·(h/a).
//! Nothing divides by d: a nucleus exactly on a grid point has dc = 0 there and contributes
//! exactly 0 from that point, which is the CPU's `d < 1e-8` skip for free (the skipped term is
//! < 1e-9 Ha/Bohr either way, `qm::forces` doc).
//!
//! # G(x) in f32: the two branches and the step at the switch
//! The derivative of the `nuclear.wgsl` split, so that the force is the gradient of the V_nuc
//! the GPU actually integrates (to the accuracy of that split):
//! - x < 1: the Maclaurin series above to k = 12 (Horner in y = x²). Alternating with
//!   decreasing terms for y ≤ 1, so the truncation is below the first omitted term,
//!   (2/√π)·2/(12!·27) = 1.7e-10 at x = 1 (4e-10 relative to |G(1)| = 0.4276), and the f32
//!   rounding is that of a well-conditioned Horner sum (Σ|terms|/|sum| ≈ 3 at x = 1).
//! - x ≥ 1: E′(x) = [(2/√π) x e^{−x²} − erf(x)]/x², written with erf = 1 − erfc:
//!   G(x) = [(2/√π) x e^{−x²} − 1 + erfc(x)] / x³,
//!   erfc(x) = e^{−x²}·f(x)/(√π x) with f = √π x e^{x²} erfc(x) a degree-12 Chebyshev fit on
//!   [1, 4) and its asymptotic series beyond — the same `erfc_scaled` as `nuclear.wgsl`
//!   (Task 10; |Δerf| ≤ 6.1e-9 with the f32 coefficients, 9.7e-10 with f64 ones). The bracket
//!   is −0.4276 at x = 1 and → −1 as x → ∞ (no cancellation), so the fit error enters G as
//!   ≤ 1.5e-8 relative at the switch — below the f32 rounding of the closed form (~2e-7).
//!   The e^{−x²} term is the *exact* derivative of erf. Task 7's Abramowitz–Stegun 7.1.26
//!   (|Δerf| ≤ 1.4e-7 on [1, 4], oscillating with x on a scale of ~0.1–1 so that |ε′| ≲ 1e-6)
//!   made the GPU's V_nuc −Z(erf + ε)/d while this force differentiated the exact erf; the
//!   mismatch budget was ≲ ∫ρ Z|ε′|/(a d) ~ 1e-5 Ha/Bohr before angular cancellation and
//!   measured −2.2e-6 (the Richardson residual below). With the fit, ε and ε′ are at the
//!   1e-8 level and the force is the gradient of the V_nuc the GPU integrates up to f32 rounding.
//!
//! At the switch both branches are exact to the fit, so the step is f32 rounding. Measured ON
//! THE SHADER (`shader_g_over_x_probe`: ρ = 1 at one grid point, G read off `electronic_forces`
//! at chosen x), Task 10: step −5.7e-8 relative between x = 0.999 and 1.001 = 0.8 f32 ulp of
//! G(1) = −0.4275933 (Task 7 with A&S: −4.05e-7 = 5.8 ulps); x = 0 gives exactly 0; max
//! relative error at the probe points (0.01 … 8, incl. 1.024) is 1.31e-7 at x = 0.9, the series
//! branch's own rounding (Task 7: 5.0e-7 at x = 1.024, the A&S error); bound 3e-7. The Rust f32
//! mirror of the shader (same decimal literals;
//! `g_over_x_mirror_is_accurate_and_continuous_at_the_switch`) agrees with the shader to
//! 1.5e-7 ≈ 2 ulps (the GPU's exp() and contraction) and its dense sweep over [0, 8] peaks at
//! 2.9e-7 at x = 4.10, where the asymptotic branch starts; G(0) is the constant to 1 ulp. Both
//! branches are even in x, as required. The kernel end to end
//! (`force_kernel_matches_f64_integrand_on_an_analytic_density`, 33³, three nuclei including
//! one on a grid point and one 1e-3 Bohr off one): ≤ 1.3e-7 Ha/Bohr vs the f64 integrand on
//! forces of 0.2–1.5, the same kernel-only (f32-rounded inputs) — the f32 partial-sum level.
//!
//! # What the GPU force is the gradient of, and the HF-vs-FD budget (measured)
//! The GPU SCF minimises its own discrete functional E_h^f32 (f32 V_nuc from the same erf
//! split, f32 orbitals/densities, f64-finished reductions). The force above is its exact
//! Hellmann-Feynman gradient up to (i) the ε′ mismatch between the erf in V_nuc and the
//! exact-erf derivative the kernel uses (Task 7's A&S: ≲ 1e-5 before angular cancellation,
//! measured −2.2e-6; now at the 1e-8 level, above); (ii) the non-stationarity of the state — first order in the
//! density residual (Δρ ≤ tol times ∫|∇V_I| ≈ 4 Ha/Bohr per unit charge) and in the LOBPCG
//! residual through 2Σ_i f_i ⟨∂ψ_i/∂R | (H − ε_i)ψ_i⟩, whose Cauchy–Schwarz bound at the f32
//! floor (‖r‖ ≈ 5e-6, ‖∂ψ/∂R‖ ≈ 1) is 2e-5 but whose actual value is far smaller because the
//! floor residual is rounding noise with no projection on the smooth ∂ψ/∂R; and (iii) the f32
//! rounding of the partial sums, ≲ 1e-6. Measured against the CPU's HF force at the same σ
//! (49³, H₂, `qm::forces` at tol 1e-9/eig 1e-8): **1.2e-7** at R = 1.4 with the GPU at tol
//! 1e-7, and 1.5e-6 at R = 1.0 / 2.5 and 1.9e-6 at 33³ with the GPU at tol 1e-6 — i.e. (ii)
//! is the Δρ term, ~4·Δρ, and the eigen-floor term is invisible.
//!
//! Finite differences of the GPU energy are another matter: the GPU total energy carries a
//! geometry-to-geometry (path-dependent) f32 scatter, measured directly as 5.8e-7 Ha between a
//! cold solve and a warm solve from a neighbouring geometry at tol 1e-7 (the energy components
//! each scatter at ~1e-6 and partially cancel, Task 6). Over 2δ that is σ_E/(2δ) ≈ 3e-5
//! Ha/Bohr at the brief's δ = 0.01 — the same size as the central-difference truncation
//! δ²|E‴|/6 = 2.7e-5 (CPU, E‴ ≈ −1.6 Ha/Bohr³) — so the GPU FD is noise-limited at δ = 0.01:
//! measured HF − FD = −7.9e-6 / −1.10e-4 / −4.35e-4 at δ = 0.01 / 0.02 / 0.04 (ratios 13.9 and
//! 3.94 — the 0.02/0.04 pair is pure δ², the 0.01 point is noise). Richardson on (0.02, 0.04)
//! gives HF − FD = −2.2e-6 (budget ≈ 0.75 σ_E/0.02 ≈ 2e-5 rms noise, truncation residual
//! < 1e-6); on (0.01, 0.02) it gives +2.6e-5, the amplified δ = 0.01 noise. The test asserts
//! the brief's δ = 0.01 bound, the (0.02, 0.04) Richardson agreement and, as the direct check,
//! the GPU HF against the CPU HF.
//!
//! # The forces do not sum to zero
//! As on the CPU, Σ_I F_I ≠ 0 on a grid: the egg-box effect breaks translational invariance
//! (moving every nucleus by the same δ changes E_h), bounded by ~4e-5 Ha/Bohr per nucleus at
//! σ = h = 0.25 (`qm::forces` doc). The CPU's H₂ tests see ~1e-13 only because their geometry
//! is mirror-symmetric about a grid plane and the f64 SCF respects that symmetry; the GPU's
//! f32 state does not respect it exactly (fixed but non-symmetric reduction orders, f32
//! fractions 0.8 and 0.2 of a cell are not exact mirror images), so the GPU H₂ sum is at the
//! f32 noise level ~1e-6 instead. Dynamics (M3/M4) must not assert total-momentum conservation
//! to better than the egg-box bound.
//!
//! # Precision and cost
//! One dispatch per nucleus: each workgroup of 256 grid-strides over the interior, accumulates
//! the vec3 integrand per thread, tree-reduces in workgroup memory and writes one vec4 partial
//! per workgroup (`force_partials` in `shaders/forces.wgsl`; the same shape and fixed workgroup
//! count as `reduce.wgsl`, hence deterministic). All nuclei go in one encoder with one copy and
//! one readback (the partials/staging pair is cached on the `GpuSolver`, grow-only — M4: forces
//! run every MD step); the CPU sums the partials per component in f64 (Neumaier) × dv and adds the
//! point nuclear–nuclear term Σ_{J≠I} Z_I Z_J (R_I − R_J)/|R_I − R_J|³ in f64
//! (`qm::nuclear`-style; coincident nuclei → `QmError::CoincidentNuclei`, lower index first).
use crate::grid::{map_read_f32, Field, GpuGrid};
use crate::kernels::{encoder, uniform, Kernels, WORKGROUP};
use crate::physics::NucleusRecord;
use crate::pointwise::check;
use crate::reduce::{neumaier, MAX_PARTIALS};
use crate::scf::{GpuSolver, GpuState};
use gpu::GpuContext;
use qm::{Nucleus, QmError};
use std::f64::consts::SQRT_2;

/// Nuclear coincidence threshold (Bohr), the same value as `qm::nuclear::nuclear_repulsion`
/// and `qm::forces`.
const ON_SITE: f64 = 1e-8;

/// Bytes per nucleus in the partials buffer: `MAX_PARTIALS` vec4<f32> (a multiple of the
/// 256-byte storage offset alignment).
const NUCLEUS_STRIDE: u64 = MAX_PARTIALS as u64 * 16;

/// Uniform of `force_partials` (see `shaders/forces.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ForceParams { cell: [i32; 4], frac: [f32; 4], p: u32, n: u32, pad0: u32, pad1: u32, h_over_a: f32, c: f32, pad2: [f32; 2] }

/// Workgroups (= partials) per dispatch: the `Reducer`'s count, min(ceil(n³/256), MAX_PARTIALS).
fn workgroups(g: &GpuGrid) -> u32 {
    (g.interior_len().div_ceil(WORKGROUP as usize) as u32).clamp(1, MAX_PARTIALS)
}

/// The forces' (partials, staging) buffer pair, cached on the `GpuSolver` because `forces` runs
/// every MD step (M4): grown to the largest nucleus count seen, never shrunk; `allocations`
/// counts the pairs created (the reuse witness of `forces_buffers_are_reused`).
#[derive(Default)]
pub(crate) struct ForceScratch { bufs: Option<(wgpu::Buffer, wgpu::Buffer, usize)>, pub(crate) allocations: usize }

impl ForceScratch {
    /// The pair for `count` nuclei (allocated or grown here only when too small).
    fn ensure(&mut self, ctx: &GpuContext, count: usize) -> (&wgpu::Buffer, &wgpu::Buffer) {
        if self.bufs.as_ref().is_none_or(|(_, _, cap)| *cap < count) {
            let bytes = count as u64 * NUCLEUS_STRIDE;
            let partials = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("force partials"), size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("force staging"), size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
            });
            self.bufs = Some((partials, staging, count));
            self.allocations += 1;
        }
        let (p, s, _) = self.bufs.as_ref().expect("allocated above");
        (p, s)
    }
}

/// The electronic Hellmann-Feynman force on each nucleus, F_I^el = ∫ρ ∇_r V_I d³r, from the
/// interior of `rho` (module doc). One dispatch per nucleus in one submission, one readback;
/// partials summed per component in f64 (Neumaier) × dv. No nuclei ⇒ empty, no GPU work.
/// Allocates its buffer pair per call; `GpuSolver::forces` uses a cached pair.
pub fn electronic_forces(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho: &Field, nuclei: &[Nucleus], sigma: f64) -> Vec<[f64; 3]> {
    electronic_forces_with(ctx, k, g, rho, nuclei, sigma, &mut ForceScratch::default())
}

/// `electronic_forces` with the buffer pair from `scratch` (grown if too small).
#[allow(clippy::too_many_arguments)]
pub(crate) fn electronic_forces_with(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho: &Field, nuclei: &[Nucleus], sigma: f64, scratch: &mut ForceScratch) -> Vec<[f64; 3]> {
    check(g, &[rho]);
    assert!(sigma > 0.0, "electronic_forces: sigma must be positive, got {sigma}");
    if nuclei.is_empty() { return Vec::new(); }
    let wg = workgroups(g);
    let a = sigma * SQRT_2;
    let bytes = nuclei.len() as u64 * NUCLEUS_STRIDE;
    let (partials, staging) = scratch.ensure(ctx, nuclei.len());
    let mut enc = encoder(ctx, "forces");
    for (i, nuc) in nuclei.iter().enumerate() {
        let rec = NucleusRecord::new(g, nuc);
        let params = ForceParams {
            cell: rec.cell, frac: [rec.frac[0], rec.frac[1], rec.frac[2], 0.0],
            p: g.p as u32, n: g.n as u32, pad0: 0, pad1: 0,
            h_over_a: (g.h / a) as f32, c: (-(nuc.z as f64) * g.h / (a * a * a)) as f32, pad2: [0.0; 2],
        };
        let u = uniform(ctx, "force_partials", &params);
        let block = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: partials, offset: i as u64 * NUCLEUS_STRIDE, size: wgpu::BufferSize::new(wg as u64 * 16),
        });
        let bg = k.bind_group(ctx, "force_partials", &[u.as_entire_binding(), rho.buffer.as_entire_binding(), block]);
        k.record(&mut enc, "force_partials", &bg, wg);
    }
    enc.copy_buffer_to_buffer(partials, 0, staging, 0, Some(bytes));
    ctx.queue.submit([enc.finish()]);
    let raw = map_read_f32(ctx, staging, nuclei.len() * MAX_PARTIALS as usize * 4);
    let dv = g.dv();
    raw.chunks(MAX_PARTIALS as usize * 4).map(|block| {
        let block = &block[..wg as usize * 4];
        std::array::from_fn(|comp| neumaier(block.iter().skip(comp).step_by(4).map(|v| *v as f64)) * dv)
    }).collect()
}

impl GpuSolver {
    /// Force on each nucleus (Hartree/Bohr) from the converged `state`: the Hellmann-Feynman
    /// electronic part ∫ρ ∇_r V_I on the GPU plus the point nuclear–nuclear repulsion in f64
    /// (module doc) — `qm::forces` for a `GpuState`. `nuclei` and `sigma` must be what the
    /// state was converged with (as the CPU: panics if `state.grid`/`state.sigma` differ from
    /// this solver's grid/`sigma`). Errors on coincident nuclei (the repulsion is undefined),
    /// lower index first.
    pub fn forces(&self, state: &GpuState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError> {
        assert_eq!(&state.grid, self.grid(), "GpuState was converged on a different grid");
        assert!((state.sigma - sigma).abs() < 1e-15, "GpuState was converged with sigma = {}, forces called with {}", state.sigma, sigma);
        // nuclear–nuclear part first (f64, and the coincidence check before any GPU work)
        let mut out = vec![[0.0f64; 3]; nuclei.len()];
        for (i, nuc) in nuclei.iter().enumerate() {
            let z = nuc.z as f64;
            for (j, other) in nuclei.iter().enumerate() {
                if i == j { continue; }
                let dv = [nuc.pos[0] - other.pos[0], nuc.pos[1] - other.pos[1], nuc.pos[2] - other.pos[2]];
                let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if d < ON_SITE { return Err(QmError::CoincidentNuclei(i.min(j), i.max(j))); }
                let s = z * other.z as f64 / (d * d * d);
                for a in 0..3 { out[i][a] += s * dv[a]; }
            }
        }
        for (o, e) in out.iter_mut().zip(self.electronic_forces_cached(&state.rho, nuclei, sigma)) {
            for a in 0..3 { o[a] += e[a]; }
        }
        Ok(out)
    }

    /// `electronic_forces` with this solver's cached buffer pair (`ForceScratch`).
    pub(crate) fn electronic_forces_cached(&self, rho: &Field, nuclei: &[Nucleus], sigma: f64) -> Vec<[f64; 3]> {
        electronic_forces_with(&self.ctx, &self.k, &self.g, rho, nuclei, sigma, &mut self.forces.borrow_mut())
    }

    /// Buffer pairs the force cache has allocated so far (grow-only; M4 reuse witness).
    pub fn force_buffer_allocations(&self) -> usize { self.forces.borrow().allocations }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Field, GpuGrid};
    use crate::kernels::Kernels;
    use qm::{Grid, ScfSettings, ScfStats, System};
    use std::f64::consts::PI;
    use std::time::Instant;

    // ---- the f32 mirror of the shader's G(x): the SAME decimal literals as forces.wgsl, so both
    // parsers round to the same f32 (a shortened literal can differ by 1 ulp — it did once);
    // `shader_g_over_x_probe` pins the mirror against the shader itself ----
    #[allow(clippy::excessive_precision, clippy::approx_constant)] // the shader's literal, on purpose
    const TWO_OVER_SQRT_PI: f32 = 1.1283791670955126;
    #[allow(clippy::excessive_precision)]
    const SQRT_PI: f32 = 1.7724538509055159;
    /// Chebyshev coefficients of √π x e^{x²} erfc(x) on [1, 4] (forces.wgsl `CH0..CH12`).
    #[allow(clippy::excessive_precision)]
    const CH: [f32; 13] = [1.8048557623e+00, 9.5036126435e-02, -3.4561538972e-02, 1.0820666035e-02, -3.0682293564e-03, 8.0784264219e-04, -2.0040632772e-04, 4.7291524862e-05, -1.0687072521e-05, 2.3243735801e-06, -4.8843173186e-07, 9.9470529692e-08, -1.9682356004e-08];
    /// s_k = 2k c_k = (−1)^k · 2/((k−1)!(2k+1)), k = 1..=12
    const S: [f32; 12] = [-2.0 / 3.0, 2.0 / 5.0, -1.0 / 7.0, 1.0 / 27.0, -1.0 / 132.0, 1.0 / 780.0, -1.0 / 5400.0,
                          1.0 / 42840.0, -1.0 / 383040.0, 1.0 / 3810240.0, -1.0 / 41731200.0, 1.0 / 498960000.0];

    fn g_series_f32(x: f32) -> f32 {
        let y = x * x;
        let mut s = S[11];
        for k in (0..11).rev() { s = S[k] + y * s; }
        TWO_OVER_SQRT_PI * s
    }

    /// √π x e^{x²} erfc(x) for x ≥ 1 as the shader computes it (Clenshaw on [1, 4), asymptotic beyond).
    fn erfc_scaled_f32(x: f32) -> f32 {
        if x < 4.0 {
            let t = (2.0 * x - 5.0) / 3.0;
            let (mut b1, mut b2) = (0.0f32, 0.0f32);
            for k in (1..=12).rev() { let b0 = 2.0 * t * b1 - b2 + CH[k]; b2 = b1; b1 = b0; }
            t * b1 - b2 + 0.5 * CH[0]
        } else {
            let u = 1.0 / (x * x);
            1.0 + u * (-0.5 + u * (0.75 + u * (-1.875 + u * (6.5625 + u * (-29.53125)))))
        }
    }

    fn g_closed_f32(x: f32) -> f32 {
        let erfc = (-x * x).exp() * erfc_scaled_f32(x) / (SQRT_PI * x);
        ((TWO_OVER_SQRT_PI * x * (-x * x).exp() - 1.0) + erfc) / (x * x * x)
    }

    fn g_over_x_f32(x: f32) -> f32 { if x < 1.0 { g_series_f32(x) } else { g_closed_f32(x) } }

    /// G(x) = E′(x)/x in f64, series to k = 20 (for x ≲ 1).
    fn g_exact_series(x: f64) -> f64 {
        let y = x * x;
        let (mut s, mut fact) = (0.0f64, 1.0f64); // fact = (k−1)!
        let mut yk = 1.0; // y^{k−1}
        for k in 1..=20 {
            if k > 1 { fact *= (k - 1) as f64; yk *= y; }
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            s += sign * 2.0 / (fact * (2 * k + 1) as f64) * yk;
        }
        (2.0 / PI.sqrt()) * s
    }

    /// G(x) in f64, closed form [(2/√π) x e^{−x²} − erf(x)]/x³ (cancels for small x).
    fn g_exact_closed(x: f64) -> f64 {
        ((2.0 / PI.sqrt()) * x * (-x * x).exp() - libm::erf(x)) / (x * x * x)
    }

    /// G(x) = E′(x)/x in f64: the series below x = 0.5, the closed form above.
    fn g_over_x_exact(x: f64) -> f64 { if x < 0.5 { g_exact_series(x) } else { g_exact_closed(x) } }

    /// The f32 mirror of the shader's G(x) against f64 on x ∈ [0, 8], the exact constant at
    /// x = 0, and the step between the two branches at the switch x = 1 (f32 rounding only
    /// since Task 10's fit; Task 7's A&S error was ≤ 3.5e-7 relative there, module doc).
    #[test]
    fn g_over_x_mirror_is_accurate_and_continuous_at_the_switch() {
        // G(0) is the constant −(2/√π)(2/3) = −0.7522528: one f32 product, no 0/0, no branch
        let g0 = -(2.0 / PI.sqrt()) * 2.0 / 3.0;
        let g0_rel = ((g_over_x_f32(0.0) as f64 - g0) / g0).abs();
        assert!(g0_rel < 1.5e-7, "G(0) = {} vs {g0} ({g0_rel:.2e} relative, > 1 f32 ulp)", g_over_x_f32(0.0));
        assert!((g_over_x_exact(0.0) - g0).abs() < 1e-15);
        // the two f64 forms agree where both are accurate (the reference is consistent)
        for x in [0.5, 0.8, 1.0] {
            assert!(((g_exact_series(x) - g_exact_closed(x)) / g_exact_closed(x)).abs() < 1e-13, "f64 reference at x = {x}");
        }
        let (mut worst, mut at) = (0.0f64, 0.0f64);
        for i in 0..=8000 {
            let x = i as f64 * 1e-3;
            let got = g_over_x_f32(x as f32) as f64;
            let want = g_over_x_exact(x);
            let rel = ((got - want) / want).abs();
            if rel > worst { worst = rel; at = x; }
        }
        let step = (g_series_f32(1.0) - g_closed_f32(1.0)) as f64;
        let step_rel = (step / g_over_x_exact(1.0)).abs();
        let series_err = ((g_series_f32(1.0) as f64 - g_over_x_exact(1.0)) / g_over_x_exact(1.0)).abs();
        let closed_err = ((g_closed_f32(1.0) as f64 - g_over_x_exact(1.0)) / g_over_x_exact(1.0)).abs();
        let g1 = g_over_x_exact(1.0).abs() as f32;
        let ulp = (g1.next_up() - g1) as f64;
        println!("G(x) f32 mirror: max rel err {worst:.2e} at x = {at:.3}; at the switch x = 1: series {series_err:.2e}, closed {closed_err:.2e}, step {step:+.3e} ({step_rel:.2e} relative, {:.1} f32 ulp of G(1) = {:.7})",
                 step.abs() / ulp, g_over_x_exact(1.0));
        // Task 10: 2.9e-7 at x = 4.10 (the asymptotic branch's first point; f32 rounding of the
        // closed form), step 3.0e-8, series 5.9e-8 / closed 1.1e-8 at the switch. Bounds 4e-7 /
        // 1e-7 / 1.5e-7 (Task 7 with A&S: 5.1e-7 at x = 1.024 and a 4.1e-7 step; bounds 6e-7 / 1e-6).
        assert!(worst < 4e-7, "max relative error {worst:.3e} at x = {at}");
        assert!(step_rel < 1e-7, "step at the switch {step_rel:.3e}");
        assert!(series_err < 1.5e-7, "series branch at x = 1: {series_err:.3e}");
    }

    /// G(x) measured ON THE SHADER: ρ = 1 at one interior grid point p₀ (0 elsewhere) and one
    /// nucleus per probe x at R = r_p₀ − (x·a, 0, 0), so dc = (x·a/h, 0, 0) and the kernel's
    /// only non-zero term is F_x = dv · c · G(x) · dc_x: G(x) = F_x/(dv·c·dc_x) read straight off
    /// `electronic_forces` (one dispatch per x, one call). x = 0 (nucleus on p₀) must give
    /// exactly 0 (dc = 0, no branch); every other probe is checked against f64 at the x the
    /// shader sees (its f32 fraction, f32 h/a and the f32 ops mimicked here — the GPU's sqrt
    /// may still differ by ≤ 1 ulp of x, ≲ 1e-7 relative in G), and the step at the switch is
    /// measured as the error jump between x = 1 − 1e-3 and 1 + 1e-3. The Rust mirror of the
    /// shader is asserted against these shader values pointwise.
    #[test]
    fn shader_g_over_x_probe() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = Grid::cube(17, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let sigma = 0.25;
        let a = sigma * SQRT_2;
        let (i0, j0, k0) = (8usize, 7usize, 9usize);
        let p0 = cg.idx(i0, j0, k0);
        let r0 = cg.point(p0);
        let xs = [0.0, 0.01, 0.1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.024, 1.5, 3.0, 8.0]; // 1.024: the mirror sweep's worst point
        let nuclei: Vec<Nucleus> = xs.iter().map(|x| Nucleus { pos: [r0[0] - x * a, r0[1], r0[2]], z: 1 }).collect();
        let mut delta = vec![0.0; cg.len()];
        delta[p0] = 1.0;
        let rho = Field::new(&ctx, &g, "delta");
        rho.upload_interior(&ctx, &g, &delta);
        let f = electronic_forces(&ctx, &k, &g, &rho, &nuclei, sigma);
        let h_over_a = (cg.h / a) as f32;
        let c32 = (-cg.h / (a * a * a)) as f32 as f64; // Z = 1, as the kernel's uniform
        let (mut worst, mut worst_at, mut mirror_worst) = (0.0f64, 0.0f64, 0.0f64);
        let mut err_at = std::collections::HashMap::new();
        println!("shader probe (17³, σ = 0.25):   x_seen        G_shader        G_f64          rel err    mirror − shader");
        for (i, (x, nuc)) in xs.iter().zip(&nuclei).enumerate() {
            let rec = NucleusRecord::new(&g, nuc);
            let dc = [(i0 as i32 - rec.cell[0]) as f32 - rec.frac[0], (j0 as i32 - rec.cell[1]) as f32 - rec.frac[1], (k0 as i32 - rec.cell[2]) as f32 - rec.frac[2]];
            assert_eq!((dc[1], dc[2]), (0.0, 0.0), "probe geometry: the offset is along x only");
            if *x == 0.0 {
                assert_eq!(dc[0], 0.0);
                assert_eq!(f[i], [0.0; 3], "nucleus on the grid point must contribute exactly 0");
                println!("  x = 0: F = {:?} (exactly 0, dc = 0)", f[i]);
                continue;
            }
            assert!(f[i][1] == 0.0 && f[i][2] == 0.0, "off-axis components must be exactly 0: {:?}", f[i]);
            let x32 = (dc[0] * dc[0]).sqrt() * h_over_a; // the shader's length(dc) · h_over_a, in f32
            let g_shader = f[i][0] / (cg.dv() * c32 * dc[0] as f64);
            let want = g_over_x_exact(x32 as f64);
            let rel = (g_shader - want) / want;
            let mirror_rel = (g_over_x_f32(x32) as f64 - g_shader) / g_shader;
            println!("  x = {x:<6}  {x32:<12.7}  {g_shader:<+14.9}  {want:<+14.9}  {rel:+.2e}  {mirror_rel:+.2e}");
            err_at.insert(x.to_bits(), rel);
            if rel.abs() > worst { worst = rel.abs(); worst_at = *x; }
            mirror_worst = mirror_worst.max(mirror_rel.abs());
        }
        let step = err_at[&1.001f64.to_bits()] - err_at[&0.999f64.to_bits()];
        let g1 = g_over_x_exact(1.0).abs() as f32;
        let ulp = ((g1.next_up() - g1) / g1) as f64;
        println!("shader G(x): max rel err {worst:.2e} at x = {worst_at}; switch step (err(1.001) − err(0.999)) {step:+.2e} relative = {:.1} f32 ulp; mirror vs shader max {mirror_worst:.2e}", step.abs() / ulp);
        // Task 10: 1.31e-7 at x = 0.9 (series branch rounding), step −5.7e-8 (0.8 ulp), mirror vs
        // shader 1.5e-7 (the GPU's exp() — Vulkan guarantees only 3 + 2|x| ulps — and FMA
        // contraction). Bounds 3e-7 / 2e-7 / 4e-7 (Task 7 with A&S: 5.0e-7, −4.05e-7, 2.7e-7;
        // bounds 6e-7 / 1e-6 / 1e-6).
        assert!(worst < 3e-7, "shader G(x) max relative error {worst:.3e} at x = {worst_at}");
        assert!(step.abs() < 2e-7, "switch step {step:.3e}");
        assert!(mirror_worst < 4e-7, "the Rust mirror differs from the shader by {mirror_worst:.3e}");
    }

    /// The CPU's integrand (`qm::forces`, f64): F_I^el = ∫ρ (−Z g′(d)/d)(r − R_I).
    fn cpu_electronic(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], sigma: f64) -> Vec<[f64; 3]> {
        nuclei.iter().map(|nuc| {
            let z = nuc.z as f64;
            let mut comp = [vec![0.0; grid.len()], vec![0.0; grid.len()], vec![0.0; grid.len()]];
            for p in 0..grid.len() {
                let r = grid.point(p);
                let dv = [r[0] - nuc.pos[0], r[1] - nuc.pos[1], r[2] - nuc.pos[2]];
                let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
                if d < 1e-8 { continue; }
                let u = d / (sigma * SQRT_2);
                let g_prime = ((2.0 / PI.sqrt()) * u * (-u * u).exp() - libm::erf(u)) / (d * d);
                let gp = -z * g_prime / d;
                for a in 0..3 { comp[a][p] = rho[p] * gp * dv[a]; }
            }
            std::array::from_fn(|a| grid.integrate(&comp[a]))
        }).collect()
    }

    /// Kernel vs the f64 integrand on an uploaded analytic density: a nucleus off-grid, one
    /// exactly on a grid point (dc = 0 there) and one 1e-3 Bohr from a grid point (the series
    /// branch at x ≈ 3e-3), Z = 1, 2, 3. End to end (f64 inputs) and kernel-only (the f32
    /// inputs the kernel sees: ρ as f32, positions as cell + f32 fraction).
    #[test]
    fn force_kernel_matches_f64_integrand_on_an_analytic_density() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = Grid::cube(33, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let sigma = 0.25;
        let on = cg.point(cg.idx(20, 15, 17));
        let nuclei = [Nucleus { pos: [0.7, 0.3, -0.2], z: 1 }, Nucleus { pos: on, z: 2 },
                      Nucleus { pos: [on[0] - 1.0 + 1e-3, on[1] + 0.5 - 2e-4, on[2] + 5e-4], z: 3 }];
        let rho_cpu = cg.sample(|[x, y, z]| 0.3 * (-((x - 0.2).powi(2) + (y + 0.1).powi(2) + z * z) / 1.5).exp() + 0.05 * (-(x * x + y * y + (z - 0.4).powi(2)) * 2.0).exp());
        let rho = Field::new(&ctx, &g, "rho");
        rho.upload_interior(&ctx, &g, &rho_cpu);
        let gpu = electronic_forces(&ctx, &k, &g, &rho, &nuclei, sigma);
        let cpu = cpu_electronic(&cg, &rho_cpu, &nuclei, sigma);
        // kernel-only reference: the f32-rounded inputs, f64 arithmetic
        let rho32: Vec<f64> = rho_cpu.iter().map(|v| *v as f32 as f64).collect();
        let rounded: Vec<Nucleus> = nuclei.iter().map(|n| {
            let rec = NucleusRecord::new(&g, n);
            Nucleus { pos: std::array::from_fn(|a| cg.origin[a] + (rec.cell[a] as f64 + rec.frac[a] as f64) * cg.h), z: n.z }
        }).collect();
        let cpu32 = cpu_electronic(&cg, &rho32, &rounded, sigma);
        let (mut worst, mut worst32) = (0.0f64, 0.0f64);
        for i in 0..nuclei.len() {
            let scale = cpu[i].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err: Vec<f64> = (0..3).map(|a| (gpu[i][a] - cpu[i][a]).abs()).collect();
            let err32: Vec<f64> = (0..3).map(|a| (gpu[i][a] - cpu32[i][a]).abs()).collect();
            println!("nucleus {i} (Z = {}): gpu {:?}\n             cpu {:?}\n             |Δ| {:.2e} {:.2e} {:.2e} (max|F| {scale:.4}); kernel-only |Δ| {:.2e} {:.2e} {:.2e}",
                     nuclei[i].z, gpu[i], cpu[i], err[0], err[1], err[2], err32[0], err32[1], err32[2]);
            assert!(scale > 1e-2, "test assumption: the force is not tiny");
            worst = worst.max(err.iter().cloned().fold(0.0, f64::max));
            worst32 = worst32.max(err32.iter().cloned().fold(0.0, f64::max));
        }
        println!("analytic density 33³: max |gpu − cpu| = {worst:.2e}, kernel-only {worst32:.2e}");
        // plan bound 1e-3 Ha/Bohr (forces); the kernel alone is at the f32 partial-sum level
        assert!(worst < 2e-6, "max |gpu − cpu| = {worst:.3e}");
        assert!(worst32 < 2e-6, "kernel-only max |gpu − cpu| = {worst32:.3e}");
    }

    fn h2(r: f64) -> System {
        System::new(vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }], 0, None)
    }

    /// A solver on an n³, h = 0.25 cube (σ = h, `ScfSettings::for_grid`).
    fn solver(n: usize) -> Option<(GpuSolver, Grid, ScfSettings)> {
        let ctx = GpuContext::new_headless()?;
        let grid = Grid::cube(n, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        Some((GpuSolver::new(ctx, &grid), grid, settings))
    }

    /// Cold GPU solve + forces, with the force evaluation timed (ms, mean of `reps`, 0 = not timed).
    fn gpu_energy_and_force(s: &mut GpuSolver, sys: &System, settings: &ScfSettings, reps: usize) -> (ScfStats, Vec<[f64; 3]>, f64) {
        let stats = s.solve(sys, settings, false).unwrap().1;
        let state = s.state().unwrap();
        let f = s.forces(state, &sys.nuclei, settings.sigma).unwrap();
        let t = Instant::now();
        for _ in 0..reps { std::hint::black_box(s.forces(state, &sys.nuclei, settings.sigma).unwrap()); }
        let ms = if reps > 0 { t.elapsed().as_secs_f64() * 1e3 / reps as f64 } else { f64::NAN };
        (stats, f, ms)
    }

    /// CPU reference forces F_x0 for H₂ at 49³, h = σ = 0.25, nucleus 1 at −R/2 (`qm::forces`
    /// at SCF tol 1e-9, eig_tol 1e-8; commit 3167095): R = 1.0, 1.4, 2.5. Regenerate with
    /// `cargo test -p qm forces --release -- --ignored --nocapture report_numbers` (the "F0" x
    /// components of the R = 1.0 / 1.4 / 2.5 rows) whenever `qm::scf`/`qm::forces` change.
    const CPU_F_X0: [(f64, f64); 3] = [(1.0, 0.5805010696), (1.4, 0.1380749345), (2.5, -0.0602453917)];

    /// Brief: 33³, R = 1.4, |F_gpu − F_cpu| < 1e-3 per component (achieved value reported),
    /// equal and opposite within 1e-5 (the f32 state is not exactly mirror-symmetric, module
    /// doc). Also the error paths: coincident nuclei, and the timing at 33³.
    #[test]
    fn h2_forces_match_cpu_at_fixed_sigma() {
        let Some((mut s, grid, settings)) = solver(33) else { return };
        let sys = h2(1.4);
        let t = Instant::now();
        let (cpu_state, cpu_stats) = qm::scf(&grid, &sys, &settings, None).unwrap();
        let f_cpu = qm::forces(&grid, &cpu_state, &sys.nuclei, settings.sigma).unwrap();
        let cpu_s = t.elapsed().as_secs_f64();
        let (gpu_stats, f_gpu, ms) = gpu_energy_and_force(&mut s, &sys, &settings, 20);
        println!("H₂ 33³ R = 1.4: GPU E {:+.8} ({} steps, Δρ {:.2e}) F0 {:?} F1 {:?}; CPU E {:+.8} ({} steps, Δρ {:.2e}, {cpu_s:.2} s) F0 {:?} F1 {:?}",
                 gpu_stats.energy.total, gpu_stats.iterations, gpu_stats.delta_rho, f_gpu[0], f_gpu[1],
                 cpu_stats.energy.total, cpu_stats.iterations, cpu_stats.delta_rho, f_cpu[0], f_cpu[1]);
        let mut worst = 0.0f64;
        for i in 0..2 { for a in 0..3 {
            let d = (f_gpu[i][a] - f_cpu[i][a]).abs();
            worst = worst.max(d);
            assert!(d < 1e-3, "F[{i}][{a}]: gpu {} cpu {} (|Δ| = {d:.3e})", f_gpu[i][a], f_cpu[i][a]);
        }}
        let sum: Vec<f64> = (0..3).map(|a| f_gpu[0][a] + f_gpu[1][a]).collect();
        println!("H₂ 33³: max |F_gpu − F_cpu| = {worst:.2e} Ha/Bohr; Σ_I F_I = {sum:?}; off-axis |F0_y|, |F0_z| = {:.2e}, {:.2e}; forces {ms:.3} ms per evaluation",
                 f_gpu[0][1].abs(), f_gpu[0][2].abs());
        // brief 1e-3; achieved 1.9e-6 (both states at tol 1e-6: first order in their Δρ ≈ 5e-7
        // times ∫|∇V| ≈ 4) → tightened to 2e-5
        assert!(worst < 2e-5, "achieved bound: max |F_gpu − F_cpu| = {worst:.3e}");
        for a in 0..3 { assert!(sum[a].abs() < 1e-5, "not equal and opposite: {sum:?}"); }
        assert!(f_gpu[0][1].abs() < 1e-5 && f_gpu[0][2].abs() < 1e-5, "off-axis force {:?}", f_gpu[0]);
        // the nuclear term's coincidence check (the SCF itself refuses such a system)
        let state = s.state().unwrap();
        let coincident = [Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }];
        assert!(matches!(s.forces(state, &coincident, settings.sigma), Err(QmError::CoincidentNuclei(0, 1))));
        // no nuclei: no forces (and no dispatch)
        assert!(s.forces(state, &[], settings.sigma).unwrap().is_empty());
    }

    /// Brief: δ = 0.01, nucleus 1 fixed, |F − FD| < 2e-3. E depends on the nuclei only through
    /// R = x₀ − x₁, so −∂E/∂x₀ is a true partial derivative. The CPU test (commit 3167095, tol
    /// 1e-9) gets HF − FD = −2.71e-5 / −1.08e-4 at δ = 0.01 / 0.02 — pure δ² truncation
    /// (E‴ ≈ −1.6 Ha/Bohr³), Richardson −5.7e-8. On the GPU the SCF is tol 1e-7 (the f32 Δρ
    /// reach) with LOBPCG at its floor, and the *energy* carries a geometry-to-geometry f32
    /// scatter σ_E of a few 1e-7 Ha (module doc), i.e. σ_E/(2δ) ≈ 1–2e-5 in FD(0.01): the same
    /// size as the truncation, so the brief's δ is noise-limited here. The test therefore
    /// (i) pins the GPU HF force against the CPU's HF value (the direct check, free of FD
    /// noise), (ii) keeps the brief's δ = 0.01 assertion, and (iii) Richardson-extrapolates
    /// δ = 0.02/0.04 (noise ≈ 0.75 σ_E/0.02 ≈ 1.5e-5, truncation residual < 1e-6) as the FD
    /// cross-check; δ = 0.01/0.02 is printed for comparison with the CPU. The energy scatter
    /// is measured directly: the same geometry solved cold and warm from a neighbour. Also the
    /// 49³ force timing.
    #[test]
    fn force_matches_finite_difference_of_gpu_energy() {
        let Some((mut s, grid, _)) = solver(49) else { return };
        let mut settings = ScfSettings::for_grid(&grid);
        settings.tol = 1e-7;
        settings.eig_tol = 1e-8; // clamped to 3× the f32 residual floor inside the solver
        let x1 = -0.7;
        let at = |x0: f64| System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [x1, 0.0, 0.0], z: 1 }], 0, None);
        let (st, f, ms) = gpu_energy_and_force(&mut s, &at(0.7), &settings, 20);
        let (hf, e0) = (f[0][0], st.energy.total);
        println!("H₂ 49³ R = 1.4 (tol 1e-7): E {e0:+.10} ({} steps, Δρ {:.2e}, last-step eigen residual {:.2e}), HF F0 {:?}, F1 {:?}; forces {ms:.3} ms per evaluation",
                 st.iterations, st.delta_rho, s.profile().eigen_last_step_residual, f[0], f[1]);
        let cpu_hf = CPU_F_X0[1].1;
        println!("HF: gpu {hf:+.10} cpu {cpu_hf:+.10} (tol 1e-9)  Δ {:+.3e}", hf - cpu_hf);
        let deltas = [0.01, 0.02, 0.04];
        let mut fd = [0.0; 3];
        for (k, d) in deltas.into_iter().enumerate() {
            let t = Instant::now();
            let (sp, _, _) = gpu_energy_and_force(&mut s, &at(0.7 + d), &settings, 0);
            let (sm, _, _) = gpu_energy_and_force(&mut s, &at(0.7 - d), &settings, 0);
            let (ep, em) = (sp.energy.total, sm.energy.total);
            fd[k] = -(ep - em) / (2.0 * d);
            println!("δ = {d:.2}: E+ {ep:+.10} ({} steps)  E− {em:+.10} ({} steps)  FD {:+.8}  HF − FD {:+.3e}  ({:.2} s)",
                     sp.iterations, sm.iterations, fd[k], hf - fd[k], t.elapsed().as_secs_f64());
        }
        let rich = |a: f64, b: f64| (4.0 * a - b) / 3.0; // FD(δ) = F + cδ² ⇒ F = (4 FD(δ) − FD(2δ))/3
        let (r12, r24) = (rich(fd[0], fd[1]), rich(fd[1], fd[2]));
        println!("Richardson (0.01, 0.02) {r12:+.8}  HF − R {:+.3e};  Richardson (0.02, 0.04) {r24:+.8}  HF − R {:+.3e};  HF − FD ratios 0.02/0.01 = {:.2}, 0.04/0.02 = {:.2} (4 for pure δ²)",
                 hf - r12, hf - r24, (hf - fd[1]) / (hf - fd[0]), (hf - fd[2]) / (hf - fd[1]));
        // energy scatter: the same geometry cold (e0) vs warm from the last solve (0.7 − 0.04)
        let e_warm = s.solve(&at(0.7), &settings, true).unwrap().1.energy.total;
        println!("energy path scatter at R = 1.4: cold {e0:+.10} warm-from-neighbour {e_warm:+.10}  Δ {:+.2e} Ha", e_warm - e0);
        assert!((hf - cpu_hf).abs() < 1e-5, "GPU HF {hf} vs CPU HF {cpu_hf}"); // achieved 1.1e-7
        assert!((hf - fd[0]).abs() < 2e-4, "HF {hf} vs FD(0.01) {} (|Δ| = {:.3e})", fd[0], (hf - fd[0]).abs()); // brief 2e-3; budget 2.7e-5 truncation + ≲ 6e-5 noise (3σ)
        assert!((hf - r24).abs() < 1e-4, "HF {hf} vs Richardson(0.02, 0.04) {r24} (|Δ| = {:.3e})", (hf - r24).abs()); // budget ≈ 1.5e-5 rms noise
    }

    /// Brief: R = 1.0 (compressed) → F_x0 > 0, R = 2.5 (stretched) → F_x0 < 0, on the 49³ grid;
    /// and both pinned against the CPU's values (tol 1e-9) — the GPU states are at tol 1e-6.
    #[test]
    fn force_sign_pushes_toward_equilibrium() {
        let Some((mut s, _, settings)) = solver(49) else { return };
        let (_, f_short, _) = gpu_energy_and_force(&mut s, &h2(1.0), &settings, 0);
        let (_, f_long, _) = gpu_energy_and_force(&mut s, &h2(2.5), &settings, 0);
        println!("H₂ 49³: R = 1.0 F0 {:?} (cpu {:+.10}, Δ {:+.2e}); R = 2.5 F0 {:?} (cpu {:+.10}, Δ {:+.2e})",
                 f_short[0], CPU_F_X0[0].1, f_short[0][0] - CPU_F_X0[0].1, f_long[0], CPU_F_X0[2].1, f_long[0][0] - CPU_F_X0[2].1);
        assert!(f_short[0][0] > 0.0, "{f_short:?}");
        assert!(f_long[0][0] < 0.0, "{f_long:?}");
        assert!((f_short[0][0] - CPU_F_X0[0].1).abs() < 2e-5, "R = 1.0: {f_short:?}"); // achieved 1.5e-6
        assert!((f_long[0][0] - CPU_F_X0[2].1).abs() < 2e-5, "R = 2.5: {f_long:?}");  // achieved 1.4e-6
    }

    /// Diagnostic (ignored; `cargo test -p qm-gpu --release -- --ignored --nocapture diag_force_vs_eig_rule`):
    /// the HF force of H₂ 49³ R = 1.4 against the CPU's (tol 1e-9) for the M3 fixed eigen tol
    /// and the Δρ-tied rule, at SCF tol 1e-7 / 1e-6 / 1e-5, cold and warm (0.02 Bohr).
    #[test]
    #[ignore]
    fn diag_force_vs_eig_rule() {
        let Some((mut s, grid, _)) = solver(49) else { return };
        let at = |x0: f64| System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
        for f in [0.0, crate::scf::EIG_TOL_FRACTION] {
            s.set_eig_knobs(f, crate::scf::EIG_TOL_MAX);
            for tol in [1e-7, 1e-6, 1e-5] {
                let mut st = ScfSettings::for_grid(&grid);
                st.tol = tol;
                st.eig_tol = 1e-8;
                for (label, warm_from) in [("cold", None), ("warm", Some(0.68))] {
                    if let Some(x) = warm_from { s.solve(&at(x), &st, false).unwrap(); }
                    let stats = s.solve(&at(0.7), &st, warm_from.is_some()).unwrap().1;
                    let state = s.state().unwrap();
                    let fx = s.forces(state, &at(0.7).nuclei, st.sigma).unwrap()[0][0];
                    println!("F = {f}: tol {tol:.0e} {label}: {} steps, last-step eig res {:.2e}, HF − CPU {:+.3e}", stats.iterations, s.profile().eigen_last_step_residual, fx - CPU_F_X0[1].1);
                }
            }
        }
    }

    /// M4: `forces` runs every MD step, so its partials/staging pair is cached on the solver
    /// (grow-only): two consecutive calls allocate nothing new and agree bit for bit; more
    /// nuclei grow the cache once, fewer reuse it.
    #[test]
    fn forces_buffers_are_reused() {
        let Some((mut s, _, settings)) = solver(33) else { return };
        let sys = h2(1.4);
        s.solve(&sys, &settings, false).unwrap();
        let state = s.state().unwrap();
        assert_eq!(s.force_buffer_allocations(), 0, "nothing is allocated before the first call");
        let f1 = s.forces(state, &sys.nuclei, settings.sigma).unwrap();
        assert_eq!(s.force_buffer_allocations(), 1);
        let f2 = s.forces(state, &sys.nuclei, settings.sigma).unwrap();
        assert_eq!(s.force_buffer_allocations(), 1, "the second call must reuse the cached buffers");
        for (a, b) in f1.iter().zip(&f2) { for c in 0..3 { assert_eq!(a[c].to_bits(), b[c].to_bits(), "{f1:?} vs {f2:?}"); } }
        // three nuclei grow it once; back to two reuses the larger pair, same result
        let three = [sys.nuclei[0], sys.nuclei[1], Nucleus { pos: [0.0, 2.0, 0.0], z: 1 }];
        let f3 = s.forces(state, &three, settings.sigma).unwrap();
        assert_eq!(s.force_buffer_allocations(), 2);
        let f4 = s.forces(state, &sys.nuclei, settings.sigma).unwrap();
        assert_eq!(s.force_buffer_allocations(), 2);
        for (a, b) in f1.iter().zip(&f4) { for c in 0..3 { assert_eq!(a[c].to_bits(), b[c].to_bits()); } }
        // the free function (no cache) gives the same electronic part
        let el = electronic_forces(&s.ctx, &s.k, &s.g, &state.rho, &three, settings.sigma);
        let el_cached = s.electronic_forces_cached(&state.rho, &three, settings.sigma);
        for (a, b) in el.iter().zip(&el_cached) { for c in 0..3 { assert_eq!(a[c].to_bits(), b[c].to_bits()); } }
        println!("forces cache: {} allocations over 5 calls; F(H₂) {f1:?}; F(3 nuclei) {f3:?}", s.force_buffer_allocations());
    }

    #[test]
    #[should_panic(expected = "GpuState was converged with sigma")]
    fn forces_reject_a_state_converged_with_another_sigma() {
        let Some(ctx) = GpuContext::new_headless() else { panic!("GpuState was converged with sigma (no adapter: skipped)") };
        let grid = Grid::cube(17, 0.5);
        let mut s = GpuSolver::new(ctx, &grid);
        let mut settings = ScfSettings::for_grid(&grid);
        settings.tol = 1e-4; settings.eig_tol = 1e-4;
        let sys = System::new(vec![Nucleus { pos: [0.0; 3], z: 1 }], 0, None);
        s.solve(&sys, &settings, false).unwrap();
        let state = s.state().unwrap();
        let _ = s.forces(state, &sys.nuclei, settings.sigma + 0.1);
    }
}
