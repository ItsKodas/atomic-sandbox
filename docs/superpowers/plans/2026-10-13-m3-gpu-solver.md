# M3 — GPU Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the M1 Kohn-Sham solve to wgpu compute so that a 49³ H₂ solve takes well under a second and a 97³ H₂O solve a few seconds, matching the CPU reference at fixed σ to f32-limited tolerances, with the density written straight into the viewer's texture — turning the M2 viewer from "wait six seconds per edit" into an interactive sandbox.

**Architecture:** New crate `qm-gpu` with the same solve contract as `qm::scf` (`System` in, density/energy/stats/forces out) but every grid quantity lives in f32 GPU storage buffers padded by 4 ghost cells per axis. ~12 WGSL compute kernels (stencil, pointwise ops, density accumulate, nuclear potential, PW92 XC, two-level reductions, multipole ghost fill, linear combinations) are orchestrated from Rust; everything small and dense (Rayleigh–Ritz, DIIS coefficients, multipole moments assembly) stays f64 on the CPU, exactly as in M1. Reductions return f32 workgroup partials that are summed in f64 on the CPU (CLAUDE.md rule 4). The CPU `qm` crate is untouched and remains the oracle: every kernel is tested against its CPU counterpart on small grids, and the SCF/forces are compared at **fixed σ** (never through the σ→0 extrapolation, which amplifies f32 noise ×88 — M1 handoff §3). The viewer's sim worker gains a GPU backend that shares the window's `wgpu::Device` and copies the converged density into `Volume3D` with a GPU-side copy.

**Tech Stack:** wgpu 30 (compute pipelines, WGSL), `gpu` crate (`GpuContext`, `Volume3D`), `qm` (types + CPU reference), nalgebra (CPU dense), bytemuck. No new crates beyond `half`-free f32 everywhere.

**Spec:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` §5 (`qm` on GPU, precision), §6 (kernel inventory), §9 (validation). **Handoff notes (binding corrections to the spec):** `docs/superpowers/notes/2026-10-13-m1-handoff.md` — V_nuc is evaluated analytically on the GPU (not via Poisson; σ = h makes the Poisson route 4e-3 Ha wrong); validation vs CPU at fixed σ; multigrid needed at ~101³+.

**Model tags:** **[Fable]** (`claude-fable-5-1`) for Tasks 4–8 and 10 (numerical solvers, SCF, forces, validation — a wrong answer looks plausible); **[Opus]** (`claude-opus-5-5`) for Tasks 1–3 and 9 (kernel plumbing with a CPU oracle, app integration). Fable tasks go through the physics-reviewer checklist. **Fallback policy (user ruling 2026-10-13):** if a Fable dispatch fails for capacity/limits, re-dispatch the same brief on Opus and note it in the ledger.

## Global Constraints

- Grid quantities on the GPU are f32 in `storage` buffers with layout `(i+4) + P*((j+4) + P*(k+4))`, P = n + 8 (4 ghost cells per side, ghosts hold 0 for wavefunctions and the Poisson interior operator). Unpadded copies are produced only for the viewer texture and readback.
- Every reduction (dot, sum, moments, Gram entries) is a two-level kernel: workgroup size 256, one f32 partial per workgroup, partials read back and summed in f64 on the CPU (Neumaier). No single-thread serial sums on the GPU, no atomics for float sums.
- Nuclear potential is analytic (erf via a ≤ 1e-7 rational approximation in WGSL), never via Poisson; `deposit_charge` stays CPU-side for the cold-start density only.
- Small dense algebra (Rayleigh–Ritz ≤ 3k×3k, DIIS bordered system, multipole moment assembly) is f64 on the CPU via nalgebra.
- The CPU `qm` crate is not modified except to expose what tests need (`pub` on already-existing items, documented in the task).
- Tolerances vs the CPU reference at fixed σ (49³, σ = h = 0.25 unless stated): per-kernel ≤ 1e-6 relative on smooth test fields; Poisson potential ≤ 1e-5 abs; LOBPCG eigenvalues ≤ 1e-6 Ha; SCF total energy ≤ 1e-4 Ha; forces ≤ 1e-3 Ha/Bohr. These are *starting* bounds: the implementer reports achieved precision and may **tighten** them, never loosen; a miss is NEEDS_CONTEXT with numbers.
- Timing targets (RX 7900 XTX, release): 49³ H₂ cold solve ≤ 0.5 s, warm ≤ 0.15 s; 97³ H₂O cold ≤ 5 s. Reported, not asserted (hardware-dependent).
- Headless GPU tests follow the M2 convention: skip only when no adapter exists; on this machine they must run.
- Commits end with the dispatching model's `Co-Authored-By` trailer.

## Review Focus

1. **f32 energy drift in the SCF**: with ~1M-point sums the energy must not wander by > 1e-5 Ha between identical solves (determinism) — pinned in Task 6 (two identical solves bit-compare energies).
2. **Padded/unpadded layout confusion**: a kernel reading ghost cells as data, or the texture copy including the padding, produces a shifted or garbage cloud — pinned in Task 1 (layout round-trip through pad/unpad) and Task 9 (viewer texture centre test).
3. **Poisson boundary with f32 moments**: high-l moments of large grids overflow or lose precision in f32 — pinned in Task 4 (moments computed with f64 accumulation of per-workgroup partials; test vs CPU moments to 1e-6 relative at l ≤ 6).
4. **LOBPCG breakdown in f32** (near-dependent subspace vectors, the absolute 1e-12 drop threshold from M1 is meaningless in f32) — pinned in Task 5 (relative threshold 1e-6·‖v‖, test with a deliberately degenerate guess).
5. **Device sharing between the render thread and the sim worker**: wgpu `Device`/`Queue` are `Clone` + `Send`; submissions from two threads are legal but the worker must not block the render thread's `get_current_texture` — pinned in Task 9 (frame-time test with a solve running).

## File Structure

```
crates/qm-gpu/Cargo.toml
crates/qm-gpu/src/lib.rs            # GpuSolver (public API), re-exports
crates/qm-gpu/src/grid.rs           # GpuGrid: padded dims, buffers, pad/unpad, readback
crates/qm-gpu/src/kernels.rs        # pipeline cache: compiles every WGSL module once per device
crates/qm-gpu/src/reduce.rs         # two-level reductions (sum, dot, batched dots), f64 CPU finish
crates/qm-gpu/src/pointwise.rs      # axpy, scale, mul, density accumulate, xc, nuclear potential
crates/qm-gpu/src/stencil.rs        # Laplacian / kinetic on padded buffers
crates/qm-gpu/src/poisson.rs        # CG + multipole ghost fill
crates/qm-gpu/src/eigen.rs          # LOBPCG (block ops on GPU, RR on CPU)
crates/qm-gpu/src/scf.rs            # SCF loop, DIIS, energy, warm state
crates/qm-gpu/src/forces.rs         # Hellmann-Feynman forces
crates/qm-gpu/src/multigrid.rs      # (Task 10) V-cycle preconditioner
crates/qm-gpu/src/shaders/*.wgsl    # one file per kernel family
crates/validate/tests/gpu_vs_cpu.rs # fixed-σ comparison suite + timing table
crates/app/src/sim.rs               # backend enum, GPU density → Volume3D copy
```

---

### Task 1: `qm-gpu` scaffold — padded `GpuGrid`, pipeline cache, two-level reductions, pointwise ops **[Opus]**

**Files:**
- Create: `crates/qm-gpu/Cargo.toml`, `crates/qm-gpu/src/{lib.rs, grid.rs, kernels.rs, reduce.rs, pointwise.rs}`, `crates/qm-gpu/src/shaders/{reduce.wgsl, pointwise.wgsl}`
- Modify: `Cargo.toml` (member), `crates/gpu/src/lib.rs` (derive `Clone` on `GpuContext` — all four wgpu handles are `Clone`)

**Interfaces:**
- Produces:
  ```rust
  // grid.rs
  pub struct GpuGrid { pub n: usize, pub h: f64, pub origin: [f64; 3], pub p: usize /* n + 8 */ }
  impl GpuGrid {
      pub fn new(grid: &qm::Grid) -> GpuGrid;                       // cubic only (asserts n0 == n1 == n2)
      pub fn padded_len(&self) -> usize;                           // p³
      pub fn interior_len(&self) -> usize;                         // n³
      pub fn pidx(&self, i: usize, j: usize, k: usize) -> usize;   // (i+4) + p*((j+4) + p*(k+4))
      pub fn dv(&self) -> f64;
  }
  /// A padded f32 field on the GPU. Ghost cells are always zero after `upload`/`zero`.
  pub struct Field { pub buffer: wgpu::Buffer, pub len: usize /* padded */ }
  impl Field {
      pub fn new(ctx: &GpuContext, g: &GpuGrid, label: &str) -> Field;               // zero-initialised
      pub fn upload_interior(&self, ctx: &GpuContext, g: &GpuGrid, data: &[f64]);    // f64 interior (qm layout) → f32 padded
      pub fn read_interior(&self, ctx: &GpuContext, g: &GpuGrid) -> Vec<f64>;        // padded f32 → f64 interior (qm layout)
      pub fn copy_from(&self, ctx: &GpuContext, src: &Field);
  }
  // kernels.rs
  pub struct Kernels { /* one ComputePipeline per entry point, bind group layouts, workgroup sizes */ }
  impl Kernels { pub fn new(ctx: &GpuContext) -> Kernels; pub fn pipeline(&self, name: &str) -> &wgpu::ComputePipeline; }
  // reduce.rs
  pub struct Reducer { /* partials buffer, staging buffer, sized for the largest padded_len */ }
  impl Reducer {
      pub fn new(ctx: &GpuContext, k: &Kernels, g: &GpuGrid) -> Reducer;
      /// Σ_interior a·b · dv, f32 workgroup partials summed in f64 (Neumaier).
      pub fn dot(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, a: &Field, b: &Field) -> f64;
      pub fn sum(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, a: &Field) -> f64;
      /// Batched: all pairwise dots of `xs` against `ys` in one submission (Gram matrices). Returns row-major.
      pub fn gram(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, xs: &[&Field], ys: &[&Field]) -> Vec<f64>;
  }
  // pointwise.rs — all operate on padded buffers, interior only, ghosts untouched (they stay 0)
  pub struct Params { pub a: f32, pub b: f32, pub p: u32, pub n: u32 }   // uniform for the kernels
  pub fn axpy(ctx, k, g, y: &Field, a: f32, x: &Field);                  // y += a·x
  pub fn scale(ctx, k, g, y: &Field, a: f32);                            // y *= a
  pub fn lincomb(ctx, k, g, out: &Field, terms: &[(f32, &Field)]);       // out = Σ cᵢ xᵢ (≤ 16 terms per dispatch, chained)
  pub fn mul_add_sq(ctx, k, g, rho: &Field, psi: &Field);                // rho += psi² (density accumulate)
  pub fn mul(ctx, k, g, out: &Field, a: &Field, b: &Field);              // out = a·b (V·ψ)
  pub fn zero(ctx, k, g, f: &Field);
  ```
  All functions take `(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, …)`; they record into a fresh encoder and submit (M3 optimises submission batching only if Task 8's timings demand it — note it, don't pre-optimise).

WGSL `reduce.wgsl`: `dot_partials` — workgroup 256, each invocation strides over interior indices (3D decode of a linear interior index → padded index), accumulates `a[p]*b[p]` in a register, tree-reduces in `var<workgroup> shared: array<f32, 256>`, writes `partials[wg_id]`. Number of workgroups = min(ceil(n³/256), 1024) so the partial count is bounded; the CPU sums partials in f64 and multiplies by dv. `sum_partials` likewise. `gram_partials` takes `xs`/`ys` as a binding array of ≤ 8 fields each (or loops over pairs with a pair index uniform — simplest: one dispatch per pair sharing one encoder).

- [ ] **Step 1: Crate + failing tests** (`grid.rs`, `reduce.rs`, `pointwise.rs` test modules)

```rust
// grid.rs tests
#[test] fn pad_unpad_round_trip_and_ghosts_zero() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let cg = qm::Grid::cube(9, 0.5);
    let g = GpuGrid::new(&cg);
    let data: Vec<f64> = (0..cg.len()).map(|i| i as f64 * 0.25 - 10.0).collect();
    let f = Field::new(&ctx, &g, "t");
    f.upload_interior(&ctx, &g, &data);
    let back = f.read_interior(&ctx, &g);
    for (a, b) in data.iter().zip(&back) { assert!((a - b).abs() < 1e-5, "{a} vs {b}"); }
    // ghosts: read the raw padded buffer and check every cell outside [4, 4+n) is exactly 0
    let raw = f.read_padded_raw(&ctx);
    for k in 0..g.p { for j in 0..g.p { for i in 0..g.p {
        let inside = (4..4 + g.n).contains(&i) && (4..4 + g.n).contains(&j) && (4..4 + g.n).contains(&k);
        if !inside { assert_eq!(raw[i + g.p * (j + g.p * k)], 0.0); }
    }}}
}
#[test] fn layout_matches_qm_grid_idx() {
    let cg = qm::Grid::cube(5, 1.0); let g = GpuGrid::new(&cg);
    assert_eq!(g.pidx(0, 0, 0), 4 + 13 * (4 + 13 * 4));
    assert_eq!(g.pidx(1, 2, 3) - g.pidx(0, 2, 3), 1);
    assert_eq!(g.pidx(0, 1, 0) - g.pidx(0, 0, 0), 13);
}
// reduce.rs tests
#[test] fn dot_matches_cpu_grid_dot_to_f32_precision() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let cg = qm::Grid::cube(33, 0.25); let g = GpuGrid::new(&cg);
    let a = cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
    let b = cg.sample(|[x, _, _]| 1.0 + 0.1 * x);
    let k = Kernels::new(&ctx); let r = Reducer::new(&ctx, &k, &g);
    let (fa, fb) = (Field::new(&ctx, &g, "a"), Field::new(&ctx, &g, "b"));
    fa.upload_interior(&ctx, &g, &a); fb.upload_interior(&ctx, &g, &b);
    let gpu = r.dot(&ctx, &k, &g, &fa, &fb);
    let cpu = cg.dot(&a, &b);
    assert!((gpu - cpu).abs() < 1e-6 * cpu.abs().max(1.0), "gpu {gpu} cpu {cpu}");
}
#[test] fn sum_of_ones_is_exact_volume() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let cg = qm::Grid::cube(49, 0.25); let g = GpuGrid::new(&cg);
    let k = Kernels::new(&ctx); let r = Reducer::new(&ctx, &k, &g);
    let f = Field::new(&ctx, &g, "ones"); f.upload_interior(&ctx, &g, &vec![1.0; cg.len()]);
    let s = r.sum(&ctx, &k, &g, &f);
    assert!((s - cg.len() as f64 * cg.dv()).abs() < 1e-9, "{s}");  // partials are exact integers in f32 up to 2^24
}
#[test] fn gram_matches_pairwise_dots() { /* 3×2 fields; compare r.gram to r.dot per pair to 1e-9 relative */ }
// pointwise.rs tests
#[test] fn axpy_scale_lincomb_mul_density_match_cpu() {
    // build three CPU fields on 17³; run each op on GPU; compare interior to the CPU formula to 1e-6 rel; assert ghosts stay 0
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p qm-gpu` → compile errors.
- [ ] **Step 3: Implement** — `Kernels::new` creates every pipeline from `include_str!` WGSL; `Field::upload_interior` builds the padded f32 Vec on the CPU (n ≤ 129 ⇒ ≤ 2.6M floats) and `write_buffer`s it; `read_interior` via staging buffer + `map_async` + `device.poll(PollType::wait_indefinitely())`; `Reducer::dot` dispatches `dot_partials`, copies partials to staging, sums in f64 with Neumaier, multiplies by `g.dv()`. Pointwise kernels: workgroup 256 over interior linear index → padded index; `Params` uniform carries `n`, `p`, scalars.
- [ ] **Step 4: Run** — `cargo test -p qm-gpu` all green on the GPU, no warnings; `cargo clippy -p qm-gpu` clean.
- [ ] **Step 5: Commit** — `feat(qm-gpu): padded fields, pipeline cache, two-level f32→f64 reductions, pointwise kernels`

---

### Task 2: `qm-gpu::stencil` — 8th-order Laplacian / kinetic on padded buffers **[Opus]**

**Files:** Create `crates/qm-gpu/src/stencil.rs`, `crates/qm-gpu/src/shaders/stencil.wgsl`; modify `lib.rs`.

**Interfaces:**
```rust
pub fn laplacian(ctx, k, g, f: &Field, out: &Field);   // out = ∇²f on the interior; reads ghosts (zero) → Dirichlet for free
pub fn kinetic(ctx, k, g, f: &Field, out: &Field);     // out = −½∇²f
```
WGSL: one unbranched kernel — because the buffer is padded by 4, every interior point can read `f[p ± m·stride]` for m = 1..4 along each axis without bounds checks; coefficients `c0..c4` as constants (same values as `qm::stencil::COEFF`, written as f32 literals); `inv_h2` from the uniform. Workgroup (8,8,4) over interior (i, j, k); compute padded index. A shared-memory tiled variant is **not** required for M3 (measure first; note in report).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn laplacian_matches_cpu_stencil() {
    // 33³ h=0.25 Gaussian exp(−r²/2); GPU laplacian vs qm::stencil::laplacian on the CPU; max abs diff over the interior < 1e-5 (f32 on values up to ~3)
}
#[test] fn dirichlet_boundary_via_zero_ghosts() {
    // field = 1 everywhere in the interior; at an interior corner the CPU stencil (Dirichlet) and the GPU (zero ghosts) must agree exactly in the sense of f32 rounding (< 1e-6 relative to |c0|·3/h²)
}
#[test] fn kinetic_is_minus_half_laplacian() { /* exact in f32: t == −0.5·l bitwise after both kernels */ }
#[test] fn ghosts_stay_zero_after_stencil() { /* read_padded_raw(out): all ghost cells 0 */ }
```
- [ ] **Step 2–5:** compile-fail → implement → `cargo test -p qm-gpu stencil` green → commit `feat(qm-gpu): 8th-order Laplacian kernel on padded buffers`. Record the kernel time at 97³ (`wgpu` timestamp queries are optional; wall-clock around `queue.submit` + `poll` is fine for the report).

---

### Task 3: nuclear potential (analytic erf) and PW92 XC kernels **[Opus]**

**Files:** Create `crates/qm-gpu/src/shaders/{nuclear.wgsl, xc.wgsl}`; add `nuclear_potential`, `xc_lsda` to `pointwise.rs` (or a new `physics.rs`); modify `lib.rs`.

**Interfaces:**
```rust
/// V_nuc(r) = −Σ_I Z_I erf(d/(σ√2))/d on the interior, d→0 limit −Z√(2/π)/σ. Nuclei passed as a storage buffer of (x, y, z, Z) f32×4, ≤ 256.
pub fn nuclear_potential(ctx, k, g, out: &Field, nuclei: &[qm::Nucleus], sigma: f64);
/// From (ρ↑, ρ↓): energy density n·ε_xc, v↑, v↓ — the PW92/Dirac formulas of `qm::xc` transcribed to WGSL in f32.
pub fn xc_lsda(ctx, k, g, rho_up: &Field, rho_dn: &Field, exc: &Field, v_up: &Field, v_dn: &Field);
```
WGSL `erf`: Abramowitz–Stegun 7.1.26 (max abs error 1.5e-7) — `t = 1/(1+0.3275911x)`, polynomial, `1 − poly·exp(−x²)`; sign-symmetric. Document the error bound; it is below f32 resolution of the potential near nuclei (|V| ~ 3Z). XC: transcribe `qm::xc::{exchange, correlation, pw92_g, f_zeta}` line by line; N_MIN guard 1e-20 → in f32 use 1e-18; `pow(x, 1/3)` via `pow` with x ≥ 0.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn nuclear_potential_matches_cpu_erf_form() {
    // 33³, two nuclei (Z=1 at (0.7,0.3,−0.2), Z=2 at (−0.7,0,0)), σ = 0.25: GPU vs qm::nuclear::nuclear_potential; max |Δ| < 2e-6·max|V| (erf approx 1.5e-7 × Z/d ≤ 2 ⇒ 3e-7 plus f32)
}
#[test] fn nuclear_potential_on_site_limit() { /* nucleus exactly on a grid point: value = −Z√(2/π)/σ within 1e-5 */ }
#[test] fn xc_matches_cpu_pointwise() {
    // densities ρ↑ = Gaussian·0.6, ρ↓ = Gaussian·0.4 on 17³ plus a zero row; compare exc, v↑, v↓ to qm::xc::lsda_pw92 to 1e-5 relative where |v| > 1e-3, abs 1e-7 elsewhere; zeros stay zero (no NaN)
}
#[test] fn xc_polarised_channel_zero_density() { /* ρ↓ ≡ 0: v↓ finite, exc matches CPU */ }
```
- [ ] **Steps 2–5:** compile-fail → implement → green → commit `feat(qm-gpu): analytic nuclear potential and LSDA PW92 kernels`.

---

### Task 4: `qm-gpu::poisson` — CG with solid-harmonic multipole boundary on the GPU **[Fable]**

**Files:** Create `crates/qm-gpu/src/poisson.rs`, `crates/qm-gpu/src/shaders/poisson.wgsl` (ghost fill + moments partials); modify `lib.rs`.

**Interfaces:**
```rust
pub struct GpuPoisson { /* r, z, p, ap, b fields; R_lm harmonic fields (49 padded Fields, precomputed on CPU, uploaded once); ghost-cell table (index, x, y, z) buffer */ }
pub struct PoissonStats { pub iterations: usize, pub residual: f64 }
impl GpuPoisson {
    pub fn new(ctx, k, g) -> GpuPoisson;
    /// Solves ∇²V = −4πρ with V(ghost) = Σ_{l≤6,m} Q_lm R_lm(r)/|r|^{2l+1}, Q_lm = ∫ρ R_lm. `v` is initial guess + output.
    pub fn solve(&mut self, ctx, k, g, r: &Reducer, rho: &Field, v: &Field, tol: f64, max_iter: usize) -> PoissonStats;
}
```
Port of `qm::poisson` (read `crates/qm/src/poisson.rs` first — the derivation and the CPU loop are the spec). Mapping:
| CPU | GPU |
|---|---|
| `multipole_moments` (49 `grid.dot(rho, R_lm)`) | `Reducer::gram(&[rho], &R_lm[..])` → 49 f64 moments on the CPU |
| ghost values `bc(r)` for every ghost cell (`add_boundary_term`) | kernel `ghost_fill`: for each ghost cell in the 4-deep shell (table precomputed on CPU: padded index + world position), evaluate Σ Q_lm R_lm(r)/|r|^{2l+1} from a 49-float uniform of moments (R_lm evaluated inline in WGSL by the same Racah recursion as the CPU, or table the 49 R_lm values per ghost cell on the CPU once per grid and upload — **choose the table**: ghost cells ≈ 6·4·n² ≈ 2.3e5 at 97³ × 49 floats = 45 MB; acceptable, zero per-solve cost beyond one dot product per cell) |
| `apply_a` = −L_int | `stencil::laplacian` on a field whose ghosts are **zero**, then `scale(−1)` — so the solver keeps TWO views: `v` with ghosts filled (for L_bc) and the interior-only operator; simplest correct scheme: compute `b = 4πρ + L_bc` once per solve by (i) writing ghost values into a scratch field whose interior is 0, (ii) `laplacian` of it (interior output = L_bc), (iii) `axpy` into b; then run CG on interior-only fields exactly as the CPU (`apply_a` never sees non-zero ghosts) |
| `grid.dot` in CG | `Reducer::dot` (one readback per iteration — acceptable for M3; note for later: keep α/β on the GPU) |
| Jacobi `inv_diag` | `scale` by the constant |

- [ ] **Step 1: Failing tests**
```rust
#[test] fn gaussian_potential_matches_cpu_solver() {
    // 33³ h=0.25, off-centre Gaussian charge (centre (1.3,−0.7,0.4), σ=0.5, q=1): run qm::poisson::PoissonSolver on the CPU and GpuPoisson on the GPU to tol 1e-6; max |V_gpu − V_cpu| < 1e-5; iterations within ±5
}
#[test] fn moments_match_cpu_to_f32_precision() { /* compare the 49 GPU moments to CPU ones: rel 1e-6 for |Q| > 1e-6 */ }
#[test] fn hartree_energy_of_gaussian() { /* ½∫ρV vs 1/(2σ√π) within 1e-4 as in qm */ }
#[test] fn warm_start_reduces_iterations() { /* perturbed ρ second solve: fewer iterations */ }
#[test] fn f32_residual_floor_is_documented() { /* request tol 1e-10: solver must stop at its f32 floor (~1e-6..1e-7) and report it, not loop to max_iter; assert iterations < max_iter and residual < 1e-5 */ }
```
- [ ] **Steps 2–5:** compile-fail → implement → `cargo test -p qm-gpu poisson` green → commit `feat(qm-gpu): Jacobi-CG Poisson solver with multipole ghost fill`. Report: iterations and ms at 49³ and 97³; the f32 residual floor measured.

---

### Task 5: `qm-gpu::eigen` — LOBPCG with GPU block operations **[Fable]**

**Files:** Create `crates/qm-gpu/src/eigen.rs`; modify `lib.rs` (and `shaders/pointwise.wgsl` if `lincomb` needs more terms).

**Interfaces:**
```rust
pub struct GpuHamiltonian<'a> { pub v: &'a Field }                      // H = −½∇² + V
pub struct EigenStats { pub iterations: usize, pub max_residual: f64 }
/// Lowest k eigenpairs; `x` holds k Fields (initial guess in, orthonormal result out). Preconditioner: JACOBI_SWEEPS damped Jacobi as in qm::eigen (ω from the stencil symbol).
pub fn lobpcg(ctx, k, g, r: &Reducer, h: &GpuHamiltonian, x: &mut [Field], evals: &mut [f64], work: &mut LobpcgWork, tol: f64, max_iter: usize) -> EigenStats;
pub struct LobpcgWork { /* hx, w, p, hs, scratch Fields sized for 3k */ }
```
Port of `qm::eigen::lobpcg` (read it first; keep the orthogonalised-P variant and the Rayleigh–Ritz on the CPU in f64 via nalgebra). Mapping: `op.apply` → `stencil::kinetic` + `mul`/`axpy`; residual norms and Gram matrices → `Reducer::gram`; `orthonormalise` (MGS) → for each vector: `gram` against kept vectors, `lincomb` to subtract, `dot` for the norm, `scale` — with a **relative** drop threshold `1e-6·‖v‖_initial` (Review Focus 4); Ritz-vector assembly `X_new = S·C` → `lincomb` with the f64 coefficients cast to f32. Preconditioner: `jacobi_precond` as in CPU (D⁻¹ computed on the GPU pointwise from `v` and λ; sweeps via `kinetic`/`axpy`).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn oscillator_spectrum_matches_cpu() {
    // 33³ h=0.25, V=½r², k=4 (1.5, 2.5×3): GPU eigenvalues vs qm::eigen on the same grid, |Δ| < 1e-6 Ha; residual < 1e-6
}
#[test] fn eigenvectors_orthonormal_in_f32() { /* Gram = I within 1e-5 */ }
#[test] fn degenerate_guess_does_not_break() { /* two identical guess vectors + one random: the relative threshold drops one and the solver still returns k converged vectors (reseeding the dropped one with noise) — assert residual < 1e-6 */ }
#[test] fn warm_start_converges_in_one_iteration() { /* like the CPU test */ }
```
- [ ] **Steps 2–5:** compile-fail → implement → green → commit `feat(qm-gpu): LOBPCG with GPU block operations and relative MGS threshold`. Report iterations/ms at 49³ k=2 and 97³ k=5, readbacks per iteration.

---

### Task 6: `qm-gpu::scf` — Kohn-Sham SCF with DIIS, energy, warm state **[Fable]**

**Files:** Create `crates/qm-gpu/src/scf.rs`; modify `lib.rs` (public `GpuSolver`).

**Interfaces:**
```rust
pub struct GpuSolver { /* ctx clone, kernels, reducer, poisson, work fields; cached GpuGrid; warm: Option<GpuState> */ }
pub struct GpuState { pub grid: qm::Grid, pub sigma: f64, pub orbitals_up: Vec<Field>, pub orbitals_dn: Vec<Field>, pub evals_up: Vec<f64>, pub evals_dn: Vec<f64>, pub rho: Field /* total, padded */, pub rho_up: Field, pub rho_dn: Field, pub v_h: Field, pub v_nuc: Field }
impl GpuSolver {
    pub fn new(ctx: GpuContext, grid: &qm::Grid) -> GpuSolver;
    /// Same contract as qm::scf: Ok((state, stats)) or Err(QmError). `warm` reuses self's cached state when grid/counts match.
    pub fn solve(&mut self, system: &qm::System, settings: &qm::ScfSettings, warm: bool) -> Result<(&GpuState, qm::ScfStats), qm::QmError>;
    /// Interior density as f32 in qm layout (for the viewer / tests) — one readback.
    pub fn density_f32(&self, state: &GpuState) -> Vec<f32>;
    /// Copies the interior density into a Volume3D texture on the GPU (no CPU round trip): unpad via a kernel into a staging buffer, then copy_buffer_to_texture.
    pub fn copy_density_to_volume(&self, state: &GpuState, volume: &gpu::Volume3D);
    /// For the viewer's auto-scale: max over the interior (two-level max reduction).
    pub fn density_max(&self, state: &GpuState) -> f32;
}
```
Port of `qm::scf::scf` (read it and its DIIS derivation first). Mapping: density build → `zero` + `mul_add_sq` per orbital; V_eff,σ → `nuclear_potential` (once per geometry) + `poisson.solve` + `xc_lsda` + `lincomb`; LOBPCG per channel (Task 5); Δρ = ∫|ρ_out − ρ_in| → needs an `abs_diff_sum` reduction kernel (add to `reduce.wgsl`); DIIS: residuals as Fields, Gram via `Reducer::gram` (stacked spin channels = two grams summed), bordered solve on CPU, mix via `lincomb`, clamp negatives via a `clamp_min(0)` pointwise kernel; energy: T_s via `kinetic` + `dot` per orbital, E_ext = `dot(rho, v_nuc)`, E_H = ½`dot(rho, v_h)` after a final Poisson solve, E_xc = `sum(exc)`, E_nn on CPU (`qm::nuclear::nuclear_repulsion`). Stats mirror `qm::ScfStats` (reuse the type). Cold-start guess: CPU `deposit_charge` at σ = 1 scaled, uploaded; orbital guesses: the CPU `initial_orbitals` helper (make it `pub(crate)` → `pub` in qm with a doc note) uploaded.

- [ ] **Step 1: Failing tests** (all `--release`)
```rust
#[test] fn hydrogen_matches_cpu_at_fixed_sigma() {
    // 33³ h=0.25 σ=0.25, tol 1e-6: GPU total energy vs qm::scf on the same grid: |ΔE| < 1e-4 Ha (report the achieved value); components each < 2e-4; ∫ρ = 1 within 1e-5
}
#[test] fn h2_matches_cpu_at_fixed_sigma() { /* R = 1.4; same bounds; DIIS steps within ±3 of CPU */ }
#[test] fn solve_is_deterministic() { /* two cold solves bit-identical energy (f32 reduction order is fixed) */ }
#[test] fn warm_start_is_cheaper_and_agrees() { /* displaced H as in CPU: warm.iterations < cold_disp.iterations and |ΔE| < 1e-5 */ }
#[test] fn empty_channel_and_lithium_run() { /* H (n_dn = 0) and Li converge; Li energy vs CPU < 5e-4 */ }
#[test] fn density_copy_to_volume_is_unpadded_and_centred() { /* copy_density_to_volume into a Volume3D(n³); read_back; compare to density_f32; argmax at the centre for H */ }
```
- [ ] **Steps 2–5:** compile-fail → implement → `cargo test -p qm-gpu scf --release` green → commit `feat(qm-gpu): Kohn-Sham SCF with DIIS on the GPU; H, H₂, Li match the CPU reference at fixed σ`. Report: achieved |ΔE| per system, steps, ms per step and per solve at 33³/49³, readbacks per step.

---

### Task 7: `qm-gpu::forces` — Hellmann-Feynman forces on the GPU **[Fable]**

**Files:** Create `crates/qm-gpu/src/forces.rs`, `crates/qm-gpu/src/shaders/forces.wgsl`; modify `lib.rs` (`GpuSolver::forces`).

**Interfaces:**
```rust
impl GpuSolver {
    /// F_I = ∫ρ ∇_r V_I + Σ_J Z_I Z_J (R_I−R_J)/d³, as qm::forces (read crates/qm/src/forces.rs — the derivation is the spec).
    pub fn forces(&self, state: &GpuState, nuclei: &[qm::Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, qm::QmError>;
}
```
Kernel `force_partials`: for nucleus I (uniform: position, Z, σ), each invocation computes ρ[p]·(−Z g′(d)/d)·(r − R_I) for its interior point (g′ as in the CPU, with the d < 1e-8 skip) and the workgroup reduces the three components into three partial arrays; CPU sums partials in f64 ×dv and adds the nuclear term (CPU, `qm::nuclear`-style). One dispatch per nucleus.

- [ ] **Step 1: Failing tests** (`--release`)
```rust
#[test] fn h2_forces_match_cpu_at_fixed_sigma() { /* 33³, R=1.4: |F_gpu − F_cpu| < 1e-3 Ha/Bohr per component (report achieved; expect ~1e-5); equal-and-opposite within 1e-5 */ }
#[test] fn force_matches_finite_difference_of_gpu_energy() { /* δ = 0.01, nucleus 1 fixed (as the CPU test): |F − FD| < 2e-3 */ }
#[test] fn force_sign_pushes_toward_equilibrium() { /* R=1.0 → F_x0 > 0; R=2.5 → < 0 */ }
```
- [ ] **Steps 2–5:** compile-fail → implement → green → commit `feat(qm-gpu): Hellmann-Feynman forces on the GPU`.

---

### Task 8: fixed-σ validation suite and timing table **[Fable]**

**Files:** Create `crates/validate/tests/gpu_vs_cpu.rs`; modify `crates/validate/Cargo.toml` (`qm-gpu`, `gpu` deps), `crates/validate/references.md` (new section), `CLAUDE.md` (run line).

Tests (all `#[cfg_attr(debug_assertions, ignore = "release-only")]`, skip if no adapter):
- `energies_match_at_fixed_sigma`: H, He, H₂ (R = 1.4), LiH (R = 3.0), H₂O (preset geometry) at 49³/σ = 0.25, tol 1e-6: table of E_cpu, E_gpu, ΔE, steps, ms; assert |ΔE| ≤ 1e-4 Ha for H/He/H₂, ≤ 5e-4 for LiH/H₂O (f32 with Z = 3/8 potentials ~ −25 Ha on-site). Report the achieved values and **tighten the asserted bounds to 3× the achieved** in the commit if they are comfortably met (tightening permitted).
- `forces_match_at_fixed_sigma`: H₂ at R = 1.0/1.4/2.5 and H₂O: |ΔF| ≤ 1e-3 per component.
- `h2_curve_shape_matches`: E(R) for R ∈ {1.0, 1.4, 1.8, 2.4, 3.2} on the GPU vs CPU: ΔE − mean(ΔE) ≤ 5e-5 (constant offset allowed; shape must match).
- `timing_table`: prints and writes `crates/validate/last-run-gpu.md` (gitignored): cold/warm ms for H₂ at 33³/49³/65³/97³ and H₂O at 57³/97³; asserts nothing except "finished".
- `references.md`: section "GPU vs CPU at fixed σ" with the achieved numbers and the explanation of why fixed σ (M1 handoff §3).

- [ ] Steps: write tests → run `cargo test -p validate --release gpu_vs_cpu` → fill the table → commit `test(validate): GPU-vs-CPU fixed-σ suite and timing table`.

---

### Task 9: app integration — GPU backend, direct density texture, filterable sampling **[Opus]**

**Files:** Modify `crates/app/src/{sim.rs, gfx.rs, viewer.rs, ui.rs, main.rs}`, `crates/app/Cargo.toml` (`qm-gpu`), `crates/gpu/src/lib.rs` (request `Features::FLOAT32_FILTERABLE` when the adapter supports it; expose `ctx.float32_filterable: bool`), `crates/render/src/volume.rs` + `shaders/volume.wgsl` (when filterable: bind the sampler and use one `textureSampleLevel`; else the manual trilinear path — two pipeline variants chosen at `VolumePass::new`).

Behaviour:
- `SimWorker::spawn(initial, backend: Backend, gpu: Option<GpuContext>)` with `enum Backend { Cpu, Gpu }`; the worker owns a `GpuSolver` when `Gpu`. `Snapshot.density` becomes `enum Density { Cpu(Arc<Vec<f32>>), Gpu(wgpu::Texture) }` — for `Gpu` the worker owns a pool of two `Volume3D`s (ping-pong), copies the converged density into the next one via `copy_density_to_volume`, and sends the texture handle; `density_max` from the GPU reduction; the viewer binds the received texture (`VolumePass::bind`) instead of uploading. `Volume3D` gains `from_texture(texture, dims)` or the worker sends the whole `Volume3D` (it is `Send`).
- UI: backend radio (CPU/GPU) in the panel; switching re-spawns the worker with the current scene; the readout shows backend + solve ms. Default: GPU when an adapter exists.
- Cancellation: the worker checks `try_recv` between SCF steps (not inside kernels) and, if a newer geometry command is queued, abandons the current solve (returns to collect) — the `Busy` bracket closes with no snapshot/error. `INTERACTIVE_TOL` re-measured for the GPU (likely 1e-5 still; document).
- Device sharing: the app passes `ctx.clone()` to the worker; the render thread must never wait on the worker's submissions — measure frame time during a 97³ H₂O solve (Review Focus 5) and record it.

- [ ] **Step 1: Failing tests**
```rust
// sim.rs
#[test] fn gpu_backend_hydrogen_snapshot_matches_cpu_backend() { /* spawn both; compare density_max within 1e-4 and energy within 1e-4 */ }
#[test] fn gpu_density_texture_is_centred_and_unpadded() { /* read back the Volume3D from the snapshot: dims n³, argmax at centre */ }
#[test] fn newer_command_cancels_in_flight_solve() { /* 97³ H₂O on GPU: send SetScene(h2) 100 ms after start; the H₂O snapshot must never arrive; H₂ snapshot arrives */ }
// render volume.rs
#[test] fn filterable_and_manual_paths_agree() { /* if ctx.float32_filterable: render the Gaussian with both pipelines; max pixel diff < 8/255 */ }
```
- [ ] **Steps 2–5:** implement → `cargo test -p app -p render --release` green → run the app (GPU backend, H₂O preset) and record: time from drag release to new cloud; frame time while solving; screenshot `docs/images/viewer_h2o_gpu.png` → commit `feat(app): GPU solver backend with direct density texture, cancellation, and filterable sampling`.

---

### Task 10: solver performance — multigrid Poisson, LOBPCG latency, erf precision **[Fable]** (triggered by Task 8; controller ruling 2026-10-13)

Task 8's table: 49³ H₂ warm 292 ms (target 150), 97³ H₂O cold 14.2 s (target 5), Poisson 44 % of a 97³ H₂ step; H₂O's LOBPCG at k = 5 costs ≈ 9 ms grid-independent + ≈ 5 ns/point per iteration (two-point fit 57³/97³) — latency (sync points + dispatch count), not bandwidth. Scope, in order of expected payoff: **(a)** LOBPCG latency — cut sync points per iteration (CGS/Cholesky-QR orthonormalisation on one measured Gram instead of per-vector MGS readbacks, fused Gram/dots kernels, α/β on the GPU where cheap) with a per-iteration dispatch/readback count before/after; **(b)** the multigrid V-cycle below as the CG preconditioner (H₂ warm → ~0.17 s); **(c)** a ≈1e-9-accurate erf (more A&S terms or a Cody rational) in `nuclear.wgsl` and `forces.wgsl`, then re-run `energies_match_at_fixed_sigma` and re-tighten `E_BOUND_H2O` if H₂O's 1.27e-5 drops (it is the leading candidate for that systematic). Re-bench on a quiet machine first (Task 8 saw contention) and record the baseline. Targets stay as the Global Constraints; report achieved.

**Files:** Create `crates/qm-gpu/src/multigrid.rs`, `shaders/multigrid.wgsl`; modify `poisson.rs` (preconditioner hook).
Design: geometric multigrid on the padded grids (coarsen by 2 with full weighting, prolong trilinear, 2nd-order 7-point Laplacian on coarse levels, damped Jacobi smoother ω = 2/3 (ruled 6/7 after measurement — see multigrid.rs), 2 pre/2 post sweeps, V-cycle to ≤ 9³), used as the CG preconditioner (PCG with MG is standard; the 8th-order operator stays in CG, the 2nd-order in MG). Tests: MG-PCG matches plain CG's solution to 1e-5 and takes ≤ ⅓ the iterations at 65³; timing at 97³.

---

## Self-review

- **Spec coverage:** §6 kernel inventory → Tasks 1–3 (stencil, reductions, density, XC, nuclear deposition replaced by analytic potential per handoff), Task 4 (Poisson + multipole), Task 5 (LOBPCG), Task 6 (SCF/DIIS/energy), Task 7 (forces); §5 precision (f32 grid / f64 small algebra / compensated reductions) → Global Constraints + Task 1; §9 layer 1 (kernel vs CPU) → every task's tests; layer 2 at fixed σ → Task 8 (the extrapolated oracles remain the CPU gate, per handoff); viewer integration → Task 9; multigrid (§6) → Task 10 conditional.
- **Review Focus coverage:** (1) determinism → Task 6 `solve_is_deterministic`; (2) padding → Task 1 round-trip/ghost test, Task 6 `density_copy_to_volume_is_unpadded_and_centred`, Task 9 texture test; (3) f32 moments → Task 4 `moments_match_cpu_to_f32_precision`; (4) LOBPCG f32 breakdown → Task 5 `degenerate_guess_does_not_break`; (5) device sharing → Task 9 frame-time measurement + cancellation test.
- **Type consistency:** `Field`, `GpuGrid`, `Kernels`, `Reducer` (Task 1) used by all later tasks with the `(ctx, k, g, …)` calling convention; `GpuPoisson::solve(ctx, k, g, r, rho, v, tol, max_iter)` used by Task 6; `lobpcg(ctx, k, g, r, h, x, evals, work, tol, max_iter)` used by Task 6; `GpuSolver::{new, solve, density_f32, copy_density_to_volume, density_max, forces}` used by Tasks 8–9; `qm::{System, ScfSettings, ScfStats, QmError, Nucleus, Grid}` reused unchanged.
- **Known risk:** the plan references CPU implementations as the algorithmic spec (ports, with mapping tables) rather than restating ~600 lines of solver code; each Fable task starts by reading the CPU module and its derivation. Per-kernel tolerances are f32-informed estimates; the implementer reports achieved precision and tightens.
