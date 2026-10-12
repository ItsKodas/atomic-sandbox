# M1 — CPU Reference Kohn-Sham Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Rust f64 real-space spin-polarized Kohn-Sham DFT solver that reproduces the LSDA hydrogen atom energy and the H₂ bond length with Hellmann-Feynman forces, serving as the test oracle for the GPU solver (M2) and dynamics (M3).

**Architecture:** Cargo workspace `atomic-sandbox` with crates `units`, `qm`, `validate` (`sim`, `gpu`, `render`, `app` come in later milestones). `qm` is built bottom-up as independently tested modules — grid, Laplacian stencil, Poisson CG, LSDA XC, nuclear charge deposition, LOBPCG eigensolver, SCF loop, forces — each with analytic tests before any physics oracle. Everything is f64 on the CPU; the GPU plan (M2) must match these results.

**Tech Stack:** Rust 2021, `nalgebra` (small dense eigenproblems only), `approx` (float asserts). No physics/chemistry crates.

**Spec:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md`

**Model tags:** each task is tagged **[Fable]** (`claude-fable-5-1`, agent `solver-implementer`) or **[Opus]** (`claude-opus-5-5`). Every Fable task's diff goes to `physics-reviewer` before it lands.

## Global Constraints

- Atomic units everywhere inside `qm` (ħ = mₑ = e = 1; Hartree, Bohr). Conversions only via `units`.
- Per-element data allowed: Z and isotope mass only.
- No physics/chemistry crates. Plumbing crates only: `nalgebra`, `approx`, `rand`, `thiserror`, `libm`.
- All arithmetic in this milestone is f64 on the CPU.
- Energy sums use compensated (Kahan/Neumaier) summation — one private kernel in `qm::grid` is the only place grid quantities are summed; `Grid::integrate` and `Grid::dot` both go through it.
- Grid index layout is `x + nx*(y + ny*z)`; `h` is the uniform spacing; Dirichlet zero boundary (isolated system).
- Never loosen a test tolerance to pass; investigate and document.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (or Opus 5.5 when an Opus agent commits).

## Review Focus

1. **Nucleus off grid centre** — energy must not change by more than the documented egg-box ripple when a nucleus is shifted by h/2. Pinned in Task 9.
2. **Odd electron counts** — H (1 electron) and Li (3) must run spin-polarized without panicking on an empty spin channel. Pinned in Task 8.
3. **Box too small** — a hydrogen atom in a 6-Bohr box must give a *higher* energy than in a 12-Bohr box, never lower (Dirichlet confinement), and the solver must report a warning, not silently return. Pinned in Task 8.
4. **Zero-density regions in XC** — PW92 has n^(−1/3) terms; n = 0 must give ε_xc = v_xc = 0, not NaN. Pinned in Task 6.
5. **Two nuclei at the same position** — `nuclear_repulsion` must return an error, not divide by zero. Pinned in Task 7.

## File Structure

```
Cargo.toml                      # workspace
crates/units/src/lib.rs         # constants, conversions, isotope masses
crates/qm/src/lib.rs            # re-exports
crates/qm/src/grid.rs           # Grid: dims, spacing, coordinates, integrate
crates/qm/src/stencil.rs        # 8th-order Laplacian (Dirichlet)
crates/qm/src/poisson.rs        # CG Poisson solver + multipole boundary
crates/qm/src/xc.rs             # LSDA PW92 exchange-correlation
crates/qm/src/nuclear.rs        # Gaussian nuclear charge, V_nuc, Z·Z/R repulsion
crates/qm/src/eigen.rs          # LOBPCG for lowest k eigenpairs of H
crates/qm/src/scf.rs            # Kohn-Sham SCF loop, total energy
crates/qm/src/forces.rs         # Hellmann-Feynman forces
crates/validate/src/lib.rs      # (empty; tests only)
crates/validate/tests/oracles.rs
crates/validate/references.md   # cited reference values
```

---

### Task 1: Workspace scaffold **[Opus]**

**Files:**
- Create: `Cargo.toml`, `crates/units/Cargo.toml`, `crates/units/src/lib.rs`, `crates/qm/Cargo.toml`, `crates/qm/src/lib.rs`, `crates/validate/Cargo.toml`, `crates/validate/src/lib.rs`, `rust-toolchain.toml`

**Interfaces:**
- Produces: workspace that builds with `cargo build --workspace`; crate names `units`, `qm`, `validate`.

- [ ] **Step 1: Write the workspace manifest**

```toml
# Cargo.toml
[workspace]
resolver = "2"
members = ["crates/units", "crates/qm", "crates/validate"]

[workspace.package]
edition = "2021"
version = "0.1.0"
license = "MIT"
repository = "https://github.com/ItsKodas/atomic-sandbox"

[workspace.dependencies]
nalgebra = "0.33"
approx = "0.5"
rand = "0.8"
thiserror = "1"

[profile.release]
debug = 1
```

```toml
# rust-toolchain.toml
[toolchain]
channel = "stable"
```

- [ ] **Step 2: Create the three crates**

```toml
# crates/units/Cargo.toml
[package]
name = "units"
edition.workspace = true
version.workspace = true
[dependencies]
```

```toml
# crates/qm/Cargo.toml
[package]
name = "qm"
edition.workspace = true
version.workspace = true
[dependencies]
units = { path = "../units" }
nalgebra = { workspace = true }
thiserror = { workspace = true }
[dev-dependencies]
approx = { workspace = true }
```

```toml
# crates/validate/Cargo.toml
[package]
name = "validate"
edition.workspace = true
version.workspace = true
[dependencies]
units = { path = "../units" }
qm = { path = "../qm" }
[dev-dependencies]
approx = { workspace = true }
```

Each `src/lib.rs` starts as:

```rust
//! See docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md
```

- [ ] **Step 3: Verify it builds**

Run: `cargo build --workspace`
Expected: `Finished` with no warnings.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml rust-toolchain.toml crates
git commit -m "chore: scaffold atomic-sandbox workspace (units, qm, validate)"
```

---

### Task 2: `units` — constants and conversions **[Opus]**

**Files:**
- Create: `crates/units/src/lib.rs` (replace stub)
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Produces:
  - `pub const BOHR_IN_ANGSTROM: f64`, `HARTREE_IN_EV: f64`, `AU_TIME_IN_FS: f64`, `KB_HARTREE_PER_K: f64`, `AMU_IN_ME: f64`
  - `pub fn bohr_to_angstrom(x: f64) -> f64`, `angstrom_to_bohr`, `hartree_to_ev`, `ev_to_hartree`, `au_time_to_fs`, `fs_to_au_time`, `kelvin_to_hartree`, `hartree_to_kelvin`, `amu_to_au_mass`
  - `pub fn isotope_mass_amu(z: u8) -> Option<f64>` — most-abundant isotope mass for Z = 1..=10.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bohr_angstrom_roundtrip() {
        assert_relative_eq!(bohr_to_angstrom(1.0), 0.529177210903, epsilon = 1e-12);
        assert_relative_eq!(angstrom_to_bohr(bohr_to_angstrom(2.5)), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn hartree_ev() {
        assert_relative_eq!(hartree_to_ev(1.0), 27.211386245988, epsilon = 1e-9);
        assert_relative_eq!(ev_to_hartree(13.605693122994), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn time_and_temperature() {
        assert_relative_eq!(au_time_to_fs(1.0), 0.024188843265857, epsilon = 1e-15);
        assert_relative_eq!(fs_to_au_time(0.25), 10.335, epsilon = 1e-3);
        assert_relative_eq!(kelvin_to_hartree(300.0), 9.5004e-4, epsilon = 1e-7);
    }

    #[test]
    fn masses() {
        assert_relative_eq!(isotope_mass_amu(1).unwrap(), 1.00782503223, epsilon = 1e-10);
        assert_relative_eq!(isotope_mass_amu(8).unwrap(), 15.99491461957, epsilon = 1e-10);
        assert!(isotope_mass_amu(0).is_none());
        assert_relative_eq!(amu_to_au_mass(1.0), 1822.888486209, epsilon = 1e-6);
    }
}
```

Add `approx = { workspace = true }` under `[dev-dependencies]` in `crates/units/Cargo.toml`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p units`
Expected: compile errors — functions not defined.

- [ ] **Step 3: Implement**

```rust
//! Atomic units are the internal unit system (ħ = mₑ = e = 1).
//! Constants: CODATA 2018. Isotope masses: AME2020, most abundant isotope.

pub const BOHR_IN_ANGSTROM: f64 = 0.529177210903;
pub const HARTREE_IN_EV: f64 = 27.211386245988;
pub const AU_TIME_IN_FS: f64 = 0.024188843265857;
pub const KB_HARTREE_PER_K: f64 = 3.166811563e-6;
pub const AMU_IN_ME: f64 = 1822.888486209;

pub fn bohr_to_angstrom(x: f64) -> f64 { x * BOHR_IN_ANGSTROM }
pub fn angstrom_to_bohr(x: f64) -> f64 { x / BOHR_IN_ANGSTROM }
pub fn hartree_to_ev(e: f64) -> f64 { e * HARTREE_IN_EV }
pub fn ev_to_hartree(e: f64) -> f64 { e / HARTREE_IN_EV }
pub fn au_time_to_fs(t: f64) -> f64 { t * AU_TIME_IN_FS }
pub fn fs_to_au_time(t: f64) -> f64 { t / AU_TIME_IN_FS }
pub fn kelvin_to_hartree(t: f64) -> f64 { t * KB_HARTREE_PER_K }
pub fn hartree_to_kelvin(e: f64) -> f64 { e / KB_HARTREE_PER_K }
pub fn amu_to_au_mass(m: f64) -> f64 { m * AMU_IN_ME }

/// Most-abundant-isotope mass in amu (AME2020). Z = 1..=10 for M1; extend as elements are validated.
pub fn isotope_mass_amu(z: u8) -> Option<f64> {
    Some(match z {
        1 => 1.00782503223,
        2 => 4.00260325413,
        3 => 7.0160034366,
        4 => 9.012183065,
        5 => 11.00930536,
        6 => 12.0,
        7 => 14.00307400443,
        8 => 15.99491461957,
        9 => 18.99840316273,
        10 => 19.9924401762,
        _ => return None,
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p units`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/units
git commit -m "feat(units): atomic-unit constants, conversions, isotope masses Z=1..10"
```

---

### Task 3: `qm::grid` — real-space grid and compensated integration **[Opus]**

**Files:**
- Create: `crates/qm/src/grid.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod grid;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct Grid { pub n: [usize; 3], pub h: f64, pub origin: [f64; 3] }
  impl Grid {
      pub fn cube(n: usize, h: f64) -> Grid;            // centred on the origin
      pub fn len(&self) -> usize;                        // n0*n1*n2
      pub fn idx(&self, i: usize, j: usize, k: usize) -> usize; // i + n0*(j + n1*k)
      pub fn point(&self, idx: usize) -> [f64; 3];
      pub fn dv(&self) -> f64;                           // h^3
      pub fn integrate(&self, f: &[f64]) -> f64;         // Σ f · h^3, Neumaier-compensated
      pub fn dot(&self, a: &[f64], b: &[f64]) -> f64;    // ∫ a b
      pub fn sample(&self, f: impl Fn([f64; 3]) -> f64) -> Vec<f64>;
  }
  ```

- [ ] **Step 1: Write failing tests** (`#[cfg(test)]` at the bottom of `grid.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn cube_is_centred() {
        let g = Grid::cube(5, 0.5);
        assert_eq!(g.len(), 125);
        let c = g.point(g.idx(2, 2, 2));
        assert_relative_eq!(c[0], 0.0); assert_relative_eq!(c[1], 0.0); assert_relative_eq!(c[2], 0.0);
        let p = g.point(g.idx(0, 0, 0));
        assert_relative_eq!(p[0], -1.0);
    }

    #[test]
    fn integrates_gaussian() {
        // ∫ exp(-r²) d³r = π^{3/2}; uniform grids are spectrally accurate for smooth decaying f.
        let g = Grid::cube(49, 0.25); // 12 Bohr box
        let f = g.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        assert_relative_eq!(g.integrate(&f), PI.powf(1.5), epsilon = 1e-9);
    }

    #[test]
    fn integrate_is_compensated() {
        let g = Grid { n: [3, 1, 1], h: 1.0, origin: [0.0; 3] };
        assert_eq!(g.integrate(&[1e16, 1.0, -1e16]), 1.0);
    }

    #[test]
    fn dot_matches_integrate() {
        let g = Grid::cube(9, 0.5);
        let a = g.sample(|[x, _, _]| x);
        let b = g.sample(|[x, _, _]| x);
        let prod: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p * q).collect();
        assert_relative_eq!(g.dot(&a, &b), g.integrate(&prod), epsilon = 1e-14);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm grid`
Expected: compile error — `Grid` not defined.

- [ ] **Step 3: Implement**

```rust
//! Uniform real-space grid. Layout: idx = i + n0*(j + n1*k). Lengths in Bohr.

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub n: [usize; 3],
    pub h: f64,
    pub origin: [f64; 3],
}

impl Grid {
    pub fn cube(n: usize, h: f64) -> Grid {
        let half = (n as f64 - 1.0) * h / 2.0;
        Grid { n: [n; 3], h, origin: [-half; 3] }
    }
    pub fn len(&self) -> usize { self.n[0] * self.n[1] * self.n[2] }
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize { i + self.n[0] * (j + self.n[1] * k) }
    pub fn point(&self, idx: usize) -> [f64; 3] {
        let i = idx % self.n[0];
        let j = (idx / self.n[0]) % self.n[1];
        let k = idx / (self.n[0] * self.n[1]);
        [self.origin[0] + i as f64 * self.h,
         self.origin[1] + j as f64 * self.h,
         self.origin[2] + k as f64 * self.h]
    }
    pub fn dv(&self) -> f64 { self.h * self.h * self.h }

    /// Neumaier-compensated sum times the volume element. The only place grid data is summed.
    pub fn integrate(&self, f: &[f64]) -> f64 {
        debug_assert_eq!(f.len(), self.len());
        let (mut sum, mut comp) = (0.0f64, 0.0f64);
        for &x in f {
            let t = sum + x;
            if sum.abs() >= x.abs() { comp += (sum - t) + x; } else { comp += (x - t) + sum; }
            sum = t;
        }
        (sum + comp) * self.dv()
    }

    pub fn dot(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let (mut sum, mut comp) = (0.0f64, 0.0f64);
        for (&p, &q) in a.iter().zip(b) {
            let x = p * q;
            let t = sum + x;
            if sum.abs() >= x.abs() { comp += (sum - t) + x; } else { comp += (x - t) + sum; }
            sum = t;
        }
        (sum + comp) * self.dv()
    }

    pub fn sample(&self, f: impl Fn([f64; 3]) -> f64) -> Vec<f64> {
        (0..self.len()).map(|i| f(self.point(i))).collect()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm grid`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): Grid with compensated integration"
```

---

### Task 4: `qm::stencil` — 8th-order Laplacian with Dirichlet boundary **[Fable]**

**Files:**
- Create: `crates/qm/src/stencil.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod stencil;`)

**Interfaces:**
- Consumes: `grid::Grid`.
- Produces:
  ```rust
  pub const COEFF: [f64; 5];  // central 8th-order second-derivative coefficients c0..c4
  /// out = ∇²f with f = 0 outside the box (Dirichlet). out.len() == f.len() == grid.len().
  pub fn laplacian(grid: &Grid, f: &[f64], out: &mut [f64]);
  /// out = -½ ∇² f  (kinetic-energy operator in atomic units)
  pub fn kinetic(grid: &Grid, f: &[f64], out: &mut [f64]);
  ```

Derivation (write this as the module doc comment): the 8th-order central approximation to d²f/dx² on a uniform grid is
(1/h²) Σ_{m=-4}^{4} c_{|m|} f_{i+m} with c₀ = −205/72, c₁ = 8/5, c₂ = −1/5, c₃ = 8/315, c₄ = −1/560 (Fornberg 1988). The 3D Laplacian is the sum over the three axes; the centre coefficient therefore appears three times.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    fn max_interior_err(n: usize, h: f64) -> f64 {
        // f = exp(-r²/2)  ⇒  ∇²f = (r² − 3) f
        let g = Grid::cube(n, h);
        let f = g.sample(|[x, y, z]| (-(x * x + y * y + z * z) / 2.0).exp());
        let exact = g.sample(|[x, y, z]| { let r2 = x * x + y * y + z * z; (r2 - 3.0) * (-r2 / 2.0).exp() });
        let mut out = vec![0.0; g.len()];
        laplacian(&g, &f, &mut out);
        // skip 4 cells at each face where the Dirichlet ghost (0) differs from the true Gaussian tail
        let mut err: f64 = 0.0;
        for k in 4..n - 4 { for j in 4..n - 4 { for i in 4..n - 4 {
            let id = g.idx(i, j, k);
            err = err.max((out[id] - exact[id]).abs());
        }}}
        err
    }

    #[test]
    fn coefficients_sum_to_zero() {
        // A constant field has zero Laplacian: c0 + 2(c1+c2+c3+c4) = 0
        let s = COEFF[0] + 2.0 * (COEFF[1] + COEFF[2] + COEFF[3] + COEFF[4]);
        assert!(s.abs() < 1e-15, "sum = {s}");
    }

    #[test]
    fn laplacian_of_gaussian() {
        assert!(max_interior_err(49, 0.25) < 1e-4);
    }

    #[test]
    fn converges_at_eighth_order() {
        let coarse = max_interior_err(25, 0.5);
        let fine = max_interior_err(49, 0.25);
        // 8th order predicts 256×; demand at least 64× to allow for the error constant
        assert!(coarse / fine > 64.0, "ratio = {}", coarse / fine);
    }

    #[test]
    fn kinetic_is_minus_half_laplacian() {
        let g = Grid::cube(9, 0.5);
        let f = g.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let (mut l, mut t) = (vec![0.0; g.len()], vec![0.0; g.len()]);
        laplacian(&g, &f, &mut l);
        kinetic(&g, &f, &mut t);
        for i in 0..g.len() { assert!((t[i] + 0.5 * l[i]).abs() < 1e-15); }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm stencil`
Expected: compile error — `laplacian` not defined.

- [ ] **Step 3: Implement**

```rust
//! 8th-order central finite-difference Laplacian on a uniform grid, Dirichlet (f = 0) outside the box.
//! d²f/dx² ≈ (1/h²) Σ_{m=-4}^{4} c_{|m|} f_{i+m},  c = [−205/72, 8/5, −1/5, 8/315, −1/560] (Fornberg 1988).
//! ∇² = Σ over the three axes.

use crate::grid::Grid;

pub const COEFF: [f64; 5] = [-205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0];

pub fn laplacian(grid: &Grid, f: &[f64], out: &mut [f64]) {
    debug_assert_eq!(f.len(), grid.len());
    debug_assert_eq!(out.len(), grid.len());
    let [n0, n1, n2] = grid.n;
    let inv_h2 = 1.0 / (grid.h * grid.h);
    let strides = [1usize, n0, n0 * n1];
    for k in 0..n2 {
        for j in 0..n1 {
            for i in 0..n0 {
                let id = grid.idx(i, j, k);
                let mut acc = 3.0 * COEFF[0] * f[id];
                let pos = [i, j, k];
                for axis in 0..3 {
                    for m in 1..=4usize {
                        let c = COEFF[m];
                        if pos[axis] + m < grid.n[axis] { acc += c * f[id + m * strides[axis]]; }
                        if pos[axis] >= m { acc += c * f[id - m * strides[axis]]; }
                    }
                }
                out[id] = acc * inv_h2;
            }
        }
    }
}

pub fn kinetic(grid: &Grid, f: &[f64], out: &mut [f64]) {
    laplacian(grid, f, out);
    for v in out.iter_mut() { *v *= -0.5; }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm stencil`
Expected: 4 passed. Record the observed convergence ratio in the commit message.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): 8th-order Laplacian stencil with Dirichlet boundary"
```

---

### Task 5: `qm::poisson` — conjugate-gradient Poisson solver with multipole boundary **[Fable]**

**Files:**
- Create: `crates/qm/src/poisson.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod poisson;`), `crates/qm/Cargo.toml` (add `libm = "0.2"` to `[dependencies]`; add `libm = "0.2"` to `[workspace.dependencies]` in the root `Cargo.toml`)

**Interfaces:**
- Consumes: `grid::Grid`, `stencil::COEFF`.
- Produces:
  ```rust
  pub struct PoissonSolver { /* preallocated work vectors */ }
  pub struct PoissonStats { pub iterations: usize, pub residual: f64 }
  impl PoissonSolver {
      pub fn new(grid: &Grid) -> PoissonSolver;
      /// Solve ∇²V = −4π ρ on the grid with V(r) outside the box given by the real solid-harmonic
      /// multipole expansion of ρ to l_max = 6: V_bc(r) = Σ_{l≤6,m} Q_lm R_lm(r)/|r|^{2l+1}.
      /// `v` is both the initial guess and the output. Converges to ‖r‖/‖b‖ < tol.
      pub fn solve(&mut self, grid: &Grid, rho: &[f64], v: &mut [f64], tol: f64, max_iter: usize) -> PoissonStats;
  }
  /// ½ ∫ ρ V
  pub fn hartree_energy(grid: &Grid, rho: &[f64], v: &[f64]) -> f64;
  ```

Derivation (module doc comment): in atomic units ∇²V = −4πρ. Split the stencil into an interior part L_int (ghost cells = 0) and a boundary part L_bc (contribution of the known ghost values, independent of V). Then −L_int V = 4πρ + L_bc, and A = −L_int is symmetric positive-definite, so conjugate gradient applies. Ghost values come from the multipole expansion of ρ in real solid harmonics to l_max = 6 (49 moments Q_lm = ∫ρ R_lm, computed with `Grid::integrate`); at l ≤ 1 this reduces to Q = ∫ρ and p = ∫ρ r. Monopole + dipole alone leaves an O((d/r)²) quadrupole error of ~1e-2 for charges 1–2 Bohr off centre in a 12-Bohr box, which the off-centre test must catch; l_max = 6 brings it to ~4e-5. Jacobi preconditioner: diag(A) = −3c₀/h² (a no-op on a uniform grid, kept so the GPU path has the hook).

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// Normalised Gaussian charge q, width σ: ρ = q (2πσ²)^{-3/2} exp(−r²/2σ²); V = q erf(r/(σ√2))/r.
    fn gaussian_case(n: usize, h: f64, q: f64, sigma: f64, centre: [f64; 3]) -> (Grid, Vec<f64>, Vec<f64>) {
        let g = Grid::cube(n, h);
        let norm = q / (2.0 * PI * sigma * sigma).powf(1.5);
        let rho = g.sample(|[x, y, z]| {
            let r2 = (x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2);
            norm * (-r2 / (2.0 * sigma * sigma)).exp()
        });
        let exact = g.sample(|[x, y, z]| {
            let r = ((x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2)).sqrt();
            if r < 1e-12 { q * (2.0 / PI).sqrt() / sigma } else { q * libm::erf(r / (sigma * 2f64.sqrt())) / r }
        });
        (g, rho, exact)
    }

    fn max_err(n: usize, h: f64, centre: [f64; 3]) -> f64 {
        let (g, rho, exact) = gaussian_case(n, h, 1.0, 0.5, centre);
        let mut v = vec![0.0; g.len()];
        let stats = PoissonSolver::new(&g).solve(&g, &rho, &mut v, 1e-10, 2000);
        assert!(stats.residual < 1e-10, "did not converge: {:?}", stats.residual);
        v.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn gaussian_potential_centred() {
        assert!(max_err(49, 0.25, [0.0; 3]) < 1e-4);
    }

    #[test]
    fn gaussian_potential_off_centre_uses_dipole_boundary() {
        assert!(max_err(49, 0.25, [1.3, -0.7, 0.4]) < 1e-4);
    }

    #[test]
    fn converges_with_h() {
        let coarse = max_err(25, 0.5, [0.0; 3]);
        let fine = max_err(49, 0.25, [0.0; 3]);
        assert!(coarse / fine > 30.0, "ratio = {}", coarse / fine);
    }

    #[test]
    fn hartree_energy_of_gaussian() {
        // E_H = q² / (2 σ √π)
        let (g, rho, _) = gaussian_case(49, 0.25, 1.0, 0.5, [0.0; 3]);
        let mut v = vec![0.0; g.len()];
        PoissonSolver::new(&g).solve(&g, &rho, &mut v, 1e-10, 2000);
        assert_relative_eq!(hartree_energy(&g, &rho, &v), 1.0 / (2.0 * 0.5 * PI.sqrt()), epsilon = 1e-4);
    }

    #[test]
    fn warm_start_takes_fewer_iterations() {
        let (g, rho, _) = gaussian_case(33, 0.25, 1.0, 0.5, [0.0; 3]);
        let mut solver = PoissonSolver::new(&g);
        let mut v = vec![0.0; g.len()];
        let cold = solver.solve(&g, &rho, &mut v, 1e-10, 2000);
        let warm = solver.solve(&g, &rho, &mut v, 1e-10, 2000);
        assert!(warm.iterations < cold.iterations / 4, "cold {} warm {}", cold.iterations, warm.iterations);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm poisson`
Expected: compile error — `PoissonSolver` not defined.

- [ ] **Step 3: Implement**

```rust
//! Poisson solver: ∇²V = −4πρ (atomic units), isolated boundary conditions.
//! −L_int V = 4πρ + L_bc, with ghost values from the multipole expansion V_bc(r) = Q/|r| + p·r/|r|³.
//! A = −L_int is SPD ⇒ Jacobi-preconditioned conjugate gradient.

use crate::grid::Grid;
use crate::stencil::COEFF;

pub struct PoissonSolver { r: Vec<f64>, z: Vec<f64>, p: Vec<f64>, ap: Vec<f64>, b: Vec<f64> }

#[derive(Debug, Clone, Copy)]
pub struct PoissonStats { pub iterations: usize, pub residual: f64 }

impl PoissonSolver {
    pub fn new(grid: &Grid) -> PoissonSolver {
        let n = grid.len();
        PoissonSolver { r: vec![0.0; n], z: vec![0.0; n], p: vec![0.0; n], ap: vec![0.0; n], b: vec![0.0; n] }
    }

    /// y = A x = −L_int x  (ghost cells treated as zero)
    fn apply_a(grid: &Grid, x: &[f64], y: &mut [f64]) {
        crate::stencil::laplacian(grid, x, y);
        for v in y.iter_mut() { *v = -*v; }
    }

    /// Adds L_bc (contribution of ghost values) to `out`. Ghost value at point r is bc(r).
    fn add_boundary_term(grid: &Grid, bc: &dyn Fn([f64; 3]) -> f64, out: &mut [f64]) {
        let [n0, n1, n2] = grid.n;
        let inv_h2 = 1.0 / (grid.h * grid.h);
        for k in 0..n2 { for j in 0..n1 { for i in 0..n0 {
            let pos = [i as isize, j as isize, k as isize];
            let mut acc = 0.0;
            for axis in 0..3 {
                for m in 1..=4isize {
                    for sign in [-1isize, 1] {
                        let mut q = pos; q[axis] += sign * m;
                        if q[axis] < 0 || q[axis] >= grid.n[axis] as isize {
                            let mut r = grid.point(grid.idx(i, j, k));
                            r[axis] += (sign * m) as f64 * grid.h;
                            acc += COEFF[m as usize] * bc(r);
                        }
                    }
                }
            }
            out[grid.idx(i, j, k)] += acc * inv_h2;
        }}}
    }

    pub fn solve(&mut self, grid: &Grid, rho: &[f64], v: &mut [f64], tol: f64, max_iter: usize) -> PoissonStats {
        let n = grid.len();
        // multipole moments
        let q = grid.integrate(rho);
        let mut p = [0.0f64; 3];
        for axis in 0..3 {
            let w: Vec<f64> = (0..n).map(|i| rho[i] * grid.point(i)[axis]).collect();
            p[axis] = grid.integrate(&w);
        }
        let bc = move |r: [f64; 3]| {
            let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            q / d + (p[0] * r[0] + p[1] * r[1] + p[2] * r[2]) / (d * d * d)
        };
        // b = 4πρ + L_bc
        for i in 0..n { self.b[i] = 4.0 * std::f64::consts::PI * rho[i]; }
        Self::add_boundary_term(grid, &bc, &mut self.b);
        // r = b − A v
        Self::apply_a(grid, v, &mut self.ap);
        for i in 0..n { self.r[i] = self.b[i] - self.ap[i]; }
        let inv_diag = -(grid.h * grid.h) / (3.0 * COEFF[0]);
        let b_norm = grid.dot(&self.b, &self.b).sqrt().max(1e-300);
        for i in 0..n { self.z[i] = inv_diag * self.r[i]; }
        self.p.copy_from_slice(&self.z);
        let mut rz = grid.dot(&self.r, &self.z);
        let mut iterations = 0;
        let mut residual = grid.dot(&self.r, &self.r).sqrt() / b_norm;
        while residual > tol && iterations < max_iter {
            Self::apply_a(grid, &self.p, &mut self.ap);
            let alpha = rz / grid.dot(&self.p, &self.ap);
            for i in 0..n { v[i] += alpha * self.p[i]; self.r[i] -= alpha * self.ap[i]; }
            for i in 0..n { self.z[i] = inv_diag * self.r[i]; }
            let rz_new = grid.dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..n { self.p[i] = self.z[i] + beta * self.p[i]; }
            residual = grid.dot(&self.r, &self.r).sqrt() / b_norm;
            iterations += 1;
        }
        PoissonStats { iterations, residual }
    }
}

pub fn hartree_energy(grid: &Grid, rho: &[f64], v: &[f64]) -> f64 { 0.5 * grid.dot(rho, v) }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm poisson`
Expected: 5 passed. Record iteration counts (cold vs warm) in the commit message.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/qm
git commit -m "feat(qm): Jacobi-PCG Poisson solver with multipole boundary conditions"
```

---

### Task 6: `qm::xc` — LSDA exchange-correlation (Perdew–Wang 92) **[Fable]**

**Files:**
- Create: `crates/qm/src/xc.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod xc;`)

**Interfaces:**
- Produces:
  ```rust
  /// Per grid point: XC energy density n·ε_xc, and potentials v_xc,↑ and v_xc,↓.
  pub fn exc_point(n_up: f64, n_dn: f64) -> (f64, f64, f64);
  pub struct XcResult { pub energy_density: Vec<f64>, pub v_up: Vec<f64>, pub v_dn: Vec<f64> }
  pub fn lsda_pw92(n_up: &[f64], n_dn: &[f64]) -> XcResult;
  ```

Derivation (module doc comment):
- Exchange (Dirac, spin-scaled): E_x[n↑,n↓] = ½(E_x[2n↑] + E_x[2n↓]), E_x[n] = −C∫n^{4/3}, C = (3/4)(3/π)^{1/3}. So the energy density is −C·2^{1/3}(n↑^{4/3} + n↓^{4/3}) and v_x,σ = −(6n_σ/π)^{1/3}.
- Correlation (PW92, Phys. Rev. B 45, 13244): with r_s = (3/4πn)^{1/3}, ζ = (n↑−n↓)/n,
  G(r_s; A, α₁, β₁..β₄) = −2A(1+α₁r_s) ln[1 + 1/(2A(β₁r_s^{1/2} + β₂r_s + β₃r_s^{3/2} + β₄r_s²))],
  ε_c(r_s,ζ) = ε_c(r_s,0) + α_c(r_s) f(ζ)/f″(0) (1−ζ⁴) + [ε_c(r_s,1) − ε_c(r_s,0)] f(ζ) ζ⁴,
  f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2]/(2^{4/3} − 2), f″(0) = 1.709921, and the fit gives −α_c.
  v_c,σ = ε_c − (r_s/3) ∂ε_c/∂r_s − (ζ − s_σ) ∂ε_c/∂ζ with s↑ = +1, s↓ = −1.
  dG/dr_s = −2Aα₁ ln(1+1/Q₁) − Q₀Q₁′/(Q₁² + Q₁) where Q₀ = −2A(1+α₁r_s), Q₁ = 2A(β₁r_s^{1/2} + β₂r_s + β₃r_s^{3/2} + β₄r_s²), Q₁′ = A(β₁r_s^{−1/2} + 2β₂ + 3β₃r_s^{1/2} + 4β₄r_s).

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn energy(nu: f64, nd: f64) -> f64 { exc_point(nu, nd).0 }

    #[test]
    fn exchange_unpolarized_is_dirac() {
        // correlation switched off by comparing against the analytic exchange with PW92 correlation subtracted
        let n = 0.1;
        let (e, vu, vd) = exc_point(n / 2.0, n / 2.0);
        let ex = -(3.0 / 4.0) * (3.0 / PI).powf(1.0 / 3.0) * n.powf(4.0 / 3.0);
        let (ec_density, vc) = correlation_only(n / 2.0, n / 2.0);
        assert_relative_eq!(e - ec_density, ex, epsilon = 1e-12);
        assert_relative_eq!(vu - vc, -(3.0 * n / PI).powf(1.0 / 3.0), epsilon = 1e-12);
        assert_relative_eq!(vu, vd, epsilon = 1e-15);
    }

    #[test]
    fn correlation_matches_pw92_reference() {
        // PW92 Table I / libxc LDA_C_PW, unpolarised: ε_c(r_s=2) = −0.04476 Ha, ε_c(r_s=10) = −0.01855 Ha
        for (rs, eps_ref) in [(2.0, -0.04476), (10.0, -0.01855)] {
            let n = 3.0 / (4.0 * PI * rs * rs * rs);
            let (ec_density, _) = correlation_only(n / 2.0, n / 2.0);
            assert_relative_eq!(ec_density / n, eps_ref, epsilon = 5e-4);
        }
    }

    #[test]
    fn fully_polarized_uses_ferromagnetic_branch() {
        let n = 0.02;
        let (e_pol, _, _) = exc_point(n, 0.0);
        assert!(e_pol.is_finite());
        // ζ = 1 ⇒ f = 1, ζ⁴ = 1 ⇒ ε_c = ε_c(r_s, 1); check against the FERRO fit directly
        let rs = (3.0 / (4.0 * PI * n)).powf(1.0 / 3.0);
        let (ec1, _) = pw92_g(rs, &FERRO);
        let (ec_density, _) = correlation_only(n, 0.0);
        assert_relative_eq!(ec_density / n, ec1, epsilon = 1e-12);
    }

    #[test]
    fn potentials_match_finite_difference() {
        for (nu, nd) in [(0.3, 0.3), (0.05, 0.2), (0.001, 0.0004), (0.1, 0.0)] {
            let (_, vu, vd) = exc_point(nu, nd);
            let d = 1e-6 * (nu + nd);
            let fd_u = (energy(nu + d, nd) - energy(nu - d, nd)) / (2.0 * d);
            assert_relative_eq!(vu, fd_u, epsilon = 1e-7, max_relative = 1e-6);
            if nd > 0.0 {
                let fd_d = (energy(nu, nd + d) - energy(nu, nd - d)) / (2.0 * d);
                assert_relative_eq!(vd, fd_d, epsilon = 1e-7, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn zero_density_gives_zero_not_nan() {
        assert_eq!(exc_point(0.0, 0.0), (0.0, 0.0, 0.0));
        let r = lsda_pw92(&[0.0, 1e-30, 0.1], &[0.0, 0.0, 0.1]);
        assert!(r.energy_density.iter().chain(&r.v_up).chain(&r.v_dn).all(|x| x.is_finite()));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm xc`
Expected: compile error — `exc_point` not defined.

- [ ] **Step 3: Implement**

```rust
//! LSDA exchange-correlation: Dirac exchange (spin-scaled) + Perdew–Wang 1992 correlation.
//! (derivation block from the task description goes here)

use std::f64::consts::PI;

pub struct Pw92Params { a: f64, a1: f64, b1: f64, b2: f64, b3: f64, b4: f64 }
pub const PARA:  Pw92Params = Pw92Params { a: 0.031091, a1: 0.21370, b1: 7.5957,  b2: 3.5876, b3: 1.6382,  b4: 0.49294 };
pub const FERRO: Pw92Params = Pw92Params { a: 0.015545, a1: 0.20548, b1: 14.1189, b2: 6.1977, b3: 3.3662,  b4: 0.62517 };
pub const MALPHA: Pw92Params = Pw92Params { a: 0.016887, a1: 0.11125, b1: 10.357,  b2: 3.6231, b3: 0.88026, b4: 0.49671 };
const FPP0: f64 = 1.709921;
const N_MIN: f64 = 1e-20;

/// Returns (G, dG/dr_s).
pub fn pw92_g(rs: f64, p: &Pw92Params) -> (f64, f64) {
    let srs = rs.sqrt();
    let q0 = -2.0 * p.a * (1.0 + p.a1 * rs);
    let q1 = 2.0 * p.a * (p.b1 * srs + p.b2 * rs + p.b3 * rs * srs + p.b4 * rs * rs);
    let q1p = p.a * (p.b1 / srs + 2.0 * p.b2 + 3.0 * p.b3 * srs + 4.0 * p.b4 * rs);
    let ln = (1.0 + 1.0 / q1).ln();
    (q0 * ln, -2.0 * p.a * p.a1 * ln - q0 * q1p / (q1 * q1 + q1))
}

fn f_zeta(z: f64) -> (f64, f64) {
    let denom = 2f64.powf(4.0 / 3.0) - 2.0;
    let f = ((1.0 + z).powf(4.0 / 3.0) + (1.0 - z).powf(4.0 / 3.0) - 2.0) / denom;
    let fp = (4.0 / 3.0) * ((1.0 + z).powf(1.0 / 3.0) - (1.0 - z).powf(1.0 / 3.0)) / denom;
    (f, fp)
}

/// (n·ε_c, v_c averaged over spin — used by tests; see exc_point for the full result)
pub fn correlation_only(n_up: f64, n_dn: f64) -> (f64, f64) {
    let (e, vu, vd) = correlation(n_up, n_dn);
    (e, 0.5 * (vu + vd))
}

fn correlation(n_up: f64, n_dn: f64) -> (f64, f64, f64) {
    let n = n_up + n_dn;
    if n < N_MIN { return (0.0, 0.0, 0.0); }
    let rs = (3.0 / (4.0 * PI * n)).powf(1.0 / 3.0);
    let z = ((n_up - n_dn) / n).clamp(-1.0, 1.0);
    let (ec0, dec0) = pw92_g(rs, &PARA);
    let (ec1, dec1) = pw92_g(rs, &FERRO);
    let (mac, dmac) = pw92_g(rs, &MALPHA);
    let (ac, dac) = (-mac, -dmac);
    let (f, fp) = f_zeta(z);
    let z3 = z * z * z;
    let z4 = z3 * z;
    let ec = ec0 + ac * f / FPP0 * (1.0 - z4) + (ec1 - ec0) * f * z4;
    let dec_drs = dec0 + dac * f / FPP0 * (1.0 - z4) + (dec1 - dec0) * f * z4;
    let dec_dz = ac / FPP0 * (fp * (1.0 - z4) - 4.0 * z3 * f) + (ec1 - ec0) * (fp * z4 + 4.0 * z3 * f);
    let common = ec - rs / 3.0 * dec_drs;
    (n * ec, common - (z - 1.0) * dec_dz, common - (z + 1.0) * dec_dz)
}

fn exchange(n_up: f64, n_dn: f64) -> (f64, f64, f64) {
    let c = (3.0 / 4.0) * (3.0 / PI).powf(1.0 / 3.0) * 2f64.powf(1.0 / 3.0);
    let e = -c * (n_up.powf(4.0 / 3.0) + n_dn.powf(4.0 / 3.0));
    (e, -(6.0 * n_up / PI).powf(1.0 / 3.0), -(6.0 * n_dn / PI).powf(1.0 / 3.0))
}

pub fn exc_point(n_up: f64, n_dn: f64) -> (f64, f64, f64) {
    let (n_up, n_dn) = (n_up.max(0.0), n_dn.max(0.0));
    if n_up + n_dn < N_MIN { return (0.0, 0.0, 0.0); }
    let (ex, vxu, vxd) = exchange(n_up, n_dn);
    let (ec, vcu, vcd) = correlation(n_up, n_dn);
    (ex + ec, vxu + vcu, vxd + vcd)
}

pub struct XcResult { pub energy_density: Vec<f64>, pub v_up: Vec<f64>, pub v_dn: Vec<f64> }

pub fn lsda_pw92(n_up: &[f64], n_dn: &[f64]) -> XcResult {
    let n = n_up.len();
    let mut r = XcResult { energy_density: vec![0.0; n], v_up: vec![0.0; n], v_dn: vec![0.0; n] };
    for i in 0..n {
        let (e, vu, vd) = exc_point(n_up[i], n_dn[i]);
        r.energy_density[i] = e; r.v_up[i] = vu; r.v_dn[i] = vd;
    }
    r
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm xc`
Expected: 5 passed. If `correlation_matches_pw92_reference` fails by more than 5e-4, re-check the reference values against Table I of Perdew & Wang, PRB 45, 13244 (1992) before touching the code; do not widen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): LSDA exchange-correlation (Dirac + PW92) with analytic potentials"
```

---

### Task 7: `qm::nuclear` — Gaussian nuclei, external potential, repulsion **[Fable]**

**Files:**
- Create: `crates/qm/src/nuclear.rs`, `crates/qm/src/error.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod nuclear; pub mod error;`)

**Interfaces:**
- Consumes: `grid::Grid`, `poisson::PoissonSolver` (tests only).
- Produces:
  ```rust
  // error.rs
  #[derive(Debug, thiserror::Error)]
  pub enum QmError {
      #[error("nuclei {0} and {1} coincide")] CoincidentNuclei(usize, usize),
      #[error("SCF did not converge after {0} iterations (Δρ = {1:.3e})")] ScfNotConverged(usize, f64),
  }
  // nuclear.rs
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct Nucleus { pub pos: [f64; 3], pub z: u8 }
  /// Positive charge density: Σ_I Z_I (2πσ²)^{-3/2} exp(−|r−R_I|²/2σ²)
  pub fn deposit_charge(grid: &Grid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64>;
  /// Potential felt by an electron: V_nuc(r) = −Σ_I Z_I erf(|r−R_I|/(σ√2))/|r−R_I|  (→ −Z√(2/π)/σ at r = R_I)
  pub fn nuclear_potential(grid: &Grid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64>;
  /// Σ_{I<J} Z_I Z_J / |R_I − R_J| for point nuclei (nuclei never see each other smeared).
  pub fn nuclear_repulsion(nuclei: &[Nucleus]) -> Result<f64, QmError>;
  ```

Rationale (module doc comment): a point nucleus has a 1/r singularity the grid cannot represent. Smearing it into a Gaussian of width σ makes V_nuc smooth with error O(σ²·ρ(R)) in the energy — ΔE ≈ (2π/3)·Z·ρ_e(R)·σ² to leading order — which validation (Task 9) removes by extrapolating σ → 0. The analytic erf form is used on the CPU; the GPU path (M2) obtains the same potential from the Poisson solver, and the test below pins that the two agree.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::poisson::PoissonSolver;
    use approx::assert_relative_eq;

    #[test]
    fn deposited_charge_integrates_to_total_z() {
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 3 }];
        let rho = deposit_charge(&g, &nuc, 0.25);
        assert_relative_eq!(g.integrate(&rho), 4.0, epsilon = 1e-8);
    }

    #[test]
    fn analytic_potential_matches_poisson_of_deposit() {
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.7, 0.3, -0.2], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 2 }];
        let sigma = 0.3;
        let rho = deposit_charge(&g, &nuc, sigma);
        let mut phi = vec![0.0; g.len()];
        PoissonSolver::new(&g).solve(&g, &rho, &mut phi, 1e-10, 3000);
        let v = nuclear_potential(&g, &nuc, sigma);
        let err = v.iter().zip(&phi).map(|(a, b)| (a + b).abs()).fold(0.0, f64::max);
        assert!(err < 1e-4, "max |V_nuc + φ| = {err}");
    }

    #[test]
    fn potential_is_coulombic_far_away() {
        let g = Grid::cube(49, 0.25);
        let nuc = [Nucleus { pos: [0.0; 3], z: 6 }];
        let v = nuclear_potential(&g, &nuc, 0.2);
        let far = g.idx(44, 24, 24); // x = 5 Bohr
        assert_relative_eq!(v[far], -6.0 / 5.0, epsilon = 1e-12);
        let on = g.idx(24, 24, 24);
        assert_relative_eq!(v[on], -6.0 * (2.0 / std::f64::consts::PI).sqrt() / 0.2, epsilon = 1e-12);
    }

    #[test]
    fn repulsion_h2() {
        let nuc = [Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }];
        assert_relative_eq!(nuclear_repulsion(&nuc).unwrap(), 1.0 / 1.4, epsilon = 1e-15);
    }

    #[test]
    fn coincident_nuclei_is_an_error() {
        let nuc = [Nucleus { pos: [0.0; 3], z: 1 }, Nucleus { pos: [0.0; 3], z: 1 }];
        assert!(matches!(nuclear_repulsion(&nuc), Err(QmError::CoincidentNuclei(0, 1))));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm nuclear`
Expected: compile error — `Nucleus` not defined.

- [ ] **Step 3: Implement**

```rust
// error.rs
#[derive(Debug, thiserror::Error)]
pub enum QmError {
    #[error("nuclei {0} and {1} coincide")]
    CoincidentNuclei(usize, usize),
    #[error("SCF did not converge after {0} iterations (Δρ = {1:.3e})")]
    ScfNotConverged(usize, f64),
}
```

```rust
// nuclear.rs
//! Nuclei as Gaussian charges of width σ (see task rationale). Lengths in Bohr, charges in e.
use crate::error::QmError;
use crate::grid::Grid;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nucleus { pub pos: [f64; 3], pub z: u8 }

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

pub fn deposit_charge(grid: &Grid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64> {
    let norm = 1.0 / (2.0 * PI * sigma * sigma).powf(1.5);
    grid.sample(|r| nuclei.iter().map(|n| n.z as f64 * norm * (-dist2(r, n.pos) / (2.0 * sigma * sigma)).exp()).sum())
}

pub fn nuclear_potential(grid: &Grid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64> {
    let s2 = sigma * 2f64.sqrt();
    grid.sample(|r| {
        nuclei.iter().map(|n| {
            let d = dist2(r, n.pos).sqrt();
            let z = n.z as f64;
            if d < 1e-10 { -z * (2.0 / PI).sqrt() / sigma } else { -z * libm::erf(d / s2) / d }
        }).sum()
    })
}

pub fn nuclear_repulsion(nuclei: &[Nucleus]) -> Result<f64, QmError> {
    let mut e = 0.0;
    for i in 0..nuclei.len() {
        for j in (i + 1)..nuclei.len() {
            let d = dist2(nuclei[i].pos, nuclei[j].pos).sqrt();
            if d < 1e-8 { return Err(QmError::CoincidentNuclei(i, j)); }
            e += nuclei[i].z as f64 * nuclei[j].z as f64 / d;
        }
    }
    Ok(e)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm nuclear`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): Gaussian nuclei, analytic external potential, point repulsion"
```

---

### Task 8: `qm::eigen` — LOBPCG for the lowest k eigenpairs **[Fable]**

**Files:**
- Create: `crates/qm/src/eigen.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod eigen;`)

**Interfaces:**
- Consumes: `grid::Grid`, `stencil::kinetic`.
- Produces:
  ```rust
  pub trait Operator { fn apply(&self, x: &[f64], y: &mut [f64]); }
  /// H = −½∇² + V(r) for a local potential on the grid.
  pub struct LocalHamiltonian<'a> { pub grid: &'a Grid, pub v: &'a [f64] }
  impl Operator for LocalHamiltonian<'_> { ... }
  pub struct EigenStats { pub iterations: usize, pub max_residual: f64 }
  /// Lowest `x.len()` eigenpairs of `op` with respect to the L² inner product `grid.dot`.
  /// `x` holds the initial guess (any linearly independent vectors) and the result, orthonormal.
  /// `precond` maps a residual to an approximate correction (None = identity).
  pub fn lobpcg(grid: &Grid, op: &dyn Operator, x: &mut [Vec<f64>], evals: &mut [f64],
                precond: Option<&dyn Fn(&[f64], &mut [f64])>, tol: f64, max_iter: usize) -> EigenStats;
  /// Jacobi preconditioner for LocalHamiltonian: r / max(diag(H) − λ, 0.1)
  pub fn jacobi_precond<'a>(grid: &'a Grid, v: &'a [f64], lambda: f64) -> impl Fn(&[f64], &mut [f64]) + 'a;
  ```

Algorithm (module doc comment, Knyazev 2001): each iteration, R = HX − XΛ; W = T·R (preconditioned residuals); S = [X, W, P] orthonormalised by modified Gram–Schmidt; solve the k×k… 3k×3k Rayleigh–Ritz problem SᵀHS c = θ c (symmetric, nalgebra `SymmetricEigen`); keep the lowest k; new X = S C; new P = S_{W,P part} C. Converged when max_i ‖r_i‖ < tol. Orthonormalising the whole subspace each iteration costs more than textbook LOBPCG but is numerically robust; the GPU plan may revisit.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use approx::assert_relative_eq;

    /// 3D isotropic harmonic oscillator: H = −½∇² + ½r², E = n + 3/2 ⇒ 1.5, 2.5 (×3), 3.5 (×6)
    fn oscillator(n: usize, h: f64) -> (Grid, Vec<f64>) {
        let g = Grid::cube(n, h);
        let v = g.sample(|[x, y, z]| 0.5 * (x * x + y * y + z * z));
        (g, v)
    }

    fn random_guess(g: &Grid, k: usize) -> Vec<Vec<f64>> {
        // deterministic pseudo-random so the test is reproducible
        let mut seed = 12345u64;
        (0..k).map(|_| g.sample(|[x, y, z]| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            (u - 0.5) * (-(x * x + y * y + z * z) / 8.0).exp()
        })).collect()
    }

    #[test]
    fn oscillator_spectrum() {
        let (g, v) = oscillator(49, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let mut x = random_guess(&g, 4);
        let mut e = vec![0.0; 4];
        let stats = lobpcg(&g, &op, &mut x, &mut e, None, 1e-6, 500);
        assert!(stats.max_residual < 1e-6, "{stats:?}");
        assert_relative_eq!(e[0], 1.5, epsilon = 1e-4);
        for i in 1..4 { assert_relative_eq!(e[i], 2.5, epsilon = 1e-4); }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let (g, v) = oscillator(33, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let mut x = random_guess(&g, 3);
        let mut e = vec![0.0; 3];
        lobpcg(&g, &op, &mut x, &mut e, None, 1e-6, 500);
        for i in 0..3 { for j in 0..3 {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!((g.dot(&x[i], &x[j]) - expect).abs() < 1e-8);
        }}
    }

    #[test]
    fn preconditioner_reduces_iterations() {
        let (g, v) = oscillator(33, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let (mut x1, mut e1) = (random_guess(&g, 2), vec![0.0; 2]);
        let plain = lobpcg(&g, &op, &mut x1, &mut e1, None, 1e-6, 500);
        let (mut x2, mut e2) = (random_guess(&g, 2), vec![0.0; 2]);
        let pc = jacobi_precond(&g, &v, 1.5);
        let pre = lobpcg(&g, &op, &mut x2, &mut e2, Some(&pc), 1e-6, 500);
        assert!(pre.iterations < plain.iterations, "plain {} pre {}", plain.iterations, pre.iterations);
    }

    #[test]
    fn warm_start_converges_immediately() {
        let (g, v) = oscillator(33, 0.25);
        let op = LocalHamiltonian { grid: &g, v: &v };
        let (mut x, mut e) = (random_guess(&g, 2), vec![0.0; 2]);
        lobpcg(&g, &op, &mut x, &mut e, None, 1e-8, 500);
        let again = lobpcg(&g, &op, &mut x, &mut e, None, 1e-6, 500);
        assert!(again.iterations <= 1, "{again:?}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm eigen`
Expected: compile error — `lobpcg` not defined.

- [ ] **Step 3: Implement**

```rust
//! LOBPCG (Knyazev 2001) with full subspace orthonormalisation. Inner product = grid.dot (L²).
use crate::grid::Grid;
use crate::stencil::kinetic;
use nalgebra::{DMatrix, SymmetricEigen};

pub trait Operator { fn apply(&self, x: &[f64], y: &mut [f64]); }

pub struct LocalHamiltonian<'a> { pub grid: &'a Grid, pub v: &'a [f64] }

impl Operator for LocalHamiltonian<'_> {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        kinetic(self.grid, x, y);
        for i in 0..y.len() { y[i] += self.v[i] * x[i]; }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EigenStats { pub iterations: usize, pub max_residual: f64 }

pub fn jacobi_precond<'a>(grid: &'a Grid, v: &'a [f64], lambda: f64) -> impl Fn(&[f64], &mut [f64]) + 'a {
    let diag_t = -1.5 * crate::stencil::COEFF[0] / (grid.h * grid.h); // diagonal of −½∇²
    move |r: &[f64], out: &mut [f64]| {
        for i in 0..r.len() { out[i] = r[i] / (diag_t + v[i] - lambda).max(0.1); }
    }
}

/// Modified Gram–Schmidt in the grid inner product; drops vectors with norm < 1e-12.
fn orthonormalise(grid: &Grid, s: &mut Vec<Vec<f64>>) {
    let mut kept = Vec::with_capacity(s.len());
    for mut v in s.drain(..) {
        for q in &kept {
            let c = grid.dot(q, &v);
            for i in 0..v.len() { v[i] -= c * q[i]; }
        }
        let nrm = grid.dot(&v, &v).sqrt();
        if nrm > 1e-12 { for x in v.iter_mut() { *x /= nrm; } kept.push(v); }
    }
    *s = kept;
}

pub fn lobpcg(grid: &Grid, op: &dyn Operator, x: &mut [Vec<f64>], evals: &mut [f64],
              precond: Option<&dyn Fn(&[f64], &mut [f64])>, tol: f64, max_iter: usize) -> EigenStats {
    let k = x.len();
    let n = grid.len();
    let mut xs: Vec<Vec<f64>> = x.to_vec();
    orthonormalise(grid, &mut xs);
    assert_eq!(xs.len(), k, "initial guess vectors are linearly dependent");
    let mut p: Vec<Vec<f64>> = Vec::new();
    let mut hx: Vec<Vec<f64>> = vec![vec![0.0; n]; k];
    let mut stats = EigenStats { iterations: 0, max_residual: f64::INFINITY };

    loop {
        // Rayleigh quotients and residuals
        for i in 0..k { op.apply(&xs[i], &mut hx[i]); evals[i] = grid.dot(&xs[i], &hx[i]); }
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut max_res = 0.0f64;
        for i in 0..k {
            let r: Vec<f64> = (0..n).map(|j| hx[i][j] - evals[i] * xs[i][j]).collect();
            max_res = max_res.max(grid.dot(&r, &r).sqrt());
            let mut wi = vec![0.0; n];
            match precond { Some(t) => t(&r, &mut wi), None => wi.copy_from_slice(&r) }
            w.push(wi);
        }
        stats.max_residual = max_res;
        if max_res < tol || stats.iterations >= max_iter { break; }
        stats.iterations += 1;

        // Subspace S = [X, W, P], orthonormalised
        let mut s: Vec<Vec<f64>> = Vec::with_capacity(3 * k);
        s.extend(xs.iter().cloned());
        s.extend(w.into_iter());
        s.extend(p.iter().cloned());
        orthonormalise(grid, &mut s);
        let m = s.len();

        // Rayleigh–Ritz: (SᵀHS) c = θ c
        let mut hs: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        for i in 0..m { op.apply(&s[i], &mut hs[i]); }
        let mut a = DMatrix::<f64>::zeros(m, m);
        for i in 0..m { for j in 0..=i {
            let v = grid.dot(&s[i], &hs[j]);
            a[(i, j)] = v; a[(j, i)] = v;
        }}
        let eig = SymmetricEigen::new(a);
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&i, &j| eig.eigenvalues[i].partial_cmp(&eig.eigenvalues[j]).unwrap());

        // New X and P from the lowest k Ritz vectors; P excludes the X-part of S (first k columns)
        let mut new_x = vec![vec![0.0; n]; k];
        let mut new_p = vec![vec![0.0; n]; k];
        for (col, &idx) in order.iter().take(k).enumerate() {
            let c = eig.eigenvectors.column(idx);
            for (si, sv) in s.iter().enumerate() {
                let coef = c[si];
                if coef == 0.0 { continue; }
                for j in 0..n { new_x[col][j] += coef * sv[j]; }
                if si >= k { for j in 0..n { new_p[col][j] += coef * sv[j]; } }
            }
        }
        xs = new_x;
        orthonormalise(grid, &mut xs);
        p = new_p;
    }
    for i in 0..k { x[i].copy_from_slice(&xs[i]); }
    stats
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm eigen`
Expected: 4 passed. Record iteration counts (plain vs preconditioned) in the commit message.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): LOBPCG eigensolver with Jacobi preconditioner"
```

---

### Task 9: `qm::scf` — spin-polarized Kohn-Sham SCF loop and total energy **[Fable]**

**Files:**
- Create: `crates/qm/src/scf.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod scf;` and re-exports: `pub use error::QmError; pub use grid::Grid; pub use nuclear::Nucleus; pub use scf::{System, ScfSettings, KsState, Energy, ScfStats, scf};`)

**Interfaces:**
- Consumes: `Grid`, `stencil::kinetic`, `poisson::{PoissonSolver, hartree_energy}`, `xc::lsda_pw92`, `nuclear::{Nucleus, nuclear_potential, nuclear_repulsion, deposit_charge}`, `eigen::{lobpcg, LocalHamiltonian, jacobi_precond}`.
- Produces:
  ```rust
  pub struct System { pub nuclei: Vec<Nucleus>, pub n_up: usize, pub n_dn: usize }
  impl System {
      /// N = ΣZ − charge. multiplicity = 2S+1; None ⇒ lowest guess (paired; one unpaired if N odd).
      pub fn new(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>) -> System;
  }
  pub struct ScfSettings { pub sigma: f64, pub tol: f64, pub max_iter: usize, pub mixing: f64, pub eig_tol: f64 }
  impl ScfSettings { pub fn for_grid(grid: &Grid) -> ScfSettings } // sigma = grid.h, tol 1e-6, max_iter 200, mixing 0.3, eig_tol 1e-6
  pub struct KsState {
      pub orbitals_up: Vec<Vec<f64>>, pub orbitals_dn: Vec<Vec<f64>>,
      pub evals_up: Vec<f64>, pub evals_dn: Vec<f64>,
      pub rho_up: Vec<f64>, pub rho_dn: Vec<f64>,
      pub v_nuc: Vec<f64>, pub v_h: Vec<f64>,
  }
  impl KsState { pub fn rho(&self) -> Vec<f64> } // ρ↑ + ρ↓
  #[derive(Debug, Clone, Copy)]
  pub struct Energy { pub kinetic: f64, pub external: f64, pub hartree: f64, pub xc: f64, pub nuclear: f64, pub total: f64 }
  #[derive(Debug, Clone)]
  pub struct ScfStats { pub iterations: usize, pub delta_rho: f64, pub energy: Energy, pub boundary_density: f64, pub warnings: Vec<String> }
  /// Runs SCF. `warm` reuses a previous state's orbitals/density as the starting point (counts must match).
  pub fn scf(grid: &Grid, system: &System, settings: &ScfSettings, warm: Option<&KsState>) -> Result<(KsState, ScfStats), QmError>;
  ```

Derivation (module doc comment): Kohn-Sham total energy with fixed grid basis
E = T_s + ∫ρV_nuc + E_H[ρ] + E_xc[ρ↑,ρ↓] + E_nn, where T_s = Σ_{iσ} ∫ψ_{iσ}(−½∇²)ψ_{iσ}, E_H = ½∫ρV_H, E_xc = ∫ n·ε_xc. Each SCF iteration: V_eff,σ = V_nuc + V_H[ρ_in] + v_xc,σ[ρ_in]; solve for the lowest n_σ orbitals per spin; ρ_out,σ = Σ_i |ψ_iσ|²; Δρ = ∫|ρ_out − ρ_in|; linear mixing ρ_in ← (1−α)ρ_in + αρ_out. The energy is evaluated from ρ_out and the orbitals. Warning (not error) when the density on the box faces exceeds 1e-5: the Dirichlet wall is confining the system.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::nuclear::Nucleus;
    use approx::assert_relative_eq;

    fn hydrogen(pos: [f64; 3]) -> System { System::new(vec![Nucleus { pos, z: 1 }], 0, None) }

    fn h_energy(grid: &Grid, sigma: f64, pos: [f64; 3]) -> (f64, ScfStats) {
        let mut s = ScfSettings::for_grid(grid);
        s.sigma = sigma; s.tol = 1e-7;
        let (_, stats) = scf(grid, &hydrogen(pos), &s, None).unwrap();
        (stats.energy.total, stats)
    }

    #[test]
    fn spin_counts() {
        let h = hydrogen([0.0; 3]);
        assert_eq!((h.n_up, h.n_dn), (1, 0));
        let li = System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None);
        assert_eq!((li.n_up, li.n_dn), (2, 1));
        let he_triplet = System::new(vec![Nucleus { pos: [0.0; 3], z: 2 }], 0, Some(3));
        assert_eq!((he_triplet.n_up, he_triplet.n_dn), (2, 0));
        let h_plus = System::new(vec![Nucleus { pos: [0.0; 3], z: 1 }], 1, None);
        assert_eq!((h_plus.n_up, h_plus.n_dn), (0, 0));
    }

    #[test]
    fn hydrogen_converges_and_conserves_charge() {
        let g = Grid::cube(49, 0.25);
        let (state, stats) = scf(&g, &hydrogen([0.0; 3]), &ScfSettings::for_grid(&g), None).unwrap();
        assert!(stats.delta_rho < 1e-6, "{stats:?}");
        assert_relative_eq!(g.integrate(&state.rho_up), 1.0, epsilon = 1e-8);
        assert_eq!(g.integrate(&state.rho_dn), 0.0);
        assert!(stats.energy.total < -0.40 && stats.energy.total > -0.50, "E = {}", stats.energy.total);
        assert!(stats.warnings.is_empty(), "{:?}", stats.warnings);
    }

    #[test]
    fn hydrogen_energy_extrapolated_to_point_nucleus() {
        // Smearing error ∝ σ²: fit E = a + bσ² + cσ⁴ through three σ and read off a.
        // Reference: NIST LSD total energy of H = −0.478671 Ha (Kotochigova et al.).
        let g = Grid::cube(49, 0.25);
        let sig = [0.25, 0.30, 0.35];
        let e: Vec<f64> = sig.iter().map(|&s| h_energy(&g, s, [0.0; 3]).0).collect();
        let x: Vec<f64> = sig.iter().map(|s| s * s).collect();
        // Lagrange interpolation at x = 0
        let a = e[0] * (x[1] * x[2]) / ((x[0] - x[1]) * (x[0] - x[2]))
              + e[1] * (x[0] * x[2]) / ((x[1] - x[0]) * (x[1] - x[2]))
              + e[2] * (x[0] * x[1]) / ((x[2] - x[0]) * (x[2] - x[1]));
        assert!((a - (-0.478671)).abs() < 5e-3, "extrapolated E = {a}, raw = {e:?}");
    }

    #[test]
    fn egg_box_ripple_is_small() {
        let g = Grid::cube(49, 0.25);
        let (e0, _) = h_energy(&g, 0.25, [0.0; 3]);
        let (e1, _) = h_energy(&g, 0.25, [0.125, 0.0, 0.0]);
        assert!((e1 - e0).abs() < 2e-3, "egg-box ΔE = {}", e1 - e0);
    }

    #[test]
    fn small_box_raises_energy_and_warns() {
        let big = Grid::cube(49, 0.25);  // 12 Bohr
        let small = Grid::cube(25, 0.25); // 6 Bohr
        let (e_big, _) = h_energy(&big, 0.25, [0.0; 3]);
        let (e_small, stats) = h_energy(&small, 0.25, [0.0; 3]);
        assert!(e_small > e_big, "confinement must raise the energy: {e_small} vs {e_big}");
        assert!(!stats.warnings.is_empty(), "expected a box-size warning");
    }

    #[test]
    fn lithium_runs_with_unequal_spin_channels() {
        let g = Grid::cube(49, 0.25);
        let li = System::new(vec![Nucleus { pos: [0.0; 3], z: 3 }], 0, None);
        let (state, stats) = scf(&g, &li, &ScfSettings::for_grid(&g), None).unwrap();
        assert_eq!(state.orbitals_up.len(), 2);
        assert_eq!(state.orbitals_dn.len(), 1);
        assert!(stats.delta_rho < 1e-6);
        assert_relative_eq!(g.integrate(&state.rho()), 3.0, epsilon = 1e-8);
    }

    #[test]
    fn warm_start_converges_in_few_iterations() {
        let g = Grid::cube(33, 0.25);
        let s = ScfSettings::for_grid(&g);
        let (state, cold) = scf(&g, &hydrogen([0.0; 3]), &s, None).unwrap();
        let (_, warm) = scf(&g, &hydrogen([0.02, 0.0, 0.0]), &s, Some(&state)).unwrap();
        assert!(warm.iterations * 3 < cold.iterations, "cold {} warm {}", cold.iterations, warm.iterations);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm scf`
Expected: compile error — `System` not defined.

- [ ] **Step 3: Implement**

```rust
//! Spin-polarized Kohn-Sham SCF on the real-space grid (derivation in the task description).
use crate::eigen::{jacobi_precond, lobpcg, LocalHamiltonian};
use crate::error::QmError;
use crate::grid::Grid;
use crate::nuclear::{deposit_charge, nuclear_potential, nuclear_repulsion, Nucleus};
use crate::poisson::{hartree_energy, PoissonSolver};
use crate::stencil::kinetic;
use crate::xc::lsda_pw92;

pub struct System { pub nuclei: Vec<Nucleus>, pub n_up: usize, pub n_dn: usize }

impl System {
    pub fn new(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>) -> System {
        let total: i32 = nuclei.iter().map(|n| n.z as i32).sum::<i32>() - charge;
        assert!(total >= 0, "negative electron count");
        let n = total as usize;
        let unpaired = match multiplicity {
            Some(m) => { assert!(m >= 1 && (m as usize - 1) <= n && (n - (m as usize - 1)) % 2 == 0, "bad multiplicity"); m as usize - 1 }
            None => n % 2,
        };
        let n_dn = (n - unpaired) / 2;
        System { nuclei, n_up: n_dn + unpaired, n_dn }
    }
}

pub struct ScfSettings { pub sigma: f64, pub tol: f64, pub max_iter: usize, pub mixing: f64, pub eig_tol: f64 }
impl ScfSettings {
    pub fn for_grid(grid: &Grid) -> ScfSettings {
        ScfSettings { sigma: grid.h, tol: 1e-6, max_iter: 200, mixing: 0.3, eig_tol: 1e-6 }
    }
}

pub struct KsState {
    pub orbitals_up: Vec<Vec<f64>>, pub orbitals_dn: Vec<Vec<f64>>,
    pub evals_up: Vec<f64>, pub evals_dn: Vec<f64>,
    pub rho_up: Vec<f64>, pub rho_dn: Vec<f64>,
    pub v_nuc: Vec<f64>, pub v_h: Vec<f64>,
}
impl KsState {
    pub fn rho(&self) -> Vec<f64> { self.rho_up.iter().zip(&self.rho_dn).map(|(a, b)| a + b).collect() }
}

#[derive(Debug, Clone, Copy)]
pub struct Energy { pub kinetic: f64, pub external: f64, pub hartree: f64, pub xc: f64, pub nuclear: f64, pub total: f64 }

#[derive(Debug, Clone)]
pub struct ScfStats { pub iterations: usize, pub delta_rho: f64, pub energy: Energy, pub boundary_density: f64, pub warnings: Vec<String> }

/// Deterministic Gaussian-enveloped pseudo-random guess, orthogonalised later by LOBPCG.
fn initial_orbitals(grid: &Grid, nuclei: &[Nucleus], k: usize, seed0: u64) -> Vec<Vec<f64>> {
    let mut seed = seed0;
    (0..k).map(|_| grid.sample(|r| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
        let env: f64 = nuclei.iter().map(|n| (-((r[0]-n.pos[0]).powi(2)+(r[1]-n.pos[1]).powi(2)+(r[2]-n.pos[2]).powi(2))/4.0).exp()).sum();
        (u + 0.5) * env
    })).collect()
}

fn density_from(grid: &Grid, orbitals: &[Vec<f64>]) -> Vec<f64> {
    let mut rho = vec![0.0; grid.len()];
    for psi in orbitals { for i in 0..rho.len() { rho[i] += psi[i] * psi[i]; } }
    rho
}

fn max_on_faces(grid: &Grid, f: &[f64]) -> f64 {
    let [n0, n1, n2] = grid.n;
    let mut m: f64 = 0.0;
    for k in 0..n2 { for j in 0..n1 { for i in 0..n0 {
        if i == 0 || j == 0 || k == 0 || i == n0 - 1 || j == n1 - 1 || k == n2 - 1 { m = m.max(f[grid.idx(i, j, k)].abs()); }
    }}}
    m
}

fn solve_channel(grid: &Grid, v_eff: &[f64], orbitals: &mut Vec<Vec<f64>>, evals: &mut Vec<f64>, eig_tol: f64) {
    if orbitals.is_empty() { return; }
    let op = LocalHamiltonian { grid, v: v_eff };
    let lambda = evals.first().copied().unwrap_or(-1.0);
    let pc = jacobi_precond(grid, v_eff, lambda);
    lobpcg(grid, &op, orbitals, evals, Some(&pc), eig_tol, 200);
}

pub fn scf(grid: &Grid, system: &System, settings: &ScfSettings, warm: Option<&KsState>) -> Result<(KsState, ScfStats), QmError> {
    let n = grid.len();
    let e_nn = nuclear_repulsion(&system.nuclei)?;
    let v_nuc = nuclear_potential(grid, &system.nuclei, settings.sigma);
    let mut poisson = PoissonSolver::new(grid);

    // starting point
    let (mut orb_up, mut orb_dn, mut ev_up, mut ev_dn, mut rho_in_up, mut rho_in_dn, mut v_h) = match warm {
        Some(w) if w.orbitals_up.len() == system.n_up && w.orbitals_dn.len() == system.n_dn => (
            w.orbitals_up.clone(), w.orbitals_dn.clone(), w.evals_up.clone(), w.evals_dn.clone(),
            w.rho_up.clone(), w.rho_dn.clone(), w.v_h.clone()),
        _ => {
            let total = (system.n_up + system.n_dn) as f64;
            let z_tot: f64 = system.nuclei.iter().map(|q| q.z as f64).sum();
            let guess = deposit_charge(grid, &system.nuclei, 1.0); // broad atomic-like blobs
            let scale = if z_tot > 0.0 { total / z_tot } else { 0.0 };
            let fu = if total > 0.0 { system.n_up as f64 / total } else { 0.0 };
            let fd = if total > 0.0 { system.n_dn as f64 / total } else { 0.0 };
            (initial_orbitals(grid, &system.nuclei, system.n_up, 1), initial_orbitals(grid, &system.nuclei, system.n_dn, 2),
             vec![-1.0; system.n_up], vec![-1.0; system.n_dn],
             guess.iter().map(|g| g * scale * fu).collect(), guess.iter().map(|g| g * scale * fd).collect(),
             vec![0.0; n])
        }
    };

    let mut iterations = 0;
    let mut delta_rho = f64::INFINITY;
    let (mut rho_out_up, mut rho_out_dn) = (rho_in_up.clone(), rho_in_dn.clone());
    while iterations < settings.max_iter {
        iterations += 1;
        let rho_in: Vec<f64> = rho_in_up.iter().zip(&rho_in_dn).map(|(a, b)| a + b).collect();
        poisson.solve(grid, &rho_in, &mut v_h, 1e-10, 5000);
        let xc = lsda_pw92(&rho_in_up, &rho_in_dn);
        let v_up: Vec<f64> = (0..n).map(|i| v_nuc[i] + v_h[i] + xc.v_up[i]).collect();
        let v_dn: Vec<f64> = (0..n).map(|i| v_nuc[i] + v_h[i] + xc.v_dn[i]).collect();
        solve_channel(grid, &v_up, &mut orb_up, &mut ev_up, settings.eig_tol);
        solve_channel(grid, &v_dn, &mut orb_dn, &mut ev_dn, settings.eig_tol);
        rho_out_up = density_from(grid, &orb_up);
        rho_out_dn = density_from(grid, &orb_dn);
        let diff: Vec<f64> = (0..n).map(|i| (rho_out_up[i] - rho_in_up[i]).abs() + (rho_out_dn[i] - rho_in_dn[i]).abs()).collect();
        delta_rho = grid.integrate(&diff);
        if delta_rho < settings.tol { break; }
        for i in 0..n {
            rho_in_up[i] += settings.mixing * (rho_out_up[i] - rho_in_up[i]);
            rho_in_dn[i] += settings.mixing * (rho_out_dn[i] - rho_in_dn[i]);
        }
    }
    if delta_rho >= settings.tol { return Err(QmError::ScfNotConverged(iterations, delta_rho)); }

    // energy from the output density and orbitals
    let rho: Vec<f64> = rho_out_up.iter().zip(&rho_out_dn).map(|(a, b)| a + b).collect();
    poisson.solve(grid, &rho, &mut v_h, 1e-10, 5000);
    let mut t = vec![0.0; n];
    let mut kinetic_e = 0.0;
    for psi in orb_up.iter().chain(orb_dn.iter()) { kinetic(grid, psi, &mut t); kinetic_e += grid.dot(psi, &t); }
    let xc = lsda_pw92(&rho_out_up, &rho_out_dn);
    let energy = {
        let external = grid.dot(&rho, &v_nuc);
        let hartree = hartree_energy(grid, &rho, &v_h);
        let xc_e = grid.integrate(&xc.energy_density);
        Energy { kinetic: kinetic_e, external, hartree, xc: xc_e, nuclear: e_nn, total: kinetic_e + external + hartree + xc_e + e_nn }
    };
    let boundary_density = max_on_faces(grid, &rho);
    let mut warnings = Vec::new();
    if boundary_density > 1e-5 { warnings.push(format!("density on box faces is {boundary_density:.2e}: box too small")); }

    Ok((KsState { orbitals_up: orb_up, orbitals_dn: orb_dn, evals_up: ev_up, evals_dn: ev_dn,
                  rho_up: rho_out_up, rho_dn: rho_out_dn, v_nuc, v_h },
        ScfStats { iterations, delta_rho, energy, boundary_density, warnings }))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm scf --release`
Expected: 7 passed (release build; debug is ~10× slower). Record the extrapolated H energy, the egg-box ΔE, and cold/warm iteration counts in the commit message. If `hydrogen_energy_extrapolated_to_point_nucleus` misses by more than 5 mHa, check (in order): Poisson tolerance, eigensolver tolerance, grid spacing convergence (rerun at h = 0.2) — do not widen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): spin-polarized Kohn-Sham SCF with LSDA; hydrogen atom validated"
```

---

### Task 10: `qm::forces` — Hellmann-Feynman forces **[Fable]**

**Files:**
- Create: `crates/qm/src/forces.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod forces; pub use forces::forces;`)

**Interfaces:**
- Consumes: `Grid`, `scf::KsState`, `nuclear::Nucleus`, `QmError`.
- Produces:
  ```rust
  /// Force on each nucleus (Hartree/Bohr) from the converged state. Includes nuclear–nuclear repulsion.
  pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError>;
  ```

Derivation (module doc comment): E_ext = ∫ρ(r) Σ_I V_I(r − R_I) dr with V_I(d) = −Z_I erf(d/(σ√2))/d. Since the grid basis does not move with R_I there is no Pulay term, so F_I = −∂E/∂R_I = −∫ρ ∂V_I/∂R_I dr = +∫ρ ∇_r V_I dr (V_I depends on r − R_I). With g(d) = erf(d/(σ√2))/d, g′(d) = [ (2/√π)(d/(σ√2)) e^{−d²/(2σ²)} − erf(d/(σ√2)) ]/d², ∇_r V_I = −Z_I g′(d) (r − R_I)/d, and g′(0) = 0 (g is even in d). Nuclear part: F_I^{nn} = Σ_{J≠I} Z_I Z_J (R_I − R_J)/|R_I − R_J|³.

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::nuclear::Nucleus;
    use crate::scf::{scf, ScfSettings, System};

    fn h2(r: f64) -> System {
        System::new(vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }], 0, None)
    }

    fn energy_and_force(g: &Grid, r: f64) -> (f64, Vec<[f64; 3]>) {
        let mut s = ScfSettings::for_grid(g);
        s.tol = 1e-9; s.eig_tol = 1e-8;
        let sys = h2(r);
        let (state, stats) = scf(g, &sys, &s, None).unwrap();
        (stats.energy.total, forces(g, &state, &sys.nuclei, s.sigma).unwrap())
    }

    #[test]
    fn h2_forces_are_equal_and_opposite() {
        let g = Grid::cube(49, 0.25);
        let (_, f) = energy_and_force(&g, 1.4);
        for a in 0..3 { assert!((f[0][a] + f[1][a]).abs() < 1e-6, "{f:?}"); }
        assert!(f[0][1].abs() < 1e-6 && f[0][2].abs() < 1e-6, "off-axis force {f:?}");
    }

    #[test]
    fn h2_force_matches_finite_difference_energy() {
        // E depends on the nuclei only through R = x₀ − x₁, so −∂E/∂x₀ = −dE/dR exactly.
        // Keep nucleus 1 fixed and move nucleus 0 by ±d so the finite difference is a true ∂E/∂x₀.
        let g = Grid::cube(49, 0.25);
        let d = 0.01;
        let mut s = ScfSettings::for_grid(&g);
        s.tol = 1e-9; s.eig_tol = 1e-8;
        let energy_at = |x0: f64| {
            let sys = System::new(vec![Nucleus { pos: [x0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None);
            scf(&g, &sys, &s, None).unwrap().1.energy.total
        };
        let f_fd = -(energy_at(0.7 + d) - energy_at(0.7 - d)) / (2.0 * d);
        let (_, f) = energy_and_force(&g, 1.4);
        assert!((f[0][0] - f_fd).abs() < 2e-3, "HF {} vs FD {}", f[0][0], f_fd);
    }

    #[test]
    fn force_sign_pushes_toward_equilibrium() {
        let g = Grid::cube(49, 0.25);
        let (_, f_short) = energy_and_force(&g, 1.0); // compressed: nuclei repel ⇒ F_x0 > 0
        let (_, f_long) = energy_and_force(&g, 2.5);  // stretched: attract ⇒ F_x0 < 0
        assert!(f_short[0][0] > 0.0, "{f_short:?}");
        assert!(f_long[0][0] < 0.0, "{f_long:?}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p qm forces --release`
Expected: compile error — `forces` not defined.

- [ ] **Step 3: Implement**

```rust
//! Hellmann-Feynman forces on Gaussian-smeared nuclei (derivation in the task description).
use crate::error::QmError;
use crate::grid::Grid;
use crate::nuclear::Nucleus;
use crate::scf::KsState;
use std::f64::consts::PI;

fn g_prime(d: f64, sigma: f64) -> f64 {
    if d < 1e-8 { return 0.0; }
    let u = d / (sigma * 2f64.sqrt());
    ((2.0 / PI.sqrt()) * u * (-u * u).exp() - libm::erf(u)) / (d * d)
}

pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError> {
    let rho = state.rho();
    let mut out = vec![[0.0f64; 3]; nuclei.len()];
    for (i, nuc) in nuclei.iter().enumerate() {
        let z = nuc.z as f64;
        // electronic part: F = ∫ ρ ∇_r V_I,  ∇_r V_I = −Z g′(d) (r − R)/d
        let mut comp = [vec![0.0; grid.len()], vec![0.0; grid.len()], vec![0.0; grid.len()]];
        for p in 0..grid.len() {
            let r = grid.point(p);
            let dv = [r[0] - nuc.pos[0], r[1] - nuc.pos[1], r[2] - nuc.pos[2]];
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            if d < 1e-8 { continue; }
            let gp = -z * g_prime(d, sigma) / d;
            for a in 0..3 { comp[a][p] = rho[p] * gp * dv[a]; }
        }
        for a in 0..3 { out[i][a] = grid.integrate(&comp[a]); }
        // nuclear–nuclear part
        for (j, other) in nuclei.iter().enumerate() {
            if i == j { continue; }
            let dv = [nuc.pos[0] - other.pos[0], nuc.pos[1] - other.pos[1], nuc.pos[2] - other.pos[2]];
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            if d < 1e-8 { return Err(QmError::CoincidentNuclei(i.min(j), i.max(j))); }
            let s = z * other.z as f64 / (d * d * d);
            for a in 0..3 { out[i][a] += s * dv[a]; }
        }
    }
    Ok(out)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p qm forces --release`
Expected: 3 passed. Record the HF-vs-FD discrepancy in the commit message; it is the egg-box/SCF-tolerance floor the GPU plan must also meet.

- [ ] **Step 5: Commit**

```bash
git add crates/qm
git commit -m "feat(qm): Hellmann-Feynman forces validated against finite-difference energies"
```

---

### Task 11: `validate` — physics oracle suite and cited references **[Opus]**

**Files:**
- Create: `crates/validate/tests/oracles.rs`, `crates/validate/references.md`
- Modify: `crates/validate/src/lib.rs` (add `pub mod extrapolate;` with the σ→0 helper below)

**Interfaces:**
- Consumes: `qm::{Grid, Nucleus, System, ScfSettings, scf, forces}`.
- Produces: `validate::extrapolate::to_zero_sigma(sigmas: &[f64; 3], energies: &[f64; 3]) -> f64` (Lagrange interpolation in σ² at σ² = 0), reused by M2/M3 validation.

These tests are slower than unit tests (tens of seconds each in release). They are the oracle suite that every later milestone must keep green: run with `cargo test -p validate --release`.

- [ ] **Step 1: Write the references file**

```markdown
# Reference values used by the validation suite

All energies in Hartree, lengths in Bohr unless stated. Each entry names the test that uses it.

| Quantity | Value | Source | Used by |
|---|---|---|---|
| Bohr radius | 0.529177210903 Å | CODATA 2018 | `units` |
| Hartree | 27.211386245988 eV | CODATA 2018 | `units` |
| H atom, LSD total energy | −0.478671 | NIST "Atomic Reference Data for Electronic Structure Calculations", Kotochigova, Levine, Shirley, Stiles, Clark (1997), LSD column | `h_atom_lsda` |
| He atom, LSD total energy | −2.834836 | same | `he_atom_lsda` |
| Li atom, LSD total energy | −7.335195 | same | `li_atom_lsda` |
| H₂ LSDA equilibrium bond length | 1.446 Bohr (0.765 Å) | NIST CCCBDB, SVWN/LSDA with a large basis set; LSDA overbinds vs experiment 1.401 Bohr | `h2_bond_length` |
| PW92 ε_c(r_s=2, ζ=0) | −0.04476 | Perdew & Wang, PRB 45, 13244 (1992), Table I / libxc `LDA_C_PW` | `qm::xc` tests |
| Isotope masses | AME2020 | Wang et al., Chinese Phys. C 45, 030003 (2021) | `units` |

Approximation gaps that are *expected* and must not be "fixed":
- LSD H atom is 21 mHa above the exact −0.5 Ha: self-interaction error of the functional.
- LSDA H₂ bond length is ~0.045 Bohr longer than experiment: LSDA overbinding.
- Gaussian nuclear smearing raises energies by ≈ (2π/3)·Z·ρ(R)·σ²; the suite extrapolates σ → 0 rather than tolerating it.
```

- [ ] **Step 2: Write the extrapolation helper and failing oracle tests**

```rust
// crates/validate/src/lib.rs
//! Headless validation suite. See references.md for every number used here.
pub mod extrapolate {
    /// Quadratic-in-σ² Lagrange interpolation evaluated at σ² = 0.
    pub fn to_zero_sigma(sigmas: &[f64; 3], energies: &[f64; 3]) -> f64 {
        let x: Vec<f64> = sigmas.iter().map(|s| s * s).collect();
        energies[0] * (x[1] * x[2]) / ((x[0] - x[1]) * (x[0] - x[2]))
            + energies[1] * (x[0] * x[2]) / ((x[1] - x[0]) * (x[1] - x[2]))
            + energies[2] * (x[0] * x[1]) / ((x[2] - x[0]) * (x[2] - x[1]))
    }
}
```

```rust
// crates/validate/tests/oracles.rs
use qm::{forces, scf, Grid, Nucleus, ScfSettings, System};
use validate::extrapolate::to_zero_sigma;

fn atom(z: u8) -> System { System::new(vec![Nucleus { pos: [0.0; 3], z }], 0, None) }

fn energy(grid: &Grid, sys: &System, sigma: f64) -> f64 {
    let mut s = ScfSettings::for_grid(grid);
    s.sigma = sigma; s.tol = 1e-7; s.eig_tol = 1e-7;
    scf(grid, sys, &s, None).expect("SCF").1.energy.total
}

fn extrapolated(grid: &Grid, sys: &System, sigmas: [f64; 3]) -> f64 {
    let e = [energy(grid, sys, sigmas[0]), energy(grid, sys, sigmas[1]), energy(grid, sys, sigmas[2])];
    to_zero_sigma(&sigmas, &e)
}

#[test]
fn h_atom_lsda() {
    let g = Grid::cube(61, 0.2); // 12 Bohr
    let e = extrapolated(&g, &atom(1), [0.2, 0.25, 0.3]);
    assert!((e - (-0.478671)).abs() < 4e-3, "H LSD: got {e}, reference −0.478671");
}

#[test]
fn he_atom_lsda() {
    let g = Grid::cube(61, 0.2);
    let e = extrapolated(&g, &atom(2), [0.2, 0.25, 0.3]);
    assert!((e - (-2.834836)).abs() < 1.5e-2, "He LSD: got {e}, reference −2.834836");
}

#[test]
fn li_atom_lsda() {
    let g = Grid::cube(71, 0.2); // 14 Bohr: Li 2s is diffuse
    let e = extrapolated(&g, &atom(3), [0.2, 0.25, 0.3]);
    assert!((e - (-7.335195)).abs() < 4e-2, "Li LSD: got {e}, reference −7.335195");
}

#[test]
fn h2_bond_length() {
    // Parabola through E(R) at three R around the minimum; LSDA R_e = 1.446 Bohr.
    let g = Grid::cube(49, 0.25);
    let rs = [1.35, 1.45, 1.55];
    let e: Vec<f64> = rs.iter().map(|&r| {
        let sys = System::new(vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }], 0, None);
        energy(&g, &sys, 0.25)
    }).collect();
    // vertex of the parabola through three equally spaced points
    let h = rs[1] - rs[0];
    let r_min = rs[1] - h * (e[2] - e[0]) / (2.0 * (e[2] - 2.0 * e[1] + e[0]));
    assert!((r_min - 1.446).abs() < 0.05, "H2 R_e = {r_min} Bohr, reference 1.446 (energies {e:?})");
}

#[test]
fn h2_force_vanishes_at_equilibrium() {
    let g = Grid::cube(49, 0.25);
    let mut s = ScfSettings::for_grid(&g);
    s.tol = 1e-9; s.eig_tol = 1e-8;
    let sys = System::new(vec![Nucleus { pos: [0.723, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.723, 0.0, 0.0], z: 1 }], 0, None);
    let (state, _) = scf(&g, &sys, &s, None).unwrap();
    let f = forces(&g, &state, &sys.nuclei, s.sigma).unwrap();
    assert!(f[0][0].abs() < 1e-2, "residual force at R_e: {f:?}");
}
```

- [ ] **Step 3: Run the suite**

Run: `cargo test -p validate --release`
Expected: 5 passed. If an atom test misses: first confirm `qm::scf` unit tests still pass, then rerun that atom at h = 0.15 to see whether the miss is grid error (shrinks) or a bug (doesn't). Report numbers; do not change tolerances or references.

- [ ] **Step 4: Record the first baseline**

Create `crates/validate/last-run.md` by hand from the test output (quantity, computed, reference, tolerance, pass/fail, grid, σ list). This file is gitignored; the numbers go in the commit message instead.

- [ ] **Step 5: Commit**

```bash
git add crates/validate
git commit -m "test(validate): LSDA atom energies and H2 bond length oracle suite with cited references"
```

---

## Self-review

- **Spec coverage (M1 scope):** §5 architecture → Tasks 1–3; §6 grid/stencil/nuclear potential/Poisson/XC/eigensolver/SCF/forces → Tasks 4–10; §9 layers 1–2 → every task's tests plus Task 11. Out of M1 scope by design: GPU (M2), dynamics/thermostat (M3), tooling/rendering (M4), multigrid preconditioner and DIIS mixing (M2 performance work; linear mixing and Jacobi-PCG are sufficient for the oracle suite).
- **Review Focus coverage:** (1) egg-box → `egg_box_ripple_is_small` (Task 9); (2) odd electron counts → `lithium_runs_with_unequal_spin_channels`, `hydrogen_converges_and_conserves_charge` (Task 9); (3) small box → `small_box_raises_energy_and_warns` (Task 9); (4) zero density → `zero_density_gives_zero_not_nan` (Task 6); (5) coincident nuclei → `coincident_nuclei_is_an_error` (Task 7).
- **Type consistency:** `Grid::{cube,len,idx,point,dv,integrate,dot,sample}` used identically in Tasks 4–11; `Nucleus { pos, z }` throughout; `ScfSettings::for_grid` + field overrides; `scf(...) -> Result<(KsState, ScfStats), QmError>`; `forces(grid, &state, &nuclei, sigma)`.
