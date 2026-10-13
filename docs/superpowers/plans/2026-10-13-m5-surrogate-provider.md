# M5 — Surrogate ("estimated") Force Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second `ForceProvider` — a linear ACE-style potential fitted to energies and forces produced by this project's own DFT solver — that runs MD at ≥ 10⁴ steps/s on small molecules, reproduces the DFT vibration frequency it was taught to 0.5 %, reports a per-atom uncertainty, and falls back to DFT (adding the sample to its training set) whenever it is outside what it knows. Switchable in the viewer as "estimated".

**Architecture:** New crate `surrogate` (pure f64, depends on `units` only): `basis` (Chebyshev radial × cosine cutoff, real solid harmonics l ≤ 3, all with analytic gradients), `descriptor` (species-channelled atomic basis A, body-2 and body-3 invariants B with a total-degree cap, and dE/dr by the chain rule), `fit` (sample store, isolated-atom reference energies, standardised ridge least squares on energies + forces with evidence-estimated noise σ_n and ridge λ, Bayesian predictive variance), `model` (coefficients, scaling, Σ, provenance; binary sidecar). In `sim`: `SurrogateKs: ForceProvider` wrapping a `Model` and a teacher provider with the fallback policy, `EvalStats.{source, uncertainty, teacher_gap}`, and one-surface energy accounting in `Dynamics` (ΔE_refit booked like thermostat work). In `app`: an "estimated" solver option, a sampling pass on the worker, readouts, sidecar save/load. Validation in `validate/tests/surrogate.rs`.

**Tech Stack:** Rust; no new crates (`rand` for sampling via the workspace; our own f64 Cholesky). Fable for everything numerical; Opus for worker/UI/files/perf.

**Spec:** `docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md` (reviewed: `docs/superpowers/notes/2026-10-13-m5-spec-review.md` — its MUST items are already in the spec and are binding here). Parent: `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` §5.1, §7, §11. Inputs: M4 handoff (to be written at merge) — H₂ at σ = h = 0.25: R_e 1.789 Bohr, k 0.200 Ha/Bohr², ω_MD 3237 cm⁻¹; egg-box ΣF ≈ 3e-4 Ha/Bohr; path scatter 6e-7 Ha / 5e-6 Ha/Bohr; H₂O at σ = 0.25 is near-linear and its warm SCF stalls (second oracle is H₃⁺).

**Model tags:** **[Fable]** Tasks 1, 2, 3, 4, 6, 7, 10; **[Opus]** Tasks 5, 8, 9. Fable→Opus fallback on dispatch failure (user ruling).

## Global Constraints

- Rule 1: the only per-element inputs are Z and isotope mass. The model's per-species coefficient channels are *learned* from solver data; isolated-atom reference energies E_atom(Z) are *computed* by the solver; sampling ranges are derived from the scene's own geometry — never from an element radius/bond table.
- Rule 2: no ML or linear-algebra frameworks; least squares, Cholesky, spherical harmonics and their gradients are hand-written and tested.
- Atomic units everywhere inside `surrogate` and `sim`; f64 throughout the model (rule 4).
- Energy–force consistency is non-negotiable: forces are the exact analytic gradient of the model energy (ΣF = 0 and Σ r×F = 0 to 1e-12; FD agreement 1e-8 relative).
- The surrogate never answers for a (species, charge, multiplicity, grid, σ) combination it has no samples for — unconditional fallback.
- One energy surface for bookkeeping: `E_conserved` uses the surrogate energy even on fallback steps; the DFT − model difference is `teacher_gap`; a refit's energy jump is booked as `ΔE_refit`.
- Validation targets are tiered (H₂: 1e-5 Ha / 2e-4 Ha/Bohr / ω 0.5 %; H₃⁺: 1e-4 Ha / 2e-3 Ha/Bohr / 2 %); never loosen — investigate.
- Timings are reported, never asserted (R3 from M4), except documented cadence invariants.

## Review Focus

1. **A wrong invariant that still fits:** with real Y_lm the body-3 contraction must be Σ_m A_{n1lm}A_{n2lm}; the complex-basis form fits training data and breaks under rotation. Pinned in Task 6 (random-rotation invariance of every B_k to 1e-12) and Task 1 (addition theorem).
2. **Forces not the gradient of the energy** (a sign in a Y_lm derivative): NVE "drift" that is really inconsistency. Pinned in Tasks 1, 2, 6 (FD vs analytic, ΣF, torque) and Task 7 (10 ps NVE < 1e-6 Ha/ps).
3. **Overconfident uncertainty:** σ_n taken from the 5e-6 path scatter instead of the ~3e-4 egg-box/SCF-lag floor → the model never falls back. Pinned in Task 3 (evidence-estimated σ_n; calibration test: empirical coverage of ±2σ on held-out samples ≥ 90 %) and Task 4 (OOD query returns Fallback).
4. **The 1 Ha cliff:** fitting total energies instead of binding energies sends E → 0 beyond the trained window. Pinned in Task 3 (reference energies; a two-atom query at r > r_c returns Σ E_atom to 1e-10).
5. **Bookkeeping across fallbacks and refits that masquerades as drift:** pinned in Task 4 (E_conserved continuous across a forced refit to 1e-12; teacher_gap carries the surface difference) and Task 7.

## File Structure

```
crates/surrogate/Cargo.toml                 # deps: units (+ dev: rand)
crates/surrogate/src/lib.rs                 # re-exports; crate doc = the model equations
crates/surrogate/src/basis.rs               # Task 1: radial R_n f_c, real solid harmonics S_lm, gradients
crates/surrogate/src/descriptor.rs          # Task 2 (body-2) + Task 6 (body-3): A, B, dB/dr, energy_and_forces
crates/surrogate/src/linalg.rs              # Task 3: f64 Cholesky, triangular solves, standardisation
crates/surrogate/src/fit.rs                 # Task 3: SampleStore, references, ridge fit, evidence, variance
crates/surrogate/src/model.rs               # Task 3: Model, Prediction, predict(); Task 5: sidecar (de)serialisation
crates/sim/src/provider.rs                  # Task 4: SurrogateKs, EvalStats.{source, uncertainty, teacher_gap}
crates/sim/src/dynamics.rs                  # Task 4: ΔE_refit accounting hook; Task 5: sampling driver helpers
crates/app/src/sim.rs, ui.rs, viewer.rs     # Task 8: "estimated" mode, sampling pass, readouts; Task 9: cadence/perf
crates/validate/tests/surrogate.rs          # Task 7
crates/validate/references.md               # Task 7: "Surrogate (M5)" section
docs/superpowers/notes/2026-10-13-m5-handoff.md   # Task 10
```

---

### Task 1: `surrogate::basis` — radial basis with cutoff, real solid harmonics, gradients **[Fable]**

**Files:** Create `crates/surrogate/Cargo.toml`, `crates/surrogate/src/{lib.rs, basis.rs}`; modify root `Cargo.toml` (member).

**Interfaces (produces):**
```rust
/// Scaled distance x(r) = 2·(1 − r/r_cut) − 1 ∈ [−1, 1) on (0, r_cut]; cutoff f_c(r) = ½(1 + cos(π r / r_cut)) for r < r_cut, else 0.
pub fn cutoff(r: f64, r_cut: f64) -> (f64, f64);                         // (f_c, df_c/dr)
/// R_n(r)·f_c(r) for n = 0..n_max (Chebyshev T_n in x(r)); returns values and d/dr. r > r_cut → zeros.
pub fn radial(n_max: usize, r: f64, r_cut: f64, out: &mut [f64], d_out: &mut [f64]);
/// Real solid harmonics S_lm(v) = |v|^l Y_lm(v̂) (Racah/"regular" normalisation, Σ_m S_lm(â)S_lm(b̂) = |a|^l|b|^l P_l(â·b̂)) for l ≤ l_max, index l² + l + m; polynomial in (x, y, z) so gradients are exact; out.len() == (l_max+1)².
pub fn solid_harmonics(l_max: usize, v: [f64; 3], out: &mut [f64], grad: &mut [[f64; 3]]);
pub fn lm_index(l: usize, m: isize) -> usize;
pub const L_MAX_SUPPORTED: usize = 4;
```
Derivation (module doc): why solid harmonics (polynomial, no 1/r singularity, gradients tabulated — spec review B2), the recursion used (Cartesian recursion for real S_lm up to l = 4 written out), the Racah normalisation and the addition theorem with the (2l+1)/4π factor absorbed, and the role of f_c in making E and F continuous at r_cut.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn cutoff_is_smooth_and_vanishes_at_r_cut() { /* f_c(0)=1, f_c(r_cut)=0, df/dr(r_cut)=0; FD of f_c vs df/dr 1e-9 at 20 random r */ }
#[test] fn radial_matches_chebyshev_recurrence_and_its_derivative() { /* T_n(x(r))·f_c vs direct cos(n·acos x)·f_c to 1e-12; FD vs d_out 1e-8 relative; zeros beyond r_cut */ }
#[test] fn solid_harmonics_satisfy_the_addition_theorem() { /* Σ_m S_lm(a)S_lm(b) == |a|^l|b|^l P_l(â·b̂) to 1e-12 for l ≤ 4, 50 random pairs */ }
#[test] fn solid_harmonic_gradients_match_finite_differences() { /* every S_lm, l ≤ 4, 30 random points: |∇S − FD| < 1e-8·max(1,|∇S|) */ }
#[test] fn solid_harmonics_rotate_as_a_representation() { /* for random rotation R: Σ_m S_lm(Rv)² == Σ_m S_lm(v)² to 1e-12 (norm invariance per l) */ }
```
- [ ] **Steps 2–5:** implement → `cargo test -p surrogate` (debug is fine; pure f64) green → commit `feat(surrogate): radial basis with cosine cutoff and real solid harmonics with analytic gradients` + Fable trailer.

---

### Task 2: body-2 descriptors, species channels, energy and forces **[Fable]**

**Files:** Create `crates/surrogate/src/descriptor.rs`, `crates/surrogate/src/model.rs` (Model struct + predict only; fit comes in Task 3); modify `lib.rs`.

**Interfaces (produces):**
```rust
#[derive(Clone, Debug, PartialEq)]
pub struct Basis { pub n_max: usize, pub l_max: usize, pub r_cut: f64, pub degree_cap: usize, pub body3: bool, pub species: Vec<u8> /* sorted, unique Z */ }
impl Basis { pub fn new(species: &[u8]) -> Basis /* n_max 8, l_max 3, r_cut 6.0, degree_cap 10, body3 false (Task 6 flips the default after measuring) */; pub fn n_features(&self) -> usize; pub fn species_index(&self, z: u8) -> Option<usize>; pub fn feature_names(&self) -> Vec<String> /* for debugging and the sidecar header */ }
/// Per-atom descriptor vector B_i (len n_features) and its gradient w.r.t. every atom position that influences it.
pub struct AtomDescriptor { pub b: Vec<f64>, pub grad: Vec<(usize /* atom j (including i) */, Vec<[f64; 3]> /* dB_k/dr_j, len n_features */)> }
pub fn describe(basis: &Basis, pos: &[[f64; 3]], z: &[u8]) -> Result<Vec<AtomDescriptor>, DescribeError /* UnknownSpecies(z) */>;
pub struct Model { pub basis: Basis, pub coeffs: Vec<Vec<f64>> /* [centre species][feature], in STANDARDISED feature space */, pub mean: Vec<f64>, pub scale: Vec<f64> /* feature standardisation */, pub ref_energy: Vec<f64> /* E_atom per species index */, pub sigma_n: f64, pub lambda: f64, pub sigma_inv_chol: Option<Vec<f64>> /* Cholesky factor of ΦᵀΦ/σ_n² + λI, lower, row-major P×P, for the variance */, pub n_samples: usize, pub provenance: Provenance }
#[derive(Clone, Debug, PartialEq)] pub struct Provenance { pub grid_n: usize, pub h: f64, pub sigma: f64, pub tol: f64, pub charge: i32, pub multiplicity: Option<u32>, pub teacher: String }
pub struct Prediction { pub energy: f64 /* total, incl. Σ E_atom */, pub forces: Vec<[f64; 3]>, pub energy_std: Option<f64>, pub force_std: Option<Vec<f64>> /* per atom, Ha/Bohr */ }
impl Model { pub fn predict(&self, pos: &[[f64; 3]], z: &[u8], with_variance: bool) -> Result<Prediction, DescribeError>; }
```
Body-2 features: for centre species s and neighbour species t, B_{n,t} = Σ_{j: Z_j = t} R_n(r_ij) f_c(r_ij) (the l = 0 atomic basis); n_features = n_max·S for body-2. E_i = ref_energy[s_i] + Σ_k c_{s_i,k}(B_k − mean_k)/scale_k. Forces by the chain rule with dR_n f_c/dr · r̂_ij, Newton's third law applied (each pair contributes to i and j).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn descriptors_are_translation_and_permutation_invariant() { /* random 5-atom 2-species cluster: translate → B identical to 1e-14; permute atoms → B permuted */ }
#[test] fn descriptor_gradients_match_finite_differences() { /* |dB/dr − FD| < 1e-8·max(1,|dB|) for every (i, k, j, component) */ }
#[test] fn forces_are_the_gradient_of_the_energy_and_conserve_momentum_and_torque() { /* random coefficients: FD energy vs forces 1e-8 rel; ΣF = 0 and Σ r×F = 0 to 1e-12 */ }
#[test] fn two_atoms_beyond_r_cut_give_the_reference_energies() { /* E == ref[s1] + ref[s2] to 1e-10, F = 0 */ }
#[test] fn unknown_species_is_an_error_not_a_guess() { /* z = 11 → Err(UnknownSpecies(11)) */ }
```
- [ ] **Steps 2–5:** implement → green → commit `feat(surrogate): body-2 species-channelled descriptors with analytic forces and reference energies` + Fable trailer.

---

### Task 3: sample store, reference energies, standardised ridge fit, evidence, predictive variance **[Fable]**

**Files:** Create `crates/surrogate/src/{linalg.rs, fit.rs}`; modify `model.rs` (variance in `predict`), `lib.rs`.

**Interfaces (produces):**
```rust
// linalg.rs — f64 only
pub fn cholesky_in_place(a: &mut [f64], n: usize) -> Result<(), LinalgError /* NotPositiveDefinite(k) */>;   // lower, row-major
pub fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]);
pub fn cholesky_logdet(l: &[f64], n: usize) -> f64;
// fit.rs
#[derive(Clone, Debug)] pub struct Sample { pub pos: Vec<[f64; 3]>, pub z: Vec<u8>, pub energy: f64 /* total E_KS */, pub forces: Vec<[f64; 3]>, pub tag: SampleTag /* LiveStep | Perturbation | HotBurst | Scan | Fallback | Reference */, pub weight: f64 }
pub struct SampleStore { pub provenance: Provenance, pub samples: Vec<Sample> }
impl SampleStore { pub fn new(provenance: Provenance) -> SampleStore; pub fn push(&mut self, s: Sample) -> Result<(), StoreError /* ProvenanceMismatch */>; pub fn len(&self) -> usize; }
pub struct FitOptions { pub lambda: Option<f64> /* None = evidence maximisation over a log grid */, pub sigma_n: Option<f64> /* None = from residuals */, pub force_weight: f64 /* relative weight of a force component vs an energy per atom, default 1.0 */, pub energy_per_atom: bool /* default true */ }
pub struct FitReport { pub rmse_energy_per_atom: f64, pub rmse_force: f64, pub sigma_n: f64, pub lambda: f64, pub log_evidence: f64, pub n_rows: usize, pub n_features: usize, pub condition_estimate: f64 }
/// Fit E_KS − Σ E_atom(Z) and forces jointly. `ref_energy[s]` must be given for every species in the store.
pub fn fit(store: &SampleStore, basis: &Basis, ref_energy: &[f64], opts: &FitOptions) -> Result<(Model, FitReport), FitError /* NoSamples | MissingReference(z) | Singular */>;
/// Incremental: accumulate (ΦᵀWΦ, ΦᵀWy) so a refit after new samples is one Cholesky; `Accumulator` is what the sidecar stores.
pub struct Accumulator { /* P×P, P, scalar sums, n_rows, per-feature running mean/var for standardisation */ }
impl Accumulator { pub fn new(basis: &Basis) -> Accumulator; pub fn add(&mut self, sample: &Sample, basis: &Basis, ref_energy: &[f64], opts: &FitOptions) -> Result<(), DescribeError>; pub fn solve(&self, opts: &FitOptions, provenance: &Provenance, ref_energy: &[f64]) -> Result<(Model, FitReport), FitError>; }
```
Derivation (module doc): the design matrix rows (one energy row per sample, 3N force rows with the −dB/dr features), standardisation (mean/scale from the energy-row features; force rows use scale only), ridge in standardised space, the Bayesian linear model with noise σ_n² and prior λ, predictive variance σ²(x) = σ_n² + φᵀ(ΦᵀΦ/σ_n² + λI)⁻¹φ (energy per atom; forces via the gradient features), evidence maximisation for λ (log grid 1e-8..1e2) and σ_n² = RSS/(n_rows − effective dof) — and why σ_n must NOT be the solver path scatter (egg-box 3e-4 Ha/Bohr and SCF lag are the real floor — spec review C2).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn cholesky_solves_a_random_spd_system() { /* 50×50 AᵀA + I; |Ax − b| < 1e-10; logdet vs product of pivots */ }
#[test] fn fit_recovers_a_synthetic_linear_model_exactly() { /* generate data from random coefficients on random clusters (no noise): recovered coefficients within 1e-10, rmse < 1e-12, variance ≈ σ_n² ≈ 0 */ }
#[test] fn fit_recovers_a_morse_dimer_to_1e6_with_body2() { /* synthetic Morse E(r) on 60 geometries r ∈ [1.0, 3.0], energies + forces: rmse_energy < 1e-6 Ha, rmse_force < 1e-5 Ha/Bohr; E(r > r_cut) = 2·E_atom */ }
#[test] fn noise_level_comes_from_residuals_not_from_a_constant() { /* add Gaussian noise σ = 3e-4 to forces: FitReport.sigma_n within 20 % of 3e-4; the predictive std at a training point ≥ 0.8·3e-4 */ }
#[test] fn predictive_variance_is_calibrated_on_held_out_samples() { /* 200 noisy samples, fit on 150: fraction of held-out energies within ±2σ ≥ 0.90 */ }
#[test] fn variance_grows_outside_the_trained_range() { /* Morse model trained on r ∈ [1.2, 2.2]: force_std at r = 3.0 ≥ 5× the std at r = 1.7 */ }
#[test] fn accumulator_refit_equals_a_batch_fit() { /* same samples: coefficients bitwise-close (1e-12) between Accumulator::solve and fit */ }
```
- [ ] **Steps 2–5:** implement → `cargo test -p surrogate` green → commit `feat(surrogate): ridge fit with evidence-estimated noise, reference energies and Bayesian predictive variance` + Fable trailer. Report the Morse numbers and the calibration coverage in the commit body.

---

### Task 4: `sim::SurrogateKs` — the provider, fallback policy, one-surface accounting **[Fable]**

**Files:** Modify `crates/sim/src/provider.rs`, `crates/sim/src/dynamics.rs`, `crates/sim/Cargo.toml` (`surrogate` dep), `crates/sim/src/lib.rs`.

**Interfaces (produces):**
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Source { Dft, Surrogate, Fallback }
// EvalStats gains:
pub source: Source,                 // Dft for CpuKs/GpuKs
pub uncertainty: Option<f64>,       // max per-atom force std (Ha/Bohr) when the surrogate answered or was consulted
pub teacher_gap: Option<f64>,       // DFT − model energy on a Fallback step (Ha)
pub struct FallbackPolicy { pub force_std_abs: f64 /* 2e-3 Ha/Bohr */, pub force_std_rel: f64 /* 0.10, capped so the threshold ≤ 1e-2 */, pub energy_std_per_atom: f64 /* 1e-3 Ha */, pub max_descriptor_distance: f64 /* OOD envelope in standardised feature space, default 3.0 */, pub variance_cadence: usize /* evaluate the O(P²) variance every k steps, k = 1 here; Task 9 tunes */ }
pub struct SurrogateKs { /* model: Option<surrogate::Model>, accumulator: surrogate::Accumulator, store: SampleStore, teacher: Box<dyn ForceProvider>, basis, ref_energy, policy, pending_refit: bool, epoch: u64, last_model_energy: Option<f64> */ }
impl SurrogateKs {
    pub fn new(teacher: Box<dyn ForceProvider>, policy: FallbackPolicy) -> SurrogateKs;
    pub fn basis(&self) -> Option<&Basis>; pub fn model(&self) -> Option<&Model>; pub fn samples(&self) -> usize; pub fn epoch(&self) -> u64;
    /// Teach from an existing sample (sampling pass, loaded sidecar). Reference energies are supplied by the caller (Task 5 computes them with the teacher on an isolated atom).
    pub fn add_sample(&mut self, s: Sample) -> Result<(), StoreError>; pub fn set_reference_energy(&mut self, z: u8, e: f64);
    /// Rebuild the model from the accumulator; returns the FitReport. Called by the worker between ticks (never inside evaluate).
    pub fn refit(&mut self) -> Result<FitReport, FitError>;
    pub fn set_policy(&mut self, p: FallbackPolicy);
}
impl ForceProvider for SurrogateKs { /* evaluate: no model or unseen (species, charge, mult) → teacher (Fallback, sample appended, pending_refit); else predict with variance per policy.cadence; if within policy → Surrogate; else teacher → Fallback + append + teacher_gap = E_dft − E_model. name() = "surrogate(<teacher>)". reset/discontinuity → teacher. */ }
// dynamics.rs
pub struct DynSnapshot { /* + */ pub source: Source, pub uncertainty: Option<f64>, pub teacher_gap: Option<f64>, pub model_epoch: u64, pub fallback_fraction_100: f64 }
```
One-surface accounting: on a Fallback step the integrator uses the DFT forces but `energy_ks` in the StepStats fed to the bookkeeping is the MODEL energy (the provider returns `ForceResult { energy: model_energy, forces: dft_forces, stats: { teacher_gap: Some(dft − model), … } }`); when `epoch` changes between two consecutive evaluates (a refit happened), `Dynamics` books ΔE_refit = E_new_model(r) − E_old_model(r) into the work term exactly like thermostat work (the provider reports `refit_jump: Option<f64>` in EvalStats for the first evaluate after a refit, computed by evaluating the old model once more at the same positions before dropping it).

- [ ] **Step 1: Failing tests** (CPU only: the teacher is `test_providers::Harmonic` or a scripted provider; plus one GPU test)
```rust
#[test] fn no_model_means_fallback_and_a_sample() { /* first evaluate → Source::Fallback, teacher called once, samples() == 1 */ }
#[test] fn window_trained_model_answers_inside_and_falls_back_outside() { /* H₂-like dimer with a Morse scripted teacher: add 40 scan samples r ∈ [1.2, 2.2], set refs, refit; evaluate at r = 1.6 → Surrogate, |E − teacher| < 1e-5, |F − teacher| < 1e-4; at r = 4.0 → Fallback, teacher_gap Some, samples()+1; refit; r = 4.0 → Surrogate */ }
#[test] fn unseen_species_or_charge_falls_back_unconditionally() { /* model trained on H₂ (charge 0); scene with charge +1 → Fallback regardless of variance */ }
#[test] fn energy_conserved_is_continuous_across_a_forced_refit() { /* Dynamics on the surrogate over the Harmonic teacher: 50 steps, add 20 samples + refit mid-run, 50 steps: |ΔE_conserved| < 1e-12 across the refit step (the jump is booked), teacher_gap None on Surrogate steps */ }
#[test] fn fallback_step_uses_dft_forces_but_books_the_model_energy() { /* scripted teacher with a known offset: position update equals the teacher-force Verlet step; energy_ks in the snapshot equals the model energy; teacher_gap equals the offset */ }
#[test] fn gpu_teacher_h2_round_trip() { /* GPU: GpuKs teacher at 33³ (release-gated): 12 scan samples + refit; a Surrogate evaluate at R_e agrees with a direct GpuKs evaluate within 1e-4 Ha / 2e-3 Ha/Bohr (body-2 only; Task 7 pins the tiered targets with body-3) */ }
```
- [ ] **Steps 2–5:** implement → `cargo test -p sim --release` green → commit `feat(sim): SurrogateKs force provider with Bayesian fallback to the DFT teacher and one-surface energy accounting` + Fable trailer.

---

### Task 5: sampling pass, reference solves, `.model` sidecar **[Opus]**

**Files:** Create `crates/sim/src/sampling.rs`; modify `crates/surrogate/src/model.rs` (sidecar), `crates/sim/src/lib.rs`, `crates/sim/Cargo.toml` (`rand` already present).

**Interfaces (produces):**
```rust
// sim::sampling
pub struct SamplingPlan { pub perturbations: usize /* 60 */, pub perturbation_sigma: f64 /* 0.15 Bohr */, pub hot_burst_steps: usize /* 200 */, pub hot_burst_t: f64 /* 2000 K */, pub scan_points_per_pair: usize /* 12 */, pub scan_range: (f64, f64) /* fractions of the CURRENT pair separation, (0.7, 1.8) — derived from the scene, rule 1 */, pub seed: u64 }
pub struct SamplingProgress { pub done: usize, pub total: usize, pub stage: &'static str, pub last_energy: f64, pub cancelled: bool }
/// Drives `teacher` over the plan, pushing samples into `surrogate`; polls `should_abort` between solves; computes and sets the isolated-atom reference energies first (one solve per species present, cached by provenance in a small in-memory map; the single atom is placed at the cell centre of the scene's grid). Returns the number of samples added.
pub fn run_sampling(scene: &Scene, plan: &SamplingPlan, teacher: &mut dyn ForceProvider, surrogate: &mut SurrogateKs, progress: &mut dyn FnMut(SamplingProgress), should_abort: &mut dyn FnMut() -> bool) -> Result<usize, ProviderError>;
// surrogate::model sidecar
impl Model { pub fn write_sidecar(&self, acc: &Accumulator, path: &Path) -> io::Result<()>; pub fn read_sidecar(path: &Path) -> io::Result<(Model, Accumulator)>; }
// format: magic "ASMODEL1", JSON header (Provenance, Basis, species, n_samples, FitReport summary, created-by), then little-endian f64 arrays (coeffs, mean, scale, ref_energy, accumulator) with lengths; bit-exact round trip.
```
Also: `DynCommand::{StartSampling(SamplingPlan), CancelSampling, ForgetModel, SaveModel(PathBuf), LoadModel(PathBuf)}` and `DynSnapshot.sampling: Option<SamplingProgress>` — the sampling runs INSIDE the worker's tick budget loop (one solve per tick slice) so the UI stays live and cancellable; while sampling, the dynamics are paused.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn sidecar_round_trips_bit_exact() { /* random Model + Accumulator → write → read → every f64 bitwise equal, header equal */ }
#[test] fn sampling_pass_yields_the_requested_counts_and_tags() { /* scripted fake teacher counting calls: H₂ scene, plan (10 perturbations, 20 burst steps, 6 scan points): samples == 10 + 20 + 6 + (1 pair) …, tags match, references set for Z = 1 from an isolated-atom call */ }
#[test] fn scan_ranges_come_from_the_scene_not_a_table() { /* two scenes with pair separations 1.4 and 2.8 Bohr: scan r ranges scale by exactly 2× */ }
#[test] fn cancelling_a_pass_leaves_a_usable_partial_set() { /* abort after 7 solves: samples() == 7 (+ refs), refit succeeds, no panic */ }
#[test] fn sampling_commands_flow_through_dynamics() { /* StartSampling → snapshot.sampling Some with progress; CancelSampling → cancelled; ForgetModel → model None */ }
```
- [ ] **Steps 2–5:** implement → `cargo test -p sim --release` and `cargo test -p surrogate` green → commit `feat(sim,surrogate): sampling pass with scene-derived scans, isolated-atom references, and a bit-exact .model sidecar` + Opus trailer.

---

### Task 6: body-3 descriptors with the real-harmonic invariant, degree cap, H₃⁺ fit **[Fable]**

**Files:** Modify `crates/surrogate/src/descriptor.rs`, `model.rs` (feature_names), `lib.rs`.

Body-3 features for centre species s: for species pair (t₁ ≤ t₂), radial pair (n₁, n₂) (symmetric when t₁ = t₂), and l ≤ l_max: B = Σ_m A_{n₁ l m, t₁} · A_{n₂ l m, t₂} with A_{nlm,t} = Σ_{j: Z_j = t} R_n(r_ij) f_c(r_ij) S_lm(r_ij)/|r_ij|^l (equivalently Y_lm — choose the solid-harmonic form and divide by |r|^l inside the radial part so gradients stay tabulated; document). Degree cap n₁ + n₂ + l ≤ degree_cap. Gradient dB/dr_j by the product rule through both A factors. `Basis::body3 = true` becomes the default if the H₃⁺ fit needs it (measure: body-2 vs body-3 rmse on the H₃⁺ fixture; record in the report and set the default accordingly).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn every_body3_feature_is_invariant_under_random_rotations_and_permutations() { /* 7-atom 3-species cluster, 20 random rotations: |B(Rr) − B(r)| < 1e-12·max(1,|B|); permutations → permuted B. THIS catches the complex-basis contraction (Review Focus 1) */ }
#[test] fn body3_gradients_match_finite_differences_and_conserve_momentum_and_torque() { /* 1e-8 rel; ΣF, Σ r×F < 1e-12 */ }
#[test] fn degree_cap_and_symmetric_pairs_give_the_documented_feature_count() { /* S = 2, n_max 8, l_max 3, cap 10 → the count the module doc derives; S = 4 → likewise */ }
#[test] fn h3plus_fixture_fits_to_the_tier_target() { /* stored fixture of 80 H₃⁺ geometries with DFT E/F (generated once with GpuKs at 33³ and committed under crates/surrogate/fixtures/ with provenance; ≤ 50 KB): body-3 fit rmse_energy < 1e-4 Ha, rmse_force < 2e-3 Ha/Bohr on 20 held-out geometries; report body-2's numbers too */ }
```
- [ ] **Steps 2–5:** implement → green → commit `feat(surrogate): body-3 invariants (real-harmonic contraction), total-degree cap, H3+ fit` + Fable trailer.

---

### Task 7: validation suite `validate/tests/surrogate.rs` **[Fable]**

**Files:** Create `crates/validate/tests/surrogate.rs`; modify `crates/validate/references.md` ("Surrogate (M5)"), `crates/validate/Cargo.toml` (`surrogate` dev-dep), `.gitignore` (`last-run-surrogate.md`).

Tests (release, GPU teacher = `GpuKs` 49³ σ = h = 0.25 tol 1e-6; skip without adapter; suite budget ≤ 10 min including sampling):
- `surrogate_reproduces_the_h2_frequency`: sampling pass on H₂ (scan 0.7–1.8× of 1.6 Bohr, 60 perturbations, 200-step 1000 K burst) → refit → surrogate-MD from the M4 start (R_e(σ) + 0.05 Bohr, dt 0.25 fs, 4000 steps = 1 ps): ω within 0.5 % of the M4 pinned DFT-MD value (read the constant from the M4 suite's reference section, cite it); E(R) on 20 fresh scan points within 1e-5 Ha, F within 2e-4 Ha/Bohr of fresh DFT.
- `surrogate_nve_is_tight`: 10 ps (40 000 steps) NVE on the surrogate: |drift| < 1e-6 Ha/ps; fluctuation equals the Verlet h²/4·E_mode to 10 %; rewind retraces to 1e-9 Bohr; steps/s reported.
- `surrogate_knows_what_it_does_not_know`: with the H₂ model: R = 4.0 Bohr → Fallback; charge +1 → Fallback; a third H added → Fallback (unseen size is allowed only if species seen — document the rule: fallback is by uncertainty; assert it is Fallback because the variance says so, print the std); after the fallbacks + refit, R = 4.0 → Surrogate within 1e-4 Ha.
- `h3plus_tier`: sampling pass on H₃⁺ (equilateral, relaxed on the σ = 0.25 surface by the M4 suite's bisection helper or 20 steps of damped MD) → body-3 model: E within 1e-4 Ha, F within 2e-3 Ha/Bohr on 20 held-out DFT points; the symmetric-stretch and bend frequencies from surrogate-MD within 2 % of DFT-MD (short DFT-MD runs, 100 steps each, zero crossings) — report.
- `active_learning_run_converges`: start with NO model, run 300 steps of H₂ MD through `SurrogateKs` with the GPU teacher and refits between steps: the fallback fraction over the last 100 steps < 10 %; teacher_gap on the last fallback < 1e-4 Ha; total DFT calls reported.
- `references.md` "Surrogate (M5)": every bound, what it measures, achieved numbers, speed table (steps/s surrogate vs DFT), citations (Drautz 2019 ACE; Bishop PRML ch. 3 Bayesian linear regression / evidence; MacKay 1992; Verlet).

- [ ] Steps: write → run → fill numbers (tighten where measured; never loosen) → commit `test(validate): surrogate suite — H2 frequency and surface reproduction, tight NVE, OOD fallback, H3+ tier, active learning` + Fable trailer.

---

### Task 8: viewer — "estimated" mode, sampling UI, readouts **[Opus]**

**Files:** Modify `crates/app/src/{sim.rs, ui.rs, viewer.rs, gfx.rs, scene.rs}`, `README.md`, `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md` (M5 section), `crates/app/presets/*.scene` unchanged.

Behaviour: solver radio gains **estimated** (enabled when a GPU or CPU teacher exists; builds `SurrogateKs::new(Box<GpuKs or CpuKs>)`); enabling with no model opens a sampling panel (plan fields with the defaults, Start/Cancel, progress bar + stage + last energy; dynamics paused while sampling, editing allowed); readouts: source per step (Surrogate green / Fallback amber / DFT blue), uncertainty vs threshold, samples, model epoch, last refit rmse, fallback fraction (last 100), teacher gap; the sparkline colours each sample by source; density: fade the last DFT cloud to 30 % with the label "density from the last DFT step (t = … fs)"; buttons: sample more, forget model, save model (next to the scene file as `<scene>.model`), load; the speed slider's effective range now matters (show fs/s). Switching teacher backend keeps the model only if the provenance matches (grid/σ/tol/charge/mult), else prompts to forget. README controls/status/what-to-try updated; screenshot `docs/images/surrogate_h2.png` with the sparkline coloured by source.

- [ ] **Step 1: Failing tests** (`sim.rs`, release-gated, GPU)
```rust
#[test] fn enabling_estimated_without_a_model_starts_a_sampling_pass() { /* snapshot.sampling Some → progresses → model epoch 1 → Play → steps with Source::Surrogate dominate (> 80 % over 200 steps) and fs/s > 50× the DFT value */ }
#[test] fn fallbacks_are_visible_in_snapshots() { /* teleport H₂ to R = 4 while estimated: next step Source::Fallback with teacher_gap Some */ }
#[test] fn model_sidecar_saves_and_loads_with_the_scene() { /* save → new worker → load → epoch and samples restored, first step Surrogate */ }
```
- [ ] **Steps 2–5:** implement → run the app yourself: sample H₂ (~1 min), press Play, watch it vibrate at ≥ 1000 fs/s, drag it far and see the fallback light up and the model learn; record the checklist; commit `feat(app): estimated mode — surrogate provider with sampling pass, uncertainty readouts, fallback visibility, model sidecars` + Opus trailer.

---

### Task 9: performance — variance cadence, species-count scaling, refit timing **[Opus]**

**Files:** Modify `crates/sim/src/provider.rs` (cadence), `crates/surrogate/src/{fit.rs, model.rs}` (spectral/low-rank Σ option if needed), `crates/surrogate/benches/` or an ignored test with a timing table.

Measure (report) steps/s for S = 1, 2, 4 species at N = 2, 6, 20 atoms with body-3 on, with variance every step vs every k = 10; refit time vs samples (P up to ~2 000); choose the default cadence and document; assert only the cadence invariant (never more than k steps between full variance checks, and always a full check on the first step after any edit/refit). If the O(P²) variance is the bottleneck at S = 4, implement the spectrally truncated Σ (top-r eigenpairs, r ≤ 64) behind a setting and measure the coverage loss on the calibration test.

- [ ] Steps: bench → cadence test → commit `perf(sim,surrogate): uncertainty cadence and timing table` + Opus trailer, with the table in the commit body and `references.md` speed section.

---

### Task 10: physics review + M5 handoff **[Fable]**

Read-only review of the whole branch against the spec and the review MUSTs (as the final whole-branch review of the SDD loop), then write `docs/superpowers/notes/2026-10-13-m5-handoff.md`: results table (H₂/H₃⁺ tiers achieved, speeds, fallback statistics), the honest limits (non-locality, charge/spin channels, egg-box, extrapolation), deferred items, and the inputs to M6 (tools) and the periodic-cell milestone.

---

## Self-review

- **Spec coverage:** §3 model → Tasks 1, 2, 6; §4 data/fit/uncertainty → Tasks 3, 5; §5 provider/fallback/accounting → Task 4; §6 UI → Task 8; §7 validation → Task 7; §8 risks → Review Focus 1–5 and Task 9; review MUSTs: A3 (Task 6 rotation test), A4 (Tasks 2/3 references), C1/C2 (Task 3 standardisation, evidence σ_n, calibration), A2/C4 (Task 6 degree cap, Task 9 cadence), D1 (Task 4 one-surface + teacher_gap + ΔE_refit), E3 (Task 7 H₃⁺ tier), E1/B4 (Task 5 scene-derived scans; Tasks 2/6 ΣF/torque).
- **Placeholder scan:** none; every test names its numbers.
- **Type consistency:** `Basis`, `Model`, `Prediction`, `Sample`, `SampleStore`, `Accumulator`, `FitOptions`, `FitReport`, `Provenance` defined in Tasks 2/3 and used by 4/5/7/8; `Source`, `FallbackPolicy`, `SurrogateKs` (Task 4) used by 5/7/8/9; `SamplingPlan`/`SamplingProgress` (Task 5) used by 8; `EvalStats.{source, uncertainty, teacher_gap, refit_jump}` (Task 4) used by `Dynamics` and the app.
- **Known risks:** the solid-harmonic recursion to l = 4 is finicky (Task 1's addition-theorem test is the guard); H₃⁺'s fixture generation needs the GPU once (commit the fixture with provenance); the sampling pass on the viewer's GPU competes with the user's own interaction (the pass is cancellable and runs in tick slices).
