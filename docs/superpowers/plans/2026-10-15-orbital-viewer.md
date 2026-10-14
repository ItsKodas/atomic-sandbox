# Orbital Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under View ▸ Density ▸ show, three new modes next to today's density:
- **Orbitals**: a list of the occupied Kohn–Sham orbitals plus the lowest few unoccupied ones, with ε, spin and occupation. A picked orbital is drawn as a ± phase isosurface or as a |ψ|² volume.
- **All orbitals coloured**: every orbital's density in its own colour, so inner shells and outer lobes show together.
- **Spin density**: ρ↑ − ρ↓ in two colours.

Colours, signs and orientations stay steady from step to step. A lone atom's shells keep their names however the grid splits them. Nothing about the view changes the solve.

**Architecture:**
- `qm::orbitals` holds the display numerics in f64: lone-atom shells by angular character, clusters by energy, Procrustes alignment, n l m labels, continuity tracks, signs and colours, spin summary, the colour-volume reference.
- `qm_gpu::orbitals::OrbitalView` produces the same products on the solver's device. It does one small scalar readback per update and copies the orbital (or a weighted set) GPU-to-GPU into display textures.
- `sim::provider` gets the opaque `OrbitalRequest`/`OrbitalDisplay` types and three default `ForceProvider` hooks. `providers::{CpuKs, GpuKs, SurrogateKs}` fill them; estimated mode has a single producer.
- A display-only eigensolve supplies the unoccupied orbitals.
- `render::volume` gains signed modes and a colour-volume pipeline. The app gains the selector, an Orbitals dock tab and the honesty text.
- With M7, core shells are reconstructed analytically from the generator's spin-restricted all-electron atom.

**Tech Stack:** Rust, wgpu/WGSL, egui, nalgebra (the small SVDs). No new crates. The f32→f16 conversion for the CPU colour upload uses `half`, already in the lock file via wgpu (plumbing, rule 2).

**Spec:** `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md` (§ numbers below refer to it). It was physics-reviewed on 2026-10-15; the review text is `docs/superpowers/notes/2026-10-15-orbital-viewer-spec-review.md`, and spec §14 maps each MUST to its change. Parent: `2026-10-12-atomic-sandbox-design.md` §8. Task 8 also builds on `2026-10-14-pseudopotential-design.md` §4.1–4.3, §4.6, §5.6 and §6.5.

**Scheduling:** start after **M6 (tools) merges into main**. File paths below are post-M6: `crates/providers/`, `crates/app/src/worker/`. If M6 moved something again, follow the code and note it in the task report. Independent of the user-data feature (settings profiles, model library). **Task 0 runs first**: its numbers are inputs to Tasks 1–5. **Task 8 is gated on M7 (pseudopotentials)**. If this plan finishes first, Task 8 is carried into M7's plan as an appendix task and this plan merges without it.

**Model tags** (CLAUDE.md "Model assignment"):
- **[Fable]**: Tasks 1, 4, 5, 8 (the alignment, phase and matching numerics, the display eigensolve, the oracles, the core reconstruction). Opus is the fallback while Fable is unavailable; the task report then says so, and the physics-reviewer pass is mandatory before merge.
- **[Opus]**: Tasks 0, 2, 3, 6, 7 (the measurement spike, GPU copy and kernels, the data path, rendering, UI).

## Global Constraints

- **Rule 1.** No per-element input and no per-Z threshold. Lone-atom shells come from angular character and occupation, n from energy order, plus the M7 generator's own partition and ghost gate. `SMEARED_ZSIGMA` is a threshold on the dimensionless Zσ. Colours are display constants.
- **Rule 3.** Atomic units in `qm`, `qm-gpu`, `sim` and `providers`. eV only in the list, through `units`.
- **Rule 4.** Every reduction is finished in f64 (`Reducer`). Clusters, SVDs, matching and labels are f64 on the CPU. Grid products are f32 (rgba16float for the colour volume).
- **Rule 5.** `sim` never interprets the payload. `qm::orbitals` knows nothing of rendering or UI. The renderer never learns what computed a payload.
- **Rule 6.** The view never changes the solve. The bitwise on/off test (Task 3, forced updates) is the guard, and the unoccupied solve (Task 4) is outside the SCF. Never loosen a tolerance to make a test pass. A failing oracle stops the task and gets reported.
- **Rule 7.** Every gap in spec §2 is visible in the UI text (Task 7 pins the strings).
- **No CPU round trip of field data on the GPU path.** One readback of O(K² + 42K) scalars per update. The unoccupied solve has its own LOBPCG readbacks.
- **Display-only constants** are the spec §10 ledger, written to `references.md` by Task 5 next to the Task 0 measurements.
- **Timings are reported, never asserted** (M4 R3).
- **Release-only for the GPU and SCF tests**: `#[cfg_attr(debug_assertions, ignore)]` as the oracles do. GPU tests skip cleanly without an adapter.
- Commits end with the attribution line from the session's system reminder. Use `git commit -o -- <paths>` while an implementer shares the checkout.

## Review Focus (each item is pinned by a named test)

1. **The view perturbs the physics.** A display write into a solver field, a shared scratch reused mid-solve, or the unoccupied solve warm-starting the SCF from its own vectors would all show up as dynamics that are "slightly different". Pinned: `orbital_view_does_not_change_the_dynamics` (Task 3, bitwise `==` over 20 steps with an update forced at every step, 20 payloads asserted) and `unoccupied_solve_leaves_scf_state_untouched` (Task 4, m = 4, 20 steps).
2. **The Procrustes direction is transposed.** For φ_j = Σ_a R_ja p_a, M = c·Rᵀ, and the correct U = V Wᵀ equals **R**. The transposed U still gives an orthogonal matrix and plausible lobes along the wrong axes. Pinned: `procrustes_recovers_a_random_rotation_of_hydrogenic_2p` (seed 0x5EED, max|R − Rᵀ| = 0.50; asserts **RᵀU = I** to 1e-8 and dominance > 0.9999) and `ne_2p_aligns_to_the_cell_axes` (Task 5).
3. **Lone-atom labels flicker because the grid splits the shell.** An energy-gap rule passes on a cube-symmetric site and fails off-grid, where the egg-box splits 2p by mHa to 10s of mHa. Pinned: `split_shell_keeps_its_labels` (Task 1, split up to the Task 0 maximum × 2) and `ne_2p_aligns_to_the_cell_axes` (Task 5, four positions, labels asserted whatever the spread).
4. **The sign rule flickers on symmetric orbitals.** A stateless largest-value rule passes on H₂'s σ_g and fails on σ_u. Pinned: `first_appearance_sign_is_deterministic_for_sigma_u` and `sign_continuity_survives_injected_flips` (Task 1), and `h2_signs_and_tracks_hold_for_100_steps` (Task 5, with m = 2 so the real σ_u is in it).
5. **Normalisation lost in the copy.** Row padding, a stride bug, or applying h³ twice gives a texture that looks fine and integrates to 0.9 or 8. Pinned: `h_atom_display_texture_is_normalised` (Task 2, Σ ψ̃² h³ = 1 ± 1e-4) and `display_copy_matches_cpu_reference` (Task 2).
6. **Spin noise painted at full scale.** Pinned: `h2_singlet_spin_density_stays_below_the_floor` (Task 5, max|m| < 1e-4·max ρ) and `spin_summary_scale_uses_the_floor` (Task 1).
7. **Fake atomic labels** on molecules, open shells or box states. Pinned: `labels_are_index_only_for_molecules`, `open_shell_atom_falls_back_to_index_labels_in_both_channels` and `unoccupied_rows_are_never_named` (Task 1), and `ne_labels_are_1s_2s_2p` (Task 5).

## File Structure

```
crates/validate/tests/orbitals_spike.rs  # Task 0 (new): #[ignore] diag_ measurements only
docs/superpowers/notes/2026-10-15-orbital-viewer-task0-spike.md  # Task 0 (new): the numbers
crates/qm/src/orbitals.rs              # Task 1 (new): shells, clusters, moments (CPU), align, labels, Tracker, spin summary, colour reference; Task 4: unoccupied (CPU)
crates/qm/src/lib.rs                   # Task 1: pub mod orbitals (qm::poisson::solid_harmonics is already pub, L_MAX = 6: reused, unchanged)
crates/qm-gpu/src/orbitals.rs          # Task 2 (new): OrbitalView (display fields, one scalar readback, rotate, products); Task 4: unoccupied (GPU)
crates/qm-gpu/src/shaders/orbital_moments.wgsl   # Task 2 (new)
crates/qm-gpu/src/shaders/colour_accumulate.wgsl # Task 2 (new)
crates/qm-gpu/src/{lib.rs,kernels.rs,reduce.rs}  # Task 2: registration; Reduction::AbsMax
crates/qm-gpu/src/scf.rs               # Task 4: pub(crate) accessors for LobpcgWork, Kernels and the effective eig tol (no behaviour change)
crates/gpu/src/volume.rs               # Task 2: ColourVolume3D (rgba16float, STORAGE_BINDING | TEXTURE_BINDING | COPY_DST | COPY_SRC)
crates/sim/src/provider.rs             # Task 3: OrbitalRequest, OrbitalDisplay, OrbitalInfo, OrbitalField, SpinSummary; trait hooks; ForceResult/SideDisplay fields
crates/sim/src/dynamics.rs             # Task 3: DynSnapshot.orbitals, rate bound, request plumbing
crates/providers/src/ks.rs             # Task 3: CpuKs/GpuKs payloads; Task 4: unoccupied
crates/providers/src/surrogate/{mod.rs,audit.rs}  # Task 3: forward the request to the side solver only
crates/app/src/worker/{mod.rs,commands.rs}        # Task 3: Command::OrbitalView, orbital_display_now on view-on while paused
crates/render/src/volume.rs, shaders/volume.wgsl  # Task 6: modes 2/3; colour pipeline
crates/render/src/shaders/volume_colour.wgsl      # Task 6 (new)
crates/app/src/{gfx.rs,viewer.rs}      # Task 6/7: bind products, OrbitalShown (fade/label like DensityShown)
crates/app/src/ui/{menus.rs,mod.rs,actions.rs,search.rs,overlays.rs}, ui/dock/orbitals.rs (new)  # Task 7
crates/app/src/screenshot.rs           # Task 7: --ui keys show=…, orbital=…, unocc=…
crates/validate/tests/orbitals.rs      # Task 5 (new)
crates/validate/references.md          # Task 5: "Orbital viewer" section
README.md, CLAUDE.md, docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md  # Task 7
docs/superpowers/notes/2026-10-15-orbital-viewer-handoff.md                     # Task 7
(M7-gated) crates/qm/src/radial/core_tables.rs, render core path                # Task 8
```

---

### Task 0: Spike: measure the grid's shell splitting, spin noise and eigensolve cost **[Opus]** (measure only)

**Files:** Create `crates/validate/tests/orbitals_spike.rs` (`#[ignore]`d `diag_` tests, the repo's pattern for measurements) and `docs/superpowers/notes/2026-10-15-orbital-viewer-task0-spike.md`. No library code changes. If something cannot be measured without one, the smallest test-only helper goes in the spike file itself.

Measurements, on the GPU solver (`qm_gpu::GpuSolver`, σ = h, the app's `INTERACTIVE_TOL` 1e-5 and the validate tol 1e-6):

1. **2p spread vs position.** Lone **O (singlet, multiplicity 1** — the triplet does not converge, `references.md` "The O reference is slow and keeps the singlet") and **Ne**, at h = 0.25 (57³) and h = 0.15 (95³, the same ~14 Bohr box). Four positions: a grid point, a cube centre (h/2, h/2, h/2), the off-grid point (0.37h, −0.21h, 0.12h), and (h/2, 0, 0). Record:
   - per channel, the three highest occupied eigenvalues (Ne: the 2p; O singlet: whatever its occupied p-like orbitals are) and their spread;
   - the 1s and 2s eigenvalues (their shift between sites);
   - SCF steps, wall time and convergence. If Ne does not converge at 57³, record that; Task 5 then reports to the controller instead of substituting another atom.
2. **Spin noise.** The H₂ singlet at 1.4 and 1.6 Bohr, 49³: max|ρ↑ − ρ↓| / max ρ and ∫|m|.
3. **Display eigensolve cost.** Using `qm_gpu::eigen::lobpcg_with_guess_tol` directly on the converged V[ρ_out] (v_nuc + v_h + v_xc built from the state):
   - H₂ and H₂O at 49³, 81³ and 97³, with blocks n_σ + 4 + 2 and n_σ + 8 + 2;
   - cold (unoccupied from `initial_orbitals`) and warm (the previous call's vectors after a 0.01 Bohr move): wall time, iterations, readbacks, final residuals;
   - all at the effective tolerance max(eig_tol, 3 × `residual_floor_estimate`). Also print that tolerance (expected ≈ 1.8e-6 at h = 0.25).
   
   Compare each wall time with the warm SCF wall time at the same size.

- [ ] **Step 1:** Write the `diag_` tests. Each prints a table and asserts nothing except that it ran.
- [ ] **Step 2:** Run `cargo test -p validate --release --test orbitals_spike -- --ignored --nocapture` and collect the tables into the notes file.
- [ ] **Step 3:** In the notes file, state for each consumer what it takes:
  - Task 1: the largest measured 2p spread S_max, used ×2 in `split_shell_keeps_its_labels`.
  - Task 4: whether `UNOCCUPIED_BUDGET` = 0.1 lets the eigensolve run in play at each size.
  - Task 5: whether Ne converges; the spreads to expect in the tags; whether the H₂ noise ratio is under 1e-4.
  
  If any number contradicts the spec (e.g. noise ≥ 1e-4·max ρ, or the eigensolve unable to reach the effective tol), **stop and report to the controller** before Task 1.
- [ ] **Step 4:** Commit `test(validate): orbital-viewer spike — 2p splitting vs atom position, closed-shell spin noise, display eigensolve cost` (notes file included).

---

### Task 1: `qm::orbitals`, the display numerics in f64 **[Fable]**

**Files:** Create `crates/qm/src/orbitals.rs`. Modify `crates/qm/src/lib.rs`. The solid harmonics come from `qm::poisson::solid_harmonics`, already public with L_MAX = 6 and Racah normalisation; reuse it without changes.

**Interfaces (produces):**
```rust
pub const TAU_PER_TOL: f64 = 100.0;   pub const TAU_MIN: f64 = 1e-4;     pub const MAX_ROTATED_CLUSTER: usize = 7;
pub const L_CHAR_MIN: f64 = 0.9;      pub const ALIGN_RANK_MIN: f64 = 0.5;
pub const MATCH_MIN: f64 = 0.5;       pub const SHARE_COLOUR_MIN: f64 = 0.9;
pub const SIGN_MOMENT_MIN: f64 = 1e-3; pub const SPIN_FLOOR: f64 = 1e-3;
pub const BOX_STATE_OUTER_WEIGHT: f64 = 0.05; /* weight within L/6 of any face */
pub const SMEARED_ZSIGMA: f64 = 0.5;  pub const UNOCC_GUARD: usize = 2;   pub const PALETTE_SLOTS: u8 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum Spin { Up, Down }
pub fn tau(tol_scf: f64) -> f64;                                         // spec §5.1
pub fn clusters(evals: &[f64], tau: f64) -> Vec<std::ops::Range<usize>>; // sorted input, single linkage
/// b_lm = ⟨ψ|S_lm(r−R)⟩ (alignment) and c_lm = ⟨ψ|Y_lm(r̂)⟩ (character), l ≤ 3, index l²+l+m — both linear in ψ.
pub struct Moments { pub b: [f64; 16], pub c: [f64; 16] }
pub fn moments(grid: &Grid, psi: &[f64], centre: [f64; 3]) -> Moments;    // CPU reference of the GPU kernel
pub fn character(m: &Moments) -> [f64; 4];                                // W_l per orbital
pub fn set_character(ms: &[Moments]) -> [f64; 4];                         // W_l(C): Frobenius, rotation-invariant
/// spec §5.3: occupied orbitals of both channels → shells (l, range) per channel, or None (open shell / unclassifiable).
pub fn lone_atom_shells(occ_moments: [&[Moments]; 2], evals: [&[f64]; 2]) -> Option<[Vec<Shell>; 2]>;
pub struct Shell { pub l: u8, pub range: std::ops::Range<usize>, pub spread: f64 }
pub struct UpdateInput<'a> {
    pub evals: [&'a [f64]; 2], pub occupations: [&'a [f64]; 2],
    pub overlaps_prev: [Option<&'a DMatrix<f64>>; 2],       // ⟨ψ_i|ψ̃_j^prev⟩ per channel
    pub cross: Option<&'a DMatrix<f64>>,                    // ⟨ψ_i↑|ψ_j↓⟩
    pub moments: Option<[&'a [Moments]; 2]>,               // Some iff lone atom
    pub first_sign: [&'a [[f64; 9]]; 2],                   // §5.4.3 windowed functionals, normalised
    pub max_abs: [&'a [f64]; 2], pub outer_weight: [&'a [f64]; 2],
    pub n_occ: [usize; 2], pub n_computed: [usize; 2],     // occupied + unoccupied incl. guards (block-edge rule)
    pub n_core: [u8; 4],                                    // N_core,l for l ≤ 3 (zeros until M7)
    pub z_sigma: Option<f64>,                               // lone atom: Z·σ for the smeared-1s tag
    pub tol_scf: f64,
}
pub struct UpdateOutput { pub rotations: [Vec<(std::ops::Range<usize>, DMatrix<f64>)>; 2], pub rows: Vec<Row>, pub scale: [Vec<f64>; 2], pub hidden_unoccupied: [usize; 2] }
pub struct Row { pub spin: Spin, pub index: usize, pub track: u32, pub slot: u8, pub energy: f64, pub occupation: f64,
                 pub label: Label, pub cluster: Option<(usize, f64 /*spread*/)>, pub rotated: bool, pub grid_split: Option<f64>,
                 pub box_state: bool, pub smeared_1s: bool }
pub enum Label { Index(usize), Atomic { n: u8, l: u8, m: &'static str } }
pub struct Tracker { /* previous tracks per channel, free slots, next id */ }
impl Tracker { pub fn new() -> Tracker; pub fn reset(&mut self); pub fn update(&mut self, input: &UpdateInput) -> UpdateOutput; }
pub struct SpinSummary { pub n_up: usize, pub n_dn: usize, pub integral: f64, pub abs_integral: f64, pub max_abs: f64, pub rho_max: f64 }
impl SpinSummary { pub fn scale(&self) -> f64 { self.max_abs.max(SPIN_FLOOR * self.rho_max) } }
pub fn apply_rotation(orbitals: &[Vec<f64>], out: &UpdateOutput, spin: Spin) -> Vec<Vec<f64>>;
pub fn colour_volume(grid: &Grid, shown: &[(&[f64], f64 /*f_i*/, [f64; 3] /*c_i*/, f64 /*P_i*/)]) -> Vec<[f64; 4]>;
pub fn first_sign_functionals(grid: &Grid, psi: &[f64], nuclei: &[[f64; 3]]) -> [f64; 9];   // windowed (spec §5.4.3)
pub fn outer_weight(grid: &Grid, psi: &[f64]) -> f64;                                       // spec §5.7
pub fn radial_nodes(grid: &Grid, psi: &[f64], centre: [f64; 3], dir: [f64; 3], floor: f64) -> usize; // test cross-check only
```
The module doc carries spec §5 in full:
- why τ is tied to tol, and why a lone atom's shells are *not* found by τ (O_h irrep argument; the egg-box; the Task 0 numbers);
- the residual bound ≤ spread/2;
- the Procrustes derivations (tr(MU), M = WΣVᵀ ⇒ U = VWᵀ; continuity tr(UᵀA) ⇒ U = WVᵀ);
- the invariance of W_l(C) for linear c;
- Sturm–Liouville labelling;
- open-shell detection from occupation;
- the windowed first-appearance sign, and the non-deterministic "kept raw sign" case;
- the block-edge rule.

- [ ] **Step 1: Failing tests** (pure f64 on small analytic grids; `cargo test -p qm --release orbitals`)
```rust
#[test] fn tau_follows_the_scf_tolerance() { /* tau(1e-5) == 1e-3; tau(1e-6) == 1e-4; tau(1e-8) == 1e-4 */ }
#[test] fn clusters_join_gaps_up_to_tau() {
    /* evals [-1.0, -0.5, -0.5+4e-4, -0.5+7e-4, -0.2]: tau 1e-3 → [0..1, 1..4, 4..5] (spread 7e-4 ≤ 2τ, the (d−1)τ bound); tau 1e-4 → five singletons */ }
#[test] fn combination_residual_is_at_most_half_the_spread() {
    /* 3 exact eigenvectors of a diagonal operator with eigenvalues −0.5 ± s/2, −0.5; 500 random unit combinations: residual at the Rayleigh quotient ≤ s/2 + 1e-14 */ }
#[test] fn moments_of_hydrogenic_functions() {
    /* 49³, h = 0.4, centre (0.13, −0.21, 0.07) (off-grid): 1s → W_0 > 0.999; 2p_x → W_1 > 0.999 and b nonzero only at (1, x) to 1e-4 relative; 3d_xy → W_2 > 0.999 */ }
#[test] fn set_character_is_invariant_under_mixing() {
    /* ideal hydrogenic 2p, 3d and 4f sets: 200 random O(2l+1) mixings each → W_l(C) equal to the unmixed value to 1e-12 (review SHOULD 5) */ }
#[test] fn procrustes_recovers_a_random_rotation_of_hydrogenic_2p() {
    /* φ_j = Σ_a R_ja p_a, R from seed 0x5EED (assert max|R − Rᵀ| > 0.3);
       before: min_k dominance < 0.9; after: dominance_k = |⟨ψ_k|x_k⟩|/‖⟨ψ_k|x⟩‖ > 0.9999 for k = x, y, z;
       ⟨ψ_k|x_k⟩ > 0 (built-in sign); and RᵀU = I to 1e-8 (U = R: review MUST 2) */ }
#[test] fn procrustes_aligns_a_rotated_3d_set() { /* five hydrogenic 3d, seeded rotation in the l = 2 real Wigner rep: dominance > 0.999 each; labels d_xy d_yz d_z² d_xz d_x²−y² */ }
#[test] fn split_shell_keeps_its_labels() {
    /* a lone-atom channel 1s(−1.0), 2s(−0.30), 2p×3 at −0.20 + {0, S, 2S} with S = 2 × Task 0's S_max (and S = 0.05 Ha),
       eigenvectors rotated arbitrarily within the p set: labels 1s 2s 2p_x 2p_y 2p_z; grid_split == Some(2S); rotated == true */ }
#[test] fn l_character_needs_the_full_shell() {
    /* runs: 3 p → one l = 1 shell; 6 p (2p + 3p adjacent) → two shells in energy order; 2 p → None (2 ≠ k(2l+1));
       a set of p_x, p_y and a 2s → the 2s breaks the run → None */ }
#[test] fn sturm_liouville_labels_a_lone_atom() {
    /* synthetic channel: 1s(−1.0), 2s(−0.30), 2p×3(−0.20), 3s(−0.05) with hydrogenic shapes → 1s 2s 2p_x 2p_y 2p_z 3s;
       with n_core = [1, 0, 0, 0] the same s states read 2s, 3s, 4s (M7 offset) */ }
#[test] fn radial_nodes_cross_check() { /* hydrogenic 3s on a fine grid (h = 0.2, 81³) along +x: 2 nodes; 2p_x: 0; (the runtime never calls it) */ }
#[test] fn open_shell_atom_falls_back_to_index_labels_in_both_channels() {
    /* C-like: ↑ {1s, 2s, p, p} (2 of 3 p occupied), ↓ {1s, 2s}: lone_atom_shells == None → every row in BOTH channels Label::Index, rotated == false (review SHOULD 7) */ }
#[test] fn labels_are_index_only_for_molecules() { /* moments = None (two nuclei): every row Label::Index, even for a 3-fold degenerate cluster */ }
#[test] fn unoccupied_rows_are_never_named() {
    /* lone atom, closed shell, with 6 unoccupied rows including a 3-fold set: occupied rows keep 1s/2s/2p labels, every unoccupied row Label::Index;
       the labels are identical for n_computed = n_occ + 1 … n_occ + 8 (the slider cannot move them; review MUST 3) */ }
#[test] fn block_edge_cluster_is_hidden() {
    /* unoccupied evals …, −0.01, 0.10, 0.10, 0.10 (last computed index inside the 3-fold set): the 3-fold set is hidden, hidden_unoccupied == 3; with one more distinct eigenvalue after it, it is shown */ }
#[test] fn first_appearance_sign_is_deterministic_for_sigma_u() {
    /* σ_u = (1s_A − 1s_B)/N with A, B at (0, ±0.7, 0): windowed monopole < 1e-12 → decided by the windowed y dipole; two calls with the input negated give the same displayed field;
       the same in a 33³ and a 65³ box at equal h: the same functional decides (box-independent; review SHOULD 10) */ }
#[test] fn no_qualifying_functional_keeps_the_raw_sign() { /* an orbital supported only > 6 Bohr from the centroid: all nine below SIGN_MOMENT_MIN → sign unchanged, documented as not deterministic */ }
#[test] fn sign_continuity_survives_injected_flips() {
    /* 100 frames: a 1s + 2p set translated 0.01 Bohr/frame along a random direction, each frame's raw input with random signs (seeded) and a random rotation inside the p cluster (molecule mode: moments None);
       every frame: ⟨ψ̃_t|ψ̃_{t−1}⟩ ≥ 0.999 per track, tracks and slots constant */ }
#[test] fn hungarian_recovers_a_permutation_and_retires_tracks() {
    /* 8 orthonormal synthetic orbitals permuted + 1e-3 noise → tracks follow; replace one by an orthogonal newcomer (|O| < 0.5) → new track id, the retired slot reused */ }
#[test] fn closed_shell_down_shares_up_colours() { /* cross overlaps 0.999 on the diagonal → ↓ rows take the ↑ slots; an unmatched ↓ (cross < 0.9) takes a fresh slot */ }
#[test] fn spin_summary_scale_uses_the_floor() { /* max_abs 1e-7, rho_max 1.0 → scale 1e-3; max_abs 0.2 → 0.2 */ }
#[test] fn colour_volume_reference_composes_weights() {
    /* two Gaussians with colours red/green, f = 1, P = own peak: at the red centre rgb ≈ (1, ε, 0), a ≈ 1 + ε; physical scaling divides by max ρ instead */ }
#[test] fn box_state_rule_is_box_independent() {
    /* the lowest Dirichlet box state of a 33³ and of a 97³ box: outer_weight ≈ 0.163 (± 0.01) for both → box; ε = +0.01 → box;
       a hydrogenic 1s (ε = −0.5) in a 20 Bohr box: outer weight < 1e-4 → not (review SHOULD 9) */ }
#[test] fn smeared_one_s_tag_follows_z_sigma() { /* z_sigma 0.25 → no tag; 0.5 → tag; derived threshold, no element list */ }
#[cfg_attr(debug_assertions, ignore)]
#[test] fn hydrogen_one_s_from_the_cpu_scf() {
    /* qm::scf H atom 33³, h = 0.25: Σψ̃²h³ = 1 ± 1e-10; label 1s ↑; sign positive by the windowed monopole; "↓: none occupied" (empty channel handled) */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail (the module does not exist).
- [ ] **Step 3:** Implement. SVDs and the symmetric eigen problems use nalgebra in f64. The Hungarian algorithm is O(K³) and hand-written (rule 2; K ≤ 64).
- [ ] **Step 4:** `cargo test -p qm --release orbitals` is green, and the existing `qm` suite is unchanged (`cargo test -p units -p qm --release`).
- [ ] **Step 5:** Commit `feat(qm): orbital display numerics — lone-atom shells by angular character, alignment, labels, continuity tracks and signs`.

---

### Task 2: `qm_gpu::orbitals::OrbitalView`, the GPU display products **[Opus]**

**Files:** Create `crates/qm-gpu/src/orbitals.rs`, `shaders/orbital_moments.wgsl` and `shaders/colour_accumulate.wgsl`. Modify `crates/qm-gpu/src/{lib.rs,kernels.rs,reduce.rs}` (`Reduction::AbsMax`: one pass, f32 partial max of |x|, exact) and `crates/gpu/src/volume.rs` (`ColourVolume3D`).

**Interfaces (produces):**
```rust
pub struct OrbitalView { /* display fields (previous displayed set) per channel, Tracker, readback counter */ }
pub enum Product<'a> { Orbital { spin: Spin, track: u32, squared: bool, out: &'a Volume3D },
                       Colour { include: &'a [u32], physical: bool, out: &'a ColourVolume3D },
                       Spin { out: &'a Volume3D }, None }
pub struct ViewUpdate { pub out: qm::orbitals::UpdateOutput, pub spin: SpinSummary, pub scale: f32, pub readbacks: usize }
impl OrbitalView {
    pub fn new(solver: &GpuSolver) -> OrbitalView;
    /// spec §4.2: one scalar readback, the CPU f64 step (`Tracker::update`), lincombs into the display fields, the product. No wait after the readback.
    pub fn update(&mut self, solver: &GpuSolver, state: &GpuState, nuclei: &[qm::Nucleus], tol_scf: f64, product: Product) -> ViewUpdate;
    pub fn reset(&mut self);
}
pub struct ColourVolume3D { pub texture: wgpu::Texture, pub view: wgpu::TextureView, pub sampler: wgpu::Sampler, pub dims: [u32; 3] }
impl ColourVolume3D { pub fn new(ctx: &GpuContext, dims: [u32; 3]) -> Self; pub fn upload_f32(&self, ctx: &GpuContext, rgba: &[[f32; 4]]) /* → f16 */; pub fn read_back(&self, ctx: &GpuContext) -> Vec<[f32; 4]>; }
```
Kernels:
- `orbital_moments`: per orbital, S_lm(r − R) (as `qm::poisson::solid_harmonics`) and Y_lm(r̂) for l ≤ 3 computed in-kernel, 32 linear workgroup partial sums. Also in the same kernel:
  - the 9 windowed first-sign functionals of spec §5.4 item 3;
  - the outer-L/6 weight of spec §5.7.
  
  The Reducer finishes them in f64, within the same readback as the overlaps (`Reducer::block_dots_in`) and `AbsMax`.
- `colour_accumulate`: one thread per interior point, loops over ≤ 64 included orbitals with (f_i c_i / P_i) in a uniform array, and `textureStore`s rgba16float.
- Rotations use `lincomb_in`.
- The orbital and spin products go through `unpad_rec` + `copy_buffer_to_texture`, exactly as `copy_density_to_volume` does.

- [ ] **Step 1: Failing tests** (`cargo test -p qm-gpu --release orbitals`; skip without an adapter)
```rust
#[test] fn abs_max_reduction_is_exact() { /* fields with the extreme negative, positive, in the padding-adjacent interior: equals the CPU max|x| exactly (f32) */ }
#[test] fn moments_match_the_cpu_reference() { /* hydrogenic 1s, 2p set, 3d_xy uploaded as f32 at an off-grid centre: GPU b, c, first-sign and outer weight vs the qm::orbitals functions on the same f32 data, max rel err ≤ 1e-5 */ }
#[test] fn overlaps_and_max_match_the_cpu() { /* 6 synthetic orbitals: |O_gpu − O_cpu| ≤ 1e-6; max|ψ| exact */ }
#[test] fn display_copy_matches_cpu_reference() { /* a rotated, sign-flipped p set: texture readback vs qm::orbitals::apply_rotation, max abs diff ≤ 1e-6 · max|ψ| */ }
#[test] fn h_atom_display_texture_is_normalised() { /* GpuSolver H atom 49³ (σ = h = 0.25, tol 1e-6): Product::Orbital → Σ texture² · h³ = 1 ± 1e-4; squared: Σ texture · h³ = 1 ± 1e-4 */ }
#[test] fn spin_texture_integrates_to_the_unpaired_count() { /* H atom: Σ m h³ = 1 ± 1e-4; SpinSummary.integral matches the texture sum to 1e-6 */ }
#[test] fn colour_volume_matches_the_cpu_reference() { /* two orbitals, both scalings: rgba16float readback vs qm::orbitals::colour_volume within f16 rounding (rel 2e-3 where a > 1e-3) */ }
#[test] fn one_readback_per_update() { /* ViewUpdate.readbacks == 1 for every product, lone atom and molecule */ }
#[test] fn update_does_not_write_solver_fields() { /* read back every GpuState field before/after 3 updates of each product: identical bytes */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement. Display fields come from their own pool (not the solver's), are sized K + 2m (spec §4.4), and are dropped on `reset` and on a grid change. Record the dispatch count per update in a doc comment.
- [ ] **Step 4:** Green. Run `cargo test -p qm-gpu --release` with the whole suite: unchanged.
- [ ] **Step 5:** Commit `feat(qm-gpu,gpu): orbital display products on the device — one scalar readback, GPU-to-GPU copies, rgba16f colour volume`.

---

### Task 3: The data path through the force boundary **[Opus]**

**Files:** Modify `crates/sim/src/{provider.rs,dynamics.rs}`, `crates/providers/src/ks.rs`, `crates/providers/src/surrogate/{mod.rs,audit.rs}` and `crates/app/src/worker/{mod.rs,commands.rs}`.

**Interfaces (produces):** in `sim::provider`:
```rust
pub enum OrbitalRequest { Off, List, Orbital { spin: Spin, track: u32, squared: bool },
                          AllColoured { include: Option<Vec<u32>> /* None = all occupied */, physical: bool, additive: bool },
                          SpinDensity }
pub struct OrbitalViewRequest { pub what: OrbitalRequest, pub unoccupied: u8, pub force_every_step: bool /* tests and paused stepping */ }
pub enum OrbitalField { Gpu { volume: Arc<Volume3D>, signed: bool, scale: f32 }, GpuColour { volume: Arc<ColourVolume3D>, scale: f32 },
                        Cpu { values: Arc<Vec<f32>>, signed: bool, scale: f32 }, CpuColour { rgba: Arc<Vec<[f32; 4]>>, scale: f32 } }
pub struct OrbitalDisplay { pub cell: Cell, pub origin: [f64; 3], pub positions: Vec<[f64; 3]>, pub rows: Vec<OrbitalInfo>,
                            pub field: Option<OrbitalField>, pub spin: SpinSummary, pub unoccupied: Option<UnoccupiedStatus>, pub notes: Vec<String> }
// ForceProvider (default no-ops):
fn set_orbital_view(&mut self, _req: OrbitalViewRequest) {}
fn orbital_display_now(&mut self) -> Option<OrbitalDisplay> { None }
// ForceResult.orbitals: Option<OrbitalDisplay>; SideDisplay.orbitals: Option<OrbitalDisplay>; DynSnapshot.orbitals: Option<OrbitalDisplay>
pub const ORBITAL_MIN_INTERVAL: Duration = Duration::from_millis(100);
```
- `OrbitalInfo` mirrors `qm::orbitals::Row`. It is re-declared in `sim` so that `sim` does not depend on `qm` for it; `providers` converts.
- Textures come from a `TexturePool`-style pool per product kind, which never overwrites a held texture.
- **Estimated mode** (spec §4.3, review SHOULD 12): `SurrogateKs` forwards the request **only to its background side solver**. Audits and refreshes carry orbitals; fallback steps and model steps carry none.

- [ ] **Step 1: Failing tests**
```rust
// providers (GPU, release)
#[test] fn orbital_view_does_not_change_the_dynamics() {
    /* H₂ at 1.6 Bohr, 49³, GpuKs, 20 velocity-Verlet steps (dt 0.25 fs) twice: view Off vs view cycling List → Orbital(σ_g ↑) → AllColoured → SpinDensity every step
       with force_every_step = true; assert exactly 20 payloads (review SHOULD 13); energy and every force component bitwise equal (==) at every step */ }
#[test] fn payload_rate_is_bounded_in_play() { /* Dynamics in play for 2 s wall with the view on, force_every_step = false: payload count ≤ 21; each paused single step: exactly one */ }
#[test] fn view_on_while_paused_needs_no_solve() { /* after one evaluate, set_orbital_view(Orbital) + orbital_display_now() → Some, solver profile.steps unchanged */ }
#[test] fn cpu_and_gpu_rows_agree() { /* H₂ 49³ at fixed σ = h, tol 1e-6: same row count, labels, occupations; |ε_gpu − ε_cpu| ≤ 1e-4 Ha */ }
#[test] fn empty_scene_and_empty_channel() { /* H⁺ (0 electrons): rows empty, note "no electrons — nothing to show"; H atom: ↓ group empty, spin integral 1 */ }
#[test] fn count_change_resets_tracks_and_moves_the_selection() {
    /* H₂ → add H far away (reset): tracks restart in energy order; a selection of a vanished track falls back to the same channel/index or the HOMO, with the note */ }
// providers (surrogate; the existing test fixtures)
#[test] fn estimated_mode_has_one_orbital_producer() {
    /* model step: orbitals None; fallback step: orbitals None; a Refresh and an Audit side display: Some, from the same tracker (track ids continue across them, never reset between) */ }
// app worker
#[test] fn worker_forwards_the_request_and_publishes_the_payload() { /* Command::OrbitalView → provider sees it; the next snapshot's orbitals is Some */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement:
  - GpuKs keeps an `Option<OrbitalView>` (built lazily, dropped on Off, on a grid change and on `reset`).
  - CpuKs runs `qm::orbitals` on its `KsState`.
  - The rate bound lives in the provider: `Instant` of the last payload, with `force_every_step` set by the driver when paused or stepping.
- [ ] **Step 4:** Green. Run `cargo test -p sim --release`, `-p providers --release` and `-p app --release`: all existing tests unchanged.
- [ ] **Step 5:** Commit `feat(sim,providers,app): orbital display payloads through the force boundary — opaque to sim, rate-bounded, one producer in estimated mode, bitwise no-impact`.

---

### Task 4: Unoccupied orbitals, a display eigensolve **[Fable]**

**Files:** Modify:
- `crates/qm/src/orbitals.rs` (CPU);
- `crates/qm-gpu/src/orbitals.rs`;
- `crates/qm-gpu/src/scf.rs`: `pub(crate)` accessors for `LobpcgWork`, `Kernels`, the xc kernel and the effective eig tol, with no behaviour change. Sharing `LobpcgWork` is safe: no state is carried, since np = 0 and HX is recomputed at entry (review NOTE);
- `crates/providers/src/ks.rs`.

**Interfaces (produces):**
```rust
pub struct Unoccupied { pub evals: [Vec<f64>; 2], pub residuals: [Vec<f64>; 2], pub converged: [Vec<bool>; 2], pub hidden: [usize; 2] }
pub fn unoccupied_cpu(state: &KsState, m: usize, settings: &ScfSettings, previous: Option<&[Vec<Vec<f64>>; 2]>) -> (Unoccupied, [Vec<Vec<f64>>; 2]);
impl OrbitalView { pub fn unoccupied(&mut self, solver: &mut GpuSolver, state: &GpuState, m: usize, settings: &ScfSettings) -> Unoccupied; }
pub const UNOCCUPIED_MIN_INTERVAL: Duration = Duration::from_secs(1);
pub const UNOCCUPIED_BUDGET: f64 = 0.1;   // of the summed SCF wall since the previous run (spec §4.5)
```
Method (spec §4.5):
1. Build v_σ = v_nuc + v_h + v_xc,σ[ρ_out] into display scratch. No Poisson solve: `GpuState.v_h` is V_H[ρ_out].
2. Run LOBPCG with block n_σ + m + `UNOCC_GUARD`, starting from [copies of the occupied orbitals, the previous unoccupied vectors or `initial_orbitals(…, seed 3/4)`]. It runs at the **effective** tolerance max(`eig_tol`, `EIG_TOL_FLOOR_FACTOR` × `residual_floor_estimate`), ≈ 1.8e-6 at h = 0.25 (review MUST 4), with the SCF's iteration cap. The CPU uses `eig_tol`.
3. Keep the unoccupied part, apply the block-edge rule (`qm::orbitals`), and discard the occupied pairs after the agreement check.

Budget: in play, run only if the last run's wall time ≤ `UNOCCUPIED_BUDGET` × the summed SCF wall since; otherwise only when paused. Task 0's cost table says where that line falls.

Before writing the module doc, derive why the occupied eigenvalues of V[ρ_out] differ from the SCF's (V[ρ_in]) by ≤ τ at a converged tol, and state the bound used.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn unoccupied_solve_leaves_scf_state_untouched() {
    /* GPU H₂ 49³, m = 4, 20 MD steps with the unoccupied solve forced every step vs never: energy/forces bitwise equal at every step (review SHOULD 13) */ }
#[test] fn display_solve_converges_at_the_effective_tolerance() {
    /* GPU H₂ and H₂O 49³, m = 4: every unoccupied row with ε < 0 converged (residual ≤ effective tol, printed ≈ 1.8e-6 at h = 0.25); none ends on the cap; the raw 1e-6 is never the stop criterion */ }
#[test] fn occupied_eigenvalues_agree_with_the_scf() { /* H₂ and Li (GPU, tol 1e-6): |ε_occ(display) − ε_occ(SCF)| ≤ tau(1e-6) = 1e-4 Ha */ }
#[test] fn unoccupied_gpu_matches_cpu() { /* H₂ 49³, σ = h, tol 1e-6, m = 2: the converged ones agree to 1e-4 Ha; both residuals ≤ their tol or flagged */ }
#[test] fn unoccupied_are_orthogonal_to_occupied() { /* |⟨ψ_occ|ψ_unocc⟩| ≤ 1e-5 (f32) on the GPU, 1e-10 on the CPU */ }
#[test] fn block_edge_hides_a_cut_set() { /* He 49³ (box states in 1-, 3-, 6-fold sets), m = 1 … 8: no shown cluster ever contains the last computed vector; "k shown" printed per m */ }
#[test] fn box_states_are_flagged() { /* He 49³ m = 4: every row with ε ≥ 0 or outer weight > 0.05 tagged box; (ε, outer weight, residual) printed for references.md */ }
#[test] fn unoccupied_budget_in_play() {
    /* H₂ 49³ in play 5 s: ≤ 1 solve per second and its summed wall ≤ 0.1 × the summed SCF wall (+ one solve); paused: one per step; 97³ H₂O: the measured wall printed, paused-only if over budget */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement. Report the growth of `LobpcgWork` to n_σ + m + 2 in the module doc (≈ +8 fields per vector).
- [ ] **Step 4:** Green. Run `cargo test -p validate --release --test gpu_vs_cpu` (unchanged solver: it must be bit-for-bit the same report).
- [ ] **Step 5:** Commit `feat(qm,qm-gpu,providers): unoccupied Kohn–Sham orbitals for display — eigensolve on the converged potential at the effective tol, block-edge rule, budget`.

---

### Task 5: The display oracles, `validate/tests/orbitals.rs` **[Fable]**

**Files:** Create `crates/validate/tests/orbitals.rs`. Modify `crates/validate/references.md` with a new section "Orbital viewer" containing:
- the spec §10 ledger with reasons;
- the Task 0 tables (2p spread vs position at h = 0.25/0.15, spin noise, eigensolve cost);
- the numbers measured below;
- the NIST LSD H 1s eigenvalue, read off the source, with its citation.

All GPU, release-only, skipping without an adapter. Seeds are overridable with `ORBITALS_SEED`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn h_atom_one_s() {
    /* 49³ h = 0.25 σ = h, tol 1e-6: Σψ̃²h³ = 1 ± 1e-4; label "1s"; ε_1s↑ printed next to the NIST LSD value (gap documented, rule 7; not asserted) */ }
#[test] fn ne_2p_aligns_to_the_cell_axes() {
    /* Ne at a grid point, the cube centre (h/2, h/2, h/2) and (0.37h, −0.21h, 0.12h), 57³, tol 1e-5 (the app's interactive tol), both spins:
       inject a seeded rotation into the three 2p fields; dominance before: min < 0.9;
       after OrbitalView::update: |⟨p_k|x_k⟩|/‖⟨p_k|x⟩‖ > 0.99 for k = x, y, z; ⟨p_k|x_k⟩ > 0;
       the 2p spread printed and carried in the row tag (grid_split) — NOT bounded: the egg-box splits the shell (review MUST 1; Task 0 numbers).
       If Ne did not converge in Task 0, this test reports that and the task stops for a ruling — no substitute atom */ }
#[test] fn ne_labels_are_1s_2s_2p() { /* at all three positions, both channels: 1s, 2s, 2p_x, 2p_y, 2p_z in energy order; 1s tagged "smeared 1s" (Zσ = 2.5 ≥ 0.5) */ }
#[test] fn h2_signs_and_tracks_hold_for_100_steps() {
    /* the app's H₂ preset (1.6 Bohr), 49³, GpuKs, NVE 100 steps dt 0.25 fs, view on (List) with m = 2 unoccupied (σ_u among them; review SHOULD 13), force_every_step:
       ⟨ψ̃_t|ψ̃_{t−1}⟩ ≥ 0.99 for σ_g ↑, σ_g ↓ and the lowest unoccupied ↑ at every step; tracks and slots constant; ↓ shares ↑'s slot;
       the raw solver's sign flips counted and printed (the rule must hold whatever that count is) */ }
#[test] fn spin_density_integrals() { /* H: ∫m = 1 ± 1e-4; Li (2↑1↓, 49³): 1 ± 1e-4; ∫|m| printed */ }
#[test] #[ignore = "the open-shell all-electron O does not converge at default grids (references.md: 'The O reference is slow and keeps the singlet'); un-ignored only under an M7 policy that pseudises O, at M7's Hund gate"]
fn o_triplet_spin_density_integrates_to_two() { /* O, multiplicity 3, pseudised O (M7), 57³: ∫m = 2 ± 1e-4 */ }
#[test] fn h2_singlet_spin_density_stays_below_the_floor() { /* H₂ at 1.4 and 1.6 Bohr: max|m| < 0.1 · SPIN_FLOOR · max ρ = 1e-4 · max ρ (review SHOULD 8); the ratio printed next to Task 0's */ }
#[test] fn li_labels_and_open_channel() { /* Li ↑: 1s, 2s; ↓: 1s — the ↑/↓ 1s share a slot (cross overlap ≥ 0.9); all s, so the atom is closed by the occupation rule (no partial l-set) */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail (before Tasks 1–4 land on the branch, or with the hooks stubbed).
- [ ] **Step 3:** Make them pass by fixing the code under test, never the bounds. Record every measured number in `references.md`.
- [ ] **Step 4:** Run `cargo test -p validate --release --test orbitals` green. The full `cargo test -p validate --release` is unchanged elsewhere. Dispatch `physics-reviewer` on Tasks 1, 4 and 5 (mandatory when Opus stood in for Fable).
- [ ] **Step 5:** Commit `test(validate): orbital viewer oracles — normalisation, Ne 2p alignment across grid positions, H₂ sign stability, spin integrals`.

---

### Task 6: Rendering, signed modes and the colour volume **[Opus]**

**Files:** Modify `crates/render/src/volume.rs` and `shaders/volume.wgsl`. Create `shaders/volume_colour.wgsl`. Modify `crates/app/src/gfx.rs` (bind the products, both crossfade ends, per kind).

**Interfaces (produces):** `VolumeUniform.mode` 2 = signed isosurface, 3 = signed volume. The uniform gains `colour_pos: [f32; 3]`, `colour_neg: [f32; 3]` and `additive: u32`, with the size assertion updated. `ColourVolumePass` (or a second pipeline inside `VolumePass`) has `bind_pair(prev, cur, depth)`, `set_params` and `draw`.

- [ ] **Step 1: Failing tests** (headless, `cargo test -p render --release`)
```rust
#[test] fn phase_isosurface_colours_lobes_by_sign() { /* synthetic p_x on 33³ (box ±4): pixel over +x lobe red − blue > 60/255; over −x blue − red > 60/255; filtered vs manual paths < 8/255 */ }
#[test] fn signed_volume_of_zero_draws_nothing() { /* all-zero field, mode 3: every pixel exactly the clear colour */ }
#[test] fn additive_colour_shows_the_inner_shell() {
    /* inner Gaussian (σ 0.5, slot colour A) inside an outer shell (r = 3 ± 0.5, colour B): centre pixel in additive has A's channel > 0.3 of B's; in weighted, B dominates the centre pixel */ }
#[test] fn colour_crossfade_is_linear() { /* mix 0.5 of two colour volumes ≈ the frame of their average within 3/255 */ }
#[test] fn fade_scales_every_mode() { /* fade 0.3 → premultiplied output ×0.3 within 2/255 in modes 2, 3 and colour */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement. Measure and print the frame time for each mode at 97³ (1400×900, headless). Report it; do not assert it.
- [ ] **Step 4:** Green, and the existing `render` and `app` tests are unchanged.
- [ ] **Step 5:** Commit `feat(render): signed phase/spin modes and an additive colour-volume pipeline`.

---

### Task 7: UI, honesty text, docs **[Opus]**

**Files:**
- Modify `crates/app/src/ui/{menus.rs,mod.rs,actions.rs,search.rs,overlays.rs}`.
- Create `ui/dock/orbitals.rs`.
- Modify `crates/app/src/{viewer.rs,screenshot.rs}`, `README.md`, `CLAUDE.md` and `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md`.
- Create `docs/superpowers/notes/2026-10-15-orbital-viewer-handoff.md`.

**Contents:**
- View ▸ Density gets `show: Density | Orbitals | All orbitals coloured | Spin density` (spec §3), with Ctrl-K entries of the same names.
- `DockTab::Orbitals` holds the list (spec §3.1): eV through `units`; swatches; include checkboxes; phase/|ψ|²; "show unoccupied" 0–8 with "m requested, k shown"; iso sliders; the spin readout block.
- `viewer::OrbitalShown` mirrors `DensityShown`: fade and label in estimated mode, "needs a DFT step", "no electrons — nothing to show".
- A legend chip shows the colour pair or the palette in use.
- New screenshot `--ui` keys: `show=orbitals|colour|spin`, `orbital=<label or #k>[↑|↓]`, `unocc=<m>`.

Pinned strings (tests assert these substrings):
- "Kohn–Sham orbital: an auxiliary object of DFT, not a measured electron wavefunction. The total density is the physical observable."
- "Spin is intrinsic (↑/↓), not a rotation. Orbitals are stationary: they move only with the nuclei."
- "rotated within a degenerate set for display"
- "split by the grid by"
- "a state of the simulation box, not of the molecule"
- "the Kohn–Sham gap is not the fundamental gap" (unoccupied rows; review NOTE)
- "the sign of an orbital is arbitrary; only the relative sign of its lobes means anything"
- "orbitals from t = … fs (last DFT solve)"
- "core: reconstructed for display" (the string is defined here, used in Task 8)

- [ ] **Step 1: Failing tests** (egui widget tests in `ui/tests.rs` style; `cargo test -p app --release`)
```rust
#[test] fn show_selector_sends_the_request() { /* clicking "Orbitals" → Action::OrbitalView(Orbital{..HOMO..}) and the dock opens on Orbitals */ }
#[test] fn rows_format_energies_in_ha_and_ev() { /* ε = −0.5 Ha row reads "−0.500" and "−13.61" (units::HARTREE_IN_EV) */ }
#[test] fn picking_a_row_selects_by_track() { /* click row 2p_y ↑ → Orbital { spin: Up, track: <its track> } */ }
#[test] fn tooltips_carry_the_honesty_text() { /* every pinned substring above appears in its widget's hover text */ }
#[test] fn estimated_mode_fades_and_labels_orbitals() { /* surrogate snapshot after a side-display payload → fade == SURROGATE_DENSITY_FADE, label starts "orbitals from t =" */ }
#[test] fn unoccupied_slider_shows_requested_and_shown() { /* payload with hidden_unoccupied 3 at m = 4 → "4 requested, 1 shown" */ }
#[test] fn search_finds_the_modes() { /* Ctrl-K "spin" → "Spin density" entry */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the viewer manual checklist rows added for the three modes (Ne lone atom, dragged so the grid splitting shows in the tag; H₂ vibrating; H₂O; estimated mode with refresh on; CPU backend). Record which were verified in a window and which only by code reading.
- [ ] **Step 5:** Take screenshots:
  - a one-atom Ne scene (`--scene` with a file written for the shot; there is no Ne preset) with `--ui "show=orbitals;orbital=2p_x↑"`;
  - `show=colour` on H₂O;
  - `show=spin` on Li.
  
  They go into the README. Then write the handoff note: the update cost at 49³, 57³ and 97³ (GPU ms, readbacks, VRAM), the unoccupied solve's wall vs budget, the frame times from Task 6, and the open items.
- [ ] **Step 6:** README: a new "Orbitals and spin" section, the controls, the honesty text, and the performance numbers. CLAUDE.md: one line under Build/App for the new flags and the `orbitals` validate test. Commit `feat(app): orbital viewer — show selector, Orbitals dock, legends and the honesty text; docs`.

---

### Task 8 (M7-gated): Core shells reconstructed for display **[Fable]**

**Gate:** these must be merged first:
- the M7 tasks for `qm::radial::atom`, `partition` and the spin-restricted mode (pseudopotential spec §4.1–4.3);
- the ghost-state gate (§4.6);
- `IonicPolicy` in `Scene`.

Otherwise this task moves to M7's plan.

**Files:**
- Create `crates/qm/src/radial/core_tables.rs`.
- Modify `crates/qm/src/orbitals.rs` (`n_core` from the partition; core rows; valence labels conditional on the ghost gate) and `crates/providers/src/ks.rs` (core rows and tables in the payload).
- Modify `crates/render/src/shaders/volume*.wgsl` (analytic core path: a 1-D table texture with one row per (Z, n, l), nucleus positions in a storage buffer, refined step inside r_core; core included in max ρ for *physical* scaling).
- Modify `crates/app` (row tags, tooltip).

**Interfaces (produces):**
```rust
pub struct CoreTable { pub z: u8, pub shells: Vec<CoreShell>, pub r_core: f64, pub dr: f64 }
pub struct CoreShell { pub n: u8, pub l: u8, pub eps: f64, pub u_over_r: Vec<f32> /* uniform r grid, dr */, pub max_abs: f64 }
/// From the spin-restricted reference atom (spec §8; review SHOULD 14): one row per spatial core orbital, occupation 2.
pub fn core_table(atom_restricted: &radial::AtomResult, part: &radial::Partition, dr: f64) -> CoreTable;
```

- [ ] **Step 1: Failing tests**
```rust
#[test] fn core_orbitals_are_normalised_on_the_radial_grid() { /* O 1s (restricted): ∫u² dr = 1 ± 1e-8 (log grid); the uniform resampling's ∫(u/r)² r² dr = 1 ± 1e-5 */ }
#[test] fn table_interpolation_error_is_relative_to_the_shell_maximum() {
    /* resampled u/r vs the log-grid value at 1000 random r < r_core: |err| ≤ 1e-5 · max|u/r| of the shell — including Ar's 2s/2p near their radial nodes, where a pointwise relative bound would fail (review SHOULD 14) */ }
#[test] fn shader_core_matches_the_cpu() { /* headless: a ray through an O nucleus sampled in the shader's core path (debug readout) vs CPU u/r·Y_lm at the same points, |err| ≤ 1e-4 · max|u/r| (f32) */ }
#[test] fn valence_labels_skip_the_core() { /* pseudised O (CoresOnly), ghost gate passed: valence rows 2s, 2p_x…; core row "1s ↑↓" (occupation 2) tagged "core: reconstructed for display", ε from the restricted AE atom */ }
#[test] fn valence_labels_need_the_ghost_gate() { /* a generation record with the ghost gate failed (synthetic): the atom is #k, no atomic names */ }
#[test] fn spin_mode_draws_the_core_neutral() { /* O under CoresOnly: the spin product is identical with and without core rows (restricted core: exact) */ }
#[test] fn physical_scaling_includes_the_core() { /* colour volume, physical: max ρ used == max over (valence ρ + reconstructed core ρ) */ }
#[test] fn hydrogen_has_no_core_rows() { /* H under All: no core rows (N_core = 0, pseudopotential spec §4.2 item 4) */ }
```
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Green. Run the physics-reviewer pass.
- [ ] **Step 5:** README: the core rows in "Orbitals and spin". Commit `feat(qm,render,app): core shells reconstructed for display from the generator's restricted all-electron atom`.

---

## Self-review (done before commit; repeated after the physics review)

- **Spec coverage:**

  | Spec | Plan |
  |---|---|
  | §3 | Tasks 6–7 |
  | §4.2–4.4 | Tasks 2–3 |
  | §4.5 | Task 4 (cost from Task 0) |
  | §5.1–5.7 | Task 1; oracles in Task 5 |
  | §6 | Task 6 |
  | §7 | Task 3's tests, Task 7's labels, Task 5 |
  | §8 | Task 8 |
  | §10 | Task 5 writes it to `references.md` |
  | §11 | Tasks 0, 2, 5, 6 |
  | §14 MUSTs | MUST 1: Tasks 0, 1 (`split_shell_keeps_its_labels`), 5 (four positions, spread not bounded) |
  | | MUST 2: Task 1 (RᵀU = I) |
  | | MUST 3: Task 1 (`unoccupied_rows_are_never_named`, `block_edge_cluster_is_hidden`), Task 4 (`block_edge_hides_a_cut_set`) |
  | | MUST 4: Task 4 (`display_solve_converges_at_the_effective_tolerance`) |

- **Exact values:**
  - H 1s norm: 1 ± 1e-4 (GPU), ± 1e-10 (CPU).
  - Ne 2p dominance: > 0.99 after, < 0.9 before, at three positions.
  - H₂: 100 steps with m = 2, overlaps ≥ 0.99.
  - Spin: H 1 ± 1e-4, Li 1 ± 1e-4, O 2 ± 1e-4 (ignored until M7 pseudises O), singlet noise < 1e-4·max ρ.
  - Bitwise no-impact: 20 forced payloads; m = 4 over 20 steps.
  - Analytic alignment: 0.9999 (p) and 0.999 (d), RᵀU = I to 1e-8.
  - W_l invariance to 1e-12.
  - τ = 1e-3 and 1e-4 at the two tolerances.
  - Box-state outer weight 0.163 ± 0.01 at 33³ and 97³.
- **Type consistency.** `OrbitalViewRequest`, `OrbitalDisplay`, `OrbitalInfo` (sim) ↔ `Row` (qm) conversion in providers. `Product` (qm-gpu) is chosen from the request in providers. `ColourVolume3D` is in `gpu` and used by qm-gpu and render. `UNOCC_GUARD` is in `qm::orbitals` and used by qm-gpu and providers.
- **Tags:**
  - Opus: the spike (Task 0), GPU, data path, render and UI.
  - Fable: numerics, eigensolve, oracles and core. Opus is the fallback while Fable is unavailable, with a mandatory physics review.
