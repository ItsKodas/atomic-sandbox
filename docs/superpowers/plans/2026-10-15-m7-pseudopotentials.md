# M7 — Self-generated Pseudopotentials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every species with a core gets a norm-conserving Troullier–Martins pseudopotential in Kleinman–Bylander form with a nonlinear core correction. It is generated at run time from this project's own radial all-electron LSDA atom by one element-independent recipe (Z in, potential out) and used by the CPU and GPU real-space solvers, the providers, the surrogate's provenance and the viewer. A bond-angle overlay shows the geometries this buys (water at 104.5°, not 177°).

**Architecture:**
- `qm::radial` (new, CPU, f64) is the whole generator: the log grid and Numerov shooting, the spin-polarised/restricted AE atom, the core/valence partition, TM construction and unscreening, NLCC, the KB form with the ghost gates, the transferability gates, the f ladder and the local-channel order, the generation record and `recipe_hash`, and the spherical-Bessel filter with h_req. Every universal constant is a field of **one `const` table, `qm::radial::ledger::LEDGER`** (spec §4.9).
- `qm::ionic` and `qm::nonlocal` (new) hold `IonicPolicy`, `IonicSetup` and the projector operator. `qm::System` carries an optional `Arc<IonicSetup>`. With `None` the solver runs today's code path bit for bit (a test pins it). `qm::scf`, `qm::forces` and `qm::nuclear` gain the pseudised terms.
- `qm-gpu` gains `ionic.rs` and `shaders/ionic.wgsl` (`local_sr`, `nlcc_deposit`, `proj_dot`, `proj_finish`, `proj_axpy`, `gather_dot3`), each checked against the CPU reference.
- Plumbing: `sim::Scene.ionic`, a process-wide `providers::pseudo::PseudoCache`, and `surrogate::Provenance.ionic` (sidecar version 4). In the app: the policy radio, species badge, valence-energy label, core-overlap tint, the ρ_pc display toggle and the bond-angle overlay.
- Validation: `crates/validate/tests/{pseudo_gen,pseudo,pseudo_oracles}.rs` (new), `gpu_vs_cpu.rs` (extended), and `references.md` sections.

**Tech Stack:** Rust 2021; nalgebra (5×5 LU, small DIIS systems); wgpu 30 / WGSL; egui 0.36. No new crates (rule 2).

**Spec:** `docs/superpowers/specs/2026-10-14-pseudopotential-design.md` ("the spec"; the § numbers below refer to it). It was physics-reviewed on 2026-10-14 with the verdict "ready with fixes", and every fix has been folded in; the review is `docs/superpowers/notes/2026-10-14-pseudopotential-spec-review.md`. Parent: `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` §6 and §11 item 1. The plan argues from the spec. Where the plan found a genuine spec defect, it is listed under **Spec questions** at the end. The plan itself was physics-reviewed on 2026-10-15 (`docs/superpowers/notes/2026-10-15-m7-plan-review.md`, verdict "ready after amendments": six MUSTs, the SHOULDs, and a ruling on each Spec question). Every amendment is folded into the tasks below, and the spec-level corrections are in the spec's §11. The section at the end lists each ruling and where it was applied. Each task's physics-review gate still checks its own diff.

**Interplay:** the orbital-viewer plan (`2026-10-15-orbital-viewer.md`, its Task 8) consumes `qm::radial::AtomResult` and `qm::radial::Partition` from Tasks 1–2 under exactly those names. That task is not part of this plan.

## Model tags and review gates

| # | task | model | physics-review gate before merge |
|---|---|---|---|
| 1 | radial AE atom | **Fable** | **yes** |
| 2 | partition + TM + unscreening | **Fable** | **yes** |
| 3 | KB form, radial KB atom, ghost gates, f ladder | **Fable** | **yes** |
| 4 | NLCC, gates T1–T7, record, `recipe_hash` | **Fable** | **yes** |
| 5 | filter, h_req, filtered re-run | **Fable** | **yes** (Spec questions 2–3 were settled by the plan review: one V_SR table, King-Smith band completion) |
| 6 | `qm::ionic` + SCF integration | **Fable** | **yes** |
| 7 | forces | **Fable** | **yes** |
| 8 | GPU kernels | Opus | no (the `gpu_vs_cpu` suite is the gate) |
| 9 | sim/providers/surrogate/app plumbing | Opus | no |
| 10 | validation oracles | **Fable** | **yes** |
| 11 | `IonicPolicy::All`, pseudo-H/He, dimer gate | **Fable** | **yes** |
| 12 | Z = 11–18 enablement | Opus | no (its gates are Task 4's code) |
| 13 | bond-angle overlay | Opus | no |

For now the controller dispatches Opus for every task. **Every Fable-tagged task (1–7, 10, 11) is gated by the `physics-reviewer` agent before it lands.** The reviewer receives the task's diff, this plan's task text and the spec sections it cites. The task report says that Opus ran a Fable task.

Order (spec §8): 1 → 2 → 3 → 4 → 5 → 6 → 7 (the CPU reference) → 8 ∥ 9 → 10 → 11 → 12 → 13.

## Global Constraints

- **Rule 1 is the crux.** The only per-element inputs are Z and the isotope mass. Every number that varies by element is an *output* of the recipe: r_c,l, the local channel, E_KB, r_nlc, the core/valence split, the configuration, h_req, r_core, which species is pseudised, and the f chosen. There is no configuration table, no radius table, nothing indexed by Z and no `match z`.
- **All universal constants live in `qm::radial::ledger::LEDGER` and nowhere else.** Tasks 2–5 and 11 read `LEDGER.<field>`. Task 4's `gate_thresholds_are_constants` enforces this mechanically for every file in `crates/qm/src/radial/`.
- **"Element X needs a slightly different value" is a design violation.** An implementer who finds that, say, nitrogen needs a slightly different f stops and reports it (spec §4.9). The answer is either a universal change, after which every element is regenerated and re-gated, or a refusal.
- **Rule 3.** Atomic units throughout `qm`, `qm-gpu`, `sim` and `providers`. Å, eV and degrees appear only in the UI and in `references.md`, converted through `units`.
- **Rule 4.** The radial code is f64 throughout. Grid integrals use `Grid::dot`/`Grid::integrate` (Neumaier). Projections c_lm use Neumaier in f64 on the CPU. On the GPU, f32 workgroup partials are finished as Spec question 7 rules, and energies and forces are finished in f64 on the CPU.
- **Rule 6.** The spec's thresholds are copied verbatim into the tests: 1e-5, 1e-10·Z² (hydrogenic, plan review M1), 1e-10, 1e-8, 1e-4 (|HF − Richardson|, plan review S2), 1 mHa, 5 mHa, 20 mHa, 10 mHa/e, < 2 mHa, > 20 mHa, 2e-3, 3e-5, ±2°, ±0.02 Å, ±0.5°, 0.02 Bohr, 5 %, 0.02 Ha, 1000 K and 1e-9 rad. A miss stops the task and gets reported; a threshold is never loosened. A threshold that is wrong on physics grounds is a spec revision, made by the physics reviewer with a derivation (spec §4.8, last paragraph).
- **Rule 7.** Pseudopotential energies are labelled *valence* energies. Refusals (Li, Na, Mg) are reported, never patched. The `CoresOnly` O–H miss, the non-spherical O splitting and core overlaps are all stated as what they are.
- **Ledger and hash.** `LEDGER` is the spec §4.9 table plus the solver knobs those sections name: the radial SCF tolerance, the r_cmp tolerance, the V_SR truncation floor (`v_tail_floor`), and two numerical knobs, the G2 root mesh (`numerics.g2_mesh`) and the King-Smith band basis size (`numerics.kspl_terms`). `recipe_hash` hashes its `Debug` rendering (FNV-1a 64). A change to any field therefore changes the hash, and with it every cached table and surrogate provenance.
- **A moved generation record bumps `LEDGER.version`** (plan review M6). Any change that moves a row of the pinned generation records in `references.md` must change `recipe_hash`. A changed constant does that by itself. A code change that moves a record without touching a constant (a new gate, the filter's h_req, the dimer gate) must bump `LEDGER.version` in the same task. `generation_record_table_matches_references_md` (Task 4) enforces it: a row present in both the pinned and the regenerated block that differs while the hash line is unchanged fails with "bump LEDGER.version". New rows (new elements) need no bump. The bumps planned here: Task 5 (`m7-tm-kb-nlcc-2`, the filter's h_req) and Task 11 (`m7-tm-kb-nlcc-3`, the dimer-gated H).
- **`IonicPolicy::None` is today's solver, bit for bit.** The oracles, `gpu_vs_cpu` at fixed σ and the dynamics suite's H₂ pins run under `None` and must not move (spec §6.1). Pinned by `none_policy_is_bitwise_today` (Task 6) and `gpu_none_policy_is_bitwise_today` (Task 8).
- **Tasks 1–5 are pure CPU and add new files only.** The new files are `crates/qm/src/radial/*.rs`, `crates/validate/tests/pseudo_gen.rs` and new sections of `crates/validate/references.md`. The only edits to existing source files are the line `pub mod radial;` in `crates/qm/src/lib.rs` and `pub(crate)` on `qm::grid::neumaier` (Task 1). **Tasks 1–5 may run while the M6 branch finishes.**
- **Before Task 6, the branch must merge the finished M6.** Once `m6-tools` → `main` has landed with its final fix wave, merge `main` into `m7-pseudopotentials` (Task 6, Step 0). Task 6 is the first task that touches `scf.rs`, `providers` and the worker.
- **Literature digits are pinned at task time from the source, never from memory.** That covers:
  - the NIST LSD totals for C, N, O and Ne, and which functional NIST used (Task 1);
  - the LSDA H₂O/NH₃ geometries from NIST CCCBDB (Task 10);
  - the AME2020 masses for Z = 11–18 (Task 12);
  - the experimental geometries of any new presets (Task 12).

  Each is a task step: "look up X, cite it, pin the digits in `references.md`". If the source cannot be reached, the task stops and reports; it never guesses.
- **Release-only tests.** Every SCF, forces or GPU test carries `#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p <crate> --release")]`, as the oracles do. GPU tests skip cleanly without an adapter (`common::gpu`).
- **Timings are reported, never asserted** (M4 R3). Generation time per element goes into the generation record.
- **Test names.** Rust test names cannot contain `-`, so the spec's `…below_3e-5` tests are named `…below_3e_5`.
- **Commits:** run `git add <paths>`, then `git commit -o -m "<message>" -- <paths>` (memory: commit pathspecs while an implementer shares the checkout). Every message ends with the session's attribution line. Never pipe a gating git command.
- **Keep `README.md` and `CLAUDE.md` current** in the same task that changes user-facing behaviour or test commands (Tasks 9–13).

## Review Focus

The controller suggested several candidates that the spec already tests in named tasks: ghost states (Task 3), the O triplet (Tasks 6 and 10), a scene coarser than h_req (Task 9) and a model across policies (Task 9). Those stay where the spec puts them. The five below are what the spec implies but no spec test exercises, most likely first:

1. **Charge or multiplicity checked against Σ Z instead of Σ Z_v.**
   - Failure: under `CoresOnly`, O⁷⁺ or O at multiplicity 9 passes today's proton check, then `System::with_ionic` panics on a negative or impossible valence count.
   - Expected: `ProviderError::Invalid` naming the valence count.
   - Test: `charge_and_multiplicity_are_checked_against_valence_electrons` (Task 9).
2. **A warm state from another ionic setup with the same orbital counts.**
   - Failure: H under `None` → `All`, or H₂O under `All` before and after an h change, passes today's grid-and-counts check and silently seeds the wrong Hamiltonian.
   - Expected: a cold solve.
   - Tests: `warm_state_from_another_ionic_setup_is_not_reused` (Task 6) and `gpu_warm_state_from_another_ionic_setup_is_cold` (Task 8).
3. **A projector sphere clipped by the box face, or an ion exactly on a grid point.**
   - Failure: the clipped sphere makes V_NL asymmetric, and LOBPCG then misbehaves. An ion on a grid point (d = 0) gives NaN in V_SR, β or ρ_pc.
   - Expected: ⟨a|V_NL b⟩ = ⟨V_NL a|b⟩ to rounding, and finite fields, energies and forces.
   - Tests: `projector_sphere_clipped_by_the_box_stays_symmetric` and `ion_on_a_grid_point_is_finite` (Task 6).
4. **A mixed scene under `CoresOnly`.**
   - Failure: an AE H at σ = h next to a pseudo-O at σ_LR = 2h puts two smearing widths and two force formulas in one molecule.
   - Expected: Hellmann–Feynman forces equal finite differences.
   - Tests: `oh_hf_force_matches_fd` (Task 7), which runs OH under `CoresOnly`, and `gpu_mixed_oh_forces_match_cpu` (Task 8).
5. **An element the recipe refuses** (Li: two valence s states; Na and Mg: semicore).
   - Expected: under `CoresOnly` the species falls back to all-electron with a note. Under `All` the scene is `Invalid`, and the message names the reason and the ONCV stage. Never a panic, never a silent pseudo.
   - Test: `refused_species_falls_back_under_cores_only_and_is_invalid_under_all` (Task 9).

## File Structure

```
crates/qm/src/lib.rs                      # T1: pub mod radial;  T6: pub mod ionic; pub mod nonlocal;
crates/qm/src/grid.rs                     # T1: `neumaier` → pub(crate)
crates/qm/src/radial/mod.rs               # T1 (new): module doc (pipeline overview), re-exports
crates/qm/src/radial/ledger.rs            # T1 (new): LEDGER — the one const table (spec §4.9); T4: hash_ledger, recipe_hash
crates/qm/src/radial/grid.rs              # T1 (new): LogGrid (integrals, cumulative, derivative, interpolation)
crates/qm/src/radial/numerov.rs           # T1 (new): Numerov on the log grid, bound_state (shooting + node count)
crates/qm/src/radial/atom.rs              # T1 (new): radial AE LSDA atom (polarised/restricted), aufbau, S9 + Δ_scan scans, DIIS, energies, scaling identity
crates/qm/src/radial/partition.rs         # T2 (new): core/valence rule, channels with the l_max cap, Refusal
crates/qm/src/radial/tm.rs                # T2 (new): TM construction, V_scr inversion, unscreening, r_cmp
crates/qm/src/radial/kb.rs                # T3 (new): local candidates, KB form, KbInput, radial KB eigenproblem (2×2 determinant), KB atom SCF, G1/G2
crates/qm/src/radial/recipe.rs            # T3 (new): generate (f ladder × local order), PseudoTable; T4: gates in the ladder; T11: DimerGate
crates/qm/src/radial/nlcc.rs              # T4 (new): partial core (LFC), r_nlc, r_core
crates/qm/src/radial/gates.rs             # T4 (new): T1–T7, transfer configurations, log-derivative δE_equiv, hardness
crates/qm/src/radial/record.rs            # T4 (new): GenerationRecord, records_table (references.md block)
crates/qm/src/radial/filter.rs            # T5 (new): spherical Bessel, transforms, King-Smith band completion, mask, FilteredTable (one V_SR table), h_req
crates/qm/src/ionic.rs                    # T6 (new): IonicPolicy, Ionic, IonicSetup, IonicError, core overlaps
crates/qm/src/nonlocal.rs                 # T6 (new): NonlocalOp (spheres, β_lm), PseudoHamiltonian; T7: gradients
crates/qm/src/nuclear.rs                  # T6: Centre, smeared_potential, deposit_centres, ionic_local_potential, partial_core_field, ionic_repulsion
crates/qm/src/scf.rs                      # T6: System.ionic, with_ionic, valence_electrons, KsState.ionic, ScfStats.core_overlaps, the ionic path
crates/qm/src/forces.rs                   # T7: pseudised local (LR + SR), nonlocal, NLCC, ion–ion terms (each a pub(crate) fn, FD-checked)
crates/qm-gpu/src/ionic.rs                # T8 (new): table/sphere upload, dispatch, GpuNonlocal
crates/qm-gpu/src/shaders/ionic.wgsl      # T8 (new): local_sr, nlcc_deposit, proj_dot, proj_finish, proj_axpy, gather_dot3
crates/qm-gpu/src/{kernels.rs,eigen.rs,scf.rs,forces.rs,lib.rs}  # T8
crates/sim/src/scene.rs                   # T9: IonicPolicy, Scene.ionic
crates/sim/src/provider.rs                # T9: CORE_OVERLAP_WARNING, PseudoSummary, EvalStats.{pseudised, core_overlaps}, set_display_partial_core
crates/sim/src/references.rs              # T9: ReferenceKey.ionic
crates/providers/src/pseudo.rs            # T9 (new): PseudoCache (process-wide), setup_for, provenance_tag; T11: the dimer gate
crates/providers/src/{lib.rs,ks.rs}       # T9: problem() via the cache, valence checks, notes, display ρ_pc
crates/providers/src/surrogate/{store.rs,mod.rs}, sampling/mod.rs  # T9: provenance tag, core-overlap samples never kept
crates/surrogate/src/{model.rs,sidecar.rs}  # T9: Provenance.ionic, sidecar version 4
crates/app/src/{scene.rs,main.rs,viewer.rs,overlay.rs,badge.rs,atom_card.rs,replay.rs,screenshot.rs}  # T9, T13
crates/app/src/ui/{mod.rs,actions.rs,menus.rs,search.rs,overlays.rs,ptable.rs}, ui/dock/{readouts.rs,selection.rs}  # T9, T12, T13
crates/app/src/worker/{commands.rs,mod.rs,tests.rs}  # T9
crates/app/replays/*.replay               # T9: the `action ionic none` pin
crates/app/presets/*.scene                # T12: new presets for accepted Z = 11–18 elements
crates/units/src/isotopes.rs              # T12: AME2020 rows Z = 11–18
crates/render/src/elements.rs             # T12: display colours Z = 11–18
crates/validate/tests/pseudo_gen.rs       # T4 (new): per-element gates and the record table; T5, T11, T12 extend
crates/validate/tests/pseudo.rs           # T6 (new): CPU grid tests; T7 extends
crates/validate/tests/pseudo_oracles.rs   # T10 (new): GPU oracles; T11 extends
crates/validate/tests/gpu_vs_cpu.rs       # T8: pseudo-O, pseudo-C, mixed OH
crates/validate/references.md             # T1, T4, T5, T7, T8, T10, T11, T12: new sections
README.md, CLAUDE.md, docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md  # T9–T13
```

---

### Task 1: `qm::radial::atom` — the radial all-electron LSDA atom **[Fable — physics-review gate]**

Spec §4.1, §4.9 (the ledger is created here, complete), §8 row 1.

**Files:**
- Create: `crates/qm/src/radial/mod.rs`, `crates/qm/src/radial/ledger.rs`, `crates/qm/src/radial/grid.rs`, `crates/qm/src/radial/numerov.rs`, `crates/qm/src/radial/atom.rs`
- Modify: `crates/qm/src/lib.rs` (add `pub mod radial;`), `crates/qm/src/grid.rs:91` (`fn neumaier` → `pub(crate) fn neumaier`)
- Modify: `crates/sim/src/sampling.rs` tests (one test pinning `qm::radial::allowed_multiplicities` to `sim::sampling::allowed_multiplicities`; `qm` is already a `sim` dev-dependency)
- Modify: `crates/validate/references.md` (new section "M7 radial atom (Task 1)")
- Test: unit tests in `crates/qm/src/radial/{grid,numerov,atom}.rs`

**Interfaces:**
- Consumes: `qm::xc::exc_point(n_up, n_dn) -> (n·ε_xc, v_up, v_dn)` (unchanged).
- Produces (later tasks use exactly these):
```rust
// ledger.rs — THE table (spec §4.9). Task 4 adds hash_ledger/recipe_hash to this file.
pub enum Xc { LsdaPw92 }
pub enum NlccMode { Lfc, Off }
pub enum LocalCandidate { HighestChannel, Polynomial }
pub struct Gates { pub t1_eig: f64, pub t2_norm: f64, pub t3_delta_e: f64, pub t4_eig: f64, pub t5_de_equiv: f64,
                   pub t5_window: f64, pub t5_mesh: f64, pub t5_rd_factor: f64, pub t6_hardness: f64, pub t6_dq: f64,
                   pub g2_eig: f64, pub g2_window: f64, pub h_req_tol: f64 }
pub struct Numerics { pub scf_tol: f64, pub scf_max_iter: usize, pub diis_history: usize, pub mixing: f64, pub g2_mesh: f64,
                      pub kspl_terms: usize, pub backtrack_max: usize }
pub struct Ledger { pub version: &'static str, pub r0: f64, pub r_max: f64, pub delta: f64, pub multiplicity_max: u32,
                    pub delta_core: f64, pub delta_scan: f64, pub l_max_above_valence: u8, pub f_ladder: [f64; 4],
                    pub local_order: [LocalCandidate; 2], pub nlcc: NlccMode, pub nlcc_crossover: f64, pub tm_degree: u8,
                    pub v_ion_tail: f64, pub gates: Gates, pub q_c_over_nyquist: f64, pub gamma_r: f64,
                    pub sigma_lr_over_h: f64, pub v_tail_floor: f64, pub h_ladder: [f64; 5], pub eps_core: f64,
                    pub dimer_z_max: u8, pub dimer_re: f64, pub dimer_omega_rel: f64, pub dimer_bracket: [f64; 2],
                    pub xc: Xc, pub numerics: Numerics }
pub const LEDGER: Ledger;
// grid.rs
pub struct LogGrid { pub r: Vec<f64>, pub x0: f64, pub delta: f64 }
impl LogGrid { pub fn new(r0: f64, r_max: f64, delta: f64) -> LogGrid; pub fn universal() -> LogGrid; pub fn len(&self) -> usize;
  pub fn integrate(&self, f: &[f64]) -> f64; pub fn cumulative(&self, f: &[f64]) -> Vec<f64>; pub fn partial(&self, f: &[f64], i_end: usize) -> f64;
  pub fn nearest(&self, r: f64) -> usize; pub fn derivative(&self, f: &[f64]) -> Vec<f64>;
  pub fn interp(&self, f: &[f64], r: f64) -> f64; pub fn interp_deriv(&self, f: &[f64], r: f64) -> f64; }
// numerov.rs
pub struct BoundState { pub e: f64, pub u: Vec<f64>, pub r_peak: f64, pub r_node: Option<f64> }
pub fn g_coeff(grid: &LogGrid, v: &[f64], l: u8, e: f64) -> Vec<f64>;
pub fn regular_start(grid: &LogGrid, l: u8, z_origin: f64) -> (f64, f64);
pub fn outward(g: &[f64], y0: f64, y1: f64, i_end: usize, delta: f64) -> (Vec<f64>, usize);
pub fn outward_sourced(g: &[f64], s: &[f64], i_end: usize, delta: f64) -> Vec<f64>;
pub fn inward(g: &[f64], i_match: usize, i_start: usize, delta: f64) -> Vec<f64>;
pub fn shooting_end(g: &[f64], delta: f64) -> usize;
pub fn bound_state(grid: &LogGrid, v: &[f64], l: u8, nodes: usize, z_origin: f64, guess: Option<f64>) -> Option<BoundState>;
pub fn eigenfunction(grid: &LogGrid, v: &[f64], l: u8, e: f64, z_origin: f64) -> BoundState;
// atom.rs
pub type XcFn = fn(f64, f64) -> (f64, f64, f64);
pub enum SpinMode { Polarised, Restricted }
pub struct ShellOcc { pub n: u8, pub l: u8, pub occ: [f64; 2] }
pub enum Occupations { Aufbau { multiplicity: u32 }, Fixed(Vec<ShellOcc>) }
pub struct AtomOptions { pub spin: SpinMode, pub occupations: Occupations, pub charge: i32, pub xc: XcFn }
impl AtomOptions { pub fn aufbau(spin: SpinMode, multiplicity: u32) -> AtomOptions; pub fn fixed(spin: SpinMode, shells: Vec<ShellOcc>) -> AtomOptions; }
pub struct Orbital { pub n: u8, pub l: u8, pub eps: [f64; 2], pub occ: [f64; 2], pub u: [Vec<f64>; 2], pub r_peak: f64, pub r_node: Option<f64> }
pub struct AtomEnergy { pub total: f64, pub kinetic: f64, pub nuclear: f64, pub hartree: f64, pub xc: f64, pub xc_potential: f64 }
pub struct AtomResult { pub z: u8, pub spin: SpinMode, pub multiplicity: u32, pub orbitals: Vec<Orbital>,
                        pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub v_h: Vec<f64>, pub energy: AtomEnergy,
                        pub iterations: usize, pub grid: LogGrid }
pub enum RadialError { NotConverged { z: u8, iterations: usize, delta: f64 }, Unbound { z: u8, n: u8, l: u8 }, Occupation(String) }
pub fn solve_atom(z: u8, opts: &AtomOptions) -> Result<AtomResult, RadialError>;
pub fn ground_state(z: u8, spin: SpinMode) -> Result<AtomResult, RadialError>;          // PW92, S9 scan + Δ_scan scan
pub fn ground_state_with(z: u8, spin: SpinMode, xc: XcFn) -> Result<AtomResult, RadialError>;
pub fn allowed_multiplicities(electrons: u32) -> Vec<u32>;
pub fn hartree_potential(grid: &LogGrid, rho: &[f64]) -> Vec<f64>;
pub fn scaling_identity_residual(a: &AtomResult) -> f64;                               // spec §4.1, / T_s
pub fn next_unoccupied(a: &AtomResult, l: u8, spin: usize) -> Option<(u8, BoundState)>;  // the lowest unoccupied state of l
```
Conventions (all later tasks): radial functions are in **u-form** (u = r·R) on `LogGrid::universal()`, normalised as ∫u² dr = 1. Densities ρ(r) are 3-D (e/Bohr³). Potentials are the electron's potential energy (Hartree). Spin index 0 = ↑, 1 = ↓. In `Restricted` mode both spin entries are equal and each holds half of the shell occupation.

- [ ] **Step 1: Look up the NIST reference (do not type digits from memory).** Open NIST "Atomic Reference Data for Electronic Structure Calculations" (Kotochigova, Levine, Shirley, Stiles & Clark, 1997; https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations).
  1. Record **which correlation functional the LSD column uses**, and **its spin interpolation** (plan review Q1(i): VWN's α_c form, ε_c = ε_P + α_c·f(ζ)/f″(0)·(1 − ζ⁴) + (ε_F − ε_P)·f(ζ)ζ⁴, which `vwn5::eps_c` codes). The plan review measured the column to be VWN5 (the Ceperley–Alder fit): VWN5 reproduces H and He to ~1e-6, while PW92 misses the 1e-5 gate by 4× (H) and 38× (He). H (ζ = 1) and He (ζ = 0) test the ferro and para fits alone; only Li and C–Ne test the interpolation, so a different NIST interpolation would make Li miss for a reason that is not the solver. See Spec question 1.
  2. Record the LSD total energies of **C, N, O and Ne** with every digit printed.
  3. Add a section "M7 radial atom (Task 1)" to `crates/validate/references.md`. It has a table of (element, LSD total, source, test) for H −0.478671, He −2.834836 and Li −7.335195 (already cited there), plus the four new rows and a line naming the functional.

  If the page cannot be reached, stop and report.

- [ ] **Step 2: Create the ledger (the complete §4.9 table, used by every later task)**

`crates/qm/src/radial/ledger.rs`:
```rust
//! The recipe's universal constants — spec §4.9, "the complete rule-1 ledger" — as ONE `const`
//! table. Every number the generator (`qm::radial`) and its consumers (the filter, `qm::ionic`, the
//! dimer gate) use is a field of [`LEDGER`]; no other file under `crates/qm/src/radial/` declares a
//! numeric constant (`gate_thresholds_are_constants`, Task 4). Nothing is indexed by Z. A change to
//! any value is a recipe change: every element is regenerated and re-gated, and `recipe_hash`
//! (Task 4: FNV-1a 64 of this table's `Debug` rendering, which includes `version` and `xc`) changes,
//! invalidating cached tables and surrogate provenances.
//!
//! `numerics` holds the solver knobs the spec names in §4.1/§4.6 (radial SCF tolerance, DIIS), the
//! G2 root-location mesh (Spec question 5: the ghost count itself is the exact inertia count; the
//! mesh only locates roots for the 1 mHa comparison) and the King-Smith band-completion basis size
//! (Spec question 3). They are numerical knobs, not rule-1 quantities: they do not change a converged
//! result beyond their tolerance, but they are hashed too.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xc { LsdaPw92 }

/// Nonlinear core correction mode: `Lfc` (Louie–Froyen–Cohen, spec §4.5); `Off` is the switch the
/// precision milestone's QMC variant needs (spec §3, "Future requirement").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NlccMode { Lfc, Off }

/// Local-channel candidates in the universal preference order (spec §4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCandidate { HighestChannel, Polynomial }

/// Gate thresholds (spec §4.6, §4.8). Energies in Hartree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    pub t1_eig: f64, pub t2_norm: f64, pub t3_delta_e: f64, pub t4_eig: f64,
    pub t5_de_equiv: f64, pub t5_window: f64, pub t5_mesh: f64, pub t5_rd_factor: f64,
    pub t6_hardness: f64, pub t6_dq: f64,
    pub g2_eig: f64, pub g2_window: f64,
    pub h_req_tol: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Numerics {
    pub scf_tol: f64, pub scf_max_iter: usize, pub diis_history: usize, pub mixing: f64,
    /// G2: the mesh (Ha) that locates KB roots for the eigenvalue comparison (Spec question 5).
    pub g2_mesh: f64,
    /// The filter: Chebyshev terms of the King-Smith band (q_c, π/h] (Spec question 3; the plan
    /// review's prototype used 24).
    pub kspl_terms: usize,
    /// DIIS steps backtracked to linear mixing before giving up (Task 1 review MUST d).
    pub backtrack_max: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ledger {
    pub version: &'static str,
    /// Radial log grid r_i = r0·e^{iδ} up to r_max (Bohr) — §4.1.
    pub r0: f64, pub r_max: f64, pub delta: f64,
    /// Ruling S9's multiplicity cap (`sim::sampling::allowed_multiplicities`; a `sim` test pins them equal).
    pub multiplicity_max: u32,
    /// Semicore promotion window and occupation-scan window (Ha) — §4.2, §4.1.
    pub delta_core: f64, pub delta_scan: f64,
    /// l_max = max occupied valence l + this — §4.2.
    pub l_max_above_valence: u8,
    /// The r_c ladder, softest first — §4.4.
    pub f_ladder: [f64; 4],
    pub local_order: [LocalCandidate; 2],
    pub nlcc: NlccMode,
    /// r_nlc where ρ_c = nlcc_crossover · ρ_v (LFC's factor 1) — §4.5.
    pub nlcc_crossover: f64,
    /// TM polynomial degree (7 coefficients) — §4.3.
    pub tm_degree: u8,
    /// r_cmp: where |V_ion + Z_v/r| < this (Ha) for good — §4.3.
    pub v_ion_tail: f64,
    pub gates: Gates,
    /// q_c = this · π/h; tables truncated at gamma_r · R (R the function's support) — §5.1.
    pub q_c_over_nyquist: f64, pub gamma_r: f64,
    /// σ_LR = this · h — §5.1. The reason is spectral (plan review Q2/Q6): V_LR = −Z_v erf(d/a)/d,
    /// a = σ_LR√2, carries a relative weight e^{−(q_c a/2)²} ≈ 6e-5 above q_c at a = 2√2·h, so σ_LR
    /// must scale with h.
    pub sigma_lr_over_h: f64,
    /// The V_SR table's floor (Ha) — §5.1: r_floor solves Z_v·erfc(r/a)/r = this, and V_SR's support
    /// radius is R_SR = max(r_cmp, r_floor) (r_floor ≈ 2.5 Bohr at h = 0.25, ≈ 3.5 at h = 0.35 for Z_v = 6; for Be and Al–S r_cmp dominates — Al 4.18, Si 3.65, P 3.24, measured in Task 2).
    pub v_tail_floor: f64,
    /// The h ladder for h_req (Bohr), coarsest first — §4.8 T8.
    pub h_ladder: [f64; 5],
    /// r_core encloses all but this fraction of the core electrons — §5.6.
    pub eps_core: f64,
    /// The dimer gate on f applies for Z ≤ dimer_z_max: R_e within dimer_re (Bohr), ω within dimer_omega_rel — §4.4, §6.4.
    pub dimer_z_max: u8, pub dimer_re: f64, pub dimer_omega_rel: f64,
    /// The dimer gate's bond-length bracket, in units of the element's outermost valence r_peak
    /// (Task 11; Spec question 9): a bound AE dimer must have its minimum inside it. Why [1, 4]
    /// (plan review Q9(i)): covalent R_e lies at ≈ 1.5–3 r_peak (H₂: 1.45 Bohr, r_peak = 1), while
    /// LSDA's spurious rare-gas minima lie beyond 4 r_peak — a universal rule, not a number chosen
    /// to make He "not applicable".
    pub dimer_bracket: [f64; 2],
    pub xc: Xc,
    pub numerics: Numerics,
}

pub const LEDGER: Ledger = Ledger {
    version: "m7-tm-kb-nlcc-1",
    r0: 1e-5, r_max: 60.0, delta: 0.008,
    multiplicity_max: 5,
    delta_core: 2.0, delta_scan: 0.1,
    l_max_above_valence: 1,
    f_ladder: [1.6, 1.4, 1.2, 1.0],
    local_order: [LocalCandidate::HighestChannel, LocalCandidate::Polynomial],
    nlcc: NlccMode::Lfc, nlcc_crossover: 1.0,
    tm_degree: 12,
    v_ion_tail: 1e-8,
    gates: Gates {
        t1_eig: 1e-3, t2_norm: 1e-10, t3_delta_e: 5e-3, t4_eig: 5e-3,
        t5_de_equiv: 20e-3, t5_window: 1.0, t5_mesh: 0.01, t5_rd_factor: 1.2,
        t6_hardness: 10e-3, t6_dq: 0.25,
        g2_eig: 1e-3, g2_window: 0.5,
        h_req_tol: 1e-3,
    },
    q_c_over_nyquist: 0.7, gamma_r: 1.5,
    sigma_lr_over_h: 2.0,
    v_tail_floor: 1e-6,
    h_ladder: [0.35, 0.30, 0.25, 0.20, 0.15],
    eps_core: 1e-3,
    dimer_z_max: 2, dimer_re: 0.02, dimer_omega_rel: 0.05, dimer_bracket: [1.0, 4.0],
    xc: Xc::LsdaPw92,
    numerics: Numerics { scf_tol: 1e-10, scf_max_iter: 400, diis_history: 6, mixing: 0.3, g2_mesh: 1e-3, kspl_terms: 24, backtrack_max: 6 },
};
```

- [ ] **Step 3: Write the failing tests**

In `crates/qm/src/radial/atom.rs`, `#[cfg(test)] mod tests`:
```rust
use super::*;
use crate::radial::numerov::bound_state;

/// The functional of the NIST LSD column, like-for-like (Spec question 1). Step 1 decides which of
/// the two lines is kept: VWN5 if NIST used Vosko–Wilk–Nusair, else PW92 (the generator's).
const NIST_XC: XcFn = vwn5::exc_point;
// const NIST_XC: XcFn = crate::xc::exc_point;

fn lsd(z: u8) -> f64 { ground_state_with(z, SpinMode::Polarised, NIST_XC).expect("converges").energy.total }

/// Spec §4.1 / §8 row 1, the first failing test.
#[test]
fn h_atom_radial_lsd() {
    let e = lsd(1);
    assert!((e - (-0.478671)).abs() < 1e-5, "H LSD = {e:.7} vs NIST −0.478671");
}

#[test]
fn he_atom_radial_lsd() {
    let e = lsd(2);
    assert!((e - (-2.834836)).abs() < 1e-5, "He LSD = {e:.7} vs NIST −2.834836");
}

#[test]
fn li_atom_radial_lsd() {
    let e = lsd(3);
    assert!((e - (-7.335195)).abs() < 1e-5, "Li LSD = {e:.7} vs NIST −7.335195");
}

// NIST LSD totals of C, N, O, Ne, read in Step 1 (spec §4.1: "read off at task time"; cited in
// references.md "M7 radial atom"). The zeros are replaced by the source's digits before the run;
// while they are 0.0 the test fails, by design (never typed from memory).
const NIST_LSD_C: f64 = 0.0;  // Step 1: replace 0.0 by the NIST digits
const NIST_LSD_N: f64 = 0.0;  // Step 1: replace 0.0 by the NIST digits
const NIST_LSD_O: f64 = 0.0;  // Step 1: replace 0.0 by the NIST digits
const NIST_LSD_NE: f64 = 0.0; // Step 1: replace 0.0 by the NIST digits

/// The second-row LSD totals to 1e-4 (spec §4.1).
#[test]
fn second_row_radial_lsd() {
    for (z, nist) in [(6u8, NIST_LSD_C), (7, NIST_LSD_N), (8, NIST_LSD_O), (10, NIST_LSD_NE)] {
        let e = lsd(z);
        assert!((e - nist).abs() < 1e-4, "Z = {z}: LSD = {e:.6} vs NIST {nist:.6}");
    }
}

/// Spec §4.1: Hund's triplet is the lowest of M = 1, 3, 5 for O in the spherical LSDA atom (PW92).
#[test]
fn o_triplet_is_lowest() {
    let e = |m| solve_atom(8, &AtomOptions::aufbau(SpinMode::Polarised, m)).expect("converges").energy.total;
    let (e1, e3, e5) = (e(1), e(3), e(5));
    assert!(e3 < e1 && e3 < e5, "O: E(M=1) = {e1:.6}, E(3) = {e3:.6}, E(5) = {e5:.6}");
    assert_eq!(ground_state(8, SpinMode::Polarised).unwrap().multiplicity, 3);
}

/// Spec §4.1 (review M2): 2T_s + E_en + E_H + 3[Σ_σ∫ρ_σ v_xc,σ − E_xc] = 0 to 1e-5 relative to T_s,
/// for every converged atom, both spin modes (PW92, the generator's functional).
#[test]
fn lsda_scaling_identity_holds() {
    for z in [1u8, 2, 3, 6, 7, 8, 10] {
        for spin in [SpinMode::Polarised, SpinMode::Restricted] {
            let a = ground_state(z, spin).expect("converges");
            let r = scaling_identity_residual(&a);
            assert!(r.abs() < 1e-5, "Z = {z} {spin:?}: residual {r:.2e} (T_s {:.6})", a.energy.kinetic);
        }
    }
}

/// The integrator alone (V_H and v_xc off): |ε_nl − (−Z²/2n²)| < 1e-10·Z² (spec §4.1 as revised by
/// plan review M1). Numerov's eigenvalue error is O(δ⁴) relative — measured ≈ (0.008, 0.047, 0.17)·δ⁴
/// for 1s, 2s, 3s — and the log grid is scale-free, so the absolute error grows as Z²: a fixed 1e-10
/// fails 7 of these 19 states at δ = 0.008 (Z = 8 2s: 1.5e-9). The worst margin under the Z² bound
/// is 2.5× (Z = 2, 3s).
#[test]
fn hydrogenic_eigenvalues_are_exact() {
    let g = LogGrid::universal();
    let shells: [(f64, &[(u8, u8)]); 4] = [
        (1.0, &[(1, 0), (2, 0), (2, 1), (3, 2)]),
        (2.0, &[(1, 0), (2, 0), (2, 1), (3, 0), (3, 2), (4, 3)]),
        (3.0, &[(1, 0), (2, 1), (3, 1), (4, 3)]),
        (8.0, &[(1, 0), (2, 0), (2, 1), (3, 2), (4, 3)]),
    ];
    for (z, list) in shells {
        let v: Vec<f64> = g.r.iter().map(|r| -z / r).collect();
        for &(n, l) in list {
            let b = bound_state(&g, &v, l, (n - l - 1) as usize, z, None).expect("bound");
            let exact = -z * z / (2.0 * (n as f64).powi(2));
            assert!((b.e - exact).abs() < 1e-10 * z * z, "Z = {z}, n = {n}, l = {l}: ε = {:.13} vs {exact:.13} (Δ {:.2e}, bound {:.1e})", b.e, b.e - exact, 1e-10 * z * z);
        }
    }
}

/// Plan review M1: the order of the integrator. The Z = 1 1s error must fall by 2⁴ = 16 when δ halves
/// from 0.016 to 0.008 — the ratio in [14, 18] (measured 16.1). An absolute bound at one δ would let
/// a wrong Numerov weight through; the ratio does not.
#[test]
fn hydrogenic_error_is_fourth_order_in_delta() {
    let err = |delta: f64| {
        let g = LogGrid::new(LEDGER.r0, LEDGER.r_max, delta);
        let v: Vec<f64> = g.r.iter().map(|r| -1.0 / r).collect();
        (bound_state(&g, &v, 0, 0, 1.0, None).expect("bound").e + 0.5).abs()
    };
    let (coarse, fine) = (err(2.0 * LEDGER.delta), err(LEDGER.delta));
    let ratio = coarse / fine;
    assert!((14.0..=18.0).contains(&ratio), "1s error {coarse:.3e} at δ = {} vs {fine:.3e} at δ = {}: ratio {ratio:.2}", 2.0 * LEDGER.delta, LEDGER.delta);
}

/// Spec §4.1: "For Z ≤ 18 the aufbau order is unambiguous and the scan is a no-op."
#[test]
fn occupation_scan_is_a_noop_below_z19() {
    for z in 1u8..=18 {
        let plain = solve_atom(z, &AtomOptions::aufbau(SpinMode::Restricted, 1)).expect("converges");
        let scanned = ground_state(z, SpinMode::Restricted).expect("converges");
        let occ = |a: &AtomResult| a.orbitals.iter().map(|o| (o.n, o.l, (o.occ[0] * 1e6).round() as i64)).collect::<Vec<_>>();
        assert_eq!(occ(&plain), occ(&scanned), "Z = {z}");
    }
}

/// The NIST-functional comparison is like-for-like; the generator's PW92 totals are reported against
/// it (rule 7), never asserted.
#[test]
fn report_pw92_against_nist() {
    for z in [1u8, 2, 3, 6, 7, 8, 10] {
        let pw = ground_state(z, SpinMode::Polarised).unwrap().energy.total;
        eprintln!("Z = {z:2}: PW92 {pw:.6}  NIST-functional {:.6}  Δ {:+.2e}", lsd(z), pw - lsd(z));
    }
}
```

Also write the test-only VWN5 module. It is needed only if Step 1 found that NIST used VWN; otherwise delete it together with the `NIST_XC` line that names it.

```rust
/// Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200 (1980), the Ceperley–Alder fit ("VWN5") — TEST-ONLY:
/// the functional of the NIST LSD column, so the radial solver is compared like-for-like (Spec
/// question 1). Verify the twelve constants against VWN 1980 (and libxc's `lda_c_vwn`) at task time.
/// Exchange: Dirac, spin-scaled (formulas in `qm::xc`'s module doc). Potentials by central
/// differences of n·ε_c with the step 1e-6·(n↑ + n↓) for both spins (plan review Q1(ii): a step
/// relative to n↓ alone loses ~1 % of v_c↓ to rounding when n↓/n↑ ≲ 1e-8; error ≪ 1e-10 Ha in the
/// totals).
#[cfg(test)]
pub(crate) mod vwn5 {
    use std::f64::consts::PI;
    fn g(x: f64, a: f64, x0: f64, b: f64, c: f64) -> f64 {
        let xx = |y: f64| y * y + b * y + c;
        let q = (4.0 * c - b * b).sqrt();
        let at = (q / (2.0 * x + b)).atan();
        a * ((x * x / xx(x)).ln() + 2.0 * b / q * at
            - b * x0 / xx(x0) * (((x - x0).powi(2) / xx(x)).ln() + 2.0 * (b + 2.0 * x0) / q * at))
    }
    fn eps_c(nu: f64, nd: f64) -> f64 {
        let n = nu + nd;
        let rs = (3.0 / (4.0 * PI * n)).powf(1.0 / 3.0);
        let zeta = ((nu - nd) / n).clamp(-1.0, 1.0);
        let x = rs.sqrt();
        let p = g(x, 0.0310907, -0.10498, 3.72744, 12.9352);
        let f = g(x, 0.01554535, -0.32500, 7.06042, 18.0578);
        let alpha = g(x, -1.0 / (6.0 * PI * PI), -0.0047584, 1.13107, 13.0045);
        let d = 2f64.powf(4.0 / 3.0) - 2.0;
        let fz = ((1.0 + zeta).powf(4.0 / 3.0) + (1.0 - zeta).powf(4.0 / 3.0) - 2.0) / d;
        let fpp0 = 8.0 / (9.0 * d);
        let z4 = zeta.powi(4);
        p + alpha * fz / fpp0 * (1.0 - z4) + (f - p) * fz * z4
    }
    pub fn exc_point(nu: f64, nd: f64) -> (f64, f64, f64) {
        let (nu, nd) = (nu.max(0.0), nd.max(0.0));
        if nu + nd < 1e-20 { return (0.0, 0.0, 0.0); }
        let c = 0.75 * (3.0 / PI).powf(1.0 / 3.0) * 2f64.powf(1.0 / 3.0);
        let ex = -c * (nu.powf(4.0 / 3.0) + nd.powf(4.0 / 3.0));
        let (vxu, vxd) = (-(6.0 * nu / PI).powf(1.0 / 3.0), -(6.0 * nd / PI).powf(1.0 / 3.0));
        let ec = |a: f64, b: f64| if a + b < 1e-20 { 0.0 } else { (a + b) * eps_c(a, b) };
        let step = 1e-6 * (nu + nd);
        let (du, dd) = (step, step);
        let vcu = (ec(nu + du, nd) - ec((nu - du).max(0.0), nd)) / (nu + du - (nu - du).max(0.0));
        let vcd = (ec(nu, nd + dd) - ec(nu, (nd - dd).max(0.0))) / (nd + dd - (nd - dd).max(0.0));
        (ex + ec(nu, nd), vxu + vcu, vxd + vcd)
    }
}
```
In `crates/sim/src/sampling.rs` tests:
```rust
#[test]
fn radial_multiplicities_mirror_ruling_s9() {
    for n in 0u32..=40 { assert_eq!(qm::radial::allowed_multiplicities(n), allowed_multiplicities(n), "N = {n}"); }
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test -p qm --release radial`
Expected: compile errors (the `radial` module does not exist).

- [ ] **Step 5: Implement `LogGrid`**

`crates/qm/src/radial/grid.rs`:
```rust
//! The universal logarithmic radial grid r_i = r0·e^{iδ} (spec §4.1; `LEDGER.r0/r_max/delta`).
//! Integrals over r become integrals over x = ln r with dr = r dx. `integrate` is the trapezoid in x:
//! for integrands f·r that vanish smoothly at both ends (every bound-state quantity here) the
//! Euler–Maclaurin corrections vanish and the rule is exponentially accurate. `cumulative` is
//! 4th order (cubic Lagrange through four neighbours); ∫₀^{r0} is neglected (≤ ρ(0)r0³ ≈ 1e-12 for Ne).
use super::ledger::LEDGER;
use crate::grid::neumaier;

#[derive(Clone, Debug, PartialEq)]
pub struct LogGrid { pub r: Vec<f64>, pub x0: f64, pub delta: f64 }

impl LogGrid {
    pub fn new(r0: f64, r_max: f64, delta: f64) -> LogGrid {
        let n = ((r_max / r0).ln() / delta).floor() as usize + 1;
        LogGrid { r: (0..n).map(|i| r0 * (i as f64 * delta).exp()).collect(), x0: r0.ln(), delta }
    }
    pub fn universal() -> LogGrid { LogGrid::new(LEDGER.r0, LEDGER.r_max, LEDGER.delta) }
    pub fn len(&self) -> usize { self.r.len() }

    /// ∫₀^{r_max} f dr (trapezoid in x, Neumaier).
    pub fn integrate(&self, f: &[f64]) -> f64 {
        debug_assert_eq!(f.len(), self.len());
        let n = self.len();
        let inner = neumaier(f.iter().zip(&self.r).map(|(a, r)| a * r));
        (inner - 0.5 * (f[0] * self.r[0] + f[n - 1] * self.r[n - 1])) * self.delta
    }

    /// c_i = ∫₀^{r_i} f dr for every i (4th order in x).
    pub fn cumulative(&self, f: &[f64]) -> Vec<f64> {
        let n = self.len();
        let g: Vec<f64> = f.iter().zip(&self.r).map(|(a, r)| a * r).collect();
        let w = self.delta / 24.0;
        let mut c = vec![0.0; n];
        c[1] = w * (9.0 * g[0] + 19.0 * g[1] - 5.0 * g[2] + g[3]);
        for i in 1..n - 2 {
            c[i + 1] = c[i] + w * (-g[i - 1] + 13.0 * g[i] + 13.0 * g[i + 1] - g[i + 2]);
        }
        c[n - 1] = c[n - 2] + w * (g[n - 4] - 5.0 * g[n - 3] + 19.0 * g[n - 2] + 9.0 * g[n - 1]);
        c
    }

    /// ∫₀^{r_{i_end}} f dr.
    pub fn partial(&self, f: &[f64], i_end: usize) -> f64 { self.cumulative(f)[i_end] }

    /// The grid index nearest to r (in x).
    pub fn nearest(&self, r: f64) -> usize {
        (((r.ln() - self.x0) / self.delta).round().max(0.0) as usize).min(self.len() - 1)
    }

    /// df/dr = (df/dx)/r, 5-point centred in x, one-sided 5-point at both ends.
    pub fn derivative(&self, f: &[f64]) -> Vec<f64> {
        let n = self.len();
        let d = 12.0 * self.delta;
        (0..n).map(|i| {
            let fx = if i >= 2 && i + 2 < n {
                (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) / d
            } else if i < 2 {
                let o = 0;
                if i == 0 { (-25.0 * f[o] + 48.0 * f[o + 1] - 36.0 * f[o + 2] + 16.0 * f[o + 3] - 3.0 * f[o + 4]) / d }
                else { (-3.0 * f[o] - 10.0 * f[o + 1] + 18.0 * f[o + 2] - 6.0 * f[o + 3] + f[o + 4]) / d }
            } else {
                let o = n - 5;
                if i == n - 1 { (3.0 * f[o] - 16.0 * f[o + 1] + 36.0 * f[o + 2] - 48.0 * f[o + 3] + 25.0 * f[o + 4]) / d }
                else { (-f[o] + 6.0 * f[o + 1] - 18.0 * f[o + 2] + 10.0 * f[o + 3] + 3.0 * f[o + 4]) / d }
            };
            fx / self.r[i]
        }).collect()
    }

    /// 4-point Lagrange interpolation in x; f[0] below r0, f[n−1] beyond r_max (callers bound the
    /// support themselves). `interp_deriv` is d/dr of the same interpolant (so a force taken with it
    /// is the exact derivative of an energy sampled with `interp`).
    pub fn interp(&self, f: &[f64], r: f64) -> f64 { self.lagrange(f, r).0 }
    pub fn interp_deriv(&self, f: &[f64], r: f64) -> f64 { self.lagrange(f, r).1 }

    fn lagrange(&self, f: &[f64], r: f64) -> (f64, f64) {
        let n = self.len();
        if r <= self.r[0] { return (f[0], 0.0); }
        if r >= self.r[n - 1] { return (f[n - 1], 0.0); }
        let s = (r.ln() - self.x0) / self.delta;
        let i0 = (s.floor() as usize).saturating_sub(1).min(n - 4);
        let t = s - i0 as f64;
        let (a, b, c, d) = (f[i0], f[i0 + 1], f[i0 + 2], f[i0 + 3]);
        let l = [-(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0, t * (t - 2.0) * (t - 3.0) / 2.0,
                 -t * (t - 1.0) * (t - 3.0) / 2.0, t * (t - 1.0) * (t - 2.0) / 6.0];
        let dl = [-((t - 2.0) * (t - 3.0) + (t - 1.0) * (t - 3.0) + (t - 1.0) * (t - 2.0)) / 6.0,
                  ((t - 2.0) * (t - 3.0) + t * (t - 3.0) + t * (t - 2.0)) / 2.0,
                  -((t - 1.0) * (t - 3.0) + t * (t - 3.0) + t * (t - 1.0)) / 2.0,
                  ((t - 1.0) * (t - 2.0) + t * (t - 2.0) + t * (t - 1.0)) / 6.0];
        let v = a * l[0] + b * l[1] + c * l[2] + d * l[3];
        let dv_ds = a * dl[0] + b * dl[1] + c * dl[2] + d * dl[3];
        (v, dv_ds / (self.delta * r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integrals_of_a_hydrogenic_density() {
        // ∫ 4 r² e^{−2r} dr = 1; the cumulative to r: 1 − e^{−2r}(1 + 2r + 2r²)
        let g = LogGrid::universal();
        let f: Vec<f64> = g.r.iter().map(|r| 4.0 * r * r * (-2.0 * r).exp()).collect();
        assert!((g.integrate(&f) - 1.0).abs() < 1e-12);
        let c = g.cumulative(&f);
        let i = g.nearest(1.3);
        let r = g.r[i];
        let exact = 1.0 - (-2.0 * r).exp() * (1.0 + 2.0 * r + 2.0 * r * r);
        assert!((c[i] - exact).abs() < 1e-11, "{} vs {exact}", c[i]);
    }
    #[test]
    fn derivative_and_interpolation_of_a_smooth_function() {
        let g = LogGrid::universal();
        let f: Vec<f64> = g.r.iter().map(|r| (-r).exp() * r).collect();
        let d = g.derivative(&f);
        for &r in &[0.01, 0.5, 2.0, 7.0] {
            let i = g.nearest(r);
            let ri = g.r[i];
            assert!((d[i] - (-ri).exp() * (1.0 - ri)).abs() < 1e-9);
            let x = r * 1.0037;
            assert!((g.interp(&f, x) - x * (-x).exp()).abs() < 1e-10);
            assert!((g.interp_deriv(&f, x) - (-x).exp() * (1.0 - x)).abs() < 1e-7);
        }
    }
}
```

- [ ] **Step 6: Implement Numerov on the log grid**

`crates/qm/src/radial/numerov.rs`:
```rust
//! The radial Schrödinger equation −½u″ + [l(l+1)/2r² + V(r)]u = εu on the log grid, by Numerov's
//! rule in x = ln r (spec §4.1; Hartree 1957, Herman & Skillman 1963, Fuchs & Scheffler 1999).
//!
//! # The transformed equation (derivation)
//! With u(r) = e^{x/2} y(x): du/dr = e^{−x/2}(y′ + y/2), d²u/dr² = e^{−3x/2}(y″ − y/4). Substituting,
//!   y″ = g(x) y,   g = (l + ½)² + 2r²(V(r) − ε)
//! — no first derivative, so Numerov applies: y_{i+1}(1 − k g_{i+1}) = 2y_i(1 + 5k g_i) − y_{i−1}(1 − k g_{i−1}),
//! k = δ²/12, local error O(δ⁶), eigenvalue error O(δ⁴). With a source, y″ = g y + S,
//! −½u″ + (V_eff − ε)u = −s(r) gives S = 2r^{3/2} s, added as k(S_{i+1} + 10S_i + S_{i−1}).
//! ∫u² dr = ∫ r² y² dx.
//!
//! # Regular start
//! u ∝ r^{l+1}(1 − Z r/(l+1) + …) at a Coulomb origin of charge Z (`z_origin`; 0 for a finite
//! pseudopotential), so y = r^{l+½}(1 − Z r/(l+1)) at the first two points (error O(r0²) relative).
//!
//! # Shooting (`bound_state`)
//! The outward solution at energy e has k(e) sign changes up to `shooting_end` (the point beyond the
//! last turning point where the WKB decay ∫√g dx exceeds 40, or k·g > 0.1 where Numerov's denominator
//! would degrade). k(e) is non-decreasing; the state with `nodes` radial nodes is the boundary between
//! k = nodes and k = nodes + 1 (the growing tail flips sign there). Bisection to 1e-13·max(1, |e|).
//! Unbound (`None`) when even e = 0 gives k ≤ nodes (r_max = 60 Bohr is the box). The eigenfunction:
//! outward to the outermost turning point, inward from `shooting_end`, scaled to match, normalised;
//! sign: positive at its outermost maximum.

use super::grid::LogGrid;

#[derive(Clone, Debug, PartialEq)]
pub struct BoundState { pub e: f64, pub u: Vec<f64>, pub r_peak: f64, pub r_node: Option<f64> }

pub fn g_coeff(grid: &LogGrid, v: &[f64], l: u8, e: f64) -> Vec<f64> {
    let lh = l as f64 + 0.5;
    grid.r.iter().zip(v).map(|(&r, &vi)| lh * lh + 2.0 * r * r * (vi - e)).collect()
}

pub fn regular_start(grid: &LogGrid, l: u8, z_origin: f64) -> (f64, f64) {
    let f = |r: f64| r.powf(l as f64 + 0.5) * (1.0 - z_origin * r / (l as f64 + 1.0));
    (f(grid.r[0]), f(grid.r[1]))
}

/// Homogeneous outward run over 0..=i_end; returns y and its sign changes. Rescales the run by
/// 1e-150 whenever |y| > 1e150 (signs and nodes unchanged).
pub fn outward(g: &[f64], y0: f64, y1: f64, i_end: usize, delta: f64) -> (Vec<f64>, usize) {
    let k = delta * delta / 12.0;
    let mut y = vec![0.0; i_end + 1];
    y[0] = y0;
    if i_end >= 1 { y[1] = y1; }
    let mut nodes = 0;
    for i in 1..i_end {
        y[i + 1] = (2.0 * y[i] * (1.0 + 5.0 * k * g[i]) - y[i - 1] * (1.0 - k * g[i - 1])) / (1.0 - k * g[i + 1]);
        if y[i] * y[i + 1] < 0.0 { nodes += 1; }
        if y[i + 1].abs() > 1e150 { for v in &mut y[..=i + 1] { *v *= 1e-150; } }
    }
    (y, nodes)
}

/// The regular particular solution of y″ = g y + s (y = 0 at the first two points: the source is
/// O(r^{l+1}) in u-form, so the particular solution is O(r^{l+3})).
pub fn outward_sourced(g: &[f64], s: &[f64], i_end: usize, delta: f64) -> Vec<f64> {
    let k = delta * delta / 12.0;
    let mut y = vec![0.0; i_end + 1];
    for i in 1..i_end {
        y[i + 1] = (2.0 * y[i] * (1.0 + 5.0 * k * g[i]) - y[i - 1] * (1.0 - k * g[i - 1])
            + k * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / (1.0 - k * g[i + 1]);
    }
    y
}

/// Decaying inward run from i_start down to i_match; element j ↔ grid index i_match + j.
pub fn inward(g: &[f64], i_match: usize, i_start: usize, delta: f64) -> Vec<f64> {
    let k = delta * delta / 12.0;
    let m = i_start - i_match;
    let mut y = vec![0.0; m + 1];
    y[m] = 1e-20;
    y[m - 1] = 1e-20 * (g[i_start].max(0.0).sqrt() * delta).exp();
    for j in (1..m).rev() {
        let i = i_match + j;
        y[j - 1] = (2.0 * y[j] * (1.0 + 5.0 * k * g[i]) - y[j + 1] * (1.0 - k * g[i + 1])) / (1.0 - k * g[i - 1]);
        if y[j - 1].abs() > 1e150 { for v in &mut y[j - 1..] { *v *= 1e-150; } }
    }
    y
}

pub fn shooting_end(g: &[f64], delta: f64) -> usize {
    let n = g.len();
    let it = (0..n).rev().find(|&i| g[i] < 0.0).unwrap_or(0);
    let k = delta * delta / 12.0;
    let mut acc = 0.0;
    for i in it..n {
        if k * g[i] > 0.1 { return i; }
        acc += g[i].max(0.0).sqrt() * delta;
        if acc > 40.0 { return i; }
    }
    n - 1
}

pub fn bound_state(grid: &LogGrid, v: &[f64], l: u8, nodes: usize, z_origin: f64, guess: Option<f64>) -> Option<BoundState> {
    let n = grid.len();
    let (y0, y1) = regular_start(grid, l, z_origin);
    let count = |e: f64| {
        let g = g_coeff(grid, v, l, e);
        outward(&g, y0, y1, shooting_end(&g, grid.delta), grid.delta).1
    };
    let mut e_hi = 0.0;
    if count(e_hi) <= nodes { return None; }
    let lf = l as f64;
    let mut e_lo = (0..n).map(|i| v[i] + lf * (lf + 1.0) / (2.0 * grid.r[i] * grid.r[i])).fold(f64::INFINITY, f64::min);
    if let Some(gs) = guess {
        let w = 0.1 * gs.abs().max(0.1);
        if count(gs - w) <= nodes { e_lo = gs - w; }
        if gs + w < 0.0 && count(gs + w) > nodes { e_hi = gs + w; }
    }
    while e_hi - e_lo > 1e-13 * e_lo.abs().max(1.0) {
        let mid = 0.5 * (e_lo + e_hi);
        if count(mid) > nodes { e_hi = mid } else { e_lo = mid }
    }
    Some(eigenfunction(grid, v, l, 0.5 * (e_lo + e_hi), z_origin))
}

pub fn eigenfunction(grid: &LogGrid, v: &[f64], l: u8, e: f64, z_origin: f64) -> BoundState {
    let n = grid.len();
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, z_origin);
    let it = (0..n).rev().find(|&i| g[i] < 0.0).unwrap_or(n / 2).clamp(2, n - 3);
    let ie = shooting_end(&g, grid.delta).max(it + 2).min(n - 1);
    let (yo, _) = outward(&g, y0, y1, it, grid.delta);
    let yi = inward(&g, it, ie, grid.delta);
    let scale = yo[it] / yi[0];
    let mut u = vec![0.0; n];
    for i in 0..=it { u[i] = yo[i] * grid.r[i].sqrt(); }
    for j in 1..yi.len() { u[it + j] = scale * yi[j] * grid.r[it + j].sqrt(); }
    let norm = grid.integrate(&u.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
    for x in &mut u { *x /= norm; }
    let umax = u.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    // r_peak feeds r_c = f·r_peak (Task 2). The 1e-3·umax threshold is a universal solver knob,
    // not rule-1 data (plan review NOTE): it only skips numerical wiggles in the far tail, and it
    // cannot move the outermost valence maximum, whose |u| is O(umax).
    let ip = (1..n - 1).rev()
        .find(|&i| u[i].abs() >= u[i - 1].abs() && u[i].abs() >= u[i + 1].abs() && u[i].abs() > 1e-3 * umax)
        .unwrap_or(0);
    if u[ip] < 0.0 { for x in &mut u { *x = -*x; } }
    let r_node = (1..=ip).rev().find(|&i| u[i - 1] * u[i] < 0.0).map(|i| {
        let t = u[i - 1] / (u[i - 1] - u[i]);
        grid.r[i - 1] + t * (grid.r[i] - grid.r[i - 1])
    });
    BoundState { e, u, r_peak: grid.r[ip], r_node }
}
```

- [ ] **Step 7: Implement the atom**

`crates/qm/src/radial/atom.rs` (library part; its module doc carries spec §4.1 in full: the potential, the derived occupations, the S9 scan and the Δ_scan rule, DIIS, the spin-restricted mode, the outputs, and the scaling-identity derivation copied from the spec):
```rust
use super::grid::LogGrid;
use super::ledger::LEDGER;
use super::numerov::{bound_state, BoundState};
use crate::xc::exc_point;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

pub type XcFn = fn(f64, f64) -> (f64, f64, f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinMode { Polarised, Restricted }

#[derive(Clone, Debug, PartialEq)]
pub struct ShellOcc { pub n: u8, pub l: u8, pub occ: [f64; 2] }

#[derive(Clone, Debug, PartialEq)]
pub enum Occupations { Aufbau { multiplicity: u32 }, Fixed(Vec<ShellOcc>) }

#[derive(Clone, Debug)]
pub struct AtomOptions { pub spin: SpinMode, pub occupations: Occupations, pub charge: i32, pub xc: XcFn }

impl AtomOptions {
    pub fn aufbau(spin: SpinMode, multiplicity: u32) -> AtomOptions {
        AtomOptions { spin, occupations: Occupations::Aufbau { multiplicity }, charge: 0, xc: exc_point }
    }
    pub fn fixed(spin: SpinMode, shells: Vec<ShellOcc>) -> AtomOptions {
        AtomOptions { spin, occupations: Occupations::Fixed(shells), charge: 0, xc: exc_point }
    }
}

#[derive(Clone, Debug)]
pub struct Orbital { pub n: u8, pub l: u8, pub eps: [f64; 2], pub occ: [f64; 2], pub u: [Vec<f64>; 2], pub r_peak: f64, pub r_node: Option<f64> }

#[derive(Clone, Copy, Debug)]
pub struct AtomEnergy { pub total: f64, pub kinetic: f64, pub nuclear: f64, pub hartree: f64, pub xc: f64, pub xc_potential: f64 }

#[derive(Clone, Debug)]
pub struct AtomResult {
    pub z: u8, pub spin: SpinMode, pub multiplicity: u32, pub orbitals: Vec<Orbital>,
    pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub v_h: Vec<f64>, pub energy: AtomEnergy,
    pub iterations: usize, pub grid: LogGrid,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RadialError {
    #[error("radial SCF for Z = {z} did not converge after {iterations} steps (Δρ = {delta:.2e})")]
    NotConverged { z: u8, iterations: usize, delta: f64 },
    #[error("Z = {z}: shell n = {n}, l = {l} is not bound in this potential")]
    Unbound { z: u8, n: u8, l: u8 },
    #[error("impossible occupation: {0}")]
    Occupation(String),
}

/// Ruling S9's rule (`sim::sampling::allowed_multiplicities`): the parity of N, from the lowest up
/// to `LEDGER.multiplicity_max`, while 2S ≤ N.
pub fn allowed_multiplicities(electrons: u32) -> Vec<u32> {
    (1 + electrons % 2..=LEDGER.multiplicity_max).step_by(2).filter(|&m| m - 1 <= electrons).collect()
}

/// V_H(r) = Q(r)/r + ∫_r^∞ 4πr′ρ dr′ (Gauss's law; both integrals cumulative, 4th order).
pub fn hartree_potential(grid: &LogGrid, rho: &[f64]) -> Vec<f64> {
    let q = grid.cumulative(&grid.r.iter().zip(rho).map(|(r, p)| 4.0 * PI * r * r * p).collect::<Vec<_>>());
    let s = grid.cumulative(&grid.r.iter().zip(rho).map(|(r, p)| 4.0 * PI * r * p).collect::<Vec<_>>());
    let s_tot = s[s.len() - 1];
    grid.r.iter().enumerate().map(|(i, r)| q[i] / r + (s_tot - s[i])).collect()
}

/// (2T_s + E_en + E_H + 3[Σ_σ∫ρ_σ v_xc,σ − E_xc]) / T_s — zero for a converged LSDA atom (spec §4.1).
pub fn scaling_identity_residual(a: &AtomResult) -> f64 {
    let e = &a.energy;
    (2.0 * e.kinetic + e.nuclear + e.hartree + 3.0 * (e.xc_potential - e.xc)) / e.kinetic
}

fn spin_counts(n: u32, m: u32) -> Result<(u32, u32), RadialError> {
    if m == 0 || m - 1 > n || (n - (m - 1)) % 2 != 0 {
        return Err(RadialError::Occupation(format!("multiplicity {m} impossible for {n} electrons")));
    }
    Ok(((n + m - 1) / 2, (n + 1 - m) / 2))
}

type Shell = (u8, u8, f64, BoundState);

/// One spin channel's integer aufbau over eigenvalues (ties to the lower l), spherically averaged:
/// the last shell's occupation is spread over its m. `cap(l)`: 2l+1 per spin channel, 2(2l+1) restricted.
fn aufbau_channel(grid: &LogGrid, v: &[f64], z: f64, electrons: f64, cap: impl Fn(u8) -> f64, prev: &[(u8, u8, f64)]) -> Result<Vec<Shell>, RadialError> {
    let guess = |n: u8, l: u8| prev.iter().find(|p| p.0 == n && p.1 == l).map(|p| p.2).filter(|e| e.is_finite());
    let mut next: Vec<Option<(u8, BoundState)>> =
        (0..=3u8).map(|l| bound_state(grid, v, l, 0, z, guess(l + 1, l)).map(|b| (l + 1, b))).collect();
    let mut out = Vec::new();
    let mut left = electrons;
    while left > 1e-12 {
        let (l, _) = next.iter().enumerate()
            .filter_map(|(l, s)| s.as_ref().map(|(_, b)| (l, b.e)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .ok_or_else(|| RadialError::Occupation(format!("{left} electrons left and no bound shell")))?;
        let (n, b) = next[l].take().expect("picked above");
        let occ = left.min(cap(l as u8));
        left -= occ;
        out.push((n, l as u8, occ, b));
        next[l] = bound_state(grid, v, l as u8, (n - l as u8) as usize, z, guess(n + 1, l as u8)).map(|b| (n + 1, b));
    }
    Ok(out)
}

fn density(grid: &LogGrid, shells: &[Shell]) -> Vec<f64> {
    (0..grid.len()).map(|i| {
        let r = grid.r[i];
        shells.iter().map(|s| s.2 * s.3.u[i] * s.3.u[i]).sum::<f64>() / (4.0 * PI * r * r)
    }).collect()
}

fn potentials(grid: &LogGrid, z: f64, rho: &[Vec<f64>; 2], xc: XcFn) -> [Vec<f64>; 2] {
    let n = grid.len();
    let total: Vec<f64> = (0..n).map(|i| rho[0][i] + rho[1][i]).collect();
    let v_h = hartree_potential(grid, &total);
    let mut v = [vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (_, vu, vd) = xc(rho[0][i], rho[1][i]);
        let c = -z / grid.r[i] + v_h[i];
        v[0][i] = c + vu;
        v[1][i] = c + vd;
    }
    v
}

fn merge(channels: &[Vec<Shell>; 2]) -> Vec<Orbital> {
    let mut keys: Vec<(u8, u8)> = channels.iter().flatten().map(|s| (s.0, s.1)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter().map(|(n, l)| {
        let get = |s: usize| channels[s].iter().find(|x| x.0 == n && x.1 == l);
        let (a, b) = (get(0), get(1));
        let peak_src = a.or(b).expect("present in one channel");
        Orbital {
            n, l,
            eps: [a.map_or(f64::NAN, |x| x.3.e), b.map_or(f64::NAN, |x| x.3.e)],
            occ: [a.map_or(0.0, |x| x.2), b.map_or(0.0, |x| x.2)],
            u: [a.map_or(Vec::new(), |x| x.3.u.clone()), b.map_or(Vec::new(), |x| x.3.u.clone())],
            r_peak: peak_src.3.r_peak, r_node: peak_src.3.r_node,
        }
    }).collect()
}

fn occupy(grid: &LogGrid, z: u8, v: &[Vec<f64>; 2], opts: &AtomOptions, prev: &[Orbital]) -> Result<(Vec<Orbital>, [Vec<f64>; 2]), RadialError> {
    let zf = z as f64;
    let prev_eps = |s: usize| prev.iter().map(|o| (o.n, o.l, o.eps[s])).collect::<Vec<_>>();
    let channels: [Vec<Shell>; 2] = match (&opts.occupations, opts.spin) {
        (Occupations::Aufbau { .. }, SpinMode::Restricted) => {
            let n_el = (z as i32 - opts.charge) as f64;
            let ch = aufbau_channel(grid, &v[0], zf, n_el, |l| 2.0 * (2 * l + 1) as f64, &prev_eps(0))?;
            let half: Vec<Shell> = ch.into_iter().map(|(n, l, o, b)| (n, l, 0.5 * o, b)).collect();
            [half.clone(), half]
        }
        (Occupations::Aufbau { multiplicity }, SpinMode::Polarised) => {
            let (up, dn) = spin_counts((z as i32 - opts.charge) as u32, *multiplicity)?;
            [aufbau_channel(grid, &v[0], zf, up as f64, |l| (2 * l + 1) as f64, &prev_eps(0))?,
             aufbau_channel(grid, &v[1], zf, dn as f64, |l| (2 * l + 1) as f64, &prev_eps(1))?]
        }
        (Occupations::Fixed(shells), spin) => {
            let mut ch: [Vec<Shell>; 2] = [Vec::new(), Vec::new()];
            for s in shells {
                for sp in 0..2 {
                    if s.occ[sp] <= 0.0 { continue; }
                    let src = if spin == SpinMode::Restricted { 0 } else { sp };
                    let g = prev.iter().find(|o| o.n == s.n && o.l == s.l).map(|o| o.eps[src]).filter(|e| e.is_finite());
                    let b = bound_state(grid, &v[src], s.l, (s.n - s.l - 1) as usize, zf, g)
                        .ok_or(RadialError::Unbound { z, n: s.n, l: s.l })?;
                    ch[sp].push((s.n, s.l, s.occ[sp], b));
                }
            }
            ch
        }
    };
    let rho = [density(grid, &channels[0]), density(grid, &channels[1])];
    Ok((merge(&channels), rho))
}

pub fn solve_atom(z: u8, opts: &AtomOptions) -> Result<AtomResult, RadialError> {
    let grid = LogGrid::universal();
    let n = grid.len();
    let zf = z as f64;
    let num = &LEDGER.numerics;
    let bare: Vec<f64> = grid.r.iter().map(|r| -zf / r).collect();
    let (mut orbitals, mut rho_in) = occupy(&grid, z, &[bare.clone(), bare], opts, &[])?;
    // the spec's Δρ measure ∫|Δρ| r² dr, and the DIIS inner product ∫ a b r² dr, on the log grid
    let w: Vec<f64> = grid.r.iter().map(|r| r * r * r * grid.delta).collect();
    let mut diis = Diis::new(num.diis_history);
    let mut delta = f64::INFINITY;
    for it in 1..=num.scf_max_iter {
        let v = potentials(&grid, zf, &rho_in, opts.xc);
        let (o, rho_out) = occupy(&grid, z, &v, opts, &orbitals)?;
        orbitals = o;
        delta = (0..2).map(|s| (0..n).map(|i| (rho_out[s][i] - rho_in[s][i]).abs() * w[i]).sum::<f64>()).sum();
        if delta < num.scf_tol {
            return Ok(finish(z, grid, opts, orbitals, rho_out, &v, it));
        }
        rho_in = diis.mix(rho_in, rho_out, &w, num.mixing);
    }
    Err(RadialError::NotConverged { z, iterations: num.scf_max_iter, delta })
}

/// The energy from the output density and orbitals (the `scf.rs` discipline): T_s = Σfε − Σ_σ∫ρ_σ V_σ^in.
fn finish(z: u8, grid: LogGrid, opts: &AtomOptions, orbitals: Vec<Orbital>, rho: [Vec<f64>; 2], v_in: &[Vec<f64>; 2], iterations: usize) -> AtomResult {
    let n = grid.len();
    let zf = z as f64;
    let shell = |f: &[f64]| grid.integrate(&f.iter().zip(&grid.r).map(|(a, r)| 4.0 * PI * r * r * a).collect::<Vec<_>>());
    let total: Vec<f64> = (0..n).map(|i| rho[0][i] + rho[1][i]).collect();
    let band: f64 = orbitals.iter().map(|o| (0..2).filter(|&s| o.occ[s] > 0.0).map(|s| o.occ[s] * o.eps[s]).sum::<f64>()).sum();
    let pot: f64 = (0..2).map(|s| shell(&(0..n).map(|i| rho[s][i] * v_in[s][i]).collect::<Vec<_>>())).sum();
    let kinetic = band - pot;
    let nuclear = -zf * grid.integrate(&(0..n).map(|i| 4.0 * PI * grid.r[i] * total[i]).collect::<Vec<_>>());
    let v_h = hartree_potential(&grid, &total);
    let hartree = 0.5 * shell(&(0..n).map(|i| total[i] * v_h[i]).collect::<Vec<_>>());
    let (mut exc, mut vr) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let (e, vu, vd) = (opts.xc)(rho[0][i], rho[1][i]);
        exc[i] = e;
        vr[i] = rho[0][i] * vu + rho[1][i] * vd;
    }
    let (xc, xc_potential) = (shell(&exc), shell(&vr));
    let up: f64 = orbitals.iter().map(|o| o.occ[0]).sum();
    let dn: f64 = orbitals.iter().map(|o| o.occ[1]).sum();
    let v = potentials(&grid, zf, &rho, opts.xc);
    AtomResult {
        z, spin: opts.spin, multiplicity: ((up - dn).abs().round() as u32) + 1, orbitals, rho, v, v_h,
        energy: AtomEnergy { total: kinetic + nuclear + hartree + xc, kinetic, nuclear, hartree, xc, xc_potential },
        iterations, grid,
    }
}

/// Pulay/DIIS on the stacked [ρ↑, ρ↓] (the 3-D `scf.rs` algorithm on 1-D vectors, inner product
/// ∫ a b r² dr): returns Σc_i(ρ_in,i + α r_i), c minimising ‖Σ c_i r_i‖ with Σ c_i = 1; B scaled by
/// its largest diagonal plus a 1e-12 relative regularisation; a failed solve drops the oldest entry.
struct Diis { m: usize, hist: Vec<(Vec<f64>, Vec<f64>)> }

impl Diis {
    fn new(m: usize) -> Diis { Diis { m: m.max(1), hist: Vec::new() } }
    fn mix(&mut self, rho_in: [Vec<f64>; 2], rho_out: [Vec<f64>; 2], w: &[f64], alpha: f64) -> [Vec<f64>; 2] {
        let n = w.len();
        let x: Vec<f64> = rho_in[0].iter().chain(rho_in[1].iter()).copied().collect();
        let y: Vec<f64> = rho_out[0].iter().chain(rho_out[1].iter()).copied().collect();
        let r: Vec<f64> = y.iter().zip(&x).map(|(a, b)| a - b).collect();
        self.hist.push((x, r));
        if self.hist.len() > self.m { self.hist.remove(0); }
        loop {
            let k = self.hist.len();
            let mut b = DMatrix::<f64>::zeros(k + 1, k + 1);
            for i in 0..k {
                for j in 0..k {
                    b[(i, j)] = (0..2 * n).map(|p| self.hist[i].1[p] * self.hist[j].1[p] * w[p % n]).sum();
                }
            }
            let scale = (0..k).map(|i| b[(i, i)]).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
            for i in 0..k {
                for j in 0..k { b[(i, j)] /= scale; }
                b[(i, i)] += 1e-12;
                b[(i, k)] = 1.0;
                b[(k, i)] = 1.0;
            }
            let mut rhs = DVector::<f64>::zeros(k + 1);
            rhs[k] = 1.0;
            match b.lu().solve(&rhs) {
                Some(c) if c.iter().all(|v| v.is_finite()) => {
                    let mut out = vec![0.0; 2 * n];
                    for i in 0..k {
                        for p in 0..2 * n { out[p] += c[i] * (self.hist[i].0[p] + alpha * self.hist[i].1[p]); }
                    }
                    for v in &mut out { *v = v.max(0.0); }
                    return [out[..n].to_vec(), out[n..].to_vec()];
                }
                _ if k > 1 => { self.hist.remove(0); }
                _ => unreachable!("one entry is linear mixing and always solves"),
            }
        }
    }
}

/// The lowest unoccupied state of angular momentum l in spin channel `spin` of a converged atom.
pub fn next_unoccupied(a: &AtomResult, l: u8, spin: usize) -> Option<(u8, BoundState)> {
    let filled = a.orbitals.iter().filter(|o| o.l == l && o.occ[spin] > 0.0).map(|o| o.n).max();
    let n = filled.map_or(l + 1, |n| n + 1);
    bound_state(&a.grid, &a.v[spin], l, (n - l - 1) as usize, a.z as f64, None).map(|b| (n, b))
}

pub fn ground_state(z: u8, spin: SpinMode) -> Result<AtomResult, RadialError> { ground_state_with(z, spin, exc_point) }

/// Restricted: the aufbau solve plus the Δ_scan scan. Polarised: that for every allowed multiplicity
/// (ruling S9); the lowest total energy wins.
pub fn ground_state_with(z: u8, spin: SpinMode, xc: XcFn) -> Result<AtomResult, RadialError> {
    let with = |m| AtomOptions { xc, ..AtomOptions::aufbau(spin, m) };
    match spin {
        SpinMode::Restricted => scanned(z, &with(1)),
        SpinMode::Polarised => {
            let (mut best, mut err): (Option<AtomResult>, Option<RadialError>) = (None, None);
            for m in allowed_multiplicities(z as u32) {
                match scanned(z, &with(m)) {
                    Ok(a) if best.as_ref().map_or(true, |b| a.energy.total < b.energy.total) => best = Some(a),
                    Ok(_) => {}
                    Err(e) => err = Some(e),
                }
            }
            best.ok_or_else(|| err.expect("at least one multiplicity was tried"))
        }
    }
}

/// Spec §4.1 "near-degenerate shells": from the aufbau solution, every pattern moving one electron
/// (in one spin channel; ½ + ½ when restricted) from the highest occupied shell to a partially filled
/// or empty shell of another l whose eigenvalue is within LEDGER.delta_scan of it is solved with
/// fixed occupations; the lowest total energy wins. No configuration table exists (rule 1).
fn scanned(z: u8, opts: &AtomOptions) -> Result<AtomResult, RadialError> {
    let base = solve_atom(z, opts)?;
    let mut best = base.clone();
    for alt in scan_alternatives(&base) {
        let o = AtomOptions { occupations: Occupations::Fixed(alt), ..opts.clone() };
        if let Ok(a) = solve_atom(z, &o) {
            if a.energy.total < best.energy.total { best = a; }
        }
    }
    Ok(best)
}

fn scan_alternatives(a: &AtomResult) -> Vec<Vec<ShellOcc>> {
    let restricted = a.spin == SpinMode::Restricted;
    let base: Vec<ShellOcc> = a.orbitals.iter().map(|o| ShellOcc { n: o.n, l: o.l, occ: o.occ }).collect();
    let mut out = Vec::new();
    for s in 0..(if restricted { 1 } else { 2 }) {
        let cap = |l: u8| (2 * l + 1) as f64 * if restricted { 2.0 } else { 1.0 };
        let total = |o: &ShellOcc| if restricted { o.occ[0] + o.occ[1] } else { o.occ[s] };
        let Some(homo) = a.orbitals.iter().filter(|o| o.occ[s] > 0.0).max_by(|x, y| x.eps[s].total_cmp(&y.eps[s])) else { continue };
        for l in (0..=3u8).filter(|&l| l != homo.l) {
            let partial = a.orbitals.iter().find(|o| o.l == l && total(&ShellOcc { n: o.n, l, occ: o.occ }) < cap(l) && o.occ[s] > 0.0);
            let (n, eps) = match partial {
                Some(o) => (o.n, o.eps[s]),
                None => match next_unoccupied(a, l, s) { Some((n, b)) => (n, b.e), None => continue },
            };
            if (eps - homo.eps[s]).abs() >= LEDGER.delta_scan { continue; }
            let mut alt = base.clone();
            let take = if restricted { [0.5, 0.5] } else if s == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
            if let Some(h) = alt.iter_mut().find(|x| x.n == homo.n && x.l == homo.l) {
                for k in 0..2 { h.occ[k] -= take[k]; }
            }
            match alt.iter_mut().find(|x| x.n == n && x.l == l) {
                Some(t) => for k in 0..2 { t.occ[k] += take[k]; },
                None => alt.push(ShellOcc { n, l, occ: take }),
            }
            alt.retain(|x| x.occ[0] + x.occ[1] > 0.0);
            out.push(alt);
        }
    }
    out
}
```
`crates/qm/src/radial/mod.rs`:
```rust
//! Self-generated pseudopotentials (M7), the generator: CPU, f64, no GPU. Spec:
//! docs/superpowers/specs/2026-10-14-pseudopotential-design.md §4–§5.1.
//! Pipeline: atom (§4.1) → partition (§4.2) → nlcc (§4.5) → tm + unscreening (§4.3) → kb + ghost
//! gates (§4.6) → transferability gates T1–T7 (§4.8) → recipe: the f ladder × local order (§4.4) →
//! record + recipe_hash (§4.9) → filter + h_req (§5.1, §4.8 T8). Every universal constant is a field
//! of `ledger::LEDGER`; nothing is indexed by Z (CLAUDE.md rule 1).
pub mod atom;
pub mod grid;
pub mod ledger;
pub mod numerov;

pub use atom::{allowed_multiplicities, ground_state, ground_state_with, hartree_potential, next_unoccupied,
               scaling_identity_residual, solve_atom, AtomEnergy, AtomOptions, AtomResult, Occupations, Orbital,
               RadialError, ShellOcc, SpinMode, XcFn};
pub use grid::LogGrid;
pub use ledger::{Ledger, LocalCandidate, NlccMode, LEDGER};
```
In `crates/qm/src/lib.rs` add `pub mod radial;`. In `crates/qm/src/grid.rs` make `neumaier` `pub(crate)`.

- [ ] **Step 8: Run the tests**

Run: `cargo test -p qm --release radial` and `cargo test -p sim --release radial_multiplicities_mirror_ruling_s9`
Expected: all PASS, with `report_pw92_against_nist` printing the PW92−NIST differences.
- If `hydrogenic_eigenvalues_are_exact` misses 1e-10·Z², or `hydrogenic_error_is_fourth_order_in_delta` falls outside [14, 18], do not change δ (a ledger value). Report the measured Δ per (Z, n, l) and the ratio. The bound already carries the derived Z² growth (plan review M1), so a miss is an integrator defect, not a tolerance question.
- If an NIST total misses its bound, first confirm the functional (Step 1), then report.

- [ ] **Step 9: Record the numbers.** In `references.md` "M7 radial atom (Task 1)", record:
  - the achieved totals and their Δ to NIST for H, He, Li, C, N, O and Ne;
  - the O E(M=1/3/5);
  - the worst scaling-identity residual;
  - the worst hydrogenic Δ/Z² and the δ-order ratio;
  - the NIST functional and its spin interpolation (Step 1);
  - the PW92−NIST-functional differences (rule 7: the functional-fit gap, stated).

- [ ] **Step 10: Commit**
```bash
git add crates/qm/src/radial crates/qm/src/lib.rs crates/qm/src/grid.rs crates/sim/src/sampling.rs crates/validate/references.md
git commit -o -m "feat(qm): radial all-electron LSDA atom on the log grid (M7 Task 1)" -- crates/qm/src/radial crates/qm/src/lib.rs crates/qm/src/grid.rs crates/sim/src/sampling.rs crates/validate/references.md
```
Before merging, hand the diff to `physics-reviewer` (Numerov transform, scaling identity, occupation rules, the NIST functional finding).

---

### Task 2: `partition` + `tm` — the core/valence rule, TM construction and unscreening **[Fable — physics-review gate]**

Spec §4.2, §4.3, §4.4 (r_c = f·r_max), §8 row 2; review M1, S5, and the §4.2-item-4 no-core branch.

**Files:**
- Create: `crates/qm/src/radial/partition.rs`, `crates/qm/src/radial/tm.rs`
- Modify: `crates/qm/src/radial/mod.rs` (add `pub mod partition; pub mod tm;` and re-exports)
- Test: unit tests in both new files

**Interfaces:**
- Consumes (Task 1): `ground_state`, `AtomResult { orbitals, v, grid, .. }`, `Orbital { n, l, eps, occ, u, r_peak, .. }`, `SpinMode::Restricted`, `bound_state`, `next_unoccupied`, `hartree_potential`, `LogGrid::{partial, derivative, nearest}`, `LEDGER.{delta_core, l_max_above_valence, f_ladder, v_ion_tail}`.
- Produces:
```rust
// partition.rs
pub struct Shell { pub n: u8, pub l: u8, pub eps: f64, pub occ: f64 }       // occ: both spins (restricted reference)
pub struct Channel { pub l: u8, pub n: u8, pub eps: f64, pub occ: f64, pub r_peak: f64, pub u: Vec<f64>, pub occupied: bool }
pub struct Partition { pub z: u8, pub core: Vec<Shell>, pub valence: Vec<Shell>, pub n_core: u32, pub z_v: f64,
                       pub semicore: bool, pub l_max: u8, pub channels: Vec<Channel> /* ascending l */, pub dropped: Vec<u8> }
pub enum Refusal { TwoStatesSameL { z: u8, l: u8, states: Vec<(u8, f64)> }, NoLadderEntryPasses { z: u8, tried: Vec<String> },
                   Radial { z: u8, message: String } }        // Task 11 adds DimerGateFailed (see Task 11)
pub fn partition(reference: &AtomResult) -> Result<Partition, Refusal>;
pub fn core_density(reference: &AtomResult, part: &Partition) -> Vec<f64>;
pub fn valence_density(reference: &AtomResult, part: &Partition) -> Vec<f64>;
pub fn has_core(z: u8) -> Result<bool, Refusal>;                                   // the CoresOnly test (Task 6)
// tm.rs
pub struct TmChannel { pub l: u8, pub eps: f64, pub r_c: f64, pub i_c: usize, pub c: [f64; 7] /* c0,c2,…,c12 */,
                       pub u: Vec<f64>, pub v_scr: Vec<f64> }
pub struct Unscreened { pub v_ion: Vec<Vec<f64>> /* per channel, same order */, pub rho_v: Vec<f64>, pub r_cmp: f64 }
pub fn tm_channel(grid: &LogGrid, l: u8, eps: f64, u_ae: &[f64], v_ae: &[f64], r_c: f64) -> Result<TmChannel, String>;
pub fn tm_channels(reference: &AtomResult, part: &Partition, f: f64) -> Result<Vec<TmChannel>, String>; // r_c rule + S5 cap
pub fn unscreen(grid: &LogGrid, z_v: f64, channels: &[TmChannel], occ: &[f64], rho_pc: &[f64]) -> Unscreened;
pub(crate) fn tm_targets(grid: &LogGrid, l: u8, eps: f64, u_ae: &[f64], v_ae: &[f64], ic: usize) -> [f64; 5];
pub(crate) fn poly_derivative(c: &[f64; 7], k: usize, r: f64) -> f64;
```
Rules fixed here and used by every later task:
- **The reference is the spin-restricted ground configuration** (spec §4.3). `partition` asserts `reference.spin == Restricted`.
- **r_c is snapped to the log grid** (`LogGrid::nearest`). Norm conservation is then exact on the same quadrature for AE and PP.
- **Channels are ascending in l.** The "highest generated channel" (`LocalCandidate::HighestChannel`, Task 3) is `channels.last()`.
- **Unoccupied l_max channel.** It is the lowest bound state of that l in the reference potential, `next_unoccupied(reference, l, 0)` (its node count follows from the core shells of that l). If unbound it is dropped and recorded in `dropped`. Its r_c is `min(f·r_peak, max over occupied channels of r_c)` (S5).
- **No-core branch** (spec §4.2 item 4, explicit): for H and He, `core` is empty, `n_core = 0` and `z_v = Z`, and `core_density` is identically 0.

- [ ] **Step 1: Write the failing tests**

`crates/qm/src/radial/tm.rs` tests:
```rust
use super::*;
use crate::radial::partition::{core_density, partition};
use crate::radial::{ground_state, SpinMode, LEDGER};

fn sq(u: &[f64]) -> Vec<f64> { u.iter().map(|x| x * x).collect() }

/// Spec §8 row 2 (review M1 and the NOTE on Task 2's first test): the seven TM conditions on O 2p.
#[test]
fn tm_oxygen_2p_satisfies_the_seven_conditions() {
    let a = ground_state(8, SpinMode::Restricted).expect("O converges");
    let part = partition(&a).expect("O is accepted by the partition");
    let ch = part.channels.iter().find(|c| c.l == 1).expect("a p channel");
    let f = LEDGER.f_ladder[0];
    let tm = tm_channel(&a.grid, 1, ch.eps, &ch.u, &a.v[0], f * ch.r_peak).expect("TM solves");
    let g = &a.grid;
    let (ic, rc) = (tm.i_c, g.r[tm.i_c]);
    // (1) norm conservation, ∫₀^{r_c}|u^PP|² = ∫₀^{r_c}|u^AE|², to 1e-10
    let (n_ae, n_pp) = (g.partial(&sq(&ch.u), ic), g.partial(&sq(&tm.u), ic));
    assert!((n_ae - n_pp).abs() < 1e-10, "norm: AE {n_ae:.12} PP {n_pp:.12}");
    // (2)–(6) p, p′, p″, p‴, p⁗ at r_c equal the AE values (TM eqs. 8–11), 1e-8 relative
    let target = tm_targets(g, 1, ch.eps, &ch.u, &a.v[0], ic);
    for k in 0..5 {
        let got = poly_derivative(&tm.c, k, rc);
        assert!((got - target[k]).abs() <= 1e-8 * target[k].abs().max(1.0), "p^({k})(r_c): {got:.12e} vs {:.12e}", target[k]);
    }
    // (2′)–(6′) the physical content of TM's C⁴ matching, independent of `tm_targets` (plan review S1:
    // the loop above compares the polynomial with the very right-hand side the 5×5 solve consumed,
    // so a sign error in the p‴ or p⁗ formula would pass it). From the coefficients alone, with
    // V = ε + (l+1)p′/r + ½(p′² + p″):
    //   V′ = (l+1)(p″/r − p′/r²) + p′p″ + ½p‴,
    //   V″ = (l+1)(p‴/r − 2p″/r² + 2p′/r³) + p″² + p′p‴ + ½p⁗,
    // compared with the AE potential's own grid derivatives; and u^PP′(r_c⁻) = u^PP·((l+1)/r + p′)
    // with the AE u′.
    let lp = 2.0; // l + 1 for the p channel
    let p = |k: usize| poly_derivative(&tm.c, k, rc);
    let v_pp = ch.eps + lp * p(1) / rc + 0.5 * (p(1) * p(1) + p(2));
    let dv_pp = lp * (p(2) / rc - p(1) / (rc * rc)) + p(1) * p(2) + 0.5 * p(3);
    let d2v_pp = lp * (p(3) / rc - 2.0 * p(2) / (rc * rc) + 2.0 * p(1) / rc.powi(3)) + p(2) * p(2) + p(1) * p(3) + 0.5 * p(4);
    let dv_ae = g.derivative(&a.v[0]);
    let d2v_ae = g.derivative(&dv_ae);
    assert!((v_pp - a.v[0][ic]).abs() < 1e-8, "V_scr(r_c⁻) {v_pp:.12} vs V_AE(r_c) {:.12}", a.v[0][ic]);
    assert!((dv_pp - dv_ae[ic]).abs() <= 1e-6 * dv_ae[ic].abs(), "V′: PP {dv_pp:.10e} vs AE {:.10e}", dv_ae[ic]);
    assert!((d2v_pp - d2v_ae[ic]).abs() <= 1e-6 * d2v_ae[ic].abs(), "V″: PP {d2v_pp:.10e} vs AE {:.10e}", d2v_ae[ic]);
    let du_pp = rc.powf(lp) * p(0).exp() * (lp / rc + p(1));
    let du_ae = g.derivative(&ch.u)[ic];
    assert!((du_pp - du_ae).abs() <= 1e-8 * du_ae.abs(), "u′(r_c): PP {du_pp:.12e} vs AE {du_ae:.12e}");
    // (7) V_scr″(0) = 0 ⇔ c₂² + (2l+5)c₄ = 0, with V_scr(0) = ε + (2l+3)c₂ (spec §4.3, review M1)
    let (c2, c4) = (tm.c[1], tm.c[2]);
    assert!((c2 * c2 + 7.0 * c4).abs() <= 1e-12 * (c2 * c2).max(1e-300), "c2² + 7c4 = {:.3e}", c2 * c2 + 7.0 * c4);
    let v0 = ch.eps + 5.0 * c2;
    assert!((tm.v_scr[0] - v0).abs() < 1e-8, "V_scr(r0) = {:.12} vs ε + 5c₂ = {v0:.12}", tm.v_scr[0]);
    let i = g.nearest(1e-3);
    assert!(((tm.v_scr[i] - v0) / (g.r[i] * g.r[i])).abs() < 1e-4, "the r² coefficient of V_scr does not vanish");
    // Unscreening (spec §4.3): "each channel's ionic potential equals −Z_v/r only beyond
    // max(r_c, r_nlc) and the core tail". At Task 2 ρ_pc = ρ_c (the r_nlc → 0 limit; Task 4 re-runs
    // this with the LFC partial core); the module doc's derivation then makes V_ion + Z_v/r beyond
    // max r_c exactly the core's Hartree tail V_H[ρ_c] − N_c/r (the 1s density beyond r). Asserted to
    // v_ion_tail/100 = 1e-10 — tight enough that r_cmp is fixed by the core tail alone — then r_cmp is
    // verified: beyond it every channel is −Z_v/r to v_ion_tail, it is minimal, and it is the radius
    // where the core tail alone drops below v_ion_tail. Every element and f:
    // `v_ion_tail_is_the_core_hartree_tail_across_the_ladder`.
    let chans = tm_channels(&a, &part, f).expect("every channel solves");
    let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
    let rho_c = core_density(&a, &part);
    let un = unscreen(g, part.z_v, &chans, &occ, &rho_c);
    let vhc = hartree_potential(g, &rho_c);
    let core_tail = |i: usize| vhc[i] - part.n_core as f64 / g.r[i];
    let n = g.len();
    let i_from = chans.iter().map(|c| c.i_c).max().unwrap() + 1;
    for (k, v) in un.v_ion.iter().enumerate() {
        let worst = (i_from..n).map(|i| (v[i] + part.z_v / g.r[i] - core_tail(i)).abs()).fold(0.0, f64::max);
        eprintln!("channel l = {}: r_c {:.4}; max |V_ion + Z_v/r − (V_H[ρ_c] − N_c/r)| beyond max r_c ({:.4}) = {worst:.2e}; V_ion(r0) {:.6}",
                  chans[k].l, chans[k].r_c, g.r[i_from], v[0]);
        assert!(worst < LEDGER.v_ion_tail / 100.0, "channel {k}: V_ion + Z_v/r is not the core tail ({worst:.2e})");
    }
    for i in (i_from..n).step_by(6).take(8) {
        eprintln!("  r {:.4}: V_ion,s + Z_v/r {:+.4e}, core tail {:+.4e}", g.r[i], un.v_ion[0][i] + part.z_v / g.r[i], core_tail(i));
    }
    let i_cmp = g.nearest(un.r_cmp);
    assert!((i_cmp..n).all(|i| un.v_ion.iter().all(|v| (v[i] + part.z_v / g.r[i]).abs() < LEDGER.v_ion_tail)), "r_cmp does not bound the tail");
    assert!(i_cmp > 0 && un.v_ion.iter().any(|v| (v[i_cmp - 1] + part.z_v / g.r[i_cmp - 1]).abs() >= LEDGER.v_ion_tail), "r_cmp is not minimal");
    let i_core = (0..n).rev().find(|&i| core_tail(i).abs() >= LEDGER.v_ion_tail).unwrap() + 1;
    eprintln!("r_cmp = {:.4}; the core tail alone falls below v_ion_tail at {:.4}", un.r_cmp, g.r[i_core]);
    assert!(i_core > i_from && i_cmp == i_core, "r_cmp {} vs the core tail's radius {}", un.r_cmp, g.r[i_core]);
}
```
`crates/qm/src/radial/partition.rs` tests:
```rust
use super::*;
use crate::radial::{ground_state, SpinMode};

fn part(z: u8) -> Result<Partition, Refusal> { partition(&ground_state(z, SpinMode::Restricted).expect("converges")) }

/// Spec §8 row 2: Li's 1s (−1.9 Ha) is within Δ_core of 2s → two valence s states → refused, the
/// message naming the ONCV stage (spec §4.2 item 4, §4.3).
#[test]
fn lithium_is_refused_two_s_states() {
    match part(3) {
        Err(r @ Refusal::TwoStatesSameL { z: 3, l: 0, .. }) => {
            let m = r.to_string();
            assert!(m.contains("two valence s states") && m.contains("ONCV"), "{m}");
        }
        other => panic!("Li must be refused with two s states, got {other:?}"),
    }
}

/// Be–Ne: the 1s is the core, Z_v = Z − 2, channels s and p only (the 3d is unbound in the neutral
/// LSDA atom — spec §4.2 item 3); derived, nothing tabulated.
#[test]
fn first_row_cores_are_the_1s() {
    for z in 4u8..=10 {
        let p = part(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
        assert_eq!(p.core.iter().map(|s| (s.n, s.l)).collect::<Vec<_>>(), vec![(1, 0)], "Z = {z}");
        assert_eq!(p.z_v, z as f64 - 2.0);
        assert_eq!(p.channels.iter().map(|c| c.l).collect::<Vec<_>>(), vec![0, 1], "Z = {z}");
        assert!(!p.semicore);
    }
}

/// The explicit no-core branch (spec §4.2 item 4): H and He have no core.
#[test]
fn hydrogen_and_helium_have_no_core() {
    for z in [1u8, 2] {
        let a = ground_state(z, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        assert!(p.core.is_empty() && p.n_core == 0 && p.z_v == z as f64, "Z = {z}: {p:?}");
        assert!(core_density(&a, &p).iter().all(|&x| x == 0.0));
        assert!(p.channels.iter().any(|c| c.l == 0 && c.occupied));
        assert!(!has_core(z).unwrap());
    }
}

/// Review S5: an unoccupied l_max channel's r_c is capped at the largest occupied r_c (Al–Ar 3d).
#[test]
fn unoccupied_channel_radius_is_capped() {
    for z in 13u8..=18 {
        let a = ground_state(z, SpinMode::Restricted).unwrap();
        let Ok(p) = partition(&a) else { continue };
        let chans = crate::radial::tm::tm_channels(&a, &p, crate::radial::LEDGER.f_ladder[0]);
        let Ok(chans) = chans else { continue };
        let occ_max = p.channels.iter().zip(&chans).filter(|(c, _)| c.occupied).map(|(_, t)| t.r_c).fold(0.0, f64::max);
        for (c, t) in p.channels.iter().zip(&chans).filter(|(c, _)| !c.occupied) {
            assert!(t.r_c <= occ_max + 1e-12, "Z = {z}, l = {}: r_c {} > cap {occ_max}", c.l, t.r_c);
        }
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release radial::tm radial::partition`
Expected: compile errors (modules missing).

- [ ] **Step 3: Implement the partition**

`crates/qm/src/radial/partition.rs` (the module doc carries spec §4.2 items 1–4, and the S9 scope note for Z > 18 verbatim):
```rust
use super::atom::{ground_state, next_unoccupied, AtomResult, SpinMode};
use super::ledger::LEDGER;
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct Shell { pub n: u8, pub l: u8, pub eps: f64, pub occ: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct Channel { pub l: u8, pub n: u8, pub eps: f64, pub occ: f64, pub r_peak: f64, pub u: Vec<f64>, pub occupied: bool }

#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    pub z: u8, pub core: Vec<Shell>, pub valence: Vec<Shell>, pub n_core: u32, pub z_v: f64,
    pub semicore: bool, pub l_max: u8, pub channels: Vec<Channel>, pub dropped: Vec<u8>,
}

fn l_name(l: u8) -> char { ['s', 'p', 'd', 'f'][l as usize] }
fn fmt_states(z: &[(u8, f64)], l: u8) -> String {
    z.iter().map(|(n, e)| format!("{n}{} at {e:.3} Ha", l_name(l))).collect::<Vec<_>>().join(", ")
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Refusal {
    #[error("Z = {z}: two valence {} states ({}) — a one-projector TM/KB potential cannot carry both; refused until the ONCV stage (spec §4.3)", l_name(*l), fmt_states(states, *l))]
    TwoStatesSameL { z: u8, l: u8, states: Vec<(u8, f64)> },
    #[error("Z = {z}: no entry of the f ladder passes the gates at this recipe version: {}", tried.join("; "))]
    NoLadderEntryPasses { z: u8, tried: Vec<String> },
    #[error("Z = {z}: the radial reference atom failed: {message}")]
    Radial { z: u8, message: String },
}

pub fn partition(reference: &AtomResult) -> Result<Partition, Refusal> {
    assert_eq!(reference.spin, SpinMode::Restricted, "the partition is taken on the spin-restricted reference (spec §4.3)");
    let z = reference.z;
    let occupied: Vec<Shell> = reference.orbitals.iter().filter(|o| o.occ[0] + o.occ[1] > 0.0)
        .map(|o| Shell { n: o.n, l: o.l, eps: o.eps[0], occ: o.occ[0] + o.occ[1] }).collect();
    let n_max = occupied.iter().map(|s| s.n).max().expect("an atom has electrons");
    // 1. seed: every occupied shell with the highest n
    let mut valence: Vec<Shell> = occupied.iter().filter(|s| s.n == n_max).cloned().collect();
    let eps_v_min = valence.iter().map(|s| s.eps).fold(f64::INFINITY, f64::min);
    // 2. promote whole lower shells while any of their subshells lies within Δ_core of the seed
    let mut semicore = false;
    for n in (1..n_max).rev() {
        let shell: Vec<Shell> = occupied.iter().filter(|s| s.n == n).cloned().collect();
        if shell.iter().any(|s| s.eps > eps_v_min - LEDGER.delta_core) {
            valence.extend(shell);
            semicore = true;
        } else {
            break;
        }
    }
    let core: Vec<Shell> = occupied.iter().filter(|s| !valence.iter().any(|v| v.n == s.n && v.l == s.l)).cloned().collect();
    let n_core = core.iter().map(|s| s.occ).sum::<f64>().round() as u32;
    let z_v = valence.iter().map(|s| s.occ).sum::<f64>();
    // 3. channels l = 0..=l_max
    let l_max = valence.iter().map(|s| s.l).max().unwrap() + LEDGER.l_max_above_valence;
    let (mut channels, mut dropped) = (Vec::new(), Vec::new());
    for l in 0..=l_max {
        let of_l: Vec<&Shell> = valence.iter().filter(|s| s.l == l).collect();
        match of_l.len() {
            0 => match next_unoccupied(reference, l, 0) {
                Some((n, b)) => channels.push(Channel { l, n, eps: b.e, occ: 0.0, r_peak: b.r_peak, u: b.u, occupied: false }),
                None => dropped.push(l),
            },
            1 => {
                let s = of_l[0];
                let o = reference.orbitals.iter().find(|o| o.n == s.n && o.l == l).expect("listed");
                channels.push(Channel { l, n: s.n, eps: s.eps, occ: s.occ, r_peak: o.r_peak, u: o.u[0].clone(), occupied: true });
            }
            _ => return Err(Refusal::TwoStatesSameL { z, l, states: of_l.iter().map(|s| (s.n, s.eps)).collect() }),
        }
    }
    Ok(Partition { z, core, valence, n_core, z_v, semicore, l_max, channels, dropped })
}

fn shell_density(reference: &AtomResult, shells: &[Shell]) -> Vec<f64> {
    let g = &reference.grid;
    (0..g.len()).map(|i| {
        let r = g.r[i];
        shells.iter().map(|s| {
            let o = reference.orbitals.iter().find(|o| o.n == s.n && o.l == s.l).expect("listed");
            s.occ * o.u[0][i] * o.u[0][i]
        }).sum::<f64>() / (4.0 * PI * r * r)
    }).collect()
}

/// ρ_c(r) (e/Bohr³); identically 0 without a core (the explicit no-core branch).
pub fn core_density(reference: &AtomResult, part: &Partition) -> Vec<f64> { shell_density(reference, &part.core) }
pub fn valence_density(reference: &AtomResult, part: &Partition) -> Vec<f64> { shell_density(reference, &part.valence) }

/// Whether §4.2 gives Z a core (`IonicPolicy::CoresOnly`, spec §4.7). A refusal propagates
/// (the caller falls back to all-electron under `CoresOnly`).
pub fn has_core(z: u8) -> Result<bool, Refusal> {
    let a = ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?;
    Ok(partition(&a)?.n_core > 0)
}
```

- [ ] **Step 4: Implement TM and unscreening**

`crates/qm/src/radial/tm.rs` (the module doc carries the TM construction of spec §4.3, the M1 derivation of V_scr(r) = ε + (2l+3)c₂ + 2[c₂² + (2l+5)c₄]r² + O(r⁴), the TM eqs. 8–11 derivation of p‴ and p⁗ by differentiating V − ε = (l+1)p′/r + ½(p′² + p″), and unscreening):
```rust
use super::atom::{hartree_potential, AtomResult};
use super::grid::LogGrid;
use super::ledger::LEDGER;
use super::partition::Partition;
use crate::xc::exc_point;
use nalgebra::{Matrix5, Vector5};

#[derive(Clone, Debug, PartialEq)]
pub struct TmChannel { pub l: u8, pub eps: f64, pub r_c: f64, pub i_c: usize, pub c: [f64; 7], pub u: Vec<f64>, pub v_scr: Vec<f64> }

#[derive(Clone, Debug, PartialEq)]
pub struct Unscreened { pub v_ion: Vec<Vec<f64>>, pub rho_v: Vec<f64>, pub r_cmp: f64 }

/// p = ln(u/r^{l+1}) and its first four derivatives at r_{ic} from the AE solution (TM eqs. 8–11):
/// p′ = u′/u − (l+1)/r; p″ = 2(V − ε) − 2(l+1)p′/r − p′²;
/// p‴ = 2V′ + 2(l+1)p′/r² − 2(l+1)p″/r − 2p′p″;
/// p⁗ = 2V″ − 4(l+1)p′/r³ + 4(l+1)p″/r² − 2(l+1)p‴/r − 2p″² − 2p′p‴.
pub(crate) fn tm_targets(grid: &LogGrid, l: u8, eps: f64, u_ae: &[f64], v_ae: &[f64], ic: usize) -> [f64; 5] {
    let lp = l as f64 + 1.0;
    let rc = grid.r[ic];
    let du = grid.derivative(u_ae);
    let dv = grid.derivative(v_ae);
    let d2v = grid.derivative(&dv);
    let p0 = (u_ae[ic] / rc.powf(lp)).ln();
    let p1 = du[ic] / u_ae[ic] - lp / rc;
    let p2 = 2.0 * (v_ae[ic] - eps) - 2.0 * lp * p1 / rc - p1 * p1;
    let p3 = 2.0 * dv[ic] + 2.0 * lp * p1 / (rc * rc) - 2.0 * lp * p2 / rc - 2.0 * p1 * p2;
    let p4 = 2.0 * d2v[ic] - 4.0 * lp * p1 / rc.powi(3) + 4.0 * lp * p2 / (rc * rc) - 2.0 * lp * p3 / rc - 2.0 * p2 * p2 - 2.0 * p1 * p3;
    [p0, p1, p2, p3, p4]
}

/// d^k/dr^k of p(r) = Σ_j c_j r^{2j}, j = 0..6.
pub(crate) fn poly_derivative(c: &[f64; 7], k: usize, r: f64) -> f64 {
    (0..7).map(|j| {
        let m = 2 * j as i32;
        let mut f = 1.0;
        for q in 0..k as i32 { f *= (m - q) as f64; }
        if f == 0.0 { 0.0 } else { c[j] * f * r.powi(m - k as i32) }
    }).sum()
}

pub fn tm_channel(grid: &LogGrid, l: u8, eps: f64, u_ae: &[f64], v_ae: &[f64], r_c: f64) -> Result<TmChannel, String> {
    let ic = grid.nearest(r_c);
    let rc = grid.r[ic];
    if u_ae[ic] <= 0.0 { return Err(format!("l = {l}: u_AE(r_c = {rc:.3}) ≤ 0 — r_c inside the outermost node")); }
    let lf = l as f64;
    let target = tm_targets(grid, l, eps, u_ae, v_ae, ic);
    let sq: Vec<f64> = u_ae.iter().map(|u| u * u).collect();
    let n_ae = grid.partial(&sq, ic);
    let dpow = |m: i32, k: usize| { let mut f = 1.0; for q in 0..k as i32 { f *= (m - q) as f64; } if f == 0.0 { 0.0 } else { f * rc.powi(m - k as i32) } };
    // for a given c2: c4 = −c2²/(2l+5) (V_scr″(0) = 0), the 5×5 linear system for c0, c6, c8, c10, c12
    let coeffs = |c2: f64| -> Option<[f64; 7]> {
        let c4 = -c2 * c2 / (2.0 * lf + 5.0);
        let ms = [0, 6, 8, 10, 12];
        let a = Matrix5::from_fn(|k, j| dpow(ms[j], k));
        let b = Vector5::from_fn(|k, _| target[k] - c2 * dpow(2, k) - c4 * dpow(4, k));
        let x = a.lu().solve(&b)?;
        Some([x[0], c2, c4, x[1], x[2], x[3], x[4]])
    };
    let u_of = |c: &[f64; 7], r: f64| r.powf(lf + 1.0) * poly_derivative(c, 0, r).exp();
    let resid = |c2: f64| -> Option<f64> {
        let c = coeffs(c2)?;
        let f: Vec<f64> = (0..grid.len()).map(|i| if i <= ic { u_of(&c, grid.r[i]).powi(2) } else { 0.0 }).collect();
        Some(grid.partial(&f, ic).ln() - n_ae.ln())
    };
    // Newton on c2 from 0 (TM), central-difference derivative, steps clamped to ±1
    let mut c2 = 0.0;
    let mut f = resid(c2).ok_or("TM: singular matching system")?;
    for _ in 0..100 {
        if f.abs() < 1e-14 { break; }
        let h = 1e-6 * c2.abs().max(1.0);
        let df = (resid(c2 + h).ok_or("TM: singular")? - resid(c2 - h).ok_or("TM: singular")?) / (2.0 * h);
        if !df.is_finite() || df == 0.0 { return Err(format!("l = {l}: TM norm equation has a flat derivative at c2 = {c2}")); }
        c2 += (-f / df).clamp(-1.0, 1.0);
        f = resid(c2).ok_or("TM: singular")?;
    }
    if !(f.abs() < 1e-12) { return Err(format!("l = {l}, r_c = {rc:.3}: norm conservation has no TM solution (residual {f:.2e})")); }
    let c = coeffs(c2).expect("solved above");
    let n = grid.len();
    let mut u = u_ae.to_vec();
    let mut v_scr = v_ae.to_vec();
    for i in 0..=ic {
        let r = grid.r[i];
        u[i] = u_of(&c, r);
        // V = ε + (l+1)p′/r + ½(p′² + p″), with p′/r expanded (no division at small r)
        let r2 = r * r;
        let p1_over_r = 2.0 * c[1] + r2 * (4.0 * c[2] + r2 * (6.0 * c[3] + r2 * (8.0 * c[4] + r2 * (10.0 * c[5] + r2 * 12.0 * c[6]))));
        let p1 = r * p1_over_r;
        let p2 = poly_derivative(&c, 2, r);
        v_scr[i] = eps + (lf + 1.0) * p1_over_r + 0.5 * (p1 * p1 + p2);
    }
    debug_assert_eq!(u.len(), n);
    Ok(TmChannel { l, eps, r_c: rc, i_c: ic, c, u, v_scr })
}

/// Spec §4.4 r_c rule: r_c,l = f·r_max,l (occupied channels); unoccupied: min(f·r_max,l, max occupied r_c) (S5).
pub fn tm_channels(reference: &AtomResult, part: &Partition, f: f64) -> Result<Vec<TmChannel>, String> {
    let g = &reference.grid;
    let snap = |r: f64| g.r[g.nearest(r)];
    let cap = part.channels.iter().filter(|c| c.occupied).map(|c| snap(f * c.r_peak)).fold(0.0, f64::max);
    part.channels.iter().map(|c| {
        let r_c = if c.occupied { f * c.r_peak } else { (f * c.r_peak).min(cap) };
        tm_channel(g, c.l, c.eps, &c.u, &reference.v[0], r_c)
    }).collect()
}

/// V_l^ion = V_l^scr − V_H[ρ_v^PP] − v_xc[ρ_v^PP + ρ_pc] (spin-restricted: each spin ½(ρ_v + ρ_pc));
/// r_cmp = the smallest grid r from which |V_ion + Z_v/r| < LEDGER.v_ion_tail for every channel.
pub fn unscreen(grid: &LogGrid, z_v: f64, channels: &[TmChannel], occ: &[f64], rho_pc: &[f64]) -> Unscreened {
    let n = grid.len();
    let rho_v: Vec<f64> = (0..n).map(|i| {
        let r = grid.r[i];
        channels.iter().zip(occ).map(|(c, f)| f * c.u[i] * c.u[i]).sum::<f64>() / (4.0 * std::f64::consts::PI * r * r)
    }).collect();
    let v_h = hartree_potential(grid, &rho_v);
    let vxc: Vec<f64> = (0..n).map(|i| { let half = 0.5 * (rho_v[i] + rho_pc[i]); exc_point(half, half).1 }).collect();
    let v_ion: Vec<Vec<f64>> = channels.iter().map(|c| (0..n).map(|i| c.v_scr[i] - v_h[i] - vxc[i]).collect()).collect();
    let bad = (0..n).rev().find(|&i| v_ion.iter().any(|v| (v[i] + z_v / grid.r[i]).abs() >= LEDGER.v_ion_tail));
    let r_cmp = bad.map_or(grid.r[0], |i| grid.r[(i + 1).min(n - 1)]);
    Unscreened { v_ion, rho_v, r_cmp }
}
```
Add `pub mod partition; pub mod tm;` to `radial/mod.rs` and re-export `partition::{core_density, has_core, partition, valence_density, Channel, Partition, Refusal, Shell}` and `tm::{tm_channel, tm_channels, unscreen, TmChannel, Unscreened}`.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p qm --release radial`
Expected: all PASS (Task 1's tests too). Beyond max r_c, V_ion + Z_v/r is the core's Hartree tail (spec §4.3), not zero; r_cmp is where that tail falls below `v_ion_tail` (O: 1.5266 at every f).

- [ ] **Step 6: Commit**
```bash
git add crates/qm/src/radial
git commit -o -m "feat(qm): core/valence partition, Troullier–Martins construction and unscreening (M7 Task 2)" -- crates/qm/src/radial
```
Before merging, hand the diff to `physics-reviewer`, citing M1 (the c₂² + (2l+5)c₄ condition), TM eqs. 8–11, the independent V/V′/V″/u′ matching checks (plan review S1), the S5 cap and the no-core branch.

---

### Task 3: `kb` + `recipe` — the KB form, the radial KB atom, the ghost gates and the f ladder **[Fable — physics-review gate]**

Spec §4.4 (local channel, f ladder), §4.6 (KB, G1, G2), §8 row 3; review NOTE (the GSS criterion re-derived, the integro-differential KB atom). **Spec questions 4 and 5 apply** (the second first-failing test, and G2's count). The plan review accepted Q4 with two test amendments (a narrow well with a scanned depth; G2 checked unconditionally on the reference screening) and amended Q5: G2's "no extra state" is the exact inertia count n_KB(E) = n_loc(E) + [s(E) < 0] − [E_KB > 0], with the mesh kept only to locate roots. Task 3 review: the expected count is the AE atom's own spectrum above the core (spec §4.6, M1), and only an extra state is refused.

**Files:**
- Create: `crates/qm/src/radial/kb.rs`, `crates/qm/src/radial/recipe.rs`
- Modify: `crates/qm/src/radial/atom.rs` (`Diis` and its `new`/`mix` become `pub(crate)`), `crates/qm/src/radial/mod.rs`, `crates/qm/src/radial/partition.rs` (`Partition::spectrum`)
- Test: unit tests in both new files

**Interfaces:**
- Consumes: Task 1 (`LogGrid`, `bound_state`, `g_coeff`, `regular_start`, `outward`, `outward_sourced`, `inward`, `shooting_end`, `hartree_potential`, `AtomResult`, `SpinMode`, `RadialError`, `LEDGER`) and Task 2 (`Partition`, `Channel`, `TmChannel`, `Unscreened`, `tm_channels`, `unscreen`, `partition`, `core_density`, `Refusal`).
- Produces:
```rust
// kb.rs
pub struct Projector { pub l: u8, pub e_kb: f64, pub gss: f64, pub beta: Vec<f64> /* u-form: δV_l·u_l^PP */, pub dv: Vec<f64>, pub u: Vec<f64>, pub i_beta: usize /* β, δV ≡ 0 from here: r_β */ }
impl Projector { pub fn support(&self, grid: &LogGrid) -> f64; } // r_β (spec §4.6)
pub struct KbPotential { pub z: u8, pub z_v: f64, pub local: LocalCandidate, pub local_l: Option<u8>, pub v_loc: Vec<f64>,
                         pub projectors: Vec<Projector>, pub rho_pc: Vec<f64>, pub r_loc: f64, pub r_cmp: f64 }
pub struct KbInput { pub z: u8, pub z_v: f64, pub v_loc: Vec<f64>, pub projectors: Vec<(u8, f64, Vec<f64>)> /* (l, E_KB, β u-form) */, pub rho_pc: Vec<f64> }
pub struct ValenceConfig { pub occ: Vec<[f64; 2]> /* index = l; (↑, ↓) */ }
pub struct KbAtom { pub eps: Vec<[f64; 2]>, pub u: Vec<[Vec<f64>; 2]>, pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub energy: f64 }
pub enum GssVerdict { GhostFree, Ghost { e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64> } }
pub struct GhostReport { pub g1: Vec<(u8, GssVerdict)>, pub g2_max_eig_error: f64, pub g2_roots: Vec<(u8, Vec<f64>)>,
                         pub g2_missing: Vec<(u8, Vec<f64>, Vec<f64>)> /* (l, AE energies, KB roots) where n_KB < n_AE */ }
// partition.rs: Partition gains `pub spectrum: Vec<Vec<f64>>` — per l ≤ l_max, the reference's bound eigenvalues above that l's core shells
pub fn core_tail(grid: &LogGrid, rho_c: &[f64], n_core: f64) -> Vec<f64>;   // τ_c = V_H[ρ_c] − N_c/r (full core)
pub fn polynomial_local(grid: &LogGrid, z_v: f64, r_loc: f64, tau_c: &[f64]) -> Vec<f64>;   // Task 2 review S3
pub fn kb_form(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, local: LocalCandidate, rho_pc: Vec<f64>, tau_c: &[f64]) -> KbPotential;
pub fn kb_form_with_local(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, v_loc: Vec<f64>, local_l: Option<u8>, rho_pc: Vec<f64>) -> KbPotential;
impl KbPotential { pub fn input(&self) -> KbInput; }
pub fn kb_lower_bound(grid: &LogGrid, v: &[f64], l: u8, proj: Option<(f64, &[f64])>) -> f64;
pub fn kb_eigenvalues(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e_lo: f64, e_hi: f64, mesh: f64) -> Vec<f64>;
pub fn kb_count_below(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> usize;   // exact (Q5)
pub fn kb_state(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> Vec<f64>;
pub fn kb_regular(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64, i_end: usize) -> Vec<f64>; // T5 (Task 4); no division (Q12)
pub fn solve_kb_atom(grid: &LogGrid, input: &KbInput, config: &ValenceConfig, spin: SpinMode) -> Result<KbAtom, RadialError>;
pub fn reference_config(part: &Partition) -> ValenceConfig;
pub fn gss_verdict(e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64>) -> GssVerdict;
pub fn ghost_gates(grid: &LogGrid, kb: &KbPotential, screen: &[f64], part: &Partition) -> Result<GhostReport, String>;
// recipe.rs
pub struct PseudoTable { pub z: u8, pub z_v: f64, pub n_core: u32, pub f: f64, pub local: LocalCandidate, pub kb: KbPotential,
                         pub channels: Vec<TmChannel>, pub screen: Vec<f64>, pub partition: Partition,
                         pub reference: std::sync::Arc<AtomResult>, pub grid: LogGrid, pub ghost: GhostReport }
                         // Task 4 adds: pub partial_core: PartialCore, pub gates: GateReport, pub record: GenerationRecord
pub fn generate(z: u8) -> Result<PseudoTable, Refusal>;
pub(crate) fn candidate(reference: &std::sync::Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, rho_pc: &[f64]) -> Result<PseudoTable, String>;
```
**The KB normalisation** (spec §4.6, review S1) is the radial one: E_KB = 1/∫u δV u dr with ∫u² dr = 1. GSS = ∫β² dr / ∫u β dr with β = δV·u. V_NL u = E_KB β ⟨β|u⟩. Task 6's grid function carries the √((2l+1)/4π) that makes this the same operator.

- [ ] **Step 1: Write the failing tests**

`crates/qm/src/radial/kb.rs` tests:
```rust
use super::*;
use crate::radial::recipe::generate;
use crate::radial::{bound_state, SpinMode, LEDGER};

/// Spec §8 row 3, first failing test: the radial KB atom (integro-differential, the 2×2 determinant)
/// reproduces the semilocal TM atom's valence eigenvalues — the AE reference ones — to < 1 mHa.
#[test]
fn kb_atom_reproduces_semilocal_eigenvalues() {
    for z in [6u8, 7, 8, 9, 10] {
        let t = generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
        let atom = solve_kb_atom(&t.grid, &t.kb.input(), &reference_config(&t.partition), SpinMode::Restricted).expect("KB atom converges");
        for c in t.partition.channels.iter().filter(|c| c.occupied) {
            let e = atom.eps[c.l as usize][0];
            assert!((e - c.eps).abs() < 1e-3, "Z = {z}, l = {}: KB {e:.6} vs semilocal/AE {:.6}", c.l, c.eps);
        }
    }
}

/// Spec §8 row 3, second first-failing test, as Spec question 4 rules (plan review: claim confirmed,
/// ruling accepted with two test amendments): the GSS criterion and G2 must catch both constructible
/// ghost cases on O's s channel:
/// (A) E_KB > 0 with a local so deep that ε₁^loc < ε_s (the textbook deep-local ghost);
/// (B) E_KB < 0 with ε₀^loc < ε_s (a local repulsive where φ_s lives, with a narrow deep well at the origin).
/// (A local with ε₀^loc above ε_s forces E_KB < 0 by the variational principle, which GSS calls ghost-free.)
/// G2 is checked on the fixed reference screening (v_loc + t.screen), unconditionally, in both cases
/// (review Q4(b)): a ghost can make the KB SCF fail, so a check behind `if let Ok(atom)` is vacuous
/// exactly when it matters. By interlacing, each case puts a KB level below ε_s besides ε_s itself
/// (A: ε₀^KB ≤ ε₁^loc < ε_s; B: ε₀^KB ≤ ε₀^loc < ε_s), so the exact count at ε_s + g2_window is ≥ 2
/// where the s channel expects 1.
#[test]
fn deliberate_bad_local_is_caught_as_ghost() {
    let t = generate(8).unwrap();
    let g = &t.grid;
    let un = crate::radial::tm::unscreen(g, t.z_v, &t.channels, &t.partition.channels.iter().map(|c| c.occ).collect::<Vec<_>>(), &t.kb.rho_pc);
    let eps_s = t.partition.channels[0].eps;
    let rc = t.channels[0].r_c;
    let top = eps_s + LEDGER.gates.g2_window;
    let screened = |v: &[f64]| v.iter().zip(&t.screen).map(|(a, b)| a + b).collect::<Vec<f64>>();
    let s_levels = |v: &[f64]| (bound_state(g, &screened(v), 0, 0, 0.0, None).map(|b| b.e), bound_state(g, &screened(v), 0, 1, 0.0, None).map(|b| b.e));
    let bump = |w: f64| g.r.iter().map(|r| (-(r / w).powi(2)).exp()).collect::<Vec<f64>>();
    // G2 on the reference screening: the exact inertia count below `top`, and the roots the mesh
    // locates there (the mesh sized for at most 2·10⁴ determinant evaluations, so the deep case B
    // stays a unit test; the count, not the mesh, is the proof)
    let g2 = |kb: &KbPotential| {
        let s = kb.projectors.iter().find(|p| p.l == 0).unwrap();
        let v = screened(&kb.v_loc);
        let proj = Some((s.e_kb, s.beta.as_slice()));
        let count = kb_count_below(g, &v, proj, 0, top);
        let lo = kb_lower_bound(g, &v, 0, proj);
        let roots = kb_eigenvalues(g, &v, proj, 0, lo, top, ((top - lo) / 2e4).max(LEDGER.numerics.g2_mesh));
        (count, roots)
    };
    // (A) V_loc = V_p^ion − A·e^{−(r/r_c)²}, A raised until two local s levels lie below ε_s
    let p_ion = &un.v_ion[1];
    let wide = bump(rc);
    let a = (1..400).map(|k| k as f64 * 0.5).find(|&a| {
        let v: Vec<f64> = p_ion.iter().zip(&wide).map(|(p, b)| p - a * b).collect();
        matches!(s_levels(&v), (_, Some(e1)) if e1 < eps_s)
    }).expect("a deep enough local exists");
    let v_a: Vec<f64> = p_ion.iter().zip(&wide).map(|(p, b)| p - a * b).collect();
    let kb_a = kb_form_with_local(g, 8, t.z_v, &t.channels, &un, v_a, None, t.kb.rho_pc.clone());
    let s = kb_a.projectors.iter().find(|p| p.l == 0).unwrap();
    assert!(s.e_kb > 0.0, "case A must have E_KB > 0, got {}", s.e_kb);
    let err = ghost_gates(g, &kb_a, &t.screen, &t.partition).expect_err("case A is a ghost");
    assert!(err.contains("G1") && err.contains("l = 0"), "{err}");
    let (e0, e1) = s_levels(&kb_a.v_loc);
    assert!(matches!(gss_verdict(s.e_kb, eps_s, e0, e1), GssVerdict::Ghost { .. }));
    let (count, roots) = g2(&kb_a);
    assert!(count >= 2, "case A: G2's inertia count below {top:.3} Ha is {count}; a ghost besides ε_s makes it ≥ 2");
    assert_eq!(roots.len(), count, "case A: mesh roots {roots:?} vs the inertia count {count}");
    // (B) V_loc = V_s^ion + 1 Ha·e^{−(r/r_c)²} − D·e^{−(r/w)²}, w = 0.05 Bohr, D scanned geometrically
    // over [1, 1e4] Ha (review Q4(a)): binding a deep level needs D·w² ≳ 1–10, while the narrow well's
    // share of ⟨φ|δV|φ⟩ grows only as D·w³·u′(0)² ∝ w, so at w = 0.15 the window D_bind < D < D_flip
    // (E_KB turning positive at D_flip) can be empty.
    let s_ion = &un.v_ion[0];
    let narrow = bump(0.05);
    let mut found = None;
    for k in 0..=80 {
        let d = 10f64.powf(k as f64 / 20.0);
        let v: Vec<f64> = (0..g.len()).map(|i| s_ion[i] + wide[i] - d * narrow[i]).collect();
        if !matches!(s_levels(&v), (Some(e0), _) if e0 < eps_s) { continue; }
        let kb_b = kb_form_with_local(g, 8, t.z_v, &t.channels, &un, v, None, t.kb.rho_pc.clone());
        let e_kb = kb_b.projectors.iter().find(|p| p.l == 0).unwrap().e_kb;
        if e_kb < 0.0 { found = Some((d, kb_b)); }
        break; // ⟨φ|δV|φ⟩ grows with D: past D_bind, E_KB < 0 here or never
    }
    let (d, kb_b) = found.expect("case B's window D_bind < D < D_flip was not found for D ≤ 1e4 Ha at w = 0.05 Bohr — report the scan (review Q4(a))");
    let sb = kb_b.projectors.iter().find(|p| p.l == 0).unwrap();
    let err = ghost_gates(g, &kb_b, &t.screen, &t.partition).expect_err("case B is a ghost");
    assert!(err.contains("G1"), "{err}");
    let (e0, e1) = s_levels(&kb_b.v_loc);
    assert!(matches!(gss_verdict(sb.e_kb, eps_s, e0, e1), GssVerdict::Ghost { .. }));
    let (count, roots) = g2(&kb_b);
    assert!(count >= 2, "case B (D = {d:.1} Ha): G2's inertia count below {top:.3} Ha is {count}; a ghost besides ε_s makes it ≥ 2");
    assert_eq!(roots.len(), count, "case B: mesh roots {roots:?} vs the inertia count {count}");
}

/// Plan review Q5: the inertia count n_KB(E) = n_loc(E) + [s(E) < 0] − [E_KB > 0] equals the number
/// of KB roots below E, for every projector channel of C–Ne at several energies, and is the plain
/// node count without a projector.
#[test]
fn inertia_count_matches_the_roots() {
    for z in [6u8, 8, 10] {
        let t = generate(z).unwrap();
        let v: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
        for p in &t.kb.projectors {
            let proj = Some((p.e_kb, p.beta.as_slice()));
            let lo = kb_lower_bound(&t.grid, &v, p.l, proj);
            for e in [-2.0, -1.0, -0.5, -0.1, 0.0] {
                let roots = kb_eigenvalues(&t.grid, &v, proj, p.l, lo, e, LEDGER.numerics.g2_mesh);
                assert_eq!(kb_count_below(&t.grid, &v, proj, p.l, e), roots.len(), "Z = {z}, l = {}, E = {e}: roots {roots:?}", p.l);
            }
        }
        let lo = kb_lower_bound(&t.grid, &v, 1, None);
        assert_eq!(kb_count_below(&t.grid, &v, None, 1, -0.05), kb_eigenvalues(&t.grid, &v, None, 1, lo, -0.05, LEDGER.numerics.g2_mesh).len());
    }
}

#[test]
fn gss_criterion_cases() {
    assert!(matches!(gss_verdict(-1.0, -1.0, Some(-0.5), Some(0.0)), GssVerdict::GhostFree));
    assert!(matches!(gss_verdict(-1.0, -1.0, Some(-2.0), Some(-0.5)), GssVerdict::Ghost { .. }));
    assert!(matches!(gss_verdict(1.0, -1.0, Some(-2.0), Some(-0.5)), GssVerdict::GhostFree));
    assert!(matches!(gss_verdict(1.0, -1.0, Some(-3.0), Some(-2.0)), GssVerdict::Ghost { .. }));
    assert!(matches!(gss_verdict(1.0, -1.0, Some(-2.0), None), GssVerdict::GhostFree)); // ε₁ unbound: the continuum edge
}

/// The polynomial local candidate (spec §4.4): −Z_v/r beyond r_loc, matched C² there, even inside.
#[test]
fn polynomial_local_is_c2_at_r_loc() {
    let g = crate::radial::LogGrid::universal();
    let (z_v, r_loc) = (6.0, 1.2);
    let v = polynomial_local(&g, z_v, r_loc);
    assert!((v[0] + 15.0 * z_v / (8.0 * r_loc)).abs() < 1e-9);
    let (dv, d2v) = (g.derivative(&v), g.derivative(&g.derivative(&v)));
    let i = g.nearest(r_loc);
    let r = g.r[i];
    assert!((v[i] + z_v / r).abs() < 1e-6 && (dv[i] - z_v / (r * r)).abs() < 1e-4 && (d2v[i] + 2.0 * z_v / r.powi(3)).abs() < 1e-2);
}
```
`crates/qm/src/radial/recipe.rs` tests:
```rust
/// Task 3 ladder (G1 + G2 only; Task 4 adds T1–T7): C–Ne generate, nothing refused, f from the ladder.
#[test]
fn second_row_generates_on_the_ladder() {
    for z in 6u8..=10 {
        let t = super::generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
        assert!(crate::radial::LEDGER.f_ladder.contains(&t.f));
        assert_eq!(t.n_core, 2);
        eprintln!("Z = {z}: f = {}, local {:?}, r_c = {:?}", t.f, t.local, t.channels.iter().map(|c| (c.l, c.r_c)).collect::<Vec<_>>());
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release radial::kb radial::recipe`
Expected: compile errors.

- [ ] **Step 3: Implement the local candidates and the KB form**

`crates/qm/src/radial/kb.rs`. Its module doc derives:
- the polynomial local;
- the KB operator and its normalisation (S1);
- GSS (with the review NOTE's interlacing argument);
- the matching determinant, and det M = E_KB·W(u_h, u_i)·s(E) with s(E) = −1/E_KB − ⟨β|G_loc(E)|β⟩;
- the inertia count n_KB(E) = n_loc(E) + [s(E) < 0] − [E_KB > 0] (Haynsworth; plan review Q5);
- the variational argument behind Spec question 4, and the interlacing that puts a second KB level below ε_s in both ghost cases.

```rust
use super::atom::{hartree_potential, Diis, RadialError, SpinMode};
use super::grid::LogGrid;
use super::ledger::{LocalCandidate, LEDGER};
use super::numerov::{bound_state, eigenfunction, g_coeff, outward, outward_sourced, regular_start, shooting_end};
use super::partition::Partition;
use super::tm::{TmChannel, Unscreened};
use crate::grid::neumaier;
use crate::xc::exc_point;
use std::f64::consts::PI;

/// One KB projector of channel l: E_KB = 1/∫u δV u dr, GSS = ∫β²/∫uβ, β = δV·u (u-form), δV = V_l^ion −
/// V_loc, u = u_l^PP. β and δV are exactly 0 from `i_beta` on: r_β = r[i_beta] is the projector's support
/// (spec §4.6), derived in [`kb_form`] (module doc, "The projector support").
#[derive(Clone, Debug, PartialEq)]
pub struct Projector { pub l: u8, pub e_kb: f64, pub gss: f64, pub beta: Vec<f64>, pub dv: Vec<f64>, pub u: Vec<f64>, pub i_beta: usize }

/// The separable potential: V_loc + Σ projectors; `local_l` is the channel used as V_loc (None for the
/// polynomial local or a test's hand-made local); `rho_pc` the partial core in E_xc; r_loc = min r_c;
/// r_cmp from the unscreening.
#[derive(Clone, Debug, PartialEq)]
pub struct KbPotential {
    pub z: u8, pub z_v: f64, pub local: LocalCandidate, pub local_l: Option<u8>, pub v_loc: Vec<f64>,
    pub projectors: Vec<Projector>, pub rho_pc: Vec<f64>, pub r_loc: f64, pub r_cmp: f64,
}

/// What the KB atom needs: (l, E_KB, β u-form) per projector.
#[derive(Clone, Debug, PartialEq)]
pub struct KbInput { pub z: u8, pub z_v: f64, pub v_loc: Vec<f64>, pub projectors: Vec<(u8, f64, Vec<f64>)>, pub rho_pc: Vec<f64> }

/// Valence occupations, index = l, (↑, ↓).
#[derive(Clone, Debug, PartialEq)]
pub struct ValenceConfig { pub occ: Vec<[f64; 2]> }

/// The converged KB atom: ε and u per (l, spin) (NaN/empty where unoccupied), spin densities, the
/// potentials of the output density, the total valence energy (with E_xc[ρ_v + ρ_pc]).
#[derive(Clone, Debug, PartialEq)]
pub struct KbAtom { pub eps: Vec<[f64; 2]>, pub u: Vec<[Vec<f64>; 2]>, pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub energy: f64 }

#[derive(Clone, Debug, PartialEq)]
pub enum GssVerdict { GhostFree, Ghost { e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64> } }

/// G1 per projector; G2's worst eigenvalue error and the located roots per l ≤ l_max; `g2_missing`:
/// (l, AE energies, KB roots) below the G2 window top where the pseudo-atom binds fewer states than the
/// all-electron atom — not a ghost, recorded for T5 and the generation record (Task 3 review S1).
#[derive(Clone, Debug, PartialEq)]
pub struct GhostReport { pub g1: Vec<(u8, GssVerdict)>, pub g2_max_eig_error: f64, pub g2_roots: Vec<(u8, Vec<f64>)>, pub g2_missing: Vec<(u8, Vec<f64>, Vec<f64>)> }

/// τ_c(r) = V_H[ρ_c](r) − N_c/r (spec §4.3): the frozen core's Hartree tail, ≤ 0, ≡ 0 without a core.
/// `rho_c` must be the *full* core density (ρ_pc never enters V_H) and N_c = Z − Z_v.
pub fn core_tail(grid: &LogGrid, rho_c: &[f64], n_core: f64) -> Vec<f64> {
    let vh = hartree_potential(grid, rho_c);
    grid.r.iter().zip(&vh).map(|(r, v)| v - n_core / r).collect()
}

/// The polynomial local (module doc): T = −Z_v/r + τ_c from r_loc outwards, the even quartic matched C² to T
/// inside. r_loc must be a grid point (the generator's min r_c is one); an off-grid caller is told, not moved.
pub fn polynomial_local(grid: &LogGrid, z_v: f64, r_loc: f64, tau_c: &[f64]) -> Vec<f64> {
    assert_eq!(tau_c.len(), grid.len(), "τ_c is a grid table");
    let i = grid.nearest(r_loc);
    let r = grid.r[i];
    assert!((r - r_loc).abs() < 1e-12 * r_loc, "r_loc = {r_loc} is not a grid point (nearest {r})");
    let (dt, d2t) = { let d = grid.derivative(tau_c); let d2 = grid.derivative(&d); (d[i], d2[i]) };
    let t0 = -z_v / r + tau_c[i];
    let t1 = z_v / (r * r) + dt;
    let t2 = -2.0 * z_v / r.powi(3) + d2t;
    let a4 = (t2 - t1 / r) / (8.0 * r * r);
    let a2 = (t1 - 4.0 * a4 * r.powi(3)) / (2.0 * r);
    let a0 = t0 - a2 * r * r - a4 * r.powi(4);
    grid.r.iter().zip(tau_c).enumerate()
        .map(|(k, (&x, &t))| if k >= i { -z_v / x + t } else { a0 + x * x * (a2 + x * x * a4) }).collect()
}

/// The KB potential with the local candidate `local` (spec §4.4): the highest generated channel, or the
/// polynomial local with the core tail `tau_c` ([`core_tail`] of the full core).
#[allow(clippy::too_many_arguments)]
pub fn kb_form(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, local: LocalCandidate, rho_pc: Vec<f64>, tau_c: &[f64]) -> KbPotential {
    let top = channels.len() - 1;
    let (v_loc, local_l) = match local {
        LocalCandidate::HighestChannel => (un.v_ion[top].clone(), Some(channels[top].l)),
        LocalCandidate::Polynomial => (polynomial_local(grid, z_v, r_loc(channels), tau_c), None),
    };
    let mut kb = kb_form_with_local(grid, z, z_v, channels, un, v_loc, local_l, rho_pc);
    kb.local = local;
    // review ruling (b) (Task 3 report): the derived support r_β, the first radius from which every V_l^ion is
    // −Z_v/r + τ_c to v_ion_tail/100 for good (module doc)
    let n = grid.len();
    let tol = LEDGER.v_ion_tail / 100.0;
    let i_beta = (0..n).rev()
        .find(|&i| un.v_ion.iter().any(|v| (v[i] + z_v / grid.r[i] - tau_c[i]).abs() >= tol))
        .map_or(0, |i| i + 1).min(n - 1);
    for p in &mut kb.projectors {
        let worst = (i_beta..n).map(|i| p.dv[i].abs()).fold(0.0, f64::max);
        assert!(worst < tol, "Z = {z}, l = {}: δV = {worst:.2e} beyond r_β = {:.4} — the local does not carry the ionic tail (a construction bug)", p.l, grid.r[i_beta]);
        for i in i_beta..n { p.dv[i] = 0.0; p.beta[i] = 0.0; }
        let denom = grid.integrate(&(0..n).map(|i| p.u[i] * p.beta[i]).collect::<Vec<_>>());
        let num = grid.integrate(&p.beta.iter().map(|b| b * b).collect::<Vec<_>>());
        p.e_kb = 1.0 / denom;
        p.gss = num / denom;
        p.i_beta = i_beta;
    }
    kb
}

fn r_loc(channels: &[TmChannel]) -> f64 { channels.iter().map(|c| c.r_c).fold(f64::INFINITY, f64::min) }

/// Every channel other than `local_l` becomes a projector δV_l = V_l^ion − V_loc. `local` is set to
/// `HighestChannel` here; [`kb_form`] overwrites it with the candidate it built and derives r_β. A
/// hand-made local (the ghost tests) has no derived support: `i_beta` is then one point past the last
/// |β| > 1e-12·max|β| (a method fallback, never a physical radius).
#[allow(clippy::too_many_arguments)]
pub fn kb_form_with_local(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, v_loc: Vec<f64>, local_l: Option<u8>, rho_pc: Vec<f64>) -> KbPotential {
    let n = grid.len();
    let projectors = channels.iter().enumerate().filter(|(_, c)| Some(c.l) != local_l).map(|(k, c)| {
        let dv: Vec<f64> = (0..n).map(|i| un.v_ion[k][i] - v_loc[i]).collect();
        let beta: Vec<f64> = (0..n).map(|i| dv[i] * c.u[i]).collect();
        let denom = grid.integrate(&(0..n).map(|i| c.u[i] * beta[i]).collect::<Vec<_>>());
        let num = grid.integrate(&beta.iter().map(|b| b * b).collect::<Vec<_>>());
        let i_beta = (support_index(&beta) + 1).min(n - 1);
        Projector { l: c.l, e_kb: 1.0 / denom, gss: num / denom, beta, dv, u: c.u.clone(), i_beta }
    }).collect();
    KbPotential { z, z_v, local: LocalCandidate::HighestChannel, local_l, v_loc, projectors, rho_pc, r_loc: r_loc(channels), r_cmp: un.r_cmp }
}

impl KbPotential {
    pub fn input(&self) -> KbInput {
        KbInput { z: self.z, z_v: self.z_v, v_loc: self.v_loc.clone(),
                  projectors: self.projectors.iter().map(|p| (p.l, p.e_kb, p.beta.clone())).collect(), rho_pc: self.rho_pc.clone() }
    }
}

impl Projector {
    /// r_β, the projector's support (spec §4.6): β ≡ 0 from here on. Derived for [`kb_form`] tables —
    /// max(max_l r_c,l, r_nlc) to the grid — and what Tasks 5 and 6 size the filter and the sphere by.
    pub fn support(&self, grid: &LogGrid) -> f64 { grid.r[self.i_beta] }
}

fn support_index(beta: &[f64]) -> usize {
    let m = beta.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    beta.iter().rposition(|b| b.abs() > 1e-12 * m).unwrap_or(0)
}
```

- [ ] **Step 4: Implement the radial KB eigenproblem**

As built (Task 3, 68b6b40 + review round 1; each deviation from the brief's numerics carries a one-line comment citing the Task 3 report):

```rust
/// β's support plus 3 points: the ε-independent matching index (an ε-dependent one would make det M jump).
fn matching_index(beta: &[f64]) -> usize { support_index(beta) + 3 }

/// The pieces of the Green-function form (module doc, "Stability") at energy e: u_h on [0, r_{i_m+2}], u_i
/// on [r_0, end] (inward from `shooting_end`, 0 beyond), the r-Wronskian W = W_r(u_h, u_i) taken where the
/// solutions are smoothest, its natural scale |u_h u_i′| + |u_h′ u_i| there, and C_h(r) = ∫₀^r β u_h dr′.
// deviation 1 (Task 3 report): det M through the Green kernel — the brief's u_h/u_p form cancels e^{2κr_c} at depth
struct Green { uh: Vec<f64>, ui: Vec<f64>, w: f64, w_scale: f64, ch: Vec<f64> }

fn green(grid: &LogGrid, v: &[f64], beta: &[f64], l: u8, e: f64, i_m: usize) -> Green {
    let n = grid.len();
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    let (yh, _) = outward(&g, y0, y1, i_m + 2, grid.delta);
    let ie = shooting_end(&g, grid.delta).max(i_m + 4).min(n - 1);
    let yi = inward_dirichlet(&g, ie, grid.delta);
    let sr = |i: usize| grid.r[i].sqrt();
    let uh: Vec<f64> = (0..=i_m + 2).map(|i| yh[i] * sr(i)).collect();
    let ui: Vec<f64> = (0..n).map(|i| if i <= ie { yi[i] * sr(i) } else { 0.0 }).collect();
    let dx = |f: &[f64], i: usize| (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) / (12.0 * grid.delta);
    // W is constant: take it where the solutions are smoothest in x, the point of least |g| (the turning
    // point when there is one). At r_m they change by e^{±√g δ} per step and the 5-point derivative's
    // relative error (√g δ)⁴/30 reaches 3e-3 (the oscillator oracle at r_m = 8.4) — harmless inside M, whose
    // row 1 carries it in both entries, but not in s
    let iw = (2..=i_m - 2).min_by(|&a, &b| g[a].abs().partial_cmp(&g[b].abs()).unwrap()).unwrap();
    let (a, b) = (uh[iw] * dx(&ui, iw), dx(&uh, iw) * ui[iw]);
    let ch = grid.cumulative(&(0..n).map(|i| if i <= i_m { beta[i] * uh[i] } else { 0.0 }).collect::<Vec<_>>());
    Green { uh, ui, w: (a - b) / grid.r[iw], w_scale: (a.abs() + b.abs()) / grid.r[iw], ch }
}

/// The solution with y(x_{i_start}) = 0, run inward (Numerov) to the first grid point: the Dirichlet
/// condition at r_ie that n_loc's Sturm count (nodes of u_h on (0, r_ie)) assumes. `numerov::inward`
/// starts instead from the WKB decay ratio, which is not that solution once the decay length exceeds the
/// box (ε → 0): the count then fell from 2 to 1 between ε = −1e-3 and −1e-4 Ha (polynomial-local Al s at
/// f = 1.0, Task 3 round 1) — impossible for a count. Deep in the gap both starts give the same, decaying,
/// solution. Rescaled by 1e-150 past |y| > 1e150 (scale-free).
// round 1 (Task 3 report): u_i vanishes at r_ie, the Dirichlet problem n_loc's Sturm count assumes
fn inward_dirichlet(g: &[f64], i_start: usize, delta: f64) -> Vec<f64> {
    let k = delta * delta / 12.0;
    let mut y = vec![0.0; i_start + 1];
    y[i_start - 1] = 1e-20;
    for i in (1..i_start).rev() {
        y[i - 1] = (2.0 * y[i] * (1.0 + 5.0 * k * g[i]) - y[i + 1] * (1.0 - k * g[i + 1])) / (1.0 - k * g[i - 1]);
        if y[i - 1].abs() > 1e150 { for v in &mut y[i - 1..] { *v *= 1e-150; } }
    }
    y
}

/// E_KB·W·s(e) = −W + 4E_KB∫₀^{r_m} β u_i C_h dr (= det M/r_m, evaluated stably) and W.
fn kb_green(grid: &LogGrid, v: &[f64], e_kb: f64, beta: &[f64], l: u8, e: f64, i_m: usize) -> (f64, f64) {
    let n = grid.len();
    let gr = green(grid, v, beta, l, e, i_m);
    let j = grid.partial(&(0..n).map(|i| if i <= i_m { beta[i] * gr.ui[i] * gr.ch[i] } else { 0.0 }).collect::<Vec<_>>(), i_m);
    (-gr.w + 4.0 * e_kb * j, gr.w)
}

/// min_r(V + l(l+1)/2r²) + min(0, E_KB∫β²): a lower bound of channel l's KB spectrum (the kinetic
/// operator is ≥ 0 and ⟨u|β⟩² ≤ ∫β² for ∫u² = 1).
pub fn kb_lower_bound(grid: &LogGrid, v: &[f64], l: u8, proj: Option<(f64, &[f64])>) -> f64 {
    let lf = l as f64;
    let floor = grid.r.iter().zip(v).map(|(r, vi)| vi + lf * (lf + 1.0) / (2.0 * r * r)).fold(f64::INFINITY, f64::min);
    floor + proj.map_or(0.0, |(e, b)| (e * grid.integrate(&b.iter().map(|x| x * x).collect::<Vec<_>>())).min(0.0))
}

/// The number of local-channel eigenvalues below e (Sturm: the sign changes of the regular solution out
/// to `shooting_end`, as `bound_state` counts them).
fn local_count(grid: &LogGrid, v: &[f64], l: u8, e: f64) -> usize {
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    outward(&g, y0, y1, shooting_end(&g, grid.delta), grid.delta).1
}

/// The exact number of KB eigenvalues of channel l below e (plan review Q5; module doc):
/// n_KB(e) = n_loc(e) + [s(e) < 0] − [E_KB > 0], s = det M/(E_KB·W(u_h,u_i)) in its stable form
/// (`kb_green`). Without a projector: node counting.
pub fn kb_count_below(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> usize {
    let n_loc = local_count(grid, v, l, e);
    let Some((e_kb, beta)) = proj else { return n_loc };
    let (det, w) = kb_green(grid, v, e_kb, beta, l, e, matching_index(beta));
    let s = det / (e_kb * w);
    debug_assert!(n_loc + usize::from(s < 0.0) >= usize::from(e_kb > 0.0), "inertia count below zero: a W/node inconsistency (n_loc {n_loc}, s {s}, E_KB {e_kb})");
    (n_loc + usize::from(s < 0.0)).saturating_sub(usize::from(e_kb > 0.0))
}

/// Every eigenvalue of channel l in [e_lo, e_hi], ascending: without a projector by node counting
/// (`bound_state`); with one, by the sign changes of det M (its stable form, `kb_green`) on a mesh, each
/// bisected to 1e-13·max(1, |e|).
/// The mesh only *locates* roots — [`kb_count_below`] is the exact count G2 gates on (Spec question 5).
pub fn kb_eigenvalues(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e_lo: f64, e_hi: f64, mesh: f64) -> Vec<f64> {
    match proj {
        None => (0..).map_while(|k| bound_state(grid, v, l, k, 0.0, None)).map(|b| b.e)
            .take_while(|&e| e < e_hi).filter(|&e| e >= e_lo).collect(),
        Some((e_kb, beta)) => {
            let i_m = matching_index(beta);
            let det = |e: f64| kb_green(grid, v, e_kb, beta, l, e, i_m).0;
            let steps = ((e_hi - e_lo) / mesh).ceil().max(1.0) as usize;
            let (mut a, mut fa) = (e_lo, det(e_lo));
            let mut roots = Vec::new();
            for k in 1..=steps {
                let b = (e_lo + k as f64 * mesh).min(e_hi);
                let fb = det(b);
                if fa * fb < 0.0 {
                    let (mut lo, mut hi, mut flo) = (a, b, fa);
                    while hi - lo > 1e-13 * lo.abs().max(1.0) {
                        let mid = 0.5 * (lo + hi);
                        let fm = det(mid);
                        if flo * fm <= 0.0 { hi = mid } else { lo = mid; flo = fm; }
                    }
                    roots.push(0.5 * (lo + hi));
                }
                a = b;
                fa = fb;
            }
            roots
        }
    }
}

/// The normalised eigenfunction at an eigenvalue e (positive at its outermost maximum). With a projector,
/// u ∝ Gβ (H_KB u = εu ⇔ (H_loc − ε)u = −E_KB β⟨β|u⟩): u ∝ u_i(r)C_h(r) + u_h(r)D_i(r), D_i(r) = ∫_r^∞ u_i β
/// — the Green-function form `kb_eigenvalues` locates the roots with (module doc, "Stability"), so the
/// state is self-consistent at exactly that root, bounded at any depth, and needs no division by W. The
/// exception is ⟨β|u⟩ = 0: e is then a local eigenvalue with its state orthogonal to β (W = 0 and Gβ
/// vanishes), and the state is the local one — taken when |W| < 1e-6 of its scale (a method literal: a
/// measure-zero coincidence for a real potential, exercised by the oscillator oracle's untouched levels).
// deviation 2 (Task 3 report): u ∝ Gβ, the root-finder's own kernel — M's null vector cancels growing parts
pub fn kb_state(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> Vec<f64> {
    let Some((_, beta)) = proj else { return eigenfunction(grid, v, l, e, 0.0).u };
    let n = grid.len();
    let i_m = matching_index(beta);
    let gr = green(grid, v, beta, l, e, i_m);
    if gr.w.abs() < 1e-6 * gr.w_scale { return eigenfunction(grid, v, l, e, 0.0).u; }
    let di = grid.cumulative(&(0..n).map(|i| if i <= i_m { gr.ui[i] * beta[i] } else { 0.0 }).collect::<Vec<_>>());
    let d_tot = di[i_m];
    let u: Vec<f64> = (0..n).map(|i| if i <= i_m { gr.ui[i] * gr.ch[i] + gr.uh[i] * (d_tot - di[i]) } else { gr.ui[i] * gr.ch[i_m] }).collect();
    let norm = grid.integrate(&u.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
    let umax = u.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let ip = (1..n - 1).rev()
        .find(|&i| u[i].abs() >= u[i - 1].abs() && u[i].abs() >= u[i + 1].abs() && u[i].abs() > LEDGER.r_peak_floor * umax)
        .unwrap_or(0);
    let sgn = if u[ip] < 0.0 { -1.0 } else { 1.0 };
    u.iter().map(|x| sgn * x / norm).collect()
}

/// The regular solution at energy e out to i_end (unnormalised): T5's log derivatives (Task 4). With a
/// projector: u = (1 − E_KB⟨β|u_p⟩)·u_h + E_KB⟨β|u_h⟩·u_p, ⟨β|·⟩ over β's support. Then E_KB⟨β|u⟩ =
/// E_KB⟨β|u_h⟩ is the coefficient of u_p — the self-consistent c, scaled by (1 − E_KB⟨β|u_p⟩) instead of
/// divided by it, so there is no pole where that factor vanishes (plan review Q12). Homogeneous of degree
/// one in u_h, so its overflow rescaling is harmless.
pub fn kb_regular(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64, i_end: usize) -> Vec<f64> {
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    let reach = proj.map_or(i_end, |(_, b)| matching_index(b).max(i_end));
    let (yh, _) = outward(&g, y0, y1, reach, grid.delta);
    let sr = |i: usize| grid.r[i].sqrt();
    let uh: Vec<f64> = (0..=reach).map(|i| yh[i] * sr(i)).collect();
    let Some((e_kb, beta)) = proj else { return uh[..=i_end].to_vec() };
    let s: Vec<f64> = (0..=reach).map(|i| 2.0 * grid.r[i].powf(1.5) * beta[i]).collect();
    let up: Vec<f64> = outward_sourced(&g, &s, reach, grid.delta).iter().enumerate().map(|(i, y)| y * sr(i)).collect();
    let n = grid.len();
    let dot = |u: &[f64]| grid.partial(&(0..n).map(|i| if i <= reach { beta[i] * u[i] } else { 0.0 }).collect::<Vec<_>>(), reach);
    let (a, c) = (1.0 - e_kb * dot(&up), e_kb * dot(&uh));
    (0..=i_end).map(|i| a * uh[i] + c * up[i]).collect()
}
```

- [ ] **Step 5: Implement the radial KB atom (SCF) and the gates**

As built (Task 3, 68b6b40 + review round 1; each deviation from the brief's numerics carries a one-line comment citing the Task 3 report):

```rust
/// The reference configuration: each occupied channel's occupation, half per spin; index l = 0..=l_max.
pub fn reference_config(part: &Partition) -> ValenceConfig {
    ValenceConfig { occ: (0..=part.l_max).map(|l| {
        let o = part.channels.iter().find(|c| c.l == l && c.occupied).map_or(0.0, |c| c.occ);
        [0.5 * o, 0.5 * o]
    }).collect() }
}

/// V_σ = V_loc + V_H[ρ↑ + ρ↓] + v_xc,σ[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc].
fn kb_potential(grid: &LogGrid, input: &KbInput, rho: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
    let n = grid.len();
    let total: Vec<f64> = (0..n).map(|i| rho[0][i] + rho[1][i]).collect();
    let v_h = hartree_potential(grid, &total);
    let mut v = [vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let c = 0.5 * input.rho_pc[i];
        let (_, vu, vd) = exc_point(rho[0][i] + c, rho[1][i] + c);
        v[0][i] = input.v_loc[i] + v_h[i] + vu;
        v[1][i] = input.v_loc[i] + v_h[i] + vd;
    }
    v
}

/// The lowest eigenvalue of channel l, by bisection on the exact count (`kb_count_below`): the infimum of
/// {e : n_KB(e) ≥ 1} over [lower bound, 0], to 1e-13·max(1, |e|); without a projector `bound_state` with
/// no node (the same bisection on the node count). `prev` (the last SCF step's ε) only narrows the
/// bracket, ±0.1·max(|prev|, 0.1) Ha, and only where the count confirms it (as `bound_state` uses a
/// guess). Deviation from the brief, whose search returned the first root in a ±0.1 Ha window around
/// `prev`: when the level moved by more than that between early SCF steps the window held the next state,
/// and H's KB atom converged to a self-consistent 2s configuration (ε = −0.0712 vs −0.2335 Ha, measured).
// deviation 3 (Task 3 report): bisection on the exact count — the brief's ±0.1 Ha window put H in a 2s state
fn lowest(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, prev: f64) -> Option<f64> {
    let guess = prev.is_finite().then_some(prev);
    if proj.is_none() { return bound_state(grid, v, l, 0, 0.0, guess).map(|b| b.e); }
    let count = |e: f64| kb_count_below(grid, v, proj, l, e);
    let mut hi = 0.0;
    if count(hi) == 0 { return None; }
    let mut lo = kb_lower_bound(grid, v, l, proj);
    if let Some(gs) = guess {
        let w = 0.1 * gs.abs().max(0.1);
        if gs - w > lo && count(gs - w) == 0 { lo = gs - w; }
        if gs + w < hi && count(gs + w) >= 1 { hi = gs + w; }
    }
    while hi - lo > 1e-13 * lo.abs().max(1.0) {
        let mid = 0.5 * (lo + hi);
        if count(mid) >= 1 { hi = mid } else { lo = mid }
    }
    Some(0.5 * (lo + hi))
}

/// The self-consistent radial KB atom (module doc). `Restricted`: both spins in the ↑ potential.
pub fn solve_kb_atom(grid: &LogGrid, input: &KbInput, config: &ValenceConfig, spin: SpinMode) -> Result<KbAtom, RadialError> {
    let n = grid.len();
    let num = &LEDGER.numerics;
    let w: Vec<f64> = grid.r.iter().map(|r| r * r * r * grid.delta).collect();
    let nl = config.occ.len();
    let mut eps = vec![[f64::NAN; 2]; nl];
    let mut us: Vec<[Vec<f64>; 2]> = vec![[Vec::new(), Vec::new()]; nl];
    let occupy = |v: &[Vec<f64>; 2], eps: &mut Vec<[f64; 2]>, us: &mut Vec<[Vec<f64>; 2]>| -> Result<[Vec<f64>; 2], RadialError> {
        let mut rho = [vec![0.0; n], vec![0.0; n]];
        for l in 0..nl {
            let proj = input.projectors.iter().find(|p| p.0 as usize == l).map(|p| (p.1, p.2.as_slice()));
            for s in 0..2 {
                if config.occ[l][s] <= 0.0 { continue; }
                if spin == SpinMode::Restricted && s == 1 && config.occ[l][0] > 0.0 {
                    // the same potential: the ↑ state
                    eps[l][1] = eps[l][0];
                    us[l][1] = us[l][0].clone();
                } else {
                    let src = if spin == SpinMode::Restricted { 0 } else { s };
                    let e = lowest(grid, &v[src], proj, l as u8, eps[l][s])
                        .ok_or(RadialError::Unbound { z: input.z, n: 0, l: l as u8, after_backtracking: false })?;
                    us[l][s] = kb_state(grid, &v[src], proj, l as u8, e);
                    eps[l][s] = e;
                }
                for i in 0..n { rho[s][i] += config.occ[l][s] * us[l][s][i] * us[l][s][i] / (4.0 * PI * grid.r[i] * grid.r[i]); }
            }
        }
        Ok(rho)
    };
    // deviation 4 (Task 3 report): ρ_in = 0 (the bare pseudo-ion), as `atom` starts (its deviation 1): the bare-ion density is too
    // compact, and taken whole as the first input it over-screens and unbinds the p shell (C–Ne, measured);
    // from ρ = 0 the first mixing step feeds in only α of it
    let mut rho_in = [vec![0.0; n], vec![0.0; n]];
    let mut diis = Diis::new(num.diis_history);
    let mut delta = f64::INFINITY;
    for _ in 0..num.scf_max_iter {
        let v = kb_potential(grid, input, &rho_in);
        let rho_out = occupy(&v, &mut eps, &mut us)?;
        delta = (0..2).map(|s| (0..n).map(|i| (rho_out[s][i] - rho_in[s][i]).abs() * w[i]).sum::<f64>()).sum();
        if delta < num.scf_tol {
            let shell = |f: &[f64]| grid.integrate(&f.iter().zip(&grid.r).map(|(a, r)| 4.0 * PI * r * r * a).collect::<Vec<_>>());
            let band = neumaier((0..nl).flat_map(|l| (0..2).filter(move |&s| config.occ[l][s] > 0.0).map(move |s| (l, s))).map(|(l, s)| config.occ[l][s] * eps[l][s]));
            let pot = neumaier((0..2).map(|s| shell(&(0..n).map(|i| rho_out[s][i] * v[s][i]).collect::<Vec<_>>())));
            let total: Vec<f64> = (0..n).map(|i| rho_out[0][i] + rho_out[1][i]).collect();
            let v_h = hartree_potential(grid, &total);
            let e_loc = shell(&(0..n).map(|i| total[i] * input.v_loc[i]).collect::<Vec<_>>());
            let e_h = 0.5 * shell(&(0..n).map(|i| total[i] * v_h[i]).collect::<Vec<_>>());
            let e_xc = shell(&(0..n).map(|i| { let c = 0.5 * input.rho_pc[i]; exc_point(rho_out[0][i] + c, rho_out[1][i] + c).0 }).collect::<Vec<_>>());
            let v_out = kb_potential(grid, input, &rho_out);
            let energy = neumaier([band, -pot, e_loc, e_h, e_xc].into_iter());
            return Ok(KbAtom { eps, u: us, rho: rho_out, v: v_out, energy });
        }
        rho_in = diis.mix(rho_in, rho_out, &w, num.mixing);
    }
    Err(RadialError::NotConverged { z: input.z, iterations: num.scf_max_iter, delta })
}

/// GSS (module doc): ghost-free iff (E_KB < 0 ∧ ε_l < ε₀^loc) ∨ (E_KB > 0 ∧ ε₀^loc < ε_l < ε₁^loc); an
/// unbound local level sits at the continuum edge, 0.
pub fn gss_verdict(e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64>) -> GssVerdict {
    let (e0, e1) = (eps0_loc.unwrap_or(0.0), eps1_loc.unwrap_or(0.0));
    let ok = if e_kb < 0.0 { eps_ref < e0 } else { e0 < eps_ref && eps_ref < e1 };
    if ok { GssVerdict::GhostFree } else { GssVerdict::Ghost { e_kb, eps_ref, eps0_loc, eps1_loc } }
}

/// G1 (GSS, every projector channel, on the reference screening `screen`), then G2 (spec §4.6): the SCF KB atom at
/// the reference configuration reproduces the valence eigenvalues to LEDGER.gates.g2_eig, and for every l ≤ l_max
/// the exact inertia count below top = min(0, ε_v,max + g2_window) does not exceed the all-electron reference's
/// bound states of l above its core below top (`Partition::spectrum`). More is a ghost (refused); fewer is an AE
/// state the pseudo-atom does not bind, not a ghost (recorded in `g2_missing` for T5 and the record). The mesh roots
/// must number the count, else the mesh missed a pair (a numerical miss, not a ghost verdict).
pub fn ghost_gates(grid: &LogGrid, kb: &KbPotential, screen: &[f64], part: &Partition) -> Result<GhostReport, String> {
    let v_ref: Vec<f64> = kb.v_loc.iter().zip(screen).map(|(a, b)| a + b).collect();
    let mut g1 = Vec::new();
    for p in &kb.projectors {
        let ch = part.channels.iter().find(|c| c.l == p.l).expect("a projector has a channel");
        let e0 = bound_state(grid, &v_ref, p.l, 0, 0.0, None).map(|b| b.e);
        let e1 = bound_state(grid, &v_ref, p.l, 1, 0.0, None).map(|b| b.e);
        g1.push((p.l, gss_verdict(p.e_kb, ch.eps, e0, e1)));
    }
    if let Some((l, v)) = g1.iter().find(|(_, v)| *v != GssVerdict::GhostFree) {
        return Err(format!("G1: ghost in l = {l}: {v:?}"));
    }
    let atom = solve_kb_atom(grid, &kb.input(), &reference_config(part), SpinMode::Restricted).map_err(|e| format!("G2: {e}"))?;
    let g2_max_eig_error = part.channels.iter().filter(|c| c.occupied)
        .map(|c| (atom.eps[c.l as usize][0] - c.eps).abs()).fold(0.0, f64::max);
    if g2_max_eig_error >= LEDGER.gates.g2_eig {
        return Err(format!("G2: KB eigenvalues miss the reference by {:.3} mHa", g2_max_eig_error * 1e3));
    }
    let eps_v_max = part.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max);
    let top = (eps_v_max + LEDGER.gates.g2_window).min(0.0);
    let mut g2_roots = Vec::new();
    let mut g2_missing = Vec::new();
    for l in 0..=part.l_max {
        let proj = kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
        let count = kb_count_below(grid, &atom.v[0], proj, l, top);
        let roots = kb_eigenvalues(grid, &atom.v[0], proj, l, kb_lower_bound(grid, &atom.v[0], l, proj), top, LEDGER.numerics.g2_mesh);
        if roots.len() != count {
            return Err(format!("G2: l = {l}: the g2_mesh located {} roots ({roots:?}) but the inertia count is {count} — report (a numerical miss, not a ghost verdict)", roots.len()));
        }
        // deviation 5 + review S1 (Task 3 report): count against the AE spectrum, refuse only an excess.
        // The all-electron atom's own states of l above its core (Partition::spectrum): the neutral LSDA atom
        // binds one (n+1)s just below 0, which the KB potential reproduces — not a ghost. More than the AE
        // atom is a ghost; fewer is an AE state the pseudo-atom does not bind (T5's, recorded)
        let ae: Vec<f64> = part.spectrum[l as usize].iter().copied().filter(|&e| e < top).collect();
        if count > ae.len() {
            return Err(format!("G2: l = {l} has {count} bound states below {top:.3} Ha (inertia count), the all-electron atom {}: an extra state (ghost); KB {roots:?}, AE {ae:?}", ae.len()));
        }
        if count < ae.len() { g2_missing.push((l, ae, roots.clone())); }
        g2_roots.push((l, roots));
    }
    Ok(GhostReport { g1, g2_max_eig_error, g2_roots, g2_missing })
}
```

- [ ] **Step 6: Implement the ladder**

`crates/qm/src/radial/recipe.rs` (module doc: spec §4.4 in full, including the S6b paragraph that Task 11 implements):
```rust
use super::atom::{ground_state, AtomResult, SpinMode};
use super::grid::LogGrid;
use super::kb::{core_tail, ghost_gates, kb_form, GhostReport, KbPotential};
use super::ledger::{LocalCandidate, LEDGER};
use super::partition::{core_density, partition, Partition, Refusal};
use super::tm::{tm_channels, unscreen, TmChannel};
use std::sync::Arc;

/// A generated, gated pseudopotential with everything it was built from. `screen` is the reference
/// screening V_H[ρ_v^PP] + v_xc[ρ_v^PP + ρ_pc] (= V_l^scr − V_l^ion, the same for every l).
#[derive(Clone, Debug)]
pub struct PseudoTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub f: f64, pub local: LocalCandidate, pub kb: KbPotential,
    pub channels: Vec<TmChannel>, pub screen: Vec<f64>, pub partition: Partition,
    pub reference: Arc<AtomResult>, pub grid: LogGrid, pub ghost: GhostReport,
}

/// Z in, potential out (spec §4.4): f from `LEDGER.f_ladder` softest first, local candidates in
/// `LEDGER.local_order`; the first (f, local) passing every gate is kept; none → refused at this recipe
/// version with the reasons listed.
pub fn generate(z: u8) -> Result<PseudoTable, Refusal> {
    let reference = Arc::new(ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?);
    let part = partition(&reference)?;
    // Task 3: the full core as the partial core (Task 2's r_nlc → 0 limit); Task 4 replaces this line by
    // nlcc::partial_core (LFC, spec §4.5)
    let rho_pc = core_density(&reference, &part);
    let mut tried = Vec::new();
    for &f in LEDGER.f_ladder.iter() {
        for &local in LEDGER.local_order.iter() {
            match candidate(&reference, &part, f, local, &rho_pc) {
                Ok(t) => return Ok(t),
                Err(why) => tried.push(format!("f = {f}, {local:?}: {why}")),
            }
        }
    }
    Err(Refusal::NoLadderEntryPasses { z, tried })
}

/// One ladder entry: TM channels at f, unscreened with `rho_pc`, the KB form with `local`, the gates.
pub(crate) fn candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, rho_pc: &[f64]) -> Result<PseudoTable, String> {
    let g = &reference.grid;
    let channels = tm_channels(reference, part, f)?;
    let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
    let un = unscreen(g, part.z_v, &channels, &occ, rho_pc);
    let screen: Vec<f64> = (0..g.len()).map(|i| channels[0].v_scr[i] - un.v_ion[0][i]).collect();
    // τ_c from the *full* core (whatever the partial core): N_c = Z − Z_v
    let tau_c = core_tail(g, &core_density(reference, part), part.n_core as f64);
    let kb = kb_form(g, part.z, part.z_v, &channels, &un, local, rho_pc.to_vec(), &tau_c);
    let ghost = ghost_gates(g, &kb, &screen, part)?;
    Ok(PseudoTable { z: part.z, z_v: part.z_v, n_core: part.n_core, f, local, kb, channels, screen,
                     partition: part.clone(), reference: reference.clone(), grid: g.clone(), ghost })
}
```
Add `pub mod kb; pub mod recipe;` to `radial/mod.rs` and re-export `kb::{…}` and `recipe::{generate, PseudoTable}`. In `atom.rs`, make `struct Diis`, `Diis::new` and `Diis::mix` `pub(crate)`.

- [ ] **Step 7: Run the tests**

Run: `cargo test -p qm --release radial`
Expected: all PASS. If `second_row_generates_on_the_ladder` refuses an element, report the `tried` list. A refusal is an outcome, not a bug to tune away (spec §7 risk 3), but the physics reviewer must see it before Task 4.

- [ ] **Step 8: Commit**
```bash
git add crates/qm/src/radial
git commit -o -m "feat(qm): Kleinman–Bylander form, radial KB atom, GSS/G2 ghost gates and the f ladder (M7 Task 3)" -- crates/qm/src/radial
```
Before merging, hand the diff to `physics-reviewer`, with Spec questions 4 and 5 named (the narrow-well scan's D, and the inertia-count derivation).

---

### Task 4: `nlcc` + gates T1–T7 + the generation record + `recipe_hash` **[Fable — physics-review gate]**

Spec §4.5, §4.8 (T1–T7), §4.9 (`recipe_hash`), §5.6 (r_core), §6.2, §8 row 4; review S4 (T5 in δE_equiv), S6a (the pinned outputs table) and M3 (r_core). **Spec question 12 applies:** the plan review (M5) rejected the plan's difference form and ruled the Wronskian form δE_equiv = ½·W(û_AE, û_PP)(r_d). Plan review M6(c) also lands here: the records test enforces the `LEDGER.version` bump rule (Global Constraints).

**Files:**
- Create: `crates/qm/src/radial/nlcc.rs`, `crates/qm/src/radial/gates.rs`, `crates/qm/src/radial/record.rs`, `crates/validate/tests/pseudo_gen.rs`
- Modify: `crates/qm/src/radial/recipe.rs` (the LFC partial core replaces the full core; T1–T7 run in `candidate`; the record), `crates/qm/src/radial/ledger.rs` (`hash_ledger`, `recipe_hash`, the grep test), `crates/qm/src/radial/mod.rs`, `.gitignore` (`/crates/validate/last-run-generation-records.md`), `crates/validate/references.md` (section "M7 generation records" with the begin/end markers); as built also `crates/qm/src/radial/{atom,kb,partition}.rs` (the confirmation pass, one Dirichlet wall, `Refusal::PartialCore`), `.gitattributes` (references.md LF) and, in review round 1, spec §4.6/§4.8/§4.9/§6.2/§11 (the T5 revision)

**Interfaces:**
- Consumes: Tasks 1–3, in particular `ShellOcc`, `AtomOptions::fixed`, `solve_atom`, `ground_state(z, Polarised)`, `kb_regular`, `solve_kb_atom`, `ValenceConfig`, `reference_config`, `PseudoTable`, `candidate`, `valence_density`, `core_density`, `numerov::{outward, regular_start, g_coeff}`.
- Produces:
```rust
// nlcc.rs
pub struct PartialCore { pub rho_pc: Vec<f64>, pub r_nlc: Option<f64>, pub a: f64, pub b: f64, pub r_core: f64, pub n_core: f64 }
pub fn partial_core(grid: &LogGrid, rho_c: &[f64], rho_v: &[f64]) -> Result<PartialCore, String>; // κ ≥ 1 → Refusal::PartialCore (review N2)
pub fn core_radius(grid: &LogGrid, rho_c: &[f64]) -> f64;                         // r_core (§5.6), 0 without a core
// gates.rs
pub struct TransferRow { pub name: String, pub de_ae: f64, pub de_pp: f64, pub max_eig_error: f64 }
pub struct T7Row { pub config: String, pub spin: u8, pub l: u8, pub top: f64, pub ae: Vec<f64>, pub n_kb: usize }
#[derive(Default)]
pub struct GateReport { pub t1: f64, pub t2: f64, pub t3: Vec<TransferRow>, pub t4: f64, pub t5: Vec<(u8, f64)> /* (l, max δE_equiv on ε_l ± t5_window) */,
                        pub t5_far: Vec<(u8, f64, f64)> /* (l, max on ε_l ± t5_report_window, E − ε_l there): reported */,
                        pub t6: Vec<(u8, u8, f64, f64)> /* (l, l′, AE, PP) slopes */, pub t7: Vec<T7Row>, pub skipped: Vec<String> }
pub fn run_gates(t: &PseudoTable) -> Result<GateReport, String>;
pub fn transfer_configs(t: &PseudoTable, hund: &AtomResult) -> Vec<TransferConfig>;
pub struct TransferConfig { pub name: &'static str, pub ae: AtomOptions, pub pp: ValenceConfig, pub spin: SpinMode }
pub fn delta_e_equiv(t: &PseudoTable, l: u8, e: f64) -> f64;
// record.rs
#[derive(Default)]
pub struct GenerationRecord { pub z: u8, pub recipe_hash: u64, pub f: f64, pub local: String, pub r_c: Vec<(u8, f64)>, pub z_v: f64,
    pub n_core: u32, pub e_kb: Vec<(u8, f64)>, pub gss: Vec<(u8, f64)>, pub r_nlc: Option<f64>, pub r_cmp: f64, pub r_core: f64,
    pub gates: GateReport, pub extra_states: Vec<(u8, f64, f64)> /* the (n+1)s, Task 3 N8 */, pub g2_missing: Vec<(u8, Vec<f64>, Vec<f64>)>,
    pub h_req: Option<f64> /* Task 5 */, pub filter: Vec<FilterRow> /* Task 5 */, pub dimer: String /* Task 11 */, pub seconds: f64 }
pub struct FilterRow { pub h: f64, pub de: f64, pub max_eig: f64, pub t3_max: f64 }   // Task 5 fills it
pub fn record_of(t: &PseudoTable, seconds: f64) -> GenerationRecord;
pub fn records_table(rows: &[(u8, Result<GenerationRecord, Refusal>)]) -> String;
// ledger.rs
pub fn hash_ledger(l: &Ledger) -> u64;   // FNV-1a 64 over format!("{l:?}")
pub fn recipe_hash() -> u64;             // hash_ledger(&LEDGER)
// Gates gains t5_report_window; LEDGER.version "m7-tm-kb-nlcc-2", t5_window 0.5, t5_de_equiv 5e-3 (review round 1)
// partition.rs — Refusal gains PartialCore { z, message }
// recipe.rs — PseudoTable gains:
pub partial_core: PartialCore, pub gates: GateReport, pub record: GenerationRecord,
pub(crate) fn candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, pc: &PartialCore) -> Result<PseudoTable, String>;
pub(crate) fn ghost_checked_candidate(/* same */) -> Result<PseudoTable, String>; // construction + G1/G2 only (tests)
// kb.rs — `lowest` is pub(crate) (T1 on unoccupied channels); solve_kb_atom gets atom's two-pass fallback
```

**As built (Task 4 report and its physics review, round 1, 2026-10-15).** The code text below is the code at the
round-1 commit. Deviations from the original brief, each reviewed:
- **T5 revised by derivation (rule 6, spec §4.8 "Window and threshold"; Spec question 12 stands).** The first run
  (union window [ε_v,min − 1, ε_v,max + 1] at 20 mHa) refused Be, B and Al–Cl at every ladder entry and O at f = 1.6,
  every maximum at the window's top edge in the continuum. The reviewer showed E_PP − E_AE = q·δE_equiv (confined O
  2p, four digits): T5 is now gated per channel on ε_l ± 0.5 Ha (both end points exact, review S1) plus the
  `g2_missing` AE energies of that l, at 5 mHa (= T4, asserted by `t5_threshold_is_the_eigenvalue_tolerance`); the
  ±1 Ha maximum is reported in `GateReport::t5_far`. Version bumped to `m7-tm-kb-nlcc-2`.
- `nlcc`: r_nlc is the outermost point with ρ_c ≥ ρ_v *and ρ_c > 0* (beyond both shooting ends the densities are
  exactly 0, and 0 ≥ 0 would put r_nlc near 60 Bohr); `partial_core` returns `Result` (κ ≥ 1 → `Refusal::PartialCore`,
  review N2); the continuity test reads "ρ_c < ρ_v or both exactly 0" after r_nlc.
- `gates`: T1 covers unoccupied channels through `kb::lowest`; T3's anion arm skips only when each atom is bound or
  `Unbound { after_backtracking: true }` and not both are bound (Task 1 re-review R1-N1), any other error fails;
  configurations the rules name but cannot build are listed in `skipped`; T7 per configuration and spin
  (`T7Row`); T5's `g2_missing` energies are the AE levels beyond the KB root count.
- `atom::solve_atom` and `kb::solve_kb_atom`: an exhausted DIIS fallback is confirmed by a plain linear-mixing pass
  from ρ = 0 (β halved for good) before `Unbound { after_backtracking: true }` (Task 1 re-review R1-S1: Cl
  3p^5.965–5.975 are bound). `kb_count_below` counts n_loc to the Green kernel's own wall (Task 3 re-review R1).
- `recipe`: `ghost_checked_candidate` (construction + G1/G2) is the Task 3 stage; the Task 3 tests whose fixtures
  needed Al or the G1/G2 ladder use it (`al_fixture`, `ghost_ladder`).
- `record`: `extra_states` (the (n+1)s, Task 3 N8) and `g2_missing`; refusal reasons have '|' replaced.
- `pseudo_gen`: ZS = 1..=18 (review N9); `references.md` CRLF is normalised before the comparison and
  `.gitattributes` keeps it LF (N8); the acceptance test asserts Li and Be refused and B–Ne passing against
  `LEDGER.gates` (review M2).

- [ ] **Step 1: Write the failing tests**

`crates/qm/src/radial/gates.rs` tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Spec §8 row 4, first failing test: AE vs KB-PP total-energy differences between the reference
    /// and the +1 ion, the s → p promotion and the spin-polarised Hund multiplet, each within 5 mHa.
    #[test]
    fn oxygen_transfer_energies_within_5mha() {
        let t = crate::radial::generate(8).expect("O generates");
        let names: Vec<&str> = t.gates.t3.iter().map(|r| r.name.as_str()).collect();
        for want in ["cation", "s→p", "Hund"] { assert!(names.contains(&want), "missing {want}: {names:?}"); }
        for row in &t.gates.t3 {
            eprintln!("O {}: ΔE_AE {:.6} ΔE_PP {:.6} (Δ {:+.3} mHa), T4 {:.3} mHa", row.name, row.de_ae, row.de_pp, (row.de_pp - row.de_ae) * 1e3, row.max_eig_error * 1e3);
            assert!((row.de_ae - row.de_pp).abs() < 5e-3, "{}: ΔE_AE {:.6} ΔE_PP {:.6}", row.name, row.de_ae, row.de_pp);
            assert!(row.max_eig_error < 5e-3, "{}: T4 eigenvalue error {:.6}", row.name, row.max_eig_error);
        }
        eprintln!("O skipped: {:?}", t.gates.skipped);
    }

    /// δE_equiv vanishes at the reference eigenvalue (beyond r_c the normalised û_AE and û_PP coincide
    /// there, so their Wronskian is zero; norm conservation then makes the mismatch second order in
    /// E − ε_l, spec §4.8 T5): at ε_l it is far below the 5 mHa gate.
    #[test]
    fn log_derivative_mismatch_vanishes_at_the_reference_energy() {
        let t = crate::radial::generate(8).unwrap();
        for c in t.partition.channels.iter().filter(|c| c.occupied) {
            let d = delta_e_equiv(&t, c.l, c.eps);
            eprintln!("O l = {}: δE_equiv(ε_l = {:.6}) = {d:.3e}", c.l, c.eps);
            assert!(d.abs() < 1e-4, "l = {}: δE_equiv(ε_l) = {d:.3e}", c.l);
        }
    }
}
```
`crates/qm/src/radial/nlcc.rs` tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::{core_density, ground_state, partition, valence_density, SpinMode, LEDGER};

    #[test]
    fn partial_core_is_continuous_with_slope_at_r_nlc() {
        let a = ground_state(8, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let (rc, rv) = (core_density(&a, &p), valence_density(&a, &p));
        let pc = partial_core(&a.grid, &rc, &rv).unwrap();
        let r = pc.r_nlc.expect("O has a core");
        let model = |x: f64| pc.a * (pc.b * x).sin() / x;
        let i = a.grid.nearest(r);
        assert!((model(r) - rc[i]).abs() < 1e-10 * rc[i]);
        let d = a.grid.derivative(&rc)[i];
        let dm = pc.a * (pc.b * (pc.b * r).cos() / r - (pc.b * r).sin() / (r * r));
        assert!((dm - d).abs() < 1e-8 * d.abs(), "slope {dm} vs {d}");
        // r_nlc is the definition — the outermost grid point with ρ_c ≥ ρ_v — checked as such, not as a
        // 5 % closeness (plan review NOTE: |d ln(ρ_c/ρ_v)/d ln r| ≈ 5 near 0.45 Bohr is ≈ 4 % per log-grid step).
        // Where both densities are exactly 0 (beyond the eigenfunctions' shooting end: O, the last 21 points,
        // r ≥ 50.76 Bohr, measured) "ρ_c ≥ ρ_v" holds as 0 ≥ 0 and says nothing; the brief's literal
        // `all(ρ_c < ρ_v)` fails there. The definition is read on points with core density (module doc), and
        // those trailing points must be joint zeros.
        assert!(rc[i] >= LEDGER.nlcc_crossover * rv[i], "ρ_c < ρ_v at r_nlc = {r}");
        assert!((i + 1..a.grid.len()).all(|j| rc[j] < LEDGER.nlcc_crossover * rv[j] || (rc[j] == 0.0 && rv[j] == 0.0)),
                "a later grid point has ρ_c ≥ ρ_v with density present");
        eprintln!("O: r_nlc = {r:.4} Bohr, A = {:.6}, B = {:.6} (x = BR = {:.6}), ρ_pc(0) = AB = {:.4} vs ρ_c(0) = {:.1}; N_c = {:.10}",
                  pc.a, pc.b, pc.b * r, pc.a * pc.b, rc[0], pc.n_core);
    }

    /// The explicit no-core branch: H has ρ_pc ≡ 0, r_nlc undefined, r_core = 0 (spec §4.2 item 4, §5.6).
    #[test]
    fn no_core_means_no_partial_core() {
        let a = ground_state(1, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let pc = partial_core(&a.grid, &core_density(&a, &p), &valence_density(&a, &p)).unwrap();
        assert!(pc.rho_pc.iter().all(|&x| x == 0.0) && pc.r_nlc.is_none() && pc.r_core == 0.0);
    }

    /// r_core is its definition (§5.6): all but ε_core of N_c inside it, not so one grid point earlier.
    #[test]
    fn core_radius_is_its_definition() {
        let a = ground_state(8, SpinMode::Restricted).unwrap();
        let p = partition(&a).unwrap();
        let rc = core_density(&a, &p);
        let g = &a.grid;
        let r_core = core_radius(g, &rc);
        let shell: Vec<f64> = g.r.iter().zip(&rc).map(|(r, x)| 4.0 * std::f64::consts::PI * r * r * x).collect();
        let c = g.cumulative(&shell);
        let n_c = c[g.len() - 1];
        let i = g.nearest(r_core);
        assert!(n_c - c[i] <= LEDGER.eps_core * n_c && n_c - c[i - 1] > LEDGER.eps_core * n_c);
        eprintln!("O: r_core = {r_core:.4} Bohr (spec §5.6 hydrogenic estimate ≈ 0.7)");
    }

    /// Task 2's tail identity re-run with the LFC partial core: beyond max(r_c, r_nlc), ρ_pc = ρ_c, so V_ion + Z_v/r is
    /// the full core's Hartree tail (spec §4.3).
    #[test]
    fn v_ion_tail_is_the_core_hartree_tail_with_lfc() {
        let t = crate::radial::generate(8).unwrap();
        let g = &t.grid;
        let n = g.len();
        let i_from = t.channels.iter().map(|c| c.i_c).max().unwrap().max(g.nearest(t.partial_core.r_nlc.unwrap())) + 1;
        let occ: Vec<f64> = t.partition.channels.iter().map(|c| c.occ).collect();
        let un = crate::radial::tm::unscreen(g, t.z_v, &t.channels, &occ, &t.partial_core.rho_pc);
        let rho_c = crate::radial::core_density(&t.reference, &t.partition);
        let vhc = crate::radial::hartree_potential(g, &rho_c);
        let tail = |i: usize| vhc[i] - t.n_core as f64 / g.r[i];
        for (k, v) in un.v_ion.iter().enumerate() {
            for i in i_from..n {
                let d = v[i] + t.z_v / g.r[i] - tail(i);
                assert!(d.abs() < crate::radial::LEDGER.v_ion_tail / 100.0, "channel {k}, r = {:.4}: V_ion + Z_v/r − τ_c = {d:.3e}", g.r[i]);
            }
        }
        // Task 2's r_cmp checks: it bounds the tail, it is minimal, and it is the core tail's 1e-8 radius (±1 point)
        // when that lies beyond i_from
        let tol = crate::radial::LEDGER.v_ion_tail;
        let dev = |v: &[f64], i: usize| (v[i] + t.z_v / g.r[i]).abs();
        let i_cmp = g.nearest(un.r_cmp);
        assert!((i_cmp..n).all(|i| un.v_ion.iter().all(|v| dev(v, i) < tol)), "r_cmp does not bound the tail");
        assert!(i_cmp > 0 && un.v_ion.iter().any(|v| dev(v, i_cmp - 1) >= tol), "r_cmp is not minimal");
        let i_tail = (0..n).rev().find(|&i| tail(i).abs() >= tol).map_or(0, |i| i + 1);
        if i_tail > i_from {
            assert!(i_cmp.abs_diff(i_tail) <= 1, "r_cmp {} vs the core tail's radius {}", un.r_cmp, g.r[i_tail]);
        } else {
            assert!(i_cmp <= i_from, "r_cmp {} beyond the first point past max(r_c, r_nlc)", un.r_cmp);
        }
        eprintln!("O with LFC: r_nlc {:.4}, first point past max(r_c, r_nlc) {:.4}, r_cmp {:.4}, core tail's 1e-8 radius {:.4}",
                  t.partial_core.r_nlc.unwrap(), g.r[i_from], un.r_cmp, g.r[i_tail]);
    }
}
```
`crates/qm/src/radial/ledger.rs` tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Spec §8 row 4: the ledger of §4.9 is one const table and nothing in `radial/` is indexed by Z.
    /// Grep-style: in every other radial source file's library code (before `#[cfg(test)]`, comments
    /// skipped) (1) no numeric `const`/`static` declaration; (2) no Z dispatch — `match z`, `z` as an
    /// index, or `z` compared with anything — unless the same line reads a `LEDGER.` field (the dimer
    /// gate's Z ≤ LEDGER.dimer_z_max is a universal rule). Task 5 adds "filter.rs" to FILES.
    #[test]
    fn gate_thresholds_are_constants() {
        const FILES: &[(&str, &str)] = &[
            ("mod.rs", include_str!("mod.rs")), ("grid.rs", include_str!("grid.rs")), ("numerov.rs", include_str!("numerov.rs")),
            ("atom.rs", include_str!("atom.rs")), ("partition.rs", include_str!("partition.rs")), ("tm.rs", include_str!("tm.rs")),
            ("kb.rs", include_str!("kb.rs")), ("recipe.rs", include_str!("recipe.rs")), ("nlcc.rs", include_str!("nlcc.rs")),
            ("gates.rs", include_str!("gates.rs")), ("record.rs", include_str!("record.rs")),
        ];
        const NUMERIC: [&str; 11] = [": f64", ": f32", ": u8", ": u16", ": u32", ": u64", ": usize", ": i32", ": i64", ": [f64", ": [u8"];
        let ident = |c: char| c.is_alphanumeric() || c == '_';
        for (name, src) in FILES {
            let lib = src.split("#[cfg(test)]").next().unwrap();
            for (k, line) in lib.lines().enumerate() {
                let t = line.trim_start();
                if t.starts_with("//") { continue; }
                let decl = ["const ", "pub const ", "pub(crate) const ", "static ", "pub static "].iter().any(|p| t.starts_with(p));
                assert!(!(decl && NUMERIC.iter().any(|n| t.contains(n))), "{name}:{}: a numeric constant outside the ledger: {t}", k + 1);
                if t.contains("LEDGER.") { continue; }
                let chars: Vec<char> = t.chars().collect();
                for (i, &c) in chars.iter().enumerate() {
                    if c != 'z' || (i > 0 && ident(chars[i - 1])) || chars.get(i + 1).is_some_and(|&n| ident(n)) { continue; }
                    let before: String = chars[..i].iter().collect();
                    let after: String = chars[i + 1..].iter().collect::<String>().trim_start().to_string();
                    let compared = ["==", "!=", "<", ">"].iter().any(|op| after.starts_with(op))
                        || ["== ", "!= ", "< ", "> ", "<= ", ">= "].iter().any(|op| before.trim_end().ends_with(op.trim_end()));
                    let indexed = before.trim_end().ends_with('[') || after.starts_with("as usize]");
                    let matched = before.trim_end().ends_with("match");
                    assert!(!(compared || indexed || matched), "{name}:{}: a Z dispatch (rule 1): {t}", k + 1);
                }
            }
        }
    }

    #[test]
    fn recipe_hash_tracks_every_switch() {
        let h = recipe_hash();
        assert_ne!(hash_ledger(&Ledger { nlcc: NlccMode::Off, ..LEDGER }), h, "the QMC switch (spec §3) must change the hash");
        assert_ne!(hash_ledger(&Ledger { gamma_r: 1.6, ..LEDGER }), h);
        assert_ne!(hash_ledger(&Ledger { version: "other", ..LEDGER }), h);
        assert_ne!(hash_ledger(&Ledger { f_ladder: [1.6, 1.4, 1.2, 0.9], ..LEDGER }), h);
        assert_eq!(hash_ledger(&LEDGER), h, "stable within a build");
    }

    /// Spec §4.8 T5 (Task 4 physics review, rule-6 revision by derivation): δE_equiv is the first-order
    /// eigenvalue error of a channel-l state wholly inside r_d, so its threshold is T4's eigenvalue tolerance.
    #[test]
    fn t5_threshold_is_the_eigenvalue_tolerance() {
        assert_eq!(LEDGER.gates.t5_de_equiv, LEDGER.gates.t4_eig);
    }
}
```
`crates/qm/src/radial/atom.rs` (Task 1 re-review R1-S1): `near_threshold_states_are_confirmed_by_linear_mixing` — Cl
3p^5.965/5.970/5.975 bound at −6.019/−4.648/−3.280 mHa (2e-6 against the review's probe), 3p^5.99 genuinely unbound.

`crates/validate/tests/pseudo_gen.rs` (new suite; module doc: what it gates, the markers, how to regenerate):
```rust
//! M7 generator gates per element (spec §6.2) and the pinned outputs table (review S6a).
//! Run: `cargo test -p validate --release --test pseudo_gen`. When the records move (a universal
//! constant changed, or Task 5/11/12 add a column or element), the test writes the new block to the
//! gitignored `crates/validate/last-run-generation-records.md` and fails; after physics review, copy it
//! between the markers in `references.md` ("M7 generation records").
//! A moved record must move `recipe_hash` (plan review M6, Global Constraints): when a row present in
//! both blocks differs and the hash line did not change, the failure says "bump LEDGER.version".
use qm::radial::{generate, record::records_table, LEDGER};
use std::collections::BTreeMap;

const BEGIN: &str = "<!-- m7-generation-records:begin -->";
const END: &str = "<!-- m7-generation-records:end -->";
/// Every Z the milestone covers (Task 4 review N9: the bump rule then guards all of them): Li, Na, Mg pinned as
/// refused by the partition, Be and Al by T5 at this recipe version; H and He are provisional until Task 11's
/// dimer gate; Al–Ar are enabled in the app by Task 12.
const ZS: [u8; 18] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

/// The block's first line, "recipe_hash = 0x… (version)".
fn hash_line(block: &str) -> &str { block.lines().next().unwrap_or("").trim() }

/// The table rows keyed by their Z cell (the header and separator rows skipped).
fn rows_by_z(block: &str) -> BTreeMap<String, String> {
    block.lines().filter(|l| l.starts_with("| ") && !l.starts_with("| Z ") && !l.starts_with("|---"))
        .filter_map(|l| l.split('|').nth(1).map(|z| (z.trim().to_string(), l.trim().to_string()))).collect()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn generation_record_table_matches_references_md() {
    let rows: Vec<_> = ZS.iter().map(|&z| (z, generate(z).map(|t| t.record))).collect();
    let table = records_table(&rows);
    // line endings normalised: references.md is checked out with CRLF where git converts (Windows,
    // core.autocrlf), `records_table` writes LF
    let refs = include_str!("../references.md").replace("\r\n", "\n");
    let pinned = refs.split(BEGIN).nth(1).and_then(|s| s.split(END).next()).map(str::trim).unwrap_or("");
    if pinned != table.trim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-generation-records.md");
        std::fs::write(path, &table).expect("write the regenerated block");
        let (old, new) = (rows_by_z(pinned), rows_by_z(&table));
        let moved: Vec<&String> = old.iter().filter(|(z, row)| new.get(*z).is_some_and(|n| n != *row)).map(|(z, _)| z).collect();
        if !pinned.is_empty() && !moved.is_empty() && hash_line(pinned) == hash_line(&table) {
            panic!("the generation records of Z = {moved:?} moved but recipe_hash did not: bump LEDGER.version (plan Global Constraints, plan review M6), then regenerate — {path}:\n{table}");
        }
        panic!("the generation records differ from references.md — review, then copy {path} between the markers:\n{table}");
    }
}

/// Spec §6.2 (as revised by the Task 4 physics review): every gate on B–Ne (T1–T7 inside `generate`; here
/// their numbers are printed and the acceptance asserted against `LEDGER.gates`, never against literals); Li
/// refused by the partition (ONCV), Be refused by T5 at every ladder entry (pinned, like Li).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn b_to_ne_pass_every_gate_li_and_be_are_refused() {
    assert!(generate(3).is_err_and(|e| e.to_string().contains("ONCV")));
    let be = generate(4).expect_err("Be is refused at this recipe version");
    eprintln!("Z = 4: {be}");
    assert!(be.to_string().contains("T5"), "Be's refusal names T5: {be}");
    let gates = &LEDGER.gates;
    for z in 5u8..=10 {
        let t = generate(z).unwrap_or_else(|e| panic!("Z = {z} refused: {e}"));
        let g = &t.gates;
        eprintln!("Z = {z}: f {} {:?} T1 {:.2e} T2 {:.1e} T3 {:?} T4 {:.2e} T5 {:?} T5 far {:?} T6 {:?}", t.f, t.local, g.t1, g.t2,
                  g.t3.iter().map(|r| (r.name.clone(), r.de_ae - r.de_pp)).collect::<Vec<_>>(), g.t4, g.t5, g.t5_far, g.t6);
        assert!(g.t1 < gates.t1_eig && g.t2 < gates.t2_norm && g.t4 < gates.t4_eig);
        assert!(g.t3.iter().all(|r| (r.de_ae - r.de_pp).abs() < gates.t3_delta_e));
        assert!(g.t5.iter().all(|(_, d)| *d < gates.t5_de_equiv));
        assert!(g.t6.iter().all(|(_, _, a, p)| (a - p).abs() < gates.t6_hardness));
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release radial` and `cargo test -p validate --release --test pseudo_gen`
Expected: compile errors (`nlcc`, `gates`, `record` missing, `PseudoTable.gates` missing).

- [ ] **Step 3: Implement NLCC and r_core**

`crates/qm/src/radial/nlcc.rs` (module doc: spec §4.5 verbatim plus the derivation of x cot x = κ, and r_core):
```rust
use super::grid::LogGrid;
use super::ledger::{NlccMode, LEDGER};
use std::f64::consts::PI;

/// The LFC partial core: ρ_pc on the grid, r_nlc (None without a core), the inner model's A and B, plus
/// the core radius r_core (§5.6) and the core charge N_c of the AE core density it was built from.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialCore { pub rho_pc: Vec<f64>, pub r_nlc: Option<f64>, pub a: f64, pub b: f64, pub r_core: f64, pub n_core: f64 }

fn shell(grid: &LogGrid, rho: &[f64]) -> Vec<f64> { grid.r.iter().zip(rho).map(|(r, x)| 4.0 * PI * r * r * x).collect() }

/// §5.6: the smallest grid r with ∫_r^∞ 4πr²ρ_c ≤ LEDGER.eps_core·N_c; 0 without a core.
pub fn core_radius(grid: &LogGrid, rho_c: &[f64]) -> f64 {
    let c = grid.cumulative(&shell(grid, rho_c));
    let n_c = c[grid.len() - 1];
    if n_c <= 0.0 { return 0.0; }
    let i = (0..grid.len()).find(|&i| n_c - c[i] <= LEDGER.eps_core * n_c).expect("the cumulative reaches N_c");
    grid.r[i]
}

/// LFC (module doc): ρ_pc = ρ_c for r ≥ r_nlc, A sin(Br)/r inside, A and B from value and slope continuity
/// (x cot x = κ). `rho_v` is the all-electron reference's valence density (f-independent). No core or
/// `NlccMode::Off`: ρ_pc ≡ 0 and r_nlc = None. An error (κ ≥ 1, no root: ρ_c not falling at r_nlc) is
/// refused by the recipe (`Refusal::PartialCore`, Task 4 review N2), not a panic.
pub fn partial_core(grid: &LogGrid, rho_c: &[f64], rho_v: &[f64]) -> Result<PartialCore, String> {
    let n = grid.len();
    let n_core = grid.integrate(&shell(grid, rho_c));
    let r_core = core_radius(grid, rho_c);
    if n_core <= 0.0 || LEDGER.nlcc == NlccMode::Off {
        return Ok(PartialCore { rho_pc: vec![0.0; n], r_nlc: None, a: 0.0, b: 0.0, r_core, n_core });
    }
    let i = (0..n).rev().find(|&i| rho_c[i] > 0.0 && rho_c[i] >= LEDGER.nlcc_crossover * rho_v[i])
        .ok_or("the core never dominates the valence density")?;
    let r = grid.r[i];
    let kappa = 1.0 + r * grid.derivative(rho_c)[i] / rho_c[i];
    if kappa >= 1.0 { return Err(format!("ρ_c is not falling at r_nlc = {r} (κ = {kappa}): x cot x = κ has no root in (0, π)")); }
    let (mut lo, mut hi) = (1e-12, PI - 1e-12);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if m / m.tan() > kappa { lo = m } else { hi = m }
    }
    let x = 0.5 * (lo + hi);
    let (b, a) = (x / r, rho_c[i] * r / x.sin());
    let rho_pc = (0..n).map(|j| if j < i { a * (b * grid.r[j]).sin() / grid.r[j] } else { rho_c[j] }).collect();
    Ok(PartialCore { rho_pc, r_nlc: Some(r), a, b, r_core, n_core })
}
```

- [ ] **Step 4: Implement the gates**

`crates/qm/src/radial/gates.rs`. Its module doc carries:
- spec §4.8's table;
- S4's Wronskian derivation of ∂D/∂E = −(2/u(r_d)²)∫₀^{r_d}u² dr, and from it the form used here (Spec question 12, as the plan review ruled): **δE_equiv(E) = ½·W(û_AE, û_PP)(r_d) = ½[û_AE û′_PP − û′_AE û_PP](r_d)**, each regular solution normalised to ∫₀^{r_d} û² dr = 1. It is pole-free (û finite, the norms positive). It equals ½(D^PP − D^AE)·√(w_AE w_PP) with w = u(r_d)²/∫₀^{r_d}u², i.e. the spec's quantity times 1 + O(δw), so the two agree to third order at ε_l. Its perturbative meaning is ⟨û_AE|V_PP − V_AE|û_PP⟩ on [0, r_d]; the KB terms cancel beyond β's support. |δE_equiv| does not depend on the sign of either û (W is bilinear). The rejected difference form ½(D_PP w_PP − D_AE w_AE) is not first-order equal: the Wronskian identity gives D′w = −2 for each potential at every E, so its slope at ε_l is ½D(w′_PP − w′_AE) with w′ = 2D″/D′², and norm conservation matches D and D′ there, not D″;
- T5's window and threshold as revised (spec §4.8 "Window and threshold"): per channel ε_l ± 0.5 Ha plus the
  `g2_missing` energies, 5 mHa = T4; ±1 Ha reported;
- the configuration rules: cation = one electron out of the highest valence shell; s→p = one electron from the valence s to the valence p, when both channels exist; anion = one electron into the lowest shell with room, kept only if bound in both atoms; Hund = the spin-polarised `ground_state` multiplet. All but Hund are spin-restricted, like the reference.

```rust
use super::atom::{ground_state, solve_atom, AtomOptions, AtomResult, RadialError, ShellOcc, SpinMode};
use super::kb::{kb_count_below, kb_regular, lowest, reference_config, solve_kb_atom, KbAtom, ValenceConfig};
use super::ledger::LEDGER;
use super::numerov::{bound_state, g_coeff, outward, regular_start};
use super::recipe::PseudoTable;

/// One T3 configuration: ΔE = E(config) − E(reference) in each atom, and T4's worst valence eigenvalue error.
#[derive(Clone, Debug, PartialEq)]
pub struct TransferRow { pub name: String, pub de_ae: f64, pub de_pp: f64, pub max_eig_error: f64 }

/// T7's count in one T3 configuration, spin and l: the AE bound levels above the core below `top` and the
/// KB inertia count there (n_kb ≤ ae.len(), else refused; fewer is recorded, not a ghost).
#[derive(Clone, Debug, PartialEq)]
pub struct T7Row { pub config: String, pub spin: u8, pub l: u8, pub top: f64, pub ae: Vec<f64>, pub n_kb: usize }

/// The gate numbers: T1 worst |Δε| (Ha), T2 worst norm difference, T3 rows, T4 worst |Δε| over the T3
/// configurations, T5 (l, max |δE_equiv|) over the gated window, `t5_far` (l, max |δE_equiv| over ε_l ±
/// `t5_report_window`, E − ε_l at that max; reported only), T6 (l, l′, AE slope, PP slope) in Ha/e, T7 rows, and what was
/// skipped with its reason (Default: the empty report of a table that has not been through the gates —
/// `recipe::ghost_checked_candidate`, tests only).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GateReport {
    pub t1: f64, pub t2: f64, pub t3: Vec<TransferRow>, pub t4: f64, pub t5: Vec<(u8, f64)>, pub t5_far: Vec<(u8, f64, f64)>,
    pub t6: Vec<(u8, u8, f64, f64)>, pub t7: Vec<T7Row>, pub skipped: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TransferConfig { pub name: &'static str, pub ae: AtomOptions, pub pp: ValenceConfig, pub spin: SpinMode }

/// AE shells: the core full in both spins plus the valence channels' (n, l) at `val[l]`.
fn ae_shells(t: &PseudoTable, val: &ValenceConfig) -> Vec<ShellOcc> {
    let mut v: Vec<ShellOcc> = t.partition.core.iter().map(|s| ShellOcc { n: s.n, l: s.l, occ: [0.5 * s.occ, 0.5 * s.occ] }).collect();
    for (l, occ) in val.occ.iter().enumerate() {
        if occ[0] + occ[1] <= 0.0 { continue; }
        let n = t.partition.channels.iter().find(|c| c.l as usize == l).expect("a channel for every occupied l").n;
        v.push(ShellOcc { n, l: l as u8, occ: *occ });
    }
    v
}

fn moved(base: &ValenceConfig, from: Option<usize>, to: Option<usize>) -> ValenceConfig {
    let mut c = base.clone();
    if let Some(l) = from { c.occ[l][0] -= 0.5; c.occ[l][1] -= 0.5; }
    if let Some(l) = to { c.occ[l][0] += 0.5; c.occ[l][1] += 0.5; }
    c
}

fn cap(l: usize) -> f64 { 2.0 * (2 * l + 1) as f64 }

fn total(c: &ValenceConfig, l: usize) -> f64 { c.occ[l][0] + c.occ[l][1] }

/// The T3 configurations (module doc), in the order cation, s→p, anion, Hund; `hund` is the AE
/// spin-polarised ground state, whose valence occupations per spin the PP's Hund configuration takes.
pub fn transfer_configs(t: &PseudoTable, hund: &AtomResult) -> Vec<TransferConfig> {
    let base = reference_config(&t.partition);
    let chans = &t.partition.channels;
    let homo = chans.iter().filter(|c| c.occupied).max_by(|a, b| a.eps.total_cmp(&b.eps)).expect("valence").l as usize;
    let restricted = |name, pp: ValenceConfig| TransferConfig { name, ae: AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &pp)), pp, spin: SpinMode::Restricted };
    let mut out = vec![restricted("cation", moved(&base, Some(homo), None))];
    let has = |l: usize| chans.iter().any(|c| c.l as usize == l);
    if has(0) && has(1) && total(&base, 0) >= 1.0 && total(&base, 1) <= cap(1) - 1.0 {
        out.push(restricted("s→p", moved(&base, Some(0), Some(1))));
    }
    let eps_of = |l: usize| chans.iter().find(|c| c.l as usize == l).expect("a channel").eps;
    let room = (0..base.occ.len()).filter(|&l| has(l) && total(&base, l) < cap(l)).min_by(|&a, &b| eps_of(a).total_cmp(&eps_of(b)));
    if let Some(l) = room { out.push(restricted("anion", moved(&base, None, Some(l)))); }
    // Hund: the AE spin-polarised multiplet and the same valence occupations per spin for the PP
    let mut pp = ValenceConfig { occ: vec![[0.0; 2]; base.occ.len()] };
    for o in hund.orbitals.iter().filter(|o| t.partition.valence.iter().any(|v| v.n == o.n && v.l == o.l)) {
        pp.occ[o.l as usize] = o.occ;
    }
    out.push(TransferConfig { name: "Hund", ae: AtomOptions::aufbau(SpinMode::Polarised, hund.multiplicity), pp, spin: SpinMode::Polarised });
    out
}

/// Why a configuration the rules name was not built (rule 7: visible, never silent).
fn not_built(t: &PseudoTable, built: &[TransferConfig]) -> Vec<String> {
    let base = reference_config(&t.partition);
    let has = |l: usize| t.partition.channels.iter().any(|c| c.l as usize == l);
    let mut out = Vec::new();
    if !built.iter().any(|c| c.name == "s→p") {
        out.push(if !has(0) || !has(1) { "s→p: no s or no p channel".to_string() }
                 else if total(&base, 0) < 1.0 { "s→p: the valence s holds no electron".to_string() }
                 else { "s→p: the valence p shell is full".to_string() });
    }
    if !built.iter().any(|c| c.name == "anion") { out.push("anion: no channel shell has room".to_string()); }
    out
}

/// The AE regular solution at e in the reference potential, u-form, to i_end.
fn ae_regular(t: &PseudoTable, l: u8, e: f64, i_end: usize) -> Vec<f64> {
    let g = &t.grid;
    let gc = g_coeff(g, &t.reference.v[0], l, e);
    let (y0, y1) = regular_start(g, l, t.z as f64);
    outward(&gc, y0, y1, i_end, g.delta).0.iter().enumerate().map(|(i, y)| y * g.r[i].sqrt()).collect()
}

/// δE_equiv(e) = ½·W(û_AE, û_PP)(r_d) = ½[û_AE û′_PP − û′_AE û_PP](r_d), each regular solution
/// normalised to ∫₀^{r_d} û² dr = 1, at r_d = LEDGER.gates.t5_rd_factor · max r_c (module doc;
/// Spec question 12 as the plan review ruled).
pub fn delta_e_equiv(t: &PseudoTable, l: u8, e: f64) -> f64 {
    let g = &t.grid;
    let r_d = LEDGER.gates.t5_rd_factor * t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max);
    let i_d = g.nearest(r_d);
    let v_pp: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
    let proj = t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
    // (û, û′) at r_d
    let at_rd = |u: &[f64]| {
        let norm = g.partial(&u.iter().map(|x| x * x).chain(std::iter::repeat(0.0)).take(g.len()).collect::<Vec<_>>(), i_d).sqrt();
        let du = (u[i_d - 2] - 8.0 * u[i_d - 1] + 8.0 * u[i_d + 1] - u[i_d + 2]) / (12.0 * g.delta * g.r[i_d]);
        (u[i_d] / norm, du / norm)
    };
    let (ae, dae) = at_rd(&ae_regular(t, l, e, i_d + 2));
    let (pp, dpp) = at_rd(&kb_regular(g, &v_pp, proj, l, e, i_d + 2));
    0.5 * (ae * dpp - dae * pp)
}

/// max |δE_equiv| over [eps − half, eps + half] on the `t5_mesh` grid with both end points exact (review S1),
/// and the energy where it occurs.
fn t5_scan(t: &PseudoTable, l: u8, eps: f64, half: f64) -> (f64, f64) {
    let mesh = LEDGER.gates.t5_mesh;
    let (lo, hi) = (eps - half, eps + half);
    let n = ((hi - lo) / mesh).ceil() as usize;
    (0..=n).map(|k| if k == n { hi } else { (lo + k as f64 * mesh).min(hi) })
        .map(|e| (delta_e_equiv(t, l, e).abs(), e)).fold((0.0, eps), |m, x| if x.0 > m.0 { x } else { m })
}

/// T7's count in one configuration (module doc): per spin and l ≤ l_max, n_KB(top) against the AE levels
/// above the core below top.
fn t7_counts(t: &PseudoTable, name: &str, ae: &AtomResult, pp: &KbAtom, spin: SpinMode) -> Result<Vec<T7Row>, String> {
    let g = &t.grid;
    let part = &t.partition;
    let valence = |o: &&super::atom::Orbital| part.valence.iter().any(|v| v.n == o.n && v.l == o.l);
    let eps_v_max = ae.orbitals.iter().filter(valence)
        .flat_map(|o| (0..2).filter(|&s| o.occ[s] > 0.0).map(|s| o.eps[s]).collect::<Vec<_>>())
        .fold(f64::NEG_INFINITY, f64::max);
    let top = (eps_v_max + LEDGER.gates.g2_window).min(0.0);
    let spins: &[usize] = if spin == SpinMode::Restricted { &[0] } else { &[0, 1] };
    let mut rows = Vec::new();
    for &s in spins {
        for l in 0..=part.l_max {
            let n_core_l = part.core.iter().filter(|c| c.l == l).count();
            let levels: Vec<f64> = (n_core_l..).map_while(|k| bound_state(g, &ae.v[s], l, k, t.z as f64, None))
                .map(|b| b.e).take_while(|&e| e < top).collect();
            let proj = t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
            let n_kb = kb_count_below(g, &pp.v[s], proj, l, top);
            if n_kb > levels.len() {
                return Err(format!("T7 {name} spin {s}, l = {l}: {n_kb} KB bound states below {top:.4} Ha, the all-electron atom {} ({levels:?}): an extra state (ghost)", levels.len()));
            }
            rows.push(T7Row { config: name.to_string(), spin: s as u8, l, top, ae: levels, n_kb });
        }
    }
    Ok(rows)
}

fn genuinely_unbound<T>(r: &Result<T, RadialError>) -> bool {
    matches!(r, Err(RadialError::Unbound { after_backtracking: true, .. }))
}

/// T1–T7 on a candidate that passed G1/G2 (module doc); the first miss refuses it with the gate's name.
pub fn run_gates(t: &PseudoTable) -> Result<GateReport, String> {
    let gates = &LEDGER.gates;
    let g = &t.grid;
    let input = t.kb.input();
    let base = reference_config(&t.partition);
    let proj_of = |l: u8| t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
    let pp_ref = solve_kb_atom(g, &input, &base, SpinMode::Restricted).map_err(|e| format!("reference KB atom: {e}"))?;
    // T1 (a regression: the reference is an exact eigenstate of the KB Hamiltonian), every channel with a
    // reference state — an unoccupied one through the lowest KB level of its l in the reference potential
    let mut t1 = 0.0f64;
    for c in &t.partition.channels {
        let e = if c.occupied { pp_ref.eps[c.l as usize][0] } else {
            lowest(g, &pp_ref.v[0], proj_of(c.l), c.l, f64::NAN).ok_or_else(|| format!("T1: the unoccupied l = {} reference state is not bound by the KB atom", c.l))?
        };
        t1 = t1.max((e - c.eps).abs());
    }
    if t1 >= gates.t1_eig { return Err(format!("T1: {:.3} mHa", t1 * 1e3)); }
    // T2 (norm conservation, TM's construction, as a regression)
    let t2 = t.channels.iter().zip(&t.partition.channels).map(|(tm, c)| {
        let sq = |u: &[f64]| u.iter().map(|x| x * x).collect::<Vec<_>>();
        (g.partial(&sq(&tm.u), tm.i_c) - g.partial(&sq(&c.u), tm.i_c)).abs()
    }).fold(0.0, f64::max);
    if t2 >= gates.t2_norm { return Err(format!("T2: norm error {t2:.2e}")); }
    // T3 / T4 / T7
    let hund = ground_state(t.z, SpinMode::Polarised).map_err(|e| format!("Hund atom: {e}"))?;
    let (mut t3, mut t4, mut t7) = (Vec::new(), 0.0f64, Vec::new());
    let configs = transfer_configs(t, &hund);
    let mut skipped = not_built(t, &configs);
    for cfg in &configs {
        if cfg.name == "Hund" {
            // the polarised ground state must be the partition's core plus valence (true for Z ≤ 18)
            let n_v: f64 = cfg.pp.occ.iter().map(|o| o[0] + o[1]).sum();
            if (n_v - t.z_v).abs() > 1e-9 {
                return Err(format!("T3 Hund: the polarised ground state holds {n_v} electrons in the partition's valence shells, Z_v = {}", t.z_v));
            }
        }
        let ae = if cfg.name == "Hund" { Ok(hund.clone()) } else { solve_atom(t.z, &cfg.ae) };
        let pp = solve_kb_atom(g, &input, &cfg.pp, cfg.spin);
        let (ae, pp) = match (ae, pp) {
            (Ok(ae), Ok(pp)) => (ae, pp),
            // not both bound (the arm above), and each atom either bound or genuinely unbound (R1-N1)
            (ae, pp) if cfg.name == "anion" && (ae.is_ok() || genuinely_unbound(&ae)) && (pp.is_ok() || genuinely_unbound(&pp)) => {
                let say = |r: Result<String, RadialError>| r.unwrap_or_else(|e| e.to_string());
                skipped.push(format!("anion: not bound in both — LSDA (no −1/r tail, self-interaction); all-electron: {}; pseudo: {}",
                                     say(ae.map(|a| format!("bound, E = {:.6} Ha", a.energy.total))),
                                     say(pp.map(|p| format!("bound, E = {:.6} Ha", p.energy)))));
                continue;
            }
            (ae, pp) => return Err(format!("T3 {}: all-electron {:?} / pseudo {:?}", cfg.name, ae.err(), pp.err())),
        };
        let de_ae = ae.energy.total - t.reference.energy.total;
        let de_pp = pp.energy - pp_ref.energy;
        let mut worst = 0.0f64;
        for o in ae.orbitals.iter().filter(|o| t.partition.valence.iter().any(|v| v.n == o.n && v.l == o.l)) {
            for s in 0..2 { if o.occ[s] > 0.0 { worst = worst.max((o.eps[s] - pp.eps[o.l as usize][s]).abs()); } }
        }
        if (de_ae - de_pp).abs() >= gates.t3_delta_e { return Err(format!("T3 {}: ΔE_AE {de_ae:.6} vs ΔE_PP {de_pp:.6}", cfg.name)); }
        if worst >= gates.t4_eig { return Err(format!("T4 {}: eigenvalue error {:.3} mHa", cfg.name, worst * 1e3)); }
        t4 = t4.max(worst);
        t3.push(TransferRow { name: cfg.name.to_string(), de_ae, de_pp, max_eig_error: worst });
        t7.extend(t7_counts(t, cfg.name, &ae, &pp, cfg.spin)?);
    }
    // T5: projector channels and the local channel (never an ungenerated l), each on ε_l ± t5_window (both end
    // points exact) plus the AE states of that l the KB atom does not bind; ε_l ± t5_report_window reported
    let (mut t5, mut t5_far) = (Vec::new(), Vec::new());
    for c in &t.partition.channels {
        let missing: Vec<f64> = t.ghost.g2_missing.iter().filter(|(l, _, _)| *l == c.l)
            .flat_map(|(_, ae, roots)| ae.iter().skip(roots.len()).copied().collect::<Vec<_>>()).collect();
        let worst = t5_scan(t, c.l, c.eps, gates.t5_window).0.max(missing.iter().map(|&e| delta_e_equiv(t, c.l, e).abs()).fold(0.0, f64::max));
        if worst >= gates.t5_de_equiv { return Err(format!("T5 l = {}: max δE_equiv {:.2} mHa on ε_l ± {} Ha", c.l, worst * 1e3, gates.t5_window)); }
        t5.push((c.l, worst));
        let (far, at) = t5_scan(t, c.l, c.eps, gates.t5_report_window);
        t5_far.push((c.l, far, at - c.eps));
    }
    // T6: ∂ε_l/∂f_l′ by ±t6_dq (restricted), AE vs PP
    let mut t6 = Vec::new();
    let occupied: Vec<u8> = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.l).collect();
    for &lp in &occupied {
        let shifted = |dq: f64| { let mut c = base.clone(); c.occ[lp as usize][0] += 0.5 * dq; c.occ[lp as usize][1] += 0.5 * dq; c };
        // a full shell cannot take +dq (Pauli): one-sided difference over [f − 2dq, f] there
        let full = total(&base, lp as usize) + gates.t6_dq > cap(lp as usize) + 1e-12;
        let (hi, lo) = if full { (0.0, -2.0 * gates.t6_dq) } else { (gates.t6_dq, -gates.t6_dq) };
        let (cp, cm) = (shifted(hi), shifted(lo));
        let ae_p = solve_atom(t.z, &AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &cp))).map_err(|e| format!("T6: {e}"))?;
        let ae_m = solve_atom(t.z, &AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &cm))).map_err(|e| format!("T6: {e}"))?;
        let pp_p = solve_kb_atom(g, &input, &cp, SpinMode::Restricted).map_err(|e| format!("T6: {e}"))?;
        let pp_m = solve_kb_atom(g, &input, &cm, SpinMode::Restricted).map_err(|e| format!("T6: {e}"))?;
        for &l in &occupied {
            let n = t.partition.channels.iter().find(|c| c.l == l).expect("a channel").n;
            let ae_eps = |a: &AtomResult| a.orbitals.iter().find(|o| o.n == n && o.l == l).expect("occupied").eps[0];
            let slope_ae = (ae_eps(&ae_p) - ae_eps(&ae_m)) / (hi - lo);
            let slope_pp = (pp_p.eps[l as usize][0] - pp_m.eps[l as usize][0]) / (hi - lo);
            if (slope_ae - slope_pp).abs() >= gates.t6_hardness { return Err(format!("T6 ∂ε_{l}/∂f_{lp}: AE {slope_ae:.5} PP {slope_pp:.5}")); }
            t6.push((l, lp, slope_ae, slope_pp));
        }
    }
    Ok(GateReport { t1, t2, t3, t4, t5, t5_far, t6, t7, skipped })
}
```

- [ ] **Step 5: Implement the record, the hash, and the ladder changes**

`crates/qm/src/radial/record.rs`:
```rust
use super::gates::GateReport;
use super::ledger::{recipe_hash, LEDGER};
use super::partition::{l_name, Refusal};
use super::recipe::PseudoTable;

/// One row of Task 5's filter ladder: h, the filtered radial atom's total-energy error, its worst
/// eigenvalue error and the worst T3 difference (Ha).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FilterRow { pub h: f64, pub de: f64, pub max_eig: f64, pub t3_max: f64 }

/// The record of one generated element. `extra_states`: (l, AE ε, KB root) for every bound state of the
/// reference beyond the channel's own below the G2 window top — the weak (n+1)s the short-range neutral
/// LSDA well binds (an LSDA artefact of the self-interaction-truncated potential; the real atom has a full
/// Rydberg series, far deeper — Task 3 review N8). `g2_missing`: AE states the pseudo-atom does not bind
/// (Task 3 review S1). `seconds`: the whole `generate` call, reported only.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationRecord {
    pub z: u8, pub recipe_hash: u64, pub f: f64, pub local: String, pub r_c: Vec<(u8, f64)>, pub z_v: f64, pub n_core: u32,
    pub e_kb: Vec<(u8, f64)>, pub gss: Vec<(u8, f64)>, pub r_nlc: Option<f64>, pub r_cmp: f64, pub r_core: f64,
    pub gates: GateReport, pub extra_states: Vec<(u8, f64, f64)>, pub g2_missing: Vec<(u8, Vec<f64>, Vec<f64>)>,
    pub h_req: Option<f64>, pub filter: Vec<FilterRow>, pub dimer: String, pub seconds: f64,
}

pub fn record_of(t: &PseudoTable, seconds: f64) -> GenerationRecord {
    let top = (t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max)
        + LEDGER.gates.g2_window).min(0.0);
    let mut extra_states = Vec::new();
    for (l, roots) in &t.ghost.g2_roots {
        let ae: Vec<f64> = t.partition.spectrum[*l as usize].iter().copied().filter(|&e| e < top).collect();
        let own = usize::from(t.partition.channels.iter().any(|c| c.l == *l));
        extra_states.extend(ae.iter().zip(roots).skip(own).map(|(a, k)| (*l, *a, *k)));
    }
    GenerationRecord {
        z: t.z, recipe_hash: recipe_hash(), f: t.f,
        local: match t.kb.local_l { Some(l) => format!("{}-local", l_name(l)), None => "polynomial".into() },
        r_c: t.channels.iter().map(|c| (c.l, c.r_c)).collect(), z_v: t.z_v, n_core: t.n_core,
        e_kb: t.kb.projectors.iter().map(|p| (p.l, p.e_kb)).collect(), gss: t.kb.projectors.iter().map(|p| (p.l, p.gss)).collect(),
        r_nlc: t.partial_core.r_nlc, r_cmp: t.kb.r_cmp, r_core: t.partial_core.r_core,
        gates: t.gates.clone(), extra_states, g2_missing: t.ghost.g2_missing.clone(), h_req: None, filter: Vec::new(), seconds,
        // Task 11's dimer_verdict overwrites this for Z ≤ dimer_z_max (pass / fail / not applicable / provisional)
        dimer: if t.z <= LEDGER.dimer_z_max { "not run".into() } else { "not applicable (Z above the dimer-gate bound)".into() },
    }
}

/// The pinned outputs table (review S6a): the hash line, then one row per Z; refusals name their reason.
/// No timings (reported elsewhere, never pinned).
pub fn records_table(rows: &[(u8, Result<GenerationRecord, Refusal>)]) -> String {
    let list = |v: &[(u8, f64)], d: usize| v.iter().map(|(l, x)| format!("{}: {x:.d$}", l_name(*l))).collect::<Vec<_>>().join(", ");
    let mut s = format!("recipe_hash = 0x{:016x} ({})\n\n", recipe_hash(), LEDGER.version);
    s += "| Z | el | f | local | r_c (Bohr) | Z_v | E_KB (Ha) | GSS (Ha) | r_nlc | r_core | h_req |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (z, r) in rows {
        let el = units::elements::symbol(*z);
        match r {
            Ok(g) => s += &format!("| {z} | {el} | {:.1} | {} | {} | {:.0} | {} | {} | {} | {:.3} | {} |\n",
                g.f, g.local, list(&g.r_c, 3), g.z_v, list(&g.e_kb, 4), list(&g.gss, 4),
                g.r_nlc.map_or("—".into(), |r| format!("{r:.3}")), g.r_core, g.h_req.map_or("—".into(), |h| format!("{h:.2}"))),
            // a refusal's reason in one cell: a '|' in it would split the row
            Err(e) => s += &format!("| {z} | {el} | refused | {} | | | | | | | |\n", e.to_string().replace('|', "¦")),
        }
    }
    s
}
```
In `ledger.rs`:
```rust
/// FNV-1a 64 over the table's `Debug` rendering (f64 `Debug` is the shortest round-trip decimal, so
/// the rendering — and the hash — is stable across builds and platforms). Every field, `version` and
/// `xc` included, enters the rendering.
pub fn hash_ledger(l: &Ledger) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in format!("{l:?}").bytes() { h ^= b as u64; h = h.wrapping_mul(0x0100_0000_01b3); }
    h
}

/// The recipe's identity (spec §4.9): one hash, one potential per element.
pub fn recipe_hash() -> u64 { hash_ledger(&LEDGER) }
```
In `recipe.rs`:
- `generate` times the whole generation (`std::time::Instant`), builds the partial core once (f-independent) with
  `partial_core(&reference.grid, &core_density(&reference, &part), &valence_density(&reference, &part))`
  (`Err` → `Refusal::PartialCore`), passes `&pc` to `candidate`, and fills `record = record_of(&t, seconds)` on the
  accepted entry.
- `candidate(…, pc)` = `ghost_checked_candidate(…, pc)?` (TM, `unscreen` and `kb_form` with `pc.rho_pc`; τ_c stays
  `core_tail` of the *full* core, Task 2 review S3, never of ρ_pc; `ghost_gates`), then `t.gates = run_gates(&t)?`.
- `kb_form`'s derived support r_β (Task 3 review ruling (b)) moves out to r_nlc with the LFC core by itself — no new
  constant (measured: r_nlc < max r_c for every accepted element).
- T7 (spec §4.8): G1, G2 of §4.6; G2's count also in the KB atom of every T3 configuration (per spin), against that
  configuration's AE atom's bound spectrum above the core. For E_KB < 0 the GSS margin ε₀^loc − ε_l is at most
  1/|E_KB| (variational; Task 3 review), so a near-singular negative E_KB that passes at the reference (polynomial Si
  f = 1.4 p: E_KB −306, margin 0.70 mHa) can grow a ghost under a transfer configuration's screening. T4 need not see
  it, because the ghost sits within mHa of the valence.
- T1 covers every channel with a reference state, occupied or not (Task 3 review N4).

Add `pub mod nlcc; pub mod gates; pub mod record;` and the re-exports to `mod.rs`. Add
`/crates/validate/last-run-generation-records.md` to `.gitignore`.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p qm --release radial`
Expected: PASS, 0 ignored.

Run: `cargo test -p validate --release --test pseudo_gen`
Expected: `b_to_ne_pass_every_gate_li_and_be_are_refused` PASS. `generation_record_table_matches_references_md` FAILS on
the first run (no block yet) and writes `last-run-generation-records.md`.

If an element misses a gate at every ladder entry and is refused, or O fails T5 at f = 1.6, follow the spec §4.8 last
paragraph: **report and have the physics reviewer re-examine**. Never adjust a threshold. (This happened at the first
run; the review revised T5 by derivation — see "As built" above.)

- [ ] **Step 7: Pin the records.** After physics review of the printed gate numbers:
  1. Add to `crates/validate/references.md` a section "M7 generation records (Task 4; regenerated by `pseudo_gen`)". It contains the begin/end markers with the content of `last-run-generation-records.md` between them, and one paragraph comparing f against the literature TM radii (spec §4.4: C ≈ 1.2, N ≈ 1.45, O ≈ 1.6 — "a check, not an input").
  2. Re-run `cargo test -p validate --release --test pseudo_gen`. Expected: PASS.

- [ ] **Step 8: Commit**
```bash
git add crates/qm/src/radial crates/validate/tests/pseudo_gen.rs crates/validate/references.md .gitignore
git commit -o -m "feat(qm,validate): NLCC, transferability gates T1–T7, generation record and recipe_hash (M7 Task 4)" -- crates/qm/src/radial crates/validate/tests/pseudo_gen.rs crates/validate/references.md .gitignore
```
Before merging, hand the diff to `physics-reviewer` with the full gate printout, Spec question 12 and the record table.

---

### Task 5: `filter` — the spherical-Bessel filter with King-Smith band completion, the one V_SR table, h_req and the filtered re-run **[Fable — physics-review gate]**

Spec §5.1, §4.8 T8, §8 row 5; review S2, S3, S7, M4. **Spec questions 2, 3 and 11 apply; the plan review settled 2 and 3 (its M2, M3) and found the first test blind to the first reciprocal shell (its M4).**
- Q2/M2: the spec's V_cmp = V_loc + Z_v/d and V_tail = −Z_v erfc(d/a)/d are each singular at the origin; only their sum V_SR is smooth. The plan's earlier switched tail removed the singularity but left the switch's high-q content on the grid, unfiltered (measured 1e-4–1e-5 of FT(0) at 2π/h, far above the 1e-8 standard). The ruling: **one table V_SR = V_loc + Z_v·erf(d/a)/d**, smooth and finite at 0, filtered like every other table, and truncated beyond R_SR = max(r_cmp, r_floor), where |V_SR| ≤ `v_tail_floor` already.
- Q3/M3: a hard cut at q_c followed by a ψ-step through the cut's Gibbs ringing re-creates high-q content (measured 10–2000× above the 1e-8 judge). The ruling: **King-Smith, Payne & Lin's band completion**. Keep F̂(q) for q ≤ q_c. Choose F̂ on (q_c, q_max], q_max = π/h (the grid Nyquist, derived), to minimise ∫_{γ_r R}^{r_max} |F|² r² dr, and truncate at γ_r R. The review's prototype reached 1.5e-8 and 1.1e-8 at h = 0.25.
- M4: for an inversion-even g, the lattice sum is S(R) = Σ_G ĝ(G) cos(G·R); the first shell's 2ĝ(2π/h)Σ_a cos(2πR_a/h) is exactly 0 at every R_a = ±h/4. The test therefore uses the shifts R ∈ {0, h/2}³, where cos = ±1, together with the ±h/4 set.
- **Task 5 physics review** (`.superpowers/sdd/2026-10-15-m7-pseudopotentials/task-5-ruling.md`):
  - The weight-1 1e-8 lattice judge is withdrawn: it was never derived, and its prototype figures were per-G ratios.
  - T8 gains three criteria: (b) the filtered T3 within 5 mHa (S2); (c) the LSDA domain, the XC clamp never engaging in
    any configuration; (d) the table egg-box force F_SR + F_NL + F_pc < `gates.h_req_force` = 2e-6 Ha/Bohr, each table
    weighted by the field it multiplies in the energy and probed at the §6.6 offsets (spec §5.1 "Translation invariance").
  - The outer step spans [γ_r R, γ_r R + 8h], and every table's radius is γ_r R + 8h.
  - 3-D sampling reads uniform tables (Δ = h/256) filled exactly.
  - S3 becomes T8(c).
  - Version `m7-tm-kb-nlcc-3`.

**Files:**
- Create: `crates/qm/src/radial/filter.rs`
- Modify: `crates/qm/src/radial/recipe.rs` (h_req and the filter rows in the record), `crates/qm/src/radial/record.rs` (`FilterRow` gains `clamped: usize` and `f_table: [f64; 3]`, and `passes()`), `crates/qm/src/radial/ledger.rs` (add `("filter.rs", include_str!("filter.rs"))` to `gate_thresholds_are_constants`; `gates.h_req_force`, `numerics.kspl_step_over_h`, `numerics.table_points_per_h`, and `version: "m7-tm-kb-nlcc-3"` — the filter's h_req moves every pinned record, Global Constraints), `crates/qm/src/radial/mod.rs`, `crates/validate/tests/pseudo_gen.rs` (h_req printout), `crates/validate/references.md` (regenerated records block; section "M7 filter (Task 5)")

**Interfaces:**
- Consumes: Tasks 1–4 (`PseudoTable`, `KbInput`, `solve_kb_atom`, `reference_config`, `transfer_configs`, `GenerationRecord`, `FilterRow`, `LEDGER.{q_c_over_nyquist, gamma_r, sigma_lr_over_h, v_tail_floor, r_max, h_ladder, numerics.kspl_terms, gates.h_req_tol, gates.t1_eig, gates.t3_delta_e}`).
- Produces (Tasks 6–8 and 11 use exactly these):
```rust
pub fn sph_bessel(l: u8, x: f64) -> f64;
pub fn sph_bessel_scaled(l: u8, x: f64) -> f64;                 // j_l(x)/x^l, finite at 0
pub fn mask(r: f64, r_in: f64, r_out: f64) -> f64;              // C^∞ step: 1 → 0 on [r_in, r_out]
pub fn filter_radial(grid: &LogGrid, f: &[f64], l: u8, q_c: f64, q_max: f64, r_cut: f64, width: f64) -> Vec<f64>; // KSPL; returns f^filt/r^l
pub struct FilteredProjector { pub l: u8, pub e_kb: f64, pub t: Vec<f64> /* F_l^filt/r^l on the log grid (radial re-run) */,
                               pub t_u: Vec<f64> /* the same on the uniform 3-D table */, pub radius: f64 }
pub struct FilteredTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub h: f64, pub sigma_lr: f64,
    pub v_sr_table: Vec<f64>, pub r_sr: f64,
    pub projectors: Vec<FilteredProjector>, pub rho_pc: Vec<f64>, pub pc_radius: f64, pub r_core: f64,
    /// 3-D sampling (spec §5.1 "Sampling"): uniform in d, spacing du = h / LEDGER.numerics.table_points_per_h,
    /// filled with the band-limited function evaluated exactly; index 0 is d = 0.
    pub du: f64, pub v_sr_u: Vec<f64>, pub rho_pc_u: Vec<f64>,
    pub check: FilterRow, pub record: GenerationRecord, pub grid: LogGrid }
impl FilteredTable {
    pub fn v_lr(&self, d: f64) -> f64;   pub fn dv_lr(&self, d: f64) -> f64;      // analytic erf
    pub fn v_sr(&self, d: f64) -> f64;   pub fn dv_sr(&self, d: f64) -> f64;      // uniform table, 4-point Lagrange and its exact derivative
    pub fn sr_radius(&self) -> f64;
    pub fn beta_radial(&self, k: usize, d: f64) -> f64; pub fn dbeta_radial(&self, k: usize, d: f64) -> f64; // uniform t_u
    pub fn rho_pc_at(&self, d: f64) -> f64; pub fn drho_pc_at(&self, d: f64) -> f64;                         // uniform
    pub fn input(&self) -> KbInput;                       // the radial KB atom on the exact log-node values (not the uniform tables)
    pub fn unfiltered(table: &PseudoTable, h: f64) -> FilteredTable;
}
/// T8(d) (spec §5.1 "Translation invariance"): [F_SR, F_NL, F_pc] in Ha/Bohr, the max over R ∈ {±h/4}³ of
/// |∇_R h³Σ_p w(r_p − R) g(r_p − R)| with g read through the uniform tables and w from `atom` (the filtered KB atom at
/// the reference configuration) hard-cut at |q| = π/h: ρ_v for V_SR; φ_l through Σ_m (f_l/(2l+1)) E_KB c_lm² for β_l
/// (an unoccupied channel: its reference pseudo-wavefunction, f = 1); ½(v_xc↑ + v_xc↓)[ρ_v + ρ_pc^filt] for ρ_pc.
pub fn table_eggbox(t: &PseudoTable, ft: &FilteredTable, atom: &KbAtom) -> [f64; 3];
/// T8(c): the number of (configuration, spin, r) points with ρ_σ + ½ρ_pc < 0 where ρ_pc < 0.
pub fn xc_domain_violations(rho: &[Vec<f64>; 2], rho_pc: &[f64]) -> usize;
pub fn filtered(table: &PseudoTable, h: f64) -> FilteredTable;
pub fn h_req(table: &PseudoTable) -> (Option<f64>, Vec<FilterRow>);
```
And in `record.rs`:
```rust
pub struct FilterRow { pub h: f64, pub de: f64, pub max_eig: f64, pub t3_max: f64,
                       /// T8(c): points where the XC clamp would engage, over the reference and every T3 configuration.
                       pub clamped: usize,
                       /// T8(d): [F_SR, F_NL, F_pc], Ha/Bohr.
                       pub f_table: [f64; 3] }
impl FilterRow {
    /// T8 (spec §4.8): (a) |ΔE|, max |Δε| < h_req_tol; (b) t3_max < t3_delta_e; (c) clamped = 0; (d) Σ f_table < h_req_force.
    /// `filter::h_req` and `ionic::check` both use this, so they cannot drift.
    pub fn passes(&self) -> bool {
        let g = &LEDGER.gates;
        self.de.abs() < g.h_req_tol && self.max_eig < g.h_req_tol && self.t3_max < g.t3_delta_e
            && self.clamped == 0 && self.f_table.iter().sum::<f64>() < g.h_req_force
    }
}
```
**The local split** (Spec question 2 as the plan review ruled; spec §5.1) uses a = σ_LR√2 and σ_LR = `LEDGER.sigma_lr_over_h`·h.
- V_LR(d) = −Z_v erf(d/a)/d is analytic per centre on the grid (Spec question 6). It is not tabulated.
- **V_SR(d) = V_loc(d) + Z_v erf(d/a)/d is one table.** It is smooth and finite at 0 (V_loc is finite and even; erf(d/a)/d → 2/(√π a)). Beyond r_cmp it equals −Z_v erfc(d/a)/d to `v_ion_tail`.
- Its support radius is R_SR = max(r_cmp, r_floor), where r_floor solves Z_v erfc(r/a)/r = `v_tail_floor` (r_floor ≈ 2.5 Bohr at h = 0.25, ≈ 3.5 at h = 0.35 for Z_v = 6; for Be and Al–S r_cmp dominates — Al 4.18, Si 3.65, P 3.24, measured in Task 2). It is derived from two existing ledger fields, so no new constant. The table is band-completed and truncated at γ_r·R_SR, where |V_SR| ≤ 1e-6 Ha: the truncation discards nothing of the exact split, so V_SR never has to be split again.
- Interaction with the filter: the grid sees V_loc below q_c plus V_LR's content above q_c, whose relative weight e^{−(q_c a/2)²} = e^{−(0.7π√2)²} ≈ 6e-5 at q_c enters the T8/S2 re-run automatically, because `input()` sums v_lr + v_sr.

**The band completion** (Spec question 3 as the plan review ruled; King-Smith, Payne & Lin, PRB 44, 13063 (1991)), the same for every ion-centred table g ∈ {V_SR, β_l, ρ_pc}, with R its support radius (r_β for β_l, spec §4.6; R_SR for V_SR; for ρ_pc the XC-weight radius of spec §4.9, Task 5 physics review S1):
1. F̂(q) = 4π∫F j_l(qr) r² dr on a Simpson mesh of [0, q_c], q_c = `LEDGER.q_c_over_nyquist`·π/h. This part is kept.
2. On (q_c, q_max], q_max = π/h, F̂ is a Chebyshev series of `LEDGER.numerics.kspl_terms` terms, chosen by least squares to minimise ∫_{γ_r R}^{r_max} |F|² r² dr. The integral runs on a uniform r mesh of spacing π/(4q_max), derived from q_max.
3. The result is truncated at γ_r R by the parameter-free ψ-step over [γ_r R, γ_r R + 8h] (`numerics.kspl_step_over_h`; Task 5 physics review: the width sets the projector leakage). The step acts only where the optimised tail is already small (review Q3: "keep the ψ-step only as the outer truncation"). The table's radius is γ_r R + 8h.

The projector radial part is F_l = δV_l u_l/r, transformed with j_l. The table stores t_l = F_l^filt/r^l, so the grid function is β_lm(r) = t_l(d)·√((2l+1)/4π)·R_lm(d), exactly the spec §4.6 S1 form. ρ_pc is transformed with j_0, and its R is its own support radius.

Every sampled ion-centred function (V_SR, β, ρ_pc) is then band-limited up to the completion's leakage. V_LR's alias at 2π/h is ∝ e^{−(πa/h)²} = e^{−8π²}.

- [ ] **Step 1: Write the failing tests** (`crates/qm/src/radial/filter.rs` tests)
```rust
use super::*;
use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
use crate::radial::generate;

/// h³ Σ_p g over a lattice of spacing h around the origin, shifted by `s`.
fn lattice_sum(h: f64, radius: f64, s: [f64; 3], g: impl Fn([f64; 3]) -> f64) -> f64 {
    let k = (radius / h).ceil() as i64 + 2;
    let mut terms = Vec::new();
    for i in -k..=k { for j in -k..=k { for l in -k..=k {
        terms.push(g([i as f64 * h - s[0], j as f64 * h - s[1], l as f64 * h - s[2]]));
    }}}
    crate::grid::neumaier(terms.into_iter()) * h * h * h
}

/// The rigid shifts of the lattice-sum test (plan review M4). For an inversion-even g, Poisson
/// summation gives S(R) = Σ_G ĝ(G)cos(G·R): the first reciprocal shell contributes
/// 2ĝ(2π/h)Σ_a cos(2πR_a/h), exactly 0 at every R_a = ±h/4 — those 8 shifts see only the √2 and √3
/// shells. R ∈ {0, h/2}³ makes cos = ±1 (the dominant alias at full weight); the ±h/4 set is kept too.
fn shifts(h: f64) -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for set in [[0.0, h / 2.0], [-h / 4.0, h / 4.0]] {
        for a in set { for b in set { for c in set { v.push([a, b, c]); } } }
    }
    v
}

Replace `filtered_lattice_sums_are_translation_invariant`, `filtered_partial_core_is_nonnegative` and
`filtered_kb_atom_within_1mha_at_h_req` with the tests below (Task 5 physics review). Keep `bessel_functions_and_transform_pair`,
`unfiltered_input_reproduces_the_kb_input`, `simpson_images_lie_beyond_the_grid`,
`hard_cut_reproduces_the_plan_review_table` and `lattice_spread_sources_report`.

```rust
/// Report for references.md (no bound; spec §5.1: the weight-1 sum is neither the energy nor the force): the
/// {0, h/2}³ lattice spreads of every filtered table at h = 0.25 for C and O, the ±h/4 set (identically 0 by
/// cubic symmetry for every reflection-even integrand), and the T8(d) forces beside them.
#[test]
fn lattice_spreads_report() { /* the wip's filtered_lattice_sums_are_translation_invariant body, eprintln only,
                                  plus `table_eggbox` at the same h */ }

/// Spec §5.1 "Sampling": every uniform 3-D table reproduces the band-limited function it samples to ≤ 1e-9 of its
/// maximum (derived bound (9/16)(π/256)⁴/24 = 5.3e-10, Bernstein) at 10⁴ random d in [0, radius].
#[test]
fn uniform_tables_match_the_band_limited_function() {
    for (z, h) in [(8u8, 0.20), (6, 0.30), (9, 0.15)] {
        let t = generate(z).unwrap();
        let ft = filtered(&t, h);
        // compare v_sr / beta_radial / rho_pc_at against Completion::value_at (exact) for each table;
        // assert max |Δ| ≤ 1e-9 · max |g| per table, printing the worst
    }
}

/// T8(d)'s probe is first-order exact in the leakage (spec §5.1): a projector that is band-limited *and* compact
/// gives no egg-box force. As built (Task 5): t = e^{−d²/s} on O's s-projector sphere ρ = γ_r r_β + 8h, with its content
/// at π/h (e^{−(π/h)²s/4}) and its value where the table ends (e^{−ρ²/s}) both ≤ 1e-13, which is feasible at h = 0.15
/// (s ∈ [0.274, 0.307]); assert F_NL < 1e-12 (measured 7.7e-13). The review's e^{−d²/0.5} has 2.7e-9 of its content
/// at π/h for h = 0.25 and is cut at 1e-8 by the h = 0.15 sphere; the probe then measures that leakage (7.8e-12 and
/// 2.5e-9), so it is printed, not asserted.
#[test]
fn table_eggbox_probe_is_blind_to_band_limited_tables() { /* as built in filter.rs */ }

/// The probe sees what it must (Task 5 physics review, Table A): O at h = 0.20 with the outer step one spacing wide
/// (the brief's construction) has F_NL = 8.0e-5 Ha/Bohr. Assert it is > 10 × h_req_force, so a blind probe fails
/// here. With the ruled 8h it is 9.2e-7.
#[test]
fn table_eggbox_probe_sees_a_one_spacing_step() {
    let t = generate(8).unwrap();
    let narrow = super::build_with_step(&t, 0.20, 1.0);  // cfg(test)-visible: the same build with width 1·h
    let f = narrow.check.f_table[1];
    eprintln!("O, h = 0.20, step width h: F_NL {f:.2e}");
    assert!(f > 10.0 * LEDGER.gates.h_req_force, "F_NL {f:.3e}");
}

/// T8(c)'s counter on a synthetic case: a minority spin of 1e-4 against ½ρ_pc = −2e-4 at one point counts once; the
/// same lobe under a majority density of 1e-1 does not count.
#[test]
fn xc_domain_counter_counts_minority_spin_violations() {
    let rho = [vec![1e-1, 1e-1], vec![1e-1, 1e-4]];
    assert_eq!(xc_domain_violations(&rho, &[-4e-4, -4e-4]), 1);
    assert_eq!(xc_domain_violations(&rho, &[-4e-4, 1.0]), 0);
}

/// Spec §4.8 T8, §6.4: the species of M7's molecules (H₂O, NH₃, CH₄, O₂) have an h_req, and every ladder row at or
/// below it passes all four T8 criteria (FilterRow::passes); every row is printed (ΔE, Δε, T3, clamped, F_SR/F_NL/F_pc).
/// Ne's h_req (None at -3: ρ_pc carries 0.97 of its 1.21 mHa at 0.15) is pinned by the records table, not here.
#[test]
fn m7_species_have_h_req_and_every_ladder_row_below_passes() {
    let mut fails = Vec::new();
    for z in [1u8, 6, 7, 8] {
        let t = generate(z).unwrap();
        for r in &t.record.filter {
            eprintln!("Z = {z} h = {:.2}: ΔE {:+.3e}, Δε {:.3e}, T3 {:.3e}, clamped {}, F {:.2e}/{:.2e}/{:.2e} → {}",
                      r.h, r.de, r.max_eig, r.t3_max, r.clamped, r.f_table[0], r.f_table[1], r.f_table[2], r.passes());
        }
        let Some(h) = t.record.h_req else { fails.push(format!("Z = {z}: no h_req")); continue };
        for r in t.record.filter.iter().filter(|r| r.h <= h + 1e-12) { if !r.passes() { fails.push(format!("Z = {z} h = {}: fails T8", r.h)); } }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}
```
As built (Task 5), spec §8 row 5's named tests are also present: `filtered_kb_atom_within_1mha_at_h_req` (T8(a, b) at every
certified h for H and C–F), `table_eggbox_force_within_budget_at_h_req` (T8(d) for H and C–F) and
`filtered_xc_domain_is_respected` (T8(c) for C–F and Si); the tests share one `generate` per element (a `OnceLock` cache).

```rust
#[test]
fn bessel_functions_and_transform_pair() {
    // closed forms vs series across x = 1
    for l in 0..=3u8 { for &x in &[0.999, 1.001, 0.3, 5.0] {
        let s = sph_bessel_scaled(l, x) * x.powi(l as i32);
        assert!((s - sph_bessel(l, x)).abs() < 1e-12, "l = {l}, x = {x}");
    }}
    // a Gaussian is band-limited enough that a cut far beyond its spectrum returns it (the band
    // completion then has nothing to cancel: its least-squares solution is ≈ 0)
    let g = LogGrid::universal();
    let f: Vec<f64> = g.r.iter().map(|r| (-r * r).exp()).collect();
    let back = filter_radial(&g, &f, 0, 40.0, 45.0, 50.0, 5.0);
    for &r in &[0.0, 0.5, 1.5] { let i = g.nearest(r.max(1e-5)); assert!((back[i] - f[i]).abs() < 1e-9); }
}

/// Plan review Q3 diagnostic (report): on the TM-class model, the ruled completion (outer step 8h) lies below the
/// hard cut at the first and √2 shells. The judge is T8(d); the old 1e-7 was not derived.
#[test]
fn band_completion_suppresses_the_first_shell() { /* as the wip, with width LEDGER.numerics.kspl_step_over_h·h;
    assert (f1/f0).abs() < 1.574e-6 && (f2/f0).abs() < 5.557e-7 (the hard cut's ratios, reproduced in
    hard_cut_reproduces_the_plan_review_table) */ }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release radial::filter`
Expected: compile errors.

- [ ] **Step 3: Implement the filter**

`crates/qm/src/radial/filter.rs`. Its module doc carries:
- spec §5.1 and the Spec question 2 ruling with its derivation: the split V_cmp = V_loc + Z_v/d, V_tail = −Z_v erfc/d is singular piece by piece (+Z_v/d and −Z_v/d at the origin), and FT[−Z_v erfc(d/a)/d] = −4πZ_v(1 − e^{−q²a²/4})/q² is the *complement* of the band-limited Gaussian part, decaying only as q⁻², so it is not "band-limited by σ_LR". The one table V_SR is smooth, equal to the erfc tail beyond r_cmp, and below `v_tail_floor` beyond R_SR;
- Poisson summation (a function band-limited below 2π/h has translation-invariant lattice sums), and why ±h/4 shifts are blind to the first shell (M4);
- King-Smith, Payne & Lin's band completion (Spec question 3): the least-squares problem, the Chebyshev band, q_max = π/h, and the uniform r mesh;
- the mask (C^∞, Gevrey-class step ψ(1−t)/(ψ(t)+ψ(1−t)), ψ(t) = e^{−1/t}: no parameter), used only as the outer truncation over [γ_r R, γ_r R + 8h] (numerics.kspl_step_over_h; Task 5 physics review: the width sets the projector leakage);
- the q mesh (uniform, dq ≤ π/(2r_max): Simpson's coarse trapezoid recurs at π/dq, so its image lies beyond r_max; Simpson per interval, so the jump at q_c is integrated piecewise);
- the solve: discrete Picard truncation (Hansen, BIT 30, 658 (1990));
- 3-D sampling: uniform tables, Δ = h/256, exact fill, 4-point Lagrange (5.3e-10, Bernstein);
- T8(c) and T8(d): the LSDA domain count and table_eggbox (spec §5.1 "Translation invariance", the weights and the first-order argument);
- S2 (T8(b)) and S3 as the LSDA domain (T8(c)).

```rust
use super::gates::transfer_configs;
use super::grid::LogGrid;
use super::kb::{reference_config, solve_kb_atom, KbInput};
use super::ledger::LEDGER;
use super::record::{FilterRow, GenerationRecord};
use super::recipe::PseudoTable;
use super::atom::{ground_state, solve_atom, SpinMode};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

pub fn sph_bessel_scaled(l: u8, x: f64) -> f64 {
    if x.abs() < 1.0 {
        // j_l(x)/x^l = Σ_k (−x²/2)^k / (k! (2l+2k+1)!!)
        let mut df = 1.0;
        for k in 1..=l as usize { df *= (2 * k + 1) as f64; }
        let (mut term, mut sum) = (1.0 / df, 1.0 / df);
        for k in 1..10usize {
            term *= -x * x / (2.0 * k as f64 * (2 * l as usize + 2 * k + 1) as f64);
            sum += term;
        }
        return sum;
    }
    sph_bessel(l, x) / x.powi(l as i32)
}

pub fn sph_bessel(l: u8, x: f64) -> f64 {
    if x.abs() < 1.0 { return sph_bessel_scaled(l, x) * x.powi(l as i32); }
    let (s, c) = (x.sin(), x.cos());
    match l {
        0 => s / x,
        1 => s / (x * x) - c / x,
        2 => (3.0 / (x * x) - 1.0) * s / x - 3.0 * c / (x * x),
        3 => (15.0 / x.powi(3) - 6.0 / x) * s / x - (15.0 / (x * x) - 1.0) * c / x,
        _ => {
            let (mut a, mut b) = (sph_bessel(2, x), sph_bessel(3, x));
            for k in 3..l { let n = (2 * k + 1) as f64 / x * b - a; a = b; b = n; }
            b
        }
    }
}

fn psi(t: f64) -> f64 { if t > 0.0 { (-1.0 / t).exp() } else { 0.0 } }

pub fn mask(r: f64, r_in: f64, r_out: f64) -> f64 {
    if r <= r_in { return 1.0; }
    if r >= r_out { return 0.0; }
    let t = (r - r_in) / (r_out - r_in);
    psi(1.0 - t) / (psi(t) + psi(1.0 - t))
}

/// A uniform Simpson mesh on [a, b] with an even number of intervals and dq ≤ π/r_max (derived from
/// the ledger's r_max: the inverse transform is alias-free out to 2r_max): (nodes, weights).
fn q_mesh(a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let n = 2 * (((b - a) * LEDGER.r_max / (2.0 * PI)).ceil() as usize).max(1);
    let dq = (b - a) / n as f64;
    let q = (0..=n).map(|k| a + k as f64 * dq).collect();
    let w = (0..=n).map(|k| dq / 3.0 * if k == 0 || k == n { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 }).collect();
    (q, w)
}

/// Chebyshev T_0..T_{m−1} at s ∈ [−1, 1].
fn chebyshev(s: f64, m: usize) -> Vec<f64> {
    let mut t = vec![1.0; m];
    if m > 1 { t[1] = s; }
    for k in 2..m { t[k] = 2.0 * s * t[k - 1] - t[k - 2]; }
    t
}

/// King-Smith, Payne & Lin's band completion (plan review Q3). F̂ is kept on [0, q_c]; on (q_c, q_max]
/// it is the Chebyshev series (LEDGER.numerics.kspl_terms terms) that minimises
/// ∫_{r_cut}^{r_max} |F|² r² dr — least squares on a uniform r mesh of spacing π/(4q_max), solved by
/// SVD with a 1e-12 relative singular-value floor. The result is truncated by the ψ-step over
/// [r_cut, r_cut + width], where the optimised tail is already small. Returns F^filt/r^l (finite at 0).
/// Transform pair: F̂(q) = 4π∫F j_l(qr) r² dr, F(r) = (1/2π²)∫F̂ j_l(qr) q² dq.
pub fn filter_radial(grid: &LogGrid, f: &[f64], l: u8, q_c: f64, q_max: f64, r_cut: f64, width: f64) -> Vec<f64> {
    let n = grid.len();
    let li = l as i32;
    let (qk, wk) = q_mesh(0.0, q_c);
    let (qb, wb) = q_mesh(q_c, q_max);
    let fk: Vec<f64> = qk.iter().map(|&q| 4.0 * PI * grid.integrate(&(0..n).map(|i| f[i] * sph_bessel(l, q * grid.r[i]) * grid.r[i] * grid.r[i]).collect::<Vec<_>>())).collect();
    let m = LEDGER.numerics.kspl_terms;
    let cheb: Vec<Vec<f64>> = qb.iter().map(|&q| chebyshev(2.0 * (q - q_c) / (q_max - q_c) - 1.0, m)).collect();
    // least squares over the tail: rows r_j ∈ [r_cut, r_max], weight √(r_j² Δr)
    let dr = PI / (4.0 * q_max);
    let rs: Vec<f64> = (0..).map(|j| r_cut + j as f64 * dr).take_while(|&r| r <= LEDGER.r_max).collect();
    let mut a = DMatrix::<f64>::zeros(rs.len(), m);
    let mut b = DVector::<f64>::zeros(rs.len());
    for (j, &r) in rs.iter().enumerate() {
        let sw = (r * r * dr).sqrt() / (2.0 * PI * PI);
        b[j] = -sw * (0..qk.len()).map(|k| wk[k] * fk[k] * qk[k] * qk[k] * sph_bessel(l, qk[k] * r)).sum::<f64>();
        for k in 0..qb.len() {
            let jb = sw * wb[k] * qb[k] * qb[k] * sph_bessel(l, qb[k] * r);
            for c in 0..m { a[(j, c)] += jb * cheb[k][c]; }
        }
    }
    let svd = a.svd(true, true);
    let floor = 1e-12 * svd.singular_values.max();
    let x = svd.solve(&b, floor).expect("the SVD was computed with U and V");
    let fb: Vec<f64> = cheb.iter().map(|t| t.iter().zip(x.iter()).map(|(tc, xc)| tc * xc).sum::<f64>()).collect();
    (0..n).map(|i| {
        let r = grid.r[i];
        let kept: f64 = (0..qk.len()).map(|k| wk[k] * fk[k] * qk[k].powi(li + 2) * sph_bessel_scaled(l, qk[k] * r)).sum();
        let band: f64 = (0..qb.len()).map(|k| wb[k] * fb[k] * qb[k].powi(li + 2) * sph_bessel_scaled(l, qb[k] * r)).sum();
        (kept + band) / (2.0 * PI * PI) * mask(r, r_cut, r_cut + width)
    }).collect()
}

#[derive(Clone, Debug)]
pub struct FilteredProjector { pub l: u8, pub e_kb: f64, pub t: Vec<f64>, pub radius: f64 }

#[derive(Clone, Debug)]
pub struct FilteredTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub h: f64, pub sigma_lr: f64,
    pub v_sr_table: Vec<f64>, pub r_sr: f64,
    pub projectors: Vec<FilteredProjector>, pub rho_pc: Vec<f64>, pub pc_radius: f64, pub r_core: f64,
    pub check: FilterRow, pub record: GenerationRecord, pub grid: LogGrid,
}

impl FilteredTable {
    fn a(&self) -> f64 { self.sigma_lr * 2f64.sqrt() }
    pub fn v_lr(&self, d: f64) -> f64 {
        let a = self.a();
        if d < 1e-10 { -self.z_v * 2.0 / (PI.sqrt() * a) } else { -self.z_v * libm::erf(d / a) / d }
    }
    pub fn dv_lr(&self, d: f64) -> f64 {
        if d < 1e-10 { return 0.0; }
        let (a, x) = (self.a(), d / self.a());
        -self.z_v * ((2.0 / PI.sqrt()) * x * (-x * x).exp() - libm::erf(x)) / (d * d)
    }
    /// The one short-range table (plan review Q2) and the exact derivative of its interpolant (so
    /// F^SR is the derivative of the sampled energy).
    pub fn v_sr(&self, d: f64) -> f64 { if d > self.r_sr { 0.0 } else { self.grid.interp(&self.v_sr_table, d) } }
    pub fn dv_sr(&self, d: f64) -> f64 { if d > self.r_sr { 0.0 } else { self.grid.interp_deriv(&self.v_sr_table, d) } }
    pub fn sr_radius(&self) -> f64 { self.r_sr }
    pub fn beta_radial(&self, k: usize, d: f64) -> f64 { let p = &self.projectors[k]; if d > p.radius { 0.0 } else { self.grid.interp(&p.t, d) } }
    pub fn dbeta_radial(&self, k: usize, d: f64) -> f64 { let p = &self.projectors[k]; if d > p.radius { 0.0 } else { self.grid.interp_deriv(&p.t, d) } }
    pub fn rho_pc_at(&self, d: f64) -> f64 { if d > self.pc_radius { 0.0 } else { self.grid.interp(&self.rho_pc, d) } }
    pub fn drho_pc_at(&self, d: f64) -> f64 { if d > self.pc_radius { 0.0 } else { self.grid.interp_deriv(&self.rho_pc, d) } }

    pub fn input(&self) -> KbInput {
        let r = &self.grid.r;
        KbInput {
            z: self.z, z_v: self.z_v,
            v_loc: r.iter().map(|&d| self.v_lr(d) + self.v_sr(d)).collect(),
            projectors: self.projectors.iter().enumerate()
                .map(|(k, p)| (p.l, p.e_kb, r.iter().map(|&d| d.powi(p.l as i32 + 1) * self.beta_radial(k, d)).collect())).collect(),
            rho_pc: r.iter().map(|&d| self.rho_pc_at(d)).collect(),
        }
    }

    pub fn unfiltered(table: &PseudoTable, h: f64) -> FilteredTable { build(table, h, false) }
}

pub fn filtered(table: &PseudoTable, h: f64) -> FilteredTable { build(table, h, true) }

fn build(t: &PseudoTable, h: f64, filter: bool) -> FilteredTable {
    let g = &t.grid;
    let n = g.len();
    let gamma = LEDGER.gamma_r;
    let sigma_lr = LEDGER.sigma_lr_over_h * h;
    let a = sigma_lr * 2f64.sqrt();
    let (q_c, q_max) = (LEDGER.q_c_over_nyquist * PI / h, PI / h);
    // a function's support radius: where |f| last exceeds 1e-12 of its maximum (r0 for f ≡ 0)
    let support = |f: &[f64]| { let m = f.iter().fold(0.0f64, |x, y| x.max(y.abs())); g.r[f.iter().rposition(|y| y.abs() > 1e-12 * m).unwrap_or(0)] };
    // the one short-range table (plan review Q2): V_SR = V_loc + Z_v erf(d/a)/d, smooth and finite at 0,
    // = −Z_v erfc(d/a)/d (to v_ion_tail) beyond r_cmp
    let v_sr: Vec<f64> = g.r.iter().zip(&t.kb.v_loc).map(|(&d, &v)| v + t.z_v * libm::erf(d / a) / d).collect();
    // R_SR = max(r_cmp, r_floor), r_floor: the outermost grid r with Z_v erfc(r/a)/r ≥ v_tail_floor
    let r_floor = g.r.iter().copied().rev().find(|&d| t.z_v * libm::erfc(d / a) / d >= LEDGER.v_tail_floor).unwrap_or(g.r[0]);
    let r_sr_support = t.kb.r_cmp.max(r_floor);
    let (v_sr_table, r_sr) = if filter {
        (filter_radial(g, &v_sr, 0, q_c, q_max, gamma * r_sr_support, h), gamma * r_sr_support + h)
    } else {
        let r = support(&v_sr);
        (v_sr, r)
    };
    let projectors = t.kb.projectors.iter().map(|p| {
        let f: Vec<f64> = (0..n).map(|i| p.beta[i] / g.r[i]).collect();   // F_l = δV·u/r = β_u/r (the 3-D radial part)
        let r_l = p.support(g);   // r_β (spec §4.6), derived — not the 1e-12 rule
        let tab = if filter { filter_radial(g, &f, p.l, q_c, q_max, gamma * r_l, h) } else { (0..n).map(|i| f[i] / g.r[i].powi(p.l as i32)).collect() };
        FilteredProjector { l: p.l, e_kb: p.e_kb, t: tab, radius: if filter { gamma * r_l + h } else { r_l } }
    }).collect();
    let r_pc = support(&t.partial_core.rho_pc);
    let (rho_pc, pc_radius) = if filter {
        (filter_radial(g, &t.partial_core.rho_pc, 0, q_c, q_max, gamma * r_pc, h), gamma * r_pc + h)
    } else {
        (t.partial_core.rho_pc.clone(), r_pc)
    };
    let mut ft = FilteredTable {
        z: t.z, z_v: t.z_v, n_core: t.n_core, h, sigma_lr, v_sr_table, r_sr,
        projectors, rho_pc, pc_radius, r_core: t.partial_core.r_core,
        check: FilterRow { h, de: 0.0, max_eig: 0.0, t3_max: 0.0 }, record: t.record.clone(), grid: g.clone(),
    };
    ft.check = filtered_rerun(t, &ft);
    ft
}
```
The u-form ↔ 3-D convention: F_l = β_u/r, t_l = F_l/r^l, and `input()` inverts it as β_u = r^{l+1}·t_l. `unfiltered_input_reproduces_the_kb_input` (below) pins the round trip. The filtered tables extend to γ_r R + h, one grid spacing beyond the review's γ_r R, because the ψ-step that ends them starts at γ_r R.

The filtered re-run (S2) and h_req (T8, Spec question 11):
```rust
/// T1 and T3 re-run on the filtered tables at the table's h: the reference energy/eigenvalues and
/// the transfer ΔE's, against the unfiltered KB atom and the AE atom.
fn filtered_rerun(t: &PseudoTable, ft: &FilteredTable) -> FilterRow {
    let base = reference_config(&t.partition);
    let un = solve_kb_atom(&t.grid, &t.kb.input(), &base, SpinMode::Restricted).expect("the unfiltered reference converged in generate");
    let Ok(fa) = solve_kb_atom(&ft.grid, &ft.input(), &base, SpinMode::Restricted) else {
        return FilterRow { h: ft.h, de: f64::INFINITY, max_eig: f64::INFINITY, t3_max: f64::INFINITY };
    };
    let max_eig = t.partition.channels.iter().filter(|c| c.occupied).map(|c| (fa.eps[c.l as usize][0] - un.eps[c.l as usize][0]).abs()).fold(0.0, f64::max);
    let hund = ground_state(t.z, SpinMode::Polarised).expect("converged in generate");
    let mut t3_max = 0.0f64;
    for cfg in transfer_configs(t, &hund) {
        let ae = if cfg.name == "Hund" { Ok(hund.clone()) } else { solve_atom(t.z, &cfg.ae) };
        if let (Ok(ae), Ok(pp)) = (ae, solve_kb_atom(&ft.grid, &ft.input(), &cfg.pp, cfg.spin)) {
            if t.gates.t3.iter().any(|r| r.name == cfg.name) {
                t3_max = t3_max.max(((pp.energy - fa.energy) - (ae.energy.total - t.reference.energy.total)).abs());
            }
        }
    }
    FilterRow { h: ft.h, de: fa.energy - un.energy, max_eig, t3_max }
}

/// T8: h_req = the coarsest ladder h such that it and every finer ladder h reproduce the unfiltered
/// energy and valence eigenvalues within LEDGER.gates.h_req_tol (Spec question 11). The rows are the
/// generation record's filter ΔE per h.
pub fn h_req(table: &PseudoTable) -> (Option<f64>, Vec<FilterRow>) {
    let rows: Vec<FilterRow> = LEDGER.h_ladder.iter().map(|&h| filtered(table, h).check).collect();
    let tol = LEDGER.gates.h_req_tol;
    let pass = |r: &FilterRow| r.de.abs() < tol && r.max_eig < tol;
    let first = (0..rows.len()).find(|&k| rows[k..].iter().all(pass));
    (first.map(|k| rows[k].h), rows)
}
```
In `recipe::generate`, after the ladder accepts a table:
```rust
let (hr, rows) = filter::h_req(&t);
t.record.h_req = hr;
t.record.filter = rows;
```
Add the test that pins the u-form ↔ 3-D convention:
```rust
#[test]
fn unfiltered_input_reproduces_the_kb_input() {
    let t = generate(8).unwrap();
    let (a, b) = (t.kb.input(), FilteredTable::unfiltered(&t, 0.2).input());
    for (pa, pb) in a.projectors.iter().zip(&b.projectors) {
        for i in (0..t.grid.len()).step_by(37) { assert!((pa.2[i] - pb.2[i]).abs() <= 1e-10 * (1.0 + pa.2[i].abs()), "β at r = {}", t.grid.r[i]); }
    }
    for i in (0..t.grid.len()).step_by(37) { assert!((a.v_loc[i] - b.v_loc[i]).abs() < 1e-9, "V_loc at r = {}", t.grid.r[i]); }
}
```
(The exact split makes V_LR + V_SR = V_loc for the unfiltered table, the V_SR table being cut only where it is below 1e-12 of its maximum, and β_u round-trips.)

- [ ] **Step 4: Run the tests**

Run: `cargo test -p qm --release radial`
Expected: PASS.

Stop and report the full FilterRow ladder per element if either test fails:
- `m7_species_have_h_req_and_every_ladder_row_below_passes`: H, C, N or O without an h_req;
- `table_eggbox_probe_sees_a_one_spacing_step`: the probe blind.

Neither is ever answered with a looser bound. Any other element's h_req, including None, is an outcome the records
table pins. B, P, S, Cl, Ar and He were not probed in review; a T8(d) failure at an element's energy h_req moves that
element's h_req and is recorded, not patched.

- [ ] **Step 5: Bump the recipe version, regenerate and pin the records.** The filter's h_req fills a column of every pinned row, so set `LEDGER.version = "m7-tm-kb-nlcc-3"` (Global Constraints, plan review M6; Task 4 took -2). Without the bump, `generation_record_table_matches_references_md` fails with "bump LEDGER.version". Then run `cargo test -p validate --release --test pseudo_gen`. It fails because the block moved, and writes `last-run-generation-records.md`. After physics review, copy that block into `references.md` and add a section "M7 filter (Task 5)". Per element and ladder h, give ΔE, max Δε, filtered T3, clamped, and F_SR / F_NL / F_pc. Then give h_req (None stated as an outcome, with the T8 criterion that failed), min ρ_pc^filt and its radius, and the weight-1 lattice spreads (report only, labelled as not a judge). Re-run: PASS.

- [ ] **Step 6: Commit**
```bash
git add crates/qm/src/radial crates/validate/tests/pseudo_gen.rs crates/validate/references.md
git commit -o -m "feat(qm): spherical-Bessel filter with King-Smith band completion, the one V_SR table, h_req and the filtered T1/T3 re-run (M7 Task 5)" -- crates/qm/src/radial crates/validate/tests/pseudo_gen.rs crates/validate/references.md
```
Before merging, hand the diff to `physics-reviewer` with Spec questions 2, 3 and 11, the FilterRow ladders (all four T8 criteria) and the `lattice_spreads_report` printout.

---

### Task 6: `qm::ionic` + SCF integration **[Fable — physics-review gate]**

Spec §4.7, §5.1–§5.3, §5.5 (`qm` bullet), §5.6, §6.3, §8 row 6; review S1, S7, M3, and the NOTE on the no-core branch. **Spec question 6 applies** (accepted by the plan review): V_LR is the analytic erf, as `nuclear_potential` and `nuclear.wgsl` already compute it, exact pointwise. **Spec question 8 applies** (accepted): `All` is usable from here, with H at its atomic-gate f, until Task 11's dimer gate. Plan review amendments here: S8 (the S1 identity bound is relative, plus a separate E_KB⟨β|φ⟩ check), S6 (the on-grid-point atom solved at tol = eig_tol = 1e-9) and M4 (the energy ripple from the shifts {0, h/2}³).

**Files:**
- Create: `crates/qm/src/ionic.rs`, `crates/qm/src/nonlocal.rs`, `crates/validate/tests/pseudo.rs`
- Modify: `crates/qm/src/lib.rs`, `crates/qm/src/nuclear.rs` (centres, the ionic local potential, the partial-core field, ion–ion repulsion, the sphere point lists), `crates/qm/src/scf.rs` (`System.ionic`, `System::with_ionic`, `valence_electrons`, `KsState.ionic`, `ScfStats.core_overlaps`, `IonicTerms`, `solve_channel` with V_NL, the warm-start guard), `crates/qm-gpu/src/scf.rs` (the `ScfStats` literal gains `core_overlaps: Vec::new()`; nothing else in `qm-gpu` until Task 8), `crates/validate/references.md` (section "M7 grid integration (Task 6)")

**Interfaces:**
- Consumes: Task 5's `FilteredTable` (`v_lr`, `v_sr`, `sr_radius`, `beta_radial`, `rho_pc_at`, `pc_radius`, `projectors`, `r_core`, `record.h_req`, `check`, `h`, `z_v`, `sigma_lr`), `filtered`, `generate`, `has_core`, `Refusal`, `recipe_hash`, `LEDGER`, and `qm::poisson::{solid_harmonics, lm_index, N_MOMENTS}` (Racah-normalised, unchanged).

**Sampling rule (spec §5.1 "Sampling"; Task 5 physics review M3):**
- `ionic_local_potential`, `partial_core_field`, `NonlocalOp::build` and Task 7's force terms read the ion tables only
  through `FilteredTable::{v_sr, dv_sr, beta_radial, dbeta_radial, rho_pc_at, drho_pc_at}`, i.e. the uniform Δ = h/256
  tables.
- Never `LogGrid::interp` on a filtered table: measured, its δ = 0.008 interpolation alone exceeds T8(d)'s whole budget.

- Produces (Tasks 7–11 use exactly these):
```rust
// ionic.rs
pub enum IonicPolicy { None, CoresOnly, All }                       // Copy, Eq, Hash
pub enum Ionic { AllElectron { sigma: f64 }, Pseudo(Arc<FilteredTable>) }
pub struct IonicSetup { pub policy: IonicPolicy, pub h: f64, pub sigma_ae: f64, pub recipe_hash: u64,
                        pub per_z: BTreeMap<u8, Ionic>, pub notes: Vec<String> }
pub enum IonicError { Refused { z: u8, refusal: Refusal }, TooCoarse { z: u8, h: f64, h_req: Option<f64> }, FilteredGate { z: u8, h: f64, why: String } }
pub const CORE_OVERLAP_WARNING: &str = "cores overlap";
impl IonicSetup {
    pub fn build(policy: IonicPolicy, h: f64, sigma_ae: f64, zs: &[u8],
                 has_core: &mut dyn FnMut(u8) -> Result<bool, Refusal>,
                 table: &mut dyn FnMut(u8) -> Result<Arc<FilteredTable>, Refusal>) -> Result<IonicSetup, IonicError>;
    pub fn from_tables(policy: IonicPolicy, h: f64, sigma_ae: f64, per_z: BTreeMap<u8, Ionic>) -> IonicSetup;   // tests; no checks
    pub fn ionic(&self, z: u8) -> &Ionic;          // panics for a Z not in per_z (a programming error)
    pub fn valence_charge(&self, z: u8) -> f64;
    pub fn pseudised(&self) -> Vec<u8>;
    pub fn same_physics(&self, other: &IonicSetup) -> bool;
}
pub fn core_overlaps(setup: &IonicSetup, nuclei: &[Nucleus]) -> Vec<(usize, usize, f64)>;
// nonlocal.rs
pub struct Sphere { pub ion: usize, pub z: u8, pub l: u8, pub points: Vec<usize>, pub beta: Vec<Vec<f64>> /* [m][point] */, pub e_kb: f64 }
pub struct NonlocalOp { pub spheres: Vec<Sphere>, pub dv: f64 }
impl NonlocalOp {
    pub fn build(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> NonlocalOp;
    pub fn project(&self, psi: &[f64]) -> Vec<Vec<f64>>;          // c[sphere][m] = h³ Σ β ψ (Neumaier)
    pub fn apply_add(&self, psi: &[f64], out: &mut [f64]);         // out += Σ E_KB c β
    pub fn energy(&self, orbitals: &[Vec<f64>]) -> f64;             // Σ_i Σ E_KB c²
    pub fn is_empty(&self) -> bool;
}
pub struct PseudoHamiltonian<'a> { pub grid: &'a Grid, pub v: &'a [f64], pub nl: &'a NonlocalOp }   // impl eigen::Operator
// nuclear.rs
pub struct Centre { pub pos: [f64; 3], pub charge: f64, pub sigma: f64 }
pub fn smeared_potential(grid: &Grid, centres: &[Centre]) -> Vec<f64>;
pub fn deposit_centres(grid: &Grid, centres: &[Centre]) -> Vec<f64>;
pub fn centres(nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<Centre>;
pub fn ionic_local_potential(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<f64>;   // V_LR + Σ V_SR
pub fn partial_core_field(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<f64>;     // Σ ρ_pc
pub fn ionic_repulsion(nuclei: &[Nucleus], setup: &IonicSetup) -> Result<f64, QmError>;        // Σ Z_v,I Z_v,J / R
pub fn points_within(grid: &Grid, centre: [f64; 3], radius: f64) -> Vec<usize>;
// scf.rs
pub struct System { pub nuclei: Vec<Nucleus>, pub n_up: usize, pub n_dn: usize, pub ionic: Option<Arc<IonicSetup>> }
impl System { pub fn with_ionic(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>, ionic: Arc<IonicSetup>) -> System; }
pub fn valence_electrons(nuclei: &[Nucleus], ionic: Option<&IonicSetup>, charge: i32) -> i64;
pub struct KsState { /* … unchanged fields …, */ pub ionic: Option<Arc<IonicSetup>> }
pub struct ScfStats { /* … unchanged fields …, */ pub core_overlaps: Vec<(usize, usize, f64)> }
```
**Meaning of `System.ionic`:**
- `None` is today's all-electron solver, the same code path bit for bit. `System::new` gives `None`, and providers use it for `IonicPolicy::None`.
- `Some(setup)` turns on the ionic path. That path asserts `setup.h == grid.h` and `setup.sigma_ae == settings.sigma`. An `AllElectron` species inside a setup is today's smeared nucleus at `sigma_ae`.
- The `Energy` struct is unchanged. `external` = ∫ρ V_loc + E_NL, the whole ion–electron energy; `nuclear` = E_ii with Z_v; the doc says so.

- [ ] **Step 0: Merge the finished M6.** Confirm `m6-tools` has landed on `main` with its final fix wave: `git log main --oneline | grep -m1 "m6-tools"`. Then run:
```bash
git merge main
```
Run the merge alone, never piped. Then run `cargo build --workspace` and `cargo test -p qm --release`. Expected: PASS, and Tasks 1–5's tests still pass.

- [ ] **Step 1: Pin today's all-electron bits (before touching `scf.rs`).** Add this test to `crates/qm/src/scf.rs` tests and run it on the unchanged code:
```rust
/// Prints the bits Task 6's `none_policy_is_bitwise_today` pins (run once, before the M7 change).
#[test]
#[ignore = "M7 Task 6 Step 1 only"]
fn none_policy_baseline() {
    let g = Grid::cube(33, 0.25);
    let s = ScfSettings::for_grid(&g);
    for (name, sys) in [("H2", System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None)),
                        ("LiH", System::new(vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None))] {
        let (st, stats) = scf(&g, &sys, &s, None).unwrap();
        let f = crate::forces(&g, &st, &sys.nuclei, s.sigma).unwrap();
        println!("const BASE_{name}_E: u64 = 0x{:016x};", stats.energy.total.to_bits());
        println!("const BASE_{name}_FX: u64 = 0x{:016x}; // {} SCF steps", f[0][0].to_bits(), stats.iterations);
    }
}
```
Run: `cargo test -p qm --release none_policy_baseline -- --ignored --nocapture`. It prints four lines, `const BASE_H2_E: u64 = 0x…;`, `const BASE_H2_FX`, `const BASE_LiH_E` and `const BASE_LiH_FX`: today's bits, measured, not invented. Step 2 pastes them verbatim.

- [ ] **Step 2: Write the failing tests**

`crates/qm/src/nonlocal.rs` tests:
```rust
use super::*;
use crate::grid::Grid;
use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
use crate::nuclear::Nucleus;
use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
use crate::radial::{filter::FilteredTable, generate};
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

fn o_setup(ft: FilteredTable) -> IonicSetup {
    let h = ft.h;
    IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(Arc::new(ft)))]))
}

/// Spec §8 row 6 first failing test (review S1): on the grid, V_NL applied to the sampled reference
/// pseudo-orbital φ_l0 = (u_l/d)·Y_l0 equals δV_l φ_l0 pointwise to 1e-6 of max|δV φ|, and ⟨φ|V_NL|φ⟩
/// equals ⟨φ|δV|φ⟩ — which catches a missing √((2l+1)/4π) (a factor 4π/(2l+1) ≥ 4.19 in V_NL).
/// Separately, the grid quadrature of the projection: |E_KB⟨β|φ⟩_grid − 1| < 1e-4. Plan review S8: an
/// absolute pointwise 1e-6 mixed that quadrature floor (the unfiltered δV has C² kinks at r_c, aliasing
/// ~1e-7 relative at |G| = 2π/0.1, while max|δVφ| is several Ha for O s) into an operator-
/// normalisation test; the relative bound keeps the discrimination. Unfiltered table on a fine grid
/// (h = 0.1); O off a grid point.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
fn v_nl_on_grid_is_the_operator_identity() {
    let t = generate(8).unwrap();
    let h = 0.1;
    let setup = o_setup(FilteredTable::unfiltered(&t, h));
    let grid = Grid::cube(61, h);
    let nuc = [Nucleus { pos: [0.013, -0.021, 0.017], z: 8 }];
    let nl = NonlocalOp::build(&grid, &nuc, &setup);
    assert!(!nl.is_empty(), "O has a projector");
    for p in &t.kb.projectors {
        let y = ((2 * p.l + 1) as f64 / (4.0 * PI)).sqrt();
        let at = |r: [f64; 3]| { let d = [r[0] - nuc[0].pos[0], r[1] - nuc[0].pos[1], r[2] - nuc[0].pos[2]]; let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12); (d, dd) };
        let phi = grid.sample(|r| { let (d, dd) = at(r); let mut sh = [0.0; N_MOMENTS]; solid_harmonics(d, &mut sh);
                                     t.grid.interp(&p.u, dd) / dd.powi(p.l as i32 + 1) * y * sh[lm_index(p.l as usize, 0)] });
        let expect: Vec<f64> = (0..grid.len()).map(|i| { let (_, dd) = at(grid.point(i)); t.grid.interp(&p.dv, dd) * phi[i] }).collect();
        let mut got = vec![0.0; grid.len()];
        nl.apply_add(&phi, &mut got);
        let worst = got.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let scale = expect.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(worst <= 1e-6 * scale, "l = {}: max |V_NL φ − δV φ| = {worst:.3e} of max |δV φ| {scale:.3e}", p.l);
        let (lhs, rhs) = (grid.dot(&phi, &got), grid.dot(&phi, &expect));
        assert!((lhs - rhs).abs() < 1e-6 * rhs.abs(), "l = {}: ⟨φ|V_NL|φ⟩ {lhs:.9} vs ⟨φ|δV|φ⟩ {rhs:.9}", p.l);
        // the projection's own quadrature: E_KB·⟨β_l0|φ_l0⟩_grid = 1 (radially ⟨β|φ⟩ = 1/E_KB exactly)
        let k = nl.spheres.iter().position(|s| s.l == p.l).expect("a sphere for this projector");
        let ekb_c = nl.spheres[k].e_kb * nl.project(&phi)[k][p.l as usize];   // m = 0 sits at index l
        assert!((ekb_c - 1.0).abs() < 1e-4, "l = {}: E_KB⟨β|φ⟩_grid = {ekb_c:.8}", p.l);
    }
}

/// Review Focus 3: a sphere clipped by the box face keeps V_NL symmetric (the clipped points carry
/// ψ = 0 under the Dirichlet wall, so the clipped operator is the exact restriction).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
fn projector_sphere_clipped_by_the_box_stays_symmetric() {
    let t = generate(8).unwrap();
    let h = t.record.h_req.unwrap();
    let setup = o_setup(crate::radial::filter::filtered(&t, h));
    let grid = Grid::cube(33, h);
    let edge = grid.origin[0] + 0.4;                         // 0.4 Bohr inside the −x face: the sphere is clipped
    let nl = NonlocalOp::build(&grid, &[Nucleus { pos: [edge, 0.1, -0.2], z: 8 }], &setup);
    let mut seed = 0x5EEDu64;
    let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
    let a: Vec<f64> = (0..grid.len()).map(|_| rnd()).collect();
    let b: Vec<f64> = (0..grid.len()).map(|_| rnd()).collect();
    let (mut na, mut nb) = (vec![0.0; grid.len()], vec![0.0; grid.len()]);
    nl.apply_add(&a, &mut na);
    nl.apply_add(&b, &mut nb);
    let (x, y) = (grid.dot(&a, &nb), grid.dot(&na, &b));
    assert!((x - y).abs() <= 1e-12 * x.abs().max(y.abs()), "⟨a|V_NL b⟩ {x} vs ⟨V_NL a|b⟩ {y}");
}
```
`crates/qm/src/scf.rs` tests (added):
```rust
// ← paste here, verbatim, the four `const BASE_…: u64 = 0x…;` lines Step 1 printed (H2 and LiH, E and FX)
```
```rust
/// Global constraint: IonicPolicy::None is today's solver, bit for bit (energies and forces).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
fn none_policy_is_bitwise_today() {
    let g = Grid::cube(33, 0.25);
    let s = ScfSettings::for_grid(&g);
    for (sys, e_bits, f_bits) in [
        (System::new(vec![Nucleus { pos: [0.7, 0.0, 0.0], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 1 }], 0, None), BASE_H2_E, BASE_H2_FX),
        (System::new(vec![Nucleus { pos: [-0.8, 0.1, 0.0], z: 3 }, Nucleus { pos: [2.2, 0.0, 0.0], z: 1 }], 0, None), BASE_LiH_E, BASE_LiH_FX),
    ] {
        assert!(sys.ionic.is_none());
        let (st, stats) = scf(&g, &sys, &s, None).unwrap();
        let f = crate::forces(&g, &st, &sys.nuclei, s.sigma).unwrap();
        assert_eq!(stats.energy.total.to_bits(), e_bits);
        assert_eq!(f[0][0].to_bits(), f_bits);
        assert!(stats.core_overlaps.is_empty() && st.ionic.is_none());
    }
}

/// Review Focus 2: a warm state converged under another ionic setup (H under None, then H under All —
/// same grid, same orbital counts) is ignored: the solve equals a cold one bit for bit.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
fn warm_state_from_another_ionic_setup_is_not_reused() {
    use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
    use crate::radial::{filter::filtered, generate};
    let h = 0.25;
    let g = Grid::cube(33, h);
    let s = ScfSettings::for_grid(&g);
    let nuc = vec![Nucleus { pos: [0.1, 0.0, 0.0], z: 1 }];
    let (ae_state, _) = scf(&g, &System::new(nuc.clone(), 0, None), &s, None).unwrap();
    let ft = filtered(&generate(1).unwrap(), h);
    let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::All, h, h, std::collections::BTreeMap::from([(1, Ionic::Pseudo(std::sync::Arc::new(ft)))])));
    let sys = System::with_ionic(nuc, 0, None, setup);
    let (_, warm) = scf(&g, &sys, &s, Some(&ae_state)).unwrap();
    let (_, cold) = scf(&g, &sys, &s, None).unwrap();
    assert_eq!(warm.energy.total.to_bits(), cold.energy.total.to_bits());
    assert_eq!(warm.iterations, cold.iterations);
}

/// Review Focus 3: an ion exactly on a grid point gives finite fields and a converged SCF. Solved at
/// tol = eig_tol = 1e-9 (plan review S6), because Task 7 extends this test with the force, whose
/// zero by inversion symmetry holds only to the eigensolver residual.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
fn ion_on_a_grid_point_is_finite() {
    use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
    use crate::radial::{filter::filtered, generate};
    let t = generate(8).unwrap();
    let h = t.record.h_req.unwrap();
    let n = (2.0 * (6.0 / h).round()) as usize + 1;         // a ≈ 12 Bohr box with a grid point at the origin
    let g = Grid::cube(n, h);
    let setup = std::sync::Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(std::sync::Arc::new(filtered(&t, h))))])));
    let nuc = vec![Nucleus { pos: [0.0; 3], z: 8 }];
    assert!(crate::nuclear::ionic_local_potential(&g, &nuc, &setup).iter().all(|v| v.is_finite()));
    assert!(crate::nuclear::partial_core_field(&g, &nuc, &setup).iter().all(|v| v.is_finite()));
    let sys = System::with_ionic(nuc, 0, Some(3), setup);
    let mut s = ScfSettings::for_grid(&g);
    (s.tol, s.eig_tol) = (1e-9, 1e-9);
    let (_st, stats) = scf(&g, &sys, &s, None).expect("converges");
    assert!(stats.energy.total.is_finite() && sys.n_up == 4 && sys.n_dn == 2);
}
```
`crates/validate/tests/pseudo.rs` (new suite, CPU). Its module doc gives the protocol: the tables come from `qm::radial` (cached per binary); `IonicSetup::build` with the real `has_core`/`filtered` closures; ScfSettings tol 1e-6 / eig 1e-7 unless stated; release-only. Its `common` helpers live in the file:
```rust
use qm::ionic::{IonicPolicy, IonicSetup};
use qm::radial::{filter::filtered, generate, has_core, PseudoTable};
use qm::{scf, Grid, Nucleus, ScfSettings, System};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn table(z: u8) -> Arc<PseudoTable> {
    static T: OnceLock<Mutex<HashMap<u8, Arc<PseudoTable>>>> = OnceLock::new();
    T.get_or_init(Default::default).lock().unwrap().entry(z).or_insert_with(|| Arc::new(generate(z).expect("generates"))).clone()
}

pub fn setup(policy: IonicPolicy, h: f64, zs: &[u8]) -> Arc<IonicSetup> {
    Arc::new(IonicSetup::build(policy, h, h, zs, &mut |z| has_core(z), &mut |z| Ok(Arc::new(filtered(&table(z), h)))).expect("setup builds"))
}

/// The odd n giving a box of ≈ `box_bohr` at spacing h (a grid point at the origin).
fn n_for(box_bohr: f64, h: f64) -> usize { 2 * (box_bohr / (2.0 * h)).round() as usize + 1 }

fn solve(h: f64, box_bohr: f64, policy: IonicPolicy, nuclei: Vec<Nucleus>, mult: Option<u32>) -> (f64, qm::KsState, qm::ScfStats) {
    let g = Grid::cube(n_for(box_bohr, h), h);
    let zs: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
    let sys = System::with_ionic(nuclei, 0, mult, setup(policy, h, &zs));
    let mut s = ScfSettings::for_grid(&g);
    s.eig_tol = 1e-7;
    let (st, stats) = scf(&g, &sys, &s, None).expect("SCF converges");
    (stats.energy.total, st, stats)
}

/// Spec §6.3: the pseudo-Ne atom (closed shell, spherical) on the 3-D grid at h = 0.15 reproduces the
/// radial KB-PP atom (filtered at the same h) to < 2 mHa.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn pseudo_ne_3d_matches_radial() {
    let h = 0.15;
    let ft = filtered(&table(10), h);
    let radial = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &qm::radial::kb::reference_config(&table(10).partition), qm::radial::SpinMode::Restricted).unwrap();
    let (e3d, _, stats) = solve(h, 12.0, IonicPolicy::CoresOnly, vec![Nucleus { pos: [0.0; 3], z: 10 }], Some(1));
    eprintln!("pseudo-Ne: E_3D {e3d:.6} vs radial {:.6} (Δ {:+.2e}), {} SCF steps", radial.energy, e3d - radial.energy, stats.iterations);
    assert!((e3d - radial.energy).abs() < 2e-3);
}

/// Spec §6.3 / §6.4: with pseudised O, Hund's rule emerges — the triplet lies > 20 mHa below the
/// singlet — with *converged* open-shell solves (M6 Task 3 found the all-electron ones unconverged at
/// the 200-step cap; an unconverged solve here is an error, not a pass).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn pseudo_o_triplet_below_singlet() {
    let h = table(8).record.h_req.expect("O has h_req").min(0.20);
    let o = || vec![Nucleus { pos: [0.0; 3], z: 8 }];
    let (e1, _, s1) = solve(h, 12.0, IonicPolicy::CoresOnly, o(), Some(1));
    let (e3, _, s3) = solve(h, 12.0, IonicPolicy::CoresOnly, o(), Some(3));
    eprintln!("pseudo-O at h = {h}: singlet {e1:.6} ({} steps), triplet {e3:.6} ({} steps), splitting {:.2} mHa", s1.iterations, s3.iterations, (e1 - e3) * 1e3);
    assert!(e3 < e1 - 0.020, "triplet {e3:.6} vs singlet {e1:.6}");
}

/// Spec §6.6 attribution (Task 5 physics review S2). The 3-D egg-box of one pseudised atom at its h_req (capped at 0.25):
/// - max |F| over the 8 offsets {±h/4}³, the gate's own probe (one atom, so |ΣF| = |F|);
/// - the energy spread over {0, h/2}³ (plan review M4);
/// - the record's T8(d) F_atom at that h, printed beside it.
/// The valence floor is ≥ |F| − F_atom. Reported and recorded in references.md; Task 7 reads it before the gate.
/// O (CoresOnly, triplet) and H (All) here, on the CPU. C and N, and h = 0.15 for all four, are measured on the GPU
/// in Task 10.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn single_atom_eggbox_floor() {
    for (z, mult, policy) in [(8u8, Some(3), IonicPolicy::CoresOnly), (1, Some(2), IonicPolicy::All)] {
        let t = table(z);
        let h = t.record.h_req.expect("an M7 species has h_req").min(0.25);
        let f_atom: f64 = t.record.filter.iter().find(|r| (r.h - h).abs() < 1e-12).expect("ladder row").f_table.iter().sum();
        let (q, half) = (h / 4.0, h / 2.0);
        let mut fmax = 0.0f64;
        for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
            let nuc = vec![Nucleus { pos: [sx, sy, sz], z }];
            let (_, st, _) = solve(h, 12.0, policy, nuc.clone(), mult);
            let g = Grid::cube(n_for(12.0, h), h);
            let f = qm::forces(&g, &st, &nuc, h).unwrap()[0];
            fmax = fmax.max((f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt());
        }}}
        let mut es = Vec::new();
        for sx in [0.0, half] { for sy in [0.0, half] { for sz in [0.0, half] {
            es.push(solve(h, 12.0, policy, vec![Nucleus { pos: [sx, sy, sz], z }], mult).0);
        }}}
        let spread = es.iter().cloned().fold(f64::MIN, f64::max) - es.iter().cloned().fold(f64::MAX, f64::min);
        eprintln!("Z = {z} at h = {h}: max |F| {fmax:.2e} Ha/Bohr over ±h/4 (tables F_atom {f_atom:.2e}; floor ≥ {:.2e}), E spread {spread:.2e} Ha over {{0, h/2}}³",
                  (fmax - f_atom).max(0.0));
        assert!(fmax.is_finite() && spread.is_finite());
    }
}
```

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test -p qm --release` and `cargo test -p validate --release --test pseudo`
Expected: compile errors (`qm::ionic`, `qm::nonlocal`, `System::with_ionic` missing).

- [ ] **Step 4: Implement `qm::ionic`**

`crates/qm/src/ionic.rs`. Its module doc covers:
- spec §4.7 (the three policies);
- that the policy is scene-wide and which species carry a potential is an output;
- the build rules: `None` is all-electron throughout. `CoresOnly` pseudises a species iff `has_core`; a refusal there falls back to all-electron with a note. `All` pseudises every species, and a refusal is `IonicError::Refused`. Both pseudo policies require h ≤ h_req and the filtered T1/T3 re-run at h (S2) to pass, else `TooCoarse` / `FilteredGate`;
- §5.6 (r_core and the overlap rule, M3).

```rust
use crate::error::QmError;
use crate::nuclear::Nucleus;
use crate::radial::filter::FilteredTable;
use crate::radial::ledger::{recipe_hash, LEDGER};
use crate::radial::Refusal;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IonicPolicy { None, CoresOnly, All }

#[derive(Clone, Debug)]
pub enum Ionic { AllElectron { sigma: f64 }, Pseudo(Arc<FilteredTable>) }

#[derive(Clone, Debug)]
pub struct IonicSetup { pub policy: IonicPolicy, pub h: f64, pub sigma_ae: f64, pub recipe_hash: u64, pub per_z: BTreeMap<u8, Ionic>, pub notes: Vec<String> }

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum IonicError {
    #[error("{} is refused by the pseudopotential recipe (IonicPolicy::All has no all-electron fallback): {refusal}", units::elements::symbol(*z))]
    Refused { z: u8, refusal: Refusal },
    #[error("h = {h} Bohr is coarser than {}'s pseudopotential allows (h_req = {}): choose a finer grid", units::elements::symbol(*z), h_req.map_or("none on the ladder".to_string(), |x| format!("{x} Bohr")))]
    TooCoarse { z: u8, h: f64, h_req: Option<f64> },
    #[error("{}'s filtered pseudopotential fails the filtered gates at h = {h} Bohr: {why}", units::elements::symbol(*z))]
    FilteredGate { z: u8, h: f64, why: String },
}

pub const CORE_OVERLAP_WARNING: &str = "cores overlap";

fn check(z: u8, t: &FilteredTable, h: f64) -> Result<(), IonicError> {
    assert!((t.h - h).abs() < 1e-15, "a table filtered at h = {} used at h = {h}", t.h);
    let finest = LEDGER.h_ladder.iter().cloned().fold(f64::INFINITY, f64::min);
    match t.record.h_req {
        Some(hr) if h <= hr + 1e-12 => {}
        // finer than the ladder: certified by this h's own filtered re-run below (plan review Q11)
        None if h < finest - 1e-12 => {}
        other => return Err(IonicError::TooCoarse { z, h, h_req: other }),
    }
    let c = &t.check;
    if !c.passes() {
        return Err(IonicError::FilteredGate { z, h, why: format!(
            "ΔE {:.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, XC clamp at {} points, table egg-box force {:.2e} Ha/Bohr",
            c.de * 1e3, c.max_eig * 1e3, c.t3_max * 1e3, c.clamped, c.f_table.iter().sum::<f64>()) });
    }
    Ok(())
}

impl IonicSetup {
    pub fn build(policy: IonicPolicy, h: f64, sigma_ae: f64, zs: &[u8],
                 has_core: &mut dyn FnMut(u8) -> Result<bool, Refusal>,
                 table: &mut dyn FnMut(u8) -> Result<Arc<FilteredTable>, Refusal>) -> Result<IonicSetup, IonicError> {
        let mut species: Vec<u8> = zs.to_vec();
        species.sort_unstable();
        species.dedup();
        let (mut per_z, mut notes) = (BTreeMap::new(), Vec::new());
        let ae = Ionic::AllElectron { sigma: sigma_ae };
        for z in species {
            let sym = units::elements::symbol(z);
            let ionic = match policy {
                IonicPolicy::None => ae.clone(),
                IonicPolicy::CoresOnly => match has_core(z) {
                    Ok(false) => ae.clone(),
                    Ok(true) => match table(z) {
                        Ok(t) => { check(z, &t, h)?; Ionic::Pseudo(t) }
                        Err(r) => { notes.push(format!("{sym}: {r} — kept all-electron (σ = {sigma_ae})")); ae.clone() }
                    },
                    Err(r) => { notes.push(format!("{sym}: {r} — kept all-electron (σ = {sigma_ae})")); ae.clone() }
                },
                IonicPolicy::All => match table(z) {
                    Ok(t) => { check(z, &t, h)?; Ionic::Pseudo(t) }
                    Err(refusal) => return Err(IonicError::Refused { z, refusal }),
                },
            };
            per_z.insert(z, ionic);
        }
        Ok(IonicSetup { policy, h, sigma_ae, recipe_hash: recipe_hash(), per_z, notes })
    }

    pub fn from_tables(policy: IonicPolicy, h: f64, sigma_ae: f64, per_z: BTreeMap<u8, Ionic>) -> IonicSetup {
        IonicSetup { policy, h, sigma_ae, recipe_hash: recipe_hash(), per_z, notes: Vec::new() }
    }

    pub fn ionic(&self, z: u8) -> &Ionic { self.per_z.get(&z).unwrap_or_else(|| panic!("Z = {z} is not in this ionic setup")) }

    pub fn valence_charge(&self, z: u8) -> f64 {
        match self.ionic(z) { Ionic::AllElectron { .. } => z as f64, Ionic::Pseudo(t) => t.z_v }
    }

    pub fn pseudised(&self) -> Vec<u8> { self.per_z.iter().filter(|(_, i)| matches!(i, Ionic::Pseudo(_))).map(|(z, _)| *z).collect() }

    /// Same policy, recipe, h, σ_AE and the same species with the same treatment: a KsState converged
    /// under one is a valid warm start for the other (the setup-equality guard of spec §5.5).
    pub fn same_physics(&self, o: &IonicSetup) -> bool {
        self.policy == o.policy && self.h == o.h && self.sigma_ae == o.sigma_ae && self.recipe_hash == o.recipe_hash
            && self.per_z.len() == o.per_z.len()
            && self.per_z.iter().zip(&o.per_z).all(|((za, a), (zb, b))| za == zb && std::mem::discriminant(a) == std::mem::discriminant(b))
    }

    fn r_core(&self, z: u8) -> f64 { match self.ionic(z) { Ionic::Pseudo(t) => t.r_core, Ionic::AllElectron { .. } => 0.0 } }
}

/// Spec §5.6: pairs (I, J, R_IJ) with R_IJ < r_core,I + r_core,J (0 for all-electron species).
pub fn core_overlaps(setup: &IonicSetup, nuclei: &[Nucleus]) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..nuclei.len() {
        for j in i + 1..nuclei.len() {
            let d = (0..3).map(|a| (nuclei[i].pos[a] - nuclei[j].pos[a]).powi(2)).sum::<f64>().sqrt();
            if d < setup.r_core(nuclei[i].z) + setup.r_core(nuclei[j].z) { out.push((i, j, d)); }
        }
    }
    out
}

#[allow(dead_code)]
fn _qm_error_is_used(_: QmError) {}
```
(Drop the last two lines if `QmError` is not otherwise needed. They exist only so the `use` list compiles while the file is written step by step.)

- [ ] **Step 5: Implement the sphere lists, the local fields and `NonlocalOp`**

In `crates/qm/src/nuclear.rs` (the module doc gains a section "Ionic species (M7)": V_loc = Σ_I [−q_I erf(d/(σ_I√2))/d] + Σ_pseudo V_SR, where q_I = Z_v and σ_I = σ_LR for pseudised species and q_I = Z, σ_I = σ_AE for all-electron ones; the exact split with the one V_SR table, sampled once per geometry within its radius R_SR·γ_r + h; Spec questions 2 and 6):
```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Centre { pub pos: [f64; 3], pub charge: f64, pub sigma: f64 }

/// −Σ q erf(d/(σ√2))/d, with the d → 0 limit −q√(2/π)/σ: `nuclear_potential` with per-centre q and σ.
pub fn smeared_potential(grid: &Grid, centres: &[Centre]) -> Vec<f64> {
    grid.sample(|r| centres.iter().map(|c| {
        let d = dist2(r, c.pos).sqrt();
        if d < ON_SITE { -c.charge * (2.0 / PI).sqrt() / c.sigma } else { -c.charge * libm::erf(d / (c.sigma * 2f64.sqrt())) / d }
    }).sum())
}

/// Σ q (2πσ²)^{−3/2} e^{−d²/2σ²}: the cold guess's blobs with valence charges.
pub fn deposit_centres(grid: &Grid, centres: &[Centre]) -> Vec<f64> {
    grid.sample(|r| centres.iter().map(|c| c.charge / (2.0 * PI * c.sigma * c.sigma).powf(1.5) * (-dist2(r, c.pos) / (2.0 * c.sigma * c.sigma)).exp()).sum())
}

pub fn centres(nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<Centre> {
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { sigma } => Centre { pos: n.pos, charge: n.z as f64, sigma: *sigma },
        Ionic::Pseudo(t) => Centre { pos: n.pos, charge: t.z_v, sigma: t.sigma_lr },
    }).collect()
}

/// Grid points within `radius` of `centre` (bounding box of indices, clipped to the box; points
/// outside the box do not exist — ψ = 0 there under the Dirichlet wall).
pub fn points_within(grid: &Grid, centre: [f64; 3], radius: f64) -> Vec<usize> {
    let lo = |a: usize| (((centre[a] - radius - grid.origin[a]) / grid.h).floor().max(0.0)) as usize;
    let hi = |a: usize| ((((centre[a] + radius - grid.origin[a]) / grid.h).ceil()) as isize).clamp(-1, grid.n[a] as isize - 1);
    let mut out = Vec::new();
    let (h0, h1, h2) = (hi(0), hi(1), hi(2));
    if h0 < 0 || h1 < 0 || h2 < 0 { return out; }
    for k in lo(2)..=h2 as usize { for j in lo(1)..=h1 as usize { for i in lo(0)..=h0 as usize {
        let p = grid.idx(i, j, k);
        if dist2(grid.point(p), centre) <= radius * radius { out.push(p); }
    }}}
    out
}

pub fn ionic_local_potential(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<f64> {
    let mut v = smeared_potential(grid, &centres(nuclei, setup));
    for n in nuclei {
        if let Ionic::Pseudo(t) = setup.ionic(n.z) {
            for p in points_within(grid, n.pos, t.sr_radius()) { v[p] += t.v_sr(dist2(grid.point(p), n.pos).sqrt()); }
        }
    }
    v
}

pub fn partial_core_field(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<f64> {
    let mut rho = vec![0.0; grid.len()];
    for n in nuclei {
        if let Ionic::Pseudo(t) = setup.ionic(n.z) {
            for p in points_within(grid, n.pos, t.pc_radius) { rho[p] += t.rho_pc_at(dist2(grid.point(p), n.pos).sqrt()); }
        }
    }
    rho
}

/// Σ_{I<J} q_I q_J / R_IJ with valence charges for pseudised species; coincident nuclei are the
/// same error as `nuclear_repulsion`.
pub fn ionic_repulsion(nuclei: &[Nucleus], setup: &IonicSetup) -> Result<f64, QmError> {
    let mut e = 0.0;
    for i in 0..nuclei.len() {
        for j in i + 1..nuclei.len() {
            let d = dist2(nuclei[i].pos, nuclei[j].pos).sqrt();
            if d < COINCIDENCE { return Err(QmError::CoincidentNuclei(i, j)); }
            e += setup.valence_charge(nuclei[i].z) * setup.valence_charge(nuclei[j].z) / d;
        }
    }
    Ok(e)
}
```
`crates/qm/src/nonlocal.rs`. Its module doc covers:
- spec §5.3 (the operator, the cost);
- the S1 normalisation β_lm = t_l(d)·√((2l+1)/4π)·R_lm(d), with R_lm from `qm::poisson` (Racah: ∫|S_lm|²dΩ = 4π/(2l+1)), so √((2l+1)/4π)·S_lm is the orthonormal real harmonic;
- precision: c_lm by Neumaier in f64;
- symmetry under clipping (Review Focus 3).

```rust
use crate::eigen::Operator;
use crate::grid::{neumaier, Grid};
use crate::ionic::{Ionic, IonicSetup};
use crate::nuclear::{points_within, Nucleus};
use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
use crate::stencil::kinetic;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct Sphere { pub ion: usize, pub z: u8, pub l: u8, pub points: Vec<usize>, pub beta: Vec<Vec<f64>>, pub e_kb: f64 }

#[derive(Clone, Debug, Default)]
pub struct NonlocalOp { pub spheres: Vec<Sphere>, pub dv: f64 }

impl NonlocalOp {
    pub fn build(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> NonlocalOp {
        let mut spheres = Vec::new();
        for (ion, n) in nuclei.iter().enumerate() {
            let Ionic::Pseudo(t) = setup.ionic(n.z) else { continue };
            for (k, p) in t.projectors.iter().enumerate() {
                assert!(p.l <= 2, "projectors up to l = 2 (Z ≤ 18; the gradients of Task 7 are explicit to l = 2)");
                let points = points_within(grid, n.pos, p.radius);
                let norm = ((2 * p.l + 1) as f64 / (4.0 * PI)).sqrt();
                let beta = (-(p.l as i32)..=(p.l as i32)).map(|m| points.iter().map(|&q| {
                    let r = grid.point(q);
                    let d = [r[0] - n.pos[0], r[1] - n.pos[1], r[2] - n.pos[2]];
                    let mut sh = [0.0; N_MOMENTS];
                    solid_harmonics(d, &mut sh);
                    t.beta_radial(k, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()) * norm * sh[lm_index(p.l as usize, m)]
                }).collect()).collect();
                spheres.push(Sphere { ion, z: n.z, l: p.l, points, beta, e_kb: p.e_kb });
            }
        }
        NonlocalOp { spheres, dv: grid.dv() }
    }

    pub fn is_empty(&self) -> bool { self.spheres.is_empty() }

    pub fn project(&self, psi: &[f64]) -> Vec<Vec<f64>> {
        self.spheres.iter().map(|s| s.beta.iter().map(|b| neumaier(s.points.iter().zip(b).map(|(&p, &bv)| bv * psi[p])) * self.dv).collect()).collect()
    }

    pub fn apply_add(&self, psi: &[f64], out: &mut [f64]) {
        for (s, c) in self.spheres.iter().zip(self.project(psi)) {
            for (b, cm) in s.beta.iter().zip(c) {
                let f = s.e_kb * cm;
                for (&p, &bv) in s.points.iter().zip(b) { out[p] += f * bv; }
            }
        }
    }

    pub fn energy(&self, orbitals: &[Vec<f64>]) -> f64 {
        neumaier(orbitals.iter().flat_map(|psi| self.spheres.iter().zip(self.project(psi)).flat_map(|(s, c)| c.into_iter().map(move |cm| s.e_kb * cm * cm)).collect::<Vec<_>>()))
    }
}

/// H = −½∇² + V + V_NL.
pub struct PseudoHamiltonian<'a> { pub grid: &'a Grid, pub v: &'a [f64], pub nl: &'a NonlocalOp }

impl Operator for PseudoHamiltonian<'_> {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        kinetic(self.grid, x, y);
        for i in 0..y.len() { y[i] += self.v[i] * x[i]; }
        self.nl.apply_add(x, y);
    }
}
```

- [ ] **Step 6: Integrate into the SCF**

In `crates/qm/src/scf.rs` (module doc: a new section "Ionic species (M7)" with spec §5.2's Hamiltonian and energy, the warm-start guard, and the `None` ⇒ unchanged guarantee):
```rust
use crate::ionic::{core_overlaps, IonicSetup, CORE_OVERLAP_WARNING};
use crate::nonlocal::{NonlocalOp, PseudoHamiltonian};
use crate::nuclear::{centres, deposit_centres, ionic_local_potential, ionic_repulsion, partial_core_field};
use crate::xc::XcResult;
use std::sync::Arc;

pub struct System { pub nuclei: Vec<Nucleus>, pub n_up: usize, pub n_dn: usize, pub ionic: Option<Arc<IonicSetup>> }

fn spin_split(total: i64, multiplicity: Option<u32>) -> (usize, usize) {
    assert!(total >= 0, "negative electron count");
    let n = total as usize;
    let unpaired = match multiplicity {
        Some(m) => { assert!(m >= 1 && (m as usize - 1) <= n && (n - (m as usize - 1)) % 2 == 0, "bad multiplicity"); m as usize - 1 }
        None => n % 2,
    };
    let n_dn = (n - unpaired) / 2;
    (n_dn + unpaired, n_dn)
}

/// Σ q_I − charge: q = Z for all-electron species (and without a setup), Z_v for pseudised ones.
pub fn valence_electrons(nuclei: &[Nucleus], ionic: Option<&IonicSetup>, charge: i32) -> i64 {
    nuclei.iter().map(|n| ionic.map_or(n.z as f64, |s| s.valence_charge(n.z)).round() as i64).sum::<i64>() - charge as i64
}

impl System {
    pub fn new(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>) -> System {
        let (n_up, n_dn) = spin_split(valence_electrons(&nuclei, None, charge), multiplicity);
        System { nuclei, n_up, n_dn, ionic: None }
    }
    /// N = Σ Z_v − charge (spec §5.2). Panics like `new` on an impossible count (providers pre-validate).
    pub fn with_ionic(nuclei: Vec<Nucleus>, charge: i32, multiplicity: Option<u32>, ionic: Arc<IonicSetup>) -> System {
        let (n_up, n_dn) = spin_split(valence_electrons(&nuclei, Some(&ionic), charge), multiplicity);
        System { nuclei, n_up, n_dn, ionic: Some(ionic) }
    }
}

/// The ionic terms of one geometry: `None` reproduces today's arrays exactly.
struct IonicTerms { v_loc: Vec<f64>, rho_pc: Option<Vec<f64>>, nl: Option<NonlocalOp>, e_nn: f64, overlaps: Vec<(usize, usize, f64)> }

impl IonicTerms {
    fn new(grid: &Grid, system: &System, settings: &ScfSettings) -> Result<IonicTerms, QmError> {
        match system.ionic.as_deref() {
            None => {
                let e_nn = nuclear_repulsion(&system.nuclei)?;
                Ok(IonicTerms { v_loc: nuclear_potential(grid, &system.nuclei, settings.sigma), rho_pc: None, nl: None, e_nn, overlaps: Vec::new() })
            }
            Some(s) => {
                assert!(s.sigma_ae == settings.sigma, "IonicSetup σ_AE {} ≠ settings σ {}", s.sigma_ae, settings.sigma);
                assert!(s.h == grid.h, "IonicSetup h {} ≠ grid h {}", s.h, grid.h);
                let e_nn = ionic_repulsion(&system.nuclei, s)?;
                let nl = NonlocalOp::build(grid, &system.nuclei, s);
                Ok(IonicTerms {
                    v_loc: ionic_local_potential(grid, &system.nuclei, s),
                    rho_pc: Some(partial_core_field(grid, &system.nuclei, s)),
                    nl: (!nl.is_empty()).then_some(nl), e_nn, overlaps: core_overlaps(s, &system.nuclei),
                })
            }
        }
    }
    /// LSDA on ρ_σ + ½ρ_pc per spin (the core is unpolarised; spec §4.5, §5.2).
    fn xc(&self, up: &[f64], dn: &[f64]) -> XcResult {
        match &self.rho_pc {
            None => lsda_pw92(up, dn),
            Some(pc) => {
                let u: Vec<f64> = up.iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
                let d: Vec<f64> = dn.iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
                lsda_pw92(&u, &d)
            }
        }
    }
}

fn same_ionic(a: Option<&IonicSetup>, b: Option<&IonicSetup>) -> bool {
    match (a, b) { (None, None) => true, (Some(x), Some(y)) => x.same_physics(y), _ => false }
}

fn solve_channel(grid: &Grid, v_eff: &[f64], nl: Option<&NonlocalOp>, orbitals: &mut Vec<Vec<f64>>, evals: &mut Vec<f64>, eig_tol: f64) -> Option<EigenStats> {
    if orbitals.is_empty() { return None; }
    let lambda = evals.first().copied().unwrap_or(-1.0);
    let pc = jacobi_precond(grid, v_eff, lambda);
    Some(match nl {
        None => lobpcg(grid, &LocalHamiltonian { grid, v: v_eff }, orbitals, evals, Some(&pc), eig_tol, 200),
        Some(nl) => lobpcg(grid, &PseudoHamiltonian { grid, v: v_eff, nl }, orbitals, evals, Some(&pc), eig_tol, 200),
    })
}
```
`KsState` gains `pub ionic: Option<Arc<IonicSetup>>` and `ScfStats` gains `pub core_overlaps: Vec<(usize, usize, f64)>`. In `scf`:
- `let terms = IonicTerms::new(grid, system, settings)?;` replaces the `e_nn`/`v_nuc` lines; `terms.v_loc` is used where `v_nuc` was.
- The warm arm adds `&& same_ionic(w.ionic.as_deref(), system.ionic.as_deref())`.
- The cold arm keeps `deposit_charge(grid, &system.nuclei, 1.0)` and `z_tot = Σ Z` when `system.ionic.is_none()`. With a setup it uses `deposit_centres(grid, &centres(nuclei, s) with sigma = 1.0)` and `z_tot = Σ valence_charge`.
- `lsda_pw92(&rho_in_up, &rho_in_dn)` → `terms.xc(&rho_in_up, &rho_in_dn)`, the same for the output.
- `solve_channel(grid, &v_up, terms.nl.as_ref(), …)`.
- The energy: `external = grid.dot(&rho, &terms.v_loc)`, plus `nl.energy(&orb_up) + nl.energy(&orb_dn)` only when `terms.nl` is `Some`. `nuclear: terms.e_nn`.
- Warnings: for each overlap, `format!("{CORE_OVERLAP_WARNING}: {i}–{j} at R = {d:.3} Bohr")`.
- `KsState { …, ionic: system.ionic.clone() }` and `ScfStats { …, core_overlaps: terms.overlaps }`.

`KsState.v_nuc` keeps its name and now holds V_loc.

In `crates/qm-gpu/src/scf.rs`, add `core_overlaps: Vec::new()` to its `ScfStats` literal. No other `qm-gpu` change: until Task 8, `GpuSolver` asserts `system.ionic.is_none()`. Add that assert with the message "the GPU solver gains pseudopotentials in M7 Task 8".

In `crates/qm/src/lib.rs`: `pub mod ionic; pub mod nonlocal;`, and re-export `ionic::{IonicPolicy, IonicSetup}` and `scf::valence_electrons`.

- [ ] **Step 7: Run the tests**

Run: `cargo test -p qm --release`
Expected: PASS, including `none_policy_is_bitwise_today`.

Run: `cargo test -p validate --release --test pseudo`
Expected: PASS. Runs are reported; record Ne's Δ, the O splitting and the egg-box rms.

Run the unchanged suites: `cargo test -p validate --release --test oracles` and `cargo test -p validate --release --test gpu_vs_cpu`
Expected: PASS with the same numbers. Policy `None` changes nothing (rule 6, spec §6.1).

If `pseudo_ne_3d_matches_radial` misses 2 mHa, report the h-series E(0.20), E(0.15), E(0.10) for Ne. The difference must be the 3-D discretisation, not a normalisation; S1's test already ruled the normalisation out. Do not loosen.

- [ ] **Step 8: Record and commit.** Add a section "M7 grid integration (Task 6)" to `references.md` with Ne's 3-D vs radial Δ, the O singlet/triplet energies and SCF step counts, and the O egg-box rms.
```bash
git add crates/qm crates/qm-gpu/src/scf.rs crates/validate/tests/pseudo.rs crates/validate/references.md
git commit -o -m "feat(qm): ionic setup, KB projectors and NLCC in the real-space SCF (M7 Task 6)" -- crates/qm crates/qm-gpu/src/scf.rs crates/validate/tests/pseudo.rs crates/validate/references.md
```
Before merging, hand the diff to `physics-reviewer` with the S1 test, the energy expression and the None-bitwise result.

**As built (Task 6 review, ruling B).** `v_nl_on_grid_is_the_operator_identity` runs at h = 0.1 (61³) and h = 0.05 (121³), in the same 6 Bohr box. It asserts:
- |ε| < 1e-4 at both;
- the pointwise and bilinear 1e-6 bounds at h = 0.05 only;
- ε(0.05) < ε(0.1)/4.

Derivation: ε is a lattice-sum error of a radial integrand whose third derivative jumps at r_c (u ∈ C⁴ ⇒ V_scr ∈ C²), so ε ∝ h⁵. A missing √((2l+1)/4π) gives ε = 3.19 at every h.

**Fix round 1 (Task 6 review, `.superpowers/sdd/2026-10-15-m7-pseudopotentials/task-6-review.md`, rulings A and C).**

**Step R1: `Occupations` (spec §5.7).** In `crates/qm/src/scf.rs`:
- Add `#[derive(Clone, Copy, Debug, PartialEq)] pub enum Occupations { Aufbau, DegenerateShells { tau: f64 } }` and `pub const DEGENERACY_TAU: f64 = 1e-4;`.
- `ScfSettings` gains `pub occupations: Occupations`; `for_grid` sets `Aufbau`.
- `KsState` gains `pub occ_up: Vec<f64>, pub occ_dn: Vec<f64>` (one weight per stored orbital; all 1.0 under `Aufbau`).
- `ScfStats` gains `pub fractional: Vec<FractionalShell>`, with `pub struct FractionalShell { pub spin: u8, pub first: usize, pub count: usize, pub f: f64, pub spread: f64 }`.
- Under `Aufbau` not one operation changes: k_σ = N_σ, `density_from`, the unweighted kinetic and `NonlocalOp::energy`. `none_policy_is_bitwise_today` must still pass.
- Under `DegenerateShells` follow §5.7's rule: grow the witness, weight ρ, T_s and E_NL. `NonlocalOp::energy` gains a weights argument, or a weighted variant.
- The warm-start guard also requires the same `occupations`.
- Document the rule, its derivation and τ's bounds in the `scf` module doc.

**Step R2: tests.**
- qm:
  - `degenerate_shell_occupations_on_levels`: a pure function on synthetic levels. A three-fold cluster with one electron gives 1/3 each. A full cluster gives integers. A cluster reaching the last vector asks for another vector.
  - `closed_shells_are_aufbau_under_the_rule`: H₂ and LiH as in `none_policy_is_bitwise_today`, under `DegenerateShells`. Integer f, and E equal to `Aufbau` within 1e-9 Ha.
  - `ion_on_a_grid_point_is_finite`: un-ignored, with `DegenerateShells`.
- validate (`pseudo.rs`):
  - `solve_with` sets `DegenerateShells` (the providers' choice under the pseudo policies).
  - `pseudo_o_triplet_below_singlet`: per spec §6.3. Radial references via `solve_kb_atom(&ft.grid, &ft.input(), &reference_config(&t.partition), Restricted)`, and the "Hund" entry of `transfer_configs(&t, &ground_state(8, Polarised)?)`. Assert `ground_state(8, Polarised)?.multiplicity == 3`.
  - `single_atom_eggbox_floor_oxygen`: un-ignored, with the wall subtraction of ruling C.
  - `pseudo_o_integer_triplet_report`: `#[ignore = "report"]`, `Aufbau` with linear mixing, reporting E_int − E_sph.

**Step R3: references.md.**
- Rewrite "Open-shell O" as the record of why §5.7 exists, keeping the integer-aufbau numbers, and add the gate numbers.
- Replace the "Single-atom 3-D floor" paragraph per ruling C.
- Fix the cost paragraph per ruling D.

**The wall helper (ruling C).** `Wall { kappa, a }` (`fit`, `energy`, `force`) in `crates/validate/tests/pseudo.rs` (duplicated in `pseudo_oracles.rs` by Task 10): the egg-box is periodic in each shift component with period h, so the lattice translates ±h·ê_b see the unshifted geometry's egg-box and differ from it only through the Dirichlet faces; each face's term decays as e^{−2κd}, κ = √(−2ε_max), so per axis E_w,b(s) = A₊(e^{2κs} − 1) + A₋(e^{−2κs} − 1), F_w,b(s) = −2κ(A₊e^{2κs} − A₋e^{−2κs}). Task 6's `eggbox_floor` fits it from E(0) and E(h·ê_b) (E(−h·ê_b) = E(h·ê_b) by the centred box's inversion symmetry) and prints raw, wall and corrected |F| per offset and the corrected energy spread; expected for H at 12 Bohr: 2.90e-5 / 2.86e-5 / ≈ 4e-7.

**Fix round 2 (Task 6 re-review round 1, ruling 1) [Fable].**

**Step R4: the state-defined rule (spec §5.7).**
- `DEGENERACY_TAU` = 1e-3, `CANDIDATE_TAU` = 0.1.
- `solve_shared` gains the phase: candidates use τ_c; at Δρ < tol the shared clusters are checked against τ, released by gap-splitting, and frozen; then the run continues to tol.
- `ScfStats.released: Vec<FractionalShell>`.
- The module doc and spec §5.7 text as above. Retract "measured ≤ 1e-6 Ha" (the review's near-cubic measurement).

**Step R5: tests.**
- `degenerate_spread_at_low_symmetry_sites` (validate, CPU).
  - Cases: O (M = 1, 3) at 0.20; C (M = 3) and N (M = 2) at 0.30; F (M = 2) at 0.15.
  - 8 Bohr box (the spread is local).
  - Sites (0, h/2, h/2), (h/2, 0, 0), (h/2, h/2, h/2) and (h/4, h/4, h/4).
  - Assert: every partly filled set is shared, with spread ≤ τ/10.
  - Reported: the step-1 spread.
- `single_atom_eggbox_floor_oxygen` additionally asserts:
  - all 19 solves shared;
  - the wall-subtracted {0, h/2}³ spread < 1e-5 Ha (it was 4.11e-3; the expected egg-box is ~3e-6).
- `degenerate_shell_occupations_on_levels`: add a release case (a cluster of spread ≥ τ is split by gaps).
- Re-run: the Hund gate, `lone_o_and_n_scan_picks_hund`, the H and O floors, and `closed_shells_are_aufbau_under_the_rule`.

**Ruling 2 (O's floor; Task 7's precondition has fired).** Before Task 7's gate: `o_floor_attribution_report` (per term F_LR, F_SR, F_NL, F_pc; 16 Bohr; h = 0.20 and 0.15), and the O₂ gate run early and recorded (`o2_eggbox_gate_early_run`, frozen ΣF, wall from 6 translates). The 3e-5 gate is not moved; a miss goes to physics review (refined-grid XC, a smoother model core, or Ono–Hirose only if F_NL dominates; else spec §7 risk 1's fallback).

---

### Task 7: forces — local (LR + SR), nonlocal and NLCC **[Fable — physics-review gate]**

Spec §5.4, §6.6, §5.6, §8 row 7; review M4 (the split in the force; superseded by the plan review's single V_SR table, M2), M5 (where the 3e-5 gate is stated) and the NOTE re-deriving the three force terms. Plan review amendments here:
- S2: the HF-vs-FD test cannot see a missing term below its 2e-3 bound, so each term also gets a frozen-state finite-difference check, exact to ~1e-8 and independent of the SCF, and |HF − Richardson| < 1e-4 is asserted (5× the derived FD floor; a spec revision, §6.6);
- S6: the on-grid-point atom's force bound is 1e-6 at tol = eig_tol = 1e-9 (inversion symmetry cancels the force only for a parity-pure state; the eigensolver residual mixes parity at first order);
- M4: the O₂ energy ripple is reported from the shifts {0, h/2}³; the force probe stays at ±h/4, where sin(2πR/h) = ±1 is maximal, so the §6.6 gate stands.

**Files:**
- Modify: `crates/qm/src/forces.rs` (dispatch on `state.ionic`; the pseudised terms as separate `pub(crate)` functions; the frozen-state FD test), `crates/qm/src/nonlocal.rs` (gradients: `solid_harmonic_gradients`, `NonlocalOp::build_with_gradients`, `Sphere.grad`), `crates/qm/src/scf.rs` (the on-grid-point force), `crates/validate/tests/pseudo.rs` (three tests), `crates/validate/references.md` (section "M7 forces (Task 7)")

**Interfaces:**
- Consumes: Task 6's `IonicSetup`, `Ionic`, `NonlocalOp`, `Sphere`, `points_within`, `partial_core_field`, `smeared_potential`, `centres`, `ionic_repulsion`, `KsState.ionic`, `ScfStats.core_overlaps`, `CORE_OVERLAP_WARNING`, and Task 5's `FilteredTable::{dv_lr, v_sr, dv_sr, sr_radius, dbeta_radial, beta_radial, drho_pc_at, pc_radius}`.
- Produces:
```rust
pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError>; // signature unchanged
// forces.rs, the pseudised terms (each the negative R-derivative of one frozen energy, review S2):
pub(crate) fn local_lr_forces(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]>;   // −∂/∂R ∫ρ V_LR (and AE species' smeared nuclei)
pub(crate) fn local_sr_forces(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]>;   // −∂/∂R h³Σρ V_SR
pub(crate) fn nonlocal_forces(grid: &Grid, orbitals: &[&[f64]], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]>; // −∂/∂R Σ E_KB c²
pub(crate) fn partial_core_forces(grid: &Grid, rho_up: &[f64], rho_dn: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]>; // −∂/∂R E_xc[ρ + ρ_pc(R)]
pub(crate) fn ion_ion_forces(nuclei: &[Nucleus], setup: &IonicSetup) -> Result<Vec<[f64; 3]>, QmError>;            // −∂/∂R Σ q_I q_J/R_IJ
pub fn solid_harmonic_gradients(r: [f64; 3]) -> [[f64; 3]; 9];      // ∇R_lm, l ≤ 2, index lm_index(l, m)
impl NonlocalOp { pub fn build_with_gradients(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> NonlocalOp; }
pub struct Sphere { /* … */ pub grad: Vec<Vec<[f64; 3]>> /* [m][point]; empty unless built with gradients */ }
```
**The force terms** (spec §5.4, with forces.rs's sign chain ∂/∂R_I = −∇_r). For a pseudised ion I:
- F^LR = ∫ρ ∇V_LR, today's integrand with Z → Z_v and σ → σ_LR.
- F^SR = ∫ρ (dV_SR/dd)(r − R_I)/d over the SR sphere, where dV_SR/dd is `interp_deriv` of the one filtered V_SR table (plan review Q2): the exact derivative of the sampled energy. There is no tail branch.
- F^NL = +2 Σ_iσ Σ_m E_KB c_m ⟨∇β_m|ψ_iσ⟩, with ∇β_m = t′(d)·N·R_lm·(r − R)/d + t(d)·N·∇R_lm.
- F^pc = ∫ ½(v_xc↑ + v_xc↓) (dρ_pc/dd)(r − R_I)/d, with v_xc evaluated on ρ_σ + ½ρ_pc.
- Ion–ion: q_I q_J (R_I − R_J)/R³.

For an all-electron species in the same setup, the local term is today's (Z, σ_AE). There is no Pulay term, for the same reason as today.

- [ ] **Step 1: Write the failing tests**

In `crates/qm/src/nonlocal.rs` tests:
```rust
#[test]
fn solid_harmonic_gradients_match_finite_differences() {
    let r = [0.37, -0.52, 0.81];
    let g = solid_harmonic_gradients(r);
    let e = 1e-6;
    for lm in 0..9 {
        for a in 0..3 {
            let (mut p, mut m) = (r, r);
            p[a] += e; m[a] -= e;
            let (mut sp, mut sm) = ([0.0; N_MOMENTS], [0.0; N_MOMENTS]);
            solid_harmonics(p, &mut sp); solid_harmonics(m, &mut sm);
            let fd = (sp[lm] - sm[lm]) / (2.0 * e);
            assert!((g[lm][a] - fd).abs() < 1e-8, "lm {lm}, axis {a}: {} vs {fd}", g[lm][a]);
        }
    }
}
```
In `crates/qm/src/forces.rs` tests (plan review S2):
```rust
/// Plan review S2: every pseudised force term is the exact negative R-derivative of its own frozen
/// energy — ρ, the spin densities and the orbitals held fixed — checked against Richardson-
/// extrapolated central differences (δ = 1e-4, 2e-4) to 1e-8 + 1e-6·|F| (the relative part covers
/// the log-grid interpolant's C⁰ kinks, ~1e-7 relative, review NOTE). Independent of the SCF: a
/// missing or sign-flipped term fails here at O(1e-3–1), while HF-vs-FD on a converged state is
/// blind below its 2e-3 bound. Two pseudo-O ions off the grid points (ion–ion, overlapping spheres);
/// the frozen state is synthetic: a density blob per ion, an s-like and a p-like orbital, spin 0.6/0.4.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
fn force_terms_are_exact_derivatives_of_frozen_energies() {
    use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
    use crate::nonlocal::NonlocalOp;
    use crate::nuclear::{centres, ionic_repulsion, partial_core_field, points_within, smeared_potential};
    use crate::radial::{filter::filtered, generate};
    use crate::xc::lsda_pw92;
    let h = 0.25;
    let g = Grid::cube(33, h);
    let t = std::sync::Arc::new(filtered(&generate(8).unwrap(), h));
    let setup = IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, std::collections::BTreeMap::from([(8, Ionic::Pseudo(t.clone()))]));
    let base = vec![Nucleus { pos: [-1.13, 0.07, 0.03], z: 8 }, Nucleus { pos: [1.16, -0.04, 0.05], z: 8 }];
    let dist = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let blob = |c: [f64; 3], w: f64| g.sample(|r| (-dist(r, c).powi(2) / w).exp());
    let rho: Vec<f64> = blob(base[0].pos, 1.0).iter().zip(blob(base[1].pos, 1.0)).map(|(a, b)| a + b).collect();
    let up: Vec<f64> = rho.iter().map(|x| 0.6 * x).collect();
    let dn: Vec<f64> = rho.iter().map(|x| 0.4 * x).collect();
    let s_orb = blob(base[0].pos, 1.5);
    let p_orb: Vec<f64> = g.sample(|r| r[0] - base[0].pos[0]).iter().zip(&s_orb).map(|(x, s)| x * s).collect();
    let orbitals: [&[f64]; 2] = [&s_orb, &p_orb];
    // the frozen energies as functions of the geometry
    let e_lr = |n: &[Nucleus]| g.dot(&rho, &smeared_potential(&g, &centres(n, &setup)));
    let e_sr = |n: &[Nucleus]| n.iter().map(|ion| crate::grid::neumaier(points_within(&g, ion.pos, t.sr_radius()).into_iter()
        .map(|p| rho[p] * t.v_sr(dist(g.point(p), ion.pos)))) * g.dv()).sum::<f64>();
    let e_nl = |n: &[Nucleus]| NonlocalOp::build(&g, n, &setup).energy(&[s_orb.clone(), p_orb.clone()]);
    let e_pc = |n: &[Nucleus]| {
        let pc = partial_core_field(&g, n, &setup);
        let u: Vec<f64> = up.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
        let d: Vec<f64> = dn.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
        g.integrate(&lsda_pw92(&u, &d).energy_density)
    };
    let e_ii = |n: &[Nucleus]| ionic_repulsion(n, &setup).unwrap();
    let terms: [(&str, Vec<[f64; 3]>, &dyn Fn(&[Nucleus]) -> f64); 5] = [
        ("LR", local_lr_forces(&g, &rho, &base, &setup), &e_lr),
        ("SR", local_sr_forces(&g, &rho, &base, &setup), &e_sr),
        ("NL", nonlocal_forces(&g, &orbitals, &base, &setup), &e_nl),
        ("ρ_pc", partial_core_forces(&g, &up, &dn, &base, &setup), &e_pc),
        ("ion–ion", ion_ion_forces(&base, &setup).unwrap(), &e_ii),
    ];
    for (name, f, e) in terms {
        for i in 0..base.len() {
            for a in 0..3 {
                let fd = |d: f64| { let (mut p, mut m) = (base.clone(), base.clone()); p[i].pos[a] += d; m[i].pos[a] -= d; -(e(&p) - e(&m)) / (2.0 * d) };
                let rich = (4.0 * fd(1e-4) - fd(2e-4)) / 3.0;
                assert!((f[i][a] - rich).abs() <= 1e-8 + 1e-6 * f[i][a].abs(), "{name}: ion {i}, axis {a}: F {:.10e} vs −dE/dR {rich:.10e}", f[i][a]);
            }
        }
    }
}
```
In `crates/validate/tests/pseudo.rs`:
```rust
/// Spec §8 row 7 first failing test (and Review Focus 4: run under CoresOnly, the mixed scene — an
/// all-electron H at σ = h beside a pseudo-O at σ_LR = 2h): Hellmann–Feynman vs central finite
/// differences of E at δ = 0.01 within forces.rs's 2e-3 bound, on both nuclei; δ = 0.02 too, and the
/// Richardson-separated estimate (4D(δ) − D(2δ))/3 within 1e-4 of HF (plan review S2: on a fixed grid
/// HF is the exact derivative of the discrete energy, `forces.rs` derives the HF−FD floor at ≈ 2e-5
/// (FD truncation), and 1e-4 is 5× that floor — a spec revision, §6.6).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn oh_hf_force_matches_fd() {
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let oh = |dx: f64, which: usize| {
        let mut v = vec![Nucleus { pos: [0.03, 0.02, 0.0], z: 8 }, Nucleus { pos: [1.86, 0.11, 0.0], z: 1 }];
        v[which].pos[0] += dx;
        v
    };
    let tight = |nuc: Vec<Nucleus>| {
        let g = Grid::cube(n_for(12.0, h), h);
        let zs: Vec<u8> = nuc.iter().map(|n| n.z).collect();
        let sys = System::with_ionic(nuc, 0, Some(2), setup(IonicPolicy::CoresOnly, h, &zs));
        let mut s = ScfSettings::for_grid(&g);
        (s.tol, s.eig_tol) = (1e-9, 1e-9);
        let (st, stats) = scf(&g, &sys, &s, None).expect("SCF converges");
        (g, sys, st, stats.energy.total, s.sigma)
    };
    let (g, sys, st, _, sigma) = tight(oh(0.0, 0));
    let hf = qm::forces(&g, &st, &sys.nuclei, sigma).unwrap();
    for which in [0usize, 1] {
        let d = |delta: f64| -(tight(oh(delta, which)).3 - tight(oh(-delta, which)).3) / (2.0 * delta);
        let (d1, d2) = (d(0.01), d(0.02));
        let rich = (4.0 * d1 - d2) / 3.0;
        eprintln!("OH nucleus {which}: HF {:.7}  FD(0.01) {d1:.7}  FD(0.02) {d2:.7}  Richardson {rich:.7}", hf[which][0]);
        assert!((hf[which][0] - d1).abs() < 2e-3, "nucleus {which}: HF {} vs FD {d1}", hf[which][0]);
        assert!((hf[which][0] - rich).abs() < 1e-4, "nucleus {which}: HF {} vs Richardson {rich} (plan review S2)", hf[which][0]);
    }
}

/// Spec §6.6 / §8 row 7 (review M5): an all-pseudo system under CoresOnly — O₂ (triplet, R = 2.28
/// Bohr) — max over the 8 rigid shifts of ±h/4 of |Σ_I F_I| < 3e-5 Ha/Bohr (the force probe at ±h/4
/// sees the first reciprocal shell at full weight: sin(2π·(±h/4)/h) = ±1); the y-component (today's
/// figure of record is −3.4e-4 for H₂O) reported, and the rms energy ripple reported from 8 further
/// shifts {0, h/2}³ (plan review M4: an energy over ±h/4 shifts of an inversion-symmetric system is
/// blind to the first shell). At 57³ × 0.25 Bohr (14 Bohr) when O's h_req allows 0.25, else at h_req
/// with the same 14 Bohr box (recorded which).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn o2_max_shift_sum_of_forces_below_3e_5() {
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let o2 = |s: [f64; 3]| vec![Nucleus { pos: [-1.14 + s[0], 0.07 + s[1], 0.03 + s[2]], z: 8 }, Nucleus { pos: [1.14 + s[0], 0.07 + s[1], 0.03 + s[2]], z: 8 }];
    let q = h / 4.0;
    let (mut worst, mut fy) = (0.0f64, Vec::new());
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let nuc = o2([sx, sy, sz]);
        let (_, st, _) = solve(h, 14.0, IonicPolicy::CoresOnly, nuc.clone(), Some(3));
        let g = Grid::cube(n_for(14.0, h), h);
        let f = qm::forces(&g, &st, &nuc, h).unwrap();
        let sum = [f[0][0] + f[1][0], f[0][1] + f[1][1], f[0][2] + f[1][2]];
        worst = worst.max((sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt());
        fy.push(sum[1]);
    }}}
    let half = h / 2.0;
    let mut es = Vec::new();
    for sx in [0.0, half] { for sy in [0.0, half] { for sz in [0.0, half] {
        es.push(solve(h, 14.0, IonicPolicy::CoresOnly, o2([sx, sy, sz]), Some(3)).0);
    }}}
    let mean = es.iter().sum::<f64>() / 8.0;
    let rms = (es.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    eprintln!("O₂ at h = {h}: max |ΣF| {worst:.2e} (±h/4), ΣF_y {fy:?}, E rms {rms:.2e} ({{0, h/2}}³)");
    assert!(worst < 3e-5, "max |Σ_I F_I| = {worst:.3e}");
}

/// Spec §5.6: pushed-together cores warn (EvalStats-bound text), the step is still taken, and the
/// forces are finite. R is set below r_core,O + r_core,O from the generated table (derived, not typed).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn core_overlap_warns_and_still_steps() {
    let h = table(8).record.h_req.expect("O has h_req").min(0.25);
    let r = 0.8 * 2.0 * table(8).record.r_core;
    let nuc = vec![Nucleus { pos: [-0.5 * r, 0.0, 0.0], z: 8 }, Nucleus { pos: [0.5 * r, 0.0, 0.0], z: 8 }];
    let (_, st, stats) = solve(h, 10.0, IonicPolicy::CoresOnly, nuc.clone(), None);
    assert_eq!(stats.core_overlaps.len(), 1);
    assert!(stats.warnings.iter().any(|w| w.starts_with(qm::ionic::CORE_OVERLAP_WARNING)), "{:?}", stats.warnings);
    let g = Grid::cube(n_for(10.0, h), h);
    assert!(qm::forces(&g, &st, &nuc, h).unwrap().iter().flatten().all(|x| x.is_finite()));
}
```
In `crates/qm/src/scf.rs` tests, extend `ion_on_a_grid_point_is_finite` with forces. An isolated atom on a grid point has every force integrand odd about R, so its force vanishes for an exactly parity-pure state. The iterative eigenvectors carry a parity admixture of first order in the eigensolver residual, so |F| ~ residual·|∇V| (plan review S6). The test therefore solves at tol = eig_tol = 1e-9 (Task 6 already does) and bounds the force by 1e-6. Concretely, after its SCF (rename `_st` to `st`):
```rust
    let f = crate::forces(&g, &st, &sys.nuclei, g.h).unwrap();
    assert!(f[0].iter().all(|x| x.is_finite() && x.abs() < 1e-6), "{:?}", f[0]);
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release nonlocal forces` and `cargo test -p validate --release --test pseudo oh_hf_force_matches_fd`
Expected: the gradient test and `force_terms_are_exact_derivatives_of_frozen_energies` fail to compile. The force tests fail: `qm::forces` ignores `state.ionic`, so the pseudised terms are missing and HF ≠ FD.

- [ ] **Step 3: Implement the gradients**

In `crates/qm/src/nonlocal.rs`:
```rust
/// ∇R_lm for l ≤ 2 in `qm::poisson`'s Racah normalisation and order (lm_index): R00 = 1;
/// (R1−1, R10, R11) = (y, z, x); R2−2 = √3xy, R2−1 = √3yz, R20 = (2z² − x² − y²)/2, R21 = √3xz,
/// R22 = (√3/2)(x² − y²). Pinned against `solid_harmonics` by finite differences.
pub fn solid_harmonic_gradients(r: [f64; 3]) -> [[f64; 3]; 9] {
    let [x, y, z] = r;
    let s3 = 3f64.sqrt();
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
        [s3 * y, s3 * x, 0.0], [0.0, s3 * z, s3 * y], [-x, -y, 2.0 * z], [s3 * z, 0.0, s3 * x], [s3 * x, -s3 * y, 0.0],
    ]
}
```
`Sphere` gains `pub grad: Vec<Vec<[f64; 3]>>` (empty from `build`). `build_with_gradients` builds the same spheres plus, per m and point, the gradient:
```rust
let d = [r[0] - n.pos[0], r[1] - n.pos[1], r[2] - n.pos[2]];
let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
let (tv, dt) = (t.beta_radial(k, dd), t.dbeta_radial(k, dd));
let lm = lm_index(p.l as usize, m);
let gr = solid_harmonic_gradients(d)[lm];
let radial = if dd > 1e-12 { dt * norm * sh[lm] / dd } else { 0.0 };   // t′(d) R_lm d̂ → 0 at d = 0 (R_lm ∝ d^l, t′(0) = 0)
[radial * d[0] + tv * norm * gr[0], radial * d[1] + tv * norm * gr[1], radial * d[2] + tv * norm * gr[2]]
```

- [ ] **Step 4: Implement the pseudised forces**

In `crates/qm/src/forces.rs` (the module doc gains a section "Pseudised ions (M7)" with the four terms, their sign derivations, the no-Pulay argument, and why each term is a separate function — each is −∂/∂R of one frozen energy, which `force_terms_are_exact_derivatives_of_frozen_energies` checks):
```rust
use crate::ionic::{Ionic, IonicSetup};
use crate::nonlocal::NonlocalOp;
use crate::nuclear::{partial_core_field, points_within};
use crate::xc::lsda_pw92;
use crate::grid::neumaier;

pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError> {
    assert_eq!(&state.grid, grid, "KsState was converged on a different grid");
    assert!((state.sigma - sigma).abs() < 1e-15, "KsState was converged with sigma = {}, forces called with {}", state.sigma, sigma);
    match state.ionic.as_deref() {
        None => all_electron_forces(grid, state, nuclei, sigma),     // today's body, moved verbatim
        Some(setup) => pseudo_forces(grid, state, nuclei, setup),
    }
}

/// Σ_p w_p · V′(d_p) (r_p − R)/d_p over `points` (all grid points when None), Neumaier per
/// component, × dv; points closer than ON_SITE contribute 0 (every radial term is smooth there).
fn radial_force(grid: &Grid, points: Option<&[usize]>, centre: [f64; 3], w: &[f64], dvdd: &dyn Fn(f64) -> f64) -> [f64; 3] {
    let all: Vec<usize>;
    let pts = match points { Some(p) => p, None => { all = (0..grid.len()).collect(); &all } };
    let mut comp: [Vec<f64>; 3] = [Vec::with_capacity(pts.len()), Vec::with_capacity(pts.len()), Vec::with_capacity(pts.len())];
    for &p in pts {
        let r = grid.point(p);
        let d = [r[0] - centre[0], r[1] - centre[1], r[2] - centre[2]];
        let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if dd < ON_SITE { continue; }
        let s = w[p] * dvdd(dd) / dd;
        for a in 0..3 { comp[a].push(s * d[a]); }
    }
    std::array::from_fn(|a| neumaier(comp[a].iter().copied()) * grid.dv())
}

fn pseudo_forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], setup: &IonicSetup) -> Result<Vec<[f64; 3]>, QmError> {
    let ii = ion_ion_forces(nuclei, setup)?;
    let rho = state.rho();
    let orbitals: Vec<&[f64]> = state.orbitals_up.iter().chain(&state.orbitals_dn).map(|v| v.as_slice()).collect();
    let parts = [
        ii,
        local_lr_forces(grid, &rho, nuclei, setup),
        local_sr_forces(grid, &rho, nuclei, setup),
        nonlocal_forces(grid, &orbitals, nuclei, setup),
        partial_core_forces(grid, &state.rho_up, &state.rho_dn, nuclei, setup),
    ];
    Ok((0..nuclei.len()).map(|i| std::array::from_fn(|a| parts.iter().map(|p| p[i][a]).sum::<f64>())).collect())
}

/// −∂/∂R_I ∫ρ V_LR: the smeared centre of every species — (Z_v, σ_LR) for pseudised ions, today's
/// (Z, σ_AE) for all-electron ones in the same setup.
pub(crate) fn local_lr_forces(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { sigma } => { let z = n.z as f64; radial_force(grid, None, n.pos, rho, &|d| -z * g_prime(d, *sigma)) }
        Ionic::Pseudo(t) => radial_force(grid, None, n.pos, rho, &|d| t.dv_lr(d)),
    }).collect()
}

/// −∂/∂R_I h³Σ_p ρ_p V_SR(|r_p − R_I|): the one V_SR table's `interp_deriv` over its sphere.
pub(crate) fn local_sr_forces(grid: &Grid, rho: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { .. } => [0.0; 3],
        Ionic::Pseudo(t) => radial_force(grid, Some(&points_within(grid, n.pos, t.sr_radius())), n.pos, rho, &|d| t.dv_sr(d)),
    }).collect()
}

/// −∂/∂R_I Σ_ψ Σ E_KB c²: +2 Σ_ψ Σ_m E_KB c_m ⟨∇β_m|ψ⟩ (real orbitals, occupation 1 each).
pub(crate) fn nonlocal_forces(grid: &Grid, orbitals: &[&[f64]], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0f64; 3]; nuclei.len()];
    let nl = NonlocalOp::build_with_gradients(grid, nuclei, setup);
    for s in &nl.spheres {
        for psi in orbitals {
            for (b, gb) in s.beta.iter().zip(&s.grad) {
                let c = neumaier(s.points.iter().zip(b).map(|(&p, &bv)| bv * psi[p])) * nl.dv;
                for a in 0..3 {
                    let gpsi = neumaier(s.points.iter().zip(gb).map(|(&p, gv)| gv[a] * psi[p])) * nl.dv;
                    out[s.ion][a] += 2.0 * s.e_kb * c * gpsi;
                }
            }
        }
    }
    out
}

/// −∂/∂R_I E_xc[ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc] at fixed ρ: +∫ ½(v_xc↑ + v_xc↓) ∇ρ_pc,I, v_xc on ρ_σ + ½ρ_pc.
pub(crate) fn partial_core_forces(grid: &Grid, rho_up: &[f64], rho_dn: &[f64], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]> {
    let pc = partial_core_field(grid, nuclei, setup);
    let up: Vec<f64> = rho_up.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
    let dn: Vec<f64> = rho_dn.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
    let xc = lsda_pw92(&up, &dn);
    let v_avg: Vec<f64> = xc.v_up.iter().zip(&xc.v_dn).map(|(a, b)| 0.5 * (a + b)).collect();
    nuclei.iter().map(|n| match setup.ionic(n.z) {
        Ionic::AllElectron { .. } => [0.0; 3],
        Ionic::Pseudo(t) => radial_force(grid, Some(&points_within(grid, n.pos, t.pc_radius)), n.pos, &v_avg, &|d| t.drho_pc_at(d)),
    }).collect()
}

/// −∂/∂R_I Σ_{I<J} q_I q_J / R_IJ with valence charges (coincident nuclei as today).
pub(crate) fn ion_ion_forces(nuclei: &[Nucleus], setup: &IonicSetup) -> Result<Vec<[f64; 3]>, QmError> {
    let mut out = vec![[0.0f64; 3]; nuclei.len()];
    for i in 0..nuclei.len() {
        for j in 0..nuclei.len() {
            if i == j { continue; }
            let dv = [nuclei[i].pos[0] - nuclei[j].pos[0], nuclei[i].pos[1] - nuclei[j].pos[1], nuclei[i].pos[2] - nuclei[j].pos[2]];
            let d = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt();
            if d < ON_SITE { return Err(QmError::CoincidentNuclei(i.min(j), i.max(j))); }
            let s = setup.valence_charge(nuclei[i].z) * setup.valence_charge(nuclei[j].z) / (d * d * d);
            for a in 0..3 { out[i][a] += s * dv[a]; }
        }
    }
    Ok(out)
}
```
`all_electron_forces` is today's `forces` body below its two asserts, moved without change, so `none_policy_is_bitwise_today` keeps passing.

- [ ] **Step 5: Run the tests**

Run: `cargo test -p qm --release`
Expected: PASS, including `none_policy_is_bitwise_today` with the same bits.

Run: `cargo test -p validate --release --test pseudo`
Expected: PASS. The suite takes ≈ 45 min on the CPU: the O₂ shifts are 16 × 57³-class solves (8 force shifts, 8 energy shifts), the OH finite differences 9 tight solves. Timings are reported.

Run: `cargo test -p validate --release --test dynamics`
Expected: PASS, unchanged (rule 6: forces changed).

**Gate precondition (Task 5 physics review).** Read Task 6's single-O floor first: the **wall-subtracted** egg-box force, not the raw probe (the raw single-H probe at 12 Bohr was 2.90e-5, of which 2.86e-5 is the Dirichlet wall; Task 6 review, ruling C).
- If the floor (max |F| − F_atom) exceeds 1.5e-5 Ha/Bohr, the gate cannot pass through any table change: that is half
  the gate, and O₂ carries two O atoms that add coherently in the worst case. Run the gate anyway, record it, and hand the
  attribution to physics review. The candidates are the XC evaluated on a refined grid, a smoother model core (a recipe
  revision), or spec §7 risk 1's fallback (report the achieved factor). Never a looser gate.
- Ono–Hirose is a candidate only if the nonlocal term's own 3-D share exceeds its T8(d) value. That share is measured as
  the ±h/4 force of E_NL at frozen orbitals.

**Wall subtraction (Task 6 review, ruling C; spec §6.6).** `o2_max_shift_sum_of_forces_below_3e_5`:
- Add the 6 translate solves ±h·ê_b, with κ from the unshifted state's highest occupied level, and fit `Wall` (Task 6's helper).
- `assert!(wall_max <= 3e-4, "the wall term {wall_max:.2e} exceeds 10× the gate: enlarge the box")` and `assert!(worst_corrected < 3e-5)`.
- Print raw, wall and corrected; subtract the wall energy from the {0, h/2}³ set.

**Occupations (Task 6 review N6).** The forces take the weights f_iσ from `KsState.occ_*` (spec §5.7): the nonlocal force is 2Σ f E_KB c⟨∇β|ψ⟩, and ρ in the local and NLCC terms is the weighted density.

If `o2_max_shift_sum_of_forces_below_3e_5` misses, record the eight vector sums and stop. The spec's plan B is the Ono–Hirose double grid for ⟨β|ψ⟩ (§5.1): a physics-review decision, then a separate task. The honest fallback (§7 risk 1) is to report the achieved factor. The bound is not moved.

- [ ] **Step 6: Record and commit.** Add a section "M7 forces (Task 7)" to `references.md` with:
  - the frozen-state per-term worst |F − (−dE/dR)| (LR, SR, NL, ρ_pc, ion–ion);
  - OH: HF, FD(0.01), FD(0.02) and Richardson for both nuclei, and |HF − Richardson|;
  - O₂: max |ΣF| and the y-components over the 8 ±h/4 shifts, the E rms over {0, h/2}³, and the h used;
  - the on-grid-point O force;
  - the core-overlap R and r_core.

```bash
git add crates/qm/src/forces.rs crates/qm/src/nonlocal.rs crates/qm/src/scf.rs crates/validate/tests/pseudo.rs crates/validate/references.md
git commit -o -m "feat(qm): pseudopotential forces — LR/SR local, nonlocal, NLCC and ion–ion terms (M7 Task 7)" -- crates/qm/src/forces.rs crates/qm/src/nonlocal.rs crates/qm/src/scf.rs crates/validate/tests/pseudo.rs crates/validate/references.md
```
Before merging, hand the diff to `physics-reviewer` with the sign derivations, the per-term frozen-state numbers and the OH/O₂ numbers.


**NLCC ruling (2026-10-16), Ruling 6.**
- The physics is unchanged. The NLCC force uses v̄ on ρ_v,σ + ½ρ_pc, the same as the SCF.
- Add an oracle: the 3-D NLCC force of a frozen spherical filtered atom, fed the sampled radial ρ_v, equals `filter::pc_lattice_force` to 1e-10 relative. It is the same lattice sum.
- `o2_max_shift_sum_of_forces_below_3e_5` follows the §6.6 outcome: a hard assertion if it passes; otherwise it is renamed `o2_eggbox_recorded` and asserts the ratchet.
- The HF/FD and Richardson checks are unchanged.

**Status (2026-10-17): implemented, awaiting physics review.** Records: `references.md` "M7 forces (Task 7)".
- Frozen-state per-term check: worst |F − (−dE/dR)| LR 8.7e-11, SR 3.2e-9, NL 5.7e-9, ρ_pc 2.8e-10, ion–ion 4.2e-11 (a mixed setup: two pseudo-O plus an all-electron H; f = 1, ⅔).
- NLCC identity oracle: bitwise at dyadic h (3/16, 5/32); 4.9e-9 at 0.20, from the grid coordinates' ulp (reported).
- On-grid-point O: |F| ≤ 3.0e-10. OH (10 Bohr, not 12: CPU small; `DegenerateShells`, the π pair ½–½): |HF − Richardson| 5.4e-7 (O), 8.3e-7 (H); 760 s.
- Deviations: `nonlocal_forces` takes the occupations (`occ`, review N6); the core-overlap solve runs on the GPU when there is one (CPU 1560 s); the §6.6 tests are `o2_eggbox_recorded` (ratchet at 0.20) and `eggbox_passes_at_0_15` (O₂ and H₂O, hard PASS) on the f64-energy instrument (Task 8a review), with no wall-fit or E-rms code of their own (`eggbox_gate`).

---

### Task 7c: the partial core's alias — nonlinear T8(d) and the NLCC crossover ladder **[Fable — physics-review gate]**

**Ruling:** `o2-eggbox-ruling.md`.

**Round 2 (NLCC ruling, `nlcc-ruling.md`).** Steps = Ruling 5, steps 1–5.
- Return to review only on step 1's row mismatch.
- The steps 2–4 failures fall back to (B) automatically.
- Status: **PASS at c = ¼, outcome (C)** (2026-10-16).
  - Step 1: `m7-tm-kb-nlcc-4` regenerated; its ¼ rows equal the 7c log's (83/83 entries); qm green (one fixture pinned to
    its c = 1 case); pseudo_gen 3/3, minimax M = 1.43e-3 / 2.81e-4 / 1.90e-5 → ¼.
  - Step 2: lone O at 16 Bohr, F_pc 1.98e-6 (0.20) and −3.6e-7 (0.15) per component against T8(e) 8.6e-7 / −5.2e-7;
    |frozen_force| at 0.15 4.2e-6 ≤ 3.5e-5.
  - Step 3: Hund oracle (−0.73 / −0.74 mHa, split 56.07 against radial 56.06), Ne −0.88 mHa, `--test pseudo` 4/4,
    gpu_vs_cpu 8/8.
  - Step 4: O₂ 4.26e-5 at 0.20 (MISS → ratchet), 9.2e-6 at 0.175, 6.7e-6 at 0.15 (PASS); H₂O 3.03e-5 at 0.20 (MISS →
    ratchet), 4.1e-6 at 0.15 (PASS).

Round 1 (2026-10-16) stopped at "no global c exists" under the first rule (the nonlinear F_pc inside T8(d)); its record
is `task-7c-report.md` and references.md "Task 7c".

---

### Task 8: GPU kernels — `local_sr`, `nlcc_deposit`, `proj_dot`/`proj_finish`/`proj_axpy`, `gather_dot3`; `gpu_vs_cpu` extended **[Opus]**

Spec §5.3, §5.5 (`qm-gpu` bullet), §6.3 (GPU vs CPU), §7 risk 6, §8 row 8. **Spec questions 6 and 7 apply:** V_LR is the analytic erf per centre, and c_lm is finished on the GPU inside H applications while energies and forces are finished in f64 on the CPU. Runs in parallel with Task 9 (different crates; Task 9 only needs `GpuKs` to accept a `System` with a setup, which this task provides).

**Task 6 review addition.** `DegenerateShells` on the GPU: the extra witness vectors, the weighted density accumulation, the weighted kinetic and E_NL reductions (f32 partials, f64 finish); `gpu_vs_cpu` gains the pseudo-O triplet at the existing bounds. Until then `GpuSolver` asserts `Occupations::Aufbau`.

**Files:**
- Create: `crates/qm-gpu/src/ionic.rs`, `crates/qm-gpu/src/shaders/ionic.wgsl`, `crates/qm-gpu/src/shaders/erf.wgsl` (a verbatim copy of `nuclear.wgsl`'s `erfc_scaled`/`erfc_ge1`/`erf_over_x` and their constants; `nuclear.wgsl` itself is not touched, so the `None` path stays bitwise)
- Modify: `crates/qm-gpu/src/kernels.rs` (SHADERS: `("ionic", concat!(include_str!("shaders/erf.wgsl"), include_str!("shaders/ionic.wgsl")))`; SPECS entries), `crates/qm-gpu/src/eigen.rs` (`GpuHamiltonian { v, nl: Option<&GpuNonlocal> }`), `crates/qm-gpu/src/scf.rs` (the ionic path in `solve_with`; `GpuState.ionic`; the warm guard; E_NL), `crates/qm-gpu/src/forces.rs` (pseudised terms), `crates/qm-gpu/src/lib.rs`, `crates/validate/tests/gpu_vs_cpu.rs`, `crates/validate/references.md` (section "GPU vs CPU: pseudopotentials (M7 Task 8)")

**Interfaces:**
- Consumes: Tasks 6–7 (`IonicSetup`, `Ionic`, `NonlocalOp::{build, build_with_gradients}`, `nuclear::{centres, points_within, Centre}`, `FilteredTable`, `System.ionic`, `KsState`), and the existing `Batch`, `Recorder`, `Kernels`, `Field`, `GpuGrid`, `Reducer`, `map_read_f32`, `neumaier`, `NucleusRecord`.
- Produces:
```rust
// qm-gpu/src/ionic.rs
pub struct GpuIonic { /* tables buffer, ion records, sphere buffers, partials/coeff buffers, force-weight buffers */ }
impl GpuIonic {
    pub fn new(ctx: &GpuContext, g: &GpuGrid, nuclei: &[Nucleus], setup: &IonicSetup) -> GpuIonic;    // per geometry
    pub fn local_potential_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field);         // V_LR (per-centre erf) + Σ V_SR
    pub fn partial_core_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field);            // Σ ρ_pc
    pub fn nonlocal(&self) -> Option<&GpuNonlocal>;
}
pub struct GpuNonlocal { /* spheres: points (padded u32), β [m][pt] f32, E_KB, partials, coeffs */ }
impl GpuNonlocal {
    pub fn apply_in(&self, b: &mut Batch, k: &Kernels, g: &GpuGrid, x: &Field, out: &Field);       // out += V_NL x
    pub fn energy(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, orbitals: &[Field]) -> f64;    // Σ E_KB c², f64 finish
}
pub fn pseudo_electronic_forces(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, ionic: &GpuIonic, rho: &Field, v_avg: &Field,
                                orbitals: &[Field], nuclei: &[Nucleus], setup: &IonicSetup) -> Vec<[f64; 3]>;
// eigen.rs
pub struct GpuHamiltonian<'a> { pub v: &'a Field, pub nl: Option<&'a GpuNonlocal> }
// scf.rs
pub struct GpuState { /* … */ pub ionic: Option<Arc<IonicSetup>> }
```
**Design choices** (spec §5.5; this task's own, stated in `ionic.rs`'s module doc):
- `local_sr` and `nlcc_deposit` sample the filtered **uniform tables** (spec §5.1 "Sampling": Δ = h/256, filled exactly
  on the CPU in f64, uploaded as f32). The tables live in one storage buffer with per-ion offsets, every table of a
  scene padded to the longest. The lookup is the same 4-point Lagrange in s = d/Δ as `FilteredTable::v_sr`. There is no
  ln, so the f32 ln r₀ caveat of the plan review is moot. Each grid point owns its value, so there are no races.
- The projector values β_m at the sphere points and the force weights (∇β, dV_SR/dd·d̂, dρ_pc/dd·d̂) are computed on the CPU in f64 by `NonlocalOp::build_with_gradients` and the `FilteredTable` derivatives. They are uploaded as f32 per geometry: ~10⁴ floats per ion (spec §5.3: "index lists uploaded per step").
- `proj_axpy` runs once per sphere and is ordered within the batch, so overlapping spheres never race.

- [ ] **Step 1: Pin today's GPU bits.** As in Task 6 Step 1, add `gpu_none_policy_baseline` (ignored) to `crates/qm-gpu/src/scf.rs` tests. It solves H₂ and LiH at 33³ (the same geometries) and prints `const GPU_BASE_…: u64 = 0x…;` lines for E and F_x0. Run it on the unchanged code (`cargo test -p qm-gpu --release gpu_none_policy_baseline -- --ignored --nocapture`).

- [ ] **Step 2: Write the failing tests**

`crates/qm-gpu/src/ionic.rs` tests (each skips without an adapter, `#[cfg_attr(debug_assertions, ignore)]`):
```rust
use super::*;
use qm::ionic::{Ionic, IonicPolicy, IonicSetup};
use qm::radial::{filter::filtered, generate};
use std::collections::BTreeMap;
use std::sync::Arc;

fn o_setup(h: f64) -> Arc<IonicSetup> {
    Arc::new(IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([
        (8, Ionic::Pseudo(Arc::new(filtered(&generate(8).unwrap(), h)))), (1, Ionic::AllElectron { sigma: h })])))
}
fn nuclei() -> Vec<Nucleus> { vec![Nucleus { pos: [0.11, -0.07, 0.05], z: 8 }, Nucleus { pos: [1.92, 0.31, -0.2], z: 1 }] }

/// The local potential (V_LR per centre + V_SR from the tables) vs `qm::nuclear::ionic_local_potential`.
#[test]
fn local_sr_matches_cpu() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let h = 0.25;
    let grid = qm::Grid::cube(33, h);
    let (k, g) = (Kernels::new(&ctx), GpuGrid::new(&grid));
    let s = o_setup(h);
    let cpu = qm::nuclear::ionic_local_potential(&grid, &nuclei(), &s);
    let ion = GpuIonic::new(&ctx, &g, &nuclei(), &s);
    let out = Field::new(&ctx, &g, "v_loc");
    let mut b = Batch::new(&ctx, "t");
    ion.local_potential_in(&mut b, &k, &g, &out);
    b.submit();
    let got = out.download_interior(&ctx, &g);
    let worst = cpu.iter().zip(&got).map(|(a, b)| (a - *b as f64).abs() / a.abs().max(1.0)).fold(0.0, f64::max);
    assert!(worst < 2e-6, "max relative |ΔV| = {worst:.2e}");
}

/// ρ_pc deposit vs `qm::nuclear::partial_core_field`.
#[test]
fn nlcc_deposit_matches_cpu() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let h = 0.25;
    let grid = qm::Grid::cube(33, h);
    let (k, g) = (Kernels::new(&ctx), GpuGrid::new(&grid));
    let s = o_setup(h);
    let cpu = qm::nuclear::partial_core_field(&grid, &nuclei(), &s);
    let ion = GpuIonic::new(&ctx, &g, &nuclei(), &s);
    let out = Field::new(&ctx, &g, "rho_pc");
    let mut b = Batch::new(&ctx, "t");
    ion.partial_core_in(&mut b, &k, &g, &out);
    b.submit();
    let max = cpu.iter().cloned().fold(0.0, f64::max);
    let worst = cpu.iter().zip(out.download_interior(&ctx, &g)).map(|(a, b)| (a - b as f64).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-6 * max, "max |Δρ_pc| = {worst:.2e} (max ρ_pc {max:.2e})");
}

/// V_NL x on random x vs `qm::nonlocal::NonlocalOp::apply_add`, and E_NL of a set of vectors.
#[test]
fn proj_apply_and_energy_match_cpu() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let h = 0.25;
    let grid = qm::Grid::cube(33, h);
    let (k, g) = (Kernels::new(&ctx), GpuGrid::new(&grid));
    let s = o_setup(h);
    let cpu_nl = qm::nonlocal::NonlocalOp::build(&grid, &nuclei(), &s);
    let mut seed = 7u64;
    let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5 };
    let x: Vec<f64> = (0..grid.len()).map(|_| rnd()).collect();
    let mut cpu = vec![0.0; grid.len()];
    cpu_nl.apply_add(&x, &mut cpu);
    let ion = GpuIonic::new(&ctx, &g, &nuclei(), &s);
    let (xf, out) = (Field::new(&ctx, &g, "x"), Field::new(&ctx, &g, "out"));
    xf.upload_interior(&ctx, &g, &x);
    let mut b = Batch::new(&ctx, "t");
    ion.nonlocal().unwrap().apply_in(&mut b, &k, &g, &xf, &out);
    b.submit();
    let got = out.download_interior(&ctx, &g);
    let scale = cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = cpu.iter().zip(&got).map(|(a, b)| (a - *b as f64).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-5 * scale, "max |ΔV_NL x| = {worst:.2e} of {scale:.2e}");
    let e_gpu = ion.nonlocal().unwrap().energy(&ctx, &k, &g, std::slice::from_ref(&xf));
    let e_cpu = cpu_nl.energy(&[x]);
    assert!((e_gpu - e_cpu).abs() < 1e-6 * e_cpu.abs(), "E_NL {e_gpu} vs {e_cpu}");
}
```
(The test helpers `Field::upload_interior` and `Field::download_interior` exist in `qm-gpu::grid`. If the names differ, use the existing ones.)

`crates/validate/tests/gpu_vs_cpu.rs` additions. They share the suite's GPU lock and skip without an adapter. Setups come from a small `pseudo_setup(policy, h, zs)` helper built like `pseudo.rs`'s, with the tables cached per binary.
```rust
/// Spec §6.3 / §8 row 8 first failing test: pseudo-O (triplet, CoresOnly) energy, GPU vs CPU, at the
/// suite's H₂O-class bound (the 3× rule re-pins it in Step 6 from the measured difference).
const E_BOUND_PSEUDO: f64 = 2e-5;
const F_BOUND_PSEUDO: f64 = 6e-5;

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn gpu_pseudo_o_energy_matches_cpu() {
    let Some(ctx) = gpu() else { return };
    let h = 0.25f64.min(pseudo_h_req(8));
    let grid = Grid::cube(n_for(12.0, h), h);
    let sys = System::with_ionic(vec![Nucleus { pos: [0.07, -0.03, 0.02], z: 8 }], 0, Some(3), pseudo_setup(IonicPolicy::CoresOnly, h, &[8]));
    let settings = settings_for(&grid);
    let (_, cpu) = qm::scf(&grid, &sys, &settings, None).unwrap();
    let mut solver = GpuSolver::new(ctx, &grid);
    let (_, gpu) = solver.solve(&sys, &settings, false).unwrap();
    let d = gpu.energy.total - cpu.energy.total;
    eprintln!("pseudo-O: CPU {:.9} GPU {:.9} Δ {d:+.2e}", cpu.energy.total, gpu.energy.total);
    assert!(d.abs() < E_BOUND_PSEUDO);
}

/// Pseudo-C (CH₄ under CoresOnly: pseudo-C, all-electron H — also a mixed scene) energies and forces.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn gpu_pseudo_c_energy_and_forces_match_cpu() {
    let Some(ctx) = gpu() else { return };
    let h = 0.25f64.min(pseudo_h_req(6));
    let grid = Grid::cube(n_for(12.0, h), h);
    let a = 2.06 / 3f64.sqrt();
    let nuc = vec![Nucleus { pos: [0.02, 0.01, -0.03], z: 6 }, Nucleus { pos: [a, a, a], z: 1 }, Nucleus { pos: [-a, -a, a], z: 1 },
                   Nucleus { pos: [-a, a, -a], z: 1 }, Nucleus { pos: [a, -a, -a], z: 1 }];
    let sys = System::with_ionic(nuc.clone(), 0, None, pseudo_setup(IonicPolicy::CoresOnly, h, &[6, 1]));
    let settings = settings_for(&grid);
    let (cs, cpu) = qm::scf(&grid, &sys, &settings, None).unwrap();
    let fc = qm::forces(&grid, &cs, &nuc, settings.sigma).unwrap();
    let mut solver = GpuSolver::new(ctx, &grid);
    let (gs, gpu) = solver.solve(&sys, &settings, false).unwrap();
    let fg = solver.forces(gs, &nuc, settings.sigma).unwrap();
    assert!((gpu.energy.total - cpu.energy.total).abs() < E_BOUND_PSEUDO);
    let fd = fc.iter().flatten().zip(fg.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("CH₄: ΔE {:+.2e}, max |ΔF| {fd:.2e}", gpu.energy.total - cpu.energy.total);
    assert!(fd < F_BOUND_PSEUDO);
}

/// Review Focus 4 on the GPU: OH under CoresOnly (σ = h H beside σ_LR = 2h O), forces GPU vs CPU.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn gpu_mixed_oh_forces_match_cpu() {
    let Some(ctx) = gpu() else { return };
    let h = 0.25f64.min(pseudo_h_req(8));
    let grid = Grid::cube(n_for(12.0, h), h);
    let nuc = vec![Nucleus { pos: [0.03, 0.02, 0.0], z: 8 }, Nucleus { pos: [1.86, 0.11, 0.0], z: 1 }];
    let sys = System::with_ionic(nuc.clone(), 0, Some(2), pseudo_setup(IonicPolicy::CoresOnly, h, &[8, 1]));
    let settings = settings_for(&grid);
    let (cs, _) = qm::scf(&grid, &sys, &settings, None).unwrap();
    let fc = qm::forces(&grid, &cs, &nuc, settings.sigma).unwrap();
    let mut solver = GpuSolver::new(ctx, &grid);
    let (gs, _) = solver.solve(&sys, &settings, false).unwrap();
    let fg = solver.forces(gs, &nuc, settings.sigma).unwrap();
    let fd = fc.iter().flatten().zip(fg.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(fd < F_BOUND_PSEUDO, "max |ΔF| {fd:.2e}");
}

/// Review Focus 2 on the GPU: a parked state from another ionic setup is not a warm start. The H
/// setup is built with `from_tables` (no checks), as the CPU twin does: after Task 11 the ungated H
/// table is provisional and `IonicSetup::build` refuses it, which is beside the point here.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only oracle; run: cargo test -p validate --release")]
fn gpu_warm_state_from_another_ionic_setup_is_cold() {
    let Some(ctx) = gpu() else { return };
    let h = 0.25;
    let grid = Grid::cube(33, h);
    let settings = settings_for(&grid);
    let nuc = vec![Nucleus { pos: [0.1, 0.0, 0.0], z: 1 }];
    let mut solver = GpuSolver::new(ctx, &grid);
    solver.solve(&System::new(nuc.clone(), 0, None), &settings, false).unwrap();
    let h_table = std::sync::Arc::new(qm::radial::filter::filtered(&qm::radial::generate(1).unwrap(), h));
    let setup = std::sync::Arc::new(qm::IonicSetup::from_tables(qm::ionic::IonicPolicy::All, h, h,
        std::collections::BTreeMap::from([(1, qm::ionic::Ionic::Pseudo(h_table))])));
    let all = System::with_ionic(nuc, 0, None, setup);
    let warm = solver.solve(&all, &settings, true).unwrap().1;
    let cold = solver.solve(&all, &settings, false).unwrap().1;
    assert_eq!(warm.energy.total.to_bits(), cold.energy.total.to_bits());
}
```
`crates/qm-gpu/src/scf.rs` tests: `gpu_none_policy_is_bitwise_today` pastes Step 1's `GPU_BASE_…` lines and asserts the H₂/LiH E and F_x0 bits, as Task 6 did on the CPU.

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test -p qm-gpu --release ionic` and `cargo test -p validate --release --test gpu_vs_cpu gpu_pseudo`
Expected: compile errors (`GpuIonic` missing). Then, after the module skeleton exists, the GPU solver asserts `system.ionic.is_none()` (Task 6) and panics on the pseudo tests.

- [ ] **Step 4: Write the kernels**

`crates/qm-gpu/src/shaders/ionic.wgsl` (prefixed by `erf.wgsl`). Its doc comments state, per kernel, what it computes and its CPU reference.
```wgsl
// Ionic species (M7): CPU references in qm::nuclear (ionic_local_potential, partial_core_field) and
// qm::nonlocal (NonlocalOp). Grid-stride over the interior; ghosts untouched.

struct IonParams { p: u32, n: u32, count: u32, table_len: u32, h: f32, inv_du: f32 }
// One pseudised ion: position as an exact cell + f32 fraction (as nuclear.wgsl), its table offsets and radii.
struct Ion {
    cell: vec4<i32>,           // .w: offset of its V_SR table
    frac: vec4<f32>,           // .xyz fraction of a cell, .w: Z_v
    radii: vec4<f32>,          // .x: sr_radius (the one V_SR table, plan review Q2); .yzw unused
    pc: vec4<f32>,             // .x: offset of its ρ_pc table (as f32-exact integer), .y: pc_radius
}

@group(0) @binding(0) var<uniform> U: IonParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> ions: array<Ion>;
@group(0) @binding(3) var<storage, read> tables: array<f32>;

fn pidx_of(idx: u32, n: u32, p: u32) -> u32 {
    let i = idx % n; let t = idx / n; let j = t % n; let k = t / n;
    return (i + 4u) + p * ((j + 4u) + p * (k + 4u));
}

// 4-point Lagrange on the uniform table (spec §5.1 "Sampling"; Δ = h/256), as qm's FilteredTable::v_sr.
// (The uniforms r0, ln_r0 and inv_delta are replaced by inv_du. V_SR, t_l = F_l/r^l and ρ_pc are even in d, so each
// uniform table carries one mirror node at d = −Δ and the stencil is centred down to d = 0: the central-interval bound
// (9/16, not 1) holds everywhere — `FilteredTable` mirrors index −1 onto index 1.)
fn interp(off: u32, d: f32) -> f32 {
    let s = d * U.inv_du;
    let i0 = u32(clamp(i32(floor(s)) - 1, 0, i32(U.table_len) - 4));
    let t = s - f32(i0);
    let a = tables[off + i0]; let b = tables[off + i0 + 1u]; let c = tables[off + i0 + 2u]; let e = tables[off + i0 + 3u];
    return a * (-(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0) + b * (t * (t - 2.0) * (t - 3.0) / 2.0)
         + c * (-t * (t - 1.0) * (t - 3.0) / 2.0) + e * (t * (t - 1.0) * (t - 2.0) / 6.0);
}

fn dist(i: vec3<i32>, ion: Ion) -> f32 {
    let dc = vec3<f32>(i - ion.cell.xyz) - ion.frac.xyz;
    return length(dc) * U.h;
}

// V_SR: the one filtered table (plan review Q2; no tail branch), added into dst (which already holds V_LR).
@compute @workgroup_size(256)
fn local_sr(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = vec3<i32>(vec3<u32>(idx % n, (idx / n) % n, idx / (n * n)));
        var v = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let ion = ions[a];
            let d = dist(i, ion);
            if (d <= ion.radii.x) { v = v + interp(u32(ion.cell.w), d); }
        }
        let p = pidx_of(idx, n, U.p);
        dst[p] = dst[p] + v;
    }
}

// ρ_pc = Σ table lookups within each ion's pc_radius (written, not added).
@compute @workgroup_size(256)
fn nlcc_deposit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = vec3<i32>(vec3<u32>(idx % n, (idx / n) % n, idx / (n * n)));
        var r = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let ion = ions[a];
            let d = dist(i, ion);
            if (d <= ion.pc.y) { r = r + interp(u32(ion.pc.x), d); }
        }
        dst[pidx_of(idx, n, U.p)] = r;
    }
}
```
The projector kernels go in the same file under their own bind-group layout (`SPECS` entries, one per entry point):
```wgsl
struct ProjParams { n_points: u32, n_m: u32, chunks: u32, pad: u32, dv: f32, e_kb: f32, pad1: f32, pad2: f32 }
@group(0) @binding(0) var<uniform> P: ProjParams;
@group(0) @binding(1) var<storage, read> points: array<u32>;      // padded indices of one sphere
@group(0) @binding(2) var<storage, read> beta: array<f32>;        // [m][point]
@group(0) @binding(3) var<storage, read> x: array<f32>;           // the vector (padded field)
@group(0) @binding(4) var<storage, read_write> partials: array<f32>; // [m][chunk]

var<workgroup> acc: array<f32, 256>;

// partials[m][chunk] = Σ_{points in chunk} β_m x (workgroup tree sum; chunk = workgroup_id.x, m = .y)
@compute @workgroup_size(256)
fn proj_dot(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = wid.y;
    var s = 0.0;
    for (var i = wid.x * 256u + lid.x; i < P.n_points; i = i + P.chunks * 256u) {
        s = s + beta[m * P.n_points + i] * x[points[i]];
    }
    acc[lid.x] = s;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w = w >> 1u) {
        if (lid.x < w) { acc[lid.x] = acc[lid.x] + acc[lid.x + w]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { partials[m * P.chunks + wid.x] = acc[0]; }
}

// coeffs[m] = E_KB · dv · Σ_chunks partials[m][·] — pairwise over ≤ 64 chunks (Spec question 7)
@group(0) @binding(5) var<storage, read_write> coeffs: array<f32>;
@compute @workgroup_size(64)
fn proj_finish(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = wid.x;
    acc[lid.x] = select(0.0, partials[m * P.chunks + lid.x], lid.x < P.chunks);
    workgroupBarrier();
    for (var w = 32u; w > 0u; w = w >> 1u) {
        if (lid.x < w) { acc[lid.x] = acc[lid.x] + acc[lid.x + w]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { coeffs[m] = P.e_kb * P.dv * acc[0]; }
}

// out[points[i]] += Σ_m coeffs[m] β_m[i] — one dispatch per sphere (spheres serialised in the batch)
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn proj_axpy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    for (var i = gid.x; i < P.n_points; i = i + nw.x * 256u) {
        var s = 0.0;
        for (var m = 0u; m < P.n_m; m = m + 1u) { s = s + coeffs[m] * beta[m * P.n_points + i]; }
        out[points[i]] = out[points[i]] + s;
    }
}

// partials[comp][chunk] = Σ w_comp x over the point list (forces: ∇β·ψ, dV_SR/dd·d̂·ρ, dρ_pc/dd·d̂·v̄)
@group(0) @binding(7) var<storage, read> w3: array<f32>;            // [point][3]
var<workgroup> acc3: array<vec3<f32>, 256>;
@compute @workgroup_size(256)
fn gather_dot3(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var s = vec3<f32>(0.0);
    for (var i = wid.x * 256u + lid.x; i < P.n_points; i = i + P.chunks * 256u) {
        s = s + vec3<f32>(w3[3u * i], w3[3u * i + 1u], w3[3u * i + 2u]) * x[points[i]];
    }
    acc3[lid.x] = s;
    workgroupBarrier();
    for (var w = 128u; w > 0u; w = w >> 1u) {
        if (lid.x < w) { acc3[lid.x] = acc3[lid.x] + acc3[lid.x + w]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { let o = 3u * wid.x; partials[o] = acc3[0].x; partials[o + 1u] = acc3[0].y; partials[o + 2u] = acc3[0].z; }
}
```
`chunks` = min(⌈n_points/256⌉, 64), so `proj_finish`'s single 64-thread workgroup covers every chunk. Each kernel's binding list is registered in `SPECS`:
- `local_sr` / `nlcc_deposit`: U, RW, R, R;
- `proj_dot`: U, R, R, R, RW;
- `proj_finish`: U, RW at binding 4 (partials read via read_write), RW at binding 5;
- `proj_axpy`: U, R, R, RW at binding 5, RW at binding 6;
- `gather_dot3`: U, R, R at binding 3, RW at binding 4, R at binding 7.

Per-centre V_LR is an entry point `smeared_potential` in `ionic.wgsl` with its own record `struct Centre { cell: vec4<i32>, frac: vec4<f32> /* .w: charge q */, inv: vec4<f32> /* .x: 1/(σ√2) */ }` bound as `@group(0) @binding(2) var<storage, read> centres: array<Centre>;` (WGSL allows the slot to be shared with `ions` because no entry point uses both). It is `nuclear_potential` with a per-centre width:
```wgsl
// V_LR(r) = −Σ q erf(d/(σ√2))/d per centre (qm::nuclear::smeared_potential)
@compute @workgroup_size(256)
fn smeared_potential(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let n = U.n;
    let total = n * n * n;
    for (var idx = gid.x; idx < total; idx = idx + nw.x * 256u) {
        let i = vec3<i32>(vec3<u32>(idx % n, (idx / n) % n, idx / (n * n)));
        var v = 0.0;
        for (var a = 0u; a < U.count; a = a + 1u) {
            let c = centres[a];
            let d = length(vec3<f32>(i - c.cell.xyz) - c.frac.xyz) * U.h;
            v = v - c.frac.w * erf_over_x(d * c.inv.x) * c.inv.x;
        }
        dst[pidx_of(idx, n, U.p)] = v;
    }
}
```
`local_potential_in` records `smeared_potential` and then `local_sr`.

- [ ] **Step 5: Wire the Rust side**

`crates/qm-gpu/src/ionic.rs`:
- `GpuIonic::new` builds the ion records from `FilteredTable`s. Tables are concatenated `f32` V_SR (`v_sr_table`) and ρ_pc per pseudised ion, padded to `table_len`.
- The lookup forms s = (ln d − ln r₀)/δ in f32 with ln r₀ ≈ −11.5 stored in f32. That offsets the index by ≈ 6e-5, which shifts d by ≈ 5e-7 relative: about the size of `local_sr_matches_cpu`'s 2e-6 bound (plan review NOTE). If that test trips on it, store the split ln r₀ (hi + lo f32) or an f32-exact index base. Never loosen the bound.
- It builds the centre records from `qm::nuclear::centres`.
- It runs `qm::nonlocal::NonlocalOp::build_with_gradients(grid, nuclei, setup)` once (f64, CPU) and uploads each sphere's padded point indices (`GpuGrid::pidx` of the interior index), β [m][pt] as f32 and ∇β [m][pt][3] as f32.
- It precomputes the force weights for every pseudised ion over its SR sphere (dV_SR/dd·d̂) and its ρ_pc sphere (dρ_pc/dd·d̂), in f64, uploaded as f32.

`GpuNonlocal::apply_in` records, per sphere: `proj_dot` (x → partials) → `proj_finish` (→ coeffs = E_KB·dv·c) → `proj_axpy` (out += Σ coeffs β). `GpuNonlocal::energy` records `proj_dot` for every (sphere, orbital) into one partials buffer, then makes one readback. On the CPU, c = Neumaier(partials)·dv in f64 and E_NL = Σ E_KB c² in f64 (rule 4).

`pseudo_electronic_forces`:
- (a) V_LR, all-electron and pseudised alike: the existing `force_partials` kernel per centre, with `c = −q·h/a³` and `h_over_a = h/a` for that centre's (q, σ). This generalises `electronic_forces_with` to `&[Centre]`; the old function stays for the `None` path.
- (b) SR and NLCC: `gather_dot3` of ρ (and of v̄ = ½(v↑ + v↓) on ρ + ½ρ_pc, which the solver keeps as a field after the final XC evaluation) against the uploaded weights.
- (c) NL: `proj_dot` partials for c and `gather_dot3` with ∇β for ⟨∇β|ψ⟩, for every (sphere, orbital), in one submission and one readback. The CPU finishes F = 2 Σ E_KB c ⟨∇β|ψ⟩ in f64.
- (d) Ion–ion in f64 with valence charges (`qm::nuclear::ionic_repulsion`'s pairs; CoincidentNuclei as today).

`crates/qm-gpu/src/scf.rs` (`solve_with`):
- Remove Task 6's `ionic.is_none()` assert.
- When `system.ionic` is `Some(s)`: build `GpuIonic::new` once per solve (geometry), fill `v_nuc` with `local_potential_in`, allocate a `rho_pc` field filled by `partial_core_in`, and run XC on (ρ↑ + ½ρ_pc, ρ↓ + ½ρ_pc) via `axpy` into two scratch fields before `xc_lsda_in`.
- Pass `GpuHamiltonian { v, nl: ionic.nonlocal() }` to LOBPCG.
- In the energy: external += `nl.energy(orbitals_up) + nl.energy(orbitals_dn)`; nuclear = `ionic_repulsion`; `core_overlaps` from `qm::ionic::core_overlaps`, with the warning text as on the CPU.
- `GpuState.ionic = system.ionic.clone()`, and the warm path requires `same_ionic` like the CPU.
- When `system.ionic` is `None`: every line as today (`nuclear_potential_in`, `xc_lsda_in` on ρ↑, ρ↓, `GpuHamiltonian { v, nl: None }`).

`GpuSolver::forces` dispatches on `state.ionic` like `qm::forces`.

`crates/qm-gpu/src/eigen.rs`: `GpuHamiltonian::apply_in` calls `hamiltonian_in` then, if `nl` is `Some`, `nl.apply_in(b, k, g, x, out)`. The preconditioner is unchanged (spec §5.5). Update every `GpuHamiltonian { v }` construction in the crate to `GpuHamiltonian { v, nl: None }`, except the SCF's.

- [ ] **Step 6: Run the tests and pin the bounds**

Run: `cargo test -p qm-gpu --release` (PASS, including `gpu_none_policy_is_bitwise_today` and the three kernel tests), then `cargo test -p validate --release --test gpu_vs_cpu` (PASS).

The 3× rule (as for the existing bounds): measure each pseudo test's |ΔE| and |ΔF|.
- If 3× the measurement is below 2e-5 / 6e-5, re-pin `E_BOUND_PSEUDO`/`F_BOUND_PSEUDO` to it.
- If a measurement exceeds the H₂O class, stop and report. Spec §7 risk 6 names the Kahan path in `proj_dot` partials as the measured remedy; never a looser bound.

Record the measured numbers, the bounds and the timing (pseudo-O solve time vs the all-electron O at the same grid) in `references.md` "GPU vs CPU: pseudopotentials (M7 Task 8)".

Run `cargo test -p validate --release --test dynamics` and `--test surrogate`: PASS, unchanged (the `None` path).

- [ ] **Step 7: Commit**
```bash
git add crates/qm-gpu crates/validate/tests/gpu_vs_cpu.rs crates/validate/references.md
git commit -o -m "feat(qm-gpu): pseudopotential kernels — local_sr, nlcc_deposit, projector dot/finish/axpy, gather forces (M7 Task 8)" -- crates/qm-gpu crates/validate/tests/gpu_vs_cpu.rs crates/validate/references.md
```


**NLCC ruling (2026-10-16), Ruling 6 (Task 8b).**
- The kernels are checked against the CPU at the gpu_vs_cpu bounds; add the NLCC force term of a lone O.
- The GPU HF ΣF qualifies against the f64-energy instrument at the current recipe (≤ 5e-6 per component). Under (C) the -3 `O2_CPU_PROBES` are stale.

---

### Task 9: `sim`/`providers`/`surrogate`/`app` plumbing **[Opus]**

Spec §4.7, §5.5 (`sim`, `surrogate`, `app` bullets), §5.6, §6.5, §8 row 9; review NOTE (the drift readouts reset on a policy switch). Runs in parallel with Task 8.

**Task 6 review addition.** The providers set `Occupations` from the policy (`None` → `Aufbau`, else `DegenerateShells { tau: DEGENERACY_TAU }`); `EvalStats` carries `fractional` and warns on a changed pattern between steps; the surrogate never trains on such a step; provenance gains (occupations, τ); the Readouts occupation line of review ruling A.6 ("open shell: n e shared equally over k degenerate levels (↑/↓), the spherical-atom ensemble; LSDA's symmetry-broken integer state lies ΔE lower (references.md)"). The atom-reference scan converges every candidate under the pseudo policies and keeps O at M = 3, N at M = 4, C at M = 3.

**Files:**
- Modify (sim): `crates/sim/src/scene.rs` (`IonicPolicy`, `Scene.ionic`), `crates/sim/src/provider.rs` (`CORE_OVERLAP_WARNING`, `PseudoSummary`, `EvalStats.{pseudised, core_overlaps}`, `ForceProvider::set_display_partial_core`), `crates/sim/src/references.rs` (`ReferenceKey.ionic`), `crates/sim/src/lib.rs`; every `Scene { … }` literal in the workspace gains `ionic: IonicPolicy::None` (74 today; the compiler lists them)
- Create (providers): `crates/providers/src/pseudo.rs`
- Modify (providers): `crates/providers/src/lib.rs`, `crates/providers/src/ks.rs` (`problem` via the cache, valence checks, notes, the two `EvalStats` fields, the ρ_pc display, the GPU footprint), `crates/providers/src/surrogate/{store.rs,mod.rs,audit.rs}` and `crates/providers/src/sampling/mod.rs` (the provenance tag; core-overlap answers never stored)
- Modify (surrogate): `crates/surrogate/src/model.rs` (`Provenance.ionic`, `describe`), `crates/surrogate/src/sidecar.rs` (version 4), every `Provenance { … }` literal (21 today) gains `ionic: "none".into()`
- Modify (qm-gpu): `crates/qm-gpu/src/scf.rs` (`copy_density_with_core_to_volume`)
- Modify (app): `crates/app/src/scene.rs`, `crates/app/src/main.rs` (`--ionic`), `crates/app/src/worker/{commands.rs,mod.rs,tests.rs}`, `crates/app/src/ui/{mod.rs,actions.rs,menus.rs,search.rs}`, `crates/app/src/ui/dock/{readouts.rs,selection.rs}`, `crates/app/src/{badge.rs,atom_card.rs,overlay.rs,viewer.rs,replay.rs,screenshot.rs}`, `crates/app/replays/*.replay`
- Modify (docs): `README.md` (a "Pseudopotentials" section), `CLAUDE.md` (the `--ionic` flag, the replay pin, the new suites), `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md` (the new controls' homes: Physics ▸ Ionic potential, View ▸ Density ▸ with partial core)

**Interfaces:**
- Consumes: Tasks 5–8 (`qm::ionic::{IonicPolicy, IonicSetup, IonicError, CORE_OVERLAP_WARNING}`, `qm::valence_electrons`, `qm::System::with_ionic`, `qm::radial::{generate, has_core, filter::filtered, PseudoTable, FilteredTable, Refusal, ledger::recipe_hash}`, `ScfStats.core_overlaps`, `GpuSolver` with setups).
- Produces:
```rust
// sim::scene
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum IonicPolicy { #[default] None, CoresOnly, All }
impl IonicPolicy { pub const ALL: [IonicPolicy; 3]; pub fn word(self) -> &'static str /* none|cores|all */;
                   pub fn parse(w: &str) -> Option<IonicPolicy>; pub fn title(self) -> &'static str; }
pub struct Scene { /* … */ pub ionic: IonicPolicy }
// sim::provider
pub const CORE_OVERLAP_WARNING: &str = "cores overlap";            // == qm::ionic::CORE_OVERLAP_WARNING (pinned)
#[derive(Clone, Debug, PartialEq)]
pub struct PseudoSummary { pub z: u8, pub z_v: f64, pub f: f64, pub local: String, pub r_c_max: f64, pub h_req: Option<f64>, pub r_core: f64, pub record: String }
pub struct EvalStats { /* … */ pub pseudised: Vec<PseudoSummary>, pub core_overlaps: Vec<(usize, usize, f64)> }
trait ForceProvider { fn set_display_partial_core(&mut self, _on: bool) {} /* default no-op */ }
// providers::pseudo
pub struct PseudoCache { /* tables, filtered, cores, generated */ }
pub fn cache() -> &'static std::sync::Mutex<PseudoCache>;
impl PseudoCache { pub fn generated(&self) -> u64; }
pub fn qm_policy(p: sim::IonicPolicy) -> qm::ionic::IonicPolicy;
pub fn table_for(z: u8) -> Result<Arc<qm::radial::PseudoTable>, qm::radial::Refusal>;   // the cache's table (Task 10's h_for; dimer-gated from Task 11)
pub fn setup_for(scene: &Scene, sigma_ae: f64, notes: &mut Vec<String>) -> Result<Option<Arc<qm::IonicSetup>>, ProviderError>; // None ⇔ nothing pseudised
pub fn provenance_tag(scene: &Scene) -> String;                     // "none" (nothing pseudised) | "cores/<hash:016x>" | "all/<hash:016x>"
pub fn summaries(setup: &qm::IonicSetup) -> Vec<PseudoSummary>;
// surrogate
pub struct Provenance { /* … */ pub ionic: String }                 // compared by PartialEq; sidecar version 4
// app
pub enum Action { /* … */ Ionic(sim::IonicPolicy), ShowPartialCore(bool) }   // script words: `ionic none|cores|all`, `partial-core true|false`
pub enum Command { /* … */ SetIonicPolicy(sim::IonicPolicy), DisplayPartialCore(bool) }
pub const DEFAULT_IONIC: sim::IonicPolicy = sim::IonicPolicy::CoresOnly;      // app::ui (spec §4.7 "default at first")
```
**Defaults**:
- A `Scene` literal and a scene file without an `ionic` key mean `None`: an old file keeps the physics it was written for, and the validate suites and worker tests keep theirs.
- The app applies `DEFAULT_IONIC` (`CoresOnly`) to presets at start-up, exactly as it applies the Accurate grid (`app::quality`). `--ionic` overrides it, and a `--scene` file keeps its own key.
- `Scene::preset` itself stays the file as written.

- [ ] **Step 1: Write the failing tests**

`crates/providers/src/surrogate/tests.rs`:
```rust
/// Spec §8 row 9 first failing test (spec §5.5): a model taught under one ionic policy is never
/// consulted under another — the surfaces differ by ≈ 38 Ha per O and a bond angle — and the
/// teacher's answer from the other policy is not kept. The tag follows the physics, not the menu:
/// H has no core, so H₂ under CoresOnly *is* today's surface (tag "none") and is answered.
#[test]
fn provenance_refuses_model_across_policies() {
    let (mut s, h) = trained(1.2, 2.2, 40, FallbackPolicy::default());     // taught under IonicPolicy::None
    let mut cores = h2(1.6);
    cores.ionic = sim::IonicPolicy::CoresOnly;
    assert_eq!(s.evaluate(&cores, &mut never()).unwrap().stats.source, Source::Surrogate, "CoresOnly H₂ = the all-electron surface");
    let mut all = h2(1.6);
    all.ionic = sim::IonicPolicy::All;
    let r = s.evaluate(&all, &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert!(r.stats.notes.iter().any(|n| n.contains("differs from the model's") && n.contains("ionic")), "{:?}", r.stats.notes);
    assert_eq!(calls(&h), 1);
    assert_eq!(s.samples(), 40, "no sample from another policy is kept");
    assert_eq!(crate::pseudo::provenance_tag(&cores), "none");
    assert!(crate::pseudo::provenance_tag(&all).starts_with("all/"));
}

/// Spec §5.5: a teacher answer with a core-overlap warning is used, never trained on.
#[test]
fn core_overlap_answers_are_never_kept() {
    struct Overlapping(Arc<Mutex<MorseDimer>>);
    impl ForceProvider for Overlapping {
        fn evaluate(&mut self, s: &Scene, a: &mut dyn FnMut() -> bool) -> Result<ForceResult, ProviderError> {
            let mut r = self.0.lock().unwrap().evaluate(s, a)?;
            r.stats.core_overlaps = vec![(0, 1, 0.5)];
            r.stats.warnings.push(format!("{}: 0–1 at R = 0.500 Bohr", sim::CORE_OVERLAP_WARNING));
            Ok(r)
        }
        fn name(&self) -> &'static str { "overlapping Morse" }
        fn reset(&mut self) {}
    }
    let teacher = Arc::new(Mutex::new(MorseDimer::default()));
    let mut s = SurrogateKs::new(Box::new(Overlapping(teacher)), FallbackPolicy::default());
    let r = s.evaluate(&h2(1.6), &mut never()).unwrap();
    assert_eq!(r.stats.source, Source::Fallback);
    assert_eq!(s.samples(), 0, "a core-overlap answer is never stored");
    assert!(r.stats.notes.iter().any(|n| n.contains("cores overlap")), "{:?}", r.stats.notes);
}
```
(Use the constructors the file already uses for `MorseDimer` and `SurrogateKs` in `trained`. If they differ from those written here, mirror `trained`.)

`crates/providers/src/ks.rs` tests:
```rust
fn o_scene(h: f64, policy: sim::IonicPolicy) -> Scene {
    let n = 2 * (6.0 / h).round() as usize + 1;
    Scene { nuclei: vec![Nucleus::new(8, [0.0; 3])], cell: Cell { n, h }, charge: 0, multiplicity: Some(3), sigma: None,
            external_field: [0.0; 3], ionic: policy }
}

/// Spec §8 row 9 (§5.5, §4.8 T8): a scene coarser than a species' h_req is refused with
/// ProviderError::Invalid naming h and h_req — before any solve.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
fn scene_with_coarser_h_than_h_req_is_invalid() {
    let h_req = qm::radial::generate(8).unwrap().record.h_req.expect("O has h_req");
    for policy in [sim::IonicPolicy::CoresOnly, sim::IonicPolicy::All] {
        let e = CpuKs::new(1e-6).evaluate(&o_scene(h_req + 0.05, policy), &mut || false).unwrap_err();
        assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("coarser") && m.contains("h_req")), "{e:?}");
    }
    // the same scene under None is posed (today's all-electron solver; not solved here)
    assert!(super::problem(&o_scene(h_req + 0.05, sim::IonicPolicy::None), 1e-6, &mut Vec::new()).is_ok());
}

/// Review Focus 1: charge and multiplicity are checked against the valence count under a pseudo
/// policy (O: 6 valence electrons), never panicking in System::with_ionic.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (generates O); cargo test -p providers --release")]
fn charge_and_multiplicity_are_checked_against_valence_electrons() {
    let h = qm::radial::generate(8).unwrap().record.h_req.unwrap();
    let mut s = o_scene(h, sim::IonicPolicy::CoresOnly);
    s.charge = 7;
    s.multiplicity = None;
    let e = super::problem(&s, 1e-6, &mut Vec::new()).unwrap_err();
    assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("charge 7") && m.contains("6 valence electrons")), "{e:?}");
    let mut s = o_scene(h, sim::IonicPolicy::CoresOnly);
    s.multiplicity = Some(9);
    let e = super::problem(&s, 1e-6, &mut Vec::new()).unwrap_err();
    assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("multiplicity 9") && m.contains("6 electrons")), "{e:?}");
    // under None the same charge is posed (8 − 7 = 1 electron)
    let mut s = o_scene(h, sim::IonicPolicy::None);
    s.charge = 7;
    s.multiplicity = None;
    assert!(super::problem(&s, 1e-6, &mut Vec::new()).is_ok());
}

/// Review Focus 5: Li (two valence s states) falls back to all-electron under CoresOnly with a note,
/// and is Invalid under All with the reason and the ONCV stage named.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (radial Li atom); cargo test -p providers --release")]
fn refused_species_falls_back_under_cores_only_and_is_invalid_under_all() {
    let li = |policy| Scene { nuclei: vec![Nucleus::new(3, [0.0; 3])], cell: Cell { n: 33, h: 0.25 }, charge: 0, multiplicity: None,
                              sigma: None, external_field: [0.0; 3], ionic: policy };
    let mut notes = Vec::new();
    let (_, sys, _) = super::problem(&li(sim::IonicPolicy::CoresOnly), 1e-6, &mut notes).expect("posed");
    assert!(sys.ionic.is_none(), "nothing pseudised: Li runs on today's all-electron path");
    assert!(notes.iter().any(|n| n.contains("Li") && n.contains("two valence s states") && n.contains("all-electron")), "{notes:?}");
    let e = super::problem(&li(sim::IonicPolicy::All), 1e-6, &mut Vec::new()).unwrap_err();
    assert!(matches!(&e, ProviderError::Invalid(m) if m.contains("refused") && m.contains("ONCV")), "{e:?}");
}

#[test]
fn core_overlap_warning_text_is_qms() { assert_eq!(sim::CORE_OVERLAP_WARNING, qm::ionic::CORE_OVERLAP_WARNING); }
```
`crates/app/src/worker/tests.rs`:
```rust
/// Spec §8 row 9: the worker generates a species' pseudopotential on first use (a note in the
/// snapshot's EvalStats) and reuses the process-wide cache afterwards; O is pseudised, H is not
/// (CoresOnly), and the energy readout's species list says so.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: runs real SCF solves; cargo test -p app --release")]
fn worker_generates_on_first_use_and_caches() {
    let _gpu = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let mut s = Scene::preset("h2o").unwrap();
    s.ionic = sim::IonicPolicy::CoresOnly;
    (s.grid_n, s.grid_h) = (81, 0.15);
    let before = providers::pseudo::cache().lock().unwrap().generated();
    let w = SimWorker::spawn(s, Backend::Gpu, Some(ctx));
    let first = settled(&w);
    assert_eq!(first.dynamics.last_error, None);
    let zs: Vec<u8> = first.dynamics.eval.pseudised.iter().map(|p| p.z).collect();
    assert_eq!(zs, vec![8], "CoresOnly: O pseudised, H all-electron");
    let after_first = providers::pseudo::cache().lock().unwrap().generated();
    assert!(after_first <= before + 1);
    if after_first == before + 1 {
        assert!(first.dynamics.eval.notes.iter().any(|n| n.contains("generated the O pseudopotential")), "{:?}", first.dynamics.eval.notes);
    }
    w.send(Command::MoveNucleus { index: 1, pos: [1.45, 1.11, 0.0] });
    let second = idle(&w);
    assert_eq!(providers::pseudo::cache().lock().unwrap().generated(), after_first, "the second solve reuses the cache");
    assert!(!second.dynamics.eval.notes.iter().any(|n| n.contains("generated the")), "{:?}", second.dynamics.eval.notes);
}

/// Review NOTE: a policy switch at run time changes E_KS by ≈ 38 Ha per O — the conserved-energy
/// and drift readouts reset as on a grid change.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: runs real SCF solves; cargo test -p app --release")]
fn policy_switch_resets_the_energy_history() {
    let _gpu = gpu_lock();
    let Some(ctx) = gpu_ctx() else { return };
    let w = SimWorker::spawn(h2_33(), Backend::Gpu, Some(ctx));
    settled(&w);
    w.send(Command::SetIonicPolicy(sim::IonicPolicy::All));
    let s = idle(&w);
    assert_eq!(s.dynamics.scene.ionic, sim::IonicPolicy::All);
    assert!(s.dynamics.drift_per_ps.is_none(), "the drift history restarts");
}
```
`crates/surrogate/src/sidecar.rs` tests: `version_4_round_trips_the_ionic_tag` (a model with `ionic = "cores/0123456789abcdef"` saves as `ASMODEL4` and loads equal) and `old_versions_read_as_all_electron` (a v3 file reads `ionic == "none"`). `crates/app/src/ui/tests.rs`: `ionic_action_round_trips` (`Action::Ionic(p).describe()` parses back for all three, and `partial-core true`). `crates/app/src/scene.rs` tests: `ionic_key_round_trips_and_defaults_to_none`.

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo build --workspace --tests`
Expected: compile errors (`Scene.ionic`, `providers::pseudo`, `Provenance.ionic` missing).

- [ ] **Step 3: `sim`**

Add `IonicPolicy` (definition above; `title`: "all-electron (σ = h)", "cores only", "all species") and `Scene.ionic` to `crates/sim/src/scene.rs`. Then run `cargo build --workspace --tests` and add `ionic: IonicPolicy::None` to every `Scene { … }` literal the compiler reports: tests, the validate suites, the providers and the app's `to_sim`.

In `provider.rs`:
- add `CORE_OVERLAP_WARNING` and `PseudoSummary`;
- add the two `EvalStats` fields (`Default`: empty);
- add the default no-op `set_display_partial_core`.

`ReferenceKey` gains `pub ionic: String`, and `ReferenceKey::new(cell, sigma, tol, teacher, z, ionic: &str)`. Every caller passes the scene's `providers::pseudo::provenance_tag(scene)` or the provenance's `ionic`. An atom reference solved under one policy is never reused under another: the valence energies differ by tens of Ha.

- [ ] **Step 4: `providers::pseudo` and the Kohn–Sham adapters**

`crates/providers/src/pseudo.rs`:
```rust
//! The process-wide pseudopotential cache (spec §5.5, `sim` bullet): tables are generated on first use
//! by `qm::radial::generate` on the calling thread (the worker or a side solver; seconds per element)
//! and cached by (Z, recipe_hash); filtered tables by (Z, recipe_hash, h). Refusals are cached too.
//! Nothing is loaded from disk or shipped (spec §3, rule 1).
use qm::ionic::{IonicPolicy as QmPolicy, IonicSetup};
use qm::radial::{filter::{filtered, FilteredTable}, generate, has_core, ledger::recipe_hash, PseudoTable, Refusal};
use sim::{IonicPolicy, ProviderError, PseudoSummary, Scene};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Default)]
pub struct PseudoCache {
    tables: HashMap<(u8, u64), Result<Arc<PseudoTable>, Refusal>>,
    filtered: HashMap<(u8, u64, u64), Arc<FilteredTable>>,
    cores: HashMap<(u8, u64), Result<bool, Refusal>>,
    generated: u64,
}

pub fn cache() -> &'static Mutex<PseudoCache> {
    static CACHE: OnceLock<Mutex<PseudoCache>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

impl PseudoCache {
    pub fn generated(&self) -> u64 { self.generated }

    fn table(&mut self, z: u8, notes: &mut Vec<String>) -> Result<Arc<PseudoTable>, Refusal> {
        let key = (z, recipe_hash());
        if let Some(t) = self.tables.get(&key) { return t.clone(); }
        let t0 = std::time::Instant::now();
        let t = generate(z).map(Arc::new);
        self.generated += 1;
        let sym = units::elements::symbol(z);
        match &t {
            Ok(t) => notes.push(format!("generated the {sym} pseudopotential: f = {}, {}, r_c ≤ {:.2} Bohr, h_req {} ({:.1} s)",
                t.f, t.record.local, t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max),
                t.record.h_req.map_or("none".into(), |h| format!("{h} Bohr")), t0.elapsed().as_secs_f64())),
            Err(r) => notes.push(format!("{sym}: {r}")),
        }
        self.tables.insert(key, t.clone());
        t
    }

    fn filtered(&mut self, z: u8, h: f64, notes: &mut Vec<String>) -> Result<Arc<FilteredTable>, Refusal> {
        let key = (z, recipe_hash(), h.to_bits());
        if let Some(t) = self.filtered.get(&key) { return Ok(t.clone()); }
        let ft = Arc::new(filtered(&self.table(z, notes)?, h));
        self.filtered.insert(key, ft.clone());
        Ok(ft)
    }

    fn has_core(&mut self, z: u8) -> Result<bool, Refusal> {
        self.cores.entry((z, recipe_hash())).or_insert_with(|| has_core(z)).clone()
    }
}

pub fn qm_policy(p: IonicPolicy) -> QmPolicy {
    match p { IonicPolicy::None => QmPolicy::None, IonicPolicy::CoresOnly => QmPolicy::CoresOnly, IonicPolicy::All => QmPolicy::All }
}

/// The table every provider uses for Z — the one place a caller outside the providers reads h_req,
/// so it reads the same table the solves use (plan review M6: from Task 11 that is the dimer-gated
/// table, never the provisional one `qm::radial::generate` returns for Z ≤ dimer_z_max).
pub fn table_for(z: u8) -> Result<Arc<PseudoTable>, Refusal> {
    cache().lock().unwrap_or_else(|e| e.into_inner()).table(z, &mut Vec::new())
}

/// `None` under IonicPolicy::None (`qm::System::new`: today's solver); otherwise the setup for the
/// scene's species at its h. IonicError → ProviderError::Invalid (refused under All, h > h_req, a
/// filtered gate), before any solve. Generation and fallback notes go to `notes`.
pub fn setup_for(scene: &Scene, sigma_ae: f64, notes: &mut Vec<String>) -> Result<Option<Arc<IonicSetup>>, ProviderError> {
    if scene.ionic == IonicPolicy::None { return Ok(None); }
    let zs: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
    // two passes, so each closure borrows alone: the core test for every species (milliseconds),
    // then the filtered tables (generation on first use)
    let cores: HashMap<u8, Result<bool, Refusal>> = zs.iter().map(|&z| (z, c.has_core(z))).collect();
    let mut gen_notes = Vec::new();
    let setup = IonicSetup::build(qm_policy(scene.ionic), scene.cell.h, sigma_ae, &zs,
        &mut |z| cores[&z].clone(),
        &mut |z| c.filtered(z, scene.cell.h, &mut gen_notes))
        .map_err(|e| ProviderError::Invalid(e.to_string()))?;
    notes.append(&mut gen_notes);
    notes.extend(setup.notes.iter().cloned());
    // nothing pseudised (H/He under CoresOnly, a refused species' fallback): today's path, bit for bit
    Ok((!setup.pseudised().is_empty()).then(|| Arc::new(setup)))
}

/// The surface the teacher solves: "none" when nothing is pseudised (policy None; CoresOnly with
/// only core-less or refused species), else "<policy word>/<recipe_hash>".
pub fn provenance_tag(scene: &Scene) -> String {
    let pseudised = match scene.ionic {
        IonicPolicy::None => false,
        IonicPolicy::All => !scene.nuclei.is_empty(),
        IonicPolicy::CoresOnly => {
            let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
            scene.nuclei.iter().any(|n| matches!(c.has_core(n.z), Ok(true)))
        }
    };
    if pseudised { format!("{}/{:016x}", scene.ionic.word(), recipe_hash()) } else { "none".into() }
}

pub fn summaries(setup: &IonicSetup) -> Vec<PseudoSummary> {
    setup.per_z.iter().filter_map(|(z, i)| match i {
        qm::ionic::Ionic::Pseudo(t) => Some(PseudoSummary {
            z: *z, z_v: t.z_v, f: t.record.f, local: t.record.local.clone(),
            r_c_max: t.record.r_c.iter().map(|(_, r)| *r).fold(0.0, f64::max), h_req: t.record.h_req, r_core: t.r_core,
            record: format!("{:#?}", t.record),
        }),
        _ => None,
    }).collect()
}
```
In `crates/providers/src/ks.rs`:
- `problem(scene, tol, notes: &mut Vec<String>)` builds settings first, then `setup_for(scene, settings.sigma, notes)?`. It checks `qm::valence_electrons(&nuclei, setup.as_deref(), scene.charge)`. A negative count under a setup gives `ProviderError::Invalid(format!("charge {} exceeds the {} valence electrons the pseudopotentials leave ({protons} protons; the cores hold the rest)", …))`; the `None` message is unchanged. The multiplicity check uses the same count ("multiplicity {m} impossible for {n} electrons").
- It builds `System::with_ionic` when there is a setup, else `System::new`.
- Both adapters: `EvalStats.pseudised = summaries(setup)`, `EvalStats.core_overlaps = stats.core_overlaps`, and the generation and fallback notes into `notes`.
- `CpuKs`/`GpuKs::set_display_partial_core(on)`. CPU: the display adds `qm::nuclear::partial_core_field` to ρ when on. GPU: `GpuSolver::copy_density_with_core_to_volume` adds the solver's `rho_pc` field into a scratch copy before the texture copy.
- `gpu_footprint_bytes` adds one field (ρ_pc) when `scene.ionic != IonicPolicy::None`.

In the surrogate (`store.rs` `scene_provenance`): `ionic: crate::pseudo::provenance_tag(scene)`. Every place that stores a teacher answer checks `!stats.core_overlaps.is_empty()` first and skips the answer with the note `"surrogate: sample not kept (cores overlap — spec §5.6)"`. Those places are `keep_sample`'s callers in `mod.rs`/`audit.rs` and the sampling pass's `push_sample` path in `sampling/mod.rs`.

`crates/surrogate/src/model.rs`: `Provenance.ionic: String`, compared in `PartialEq`; `describe()` appends `, ionic {}`. `sidecar.rs`:
- magic `ASMODEL4`, the only version written; the module doc's format section gains "Version 4";
- `provenance.ionic` is written as a string;
- versions 1–3 read `ionic = "none"`: every model before M7 was all-electron.

- [ ] **Step 5: The app**

Each item mirrors an existing pattern; the file in brackets is where it lives.
- [`scene.rs`] `pub ionic: sim::IonicPolicy`. `parse` accepts `ionic none|cores|all`; a missing key means `None`. `to_text` writes the line when not `None`. `to_sim`/`from_sim` carry it.
- [`main.rs`, `screenshot.rs`] `--ionic <word>`; default `ui::DEFAULT_IONIC` for presets (as `--quality`). A `--scene` file keeps its key.
- [`worker/commands.rs`, `worker/mod.rs`] `Command::SetIonicPolicy(p)` is one of the commands that "pause the run and replace the scene with the driver's current one with the setting changed" (as `SetSigma`). It resets the provider (`reset`, cold) and the conserved-energy/drift history exactly as `SetGrid` does. `Command::DisplayPartialCore(on)` calls `provider.set_display_partial_core(on)`.
- [`ui/actions.rs`, `ui/mod.rs`] `Action::Ionic(p)` → `Command::SetIonicPolicy(p)` and `UiState.ionic = p`; script word `ionic <word>`. `Action::ShowPartialCore(b)` → `Command::DisplayPartialCore(b)`; script word `partial-core <bool>`.
- [`ui/menus.rs`] Physics ▸ "Ionic potential" (anchor `m-ionic`) is a `theme::segmented` over `IonicPolicy::ALL` with `title()`, and a tooltip. Constant `IONIC_TIP`: "how the nuclei and their core electrons enter: all-electron — every electron on the grid, nuclei smeared at σ = h (heavy atoms lose most of their energy to the smearing); cores only — each species with a core gets a pseudopotential generated from this program's own all-electron atom (H, He stay all-electron); all species — H and He too. Pseudopotential energies are valence energies: not comparable with all-electron totals; energy differences are the physics". It also shows the honesty line "E_KS is a valence energy under a pseudopotential (spec §6.5)". View ▸ Density gains the checkbox "with partial core" (tooltip: "adds ρ_pc — a display of the generator's core model, not solved").
- [`ui/search.rs`] `e("Ionic potential", "pseudopotential all-electron cores valence nlcc", Menu(M::Physics), "m-ionic")` and `e("Partial core density", "nlcc core density display", Menu(M::View), "m-density")`.
- [`ui/dock/readouts.rs`] When `eval.pseudised` is non-empty, the energy row label reads "E_KS (valence, pseudopotential)".
- [`badge.rs`, `atom_card.rs`, `ui/dock/selection.rs`] For a pseudised species, the badge appends "· pseudo, Z_v = 6, r_c = 1.35 Bohr, needs h ≤ 0.20" (from `PseudoSummary`). The Selection tab shows the full generation record (`record`) in a collapsing section "pseudopotential (generation record)".
- [`overlay.rs`, `viewer.rs`] `core_overlap_lines(pos, pairs)` draws each overlapping pair as a red tint segment (`[1.0, 0.35, 0.3, 0.8]`) when `eval.core_overlaps` is non-empty. A toast shows the warning text.
- [`ui/menus.rs`] Scene ▸ Quality: when a level's h exceeds a pseudised species' `h_req` (from the latest `pseudised`), the level is marked "needs h ≤ …" (the T8 warning; the provider refuses such a scene).
- [`replay.rs`, `crates/app/replays/*.replay`] `action ionic …` is the third pin. `replays_pass` refuses a script without it, with the same message shape as the quality/walls pins, and `--record` writes it. Every existing script gets `action ionic none` next to its quality pin, so its recorded physics is unchanged.

- [ ] **Step 6: Run the tests**

Run, in order:
- `cargo test -p sim --release`, `cargo test -p surrogate --release` and `cargo test -p providers --release`: PASS.
- `cargo test -p app --release`: PASS, including the replay scripts with their new pin and the two worker tests.
- `cargo test -p validate --release --test surrogate` and `--test tools`: PASS, unchanged (the scenes are `None`).

- [ ] **Step 7: Docs.**
- `README.md` gets a "Pseudopotentials" section: what the three policies do; that energies are valence energies; that Li/Na/Mg are refused until the ONCV stage; the core-overlap tint; the partial-core display; generation on first use (seconds, cached per run); and that under Balanced (0.25) and Fast (0.30) a scene whose pseudised species has h_req < h is refused with the h_req named — correct behaviour, the T8 certificate (plan review Q11 NOTE).
- `CLAUDE.md` gets the `--ionic` flag, the `action ionic` replay pin, and the default (CoresOnly for presets in the app, `None` for `Scene` literals and old files).
- The UI checklist gets rows for the new controls.

- [ ] **Step 8: Commit**
```bash
git add crates/sim crates/providers crates/surrogate crates/qm-gpu/src/scf.rs crates/app crates/validate README.md CLAUDE.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md
git commit -o -m "feat(sim,providers,surrogate,app): ionic policy, pseudopotential cache, provenance and the panel (M7 Task 9)" -- crates/sim crates/providers crates/surrogate crates/qm-gpu/src/scf.rs crates/app crates/validate README.md CLAUDE.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md
```


**NLCC ruling (2026-10-16), Ruling 6.**
- Show h_eb per element, and a line when the scene's h > h_eb, or h_eb is None: "egg-box force not certified at this grid".
- Never refuse on it.
- The README section ("Pseudopotentials", "Egg-box, honestly"; `nlcc-ruling.md`, with the measured -4 numbers of references.md "Task 7c round 2").

---

### Task 10: validation oracles — atoms, geometries, egg-box record, dynamics re-pins **[Fable — physics-review gate]**

Spec §6.3–§6.7, §8 row 10; review S8 (the basis-set-limit LSDA reference; the angle gated under `All`), M5. **Spec question 8 applies** (accepted by the plan review, conditional on its M6): `All` is usable from Task 6, so this task gates the geometries under `All` with H at its atomic-gate f, and Task 11 re-runs them after the dimer gate. M6's fix makes that sound: `h_for` reads the providers' cached table (the one the solves use, dimer-gated from Task 11) rather than `qm::radial::generate`, and Task 11's `LEDGER.version` bump moves the hash of every table, number and model made here. Plan review S5: the Ne h-series is asserted one-signed, its sign reported, not assumed downward.

**Task 6 review additions.**
- `gpu_oxygen_reference_multiplicity` runs under `DegenerateShells` and asserts spec §6.4's O row (review A.7): both multiplicities converged, E(3) < E(1) − 20 mHa, each within 2 mHa of the radial KB atom in the same configuration, 14 Bohr.
- `h2o_cores_only_eggbox_is_recorded` adds the wall subtraction (`Wall`, duplicated from `pseudo.rs`); its doc line becomes "recorded against the smeared-H egg-box ≈ 5e-7 per H (Task 6 review; the earlier ≤ 4e-5 was the wall term)".
- Every egg-box probe is also reported at h = 0.15 (the app default) and, for C/N/H scenes, at 0.30 (Fast; review C.3, N5) — reported, never moved as the gate's h.

**Files:**
- Create: `crates/validate/tests/pseudo_oracles.rs` (GPU suite)
- Modify: `crates/providers/src/sampling/tests.rs` (`gpu_oxygen_reference_multiplicity` un-ignored, under `CoresOnly`), `crates/validate/references.md` (section "M7 oracles (Task 10)": the pinned literature digits and every measured number), `CLAUDE.md` (the new suites, their run commands and wall times), `.gitignore` (`/crates/validate/last-run-pseudo.md`)

**Interfaces:**
- Consumes: Tasks 6–9 (`sim::Scene.ionic`, `providers::GpuKs` with setups, `providers::pseudo`), `validate/tests/common` (`gpu`, `Locked`, `scene`, `relax`, `solve`, `fit_line`, `au_per_ps`, `timing_row`, `DT_FS`, `never`), `sim::{Dynamics, DynCommand}`.
- Produces: `crates/validate/tests/pseudo_oracles.rs` with the helpers `pseudo_scene(nuclei: Vec<Nucleus>, h: f64, box_bohr: f64, policy: IonicPolicy) -> Scene`, `angle_deg(s: &Scene, vertex: usize, a: usize, b: usize) -> f64` and `h_for(zs: &[u8], wanted: f64) -> f64` (the finest of `wanted` and every species' h_req, read from `providers::pseudo::table_for`). Task 11 extends the file.

- [ ] **Step 1: Look up and pin the literature digits (spec §6.4; review S8).** From NIST CCCBDB ("Calculated geometries", method **SVWN**, the largest available basis of the aug-cc-pVTZ/aug-cc-pVQZ class), record:
  - H₂O r(O–H) in Å and ∠HOH in degrees;
  - NH₃ ∠HNH;
  - the basis name of each;
  - **which SVWN** each row is: CCCBDB lists both SVWN and SVWN5, and Gaussian's "SVWN" is VWN-RPA (VWN3), not VWN5 (plan review NOTE). Pin the one used and name it in `references.md`. The geometry difference (~1e-3 Å, ~0.1°) is far inside ±0.02 Å / ±2°;
  - for H₂O also the Johnson, Gill & Pople, *J. Chem. Phys.* 98, 5612 (1993) SVWN/6-31G* geometry, reported alongside and not pinned (S8).

  Write them into `references.md` "M7 oracles (Task 10)" with the source and access date, and as these constants at the top of `pseudo_oracles.rs`:
```rust
/// NIST CCCBDB, SVWN/<basis read in Step 1> (cited in references.md "M7 oracles"): the basis-set-limit LSDA water.
const LSDA_H2O_ROH_ANGSTROM: f64 = 0.0; // Step 1: replace 0.0 by the CCCBDB digits
const LSDA_H2O_ANGLE_DEG: f64 = 0.0;     // Step 1: replace 0.0 by the CCCBDB digits
const LSDA_NH3_ANGLE_DEG: f64 = 0.0;     // Step 1: replace 0.0 by the CCCBDB digits
/// CH₄: the tetrahedral angle acos(−1/3) — a symmetry, functional-independent (spec §6.4).
const TETRAHEDRAL_DEG: f64 = 109.471_220_634_490_7;
```
These zeros, like Task 1's `NIST_LSD_*`, are the plan's only value-free lines, by the controller's instruction: the digits are read from the source at task time and never typed from memory. Every assertion below fails while they are 0.0. If CCCBDB cannot be reached, stop and report.

- [ ] **Step 2: Write the failing tests** (`crates/validate/tests/pseudo_oracles.rs`). The module doc covers the protocol: GPU (`Locked`, side solvers off), SCF tol 1e-6, relaxation by `common::relax` (the M6 relax tool's quench, "the M6 relax tool" of spec §6.4), h = 0.25 when every species' h_req allows it, else the finest h_req, plus h = 0.20 reported. Release-only; skips without an adapter; each test appends its row to `last-run-pseudo.md`.
```rust
mod common;
use common::*;
use sim::{Cell, IonicPolicy, Nucleus, Scene};

/// The finest of `wanted` and every species' h_req, from the providers' cache: the table the solves
/// themselves use (plan review M6 — `qm::radial::generate` would return the ungated, provisional H).
fn h_for(zs: &[u8], wanted: f64) -> f64 {
    zs.iter().filter_map(|&z| providers::pseudo::table_for(z).ok()?.record.h_req).fold(wanted, f64::min)
}
fn pseudo_scene(nuclei: Vec<Nucleus>, h: f64, box_bohr: f64, policy: IonicPolicy) -> Scene {
    let n = 2 * (box_bohr / (2.0 * h)).round() as usize + 1;
    Scene { nuclei, cell: Cell { n, h }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: policy }
}
fn angle_deg(s: &Scene, v: usize, a: usize, b: usize) -> f64 {
    let d = |i: usize| [0, 1, 2].map(|k| s.nuclei[i].pos[k] - s.nuclei[v].pos[k]);
    let (x, y) = (d(a), d(b));
    let dot: f64 = (0..3).map(|k| x[k] * y[k]).sum();
    let n = |u: [f64; 3]| u.iter().map(|c| c * c).sum::<f64>().sqrt();
    (dot / (n(x) * n(y))).clamp(-1.0, 1.0).acos().to_degrees()
}
fn dist_angstrom(s: &Scene, i: usize, j: usize) -> f64 {
    units::bohr_to_angstrom((0..3).map(|k| (s.nuclei[i].pos[k] - s.nuclei[j].pos[k]).powi(2)).sum::<f64>().sqrt())
}
/// A start away from the answer (112°, 1.85 Bohr), so the relax has to find the minimum.
fn water(policy: IonicPolicy, h: f64) -> Scene {
    let (r, half) = (1.85, 56f64.to_radians());
    pseudo_scene(vec![Nucleus::new(8, [0.0, 0.0, 0.0]), Nucleus::new(1, [r * half.sin(), r * half.cos(), 0.0]),
                      Nucleus::new(1, [-r * half.sin(), r * half.cos(), 0.0])], h, 14.0, policy)
}

/// Spec §8 row 10 first failing test (§6.4; review S8): relaxed water under `All` — ∠HOH within ±2°
/// and r(O–H) within ±0.02 Å of the basis-set-limit LSDA geometry; the `CoresOnly` numbers (σ = h H
/// beside pseudo-O) recorded, not gated: their miss is the measured cost of the remaining smearing.
/// Today's smeared surface relaxes to ≈ 177° (`h2o_under_none_is_the_smeared_177`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_angle_within_2deg_of_lsda() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8, 1], 0.25);
    let mut log = String::new();
    let (mut p, _) = Locked::new(ctx.clone(), 1e-6);
    let (all, _, steps) = relax(&mut p, water(IonicPolicy::All, h), &mut log);
    let (ang, roh) = (angle_deg(&all, 0, 1, 2), dist_angstrom(&all, 0, 1));
    let (core, _, _) = relax(&mut p, water(IonicPolicy::CoresOnly, h), &mut log);
    eprintln!("H₂O at h = {h}: All {ang:.2}°, {roh:.4} Å ({steps} relax steps); CoresOnly {:.2}°, {:.4} Å (recorded); LSDA {LSDA_H2O_ANGLE_DEG}°, {LSDA_H2O_ROH_ANGSTROM} Å",
              angle_deg(&core, 0, 1, 2), dist_angstrom(&core, 0, 1));
    assert!((ang - LSDA_H2O_ANGLE_DEG).abs() <= 2.0, "∠HOH {ang:.2}° vs {LSDA_H2O_ANGLE_DEG}°");
    assert!((roh - LSDA_H2O_ROH_ANGSTROM).abs() <= 0.02, "r(O–H) {roh:.4} Å vs {LSDA_H2O_ROH_ANGSTROM} Å");
}

/// The discriminating control: the same relax under `None` lands on the smeared near-linear
/// minimum (spec §1.2: 177.2°) — the test above would fail there. Ignored (minutes; a record).
#[test]
#[ignore = "control for h2o_angle_within_2deg_of_lsda: the all-electron σ = 0.25 surface is near-linear (spec §1.2)"]
fn h2o_under_none_is_the_smeared_177() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let (mut p, _) = Locked::new(ctx, 1e-6);
    let (s, _, _) = relax(&mut p, water(IonicPolicy::None, 0.25), &mut String::new());
    let ang = angle_deg(&s, 0, 1, 2);
    eprintln!("H₂O under None: {ang:.2}°");
    assert!(ang > 170.0);
}

/// Spec §6.4: NH₃ ∠HNH within ±2° of the LSDA value (All).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn nh3_angle_within_2deg_of_lsda() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[7, 1], 0.25);
    let (r, t) = (1.95, 112f64.to_radians());           // start: flattened pyramid
    let hs: Vec<Nucleus> = (0..3).map(|k| { let phi = k as f64 * 2.0 * std::f64::consts::PI / 3.0;
        Nucleus::new(1, [r * (t / 2.0).sin() * phi.cos(), r * (t / 2.0).sin() * phi.sin(), -r * (t / 2.0).cos() * 0.6]) }).collect();
    let mut nuc = vec![Nucleus::new(7, [0.0; 3])];
    nuc.extend(hs);
    let (mut p, _) = Locked::new(ctx, 1e-6);
    let (s, _, _) = relax(&mut p, pseudo_scene(nuc, h, 14.0, IonicPolicy::All), &mut String::new());
    let angles = [angle_deg(&s, 0, 1, 2), angle_deg(&s, 0, 2, 3), angle_deg(&s, 0, 1, 3)];
    eprintln!("NH₃ at h = {h}: {angles:?} vs {LSDA_NH3_ANGLE_DEG}°");
    for a in angles { assert!((a - LSDA_NH3_ANGLE_DEG).abs() <= 2.0, "∠HNH {a:.2}°"); }
}

/// Spec §6.4: CH₄ relaxed from a distorted start — every ∠HCH within ±0.5° of 109.47°.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn ch4_angle_is_tetrahedral_within_half_degree() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[6, 1], 0.25);
    let a = 2.06 / 3f64.sqrt();
    let nuc = vec![Nucleus::new(6, [0.0; 3]), Nucleus::new(1, [a * 1.05, a, a]), Nucleus::new(1, [-a, -a * 0.97, a]),
                   Nucleus::new(1, [-a, a, -a * 1.03]), Nucleus::new(1, [a, -a, -a])];
    let (mut p, _) = Locked::new(ctx, 1e-6);
    let (s, _, _) = relax(&mut p, pseudo_scene(nuc, h, 14.0, IonicPolicy::All), &mut String::new());
    for (i, j) in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)] {
        let ang = angle_deg(&s, 0, i, j);
        assert!((ang - TETRAHEDRAL_DEG).abs() <= 0.5, "∠H{i}CH{j} = {ang:.3}°");
    }
}

/// Spec §6.3: the h-convergence of pseudo-Ne on the GPU — monotone over the ladder h ≤ h_req, i.e.
/// the successive differences E(h_{k+1}) − E(h_k) all have one sign; that sign is reported, not
/// assumed (plan review S5: the 8th-order FD symbol underestimates q², pulling coarse-h energies
/// down, while the per-h filter smooths V_SR, pushing them up — the net sign is not derivable a
/// priori). |E(0.25) − E(0.15)| reported.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn ne_h_convergence_is_monotone() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h_req = providers::pseudo::table_for(10).unwrap().record.h_req.unwrap();
    let mut es = Vec::new();
    for &h in qm::radial::LEDGER.h_ladder.iter().filter(|&&h| h <= h_req) {
        let (mut p, _) = Locked::new(ctx.clone(), 1e-6);
        let s = pseudo_scene(vec![Nucleus::new(10, [0.0; 3])], h, 12.0, IonicPolicy::CoresOnly);
        es.push((h, solve(&mut p, &s, "Ne").energy));
    }
    let diffs: Vec<f64> = es.windows(2).map(|w| w[1].1 - w[0].1).collect();
    // 1e-6: the SCF tolerance, as the original one-sided check allowed
    let sign = if diffs.iter().all(|&d| d <= 1e-6) { "downward (E falls as h refines)" } else if diffs.iter().all(|&d| d >= -1e-6) { "upward (E rises as h refines)" } else { "mixed" };
    eprintln!("pseudo-Ne E(h): {es:?}; successive differences {diffs:?}: {sign}");
    assert_ne!(sign, "mixed", "not monotone: {es:?}");
}

/// Spec §6.6 (review M5): H₂O under `CoresOnly` keeps two smeared H's whose egg-box force is
/// ≤ 4e-5 each — |Σ F| over the 8 shifts is recorded against that floor ("expected ≈ 4e-5, the
/// smeared-H residual"), not gated. The y-component reported next to today's −3.4e-4.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_cores_only_eggbox_is_recorded() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8], 0.25);
    let (mut p, _) = Locked::new(ctx, 1e-6);
    let (base, _, _) = relax(&mut p, water(IonicPolicy::CoresOnly, h), &mut String::new());
    let q = h / 4.0;
    let mut sums = Vec::new();
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let mut s = base.clone();
        for n in &mut s.nuclei { n.pos[0] += sx; n.pos[1] += sy; n.pos[2] += sz; }
        let f = solve(&mut p, &s, "shift").forces;
        sums.push([0, 1, 2].map(|k| f.iter().map(|x| x[k]).sum::<f64>()));
    }}}
    let max = sums.iter().map(|v| v.iter().map(|c| c * c).sum::<f64>().sqrt()).fold(0.0, f64::max);
    eprintln!("H₂O CoresOnly egg-box: max |ΣF| {max:.2e} (smeared-H floor up to 8e-5); ΣF_y {:?}", sums.iter().map(|v| v[1]).collect::<Vec<_>>());
    assert!(max.is_finite());
}

/// Spec §6.7: NVE drift on a pseudo system (O₂, CoresOnly = all-pseudo, triplet) at the dynamics
/// suite's bound — |slope of E_total| ≤ 2e-4 Ha/ps (`dynamics.rs` DRIFT_BOUND) over 2000 steps.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_o2_nve_drift_is_bounded() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8], 0.25);
    let mut s = pseudo_scene(vec![Nucleus::new(8, [-1.17, 0.0, 0.0]), Nucleus::new(8, [1.17, 0.0, 0.0])], h, 10.0, IonicPolicy::CoresOnly);
    s.multiplicity = Some(3);
    let (p, _) = Locked::new(ctx, 1e-6);
    let mut d = sim::Dynamics::new(s, Box::new(p), DT_FS, 7);
    let (mut t, mut e) = (Vec::new(), Vec::new());
    for i in 1..=2000 {
        d.apply(sim::DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert!(snap.last_error.is_none(), "step {i}: {:?}", snap.last_error);
        t.push(snap.sim_time / au_per_ps());
        e.push(snap.total);
    }
    let (slope, _, _) = fit_line(&t, &e);
    eprintln!("pseudo-O₂ NVE: drift {slope:+.2e} Ha/ps");
    assert!(slope.abs() <= 2e-4);
}

/// Spec §6.7: CSVR on a pseudo system reaches its target at the dynamics suite's bounds
/// (T = 300 K, τ = 5 fs, 2800 steps, settle 100, |⟨T⟩/T − 1| ≤ 0.2) — O₂, CoresOnly.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_o2_csvr_reaches_target() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8], 0.25);
    let mut s = pseudo_scene(vec![Nucleus::new(8, [-1.14, 0.0, 0.0]), Nucleus::new(8, [1.14, 0.0, 0.0])], h, 10.0, IonicPolicy::CoresOnly);
    s.multiplicity = Some(3);
    let (p, _) = Locked::new(ctx, 1e-5);
    let mut d = sim::Dynamics::new(s, Box::new(p), DT_FS, 7);
    d.apply(sim::DynCommand::SetThermostat(Some((300.0, 5.0))));
    let mut temps = Vec::new();
    for i in 1..=2800 {
        d.apply(sim::DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert!(snap.last_error.is_none(), "step {i}: {:?}", snap.last_error);
        if i > 100 { temps.push(snap.temperature); }
    }
    let mean = temps.iter().sum::<f64>() / temps.len() as f64;
    eprintln!("pseudo-O₂ CSVR: ⟨T⟩ = {mean:.1} K");
    assert!((mean / 300.0 - 1.0).abs() <= 0.2);
}
```
In `crates/providers/src/sampling/tests.rs`, `gpu_oxygen_reference_multiplicity` loses its `#[ignore]`. Its scene gains `ionic: IonicPolicy::CoresOnly` on a grid within O's h_req. It asserts the reference multiplicity is 3, with converged triplet and singlet solves and E(3) < E(1) − 0.020 (§6.4 "O atom: triplet < singlet, > 20 mHa"). Its doc records the M6 all-electron numbers it replaces.

- [ ] **Step 3: Verify the tests discriminate**

Run: `cargo test -p validate --release --test pseudo_oracles -- --ignored h2o_under_none_is_the_smeared_177`
Expected: PASS at ≈ 177°, the known limitation. `h2o_angle_within_2deg_of_lsda` would fail there, so it tests something.

- [ ] **Step 4: Run the suite**

Run: `cargo test -p validate --release --test pseudo_oracles` and `cargo test -p providers --release gpu_oxygen_reference_multiplicity`
Expected: PASS.

Any miss is reported with its numbers and the task stops (rule 6). In particular:
- a `CoresOnly` angle outside ±2° is a *recorded* number, not a failure (spec §6.4);
- an `All` miss goes to the physics reviewer with the h = 0.20 run, the H f used and Spec question 8 (Task 11's dimer gate may move H's f).

- [ ] **Step 5: Record.** `references.md` "M7 oracles (Task 10)" gets:
  - the pinned digits with their source and which SVWN (VWN-RPA or VWN5) they are;
  - every measured angle and length (`All` and `CoresOnly`, h = 0.25 and 0.20);
  - the Ne h-series, the sign of its convergence, and |E(0.25) − E(0.15)| (spec §6.3: "the honest statement of what h = 0.25 costs");
  - the O splitting vs the radial spherical value (documented, not gated: the 3-D integer-occupation atom is non-spherical);
  - the `CoresOnly` H₂O egg-box against the floor;
  - the drift and ⟨T⟩;
  - timings.

  `CLAUDE.md` gains the lines for `--test pseudo_gen`, `--test pseudo` and `--test pseudo_oracles`, with measured wall times. Add `/crates/validate/last-run-pseudo.md` to `.gitignore`.

- [ ] **Step 6: Commit**
```bash
git add crates/validate/tests/pseudo_oracles.rs crates/providers/src/sampling/tests.rs crates/validate/references.md CLAUDE.md .gitignore
git commit -o -m "test(validate): M7 oracles — H2O/NH3/CH4 geometries, Ne/O atoms, egg-box record, pseudo dynamics (M7 Task 10)" -- crates/validate/tests/pseudo_oracles.rs crates/providers/src/sampling/tests.rs crates/validate/references.md CLAUDE.md .gitignore
```
Before merging, hand the numbers to `physics-reviewer`.


**NLCC ruling (2026-10-16), Ruling 6.**
- The geometry oracles are unaffected (δθ ≈ 0.03°, δr ≈ 1e-4 Bohr against ±2° and ±0.02 Å), but run on -4.
- The egg-box record is §6.6 per the outcome.
- Any pseudo dynamics test bounds momentum drift from the recorded G, or turns the COM correction on; never from 3e-5.

---

### Task 11: `IonicPolicy::All` — pseudo-H/He, the dimer gate on f, H₂, the H₂O egg-box gate, the `h2o` preset gate, the default **[Fable — physics-review gate]**

Spec §4.4 (S6b, the dimer gate), §4.7 (`All` as the intended default once its gates pass), §6.4 (H₂ under `All`), §6.6 (the egg-box gate under `All`), §6.7, §8 row 11. **Spec question 9 applies** (accepted by the plan review with notes): the dimer gate is a 3-D computation that the radial code cannot run. It is supplied as a callback, and He's AE dimer has no minimum in the bracket.

**Task 6 review addition.** `h2o_all_max_shift_sum_of_forces_below_3e_5` adds the wall subtraction and its assertions (`wall_max ≤ 3e-4`, corrected < 3e-5; raw, wall and corrected printed; the {0, h/2}³ energies wall-subtracted), and reports the probe at h = 0.15 beside it.

**Plan review M6 lands here: two different H potentials must never share one `recipe_hash`.** `generate_with(z, gate)` changes H's f, and with it the table, h_req and E_KB, while `generate(z)` without a gate still returns the ungated H. Before this fix the ledger already held the dimer fields, so the hash did not move: `pseudo_gen` pinned the ungated H, Task 10's `h_for` read it, `PseudoCache` and the surrogate's provenance used the gated one, and `IonicSetup::same_physics` (hash only) let a warm start cross them. The fix:
- (a) For Z ≤ `dimer_z_max`, `generate` (no gate) returns a table marked **provisional** (`PseudoTable.provisional`, carried into `FilteredTable.provisional`), and `IonicSetup::build` refuses it (`IonicError::Provisional`). Only the providers' cache, which always passes the gate, makes usable H/He tables. `pseudo_gen` (CPU, no gate) shows the H/He rows as "provisional".
- (b) `LEDGER.version` becomes `"m7-tm-kb-nlcc-3"`: every table, Task 10 number and surrogate model made under the ungated H is invalid by hash.
- (c) The global rule and its test exist since Task 4 (Global Constraints).

**Files:**
- Modify: `crates/qm/src/radial/recipe.rs` (`DimerGate`, `DimerPair`, `generate_with`; `generate` = `generate_with(z, None)`; `PseudoTable.provisional`), `crates/qm/src/radial/filter.rs` (`FilteredTable.provisional`), `crates/qm/src/radial/record.rs` (the dimer column in `records_table`), `crates/qm/src/radial/ledger.rs` (`version: "m7-tm-kb-nlcc-3"`), `crates/qm/src/ionic.rs` (`IonicError::Provisional`, refused in `check`), `crates/qm/src/radial/mod.rs` (re-export `generate_with`, `DimerGate`, `DimerPair`), `crates/providers/src/pseudo.rs` (`GridDimerGate`, `DimerBackend`, `set_dimer_backend`; `PseudoCache::table` generates through the gate), `crates/app/src/main.rs` (sets the dimer backend to the window's GPU, or CPU without one), `crates/app/src/ui/mod.rs` (`DEFAULT_IONIC`, Step 6), `crates/validate/tests/{pseudo_gen,pseudo_oracles}.rs`, `crates/validate/tests/common/mod.rs` (`set_dimer_backend` at suite start), `crates/validate/references.md` (sections "M7 `All` (Task 11)" and "M7 default policy"; the regenerated records block), `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: Tasks 1–10, plus `qm::extrapolate_sigma` and the σ-ladder rule of `validate::sigma_ladder` (h·{1.0, 1.2, 1.4, 1.6}; restated in the gate's doc and not imported, since providers do not depend on validate).
- Produces:
```rust
// qm::radial::recipe
pub struct DimerPair { pub pp: (f64, f64), pub ae: (f64, f64) }        // (R_e Bohr, ω a.u.) pseudised and AE σ → 0
pub trait DimerGate { fn dimers(&mut self, table: &PseudoTable) -> Result<DimerPair, String>; }
pub fn generate_with(z: u8, dimer: Option<&mut dyn DimerGate>) -> Result<PseudoTable, Refusal>;
pub struct PseudoTable { /* … */ pub provisional: bool }                // Z ≤ dimer_z_max generated without the gate (M6)
// qm::radial::filter
pub struct FilteredTable { /* … */ pub provisional: bool }              // copied from the PseudoTable
// qm::ionic
pub enum IonicError { /* … */ Provisional { z: u8 } }                  // build refuses a provisional table
// providers::pseudo
pub enum DimerBackend { Cpu, Gpu(gpu::GpuContext) }
pub fn set_dimer_backend(b: DimerBackend);                              // process-wide; default Cpu
pub struct GridDimerGate { pub backend: DimerBackend, ae_cache: std::collections::HashMap<u8, Result<(f64, f64), String>> }
impl qm::radial::recipe::DimerGate for GridDimerGate { /* … */ }
pub fn pseudo_dimer(table: &PseudoTable, backend: &DimerBackend) -> Result<(f64, f64), String>;   // (R_e, ω a.u.) at the finest ladder h
```
**The gate** (Spec question 9 ruling):
- It applies for Z ≤ `LEDGER.dimer_z_max`, inside the f ladder after the atomic gates pass, in the same descending order.
- On the finest ladder h (`LEDGER.h_ladder` last, 0.15 Bohr, 12 Bohr box), both dimers are located by bisection of the bond force within `LEDGER.dimer_bracket`·r_peak (the outermost valence maximum) to `LEDGER.dimer_re`/10. k is the central difference of the force with step h, and ω = √(k/μ) with μ from the most abundant isotope's mass (the one per-element datum allowed).
- The AE dimer's force at each R is extrapolated σ → 0 by `qm::extrapolate_sigma` over σ = h·{1.0, 1.2, 1.4, 1.6} (the oracle rule).
- Pass ⇔ |R_e^PP − R_e^AE| < `dimer_re` and |ω^PP/ω^AE − 1| < `dimer_omega_rel`.
- If the AE dimer has no force zero in the bracket (He₂: LSDA's well lies near 5 Bohr, outside [1, 4]·r_peak(1s) ≈ [0.57, 2.3]), the gate is "not applicable" for that element, a derived outcome recorded in the generation record. The record then also holds the PP and AE bond forces at both bracket ends (plan review Q9(ii), report only): the sandbox's He–He collisions live on that repulsive wall.
- The bracket is physical, not tuned (Q9(i), the ledger doc): covalent R_e ≈ 1.5–3 r_peak (H₂: 1.45), and LSDA's spurious rare-gas minima lie beyond 4 r_peak.
- k from F(R_e ± h) with h = 0.15 carries an anharmonic bias: for a Morse curve k_FD = k(1 + 7a²h²/6), so ω is ≈ +1.4 % for H₂ (Q9(iii)). It cancels inside the gate (the same procedure on both sides), but not against the curve-fitted 4405 cm⁻¹ of `pseudo_h2_omega_within_5pct_of_sigma0_curve`, whose doc states it.
- Extrapolating the AE force at each R over σ = h·{1, 1.2, 1.4, 1.6} is consistent (d/dR of the energy ansatz has the same σ dependence), and Zσ_max = 0.24 for H at h = 0.15 is inside the asymptotic regime (Q9).
- Costs (reported): ≈ 60 solves per f on the GPU (≈ 30 s); on the CPU fallback ≈ 40 min per element. The AE curve is cached per Z, so later f's cost only the PP solves.

- [ ] **Step 1: Write the failing tests**

`crates/validate/tests/pseudo_oracles.rs`:
```rust
/// Spec §8 row 11 first failing test (§6.4): pseudo-H₂ under `All` against this solver's own σ → 0
/// all-electron curve (references.md: R_e 1.451 Bohr, ω 4405 ± 60 cm⁻¹, that number's stated bias):
/// R_e within 0.02 Bohr, ω within 5 %; reported against LSDA (4201) and experiment (4401).
/// Stated bias (plan review Q9(iii)): `pseudo_dimer` takes k from F(R_e ± h), h = 0.15, and for a
/// Morse curve k_FD = k(1 + 7a²h²/6), so this ω reads ≈ +1.4 % high against the curve-fitted 4405 —
/// inside the 5 % bound, not removed from it.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_h2_omega_within_5pct_of_sigma0_curve() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    providers::pseudo::set_dimer_backend(providers::pseudo::DimerBackend::Gpu(ctx.clone()));
    let t = qm::radial::generate_with(1, Some(&mut providers::pseudo::GridDimerGate::new(providers::pseudo::DimerBackend::Gpu(ctx.clone())))).expect("pseudo-H generates");
    let (r_e, w) = providers::pseudo::pseudo_dimer(&t, &providers::pseudo::DimerBackend::Gpu(ctx)).expect("pseudo-H₂ is bound");
    let cm = w * HARTREE_IN_CM;
    eprintln!("pseudo-H₂ (f = {}): R_e {r_e:.4} Bohr, ω {cm:.0} cm⁻¹ — AE σ→0 1.451 / 4405; LSDA 4201, expt 4401; dimer gate: {}", t.f, t.record.dimer);
    assert!((r_e - 1.451).abs() <= 0.02, "R_e {r_e:.4}");
    assert!((cm / 4405.0 - 1.0).abs() <= 0.05, "ω {cm:.0} cm⁻¹");
}

/// Spec §6.6 (review M5) under `All`: relaxed H₂O, 8 rigid shifts of ±h/4 — max |Σ_I F_I| < 3e-5
/// Ha/Bohr (the pseudopotential is the only egg-box source); the y-component reported, and the E rms
/// from 8 further shifts {0, h/2}³ (plan review M4: the ±h/4 energies are blind to the first shell;
/// the ±h/4 forces are not).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_all_max_shift_sum_of_forces_below_3e_5() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8, 1], 0.25);
    let (mut p, _) = Locked::new(ctx, 1e-6);
    let (base, _, _) = relax(&mut p, water(IonicPolicy::All, h), &mut String::new());
    let shifted = |s: [f64; 3]| { let mut sc = base.clone(); for n in &mut sc.nuclei { for k in 0..3 { n.pos[k] += s[k]; } } sc };
    let q = h / 4.0;
    let (mut worst, mut fy) = (0.0f64, Vec::new());
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let r = solve(&mut p, &shifted([sx, sy, sz]), "shift");
        let sum = [0, 1, 2].map(|k| r.forces.iter().map(|f| f[k]).sum::<f64>());
        worst = worst.max(sum.iter().map(|c| c * c).sum::<f64>().sqrt());
        fy.push(sum[1]);
    }}}
    let half = h / 2.0;
    let mut es = Vec::new();
    for sx in [0.0, half] { for sy in [0.0, half] { for sz in [0.0, half] {
        es.push(solve(&mut p, &shifted([sx, sy, sz]), "energy shift").energy);
    }}}
    let mean = es.iter().sum::<f64>() / 8.0;
    let rms = (es.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    eprintln!("H₂O All at h = {h}: max |ΣF| {worst:.2e} (±h/4; today's ΣF_y −3.4e-4), ΣF_y {fy:?}, E rms {rms:.2e} ({{0, h/2}}³)");
    assert!(worst < 3e-5);
}

/// Spec §6.7: `thermostat_h2o_preset_known_limitation` becomes a gate under `All` — the `h2o`
/// preset (physical geometry) is within 0.02 Ha of the relaxed minimum, 600 NVE steps from rest
/// stay below 1000 K, and no warm SCF needed a cold retry.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_preset_dynamics_gate() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8, 1], 0.25);
    let preset = pseudo_scene(vec![Nucleus::new(8, [0.0, 0.0, 0.0]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])], h, 14.25, IonicPolicy::All);
    let (mut p, _) = Locked::new(ctx.clone(), 1e-6);
    let e_preset = solve(&mut p, &preset, "preset").energy;
    let (relaxed, _, _) = relax(&mut p, preset.clone(), &mut String::new());
    let e_min = solve(&mut p, &relaxed, "relaxed").energy;
    eprintln!("h2o preset under All: E − E_min = {:.4} Ha", e_preset - e_min);
    assert!(e_preset - e_min <= 0.02);
    let (p, _) = Locked::new(ctx, 1e-5);
    let mut d = sim::Dynamics::new(preset, Box::new(p), DT_FS, 7);
    let mut t_max = 0.0f64;
    for i in 1..=600 {
        d.apply(sim::DynCommand::Step(1));
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(s.last_error.is_none(), "step {i}: {:?}", s.last_error);
        assert!(!s.eval.notes.iter().any(|n| n.contains("retried cold")), "step {i}: a warm SCF stalled: {:?}", s.eval.notes);
        t_max = t_max.max(s.temperature);
    }
    eprintln!("h2o preset NVE from rest: max T {t_max:.0} K");
    assert!(t_max < 1000.0);
}
```
`crates/validate/tests/pseudo_gen.rs`:
```rust
/// Spec §8 row 11: H and He pass the atomic gates (the explicit no-core branch: Z_v = Z, no NLCC);
/// their rows join the pinned records (ZS gains 1 and 2; the block is regenerated). Generated here
/// without the dimer gate, they are provisional (plan review M6): usable by no setup.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn hydrogen_and_helium_pass_the_atomic_gates() {
    for z in [1u8, 2] {
        let t = qm::radial::generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
        assert_eq!((t.n_core, t.z_v), (0, z as f64));
        assert!(t.partial_core.r_nlc.is_none() && t.partial_core.r_core == 0.0);
        assert!(t.provisional && t.record.dimer.starts_with("provisional"), "Z = {z}: {}", t.record.dimer);
        let e = qm::IonicSetup::build(qm::ionic::IonicPolicy::All, 0.15, 0.15, &[z], &mut |_| Ok(false),
            &mut |z| Ok(std::sync::Arc::new(qm::radial::filter::filtered(&qm::radial::generate(z)?, 0.15)))).unwrap_err();
        assert!(matches!(e, qm::ionic::IonicError::Provisional { .. }), "Z = {z}: {e}");
        eprintln!("Z = {z}: f {}, {}, h_req {:?}, gates {:?}", t.f, t.record.local, t.record.h_req, t.gates);
    }
}
```
and `ZS` becomes `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. The records block is regenerated with the dimer column (`records_table` gains a last column "dimer" showing `record.dimer`). Generation there runs without a gate (`generate`), so the H/He rows read "provisional (dimer gate not run)". The dimer-gated f is recorded by `pseudo_oracles` (below) and in references.md "M7 `All`". The block's hash line changes with the version bump.

`crates/providers/src/pseudo.rs` tests (GPU, skip without an adapter):
```rust
/// The dimer gate runs inside the ladder for H (and is not applicable for He), and its verdict is in
/// the generation record.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only (GPU dimer solves); cargo test -p providers --release")]
fn dimer_gate_runs_for_hydrogen_and_is_not_applicable_for_helium() {
    let Some(ctx) = gpu::GpuContext::new_headless() else { return };
    let mut gate = GridDimerGate::new(DimerBackend::Gpu(ctx));
    let h = qm::radial::generate_with(1, Some(&mut gate)).expect("H generates");
    assert!(h.record.dimer.contains("pass"), "{}", h.record.dimer);
    let he = qm::radial::generate_with(2, Some(&mut gate)).expect("He generates");
    assert!(he.record.dimer.starts_with("not applicable"), "{}", he.record.dimer);
    // review Q9(ii): the not-applicable record carries the PP and AE bond forces at both bracket ends
    assert!(he.record.dimer.contains("PP F") && he.record.dimer.contains("AE F"), "{}", he.record.dimer);
    assert!(!h.provisional && !he.provisional, "gated tables are usable");
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p validate --release --test pseudo_oracles pseudo_h2_omega`
Expected: compile error (`generate_with`, `GridDimerGate`, `pseudo_dimer` missing).

- [ ] **Step 3: Implement the gate in the ladder** (`crates/qm/src/radial/recipe.rs`)
```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimerPair { pub pp: (f64, f64), pub ae: (f64, f64) }

/// The molecular gate on f (spec §4.4, review S6b; Spec question 9): the caller computes the
/// pseudised homonuclear dimer and the element's own all-electron σ → 0 dimer on the grid.
/// `Err(why)`: the AE dimer has no bound minimum in the bracket — the gate is not applicable.
pub trait DimerGate { fn dimers(&mut self, table: &PseudoTable) -> Result<DimerPair, String>; }

pub fn generate(z: u8) -> Result<PseudoTable, Refusal> { generate_with(z, None) }

/// The whole ladder (Tasks 3–5 and this one): reference atom → partition → partial core → for each
/// (f, local): TM/KB/gates (`candidate`), then the dimer gate for Z ≤ LEDGER.dimer_z_max; the first
/// passing entry gets h_req and its filter rows; none → refused with every reason.
pub fn generate_with(z: u8, mut dimer: Option<&mut dyn DimerGate>) -> Result<PseudoTable, Refusal> {
    let t0 = std::time::Instant::now();
    let reference = Arc::new(ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?);
    let part = partition(&reference)?;
    let pc = partial_core(&reference.grid, &core_density(&reference, &part), &valence_density(&reference, &part));
    let mut tried = Vec::new();
    for &f in LEDGER.f_ladder.iter() {
        for &local in LEDGER.local_order.iter() {
            let mut t = match candidate(&reference, &part, f, local, &pc) {
                Ok(t) => t,
                Err(why) => { tried.push(format!("f = {f}, {local:?}: {why}")); continue; }
            };
            if z <= LEDGER.dimer_z_max && !dimer_verdict(&mut t, dimer.as_deref_mut()) {
                tried.push(format!("f = {f}, {local:?}: dimer gate {}", t.record.dimer));
                continue;
            }
            let (hr, rows) = super::filter::h_req(&t);
            t.record.h_req = hr;
            t.record.filter = rows;
            t.record.seconds = t0.elapsed().as_secs_f64();
            return Ok(t);
        }
    }
    Err(Refusal::NoLadderEntryPasses { z, tried })
}

/// Fills `record.dimer`; false ⇔ the gate ran and failed (the ladder moves on). Without a gate the
/// table is provisional (plan review M6): reported, never usable — `IonicSetup::build` refuses it,
/// so an ungated H can never stand in for the gated one under the same recipe_hash.
fn dimer_verdict(t: &mut PseudoTable, gate: Option<&mut dyn DimerGate>) -> bool {
    let Some(g) = gate else {
        t.provisional = true;
        t.record.dimer = "provisional (dimer gate not run)".into();
        return true;
    };
    match g.dimers(t) {
        Ok(p) => {
            let ok = (p.pp.0 - p.ae.0).abs() < LEDGER.dimer_re && (p.pp.1 / p.ae.1 - 1.0).abs() < LEDGER.dimer_omega_rel;
            t.record.dimer = format!("{}: PP R_e {:.4} Bohr, ω {:.6} a.u. vs AE σ→0 {:.4} / {:.6}", if ok { "pass" } else { "fail" }, p.pp.0, p.pp.1, p.ae.0, p.ae.1);
            ok
        }
        Err(why) => { t.record.dimer = format!("not applicable: {why}"); true }
    }
}
```
`candidate` fills the record (`record_of`) and sets `provisional: false` before `dimer_verdict` runs, and h_req is computed only for the accepted entry, after the dimer gate (informational for a provisional table). This replaces `generate`'s body from Tasks 3–5. `PseudoTable` gains `pub provisional: bool`; `filter::build` copies it into `FilteredTable.provisional`. In `ledger.rs` set `version: "m7-tm-kb-nlcc-3"` (M6(b)).

In `crates/qm/src/ionic.rs`, `IonicError` gains a variant and `check` refuses a provisional table first, before the h_req test:
```rust
    #[error("{}'s pseudopotential is provisional: the dimer gate on f (spec §4.4) has not run — tables for Z ≤ {} come from the providers' cache, which runs it", units::elements::symbol(*z), LEDGER.dimer_z_max)]
    Provisional { z: u8 },
```
```rust
fn check(z: u8, t: &FilteredTable, h: f64) -> Result<(), IonicError> {
    if t.provisional { return Err(IonicError::Provisional { z }); }
    // … the h and h_req/filtered-gate checks of Task 6, unchanged …
}
```
`IonicSetup::from_tables` stays unchecked (tests only).

`records_table` gains the last column "dimer" (`record.dimer`), so the header row becomes `| Z | el | f | local | r_c (Bohr) | Z_v | E_KB (Ha) | GSS (Ha) | r_nlc | r_core | h_req | dimer |`, every accepted row ends with `| {} |` filled by `g.dimer`, and a refusal row gets one more empty cell.

- [ ] **Step 4: Implement the grid dimer gate** (`crates/providers/src/pseudo.rs`)
```rust
#[derive(Clone)]
pub enum DimerBackend { Cpu, Gpu(gpu::GpuContext) }

static BACKEND: Mutex<Option<DimerBackend>> = Mutex::new(None);
pub fn set_dimer_backend(b: DimerBackend) { *BACKEND.lock().unwrap_or_else(|e| e.into_inner()) = Some(b); }
fn dimer_backend() -> DimerBackend { BACKEND.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or(DimerBackend::Cpu) }

pub struct GridDimerGate { pub backend: DimerBackend, ae_cache: HashMap<u8, Result<(f64, f64), String>> }
impl GridDimerGate { pub fn new(backend: DimerBackend) -> GridDimerGate { GridDimerGate { backend, ae_cache: HashMap::new() } } }

fn finest_grid() -> (qm::Grid, f64) {
    let h = *LEDGER.h_ladder.last().unwrap();
    (qm::Grid::cube(2 * (6.0 / h).round() as usize + 1, h), h)
}

/// The bond force on the nucleus at +R/2 (positive = repulsive), at smearing σ for all-electron systems.
fn bond_force(backend: &DimerBackend, grid: &qm::Grid, sys: &qm::System, sigma: f64) -> Result<f64, String> {
    let mut s = qm::ScfSettings::for_grid(grid);
    s.sigma = sigma;
    let f = match backend {
        DimerBackend::Cpu => { let (st, _) = qm::scf(grid, sys, &s, None).map_err(|e| e.to_string())?; qm::forces(grid, &st, &sys.nuclei, sigma).map_err(|e| e.to_string())? }
        DimerBackend::Gpu(ctx) => {
            let mut solver = qm_gpu::GpuSolver::new(ctx.clone(), grid);
            let (st, _) = solver.solve(sys, &s, false).map_err(|e| e.to_string())?;
            solver.forces(st, &sys.nuclei, sigma).map_err(|e| e.to_string())?
        }
    };
    Ok(f[1][0])
}

fn dimer_nuclei(z: u8, r: f64) -> Vec<qm::Nucleus> { vec![qm::Nucleus { pos: [-0.5 * r, 0.0, 0.0], z }, qm::Nucleus { pos: [0.5 * r, 0.0, 0.0], z }] }

/// R_e by bisection of F(R) in [lo, hi] to LEDGER.dimer_re/10, then k = −dF/dR (central, step h)
/// and ω = √(k/μ), μ = m/2 (the most abundant isotope's mass).
fn locate(z: u8, lo: f64, hi: f64, h: f64, force: &mut dyn FnMut(f64) -> Result<f64, String>) -> Result<(f64, f64), String> {
    let (mut a, mut b) = (lo, hi);
    let (fa, fb) = (force(a)?, force(b)?);
    if !(fa > 0.0 && fb < 0.0) { return Err(format!("no bound minimum in [{lo:.3}, {hi:.3}] Bohr (F = {fa:+.2e}, {fb:+.2e})")); }
    while b - a > LEDGER.dimer_re / 10.0 {
        let m = 0.5 * (a + b);
        if force(m)? > 0.0 { a = m } else { b = m }
    }
    let r_e = 0.5 * (a + b);
    let k = -(force(r_e + h)? - force(r_e - h)?) / (2.0 * h);
    let mu = 0.5 * units::amu_to_au_mass(units::isotope_mass_amu(z).expect("listed"));
    Ok((r_e, (k / mu).sqrt()))
}

fn bracket(t: &PseudoTable) -> (f64, f64) {
    let r_peak = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.r_peak).fold(0.0, f64::max);
    (LEDGER.dimer_bracket[0] * r_peak, LEDGER.dimer_bracket[1] * r_peak)
}

pub fn pseudo_dimer(t: &PseudoTable, backend: &DimerBackend) -> Result<(f64, f64), String> {
    let (grid, h) = finest_grid();
    let ft = Arc::new(filtered(t, h));
    let setup = Arc::new(IonicSetup::from_tables(QmPolicy::All, h, h, std::collections::BTreeMap::from([(t.z, qm::ionic::Ionic::Pseudo(ft))])));
    let (lo, hi) = bracket(t);
    locate(t.z, lo, hi, h, &mut |r| bond_force(backend, &grid, &qm::System::with_ionic(dimer_nuclei(t.z, r), 0, None, setup.clone()), h))
}

/// The pseudised dimer's bond force at R (finest ladder h), for the not-applicable record.
fn pseudo_bond_force(t: &PseudoTable, backend: &DimerBackend, r: f64) -> Result<f64, String> {
    let (grid, h) = finest_grid();
    let ft = Arc::new(filtered(t, h));
    let setup = Arc::new(IonicSetup::from_tables(QmPolicy::All, h, h, std::collections::BTreeMap::from([(t.z, qm::ionic::Ionic::Pseudo(ft))])));
    bond_force(backend, &grid, &qm::System::with_ionic(dimer_nuclei(t.z, r), 0, None, setup), h)
}

impl qm::radial::recipe::DimerGate for GridDimerGate {
    fn dimers(&mut self, t: &PseudoTable) -> Result<qm::radial::recipe::DimerPair, String> {
        let backend = self.backend.clone();
        let ae = self.ae_cache.entry(t.z).or_insert_with(|| {
            let (grid, h) = finest_grid();
            let sigmas = [1.0, 1.2, 1.4, 1.6].map(|f| f * h);      // the oracle's σ ladder (validate::sigma_ladder)
            let (lo, hi) = bracket(t);
            locate(t.z, lo, hi, h, &mut |r| {
                let sys = qm::System::new(dimer_nuclei(t.z, r), 0, None);
                let fs: Vec<f64> = sigmas.iter().map(|&s| bond_force(&backend, &grid, &sys, s)).collect::<Result<_, _>>()?;
                Ok(qm::extrapolate_sigma(&sigmas, &[fs[0], fs[1], fs[2], fs[3]]))
            })
        }).clone();
        let ae = match ae {
            Ok(ae) => ae,
            // not applicable: record both bond forces at both bracket ends (plan review Q9(ii), report only;
            // the AE ones are in `why`, from `locate`)
            Err(why) => {
                let (lo, hi) = bracket(t);
                let pp = |r: f64| pseudo_bond_force(t, &self.backend, r).map_or_else(|e| format!("error: {e}"), |f| format!("{f:+.3e}"));
                return Err(format!("AE F at the bracket ends: {why}; PP F({lo:.3}) {}, PP F({hi:.3}) {} Ha/Bohr", pp(lo), pp(hi)));
            }
        };
        Ok(qm::radial::recipe::DimerPair { pp: pseudo_dimer(t, &self.backend)?, ae })
    }
}
```
`PseudoCache::table` calls `qm::radial::generate_with(z, Some(&mut GridDimerGate::new(dimer_backend())))`. The gate only runs for Z ≤ `LEDGER.dimer_z_max`: the ladder decides that, not the cache. In `crates/app/src/main.rs`, after the GPU context exists: `providers::pseudo::set_dimer_backend(DimerBackend::Gpu(ctx.clone()))`. Without a GPU the default `Cpu` applies, and the status line says the first `All` use generates H for minutes. `common::Locked::new` calls `set_dimer_backend(Gpu(ctx))` once.

- [ ] **Step 5: Run the tests and re-run Task 10's oracles**

Run: `cargo test -p providers --release dimer_gate`, `cargo test -p validate --release --test pseudo_gen` (the version bump and the H/He rows move the block: regenerate and pin it after review), and `cargo test -p validate --release --test pseudo_oracles`, which includes Task 10's geometry tests, now with the dimer-gated H (Task 10's numbers were made under the old hash; their re-run here is what references.md keeps).
Expected: PASS. Report any miss with numbers (rule 6).

Run the regression suites: `cargo test -p validate --release --test dynamics`, `--test surrogate`, `--test tools`, and `cargo test -p app --release`.
Expected: PASS, unchanged.

- [ ] **Step 6: The default policy decision (spec §4.7: "the intended default once its gates pass")**
  - **Every gate of this task and Task 10 passes under `All`:** set `ui::DEFAULT_IONIC = sim::IonicPolicy::All`. Update the app's start-up text and README ("presets start with every species pseudised; Physics ▸ Ionic potential switches") and CLAUDE.md's default line.
  - **Any gate failed:** `DEFAULT_IONIC` stays `CoresOnly`, and the failing gate and its numbers are recorded.

  Either way, add the section "M7 default policy" to `references.md` listing each gate with its number and the decision.

- [ ] **Step 7: Commit**
```bash
git add crates/qm/src/radial crates/qm/src/ionic.rs crates/providers/src/pseudo.rs crates/app crates/validate README.md CLAUDE.md
git commit -o -m "feat(qm,providers,validate): IonicPolicy::All — dimer gate on f (ungated H/He provisional, recipe v3), pseudo-H2, H2O egg-box and preset gates, default policy (M7 Task 11)" -- crates/qm/src/radial crates/qm/src/ionic.rs crates/providers/src/pseudo.rs crates/app crates/validate README.md CLAUDE.md
```
Before merging, hand the diff to `physics-reviewer` with Spec question 9, the dimer numbers, the M6 provisional/version fix, and the decision.


**NLCC ruling (2026-10-16), Ruling 6.**
- Pseudo-H has no core. H₂ under All gets the hard §6.6 gate (single-H floor 4.1e-7, so pass expected).
- H₂O under All falls under the partial-core rule. Keep `h2o_all_max_shift_sum_of_forces_below_3e_5` only if it passes; otherwise `h2o_all_eggbox_recorded` with the ratchet.
- The default-policy decision does not depend on the egg-box outcome.

---

### Task 12: Z = 11–18 enablement **[Opus]**

Spec §4.2 (Na/Mg semicore → refused; Al–Ar n = 2 core, the 3d unbound in the neutral LSDA atom → dropped, s + p channels with p local; the S5 cap binds only for Be's 2p), §5.5 (`units` bullet: the mass is the only datum added), §6.2, §8 row 12.

**Files:**
- Modify: `crates/units/src/isotopes.rs` (`TABLE` grows to Z = 18 with AME2020 rows; the module doc lists the new isotopes and the most abundant ones), `crates/render/src/elements.rs` (display colours for Z = 11–18), `crates/app/src/ui/ptable.rs` (`Z_ENABLED = 18`; tooltips), `crates/app/src/scene.rs` (new presets), `crates/app/presets/*.scene` (new files), `crates/sim/src/scene.rs` (`Nucleus::new` doc: "Panics for Z outside the table (1..=18)"), `crates/validate/tests/pseudo_gen.rs`, `crates/validate/references.md`, `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `qm::radial::generate`, the records table (Task 4), the policy plumbing (Task 9), `units::isotopes` (unchanged API).
- Produces: `units::isotopes_of(z)`, `most_abundant_a(z)` and `isotope_mass_amu_a(z, a)` answering for Z ≤ 18; `ptable::enabled(z)` for Z ≤ 18; presets for the accepted elements.

- [ ] **Step 1: Look up and pin the masses (do not type them from memory).**
  1. From AME2020 `mass_1.mas20` (Wang et al., *Chinese Phys. C* 45, 030003 (2021), the atomic-mass column, as the existing table), transcribe the stable isotopes of Z = 11–18: ²³Na; ²⁴Mg ²⁵Mg ²⁶Mg; ²⁷Al; ²⁸Si ²⁹Si ³⁰Si; ³¹P; ³²S ³³S ³⁴S ³⁶S; ³⁵Cl ³⁷Cl; ³⁶Ar ³⁸Ar ⁴⁰Ar. Take every printed digit, in μu → u, exactly as the existing rows.
  2. The most abundant ones (the default isotope label) are ²³Na, ²⁴Mg, ²⁷Al, ²⁸Si, ³¹P, ³²S, ³⁵Cl and ⁴⁰Ar.
  3. A second transcription pins each row in the `units` tests at 1e-9 u, as the existing rows are.
  4. Cite the source in the module doc and in `references.md`.

- [ ] **Step 2: Write the failing tests**

`crates/units/src/isotopes.rs` tests:
```rust
/// Spec §8 row 12 first failing test: the mass table covers Z ≤ 18 — every Z has at least one
/// isotope, a most-abundant default, and a mass within 0.5 u of A (atomic masses: binding < 0.1 u
/// per nucleon-mass unit at these A) — the only datum added (rule 1).
#[test]
fn isotope_mass_table_covers_z_up_to_18() {
    for z in 1u8..=18 {
        let isos = isotopes_of(z);
        assert!(!isos.is_empty(), "Z = {z} has no isotopes");
        let a = most_abundant_a(z).unwrap_or_else(|| panic!("Z = {z} has no default isotope"));
        assert!(isos.contains(&a));
        for &a in isos {
            let m = isotope_mass_amu_a(z, a).unwrap();
            assert!((m - a as f64).abs() < 0.5, "Z = {z}, A = {a}: {m}");
        }
    }
    assert!(most_abundant_a(19).is_none(), "Z > 18 is not enabled");
}
```
`crates/validate/tests/pseudo_gen.rs`:
```rust
/// Spec §8 row 12: Al–Ar pass every gate (n = 2 core; the 3d unbound and dropped, spec §4.2 item 3) or are
/// refused with a reason; Na and Mg are refused with the two-s-state reason naming ONCV (spec §4.2).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn al_through_ar_pass_gates_or_are_refused_with_reason() {
    for z in [11u8, 12] {
        let e = qm::radial::generate(z).expect_err("Na and Mg carry a semicore n = 2 shell");
        assert!(e.to_string().contains("two valence s states") && e.to_string().contains("ONCV"), "Z = {z}: {e}");
    }
    for z in 13u8..=18 {
        match qm::radial::generate(z) {
            Ok(t) => {
                assert_eq!(t.n_core, 10, "Z = {z}: the n = 2 shell is the core");
                assert_eq!(t.partition.dropped, vec![2], "Z = {z}: the 3d is unbound in the neutral LSDA atom (spec §4.2 item 3)");
                assert!(t.partition.channels.iter().all(|c| c.occupied), "Z = {z}: no unoccupied channel");
                // reported, not gated (T5 excludes an ungenerated l; rule 7): the d-wave scattering of the local
                // potential against the AE atom, δE_equiv on T5's window and mesh at r_d (`delta_e_equiv` with no
                // l = 2 projector is the local potential's); pinned in references.md's generation records as
                // "d scattering (ungated)"
                let gates = &qm::radial::LEDGER.gates;
                let e_min = t.partition.channels.iter().map(|c| c.eps).fold(f64::INFINITY, f64::min) - gates.t5_window;
                let e_max = t.partition.channels.iter().map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max) + gates.t5_window;
                let steps = ((e_max - e_min) / gates.t5_mesh).round() as usize;
                let d_worst = (0..=steps).map(|k| qm::radial::gates::delta_e_equiv(&t, 2, e_min + k as f64 * gates.t5_mesh).abs()).fold(0.0, f64::max);
                eprintln!("Z = {z}: d scattering (ungated): max |δE_equiv| {:.2} mHa", d_worst * 1e3);
                eprintln!("Z = {z}: f {}, {}, h_req {:?}", t.f, t.record.local, t.record.h_req);
            }
            Err(e) => {
                let m = e.to_string();
                assert!(!m.is_empty() && (m.contains("ladder") || m.contains("valence")), "Z = {z}: a refusal must name its reason: {m}");
                eprintln!("Z = {z} refused: {m}");
            }
        }
    }
}
```
`ZS` grows to `1..=18`. The pinned records block is regenerated (Step 5).

`crates/app/src/ui/ptable.rs` tests: `tip(11)` names the refusal and what the fallback costs ("refused: two valence s states — needs the ONCV stage; under Cores only it falls back to all-electron at σ = h, where Zσ ≈ 1.7 even at h = 0.15: qualitatively wrong, as O was before M7" — plan review NOTE, rule 7), `tip(12)` the same for Mg (Zσ ≈ 1.8), `tip(16)` reads "pseudopotential (generated on first use)", `enabled(18)` is true and `enabled(19)` is false.

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test -p units --release isotope_mass_table_covers_z_up_to_18`
Expected: FAIL at Z = 11.

Run: `cargo test -p validate --release --test pseudo_gen al_through_ar`
Expected: runs. It passes or reports per element, because the generator is Task 4's code; this task wires the elements through the app.

- [ ] **Step 4: Implement**
- **`units`:** add the Step 1 rows to `TABLE` (index 11–18) and grow the array to `[&[Row]; 19]`; set the most-abundant defaults; update the module doc's isotope list.
- **`render::elements::colour`:** Jmol CPK display colours — Na `[0.67, 0.36, 0.95]`, Mg `[0.54, 1.0, 0.0]`, Al `[0.75, 0.65, 0.65]`, Si `[0.94, 0.78, 0.63]`, P `[1.0, 0.50, 0.0]`, S `[1.0, 1.0, 0.19]`, Cl `[0.12, 0.94, 0.12]`, Ar `[0.50, 0.82, 0.89]` (display only, rule 1).
- **`ptable.rs`:** `Z_ENABLED = 18`. `tip(z)` for 11..=18 reads from `qm::radial::has_core`/`generate` results cached in the providers' `PseudoCache` when available, else "pseudopotential (generated on first use)". A refused species' tip states the all-electron fallback's Zσ at h = 0.15 (computed as z·0.15, not tabulated) and that it is qualitatively wrong there. The panel note changes to "Z 1–18 can be placed; heavier elements need the next pseudopotential stage (ONCV, semicore)". README: Na/Mg place, but under Cores only they run all-electron at Zσ ≈ 1.7–1.8 (h = 0.15), the regime spec §1 shows is wrong; under All they are refused.
- **Presets:** for each accepted element among S, Cl, P and Si, add one molecule preset at its experimental geometry: H₂S, HCl, PH₃ and SiH₄. Look up r and ∠ in NIST CCCBDB "Experimental geometry" at task time and cite them in the preset file's header comment, as `h2o.scene` does.
  - Use a 14 Bohr box at the preset's default grid of 57 × 0.25 (the app's Accurate default then applies at start-up).
  - Add each to `Scene::PRESETS` and `Scene::preset`, and to `--preset`.
  - A refused element gets no preset; the README says why.

- [ ] **Step 5: Run, regenerate, record**

Run: `cargo test -p units --release`, `cargo test -p validate --release --test pseudo_gen` (pin the regenerated block after review) and `cargo test -p app --release`.
Expected: PASS.

`references.md` "M7 generation records" now covers Z = 1–18. Add a paragraph on Al's Δ_core margin (review NOTE: 0.27 Ha — "universal, but close"), with the measured ε_2p and ε_3s.

- [ ] **Step 6: Docs and commit.** README: the element range (Z ≤ 18), refused Na/Mg and why, the new presets. CLAUDE.md: the preset list, `--preset` values and the table range.
```bash
git add crates/units crates/render/src/elements.rs crates/app crates/sim/src/scene.rs crates/validate README.md CLAUDE.md
git commit -o -m "feat(units,app,validate): enable Z = 11–18 — AME2020 masses, presets, periodic table; Na/Mg refused until ONCV (M7 Task 12)" -- crates/units crates/render/src/elements.rs crates/app crates/sim/src/scene.rs crates/validate README.md CLAUDE.md
```

---

### Task 13: Bond-angle overlay **[Opus]**

Spec §8 row 13 (user request 2026-10-15: "with M7, you should display the bond angles on the bond chains that are drawn between atoms"). The overlay is display only: pure geometry from the nuclear positions, with no physics input. **Spec question 10 applies, now resolved:** the M6 bond-strength kind exists as `sim::BondKind { Shared, ClosedShell, Indeterminate }` with `Bond::kind() -> BondKind` (and `PairReport::kind() -> Option<BondKind>`), from the sign of ∇²ρ_b against its noise band (`crates/sim/src/bonds.rs` on the M6 branch, merged in Task 6 Step 0). M6 draws a shared bond solid, a closed-shell contact dashed and an indeterminate one dotted (`overlay::pattern`). The angle styling uses the kinds of both bonds directly (plan review Q10); there is no stand-in. The plan review's physics caveat is applied too: under pseudopotentials ∇²ρ_b is evaluated on the valence (+ρ_pc) density, so a bond point inside an r_c sphere gets a kind that is not the all-electron one, and it is labelled "valence-density kind".

**Files:**
- Modify: `crates/app/src/overlay.rs` (`AngleStyle`, `AngleMark`, `bond_angles`, `shown_angles`, `angle_style`, `angle_arc_lines`, `angle_label`), `crates/app/src/viewer.rs` (arcs in `overlay_lines`, labels in `label_overlay`), `crates/app/src/ui/{mod.rs,actions.rs,menus.rs,search.rs}` (`UiState.show_angles`, `Action::ShowAngles`, the View toggle, the search entry, `valence_density_kind`), `crates/app/src/ui/dock/selection.rs` (the angle list with the surface note; the valence-density clause on bond kinds), `crates/app/src/screenshot.rs` (`--ui` key `angles=on|off`), `crates/app/replays/h2o_angle_all.replay` (new), `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md`, `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md`, `README.md`

**Interfaces:**
- Consumes: `sim::BondGraph { bonds, n_nuclei, .. }`, `sim::Bond { i, j, s, multi_centre, laplacian_bcp, laplacian_noise, .. }`, `sim::Bond::kind`, `sim::BondKind`, `sim::PairReport::{kind, s}`, `render::lines::{segment, LineVertex}`, `overlay::{BOND, RING_BOND, BOND_ALPHA, STALE_BOND_ALPHA, MIN_ALPHA, pattern}` (M6), `tools::project`, `EvalStats.pseudised` (`PseudoSummary { z, r_c_max, .. }`, Task 9), `UiState`, `Action`.
- Produces:
```rust
// crates/app/src/overlay.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngleStyle { Shared, Contact, Indeterminate, Ring }
#[derive(Clone, Debug, PartialEq)]
pub struct AngleMark { pub vertex: usize, pub a: usize, pub b: usize, pub radians: f64, pub style: AngleStyle }
pub const ANGLE_ARC_RADIUS: f32 = 0.45;             // Bohr (display)
pub const ANGLE_ARC_SEGMENTS: usize = 16;            // display
pub const ANGLE_DECLUTTER_BONDS: usize = 3;          // more bonds than this at a vertex: labelled only when selected/hovered
pub fn bond_angles(pos: &[[f64; 3]], graph: &sim::BondGraph) -> Vec<AngleMark>;
pub fn shown_angles<'a>(all: &'a [AngleMark], graph: &sim::BondGraph, selected: Option<usize>, hovered: Option<usize>) -> Vec<&'a AngleMark>;
pub fn angle_style(a: &sim::Bond, b: &sim::Bond) -> AngleStyle;
pub fn angle_arc_lines(pos: &[[f64; 3]], m: &AngleMark, stale: bool) -> Vec<LineVertex>;
pub fn angle_label(m: &AngleMark) -> String;                        // "104.4°"
pub fn angle_label_anchor(pos: &[[f64; 3]], m: &AngleMark) -> [f64; 3];
// crates/app/src/ui
pub fn angle_surface_note(pseudised: &[u8], zs: [u8; 3]) -> &'static str;
pub fn valence_density_kind(pos: &[[f64; 3]], zs: &[u8], i: usize, j: usize, s: f64, pseudised: &[sim::PseudoSummary]) -> bool;
pub const VALENCE_KIND_NOTE: &str;
pub enum Action { /* … */ ShowAngles(bool) }                        // script word `angles true|false`
pub struct UiState { /* … */ pub show_angles: bool }                // default true
```
**Rules** (the spec row's declutter, made exact):
- An angle exists for every pair of drawn bonds meeting at a vertex.
- A vertex with ≤ `ANGLE_DECLUTTER_BONDS` (3) bonds shows all its angles: water, NH₃ and every "small molecule".
- A vertex where more than 3 bonds meet (CH₄'s C) shows its angles only when that atom is selected or hovered.
- Angles are drawn between the positions drawn now and faded with the bonds when the graph is stale (§3.4 of the M6 spec, as `bond_lines`).
- The style follows the two bonds (plan review Q10), first match wins:
  - either bond multi-centre → `Ring`: dashed in the ring colour, as today;
  - either bond `ClosedShell` → `Contact`: the contact's dash, faint;
  - either bond `Indeterminate` → `Indeterminate`: its own neutral dotted style, never drawn as shared (rule 7: the sign is not resolved);
  - both `Shared` → `Shared`: solid.
  `Bond::kind()` is never absent: a Laplacian that could not be evaluated is NaN, hence `Indeterminate`.
- The angle is computed as atan2(|u×w|, u·w), well conditioned near 0° and 180°.
- **The valence-density caveat** (plan review Q10). The bond point of i–j is pos_i + s·(pos_j − pos_i). When it lies within `r_c_max` of a pseudised end atom, the kind is a "valence-density kind": the pseudo-density there is smooth by construction, so the QTAIM sign is not the all-electron one (for O–H the bond point is ≈ 1.4 Bohr from O, inside pseudo-O's r_c of 1.4–1.9). The Selection tab's kind line and the angle toggle's tooltip say so.

- [ ] **Step 1: Write the failing tests** (`crates/app/src/overlay.rs` tests)
```rust
/// A shared-shell bond (∇²ρ_b = −1 against a 0.1 band) unless `laplacian` says otherwise.
fn bond(i: usize, j: usize, ring: bool, laplacian: f64) -> sim::Bond {
    sim::Bond { i, j, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: sim::bonds::Route::Saddle, multi_centre: ring, laplacian_bcp: laplacian, laplacian_noise: 0.1 }
}

fn graph(n: usize, pairs: &[(usize, usize)], ring: bool) -> sim::BondGraph {
    sim::BondGraph {
        bonds: pairs.iter().map(|&(i, j)| bond(i, j, ring, -1.0)).collect(),
        components: vec![(0..n).collect()], step: 1, sim_time: 0.0, n_nuclei: n,
    }
}

/// Spec §8 row 13 first failing test: the water preset's angle from its nuclear positions, to 1e-9 rad.
#[test]
fn angles_of_water_preset_match_geometry() {
    let s = crate::scene::Scene::preset("h2o").unwrap();
    let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
    let marks = bond_angles(&pos, &graph(3, &[(0, 1), (0, 2)], false));
    assert_eq!(marks.len(), 1);
    let m = &marks[0];
    assert_eq!((m.vertex, m.a, m.b), (0, 1, 2));
    // the preset: O at the origin, H at (±1.43, 1.11, 0) — the half-angle is atan2(1.43, 1.11)
    let expected = 2.0 * 1.43f64.atan2(1.11);
    assert!((m.radians - expected).abs() < 1e-9, "{} vs {expected}", m.radians);
    assert_eq!(angle_label(m), format!("{:.1}°", expected.to_degrees()));
    assert_eq!(shown_angles(&marks, &graph(3, &[(0, 1), (0, 2)], false), None, None).len(), 1, "two bonds: always shown");
}

/// Spec §8 row 13: where four bonds meet (CH₄), only the selected (or hovered) atom's angles show.
#[test]
fn four_bonds_only_selected_atom_labelled() {
    let a = 2.06 / 3f64.sqrt();
    let pos = vec![[0.0; 3], [a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]];
    let g = graph(5, &[(0, 1), (0, 2), (0, 3), (0, 4)], false);
    let marks = bond_angles(&pos, &g);
    assert_eq!(marks.len(), 6);
    let tetra = (-1.0f64 / 3.0).acos();
    assert!(marks.iter().all(|m| (m.radians - tetra).abs() < 1e-9));
    assert!(shown_angles(&marks, &g, None, None).is_empty(), "four bonds, nothing selected: no labels");
    assert_eq!(shown_angles(&marks, &g, Some(0), None).len(), 6);
    assert_eq!(shown_angles(&marks, &g, None, Some(0)).len(), 6);
    assert!(shown_angles(&marks, &g, Some(1), None).is_empty(), "an H has one bond: no angle at it");
}

#[test]
fn three_bonds_show_without_selection_and_ring_angles_are_dashed() {
    let pos = vec![[0.0; 3], [1.9, 0.0, -0.6], [-0.95, 1.65, -0.6], [-0.95, -1.65, -0.6]];
    let g = graph(4, &[(0, 1), (0, 2), (0, 3)], false);
    let marks = bond_angles(&pos, &g);
    assert_eq!(shown_angles(&marks, &g, None, None).len(), 3);
    assert!(marks.iter().all(|m| m.style == AngleStyle::Shared));
    let ring = graph(3, &[(0, 1), (1, 2), (0, 2)], true);
    let h3 = vec![[0.0, 0.0, 0.0], [1.65, 0.0, 0.0], [0.825, 1.429, 0.0]];
    assert!(bond_angles(&h3, &ring).iter().all(|m| m.style == AngleStyle::Ring));
}

/// Plan review Q10: the angle takes the kinds of both bonds — a closed-shell contact makes it the
/// contact's faint dash, an indeterminate one its own dotted style (never drawn as shared), a ring
/// bond wins over both.
#[test]
fn angle_style_follows_the_bond_kinds() {
    let (shared, contact, indet) = (bond(0, 1, false, -1.0), bond(0, 2, false, 0.5), bond(0, 3, false, 0.05));
    assert_eq!(angle_style(&shared, &shared), AngleStyle::Shared);
    assert_eq!(angle_style(&shared, &contact), AngleStyle::Contact);
    assert_eq!(angle_style(&contact, &indet), AngleStyle::Contact);
    assert_eq!(angle_style(&indet, &shared), AngleStyle::Indeterminate);
    assert_eq!(angle_style(&bond(0, 4, false, f64::NAN), &shared), AngleStyle::Indeterminate, "a Laplacian that could not be evaluated");
    assert_eq!(angle_style(&bond(0, 1, true, 0.5), &shared), AngleStyle::Ring);
    // the patterns are M6's `pattern` along the arc (length 0.45·θ ≈ 0.83 Bohr here): solid draws all
    // ANGLE_ARC_SEGMENTS chords; the dash (0.22 on, 0.14 off) leaves out the gaps; the dots (0.05 on,
    // 0.12 off) leave out more
    let pos = vec![[0.0; 3], [1.8, 0.0, 0.0], [-0.5, 1.7, 0.0]];
    let arc = |style| angle_arc_lines(&pos, &AngleMark { vertex: 0, a: 1, b: 2, radians: angle_between(pos[1], pos[2]), style }, false).len();
    let (solid, dashed, dotted) = (arc(AngleStyle::Shared), arc(AngleStyle::Contact), arc(AngleStyle::Indeterminate));
    assert!(dotted < dashed && dashed < solid, "solid {solid}, dashed {dashed}, dotted {dotted} vertices");
}

/// Nothing when the graph belongs to another nucleus count (as `bond_lines`).
#[test]
fn a_graph_of_another_scene_draws_no_angles() {
    assert!(bond_angles(&[[0.0; 3], [1.0, 0.0, 0.0]], &graph(3, &[(0, 1), (0, 2)], false)).is_empty());
}
```
`crates/app/src/ui/tests.rs`:
```rust
#[test]
fn show_angles_action_round_trips_and_defaults_on() {
    let st = UiState::default();
    assert!(st.show_angles, "bond angles are on by default (spec §8 row 13)");
    for b in [true, false] { assert_eq!(Action::parse(&Action::ShowAngles(b).describe()), Ok(Action::ShowAngles(b))); }
}

#[test]
fn angle_note_names_the_surface() {
    assert_eq!(angle_surface_note(&[8, 1], [8, 1, 1]), "pseudopotential surface");
    assert!(angle_surface_note(&[8], [8, 1, 1]).contains("smeared"));
    assert!(angle_surface_note(&[], [8, 1, 1]).contains("smeared"));
}

/// Plan review Q10: a bond point inside a pseudised atom's r_c gets a valence-density kind. O at the
/// origin with r_c,max = 1.6, H at 1.81 Bohr: the bond point at s = 0.75 (1.36 Bohr from O) is inside;
/// at s = 0.95 (1.72 Bohr) it is not, and nothing is inside when nothing is pseudised.
#[test]
fn valence_density_kind_inside_an_r_c_sphere() {
    let pos = [[0.0; 3], [1.81, 0.0, 0.0]];
    let o = sim::PseudoSummary { z: 8, z_v: 6.0, f: 1.6, local: "p-local".into(), r_c_max: 1.6, h_req: Some(0.2), r_core: 0.7, record: String::new() };
    assert!(valence_density_kind(&pos, &[8, 1], 0, 1, 0.75, std::slice::from_ref(&o)));
    assert!(!valence_density_kind(&pos, &[8, 1], 0, 1, 0.95, std::slice::from_ref(&o)));
    assert!(!valence_density_kind(&pos, &[8, 1], 0, 1, 0.75, &[]));
}
```
(Adapt the `Action::parse`/`describe` call shapes to the ones the file already uses for `ShowBonds`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p app --release overlay ui::tests::show_angles`
Expected: compile errors.

- [ ] **Step 3: Implement the geometry and the lines** (`crates/app/src/overlay.rs`)
```rust
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }

/// atan2(|u × w|, u·w): exact to rounding at every angle (acos loses digits near 0° and 180°).
fn angle_between(u: [f64; 3], w: [f64; 3]) -> f64 {
    let c = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(u[0] * w[0] + u[1] * w[1] + u[2] * w[2])
}

/// The style of the angle between two bonds (plan review Q10; M6's `sim::BondKind`): a multi-centre
/// bond → `Ring`; else a closed-shell contact → `Contact`; else an unresolved sign → `Indeterminate`
/// (never drawn as shared, rule 7); else `Shared`.
pub fn angle_style(a: &sim::Bond, b: &sim::Bond) -> AngleStyle {
    let kinds = [a.kind(), b.kind()];
    if a.multi_centre || b.multi_centre { AngleStyle::Ring }
    else if kinds.contains(&sim::BondKind::ClosedShell) { AngleStyle::Contact }
    else if kinds.contains(&sim::BondKind::Indeterminate) { AngleStyle::Indeterminate }
    else { AngleStyle::Shared }
}

pub fn bond_angles(pos: &[[f64; 3]], graph: &sim::BondGraph) -> Vec<AngleMark> {
    if graph.n_nuclei != pos.len() { return Vec::new(); }
    let mut out = Vec::new();
    for v in 0..pos.len() {
        let at: Vec<(usize, &sim::Bond)> = graph.bonds.iter().filter_map(|b| if b.i == v { Some((b.j, b)) } else if b.j == v { Some((b.i, b)) } else { None }).collect();
        for x in 0..at.len() {
            for y in x + 1..at.len() {
                let ((a, ba), (b, bb)) = (at[x], at[y]);
                let (a, b, ba, bb) = if a < b { (a, b, ba, bb) } else { (b, a, bb, ba) };
                out.push(AngleMark { vertex: v, a, b, radians: angle_between(sub(pos[a], pos[v]), sub(pos[b], pos[v])), style: angle_style(ba, bb) });
            }
        }
    }
    out
}
```
```rust
pub fn shown_angles<'a>(all: &'a [AngleMark], graph: &sim::BondGraph, selected: Option<usize>, hovered: Option<usize>) -> Vec<&'a AngleMark> {
    let bonds_at = |v: usize| graph.bonds.iter().filter(|b| b.i == v || b.j == v).count();
    all.iter().filter(|m| bonds_at(m.vertex) <= ANGLE_DECLUTTER_BONDS || selected == Some(m.vertex) || hovered == Some(m.vertex)).collect()
}

pub fn angle_label(m: &AngleMark) -> String { format!("{:.1}°", m.radians.to_degrees()) }

/// An arc of radius ANGLE_ARC_RADIUS around the vertex from the direction of `a` to that of `b`,
/// faded with stale bonds. The pattern is M6's bond pattern measured along the arc (`pattern`:
/// Shared solid, Contact dashed and faint as M6's weakest contact, Indeterminate dotted); a Ring
/// angle is dashed in the ring colour. Each drawn piece is split into chords at the
/// ANGLE_ARC_SEGMENTS-per-arc resolution. Parametrised as u·cos(tθ) + n̂·sin(tθ), n̂ the unit vector
/// in the (u, w) plane perpendicular to u (well defined below 180°; at 180° any perpendicular from
/// `plane_basis`).
pub fn angle_arc_lines(pos: &[[f64; 3]], m: &AngleMark, stale: bool) -> Vec<LineVertex> {
    let mut out = Vec::new();
    let Some((v, u, n)) = arc_frame(pos, m) else { return out };
    let base = if stale { STALE_BOND_ALPHA } else { BOND_ALPHA };
    let (rgb, alpha, dash) = match m.style {
        AngleStyle::Shared => (BOND, base, pattern(sim::BondKind::Shared)),
        AngleStyle::Contact => (BOND, base * MIN_ALPHA, pattern(sim::BondKind::ClosedShell)),
        AngleStyle::Indeterminate => (BOND, base, pattern(sim::BondKind::Indeterminate)),
        AngleStyle::Ring => (RING_BOND, base, pattern(sim::BondKind::ClosedShell)),
    };
    let colour = [rgb[0], rgb[1], rgb[2], alpha];
    let theta = m.radians as f32;
    let len = ANGLE_ARC_RADIUS * theta;
    let point = |s: f32| { let t = s / len * theta; v + ANGLE_ARC_RADIUS * (u * t.cos() + n * t.sin()) };
    let pieces: Vec<(f32, f32)> = match dash {
        None => vec![(0.0, len)],
        Some((on, gap)) => {
            let (mut p, mut s) = (Vec::new(), 0.0f32);
            while s < len { let e = (s + on).min(len); p.push((s, e)); s = e + gap; }
            p
        }
    };
    for (s0, s1) in pieces {
        let k = ((s1 - s0) / len * ANGLE_ARC_SEGMENTS as f32).ceil().max(1.0) as usize;
        for c in 0..k {
            let (a, b) = (s0 + (s1 - s0) * c as f32 / k as f32, s0 + (s1 - s0) * (c + 1) as f32 / k as f32);
            lines::segment(&mut out, point(a).to_array(), point(b).to_array(), colour);
        }
    }
    out
}

/// (vertex, û, n̂) of an angle's arc; None for a degenerate angle (coincident nuclei, θ ≈ 0).
fn arc_frame(pos: &[[f64; 3]], m: &AngleMark) -> Option<(Vec3, Vec3, Vec3)> {
    let v = Vec3::from_array(pos[m.vertex].map(|x| x as f32));
    let u = (Vec3::from_array(pos[m.a].map(|x| x as f32)) - v).normalize_or_zero();
    let w = (Vec3::from_array(pos[m.b].map(|x| x as f32)) - v).normalize_or_zero();
    if u == Vec3::ZERO || w == Vec3::ZERO || m.radians < 1e-4 { return None; }
    let perp = w - u * u.dot(w);
    let n = if perp.length() > 1e-4 { perp.normalize() } else { plane_basis(u).0 };
    Some((v, u, n))
}

/// Where the label goes: the arc's midpoint, pushed out by a third of the radius.
pub fn angle_label_anchor(pos: &[[f64; 3]], m: &AngleMark) -> [f64; 3] {
    let Some((v, u, n)) = arc_frame(pos, m) else { return pos[m.vertex] };
    let h = 0.5 * m.radians as f32;
    (v + 1.33 * ANGLE_ARC_RADIUS * (u * h.cos() + n * h.sin())).to_array().map(|x| x as f64)
}
```

- [ ] **Step 4: Wire the viewer and the UI**
- `viewer.rs::overlay_lines`: inside the existing `show_bonds` branch, when `self.ui.show_angles`, add `angle_arc_lines` for every `shown_angles(&bond_angles(&pos, g), g, self.tool.selected, hovered)`. `hovered` is the Select tool's pick, as `badge` computes it.
- `label_overlay`: for the same marks, draw `angle_label(m)` with the existing `pill` at `tools::project(angle_label_anchor(..))`, alpha as the molecule labels (faded when stale).
- `UiState.show_angles` (default `true`). `Action::ShowAngles(b)` → `st.show_angles = b`; script word `angles`, handled where `"bonds" | "labels" | "badge"` are.
- View ▸ Bonds and molecules: checkbox "bond angles", with the tooltip "the angle between each pair of drawn bonds at an atom (geometry only, from the nuclear positions). Where more than three bonds meet, select or hover the atom to see its angles. An angle is drawn in the style of its weaker bond: dashed and faint at a closed-shell contact, dotted where the bond's kind is not resolved. Under all-electron nuclei (σ = h) the angles are the smeared surface's — water relaxes toward 177° there; with every species pseudised it relaxes to {θ}°". Here {θ} is the `All` water angle **measured in Task 10/11 and recorded in `references.md` "M7 oracles"**, quoted to 0.1° (plan review NOTE: the tooltip quotes the measurement, not an expectation written before it). Read it from `references.md` at task time; if the `All` gate failed, the sentence names the measured value and the miss.
- [`ui/mod.rs`] the valence-density caveat (plan review Q10):
```rust
/// Spec §5 and plan review Q10: under pseudopotentials ∇²ρ_b is evaluated on the valence (+ρ_pc)
/// density, which is smooth by construction inside each r_c — so a bond point there has a kind that
/// is not the all-electron one.
pub const VALENCE_KIND_NOTE: &str = "valence-density kind: this bond point lies inside a pseudopotential's core radius r_c, where the valence density is smooth by construction — the sign of ∇²ρ_b here is not the all-electron one";

/// Whether the bond point of i–j (pos_i + s·(pos_j − pos_i)) lies within r_c,max of a pseudised end atom.
pub fn valence_density_kind(pos: &[[f64; 3]], zs: &[u8], i: usize, j: usize, s: f64, pseudised: &[sim::PseudoSummary]) -> bool {
    let p: [f64; 3] = std::array::from_fn(|k| pos[i][k] + s * (pos[j][k] - pos[i][k]));
    [i, j].into_iter().any(|a| pseudised.iter().find(|ps| ps.z == zs[a])
        .is_some_and(|ps| (0..3).map(|k| (p[k] - pos[a][k]).powi(2)).sum::<f64>().sqrt() < ps.r_c_max))
}
```
- [`ui/dock/selection.rs`] in M6's bond list, where a bonded pair's hover text names its kind (`kind_words`), append "\n" + `VALENCE_KIND_NOTE` when `valence_density_kind(&pos, &zs, r.i, r.j, r.s?, &eval.pseudised)`, and suffix the line with " (valence-density kind)".
- Search entry: `e("Bond angles", "angle degrees geometry bonds HOH water", Menu(M::View), "m-bonds")`.
- Dock ▸ Selection: under the selected atom's bonds, list each angle at it as "∠H–O–H 104.4°", followed by `angle_surface_note(&pseudised, [z_a, z_v, z_b])`:
```rust
pub fn angle_surface_note(pseudised: &[u8], zs: [u8; 3]) -> &'static str {
    if zs.iter().all(|z| pseudised.contains(z)) { "pseudopotential surface" }
    else { "includes smeared all-electron nuclei (σ = h): the angle carries the smearing" }
}
```
- `screenshot.rs`: `--ui` key `angles=on|off`.

- [ ] **Step 5: The checklist screenshot (spec row 13: "screenshot of relaxed H₂O under `All` in the checklist")**

Add `crates/app/replays/h2o_angle_all.replay` with the three pins (`action quality accurate`, `action walls off 4.5`, `action ionic all`), then `action relax` and `wait idle`. Water's O carries two bonds, so its angle shows without a selection.

Run: `cargo run -p app --release -- --replay crates/app/replays/h2o_angle_all.replay --preset h2o --screenshot docs/images/m7_water_angle_all.png`
Expected: exit 0, and the PNG shows the arc and label at O.

Under `--ionic` defaults that are not `All`, the script's pin decides. Record the item, the screenshot path and the measured angle in `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md`. In the same checklist entry, record the visual oracle of spec §6.4 ("lone pairs: two lobes on the O isosurface of water at the published density threshold"): a second screenshot of the relaxed water under `All` in the isosurface view (`--ui "opacity=…"` at the threshold the design spec §8 names), marked verified or not. Add the new control's home to the UI checklist.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p app --release`
Expected: PASS, including `replays_pass` with the new script.

- [ ] **Step 7: README and commit.** README "Bonds and molecules" gains the angle overlay: what it shows, the declutter rule and the surface note.
```bash
git add crates/app README.md docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md docs/images/m7_water_*.png
git commit -o -m "feat(app): bond-angle overlay on the drawn bonds (M7 Task 13)" -- crates/app README.md docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md docs/images
```

---

## Spec coverage (self-review, 2026-10-15)

| spec | where |
|---|---|
| §4.1 radial atom: log grid, Numerov/shooting, spin modes, derived aufbau, S9 + Δ_scan scans, DIIS, scaling identity, hydrogenic check, NIST oracles | Task 1 |
| §4.2 partition (Δ_core, channels, the S5 cap, refusals, no-core branch) | Task 2 (Na/Mg/Al–Ar exercised in Task 12) |
| §4.3 TM (seven conditions, M1), V_scr inversion, unscreening with ρ_pc, r_cmp | Task 2; LFC re-run Task 4 |
| §4.4 f ladder, local-candidate order, pinned outputs table (S6a), dimer gate (S6b) | Tasks 3, 4, 11 |
| §4.5 NLCC (LFC, r_nlc, Off switch) | Task 4 |
| §4.6 KB form + normalisation (S1), GSS (G1), radial KB atom (G2) | Task 3; grid identity Task 6 |
| §4.7 policies | Tasks 6, 9, 11 |
| §4.8 T1–T7, T8 (h_req), the filtered T1/T3 re-run (S2) | Tasks 4, 5 |
| §4.9 ledger, `recipe_hash`, the QMC switch | Task 1 (table), Task 4 (hash, grep test) |
| §5.1 LR/SR split with the one V_SR table (Spec question 2), King-Smith band completion (Spec question 3), filter (S3, S7), σ_LR = 2h (spectral) | Task 5; grid sampling Task 6; GPU Task 8 |
| §5.2 Hamiltonian and energy, N = Σ Z_v | Task 6 |
| §5.3 projector application, cost | Task 6 (CPU), Task 8 (GPU) |
| §5.4 forces | Task 7 (CPU), Task 8 (GPU) |
| §5.5 crate changes, `PseudoCache`, provenance, panel, badge, valence label, drift reset, README | Tasks 6, 8, 9 |
| §5.6 core overlap (r_core, warning, tint, never trained on) | Tasks 4, 6, 7, 9 |
| §6.1 unchanged suites | Global constraint + Tasks 6–9 runs |
| §6.2 generator gates on Be–Ne | Task 4 (`pseudo_gen`) |
| §6.3 Ne 3-D vs radial, O triplet, GPU vs CPU | Tasks 6, 8, 10 |
| §6.4 H₂O/NH₃/CH₄, O Hund, lone pairs (visual, checklist), H₂ under `All` | Tasks 10, 11; the lone-pair isosurface is a checklist item in Task 13 Step 5's checklist entry |
| §6.5 valence-energy label, ρ_pc display toggle, generation record in the panel | Task 9 |
| §6.6 egg-box gates (CoresOnly all-pseudo, `All` H₂O), CoresOnly H₂O recorded | Tasks 7, 10, 11 |
| §6.7 `h2o` preset gate, NVE/CSVR on pseudo systems | Tasks 10, 11 |
| §8 rows 1–13 and their first failing tests | Tasks 1–13, names verbatim (`-` → `_` in Rust identifiers) |

The lone-pair check of §6.4 is a visual item. Task 13's checklist entry (Step 5) also records "two lobes on the O isosurface of water under `All` at the published threshold": a screenshot with the isosurface view (`--ui "opacity=…"`) next to the angle screenshot.

## Spec questions — resolved (plan review 2026-10-15)

The plan review (`docs/superpowers/notes/2026-10-15-m7-plan-review.md`, verdict "ready after amendments") ruled on each question. The spec-level consequences are in the spec's §11. One line per question: the ruling, then where it was applied.

1. **NIST LSD column is VWN — ACCEPT, amended.** The review measured the column as VWN5 (H/He to ~1e-6; PW92 misses by 4×/38×). Step 1 also records NIST's spin interpolation. The test-only VWN5 takes its v_c step as 1e-6·(n↑+n↓), and stays outside LEDGER/`recipe_hash`. Applied: Task 1 Steps 1, 3 and 9.
2. **V_cmp = V_loc + Z_v/d is singular — claim CONFIRMED; the plan's switched tail REJECTED; AMENDED to one table.** V_LR = −Z_v erf(d/a)/d is analytic. V_SR = V_loc + Z_v erf(d/a)/d is one smooth table, band-completed and truncated at γ_r·R_SR, with R_SR = max(r_cmp, r_floor) and r_floor from `v_tail_floor`. Applied: Task 1 ledger docs; Task 5 (`FilteredTable.{v_sr_table, r_sr}`, `build`, the lattice test includes V_SR); Task 6 (`ionic_local_potential`); Task 7 (F^SR from `interp_deriv`); Task 8 (`local_sr` with no tail branch); spec §5.1, §5.4, §4.9.
3. **"King-Smith's smooth cutoff" — AMEND: King-Smith, Payne & Lin's band completion.** Keep F̂ ≤ q_c. Choose F̂ on (q_c, π/h] (Chebyshev, `numerics.kspl_terms` = 24) to minimise ∫_{γ_r R}^∞|F|²r²dr. Truncate by the ψ-step over [γ_r R, γ_r R + h]. The 1e-8 judge is kept; the remedies if it misses are Ono–Hirose or reporting the factor. Applied: Task 1 (ledger `numerics.kspl_terms`); Task 5 (`filter_radial`, `band_completion_suppresses_the_first_shell`); spec §5.1.
4. **`deliberate_bad_local_is_caught_as_ghost` as worded cannot fail — claim CONFIRMED; ruling ACCEPTED with test amendments.** The narrow well uses w = 0.05 Bohr and D scanned geometrically to 1e4 Ha, and the test asserts the window was found. G2 is checked unconditionally on the reference screening (v_loc + screen) in both cases A and B. Applied: Task 3 Step 1; spec §8 row 3 and §4.6.
5. **G2 by determinant zeros on a mesh — AMEND: the exact inertia count.** n_KB(E) = n_loc(E) + [s(E) < 0] − [E_KB > 0], with s = det M/(E_KB·W(u_h,u_i)). `g2_mesh` stays (hashed) only to locate roots, and the mesh root count must equal the inertia count. Applied: Task 3 (`kb_count_below`, `ghost_gates`, `inertia_count_matches_the_roots`); spec §4.6.
6. **V_LR is the analytic erf — ACCEPT.** The Poisson-floor wording is stale. σ_LR = 2h is re-justified spectrally: e^{−(q_c a/2)²} ≈ 6e-5. Applied: Tasks 6 and 8 (as planned); the ledger doc of `sigma_lr_over_h` (Task 1); spec §4.9, §5.1, §5.2, §5.5.
7. **c_lm finished on the GPU inside H; E_NL and forces in f64 on the CPU — ACCEPT.** Applied: Task 8 (unchanged).
8. **`All` usable from Task 6, Task 11 re-runs Task 10 — ACCEPT, conditional on M6.** Applied: Task 6 header; Task 10 (`h_for` and the Ne h_req read `providers::pseudo::table_for`); Task 11 (provisional ungated H/He, `LEDGER.version` bump, the Task 10 re-run).
9. **Dimer gate as a callback, `dimer_bracket`, He not applicable — ACCEPT with notes.** (i) [1, 4]·r_peak is justified physically in the ledger doc. (ii) The not-applicable record carries the PP and AE bond forces at both bracket ends. (iii) The FD-k anharmonic bias (+1.4 % for H₂) is stated in the H₂ test's doc. Applied: Task 1 ledger doc; Task 11 (gate rules, `GridDimerGate`, the tests).
10. **The M6 bond-strength kind — AMEND: it exists.** `sim::BondKind { Shared, ClosedShell, Indeterminate }` with `Bond::kind()` comes from the M6 branch, merged before Task 6. Angles take both bonds' kinds: Ring, then Contact, then Indeterminate, else Shared; there is no stand-in. Bond points inside an r_c sphere are labelled "valence-density kind". Applied: Task 13 (`AngleStyle`, `angle_style`, `angle_arc_lines`, `valence_density_kind`, the tooltip and Selection tab); spec §8 row 13.
11. **h_req = the coarsest ladder h such that it and every finer one pass — ACCEPT.** Applied: Task 5 (unchanged); the README notes the Balanced/Fast refusals (Task 9 Step 7).
12. **Pole-free δE_equiv — defect CONFIRMED; the plan's difference form REJECTED; the Wronskian form ruled.** δE_equiv = ½·W(û_AE, û_PP)(r_d), each û normalised to ∫₀^{r_d}û² = 1. `kb_regular` drops its division. The 20 mHa threshold is unchanged. Applied: Task 3 (`kb_regular`); Task 4 (module doc, `delta_e_equiv`); spec §4.8 T5.

The MUSTs and SHOULDs that are not Spec questions:
- M1 (hydrogenic bound 1e-10·Z² and the δ-order ratio in [14, 18]) → Task 1 and spec §4.1;
- M4 (shifts {0, h/2}³ for the lattice sums and every energy ripple) → Tasks 5, 6, 7 and 11, and spec §6.6/§8 row 5;
- M6 (one H per `recipe_hash`) → Global Constraints, Task 4's records test, Task 5's and Task 11's version bumps, Task 11's provisional tables;
- S1 (independent V/V′/V″/u′ matching) → Task 2;
- S2 (per-term frozen FD checks; |HF − Richardson| < 1e-4) → Task 7 and spec §6.6;
- S3/S4 → Task 3 (Q4/Q5); S5 (one-signed Ne series) → Task 10 and spec §6.3; S6 (on-grid-point force 1e-6 at tol 1e-9) → Tasks 6 and 7;
- S7 → Task 13 (Q10); S8 (the relative S1 bound plus E_KB⟨β|φ⟩) → Task 6 and spec §8 row 6.

The NOTEs that named a change were applied:
- the r_peak comment → Task 1;
- the r_nlc definition check and its AE-valence reading → Task 4 and spec §4.5;
- the f32 ln r₀ remedy → Task 8;
- which SVWN was pinned → Task 10;
- the Na/Mg fallback's Zσ → Task 12;
- the measured angle in the tooltip → Task 13.
