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

**Spec:** `docs/superpowers/specs/2026-10-14-pseudopotential-design.md` ("the spec"; the § numbers below refer to it). It was physics-reviewed on 2026-10-14 with the verdict "ready with fixes", and every fix has been folded in; the review is `docs/superpowers/notes/2026-10-14-pseudopotential-spec-review.md`. Parent: `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` §6 and §11 item 1. The plan argues from the spec. Where the plan found a genuine spec defect, it is listed under **Spec questions** at the end with the ruling this plan implements and the task that depends on it. That task's physics-review gate must confirm the ruling before the task lands.

**Interplay:** the orbital-viewer plan (`2026-10-15-orbital-viewer.md`, its Task 8) consumes `qm::radial::AtomResult` and `qm::radial::Partition` from Tasks 1–2 under exactly those names. That task is not part of this plan.

## Model tags and review gates

| # | task | model | physics-review gate before merge |
|---|---|---|---|
| 1 | radial AE atom | **Fable** | **yes** |
| 2 | partition + TM + unscreening | **Fable** | **yes** |
| 3 | KB form, radial KB atom, ghost gates, f ladder | **Fable** | **yes** |
| 4 | NLCC, gates T1–T7, record, `recipe_hash` | **Fable** | **yes** |
| 5 | filter, h_req, filtered re-run | **Fable** | **yes**, plus Spec questions 2–3 confirmed *before* it starts |
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
- **Rule 6.** The spec's thresholds are copied verbatim into the tests: 1e-5, 1e-10, 1e-8, 1 mHa, 5 mHa, 20 mHa, 10 mHa/e, < 2 mHa, > 20 mHa, 2e-3, 3e-5, ±2°, ±0.02 Å, ±0.5°, 0.02 Bohr, 5 %, 0.02 Ha, 1000 K and 1e-9 rad. A miss stops the task and gets reported; a threshold is never loosened. A threshold that is wrong on physics grounds is a spec revision, made by the physics reviewer with a derivation (spec §4.8, last paragraph).
- **Rule 7.** Pseudopotential energies are labelled *valence* energies. Refusals (Li, Na, Mg) are reported, never patched. The `CoresOnly` O–H miss, the non-spherical O splitting and core overlaps are all stated as what they are.
- **Ledger and hash.** `LEDGER` is the spec §4.9 table plus the solver knobs those sections name: the radial SCF tolerance, the r_cmp tolerance and the V_tail floor. `recipe_hash` hashes its `Debug` rendering (FNV-1a 64). A change to any field therefore changes the hash, and with it every cached table and surrogate provenance.
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
crates/qm/src/radial/filter.rs            # T5 (new): spherical Bessel, transforms, mask, FilteredTable, h_req
crates/qm/src/ionic.rs                    # T6 (new): IonicPolicy, Ionic, IonicSetup, IonicError, core overlaps
crates/qm/src/nonlocal.rs                 # T6 (new): NonlocalOp (spheres, β_lm), PseudoHamiltonian; T7: gradients
crates/qm/src/nuclear.rs                  # T6: Centre, smeared_potential, deposit_centres, ionic_local_potential, partial_core_field, ionic_repulsion
crates/qm/src/scf.rs                      # T6: System.ionic, with_ionic, valence_electrons, KsState.ionic, ScfStats.core_overlaps, the ionic path
crates/qm/src/forces.rs                   # T7: pseudised local (LR+V_cmp+V_tail), nonlocal, NLCC terms
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
pub struct Numerics { pub scf_tol: f64, pub scf_max_iter: usize, pub diis_history: usize, pub mixing: f64, pub g2_mesh: f64 }
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
  1. Record **which correlation functional the LSD column uses**. If it is the Vosko–Wilk–Nusair parametrisation, see Spec question 1.
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
//! `numerics` holds the solver knobs the spec names in §4.1/§4.6 (radial SCF tolerance, DIIS) and the
//! plan's G2 zero-count mesh (Spec question 5): they do not change a converged result beyond their
//! tolerance, but they are hashed too.

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
pub struct Numerics { pub scf_tol: f64, pub scf_max_iter: usize, pub diis_history: usize, pub mixing: f64, pub g2_mesh: f64 }

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
    /// q_c = this · π/h; mask at gamma_r · r_c — §5.1.
    pub q_c_over_nyquist: f64, pub gamma_r: f64,
    /// σ_LR = this · h — §5.1.
    pub sigma_lr_over_h: f64,
    /// V_tail is sampled out to where it is below this (Ha) — §5.1.
    pub v_tail_floor: f64,
    /// The h ladder for h_req (Bohr), coarsest first — §4.8 T8.
    pub h_ladder: [f64; 5],
    /// r_core encloses all but this fraction of the core electrons — §5.6.
    pub eps_core: f64,
    /// The dimer gate on f applies for Z ≤ dimer_z_max: R_e within dimer_re (Bohr), ω within dimer_omega_rel — §4.4, §6.4.
    pub dimer_z_max: u8, pub dimer_re: f64, pub dimer_omega_rel: f64,
    /// The dimer gate's bond-length bracket, in units of the element's outermost valence r_peak
    /// (Task 11; Spec question 9): a bound AE dimer must have its minimum inside it.
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
    numerics: Numerics { scf_tol: 1e-10, scf_max_iter: 400, diis_history: 6, mixing: 0.3, g2_mesh: 1e-3 },
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

/// The integrator alone (V_H and v_xc off): ε_nl = −Z²/2n² to 1e-10.
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
            assert!((b.e - exact).abs() < 1e-10, "Z = {z}, n = {n}, l = {l}: ε = {:.13} vs {exact:.13} (Δ {:.2e})", b.e, b.e - exact);
        }
    }
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
/// differences of n·ε_c (relative step 1e-6; error ≪ 1e-10 Ha in the totals).
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
        let d = |x: f64| 1e-6 * x.max(1e-12);
        let (du, dd) = (d(nu), d(nd));
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
- If `hydrogenic_eigenvalues_are_exact` misses 1e-10, do not change δ (a ledger value). Report the measured Δ per (Z, n, l): Numerov's eigenvalue error is O(δ⁴) and grows with Z² in absolute terms, so the physics reviewer decides between a recipe change and a spec revision.
- If an NIST total misses its bound, first confirm the functional (Step 1), then report.

- [ ] **Step 9: Record the numbers.** In `references.md` "M7 radial atom (Task 1)", record:
  - the achieved totals and their Δ to NIST for H, He, Li, C, N, O and Ne;
  - the O E(M=1/3/5);
  - the worst scaling-identity residual;
  - the worst hydrogenic Δ;
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
    // (7) V_scr″(0) = 0 ⇔ c₂² + (2l+5)c₄ = 0, with V_scr(0) = ε + (2l+3)c₂ (spec §4.3, review M1)
    let (c2, c4) = (tm.c[1], tm.c[2]);
    assert!((c2 * c2 + 7.0 * c4).abs() <= 1e-12 * (c2 * c2).max(1e-300), "c2² + 7c4 = {:.3e}", c2 * c2 + 7.0 * c4);
    let v0 = ch.eps + 5.0 * c2;
    assert!((tm.v_scr[0] - v0).abs() < 1e-8, "V_scr(r0) = {:.12} vs ε + 5c₂ = {v0:.12}", tm.v_scr[0]);
    let i = g.nearest(1e-3);
    assert!(((tm.v_scr[i] - v0) / (g.r[i] * g.r[i])).abs() < 1e-4, "the r² coefficient of V_scr does not vanish");
    // unscreened V_ion → −Z_v/r beyond max(r_c, r_nlc) to 1e-8; at Task 2 the partial core is the full
    // core (ρ_pc = ρ_c, the r_nlc → 0 limit); Task 4 re-runs this with the LFC partial core
    let chans = tm_channels(&a, &part, f).expect("every channel solves");
    let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
    let un = unscreen(g, part.z_v, &chans, &occ, &core_density(&a, &part));
    let i_from = chans.iter().map(|c| c.i_c).max().unwrap() + 1;
    for (k, v) in un.v_ion.iter().enumerate() {
        for i in i_from..g.len() {
            let d = v[i] + part.z_v / g.r[i];
            assert!(d.abs() < 1e-8, "channel {k}, r = {:.4}: V_ion + Z_v/r = {d:.3e}", g.r[i]);
        }
    }
    assert!(un.r_cmp <= g.r[i_from], "r_cmp = {} beyond max r_c", un.r_cmp);
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
Expected: all PASS (Task 1's tests too). If `tm_oxygen_2p_satisfies_the_seven_conditions` misses the V_ion tail bound just beyond r_c, report the measured |V_ion + Z_v/r| profile. The core's Hartree tail is the expected cause (spec §4.3 "and the core tail"), but a fix is a physics-review decision, not a tolerance change.

- [ ] **Step 6: Commit**
```bash
git add crates/qm/src/radial
git commit -o -m "feat(qm): core/valence partition, Troullier–Martins construction and unscreening (M7 Task 2)" -- crates/qm/src/radial
```
Before merging, hand the diff to `physics-reviewer`, citing M1 (the c₂² + (2l+5)c₄ condition), TM eqs. 8–11, the S5 cap and the no-core branch.

---

### Task 3: `kb` + `recipe` — the KB form, the radial KB atom, the ghost gates and the f ladder **[Fable — physics-review gate]**

Spec §4.4 (local channel, f ladder), §4.6 (KB, G1, G2), §8 row 3; review NOTE (the GSS criterion re-derived, the integro-differential KB atom). **Spec questions 4 and 5 apply** (the second first-failing test and G2's "node counting").

**Files:**
- Create: `crates/qm/src/radial/kb.rs`, `crates/qm/src/radial/recipe.rs`
- Modify: `crates/qm/src/radial/atom.rs` (`Diis` and its `new`/`mix` become `pub(crate)`), `crates/qm/src/radial/mod.rs`
- Test: unit tests in both new files

**Interfaces:**
- Consumes: Task 1 (`LogGrid`, `bound_state`, `g_coeff`, `regular_start`, `outward`, `outward_sourced`, `inward`, `shooting_end`, `hartree_potential`, `AtomResult`, `SpinMode`, `RadialError`, `LEDGER`) and Task 2 (`Partition`, `Channel`, `TmChannel`, `Unscreened`, `tm_channels`, `unscreen`, `partition`, `core_density`, `Refusal`).
- Produces:
```rust
// kb.rs
pub struct Projector { pub l: u8, pub e_kb: f64, pub gss: f64, pub beta: Vec<f64> /* u-form: δV_l·u_l^PP */, pub dv: Vec<f64>, pub u: Vec<f64> }
pub struct KbPotential { pub z: u8, pub z_v: f64, pub local: LocalCandidate, pub local_l: Option<u8>, pub v_loc: Vec<f64>,
                         pub projectors: Vec<Projector>, pub rho_pc: Vec<f64>, pub r_loc: f64, pub r_cmp: f64 }
pub struct KbInput { pub z: u8, pub z_v: f64, pub v_loc: Vec<f64>, pub projectors: Vec<(u8, f64, Vec<f64>)> /* (l, E_KB, β u-form) */, pub rho_pc: Vec<f64> }
pub struct ValenceConfig { pub occ: Vec<[f64; 2]> /* index = l; (↑, ↓) */ }
pub struct KbAtom { pub eps: Vec<[f64; 2]>, pub u: Vec<[Vec<f64>; 2]>, pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub energy: f64 }
pub enum GssVerdict { GhostFree, Ghost { e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64> } }
pub struct GhostReport { pub g1: Vec<(u8, GssVerdict)>, pub g2_max_eig_error: f64, pub g2_roots: Vec<(u8, Vec<f64>)> }
pub fn polynomial_local(grid: &LogGrid, z_v: f64, r_loc: f64) -> Vec<f64>;
pub fn kb_form(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, local: LocalCandidate, rho_pc: Vec<f64>) -> KbPotential;
pub fn kb_form_with_local(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, v_loc: Vec<f64>, local_l: Option<u8>, rho_pc: Vec<f64>) -> KbPotential;
impl KbPotential { pub fn input(&self) -> KbInput; }
pub fn kb_lower_bound(grid: &LogGrid, v: &[f64], l: u8, proj: Option<(f64, &[f64])>) -> f64;
pub fn kb_eigenvalues(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e_lo: f64, e_hi: f64, mesh: f64) -> Vec<f64>;
pub fn kb_state(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> Vec<f64>;
pub fn kb_regular(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64, i_end: usize) -> Vec<f64>; // T5 (Task 4)
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

/// Spec §8 row 3, second first-failing test, as Spec question 4 rules: the GSS criterion and G2 must
/// catch both constructible ghost cases on O's s channel:
/// (A) E_KB > 0 with a local so deep that ε₁^loc < ε_s (the textbook deep-local ghost);
/// (B) E_KB < 0 with ε₀^loc < ε_s (a local repulsive where φ_s lives but with a deep well at the origin).
/// (A local with ε₀^loc above ε_s forces E_KB < 0 by the variational principle, which GSS calls ghost-free.)
#[test]
fn deliberate_bad_local_is_caught_as_ghost() {
    let t = generate(8).unwrap();
    let g = &t.grid;
    let un = crate::radial::tm::unscreen(g, t.z_v, &t.channels, &t.partition.channels.iter().map(|c| c.occ).collect::<Vec<_>>(), &t.kb.rho_pc);
    let eps_s = t.partition.channels[0].eps;
    let rc = t.channels[0].r_c;
    let screened = |v: &[f64]| v.iter().zip(&t.screen).map(|(a, b)| a + b).collect::<Vec<f64>>();
    let s_levels = |v: &[f64]| (bound_state(g, &screened(v), 0, 0, 0.0, None).map(|b| b.e), bound_state(g, &screened(v), 0, 1, 0.0, None).map(|b| b.e));
    let bump = |w: f64| g.r.iter().map(|r| (-(r / w).powi(2)).exp()).collect::<Vec<f64>>();
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
    // (B) V_loc = V_s^ion + 1 Ha·e^{−(r/r_c)²} − D·e^{−(r/0.15)²}, D raised until ε₀^loc < ε_s
    let s_ion = &un.v_ion[0];
    let narrow = bump(0.15);
    let mut found = false;
    for k in 1..2000 {
        let d = k as f64;
        let v: Vec<f64> = (0..g.len()).map(|i| s_ion[i] + wide[i] - d * narrow[i]).collect();
        if !matches!(s_levels(&v), (Some(e0), _) if e0 < eps_s) { continue; }
        let kb_b = kb_form_with_local(g, 8, t.z_v, &t.channels, &un, v, None, t.kb.rho_pc.clone());
        let sb = kb_b.projectors.iter().find(|p| p.l == 0).unwrap();
        if sb.e_kb >= 0.0 { continue; }
        let err = ghost_gates(g, &kb_b, &t.screen, &t.partition).expect_err("case B is a ghost");
        assert!(err.contains("G1"), "{err}");
        // G2 alone finds the extra s state too (the direct proof, independent of G1)
        let atom = solve_kb_atom(g, &kb_b.input(), &reference_config(&t.partition), SpinMode::Restricted);
        if let Ok(atom) = atom {
            let lo = kb_lower_bound(g, &atom.v[0], 0, Some((sb.e_kb, &sb.beta)));
            let roots = kb_eigenvalues(g, &atom.v[0], Some((sb.e_kb, &sb.beta)), 0, lo, eps_s + LEDGER.gates.g2_window, LEDGER.numerics.g2_mesh);
            assert!(roots.len() >= 2, "G2 must see the ghost below ε_s: roots {roots:?}");
        }
        found = true;
        break;
    }
    assert!(found, "case B was constructible");
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
- the matching determinant;
- the variational argument behind Spec question 4.

```rust
use super::atom::{hartree_potential, Diis, RadialError, SpinMode};
use super::grid::LogGrid;
use super::ledger::{LocalCandidate, LEDGER};
use super::numerov::{bound_state, g_coeff, inward, outward, outward_sourced, regular_start, shooting_end};
use super::partition::Partition;
use super::tm::{TmChannel, Unscreened};
use crate::xc::exc_point;
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct Projector { pub l: u8, pub e_kb: f64, pub gss: f64, pub beta: Vec<f64>, pub dv: Vec<f64>, pub u: Vec<f64> }

#[derive(Clone, Debug, PartialEq)]
pub struct KbPotential {
    pub z: u8, pub z_v: f64, pub local: LocalCandidate, pub local_l: Option<u8>, pub v_loc: Vec<f64>,
    pub projectors: Vec<Projector>, pub rho_pc: Vec<f64>, pub r_loc: f64, pub r_cmp: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KbInput { pub z: u8, pub z_v: f64, pub v_loc: Vec<f64>, pub projectors: Vec<(u8, f64, Vec<f64>)>, pub rho_pc: Vec<f64> }

#[derive(Clone, Debug, PartialEq)]
pub struct ValenceConfig { pub occ: Vec<[f64; 2]> }

#[derive(Clone, Debug, PartialEq)]
pub struct KbAtom { pub eps: Vec<[f64; 2]>, pub u: Vec<[Vec<f64>; 2]>, pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub energy: f64 }

#[derive(Clone, Debug, PartialEq)]
pub enum GssVerdict { GhostFree, Ghost { e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64> } }

#[derive(Clone, Debug, PartialEq)]
pub struct GhostReport { pub g1: Vec<(u8, GssVerdict)>, pub g2_max_eig_error: f64, pub g2_roots: Vec<(u8, Vec<f64>)> }

/// V = −Z_v/r for r ≥ r_loc; inside a0 + a2 r² + a4 r⁴ with value, slope and curvature of −Z_v/r at
/// r_loc: a4 = −3Z_v/(8 r_loc⁵), a2 = 5Z_v/(4 r_loc³), a0 = −15Z_v/(8 r_loc) (finite, zero slope at 0).
pub fn polynomial_local(grid: &LogGrid, z_v: f64, r_loc: f64) -> Vec<f64> {
    let (a0, a2, a4) = (-15.0 * z_v / (8.0 * r_loc), 5.0 * z_v / (4.0 * r_loc.powi(3)), -3.0 * z_v / (8.0 * r_loc.powi(5)));
    grid.r.iter().map(|&r| if r >= r_loc { -z_v / r } else { a0 + r * r * (a2 + r * r * a4) }).collect()
}

pub fn kb_form(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, local: LocalCandidate, rho_pc: Vec<f64>) -> KbPotential {
    let (v_loc, local_l) = match local {
        LocalCandidate::HighestChannel => (un.v_ion[channels.len() - 1].clone(), Some(channels[channels.len() - 1].l)),
        LocalCandidate::Polynomial => {
            let r_loc = channels.iter().map(|c| c.r_c).fold(f64::INFINITY, f64::min);
            (polynomial_local(grid, z_v, r_loc), None)
        }
    };
    let mut kb = kb_form_with_local(grid, z, z_v, channels, un, v_loc, local_l, rho_pc);
    kb.local = local;
    kb
}

/// Every channel other than `local_l` becomes a projector δV_l = V_l^ion − V_loc.
pub fn kb_form_with_local(grid: &LogGrid, z: u8, z_v: f64, channels: &[TmChannel], un: &Unscreened, v_loc: Vec<f64>, local_l: Option<u8>, rho_pc: Vec<f64>) -> KbPotential {
    let n = grid.len();
    let projectors = channels.iter().enumerate().filter(|(_, c)| Some(c.l) != local_l).map(|(k, c)| {
        let dv: Vec<f64> = (0..n).map(|i| un.v_ion[k][i] - v_loc[i]).collect();
        let beta: Vec<f64> = (0..n).map(|i| dv[i] * c.u[i]).collect();
        let denom = grid.integrate(&(0..n).map(|i| c.u[i] * beta[i]).collect::<Vec<_>>());
        let num = grid.integrate(&beta.iter().map(|b| b * b).collect::<Vec<_>>());
        Projector { l: c.l, e_kb: 1.0 / denom, gss: num / denom, beta, dv, u: c.u.clone() }
    }).collect();
    let r_loc = channels.iter().map(|c| c.r_c).fold(f64::INFINITY, f64::min);
    KbPotential { z, z_v, local: LocalCandidate::HighestChannel, local_l, v_loc, projectors, rho_pc, r_loc, r_cmp: un.r_cmp }
}

impl KbPotential {
    pub fn input(&self) -> KbInput {
        KbInput { z: self.z, z_v: self.z_v, v_loc: self.v_loc.clone(),
                  projectors: self.projectors.iter().map(|p| (p.l, p.e_kb, p.beta.clone())).collect(), rho_pc: self.rho_pc.clone() }
    }
}
```

- [ ] **Step 4: Implement the radial KB eigenproblem**

```rust
/// Last index where |β| exceeds 1e-12·max|β| (β = δV·u is compact: δV → 0 beyond r_cmp), plus 3:
/// the e-independent matching index (an e-dependent one would make det(e) jump).
fn matching_index(beta: &[f64]) -> usize {
    let m = beta.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    beta.iter().rposition(|b| b.abs() > 1e-12 * m).unwrap_or(0) + 3
}

struct KbSolve { det: f64, m: [[f64; 2]; 2], uh: Vec<f64>, up: Vec<f64>, ui: Vec<f64>, i_m: usize }

/// The radial KB equation at energy e (spec §4.6, review NOTE): u = a·u_h + c·u_p for r ≤ r_m
/// (u_h regular homogeneous; u_p regular with source −β: −½u″ + (V_eff − e)u = −β), u = b·u_i
/// beyond (u_i decaying). Value/slope matching at r_m and c = E_KB⟨β|u⟩ give M(a, c)ᵀ = 0 with
///   M = [[W(u_h,u_i), W(u_p,u_i)], [E_KB⟨β|u_h⟩, E_KB⟨β|u_p⟩ − 1]],  W = u₁u₂′ − u₁′u₂ in x at r_m
/// (the common 1/r cancels): no poles; eigenvalues are the zeros of det M.
fn kb_solve(grid: &LogGrid, v: &[f64], e_kb: f64, beta: &[f64], l: u8, e: f64, i_m: usize) -> KbSolve {
    let n = grid.len();
    let g = g_coeff(grid, v, l, e);
    let (y0, y1) = regular_start(grid, l, 0.0);
    let i_out = i_m + 2;
    let (yh, _) = outward(&g, y0, y1, i_out, grid.delta);
    let s: Vec<f64> = (0..=i_out).map(|i| 2.0 * grid.r[i].powf(1.5) * beta[i]).collect();
    let yp = outward_sourced(&g, &s, i_out, grid.delta);
    let ie = shooting_end(&g, grid.delta).max(i_m + 4).min(n - 1);
    let yi = inward(&g, i_m - 2, ie, grid.delta);
    let sr = |i: usize| grid.r[i].sqrt();
    let uh: Vec<f64> = (0..=i_out).map(|i| yh[i] * sr(i)).collect();
    let up: Vec<f64> = (0..=i_out).map(|i| yp[i] * sr(i)).collect();
    let mut ui = vec![0.0; n];
    for (j, y) in yi.iter().enumerate() { ui[i_m - 2 + j] = y * sr(i_m - 2 + j); }
    let dx = |f: &[f64], i: usize| (f[i - 2] - 8.0 * f[i - 1] + 8.0 * f[i + 1] - f[i + 2]) / (12.0 * grid.delta);
    let w = |a: &[f64], b: &[f64]| a[i_m] * dx(b, i_m) - dx(a, i_m) * b[i_m];
    let dot = |u: &[f64]| grid.partial(&(0..n).map(|i| if i <= i_m { beta[i] * u[i] } else { 0.0 }).collect::<Vec<_>>(), i_m);
    let m = [[w(&uh, &ui), w(&up, &ui)], [e_kb * dot(&uh), e_kb * dot(&up) - 1.0]];
    KbSolve { det: m[0][0] * m[1][1] - m[0][1] * m[1][0], m, uh, up, ui, i_m }
}

/// min(V_eff) + min(0, E_KB·∫β²): a lower bound of the KB spectrum of channel l.
pub fn kb_lower_bound(grid: &LogGrid, v: &[f64], l: u8, proj: Option<(f64, &[f64])>) -> f64 {
    let lf = l as f64;
    let floor = grid.r.iter().zip(v).map(|(r, vi)| vi + lf * (lf + 1.0) / (2.0 * r * r)).fold(f64::INFINITY, f64::min);
    floor + proj.map_or(0.0, |(e, b)| (e * grid.integrate(&b.iter().map(|x| x * x).collect::<Vec<_>>())).min(0.0))
}

/// Every eigenvalue of channel l in [e_lo, e_hi]: without a projector by node counting
/// (`bound_state`); with one, by the sign changes of det M on a mesh, each bisected to
/// 1e-13·max(1,|e|). Node counting does not order a nonlocal operator's eigenvalues (Spec question 5).
pub fn kb_eigenvalues(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e_lo: f64, e_hi: f64, mesh: f64) -> Vec<f64> {
    match proj {
        None => (0..).map_while(|k| bound_state(grid, v, l, k, 0.0, None)).map(|b| b.e)
            .take_while(|&e| e < e_hi).filter(|&e| e >= e_lo).collect(),
        Some((e_kb, beta)) => {
            let i_m = matching_index(beta);
            let det = |e: f64| kb_solve(grid, v, e_kb, beta, l, e, i_m).det;
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

/// The normalised eigenfunction at an eigenvalue e (positive at its outermost maximum).
pub fn kb_state(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, e: f64) -> Vec<f64> {
    let Some((e_kb, beta)) = proj else { return crate::radial::numerov::eigenfunction(grid, v, l, e, 0.0).u };
    let n = grid.len();
    let s = kb_solve(grid, v, e_kb, beta, l, e, matching_index(beta));
    let row = if s.m[0][0].abs() + s.m[0][1].abs() >= s.m[1][0].abs() + s.m[1][1].abs() { 0 } else { 1 };
    let (a, c) = (s.m[row][1], -s.m[row][0]);
    let mut u = vec![0.0; n];
    for i in 0..=s.i_m { u[i] = a * s.uh[i] + c * s.up[i]; }
    let b = u[s.i_m] / s.ui[s.i_m];
    for i in s.i_m + 1..n { u[i] = b * s.ui[i]; }
    let norm = grid.integrate(&u.iter().map(|x| x * x).collect::<Vec<_>>()).sqrt();
    let umax = u.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let ip = (1..n - 1).rev().find(|&i| u[i].abs() >= u[i - 1].abs() && u[i].abs() >= u[i + 1].abs() && u[i].abs() > 1e-3 * umax).unwrap_or(0);
    let sgn = if u[ip] < 0.0 { -1.0 } else { 1.0 };
    u.iter().map(|x| sgn * x / norm).collect()
}

/// The regular solution at energy e out to i_end (unnormalised): T5's log derivatives (Task 4).
/// With a projector: u = u_h + c·u_p, c = E_KB⟨β|u_h⟩/(1 − E_KB⟨β|u_p⟩), ⟨β|·⟩ over β's support.
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
    let c = e_kb * dot(&uh) / (1.0 - e_kb * dot(&up));
    (0..=i_end).map(|i| uh[i] + c * up[i]).collect()
}
```

- [ ] **Step 5: Implement the radial KB atom (SCF) and the gates**

```rust
pub fn reference_config(part: &Partition) -> ValenceConfig {
    ValenceConfig { occ: (0..=part.l_max).map(|l| {
        let o = part.channels.iter().find(|c| c.l == l && c.occupied).map_or(0.0, |c| c.occ);
        [0.5 * o, 0.5 * o]
    }).collect() }
}

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

/// Lowest eigenvalue of channel l: a ±0.1 Ha window around `prev` at the G2 mesh, else the full
/// window [lower bound, 0].
fn lowest(grid: &LogGrid, v: &[f64], proj: Option<(f64, &[f64])>, l: u8, prev: f64) -> Option<f64> {
    let mesh = LEDGER.numerics.g2_mesh;
    if prev.is_finite() {
        if let Some(&e) = kb_eigenvalues(grid, v, proj, l, prev - 0.1, (prev + 0.1).min(0.0), mesh).first() { return Some(e); }
    }
    kb_eigenvalues(grid, v, proj, l, kb_lower_bound(grid, v, l, proj), 0.0, mesh).first().copied()
}

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
                let src = if spin == SpinMode::Restricted { 0 } else { s };
                let e = lowest(grid, &v[src], proj, l as u8, eps[l][s]).ok_or(RadialError::Unbound { z: input.z, n: 0, l: l as u8 })?;
                let u = kb_state(grid, &v[src], proj, l as u8, e);
                for i in 0..n { rho[s][i] += config.occ[l][s] * u[i] * u[i] / (4.0 * PI * grid.r[i] * grid.r[i]); }
                eps[l][s] = e;
                us[l][s] = u;
            }
        }
        Ok(rho)
    };
    let zero = [vec![0.0; n], vec![0.0; n]];
    let mut rho_in = occupy(&kb_potential(grid, input, &zero), &mut eps, &mut us)?;
    let mut diis = Diis::new(num.diis_history);
    let mut delta = f64::INFINITY;
    for _ in 0..num.scf_max_iter {
        let v = kb_potential(grid, input, &rho_in);
        let rho_out = occupy(&v, &mut eps, &mut us)?;
        delta = (0..2).map(|s| (0..n).map(|i| (rho_out[s][i] - rho_in[s][i]).abs() * w[i]).sum::<f64>()).sum();
        if delta < num.scf_tol {
            // E = (T_s + E_NL) + ∫ρ V_loc + E_H + E_xc[ρ + ρ_pc], with T_s + E_NL = Σfε − Σ_σ∫ρ_σ V_σ^in
            let shell = |f: &[f64]| grid.integrate(&f.iter().zip(&grid.r).map(|(a, r)| 4.0 * PI * r * r * a).collect::<Vec<_>>());
            let band: f64 = (0..nl).map(|l| (0..2).filter(|&s| config.occ[l][s] > 0.0).map(|s| config.occ[l][s] * eps[l][s]).sum::<f64>()).sum();
            let pot: f64 = (0..2).map(|s| shell(&(0..n).map(|i| rho_out[s][i] * v[s][i]).collect::<Vec<_>>())).sum();
            let total: Vec<f64> = (0..n).map(|i| rho_out[0][i] + rho_out[1][i]).collect();
            let v_h = hartree_potential(grid, &total);
            let e_loc = shell(&(0..n).map(|i| total[i] * input.v_loc[i]).collect::<Vec<_>>());
            let e_h = 0.5 * shell(&(0..n).map(|i| total[i] * v_h[i]).collect::<Vec<_>>());
            let e_xc = shell(&(0..n).map(|i| { let c = 0.5 * input.rho_pc[i]; exc_point(rho_out[0][i] + c, rho_out[1][i] + c).0 }).collect::<Vec<_>>());
            let v_out = kb_potential(grid, input, &rho_out);
            return Ok(KbAtom { eps, u: us, rho: rho_out, v: v_out, energy: band - pot + e_loc + e_h + e_xc });
        }
        rho_in = diis.mix(rho_in, rho_out, &w, num.mixing);
    }
    Err(RadialError::NotConverged { z: input.z, iterations: num.scf_max_iter, delta })
}

/// GSS (PRB 44, 8503): ghost-free iff (E_KB < 0 ∧ ε_l < ε₀^loc) ∨ (E_KB > 0 ∧ ε₀^loc < ε_l < ε₁^loc);
/// an unbound local level sits at the continuum edge, 0.
pub fn gss_verdict(e_kb: f64, eps_ref: f64, eps0_loc: Option<f64>, eps1_loc: Option<f64>) -> GssVerdict {
    let (e0, e1) = (eps0_loc.unwrap_or(0.0), eps1_loc.unwrap_or(0.0));
    let ok = if e_kb < 0.0 { eps_ref < e0 } else { e0 < eps_ref && eps_ref < e1 };
    if ok { GssVerdict::GhostFree } else { GssVerdict::Ghost { e_kb, eps_ref, eps0_loc, eps1_loc } }
}

/// G1 (GSS, every projector channel, on the reference screening `screen`), then G2 (the SCF KB atom
/// at the reference configuration reproduces the valence eigenvalues to LEDGER.gates.g2_eig, and
/// no channel l ≤ l_max has more bound states below min(0, ε_v,max + g2_window) than the channels
/// generated for it — the zero count of det M, Spec question 5).
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
    for l in 0..=part.l_max {
        let proj = kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
        let roots = kb_eigenvalues(grid, &atom.v[0], proj, l, kb_lower_bound(grid, &atom.v[0], l, proj), top, LEDGER.numerics.g2_mesh);
        let expected = part.channels.iter().filter(|c| c.l == l && c.eps < top).count();
        if roots.len() != expected {
            return Err(format!("G2: l = {l} has {} bound states below {top:.3} Ha ({roots:?}), expected {expected}", roots.len()));
        }
        g2_roots.push((l, roots));
    }
    Ok(GhostReport { g1, g2_max_eig_error, g2_roots })
}
```

- [ ] **Step 6: Implement the ladder**

`crates/qm/src/radial/recipe.rs` (module doc: spec §4.4 in full, including the S6b paragraph that Task 11 implements):
```rust
use super::atom::{ground_state, AtomResult, SpinMode};
use super::grid::LogGrid;
use super::kb::{ghost_gates, kb_form, GhostReport, KbPotential};
use super::ledger::{LocalCandidate, LEDGER};
use super::partition::{core_density, partition, Partition, Refusal};
use super::tm::{tm_channels, unscreen, TmChannel};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct PseudoTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub f: f64, pub local: LocalCandidate, pub kb: KbPotential,
    pub channels: Vec<TmChannel>, pub screen: Vec<f64>, pub partition: Partition,
    pub reference: Arc<AtomResult>, pub grid: LogGrid, pub ghost: GhostReport,
}

/// Z in, potential out (spec §4.4): f from LEDGER.f_ladder softest first, local candidates in
/// LEDGER.local_order; the first (f, local) passing every gate is kept; none → refused at this
/// recipe version (the reasons listed). Task 4 adds T1–T7 to `candidate`; Task 11 the dimer gate.
pub fn generate(z: u8) -> Result<PseudoTable, Refusal> {
    let reference = Arc::new(ground_state(z, SpinMode::Restricted).map_err(|e| Refusal::Radial { z, message: e.to_string() })?);
    let part = partition(&reference)?;
    // Task 3: the full core as the partial core (Task 2's r_nlc → 0 limit); Task 4 replaces this
    // line by nlcc::partial_core (LFC, spec §4.5)
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

pub(crate) fn candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, rho_pc: &[f64]) -> Result<PseudoTable, String> {
    let g = &reference.grid;
    let channels = tm_channels(reference, part, f)?;
    let occ: Vec<f64> = part.channels.iter().map(|c| c.occ).collect();
    let un = unscreen(g, part.z_v, &channels, &occ, rho_pc);
    let screen: Vec<f64> = (0..g.len()).map(|i| channels[0].v_scr[i] - un.v_ion[0][i]).collect();
    let kb = kb_form(g, part.z, part.z_v, &channels, &un, local, rho_pc.to_vec());
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
Before merging, hand the diff to `physics-reviewer`, with Spec questions 4 and 5 named.

---

### Task 4: `nlcc` + gates T1–T7 + the generation record + `recipe_hash` **[Fable — physics-review gate]**

Spec §4.5, §4.8 (T1–T7), §4.9 (`recipe_hash`), §5.6 (r_core), §6.2, §8 row 4; review S4 (T5 in δE_equiv), S6a (the pinned outputs table) and M3 (r_core). **Spec question 12 applies** (the pole-free form of δE_equiv).

**Files:**
- Create: `crates/qm/src/radial/nlcc.rs`, `crates/qm/src/radial/gates.rs`, `crates/qm/src/radial/record.rs`, `crates/validate/tests/pseudo_gen.rs`
- Modify: `crates/qm/src/radial/recipe.rs` (the LFC partial core replaces the full core; T1–T7 run in `candidate`; the record), `crates/qm/src/radial/ledger.rs` (`hash_ledger`, `recipe_hash`, the grep test), `crates/qm/src/radial/mod.rs`, `.gitignore` (`/crates/validate/last-run-generation-records.md`), `crates/validate/references.md` (section "M7 generation records" with the begin/end markers)

**Interfaces:**
- Consumes: Tasks 1–3, in particular `ShellOcc`, `AtomOptions::fixed`, `solve_atom`, `ground_state(z, Polarised)`, `kb_regular`, `solve_kb_atom`, `ValenceConfig`, `reference_config`, `PseudoTable`, `candidate`, `valence_density`, `core_density`, `numerov::{outward, regular_start, g_coeff}`.
- Produces:
```rust
// nlcc.rs
pub struct PartialCore { pub rho_pc: Vec<f64>, pub r_nlc: Option<f64>, pub a: f64, pub b: f64, pub r_core: f64, pub n_core: f64 }
pub fn partial_core(grid: &LogGrid, rho_c: &[f64], rho_v: &[f64]) -> PartialCore;
pub fn core_radius(grid: &LogGrid, rho_c: &[f64]) -> f64;                         // r_core (§5.6), 0 without a core
// gates.rs
pub struct TransferRow { pub name: String, pub de_ae: f64, pub de_pp: f64, pub max_eig_error: f64 }
pub struct GateReport { pub t1: f64, pub t2: f64, pub t3: Vec<TransferRow>, pub t4: f64, pub t5: Vec<(u8, f64)> /* (l, max δE_equiv) */,
                        pub t6: Vec<(u8, u8, f64, f64)> /* (l, l′, AE, PP) slopes */, pub skipped: Vec<String> }
pub fn run_gates(t: &PseudoTable) -> Result<GateReport, String>;
pub fn transfer_configs(t: &PseudoTable, hund: &AtomResult) -> Vec<TransferConfig>;
pub struct TransferConfig { pub name: &'static str, pub ae: AtomOptions, pub pp: ValenceConfig, pub spin: SpinMode }
pub fn delta_e_equiv(t: &PseudoTable, l: u8, e: f64) -> f64;
// record.rs
pub struct GenerationRecord { pub z: u8, pub recipe_hash: u64, pub f: f64, pub local: String, pub r_c: Vec<(u8, f64)>, pub z_v: f64,
    pub n_core: u32, pub e_kb: Vec<(u8, f64)>, pub gss: Vec<(u8, f64)>, pub r_nlc: Option<f64>, pub r_cmp: f64, pub r_core: f64,
    pub gates: GateReport, pub h_req: Option<f64> /* Task 5 */, pub filter: Vec<FilterRow> /* Task 5 */, pub dimer: String /* Task 11 */,
    pub seconds: f64 }
pub struct FilterRow { pub h: f64, pub de: f64, pub max_eig: f64, pub t3_max: f64 }   // Task 5 fills it
pub fn record_of(t: &PseudoTable, seconds: f64) -> GenerationRecord;
pub fn records_table(rows: &[(u8, Result<GenerationRecord, Refusal>)]) -> String;
// ledger.rs
pub fn hash_ledger(l: &Ledger) -> u64;   // FNV-1a 64 over format!("{l:?}")
pub fn recipe_hash() -> u64;             // hash_ledger(&LEDGER)
// recipe.rs — PseudoTable gains:
pub partial_core: PartialCore, pub gates: GateReport, pub record: GenerationRecord,
pub(crate) fn candidate(reference: &Arc<AtomResult>, part: &Partition, f: f64, local: LocalCandidate, pc: &PartialCore) -> Result<PseudoTable, String>;
```

- [ ] **Step 1: Write the failing tests**

`crates/qm/src/radial/gates.rs` tests:
```rust
/// Spec §8 row 4, first failing test: AE vs KB-PP total-energy differences between the reference
/// and the +1 ion, the s → p promotion and the spin-polarised Hund multiplet, each within 5 mHa.
#[test]
fn oxygen_transfer_energies_within_5mha() {
    let t = crate::radial::generate(8).expect("O generates");
    let names: Vec<&str> = t.gates.t3.iter().map(|r| r.name.as_str()).collect();
    for want in ["cation", "s→p", "Hund"] { assert!(names.contains(&want), "missing {want}: {names:?}"); }
    for row in &t.gates.t3 {
        assert!((row.de_ae - row.de_pp).abs() < 5e-3, "{}: ΔE_AE {:.6} ΔE_PP {:.6}", row.name, row.de_ae, row.de_pp);
        assert!(row.max_eig_error < 5e-3, "{}: T4 eigenvalue error {:.6}", row.name, row.max_eig_error);
    }
}

/// δE_equiv vanishes at the reference eigenvalue to first order (norm conservation ⇒ equal ∂D/∂E,
/// spec §4.8 T5 derivation): at ε_l it is far below the 20 mHa gate.
#[test]
fn log_derivative_mismatch_vanishes_at_the_reference_energy() {
    let t = crate::radial::generate(8).unwrap();
    for c in t.partition.channels.iter().filter(|c| c.occupied) {
        let d = delta_e_equiv(&t, c.l, c.eps);
        assert!(d.abs() < 1e-4, "l = {}: δE_equiv(ε_l) = {d:.3e}", c.l);
    }
}
```
`crates/qm/src/radial/nlcc.rs` tests:
```rust
use super::*;
use crate::radial::{core_density, ground_state, partition, valence_density, LogGrid, SpinMode, LEDGER};

#[test]
fn partial_core_is_continuous_with_slope_at_r_nlc() {
    let a = ground_state(8, SpinMode::Restricted).unwrap();
    let p = partition(&a).unwrap();
    let (rc, rv) = (core_density(&a, &p), valence_density(&a, &p));
    let pc = partial_core(&a.grid, &rc, &rv);
    let r = pc.r_nlc.expect("O has a core");
    let model = |x: f64| pc.a * (pc.b * x).sin() / x;
    let i = a.grid.nearest(r);
    assert!((model(r) - rc[i]).abs() < 1e-10 * rc[i]);
    let d = a.grid.derivative(&rc)[i];
    let dm = pc.a * (pc.b * (pc.b * r).cos() / r - (pc.b * r).sin() / (r * r));
    assert!((dm - d).abs() < 1e-8 * d.abs(), "slope {dm} vs {d}");
    assert!((rc[i] - LEDGER.nlcc_crossover * rv[i]).abs() <= 0.05 * rc[i], "r_nlc sits at the ρ_c = ρ_v crossing");
}

/// The explicit no-core branch: H has ρ_pc ≡ 0, r_nlc undefined, r_core = 0 (spec §4.2 item 4, §5.6).
#[test]
fn no_core_means_no_partial_core() {
    let a = ground_state(1, SpinMode::Restricted).unwrap();
    let p = partition(&a).unwrap();
    let pc = partial_core(&a.grid, &core_density(&a, &p), &valence_density(&a, &p));
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

/// Task 2's tail condition re-run with the LFC partial core: V_ion → −Z_v/r beyond max(r_c, r_nlc).
#[test]
fn v_ion_is_coulombic_beyond_rc_and_r_nlc_with_lfc() {
    let t = crate::radial::generate(8).unwrap();
    let g = &t.grid;
    let i_from = t.channels.iter().map(|c| c.i_c).max().unwrap().max(g.nearest(t.partial_core.r_nlc.unwrap())) + 1;
    let occ: Vec<f64> = t.partition.channels.iter().map(|c| c.occ).collect();
    let un = crate::radial::tm::unscreen(g, t.z_v, &t.channels, &occ, &t.partial_core.rho_pc);
    for (k, v) in un.v_ion.iter().enumerate() {
        for i in i_from..g.len() {
            assert!((v[i] + t.z_v / g.r[i]).abs() < 1e-8, "channel {k}, r = {:.4}", g.r[i]);
        }
    }
}
```
`crates/qm/src/radial/ledger.rs` tests:
```rust
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
```
`crates/validate/tests/pseudo_gen.rs` (new suite; module doc: what it gates, the markers, how to regenerate):
```rust
//! M7 generator gates per element (spec §6.2) and the pinned outputs table (review S6a).
//! Run: `cargo test -p validate --release --test pseudo_gen`. When the records move (a universal
//! constant changed, or Task 5/11/12 add a column or element), the test writes the new block to the
//! gitignored `crates/validate/last-run-generation-records.md` and fails; after physics review, copy it
//! between the markers in `references.md` ("M7 generation records").
use qm::radial::{generate, record::records_table};

const BEGIN: &str = "<!-- m7-generation-records:begin -->";
const END: &str = "<!-- m7-generation-records:end -->";
/// Li is listed to pin its refusal; Be–Ne are the elements the spec expects accepted (§6.2).
const ZS: [u8; 8] = [3, 4, 5, 6, 7, 8, 9, 10];

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn generation_record_table_matches_references_md() {
    let rows: Vec<_> = ZS.iter().map(|&z| (z, generate(z).map(|t| t.record))).collect();
    let table = records_table(&rows);
    let refs = include_str!("../references.md");
    let pinned = refs.split(BEGIN).nth(1).and_then(|s| s.split(END).next()).map(str::trim).unwrap_or("");
    if pinned != table.trim() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-generation-records.md");
        std::fs::write(path, &table).expect("write the regenerated block");
        panic!("the generation records differ from references.md — review, then copy {path} between the markers:\n{table}");
    }
}

/// Spec §6.2: every gate on Be–Ne (T1–T7 inside `generate`; here their numbers are printed and the
/// spec's acceptance asserted); Li refused with its reason.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn be_to_ne_pass_every_gate_and_li_is_refused() {
    assert!(generate(3).is_err_and(|e| e.to_string().contains("ONCV")));
    for z in 4u8..=10 {
        let t = generate(z).unwrap_or_else(|e| panic!("Z = {z} refused: {e}"));
        let g = &t.gates;
        eprintln!("Z = {z}: f {} {:?} T1 {:.2e} T2 {:.1e} T3 {:?} T4 {:.2e} T5 {:?} T6 {:?}", t.f, t.local, g.t1, g.t2,
                  g.t3.iter().map(|r| (r.name.clone(), r.de_ae - r.de_pp)).collect::<Vec<_>>(), g.t4, g.t5, g.t6);
        assert!(g.t1 < 1e-3 && g.t2 < 1e-10 && g.t4 < 5e-3);
        assert!(g.t3.iter().all(|r| (r.de_ae - r.de_pp).abs() < 5e-3));
        assert!(g.t5.iter().all(|(_, d)| *d < 20e-3));
        assert!(g.t6.iter().all(|(_, _, a, p)| (a - p).abs() < 10e-3));
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release radial` and `cargo test -p validate --release --test pseudo_gen`
Expected: compile errors (`nlcc`, `gates`, `record` missing, `PseudoTable.gates` missing).

- [ ] **Step 3: Implement NLCC and r_core**

`crates/qm/src/radial/nlcc.rs` (module doc: spec §4.5 verbatim plus the derivation of x cot x = κ):
```rust
use super::grid::LogGrid;
use super::ledger::{NlccMode, LEDGER};
use std::f64::consts::PI;

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

/// LFC (PRB 26, 1738): ρ_pc = ρ_c for r ≥ r_nlc, A sin(Br)/r inside. Value continuity gives
/// A = ρ_c(R)R/sin(BR); slope continuity gives x cot x = κ with x = BR, κ = 1 + Rρ_c′(R)/ρ_c(R) < 1,
/// and x cot x falls monotonically from 1 (x → 0) to −∞ (x → π): one root in (0, π), by bisection.
/// r_nlc is the outermost grid point with ρ_c ≥ LEDGER.nlcc_crossover·ρ_v. No core or NlccMode::Off:
/// ρ_pc ≡ 0 and r_nlc = None (the explicit no-core branch).
pub fn partial_core(grid: &LogGrid, rho_c: &[f64], rho_v: &[f64]) -> PartialCore {
    let n = grid.len();
    let n_core = grid.integrate(&shell(grid, rho_c));
    let r_core = core_radius(grid, rho_c);
    if n_core <= 0.0 || LEDGER.nlcc == NlccMode::Off {
        return PartialCore { rho_pc: vec![0.0; n], r_nlc: None, a: 0.0, b: 0.0, r_core, n_core };
    }
    let i = (0..n).rev().find(|&i| rho_c[i] >= LEDGER.nlcc_crossover * rho_v[i]).expect("the core dominates at the nucleus");
    let r = grid.r[i];
    let kappa = 1.0 + r * grid.derivative(rho_c)[i] / rho_c[i];
    let (mut lo, mut hi) = (1e-12, PI - 1e-12);
    for _ in 0..200 {
        let m = 0.5 * (lo + hi);
        if m / m.tan() > kappa { lo = m } else { hi = m }
    }
    let x = 0.5 * (lo + hi);
    let (b, a) = (x / r, rho_c[i] * r / x.sin());
    let rho_pc = (0..n).map(|j| if j < i { a * (b * grid.r[j]).sin() / grid.r[j] } else { rho_c[j] }).collect();
    PartialCore { rho_pc, r_nlc: Some(r), a, b, r_core, n_core }
}
```

- [ ] **Step 4: Implement the gates**

`crates/qm/src/radial/gates.rs`. Its module doc carries:
- spec §4.8's table;
- S4's Wronskian derivation of ∂D/∂E = −(2/u(r_d)²)∫₀^{r_d}u² dr, and from it the pole-free form used here, δE_equiv(E) = [u u′/(2∫₀^{r_d}u²)]_PP − [u u′/(2∫₀^{r_d}u²)]_AE at r_d (Spec question 12: to first order this equals (D^PP − D^AE)·u²/(2∫u²), and it has no poles);
- the configuration rules: cation = one electron out of the highest valence shell; s→p = one electron from the valence s to the valence p, when both channels exist; anion = one electron into the lowest shell with room, kept only if bound in both atoms; Hund = the spin-polarised `ground_state` multiplet. All but Hund are spin-restricted, like the reference.

```rust
use super::atom::{ground_state, solve_atom, AtomOptions, AtomResult, Occupations, ShellOcc, SpinMode};
use super::kb::{kb_regular, reference_config, solve_kb_atom, ValenceConfig};
use super::ledger::LEDGER;
use super::numerov::{g_coeff, outward, regular_start};
use super::recipe::PseudoTable;

#[derive(Clone, Debug, PartialEq)]
pub struct TransferRow { pub name: String, pub de_ae: f64, pub de_pp: f64, pub max_eig_error: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct GateReport { pub t1: f64, pub t2: f64, pub t3: Vec<TransferRow>, pub t4: f64, pub t5: Vec<(u8, f64)>, pub t6: Vec<(u8, u8, f64, f64)>, pub skipped: Vec<String> }

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

pub fn transfer_configs(t: &PseudoTable, hund: &AtomResult) -> Vec<TransferConfig> {
    let base = reference_config(&t.partition);
    let cap = |l: usize| 2.0 * (2 * l + 1) as f64;
    let total = |c: &ValenceConfig, l: usize| c.occ[l][0] + c.occ[l][1];
    let chans = &t.partition.channels;
    let homo = chans.iter().filter(|c| c.occupied).max_by(|a, b| a.eps.total_cmp(&b.eps)).expect("valence").l as usize;
    let restricted = |name, pp: ValenceConfig| TransferConfig { name, ae: AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &pp)), pp, spin: SpinMode::Restricted };
    let mut out = vec![restricted("cation", moved(&base, Some(homo), None))];
    let has = |l: usize| chans.iter().any(|c| c.l as usize == l);
    if has(0) && has(1) && total(&base, 0) >= 1.0 && total(&base, 1) <= cap(1) - 1.0 {
        out.push(restricted("s→p", moved(&base, Some(0), Some(1))));
    }
    let room = (0..base.occ.len()).filter(|&l| has(l) && total(&base, l) < cap(l))
        .min_by(|&a, &b| chans.iter().find(|c| c.l as usize == a).unwrap().eps.total_cmp(&chans.iter().find(|c| c.l as usize == b).unwrap().eps));
    if let Some(l) = room { out.push(restricted("anion", moved(&base, None, Some(l)))); }
    // Hund: the AE spin-polarised multiplet and the same valence occupations per spin for the PP
    let mut pp = ValenceConfig { occ: vec![[0.0; 2]; base.occ.len()] };
    for o in hund.orbitals.iter().filter(|o| t.partition.valence.iter().any(|v| v.n == o.n && v.l == o.l)) {
        pp.occ[o.l as usize] = o.occ;
    }
    out.push(TransferConfig { name: "Hund", ae: AtomOptions::aufbau(SpinMode::Polarised, hund.multiplicity), pp, spin: SpinMode::Polarised });
    out
}

/// The AE regular solution at e in the reference potential, u-form, to i_end.
fn ae_regular(t: &PseudoTable, l: u8, e: f64, i_end: usize) -> Vec<f64> {
    let g = &t.grid;
    let gc = g_coeff(g, &t.reference.v[0], l, e);
    let (y0, y1) = regular_start(g, l, t.z as f64);
    outward(&gc, y0, y1, i_end, g.delta).0.iter().enumerate().map(|(i, y)| y * g.r[i].sqrt()).collect()
}

/// δE_equiv(e) at r_d = LEDGER.gates.t5_rd_factor · max r_c (pole-free form, module doc).
pub fn delta_e_equiv(t: &PseudoTable, l: u8, e: f64) -> f64 {
    let g = &t.grid;
    let r_d = LEDGER.gates.t5_rd_factor * t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max);
    let i_d = g.nearest(r_d);
    let v_pp: Vec<f64> = t.kb.v_loc.iter().zip(&t.screen).map(|(a, b)| a + b).collect();
    let proj = t.kb.projectors.iter().find(|p| p.l == l).map(|p| (p.e_kb, p.beta.as_slice()));
    let term = |u: &[f64]| {
        let norm = g.partial(&u.iter().map(|x| x * x).chain(std::iter::repeat(0.0)).take(g.len()).collect::<Vec<_>>(), i_d);
        let du = (u[i_d - 2] - 8.0 * u[i_d - 1] + 8.0 * u[i_d + 1] - u[i_d + 2]) / (12.0 * g.delta * g.r[i_d]);
        u[i_d] * du / (2.0 * norm)
    };
    term(&kb_regular(g, &v_pp, proj, l, e, i_d + 2)) - term(&ae_regular(t, l, e, i_d + 2))
}

pub fn run_gates(t: &PseudoTable) -> Result<GateReport, String> {
    let gates = &LEDGER.gates;
    let g = &t.grid;
    let input = t.kb.input();
    let base = reference_config(&t.partition);
    let pp_ref = solve_kb_atom(g, &input, &base, SpinMode::Restricted).map_err(|e| format!("reference KB atom: {e}"))?;
    // T1 (a regression: the reference is an exact eigenstate of the KB Hamiltonian)
    let t1 = t.partition.channels.iter().filter(|c| c.occupied).map(|c| (pp_ref.eps[c.l as usize][0] - c.eps).abs()).fold(0.0, f64::max);
    if t1 >= gates.t1_eig { return Err(format!("T1: {:.3} mHa", t1 * 1e3)); }
    // T2 (norm conservation, TM's construction, as a regression)
    let t2 = t.channels.iter().zip(&t.partition.channels).map(|(tm, c)| {
        let sq = |u: &[f64]| u.iter().map(|x| x * x).collect::<Vec<_>>();
        (g.partial(&sq(&tm.u), tm.i_c) - g.partial(&sq(&c.u), tm.i_c)).abs()
    }).fold(0.0, f64::max);
    if t2 >= gates.t2_norm { return Err(format!("T2: norm error {t2:.2e}")); }
    // T3 / T4
    let hund = ground_state(t.z, SpinMode::Polarised).map_err(|e| format!("Hund atom: {e}"))?;
    let (mut t3, mut t4, mut skipped) = (Vec::new(), 0.0f64, Vec::new());
    for cfg in transfer_configs(t, &hund) {
        let ae = if cfg.name == "Hund" { Ok(hund.clone()) } else { solve_atom(t.z, &cfg.ae) };
        let pp = solve_kb_atom(g, &input, &cfg.pp, cfg.spin);
        let bound = |ae: &AtomResult| ae.orbitals.iter().all(|o| (0..2).all(|s| o.occ[s] == 0.0 || o.eps[s] < 0.0));
        match (ae, pp) {
            (Ok(ae), Ok(pp)) if bound(&ae) && pp.eps.iter().zip(&cfg.pp.occ).all(|(e, o)| (0..2).all(|s| o[s] == 0.0 || e[s] < 0.0)) => {
                let de_ae = ae.energy.total - t.reference.energy.total;
                let de_pp = pp.energy - pp_ref.energy;
                let mut worst = 0.0f64;
                for o in ae.orbitals.iter().filter(|o| t.partition.valence.iter().any(|v| v.n == o.n && v.l == o.l)) {
                    for s in 0..2 { if o.occ[s] > 0.0 { worst = worst.max((o.eps[s] - pp.eps[o.l as usize][s]).abs()); } }
                }
                if (de_ae - de_pp).abs() >= gates.t3_delta_e { return Err(format!("T3 {}: ΔE_AE {de_ae:.5} vs ΔE_PP {de_pp:.5}", cfg.name)); }
                if worst >= gates.t4_eig { return Err(format!("T4 {}: eigenvalue error {:.3} mHa", cfg.name, worst * 1e3)); }
                t4 = t4.max(worst);
                t3.push(TransferRow { name: cfg.name.to_string(), de_ae, de_pp, max_eig_error: worst });
            }
            (ae, pp) if cfg.name == "anion" => skipped.push(format!("anion: unbound in LSDA (AE {}, PP {})", ae.is_ok(), pp.is_ok())),
            (ae, pp) => return Err(format!("T3 {}: AE {:?} / PP {:?}", cfg.name, ae.err(), pp.err())),
        }
    }
    // T5: projector channels and the local channel (never an ungenerated l)
    let e_min = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::INFINITY, f64::min) - gates.t5_window;
    let e_max = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.eps).fold(f64::NEG_INFINITY, f64::max) + gates.t5_window;
    let mut t5 = Vec::new();
    for c in &t.partition.channels {
        let steps = ((e_max - e_min) / gates.t5_mesh).round() as usize;
        let worst = (0..=steps).map(|k| delta_e_equiv(t, c.l, e_min + k as f64 * gates.t5_mesh).abs()).fold(0.0, f64::max);
        if worst >= gates.t5_de_equiv { return Err(format!("T5 l = {}: max δE_equiv {:.2} mHa", c.l, worst * 1e3)); }
        t5.push((c.l, worst));
    }
    // T6: ∂ε_l/∂f_l′ by ±t6_dq (restricted), AE vs PP
    let mut t6 = Vec::new();
    let occupied: Vec<u8> = t.partition.channels.iter().filter(|c| c.occupied).map(|c| c.l).collect();
    for &lp in &occupied {
        let shifted = |dq: f64| { let mut c = base.clone(); c.occ[lp as usize][0] += 0.5 * dq; c.occ[lp as usize][1] += 0.5 * dq; c };
        // a full shell cannot take +dq (Pauli): one-sided difference over [f − 2dq, f] there
        let full = base.occ[lp as usize][0] + base.occ[lp as usize][1] + gates.t6_dq > 2.0 * (2 * lp as usize + 1) as f64 + 1e-12;
        let (hi, lo) = if full { (0.0, -2.0 * gates.t6_dq) } else { (gates.t6_dq, -gates.t6_dq) };
        let (cp, cm) = (shifted(hi), shifted(lo));
        let ae_p = solve_atom(t.z, &AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &cp))).map_err(|e| format!("T6: {e}"))?;
        let ae_m = solve_atom(t.z, &AtomOptions::fixed(SpinMode::Restricted, ae_shells(t, &cm))).map_err(|e| format!("T6: {e}"))?;
        let pp_p = solve_kb_atom(g, &input, &cp, SpinMode::Restricted).map_err(|e| format!("T6: {e}"))?;
        let pp_m = solve_kb_atom(g, &input, &cm, SpinMode::Restricted).map_err(|e| format!("T6: {e}"))?;
        for &l in &occupied {
            let n = t.partition.channels.iter().find(|c| c.l == l).unwrap().n;
            let ae_eps = |a: &AtomResult| a.orbitals.iter().find(|o| o.n == n && o.l == l).unwrap().eps[0];
            let slope_ae = (ae_eps(&ae_p) - ae_eps(&ae_m)) / (hi - lo);
            let slope_pp = (pp_p.eps[l as usize][0] - pp_m.eps[l as usize][0]) / (hi - lo);
            if (slope_ae - slope_pp).abs() >= gates.t6_hardness { return Err(format!("T6 ∂ε_{l}/∂f_{lp}: AE {slope_ae:.4} PP {slope_pp:.4}")); }
            t6.push((l, lp, slope_ae, slope_pp));
        }
    }
    Ok(GateReport { t1, t2, t3, t4, t5, t6, skipped })
}
```

- [ ] **Step 5: Implement the record, the hash, and the ladder changes**

`crates/qm/src/radial/record.rs`:
```rust
use super::gates::GateReport;
use super::ledger::recipe_hash;
use super::partition::Refusal;
use super::recipe::PseudoTable;

#[derive(Clone, Debug, PartialEq)]
pub struct FilterRow { pub h: f64, pub de: f64, pub max_eig: f64, pub t3_max: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRecord {
    pub z: u8, pub recipe_hash: u64, pub f: f64, pub local: String, pub r_c: Vec<(u8, f64)>, pub z_v: f64, pub n_core: u32,
    pub e_kb: Vec<(u8, f64)>, pub gss: Vec<(u8, f64)>, pub r_nlc: Option<f64>, pub r_cmp: f64, pub r_core: f64,
    pub gates: GateReport, pub h_req: Option<f64>, pub filter: Vec<FilterRow>, pub dimer: String, pub seconds: f64,
}

pub fn record_of(t: &PseudoTable, seconds: f64) -> GenerationRecord {
    let lch = |l: u8| ['s', 'p', 'd', 'f'][l as usize];
    GenerationRecord {
        z: t.z, recipe_hash: recipe_hash(), f: t.f,
        local: match t.kb.local_l { Some(l) => format!("{}-local", lch(l)), None => "polynomial".into() },
        r_c: t.channels.iter().map(|c| (c.l, c.r_c)).collect(), z_v: t.z_v, n_core: t.n_core,
        e_kb: t.kb.projectors.iter().map(|p| (p.l, p.e_kb)).collect(), gss: t.kb.projectors.iter().map(|p| (p.l, p.gss)).collect(),
        r_nlc: t.partial_core.r_nlc, r_cmp: t.kb.r_cmp, r_core: t.partial_core.r_core,
        gates: t.gates.clone(), h_req: None, filter: Vec::new(), dimer: "not applicable".into(), seconds,
    }
}

/// The pinned outputs table (review S6a): one row per Z; refusals name their reason. No timings
/// (reported elsewhere, never pinned).
pub fn records_table(rows: &[(u8, Result<GenerationRecord, Refusal>)]) -> String {
    let lch = |l: u8| ['s', 'p', 'd', 'f'][l as usize];
    let list = |v: &[(u8, f64)], d: usize| v.iter().map(|(l, x)| format!("{}: {x:.d$}", lch(*l))).collect::<Vec<_>>().join(", ");
    let mut s = format!("recipe_hash = 0x{:016x} ({})\n\n", recipe_hash(), super::LEDGER.version);
    s += "| Z | el | f | local | r_c (Bohr) | Z_v | E_KB (Ha) | GSS (Ha) | r_nlc | r_core | h_req |\n|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (z, r) in rows {
        let el = units::elements::symbol(*z);
        match r {
            Ok(g) => s += &format!("| {z} | {el} | {:.1} | {} | {} | {:.0} | {} | {} | {} | {:.3} | {} |\n",
                g.f, g.local, list(&g.r_c, 3), g.z_v, list(&g.e_kb, 4), list(&g.gss, 4),
                g.r_nlc.map_or("—".into(), |r| format!("{r:.3}")), g.r_core, g.h_req.map_or("—".into(), |h| format!("{h:.2}"))),
            Err(e) => s += &format!("| {z} | {el} | refused | {e} | | | | | | | |\n"),
        }
    }
    s
}
```
In `ledger.rs`:
```rust
/// FNV-1a 64 over the table's `Debug` rendering (f64 `Debug` is the shortest round-trip decimal, so
/// the rendering — and the hash — is stable across builds and platforms).
pub fn hash_ledger(l: &Ledger) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in format!("{l:?}").bytes() { h ^= b as u64; h = h.wrapping_mul(0x0100_0000_01b3); }
    h
}
pub fn recipe_hash() -> u64 { hash_ledger(&LEDGER) }
```
In `recipe.rs`:
- `generate` builds the partial core: `let pc = partial_core(&reference.grid, &core_density(&reference, &part), &valence_density(&reference, &part));`
- it times the whole generation (`std::time::Instant`) and passes `&pc` to `candidate`.
- `candidate(…, pc: &PartialCore)` passes `pc.rho_pc` to `unscreen` and `kb_form`. After `ghost_gates` it runs `let gates = run_gates(&table)?;`, builds the table with `partial_core: pc.clone()` and `gates`, and fills `record = record_of(&t, seconds)` on the accepted entry.

Add `pub mod nlcc; pub mod gates; pub mod record;` and the re-exports to `mod.rs`. Add `/crates/validate/last-run-generation-records.md` to `.gitignore`.

- [ ] **Step 6: Run the tests**

Run: `cargo test -p qm --release radial`
Expected: PASS.

Run: `cargo test -p validate --release --test pseudo_gen`
Expected: `be_to_ne_pass_every_gate_and_li_is_refused` PASS. `generation_record_table_matches_references_md` FAILS on the first run (no block yet) and writes `last-run-generation-records.md`.

If an element misses a gate at every ladder entry and is refused, or O fails T5 at f = 1.6, follow the spec §4.8 last paragraph: **report and have the physics reviewer re-examine**. Never adjust a threshold.

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

### Task 5: `filter` — the spherical-Bessel filter, the V_cmp/V_tail split, h_req and the filtered re-run **[Fable — physics-review gate]**

Spec §5.1, §4.8 T8, §8 row 5; review S2, S3, S7, M4. **Spec questions 2, 3 and 11 apply, and the physics reviewer must confirm Spec questions 2–3 before this task starts.** The spec's V_cmp = V_loc + Z_v/d is singular at the origin; Spec question 2 gives the smooth split this task implements. Spec question 3 makes "King-Smith's smooth cutoff" concrete.

**Files:**
- Create: `crates/qm/src/radial/filter.rs`
- Modify: `crates/qm/src/radial/recipe.rs` (h_req and the filter rows in the record), `crates/qm/src/radial/record.rs` (no change of type; the table shows h_req), `crates/qm/src/radial/ledger.rs` (add `("filter.rs", include_str!("filter.rs"))` to `gate_thresholds_are_constants`), `crates/qm/src/radial/mod.rs`, `crates/validate/tests/pseudo_gen.rs` (h_req printout), `crates/validate/references.md` (regenerated records block; section "M7 filter (Task 5)")

**Interfaces:**
- Consumes: Tasks 1–4 (`PseudoTable`, `KbInput`, `solve_kb_atom`, `reference_config`, `transfer_configs`, `GenerationRecord`, `FilterRow`, `LEDGER.{q_c_over_nyquist, gamma_r, sigma_lr_over_h, v_tail_floor, h_ladder, gates.h_req_tol, gates.t1_eig, gates.t3_delta_e}`).
- Produces (Tasks 6–8 and 11 use exactly these):
```rust
pub fn sph_bessel(l: u8, x: f64) -> f64;
pub fn sph_bessel_scaled(l: u8, x: f64) -> f64;                 // j_l(x)/x^l, finite at 0
pub fn mask(r: f64, r_in: f64, r_out: f64) -> f64;              // C^∞ step: 1 → 0 on [r_in, r_out]
pub fn mask_deriv(r: f64, r_in: f64, r_out: f64) -> f64;
pub fn filter_radial(grid: &LogGrid, f: &[f64], l: u8, q_c: f64, r_in: f64, r_out: f64) -> Vec<f64>; // returns f^filt/r^l
pub struct FilteredProjector { pub l: u8, pub e_kb: f64, pub t: Vec<f64> /* F_l^filt(r)/r^l */, pub radius: f64 }
pub struct FilteredTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub h: f64, pub sigma_lr: f64,
    pub v_cmp: Vec<f64>, pub cmp_radius: f64, pub tail_in: f64, pub tail_out: f64, pub tail_radius: f64,
    pub projectors: Vec<FilteredProjector>, pub rho_pc: Vec<f64>, pub pc_radius: f64, pub r_core: f64,
    pub check: FilterRow /* the filtered T1/T3 re-run at this h (S2) */, pub record: GenerationRecord, pub grid: LogGrid }
impl FilteredTable {
    pub fn v_lr(&self, d: f64) -> f64;   pub fn dv_lr(&self, d: f64) -> f64;      // −Z_v erf(d/(σ_LR√2))/d
    pub fn v_tail(&self, d: f64) -> f64; pub fn dv_tail(&self, d: f64) -> f64;
    pub fn v_sr(&self, d: f64) -> f64;   pub fn dv_sr(&self, d: f64) -> f64;      // V_cmp^filt + V_tail
    pub fn sr_radius(&self) -> f64;
    pub fn beta_radial(&self, k: usize, d: f64) -> f64; pub fn dbeta_radial(&self, k: usize, d: f64) -> f64; // t_k and dt_k/dd
    pub fn rho_pc_at(&self, d: f64) -> f64; pub fn drho_pc_at(&self, d: f64) -> f64;
    pub fn input(&self) -> KbInput;                                               // the radial KB atom on the filtered tables
    pub fn unfiltered(table: &PseudoTable, h: f64) -> FilteredTable;              // the identity filter (Task 6's S1 test)
}
pub fn filtered(table: &PseudoTable, h: f64) -> FilteredTable;
pub fn h_req(table: &PseudoTable) -> (Option<f64>, Vec<FilterRow>);
```
**The split** (Spec question 2) uses a = σ_LR√2, σ_LR = `LEDGER.sigma_lr_over_h`·h, r_c = max_l r_c,l and γ = `LEDGER.gamma_r`.
- The short-range part V_SR(d) = V_loc(d) + Z_v erf(d/a)/d is smooth and finite at 0.
- The tail is V_tail(d) = (1 − m(d; r_c/γ, r_c))·(−Z_v erfc(d/a)/d). It is analytic, zero near the origin and never masked outward; it is sampled to where |V_tail| < `LEDGER.v_tail_floor`.
- The compact part is V_cmp = V_SR − V_tail. It is smooth and vanishes beyond max(r_c, r_cmp). It is filtered (hard cut at q_c = `LEDGER.q_c_over_nyquist`·π/h) and masked by m(d; R, γR) with R = max(r_c, r_cmp).

The projector radial part is F_l = δV_l u_l/r. It is filtered with j_l and masked over [R_l, γR_l], with R_l its support radius (where |β| falls below 1e-12 of its maximum). The table stores t_l = F_l^filt/r^l, so the grid function is β_lm(r) = t_l(d)·√((2l+1)/4π)·R_lm(d), exactly the spec §4.6 S1 form. ρ_pc is filtered with j_0 and masked over [r_c, γr_c].

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

fn shifts(h: f64) -> Vec<[f64; 3]> {
    let q = h / 4.0;
    let mut v = Vec::new();
    for a in [-q, q] { for b in [-q, q] { for c in [-q, q] { v.push([a, b, c]); } } }
    v
}

/// Spec §8 row 5 (review S7): for each filtered table g, h³Σ_p g(r_p − R) and h³Σ_p g(r_p − R)R_lm(r_p − R)
/// are invariant in R to < 1e-8 relative over the 8 sub-grid shifts (Poisson summation). For a
/// projector the invariant tested is the plain sum (l = 0) and the norm h³Σ β_lm² (every l, m).
#[test]
fn filtered_lattice_sums_are_translation_invariant() {
    for z in [6u8, 8] {
        let t = generate(z).unwrap();
        let h = 0.25;
        let ft = filtered(&t, h);
        let norm3 = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let spread = |vals: &[f64], scale: f64| (vals.iter().cloned().fold(f64::MIN, f64::max) - vals.iter().cloned().fold(f64::MAX, f64::min)) / scale;
        let check = |what: &str, radius: f64, g: &dyn Fn([f64; 3]) -> f64, scale_of: &dyn Fn(&[f64]) -> f64| {
            let vals: Vec<f64> = shifts(h).into_iter().map(|s| lattice_sum(h, radius, s, g)).collect();
            let rel = spread(&vals, scale_of(&vals));
            assert!(rel < 1e-8, "Z = {z} {what}: relative spread {rel:.2e} over the 8 shifts ({vals:?})");
        };
        let mean_abs = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64;
        check("V_cmp", ft.cmp_radius, &|d| { let r = norm3(d); if r <= ft.cmp_radius { ft.grid.interp(&ft.v_cmp, r) } else { 0.0 } }, &mean_abs);
        check("ρ_pc", ft.pc_radius, &|d| ft.rho_pc_at(norm3(d)), &mean_abs);
        for (k, p) in ft.projectors.iter().enumerate() {
            let nrm = ((2 * p.l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
            for m in -(p.l as i32)..=(p.l as i32) {
                let beta = |d: [f64; 3]| { let mut sh = [0.0; N_MOMENTS]; solid_harmonics(d, &mut sh); ft.beta_radial(k, norm3(d)) * nrm * sh[lm_index(p.l as usize, m)] };
                if p.l == 0 { check("β sum", p.radius, &beta, &mean_abs); }
                check("β norm", p.radius, &|d| beta(d).powi(2), &mean_abs);
            }
        }
    }
}

/// Review S3: the filtered partial core stays ≥ −1e-8 (a ringing King-Smith cut could go negative;
/// `qm::xc::exc_point` clamps any residue at 0 — documented there).
#[test]
fn filtered_partial_core_is_nonnegative() {
    for z in [4u8, 6, 8, 10] {
        let t = generate(z).unwrap();
        for &h in LEDGER.h_ladder.iter() {
            let ft = filtered(&t, h);
            let min = ft.rho_pc.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(min >= -1e-8, "Z = {z}, h = {h}: min ρ_pc^filt = {min:.3e}");
        }
    }
}

/// Spec §8 row 5 / §4.8 T8 (review S2): at h_req the filtered radial KB atom reproduces the
/// unfiltered total energy and every valence eigenvalue within 1 mHa (and h_req is on the ladder).
#[test]
fn filtered_kb_atom_within_1mha_at_h_req() {
    for z in [6u8, 7, 8, 9, 10] {
        let t = generate(z).unwrap();
        let h = t.record.h_req.unwrap_or_else(|| panic!("Z = {z}: no h on the ladder passes"));
        assert!(LEDGER.h_ladder.contains(&h));
        let ft = filtered(&t, h);
        assert!(ft.check.de.abs() < 1e-3 && ft.check.max_eig < 1e-3, "Z = {z} at h_req = {h}: ΔE {:.3e}, max Δε {:.3e}", ft.check.de, ft.check.max_eig);
        assert!(ft.check.t3_max < 5e-3, "Z = {z}: filtered T3 {:.3e}", ft.check.t3_max);
    }
}

#[test]
fn bessel_functions_and_transform_pair() {
    // closed forms vs series across x = 1
    for l in 0..=3u8 { for &x in &[0.999, 1.001, 0.3, 5.0] {
        let s = sph_bessel_scaled(l, x) * x.powi(l as i32);
        assert!((s - sph_bessel(l, x)).abs() < 1e-12, "l = {l}, x = {x}");
    }}
    // a Gaussian is band-limited enough that a cut far beyond its spectrum returns it
    let g = LogGrid::universal();
    let f: Vec<f64> = g.r.iter().map(|r| (-r * r).exp()).collect();
    let back = filter_radial(&g, &f, 0, 40.0, 50.0, 55.0);
    for &r in &[0.0, 0.5, 1.5] { let i = g.nearest(r.max(1e-5)); assert!((back[i] - f[i]).abs() < 1e-9); }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release radial::filter`
Expected: compile errors.

- [ ] **Step 3: Implement the filter**

`crates/qm/src/radial/filter.rs`. Its module doc carries:
- spec §5.1 and the Spec question 2 split, with its derivation: V_SR is smooth; V_tail vanishes near 0; V_cmp is compact beyond max(r_c, r_cmp) because V_SR = tail there to `v_ion_tail`;
- Poisson summation (a function band-limited below 2π/h has translation-invariant lattice sums);
- the mask (C^∞, Gevrey-class step ψ(1−t)/(ψ(t)+ψ(1−t)), ψ(t) = e^{−1/t}: no parameter);
- the q mesh (uniform, dq ≤ π/r_max: derived from the ledger's r_max, Simpson);
- S2/S3.

```rust
use super::gates::transfer_configs;
use super::grid::LogGrid;
use super::kb::{reference_config, solve_kb_atom, KbInput};
use super::ledger::LEDGER;
use super::record::{FilterRow, GenerationRecord};
use super::recipe::PseudoTable;
use super::atom::{ground_state, solve_atom, SpinMode};
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

pub fn mask_deriv(r: f64, r_in: f64, r_out: f64) -> f64 {
    if r <= r_in || r >= r_out { return 0.0; }
    let t = (r - r_in) / (r_out - r_in);
    let (a, b) = (psi(t), psi(1.0 - t));
    let (da, db) = (a / (t * t), b / ((1.0 - t) * (1.0 - t)));   // ψ′(t) = ψ(t)/t²
    -(db * a + b * da) / ((a + b) * (a + b)) / (r_out - r_in)
}

fn q_mesh(q_c: f64) -> Vec<f64> {
    let n = 2 * ((q_c * LEDGER.r_max / (2.0 * PI)).ceil() as usize).max(1);
    (0..=n).map(|k| q_c * k as f64 / n as f64).collect()
}

/// F → hard cut at q_c → back → × mask(r_in, r_out); returns F^filt/r^l (finite at 0).
pub fn filter_radial(grid: &LogGrid, f: &[f64], l: u8, q_c: f64, r_in: f64, r_out: f64) -> Vec<f64> {
    let n = grid.len();
    let q = q_mesh(q_c);
    let ft: Vec<f64> = q.iter().map(|&qk| 4.0 * PI * grid.integrate(&(0..n).map(|i| f[i] * sph_bessel(l, qk * grid.r[i]) * grid.r[i] * grid.r[i]).collect::<Vec<_>>())).collect();
    let dq = q[1] - q[0];
    let simpson = |k: usize| if k == 0 || k == q.len() - 1 { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
    (0..n).map(|i| {
        let r = grid.r[i];
        let s: f64 = (0..q.len()).map(|k| simpson(k) * ft[k] * q[k].powi(l as i32 + 2) * sph_bessel_scaled(l, q[k] * r)).sum();
        s * dq / 3.0 / (2.0 * PI * PI) * mask(r, r_in, r_out)
    }).collect()
}

#[derive(Clone, Debug)]
pub struct FilteredProjector { pub l: u8, pub e_kb: f64, pub t: Vec<f64>, pub radius: f64 }

#[derive(Clone, Debug)]
pub struct FilteredTable {
    pub z: u8, pub z_v: f64, pub n_core: u32, pub h: f64, pub sigma_lr: f64,
    pub v_cmp: Vec<f64>, pub cmp_radius: f64, pub tail_in: f64, pub tail_out: f64, pub tail_radius: f64,
    pub projectors: Vec<FilteredProjector>, pub rho_pc: Vec<f64>, pub pc_radius: f64, pub r_core: f64,
    pub check: FilterRow, pub record: GenerationRecord, pub grid: LogGrid,
}

fn erfc_tail(z_v: f64, a: f64, d: f64) -> f64 { -z_v * libm::erfc(d / a) / d }
fn derfc_tail(z_v: f64, a: f64, d: f64) -> f64 {
    z_v * (2.0 / (PI.sqrt() * a * d) * (-(d / a).powi(2)).exp() + libm::erfc(d / a) / (d * d))
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
    pub fn v_tail(&self, d: f64) -> f64 {
        if d <= self.tail_in || d > self.tail_radius { 0.0 } else { (1.0 - mask(d, self.tail_in, self.tail_out)) * erfc_tail(self.z_v, self.a(), d) }
    }
    pub fn dv_tail(&self, d: f64) -> f64 {
        if d <= self.tail_in || d > self.tail_radius { return 0.0; }
        let m = mask(d, self.tail_in, self.tail_out);
        -mask_deriv(d, self.tail_in, self.tail_out) * erfc_tail(self.z_v, self.a(), d) + (1.0 - m) * derfc_tail(self.z_v, self.a(), d)
    }
    pub fn v_sr(&self, d: f64) -> f64 {
        (if d <= self.cmp_radius { self.grid.interp(&self.v_cmp, d) } else { 0.0 }) + self.v_tail(d)
    }
    pub fn dv_sr(&self, d: f64) -> f64 {
        (if d <= self.cmp_radius { self.grid.interp_deriv(&self.v_cmp, d) } else { 0.0 }) + self.dv_tail(d)
    }
    pub fn sr_radius(&self) -> f64 { self.cmp_radius.max(self.tail_radius) }
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
    let q_c = LEDGER.q_c_over_nyquist * PI / h;
    let r_c = t.channels.iter().map(|c| c.r_c).fold(0.0, f64::max);
    let (tail_in, tail_out) = (r_c / gamma, r_c);
    // V_SR (smooth), V_tail (analytic, switched on over [r_c/γ, r_c]), V_cmp = V_SR − V_tail (compact)
    let v_sr: Vec<f64> = g.r.iter().zip(&t.kb.v_loc).map(|(&d, &v)| v + t.z_v * libm::erf(d / a) / d).collect();
    let v_tail: Vec<f64> = g.r.iter().map(|&d| if d <= tail_in { 0.0 } else { (1.0 - mask(d, tail_in, tail_out)) * erfc_tail(t.z_v, a, d) }).collect();
    let raw_cmp: Vec<f64> = (0..n).map(|i| v_sr[i] - v_tail[i]).collect();
    let big_r = r_c.max(t.kb.r_cmp);
    let support = |f: &[f64]| { let m = f.iter().fold(0.0f64, |x, y| x.max(y.abs())); g.r[f.iter().rposition(|y| y.abs() > 1e-12 * m).unwrap_or(0)] };
    let (v_cmp, cmp_radius) = if filter { (filter_radial(g, &raw_cmp, 0, q_c, big_r, gamma * big_r), gamma * big_r) } else { (raw_cmp.clone(), support(&raw_cmp)) };
    let tail_radius = g.r.iter().copied().rev().find(|&d| (1.0 - mask(d, tail_in, tail_out)) * erfc_tail(t.z_v, a, d).abs() >= LEDGER.v_tail_floor).unwrap_or(tail_out);
    let projectors = t.kb.projectors.iter().map(|p| {
        let f: Vec<f64> = (0..n).map(|i| p.beta[i] / g.r[i]).collect();   // F_l = δV·u/r = β_u/r (the 3-D radial part)
        let r_l = support(&p.beta);
        let tab = if filter { filter_radial(g, &f, p.l, q_c, r_l, gamma * r_l) } else { (0..n).map(|i| f[i] / g.r[i].powi(p.l as i32)).collect() };
        FilteredProjector { l: p.l, e_kb: p.e_kb, t: tab, radius: if filter { gamma * r_l } else { r_l } }
    }).collect();
    let pc_radius = if filter { gamma * r_c } else { support(&t.partial_core.rho_pc).max(g.r[0]) };
    let rho_pc = if filter { filter_radial(g, &t.partial_core.rho_pc, 0, q_c, r_c, gamma * r_c) } else { t.partial_core.rho_pc.clone() };
    let mut ft = FilteredTable {
        z: t.z, z_v: t.z_v, n_core: t.n_core, h, sigma_lr, v_cmp, cmp_radius, tail_in, tail_out, tail_radius,
        projectors, rho_pc, pc_radius, r_core: t.partial_core.r_core,
        check: FilterRow { h, de: 0.0, max_eig: 0.0, t3_max: 0.0 }, record: t.record.clone(), grid: g.clone(),
    };
    ft.check = filtered_rerun(t, &ft);
    ft
}
```
The u-form ↔ 3-D convention: F_l = β_u/r, t_l = F_l/r^l, and `input()` inverts it as β_u = r^{l+1}·t_l. `unfiltered_input_reproduces_the_kb_input` (below) pins the round trip.

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
(The exact split makes V_LR + V_cmp + V_tail = V_loc for the unfiltered table, and β_u round-trips.)

- [ ] **Step 4: Run the tests**

Run: `cargo test -p qm --release radial`
Expected: PASS.

If `filtered_lattice_sums_are_translation_invariant` misses 1e-8, report the spread per table and h, and stop. The mask's spectral leakage is the expected cause. The remedy is the physics reviewer's call: a smoother mask is a ledger change; the spec's plan B, the Ono–Hirose double grid, belongs to Task 6. It is never a looser bound.

- [ ] **Step 5: Regenerate and pin the records.** Run `cargo test -p validate --release --test pseudo_gen`. It fails, because the h_req column moved, and writes `last-run-generation-records.md`. After physics review, copy that block into `references.md` and add a section "M7 filter (Task 5)" with each element's filter ΔE per ladder h (`record.filter`), h_req, and min ρ_pc^filt. Re-run: PASS.

- [ ] **Step 6: Commit**
```bash
git add crates/qm/src/radial crates/validate/tests/pseudo_gen.rs crates/validate/references.md
git commit -o -m "feat(qm): spherical-Bessel filter, V_cmp/V_tail split, h_req and the filtered T1/T3 re-run (M7 Task 5)" -- crates/qm/src/radial crates/validate/tests/pseudo_gen.rs crates/validate/references.md
```
Before merging, hand the diff to `physics-reviewer` with Spec questions 2, 3 and 11, the lattice-sum spreads and the filter table.

---

### Task 6: `qm::ionic` + SCF integration **[Fable — physics-review gate]**

Spec §4.7, §5.1–§5.3, §5.5 (`qm` bullet), §5.6, §6.3, §8 row 6; review S1, S7, M3, and the NOTE on the no-core branch. **Spec question 6 applies:** V_LR is the analytic erf, as `nuclear_potential` and `nuclear.wgsl` already compute it.

**Files:**
- Create: `crates/qm/src/ionic.rs`, `crates/qm/src/nonlocal.rs`, `crates/validate/tests/pseudo.rs`
- Modify: `crates/qm/src/lib.rs`, `crates/qm/src/nuclear.rs` (centres, the ionic local potential, the partial-core field, ion–ion repulsion, the sphere point lists), `crates/qm/src/scf.rs` (`System.ionic`, `System::with_ionic`, `valence_electrons`, `KsState.ionic`, `ScfStats.core_overlaps`, `IonicTerms`, `solve_channel` with V_NL, the warm-start guard), `crates/qm-gpu/src/scf.rs` (the `ScfStats` literal gains `core_overlaps: Vec::new()`; nothing else in `qm-gpu` until Task 8), `crates/validate/references.md` (section "M7 grid integration (Task 6)")

**Interfaces:**
- Consumes: Task 5's `FilteredTable` (`v_lr`, `v_sr`, `sr_radius`, `beta_radial`, `rho_pc_at`, `pc_radius`, `projectors`, `r_core`, `record.h_req`, `check`, `h`, `z_v`, `sigma_lr`), `filtered`, `generate`, `has_core`, `Refusal`, `recipe_hash`, `LEDGER`, and `qm::poisson::{solid_harmonics, lm_index, N_MOMENTS}` (Racah-normalised, unchanged).
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
/// pseudo-orbital φ_l0 = (u_l/d)·Y_l0 equals δV_l φ_l0 pointwise (< 1e-6), and ⟨φ|V_NL|φ⟩ equals
/// ⟨φ|δV|φ⟩ — which catches a missing √((2l+1)/4π) (a factor 4π/(2l+1) in V_NL). Unfiltered table on
/// a fine grid (h = 0.1) so the quadrature is not what is measured; O off a grid point.
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
        assert!(worst < 1e-6, "l = {}: max |V_NL φ − δV φ| = {worst:.3e}", p.l);
        let (lhs, rhs) = (grid.dot(&phi, &got), grid.dot(&phi, &expect));
        assert!((lhs - rhs).abs() < 1e-6 * rhs.abs(), "l = {}: ⟨φ|V_NL|φ⟩ {lhs:.9} vs ⟨φ|δV|φ⟩ {rhs:.9}", p.l);
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

/// Review Focus 3: an ion exactly on a grid point gives finite fields and a converged SCF.
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
    let (_, stats) = scf(&g, &sys, &ScfSettings::for_grid(&g), None).expect("converges");
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

/// Spec §8 row 6 (moved from Task 5 by review S7): the O atom's rigid-shift energy ripple on the
/// grid after the filter — 8 sub-grid shifts of ±h/4; reported (recorded in references.md, Task 10).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn o_atom_eggbox_on_grid_after_filter() {
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let q = h / 4.0;
    let mut es = Vec::new();
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let (e, _, _) = solve(h, 12.0, IonicPolicy::CoresOnly, vec![Nucleus { pos: [sx, sy, sz], z: 8 }], Some(3));
        es.push(e);
    }}}
    let mean = es.iter().sum::<f64>() / 8.0;
    let rms = (es.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    eprintln!("pseudo-O egg-box at h = {h}: E rms {rms:.2e} Ha, spread {:.2e} Ha over 8 shifts", es.iter().cloned().fold(f64::MIN, f64::max) - es.iter().cloned().fold(f64::MAX, f64::min));
    assert!(rms.is_finite());
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
    match t.record.h_req {
        Some(hr) if h <= hr + 1e-12 => {}
        other => return Err(IonicError::TooCoarse { z, h, h_req: other }),
    }
    let c = &t.check;
    if c.de.abs() >= LEDGER.gates.h_req_tol || c.max_eig >= LEDGER.gates.t1_eig || c.t3_max >= LEDGER.gates.t3_delta_e {
        return Err(IonicError::FilteredGate { z, h, why: format!("ΔE {:.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa", c.de * 1e3, c.max_eig * 1e3, c.t3_max * 1e3) });
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

In `crates/qm/src/nuclear.rs` (the module doc gains a section "Ionic species (M7)": V_loc = Σ_I [−q_I erf(d/(σ_I√2))/d] + Σ_pseudo V_SR, where q_I = Z_v and σ_I = σ_LR for pseudised species and q_I = Z, σ_I = σ_AE for all-electron ones; the exact split; Spec question 6):
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

---

### Task 7: forces — local (LR + V_cmp + V_tail), nonlocal and NLCC **[Fable — physics-review gate]**

Spec §5.4, §6.6, §5.6, §8 row 7; review M4 (the same split in the force), M5 (where the 3e-5 gate is stated) and the NOTE re-deriving the three force terms.

**Files:**
- Modify: `crates/qm/src/forces.rs` (dispatch on `state.ionic`; the pseudised terms), `crates/qm/src/nonlocal.rs` (gradients: `solid_harmonic_gradients`, `NonlocalOp::build_with_gradients`, `Sphere.grad`), `crates/validate/tests/pseudo.rs` (three tests), `crates/validate/references.md` (section "M7 forces (Task 7)")

**Interfaces:**
- Consumes: Task 6's `IonicSetup`, `Ionic`, `NonlocalOp`, `Sphere`, `points_within`, `partial_core_field`, `KsState.ionic`, `ScfStats.core_overlaps`, `CORE_OVERLAP_WARNING`, and Task 5's `FilteredTable::{dv_lr, dv_sr, sr_radius, dbeta_radial, beta_radial, drho_pc_at, pc_radius}`.
- Produces:
```rust
pub fn forces(grid: &Grid, state: &KsState, nuclei: &[Nucleus], sigma: f64) -> Result<Vec<[f64; 3]>, QmError>; // signature unchanged
pub fn solid_harmonic_gradients(r: [f64; 3]) -> [[f64; 3]; 9];      // ∇R_lm, l ≤ 2, index lm_index(l, m)
impl NonlocalOp { pub fn build_with_gradients(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> NonlocalOp; }
pub struct Sphere { /* … */ pub grad: Vec<Vec<[f64; 3]>> /* [m][point]; empty unless built with gradients */ }
```
**The force terms** (spec §5.4, with forces.rs's sign chain ∂/∂R_I = −∇_r). For a pseudised ion I:
- F^LR = ∫ρ ∇V_LR, today's integrand with Z → Z_v and σ → σ_LR.
- F^SR = ∫ρ (dV_SR/dd)(r − R_I)/d over the SR sphere, where dV_SR/dd is the derivative of the sampled interpolant of the filtered V_cmp plus the analytic d/dd of V_tail.
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
In `crates/validate/tests/pseudo.rs`:
```rust
/// Spec §8 row 7 first failing test (and Review Focus 4: run under CoresOnly, the mixed scene — an
/// all-electron H at σ = h beside a pseudo-O at σ_LR = 2h): Hellmann–Feynman vs central finite
/// differences of E at δ = 0.01 within forces.rs's 2e-3 bound, on both nuclei; δ = 0.02 too, and the
/// Richardson-separated estimate (4D(δ) − D(2δ))/3 reported.
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
    }
}

/// Spec §6.6 / §8 row 7 (review M5): an all-pseudo system under CoresOnly — O₂ (triplet, R = 2.28
/// Bohr) — max over the 8 rigid shifts of ±h/4 of |Σ_I F_I| < 3e-5 Ha/Bohr; the y-component (today's
/// figure of record is −3.4e-4 for H₂O) and the rms energy ripple reported. At 57³ × 0.25 Bohr (14 Bohr) when
/// O's h_req allows 0.25, else at h_req with the same 14 Bohr box (recorded which).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo")]
fn o2_max_shift_sum_of_forces_below_3e_5() {
    let h = 0.25f64.min(table(8).record.h_req.expect("O has h_req"));
    let q = h / 4.0;
    let (mut worst, mut es, mut fy) = (0.0f64, Vec::new(), Vec::new());
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let nuc = vec![Nucleus { pos: [-1.14 + sx, 0.07 + sy, 0.03 + sz], z: 8 }, Nucleus { pos: [1.14 + sx, 0.07 + sy, 0.03 + sz], z: 8 }];
        let (e, st, _) = solve(h, 14.0, IonicPolicy::CoresOnly, nuc.clone(), Some(3));
        let g = Grid::cube(n_for(14.0, h), h);
        let f = qm::forces(&g, &st, &nuc, h).unwrap();
        let sum = [f[0][0] + f[1][0], f[0][1] + f[1][1], f[0][2] + f[1][2]];
        worst = worst.max((sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt());
        es.push(e);
        fy.push(sum[1]);
    }}}
    let mean = es.iter().sum::<f64>() / 8.0;
    let rms = (es.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    eprintln!("O₂ at h = {h}: max |ΣF| {worst:.2e}, ΣF_y {fy:?}, E rms {rms:.2e}");
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
In `crates/qm/src/scf.rs` tests, extend `ion_on_a_grid_point_is_finite` with forces. An isolated atom on a grid point has every force integrand odd about R, so its force is zero to rounding. Concretely, after its SCF:
```rust
    let (st, _) = scf(&g, &sys, &ScfSettings::for_grid(&g), None).unwrap();
    let f = crate::forces(&g, &st, &sys.nuclei, g.h).unwrap();
    assert!(f[0].iter().all(|x| x.is_finite() && x.abs() < 1e-8), "{:?}", f[0]);
```
(Restructure the test so the SCF result is bound once and used for both checks.)

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p qm --release nonlocal` and `cargo test -p validate --release --test pseudo oh_hf_force_matches_fd`
Expected: the gradient test fails to compile. The force tests fail: `qm::forces` ignores `state.ionic`, so the pseudised terms are missing and HF ≠ FD.

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

In `crates/qm/src/forces.rs` (the module doc gains a section "Pseudised ions (M7)" with the four terms, their sign derivations and the no-Pulay argument):
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
    let rho = state.rho();
    let pc = partial_core_field(grid, nuclei, setup);
    let up: Vec<f64> = state.rho_up.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
    let dn: Vec<f64> = state.rho_dn.iter().zip(&pc).map(|(a, c)| a + 0.5 * c).collect();
    let xc = lsda_pw92(&up, &dn);
    let v_avg: Vec<f64> = xc.v_up.iter().zip(&xc.v_dn).map(|(a, b)| 0.5 * (a + b)).collect();
    let mut out = vec![[0.0f64; 3]; nuclei.len()];
    for (i, n) in nuclei.iter().enumerate() {
        let mut f = [0.0; 3];
        let mut add = |g: [f64; 3]| for a in 0..3 { f[a] += g[a]; };
        match setup.ionic(n.z) {
            Ionic::AllElectron { sigma } => {
                let z = n.z as f64;
                add(radial_force(grid, None, n.pos, &rho, &|d| -z * g_prime(d, *sigma)));
            }
            Ionic::Pseudo(t) => {
                add(radial_force(grid, None, n.pos, &rho, &|d| t.dv_lr(d)));
                add(radial_force(grid, Some(&points_within(grid, n.pos, t.sr_radius())), n.pos, &rho, &|d| t.dv_sr(d)));
                add(radial_force(grid, Some(&points_within(grid, n.pos, t.pc_radius)), n.pos, &v_avg, &|d| t.drho_pc_at(d)));
            }
        }
        out[i] = f;
    }
    // nonlocal: F_I += 2 Σ_ψ Σ_m E_KB c_m ⟨∇β_m|ψ⟩ (real orbitals, occupation 1 each)
    let nl = NonlocalOp::build_with_gradients(grid, nuclei, setup);
    for s in &nl.spheres {
        for psi in state.orbitals_up.iter().chain(&state.orbitals_dn) {
            for (b, gb) in s.beta.iter().zip(&s.grad) {
                let c = neumaier(s.points.iter().zip(b).map(|(&p, &bv)| bv * psi[p])) * nl.dv;
                for a in 0..3 {
                    let gpsi = neumaier(s.points.iter().zip(gb).map(|(&p, gv)| gv[a] * psi[p])) * nl.dv;
                    out[s.ion][a] += 2.0 * s.e_kb * c * gpsi;
                }
            }
        }
    }
    // ion–ion with valence charges
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
Expected: PASS. The suite takes ≈ 30 min on the CPU: the O₂ shifts are 8 × 57³-class solves, the OH finite differences 9 tight solves. Timings are reported.

Run: `cargo test -p validate --release --test dynamics`
Expected: PASS, unchanged (rule 6: forces changed).

If `o2_max_shift_sum_of_forces_below_3e_5` misses, record the eight vector sums and stop. The spec's plan B is the Ono–Hirose double grid for ⟨β|ψ⟩ (§5.1): a physics-review decision, then a separate task. The honest fallback (§7 risk 1) is to report the achieved factor. The bound is not moved.

- [ ] **Step 6: Record and commit.** Add a section "M7 forces (Task 7)" to `references.md` with:
  - OH: HF, FD(0.01), FD(0.02) and Richardson for both nuclei;
  - O₂: max |ΣF| and the y-components over the 8 shifts, the E rms, and the h used;
  - the core-overlap R and r_core.

```bash
git add crates/qm/src/forces.rs crates/qm/src/nonlocal.rs crates/qm/src/scf.rs crates/validate/tests/pseudo.rs crates/validate/references.md
git commit -o -m "feat(qm): pseudopotential forces — LR/compact/tail local, nonlocal and NLCC terms (M7 Task 7)" -- crates/qm/src/forces.rs crates/qm/src/nonlocal.rs crates/qm/src/scf.rs crates/validate/tests/pseudo.rs crates/validate/references.md
```
Before merging, hand the diff to `physics-reviewer` with the sign derivations and the OH/O₂ numbers.

---

### Task 8: GPU kernels — `local_sr`, `nlcc_deposit`, `proj_dot`/`proj_finish`/`proj_axpy`, `gather_dot3`; `gpu_vs_cpu` extended **[Opus]**

Spec §5.3, §5.5 (`qm-gpu` bullet), §6.3 (GPU vs CPU), §7 risk 6, §8 row 8. **Spec questions 6 and 7 apply:** V_LR is the analytic erf per centre, and c_lm is finished on the GPU inside H applications while energies and forces are finished in f64 on the CPU. Runs in parallel with Task 9 (different crates; Task 9 only needs `GpuKs` to accept a `System` with a setup, which this task provides).

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
- `local_sr` and `nlcc_deposit` sample the filtered **tables** on the GPU. Tables live in one storage buffer with per-ion offsets; the lookup is the same 4-point Lagrange in x = ln d as `LogGrid::interp`. Each grid point owns its value, so there are no races.
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

/// Review Focus 2 on the GPU: a parked state from another ionic setup is not a warm start.
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
    let all = System::with_ionic(nuc, 0, None, pseudo_setup(IonicPolicy::All, h, &[1]));
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

struct IonParams { p: u32, n: u32, count: u32, table_len: u32, h: f32, ln_r0: f32, inv_delta: f32, r0: f32 }
// One pseudised ion: position as an exact cell + f32 fraction (as nuclear.wgsl), its table offsets and radii.
struct Ion {
    cell: vec4<i32>,           // .w: offset of its V_cmp table
    frac: vec4<f32>,           // .xyz fraction of a cell, .w: Z_v
    radii: vec4<f32>,          // cmp_radius, tail_in, tail_out, tail_radius
    pc: vec4<f32>,             // .x: offset of its ρ_pc table (as f32-exact integer), .y: pc_radius, .z: 1/(σ_LR√2)
}

@group(0) @binding(0) var<uniform> U: IonParams;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> ions: array<Ion>;
@group(0) @binding(3) var<storage, read> tables: array<f32>;

fn pidx_of(idx: u32, n: u32, p: u32) -> u32 {
    let i = idx % n; let t = idx / n; let j = t % n; let k = t / n;
    return (i + 4u) + p * ((j + 4u) + p * (k + 4u));
}

// 4-point Lagrange in s = (ln d − ln r0)/δ, as qm::radial::LogGrid::interp (f[0] below r0)
fn interp(off: u32, d: f32) -> f32 {
    if (d <= U.r0) { return tables[off]; }
    let s = (log(d) - U.ln_r0) * U.inv_delta;
    let i0 = u32(clamp(i32(floor(s)) - 1, 0, i32(U.table_len) - 4));
    let t = s - f32(i0);
    let a = tables[off + i0]; let b = tables[off + i0 + 1u]; let c = tables[off + i0 + 2u]; let e = tables[off + i0 + 3u];
    return a * (-(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0) + b * (t * (t - 2.0) * (t - 3.0) / 2.0)
         + c * (-t * (t - 1.0) * (t - 3.0) / 2.0) + e * (t * (t - 1.0) * (t - 2.0) / 6.0);
}

fn psi_m(t: f32) -> f32 { if (t > 0.0) { return exp(-1.0 / t); } return 0.0; }
fn mask(r: f32, r_in: f32, r_out: f32) -> f32 {
    if (r <= r_in) { return 1.0; }
    if (r >= r_out) { return 0.0; }
    let t = (r - r_in) / (r_out - r_in);
    return psi_m(1.0 - t) / (psi_m(t) + psi_m(1.0 - t));
}

fn dist(i: vec3<i32>, ion: Ion) -> f32 {
    let dc = vec3<f32>(i - ion.cell.xyz) - ion.frac.xyz;
    return length(dc) * U.h;
}

// V_SR = V_cmp^filt (table) + (1 − m)·(−Z_v erfc(d/(σ_LR√2))/d), added into dst (which already holds V_LR).
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
            if (d > ion.radii.y && d <= ion.radii.w) {
                let x = d * ion.pc.z;
                let erfc = select(1.0 - x * erf_over_x(x), erfc_ge1(x), x >= 1.0);
                v = v - (1.0 - mask(d, ion.radii.y, ion.radii.z)) * ion.frac.w * erfc / d;
            }
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
- `GpuIonic::new` builds the ion records from `FilteredTable`s. Tables are concatenated `f32` V_cmp and ρ_pc per pseudised ion, padded to `table_len`.
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

---

### Task 9: `sim`/`providers`/`surrogate`/`app` plumbing **[Opus]**

Spec §4.7, §5.5 (`sim`, `surrogate`, `app` bullets), §5.6, §6.5, §8 row 9; review NOTE (the drift readouts reset on a policy switch). Runs in parallel with Task 8.

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
- `README.md` gets a "Pseudopotentials" section: what the three policies do; that energies are valence energies; that Li/Na/Mg are refused until the ONCV stage; the core-overlap tint; the partial-core display; generation on first use (seconds, cached per run).
- `CLAUDE.md` gets the `--ionic` flag, the `action ionic` replay pin, and the default (CoresOnly for presets in the app, `None` for `Scene` literals and old files).
- The UI checklist gets rows for the new controls.

- [ ] **Step 8: Commit**
```bash
git add crates/sim crates/providers crates/surrogate crates/qm-gpu/src/scf.rs crates/app crates/validate README.md CLAUDE.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md
git commit -o -m "feat(sim,providers,surrogate,app): ionic policy, pseudopotential cache, provenance and the panel (M7 Task 9)" -- crates/sim crates/providers crates/surrogate crates/qm-gpu/src/scf.rs crates/app crates/validate README.md CLAUDE.md docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md
```

---

### Task 10: validation oracles — atoms, geometries, egg-box record, dynamics re-pins **[Fable — physics-review gate]**

Spec §6.3–§6.7, §8 row 10; review S8 (the basis-set-limit LSDA reference; the angle gated under `All`), M5. **Spec question 8 applies:** `All` is usable from Task 6, so this task gates the geometries under `All` with H at its atomic-gate f, and Task 11 re-runs them after the dimer gate.

**Files:**
- Create: `crates/validate/tests/pseudo_oracles.rs` (GPU suite)
- Modify: `crates/providers/src/sampling/tests.rs` (`gpu_oxygen_reference_multiplicity` un-ignored, under `CoresOnly`), `crates/validate/references.md` (section "M7 oracles (Task 10)": the pinned literature digits and every measured number), `CLAUDE.md` (the new suites, their run commands and wall times), `.gitignore` (`/crates/validate/last-run-pseudo.md`)

**Interfaces:**
- Consumes: Tasks 6–9 (`sim::Scene.ionic`, `providers::GpuKs` with setups, `providers::pseudo`), `validate/tests/common` (`gpu`, `Locked`, `scene`, `relax`, `solve`, `fit_line`, `au_per_ps`, `timing_row`, `DT_FS`, `never`), `sim::{Dynamics, DynCommand}`.
- Produces: `crates/validate/tests/pseudo_oracles.rs` with the helpers `pseudo_scene(nuclei: Vec<Nucleus>, h: f64, box_bohr: f64, policy: IonicPolicy) -> Scene`, `angle_deg(s: &Scene, vertex: usize, a: usize, b: usize) -> f64` and `h_for(zs: &[u8], wanted: f64) -> f64` (the coarsest of `wanted` and every species' h_req). Task 11 extends the file.

- [ ] **Step 1: Look up and pin the literature digits (spec §6.4; review S8).** From NIST CCCBDB ("Calculated geometries", method **SVWN**, the largest available basis of the aug-cc-pVTZ/aug-cc-pVQZ class), record:
  - H₂O r(O–H) in Å and ∠HOH in degrees;
  - NH₃ ∠HNH;
  - the basis name of each;
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

fn h_for(zs: &[u8], wanted: f64) -> f64 {
    zs.iter().filter_map(|&z| qm::radial::generate(z).ok()?.record.h_req).fold(wanted, f64::min)
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

/// Spec §6.3: the h-convergence of pseudo-Ne on the GPU — monotone over the ladder h ≤ h_req; |E(0.25) − E(0.15)| reported.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn ne_h_convergence_is_monotone() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h_req = qm::radial::generate(10).unwrap().record.h_req.unwrap();
    let mut es = Vec::new();
    for &h in qm::radial::LEDGER.h_ladder.iter().filter(|&&h| h <= h_req) {
        let (mut p, _) = Locked::new(ctx.clone(), 1e-6);
        let s = pseudo_scene(vec![Nucleus::new(10, [0.0; 3])], h, 12.0, IonicPolicy::CoresOnly);
        es.push((h, solve(&mut p, &s, "Ne").energy));
    }
    eprintln!("pseudo-Ne E(h): {es:?}");
    assert!(es.windows(2).all(|w| w[1].1 <= w[0].1 + 1e-6), "not monotone: {es:?}");
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
  - the pinned digits with their source;
  - every measured angle and length (`All` and `CoresOnly`, h = 0.25 and 0.20);
  - the Ne h-series and |E(0.25) − E(0.15)| (spec §6.3: "the honest statement of what h = 0.25 costs");
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

---

### Task 11: `IonicPolicy::All` — pseudo-H/He, the dimer gate on f, H₂, the H₂O egg-box gate, the `h2o` preset gate, the default **[Fable — physics-review gate]**

Spec §4.4 (S6b, the dimer gate), §4.7 (`All` as the intended default once its gates pass), §6.4 (H₂ under `All`), §6.6 (the egg-box gate under `All`), §6.7, §8 row 11. **Spec question 9 applies:** the dimer gate is a 3-D computation that the radial code cannot run. It is supplied as a callback, and He's AE dimer has no minimum in the bracket.

**Files:**
- Modify: `crates/qm/src/radial/recipe.rs` (`DimerGate`, `DimerPair`, `generate_with`; `generate` = `generate_with(z, None)`), `crates/qm/src/radial/mod.rs` (re-export `generate_with`, `DimerGate`, `DimerPair`), `crates/providers/src/pseudo.rs` (`GridDimerGate`, `DimerBackend`, `set_dimer_backend`; `PseudoCache::table` generates through the gate), `crates/app/src/main.rs` (sets the dimer backend to the window's GPU, or CPU without one), `crates/app/src/ui/mod.rs` (`DEFAULT_IONIC`, Step 6), `crates/validate/tests/{pseudo_gen,pseudo_oracles}.rs`, `crates/validate/tests/common/mod.rs` (`set_dimer_backend` at suite start), `crates/validate/references.md` (sections "M7 `All` (Task 11)" and "M7 default policy"), `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: Tasks 1–10, plus `qm::extrapolate_sigma` and the σ-ladder rule of `validate::sigma_ladder` (h·{1.0, 1.2, 1.4, 1.6}; restated in the gate's doc and not imported, since providers do not depend on validate).
- Produces:
```rust
// qm::radial::recipe
pub struct DimerPair { pub pp: (f64, f64), pub ae: (f64, f64) }        // (R_e Bohr, ω a.u.) pseudised and AE σ → 0
pub trait DimerGate { fn dimers(&mut self, table: &PseudoTable) -> Result<DimerPair, String>; }
pub fn generate_with(z: u8, dimer: Option<&mut dyn DimerGate>) -> Result<PseudoTable, Refusal>;
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
- If the AE dimer has no force zero in the bracket (He₂: LSDA's well lies near 5 Bohr, outside [1, 4]·r_peak(1s) ≈ [0.57, 2.3]), the gate is "not applicable" for that element, a derived outcome recorded in the generation record.
- Costs (reported): ≈ 60 solves per f on the GPU (≈ 30 s); on the CPU fallback ≈ 40 min per element. The AE curve is cached per Z, so later f's cost only the PP solves.

- [ ] **Step 1: Write the failing tests**

`crates/validate/tests/pseudo_oracles.rs`:
```rust
/// Spec §8 row 11 first failing test (§6.4): pseudo-H₂ under `All` against this solver's own σ → 0
/// all-electron curve (references.md: R_e 1.451 Bohr, ω 4405 ± 60 cm⁻¹, that number's stated bias):
/// R_e within 0.02 Bohr, ω within 5 %; reported against LSDA (4201) and experiment (4401).
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
/// Ha/Bohr (the pseudopotential is the only egg-box source); the y-component and E rms reported.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_all_max_shift_sum_of_forces_below_3e_5() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8, 1], 0.25);
    let (mut p, _) = Locked::new(ctx, 1e-6);
    let (base, _, _) = relax(&mut p, water(IonicPolicy::All, h), &mut String::new());
    let q = h / 4.0;
    let (mut worst, mut fy, mut es) = (0.0f64, Vec::new(), Vec::new());
    for sx in [-q, q] { for sy in [-q, q] { for sz in [-q, q] {
        let mut s = base.clone();
        for n in &mut s.nuclei { n.pos[0] += sx; n.pos[1] += sy; n.pos[2] += sz; }
        let r = solve(&mut p, &s, "shift");
        let sum = [0, 1, 2].map(|k| r.forces.iter().map(|f| f[k]).sum::<f64>());
        worst = worst.max(sum.iter().map(|c| c * c).sum::<f64>().sqrt());
        fy.push(sum[1]);
        es.push(r.energy);
    }}}
    let mean = es.iter().sum::<f64>() / 8.0;
    let rms = (es.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / 8.0).sqrt();
    eprintln!("H₂O All at h = {h}: max |ΣF| {worst:.2e} (today's ΣF_y −3.4e-4), ΣF_y {fy:?}, E rms {rms:.2e}");
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
/// their rows join the pinned records (ZS gains 1 and 2; the block is regenerated).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_gen")]
fn hydrogen_and_helium_pass_the_atomic_gates() {
    for z in [1u8, 2] {
        let t = qm::radial::generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}"));
        assert_eq!((t.n_core, t.z_v), (0, z as f64));
        assert!(t.partial_core.r_nlc.is_none() && t.partial_core.r_core == 0.0);
        eprintln!("Z = {z}: f {}, {}, h_req {:?}, gates {:?}", t.f, t.record.local, t.record.h_req, t.gates);
    }
}
```
and `ZS` becomes `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. The records block is regenerated with the dimer column. Generation there runs without a gate (`generate`), so the H/He rows read "dimer: not run". The dimer-gated f is recorded by `pseudo_oracles` (below) and in references.md "M7 `All`".

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

/// Fills `record.dimer`; false ⇔ the gate ran and failed (the ladder moves on).
fn dimer_verdict(t: &mut PseudoTable, gate: Option<&mut dyn DimerGate>) -> bool {
    let Some(g) = gate else { t.record.dimer = "not run (no dimer gate supplied)".into(); return true };
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
`candidate` fills the record (`record_of`) before `dimer_verdict` runs, and h_req is computed only for the accepted entry, after the dimer gate. This replaces `generate`'s body from Tasks 3–5.

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
        }).clone()?;
        Ok(qm::radial::recipe::DimerPair { pp: pseudo_dimer(t, &self.backend)?, ae })
    }
}
```
`PseudoCache::table` calls `qm::radial::generate_with(z, Some(&mut GridDimerGate::new(dimer_backend())))`. The gate only runs for Z ≤ `LEDGER.dimer_z_max`: the ladder decides that, not the cache. In `crates/app/src/main.rs`, after the GPU context exists: `providers::pseudo::set_dimer_backend(DimerBackend::Gpu(ctx.clone()))`. Without a GPU the default `Cpu` applies, and the status line says the first `All` use generates H for minutes. `common::Locked::new` calls `set_dimer_backend(Gpu(ctx))` once.

- [ ] **Step 5: Run the tests and re-run Task 10's oracles**

Run: `cargo test -p providers --release dimer_gate`, `cargo test -p validate --release --test pseudo_gen` (regenerate and pin the records block after review), and `cargo test -p validate --release --test pseudo_oracles`, which includes Task 10's geometry tests, now with the dimer-gated H.
Expected: PASS. Report any miss with numbers (rule 6).

Run the regression suites: `cargo test -p validate --release --test dynamics`, `--test surrogate`, `--test tools`, and `cargo test -p app --release`.
Expected: PASS, unchanged.

- [ ] **Step 6: The default policy decision (spec §4.7: "the intended default once its gates pass")**
  - **Every gate of this task and Task 10 passes under `All`:** set `ui::DEFAULT_IONIC = sim::IonicPolicy::All`. Update the app's start-up text and README ("presets start with every species pseudised; Physics ▸ Ionic potential switches") and CLAUDE.md's default line.
  - **Any gate failed:** `DEFAULT_IONIC` stays `CoresOnly`, and the failing gate and its numbers are recorded.

  Either way, add the section "M7 default policy" to `references.md` listing each gate with its number and the decision.

- [ ] **Step 7: Commit**
```bash
git add crates/qm/src/radial/recipe.rs crates/providers/src/pseudo.rs crates/app crates/validate README.md CLAUDE.md
git commit -o -m "feat(qm,providers,validate): IonicPolicy::All — dimer gate on f, pseudo-H2, H2O egg-box and preset gates, default policy (M7 Task 11)" -- crates/qm/src/radial/recipe.rs crates/providers/src/pseudo.rs crates/app crates/validate README.md CLAUDE.md
```
Before merging, hand the diff to `physics-reviewer` with Spec question 9, the dimer numbers, and the decision.

---

### Task 12: Z = 11–18 enablement **[Opus]**

Spec §4.2 (Na/Mg semicore → refused; Al–Ar n = 2 core; the S5 cap on the 3d channel), §5.5 (`units` bullet: the mass is the only datum added), §6.2, §8 row 12.

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
/// Spec §8 row 12: Al–Ar pass every gate (n = 2 core, the 3d channel capped per review S5) or are
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
                let occ_max = t.channels.iter().zip(&t.partition.channels).filter(|(_, c)| c.occupied).map(|(tm, _)| tm.r_c).fold(0.0, f64::max);
                for (tm, c) in t.channels.iter().zip(&t.partition.channels).filter(|(_, c)| !c.occupied) {
                    assert!(tm.r_c <= occ_max + 1e-12, "Z = {z}, l = {}: r_c {} above the S5 cap {occ_max}", c.l, tm.r_c);
                }
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

`crates/app/src/ui/ptable.rs` tests: `tip(11)` names the refusal ("refused: two valence s states — needs the ONCV stage; all-electron under Cores only"), `tip(16)` reads "pseudopotential (generated on first use)", `enabled(18)` is true and `enabled(19)` is false.

- [ ] **Step 3: Run them to verify they fail**

Run: `cargo test -p units --release isotope_mass_table_covers_z_up_to_18`
Expected: FAIL at Z = 11.

Run: `cargo test -p validate --release --test pseudo_gen al_through_ar`
Expected: runs. It passes or reports per element, because the generator is Task 4's code; this task wires the elements through the app.

- [ ] **Step 4: Implement**
- **`units`:** add the Step 1 rows to `TABLE` (index 11–18) and grow the array to `[&[Row]; 19]`; set the most-abundant defaults; update the module doc's isotope list.
- **`render::elements::colour`:** Jmol CPK display colours — Na `[0.67, 0.36, 0.95]`, Mg `[0.54, 1.0, 0.0]`, Al `[0.75, 0.65, 0.65]`, Si `[0.94, 0.78, 0.63]`, P `[1.0, 0.50, 0.0]`, S `[1.0, 1.0, 0.19]`, Cl `[0.12, 0.94, 0.12]`, Ar `[0.50, 0.82, 0.89]` (display only, rule 1).
- **`ptable.rs`:** `Z_ENABLED = 18`. `tip(z)` for 11..=18 reads from `qm::radial::has_core`/`generate` results cached in the providers' `PseudoCache` when available, else "pseudopotential (generated on first use)". The panel note changes to "Z 1–18 can be placed; heavier elements need the next pseudopotential stage (ONCV, semicore)".
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

Spec §8 row 13 (user request 2026-10-15: "with M7, you should display the bond angles on the bond chains that are drawn between atoms"). The overlay is display only: pure geometry from the nuclear positions, with no physics input. **Spec question 10 applies:** the "M6 bond-strength kind" does not exist on any branch yet, so the weak-contact style hook is `angle_style`.

**Files:**
- Modify: `crates/app/src/overlay.rs` (`AngleMark`, `bond_angles`, `shown_angles`, `angle_style`, `angle_arc_lines`, `angle_label`), `crates/app/src/viewer.rs` (arcs in `overlay_lines`, labels in `label_overlay`), `crates/app/src/ui/{mod.rs,actions.rs,menus.rs,search.rs}` (`UiState.show_angles`, `Action::ShowAngles`, the View toggle, the search entry), `crates/app/src/ui/dock/selection.rs` (the angle list with the surface note), `crates/app/src/screenshot.rs` (`--ui` key `angles=on|off`), `crates/app/replays/h2o_angle_all.replay` (new), `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md`, `docs/superpowers/notes/2026-10-14-ui-redesign-checklist.md`, `README.md`

**Interfaces:**
- Consumes: `sim::BondGraph { bonds, n_nuclei, .. }`, `sim::Bond { i, j, multi_centre, .. }`, `render::lines::{segment, LineVertex}`, `overlay::{BOND, RING_BOND, BOND_ALPHA, STALE_BOND_ALPHA}`, `tools::project`, `EvalStats.pseudised` (Task 9), `UiState`, `Action`.
- Produces:
```rust
// crates/app/src/overlay.rs
#[derive(Clone, Debug, PartialEq)]
pub struct AngleMark { pub vertex: usize, pub a: usize, pub b: usize, pub radians: f64, pub dashed: bool }
pub const ANGLE_ARC_RADIUS: f32 = 0.45;             // Bohr (display)
pub const ANGLE_ARC_SEGMENTS: usize = 16;            // display
pub const ANGLE_DECLUTTER_BONDS: usize = 3;          // more bonds than this at a vertex: labelled only when selected/hovered
pub fn bond_angles(pos: &[[f64; 3]], graph: &sim::BondGraph) -> Vec<AngleMark>;
pub fn shown_angles<'a>(all: &'a [AngleMark], graph: &sim::BondGraph, selected: Option<usize>, hovered: Option<usize>) -> Vec<&'a AngleMark>;
pub fn angle_style(a: &sim::Bond, b: &sim::Bond) -> bool;          // dashed
pub fn angle_arc_lines(pos: &[[f64; 3]], m: &AngleMark, stale: bool) -> Vec<LineVertex>;
pub fn angle_label(m: &AngleMark) -> String;                        // "104.4°"
pub fn angle_label_anchor(pos: &[[f64; 3]], m: &AngleMark) -> [f64; 3];
// crates/app/src/ui
pub fn angle_surface_note(pseudised: &[u8], zs: [u8; 3]) -> &'static str;
pub enum Action { /* … */ ShowAngles(bool) }                        // script word `angles true|false`
pub struct UiState { /* … */ pub show_angles: bool }                // default true
```
**Rules** (the spec row's declutter, made exact):
- An angle exists for every pair of drawn bonds meeting at a vertex.
- A vertex with ≤ `ANGLE_DECLUTTER_BONDS` (3) bonds shows all its angles: water, NH₃ and every "small molecule".
- A vertex where more than 3 bonds meet (CH₄'s C) shows its angles only when that atom is selected or hovered.
- Angles are drawn between the positions drawn now and faded with the bonds when the graph is stale (§3.4 of the M6 spec, as `bond_lines`).
- An angle is dashed when either of its bonds is dashed (multi-centre today; Spec question 10).
- The angle is computed as atan2(|u×w|, u·w), well conditioned near 0° and 180°.

- [ ] **Step 1: Write the failing tests** (`crates/app/src/overlay.rs` tests)
```rust
fn graph(n: usize, pairs: &[(usize, usize)], ring: bool) -> sim::BondGraph {
    sim::BondGraph {
        bonds: pairs.iter().map(|&(i, j)| sim::Bond { i, j, rho_bcp: 0.2, enhancement: 1.5, s: 0.5, route: sim::bonds::Route::Saddle, multi_centre: ring }).collect(),
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
    assert_eq!(shown_angles(&bond_angles(&pos, &g), &g, None, None).len(), 3);
    let ring = graph(3, &[(0, 1), (1, 2), (0, 2)], true);
    let h3 = vec![[0.0, 0.0, 0.0], [1.65, 0.0, 0.0], [0.825, 1.429, 0.0]];
    assert!(bond_angles(&h3, &ring).iter().all(|m| m.dashed));
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

/// Dashed when either bond is drawn dashed: a multi-centre (ring) bond today. The M6 bond-strength
/// display (weak/closed-shell contacts drawn faint or dashed), when it lands, extends this one
/// function (Spec question 10).
pub fn angle_style(a: &sim::Bond, b: &sim::Bond) -> bool { a.multi_centre || b.multi_centre }

pub fn bond_angles(pos: &[[f64; 3]], graph: &sim::BondGraph) -> Vec<AngleMark> {
    if graph.n_nuclei != pos.len() { return Vec::new(); }
    let mut out = Vec::new();
    for v in 0..pos.len() {
        let at: Vec<(usize, &sim::Bond)> = graph.bonds.iter().filter_map(|b| if b.i == v { Some((b.j, b)) } else if b.j == v { Some((b.i, b)) } else { None }).collect();
        for x in 0..at.len() {
            for y in x + 1..at.len() {
                let ((a, ba), (b, bb)) = (at[x], at[y]);
                let (a, b, ba, bb) = if a < b { (a, b, ba, bb) } else { (b, a, bb, ba) };
                out.push(AngleMark { vertex: v, a, b, radians: angle_between(sub(pos[a], pos[v]), sub(pos[b], pos[v])), dashed: angle_style(ba, bb) });
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

/// An arc of radius ANGLE_ARC_RADIUS around the vertex from the direction of `a` to that of `b`
/// in ANGLE_ARC_SEGMENTS steps, in the bond colour; every other segment skipped when dashed; faded
/// with stale bonds. Parametrised as u·cos(tθ) + n̂·sin(tθ), n̂ the unit vector in the (u, w) plane
/// perpendicular to u (well defined below 180°; at 180° any perpendicular from `plane_basis`).
pub fn angle_arc_lines(pos: &[[f64; 3]], m: &AngleMark, stale: bool) -> Vec<LineVertex> {
    let mut out = Vec::new();
    let Some((v, u, n)) = arc_frame(pos, m) else { return out };
    let rgb = if m.dashed { RING_BOND } else { BOND };
    let colour = [rgb[0], rgb[1], rgb[2], if stale { STALE_BOND_ALPHA } else { BOND_ALPHA }];
    let theta = m.radians as f32;
    let point = |t: f32| v + ANGLE_ARC_RADIUS * (u * (t * theta).cos() + n * (t * theta).sin());
    for k in 0..ANGLE_ARC_SEGMENTS {
        if m.dashed && k % 2 == 1 { continue; }
        let (t0, t1) = (k as f32 / ANGLE_ARC_SEGMENTS as f32, (k + 1) as f32 / ANGLE_ARC_SEGMENTS as f32);
        lines::segment(&mut out, point(t0).to_array(), point(t1).to_array(), colour);
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
- View ▸ Bonds and molecules: checkbox "bond angles", with the tooltip "the angle between each pair of drawn bonds at an atom (geometry only, from the nuclear positions). Where more than three bonds meet, select or hover the atom to see its angles. Under all-electron nuclei (σ = h) the angles are the smeared surface's — water relaxes toward 177° there; with pseudopotentials it is ≈ 105°".
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
| §5.1 LR/SR split, V_cmp/V_tail (Spec question 2), filter (S3, S7), σ_LR = 2h | Task 5; grid sampling Task 6; GPU Task 8 |
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

## Spec questions

Each question below names a spec defect or under-specification the plan found, the ruling the plan implements, and the task whose physics-review gate must confirm the ruling. None changes a spec threshold.

1. **Which functional is the NIST LSD column?** (Task 1)
   - Defect: the 1e-5 NIST gates (H −0.478671, He −2.834836, Li −7.335195) compare a PW92 solver with a column computed with Vosko–Wilk–Nusair correlation (to be verified in Step 1). The VWN–PW92 fit difference can exceed 1e-5.
   - Ruling: if NIST is VWN, the 1e-5/1e-4 oracles run on a test-only VWN5 functional, like for like, and the generator's PW92 totals are reported against NIST (rule 7).
2. **V_cmp = V_loc + Z_v/d is singular at the origin** (§5.1, from review M4) (Task 5)
   - Defect: V_loc is finite at 0, so V_cmp is not "smooth at the origin" as stated. It has a Coulomb singularity, it is not band-limited, and V_tail = −Z_v erfc/d is singular there too; sampled at a grid point on the ion it is −∞.
   - Ruling: keep the exact LR/SR split and switch the tail on smoothly. V_tail = (1 − m(d; r_c/γ_r, r_c))·(−Z_v erfc(d/(σ_LR√2))/d) and V_cmp = V_SR − V_tail, with V_SR = V_loc + Z_v erf(·)/d. Both pieces are then smooth; V_cmp is compact within max(r_c, r_cmp), and V_tail is never masked outward. No new constant: m is the parameter-free C^∞ step, and γ_r is the ledger's.
3. **"King-Smith's smooth cutoff" is not specified beyond q_c and γ_r** (§5.1) (Task 5)
   - Ruling: a hard cut at q_c = 0.7π/h, then the C^∞ mask ψ(1−t)/(ψ(t)+ψ(1−t)), ψ(t) = e^{−1/t}, over [R, γ_r R]. Only the two ledger constants are used.
   - The judge is Task 5's lattice-sum test (1e-8 relative). If it misses, the reviewer chooses a smoother mask (a ledger change) or plan B (Ono–Hirose), never a looser bound.
4. **`deliberate_bad_local_is_caught_as_ghost` as worded cannot fail GSS** (§8 row 3) (Task 3)
   - Defect: a local with ε₀^loc above ε_s implies E_KB < 0 by the variational principle (E_KB > 0 ⇔ ⟨φ|δV|φ⟩ > 0 ⇒ ⟨φ|H_loc|φ⟩ < ε_s ⇒ ε₀^loc < ε_s), and GSS calls that case ghost-free.
   - Ruling: the test constructs the two real ghost cases instead: E_KB > 0 with ε₁^loc < ε_s (the textbook deep local), and E_KB < 0 with ε₀^loc < ε_s.
5. **G2's "node counting" does not order the eigenvalues of a nonlocal operator** (§4.6) (Task 3)
   - Ruling: count the zeros of the 2×2 matching determinant on a 1 mHa mesh, adding `LEDGER.numerics.g2_mesh` = 1e-3 to the ledger, and bisect each zero. Channels without a projector keep node counting.
6. **V_LR "from the Poisson solve of a deposited Gaussian"** (§5.1/§5.2) (Tasks 6, 8)
   - Defect: both solvers compute the nuclear potential by the analytic erf (`nuclear_potential`, `nuclear.wgsl`) since M3 Task 10, not by a Poisson solve.
   - Ruling: V_LR is the analytic erf per centre (q, σ). It is exact pointwise, and σ_LR = 2h is kept as the spec's numerical width.
7. **c_lm "finished in f64 on the CPU" (§5.3) would put a readback in every Hamiltonian application** of the GPU LOBPCG (Task 8)
   - Ruling: inside H, f32 workgroup partials are finished on the GPU by a pairwise kernel (≤ 64 terms). E_NL and the forces are finished in f64 on the CPU from one readback (rule 4).
   - `gpu_vs_cpu` decides, and the Kahan path (§7 risk 6) is the measured remedy.
8. **Task 10 gates the geometries under `All`, but pseudo-H's dimer gate is Task 11** (Tasks 10, 11)
   - Ruling: the no-core branch is generic, so `All` is usable from Task 6 (H at its atomic-gate f). Task 10 gates with that H, and Task 11 re-runs Task 10's oracles once the dimer gate may have moved H's f.
9. **The dimer gate (S6b) is a 3-D computation inside the f ladder**, which `qm::radial` cannot run; He₂ has no usable AE σ → 0 reference (Task 11)
   - Ruling, the gate: a `DimerGate` callback. Providers compute both dimers at the finest ladder h (0.15) by force bisection within `LEDGER.dimer_bracket` = [1, 4]·r_peak, a new ledger field. The AE side is extrapolated σ → 0 with the oracle's σ ladder.
   - Ruling, He: no AE force zero in the bracket makes the gate "not applicable", a derived outcome recorded in the generation record.
   - The cost (≈ 30 s on the GPU, ≈ 40 min on the CPU per element) is stated.
10. **"The M6 bond-strength kind" (Task 13) does not exist on any branch yet** (`m6-bond-strength` has no commits) (Task 13)
    - Ruling: angles inherit the dashed style from multi-centre bonds and the fade from stale graphs. The bond-strength work extends the single `angle_style` function when it lands.
11. **h_req's "coarsest h at which …" is ambiguous if the ladder is not monotone** (§4.8 T8) (Task 5)
    - Ruling: the coarsest ladder h such that it and every finer ladder h pass, so a scene at any allowed ladder h is certified.
12. **"δE_equiv has no poles"** holds only for the form δE = Δ[u u′/(2∫₀^{r_d}u²)], each side with its own u (§4.8 T5) (Task 4)
    - Defect: (D^PP − D^AE)·u_AE² keeps D^PP's pole.
    - Ruling: the pole-free difference form, which is first-order equal to the spec's.
