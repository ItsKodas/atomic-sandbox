# Live orbitals, Phase A (Harris display orbitals) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** under the estimated backend with the GPU teacher, the Orbitals, All orbitals coloured and Spin density views show orbitals solved at the **current** nuclei in the potential of the superposed isolated atoms (Harris, non-self-consistent), refreshed while playing and after a paused geometry change, honestly labelled, and never counted as DFT.

**Architecture:** `qm` gets a one-shot occupation rule; `qm-gpu` gets `GpuSolver::harris` writing into its own `HarrisState` (a `GpuState` the existing `OrbitalView` products path reads), never the SCF's warm state. `providers` builds ρ_in from the cloud's `AtomProfile`s (moved to `sim` so they can cross the boundary) and implements the one new `ForceProvider::harris_display`: `GpuKs` solves now, `SurrogateKs` queues a `JobKind::HarrisDisplay` on its background auditor. The app worker schedules the jobs (cadence, budget, debounce, Model only) and labels them. Display only.

**Tech Stack:** Rust; `qm`, `qm-gpu` (wgpu, no new WGSL), `sim`, `providers`, `app` (egui), `validate` (new `--test harris`).

**Spec:** `docs/superpowers/specs/2026-11-14-live-orbitals-design.md` (approved 2026-11-14; §9's builder-check rulings are binding). This plan covers §4, §6, §7 V-A1–V-A4 and §7's gates. Phase B (§5) is out of scope.

## Global Constraints

- **Model: Opus for every task** (user's standing choice, memory "No Fable"); tasks marked *physics review before merge* go to the `physics-reviewer` agent (Opus) before they land.
- Worktree `D:\Projects\Physics\.claude\worktrees\live-orbitals`, branch `live-orbitals`. One heavy cargo job at a time; never kill the user's `app.exe` or cargo; with another job running, build at BelowNormal `-j 4`.
- Rule 1: every constant here (1 s, 2 s, 300 ms, 15 %, 1e-6 Bohr, 0.1 e/Bohr³) is a display choice, the same for every element; the atoms' densities come only from this app's own isolated-atom solves.
- Rule 3: atomic units inside `qm`, `qm-gpu`, `sim`, `providers`; fs and eV only in `app` labels.
- Rule 4: grid fields f32 on the GPU; ρ_in assembled in f64 on the CPU with Neumaier sums; eigenvalues, occupations, overlaps f64.
- Rule 5: the **only** `sim` trait change is one new default-`None` `ForceProvider::harris_display`; the rest of the `sim` change is data (`AtomProfile` moved in, `OrbitalKind`, `ExperienceUse`, `HarrisAtoms`, `HarrisAnswer`, `SideKind::Harris`, `SideDisplay::display: Option<_>`). Nothing downcasts a provider.
- Rule 7: every place listed in spec §6 states "not self-consistent", ε approximate ("≈" + tooltip), shapes qualitative, density/spin are this solve's output, unoccupied mostly box states, and the rescale on a charged scene.
- G4 / anti-goals: a Harris job is never a fallback, audit, teacher solve, sample or `TeachEvent`; it never runs on the stepping thread; Activity never titles it "DFT fallback".
- Every control that reaches the worker goes through `ui::Action` (recorded by `--record`); replay scripts keep parsing.
- Heavy tests are release-only: `#[cfg_attr(debug_assertions, ignore)]`.
- Phase B hook: `OrbitalKind::Harris { experience: Option<ExperienceUse>, .. }` with `experience` always `None` in Phase A.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; commit with pathspecs (`git commit -o -- <paths>`). README "Orbitals and spin" + estimated-mode text, `docs/dev-reference.md` and the CLAUDE.md suite table are kept current (Tasks 7–9).

## Review Focus

1. **A paused drag** (dozens of moves in a second): exactly one Harris job after the last move + 300 ms; a queued job is replaced, an in-flight one yields to the next command; the shown payload is the last geometry's. Tests: Task 6 `debounce_fires_once_after_the_last_move`, Task 5 `a_newer_harris_request_replaces_the_queued_one`.
2. **Model only** never starts a DFT side solve for display: the view-change refresh, the interrupted-refresh retry and a changed unoccupied count all become Harris jobs. Test: Task 6 `model_only_view_paths_never_request_a_dft_refresh`.
3. **A pseudised scene with "show partial core" on**: the atom profiles are valence-only, ∫ρ_in = N (not N + core). Tests: Task 5 `atom_jobs_solve_without_the_partial_core`, Task 4 `superposition_integrates_to_the_valence_count`.
4. **Charged or electron-less scenes** (H₃⁺; H⁺ with N = 0): H₃⁺ is rescaled and says so; N = 0 gives the "no electrons" note, never a panic or a solve. Tests: Task 4 `charged_scene_is_rescaled_and_labelled`, `no_electrons_is_a_note_not_a_solve`.
5. **A DFT payload and a Harris payload at one geometry, in either arrival order**: the self-consistent one is shown; at different geometries the newer job wins. Test: Task 3 `side_replaces_prefers_converged_at_the_same_geometry`.

## Rulings made by this plan (spec §10 and gaps; check them)

- **Request path.** The one new `ForceProvider::harris_display` serves both sides: `GpuKs` answers `Solved`, `SurrogateKs` answers `Queued` and delivers a `SideKind::Harris` `SideDisplay`. The worker reaches it through `Dynamics::provider_mut()`. `Trainable` is unchanged.
- **Profiles.** `AtomProfile`/`radial_profile` move to `sim` unchanged (they cross the boundary as `HarrisAtoms`, and `validate` needs them). ρ_pc is kept out by **requesting**: Atom/Reference jobs always solve with the display partial core off, so the cloud's superposition is valence-only even with "show partial core" on.
- **Unoccupied states** join the one LOBPCG (no `occupied_disagreement` step); `OrbitalView::adopt_unoccupied` copies them. One `OrbitalView` serves both kinds, so tracks continue.
- **One-shot `DegenerateShells`:** phase 1 at τ_c on the Harris ε, then phase 2's release test once (Task 1).
- **Probe:** the existing display slot, with a `DisplayJob` kind.
- **Views:** Harris runs for `Orbital`, `AllColoured`, `SpinDensity`, never for `List` or `Off`.
- **Two side slots:** the surrogate holds a DFT and a Harris display separately, and the driver drains both per tick. Precedence: at the same geometry (1e-6 Bohr) Converged beats Harris; otherwise the newer job wins. A Harris payload is unfaded only at the current geometry.
- **Model only** reaches the worker as `Command::SetModelOnly` from the existing `Action::ModelOnly`. The new setting is `Action::OrbitalRefresh(s)`.
- **V-A4:** bond experience's pure store moves to `sim` so `validate` can use it. "Near the bond length" means SCF ρ_bcp ≥ 0.1 e/Bohr³.

---

### Task 1: One-shot occupations (Model: Opus; physics review before merge)

**Files:** modify `crates/qm/src/scf.rs` (next to `degenerate_shell_occupations` ≈ :275; module doc "Occupations" gets a paragraph), `crates/qm/src/lib.rs` (export).

**Interfaces — produces:**
- `pub struct OneShot { pub f: Vec<f64>, pub shared: Option<FractionalShell>, pub released: bool, pub window_short: bool }`
- `pub fn one_shot_occupations(evals: &[f64], n: usize, spin: u8, rule: Occupations) -> OneShot` — `f` in the order of `evals` (any order), Σf = n.
  - `n == 0`: all zeros.
  - `Aufbau`: 1 on the n lowest by rank. Panics if `evals.len() < n`.
  - `DegenerateShells { tau }`: phase 1 = `degenerate_shell_occupations(evals, n, CANDIDATE_TAU)`; when it returns `None` (the top computed level within τ_c of ε_F), the cluster is every computed level within τ_c of ε_F shared equally and `window_short = true`. Then phase 2 once: a shared cluster whose spread ≥ `tau` is split by `released_occupations` (`released = true`). Panics if `evals.len() <= n`.
  - The `match` on `Occupations` stays exhaustive (no `_` arm), so the Fermi–Dirac variant of the approved smearing work fails to compile here until its arm is written.

- [ ] Test `one_shot_aufbau_fills_the_lowest`: evals [−0.6, −0.2, −0.1, 0.3] shuffled, n = 2 → f = 1 on −0.6 and −0.2, Σf = 2.
- [ ] Test `one_shot_shares_a_degenerate_p_shell`: [−1.0, −0.40000, −0.40004, −0.39998, 0.2], n = 3, `DegenerateShells { tau: DEGENERACY_TAU }` → f = 1, 2/3, 2/3, 2/3, 0; `shared = Some(FractionalShell { first: 1, count: 3, .. })`; `released == false`.
- [ ] Test `one_shot_releases_a_split_cluster`: [−1.0, −0.402, −0.400, −0.395, 0.2], n = 2 (spread 7e-3 ≥ τ) → `released`, f equals `released_occupations` of the cluster, Σf = 2 to 1e-12.
- [ ] Test `one_shot_marks_a_short_window`: [−1.0, −0.40, −0.39, −0.35], n = 2 (top level within τ_c = 0.1 of ε_F) → `window_short`, the three levels within 0.1 share the remaining electron, Σf = 2.
- [ ] Run `cargo test -p qm --release one_shot` → PASS. Commit `feat(qm): one-shot occupations for a single diagonalisation (Harris display)`.

### Task 2: `GpuSolver::harris` and `HarrisState` (Model: Opus; physics review before merge)

**Files:** create `crates/qm-gpu/src/harris.rs` (module doc: the physics of spec §4.1, the term order, what is and is not touched); modify `crates/qm-gpu/src/lib.rs`, `crates/qm-gpu/src/scf.rs` (a crate-private accessor for the Poisson solver, kernels, reducer and the grown `LobpcgWork` that does not require a converged state — beside `display_work` ≈ :1240), `crates/qm-gpu/src/orbitals.rs` (`adopt_unoccupied`; factor the tail of `unoccupied` — residuals, sort, τ-set face density, `block_edge` — into a helper both use).

**Interfaces:**
- Consumes Task 1's `qm::one_shot_occupations`.
- Produces:
  - `pub struct HarrisState` (own fields, own `GpuIonic` and v_nuc, its vectors kept as the next call's guess); `HarrisState::new() -> HarrisState`, `fn state(&self) -> Option<&GpuState>`, `fn unoccupied(&self) -> ([&[Field]; 2], Option<&Unoccupied>)`, `fn clear(&mut self)`.
  - `pub struct HarrisInput<'a> { pub system: &'a qm::System, pub settings: &'a qm::ScfSettings, pub rho_in: [&'a [f64]; 2], pub unoccupied: usize }` (ρ_in↑, ρ_in↓ on the interior in `qm::Grid` layout).
  - `pub struct HarrisSolve { pub evals: [Vec<f64>; 2], pub occ: [Vec<f64>; 2], pub n_occ: [usize; 2], pub notes: Vec<String>, pub iterations: [usize; 2], pub at_cap: [bool; 2], pub cold: bool, pub wall: Duration }` (`evals`/`occ` every computed level, ascending; `notes`: a released shell, a short window, an eigensolve at its cap).
  - `pub fn GpuSolver::harris(&mut self, hs: &mut HarrisState, input: &HarrisInput, should_abort: &mut dyn FnMut() -> bool) -> Result<Option<HarrisSolve>, QmError>` — `Ok(None)` when aborted (then `hs.state()` is `None`, its vectors stay as the next guess).
  - `pub fn OrbitalView::adopt_unoccupied(&mut self, solver: &GpuSolver, vectors: [&[Field]; 2], info: Unoccupied)` — copies the vectors into this view's own pool fields (the view never borrows `hs`), replaces `unocc`/`unocc_info`; the next `update` shows them as rows n_σ...
- Algorithm (the parts the signature does not determine):
  1. Upload ρ_in,σ; ρ = ρ↑ + ρ↓. v_nuc: `nuclear_potential_in(nuclei, settings.sigma)` all-electron, else `hs`'s own `GpuIonic` (`new`/`update` for these nuclei and setup) `local_potential_in`; ρ_pc via its `partial_core_in` when the setup has a core.
  2. V_H[ρ_in]: the solver's `GpuPoisson::solve` at its `poisson_tol`, `POISSON_MAX_ITER`, warm from `hs`'s previous V_H (0 when cold). `xc_in(ρ↑, ρ↓, core)`; v_σ = v_nuc + V_H + v_xc,σ by `lincomb_in` in the SCF's term order (`run` ≈ :975).
  3. Per channel with n_σ + m > 0: one `lobpcg` for k_σ = n_σ + m + `UNOCC_GUARD` vectors (≥ 2 above N_σ, so the `DegenerateShells` witness is always present), `GpuHamiltonian { v, nl: hs ionic's nonlocal }`, tol `effective_eig_tol(settings)`, `EIGEN_MAX_ITER`, `Some(should_abort)`. Guess: `hs`'s vectors when warm and k_σ matches, else `qm::scf::initial_orbitals(grid, nuclei, k_σ, 5 + ch)`. Occupied and unoccupied in this one solve: there is no separate unoccupied step and so no `occupied_disagreement` check (spec §4.2, the builder's choice).
  4. `one_shot_occupations` on the sorted ε; the levels with f > 0 (the lowest, by construction) become `state.orbitals_σ` with `occ_σ`; the rest become the `Unoccupied` record (the factored helper) and `hs`'s unoccupied vectors.
  5. ρ_out,σ = Σ f_i ψ_i² (`mul_add_sq_w_in`) into `state.rho_up/rho_dn`, `rho` their sum; `state.v_h` = V_H[ρ_in], `state.v_nuc`, `ionic`, `occupations` from the input.
  It never reads or writes `GpuSolver::warm`, `converged`, `prev_rho` or the solver's cached `ionic`.

- [ ] Test (qm-gpu, release) `harris_on_the_converged_density_reproduces_the_scf`: H₂ R = 1.4, 33³, h = 0.25: solve, feed the state's ρ_up/ρ_dn (`ks_state`) as ρ_in → every occupied |ε_H − ε_SCF| ≤ 1e-5 Ha; (1/n_σ)‖⟨ψ_H|ψ_SCF⟩‖_F² ≥ 1 − 1e-6; the state's ρ_up/ρ_dn are the output Σ f|ψ|² (h³Σρ_σ = n_σ to 1e-5), which is what the Spin density product reads (spec §4.1: never ρ_in's split).
- [ ] Test `harris_leaves_the_scf_state_bitwise_unchanged`: H₂ 33³: solve; `ks_state` A; two `harris` calls; `ks_state` B == A bit for bit; then a warm `solve` from both a Harris-touched and an untouched twin solver gives the same iterations and bitwise energy.
- [ ] Test `harris_ionic_cache_is_its_own`: pseudo-O at the origin under `IonicPolicy::All` 49³: SCF solve at the origin, then `harris` with the atom moved 0.3 Bohr → the solver's `display_work(k).ionic` nuclei are still the SCF's (the cache was not rebuilt).
- [ ] Test `harris_abort_leaves_no_state`: an abort on the first poll → `Ok(None)`, `hs.state().is_none()`; the next call succeeds warm.
- [ ] Test `adopted_unoccupied_rows_follow_the_occupied`: H₂ 33³, m = 2 → `OrbitalView::update` on `hs.state()` after `adopt_unoccupied` gives 2 + 2 rows, the unoccupied ε ascending and above the HOMO.
- [ ] `cargo test -p qm-gpu --release` green; physics review (term order, partial core only in xc, the one-shot occupations, the shared-work safety argument); commit `feat(qm-gpu): Harris display solve into its own state`.

### Task 3: `sim` data, the one trait method, the driver's precedence (Model: Opus)

**Files:** create `crates/sim/src/profiles.rs` (move `AtomProfile`, `radial_profile`, `TAIL_FRACTION` and the `at`/`reach`/`fits` impl from `crates/app/src/cloud.rs` unchanged, with their tests `a_profile_fits_only_its_own_ionic_policy` and the profile half of `superposition_integrates_…`); modify `crates/sim/src/lib.rs`, `crates/sim/src/provider.rs`, `crates/sim/src/dynamics.rs` (`poll_side_display` ≈ :1800, module doc "Orbital display" ≈ :294 and "Side densities"), `crates/app/src/cloud.rs` (`pub use sim::{AtomProfile, radial_profile};` — call sites unchanged), every `OrbitalDisplay { .. }` literal (`kind: OrbitalKind::Converged`), `crates/app/src/ui/mod.rs:1726` and `crates/app/src/viewer.rs:109` (the new `SideKind` arm).

**Interfaces — produces:**
- `#[derive(Clone, Debug, Default, PartialEq)] pub enum OrbitalKind { #[default] Converged, Harris { experience: Option<ExperienceUse>, rescale: f64, wall_s: f64 } }`; `OrbitalDisplay` gains `pub kind: OrbitalKind`.
- `#[derive(Clone, Debug, PartialEq)] pub struct ExperienceUse { pub atoms: Vec<usize>, pub of: usize }` (Phase B fills it; doc says so).
- `pub struct HarrisAtoms { pub profiles: Vec<AtomProfile> }`.
- `pub enum HarrisAnswer { Solved(Result<OrbitalDisplay, ProviderError>), Queued { job: u64 } }`.
- `ForceProvider::harris_display(&mut self, _scene: &Scene, _atoms: &Arc<HarrisAtoms>, _should_abort: &mut dyn FnMut() -> bool) -> Option<HarrisAnswer> { None }` — doc: display-only orbitals at `scene` in the potential of the superposed `atoms` (spec §4); a Kohn–Sham provider solves now (`Solved`), a provider with a background teacher queues it (`Queued`, the payload arrives as a `SideKind::Harris` `SideDisplay` with that job number); `None`: no orbitals here. Never a solve of the energy, never a sample.
- `SideKind::Harris`; `SideDisplay::display: Option<Display>` (`None` for Harris). The two existing constructors in `audit.rs` wrap theirs in `Some`.
- `pub const HARRIS_SAME_GEOMETRY: f64 = 1e-6;` (Bohr, max |ΔR| per nucleus).
- `fn side_replaces(shown: Option<(&OrbitalDisplay, u64)>, incoming: &OrbitalDisplay, job: u64) -> bool` (private, in `dynamics.rs`): same nucleus count and positions within `HARRIS_SAME_GEOMETRY` → `Converged` beats `Harris` whatever the job order, and within one kind the newer job wins; otherwise the newer job wins. Replaces the `older` test at :1809; `answers_view` still applies.
- `poll_side_display` drains `take_side_display` until `None` (a provider may hold a DFT and a Harris display at once — Task 5); `side_display` (the density) is set only from `Some(display)`, so `SideDensity::kind` is never `Harris` (the `viewer.rs`/`ui/mod.rs` arms are `unreachable!("a Harris job carries no density")`).

- [ ] Test (sim) `side_replaces_prefers_converged_at_the_same_geometry`: positions P; shown Harris job 10 at P, incoming Converged job 9 at P → true; shown Converged 9 at P, incoming Harris 10 at P → false; shown Harris 10 at P, incoming Converged 9 at P + 0.01 → false; incoming Harris 11 at P + 0.01 over Converged 9 at P → true; 2 nuclei vs 3 → newer wins.
- [ ] Test (sim, dynamics tests beside :2712) `harris_side_display_sets_orbitals_not_density`: a test provider hands a `SideDisplay { display: None, kind: SideKind::Harris, orbitals: Some(o with kind Harris) }` → `snapshot().orbitals` is it, `orbitals_side == Some(SideKind::Harris)`, `side_display` unchanged (still the earlier audit's), `measure_side_charges` never sees it.
- [ ] Test `both_side_displays_of_one_tick_are_taken`: provider holds an Audit display (job 5) then a Harris (job 6) → after one tick `side_display.kind == Audit` and the orbitals are the Harris job's (different geometry, newer).
- [ ] `cargo test -p sim --release` and `cargo build --workspace` green; commit `feat(sim): Harris display kind, the harris_display entry point, side precedence`.

### Task 4: ρ_in and `GpuKs::harris_display` (Model: Opus; physics review before merge)

**Files:** create `crates/providers/src/harris.rs` (module doc: spec §4.1's ρ_in rule, the rescale, the partial core, the VRAM term); modify `crates/providers/src/lib.rs`, `crates/providers/src/ks.rs` (`problem` ≈ :356 becomes `pub` for the validate suite; `GpuKs` gains `harris: HarrisState`; factor `orbital_payload`'s tail from "the product textures" to the shares into a helper both payloads use; `harris_footprint_bytes` beside `orbital_view_footprint_bytes` ≈ :328), `crates/providers/src/orbitals.rs` (`UnoccupiedState` learns that the view's unoccupied vectors are a Harris solve's: the next converged payload clears them before its own plan), `crates/sim/src/activity.rs` (the display slot gets a kind).

**Interfaces:**
- Consumes Task 2 (`GpuSolver::harris`, `HarrisState`, `HarrisInput`, `adopt_unoccupied`) and Task 3 (`HarrisAtoms`, `HarrisAnswer`, `OrbitalKind`, `AtomProfile`).
- Produces:
  - `pub struct RhoIn { pub up: Vec<f64>, pub dn: Vec<f64>, pub rescale: f64, pub per_atom: Vec<f64> }` (`per_atom`: each atom's on-grid electrons before renormalisation, for the debug readout).
  - `pub fn superposition(grid: &qm::Grid, nuclei: &[qm::Nucleus], profiles: &[AtomProfile], counted: &[f64], n_up: usize, n_dn: usize) -> Result<RhoIn, String>`: per atom, splat `p.at(|r − R_A|)` within `p.reach()` onto the interior nodes (f64), S_A = h³ Σ (Neumaier), scale by `counted[A] / S_A`; sum; `rescale = (n_up + n_dn) / Σ counted`; ρ_σ = rescale · Σ · n_σ / N. `Err` when a species has no profile, an S_A ≤ 0, or N = 0 ("no electrons").
  - `pub fn harris_footprint_bytes(n: usize, n_up: usize, n_dn: usize, m: usize, pseudised: bool) -> u64`: fields = Σ_σ (n_σ + m + `UNOCC_GUARD`) + 13 (ρ_in↑↓, ρ, V_H, v_nuc, v↑↓, exc, xc↑↓, ρ_out↑↓ and one scratch) + 3 if pseudised (ρ_pc and the two core inputs) + 8(m + `UNOCC_GUARD`) (`LobpcgWork` growth), × `field_bytes(n)` × 1.2.
  - `impl ForceProvider for GpuKs { fn harris_display(..) -> Option<HarrisAnswer> }`: `problem(scene, tol, notes)`; the solver for the scene's grid (the rule `evaluate` uses to build or rebuild it; a rebuild drops `harris` and the view); the GPU memory guard with `solve_footprint_bytes + orbital_view_footprint_bytes + harris_footprint_bytes` → over budget: `Solved(Ok(orb.refused(.., note)))`, no solve; `counted` = Z or `IonicSetup::valence_charge(z)`; `superposition` (its `Err` → `Solved(Ok(payload with that note and no field))`); the probe's display slot with kind Harris; `solver.harris`; `adopt_unoccupied`; the shared payload helper on `harris.state()`; `kind = Harris { experience: None, rescale, wall_s }`; a note when |rescale − 1| > 1e-9; abort → `Solved(Err(ProviderError::Cancelled))`. It never sets `density_current`, never touches `last`, never starts the probe's solve slot, and keeps one `OrbitalView` with the converged payloads (tracks continue between them).
  - `sim::activity`: `pub enum DisplayJob { Unoccupied(u32), Harris }`; `ActivityProbe::display_solve(self: &Arc<Self>, job: DisplayJob) -> DisplayBusy` (the existing call sites pass `Unoccupied(m)`); `ProbeView::display: Option<(f64, DisplayJob)>`; `DisplaySolve` gains `job: DisplayJob` (one slot: the two never overlap on one teacher, and only the auditor's teacher runs Harris).

- [ ] Test (providers) `superposition_integrates_to_the_valence_count`: two analytic 1s profiles (as `cloud.rs`'s test builds them) at off-grid positions, 61³ × 0.2, counted [1, 1], n = (1, 1) → h³Σρ_σ = 1 each to 1e-10, `rescale` = 1; with counted [6, 1] (a pseudo-O-like valence count) h³Σρ = 7 to 1e-10.
- [ ] Test `charged_scene_is_rescaled_and_labelled`: H₃⁺ (`validate` common's triangle, side 2.1) on the GPU teacher with H profiles → payload `kind == Harris { rescale ≈ 2/3 (1e-12), .. }` and a note naming the rescale.
- [ ] Test `no_electrons_is_a_note_not_a_solve`: H⁺ alone → `Solved(Ok(p))`, `p.field.is_none()`, `p.notes` names "no electrons", the probe's display slot never started.
- [ ] Test `harris_is_not_a_teacher_solve`: GPU teacher with probe slot Side: one `harris_display` → `probe.view().side.solves` unchanged, `display_solves` +1, the last `DisplaySolve.job == DisplayJob::Harris`; `sample_density` answers as before the call (bitwise).
- [ ] Test `harris_over_the_vram_budget_is_a_note`: budget = solve + view footprints + 1 byte → refused payload with the GB note, no solve.
- [ ] Test `converged_payload_after_harris_never_shows_harris_unoccupied`: m = 2: Harris payload, then `evaluate` + `orbital_display_now` at another geometry → its unoccupied rows are its own solve's or none, never the Harris vectors (their ε differ from the Harris payload's).
- [ ] `cargo test -p providers --release` green; physics review (ρ_in normalisation and rescale, the valence count, the partial core path); commit `feat(providers): Harris display orbitals on the GPU teacher`.

### Task 5: The auditor's Harris job (Model: Opus)

**Files:** modify `crates/providers/src/surrogate/audit.rs` (`JobKind` ≈ :396 region, `AuditJob`, the thread loop ≈ :253–323, `pump_side_queue` ≈ :470, `handle_answer`), `crates/providers/src/surrogate/mod.rs` (fields; `ForceProvider for SurrogateKs`; `take_side_display` ≈ :2394; `side_job_in_flight` ≈ :3161; module doc "Side densities" and "Orbitals").

**Interfaces:**
- Consumes Task 3's `harris_display`, `HarrisAnswer`, `SideKind::Harris`; Task 4's `GpuKs::harris_display`.
- Produces:
  - `JobKind::HarrisDisplay`; `AuditJob::harris: Option<Arc<HarrisAtoms>>`; `SurrogateKs::harris_queued: Option<(Scene, Arc<HarrisAtoms>, u64)>` (at most one; a newer request replaces it); `SurrogateKs::side_harris: Option<SideDisplay>` beside `side_display`.
  - `SurrogateKs::harris_display(scene, atoms, _)`: `None` without an auditor, for an empty scene, or while a sampling pass is active; else `side_jobs += 1`, replace `harris_queued`, `pump_side_queue`, `Some(Queued { job })`.
  - `pump_side_queue`: the `side_queue` first (refreshes, atoms, references) as today; the Harris job only when that queue is empty and no audit is due (`!audit_due`).
  - The auditor thread, for `HarrisDisplay`: no `reset`/`discontinuity`, `last_atom` untouched; the job's view is set as for a refresh; abort = generation changed **or** the display interrupt is raised; calls `teacher.harris_display`. For `Atom`/`Reference` jobs the partial core is always off (`teacher.set_display_partial_core(job.core && !atom)`): every `AtomProfile` is the valence density (spec §4.1; the cloud's superposition is then valence-only with "show partial core" on — document it in the cloud's label doc).
  - `handle_answer`: `(HarrisDisplay, Ok(orbitals: Some(o)))` → `side_harris = Some(SideDisplay { display: None, orbitals: Some(o), kind: SideKind::Harris, .. })`; an error or `Cancelled` → a pending note only (no `TeachEvent`, no counter). `take_side_display` returns `side_display` first, then `side_harris`. `side_job_in_flight` is true while a Harris job is queued or running.

- [ ] Test (providers surrogate tests) `harris_job_is_never_a_fallback_or_audit`: estimated H₂ on the GPU teacher with H profiles: `harris_display` → `Queued`; poll until the side display arrives → `kind == Harris`, the teach events, audit history, fallback counts and `samples()` are unchanged.
- [ ] Test `a_newer_harris_request_replaces_the_queued_one`: auditor busy with a refresh; three `harris_display` calls at R = 1.40, 1.45, 1.50 → exactly one Harris answer, at 1.50.
- [ ] Test `a_due_audit_goes_before_harris`: `audit_due` set and a Harris queued when the auditor frees → the audit is dispatched first.
- [ ] Test `harris_never_resets_the_auditor_teacher`: a refresh at R, a Harris at R + 0.05, a refresh at R → the second refresh's SCF iterations equal a warm refresh's without the Harris between (the teacher's warm state survived).
- [ ] Test `atom_jobs_solve_without_the_partial_core`: `set_display_partial_core(true)`, `request_atom_density(8, accepted scene)` → its density integrates to Z_v = 6 (± the grid's 1e-2), not 6 + ∫ρ_pc.
- [ ] `cargo test -p providers --release` green; then (gate for the queue change) `cargo test -p validate --release --test surrogate` green; commit `feat(providers): Harris display jobs on the background auditor`.

### Task 6: The worker's schedule, profiles, Model only and the setting (Model: Opus)

**Files:** create `crates/app/src/worker/harris.rs`; modify `crates/app/src/worker/mod.rs` (call `harris_jobs` beside `cloud_jobs` ≈ :981, the idle wait's wake beside `views_wake` ≈ :879, `cloud_jobs` requests atom densities when Harris needs them), `crates/app/src/worker/commands.rs` (`Command::SetModelOnly(bool)`, `Command::SetOrbitalRefresh(Duration)`, `orbital_view` ≈ :487), `orbital_retry` ≈ :1013, `crates/app/src/ui/actions.rs` (`Action::OrbitalRefresh(f64)`, replay word `orbital-refresh <s>`; `model_only_on/off/by_hand` push `SetModelOnly`), `crates/app/src/ui/mod.rs` (`UiState::orbital_refresh_s`, default 2.0; persisted like `cloud_refresh_s` if that is), `crates/app/src/ui/dock/view.rs` (`cloud_controls`: "refresh orbitals every N s" slider 1–10 s beside the cloud refresh, estimated only), `crates/app/src/replay/tests.rs`.

**Interfaces:**
- Consumes Task 3's `ForceProvider::harris_display` (through `Dynamics::provider_mut()`), `HarrisAtoms`, `OrbitalKind::Harris { wall_s, .. }`.
- Produces:
  - Constants: `HARRIS_BUDGET = 0.15`, `HARRIS_MIN_INTERVAL = 1 s`, `HARRIS_DEBOUNCE = 300 ms`, `HARRIS_EVERY_DEFAULT = 2 s`.
  - `pub(super) fn harris_view(what: &OrbitalRequest) -> bool`: true for `Orbital`, `AllColoured`, `SpinDensity`; false for `Off`, `List` (the Density view never starts one).
  - `pub(super) fn harris_eligible(estimated_gpu: bool, what: &OrbitalRequest, scene: &Scene, profiles: &[AtomProfile], pass_active: bool) -> Result<(), &'static str>` with the notes `HARRIS_WAITING_ATOMS` ("orbital display: waiting for the isolated-atom densities (one solve per species, shared with the cloud)") and `HARRIS_PAUSED_PASS` ("orbital display solves paused: the sampling pass is using the background solver").
  - `pub(super) struct HarrisSchedule` with `fn due(&mut self, now: Instant, paused: bool, positions: &[[f64; 3]], view_changed: bool) -> bool`, `fn sent(&mut self, now: Instant, positions: &[[f64; 3]])`, `fn landed(&mut self, wall: Duration)`, `fn set_every(&mut self, d: Duration)`, `fn wake(&self) -> Option<Instant>`. Rules: `view_changed` → due now; paused → due once the positions differ from the last sent by > `HARRIS_SAME_GEOMETRY` and the last change is ≥ `HARRIS_DEBOUNCE` old; playing → due when the positions changed and the time since the last send ≥ max(`HARRIS_MIN_INTERVAL`, every, last_wall / `HARRIS_BUDGET`).
  - `pub(super) enum ViewJob { Refresh, Harris, Nothing }` and `pub(super) fn view_change_job(model_only: bool, harris_ok: bool, on: bool) -> ViewJob`: Model only → `Harris` (or `Nothing` while not eligible — never `Refresh`); otherwise `Refresh` as today. `orbital_view` and `orbital_retry` both use it.
  - `Worker::harris_jobs(&mut self, snap: &DynSnapshot)`: eligibility; `HarrisAtoms` rebuilt only when `self.profiles` (fitted to the scene's key) changes; `due` → `provider_mut().harris_display(&snap.scene, &atoms, &mut || false)`; `Queued` → `sent`; a new `snap.orbitals_seq` with `OrbitalKind::Harris { wall_s, .. }` → `landed`. The eligibility note, when not `Ok` and an orbital view is on under the estimated GPU backend, is published as the app snapshot's new `harris_note: Option<&'static str>` (`crate::sim::Snapshot`) for the labels (Task 7).
  - `cloud_jobs`: asks for a missing atom density when `cloud_superposition || (estimated && harris_view(&snap.orbital_view.what))`.

- [ ] Test (app, pure) `debounce_fires_once_after_the_last_move`: paused; positions change at t = 0, 50, …, 500 ms → `due` false until t = 800 ms, true once, then false.
- [ ] Test `playing_rate_follows_budget_and_setting`: every = 2 s, last_wall 0.6 s → the interval is 4 s (0.6 / 0.15); last_wall 0.1 s → 2 s; every = 1 s, wall 0.05 s → 1 s (the floor).
- [ ] Test `model_only_view_paths_never_request_a_dft_refresh`: `view_change_job(true, true, true) == Harris`, `(true, false, true) == Nothing`, `(false, _, true) == Refresh`; and in `worker/tests.rs` with the estimated GPU provider (release): Model only on, view → All orbitals coloured, a paused edit → the Activity log has no "DFT fallback"/refresh solve, a Harris payload arrives.
- [ ] Test `harris_stops_with_the_view_and_the_pass`: `harris_eligible` with `Off`, `List`, the Density mode's request, or `pass_active` → `Err`; with a species lacking a profile → `Err(HARRIS_WAITING_ATOMS)`.
- [ ] Test (replay) `orbital_refresh_round_trips`: `Action::OrbitalRefresh(3.0)` ↔ `"orbital-refresh 3"`; an existing replay script still parses.
- [ ] `cargo test -p app --release --bin app -- worker:: ui::tests replay` green; commit `feat(app): schedule Harris orbital display solves; Model only and the refresh setting`.

### Task 7: Labels, Activity, docs wording (Model: Opus)

**Files:** modify `crates/app/src/viewer.rs` (`OrbitalShown` ≈ :155–215), `crates/app/src/ui/dock/orbitals.rs` (notes ≈ :37–70, the ε/HOMO/LUMO/gap cells, the details readout), `crates/app/src/ui/dock/view.rs` (≈ :182 "show" info, :194–223), `crates/app/src/ui/activity/jobs.rs` (≈ :125–134), `crates/app/src/ui/toolbar.rs` (`MODEL_ONLY_TIP` :41), `crates/app/src/ui/tests.rs`, `README.md` ("Orbitals and spin", the estimated-mode paragraph ≈ :42), `docs/dev-reference.md` ("Orbital viewer").

**Interfaces:**
- Consumes `OrbitalKind::Harris`, `DisplayJob::Harris`, Task 6's eligibility notes.
- Produces (exact strings are the builder's within spec §6; these substrings are pinned by tests):
  - `pub const HARRIS_NOTE` ⊃ "not self-consistent"; `HARRIS_EPS_TIP` ⊃ "approximate"; `HARRIS_SHAPES` ⊃ "qualitative"; `HARRIS_DENSITY_NOTE` ⊃ "this solve's output, not a DFT density"; `HARRIS_UNOCC_NOTE` ⊃ "box states"; `HARRIS_RESCALED` ⊃ "rescaled to the scene's".
  - `OrbitalShown` gains `harris: Option<(f64 /*rescale*/, Option<ExperienceUse>)>` and `current: bool`; `fade()` is 1.0 for a Harris payload at the current positions (`HARRIS_SAME_GEOMETRY`), `SURROGATE_DENSITY_FADE` otherwise; `label` for Harris: "orbitals solved once at t = {t:.2} fs in the potential of superposed atoms, not self-consistent …", "+ experience on k of n atoms" when `experience` is `Some` (Phase B), the rescale sentence when |rescale − 1| > 1e-9, "(an earlier geometry)" when not current; with no Harris payload yet and `Snapshot::harris_note` set, the label is that note (instead of the faded "last DFT solve" text alone).
  - ε, HOMO, LUMO and gap of a Harris payload print with "≈" and carry `HARRIS_EPS_TIP`; the details readout shows the rescale factor, the per-atom electrons before renormalisation are not shown (log only), and the solve's wall.
  - jobs.rs: `DisplayJob::Harris` → title "Orbital display solve", detail "orbital display solve (not self-consistent) · {s:.1} s"; never "DFT fallback".
  - `MODEL_ONLY_TIP` adds: opening Orbitals, All orbitals coloured or Spin density runs display-only orbital solves (not self-consistent, no SCF, nothing learned) plus the one isolated-atom solve per species the cloud already makes; no DFT for forces or motion.

- [ ] Test (ui) `harris_labels_say_what_they_are`: a snapshot with a Harris payload at the current positions → `fade() == 1.0`, label ⊃ `HARRIS_NOTE`, no "last DFT solve"; positions moved 0.1 → faded, label ⊃ "earlier geometry"; rescale 0.667 → ⊃ `HARRIS_RESCALED`; no payload and `harris_note = Some(HARRIS_PAUSED_PASS)` → the label is that note.
- [ ] Test `harris_eps_cells_are_approximate`: the Orbitals list of a Harris payload renders ε text starting "≈" and the hover text ⊃ `HARRIS_EPS_TIP`; a Converged payload's has no "≈".
- [ ] Test `harris_job_row_is_not_a_fallback`: probe view with `display = Some((0.3, DisplayJob::Harris))` → one row titled "Orbital display solve", detail ⊃ "not self-consistent"; no row titled "DFT fallback".
- [ ] README: "Orbitals and spin" gains a paragraph on live orbitals in estimated mode (what is solved, the labels, Model only's promise, the refresh setting, costs filled in by Task 8); the estimated-mode paragraph mentions it; `docs/dev-reference.md` "Orbital viewer" gains the Harris path, the `--test harris` gate and the `orbital-refresh` replay word.
- [ ] `cargo test -p app --release --bin app -- ui::tests` green; commit `feat(app): label Harris display orbitals honestly; docs`.

### Task 8: `--test harris`: V-A1, V-A2, V-A3 (Model: Opus; physics review before merge)

**Files:** create `crates/validate/tests/harris.rs`; modify `crates/validate/references.md` (new section "Live orbitals (Harris display; spec 2026-11-14)"), `README.md` (the cost sentence), `docs/dev-reference.md` "Test suites", `CLAUDE.md` (the suite table row: `--test harris`, its time).

**Interfaces:**
- Consumes `qm_gpu::{GpuSolver, HarrisState, HarrisInput}`, `providers::harris::superposition`, `providers::ks::problem` (pub, Task 4), `sim::{AtomProfile, radial_profile}`, `GpuKs::harris_display`.
- Helper `fn atom_profile(ctx: &GpuContext, z: u8, scene: &Scene) -> AtomProfile`: the isolated neutral atom on the scene's cell (σ, ionic policy) with `GpuKs` (display on, partial core off), read back, `radial_profile` — the app's Atom-job path.
- Helper `fn overlap(h: &qm::KsState, s: &qm::KsState) -> [f64; 2]`: per channel (1/n_σ)‖C‖_F², C_ij = h³⟨ψ_i^H|ψ_j^SCF⟩ over the f > 0 levels (f64, Neumaier).
- Geometries (inline; cited in `references.md`): H₂ R = 1.4; H₂O O–H 1.809 Bohr, ∠HOH 104.5°; NH₃ N–H 1.913 Bohr, ∠HNH 106.7°; H₂ + H: H₂ R = 1.4 on x centred at 0, H at (3.0, 0, 0), multiplicity auto (doublet); H₃⁺ equilateral side 2.1, charge +1; pseudo-O: one O at the origin. Grids: H₂, H₂ + H, H₃⁺ at 81³ × 0.15; H₂O, NH₃, pseudo-O at 97³ × 0.15; ionic `Accepted` (the app default), tol 1e-6.

- [ ] Test `va1_identity_on_the_converged_density` (gate): H₂, H₂O, pseudo-O: SCF, then `harris` with ρ_in = the state's ρ_up/ρ_dn → max occupied |Δε| ≤ 1e-5 Ha, `overlap` ≥ 1 − 1e-6 per channel. Prints both per case.
- [ ] Test `va2_superposed_atoms_against_scf` (record + gate): H₂, H₂O, NH₃, H₂ + H, H₃⁺: profiles, `superposition`, `harris` vs SCF → print overlap, ε error per level, ∫|ρ_out − ρ_SCF| h³ / N; assert overlap ≥ 0.95 per channel for every case **except** H₃⁺ (recorded only). A failure is investigated, never loosened.
- [ ] Test `va3_warm_refresh_cost` (record): through `GpuKs::harris_display` on one teacher: a cold call, then five warm calls each after moving atom 0 by +0.01 Bohr; print min/median wall (superposition, solve and payload separately) for H₂ and H₂ + H at 81³ and H₂O at 97³; record against the ≤ 0.3 s target (no assert).
- [ ] Run `cargo test -p validate --release --test harris` (one heavy job) → V-A1, V-A2 pass; numbers into `references.md` (date, hardware, grid, every case's rows, the target met or not and what the cadence then does), the README cost sentence, dev-reference "Test suites" and the CLAUDE.md row.
- [ ] Physics review of the numbers and the ε gap's documentation (rule 7); commit `test(validate): live orbitals V-A1–V-A3 (Harris identity, superposition, cost)`.

### Task 9: V-A4, bonds from the Harris density (record only) (Model: Opus; physics review before merge)

**Files:** create `crates/sim/src/bond_experience.rs` (move the pure part of `crates/app/src/bond_experience.rs` unchanged: `Example`, `Key`, `pair_key`, `neighbour_counts`, `examples`, `Store`, `Verdict`, `MATCH_R`, `MIN_AGREE`, `COUNT_CAP`, `CAP` and their unit tests; the app module keeps the file I/O, `Place`, `compose`, `Experience` and re-exports the rest); modify `crates/validate/tests/harris.rs`, `crates/validate/references.md`, `README.md` (one sentence pointing at the table, under "Orbitals and spin").

**Interfaces:**
- Consumes `sim::bonds::{Detector, Promolecule}`, `sim::references::solve_atom` (the bond references for H, N, O on each cell), `sim::charges::hirshfeld`, `sim::bonds::trilinear_blend`, Task 8's helpers, `sim::bond_experience::{Store, examples, pair_key, neighbour_counts}`.
- Frames: (a) a sampling pass on H₂O (`common::surrogate::student`, `sample_and_fit` with `SamplingPlan { perturbations: 60, hot_burst_steps: 0, scan_points_per_pair: 0, ..Default::default() }`), its samples in store order: the first 40 train experience, the last 20 are held out; (b) an H₂ scan R = 0.8, 1.0, …, 6.0; (c) H₂ + H, the third H at 6.0, 5.75, …, 2.0 Bohr from H₂'s centre on its axis. For (b) and (c) even-index frames train experience, odd-index frames are held out. Grid 81³ × 0.15 (97³ for H₂O), `Accepted`.
- Per held-out frame: the SCF density and the Harris ρ_out (both read back, sampled trilinearly) each through `Detector` with the driver's thresholds (unchanged) → `BondGraph`; Hirshfeld charges of both; experience `Store::predict(pair_key(..), R)` per pair within `sim::bonds::R_MAX`.
- Scored per pair against SCF: bonded/not-bonded agreement; `BondKind` agreement; |Δρ_bcp|; |Δ enhancement| (the pile-up); per atom |Δq|. Experience: coverage (answers at all: ≥ `MIN_AGREE` agreeing) and agreement where it answers. Split "near the bond length" = SCF bonded with ρ_bcp ≥ 0.1 e/Bohr³; "stretched/borderline" = the rest within `R_MAX` (a display split, recorded with the table).

- [ ] Run `cargo test -p sim --release bond_experience` and `cargo test -p app --release --bin app -- bond_experience` after the move → unchanged results.
- [ ] Test `va4_harris_bonds_against_scf_and_experience` (record only, no assert beyond "every frame solved"): prints one table per frame set and split: pairs, Harris agreement %, kind agreement %, mean |Δρ_b|, mean |Δ pile-up|, mean |Δq|, experience coverage %, experience agreement %.
- [ ] Run `cargo test -p validate --release --test harris va4` (one heavy job); the table into `references.md` "Live orbitals" with "measurement for spec §9 Q2, not a gate"; the README sentence. Physics review of the scoring; commit `test(validate): V-A4 bonds from the Harris density vs SCF and bond experience (record)`.

### Task 10: Gates on the merged branch (Model: Opus, `validation-runner`; no edits to physics)

**Files:** none (numbers into `references.md` only if a suite's recorded timing changes).

- [ ] Serially, one heavy job at a time, each green:
  `cargo test -p units -p qm -p sim -p surrogate --release`; `cargo test -p providers --release`; `cargo test -p qm-gpu --release`;
  `cargo test -p validate --release --test gpu_vs_cpu`; `--test surrogate`; `--test orbitals`; `--test harris`;
  `cargo test -p app --release` (the replay scripts included).
- [ ] Hand-test list for the user (do not block on it): estimated H₂ + H playing with All orbitals coloured (orbitals refresh, dent as the H approaches, labels say "not self-consistent", ≈ on ε); Model only on → no DFT row in Activity, only "Orbital display solve"; a paused drag → one solve 0.3 s after letting go; H₃⁺ label names the rescale; a sampling pass shows the pause note.
- [ ] DEVLOG entry at the merge (memory "DEVLOG upkeep"); README milestone line.

**Follow-on (not this plan):** Phase B (spec §5) starts with §5.1b's pre-registered spike (ℓ, W_min on one H₂O sampling pass; stop if ℓ ≤ 0.5 Bohr gives < 30 % held-out L1 gain), then its own plan; it fills `ExperienceUse` and V-B1–V-B3.
