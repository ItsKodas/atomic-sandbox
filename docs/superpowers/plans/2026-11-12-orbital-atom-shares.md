# Orbitals by atom — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every orbital row shows how much of it sits on each atom (Hirshfeld shares), the list is grouped by atom, and selecting atoms dims the rows that are not mostly on them.

**Architecture:** `sim` owns the Hirshfeld weights. It already builds them for the partial charges from its private `AtomReferences`, and it hands the provider one radial table per nucleus as plain data, so the provider needs no `refs` (CLAUDE.md rule 5). The provider computes s_iA for the rows it displays — on the CPU from `CpuView`'s ψ̃ (f64), on the GPU with one new reduction kernel — averages them over degenerate sets, and ships them on `OrbitalInfo`. The app groups and dims rows from the shares and the selection. Display only.

**Tech Stack:** Rust; `sim`, `providers`, `qm-gpu` (WGSL), `app` (egui 0.36); `validate` orbital suite.

**Spec:** `docs/superpowers/specs/2026-11-12-orbital-atom-shares-design.md` (binding; read its "Calls made after the builder check").

## Global Constraints

- All tasks on **Opus** (user memory: no Fable); `physics-reviewer` (Opus) reviews Tasks 1 and 3 before they land.
- Branch `docking`, worktree `D:\Projects\Physics\.claude\worktrees\docking`, after docking Task 11 completes; one heavy cargo job at a time; never kill the user's `app.exe`; build at BelowNormal `-j 4` when another job runs.
- Rule 1: no per-element data or thresholds. Rule 3: atomic units inside, % only in the UI. Rule 4: the GPU kernel in f32; the share sums are compensated (Neumaier `Acc` on the CPU; pairwise/Kahan on the GPU). Rule 7: the tooltip says the share is a split of space (Hirshfeld), not measured.
- Display only: nothing reaches a solve, a force, the surrogate or a sample; nothing becomes a `ui::Action`.
- Fixed values (from the spec): majority **0.55** (grouping and selection); shared-heading atoms have share **≥ 0.25**, largest first, ties by atom number; hover lists atoms with share **≥ 0.01**; the shares' own check is |Σ_A s_iA − 1| ≤ `sim::charges::SUM_TOL` (1e-3), else no shares for that payload.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git commit -o -- <paths>`; README (Orbitals bullet) and `docs/dev-reference.md` (Orbital viewer line) updated in Task 4; `crates/validate/references.md` gets the oracle numbers in Task 5.

## Review Focus

1. A scene edit (atom added or removed) between the tables being sent and the payload being built: the tables no longer match the nuclei → no shares (never shares indexed to the wrong atoms). Test in Task 2.
2. A pseudised atom (`Accepted` policy: H and O pseudised): its table is the pseudo-atom's valence density, the orbitals are valence-only, and shares still sum to 1. Test in Task 5 (H₂O, accepted).
3. A degenerate set split across the 0.55 line: every member gets the same averaged share and the same heading. Test in Task 2 (synthetic set) and Task 5 (CH₄ t₂).
4. Estimated mode: the side solve's geometry differs from the scene's → the teacher uses its own job positions with the same per-nucleus tables. Test in Task 2.
5. During play, H₂'s σ_g stays `shared` on every step (exact tie, grid noise): a pure function with the 0.55 margin. Test in Task 4.

---

### Task 1: The Hirshfeld weights as data, and the CPU share reference (Opus; physics review)

**Files:** modify `crates/sim/src/charges.rs` (factor the per-node weights out of `hirshfeld`), `crates/sim/src/provider.rs` (`ShareTables`, `OrbitalInfo::shares`), `crates/sim/src/lib.rs` (exports).

**Interfaces — produces:**
- `pub struct ShareTables { pub tables: Vec<std::sync::Arc<RadialDensity>>, pub z: Vec<u8> }` in `sim::provider`: one free-atom table per nucleus in scene order (`tables[i]` for nucleus `i`), with `z` to detect a changed scene.
- `pub struct Weights<'a>` in `sim::charges`, `pub fn Weights::new(tables: &'a [&'a RadialDensity], pos: &'a [[f64; 3]]) -> Self` and `pub fn Weights::at(&self, r: [f64; 3], out: &mut [f64])`: w_A(r) for every A, Σ_A w_A = 1; the nearest atom takes 1 where Σ ρ_B⁰ underflows (today's rule). `hirshfeld` uses it, with unchanged results.
- `pub fn orbital_shares(w: &Weights, cell: &Cell, psi: &[f64]) -> Vec<f64>`: s_A = h³ Σ_nodes w_A |ψ|², Neumaier-summed, **not** renormalised (the caller checks Σ against 1).
- `pub fn set_average(rows: &[OrbitalInfo], shares: &mut [Option<Vec<f64>>])`: within each channel, rows of one degenerate set (consecutive indices with `cluster == Some((d, _))`, d rows) get their members' mean.
- `OrbitalInfo` gains `pub shares: Option<Vec<f64>>` (per nucleus; `None` = not computed). Every existing constructor sets `None`.

- [ ] Test `hirshfeld_is_unchanged_by_the_weights_refactor`: today's charges for H₂ and HeH⁺ fixtures (copy the numbers the current code gives before refactoring) equal to 1e-12.
- [ ] Test `orbital_shares_sum_to_the_norm`: a normalised synthetic ψ on a 33³ grid across two H tables → |Σ s − 1| ≤ 1e-10; ψ mirror-symmetric between the two nuclei → |s₀ − s₁| ≤ 1e-12.
- [ ] Test `a_far_atom_owns_its_orbital`: two H tables 12 Bohr apart, ψ = a normalised 1s-like Gaussian on nucleus 0 → s₀ ≥ 0.999.
- [ ] Test `set_average_shares_one_answer`: three rows with `cluster = Some((3, _))`, shares (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) → all three (1/3, 1/3, 1/3); a non-degenerate row is untouched; ↑ and ↓ are averaged separately.
- [ ] `cargo test -p sim --release` green; physics review (weights, normalisation, the underflow rule, pseudised tables); commit `feat(sim): Hirshfeld weights as data; orbital shares (CPU reference)`.

### Task 2: Tables to the provider; CPU shares on the payload (Opus)

**Files:** `crates/sim/src/provider.rs` (`ForceProvider::set_share_tables`), `crates/sim/src/dynamics.rs` (build and send the tables), `crates/providers/src/ks.rs` and `crates/providers/src/orbitals.rs` (CPU shares), `crates/providers/src/surrogate/mod.rs` + `audit.rs` (forward to the teacher).

**Interfaces:**
- Consumes Task 1's `ShareTables`, `Weights`, `orbital_shares`, `set_average`, `OrbitalInfo::shares`.
- Produces `fn set_share_tables(&mut self, t: Option<std::sync::Arc<ShareTables>>) {}` on `ForceProvider` (default no-op). `Dynamics` sends `Some` whenever the reference tables for every nucleus of the current scene exist (the same `ReferenceKey` the charges use, `refresh_bonds` ≈ dynamics.rs:2886), and `None` when any is missing or the scene's Z list changes; it resends after a scene edit.
- `CpuKs::orbital_payload` (ks.rs ≈546): with tables whose `z` equals the evaluated scene's Z list, compute shares for each displayed row from `CpuView.prev[ch][row.index]` at the evaluated positions; if any row's |Σ s − 1| > `SUM_TOL`, set every row's `shares = None` and log once ("orbital shares off: Σ = …"); then `set_average`.
- `SurrogateKs` forwards the tables to its teacher(s); the side solve uses its job's positions (the teacher evaluates the job's scene).

- [ ] Test (providers) `cpu_payload_carries_shares`: `CpuKs` H₂ 33³ with tables for both H → σ_g shares within 1e-6 of 0.5/0.5, Σ = 1 within 1e-9.
- [ ] Test `stale_tables_give_no_shares`: tables for [1, 1], scene changed to [1, 1, 1] before the payload → every row's `shares` is `None`.
- [ ] Test `side_solve_shares_use_the_job_geometry`: the estimated provider's side display for an H₂ job at R = 1.6 with the scene at R = 1.4 → shares symmetric (computed at the job's positions).
- [ ] Test (sim) `dynamics_sends_tables_when_references_exist`: a test provider records `set_share_tables` calls; before the references `None`, after them `Some` with `z` = scene; after adding an atom, `None` until its reference exists.
- [ ] `cargo test -p sim -p providers --release` green; commit `feat(providers): orbital shares on the CPU payload; tables from the driver`.

### Task 3: The GPU share kernel (Opus, `gpu-kernel-engineer`; physics review)

**Files:** create `crates/qm-gpu/src/shaders/orbital_shares.wgsl`; modify `crates/qm-gpu/src/orbitals.rs` (a `shares` pass in `OrbitalView` beside `orbital_moments`, returned in `ViewUpdate`), `crates/providers/src/ks.rs` (`GpuKs::orbital_payload` ≈834 fills `shares` like the CPU path).

**Interfaces:**
- Produces `pub fn OrbitalView::shares(&mut self, solver: &GpuSolver, tables: &ShareTablesGpu, positions: &[[f64; 3]]) -> Vec<[Vec<f64>; 2]>` (per channel, per shown row, per nucleus) — one readback of N_rows × N_atoms scalars, no field crosses to the CPU (module doc rule). `ShareTablesGpu`: the radial tables packed into one storage buffer (offset, length, r-step per atom), uploaded when the `Arc<ShareTables>` changes.
- The kernel evaluates w_A at each node from the packed tables with the same linear interpolation and underflow rule as `Weights::at`, multiplies by ψ̃², and reduces per (row, atom) with the existing `Reducer` pattern (workgroup partials, f32, then pairwise on the CPU in f64).

- [ ] Test (qm-gpu or providers, release) `gpu_shares_match_cpu`: the same H₂ 49³ state on both providers → |s_gpu − s_cpu| ≤ 1e-4 per row and atom; Σ_A s_gpu within 1e-4 of 1.
- [ ] Test `gpu_shares_cost_is_reported` (`#[ignore]`-free, report-only): H₂O 95³, 8 rows × 3 atoms, print the kernel wall time; assert ≤ 50 ms on this machine (it must not delay a step: payloads are already off the stepping path).
- [ ] `cargo test -p qm-gpu -p providers --release` green; physics review; commit `feat(qm-gpu): orbital shares on the GPU`.

### Task 4: The Orbitals list grouped by atom, with selection dimming (Opus)

**Files:** `crates/app/src/ui/dock/orbitals.rs` (`list`, `row`; a pure grouping fn), `crates/app/src/ui/dock/selection.rs` (extract the atom name `"O #0"` into `pub(crate) fn atom_name(z: u8, k: usize) -> String` and use it in both panels), `crates/app/src/ui/tests.rs`, README, dev-reference.

**Interfaces:**
- `pub fn groups(rows: &[&OrbitalInfo], z: &[u8], unocc_current: bool) -> Vec<Group>` where `pub enum Owner { Atom(usize), Shared(Vec<usize>), Box, Earlier, Unknown }` and `pub struct Group { pub owner: Owner, pub rows: Vec<usize> }`. Box = `box_state` rows; Earlier = unoccupied rows while `unoccupied.current` is false; Unknown = `shares == None`. Owner order: atoms by number, then shared groups by their first atom, then Box, Earlier, Unknown. Row order within a group: today's (channel structure, occupied before unoccupied, by energy). The ↑↓ fold (`paired`, `merged`) stays and uses the ↑ member's shares.
- `pub fn lit(shares: &[f64], selected: &[usize]) -> bool`: Σ over selected ≥ 0.55; nothing selected → true.
- Selection = `PanelInput::selected` plus `UiState::edit_group`. Unlit rows draw with `theme::INK_FAINT`, stay clickable.
- Each row shows its largest share as `H #0 50 %` (no column in a one-nucleus scene: goal 4); hover lists atoms ≥ 1 % and "a split of space between the atoms (Hirshfeld, like the partial charges), not something measured". With shares absent for the whole payload, the list is today's, plus a note: "atom shares need the atom references: being solved" (or the payload's reason).

- [ ] Test `groups_h2o_like`: synthetic rows with shares O 0.92 / O 0.60 / (O 0.5, H 0.25, H 0.25) / H 0.9 → headings `O #0` (two rows), `shared · O #0–H #1–H #2`, `H #1`.
- [ ] Test `an_exact_tie_reads_shared`: (0.5000001, 0.4999999) and (0.54, 0.46) → `Shared([0, 1])`; (0.56, 0.44) → `Atom(0)`.
- [ ] Test `degenerate_rows_stay_together` (input already set-averaged by the provider): three rows with equal averaged shares land in one group.
- [ ] Test `selection_dims_but_keeps_rows` (UI, probes): select H #1 → the H #1 rows are lit, the O rows are drawn faint and still clickable (`PickOrbital` arrives); the number of rows drawn is unchanged; deselect → nothing faint.
- [ ] Test `box_and_earlier_rows_have_no_share`: a `box_state` row and a non-current unoccupied row go under `Box` / `Earlier` and show no percentage.
- [ ] Test `play_keeps_h2_shared` (Review Focus 5): 20 payloads with σ_g shares 0.5 ± 1e-6 noise → owner `Shared` every time.
- [ ] Test `one_nucleus_lists_as_today` (spec goal 4): a He payload with shares [1.0] draws no group headings and no share text.
- [ ] `cargo test -p app --release --bin app -- ui::` green; README Orbitals bullet and dev-reference updated; commit `feat(orbitals): grouped by atom, selection dims the rest`.

### Task 5: Oracles in the orbital display suite (Opus; physics review)

**Files:** `crates/validate/tests/orbitals.rs`, `crates/validate/references.md` (a short "Orbital shares" table).

- [ ] `h2_sigma_g_is_shared_evenly` (49³, h 0.25): |s₀ − s₁| ≤ 1e-4, Σ within 1e-4 of 1 (GPU).
- [ ] `lih_li_core_sits_on_li` (`ionic none`, 57³): the lowest occupied row ≥ 0.95 on Li; record the measured value.
- [ ] `ch4_t2_members_share_one_answer` (Review Focus 3): the three t₂ rows' averaged shares are equal to 1e-12 (set average), each H's share within 1e-3 of the others (symmetry), measured at two lattice offsets.
- [ ] `h2o_accepted_shares_sum_to_one` (Review Focus 2; `Accepted` policy, 81³): every occupied row's Σ within `SUM_TOL` of 1; record the lone-pair rows' O shares.
- [ ] `cargo test -p validate --release --test orbitals` green; numbers in references.md with the gap/meaning notes (rule 7); commit `test(validate): orbital share oracles`.

Gate at the end: `cargo test -p units -p qm -p sim -p surrogate --release`, `cargo test -p providers --release`, `cargo test -p validate --release --test orbitals`, `cargo test -p app --release`.
