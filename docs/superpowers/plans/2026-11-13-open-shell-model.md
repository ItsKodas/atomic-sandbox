# One Model for Radicals and Closed Shells — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The shared collection's model fits radicals and closed shells together (body-4 degree 3, grown with the store), a failed trust recovery no longer ratchets holds, and samples the model still cannot reproduce refuse only the atoms near them.

**Architecture:** The body-4 degree D₄ becomes a `Basis` field chosen per refit by the provider from the sample floor (surrogate crate + `providers::surrogate`). `recover_trust` restores its entry state when it does not end trusted. After every fit one residual pass also yields the misfit set M; the trust rule judges the residual rows outside M plus a new `Misfit` row, the class rule's statistics leave M out, and a new per-atom `NearMisfit` refusal fences M's atoms in `evaluate`, `geometry_refusal`, `would_answer` and `atom_verdicts`.

**Tech Stack:** Rust workspace (`surrogate`, `providers`, `sim`, `app`, `validate`); tests are release-only (`#[cfg_attr(debug_assertions, ignore)]` where heavy).

**Spec:** `docs/superpowers/specs/2026-11-13-open-shell-model-design.md` (read it first; section numbers below are its).

## Global Constraints

- CLAUDE.md rules 1–8. Nothing may name an element, molecule, multiplicity or spin (rule 1). No bound or tolerance is loosened (rule 6): 1e-3 Ha/atom energy, max(2e-3, 0.1|F|) ≤ 1e-2 Ha/Bohr force, `QUARANTINE_MAX_FRACTION` = 0.1, `QUARANTINE_FRACTION` = 0.1, `FENCE_VOTE_LIMIT` = 3, `QUARANTINE_ROUNDS` = 5 all stay.
- The trust rules are safety rules: every change is visible in the Model tab's checklist and the Activity tab, none silent.
- D₄ ceiling 3, start 2; D₄ = 4 is out.
- One heavy job at a time. Never kill the user's running `app.exe`; never build into a target dir whose `app.exe` is running (check `Get-Process app -ErrorAction SilentlyContinue | Select Path` first). BelowNormal priority, `-j 4` builds are fine while a suite runs.
- No PowerShell file edits (Get/Set-Content mangle UTF-8). Edit with the Edit/Write tools or Python.
- All subagents on Opus (user instruction 2026-10-21, overrides the Fable tags).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; while another agent shares the checkout commit with `git commit -o -- <paths>`.
- README, `crates/validate/references.md`, `docs/dev-reference.md`, DEVLOG.md updated in the merge (Task 9).

## Review Focus

1. **A NaN prediction on a fitted sample** must still fail `EnergyResidual` / `ForceResidual` (as today, activity Task 3b), never be absorbed into M as "a misfit". Test: Task 5 `a_nan_residual_is_not_a_misfit`.
2. **An environment key held only by M's samples** must be refused by the class rule (it becomes an unseen key), never trusted for want of statistics. Test: Task 5 `a_key_held_only_by_misfits_is_refused`.
3. **Old and foreign model files:** a D₄ = 2 file loads and answers at 2; a refit of that provider with enough samples steps to 3; a file claiming `body4_cap` 1, 4 or a value above its own degree cap is refused, never silently refitted. Tests: Task 2 `a_body4_file_loads_at_its_own_degree`, Task 3 `a_loaded_degree_2_model_steps_up_at_its_next_refit`.
4. **Background refits across a D₄ step:** a fork that chose a new D₄ installs cleanly; a live D₄ change while the fork runs is a conflict, not a merge. Test: Task 3 `a_background_refit_carries_its_body4_degree`.
5. **A standing M re-runs the whole recovery at every refit** (recoverable_fail stays true on the all-sample residuals): the refit stays idempotent and its cost is measured, not assumed. Measured in Task 4 (attach refit with M non-empty) and checked in Task 8 (second refit gives the same holds and M).

---

### Task 1: No ratchet in the trust recovery (§4.2)

**Files:**
- Modify: `crates/providers/src/surrogate/quarantine.rs` (`recover_trust`, `note_quarantined`)
- Test: `crates/providers/src/surrogate/tests.rs` (next to `a_stray_sample_is_quarantined_then_superseded_and_trust_returns`, ~l. 924)

**Interfaces:**
- Consumes: `save_state`/`restore_state` (`trust.rs`), `recoverable_fail`, `sync_collection_status` (`shared.rs`; writes the store's codes to the collection after `refit`).
- Produces: `recover_trust(&mut self, report: FitReport) -> FitReport` with the post-condition "ends with `!recoverable_fail()` or with the store's quarantine, `degree_cap`, `species_dirty` and fit exactly as on entry". Later tasks rely on this post-condition (M is computed on the entry fit when the recovery fails).

- [ ] **Step 1: Write the failing tests**

`a_recovery_that_cannot_restore_trust_holds_nothing` (per-scene, Morse H₂ via `trained(1.2, 2.2, 100, FallbackPolicy::default())`): add 14 strays at distinct bond lengths in [1.25, 2.15] with energy offsets graded `0.2 · 0.75^k` Ha (k = 0..13) so that each quarantine round names a few, five rounds cannot hold them all within the 10 % cap, and the fit still fails. Assert after `s.refit()`:
```rust
assert!(!s.trusted());
assert_eq!(s.quarantined(), 0, "a failed recovery holds nothing");
assert!(s.recovery_note().unwrap().contains("recovery did not restore trust; no samples held"), "{:?}", s.recovery_note());
assert_eq!(s.refit().map(|r| r.n_samples).ok(), Some(114), "idempotent: every sample stays fitted");
```
Before the assertions, print the number of holds the rounds made (`println!`) so a reader can see the test exercises the ratchet; if the unchanged code holds 0 samples on this store, adjust the offsets until it holds ≥ 1 (Step 2 checks this).

`a_failed_recovery_leaves_the_collection_codes_as_they_were` (collection mode, set up as `a_reopened_collection_reproduces_the_same_fit` does with `collection_root`): the same kind of graded strays in a collection with two pre-existing `Q_HELD` samples; after `refit()` reopen the collection with `surrogate::Collection::open` and assert its `quarantine` equals the codes before the refit, and the provider's `quarantined()` is still 2.

`a_recovery_that_restores_trust_still_holds` — the existing `a_stray_sample_is_quarantined_then_superseded_and_trust_returns` must stay green unchanged (one stray is held, trust returns).

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p providers --release a_recovery_that_cannot_restore_trust_holds_nothing a_failed_recovery_leaves_the_collection_codes -- --nocapture`
Expected: FAIL on `quarantined() == 0` (the printed holds ≥ 1).

- [ ] **Step 3: Implement the entry-state restore in `recover_trust`**

At the top of `recover_trust` capture `(self.save_state(), self.quarantine.clone(), self.species_dirty, self.recovery.clone())`. Steps 1, 1b and 3 keep their early returns on success. On both remaining exits (after step 3's fallback to the capped result and after step 2), if `self.recoverable_fail()` still holds: restore the captured state, quarantine codes, `species_dirty`, and set `self.recovery = Some("recovery did not restore trust; no samples held".into())`; return the entry report. `quarantine_rounds` is unchanged (it already stops at `QUARANTINE_MAX_FRACTION`). `judge_votes` and `release_reproduced` (run before the recovery in `refit_fits`) are unchanged.

- [ ] **Step 4: Run the tests and the recovery neighbours**

Run: `cargo test -p providers --release quarantin recover stray degree_cap_retry -- --nocapture`
Expected: PASS, including `a_degree_cap_retry_that_leaves_too_few_samples_is_not_a_recovery` and `a_recovery_cap_that_spreads_the_misfit_does_not_leave_the_collection_stuck`. If the latter's pinned hold counts change, stop and report the numbers: that test is about the recovery and its new numbers need the controller's ruling, not an edit.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "fix(providers): a trust recovery that does not restore trust holds nothing (no ratchet)" -- crates/providers/src/surrogate/quarantine.rs crates/providers/src/surrogate/tests.rs
```

---

### Task 2: The body-4 degree as part of the basis (§4.1, surrogate crate)

**Files:**
- Modify: `crates/surrogate/src/descriptor.rs` (`Basis`, `Basis::new`, `body4_cap`, `BODY4_DEGREE_CAP` and its doc, module doc "Body-4 features" Q′ table)
- Modify: `crates/surrogate/src/lib.rs` (export `BODY4_DEGREE_START`)
- Modify: `crates/surrogate/src/sidecar.rs` (reader at ~l. 1146–1160, writer ~l. 678, module doc ~l. 204, test `a_body4_basis_round_trips_as_version_8`)
- Test: `descriptor.rs` tests (`body4_counts_and_layout`, the FD and rotation tests), `sidecar.rs` tests

**Interfaces:**
- Produces: `Basis { …, pub body4_degree: usize, … }` (after `body4`); `pub const BODY4_DEGREE_START: usize = 2;` `pub const BODY4_DEGREE_CAP: usize = 3;` (the ceiling); `Basis::body4_cap(&self) -> usize = self.degree_cap.min(self.body4_degree)`. `Basis::new` sets `body4_degree: BODY4_DEGREE_START`, so every per-scene basis and every existing literal keeps its layout bit for bit.

- [ ] **Step 1: Write the failing tests**

In `body4_counts_and_layout` add the D₄ = 3 table next to the D₄ = 2 one:
```rust
let q3 = |sp: &[u8]| { let mut b = Basis::new(sp); b.body4 = true; b.body4_degree = 3; b.n_independent_columns() };
assert_eq!((q3(&[1]), q3(&[1, 8]), q3(&[1, 7, 8]), q3(&[1, 6, 7, 8])), (52, 468, 1902, 5344));
```
(the D₄ = 2 asserts 38 / 284 / 1 032 / 2 688 stay). If 5 344 is not what the code gives, the module doc's estimate was wrong: pin the computed value and correct the doc and the spec's §4.1 number in the same commit.

`body4_forces_match_finite_differences` and `body4_features_are_rotation_invariant`: run each at `body4_degree` 2 and 3 (loop over both, same tolerances).

`a_body4_file_loads_at_its_own_degree` (sidecar): a `random_sidecar_with(37, true, true)` whose basis has `body4_degree = 3` round-trips (header contains `"body4_cap":3`, the read model's `basis.body4_degree == 3`, predictions equal bit for bit); the existing D₄ = 2 round trip still gives `"body4_cap":2`. Replace the old "cap 2 → 3 is refused" case with: header edited to `"body4_cap":4` refused, `"body4_cap":1` refused (both `InvalidData`); the flag case stays.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p surrogate --release body4 -- --nocapture`
Expected: compile error (`body4_degree` unknown).

- [ ] **Step 3: Implement**

Add the field and constants; `for_each_body4` reads `self.body4_cap()` as today. Sidecar reader: when `body4`, read `body4_cap`, refuse unless `BODY4_DEGREE_START <= cap && cap <= BODY4_DEGREE_CAP`, build the basis with `body4_degree = cap`, then keep the existing `cap != basis.body4_cap()` refusal (it now catches a `degree_cap` below the stated degree). Non-body-4 files get `body4_degree: BODY4_DEGREE_START`. The format version does not change (the field already exists). Update the doc comments: D₄ = min(`degree_cap`, the basis's `body4_degree`), chosen per refit by the provider between `BODY4_DEGREE_START` and `BODY4_DEGREE_CAP` (spec §4.1); the Q′ table for both degrees.

- [ ] **Step 4: Run the crate's suite**

Run: `cargo test -p surrogate --release`
Expected: PASS (fix any other `Basis { … }` literal the compiler names by adding `body4_degree: BODY4_DEGREE_START`).

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(surrogate): the body-4 degree is part of the basis (start 2, ceiling 3); a file loads at its own degree" -- crates/surrogate
```

---

### Task 3: The provider grows D₄ with the store (§4.1, providers)

**Files:**
- Modify: `crates/providers/src/surrogate/mod.rs` (`SurrogateKs` fields ~l. 1093/1183/1428, `default_basis`/`fit_basis` ~l. 2463–2486, `fit_floor` ~l. 2435 and its cache, the status builder that fills `TrainableStatus::fit`)
- Modify: `crates/providers/src/surrogate/trust.rs` (`FitState`, `save_state`, `restore_state`, `fit_store_with` rebuild decision)
- Modify: `crates/providers/src/surrogate/store.rs` (`install_sidecar`)
- Modify: `crates/providers/src/surrogate/background.rs` (`Snapshot`, `background_conflict`, fork copy ~l. 146/162, `install_fork` ~l. 340/356)
- Modify: `crates/sim/src/provider.rs` (`TrainableStatus`: add `pub body4_degree: Option<usize>`), `crates/app/src/worker/model.rs` (carry it), `crates/app/src/ui/dock/model.rs` ~l. 677 (fit line), `crates/app/src/screenshot.rs` ~l. 927
- Test: `crates/providers/src/surrogate/tests.rs` (collection-mode tests near `collection_mode_learns_across_scenes`), `crates/providers/src/surrogate/tests/refit_cost.rs` (background test neighbours)

**Interfaces:**
- Consumes: `Basis::body4_degree`, `BODY4_DEGREE_START`, `BODY4_DEGREE_CAP` (Task 2); `fit_floor(&self, basis: &Basis) -> usize` (the `Samples` floor of the non-excluded samples at that basis).
- Produces: field `body4_degree: usize` (initial `BODY4_DEGREE_START`); `pub fn body4_degree(&self) -> Option<usize>` (`Some` only while the standing model's basis has body-4); `pub(super) fn grown_body4_degree(&self, species: &[u8]) -> usize`; `#[cfg(test)] pub(super) body4_force: Option<usize>` (a test-only override of the growth rule, used by Tasks 4 and 8, like `body4_off`). `default_basis` and `fit_basis` set `b.body4_degree = self.body4_degree`.

- [ ] **Step 1: Write the failing tests**

`a_collection_steps_its_body4_degree_up_when_the_samples_meet_the_floor`: a collection-mode provider of H₂/H₃ samples (the `collection_root` setup). Compute `f2 = s.fit_floor(&basis_at(2))` and `f3 = s.fit_floor(&basis_at(3))` for the store's species (assert `f2 < f3`), keep `n` samples with `f2 + 30 <= n < f3`, refit: assert `s.body4_degree() == Some(2)` and the model's basis has `body4_degree == 2`. Add samples until `n >= f3 + 30`, refit: `Some(3)`. Then `forget` samples back below `f3` (or hold them with `Q_HELD` codes in the collection), refit: `Some(2)`.

`a_per_scene_model_keeps_its_basis`: `trained(1.2, 2.2, 40, …)`'s model basis equals `Basis::new(&[1])` with the recovery cap logic as before (`body4 == false`, `body4_degree == BODY4_DEGREE_START`), and `body4_degree()` is `None`.

`a_loaded_degree_2_model_steps_up_at_its_next_refit`: write a body-4 sidecar at D₄ = 2 from a collection-mode provider whose store meets `f3`; `install_sidecar` it into a fresh provider: `body4_degree() == Some(2)`, it answers as the saved model did (same prediction at one geometry); after `refit()`: `Some(3)`.

`a_background_refit_carries_its_body4_degree`: start a background refit whose fork steps 2 → 3, install: live `body4_degree() == Some(3)`, no conflict note. Then a snapshot with `body4_degree` 2 against a live provider at 3: `background_conflict` returns `Some(_)` naming "the body-4 degree".

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release body4_degree degree_2_model per_scene_model_keeps -- --nocapture`
Expected: compile errors / FAIL.

- [ ] **Step 3: Implement**

`grown_body4_degree`: without a collection, `BODY4_DEGREE_START`; under `#[cfg(test)]` `body4_force` wins; else the highest d from `BODY4_DEGREE_CAP` down to `BODY4_DEGREE_START` with `fitted >= self.fit_floor(&basis_at_d)`, where `fitted` is the count of non-excluded samples (the number the `Samples` row compares, `FitReport::n_samples`) and `basis_at_d` is `fit_basis(species)` with that degree. Add `body4_degree` to `fit_floor`'s fingerprint tuple and give the cache one slot per degree (two calls per refit must not evict each other). In `fit_store_with`, compute it before the rebuild decision; a degree different from the accumulator's basis forces a rebuild, the chosen degree is written to `self.body4_degree` when the fit succeeds. `FitState` carries `body4_degree` (a failed recovery attempt restores it). `install_sidecar` sets `self.body4_degree` from the loaded basis (old D₄ = 2 files load at 2: no refit, no refusal). Background: `Snapshot` gets `body4_degree`; `background_conflict` returns `why("the body-4 degree")` when live ≠ snapshot; the fork copies it in and `install_fork` copies it back. The recovery's degree-cap-6 basis needs nothing extra: `body4_cap()` = min(6, `body4_degree`) is the chosen D₄ (spec §4.1). `TrainableStatus::body4_degree` from `body4_degree()`; the Model tab fit line appends `, body-4 degree {d}` when `Some(d)`; the screenshot line likewise.

- [ ] **Step 4: Run the tests and the providers suite**

Run: `cargo test -p providers --release` (~5.5 min)
Expected: PASS. A pinned Q′ or column count that changes because a collection test's store now meets the D₄ = 3 floor is updated to the new layout with a one-line comment (`// D₄ = 3: the store meets its floor (spec §4.1)`) — a pinned count, not a tolerance. Any other changed number: stop and report.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(providers): a collection's body-4 degree grows with the store (2 → 3 at its sample floor); provider state, background-safe, shown on the fit line" -- crates/providers crates/sim/src/provider.rs crates/app/src
```

---

### Task 4: Refit cost at D₄ = 3 and the stop condition (§5.6, G4)

**Files:**
- Modify: `crates/providers/src/surrogate/tests/refit_cost.rs` (new runs in `refit_cost_on_a_collection`)
- Modify: `crates/validate/references.md` (new section "Refit cost at body-4 degree 3 (2026-11-…)")

**Interfaces:**
- Consumes: `body4_force` (Task 3), `open_copy`, `provider_on`, `phases`, `replica` (existing helpers in `refit_cost.rs`; `COLLECTION_DIR` points at the user's collection copy).

- [ ] **Step 1: Add the runs**

`REFIT_COST=d4`: on the user's S = 3 collection copy, one cold attach `refit()` and one incremental refit, each at `body4_force = Some(2)` and `Some(3)`, timed (`phases` output). Also report whether M is non-empty after the attach refit, and the wall time of a second `refit()` with no new samples (Review Focus 5: the recovery reruns while M stands).

`REFIT_COST=s4`: a synthetic S = 4 store of ≈ 10 000 samples, for cost only: copy the collection, change every N (Z = 7) of the samples with even index to Z = 6 (labels untouched — the labels are meaningless for that species, the matrix shapes are what is measured; say so in the output and in `references.md`), then replicate with `replica(…, 1e-4)` to ≥ 10 000 samples, `read_only = true`. Time one cold attach refit at `Some(2)` and at `Some(3)`.

- [ ] **Step 2: Run (heavy; alone, BelowNormal)**

Run: `REFIT_COST=d4,s4 COLLECTION_DIR=<copy> cargo test -p providers --release refit_cost_on_a_collection -- --ignored --nocapture --test-threads 1`
Expected: timings printed for every case.

- [ ] **Step 3: Apply the stop condition**

If the S = 4 cold refit at D₄ = 3 exceeds **300 s** on the 9800X3D: stop the plan here and bring the table to the user with the two options of spec §5.6 (D₄ = 3 for many-body centres only, or a smaller n_max in the body-4 block). Otherwise record the table (S, samples, D₄, Q′, cold attach, incremental, second refit with M) in `references.md`.

- [ ] **Step 4: Commit**

```bash
git commit -o -m "test(providers): refit cost at body-4 degree 3 (S = 3 collection, synthetic S = 4 at 10k samples)" -- crates/providers/src/surrogate/tests/refit_cost.rs crates/validate/references.md
```

---

### Task 5: The misfit set M and the `Misfit` trust row (§4.3, first half)

**Files:**
- Modify: `crates/sim/src/activity.rs` (`TrustKey::Misfit`, after `ForceResidual`)
- Modify: `crates/providers/src/surrogate/trust.rs` (`Misfit` struct, the residual pass, `checks`, `untrusted_check`, predicates, `compute_class_loo_full`'s member filter, `FitState`)
- Modify: `crates/providers/src/surrogate/mod.rs` (field `misfit`), `policy.rs` ~l. 342, `store.rs` ~l. 132/445, `background.rs` ~l. 146/340 (the other `training_residuals` assignment and copy sites)
- Modify: `crates/app/src/ui/activity/words.rs` and every `TrustKey` match the compiler names in `crates/app` (checklist row words)
- Test: `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: `recover_trust`'s post-condition (Task 1).
- Produces:
  ```rust
  pub struct Misfit {
      /// Fitted samples (store indices) whose residual ratio exceeds 1 on the fit that stands.
      pub samples: Vec<usize>,
      /// (sample, atom) of every misfit atom: a force misfit's trained atoms over their bound;
      /// every trained atom of an energy misfit.
      pub atoms: Vec<(usize, usize)>,
      /// The training residuals over the fitted samples not in M (what the residual rows judge).
      pub outside: TrainingResiduals,
      /// The fitted samples counted (the limit's base).
      pub fitted: usize,
  }
  ```
  `pub(super) fn compute_residuals(&self) -> Option<(TrainingResiduals, Misfit)>` (one prediction pass; replaces `compute_training_residuals` at every assignment site, which now set `training_residuals` and `misfit` together); `pub fn misfit(&self) -> Option<&Misfit>`; `pub(super) fn in_misfit(&self, i: usize) -> bool`. `training_residuals` keeps meaning "over every fitted sample" (the recovery and `worst_offenders` read it; spec §4.2). M is never persisted.

- [ ] **Step 1: Write the failing tests**

Use a test-only `#[cfg(test)] pub(super) skip_recovery: bool` on `SurrogateKs` (checked at the top of the recovery calls in `refit_fits`) so the misfit mechanics are tested alone; the end-to-end case uses the real recovery.

`a_small_misfit_leaves_the_model_trusted`: `trained(1.2, 2.2, 120, …)`, `skip_recovery = true`, add 3 strays (+0.01 Ha, i.e. 5× the per-atom bound, at r = 1.55, 1.63, 1.71; indices 120–122), refit. Assert `s.trusted()`; `[120, 121, 122]` ⊆ `s.misfit().unwrap().samples` (a good sample the strays pull over its bound may join M — print M); the checklist's `Misfit` row is met with value |M| and limit 12.3 (0.1 × 123); the `EnergyResidual` row's value ≤ 1e-3 (outside M); `s.training_residuals().unwrap().energy_per_atom > 1e-3` (all-sample, unchanged meaning). This store is "the Task 5 store" below.

`a_large_misfit_untrusts_the_model` (real recovery): 40 good + 6 equal strays (+0.1 Ha, distinct r): 6 > 4.6 so the rounds hold nothing (Task 1) and M = 6. Assert `!trusted()`, `untrusted_check().unwrap().0 == TrustKey::Misfit`, the text contains "training samples the model cannot reproduce" and `format!("{} ×", shared::formula(&[1, 1]))`; |M| ≥ 6.

`the_class_rule_leaves_misfits_out`: the store of the first test; the pair class is trusted (`class_loo()`), and no M index appears in `class_worst` / `class_blockers`.

`a_key_held_only_by_misfits_is_refused` (Review Focus 2): a store whose only H₃ samples are misfits (3 H₃ samples with +0.01 Ha + 120 good H₂, `skip_recovery`; assert the three are in M): for an H₃ query, `s.class_untrusted(&env_keys(&pos, &z, rc2))` is `Some((_, true))` — refused by the class rule's sample floor (call it directly: rules 5a–5d may fire first in `evaluate`).

`a_nan_residual_is_not_a_misfit` (Review Focus 1): force a NaN prediction the way `a_nan_prediction_reaches_the_training_residuals_and_the_class_rule` (~l. 4682) does; assert the sample is not in M and the `EnergyResidual` (or `ForceResidual`) row fails with a NaN value.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release misfit class_rule_leaves key_held_only nan_residual_is_not -- --nocapture`
Expected: compile errors.

- [ ] **Step 3: Implement**

Residual pass: per fitted sample (non-excluded, accumulated prefix, fitted rows only — exactly the quantities of today's pass) compute the energy ratio `|ΔE|/N / energy_std_per_atom` (when `use_energy`) and each trained atom's force ratio `d / bound`. A sample is in M iff its largest ratio is `> 1.0` and finite; a NaN ratio stays out of M and goes into `outside` (so the residual rows fail on it, as today). Misfit atoms as in the struct doc. `outside` accumulates exactly as `compute_training_residuals` did, over the non-M samples. New predicate `misfit_exceeds(m: &Misfit) -> bool { !((m.samples.len() as f64) <= QUARANTINE_MAX_FRACTION * m.fitted as f64) }`. `checks()`: the `EnergyResidual` and `ForceResidual` rows read `misfit.outside` (falling back to `training_residuals` when there is no misfit record), then a `Misfit` row with value `|M|` and limit `0.1 × fitted`. Refusal text for `TrustKey::Misfit`: `"{n} training samples the model cannot reproduce ({scenes}) > {limit:.1} (10 % of the fitted samples)"`, where scenes is the per-composition count of M's samples (`shared::formula(&s.z)` for the name, then ` ×count`, comma-separated, largest first: `H2 ×6` in whatever form `formula` writes). App: the checklist row label "training samples it cannot reproduce" and the Activity refusal words for `Untrusted(Misfit)`. Class rule: `compute_class_loo_full`'s member filter adds `&& !self.in_misfit(*i)` (its leave-one-out sums, its keys and the aleatoric `train` sums all come from that one member list). `FitState` carries `misfit`. Do **not** change `residuals_fail`, `recoverable_fail` or `worst_offenders`.

- [ ] **Step 4: Run the tests and the providers + app compile**

Run: `cargo test -p providers --release` then `cargo build -p app --release` (check `app.exe` first)
Expected: PASS. `the_refusal_strings_are_unchanged` must stay green (the old strings are byte-identical; the new key adds one). `a_nan_in_any_trust_input_refuses_trust` (~l. 4638) and `the_trust_checks_and_the_refusal_agree_at_equality_and_nan` (~l. 4264) set `training_residuals` by hand: the rows now read `misfit.outside`, so each such case also sets the same field of `misfit.outside` and gains a `Misfit`-row case (|M| NaN-free, at its limit → met, one over → `TrustKey::Misfit`). Extending them to the new input is required; dropping a case is not.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(providers): local misfit — samples the fit cannot reproduce are counted (Misfit row), judged apart from the residual rows and left out of the class statistics" -- crates/sim/src/activity.rs crates/providers crates/app/src
```

---

### Task 6: The `NearMisfit` refusal, per atom, and its admission path (§4.3, second half)

**Files:**
- Modify: `crates/sim/src/activity.rs` (`TeachRule::NearMisfit` after `NearQuarantine`; `TeachOutcome::MisfitDisagrees` after `FenceDisagrees`)
- Modify: `crates/providers/src/surrogate/quarantine.rs` (`near_misfit_atoms`, `admit`'s new branch)
- Modify: `crates/providers/src/surrogate/mod.rs` (`Handoff.near_misfit`, `evaluate` ~l. 2047 and the three other `Handoff` sites ~l. 1936–1991, `admit_and_keep` call ~l. 1639)
- Modify: `crates/providers/src/surrogate/store.rs` (`admit_and_keep` signature), `audit.rs` (~l. 817–821; `history_rule_code` 23 and its table), `trust.rs` (`geometry_refusal`), `views.rs` (`fence_verdict`, `any_fence`), `policy.rs` (`atom_verdicts`), `inspect.rs` (~l. 155, 311)
- Modify: `crates/app/src/ui/activity/words.rs` (short word "misfit", sentence "an atom here is next to a training sample the model cannot reproduce")
- Test: `crates/providers/src/surrogate/tests.rs`

**Interfaces:**
- Consumes: `Misfit::atoms` (Task 5), the descriptor cache (`ensure_desc_cache`, `desc_cache`, `desc_basis`), `spread_distance`, `QUARANTINE_FRACTION`, `policy.max_descriptor_distance`.
- Produces: `pub(super) fn near_misfit_atoms(&self, pos: &[[f64; 3]], z: &[u8]) -> Vec<bool>` — read-only, one entry per query atom: true when the atom's descriptor is within `QUARANTINE_FRACTION × max_descriptor_distance` (`spread_distance`, the same per-atom distance today's fence uses) of a misfit atom of its species; misfit atoms are read from the cache when it covers the model's basis, else described on the fly (`describe_sample`), as `quarantine_distances_cached` does. Empty-misfit fast path returns all false without describing. `admit(…, near: &[usize], near_misfit: bool, …)` and `admit_and_keep(…, near: &[usize], near_misfit: bool, …)`.

- [ ] **Step 1: Write the failing tests**

All on the Task 5 store (120 good H₂ + 3 strays, `skip_recovery`), in a big cell (`in_big_cell`). Precondition, asserted with a message: the model's answer at r = 1.63 disagrees with the exact Morse teacher (`SurrogateKs::agree` false) — if it does not, raise the strays' offset until it does and say so in the test's doc.

`a_geometry_near_a_misfit_is_refused_and_far_ones_answer`: `evaluate(h2(1.63))` → `Source::Fallback`, last ledger rule `TeachRule::NearMisfit`; `evaluate(h2(2.1))` → `Source::Surrogate`. `geometry_refusal` at r = 1.63 is `Some` with `NearMisfit`; `would_answer` at r = 1.63 is `Err((NearMisfit, _))`.

`atom_verdicts_mark_exactly_the_atoms_near_a_misfit`: a 4-atom scene, an H₂ at r = 1.63 and an H₂ at r = 2.1 ten Bohr apart: `atom_verdicts` gives `atoms[0..2] == Some(NearMisfit)`, `atoms[2..4] == None`, `geometry == None`; `would_answer` refuses the scene (any atom near). The existing `atom_verdicts_split_would_answer_per_atom` sweep stays green.

`an_answer_near_a_misfit_is_kept_only_if_it_agrees`: the teacher (exact Morse) at r = 1.63 disagrees with the model there (the fit carries the strays): outcome `TeachOutcome::MisfitDisagrees`, `samples()` unchanged, no fence vote recorded (`fence_votes` empty, `vote_pending` empty); 20 such steps, then a refit, leave `misfit().samples` as it was before them. With a teacher that agrees (set the policy's `energy_std_per_atom` to 1.0 so the model's answer agrees), the answer is kept with `TeachOutcome::Kept`.

`a_misfit_fence_lifts_when_a_refit_fits_it`: remove the strays' offset by superseding them (forget the three samples), refit: `misfit()` empty, r = 1.63 answered by the model.

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p providers --release near_a_misfit atom_verdicts_mark misfit_fence -- --nocapture`
Expected: compile errors.

- [ ] **Step 3: Implement**

`evaluate`: after the near-quarantine block, `if reason.is_none()` and any `near_misfit_atoms` entry is true → `Refusal::new(format!("surrogate: atom {i} is within {:.2} training spreads of a training sample the model cannot reproduce", limit), TeachRule::NearMisfit, Some(i))` with i the first near atom; carry `near_misfit: bool` in the `Handoff` (every `Handoff` site computes it as the near-quarantine one is computed there; `false` where `near` is empty for want of a model). `geometry_refusal` (draft mode included) and `views::fence_verdict` refuse with `NearMisfit` after their near-quarantine test; `any_fence` is also true when `misfit` is non-empty. `atom_verdicts`: in the fence section, set `refuse(&mut atoms, i, TeachRule::NearMisfit)` for each near atom (per atom, not the geometry). `admit`: after the near-quarantine branch (which runs first when `near` is non-empty), `if near_misfit`: when the model's answer agrees (`Self::agree`) fall through to the normal path; else push `"surrogate: sample not kept — next to a training sample the model cannot reproduce, and the teacher disagrees with the model there (a later refit at more samples or a higher body-4 degree may fit it)"` and return `no(TeachOutcome::MisfitDisagrees)`; no fence vote. `history_rule_code(NearMisfit) = 23` (add the table row). Sampling-pass answers keep their own path (`add_sample`): unchanged, as the spec leaves them.

- [ ] **Step 4: Run the tests and the providers suite**

Run: `cargo test -p providers --release` then `cargo test -p app --release --bin app` (compile + unit tests; the full app suite runs in Task 9)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(providers): NearMisfit — atoms next to a training sample the model cannot reproduce are refused per atom; answers there kept only if they agree" -- crates/sim/src/activity.rs crates/providers crates/app/src
```

---

### Task 7: Physics review of the trust changes (§5.7)

**Files:**
- Read: the diffs of Tasks 1, 5, 6 (`git diff <base>..HEAD -- crates/providers/src/surrogate crates/sim/src/activity.rs`), the spec §4.2–4.3
- Modify: `crates/validate/references.md` (the review's rulings, new section "Local misfit (2026-11-…)")

- [ ] **Step 1: Dispatch the `physics-reviewer` agent (Opus)** with the spec, the diff and these questions, in this order: (1) does leaving M out of the class rule's leave-one-out and aleatoric statistics leave any path to answering a misfit region as trusted, given `NearMisfit` and the audit fences; (2) should M's samples be down-weighted (`Sample::weight`) rather than kept at full weight; (3) should the misfit near distance equal `QUARANTINE_FRACTION × max_descriptor_distance`; (4) is the energy-misfit "every trained atom" rule sound for Sandbox's per-atom freezing.
- [ ] **Step 2: Act on the verdict.** A finding against (1) blocks the merge: fix it as a new task with its own test, then re-review. Answers to (2)–(4) that change behaviour come to the user before they are built (the standing rule on trust changes); answers that keep the design are recorded in `references.md` with their justification.
- [ ] **Step 3: Commit** `references.md` with `docs(validate): physics review of the local-misfit trust changes`.

---

### Task 8: Acceptance on the user's collection (§5.1–5.2)

**Files:**
- Modify: `crates/providers/src/surrogate/tests/residual_diag.rs` (a new `#[ignore]`d test `acceptance_on_the_collection`)
- Modify: `crates/validate/references.md` ("Raising the body-4 degree": the acceptance numbers)

**Interfaces:**
- Consumes: `COLLECTION_DIR` (the 181-hold copy of 2026-11-12 in the session scratchpad, `coll/lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5`; copy it fresh before each run — the test writes codes), `body4_force` (Task 3), `misfit()` (Task 5), `report()` and its "held by scene" print.

- [ ] **Step 1: Write the test**

Attach the copy with the real growth rule; record holds before. After the attach refit assert: `trusted()`; held after ≤ held before; released ≥ 68; print M by scene. Second `refit()`: the same quarantine codes and the same `misfit().samples`. Then the §5.2 comparison: a second provider on a fresh copy with `body4_force = Some(2)`, and per scene print the force rms and the per-scene leave-one-out force RMS at both degrees; assert for every scene `rms3 <= rms2 || (rms3 < 1e-4 && rms2 < 1e-4)` and the same for the LOO RMS.

- [ ] **Step 2: Run (heavy; alone, BelowNormal)**

Run: `COLLECTION_DIR=<fresh copy> cargo test -p providers --release acceptance_on_the_collection -- --ignored --nocapture --test-threads 1`
Expected: PASS. A failed assertion is investigated (systematic-debugging), never met by loosening; a scene whose rms rises above the floor is brought to the user with its numbers.

- [ ] **Step 3: Record and commit**

The acceptance table in `references.md` (held before/after, released, M by scene, both rms columns per scene, wall times). Commit `test(providers): acceptance of the open-shell model on the user's collection copy`.

---

### Task 9: Suites, docs and merge (§5.5, §6)

**Files:**
- Modify: `README.md` (Q′ table per D₄, refit timings from Task 4, the `Misfit` trust row, the `NearMisfit` refusal, "a failed recovery holds nothing"), `docs/dev-reference.md` (trust-rule table and refusal list), `DEVLOG.md` (timeline, durations, failures, wins), `crates/providers/src/surrogate/mod.rs` module doc ("Trust recovery and quarantine", the decision list: rule rows for `Misfit` and `NearMisfit`), the spec's status line ("implemented")

- [ ] **Step 1: Run the gates one at a time** (each alone; record pass/fail and time):
  1. `cargo test -p units -p qm -p sim -p surrogate --release`
  2. `cargo test -p providers --release`
  3. `cargo test -p validate --release --test surrogate`
  4. `cargo test -p validate --release --test surrogate trust_baseline_report -- --ignored --nocapture`, then with `BASELINE_AUDIT_EVERY=10`: compare the missed large errors (an audit error beyond the fit's in-domain p99 while trusted) with the numbers of record in `references.md` "M8a calibrated verdict"; **no rise**, and the stuck-then-escape repro still answered honestly
  5. `cargo test -p app --release` (the replay scripts included)
  Expected: all green. Any red: systematic-debugging, never a loosened tolerance.
- [ ] **Step 2: Update the docs** listed above (README in the same commit as the merge, per CLAUDE.md).
- [ ] **Step 3: Final whole-branch review** (superpowers:requesting-code-review, Opus), fix what it confirms.
- [ ] **Step 4: Merge** with superpowers:finishing-a-development-branch (standing approval to merge reviewed milestones and push; run the merge alone, no pipes on the gating git steps).
