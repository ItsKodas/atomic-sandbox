# Warm-Started Recovery Trials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search each of a trust recovery's trial fits from the previous fit's hyperparameters, and re-search every trial that would stand from scratch, judging it on that cold fit.

**Architecture:** Two pieces.
- In `surrogate`, an optional search start on `Accumulator::refit` that reaches the existing `estimate_spectral(opts, start)` hook.
- In `providers`, `recover_trust` and `quarantine_rounds` pass the previous fit's (λ, s_E, s_F) to every trial, and confirm every acceptance with a cold refit.

**Tech Stack:** Rust; crates `surrogate` (`src/fit.rs`) and `providers` (`src/surrogate/quarantine.rs`, `src/surrogate/trust.rs`).

**Spec:** `docs/superpowers/specs/2026-11-17-recovery-warm-start-design.md`

## Global Constraints
- A refit with no start is bit for bit today's. No existing test is re-pinned or loosened (CLAUDE.md rule 6).
- Every fit that stands, recovery or not, is a cold search: a function of the accumulated statistics alone.
- No per-element parameters (rule 1). f64 on the CPU throughout.
- Serialise GPU test binaries: before `cargo test -p providers` or `-p validate`, wait while any other `target\release\deps\*-<hash>.exe` test binary is running. CPU-only builds and tests may run at BelowNormal priority with `-j 6`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus
1. A trial that passes warm but fails cold must continue the recovery exactly as a failed trial does today: no half-installed state, codes, `species_dirty` or note.
2. A background refit (`background.rs`) runs the same recovery on a copy, so the start and the confirmation must work there too, and be cancellable.
3. A start from a fit on another basis (the cap toggled) must not error: a non-finite or non-positive start is filtered, and a positive one converges.
4. The standing fit's `WarmState`, which later incremental refits continue from, must come from the cold re-search, not from the warm trial.
5. The dense (small-basis) path: if it has no start hook, a trial there is simply cold (no confirmation needed, nothing to change).

---

### Task 1: The search start in `surrogate`

**Files:** Modify `crates/surrogate/src/fit.rs`; tests in the same file's test module.

**Interfaces:**
- Produces: `Accumulator::refit_from(&mut self, opts: &FitOptions, provenance: &Provenance, ref_energy: &[f64], start: Option<SearchStart>) -> Result<(Model, FitReport), FitError>`.
  - `pub struct SearchStart { pub lambda: f64, pub sigma_e: f64, pub sigma_f: f64 }`.
  - `refit(..)` = `refit_from(.., None)`.
  - `FitReport` (or the model) exposes the fit's (λ, σ_E, σ_F) as a `SearchStart` via `FitReport::search_start(&self) -> SearchStart`.
  - `RefitKind::Spectral` gains `from_start: bool`.
- A started refit skips the warm rank-one path (it is always a search).

- [ ] **Step 1: Write the failing tests.**
  - `a_started_search_reaches_the_cold_stopping_point`: on an existing spectral-path fixture, cold refit vs a refit started from the cold result perturbed ×1.5 (λ) and ×0.8 (σ's). Assert λ, s_E², s_F² each within 3e-3 relative of the cold result (the 1e-3 stopping test on both sides), and that the started search's rounds are ≤ the cold rounds.
  - `no_start_is_todays_refit`: `refit_from(.., None)` vs `refit`: the coefficients bit for bit.
- [ ] **Step 2:** Run `cargo test -p surrogate --release started_search no_start`. Expected: FAIL (not defined).
- [ ] **Step 3:** Implement. Feed `start` into `estimate_spectral` in `solve_spectral`, and update its doc: a started search is a recovery trial's and never stands. Set `from_start`.
- [ ] **Step 4:** `cargo test -p surrogate --release`. Expected: PASS, every existing test included.
- [ ] **Step 5:** Commit.

### Task 2: Warm trials and cold confirmation in the recovery

**Files:**
- Modify `crates/providers/src/surrogate/quarantine.rs` (`recover_trust`, `quarantine_rounds`, `rounds_by`) and `crates/providers/src/surrogate/trust.rs` (`fit_store_with` takes the start).
- Tests: `crates/providers/src/surrogate/tests/recovery_loo.rs`, or a new `tests/recovery_warm.rs`.
- Modify `crates/validate/references.md`: measured timings.

**Interfaces:** Consumes Task 1's `refit_from`, `SearchStart` and `FitReport::search_start`.

- [ ] **Step 1: Write the failing tests.**
  - `a_recoverys_standing_fit_is_a_cold_search`. Drive a recovery that succeeds by quarantine on an existing recovery fixture (reuse `recovery_loo.rs`'s builders). Then, on a fresh provider with the same samples and the resulting quarantine codes and degree cap, run a cold refit. Assert the hyperparameters and coefficients are identical bit for bit, and that at least one trial ran `from_start`.
  - `a_trial_that_fails_cold_is_a_failed_trial`: a test seam (`#[cfg(test)]` flag) forces the cold confirmation to fail once. Assert the recovery's codes, cap, note and fit equal those of the same recovery with that trial failing outright (the training-ranking seam is the pattern to follow).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
  - Every trial's `fit_store_deferred` gets the most recent fit's `search_start()`.
  - Add `confirm_cold(&mut self, accept: impl Fn(&Self) -> bool) -> Option<FitReport>`: it refits with no start on the same state and re-runs `accept`.
  - Call it at every acceptance point listed in the spec, including `quarantine_rounds`' choice between rankings.
  - Leave the restore-to-entry path alone.
- [ ] **Step 4:** `cargo test -p providers --release`. Expected: PASS, every existing test unchanged.
- [ ] **Step 5: Measure on the user's collection copy** (read-only, never the user's folder; the profile copy under the session scratchpad works).
  - Record `refit_golden` on main 7e420ad (a detached worktree build).
  - Compare this branch against it, all three cases. Expected: every verdict identical, within the bars.
  - Report the recovery case's wall time and its search rounds per trial, before and after.
  - Add a "Warm-started recovery trials" entry to references.md.
- [ ] **Step 6:** `cargo test -p validate --release --test surrogate`. Expected: PASS.
- [ ] **Step 7:** Commit the code, the tests and references.md, plus the spec and this plan if they are not already committed.
