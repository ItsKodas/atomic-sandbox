# Leave-one-out recovery and per-sample forget — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Two changes to the estimated model:
- Trust recovery quarantines the sample that breaks the fit (ranked by its leave-one-out residual), not that sample's victims.
- The user can forget, and restore, one sample from the Model tab.

**Architecture:**
- `providers::surrogate`:
  - `quarantine_rounds` reads per-sample leave-one-out ratios.
  - The trust pass caches those ratios for each fit.
  - A new persisted code `Q_FORGOTTEN` is honoured across the fit, release, fences, the cap, the collection and the sidecar.
- `sim::Trainable` gains forget, restore and list methods.
- The app adds `Action::ForgetSample` / `Action::RestoreSample` and a Model-tab section.

**Spec:** `docs/superpowers/specs/2026-11-12-sample-recovery-design.md` (read it first; it is the authority). Its "Amendments (2026-11-12/13)" section records the rulings made during implementation; they override the plan where they differ.

## Global Constraints

- CLAUDE.md applies:
  - rule 1: no per-element parameters;
  - rule 3: atomic units;
  - rule 5: the `ForceProvider` boundary, no downcasts;
  - rule 6: never loosen a tolerance;
  - rule 7: honest notes.
- No change to `QUARANTINE_ROUNDS`, `QUARANTINE_MAX_FRACTION`, `RECOVERY_DEGREE_CAP`, any bound, or the trust rule.
- Every control that reaches the worker goes through `ui::Action`.
- Tests: heavy tests are release-only (`#[cfg_attr(debug_assertions, ignore)]`). Run one heavy job at a time.
- **Never kill the user's `app.exe`.** Build at BelowNormal priority with `-j 4` if other cargo jobs run.
- Never touch `D:\Projects\Physics\models` or any user collection. Measurements run on copies in the scratchpad.
- Edit files with Edit/Write or Python, never with PowerShell `Set-Content`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Keep README.md current. Update `docs/dev-reference.md` and `crates/validate/references.md` where the behaviour they describe changes.

## Review Focus

1. **A store where many samples have an undefined leave-one-out residual.**
   - Recovery must not hold them all at once past the cap. The cap check runs before holding.
   - The trust rule's counts must still judge the result.
2. **A stale leave-one-out cache.** It must never come from an earlier fit, for example after a release or a forget changed the fit.
3. **Forgetting a sample while a background refit is in flight.** The status must reach the collection, and the refit that lands afterwards must not resurrect the sample.
4. **Forgetting the only sample of a species or environment key.** The trust counts may then refuse the model. That is fine and honest, but it must not panic, and the note must say why.
5. **An old build or an old sidecar meeting code 7, and a new build reading an old file.** Old files read as before.

---

### Task 1: Leave-one-out ranking for recovery rounds (Opus; physics review before merge)

**Files:**
- Modify `crates/providers/src/surrogate/quarantine.rs`:
  - `worst_offenders`;
  - `quarantine_rounds`;
  - `note_quarantined` and the step-3 note text.
- Modify `crates/providers/src/surrogate/trust.rs`: the class pass caches the per-sample leave-one-out ratios.
- Modify the provider struct in `crates/providers/src/surrogate/mod.rs`: the cache field and its invalidation. Update the module doc "Trust recovery and quarantine".
- Add tests under `crates/providers/src/surrogate/tests/`: a new `recovery_loo.rs`, wired like the existing test modules.
- Update `crates/validate/references.md` with the measurement.

**Produces (Task 2 relies on these):**
- `pub(super) fn loo_ratios(&mut self) -> Option<&[(usize, f64)]>`, or an equivalent accessor. It returns per-sample `(store index, leave-one-out ratio)` for the current fit, computing them if the cache is stale.
- `f64::INFINITY` stands for an undefined residual.

**Steps:**
- [ ] Read the spec R1, `quarantine.rs` lines 295–600, and `trust.rs` around the class pass (`leave_one_out_many`, about line 640).
- [ ] Write the failing unit test: a high-leverage outlier.
  - Build a small store with a fitted pair channel, for example many H₂ near equilibrium plus a bent or perturbed family, so that one strongly compressed sample with a large force bends the pair term.
  - Today's recovery quarantines victims and fails, or holds many. The new ranking holds the outlier alone and trusts.
  - Assert:
    - the outlier is held;
    - the victims are not;
    - the model is trusted after the refit.

  Prefer an existing test fixture or teacher (`tests/` helpers) over hand-made forces. If no cheap analytic teacher can show the effect, say so in the report and pin the closest honest variant.
- [ ] Run it and see it fail on the current code.
- [ ] Implement the ratio exactly as spec R1 defines it, with rulings R1a and R1b.
  - The cache is keyed to the fit, for example by a fit generation counter bumped wherever the model is replaced or restored (`save_state` / `restore_state`).
  - A stale cache recomputes.
- [ ] Rewrite the recovery note text as the spec gives it, including the worst leave-one-out ratio, rounded.
- [ ] Run `cargo test -p providers --release`. Investigate every changed pinned recovery count, following spec R1 "Existing recovery tests".
- [ ] Add an `#[ignore]`d measurement test. It reads a collection folder from the environment variable `SURROGATE_RECOVERY_SNAPSHOT`, attaches it, refits and prints:
  - the held set's size;
  - whether #1804 is held;
  - the worst training ratio;
  - the trust verdict;
  - the wall time.
- [ ] Run the measurement against a **copy** of `C:\Users\koda\AppData\Local\Temp\claude\ab332b822dafabae\d8535878-a71d-4fab-8d07-5ec70933a386\scratchpad\pre1805`. Copy it to your own scratch folder first and never write to the original. If `pre1805` is not a collection folder, look for the snapshot named in the scratchpad's `collection_debug.rs`. Record the numbers in `references.md`.
- [ ] Run `cargo test -p validate --release --test surrogate` (about 12–17 min, alone).
- [ ] Commit.

### Task 2: Forget and restore one sample (Opus)

**Files:**
- `crates/surrogate/src/quarantine.rs`: `Q_FORGOTTEN = 7`, with doc.
- The collection reader: accept code 7.
- The sidecar reader and writer: round-trip 7.
- `crates/providers/src/surrogate/quarantine.rs` and `mod.rs`:
  - `forgotten()`;
  - `excluded()` includes 7;
  - `release_reproduced` and supersede skip 7;
  - fences include 7;
  - the cap in `quarantine_rounds` counts held + forgotten.
- `crates/sim/src/provider.rs` (`Trainable`): add these, with defaults that return `Err("not supported")` or empty:

  ```rust
  fn forget_sample(&mut self, index: usize) -> Result<(), String>;
  fn restore_sample(&mut self, index: usize) -> Result<(), String>;
  fn sample_suspects(&self) -> Vec<SampleSuspect>;
  fn forgotten_samples(&self) -> Vec<SampleSuspect>;
  ```

  `SampleSuspect { index: usize, formula: String, origin: String, loo_ratio: Option<f64> }` lives in `sim`.
- The worker in `crates/app/src/worker/`: handle the two Actions, then refit (background where the worker already refits in the background).
- `crates/app/src/ui/actions.rs`: `Action::ForgetSample(usize)`, `Action::RestoreSample(usize)`.
- `crates/app/src/ui/dock/model.rs`: the "Worst-fitting samples" section, as spec R2 describes it.
- Replay or record support, if `Action` needs it.
- README: Model tab controls.
- `docs/dev-reference.md`.

**Consumes:** Task 1's per-sample leave-one-out ratio accessor.

**Steps:**
- [ ] Failing tests in `providers`:
  - a forgotten sample is not fitted;
  - it is not released by a refit that reproduces it;
  - it survives a collection reopen and a sidecar round trip;
  - it is fitted again after restore;
  - the cap counts it.
- [ ] Implement in `surrogate` and `providers`. Check what an older build does on reading code 7, and document it in the code 7 doc.
- [ ] `Trainable` methods with defaults; the provider's implementation; the worker handling.
- [ ] UI section and Actions. App unit tests:
  - the row text;
  - Forget and Restore buttons emit the Actions;
  - the "no model" and "no ranking yet" texts.
- [ ] Run `cargo test -p providers --release`, then `cargo test -p app --release`, one at a time.
- [ ] README and dev-reference. Commit.
