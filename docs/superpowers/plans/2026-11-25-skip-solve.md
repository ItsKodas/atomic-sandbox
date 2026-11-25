# Skip the current solve: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Skip button on the Activity tab's live-solve rows. It gives up the running solve:
- in a sampling pass, like the time limit;
- in play or training, then pausing (the default) or retrying cold.

**Architecture:**
- **The request:** a shared `SkipRequest` atomic, set when the UI sends `DynCommand::SkipSolve`, before the command is queued. The command aborts the running solve through the existing abort path (`Command::aborts`).
- **Taking it:** whoever handles the resulting `ProviderError::Cancelled` takes the request.
  - The pass's `Sampler::sample_solve` turns it into a time-limit-style give-up with `ScfEnd::Skipped`.
  - `sim::Dynamics` turns it into a skipped step: provider reset, an activity event, then pause or continue.
- **The UI:** the button and its ▾ setting are display-side, and `Action::SkipSolve { then }` carries the choice.

**Tech stack:** Rust, `sim` / `providers` / `app` (egui).

**Spec:** `docs/superpowers/specs/2026-11-25-skip-solve-design.md` (approved by the user 2026-11-25).

**Model:** all tasks **Opus** (memory "no Fable").

## Global constraints

- A skipped solve is **never kept**: not as a sample, a reference or an audit verdict. No trust rule, threshold, stall stop, iteration cap or time limit changes. A skip does **not** double the time limit; only time-outs do.
- **Wording (spec, verbatim):**
  - button hover: `give up on this solve now: nothing from it is kept`;
  - Pause note: `you skipped the solve at step N — nothing was kept; Play retries it from a cold start`;
  - Retry note: `skipped at step N — retrying cold`;
  - setting: `After a skip during play: Pause (default) / Retry cold`;
  - log word: `skipped by you` (`ScfEnd::Skipped.word()` = `"skipped"`);
  - disabled-state hovers:
    - CPU: `a CPU solve can't be stopped mid-solve`;
    - reference solves and shell checks: `the fit and the shell fence rest on this solve`.
- **Profile key:** `skip_then = pause|retry`, default `pause`. It is a display setting in `UiState`, not an `Action`; the Skip **action** carries `then` and is recorded.
- **Replays:** no replay script with a skip (its timing isn't reproducible). `Action::SkipSolve` is recorded and may replay as an ordinary command.
- **Atomic units and rule 1:** untouched, since no physics changes.
- **GPU:** the user is using the GPU, so only CPU tests run (the focused commands in each task). The full gates (`cargo test -p app --release`, `cargo test -p providers --release`, `cargo test -p sim --release`) wait until the coordinator says the GPU is free. Builds run at BelowNormal with `-j 4`.
- **Commits:** pathspec commits (`git commit -o -- <paths>`), ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review focus

1. **A skip that arrives after the solve already finished** (a race: the user clicks just as it converges). Nothing must be skipped or paused, the request must not linger to hit the *next* solve, and a note says `nothing to skip`. Owned by Task 1: `a_late_skip_is_dropped_and_never_hits_the_next_solve`.
2. **Two skips clicked quickly.** Only the running solve is skipped, once; the second request finds nothing to skip. Owned by Task 1, same test, with a second request.
3. **A skip during a reference solve or shell check, despite the disabled button** (a stale frame). It is not dropped as a failure: it is abandoned like a Pause and re-queued, with a note `reference solves can't be skipped; it restarts`. Owned by Task 2: `a_skip_during_a_reference_solve_requeues_it`.
4. **A skip during a relax.** The button is disabled (hover `a relax: use Stop relax`). A stray request is dropped with `nothing to skip`. Owned by Task 3 (UI) and Task 1 (a dropped request).
5. **Retry cold on the same geometry.** It must really be cold: the provider is `reset()`, and the half-converged parked state is never used as a warm start. Owned by Task 1: `retry_resolves_the_same_step_cold`.

---

### Task 1: The skip request and skipped steps in `sim` — **Opus**

**Files:**
- Modify:
  - `crates/sim/src/activity.rs`: `ScfEnd::Skipped`, with its `word()`;
  - `crates/sim/src/dynamics.rs`: `DynCommand::SkipSolve(SkipThen)`, the `SkipRequest` type, and handling at the `Err(ProviderError::Cancelled)` sites of the step paths (around :1781 and :1841);
  - `crates/sim/src/lib.rs`: re-exports.
- Test: `crates/sim/src/dynamics/tests.rs`.

**Interfaces**
- Produces:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum SkipThen { Pause, Retry }
  #[derive(Clone, Default)] pub struct SkipRequest(Arc<AtomicU8>); // 0 none, 1 Pause, 2 Retry
  impl SkipRequest { pub fn request(&self, then: SkipThen); pub fn take(&self) -> Option<SkipThen>; pub fn peek(&self) -> Option<SkipThen>; pub fn clear(&self) }
  impl Dynamics { pub fn set_skip_request(&mut self, r: SkipRequest) }    // the worker installs the shared one
  // DynCommand::SkipSolve(SkipThen) — applied between steps: if the request is still set (nothing took it), clear it and note "nothing to skip"
  ScfEnd::Skipped  // word() "skipped"
  ```
- In a play or training step (not a relax), when the evaluation returns `Cancelled` and `skip.take()` is `Some(then)`:
  - the step is not taken (state as at a Pause abandon);
  - `provider.reset()` is called, so the next evaluation is cold;
  - a `TeachEvent { outcome: TeachOutcome::Failed, scf_end: Some(ScfEnd::Skipped), step: Some(n), .. }` is pushed with a note in the spec's wording. Mark it a teaching event only when the provider reported the running solve as a teacher solve. Otherwise push only the note on the activity record. The implementer picks the field the snapshot already uses for "fallback running".
  - `Pause` pauses the run with the Pause note; `Retry` keeps playing with the Retry note.
- During a relax, a cancelled step with a pending skip is handled as today's Pause abandon. The request is cleared, with a note `nothing to skip`.

- [ ] **Step 1: Write the failing tests**, using the existing test providers. `ScriptedMorse` and the slow and failing providers in `test_providers.rs` cover this; add a "slow until aborted" one if none exists.
  - `skip_with_pause_pauses_and_keeps_nothing`: playing, a request `Pause` arrives during a step. The time and positions are unchanged, `playing == false`, an event with `scf_end == Some(ScfEnd::Skipped)` is present, and the provider saw `reset()`.
  - `skip_with_retry_keeps_playing`: same, with `Retry`. Still playing, and the next tick solves the same positions.
  - `retry_resolves_the_same_step_cold` (Review focus 5): the provider records whether each evaluation started cold. The evaluation after a Retry skip is cold, at bit-equal positions.
  - `a_late_skip_is_dropped_and_never_hits_the_next_solve` (Review focus 1 and 2):
    - a request is set while no solve runs, then `DynCommand::SkipSolve(Pause)` is applied: the request is cleared and the note says `nothing to skip`;
    - the next step completes normally, still playing;
    - two requests before one cancelled solve skip it once and leave nothing pending.
  - `a_skip_during_relax_is_nothing_to_skip`: the relax keeps its state and the request is cleared.
- [ ] **Step 2: Run them and see them fail.** `cargo test -p sim --release skip` fails on the missing types.
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Run them and see them pass,** together with `cargo test -p sim --release dynamics::tests` (CPU, the existing dynamics unit tests).
- [ ] **Step 5: Commit.** `feat(sim): skip the running solve — a skip request, skipped steps that pause or retry cold`

### Task 2: A skipped sample in the sampling pass — **Opus**

**Files:**
- Modify: `crates/providers/src/sampling/mod.rs`, in `Sampler::sample_solve` (around :1555–1610), the failure classification (around :1539 and :1672) and the module doc's "The time limit" paragraph. Add a sentence on skips.
- Modify: the place the pass's driver receives the `SkipRequest`, either `SurrogateKs::pass_step` or the `Sampler` constructor. The implementer picks it and installs the same handle the worker gives `Dynamics`.
- Test: `crates/providers/src/sampling/tests.rs`, next to `a_sample_solve_past_the_time_limit_is_given_up_like_a_stall` (:2285).

**Interfaces**
- Consumes (Task 1): `SkipRequest`, `SkipThen`, `ScfEnd::Skipped`.
- Produces: `impl Sampler { pub fn with_skip_request(self, r: SkipRequest) -> Sampler }`.
- `Err(Cancelled)` with `skip.take().is_some()` follows the time limit's path, except:
  - `last_failure = Some((ScfEnd::Skipped, 0))`, and `timeouts_in_a_row` is **not** incremented;
  - the error text reads `skipped by you after {:.1} s`.

  The stalled-solve rules then apply exactly as for `TimeLimit`: add `Skipped` to the `Stalled | Cap | TimeLimit` matches at :1468 and :1919, the stretched-scan tail skip and the burst restart.
- **Untimed jobs** (references, shell checks): a pending skip is **not** taken. The solve is abandoned like a Pause, the job stays queued, and the note says `reference solves can't be skipped; it restarts`. The request is cleared.
- The `then` value doesn't matter in a pass: the pass continues.

- [ ] **Step 1: Write the failing tests.**
  - `a_skipped_sample_is_given_up_like_a_time_out_without_doubling_the_limit`: mirror the time-limit test, with a skip instead of the clock.
    - The sample is not kept and `failed` goes up by 1.
    - The event has `ScfEnd::Skipped`.
    - The next solve is cold.
    - `time_limit_s()` is unchanged from before the skip.
  - `a_skipped_stretched_scan_point_drops_the_further_points`: the further-out points become skip rows, as for a stall.
  - `a_skip_during_a_reference_solve_requeues_it` (Review focus 3): the reference job is still queued, nothing is counted failed, the request is cleared, and the note is present.
- [ ] **Step 2: Run them and see them fail.** `cargo test -p providers --release sampling::tests::a_skip` and `…::a_skipped` (CPU; the scripted teachers as in the time-limit test).
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Run them and see them pass,** together with the existing time-limit tests: `cargo test -p providers --release sampling::tests::a_sample_solve_past_the_time_limit`.
- [ ] **Step 5: Commit.** `feat(providers): a skipped pass sample is given up like a time-out, without doubling the limit`

### Task 3: The worker, the Activity button, the setting and docs — **Opus**

**Files:**
- Modify:
  - `crates/app/src/worker/commands.rs`: `Command::aborts` includes `DynCommand::SkipSolve(_)`, plus the batch, edit and pause classification beside it;
  - `crates/app/src/worker/mod.rs`:
    - the worker owns one `SkipRequest`, installed on every `Dynamics` it builds and on the sampler;
    - `SimWorker::send` calls `request(then)` **before** queuing `SkipSolve`, as it counts commands for the display interrupt;
    - module doc "What aborts a running step": add Skip;
  - `crates/app/src/ui/actions.rs`: `Action::SkipSolve { then: SkipThen }` → `Command::Dyn(DynCommand::SkipSolve(then))`;
  - `crates/app/src/ui/activity/jobs.rs`: `JobRow` gains `skip: Option<SkipControl>`, `pub enum SkipControl { Enabled, Disabled(&'static str) }`. It is set on the "Sampling pass" row and on the "DFT fallback" / "Solving" row.
    - Disabled for a CPU engine, with the spec's hover.
    - Disabled during a pass stage that solves references or shell checks, with the spec's hover. Find the stage names from `words::pass_stage_of`.
    - Disabled during a relax, with hover `a relax: use Stop relax`.
    - No skip on the orbital display row.
  - `crates/app/src/ui/activity/mod.rs` or `jobs.rs` rendering: the small Skip button plus the ▾ setting. The ▾ writes `UiState::skip_then`.
  - `crates/app/src/ui/profile.rs`: `skip_then = pause|retry`;
  - `crates/app/src/ui/activity/log.rs`: skipped solves appear under the "failed" chip with the word "skipped";
  - `README.md`: the Activity tab section gets Skip, its setting, and when it's disabled;
  - `docs/dev-reference.md`: the Activity bullet's probes and keys;
  - `DEVLOG.md`: at merge.
- Test: `crates/app/src/worker/tests.rs`, `crates/app/src/ui/tests.rs` (the Activity and profile modules).

**Interfaces**
- Consumes: Task 1's `SkipRequest`, `SkipThen`, `DynCommand::SkipSolve`; Task 2's `Sampler::with_skip_request`.
- Produces: `Action::SkipSolve { then: SkipThen }`, `UiState::skip_then: SkipThen` and `JobRow::skip`. The button's probe id is `activity-skip`, and the ▾'s is `activity-skip-then`.

- [ ] **Step 1: Write the failing tests.**
  - `worker::tests::skip_during_play_pauses_and_keeps_nothing`: a `SimWorker::spawn_test` with a slow scripted provider, playing. Send `Action::SkipSolve { then: Pause }` through `actions::apply`, then the commands. The snapshot reaches paused with the skipped note, and the time didn't advance.
  - `worker::tests::skip_during_play_with_retry_keeps_playing`.
  - `ui::tests::activity::the_skip_button_is_on_both_live_rows_and_disabled_with_reasons`:
    - enabled on a GPU pass row and a GPU fallback row;
    - disabled with the exact hovers for a CPU engine, a reference stage and a relax;
    - absent on the orbital row.
  - `ui::tests::profile::skip_then_round_trips` (default `pause`).
  - `ui::tests::activity::a_skip_is_recorded_with_its_choice`: the recorder writes the action with `then`.
- [ ] **Step 2: Run them and see them fail.** `cargo test -p app --release --bin app -- worker::tests::skip ui::tests::activity ui::tests::profile` (CPU; `spawn_test` uses scripted providers).
- [ ] **Step 3: Implement it, and update README and dev-reference.**
- [ ] **Step 4: Run them and see them pass.** Then render `--screenshot out.png --ui "tab=activity"` on a CPU scene and check the PNG: the disabled Skip appears with its ▾.
- [ ] **Step 5: Commit.** `feat(app): Skip on the Activity tab's live solve, with Pause / Retry cold after a skip during play`

## Execution order and gates

Tasks 1 → 2 → 3. When the coordinator says the GPU is free, run the full gates once, serially:

- `cargo test -p sim --release`
- `cargo test -p providers --release`
- `cargo test -p app --release`

The known environmental failure `ks::tests::the_level_gated_stall_stop_on_the_measured_cases` is expected. The branch merges after the gates.
