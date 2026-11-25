# Skip the current solve: design (short)

**Date:** 2026-11-25 · **Status:** for approval · **Requested by:** the user (2026-11-25: "a small button to manually skip the current solve … sometimes a human can do a better job at spotting a failed solve before the threshold says so").

## Intent

When the Activity tab's live iteration log shows a solve that is clearly not going to converge, the user can give up on it at once. Today they have to wait for the SCF stall stop, the iteration cap, or the sampling pass's time limit.

The button applies to **sampling passes, play and training** ("both and training yeah").

## Approaches

The user chose **A as the default, with a setting for B.**

- **A. A skip during play or training pauses the run.**
  - The step is not taken and nothing from the solve is kept.
  - A note says so.
  - Play retries the step from a cold start.
- **B. A skip during play or training retries at once.**
  - The solve is dropped and restarted cold on the same geometry, without pausing.

During a sampling pass, both options behave the same way (the design below).

## Design

1. **The button.** A small **Skip** button on the Activity tab's live-solve job row, beside the iteration readout, as the row's second control:
   - on the "Sampling pass" row, beside Cancel and Pause;
   - on the "DFT fallback" / "Solving" row.

   Its hover text: "give up on this solve now: nothing from it is kept".
2. **A sampling-pass sample** (perturbations, scan points, burst steps, teach-around and targeted solves) is given up through the **same path as the pass's time limit** (`providers::sampling`, "The time limit"):
   - it is abandoned at the teacher's next `should_abort` poll;
   - it is logged `failed` with a new `ScfEnd::Skipped` ("skipped by you"), and is never kept;
   - the stalled-solve rules apply: a stretched scan's further-out points are skipped, and a burst step restarts the burst;
   - the next solve is cold.

   A skip does **not** double the time limit, since only time-outs do that.
3. **A solve during play or training.** These are the DFT-mode step, the Estimated-mode fallback, and training's fallback.
   - The solve is abandoned at the next SCF iteration, through the same path an edit uses today ("edits abandon it at the next SCF iteration").
   - The step is not taken and nothing is kept.
   - It is logged as a teaching event `Skipped` when it was a teaching solve.
   - **Setting "Pause" (default):** the run pauses. The Activity row and a toast say "you skipped the solve at step N — nothing was kept; Play retries it from a cold start".
   - **Setting "Retry":** the same solve restarts cold without pausing, and the note says "skipped at step N — retrying cold".
4. **The setting.** A small ▾ beside the Skip button, "After a skip during play: Pause (default) / Retry cold". It is stored in `UiState` and the profile (`skip_then = pause|retry`). It is a display setting: the **Skip action carries the choice** (`Action::SkipSolve { then: Pause | Retry }`), so `--record` sees exactly what happened.
5. **Where Skip is disabled**, with the reason on hover:
   - **CPU solves:** the CPU teacher only checks before the solve, so it cannot stop mid-solve. Hover: "a CPU solve can't be stopped mid-solve".
   - **Reference solves and shell checks**, which the time limit also never cuts: "the fit and the shell fence rest on this solve".
   - **Orbital display solves**, which have no SCF and replace themselves anyway.
   - **Background audits**, which are not on the live row.
6. **Honesty.**
   - A skipped solve is never kept as a sample, a reference or an audit verdict.
   - Skips appear under the solve log's "failed" chip with the word "skipped".
   - Counters count a skip as a failed solve, labelled as such.
   - No trust rule or threshold changes.

## Non-goals

- Skipping background audits or reference scans.
- Automatic skip suggestions (detecting "looks failed" for the user).
- Changing the stall stop, iteration cap or time limit.

## Tests

Worker tests with scripted providers, as the existing relax and pause tests do (CPU, no GPU):

- **Sampling pass:** a skip mid-pass gives `ScfEnd::Skipped`. The sample is not kept, the next solve is cold, a stretched scan's tail is skipped, and the time limit is unchanged.
- **Play with "Pause":** paused, the step not taken, nothing kept. Play then solves the same step cold.
- **Play with "Retry":** not paused, a cold solve of the same geometry.
- **Disabled states:** CPU backend, a reference solve, a shell check.
- **UI:** the button on both rows, the ▾ setting round-tripping in the profile, and `Action::SkipSolve` recorded with its `then`.
- **Gate:** the "viewer / UI / worker" row (`cargo test -p app --release`) and `cargo test -p providers --release`, run once the GPU is free. A replay script with a skip is not added: a skip's timing isn't reproducible.

## Left to the builder

- The exact button styling and the ▾ widget.
- How the worker carries the skip to the running solve: an abort flag beside the existing edit-abandon path.
- The toast wording, within the sentences above.
