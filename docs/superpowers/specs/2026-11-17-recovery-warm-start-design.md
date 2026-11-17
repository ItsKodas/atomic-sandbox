# Warm-started recovery trials (design, short)

**Status:** approved by the user 2026-11-17 ("yes build the warm-start option", then "B").

## Intent
A trust recovery (`SurrogateKs::recover_trust`: the degree-cap retry, the default-basis retry, the quarantine rounds by both rankings) runs up to about a dozen fits. Each runs the full spectral hyperparameter search from the data's own starting levels. After the refit speed-up (merge 7e46006) a recovery still takes about 31 s on the user's BEST collection. Starting each trial's search from the previous fit's settings should cut the search rounds, for an estimated 20–40 % off a recovery.

## The constraint kept
`Accumulator::solve_spectral` deliberately starts from the data's own levels. That makes a search a function of the accumulated statistics alone, so a reopened collection's first fit equals the session's last search bit for bit. **Every fit that stands keeps this property.** Only the trial fits, which a recovery throws away or confirms, are warm-started.

## Approaches
- **A. Warm-start every fit in a recovery.**
  - For: the biggest saving.
  - Against: the fit that stands depends on the path taken to it, by about the search tolerance (1e-3 relative in λ, s_E², s_F²). A reopen no longer reproduces it exactly.
  - Not chosen.
- **B. Warm-start only the trials, and redo the winner from scratch.** Chosen by the user.
  - A trial is searched from the previous fit's (λ, s_E, s_F).
  - When a trial passes the check that would accept it, it is re-searched cold, and the acceptance is judged on the cold fit.
  - Cost: one extra cold search per accepted trial, about 2–4 s.
  - Every standing fit stays a function of the statistics alone.
- **C. Skip a recovery that already failed on the same samples.** Declined (the user, 2026-11-17).

## Design (B)
- **Accumulator.** `surrogate::Accumulator` gains a way to refit with a search start `(λ, s_E, s_F)`, fed to the existing `estimate_spectral(opts, start)` hook, plus the dense path's equivalent if it has one.
  - A refit with no start is bit for bit today's.
  - A warm-started search converges to the same stopping test as a cold one (λ, s_E², s_F² each within the search's 1e-3), and is marked as warm-started in its `RefitKind`.
- **Trials.** The quarantine's trials — every round of both rankings, on the standing basis (step 2) and in step 3 (its default-basis fit and its rounds) — are searched from the hyperparameters of the most recent fit in the chain. A basis change (cap on or off) still passes the start: the search filters a non-finite start and converges from any positive one.
  - **Fix round 1 (the coordinator, 2026-11-17: option B must never make a refit slower than main).** The cap retries, steps 1 and 1b, are searched cold directly, with no start and nothing to confirm: they are single trials that usually stand, and a warm trial that stands pays its confirmation on top (measured +4 / +5 rounds on the golden's cold-attach and append cases before this round). Their result is exactly main's.
- **Acceptance.** At every point where a warm trial would stand, the trial is refitted cold on the same state and the same check is re-run on the cold fit. This covers step 3's acceptance, the rankings' choice and the "passes the residuals" keep (steps 1 and 1b are cold already).
  - Only if the cold fit passes does the trial stand.
  - If the cold fit fails, the trial counts as failed, and the recovery continues exactly as it does today after a failed trial.
  - Restoring the entry state needs no re-search: that fit was already cold.
- **Where outcomes can differ from today.** A trial whose warm fit fails a check its cold fit would pass, or a near-tie in the offender ranking computed from the warm fit, can lead a recovery down a different path. It is the same class of change the speed-up round measured: hyperparameters at the 1e-3 level, with every trust verdict unchanged on the golden.

## Acceptance
- Cold refits and the non-recovery paths are bit for bit unchanged. The existing goldens and `fit.rs` tests still pass.
- A recovery's standing fit equals a cold refit of the same store state bit for bit (the reopen property).
- `refit_golden` compare, all three cases, against a record made on main 7e420ad with the same collection copy: every verdict identical, within the ruling-2026-11-17 bars.
- The recovery case's wall time is measured before and after, together with the search rounds per trial, and recorded in `crates/validate/references.md`.
- Gates: `cargo test -p surrogate --release`, `cargo test -p providers --release`, `cargo test -p validate --release --test surrogate`.

## Left to the builder
How the start travels (a field on the accumulator, or a parameter on a `refit_from`), and the name of the `RefitKind` marker.
