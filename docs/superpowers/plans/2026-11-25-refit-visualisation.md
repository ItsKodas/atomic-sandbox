# Refit Visualisation Implementation Plan

**Goal:** show what a refit is actually doing, stage by stage, from its own counters (approved spec
`docs/superpowers/specs/2026-11-22-refit-visualisation-design.md`). The full picture is in the 3D
view while there is no model (the first fit at launch); a compact strip is in Activity's refit card
for every other refit. The time words come from the running stage's own counter, never from the
last refits' walls.

**Constraints:** display only. Every fit, leave-one-out and verdict stays bit for bit (the existing
fit goldens and tests pass unchanged; new tests compare metered and unmetered fits bit for bit). The
counters are relaxed atomics read by the probe. No new control reaches the worker (nothing new goes
through `ui::Action`). Atomic units inside `surrogate`/`providers`.

**Model:** Opus (UI, plumbing, tests with pinned values); no solver maths changes.

## Where the time goes (measured before deciding the counters)

`providers` `surrogate::tests::refit_profile`, new run `channels` (`REFIT_PROFILE=channels`, built
with `--features profile`), on a read-only copy of the user's installed H/Li/O/S/Cl collection
(2 708 stored, 2 655 fitted, 6 163 atoms), K = 3, 12 worker threads:

| stage (first channel refit, 79.9 s wall) | s | calls / note |
|---|---|---|
| **learning the mix W** (`channel_lifecycle`) | **53.2** | its dense reference search: **221 Choleskys** (`TwoNoise::solve`) 53.1 s; its gathering < 0.6 s |
| the fit's spectral search (2 searches) | 16.7 | 12 spectra 9.2 s, 14 solves 5.3 s, condition estimates 1.7 s; Q′ 4 140, 5 rounds |
| the trust recovery (a degree-cap retry: one more fit) | 10.9 | inside the line above and below |
| checking each sample (class leave-one-out) | 4.2 | 2 655 samples, 43 cpu-s on 12 threads |
| null space | 1.4 | |
| active set | 0.9 | |
| counting environments (body-4 degree floor, floor columns, class keys) | 2.4 | |
| gathering samples (`add_many`, every fit) | 0.6 | |
| training residuals | 0.25 | 2 calls |

A second refit with nothing new: 8.1 s — warm solve 0.2 s, the standing fit's trust passes 7.3 s
(leave-one-out 4.4 s, null space 1.4 s, active set 0.9 s, counts 0.5 s).

So the counters are:
- **the search** (rounds; decompositions as each finishes, the library kernels one step each): the
  W learn's 221 Choleskys and the fit's spectra and solves are most of a channel refit;
- **checking each sample**: one count per sample, with its ratio;
- **gathering**: one count per batch (a second at most, as the spec expected);
- **trust**: the recovery's round and the samples held;
- the passes without a counter are named (counting environments, training residuals, null space,
  active set) so no second is misattributed.

The species basis runs the same code minus the W learn (and with the warm/spectral search on body
4, the dense search per scene): the stages are the same, the mix row never shows.

## Files

- `crates/surrogate/src/progress.rs` (new): `Meter` trait (`described`, `search_round`,
  `decomposition`), a thread-local `install` guard, free functions that are no-ops without a meter;
  `Recorder` for tests (`test-hooks`).
- `crates/surrogate/src/fit.rs`: `add_many` counts per batch (per sample off body 4); the dense and
  spectral searches report their rounds; `TwoNoise::{solve, solve_large, spectrum}` a decomposition
  each. `crates/surrogate/src/loo.rs`: `leave_one_out_many_with(model, samples, each)` (the same
  chunks, the callback sees each result on its worker).
- `crates/sim/src/refit_live.rs` (new): `RefitMeter` (relaxed atomics + the ratio buffer, a
  `Mutex<Vec<f32>>` taken once per sample by a worker and once per frame by the UI), `RefitLive`
  (the Copy snapshot), `LiveStage`, `OtherPass`, `TrustRound`, the stage's time left / fraction /
  decomposition fill. `crates/sim/src/activity.rs`: the probe holds one; `RefitView::live`;
  `refit_described/search_round/decomposition/check_begin/checked/other/mix/trust_round`,
  `refit_ratios`, `last_refit_live`.
- `crates/providers/src/surrogate/{mod,trust,quarantine,represent}.rs`: `ProbeMeter`; `refit()`
  sets `meter` (the live probe or a background copy's stage probe) and installs it for its duration;
  `leave_one_out_metered` in the class pass and `ensure_loo_ratios` (each sample's ratio is
  `loo_ratio`'s first number, the Worst-fitting list's); `meter_other` at the counted-less passes;
  the mix flag around `learn_mix`; the trust round in `recover_trust` / `rounds_by`.
- `crates/app/src/ui/refit_picture.rs` (new): `in_view`, `free_rect` / `picture_rect` (the room the
  zones leave), `blocks` (one per sample, one per k past the room), `cell` / `fill_of`, the view
  picture (`show`, fades out over 0.6 s), the Activity `strip`, `demo` for the screenshot key.
- `crates/app/src/ui/activity/{words,jobs,headline,mod}.rs`, `ui/dock/model.rs`: stage words
  replace the history countdown; the strip under the refit card. `ui/overlays.rs` calls the picture.
  `ui/mod.rs`: `PanelInput::refit_ratios`, `UiState::refit_picture`. `viewer.rs`: the ratios read
  once per frame while a refit runs; `demo_refit`. `screenshot.rs`: `--ui "refit=<stage>"`.

## Tests

- `surrogate`: `progress::tests` (own thread only, nesting); `fit::tests::a_metered_fit_counts_its_stages_and_computes_the_same_bits`
  (every sample once; rounds 1..=R; R spectra and R + 1 solves; the dense search's Choleskys; both
  fits bit for bit); `loo::tests::leave_one_out_many_is_the_serial_loop_bit_for_bit` extended (the
  callback sees each sample once with its own result, 1 and 3 threads).
- `sim`: `refit_live::tests` (gathering rate and estimate, the search's counts and fill, the ratio
  buffer from threads, the mix booked to its own slot, a new refit resets); `activity::tests::the_live_stage_counters_are_published_and_restarted`.
- `providers`: `surrogate::tests::a_metered_refit_publishes_its_stages_and_fits_the_same_bits`
  (a probe vs none: the same bits; Q′, rounds, the ratios equal `loo_ratios()` as f32; nothing
  published outside a refit).
- `app`: `refit_picture::tests` (free rect, blocks, cells, fill, stage rows),
  `ui::tests::{the_refit_stage_words_card_headline_and_banner, the_refit_estimate_is_drawn_in_the_activity_card_and_the_model_banner, the_refit_picture_fills_the_view_only_while_there_is_no_model}`.
- Gates: `cargo test -p surrogate --release`, `cargo test -p providers --release`,
  `cargo test -p app --release --bin app -- ui:: worker::`; a rendered check with
  `--screenshot … --ui "refit=check"` (and `tab=activity`).
