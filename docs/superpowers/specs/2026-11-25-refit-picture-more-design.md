# The refit picture after a pass and on Refit now: design (short)

**Date:** 2026-11-25 · **Status:** approved by the user ("yes go ahead with the refit picture") · **Requested by:** the user (2026-11-25: "After each training, it does a larger refit, you should display the view screen UI for that as well" … "As well as when you manually press 'Refit Now'").

Builds on the refit picture (`2026-11-22-refit-visualisation-design.md`, code `crates/app/src/ui/refit_picture.rs`).

## Intent

Today the full refit picture fills the 3D view only while no model stands (`refit_picture::in_view`: a refit runs and the model isn't fitted). Every other refit shows only Activity's compact strip. The refits the user waits on after a training pass, and the one they start by hand, are the long ones they want to watch.

## Approaches

- **A (chosen). Tag each refit with its cause** where it is started, publish it with the refit's live view, and let the picture's rule read it.
- B. Infer it in the UI (a pass just ended / the user just clicked) — rejected: timing-based guesses misfire when a refit is queued behind another.
- C. Show the picture for every refit — rejected: background refits during play happen every few seconds and would cover the view.

## Design

1. **A refit's cause** — `sim::RefitCause { First, AfterPass, Requested, Background }`, set by whoever starts the refit and published with the live refit view (`RefitView::cause`).
   - `First`: no fitted model stands (today's case; still decided by the model state as now).
   - `AfterPass`: the refit a sampling ("training") pass starts when it ends.
   - `Requested`: the refit `Action::RefitNow` starts — both "Refit now" buttons (the Forget/Restore banner and the manual-cadence one) and Activity's job row.
   - `Background`: every other refit (mid-pass, during play, the delayed Forget/Restore refit that wasn't hurried).
   - A refit that merges several requests takes the most visible cause among them (Requested > AfterPass > Background).
2. **The picture's rule** (`refit_picture::in_view`): a refit runs and (no fitted model stands, or its cause is `AfterPass` or `Requested`). Everything else is unchanged: the room the zones leave free, never input (the camera turns under it), no picture while a decision card (the model offer) is up, the fade-out when the fit lands, the Activity strip for every refit.
3. **Rule 5:** the cause travels as data through the existing refit probe; `sim` still doesn't know what computes forces, nothing downcasts a provider.
4. **Display only:** no fit, trust rule, sample or timing changes.

## Tests

- `refit_picture`: the view shows the picture for `First`, `AfterPass`, `Requested`; not for `Background` with a fitted model; not while the model offer is up (UI tests with a scripted probe, as `the_refit_picture_fills_the_view_only_while_there_is_no_model` does).
- The cause is set right where refits start: a pass's closing refit is `AfterPass`; `Action::RefitNow`'s is `Requested`; a mid-pass/play refit is `Background` (provider or worker tests with scripted collections, CPU).
- Gate: `cargo test -p app --release` (viewer / UI / worker), `cargo test -p providers --release` if providers change, run once on merged code.

## Left to the builder

Where exactly each refit is started and how the cause is threaded there; the field names.
