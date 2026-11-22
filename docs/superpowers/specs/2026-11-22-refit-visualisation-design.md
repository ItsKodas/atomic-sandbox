# Refit visualisation — design (short)

**Approved by the user 2026-11-22** ("I like that plan", "lets do that"). Build after the weekly usage reset (24 Nov), beside the faster trust recovery.

## Why

The first fit at launch takes 1½–2 min on a four-element collection, and a trust recovery up to 6 min. The only display is a bar whose estimate comes from the last refits. It is often badly wrong: "about 0:05 left" four minutes into a six-round recovery. The user wants to see what the refit is actually doing.

## What (the user's choice: show where the time really goes, not per-sample blocks alone)

A refit's time is not spent per sample: summing 2 800 samples takes under 10 ms, while the hyperparameter search's decompositions of the Q′ × Q′ matrix (Q′ = 6 656) take most of the time. So the picture shows the real stages:

1. **Gathering:** one block per sample, filling as the samples are read or described.
2. **Solving the model:** a square grid (the matrix) filling as each decomposition sweeps it, with its number ("search 3 of 5"). Progress counters go only in the code we write (QL sweeps per block, the spectrum loop); a library call shows as one step.
3. **Checking each sample:** the leave-one-out pass. The sample blocks light up one by one with their ratio: green within the bound, amber beyond it. These are the same ratios as the worst-fitting list.
4. **Trust:** in a recovery, its round ("round 4 of up to N") and the samples it holds out.

**Where (the user's choice):**
- **The 3D view** shows the full picture only while nothing else needs it: the first fit at launch, or any refit while there is no model. Play is refused then anyway. It fades out when the fit lands.
- **Activity's refit row** shows a compact strip of the same four stages and counters for every other refit (background, while playing), so the view stays on the simulation.

**The estimate** comes from the stage counters, not from the last refits' walls. A stage that has not started yet has no number.

## Not doing

- Per-sample blocks alone (they would fill in a second and then sit still).
- Any change to what a refit computes. Display only; the counters are relaxed atomics, read by the probe.

## Approaches considered

1. **The stage picture in the view at the first fit, a strip in Activity otherwise (chosen).**
2. **Per-sample blocks only:** dishonest about where the time goes.
3. **Activity only:** at launch the view is empty and Play is blocked, so the view is the natural place.

## Left to the builder

- The layout and colours, within the theme. Blocks wrap to the view; at more than a few thousand samples, one block per k samples.
- How the per-sample leave-one-out ratios reach the UI: a shared buffer filled by the LOO workers, read once per frame.

## Gates

- `cargo test -p app --release`.
- `cargo test -p surrogate --release` and `cargo test -p providers --release` for the counters.
- No change to fit results: the existing fit goldens must stay bitwise.
