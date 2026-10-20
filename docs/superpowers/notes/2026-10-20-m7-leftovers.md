# M7 leftovers (2026-10-20, branch `m7-leftovers`)

The successor note of the M7 final review ledger
(`.superpowers/sdd-archive/Physics-m7/sdd/2026-10-15-m7-pseudopotentials/final-review.md`) for the minors that
were parked there (M1, M2, M5, M6), and of the plan's unfinished visual check (Task 13 Step 5). The ledger's wording
and suggested fixes were followed; each item below is either fixed with a test or ruled.

## M1 — the training views ignored the shell fence — **fixed** (`002418a`)

`SurrogateKs::fence_verdict` (`providers/src/surrogate/views.rs`; the last rule of `would_answer`, and the
pair curve's refusal marks) read `near_audit_fence` alone, so the Activity tab drew a fenced shell band
[5/6 r_cut, r_cut) as answered while strict mode refused it. It was also consulted only while samples were
quarantined, so an audit fence with nothing quarantined never reached the curve either (found while fixing M1).
It now takes `evaluate`'s order — a quarantined geometry, then `audit_fence_refusal` (the shell fence, then the
audit fence) — with the same rule and atom, whenever any fence stands.
Test: `surrogate::tests::the_training_curve_refuses_a_fenced_shell_band_as_evaluate_does` (a model trained on
1.2–5.0 Bohr answers just inside the band — the review's LiH case; after a failing H–H shell check every band point
is refused and names the shell fence's atom; `evaluate` agrees; an audit fence with nothing quarantined is on the
curve). It fails on the old code.

## M2 — shell checks judged against no model or a stale one — **fixed** (`a964f7e`, `0eaa4e7`)

`finish_pass` judged the pass's shell checks whether or not its refit ran. A check is now judged only against the
pass's **own** refit; with none (no samples, below Q′, the refit failed) each check is reported "could not be judged
(<why>) — the cutoff shell of this pair stays unmeasured" and nothing is fenced. A check still queued when the pass
ends (cancelled, a failed reference; `Sampler::pending_shell_checks`) is reported "the pass ended before its shell
check …". The `check_ratios == None` branch reports instead of a silent `continue`. Physics review (ship): a check
judged against a refit that left the model untrusted, or that holds none of the pass's samples, keeps its verdict
and fence (the safe side) and now says "(<caveat>: possibly the fit's error, not the cutoff's — the next pass's
shell check measures it again)".
Test: `surrogate::tests::shell_checks_are_judged_against_the_passs_own_refit_or_said_unmeasured`.

**Ruling on discarding the measurement.** A solved check with no fit of its own is dropped with the note rather than
held for a later refit: its verdict would then be a later, differently trained fit's, and the cost is one teacher
solve — the next pass over the scene solves it again (its scan crosses the limit again).

**Left (physics review Minors 3 and 4, recorded, not fixed):** two more ways a shell check is lost without the
"stays unmeasured" wording, both older than M2 — a shell geometry outside the cell is planned as a plain skip
(`sampling/mod.rs`, `Job::Skip("scans", …, None)`), and a stalled scan tail turns the pair's `Shell` job into a skip
(`skip_scan_tail`, counted in "(n not solved)" only). And with a fence standing, the views re-describe each fenced
case per answered curve point (`near_audit_fence`); cheap for dimers, measured only by
`training_views_build_time_is_reported`.

## M5 — the provenance tag encodes the dimer gate's procedure, not its verdict — **ruled: stays** (`5ddb10b`)

`"all/<recipe_hash>/…"` hashes the ledger and `DIMER_PROCEDURE`, not which f the H and He ladders settled on.

- **On one machine** this is exact: the verdict is a deterministic function of the procedure on that backend, and an
  `All` model cannot exist with H refused (`setup_for` errors).
- **Across machines and adapters** (the shipped `models/`; a disk-cache verdict replayed on another adapter) it holds
  only while every verdict lies outside the backend spread: 2.2e-3 Bohr in R_e (GPU vs CPU all-electron σ → 0 R_e),
  about 0.5 % in ω. The nearest verdict today is H f = 1.4 polynomial, 0.0037 Bohr outside its bound; the passing
  f = 1.2 is 0.005 Bohr and 1.55 % inside.
- **Why not fold the verdict into the tag:** `tag_of` must stay lock-free on the UI thread (final review I3), so it
  would have to wait for H and He to be generated, and every `All` tag would generate He (≈ 40 min on a CPU) — for no
  difference while the margin holds.
- **The premise is enforced:** `providers::pseudo::verdict_is_backend_robust` (margins `DIMER_BACKEND_MARGIN_RE` =
  2.5e-3 Bohr, `DIMER_BACKEND_MARGIN_OMEGA` = 0.5 %) is asserted on every rung the ladder asks for in
  `dimer_gate_runs_for_hydrogen_and_is_not_applicable_for_helium` (GPU, `cargo test -p providers --release`); the
  rule itself is unit-tested by `the_backend_margin_rule_reads_both_criteria`. A recipe change that brings a verdict
  within the margin fails there, and the tag must then carry the verdict. He's "not applicable" is not a pair verdict;
  its margins (AE F +7.27 / +0.13 Ha/Bohr at the bracket ends, wall agreement 0.3 % against 5 %) are far from any
  backend spread.
- **Model files:** the tag is unchanged, so every existing `.model` loads or is refused exactly as before — no file
  is silently mismatched by this change, and none is newly refused.

## M6 — the no-GPU offer could start a ≈ 40-min CPU generation without saying so — **fixed** (`8dee9f0`)

`providers::pseudo::GenerationCost` (`Ready` / `Seconds` / `GpuGate` / `CpuGate`) now rides on `generating()`: the
status line says "generating the H pseudopotential (≈ 40 min on the CPU: its dimer check (no GPU))" when it is, and
"(seconds)" when the disk cache replays the gate (before, it always said ≈ 1 min for H and He). The lock-free
`generation_cost(z)` (only `try_lock` on the cache) feeds the model offer: a switch to `All` whose H or He would go
through the gate on the CPU carries the row note "applying generates the H pseudopotential on the CPU first — about
40 min (the dimer check; this machine has no GPU for it), and the first solve waits for it". Said, not blocked —
the user may want it. README updated.
Tests: `pseudo::tests::the_generation_cost_names_the_cpu_gate`,
`ui::model_offer::tests::the_ionic_row_names_a_cpu_dimer_gate_before_apply`, and the CPU line in
`ui::tests::the_default_is_accepted_species_and_generation_is_said`.

## Task 13 Step 5 — water under `All`: the angle overlay and the lone pairs — **shot, awaiting the user's verdict**

Recorded in `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md` ("M7 Task 13"). In short:

- Relaxed water under `--ionic all` (95³ × 0.15, the new screenshot key `--ui wait=relax`): O–H 0.979 Å,
  ∠H–O–H 104.36°, the arc and "104.4°" at O (`docs/images/m7_water_angle_all.png`, volume opacity 0.15; at the
  default opacity the strands and the arc are lost in the volume's glow —
  `m7_water_angle_all_default_opacity.png`); the Selection tab names the valence-density kind
  (`m7_water_angle_all_selection.png`).
- Lone pairs: **not two lobes on the density isosurface** at any of 0.05–0.4 × max, in the default view or edge-on
  (`m7_water_iso_all.png`, `m7_water_iso_all_edge.png`). Design §8 publishes no threshold. The total density has no
  maxima at the lone pairs; they show in the HOMO 1b₁ (`m7_water_homo_all.png`: two lobes above and below the plane).
- Found and fixed on the way (`0b21cef`): the Orbitals tab's "smeared nuclei: Zσ = 1.2" note counted the pseudised O.
- No `h2o_angle_all.replay`: `--replay` ignores display keys, and the script would add the relax, the O reference and
  the H dimer gate (minutes) to `replays_pass`; `wait=relax` reproduces the shot.
