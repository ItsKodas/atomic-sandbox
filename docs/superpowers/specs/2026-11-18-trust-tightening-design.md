# Trust tightening: an evidence floor and tail-aware inflations (2026-11-18, rev. 2)

Design note only. No code, tolerance, tier or test changes here. Atomic units: Ha, Ha/Bohr.

- A force ratio is e_j/b_j, where e_j = ‖ΔF_j‖/√3 and b_j = min(max(2e-3, 10 %|F_j|), 1e-2) is the
  policy's per-atom bound. Likewise s_j = σ_epi,j/√3.
- An energy ratio is |ΔE|/N over the tier, 1e-3 Ha/atom.

Rev. 1 takes the physics review (`.superpowers/spec-review/trust-tightening-physics-review-1.md`,
APPROVE WITH CHANGES): C1, I1–I5 and M1–M3. Changes from rev. 0:

- The mean-of-ratios λ_A is replaced by the tail-aware λ_T (forces) and λ_E (energy).
- The floor counts fits, not generations, and needs ≥ 60 points as well as ≥ 12 fits.
- Fenced points and points freshened by a refusal are left out of the inflation's sample.
- The calibration ticks get their own cadence.
- A read-only report on the user's BEST collection is a go/no-go gate before implementation.

Rev. 2 takes the re-review of rev. 1 (APPROVE WITH CHANGES; all three departures accepted):

- λ_E's max condition is 1× the tier.
- In the app, ticks are served only on the auditor; a busy auditor means a skipped, counted tick.
- The cadence-fresh flag reads its own counter, which nothing resets.

## 1. Intent and decisions

The user, 2026-11-17: **"Tighten trust"**.

- (a) "require more than 3 evidence points (checks) before a class verdict can open";
- (b) "widen the model's uncertainty using a variance inflation calibrated from measured audit
  residuals".

The user, 2026-11-18, on (a)'s reading: **"Keep it, gated"**.

- A class may still open on the zero-evidence leave-one-out (LOO) rule after a sampling pass.
- Below the floor, the calibrated verdict cannot open a class alone.
- This is settled and not reopened here.

The aim is to close the trust gap that `the_stuck_escape_repro_is_answered_honestly` catches on some
trajectories. Its tolerances stay as they are (rule 6). Every change below can only refuse more often
than today.

## 2. The mechanism (bisect: `scf-stall/.superpowers/sdd/2026-11-13-fermi-smearing/stuck-escape-bisect.md`)

**What flipped.**

- 762159b moved converged labels by about 1e-7 Ha, inside the 1e-6 SCF tolerance. The teacher is
  unchanged at the failing geometries.
- The closed loop of label, fit, answers and trajectory amplifies that shift through the class LOO on
  a κ₂ ≈ 6e12 fit. Already at the tail sample, the pair LOO max moves from 15.084× to 15.120×.
- By the end of the NVE run, the many-body verdict has flipped:
  - 07939ac: untrusted, κ 5.85;
  - 762159b: trusted on 3 evidence points, LOO force rms 0.32×, max 3.07×.
- With audits off, that count can never exceed 3. The fallback weight cap max(n_a, 3) holds the
  count n_a + min(n_f, cap) at 3, however many of the run's 10 rule-4b fallbacks there are.

**The hot run (762159b).** It records 0 rule-4b points and gives 657 answers, 248 of them judged.
Twelve are beyond the gate (steps 275–277, 485–489, 700–703):

- 11 of the 12 break the energy tier (worst 2.38e-3).
- 7 break it alone: 485–489, 700 and 701, at |ΔE|/N 1.03–1.39e-3 with force ratios 1.56–2.89.
- The worst force ratio is 4.48.

So energy, not force, is the main failure mode.

**The judged z** = max_j ‖ΔF_j‖/√(σ_epi,j² + 3σ_f²) (`se-762159b.log` line 30). This is an upper
bound on a per-key z. I recomputed the review's numbers from the log, and they reproduce:

| quantity | value |
|---|---|
| median | 2.65 |
| rms | 5.82 |
| p95 | 12.0 |
| max | 34.2 |
| CV²(z²) | 10.05 (a χ²₁ variable gives 2) |
| answers with z > 18 | 5 of 248 |
| lag-1 autocorrelation of z² | 0.63 |
| CV² of 25-answer block means | 0.84–1.0 (0.84 over 9 whole blocks; 0.95 or 1.0 depending on how the partial block is treated) |
| τ_int of z² | 2.8 (empirical 1 + 2Σρ_k, the re-review) |

- **AR(1) gives a larger τ_int.** From ρ₁ alone, (1 + ρ)/(1 − ρ) gives 4.4. That overestimates, so
  τ_int 2.8–4.4 is used below as a conservative range. No requirement changes.
- **s ≈ 0.17 b at the z > 18 answers** cannot be checked: the log prints no stds.

**Why the existing λ did not stop it.**

- λ_sum is a ratio of sums, √(Σ w̄²/Σ s̄²), over audits and rule-4b/6/7 fallbacks (`calibrate.rs`).
  It is dominated by the high-std points, so it cannot see confident-point overconfidence.
- No comparison of any answer ever reached the history: there were no audits, and the class was not
  refused.

**Two weaknesses follow.**

1. A class verdict opens on few, correlated points.
2. No inflation sees the tail of the population that rule 6 lets through, in force or in energy.

Neither fix makes the trajectory less chaotic. They change what a flip can admit.

## 3. (a) The evidence floor

**The verdict.** "Evidence" is today's weighted count, and the floor is the new count below.

- **Floor met:** today's calibrated verdict, unchanged. It enters at V ≤ `ENTER`, stays at
  V ≤ `LEAVE`, and keeps the worst-atom guard on entry.
- **Evidence ≥ 3, floor not met:** trusted only if **both** of these pass:
  - the rule of before (`trust::class_refusal`: LOO force rms and max ≤ 1, LOO energy max ≤ 1);
  - today's calibrated verdict.

  The worst-atom guard then applies both ways, as today below 3.
- **Evidence < 3:** today's rule, unchanged.

At every count this is today's verdict ANDed with a further condition, or identical to today. The
measured case, LOO max 3.07× on 3 points, stays shut.

**The floor: n_fit ≥ 12 distinct fits *and* n_pts ≥ 60 points (both my call).** Both are counted
over the key's attributed verdict points (audits and codes 5/6) inside its evidence window.

- **The criterion.** Two sigma of the verdict value's scatter must fit inside the √2 band. For an rms
  over n draws whose squares have coefficient of variation CV², the relative scatter is ≈ ½√(CV²/n).
  So the condition is n ≥ CV²/(√2 − 1)² = 5.83·CV² (review I2).
- **12 fits.** This holds when each fit pools one decorrelated block. In the harness a fit is one
  25-step refit interval, about 6–9 τ_int. The measured block CV² is 0.84, which needs n ≥ 4.9, so 12
  has margin.
- **60 points.** This covers fits of only 1–3 points, as in the app, where the worker refits for
  every kept sample. The per-answer CV² proxy of 10 needs n ≥ 59.
  - The 60 points are correlated: at τ_int 2.8–4.4 they are 14–21 effective draws.
  - That is why both conditions are needed. Neither alone holds in both regimes.
- **Why W keeps its per-point rms.** W is not recomputed from per-fit mean squares (the review's
  other option). Re-weighting per fit can loosen: one fit holding many bad points gets diluted by
  benign one-point fits.
- **A correction to rev. 0.** Its "one point must be 2.1×/1.1× the rms" was wrong. Replacing one of n
  draws moves the rms across the band when that draw is √(n+1)× the rms: 2.0× at n = 3, 3.6× at
  n = 12 (review).

**A fit, not a generation (review I1).**

- `model_gen` is also bumped by:
  - `set_policy` (policy.rs:342; the quiet-refits branch removes this bump for draft and
    audit-cadence changes);
  - restore (trust.rs:1803), forget (store.rs:124) and install/attach (shared.rs:311, store.rs:452).

  Counting generations would let a parked scene reach 12 by settings changes alone.
- Each point therefore records a **fit id**, written as an optional history key. It is a counter
  bumped only by a refit that solved new coefficients: never by a policy change, restore, forget or
  install.
- A point without a fit id does not count. Every point saved before this change is such a point.
- **Over-count, documented.** One key's refit advances every key's fit id. A parked key whose own
  fallbacks are thinned still gains fits from other keys' samples.
  - A stricter count would use the key's own sample epoch: n_k at judgement time. It needs a
    per-key list per point.
  - It is held in reserve, in case the BEST report (§6) or the repro shows the over-count matters.
- Species-less (legacy) points never count toward the floor, as in rev. 0. They still enter W and the
  guard.

**What stays at 3.** These uses of `MIN_EVIDENCE` are unchanged, because raising any of them would
loosen or is a different question:

- the fallback weight cap, so W does not move;
- λ_sum's threshold;
- the segment-energy count (trust.rs ≈ 1065);
- the class floor in `key_inflation`.

New wording:

- a `ClassKey::Floor` refusal: "within the bound over n_fit of 12 fits / n_pts of 60 checks, and its
  leave-one-out is X× the bound";
- the worst-atom wording follows the floor (`refusal_words`, `app/src/ui/activity/words.rs`).

**Saved models and collections are re-judged at load, with no grandfathering (my call).**

- Old points carry no fit id, so a saved key with evidence ≥ 3 and LOO max > 1 loads untrusted.
- **The knife edge returns there (review I3).** Below the floor the LOO max decides again, the max rule
  that the calibrated verdict replaced. On record: the user's water + H₂ file at LOO max 12.0×/10.6×.
- **The dead end.** A refused key's fallbacks at a parked scene are thinned: no sample, no refit, no
  new fit. Its count cannot grow there, by design: refits on near-copies would be the I1 over-count.
  The way out is the existing one for a LOO-blocked class:
  - the teach prompt, after `TEACH_PROMPT_STREAK` class fallbacks (teach-around brings LOO ≤ 1);
  - or moving the scene.
- That is why the BEST report is a precondition (§6).

## 4. (b) Tail-aware inflations λ_T and λ_E

**Rules 6 and 7.** For an atom j of key k:

```text
λ_F(j) = max(λ_sum(k), λ_T(k))
rule 6:  λ_F(j) · s_j ≤ b_j
rule 7:  max(max_j λ_F(j), max_k λ_E(k)) · σ_E,epi/N ≤ tier
```

- b_j and the tier never move.
- Since λ ≥ λ_sum ≥ 1, no inflation ever shrinks the posterior's std.
- Any NaN among the terms makes the max NaN (written as trust.rs:105 does), and a NaN or ∞ factor
  refuses (M1).
- **A sparse key** (fewer than 3 points in its own sample) takes its class's λ_T and λ_E as floors, as
  `key_inflation` does for λ_sum (M2).

**λ_T, in bound units (review C1, change 2).** Over the key's atoms in its sample P_A (below), inside
the window:

- Let r_j = s_j/b_j, and let A(λ) = {atoms with λ·r_j ≤ 1}: the measured atoms rule 6 would let
  through.
- A set A is *good* when both hold:
  - **max condition:** max over A of e_j/b_j ≤ `WORST_ATOM` (3), at every size;
  - **rms condition:** the rms over A's points of the mean (e_j/b_j)² of each point's admitted atoms,
    × (1 + 1/√(2n_A)), is ≤ `ENTER`. This condition applies only when A itself meets the floor (12
    fits, 60 points).
- **λ_T** = the smallest λ ≥ 1 such that A(λ′) is good for every λ′ ≥ λ. The sets are nested (sorted
  by r), so this is one pass over the sorted atoms. If a bad atom has r = 0, λ_T = ∞.
- Restricting the rms condition to sets that meet the floor is my addition to the review's
  definition. Without it, a single admitted atom at 0.8× the bound would make λ_T = ∞.
- **Why bound units.** Only errors that matter to the policy inflate λ_T. A harmless confident point
  (e = 0.3 b at s = ā) leaves λ_T = 1. Under rev. 0's mean of ratios with equal weight per generation,
  that same point gave λ_A ≈ 5.3. That was the knife edge which `calibrate.rs` lines 259–263 rejected
  (review M3), so the mean of ratios is gone.
- **What λ_T does.** It applies the verdict and the worst-atom guard to exactly the atoms rule 6
  would admit.
- **What it does not do (C1, change 3, corrected).** It bounds the tail *of what was measured*. An
  answer at a geometry like none in P_A is still unmeasured.
  - Rev. 0 claimed λ_A addressed overconfidence at confident geometries. That was true only on
    average: a mean of ratios gives ≈ 7.1 on this run, where the 4.48× answer needs about 7.7.

**λ_E, the same construction in tier units (review C1, change 1).**

- Per point, r_p = σ_E,epi,p/N_p/tier and e_p = |ΔE_p|/N_p/tier. Each point counts for every key it
  holds, as the LOO energy does.
- **Max condition: e_p ≤ 1, i.e. 1× the tier** (the controller's ruling on the re-review's
  recommendation).
  - The max condition is the answer-level criterion the validation asserts. On the force side
    `WORST_ATOM` = `REPRO_FORCE_RATIO` = 3 already. The gate checks every answer's energy at 1× the
    tier, so λ_E's max is 1.
  - It only tightens: an admitted energy beyond the tier is an answer the gate already calls wrong.
  - The over-refusal cost is small: benign energy errors are about 1e-5 to 1e-4 Ha/atom.
  - Rev. 1's 4.53 (equal rarity with `WORST_ATOM`) never fired on the energy-only failures, e_p
    1.03–1.39. The rms condition does not catch them either: it needs the 60-point floor, and the
    hot run's admitted rms is ≈ 0.26 < `ENTER`, which gives λ_E = 1.
- **Rms condition:** at `ENTER`, applied as for λ_T.
- **The aleatoric floor** (N·s_E, about 1e-5 Ha per atom) is 1 % of the tier and is not subtracted.
- **The epistemic energy std** is not in the history: `pred_energy_std` is the total std
  (history.rs:22). Each point gains an optional `energy_std_epistemic` key, and a point without it is
  not in λ_E's sample.
- The M5 bias of z_E is moot here, because nothing divides by σ_E.

**The sample P_A: chosen without reading σ or λ (the self-reference guard).** A point is in P_A only
if a history flags word (one new optional key, bits below) marks it **not fenced** and **fresh by
cadence**, and it is one of these:

- **(i) Class-refused in-envelope steps (codes 5, 6 and 21).** The class verdict is decided by measured
  errors and LOO, never by this step's σ. Every such step is recorded, so the union of the three codes
  is complete whatever λ is.
- **(ii) Calibration ticks.** These sample a trusted key.

**Fenced (review I4).** The class reason is set at mod.rs:2469. The quarantine, misfit and audit fences
are consulted only while no reason is set (2478, 2492, 2516), so codes 5/6/21 can hold geometries that
the key will never answer.

- `near` and `near_misfit` are computed every step. The audit fence is evaluated for the bit when the
  class has already refused.
- A fenced point is excluded from P_A.

**Fresh by cadence (review I4).** The variance, and with it rule 5d's novelty test (2387, 2429), is
computed on a step when either:

- the consult counter reaches `variance_cadence` (σ-blind); or
- the previous verdict refused or the counts changed (σ-coupled).

`consults` resets on every fresh step, σ-coupled ones included (mod.rs:2388), so its phase reads σ.
The flag therefore uses **a separate counter that nothing resets**: a step is cadence-fresh when that
counter is a multiple of `variance_cadence`. Only points recorded on cadence-fresh steps, or on ticks,
enter P_A. A tick decides before the variance is computed and forces it.

**Ticks on their own cadence (review I5): every 25th in-envelope step, per provider (my call).**

- The cadence does not depend on `audit_every`. The default `audit_every` of 2000 would let a trusted
  key's 300-point window never age, and the gate's 0 gives no ticks at all.
- **Why 25.**
  - It is one fit interval of the harness and about 6–9 τ_int of z², so successive ticks are near
    independent.
- **How ticks are served.**
  - **In the app,** ticks go asynchronously to the background auditor only, never in-step.
  - A tick that falls due while the auditor is busy is **skipped, not deferred**, whatever answers
    the step, so the skip does not read σ. Skips are counted and reported.
  - A tick at a fallback needs no solve: the teacher's answer and the model's std are both at hand.
  - A tick at an answered step is an extra audit, judged by rule 8 as well.
  - **In-step ticks exist only on paths with no auditor:** the CPU in-step path and the harness.
- **The effective cadence** is max(25, t_solve/t_step) in-envelope steps:
  - ≈ 25 at 12 ms/step;
  - 50–55 at 5 ms/step (solves of 130–270 ms);
  - a trusted key's window turns over in 7500–16000 steps.
- **Cost.**
  - Wall time ≈ 0, plus contention for the GPU.
  - σ-coupling ≈ 1 %: the auditor is otherwise busy only with rule-8 audits and their bursts, at
    `audit_every` 2000.
  - A tick lands in a 3–5-step tail cluster with a chance of ≈ 6–10 % (rev. 1: 12–20 %). That is
    acceptable because the floor and the class-refused points do most of the catching.
- **Rejected: ticks only for keys near their bound.** Choosing by nearness reads σ.
- Rule 8's own cadence is unchanged.

**Left out of P_A**, because λ selects them:

- ordinary audits;
- codes 14 and 15;
- a new **code 24**, "refused by λ_T or λ_E alone (it passes at λ_sum)", recorded for the Log and the
  measurement.

**No lockout follows.** A refusing key's fallbacks are free ticks or class-refused points, so the
inflation is always re-measured on steps it did not choose. A λ_T of ∞ lasts until the offending
point ages out of the window.

**Audits are never judged by λ.** Rule 8 compares errors with b_j and the tier only.

**The inflations move only at refits, installs and policy changes**, never on a step they decided.
They have no hysteresis.

**λ_T and λ_E enter the class verdict only through its jurisdiction**, as λ_sum does today: a
class-refused step that the inflated rule 6 would refuse is coded 21. A refusal they decide names the
factor and its measured cause, e.g. "λ_T 7.7: a measured atom at 4.5× the bound with the model's std
0.13× of it".

## 5. Interactions

- **Hysteresis.**
  - The band is unchanged when the floor is met.
  - Below the floor, the LOO half has no band, as today.
  - When aging drops a trusted key below the floor, it can leave through the LOO half.
- **Re-judging on new evidence.** The floor's counts are recomputed with the verdict. Leaving is
  immediate. Entering between refits still needs a full window (`ClassKey::Window`).
- **Misfit fence and M.** Fenced points are out of P_A. M stays in the LOO RMS sums that the LOO half
  reads.
- **Teach-around** (`teach_verdict`).
  - Acceptance already requires LOO ≤ 1, which implies the LOO half.
  - Teaching checks are not ticks.
  - `teaching_around_a_blocker_restores_its_class` should not move; if it does, that is a finding.
- **Collections and migration (Task 14).** Collections are re-judged at install. A re-solved sample
  is a sample, not a point.
- **Stray atoms (branch `stray-atoms`, I4/I6).** Compatible.
  - Rule 7 and audits there divide by n_bonded. The inflations multiply σ_E,epi before the division,
    and λ_E's e_p and r_p use the same N, so set N = n_bonded there.
  - Stray atoms are class 0: they carry no key, no P_A atoms and no fits.
- **Draft mode.** `DraftAudit` points are not in P_A.

## 6. Validation

**Task 0, before any implementation: the go/no-go (review I3).**

- An `#[ignore]`d, CPU-only report opens
  `models/collections/lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5` read-only (Task 13's
  read-only open) and never writes it.
- It computes the new verdict as pure functions over the stored history and the fit's LOO.
- Per key it prints:
  - LOO force rms and max, LOO energy max;
  - evidence, n_fit (0 for every legacy point) and n_pts;
  - the verdict today and under this design;
  - λ_sum. Legacy points carry no flags, so λ_T and λ_E start at 1.
- **The count of keys that lose trust at load, with their scenes' words, goes to the user.**
  Implementation proceeds only on the user's go.

**Gates (rule 6).**

- `the_stuck_escape_repro_is_answered_honestly`, with every constant as it is: `REPRO_FORCE_RATIO`
  3, tier 1e-3, `REPRO_MIN_JUDGED_HOT` 10, `REPRO_BOOKING_RESIDUAL` 1e-9.
- **Risks to the gate:**
  - **The check is two-sided.** It needs ≥ 10 judged answers or ≥ 10 warranted refusals. Code-24
    refusals of a trusted class count toward neither, and 07939ac's refused steps sat at rms 0.46,
    not "warranted".
  - **The harness has no auditor, so its ticks are in-step.** A tick at an answered step there
    becomes an in-step audit, which changes the trajectory and which steps are judged answers.
  - **An unmeasured first tail cluster can still be answered.** A tick lands in a 3–5-step cluster
    with a chance of ≈ 6–10 % in the app, and ≈ 12–20 % at the harness's in-step 25.
- **If the gate fails**, the next step is policy (review Q8), never a tolerance.
- **Suites:**
  - `cargo test -p providers --release`;
  - `validate --test surrogate` and `--test surrogate_water`;
  - `cargo test -p app --release` (the words).

**Pinned tests expected to move.** The rule changes, not a tolerance; each moved value gets the new
rule's derivation in its doc.

- `a_class_refused_for_one_outlier_earns_the_calibrated_verdict_and_holds`;
- `a_class_refused_on_its_evidence_reenters_after_a_teaching_refit`;
- `evidence_ages_by_model_generation_both_ways` (the window keeps generations; the floor counts
  fits);
- `h2_plus_h_falls_back_on_the_grade_until_enough_is_learned` (26/40, 40/40);
- `active_learning_run_converges` (its DFT-call count, now with ticks);
- `surrogate_knows_what_it_does_not_know`;
- `references.md` "M8a calibrated verdict".

The evidence-count test near tests.rs:9070 should not move: the cap stays at 3.

**New unit tests.**

1. `a_class_below_the_floor_needs_its_leave_one_out_as_well`:
   - 11 fits/80 points or 12 fits/59 points, V ≤ `ENTER`, LOO max 3.07 → untrusted
     (`ClassKey::Floor`);
   - 12 fits/60 points → trusted;
   - LOO max 0.9 below the floor → trusted.
2. `the_floor_counts_fits_not_generations`: 12 policy changes, a restore and an install at one
   geometry add no fit; 50 points of one fit count 1.
3. `points_without_a_fit_id_or_species_do_not_open_a_key`: they raise W and trip the guard, and add
   nothing to the floor.
4. `a_class_saved_open_on_three_points_loads_closed`: one with 12 fits and 60 points inside the band
   stays trusted.
5. `the_tail_inflation_refuses_the_measured_tail`:
   - 80 benign points over 16 fits at r ∈ [0.05, 0.9], e/b ≤ 0.5, plus one atom at r = 0.13 with e/b = 4.48;
   - λ_T is just above 1/0.13 ≈ 7.7, and rule 6 refuses r > 0.13;
   - λ_sum stays below 2.
6. `a_harmless_confident_point_does_not_inflate`: e = 0.3 b at r = ā/b = 0.02 among 12 benign fits →
   λ_T = 1 (the M3 case).
7. `the_energy_inflation_refuses_energy_only_failures`:
   - fewer points than the floor: some at e_p 1.03–1.39 and r_p 0.2, beside benign ones at
     e_p ≤ 0.1 and r_p up to 0.9;
   - λ_E is just above 1/0.2 = 5, from the max condition alone, because the rms condition does not
     apply below the floor;
   - rule 7 refuses at r_p > 0.2 while the forces pass;
   - with rev. 1's 4.53, λ_E would be 1.
8. `the_inflations_never_read_what_they_selected`: λ_T and λ_E stay bit-identical under added code
   14/15/24 points, ordinary audits, fenced points, or points fresh only by a refusal.
9. `no_inflation_shrinks_the_std`: λ ≥ λ_sum ≥ 1 on random histories.
10. `a_calibration_tick_does_not_read_the_std`:
    - σ scaled ×10 gives the same ticks;
    - with an auditor, a tick due while it is busy is skipped and counted, whether the step was
      answered or not, and is never served in-step;
    - with no auditor, ticks are served in-step.
11. `non_finite_and_zero_rules` (M1):
    - e = s = 0 adds nothing;
    - s = 0 with e/b > 3 gives λ_T = ∞, and rule 6 refuses the key's atoms, s = 0 included;
    - NaN anywhere gives NaN, which refuses (NaN is never dropped by `max`).
12. `a_sparse_key_takes_its_class_inflations_as_floors` (M2).

**Measurements (no gates).**

- **The repro, per run and key:**
  - n_fit, n_pts, λ_sum, λ_T, λ_E;
  - ticks taken, skipped while the auditor was busy, and served in-step (harness);
  - code-24 refusals;
  - judged z, raw and inflated (‖ΔF‖/(λ_F σ_epi));
  - worst ratios;
  - solves and wall time against 1036 s (07939ac) and 1378 s (762159b).
- **`trust_baseline_report`:** z rms per class and missed large errors, before and after.
- **The BEST report of Task 0**, re-run after implementation.

## 7. Approaches, what is left, open questions

**Approaches rejected.**

- **Raise `MIN_EVIDENCE` everywhere.** It loosens λ_sum's threshold, moves W's cap and drops today's
  measured refusals.
- **Widen λ_sum.** A ratio of sums cannot see a confident tail.
- **Rev. 0's mean-of-ratios λ_A.** It fixes only overconfidence spread evenly, and revives the
  knife edge ruled out before (M3).
- **Horvitz–Thompson weighted audits.** Unbiased, but one audit stands for about 2000 steps.

**Chosen:** the gated floor, with λ_T and λ_E on a σ-blind sample.

**Left to the implementer.**

- The history keys: fit id, flags word, `energy_std_epistemic`.
- Code 24 in `history_rule_name`.
- Where the tick counter and the never-reset cadence counter live (the async auditor, the in-step path).
- The panel readouts.
- The README line at the merge.

**Open questions for the physics reviewer.**

1. **λ_T's rms condition uses `ENTER`** (the review's choice). Should a trusted key's admitted set be
   held to `LEAVE` instead, to match the verdict's stay band?
2. **The fit-id over-count from other keys' refits.** Should the stricter per-key sample epoch be the
   default rather than the reserve?
3. **Ticks every 25 steps.** Should the cadence scale with each key's measured τ_int?
