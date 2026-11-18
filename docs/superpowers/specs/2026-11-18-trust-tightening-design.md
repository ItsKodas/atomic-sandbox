# Trust tightening: an evidence floor and an answer-population inflation (2026-11-18)

Design note only. Nothing here changes code, a tolerance, a tier or a test. Atomic units throughout:
energies in Ha, forces in Ha/Bohr. Every force ratio is ‖ΔF_j‖/√3 over the policy's per-atom bound
b_j = min(max(2e-3, 10 %|F_j|), 1e-2). Every energy ratio is |ΔE|/N over the tier, 1e-3 Ha/atom.

## 1. Intent and the user's decision

The user decided on 2026-11-17 to **"Tighten trust"**:

- (a) "require more than 3 evidence points (checks) before a class verdict can open";
- (b) "widen the model's uncertainty using a variance inflation calibrated from measured audit
  residuals".

The aim is to close the trust-policy gap that `the_stuck_escape_repro_is_answered_honestly`
(`crates/validate/tests/surrogate.rs`) catches on some trajectories. The gate's tolerances stay
exactly as they are (rule 6). Each change below may only refuse more often than today. None loosens
anything.

## 2. The mechanism (bisect: `scf-stall/.superpowers/sdd/2026-11-13-fermi-smearing/stuck-escape-bisect.md`)

- **First bad commit: 762159b** (Task 9d fix round 3). It moved converged GPU labels by about 1e-7 Ha,
  which is inside the 1e-6 SCF tolerance. The teacher is unchanged at the failing geometries: E
  agrees to 1e-7 or better, F to 1e-6.
- **The closed loop turns that shift into a verdict flip.** Labels feed the fit, the fit decides which
  steps the model answers, and the answers shape the trajectory. The class leave-one-out runs on a fit
  with κ₂ ≈ 6e12. There, the 1e-7 shift already moves the tail sample's pair LOO max in the third
  digit (15.084× → 15.120×).
- **By the end of the edited NVE run the many-body verdict has flipped.**
  - 07939ac: untrusted, κ 5.85.
  - 762159b: trusted, κ 3.25, LOO force rms 0.32× and max 3.07×.
  - Both rest on **"3 evidence"**. The run has no audits (`audit_every` 0), so the weighted count
    n_a + min(n_f, max(n_a, `MIN_EVIDENCE`)) cannot exceed 3, whatever the number of rule-4b
    fallbacks. The NVE run had 10 rule-4b fallbacks in all, and their would-be answers had rms
    0.27× the bound.
- **The hot run then fails.** At 762159b it starts with many-body trusted and records 0 rule-4b
  points. It gives 657 answers, 248 of them judged. Twelve are beyond the gate, at steps 275–277,
  485–489 and 700–703:
  - worst |ΔE|/N 2.38e-3 against the 1e-3 tier;
  - worst force ratio 4.48 against `REPRO_FORCE_RATIO` 3;
  - grades γ 1.06–1.89;
  - the test's z = max_j ‖ΔF_j‖/√(σ_epi,j² + 3σ_f²) up to **34.2**, where a calibrated posterior
    gives about 1.
- **Why the existing λ did not stop it.** The class's existing variance inflation λ
  (`calibrate.rs`, "The variance inflation") is a **ratio of sums**, √(Σ w̄²/Σ s̄²). It is computed
  over audits and over the rule 4b, 6 and 7 fallbacks. By design that ratio is dominated by the
  points whose std is a large fraction of the bound. The failing answers sit at the other end: the
  epistemic std per component is about 0.13 of the bound (estimated: z ≈ 34 at a ratio of 4.48),
  with errors 2–4.5× the bound. No comparison of such an answer ever reached the history, because
  there were no audits and the class was not refused.

Two weaknesses follow:

1. A class verdict can open on 3 correlated points.
2. The inflation is blind to overconfidence at confident geometries.

(a) and (b) address one each. Neither makes the trajectory less chaotic. A verdict sitting on its
threshold can still flip on 1e-7 of label noise. What changes is what a flip can admit.

## 3. (a) The evidence floor

**Reading of the decision (user ruling 2026-11-18: "Keep it, gated").** The user chose this reading over the literal one: a class may still open on the zero-evidence LOO path after a pass, and below the floor the calibrated verdict cannot open a class alone. The verdict that opened on 3 points is the
*calibrated* one, `ClassCalibration::calibrated()`, true at evidence ≥ `MIN_EVIDENCE` = 3. The
zero-evidence path is the rule of before, `trust::class_refusal`: LOO force rms ≤ 1 and max ≤ 1, LOO
energy max ≤ 1. That path is how every class opens after a sampling pass, and it **refused** this
class (LOO max 3.07× > 1).

So the floor gates the calibrated opening. Below it, the calibrated verdict can still refuse, but it
cannot open a class by itself. The verdict becomes:

- **n_ind ≥ `OPEN_GENERATIONS`**: the calibrated verdict of today, unchanged. It enters at V ≤
  `ENTER`, stays at V ≤ `LEAVE`, and keeps the worst-atom guard on entry.
- **n_ind below the floor and evidence ≥ 3**: trusted only if **both** verdicts pass:
  - the rule of before (`class_refusal`);
  - today's calibrated verdict with its hysteresis.

  The worst-atom guard then applies both ways, as it does today below 3.
- **Evidence < 3**: today's rule, unchanged.

At every count this refuses at least as often as today:

- At 3 ≤ evidence with n_ind below the floor it is the AND of today's verdict and a further
  condition.
- Elsewhere it is identical to today.

Moving those classes onto the rule of before *alone* (approach A in §7) would have dropped today's
measured refusal at evidence 3 to 11, so that would loosen.

**The value: `OPEN_GENERATIONS` = 12 (my call).**

- *Why 12.* The verdict value is an rms of n draws, and its relative one-sigma scatter is about
  1/√(2n) for χ²₁-like draws (`calibrate.rs`, "Why 300", requirement 1). The hysteresis band is the
  factor √2 between `ENTER` and `LEAVE`.
  - Today the band equals one sigma at n = 3. Then one point can carry the verdict across it: the
    measured case, 07939ac against 762159b.
  - Requiring the band to hold **two** sigma gives 2/√(2n) ≤ √2 − 1, so n ≥ 11.7, so 12.
  - At 12, one point holds 1/12 of the variance instead of 1/3. For a single χ²₁ draw to move the rms
    across the band it must be about 2.1× the rms instead of about 1.1×.
- *Its limits (rule 7).*
  - The argument assumes Gaussian-like draws. The many-body error is heavy-tailed: refused-step
    max/rms is 4–6, against about 1.9 for a Gaussian. The tail is the worst-atom guard's job, and
    open question 2 asks for the kurtosis to be measured.
  - It does not bound the *first* entry against chaos. It bounds how little evidence an entry can
    rest on.
- *In the measured case.* The NVE run's 10 rule-4b points are fewer than 12 even counted as points.
  With LOO max 3.07 the class would have stayed untrusted, and the hot run would have begun refused,
  as at 07939ac (299 rule-4b refusals). Whether the hot run then passes is for the measurement (§6)
  to say, not this note.

**What counts as an independent point: one model generation (my call).**

- n_ind is the number of distinct `AuditPoint::model_gen` values among the key's verdict points
  (audits and codes 5/6) inside its evidence window.
- *Why generations.*
  - `AuditPoint` carries no positions or descriptors, so a geometry test would need a history format
    change.
  - Within one generation the points are consecutive MD steps under one model, which are serially
    correlated.
  - A generation is already the window's unit ("a generation is never split").
  - Generations follow geometric distinctness. In the app the worker refits whenever a sample is
    kept, and a rule-4b fallback "like one already held" is thinned (not kept), so a new generation
    needs a new kept geometry. In the test harness it means one refit interval (25 steps) with a
    refusal in it.
- *A parked scene keeps one generation.* Its class cannot open by calibration. That is the safe side,
  and it matches today's rule that "entering between refits needs a full window".
- **Species-less (legacy) points never count toward n_ind (my call).** Today they count for every key
  of their class, which is the conservative reading *for refusing*. Counting them toward a floor
  would let them help open a key whose atoms they may not hold. They still enter W and the guard as
  today.

**What stays at 3.** `MIN_EVIDENCE` is used for five things, and only one of them is this decision:

- `ClassCalibration::calibrated` moves to the conjunction above.
- **These stay at 3:**
  - the fallback weight cap max(n_a, 3), so W's values do not move;
  - λ's threshold (`variance_inflation`). Raising it would set λ = 1 for keys with 3 to 11 points,
    which loosens;
  - the segment-energy count (`trust.rs` near line 1065), a different question;
  - the sparse-key class floor in `key_inflation`.
- The worst-atom wording (`refusal_words`, `app/src/ui/activity/words.rs`) changes from "while fewer
  than 3 comparisons judge the class" to the floor's words.
- The implementer adds a new `ClassKey::Floor` refusal, "measured within the bound over
  n_ind of 12 independent checks, and its leave-one-out is X× the bound". It is shown when the
  conjunction's LOO half decides.

**Saved models and collections: re-judged at load, no grandfathering (my call).**

- The sidecar's `class_memory` header is still read as the hysteresis state.
- At install the verdicts are recomputed under the new rule, as they are at every install today
  (`refresh_class_loo`).
- A key trusted under the old rule with fewer than 12 generations and LOO max > 1 loads untrusted.
  762159b's many-body class is one such key.
- *Why not grandfather.* The saved verdict was taken under a rule now known to admit answers 4.5×
  the bound. Re-earning trust costs teacher steps whose physics is correct. No format change is
  needed: `model_gen` is in every saved point, and an install moves the provider's generation past
  the installed history's.

## 4. (b) The answer-population inflation λ_A

**What λ_A measures (my call).** The calibration of the population rule 6 actually lets through.

- It is the classical variance-recalibration factor E[z²] (Levi et al. 2022 and Busk et al. 2022, as
  cited in `calibrate.rs` and not verified here). The method stands on the measurement.
- It is a **mean of ratios**, so a confident point counts as much as an unsure one. That is the case
  the ratio of sums under-weights.
- λ_sum is the existing λ, unchanged. Rules 6 and 7 apply

  ```text
  λ(j) = max(λ_sum(k_j), λ_A(k_j))        k_j = atom j's environment key; a sparse key keeps today's class floor
  rule 6:  λ(j) · σ_epi,j/√3 ≤ b_j          rule 7:  max_j λ(j) · σ_E,epi/N ≤ tier      (b_j and the tier unchanged)
  ```

  Since λ ≥ λ_sum ≥ 1, the inflation **never shrinks** the posterior's std. b_j and the tier never
  move.

**The sample P_A, chosen without reading σ or λ: the self-reference guard (my call).** A point is in
P_A only if its membership does not depend on the std it will calibrate.

- **(i) Class-refused in-envelope steps: history codes 5, 6 and 21.** Rules 2–5e and the fences
  passed, and rule 4b refused.
  - The class verdict is decided by measured errors and LOO, never by this step's σ.
  - Every such step is recorded, as 5/6 when rules 6 and 7 pass and as 21 when they refuse. So the
    union is complete whatever λ is.
  - This is the population an untrusted key will answer once it enters. In the gated test, with
    audits off, it is the only source.
- **(ii) Calibration ticks.** These sample a *trusted* key without reading its std.
  - Every in-envelope step advances a counter, and every `audit_every`-th one is a tick.
  - A tick counts only if the auditor is free at that step, **whatever answers the step**.
  - A tick on a fallback costs no teacher solve: the teacher's answer and the model's prediction with
    its std are both at hand. On an answered step it is one extra audit, which is also judged as a
    rule-8 audit and so only adds checking.
  - Rule 8's own cadence is unchanged (nothing loosened). The extra cost is at most one auditor solve
    per `audit_every` in-envelope steps.
  - The history gains a per-point `calibration` flag, written as a new optional key that reads false
    when absent. Older builds ignore it.
- **Excluded:**
  - ordinary audits, because they are selected by λ (only answered steps are audited);
  - codes 14 and 15 (rules 6/7 alone), selected by λ;
  - a new **code 24**, "refused by λ_A alone (it passes at λ_sum)", recorded for the Log and the
    measurement only. It is selected by λ_A itself.
- **No lockout follows from the exclusions.**
  - While λ_A refuses a key's steps, those steps are fallbacks, and the ticks keep sampling them
    without cost.
  - A refused key's class-refused steps are all in P_A.
  - So λ_A is always re-measured on steps it did not choose.
  - With audits off *and* the key trusted there are no ticks, and λ_A stays at its value at entry.
    That is a known blind spot, the same one today's audit-free runs have.
- **Audits are never judged by λ.** Rule 8 compares the error with b_j and the tier only, so no audit
  passes or fails by a bound it calibrates.
- **λ_A moves only at refits, installs and policy changes**, with λ_sum, never on a step it decided.
- **A residual dependence (open question 7).** The auditor's busy state depends on earlier ticks.

**The estimator (my call).** Per key k, over its P_A points inside the same window of generations
(≥ `WINDOW_POINTS`), and with the class's aleatoric level ā_c as today:

```text
z̃_p² = mean over the point's atoms j of key k of  max(0, e_j² − ā_c²) / max(s_j², ā_c²)     e_j = ‖ΔF_j‖/√3, s_j = σ_epi,j/√3
λ_A² = mean over the window's generations g of ( mean over g's points of z̃_p² )
λ_A  = max(1, λ_A) · (1 + 1/√(2 n_gen))                       applied from 3 P_A points on; 1 below; ∞ if a comparison is not a number
```

- **The numerator subtracts the misfit floor** (M8a Task 5 review I2). An error rule 6 cannot act on
  is not overconfidence.
- **The denominator is floored at ā_c.** The model cannot claim confidence below its own misfit, and
  a σ → 0 atom cannot make z infinite.
  - At the failing answers the floor does not bind: s ≈ 0.13 of the bound against ā ≈ 0.02 (σ_F
    4.5e-5 Ha/Bohr; bound ≥ 2e-3).
- **One draw per generation.** A refusal repeated over a parked or heating stretch (3313
  near-identical pair points in the baseline) counts once, and a single point is averaged within its
  generation.
- **No cap (as for λ_sum).** A large λ_A only sends steps to the teacher, and P_A keeps re-measuring
  it.
- **Applied from 3 points (my call).** λ_A can only widen, so applying it on little evidence errs on
  the safe side. Waiting for 12 would leave a key that opens on the conjunction path (§3) at λ_A = 1,
  which is the gap. The price is over-refusal from noise, and the hot run's two-sided check can fail
  that way (§6).
- **The energy uses the same scale** (rule 7 already takes the largest λ among the atoms: one scale of
  the posterior covariance). Open question 6.
- **λ_A does not enter the verdict's values.** V, W and R are measured errors. It changes the
  verdict's *jurisdiction*, as λ_sum does today: a class-refused step that the inflated rule 6 would
  also refuse is coded 21, not 5/6. The words of a refusal decided by λ_A say "(the model's std ×
  λ_A, its calibration measured where it would answer)". κ stays a readout.

## 5. Interaction with existing pieces

- **Hysteresis.**
  - At or above the floor the band is unchanged.
  - Below it the LOO half has no band, as today. When aging drops a trusted key below the floor, it
    can leave through the LOO half.
  - λ_A has no hysteresis: it is the posterior's and moves with it.
- **Re-judging on new evidence** (`rejudge_classes`).
  - n_ind is recomputed with the verdict.
  - Leaving is immediate. Entering between refits still needs a full window (`ClassKey::Window`).
  - λ_A does not move between refits.
- **The misfit fence and M.** Steps the fences refuse are not in-envelope, so they are not in P_A.
  M stays in the LOO RMS sums that the conjunction's LOO half reads, as today.
- **Teach-around acceptance** (`teach_verdict`).
  - Acceptance already requires LOO max and rms ≤ 1 and the class trusted, so the conjunction's LOO
    half is implied.
  - Teaching checks that are history audits are not ticks, so they are not in P_A.
  - `teaching_around_a_blocker_restores_its_class` should not move. If it does, it is a finding.
- **Shared collections and the migration engine (Task 14).** Collections re-judge at install as in
  §3. A re-solved sample is a new sample, not a point, so it does not count toward n_ind.
- **Stray atoms (branch `stray-atoms`, spec review I4).** The two designs are compatible:
  - Rule 7 and the audits there divide by n_bonded. λ multiplies σ_E,epi before that division, so the
    two commute.
  - λ_A is computed from per-atom forces, so n does not enter it.
  - Stray atoms are class 0, so they carry no key and add to no P_A or n_ind. That branch's audit
    attribution (I6) must keep a stray atom's comparisons out of the keys, which it already intends.
- **Draft mode.** Its held-out `DraftAudit` points are not ticks, so they are not in P_A. Draft mode
  still answers anyway.

## 6. Validation

**Gates (rule 6).**

- **`the_stuck_escape_repro_is_answered_honestly` must pass with every constant as it is:**
  `REPRO_FORCE_RATIO` 3, the tier 1e-3, `REPRO_MIN_JUDGED_HOT` 10, `REPRO_BOOKING_RESIDUAL` 1e-9.
- **The hot check is two-sided.** It needs ≥ 10 judged answers, *or* ≥ 10 class-refused steps whose
  would-be answers have rms > `ENTER`. Over-tightening can therefore fail it as well:
  - 07939ac's refused steps sat at rms 0.46, which is not "warranted";
  - so a policy that answers fewer than 10 times fails.
- **If the gate fails**, the failure is reported with the numbers below. It is not answered by any
  tolerance change. The bisect's option 2 (path-robust replicas) remains the user's call.
- **Suites to run:**
  - `cargo test -p providers --release`;
  - `--test surrogate` and `--test surrogate_water` in `validate`;
  - `cargo test -p app --release` (the words).

**Pinned tests expected to move, and why.**

- The rule changes, not a tolerance. Each moved value gets the new rule's derivation in its doc.
- In `providers/src/surrogate/tests.rs`:
  - `a_class_refused_for_one_outlier_earns_the_calibrated_verdict_and_holds`: entered after 10
    fallback steps with LOO max 1.08. It now needs 12 generations or LOO ≤ 1.
  - `a_class_refused_on_its_evidence_reenters_after_a_teaching_refit`;
  - `evidence_ages_by_model_generation_both_ways`.
- In the module docs:
  - `h2_plus_h_falls_back_on_the_grade_until_enough_is_learned`: 26/40 and 40/40 answers may drop
    under λ_A;
  - the numbers under "M8a calibrated verdict" in `references.md`.
- In `validate` `surrogate`:
  - `active_learning_run_converges` (its DFT-call count);
  - `surrogate_knows_what_it_does_not_know`.
- The counting test near `tests.rs:9070` (evidence = 3 for any n_f with no audits) should **not**
  move, because the cap stays at 3.

**New unit tests (names; the assertions in words).**

1. `a_class_below_the_floor_needs_its_leave_one_out_as_well`: eleven generations, V ≤ `ENTER`, LOO max
   3.07 → untrusted (`ClassKey::Floor`). The same at twelve → trusted. LOO max 0.9 at eleven →
   trusted.
2. `the_floor_counts_generations_not_points`: 50 rule-4b points in one generation → n_ind 1, and a
   class with LOO max > 1 is refused.
3. `species_less_points_do_not_open_a_key`: legacy points raise W and trip the guard but add nothing
   to n_ind.
4. `a_class_saved_open_on_three_points_loads_closed`: a sidecar with `class_memory` trusted, 3
   generations and LOO max 3.07 → untrusted after install. One with 12 generations inside the band →
   still trusted.
5. `the_answer_inflation_sees_confident_overconfidence`: ā = 0 and one point per generation. Twelve
   class-refused points at s = e = 0.9 of the bound (z̃ = 1), with twelve more at s = 0.1 and e = 1.0
   (z̃ = 10). Result: λ_sum = √(21.72/9.84)·1.144 ≈ 1.70 and λ_A = √50.5·1.144 ≈ 8.1. Rule 6 refuses
   an atom at s = 0.2 (λ_A s = 1.6), which λ_sum admits (0.34).
6. `the_answer_inflation_never_reads_what_it_selected`: adding any number of code 14, 15 or 24 points,
   or ordinary non-tick audits, leaves λ_A bit-identical.
7. `the_inflation_never_shrinks_the_std`: for random histories, λ ≥ λ_sum ≥ 1. Exactly λ_sum when
   P_A is calibrated (z̃ ≤ 1).
8. `a_calibration_tick_does_not_read_the_std`: two providers differing only in σ (scaled ×10) tick at
   the same steps.
9. `a_key_refused_by_its_answer_inflation_is_re_measured`: λ_A large on early points. Later
   class-refused points at z̃ ≈ 1 over enough generations bring it under 1.5 at a refit.
10. `a_non_finite_comparison_makes_the_answer_inflation_infinite`: no NaN reaches rule 6 as a pass.

**Measurements to report (no gates).**

- **The gated repro, per run:**
  - per key: n_ind, λ_sum and λ_A at the end;
  - code-24 refusals;
  - the judged answers' z, raw and inflated (‖ΔF‖/(λ·σ_epi));
  - worst ratios;
  - teacher solves and wall time against the records (07939ac 1036 s, 762159b 1378 s).
- **`trust_baseline_report`** (audits every 50): z rms per class and missed large errors, before and
  after.
- **The user's BEST collection.** No fixture exists in `validate`. An `#[ignore]`d CPU-only report
  would open `models/collections/lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5` read-only,
  never writing it (Task 13's read-only open). It would print per key: n_ind, λ_sum, λ_A, and the
  verdict under the old and new rules. This shows what the user's own keys lose at load before they
  meet it.

## 7. Approaches considered, what is left, open questions

- **A. Raise `MIN_EVIDENCE` 3 → 12 everywhere. Rejected.**
  - It raises λ's threshold, which loosens: λ = 1 at 3 to 11 points.
  - It moves the fallback weight cap, which shifts W.
  - It touches the unrelated segment-energy count.
  - It sends classes with 3 to 11 points to the max rule alone, dropping today's measured refusal.
- **B. Widen the existing λ_sum** (more points, a larger scatter factor). **Rejected.** A ratio of
  sums cannot see confident-point overconfidence, whatever its factor: the failing answers had z 10–34
  at s ≈ 0.13 of the bound.
- **C. Inverse-probability-weighted audits** (Horvitz–Thompson: each audit weighted by the answered
  steps it stands for). Unbiased, but one audit weighs about 2000 steps, so λ_A would be very noisy.
  The σ-blind ticks give the same freedom from selection at a known, small cost.
- **Chosen: the conjunction floor with λ_A on a σ-blind sample.** It is the smallest change that
  implements both halves of the decision and refuses at least as often as today everywhere.

**Left to the implementer.**

- Where the in-envelope counter lives: the variance cadence, the async auditor, and the CPU in-step
  path, where the auditor is always free.
- The history key name for the `calibration` flag.
- The code-24 name in `history_rule_name` and the Log.
- The panel's readout of n_ind, λ_A and which λ decided.
- Confirming that codes 5/6/21 are only written when rules 2–5e and the fences passed.
- The README line ("trust now needs 12 independent checks") at the merge.

**Open questions for the physics reviewer.**

1. **Is (a) read right?** *Resolved by the user 2026-11-18: keep the LOO opening, gated (this design's reading).* The literal reading would open *no* class on fewer than 12 checks,
   including the LOO-only opening after a pass. That costs the default H₂ pass's pair class 12
   generations of fallback. Counted in generations, a thinned, vibrating H₂ may never get there;
   counted in points it is about 12 correlated steps. This design keeps the zero-evidence LOO path,
   whose exposure is the measured "32 answers, worst 2.7×" from before Task 2. The user may need to
   rule on it.
2. **Is 12 enough for a heavy-tailed error?** The 2σ band argument assumes χ²₁ draws. Please measure
   the kurtosis of w̄_p on the repro's many-body points and say whether the floor should be corrected
   for it.
3. **Is a generation a valid lower bound on the effective sample size?** Consider points of different
   generations at one geometry, and the harness's 25-step refits against the app's refit per kept
   sample.
4. **The z̃ form.** Is flooring the denominator at ā_c right, as opposed to σ_F or a fraction of b_j?
   Is pooling per generation right?
5. **λ_A from 3 points.** Is the over-refusal it may cause acceptable against the hot run's two-sided
   check?
6. **Is one scale enough for the energy?** The failures carry 2.38e-3 Ha/atom energies. The task 1
   review (M5) found z_E biased, so an energy-own λ_E would need its own estimator.
7. **The ticks' residual coupling** to σ through the auditor's busy state. Is it negligible, or
   should a tick on a fallback be served whatever the auditor is doing?
8. **This design cannot promise the gate passes.** Its remaining exposure is twofold: a key opened on
   the LOO path at zero evidence (λ_A = 1), and a key trusted with audits off (λ_A frozen at entry).
   If the measurement fails there, is the next step policy (open question 1) or the test's design
   (the bisect's option 2)?
