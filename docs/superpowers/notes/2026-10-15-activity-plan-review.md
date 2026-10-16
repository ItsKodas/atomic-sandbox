# Activity tab plan: physics review (Tasks 2, 3, 4 and the labels)

**Reviewer:** physics-reviewer (read-only), 2026-10-15
**Reviewed:** `docs/superpowers/plans/2026-10-15-activity-tab.md` (the plan) and `docs/superpowers/specs/2026-10-15-activity-tab-design.md` (the spec), at `6dd3904`. Code cited at `6dd3904` (`crates/providers/src/surrogate/*`, `crates/providers/src/sampling/mod.rs`, `crates/surrogate/src/{sidecar,model,fit}.rs`, `crates/app/src/worker/model.rs`). The Task 1 implementer was editing `surrogate/mod.rs` and `trust.rs` in the worktree during this review, so line numbers are from `git show HEAD:`.

## Verdict: ready after amendments

Task 2 cannot change a policy decision as written. Every decision site keeps its predicate, `admit` keeps its control flow, `keep_sample` keeps its side effects, and the `Refusal` texts are byte-identical by construction. All but one refusal path get a code (the exception is S11). Task 4's `would_answer` follows `evaluate`'s priority order exactly and touches no policy state. The parity axes (A4) and the epistemic band (A5) are the right quantities.

The plan needs changes in three places. First, several displayed checks are re-derived next to the policy instead of read from it; the audit check and the class line already disagree with the policy. Second, the lifetime counter's "unrecorded" caveat is lost at the first autosave. Third, two labels state numbers the data cannot support: the pair curve's rule-7 lines, and the log's step and dispatch-step stamps.

Counts: **MUST 6, SHOULD 18, NOTE 12.**

---

## MUST

### M1. The audit trust check has a unit mismatch, names the wrong failure, and says "passed" when no audit has run (Task 3, rule 7)

- **Where:** plan Task 3 Step 3, line 1294 (`value: |audit_failed|, limit: energy_std_per_atom`); Task 8 `trust_line`, lines 2869-2870; spec §5.3, line 246; mock-up line 95. The plan asks for this ruling at line 1318.
- **The policy:** the audit fails when `|gap|/N > energy_std_per_atom` **or** the worst atom's force residual exceeds its bound (`mod.rs:929-930` in the step, `audit.rs:458-465` in the background). `audit_failed` holds the **total** gap in Ha, `audit_force` holds `(d, b)` when the forces failed it, and `fit_store` clears both at every refit (`trust.rs:377-378`).
- **What the plan shows:**
  - A total gap in Ha is set against a per-atom limit in Ha/atom. For H2, a failing audit at gap 1.5e-3 Ha (0.75 mHa/atom) is not a failure on energy; it would read "value 1.5e-3, limit 1.0e-3".
  - A force-only failure prints "found the model off by 0.3 mHa", a number inside the tier, next to a cross.
  - `met: self.audit_failed.is_none()` is true when **no audit has ever run**, and again right after a refit cleared an earlier failure. The plain line then says "Last check against DFT — passed" in both cases. Neither is true.
- **Ruling:**
  1. Add a readout field `audit_atoms: usize`, set wherever `audit_failed` is set (`mod.rs:949`, `audit.rs:473`). It decides nothing.
  2. When `audit_force` is `Some((d, b))`: the check is `(value d, limit b)`, worded "found forces off by d, allowed b mHa/Bohr".
  3. Otherwise: the check is `(|gap|/N, energy_std_per_atom)`, worded "found the energy off by x mHa/atom, allowed 1 mHa/atom".
  4. Word the met line from `last_audit`, not from `audit_failed`:
     - "no failed check since the last refit — last check: passed / box-affected, not judged / failed before the refit / none yet (one every N answered steps)".
  5. The Task 8 fixture (line 2668) and the Task 3 test assertions change with this.
- **Verified:** derivation from `mod.rs:929-963`, `audit.rs:456-487`, `trust.rs:109-114`, and the limit case "fresh model, no audit yet".

### M2. `trust_checks` re-derives the trust predicates beside `untrusted_check`; derive one from the other (Task 3)

- **Where:** plan lines 1281-1296 and 1314. They duplicate `trust.rs:76-116`, and `residuals_fail` / `noise_fails` (`trust.rs:263-272`) are already a third copy.
- **The problem:** the plan writes each `met` as a hand-negated comparison (`>=` for `<`, `<=` for `>`), then relies on the test (lines 1223-1250) to catch drift. That test does not exercise equality or NaN. The two forms differ for NaN. For example, `residual_dof_e = NaN` makes `NaN < 30` false, so the policy trusts, while `NaN >= 30` is also false, so the checklist shows a cross and "this is why" on a trusted model. Any later edit to one copy also drifts silently.
- **Required:**
  1. Build the list once, with `met = !(the refusal predicate)`, using one named predicate per condition shared with `residuals_fail` and `noise_fails`.
  2. Define `untrusted_check` as the first unmet entry of that list, formatted per key with today's strings (byte-identical).
  3. "The first unmet check is the refusal" then holds by construction, and the test only pins the strings.
- **Verified:** read `trust.rs:76-116` against plan lines 1285-1294. The order and constants match today (Samples, ResidualDofE, ResidualDofF, EnergyResidual, ForceResidual, ForceNoise, Audit; the residual pair only `if let Some(t)`), so the only defects are structural.

### M3. The class line re-derives `class_untrusted`'s branch and mislabels energy-only failures (Task 8 on a Task 3 number)

- **Where:** plan lines 2874-2883 (`class_line`); spec §5.3, line 248.
- **The policy:** `class_untrusted` (`trust.rs:202-225`) names forces when `force_max > 1 || force_rms > 1`, and otherwise names the **LOO energy** ratio.
- **The defect:** `class_line`'s third branch always prints "leave-one-out force error {force_max}x the bound". For a class failing only on energy (for example the first H-He sample, LOO energy 1.1e3x the tier, mod.rs module doc lines 336-338), it shows "force error 0.4x the bound" next to a cross. The UI also repeats `samples < q_prime`.
- **Required:** Task 3 exports the class verdict in structured form. For example, add `ClassLoo::refusal: Option<(ClassKey {Samples, Force, Energy}, value, limit)>`, filled by the same code path as `class_untrusted`. The UI only words it.

### M4. The lifetime counter's "earlier solves unrecorded" caveat is lost at the first save (Task 2 Step 5, spec A2)

- **Where:** plan lines 1155-1161 (`TeacherSolves` has no flag) and line 1186 (`save_sidecar` always writes `Some`).
- **The defect:**
  1. A model loaded from a file without the key starts at `unrecorded_before: true` with zeros.
  2. The session autosave (every backend switch round-trips through the file, spec A2) writes the key with the since-load counts and no flag.
  3. The next load reads `unrecorded_before: false`.
  4. The counter (plan line 2318) then shows an exact `total()` that can be far **below** `samples`, for example "teacher solves for this model 12" on a 412-sample model.
- **Required:** persist the flag, for example `"teacher_solves": {..., "complete": false}`. Alternatively, do not write the key while `unrecorded_before` is set. Pin the round trip load, save, load in `solve_counts_survive_the_sidecar_and_forget_resets_them`.
- **Verified:** limit case of a pre-activity file, followed by one autosave.

### M5. The pair curve's rule-7 lines and its refusal encoding (Task 9 on Task 4 data; spec §9.2-9.3). Ruling on the band question

- **Where:** plan line 2056 ("the dashed +-2·`energy_tier` lines labelled 'rule 7 refuses beyond this'", "shading under the x axis ... 4 px strip"); spec lines 147 and 285-286.
- **(a) The dashed lines.** Rule 7 compares the band's **half-width** sigma_epi/N with the tier. The dashed lines must therefore be the envelope E(r) +- N·tier drawn around the curve. Alternatively, plot sigma_epi(r)/N on its own lane. Horizontal lines at y = +-2 mHa in a plot of E - sum E_ref compare an energy with a width. For H2 the curve crosses that strip only where E - 2E_ref is about 0 (the repulsive wall and near r_cut), so the label "rule 7 refuses beyond this" would point at meaningless places.
- **(b) Visibility.** A trained H2 model has sigma_epi of order 1e-6 to 1e-5 Ha (module doc line 147: s about 1e-6 to 1e-5 for forces). A dimer curve spans at least 0.2 Ha. In a 150 px plot the band is 7.5e-3 px wide and the +-2 mHa envelope 1.5 px (computed). "Band invisible" then reads as "no uncertainty". Show the comparison on a log-y sigma lane under the curve:
  - sigma_epi/N with the tier line and the floor s_E;
  - the rule-6 ratio max_j sigma_F,epi/(sqrt(3)·bound_j) against 1, because rule 6 usually fires before rule 7.
- **(c) Ruling on the band.** +-sigma_epi is the correct band. It is the model's reducible uncertainty, the exact quantity rule 7 judges. The aleatoric floor N·s_E is drawn separately and labelled as the teacher's scatter. This approves A5.

  However, the band does **not** answer "will it fall back here". Rules 4, 4b, 5a-5e, 6 and the fence refuse independently of it, and rule 6 is usually the binding one. So the curve **must** mark where the policy would refuse, from `would_answer` (never from sigma), as the dominant encoding:
  - the curve solid only where `would_answer` is `Ok`;
  - dashed, with the band dimmed, where it refuses;
  - the refusing rule labelled on the stretch.

  A 4 px strip under the axis beside a uniform band invites exactly the "thin band = answered" reading.
- **(d) y-range.** Take it from the answered points and the dots. The refused extrapolation (r toward 0.5 Bohr) must not set the scale.

### M6. The log's "step" and "dispatched at step d" stamps are not what they say (Task 5/6 labels)

- **Where:** plan lines 1831-1832 (`step_of = snap.step` at publish; `dispatch = step - audit_lag`); spec line 128 ("dispatched at step 1 600, judged at 1 640"); plan line 2330.
- **(a) Wrong units.** `AuditRecord::unaudited_steps` counts **answered** steps (`self.answered`, incremented only when the model answered or would have, `mod.rs:1172-1175`). Fallback steps between dispatch and verdict are not in it. So `driver_step - lag` is not the dispatch step whenever a fallback happened in between.
- **(b) Wrong moment.** Events are stamped with the step at **publish**, not at the solve. At speed n, one tick of fallbacks gives n rows with the same step.
- **Required:**
  - Stamp each event in `Dynamics` right after the `evaluate` that produced it. The driver knows its step there, and `Trainable::take_teach_events` lives in `sim`, so rule 5 holds.
  - Show the audit lag as what it is: "judged k answered steps after its dispatch". Drop the computed dispatch step.
  - Pass events keep `pass`.

---

## SHOULD

- **S1. Make `would_answer` read-only by type and pin it by behaviour (Task 4, plan lines 1474-1507, 1349-1397).**
  - Its only `&mut` use is `near_quarantined` -> `ensure_desc_cache` (a cache; `quarantine.rs:280-308`). Call `ensure_desc_cache()` once at the top of `training_views` and add a `&self` `quarantine_distances` that requires a filled cache. `would_answer` then becomes `&self`, and the compiler enforces "no store, quarantine, cadence, audit counter or streak change". No RNG is on this path (checked).
  - Add a pin: two identical providers run the same 200-step `evaluate` sequence, one building `training_views` between every step. Sources, energies, forces, `variance_checks()`, `samples()` and the event stream must be bit-identical.
  - In the agreement sweep, also compare the atom index. Add a quarantined (fence) case, an untrusted case and a draft case (see S3).
- **S2. Cost of the curves (Task 4, lines 1585-1596; Review Focus 5).**
  - Each curve point runs `predict(..., true)` twice: once for the band, and again inside `would_answer`. Pass the band's prediction into `would_answer`.
  - Evaluate refusals in priority order with early return: envelope, then novelty, then class, then the variance last. This gives the same verdict as `evaluate`'s "compute all, take first" (`mod.rs:1159-1171`) and skips the O(Q'^2) variance wherever the envelope already refuses.
  - The build runs on the worker thread between steps. At S = 4, a multi-second build every 2 s would visibly throttle the dynamics. Read the printout against that, not only against the < 1 % teacher ratio.
- **S3. Pair-curve captions (spec §9.4, plan line 2056).**
  - "In a larger molecule the pair's energy also depends on its neighbours" implies a per-pair energy exists in the molecule. It does not: the model is a sum of per-atom terms whose descriptors couple every neighbour, and even the dimer carries body-3 self-terms (module doc lines 68-70). Word it as: "the model's energy of an isolated A-B pair; not a pair potential: in a molecule the model has no per-pair energy". The shading and the band are also for the isolated pair; say so.
  - In draft mode the model answers everywhere rules 1-3 pass. The caption must then say "shading: where it would fall back outside draft mode".
  - `would_answer` also ignores rule 0, audits and the cadence's reuse of an accepting verdict ("with a fresh uncertainty check").
- **S4. Use the fit's own references (`model.ref_energy`), not `self.refs` (plan lines 1517, 1582).** The fit subtracts `reference_vector` at fit time (`trust.rs:335`). `self.refs` can differ when `refs_dirty` (a pass re-solved a legacy reference), which shifts the curve's asymptote off 0. `self.refs[&za]` can also panic.
- **S5. Force-parity thinning aliases (plan lines 1569-1572).** `step_by(stride)` over interleaved x, y, z components keeps a single Cartesian component when the stride is a multiple of 3. That happens at stride 3, i.e. 40 001-60 000 components (computed), roughly 6 700-10 000 dimer samples. Thin whole atom vectors, or use a stride coprime to 3.
- **S6. Burst-temperature histogram (spec line 148, plan line 2057).**
  - It is built from the 5 000-row log ring, so a long session silently drops early bursts while the caption says "this session". Keep a separate per-session histogram accumulator in `ActivityState`.
  - `record` runs after `csvr.apply` (`sampling/mod.rs:809-813`), so `burst_t` is the post-thermostat temperature. Pin which one is meant (post-Verlet v(t+dt) before CSVR is the Verlet state at the solved geometry) and document it.
  - Add to the caption that consecutive burst points are correlated (tau = 5 fs at dt = 0.25 fs). For N = 2 the instantaneous T has relative std sqrt(2/3), about 0.82 (f = 3N - 3 = 3, `Scene::dof`), as the spec already notes.
- **S7. 5e at gamma = infinity is labelled "5d" (plan table line 990 vs 992; `rule_label` line 2766).** Both map to `NewEnvironment`, but the module doc (lines 89-93) says they are different tests. Add a code such as `UnspannedAtom`, labelled "5e new environment"; the plain sentence can stay the same.
- **S8. The `ForceNoise` plain line (plan line 2868; spec line 245) misdescribes sigma_f.** sigma_f is "label noise and basis misfit together" (module doc lines 276-280). "Its force labels agree with each other" blames the teacher for what may be model misfit. Use: "Its forces fit the data to within the force tier on average — scatter sigma_f x, allowed 2 mHa/Bohr".
- **S9. "Enough samples — n of Q'" uses the fit's `n_samples`, while the counter shows the store length.** Two different "samples" numbers sit side by side. Word the check as "samples in the fit". On the counter's hover say "held; F in the current fit" (the Model tab's "fit N of S held").
- **S10. `stuck_info` invents a rule (plan line 1303).** `map_or(TeachRule::NearQuarantine, ...)` names the fence when no reason is stored. Make it `Option<TeachRule>`.
- **S11. The background auditor's failed and discarded solves leave no event (`audit.rs:417-418`, `443-446`; plan line 1118).** The counter's hover says "kept or not", but these auditor solves are not counted, while a failed main-teacher solve is (plan lines 1074-1085). Either push `Failed` and discarded events with origin Audit, or word the counter as "completed, judged solves". Cancelled solves are excluded too; say so.
- **S12. The event's `detail` is only `notes[0]` (plan line 1093).** The admission note with its numbers is lost from the hover, for example "0.42 training spreads from a stored one (<= 0.75...)" or "face density 3.1e-5 > 2.0e-5". Carry every provider note of the step.
- **S13. Sentences that can be false (plan lines 2796, 2803-2806, 2841-2845).**
  - Rules 6/7 say "— this answer teaches it" even when the outcome is `BoxNotKept` or `FenceDisagrees`. Condition the phrase on `outcome.is_kept()`. The box clause appended to that sentence is also ungrammatical ("...teaches it and is touching the box edge").
  - `NewSpecies`: "it learns it at the next refit" overpromises. The refit needs El's reference (else `MissingReference` keeps the old model), and the enlarged Q' untrusts the model until El samples accumulate.
  - `NotOffered`: "not used" is false, because the step uses the answer. Say "not kept for this model".
  - Write "one in ten" and "three fresh answers" from `THIN_EVERY` and `FENCE_VOTES`, not as literals.
- **S14. Old rows of rule 4 take today's numbers.** `rule_sentence(R::Untrusted(_), ..., untrusted)` (plan line 2798) is fed the current first unmet line, so an hour-old row shows current values. Use key-only wording for log rows, or carry the `TrustCheck` in the event.
- **S15. `box_blocks_trust` keys on string prefixes (`worker/model.rs:120-132`).** Use the first unmet `TrustKey` once Task 3 exposes it. Put the box first only when that key is a fit condition (EnergyResidual, ForceResidual, ForceNoise), not for a failed audit: an audit is judged against a fresh, non-box teacher answer, so "the box is why" is wrong there.
- **S16. "Solves this session" on the DFT backends never ticks for plain DFT steps,** because it counts teaching solves only (spec §3.3, line 70). Label it "teaching solves this session".
- **S17. Sidecar reader (plan lines 1174-1183).** A malformed optional `teacher_solves` makes `read` fail for the whole file through `?`. An optional key should read as `None`, with a note, and never refuse a model. Use `Json::u64()` (sidecar.rs:891) rather than `usize()? as u64`. Older readers ignore the key (look-up by name, sidecar.rs:853-858): confirmed back-compatible, and nothing gates on it (`install_sidecar`, `Dynamics::LoadModel`).
- **S18. One failed `predict` drops every view (plan line 1586, `.ok()?`).** Skip that curve instead, and report how many samples the parity plot omitted (line 1542 `continue`s silently).

## NOTE

- **N1.** An essential (rule 1, 2 or 4-short) box-affected answer near a fence takes the vote path (`quarantine.rs:443, 450-485`), so `KeptByVote` and `KeptReleased` can be box-affected. The event's `box_affected` flag covers this; the outcome words do not.
- **N2.** A relax verification whose gate refuses (rules 1-4) is logged as origin Fallback with the gate's rule (`mod.rs:1086, 1095-1101`), so the "check" origin and `verification` count miss it. Defensible; worth a line in the doc.
- **N3.** Rule-0 events are pushed on every out-of-cell `evaluate`. A drag held against the margin can fill the queue with NotSolved rows. It is bounded, and they are not counted.
- **N4.** `burst_t` (K) and `solve_ms` (ms) in `sim::activity` records follow existing precedent (`SamplingPlan`, `EvalStats`). This is not an atomic-units violation in a decision path; it is fine.
- **N5.** `d_box(h)` is the hydrogen 1s-tail estimate (checked: 5.35 Bohr at h = 0.15, 4.84 at h = 0.25). It applies to every element and mixes the unconfined tail with the reported (pinched) face density. This is the existing `box_advice` caveat; it is stated, and it is not per-element data.
- **N6.** "b of n touch them" divides box-affected samples **in the fit** by **held** samples. This is the existing M6 text; the wording could say so.
- **N7.** `mha()` (plan lines 2854-2858) gives one significant figure below 0.1 mHa: sigma_f 4.5e-5 Ha/Bohr shows as "0.04" or "0.05". Format with two significant figures.
- **N8.** The Pair class means "every atom has at most one neighbour within r_cut" (`env_class`). Two separated H2 molecules are "pair" class, so "two-atom arrangements" is approximate.
- **N9.** A fit point's `trusted` flag is frozen at the refit. A later failed audit does not hollow it; the caption could say "trusted at the refit".
- **N10.** In-step audits on the CPU teacher show the headline "DFT fallback".
- **N11.** `install_sidecar` must set `ledger.counts` only on its success path (after the validation returns).
- **N12.** `training_views` histograms count fitted samples only (`st == Fitted`). The coverage caption should say "in the current fit".

## Checks on the plan's specific questions

- **Task 2 cannot change a decision.** Verified site by site:
  - `gate` (`policy.rs:177-210`) and `envelope_outside` (`policy.rs:216-293`);
  - `variance_check` (`policy.rs:307-348`) and the novelty in `evaluate` (`mod.rs:1132-1139`);
  - `class_untrusted` (`trust.rs:202-225`) and the fence (`mod.rs:1165-1171`);
  - the audit and `forced` overrides (`mod.rs:1172-1195`), `admit` (`quarantine.rs:407-517`) and `keep_sample` (`store.rs:267-274`).

  Each gains a wrapper or a return value. No predicate, constant or order changes. The thinning counter `since_kept` and the fence state are mutated exactly as before. The rule-0 change keeps the same `Err`.
- **Every refusal path has a code:** rule 0; 1-4; 4b (four branches); 5a, 5b, 5c, 5e-void, 5e-infinity, 5e; 5d; 6; 7; the fence; the audit in the step and in the background; relax verification. The admit outcomes cover `!keep`, the no-model cap, the box, the vote kept / disagree / agree, thinning, kept, and the store error. The pass covers references, samples, skips, the burst tail and burst failures. The one missing path is the auditor's failure (S11).
- **Is each code the rule that fired?** Yes. The codes ride on the same `Option` chain `envelope.or(novel).or(class).or(variance)`, then the fence (`mod.rs:1159-1171`), then the draft, audit and forced overrides. `would_answer` (plan lines 1475-1506) mirrors that order exactly.
- **`KeptBoxAffected` matches the essential-box exemption.** The final `true` of `admit` is reachable with `face > rule` only when `essential` (`quarantine.rs:443`). The pass uses the same `box_affected_face_density(scene.cell.h)` as `is_box`.
- **Parity axes (A4): approved.** (E - sum E_ref)/N is the fit's target with `energy_per_atom` (fit.rs:18, 1203). y - x is the training-residual quantity `|E_model - E|/N` (`trust.rs:131`), so the +-tier band about y = x is exactly that trust condition.
- **Band (A5): approved, subject to M5.** The floor is N·s_E for `energy_per_atom` (otherwise s_E), as in `variance_check` (`policy.rs:333-334`). The plan's `floor2` matches both cases.
- **Per-solve cost.** No view, check or event calls a teacher. One `TeachEvent` (one `String`, about 200 B) per teacher solve of 100 ms or more. Nothing on a model-answered step. Per publish: draining an empty `VecDeque` (no allocation) and a 7-entry `Vec` in `status()`. One relaxed `fetch_add` per SCF step. The only heavy item is the S = 4 curve build (S2).
- **Sidecar key.** It is back-compatible (unknown keys ignored), byte-identical when absent (pinned), and never keyed on. M4 and S17 still apply.

## Not checked

- Task 1 and Tasks 5-9 beyond their labels and the data they draw (Opus scope). This includes probe accuracy across `retry_cold_on_stall`, where a stall retry keeps counting SCF steps across both attempts.
- I did not run any test or build. Every finding comes from reading the code at `6dd3904`, or from the derivations and arithmetic stated above.
- The S = 4 view build time: it is not measured yet; the plan prints it in Task 4.
