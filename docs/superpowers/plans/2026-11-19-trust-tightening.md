# Trust Tightening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the user's "Tighten trust" decision (2026-11-17, approved spec rev. 2 on
2026-11-19). It has two halves:

- **The evidence floor.** A class verdict can no longer open on fewer than 12 fits and 60 points
  unless the leave-one-out rule also passes.
- **Two new inflations, λ_T for forces and λ_E for energy.** Rules 6 and 7 apply them, measured on a
  sample chosen without reading σ.

Together they close the gap that `the_stuck_escape_repro_is_answered_honestly` catches, with its
tolerances unchanged.

**Architecture:**

- **History (`surrogate::history`).** Each point gains three per-point fields: a fit id, a flags word
  (tick / fenced / cadence-fresh), and the epistemic energy std.
- **The calibration (`providers::surrogate::calibrate`).** It gains:
  - the floor counts and the conjunction verdict;
  - the pure functions λ_T (bound units) and λ_E (tier units) over the answer sample P_A.
- **The provider.** It records the new fields in `evaluate` and keeps a never-reset cadence counter.
  It runs calibration ticks every 25 in-envelope steps, asynchronously on the auditor, skipped and
  counted when the auditor is busy. Rules 6/7 use λ_F = max(λ_sum, λ_T) and λ_E.
- **Task 0 comes before all of it:** a read-only report on the user's BEST collection, which is a
  go/no-go.

**Tech Stack:** Rust workspace crates `surrogate`, `providers`, `sim`, `app`, `validate`; f64 CPU
throughout (no GPU code changes).

**Spec:** `docs/superpowers/specs/2026-11-18-trust-tightening-design.md`, rev. 2, approved at e6d79a80
("the spec", cited §n). Its reviews:

- `.superpowers/spec-review/trust-tightening-physics-review-1.md`;
- the re-review ruling, folded into the spec as rev. 2.

Project rules: `CLAUDE.md`. Mechanism: the bisect file named in spec §2.

## Global Constraints

**The floor (spec §3).**

- Constants: `pub const FLOOR_FITS: usize = 12;` and `pub const FLOOR_POINTS: usize = 60;`.
- Both are counted over a key's attributed verdict points (audits and codes 5/6) inside its evidence
  window. A point counts toward the fits only if it carries a fit id, and toward either count only if
  it carries species.
- **Floor met:** today's calibrated verdict, unchanged (`ENTER` 1/√2 to enter, `LEAVE` 1 to stay, the
  worst-atom guard on entry).
- **Evidence ≥ 3 with the floor not met:** trusted only if **both** pass:
  - `trust::class_refusal` (LOO force rms and max ≤ 1, LOO energy max ≤ 1);
  - today's calibrated verdict.

  The worst-atom guard then applies both ways.
- **Evidence < 3:** today's rule.
- **These uses of `MIN_EVIDENCE` stay at 3:** the fallback weight cap, λ_sum's threshold, the
  segment-energy count, and `key_inflation`'s class floor.
- **A fit id** is bumped only by a refit that solved new coefficients. Never by `set_policy`, restore,
  forget, install, attach, or a worker re-spawn.
- **Saved models and collections are re-judged at load, with no grandfathering.**

**λ_T (spec §4, forces, bound units).**

- r_j = s_j/b_j, where s_j = σ_epi,j/√3 and b_j = min(max(2e-3, 10 %|F_j|), 1e-2); e_j = ‖ΔF_j‖/√3.
- A(λ) = {P_A atoms of the key with λ·r_j ≤ 1}.
- A set A is good when both hold:
  - max over A of e_j/b_j ≤ `WORST_ATOM` = **3**, at every size;
  - when A meets the floor (12 fits, 60 points): rms over A's points of the mean (e_j/b_j)² of each
    point's admitted atoms, × (1 + 1/√(2n_A)), is ≤ `ENTER`.
- λ_T = the smallest λ ≥ 1 with A(λ′) good for every λ′ ≥ λ. If a bad atom has r = 0, λ_T = ∞.

**λ_E (spec §4, energy, tier units).**

- r_p = σ_E,epi,p/N_p/tier; e_p = |ΔE_p|/N_p/tier. Each point counts for every key it holds.
- **Max condition: e_p ≤ 1, i.e. 1× the tier** (not 4.53). The rms condition is as for λ_T.
- The aleatoric floor is not subtracted.
- A point without `energy_std_epistemic` is not in λ_E's sample.

**Rules 6 and 7.**

- λ_F(j) = max(λ_sum(k), λ_T(k)).
- Rule 6: λ_F(j)·s_j ≤ b_j.
- Rule 7: max(max_j λ_F(j), max_k λ_E(k))·σ_E,epi/N ≤ tier.
- b_j and the tier never move, and λ ≥ λ_sum ≥ 1.
- A sparse key (fewer than 3 of its own P_A points) takes its class's λ_T and λ_E as floors.
- The inflations move only at refits, installs and policy changes.

**P_A** is the set of points with flags *not fenced* and *cadence-fresh*, and either codes 5, 6 or
21, or a calibration tick. Ordinary audits, codes 14, 15 and 24, fenced points, and points fresh only
by a refusal are excluded. New code **24** means "refused by λ_T or λ_E alone (it passes at λ_sum)".

**Ticks.**

- Every 25th in-envelope step, per provider, independent of `audit_every`.
- **In the app:** served asynchronously on the auditor only, never in-step. When the auditor is busy,
  the tick is **skipped and counted**, whatever answers the step, and never deferred.
- In-step ticks happen only on paths with no auditor (the CPU in-step path, the harness).
- A tick at a fallback needs no solve. A tick at an answered step is also judged by rule 8.
- Rule 8's own cadence is unchanged.

**Cadence-fresh.** A separate counter that nothing resets: a step is cadence-fresh when that counter
is a multiple of `variance_cadence`. `consults` is not used.

**NaN and zero (M1).**

- e = s = 0 adds nothing.
- s = 0 with e/b > 3 gives λ_T = ∞, and rule 6 refuses the key's atoms, s = 0 included (∞·0 counts as
  ∞).
- Any NaN gives NaN (as trust.rs:105 does, never through `f64::max`), and a NaN or ∞ factor refuses.

**Never:**

- no tolerance, tier, bound or gate constant changes (rule 6): `REPRO_FORCE_RATIO` 3, tier 1e-3,
  `REPRO_MIN_JUDGED_HOT` 10, `REPRO_BOOKING_RESIDUAL` 1e-9;
- no per-element parameter (rule 1): Z is a key only.

**Formats.**

- History arrays are new and optional, read as absent.
- If a reader rejects unknown trailing arrays (the standalone encoding does: "trailing bytes … are an
  error"), its magic bumps symbolically: `ASAUDIT<n+1>`, and the sidecar `ASMODEL<K+1>` if its reader
  rejects them too. The value is read in Task 1 step 1, never assumed. It is registered in
  `app::update::formats`.

**Agents and machines.**

- Every subagent runs on Opus (memory "No Fable"). Tags give the CLAUDE.md category: **Fable** is
  physics-critical and gets a `physics-reviewer` pass on its diff; **Opus** is engineering.
- One heavy GPU binary at a time. Never kill the user's cargo or `app.exe`. Build at BelowNormal with
  `-j 4` when another job runs.
- Heavy tests stay `#[cfg_attr(debug_assertions, ignore)]`.
- Never write `D:\Projects\Physics\models` (Task 0 reads a scratchpad copy).
- Commit with `git commit -o -- <paths>`. Every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

These are five failure modes that would bite a user and that the tasks' own tests would not exercise.
Each has a test added to its owning task.

1. **A downgrade erases the evidence.** An older build reads a new `history.bin` as unreadable, starts
   empty, and overwrites it on save.
   - Test `a_newer_history_is_never_overwritten_by_this_reader` (Task 1): a file with the next magic
     is refused, the collection opens with the existing "newer file" behaviour, and `save_history`
     leaves its bytes unchanged.
   - Plus `app::update::formats` lists the bumped magic, which the downgrade warning reads.
2. **A loaded key is refused at a parked scene forever and nobody says why** (spec §3, the dead end).
   - Test `a_parked_key_below_the_floor_raises_the_teach_prompt_and_says_why` (Task 4): after
     `TEACH_PROMPT_STREAK` (5) class fallbacks at one geometry, with LOO max > 1 and the floor unmet,
     the teach prompt is offered. The note names the floor (`ClassKey::Floor`), n_fit/12, n_pts/60 and
     the way out.
3. **A trusted key is locked by λ_T = ∞ while the panel says "trusted".**
   - Test `a_key_locked_by_its_tail_inflation_says_so_and_keeps_measuring` (Task 6): after one tick
     at e/b = 4 with r = 0, every atom of the key is refused under code 24. The note names λ_T = ∞,
     the measured atom and "lifts as the window ages". Ticks and class-refused points keep being
     recorded meanwhile.
4. **Calibration starves at full speed** (the auditor is saturated, so every tick is skipped).
   - Test `skipped_ticks_are_counted_and_shown` (Task 7): with an auditor kept busy, 100 due ticks
     produce 100 skips, `ticks_skipped()` = 100, and none is served in-step.
   - The Model tab line "calibration checks skipped: 100" (Task 8).
5. **Reloads manufacture fits.** A save/load, a collection re-attach or a worker re-spawn advances the
   fit id, so the floor is reached by restarts.
   - Test `reloading_reattaching_and_respawning_add_no_fit` (Task 2): n_fit is identical before and
     after a sidecar save+load, a detach+attach, `fork_for_refit` with a carried model, `restore`, and
     12 `set_policy` calls.

---

### Task 0: The BEST-collection go/no-go report (read-only, CPU-only) — **Opus**

**Execution STOPS after this task until the user says go.**

**Files:**

- Modify `crates/providers/src/surrogate/trust.rs`: a readout only, no policy change.
- Modify `crates/validate/tests/surrogate.rs`: the new `#[ignore]`d test.
- Create `docs/superpowers/notes/2026-11-19-trust-tightening-go-no-go.md`: the report and the count.

**Interfaces:**

- Produces:
  - `pub fn key_class_stats(&self) -> Vec<(surrogate::EnvKey, providers::surrogate::ClassStats)>`
    on `SurrogateKs`: the per-key LOO statistics the verdict reads. Empty without a fit.
  - The test `trust_tightening_go_no_go_report`.
- Consumes:
  - `SurrogateKs::class_loo() -> &[ClassLoo]` (`SpeciesVerdict`: trusted, refusal, evidence);
  - `audit_history()`;
  - `surrogate::Collection::open(root, key)`;
  - `SurrogateKs::attach(c)`.
- `Collection::open_read_only` does not read `history.bin`, so the test opens a **scratchpad copy**.

- [ ] **Step 1: Write the readout's unit test** `key_class_stats_match_the_verdicts_inputs` in
  `crates/providers/src/surrogate/tests.rs`. On the Morse fixture, every key in `class_loo()` species
  appears once, and its `force_max`/`force_rms` equal the class pass's per-key sums.
  - Run `cargo test -p providers --release key_class_stats` → FAIL (no such method).
- [ ] **Step 2: Implement `key_class_stats`** from the per-key `LooSums` already computed in
  `compute_class_loo_full`, kept on `ClassJudge`. Run → PASS.
- [ ] **Step 3: Write the report test.**
  - **Setup:**
    - Copy `D:\Projects\Physics\models\collections\lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5`
      without `.lock` into `<scratchpad>/best-copy/<slug>/`.
    - Assert the source folder's file sizes and mtimes are unchanged at the end.
    - Build `SurrogateKs::new(Box::new(NoTeacher), FallbackPolicy::default())`. `NoTeacher::evaluate`
      returns `Err`, so no DFT runs.
    - Read the key from `key.txt`, then `Collection::open(copy_root, &key)`, `attach`, `refit`.
  - **Per key it prints:** today's `trusted` and `refusal`; `evidence`; the number of distinct
    `model_gen` (information only); the LOO force rms and max and the energy max; λ_sum (`inflation`);
    the new verdict.
  - **The new verdict, computed in the test:** if today trusted and evidence ≥ 3, then also require
    `force_rms ≤ 1 && force_max ≤ 1 && energy_max ≤ 1`. n_fit = 0 for every legacy point, so the floor
    is never met; λ_T = λ_E = 1, since legacy points carry no flags.
  - **Output:** the count of keys that are trusted today and untrusted under the new rule, plus the
    table written to the note.
  - Marked `#[ignore = "go/no-go measurement: cargo test -p validate --release --test surrogate trust_tightening_go_no_go_report -- --ignored --nocapture"]`,
    with no GPU context requested.
- [ ] **Step 4: Run it.** Expected: it completes and prints the table and the count. No assertion on
  the count; it is a measurement.
- [ ] **Step 5: Write the note.**
  - The table, the count, and per losing key its scene words (`key_words`) and today's evidence.
  - The caveat: the report shows only the floor's effect at load, since λ_T and λ_E cannot be
    previewed on legacy points.
  - Commit the note, the readout and the test.
- [ ] **Step 6: STOP.**
  - Send the user the count of keys that lose trust, and the note's path.
  - Do not start Task 1 until the user says go. If they say no-go, the spec goes back to them.

---

### Task 1: History fields: fit id, flags, epistemic energy std — **Opus**

**Files:**

- Modify:
  - `crates/surrogate/src/history.rs`: `AuditPoint`, `HistoryArrays`, encode/decode, the module doc
    "Standalone encoding";
  - `crates/surrogate/src/sidecar.rs`: the arrays under `audit_history`, the "Version" doc;
  - `crates/app/src/update/formats.rs`: only if a magic bumps.
- Test: `crates/surrogate/src/history.rs` tests and `crates/surrogate/src/sidecar.rs` tests.

**Interfaces (Produces):**

```rust
pub struct PointFlags(pub u8);   // TICK = 1, FENCED = 2, CADENCE_FRESH = 4; f64-integral in the array
impl PointFlags { pub const TICK: u8 = 1; pub const FENCED: u8 = 2; pub const CADENCE_FRESH: u8 = 4;
                  pub fn has(self, bit: u8) -> bool }
// AuditPoint gains:
pub fit_id: Option<u64>,                 // None = absent (legacy) — does not count toward the floor
pub flags: PointFlags,                   // 0 when absent — not in P_A
pub energy_std_epistemic: Option<f64>,   // Ha, total-energy epistemic std; None = not in λ_E's sample
// HistoryArrays gains three optional per-point arrays:
pub const FIT: &str = "audit_fit"; pub const FLAGS: &str = "audit_flags"; pub const ENERGY_STD_EPI: &str = "audit_energy_std_epistemic";
```

- [ ] **Step 1: Read both readers' handling of unknown trailing arrays.**
  - The standalone reader rejects them (`decode`: "trailing bytes after the last array").
  - Decide each bump: history magic `ASAUDIT<n+1>`; the sidecar `ASMODEL<K+1>` only if its reader
    rejects unknown arrays.
  - Write the values read into this plan's "Rulings" section.
- [ ] **Step 2: Write the failing tests.**
  - `the_new_fields_round_trip`: standalone and sidecar; Some, None and every flag bit.
  - `a_legacy_history_reads_with_absent_fields`: an `ASAUDIT1` fixture gives fit_id None, flags 0 and
    energy_std_epistemic None.
  - `hostile_new_arrays_are_rejected`: a wrong length, a non-integral fit or flag, fit ≥ 2⁵³, a flag
    above 7, a negative or NaN energy std. Each is an `Err`, never a panic.
  - `a_newer_history_is_never_overwritten_by_this_reader` (Review Focus 1).
  - Run `cargo test -p surrogate --release history` → FAIL.
- [ ] **Step 3: Implement.** Fields, arrays and encode/decode, with lengths checked before
  allocation. Bump the magics as ruled in step 1 and register them in `formats.rs`. Run → PASS. Also
  run `cargo test -p app --release update::`.
- [ ] **Step 4: Commit.**

---

### Task 2: The fit id — **Opus**

**Files:**

- Modify:
  - `crates/providers/src/surrogate/mod.rs`: the field `fit_id: u64`;
  - `crates/providers/src/surrogate/trust.rs`: `refit` near l. 1932, `restore` near l. 1803;
  - `crates/providers/src/surrogate/store.rs`: `install_sidecar`, forget near l. 124;
  - `crates/providers/src/surrogate/shared.rs`: attach near l. 311;
  - `crates/providers/src/surrogate/audit.rs`: `record_history_point` writes `Some(self.fit_id)`
    when a model exists;
  - `crates/surrogate/src/sidecar.rs`: header key `fit_id`.
- Test: `crates/providers/src/surrogate/tests.rs`.

**Interfaces:**

- Produces: `pub fn fit_id(&self) -> u64` on `SurrogateKs`.
- The id rules:
  - A refit that solved coefficients sets `fit_id = max(fit_id, max audit_fit in history) + 1`.
  - Restore brings back the restored fit's id.
  - `install_sidecar` takes the sidecar's `fit_id`, defaulting to the history's maximum when absent.
  - Attach sets the next refit's base from the collection history's maximum.
  - `set_policy`, forget and `fork_for_refit` leave it unchanged.

- [ ] **Step 1: Write the failing tests.**
  - `a_refit_advances_the_fit_id_and_nothing_else_does`.
  - `reloading_reattaching_and_respawning_add_no_fit` (Review Focus 5).
  - `fit_ids_stay_unique_across_an_install`: a history holding ids up to 40 → the next refit is 41.
  - Run `cargo test -p providers --release fit_id` → FAIL.
- [ ] **Step 2: Implement** as above. The background refit's result carries the id it was solved
  under. Run → PASS.
- [ ] **Step 3: Commit.**

---

### Task 3: Recording the flags: cadence-fresh counter, fenced bit, epistemic energy std — **Fable**

Physics-reviewed for σ-blindness.

**Files:**

- Modify:
  - `crates/providers/src/surrogate/mod.rs` (`evaluate`, l. ≈2380–2530): the never-reset
    `cadence_steps: u64` and `fenced`;
  - `crates/providers/src/surrogate/audit.rs`: `record_history_point` gains `flags: PointFlags` and
    `energy_std_epistemic: Option<f64>`;
  - every caller of `record_history_point` (grep).
- Test: `crates/providers/src/surrogate/tests.rs`.

**Interfaces:**

```rust
pub(super) fn record_history_point(&mut self, kind: PointKind, pos: &[[f64; 3]], z: &[u8], stds: Option<(f64, &[f64], &[f64])>, model_energy: f64, model_forces: &[[f64; 3]], energy: f64, forces: &[[f64; 3]], box_affected: bool, draft: bool, unfit: bool, flags: PointFlags, energy_std_epistemic: Option<f64>)
```

- **Cadence-fresh:** `cadence_steps` advances on every consulted step and is never reset. The flag is
  set when `cadence_steps % variance_cadence == 0`.
- **Fenced:** set when `near` is non-empty, `near_misfit` holds, or the audit fence would refuse. All
  three are evaluated even when the class reason is already set, without mutating any state.
- **Epistemic energy std:** √(energy_std² − (N·s_E)²), clamped at 0, with s_E the fit's.

- [ ] **Step 1: Write the failing tests.**
  - `the_cadence_flag_does_not_follow_refusals`: a run whose rule-6 refusals make extra fresh steps
    → the flagged steps are exactly every `variance_cadence`-th consult, identical to a run with σ
    scaled ×10.
  - `a_class_refused_step_near_a_quarantined_sample_is_flagged_fenced`: code 5/6 with
    `FENCED` set.
  - `the_recorded_epistemic_energy_std_matches_the_prediction`: equal to
    `√(pred.energy_std² − (N s_E)²)` within 1e-15 relative.
  - Run `cargo test -p providers --release cadence_flag` → FAIL.
- [ ] **Step 2: Implement.** Run the three tests → PASS. Run all `providers` tests: nothing else may
  move, since no verdict reads the flags yet.
- [ ] **Step 3: `physics-reviewer` pass on the diff** (σ-blindness of both bits), then commit.

---

### Task 4: The evidence floor — **Fable**

**Files:**

- Modify:
  - `crates/providers/src/surrogate/calibrate.rs`: `ClassCalibration`, `summarise`, `verdict`,
    `refusal_words`, the module doc;
  - `crates/sim/src/provider.rs`: `ClassKey::Floor`, plus its doc;
  - `crates/providers/src/surrogate/mod.rs`: the module doc, rule 4b;
  - `crates/app/src/ui/activity/words.rs`: the worst-atom and Floor words.
- Test: `crates/providers/src/surrogate/tests.rs`.

**Interfaces (Produces):**

```rust
pub const FLOOR_FITS: usize = 12;
pub const FLOOR_POINTS: usize = 60;
// ClassCalibration gains:
pub fits: usize,          // distinct fit ids among attributed verdict points in the window
pub floor_points: usize,  // attributed verdict points in the window
impl ClassCalibration { pub fn floor_met(&self) -> bool }   // fits ≥ 12 && floor_points ≥ 60
// sim::ClassKey::Floor — value = the LOO ratio that decided, limit 1
```

- **`verdict` order:** samples; the worst-atom guard (`!was_trusted || !floor_met()`); then
  - `!calibrated()`: the rule of before;
  - `calibrated() && !floor_met()`: the rule of before first, refused as `ClassKey::Floor`, then
    today's calibrated checks;
  - `floor_met()`: today's calibrated checks.
- **Words:** "within the bound over {fits} of 12 fits / {points} of 60 checks, and its leave-one-out
  is {x:.1}× the bound; {n} samples".

- [ ] **Step 1: Write the failing tests.**
  - `a_class_below_the_floor_needs_its_leave_one_out_as_well`: 11 fits/80 points and 12 fits/59
    points, V ≤ ENTER, LOO max 3.07 → Untrusted { key: Floor }. 12 fits/60 points → Trusted. LOO max
    0.9 below the floor → Trusted.
  - `points_without_a_fit_id_or_species_do_not_open_a_key`: they raise W and trip the guard; fits and
    floor_points are unchanged.
  - `the_worst_atom_guard_applies_both_ways_below_the_floor`.
  - `a_class_saved_open_on_three_points_loads_closed`: a sidecar with `class_memory` trusted, 3 fits
    and LOO max 3.07 → untrusted after `install_sidecar`. One with 12 fits/60 points inside the band →
    trusted.
  - `a_parked_key_below_the_floor_raises_the_teach_prompt_and_says_why` (Review Focus 2).
  - Run `cargo test -p providers --release floor` → FAIL.
- [ ] **Step 2: Implement.** Run → PASS.
- [ ] **Step 3: Update the pinned tests this rule moves.** Write the new rule's derivation in each
  doc; no tolerance changes.
  - `a_class_refused_for_one_outlier_earns_the_calibrated_verdict_and_holds` (tests.rs:8914);
  - `a_class_refused_on_its_evidence_reenters_after_a_teaching_refit` (9164);
  - `evidence_ages_by_model_generation_both_ways` (9096): the window keeps generations, and the floor
    counts fits.

  The evidence-count test near 9070 must **not** move.
- [ ] **Step 4: Run everything.** `cargo test -p providers --release` (~5.5 min) and
  `cargo test -p app --release --bin app -- activity` (the words) → PASS.
- [ ] **Step 5: `physics-reviewer` pass, then commit.**

---

### Task 5: λ_T and λ_E as pure functions over P_A — **Fable**

**Files:**

- Modify `crates/providers/src/surrogate/calibrate.rs`: P_A selection, the two estimators,
  `ClassCalibration` fields, the module doc section "The tail-aware inflations".
- Test: `crates/providers/src/surrogate/tests.rs`, calibrate block.

**Interfaces (Produces):**

```rust
pub(super) fn in_answer_sample(p: &AuditPoint) -> bool            // flags ¬FENCED ∧ (CADENCE_FRESH ∨ TICK) ∧ (code ∈ {5,6,21} ∨ TICK)
pub fn tail_inflation(atoms: &[TailAtom], floor_met: impl Fn(&[TailAtom]) -> bool) -> f64   // λ_T
pub fn energy_inflation(points: &[TailPoint], floor_met: impl Fn(&[TailPoint]) -> bool) -> f64 // λ_E
pub struct TailAtom { pub point: usize, pub fit: Option<u64>, pub r: f64, pub e: f64 }     // r = s/b, e = e/b
pub struct TailPoint { pub point: usize, pub fit: Option<u64>, pub r: f64, pub e: f64 }    // r, e in tier units
// ClassCalibration gains: pub tail_inflation: f64, pub energy_inflation: f64, pub answer_points: usize
```

- **The algorithm is determined by the spec, so no body is given here.**
  - Sort the atoms by r ascending, grouping ties. Each A(λ) is then a prefix P_m, and the prefix
    grows as λ falls.
  - Let m* be the largest m such that every prefix P_1 … P_m is good.
  - If m* covers every atom with r ≤ 1, λ_T = 1.
  - Otherwise λ_T = the next float above 1/r_{m*+1}. If r_{m*+1} = 0, λ_T = ∞.
  - Each prefix's goodness is evaluated incrementally (running max, running per-point sums), so the
    whole computation costs one sort plus one pass.
- **Excluded from P_A by `in_answer_sample`:** draft-mode points (`draft`) and `DraftAudit`.
- **The max conditions:** 3 for λ_T and **1** for λ_E.
- **The rms condition:** only when the set meets the floor.
- **The window:** the same window of generations as today.
- **NaN and zero:** per the Global Constraints.

- [ ] **Step 1: Write the failing tests.**
  - `the_tail_inflation_refuses_the_measured_tail`: 80 benign points over 16 fits, r ∈ [0.05, 0.9],
    e/b ≤ 0.5, plus one atom at r = 0.13, e/b = 4.48. λ_T is in (1/0.13, 1/0.13·(1 + 1e-12)], and
    λ_sum stays below 2.
  - `a_harmless_confident_point_does_not_inflate`: e = 0.3 b at r = 0.02 among 12 benign fits →
    λ_T = 1.
  - `the_energy_inflation_refuses_energy_only_failures`: fewer points than the floor, some at e_p
    1.03–1.39 and r_p 0.2, benign ones at e_p ≤ 0.1 and r_p up to 0.9. λ_E is just above 5, from the
    max condition alone. With a max of 4.53 it would be 1.
  - `the_inflations_never_read_what_they_selected`: added code 14/15/24 points, ordinary audits,
    fenced points, or points fresh only by a refusal leave both bit-identical.
  - `no_inflation_shrinks_the_std`: on 1000 seeded random histories, max(λ_sum, λ_T) ≥ λ_sum ≥ 1 and
    λ_E ≥ 1.
  - `non_finite_and_zero_rules`, per M1.
  - Run `cargo test -p providers --release inflation` → FAIL.
- [ ] **Step 2: Implement.** Run → PASS.
- [ ] **Step 3: `physics-reviewer` pass** on the estimators against spec §4, then commit.

---

### Task 6: Rules 6/7 on λ_F and λ_E; code 24; sparse keys — **Fable**

**Files:**

- Modify:
  - `crates/providers/src/surrogate/trust.rs`: `key_inflation` becomes `key_inflations` (λ_F, λ_E),
    with the class floors;
  - `crates/providers/src/surrogate/policy.rs`: `atom_inflation` and `variance_check`, l. 660–720;
  - `crates/providers/src/surrogate/audit.rs`: `HISTORY_RULE_TAIL_INFLATION = 24` and
    `history_rule_name`;
  - `crates/providers/src/surrogate/mod.rs`: `evaluate` records code 24 when the step passes at
    λ_sum.
- Test: `crates/providers/src/surrogate/tests.rs`.

**Interfaces:**

```rust
pub(super) fn key_inflations(&self, k: &surrogate::EnvKey) -> (f64, f64)   // (λ_F = max(λ_sum, λ_T), λ_E), class floors for a sparse key (< 3 own P_A points)
pub const HISTORY_RULE_TAIL_INFLATION: u8 = 24;
```

- **Rule 7** uses max(max_j λ_F(j), max_k λ_E(k)). The max propagates NaN.
- **λ_E's N and rule 7's N are the same normalisation.** If branch `stray-atoms` merges first, that
  is n_bonded (spec §5); the implementer checks `main` at the start of this task.
- **Refusal words** name the factor, e.g. "(the model's std × λ_T 7.7: a measured atom at 4.5× the
  bound with the model's std 0.13× of it)". The text is byte-identical when λ = λ_sum.

- [ ] **Step 1: Write the failing tests.**
  - `a_sparse_key_takes_its_class_inflations_as_floors` (M2).
  - `a_step_refused_only_by_the_tail_inflation_is_code_24`.
  - `a_nan_inflation_refuses`.
  - `a_key_locked_by_its_tail_inflation_says_so_and_keeps_measuring` (Review Focus 3).
  - `rule_7_uses_the_energy_inflation_while_forces_pass`.
  - Run `cargo test -p providers --release inflations` → FAIL.
- [ ] **Step 2: Implement.** Run → PASS. Run all `providers` tests.
  - `h2_plus_h_falls_back_on_the_grade_until_enough_is_learned` (tests.rs:3994) may move from
    26/40 and 40/40. Re-pin it with the reason "λ_T/λ_E (spec 2026-11-18 §4)" and the measured counts.
- [ ] **Step 3: `physics-reviewer` pass, then commit.**

---

### Task 7: Calibration ticks — **Opus**

Physics-reviewed for σ-blindness.

**Files:**

- Modify:
  - `crates/providers/src/surrogate/mod.rs`: the in-envelope tick counter, `TICK_EVERY`, and its
    module-doc section;
  - `crates/providers/src/surrogate/audit.rs`: an auditor job tagged as a tick, and the in-step path
    when there is no auditor.
- Test: `crates/providers/src/surrogate/tests.rs`.

**Interfaces (Produces):**

```rust
pub const TICK_EVERY: usize = 25;     // in-envelope steps, per provider; independent of audit_every
pub fn ticks_taken(&self) -> u64; pub fn ticks_skipped(&self) -> u64; pub fn ticks_in_step(&self) -> u64;
```

- **In-envelope:** rules 2–5e and the fences passed. The tick is decided before the variance and
  forces it.
- **At a fallback:** the point is recorded with `TICK`.
- **At an answered step:**
  - with an auditor that is free: an async audit job carrying `TICK`, judged by rule 8 too;
  - with an auditor that is busy: skipped and counted, whatever answered;
  - with no auditor: in-step.
- An abandoned job (Forget, install) counts as skipped.

- [ ] **Step 1: Write the failing tests.**
  - `a_calibration_tick_does_not_read_the_std`:
    - σ scaled ×10 gives the same tick steps;
    - with a busy auditor, a due tick is skipped whether the step was answered or not, and never
      served in-step;
    - with no auditor, ticks are served in-step.
  - `skipped_ticks_are_counted_and_shown` (Review Focus 4).
  - `rule_8s_cadence_is_unchanged_by_ticks`: the same audit steps with ticks on and off.
  - Run `cargo test -p providers --release tick` → FAIL.
- [ ] **Step 2: Implement.** Run → PASS. Run all `providers` tests.
- [ ] **Step 3: `physics-reviewer` pass, then commit.**

---

### Task 8: The app: readouts and words — **Opus**

**Files:**

- Modify:
  - `crates/app/src/ui/activity/words.rs`;
  - the Model-tab class panel (grep `SpeciesVerdict` in `crates/app/src/ui/`);
  - `crates/sim/src/provider.rs`: `SpeciesVerdict` gains `fits`, `floor_points`, `tail_inflation`,
    `energy_inflation`;
  - `crates/providers/src/surrogate/trust.rs`: fills them in.
- Test: the app's words tests and a `--screenshot` key, if the panel has one (`docs/dev-reference.md`).

**Interfaces:**

- Consumes `SpeciesVerdict` and `ticks_*()`.
- Produces:
  - the panel lines "independent checks: {fits} of 12 fits, {points} of 60";
  - "λ_T {x}, λ_E {y}";
  - "calibration checks skipped: {n}".

- [ ] **Step 1: Write the failing tests.** Byte-pinned words for `ClassKey::Floor`, the code-24 note,
  and the skipped-ticks line. Run `cargo test -p app --release --bin app -- activity` → FAIL.
- [ ] **Step 2: Implement.** Run → PASS. Run `cargo test -p app --release` (~10 min, incl. the
  replays).
- [ ] **Step 3: Commit.** List for the user's hand-test: a loaded collection key below the floor
  (its words and the teach prompt), and the skipped-ticks line at full speed.

---

### Task 9: The gate, the measurements, the docs — **Opus** (validation-runner)

Moved pins are physics-reviewed.

**Files:**

- Modify:
  - `crates/validate/tests/surrogate.rs`: report lines only, never a constant;
  - `crates/validate/references.md`;
  - `README.md`;
  - `DEVLOG.md`;
  - the docs that cite the floor or λ: the `calibrate.rs` and `mod.rs` module docs, already updated
    in Tasks 4–7, checked here.

- [ ] **Step 1: Add the repro's measurement lines.** Per run and key: n_fit, n_pts, λ_sum, λ_T, λ_E;
  ticks taken / skipped / in-step; code-24 refusals; judged z raw and inflated (‖ΔF‖/(λ_F σ_epi));
  worst ratios; solves; wall time. These are printouts, with no new assertions.
- [ ] **Step 2: The gate, run alone on the GPU.**
  - `cargo test -p validate --release --test surrogate the_stuck_escape_repro_is_answered_honestly
    -- --nocapture` → **PASS with every constant unchanged**.
  - On FAIL, report the per-key numbers and stop. The next step is policy (spec §6), never a
    tolerance.
- [ ] **Step 3: The remaining suites, one at a time.**
  - `cargo test -p validate --release --test surrogate` (~12–17 min);
  - `--test surrogate_water`;
  - `cargo test -p units -p qm -p sim -p surrogate --release`;
  - `cargo test -p providers --release`.

  Each moved pin, `active_learning_run_converges`'s DFT-call count and
  `surrogate_knows_what_it_does_not_know` among them, is re-pinned with its reason and reviewed by the
  `physics-reviewer`.

  `teaching_around_a_blocker_restores_its_class` must **not** move (spec §5). If it does, that is a
  finding to report, not a re-pin.
- [ ] **Step 3b: Task 7's carried checks (Task 7 review I1, M4).**
  - **I1/R1 (the dead time):** in the repro's tick lines (`SurrogateKs::tick_counts`), compare the
    **population** answered share, `population_answered / population` (every in-envelope step), with
    the answered share among **due** ticks, `answered_due / due`, and among taken ticks. Also
    compare the take rates at answered steps (`answered_skipped_busy / answered_due`) and refused
    ones. A due share materially off the population's is a σ-coupling of which steps become ticks;
    a lower take rate at answered steps is one through skips (a job outliving D, or rule 8). Report
    D (the timed dead time) beside them.
  - **M4:** check whether a parked locked key's λ_T lifts by the 4096-point history ring (code-24
    points evicting the bad tick) before the window ages it, and report which.
- [ ] **Step 4: The measurements.**
  - `trust_baseline_report` (`--ignored`): z rms per class and missed large errors, before and after.
  - The Task 0 report re-run on a fresh copy.
- [ ] **Step 5: The docs.**
  - `references.md`: a new section "Trust tightening (2026-11-19)" with the gate's numbers against
    07939ac (1036 s) and 762159b (1378 s), and the measurements. The stuck-escape record is updated.
  - `README.md`: "trust now needs 12 independent checks (fits) and 60 comparisons before measured
    evidence alone opens a class; rules 6/7 also apply tail-aware inflations".
  - `DEVLOG.md`: timeline, durations, failures, wins.
- [ ] **Step 6: Commit.** Hand off for the merge (`superpowers:finishing-a-development-branch`).

---

## Rulings made in this plan (spec gaps)

- **Formats.** The spec says older builds ignore the new keys. That holds for neither the standalone
  history (trailing arrays are an error) nor, possibly, the sidecar. Task 1 bumps the magics
  symbolically and registers them in `app::update::formats`, so the downgrade path warns and never
  overwrites a newer file.
- **Task 0's read path.** `Collection::open_read_only` does not read `history.bin`, so Task 0 opens a
  scratchpad copy normally. It also needs one read-only readout (`key_class_stats`), because the
  per-key LOO statistics are not public.
- **Fit ids across sessions.** The spec does not say how ids stay unique. A refit takes
  max(own, history max) + 1, and an install or restore brings back the fit's own id (Task 2).
  Task 2's rulings where the brief met the code:
  - The id is stamped once per `refit` (at its end), not per solve: a refit's intermediate fits
    (the recovery's trials and restores) do not count, and a refit whose standing model has the
    same coefficients, bit for bit, as the one before is not a new fit (the spec's "solved new
    coefficients"; a user's repeated Refit or a refit whose new samples are all held out adds none).
  - New coefficients count only from new teacher data (fix round 1, review I1): a watermark
    `fit_id_samples` (the store length when the id was last taken) — a refit takes an id only when
    its standing fit holds a member at or past it, then moves the watermark to the store length.
    A forget + restore, the recovery's holds and releases or a degree-cap retry re-solve old data
    and add no fit. Set at an install (the file's fitted prefix), an attach (the collection's
    length), a restore (the saved value) and a forget (0). It can only under-count.
  - A standing model's id is ≥ 1 (fix round 1, M1): a legacy install (no key, no id in the history)
    takes 1, and a first fit — a collection with no id on record included — takes ≥ 1. So a
    point's `Some(0)` never occurs; Task 3 on, `None` alone means absent.
  - An install's rebuild (`fit_store`, lone atoms or a membership mismatch) stamps nothing: the
    file's id stands.
  - An attach keeps the collection history's largest id (Review Focus 5: a re-attach or a re-spawn
    adds no fit), its own refit or a later one of what it holds included (the watermark); a refit
    with new samples takes the next. If the collection grew after its newest audited fit, its fit is
    a new one under the old id: an under-count, the conservative side of the floor.
  - The sidecar's `fit_id` header key is optional and bumps no version (every reader ignores header
    keys it does not read); a file without it installs at its history's largest id.
- **The magic values read (Task 1 step 1).**
  - The standalone history: `surrogate::history::MAGIC` = `ASAUDIT1`, the only version. Its decoder
    refuses anything after the optional species array ("trailing bytes after the last array"), so
    it bumps: `ASAUDIT2` (n + 1 = 2). The `history` format number goes from 1 to 2.
  - The sidecar: `surrogate::sidecar::MAGICS.len()` = 9, newest `ASMODEL9` (bodies 6, 7, 8). Its
    reader takes exactly the declared arrays its layout names (`read_arrays`: "N arrays declared,
    format version V has M"), so three more history arrays are refused by a version-9 build: it
    bumps too. K + 1 = 10, whose magic under the layout fixed in version 9 is `ASMOD010`. The
    `sidecar` format number goes from 9 to 10.
  - Both bumps apply only when the new arrays are written: a history none of whose points carries
    a fit id, a flag or an epistemic energy std is still written as `ASAUDIT1` / `ASMODEL9`, bit for
    bit (the version-8 precedent: a basis without body-4 stayed version 7). An older build keeps
    reading every file that holds nothing it cannot read.
  - An absent fit id or energy std is stored as **−1** in its array (any other negative, a NaN or an
    infinity is refused); an absent flags word is 0.
  - This build refuses an audit history of a newer build (`ASAUD…` that is not `ASAUDIT1`/`2`) and
    never overwrites it: the collection opens with an empty history and a note, and
    `save_history` leaves `history.bin` as it is (Review Focus 1). The downgrade warning says so for
    a target whose `history` number is 2 or more; older targets still overwrite, and are warned as
    before.
