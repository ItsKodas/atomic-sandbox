# Stray Atoms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In strict mode, let the model answer a bonded part plus stray atoms once a measured composition check has
cleared that combination for fragment geometries like the current one. Until then, keep such samples out of the fit.

**Architecture:** A new `providers::surrogate::stray` module holds the pure logic: the split into bonded part and
strays, the signature, the static floor, the check's verdict, and the cleared region. A per-session `StrayBook` on
`SurrogateKs` holds each signature's state. Rule 5a's fallback queues two kinds of auditor job, a fragment-alone solve
and an atom-at-its-position solve; their answers are judged into the book. The policy then exempts a cleared, in-region
scene from 5a. Stray-atom samples carry an in-memory hold-out code until a check covers them.

**Tech Stack:** Rust workspace (`qm`, `qm-gpu`, `sim`, `providers`, `surrogate`, `app`, `validate`), wgpu/Vulkan, egui 0.36.

**Spec:** `docs/superpowers/specs/2026-11-18-stray-atom-composition-design.md` (approved 2026-11-18). Read it first; the
section numbers below are the spec's.

## Global Constraints

- Atomic units inside `qm`, `sim` and `providers` (Hartree, Bohr). mHa appears only in UI words.
- No per-element parameter (CLAUDE.md rule 1). Clearance is keyed by Z multisets and measured.
- Strict mode only. Draft mode is unchanged.
- **Without a background auditor** (the CPU backend's synchronous audits), no check runs: rule 5a stands as today.
  Stray samples are still held out (§4.3 C1). My ruling: a synchronous fragment solve would stall the step.
- K = 3 passing checks, at bonded geometries beyond the store's thinning radius from each other; a pass differing only
  in stray distance does not count.
- Frontier margin: 27.6 kT = 138 mHa = 0.138 Ha at kT = `qm::ELECTRON_KT` (write it as `27.6 * qm::ELECTRON_KT`).
  Level-matching tolerance: 1.8e-3 Ha (the box splitting).
- Energy test: |E_scene − E_frag − Σ_s E_atom(R_s)| ≤ `policy.energy_std_per_atom` × n_stray. Force test:
  `FallbackPolicy::force_residual` per atom (bonded: scene vs fragment; stray: scene vs its atom solve).
- Box rule: a stray atom within 4.0 Bohr of a face is floor-refused before a solve. Any check solve with face density
  above `box_affected_face_density(h)` makes the check void.
- Every per-atom normalisation for a cleared answer uses n_bonded: rule 7, its (Nσ_e)² floor, and the audit criterion.
- Session-only state; nothing new is written to a collection or model file. A held-out stray sample is never rewritten.
- Model per task: all subagents on Opus (user memory, overrides the Fable tags). Physics-reviewer before Tasks 3–5 land.
- One heavy GPU test binary at a time; no GPU tests while the user's `app.exe` runs (check `Get-Process app` first).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. README updated in Task 6.

## Review Focus

1. **A stray atom that wanders back and out repeatedly** (a vibrating near-cutoff H). Expect no flicker of
   model ↔ teacher without a booked jump, and no sample flood. Test in Task 4: `reentry_and_exit_book_every_switch`.
2. **Two different stray species leaving one molecule** (H and O from water). Expect a floor refusal naming mixed
   stray species, not a check. Test in Task 2: `floor_refuses_mixed_stray_species`.
3. **A stray atom drifting toward a face during a cleared run.** Expect the step to fall back as soon as it is within
   4 Bohr, with the reason shown. Test in Task 4: `a_stray_near_a_face_falls_back_inside_a_cleared_region`.
4. **The user deletes or edits an atom while a check is in flight.** Expect the late answer to be discarded (scene
   generation changed), with no verdict and no crash. Test in Task 3: `a_check_answer_after_an_edit_is_dropped`.
5. **A loaded collection full of old stray samples.** Expect them held out, the fit to proceed without them, and the
   files byte-identical. Test in Task 3: `loaded_stray_samples_are_held_out_not_rewritten`.

---

### Task 1: Per-spin frontier levels in the solve statistics

**Files:**
- Modify: `crates/sim/src/provider.rs` (`EvalStats`, beside `partial` at ~647)
- Modify: `crates/providers/src/ks.rs` (CpuKs and GpuKs fill it)
- Test: `crates/providers/src/ks.rs` tests

**Interfaces:**
- Produces: `pub struct Frontier { pub homo: Option<f64>, pub lumo: Option<f64> }` in `sim::provider` (re-exported
  from `sim`), and `EvalStats::frontier: [Frontier; 2]` (↑ then ↓, Ha). `homo` is the highest eigenvalue with f ≥ 1/2,
  `lumo` the lowest with f < 1/2 among the computed block; `None` when the channel has no such level. Default:
  `[Frontier { homo: None, lumo: None }; 2]`.

- [ ] **Step 1:** Write the failing tests `cpu_h2_reports_frontier_levels` and `gpu_h2_reports_frontier_levels` (the
  GPU one `#[cfg_attr(debug_assertions, ignore)]`). H₂ at 1.4 Bohr, 41³ × 0.25, automatic. Assert ↑ and ↓ `homo` are
  equal to within 1e-6, `homo < lumo` in each channel, `homo` within 1e-6 of the solve's own lowest eigenvalue, and
  CPU vs GPU `homo` agree within 1e-3.
- [ ] **Step 2:** Run `cargo test -p providers --release frontier`. Expect a compile FAIL (no field).
- [ ] **Step 3:** Add `Frontier` and the field. Fill it in both providers from `evals_up/dn` and `occ_up/dn` (CPU
  `KsState`; GPU state at the same place `partial` is filled). When a channel has no unoccupied computed level, `lumo`
  is `None`.
- [ ] **Step 4:** Run the tests and expect PASS. Then run `cargo test -p providers --release` (the full gate) and expect
  0 failures.
- [ ] **Step 5:** Commit `feat(sim, providers): per-spin frontier levels in EvalStats`.

### Task 2: `stray` module: split, signature, floor, verdict, region (pure logic)

**Files:**
- Create: `crates/providers/src/surrogate/stray.rs` (module doc: spec §2 and §4 condensed, with its derivations)
- Modify: `crates/providers/src/surrogate/mod.rs` (`mod stray;`)
- Test: `crates/providers/src/surrogate/tests/stray.rs` (new, registered like `tests/policy_cost.rs`)

**Interfaces:**
- Produces:
  - `pub(super) struct Split { pub bonded: Vec<usize>, pub stray: Vec<usize> }` and
    `pub(super) fn split(pos: &[[f64; 3]], r_cut: f64) -> Option<Split>`. `None` unless n ≥ 2, at least one stray,
    and at least one bonded atom.
  - `#[derive(Clone, PartialEq, Eq, Hash, Debug)] pub(super) struct Signature { pub bonded_z: Vec<u8>, pub stray_z: Vec<u8>, pub charge: i32, pub surface: String }`,
    both Z lists sorted. `surface` is the store provenance's surface key (`Provenance`'s grid/h/σ/tol/teacher/ionic,
    formatted as `lone::same_reference_settings` compares them).
  - `pub(super) fn floor_refusal(scene: &Scene, split: &Split, m_ref: &dyn Fn(u8) -> Option<(u32, Option<bool>)>, fenced: &dyn Fn(u8, u8) -> bool, cell_faces: &dyn Fn([f64; 3]) -> f64) -> Option<String>`.
    It returns the reason words or `None`. `m_ref(z)` is the reference multiplicity and its partial flag (`lone`'s
    source); `cell_faces(r)` is the distance to the nearest face (Bohr). The checks run in spec §4.2's order: charge;
    forced multiplicity; the spin rule at automatic; ≥ 2 strays not `closed_spin_channels` or of mixed species; a
    shell fence on (stray Z, any other Z); a stray within 4.0 Bohr of a face.
  - `pub(super) struct PartAnswer { pub energy: f64, pub forces: Vec<[f64; 3]>, pub frontier: [sim::Frontier; 2], pub partial: Vec<sim::PartialLevel>, pub clean: bool }`
    (`clean`: no warning, trainable, face density within the box rule).
  - `pub(super) enum CheckVerdict { Pass { residual: f64, margin: f64 }, Fail(String), Void(String) }`, and
    `pub(super) fn judge(policy: &FallbackPolicy, scene: &PartAnswer, frag: &PartAnswer, atoms: &[PartAnswer], split: &Split) -> CheckVerdict`,
    with the tests and constants from Global Constraints. `residual` is the absolute energy residual (Ha); `margin` is
    the smallest frontier margin over channels and strays (Ha). A missing `lumo` or `homo` that a margin needs gives
    `Void("frontier level not computed")`.
  - `pub(super) struct Region { env: surrogate::Envelope, points: Vec<Vec<Vec<f64>>> }` with
    `fn new(basis: &Basis) -> Region`, `fn add(&mut self, bonded_desc: &[Vec<f64>], species: &[usize])`,
    `fn distinct(&self, bonded_desc: &[Vec<f64>], thin: f64) -> bool` (beyond `thin` from every stored point, in the
    store's thinning metric), `fn contains(&self, bonded_desc: &[Vec<f64>], species: &[usize]) -> bool` (every bonded
    atom within its species' per-feature `[min, max]`; a species absent from the region → false).
  - `pub(super) enum StrayState { Checking { passes: usize, region: Region }, Cleared { region: Region, residual: f64, margin: f64 }, Refused(String) }`,
    and `pub(super) struct StrayBook { map: HashMap<Signature, StrayState> }` with `fn state(&self, s: &Signature) -> Option<&StrayState>`,
    `fn record(&mut self, s: Signature, v: CheckVerdict, desc: &[Vec<f64>], species: &[usize], basis: &Basis, thin: f64)`
    (Pass: if distinct, `passes += 1` and add to the region; at 3, Cleared; when already Cleared, add to the region.
    Fail: Refused. Void: no change), `fn refuse(&mut self, s: &Signature, why: String)`, `fn clear(&mut self)`.
  - `pub(super) const STRAY_CHECKS: usize = 3;` `pub(super) const STRAY_FACE_MARGIN: f64 = 4.0;`
    `pub(super) const STRAY_LEVEL_TOL: f64 = 1.8e-3;` `pub(super) fn frontier_margin_min() -> f64 { 27.6 * qm::ELECTRON_KT }`.

- [ ] **Step 1:** Write the failing tests:
  - `split_finds_bonded_and_stray`: H₂ at 1.4 Bohr plus H at 8 Bohr gives bonded [0, 1], stray [2]. All-lone and
    all-bonded give `None`.
  - `floor_refuses_charge_forced_m_and_spin`: charge +1 refuses; forced M refuses; H₂ + 2 H at automatic (M = 1 vs
    needed 3) refuses with words containing "spin"; H₂ + H at automatic passes.
  - `floor_refuses_mixed_stray_species`: H₂O bonded, strays H and O, refuses naming "mixed".
  - `floor_refuses_a_stray_near_a_face` at 3.9 Bohr; 4.1 passes.
  - `judge_passes_an_exact_split`; `judge_fails_an_energy_offset` (offset 2 × tolerance); `judge_fails_a_narrow_margin`
    (stray homo 0.1 Ha below frag lumo); `judge_fails_mismatched_partial_levels`;
    `judge_voids_an_unclean_solve_and_a_missing_lumo`.
  - `region_clears_after_three_distinct_passes`; `passes_at_one_geometry_do_not_clear`;
    `region_contains_its_points_and_not_beyond`; `one_fail_refuses_for_good`.
- [ ] **Step 2:** Run `cargo test -p providers --release stray`. Expect FAIL (module missing).
- [ ] **Step 3:** Implement `stray.rs` to the interfaces. Reuse `lone::closed_spin_channels` and `no_neighbours`; do not
  copy them.
- [ ] **Step 4:** Run the same command and expect PASS.
- [ ] **Step 5:** Commit `feat(providers): the stray module - split, signature, floor, check verdict, cleared region`.

### Task 3: Checks on the auditor, the book, and the sample hold-out (C1)

**Files:**
- Modify: `crates/providers/src/surrogate/audit.rs` (`JobKind::Fragment`, `JobKind::AtomAt(u8)`; receive and route)
- Modify: `crates/providers/src/surrogate/mod.rs` (`stray_book: StrayBook` and `stray_pending` on `SurrogateKs`; the
  rule-5a fallback path queues a check)
- Modify: `crates/providers/src/surrogate/quarantine.rs` (`Q_STRAY`, in memory only, like `Q_LONE`; released on cover)
- Modify: `crates/providers/src/surrogate/store.rs` (the keep path sets `Q_STRAY` on a stray sample; load sets it too)
- Test: `crates/providers/src/surrogate/tests/stray.rs`

**Interfaces:**
- Consumes: Task 2 (everything), Task 1's `EvalStats::frontier`.
- Produces:
  - `pub(super) struct StrayCheck { sig: Signature, split: Split, scene: PartAnswer, frag: Option<PartAnswer>, atoms: Vec<Option<PartAnswer>>, scene_gen: u64, sample: Option<usize> }`
    on `SurrogateKs.stray_pending: Option<StrayCheck>` (at most one in flight; a newer fallback does not replace it).
  - `fn queue_stray_check(&mut self, scene: &Scene, ans: &ForceResult, split: Split, sig: Signature, kept: Option<usize>)`.
    It pushes `(fragment scene, JobKind::Fragment)` and one `(atom scene, JobKind::AtomAt(z))` per stray onto
    `side_queue`, behind any audit. Fragment: the bonded nuclei only, the same `Cell`, charge 0, multiplicity automatic.
    Atom: one nucleus at its scene position, multiplicity `m_ref(z)`. Both jobs keep the **scene's per-species ionic
    setup** (review M5): never recompute the surface tag from the fragment's own species.
  - `fn take_stray_answer(&mut self, ans: AuditAnswer)`: fills the pending check. When it is complete, it calls
    `stray::judge`, then `stray_book.record`, then `self.release_stray_samples(&sig)` on Cleared or a covering pass,
    and records `StrayOutcome` for the Activity (Task 6).
  - `fn release_stray_samples(&mut self, sig: &Signature)`: `Q_STRAY` → `Q_NONE` for the store samples of `sig` whose
    bonded descriptors the region contains, then `arm_refit`.
  - `pub(super) const Q_STRAY: u8` = the next value no in-memory code uses **and** no collection status record can
    hold (collection code 9 is the migration's, `surrogate::collection`). It is never written; `views.rs` and
    `shared.rs` treat it like `Q_LONE`. A test asserts it differs from every `Q_*` and every collection code.
- `scene_gen`: the store's scene-edit generation (the one `poll_side_jobs` already compares for Refresh jobs). A
  mismatch drops the check silently.

- [ ] **Step 1:** Write the failing tests (Morse teacher with an auditor, as the existing async-audit tests build one):
  - `a_stray_fallback_queues_fragment_and_atom_jobs`: H₂ + H at 8 Bohr; one strict step queues both jobs; the
    sample is kept with `Q_STRAY`.
  - `three_distinct_passes_clear_and_release`: three fallbacks at H–H 1.35/1.45/1.55 Bohr clear the signature, and
    the three samples go to `Q_NONE`.
  - `a_composite_offset_refuses_and_holds_out_for_good`: a Morse wrapper adds 5e-3 Ha when a stray is present.
    Refused, and the samples stay `Q_STRAY`.
  - `floor_refused_samples_are_never_kept`: two stray H at automatic; the sample is not kept and no job is queued.
  - `a_check_answer_after_an_edit_is_dropped` (Review Focus 4).
  - `loaded_stray_samples_are_held_out_not_rewritten` (Review Focus 5): a collection with a stray sample loads
    with it `Q_STRAY`, and `samples.bin` hashes identical before and after.
  - `no_auditor_no_check_but_held_out`.
- [ ] **Step 2:** Run `cargo test -p providers --release stray`. Expect FAIL.
- [ ] **Step 3:** Implement to the interfaces. The keep-path hold-out goes in `keep_sample_masked`, beside the lone
  check (`store.rs` ~647): a stray sample whose signature is not Cleared-and-covering is pushed with `Q_STRAY`. A
  floor-refused one is not kept (outcome `TeachOutcome::StrayNotKept`, added in `sim::activity`).
- [ ] **Step 4:** Run the tests and expect PASS. Then run `cargo test -p providers --release` and expect 0 failures.
- [ ] **Step 5:** Physics review (physics-reviewer, Opus) of Tasks 2–3's diff against spec §4.2–4.3, then any fix round.
- [ ] **Step 6:** Commit `feat(providers): stray-atom composition checks on the auditor; stray samples held out until measured`.

### Task 4: Answering a cleared, in-region scene (5a exemption, n_bonded)

**Files:**
- Modify: `crates/providers/src/surrogate/policy.rs` (`envelope_outside` ~470 and `atom_verdicts` ~580: skip 5a for a
  cleared, in-region signature; strays exempt from 5b–5e)
- Modify: `crates/providers/src/surrogate/mod.rs` (`evaluate`: the cleared decision before the envelope; rule 7 and
  its floor with n_bonded; the step note)
- Modify: `crates/providers/src/surrogate/inspect.rs` (`would_answer` mirrors it)
- Test: `crates/providers/src/surrogate/tests/stray.rs`

**Interfaces:**
- Consumes: Task 3's `stray_book`, Task 2's `Region::contains`.
- Produces: `fn stray_cleared(&self, pos: &[[f64; 3]], z: &[u8], prov: &Provenance) -> Option<(Split, f64, f64)>`
  (the split, the residual and the margin when cleared and in-region, after re-checking the face margin; `None`
  when a stray species is not in the model's basis, rule 2, review M2). The step
  note is exactly: `"stray atoms: model for the bonded part + free atoms (composition measured, 3 checks; worst residual {:.2} mHa, frontier margin {:.0} mHa)"`.

- [ ] **Step 1:** Write the failing tests:
  - `a_cleared_scene_is_the_models`: after clearing, a fourth in-region geometry is `Source::Surrogate`. Its energy is
    within 4 ulp of `predict(frag) + Σ ref_energy`, bonded forces are bit for bit, stray forces exactly 0, and the note
    is present.
  - `outside_the_region_falls_back_and_checks`: H–H 1.9 Bohr.
  - `rule7_uses_n_bonded`: a variance that passes at n = 3 but fails at n = 2 falls back.
  - `reentry_and_exit_book_every_switch` (Review Focus 1): the stray H on a path 8 → 5 → 8 Bohr. Every
    model ↔ teacher switch carries a `transition_jump` booking, and `E_conserved` has no unbooked step.
  - `a_stray_near_a_face_falls_back_inside_a_cleared_region` (Review Focus 3).
  - `would_answer_agrees_with_evaluate` on those scenes.
- [ ] **Step 2:** Run `cargo test -p providers --release stray`. Expect FAIL.
- [ ] **Step 3:** Implement to the interfaces.
- [ ] **Step 4:** Run the tests and expect PASS. Then run `cargo test -p providers --release` and expect 0 failures.
- [ ] **Step 5:** Commit `feat(providers): the model answers a cleared stray-atom scene inside its measured region`.

### Task 5: Audits of cleared answers and their attribution (I6)

**Files:**
- Modify: `crates/providers/src/surrogate/audit.rs` and `trust.rs` (the audit judge with n_bonded; the failure path)
- Test: `crates/providers/src/surrogate/tests/stray.rs`

**Interfaces:**
- Consumes: Tasks 3–4.
- Produces: on a failed audit of a cleared stray answer, three things happen in order. `stray_book.refuse(sig, why)`
  and `Q_STRAY` on that signature's samples; `queue_stray_check` at the audited geometry with the audit's own answer
  as `scene`; and `FailedAudit` entering the model's audit record **only if** that check's fragment solve shows the
  model wrong on the fragment by `FallbackPolicy::agree` (`TeachOutcome::StrayAuditAttributed { model: bool }`).

- [ ] **Step 1:** Write the failing tests: `a_failed_stray_audit_refuses_and_blames_by_a_check` (composite offset after
  clearance: refused, and the model untouched) and `a_failed_audit_with_a_wrong_fragment_counts_against_the_model`
  (the Morse fragment perturbed: the model's record takes it).
- [ ] **Step 2:** Run `cargo test -p providers --release stray_audit`. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and expect PASS. Then run `cargo test -p providers --release` and expect 0 failures.
- [ ] **Step 5:** Physics review of Tasks 4–5 (n_bonded, booking, attribution), then any fix round.
- [ ] **Step 6:** Commit `feat(providers): audits of stray-atom answers; a failure refuses the signature and is attributed by a check`.

### Task 6: What the user sees (Activity, trust card, README)

**Files:**
- Modify: `crates/sim/src/activity.rs` (`TeachOutcome::StrayChecking { n: usize, of: usize }`,
  `StrayCheckingNew`, `StrayRefused`, `StrayNotKept`, `StrayAuditAttributed`)
- Modify: `crates/sim/src/provider.rs` (`ModelStatus::stray: Vec<(String, String)>`, per signature: the label
  "H₂ + H" and the state words; `stray_held_out: usize`)
- Modify: `crates/app/src/ui/activity/words.rs`, `crates/app/src/ui/dock/model.rs` (the trust card lines)
- Modify: `README.md` (one paragraph under the model's fallback rules)
- Test: `crates/app/src/ui/tests.rs`

**Interfaces:**
- Words, exactly (spec §5): outcomes `"checking composition {n} of 3"`, `"checking composition (new geometry)"`,
  `"composition fails"`, and `"not kept: stray atoms"`. Trust card `"Stray atoms: {label} cleared (3 checks)"` /
  `"Stray atoms: {label} refused ({why})"`, and `"{k} stray-atom samples held out of the fit until measured"`.

- [ ] **Step 1:** Write failing app tests: `stray_outcomes_have_their_words` and `the_trust_card_lists_stray_signatures`.
- [ ] **Step 2:** Run `cargo test -p app --release --bin app -- stray`. Expect FAIL.
- [ ] **Step 3:** Implement. Add the README paragraph.
- [ ] **Step 4:** Run the tests and expect PASS. Then run `cargo test -p app --release` and expect 0 failures.
- [ ] **Step 5:** Commit `feat(app, sim): stray-atom checks in the Activity and the Model tab; README`.

### Task 7: Validation on the real teacher, and dynamics

**Files:**
- Modify: `crates/validate/tests/surrogate.rs` (module doc entry plus tests), `crates/validate/tests/dynamics.rs`
- Modify: `crates/validate/references.md` (measured numbers), `CLAUDE.md` suite times if they move

- [ ] **Step 1:** Write the tests (heavy, release-only):
  - `stray_h_clears_and_matches_dft`: GPU, H₂ + H at 8 Bohr, automatic, 61³ × 0.20, Accepted. It clears, and at an
    unchecked fourth in-region geometry |E_model − E_DFT|/n_bonded ≤ the audit tier. Print the residuals and margins.
  - `stray_li_beside_f2_is_refused`: F₂ (2.68 Bohr) + Li at 8 Bohr. Refused by the margin, level, or energy test.
    Print which test refused, the margin and the transfer, with no pre-judged size.
  - `stretched_h2_margin_is_reported`: H₂ at 2.5 Bohr + H. Print the σu − 1s margin; assert only that it is finite.
  - In dynamics: `stray_h_returning_conserves_energy`. NVE, H₂ + H sent back through r_cut and out, under the dynamics
    suite's drift bound, with the booked jumps reported separately.
- [ ] **Step 2:** Run them one binary at a time: `cargo test -p validate --release --test surrogate stray`, then
  `--test dynamics stray`. Expect PASS, and record the numbers in `references.md`.
- [ ] **Step 3:** Gates: `cargo test -p providers --release`, `cargo test -p validate --release --test surrogate`
  (the whole binary), and `--test dynamics`. Expect 0 failures, except failures already present on the base
  (`the_stuck_escape_repro_is_answered_honestly`, owned by the trust-tightening branch; note it, don't fix it here).
- [ ] **Step 4:** Commit `test(validate): stray atoms on the real teacher (H2 + H clears, F2 + Li refused) and NVE re-entry`.
