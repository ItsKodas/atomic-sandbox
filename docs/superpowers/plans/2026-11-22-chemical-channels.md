# Chemical channels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a surrogate whose body-3 and body-4 terms are built from K learnt chemical channels, so refits stop exploding with the element count. Shipped behind a switch, with fresh collections.

**Architecture:**
- `surrogate::descriptor::Basis` gains an optional `ChannelMix` (K, W S×K).
- Body 2 stays per species pair; bodies 3 and 4 mix the neighbour densities per channel.
- W is learnt from a low-order species fit by an SVD, then frozen and stored with the collection.
- The fit stays linear with the exact posterior, plus a per-element guard and a linearised W-variance term.

**Tech stack:** Rust (surrogate, providers, app), faer for dense linear algebra as now.

**Spec:** `docs/superpowers/specs/2026-11-22-chemical-channels-design.md` (with `docs/superpowers/research/2026-11-22-chemical-channels-literature.md`).

## Physics review changes (2026-11-22, binding)

The spec's section "Physics review, 2026-11-22: binding changes" (C1, C2, I1–I7) overrides the task text below where they differ. In particular:
- Task 1 adds I1's oracle, permutation, relabel and finite-difference tests.
- Task 2 follows I2/I3/I4 (the W recipe, K by held-out RMSE, the reduced low-order basis) and the extended gate I7.
- Task 4 keeps the envelope and the 5a/5b logic (no geometric replacement).
- Task 5 implements C1 (Σ_W = R·Σ_low·Rᵀ) and N_W, with no N_trust.
- Task 6 implements C2's per-key floor q_k and the I5 lifecycle.

## Global constraints

- Rule 1 exception (CLAUDE.md): W is learnt only from our DFT samples, never from tabulated properties.
- No old-collection compatibility or migration. Channel collections are new keys and start empty.
- f64 on the CPU for the fit and the model. No tolerance is loosened. Trust-floor meanings are unchanged.
- Tests run as you go:
  - every change: `cargo test -p surrogate --release` plus the task's new tests (~1 min);
  - end of day 2: `cargo test -p providers --release -- surrogate:: sampling::`, and `cargo test -p app --release --bin app -- ui:: worker::`;
  - before merging: `cargo test -p validate --release --test surrogate`, the full `-p app` and the full `-p providers`.
- Skip oracles, gpu_vs_cpu, dynamics, pseudo×3, harris, orbitals, charges and tools (untouched) unless a task touches qm, qm-gpu or sim's integrator.
- One heavy job at a time, at BelowNormal priority with `-j 4`. Never kill the user's processes.
- Agents work in separate git worktrees. Commits end with the Claude attribution line.
- Model tags: physics-review tasks and Task 5's uncertainty are Fable by the tags, but run on Opus (memory: no-fable-tokens).

## Review focus

1. **A brand-new element in a scene:** refused by the model (DFT) until N_W samples exist, never answered on a missing W row.
2. **Signed W:** the isolated-atom / lone-atom detection and the coordination ratio never read channel values (a neighbour present with a zero channel sum must not read as "isolated").
3. **K larger than the species count** (early collections with 1–2 elements): W is capped at rank ≤ S; no NaN or degenerate columns.
4. **Relearning W after the species set changes:** every cached feature and model is invalidated. No mixed-W rows in one fit.
5. **Determinism:** the same samples give the same W (sign convention) and bit-identical fits on the same machine.

---

### Task 0: delete the legacy compatibility code (prep, can run now)

**Files:**
- Delete:
  - `crates/surrogate/src/migration.rs`
  - `crates/providers/src/sampling/migrate.rs`
  - `crates/app/src/ui/migration.rs`
  - `crates/validate/tests/migration.rs`
- Modify (remove the legacy paths):
  - `crates/providers/src/surrogate/shared.rs`: `legacy_tag_for`, `predecessor_of`, `MigrationSlot`, `legacy_checkpoint_refusal`, `PASS_STARTED_WITH_PREVIOUS_SOLVER`, `LAST_PRE_FD_RELEASE`, `legacy_model_refusal`, and their call sites in `providers/src/surrogate/mod.rs`; the re-solved sample review (quarantine code 9);
  - `legacy_refs` (pre-S9 references) in `mod.rs`, `store.rs`, `shared.rs`, `background.rs`;
  - the calibration legacy points in `calibrate.rs`;
  - old sidecar magics and body versions (`sidecar.rs` `MAGICS`, `BODIES_V9`/`V10`, the v1 / v8 / pre-v5 branches);
  - the collection record v1 reading (`collection.rs`);
  - the migration lines in `app/src/update/formats.rs`, `pseudo::legacy_tag_of`, the app's migration offer, action and command, and the worker plumbing.
- Docs: README (the migration paragraph), `docs/dev-reference.md` (migration), CLAUDE.md's V11 row in the suite table.

- [x] **Step 1:** delete and trim; `cargo build --workspace` is clean, with no dead-code warnings left by the removal.
- [x] **Step 2:** delete the tests that only exercised removed behaviour; keep every test that exercises current behaviour.
- [x] **Step 3:** `cargo test -p surrogate --release`, `cargo test -p providers --release`, and `cargo test -p app --release --bin app -- ui:: worker::` pass.
- [x] **Step 4:** commit `refactor: remove old-collection migration and legacy formats (no compatibility, user 2026-11-22)`.

### Task 1: channel-mixed descriptors (day 1)

**Files:** `crates/surrogate/src/descriptor.rs` (Basis, body2_index unchanged, body3_terms, n_body3, for_each_body4, add_neighbour, describe_with, describe_values, touched_columns, feature_names, column_classes); tests in the same module.

**Interfaces:**
- Produces:
  - `pub struct ChannelMix { pub k: usize, pub w: Vec<f64> }`, with `w[t * k + c]` and t the species index;
  - `Basis::with_channels(self, mix: ChannelMix) -> Basis`;
  - `Basis::channels(&self) -> Option<&ChannelMix>`;
  - `Basis::n_independent_columns()` reflecting the channel P₃/P₄.

- [ ] **Step 1, the failing test `identity_channels_reproduce_the_species_basis`:** for random clusters of 2–4 species, `Basis::with_channels(ChannelMix { k: S, w: identity })` gives features equal to the species basis (body 3/4 reordered to the channel indexing; compare by `feature_names` mapping), to 1e-10 relative. Also test energies and forces through a random coefficient vector.
- [ ] **Step 2:** implement Ā_k = Σ_t W[t, k]·A_t, inside `add_neighbour` (spread φ into K rows weighted by W[z_j]). Index body 3/4 by (k, n, l). Drop the composition skips for bodies 3/4 when channels are on. Leave body 2 untouched.
- [ ] **Step 3, the test `channel_columns_count`:** Q′ = n_max·S(S+1)/2 + S·(P₃(K) + P₄(K)), for S ∈ {1..5}, K ∈ {1..6}, with K capped at S if larger (Review focus 3).
- [ ] **Step 4:** run `cargo test -p surrogate --release`; it passes. Commit.

### Task 2: learn W, the accuracy harness, the day-1 go / no-go (day 1)

**Files:**
- Create `crates/surrogate/src/channels.rs`: `pub fn learn_mix(samples, basis_species, k) -> ChannelMix`, the SVD of the unfolded body-3 coefficients of a low-order species fit (`dense::` routines), with the sign convention. Determinism test.
- Create `crates/providers/src/surrogate/tests/channels_gate.rs`: `#[ignore]` harness `channels_gate` reading `COLLECTION_DIR` (a copy), as `refit_profile` does. It fits the species model and the channel model at K = 3/4/6 and prints a table:
  - energy and force RMSE, worst LOO ratio;
  - Q′, cold refit wall;
  - the evidence for each K.
- Test `learn_mix_is_deterministic_and_rank_capped`.

- [x] **Step 1:** write the tests; implement `learn_mix`; the tests pass. (Task 2a, 33f56405.)
- [x] **Step 2:** run the harness on a copy of the user's 5-element collection (copy from `%LOCALAPPDATA%\atomic-sandbox\models\collections\<key>`, never in place). (2026-11-24: K = 3/4 plus the rare-element probe; K = 6 is capped to S = 5, which is the species control.)
- [x] **Step 3, the GATE:** (2026-11-24: **GO** at K = 3 and K = 4. On this ≤ 3-atom collection W came out species-diagonal; see the references section's caveat.) GO if at K ≤ 4 the force RMSE is within 10 % of the species model's and the cold refit takes ≤ 25 % of its time. Write the numbers to `crates/validate/references.md` ("Chemical channels: the day-1 gate"). On NO-GO, stop and report to the controller.

### Task 3: fresh-collection format and switch (day 2)

**Files:**
- `crates/surrogate/src/collection.rs`: the key gains basis kind and K; `channels.txt` holds W, versioned; a record field for optional extra targets, versioned and empty for now;
- `crates/surrogate/src/sidecar.rs`: a new magic with K and W;
- `crates/providers/src/surrogate/shared.rs`: open, create and attach for the channel basis;
- app: Model tab ▸ **Model basis** (Species / Channels), as `ui::Action::ModelBasis` and a script word, with a new collection for each choice.

- [x] **Steps:** round-trip tests (`channels_txt_round_trips`, `a_channel_collection_has_its_own_key`), then the implementation, tests, commit. (2026-11-24: also `a_mismatched_species_set_is_refused` (review I5: rows keyed by Z), `a_sample_with_extra_targets_reads_as_the_same_sample` (flag bit 2, never written yet), `a_channel_model_sidecar_round_trips_its_mix` (sidecar version 11, `ASMOD011`), providers `a_channel_collection_attaches_as_its_own_and_fits_only_with_its_mix`, app `the_model_basis_switch_sends_its_command_and_has_a_script_word`. W is accepted from outside (`SurrogateKs::set_collection_mix`); learning it when the collection is ready, leaving rowless elements out of the fit and relearning — review I5's lifecycle — is Task 6's. A channel collection without a W gathers samples and forms no model.)

### Task 4: model hot path, fit, novelty, LOO, envelope (day 2)

**Files:**
- `crates/surrogate/src/model.rs`: the body-3/4 adjoint through W; variance support falling back to centre blocks;
- `fit.rs`: column classes are body-2 ties only, per-feature scales;
- `novelty.rs`, `loo.rs`, `grade.rs`;
- `envelope.rs` and `providers/src/surrogate/policy.rs`: the isolated-atom test and coordination use geometry or raw per-species counts (Review focus 2).

- [x] **Step 1, tests:**
  - `channel_model_forces_are_minus_energy_gradient` (finite differences, 1e-7);
  - `a_neighbour_with_zero_channel_sum_is_not_isolated`;
  - the identity-W fit equals the species fit (energies and forces to 1e-9).
  - (added) `channel_hot_path_matches_the_design_rows`, `channel_variance_matches_the_dense_posterior`, `the_channel_support_covers_every_scene_of_it`, `channel_loo_matches_a_downdate_of_the_fit`.
- [x] **Step 2:** implement until they pass. Run `cargo test -p surrogate --release`. Commit. (2026-11-24: the hot path pulls the body-3/4 adjoint back through W; the variance support takes the channels a composition reaches, K(C) — the centres' whole channel blocks for a dense W. fit, novelty, LOO and grade are generic in the column classes and needed no change; the envelope and `policy.rs` none, by the physics review's "Rules 5a/5b are unaffected". Species behaviour bit for bit.)

### Task 5: uncertainty guards (day 2; physics review before landing)

**Files:**
- `crates/providers/src/surrogate/trust.rs`: a per-element evidence floor N_trust, and the "missing W row" refusal through rule 3;
- `crates/surrogate/src/model.rs` and `fit.rs`: the linearised W variance gᵀΣ_W g in the predictive std (the form the physics review approves);
- constants N_W and N_trust, each with its reason.

**Replaced** by the spec's "Physics review 2, 2026-11-23: Task 5 replaced" (refuse rather than widen; no gᵀΣ_W g, no N_trust). Tasks 5 and 6 landed as one piece (2026-11-24; `providers/src/surrogate/represent.rs` module doc).

- [x] **Step 1, tests** (the review-2 list): `representation_status_is_a_flag_not_a_value` (surrogate `channels`), `channels_txt_round_trips` and `a_mismatched_species_set_is_refused` (surrogate `collection`, channels.txt version 2), and providers `tests/channels_life.rs`: `a_many_body_atom_with_an_unrepresented_neighbour_is_refused` (HOCl, H₂O·HCl, LiOH refused by rule 3b where 5d/5e pass at a geometry Task 4's fit holds; the excluded samples leave water's coefficients, report, n_k/q_k and prediction bit-identical), `pair_atoms_with_an_unrepresented_neighbour_still_answer` (HCl, LiH, ClO bit-identical), `a_rare_element_is_unrepresented_until_n_w_then_relearns` (S unrepresented with a zero row, refused, kept; relearnt at N_W with class memory cleared and `history_from` moved; then O{H, S} refused by its n_k = 0 floor). The two Σ_W error-bar tests are dropped (no Σ_W term).
- [x] **Step 2:** implemented: the represented flag (N_W ∧ ρ_t ≤ 0.01, NaN fails, stored by Z), W from represented species only (`MixSetup::eligible`; the raw mix stays for the gate), rule 3b (`TeachRule::UnrepresentedNeighbour`, history code 25, draft mode too, kept and thinned), `Q_UNREP` (out of the channel fit, in the low-order fit and N_W). Physics-reviewer pass: still due (not dispatched from the implementing agent).

**Physics review of Tasks 5–6 (2026-11-24, "land after fixes"; `scratchpad/task56-review.md` of the session):** fixed I1 (a learnt W is staged and committed only with its first successful fit; a background copy's W is taken only when its refit succeeded and its model is fitted on it; rule 3b refuses every many-body atom while the standing model's channels are not the committed mix's), I2 (a relearn that reproduces the same species, K, W bits and represented flags updates only the counts: no class-memory reset, no `history_from` move), M1 (the reader refuses a represented row with ρ > 0.01 or NaN), M2 (`set_collection_mix` sets `history_from`), M3 (attach moves `model_gen` past `history_from`), M4 (one reach for rule 3b, the exclusion and the census), M5 (relative QR pivot test), M6 (a failed learn is retried only when the N_W set changes or a count doubles). Left: M7 (no cancel token in `learn_mix`), M8 (the C2 shadow count describes every fitted sample in the species basis at each fit: measure in Task 7, or count structurally), M9 (a mix row for a purged species is refused rather than dropped; dropping it would let `with_channels` replace K > S by the identity silently, so it needs care).

- [x] **Blocker — review I3 / T7:** the gate rerun through the full decision list (below) must pass before the Channels default, any README or release mention of the channel basis, and the merge that closes this plan (Task 7).
  - **PASS, 2026-11-25** (`providers` `surrogate::tests::channels_decisions`; `references.md` "T7: through the decision list").
  - Full split: K = 3 and species answer the same 453 of 559 geometries, with 0 admitted-but-wrong atoms each.
    - W is learnt through the provider with the N_W mask: 54–57 s end to end, against 3 334 s for the species refit.
    - The z ratio is 1.002.
  - S at 5 %:
    - S fails N_W, so K_eff = 2.
    - **Rule 3b refuses all 60 held-out H₂S scenes (180 atoms).** No admitted atom has an unrepresented neighbour.
    - The channel model admits 0 wrong atoms, against the species model's 0.
  - S at 20 %: the channel model admits 0 wrong atoms. Per the coordinator's ruling the species fit is not run (36 GB, ~50 min), since 0 cannot be beaten.
  - Not shown: an unrepresented neighbour the envelope cannot also see (HOCl-type). That case is pinned only by the synthetic `channels_life` test.

**Not done here:** the gate rerun through the full decision list (review T7: admitted-but-wrong ≤ the species model's, no admitted many-body atom with an unrepresented neighbour, z within 1 + 1/√(2n)) — see the Task 5–6 report (run 2026-11-25: the blocker above). K is the collection key's; review I3's choice of K stays the gate's.

### Task 6: trust floors, new-species handling, sampling (day 2)

**Files:** providers `trust::key_column_counts`, `unseen_key_floor`, `fit_column_union`, `mod.rs` `fit_floor` / `grown_body4_degree`, `calibrate.rs`, `quarantine.rs:2314`, `sampling/mod.rs` `min_samples_for_trust_basis`, `teach.rs` `richer_basis`, `store.rs` / `shared.rs` dirtying on a new species (relearn W: Review focus 4). From Task 3: a channel collection starts without W and refuses every fit (`shared.rs` `channel_refusal`); this task learns W with `channels::learn_mix` once N_W is met and installs it through `SurrogateKs::set_collection_mix` (which writes `channels.txt`), leaves a rowless element's samples out of the fit instead of refusing it (review I5a), relearns on doubling, and resets class hysteresis on a relearn.

- [x] **Steps** (2026-11-24, with Task 5):
  - floors: C2 per key, `trust::key_column_counts_c2` = min(channel columns touched, shadow species basis columns touched) over fitted samples — the class pass (`key_q`), the whole-model floor (`fit_column_union`, `sample_floor`), `fit_floor` (so `grown_body4_degree` and the readiness), `richer_basis` (now keeps the mix) and the pass floor (`sampling::min_samples_for_trust_channels`, `Sampler::with_channel_k`); `unseen_key_floor` stays Q′; the no-model keep cap (quarantine.rs) is the channel basis's own Q′, unchanged in meaning. Test `c2_floors_keep_todays_floor_up_to_k_species`.
  - lifecycle (`represent` module doc): census of the low-order fit's samples at every refit start; learn at the first N_W, relearn when a species first meets N_W or a species that met it doubles its many-body samples; a new element enters with a zero row (`collection::mix_for`) without a relearn; before any W the fit uses one all-zero channel (a body-2 model; Task 3's "no fit without W" is gone); a relearn clears the class memory and sets `history_from`, and `calibrate::calibrate_from` leaves earlier-W points out. A background refit's copy learns, the install writes `channels.txt`. `a_new_species_relearns_the_mix_and_refits_from_scratch` is `a_rare_element_is_unrepresented_until_n_w_then_relearns`.
  - runs (2026-11-24): `-p surrogate` all pass; `-p providers` 640 pass, 1 known environmental failure (`ks::tests::the_level_gated_stall_stop_on_the_measured_cases`, also on main); `-p app --bin app -- ui:: worker::` 538 pass. The gate rerun through the decision list (review T7) is not run yet (Task 7).
  - commit.

### Task 7: preview, suites, merge (day 3)

- [x] **Step 0 (2026-11-25, after the T7 PASS):** Channels K = 3 is the default for new collections. `sim::ModelBasis::default()` is `Channels(CHANNELS_K)`; the UI state, the worker and the viewer's re-send of a non-default choice all follow it. The Model tab lists Channels first and keeps Species. The UI test pins the new default and its reason. Replays, screenshots and `Scene::preset` paths run without a collections folder, so the default does not reach them. The provider's own `collection_basis` still starts at Species for the providers and validate tests. No old collection is migrated or deleted.
- [ ] **Step 1:** preview build: a fresh channel collection, plus a copy of the user's real collection opened as a channel collection for the comparison.
- [x] **Step 2:** `validate --test surrogate` (the key trust tests; the stuck-escape failure is pre-existing); the full app and full providers runs.
  - Runs of 2026-11-25, with the Channels default and main's view overlay zones merged in, at BelowNormal and `-j 4`, one heavy job at a time:
    - `-p app`: 858 passed, 0 failed, 4 ignored, plus `surface_reconfigure` 1 passed. 17.5 min. The run before the merge, on the flip alone, gave 847 / 0.
    - `-p providers`: 642 passed, 1 failed, 34 ignored, in 10.6 min. The failure is the known environmental `ks::tests::the_level_gated_stall_stop_on_the_measured_cases`.
    - `validate --test surrogate`: 10 passed, 1 failed, 5 ignored, in 27 min. The failure is the pre-existing `the_stuck_escape_repro_is_answered_honestly`: refused steps at rms 0.411 of the bound, as on main.
  - Review M8 (the C2 shadow count's cost per refit) is not measured separately. No existing log times it; it is inside T7's 54–57 s end-to-end channel refit.
- [ ] **Step 3:** README, DEVLOG and dev-reference updated (done 2026-11-25, with the default flip); merge; push. Merge and push are the coordinator's.
