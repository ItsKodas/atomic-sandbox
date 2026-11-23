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

- [ ] **Steps:** round-trip tests (`channels_txt_round_trips`, `a_channel_collection_has_its_own_key`), then the implementation, tests, commit.

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

- [ ] **Step 1, tests:**
  - `a_new_element_is_refused_until_it_has_enough_samples`;
  - `rare_element_error_bars_widen` (σ for an element with few samples ≥ σ for the same geometry type with many);
  - `well_sampled_error_bars_are_unchanged_within_5_percent`.
- [ ] **Step 2:** implement; physics-reviewer pass; commit.

### Task 6: trust floors, new-species handling, sampling (day 2)

**Files:** providers `trust::key_column_counts`, `unseen_key_floor`, `fit_column_union`, `mod.rs` `fit_floor` / `grown_body4_degree`, `calibrate.rs`, `quarantine.rs:2314`, `sampling/mod.rs` `min_samples_for_trust_basis`, `teach.rs` `richer_basis`, `store.rs` / `shared.rs` dirtying on a new species (relearn W: Review focus 4).

- [ ] **Steps:**
  - check each floor's rationale against channel Q′ and fix the reasoning text;
  - test `a_new_species_relearns_the_mix_and_refits_from_scratch`;
  - shortened providers and app runs (end of day 2);
  - commit.

### Task 7: preview, suites, merge (day 3)

- [ ] **Step 1:** preview build: a fresh channel collection, plus a copy of the user's real collection opened as a channel collection for the comparison.
- [ ] **Step 2:** `validate --test surrogate` (the key trust tests; the stuck-escape failure is pre-existing); the full app and full providers runs.
- [ ] **Step 3:** README, DEVLOG and dev-reference updated; merge; push.
