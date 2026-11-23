# Physics/trust review: chemical channels Tasks 5+6 (refuse rather than widen)

Branch worktree-agent-a34cdf545910cd8b1 (commit 24b2c9ee), worktree D:/Projects/Physics/.claude/worktrees/agent-a34cdf545910cd8b1.
Binding text: spec "Physics review 2, 2026-11-23" items 1-10, C2, I5; review R1-R8.
Paths below are relative to the worktree. No tests run; read-only review.

**Verdict: land after fixes.** The anti-goal holds on every path I traced, with two independent guards (rule 3b, and the n_k = 0 key floor over fitted samples). One path, a fit that fails after a relearn, leaves only the second guard (I1). The species basis is untouched.

## What I verified (sound)

- **Item 1, no g^T Sigma_W g.** Nothing adds Sigma_W to a std. Sigma_W and eta are computed in `represented_mix` and written as diagnostics only.
- **Item 2, the represented flag** (`surrogate/src/channels.rs` `is_represented`, `represented_mix`).
  - It is N_W AND rho <= 0.01, with NaN failing (`rho <= RHO_MAX` is false for NaN).
  - The rho formula is right. Re-derived: row t of M~ minus its rank-K truncation has squared norm sum_{j>=K} U[t,j]^2 lambda_j, and G_tt = sum_j U[t,j]^2 lambda_j, so rho_t is that sum over G_tt.
  - 0.01 = 0.1^2 follows from the 10 % relative force bound.
  - The flag is stored by Z in `channels.txt` v2 and never read from W (`LearntMix::represents` reads the flag).
  - The reader is strict in two ways: an unrepresented row must be exactly 0, and represented requires N_W.
- **Item 3, W from represented species only.**
  - Rows failing N_W are zeroed before the Gram.
  - rho-failing rows are zeroed after the SVD and before the QR. This is the single pass I asked for in R2.
  - Unrepresented rows are set to exact +0.0 after the sign flip.
  - W = I on the represented species when K_eff covers them, and with every species represented it is the raw W = I.
- **Item 4, rule 3b** (`policy.rs` `gate`, `represent.rs` `unrepresented_many_body`).
  - It fires for an atom with >= 2 neighbours within r_cut and an unrepresented species among them. It uses the same `neighbour_counts` / `neighbour_species` that build EnvKey and the class.
  - It sits after rules 2/3 and before rule 4, with `untrusted: false`. In `evaluate` the draft branch only matches `untrusted: true`, so draft mode goes to the teacher (verified at mod.rs:3078-3094).
  - The sample is kept, thinned, never a copy.
  - `would_answer`, `first_refusal`, inspect and the atom verdicts all pass `pos`. The pair curves pass `&[]` (dimers, correct).
  - The only `predict` calls that do not go through the gate are audits and bookkeeping, none of them answers.
  - It is complete for forces: E_i of a pair atom with an unrepresented neighbour is body 2. Any atom whose E_i holds a 3/4-body term involving X has >= 2 neighbours including X, so it is refused. Second-shell X never enters E_i.
- **Item 5, Q_UNREP.**
  - It is set and cleared in `fit_store_with` before every fit (trust.rs:2133), so recovery trials, votes, releases, held-out promotions and background copies all pass through it.
  - The incremental path adds only Q_NONE samples. A change below the prefix forces a rebuild.
  - `excluded()` covers it, so the envelope, the LOO, key counts, `fit_column_union` and `fit_floor` all leave it out.
  - It is never persisted: `persisted_code`, `admit`, `sync_collection_status` and `write_status_now` all map it to Q_NONE.
  - The census and the low-order learn take Q_NONE + Q_UNREP.
  - T3 pins water's coefficients bit for bit.
- **Item 6 / C2.** `key_column_counts_c2` = per-key min(channel touched, shadow species touched) over fitted samples, with union = min of the unions. It is applied at:
  - the class pass `key_q` (and the misfit `all_fitted`);
  - `fit_column_union` / `sample_floor`;
  - `fit_floor` (so `grown_body4_degree` and readiness);
  - `richer_basis` (which now carries the mix, with W in its cache key);
  - the pass floor `min_samples_for_trust_channels`.

  `unseen_key_floor` stays Q'. I found no remaining non-C2 trust count (grep of `key_column_counts(`, `column_classes()`, `n_independent_columns()`). The no-model keep cap uses Q', which is correct: it is a cap, not a trust floor.
- **Item 8, evidence.**
  - `history_from = model_gen + 1`. A successful synchronous fit makes model_gen exactly that.
  - In a background copy: `f.model_gen + 1`, the copy's fit bumps it, and the install takes max(live, copy). Points the live provider recorded during the run carry `snap.model_gen`, which is below history_from, so they are excluded.
  - `calibrate_from(.., history_floor())` is used at both judge sites (trust.rs:878, 1181). Those are the only providers callers, so counts, windows and every lambda drop old-W points.
  - `install_mix` clears class memory and the floor cache, and `commit_class_memory` then records the ENTER verdicts.
- **Species basis untouched.** Every new path is behind one of these checks:
  - `channel_collection()` returns None;
  - `basis.channels().is_none()`;
  - `channel_k` is None;
  - `history_floor()` is 0 (`calibrate_from(..,0)` filters `>= 0`).

  `save_mix` refuses a species collection before `set_collection_mix` bumps anything. The `richer_basis` hash adds `None`, which changes a cache key value only. The claim that the suites pass unchanged is consistent with the code.

## Rulings on the implementer's concerns

1. **Body-2 model on one all-zero channel before any W: acceptable and honest.**
   - With W = 0, `add_neighbour` contributes nothing to bodies 3/4. The model is exactly a body-2 model.
   - `represents()` is false for every species, so every sample with a many-body atom is Q_UNREP and every many-body scene is refused by 3b.
   - The dead columns only raise Q' (unseen-key floor, no-model cap): the conservative side. They add nothing to any prediction or variance, because their features are 0 everywhere.
   - It also removes Task 3's dead end: a dimer collection could never form a model before.
   - Liveness: the census and the learn run at the start of every refit, model or no model. A many-body-only collection learns before its first fit once N_W is met.
2. **The doubling trigger includes N_W-met, rho-failing species: correct.** I5c names the element's informative count. Without it a rho-failed species could never become represented. See I2 for the cost of each relearn.
3. **The exclusion checks every atom, including capped-segment edge atoms: correct and conservative.** An edge atom's E_i enters interior forces. It matches R4's "the whole sample".
4. **3b answers thinned like rule 5: acceptable.**
   - Distinct geometries are what N_W needs, and the census measures N_W in the species low-order basis.
   - Caveat: thinning distance is measured in the channel basis, which sees the unrepresented neighbour only through body 2. Pair distances still change with any geometric change in small molecules. At worst this slows N_W; it cannot make an answer.
5. **K from the key, no I3 choice in the provider: fine for this piece.** It must be decided before Channels becomes the default.
6. **Census cost per refit:** a describe pass in the small I4 basis plus O(atoms^2) per sample. Acceptable, but measure it in Task 7. See M8 for the larger cost: the C2 shadow count at every fit.
8. **`set_collection_mix` bumps model_gen: correct.** It is a new basis, so a background refit on the old W must be discarded, and calibrate already treats generation bumps that are not fits via `fit_id`. It should also set `history_from` (M2).

## Important

**I1. A relearn whose first fit fails, or a background copy installed with an error, leaves the new W's represented flags governing rule 3b while the old-W model stands.**

- Where:
  - `crates/providers/src/surrogate/represent.rs:335-366`: `install_mix` sets `sh.mix` and writes `channels.txt` before any fit.
  - `crates/providers/src/surrogate/trust.rs:1500-1501`: `channel_lifecycle()` then `fit_store_deferred()?` returns on error, and the refit contract keeps the previous model.
  - `crates/providers/src/surrogate/background.rs:459-471`: `install_fork` is called with `ok = false` too (background.rs:363) and copies `copy.mix` regardless. A copy whose fit errored still holds the old model.
- Effect: `represents()` (represent.rs:166) reads `sh.mix`, not the model's W.
  - Scenario: species X is unrepresented in the old W and represented in the new one.
  - 3b stops refusing HOCl-type scenes, but the standing model has X's row at zero and fitted none of those samples.
  - What stops the confident wrong answer is only the n_k = 0 floor: every class-2 key containing X had n_k = 0 in the old fit, because all such samples were Q_UNREP.
  - That floor does hold, so this is not Critical. But it removes the first of the two guards that spec item 7 relies on.
  - Separately, `channels.txt` on disk now names a W that no model was fitted with.
- Fix (either):
  - (a) Stage the learnt mix. Keep the old `sh.mix`, fit with the candidate (e.g. pass it to `fit_basis` / `default_basis` through a pending field), and commit `sh.mix`, `save_mix`, `history_from` and the class-memory clear only after `fit_store_deferred` succeeds. On error, drop the candidate and set `learn_failed`. In `install_fork`, take `copy.mix` only when `ok` and the copy model's `basis.channels()` equals `mix_for(copy.mix, species)`.
  - (b) As belt and braces regardless: in `unrepresented_neighbour`, when the standing model's `basis.channels()` differs from `collection_mix_for(&model.basis.species)`, treat every species as unrepresented, so every many-body atom is refused.
- Test to add: plant a mix that represents Cl, force the first fit to fail (e.g. the existing failure seam), and assert HOCl is still refused by 3b and that `channels.txt` is unchanged.

**I2. Every relearn resets trust even when it reproduces the same W.**

- Where: `crates/providers/src/surrogate/represent.rs:111-133` (the trigger), 335 (`history_from`), 362 (class memory).
- The doubling trigger fires each time any N_W-met species' many-body count doubles. H's doubles whenever the collection doubles.
- On the gate's data W is species-diagonal, so each such relearn returns W = I on the same represented set: a bit-identical basis.
- Yet `history_from` moves and the class memory clears. All calibrated evidence (earned with DFT audits) is discarded at 2, 4, 8 times the collection size, and every key goes back to ENTER, for a model in exactly the basis that was measured.
- My item 8 meant "a new W generation". A refit in an unchanged basis is an ordinary refit, and ordinary refits keep their evidence.
- This also biases T7's trust-outcome comparison against the channel model.
- Fix: in `channel_lifecycle`, when the learnt (species, k, W bits, represented) equal the standing mix's, update only `many_body` / `n_w_met` (and `channels.txt`), and keep `history_from`, class memory and the floor cache.

**I3. T7 (the gate rerun through the full decision list) is not run, and the Channels choice is already user-visible** (`crates/app/src/ui/dock/model.rs:1268`).

- Spec item 10 makes it a required test:
  - admitted-but-wrong atoms <= the species model's;
  - zero admitted many-body atoms with an unrepresented neighbour;
  - z within 1 + 1/sqrt(2n).
- The unit tests (T1-T6) pin the mechanisms on a synthetic Morse teacher, not the outcome on real DFT data.
- Also, as the report notes, `channels_gate.rs` still learns the raw mix (`eligible: None`). It must pass the census's N_W mask, or it measures a mix the provider never installs.
- Acceptable to land this piece with T7 as a hard blocker of Task 7: before the Channels default, any README/release mention, and the merge that closes the feature.

## Minor

- **M1** `crates/surrogate/src/collection.rs:501`: the reader does not enforce that represented implies rho finite and <= RHO_MAX. A hand-edited or raw-mix file can flag a rho = 1 row as represented. Add the check. The tests that write `gate_mix(true)` model Task 4's semantics and never read it back, so they can mark that mix raw.
- **M2** `crates/providers/src/surrogate/shared.rs:498-507`: `set_collection_mix` bumps model_gen and clears class memory but leaves the given `history_from`, which is 0 in the tests. After an external mix, old-W points keep counting. Set the mix's `history_from` to the bumped model_gen before `save_mix`.
- **M3** `crates/providers/src/surrogate/shared.rs:331`: at attach, model_gen = max(own, newest point) + 1 ignores the `history_from` in `channels.txt`. If model_gen was bumped without audit points before the learn (policy edits, restores), the new session's new-W points can carry gens below `history_from` and be excluded for several refits. That is conservative but stalls evidence. Take the max with `history_from` too.
- **M4** `crates/providers/src/surrogate/represent.rs:175` (3b: the model's r_cut) vs 217 and 266 (exclusion and census: `collection_r_cut`). They are equal today. Use one source, so a future Model-reach setting cannot let the exclusion see fewer neighbours than the query rule.
- **M5** `crates/surrogate/src/channels.rs:692`: the QR rank test is r_cc > 0. When the zeroed rho-failing rows carried a channel, a near-dependent B_rep passes with a noise column. That is harmless for capacity (a column gauge) but ill-conditioned. Use a relative threshold (r_cc > S * eps * max r) and return `DegenerateGap` below it.
- **M6** `crates/providers/src/surrogate/represent.rs:315`: `learn_failed` compares the whole census, which changes with almost every new sample. A learn that keeps failing (about 60 s at S = 5) is retried at nearly every refit. Retry only when the N_W set changes or a count doubles.
- **M7** `learn_mix` has no cancel token. A background copy's cancel waits for the low-order fit, and a synchronous refit blocks for it. Interactivity (rule 2) is affected, not physics.
- **M8** `crates/providers/src/surrogate/trust.rs:354`: the C2 shadow count describes every fitted sample in the species basis at every fit (`fit_column_union` runs in each deferred fit). At S = 5 with body 4 that is the species model's full describe cost, several times per refit. q_sp could be computed structurally from each atom's neighbour-species set, or cached per sample. Measure it in Task 7.
- **M9** `crates/surrogate/src/collection.rs` `mix_for`: a mix row for a species the fit lacks is refused, and the relearn needs some N_W. If a species is purged and nothing meets N_W, the collection cannot fit. Dropping the extra row is exact (an absent species changes no feature), so prefer that over a refusal.

## Not checked

- T7 on the 5-element collection copy: not run. It is too heavy, and I was told not to run tests.
- Wall-clock cost of the census and the C2 shadow counts on a large collection: not measured.
- Per-species row stability of W across fold relearns (my review's open item).
- Behaviour with a read-only second session on the same collection while the writer relearns. I traced only that `save_mix` refuses read-only handles, so their W stays session-only and their points are not written.
