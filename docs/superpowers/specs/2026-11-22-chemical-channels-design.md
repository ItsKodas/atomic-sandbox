# Chemical channels for the surrogate — design

**Approved direction:** user, 2026-11-22 ("this plan sounds great", "This model problem is now the most important thing"). Rule 1's exception was signed off the same day (CLAUDE.md). Research: `docs/superpowers/research/2026-11-22-chemical-channels-literature.md`.

## Why

The linear ACE surrogate has one column block per element combination. Q′ went from 6 656 at 4 elements to 14 670 at 5 (H, Li, O, S, Cl), and a cold refit of the user's collection took 1 869 s (harness, measured). Each element multiplies refit time by roughly 10–20×, and memory (Q′² × 8 B per copy) runs out at 7–8 elements. The goal is a model whose cost grows gently with the element count, for many-element and molecules-with-molecules simulation.

## Goals

- Many-body terms built from K learnt chemical channels instead of neighbour species, so P₃ and P₄ depend on K, not S.
- The same accuracy as the species basis on the user's collection at K ≤ 4 (the day-1 gate below).
- An exact linear Bayesian posterior, as now, with uncertainty that stays honest for elements with few samples.
- Every trust mechanism (audits, fallbacks, novelty, grade, classes, floors, stray atoms) working on the new model.
- A sample format with room for optional extra targets later. That part only: no bonds or density work.

## Non-goals and anti-goals

- **No migration or compatibility with old collections** (the user: "please do not ever care about them"). The channel model starts fresh collections; the old ones sit unused and untouched.
- **Not flat in the centre element yet.** Per-centre coefficient blocks stay, so Q′ ≈ S·P(K) grows linearly in S and cost grows as S³. Putting centres through channels, or exploiting the loose coupling between centre blocks, is a later step.
- **Anti-goal: a confident wrong answer.** A frozen mix must never make the model trust itself for an element it has barely seen.

## Constraints

- **Rule 1 (signed-off exception):** W, the per-element channel weights, is learnt only from our own DFT samples, never from tabulated properties. No electronegativity initialisation.
- Atomic units; f64 for the fit and the model on the CPU, as now.
- The tolerances stay as they are. The trust floors keep their meaning.

## The model

- **Body 2 is unchanged:** a full block per (centre, neighbour) species pair, with its tie classes. It is small (n_max·S(S+1)/2) and carries most of the pair physics.
- **Bodies 3 and 4 use channels.** The neighbour density is mixed per channel, Ā_k(n, l, m) = Σ_t W[t, k]·A_t(n, l, m), and the body-3/4 products are taken over channel indices (k, n, l) instead of (t, n, l). This uses full channel products: K is small, so O(K^ν) is fine, and the paper's within-channel O(K) form is not needed.
- **Per-centre blocks stay:** coefficients are c[s][feature], where the features are channel features.
- **Q′** = n_max·S(S+1)/2 + S·(P₃(K) + P₄(K)). At K = 4 that is about today's 4-element per-centre size (an estimate; measure it).

## Learning W (then frozen)

1. Fit a **low-order full-species model** (body 2 + body 3 on species, a cheap Q′) on the collection's samples.
2. Take its body-3 coefficient tensor and unfold it along one neighbour-species mode. The rows are neighbour species t; the columns are everything else (centre s, partner t′, n, n′, l), symmetrised over the two neighbour slots.
3. W = the top-K left singular vectors of that matrix, scaled by their singular values: a mode-1 Tucker or SVD reduction.
4. Fix each column's sign so its largest entry is positive, for determinism.
5. Choose **K by evidence:** fit the channel model at K ∈ {3, 4, 6} and keep the smallest K within the evidence (or LOO) margin of the best. Record the choice.
6. **Freeze W** and store it in the collection as a versioned `channels.txt`. Every refit uses it until it is relearnt.

**When W is relearnt:** when the collection's species set changes, or on the user's Refit-all. A relearn recomputes every sample's features and fits from scratch. A Gauss–Newton refinement of W is a later option.

**A new element** has no W row until it has at least N_W samples. That is the per-element guard, a constant with its reason. Until then:
- scenes containing it are refused by the model (rule 3, as an unknown species is today) and go to DFT;
- once enough samples exist, W is relearnt with the new row.

## Uncertainty with a frozen W (the risky part; physics review)

1. **The per-element guard** above, plus a **per-element evidence floor** in trust: an element with fewer than N_trust fitted samples keeps every key that contains it on DFT, whatever the posterior says.
2. **Linearised W uncertainty (Laplace):** the predictive variance gains gᵀ Σ_W g, where g = ∂E/∂W at the frozen W and Σ_W comes from the SVD fit's residual (the first version may use a conservative scalar multiple of the identity per row, scaled by 1/n_samples(element)). The physics review rules on the exact form. It must widen error bars for rare elements and add almost nothing for well-sampled ones.

## What changes in the code (from the codebase map)

- **`surrogate::descriptor`:**
  - `Basis` gains `channels: Option<ChannelMix { k, w: Vec<f64> /* S×K */ }>`;
  - body 3 and body 4 are indexed by channel;
  - `add_neighbour` spreads into K channels with weights W[z_j];
  - the composition skips (`present`, `active3`, `active4`) go for bodies 3 and 4;
  - `touched_columns` covers the centre block;
  - `feature_names` prints channels.
- **`surrogate::model`:** the hot-path pair sweep is unchanged for body 2. The body-3/4 adjoint takes the neighbour gradient through W. The variance-support cache falls back to centre blocks (a speed loss, accepted).
- **`surrogate::fit`, `novelty`, `loo`, `grade`:** column classes are body-2 ties only. Scales are per feature. These are otherwise generic in Q′.
- **`surrogate::envelope`, `policy`:** with signed W, the isolated-atom test and the coordination ratio use the geometric test (`energy_row_atoms_at`) or raw per-species counts, never channel values.
- **sidecar/collection:**
  - a new collection key component (basis kind and K), so collections start fresh;
  - a new sidecar magic;
  - `channels.txt`.
- **providers `trust`, `calibrate`, `quarantine`, `sampling`, `teach`:** the sample floors are recomputed from channel Q′ (the code is data-driven; check each one's rationale). New-species handling as above.
- **app:** Model tab ▸ **Model basis: Species / Channels (K)**, default Channels for new collections once it is gated. Recorded as a `ui::Action`.
- **Unaffected** (geometry or Z labels only): the knowledge map, pair and triple coverage, stray signatures, `EnvKey`, history, `ref_energy` per element.

## Deleted with it (no compatibility)

The FD migration (`surrogate/migration.rs`, `providers/sampling/migrate.rs`, the app's migration UI and test), the old sidecar magics and body versions, pre-S9 references, calibration legacy points and collection record v1 reading. The exact list is in the codebase map. Prep task 0 does this first.

## Gates

- **Identity test:** W = I with K = S reproduces the species basis's features, energies and forces to rounding (1e-10 relative). This is the correctness anchor for the channel code.
- **Day-1 go / no-go**, on a copy of the user's 5-element collection, the channel model at K = 3/4/6 against the species model:
  - force and energy RMSE and LOO ratios;
  - Q′ and cold refit time.

  **GO** if at K ≤ 4 the force RMSE is within 10 % of the species model's and the cold refit takes ≤ 25 % of its time. Otherwise report and stop.
- **Uncertainty sanity:** on held-out samples of an under-sampled element, the predicted σ covers the error (z-scores within the class calibration bounds). The sample-count guard is exercised.
- **Suites** (trimmed; qm, qm-gpu and the integrator are untouched):
  - surrogate unit tests and the new checks on every change;
  - providers (surrogate and sampling) and app (`ui::` and the worker model tests) at the end of day 2;
  - validate surrogate (the key trust tests), one full app run and one full providers run before merging.

## Approaches considered

1. **Neighbour channels with per-centre blocks and W from an SVD of a low-order fit (chosen).** It is supported by the literature, linear-algebra only, and keeps the exact posterior.
2. **Random fixed W (the paper's linear form):** needs a large K for accuracy, which defeats the size goal.
3. **Joint non-linear learning of W:** loses the exact posterior, and the trust system depends on it.
4. **Centre channels too:** flat in S, but untested here. A later step.
