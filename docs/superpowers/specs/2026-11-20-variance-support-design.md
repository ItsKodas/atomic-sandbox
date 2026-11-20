# The predictive variance on the scene's support (design)

**Status:** approved by the user on 2026-11-20 ("Let's do it, and if you think it will fix it, implement it"). **Amended 2026-11-20** after the implementation and the physics review (rounds 1 and 2, the coordinator's rulings): the support is keyed by species composition, the factor is built in double-double and kept as R⁻¹, every build is self-checked, `describe` skips absent channel species, and the accuracy statement and the timing are the measured ones. The original text is in git history (b883c1c0).

## Why

Large shared models are slow in Estimated mode even on small scenes. The measurement, on a copy of the user's BEST collection with an H₂ geometry, 2000 calls each:

| Model | Species | Q′ (independent columns) | `predict` | `predict` with variance |
|---|---|---|---|---|
| BEST (H, O, P) | 3 | 1902 | 32 µs | **729 µs** |
| `models/hydrogen.model` | 1 | 29 | 1.8 µs | 5.1 µs |

The variance (`Model::predictive_std`, `crates/surrogate/src/model.rs`) computes 1 + 3n quadratic forms vᵀA⁻¹v against the full Q′ × Q′ Cholesky factor L (`sigma_inv_chol`). Its cost is (1 + 3n)·Q′²/2 per call, whatever the scene contains. An H₂ scene's query vectors, though, are non-zero on only a few dozen of the 1902 columns: the H-centre features with H neighbours.

## The rule (exact)

Let S be a set of reduced columns that contains the support of every query vector v of the call. Then

vᵀA⁻¹v = v_Sᵀ (A⁻¹)_SS v_S = |R v_S|²

where X = L⁻¹E_S (Q′ × |S|) and XᵀX = (A⁻¹)_SS = RᵀR, R upper triangular. The value is the same quantity, a sum of squares as today, at a cost of |S|²/2 per vector instead of Q′²/2.

**How R is applied (amended).** With W = R⁻¹ and J the reversal of S, F = J W J is lower triangular and is the Cholesky factor of J C J, C = A_SS − A_SN A_NN⁻¹ A_NS: the trailing block of the Cholesky factor of A ordered (N, reversed S). The forms are ‖F⁻¹ J v_S‖², the full path's own forward substitution through a reordered exact factor of the same A (its N block meets a zero right-hand side). F is computed in **double-double** (X, XᵀX, its Cholesky factor and the inverse; Dekker/Knuth error-free transformations, no FMA, bitwise deterministic for any thread count) and rounded once to f64.

**Why not the f64 route first written here.** Measured on fitted factors against a double-double reference, ‖R v_S‖² with R from the f64 Householder QR of X is off by up to 3.5e-11 relative, where the full substitution is ≤ 2.2e-13; f64 eliminations of L to C's factor are off by up to 1.8e-10 (Householder) and 5.9e-12 (Givens). The cause is cancellation: the unit columns L⁻¹e_s carry the prior's variance (of order 1/λ), while a query in a direction the data determine has a variance orders of magnitude smaller.

**Accuracy (amended).** The support path is in the same accuracy class as the full path: both are measured against a double-double reference on the BEST fit, and every build is guarded by the 1e-10 self-check (below). On the BEST fit, over every form of 1 000 seeded H₂/H₃ geometries:

| | max relative error vs double-double | median |
|---|---|---|
| support path | 3.18e-12 | 4.33e-13 |
| full path | 3.26e-12 | 4.21e-13 |

The two paths can therefore differ by about the sum of their errors: 1.93e-12 on the stds, 4.25e-12 on the forms on BEST. The unit tests keep the 1e-12 relative + 1e-15 pin on the stds on their own fixtures (measured 4.4e-13); it was not loosened. It is not bit for bit today's value.

## Design

- **Where:** `surrogate::model`, `surrogate::linalg` and `surrogate::descriptor`. `predictive_std` keeps its signature. Outside the crate, only the providers' tests and the would-answer doc changed.
- **Support (amended): keyed by species composition.**
  - For the scene's species set C, S(C) is every reduced column whose centre species and channel species all lie in C:
    - body-2: the neighbour's species;
    - body-3: both channels';
    - body-4: all three.
  - A feature of an absent channel species is an empty sum (or a product with one): exactly +0.0, as is its gradient. So S(C) contains every query vector's support.
  - On the BEST basis: |S({H})| = 52, any pair 468, all three 1902.
  - The query vectors are accumulated straight onto S(C). A non-zero landing outside S(C) is impossible by that argument; if one ever occurs, the call takes the full path, with a `debug_assert` and a once-per-entry stderr note and counter.
- **Cache (amended): one entry per composition.** It lives in the `Model` beside `representatives`, behind a `Mutex`: a map from species bitmask to the factor F or "full path", so at most 2^S − 1 entries.
  - **Determinism.** The entry's path and every std are a pure function of (L, geometry, species set), whatever the call order, thread or clone. The reasons:
    - S(C) depends on C alone;
    - the build is deterministic;
    - the self-check uses a canonical cluster of the composition, not the triggering call.
  - **A call with no neighbour inside r_cut** (all query vectors zero) answers +0.0 forms, exactly as the full path does, and settles nothing.
  - **Clone** copies the entries. Equality ignores them.
  - **Stale L.** A fingerprint of L (its size, last row and 32 diagonal entries) is checked on every call and empties the cache when L changes in place. The doc still says to replace the cell then.
  - **A refit** makes a new `Model`.
- **When the full path stays:**
  - |S(C)| > Q′/2;
  - `sigma_inv_chol` is `None` (no variance, as today);
  - L has a non-finite entry or a zero pivot (checked once per model, only before a build);
  - a build fails;
  - the self-check fails.
- **Self-check (added).** Every new factor is compared with the full path on every form of its composition's canonical cluster: one atom of each species and one more, all within r_cut. It is used only if they agree within 1e-10 relative (NaN fails) and at least one form is non-zero. Otherwise the composition takes the full path for good. The largest difference seen is kept, and a fallback is reported once on stderr.
- **`describe` (added).** Body-3/4 terms with a channel species absent from the scene are skipped. Their value and gradients are the +0.0 they would compute, bit for bit for finite positions. E and F are not derived from `describe`.
- **Threads.** The model is shared with the auditor's thread.
  - The lock is held only to read or insert an entry, never during a build. No lock is held across rayon.
  - Racing builders insert the same bits.
  - A poisoned lock is recovered.

## Tests (TDD)

- **`support_variance_matches_the_full_path`:** seeded random and fitted models (several species, body orders up to 3) and seeded scenes. energy_std, force_std and force_std_epistemic each agree with the full path within 1e-12 relative plus 1e-15 absolute.
- **Composition tests** (amended; they replace the growth test):
  - `one_factor_per_composition`;
  - `the_composition_support_covers_every_scene_of_it`: sizes on the BEST basis, and every geometric support inside S(C), with body-2, body-3 and body-4 models;
  - `stds_do_not_depend_on_the_call_history`: bit-identical across call orders and clones;
  - `concurrent_calls_agree_with_the_full_path`.
- **Cache-lifetime and fallback tests:**
  - `a_new_model_starts_with_an_empty_cache`;
  - `large_support_uses_the_full_path`: bit-identical;
  - `a_factor_changed_in_place_drops_the_stale_entries`;
  - `an_unusable_factor_takes_the_full_path`;
  - `a_failed_self_check_takes_the_full_path`;
  - `a_call_without_neighbours_settles_nothing`;
  - `a_nan_difference_fails_the_self_check`.
- **`linalg` tests:**
  - the double-double arithmetic;
  - `support_factor_gives_the_full_quadratic_forms`;
  - `support_factor_keeps_its_accuracy_where_the_products_cancel`;
  - `describe`'s `the_composition_skip_changes_no_bit`.
- **Providers:** `support_decisions`. A two-species session in which the support path is taken decides identically with the support path forced on and forced off.
- **The existing pinned tests stay green:** `cargo test -p surrogate --release`, `-p providers` and `-p sim`. Every decision is unchanged, including the scripted golden session (`a_scripted_session_decides_exactly_as_before`). No std pin moved.
- **Measurements** (in `crates/validate/references.md`, not gates):
  - the H₂-on-BEST timing, `tests/variance_timing.rs`;
  - the accuracy comparison above, `support_accuracy_on_a_model_file`.

## Measured result (amended)

H₂ on the BEST fit, 2000 calls each, |S({H})| = 52:

| | before | after |
|---|---|---|
| `predict` with variance | 679 µs | **66–76 µs** |
| `predict` | 50 µs | 38–44 µs (unchanged code) |
| first call (L checked, {H} built and self-checked) | | 13–18 ms |

**The target (with variance ≤ 60 µs) is missed.** The rest of a variance call is two things:
- `energy_and_forces`: ~40 µs, a non-goal below;
- `describe` with gradients: 17–24 µs after the composition skip.

Those two alone are about 60 µs. The variance's own forms now take about 6–10 µs.

## Non-goals

- Fallbacks in trained areas on the BEST model. They come from the H₂ vs H₃/H₄ fit conflict, a separate design.
- The plain `predict` at 32 µs, a descriptor cost that is not the bottleneck.
- No per-element data and no change to any trust rule or bound.
