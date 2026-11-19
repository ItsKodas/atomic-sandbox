# The predictive variance on the scene's support (design)

**Status:** approved by the user on 2026-11-20 ("Let's do it, and if you think it will fix it, implement it").

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

where X = L⁻¹E_S (Q′ × |S|, from |S| forward substitutions of unit vectors) and X = QR is its thin QR factorisation (Householder; R is |S| × |S| upper triangular). The value is the same quantity, a sum of squares as today, at a cost of |S|² per vector instead of Q′²/2. It is not bit for bit today's value: it agrees to rounding, with a relative error of about 1e-13.

## Design

- **Where:** `surrogate::model` and `surrogate::linalg`. `predictive_std` keeps its signature, and nothing outside the crate changes.
- **Support:** the union of the reduced query vectors' non-zero indices. Values that are exactly 0.0 are off-support, so restricting is exact.
- **Cache:** in the `Model`, beside `representatives`, behind a `Mutex`. It is derived data, so `Clone` shares or rebuilds it, and the doc says to replace the cell if `sigma_inv_chol` changes in place. The cache holds the current support S and its R.
  - If a call's support ⊆ S, it uses R.
  - If not, S grows to the union and R is rebuilt, so S grows monotonically to the composition's reachable set instead of churning as neighbours move in and out.
  - A refit makes a new `Model`, so the cache never outlives its L.
- **When the full path stays:** when |S| > Q′/2 (the support gains nothing), and when `sigma_inv_chol` is `None` (no variance, as today).
- **Threads:** the model is shared with the auditor's thread, so the cache must be `Sync`. A rebuild under contention must not deadlock or give a torn R.

## Tests (TDD)

- `support_variance_matches_the_full_path`:
  - on seeded random models (several species, body orders up to 3) and seeded scenes;
  - energy_std, force_std and force_std_epistemic each agree with the full path within 1e-12 relative plus 1e-15 absolute.
- `the_support_grows_and_never_shrinks`: a scene of H then H with O nearby.
  - The second call's support is a strict superset and the values still match.
  - A third call on plain H reuses the grown R without a rebuild (counted).
- `a_new_model_starts_with_an_empty_cache`: a clone after a refit is the case named.
- `large_support_uses_the_full_path`: |S| > Q′/2 uses the full path and gives bit-identical values.
- The existing pinned tests stay green: `cargo test -p surrogate --release` and `cargo test -p providers --release`. Every decision is unchanged, including the scripted golden session (`a_scripted_session_decides_exactly_as_before`). A test that pins an std bit for bit and moves at the 1e-15 level is re-pinned with the reason "variance on the support (spec 2026-11-20), rounding only", and the report lists each one.
- **Measurement** (in the report, not a gate): the H₂-on-BEST timing above, rerun on the new path. The target is with variance ≤ 60 µs.

## Non-goals

- Fallbacks in trained areas on the BEST model. They come from the H₂ vs H₃/H₄ fit conflict, a separate design.
- The plain `predict` at 32 µs, a descriptor cost that is not the bottleneck.
- No per-element data and no change to any trust rule or bound.
