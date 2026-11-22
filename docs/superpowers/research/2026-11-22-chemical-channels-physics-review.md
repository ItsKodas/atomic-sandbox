# Physics review: chemical channels (spec + plan 2026-11-22)

Reviewed: `docs/superpowers/specs/2026-11-22-chemical-channels-design.md`,
`docs/superpowers/plans/2026-11-22-chemical-channels.md`,
`docs/superpowers/research/2026-11-22-chemical-channels-literature.md`, against the current
`surrogate::{descriptor, basis, envelope, novelty, loo, fit, model, grade}` and
`providers::surrogate::{trust, policy, calibrate}`.

**Verdict: sound after changes.** The channel construction is correct physics: rotation-invariant,
permutation-correct, C¹, and forces stay −∇E. Two parts of the design are not sound as written:
- the W-uncertainty term, whose proposed Σ_W is not an uncertainty and fails in exactly the
  rare-element case it exists for;
- the trust sample floors, which a dense W inflates by about 20× (H{H}: 74 → 1 652 samples).

Both have derivable fixes that need no new tuned constants. Several cost claims and the day-1 gate
also need tightening.

Column counts below are from a script that copies `n_body3` and `for_each_body4` with S replaced by
K. It reproduces the spec's 6 656 (S = 4) and 14 670 (S = 5) at degree_cap 6 and D₄ = 3, so the user's
collection is at the recovery degree cap. All counts are at that basis unless stated.

| | S = 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| species Q′ | 6 656 | 14 670 | 28 284 | — | — | — |
| channels K = 3 | 3 296 | 4 140 | 4 992 | 6 720 | 8 480 | 10 272 |
| channels K = 4 | 6 656 | 8 340 | 10 032 | 13 440 | 16 880 | 20 352 |
| channels K = 6 (capped at S) | 6 656 | 14 670 | 28 284 | 37 776 | 47 300 | 56 856 |

Per centre, P₃ + P₄ is 351 + 453 at K = 3, 616 + 1 028 at K = 4 and 1 368 + 3 318 at K = 6.

---

## Critical

### C1. Σ_W "from the SVD fit's residual" is not an uncertainty of W, and it vanishes for the rare element

Spec: "Uncertainty with a frozen W" item 2. Plan: Task 5.

**What is wrong.**
- The residual of the rank-K truncation, ‖M − W V_Kᵀ‖, measures misspecification: how well K
  channels reproduce a species' coefficient fibre. It does not measure how well the data determine W.
  It does not shrink with data, and it is in coefficient units, not W units.
- For a barely-sampled species the low-order fit shrinks its fibre M[t,:] towards 0. Because
  W[t,:] = M[t,:]·V_K, its W row is small too. That makes three things small together:
  - the truncation residual;
  - g = ∂E/∂W[t,:], because the adjoint contracts with that species' small channel weights through Ā;
  - the species' contribution to every channel feature, so the c-posterior variance and the envelope
    cannot see it either.

  So gᵀΣ_W g → 0 exactly where the anti-goal says it must be large (the research notes' "shrunk row"
  risk, §3).
- The fallback, "scalar × I per row scaled by 1/n_samples(element)", has three faults:
  - It is not scale-covariant: W's columns differ in scale by σ_k, so an isotropic I_K means
    something different per channel.
  - n counted at query time keeps shrinking the term while W stays frozen at its learn-time error.
  - n counts any sample containing the element. HCl dimers say nothing about Cl's angular many-body
    row.

**Fix: the simplest defensible form, exact linear propagation of the low-order fit's own posterior.**
1. The low-order species fit has posterior N(m, Σ_low) over its standardised reduced coefficients θ.
   The unfolded matrix is a fixed linear map of θ, M = 𝓛(θ). 𝓛 un-standardises, applies the
   weighting of I2(b), unpacks the stored c₁ ≤ c₂ terms with off-diagonals halved (I2(a)), and
   symmetrises the two neighbour slots.
2. Take the SVD at θ = m: M(m) = U Σ Vᵀ. Define W = M(θ)·V_K Σ_K⁻¹, which equals U_K at the mean. For
   fixed (V_K, Σ_K) this is linear in θ: vec W = R θ.
3. Then Σ_W = R Σ_low Rᵀ.
   - It is (S·K)², at most 40 × 40 at S = 10, K = 4, with cross-species terms included.
   - It costs S·K triangular solves through the low-order factor.
   - It is first order: the uncertainty of the subspace V_K itself is second order for the dominant
     species. State that.
4. Behaviour, with no new constant:
   - A species the data barely see has a fibre posterior that reverts to the prior (1/λ in
     standardised units, large in raw units because its feature scale is small). So Σ_W,t is large
     even when its mean row is small.
   - A well-sampled species has Σ_W,t ∝ ~1/n and is negligible.
5. **Freeze Σ_W with W.** It is the error of the frozen W, so it does not shrink as later samples
   arrive. It is recomputed only at a relearn (I5c).
6. Predictive term: σ²_epi = φ̃ᵀΣ_c φ̃ + g_Wᵀ Σ_W g_W, with g_W = ∂(prediction)/∂vec W at (W, m_c).
   - **Energy:** ∂E/∂W[t,k] = Σ_i Σ_{n,lm} w_i^ch[k,n,lm]·A_i[t,n,lm]. This is the hot path's own
     adjoint contracted with the per-species A, so it is exact and costs one contraction. Keep A per
     species in the scratch (M1).
   - **Forces:** along any direction δW, E(W + hδW) is a polynomial of degree ≤ 3 in h (Ā enters
     quadratically in body 3 and cubically in body 4). The 5-point central stencil is therefore exact
     to rounding. Only the rows of species present in the scene are non-zero, so a variance
     evaluation costs |C|·K directions × 4 hot-path calls, on the existing variance cadence.
7. **Where it enters.** Add it to σ_epi before calibration, and record it in the history's model std
   (`AuditPoint` stds are what λ_sum, λ_T and λ_E are measured from). Otherwise λ is measured against
   the wrong std.
   - Calibration stays floored at 1 (never deflated), so the term can only widen.
   - If it overstates, λ stays 1 and only fallbacks rise (cost, not safety).
   - If it understates, the existing per-key λ inflates.
   - Rules 6 and 7 read it for both σ_F and σ_E.
8. **Known approximation.** This is block-diagonal Laplace. It ignores the c↔W cross term, so it
   double-counts the K² gauge directions δW = W·δT, which a refit of c would absorb. That is
   conservative and small for well-sampled species. The "well-sampled within 5 %" test measures it;
   if that test fails, project g_W off span{∂E/∂W · W E_ab}.

**N_W, with no new number.** A species gets a W row when the low-order species fit holds at least one
many-body key with that species as a neighbour that meets its own sample floor (n_k ≥ q_k in the
low-order basis). This is today's floor rule applied to the fit W is learnt from: the row is projected
only from coefficients the existing machinery calls sample-determined. Count informative samples:
atoms in the many-body class whose neighbour set contains t.

**N_trust.** Redundant once C2 is fixed:
- The per-key floors then equal the species model's for every key with |T| ≤ K, so the channel model
  never trusts a key on fewer samples than the species model would have.
- The W term covers the posterior side.
- The per-key calibration covers the measured side.

Either drop N_trust or define it as N_W. Do not add a second tuned count.

**Verified by:**
- the derivation above;
- the limit case of a species whose fibre is at the prior (Σ_W → prior, row → 0, so the variance
  stays finite and large instead of 0);
- reading `calibrate.rs`: λ is floored at 1, measured from the recorded stds, and keys below
  MIN_EVIDENCE take their class's λ as a floor.

### C2. The sample floors explode under a dense W, and trust becomes unreachable

Spec: "touched_columns covers the centre block". Plan: Task 6, "recomputed from channel Q′".

**What is wrong.**
- `trust::key_column_counts` holds each key to n_k ≥ q_k, where q_k is the set of independent columns
  non-zero on its atoms (`surrogate::touched_columns`). The whole-model `TrustKey::Samples` uses the
  union of those sets.
- With a dense learnt W, Ā_k = Σ_t W[t,k] A_t is non-zero for every k as soon as any neighbour is
  present. So every key touches its centre's whole channel block.
- Numbers at the collection basis (D = 6, D₄ = 3):
  - H{H} (H₂, H₃): species q = 8 + 43 + 23 = 74; channels at K = 4: 8 + 616 + 1 028 = **1 652**, 22×.
  - The whole-model union becomes the full Q′ of every centre present: **8 340** at S = 5, K = 4.
    That is more samples than the user's collection holds (CLAUDE.md's V11 row cites a 1 657-sample
    Accurate collection), so `Samples` would refuse the whole model.
- The constraint "the trust floors keep their meaning" is broken, and the day-1 gate does not measure
  it. The gate could GO on RMSE and the product would then answer nothing.

**Fix: derived, not tuned.** An atom of key k (centre s, neighbour species set T) has channel features
b^ch = M_T b^sp_T.
- M_T is a fixed linear map that depends only on W and T: Ā_k involves only A_t for t ∈ T, and body
  3 and body 4 are W-multilinear in them. Gradients are mapped the same way.
- So the key's predictions depend on c_s only through M_Tᵀ c_s, whose dimension is at most
  min(q^sp_T, q^ch).
- Hold the key to

  q_k^ch = n_max·|T| + P₃(min(|T|, K)) + P₄(min(|T|, K))

  computed as today's `touched_columns` in a shadow species basis restricted to T, capped by the
  touched channel columns. It is exact for generic W, meaning the T rows of W have rank min(|T|, K).
- For |T| ≤ K (almost every real key) this is exactly today's floor, so nothing is loosened. For
  |T| > K it is honestly smaller, because the key really has fewer parameters.
- The whole-model floor is the per-centre sum of the same bound over the union of T seen at that
  centre.
- The calibrate.rs doc rejected the rank "per key per refit" because it needs an eigendecomposition.
  This bound is combinatorial and costs nothing.

**Verified by:** the column counts above, and reading `key_column_counts`, `sample_floor` and the
calibrate.rs "sample floor per key" rationale.

---

## Important

### I1. The identity test (W = I, K = S) does not pin the channel code

- I is symmetric, so a row-major vs column-major mix-up (`w[t*k + c]` vs `w[c*S + t]`) passes.
- W[t_j, k] = δ, so a gradient that skips W (the species-style `if t.t1 != tj` test kept in the
  gradient loop) passes.
- A leftover composition skip (`present`, `active3`, `active4`) is still exact when W = I. With a
  dense W it zeroes channel terms that are not zero.
- It never runs K < S, so the n_body3 / n_body4 closed forms with K vs S, the column classes and the
  sidecar layout go untested.
- The FD force test cannot catch a consistently transposed W either: forces stay −∇E of the wrong
  model.

**Fix: add an algebraic oracle with random dense non-symmetric W, at K ∈ {1, 2, S−1, S}.**
- Body 3: B^ch_{(k₁n₁),(k₂n₂),l} = Σ_{t₁,t₂} W[t₁,k₁] W[t₂,k₂] B^sp_{(t₁n₁),(t₂n₂),l}, with B^sp
  expanded to the full symmetric matrix.
- Body 4: the triple sum, the same way.
- Gradients: the same identity, term by term.
- Also test:
  - a permutation-matrix W at K = S (permuted features);
  - species relabelling (permute the species order and W's rows together: features unchanged);
  - composition: a scene without species t gives the features of W with row t zeroed.
- Keep `n_species()` = S (body 2, the envelope, isolation) and add `n_channels()` = K, so no consumer
  silently gets K where it meant S.
- Run `channel_model_forces_are_minus_energy_gradient` with dense random W at K < S.

### I2. The recipe for learning W needs six corrections

a. **Unpack the stored c₁ ≤ c₂ coefficients as a symmetric quadratic form.**
   - E = Σ_{c₁≤c₂} c B = Σ_{c₁,c₂} C A_{c₁}·A_{c₂}, so C_cc = c_cc and C_{c₁c₂} = c_{c₁c₂}/2 for
     c₁ ≠ c₂. This includes the same-species, different-n pairs.
   - Without the halving, hetero entries carry 2× weight and bias the subspace, silently.
   - Body 4: divide by the multiplicity of the unordered triple.

b. **Weight by energy relevance, not by raw coefficient size.**
   - Raw coefficients are large exactly for features that are small, such as an usually-distant
     neighbour. An unweighted SVD therefore over-weights them.
   - Take a_t = the pooled RMS of the body-2 coordination feature b2[(t, 0)] over the energy rows
     (its existing standardisation scale). Form M̃ = D_a M D_a over both neighbour slots, SVD it, and
     set W = D_a⁻¹ Ũ_K.
   - This is the "weighting by data counts" you asked about. The pooled scale includes frequency, so
     rare, noisy fibres pull the subspace less, and their rows are still projected in.
   - Precision weighting (dividing by the posterior std) is the alternative. Compare weighted against
     unweighted in the gate (subspace angle and accuracy).

c. **Drop "scaled by singular values" and store U_K (orthonormal, dimensionless).**
   - Under the fit's per-feature standardisation, scaling W's column k by σ_k scales every body-3
     feature (k, k′) by σ_kσ_k′ and its scale by the same factor. Body 4 behaves the same way.
   - So the standardised features, the prior, the evidence, novelty τ and the envelope's spread units
     are all unchanged. The grade is invariant under any invertible linear map (grade.rs doc).
   - Column scale is therefore an exact gauge. Keeping σ_k only makes W carry Hartree and body-4
     features carry ~σ³ (dynamic range in the raw Gram).
   - Likewise, only col-span(W) sets the model's span: the degree caps involve n and l, never k, so
     every invertible K × K transform maps the term set to itself. A rotation within the span changes
     only the prior. The K = S control (I3) measures how much that matters.

d. **Sign convention.** "Largest entry positive" by value fixes nothing: v and −v usually both have
   a positive largest entry. Use the entry of largest |value|, ties broken by the lowest species
   index. Note that the sign is an exact prediction gauge, so this is for determinism and the file
   only.

e. **Determinism.** Form the S × S Gram M̃M̃ᵀ with compensated sums and decompose it with the existing
   Jacobi `linalg::symmetric_eigen`. That is S ≤ 10, bitwise deterministic, with no threaded SVD.
   Squaring the condition number is harmless at S ≤ 10 in f64, and only the subspace is used.

f. **K ≥ S and rank.**
   - Set K_eff = min(K, S, numerical rank), where the rank counts σ_k > S·ε·σ₁.
   - When K_eff = S, set **W = I**: exactly the species span, and the identity test proves the code
     path. Early 1–2-element collections then never run an SVD of a nearly degenerate fibre matrix.
   - A rank below K would otherwise give exact linear dependencies among the channel columns. The
     ridge and novelty tolerate them, but they are wasted columns.
   - Report the gap σ_K/σ_{K+1} and a per-species truncation residual
     ρ_t = ‖M̃_t − W̃_t V_Kᵀ‖² / ‖M̃_t‖². A species with large ρ_t is poorly represented at that K,
     which is the "highly correlated vs distinct species" diagnostic.
   - Optional: 1–2 HOOI steps (project the t′ mode onto W, then re-take the SVD of the t mode). They
     are deterministic and cheap, and they improve the shared-factor Tucker.

### I3. Choosing K by evidence degenerates, and K = 6 at S = 5 is the species model

- With thousands of force rows, any accuracy difference is worth hundreds to thousands of nats of
  evidence. "The smallest K within the margin of the best" therefore picks the largest K.
- At S = 5, K = 6 is capped to 5: the species span at species cost (14 670). The spec's relearn also
  fits all three K, so it costs more than a species cold refit:

  | stage | fraction of a species cold refit |
  |---|---|
  | K = 3 | 0.02–0.08 |
  | K = 4 | 0.18–0.32 |
  | K = 5 | 1.0 |
  | low-order fit, Q′ 4 895 | 0.04–0.11 |

  Each range runs from (Q′ ratio)³ to (Q′ ratio)². The cold-refit stages scale between Q′² (LOO,
  accumulation: n·m·Q′²) and Q′³ (the spectral search).

**Fix.**
- Bound K ≤ K_max(S), where K_max is the largest K whose Q′ is below the cost budget.
- Choose K by held-out force RMSE within the gate's own 10 %. Do it once at a relearn, not per
  refit.
- Run K = S only as the **control**. It spans exactly the species space, so any accuracy difference
  from the species model is the prior / standardisation gauge. If that difference is more than about
  3 %, every K comparison is confounded.
- If evidence is used anywhere, state its margin in nats with a citation (Kass & Raftery's "strong",
  ln 20 ≈ 3).

### I4. The W-learning fit reintroduces the scaling problem, and the absolute scaling should be stated

- The low-order fit (species body 2 + body 3) has Q′ = n_max·S(S+1)/2 + S·P₃(S), which grows like S³
  (so the solve grows like S⁹). At the collection's own body-3 basis (D = 6) that is 4 895 at S = 5
  and **38 040 at S = 10**: larger than today's whole 5-element model.
- **Fix:** specify a reduced low-order basis.
  - n_max 4, l_max 2, D 2 gives 795 / 2 976 / 5 640 at S = 5 / 8 / 10.
  - n_max 6, l_max 2, D 3 gives 1 370 / 5 312 / 10 190.
  - Gate that its W matches the D = 4 W at S = 5, by subspace angle and accuracy.
- **State the absolute scaling.** The per-element refit multiplier falls from ~10× to ~1.7× at S = 5,
  so that claim holds. But:
  - At K = 4 the channel model at S = 10 is Q′ 16 880, larger than today's S = 5. That is about
    2.3 GB per Q′² copy and a cold refit of roughly 1.3–1.5× today's 1 869 s.
  - Channels buy about +4–5 elements at K = 4 (about +7 at K = 3) before the memory wall, not "many
    elements". The flat-in-S step is centre channels.
- **Check:** the spec's "K = 4 ≈ today's 4-element per-centre size" holds: 1 644 vs 1 664 per centre.
  But Q′ at S = 5 is 8 340, so the 25 % time gate is tight (I3).

### I5. The new-element lifecycle has gaps that make W stale or wrong

a. **Fitting.** Samples containing a species with no W row must be stored but left out of the fit.
   Otherwise that species' many-body energy cannot be represented: its neighbours' centre blocks
   absorb it (bias), or it fills the misfit set M. Rule 3's refusal covers answering, not fitting.

b. **When to relearn.** Do not cold-relearn when an element first arrives. "When the species set
   changes" means a cold refit at 1 sample and another at N_W. Relearn when it reaches N_W.

c. **Staleness.** A W row is learnt at its weakest (N_W samples) and then frozen while the samples
   grow 100×, until the next species change or Refit-all.
   - Relearn when an element's informative count has doubled since its row was learnt. Doubling cuts
     the row's std by √2, the hysteresis band: O(log n) relearns.
   - Recompute Σ_W only then.

d. **Row identity.** Key channels.txt rows by Z (a label only, rule 1). The loader refuses a species
   set that differs from the basis. Inserting an element with a lower Z otherwise shifts every row
   index (`w[t*k + c]`), and O's row is silently applied to Li.

e. **A relearn is a new model family.** Reset `class_memory` so every key re-enters at ENTER (1/√2)
   rather than staying trusted up to LEAVE (1) on evidence measured under the old W. Bump the fit id
   and the generation so the evidence windows age as after any refit.

### I6. The LOO is W-leaky

- W is learnt from all samples, including the one the closed-form LOO leaves out (`loo.rs` holds the
  hyperparameters fixed; W is now a fitted quantity outside it).
- For abundant elements the leak is about K/n_t and negligible. For a rare element it is not, and the
  uncalibrated class rule and the LOO floor R_c read exactly those numbers for rare keys, which are
  below 12 fits / 60 points.
- **Fix:** the N_W rule (C1) keeps n_t above the low-order floor. The gate measures the leak (I7):
  fold-relearn on the rarest element, compared with the fixed-W LOO.

### I7. The day-1 gate must measure trust outcome and calibration, not only aggregate RMSE

"Force RMSE within 10 % at K ≤ 4" and "cold refit ≤ 25 %" are sensible thresholds, but each needs
to be defined more tightly, and the gate needs more measurements.

- **Held-out definition.** Use group-held-out by scene or trajectory (MD samples are correlated),
  with W relearnt per fold. Training RMSE is meaningless here, and fixed-W LOO is optimistic (I6).
  Report the fold spread, and decide only when the spread is smaller than the 10 % margin.
- **Energy.** Energy RMSE within 10 % too. Intermolecular energetics are the stated motivation.
- **Breakdowns.** Per element and per key, not aggregate: H and O dominate the aggregate.
  Suggested: no element's force or energy RMSE more than 20 % worse, and no key's LOO ratio crossing
  1.
- **Trust outcome**, with C2's floors:
  - per key: n_k, q_k, the verdict, LOO max and RMS;
  - the `would_answer` verdict at `refit_golden`'s probes, against the species model.

  A GO with fewer keys trusted or fewer probes answered is not a GO.
- **Calibration.** On the held-out folds: RMS of z = error/σ_total per key, and the count of
  *admitted-but-wrong* atoms (σ within rule 6's bound at λ = 1, error beyond WORST_ATOM × bound). The
  channel model must not exceed the species model's z by more than (1 + 1/√(2n)), nor exceed its
  admitted-but-wrong count.
- **Leave-element-mostly-out**, moved forward from Task 5 because it decides viability:
  - Subsample the rarest element (by informative count) to 5 % and 20 %, relearn W and fit.
  - On its held-out samples, compare z, admitted-but-wrong and σ(5 %) ≥ σ(100 %) at the same
    geometries, against the species model on the same split.
  - This replaces `rare_element_error_bars_widen`'s "same geometry type, different element"
    comparison, which is ill-posed (different elements have different features). Use the same
    element, subsampled.
- **Timing.** End to end per relearn: the low-order fit, the SVD, re-describing every sample, the K
  choice and the final fit. Report each stage (as `refit_profile` does), the warm append-20 refit and
  peak memory.
- **Controls.** The K = S control (I3). The W = I path must reproduce the species model bit for bit
  (M3).

---

## Minor

- **M1. Hot path.**
  - Accumulate per-species A as now, then mix Ā = WᵀA once per centre (S·K·n_max·nlm). This is
    cheaper than spreading every neighbour into K rows, and it keeps A_t for g_W (C1).
  - Pre-contract the adjoint per centre, w_sp[t] = Σ_k W[t,k] w_ch[k], so the neighbour force loop at
    model.rs:1230–1246 is unchanged. Otherwise it costs K× more.
- **M2. Variance support.** `model::composition_support` (model.rs:706–729) reads body-3/4 term
  indices as species bits. With channels the support must be "all body-3/4 columns of the centres in
  C" plus the body-2 pairs in C. The OUTSIDE guard (model.rs:664–685) makes the old function safe
  (it falls back to the full path) but silently slow. Rewrite it and test it.
- **M3. Identity-fit tolerance.** With W = I, Ā = 1·A + (+0) is exact, so the channel path can be
  bit-identical to the species path. Assert bitwise. If the summation order differs, use the
  `refit_golden` bar (1 % of the tiers): "1e-9" on a κ(A) ≈ 1e13 system (references.md, refit
  speed-up) is a flaky tolerance, not a derivation.
- **M4. Defence in depth.** In the species basis, an unseen neighbour pair at a centre (O{H, Cl} with
  only O{H} and O{Cl} trained) made body-3/4 columns that were zero in training non-zero. That fired
  5c (u_k = 0) and 5d. Channels populate every column, so only body 2 and the EnvKey floor (n_k = 0
  for a new neighbour set) remain. That is sufficient, but pin it: add a test that such a key is
  refused in a channel collection.
- **M5. Rules 5a/5b: confirmed unaffected.**
  - `envelope::isolated_atom` (envelope.rs:119–121) and `Envelope::coordination_ratio`
    (envelope.rs:220–240) read only b2[(t, 0)] = C_t, and body 2 stays per species.
    `energy_row_atoms_at` is geometric.
  - With signed W nothing reads a channel value for isolation. The premise E_i → E_ref + O(C) as
    C → 0 still holds, because every channel feature is a polynomial in the A's.
  - No change is needed in envelope.rs or policy.rs. Keep `a_neighbour_with_zero_channel_sum_is_not_isolated`
    as a regression test, and debug-assert that both functions index below `n_body2()`.
  - The spread rule (5c) handles signed features (a min/max box).
- **M6. Cross-centre ties.** There are no new exact ties. Body 3/4 stay untied: the centre is the
  vertex. The body-3 j = k self-terms R_nR_n′f_c²(r/r_c)^{2l} are not in the body-2 span (f_c² is not
  in span{T_n f_c}), as before. Data-dependent dependencies (dimer-only data) are unchanged in kind,
  and novelty handles them.
- **M7. Low-order fit.** Learn W from the posterior **mean** of a fit with evidence-chosen λ. Record
  that fit's Q′, λ, σ and the per-species ρ_t in channels.txt so a relearn can be audited.
- **M8.** A centre-mode SVD of the same low-order tensor is free. It previews how compressible the
  centres are for the later "centre channels" step.

---

## Item-by-item answers

1. **Construction.**
   - Rotation-invariant: Ā_k is a scalar-weighted sum of same-l objects, so A_l → D^l A_l carries
     over, and the Σ_m and Gaunt contractions are invariant.
   - Permutation-correct: equal species have equal weights.
   - C¹, with forces = −∇E exactly, provided the gradient carries W[t_j, k] (I1).
   - Exact dependencies: only when rank(W) < K. Cap K (I2f).
   - The identity test does not pin correctness (I1).
2. **Learning W.** Sound as a mode-1 HOSVD of a partially symmetric Tucker tensor (the left singular
   subspace equals col-span(W) when the model is exact). It needs: the symmetric unpacking, energy
   weighting, orthonormal U_K (column scale and sign are exact gauges), the |·| sign rule, a
   deterministic S × S eigensolve, K_eff = min(K, S, rank) with W = I at K_eff = S, and the ρ_t /
   gap diagnostics (I2).
   - A very rare species is handled by N_W plus Σ_W reverting to the prior (C1).
   - Highly correlated species share a channel by design; ρ_t shows when one does not fit.
3. **Uncertainty.** Use the posterior propagation of C1. Freeze it with W. Put it in σ_epi before
   calibration and record it in the history. N_W comes from the low-order floor; drop N_trust given
   C2. The spec's sanity gate is not adequate; replace it with I7's leave-element-mostly-out (z,
   admitted-but-wrong, monotonicity).
4. **Signed W and rules 5a/5b.** Already honest. Body 2 stays per species, and no geometric
   replacement is needed (M5).
5. **Day-1 gate.** The thresholds are sensible once they are measured held-out per fold, with energy,
   per element, the trust outcome, calibration, the leave-element-mostly-out and end-to-end timing
   (I7). K = 6 at S = 5 is the control, not a candidate (I3).
6. **Confidently wrong, or a wrong scaling claim.**
   - Confidently wrong: the SVD-residual Σ_W (C1), the W-leaky LOO (I6), fitting samples with no W
     row (I5a), the stale row (I5c), misaligned rows (I5d), hysteresis carried across a relearn
     (I5e).
   - Wrong scaling: the floors (C2), the S⁹ low-order fit (I4), the K-selection cost (I3), and
     absolute S = 10 ≈ 1.15× today's S = 5 Q′ (I4).

## Not checked

- Cold-refit stage timings at channel Q′: inferred from Q′² / Q′³ scaling, not measured. The gate
  measures them.
- Whether the user's collection's rarest element meets the proposed N_W. That needs its key counts.
- The literature claims (Darby SI Fig. 4, Lopanitsyna Fig. 2) were not re-read. I relied on the
  research notes.
- The cost of `grade.rs`'s MaxVol at channel P. It is invariant to the W gauge (standardised; its doc
  says it is invariant under invertible maps), but not timed.
- sidecar.rs and collection.rs format details, and the teach/sampling floors beyond
  `key_column_counts` and `sample_floor` (`fit_floor`, `min_samples_for_trust_basis`,
  `richer_basis`). Apply C2's bound to each.
