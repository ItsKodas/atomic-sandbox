# Physics review: chemical channels Task 4 diff, Task 2a’s Σ_W, Task 5 redesign (2026-11-24)

Reviewed: task4.diff (branch worktree-agent-ae9d5c96d5f820f2d, commit 007b9e3f) with the worktree’s
descriptor.rs / model.rs / sidecar.rs; crates/surrogate/src/channels.rs module doc on main;
crates/validate/references.md "Chemical channels: the day-1 gate"; the spec and the 2026-11-22 review.
No files edited, no heavy tests run. One numeric check run (B).

---------------------------------------------------------------------------------------------------
## A. Task 4 (channel-aware hot path, variance support): LAND AFTER FIXES (one small fix)

### Derivation check (passes)
Ā_{(k,n),lm}(i) = Σ_{j≠i} W[s_j,k] φ_{n,lm}(d_ij), d_ij = r_j − r_i.
∂E_i/∂r_j = Σ_{k,n,lm} w_{(k,n),lm} W[s_j,k] ∇φ_{n,lm}(d_ij) = Σ_{n,lm} w̄_{(s_j,n),lm} ∇φ,
w̄_{(t,n)} = Σ_k W[t,k] w_{(k,n)};  ∂E_i/∂r_i = −Σ_j ∂E_i/∂r_j (Ā depends on r only through d_ij).
- model.rs (worktree) 1296–1331: the pull-back reads the W row of the NEIGHBOUR species t
  (wt[t*kk..]), the same row-major layout as add_neighbour (descriptor.rs:866), describe
  (descriptor.rs:1020) and reached_channels (w[t*kk+k]). F_j −= g, F_i += g: correct sign
  (F = −∇E), ΣF = 0 by construction.
- A repeated channel index in body 3 (c1 == c2) accumulates 2c·a in w (model.rs 1269–1270), the
  correct derivative of c·Σa²; the body-4 product rule likewise (1286–1289). The j = k self products
  are inside Ā automatically; the centre is excluded from its own Ā (j ≠ i). No missing self term.
- W = I path: adj = &w directly, the species path bit for bit (identity fit measured 0 difference).

### Variance support: exact, no miss possible
K(C) = {k : ∃ t ∈ C, W[t,k] ≠ 0}. For k ∉ K(C): add_neighbour skips w == 0.0 (descriptor.rs:876;
−0.0 == 0.0, so −0.0 is skipped too), so Ā_k stays +0.0; every body-3/4 value with that factor is a
sum of ±0 products; every gradient entry either skips the factor (describe 1029/1034/1043, 1073/1080:
weight == 0) or multiplies by Ā_k = 0. reached_channels uses the same predicate (!= 0.0), so −0.0 is
"not reached" consistently, and a tiny or subnormal non-zero entry IS reached (a superset: including a
column whose value underflows to 0 is harmless). Non-finite W is refused at with_channels
(descriptor.rs:490), so a NaN cannot hide a non-zero. K ≤ S is enforced (K > S → I,
descriptor.rs:491), so 1 << k cannot overflow. W = I gives K(C) = C, the species definition column
for column; a dense W gives every channel (the centre’s whole block).
Verified by reading the three code paths and the limit cases (zero row, −0.0, subnormal, W = I).

### Tests: sufficient
The chain closes: Task 1’s algebraic oracle (random dense non-symmetric W) pins the features and its
FD test the descriptor gradients; Task 4’s design-row match (1e-12; dense K < S, K = S, identity,
species) ties the hot path’s energy and forces to features·θ; the FD test (2.4e-12) ties the forces to
−∇E; the double-double dense-posterior test with a block W (K(C) a strict subset) pins the support
and the stds; the support test checks that every query vector vanishes off S(C). Moving the oracle to
double-double instead of widening the bound was right.

### Important
- **A-I1. The sidecar silently drops W.** sidecar.rs (worktree) ~1268 decodes `channels: None`, and
  `encode` (sidecar.rs:654) has no guard. A channel model with K = S and a dense W (allowed:
  descriptor.rs:491 only replaces K > S) has exactly the species basis’s layout (same Q′, same
  coefficient count), so a write/read round trip yields a species model carrying the channel model’s
  coefficients: different predictions, no error, i.e. a confident wrong answer. K < S fails loudly
  on the coefficient count; only K = S is silent. No production path writes a channel model yet
  (only tests and the gate build one), hence Important rather than Critical. Fix: `encode` returns
  InvalidInput when `model.basis.channels().is_some()` until Task 3 stores W; add a test.

### Minor
- A-M1. `identity_mix_fit_equals_the_species_fit` asserts 1e-9 but measures 0: assert bitwise
  (review M3), so a future summation-order change is a visible decision, not silent slack.
- A-M2. Pin the predicate consistency: add a 1e-300 entry (and a −0.0) to a support-test mix and
  assert the 1e-300 channel is in S(C). Both paths use `== 0.0` today; a later threshold in
  add_neighbour would keep the superset, but a threshold in reached_channels would miss columns.
- A-M3. The pull-back loops over all S species, absent ones included (S·K·n_max·nlm ≈ 2.6e3 madds
  per centre at S 5, K 4): negligible beside the Gaunt loops; optional.
- A-M4. Full-path cost with a dense W (≥ 3 species at S = 5): (3N+1) solves at Q′ 4 140 cost about
  1/12 of the species model’s full path at 14 670, so not a regression; measure in Task 7 as planned.
- A-M5. The FD bound 1e-7 against 2.4e-12 measured is loose, but the 1e-12 design-row test is the
  tight check.

---------------------------------------------------------------------------------------------------
## B. Task 2a’s Σ_W (exact first-order off-span derivative): ACCEPT

The exact first-order eigenvector derivative of the symmetric Gram G = M̃M̃ᵀ is
δũ_k = Σ_{m≠k} ũ_m (ũ_mᵀ δG ũ_k)/(λ_k − λ_m), δG = δM̃M̃ᵀ + M̃δM̃ᵀ, so
ũ_mᵀ δG ũ_k = ũ_mᵀ δM̃ p_k + ũ_kᵀ δM̃ p_m with p = M̃ᵀũ. That is the implemented γ_mk. It reduces to
the frozen form only when p_m = 0 (λ_m = 0), as the doc says. My C1 frozen form was wrong off the span
at first order whenever σ_{K+1} ≠ 0.

Numeric check run (numpy; S 5, N 40, σ 3.8/2.3/1.1/0.87/0.3, K 2): the exact formula matches a
central difference of the eigenvector to 5e-11…3e-10 (off-span part); the frozen form is 40 % and
64 % off for k = 1 and 2. This agrees with the implementer’s MC (frozen ~2× low, exact within 3.3 %).

Keeping the frozen in-span part is acceptable: in-span changes are the K² gauge directions that a
refit of c absorbs, so they can only overstate. D_a weighting does not break the split: D_a⁻¹ maps
span(Ũ_K) onto col-span(W) and ũ_m (m > K) outside it, so "off-span" is the span-changing part in any
metric. η (Frobenius spread over the gap, Wedin) is a sufficient-condition diagnostic, correctly
reported. Given C, Σ_W stays a recorded diagnostic, not an error-bar term.

---------------------------------------------------------------------------------------------------
## C. Task 5 redesign

### What the gate changes
- η = 242–6 030: Σ_W is far outside first order, where it UNDERSTATES (2× already at η 1.9; unmeasured
  beyond). gᵀΣ_W g would be a cosmetic widening, neither a bound nor calibrated.
  **Withdraw C1’s predictive term.** Keep Σ_W, η and ρ_t as diagnostics in channels.txt.
- W is species-diagonal; the Li and Cl rows are zero (ρ = 1) because they occur only in diatomics, so
  their body-3 fibre carries no many-body information (a one-neighbour body-3 feature is close to the
  body-2 span). The SVD is doing the right thing; the trust system must know about it.
- The z_epi overconfidence (median 3.5–6) is the SAME for the species model: it pre-exists, and it is
  what calibrate’s per-class/per-key λ (λ_sum, λ_T, λ_E) already scales. Not a channel regression.
  Task 5 need not fix it, only not make it worse.

### Critical: the silent path a zero row opens today (without Task 5)
For O in HOCl, Ā sees H only (Cl row 0). 5d and 5e are blind: if O–Cl pairs exist in training, O’s
descriptor in HOCl is the sum of an O{H} row and an O{Cl} row (whose body-3/4 part is 0), inside the
training span with grade ≈ 1. The only guard is the EnvKey floor (n_k = 0 → Q′). Once O{H,Cl}
samples are fitted:
  (i) q_k from the touched columns leaves out the Cl many-body columns (they are 0), so the floor is
      n_max·2 + P₃(1) + P₄(1), reached quickly;
  (ii) worse, those samples’ H–O–Cl many-body energy, which cannot be represented, is absorbed by O’s
      H-channel coefficients, biasing O{H}, i.e. WATER, a well-sampled key that then answers
      confidently.
So yes: the model would treat Cl’s many-body environment as "seen", and corrupt a trusted key.

### Proposed Task 5 rules (no Σ_W term; refusal, not widening)
R1. Representation status per species, computed at learn time, frozen with W, stored in channels.txt
    keyed by Z (a flag, never inferred from W’s values). t is REPRESENTED iff
    (a) N_W (review C1): some many-body EnvKey (class 2) whose neighbour set contains t has n_k ≥ q_k
        in the low-order fit: the existing floor applied to the data W is learnt from; and
    (b) ρ_t ≤ ρ_max = 0.01. Derivation: √ρ_t is the relative RMS error of t’s weighted body-3
        coefficient fibre after truncation; require it ≤ the existing 10 % relative force bound
        (force_std_rel), so ρ ≤ 0.1². Conservative: the many-body part is a fraction of the force,
        and a refit of c can absorb some of the discarded part. NaN ρ (no fibre) fails.
    On the gate’s data H, O and S pass (a); Li and Cl fail (a). No tuned number decides that case.
R2. Learn W from the represented species only: zero the M̃ rows of species failing (a) before the Gram,
    so a pair-only species neither spends a channel (Li took K = 4’s fourth) nor tilts the subspace.
    Single pass: species failing (b) after the SVD are marked unrepresented and their W rows set to
    exact 0. K_eff ≤ S_rep; W = identity on the represented species when K ≥ S_rep (I2f generalised).
    The K choice (I3) prefers the smallest cost-bounded K at which every (a)-species passes (b), and
    reports the ones that fail at K_max.
R3. Query rule (a new rule 3b beside rule 3; teacher’s surface; its own history code and words): a
    scene with any many-body-class atom (≥ 2 neighbours within r_cut) whose neighbour set contains an
    unrepresented species → teacher; the sample is kept as rule 3 keeps its samples. The reason names
    the species and its N_W progress. Pair-class atoms with an unrepresented neighbour are judged as now
    (body 2 carries them; the gate’s HCl, HLi and LiO are accurate). Lone and stray exemptions
    unchanged.
R4. Fit exclusion (I5a generalised): a sample holding any fitted many-body atom with an unrepresented
    neighbour is stored but left OUT of the channel fit, the whole sample, because its energy row and
    every neighbour’s force carry that atom’s E_i. It still enters the low-order fit and counts toward
    N_W. This closes the water-bias path above.
R5. Floors (C2, unchanged form): q_k = min(touched channel columns of k’s atoms, touched columns of the
    shadow species basis restricted to T), with n_k and q_k over FITTED samples only. Zero rows make a
    pair atom with an unrepresented neighbour honestly n_max-only. Unseen key → Q′ (unchanged).
R6. Novelty unchanged, with its blind spots stated: 5d/5e cannot see zero-row neighbours (covered by
    R3), nor, under a dense W, a new combination of represented species (covered only by the EnvKey
    n_k = 0 floor, review M4). Both pinned by tests.
R7. Lifecycle: an unrepresented species reaching N_W triggers a relearn; a represented species’
    many-body informative count doubling triggers one (I5c); a new species enters with an exact-zero
    row, unrepresented, with no relearn (I5b). A relearn is a new W generation: class memory reset
    (ENTER), and audit-history points of earlier W generations left out of the key evidence counts and
    the λ samples (I5e extended).
R8. Calibration unchanged. Task 5 asserts that the channel model’s held-out z is no worse than the
    species model’s by more than (1 + 1/√(2n)) (I7), as the gate already measured (equal).

### Tests that prove the anti-goal
- T1 `a_many_body_atom_with_an_unrepresented_neighbour_is_refused`: on an H/Li/O/S/Cl store shaped
  like the gate’s, HOCl, H₂O·HCl (Cl within r_cut of water’s H) and LiOH are refused by R3 with its
  reason, and their samples kept; assert 5d and 5e would NOT have fired there (documents why R3 is
  needed).
- T2 `pair_atoms_with_an_unrepresented_neighbour_still_answer`: the HCl and HLi verdicts and
  predictions are bit-identical to the pre-Task-5 model’s.
- T3 `excluded_samples_do_not_move_represented_keys`: a fit with and without added HOCl samples gives
  bit-identical coefficients, n_k and q_k; water predictions unchanged.
- T4 `a_rare_element_is_unrepresented_until_n_w_then_relearns`: S thinned to 5 % → unrepresented
  (N_W unmet) and H₂S many-body atoms refused; add H₂S samples until H{H,S} meets the low-order floor
  → relearn → S represented, class memory reset, its keys at ENTER.
- T5 `representation_status_is_a_flag_not_a_value`: planted tensors with ρ = 0.5 (fails), ρ < 0.01
  (passes) and NaN (fails); an N_W-unmet species with a small noisy non-zero fibre is still
  unrepresented and its row is exactly 0.
- T6 `an_unseen_combination_of_represented_species_is_refused` (M4): O{H,S} with O{H} and H{H,S}
  trained → refused by the n_k = 0 floor.
- T7 the gate rerun through the decision list (possible now that Task 4 has landed),
  leave-element-mostly-out (S at 5 % and 20 %):
  - admitted-but-wrong atoms (rule 6 at λ 1, error > WORST_ATOM × bound) ≤ the species model’s on the
    same split;
  - zero admitted many-body atoms with an unrepresented neighbour;
  - z within (1 + 1/√(2n)) of the species model’s.
- T8 a channels.txt round trip of the flags, ρ_t and η, keyed by Z; a mismatched species set refused.
- Drop `rare_element_error_bars_widen` (ill-posed, I7) and the "well sampled within 5 %" test (there is
  no Σ_W term). Replace the latter with bit-identity of the stds for represented-only scenes against
  Task 4.
- Spec: amend the binding section. C1’s gᵀΣ_W g is withdrawn (η measured 242–6 030). I5’s "no W row →
  refuse every scene, drop it from the fit" is replaced by R1–R4 (pair-only use allowed).

---------------------------------------------------------------------------------------------------
## Not checked
- Per-species η_t (whether the H and O rows are themselves well determined under η ≫ 1): not computed.
  I do not gate on it: a violated global Wedin bound does not imply a wrong subspace, and the gate’s
  per-fold relearns gave consistent accuracy. The gate should report the per-species row angle between
  the fold-relearnt W’s as the direct measurement.
- Whether ρ_max = 0.01 leaves channel SHARING useful in a many-element collection: there is no such
  data (the gate’s W is diagonal). If sharing species fail it, the honest outcome is DFT for their
  many-body atoms, or a larger K.
- N_W counts MD near-duplicates as independent samples (a weakness every floor already has).
- No tests run (as instructed); Task 4’s measured numbers are taken from its report.
