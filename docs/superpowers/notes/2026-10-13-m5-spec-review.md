# M5 surrogate-provider spec — physics/numerics review

Reviewer: Fable 5.1 (physics-reviewer role, read-only). Date 2026-10-13. Subject:
`docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md` (draft). Inputs consulted:
CLAUDE.md rules 1/2/4/6/7; parent design §5.1, §7, §9, §11; M3 handoff (f32 floors, path scatter
6e-7 Ha / 5e-6 Ha/Bohr, egg-box ΣF ≈ 3e-4 Ha/Bohr); M4 Task 7 report (H₂ at σ = h = 0.25:
R_e 1.789 Bohr, k 0.200 Ha/Bohr², ω_MD 3236.9 cm⁻¹, NVE drift −4e-6 Ha/ps, fluctuation 1.9e-6 Ha;
the H₂O-on-this-surface finding); `crates/sim/src/provider.rs` (`ForceProvider`, `EvalStats`,
`ForceResult`) and `dynamics.rs` (`energy_conserved` = E_KS + K + U_spring + U_field − W).

**Verdict: sound with changes.** Linear ACE (body order 2–3, species-channelled, ridge fit,
Bayesian variance, DFT fallback) is the right first rung and the right architecture behind
`ForceProvider`. The spec has one mathematical error (the real-harmonic invariant), one
missing physics term (atomic reference energies), one wrong noise floor (path scatter instead
of egg-box/SCF-lag), an under-counted descriptor budget that breaks the 10³ steps/s target
through the uncertainty evaluation, a validation system (H₂O) that the M4 suite has already
shown cannot be sampled on this surface, and an energy-bookkeeping choice that makes the
suite's drift assertion untestable. All are fixable in the spec before planning. Grades below
are MUST (spec must change), SHOULD (change unless there is a reason recorded in the spec),
COULD (worth a line or a flag).

---

## A. Model adequacy

**A1. Is body order 3 with l_max 3, n_max 8, r_c 6 Bohr enough?** For the systems named, yes
for energies to ~1e-4 Ha and forces to ~2e-3 Ha/Bohr *inside the sampled region*, with the
following structure:

- **H₂ (and any diatomic).** Only pair terms carry information: every atom has one neighbour,
  so the body-3 invariant Σ_m A_{n1lm}A_{n2lm} collapses to R_{n1}(r)R_{n2}(r)·(2l+1)/4π — a
  function of r alone, redundant with (not identical to) the pair channel. Within the trained
  window 1.2–2.2 Bohr the model is a degree-≤8 polynomial in r times f_c; a Morse-like well
  with a ≈ √(k/2D) ≈ 1.05 Bohr⁻¹ (k = 0.200, D ≈ 0.09 on this surface) over a ±0.5 Bohr window
  has a·x ≤ 0.5, so the degree-8 truncation error is far below 1e-6 Ha. H₂ is a *1-D fit* and
  should be held to much tighter numbers than the spec's 1e-4 / 2e-3 (see E3): a 2e-3 Ha/Bohr
  force error that varies over a 1 Bohr correlation length is a 2e-3 Ha/Bohr² error on
  k = 0.200, i.e. 1 % in k and **0.5 % in ω** — the spec's "ω within 1 %" and "forces < 2e-3"
  are not independent targets; the force target must be tighter for the ω target to be safe.
- **H₃⁺, H₂O (one centre with two neighbours).** The complete set of symmetric functions of
  (r₁, r₂, θ) is spanned by R_{n1}(r₁)R_{n2}(r₂)P_l(cos θ): body order 3 **is complete** for a
  2-neighbour centre as n, l → ∞. With l_max = 3 the angular dependence is a cubic in cos θ.
  For a bend from 104° to linear, cos θ ∈ [−1, −0.24]; a cubic captures a bending potential
  to ~1e-3 Ha, and ~1e-4 needs l_max 4–6 (cheap: each extra l adds (2l+1) harmonics and one
  more l-slot per (n1,n2) pair). **Body order 4 is not what the bend needs**; raise l_max
  first. Body order 4 (CG-coupled bispectrum) matters for 3+-neighbour centres (NH₃ pyramid
  inversion, CH₄ Jahn–Teller-like distortions, 20-atom clusters) where the sum of pairwise
  angular terms cannot resolve all three-fold arrangements. Keep it off by default (spec is
  right), but say "l_max before body order 4" in §8. SHOULD.
- **Where linear ACE breaks (document in §2/§7, these are not fixable by more data):**
  (i) *Charge transfer and non-locality.* E = ΣE_i with r_c = 6 Bohr is exact for a molecule
  that fits inside every atom's cutoff (all M5 oracles do) and an approximation beyond; the
  DFT energy of a 20-atom cluster is not a sum of 6-Bohr-local terms (electrostatics of
  partial charges decays as 1/r). Expect a few 1e-3 Ha of irreducible error on clusters; the
  spec's 1e-4 target is for ≤ 4 atoms. (ii) *Dissociation and spin.* The per-(charge,
  multiplicity) channel means the model learns one spin solution; where LSDA's lowest state
  changes character (H₂ past ~3 Bohr, the near-linear H₂O level approach from Task 7) the
  teacher's surface has a kink or a stall and the surrogate learns whatever samples it was
  given there. (iii) *Egg-box.* The DFT forces contain a non-translation-invariant component
  (ΣF ≈ 3e-4 Ha/Bohr for O; the energy ripple is of order ΣF·h/4 ≈ 2e-5 Ha) that an exactly
  translation-invariant model **cannot fit**. This is the surrogate's irreducible force
  residual and it is *above* the spec's noise figure — see C2. It is also a feature: the
  surrogate is smoother than its teacher under translation, and ΣF = 0 holds to rounding.

**A2. Species channels.** The spec is internally consistent but under-specified. Standard ACE
(and what the formula E_i = Σ_k c_{Z_i,k} B_k with φ(r_ij, Z_j) implies) is: **coefficients
indexed by the centre species Z_i; one-particle functions indexed by the neighbour species
Z_j**; body-3 pairs are over the combined index (n, Z_j), symmetric: (n₁,Z₁) ≤ (n₂,Z₂). Write
that out. Two consequences the spec must own:

- The symmetric pair must be stored **once** (n₁ ≤ n₂ within a species, Z₁ ≤ Z₂ across): the
  product is symmetric, so storing both orders gives exactly duplicate columns and a singular
  ΦᵀΦ that only the ridge term hides. MUST (part of A3).
- The "**low hundreds per species**" count is true only for one species. Counting
  P = n_max·S + [n_max·S (n_max·S + 1)/2]·(l_max + 1) per centre species:

  | neighbour species S | body-2 | body-3 | P per centre | total S·P |
  |---|---|---|---|---|
  | 1 (H₂, H₃⁺) | 8 | 144 | 152 | 152 |
  | 2 (LiH, H₂O) | 16 | 544 | 560 | 1 120 |
  | 3 (H/C/O) | 24 | 1 200 | 1 224 | 3 672 |
  | 4 (H/C/N/O) | 32 | 2 112 | 2 144 | 8 576 |

  A 300-sample H₂O pass gives 300 × (1 + 9) = 3 000 rows for 1 120 unknowns; a 20-atom
  H/C/N/O cluster gives 61 rows per sample for 8 576 unknowns — hundreds of samples at
  seconds each before the posterior leaves the prior. With a **total-degree cap**
  n₁ + n₂ + l ≤ D (the standard ACE sparsification; n from 0) the counts fall to 288 (S = 2,
  D = 8), 400 (D = 10), 1 088 (S = 4, D = 8). MUST: correct the count and adopt the cap (D a
  setting, default 10), because the uncertainty cost in C4 scales as P².

**A3. The real-harmonic invariant is written wrongly.** §3 says body-3 = "contractions
A_{n1 l m} A_{n2 l −m} summed over m". That is the **complex**-harmonic form (Σ_m (−1)^m
A_{lm}A_{l,−m} = Σ_m A_{lm}A*_{lm}). With **real** Y_lm, the (2l+1)-vector (Y_{l,−l} … Y_{l,l})
transforms under a real orthogonal matrix, so the rotation invariant is the plain Euclidean
dot product **Σ_m A_{n1 l m} A_{n2 l m}**. Pairing m with −m in the real basis pairs e.g.
Y_{1,1} ∝ x with Y_{1,−1} ∝ y and gives (Σ_j x_j)(Σ_j y_j), which is not invariant. MUST:
fix the formula; add the addition-theorem unit test Σ_m Y_lm(â)Y_lm(b̂) = (2l+1)/(4π)·P_l(â·b̂)
to 1e-12 and a random-rotation invariance test on B_k (B5).

**A4. Constant term / reference energies.** Every feature → 0 as all neighbours leave r_c,
so the model's energy → 0 Ha there, while E_KS(H₂) ≈ −0.985 Ha. Trained on 1.2–2.2 Bohr the
fit represents −0.985 Ha locally and the ridge prior drags it to 0 across 2.2–6 Bohr: a ~1 Ha
cliff, spurious forces of order 0.3 Ha/Bohr, and an "isolated atoms" limit that is wrong by
exactly the atomic energies. MUST: fit **E_KS − Σ_i E_atom(Z_i; σ, h, channel)**, with
E_atom computed by the project's own solver (an isolated-atom solve at the scene's σ, h and
the atom's lowest-energy multiplicity — Z-only physics, rule 1 holds; cache it in the model
header). Then the model learns binding energies (0.1 Ha scale), zero features mean
"non-interacting" (physically right, and it resolves the §8 "descriptors go to zero" worry
for the *separated* case), and the ridge prior is a sensible prior. For charged channels use
neutral-atom references and let the fit absorb the ionisation offset (the channel is a
separate model anyway). A learned per-centre-species constant is an acceptable fallback but
it is collinear with nothing that pins it except the energy rows, which are 1 in 3N+1.

## B. Descriptors and derivatives

**B1. Radial basis.** Chebyshev T_n(x) in a scaled distance with a cosine envelope is fine.
Say which x(r): linear x = 2r/r_c − 1 is adequate here because samples never reach the
repulsive wall (3000 K CSVR does not compress H–H below ~1.1 Bohr); ACE's exponential scaling
x = 1 − 2(e^{−λ(r/r_c − 1)} − 1)/(e^{λ} − 1) concentrates resolution at short range and
would be the thing to try if n_max convergence on the H₂ scan is slow. SHOULD: record the
choice and the n_max-convergence test (H₂ scan residual vs n_max = 4, 6, 8, 10). f_c(r) =
½(1 + cos(πr/r_c)) has f_c(r_c) = f_c′(r_c) = 0, so energies and forces are C¹ at the cutoff
(C² is not needed for Verlet at these dt). Note f_c(4 Bohr) = 0.25 exactly for r_c = 6 — the
OOD test point is well inside the envelope (relevant to C3).

**B2. Real Y_lm and gradients.** Do not differentiate Y_lm(r̂) on the sphere; write
φ_{nlm} = [R_n(r)f_c(r)/r^l]·S_lm(x,y,z) with S_lm the real **solid** harmonics (homogeneous
polynomials of degree l; 16 closed forms for l ≤ 3), whose Cartesian gradients are
polynomials. The 1/r^l is harmless (r ≥ 1 Bohr in any sample). Normalise the real Y_lm to
∫|Y|²dΩ = 1 so the addition theorem test (A3) has the (2l+1)/4π constant; normalisation is
otherwise absorbed by the coefficients, but it matters for the ridge (C1). SHOULD: tabulate
S_lm and ∇S_lm by hand for l ≤ 3, unit-test each against finite differences (1e-8 relative,
step 1e-5 Bohr, f64) *and* against a random rotation (covariance).

**B3. l = 0 and self-pairs.** Body-2 B_n = A_{n00} and body-3 with l = 0, B_{n1 n2 0} =
A_{n1 00}A_{n2 00}, overlap: the latter contains the self-terms R_{n1}(r_ij)R_{n2}(r_ij)
(functions of one distance) plus the genuine angle-independent 3-body cross terms. This is
not double counting — ACE's density trick includes self-interaction deliberately and the
products R_{n1}R_{n2} are not in the span of the R_n — but it is a conditioning hazard for
diatomic-heavy data (many columns that are functions of one variable sampled at few distinct
values). The cure is C1 (standardisation + evidence-chosen λ), not dropping l = 0. COULD: a
note.

**B4. Forces.** F_k = −Σ_i ∂E_i/∂r_k with ∂E_i/∂r_k = Σ_{nlm}(∂E_i/∂A_{nlm})·∂A_{nlm}/∂r_ij·
(δ_jk − δ_ik). Compute the adjoint ∂E_i/∂A once per atom (cost ∝ number of B × (2l+1)),
then one 3-vector per neighbour. Newton's third law is exact: assert **ΣF = 0 and Σ r×F = 0
to 1e-12** — a test the DFT provider cannot pass (egg-box) and a clean witness that the
derivative plumbing is right. MUST as the first derivative test alongside FD.

**B5. Cost for 20 atoms (one thread, f64).** ~15 neighbours per atom within 6 Bohr in a
compact cluster. Per neighbour: 8 R_n + 16 S_lm with gradients ≈ 500 flop; A accumulation
128 × S slots; B products ≈ 2 000 × ⟨2l+1⟩ ≈ 10⁴ flop per atom (S = 4, uncapped); adjoint
the same again. ≈ 4e4 flop per atom → ~1e6 flop per step ≈ 0.2–0.5 ms → **2–5 × 10³ steps/s
for the energy and forces alone**, consistent with the spec's ≥ 10³. H₂: a few μs,
≥ 10⁵ steps/s (the spec's 10⁴ is conservative). **The uncertainty evaluation, not the
forces, is the budget problem** — C4.

## C. Fit and uncertainty

**C1. Conditioning and scaling.** Energies are O(1) Ha, binding energies O(0.1), Chebyshev
values O(1), force-row features O(1) Bohr⁻¹ times O(1), body-3 features quadratic in the
neighbour count (200× larger on a 15-neighbour atom than on H₂). A single λ on raw columns
penalises in arbitrary units and the normal equations square an already poor condition
number (Chebyshev on [0, 6] sampled only on [1.2, 2.2] is a near-monomial family; cond(ΦᵀΦ)
1e10+ is likely). MUST: **standardise columns** (subtract the training mean, divide by the
training standard deviation; keep means/scales in the model) before forming the ridge
system; keep **ΦᵀΦ, Φᵀy, Φᵀ1, Σy², N as the sufficient statistics in f64** (Neumaier or
blocked accumulation — 10⁴ rows is not a hard sum) so standardisation and refits are exact
algebra on the stored sums; solve the standardised (ΦᵀΦ/σ_n² + λI) by **Cholesky in f64**
(P = 560: 6e7 flop; P = 2 144: 3e9 flop ≈ 1 s — acceptable for a between-ticks refit, and
the total-degree cap of A2 makes it 10× cheaper). A QR of Φ is not needed: with
standardisation and λ ≥ 1e-8 the normal equations are fine in f64, and keeping Φ (10⁴ rows ×
2 000 columns = 160 MB) buys nothing the stats don't. Store the sufficient statistics (not
just coefficients) in the `.model` sidecar, otherwise "sample more" after a reload cannot
refit. SHOULD.

**C2. The noise level is wrong by two orders of magnitude.** §4 weights samples by the
solver's path scatter (6e-7 Ha, 5e-6 Ha/Bohr). Those are the *reproducibility* of the
teacher, not the residual the model can reach. The irreducible force residual is set by the
egg-box component (3e-4 Ha/Bohr on O, less on H, **not representable** by a translation-
invariant model) and by SCF lag at interactive tolerance (M3 handoff: ~2e-4 Ha/Bohr first
order at tol 1e-5). Using σ_n = 5e-6 makes the predictive variance overconfident by
(3e-4/5e-6)² ≈ 3 600 in variance (60× in std) and the ridge negligible; the fallback would
then almost never fire where it should. MUST: (a) estimate **σ_n,E and σ_n,F from the fit
residuals** by type-II maximum likelihood (MacKay's fixed point: γ = Σ_k d_k/(d_k + λ),
α = γ/‖w‖², σ² = ‖r‖²/(N − γ)) or, minimally, from the held-out residual RMSE — this also
chooses λ; (b) **sample at tol 1e-6**, not the viewer's 1e-5 (0.1 s vs 0.06 s per H₂ solve;
removes the lag term); (c) weight energy rows per atom (target E/N) so 1 energy row is not
drowned by 3N force rows — the standard ACE practice, and it is what pins the absolute
offset the forces cannot see. The path scatter stays as the *lower bound* on σ_n (a sanity
assert: fitted σ_n,F ≥ 5e-6 and ≤ 1e-2).

**C3. Is the Bayesian variance a trustworthy OOD signal?** For a linear model
φᵀΣφ is exactly the leverage of the query in the standardised feature space: it is large
when φ has a component outside the subspace the data span (variance → 1/λ per unit of
standardised feature there) and small when φ lies inside it, *regardless of whether the
true function is linear in φ there*. So it is a good "have I seen this direction" detector
and a poor "is my model right here" detector. Concretely for the §7 test (H₂ trained on
1.2–2.2, queried at 4 Bohr): f_c = 0.25 and the Chebyshev vector at x = 1/3 is O(1) and
points in a direction the window never populated, so the variance **will** flag it strongly
once λ is evidence-chosen (with a hand-set large λ it might not — another reason for C2a).
Its blind spots are (i) *interpolation across a gap* (trained at 1.4 and 2.4, queried at
1.9: in span, low variance, model possibly wrong — mitigated only by the smoothness of a
degree-8 polynomial, which is real but not a guarantee); (ii) *small features*: after the
A4 reference subtraction, near-zero features mean "separated", which is in-distribution, so
this one is handled; (iii) *compression beyond the data* where the Chebyshev extrapolation
happens to stay near the span. SHOULD add, in order of cost: (1) a **species / charge /
multiplicity key** that must match exactly (spec has it); (2) a **feature envelope** per
centre species — min/max of every standardised feature over the training set, with a 10 %-
of-range margin; any feature outside → fall back. Cost P per atom, interpretable, and it
catches (iii); (3) a **raw-geometry envelope**: fall back if any pair distance is below
0.9× the smallest trained distance for that species pair, or any neighbour count exceeds the
trained maximum — ten lines of code and exactly the kinds of events (collision, new
contact) the user will create by dragging; (4) distance-to-data in feature space is
expensive at 2e5 per-atom vectors × P = 600 (1e8 flop per query) — do it only as a
32-dimensional random projection if (2)–(3) prove insufficient. COULD. The bootstrap
ensemble behind a flag is a fine validation-time check; it is not an OOD signal a linear
model lacks, so do not promote it to the default.

**C4. Uncertainty evaluation cost (the real budget problem).** One quadratic form
φᵀΣφ = ‖L⁻¹φ‖² costs P²/2 flop. Per step the force variance needs the Jacobian row
J_{iα} = ∂φ_total/∂r_{iα} for every atom and component (3N forms) plus one for the energy:
20 atoms × 3 × P²/2 = 9e6 flop at P = 560 (≈ 2 ms) and **1.4e8 flop at P = 2 144 (≈ 30 ms)**
— the 10³ steps/s target is gone at four species. MUST pick (and the spec should state) a
budget: (a) the total-degree cap (A2); (b) **cadence** — evaluate the variance every K steps
(K = 10, and every step when the last value was above half the threshold or the geometry
moved > 0.1 Bohr since the last check); (c) a cheap upper bound per step — the largest
eigenvalue direction only — with the full form on the cadence; (d) a spectral truncation
Σ ≈ Σ_{k≤K} s_k u_k u_kᵀ + s_min·I (K ≈ 50 top-variance directions from one symmetric
eigendecomposition at refit) at K·P per form. (a)+(b) suffice for M5; (d) is the scalable
route. Note the energy-variance threshold (C5) is one form per step and always affordable.

**C5. The fallback threshold.** std_F < max(2e-3 Ha/Bohr, 10 % |F|): the 2e-3 floor is
sane against the physics scales — the H₂ 300 K thermal force is √(k·k_BT) ≈ 1.4e-2
Ha/Bohr (floor = 15 % of it), the egg-box 3e-4 and SCF lag 2e-4 are below it, and the
path scatter 5e-6 is irrelevant. Two fixes: (1) the **relative term needs a cap** — on a
hot collision |F| ≈ 0.5 Ha/Bohr makes 10 % = 5e-2, far too loose exactly where the model is
least trained; use max(2e-3, min(0.1 |F|, 1e-2)) SHOULD; (2) add an **energy criterion**
std_E < 1e-4 Ha (one quadratic form, cheap) because the energy bookkeeping (D1) is what the
user and the suite watch, and force variance can be small on a flat, wrongly-offset region.
SHOULD. Also note that the predictive std includes σ_n: with σ_n,F ≈ 5e-4 (fitted) the
epistemic part allowed by the 2e-3 floor is √(4e-6 − 2.5e-7) ≈ 1.9e-3 — fine; state it so
nobody later "fixes" the floor to equal σ_n.

**C6. Calibration test (missing).** A Bayesian variance that is never checked for
calibration is a readout, not an uncertainty. SHOULD: a unit test that fits a synthetic
linear model with injected Gaussian noise and asserts (i) coefficients recovered to 1e-10
noiseless, (ii) held-out residual RMSE within 20 % of the injected σ, (iii) the 2σ predictive
interval covers ≈ 95 % (90–99 %) of held-out truths, (iv) variance is monotone in distance
from a 1-D training window.

## D. Dynamics integration

**D1. Energy bookkeeping across fallbacks and refits.** The spec lets the DFT-vs-model
energy gap "show up as a jump in E_conserved". Two distinct things are then mixed into the
primary health indicator: the integrator's conservation on whatever surface it is on
(what `drift_per_ps` is for, and what the suite must assert) and the model's error (what the
user wants to see). A refit between ticks is a third source: the whole surface changes by
δE(x), so E_conserved jumps even with no fallback, and after enough refits the least-squares
slope is dominated by them. The per-step kick itself is harmless: a 2e-3 Ha/Bohr force
discontinuity for one step is Δp = 2e-3 × 10.3 a.u. ≈ 0.02, ΔE ≈ p·Δv ≈ 1.5e-5 Ha for H at
300 K. MUST: account for surface changes the way the driver already accounts for the
thermostat's W — **E_conserved is computed on the surrogate's surface at every step** (on a
fallback step the forces are the teacher's but the energy term is the model's prediction;
the inconsistency costs ΔF·Δx ≈ 2e-3 × 1e-3 ≈ 2e-6 Ha per fallback step, negligible), and
at the instant a refit is installed the driver evaluates **both** models at the current
geometry and adds ΔE_refit = E_new(x_t) − E_old(x_t) to an accumulated accounting term, so
E_conserved is continuous and `drift_per_ps` keeps meaning "integrator". The DFT–model
difference at a fallback step is then reported as its own readout, **`teacher_gap`** (last
value and running RMS) — a free, honest measurement of the surrogate's error along the
actual trajectory, and the number the UI copy should point at. A blended switch (w·F_DFT +
(1−w)·F_model) is not conservative and buys nothing; Δ-learning on a cheap baseline is not
available here except for exactly the reference-energy subtraction of A4, which is the
minimal and correct baseline. COULD: a nuclear-repulsion-plus-atomic-densities
electrostatic baseline later.

**D2. Refits and reproducibility.** A refit changes the surface mid-trajectory. `Rewind`
retraces only within one model epoch; the spec should say so, make refits **synchronous
and logged** (a refit is a history event like `discontinuity`), and pin the model during a
rewind. SHOULD. Determinism of the fit itself across accumulation orders is not promised
(f64 sums); determinism of MD given a fixed model is, and is testable.

**D3. What the NVE suite should assert on the surrogate.** The surface is smooth, analytic
and f64; Verlet's drift on it is ~1e-9 Ha/ps. The spec's 1e-6 Ha/ps over 10 ps (40 000 steps
≈ seconds) is fine and tight enough to catch an energy–force inconsistency. Add: the
fluctuation equals Verlet's exact h²/4·E_mode **with no scatter term** (assert within 20 %
— a positive test that the model is a gradient field; the M4 bound needed 3× and a scatter
allowance); **rewind to 1e-9 Bohr** (M4: 5.8e-6 on the GPU); ΣF = 0 to 1e-12 along the run;
and a run **with fallbacks enabled** on a model trained deliberately short of the amplitude
reached, asserting that E_conserved (with D1 accounting) still drifts < 1e-5 Ha/ps while
`teacher_gap` RMS < 1e-3 Ha and the fallback fraction falls to zero after refits
(active learning end-to-end).

**D4. `Display`.** Fading the last DFT density with a timestamp is honest and sufficient;
on a fallback step the density is fresh — show it, then fade again. Say in the spec that
the M6 bond graph (read off the density) is therefore unavailable in estimated mode except
at fallback steps. COULD: a superposition of the project's own isolated-atom densities as a
clearly labelled "not solved" cartoon cloud — it is Z-only, it is cheap, and the user will
otherwise watch bare spheres for thousands of steps.

## E. Honesty and rules

**E1. Rule 1.** Nothing in the spec hand-sets per-element behaviour: coefficients are
indexed by Z categorically and learned; r_c, n_max, l_max, λ, thresholds are global;
reference energies (A4) are computed by the solver. Two traps for the implementer, to be
written into the spec as rules: (i) **sampling ranges must be derived from the scene**
(bond scans over fractions of the *current* distances, e.g. 0.8–1.5×; displacement σ_r
global), never from covalent/van-der-Waals radius tables — the obvious shortcut and a
violation; (ii) the `.model` header records solver version, σ, h, tol, species, channel and
the reference energies, and the fit accepts samples **only from `ForceProvider` evaluations
of this project's solvers** — a model file made elsewhere is the one loophole in "learned
from Z-only physics", and it is a social one, closed by provenance in the header. MUST (i),
SHOULD (ii).

**E2. "The physics cannot be made cheaper" (§1) is not true as stated.** The M3 handoff
lists latency levers (≈ 200 readbacks per MD step, per-dispatch buffer creation) worth
2–5× on DFT-MD itself. The surrogate is still the right call — it is the only route to
1000× — but the spec should not overclaim. COULD: one sentence.

**E3. Validation targets (§7).** Achievable for H₂, and the suite should be tiered:

- **H₂ (1-D, 152 coefficients, uncapped):** E to **1e-5 Ha**, forces to **2e-4 Ha/Bohr**
  (above the H egg-box floor), ω_surrogate-MD vs the M4 pinned 3236.9 cm⁻¹ within **0.5 %**
  (the force target then guarantees it; the 1 % in the spec would pass with a visibly wrong
  k). The training set is a scan + 300 K samples, so the test range must be stated as the
  scan range and the MD amplitude must stay inside it.
- **Second system: H₃⁺, not H₂O.** MUST. Task 7 showed the σ = h = 0.25 H₂O surface has
  its minimum near linear, 0.13 Ha below the preset, and the warm GPU SCF stalls near 170°:
  the teacher cannot produce the hot-MD samples §4 asks for, and "reproduces H₂O" on that
  surface means reproducing a near-linear molecule, which is not what anyone reading the
  test will think it means. H₃⁺ (M4's own choice) exercises exactly what H₂ cannot — the
  3-body features through the e′ bend and asymmetric stretch — at 80 ms per sample.
  Targets 1e-4 Ha / 2e-3 Ha/Bohr RMSE, 5e-3 max. H₂O enters the suite when the cold-retry
  on non-convergence (M4 handoff item 2) has landed, and then on the surface it is on.
- **LiH** is the two-species test (S = 2, 560 coefficients, the channel bookkeeping); no
  physics beyond H₂ — a scan + perturbations suffices, same targets as H₂ loosened 3×.
- **OOD test** as specified (R = 4 Bohr) plus a *compression* case (R = 1.0 Bohr) and a
  *new-species* case (add Li to an H₂ model → fallback by key, no evaluation of the model).
- **Speed** reported with the uncertainty cadence stated, for S = 1, 2, 4.
- **Rule 7 line:** the surrogate inherits the σ = 0.25 surface's gaps (ω −26.5 % to
  experiment, H₂O near-linear) and adds its fit error; the readout's uncertainty is model
  uncertainty only. Spec has this; keep it, and add that the egg-box is *removed*, not
  learned, so surrogate and teacher differ by it by construction.

**E4. A cheaper first rung?** A per-scene harmonic/Morse surrogate is not cheaper in any
way that matters: it needs the same sample plumbing, provider, fallback and UI, and it has
no angles, so it would be thrown away at H₃⁺. The correct de-risking is **inside ACE**:
land body order 2 only (pair terms, 8 coefficients per species pair) end to end on H₂ —
descriptors → fit → variance → provider → fallback → readout → ω test — before body
order 3 exists. That is a strict subset of the planned code and it exercises every risky
seam except the angular derivatives. SHOULD: order the plan that way (F).

## F. Plan guidance

New crate `surrogate` (depends on `units` only; pure f64, fast in debug) for the model, fit
and variance; `SurrogateKs` lives in `sim/src/provider.rs` beside the others (`sim` already
depends on `qm`/`qm-gpu`). Each task's first failing test is the one named.

1. **Radial basis, cutoff, real solid harmonics with gradients** — Fable. First test: FD vs
   analytic for R_n·f_c and for each S_lm (l ≤ 3) at random points, 1e-8 relative; addition
   theorem Σ_m Y_lm(â)Y_lm(b̂) = (2l+1)/4π·P_l(â·b̂) to 1e-12.
2. **Body-2 descriptors and forces, species-channelled** — Fable. First test: translation
   and permutation invariance to 1e-14; FD vs analytic dB/dr 1e-8; ΣF = 0 and Σr×F = 0 to
   1e-12 on a random 5-atom, 2-species cluster.
3. **Reference energies, sample store, standardised ridge fit, evidence σ_n/λ, predictive
   variance** — Fable. First test: exact recovery of a synthetic linear model (1e-10); the
   calibration test of C6; the H₂ scan (synthetic Morse, then real samples from the
   `validate` common fixtures) fitted to 1e-6 Ha with body-2 only.
4. **`SurrogateKs` provider, fallback policy, D1 accounting, `EvalStats.source/
   uncertainty/teacher_gap`** — Fable. First test: window-trained H₂ model queried at
   R = 4 Bohr returns `Fallback`, calls the teacher once, appends the sample; after refit the
   same R returns `Surrogate`; E_conserved continuous across a forced refit to 1e-12.
5. **Sampling pass on the worker (perturbations, hot CSVR bursts, scene-derived scans),
   isolated-atom reference solves cached, `.model` sidecar with sufficient statistics and
   provenance header** — Opus. First test: sidecar round-trip bit-exact; a pass over H₂
   with a scripted fake teacher yields the requested counts and tags; cancellation leaves a
   usable partial set.
6. **Body-3 descriptors and forces, symmetric (n,Z) pair indexing, total-degree cap** —
   Fable. First test: random-rotation invariance of every B_k to 1e-12 (this is the test
   that catches A3); FD vs analytic 1e-8; H₃⁺ fixture fitted to 1e-4 Ha / 2e-3 Ha/Bohr.
7. **Validation `surrogate.rs`** — Fable. First test: H₂ ω within 0.5 % of the M4 pinned
   value; then E/F reproduction (tiered per E3), NVE 10 ps < 1e-6 Ha/ps with the exact
   h²/4 fluctuation, rewind 1e-9, OOD ×3, the fallback-enabled active-learning run, the
   speed table. References and bound derivations to `references.md`.
8. **UI: "estimated" radio, sampling progress, readouts (source, std vs threshold,
   samples, last refit, fallback fraction, teacher gap), sparkline colouring, density fade
   label, "sample more"/"forget"** — Opus. First test: an `app::sim` worker test that
   enabling "estimated" with no model starts a pass and the snapshot carries its progress;
   the viewer checklist gains the estimated-mode items.
9. **Performance: uncertainty cadence/spectral form, S = 4 20-atom timing, refit timing**
   — Opus. First test: a benchmark that reports (not asserts) steps/s for S = 1, 2, 4 and
   asserts the variance cadence never lets more than K steps pass between full checks.
10. **Physics review + M5 handoff note** — Fable (physics-reviewer), after 7.

Dependencies to record in the plan: the warm-SCF cold-retry (M4 handoff item 2) gates any
sampling on H₂O; the M4 `Dynamics` accounting hook for ΔE_refit is a small `sim` change
that belongs to task 4, not to the UI task.

## Summary of graded recommendations

MUST
- A3: body-3 invariant with real harmonics is Σ_m A_{n1lm}A_{n2lm}; symmetric (n,Z) pairs
  stored once; addition-theorem and rotation tests.
- A4: fit E_KS − Σ E_atom(Z) with solver-computed isolated-atom references.
- C2: σ_n from residuals/evidence (not the 5e-6 path scatter); sample at tol 1e-6; per-atom
  energy weighting. C1: standardise features; f64 sufficient statistics + Cholesky.
- A2/C4: correct the descriptor count (560 / 2 144 per centre species at S = 2 / 4); add
  the total-degree cap; budget the P² uncertainty evaluation (cadence or spectral form).
- D1: E_conserved on one surface with ΔE_refit accounted like W; `teacher_gap` readout.
- E3: second oracle is H₃⁺; H₂O conditional on the cold-retry; tier the targets (H₂ 1e-5 Ha /
  2e-4 Ha/Bohr / ω 0.5 %).
- E1(i): sampling ranges derived from the scene, never from radius tables. B4: ΣF = 0 test.

SHOULD
- A1: raise l_max before enabling body order 4; document non-locality/spin/egg-box limits.
- B1/B2: record x(r), n_max convergence; solid-harmonic form with tabulated gradients.
- C3: feature envelope + raw-geometry envelope as OOD guards; C5: cap the relative threshold
  at 1e-2 and add an energy-std criterion; C6: calibration test.
- D2: synchronous, logged refits; rewind pinned to a model epoch. D3: the surrogate NVE
  assertions listed. E1(ii): provenance in the sidecar. E4/F: body-2 end to end first.

COULD
- B3 note on l = 0 self-pairs; C3(4) projected distance-to-data; D1 electrostatic baseline;
  D4 atomic-density cartoon cloud; E2 soften "cannot be made cheaper".
