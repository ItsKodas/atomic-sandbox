# M5 — Surrogate ("estimated") force provider: design

Status: reviewed 2026-10-13 (physics review: `docs/superpowers/notes/2026-10-13-m5-spec-review.md`, verdict "sound with changes" — all MUST items folded in below). Child of `2026-10-12-atomic-sandbox-design.md` (§5.1 `ForceProvider`, §11 roadmap "ML potentials loaded as data"). User decision 2026-10-13: this is the milestone after M4; the tools list moves to M6.

## 1. Purpose

M4 delivered honest Born-Oppenheimer MD: every step is a full Kohn-Sham solve, ~0.1 s for H₂ on a 49³ grid, so the sandbox runs at ~2.5 fs of simulated time per wall second. A bond vibration takes four seconds to watch; nothing larger than a few atoms is interactive. The physics cannot be made cheaper; it can be *learned*. M5 adds a second `ForceProvider` that reproduces the DFT energy surface where it has been taught it — at microseconds per step — and says clearly when it is guessing.

Success looks like: press "estimated", and the same H₂ vibrates at thousands of fs per wall second with the same frequency the DFT run gave; drag it somewhere the model has never seen and the panel shows the uncertainty climb, the step quietly falls back to DFT, and the model gets better. Nothing per-element is hand-set; every coefficient is learned from Z-only physics produced by this project's own solver.

## 2. Non-goals

- Not a general-purpose pretrained potential. The model is fitted to this user's scenes, on this solver, at the scene's σ and h, and it is labelled as such.
- Not a replacement for the DFT provider. DFT remains the reference, the fallback, and the teacher.
- No external ML frameworks (rule 2). Training is linear least squares written in Rust; no autodiff library.
- No pseudopotentials, periodic cells, or elements beyond Z = 10 (unchanged from M4).

## 3. Model

**Atomic cluster expansion (ACE)-style linear model.** The total energy is a sum of atomic energies, each a linear function of body-ordered, rotation- and permutation-invariant descriptors of the atom's neighbourhood:

E = Σ_i E_i,  E_i = Σ_k c_{Z_i,k} B_k(neighbourhood of i)

- **Neighbourhood:** atoms within a cutoff r_c (default 6 Bohr, smooth cosine envelope f_c(r) so energies and forces go to zero continuously at r_c).
- **One-particle basis** φ_{nlm}(r_ij, Z_j) = R_n(r_ij)·Y_lm(r̂_ij)·δ-channel on Z_j: radial functions R_n are a fixed orthogonal set (Chebyshev in a scaled distance) with n ≤ n_max (default 8), real spherical harmonics with l ≤ l_max (default 3). The Z_j channel means the model keeps separate coefficients per neighbour species — a categorical index, not a hand-set number, so rule 1 holds: the per-element content is *learned*.
- **Atomic basis** A_{nlm,Z} = Σ_j φ_{nlm}(r_ij, Z_j) (density projection; cost linear in neighbours).
- **Invariant products** B_k: body order 2 (A with l = 0), 3 — with *real* spherical harmonics the rotation invariant is Σ_m A_{n1 l m} A_{n2 l m} (the A_{lm}A_{l,−m} form belongs to the complex basis and is NOT invariant with real Y_lm); symmetric (n₁Z₁, n₂Z₂) pairs stored once — and optionally 4 (Clebsch–Gordan coupling; default off; try raising l_max before body-4). Descriptor counts are not small: 560 per centre species at 2 species and 2 144 at 4 species for n_max 8 / l_max 3, so a **total-degree cap** (n₁ + n₂ + l ≤ D, default D = 10) is part of the basis definition. Tests: addition theorem for the Y_lm, and invariance of every B_k under random rotations/permutations to 1e-12.
- **Forces** are the analytic gradient −∂E/∂r_i, implemented by hand (the chain rule through B_k, A, R_n, Y_lm). Energy–force consistency is what makes NVE conservation and `Rewind` work on the surrogate; it is tested to rounding by finite differences.
- **Reference energies:** the model fits the *binding* energy E_KS − Σ_i E_atom(Z_i), where E_atom(Z) is the isolated-atom energy computed by the project's own solver at the same grid/σ (Z-only, cached per settings). Without this the features → 0 far from any neighbour would send the energy to zero instead of to the separated atoms — a ~1 Ha cliff between the trained window and r_c.
- **Why linear:** a linear model is fitted by one least-squares solve (seconds), gives a closed-form predictive variance (below), is deterministic, and is small enough to write and review in Rust. Nonlinear message-passing models are better at scale and are a later rung, behind the same interface.

## 4. Training data and the fit

- **Samples** come only from the project's own solvers (`GpuKs` default; `CpuKs` allowed) at the scene's grid, σ and tolerance: each sample is (positions, Z, charge, multiplicity) → (E_KS, forces, EvalStats). Sources: (a) the live DFT run's own steps (every M4 evaluate is a free sample); (b) a **sampling pass** the user triggers: random displacements of the current scene (Gaussian, σ_r 0.05–0.3 Bohr), short hot DFT-MD bursts (CSVR at 1000–3000 K, a few hundred steps), and scans along each pair within r_c whose ranges are derived from the scene itself (fractions of the current separations), never from any element radius table (rule 1); (c) fallback steps (below). Samples are tagged with the solve settings; a scene on a different grid/σ starts a different model (the surface is different).
- **Fit:** ridge-regularised weighted least squares on energies (weighted per atom) and force components jointly (forces are 3N equations per sample and carry most of the information). Samples are taken at tol 1e-6 (the interactive 1e-5 carries ~2e-4 Ha/Bohr of SCF lag). Features are standardised before the solve; sufficient statistics ΦᵀΦ, Φᵀy are accumulated in f64 and solved by Cholesky; refit after new samples is one solve. The **noise level σ_n is estimated from the fit residuals / marginal likelihood (evidence)**, not from the solver's path scatter: the real floor is the egg-box force (ΣF ≈ 3e-4 Ha/Bohr, not representable by a translation-invariant model) and SCF lag, ~60× above the 5e-6 scatter — using the scatter would make the uncertainty 60× overconfident.
- **Uncertainty:** Bayesian linear regression's predictive variance σ²(x) = σ_n² + φ(x)ᵀ Σ φ(x) with Σ = (ΦᵀΦ/σ_n² + λI)⁻¹ — one matrix, no ensemble, exact for a linear model. Reported per atom (force variance) and per step (energy variance). Cost: the φᵀΣφ product is O(P²) per atom — ~30 ms/step for 20 atoms at four species — so it is evaluated on a **cadence** (every k steps, default 10, and always on the first step after an edit) or with a spectrally truncated Σ; the per-step force prediction itself stays O(P). Out-of-distribution guards that the variance alone misses: a distance-to-data term (nearest training descriptor) and an unconditional fallback for any (species, charge, multiplicity) combination the model has not seen. A **bootstrap ensemble** (e.g. 8 refits on resampled data) is kept behind a setting as an independent calibration check during validation.
- **Charge/spin:** the model is per (charge, multiplicity) channel; electrons are not modelled explicitly. A scene edit that changes the electron count invalidates the model for that scene (falls back to DFT until retrained). Documented limitation: the surrogate cannot see electronic state changes that DFT can (level crossings during hot dynamics) except through the data it was given.

## 5. The provider and the fallback policy

`SurrogateKs: ForceProvider` wraps a `Model` plus a `Box<dyn ForceProvider>` teacher (the DFT provider):

- `evaluate(scene)`: compute descriptors, predict E and F and their variances. If the normalised force uncertainty of every atom is below a threshold (default: predicted force std < max(2e-3 Ha/Bohr, 10 % of |F|)) → return the prediction with `EvalStats` marking `source = Surrogate`, `uncertainty = …`. Otherwise → call the teacher, append the sample, mark `source = Fallback`, and schedule a refit (cheap; runs on the worker thread between ticks, never on the render thread).
- The panel's time controls are unchanged; the driver does not know which provider answered. `EvalStats` gains `source` and `uncertainty` so the readouts and the sparkline can colour steps by source.
- Energy bookkeeping: `E_conserved` is computed on ONE surface — the surrogate's — even on fallback steps (the surrogate energy is still evaluated and used for the invariant; the DFT forces move the atoms). The difference DFT − model on a fallback step is published as a separate `teacher_gap` readout, and the energy change caused by a refit (the surface moved under the atoms) is booked like thermostat work (`ΔE_refit`). The drift readout then measures the integrator on the model surface, the teacher gap measures the model, and the surrogate NVE assertion in §7 is testable.
- `reset()`/`discontinuity()` pass through to the teacher; the model persists (it is a function of positions, not of solver state).
- `Display`: the surrogate has no density. The viewer shows the last DFT density (greyed, labelled "from the last DFT step at t = …"), or none. Visual honesty beats a stale cloud: default is to fade it.

## 6. UI

- Solver radio gains **"estimated"** (alongside CPU/GPU). Enabling it with no model prompts a sampling pass (progress bar, cancellable, running on the worker; the user can keep editing while it samples).
- Readouts: surrogate/fallback indicator per step, uncertainty (Ha/Bohr) with the threshold, number of training samples, last refit time, fraction of steps that fell back over the last 100.
- A "sample more" button and a "forget model" button. Models are saved next to the scene file (`.model` sidecar, binary + a JSON header with grid/σ/tol/species/charge) and reloaded with it.
- Speed: with the surrogate answering, `steps/tick` can be raised to the 1000 cap; the fs/s readout shows the real number.

## 7. Validation (`validate/tests/surrogate.rs`, Fable)

- **Reproduces its teacher (tiered):** H₂ — train on a bond scan + 300 K DFT-MD samples; E(R) within 1e-5 Ha over the trained range, forces within 2e-4 Ha/Bohr, surrogate-MD vibration frequency within 0.5 % of the DFT-MD value (M4 suite). H₃⁺ (the second system — σ = 0.25 water is near-linear and its warm SCF stalls, Task 7) — E within 1e-4 Ha, forces within 2e-3 Ha/Bohr, bend/stretch frequencies within 2 %. H₂O only once the cold-retry solver item lands, same targets as H₃⁺.
- **Energy–force consistency:** first derivative tests are ΣF = 0 and Σ r × F = 0 to 1e-12 on random clusters (translation and rotation invariance of the model), then finite-difference forces vs analytic to 1e-8 relative; NVE drift on the surrogate < 1e-6 Ha/ps over 10 ps (it is a cheap, smooth surface — the bound is tight).
- **Knows what it doesn't know:** on geometries outside the training range (R = 4 Bohr for an H₂ model trained on 1.2–2.2) the predictive std exceeds the threshold and `evaluate` falls back; after adding the fallback samples, the same geometry is answered by the surrogate.
- **Speed:** steps per second reported (not asserted): expected ≥ 10⁴ for H₂, ≥ 10³ for a 20-atom cluster.
- **Honesty gap documented:** the surrogate inherits every DFT approximation (smearing, LSDA) and adds its own fit error; the panel's uncertainty is a *model* uncertainty, not a physics one.

## 8. Risks

- Descriptor derivatives are the error-prone part (signs in the spherical-harmonic gradients). Mitigation: the FD-vs-analytic test is the first test written; the Y_lm and R_n derivatives get their own unit tests.
- Linear ACE at body order 3 may underfit bent/branched geometries (water's bend). Mitigation: body order 4 behind a flag; measure on H₂O before deciding.
- Extrapolation confidence: a linear model's variance can be small far from data if the descriptor goes to zero (isolated atoms). Mitigation: add a distance-to-data term (nearest-neighbour descriptor distance) to the fallback test, and always fall back on species/charge the model has not seen.
- The fallback makes the frame rate bursty (a DFT step mid-run). Mitigation: show it; let the user choose "never fall back (unsafe)" only with a visible warning.

## 9. Model policy

Fable: descriptors + analytic derivatives, the fit and its uncertainty, the fallback policy, validation. Opus: the provider plumbing, the sampling pass on the worker, UI, sidecar files, README.
