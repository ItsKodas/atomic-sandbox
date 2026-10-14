# Physics review — precision spec and plan (09b79c9)

**Date:** 2026-10-15. **Reviewed:** `docs/superpowers/specs/2026-10-15-precision-design.md`, `docs/superpowers/plans/2026-10-15-precision.md` at 09b79c9.
**Verdict:** "ready with fixes" — Stages 1–3 can start; Stage 4 not until M1/M2 are folded in. Rulings (the controller's) are in brackets. How each item was applied: spec §18.

## MUST

- **M1** (§5.2.1, Task 4.4): the KB operator ⟨r|V_KB|r′⟩ = E_KB β(r)β(r′)ΣYY* needs a radial × angular integral of Ψ; a 12-point sphere rule only evaluates the semilocal form δV_l(r)P_l at the electron's radius. `nonlocal_quadrature_reproduces_the_kb_operator` is ill-posed (for f = R·Y_lm, E_KB⟨β|f⟩² ≠ ∫δV_l R² r² dr unless R ∝ φ_l). [Ruling: use M7's semilocal V_l^ion in QMC (standard practice); test the quadrature against ∫δV_l R² r² dr (degree 2l ≤ 4, exact); strike "the grid orbitals are the correct pseudo-orbitals of the Hamiltonian DMC solves" — KB on the grid vs semilocal in DMC is a trial-function difference only.]
- **M2** (§5.2.1, Tasks 4.4/4.7): M7 potentials are unscreened with v_xc[ρ_v + ρ_pc] from a spin-restricted reference relying on NLCC; QMC has no ρ_pc, so V_ion as-is leaves valence electrons without the core–valence XC the unscreening removed (QMC potentials like BFD, Burkatzki et al. 2007, are HF-generated without NLCC). [Ruling: add an M7 recipe variant `nlcc off` with an HF reference, in `recipe_hash` and the `pseudo` key line; the trial orbitals use the same variant; Task 4.7 becomes unconditional; add this potential decision to the Task 4.0 go/no-go; also add one line to `docs/superpowers/specs/2026-10-14-pseudopotential-design.md` (§8 or §3) noting the QMC variant as a future requirement.]
- **M3** (Task 3.2): "ΣF = 0 to 1e-10" fails for a space-fixed frozen floating basis (ΣF_I = −dE/d(rigid translation of nuclei) at fixed centres ≠ 0 for an incomplete basis; the frozen surface is not rotation-invariant → NVE momentum drift). [Ruling: attach the centres to their nuclei and add the Pulay term Σ_k ∂E/∂s_k·∂s_k/∂R_I (analytic, same derivatives as gradient optimisation) → ΣF = 0 exactly; name the ion–ion term in F_I = −cᵀ∂H/∂R_I c (cᵀSc = 1); ECG force labels in Δ fits (§7.3 "σ = 0") are frozen-basis derivatives — give them a measured σ_F against central differences of warm-start re-optimised energies.]
- **M4** (§7.4, Task 5.2): χ² cannot catch understated σ_i when s_E is evidence-chosen (s_E rises to absorb 3× understated noise → reduced χ² ≈ 1; `chi_square_trust_rejects_overconfidence` fails as designed). [Ruling: flag s_E² ≫ median σ_i² ("label errors understated or model underfits"); rely on `error_bars_are_honest` for calibrating σ_i; keep the overstated-case check.]

## SHOULD

- **S1** (§3.3/§9, Task 4.6): the 4 kcal/mol water bound is extrapolated (cited studies used HF/DFT nodes with all-electron or BFD potentials). [Ruling: hard gate = internal consistency (τ → 0, σ, decomposition); |ΔTAE| ≤ 4 is the stage-4 acceptance *reported* to the user, pinned "measured first".]
- **S2** (§7.1): pick the O reference multiplicity by a DMC scan, not the LSDA S9 scan.
- **S3** (§6.2): the last frame can be 30 s newer than `state.ckpt` — resume truncates to the checkpoint frame, or checkpoint with each refresh/label.
- **S4** (§6.5): "R1 ECG ±1 µHa" is accuracy at the oracle geometry — show the measured frozen-basis error within a frozen segment.
- **S5** (Task 4.4): "E_TM(τ) ≥ E_TM(τ→0) − 3σ" does not follow from Casula (the T-move bound is relative to the exact E₀ of the PP Hamiltonian; time-step error has either sign — report it); build the 30 Bohr dimer trial as a product of two single-atom trials (one grid solve would need a ~44 Bohr box at h = 0.1).
- **S6** (Task 5.6): add the measured τ bias to `water_live_reproduces_dmc_atomization`.
- **S7** (Task 3.1): pin He at −2.903724377034 with the variational bound.
- **S8** (§3.3): Nemec 3.2 kcal/mol / 95 % and Petruzielo 3.0/2.1/1.2 are unverified (Petruzielo's single-determinant figure possibly 3.1); also unverified: Pavanello/Pachucki HeH⁺ digits, Nemec's 40 CPU-h, Ma Z^5.2, Filippi–Umrigar R_e — verify with WebSearch/WebFetch or mark "unverified"; state which nodes and potentials each study used.

## NOTE

- Task 4.6's all-electron h = 0.05 test has Zσ = 0.4 for O — core still distorted, Ma's cusp correction assumes correct orbitals outside r_c → a bound, not a decomposition.
- E_DFT labels carry the egg-box (not invariant) → a noise floor in s_E, report it.
- §6.5 R2 shows σ_Δ while §7.3's total is √(σ_base² + σ_Δ²) — show the total, and the force σ from ∇φᵀΣ∇φ (Δ forces are never checked against DMC).
- `DELTA_ANSWER_STD` "1 mHa total and 0.5 mHa per atom" — say which binds.
- Task 4.5's 1e-5 Ha absolute E_L tolerance fails near nuclei in f32 → relative to |V_en|.
- The jobs "free X MB" footer never touches `models/precise/` (user-data §5.5).

## Reviewer's estimate (for the spec's honesty section)

H₂O TAE_e at the end of Stage 5, with M1/M2 fixed, underbound by ~3 ± 1.5 kcal/mol (fixed node from single-determinant LSDA nodes ~1.5–3; LSDA-TM no-NLCC PP ~0.5–2 unverified; T-moves vs locality ~0.3; core–valence 0.38; the live Δ adds ≤ 0.6) — ~60–65 % chance of ≤ 4; with M7's NLCC/KB potentials as-is, unpredictable, 5–10 plausible.
