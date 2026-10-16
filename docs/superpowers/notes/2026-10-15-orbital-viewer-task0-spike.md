# Orbital viewer — Task 0 spike: grid shell splitting, spin noise, display-eigensolve cost

Measure-only spike for the orbital-viewer plan (`docs/superpowers/plans/2026-10-15-orbital-viewer.md`, Task 0), run 2026-10-14 on branch `orbital-viewer` (from main 66a714b). The code is `crates/validate/tests/orbitals_spike.rs`: three `#[ignore]`d `diag_` tests that print tables and assert nothing. No library code was changed.

```
cargo test -p validate --release --test orbitals_spike -- --ignored --nocapture --test-threads=1
```

**Four results contradict the spec or plan. Each is flagged ⚠ below and listed in "For the controller" at the end:**

- In Ne, the 2s level lies **above** the 2p shell. The O singlet therefore comes out as 1s² 2p⁶.
- The display eigensolve **does not reach the effective tolerance** within the SCF's 200-iteration cap in several cases.
- Its cost is 4–50 times a warm SCF, so it can never run during play under `UNOCCUPIED_BUDGET` = 0.1.
- With a 1000-iteration cap, one display solve lost the occupied ground state (H₂ at 97³).

## Method (common to all three)

- **Solver.** GPU solver `qm_gpu::GpuSolver`, LSDA, σ = h, `ScfSettings::for_grid`. Mixing 0.3, DIIS 6, `eig_tol` 1e-6, `max_iter` 200.
- **SCF tolerance.** Either the app's `INTERACTIVE_TOL` 1e-5 or the validate tolerance 1e-6, as stated per table.
- **Grids.** `Grid::cube(n, h)` with odd n, so the origin is a grid point.
- **Starts.** Every lone-atom solve is cold, using a fresh solver.
- **CPU-side analysis.** Eigenvalues are the solver's own `GpuState::evals_*`. Densities and orbitals are read back as f32, and every reduction is finished in f64 on the CPU.
- **Angular character.** W_l comes from the linear moments c_lm = ⟨ψ|Y_lm⟩ about the nucleus, for l ≤ 3. This is spec §5.2 with the real unit harmonics written out in the test. The p set is the orbitals with W_1 ≥ `L_CHAR_MIN` = 0.9; it is not identified by index (see 1b).
- **"Spread".** The maximum minus the minimum eigenvalue of the p-dominant orbitals in one channel. The tables show the larger of the two channels.

## 1. 2p spread versus position (lone Ne, lone O singlet)

**Setup.**
- Grids: 57³ at h = 0.25 (box 14.00 Bohr) and 95³ at h = 0.15 (box 14.10 Bohr).
- Nuclear positions:
  - a grid point (0, 0, 0);
  - a cube centre (h/2, h/2, h/2);
  - "off-grid" (0.37h, −0.21h, 0.12h);
  - (h/2, 0, 0).
- Tolerances: 1e-5 and 1e-6. The spreads agree between the two tolerances to the digits shown; the 1e-6 run is quoted.

### 1a. Spreads (Ha)

| atom | grid | grid point | cube centre | off-grid | (h/2,0,0) | 1s shift between sites | 2s shift between sites |
|---|---|---|---|---|---|---|---|
| Ne (5↑5↓) | 57³, h 0.25 | 3.8e-8 | 1.4e-7 | **4.97e-4** | **7.05e-4** | 2.25e-3 | 3.4e-4 |
| Ne | 95³, h 0.15 | 7.0e-8 | 3.0e-8 | 8.24e-5 | 1.17e-4 | 7.7e-4 | 8.2e-5 |
| O singlet (4↑4↓) | 57³, h 0.25 | 3.3e-8 | 2.9e-8 | 1.10e-4 | 1.55e-4 | 1.03e-3 | (2s unoccupied) |
| O singlet | 95³, h 0.15 | 5.3e-8 | 2.2e-7 | 1.49e-5 | 2.11e-5 | 3.1e-4 | (2s unoccupied) |

**On cube-symmetric sites the p shell is degenerate to f32/SCF noise (≤ 2.2e-7 Ha), as the O_h argument says.** The splitting off those sites scales as about h^3.5 for Ne (7.05e-4 → 1.17e-4) and h^3.9 for O.

The measured self-consistent spreads are much smaller than the reviewer's screened one-electron model predicted:

| quantity | measured | model | measured is smaller by |
|---|---|---|---|
| Ne 2p spread, off-grid | 4.97e-4 Ha | 1.3e-2 | ≈ 26× |
| Ne 2p spread, (h/2,0,0) | 7.05e-4 Ha | 2.0e-2 | ≈ 28× |
| O 2p spread, off-grid | 1.10e-4 Ha | 2.5e-3 | ≈ 23× |
| O 2p spread, (h/2,0,0) | 1.55e-4 Ha | 3.7e-3 | ≈ 24× |
| 1s shift between sites, Ne | 2.25e-3 Ha | 0.50 Ha | ≈ 220× |
| 1s shift between sites, O | 1.0e-3 Ha | 0.26 Ha | ≈ 250× |

So spec §2 item 5 ("the 1s of O moves by tenths of a Ha between sites") should be corrected to "≈ 1 mHa (O) and 2 mHa (Ne) at h = 0.25".

Even so, the measured spreads still exceed τ at the validate tolerance (`TAU_MIN` = 1e-4 Ha) by up to 7×. At the interactive tolerance (τ = 1 mHa) they reach 0.7 τ. **MUST 1 (group a lone atom's shells by angular character, not by energy gap) therefore stands.** The splitting is smaller than the review assumed, but it is not below τ at both tolerances.

**Angular character on real data.** Every p orbital at every site and grid has W_1 = 1.0000 (to 4 decimals). The 2s has W_0 = 1.0000. The 1s has W_0 = 0.9973 at the worst site (Ne 57³ off-grid: the smeared 1s is slightly distorted by the egg-box), and 0.9985–1.0000 elsewhere. The margin over `L_CHAR_MIN` = 0.9 is very large.

### 1b. ⚠ The 2s lies above the 2p (smeared nucleus, Zσ ≥ 2)

Ne eigenvalues per channel at the grid point (the other sites agree to the shifts above):

| grid | 1s (W_0) | 2p ×3 (W_1) | 2s (W_0) |
|---|---|---|---|
| 57³, h 0.25 | −5.900786 (1.0000) | −0.845152 | **−0.298213** |
| 95³, h 0.15 | −9.664869 (1.0000) | −0.819040 | **−0.394461** |

- **Energy order.** It is **1s, 2p, 2p, 2p, 2s**. For comparison, the point-nucleus LSDA values are 2s ≈ −1.32 Ha and 2p ≈ −0.50 Ha.
- **Cause.** The σ = h Gaussian nucleus (Zσ = 2.5 for Ne at h = 0.25, 1.5 at h = 0.15) removes the deep potential that the penetrating 2s relies on. It also makes the 1s far wider than the physical one (−5.9 / −9.7 Ha against ≈ −30.3), so the 2p is screened less and sits deeper than the physical 2p.
- **Not a grid artefact.** This is a real property of today's smeared-nucleus functional, and it persists at h = 0.15.
- **O singlet (4↑4↓).** For the same reason the aufbau fills **1s and all three 2p** in each channel, and the 2s is unoccupied. Measured: ↑ = ↓ = {−4.3646 (s), −0.36918 ×3 (p)} at 57³ and {−7.0697 (s), −0.37175 ×3 (p)} at 95³. So the "O singlet" is a **closed-shell 1s² 2p⁶** configuration, ∫|m| ≤ 2e-6. This explains the M5 handoff's "O singlet ≈ half its resolved energy (smeared core)". It also means that the brief's "three highest occupied eigenvalues" of O are 2p only; there is no occupied 2s.

**What this changes:**

- **The labelling rule itself survives.**
  - n = l + 1 + (rank among same-l shells), and runs are cut by character, not energy.
  - Ne: the runs s(1), p(3), s(1) give 1s, 2p, 2s. O singlet: s(1), p(3) give 1s, 2p, and the atom passes the occupation test (complete p set in both channels).
- **Plan Task 5 `ne_labels_are_1s_2s_2p` contradicts the measurement.** It asserts "1s, 2s, 2p_x, 2p_y, 2p_z **in energy order**". The measured energy order is 1s, 2p ×3, 2s. The labels should be asserted as a set, or in the measured order.
- **Plan Task 1's synthetic channels** all use the physical order 1s < 2s < 2p. A case with 2s above 2p should be added, so that a run-walking implementation cannot assume s runs are contiguous.
- **Spec §2 item 5 / the `smeared 1s` tag.** The smearing also inverts 2s/2p, not only widening the 1s. Rule 7 wants the 2s row (and the level order) explained, for example "2s above 2p: smeared nucleus".
- **Spec §11 and the plan's O-singlet framing.** They describe the O singlet as an open-shell O that keeps the singlet. On this grid it is a closed 2p⁶ shell, not an open p⁴.

### 1c. SCF convergence

| atom | grid | tol | steps (grid point / centre / off-grid / (h/2,0,0)) | wall per solve |
|---|---|---|---|---|
| Ne | 57³ | 1e-5 | 14 / 14 / 14 / 14 | 0.8–1.4 s |
| Ne | 57³ | 1e-6 | 14 / 15 / 14 / 15 | — |
| Ne | 95³ | 1e-5 | 14 / 14 / 14 / 14 | — |
| Ne | 95³ | 1e-6 | 20 / 16 / 17 / 21 | 2.0–2.4 s |
| O singlet | 57³ | 1e-5 | **61** / 11 / 12 / 11 | 0.5–1.4 s |
| O singlet | 57³ | 1e-6 | **61** / 13 / 15 / 14 | — |
| O singlet | 95³ | 1e-5 | 14 / 13 / 13 / 13 | — |
| O singlet | 95³ | 1e-6 | 18 / 16 / 15 / 15 | 1.6–2.1 s |

- **Ne converges at every site, grid and tolerance.** Task 5 does not need a ruling on Ne convergence.
- **The O singlet's 61-step solve** occurs at the 57³ grid point at both tolerances.
- **Wall times** were taken with other agents' GPU work running (§3); they are indicative only.
- **O triplet (57³, grid point, tol 1e-5).** It does **not** converge: 200 steps in 16.3 s, Δρ oscillating around 1e-4 (minimum 8.3e-5). This is consistent with `references.md`/M5.

## 2. Closed-shell spin noise (`SPIN_FLOOR`)

**Setup.** H₂ singlet at 49³, h = 0.25. For each tolerance: one cold solve, then two warm solves after rigid moves of +0.01 and +0.02 Bohr. Here max|m| = max|ρ↑ − ρ↓|, max ρ = max(ρ↑ + ρ↓), and ∫|m| = Σ|m| h³.

| R (Bohr) | tol | max|m|/max ρ (cold / +0.01 / +0.02) | ∫|m| | ∫m | \|ε↑ − ε↓\| |
|---|---|---|---|---|---|
| 1.4 | 1e-5 | 2.00e-7 / 1.67e-7 / 2.34e-7 | ≤ 1.32e-7 | ≤ 1.8e-8 | ≤ 1.3e-9 |
| 1.4 | 1e-6 | 2.00e-7 / 2.00e-7 / 1.67e-7 | ≤ 1.32e-7 | ≤ 3.4e-8 | ≤ 5.0e-9 |
| 1.6 | 1e-5 | 2.35e-7 / 2.35e-7 / 2.74e-7 | ≤ 1.35e-7 | ≤ 2.5e-8 | ≤ 3.4e-9 |
| 1.6 | 1e-6 | 2.35e-7 / 1.96e-7 / 2.74e-7 | ≤ 1.35e-7 | ≤ 4.4e-8 | ≤ 5.5e-9 |

The same diagnostic was taken on the closed lone atoms from §1, worst over sites and tolerances:

| atom | grid | max|m|/max ρ | ∫|m| |
|---|---|---|---|
| Ne | 57³ | 1.1e-6 (off-grid) | 2.2e-5 |
| Ne | 95³ | 1.0e-7 | 8.3e-7 |
| O 1s²2p⁶ | 57³ | 1.1e-7 | — |
| O 1s²2p⁶ | 95³ | 2.5e-6 | 9.5e-5 |

**Conclusion (`SPIN_FLOOR`, P4; Task 5).**
- The closed-shell noise ratio is **≤ 2.7e-7 for H₂**, about 370× under the 1e-4 criterion (`0.1 · SPIN_FLOOR · max ρ`).
- It is **≤ 2.5e-6 for any closed atom measured**, ≥ 40× under the criterion.
- So `SPIN_FLOOR` = 1e-3 holds, and Task 5's assertion `h2_singlet_spin_density_stays_below_the_floor` (< 1e-4) has a wide margin.
- The noise is f32/eigen-residual sized: ε↑ and ε↓ agree to ≤ 5.5e-9 Ha.

## 3. Display eigensolve cost (`UNOCCUPIED_BUDGET`; P1)

**⚠ All wall times in this section were measured under contention.** Other agents' GPU processes were running throughout: a surrogate repro binary and the `app`/`sim` test suites, at 20–30 % 3D-engine utilisation each. The GPU was never quiet during this spike. Absolute walls may be up to ~2× high. **Iteration counts, stop reasons and residuals do not depend on contention.** They were identical in a second full run, whose walls agreed within about ±20 %. The SCF baseline in each row was measured in the same run, so the ratios are less affected than the absolute walls. The conclusions hold with a 2× margin either way. Re-measure the walls on a quiet GPU before quoting them in the README (Task 7).

**Setup.**
- **Tool.** `qm_gpu::eigen::lobpcg_with_guess_tol` with the test's own `Kernels`, `Reducer` and `LobpcgWork`.
- **Potential.** V_σ = v_nuc + v_h + v_xc,σ[ρ_out↑, ρ_out↓], built from the converged `GpuState`: one `xc_lsda` and one `lincomb` per channel.
- **Block.** The n_σ occupied copies plus m + `UNOCC_GUARD` vectors (m = 4 or 8, guard 2). Both channels are solved.
- **Tolerance.** The SCF's effective tol, max(`eig_tol`, 3 × `residual_floor_estimate`). Guess tol 1.5 × tol and cap 200, both copied from `qm_gpu::scf`.
- **⚠ Effective tolerance value.** At h = 0.25, `residual_floor_estimate` = 1.807e-6, so the **effective tol is 5.42e-6, not ≈ 1.8e-6** as spec §4.5 and §14 MUST 4 and the brief state: 1.8e-6 is the floor itself, not 3 × the floor. This is a wording fix to the spec; the formula is right.
- **Cold.** The unoccupied guess is `initial_orbitals`, seeds 3 (↑) and 4 (↓). Generating it on the CPU takes 25–340 ms, which is included.
- **Warm.** The previous cold call's unoccupied vectors, after a further 0.01 Bohr move and a fresh SCF.
- **Geometry.** H₂ at R = 1.4, moved rigidly. H₂O at the `h2o` preset, with only O moved. h = 0.25 and SCF tol `INTERACTIVE_TOL` 1e-5 throughout.
- **"Shown rows ≤ tol".** Every vector's own residual ‖Hx − εx‖/‖x‖ is recomputed afterwards (untimed). The column checks the first n_σ + m vectors, the rows the viewer would show.
- **Columns.** "× SCF" is the ratio to the SCF solve at the same size and geometry. "× step" is the ratio to one SCF step (that SCF's wall divided by its step count).

| system | n³ | m | guess | wall (both channels) | × SCF | × step | iterations ↑/↓ | stop | shown rows ≤ tol |
|---|---|---|---|---|---|---|---|---|---|
| H₂ | 49 | 4 | cold | 1.56 s | 11.2 | 67 | 200/200 | cap | yes |
| H₂ | 49 | 8 | cold | 6.94 s | 50 | 300 | 199/171 | floor | yes |
| H₂ | 49 | 4 | warm | 0.66 s | 4.3 | 26 | 93/66 | floor | yes |
| H₂ | 49 | 8 | warm | 1.78 s | 11.6 | 70 | 48/47 | converged | yes |
| H₂ | 81 | 4 | cold | 2.05 s | 10.4 | 63 | 200/200 | cap | yes |
| H₂ | 81 | 8 | cold | 8.14 s | 41 | 249 | 200/200 | cap | **no** (max 7.2e-5) |
| H₂ | 81 | 4 | warm | 1.80 s | 8.7 | 52 | 200/200 | cap | yes |
| H₂ | 81 | 8 | warm | 3.42 s | 16.6 | 99 | 73/107 | floor | **no** (1.4e-5) |
| H₂ | 97 | 4 | cold | 2.55 s | 12.7 | 76 | 200/200 | cap | yes |
| H₂ | 97 | 8 | cold | 10.1 s | 50 | 302 | 200/200 | cap | **no** (4.4e-4) |
| H₂ | 97 | 4 | warm | 2.27 s | 16.0 | 96 | 200/200 | cap | **no** (1.2e-5) |
| H₂ | 97 | 8 | warm | 7.62 s | 54 | 323 | 200/109 | cap / floor | **no** (7.3e-6) |
| H₂O | 49 | 4 | cold | 6.58 s | 7.9 ᶜ | 102 | 200/200 | cap | yes |
| H₂O | 49 | 8 | cold | 24.3 s | 29 ᶜ | 378 | 200/200 | cap | yes |
| H₂O | 49 | 4 | warm | 3.37 s | 1.6 ˢ | 117 | 83/125 | floor | yes |
| H₂O | 49 | 8 | warm | 17.1 s | 8.0 ˢ | 592 | 183/85 | floor | yes |
| H₂O | 81 | 4 | cold | 7.38 s | 1.2 ˢ | 197 | 200/200 | cap | yes |
| H₂O | 81 | 8 | cold | 29.6 s | 4.6 ˢ | 788 | 200/200 | cap | **no** (1.6e-4) |
| H₂O | 81 | 4 | warm | 6.87 s | 5.0 ᶜ | 65 | 200/170 | cap / floor | yes |
| H₂O | 81 | 8 | warm | 16.9 s | 12.4 ᶜ | 161 | 108/121 | floor | yes |
| H₂O | 97 | 4 | cold | 11.6 s | 1.2 ˢ | 138 | 200/200 | cap | **no** (7.0e-6) |
| H₂O | 97 | 8 | cold | 33.6 s | 3.3 ˢ | 397 | 200/200 | cap | **no** (1.7e-4) |
| H₂O | 97 | 4 | warm | 11.8 s | 3.9 ᶜ | 51 | 200/200 | cap | yes |
| H₂O | 97 | 8 | warm | 20.4 s | 6.8 ᶜ | 88 | 116/118 | floor | yes |

Footnotes on the H₂O baselines:
- **ᶜ** The warm SCF **failed** (200 steps, Δρ stalled near 1e-4, minimum 1.0–1.6e-5). The baseline is the cold re-solve at the same geometry: 13 steps, 0.84 / 1.37 / 3.0 s at 49³ / 81³ / 97³.
- **ˢ** The warm SCF converged only after 74 (49³), 171 (81³) or 119 (97³) steps (2.1 / 6.4 / 10.1 s). That inflates the "× SCF" column. Against the 13-step cold H₂O SCF, the H₂O ratios are 4–30×.

**The H₂O warm-SCF stall** after a 0.01 Bohr O move, at tol 1e-5, reproduces at 49³, 81³ and 97³. It is the known M4 item, "Warm-SCF stall on fast-changing density" (`docs/superpowers/notes/2026-10-13-m4-handoff.md`; the app retries cold, R15). It is not an orbital-viewer issue, but it is still open.

**SCF baselines for H₂.**

| grid | warm SCF | per step | cold SCF |
|---|---|---|---|
| 49³ | 6 steps, 139–154 ms | 23–26 ms | 9 steps, 0.27 s |
| 81³ | 6 steps, 196–207 ms | 33–34 ms | 0.36 s |
| 97³ | 6 steps, 142–200 ms | 24–33 ms | 0.42 s |

**A longer cap (1000) for the cold m = 4 solve** only lowers the cost if the solve converges. It converges at the floor after:

| system | iterations ↑ | iterations ↓ | wall |
|---|---|---|---|
| H₂ 49³ | 338 | 267 | 2.5 s |
| H₂ 81³ | 578 | 803 | 6.1 s |
| H₂ 97³ | 1000 (cap) | 916 | 11.0 s ⚠ |
| H₂O 49³ | 291 | 355 | 10.6 s |
| H₂O 81³ | 543 | 379 | 18.2 s |
| H₂O 97³ | 522 | 1000 (cap) | 47.5 s |

⚠ In the H₂ 97³ run, the ↑ channel **lost the occupied σ_g eigenpair**. It returned the seven lowest *box* states (0.0073 …) instead of −0.336 + six box states, so |Δε_occ| = 0.34 Ha. The shown-row residuals were ≤ tol, so a residual test alone would accept it. The loss is deterministic (it reproduced in the second run), with 0 MGS fallbacks and a coefficient amplification α of up to 18.6.

**Why it is slow.** In LSDA, H₂ has no bound unoccupied state, and H₂O has one (ε ≈ −0.002 Ha). Everything else in the block is a **box state**: a dense, degenerate-rich spectrum spaced 1e-3 to 1e-2 Ha.
- Example, H₂ at 97³: ε = 0.0073, 0.0146, 0.0244 ×2, 0.0668 ×2, 0.0676, 0.0713, 0.0793, 0.0808.
- LOBPCG's rate is set by the relative gap to the first vector outside the block, so the last shown vectors and the guards converge slowest.
- The occupied vectors and the first 2–4 unoccupied ones reach ≤ 5e-6 quickly; the shortfall is always in the top shown vector or two and in the guards.
- The block edge often cuts a degenerate pair. For example, at H₂ 81³ with m = 4, the 0.091385 pair sits at the edge, and the guard pair reads 0.091385 / 0.0927 with residuals 1e-4 to 1e-2. That is the case review MUST 3 anticipated: it is hidden by the "last cluster" rule, but it still holds LOBPCG at the cap.

**Occupied eigenvalues.** Those returned by the display solve agree with the SCF's to ≤ 5.0e-7 Ha in 59 of 60 channel solves, well inside τ(1e-6) = 1e-4. That is a 200× margin for Task 4's `occupied_eigenvalues_agree_with_the_scf`. The exception is the lost σ_g of the 1000-iteration H₂ 97³ run above.

**Recommendation for Task 4.** A display solve whose occupied eigenvalues disagree with the SCF's by more than τ should be discarded at run time and not shown. This is a runtime guard as well as a test. The MGS fallback never triggered in any of the 60 channel solves.

**Conclusion (`UNOCCUPIED_BUDGET` = 0.1; Task 4).**
- **The budget is never met at any size.** In play the rule allows ≈ 0.1 × the SCF wall per 1 s interval, i.e. ≤ ~100 ms.
- **Cheapest measured case:** warm H₂ at 49³ with m = 4, at 0.66 s = 4.3 warm SCFs. Most cases are 10–50 warm SCFs. Even if contention doubled every wall, the best case would still be 2× a warm SCF, 20× over the budget.
- **So with the spec's rule, unoccupied orbitals are paused-only at every grid size.**
- **Cost when paused:** 0.7–10 s (H₂) and 3–34 s (H₂O).
- **⚠ Convergence:** the 200-iteration cap leaves the top shown rows above the effective tolerance in 8 of 24 runs, and in most m = 8 runs at 81³/97³. Those rows will carry `not converged (residual …)` tags. The spec allows that, but the plan's step "If … the eigensolve [is] unable to reach the effective tol, stop and report" is triggered.
- **Options for the controller** (not decided here):
  - accept paused-only plus "not converged" tags;
  - judge convergence on the shown rows only and let the guards stay loose (a LOBPCG option, i.e. a solver change under rule 6);
  - raise the display cap, for example to 1000. Cold H₂ m = 4 then converges at 49³/81³ in 2.5–6 s, but at 97³ the long run lost the occupied ground state, so a higher cap needs the occupied-agreement guard above;
  - default m to 2–4 rather than 8;
  - cache the unoccupied set per geometry while paused.

## For each consumer

- **Task 1 (`split_shell_keeps_its_labels`).**
  - **S_max = 7.05e-4 Ha**: Ne, 57³, h = 0.25, at (h/2, 0, 0). The largest spread measured at any of the four sites, either grid and either tolerance.
  - The test uses **S = 2 × S_max = 1.41e-3 Ha**, together with the plan's extra S = 0.05 Ha case.
  - Only four sites were sampled. The ×2 margin covers an unsampled worse site; the off-grid point gives 0.7× the (h/2,0,0) value.
  - **Add a channel with 2s above 2p** (§1b).
- **Task 4 (`UNOCCUPIED_BUDGET`).** At 0.1 the eigensolve never runs during play, at 49³, 81³ or 97³, for H₂ or H₂O. The cheapest case is 4.3× a warm SCF. Paused-only is the operating mode. The block's top rows can end above tol at the 200 cap: see the options in §3. The effective tol at h = 0.25 is 5.42e-6.
- **Task 5.**
  - **Ne** converges at 57³ (and 95³) at all four sites and both tolerances.
  - **Tag values to expect:** "split by the grid by" ≈ 0.5 mHa off-grid and 0.7 mHa at (h/2,0,0), and ≈ 0 (≤ 1.4e-7 Ha) on the grid point and cube centre. At 95³: 0.08 and 0.12 mHa.
  - **H₂ noise ratio:** ≤ 2.7e-7, under 1e-4.
  - **⚠ `ne_labels_are_1s_2s_2p`** must not assert "1s 2s 2p in energy order": the measured order is 1s, 2p ×3, 2s (§1b).
- **Spec corrections (rule 7 honesty).**
  - §2 item 5: the 1s moves ≈ 1–2 mHa between sites, not tenths of a Ha.
  - §4.5 and §14: the effective tol is ≈ 5.4e-6, not 1.8e-6.
  - The smeared nucleus inverts 2s/2p for Zσ ≳ 1.5 (Ne, and the O singlet becomes 1s² 2p⁶). This needs a tag or a §2 line next to the smeared-1s one.

## For the controller (stop-and-report items)

1. **2s above 2p** (Ne, both grids). The O singlet is 1s² 2p⁶, a closed shell.
   - The plan's `ne_labels_are_1s_2s_2p` "in energy order" contradicts the measurement.
   - The labelling rule (character + per-l rank) still gives the right names.
2. **The display eigensolve misses the effective tol at the 200 cap** in 8 of 24 runs: the top shown rows, at 81³/97³ and m = 8. It costs 4–50 warm SCFs, so `UNOCCUPIED_BUDGET` = 0.1 makes it paused-only everywhere. Under contention; the iteration counts are not timing-dependent.
3. **The effective tolerance is 5.42e-6 at h = 0.25** (3 × 1.807e-6), not ≈ 1.8e-6 as in the spec.
4. **The 2p spread is ≈ 25× smaller than the review's model** (S_max 7.05e-4 Ha), but still above `TAU_MIN`, so MUST 1 (grouping by angular character) stands.
5. **A long (1000-iteration) display solve lost the occupied ground state** (H₂ 97³ ↑): |Δε_occ| = 0.34 Ha, with the shown residuals ≤ tol. It is deterministic. Task 4 needs a runtime occupied-eigenvalue agreement guard, especially if the cap is raised.
6. **The H₂O warm-SCF stall** (the known M4 item) reproduces at 49³/81³/97³ after a 0.01 Bohr move at tol 1e-5. It is out of scope here, but it affects any H₂O play measurement, Task 4's included.
