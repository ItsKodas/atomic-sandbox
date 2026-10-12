# Reference values used by the validation suite

All energies in Hartree, lengths in Bohr, forces in Hartree/Bohr unless stated. Each entry names
the test that uses it. Bounds are pinned here and in the tests; they are never loosened to make a
test pass (CLAUDE.md rule 6) — a miss is investigated and reported.

| Quantity | Value | Source | Used by |
|---|---|---|---|
| Bohr radius | 0.529177210903 Å | CODATA 2018 | `units` |
| Hartree | 27.211386245988 eV | CODATA 2018 | `units` |
| H atom, LSD total energy | −0.478671 | NIST "Atomic Reference Data for Electronic Structure Calculations", Kotochigova, Levine, Shirley, Stiles, Clark (1997), LSD column | `h_atom_lsda` (bound 4e-3) |
| He atom, LSD total energy | −2.834836 | same | `he_atom_lsda` (bound 1.5e-2; h = 0.10 by ruling 2026-10-13) |
| Li atom, LSD total energy | −7.335195 | same | `li_atom_lsda` (bound 4e-2) — `#[ignore]`d, see "Known limitation" below |
| H₂ LSDA equilibrium bond length | 1.446 Bohr (0.765 Å) | NIST CCCBDB, SVWN/LSDA with a large basis set; LSDA overbinds vs experiment 1.401 Bohr | `h2_bond_length` (bound ±0.05), `h2_force_vanishes_at_equilibrium` (bound 1e-2 Ha/Bohr) |
| PW92 ε_c(r_s=2, ζ=0) | −0.044760 | Perdew & Wang, PRB 45, 13244 (1992), eqs. 8–10 with the Table I parameters, evaluated exactly; libxc `LDA_C_PW` agrees (the paper's rounded −0.04476 is within 1e-6) | `qm::xc` tests (tolerance 2e-5) |
| PW92 ε_c(r_s=10, ζ=0) | −0.018572 | same (the plan's −0.01855 was a misquote; controller ruling 2026-10-12) | `qm::xc` tests (tolerance 2e-5) |
| Isotope masses | AME2020 | Wang et al., Chinese Phys. C 45, 030003 (2021) | `units` |

## Approximation gaps that are *expected* and must not be "fixed"

- LSD H atom is 21 mHa above the exact −0.5 Ha: self-interaction error of the functional.
- LSDA H₂ bond length is ~0.045 Bohr longer than experiment: LSDA overbinding.
- Gaussian nuclear smearing raises energies by ≈ 2π·Z·ρ(R)·σ² + O(σ³) + O(σ⁴) (coefficient
  derived in `qm::nuclear`; the σ³ term comes from the density cusp). The suite extrapolates
  σ → 0 with `qm::extrapolate_sigma`, a four-point fit E = a + bσ² + cσ³ + dσ⁴ on the ladder
  σ = h·(1, 1.2, 1.4, 1.6), rather than tolerating the shift. The dropped σ⁵ term biases the fit
  by an amount that scales as ρ(0)·Z⁴·σ_max⁵ (`validate` module doc), so the grid spacing per
  oracle is set by Z·h: H at h = 0.15 (predicted bias ≲ 0.6 mHa, measured +0.96); He at
  h = 0.10 (+12.3 mHa; +45.9 at h = 0.15); Li is a known limitation (below).
  At finite σ the smearing is also repulsive in the bond coordinate (2πσ² Σ_I ρ(R_I) grows as the
  bond shortens), so H₂'s R_e and the force at R_e are extrapolated in σ as well.

## Oracle designs (σ → 0 extrapolation) and measured values (Task 11, 2026-10-13, release)

| Oracle | Grid | Box | σ ladder | SCF tol / eig_tol | Fit / location | Measured → extrapolated | Reference, bound | Pass |
|---|---|---|---|---|---|---|---|---|
| `h_atom_lsda` | 81³, h = 0.15 | 12 Bohr | 0.15, 0.18, 0.21, 0.24 | 1e-7 / 1e-7 | 4-point on E(σ) | E(σ) = −0.452081 / −0.443880 / −0.435458 / −0.426968 → **−0.477710** | −0.478671, 4e-3 | yes (+0.96 mHa) |
| `h2_bond_length` | 81³, h = 0.15 | 12 Bohr | same | 1e-7 / 1e-8 | R_e(σ) = root of the Hellmann-Feynman F_x0(R; σ) on the fixed scan R ∈ {1.45, 1.60, 1.75, 1.90} (Lagrange cubic through all four, root in the sign-change interval); then 4-point on R_e(σ) | R_e(σ) = 1.6008 / 1.6530 / 1.7092 / 1.7685 → **1.4511** (1.4510602 at tol 1e-7 vs 1.4510614 at 1e-8) | 1.446, ±0.05 | yes (+0.005) |
| `h2_force_vanishes_at_equilibrium` | 81³, h = 0.15 | 12 Bohr | same | 1e-9 / 1e-8 | 4-point on F_x0(1.446; σ) | F(σ) = +0.0516 / +0.0690 / +0.0874 / +0.1066 → **+0.0010** | 0, 1e-2 | yes |
| `he_atom_lsda` | 101³, h = 0.10 | 10 Bohr | 0.10, 0.12, 0.14, 0.16 | 1e-7 / 1e-7 | 4-point on E(σ) | E(σ) = −2.600399 / −2.534062 / −2.468010 / −2.403228 → **−2.822561** | −2.834836, 1.5e-2 | yes (+12.3 mHa, 2.7 mHa margin) |
| He at the first-ruled h = 0.15 (not a test) | 81³, h = 0.15 | 12 Bohr | 0.15, 0.18, 0.21, 0.24 | 1e-7 / 1e-7 | 4-point on E(σ) | E(σ) = −2.435719 / −2.340665 / −2.250503 / −2.165700 → −2.788945 | −2.834836, 1.5e-2 | **no** (+45.9 mHa) — why the oracle runs at h = 0.10 |
| `li_atom_lsda` (`#[ignore]`) | 107³, h = 0.15 | 15.9 Bohr | 0.15, 0.18, 0.21, 0.24 | 1e-7 / 1e-7 | 4-point on E(σ) | E(σ) = −5.730663 / −5.419456 / −5.138109 / −4.883871 → −7.004270 | −7.335195, 4e-2 | **no** (+331 mHa) — known limitation |

## Known limitation: Li (and any Z ≥ 3) absolute energy on the Gaussian-smeared all-electron rung

`li_atom_lsda` is `#[ignore]`d by ruling (2026-10-13); its bound (4e-2) and reference are
unchanged. Measured: a = −7.004270, **+331 mHa**, with a clean SCF (11–13 DIIS steps, Δρ ≤ 8e-8,
no face warning in 15.9 Bohr). Cause: the four-point fit drops the (Zσ)⁵ term of the smearing
series, so its bias scales as ρ(0)·Z⁴·σ_max⁵ (`validate` module doc). With ρ(0) ≈ 13.8 and
Z = 3 the first-order model predicts +207 mHa at h = 0.15 (measured +331 with density
relaxation; the same model predicted +20 / +4 mHa for He at h = 0.15 / 0.10 vs measured
+46 / +12), and at Z·σ_max = 0.72 the fitted σ² coefficient (111) is under half the physical
2πZρ(0) ≈ 260 — the series is far from asymptotic. Meeting 40 mHa needs Z·h ≲ 0.2, i.e. h ≲ 0.07
in a 16 Bohr box (≥ 230³ ≈ 12 M points, hours per σ on the CPU reference); h = 0.10 (161³) is
predicted to miss (+50 mHa). This is the known cost of representing a Z = 3 core by a Gaussian
that the grid must resolve, not a defect of the SCF, Poisson, eigensolver or XC code (Li's SCF
convergence and spin bookkeeping stay covered by `qm::scf::tests::lithium_runs_with_unequal_spin_channels`).
Remedies: self-generated pseudopotentials (design spec §11 item 1), which remove the 1s cusp and
the Z⁴ scaling with it; or h ≲ 0.07 once the GPU solver makes it affordable. Re-enable the test
when either lands. Run it meanwhile with
`cargo test -p validate --release --test oracles -- --ignored --nocapture li_atom_lsda`.

Attribution of the He/Li misses (Task 11 report): the quartic fitted to the He h = 0.10 ladder
reproduces the h = 0.15 ladder's E(0.15) to 0.3 mHa while the two intercepts differ by 34 mHa, so
the grid error between the two spacings is < 0.5 mHa at equal σ and the miss is the fit's
range-dependent bias (the smearing series is pre-asymptotic at Z·σ_max = 0.48 for He and 0.72
for Li). E(σ) − a is 0.77→0.71 (H), 0.73→0.67 (He), 0.50→0.48 (Li) of the first-order curve
2πZρ(0)σ²-series — the density-relaxation signature Task 9 verified for H against a radial
Numerov reference. No tolerance was changed.

Why the force and not a parabola through E(R) for R_e: the vertex of a parabola through three
E(R) values spaced Δ is biased by E‴Δ²/(6E″) ≈ 0.016 Bohr at Δ = 0.15 (E‴ ≈ −1.6 Ha/Bohr³,
E″ ≈ 0.37 from Task 10), while the Hellmann-Feynman force is the exact derivative of the SCF
energy to 6e-8 Ha/Bohr (Task 10) and its root is found by interpolating a smooth, nearly linear
function. Why the interpolant must use the same nodes at every σ: the intercept weights of the
four-point fit are (17.7, −36.8, 27.0, −6.9), |w|₁ ≈ 88, so an error in R_e(σ) that is not
smooth in σ is amplified ~90×. A 3-point parabola whose bracketing triple switches with σ
differs from the cubic by ≤ 1.4e-3 Bohr per σ yet extrapolates to 1.539 Bohr (a fail); the
fixed-node cubic gives 1.451. Egg-box noise in F (≤ 4e-5 Ha/Bohr) moves a root by ≤ 1e-4 Bohr.
Why four scan points and not three: a fixed quadratic over a 0.30 Bohr span has a ~1 mBohr
interpolation error that changes with σ as the root crosses the span; on the measured data the
triples {1.60, 1.75, 1.90} / {1.45, 1.60, 1.75} extrapolate to 1.427 / 1.440 — 0.01–0.02 Bohr
from the cubic's 1.451, above the 1e-3 Bohr change allowed for a runtime cut. Why SCF tol 1e-7
(from 1e-8): the extrapolated R_e changes by −1.2e-6 Bohr while the 16 SCFs take 150 instead of
217 DIIS steps; with eig_tol = 1e-8 the orbitals (hence F) are already frozen before Δρ reaches
1e-8, so the extra steps changed nothing.

## Runtime (release, Ryzen 7 9800X3D, one thread per test; the harness runs tests concurrently)

| Test | Wall | Conditions |
|---|---|---|
| `h_atom_lsda` | 131 s | alone (4 SCFs, 10–13 DIIS steps each) |
| `h2_force_vanishes_at_equilibrium` | 752 s | 5 tests concurrent (4 SCFs at tol 1e-9, 13–23 steps) |
| `h2_bond_length` | 1488 s alone at tol 1e-7 (16 SCFs, 150 DIIS steps); was 2235 s (5 concurrent) / 2269 s (3 concurrent) at tol 1e-8 (217 steps) | see `last-run.md` for the suite-run time |
| `he_atom_lsda` (h = 0.10) | 823 s alone (1048 s with 5 concurrent) | 101³, 4 SCFs, 13–18 steps |
| He at h = 0.15 (not a test) | 275 s | 5 tests concurrent |
| `li_atom_lsda` (ignored) | 1158 s | 5 tests concurrent (107³, 11–13 steps) |

The committed suite (`cargo test -p validate --release`) is bounded by `h2_bond_length`; the
Task 11 report records the solo workspace time. Do not mark the oracles `#[ignore]` — they are
the gate every solver change must pass.

## GPU vs CPU at fixed σ (M3 Task 8, `tests/gpu_vs_cpu.rs`, 2026-10-13, release, RX 7900 XTX)

Run with `cargo test -p validate --release --test gpu_vs_cpu` (~3.5 min, four tests concurrent;
skips without a GPU adapter; `--nocapture` prints the tables; `timing_table` also writes the
gitignored `last-run-gpu.md`).

**Why fixed σ and not the σ → 0 oracles** (M1 handoff §3). The oracles above extrapolate E(σ)
through a four-point fit whose intercept weights sum to |w|₁ ≈ 88, so 1e-6 Ha of noise in any
E(σ) is ~1e-4 in the extrapolated energy, and He passes with 2.7 mHa of margin. The GPU keeps
every grid quantity in f32 and its total energy has a *path-dependent* scatter — two correct
trajectories to the same fixed point differ by ~6e-7 Ha for H₂ at 49³ (Task 7: cold vs
warm-from-neighbour) and 5.4e-6 Ha for H₂O (Task 8: the two charge-renormalisation targets,
27 vs 17 steps) — which the ×88 would turn into a mHa-level fake failure. So the GPU is
validated against the CPU f64 solver (which does pass the oracles) on the **same discrete
problem**: same grid, σ = h = 0.25, tol 1e-6, cold starts on both sides. Nothing is extrapolated.
The σ → 0 oracles remain the CPU-reference gate until pseudopotentials land.

**Energies** (`energies_match_at_fixed_sigma`, 49³ = 12 Bohr box, Ha). Bounds: brief 1e-4
(H/He/H₂), 5e-4 (LiH/H₂O) → tightened to 3× the larger of the achieved |ΔE| and the system's
f32 path scatter, rounded up (the constants in the test say why per system).

| system | E_cpu | E_gpu | ΔE (gpu − cpu) | worst component | steps cpu / gpu | wall cpu / gpu | bound (was) |
|---|---|---|---|---|---|---|---|
| H | −0.424224232 | −0.424224380 | **−1.5e-7** | external 9.1e-7 | 9 / 8 | 5.1 s / 0.47 s | 1e-6 (1e-4) |
| He | −2.138685118 | −2.138686124 | **−1.0e-6** | external 2.1e-6 | 11 / 11 | 9.3 s / 0.61 s | 3e-6 (1e-4) |
| H₂ R = 1.4 | −0.962715541 | −0.962714949 | **+5.9e-7** | hartree 9.2e-7 | 9 / 11 | 8.1 s / 0.64 s | 2e-6 (1e-4) |
| LiH R = 3.0 (`lih` preset) | −5.320420938 | −5.320421249 | **−3.1e-7** | external 2.1e-6 | 13 / 14 | 32 s / 2.1 s | 3e-6 (5e-4) |
| H₂O (`h2o` preset) | −37.528379872 | −37.528392550 | **−1.27e-5** | external 1.9e-5 (kinetic +6.2e-6) | 17 / 17 | 88 s / 6.8 s | 4e-5 (5e-4) |

(CPU walls with four tests running concurrently.) H₂O's 1.27e-5 is 3.4e-7 of |E| and 2e-7 of
|E_ext| = 97 Ha — the f32 level of a Z = 8 on-site potential (~−25 Ha) times a density stored
in f32, and 2.3× the GPU's own H₂O path scatter (5.4e-6); the CPU value is stable to 1e-9 (tol
1e-7 changes the GPU's by 3.7e-7 only). LiH and the CPU both warn "box too small" (4.5e-5 on
the faces of the 12 Bohr box; Li's 2s) — identical on both sides, not a comparison issue.
Every GPU solve ended with `eigen_last_step_residual` ≈ 5e-6 (the 3× f32 floor the solver asks
for), no LOBPCG/Poisson warnings; LiH had 3 of 28 eigensolves accepted at the floor (a note).

**Forces** (`forces_match_at_fixed_sigma`, 49³, Hellmann-Feynman, both states at tol 1e-6,
Ha/Bohr). Bound: brief 1e-3 → 5e-6 (H₂), 6e-5 (H₂O).

| system | nucleus | F_gpu | F_cpu | max \|ΔF\| |
|---|---|---|---|---|
| H₂ R = 1.0 | 0 (x) | +0.58049955 | +0.58050090 | **1.35e-6** |
| H₂ R = 1.4 | 0 (x) | +0.13807482 | +0.13807482 | **1.4e-8** |
| H₂ R = 2.5 | 0 (x) | −0.06024684 | −0.06024519 | **1.66e-6** |
| H₂O | O (y) | −0.34097500 | −0.34099228 | **1.73e-5** |
| H₂O | H (x, y) | +0.26040852, +0.17032517 | +0.26040954, +0.17032586 | 1.0e-6 |

The H₂ differences are the Δρ term of two tol-1e-6 states (Task 7: ~4·Δρ); R = 1.4 happens to
land at 1.4e-8. H₂O's 1.7e-5 on the oxygen is the same first-order term with ∫|∇V_O| ∝ Z = 8.
Σ_I F_I,y for H₂O is −3.2e-4 (GPU) / −3.4e-4 (CPU): the egg-box force of a Z = 8 nucleus at
σ = h = 0.25, present on both solvers and agreeing to 2e-5 — dynamics must not assert momentum
conservation below that (see `qm::forces`).

**H₂ curve shape** (`h2_curve_shape_matches`, 49³, R ∈ {1.0, 1.4, 1.8, 2.4, 3.2}, GPU cold at
every R and warm along R). ΔE = E_gpu − E_cpu: cold −5.4e-7 / +5.9e-7 / +8.2e-7 / +7.6e-7 /
+5.3e-7 (mean +4.3e-7), warm −5.4e-7 / −4.9e-7 / −4.7e-7 / +7.1e-7 / +1.1e-7 (mean −1.4e-7);
max |ΔE − mean| **9.7e-7 cold, 8.4e-7 warm** — the per-point path scatter. Bound: brief 5e-5 →
5e-6. Both solvers give E(1.4) < E(1.0) and E(3.2) > E(1.8) (the σ-shifted minimum is near 1.8).

**Timing** (`timing_table`, uncontended, σ = h = 0.25, tol 1e-6; "cold" = second cold solve on a
warmed solver, "warm" = from that state with the molecule shifted 0.02 Bohr; phase walls from
`SolveProfile`). The full table with the first-solve rows, readbacks and iteration counts is
written to `last-run-gpu.md`; the Task 8 report has the run of record.

| system | grid | solve | steps | wall | ms/step | Poisson (share) | LOBPCG (share) | readbacks/step | CG it | LOBPCG it |
|---|---|---|---|---|---|---|---|---|---|---|
| H₂ | 33³ | cold / warm | 10 / 8 | 372 / 215 ms | 37 / 27 | 44 % / 47 % | 54 % / 51 % | 217 / 160 | 575 / 350 | 290 / 154 |
| H₂ | 49³ | cold / warm | 11 / 8 | **552 / 292 ms** | 50 / 37 | 43 % / 44 % | 54 % / 54 % | 265 / 188 | 800 / 400 | 382 / 194 |
| H₂ | 65³ | cold / warm | 10 / 8 | 740 / 384 ms | 74 / 48 | 43 % / 41 % | 53 % / 58 % | 336 / 203 | 950 / 425 | 430 / 216 |
| H₂ | 97³ | cold / warm | 13 / 8 | 1094 / 463 ms | 84 / 58 | **44 %** / 40 % | 49 % / 59 % | 325 / 228 | 1300 / 500 | 467 / 231 |
| H₂O | 57³ | cold / warm | 23 / 13 | 8.9 / 5.0 s | 386 / 385 | 7 % / 5 % | **93 % / 95 %** | 554 / 518 | 1600 / 650 | 829 / 474 |
| H₂O | 97³ | cold / warm | 25 / 15 | **14.2 / 7.6 s** | 568 / 504 | 7 % / 5 % | **92 % / 95 %** | 610 / 516 | 2125 / 825 | 958 / 530 |

`GpuSolver::new`: 18–25 ms (33³/49³), 0.11 s (65³), 0.31 s (97³: 49 R_lm ghost-table fields).
Cold-start guess (CPU f64 upload): 4 ms (49³ H₂), 57 ms (97³ H₂), 0.22 s (97³ H₂O). With the
suite's four tests concurrent the GPU rows are 5–20 % slower (49³ H₂ 604 / 315 ms), and the
H₂O rows vary ±15 % between runs (12.0–14.2 s at 97³).

Plan targets: 49³ H₂ cold ≤ 0.5 s — **missed by 10 %** (552 ms; Task 6's bench had 497 ms);
warm ≤ 0.15 s — **missed** (292 ms, 1.9×); 97³ H₂O cold ≤ 5 s — **missed** (14.2 s, 2.8×).
**Task 10 trigger** (multigrid Poisson if Poisson ≥ 40 % of a 97³ step or 97³ H₂O cold > 5 s):
**triggered on both criteria** — Poisson is 44 % of a 97³ H₂ step and the 97³ H₂O cold solve
is 14.2 s. But the split says what a Poisson-only change can and cannot do: for H₂ (k = 1)
Poisson and LOBPCG are ~45 / 50 % and both are readback-bound (265–325 readbacks per step at
~0.15 ms); for H₂O (k = 5 per channel) **LOBPCG is 92–95 %** of the wall at 11–13.6 ms per
LOBPCG iteration (vs 1.2 ms at k = 1: the k² Gram/Rayleigh–Ritz block ops over 3k = 15 vectors,
bandwidth-bound at 97³), and Poisson is 7 %. Multigrid alone would take 97³ H₂O from 14.2 s to
~13.5 s; the 5 s target needs the eigensolver (fewer readbacks per iteration — CGS/Cholesky-QR,
Task 5's deferred minor — fused Gram kernels, and fewer iterations per call: 19 per call for
H₂O vs 36 for H₂ already, so it is the per-iteration cost).

**Evidence that the ρ_in renormalisation target matters** (Task 6 fix, ∫ρ_out,σ instead of
n_σ; the ruled one-off run, not committed): H₂O 49³ cold at tol 1e-6 takes **27 steps with the
old n_σ target vs 17 with ∫ρ_out** (tol 1e-7: 43 vs 33) — the N|ε| floor (N = 10) costs ten
DIIS steps even below tol. The H₂ tol-1e-7 test in `qm-gpu::scf` does *not* discriminate: with
the old target it still passes (11 steps, Δρ_final 9.4e-8 vs 8.8e-8), because N|ε| ≈ 1.0e-7
for H₂ at 33³ sits at the tolerance rather than above it; at tol 5e-8 both targets overrun the
CPU's step count (+7 / +12) as the f32 Δρ floor (~3e-8·N) is reached, so a tighter H₂ probe is
not usable either. The discriminating regression test is the H₂O step count above (GPU 17 =
CPU 17 in `energies_match_at_fixed_sigma`).
