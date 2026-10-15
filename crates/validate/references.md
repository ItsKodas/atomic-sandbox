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
| Li atom, LSD total energy | −7.343957 | same, LSD column (−7.335195 is the LDA column; corrected 2026-10-15, M7 Task 1 review) | `li_atom_lsda` (bound 4e-2) — `#[ignore]`d, see "Known limitation" below |
| The functional behind these NIST rows | VWN (Vosko–Wilk–Nusair, the Ceperley–Alder fit) | NIST procedure page; M7 Task 1 | the 3-D oracles run PW92 against VWN numbers: the measured functional gap PW92 − VWN5 of the same radial solver, H −0.04, He +0.38, Li +0.67 mHa ("M7 radial atom (Task 1)"), is well inside their bounds (rule 7) |
| H₂ LSDA equilibrium bond length | 1.446 Bohr (0.765 Å) | NIST CCCBDB, SVWN/LSDA with a large basis set; LSDA overbinds vs experiment 1.401 Bohr | `h2_bond_length` (bound ±0.05), `h2_force_vanishes_at_equilibrium` (bound 1e-2 Ha/Bohr) |
| PW92 ε_c(r_s=2, ζ=0) | −0.044760 | Perdew & Wang, PRB 45, 13244 (1992), eqs. 8–10 with the Table I parameters, evaluated exactly; libxc `LDA_C_PW` agrees (the paper's rounded −0.04476 is within 1e-6) | `qm::xc` tests (tolerance 2e-5) |
| PW92 ε_c(r_s=10, ζ=0) | −0.018572 | same (the plan's −0.01855 was a misquote; controller ruling 2026-10-12) | `qm::xc` tests (tolerance 2e-5) |
| Isotope masses (22 isotopes, Z ≤ 10) | AME2020 **atomic** masses, Table I's atomic-mass column: ¹H 1.007825031898 u, ²H 2.014101777844 u, … (every entry pinned at 1e-9 u) | Wang, Huang, Kondev, Audi & Naimi, Chinese Phys. C 45, 030003 (2021); machine-readable `mass_1.mas20` (AMDC, 3 Mar 2021) | `units::isotopes` |
| m_D/m_H (atomic masses) | 1.9984637; √ = 1.4136703 — the H₂/D₂ period ratio `sim::integrator` pins on the Morse dimer (M6 Task 5) and `tools` pins on the DFT surface (M6 Task 7) | the two AME2020 masses above | `units::isotopes`, `sim::integrator` (bound 2e-3 raw, 2e-4 Verlet-corrected), `tools::d2_vibrates_slower_than_h2_by_the_mass_ratio` (3e-4 Verlet-corrected; measured −4.6e-5) |

## Approximation gaps that are *expected* and must not be "fixed"

- LSD H atom is 21 mHa above the exact −0.5 Ha: self-interaction error of the functional.
- Nuclear masses are the AME2020 **atomic** masses (electrons included), the usual
  Born-Oppenheimer choice since each nucleus moves with its electrons: ¹H is 1837.15 mₑ, not the
  proton's 1836.15 (5.4e-4 relative, smaller for heavier elements), and in a cation one spare
  electron mass is carried (1.8e-4 relative for H₃⁺) — below every bound in this suite. With
  nuclear masses the H₂/D₂ frequency ratio would be √(2.013553212745/1.007276466621) = 1.41386
  instead of the atomic-mass 1.41367 (M6 spec §5.5); experiment's 1.41268 differs by the
  adiabatic/non-adiabatic corrections no BO code reproduces.
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
| `li_atom_lsda` (`#[ignore]`) | 107³, h = 0.15 | 15.9 Bohr | 0.15, 0.18, 0.21, 0.24 | 1e-7 / 1e-7 | 4-point on E(σ) | E(σ) = −5.730663 / −5.419456 / −5.138109 / −4.883871 → −7.004270 | −7.343957, 4e-2 | **no** (+340 mHa) — known limitation |

## Known limitation: Li (and any Z ≥ 3) absolute energy on the Gaussian-smeared all-electron rung

`li_atom_lsda` is `#[ignore]`d by ruling (2026-10-13); its bound (4e-2) and reference are
unchanged (the reference digits corrected to NIST's LSD column on 2026-10-15, M7 Task 1 review: −7.343957,
not the LDA column's −7.335195). Measured: a = −7.004270, **+340 mHa**, with a clean SCF (11–13 DIIS steps, Δρ ≤ 8e-8,
no face warning in 15.9 Bohr). Cause: the four-point fit drops the (Zσ)⁵ term of the smearing
series, so its bias scales as ρ(0)·Z⁴·σ_max⁵ (`validate` module doc). With ρ(0) ≈ 13.8 and
Z = 3 the first-order model predicts +207 mHa at h = 0.15 (measured +340 with density
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

## GPU vs CPU at fixed σ (M3 Task 8, `tests/gpu_vs_cpu.rs`, 2026-10-13, release, RX 7900 XTX; Task 10 update below)

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

(CPU walls with four tests running concurrently.) H₂O's 1.27e-5 is at the f32 level — 3.4e-7
of |E| and 2e-7 of |E_ext| = 97 Ha (a Z = 8 on-site potential of ~−25 Ha against an f32
density), 2.3× the GPU's own H₂O path scatter (5.4e-6), while the CPU value is stable to 1e-9
and tol 1e-7 moves the GPU's by only 3.7e-7 — but **its mechanism is not isolated**. Four of
the five ΔE are negative, and H₂O's total (−1.27e-5) is two thirds of its E_ext shift
(−1.9e-5): a different fixed point of the *same* functional changes the total only to second
order, so "total ≈ component" is the signature of a slightly different Hamiltonian, not of
f32 path scatter alone. Leading candidate: the A&S erf in `nuclear.wgsl` (≤ 1.8e-7 relative for
x ≥ 1; Z/d ≈ 16 Ha × 1.8e-7 × ~8 electrons near O ≈ 2e-5 — the right size and sign-consistent);
its upgrade is scheduled in Task 10. The 4e-5 bound stands until then. LiH and the CPU both
warn "box too small" (4.5e-5 on the faces of the 12 Bohr box; Li's 2s) — identical on both
sides, not a comparison issue. Every GPU solve ended with `eigen_last_step_residual` ≈ 5e-6
(the 3× f32 floor the solver asks for), no LOBPCG/Poisson warnings; LiH had 3 of 28 eigensolves
accepted at the floor (a note). **Asserted step-count guard:** H₂O's GPU step count must be
≤ the CPU's + 5 (`H2O_STEP_MARGIN`; achieved 17 = 17) — the renormalisation regression test,
see the evidence paragraph below.

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
~0.15 ms); for H₂O (k = 5 per channel) **LOBPCG is 92–95 %** of the wall and Poisson is 7 %.
The LOBPCG iteration at k = 5 costs 9.9 ms at 57³ and 13.6 ms at 97³ while the point count
grows 4.93× (185k → 913k): a linear fit gives **≈ 9 ms grid-independent + ≈ 5 ns/point** —
i.e. the cost is *latency-first*, not bandwidth: the 2k + 1 = 11 readbacks per iteration and
the O(k²) Gram/MGS dispatches over 3k = 15 vectors (hundreds of small dispatches, each with its
own uniform + bind group) dominate, and the per-point work (≈ 4.6 ms at 97³) is the minor
part. (For comparison k = 1 costs 1.2 ms/it at 97³.) Multigrid alone would take 97³ H₂O from
14.2 s to ~13.5 s; the 5 s target needs the eigensolver's *sync-point count* cut — CGS/
Cholesky-QR instead of MGS (2k + 1 → ~3 readbacks per iteration, Task 5's deferred minor),
fused block-Gram kernels (one dispatch per k × 3k block instead of per-pair dots), H·X reuse
— not its arithmetic; the iteration count per call is already lower than H₂'s (19 vs 36).
Task 5's standalone bench (`bench_lobpcg_49_97`, oscillator, 97³ at h = 0.2, k = 5) reported
9.75 ms/it at the time; the SCF's 13.6 ms/it at 97³ H₂O (h = 0.25, k = 5, real potential) is
1.4× that. Whether that is the different problem (V and h; the SCF also calls with warm X at
tol 5.4e-6) or a regression could not be settled in the Task 8 fix round: the bench re-run
gave 43–75 ms/it with the machine visibly contended (the SCF bench's 49³ H₂ cold took 3.9 s
against the 552 ms measured an hour earlier on the same build), so it is **unexplained** —
re-run `bench_lobpcg_49_97` on a quiet machine before Task 10 touches the eigensolver.

### Task 10 (2026-10-13): LOBPCG latency, multigrid Poisson, erf precision — numbers of record

Three solver changes (Task 10 report, commits A–C on `m3-gpu-solver`): LOBPCG with one
readback per iteration (Rayleigh–Ritz on the raw [X, W, P] with measured A and B, a fused
Gram kernel, fused H and Jacobi kernels), a geometric multigrid V-cycle as the Poisson CG
preconditioner (9–15 PCG iterations instead of 125–325), and an erfc fit accurate to 6.1e-9 in
place of Abramowitz–Stegun 7.1.26 (1.4e-7). The suite's bounds were tightened, never loosened.

**Energies** (same protocol as above). Achieved |ΔE|: H +4.7e-9, He +4.2e-8, H₂ +1.4e-8,
LiH +1.1e-7, **H₂O +2.35e-7** (steps cpu / gpu 17 / 16; worst component external 4.7e-5).
**The H₂O systematic is resolved, and the hedge above pointed at the wrong candidate:** the
multigrid Poisson alone (commit B, before the erf change) took H₂O from −1.33e-5 to
+1.84e-6. The old Jacobi-CG ran 60–100 iterations per solve into its f32 recurrence floor,
whose *true* residual was ~1e-5 relative (9.8e-6 at 49³, 1.8e-5 at 97³); that smooth V_H
error biased the fixed point. With MG-PCG the floor is 2.3e-6 (49³) / 3.2e-6 (97³) and the
H₂O components now cancel to second order (external −4.4e-5, hartree +2.9e-5, kinetic
+2.1e-5) — the signature of the *same* functional, as a correct f32 solver should show. The
erf upgrade (commit C) then moved the H₂O total from +1.84e-6 to +2.35e-7. Reading: the
*mechanism* is resolved — the V_H f32 floor (≈ 1.2e-5) was the dominant term and MG-PCG
removed it. The erf's *share* (≈ 1.6e-6) is indicative, not resolved, because it sits inside
H₂O's own 5.4e-6 path scatter — the A&S error was real (1.4e-7 absolute, smooth in x) but its
energy effect cannot be separated from the scatter here.
Bounds (3× max(achieved, path scatter), the scatter now binding everywhere): H₂O 4e-5 →
**2e-5**, He 3e-6 → **2e-6**; H 1e-6, H₂ 2e-6, LiH 3e-6 unchanged. H₂O step guard: 16 ≤ 17 + 5.

**Forces**: max |ΔF| 2.8e-6 / 3.0e-6 / 1.8e-6 at R = 1.0 / 1.4 / 2.5 (bound 5e-6) and 3.1e-6 for H₂O (bound 6e-5; Σ_I F_I,y = −3.5e-4 / −3.4e-4, the egg-box force as before). **H₂ curve shape**: max |ΔE − mean| 7.9e-8 cold, 5.0e-8 warm (bound 5e-6; Task 8: 9.7e-7 / 8.4e-7).

**Timing** (`timing_table` solo, the mode of record; σ = h = 0.25, tol 1e-6; cold = second
cold solve on a warmed solver, warm = from that state with the molecule shifted 0.02 Bohr):

| system | grid | solve | steps | wall | ms/step | Poisson (share) | LOBPCG (share) | readbacks/step | CG it | LOBPCG it |
|---|---|---|---|---|---|---|---|---|---|---|
| H₂ | 33³ | cold / warm | 10 / 8 | 164 / 92 ms | 16 / 12 | 39 % / 40 % | 54 % / 54 % | 52 / 39 | 65 / 39 | 290 / 154 |
| H₂ | 49³ | cold / warm | 10 / 8 | **206 / 108 ms** | 21 / 14 | 36 % / 37 % | 57 % / 57 % | 61 / 42 | 67 / 35 | 375 / 192 |
| H₂ | 65³ | cold / warm | 10 / 9 | 256 / 145 ms | 26 / 16 | 30 % / 38 % | 62 % / 57 % | 65 / 42 | 64 / 38 | 420 / 213 |
| H₂ | 97³ | cold / warm | 10 / 8 | 412 / 195 ms | 41 / 24 | 25 % / 32 % | 62 % / 65 % | 69 / 47 | 64 / 35 | 462 / 228 |
| H₂O | 57³ | cold / warm | 15 / 12 | 1.80 / 1.03 s | 120 / 86 | 6 % / 8 % | **91 % / 91 %** | 73 / 55 | 85 / 57 | 776 / 433 |
| H₂O | 97³ | cold / warm | 15 / 12 | **3.57 / 1.75 s** | 238 / 146 | 4 % / 5 % | **90 % / 94 %** | 79 / 56 | 85 / 51 | 872 / 454 |

Plan targets: 49³ H₂ cold ≤ 0.5 s — **met** (206 ms; Task 8: 552); warm ≤ 0.15 s — **met**
(108 ms; Task 8: 292); 97³ H₂O cold ≤ 5 s — **met** (3.57 s; Task 8: 14.2). `GpuSolver::new`
unchanged (30 ms at 49³, 0.3 s at 97³). The H₂O rows are LOBPCG-bound (90 %) at
≈ 3.5 ms per k = 5 iteration — ≈ 70 dispatches and one readback each — so the next lever
there is the per-dispatch overhead (a per-op uniform ring, wider lincombs), not sync points.

Per-iteration costs (benches, solo): LOBPCG 97³ k = 5 12.3 → 3.4 ms/it (11 → 1 readback,
≈ 500 → 70 dispatches), 49³ k = 2 2.40 → 0.68 ms/it; Poisson 49³ 186 it / 77 ms → 9 it /
14 ms, 97³ 325 it / 143 ms → 9 it / 18 ms. The GPU shader's erf: max relative error of
G(x) = E′(x)/x on the shader 5.0e-7 → 1.31e-7 (now the series branch's f32 rounding), the
switch step 5.8 → 0.8 ulp; V_nuc 1.3e-7 → 1.02e-7 of max|V| (f32 rounding, no systematic
part). What the SCF taught (Task 10 report §2): with the eigen tol at 3× the f32 floor the
warm orbitals *freeze* once their residual sits below it and the mixing alone finishes —
and must, because every orbital update injects ~1e-7 of Δρ; a 1.5× hysteresis band keeps
the freeze from being decided at the rounding level (10 vs 15 SCF steps between two correct
eigensolvers before it).

**Forward item (M4): orbital lag at the interactive tolerance.** At the viewer's tol 1e-5 the
eigen tol of 3× the f32 floor (≈ 5.4e-6 residual, band to 8.1e-6) admits an orbital set whose
density is ~5e-5 from self-consistent — more than the SCF tolerance — so frozen orbitals can be
re-solved by the potential's drift late in a warm solve, which moves ρ_out by ~8e-5 and
resets the DIIS: measured on the app's warm H₂ 49³ (nucleus moved 0.02 Bohr), 15 mixing
steps vs 9 cold, at 210 vs 304 ms and the same energy. Correctness is preserved (energy second
order, forces within their bounds); it is a cost trade-off that matters once dynamics makes
warm starts the common case. Remedies to evaluate: a per-call eigen tol tied to the current
Δρ, or a cheaper floor (error-compensated Ritz lincombs). Notes:
`docs/superpowers/notes/2026-10-13-m3-task10-orbital-lag.md`.

**Evidence that the ρ_in renormalisation target matters** (Task 6 fix, ∫ρ_out,σ instead of
n_σ; the ruled one-off run, not committed): H₂O 49³ cold at tol 1e-6 takes **27 steps with the
old n_σ target vs 17 with ∫ρ_out** (tol 1e-7: 43 vs 33) — the N|ε| floor (N = 10) costs ten
DIIS steps even below tol. The H₂ tol-1e-7 test in `qm-gpu::scf` does *not* discriminate: with
the old target it still passes (11 steps, Δρ_final 9.4e-8 vs 8.8e-8), because N|ε| ≈ 1.0e-7
for H₂ at 33³ sits at the tolerance rather than above it; at tol 5e-8 both targets overrun the
CPU's step count (+7 / +12) as the f32 Δρ floor (~3e-8·N) is reached, so a tighter H₂ probe is
not usable either. The energy does not catch it either: with the old target H₂O lands at
−37.528387142, −7.3e-6 from the CPU, inside the 4e-5 bound. The discriminating regression test
is therefore the **asserted** H₂O step-count guard in `energies_match_at_fixed_sigma`: GPU
steps ≤ CPU steps + 5 (achieved 17 vs 17; the old target's 27 fails it by 5).

## Dynamics (M4 Task 7, `tests/dynamics.rs`, 2026-10-13, release, RX 7900 XTX)

Run with `cargo test -p validate --release --test dynamics` (~6 min, four tests
concurrent with their GPU evaluations serialised; skips without a GPU adapter; `--nocapture`
prints every number; each test appends its row to the gitignored `last-run-dynamics.md`). Wall
times are reported, never asserted (ruling R3).

**Design.** Every test runs on one fixed discrete problem — σ = h = 0.25, 49³ (12 Bohr box) for
H₂ and H₃⁺ at tol 1e-6 (1e-5, the viewer's, for the thermostat) — and compares the *dynamics*
against *statics of the same surface*, so no bound depends on the distance of that surface from
the σ → 0 LSDA limit or from experiment; those gaps are reported under rule 7. H₂ runs at 49³,
not the brief's 33³: at 33³ both solvers warn "box too small" for H₂ (face density ~4e-5,
Task 2 report) and the tests assert the warning's absence; the warm step costs about the same
(latency-bound). Integrator: velocity Verlet (Verlet, Phys. Rev. 159, 98 (1967); Swope,
Andersen, Berens & Wilson, J. Chem. Phys. 76, 637 (1982)); thermostat: CSVR (Bussi, Donadio &
Parrinello, J. Chem. Phys. 126, 014101 (2007)).

| Quantity | Value | Source | Used by |
|---|---|---|---|
| H₂ ω_e, experiment | 4401.21 cm⁻¹ | NIST CCCBDB (experimental vibrational data, Huber & Herzberg) | reported against (not asserted) |
| H₂ ω_e, LSDA (SVWN) | 4201 cm⁻¹ (aug-cc-pVQZ; 4203 aug-cc-pVTZ, 4206 6-311G**) | NIST CCCBDB, calculated vibrational frequencies, SVWN rows | reported against (not asserted) |
| H₂ E″(R_e) at σ → 0, this solver | ≈ 0.37 Ha/Bohr² → **4405 ± 60 cm⁻¹** with μ = 918.58 mₑ | M1 Task 10 (h = 0.15, extrapolated curve): a two-significant-figure number that exists only inside the parabola-vertex bias argument above — ±0.005 on E″ is ±30 cm⁻¹, and a parabola through points 0.15 Bohr apart overestimates k by ~1.3 % (another +30) — so it **cannot discriminate experiment (4401) from LSDA (4201)**; a σ-extrapolated 3-point k at Δ ≤ 0.05 in the oracle suite would | the "LSDA h = 0.15 extrapolated value" the brief asks to report against |
| E_h/(hc) | 219 474.631 cm⁻¹ | CODATA 2018 | the cm⁻¹ conversion of ω (a.u.) |

### `h2_vibration_frequency_matches_curvature_of_the_same_surface`

What it measures: that the forces the integrator uses are the gradient of the energy the
surface has, independently of the smearing — ω_MD from the trajectory against
ω_static = √(k/μ) from the energies. R_e(σ) by force bisection on F_x(nucleus 0) = −dE/dR
(bracket [1.6, 2.0], 12 halvings to 1e-4 Bohr), then three *cold* solves at R_e − Δ, R_e,
R_e + Δ (Δ = 0.05, the MD's own amplitude; cold solves are bit-reproducible and their mutual
path scatter is 8e-8 Ha, Task 10) for k = [E₊ − 2E₀ + E₋]/Δ²; 400 Verlet steps at dt = 0.25 fs
from R_e + Δ at rest; ω_MD from the upward zero crossings of R(t) − R_e (upward only, so an
offset of the oscillation centre cancels exactly).

| quantity | value |
|---|---|
| R_e(σ = 0.25, 49³) | **1.78894 Bohr** (the σ → 0 oracle: 1.4511; the smearing is repulsive in the bond coordinate) |
| E(R_e − Δ), E(R_e), E(R_e + Δ) | −0.984638823, −0.984900045, −0.984661190 Ha (9 SCF steps each) |
| k from energies / from forces −[F₊ − F₋]/2Δ | **0.20003** / 0.20035 Ha/Bohr² (ratio 1.0016 = the two truncation terms, E⁗Δ²/12 vs E⁗Δ²/6) |
| μ (¹H atom masses), E_mode = E₊ − E₀ | 918.58 mₑ, 2.3885e-4 Ha |
| ω_static | 1.475675e-2 a.u. = **3238.7 cm⁻¹** |
| ω_MD (9 upward crossings, period 10.3049 fs) | 1.474855e-2 a.u. = **3236.9 cm⁻¹** |
| **ω_MD/ω_static − 1** | **−5.55e-4** (over 2000 steps, 48 crossings: −5.51e-4) |
| h = ω dt; Verlet's ω̃/ω − 1 = h²/24 + …; ω_MD/ω̃ − 1 | 0.1525; +9.72e-4; −1.53e-3 (anharmonic softening at this amplitude plus the k truncation) |

Bound `OMEGA_BOUND`: brief 3 % → **5e-3**. Not 3× the achieved (1.7e-3): the achieved −5.6e-4
is a partial cancellation of three known systematic terms — Verlet +9.7e-4 (h²/24, exact), the
3-point truncation of k (k_F/k_E = 1.0016 → −0.8e-3 on the ratio) and the anharmonic softening
at this amplitude (the remainder, −0.7e-3; ω_MD/ω̃ − 1 = −1.5e-3 is truncation + anharmonic
together, and a Morse well of this k and D ≈ 0.09 would give −1.3e-3) — whose aligned sum is
2.5e-3, and a bound must not rely on the cancellation; the cold-solve scatter in k adds 3e-4 on
ω. 5e-3 is 2× that sum, 9× the achieved, 6× tighter than the brief. Also asserted: the force-based and energy-based k agree to 1 %
(forces are the gradient of the energy), no "box too small" warning, the trajectory reaches
±0.8–0.9 Δ.

**Approximation gap (reported, rule 7).** ω_MD = 3237 cm⁻¹ at σ = 0.25 against 4401 cm⁻¹
experiment (−26.5 %) and 4201 cm⁻¹ LSDA (−22.9 %). The gap is the Gaussian nuclear smearing,
not the integrator: the same solver's σ → 0 curvature (E″ ≈ 0.37, M1 Task 10) gives
4405 ± 60 cm⁻¹ — a two-digit figure from a bias argument that **cannot tell experiment (4401)
from LSDA (4201)**, so it says only that the σ → 0 well is ≈ 1.85× stiffer than the σ = 0.25
one, not that this LSDA solver reproduces experiment — and the smearing's repulsive
2πσ²Σ_I ρ(R_I) term both lengthens the bond (1.789 vs 1.451 Bohr) and flattens the well
(k 0.20 vs 0.37). It shrinks with σ = h; the σ → 0 oracles, not this suite, are where the
frequency is physical (a σ-extrapolated 3-point k at Δ ≤ 0.05 would pin the LSDA value in
minutes).

### `nve_energy_drift_is_bounded`

What it measures: the integrator's secular energy error and its bounded fluctuation on real DFT
forces, through the `Dynamics` driver end to end (one `Step(1)` + `tick` per step; asserted
along the way: `energy_conserved` bitwise equal to `total` in NVE without tools, no errors, the
step counter, the history's last point equal to the recorded total, the driver's
`drift_per_ps` equal to a refit of the recorded series). Same start as above, 2000 steps =
0.5 ps.

Drift bound (brief, unchanged) 2e-4 Ha/ps. Budget: Verlet's energy error is a bounded
oscillation, not a drift; what can look like a drift is the work of the force irreproducibility
δF along the path, δW ≈ δF·v·dt per step ≈ 3e-6 × 1e-3 × 10 = 3e-8 Ha, random-walking to
~1.3e-6 Ha over 2000 steps, i.e. ≲ 3e-6 Ha/ps apparent — the brief's 6e-7·√2000 figure treats
the per-solve *energy* scatter as a random walk, which it is not (an energy error at one step is
not integrated; only force errors are), so the bound has far more than its stated 3× margin.
Fluctuation bound **derived, not pinned** (controller ruling after R6): Verlet's true energy
swings by exactly h²/4 of the mode energy peak to peak (`sim::integrator` module doc;
E ∈ [Ẽ, Ẽ/(1 − h²/4)]), E_mode = E₊ − E₀ from the same cold solves, h = ω_static·dt, plus the
GPU energy's f32 path scatter (6e-7 Ha per solve, M3 Task 7; white on the readout, so ±1 scatter
peak to peak): bound = 3 × (h²/4·E_mode + 2·6e-7) = **3 × (1.39e-6 + 1.2e-6) = 7.8e-6 Ha**,
measured as the peak-to-peak residual about the least-squares line (the drift has its own
bound). The brief's fixed 1e-5 would have had only 1.3× margin against the full
(un-detrended) swing had the surface been as stiff as the σ → 0 one (5e-6 + scatter); on the
σ = 0.25 surface it is met with 5×.

| quantity | value |
|---|---|
| **drift** (driver `drift_per_ps` = refit to 1e-9) | **−4.0e-6 Ha/ps** (bound 2e-4: 50× margin) |
| E_total start / mean / end | −0.984661276 / −0.984662789 / −0.984663150 Ha |
| **fluctuation**, detrended peak to peak | **1.92e-6 Ha** (derived bound 7.8e-6; brief's 1e-5 met); raw peak to peak 3.57e-6 |
| ω_MD over the 2000 steps (48 crossings) | ω_MD/ω_static − 1 = −5.51e-4 |
| ⟨T⟩ (COM removed, N_f = 3) | 24.9 K (harmonic expectation E_mode/(3k_B) = 25.1 K) |

### `thermostat_reaches_target_temperature`

What it measures: that CSVR on real DFT forces holds the target with the right K̄ and DOF,
along the realistic path (R7): `SetThermostat` on a molecule at rest → Maxwell–Boltzmann draw at
T (the readout starts at exactly 300 K, COM at rest — asserted) → CSVR each step.

**System: H₃⁺ at 49³, not the brief's H₂O at 33³** (the H₂O finding is below). H₃⁺ (2 electrons,
closed shell, one occupied level) has the same N_f = 3N − 3 = 6, no occupied near-degeneracy
near its minimum, and costs a k = 1 solve per step (82 ms at 49³, tol 1e-5), which buys the
window the statistics need. It is relaxed on the test surface first by quenched Verlet
(velocities zeroed whenever ΣF·v < 0, dt 0.5 fs, to max|F| < 1e-3 Ha/Bohr): from an
equilateral triangle of side 1.9 Bohr it converges in 6 steps to **side 2.1174 Bohr,
E = −1.123993 Ha, max|F| 9e-4** (the exact R_e is 1.65; the σ = 0.25 widening is the same
≈ 1.3× as H₂'s 1.79/1.45).

**Statistics, and why τ = 5 fs over 675 fs and not the brief's τ = 10 fs over 100 fs.** The
instantaneous kinetic temperature of 6 DOF fluctuates by σ_T/T = √(2/N_f) = 58 % (measured
61 %), so only a window mean is testable. That mean is governed by the slow part of
K — the vibrational mode energies E_i (exponential, Var (k_BT)²) and the rotational kinetic
energies (χ²₁, Var (k_BT)²/2): S = ΣE_i/2 + ΣK_rot, S̄ = 3k_BT, Var S = 2.25 (k_BT)², σ_S/S̄ =
50 % — which CSVR relaxes with time constant τ (dK = (K̄ − K)dt/τ + noise, Bussi eq. 9), so
over a window T_w ≫ τ, σ_mean/T ≈ 0.50·√(2τ/T_w). A toy model of exactly this ensemble
(3 harmonic modes + 3 free kinetic DOF, Verlet dt = 0.25 fs, CSVR each step, 300 seeds; Task 7
report, fix round 1) gives **6.7 % at τ = 5 fs over 675 fs** (9.4 % at τ = 10 fs, 3.0 % at
1 fs) with ⟨T⟩ = 1.006 ± 0.004 of the target — the formula holds for uncoupled modes. At the
brief's τ = 10 fs over the 100 fs window that 600 steps leave, σ_mean/T = 21 %: the 20 % bound
would have been a 1σ statement, failed by a third of fresh trajectories; τ = 10 fs needs a
1.1 ps window (≈ 4500 steps, 6 min) for 3σ. With τ = 5 fs over 2700 steps = 675 fs the
analytic σ_mean/T is 6.1 % (bound = 3.3σ for uncoupled modes); the test asserts that design
condition (analytic σ ≤ bound/3) so the window cannot be shortened unnoticed. CSVR integrates
its OU step exactly for any dt/τ and is canonical for every τ (Bussi 2007 §II), so the question
asked does not depend on τ. Settle window 100 steps = 25 fs = 5τ, longer than any H₃⁺ period,
for the in-phase Maxwell–Boltzmann start's K → V transient (Task 4 note).

**What two independent seeds say (fix round 1) — the gate is empirically 1.5–3σ, not 3.3σ.**
Seed 7 (the run of record): ⟨T⟩ = 335.1 K (+11.7 %); seed 11: 240.4 K (−19.9 %, passing by
0.1 %). Each run's own estimates agree with the analytic σ (blocks of 100 steps: 17.0 / 14.1 K;
σ_T√(2τ_int/T_w) with τ_int = 3.5 / 3.9 fs to the first zero of ρ: 18.5 / 16.2 K), yet the two
means differ by 95 K = 3.7σ of their difference. The autocorrelation of T explains why the
in-run estimates cannot see the width: ρ(lag) = 0.75 / 0.11 / 0.18 / 0.05 / −0.10 (seed 7) and
0.73 / 0.27 / 0.21 / 0.11 / −0.07 (seed 11) at 1 / 5 / 10 / 25 / 50 fs — a plateau of
≈ 0.05–0.1 out to 25 fs that neither 25 fs blocks nor a first-zero τ_int (12 fs) integrate. A
slow component carrying ~10 % of the variance with a correlation time of order 100 fs
(rotation–vibration exchange on the egg-box, large-amplitude motion of the soft smeared
molecule — the toy has neither) gives σ_mean ≈ 35 K and puts both seeds inside 1.7σ. No bias is
indicated: the deviations have opposite signs, and no positive systematic exists in the
mechanism (CSVR pins the velocity marginal at the true β, so ⟨K⟩ = K̄ exactly; Verlet's shadow
term lands in ⟨V⟩ = K̄/(1 − h²/4), as the NVE run shows at 24.9 vs 25.1 K; the warm-path heating
+1.8e-5 Ha/ps shifts the OU mean by 3e-5; the settle residual is −0.4 K). The 20 % bound stays
(never loosened); read it as a gross-error detector — a DOF or K̄ error of 2× fails it outright
— that is deterministic on one adapter and may trip on another by sampling alone. A 3 ps run
(≈ 12 000 steps, ≈ 15 min solo) would resolve the slow component; it is the handoff's
recommendation.

| quantity | value |
|---|---|
| **⟨T⟩ over steps 100..2800 (675 fs)** | **335.1 K (+11.7 % of 300 K; bound ±20 %; 1.9σ)** — mean over the first 100 steps 247 K (the in-phase start's K → V transient) |
| σ_mean: analytic / from 27 blocks of 100 steps / from τ_int | 18.2 K (6.1 %) / 17.0 K / 18.5 K (τ_int 3.5 fs) — all blind to the slow plateau above |
| **second seed** (`DYNAMICS_SEED=11`, reported, not asserted) | **⟨T⟩ = 240.4 K (−19.9 %)**; blocks 14.1 K, τ_int 3.9 fs → 16.2 K; sample σ_T 150 K (50 %); first-100-step mean 161 K |
| sample σ_T, T range | 183 K (61 %; canonical 58 %), [19, 1188] K |
| E_conserved drift (Bussi's effective energy; reported) | **+1.8e-5 Ha/ps at the viewer's tol 1e-5** (H₂ NVE at tol 1e-6: −4e-6) — what the viewer's drift readout will show in NVE/NVT at the interactive tolerance, well inside its amber threshold (1e-4) |
| first steps after the draw | T = 302, 279, 266, 236 K |

**H₂O at the `h2o` preset on the σ = 0.25 surface — documented limitation
(`thermostat_h2o_preset_known_limitation`, `#[ignore]`d: it now runs to the end, but is not a gate; run it with
`cargo test -p validate --release --test dynamics -- --ignored --nocapture thermostat_h2o`).**
Numbers of record are at the preset's own grid, **57³ = 14.25 Bohr** (`crates/app/presets/
h2o.scene`), tol 1e-5, cold GPU solves, no box warning; the first measurements were at the
brief's 33³ (8 Bohr, "box too small": face density 1.9e-4 at the preset, 7e-4 stretched) and are
marked as such — 0.045 Ha of the energy there is box error at the preset alone.

| quantity (57³ unless marked) | value |
|---|---|
| E at the preset geometry (O–H 1.81 Bohr, 104.5°) | −37.529795 Ha (13 SCF steps; max\|F\| 0.34 Ha/Bohr) |
| E at the 172° geometry the 33³ run reached (O–H 2.21/2.22) | −37.684659 Ha (15 steps) → **ΔE = 0.155 Ha** |
| relaxed minimum (quenched Verlet from the preset, 74 steps, no stall) | **O–H 2.338 Bohr, 177.2°, E = −37.689800 Ha: 0.160 Ha below the preset** |
| the same at 33³ (box too small) | 0.13 Ha, minimum beyond 172° (−37.616 and still falling) |
| release as a temperature, 2ΔE/(N_f k_B), N_f = 6 | ≈ 16 800 K (an upper bound: the energy flows through V as well as K); 33³ MD: T 518 / 1338 / 2376 / 3235 / 3818 K over the first five steps, back to ~300 K by step 20 under CSVR τ = 1 fs |
| **after R15 (cold retry), 57³ MD from the preset, τ = 10 fs, MB start, seed 7, 600 steps** | **all 600 steps complete**; the warm SCF stalls at step 2 (Δρ 8.5e-5 after 200 iterations — the same deterministic warm attempt that used to end the run) and at step 281, and both cold retries converge (the costlier step 24 s vs ≈ 1 s); T = 583 / 1899 / 3876 / 6097 K over the first four steps, 11 539 K at step 10, **3381 K at step 100**, 862 / 294 / 44 K at 200 / 400 / 600; E_KS −37.5298 → −37.6853 Ha (ΔE_KS −0.156 Ha; final O–H 2.41/2.42 Bohr, ≈ 158°), thermostat work −0.154 Ha; **⟨T⟩ steps 200–600 = 544 K (+81 %)**, block means 784 / 595 / 480 / 316 K; E_conserved drift −9.4e-4 Ha/ps (tol 1e-5 SCF lag and Verlet error at thousands of kelvin, not separated); 1.03 s/step (9.1 SCF steps, 369 readbacks), wall 618 s |
| before R15 (Task 7 fix round 1), same run | paused at **step 2**: "SCF did not converge after 200 iterations (Δρ = 8.5e-5)" — the same stall occurs in plain NVE from the preset, with or without thermal velocities and at tol 1e-6 as well (Δρ 6.7e-5 / 9.6e-5 / 7.6e-5; one tol-1e-6 case needed 148 SCF iterations at step 2 and failed at step 3), so it is the warm, density-extrapolated DIIS path on the preset's fast-changing density (0.34 Ha/Bohr forces; T passes 13 000 K and O–H 3.3 Bohr within 6 fs in NVE), not the thermostat or the tolerance; at dt 0.5 fs the same path converges every step but with 52–179 iterations on some, and at 33³ the 0.25 fs step converges — a DIIS stagnation at Δρ ≈ 1e-4 whose trigger is unresolved |
| warm-SCF stall, 33³ | step 110 at 172° (Δρ 6.0e-5); a cold GPU solve there converges in 13 steps (16 at tol 1e-6) and the CPU in 14; quenched relaxation stalls at step 39 (~150°) |

Reading: the Z = 8 nucleus smeared to σ = 0.25 Bohr is a very soft core, and the σ = h = 0.25
surface's water is a near-linear molecule with 2.34 Bohr bonds, 0.16 Ha (4.4 eV) below the
physical geometry it is loaded in — the same widening that takes H₂ from 1.45 to 1.79 Bohr.
Consequences: (1) the viewer's `h2o` preset is not an equilibrium of the surface it is simulated
on — pressing Play releases 0.16 Ha, the temperature readout climbs past 10 000 K within ten
steps (2.5 fs) while the molecule opens toward linear, and a 300 K thermostat at τ = 10 fs
brings it back to ~300 K only after ~100 fs; (2) the warm SCF still stalls on such a path, now
handled by R15's cold retry in `GpuKs::evaluate`/`CpuKs::evaluate` — a stalled step costs a
cold solve more (24 s at 57³ instead of ≈ 1 s) and the run continues; the stagnation itself
remains a solver item (fractional occupations are the real fix near level crossings), and a
cold retry that also fails still pauses the run with the last good state (Review Focus 3);
(3) the test stays `#[ignore]`d: it completes, but its ⟨T⟩ over the brief's window measures how
fast 0.16 Ha drains (544 K, +81 %), and it takes 10 min — a gate would start from the relaxed
minimum, as the H₃⁺ test does; (4) a smaller σ (finer h) moves the H₂O minimum back toward the
physical geometry.

### `rewind_retraces_on_the_gpu`

What it measures: time reversibility of the whole force–integrator path on the GPU. H₂ from
R₀ = 1.84 Bohr at rest, 50 steps, `flip_velocities`, 50 steps, flip → compared with the start.
Verlet is exactly reversible in f64 (`sim::integrator::tests::verlet_is_time_reversible`, 1e-10
over 200 steps); here the backward forces are *re-solved* from different warm (extrapolated)
states, so they are not bitwise the forward ones and the error is the warm-path force
irreproducibility δF integrated: δr ≈ (δF dt/m)·dt·√(N³/3) — 1e-4 Bohr at δF = 3e-6 Ha/Bohr
(the GPU-vs-CPU figure, which contains the CPU's own Δρ term), 3e-5 at 1e-6.

| quantity | value |
|---|---|
| at the turn: \|R − R₀\|, v_x(0) | 0.0374 Bohr, −3.53e-4 Bohr/a.u. |
| **max \|Δr\| after n → flip → n → flip** | **5.80e-6 Bohr** (brief 1e-4) |
| max \|Δv\| | 1.79e-8 Bohr/a.u. (1.9e-7 Bohr per step) |
| E_total back − start | −1.4e-8 Ha |
| implied rms δF between the passes | ≈ 1.7e-7 Ha/Bohr |

Bound `REWIND_BOUND`: 1e-4 → **5e-5** (2× tighter than the brief, 9× the achieved; δF ≈ 1.5e-6
at the bound). Not 3× the achieved: the GPU is bit-reproducible on one adapter, so 5.8e-6 is a
single sample of an unmodelled, adapter-dependent quantity (the backward leg's first density
extrapolation points the wrong way in time, and the SCF converges to tol from a different
guess); what limits it is the warm-path force scatter, i.e. the SCF tolerance and the f32 floor,
not the integrator. The achieved 5.8e-6 is the RX 7900 XTX figure.

### Runtime (in-suite, four tests concurrent, GPU evaluations serialised)

| test | system | grid | tol | solves | GPU s | ms/solve | SCF steps/solve | readbacks/solve | test wall s | result |
|---|---|---|---|---|---|---|---|---|---|---|
| `rewind_retraces_on_the_gpu` | H₂ | 49³ | 1e-6 | 101 | 9.0 | 89 | 5.5 | 206 | 11.4 | max Δr = 5.8e-6 Bohr (bound 5e-5) |
| `h2_vibration_…` | H₂ | 49³ | 1e-6 | 417 | 33.9 | 81 | 5.6 | 207 | 50.7 | ω_MD/ω_static − 1 = −5.55e-4 |
| `nve_energy_drift_is_bounded` | H₂ | 49³ | 1e-6 | 2018 | 120.3 | 60 | 5.5 | 202 | 298.9 | drift −4.0e-6 Ha/ps, fluctuation 1.9e-6 Ha |
| `thermostat_reaches_target_temperature` | H₃⁺ | 49³ | 1e-5 | 2809 | 188.9 | 67 | 5.5 | 239 | 352.0 | ⟨T⟩ = 335 K (+11.7 %, 1.9σ) |
| **suite** | | | | 5345 | **352** | 66 | | | **358 (5 m 58 s)** | 4 passed, 1 ignored |

(Solo, the H₂ MD step is 63–66 ms and the H₃⁺ step 82 ms; in-suite the MD walls are 142 / 121 ms per step because the two long runs alternate on the GPU. The fix-round-1 re-run, made concurrently with a second-seed thermostat run on a GPU that was also otherwise busy, took 23 min with 230–440 ms per solve and bit-identical physics — a reminder that the walls here are machine-state, not physics; R3.)

The suite's GPU time is the sum of the "GPU s" column (one evaluation at a time); the test
walls include waiting. Budget ≤ 8 min: 6 min. The per-step cost is latency-bound
(~5.5 SCF steps and ~200–240 readbacks per solve at 63–82 ms), so the 2000-step NVE run and the
2800-step thermostat run dominate.

## Surrogate (M5 Task 7, `tests/surrogate.rs`, 2026-10-14, release, RX 7900 XTX)

Run with `cargo test -p validate --release --test surrogate` (2 min 51 s in-suite, five tests
concurrent with their GPU teacher solves serialised, GPU sum 165 s; skips without a GPU adapter;
`--nocapture` prints every number; each test appends its row to the gitignored
`last-run-surrogate.md`; `SURROGATE_SEED` overrides the sampling/held-out seed for reported
second-seed runs). Wall times are reported, never asserted (ruling R3).

**Design.** The teacher is `GpuKs` on the M4 dynamics suite's discrete problem — σ = h = 0.25,
49³ (12 Bohr box), tol 1e-6 — so the M4 pins (R_e = 1.78894 Bohr, k = 0.20003 Ha/Bohr², ω_MD =
3236.9 cm⁻¹) are the surface the models learn, and H₃⁺ runs at the same 49³ (the budget allowed
it; the 33³ box of the Task 6 fixture adds a confinement energy that changes with the
molecule's distance to the faces — see the measured floor below). Every test builds its own
`SurrogateKs` over its own locked teacher; the passes are seeded and the GPU bit-reproducible,
so the three H₂ models are the same model (identical fit reports). The model is linear ACE
(Drautz, PRB 99, 014104 (2019)): E = Σ_i E_atom(Z_i) + Σ_k c_k B_k on body-2/body-3 invariants
(n_max 8, l_max 3, r_cut 6, degree cap 4: P = Q′ = 29 for one species), fitted by Bayesian ridge
regression with the evidence choosing λ and the two noise levels s_E, s_F (Bishop, *Pattern
Recognition and Machine Learning*, §3.3–3.5; MacKay, Neural Comput. 4, 415 (1992)); the
fallback is by the posterior predictive std and the training envelope. Integrator: velocity
Verlet (Verlet, Phys. Rev. 159, 98 (1967); Swope, Andersen, Berens & Wilson, J. Chem. Phys. 76,
637 (1982)). Nothing per element enters beyond Z as the species label and the isotope mass
(rule 1); the H reference energy is the teacher's isolated atom by ruling S9 (E_H(49³) =
−0.424224 Ha, doublet — above LSDA's −0.479 by the smearing and the box, as the H₂ R_e is above
the σ → 0 value).

**Honesty gap (spec §7, rule 7).** The surrogate inherits every approximation of its teacher —
the Gaussian smearing (H₂ ω 3237 vs 4401 cm⁻¹ experiment; H₃⁺ stretch 2509 and bend 1960 cm⁻¹
vs 3178 / 2521 cm⁻¹ experiment, the same ≈ 0.78 softening), LSDA, the grid — and adds its own
fit error; its reported uncertainty is the *model's* std relative to the teacher, not a physics
one. The teacher itself has a floor: the SCF path scatter (6e-7 Ha per H₂ solve, M3) and the
egg-box ripple, **measured here for H₃⁺ at 49³ as 3.1e-7 Ha / 2.2e-6 Ha/Bohr rms** over 8 rigid
sub-grid shifts (±h/4 per axis, cold solves) — 500× below Task 6's 33³ figure (1.67e-4 /
2.53e-4), which was therefore the 8 Bohr box's confinement, not the grid. "Model vs fresh DFT"
differences are √(model error² + floor²); at 49³ the floor is inert for every bound below.

| Quantity | Value | Source | Used by |
|---|---|---|---|
| H₂ DFT-MD ω_MD at σ = h = 0.25, 49³, from R_e + 0.05 at rest, dt 0.25 fs | 1.474855e-2 a.u. = 3236.9 cm⁻¹ | this file, "Dynamics (M4)" (400 steps, 9 crossings; the 2000-step ratio differs by 4e-6) | `surrogate_reproduces_the_h2_frequency` (`OMEGA_MD_M4`) |
| H₂ R_e(σ = 0.25, 49³) | 1.78894 Bohr | "Dynamics (M4)" | the MD start; sanity check (re-measured 1.78894, Δ 3e-6) |
| H₃⁺ ν₁ (A₁′ stretch), ν₂ (E′ bend), experiment | 3178.3, 2521.3 cm⁻¹ | Oka, Phys. Rev. Lett. 45, 531 (1980); Lindsay & McCall, J. Mol. Spectrosc. 210, 60 (2001) | reported against (not asserted) |
| Verlet energy swing of a harmonic mode started from rest at x = Δ | exactly h²/4·E_mode peak to peak, h = ω dt | `sim::integrator` module doc: invariant ½mv² + ½kx²(1 − h²/4) ⇒ E ∈ [Ẽ, Ẽ/(1 − h²/4)], Ẽ = E_mode(1 − h²/4) | `surrogate_nve_is_tight` |

### `surrogate_reproduces_the_h2_frequency`

What it measures: that the model reproduces the teacher's H₂ curve and its dynamics after one
sampling pass. Pass on H₂ at R = 1.6 (the brief's scene): 12-point scan 0.7–1.8 × 1.6 = 1.12–2.88
Bohr, 60 Gaussian perturbations of 0.15 Bohr, 200-step burst at 1000 K (272 samples, 273
teacher solves, 67–131 s in-suite, ≈ 30 s of GPU), one refit (18 ms). Then (i) E and F at 20
fresh on-axis points 1.325 + 0.05k Bohr — never a training scan point: 50j − 160k = −205 has no
integer solution — against fresh DFT; (ii) surrogate-MD from the M4 start (DFT R_e + 0.05 at
rest, dt 0.25 fs, 4000 steps), every step answered by the model (asserted), ω from the upward
zero crossings against the M4 pinned ω_MD.

| quantity | value |
|---|---|
| fit (272 samples, Q′ 29, 1904 rows) | rmse E/atom 1.11e-5, F 4.47e-5; s_E 1.11e-5/atom, s_F 4.48e-5; λ 6.75; γ 9.1 (E 1.5, F 7.6); dof_E 270, dof_F 1624; training residuals E/atom 1.35e-4, F 5.01e-4 (bound 6.2e-3) → trusted |
| **20 fresh points: rms \|ΔE\|** | **2.54e-6 Ha** (max 5.18e-6 at R = 2.125; tier 1e-5) — a smooth residual curve, ±5e-6 across 1.3–2.3 Bohr |
| **20 fresh points: rms per-component \|ΔF\|** | **2.04e-5 Ha/Bohr** (max 1.27e-4 at R = 2.275, the sparse end of the scan; tier 2e-4) |
| model std along the fresh points | F 4.5e-5 per component (the policy bound is 2e-3), epistemic E 7–11e-7 per atom |
| surrogate-MD | 4000/4000 Surrogate steps, 0 teacher calls; 97 upward crossings, period 10.3050 fs (M4 DFT-MD: 10.3049); R − R_e ∈ [−0.0478, +0.0500] |
| **ω_sur/ω_MD(M4) − 1** | **−7.0e-6** (ω_sur = 1.474845e-2 a.u. = 3236.9 cm⁻¹). **On the M6 branch: −6.516e-6** — identical before and after the providers refactor (Task 1 review I1) and re-measured in M6 Task 7's run; the shift from main's −7.0e-6 is Task 5's AME2020 retabulation and the M5b merge, both legitimate |
| model statics (bisection + 3-point k on the model, 0 teacher calls) | R_e^model − R_e^DFT = +1.9e-5 Bohr; **k_model/k_DFT − 1 = −2.2e-5**; E_mode 2.3869e-4 vs DFT 2.3885e-4 |
| decomposition | ω_sur/ω_static^model − 1 = −5.51e-4 (M4's DFT value −5.55e-4: the same Verlet +9.72e-4, truncation and anharmonic −1.52e-3 terms on the same curve) |
| speed | 1.00e5 steps/s (10.0 µs/step, N = 2) against the teacher's 102 ms/solve — the Task 7 run; **2.4e5 since Task 9** (variance cadence 4, value-only descriptors; "Speed" below) |

Bounds. Tiers `E_TIER_H2` = 1e-5 Ha and `F_TIER_H2` = 2e-4 Ha/Bohr on the rms (the statement
Task 6's fixture made): kept at the spec's values — the measured maxima (5.2e-6, 1.27e-4) are
more than half of them, so there is no honest headroom to tighten. The spec's per-point
"within" is asserted on the energy as well (max |ΔE| ≤ 1e-5; achieved 5.2e-6, deterministic per
adapter); the force max (1.27e-4, 63 % of the tier, at the sparse end of the scan) is reported,
not asserted — a pin on this adapter.
`OMEGA_BOUND`: the spec's 0.5 % is a 1 % statement on the model's k(R_e) — Verlet's +h²/24 and
the anharmonic shift are identical in both runs (same start, same dt) and cancel exactly; a
constant force offset does not move ω, a slope dΔF/dR = δk does, and at the force tier over the
0.1 Bohr the oscillation spans a worst-case slope is 2e-3 Ha/Bohr² = 1 % of k, so the force
tier implies the spec's 0.5 % (the tightened 1e-3 is a fit-noise statement instead: a model
exactly at the force tier with a slope could pass F and fail ω). The fit averages the teacher's force
scatter s_F = 4.5e-5 over ~270 samples spread ~0.15 Bohr rms, so the expected δk/k ≈
s_F/(k σ_R √n) ≈ 1e-4 and δω/ω ≈ 5e-5. Achieved −7.0e-6 → **1e-3** (20× the modelled scatter,
5× tighter than the spec; never loosened, rule 6). Also asserted: no "box too small" warning on
the statics and the fresh points (the pass's stretched/hot samples touch the 1e-5 face-density
threshold on 2 of 273 solves, 1.04e-5 and 1.43e-5 — reported, not asserted: those samples are
the teacher's answer for this box, and model and DFT are compared on the same surface).

### `surrogate_nve_is_tight`

What it measures: velocity Verlet on the surrogate's own surface — smooth, deterministic, no
SCF, no f32 path — so the integrator's energy error is the pure shadow-Hamiltonian oscillation
and anything else is the model's forces failing to be the gradient of its energy. 10 ps =
40 000 steps from the model's own R_e + 0.05 at rest (its statics by the M4 helper on the model:
R_e 1.78896, k 0.20003, E_mode 2.3869e-4 — 0 teacher calls, asserted); then 1000 steps → flip →
1000 → flip.

| quantity | value |
|---|---|
| **drift** (least-squares slope of E_total) | **−2.6e-11 Ha/ps** (bound 1e-8; spec 1e-6) |
| **fluctuation**, detrended peak to peak / h²/4·E_mode | **1.4319e-6 / 1.3880e-6 = 1.032** (h = 0.1525; bound 1 ± 0.10); raw peak to peak 1.432e-6 |
| ω_MD over 970 crossings vs the model's ω_static | −5.52e-4 (the M4 decomposition again) |
| **rewind** max \|Δr\| after 1000 → flip → 1000 → flip | **1.5e-14 Bohr** (bound 1e-12; \|R − R₀\| at the turn 0.051); max \|Δv\| 1.5e-17 |
| teacher calls after the pass | 0 (statics, NVE, rewind) |
| speed | 0.9–1.1e5 steps/s (9–11 µs/step, N = 2; the pass ran concurrently with the other tests) — the Task 7 run; **2.39–2.43e5 since Task 9** ("Speed" below) |

Bounds. Fluctuation: for a harmonic mode started from rest at x = Δ the Verlet energy swings
by exactly h²/4·E_mode peak to peak (table above); the 3.2 % excess is the anharmonicity of
the H₂ well at this amplitude — with k = 0.20 and D ≈ 0.09 Ha the Morse α = √(k/2D) ≈ 1.05
Bohr⁻¹, so |F|² at the compressed turning point exceeds the stretched one by ~2·1.5αΔ ≈ 16 %
and the |F|²-weighted turning-point term of the Verlet error grows by a few %, the measured
size and sign — plus the 3-point truncation of k (E⁗Δ²/12 ≈ 0.1 %); `FLUCT_TOL` = 10 %
(controller) is kept — 3× the measured deviation, and the anharmonic term is amplitude-
dependent, not modelled to better than that. **The swing ratio is also what bounds a
force–energy mismatch of the model in this test**: 10 % of h²/4·E_mode = 1.4e-7 Ha over Δ =
0.05 Bohr is |δF| ≲ 3e-6 Ha/Bohr (δk/k ≲ 5e-4). Drift: the run is on-axis from rest — 1-D
motion under a central force — so any mismatch δF(R) between the model's forces and
−dE_model/dR is the gradient of some Φ(R) and E_total = [K + Φ]_shadow + [E_model − Φ](R): a
conserved part plus a periodic function of R, never a secular term (a non-conservative mismatch
needs a 3-D trajectory to pump energy). The drift therefore measures f64 rounding plus the slope
the 1.4e-6 Ha Verlet oscillation leaks into a least-squares line over a non-integer number of
periods, ≤ A/(T·N_periods) ≈ 1.4e-10 Ha/ps; the achieved −2.6e-11 is at that floor.
`DRIFT_BOUND` = **1e-8 Ha/ps** (spec 1e-6): 70× the leakage floor — a rounding/bookkeeping
statement about the integrator on an analytic surface, and nothing more is claimed for it
(rule 7; the earlier "28·δF per ps" sensitivity claim was wrong for a 1-D run and is withdrawn).
Rewind: the model is bitwise reproducible, so only rounding accumulates, √2000 × ε
× R ≈ 5e-15 Bohr (achieved 1.5e-14); `REWIND_BOUND` = **1e-12** (controller 1e-9), 100× the
rounding walk and 4e5× below M4's GPU figure, where the backward forces are re-solved.

### `surrogate_knows_what_it_does_not_know`

What it measures: the fallback policy on the trained H₂ model (the decision list in
`providers::surrogate`). The rule the policy applies is **fallback by uncertainty**: an unseen
species or a different charge/grid/σ/multiplicity is refused unconditionally (rules 2–3), while
an unseen *size* or *geometry* of known species is allowed only when the model's own posterior
and training envelope accept it (rules 5–7).

| probe | verdict (asserted where bold) | numbers |
|---|---|---|
| H₂ R = 1.8 (in range) | **Surrogate** | uncertainty 4.5e-5 Ha/Bohr per component (bound 2e-3), epistemic E 6.8e-7/atom |
| (a) H₂ R = 4.0 (scan ended at 2.88) | **Fallback, model consulted** (uncertainty present) — rule 5, envelope: atom 0 is 29.3 training spreads outside on b3[n=0;n=1;l=3] | force std 1.26e-2 per component (bound 4.6e-3 at \|F\| = 4.6e-2 — the variance would also have refused), epistemic E 3.4e-3/atom (bound 1e-3); teacher − model **−5.51e-2 Ha**; sample kept (273) |
| (b) charge +1 at R = 1.8 | **Fallback, unconditional** (rule 3, note names "charge 1"), no uncertainty, no gap | **sample not kept** (273) |
| (d) walk R = 3.6, 3.8, 4.0, 4.2, 4.4 | 5 Fallbacks by the envelope (18–32 spreads) | gaps −1.1e-2, −2.7e-2, −5.5e-2, −9.7e-2, −1.48e-1 Ha; refit over 278 samples (74 ms): s_E 1.41e-5/atom, s_F 4.50e-5, training residuals 1.66e-4 / 5.68e-4 → trusted |
| **(d) R = 4.0 after the refit** | **Surrogate**; **ΔE vs cold DFT +5.2e-5 Ha** (bound 1e-4), max \|ΔF\| 6.7e-5 | uncertainty 4.6e-5, epistemic E 5.5e-6/atom; H₂ at 1.8 still Surrogate |
| (c) neutral H₃, side 1.8 (3 electrons, doublet) | **Fallback, model consulted** — rule 5: atom 1 is 19.9 spreads outside on b3[n=0;n=2;l=0] | force std 7.4 Ha/Bohr, epistemic E 4.4 Ha/atom, teacher − model **−27.0 Ha** (−2.4 Ha with the pre-walk model): the dimer model has no opinion worth having about a three-centre radical, and says so |
| after one refit with the H₃ sample (reported, not asserted) | **untrusted**: "max training force residual 3.2e-2 > 4.2e-3 Ha/Bohr"; H₂ at 1.8 → Fallback (rule 4) | the evidence settled on s_F 2.4e-3 (from 4.5e-5), γ_F 1.0: one sample of a new bonding situation that Q′ = 29 dimer-shaped columns cannot carry makes the S8 trust rule refuse the whole model until a pass on H₃ teaches it — the designed answer, and the reason the H₃ probe runs *after* the R = 4.0 check (the first run had it before, and the walk's refit was then untrusted) |

`OOD_E_BOUND` = 1e-4 Ha (controller) kept: the achieved 5.2e-5 has only 2× headroom — the
region 3.6–4.4 Bohr is taught by five samples, and the model's ΔE there is at the fit's s_E
(1.4e-5/atom) times a leverage of ~2; more walk samples would shrink it, which is the
provider's active-learning job, not this test's.

### `h3plus_tier`

What it measures: the second system of spec §7 (σ = 0.25 water is near-linear and stalls the
warm SCF — M4). H₃⁺ (charge +1, 2 electrons, closed shell) relaxed on the teacher surface by the
M4 helper (quenched Verlet: 6 steps from side 1.9 to **2.1174 Bohr, E = −1.123993 Ha**, max|F|
9e-4; the M4 thermostat test found the same side at tol 1e-5); the egg-box floor measured in
place (above); one default sampling pass (60 perturbations, 3 × 12 scan points over 0.7–1.8 ×
2.117 = 1.48–3.81 Bohr, 200-step 2000 K burst: 296 samples, 297 solves) → body-3 model; 20
held-out geometries (Gaussian 0.15 Bohr per coordinate, seed ⊕ `HELD_OUT_SALT`) against fresh
DFT; the symmetric stretch (A₁′: every radius × 1.03, at rest — the trajectory stays equilateral
by symmetry) and the bend (one E′ component: atom 0 out by 3 %, atoms 1–2 in by 1.5 %,
recentred; C₂ᵥ is preserved, Q = (2d₀ − d₁ − d₂)/√6 oscillates at ω_bend with whatever
breathing admixture the anharmonic coupling adds — identically in both runs) from the same
starts through the teacher (DFT-MD) and through the surrogate, 320 steps = 80 fs each.

| quantity | value |
|---|---|
| fit (296 samples, Q′ 29, 2960 rows) | rmse E/atom 5.10e-6, F 2.01e-5; s_E 5.12e-6/atom, s_F 2.02e-5; λ 1.95; γ 27.3 (E 1.5, F 25.8); dof_E 294, dof_F 2638; training residuals 7.25e-5 / 3.20e-4 → trusted |
| **egg-box floor at 49³** (8 rigid shifts, cold) | **E rms 3.10e-7 Ha**, force components rms 2.22e-6 Ha/Bohr (33³, Task 6: 1.67e-4 / 2.53e-4) |
| **20 held-out: rms \|ΔE\|** | **3.58e-6 Ha raw** (max 1.26e-5 on the most compressed draw, side 1.51); floor-corrected √(rms² − floor²) = 3.57e-6 |
| **20 held-out: rms per-component \|ΔF\|** | **8.91e-6 Ha/Bohr** (max 3.87e-5 at \|F\| = 0.137) |
| model std along the held-out set | F 2.0e-5 per component, epistemic E 3.1–4.5e-7/atom |
| **symmetric stretch** (DFT-MD 6 crossings, period 13.295 fs) | ω_DFT **2508.9 cm⁻¹**, ω_sur 2508.1; **ω_sur/ω_DFT − 1 = −3.0e-4** |
| **bend** (DFT-MD 4 crossings, period 17.017 fs) | ω_DFT **1960.1 cm⁻¹**, ω_sur 1959.3; **−4.5e-4** |
| surrogate-MD | 320/320 Surrogate steps on both modes |
| **second seed** (`SURROGATE_SEED=11`: another pass — perturbations, burst — and other held-out draws; reported, not asserted) | fit s_E 4.81e-6/atom, s_F 2.10e-5; **held-out rms \|ΔE\| 1.73e-6 Ha (max 5.09e-6), rms \|ΔF\| 9.96e-6 Ha/Bohr (max 6.69e-5); stretch −2.4e-4, bend −4.6e-4** — inside the 8× / 22× / 11× margins (17× / 20× / 11× on this seed); same relaxed side and floor (deterministic) |
| speed | 6.4–7.6e4 steps/s (13–16 µs/step, N = 3) against the teacher's 72 ms/solve (k = 1 solve) |

Bounds. Energy tier judged as √(rms² − floor²) (controller ruling: the floor is the teacher's
own scatter, which the held-out DFT values carry and no model removes; at 49³ the correction
is 1e-9 and the raw rms is the number — reported against the tier as well). Achieved 3.6e-6
(max 1.3e-5) at a fit noise of 5e-6/atom, so `E_TIER_H3` is tightened from the spec's 1e-4 to
**3e-5 Ha** (8× the rms, 2.4× the max) and `F_TIER_H3` from 2e-3 to **2e-4 Ha/Bohr** (22× the
rms, 5× the max): the held-out draws are seeded and the teacher is bit-reproducible per
adapter; the second seed (table) moved the energy rms by 2× (1.73e-6 vs 3.58e-6: the 1/√(2n)
statistic of a 20-draw rms plus a different pass) and the force rms by 12 %, both well inside
the margins. `FREQ_BOUND`: a period from 4–6
crossings located by linear interpolation at 44–68 samples per period is good to ~1e-3
(the bend's 4 crossings are the limit), achieved −3.0e-4 / −4.5e-4 → spec 2 % tightened to
**5e-3**. Also asserted: both surrogate-MD runs answered by the model throughout; no box
warning on the relax, floor and DFT-MD solves (the pass's hot burst touched the 1e-5
face-density threshold once, 1.02e-5 — reported). Approximation gap (rule 7): 2509 / 1960
cm⁻¹ against experiment's 3178 / 2521 — the σ = 0.25 softening, ≈ 0.78 on both, the same as
H₂'s 3237/4401 — and against each other the bend/stretch ratio is 0.781 (experiment 0.793).

### `active_learning_run_converges`

What it measures: the provider learning on the fly with no pass — the live path of spec §5.
H₂ at the M4 R_e + 0.05 along a generic axis (0.6, 0.64, 0.48), rotating with k_B·300 K of
kinetic energy about its centre (period 194 fs; the descriptors are rotation invariant, so the
rotation adds no features — it turns the egg-box ripple into a noise the fit must classify as
such), a references-only pass first (every count 0: the one job is the isolated H atom by
ruling S9 — no fit exists without E_atom(H), and this is how the worker's pass computes it),
then 300 Verlet steps at 0.25 fs through `SurrogateKs` with `audit_every` = 50 (S7) and
`refit()` whenever `pending_refit()`.

| quantity | value |
|---|---|
| timeline (25 steps per group, S = Surrogate, F = Fallback) | `FFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFSSSSSSSSSSSSSSSSSSS SSSSSSSSSSSSSSSSSSSSSSSSS SSSSSFSSS…` — 31 rule-1/rule-4 fallbacks, then an audit every 50 answered steps |
| first trusted fit | before step 32: n = 32 ≥ Q′ = 29, dof_E 31 ≥ 30, dof_F 188; s_E 1.8e-7/atom, s_F 2.4e-6; training residuals 3.9e-7 / 4.4e-6 (the single-orientation egg-box noise of a slowly rotating dimer is at the 1e-7 level) |
| **fallback fraction over the last 100 steps** | **2.0 %** (2 audits; bound 5 %, controller 10 %) |
| **\|teacher_gap\| on the last fallback** (step 281, an audit) | **5.1e-6 Ha** (bound 3e-5, controller 1e-4) |
| audit gaps (steps 81, 131, 181, 231, 281) | +5.8e-6, +4.9e-6, +1.1e-6, +2.6e-6, +5.1e-6 Ha — all finite, none failed (bound 1e-3, reported) |
| refit jumps booked on the first evaluate after each refit | −7.9e-7 (onto the model surface at step 32), then +2.1e-7, +1.7e-7, +2.6e-8, +6.8e-8, +1.5e-7 Ha |
| final model (37 samples) | s_E 8.0e-7/atom, s_F 4.3e-6, training residuals 2.6e-6 / 1.2e-5 → trusted; bond 1.795 Bohr, T 249 K |
| **total DFT calls** | **38** for 300 steps (1 reference + 1 start + 31 fallbacks + 5 audits); 37 refits of ≤ 8 ms; solo wall 3.8 s |

Bounds. `AL_FALLBACK_BOUND`: the controller's 10 % is tightened to **5 %** — the achieved 2 % is
the audit cadence's own floor (2 audits per 100 answered steps), and 5 % leaves room for two
variance fallbacks on top. `AL_GAP_BOUND`: 1e-4 → **3e-5 Ha** — an empirical margin, not a
budget: 5× the largest of the five audit gaps (1.1–5.8e-6), which sit 3–7× above the N·s_E
floor of 1.6e-6 (the mild misspecification of a 29-column fit over a 0.1 Bohr window with 32–37
samples; the test has no RNG and the adapter noise is ~1e-6, so it will not flake; if it ever
trips, more samples, never a looser bound). `AUDIT_GAP_BOUND` = 1e-3 kept as the
controller's reporting bound (a failed audit — |gap|/N > 1e-3 per atom — would untrust the
model and show in the timeline as a run of F's). Also asserted: the last fallback carries a
gap (the model was trusted there), the model is trusted at the end, no box warning.

### Speed (steps per second, release, Ryzen 7 9800X3D / RX 7900 XTX; reported, never asserted)

| system | surrogate (hot path, f64, one thread) | DFT teacher (GpuKs 49³, warm, tol 1e-6) | ratio |
|---|---|---|---|
| H₂ (N = 2) | **2.39–2.43e5 steps/s** (≈ 4.2 µs/step; after Task 9 — the Task 7 run had 0.9–1.1e5) | 9.8 steps/s (102 ms/solve, 7.8 SCF steps, 338 readbacks) | ~2.4e4 |
| H₃⁺ (N = 3) | **1.36e5 steps/s** (7.4 µs/step; final fix wave, machine under load — the Task 7 run had 6.4–7.6e4) | 13.9 steps/s (72 ms/solve, k = 1) | ~1e4 |

The spec's expectation (≥ 10⁴ for H₂) is met by 24×. In the Task 7 run the step was dominated by
the descriptor and the O(Q′²) predictive variance at `variance_cadence` = 1; Task 9 tuned both
(below). The final fix wave's suite run (2026-10-14, the GPU shared with other applications:
teacher solves 128–190 ms instead of 72–107, suite 4 min 52 s) measured 1.73–1.95e5 steps/s for
H₂ and 1.36e5 for H₃⁺, every accuracy number identical to the run of record (rms ΔE 2.54e-6,
rms ΔF 2.04e-5, ω −7.03e-6, drift −2.63e-11 Ha/ps, swing 1.032, rewind 1.5e-14, AL 38 calls /
2 %, H₃⁺ 3.58e-6 / 8.91e-6, modes −3.0e-4 / −4.5e-4) and the OOD relearn ΔE(4.0) 4.8e-5 (record
5.2e-5; in this run the walk's R = 4.4 answer was box-affected, 2.3e-5 > 2e-5, and not kept).

#### Task 9 (2026-10-14): cost by species, atoms and samples; variance cadence

`tests/surrogate_perf.rs` (ignored, CPU only, one thread per model; reported, never asserted;
`cargo test -p validate --release --test surrogate_perf -- --ignored --nocapture`; the run's table
also goes to `last-run-surrogate.md`). Synthetic pair-Morse teacher with a 1e-5 Ha / 1e-4 Ha/Bohr
scatter on random clusters (the costs depend on Q′, N and P, not on what is fitted); body-2 +
body-3 at D = 4; every step answered. "Steps/s" = `SurrogateKs::evaluate` (predict + the
envelope's value-only describe) on an N-atom cluster jittered each step, at variance cadence k.

| S | Q′ | P | first refit | N | predict µs | + variance µs | describe (with gradients) µs | values only µs | steps/s k = 1 | k = 4 | k = 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 29 | 89 | 6 ms | 2 | 1.4 | 2.4 | 1.6 | 1.0 | 148 789 | 260 049 | 309 394 |
| 1 | 29 | 89 | | 6 | 9.0 | 10.4 | 13.0 | 6.6 | 28 074 | 48 267 | 56 279 |
| 1 | 29 | 89 | | 20 | 107 | 224 | 271 | 74 | 2 311 | 3 896 | 4 732 |
| 2 | 176 | 236 | 0.76 s | 2 | 2.7 | 49 | 2.6 | 1.6 | 18 504 | 56 126 | 102 240 |
| 2 | 176 | 236 | | 6 | 12.3 | 168 | 26 | 9.6 | 5 658 | 16 649 | 24 815 |
| 2 | 176 | 236 | | 20 | 102 | 685 | 221 | 87 | 1 052 | 2 642 | 3 573 |
| 4 | 1232 | 1292 | **260 s** | 4 | 16.9 | 4 464 | 17 | 7.0 | 218 | 834 | 2 022 |
| 4 | 1232 | 1292 | | 6 | 26.5 | 7 128 | 46 | 14.7 | 145 | 558 | 1 290 |
| 4 | 1232 | 1292 | | 20 | 178 | 25 247 | 874 | 122 | 40 | 145 | 347 |

Refit wall against stored samples (first fit over the store / incremental after 25 more samples,
the viewer's refit cadence): S = 1 (Q′ 29, 2-atom samples) 7 / 5 ms at P = 250, 8 / 8 at 500,
15 / 7 at 1000, 26 / 11 at 2000 (the viewer measured 5–22 ms at 290–400 H₂ samples); S = 2 (Q′
176, 6-atom samples) 870 / 828 ms, 731 / 627, 857 / 644, 1 028 / 596 — the evidence search's
Choleskys of the Q′ × Q′ system dominate, not P.

Reading it:
- **Variance cadence default k = 4** (`FallbackPolicy::variance_cadence`; Task 9). The variance is
  O(N·Q′²) per evaluation: ≈ 2 µs at S = 1, 50–700 µs at S = 2, 4.5–25 ms at S = 4, i.e. 60–99 % of a
  step from S = 2 up. k = 4 buys 1.7× (S = 1) to 3.6× (S = 4) and bounds the reuse of a verdict
  to three steps of 0.25 fs (a 2000 K hydrogen moves ≈ 0.13 Bohr in four steps; the envelope and
  the quarantine fence are still checked every step, an edit, a refit, a policy change or a
  rejection always re-checks — the invariant `the_variance_is_checked_at_least_every_k_…` asserts).
  k = 10 would buy another 1.2–2.4× at S ≥ 2 with a 2.5 fs blind window; not taken.
- **The envelope's descriptor** now uses `surrogate::describe_values` (values without the gradient
  lists, bit-identical): at N = 20 74–122 µs instead of 271–874 µs, which had been more than the
  prediction itself.
- **S = 4 is not interactive yet**: 145 steps/s at N = 20 even at k = 4, and a first refit of
  260 s (Q′ 1232). The plan's spectrally truncated Σ (top-r eigenpairs, behind a setting, with the
  calibration's coverage measured) is the remedy for the variance; the refit needs the evidence
  search on the reduced system too. Not done in Task 9 (re-scoped); recorded for the handoff.

#### The viewer's estimated mode (H₂, RX 7900 XTX, release)

| audit cadence / mode | Task 8 (synchronous audit) | Task 9 (background auditor, ruling S14; k = 4) |
|---|---|---|
| audit every 2000 (default), worker test H₂ 41³, Δsim/Δwall over 3 s | 3 474 fs/s (×1 112 the GPU DFT run) | 24 739–27 188 fs/s after item 2, **38 816 fs/s (×9 643)** after item 5 |
| audits off, same test | 31 655 fs/s | 43 561 fs/s |
| headless screenshot run, H₂ 49³, speed 1000 | 4 000 fs/s | 39 841–41 969 fs/s over 2 s (0.005–0.009 ms a step) |

The validate suite's own surrogate MD (H₂ 49³, policy default, audits off in those tests) went
from 0.9–1.1e5 to **2.39–2.43e5 steps/s** (`surrogate_nve_is_tight`, `…_h2_frequency`, run of
2026-10-14 after Task 9), every accuracy number unchanged (rms ΔE 2.5e-6, rms ΔF 2.0e-5,
ω ratio −7.0e-6, drift −2.6e-11 Ha/ps, AL 38 DFT calls, H₃⁺ tiers).

One audit at 41³ is a ≈ 0.13–0.27 s GPU solve on the second teacher; the model answers 6–17
steps meanwhile at speed 1 (paced). At speed 1000 the model answers ≈ 1.6e5 steps/s, so 2000
answered steps take ≈ 12 ms and the **auditor is saturated**: audits run back to back, the
effective cadence is one audit per teacher solve (≈ 2–4·10⁴ answered steps), and a failed audit
arrives that many steps after its dispatch (the panel's "audit lag"). `audit_every` governs only at
paced speeds. A `GpuContext` now costs 0.92–1.09 s instead of 2.2–2.6 s (Vulkan-only instance
first, `gpu/tests/context_timing.rs`).

#### Sampling-pass cost (`app` `diag_pass_rate`, RX 7900 XTX)

| scene | first (cold) evaluation | reference solves | perturbations | scans | hot burst |
|---|---|---|---|---|---|
| H₂ 41³ | 0.20 s | 0.14 s | 0.17 s | 0.16 s | 0.08 s |
| H₂ 49³ | 0.23 s | 0.20 s | 0.21 s | 0.19 s | 0.10 s |
| H₂O 41³ | 0.81 s | 13.0 s each (H; O at 1, 3, 5) | 1.28 s | — | — |
| H₂O 57³ | 1.11 s | 15.3 s each | 1.87 s | — | — |

The viewer's first pass (`Command::EnsureModel`) keeps to `PASS_TARGET_S` = 120 s at an expected
per-solve cost of 1.6 × the first evaluation until a pass is timed: H₂'s default 273-solve pass
starts as is (≈ 100 s estimated, 24–34 s actual at the old teacher tol 1e-5; 39–44 s at 1e-6 in
the S18 session below, where 1e-5 took 41.5 s too); H₂O at 57³ (300 solves, the user's 12-minute
pass) is offered at its trust minimum, 210 solves, with "Start anyway". The O-atom reference scan
(open-shell, cold, 200 SCF iterations at 41–57³: S10) is not in the estimate — the live ETA shows
it once the pass runs. The measured per-solve rate a later pass ("Sample more") is estimated with
excludes the reference stage (the references are cached by then; Task 9 fix round 1).

**Sample tolerance 1e-6 (ruling S18, final fix wave, 2026-10-14).** The estimated solver's own
teacher (pass, fallbacks, background auditor) now solves at tol 1e-6; the DFT backends keep
1e-5. Measured with the headless screenshot run (`--preset h2 --estimated --sample`, H₂ 49³, one
session, the two builds alternated, `SAMPLE_TOL` the only difference):

| teacher tol | full default pass (268–273 solves) | fit of record |
|---|---|---|
| 1e-6 | **38.9 / 39.4 / 44.4 s** (≈ 145–166 ms a solve) | 273 samples: rmse_E 1.40e-5 Ha/atom, **rmse_F 4.43e-5 Ha/Bohr**; 268: 1.39e-5 / 4.47e-5 |
| 1e-5 | 41.5 s (one more run at 89.7 s under a loaded GPU, discarded) | 268 samples: 1.39e-5 / 4.47e-5 — the same to three digits |

So on the GPU teacher the tighter tolerance costs no measurable pass time (the M3 finding that
the GPU wall time is flat in tol) and does not change the labels at the fit's resolution: the
GPU's warm answers already sit at its f32 path scatter at 1e-5 (`diag_interactive_tol`), so the
spec's "~2e-4 Ha/Bohr of SCF lag at 1e-5" is not what this teacher shows. The ruling stands on
its own grounds — the labels are permanent, the CPU teacher and heavier species are not
measured here, and the suite validates at 1e-6 — and costs nothing. The fit agrees with the
validate suite's 1e-6 pass (rmse_F 4.47e-5) and the earlier in-window 1e-5 fit (3.9e-5 at 350
samples). Measurement caveat (rule 7): the automatic first pass is sized from the first cold
evaluation × 1.6, and in this session that evaluation varied enough (on a GPU shared with other
applications) to scale the H₂ pass to 72, 88 or 115 solves on some runs; only full passes are
in the table.

### M6 Task 9: the automatic pass sized from a median; the far-drag regression at 49³ (`app` worker tests, 2026-10-14, release, RX 7900 XTX)

**Pass sizing.** The automatic first pass (`Command::EnsureModel`) now starts with the panel's
plan whole and is shrunk once, after its first `PASS_SIZING_SOLVES` = 3 solves, to
`PASS_TARGET_S` at their **median** wall time (`app::worker::pass_resize`,
`Trainable::rescale_pass` → `providers::sampling::Sampler::rescale`: only jobs not yet run are
dropped, the references are kept, never below `min_samples_for_trust`). `PASS_SOLVE_PER_COLD`
(1.6 × the cold first evaluation, above) is deleted: the offer rule — "Start anyway" when even
the trust floor exceeds the target — now uses the cold first evaluation as measured (1×). The
median ignores one slow solve among three (the cold reference, a solve slowed by GPU sharing —
the 72–115-solve spread above came from a single cold evaluation). Numbers (scripted teacher,
`the_first_pass_is_sized_from_the_median_of_its_first_three_solves`; the timing is the
worker's tick wall time, so a solve's measured cost includes the tick's overhead):

| first three pass solves | plan | jobs before → after | at 0.9 s a solve it would have been |
|---|---|---|---|
| 0.9, 0.1, 0.1 s | 3000 perturbations (3213 jobs) | 3213 → **1193–1195** over runs (bound 1000–1200) | 133 |
| 1.0, 1.0, 1.0 s | default (273 jobs) | 273 → **119** (bound 110–120) | — |

`Sampler::rescale` on H₂ (41³, `rescale_never_drops_references_or_goes_below_the_trust_minimum`):
to 3 solves → 60 jobs = trust floor 59 + 1 reference (exact; `scaled_to`'s ceilings trimmed);
after the reference and two perturbations, to ≤ 120 → 120 jobs (26 perturbations, 6 of 12 scan
points — 0.7 … 1.8 r₀ ends kept — 87 burst steps), then run to exactly 120. The exact trim to the
floor takes the burst's last steps, then the last queued perturbations, before any scan end
(review M4, `the_trim_to_the_trust_floor_keeps_both_scan_ends`: 60 perturbations + 60 scan
points, no burst → 29 + 30 = the floor 59, both scan ends solved; the test also caught the
floor branch ending one job over the floor before fix round 1). The re-sizing goes through the
driver (`Dynamics::rescale_sampling`), which publishes the new total at once: before fix round 1
a tick without a pass step (an edit's slow re-evaluation right after the re-sizing) showed the
old total again — `[3213, 1195, 3213, 1195]` in the worker test, now `[3213, 1193]`. Water's floor plan
is 210 solves (4 of them references): at 57³ the offer stands (210 × 1.11 s ≈ 233 s > 126 s;
the shown estimate is below the M5 one, 1.78 s a solve, and the measured 1.87 s perturbations —
the live ETA corrects it once the pass runs). The panel now says what the number is: "≈ N at
this scene's first solve (… s); a pass's own solves differ (references, warm starts) — the time
left is shown once it runs", and the second click's "Start anyway (≈ N at the first solve / at
the last pass's rate)". Not "≥ N": the cold solve is no lower bound (H₂'s pass solves run at
0.45–0.9× of it, water's perturbations at 1.6–1.7×, its O references at 12–14×).

**The thresholds moved with the estimate (fix round 1, review I1).** The offer (before the
automatic pass: the trust floor against 120 s × 1.05) and the panel's second click (the plan
against 120 s) compare at 1× the cold first solve where M5 used 1.6×. A scene whose trust floor
costs 79–126 s at the cold rate therefore now starts without asking, and a water-like scene then
runs ≈ 1.7× that plus its references (up to ≈ 3.5 min). Accepted, not restored: 1.6 was fitted
on water and wrong for H₂ by 3×, which is why the spec deleted it; the live ETA replaces the
estimate after two pass solves and Cancel keeps what was sampled. The user-facing rule is now
"a pass whose trust floor exceeds 2 min at the scene's first solve asks first" (CLAUDE.md); water
still asks at 41³ (210 × 0.81 s ≈ 170 s) and 57³ (≈ 233 s). H₂O's first three solves are its
H reference and two O multiplicities (13–15 s each): its median sizes the pass to the floor.

**Far-drag regression** (`after_a_far_drag_the_surrogate_answers_again`, `test_plan` pass, then
R = 8 → R = 3.3 and Play):

| grid (box) | R = 8 | fallback steps until the model answers at R = 3.3 | refits | samples after | bound |
|---|---|---|---|---|---|
| 41³ (10 Bohr), M5 final fix wave / re-measured now | Fallback, box-affected, not kept | **15** / **15** (deterministic, 2 runs) | 0 | 83 (the fallbacks box-affected, not kept) | ≤ 20 (M5) |
| **49³ (12 Bohr), M6 Task 9** | Fallback, box-affected (face density 6.3e-4 > 2e-5), not kept | **1** — the teleport's own evaluation (5 runs out of 5) | **0** after Play (the teleport round's refit comes first, below) | 84 (that sample kept) | **≤ 2** (measured × 2, spec §9); refits ≤ 2 unchanged |

At 41³ the pass's stretched scan end (1.8 × 1.6 = 2.9 Bohr) is box-affected and left out of the
fit (M5 final review I1), so R = 3.3 starts outside the trained envelope and its own answers
there are box-affected too; at 49³ the scan is clear of the box and R = 3.3 is inside the
envelope after one kept sample. The refit that brings the model back is the teleport round's:
paused, the worker refits a kept sample at once (`refit_if_pending` throttles only while
playing), before the test reads its refit count — so the count does not depend on wall time (the
play-phase throttle, 0.5 s, would matter only if step 1 fell back too, and a slower GPU would
then need fewer steps, not more; fix round 1, review M2, corrects the report's "slower GPU"
concern). The margin under the tight bound: step 1's worst atom force uncertainty is **0.015**
of its bound. This is a new measurement on a different grid, not a loosened or tightened
tolerance on the old one (rule 6). The 41³ recovery (15 steps) is now a recorded measurement
only: `a_far_geometry_in_a_small_box_falls_back_and_is_not_trained_on` still covers 41³, but
only R = 8 not being trained on there, not the recovery.

### Policy constants of the estimated solver (M5 Tasks 4–9; for the final review)

Behavioural numbers of `providers::surrogate` and the app worker — none per element, all in atomic
units. "Why" is the one-line justification; the module docs carry the derivations.

| constant | value | why |
|---|---|---|
| `FallbackPolicy::force_std_abs` / `force_std_rel` / `FORCE_STD_BOUND_MAX` | 2e-3 Ha/Bohr / 10 % / 1e-2 | the spec's force tier; relative for large forces, capped so a large force never buys an unbounded error |
| `energy_std_per_atom` | 1e-3 Ha | the spec's energy tier per atom; the trust rule's training-residual bound and the audit's verdict use the same number |
| `max_descriptor_distance` | 3.0 training spreads | the envelope (rule 5c): what the variance cannot see (all body-2 features vanish as a pair separates) |
| rule 0: the cell check under the surrogate (M6 Task 11) | `Scene::check_inside_cell`, 3h from every face | the teacher's own R12 check, same error and message, before any rule (draft mode too): a model answer is a function of the positions alone and carried a dissociated pair to \|x\| = 287 Bohr (diagnosis 2026-10-15 §3); the store, envelope and audits are of in-cell geometries |
| rule 5a: an isolated atom in a scene of ≥ 2 atoms (M6 Task 11) | no neighbour within r_cut = 6 Bohr → fallback | the model's free-atom limit is Σ E_ref with F = 0 (ruling S5), not the teacher's dissociated state at the scene's spin (restricted H₂ singlet +4.6e-2 Ha above 2 E_ref at r = 6.4, measured); a one-atom scene is exempt (its answer is E_ref, the teacher's own atom, S9). Not an envelope limit: on even with `max_descriptor_distance` = ∞ |
| `COORDINATION_RATIO` (rule 5b, M6 Task 11) | 3 | the vanishing direction judged relatively: an atom whose coordination C_t = Σ_{j∈t} f_c(r_ij) is > 3× below the training minimum of its species is outside. One kept stretched-pair sample (r = 5.125, f_c 0.05) put C = 0 at 0.51 spreads (old rule never fired; now: fires at C < 0.0172, r > 5.50 Bohr for one H neighbour, and at C = 0). Factor = the spread limit's "3× beyond what was seen", multiplicatively — no new tuning; where the old premise held nothing changes (the spread rule fires first iff u < 2m/9: H₂ pass m 0.531, u 0.078 = 0.15 m → spread fires at r > 3.8, ratio would at r > 4.34). Only the n = 0 channels (the n ≥ 1 and l ≥ 1 features vanish inside full neighbourhoods by symmetry). Derivation: `surrogate::envelope` module doc |
| rule 6: the force std compared (M6 Task 11) | epistemic only: force_std_epistemic/√3 | the posterior's force variance splits exactly, force_std² = 3σ_f² + Σ_a g̃ᵀΣg̃ (accumulated separately in `Model::predictive_std`; test 1e-14); σ_f is label noise a fallback cannot reduce — as rule 7 already excludes the energy floor. With σ_f in it, a fit with σ_f ≳ 2e-3 fell back on every pass through equilibrium (absolute bound) and answered near the turning points: the user's cycle |
| trust: force noise (M6 Task 11) | σ_f ≤ `force_std_abs` (2e-3 Ha/Bohr) | σ_f is the fit's homoscedastic force residual — label noise *and* basis misfit together, one number for every geometry. A constant carries no geometry-dependent information, so it cannot inform a per-step decision (rule 6 leaves it out) and is gated once, globally, by trust: a model whose force residual scatter exceeds the absolute tier is untrusted ("force noise σ_f …") and goes through the recovery (offenders ranked against the absolute tier, quarantine within the 10 % cap). With both gates the per-component expected error of an answer is √(s² + σ_f²) ≤ √(b² + σ_f²) ≤ √2·b at the 2e-3 floor (s the epistemic std ≤ b by rule 6). User's model 9.6e-3 (trusted before: the relative residual bound passes up to 10 % of \|F\|, capped at 1e-2); validated fits: H₂ pass 4.49e-5, H₃⁺ 2e-5 |
| teacher-surface fallbacks (M6 Task 11) | rule 5, the first evaluation after an edit, \|gap\|/N > `energy_std_per_atom` | the model's energy there is an extrapolation (+578 Ha in the user's run), a guess the history must not be anchored on (a 15 Ha offset carried for good), or off by more than its tier (rule-7 fallbacks 3.9e-2 Ha/atom in the repro): E_total is the teacher's on every such step. The switch is booked as J = E₁ − E₀ + ½(m̃₀ + m̃₁)·Δr, so E_conserved moves by Verlet's dt²Σ Δ\|F\|²/8m only (GPU repro: residual 1.6e-16 Ha); no model energy at an extrapolated point enters |
| rule 5d: new environment (M6 Task 11 round 2) | eigenvalue ≤ τ = Q′·ε·tr(G̃) is unspanned; a query row with Σ_null (u·v)² > N_null·τ falls back | controller ruling on the round-1 finding: the H₂-only model answered H₂ + H (third atom 2–5 Bohr) with forces 4–22× their bound under a confident posterior. The dimer store has no zero feature column (a single neighbour's body-3 features are functions of its distance) but leaves 17 of 29 standardised directions unspanned (\|λ\|/tr ≤ 3.1e-16, next 5.2e-13); along them the coefficient is the prior's 0 and the variance ‖φ_⊥‖²/λ looks small. τ: the Gram's accumulation and the Jacobi eigensolver's rounding, ≤ Q′ε·tr in Frobenius norm; N_null·τ: the most null energy any training row can carry. No tuned constant. Measured: dimers ν ≤ 3.3e-6, H₂ + H ν ≥ 5.6e8 (`surrogate` `tests/novelty.rs`) |
| rule 5e: extrapolation grade, `GRADE_MAX` (M6 Task 11 round 4) | γ_max = 2 | the per-atom (neighbourhood) D-optimality / MaxVol grade of Gubaev, Podryabinkin & Shapeev, J. Chem. Phys. 148, 241727 (2018) and the MLIP package (Novikov et al., Mach. Learn.: Sci. Technol. 2, 025002 (2021)); Podryabinkin & Shapeev, Comput. Mater. Sci. 140, 171 (2017) introduced it for whole configurations. Per atom because the model is a sum of per-atom terms: size-independent, and one extrapolating atom inside an interpolating sum is caught. Per species a MaxVol active set of the training atoms' standardised descriptors (rebuilt every refit; MaxVol tolerance 1 + 1e-3, so every training row has γ ≤ 1.001); a query atom's γ = max_j \|(xᵀA⁻¹)_j\|, ∞ outside the training atoms' per-atom span — the per-atom analogue of 5d, not the same test (5d checks the design rows, energy sums and force rows; a query's force rows can leave the design span while every atom stays inside the per-atom span), both kept; MaxVol not converged within 1000 swaps voids γ ≤ 1 + δ, so every query falls back until a refit converges. γ ≤ 1 is interpolation inside the active rows' parallelotope; γ_max = 2 is chosen, consistent with the MLIP practice of selecting at γ > 1–2 (a specific page/table could not be verified in this task: no web access): at most a factor 2 beyond what was learned along any active direction. As a bound on force errors the grade is a heuristic (the per-atom energies at the active rows are never observed, force rows are not graded), supported by the transition test and the repro. The ≤ 3× repro outcome comes from this rule, not from calibration: logged force z-scores up to 9.8 show the posterior ~10× overconfident on learned three-body geometries (follow-up: a variance inflation fitted from audit residuals). Measured: validated models answer at γ ≤ 1 (suite pinned values unchanged, active learning 38 DFT calls before and after); H₂ at 1.6 in its window γ 0.98, a dimer 0.2 Bohr past the 1.2–2.2 window γ 38; H₂ + H after k learned H₃ samples (pair Morse): 0/40 answered for k ≤ 10 (forces there 33–65× the bound at k ≤ 3), 26/40 at k = 20, 40/40 at k = 80, answered ≤ 0.03× the bound; GPU hot repro: 737 grade fallbacks, worst answered force 1.72× the bound (round 3: 20.6×). Cost: H₂ surrogate step 6.3 → 7.7 µs |
| rule 4b: environment class (M6 Task 11 round 6) | per atom its neighbour count within r_cut; per geometry the largest: 0 isolated (exempt), 1 pair, ≥ 2 many-body | controller ruling: universal, element-agnostic, no constant beyond r_cut (`providers::surrogate::env_class`). Isolated is exempt: rule 5a refuses it in a scene of ≥ 2 atoms, and a one-atom scene's answer is E_ref, the teacher's own atom (S9) |
| rule 4b: leave-one-out residuals (round 6) | r_s = (I − H_ss)⁻¹ e_s, H_ss = Φ_s A⁻¹ Φ_sᵀ W_s, per sample (its 1 + 3N rows at once) | exact at the fit's hyperparameters and standardisation (Woodbury; derivation in `surrogate::loo`): the block form of e/(1 − h); cost O(m Q′² + m³) per sample of m = 1 + 3N rows from the model's Cholesky factor. Verified against an explicit downdate of the model's own L·Lᵀ (a check of the Woodbury algebra and the rows, not of the row weights — those come from the same model on both sides), 40 noisy weighted dimers/trimers, κ(A) 1.1e4: **2.4e-12** relative (`loo_matches_a_downdate_of_the_fit`); the novelty test (rule 5d) builds its rows with the same function as the LOO (`loo::standardised_rows`, final review M7); rotation-covariant at fixed hyperparameters to 1.7e-12 (`loo_is_rotation_covariant`) |
| rule 4b: trust condition (round 6) | class trusted iff ≥ Q′ samples and, over its samples, LOO force ‖r_F,j‖/√3 ÷ `force_bound(\|F_j\|)` RMS ≤ 1 and max ≤ 1, LOO energy \|r_E\|/N ≤ `energy_std_per_atom` | the existing bounds (rule 6's force bound, the energy tier) and the trust rule's minimum count, nothing new. Pinned transition on the 40 held-out H₂ + H geometries: many-body untrusted at 0/10/20/28 learned H₃ samples (at 10: LOO F 9.0× max), trusted from 29 = Q′ (LOO F max 0.12×), 40/40 answered, worst 0.11× (`the_many_body_class_earns_trust_as_samples_arrive`). Validated models: H₂ pass pair n 272, LOO F rms 0.009× max 0.100×, E 0.152×; H₃⁺ pass many-body n 296, rms 0.008× max 0.094×, E 0.078× — both trusted, suite numbers unchanged. Cost of the max (rule 7): one sample alone in its region untrusts its class (the repro's tail sample, 14.9×; a first H–He sample untrusts H₂ with it, energy 1.1e3×) until the region is learned |
| variance cadence: coordination change (M6 Task 11 round 2) | fresh verdict whenever any atom's neighbour count within r_cut changes | a reused verdict answered a three-body geometry (round-1 hot repro: 1.32e-3 Ha/atom); a neighbour arriving is a new environment, not "four steps of motion" |
| audits judge forces (M6 Task 11 round 2) | fail if any atom's ‖ΔF‖/√3 > min(max(2e-3, 10 %\|F_teacher\|), 1e-2) — strict, 1× | the variance rule's own bound; an energy-only audit passed the round-1 repro's 4× force errors. A force failure takes the failed-audit path (teacher surface, the flip booked, untrusted "audit found forces off by …" until the next refit). Strict like the energy audit (review round 3, M1): a false alarm costs one untrust and a refit, never a wrong answer; for a calibrated posterior with per-component std s the false-alarm rate is P(χ²₃ > 3b²/s²) = 0.39 at s = b, 7e-3 at s = b/2, ≈ 0 where trained (s ≈ 1e-6–1e-5 ≪ b) |
| Gram accumulation (M6 Task 11 round 3) | Neumaier-compensated G, b, yᵀy (`Accumulator::sums`) | CLAUDE.md rule 4; the null-space threshold τ = Q′·ε·tr needs the Gram's rounding at the eigensolver's level. Max \|λ_null\|/tr (τ/tr = 6.44e-15), Morse dimer stores: naive 4.2e-16 / 1.0e-15 / 2.1e-15 at 300 / 3000 / 30 000 (margin 15× / 6× / 3×; at 30 000 the fit failed `Singular`), compensated 1.7e-16 / 1.8e-16 / 2.0e-16 (37× / 35× / 32×) — flat, so no √N_rows factor (`surrogate` `tests/novelty.rs`) |
| stuck, box variant (M6 Task 11) | the last `STUCK_STEPS` fallbacks all box-affected | "every answer here touches the box wall — enlarge the grid (or turn container walls on)": those answers are never kept, so nothing is learned — the user's 140 fallbacks under "trusted" |
| `MIN_RESIDUAL_DOF` | 30 | both noise levels known to ≈ 25 % before the variance means anything (Task 3 review) |
| `variance_cadence` | 4 | Task 9 timing table above: 1.7–3.6× steps/s; a verdict reused ≤ 3 steps (≈ 0.13 Bohr for hot H); rejections never reused |
| `audit_every` | 2000 answered steps | ruling S7 (catch model error the evidence absorbed as noise); 2000 since S14 — now background, so the cadence costs no speed. **Only at paced speeds:** at speed 1000 the auditor is saturated (2000 steps ≈ 12 ms, one solve 130–270 ms), the effective cadence is one audit per teacher solve (≈ 2–4·10⁴ steps) and a failed audit arrives that late, whatever this value — 200 would change nothing there; at paced speeds it would triple the second teacher's GPU share |
| `BOX_AFFECTED_FACE_DENSITY` × (h / 0.25 Bohr)² (`box_affected_face_density`) | 2e-5 e/Bohr³ at h = 0.25 | **a calibration at h = 0.25 for hydrogen, h-scaled (ruling S17).** The reported face density is the outermost grid plane, one h inside the Dirichlet wall, where the confined orbital is pinched: ρ_face ≈ 4κ²h² ρ₀(wall). With ΔE ≈ π d ρ₀(wall) for a nucleus d from the wall (1-D wall against e^{−κx}, patch area π d/κ), ΔE_conf ≈ π d ρ_face / (4κ²h²) ≈ 50–80 Ha per e/Bohr³ for H₂ (κ ≈ 0.85, d = 2.6–4 Bohr) at h = 0.25 — measured 40–55 (41³ and 49³, `diag_confinement_vs_face_density`). So the tolerated density scales as h² (the same density is 4× the confinement at h = 0.125, so a quarter of it is tolerated there; direction corrected in Task 9 fix round 2); the 1/κ² dependence is not modelled (handoff: an energy-relative rule, ΔE_conf,est vs f·N·`energy_std_per_atom` with κ from the HOMO eigenvalue). 2e-5 at h = 0.25 ≈ 1e-3 Ha of confinement, half the H₂ audit threshold (N × 1e-3): acceptable, not tight (the review prefers 1e-5 = the solvers' warning level); 49³ hot-burst solves that warn at 1.0–1.4e-5 stay trainable. (Fix round 4's 1e-3 ≈ 5e-2 Ha let confinement through: Task 9 item 1.) Beyond it the teacher's surface depends on position and on orientation vs the box axes, so no invariant model can represent it: honest, not over-conservative |
| `QUARANTINE_FRACTION` | 0.1 × `max_descriptor_distance` (0.3 spreads) | **a geometric choice, not a measurement**: the fence covers the unrepresentable geometry and its immediate neighbourhood, not the region the model still fits (Task 8 fix round 3) |
| `FENCE_VOTES` / `FENCE_VOTE_LIMIT` | 3 / 3 per fenced sample | Task 9 fix round 1 (I3): three mutually consistent fresh answers that disagree with the model inside a fence are evidence, not noise; one is kept and the next refit decides (kept if reproduced, else quarantined and the region closed to votes — one sample per unrepresentable region, not a chain); a choice, not a measurement |
| `QUARANTINE_ROUNDS` / `QUARANTINE_MAX_FRACTION` | 5 / 10 % of the store | a model that cannot fit its data broadly (the misspecification test) must stay untrusted; ten per cent is "a few strays", not "the data" |
| `RECOVERY_DEGREE_CAP` | 6 | the documented bent-molecule setting (Task 6: D = 6 within 10 % of D = 10 on H₃⁺), tried once before quarantine |
| `THIN_FRACTION` / `THIN_EVERY` | 0.25 × `max_descriptor_distance` (0.75 spreads) / 10 | a fallback answer within 0.75 spreads of a stored one teaches little; one in ten of those thinned candidates (the counter advances only on one, restarts at every kept sample) is still kept so a parked molecule's region keeps learning |
| `NO_MODEL_SAMPLE_FACTOR` | 4 × Q′ (116 for H at D = 4) | final review B2: with no model (a pass offer waiting, a refused autoload) the worker cannot refit and the envelope cannot thin, so DFT steps kept a sample each with no bound; 4 Q′ is past the trust floor Q′ + 30 for every species set (Q′ ≥ 29), beyond it one in `THIN_EVERY` with a "waiting for a sampling pass" note; a choice, not a measurement |
| app `SAMPLE_TOL` | SCF tol 1e-6 for the surrogate's own teacher (pass, fallbacks, background auditor); the DFT backends keep `INTERACTIVE_TOL` 1e-5 | ruling S18 (spec §4, review MUST C2(b)): labels are permanent and at 1e-5 the SCF lag would set the evidence's noise floor. The auditor is included because its answer is admitted as a training sample; the audit gap alone would tolerate 1e-5 (energy lag ≤ 3e-6 Ha, M3 `diag_interactive_tol`, against N × 1e-3). Part of the provenance: a model taught at 1e-5 is refused with both tolerances named (`sim` test `a_saved_model_reloads_…`). Cost: "Sampling-pass cost" above |
| `STUCK_STEPS` | 50 consultations | ≈ 12 fs of a trusted model falling back at one place before the panel says "not answering here" (Task 9) |
| app `REFIT_INTERVAL` / `REFIT_SAMPLES` | 0.5 s / 25 samples | a fallback burst refits a few times a second, not every step; a refit is 5–26 ms at S = 1, ~1 s at S = 2 |
| app `PASS_TARGET_S` / `PASS_SIZING_SOLVES` | 120 s / 3 | the user's 12-minute pass with no warning (Task 9 item 4); M6 Task 9: sized from the median of the pass's first three solves (one outlier moves nothing), replacing `PASS_SOLVE_PER_COLD` = 1.6; per-solve cost measured above |

### Box rule on the Accurate grid (M6 Task 11 requirement 6, a measurement; `surrogate` `box_rule_on_the_default_and_accurate_grids`, `--ignored`, 2026-10-15, RX 7900 XTX, GPU shared with another agent)

The box rule's threshold scales as (h/0.25)², so on the coming Accurate default (81³ × 0.15, the
same 12 Bohr cube as 49³ × 0.25) it is 7.2e-6 e/Bohr³. Question: does it exclude an ordinary
centred H₂? Default pass (plan seed 7) and 1000 surrogate steps at 300 K (CSVR τ 20 fs after 400
warm-up steps, every 10th geometry solved by a separate teacher), container walls off and on
(`Walls::max_half_width`):

| grid | rule e/Bohr³ | pass: box-affected / held | 300 K walls off: box-affected / probed, max face ρ (× rule) | 300 K walls on |
|---|---|---|---|---|
| 49³ × 0.25 | 2.0e-5 | 0 / 272 | 0 / 100, 1.52e-6 (0.08×) | 0 / 100, 1.53e-6 (0.08×) |
| 81³ × 0.15 | 7.2e-6 | 0 / 272 | 0 / 100, 3.80e-7 (0.05×) | 0 / 100, 3.45e-7 (0.05×) |

No: the rule excludes nothing of an ordinary centred H₂ on either grid, and the margin is wider
at 81³ (the reported plane is one h from the wall, so the reported density falls faster than h²
when h shrinks at a fixed box). Walls on/off is the same measurement here: the thermostat keeps
the centre of mass at rest (R10) and the molecule starts centred, so it never reaches the walls'
layer; the pass does not see the walls at all (its burst is the sampler's own MD). What the box
rule does exclude is the user's case — atoms within ≈ 3–4 Bohr of a face (an H₃ in the 12 Bohr
cell: face densities 2e-5 … 5e-4 in the repro below).

### `the_stuck_escape_repro_is_answered_honestly` (M6 Task 11, rounds 1–6, 2026-10-15, RX 7900 XTX shared with another agent)

The user's report (diagnosis `docs/superpowers/notes/2026-10-15-stuck-escape-diagnosis.md`) made
deterministic: the default H₂ pass at 49³ (272 samples, s_F 4.49e-5), the hot-H₂ tail sample the
user's model held (r = 5.125 Bohr on the body diagonal, face density 7.9e-6 — kept; coordination
minimum 0.531 → 0.0516), then two runs on that model through `Dynamics` with the worker's refit
cadence, each CSVR 4000 K for 400 fs, one H added 3 Bohr from the centre, 1000 steps: NVE, then
with the thermostat left on (the run sees the samples the first one kept). Deterministic per
adapter; the asserted parts in the test doc.

| quantity | round 1 (rules 0, 5a–5c, teacher surface) | round 2 (+ 5d, coordination cadence, force audits) |
|---|---|---|
| NVE: model answers / judged / box-affected | 72 / 6 / 12 | 58 / 7 / 8 |
| NVE: worst judged \|ΔE\|/N; force residual / bound | 3.3e-4; **4.34** | 5.8e-4; 1.89 |
| hot: worst judged \|ΔE\|/N; force residual / bound | **1.32e-3; 22** (exploratory run) | 2.3e-4; 1.64 (325 answers, 3 judged) — round 3 (compensated Gram): **7.52e-3; 20.6** (220 answers, z up to 33) — round 4 (+ rule 5e): 3.8e-5; 1.72 (87 answers) |
| NVE fallbacks by rule | 5c 498, 5a 46, 5b 50, 6 52, 7 48 | 5c 638, 5d 304 |
| step at a surface change vs Verlet's dt²Σ Δ\|F\|²/8m | residual ≤ 1.6e-16 Ha | ≤ 2.4e-16 Ha (bound 1e-9) |
| outcome | NVE paused at the cell margin, step 766 | no pause in either run |

**Round 3 → 4: the hot run.** In round 3 the NVE run's kept H₃ fallbacks spanned the directions
rule 5d guards, and the hot run's model answered three-body geometries with a confident posterior
and errors up to 20.6× the force bound and 7.5e-3 Ha/atom (round 2's pass was its trajectory).
Round 4's extrapolation grade (rule 5e, γ_max = 2) fixes it: NVE run 40 answers, 5 judged, worst
1.9e-4 Ha/atom and 0.29× the force bound (z ≤ 4.1), fallbacks 5c 580, 5d 120, 5e 260; hot run 87
answers, 1 judged, 3.8e-5 Ha/atom and 1.72× (z 9.8), fallbacks 5e 737, 5a 102, 5b 42, 5c 32;
box-affected checks worst 7.2e-3 Ha/atom (was 8.7e-2). Both runs asserted in
`the_stuck_escape_repro_is_answered_honestly`.

**Round 5: the hot run made checkable — and the gap it shows.** The review asked for ≥ 10 judged
answers in the hot run (round 4 judged 1 of 87). With the container walls at 3 Bohr (keeping the
H₃ compact, off the faces) and every answer checked: 747 answers, 196 judged, 551 box-affected;
**12 judged answers beyond the bounds — force 3.1–6.2× the bound, energy up to 2.6e-3 Ha/atom —
at grades γ 0.86–1.16**, i.e. inside or at the edge of the active set's parallelotope (γ ≤ 1: 137
answers, worst 4.9×). No γ_max removes them; the force z-scores reach 24. It is model error in
the interior of a sparsely learned three-body region (a few dozen H₃ samples against 29
directions; the training residuals pass), which neither the grade (a bound on extrapolation),
the posterior (overconfident) nor S8 (residuals at the training points only) sees; the force
audits would untrust such a model, but between audits it answers. Not fixed: the controller's
ruling is needed (more data per direction before trust, a calibrated posterior, or a richer
basis for H₃). The γ data are logged by the repro (`judged answers by grade`).

**Round 6: class-wise leave-one-out trust (rule 4b) — the hot run green.** Same runs, the strict
hot run asserted (walls 3 Bohr, every answer judged, ≥ 10 judged). The tail sample alone in its
region makes the pair class untrusted (LOO force max 14.9×, energy 9.5× the tier), so all 1600
heating steps of each run fall back (they did not before). **NVE**: 90 answers, 5 judged, worst
1.71e-4 Ha/atom and 0.52× the force bound (γ 0.89–1.38, z ≤ 4.7); 18 box-affected; end classes
pair n 442 untrusted (LOO E 20.6×), many-body n 50 trusted (LOO F rms 0.24×, max 0.53×).
**Hot**: 140 answers, **58 judged, worst 5.19e-4 Ha/atom and 2.69× the force bound** (γ ∈
[0.62, 1.98]; γ ≤ 1: 46 answers, worst 0.89×), 82 box-affected (worst 2.5e-3 Ha/atom); fallbacks
4b 626, 5e 184, 5c 30, 5b 20; end classes pair n 607 trusted (max 0.21×), many-body n 151
trusted (rms 0.33×, max 0.98×). The force z-scores still reach 10.8 — the posterior is not
calibrated; the class rule, not the variance, is what keeps the answers within 3×. Booking
residual ≤ 2.1e-16 Ha. Teacher solves for the whole test 5398, GPU 651 s (every heating step
now falls back while the tail sample is unexplained; round 5's total was not recorded). Active learning (`active_learning_run_converges`)
unchanged: 36 fallbacks / 38 DFT calls (its store is pair-only, LOO max 0.007×).

**Known limitation: the max over a class (rule 7).** Rule 4b judges each class by its *maximum*
LOO error, so one poorly learned sample (the stretched-H₂ tail above) leaves the whole class
untrusted and estimated mode falls back to DFT more often until the region is learned. This is the
deliberately cautious choice for M6. A calibrated replacement (jackknife+ / conformal intervals
on the LOO residuals) was built and measured but parked: with γ_max 2 it answered unprobed
three-body geometries at up to 8× the force bound, and with γ_max 1 it was too cautious for H₃⁺
and active learning. Both patches are kept for after the M6 pause.

The force assertion is ≤ 3× the bound (`REPRO_FORCE_RATIO`): rule 6 bounds the epistemic *std*,
so a calibrated posterior's residual exceeds its bound at 1σ routinely, and beyond 3× with
P(χ²₃/3 > 9) = 6e-6 per atom. The residual 1–2× cases are H₃ geometries the model has learned
from the first run's kept fallbacks (samples 273 → 315): one sample spans its direction, so rule 5d
lets the posterior judge there, as designed. Box-affected checks are not judged (the box rule);
the hot run's worst there is 4.6e-2 Ha/atom of confinement and model error together.

### Runtime (in-suite, five tests concurrent, GPU evaluations serialised; run of record 2026-10-14)

| test | system | grid | tol | teacher solves | GPU s | ms/solve | SCF steps/solve | readbacks/solve | test wall s | result |
|---|---|---|---|---|---|---|---|---|---|---|
| `active_learning_run_converges` | H₂ | 49³ | 1e-6 | 38 | 3.0 | 79 | 6.1 | 244 | 70.2 | 36 fallbacks/300 (2 % in the last 100), last gap +5.1e-6, 38 DFT calls |
| `surrogate_knows_what_it_does_not_know` | H₂/H₃ | 49³ | 1e-6 | 283 | 31.3 | 110 | 8.1 | 360 | 129.2 | far Fallback (envelope), ion Fallback, walk + refit ΔE(4.0) +5.2e-5; H₃ Fallback (envelope), then untrusted by the S8 rule |
| `surrogate_nve_is_tight` | H₂ | 49³ | 1e-6 | 273 | 29.2 | 107 | 7.9 | 346 | 131.5 | drift −2.6e-11 Ha/ps, swing ratio 1.032, rewind 1.5e-14 Bohr, 9.1e4 steps/s |
| `surrogate_reproduces_the_h2_frequency` | H₂ | 49³ | 1e-6 | 310 | 31.6 | 102 | 7.8 | 338 | 133.0 | rms ΔE 2.5e-6, rms ΔF 2.0e-5; ω_sur/ω_MD − 1 = −7.0e-6; 1.0e5 steps/s |
| `h3plus_tier` | H₃⁺ | 49³ | 1e-6 | 975 | 69.9 | 72 | 6.7 | 257 | 165.0 | rms ΔE 3.6e-6 (floor 3.1e-7), rms ΔF 8.9e-6; ω ratios −3.0e-4 / −4.5e-4 |
| **suite** | | | | 1879 | **165** | 88 | | | **171 (2 m 51 s)** | 5 passed |

The suite's GPU time is the sum of the "GPU s" column; the test walls include waiting (the
four sampling passes alternate on the GPU, so each pass takes 105–131 s in-suite against ≈ 30 s
of its own GPU time). Budget ≤ 10 min solo: 2 min 51 s in-suite (the passes dominate; the
surrogate's own 40 000 + 4000 + 10 000 + 640 steps cost ≈ 1 s in all).

## Viewer extras (M5b, branch `m5b-viewer-extras`, 2026-10-14, release; `sim` and `app` unit tests)

Not a `validate` binary: the walls are tested in `crates/sim/src/dynamics/tests.rs` on analytic
providers (no grid), the cloud builder in `crates/app/src/cloud.rs`. Derivations: `sim::dynamics`
module doc, "Container walls"; `app::cloud` module doc. Atomic units; nothing per element.

### Constants

| constant | value | why |
|---|---|---|
| `sim::dynamics::WALL_LAYER` (D) | 1 Bohr | the soft layer sits inside the slider's face L, so the stopping distance is not limited to the h/2 between the face and the providers' 3h margin: a wall in that h/2 alone would need k = 2·1 Ha/(h/2)² = 128 Ha/Bohr² to stop 1 Ha at h = 0.25, ω dt = 2.73 for ¹H at 0.25 fs — Verlet-unstable (> 2) |
| `WALL_ONSET` (a) | 0.5 Bohr | the curvature ramps u'' = k·min(d/a, 1) (C²): a C¹ wall (½ k d², u'' jumping 0 → k) has no Verlet shadow Hamiltonian across the jump — measured, K random-walked ≈ 1 % per bounce, ≈ 20 % after ~300 bounces of a free ¹H at 0.05 Ha |
| `WALL_K` (k) | 3.5 Ha/Bohr² | chosen for u(D) = k (D²/2 − aD/2 + a²/6) = 1.021 Ha: a nucleus with ≤ 1 Ha per axis turns before the face (δ(1 Ha) = a/2 + √(2/k − a²/12) = 0.992 Bohr); ω = √(k/m) = 0.0437 a.u.⁻¹ for ¹H, ω dt = 0.451 at 0.25 fs (0.902 at 0.5 fs), well inside Verlet's 2. The margin guard (R12) fires only above u(D + h/2) = 1.376 Ha (h = 0.25) |
| `Walls::max_half_width` | `Cell::max_coord()` − h/2 = (n−1)h/2 − 3h − h/2 | the h/2 is the room for Verlet's overshoot past the exact turning point (measured below) |
| app `ui::MIN_WALL_HALF_WIDTH` | 1.5 Bohr | the panel's smallest box: below it the 1 Bohr layer is most of the box (a choice, not a measurement; the driver accepts any L > 0 — the layer start is clamped at 0, a harmonic trap) |
| app `cloud::BLEND_DISTANCE` | 0.5 Bohr | display only: the atomic superposition's weight against the last real density is max displacement / 0.5 Bohr (a bond-length-scale move makes a solved cloud visibly wrong); a choice |
| app `cloud::TAIL_FRACTION` | 1e-6 of the profile's peak | display only: the superposition's reach; what the cut tail carries is < 1e-4 e for the atoms here |

### Test bounds and the measurements behind them (re-measured for fix round 1; one number each)

| test | asserted bound | derivation | measured |
|---|---|---|---|
| `a_free_nucleus_bounces_off_the_wall_with_e_conserved_flat` — turning depth | ∈ [0.95, 1.02] × δ(E) | δ from u(δ) = E (0.350 Bohr at 0.05 Ha, cubic branch); the upper 2 % is the shadow-orbit overshoot, 1/√(1 − (ω dt)²/4) − 1 = 2.6 % at most for the harmonic branch and less on the softer cubic one | 0.3509 / 0.3500 = 1.0026 |
| same — E_conserved during contact | ≤ 1.05 × (ω dt)²/4 × E | Verlet's swing for a harmonic mode of energy E, H − H̃ ≤ (ω dt)²/4 · E with the stiffest curvature k (the cubic onset is softer) | 5.36e-4 Ha vs bound 2.67e-3 |
| same — energy after each of 306 bounces | \|K/E − 1\| < 2e-3 | the C² wall's error is bounded, not a random walk: the bound is ~2× the measured maximum, far below the C¹ wall's 0.2 | **1.05e-3** (E = 0.05 Ha, 20 000 steps) |
| same — speed after the bounces | \|Δ\|v\|\| < 1e-3 \|v\| | half the energy bound | 4.6e-4 |
| `shrinking_the_box_while_playing_books_positive_work` | \|K − W\| ≤ (ω dt)²/4 · W and max \|ΔE_c\| ≤ (ω dt)²/4 · W; each move booked to 1e-15 | a moving wall, like a moving drag target, books ΔU exactly for H while Verlet conserves H̃: residual ≈ (ω dt)²/4 of the work | W 8.58e-4, K 8.60e-4, ΔE_c 4.38e-6 vs bound 4.36e-5 |
| `the_wall_stops_a_one_hartree_nucleus_inside_the_margin` | 1 Ha per axis (corner): never past `max_coord`, no error over 400 steps, dt 0.25 and 0.5 fs; 1.5 Ha per axis: the guard fires | u(D) = 1.021 Ha; guard at u(D + h/2) = 1.376 Ha | deepest 3.1231 / 3.1389 Bohr (face 3.125, margin 3.25) at dt 0.25 / 0.5 fs: overshoot past the exact turn 0.006 / 0.022 Bohr |
| `superposition_integrates_to_the_electrons_and_peaks_at_the_nuclei` | \|Σρh³ / (2 × profile electrons) − 1\| < 5e-3; \|Σρh³ − 2\| < 2e-2 | grid quadrature of the cusped 1s at h = 0.2 (0.4 % for one atom) | 2.009 vs 2 × 0.99975 |
| `draft_mode_books_the_jump_of_an_untrusted_refit` (fix round 1, review I1) | jump booked (\|jump\| ≥ 1e-4 Ha) and \|ΔE_c\| ≤ 1e-12 Ha across the refresh | the refresh re-evaluates at unchanged positions, so booking E_new(r) − E_old(r) is exact up to rounding (the trusted-refit precedent: 1e-12) | jump −9.98e-3 Ha, ΔE_c 0 (without the fix: −9.98e-3, unbooked) |

## Tools (M6, branch `m6-tools`; Task 3: bond detection, 2026-10-14, release; `sim` unit tests on analytic densities, one CPU diagnostic)

The detector (`sim::bonds`, spec §3) decides each pair within R_max by three element-agnostic
conditions on the bias-corrected density ρ̂ and the promolecule ρ_pro of the solved isolated-atom
references (`sim::references`). Every constant is universal (rule 1); the references are solved
by the scene's own solver and shell-binned from its grid, never tabulated.

### Constants and their derivation

| constant | value | derivation |
|---|---|---|
| M = `SEGMENT_SAMPLES` | 33 chord samples, s = k/32 | ≈ h/2 spacing on a 4 Bohr chord; the endpoints are the nuclei |
| ε_⊥ = `EPS_TRANSVERSE` | 0.07 | ≥ 3 × the transverse-ratio noise: measured spread 0.018 here (3× = 0.054); the re-review measured 0.022 (3× = 0.066); **Task 7, the real surface** (H₂ at R_e, 12 placements): 0.0079 (3× = 0.024) — it stands |
| δ = `DELTA` | **0.11** (spec: 0.10, provisional) | ≥ 3 × the relative spread of the corrected q: measured 0.0629 on a mean of 1.852 (3.4 %, 3× = 0.102 > 0.10), so the constant was **raised** under the spec's own rule (a constant is raised, never lowered, when a measurement demands). The re-review's run gave 0.051/1.82 = 2.8 % → 0.084. **Task 7, the real surface** (H₂ at R_e, 12 placements): spread 0.0208 on 1.697 (1.2 %; 3× = 0.062 absolute, 0.037 relative) — δ = 0.11 stands; the fixture stays the binding measurement |
| ρ_floor = `RHO_FLOOR` | 1e-2 e/Bohr³ | below Bader's closed-shell range; two H 5 Bohr apart: ρ_mid ≈ 4e-3 (the fixture: 2e-6), two at 2.5 Bohr: ≈ 0.05 |
| R_max = `R_MAX` | 8 Bohr | beyond it no two atoms in range share density above the floor; pairs beyond are reported `TooFar` and never sampled |
| dr (shell bins) | h/2 | one bin per node shell near the nucleus (no node between r = 0 and h), several shells per bin beyond; the geometric mean of ρ per bin — exact for an exponential tail under log-linear interpolation, where an arithmetic mean would sit high by Var_b(r)·2κ² (0.26 % H, 0.74 % He with 2κ = 2Z_eff = 3.375 — a quarter of the trilinear bias; review M1 corrected the first draft's 3 %) |
| march step | h/2, re-steered from ±h/2 central differences on the three axes (7 samples per step) | a half-cell step resolves a ring point to ≈ 0.1 Bohr; the "within h of a nucleus" stop is one cell from the hill's top |
| f_g = `MARCH_STATIONARY_FRACTION` | 0.25 | a quadratic top reached from 0.5 Bohr away: the gradient falls below a quarter of its first-step value within ≈ 0.12 Bohr of the point (measured on the H₃⁺-like fixture: the march stops 0.11 Bohr from the centroid after 4 half-steps) |
| step cap | 2R_max/h = 64 steps at h = 0.25 | twice the longest chord |
| condition 4, attribution (Task 7) | **no constant**: at p* no third nucleus's promolecular term may exceed the weaker of the pair's own two | a bond point of A–B lies where A's and B's basins meet (Bader 1990); the promolecule's stockholder shares (Hirshfeld, Theor. Chim. Acta 44, 129 (1977)) say whose space a point is. Added from the real water surface (below); margins there 30–45× both ways |
| transverse basis e₁ | toward the centroid of the other nuclei, ⊥ u (lab construction when degenerate) | the direction a neighbour's hill or a ring point lies in; a lab-fixed basis measures a directional rise anywhere between 1 and cos 45° of its value depending on how the molecule is turned, which would make `multi_centre` orientation-dependent |

### The fixture noise behind δ and ε_⊥ (`the_same_operator_cancels_the_interpolation_bias`)

The bonded-pair fixture (two 0.3·exp(−r²/0.5) atoms 1.6 Bohr apart plus a 0.15·exp(−r²/0.5)
bump at the midpoint; exact q = 1.8992) evaluated on a 49³ h = 0.25 grid and sampled trilinearly
(the Kohn-Sham providers' operator), the promolecule from shell-binned tables of the same gridded
atom, over 24 placements (shifts 0, h/4, h/2, 3h/4 along x × 6 orientations: x, y, z, (1,1,0),
(1,0,1), (1,1,1)):

| quantity | spread (max − min) | mean | the re-review's run | ceiling asserted |
|---|---|---|---|---|
| corrected q = ρ/ρ_pro^tri at **fixed s = 0.5** (the same-operator residual itself) | **0.0700** | 1.8931 (0.3 % below the exact 1.8992) | — | ≤ the derivation's residual between a node and a cell centre (h²/8)·Σ_a\|D_aa\|·q, D_aa = ∂²_aρ/ρ − ∂²_aρ_pro/ρ_pro on the lab axes (D_u = −4.848, D_⊥ = 0 exactly for equal widths) = **0.0719** for this fixture; mean < exact (review I1, M3). **Split in Task 7** (the Task 3 re-review's item (a)): 0.0700 = 0.0664 (u ∥ x, node → cell centre: the residual) + 0.0037 (chords along y, z, where the x-shift is across the chord and the residual vanishes identically — every term has the same x-profile — so only the shell table's own error moves q); asserted separately (Task 7 review M2): with ε the table's error through the same operator (its promolecule over the exact atoms', blended at the same nodes; ε ∈ [−7.4e-3, −3.8e-3]) and Q = q·(1 + ε) the pure same-operator ratio — u ∥ x: q spread 0.0664 and Q spread 0.0674 ≤ 0.0719; y/z: Q = the exact q to 4.4e-16 at every shift (the residual vanishes identically; the 0.0037 is all ε); all 24: q spread 0.0700 ≤ spread(Q)/(1 + ε_min) + Q_max·Δ(1/(1 + ε)) = 0.0748 |
| the detector's q(p*) (p* = the sampled chord minimum) | **0.0629** | 1.8521 (2.48 % below exact — the safe direction: residual at the chosen points 0.034 + wander 0.014) | 0.051, −2.6 % | measure-and-record (δ follows from it); derived compound ceiling (the reviewer's) = the fixed-point residual 0.072 + the wander of the sampled minimum: along the chord q − 1 = 0.8992·(1 − 5.12x²), and the minimum search wanders ≤ 2 samples of 0.05 Bohr (observed s = 0.4375 … 0.5312), so up to x = 0.1 Bohr where q is lower by ½\|q″\|x² = 0.044 → **0.116**; \|mean − exact\| ≤ 0.06 and mean < exact |
| raw ρ_tri(p*)/ρ_pro^exact(p*) | 0.1078 | 1.8763 | 0.142 | ≥ 0.10 (what the correction removes) |
| transverse ratio (ρ̂ at p* ± h e_k over ρ̂(p*)) | 0.0179 | 0.8848 | 0.022 | ≤ 0.03 |

What the per-placement table shows (printed by the test): at fixed s = 0.5 the residual alone
spreads the values by 0.070 over the 24 placements (u ∥ x alone: 1.9074 / 1.8624 / 1.8411 /
1.8624 across the four x-shifts, 0.066; 0.036 for (1,1,0), 0.020 for (1,1,1)) — the u ∥ x part
inside the leading-order 0.0719, the fourth-order terms at h/w = 0.5 trimming it, and the last
0.0037 the table's own (1.9111 at the y/z chords' cell centres against 1.9074 at the node: the
0.0719 does not cover it, which is why Task 7 split the assertion, above); the detector's p* is the
**sampled minimum, which
wanders ±1–2 chord samples** (s = 0.4375 … 0.5312) along the fixture's flat bottom to the sample
with the largest negative residual, where q itself is lower by up to 0.044 — which is why no
placement reaches 1.907 and the detector's spread (0.063) sits below the fixed-point range. A
real H₂ chord is as flat (Task 2: 0.1853 at the midpoint, 0.1861 one sample away), so the
wander is expected on the real surface too; Task 7 measures it there. The brief's 0.06 ceiling
was the re-review's single run plus headroom; the first draft of this section asserted the
fixed-point residual against the detector statistic (review I1) — now each statistic has its
own derived ceiling, above. The measurement is robust (the reviewer's 120-placement scan gives
the same 0.0629), so δ = 0.11 stands.

### The shell-binned table (`shell_binning_a_gaussian_atom_reproduces_its_profile_log_linearly`)

A Gaussian atom 0.3·exp(−r²/2·0.8²) on 49³ h 0.25 through `solve_atom`: relative error of the
table vs the profile 0 (r = 0), 1.6e-3 (0.3), 7.5e-4 (0.7), **4.4e-3** (1.4), 2.9e-3 (2.9), and
5.0e-3 at worst over a sweep of r ∈ [0.3, 5] (471 points); electrons 2.4142 vs 2.4191 (−0.2 %;
the first draft's trapezoid over the flat first segment gave +0.5 %, review M5 — Simpson with
the r² interpolant's midpoint now). The asserted bound is the scheme's own: log-linear
interpolation between entries Δ apart errs by ≤ Δ²/(8w²), with Δ the spacing of consecutive bin
*means* — up to Δ_max = 0.143 = 1.14 dr beyond r = 1, read off the table (review M2: the first
draft assumed Δ ≤ dr) — plus the bin's residual Var_b(r)/(2w²) ≤ dr²/(8w²) at worst:
(Δ_max² + dr²)/(8w²) = **7.0e-3**. The brief's 2e-3 does not follow from log-linear
interpolation at dr = h/2 (the interpolation term alone is 3.7e-3 at r = 1.4); the bound is
the derived one, not a loosened pin. The first segment (r = 0 to the first shell at r̄ ≈ 0.32) is
interpolated linearly in r² (the smeared density is even at the nucleus): linear in r would
miss by ≈ 0.6 % there. Beyond the last entry (half-extent − 3h) the last segment's log-slope is
continued instead of a step to 0 (review M4): a chord point beyond one atom's table keeps that
atom's exponential tail, and a promolecule of 0 occurs only outside the grid.

### The analytic fixtures (expected verdicts from an independent 501-point scan inside each test)

| fixture | scan | detector |
|---|---|---|
| bonded pair (`a_bonded_pair_passes_all_three_conditions`) | Route S, min at s = 0.5, q = 1.8992, transverse 0.887, Bonded | same; s 0.5, q within 2 % |
| depleted pair, −0.42·g(mid) (`a_depleted_overlap_is_not_a_bond`) | Route S, ρ̂(p*) = 0.0408 > floor, q = 0.245 → NotAccumulated | same |
| water-like (`a_third_atom_…`) | O–H: chord monotone (ends 6.52 / 0.39) → Route D, q max 1.33 at s = 0.59, transverse on q 0.98, Bonded; H–H: min 0.278 at s = 0.5, rise 1.68 toward O, the march runs into O's hill → NotATransverseMaximum (stop `Nucleus`) | bonds {(0,1), (0,2)} on Route D; the H–H march ends 0.235 Bohr from O (the seventh half-step point from the chord midpoint, six positions evaluated; the scan's 0.02 Bohr march ends at 0.250) |
| H₃⁺-like ring, bump 0.25 (`a_three_centre_ring_is_kept_by_the_march`) | edge min 0.316 at s = 0.5, ρ(centroid)/ρ(mid) = 1.148, rise 1.111 toward the centroid, q = 2.44; the march stalls on the ring point (stop `Stationary`) → Bonded { multi_centre: true } | three such bonds; each march ends 0.11 Bohr from the centroid, 1.16 Bohr from the nearest nucleus, after 4 half-steps; components [[0,1,2]] |
| ring, bump 0.10 / 0.15 | rise 1.056 (inside ε_⊥: plain saddle, `multi_centre: false`) / 1.080 (**above** ε_⊥ = 0.07: the march runs, `multi_centre: true`) | identical. The brief named 0.15 as the inside-ε run ("+8 %"); 8 % > 7 %, so 0.10 is the inside-ε run and 0.15 is checked against the scan |
| rotations (`the_march_is_rotation_invariant`) | water about its H–H axis by 45°, 90°, and a seeded random rotation (StdRng 7); the ring tilted 45° about x and the same random rotation | every verdict, route and stop identical; rho_min, enhancement, s equal to 1e-9; march ends equal under the rotation to 0.05 Bohr — measured \|Δ\| water 0.0000 (45°), 0.0000 (90°), 0.0064 (random); ring 0.0057, 0.0057, 0.0000 (tilted), 0.0233, 0.0179, 0.0056 (random) |
| grazed water (`a_neighbours_bond_seen_from_the_side_is_not_the_pairs`, Task 7: the real surface's shape — O valence 0.94·e^{−r²/1.08} filling the H–H chord, H 0.12·e^{−r²/0.7}, O–H lobes 0.09·e^{−r²/0.15} at 0.88 of each O–H) | O–H: Route D, q 1.461 at s 0.918 (0.15 Bohr from H), Bonded; H–H: Route D, q 1.439 at s 0.956 (0.13 Bohr from an H), transverse 0.969, ρ̂ 0.258 — conditions 1–3 pass — and condition 4 from the analytic terms (O 0.062, near H 0.117, far H 3e-6) → `NeighbourRegion { atom: 0 }` | identical verdicts; q within 2 %; bonds {(0,1),(0,2)}; turned 45° about H–H and by the seeded random rotation: identical (s up to the chord's mirror tie) |

### H₂ on the real surface — which route (CPU diagnostic `diag_h2_bond_route_on_the_cpu`, 33³, σ = h = 0.25, tol 1e-5; Task 2's flag)

The H reference through `solve_atom` on `CpuKs`: E = −0.414255 Ha (confinement-raised, as M5
measured), 26 shell entries to r = 3.25, 0.963 electrons in the table (the rest beyond the
33³ margin); ρ̄(0) = 0.1264, ρ̄(0.8) = 0.0544, ρ̄(1.6) = 0.0140.

| R | raw chord | corrected chord ρ̂ | route | q(p*) | transverse | verdict |
|---|---|---|---|---|---|---|
| 1.6 (the app's start) | endpoints 0.1782, max 0.1956 at s = 6/32, midpoint 0.1853 — no interior minimum below both ends (Task 2) | endpoints **0.1807**, interior min 0.1843 at s = 15/32: still not below both ends | **Route D**: q's interior maximum 1.703 at s = 0.5 (ends 1.287) | 1.703 (ρ_pro 0.1088) | 0.982 (on q) | **Bonded** |
| 1.789 = R_e(σ) | endpoints 0.1597, max 0.1720, midpoint 0.1544 | endpoints 0.1634, interior min **0.1536** at s = 17/32 (below both) | **Route S** | 1.640 (ρ_pro 0.0937) | 0.931 (on ρ̂) | **Bonded** |

So the correction lifts the smeared endpoints (+1.4 % at 1.6, +2.3 % at 1.789) but at the
compressed geometry the dip between the two density maxima at x ≈ ±0.5 (inside the nuclei at
±0.8: the smeared molecule's density peaks are not at the nuclei) stays above the endpoints, and
H₂ takes Route D there; at its equilibrium it takes Route S. Both are Bonded with q ≈ 1.6–1.7
(exact H₂ at 1.4: 1.72). Task 7 pins the route per geometry on the GPU; the detector reports it.
**On the GPU (Task 7, 49³, tol 1e-6):** at R_e = 1.78894 Route S at all 12 placements, q 1.682–1.703.
The route decision at 1.6 rests on a 2 % margin (0.1843 vs 0.1807) against ≈ 1 % wiggles of the
corrected chord (0.1898 0.1894 0.1870 0.1852 0.1844 0.1843 0.1853 …), so along H₂'s vibration
(1.6 → 1.79 → …) the route may flip between D and S; both give Bonded with q 1.64–1.70 and, for a
symmetric pair, the same p*. Task 7 should therefore pin the **verdict and q** at 1.6, not the
route (Task 3 review, check 4).

### Where the references are solved (Task 3 review I2, fix round 1)

The first draft solved each species' scan on the **stepping** provider inside `tick` (resetting
it before and after, on paused ticks too, and retrying after an abort): every edit that queued a
species cost the warm start (`moving_a_nucleus_warm_starts`: warm readbacks = cold, 477 = 477)
and a reference starting the moment a pass stopped held Cancel for 18.5 s
(`pause_and_cancel_abort_a_sampling_solve_promptly`). Now they are side jobs (`sim::dynamics`
module doc, "Bonds and references"): the GPU surrogate's background auditor solves them in its
job slot; otherwise `ForceProvider::side_solver` (a fresh `GpuKs`/`CpuKs`) on its own thread. No
job starts while a pass runs; a failure is not retried; Pause does not abort them (nothing waits
for them). A pending reference is background work (fix round 2): its progress is
`DynSnapshot::bond_reference` ("bond references: O 2/3…" in the panel), it never sets the busy
state, `--screenshot` does not wait for it, and an idle worker polls it once a budget,
publishing only on progress or landing. The worker test
`a_reference_job_neither_resets_the_warm_start_nor_delays_a_command` (H₂O 57³, RX 7900 XTX) pins
it: Pause while playing with the references solving → paused snapshot in 42–125 ms (bound 1.5
s); the graph after the references (55–105 s, the O scan below); a 0.02 Bohr teleport after them
warm 380–381 readbacks vs cold 752 (bound 0.85 × cold); estimated: Cancel of a pass with the
references queued 19–20 ms, a Pause with O in the auditor's slot 78 ms.
`moving_a_nucleus_warm_starts` (H₂ 49³): warm 381 vs cold 477 again (the first draft: 477 = 477).

**A pass ended below Q′ keeps its samples and does not refit (fix round 2).** Because the
driver's references seed the pass, a Cancel early in a pass lands in its geometry stage with a
few samples; the end-of-pass refit plus its trust recovery then took 45–50 s for 3 H₂O samples
at 97³ (one fit 6.2 s, the recovery's retries 43 s), on the worker thread inside the Cancel.
Below Q′ (the basis's independent columns; the trust rule needs n ≥ Q′) the model cannot earn
trust anyway, so the samples are kept, the refit skipped, and the note says "N samples kept —
not enough to fit (Q′ = M); sample more to train". No constant changed. Measured: H₂ 41³, 3
samples (Q′ = 29) → Cancel < 1 ms; H₂O 97³ re-pinned `pause_and_cancel_…` (references in after
88 s of background work, 3 samples, Q′ = 176) → Pause 8 ms, Cancel 150 ms (bounds 1.5 s).

**The O reference is slow and keeps the singlet (measured, GPU, tol 1e-5, a fresh `GpuKs`):**
57³ — singlet 2.4 s, triplet and quintet stop unconverged at the 200-iteration cap (Δρ 1.4e-4,
1.8e-3) after 19 s and 49 s; 97³ — 1.9 s, then 40 s and 87 s unconverged (Δρ 2.9e-4, 3.5e-4).
So the scan takes 70 s / 130 s and keeps multiplicity 1 for O, against the LSDA (and Hund) triplet
ground state: the open-shell p⁴ atom's SCF does not settle (degenerate p occupations). Out
of M6 scope — solver work for the pseudopotential milestone, whose Hund oracle is one of its
gates (noted in its spec §6.3); it also governs every sampling pass's O reference since M5. On the real H₂O surface at 57³ with that singlet table the detector reports
O–H Bonded on Route D (q 1.466 at s = 0.875, transverse 0.977) and **also H–H Bonded** on Route
D (q 1.426 at s = 0.0625, next to an H end, transverse 1.024): a spurious H–H bond that Task 7's
real-surface oracles must resolve (the singlet promolecule is a suspect). **Resolved in Task 7**
(below, "Water and condition 4"): the H–H point is the O–H bond's own deformation lobe seen from
the side, and the detector gained condition 4 (attribution); the singlet table is not the cause
(scaling it by 0.9 / 1.1 leaves the pair bonded at q 1.49 / 1.37).

**Open plumbing (review M7).** `AtomReferences::seed_energy` is used only by the surrogate's
sidecar load into its own cache; the driver's cache is never seeded from a loaded model, so a
session with a loaded model solves each species' table once more (off the stepping path, so it
costs no warm state). Tasks 7/10 must not assume the sidecar seeds the detector.

## Tools (M6 Task 7): the real surface (`tests/tools.rs`, 2026-10-14, release, RX 7900 XTX)

Protocol: `GpuKs` at σ = h = 0.25 Bohr, SCF tol 1e-6, 49³ unless stated (57³ water, the `h2o`
preset's grid; 65³ for the H₂ dissociation scan and LiH), behind the suite's GPU lock **with side
solvers**, so the driver solves its isolated-atom references off the stepping path as in the app.
Bond verdicts are read from the `Dynamics` snapshot's `PairReport`s; the chords behind them are
printed every run. Every number below is from the run of record and is deterministic per adapter
(bit-reproducible GPU); the module doc of `tests/tools.rs` carries the same derivations.

### Constants of the suite and of the tools they test (spec §11's list)

| constant | value | derivation |
|---|---|---|
| `NOISE_Q` (tools) | 0.034 (relative) | the larger of the gridded fixture's relative q noise (Task 3: 0.0629/1.8521 = 0.0340, 24 placements) and the real surface's (H₂ at R_e, 12 placements: 0.0208/1.6971 = 0.0123); the H₂ test asserts the live value ≤ it, so a noisier surface fails there first |
| `H2_ENHANCEMENT_MIN` | 1.3 | spec §11 (exact H₂ at 1.4 Bohr: 1.72) |
| `D2_BOUND` | 3e-4 (spec 1e-3) | the residual budget of the Verlet-corrected ratio is ≈ 1e-4 (h²-anharmonic cross term ≈ 3e-5, crossing interpolation ≈ 1e-5, warm-path SCF scatter ≈ 1e-5; the mass scales out of the classical equation exactly); 3× the budget, 6.5× the measured 4.6e-5; the raw ratio's +4.8e-4 Verlet systematic is outside it |
| `K_SOFT` | 0.05 Ha/Bohr² | spec §11, a stated lower bound for H₃⁺'s modes; **measured** the smallest vibrational Hessian eigenvalue, 0.1435 (asserted ≥) |
| `E_CONSERVED_BOUND` | 1e-10 Ha | spec §11 / Task 6 (P8: every FIRE step's ΔE_KS + ΔK booked) |
| `VERIFIED_F_BOUND` | 3e-3 Ha/Bohr | spec §11 (the M5 force tier) |
| `SURROGATE_R_BOUND` | 2·F_TOL/k = 1e-2 Bohr | spec §11, k = 0.200 (M4) |
| F_TOL (`sim::relax`) | 1e-3 Ha/Bohr on max_i \|F_i\|, the per-atom vector norm of the augmented force | the brief; well above the egg-box ripple (ΣF ≈ 3e-4 at h = 0.25, 2.2e-6 rms per component at 49³), so reachable; the norm (not the largest component) so that \|F\|₂² ≤ N·F_TOL², which the energy comparison needs — `common::relax` uses the same definition since this task |
| FIRE N_min, f_inc, f_dec, α_start, f_α | 5, 1.1, 0.5, 0.1, 0.99 | Bitzek, Koskinen, Gähler, Moseler & Gumbsch, PRL 97, 170201 (2006), shown there to be system-independent |
| FIRE Δt start, Δt_max | the MD dt (0.25 fs); min(10 dt, 1.0 fs) | ruling P3: Verlet is stable for ωΔt < 2 and a compressed H–X stretch reaches ω ≈ 2ω_H₂(σ = 0.25) = 2.95e-2 a.u.: Bitzek's 10 × 0.25 fs = 2.5 fs gives ωΔt = 3.05, 1.0 fs = 41.3 a.u. gives 1.22. **Not covered: the container walls** (on by default in the app; final review M1): k_w = 3.5 Ha/Bohr² gives ω = 0.0436 a.u. for ¹H (ωΔt_max = 1.80), and a bonded H pressed into a wall along its bond ω = 0.0527 (ωΔt_max = 2.18 > 2, Verlet-unstable). FIRE's P ≤ 0 rule recovers by halving Δt — uphill steps, not a wrong minimum; a wall cap ≈ 1.2/√(k_w/m_min) ≈ 0.66 fs is not applied (frequency not measured) |
| FIRE Δt_min | 0.02 Δt_max | after Guénolé et al., Comput. Mater. Sci. 175, 109069 (2020), scaled to Δt_max: as recalled (not checked against the paper) FIRE 2.0 sets Δt_min = 0.02 Δt_MD — 0.005 fs here against this 0.02 fs; the larger floor only lets the stall detector speak sooner. A power that flips sign every step near the egg-box floor cannot shrink Δt indefinitely before the stall detector speaks |
| FIRE backtrack | r ← ½(r_before + r_verlet), v = 0 on P ≤ 0 | Guénolé 2020's uphill correction (spec MUST-7) |
| FIRE stall / cap | no 1 % improvement of max\|F_aug\| in 200 steps / 2000 steps | spec §6.2: a soft mode on the egg-box ripple is reported instead of spun on |

### The real-surface noise beside the fixture noise (`h2_is_bonded_at_equilibrium_and_not_at_six_bohr`)

| quantity | gridded fixture (Task 3, 24 placements) | H₂ at R_e = 1.78894 on the real surface (12 placements: centre shifted 0, h/4, h/2, 3h/4 along x × axis x, xy, xyz) | bound (asserted on the real surface) |
|---|---|---|---|
| q(p*) spread | 0.0629 on 1.8521 (3.4 %) | **0.0208** on 1.6971 (**1.2 %**; the sampled minimum wanders s = 0.469 … 0.5625; the outlier is the xyz chord through a node, q 1.682) | ≤ δ/3 = 0.0367 (1.8× margin); relative ≤ `NOISE_Q` |
| transverse-ratio spread | 0.0179 | **0.0079** (mean 0.932) | ≤ ε_⊥/3 = 0.0233 (3.0× margin) |

**δ verdict: δ = 0.11 stands** (3 × the real-surface spread = 0.062 absolute, 3.7 % relative —
both below δ); the binding measurement remains the fixture's 3.4 %. **ε_⊥ = 0.07 stands** (3 ×
0.0079 = 0.024). Neither is lowered: the fixture's larger noise is a real regime (a strongly
deformed pair), and a constant is only ever raised.

### The bond oracles

| test | measured (run of record) | asserted | margin / derivation |
|---|---|---|---|
| H₂ at R_e (49³) | `bonds_pending` [1] → [] after the H reference (0.2 s on the side solver); one bond, Route S, s 0.469, ρ̂(p*) 0.1486, ρ_pro 0.0872, q **1.703**, transverse 0.931 | one bond, Route S, not multi-centre, q ≥ 1.3 | q − 1.3 = 0.40 = 19× the q spread |
| H₂ at 6.0 Bohr (65³) | `BelowFloor`: Route S, ρ̂(p*) 0.0053 < 0.01, q 2.215 | no bond; the pair sampled | ρ̂ is half the floor |
| H₂ dissociation scan (65³) | q = 1.708 / 1.598 / 1.637 / 1.959 at R = 1.8 / 2.5 / 3.5 / 5.0, all `Bonded` (Route S; ρ̂ 0.147 / 0.077 / 0.034 / 0.0109) | reported (the spec's "monotone fall, crossing of 1 + δ" is not met — findings below) | — |
| He₂ at 3.0 (49³) | Route S, ρ̂(p*) **0.0570** > floor, ρ_pro 0.0587, q **0.9705**, transverse 0.946 → `NotAccumulated`; 12 placements q 0.9705 … 0.9723 (spread 0.0017, 0.18 %) | `NotAccumulated` with Route S and ρ̂ > floor at all 12; (1 + δ)/q_max − 1 ≥ 3·`NOISE_Q`; q_max ≤ 1 − 3·(He₂'s own relative spread) | verdict: 14.2 % vs 10.2 %; depletion: 0.9723 ≤ 0.9946 (2.8 % below 1 = 15× its spread) and ≤ 0.98 (review M4: 4.5× its spread above the run of record). The brief's q ≤ 1 − 3·`NOISE_Q` = 0.898 is **not met** (findings) |
| H₃⁺ ring (49³) | relaxed side 2.1174 (quench, 6 steps from 1.9); ρ̂(centroid)/ρ̂(edge midpoint) = 0.1302/0.1135 = **1.147**; edges q 1.511 / 1.511 / 1.503, transverse **1.099 / 1.099 / 1.100** → march, stop `Stationary` 0.112 / 0.112 / 0.187 Bohr from the centroid, 1.158 / 1.158 / 1.038 from the nearest nucleus → three `Bonded { multi_centre: true }` | three bonds, Route S each, `multi_centre` == (rise > 1 + ε_⊥), march end ≥ h from every nucleus, components [[0,1,2]] | rise − 1.07 = 0.029 = 3.7× the real transverse spread; march ends ≥ 4h from the nuclei. The reviewer's LCAO estimate (+5–7 %) was low: the smeared ring's centroid sits 15 % above the edge midpoint |
| HeH⁺ at 1.46, +1 (49³) | **`Bonded`**, Route D, s 0.656 (0.96 from He, 0.50 from H), ρ̂ 0.271, ρ_pro 0.193, q **1.405**, transverse 0.967 | reported; the report exists, filled, not `TooFar` | spec §3.6 expected `NotAccumulated` (findings) |
| H₂O, preset geometry (57³) and turned 45° about H–H / a seeded random rotation, each shifted off the grid | bonds exactly {(0,1),(0,2)}, Route D: q 1.466, 1.466 (preset); 1.452, 1.446 (45°); 1.464, 1.443 (random) at s 0.81–0.875 (0.23–0.34 Bohr from H), transverse 0.977–0.991; H–H **`NeighbourRegion { atom: 0 }`**: Route D, s 0.0625 / 0.094 / 0.125, q 1.426 / 1.435 / 1.435, transverse 1.024 / 1.004 / 0.972 | bonds == {(0,1),(0,2)} and the H–H verdict ∈ {`NoInteriorMinimum`, `NotATransverseMaximum`, `NotAccumulated`, `NeighbourRegion { atom: 0 }`} at all three | at the H–H point O's promolecular term is 32× the far H's (0.0643 vs 0.0020; 0.0758 vs 0.0024; 0.0890 vs 0.0028); at the O–H points O outweighs the other H 40–45× |
| LiH (65³, centred) | R_e(σ) = **3.366** by force bisection on [2.6, 3.6] (experiment 3.015); **`Bonded`**, Route S, s 0.469 (1.58 Bohr from Li), ρ̂(p*) **0.0423**, ρ_pro 0.0244, q **1.738**, transverse 0.961 | reported; not `TooFar` | literature ρ_b ≈ 0.04 (Bader 1990) — matched; spec §3.6 left the ionic verdict open (findings) |

The references these used (side solver, GPU, tol 1e-6): H at 49³ −0.424670 (doublet, 0.9978
electrons in the table), at 65³ −0.424773 (0.9986); He at 49³ −2.138685 (singlet; its triplet solve
warned "box too small", 2.67e-5 — the kept singlet did not); O at 57³ −36.578709 (**singlet**: the
triplet and quintet stop at the 200-iteration cap, Task 3; 7.98 electrons; ≈ 57 s of GPU); Li at
65³ −4.805425 (doublet, 2.96 electrons in the table).

### Water and condition 4 (Review Focus 1: "no H–H line, ever")

**What happened.** With conditions 1–3 alone the real water surface had an H–H bond (Task 3 fix
round 2, reproduced here): the H–H chord has no interior minimum — ρ̂ rises from 0.223 at each H to
0.316 at the midpoint, oxygen's valence — so it takes Route D, and q's interior maximum lies at
s = 0.0625, **0.179 Bohr from an H**: q 1.426 against 1.356 at the H itself, ρ̂ 0.252, transverse
ratio on q 1.024 < 1 + ε_⊥. Every condition passed.

**Root cause.** That maximum is the O–H bond's own deformation lobe, seen from the side. The O–H
chord finds its q maximum at s = 0.875, **0.226 Bohr from the same H** (q 1.466); a map of q in the
molecular plane around the H (0.1 Bohr grid, `diag` run) shows one lobe, peak q ≈ 1.46 at ≈ 0.22
Bohr from H toward O. The two chords leave the H 38° apart; the H–H chord passes 0.139 Bohr from the
O–H point (the two points are 0.14 Bohr apart in space, the H–H point 0.11 Bohr off the O–H axis) —
less than one grid spacing — and the H–H chord's q maximum is the lobe's flank
(q 1.426 on the chord, 1.461 0.1 Bohr off it toward the O–H axis). The ±h transverse probes see
+2.4 %, inside ε_⊥ = 7 %, as they must for a test that has to tolerate 3 % noise. Both points lie
inside the H's smeared nucleus (σ = 0.25): at h = 0.25 the O–H bond point itself has been pulled
onto the hydrogen (spec §3.6: the topology right, ρ_b not physical).

**What was ruled out.** (i) **The singlet O reference:** scaling the O table by 0.9 / 1.1 leaves
the H–H pair `Bonded` (q 1.488 at s 0.125 / 1.375 at s 0.0625) — a 10 % promolecule error moves q
by 4 %, while the pair clears 1 + δ by 28 %; any O reference would give the same verdict. (ii) **An
exclusion radius about each nucleus tied to σ and h** (the brief's first suggestion): the legitimate
O–H point (0.226 Bohr from H) and the spurious H–H point (0.179) differ by 0.05 Bohr in their
distance from H (0.14 Bohr apart in space), both inside one smearing width; any radius between them would be fitted to water, and any radius above 0.23
removes the O–H bond — whose q maximum is genuinely there on this surface (in the 3-D map, not only
on the chord). (iii) Tuning δ or ε_⊥: never (rule 6) — and pointless, the H–H q is 1.43.

**The fix: condition 4, attribution** (`sim::bonds`; spec §3.1, ruling P11). At p* the pair's own
atoms must be the two largest terms of the promolecule; a third nucleus whose term exceeds the
weaker partner's gives `NeighbourRegion { atom }`. Derivation: a bond point of A–B lies on the
interatomic surface where A's and B's basins meet (Bader 1990); the promolecule's stockholder shares
w_C = ρ̄_C/ρ_pro (Hirshfeld, Theor. Chim. Acta 44, 129 (1977)) are the parameter-free estimate of
whose space a point is, and where a third atom outweighs the weaker partner the point is in that
atom's space — a feature there is a bond of its own seen from the side. No constant (rule 1); the
weights come from the solved tables at the exact positions, so they carry no grid noise and are
rotation-invariant. **Margins on the real surface:** at the H–H point the terms are O 0.0643, near
H 0.1107, far H 0.0020 (O 32× the partner; 32× at both other placements); at the O–H points O 0.080–
0.107, the other H 0.0020–0.0024 (the pair wins by 40–45×). Checked last, only for a pair that passed
1–3, so the Task 3 verdicts are unchanged (the water-like fixture's H–H pair is still rejected by its
march into O, the ring keeps its edges, rotation invariance holds). It also removes the end-to-end
chord of a linear A–B–C (its interior minimum is the A–B bond point, where B, not C, is the
second-largest term). Pinned by `sim::bonds::tests::a_neighbours_bond_seen_from_the_side_is_not_the_pairs`
(a fixture of the real surface's shape — O valence filling the H–H chord, the O–H lobes 0.22 Bohr
from each H — whose H–H chord passes conditions 1–3 (Route D, q 1.44, transverse 0.97) and is
rejected by the scan's independent condition 4 from the analytic atom terms; turned 45° about H–H
and by a seeded random rotation: identical verdicts) and by the water oracle above.

### Findings (rule 7) — reported, not pinned

- **H₂'s dissociation stays on the spin-symmetric SCF solution — an artefact of the start, not of
  the detector, the physics, or LSDA's minimum** (Task 7 review I1). q falls from 1.708 (R = 1.8) to
  1.598 (2.5) and rises again to 1.637 (3.5), 1.959 (5.0), 2.215 (6.0). The solver is
  spin-unrestricted LSDA, but `multiplicity: None` gives n↑ = n↓ (`qm/src/scf.rs` `System`) and the
  cold start sets ρ↑ = ρ↓ = guess/2 (`qm/src/scf.rs` cold path; `qm-gpu/src/scf.rs` likewise), so
  nothing breaks the spin symmetry and the SCF converges to the spin-symmetric stationary point: one
  doubly occupied σ_g ∝ φ_A + φ_B, 4φ²/(1 + S) at the midpoint against the promolecule's 2φ², q →
  2/(1 + S) → 2 as S → 0 (and above 2 here: the unpolarised orbital is more diffuse than the fully
  polarised reference atom's). Past the Coulson–Fischer point (≈ 3 Bohr for LSDA) unrestricted LSDA
  has a **lower** broken-symmetry solution (ρ↑ on one atom, ρ↓ on the other) whose density tends to
  the promolecule, q → 1: at 5 Bohr ρ̂ ≈ ρ_pro ≈ 0.0056, below the floor — no bond (exact
  Heitler–London gives q = (1 + S)/(1 + S²) ≈ 1.09 at S ≈ 0.097). The same artefact leaves the
  stretched pair +4.6e-2 Ha above ΣE_ref, so the stretched-H₂ energies and forces are wrong too. The
  spec's prediction (monotone fall, a crossing of 1 + δ) assumed the right solution; on this solver
  the pair leaves the bond list only at the **floor** between 5 and 6 Bohr (ρ̂ 0.0109 → 0.0053), and
  at 5 Bohr a bond line is drawn. The detector is unchanged (it reports the density it is given);
  the README states the caveat, and the solver follow-up (for an even-N lowest multiplicity also try
  a spin-asymmetric start and keep the lower energy) is recorded in the M6 notes.
- **He₂'s depletion is small on this surface:** q = 0.971 at 3.0 Bohr (−2.9 %; the exact-atom
  promolecule estimate of the midpoint, 0.039, is 0.0587 for the smeared LSDA atoms). The verdict is
  robust (14 % below 1 + δ), the depletion resolved against He₂'s own noise (15×), but not against
  the gridded fixture's 3.4 %, which the brief's pin used; that noise is the residual of a strongly
  deformed pair, an order larger than He₂'s (measured 0.18 %). Recorded as a deviation.
- **HeH⁺ is `Bonded`** (q 1.405, Route D), against the spec's expected `NotAccumulated` (§3.6: q
  0.6–0.9 from unsmeared estimates of the neutral promolecule). At σ = 0.25 the ion's density 0.96
  Bohr from He is 0.27, the neutral promolecule's 0.19. HeH⁺ is a bound ion (D_e ≈ 2 eV), so the
  verdict is chemically defensible; QTAIM classes the bond closed-shell-like. Not asserted (§3.6).
- **LiH is `Bonded`** (Route S, q 1.74) with ρ_b = 0.0423 — the literature's ≈ 0.04 — at a smeared
  R_e of 3.366 Bohr (experiment 3.015). The ionic case was the open question; the accumulation test
  passes on this surface. Not asserted (§3.6). The brief placed Li at the origin; on the 65³ grid
  that put the H 3.3 Bohr toward a face and every solve warned "box too small" (1.1–1.7e-5), so the
  molecule is centred (no warning; R_e moved from 3.324 to 3.366 Bohr — the off-centre value was
  box-affected).
- **FIRE does not beat the quench on H₃⁺:** 25 evaluations against the 0.5 fs quench's 16 (1.56×)
  and the 0.25 fs quench's 29 (0.86×). The spec's "≤ 0.5 × the quench's" (and §6.1's "a few hundred
  steps where the quench needs thousands") describes many-mode, ill-conditioned relaxations; a
  three-atom triangle kicked 0.2 Bohr has no soft mode (spec §6.1 and §11 rewritten to these measured ratios, review M5), and FIRE spends its first N_min-limited
  steps growing Δt from the MD dt. Reported, not pinned (a ratio pin would pin Bitzek's start-up
  transient); the energy comparison — the spec's real statement — holds with a 17× margin.

### Isotopes and relax

| test | measured | asserted | margin / derivation |
|---|---|---|---|
| `d2_vibrates_slower_than_h2_by_the_mass_ratio` (49³) | H₂: 9 upward crossings, w = 1.474855e-2 a.u. (3236.9 cm⁻¹; = M4's ω_MD to 7 digits), ω = 1.473428e-2; D₂: 7 crossings, w = 1.042824e-2 (2288.7 cm⁻¹), ω = 1.042319e-2; ω_H/ω_D = **1.4136050** vs √(m_D/m_H) = 1.4136703 → **−4.6e-5**; raw w_H/w_D 1.4142897 → +4.4e-4 (Verlet's (h_H² − h_D²)/24 predicts +4.8e-4); nuclear-mass BO 1.41386 (+1.8e-4 from ours), experiment 1.41268 (−6.5e-4) | \|ratio/expected − 1\| ≤ 3e-4; no box warning; ≥ 5 crossings each | 6.5× the measured; budget above |
| `fire_relaxes_h3_plus_to_the_quenched_minimum` (49³) | start: side 2.117, each nucleus kicked 0.2 Bohr (seed 7) → sides 1.835/2.140/2.066; quench 14 steps (16 evaluations), cold E −1.123994020; FIRE 24 steps (25 evaluations), Δt → 0.432 fs, cold E −1.123992261, cold max\|F\| 9.2e-4; **\|E_f − E_q\| = 1.76e-6**; Hessian eigenvalues (Ha/Bohr²) six in [−3e-4, 8e-4], then **0.1435, 0.1476, 0.2392** (3-point breathing curvature 0.2393); sides differ by ≤ 0.0055 Bohr (slack 2·F_TOL/k_min = 0.014); max \|ΔE_conserved\| = **0** over 24 snapshots; W = E_KS(end) − E_KS(start) = −4.584121125e-3 | converged; cold max\|F\| < F_TOL; \|E_f − E_q\| ≤ 3·F_TOL²/(2·K_SOFT) = 3e-5; k_min ≥ K_SOFT; \|ΔE_conserved\| ≤ 1e-10; sim_time, step = 0 | 17× on the energy (the bound at the measured k_min, 1.0e-5, would still be 6×); k_min 2.9× K_SOFT. Review M3: the bound covers the vibrational modes; the translations carry the egg-box ripple, amplitude ≈ \|ΣF\|·h/(2π) ≈ 1.2e-5 Ha at ΣF ≈ 3e-4 — at 49³ the measured egg-box floor of this H₃⁺ is 3.1e-7 Ha rms (M5 `h3plus_tier`), so the spec's 3e-5 stands without it |
| `fire_on_the_surrogate_is_verified_by_the_teacher` (49³) | the surrogate suite's H₂ pass: 272 samples, trusted; Relax from R = 1.4: 21 FIRE steps, model max\|F\| 1.6e-4, **verified max\|F\| 1.5e-4**; R = 1.78818 vs the DFT R_e 1.78894: **ΔR = −7.6e-4**; one teacher solve during the relax | converged; verified ≤ 3e-3; \|ΔR\| ≤ 1e-2 | 20× and 13× |

### Runtime

Nine tests concurrent, GPU evaluations serialised. **Run of record: 9.7 min wall (581 s in the
harness)** — with two viewer windows (`app.exe`) open on the same GPU from another checkout, so the
per-solve times ran 4–5× the quiet machine's (H₂ at 49³: 249 ms per solve in the D₂ test against 54
ms in an earlier run of the same binary on a quiet GPU, whose nine tests — without He₂'s placements
and FIRE's Hessian — took 3.0 min). Per test (`last-run-tools.md`): H₂ bonds 18 stepping solves
(the 12 placements and the 65³ scan, cold), He₂ 12, water 3 at 57³ plus the O and H references on
the side solver (O ≈ 57 s of GPU when quiet, 88 s shared), LiH 12 bisection solves at 65³ plus one,
D₂ 819 (two 400-step runs and the surface), FIRE 26 stepping + 16 quench + 29 quench at 0.25 fs + 22
cold checks and Hessian solves, the surrogate relax 274 (the pass's 272 and one verification). The
numbers above are identical between the shared and the quiet runs (deterministic per adapter).

### Citations (spec §14)

Bader, *Atoms in Molecules: A Quantum Theory* (Oxford 1990) — the bond critical point, the
interatomic surface, LiH's ρ_b; Bader & Essén, J. Chem. Phys. 80, 1943 (1984) — ρ_b ranges;
Hirshfeld, Theor. Chim. Acta 44, 129 (1977) — the stockholder partition behind condition 4; Bitzek,
Koskinen, Gähler, Moseler & Gumbsch, PRL 97, 170201 (2006) — FIRE; Guénolé et al., Comput. Mater.
Sci. 175, 109069 (2020) — the half-step backtrack and Δt_min; Wang, Huang, Kondev, Audi & Naimi,
Chinese Phys. C 45, 030003 (2021) — AME2020 atomic masses (the spare electron mass of a cation, 1.8e-4
relative for H₃⁺, is in "Approximation gaps" at the top); NIST CCCBDB (Huber & Herzberg 1979) —
ω_e(H₂) = 4401.21, ω_e(D₂) = 3115.50 cm⁻¹; the M4 σ = 0.25 surface (R_e 1.789, k 0.200, ω 3237 cm⁻¹,
H₃⁺ side 2.117).

### The bond verdicts at the app's Accurate grid (h = σ = 0.15; first a one-off at the second merge of `main`, 2026-10-14; a gate since the M6 final fix wave, 2026-10-15)

The app's default grid became Accurate (`app::quality`), while the tools suite pins h = 0.25. The
`bonds_at_the_accurate_grid` test (`tests/tools.rs`; GpuKs tol 1e-6, side-solver
references at the same grid) re-asks the qualitative verdicts there with the detector's constants
unchanged (δ, ε_⊥, the floor — not tuned). Run of record (RX 7900 XTX, release, 48 s wall):

| system | grid | verdict | q (h = 0.15) | h = 0.25 pin |
|---|---|---|---|---|
| H₂ at R_e(σ = 0.15) = 1.6008 | 81³ × 0.15 | `Bonded`, Route S, s 0.469, ρ̂ 0.195, transverse 0.968 | 1.7317 | 1.703 at R_e 1.789 |
| He₂ at 3.0 | 81³ × 0.15 | `NotAccumulated`, Route S, ρ̂ 0.049 | 0.9609 | 0.9705 |
| H₂O O–H (both), preset geometry | 95³ × 0.15 | `Bonded`, Route D, s 0.8125, ρ̂ 0.342 | 1.5381 | 1.44–1.47 (57³) |
| H₂O H–H | 95³ × 0.15 | **`NotATransverseMaximum`** (Route S, transverse 1.231 > 1 + ε_⊥, the march ends on a nucleus) | 1.3293 | `NeighbourRegion { atom: 0 }` |

All four verdicts hold (two O–H bonds and no H–H, H₂ bonded, He₂ not). One difference is recorded
rather than smoothed over (rule 7): at h = 0.15 water's H–H chord has a Route S saddle whose
transverse rise (1.231) exceeds 1 + ε_⊥, and the multi-centre march climbs to within h of a
nucleus (end (i)) — condition 1 rejects the pair (`NotATransverseMaximum`: the saddle sits on the
flank of the O–H density, not on a ridge of its own), before condition 4's attribution, which
rejected it at 0.25 (Route D there), is reached. Its q 1.33 is above 1 + δ, so at this grid the
"no H–H bond" verdict rests on condition 1. He₂'s contact is more depleted (q 0.961) and H₂'s
enhancement slightly higher at the finer grid; water's O–H points move toward the H (s 0.81).

**Un-ignored at the M6 final fix wave** (final review core I2: the app default is what users see,
and δ, ε_⊥ come from h = 0.25 noise only). Re-run 2026-10-15 (RX 7900 XTX shared with another
agent, 58.5 s test wall): every verdict and number in the table above reproduced to the printed
digits. Condition 4 is now measured at this grid too — the second line of defence behind the H–H
verdict, asserted: at the H–H p* (s 0.25, 0.715 Bohr from the nearest nucleus) the promolecular
terms are **O 0.1430, H₁ 0.0653, H₂ 0.0048** — oxygen outweighs the weaker H by a factor ≈ 30, so
condition 4 would reject the pair (`NeighbourRegion { atom: 0 }`) if condition 1 did not. At each
O–H p* (0.339 Bohr from its H): O 0.0963, its H 0.1237, the other H 0.0021 — the pair's own atoms
are the two largest terms by a factor ≈ 46 over the third (no `NeighbourRegion`). The O–H ∇²ρ_b
−2.807 sits at 1.03 × its band (the bond-strength section's intrinsic water flicker).

## Bond strength (user request 2026-10-15, branch `m6-bond-strength`; `sim` unit tests and `tests/tools.rs`, release, RX 7900 XTX shared with another agent)

The detector's verdict is binary; each `Bonded` and `NotAccumulated` pair now also carries ∇²ρ at
its bond point and the QTAIM kind by its sign — shared shell (∇²ρ_b < 0, covalent) or closed
shell (∇²ρ_b > 0: weak, stretched, ionic-like) — Bader & Essén, J. Chem. Phys. 80, 1943 (1984);
Bader, *Atoms in Molecules: A Quantum Theory* (Oxford 1990), ch. 7. Derivation, operator choice and
the noise model: `sim::bonds` module doc, "Bond strength". No tuned constant: the classification is
a sign; the only number is the noise band, 3 × (an upper estimate made at every bond point;
`LAPLACIAN_SAFETY` = 3) — ≈ 2× more conservative than δ's rule of 3 × a measured spread (below).

### The operator (`sim` unit tests)

| test | what | measured |
|---|---|---|
| `the_laplacian_stencil_mirrors_the_solvers` | `LAPLACIAN_COEFF` = `qm::stencil::COEFF`; the one-node sum through samples = `qm::stencil::laplacian` (interior and Dirichlet faces) | ≤ 1e-12 relative; Σ\|c\| = 19.5048 |
| `the_laplacian_at_a_point_converges_at_second_order` | Gaussian w = 0.5; exact samples, stencil on p, h = 0.1 | worst 1.56e-6 vs the derived truncation bound 6.06e-6 |
| | gridded, the 8-node blend at the cell centres, h = 0.2 / 0.1 / 0.05 | worst 0.287 / 0.0748 / 0.0181: **observed orders 1.94, 2.05** (expected 2) |
| | the estimate's interpolation term vs the actual error at 64 random points | worst error/(interpolation + rounding + truncation) 0.62 / 0.68 / 0.81 (≤ 1 asserted) |
| `the_bond_kind_follows_the_sign_of_the_laplacian` | two atoms 2.4 apart + a wide bump b (exact samples, h = 0.25): sign change at b* = 0.12395 | b = 0.2: −0.2277 (exact −0.2282), band 0.028, shared; b = 0.05: +0.2223 (+0.2218), band 0.034, closed-shell; b*: +0.0004, band 0.031, indeterminate; depleted pair (`NotAccumulated`) +0.3144 (+0.3140), closed-shell |

### The noise, measured the way δ was (`the_laplacian_at_the_bond_point_converges_and_its_noise_is_measured`)

The bonded-pair fixture of δ's measurement at its 24 placements (4 shifts × 6 orientations):

| h | ∇²ρ_b mean (exact at the midpoint −2.0936) | spread over placements | estimated noise mean / max (interpolation mean, location mean) | worst \|∇²ρ_b − exact\| / noise | spread / (2 × max noise) |
|---|---|---|---|---|---|
| 0.25 (49³) | −2.0850 | 0.1994 | 0.372 / 0.446 (0.325, 0.047) | 0.287 | 0.224 |
| 0.125 (97³) | −2.0994 | 0.0616 | 0.111 / 0.132 (0.075, 0.036) | 0.277 | 0.234 |

Both asserted ≤ 1: the per-point estimate covers the measured placement noise with a factor ≈ 3.5
to spare, before the 3× of the band. So the band is **not** δ's rule (3 × a measured spread) but 3 ×
an upper estimate: on the fixture at h = 0.25 the mean band is 1.12 against δ-style 3 × spread =
0.60, on the real H₂ at 1.45 (81³ × 0.15) 0.29 against 0.146 — ≈ 2× more conservative, erring
toward "not resolved". The estimate leaves out the density's own errors (SCF convergence, the
egg-box error of the solved ρ); those are covered only empirically, by the 12-placement H₂ checks
below. The 2⁻²⁴ rounding term is a floor, not a bound (orbital-summed f32 errors are not white).

**Why per point and not one fixture constant (as δ).** The spread of ∇²ρ_b/(ρ_b h²) over the
placements is 11.7 (h = 0.25) and 13.1 Bohr⁻⁴ (h = 0.125), a 3× band of 39.3·ρ_b·h². On the
measured cases:

| case | grid | ρ_b | \|∇²ρ_b\| | constant band 39.3·ρ_b·h² | per-point band | kind (constant / per-point) |
|---|---|---|---|---|---|---|
| H₃ long contact R₂ = 2.0 | 107³ × 0.15 | 0.104 | 0.179 | 0.092 | 0.090 | shared / shared |
| H₃ R₂ = 2.25 | 107³ × 0.15 | 0.077 | 0.073 | 0.068 | 0.054 | shared / shared |
| H₃ R₂ = 2.5 | 107³ × 0.15 | 0.057 | 0.020 | 0.050 | 0.034 | indet. / indet. |
| H₃ R₂ = 2.55 | 107³ × 0.15 | 0.054 | 0.013 | 0.048 | 0.031 | indet. / indet. |
| H₂ R = 4.0 | 107³ × 0.15 | 0.022 | 0.0076 | 0.020 | 0.0011 | indet. / shared |
| H₂ R = 4.5 | 107³ × 0.15 | 0.015 | 0.0022 | 0.013 | 0.0005 | indet. / shared |
| H₃ R₂ = 3.0 (`NotAccumulated`) | 107³ × 0.15 | 0.033 | 0.017 | 0.029 | 0.014 | indet. / closed |
| H₃ R₂ = 3.25 (`NotAccumulated`) | 107³ × 0.15 | 0.025 | 0.0205 | 0.022 | 0.0096 | indet. / closed |
| H₂O O–H | 95³ × 0.15 | 0.342 | 2.807 | 0.30 | 2.731 | shared / shared (margin 1.03) |
| HeH⁺ | 49³ × 0.25 | 0.271 | 1.616 | 0.67 | 1.756 | shared / indet. |

The two give the same kinds on every drawn H–H bond of the user's case; the constant is wider near
the floor (H₂ R = 4.0–4.5, H₃'s `NotAccumulated` contact at 3.0–3.25), and the per-point band is
2–9× **wider** at the asymmetric Route-D points (O–H, HeH⁺). That is the reason for the per-point
estimate: a symmetric fixture — bond point at a midpoint, L stationary along the chord — cannot see
the location error at an asymmetric (Route D) bond point, and its ρ_b·h² scaling carries the
fixture's length scale everywhere; the per-point estimate adapts to the local curvature and slope
of L. (Physics review round 1 corrected an earlier claim here that the constant "would have marked
every diffuse stretched contact indeterminate": on these numbers it does not.)

**The real surface's noise** (12 placements of H₂ as in the q-noise measurement; each the same
molecule shifted 0, h/4, h/2, 3h/4 along x × axis along x, xy, xyz):

| grid | R | ∇²ρ_b mean | spread | estimated noise mean / max | spread / (2 × max noise) | f32 rounding term |
|---|---|---|---|---|---|---|
| 49³ × 0.25 | R_e 1.789 | −0.5530 | 0.0463 | 0.095 / 0.134 | 0.173 (≤ 1 asserted) | ≈ 3e-5 |
| 81³ × 0.15 | 1.45 (the preset) | −1.1685 | 0.0488 | 0.096 / 0.156 | 0.156 (≤ 1 asserted) | ≈ 3e-5 |

### The oracles (`tests/tools.rs`, run of record 2026-10-15)

| system | grid | ρ_b | ∇²ρ_b | band | kind | asserted |
|---|---|---|---|---|---|---|
| H₂ at the preset's 1.45 | 81³ × 0.15 | 0.232 | −1.198 | 0.469 | shared (12/12 placements) | shared |
| H₂ at R_e(σ = 0.25) 1.789 | 49³ × 0.25 | 0.149 | −0.570 | 0.401 | shared (12/12) | shared |
| He₂ at 3.0 (`NotAccumulated`) | 49³ × 0.25 | 0.057 | +0.167 | 0.030 | closed-shell (12/12) | closed-shell |
| He₂ at 3.0 (`NotAccumulated`) | 81³ × 0.15 | 0.049 | +0.203 | 0.038 | closed-shell | (one-off, printed) |
| H₃⁺ edges (ring) | 49³ × 0.25 | 0.113 | −0.285 / −0.288 | 0.110 / 0.193 | shared | — |
| HeH⁺ at 1.46 (Route D) | 49³ × 0.25 | 0.271 | −1.616 | 1.756 | indeterminate | — |
| LiH at R_e(σ) 3.366 | 65³ × 0.25 | 0.042 | +0.075 | 0.181 | indeterminate (the closed-shell sign, unresolved) | — |
| H₂O O–H, 3 placements × 2 | 57³ × 0.25 | 0.274–0.296 | −1.50 … −2.02 | 1.45–1.97 | shared 5, indeterminate 1 | never closed-shell |
| H₂O O–H (preset geometry) | 95³ × 0.15 | 0.342 | −2.807 | 2.731 | shared (\|∇²ρ_b\|/band 1.03) | (one-off, printed) |

**Water and the smearing (the brief's question).** The O–H bond points take Route D and sit near
the hydrogen: at 95³ × 0.15, 0.339 Bohr = 2.3σ from H and 1.471 Bohr = 9.8σ from O; at 57³ × 0.25,
0.23–0.34 Bohr = 0.9–1.4σ from H and 5.9–6.3σ from O. ∇²ρ changes steeply along the chord there:
the noise is interpolation 0.39–0.49 + location 0.09–0.20 e/Bohr⁵ at h = 0.25 and 0.45 + 0.46 at
0.15, a third of the value. So the sign is negative at every point (the QTAIM verdict for O–H;
asserted as the raw sign, ∇²ρ_b < 0 at all six 57³ points, where |∇²ρ_b|/noise ≈ 3), but only
just resolved as a kind (|∇²ρ_b|/band 0.93–1.03), and the number mixes the bond with the
hydrogen's own concentration — not comparable with literature values. During MD an O–H line may
switch between solid and dotted; recorded, not hidden (rule 7). **Not a smearing artefact, and M7
will not fix it:** at h = 0.15 the point is 2.3σ from H (erf(1.63) ≈ 0.98 of the H charge
enclosed; the potential within ≈ 2 % of a point charge's); the location term is intrinsic to
hydrogen's density slope — dL/dr ≈ 8.1 e/Bohr⁶ at r = 0.339 × the chord step 0.057 Bohr = 0.46,
the measured 0.465 — and a sharper unsmeared H makes it larger. **Known follow-up, not done in
M6:** a refined bond point, the parabolic vertex of q (Route D) or ρ̂ (Route S) through the chord
samples k − 1, k, k + 1 with L evaluated there. No hysteresis on the kind (an underived margin; a
history-dependent display).

**H₂ stretched** (107³ × 0.15, multiplicity None — the spin-symmetric solution, as the app):

| R | 1.45 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 4.75 | 5.0 | 5.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| ρ_b | 0.232 | 0.128 | 0.079 | 0.051 | 0.033 | 0.022 | 0.015 | 0.012 | 0.0102 | 0.0084 |
| ∇²ρ_b | −1.197 | −0.358 | −0.131 | −0.052 | −0.021 | −0.0076 | −0.0022 | −0.0009 | −0.0000 | — (`BelowFloor`) |
| band | 0.468 | 0.103 | 0.033 | 0.011 | 0.004 | 0.0011 | 0.0005 | 0.0006 | 0.0006 | — |
| kind | shared | shared | shared | shared | shared | shared | shared | shared | indeterminate | — |

**No detected-but-closed-shell window exists for H₂ at the default grid:** ∇²ρ_b reaches zero at
R ≈ 5.0, at about the same R as ρ_b meets the 0.01 floor (the scan resolves both to ≈ ±0.25 Bohr). The LCAO estimate of the module doc (∇²ρ/ρ =
2κ² − 8κ/R, sign change at R ≈ 4/κ, 4 Bohr for κ = 1) places it earlier; the spin-symmetric
solution's orbital is more diffuse than a free 1s (κ < 1; the dissociation caveat of "Tools (M6
Task 7)"), which moves it out. Asserted: shared at every R ≤ 4.5 (|∇²ρ_b| ≥ 3.9 × band).

**Linear H₃, H₀–H₁ = 1.45, H₁…H₂ = R₂** (107³ × 0.15; the user's report — "three H in a near-line,
bonds drawn on both sides, labelled H₃"):

| R₂ | long contact verdict | ρ_b | ∇²ρ_b | band | kind | components |
|---|---|---|---|---|---|---|
| 2.00 | `Bonded` | 0.104 | −0.179 | 0.090 | shared | one (H₃) |
| 2.25 | `Bonded` | 0.077 | −0.073 | 0.054 | shared | one (H₃) |
| 2.50 | `Bonded` | 0.057 | −0.020 | 0.034 | indeterminate | one (H₃) |
| 2.55 | `Bonded` | 0.054 | −0.013 | 0.031 | indeterminate | one (H₃) |
| 2.60 | `NotAccumulated` | 0.051 | −0.007 | 0.029 | indeterminate | H₂ + H |
| 2.65 | `NotAccumulated` | 0.048 | −0.002 | 0.026 | indeterminate | H₂ + H |
| 2.70 | `NotAccumulated` | 0.046 | +0.002 | 0.024 | indeterminate | H₂ + H |
| 2.75 | `NotAccumulated` | 0.043 | +0.006 | 0.022 | indeterminate | H₂ + H |
| 3.00 | `NotAccumulated` | 0.033 | +0.017 | 0.014 | closed-shell | H₂ + H |
| 3.25–4.00 | `NotAccumulated` | 0.025–0.012 | +0.018 … +0.021 | 0.008–0.010 | closed-shell | H₂ + H |
| ≥ 4.25 | `BelowFloor` | ≤ 0.009 | — | — | — | H₂ + H |

The short bond is shared throughout (∇²ρ_b −1.06 … −1.20, band ≤ 0.47). **So the contact the user
saw is shared-shell on this density while it is a bond.** That is not by itself the LSDA error: near
the exact H + H₂ transition state (1.757/1.757 Bohr) exact densities also show a bond path between
all three; what LSDA mainly gets wrong is the energetics (no barrier) and the magnitudes, and how
much of this contact's density is delocalisation error is not measured here. What tells the contact
apart on the display is its ρ_b — 0.06–0.10 against the short bond's 0.22–0.23 — a **modest** cue,
since ρ_b is LSDA's own and inflated by the same error: at R₂ = 2.0 `strength_fraction` is 0.69
(ρ_b 0.104) against 0.91 for the short bond (0.218), i.e. width 0.78 vs 0.93 of full (−16 %) and
alpha 0.80 vs 0.94 of full; the sign goes indeterminate (dotted) at the edge of the bonded range
(2.5–2.55, the longest bonded R₂ 2.55). ∇²ρ_b of the contact crosses
zero at R₂ ≈ 2.68 — after the detector has dropped it (2.60) — so in H₃ too there is **no
bonded closed-shell window**. Asserted (only where |∇²ρ_b| ≥ 2 × band): the short bond shared at
every R₂; the long contact bonded and shared at R₂ = 2.0; `NotAccumulated` and closed-shell at
3.25–4.0; the longest bonded R₂ in [2.5, 2.75).

### The display (a display scale, not physics; `app::overlay`)

Line width and opacity follow log ρ_b from the detector's floor 0.01 to `RHO_STRONG` = 0.3
e/Bohr³ (the upper end of covalent ρ_b; H₂'s ≈ 0.27 near the exact density, the brief's number —
this solver's is 0.232 at 1.45 Bohr), to 30 % width and 35 %
alpha at the floor; shared solid, closed-shell dashed (0.22/0.14 Bohr), indeterminate dotted
(0.05/0.12 Bohr); a multi-centre ring keeps its violet colour. Screenshot of record: linear H₃
(1.45 + 2.5) at 81³ × 0.15 — the short bond solid, the long contact thin, faint and dotted.

## Drag force (wheel) (user request 2026-10-14, branch `drag-force-wheel`; `sim` unit tests)

The wheel while dragging sets the drag spring's force cap ("pull"); k = `K_DRAG` = 0.5 Ha/Bohr² and
γ = 2√(k m) are unchanged, so the integrator's stability is (ω dt = 0.17, γ dt/m = 0.34 < 2 for ¹H
at 0.25 fs). Derivation: `sim::dynamics` module doc, "Tool forces" item 2. Far from the target a
dragged nucleus runs at v_t = F/γ with kinetic energy F²/(8k), mass-independent; ¹H (m = 1837.15
mₑ, γ = 60.6) is the worst case for the per-step jump. Not a `validate` binary: analytic providers.

| constant | value | why |
|---|---|---|
| `sim::dynamics::F_DRAG_MAX` (default pull) | 0.1 Ha/Bohr (5.1 eV/Å) | R12, unchanged: about the largest restoring force of the H–H bond (Morse D_e a/2 ≈ 0.175 × 1.03 / 2 = 0.09 Ha/Bohr), so a default drag can break a bond, slowly |
| **Drag force (wheel)** — ceiling `F_DRAG_HI` | 1.0 Ha/Bohr (51.4 eV/Å, 82 nN) | ¹H at terminal speed v_t = 1/60.6 = 0.0165 a.u. moves v_t dt = 0.17 Bohr per 0.25 fs step = 0.68 h at h = 0.25 — below one grid step per solve, so the SCF warm start stays a nearby guess (cost, not correctness); K = F²/(8k) = 0.25 Ha, a quarter of the 1 Ha per axis the walls stop, and a full-strength pull into a wall rests at δ = a/2 + F/k_w = 0.54 Bohr inside the 1 Bohr layer; injected power F_s·v ≤ F_max\|v\|, balanced by the damping at terminal speed (F v_t in, γ v_t² out); the critically damped approach from terminal speed (x₀ = −d_c, v₀ = ω d_c/2) does not overshoot the (cell-clamped) target. ≈ 10× the H–H bond's maximum restoring force and ≈ 4× the strongest bond of H–Ne (N₂: D_e a/2 = 0.364 × 1.423/2 = 0.26 Ha/Bohr, a from ω_e = 2358.6 cm⁻¹); pulling the lighter atom of a bond tears it in a few fs (the heavy end of an X–H bond drags the H along) |
| **Drag force (wheel)** — floor `F_DRAG_LO` | 0.01 Ha/Bohr (0.51 eV/Å, 0.82 nN) | a gentle pull (≈ 1/9 of the H–H maximum restoring force: stretches without tearing; the single-molecule force-spectroscopy scale) still ≈ 30× the grid's egg-box net force (~3e-4 Ha/Bohr at h = 0.25); lower is useless interactively — at the floor ¹H moves 1.7e-3 Bohr per step, ≈ 150 fs (≈ 50 s of wall at a DFT run's ~3 fs/s) per Bohr |
| app wheel step (`ui::DRAG_WHEEL_STEP`) | × 1.25 per wheel line (pixel wheels: 40 px = 1 line, as the zoom) | a UI choice: 10.3 lines from the default to either end of the decade-each-way range |

| test | asserted bound | derivation | measured |
|---|---|---|---|
| `drag_force_setting_applies_the_new_cap_and_books_the_change` | the step after a raise (0.1 → 0.5) and a lower (0.5 → 0.02) at d > d_c: Δv = (F_new/m) dt to 1e-17 / 1e-18; first step of the next drag from rest x₁ = ½ (F/m) dt² to 1e-16; each change booked to 1e-14 Ha; no stale forces, no history restart | an undamped spring beyond d_c is a constant force F_max (both kicks); U changes at fixed r by U(d; F_new) − U(d; F_old), booked into W | Δv 2.812870e-3 = (0.5/m) dt |
| `drag_force_setting_outside_the_range_is_refused` | 0, −0.1, 0.0099, 1.01, NaN, ∞ refused (`last_rejection`), nothing changes; 0.01 and 1.0 accepted | the range above | — |
| `drag_force_booking_matches_the_closed_forms_across_the_corner` (review M4) | ΔW at fixed r to 1e-15: linear → quadratic (k d − F_o)²/(2k) = 0.0225 Ha at d = 0.5, F 0.1 → 0.5; quadratic → quadratic 0 exactly; back across −0.0225; linear → linear (F_n − F_o)(d − (F_n + F_o)/(2k)) | the Huber potential's three regimes | — |
| `a_pull_transient_leaves_the_closed_form_damping_residual` (review I1) | start from rest and full-speed stop (cap → 0.01), at 0.1 and 1.0 Ha/Bohr: ΔE_conserved = −c · ½ m e₀² to 1e-9 relative; arrival (Huber corner) term \|·\| ≤ 0.3 × the start's over D = 3…12 Bohr; the drift readout of one start at the ceiling < −1e-3 Ha/ps (red) | linear regime, a = ω dt = 0.1705 for ¹H: e ↦ (1 − a)² e, R = −½ m a³(2 − a) e² per step (exact algebra of Verlet + explicit damping kicks + the trapezoidal booking ½(f₁ + f₂)·Δr), summed: c = a³(2 − a)/(1 − (1 − a)⁴) = 1.72221e-2; the arrival bound is empirical (phase-dependent C¹ corner) | start −4.305526e-5 / −4.305526e-3 Ha, stop −3.487476e-5 / −4.219846e-3 (all = closed form); arrival worst +8.85e-4 Ha = 0.205 × 4.3055e-3; drift −1.307e-4 Ha/ps (0.1, amber) / **−1.221e-2 Ha/ps (1.0, red)** over the 2000-step window = 0.5 ps after one start |
| `drag_force_changes_stay_within_the_damping_residual_bound` (review I2; replaces the round-0 test with the (ω dt)²/4 · W_user bound, which was derived for a nucleus trailing a moving target and which a wheel flick exceeds 1.9×) | raise 0.1 → 1 at step 30, drop to 0.01 mid-flight at 60, a wheel flick 0.01 ↔ 1 every 4 steps for six cycles, hold 0.3; each change booked to 1e-14; max \|ΔE_conserved\| ≤ Σ_changes c · ½ m \|e₀\|² × (1 + 1e-9) | each transient's residual is the partial sum of its own geometric series (≤ the full one), so the sum over the transients the changes start is an exact upper bound in the linear regime (target 60 Bohr away: no arrival) | 4.1017e-2 Ha vs bound 4.2774e-2 (ratio 0.959) |
| fix round 1, not kept: the damping booked per kick by the kinetic energy it changes, F_d·(v_before + v_after)/2 · dt/2 | — | linear-regime algebra: R = −m a² b e/2 per step (b = F dt/2m), a start from rest sums to −a²/(2 − a) · ½ m v_t²; still one-signed for a start, opposite in sign for a stop | start −3.9727e-5 / −3.9727e-3 Ha (0.1 / 1.0, vs −4.3055e-5 / −4.3055e-3 with the trapezoid: 8 % less); stop +3.58e-6 / +3.93e-5 (vs −3.49e-5 / −4.22e-3); flick (12 Bohr target, 0.01 ↔ 1 every 4 steps, 6 cycles) −1.542e-2 (vs −3.343e-2). Does not remove the one-signed start term: booking kept as it was |
| `a_drag_pull_change_during_a_relax_is_applied_booked_and_restarts_fire` (merge of `main` into `m6-tools`: the pull × relax) | one ¹H in a k = 0.3 well, dragged toward x = 3 from x = −2, relaxing; the pull raised 0.1 → 0.5 once FIRE's Δt has grown: accepted, booked at fixed positions to 1e-15, FIRE restarted (α 0.1, Δt 0.25 fs), the augmented force = 0.5 − k x to 1e-12; \|ΔE_conserved\| ≤ 1e-10 Ha over every FIRE step; converged at \|x − F/k\| ≤ F_TOL/k | beyond d_c = F/K_DRAG the capped spring is the constant force F, so the balance is k x = F; every FIRE step books ΔU_spring at the current cap (`tool_potentials`) | 6 steps before the change, converged after 29 at x = 1.668122 (F/k = 1.666667); max \|ΔE_conserved\| 2.2e-16 Ha |

## Quality presets (user request 2026-10-14, branch `empty-preset`; report-only)

The app's Scene ▸ Quality levels (`crates/app/src/quality.rs`) trade speed for accuracy through the
grid spacing h, with σ = h and the box kept (12 Bohr for the H/H₂/He/empty presets). What each buys
on H₂ and what it costs, measured by `crates/validate/tests/quality.rs`
(`h2_bond_length_per_quality`, `#[ignore]`d, run with `-- --ignored --nocapture`; 2026-10-14, RX 7900
XTX, release): `GpuKs` at tol 1e-6, R_e(σ = h) by bisection of the Hellmann-Feynman F_x(nucleus 0)
in [1.40, 2.20] Bohr to 1e-4 with linear interpolation in the last bracket (15 warm solves); then a
cold solve at R_e (solver built, state reset) and a warm re-solve 0.01 Bohr away, after one untimed
warm-up solve that compiles the pipelines.

| level | h = σ (Bohr) | grid (12 Bohr box) | H₂ R_e(σ = h) | vs 0.7414 Å (experiment) | cold solve at R_e | warm re-solve (+0.01 Bohr) |
|---|---|---|---|---|---|---|
| Fast | 0.30 | 41³ | 1.8931 Bohr = 1.0018 Å | +0.260 Å | 0.12 s (9 SCF steps) | 0.07 s (6) |
| Balanced (old default) | 0.25 | 49³ | 1.7889 Bohr = 0.9467 Å | +0.205 Å | 0.15 s (9) | 0.07 s (6) |
| **Accurate (default)** | 0.15 | 81³ | 1.6008 Bohr = 0.8471 Å | +0.106 Å | 0.53 s (13) | 0.28 s (9) |
| Precise | 0.10 | 121³ | 1.5241 Bohr = 0.8065 Å | +0.065 Å | 0.76 s (13) | 0.46 s (8) |
| σ → 0 (the oracle's extrapolation, not a level) | — | 81³ ladder | 1.4511 Bohr = 0.7679 Å | +0.027 Å (the LSDA functional's own error: LSDA 1.446) | — | — |

Cross-checks: Balanced reproduces the M4 pin R_e(σ = 0.25, 49³) = 1.78894 (Δ < 1e-4), Accurate the
oracle scan's R_e(σ = 0.15) = 1.6008 on the same 81³ × 0.15 grid (CPU, tol 1e-7). The gap to
0.741 Å is almost all the Gaussian smearing's repulsive 2πσ²Σρ(R_I) term (it shrinks ≈ ∝ σ²; the σ → 0
extrapolation leaves the LSDA functional's +0.005 Bohr); rule 7: the app labels each level with its
number. The app's per-level cost readout is a rough estimate: the scene's last solve × (n′/n)³. It
*over*estimates finer levels from a coarse solve — the GPU is latency-bound (dispatch and readback
overhead) below ~10⁶ points, so the time grows much slower than n³ there: from the 49³ cold solve it
predicts 0.15 × (81/49)³ = 0.68 s for 81³ (measured 0.53) and from 81³ 0.53 × (121/81)³ = 1.77 s
for 121³ (measured 0.76), although the SCF step count also grows (9 → 13). Scaled down from a large
grid it underestimates; either way it can be well off (review I1, rule 7: the readout says "≈" and
"rough").

## M7 radial atom (Task 1)

`qm::radial` (branch `m7-pseudopotentials`, 2026-10-15, release, Ryzen 7 9800X3D): the spherical
all-electron Kohn–Sham atom on the universal log grid r_i = 1e-5·e^{0.008 i} Bohr up to 60 Bohr
(`radial::LEDGER`), Numerov in x = ln r, DIIS to Σ_σ∫|Δρ_σ|r²dr < 1e-10. Tests in
`crates/qm/src/radial/{grid,numerov,atom}.rs` and `sim::sampling::tests`.

**Source and functional.** NIST "Atomic Reference Data for Electronic Structure Calculations"
(Kotochigova, Levine, Shirley, Stiles & Clark 1997; https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations,
element pages and "Procedure", read 2026-10-15). The LDA/LSD columns use **Vosko–Wilk–Nusair**
correlation (the Ceperley–Alder fit, "VWN5"; procedure page "The local-density functional", parameter
table on "The correlation term": P 0.0310907 / −0.10498 / 3.72744 / 12.9352, F 0.01554535 / −0.32500 /
7.06042 / 18.0578, α_c −1/(6π²) / −0.0047584 / 1.13107 / 13.0045), with Dirac exchange, non-relativistic.
So the like-for-like oracle runs a **test-only VWN5** (`radial::atom::vwn5`, analytic potentials, checked
against finite differences); the generator's functional stays PW92 (rule: generation = solver) and its
totals are *reported* against the VWN5 ones (below), never asserted against NIST. NIST lists the
**majority spin first** in the LSD column ("down" in their convention; our spin index 0).

**Citation correction (rule 7).** NIST's Li page prints LDA **−7.335195** and LSD **−7.343957**. The
"Li atom, LSD total energy −7.335195" row at the top of this file (and the M7 plan's Task 1 brief)
was the **LDA** column. The radial tests use each digit where it belongs (`li_atom_radial_lsd` vs
−7.343957; `restricted_radial_lda_matches_nist_lda` vs −7.335195). Review round 1 (2026-10-15)
corrected the top row, the oracle-design row and "Known limitation", and the 3-D `li_atom_lsda`
oracle now asserts −7.343957 with its bound 4e-2 unchanged; its verdict is unchanged (still
`#[ignore]`d: +340 mHa against the corrected reference, +331 against the old one).

| Element | NIST LSD total | NIST LDA total | Radial VWN5 LSD (M) → Δ | Radial VWN5 restricted → Δ | Test, bound |
|---|---|---|---|---|---|
| H | −0.478671 | −0.445671 | −0.47867076 (2) → +2.4e-7 | −0.44567052 → +4.8e-7 | `h_atom_radial_lsd` 1e-5; LDA 1e-5 |
| He | −2.834836 | −2.834836 | −2.83483562 (1) → +3.8e-7 | same | `he_atom_radial_lsd` 1e-5; LDA 1e-5 |
| Li | −7.343957 | −7.335195 | −7.34395669 (2) → +3.1e-7 | −7.33519519 → −1.9e-7 | `li_atom_radial_lsd` 1e-5; LDA 1e-5 |
| C | −37.470031 | −37.425749 | −37.47003065 (3) → +3.5e-7 | −37.42574854 → +4.6e-7 | `second_row_radial_lsd` 1e-4; LDA 1e-4 |
| N | −54.136799 | −54.025016 | −54.13679866 (4) → +3.4e-7 | −54.02501614 → −1.4e-7 | same |
| O | −74.527410 | −74.473077 | −74.52740981 (3) → +1.9e-7 | −74.47307680 → +2.0e-7 | same |
| Ne | −128.233481 | −128.233481 | −128.23348127 (1) → −2.7e-7 | same | same |

Every total, every energy component (T_s, E_en, E_H, E_xc; worst |Δ| 4.9e-7) and every occupied
eigenvalue of both columns (worst |Δε| 4.9e-7, LSD both spins) agrees with NIST to its printed
rounding (5e-7). `o_lsd_eigenvalues_match_nist` pins O's six LSD eigenvalues (1s −18.766963 /
−18.713693, 2s −0.915554 / −0.801638, 2p −0.381133 / −0.272349; bound 1e-5).

**The functional-fit gap (rule 7): PW92 − VWN5**, same solver, spin-polarised ground states:
H −4.0e-5, He +3.80e-4, Li +6.72e-4, C +1.77e-3, N +2.41e-3, O +2.58e-3, Ne +3.56e-3 Ha
(PW92: −0.478711, −2.834455, −7.343284, −37.468257, −54.134387, −74.524830, −128.229917).
These are the generator's own all-electron references; the NIST numbers are VWN.

**O multiplicity scan (PW92, spherical, `o_triplet_is_lowest`):** E(M=1) = −74.470692,
**E(M=3) = −74.524830**, E(M=5) = −74.184436 → triplet lowest (ΔE vs singlet −54.1 mHa). With VWN5:
−74.473077 / −74.527410 / −74.187140. Hund's multiplicity emerges for every Z ≤ 18 (H 2, He 1, Li 2,
Be 1, B 2, C 3, N 4, O 3, F 2, Ne 1, Na 2, Mg 1, Al 2, Si 3, P 4, S 3, Cl 2, Ar 1). Since review
round 1 no neutral candidate with Z ≤ 18 is skipped (He M=3 −2.120257; Li M=4 −5.270233, which the
DIIS-only SCF lost); a candidate that fails is recorded in `AtomResult::skipped` with its reason
(`failed_scan_candidates_are_recorded`).

**SCF fallback (review M1, round 1).** A DIIS step that unbinds an occupied shell returns to the last
good step and mixes linearly, β = 0.3/2^k, history cleared, up to `LEDGER.numerics.backtrack_max` = 6
consecutive halvings; only then is the shell reported `Unbound { after_backtracking: true }`.
`diis_failure_falls_back_to_linear_mixing` (restricted, fixed occupations; ε pinned to the review's
independent linear-mixing probe, 1e-5): H 1s^1.25 ε_1s −0.137944 (E −0.491827, 20 steps); O 1s²2s²2p^4.5
ε_2p −0.112286 (E −74.581468, 20); Li 1s²2p¹ ε_2p −0.058122 (E −7.270213, 29) — all three failed before.
Also converging now: H 1s^1.5 (ε −0.056488), C 2p^2.5 (−0.037859), Li 1s²2s^0.25 2p^0.75.

**The integer anions are unbound in the spherical LSDA atom (rule 7; `integer_anions_are_genuinely_unbound`).**
Every first- and second-row anion Z = 1…18 at every allowed multiplicity ends with a genuine `Unbound`
after the full fallback (the HOMO of the fractional sequence crosses 0 between q = −0.5 and −1: H
ε_1s −0.2335 / −0.1379 / −0.0565 at q = 0 / −0.25 / −0.5). That is the known LSDA anion failure — no
−1/r tail, self-interaction (Shore, Rose & Zaremba, "Failure of the local exchange approximation in the
evaluation of the H⁻ ground state", Phys. Rev. B 15, 2858 (1977); bibliographic data verified via Crossref,
DOI 10.1103/PhysRevB.15.2858, at Task 4 (2026-10-15) — the paper's text was not read; the claim cited is its title's) — not a solver defect, and T3's "−1 anion if bound in
both" must report it as such rather than skip it silently or confine the electron.

**LSDA scaling identity** (`lsda_scaling_identity_holds`, bound 1e-5 relative to T_s): worst
|residual| **6.4e-11** (H restricted; Z ≤ 18, both modes, ≤ 1.9e-11 for Z ≥ 2). Before the inner
sphere ∫₀^{r0} was added to `LogGrid::integrate` it was 3.0e-8 for Ne — the Coulomb-weighted
integrands E_en and ∫ρV are ∝ r at the origin, and 2πZρ(0)r0² ≈ 4e-6 Ha for Ne was missing from each
(cancelling in the total); `coulomb_weighted_integral_includes_the_inner_sphere` pins it.

**Hydrogenic eigenvalues** (`hydrogenic_eigenvalues_are_exact`): bound |ε − (−Z²/2n²)| < 1e-10·Z² —
**spec revision M1 (plan review 2026-10-15): the error is relative, ∝ δ⁴, so the absolute bound
scales as Z²** — plus the order check (Z = 1 1s error ratio δ 0.016 → 0.008 in [14, 18]; measured
16.05). Measured at δ = 0.008: Δε/Z² from −4.0e-12 (4f) to −4.0e-11 (Z = 2 3s, the worst, −1.59e-10
absolute, margin 2.5×); Z = 8: 1s −1.11e-9, 2s −1.55e-9, 2p −5.1e-10, 3d −3.4e-10, 4f −2.6e-10.
Order (`numerov_eigenvalue_error_is_fourth_order`, Z = 1 1s, δ = 0.064 … 0.004): −6.67e-8, −4.17e-9,
−2.60e-10, −1.62e-11, −8.0e-13; Z = 8 1s: −4.27e-6, −2.67e-7, −1.67e-8, −1.11e-9, −1.67e-10 — clean
δ⁴ to δ = 0.008, then a ≈ 1e-10 floor at Z = 8 (≈ 3e-12 relative; recurrence roundoff, it does not move
with r0 ∈ {1e-5, 1e-6, 1e-7}).

**Grid helpers: bounds derived from the truncation terms (ruling (a), Task 1 review 2026-10-15).** The
plan brief's absolute bounds (1e-11 on `cumulative`, 1e-10 on `interp`) were not derived and were
missed by exactly the prescribed methods' leading truncation terms; they are replaced by tests that
assert the error *is* that term (a wrong weight changes its coefficient):
`cumulative_matches_its_truncation_term` — c_i − exact = −(11/720)δ⁴[g‴(x_i) − g‴(x_0)], g = f·r,
f = 4r²e^{−2r}: at r = 1.300924 predicted +2.30701e-10, measured +2.30690e-10 (bound 1 % of the term);
`interpolation_matches_its_truncation_term` — p − f = −[t(t−1)(t−2)(t−3)/24]δ⁴D⁴f, f = re^{−r}:
predicted / measured −1.9144e-13 / −1.9100e-13 (r ≈ 0.01), 9.4454e-12 / 9.4597e-12 (0.5),
−1.19969e-10 / −1.19547e-10 (2), −3.3767e-11 / −3.2979e-11 (7) (bound 5 % + 1e-15). Order test:
1.49e-5 / 9.42e-7 / 5.90e-8 at δ = 0.128 / 0.064 / 0.032 (orders 3.99, 4.00). No downstream task needs
better than ~1e-8 relative from either helper (review table: V_H, the T2 norm — the same rule on both
sides —, KB integrals, the Bessel filter). `integrate` (1e-12), `derivative` (1e-9) and
`interp_deriv` (1e-7) pass as before.

**Cost** (one thread): one SCF solve 20–30 DIIS steps (H 16–18, Li restricted 26–30, Na 24–29), 0.03–0.25 s;
`ground_state` (multiplicity + Δ_scan scans): ≤ 0.13 s restricted, 0.04–1.5 s polarised (Ar 1.45 s).
`cargo test -p qm --release radial`: 25 pass, 0 ignored, ≈ 6 s (after review round 1).

## M7 generation records (Task 4; regenerated by `pseudo_gen`)

`qm::radial::generate` (branch `m7-pseudopotentials`, 2026-10-15, release, Ryzen 7 9800X3D): Z in, the first
(f, local) of the universal ladder that passes G1/G2 and T1–T7 (spec §4.4, §4.6, §4.8), with the LFC partial
core (§4.5). The block between the markers is the pinned outputs table (review S6a) for every Z = 1–18, compared
verbatim by `crates/validate/tests/pseudo_gen.rs` `generation_record_table_matches_references_md`. When it moves,
the test writes the new block to the gitignored `crates/validate/last-run-generation-records.md` and fails; a row
that moves while the hash line does not fails with "bump LEDGER.version" (plan review M6; exercised by hand at
Task 4: perturbing N's row gave exactly that message). Copy only after physics review. Columns: r_c per generated
channel; E_KB and GSS per projector (Ha); r_nlc the LFC radius, r_core the §5.6 overlap radius (Bohr); h_req is
Task 5's. H and He are provisional until Task 11's dimer gate; Al–Ar are enabled in the app by Task 12.

<!-- m7-generation-records:begin -->
recipe_hash = 0xddd92db0e4be8d38 (m7-tm-kb-nlcc-3)

| Z | el | f | local | r_c (Bohr) | Z_v | E_KB (Ha) | GSS (Ha) | r_nlc | r_core | h_req |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | H | 1.4 | s-local | s: 1.479 | 1 |  |  | — | 0.000 | 0.35 |
| 2 | He | 1.6 | s-local | s: 0.915 | 2 |  |  | — | 0.000 | 0.20 |
| 3 | Li | refused | Z = 3: two valence s states (1s at -1.878 Ha, 2s at -0.106 Ha) — a one-projector TM/KB potential cannot carry both; refused until the ONCV stage (spec §4.3) | | | | | | | |
| 4 | Be | refused | Z = 4: no entry of the f ladder passes the gates at this recipe version: f = 1.6, HighestChannel: T5 l = 1: max δE_equiv 38.05 mHa on ε_l ± 0.5 Ha; f = 1.6, Polynomial: T5 l = 0: max δE_equiv 96.20 mHa on ε_l ± 0.5 Ha; f = 1.4, HighestChannel: T5 l = 0: max δE_equiv 6.16 mHa on ε_l ± 0.5 Ha; f = 1.4, Polynomial: T5 l = 0: max δE_equiv 22.46 mHa on ε_l ± 0.5 Ha; f = 1.2, HighestChannel: T5 l = 0: max δE_equiv 6.56 mHa on ε_l ± 0.5 Ha; f = 1.2, Polynomial: T5 l = 1: max δE_equiv 14.05 mHa on ε_l ± 0.5 Ha; f = 1, HighestChannel: T5 l = 0: max δE_equiv 6.53 mHa on ε_l ± 0.5 Ha; f = 1, Polynomial: T5 l = 0: max δE_equiv 6.26 mHa on ε_l ± 0.5 Ha | | | | | | | |
| 5 | B | 1.0 | p-local | s: 1.514, p: 1.564 | 3 | s: 3.6291 | s: 4.0810 | 0.749 | 1.322 | 0.35 |
| 6 | C | 1.2 | p-local | s: 1.467, p: 1.432 | 4 | s: 2.0335 | s: 5.1324 | 0.575 | 1.074 | 0.30 |
| 7 | N | 1.4 | polynomial | s: 1.421, p: 1.354 | 5 | s: 7.6284, p: -1.4575 | s: 2.9650, p: -3.6527 | 0.460 | 0.908 | 0.30 |
| 8 | O | 1.4 | p-local | s: 1.220, p: 1.136 | 6 | s: 0.9375 | s: 8.4165 | 0.379 | 0.786 | 0.20 |
| 9 | F | 1.6 | p-local | s: 1.230, p: 1.126 | 7 | s: 0.6711 | s: 8.8822 | 0.321 | 0.691 | 0.15 |
| 10 | Ne | 1.6 | p-local | s: 1.100, p: 0.991 | 8 | s: 0.5287 | s: 11.4438 | 0.278 | 0.618 | — |
| 11 | Na | refused | Z = 11: two valence s states (2s at -2.063 Ha, 3s at -0.103 Ha) — a one-projector TM/KB potential cannot carry both; refused until the ONCV stage (spec §4.3) | | | | | | | |
| 12 | Mg | refused | Z = 12: two valence s states (2s at -2.903 Ha, 3s at -0.175 Ha) — a one-projector TM/KB potential cannot carry both; refused until the ONCV stage (spec §4.3) | | | | | | | |
| 13 | Al | refused | Z = 13: no entry of the f ladder passes the gates at this recipe version: f = 1.6, HighestChannel: T5 l = 0: max δE_equiv 10.84 mHa on ε_l ± 0.5 Ha; f = 1.6, Polynomial: T5 l = 0: max δE_equiv 27.52 mHa on ε_l ± 0.5 Ha; f = 1.4, HighestChannel: T5 l = 0: max δE_equiv 7.51 mHa on ε_l ± 0.5 Ha; f = 1.4, Polynomial: T5 l = 1: max δE_equiv 11.10 mHa on ε_l ± 0.5 Ha; f = 1.2, HighestChannel: T5 l = 1: max δE_equiv 8.81 mHa on ε_l ± 0.5 Ha; f = 1.2, Polynomial: T5 l = 0: max δE_equiv 6.04 mHa on ε_l ± 0.5 Ha; f = 1, HighestChannel: T5 l = 1: max δE_equiv 6.40 mHa on ε_l ± 0.5 Ha; f = 1, Polynomial: T5 l = 0: max δE_equiv 10.69 mHa on ε_l ± 0.5 Ha | | | | | | | |
| 14 | Si | 1.0 | p-local | s: 1.777, p: 2.153 | 4 | s: 13.4867 | s: 2.2095 | 1.211 | 1.806 | 0.35 |
| 15 | P | 1.2 | p-local | s: 1.880, p: 2.188 | 5 | s: 7.6041 | s: 2.0005 | 1.057 | 1.615 | 0.35 |
| 16 | S | 1.2 | p-local | s: 1.694, p: 1.925 | 6 | s: 5.5355 | s: 2.3949 | 0.937 | 1.455 | 0.35 |
| 17 | Cl | 1.4 | polynomial | s: 1.792, p: 1.988 | 7 | s: 4.3175, p: -21.8515 | s: 3.2008, p: -0.2870 | 0.845 | 1.333 | 0.30 |
| 18 | Ar | 1.4 | polynomial | s: 1.654, p: 1.806 | 8 | s: 3.6766, p: -13.7063 | s: 3.8382, p: -0.3113 | 0.767 | 1.220 | 0.25 |
<!-- m7-generation-records:end -->

**Status (review round 1, recipe `m7-tm-kb-nlcc-2`).** T5 is gated per channel on ε_l ± 0.5 Ha (both end points
exact) plus every `g2_missing` AE energy of that l, at 5 mHa = T4; the maximum over ε_l ± 1 Ha is reported
(`GateReport::t5_far`). This is the Task 4 physics review's rule-6 revision by derivation (spec §4.8 "Window and
threshold": δE_equiv is the first-order eigenvalue error of a channel-l state wholly inside r_d, E_PP − E_AE =
q·δE_equiv verified on Dirichlet-confined O 2p to four digits), not a loosening; the first version and its refusals
are recorded below. In outcome the revision accepts every entry the first version accepted, and more (B, Si–Cl); it
removes an extrapolation the measurement falsified, without lowering the stated standard (5 mHa over the ±0.5 Ha
bonding range). Accepted: H, He, B, C, N, O, F, Ne, Si, P, S, Cl, Ar. Refused: Li, Na, Mg (two valence s states,
partition), Be and Al (T5 at every ladder entry; pinned, like Li). Every row of the block was regenerated by this
code and agrees with the reviewer's independent scratch probe (f, local and T5 per channel to the printed digits; no
differing row).

**f against the literature TM radii (spec §4.4: "a check, not an input").** The classical hand-chosen TM radii
correspond to f ≈ 1.2 (C), ≈ 1.45 (N), ≈ 1.6 (O). The recipe lands C at 1.2 (p local), matching; N at 1.4 with the
polynomial local (1.4 p-local fails T5 at 5.10 mHa, at ε_p + 0.5) and O at 1.4 (1.6 fails at 6.66 mHa, at
ε_p + 0.5): both harder than the literature. The classical radii themselves, measured with this recipe's LSDA-PW92
+ LFC by the Task 4 reviewer, give at ε_p + 0.5 Ha: O (r_c = 1.45) ≈ 12.2 mHa (≈ 55 on the first version's
window), N (1.50) ≈ 8.7, C (1.50) ≈ 5.2 p-local / ≈ 4.3 polynomial. So "harder than the literature" for N and O
is the spec's original estimate (2–5 mHa at ±0.3–0.5 Ha) being low for TM potentials, not a recipe defect
(rule 7: recorded, not corrected).

### Gate numbers (round 1, measured; reported, not pinned)

T3 Δ = ΔE_PP − ΔE_AE (cation / s→p / Hund). T5 gated: max over ε_l ± 0.5 Ha; "far": max over ε_l ± 1 Ha (at
E − ε_l = +1.00 in every case except F s, −1.00). T6: worst |slope_PP − slope_AE| over (l, l′). T7: rows with equal
counts / rows. Thresholds (`LEDGER.gates`): T1 1 mHa, T2 1e-10, T3/T4 5 mHa, T5 5 mHa, T6 10 mHa/e.

| Z | f, local | T1 (Ha) | T2 | T3 Δ (mHa) | T4 (mHa) | T5 s / p (mHa) | far s / p (mHa) | T6 (mHa/e) | T7 |
|---|---|---|---|---|---|---|---|---|---|
| H | 1.4, s-local | 1.2e-9 | 3.3e-16 | −0.751 / — / +0.310 | 0.050 | 4.915 / — | 24.8 / — | 0.35 | 6/6 |
| He | 1.6, s-local | 4.2e-9 | 5.3e-15 | −1.224 / — / 0.000 | 3.865 | 2.186 / — | 10.4 / — | 1.74 | 6/6 |
| B | 1.0, p-local | 4.3e-9 | 3.0e-15 | −0.273 / −0.144 / −0.190 | 0.709 | 2.788 / 3.380 | 17.5 / 19.3 | 0.24 | 12/12 |
| C | 1.2, p-local | 7.0e-9 | 3.6e-15 | −0.322 / −0.276 / −1.120 | 1.984 | 1.508 / 3.983 | 9.6 / 21.4 | 0.36 | 12/12 |
| N | 1.4, polynomial | 5.0e-9 | 4.8e-15 | −0.267 / −0.498 / −3.077 | 3.708 | 1.191 / 4.151 | 5.6 / 21.8 | 1.18 | 12/12 |
| O | 1.4, p-local | 1.3e-8 | 2.6e-15 | −0.176 / −0.713 / −1.313 | 2.140 | 0.618 / 3.114 | 3.7 / 15.7 | 1.22 | 12/12 |
| F | 1.6, p-local | 1.2e-8 | 3.2e-15 | −0.113 / −1.026 / −0.326 | 2.283 | 0.174 / 4.584 | 0.7 / 22.3 | 2.51 | 12/12 |
| Ne | 1.6, p-local | 1.6e-8 | 5.7e-15 | +0.003 / — / 0.000 | 2.525 | 0.071 / 3.290 | 0.7 / 15.9 | 2.60 | 9/9 |
| Si | 1.0, p-local | 2.0e-9 | 6.7e-16 | −0.089 / +0.035 / −0.299 | 0.691 | 2.353 / 4.176 | 12.8 / 24.1 | 0.22 | 12/12 |
| P | 1.2, p-local | 3.5e-9 | 4.3e-15 | −0.143 / +0.096 / −0.826 | 1.014 | 4.810 / 4.469 | 22.2 / 22.7 | 0.30 | 12/12 |
| S | 1.2, p-local | 4.1e-9 | 1.4e-15 | −0.163 / +0.066 / −0.451 | 0.672 | 4.098 / 3.309 | 21.1 / 17.9 | 0.32 | 11/12 |
| Cl | 1.4, polynomial | 4.4e-9 | 6.7e-15 | −0.204 / +0.282 / −0.130 | 0.591 | 0.872 / 4.191 | 2.4 / 20.4 | 0.37 | 12/12 |
| Ar | 1.4, polynomial | 5.2e-9 | 1.2e-15 | −0.202 / — / 0.000 | 0.488 | 0.990 / 2.885 | 3.6 / 14.7 | 0.41 | 9/9 |

Thin T5 margins are what the ladder produces (it keeps the softest passing entry): H 1.7 %, P 3.8 %, F 8 %.
δE_equiv at ε_l is ≈ 1e-8 everywhere (O: 6.3e-9 s, 9.4e-9 p; bound 1e-4). H and He are provisional (Task 11).

**T7.** Every row has n_KB ≤ n_AE; the one row with fewer is S⁺ (cation, restricted, l = 2: the AE ion binds a 3d
at −0.1496 Ha below the window top −0.1208, the KB atom none), recorded in `T7Row`, not a ghost.

**Dropped d (Al–Ar), rule 7.** The 3d is unbound in the neutral LSDA atom, so no d channel is generated and the local
potential carries the d scattering. Its δE_equiv (the q = 1 eigenvalue-error equivalent) is 34–108 mHa at ε_p (Si 70,
P 55, S 84, Cl 79, Ar 108; Al 34, given for reference — Al is refused at this recipe version; Task 4 re-review),
against 2–5 mHa for the gated s and p channels. With the d weight at polarisation level, ordinary bonding (SiH₄, PH₃,
H₂S, HCl) is affected at the few-mHa level. Configurations with substantial d character are outside this recipe's
validated transferability until a d channel is generated from an ionised or scattering reference (ONCV stage). These
are cations in which the all-electron 3d binds (measured: S⁺, AE 3d −0.150 Ha, none in the KB atom), and hypervalent
or high-oxidation-state compounds (SF₆, SO₄²⁻, PCl₅, ClO₄⁻).

Skipped configurations (rule 7, `GateReport::skipped`): s→p for H, He (no p channel), Ne, Ar (p full); the anion
for He, Ne, Ar (no room). The anion of H, B, C, N, O, F, Si, P, S, Cl is **unbound in both atoms**: AE and KB each
end `Unbound { after_backtracking: true }` after the DIIS fallback and the linear-mixing confirmation pass. This is
the LSDA anion failure: no −1/r tail, self-interaction (Shore, Rose & Zaremba, "Failure of the local exchange
approximation in the evaluation of the H⁻ ground state", Phys. Rev. B 15, 2858 (1977); bibliographic data verified
via Crossref, DOI 10.1103/PhysRevB.15.2858, on 2026-10-15 — the paper's text was not read; the claim cited is its
title's). The electron is never confined into binding (that would be a new recipe ingredient).

The (n+1)s (Task 3 review N8; `GenerationRecord::extra_states`): the weak s level the short-range neutral LSDA well
binds, AE → KB (Ha) — an LSDA artefact of the self-interaction-truncated potential; the real atom has a full
Rydberg series, far deeper: B 3s −0.005766 → −0.005787; C −0.006094 → −0.006117; N −0.005465 → −0.005437; O
−0.004502 → −0.004524; F −0.003449 → −0.003454; Ne −0.002434 → −0.002443; Si 4s −0.013760 → −0.013799; P −0.013528
→ −0.013658; S −0.012500 → −0.012654; Cl −0.011102 → −0.011067; Ar −0.009549 → −0.009501. No element has a
`g2_missing` entry. The 60-Bohr box shifts Be 3s by 2.6e-5 Ha (Task 3 review probe); harmless, because AE and KB
counts use the same box.

### NLCC and r_core (spec §4.5, §5.6)

r_nlc is the outermost grid point with ρ_c ≥ ρ_v (AE valence density) and ρ_c > 0; r_core encloses all but 1e-3 of
N_c (Bohr):

| | B | C | N | O | F | Ne | Si | P | S | Cl | Ar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r_nlc | 0.749 | 0.575 | 0.460 | 0.379 | 0.321 | 0.278 | 1.211 | 1.057 | 0.937 | 0.845 | 0.767 |
| r_core | 1.322 | 1.074 | 0.908 | 0.786 | 0.691 | 0.618 | 1.806 | 1.615 | 1.455 | 1.333 | 1.220 |

H and He: no core (r_nlc —, r_core 0). O: r_core 0.786 against the spec's hydrogenic estimate ≈ 0.7; r_nlc 0.379
against the plan review's ≈ 0.45; LFC A = 0.693490, B = 6.898759 Bohr⁻¹ (x = BR = 2.618), ρ_pc(0) = AB = 4.78
e/Bohr³ against ρ_c(0) = 292.4; value continuity 1e-10, slope 1e-8 relative
(`partial_core_is_continuous_with_slope_at_r_nlc`). With LFC the Task 2 tail identity V_ion + Z_v/r = τ_c holds
beyond max(r_c, r_nlc) to 1e-10 and O's r_cmp = 1.5266 is the core tail's 1e-8 radius
(`v_ion_tail_is_the_core_hartree_tail_with_lfc`). r_nlc < max r_c for every accepted element, so r_β = max r_c.

**Cost:** one `generate` with every gate takes 0.4 s (He) to 15 s (Si); a refused element runs all 8 entries.

### T5 as first written (recipe `m7-tm-kb-nlcc-1`, Task 4 commit 4042405; superseded in review round 1)

Kept as the record of why T5 was revised (spec §4.8 "Window and threshold", §11). The first version gated the
union window [ε_v,min − 1, ε_v,max + 1] Ha at 20 mHa; it refused Be, B and Al–Cl at every ladder entry and moved O
off f = 1.6.

Spec §4.8 T5: max |δE_equiv| over E ∈ [ε_v,min − 1, ε_v,max + 1] Ha on a 10 mHa mesh, at r_d = 1.2·max r_c, in
the Wronskian form ½W(û_AE, û_PP)(r_d) (Spec question 12).

Every measured maximum lies **at the top edge of the window, E = ε_v,max + 1 Ha** (positive-energy scattering).
None lies inside the window. Per element, at f = 1.6 and f = 1.0 with the p local:

| Z | f | r_d | s: max (E − ε_s) / max over ±0.5 Ha | p: max (E − ε_p) / max over ±0.5 Ha |
|---|---|---|---|---|
| Be | 1.6 | 3.917 | 78.4 (+1.00) / 5.0 | 82.4 (+0.87) / 24.8 |
| Be | 1.0 | 2.443 | 48.3 (+1.00) / 6.5 | 26.5 (+0.87) / 3.0 |
| B | 1.6 | 3.008 | 22.5 (+1.21) / 4.5 | 86.7 (+1.00) / 33.4 |
| B | 1.0 | 1.876 | 30.2 (+1.21) / 6.8 | 19.4 (+1.00) / 3.4 |
| C | 1.6 | 2.348 | 8.3 (+1.30) / 2.3 | 66.8 (+1.00) / 18.4 |
| N | 1.6 | 1.953 | 4.5 (+1.41) / 1.2 | 47.8 (+1.00) / 10.8 |
| O | 1.6 | 1.678 | 4.8 (+1.53) / 0.4 | 31.7 (+1.00) / 7.1 |
| F | 1.6 | 1.476 | 4.9 (+1.67) / 1.1 | 22.2 (+1.00) / 7.5 |
| Ne | 1.6 | 1.320 | 5.6 (+1.82) / 1.9 | 15.7 (+1.00) / 7.8 |
| Al | 1.6 | 5.060 | 137.1 (+1.18) / 28.1 | 19.1 (+1.00) / 1.5 |
| Ar | 1.6 | 2.483 | 67.3 (+1.50) / 33.5 | 15.6 (+1.00) / 4.1 |

All values in mHa.

**O at f = 1.6** (r_c,s = 1.398, r_c,p = 1.301 Bohr; the classical TM O used 1.45): the miss is the local p
channel, where KB equals semilocal exactly (both columns agree), at the top edge.
- δE_equiv(E) at E − ε_p = −1.53, −1.03, −0.53, −0.03, +0.47, +0.77, +0.97 Ha: 9.6, 7.1, 3.1, 0.0, 5.7, 17.7,
  29.7 mHa.
- At ±0.3 Ha it is 1.5–1.6 mHa and at ±0.5 Ha 3.1–5.7 mHa, close to the 2–5 mHa the spec's own estimate gives
  there (§4.8, "Why T5 is gated in energy").
- At the window top (E = +0.662 Ha) it is 31.7 mHa, against the estimate "≈ 22 mHa for a 0.05 rad mismatch".

The other refusals by reason, at every entry of the ladder (f, local: T5 max in mHa unless stated):
- Be:
  - p local: 78.4, 53.7, 47.9, 48.3 for f = 1.6, 1.4, 1.2, 1.0;
  - polynomial: 304.6, 61.0, 45.1 (p), 42.0.
- B:
  - p local: 22.5, 21.4, 26.0, 30.2;
  - polynomial: T4 cation 5.35 mHa at 1.6, then T5 55.5, 36.6 (p), 29.8.
- Al–Cl (Task 12's elements): T5 at every entry, 20.2–181.9 mHa (`ladder_report_h_through_ar` keeps their
  G1/G2 stage).

**Checks that the numbers are the gate's and not the code's:**
1. δE_equiv(ε_l) ≈ 1e-8 for every channel tried.
2. The independent D form, ½(D_PP − D_AE)·√(w_AE w_PP)·sgn(û_AE û_PP), equals the Wronskian form to every
   printed digit at every E sampled.
3. The p (local) channel's δE_equiv equals the semilocal TM potential's exactly.
4. The growth away from ε_l is second order near it and faster towards the edge.
5. For the s channels the semilocal TM potential is *worse* than the KB form at the top edge (O s at f = 1.6:
   SL 80.6 vs KB 4.4 mHa at E = +0.63 Ha). So T5's s misses (Be, B, Al–Ar) are not a KB artefact.

### The near-threshold Unbound flag (Task 1 re-review R1-S1, fixed in Task 4)

The DIIS pass's fallback alone reported Cl 1s²2s²2p⁶3s²3p^q as genuinely unbound for q = 5.965–5.975, which
are bound. DIIS lands on an anchor whose 3p is just below 0, and every linear step from it ratchets the level
towards 0 (q = 5.97: −3.39 mHa at step 5, then −1.17, −0.54, −0.051 and −0.0049 mHa at the successive anchors).

`solve_atom` (and `solve_kb_atom`) now confirm with a second pass before reporting
`Unbound { after_backtracking: true }`. The second pass uses plain linear mixing from ρ = 0, β halved for good
at each failure.

Measured ε_3p (`near_threshold_states_are_confirmed_by_linear_mixing`, bound 2e-6 against the review's
independent linear-mixing probe), with 72 linear-pass steps each:
- q = 5.965: −6.019 mHa;
- q = 5.970: −4.648 mHa;
- q = 5.975: −3.280 mHa.

q = 5.95, 5.96, 5.98 and 5.985 are unchanged (−10.155, −7.394, −1.917 and −0.557 mHa). q = 5.99 is unbound by
both passes. Every integer anion Z = 1–18 at every allowed multiplicity still ends genuinely unbound, ≤ 0.55 s
each.

T3's anion arm files a configuration as "not bound in both" only on that flag, of either atom (R1-N1). Any
other error fails the gate.

## M7 filter (Task 5; recipe `m7-tm-kb-nlcc-3`)

`qm::radial::filter` (branch `m7-pseudopotentials`, 2026-10-15, release): the spherical-Bessel filter with King-Smith,
Payne & Lin band completion (24 Chebyshev terms, discrete Picard truncation), the outer C^∞ step over [γ_r R, γ_r R + 8h],
uniform 3-D sampling tables (Δ = h/256), and T8 per ladder h (spec §4.8, §5.1; Task 5 physics review,
`.superpowers/sdd/2026-10-15-m7-pseudopotentials/task-5-ruling.md`). Printed by `pseudo_gen`
(`generation_record_table_matches_references_md`); the records block above pins h_req only.

T8 per element and ladder h — (a) ΔE = E_filt − E_unfilt and max |Δε| of the reference KB atom, < 1 mHa (= T1);
(b) filtered T3 = max |ΔE_PP^filt − ΔE_AE|, < 5 mHa (≈ the unfiltered T3 error at fine h); (c) clamped = XC-domain
violations over the reference and every T3 configuration, = 0; (d) the table egg-box force F_SR / F_NL / F_pc (Ha/Bohr,
the §6.6 probe at {±h/4}³, each table with its own energy weight), sum < 2e-6. Energies in Ha. "fails" marks a row that
misses one of (a)–(d); h_req = the coarsest h from which every finer row passes.

| Z | h | ΔE | max Δε | T3 | clamped | F_SR | F_NL | F_pc | T8 |
|---|---|---|---|---|---|---|---|---|---|
| H | 0.35 | +1.12e-6 | 1.34e-5 | 7.52e-4 | 0 | 5.85e-10 | 0.00e0 | 0.00e0 | pass |
| H | 0.30 | -1.70e-5 | 8.42e-6 | 7.34e-4 | 0 | 1.67e-11 | 0.00e0 | 0.00e0 | pass |
| H | 0.25 | -4.05e-7 | 1.43e-7 | 7.51e-4 | 0 | 2.65e-10 | 0.00e0 | 0.00e0 | pass |
| H | 0.20 | -2.27e-7 | 1.11e-7 | 7.51e-4 | 0 | 7.97e-11 | 0.00e0 | 0.00e0 | pass |
| H | 0.15 | -5.86e-8 | 3.22e-8 | 7.51e-4 | 0 | 1.71e-11 | 0.00e0 | 0.00e0 | pass |
| He | 0.35 | +1.88e-2 | 3.49e-3 | 7.37e-3 | 0 | 6.07e-8 | 0.00e0 | 0.00e0 | fails |
| He | 0.30 | -5.87e-3 | 7.38e-4 | 3.92e-4 | 0 | 2.77e-8 | 0.00e0 | 0.00e0 | fails |
| He | 0.25 | -3.80e-3 | 4.22e-4 | 2.25e-4 | 0 | 8.91e-9 | 0.00e0 | 0.00e0 | fails |
| He | 0.20 | -5.24e-5 | 4.87e-6 | 1.22e-3 | 0 | 3.41e-9 | 0.00e0 | 0.00e0 | pass |
| He | 0.15 | +1.94e-6 | 1.30e-7 | 1.22e-3 | 0 | 2.14e-9 | 0.00e0 | 0.00e0 | pass |
| B | 0.35 | +4.47e-4 | 2.18e-5 | 3.03e-4 | 0 | 4.93e-10 | 1.14e-6 | 8.79e-9 | pass |
| B | 0.30 | +3.67e-5 | 2.81e-5 | 2.09e-4 | 0 | 8.30e-10 | 3.01e-7 | 9.05e-9 | pass |
| B | 0.25 | +3.00e-5 | 3.27e-6 | 2.65e-4 | 0 | 2.63e-10 | 4.24e-8 | 2.52e-9 | pass |
| B | 0.20 | +1.38e-5 | 5.35e-7 | 2.72e-4 | 0 | 1.59e-11 | 1.73e-8 | 2.75e-10 | pass |
| B | 0.15 | +3.18e-6 | 1.38e-7 | 2.72e-4 | 0 | 2.51e-11 | 3.07e-9 | 1.15e-11 | pass |
| C | 0.35 | +5.17e-3 | 1.27e-3 | 1.80e-3 | 0 | 4.70e-9 | 7.63e-6 | 1.04e-7 | fails |
| C | 0.30 | -3.42e-5 | 8.68e-5 | 1.10e-3 | 0 | 9.17e-10 | 1.22e-6 | 5.63e-8 | pass |
| C | 0.25 | +2.18e-4 | 5.04e-5 | 1.12e-3 | 0 | 2.86e-9 | 5.43e-7 | 3.82e-8 | pass |
| C | 0.20 | +5.43e-5 | 2.38e-6 | 1.12e-3 | 0 | 2.41e-10 | 1.43e-7 | 2.02e-9 | pass |
| C | 0.15 | +1.54e-5 | 5.96e-7 | 1.12e-3 | 0 | 3.80e-11 | 7.68e-9 | 7.88e-11 | pass |
| N | 0.35 | +5.86e-3 | 5.60e-4 | 3.75e-3 | 0 | 6.52e-10 | 7.89e-6 | 9.60e-6 | fails |
| N | 0.30 | +1.39e-4 | 2.79e-4 | 3.45e-3 | 0 | 3.32e-10 | 8.41e-7 | 8.74e-7 | pass |
| N | 0.25 | +7.09e-4 | 1.79e-5 | 3.08e-3 | 0 | 4.58e-10 | 7.42e-7 | 1.08e-7 | pass |
| N | 0.20 | +3.80e-4 | 1.16e-5 | 3.10e-3 | 0 | 7.22e-10 | 5.76e-8 | 8.03e-8 | pass |
| N | 0.15 | +7.07e-5 | 1.57e-6 | 3.08e-3 | 0 | 1.96e-10 | 1.15e-8 | 1.57e-10 | pass |
| O | 0.35 | +1.13e-1 | 1.83e-2 | 2.39e-2 | 14 | 3.80e-8 | 5.67e-6 | 5.01e-5 | fails |
| O | 0.30 | +6.54e-2 | 1.08e-2 | 1.37e-2 | 0 | 5.03e-8 | 4.80e-9 | 2.14e-5 | fails |
| O | 0.25 | +4.88e-3 | 8.08e-4 | 1.45e-3 | 0 | 8.35e-9 | 5.24e-6 | 3.41e-7 | fails |
| O | 0.20 | +6.61e-5 | 1.61e-4 | 1.31e-3 | 0 | 1.51e-8 | 9.32e-7 | 6.70e-9 | pass |
| O | 0.15 | +2.47e-4 | 3.56e-6 | 1.32e-3 | 0 | 1.26e-9 | 3.03e-8 | 2.48e-8 | pass |
| F | 0.35 | +2.51e-1 | 3.07e-2 | 4.01e-2 | 38 | 7.57e-8 | 7.76e-6 | 2.87e-5 | fails |
| F | 0.30 | +1.66e-1 | 2.22e-2 | 2.79e-2 | 8 | 1.22e-7 | 5.84e-8 | 6.64e-5 | fails |
| F | 0.25 | +2.29e-2 | 2.50e-3 | 2.53e-3 | 0 | 3.64e-8 | 5.61e-6 | 3.60e-5 | fails |
| F | 0.20 | -9.10e-4 | 3.48e-4 | 1.36e-3 | 0 | 3.03e-8 | 7.06e-7 | 2.11e-6 | fails |
| F | 0.15 | +9.24e-4 | 2.49e-6 | 1.03e-3 | 0 | 4.35e-9 | 1.99e-8 | 8.77e-9 | pass |
| Ne | 0.35 | +3.22e-1 | 2.61e-2 | 1.45e-2 | 82 | 5.57e-9 | 5.55e-5 | 1.12e-3 | fails |
| Ne | 0.30 | +3.64e-1 | 4.09e-2 | 1.69e-2 | 26 | 1.54e-7 | 6.65e-6 | 9.88e-5 | fails |
| Ne | 0.25 | +1.72e-1 | 2.01e-2 | 7.59e-3 | 0 | 1.88e-7 | 7.55e-6 | 5.26e-5 | fails |
| Ne | 0.20 | -4.22e-3 | 1.36e-3 | 3.02e-5 | 0 | 2.32e-8 | 5.82e-7 | 9.76e-7 | fails |
| Ne | 0.15 | +1.35e-3 | 3.23e-5 | 3.52e-6 | 0 | 2.31e-8 | 1.71e-7 | 2.45e-7 | fails |
| Si | 0.35 | +2.08e-4 | 1.96e-5 | 3.01e-4 | 0 | 2.79e-11 | 1.90e-7 | 2.97e-10 | pass |
| Si | 0.30 | +3.91e-5 | 2.94e-6 | 2.99e-4 | 0 | 1.54e-10 | 4.13e-8 | 6.65e-10 | pass |
| Si | 0.25 | +2.48e-5 | 2.59e-7 | 2.99e-4 | 0 | 2.07e-11 | 7.21e-9 | 2.94e-10 | pass |
| Si | 0.20 | +7.56e-6 | 1.96e-7 | 2.99e-4 | 0 | 1.11e-11 | 3.46e-10 | 4.04e-11 | pass |
| Si | 0.15 | +2.06e-6 | 7.14e-8 | 2.99e-4 | 0 | 2.40e-12 | 4.29e-11 | 1.44e-11 | pass |
| P | 0.35 | +3.35e-4 | 3.38e-5 | 8.29e-4 | 0 | 1.34e-10 | 1.77e-7 | 7.71e-9 | pass |
| P | 0.30 | +1.64e-4 | 2.64e-6 | 8.31e-4 | 0 | 1.41e-10 | 2.52e-8 | 4.79e-9 | pass |
| P | 0.25 | +5.59e-5 | 1.37e-6 | 8.26e-4 | 0 | 4.23e-11 | 1.23e-9 | 1.52e-10 | pass |
| P | 0.20 | +1.74e-5 | 3.33e-7 | 8.26e-4 | 0 | 1.70e-11 | 1.22e-9 | 1.01e-10 | pass |
| P | 0.15 | +5.21e-6 | 6.36e-8 | 8.26e-4 | 0 | 5.55e-13 | 2.26e-11 | 9.61e-13 | pass |
| S | 0.35 | +5.03e-4 | 2.21e-5 | 4.63e-4 | 0 | 1.23e-10 | 1.17e-7 | 6.87e-10 | pass |
| S | 0.30 | +3.97e-4 | 3.54e-5 | 4.50e-4 | 0 | 1.17e-10 | 1.76e-8 | 3.09e-9 | pass |
| S | 0.25 | +1.26e-4 | 2.39e-6 | 4.52e-4 | 0 | 1.52e-10 | 4.46e-9 | 2.51e-9 | pass |
| S | 0.20 | +5.23e-5 | 2.39e-7 | 4.51e-4 | 0 | 9.33e-11 | 7.29e-9 | 5.81e-10 | pass |
| S | 0.15 | +8.88e-6 | 8.14e-8 | 4.51e-4 | 0 | 2.69e-11 | 5.62e-11 | 4.23e-11 | pass |
| Cl | 0.35 | +1.35e-3 | 3.10e-5 | 2.92e-4 | 0 | 1.02e-10 | 8.93e-8 | 7.36e-8 | fails |
| Cl | 0.30 | +4.39e-4 | 2.95e-5 | 2.56e-4 | 0 | 1.34e-10 | 8.50e-8 | 9.74e-9 | pass |
| Cl | 0.25 | +3.11e-4 | 3.94e-6 | 2.79e-4 | 0 | 1.54e-11 | 1.96e-8 | 7.75e-9 | pass |
| Cl | 0.20 | +7.87e-5 | 5.03e-7 | 2.82e-4 | 0 | 4.81e-11 | 2.12e-9 | 2.02e-11 | pass |
| Cl | 0.15 | +2.32e-5 | 7.36e-8 | 2.82e-4 | 0 | 2.31e-11 | 1.34e-10 | 3.37e-11 | pass |
| Ar | 0.35 | +2.70e-3 | 1.40e-4 | 1.96e-4 | 0 | 1.16e-9 | 1.39e-7 | 8.21e-8 | fails |
| Ar | 0.30 | +1.10e-3 | 7.27e-5 | 2.24e-4 | 0 | 5.50e-10 | 6.44e-8 | 1.10e-7 | fails |
| Ar | 0.25 | +3.86e-4 | 3.87e-6 | 2.01e-4 | 0 | 1.73e-11 | 1.51e-8 | 8.92e-9 | pass |
| Ar | 0.20 | +1.27e-4 | 1.41e-6 | 2.03e-4 | 0 | 1.61e-11 | 1.17e-8 | 1.25e-9 | pass |
| Ar | 0.15 | +3.08e-5 | 5.76e-8 | 2.02e-4 | 0 | 5.38e-11 | 3.81e-10 | 2.31e-11 | pass |

**h_req** (an outcome per element, never a refusal of the potential): H 0.35, He 0.2, B 0.35, C 0.3, N 0.3, O 0.2, F 0.15, Ne **None**, Si 0.35, P 0.35, S 0.35, Cl 0.3, Ar 0.25. Ne has no h_req at this recipe version: at h = 0.15 it fails T8(a), ΔE = +1.35e-3 Ha (> 1 mHa); its (b)–(d) pass (T3 3.52e-6, clamped 0, F_atom 4.39e-07). The review's attribution (Table C below, step width h): at 0.15 the partial core carried 0.97 of Ne's 1.21 mHa. Against the brief's construction (step h, Task 5 WIP) Ar moves from 0.30 to 0.25 (T8(a): ΔE 1.10 mHa at 0.30 with the 8h step); He fails (a) at 0.25 (ΔE −3.80 mHa) under both. A scene finer than the ladder is certified by (a)–(d) at its own h (plan Task 6 `check`).

**Table extents at h_req** (Bohr; R_pc is the XC-weight support of spec §4.9, 4π∫_r^∞|v̄_xc|ρ_pc r′² dr′ ≤ 1e-6 Ha; every radius is γ_r R + 8h) and **min ρ_pc^filt** (e/Bohr³, at r; interior Gibbs lobes, allowed — T8(c) checks n_σ):

| Z | h_req | R_pc | V_SR radius | β radius | ρ_pc radius | min ρ_pc^filt |
|---|---|---|---|---|---|---|
| H | 0.35 | — | 7.658 | — | — | — |
| He | 0.2 | — | 4.535 | — | — | — |
| B | 0.35 | 2.241 | 7.896 | 5.146 | 6.162 | −1.05e-3 at 1.220 |
| C | 0.3 | 1.850 | 6.848 | 4.600 | 5.175 | −1.94e-3 at 0.960 |
| N | 0.3 | 1.576 | 6.884 | 4.531 (s, p) | 4.764 | −3.29e-2 at 0.780 |
| O | 0.2 | 1.376 | 4.654 | 3.430 | 3.664 | −3.89e-3 at 1.023 |
| F | 0.15 | 1.220 | 3.527 | 3.045 | 3.030 | −7.54e-3 at 0.851 |
| Ne | None (0.15 shown) | 1.100 | 3.527 | 2.850 | 2.850 | −1.25e-2 at 0.755 |
| Si | 0.35 | 3.291 | 8.277 | 6.030 | 7.736 | −2.42e-4 at 2.428 |
| P | 0.35 | 2.919 | 7.979 | 6.082 | 7.178 | −3.31e-4 at 2.206 |
| S | 0.35 | 2.651 | 8.020 | 5.688 | 6.777 | −1.26e-3 at 2.086 |
| Cl | 0.3 | 2.428 | 6.920 | 5.382 (s, p) | 6.042 | −1.43e-3 at 1.806 |
| Ar | 0.25 | 2.223 | 5.821 | 4.709 (s, p) | 5.335 | −8.96e-4 at 1.602 |

The uniform tables hold ≈ 4–6 × 10³ entries per table. They reproduce the exact band-limited function to ≤ 1.8e-11 of
max|g| (O 0.20, C 0.30, F 0.15; derived bound 5.3e-10, `uniform_tables_match_the_band_limited_function`).

**Probe checks.** O at h = 0.20 with the outer step one spacing wide (the brief's construction): F_NL = 8.01e-5 Ha/Bohr
(review 8.0e-5; with 8h, 9.3e-7). A projector band-limited and compact to 1e-13 (e^{−d²/0.289}, O's sphere at h = 0.15):
F_NL = 7.7e-13; the plan text's e^{−d²/0.5} at h = 0.25 still has 2.7e-9 of its content at π/h and gives 7.8e-12 — the
probe measures that leakage.

**Weight-1 lattice spreads (report only — not a judge; spec §5.1 "Translation invariance").** h = 0.25, the {0, h/2}³
shifts (the eight ±h/4 sums are identical by symmetry: 0.00 for every table): C — V_SR 1.77e-9, ρ_pc 4.95e-8, β_s sum
1.86e-7, β_s norm 2.45e-10; O — V_SR 8.63e-9, ρ_pc 2.32e-6, β_s sum 1.23e-6, β_s norm 3.45e-9. Poisson's 12|ĝ(G₁)| +
16|ĝ(G√3)| reproduces each to the printed digits (`lattice_spread_sources_report`); the uniform tables add ≤ 2.6e-11. For
comparison, the brief's construction (step h, log-grid interpolation) gave C/O V_SR 9.2e-8/2.1e-7, β_s sum 1.0e-5/6.7e-5.
On the TM-class model of plan review Q3 (h = 0.25, r_c = 1.4) the 8h completion's first and √2 shells are 3.21e-9 and
1.48e-10 of F(0) (hard cut 1.57e-6 / 5.56e-7, reproduced to the review's digits).

**Why ρ_pc keeps the band completion and Ne has no h_req** (Task 5 physics review, measured in its scratch probe; step
width h). Table C, T8(a)'s ΔE (Ha) with one table filtered at a time (the parts are not additive):

| Z, h | all | only ρ_pc | only β | only V_loc (V_LR + V_SR) |
|---|---|---|---|---|
| C 0.30 | −1.95e-4 | +3.60e-4 | −3.08e-4 | −4.05e-4 |
| N 0.30 | −2.22e-4 | −5.00e-4 | +1.63e-5 | +2.10e-4 |
| O 0.25 | +3.88e-3 | −7.01e-4 | +3.80e-4 | +4.01e-3 |
| O 0.20 | +1.31e-4 | +6.23e-4 | +1.49e-4 | −6.66e-4 |
| O 0.15 | +1.86e-4 | +1.71e-4 | −1.15e-5 | +1.65e-5 |
| F 0.25 | +1.92e-2 | +5.64e-4 | +7.36e-4 | +1.74e-2 |
| F 0.15 | +8.64e-4 | +8.13e-4 | −6.90e-6 | +4.14e-5 |
| Ne 0.20 | −7.42e-3 | +6.34e-4 | −4.37e-5 | −8.11e-3 |
| Ne 0.15 | +1.22e-3 | +9.68e-4 | −2.80e-5 | +2.40e-4 |

At coarse h the local potential dominates (F and Ne are hard p-local potentials); at h = 0.15 the partial core does: LFC's
ρ_pc equals the true 1s tail beyond r_nlc, decaying as e^{−2Z_eff r}, so its spectrum is still large at q_c. Table D, a
positivity-preserving window for ρ_pc (a Gaussian of σ = h), ρ_pc-only ΔE (Ha):

| Z | h = 0.25 | 0.20 | 0.15 |
|---|---|---|---|
| C | +3.99e-2 | +2.96e-2 | +1.92e-2 |
| N | +7.63e-2 | +5.83e-2 | +3.92e-2 |
| O | +1.28e-1 | +9.94e-2 | +6.92e-2 |
| F | +1.92e-1 | +1.52e-1 | +1.09e-1 |
| Ne | +2.69e-1 | +2.13e-1 | +1.56e-1 |

19–270 mHa against 0.2–1.0 mHa for the band-completed ρ_pc: a nonnegative kernel must damp the band below q_c that the
partial core needs. Rejected; the requirement is the LSDA domain (T8(c)), never violated at a certified h (O at 0.35 and
F, Ne at 0.30–0.35 do violate it — all above their h_req). A smoother model core is a recipe revision for the ONCV stage.

## M7 grid integration (Task 6; recipe `m7-tm-kb-nlcc-3`)

`qm::ionic` (policies, `IonicSetup::build` with `check()` = T8 at the scene's h, core overlap), `qm::nonlocal` (the KB
spheres, V_NL in LOBPCG, E_NL), `qm::nuclear` (centres, V_LR + the one V_SR table, ρ_pc, E_ii with Z_v, the sphere
lists) and the SCF integration (`System.ionic`, N = Σ Z_v − charge, LSDA on ρ_σ + ½ρ_pc, the warm-start setup guard).
CPU, f64, branch `m7-pseudopotentials`, 2026-10-15, release; `cargo test -p qm --release` and
`cargo test -p validate --release --test pseudo`. Forces on pseudised states are Task 7's: until then `qm::forces`
refuses an ionic state (an all-electron integrand on a pseudo state would be a plausible wrong number).

**`IonicPolicy::None` is today's solver, bit for bit.** `none_policy_is_bitwise_today` pins E and F_x of H₂ (R = 1.4) and
LiH (33³, h = σ = 0.25) to the bits measured on the code before M7 (86974e3): H₂ `0xbfee7a76b2327147` /
`0x3fc07b1358ca116a` (10 SCF steps), LiH `0xc014e13070f24df5` / `0xbf97397795d2fce0` (13). `--test gpu_vs_cpu` before
and after: every energy, component, step count and bound identical (only the wall-time columns differ); 4/4 pass.

**The S1 operator identity** (`v_nl_on_grid_is_the_operator_identity`, O, unfiltered table, off a grid point). With
ε = E_KB⟨β|φ⟩_grid − 1, V_NLφ − δVφ = ε·δVφ pointwise and ⟨φ|V_NL|φ⟩ = (1 + ε)⟨φ|δV|φ⟩: all three of the plan's checks
measure the one quadrature ε. The TM junction at r_c is C² (V‴ jumps), so the transform of δV·φ² falls as q⁻⁵ and
ε ∝ h⁵ with an offset-dependent sign (aliasing). Lattice sum of the radial functions alone, at offsets
(0.013, −0.021, 0.017) / (h/2)³:

| h | 0.2 | 0.141 | 0.1 | 0.0707 | 0.05 | 0.0354 |
|---|---|---|---|---|---|---|
| ε | +8.7e-5 / −1.2e-4 | −5.9e-6 / −9.3e-6 | −2.7e-6 / +3.8e-6 | −3.9e-8 / +8.5e-7 | −4.9e-8 / −1.2e-7 | +1.6e-8 / +2.1e-8 |

The plan review's S8 estimate (~1e-7 at h = 0.1) was 30× low. The operator (`NonlocalOp`) is checked at its offset as
follows.
- h = 0.1: ε = −2.692e-6, max |V_NLφ − δVφ| = 2.69e-6 of max |δVφ| = 4.480 Ha. Only S8's separate |ε| < 1e-4 is
  asserted here.
- h = 0.05: ε = −8.01e-8, pointwise 2.69e-7 relative, ⟨φ|V_NL|φ⟩ 1.0666976699 vs ⟨φ|δV|φ⟩ 1.0666977884. The plan's
  1e-6 bounds are unchanged and asserted here.
- ε falls 34× from h = 0.1 to 0.05. A missing √((2l+1)/4π) would give ε = 3.19, independent of h.

The S1 deviation (bounds at h = 0.05, not 0.1) is for physics review. The clipped sphere (0.4 Bohr inside a face, ~59 % of
its points kept) is symmetric to 1e-12 relative.

**The local split and the partial core on the grid.**
- `local_potential_is_the_exact_split`: O, identity filter, h = 0.2. V_grid against the radial V_loc gives 1.02e-7 Ha
  away from r_c,p (derived bound 9.0e-6), and 1.11e-6 within three log intervals of the C² junction r_c,p = 1.1355
  (J3 = 214 Ha/Bohr³; bound 1.2·Δ·4.5·s²|J3| = 7.5e-5). Both are double-interpolation residues of the identity table,
  which the solver never samples. Beyond the V_SR table, V_grid equals the analytic erf to 1e-15. Beyond 6 Bohr,
  |V + Z_v/d| = 0.
- `partial_core_field_carries_the_model_core`: O at h = 0.2, filtered, off a grid point. ∫ρ_pc on the grid is
  0.6711674559, against 4π∫ρ_pc r²dr = 0.6711674558 (1.9e-10 relative). The no-core branch (H under `All`) adds exactly
  0.
- O r_core = 0.786 Bohr (spec §5.6). O–O at 2.28 and O–H at 1.81 Bohr are silent.

**Pseudo-Ne, 3-D against radial** (`pseudo_ne_3d_matches_radial`, spec §6.3; tol 1e-6 / eig 1e-7, 81³, 12 Bohr box).
- E_3D = −35.923019 against the radial KB atom on the same filtered table, −35.922123: **Δ = −0.90 mHa** (< 2 mHa),
  17 SCF steps.
- ε_3D: 2s −1.322333 and 2p −0.497696 (threefold to 2e-15), against radial −1.322634 and −0.497847.
- Face density 3.8e-8; no warnings.
- `IonicSetup::build` refuses Ne at h = 0.15 (TooCoarse, h_req None: T8(a) misses there, an outcome; Task 5 ruling
  item 4). The comparison therefore uses `from_tables`.
- At h = 0.125, finer than the ladder, Ne is certified by its own T8 row: ΔE +0.923 mHa, Δε 0.004 mHa, T3 0.002 mHa,
  clamped 0, F_table 1.6e-8 Ha/Bohr.

**Open-shell O: why spec §5.7 exists (round 0, integer aufbau).** Neither O multiplicity converges under integer aufbau
occupations. This is the same failure M6 Task 3 recorded for the all-electron atom (above, "Tools (M6)"); pseudisation
does not remove it. Measured on pseudo-O, h = 0.20.
- **Singlet** (41³, DIIS, eig_tol 1e-9): the 2p hole flips orientation every step. The minority quadrupole's diagonal
  swaps axes, and Δρ stays 1.2–2.7 for 60 steps, from a random or an axis-aligned start. In its own potential the empty p
  falls below an occupied one: there is no aufbau-consistent fixed point.
- **Triplet**: it is aufbau-consistent. At a loosely converged state the occupied minority 2p (−0.27127 Ha) lies
  11.8 mHa below the two empty ones (−0.25951, degenerate to 1e-6), and the majority 2p are −0.40357 ×2 and −0.33577.
  - The hole's orientation is a near-zero mode of the SCF map.
  - Linear mixing (α = 0.3, LOBPCG at 1e-9 every step, 41³) contracts geometrically to Δρ = 7.3e-5. It then contracts by
    only 2.7e-4 per step (λ_J ≈ 0.9991), while the minority quadrupole's off-diagonals drift (Q_xy 0.02265 → 0.02189,
    Q_xz −0.04075 → −0.03961 over 100 steps): the hole is rotating toward the grid's axis.
  - DIIS stalls at Δρ ≈ 8e-5 (41³) / 5e-4 (61³, on a grid point) and then destabilises (spikes to 1e-2–7e-2).
  - Guard vectors (k + 2 or k + 4 LOBPCG vectors) do not help. The extra vectors are box continuum states and LOBPCG
    hits its cap. The eigensolver is not the cause.

**Spec §5.7 as built (round 1; `Occupations::DegenerateShells`, τ = 1e-4 Ha, under the pseudo policies only).** `None`
stays integer aufbau: `none_policy_is_bitwise_today` still passes. Closed shells under the rule
(`closed_shells_are_aufbau_under_the_rule`) get integer f and the `Aufbau` energy: H₂ Δ −4.1e-12, LiH +3.7e-10 Ha.
- **The Hund gate** (`pseudo_o_triplet_below_singlet`, spec §6.3/§6.4; CoresOnly, h = 0.20, 14 Bohr, 71³, the atom at
  (0.013, −0.0091, 0.0052), tol 1e-6 / eig 1e-7, cold):

  | | 3-D | radial KB, same filtered table | Δ | SCF steps |
  |---|---|---|---|---|
  | M = 1 (2p shared 2/3 per spin at −0.338147, spread 2.2e-8) | −16.450188 | −16.449456 (restricted reference) | **−0.73 mHa** | 11 |
  | M = 3 (minority 2p shared 1/3 at −0.270708, spread 4.2e-7; majority 2p −0.381758 ×3) | −16.505645 | −16.504909 (T3 Hund) | **−0.74 mHa** | 11 |

  Splitting E(1) − E(3) = **55.456 mHa** (radial 55.453; all-electron spherical 54.138, "M7 radial atom"); witnesses are
  box continuum states at +0.031–0.040 Ha. The review's prototype gave −0.73 / −0.74 and 55.457; reproduced.
- **LSDA's integer lowering** (`pseudo_o_integer_triplet_report`, report; same grid and position, integer aufbau, linear
  mixing α = 0.3): E_int = −16.5097425 after 37 steps (energy stationary to 6.4e-8 over 20 steps; Δρ still 2.7e-4),
  E_sph = −16.5056447: **E_int − E_sph = −4.10 mHa**, the "few mHa" of spec §6.3 (rule 7).
- **On a grid point** (`ion_on_a_grid_point_is_finite`, 61³, tol = eig_tol = 1e-9): the triplet converges in 17 steps,
  E = −16.50543512, minority 2p shared 1/3 with spread 8e-16 (cubic site). The max LOBPCG residual 9.1e-7 is the
  continuum witness stopped at the 200-iteration cap (reported, spec §5.7).
- **Lone O and N scan** (`lone_o_and_n_scan_picks_hund`, 12 Bohr, every multiplicity converged): O at h = 0.20: M = 1
  −16.449979 (11 steps), **M = 3 −16.505435** (11), M = 5 −16.137895 (14); N at h = 0.30: M = 2 −10.146469 (12),
  **M = 4 −10.248716** (12, integer: the rule is inert). Hund's multiplicity is the lowest for both.
- **Site dependence: OPEN, for physics review.** At a site of lower than cubic symmetry the rule's τ test can see the
  grid's splitting of the p shell. At O on (0, h/2, h/2) (tetragonal site, 61³, tol 1e-9), the first SCF step's minority
  p levels, computed from the cold spherical guess, are split by **1.5e-4 Ha > τ** (−1.0524676 against −1.0523148: the
  compact first-step orbitals feel the lattice far more than the converged ones, whose splitting near a cubic site is
  ≤ 1e-6). The witness test passes at once, the occupation is integer, the state self-polarises, and it converges in
  23 steps to the integer state E = −16.509541550, 4.11 mHa below the shared one (= E_int − E_sph). The {0, h/2}³ set:
  the cubic sites (0 and (h/2)³) are shared (−16.505435, −16.505428); (h/2)·ê-type sites lie 1.45 mHa lower and
  (h/2)(ê + ê′)-type sites 4.11 mHa lower. A lone O's energy surface is therefore corrugated by 4.1 mHa on the lattice
  under the rule as written; molecules without a symmetry-degenerate partly filled set are unaffected (the rule is
  inert). Not tuned here (rule 6): τ's lower bound (spec §5.7) does not hold at every site and SCF step.

**Single-atom 3-D floor** (Task 5 ruling S2; Task 6 review ruling C: the Dirichlet wall subtracted, spec §6.6).
- The force is −∂E/∂R of the frozen R-dependent energy terms (∫ρV_loc, E_NL with the weights f_iσ, E_xc[ρ_σ + ½ρ_pc],
  E_ii), Richardson δ = 1e-4 / 2e-4, at tol = eig_tol = 1e-9. `qm::forces` is Task 7's. The wall is fitted from E(0)
  and E(h·ê_b) (E(−h·ê_b) = E(h·ê_b) by the centred box's inversion), κ = √(−2ε_max).
- **H (`All`) at h = 0.25, 12 Bohr:** raw max |F| over {±h/4}³ **2.90e-5**, wall **2.86e-5** (1.651e-5 per component;
  κ = 0.7316, E(h·ê_b) − E(0) = 8.338e-6 Ha), **egg-box (wall-subtracted) 4.13e-7 Ha/Bohr** (2.38e-7 per component;
  the review predicted 2.4e-7). Tables' F_atom 2.65e-10, so the valence floor is ≈ 4.1e-7, 1.4 % of the 3e-5 gate.
  Energy spread over {0, h/2}³: raw 6.26e-6, **wall-subtracted 5.4e-8 Ha**. The earlier reading of 2.90e-5 as "the whole
  §6.6 budget from one H" was the box wall (review ruling C).
- The force is the energy's derivative (round 0, H at R = (h/4)·(1,1,1)): HF F_x = −1.6753e-5 against −dE/dX from tight
  SCF energies (Richardson of δ = 0.01, 0.02) = −1.6743e-5, a difference of 9.4e-9. FD cross-check δ = 2e-3: 4.3e-10.
- **O (`CoresOnly`, triplet) at h = 0.20, 12 Bohr:** the 8 probes at {±h/4}³ (trigonal sites) converge shared (minority 2p 1/3 each) in 20 steps. Raw max |F| **4.81e-5**,
  wall **2.38e-5** (1.375e-5 per component; κ = 0.7348, E(h·ê_b) − E(0) = 5.536e-6 Ha), **egg-box (wall-subtracted)
  2.43e-5 Ha/Bohr** (1.403e-5 per component), against the tables' F_atom 9.54e-7: the **valence floor is ≥ 2.34e-5**, above
  Task 7's 1.5e-5 precondition (half the gate; O₂ has two O). Consistent with the energies: between the two cubic sites,
  both shared, E((h/2)³) − E(0) = 6.8e-6 Ha of which ≈ 4.2e-6 is the wall, leaving ≈ 2.6e-6 = 6A₁, A₁ ≈ 4.3e-7, i.e.
  A₁·2π/h ≈ 1.35e-5 per component. FD cross-check (δ = 2e-3): 6.2e-8. The {0, h/2}³ energy spread, 4.11e-3 Ha, is not an
  egg-box: it is the occupation pattern switching with the site (above, "Site dependence"): shared 1/3 at the cubic sites,
  1/2 over a pair at (h/2)·ê sites (−1.45 mHa), integer at (h/2)(ê + ê′) sites (−4.11 mHa); the lattice translates h·ê_b
  are shared (spread 6.3e-6). 2880 s for 19 solves at 4 at a time.
- **Box bias** (Task 6 review S4, its measurement): pseudo-H E(12 Bohr) − E(16 Bohr) = +0.30 mHa and ε_1s +1.2 mHa at a
  face density of 4.3e-7, below `BOUNDARY_DENSITY_WARN` = 1e-5: the face-density warning does not protect mHa- or
  force-level work for a −0.27 Ha level.

**Cost** (`pseudo_cost_report`, report-only; Ryzen 7 9800X3D, one thread per solve; spec §5.3 as corrected by the review).
- O at h = 0.20 on 61³: one s projector, a sphere of 21,247 points (radius γ_r r_β + 8h = 3.430 Bohr), built in 1.3 ms.
  **Flops:** 4(2l+1)·n_pts = 8.5e4 per orbital against ≈ 52·N = 1.2e7 for the stencil plus V: **0.7 % per s projector**.
  **Cost:** one H·ψ takes 1.64 ms local and 2.10 ms with V_NL, **22 % of one Hψ** on this CPU reference, because the
  Neumaier-compensated gather over scattered indices is latency-bound (~11 ns per sphere point) while the stencil streams
  (~7 ns per grid point). The GPU share is Task 8's measurement.
- Closed-shell Ne at h = 0.20 (61³, tol 1e-6 / eig 1e-7): pseudo **177 s** (20 SCF steps, 8.9 s/step, E_valence
  −35.934891) against all-electron σ = h **203 s** (12 steps, 16.9 s/step, E −65.938035); ratio 0.87: cheaper per step,
  more steps (review N4).
- Memory (round 1): one O solve at 61³ under `DegenerateShells` holds ≈ 170–200 MB (LOBPCG's 3k-block workspace with
  k = N_σ + witnesses, and the DIIS history); the floor test runs at most 4 solves at once (`MAX_PARALLEL_SOLVES`).
