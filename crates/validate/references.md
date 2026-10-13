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
(`thermostat_h2o_preset_known_limitation`, `#[ignore]`d; run it with
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
| release as a temperature, 2ΔE/(N_f k_B), N_f = 6 | ≈ 16 000 K (an upper bound: the energy flows through V as well as K); 33³ MD: T 518 / 1338 / 2376 / 3235 / 3818 K over the first five steps, back to ~300 K by step 20 under CSVR τ = 1 fs |
| warm-SCF stall, 57³ MD from the preset (τ = 10 fs, MB start) | **step 2**: "SCF did not converge after 200 iterations (Δρ = 8.5e-5)" — the same stall occurs in plain NVE from the preset, with or without thermal velocities and at tol 1e-6 as well (Δρ 6.7e-5 / 9.6e-5 / 7.6e-5; one tol-1e-6 case needed 148 SCF iterations at step 2 and failed at step 3), so it is the warm, density-extrapolated DIIS path on the preset's fast-changing density (0.34 Ha/Bohr forces; T passes 13 000 K and O–H 3.3 Bohr within 6 fs in NVE), not the thermostat or the tolerance; at dt 0.5 fs the same path converges every step but with 52–179 iterations on some, and at 33³ the 0.25 fs step converges — a DIIS stagnation at Δρ ≈ 1e-4 whose trigger is unresolved |
| warm-SCF stall, 33³ | step 110 at 172° (Δρ 6.0e-5); a cold GPU solve there converges in 13 steps (16 at tol 1e-6) and the CPU in 14; quenched relaxation stalls at step 39 (~150°) |

Reading: the Z = 8 nucleus smeared to σ = 0.25 Bohr is a very soft core, and the σ = h = 0.25
surface's water is a near-linear molecule with 2.34 Bohr bonds, 0.16 Ha (4.4 eV) below the
physical geometry it is loaded in — the same widening that takes H₂ from 1.45 to 1.79 Bohr. The
driver pauses with the last good state on the stall (Review Focus 3, exercised for real).
Consequences for the handoff: (1) the viewer's `h2o` preset is not an equilibrium of the surface
it is simulated on — pressing Play releases 0.16 Ha and the temperature readout spikes to
thousands of kelvin within a few steps while the molecule opens toward linear; (2) **warm SCF
robustness on such a path is a solver item** (a cold-restart fallback in `GpuKs::evaluate` on
non-convergence turns a paused run into a slow step; fractional occupations are the real fix
near level crossings); (3) a smaller σ (finer h) moves the H₂O minimum back toward the physical
geometry.

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
