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
| **ω_sur/ω_MD(M4) − 1** | **−7.0e-6** (ω_sur = 1.474845e-2 a.u. = 3236.9 cm⁻¹) |
| model statics (bisection + 3-point k on the model, 0 teacher calls) | R_e^model − R_e^DFT = +1.9e-5 Bohr; **k_model/k_DFT − 1 = −2.2e-5**; E_mode 2.3869e-4 vs DFT 2.3885e-4 |
| decomposition | ω_sur/ω_static^model − 1 = −5.51e-4 (M4's DFT value −5.55e-4: the same Verlet +9.72e-4, truncation and anharmonic −1.52e-3 terms on the same curve) |
| speed | 1.00e5 steps/s (10.0 µs/step, N = 2) against the teacher's 102 ms/solve |

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
| speed | 0.9–1.1e5 steps/s (9–11 µs/step, N = 2; the pass ran concurrently with the other tests) |

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
`sim::surrogate_ks`). The rule the policy applies is **fallback by uncertainty**: an unseen
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
| H₂ (N = 2) | **0.9–1.1e5 steps/s** (9–11 µs/step) | 9.8 steps/s (102 ms/solve, 7.8 SCF steps, 338 readbacks) | ~1e4 |
| H₃⁺ (N = 3) | **6.4–7.6e4 steps/s** (13–16 µs/step) | 13.9 steps/s (72 ms/solve, k = 1) | ~5e3 |

The spec's expectation (≥ 10⁴ for H₂) is met by 10×; the surrogate's step is dominated by the
descriptor and the O(Q′²) predictive variance at `variance_cadence` = 1 (Task 9 tuned it, below).

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
starts as is (≈ 100 s estimated, 24–34 s actual); H₂O at 57³ (300 solves, the user's 12-minute
pass) is offered at its trust minimum, 210 solves, with "Start anyway". The O-atom reference scan
(open-shell, cold, 200 SCF iterations at 41–57³: S10) is not in the estimate — the live ETA shows
it once the pass runs. The measured per-solve rate a later pass ("Sample more") is estimated with
excludes the reference stage (the references are cached by then; Task 9 fix round 1).

### Policy constants of the estimated solver (M5 Tasks 4–9; for the final review)

Behavioural numbers of `sim::surrogate_ks` and the app worker — none per element, all in atomic
units. "Why" is the one-line justification; the module docs carry the derivations.

| constant | value | why |
|---|---|---|
| `FallbackPolicy::force_std_abs` / `force_std_rel` / `FORCE_STD_BOUND_MAX` | 2e-3 Ha/Bohr / 10 % / 1e-2 | the spec's force tier; relative for large forces, capped so a large force never buys an unbounded error |
| `energy_std_per_atom` | 1e-3 Ha | the spec's energy tier per atom; the trust rule's training-residual bound and the audit's verdict use the same number |
| `max_descriptor_distance` | 3.0 training spreads | the envelope (rule 5): what the variance cannot see (all body-2 features vanish as a pair separates) |
| `MIN_RESIDUAL_DOF` | 30 | both noise levels known to ≈ 25 % before the variance means anything (Task 3 review) |
| `variance_cadence` | 4 | Task 9 timing table above: 1.7–3.6× steps/s; a verdict reused ≤ 3 steps (≈ 0.13 Bohr for hot H); rejections never reused |
| `audit_every` | 2000 answered steps | ruling S7 (catch model error the evidence absorbed as noise); 2000 since S14 — now background, so the cadence costs no speed. **Only at paced speeds:** at speed 1000 the auditor is saturated (2000 steps ≈ 12 ms, one solve 130–270 ms), the effective cadence is one audit per teacher solve (≈ 2–4·10⁴ steps) and a failed audit arrives that late, whatever this value — 200 would change nothing there; at paced speeds it would triple the second teacher's GPU share |
| `BOX_AFFECTED_FACE_DENSITY` × (h / 0.25 Bohr)² (`box_affected_face_density`) | 2e-5 e/Bohr³ at h = 0.25 | **a calibration at h = 0.25 for hydrogen, h-scaled (ruling S17).** The reported face density is the outermost grid plane, one h inside the Dirichlet wall, where the confined orbital is pinched: ρ_face ≈ 4κ²h² ρ₀(wall). With ΔE ≈ π d ρ₀(wall) for a nucleus d from the wall (1-D wall against e^{−κx}, patch area π d/κ), ΔE_conf ≈ π d ρ_face / (4κ²h²) ≈ 50–80 Ha per e/Bohr³ for H₂ (κ ≈ 0.85, d = 2.6–4 Bohr) at h = 0.25 — measured 40–55 (41³ and 49³, `diag_confinement_vs_face_density`). So the tolerated density scales as h² (the same density is 4× the confinement at h = 0.125, so a quarter of it is tolerated there; direction corrected in Task 9 fix round 2); the 1/κ² dependence is not modelled (handoff: an energy-relative rule, ΔE_conf,est vs f·N·`energy_std_per_atom` with κ from the HOMO eigenvalue). 2e-5 at h = 0.25 ≈ 1e-3 Ha of confinement, half the H₂ audit threshold (N × 1e-3): acceptable, not tight (the review prefers 1e-5 = the solvers' warning level); 49³ hot-burst solves that warn at 1.0–1.4e-5 stay trainable. (Fix round 4's 1e-3 ≈ 5e-2 Ha let confinement through: Task 9 item 1.) Beyond it the teacher's surface depends on position and on orientation vs the box axes, so no invariant model can represent it: honest, not over-conservative |
| `QUARANTINE_FRACTION` | 0.1 × `max_descriptor_distance` (0.3 spreads) | **a geometric choice, not a measurement**: the fence covers the unrepresentable geometry and its immediate neighbourhood, not the region the model still fits (Task 8 fix round 3) |
| `FENCE_VOTES` / `FENCE_VOTE_LIMIT` | 3 / 3 per fenced sample | Task 9 fix round 1 (I3): three mutually consistent fresh answers that disagree with the model inside a fence are evidence, not noise; one is kept and the next refit decides (kept if reproduced, else quarantined and the region closed to votes — one sample per unrepresentable region, not a chain); a choice, not a measurement |
| `QUARANTINE_ROUNDS` / `QUARANTINE_MAX_FRACTION` | 5 / 10 % of the store | a model that cannot fit its data broadly (the misspecification test) must stay untrusted; ten per cent is "a few strays", not "the data" |
| `RECOVERY_DEGREE_CAP` | 6 | the documented bent-molecule setting (Task 6: D = 6 within 10 % of D = 10 on H₃⁺), tried once before quarantine |
| `THIN_FRACTION` / `THIN_EVERY` | 0.25 × `max_descriptor_distance` (0.75 spreads) / 10 | a fallback answer within 0.75 spreads of a stored one teaches little; one in ten is still kept so a parked molecule's region keeps learning |
| `STUCK_STEPS` | 50 consultations | ≈ 12 fs of a trusted model falling back at one place before the panel says "not answering here" (Task 9) |
| app `REFIT_INTERVAL` / `REFIT_SAMPLES` | 0.5 s / 25 samples | a fallback burst refits a few times a second, not every step; a refit is 5–26 ms at S = 1, ~1 s at S = 2 |
| app `PASS_TARGET_S` / `PASS_SOLVE_PER_COLD` | 120 s / 1.6 | the user's 12-minute pass with no warning (Task 9 item 4); per-solve cost measured above |

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
