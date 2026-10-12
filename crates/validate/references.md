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
