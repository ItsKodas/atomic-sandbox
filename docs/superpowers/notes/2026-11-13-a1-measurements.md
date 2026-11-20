# A1: the remaining measurements at kT = 5 mHa (2026-11-14)

Task 3 of `docs/superpowers/plans/2026-11-13-fermi-smearing.md`, measured on the uncommitted prototype
(`qm::scf::diag_smear`, the `SCF_SMEAR` knob, the `SCF_TRACE2` instrumentation and the harnesses in
`crates/providers/src/scf_stall_diag.rs`) before any production code exists. The prototype is the V0 note's (§1.1:
FD in the overflow-free form, μ from the two bisections, both witness arms, S from x); it now lives on the side
branch `smearing-prototype` (never merged). Everything here is GPU (RX 7900 XTX, release, BelowNormal, one job at a
time with 3-minute gaps), so the wall times are indicative: the GPU was shared with other agents' jobs and the
user's session.

Spec: `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md` (kT = 5 mHa, the section 5.1 partial-level stop
of 1e-5). Prior measurements: `2026-11-13-v0-open-shell-stability.md` (V0) and `2026-11-13-scf-stall-diagnosis.md`
(D).

## Summary

- **Step 0, the V2 pins at 5 mHa: every case converges** (no stall stop, no cap), in 11–16 iterations except
  all-electron H₂O + 2 H (33; 434 s of GPU). New budgets ≈ 1.5× and the F/TS pins are in section 0 and in the
  spec's V2 table (amended). Three default-policy cases that were gapped at 1e-3 now carry a partial level
  (seeds 1, 3, 8). The H₂ and H₂O controls are unchanged to every printed digit.
- **Step 1, the V3 aspect: H₄ 1.8 × 1.95 Bohr, atom 0 along x** (41³ × 0.30, σ = h, all-electron): c = 0.052
  Ha/Bohr, d(TS)/dx = −2.26e-2 Ha/Bohr, the plain |FD(δ) − FD(2δ)| = 2.9e-5, all three in range over the whole 4δ
  span. Task 6 confirms it on the CPU.
- **Step 2, the V5 NVE case: N₃ fails, CO₂⁺ qualifies.** N₃ is linear ²Π_g (TS = 2 kT ln 2 to 1e-9) and its bend is
  restoring, but its π split never reaches 10 kT on one state (39 mHa at 60°, then a state change). CO₂⁺ reaches
  47.3 mHa (9.5 kT) at a 70° bend on a smooth, restoring surface, with h²/4·E_mode = 1.4e-5 Ha: CO₂⁺, 71³ × 0.20,
  β = 70° at R_e = 2.226, dt = 0.25 fs, ≥ 2 bend periods (≈ 600 steps). The spec's NVE text now names CO₂⁺
  (amended in review fix round 1).
- **Step 3, the witness cost:** all-electron H₂ / H₃⁺ at 49³ cost 3.1× / 2.2× cold and 1.4× / 1.75× warm (one extra
  vector per channel); pseudised H₂O costs 1.76× / 1.29× cold and 1.31× / 1.09× warm at 57³ × 0.20 / 81³ × 0.15
  (two extra vectors per channel instead of one).
- **Step 4, E‴:** max (1/12)|E‴| = 0.40 (O–H in H₂O) and 0.35 (N–H in NH₃) over 0.8–1.0 R_e, below κ_max = 1. No
  stop.

## 0. The V2 pins at kT = 5 mHa (spec §9 "V2 cases", amended)

**Method.** The diagnosis harness `scf_stall_diag` unchanged (D§2): every scene through `providers::ks::problem`,
cold, 81³ × 0.15, the app's interactive tolerance 1e-5 raised by `effective_gpu_tol` (which leaves 1e-5 for every
case here: N ≤ 26 valence electrons), the stall stop at its default W = 50, the cap at 200, `SCF_SMEAR=5e-3`.
- Default policy (`Accepted`: pseudo-H, pseudo-O): the cases' own `DegenerateShells`, so the prototype's FD runs in
  phase 1.
- `ionic none`: `DIAG_DS=1` (the witness vectors the FD needs; spec §4.2).
- **The section 5.1 stop.** Every V2 case's tolerance is already 1e-5, so the partial-level rule
  max(tol_eff, 1e-5) is the same number on the cases that keep a partial level and on the gapped ones; no separate
  `DIAG_TOL` run was needed.
- TS = kT·Σ s(f) from the converged occupations; F = E − TS.
- A cold GPU solve is bit-reproducible (spec §5.1), so one run per case.

**Result: every case converges, none by the stall stop or the cap.**

| policy | case | 1e-3 (D§6): iterations, F, TS | **5 mHa: iterations** | **5 mHa: E (Ha)** | **5 mHa: F (Ha)** | **5 mHa: TS (Ha)** | new budget | wall (s) |
|---|---|---|---|---|---|---|---|---|
| default | H₄ rhombus | 10, −2.079319, 0 | **12** | −2.079295 | **−2.079322** | **0.000027** | ≤ 20 | 21.5 |
| default | H₄ lopsided tetra | 32, −1.932837, 2.7e-3 | **16** | −1.929941 | **−1.943830** | **0.013889** | ≤ 25 | 15.4 |
| default | H₄ seed 1 | 12, −2.078114, 1e-9 | **13** | −2.076526 | **−2.078602** | **0.002076** | ≤ 20 | 8.3 |
| default | H₅ seed 3 | 18, −2.563330, 2e-7 | **15** | −2.561243 | **−2.564249** | **0.003006** | ≤ 25 | 17.6 |
| default | H₈ seed 8 | 14, −4.220373, 2e-16 | **14** | −4.220231 | **−4.220394** | **0.000163** | ≤ 25 | 49.6 |
| `ionic none` | H₅ ring | 13, −2.465136, 1.39e-3 | **11** | −2.463750 | **−2.470682** | **0.006932** | ≤ 20 | 13.7 |
| `ionic none` | H₆ seed 5 | 47, −2.967311, 6.8e-4 | **15** | −2.963540 | **−2.973255** | **0.009715** | ≤ 25 | 24.2 |
| `ionic none` | H₂O + 2 H | 79, −47.339721, 2.5e-3 | **33** | −47.336856 | **−47.350019** | **0.013163** | ≤ 50 | **433.5** |
| `ionic none` | lopsided tetra | 28, −1.813947, 2.7e-3 | **15** | −1.810974 | **−1.824952** | **0.013978** | ≤ 25 | 38.3 |
| default | H₂ control | 9, E −1.136119 | 9 | −1.136119 | −1.136119 | 0 | (control) | 1.3 |
| default | H₂O control | 13, E −17.476023 | 13 | −17.476023 | −17.476023 | 1.4e-12 | (control) | 24.1 |

- **Budgets.** The spec's rule, unchanged: about 1.5× the measured count, here ⌈1.5 n⌉ rounded up to a multiple of 5.
- **F/TS pins.** The 5 mHa values above, within 1e-5 Ha (the spec's rule). The harness prints six decimals, so the
  pins carry a ±5e-7 rounding inside that tolerance; TS is written as E − F of those printed values (±1e-6; review
  fix round 1: two significant figures were coarser than the 1e-5 tolerance).
- **The controls are unchanged** from D§6 to every printed digit (H₂ −1.136119, H₂O −17.476023; TS 0 and 1.4e-12):
  a gapped solve does not see kT.
- **What moved, and why.**
  - Every partly occupied case converges in fewer or the same iterations at 5 mHa (the slope 1/(4kT) is 5× smaller,
    so the occupation feedback is softer); the rhombus and seed 1 take one or two more.
  - TS moves non-linearly where a level is pinned near μ. The exactly degenerate H₅ ring is the closed form
    2 kT ln 2 = 6.931e-3 (E unchanged, −2.463750 at both kT). The lopsided tetrahedra carry a pinned pair in both
    channels: 0.013889 and 0.013978. Both **exceed 4 kT ln 2 = 0.013863**, the most one pinned pair per channel
    can give (2 ln 2 of entropy per channel at f = ½), so a third level lies near μ (a second splitting t₂ within a
    few kT) and carries part of TS. Not resolved further here (the harness prints the totals only).
  - **Three default-policy cases that were gapped at 1e-3 now carry a partial level:** seed 1 (TS 1e-9 → 2.08e-3),
    seed 3 (2e-7 → 3.01e-3) and seed 8 (2e-16 → 1.63e-4). Their frontier gaps are a few kT at 5 mHa, so they are
    pinned states under the 5 mHa rule, and their E is 0.1–2.1 mHa above the 1e-3 E. These are the cases the
    section 5.1 stop governs.
- **Cost.** All-electron H₂O + 2 H converged in 33 iterations (79 at 1e-3) but took **434 s** of GPU time, by far
  the slowest case: 13 s per iteration against 0.6–2.6 s for the other cases. V2 runs it once per
  suite run; section 3 has the per-solve cost of the witness.

## 1. V3: the H₄ aspect (spec §9 "V3 force oracle")

**Method** (`a1_measure`, `A1_MODE=v3`). All-electron H₄ rectangle a (along x) × b (along y) in the xy plane,
centred on its centroid, 41³ × 0.30 (12 Bohr box), σ = h, `DegenerateShells` + FD at 5 mHa (`DIAG_DS=1`
equivalent), auto multiplicity (n↑ = n↓ = 2), tol 1e-6, GPU. Atom 0, the corner at (−a/2, −b/2, 0), is displaced
along x by x ∈ {0, ±δ/2, ±δ, ±2δ}, δ = 0.01 Bohr: the centre cold, the rest warm in the order 0, +δ/2, +δ, +2δ,
0 (again), −δ/2, −δ, −2δ; the second centre solve is the one used, the first gives the cold/warm path scatter.
Δ is the frontier splitting (rank N_σ − 1 → N_σ, averaged over the two channels, which agree to ≤ 2e-6 Ha); c = dΔ/dx
and d(TS)/dx are central differences; HF is the GPU Hellmann–Feynman force on atom 0 along x at the centre.

**The TS slope.** Each channel holds one electron in the frontier pair, f = 1/(1 + e^{∓2u}) with u = Δ/(4kT), and
S_σ = 2[ln(2 cosh u) − u tanh u]. So d(TS)/dΔ = −u sech²u for the two channels together, whose maximum is 0.448 at
u = 0.772 (Δ = 15.4 mHa): the spec's |d(TS)/dx| ≤ 0.45·c. The measured slopes follow it to 3 digits
(u = 0.202, 0.413, 0.647: u sech²u = 0.194, 0.351, 0.440 against the measured |d(TS)/dx|/c = 0.194, 0.350, 0.437).

**Quick scan** (centre and ±δ):

| a × b | Δ (mHa) | u = Δ/4kT | c = dΔ/dx (Ha/Bohr) | d(TS)/dx (Ha/Bohr) | HF + dE/dx (Ha/Bohr) | note |
|---|---|---|---|---|---|---|
| 1.8 × 1.82 | 1.61 | 0.08 | — | — | — | the −δ warm solve stopped at Δρ 1.8e-6 > 1e-6 (the f32 floor of a two-channel pinned pair, V0 note D) |
| 1.8 × 1.85 | 4.04 | 0.20 | 0.0414 | −8.04e-3 | −8.04e-3 | |
| 1.8 × 1.90 | 8.26 | 0.41 | 0.0452 | −1.58e-2 | −1.58e-2 | |
| 2.2 × 2.22 | 1.45 | 0.07 | 0.0362 | −2.62e-3 | −2.63e-3 | |
| 2.2 × 2.26 | 4.36 | 0.22 | — | — | — | the warm centre re-solve stopped at Δρ 3.2e-6 |
| 2.6 × 2.62, 2.6 × 2.66 | 1.30, 3.90 | 0.06, 0.20 | — | — | — | the warm centre re-solve fails at Δρ ≈ 1e-3 (best 2.5e-5, 1.2e-4): the restricted state is not stable at this stretch (the diradical regime of D§6) |

c hardly depends on the aspect at a given a (it is set by the side's hopping slope: 0.036 at a = 2.2, 0.041–0.052
at a = 1.8), and it sits inside the window [5e-3, 1e-1] everywhere; b − a sets Δ and so u.

**Full stencil** (7 points; δ = 0.01):

| quantity | 1.8 × 1.85 | **1.8 × 1.95** | requirement |
|---|---|---|---|
| Δ at x = 0 (mHa); u | 4.04; 0.20 | **12.93; 0.65** | a few kT |
| c over the 4δ span, per sub-interval (Ha/Bohr) | 0.0393 … 0.0431 | **0.0498 … 0.0540** | ∈ [5e-3, 1e-1] |
| \|d(TS)/dx\| over the 4δ span (Ha/Bohr) | 6.6e-3 … 9.5e-3 | **2.14e-2 … 2.39e-2** | ≥ 2e-3 |
| plain \|FD(δ) − FD(2δ)\| (Ha/Bohr) | 2.47e-5 | **2.88e-5** | ≤ 1e-3 |
| implied F‴ = 2\|FD(δ) − FD(2δ)\|/δ² (Ha/Bohr³) | 0.49 | 0.58 | (see the F‴ note below) |
| HF (Ha/Bohr) | −8.18269e-2 | −4.63969e-2 | |
| R(δ/2, δ), R(δ, 2δ) (Ha/Bohr) | −8.18297e-2, −8.18183e-2 | −4.63777e-2, −4.64004e-2 | (CPU test: agree within 2.5e-5) |
| \|R(δ/2, δ) − R(δ, 2δ)\| | 1.1e-5 | 2.3e-5 | ≤ 2.5e-5 on the CPU |
| HF − R_F(δ/2, δ) | +2.8e-6 | −1.9e-5 | ≤ 1e-4 |
| HF − R_E(δ/2, δ) | −8.0e-3 | **−2.27e-2** | ≥ 5e-4 |
| (FD(2δ) − HF)/(FD(δ) − HF) | not asserted (\|FD(δ) − HF\| = 3.8e-7 < 1e-5) | 3.18 (\|FD(δ) − HF\| = 1.3e-5) | ∈ [3, 5] when \|FD(δ) − HF\| ≥ 1e-5 |
| cold − warm F at the centre | 9.4e-8 | −1.3e-7 | (the GPU path scatter) |

1.8 × 1.90 ran the same stencil but its +δ warm solve stopped at Δρ 2.1e-6 (tol 1e-6): the f32 floor again, not a
property of the aspect (its quick scan converged at the same point).

**F‴ (corrected in review fix rounds 1 and 2).** The spec's earlier window estimate 0.2c³/(kT)² is **1.12** Ha/Bohr³
at c = 0.052 (this note first misstated it as 0.57). That estimate used the wrong functional: −kT ln(2 cosh(Δ/2kT))
per channel is the canonical two-state free energy, not the FD (grand-canonical, μ at mid-gap) one, so it is not
an FD upper bound but a different function that happens to be 4× larger. The pair's FD free energy, the same F whose temperature derivative reproduces the measured d(TS)/dx to
3 digits above, is −2kT ln(2 cosh u) per channel with u = Δ/(4kT), whose third derivative in x is
sech²u·tanh u·c³/(8(kT)²) for the two channels (maximum coefficient 0.048, a quarter of 0.2). At u = 0.65 and
c = 0.052 that is **0.27** Ha/Bohr³ (checked numerically against a finite difference of the closed form), and 0.07
for 1.8 × 1.85. The measured 0.58 and 0.49 therefore also hold the bonds' own anharmonicity along x (≈ 0.3–0.4 on
both aspects), and the GPU plain-FD differences (2.5–2.9e-5) carry ~1e-5 of f32 noise, so the split is not
resolved here. The measured total, 0.58, is what sets the δ-scaling truncation e = F‴δ²/6 ≈ 1e-5 (spec §9 V3,
pre-registered in fix round 1). The re-review agreed (0.27); the spec now states F‴ ≤ 0.048·c³/(kT)², and the pre-registered window
[5e-3, 1e-1] Ha/Bohr is unchanged.

**Candidate (for Task 6 to confirm on the CPU at tol 1e-9): a × b = 1.8 × 1.95 Bohr, atom 0 at (−0.9, −0.975, 0),
displaced along x** (toward atom 1, which shortens the 1.8 side and widens Δ; c > 0). Centred coordinates:
(−0.9, −0.975, 0), (0.9, −0.975, 0), (0.9, 0.975, 0), (−0.9, 0.975, 0). On the GPU prototype:
c = 0.0518 Ha/Bohr, d(TS)/dx = −2.26e-2 Ha/Bohr (11× the 2e-3 floor; 0.437c, near the 0.448c maximum), the plain
FD difference 2.9e-5 (35× under 1e-3), and HF − R_E = −2.27e-2 (45× the 5e-4 the test needs to tell F's derivative
from E's).
- The GPU Richardson numbers are at the f32 floor: a path scatter of ~1e-7 Ha in F becomes ~1e-5 Ha/Bohr in FD(δ/2),
  so |R(δ/2, δ) − R(δ, 2δ)| = 2.3e-5 and the ratio 3.18 are noise-limited here, not truncation. The CPU f64 test
  (Task 6) is the one that certifies them.
- For Task 11's work integral (0.2 Bohr along the same axis, x ∈ [−0.1, +0.1]), Δ runs from about 7.7 to 18.1 mHa
  (u 0.39–0.91) at this c, which keeps |d(TS)/dx| between ≈ 0.35c and 0.45c, so ΔTS ≈ 4 mHa over the path
  (estimate from c, not measured).

## 2. V5: the NVE case (spec §9 "V5 new tests" 2)

**Method** (`a1_measure`, `A1_MODE=n3`). Symmetric linear triatomic X–Y–X in the xy plane, centre atom at the
origin (a grid node), ends at (∓R cos(β/2), −R sin(β/2), 0), so β is the deviation from linear (bond angle
180° − β) and the bend keeps C₂ᵥ. `Accepted` policy (pseudised throughout), auto multiplicity, FD at 5 mHa, tol 1e-5
(the section 5.1 stop for a partial level), GPU, 71³ × 0.20 (14 Bohr box; h_req of N and O is 0.20). Linear scan
cold then warm; the bends warm in increasing β. Reported per solve: F, TS, the frontier levels ε(f) of each channel,
the face density, and the generalised force −dF/dβ = Σ_I F_I·∂R_I/∂β from the HF forces.

### 2.1 N₃ (all pseudo-N)

**Linear: R_e, the state, and the box.**

| R (Bohr) | iterations | F (Ha) | TS (Ha) | −dF/dR, symmetric stretch (Ha/Bohr) | face density |
|---|---|---|---|---|---|
| 2.18 | 15 (cold) | −30.36325325 | 6.931472e-3 | +9.243e-2 | 4.3e-6 |
| 2.22 | 20 | −30.36519516 | 6.931472e-3 | +6.84e-3 | 4.6e-6 |
| 2.224 | 15 (cold) | −30.36520552 | 6.931472e-3 | (β = 0: −dF/dβ = −1.2e-7) | 4.7e-6 |
| 2.24 | 13 | −30.36494407 | 6.931472e-3 | | 4.8e-6 |
| 2.26 | 26 | −30.36396190 | 6.931472e-3 | −6.66e-2 | 5.0e-6 |
| 2.30 | 35 | −30.36001102 | 6.931472e-3 | −1.293e-1 | 5.3e-6 |

- **R_e = 2.224 Bohr** (1.177 Å; the force interpolates to zero at 2.2237; experiment 1.181 Å). The symmetric-stretch
  constant k_R = 1.84 Ha/Bohr² gives ω_s = √(k_R/2m_N) = 6.0e-3 a.u. (1318 cm⁻¹; experiment ν₁ ≈ 1320).
- **²Π_g with the hole in the ↓ π_g pair.** At linear the ↓ channel (7 electrons) holds one electron in a degenerate
  pair at −0.260639 Ha, f = 0.5000 / 0.5000; the ↑ channel (8 electrons) has the same π_g pair full at −0.293538
  (× 2) and a gap of 0.229 Ha to the π_u* pair. **TS = 6.931472e-3 Ha = 2 kT ln 2 (6.931472e-3) at every R**, within
  1e-9 Ha (the requirement: 1e-5). The g label is from the level order (the half-filled pair is the HOMO pair, above
  the σ levels and the ↓ π_u pair), not from a parity check of the orbitals.
- **Linear is a minimum: the bend force is restoring.** −dF/dβ = −1.11e-2, −2.23e-2 and −4.59e-2 Ha/rad at β = 5°,
  10° and 20° (R = 2.24), linear in β with k_β = 0.127–0.132 Ha/rad², matching the energies (F(β) − F(0) = 0.484,
  1.940, 7.874 mHa at 5°, 10°, 20°: k_β = 0.127–0.129). With E = ½k_β β², the C₂ᵥ bend frequency of a symmetric
  X₃ is ω_b² = 6k_β/(m R²) (m the atom mass), so **ω_b = 2.48e-3 a.u. (543 cm⁻¹**; experiment ν₂ ≈ 457) and
  **h_b = ω_b·dt = 0.0256** at the suites' dt = 0.25 fs = 10.335 a.u.
- **The box.** Face density 1.6–5.3e-6 over every N₃ solve (linear and bent), against
  `box_affected_face_density(0.20)` = 1.28e-5: 71³ × 0.20 holds it with a factor ≥ 2.4.

**The bend: the π split, TS and the bend energy** (R = 2.224; 5°–20° at R = 2.24):

| β | ↓ π pair ε (f) | ↓ split (mHa) | ↑ frontier (f) | TS (mHa) | F − F(0) (mHa) | −dF/dβ (Ha/rad) | iterations | face |
|---|---|---|---|---|---|---|---|---|
| 0° | −0.260639 (0.5000), −0.260639 (0.5000) | 0 | π_g full, gap 229 mHa | 6.931 | 0 | −1.2e-7 | 15 | 4.7e-6 |
| 5° | −0.260509 (0.5060), −0.260268 (0.4940) | 0.24 | gapped | 6.931 | 0.48 | −1.11e-2 | 17 | 4.8e-6 |
| 10° | −0.260212 (0.5239), −0.259254 (0.4761) | 0.96 | gapped | 6.920 | 1.94 | −2.23e-2 | 16 | 4.7e-6 |
| 20° | −0.258974 (0.5960), −0.255086 (0.4040) | 3.89 | gapped (ε_LUMO − ε_HOMO 185 mHa) | 6.746 | 7.87 | −4.59e-2 | 18 | 4.6e-6 |
| 30° | −0.256668 (0.7133), −0.247553 (0.2867) | 9.12 | gapped (154 mHa) | 5.992 | 18.77 | −7.48e-2 | 19 | 4.2e-6 |
| 45° | −0.250791 (0.8921), −0.229667 (0.1079) | 21.1 | gapped (101 mHa) | 3.426 | 44.92 | −1.288e-1 | 16 | 3.6e-6 |
| 60° | −0.241978 (0.9807), −0.202551 (0.0187); next −0.185238 (0.0006) | 39.4 | **−0.257116 (0.9914), −0.211777 (0.0132): 45 mHa** | 1.712 | 89.38 | −2.176e-1 | 18 | 2.9e-6 |
| 75° | −0.244406 (0.9544), −0.213902 (0.0448) | **30.5** | **−0.253208 (0.9695), −0.221089 (0.0490): 32 mHa** | 3.996 | 131.07 | **−2.45e-2** | 19 | 1.6e-6 |

- The ↓ split grows as β² (s/β² = 0.032–0.033 Ha/rad² from 5° to 45°), so the spec's "about 10 kT" (50 mHa at
  5 mHa) would need β ≈ 70° on that law. **It is never reached.** The split peaks near 60° at 39 mHa (7.9 kT); by then
  the ↑ channel's in-plane π_u* component has come down to 45 mHa (9 kT) above its HOMO, so the ↑ channel carries
  part of TS and the state is no longer "one π pair". Between 60° and 75° the electronic structure changes: the ↓
  split falls back to 30.5 mHa, TS rises again to 4.0 mHa, and the restoring force collapses from 0.218 to
  0.024 Ha/rad while F still rises by 41.7 mHa (the trapezoid of the two forces gives 31.7 mHa: the 10 mHa mismatch
  says the path between them is not one smooth surface at this spacing). A start beyond ≈ 60° is outside one
  smooth state.
- **h²/4·E_mode is not the limit.** At 60° the bend holds E_mode = 89.4 mHa: h_b²/4·E_mode = 1.6e-4 × 0.0894 =
  1.5e-5 Ha. Even if all of it went into the symmetric stretch (the only other C₂ᵥ-symmetric mode a symmetric start
  at rest can feed; h_s = ω_s·dt = 0.062), h_s²/4·E = 9.6e-4 × 0.0894 = 8.6e-5 Ha, under 1e-4. At 45°: 7.4e-6 and
  4.3e-5. So every bend up to 60° meets the energy condition; what fails is the split.

**N₃ does not qualify** as the spec words the case: the 10 kT split is not reached on one smooth state. Per the
brief, the substitute the review suggested was measured next.

### 2.2 CO₂⁺ (pseudo-C centre, pseudo-O ends, charge +1; 15 valence electrons, isovalent with N₃)

**Linear.** −dF/dR = +9.76e-2 (R = 2.18), −1.76e-3 (2.226), −2.88e-2 (2.24) Ha/Bohr: **R_e = 2.225 Bohr** (1.178 Å;
experiment 1.177 Å), k_R = 2.11 Ha/Bohr², ω_s = √(k_R/2m_O) = 6.01e-3 a.u. (1319 cm⁻¹; experiment ν₁ ≈ 1244).
**²Π_g**: the ↓ π_g pair (−0.690458 Ha, f = 0.5000 / 0.5000) holds the hole, the ↑ channel is gapped by 335 mHa,
and **TS = 6.931472e-3 Ha = 2 kT ln 2 within 1e-9** at every R. Face density 3.5e-7–1.1e-6 over every CO₂⁺ solve
(rule 1.28e-5: a factor ≥ 11).

**The bend** (R = 2.226 except where marked; tol 1e-5, warm in increasing β):

| β | ↓ π pair ε (f) | ↓ split (mHa) | next ↓ level above the pair (mHa) | ↑ HOMO–LUMO (mHa) | TS (mHa) | F − F(0) (mHa) | −dF/dβ (Ha/rad) | −dF/dR (Ha/Bohr) | iterations, s |
|---|---|---|---|---|---|---|---|---|---|
| 0° | −0.690458 (0.5000) ×2 | 0 | 311 | 335 | 6.931 | 0 | −5.0e-7 | −1.8e-3 | 17, 18 |
| 10° | −0.690041 (0.5178), −0.689329 (0.4822) | 0.71 | 310 | 316 | 6.925 | 1.636 | −1.872e-2 | +8.0e-4 | 17, 14 |
| 20° (R 2.21) | −0.689957 (0.5740), −0.686976 (0.4260) | 2.98 | 260 | 286 | 6.822 | 6.75 † | −3.90e-2 | | 19, 17 |
| 40° (R 2.21) | −0.684065 (0.7828), −0.671245 (0.2172) | 12.8 | 181 | 209 | 5.234 | 27.86 † | −8.36e-2 | | 18, 26 |
| 60° | −0.671687 (0.9599), −0.639932 (0.0401) | 31.8 | 90.6 | 133 | 1.683 | 66.63 | −1.529e-1 | +1.000e-1 | 19, 36 |
| **70°** | **−0.662983 (0.9912), −0.615731 (0.0088)** | **47.3** | **39.6** | **94.9** | **0.514** | **98.75** | **−2.207e-1** | **+1.560e-1** | 17, 51 |
| 75° | −0.657559 (0.9966), −0.600241 (0.0031) | 57.3 | **10.6** | 67.8 | 0.329 | 120.1 | −2.696e-1 | +1.945e-1 | 17, 25 |
| 80° | −0.652144 (0.9927), −0.602902 (0.0071) | 49.2 | (a level crosses in) | **37.2, ↑ partial** | 1.568 | 146.1 | −3.236e-1 | +2.414e-1 | 17, 32 |

† relative to F(0) at R = 2.21 (−37.98543967).

- **Restoring at every angle**, linear in β at small β: k_β = 0.1073 Ha/rad² from the force at 10° and 0.1074 from
  the energy. The surface is smooth up to 70°: the trapezoid of the forces at 60° and 70° gives 32.6 mHa against
  the measured F(70°) − F(60°) = 32.1 mHa (0 → 10°: 1.63 against 1.636). Beyond 70° the next ↓ level comes within
  2 kT of the pair at 75°, and at 80° the ↑ channel becomes partial and TS rises again (a state change, as N₃ at
  75°).
- The split grows a little faster than β² (s/β² = 0.023, 0.025, 0.026, 0.029, 0.032 Ha/rad² at 10°, 20°, 40°, 60°,
  70°) and **reaches
  47.3 mHa = 9.5 kT at 70°**, with the ↑ channel gapped at 19 kT and the next ↓ level 63 mHa = 12.7 kT above μ.
- **h²/4·E_mode.** For a symmetric X–Y–X bend (end mass m, centre mass M, bond R, E = ½k_β β²),
  ω_b² = 2k_β(1 + 2m/M)/(mR²). With m_O = 29156.95 mₑ, m_C = 21874.66 mₑ, R = 2.226: **ω_b = 2.335e-3 a.u.
  (512 cm⁻¹; experiment ν₂ ≈ 511)**, h_b = ω_b·dt = **0.0241** at dt = 0.25 fs = 10.335 a.u., h_b²/4 = 1.46e-4.
  E_mode = F(70°) − F(0) = 98.75 mHa, so **h_b²/4·E_mode = 1.4e-5 Ha** (≤ 1e-4). The symmetric stretch is the only
  other mode a C₂ᵥ start at rest feeds (h_s = 0.0621, h_s²/4 = 9.65e-4): the start's stretch force +0.156 Ha/Bohr puts
  G²/(2k_R) = 5.8 mHa in it (h_s²/4·E = 5.6e-6), and even the whole 98.75 mHa there would give 9.5e-5, still ≤ 1e-4.
- **TS swing**: 0.514 mHa at the start, 6.931 mHa at each linear crossing: max TS − min TS = 6.4 mHa (the test needs
  ≥ 1 mHa).

### 2.3 The V5 NVE case (recommended; Task 11 pre-registers the bounds and runs it)

- **Molecule:** CO₂⁺ (C centre, two O; charge +1; auto multiplicity = doublet), `Accepted` (pseudised C and O).
  **N₃ is replaced** (section 2.1: its π split never reaches 10 kT on one state).
- **Box:** 71³ × 0.20 (14 Bohr; h_req of O). Face density ≤ 1.1e-6 over 0°–80°, ≤ 1/11 of
  `box_affected_face_density(0.20)` = 1.28e-5.
- **Start:** C at a grid node (the origin), O at (∓R_e cos 35°, −R_e sin 35°, 0) with R_e = 2.226 Bohr, i.e. bond angle
  110° (**β = 70°**), at rest. ↓ π split 47.3 mHa (9.5 kT), TS 0.514 mHa.
- **dt = 0.25 fs** (the suites' `DT_FS`, 10.335 a.u.); h_b = 0.0241; **h_b²/4·E_mode = 1.4e-5 Ha** (E_mode = 98.75 mHa).
- **Steps: ≥ 2 bend periods, ≈ 600** (review fix round 1; the first draft said 300). The harmonic period is
  2π/ω_b = 2691 a.u. = 260 steps. The large amplitude **stiffens** the bend: F(70°) − F(0) = 98.75 mHa against
  ½k_ββ² = 80 mHa, so the period is shorter than harmonic and 600 steps hold more than two. 300 steps = 0.075 ps
  cannot resolve a drift at `DRIFT_BOUND` = 2e-4 Ha/ps against the Verlet swing. **Not measured:** the warm
  per-step cost at 71³; the bend scan's warm solves took 14–51 s per 10° jump, and an MD step moves the bend by at
  most h_b·β₀ ≈ 1.7° (at the linear crossing), so a step is cheaper, but 600 steps may exceed one 15-min GPU batch.
  Task 11 times the first steps and measures σ_path under the 1e-5 stop before fixing the count and the
  fluctuation bound 3 × (h²/4·E_mode + 2σ_path), splitting the run into checkpointed batches if needed.
- The trajectory stays C₂ᵥ by symmetry (up to f32 noise), so the asymmetric stretch is not fed.
- Not checked here: a run of the trajectory itself (Task 11), and whether 70° at R_e or the bent geometry's own R
  (the stretch force +0.156 Ha/Bohr) is the better start; at R_e the stretch carries 5.8 mHa, which is harmless for
  h_s²/4·E.

**The spec's NVE text is amended** (review fix round 1, §9 "V5 new tests" 2): the case is CO₂⁺ with the numbers
above, and the premise "the bend is soft (h ≈ 0.02) with a few mHa in it, so h²/4·E_mode ≲ 1e-6 Ha" is gone
(h_b = 0.024 with ≈ 0.1 Ha in it, h²/4·E_mode = 1.4e-5 Ha). This is the substitution the brief allows ("If N₃ fails,
substitute another linear ²Π radical (the review suggests CO₂⁺) with the same properties"), so it is not a STOP;
the reviewer should confirm it.

## 3. The witness cost (spec §4.2 "Cost")

**Method** (`a1_measure`, `A1_MODE=cost`). One GPU solver per grid; per arm one discarded cold warm-up solve, then 5
cold solves of the same geometry, then 5 warm solves with nucleus 0 moved +0.005 Bohr along x and back alternately (an
MD-sized step). Wall time of `solve_observed` only (no forces), median of 5. Arms: **aufbau** (`ionic none` as today:
N_σ vectors) against **witness** (`DegenerateShells` + FD at 5 mHa, so both witness arms run: the configuration
production will run on every all-electron scene); for pseudised H₂O, **DS** (today's `Accepted` path) against **FD**
(5 mHa). The GPU was otherwise idle (the PC had just restarted), so these times are cleaner than the others here.

| case | grid | tol | arm | cold: median (ms), iterations | warm: median (ms), iterations | vectors per channel (N_σ + extra) | E (Ha) |
|---|---|---|---|---|---|---|---|
| H₂ (R = 1.4, all-electron, σ = h) | 49³ × 0.25 | 1e-6 | aufbau | 230, 10 | 222, 10–11 | 1 + 0 | −0.96271551 |
| | | | witness | **708**, 10 | **308**, 7–8 | 1 + **1** | −0.96271550 |
| H₃⁺ (side 1.9, all-electron) | 49³ × 0.25 | 1e-6 | aufbau | 248, 11 | 188, 6 | 1 + 0 | −1.11755146 |
| | | | witness | **550**, 10 | **329**, 6 | 1 + **1** | −1.11755141 |
| H₂O (`h2o` preset, pseudised) | 57³ × **0.20** | 1e-5 | DS | 3082, 13 | 1713, 8–13 | 4 + 1 | −17.47681155 |
| | | | FD | **5437**, 13 | **2252**, 8–9 | 4 + **2** | −17.47681007 |
| H₂O | 81³ × 0.15 | 1e-5 | DS | 5383, 13 | 2594, 8–9 | 4 + 1 | −17.47609498 |
| | | | FD | **6924**, 13 | **2817**, 8–9 | 4 + **2** | −17.47609551 |

- **All-electron: the witness costs 2.2–3.1× per cold solve and 1.4–1.75× per warm solve** at 49³ (H₂ 3.1× / 1.39×,
  H₃⁺ 2.2× / 1.75×). It is one extra vector per channel: the 36 kT arm (μ + 0.18 Ha) is met by the first unoccupied
  level (H₂: +0.039 Ha with μ mid-gap at −0.149; H₃⁺: −0.349 with μ at −0.590). E is unchanged within 5e-8 Ha (a
  gapped solve; TS ≤ 4e-17).
- **Pseudised H₂O: FD costs 1.76× (57³) and 1.29× (81³) per cold solve, 1.31× and 1.09× warm.** At 5 mHa the second
  arm binds in every channel (spec §4.2): FD needs 2 extra vectors per channel where DS's 0.1 Ha arm needed 1. E agrees
  within 1.5e-6 Ha (the 1e-5 stop's scatter).
- **The grid "57³" of the brief at h = 0.25 cannot be run pseudised**: `problem` refuses pseudo-O at h = 0.25 > h_req
  0.20 ("choose a finer grid"). 57³ × 0.20 (an 11.2 Bohr box) keeps the brief's point count, which sets the cost.
- For comparison, V0 (the 2026-11-14 section C) measured the 3p atoms' cold solves at 5–80× DS in a 14 Bohr box,
  where the arm needs 7–15 extra vectors, and the 2p atoms and molecules at 1–5×.
- The slow V2 case of section 0 (all-electron H₂O + 2 H, 434 s for 33 iterations) is not explained by these numbers;
  its per-solve cost was not broken down (not measured: the extra vectors it grew).

## 4. E‴ of the stiffest bonds (spec §6.3, κ_max = 1 Ha/Bohr³)

**Method** (`a1_measure`, `A1_MODE=e3`). Default pseudo surface (`Accepted`: pseudised O, N and H), 81³ × 0.15,
tol 1e-6 (gapped: TS ≤ 1e-11), GPU. R_e from one secant step on the bond force with all bonds equal (H₂O at
104.5°, NH₃ at 106.7°); then one bond (the others at R_e, the angles fixed) at 7 points R = R_e·(0.8 + 0.2k/6),
warm from R_e inward. E‴ from least-squares polynomial fits of F(R) (degree 3, 4, 5) and, independently, of the HF
bond force (−dF/dR; degree 2, 3, 4; E‴ = −F″).

| bond | R_e (Bohr) | force at R_e | F(R) deg 5: E‴ at 0.8 R_e … R_e (Ha/Bohr³) | force deg 4: same | **max (1/12)\|E‴\|** (deg 5 / force deg 4) | fit residuals |
|---|---|---|---|---|---|---|
| O–H in H₂O | 1.8488 | +3.1e-4 | −4.77 … −1.80 | −4.83 … −1.80 | **0.398 / 0.402** | 5.3e-7 Ha / 3.2e-5 Ha/Bohr |
| N–H in NH₃ | 1.9367 | −6.0e-5 | −4.15 … −1.40 | −4.20 … −1.42 | **0.346 / 0.350** | 3.1e-7 Ha / 1.7e-5 Ha/Bohr |

- **Both are below κ_max = 1 Ha/Bohr³** (a factor 2.5 and 2.9), so no stop. The energy and force fits agree to 1 %.
- |E‴| grows toward short range (the repulsive wall): at R_e it is 1.8 (O–H) and 1.4 (N–H), i.e. (1/12)|E‴| = 0.15
  and 0.12, matching "M7 forces" (OH, |E‴| ≈ 2.2 at R_e on the `CoresOnly` surface at h = 0.20: 0.18). The spec's
  "≈ 0.18" is the value at R_e; at 0.8 R_e it doubles.
- The low-degree fits (cubic in F, quadratic in the force) average E‴ over the range (−2.87 and −2.45) and miss the
  data by 6e-5 Ha and 2e-3 Ha/Bohr, so they are not used.

## 5. Reproducing

`cargo test -p providers --release <test> -- --exact --ignored --nocapture` on `smearing-prototype` (the harness,
`crates/providers/src/scf_stall_diag.rs`):
- Step 0: `scf_stall_diag::scf_stall_diag` with `SCF_SMEAR=5e-3` and `DIAG_CASES` = the default-policy cases and
  controls; then `DIAG_POLICY=none DIAG_DS=1` with the four all-electron cases.
- Steps 1–4: `scf_stall_diag::a1_measure` with `A1_MODE` = `v3` (`A1_AB`, `A1_PTS=quick|full`), `n3` (`A1_Z`,
  `A1_ZC`, `A1_CHARGE`, `A1_RS`, `A1_R`, `A1_BETAS`), `cost` (`A1_CASE`, `A1_N`, `A1_H`, `A1_TOL`) and `e3`
  (`A1_MOL`); `A1_KT` defaults to 5e-3.
- About 40 min of GPU in 15 batches of ≤ 9 min, with ≥ 3-minute gaps. The work paused after the CO₂⁺ 60°–80° batch
  (the PC was shut down for a GPU temperature scare) and resumed after the restart with the cost and E‴ batches.

## 6. Not measured, or not verified

- **No CPU confirmation** of anything here: V3's aspect is confirmed in Task 6, the NVE case runs in Task 11.
- **The NVE trajectory itself** and its per-step cost at 71³ (section 2.3).
- **The g/u label** of N₃'s and CO₂⁺'s half-filled pair is from the level order, not an orbital parity check.
- **Statistics:** one cold solve per V2 case (a cold GPU solve is bit-reproducible); one stencil per V3 aspect.
- **Why all-electron H₂O + 2 H takes 13 s per iteration** (section 0).
