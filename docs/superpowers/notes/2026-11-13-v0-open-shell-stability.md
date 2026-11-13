# V0: lone open-shell atoms under Fermi–Dirac smearing (2026-11-13)

Gate V0 of `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md` (at ad774a9): does Fermi–Dirac (FD) smearing at
kT = 1e-3 Ha keep the symmetric open-shell atomic states stable, on the GPU, at four lattice sites and on both grids?
This note holds the measurement and the verdict against the pre-registered stop. No production code was changed. The
prototype and harness stay uncommitted in the `scf-stall` worktree.

## Verdict

> **2026-11-14:** the user ruled kT = 5 mHa with a 1e-5 scan stop; the re-run under that ruling is the last
> section, "2026-11-14: V0 at kT = 5 mHa".

**V0 fails at kT = 1e-3 Ha.** The pre-registered stop applies, and the user rules among (i)–(iii).

Every accepted open-shell species fails at least one rule at kT = 1e-3. The gapped controls pass (N at M = 4, O and C
at M = 5). Two mechanisms are involved, and they need different remedies:

1. **Physical symmetry breaking at M = 1 (O) and M = 2 (N).** At 1e-3 the SCF lands on a polarised integer state at
   some sites and on the symmetric ensemble at others. The two lie 21–42 mHa apart.
   - O M = 1 on the h_req grid: a 21.1 mHa site spread (face polarised, generic symmetric; the node and centre never
     converge).
   - N M = 2: a 38.5 mHa spread at h = 0.15 and 28.9 mHa at h_req.
   - The CPU f64 cross-check confirms this is physics, not f32. O M = 1 at the generic site fails on the CPU too
     (Δρ = 2.9e-2 after 100 iterations).
2. **The GPU f32 noise floor (spec risk 8), in every open-shell case.** FD's slope at a degenerate shell turns the f32
   eigenvalue jitter (δε ≈ 1e-9–1e-7 Ha) into a Δρ floor of about 1e-6–1e-5. That equals or exceeds the reference
   scan's GPU stop tolerance of 1e-6 (`SAMPLE_TOL`; `effective_gpu_tol` leaves it at 1e-6 for N ≤ 10).
   - The symmetric state is reached, but it converges late or never. Examples: C M = 1 at 200/200/200/200
     iterations, F M = 2 at 3 of 4 sites, O M = 3 at h_req at 126/200/106/32.
   - The spec's own jitter rule (floor ≤ stop/3, §4) fails at 1e-3 for every open-shell case measured except B at
     M = 4, whose median 3.1e-7 is just under 3.3e-7.
   - It still fails at the largest kT measured for O M = 1 (8 mHa: 1.1e-6), C M = 1 (8 mHa: 2.8e-6), F M = 4
     (5 mHa: 5.5e-7) and N M = 2 (3 mHa: 2.8e-6).
   - The CPU f64 cross-check shows the difference. O M = 1 at 2 mHa converges on the CPU in 15 and 27 iterations,
     while the GPU fails at both sites (best Δρ 3.6e-6, 1.4e-6).

**Failing at 1e-3 (scan tolerance 1e-6, four sites, both grids):**
- pseudo O at M = 1 and M = 3;
- pseudo C at M = 1, and at M = 3 on its h_req grid;
- pseudo F at M = 2 and M = 4;
- pseudo N at M = 2;
- pseudo B at M = 2 on its h_req grid only: one site takes 71 iterations, everything else passes;
- all-electron O at M = 3 (both grids), and at M = 1 on Balanced (also failing under `DegenerateShells`).

**Passing at 1e-3:**
- B at M = 4, and B at M = 2 on Accurate;
- C at M = 3 on Accurate;
- all-electron O at M = 1 on Accurate (a gapped closed shell on this surface, §6);
- the gapped controls.

**Smallest kT at which each failing case holds** (tested set 1, 2, 3, 5 and 8 mHa; 1e-4 for the integer limit):

| case | symmetric at all 4 sites, both grids | full gate at the scan's tol 1e-6 | gate at the interactive tol 1e-5 (≤ 60 iterations to Δρ < 1e-5, from the Δρ histories) |
|---|---|---|---|
| O M = 1 | 2 mHa (polarised at 1e-3) | **none ≤ 8 mHa.** 5 mHa passes Accurate and misses h_req at one site (72 iterations); 8 mHa misses both | 3 mHa |
| O M = 3 | 1 mHa (half-polarised at 1e-4) | 5 mHa | 5 mHa |
| C M = 1 | 1 mHa (polarised at 1e-4) | **none ≤ 8 mHa** (2–3 of 4 sites ≤ 60 iterations at 5 and 8 mHa) | 2 mHa |
| C M = 3 (h_req) | 1 mHa | 3 mHa | 1 mHa |
| F M = 2 | 1 mHa | 3 mHa | 1 mHa |
| F M = 4 | 3 mHa (symmetric at 1 mHa, half-polarised at the face at 2 mHa, symmetric from 3 mHa) | **none ≤ 8 mHa.** 5 mHa misses one site; at 8 mHa every site hits the witness cap (§7) | 5 mHa |
| N M = 2 | 2 mHa (polarised at 1e-3) | **none ≤ 8 mHa** | 2 mHa |
| B M = 2 (h_req) | 1 mHa | 3 mHa | 1 mHa |
| AE O M = 3 | 1 mHa | 8 mHa on Accurate. Balanced: every M = 3 site converges at 8 mHa, but M = 1 fails one site, so the scan's winner is not safe | 3 mHa |

Above 2–3 mHa the binding constraint is no longer the physics. Under the reference scan's stop rule, **no kT ≤ 8 mHa
passes every V0 rule for every species.** What blocks it is the GPU noise floor: the Δρ floor falls with kT, but
slowly, and at 5–8 mHa it is still 1–3e-6 for the M = 1 / N M = 2 / F M = 4 cases. If convergence is judged at 1e-5, the threshold set
by the states and convergence is **5 mHa** (O M = 3 and F M = 4). The **symmetric-state** threshold alone is
**3 mHa** (F M = 4). The kT at which the symmetric ensemble becomes the *lowest-F* state (§5.3) is 2.6–11 mHa by case.

**Site spread at kT = 1e-3** (the states' F at the four sites, from the tol 1e-5 run so that every state ends; Table 1):
- **Polarising cases:**
  - O M = 1: 21.1 mHa at h_req (face polarised −16.074985, generic symmetric −16.053872). At h = 0.15 it is
    0.0012 mHa over the three symmetric sites, but the centre polarises and does not converge.
  - N M = 2: 38.5 mHa at h = 0.15 (node and generic polarised −9.895791, centre and face symmetric −9.857277). At
    h_req it is 28.9 mHa (a half-polarised node −9.867162 against polarised −9.896083).
- **Noise-limited cases** (the symmetric state everywhere): the spread is within the rule wherever every site
  converged.

| case | h_req spread (mHa) | h = 0.15 spread (mHa) | bound (mHa) |
|---|---|---|---|
| O M = 3 | 0.0067 | 0.0012 | 0.0099 / 0.0026 |
| C M = 1 | 0.0413 | 0.0022 | 0.0432 / 0.0036 |
| C M = 3 | 0.0494 | 0.0018 | |
| F M = 2 | — | 0.0021 | 0.0046 |
| F M = 4 | — | 0.013 (3 sites; centre never converges) | |
| B M = 2 | 0.0259 | 0.0038 | 0.0272 / 0.0049 |
| AE O M = 3 | 1.2450 (Balanced) | 0.287 (3 sites) | 1.2458 |

The winning M is unchanged wherever the states converge: O 3, C 3, N 4, F 2, B 2 (AE O 1, §6). At 1e-3 the scan
still picks a **wrong** M wherever the winner failed to converge and a higher M did. Examples: O at h_req picks M = 5
at the centre; F picks M = 4 at 3 sites; C at h_req picks M = 5 / M = 1 at two sites.

## 1. Method

### 1.1 Prototype (uncommitted; made to match the spec)

The diagnosis's `SCF_SMEAR` prototype differed from spec §4.1–4.3 in four ways:
- a ±40 clip;
- a single-sided 200-step bisection;
- renormalised f;
- no 36 kT witness arm.

For this measurement I replaced it with `qm::scf::diag_smear` (`crates/qm/src/scf.rs`), called by both solvers inside
`solve_shared`'s phase 1 (`crates/qm/src/scf.rs` CPU, `crates/qm-gpu/src/scf.rs` `Channel::solve_shared` GPU). It
follows the spec:
- FD in the overflow-free form;
- μ_σ from **two bisections** (the ≥ and ≤ arms) on the bracket [ε_min − 50w, ε_max + 50w], stopping when the
  midpoint hits an endpoint, with μ their midpoint (the plateau midpoint);
- no renormalisation;
- a NaN check;
- S from x, s = ln(1 + e^{−|x|}) + |x|·f_<, Neumaier-summed;
- **both witness arms**: today's ε_N + 0.1 Ha arm, then growth until ε_k − μ ≥ 36 kT, within the existing
  `MAX_WITNESS_GROWTH` = 16.

Two diagnostic hooks:
- `Kind::Mv` adds Marzari–Vanderbilt cold smearing: f = ½ erfc(−u) + e^{−u²}/√(2π), u = (μ − ε)/σ − 1/√2, with the
  generalised entropy s̃ = −u e^{−u²}/√(2π), F = E − σΣs̃. Its tail arm is 8σ. erfc is an f64 series / continued
  fraction.
- The GPU loop records each step's levels and occupations for the harness (`diag_smear::TRACE`).

The closed forms check out: TS at the exactly degenerate node equals the spec's table to 7 digits (O M = 3
1.909543e-3, M = 1 3.819085e-3 at kT = 1e-3).

### 1.2 Harness

The harness is `v0_open_shell` in `crates/providers/src/scf_stall_diag.rs` (`#[ignore]`, env-driven). Each lone atom
goes through `providers::ks::problem` exactly as `GpuKs` builds it:
- the `Accepted` policy (`none` for all-electron), forced M;
- tol = `SAMPLE_TOL` = 1e-6 raised by `effective_gpu_tol`, which leaves 1e-6;
- `DegenerateShells` for the witness path. All-electron also runs it, because FD sits in its phase 1.

The cases:
- **Box.** 14 Bohr, odd n, so the origin is a grid node: 71³ at h = 0.20, 95³ at 0.15, 47³ at 0.30, 41³ at 0.35,
  57³ at 0.25.
- **Sites.** The node (0, 0, 0), the cube centre (h/2)(1, 1, 1), the face (0, h/2, h/2), and a generic site
  (0.37, 0.13, 0.29)·h.
- **Grids.** h_req and Accurate (0.15): B 0.35, C 0.30, N 0.20, O 0.20, F 0.15 (one grid). All-electron O runs
  Balanced 0.25 and Accurate 0.15, σ = h.
- **Multiplicities.** Every multiplicity `allowed_multiplicities` gives the scan: B, N and F at 2 and 4; C and O at 1,
  3 and 5.
- **Modes.** `DegenerateShells` (the baseline), then FD at 1e-4, 1, 2, 3, 5 and 8 mHa. The gapped M (O 5, C 5, N 4)
  ran at DS and 1e-3, plus the full kT list for O at h_req and for C on both grids. Their F does not depend on kT (TS ≤ 1e-9; e.g. O M = 5:
  −15.756807 at 1e-4 through 5e-3).

Each cold solve is followed by a warm re-solve from its converged state. The iteration cap is 200 for the first O and
C passes and 100 afterwards. The stall stop is at its default; it never fired, because these Δρ are far below
1e-3·N.

The rules are applied as the spec states them:
- **Convergence**: all four cold solves converge within 60 iterations.
- **Site spread**: the FD F spread over the four sites must be ≤ the DS E spread + 6σ_path + 1e-6 Ha. σ_path is the
  RMS over the sites of (F_warm − F_cold), divided by √2.
- **Same winner**: the lowest-F M is the same at every site and equals the DS winner. A gapped M's 1e-3 value stands in
  at the other kT.

The state is classed from the p-shell occupations and TS against the closed form:
- **S**: symmetric, every p level equal;
- **P**: polarised, an integer pattern;
- **p0.xx**: partly polarised, by the occupation spread.

✗ in the tables marks the end state of a solve that did not converge.

The noise floor was measured separately. Each case was solved cold at the node and h_req, then run for 14 warm steps at
tol = 1e-12. From steps 3 onwards the harness took the median and maximum Δρ, and the step-to-step RMS change of the
p-shell splittings (δε). It also computed the spec's estimate 2δε/(4w)·4/π (∫||ψ_i|² − |ψ_j|²| = 4/π for two p
orbitals with the same radial part).

Two extra checks:
- **CPU f64** (`qm::scf_observed`, the same `diag_smear`): O M = 1 at h_req in a 10 Bohr box (51³), at the node and
  generic sites, at 1 and 2 mHa. Alongside it, the GPU ran the same box and sites at 1, 2 and 5 mHa, M = 1 and 3.
- **Every kT = 1e-3 case at tol 1e-5** (the interactive tolerance), so that the noise-limited solves end and the
  states' energies and spreads can be read.

GPU jobs ran one at a time at BelowNormal priority. The GPU was shared with the user's session, so the wall times
quoted are indicative only.

### 1.3 Not measured (outside this run)

- Si, S and Cl are accepted open-shell species (3p; P is gapped at M = 4) but are not in the spec's V0 list. They were
  not run. By the spec's compactness argument their U is smaller than that of the 2p elements.
- The molecular V0 cases (OH bent and linear, O₂ singlet and triplet, H₂O, NH at auto) and the lopsided-tetrahedron
  seeding were not run. Neither were A1's other items: V3's aspect, V5's NVE case, and the all-electron witness cost.
- Statistics: one cold solve per site and kT. The noise-limited outcomes (converges in 30 or in 100+ iterations) are
  stochastic. One draw per cell is not enough to separate, for example, a pass at 3 mHa from a fail at 2 mHa for
  O M = 3 at h = 0.15. The pattern across kT and sites is the evidence, not a single cell.

## 2. Table 1: kT = 1e-3, per species × M × grid

The F values (Ha) at node / centre / face / generic come from the tol 1e-5 run. The state, iterations and winner come
from the tol 1e-6 gate run. The DS spread is the `DegenerateShells` E spread at the same sites. Spreads are in mHa.

| case | grid | DS spread | F node / centre / face / generic (tol 1e-5) | F spread | bound | state per site (gate run) | iterations (gate run, tol 1e-6) | winner per site (gate run) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| O M = 1 | 0.20 (71³) | 0.0061 | ✗ / ✗ / −16.074985 / −16.053872 | **21.1** | 0.0083 | P✗ / P✗ / P / S✗ | 200, 200, 85, 200 | 3, 5, 3, 3 | **fail** (state, conv, spread, winner) |
| O M = 1 | 0.15 (95³) | 0.0015 | −16.052684 / ✗ / −16.052683 / −16.052684 | 0.0012 of 3 | 0.0033 | S✗ / P✗ / S✗ / S✗ | 100 × 4 | 3 × 4 | **fail** (state at the centre, conv) |
| O M = 3 | 0.20 | 0.0067 | −16.108035 / −16.108028 / −16.108031 / −16.108032 | 0.0067 | 0.0099 | S / S✗ / S / S | 126, 200, 106, 32 | 3, 5, 3, 3 | **fail** (conv, winner) |
| O M = 3 | 0.15 | 0.0009 | −16.106838 / −16.106837 / −16.106838 / −16.106838 | 0.0012 | 0.0026 | S × 4 | 35, 75, 44, 83 | 3 × 4 | **fail** (conv) |
| O M = 5 | 0.20 / 0.15 | 0.0387 / 0.0195 | gapped, F = E (−15.756807 / −15.755790 at the node) | 0.0389 / 0.0202 | 0.0402 / 0.0209 | g × 4 | 13–15 | — | pass |
| C M = 1 | 0.30 (47³) | 0.0414 | −5.495496 / −5.495538 / −5.495524 / −5.495521 | 0.0413 | 0.0432 | S✗ / S✗ / S✗ / S | 200, 200, 200, 173 | 3, 5, 3, 1 | **fail** (conv, winner) |
| C M = 1 | 0.15 | 0.0023 | −5.494743 / −5.494741 / −5.494742 / −5.494742 | 0.0022 | 0.0036 | S✗ × 4 | 200 × 4 | 3 × 4 | **fail** (conv) |
| C M = 3 | 0.30 | 0.0497 | −5.539407 / −5.539456 / −5.539440 / −5.539435 | 0.0494 | 0.0516 | S / S✗ / S / S✗ | 20, 200, 29, 200 | as above | **fail** (conv) |
| C M = 3 | 0.15 | 0.0020 | −5.538642 / −5.538640 / −5.538640 / −5.538641 | 0.0018 | 0.0033 | S × 4 | 21, 13, 12, 16 | 3 × 4 | pass |
| C M = 5 | 0.30 / 0.15 | 0.2208 / 0.0014 | gapped | 0.2208 / 0.0014 | 0.2220 / 0.0026 | g × 4 | 10–13 | — | pass |
| N M = 2 | 0.20 (71³) | 0.0078 | −9.867162 / −9.896079 / ✗ / −9.896083 | **28.9** | 0.0096 | p0.50✗ / P / p0.51✗ / P | 100, 73, 100, 93 | 4 × 4 | **fail** (state, conv, spread) |
| N M = 2 | 0.15 | 0.0012 | −9.895791 / −9.857277 / −9.857278 / −9.895791 | **38.5** | 0.0029 | P / S✗ / S✗ / P | 60, 100, 100, 80 | 4 × 4 | **fail** (state, conv, spread) |
| N M = 4 (control) | 0.20 / 0.15 | 0.0081 / 0.0010 | gapped (−9.956245 / −9.955945 at the node) | 0.0081 / 0.0010 | 0.0098 / 0.0022 | g × 4 | 13–18 | — | pass |
| F M = 2 | 0.15 | 0.0021 | −24.410831 / −24.410833 / −24.410830 / −24.410832 | 0.0021 | 0.0046 | S✗ / S / S✗ / S✗ | 100, 51, 100, 100 | 4, 2, 4, 4 | **fail** (conv, winner) |
| F M = 4 | 0.15 | 0.0173 | −23.919496 / ✗ / −23.919483 / −23.919490 | 0.013 of 3 | 0.0199 | S / S✗ / S / S | 96, 100, 34, 64 | as above | **fail** (conv) |
| B M = 2 | 0.35 (41³) | 0.0259 | −2.678132 / −2.678106 / −2.678114 / −2.678121 | 0.0259 | 0.0272 | S × 4 | 18, **71**, 33, 54 | 2 × 4 | **fail** (one site > 60) |
| B M = 2 | 0.15 | 0.0038 | −2.678035 / −2.678031 / −2.678032 / −2.678034 | 0.0038 | 0.0049 | S × 4 | 49, 36, 29, 23 | 2 × 4 | pass |
| B M = 4 | 0.35 / 0.15 | 0.046 / 0.004 | symmetric (f = ⅔) | 0.0463 / 0.0043 | 0.0477 / 0.0058 | S × 4 | 12–17 | — | pass |
| AE O M = 3 | 0.25 (57³) | 1.2402 | −36.366762 / −36.365517 / −36.365933 / −36.366087 | 1.2450 | 1.2458 | S (↓ ⅔ × 3) | 32, 100, 78, 100 | M = 1 wins where it converges (§6) | **fail** (conv) |
| AE O M = 3 | 0.15 | 0.6260 | ✗ / −45.157647 / −45.157859 / −45.157934 | 0.287 of 3 | 0.6311 | S | 100 × 4 | — | **fail** (conv) |
| AE O M = 1 | 0.15 / 0.25 | 0.577 / — | gapped 1s²2p⁶ (§6) | 0.574 / — | 0.579 | g | 26, 24, 22, 27 / 100, 13, 100, 100 | 1 × 4 / mixed | pass / fail (DS fails the same 3 sites) |

The AE O "state" column of the analysis script says "part" / "pol" for M = 3 / M = 1. That is a rank-ordering
artefact: on this σ = h surface the AE 2s lies *above* 2p (§6). Read through TS/TS₀ = 1.00, M = 3 is the symmetric
↓ ⅔ shell.

## 3. kT scan (tol 1e-6 gate run)

Each cell gives the gate result at that kT, the number of sites converged ≤ 60 iterations / converged at all / solved,
and the states reached by the converged solves. Columns are 1e-4, 1, 2, 3, 5 and 8 mHa. **PASS** = all three rules
hold at that grid. A gapped M (O 5, C 5) shows "fail" when the species' winner rule fails at that kT, although its own
solves pass.

| case | h | M | 0.1 mHa | 1 mHa | 2 mHa | 3 mHa | 5 mHa | 8 mHa |
|---|---|---|---|---|---|---|---|---|
| O | 0.15 | 1 | — | fail 0/0/4 — | fail 0/0/4 — | fail 1/1/4 sym | **PASS** 4/4/4 sym | fail 2/2/4 sym |
| O | 0.15 | 3 | — | fail 2/4/4 sym | fail 0/0/4 — | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| O | 0.15 | 5 | — | **PASS** 4/4/4 gapped | — | — | — | — |
| O | 0.2 | 1 | fail 2/2/4 pol | fail 0/1/4 part | fail 1/2/4 sym | fail 1/3/4 sym | fail 3/4/4 sym | fail 3/4/4 sym |
| O | 0.2 | 3 | fail 0/3/4 part,sym | fail 1/3/4 sym | fail 1/1/4 sym | fail 2/2/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| O | 0.2 | 5 | fail 4/4/4 gapped | fail 4/4/4 gapped | fail 4/4/4 gapped | fail 4/4/4 gapped | **PASS** 4/4/4 gapped | fail 0/0/2 — |
| C | 0.15 | 1 | fail 4/4/4 pol | fail 0/0/4 — | fail 1/2/4 sym | fail 1/2/4 sym | fail 2/4/4 sym | fail 2/4/4 sym |
| C | 0.15 | 3 | fail 0/0/4 — | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| C | 0.15 | 5 | fail 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped |
| C | 0.3 | 1 | fail 0/0/4 — | fail 0/1/4 sym | fail 0/3/4 sym | fail 1/3/4 sym | fail 3/4/4 sym | fail 2/3/4 sym |
| C | 0.3 | 3 | fail 0/0/4 — | fail 2/2/4 sym | fail 3/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| C | 0.3 | 5 | fail 4/4/4 gapped | fail 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped | **PASS** 4/4/4 gapped |
| F | 0.15 | 2 | — | fail 1/1/4 sym | fail 2/2/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| F | 0.15 | 4 | — | fail 1/3/4 sym | fail 1/1/4 sym | fail 1/1/4 sym | fail 3/3/4 sym | fail 0/0/4 — |
| N | 0.15 | 2 | — | fail 1/2/4 pol | fail 0/0/4 — | fail 3/3/4 sym | fail 0/3/4 sym | fail 2/3/4 sym |
| N | 0.15 | 4 | — | **PASS** 4/4/4 gapped | — | — | — | — |
| N | 0.2 | 2 | — | fail 0/2/4 pol | fail 1/1/4 sym | fail 1/2/4 sym | fail 2/2/4 sym | fail 1/2/4 sym |
| N | 0.2 | 4 | — | **PASS** 4/4/4 gapped | — | — | — | — |
| B | 0.15 | 2 | — | **PASS** 4/4/4 sym | fail 2/2/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| B | 0.15 | 4 | — | **PASS** 4/4/4 sym | fail 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| B | 0.35 | 2 | — | fail 3/4/4 sym | fail 3/3/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| B | 0.35 | 4 | — | **PASS** 4/4/4 sym | fail 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym | **PASS** 4/4/4 sym |
| O AE | 0.15 | 1 | — | **PASS** 4/4/4 pol | fail 4/4/4 pol | **PASS** 4/4/4 pol | fail 4/4/4 pol | **PASS** 4/4/4 pol |
| O AE | 0.15 | 3 | — | fail 0/0/4 — | fail 0/1/4 part | fail 1/2/4 part | fail 2/2/4 part | **PASS** 4/4/4 part |
| O AE | 0.25 | 1 | — | fail 1/1/4 pol | fail 4/4/4 pol | fail 4/4/4 pol | fail 4/4/4 pol | fail 3/3/4 pol |
| O AE | 0.25 | 3 | — | fail 1/2/4 part | fail 1/1/4 part | fail 2/2/4 part | fail 3/3/4 part | fail 4/4/4 part |

Modes: `fd<w>` is FD at kT = w Ha, `mv<w>` is MV at σ = w Ha, `ds` is `DegenerateShells`.

Read across a row: a PASS at one kT next to a fail at the next is the noise-limited convergence of §1.3, not a change
of state. The states (and the ✗ end states of the solves that did not converge) are listed per site in the appendix.

## 4. The GPU noise floor (spec risk 8, the jitter rule)

Setup: warm steps 3–14 at tol 1e-12 from the converged node state at h_req. A row is missing where that kT's cold node
solve did not converge. The stop tolerance is 1e-6 (`SAMPLE_TOL` through `effective_gpu_tol`). The spec rule is a floor
≤ tol/3 = 3.3e-7.

| case | h | M | mode | median Δρ | max Δρ | δε ↑, ↓ (Ha) | max f(1−f) | spec estimate 2δε/(4w)·4/π | stop tol | tol/3 | median ≤ tol/3? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.35 | 2 | fd1e-3 | 1.1e-06 | 2.9e-06 | 6.0e-09, — | 0.222 | 3.8e-06 | 1.0e-06 | 3.3e-07 | NO |
| B | 0.35 | 2 | fd2e-3 | 4.6e-07 | 1.4e-06 | 3.3e-09, — | 0.222 | 1.1e-06 | 1.0e-06 | 3.3e-07 | NO |
| B | 0.35 | 2 | fd3e-3 | 5.1e-07 | 8.5e-07 | 2.2e-09, 1.7e-09 | 0.222 | 4.7e-07 | 1.0e-06 | 3.3e-07 | NO |
| B | 0.35 | 2 | fd5e-3 | 8.8e-08 | 2.0e-07 | 1.2e-09, 1.2e-09 | 0.222 | 1.6e-07 | 1.0e-06 | 3.3e-07 | yes |
| B | 0.35 | 2 | fd8e-3 | 1.4e-07 | 1.6e-07 | 7.2e-10, 2.9e-10 | 0.222 | 5.8e-08 | 1.0e-06 | 3.3e-07 | yes |
| B | 0.35 | 4 | fd1e-3 | 3.1e-07 | 2.1e-06 | 2.8e-09, — | 0.222 | 1.8e-06 | 1.0e-06 | 3.3e-07 | yes |
| B | 0.35 | 4 | fd2e-3 | 4.5e-07 | 8.1e-07 | 2.4e-09, — | 0.222 | 7.5e-07 | 1.0e-06 | 3.3e-07 | NO |
| B | 0.35 | 4 | fd3e-3 | 1.9e-07 | 3.0e-07 | 1.8e-09, — | 0.222 | 3.7e-07 | 1.0e-06 | 3.3e-07 | yes |
| B | 0.35 | 4 | fd5e-3 | 5.8e-08 | 8.9e-08 | 2.4e-10, — | 0.222 | 3.1e-08 | 1.0e-06 | 3.3e-07 | yes |
| B | 0.35 | 4 | fd8e-3 | 1.5e-07 | 1.5e-07 | 4.4e-10, — | 0.222 | 3.5e-08 | 1.0e-06 | 3.3e-07 | yes |
| C | 0.3 | 1 | fd2e-3 | 2.9e-06 | 4.7e-06 | 6.1e-09, 5.5e-09 | 0.222 | 1.9e-06 | 1.0e-06 | 3.3e-07 | NO |
| C | 0.3 | 1 | fd5e-3 | 1.5e-06 | 1.6e-06 | 4.2e-09, 3.4e-09 | 0.222 | 5.3e-07 | 1.0e-06 | 3.3e-07 | NO |
| C | 0.3 | 1 | fd8e-3 | 2.8e-06 | 3.4e-06 | 1.3e-08, 2.3e-08 | 0.222 | 1.9e-06 | 1.0e-06 | 3.3e-07 | NO |
| C | 0.3 | 3 | fd1e-3 | 3.6e-06 | 5.7e-06 | 4.7e-09, — | 0.222 | 3.0e-06 | 1.0e-06 | 3.3e-07 | NO |
| C | 0.3 | 3 | fd2e-3 | 5.5e-07 | 7.4e-07 | 4.5e-09, — | 0.222 | 1.4e-06 | 1.0e-06 | 3.3e-07 | NO |
| C | 0.3 | 3 | fd3e-3 | 6.1e-07 | 7.1e-07 | 3.1e-09, — | 0.222 | 6.6e-07 | 1.0e-06 | 3.3e-07 | NO |
| C | 0.3 | 3 | fd5e-3 | 2.7e-07 | 4.4e-07 | 3.2e-09, 2.6e-09 | 0.222 | 4.1e-07 | 1.0e-06 | 3.3e-07 | yes |
| C | 0.3 | 3 | fd8e-3 | 1.9e-07 | 2.0e-07 | 8.9e-10, 7.5e-10 | 0.222 | 7.0e-08 | 1.0e-06 | 3.3e-07 | yes |
| F | 0.15 | 2 | fd2e-3 | 1.4e-06 | 2.0e-06 | 1.6e-09, 4.4e-09 | 0.222 | 1.4e-06 | 1.0e-06 | 3.3e-07 | NO |
| F | 0.15 | 2 | fd3e-3 | 2.3e-07 | 5.2e-07 | 2.1e-10, 3.1e-09 | 0.222 | 6.6e-07 | 1.0e-06 | 3.3e-07 | yes |
| F | 0.15 | 2 | fd5e-3 | 1.4e-06 | 2.2e-06 | 8.7e-09, 1.2e-08 | 0.222 | 1.6e-06 | 1.0e-06 | 3.3e-07 | NO |
| F | 0.15 | 2 | fd8e-3 | 3.0e-07 | 4.0e-07 | 1.3e-09, 3.3e-09 | 0.222 | 2.7e-07 | 1.0e-06 | 3.3e-07 | yes |
| F | 0.15 | 4 | fd1e-3 | 5.8e-06 | 2.9e-05 | 3.5e-08, 6.8e-08 | 0.222 | 4.3e-05 | 1.0e-06 | 3.3e-07 | NO |
| F | 0.15 | 4 | fd3e-3 | 1.0e-06 | 1.3e-06 | 4.5e-09, 7.1e-09 | 0.222 | 1.5e-06 | 1.0e-06 | 3.3e-07 | NO |
| F | 0.15 | 4 | fd5e-3 | 5.5e-07 | 1.2e-06 | 2.2e-09, 1.6e-08 | 0.222 | 2.0e-06 | 1.0e-06 | 3.3e-07 | NO |
| N | 0.2 | 2 | fd3e-3 | 2.8e-06 | 4.3e-06 | 7.8e-09, 8.7e-09 | 0.222 | 1.8e-06 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 1 | fd3e-3 | 3.7e-06 | 5.6e-06 | 9.9e-09, 6.2e-09 | 0.222 | 2.1e-06 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 1 | fd5e-3 | 1.7e-06 | 2.2e-06 | 1.0e-08, 9.1e-09 | 0.222 | 1.3e-06 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 1 | fd8e-3 | 1.1e-06 | 1.2e-06 | 4.3e-09, 5.1e-09 | 0.222 | 4.0e-07 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 3 | fd2e-3 | 2.4e-06 | 3.4e-06 | 6.9e-09, 1.1e-08 | 0.222 | 3.5e-06 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 3 | fd3e-3 | 9.4e-07 | 2.1e-06 | 2.7e-09, 1.1e-08 | 0.222 | 2.3e-06 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 3 | fd5e-3 | 1.5e-07 | 2.0e-07 | 1.8e-09, 2.2e-09 | 0.222 | 2.8e-07 | 1.0e-06 | 3.3e-07 | yes |
| O | 0.2 | 3 | fd8e-3 | 1.9e-07 | 2.2e-07 | 1.6e-09, 1.3e-09 | 0.222 | 1.2e-07 | 1.0e-06 | 3.3e-07 | yes |
| O | 0.2 | 3 | mv6e-3 | 1.8e-06 | 3.2e-06 | 5.1e-09, 1.2e-08 | 0.222 | 1.3e-06 | 1.0e-06 | 3.3e-07 | NO |
| O | 0.2 | 3 | mv1.5e-2 | 4.1e-07 | 7.5e-07 | 2.8e-09, 9.3e-09 | 0.222 | 3.9e-07 | 1.0e-06 | 3.3e-07 | NO |

- Under `DegenerateShells` the warm Δρ is exactly 0 (the equal share is flat; a warm DS solve ends in 1 step).
- Under FD the floor is 1e-7–6e-6 and falls with kT only slowly. For O M = 1, C M = 1, N M = 2 and F M = 4 it does not
  reach tol/3 at any kT measured.
- The spec's estimate tracks the measured median within a factor of about 3.
- The eigenvalue jitter itself is 1e-9–7e-8 Ha. It is largest for F at M = 4 (3.5e-8 / 6.8e-8 at 1 mHa).

## 5. States, U, and where the symmetric ensemble is the lowest F

### 5.1 The integer (kT = 1e-4) states, and U from their splittings

U is the level splitting per unit occupation difference. U > 0 means the occupied level drops (aufbau-stable). The
same-spin value is U↑↑; the cross-spin value is U↑↓, from the other channel's splitting.

| case (grid, site) | occupations at 1e-4 | splitting | U↑↑ | U↑↓ | E_pol − E_sym (DS) |
|---|---|---|---|---|---|
| O M = 3 (0.20, face) | ↓ (½, ½, 0): a half-polarised branch | ↓ 9.78 mHa at Δf = ½; ↑ 34.06 mHa (the ↑ level on the ↓-occupied axes higher) | **+19.6 mHa** | **−68.1 mHa** | −1.50 mHa |
| F M = 4 (0.15, node) | ↓ (1, 0, 0) | ↓ 14.39 mHa; ↑ 90.82 mHa | +14.4 | −90.8 | −5.01 |
| O M = 1 (0.20, face) | ↑ = ↓ = (1, 1, 0), the same pattern in both channels | occupied −0.3780 / −0.3095, empty −0.3041 (5.4 mHa above) | — | — | −24.75 |
| C M = 1 (0.15, node) | ↑ = ↓ = (1, 0, 0) | empty levels +9.2 / +53.9 mHa above the occupied | — | — | −21.46 |
| N M = 2 (0.20, node) | ↑ (1, 1, 0), ↓ (0, 0, 1)-like: the opposite pattern (the spin mode) | ↑ 60.8 mHa, ↓ 66.7 mHa | — | — | −42.34 |

No integer state was reached for B, C M = 3 or F M = 2. B M = 2 stays symmetric and converges at 1e-4 (64
iterations); C M = 3, F M = 2 and B M = 4 stay symmetric at 1e-4 without converging.

Against the §4.7(d) premises:
- U↑↑ > 0 and U_c = U↑↑ + U↑↓ < 0 hold for O (U_c ≈ −48 mHa, U_s ≈ +88 mHa) and F (U_c ≈ −76, U_s ≈ +105).
- The linear thresholds are then (2/9)·U:
  - O M = 3: 4.3 mHa (spec 2.6);
  - the O M = 1 spin mode: 19.5 mHa (spec ≳ 5);
  - F M = 4: 3.2 mHa.
- **The SCF does not follow the linear estimate.** At 1e-3, well below these, DIIS converges onto the symmetric
  stationary state at every site for O M = 3, F M = 4, C M = 1 and M = 3, F M = 2 and B. Only O M = 1 and N M = 2
  polarise at 1e-3, and both are symmetric everywhere from 2 mHa.
- The polarised M = 1 states of O and C carry the **same** pattern in both channels (a charge-like pattern), with the
  occupied levels split by 68.6 mHa (O). This is not the "degenerate occupied pair" pattern the spec's U_c < 0
  argument concerns. N's M = 2 state carries the opposite pattern (the spin mode).

### 5.2 The CPU f64 cross-check (worst case: O M = 1)

O M = 1 at h_req in a 10 Bohr box (51³). On the CPU each solve took 83–670 s.

| site | kT | CPU f64 | GPU f32, same box |
|---|---|---|---|
| node | 1 mHa | converges, 18 iterations, symmetric, F −16.052350 | 100 iterations without converging, polarised (1, 1, 0) |
| generic | 1 mHa | **100 iterations without converging, Δρ 2.9e-2** (unstable) | converges, 83 iterations, partial (TS 1.39e-3), F −16.073606 |
| node | 2 mHa | converges, 15 iterations, symmetric, F −16.056169 | 100 iterations, best Δρ 3.6e-6, symmetric |
| generic | 2 mHa | converges, 27 iterations, symmetric, F −16.056158 | 100 iterations, best Δρ 1.4e-6, symmetric |

- At 1 mHa the M = 1 instability is physical. At the symmetric node f64 holds the saddle, but the lattice at the
  generic site seeds the unstable mode and the f64 SCF oscillates.
- At 2 mHa the f64 SCF converges at both sites in ≤ 27 iterations, while the f32 GPU stalls at Δρ ≈ 1e-6. What remains
  on the GPU at 2 mHa is the noise floor of §4.
- On the GPU in this box, O M = 3 stalls at the node at 1 mHa (best 2.1e-6; the generic site converges in 48) and at
  both sites at 2 mHa (best 7e-6, 2e-5). It converges at both only at 5 mHa (15 and 32 iterations).

### 5.3 Where the symmetric ensemble is the lowest F

F_sym = E_sym − kT·S₀, with S₀ the closed form (3.819 at M = 1 and N M = 2, 1.9095 for one open channel), and
F_pol ≈ E_pol − kT·S_pol. The symmetric ensemble is the *global* minimum of F only above kT_× = (E_sym − E_pol)/(S₀ −
S_pol). The estimate holds S_pol at its 1e-4 value; a polarised state with a gap of order kT gains entropy too, so the
true crossing lies somewhat higher.

| case | kT_× |
|---|---|
| O M = 3, against the (½, ½, 0) branch, S_pol = 2 ln 2 | 2.9 mHa |
| F M = 4 | 2.6 mHa |
| C M = 1 | 5.6 mHa |
| O M = 1 | 6.5 mHa |
| N M = 2 | 11.1 mHa |

Below kT_× the symmetric state the SCF finds is a stationary point but not the lowest F. Which one a solve reaches is
then path- and site-dependent, which is what §2 shows for O M = 1 and N M = 2 at 1 mHa.

**None of these polarised states changes a scan winner.** The winning M lies 25–60 mHa lower in every case: O M = 3,
C M = 3, F M = 2, N M = 4, B M = 2. Of the winners, only O M = 3 has a measured kT_× (2.9 mHa).

## 6. All-electron O (σ = h)

On the all-electron surface at σ = h = 0.15 / 0.25, **the AE 2s lies above the 2p**. The ↑ levels at M = 3 are 2p
−0.4967 (× 3) and 2s −0.3293 at h = 0.15. This is the smeared-nucleus effect behind M6's "singlet lowest".

So:
- **AE M = 1 is the gapped closed shell 1s² 2p⁶ (2s⁰).** Its gap is 113 mHa, and FD equals DS to 2e-6 Ha.
- **AE M = 1 is also the scan winner on both grids** (−45.255165 against −45.156365 for M = 3 at 0.15).
- AE M = 3 is the symmetric ↓ shell, 2 electrons in 3 p (TS/TS₀ = 1.00). It suffers the same noise-limited
  convergence:
  - at 1e-3, 0/4 converge at 0.15 and 2/4 at 0.25 (tol 1e-6);
  - all four sites converge at 8 mHa on both grids (16–27 iterations);
  - at tol 1e-5 and 1e-3, Balanced converges with a spread of 1.2450 mHa against a bound of 1.2458 (the AE egg-box,
    DS 1.2402).
- AE M = 1 at Balanced fails 3 of 4 sites under **`DegenerateShells` too**: a pre-existing failure, not FD's.

The analysis script's "part" / "pol" labels for AE are an artefact of assuming 1s < 2s < 2p.

## 7. The witness cap at 8 mHa

At kT = 8 mHa the spec's second witness arm needs ε_k ≥ μ + 36 kT = μ + 0.288 Ha. For the channels whose μ sits at a
diffuse level, the 14 Bohr box does not supply that within `MAX_WITNESS_GROWTH` = 16 extra vectors:
- O M = 5's ↑ (2s 2p³ 3s);
- F M = 4's ↑ (2s 2p³ 3s).

Every such solve fails with "no gap witness within 16 extra levels of spin 0", after 250–290 s of growth (O M = 5 at
h_req: node and centre; the face and generic sites were not reached before that batch was restarted. F M = 4: all 4
sites, within 2 steps each). At 5 mHa (0.18 Ha) they pass. **kT = 8 mHa therefore fails
V0's convergence rule for two scan multiplicities outright**, independently of the noise floor. It would also make
NOTE-6's all-electron H⁻ risk worse.

## 8. Cold smearing (Marzari–Vanderbilt), measured on O at h_req

Setup: four sites, cold and warm, tol 1e-6, at σ = 3, 6, 15 and 24 mHa. MV's f has a maximum slope of 0.755/σ against
FD's 0.25/kT. So for the noise amplification and the linear instability, MV at σ behaves like FD at kT ≈ σ/3.

| M | σ | converged ≤ 60 / converged / 4 | states | F spread | warm Δρ₁ | noise median |
|---|---|---|---|---|---|---|
| 1 | 3 mHa | 0/0/4 | polarised | — | — | — |
| 1 | 6 mHa | 0/0/4 | symmetric (every solve stalls) | — | — | — |
| 1 | 15 mHa | 1/2/4 (35, 84) | symmetric | — | 1.7e-6 | — |
| 1 | 24 mHa | 1/1/4 (48) | symmetric | — | 1.4e-6 | — |
| 3 | 3 mHa | 1/2/4 | symmetric | — | 8.5e-6 | — |
| 3 | 6 mHa | 1/1/4 | symmetric | — | 2.2e-6 | 1.8e-6 |
| 3 | **15 mHa** | **4/4/4** (24, 16, 19, 27) | symmetric | 0.0071 mHa | 1.4e-6 | 4.1e-7 (> tol/3) |
| 3 | 24 mHa | 2/2/4 | symmetric | — | 1.1e-6 | — |

- **Cold smearing is no better than FD at equal slope**, which is the spec's own diagnosis of the failure (set by the
  slope at μ). MV at 15 mHa matches FD at 5 mHa: M = 3 passes, M = 1 does not.
- The energy bias at the degenerate atom is of the same order as FD's. MV at 15 mHa gives generalised TS 7.55 mHa at
  O M = 3, against FD at 5 mHa 9.55 mHa. E is unchanged (−16.10612 in both). MV's O(σ³) advantage concerns
  non-degenerate pinned levels, not an exactly degenerate shell.
- MV's Σf(μ) is not monotone (f peaks at 1.083). At 24 mHa O M = 3's TS (6.20e-3) disagrees with the
  synthetic-degenerate closed form (1.207e-2) while E agrees. μ is not unique there, and the solver and the closed form
  evidently took different roots. Not investigated further; it is a cost of (iii) if pursued.

## 9. Against the pre-registered stop

**V0 fails**, so the stop applies. The numbers for the user's ruling, one global kT (rule 1). I make no choice.

**(i) Raise kT.**
- It costs a lone-atom bias of −kT·S₀ at the symmetric state: O M = 3 −1.91·kT and M = 1 −3.82·kT. At 5 mHa that is
  −9.5 / −19.1 mHa, and the same scaling applies to pinned molecular levels.
- The smallest kT passing every V0 rule **as specified** (the scan's GPU tol 1e-6 and the jitter rule): **none ≤ 8
  mHa.**
  - O M = 1, C M = 1, N M = 2 and F M = 4 never converge at all four sites within 60 iterations on both grids.
  - 8 mHa breaks the witness cap (§7).
- If convergence is judged at the interactive 1e-5: 5 mHa (O M = 3, F M = 4; Table "Verdict").
- The symmetric state at every site: 3 mHa.
- The symmetric state as the global F minimum: 2.9 mHa for the winning multiplicities (O M = 3); 11 mHa for every M
  (N M = 2).

**(ii) Keep 1e-3 and document.**
- Not available as the spec words it. The rule requires the scan to converge at every M at every site.
  - At 1e-3, C M = 1 converges at 0 of 4 sites in 200 iterations at h = 0.15, and at 1 of 4 (173 iterations) at
    h_req. O M = 1 converges at 1 of 4 at h_req.
  - The scan then returns a **wrong** M at some sites: O at h_req gives M = 5 at the centre, F gives M = 4 at 3 sites,
    C at h_req gives M = 5 and M = 1.
- The site dependence to document would be:
  - 21–39 mHa for the polarising non-winning multiplicities (O M = 1, N M = 2);
  - ≤ 0.05 mHa and 1.25 mHa (AE egg-box) for the winners, where they converge.

**(iii) Cold smearing.**
- At the widths measured it gives no improvement over FD at equal slope (§8).
- It carries an O(σ) term at exact degeneracies and a non-unique μ at σ ≳ 20 mHa.

**Observation, not an option the spec lists.** The dominant failure on the GPU is the noise floor against the 1e-6 stop:
- the CPU converges O M = 1 at 2 mHa in 15–27 iterations where the GPU cannot (§5.2);
- the jitter rule in §4 is what fails for most cases.

Any remedy that lowers the f32 floor at a smeared degenerate shell would move every threshold above down to the
physical ones (2–3 mHa for the states). Examples: an FD-aware stop rule, or a tighter final eigensolve. That would need
its own spec and review.

## 10. Reproducing

Prototype (uncommitted, worktree `scf-stall`):
- `qm::scf::diag_smear` (`crates/qm/src/scf.rs`), with its call in the CPU `solve_shared`;
- the GPU call and per-step trace in `crates/qm-gpu/src/scf.rs` (`Channel::solve_shared`, the SCF loop's `DIAG` blocks);
- the harness `v0_open_shell` in `crates/providers/src/scf_stall_diag.rs`.

Run it as `cargo test -p providers --release v0_open_shell -- --ignored --nocapture`, with `V0_Z`, `V0_POLICY`,
`V0_GRIDS`, `V0_MS`, `V0_MODES` (ds, fd<w>, mv<w>), `V0_SITES`, `V0_BOX`, `V0_TOL`, `V0_WARM`, `V0_NOISE`, `V0_CPU` and
`V0_MAXIT` (module comment). The batches run here are listed below.

1. **Main runs** (tol 1e-6):
   - O and C, all M, both grids, DS and FD at 0.1, 1, 2, 3, 5 and 8 mHa, cap 200 (O at h = 0.15: cap 100, M = 5 at DS
     and 1 mHa only);
   - F, N M = 2, B: DS and FD at 1–8 mHa, cap 100; N M = 4: DS and 1 mHa;
   - AE O M = 1 and 3 at 0.25 and 0.15, DS and 1–8 mHa.
2. **Noise floor** at the node, h_req, every open M (`V0_NOISE=1 V0_WARM=0`).
3. **Integer limit**: 0.1 mHa at the node for B, N and F.
4. **Cold smearing**: O M = 1 and 3 at σ = 3, 6, 15 and 24 mHa, with noise.
5. **The 10 Bohr GPU control** and the CPU f64 runs.
6. **Every kT = 1e-3 case at tol 1e-5.**

In total there were about 1 700 GPU solves over about 5.3 h of wall time, shared with the user's session.

## Appendix: end state per site and the steps to reach Δρ < 1e-6 / 3e-6 / 1e-5 (cold solves, tol 1e-6 gate runs)

Key:
- S: symmetric.
- P: polarised integer.
- p0.xx: partly polarised (the p-occupation spread).
- g: gapped.
- sh: `DegenerateShells` shared.
- ✗: the end state of a solve that did not converge.

The AE rows use the pseudo ordering; read them through §6.

| case | h | M | mode | end state node/centre/face/generic | steps to Δρ < 1e-6 | < 3e-6 | < 1e-5 |
|---|---|---|---|---|---|---|---|
| B | 0.15 | 2 | ds | sh/sh/sh/sh | 17,13,13,13 | 17,12,11,11 | 14,11,10,10 |
| B | 0.15 | 2 | fd1e-3 | S/S/S/S | 49,36,29,23 | 26,28,27,23 | 19,23,18,17 |
| B | 0.15 | 2 | fd2e-3 | S/S/S✗/S✗ | 18,26,–,– | 16,21,63,– | 14,16,30,– |
| B | 0.15 | 2 | fd3e-3 | S/S/S/S | 13,51,15,17 | 13,11,13,13 | 11,10,13,11 |
| B | 0.15 | 2 | fd5e-3 | S/S/S/S | 13,20,21,14 | 13,20,19,13 | 10,18,17,11 |
| B | 0.15 | 2 | fd8e-3 | S/S/S/S | 12,13,12,13 | 11,12,11,12 | 9,11,10,10 |
| B | 0.15 | 4 | ds | sh/sh/sh/sh | 13,11,12,12 | 12,11,11,11 | 11,10,10,10 |
| B | 0.15 | 4 | fd1e-3 | S/S/S/S | 13,13,17,13 | 11,11,11,11 | 11,10,10,10 |
| B | 0.15 | 4 | fd2e-3 | S/S/S/S | 13,12,11,12 | 12,11,11,11 | 11,10,10,10 |
| B | 0.15 | 4 | fd3e-3 | S/S/S/S | 13,12,11,12 | 11,11,11,11 | 11,10,10,10 |
| B | 0.15 | 4 | fd5e-3 | S/S/S/S | 11,21,18,20 | 10,19,18,19 | 10,19,13,17 |
| B | 0.15 | 4 | fd8e-3 | S/S/S/S | 12,12,12,13 | 11,11,10,11 | 10,10,9,10 |
| B | 0.35 | 2 | ds | sh/sh/sh/sh | 11,12,12,12 | 10,12,11,12 | 9,12,10,11 |
| B | 0.35 | 2 | fd1e-3 | S/S/S/S | 18,71,33,54 | 14,35,28,23 | 9,28,24,21 |
| B | 0.35 | 2 | fd2e-3 | S/S/S/S✗ | 16,31,46,– | 10,20,31,– | 9,15,28,63 |
| B | 0.35 | 2 | fd3e-3 | S/S/S/S | 12,15,22,23 | 11,13,20,23 | 9,12,19,21 |
| B | 0.35 | 2 | fd5e-3 | S/S/S/S | 12,14,14,13 | 11,11,13,13 | 9,10,12,11 |
| B | 0.35 | 2 | fd8e-3 | S/S/S/S | 13,12,12,12 | 10,12,11,11 | 9,10,10,10 |
| B | 0.35 | 4 | ds | sh/sh/sh/sh | 12,12,11,12 | 11,10,10,11 | 10,9,9,10 |
| B | 0.35 | 4 | fd1e-3 | S/S/S/S | 12,15,16,16 | 11,11,12,11 | 10,10,9,10 |
| B | 0.35 | 4 | fd2e-3 | S/S/S/S | 12,12,17,12 | 11,10,11,11 | 10,10,9,10 |
| B | 0.35 | 4 | fd3e-3 | S/S/S/S | 12,12,11,11 | 11,10,10,11 | 10,9,9,10 |
| B | 0.35 | 4 | fd5e-3 | S/S/S/S | 12,11,14,11 | 11,11,11,11 | 10,9,9,9 |
| B | 0.35 | 4 | fd8e-3 | S/S/S/S | 11,11,11,11 | 10,10,10,10 | 9,10,10,9 |
| C | 0.15 | 1 | ds | sh/sh/sh/sh | 10,11,12,12 | 10,10,12,12 | 10,9,11,10 |
| C | 0.15 | 1 | fd1e-3 | S✗/S✗/S✗/S✗ | –,–,–,– | 32,36,36,72 | 19,19,28,17 |
| C | 0.15 | 1 | fd1e-4 | P/P/P/P | 26,29,24,28 | 25,29,24,26 | 24,27,22,26 |
| C | 0.15 | 1 | fd2e-3 | S/S/S✗/S✗ | 77,33,–,– | 53,20,168,26 | 27,12,16,13 |
| C | 0.15 | 1 | fd3e-3 | S/S✗/S✗/S | 20,–,–,101 | 15,21,20,23 | 12,11,14,12 |
| C | 0.15 | 1 | fd5e-3 | S/S/S/S | 16,115,44,117 | 13,13,41,31 | 12,11,28,22 |
| C | 0.15 | 1 | fd8e-3 | S/S/S/S | 20,130,101,31 | 18,20,93,22 | 16,16,89,14 |
| C | 0.15 | 3 | ds | sh/sh/sh/sh | 12,13,13,12 | 11,10,10,12 | 10,10,10,10 |
| C | 0.15 | 3 | fd1e-3 | S/S/S/S | 21,13,12,16 | 11,11,12,11 | 10,10,10,10 |
| C | 0.15 | 3 | fd1e-4 | S✗/S✗/S✗/S✗ | –,–,–,– | –,–,40,– | 19,181,40,160 |
| C | 0.15 | 3 | fd2e-3 | S/S/S/S | 12,12,13,11 | 12,11,11,11 | 10,11,10,10 |
| C | 0.15 | 3 | fd3e-3 | S/S/S/S | 12,12,14,12 | 11,11,11,12 | 10,11,10,10 |
| C | 0.15 | 3 | fd5e-3 | S/S/S/S | 12,12,14,12 | 12,12,13,12 | 12,12,13,11 |
| C | 0.15 | 3 | fd8e-3 | S/S/S/S | 12,12,20,12 | 12,12,18,11 | 11,10,17,9 |
| C | 0.15 | 5 | ds | sh/sh/sh/sh | 13,12,12,12 | 12,11,11,11 | 10,9,9,10 |
| C | 0.15 | 5 | fd1e-3 | g/g/g/g | 13,12,12,12 | 12,11,11,11 | 10,9,9,10 |
| C | 0.15 | 5 | fd1e-4 | g/g/g/g | 13,12,12,12 | 12,11,11,11 | 10,9,9,10 |
| C | 0.15 | 5 | fd2e-3 | g/g/g/g | 13,12,12,12 | 12,11,11,11 | 10,9,9,10 |
| C | 0.15 | 5 | fd3e-3 | g/g/g/g | 13,12,12,12 | 12,11,11,11 | 10,9,9,10 |
| C | 0.15 | 5 | fd5e-3 | g/g/g/g | 11,12,11,11 | 11,11,11,10 | 10,10,10,10 |
| C | 0.15 | 5 | fd8e-3 | g/g/g/g | 11,12,12,13 | 10,11,11,11 | 10,10,10,10 |
| C | 0.3 | 1 | ds | sh/sh/sh/sh | 11,10,11,11 | 11,10,10,10 | 11,8,9,9 |
| C | 0.3 | 1 | fd1e-3 | S✗/S✗/S✗/S | –,–,–,173 | 127,96,72,107 | 27,24,48,63 |
| C | 0.3 | 1 | fd1e-4 | P✗/P✗/P✗/P✗ | –,–,–,– | –,–,–,– | –,–,–,– |
| C | 0.3 | 1 | fd2e-3 | S/S✗/S/S | 88,–,109,123 | 87,111,26,90 | 46,22,16,38 |
| C | 0.3 | 1 | fd3e-3 | S✗/S/S/S | –,185,34,106 | 70,23,33,36 | 12,12,21,24 |
| C | 0.3 | 1 | fd5e-3 | S/S/S/S | 19,91,55,46 | 14,16,50,17 | 11,14,33,14 |
| C | 0.3 | 1 | fd8e-3 | S/S/S/S✗ | 21,18,101,– | 18,14,24,35 | 14,12,13,29 |
| C | 0.3 | 3 | ds | sh/sh/sh/sh | 12,11,12,12 | 12,11,12,12 | 12,11,12,12 |
| C | 0.3 | 3 | fd1e-3 | S/S✗/S/S✗ | 20,–,29,– | 13,182,27,29 | 11,11,15,19 |
| C | 0.3 | 3 | fd1e-4 | S✗/S✗/p0.03✗/S✗ | –,–,–,– | 103,–,–,– | 30,23,63,– |
| C | 0.3 | 3 | fd2e-3 | S/S/S/S | 13,17,13,200 | 11,12,13,69 | 11,11,10,12 |
| C | 0.3 | 3 | fd3e-3 | S/S/S/S | 11,12,13,13 | 11,11,11,13 | 11,11,11,10 |
| C | 0.3 | 3 | fd5e-3 | S/S/S/S | 12,13,14,12 | 12,13,13,12 | 10,10,12,12 |
| C | 0.3 | 3 | fd8e-3 | S/S/S/S | 12,13,12,12 | 12,10,11,10 | 10,9,11,9 |
| C | 0.3 | 5 | ds | sh/sh/sh/sh | 10,10,10,11 | 10,9,9,11 | 10,8,8,10 |
| C | 0.3 | 5 | fd1e-3 | g/g/g/g | 10,10,10,11 | 10,9,9,11 | 10,8,8,10 |
| C | 0.3 | 5 | fd1e-4 | g/g/g/g | 10,10,10,11 | 10,9,9,11 | 10,8,8,10 |
| C | 0.3 | 5 | fd2e-3 | g/g/g/g | 10,10,10,11 | 10,9,9,11 | 10,8,8,10 |
| C | 0.3 | 5 | fd3e-3 | g/g/g/g | 10,10,10,11 | 10,9,9,11 | 10,8,8,10 |
| C | 0.3 | 5 | fd5e-3 | g/g/g/g | 10,11,10,11 | 10,10,10,11 | 9,9,9,10 |
| C | 0.3 | 5 | fd8e-3 | g/g/g/g | 11,10,16,11 | 11,10,16,11 | 9,9,16,10 |
| N | 0.15 | 2 | ds | sh/sh/sh/sh | 16,14,13,16 | 16,13,13,16 | 14,13,13,14 |
| N | 0.15 | 2 | fd1e-3 | P/S✗/S✗/P | 60,–,–,80 | 60,95,48,80 | 60,37,48,78 |
| N | 0.15 | 2 | fd2e-3 | S✗/S✗/S✗/S✗ | –,–,–,– | 50,51,73,48 | 42,23,33,28 |
| N | 0.15 | 2 | fd3e-3 | S✗/S/S/S | –,51,25,23 | 35,50,24,17 | 26,29,22,17 |
| N | 0.15 | 2 | fd5e-3 | S/S/S✗/S | 84,67,–,100 | 27,23,17,25 | 17,14,14,14 |
| N | 0.15 | 2 | fd8e-3 | S/S/S✗/S | 13,19,–,– | 13,15,16,28 | 12,14,14,14 |
| N | 0.15 | 4 | ds | sh/sh/sh/sh | 16,18,16,16 | 16,18,16,16 | 16,18,16,16 |
| N | 0.15 | 4 | fd1e-3 | g/g/g/g | 16,18,16,16 | 16,18,16,16 | 16,18,16,16 |
| N | 0.2 | 2 | ds | sh/sh/sh/sh | 18,26,19,20 | 18,25,19,18 | 17,24,19,17 |
| N | 0.2 | 2 | fd1e-3 | p0.50✗/P/p0.51✗/P | –,73,–,93 | 95,72,–,93 | 94,72,–,90 |
| N | 0.2 | 2 | fd2e-3 | S✗/S/S✗/S✗ | –,24,–,– | 21,24,–,– | 21,22,44,28 |
| N | 0.2 | 2 | fd3e-3 | S/S/S✗/S✗ | 52,100,–,– | 42,30,19,20 | 30,28,18,20 |
| N | 0.2 | 2 | fd5e-3 | S✗/S/S/S✗ | –,28,24,– | 22,21,24,– | 18,20,20,21 |
| N | 0.2 | 2 | fd8e-3 | S✗/S/S✗/S | –,22,–,84 | 21,21,45,57 | 20,20,25,15 |
| N | 0.2 | 4 | ds | sh/sh/sh/sh | 16,15,17,13 | 15,14,14,12 | 14,13,14,12 |
| N | 0.2 | 4 | fd1e-3 | g/g/g/g | 16,15,17,13 | 15,14,14,12 | 14,13,14,12 |
| O AE | 0.15 | 1 | ds | sh/sh/sh/sh | 31,22,23,25 | 25,22,21,21 | 20,22,20,20 |
| O AE | 0.15 | 1 | fd1e-3 | P/P/P/P | 26,24,22,27 | 24,24,21,21 | 20,21,19,20 |
| O AE | 0.15 | 1 | fd2e-3 | P/P/P/P | 27,20,20,23 | 25,20,19,19 | 22,18,18,18 |
| O AE | 0.15 | 1 | fd3e-3 | P/P/P/P | 27,20,21,21 | 22,19,18,20 | 21,19,17,19 |
| O AE | 0.15 | 1 | fd5e-3 | P/P/P/P | 18,20,19,20 | 17,17,16,17 | 16,16,14,15 |
| O AE | 0.15 | 1 | fd8e-3 | P/P/P/P | 18,19,19,20 | 16,17,16,17 | 14,15,14,13 |
| O AE | 0.15 | 3 | ds | sh/sh/sh/sh | 21,18,18,16 | 18,16,17,15 | 16,15,15,15 |
| O AE | 0.15 | 3 | fd1e-3 | p0.67✗/p0.67✗/p0.66✗/p0.67✗ | –,–,–,– | –,50,–,– | –,46,63,89 |
| O AE | 0.15 | 3 | fd2e-3 | p0.67✗/p0.67/p0.67✗/p0.67✗ | –,73,–,– | –,19,–,74 | 23,15,18,72 |
| O AE | 0.15 | 3 | fd3e-3 | p0.67/p0.67✗/p0.67/p0.67✗ | –,–,80,– | 23,27,66,36 | 16,15,43,18 |
| O AE | 0.15 | 3 | fd5e-3 | p0.67/p0.67/p0.67✗/p0.67✗ | 16,18,–,– | 14,17,–,16 | 14,15,16,15 |
| O AE | 0.15 | 3 | fd8e-3 | p0.67/p0.67/p0.67/p0.67 | 22,27,24,21 | 14,15,16,15 | 14,14,14,15 |
| O AE | 0.25 | 1 | ds | sh✗/sh/sh✗/sh✗ | –,13,–,– | –,13,–,– | –,13,–,– |
| O AE | 0.25 | 1 | fd1e-3 | P✗/P/P✗/P✗ | –,13,–,– | –,13,–,– | –,13,–,– |
| O AE | 0.25 | 1 | fd2e-3 | P/P/P/P | 14,15,14,14 | 13,13,13,13 | 13,11,11,11 |
| O AE | 0.25 | 1 | fd3e-3 | P/P/P/P | 14,16,14,15 | 13,13,13,14 | 12,11,11,11 |
| O AE | 0.25 | 1 | fd5e-3 | P/P/P/P | 14,14,14,14 | 13,13,13,13 | 12,11,11,11 |
| O AE | 0.25 | 1 | fd8e-3 | P/P✗/P/P | 17,–,18,17 | 15,–,18,17 | 13,–,15,17 |
| O AE | 0.25 | 3 | ds | sh/sh/sh/sh | 23,15,16,16 | 23,14,15,15 | 23,13,14,13 |
| O AE | 0.25 | 3 | fd1e-3 | p0.67/p0.67✗/p0.69/p0.67✗ | 32,–,78,– | 18,31,59,– | 17,17,52,40 |
| O AE | 0.25 | 3 | fd2e-3 | p0.67✗/p0.67/p0.67✗/p0.67✗ | –,32,–,– | 25,15,20,– | 17,15,19,16 |
| O AE | 0.25 | 3 | fd3e-3 | p0.67/p0.67✗/p0.67✗/p0.67 | 16,–,–,59 | 16,17,21,20 | 16,14,18,19 |
| O AE | 0.25 | 3 | fd5e-3 | p0.67/p0.67/p0.67✗/p0.67 | 26,24,–,42 | 26,21,19,21 | 26,19,15,14 |
| O AE | 0.25 | 3 | fd8e-3 | p0.67/p0.67/p0.67/p0.67 | 16,17,22,22 | 16,17,17,15 | 16,17,14,14 |
| O | 0.15 | 1 | ds | sh/sh/sh/sh | 13,13,13,14 | 11,11,11,12 | 10,10,10,10 |
| O | 0.15 | 1 | fd1e-3 | S✗/P✗/S✗/S✗ | –,–,–,– | 82,–,58,51 | 56,–,43,48 |
| O | 0.15 | 1 | fd2e-3 | S✗/S✗/S✗/S✗ | –,–,–,– | 60,–,25,33 | 29,–,21,30 |
| O | 0.15 | 1 | fd3e-3 | S✗/S✗/S✗/S | –,–,–,46 | –,92,66,32 | 45,22,20,21 |
| O | 0.15 | 1 | fd5e-3 | S/S/S/S | 30,50,29,16 | 15,17,21,16 | 15,17,18,14 |
| O | 0.15 | 1 | fd8e-3 | S/S/S✗/S✗ | 39,21,–,– | 18,16,17,18 | 11,15,13,11 |
| O | 0.15 | 3 | ds | sh/sh/sh/sh | 22,21,16,16 | 22,21,15,16 | 22,19,13,14 |
| O | 0.15 | 3 | fd1e-3 | S/S/S/S | 35,75,–,83 | 33,61,36,31 | 24,56,30,27 |
| O | 0.15 | 3 | fd2e-3 | S✗/S✗/S✗/S✗ | –,–,–,– | –,22,–,– | 21,12,–,– |
| O | 0.15 | 3 | fd3e-3 | S/S/S/S | 44,55,25,43 | 14,15,22,22 | 12,12,22,21 |
| O | 0.15 | 3 | fd5e-3 | S/S/S/S | 22,28,19,23 | 21,26,19,21 | 16,21,19,19 |
| O | 0.15 | 3 | fd8e-3 | S/S/S/S | 15,16,18,14 | 13,15,16,13 | 12,13,15,13 |
| O | 0.15 | 5 | ds | sh/sh/sh/sh | 14,17,17,16 | 12,15,16,16 | 12,14,12,14 |
| O | 0.15 | 5 | fd1e-3 | g/g/g/g | 13,14,14,14 | 13,13,13,13 | 12,12,12,12 |
| O | 0.2 | 1 | ds | sh/sh/sh/sh | 12,12,15,12 | 11,11,15,11 | 9,9,15,10 |
| O | 0.2 | 1 | fd1e-3 | P✗/P✗/P/S✗ | –,–,85,– | –,–,78,– | –,–,76,31 |
| O | 0.2 | 1 | fd1e-4 | P✗/P✗/P/P | –,–,39,32 | –,–,39,32 | –,–,38,32 |
| O | 0.2 | 1 | fd2e-3 | S✗/S/S/S✗ | –,50,194,– | 33,31,52,89 | 32,28,41,70 |
| O | 0.2 | 1 | fd3e-3 | S/S/S✗/S | 37,92,–,179 | 32,58,153,62 | 23,19,24,53 |
| O | 0.2 | 1 | fd5e-3 | S/S/S/S | 72,29,51,53 | 20,16,24,24 | 15,15,16,22 |
| O | 0.2 | 1 | fd8e-3 | S/S/S/S | 30,40,157,25 | 23,26,20,22 | 18,15,11,20 |
| O | 0.2 | 3 | ds | sh/sh/sh/sh | 17,24,28,22 | 17,24,28,21 | 17,24,28,20 |
| O | 0.2 | 3 | fd1e-3 | S/S✗/S/S | 126,–,106,32 | 111,111,16,32 | 36,100,16,23 |
| O | 0.2 | 3 | fd1e-4 | S/S/p0.50/p0.50✗ | 92,168,71,– | 47,58,71,– | 44,55,71,100 |
| O | 0.2 | 3 | fd2e-3 | S/S✗/S✗/S✗ | 27,–,–,– | 27,–,–,21 | 22,28,19,17 |
| O | 0.2 | 3 | fd3e-3 | S/S/S✗/S✗ | 35,27,–,– | 27,22,132,– | 27,22,127,– |
| O | 0.2 | 3 | fd5e-3 | S/S/S/S | 20,27,21,20 | 19,27,21,19 | 16,27,17,16 |
| O | 0.2 | 3 | fd8e-3 | S/S/S/S | 15,29,16,13 | 13,29,15,13 | 11,29,15,13 |
| O | 0.2 | 5 | ds | sh/sh/sh/sh | 14,14,14,13 | 12,13,13,12 | 11,11,12,11 |
| O | 0.2 | 5 | fd1e-3 | g/g/g/g | 15,14,13,13 | 12,12,13,12 | 11,12,12,12 |
| O | 0.2 | 5 | fd1e-4 | g/g/g/g | 15,14,13,13 | 12,12,13,12 | 11,12,12,12 |
| O | 0.2 | 5 | fd2e-3 | g/g/g/g | 13,14,13,13 | 12,12,12,12 | 11,12,12,12 |
| O | 0.2 | 5 | fd3e-3 | g/g/g/g | 14,14,16,17 | 13,12,14,16 | 11,12,12,15 |
| O | 0.2 | 5 | fd5e-3 | g/g/g/g | 13,13,13,16 | 12,12,12,13 | 11,12,12,12 |
| O | 0.2 | 5 | fd8e-3 | g✗/g✗/—/— | –,–,–,– | –,–,–,– | –,–,–,– |
| F | 0.15 | 2 | ds | sh/sh/sh/sh | 16,16,18,18 | 15,15,17,17 | 15,13,16,16 |
| F | 0.15 | 2 | fd1e-3 | S✗/S/S✗/S✗ | –,51,–,– | –,27,33,– | 34,22,31,25 |
| F | 0.15 | 2 | fd2e-3 | S/S✗/S/S✗ | 46,–,22,– | 22,–,21,25 | 17,15,13,17 |
| F | 0.15 | 2 | fd3e-3 | S/S/S/S | 24,25,48,20 | 16,21,17,16 | 15,17,14,16 |
| F | 0.15 | 2 | fd5e-3 | S/S/S/S | 23,20,30,19 | 17,20,18,17 | 13,17,16,15 |
| F | 0.15 | 2 | fd8e-3 | S/S/S/S | 19,17,23,17 | 17,16,17,17 | 16,14,14,17 |
| F | 0.15 | 4 | ds | sh/sh/sh/sh | 14,14,14,14 | 14,14,13,13 | 12,12,12,12 |
| F | 0.15 | 4 | fd1e-3 | S/S✗/S/S | 96,–,34,64 | 62,–,32,54 | 46,–,30,54 |
| F | 0.15 | 4 | fd2e-3 | S✗/S/p0.50✗/S✗ | –,55,–,– | –,55,84,29 | –,20,84,22 |
| F | 0.15 | 4 | fd3e-3 | S/S✗/S✗/S✗ | 37,–,–,– | 21,99,–,– | 12,96,–,27 |
| F | 0.15 | 4 | fd5e-3 | S/S/S/S✗ | 26,26,42,– | 12,23,34,37 | 11,20,30,12 |
| F | 0.15 | 4 | fd8e-3 | S✗/S✗/S✗/S✗ | –,–,–,– | –,–,–,– | –,–,–,– |

## 2026-11-14: V0 at kT = 5 mHa with the scan stop at 1e-5 (the user's ruling)

The user ruled on 2026-11-14: option (i), **kT = 5 mHa (Fermi–Dirac)**, with the lone-atom reference scan's
convergence stop at **1e-5** instead of 1e-6. This section re-runs V0 under that ruling, adds the species and cases
§1.3 left out (Si, S, Cl, and P as N's analogue; the spec's molecular cases), measures the 36 kT witness arm in
every case, and measures what stopping at 1e-5 costs in F and in the forces. No production code was changed; the
prototype and the harness stay uncommitted.

### Method

- **Prototype:** unchanged (`qm::scf::diag_smear`, §1.1).
- **Harness:** a new `v0_cases` test in `crates/providers/src/scf_stall_diag.rs` (uncommitted), next to
  `v0_open_shell`. Atoms and molecules go through `providers::ks::problem` exactly as `GpuKs` builds them (`Accepted`,
  or `none` for all-electron), at tol 1e-5 raised by `effective_gpu_tol` (which leaves 1e-5 for every case here).
  Iteration cap 100; the V0 rule is ≤ 60. Stall stop at its default (it never fired).
- **Cases:** the box, sites and grids of §1.2 (14 Bohr, odd n; node, centre, face, generic; h_req and 0.15).
  - Pseudo B, C, N, O, F (as before), and **Si, P, S, Cl** (h_req 0.35 for all four), every multiplicity
    `allowed_multiplicities` gives the scan: 1, 3, 5 for C, O, Si, S; 2, 4 for B, N, F, P, Cl.
  - All-electron O at M = 1, 3 and 5, Balanced (0.25) and Accurate (0.15), σ = h.
  - **Molecules** (spec §9 "V0 in detail"), pseudo throughout (H pseudised by the GPU dimer gate), the centroid at the
    site, on 0.20 (the finest h_req of their species) and 0.15:
    - OH at auto (M = 2), R = 1.834 Bohr. **OH is a diatomic, so "bent" has no meaning for it.** I read the spec's
      "OH (bent, linear)" as the two cases that differ physically: the bond along a lattice axis (`oh_lin`; the π
      pair degenerate by the lattice's C4) and along a low-symmetry direction (1, 0.37, 0.21)/|·| (`oh_gen`; the π
      pair split by the egg-box only). If the spec meant a bent radical such as HO₂ or H₂O⁺, that case is not run.
    - O₂ at auto (M = 1, the restricted singlet the scene solve gives) and at M = 3, R = 2.282 Bohr, along x.
    - H₂O at auto, O–H 1.81 Bohr, 104.4° (the diagnosis geometry).
    - NH at auto (M = 1) and, for reference, at M = 3, R = 1.958 Bohr, along x.
- **Per site and mode** (DS and FD at 5 mHa):
  1. cold at tol 1e-5;
  2. warm at tol 1e-5 (σ_path = RMS(F_warm − F_cold)/√2, as in §1.2);
  3. FD only, **noise**: warm steps at tol 1e-12, abandoned after 15 steps (so the state is parked, not dropped);
     the median and maximum Δρ of steps 3–15 are the floor. Every one of these steps is a tight eigen step (below
     Δρ = eig_tol/`EIG_TOL_FRACTION` ≥ 1e-6/0.0015 = 6.7e-4 the Δρ-tied eigen tolerance is clamped at the effective
     `eig_tol`), so this is the floor the stop test sees;
  4. FD only, **tight**: a warm re-solve at tol 3e-6 from the parked floor state. F_tight − F_cold and the largest
     force-component change measure the error of stopping at 1e-5 (the residual of the tight state is ≤ 3e-6, so
     these are upper estimates of the stop error plus the path scatter);
  5. molecules only: a further warm re-solve at **tol 1e-6** (cap 40), the sample teacher's tolerance, to see whether
     a pinned molecule can reach it at all.
- **Witness:** per channel, the extra vectors k − N_σ the solve ended with, the reach (ε_max − μ)/kT, and
  ε_max − ε_(N). The second arm needs (ε_max − μ)/kT ≥ 36; `MAX_WITNESS_GROWTH` = 16.
- **Rules:** as pre-registered (spec §9): convergence ≤ 60 iterations at every site; site spread of F ≤ DS spread of
  E + 6σ_path + 1e-6 Ha; the same winning M at every site; the jitter rule, floor ≤ stop/3 = 3.3e-6.
- GPU jobs ran one at a time at BelowNormal priority, sharing the GPU with the user's session.

### Verdict at kT = 5 mHa with the 1e-5 stop

**States, site spread and winner: pass for every species.** Every accepted open-shell species (B, C, N, O, F, Si, P,
S, Cl) and all-electron O end, in every converged solve at every site, on both grids and at every scanned M, in the
symmetric ensemble or a gapped state (TS equal to
the closed form to ≤ 1e-6 relative, or 0.3–0.6 % above it where a diffuse level is also partly occupied: F and Cl at
M = 4; O and S at M = 5 carry 6–7e-5 Ha from that level alone). No converged solve polarises. The F spread is within the DS spread + 6σ_path + 1e-6 Ha everywhere it can be formed, and
the scan's winning M is the same at every site and equals the DS winner: B 2, C 3, N 4, O 3, F 2, Si 3, P 4, S 3, Cl 2. All-electron O wins at M = 1
(the gapped 1s² 2p⁶ of §6) on both grids; on Balanced, DS picked M = 3 at three of four sites because its M = 1
stalls there (the pre-existing failure of §6), and FD fixes that.

**Convergence (≤ 60 iterations at every site): two misses.** S at M = 1 on Accurate (61 iterations at the node, after a 30-step plateau at 1.3–2.5e-5), and all-electron O at M = 5 (not converged in 100 at 2 of 4 sites on Balanced and 3 of 4 on Accurate; best Δρ ≈ 1.7e-5). Neither is a scan winner, and the scan leaves a failed multiplicity out, so no winner changes; both are V0 failures as pre-registered.

**Jitter rule (floor median ≤ stop/3 = 3.3e-6): fails for the M = 1-type states.** It fails for O, C and S at M = 1 on both grids, Si at M = 1 on Accurate, N and P at M = 2 on both grids, all-electron O at M = 5, one Si M = 3 site (Accurate, generic: 6.2e-6), and the O₂ singlet and NH at auto. For these the 1e-5
stop is reached by a downward fluctuation of a floor whose median is 1e-6 to 2.8e-5, not with a factor-3 margin;
every other case clears it by 4.5× to > 1e4×. See "D. The noise floor".

**Molecules: pass, with the same jitter caveat for the two-pair singlets.** OH (axis and off-axis), the O₂ triplet,
H₂O and NH at M = 3 pass every rule. The O₂ singlet and NH at auto (two half-filled pairs, TS = 4 kT ln 2 =
13.863 mHa at every site) converge within 19 and 42 iterations, are symmetric at every site, and miss only the jitter
rule. NH at auto has no DS spread to compare with: **`DegenerateShells` stalls on singlet NH at every site on both
grids** (stall stop at 57 iterations, best Δρ 7e-2), a pre-existing DS failure that FD fixes. Its FD spread (0.0104 /
0.0087 mHa) equals the egg-box spread of the same nuclei at M = 3 under DS (0.0106 / 0.0090).

**Witness (36 kT): fits in every case.** At most 15 of the 16 allowed extra vectors (S at M = 5, h_req); 7–12 for Si
and S at M = 3, P, F and Cl at M = 4 and the M = 5 states; 2–5 for B, C, N, O, F, Cl at their winning M, the
molecules and all-electron O at M = 1, 3. The cold-solve cost of the 3p atoms is 5–80× DS. Spec §4.2 (ruling 43) keeps 36 kT and 16 and lists the
options.

**The 1e-5 stop costs ≤ 1.7 µHa in F and ≤ 8.8e-7 Ha/Bohr in the forces of the partial-level solves on the pseudo
surface** (≤ 3.2e-6 including gapped ones; ≤ 2.9 µHa and ≤ 2.1e-6 all-electron), at or below the f32 path scatter. See "B".

### A1. Table: lone atoms at kT = 5 mHa, tol 1e-5 (FD rows; the DS rows give the spread bound)

Columns: the end state (S symmetric, TS against the closed form; P a gapped closed shell, TS ≈ 0; g gapped; tsX a
non-closed-form TS of X mHa); the cold iterations at node/centre/face/generic (✗ = not converged in 100); the F
spread against the pre-registered bound (DS spread + 6σ_path + 1e-6 Ha; "nan" where DS did not converge at every
site); σ_path; the noise floor (median of warm steps 3–15 at tol 1e-12, range over the four sites; 0 = the noise run
converged at once, floor < 1e-10); the largest single noisy Δρ; how many sites have a median ≤ 3.3e-6 (the jitter
rule against the 1e-5 stop); the witness; TS against the closed form; the stop error (B below).

| case | M | h | state | cold iterations (node/centre/face/generic) | F spread / bound (mHa) | σ_path (µHa) | noise median, range over sites | noise max | sites with median ≤ 3.3e-6 | witness: max extra (min reach, kT) | TS / closed form | stop error: max \|ΔF\| (µHa), max \|ΔF_I\| (Ha/Bohr) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O (AE) | 1 | 0.25 | P | 12/11/11/11 | 1.0309 / nan | 0.62 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (72) | 0.0000–0.0000 | 1.65, 1.6e-06 | pass (no DS spread: DS stalls) |
| O (AE) | 1 | 0.15 | P | 16/16/14/15 | 0.5802 / 0.5820 | 0.79 | 6.5e-12–2.8e-07 | 6.3e-07 | 4/4 | 2 (82) | 0.0003–0.0003 | 2.91, 2.1e-06 | pass |
| O (AE) | 3 | 0.25 | S | 26/19/15/14 | 1.2440 / 1.2470 | 0.97 | 6.4e-07–1.4e-06 | 2.0e-06 | 4/4 | 5 (45) | 0.9999–1.0000 | 1.67, 6.3e-07 | pass |
| O (AE) | 3 | 0.15 | S | 14/15/16/15 | 0.6247 / 0.6333 | 1.06 | 1.0e-07–1.5e-06 | 2.4e-06 | 4/4 | 5 (43) | 1.0000–1.0000 | 1.68, 2.0e-06 | pass |
| O (AE) | 5 | 0.25 | ts19.141,ts19.148 | 35/✗/28/✗ | 0.9135 / 1.3690 | 0.54 | 5.4e-06–6.5e-06 | 1.5e-05 | 0/2 | 9 (41) | TS 1.9e-02 Ha | 2.55, 5.9e-07 | **conv, jitter 2/2** |
| O (AE) | 5 | 0.15 | ts20.720,ts20.731,ts20.743 | ✗/21/62/72 | 0.2852 / 0.5854 | 0.50 | 3.3e-06–6.2e-06 | 9.9e-06 | 1/3 | 9 (41) | TS 2.1e-02 Ha | 1.47, 2.1e-06 | **conv, jitter 2/3** |
| B | 2 | 0.35 | S | 9/10/12/11 | 0.0259 / 0.0273 | 0.06 | 7.7e-08–2.6e-07 | 1.3e-06 | 4/4 | 4 (53) | 1.0000–1.0000 | 0.14, 1.1e-08 | pass |
| B | 2 | 0.15 | S | 10/18/17/11 | 0.0039 / 0.0050 | 0.03 | 9.2e-08–3.8e-07 | 1.1e-06 | 4/4 | 4 (54) | 1.0000 | 0.08, 5.0e-09 | pass |
| B | 4 | 0.35 | S | 10/9/9/9 | 0.0461 / 0.0475 | 0.05 | 8.4e-08–2.0e-07 | 2.9e-07 | 4/4 | 3 (56) | 1.0000–1.0000 | 0.06, 8.8e-08 | pass |
| B | 4 | 0.15 | S | 10/19/13/17 | 0.0043 / 0.0056 | 0.04 | 1.7e-11–4.8e-07 | 5.6e-07 | 4/4 | 4 (57) | 1.0000–1.0000 | 0.08, 4.7e-09 | pass |
| C | 1 | 0.3 | S | 11/14/33/14 | 0.0415 / 0.0432 | 0.14 | 3.7e-06–1.2e-05 | 2.2e-05 | 0/4 | 3 (50) | 1.0000–1.0000 | 0.32, 3.2e-08 | **jitter 4/4** |
| C | 1 | 0.15 | S | 12/11/28/22 | 0.0019 / 0.0039 | 0.10 | 6.2e-07–9.0e-06 | 2.9e-05 | 2/4 | 3 (50) | 1.0000 | 0.24, 1.3e-08 | **jitter 2/4** |
| C | 3 | 0.3 | S | 10/10/12/12 | 0.0495 / 0.0511 | 0.06 | 1.5e-07–2.7e-07 | 4.8e-07 | 4/4 | 4 (54) | 1.0000–1.0000 | 0.36, 3.8e-08 | pass |
| C | 3 | 0.15 | S | 12/12/13/11 | 0.0020 / 0.0036 | 0.09 | 4.6e-08–1.3e-07 | 2.2e-07 | 4/4 | 4 (53) | 1.0000 | 0.18, 1.8e-08 | pass |
| C | 5 | 0.3 | g | 9/9/9/10 | 0.2208 / 0.2220 | 0.04 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (41) | TS 1.1e-13 Ha | 0.10, 0.0e+00 | pass |
| C | 5 | 0.15 | g | 10/10/10/10 | 0.0013 / 0.0026 | 0.04 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (41) | TS 1.1e-13 Ha | 0.10, 0.0e+00 | pass |
| N | 2 | 0.2 | S | 18/20/20/21 | 0.0075 / 0.0111 | 0.37 | 1.6e-06–1.2e-05 | 4.1e-05 | 1/4 | 3 (56) | 1.0000 | 0.64, 7.3e-08 | **jitter 3/4** |
| N | 2 | 0.15 | S | 17/14/14/14 | 0.0015 / 0.0035 | 0.22 | 2.2e-06–7.0e-06 | 1.1e-05 | 3/4 | 3 (56) | 1.0000 | 0.44, 4.9e-08 | **jitter 1/4** |
| N | 4 | 0.2 | g | 12/12/11/12 | 0.0083 / 0.0097 | 0.11 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (40) | TS 6.4e-16 Ha | 0.29, 0.0e+00 | pass |
| N | 4 | 0.15 | g | 16/16/16/13 | 0.0009 / 0.0022 | 0.05 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (40) | TS 6.5e-16 Ha | 0.12, 0.0e+00 | pass |
| O | 1 | 0.2 | S | 15/15/16/22 | 0.0056 / 0.0096 | 0.42 | 5.2e-06–9.3e-06 | 9.5e-05 | 0/4 | 2 (74) | 1.0000 | 1.00, 1.9e-07 | **jitter 4/4** |
| O | 1 | 0.15 | S | 15/17/18/14 | 0.0012 / 0.0042 | 0.28 | 1.5e-06–1.4e-05 | 3.8e-05 | 2/4 | 2 (74) | 1.0000 | 0.37, 2.2e-07 | **jitter 2/4** |
| O | 3 | 0.2 | S | 16/27/17/16 | 0.0062 / 0.0103 | 0.45 | 9.5e-08–5.0e-07 | 1.2e-06 | 4/4 | 3 (41) | 1.0000–1.0000 | 0.85, 1.3e-07 | pass |
| O | 3 | 0.15 | S | 16/21/19/19 | 0.0010 / 0.0035 | 0.26 | 6.9e-08–8.2e-07 | 2.3e-06 | 4/4 | 3 (41) | 1.0000 | 0.60, 8.1e-08 | pass |
| O | 5 | 0.2 | ts0.063 | 11/12/12/12 | 0.0378 / 0.0405 | 0.10 | 9.7e-11–2.1e-10 | 2.3e-10 | 4/4 | 9 (46) | TS 6.3e-05 Ha | 0.49, 1.0e-10 | pass |
| O | 5 | 0.15 | ts0.063 | 11/13/12/12 | 0.0198 / 0.0219 | 0.11 | 7.3e-11–2.8e-10 | 2.8e-10 | 4/4 | 9 (46) | TS 6.3e-05 Ha | 0.22, 4.3e-07 | pass |
| F | 2 | 0.15 | S | 13/17/16/15 | 0.0016 / 0.0058 | 0.45 | 1.4e-07–2.2e-06 | 3.3e-06 | 4/4 | 2 (47) | 1.0000 | 1.67, 1.9e-07 | pass |
| F | 4 | 0.15 | S | 11/20/30/12 | 0.0173 / 0.0205 | 0.37 | 3.3e-07–1.2e-06 | 4.1e-06 | 4/4 | 9 (45) | 1.0033–1.0033 | 1.13, 1.5e-07 | pass |
| Si | 1 | 0.35 | S | 15/12/20/15 | 0.0517 / 0.0529 | 0.05 | 9.1e-07–1.7e-06 | 1.9e-05 | 4/4 | 9 (56) | 1.0000–1.0000 | 0.13, 4.6e-09 | pass |
| Si | 1 | 0.15 | S | 14/11/28/36 | 0.0113 / 0.0125 | 0.04 | 1.1e-06–5.2e-06 | 6.2e-06 | 2/4 | 9 (54) | 1.0000–1.0000 | 0.12, 3.4e-09 | **jitter 2/4** |
| Si | 3 | 0.35 | S | 10/11/11/10 | 0.0459 / 0.0478 | 0.09 | 7.5e-08–1.9e-07 | 2.4e-07 | 4/4 | 8 (57) | 1.0000–1.0000 | 0.16, 7.2e-09 | pass |
| Si | 3 | 0.15 | S | 12/12/12/22 | 0.0102 / 0.0114 | 0.05 | 0.0e+00–6.2e-06 | 8.5e-06 | 3/4 | 9 (58) | 1.0000–1.0000 | 0.15, 1.8e-07 | **jitter 1/4** |
| Si | 5 | 0.35 | g | 10/10/10/10 | 0.0558 / 0.0568 | 0.01 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 12 (63) | TS 3.6e-11 Ha | 0.02, 6.8e-08 | pass |
| Si | 5 | 0.15 | g | 11/11/11/12 | 0.0104 / 0.0114 | 0.02 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 10 (58) | TS 2.5e-11 Ha | 0.04, 0.0e+00 | pass |
| P | 2 | 0.35 | S | 11/15/18/20 | 0.0297 / 0.0322 | 0.18 | 1.5e-06–1.3e-05 | 1.7e-05 | 1/4 | 7 (61) | 1.0000–1.0000 | 0.38, 1.3e-08 | **jitter 3/4** |
| P | 2 | 0.15 | S | 10/27/25/29 | 0.0070 / 0.0082 | 0.06 | 5.5e-07–3.5e-06 | 2.0e-05 | 3/4 | 7 (61) | 1.0000–1.0000 | 0.12, 5.0e-09 | **jitter 1/4** |
| P | 4 | 0.35 | g | 13/13/13/11 | 0.0243 / 0.0261 | 0.05 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 10 (63) | TS 2.6e-12 Ha | 0.11, 3.2e-06 | pass |
| P | 4 | 0.15 | g | 11/11/11/11 | 0.0055 / 0.0070 | 0.04 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 10 (64) | TS 1.9e-12 Ha | 0.12, 0.0e+00 | pass |
| S | 1 | 0.35 | S | 9/10/14/14 | 0.0175 / 0.0197 | 0.22 | 7.4e-07–5.6e-06 | 1.3e-05 | 1/4 | 3 (72) | 1.0000 | 0.40, 1.2e-07 | **jitter 3/4** |
| S | 1 | 0.15 | S | 61/35/38/41 | 0.0043 / 0.0054 | 0.09 | 2.5e-06–7.2e-06 | 2.3e-05 | 1/4 | 4 (73) | 1.0000 | 0.25, 1.6e-08 | **conv, jitter 3/4** |
| S | 3 | 0.35 | S | 13/11/11/11 | 0.0169 / 0.0185 | 0.07 | 1.3e-07–4.1e-07 | 4.7e-07 | 4/4 | 9 (52) | 1.0000–1.0000 | 0.50, 5.6e-08 | pass |
| S | 3 | 0.15 | S | 11/22/11/12 | 0.0037 / 0.0051 | 0.09 | 2.6e-07–5.1e-07 | 8.0e-07 | 4/4 | 8 (52) | 1.0000 | 0.24, 5.8e-07 | pass |
| S | 5 | 0.35 | ts0.072,ts0.073 | 10/11/12/11 | 0.1062 / 0.1079 | 0.17 | 1.2e-10–2.0e-07 | 2.1e-07 | 4/4 | 15 (45) | TS 7.3e-05 Ha | 0.31, 5.5e-08 | pass |
| S | 5 | 0.15 | ts0.064 | 10/12/54/12 | 0.0205 / 0.0218 | 0.06 | 1.5e-10–1.2e-07 | 2.5e-07 | 4/4 | 10 (44) | TS 6.4e-05 Ha | 0.22, 2.8e-07 | pass |
| Cl | 2 | 0.35 | S | 9/9/9/10 | 0.0096 / 0.0118 | 0.22 | 2.6e-07–5.2e-07 | 8.3e-07 | 4/4 | 2 (50) | 1.0000–1.0000 | 0.54, 3.8e-08 | pass |
| Cl | 2 | 0.15 | S | 12/9/20/12 | 0.0018 / 0.0045 | 0.17 | 1.6e-07–1.6e-06 | 1.9e-06 | 4/4 | 3 (51) | 1.0000 | 0.54, 2.6e-08 | pass |
| Cl | 4 | 0.35 | S | 11/14/14/12 | 0.1085 / 0.1111 | 0.25 | 6.3e-07–1.3e-06 | 2.1e-06 | 4/4 | 10 (46) | 1.0060–1.0061 | 0.49, 4.6e-07 | pass |
| Cl | 4 | 0.15 | S | 11/23/30/23 | 0.0208 / 0.0225 | 0.11 | 1.9e-07–6.3e-07 | 8.3e-07 | 4/4 | 10 (47) | 1.0053–1.0053 | 0.15, 2.5e-08 | pass |

Notes on the table:
- **All-electron O at M = 1 on Balanced (0.25)**: DS stalls at three of four sites (the pre-existing failure of §6),
  so there is no DS spread; FD converges at all four in 11–12 iterations to the gapped 1s² 2p⁶ (2s above 2p on the
  σ = h surface, §6). Its FD spread, 1.03 mHa, is the all-electron egg-box (M = 3's DS spread on the same grid is
  1.24 mHa). On Balanced the DS winner was M = 3 at three sites only because its M = 1 failed there; FD's winner is
  M = 1 at every site.
- **All-electron O at M = 5** is a new case (not run at 1e-3). Its 6th ↑ electron sits in a diffuse, triply
  degenerate level at −0.02 Ha (f ≈ ⅓ each; TS 19–21 mHa). It is the worst noise case measured (best Δρ 1.7e-5 in
  100 iterations at three sites) and **fails the convergence rule at 2 of 4 sites on Balanced and 3 of 4 on
  Accurate**. It lies 670 mHa above the M = 1 winner, and the scan leaves a failed multiplicity out, so the winner is
  unaffected, but it is a V0 convergence failure as pre-registered.
- **S at M = 1 on Accurate** took 61 iterations at the node (rule: 60), after a 30-step plateau at 1.3–2.5e-5: a
  convergence failure by one iteration. Its other sites took 35–41.
- The DS warm re-solves of the diffuse M = 5 / M = 4 cases take ~25 iterations because DS's release fires again;
  that is DS's behaviour, not FD's.

### A2. Table: the spec's molecular cases at kT = 5 mHa

| case | M | h | state | cold iterations (node/centre/face/generic) | F spread / bound (mHa) | σ_path (µHa) | noise median, range over sites | noise max | sites with median ≤ 3.3e-6 | witness: max extra (min reach, kT) | TS / closed form | stop error: max \|ΔF\| (µHa), max \|ΔF_I\| (Ha/Bohr) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H₂O | auto | 0.2 | g | 16/13/13/13 | 0.0042 / 0.0051 | 0.07 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (40) | TS 5.9e-12 Ha | 0.19, 0.0e+00 | pass |
| H₂O | auto | 0.15 | g | 13/13/14/13 | 0.0030 / 0.0057 | 0.19 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 2 (40) | TS 5.9e-12 Ha | 0.46, 0.0e+00 | pass |
| NH | auto | 0.2 | ts13.863 | 15/14/16/16 | 0.0104 / nan | 0.22 | 4.5e-06–9.6e-06 | 6.9e-05 | 0/4 | 2 (51) | TS 1.4e-02 Ha | 0.56, 8.0e-08 | **jitter 4/4** (spread: DS stalls; vs NH M = 3 DS 0.0106) |
| NH | auto | 0.15 | ts13.863 | 15/15/42/19 | 0.0087 / nan | 0.21 | 2.9e-06–9.6e-06 | 1.8e-05 | 1/4 | 2 (51) | TS 1.4e-02 Ha | 0.47, 6.0e-08 | **jitter 3/4** (spread: DS stalls; vs NH M = 3 DS 0.0090) |
| NH | 3 | 0.2 | ts0.000 | 20/14/17/14 | 0.0106 / 0.0121 | 0.08 | 0.0e+00–8.0e-12 | 8.0e-12 | 4/4 | 3 (48) | TS 2.9e-07 Ha | 0.12, 4.4e-07 | pass |
| NH | 3 | 0.15 | ts0.000 | 15/14/14/14 | 0.0090 / 0.0102 | 0.04 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 4 (48) | TS 2.9e-07 Ha | 0.12, 1.0e-12 | pass |
| O₂ | auto | 0.2 | ts13.863 | 16/19/19/19 | 0.0090 / 0.0137 | 0.47 | 8.9e-07–2.8e-05 | 5.2e-05 | 2/4 | 2 (53) | TS 1.4e-02 Ha | 1.19, 2.3e-07 | **jitter 2/4** |
| O₂ | auto | 0.15 | ts13.863 | 14/16/18/18 | 0.0018 / 0.0085 | 0.66 | 6.8e-07–8.1e-06 | 1.3e-05 | 2/4 | 2 (53) | TS 1.4e-02 Ha | 1.56, 1.4e-07 | **jitter 2/4** |
| O₂ | 3 | 0.2 | g | 13/17/13/16 | 0.0076 / 0.0093 | 0.18 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 3 (40) | TS 2.0e-12 Ha | 0.59, 0.0e+00 | pass |
| O₂ | 3 | 0.15 | g | 15/15/13/14 | 0.0037 / 0.0058 | 0.25 | 0.0e+00–0.0e+00 | 0.0e+00 | 4/4 | 3 (40) | TS 2.0e-12 Ha | 0.63, 0.0e+00 | pass |
| OH off-axis | auto | 0.2 | ts6.931 | 14/18/35/14 | 0.0062 / 0.0103 | 0.44 | 5.8e-08–1.6e-06 | 5.7e-06 | 4/4 | 3 (53) | TS 6.9e-03 Ha | 0.92, 8.8e-07 | pass |
| OH off-axis | auto | 0.15 | ts6.931 | 43/14/14/14 | 0.0047 / 0.0078 | 0.36 | 5.4e-08–9.5e-07 | 1.6e-06 | 4/4 | 4 (53) | TS 6.9e-03 Ha | 0.76, 1.1e-07 | pass |
| OH along x | auto | 0.2 | ts6.931 | 14/14/14/14 | 0.0057 / 0.0079 | 0.25 | 2.9e-08–5.2e-07 | 9.1e-07 | 4/4 | 3 (53) | TS 6.9e-03 Ha | 0.70, 1.0e-07 | pass |
| OH along x | auto | 0.15 | ts6.931 | 14/14/21/14 | 0.0033 / 0.0049 | 0.20 | 7.4e-08–2.6e-07 | 5.5e-07 | 4/4 | 4 (53) | TS 6.9e-03 Ha | 0.68, 7.7e-08 | pass |

- **OH** (axis and off-axis): the ↓ π pair holds one hole, TS = 2 kT ln 2 = 6.931 mHa at every site (off-axis the
  egg-box splits the pair by 5e-6 Ha, f = 0.5001/0.4999). Passes every rule; reaches 1e-6 from the floor in 1–3
  steps.
- **O₂ singlet at auto** and **NH at auto**: two half-filled pairs, TS = 4 kT ln 2 = 13.863 mHa at every site, E equal
  to the DS E (O₂: −32.448841 against −32.448843 at the node, h = 0.15), i.e. the migration's class T. They converge in
  ≤ 19 (O₂) and ≤ 42 (NH) iterations at 1e-5 but miss the jitter rule (floor medians up to 2.8e-5 and 9.6e-6). From
  the floor, a re-solve to 1e-6 took 1–40 iterations, and failed once (O₂, h = 0.20, centre: best 1.0e-6 in 40).
  **`DegenerateShells` stalls on singlet NH at every site on both grids** (stall stop at 57 iterations, best Δρ
  7e-2), so its spread is compared with the same nuclei's DS spread at M = 3 (0.0106 / 0.0090 mHa): FD gives 0.0104 /
  0.0087.
- **O₂ triplet, H₂O, NH at M = 3**: gapped (TS ≤ 3e-7 Ha); pass. H₂O's frontier gap on the pseudo surface at h = 0.15
  is 0.252 Ha (HOMO −0.2710, LUMO −0.0189), TS 5.9e-12 Ha.
- Not run: the lopsided-tetrahedron seeding (§4.7(e) of the spec, item of A1), and a bent radical if the spec's
  "OH (bent)" meant one (method).

### C. The witness at 5 mHa

The 36 kT arm fits under `MAX_WITNESS_GROWTH` = 16 in every case (table column "witness"):
- 2–5 extra vectors for B, C, N, O, F and Cl at their winning M, S at M = 1, the molecules, and all-electron O at
  M = 1 and 3;
- 7–12 for Si at every M, S at M = 3, P at both M, F and Cl at M = 4, O and all-electron O at M = 5;
- **15 for S at M = 5 on h_req** (the 4s-like ↑ level against the box states), the only case near the cap.

At 8 mHa (§7) O at M = 5 and F at M = 4 exceeded 16; at 5 mHa they need 9. The reach lands at 40–82 kT, because one
added vector often jumps across a cluster of near-degenerate box states. Since box states scale as L³, a box a few
per cent larger than 14 Bohr can push S at M = 5 over the cap; the scan then leaves that multiplicity out, and every
winner needs ≤ 10.

**Cost.** Mean cold-solve time, FD against DS (GPU shared with the user's session, so indicative): 2p atoms and the
molecules 1–5× (H₂O 15 s against 8 s at 95³; O₂ triplet 34 against 7); the 3p atoms 5–80× (Si at M = 5: 80 s against
1–2 s; S at M = 3, 0.15: 60 against 4; Cl at M = 4, 0.15: 227 against 47). The time is the one-vector-at-a-time growth,
each growth re-solving the block; warm solves keep the block and take 1 step.

**Verdict on the arm: no change is needed for V0.** The options, if the user wants margin or speed, are in spec §4.2
(a larger cap; multi-vector growth; or a 25 kT arm derived from the stop, whose neglected charge is ≈ 4e-12 e), none
of them a tolerance change.

### D. The noise floor at 5 mHa, and the 1e-5 margin

The f32 floor (median Δρ of warm steps at tol 1e-12, tight eigen steps) against the 1e-5 stop:
- **Gapped states** (O₂ triplet, H₂O, NH at M = 3, N and P at M = 4, Si at M = 5, all-electron O at M = 1): ≤ 3e-10, or
  converged at once. Margin > 3e4.
- **One open channel, three levels at f = ⅓ or ⅔** (the winners of B, C, O, F, Si, S, Cl; F and Cl at M = 4), and OH:
  medians 5e-8 to 2.2e-6; margin 4.5–150×. One exception: Si at M = 3 on Accurate, generic site (6.2e-6).
- **Two open channels or two half-filled pairs** (O, C, Si, S at M = 1; N and P at M = 2; the O₂ singlet and NH at
  auto) and all-electron O at M = 5: medians **1e-6 to 2.8e-5**, single steps up to 9.5e-5. The jitter rule fails at
  most of their sites. The 1e-5 stop is still reached within the 60-iteration rule at every site except S at M = 1
  (61) and all-electron O at M = 5, because the floor fluctuates and one step below 1e-5 ends the solve.
- The floor is not stationary across runs: O at M = 1, h_req, node had a median of 1.7e-6 in the 1e-3 study's 5 mHa
  row (§4) and 6.4e-6 here.

**Is 1e-5 a safe margin?** For gapped and single-open-channel states (every scan winner, OH, H₂O): yes, ≥ 4× and
usually ≥ 20×. For the M = 1-type states it is not a factor-3 margin; it is a first-passage stop that converged in
9–61 iterations here. None of these is a scan winner, and a failed multiplicity is left out of the scan, but the O₂
singlet and NH at auto are scene states a user can build, so a live solve of them can take ~40 iterations and
occasionally hit the stall stop. A factor-3 margin over the worst median (2.8e-5) would need a stop near 1e-4, which I
do not propose: it is the user's value, the energy and
force cost of 1e-5 is already below the f32 scatter, and the remedy for the noise is an FD-aware stop criterion or a
tighter final eigensolve (§9 above), not a looser Δρ.

### B. Where the 1e-5 stop applies (my ruling), and what it costs

**Ruling.** On the GPU, under FD, a step is converged at `max(tol_eff, 1e-5)` when its occupations have a partial
level (min(f, 1 − f) ≥ 1e-6 in either channel), and at `tol_eff` otherwise. `tol_eff` is today's
`effective_gpu_tol` (max(tol, 2·5e-8·N)) under an ionic setup, `tol` otherwise. The CPU f64 path keeps `tol`. It
applies to every GPU SCF, not only the reference scan. The spec's section 5.1 states it; ruling 42 records it.

**Why every partial-level solve and not only the scan.** The floor belongs to the partly occupied level, not to
the lone atom. The molecules show it:
- the O₂ singlet at auto and NH at auto (two half-filled pairs, TS = 4 kT ln 2) have floor medians up to
  2.8e-5 and 9.6e-6, and a warm re-solve to the sample tolerance 1e-6 from the floor took 1–40 iterations, with one
  failure at 40 (O₂ at h = 0.20, best 1.0e-6);
- the lone M = 1 atoms reached 1e-6 at almost no site in the 1e-3 study (§2, §3 above).

The sample teacher (1e-6) would stall on exactly these states in a collection, the migration re-solves at 1e-6,
and an O₂ or OH in the user's H/N/O collection is a real case. Restricting the stop to the scan would leave those to
the stall stop.

**Why not every FD solve.** A gapped solve has no FD floor: H₂O, the O₂ triplet, NH at M = 3 and the gapped atoms reach
1e-6 in one warm step (their noise runs sit at Δρ ≤ 3e-10). Keeping their stop at 1e-6 keeps every gapped pin and
label exactly as today.

**Interaction with the existing rule and the provenance.**
- `effective_gpu_tol` is unchanged and still applied first; the partial-level 1e-5 is a further floor on top of it.
  At N ≤ 100 valence electrons the storage floor term (2·5e-8·N ≤ 1e-5) never exceeds the new one.
- `Provenance::tol` and `ReferenceKey::tol` keep the nominal tolerance, as they already do for `effective_gpu_tol`.
- The rule is named in the tag: `…/fd5e-3/stop2x5e-8/pin1e-5` and `none/fd5e-3/pin1e-5`.
- `ScfStats` reports the tolerance the converged step was tested against, so `EvalStats::scf_tol_used` is 1e-5 for
  such a solve and `branch_jump` uses the 1e-5 noise tier for it.

**Effect on energies and forces.**
- *Estimate.* F is stationary in ρ, so a residual δρ of L1 norm 1e-5 changes F at second order: roughly its Hartree
  self-energy over a ~1 Bohr region, ‖δρ‖²/R ≈ 1e-10 Ha. The force on nucleus I is first order: at most
  ‖δρ‖₁·max|∇V_loc,I| ≈ 1e-5 × 2 Ha/Bohr = 2e-5 Ha/Bohr for pseudo-O (Z_v = 6, r_c ≈ 1.1 Bohr), a strict upper bound.
- *Measured* (cold at 1e-5 against a warm re-solve to 3e-6 from the floor state, every FD case above):
  - pseudo surface, |ΔF| ≤ 1.7 µHa (F at M = 2); |ΔF_I| ≤ 8.8e-7 Ha/Bohr per component on every solve with a partial
    level (OH off-axis), ≤ 3.2e-6 including gapped ones (P at M = 4, generic site, whose cold stop was at
    Δρ = 9.1e-6);
  - all-electron O, |ΔF| ≤ 2.9 µHa, |ΔF_I| ≤ 2.1e-6 Ha/Bohr;
  - the warm/cold path scatter σ_path for comparison: 0.01–0.7 µHa pseudo, 0.5–1.1 µHa all-electron.
- So the stop error is not resolved above the f32 path scatter in F (the 1e-10 Ha estimate is far below it), and the
  forces' sits ≥ 6× under the linear bound (≥ 20× on the partial-level solves). Against the surface's tolerances:
  ≥ 15× under `PSEUDO_FORCE_NOISE` (5e-5; ≥ 57× on the partial-level solves), ≥ 47× under the migration's
  ε_F (1.5e-4), and ≤ ε_E/3 in energy.

### E. Not measured, or not verified

- **The CPU f64 cross-check at 5 mHa** of the worst cases (S at M = 1 on Accurate, all-electron O at M = 5) was not
  run, to keep the PC usable; whether S M = 1's 30-step plateau is f32 noise or a slow physical mode is therefore
  not separated. The 1e-3 study's CPU check (§5.2) showed the f32 floor is what held O at M = 1 at 2 mHa.
- **The lopsided-tetrahedron seeding** (spec §4.7(e)) and **a bent radical**, if "OH (bent)" meant one.
- **Statistics:** one cold solve per site and case; the noise-limited iteration counts are stochastic (§1.3), and the
  floor medians move by up to 4× between runs of the same case.
- **The integer-limit U** was not re-measured; §5.1's values stand.
- **Wall times** are indicative: the GPU was shared with the user's session and, after a pause for a suspected GPU
  overheating, the last batches (all-electron O at M = 5, P, C, N, B) ran one at a time with 3-minute gaps.

### F. Reproducing

`cargo test -p providers --release v0_cases -- --ignored --nocapture` (uncommitted harness), with `V0_Z` or `V0_MOL`,
`V0_POLICY`, `V0_GRIDS`, `V0_MS`, `V0_MODES` (default `ds,fd5e-3`), `V0_SITES`, `V0_TOL` (1e-5), `V0_TIGHT` (3e-6),
`V0_TIGHT2` (molecules: 1e-6), `V0_NOISE`, `V0_NOISE_STEPS` (15), `V0_MAXIT` (100). Batches: O, Si, S, Cl, F, the
molecules, all-electron O (M = 1, 3 and then M = 5), P (M = 2, M = 4), C (M = 1, 3, then 5), N and B; about 520 GPU
solves (cold, warm, noise and tight), ≈ 4.5 h of GPU time.
