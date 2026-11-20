# SCF stall on small H clusters: diagnosis (2026-11-13)

User report: under the app's defaults (GPU, Accurate h = σ = 0.15, `Accepted` ionic policy), a cluster of loosely
bound H atoms keeps Δρ ≈ 0.3 with the iteration energy swinging ~30 mHa, and the stall stop ends the solve after
50 iterations. This note reproduces that headlessly, finds the cause, and tests the proposed fix. **No committed
solver behaviour was changed.** The diagnostic instrumentation (below) lives only in the `scf-stall` worktree,
uncommitted.

## 1. Code map: how occupations, mixing and the stall stop work today

The default policy matters. Under `Accepted` with a GPU, **H is pseudised** (`crates/providers/src/pseudo.rs:210`).
Whenever anything is pseudised, `providers::ks::problem` switches the whole scene to
`Occupations::DegenerateShells { τ = 1e-3 }` (`crates/providers/src/ks.rs:356`, `:384`). So every all-H scene the user
builds in the app runs the `DegenerateShells` rule, not integer aufbau. All-electron scenes (`--ionic none`) run
`Aufbau`.

| What | Where (HEAD line numbers) |
|---|---|
| `Occupations` enum, `DEGENERACY_TAU` = 1e-3, `CANDIDATE_TAU` (τ_c) = 0.1 Ha, `MAX_WITNESS_GROWTH` = 16 | `crates/qm/src/scf.rs:251`, `:255`, `:258`, `:263` |
| Phase 1 rule `degenerate_shell_occupations`: levels within τ_c of ε_F (the N_σ-th level) form a cluster. If the cluster straddles ε_F it is shared **equally**. | `crates/qm/src/scf.rs:275-296` |
| Phase 2 `decide` / `released_occupations`: at the first convergence, a shared cluster with spread ≥ τ is "released" (aufbau over groups split at gaps ≥ τ). The pattern **by rank** is then frozen, the DIIS history is cleared, and the SCF converges again. | `crates/qm/src/scf.rs:789-799`, `:301-316`; CPU loop `:906-920`; GPU loop `crates/qm-gpu/src/scf.rs:1064-1079` |
| Per-channel eigensolve plus witness growth (GPU `Channel::solve_shared`; CPU `solve_shared`) | `crates/qm-gpu/src/scf.rs:641-666`; `crates/qm/src/scf.rs:739-760` |
| Mixer: Pulay/DIIS on the stacked spin densities. History m = 6, step α = 0.3 along the minimised residual (`ρ_new = Σcᵢ(ρᵢ + α rᵢ)`), clamped at 0, charge renormalised per channel. No Kerker preconditioner. | `crates/qm/src/scf.rs:37-62`, `:371-374` (`for_grid`: α 0.3, m 6, tol 1e-6 / app 1e-5, 200 iterations); GPU `Diis` `crates/qm-gpu/src/scf.rs:393-457`, mix call `:1101` |
| Eigen tolerance tied to Δρ (`EIG_TOL_FRACTION` 0.0015, `EIG_TOL_MAX` 1e-3). The DIIS history is reset when frozen orbitals are re-solved. | `crates/qm-gpu/src/scf.rs:259-276`, `:1087-1097` |
| Stall stop (`StallWatch`): stop when the best Δρ has not halved in W = 50 iterations and best Δρ ≥ 1e-3·N | `crates/qm/src/scf.rs:394`, `:406`, `:427-472`; GPU `crates/qm-gpu/src/scf.rs:1080-1086` |
| Spin: n↑, n↓ fixed per solve (`multiplicity` None ⇒ lowest). The cold guess splits ρ↑ : ρ↓ = n↑ : n↓. Nothing moves electrons between channels. | `crates/qm/src/scf.rs:324-336` |

The CPU (`qm::scf`) and GPU (`qm_gpu::scf`) implement the same rule step for step (the GPU imports `decide`,
`ascending`, `shared_runs` and `degenerate_shell_occupations`).

## 2. Reproduction

Diagnostic harness, uncommitted: `crates/providers/src/scf_stall_diag.rs` (registered as `#[cfg(test)] mod` in
`crates/providers/src/lib.rs`, `#[ignore]`). Run it with
`cargo test -p providers --release scf_stall_diag -- --ignored --nocapture`.

It builds each scene through `providers::ks::problem` exactly as `GpuKs` does, with the app's interactive tolerance
1e-5 raised by `effective_gpu_tol` under a setup. Grid 81³ × 0.15 (the 12 Bohr preset box at Accurate). Each solve
starts cold, with the stall stop at its default W = 50. The GPU dimer backend and the user's disk cache are set, as
in the app. Environment knobs: `DIAG_POLICY`, `DIAG_CASES`, `DIAG_MIX`, `DIAG_HIST`, `DIAG_STALL`, `DIAG_DS`, `DIAG_N/H`,
`DIAG_CPU`.

Two env-gated additions in `crates/qm-gpu/src/scf.rs`, also uncommitted:
- `SCF_TRACE2` prints, per iteration, Δρ (per spin), the Harris–Foulkes energy, the DIIS depth and every level
  ε(f) per spin.
- `SCF_SMEAR=kT` replaces the phase-1 occupations with Fermi–Dirac at kT, with μ bisected per spin channel (fixed
  N↑, N↓). No shell is reported, so no release happens.

### Geometries (Bohr, before centring; the harness centres each cluster on its centroid)

- H₄ square a = 1.8, 2.5; rectangle 1.4 × 2.2; **rhombus** side 1.8, 60°: (0,0,0), (1.8,0,0), (2.7,1.5588,0), (0.9,1.5588,0)
- H₄ chains d = 1.6, 2.0; **H₄ lopsided tetrahedron**: (0,0,0), (1.9,0,0), (0.7,1.7,0), (0.9,0.6,1.8)
- H₅ chain 1.8, H₅ ring side 1.8; H₆ chain 1.8, rings side 1.8 / 2.2
- Random "loosely bound" clusters: each new atom is attached to a random existing one at 1.5–2.5 Bohr, with every
  pair ≥ 1.5 Bohr (`random_cluster`, seeds 1–8, H₄–H₈; coordinates in §2.1)
- H₂O + 2 H: O (0,0,0), H (±1.43, 1.11, 0), H (0.5, −2.2, 1.0), H (−0.9, −2.0, −1.6)
- Controls: H₂ at 1.434 Bohr, H₂O preset

### 2.1 Centred coordinates as solved (Bohr; all Z = 1 unless marked)

- H₄ rhombus: (−1.350, −0.779, 0), (0.450, −0.779, 0), (1.350, 0.779, 0), (−0.450, 0.779, 0)
- H₄ lopsided tetra: (−0.875, −0.575, −0.450), (1.025, −0.575, −0.450), (−0.175, 1.125, −0.450), (0.025, 0.025, 1.350)
- H₄ random seed 1: (0.764, −0.857, −0.684), (−0.861, 0.614, −0.003), (1.272, 1.544, 0.249), (−1.175, −1.300, 0.438)
- H₅ random seed 3: (1.144, −0.101, −1.151), (0.081, 1.439, −0.157), (−0.899, 0.101, −1.136), (−0.476, −0.313, 0.515), (0.150, −1.126, 1.929)
- H₈ random seed 8: (0.528, 0.790, −1.058), (0.502, 1.743, 0.900), (1.709, 1.027, 0.330), (0.194, −0.661, 1.061), (−1.519, −2.179, 1.325), (0.222, −2.303, 1.030), (−0.815, 1.811, −1.118), (−0.821, −0.227, −2.470)
- H₆ random seed 5 (all-electron failure): (0.324, −1.661, 1.023), (−0.220, −0.273, 2.268), (1.768, −2.667, 0.858), (0.677, 0.067, −0.331), (−0.621, 1.375, −1.330), (−1.929, 3.159, −2.488)
- H₅ ring side 1.8 (all-electron failure): radius 1.531, in the xy plane
- H₂O + 2 H (all-electron failure): O (0.080, 0.396, 0.120), H (1.510, 1.506, 0.120), (−1.350, 1.506, 0.120), (0.580, −1.804, 1.120), (−0.820, −1.604, −1.480)

### Survey, default app settings (`Accepted` → pseudo-H, `DegenerateShells`), GPU 81³ × 0.15, cold

| case | outcome | iterations | E (Ha) / last Δρ, E_HF swing |
|---|---|---|---|
| H₄ square 1.8 / 2.5 | conv (symmetric pair shared, spread 1e-8) | 10 / 11 | −1.999443 / −2.028310 |
| H₄ rectangle, chains 1.6 / 2.0 | conv | 9, 10, 10 | |
| **H₄ rhombus** | **STALL** | 59 | Δρ 0.43–1.04, E_HF −1.981 ↔ −2.086 (**100 mHa**) |
| **H₄ lopsided tetra** | **STALL** | 61 | Δρ 1.2–2.1 (best 0.92) |
| H₅ chain, ring; H₆ chain, rings | conv | 9–12 | |
| **H₄ random seed 1** | **STALL** | 62 | Δρ 0.2–1.4, E_HF −2.008 ↔ −2.111 |
| H₄ random seed 2 | conv | 44 | |
| **H₅ random seed 3** | **STALL** | 57 | Δρ 0.21–0.30, E_HF −2.5145 ↔ −2.5520 (**37 mHa**, matches the screenshot) |
| H₅ seed 4, H₆ seeds 5/6, H₇ seed 7 | conv **after a shell release** | 32, 55, 29, 44 | |
| **H₈ random seed 8** | **STALL** | 70 | Δρ 0.037 ↔ 0.78, E_HF −4.1067 ↔ −4.1450 (38 mHa) |
| H₂O + 2 H | conv after a release | 32 | −18.402048 |
| H₂ / H₂O controls | conv | 9 / 13 | −1.136119 / −17.476023 |

**5 of 23 stall** (all 5 are lopsided clusters; every symmetric chain and ring converges), and 6 more need a release
and 30–55 iterations. The symptom matches the report: Δρ of order 0.2–1, and a period-2/3 energy oscillation of
30–100 mHa.

All-electron (`--ionic none`, `Aufbau`), same grid: a **different** set fails. 3 of 23 fail: H₅ ring stalls (Δρ
0.25–1.2), H₆ seed 5 hits the 200 cap at Δρ ≈ 1e-3 (below the stall level), and H₂O + 2 H stalls (best 0.76). The 5
cases above converge under aufbau in 10–15 iterations. The lopsided tetra takes 56.

Coarse grid 41³ × 0.30 (pseudo-H, GPU): rhombus, seeds 1, 2 and 3 stall, seed 5 hits the cap, and the lopsided tetra
converges after a release in 78 iterations. The failures are not specific to the fine grid.

## 3. Per-iteration evidence (SCF_TRACE2)

`occ` lists the occupations by ascending level per spin: `1` full, `0` empty, `s` shared. `gap` = ε_{N+1} − ε_N.
"edge" = the distance of the nearest level from the τ_c window edge |ε − ε_F| = 0.1 Ha.

**H₄ rhombus** (n↑ = n↓ = 2). Levels ↑ = ↓: the shared iterations look like it 11 (−0.6158, −0.1891 (0.5),
−0.0903 (0.5), +0.0355), with the HOMO–LUMO pair 0.0988 Ha apart, inside τ_c, so both levels hold ½. The integer
iterations look like it 30 (−0.6167, −0.1907 (1), −0.0907 (0), +0.0350), with a gap of **0.1000–0.1015** Ha, outside
τ_c, so aufbau applies. The pattern flips 1ss0 ↔ 1100 throughout: iterations 3–5 s, 6–7 integer, 8 s, 9–10 integer,
11–13 s, 14 integer, 15–16 s, … The energy alternates −1.982 (shared) ↔ −2.081 (integer). Δρ never drops below 0.41,
even across 5 consecutive integer iterations (27–31), because the DIIS history holds residuals from the other branch.

**H₄ random seed 1**: the same pattern. The gap sits at 0.0921–0.1070, edge distance 0.0000–0.007 Ha.
1ss0 ↔ 1100, E −2.0085 ↔ −2.111.

**H₅ random seed 3** (n↑ 3, n↓ 2). In the ↑ channel the level two below ε_F sits at the **lower** window edge:
ε_F = −0.1616 and the level at −0.2607 to −0.2625 is 0.0991–0.1009 Ha below it. When that level is inside, the cluster
is 3 levels sharing 2 electrons (f = ⅔ each). When it is outside, the cluster is 2 levels sharing 1 electron (½
each). The pattern alternates 1sss0 (⅔) ↔ 11ss0 (½), E −2.5145 ↔ −2.5520 (37 mHa).

**H₈ random seed 8** (n↑ = n↓ = 4). A level sits at the **upper** edge, ε_F + 0.0999…0.1002, edge distance
0.0000–0.0007 Ha for 70 iterations. Three shared levels (111sss0) alternate with two (111ss00), Δρ 0.04 ↔ 0.78.

**H₄ lopsided tetra**, a different mechanism, at phase 2. Phase 1 converges cleanly (Δρ 5.0e-6 at it 10) with a
3-level ↑/↓ cluster sharing 1 electron (f = ⅓; levels −0.1488, −0.1348, −0.0975, spread 0.051 ≥ τ). The cluster is
released to integer aufbau "11000", frozen by rank. Two things follow:
1. **Iterations 11–16: ρ_in does not move at all.** Δρ stays at 1.665 and E_HF at −1.948271 to every printed digit,
   for exactly `diis_history` = 6 iterations. The release clears the DIIS history and *then* pushes the phase-1 entry
   (ρ_in at convergence, residual 5e-6 under the old occupations). With the new occupations every later residual is
   ≈ 1.67. DIIS minimises ‖Σcᵢrᵢ‖, so it puts c ≈ 1 on the stale entry and returns the same ρ_in until that entry
   ages out of the history. The CPU has the same order (`crates/qm/src/scf.rs:915` then `:927`). That wastes 6
   iterations on every release; on its own it does not cause the stall.
2. From it 17 on, integer aufbau has no stable fixed point from this path. The frozen-by-rank HOMO/LUMO gap swings
   from 0.0007 to 0.08 Ha, the orbitals swap character between ranks (the aufbau flip in disguise), and Δρ stays at
   1.2–3.0 until the stall stop.

## 4. Root cause

**The `DegenerateShells` phase-1 occupation rule is a discontinuous function of the eigenvalues at
|ε − ε_F| = τ_c = 0.1 Ha.** Whenever a self-consistent level of a low-symmetry cluster lies within ~1e-3 Ha of either
window edge, the occupations jump between iterations: ½ ↔ 1/0 at the upper edge, ⅔ ↔ ½ at the lower edge, 3 ↔ 2
shared levels. Neither branch is self-consistent. Each branch's density pushes that level back across the edge,
which is the "no fixed point" situation. A 0.1 Ha gap is common in loosely bound H clusters; their HOMO–LUMO gaps
are 0.03–0.15 Ha. Symmetric chains and rings rarely land a level at exactly 0.1 Ha and converge. Lopsided clusters,
which is what a user builds by hand, regularly do.

The equal-share rule also weights levels up to 0.1 Ha apart identically (a kT of ~30 000 K). That is unphysical as an
intermediate state, and the release then has to undo it. When the true state has a frontier gap ≲ kT (the lopsided
tetra), the phase-2 integer pattern is the classic aufbau level-flipping problem. The secondary DIIS-after-release
ordering defect (§3) adds 6 dead iterations to every release.

Ruled out or secondary:
- **The mixer is not the cause.** On the 5 stallers, with the stall stop off and the cap at 200: α = 0.1 (DIIS 6)
  converges 1 of 5 (the lopsided tetra, 85 iterations). α = 0.05 converges 1 of 5 (the lopsided tetra, 93 iterations,
  at a *different* energy, −1.956852 vs −1.944984). Linear mixing at α = 0.1 converges 0 of 5. The rhombus, seed 1,
  seed 3 and seed 8 oscillate with the same period at every α: the occupation flip is a discontinuity no step size
  removes.
- **Spin flipping between channels: no.** n↑ and n↓ are fixed per solve, and in every staller ↑ and ↓ flip together
  (n↑ = n↓, symmetric guess) or only the open channel flips (H₅ seed 3: ↓ is integer with Δρ↓ ≈ 2e-3, while ↑
  carries Δρ↑ ≈ 0.23).
- **Eigensolver: no.** The CPU reference, which uses a fixed tight eig_tol of 1e-6 at every step, stalls the same way
  (§5). On the GPU the levels are stable to ~1e-4 Ha between iterations on the same branch. The traces show the
  occupations changing, not the eigenvalues jumping.
- **Hartree sloshing / box size: no.** Symmetric clusters of the same size and box converge in 9–12 iterations, and a
  gentler mixer does not help.
- **Warm vs cold: no.** Every reproduction is a cold solve. A warm start cannot help, because the rule decides on each
  step's levels in phase 1.
- **GPU-specific: no** (see §5, the CPU run). The rule is shared code (`qm::scf::degenerate_shell_occupations`), and
  the CPU applies it the same way.

## 5. CPU check (f64 reference, 41³ × 0.30, pseudo-H, `DegenerateShells`)

The same scenes go through `qm::scf_observed`, f64, with the fixed eig_tol 1e-6 (no Δρ-tied loosening) and the same
settings:

| case | GPU 41³ | CPU 41³ |
|---|---|---|
| H₄ chain 1.6 (control) | conv 10, E −2.221083 | conv 9, E −2.221083 |
| H₄ rhombus | STALL 56, best Δρ 0.42 | **STALL 57, best Δρ 0.42**, E_HF −1.982 ↔ −2.083 |
| H₄ random seed 1 | STALL 57, best 0.17 | **STALL 57, best 0.17**, E_HF −2.010 ↔ −2.118 |
| H₅ random seed 3 | STALL 82, best 0.12 | **STALL 57, best 0.21**, E_HF −2.515 ↔ −2.553 |

The CPU reference stalls with the same best Δρ and the same energy branches. **The defect is in the shared occupation
rule, not in the GPU or f32 numerics.** The CPU also runs every step at its tight eigen tolerance, so the
eigensolver's Δρ-tied loosening is ruled out too. (CPU: 130–180 s per stalled solve.)

## 6. Experiment: Fermi–Dirac smearing

`SCF_SMEAR=1e-3` Ha (≈ 316 K). FD occupations replace the phase-1 equal share, μ is bisected per spin channel (fixed
N↑, N↓), and there is no release. The witness vectors that `DegenerateShells` already grows reach at least ε_F + 0.1 Ha
= ε_F + 100 kT, so the FD tail is complete. The Mermin free energy F = E − TS uses
S = −Σ[f ln f + (1−f) ln(1−f)].

### Default policy (pseudo-H, 81³ × 0.15): all 23 converge

| case | before (iterations, outcome) | smeared: iterations | E (Ha) | F = E − TS | TS (Ha) |
|---|---|---|---|---|---|
| H₄ rhombus | 59, STALL | **10** | −2.079319 | −2.079319 | 0 |
| H₄ lopsided tetra | 61, STALL | **32** | −1.930104 | −1.932837 | 2.7e-3 |
| H₄ random seed 1 | 62, STALL | **12** | −2.078114 | −2.078114 | 1e-9 |
| H₅ random seed 3 | 57, STALL | **18** | −2.563330 | −2.563330 | 2e-7 |
| H₈ random seed 8 | 70, STALL | **14** | −4.220373 | −4.220373 | 2e-16 |
| H₅ seed 4 (release before) | 32 | 14 | −2.655883 (same) | | 0 |
| H₆ seed 5 (release before) | 55 | 31 | −3.147893 (−3.148291 before) | −3.148504 | 6.1e-4 |
| H₆ seed 6 (release) | 29 | 12 | −3.093191 (same) | | 2e-8 |
| H₇ seed 7 (release) | 44 | 17 | −3.719251 (−3.719250) | | 2e-14 |
| H₂O + 2 H (release) | 32 | 16 | −18.402048 (same) | | 4e-16 |
| H₄ square 1.8 / 2.5 (exactly degenerate, shared before) | 10 / 11 | 14 / 14 | **unchanged** −1.999443 / −2.028310 | −2.002216 / −2.031082 | 2.77e-3 (= 4 kT ln 2) |
| H₅ ring (degenerate ↓ pair) | 10 | 11 | unchanged −2.638384 | −2.639770 | 1.39e-3 (= 2 kT ln 2) |
| other converging cases | 9–12 | same counts | **identical E** | | 0 |
| **H₂ control** | 9 | 9 | −1.136119 → −1.136119 (**ΔE = 0**) | | 0 |
| **H₂O control** | 13 | 13 | −17.476023 → −17.476022 (**1e-6**, the SCF tolerance) | | 0 |

(The wall times are not comparable between the surveys, because the GPU was shared with the user's session. Compare
iteration counts. The FD bisection itself is a few µs per step.)

Points to note:
- **Four of the five stallers converge to an integer-occupied state with a real gap** (TS ≤ 2e-7). The aufbau
  answer existed all along; the τ_c window rule kept the SCF away from it. Smearing at 316 K changes nothing in a
  gapped system: H₂ and H₂O are bit-for-bit or within the SCF tolerance, and TS is 0 to machine precision.
- **Exactly degenerate shells.** The internal energy E equals the `DegenerateShells` ensemble energy, because FD gives
  the same ½/½. F differs by TS = kT·N_deg·ln 2 (2.8 mHa for the H₄ square). Which number gets reported is a choice
  (§7).
- **Smearing alone does not make the SCF find the lowest solution.** For the lopsided tetra, smearing converges to a
  spin-symmetric state with the frontier pair pinned at the Fermi level (−0.1296 / −0.1290 Ha, f = 0.57 / 0.43) at
  F = −1.932837. The no-smearing α = 0.1 run instead reaches a spin-broken integer state (↑ and ↓ levels differ, gap
  0.031 / 0.027 Ha) at E = −1.944984 (12 mHa lower), and α = 0.05 another at −1.956852. This is LSDA's
  broken-symmetry / static-correlation regime (a diradical-like H₄), and it is physics rather than a numerical
  artefact. A spin-symmetric cold guess (ρ↑ = ρ↓ when n↑ = n↓) stays spin-symmetric under smearing. Rule 7: the gap
  between the restricted fractional state and the broken-symmetry integer one should be documented, not hidden. A
  symmetry-breaking nudge in the cold guess is a separate decision.

### All-electron (`--ionic none` with `DegenerateShells` for the witness vectors), 81³ × 0.15

| case | Aufbau (today) | DS, no smearing | DS + FD 1e-3 | E / F with smearing |
|---|---|---|---|---|
| H₅ ring | STALL 55 | conv 11 (shares the degenerate ↓ pair) | conv 13 | E −2.463750 / F −2.465136 |
| H₆ random seed 5 | cap 200 (Δρ 1.2e-3) | conv 54 (2 releases) | conv **47** | E −2.966631 / F −2.967311 (integer −2.967056) |
| H₂O + 2 H | STALL 64 | STALL 61 | conv **79** | E −47.337254 / F −47.339721, TS 2.5e-3 |
| H₄ lopsided tetra | conv 56 | conv 70 | conv **28** | E −1.811214 / F −1.813947 (integer −1.826310, again lower) |

Smearing fixes the all-electron aufbau stalls too. That is the textbook level-flipping case: H₅ ring has a degenerate
e-pair holding one ↓ electron under integer aufbau. All-electron H₂O + 2 H converges, but slowly (79 iterations):
its frontier is pinned (TS 2.5e-3), and α = 0.3 DIIS without Kerker takes a while. That is acceptable, but noted.

## 7. Recommended fix and its physics

**Replace the phase-1 equal share inside the τ_c window (and the phase-2 release) with Fermi–Dirac occupations at a
small fixed electronic temperature, kT = 1e-3 Ha, μ_σ by bisection per spin channel. Report and minimise the Mermin
free energy F = E − TS, keeping the existing witness growth (at least ε_F + τ_c = 100 kT) as the FD tail.** Use one
rule for all-electron and pseudo scenes (Aufbau then becomes the kT → 0 limit).

Physics consequences:
- **Variational functional and forces.** With FD occupations the stationary functional is the Mermin free energy
  F[{ψ},{f}] = E_KS[{ψ},{f}] − kT·S[{f}] at fixed N_σ. ∂F/∂f_i = ε_i − μ_σ + kT ln(f_i/(1−f_i)) = 0 is exactly the FD
  condition, and ∂F/∂ψ = 0 gives the KS equations. F is therefore stationary in both ψ and f, and the
  Hellmann–Feynman force −dF/dR = −∂F/∂R|explicit is exactly what the existing force code computes. That code already
  weights the local density and the nonlocal projector sums by f_iσ (`crates/qm/src/forces.rs:87`, `:236`;
  `crates/qm-gpu/src/forces.rs:268`); no Pulay-like occupation term appears. The forces are the derivative of **F,
  not E**. With F as the potential energy, BO-MD conserves F + K_nuc, the surrogate's energy and force labels stay
  consistent (one surface), and relax and FIRE minimise F. Reporting E with these forces would be inconsistent at the
  level of d(TS)/dR (0 in gapped systems, ~mHa/Bohr when a level crosses the Fermi level).
- **Energy shift.** kT = 1e-3: 0 on every gapped system measured (H₂, H₂O, chains, rings, 4 of 5 stallers: TS ≤ 2e-7
  Ha). In an exactly degenerate shell F = E_ensemble − kT·N ln 2 (2.8 mHa for the H₄ square's two pairs, 1.4 mHa for
  one pair). If the readout must stay comparable with today's numbers, show E_0 ≈ ½(E + F) (Gillan's kT → 0
  estimate). It is exact to O(T²) for a smooth density of states but off by TS/2 for an exact degeneracy. The cleanest
  honest readout is F, labelled with T_el, with E alongside.
- **What changes vs `DegenerateShells`.** For a symmetry-degenerate shell (lone pseudo-O, M = 1 or 3) FD gives the
  same equal-weight ensemble (exact degeneracy ⇒ equal f), plus the −TS term. The lattice-split pairs (≤ 5.3e-5 Ha
  spread) get f within 1.3 % of equal at kT = 1e-3 instead of exactly equal. The spec §5.7 argument
  ("levels within τ are thermally co-occupied at the sandbox's temperatures") is the FD rule itself, made continuous.
  The two-phase decision, the release and the "pattern frozen by rank" logic disappear, and with them the 6-iteration
  DIIS dead time after a release and the path dependence (the lopsided tetra converged to −1.9450 or −1.9569
  depending on α).
- **Choice of kT.** It should be a solver constant, a smearing width rather than per-element data. 1e-3 Ha already
  matches `DEGENERACY_TAU`'s 300 K justification. Larger kT converges more robustly but biases F in metallic/pinned
  cases. Smaller kT approaches today's discontinuity: the FD slope 1/(4kT) is what DIIS has to handle. Gaussian or
  Methfessel–Paxton smearing would cut the E(T) bias for pinned states, but FD is the one with a true variational F
  and an entropy with physical meaning (it is the electronic temperature). Recommend FD.
- **Spin.** Per-channel μ_σ keeps the user's multiplicity (fixed-spin-moment). A joint μ would let the moment float.
  That is physical but changes what `multiplicity` means and the solver's counts and normalisation; leave it for later.
- **Broken symmetry is not addressed.** Smearing converges to the nearest solution. For near-degenerate clusters
  (lopsided H₄) that can be the spin-restricted fractional state, 12 mHa above an LSDA broken-symmetry state. Document
  this. A separate decision is whether the cold guess should carry a small spin perturbation.

Secondary fix, independent of smearing: at a release (if the two-phase rule is kept for any reason), clear the DIIS
history **after** recording the step, or drop the current entry, so the first post-release mix starts from the new
map. Both solvers have the same order (`crates/qm/src/scf.rs:915`/`:927`, `crates/qm-gpu/src/scf.rs:1073`/`:1098`).

### Validation the fix would need

1. A regression test (this harness reduced): the 5 stallers and the 3 all-electron failures converge within, say, 40
   iterations on the GPU at 81³ × 0.15. The rhombus, seed 1 and seed 3 are cheap, ~1 s each when they converge.
2. `qm` unit suite plus `validate --test oracles` and `--test gpu_vs_cpu` (a solver change). Every σ → 0 oracle is a
   gapped closed shell or a single-electron atom, so F = E to ~1e-16 and they should not move. The lone pseudo-atom
   tests (`pseudo_o_triplet_matches_cpu`, `degenerate_spread_at_low_symmetry_sites`, the O reference scan
   `gpu_oxygen_reference_multiplicity`) **will** move by −kT·N ln 2 if F is reported. Their references need the TS
   term stated in `references.md` (never a loosened tolerance). The bitwise pins (`gpu_none_policy_is_bitwise_today`,
   the stall-stop bitwise tests) change by construction if Aufbau becomes FD; re-pin with a documented reason, or
   keep Aufbau bitwise as the kT = 0 branch.
3. A new physics oracle for consistency: a finite-difference check that F(R + δ) − F(R − δ) ≈ −2δ·F_HF on a smeared,
   pinned case (the lopsided H₄, or stretched H₂ near the Coulson–Fischer point). Also an NVE run on a cluster with a
   level crossing the Fermi level, checking that F + K is conserved while E + K is not.
4. `--test dynamics` (the energy that is conserved changes to F), `--test pseudo`, `--test pseudo_oracles` (the lone-O
   ensemble), and `--test surrogate` (labels become F; the provenance tag's "shared τ/τ_c" ionic tag must change, so
   old models are refused, never mixed).
5. CPU ↔ GPU parity of the smeared rule (`gpu_vs_cpu`), including μ bisection on the f64 side.
6. Docs: `qm::scf` "Occupations" rewritten, spec §5.7 amended, README and readout label ("free energy F at T_el =
   316 K"), `references.md` gaps (TS of degenerate shells; the restricted vs broken-symmetry H₄ gap).

## Files touched (uncommitted, worktree only)

- `crates/qm-gpu/src/scf.rs`: `SCF_SMEAR` (FD in `Channel::solve_shared` phase 1) and `SCF_TRACE2` (per-iteration
  levels and occupations), both env-gated diagnostics.
- `crates/providers/src/scf_stall_diag.rs` + `mod` line in `crates/providers/src/lib.rs`: the `#[ignore]` harness.
