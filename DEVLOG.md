# DEVLOG: how Atomic Sandbox was built

This is the build history of Atomic Sandbox, written for people who want to know how a project like this
came together when the code is written by Claude agents and steered by one human. It is a log, not a
sales page. The wins are here, and so are the dead ends, the mistakes and the things that were measured
and turned out worse than hoped.

Where a number appears, it comes from git, a handoff note, a build ledger or `crates/validate/references.md`.
Where something is my estimate, it says so. All times are local (UTC+10), as git recorded them.

---

## What this is

Atomic Sandbox is an interactive 3D sandbox that simulates matter from the atomic level up. Nuclei are
classical particles. Electrons are solved with real-space Kohn-Sham density functional theory (LSDA) on a
GPU grid, and the nuclei move under the resulting forces (Born-Oppenheimer molecular dynamics). It is
written from scratch in Rust and wgpu, with hand-written physics: no chemistry or MD libraries.

The point of the project is emergence. The only per-element data allowed is the atomic number Z and the
isotope mass. Bond lengths, bond angles, bonds, molecule shapes and the periodic table have to come out of
the equations, not out of a lookup table. That rule (rule 1 in `CLAUDE.md`) shaped most of the hard
problems below.

How it was built:

- **One coordinating Claude session** (the "controller") held the plan, wrote briefs, dispatched agents,
  made rulings and kept a ledger. It did not write most of the solver code itself.
- **Implementer agents** (Fable 5.1 for physics and numerics, Opus 5.5 for GPU, UI and plumbing, and for
  everything once Fable hit its usage limit on Oct 14) wrote the code test-first against CPU f64 references.
- **Reviewer agents** (physics reviewers, read-only) checked derivations, units, signs and tolerances.
  Every task went through a review and, usually, one or more fix rounds before it counted as done. Every
  milestone ended with a whole-branch review and a fix wave.
- **The user** play-tested the app between tasks, filed bug reports from their window logs, asked for
  features, made the big direction calls (for example "no usable frame rates, build a surrogate"), and
  paused things when the machine became unusable.

---

## By the numbers

All measured on the repository, unless marked as an estimate.

| What | Number | Source |
|---|---|---|
| Calendar span | Oct 12 21:24 to Oct 16 23:15 (the latest commit on any branch): 4 days 1 h 51 m | `git log` across branches |
| Calendar days with commits on `main` | 5 (commits per day: 17, 99, 152, 25, 4) | `git log main` |
| Commits on `main` | 297 (278 non-merge) | `git rev-list` |
| Commits on all local branches (non-merge) | 332 | `git rev-list --all --no-merges` |
| Commits ahead of `main` | 25 on `m7-pseudopotentials`, 23 on `activity-tab` | `git log main..<branch>` |
| Rust on `main` | 81,432 lines in 136 files (includes tests, 7,484 lines under `tests/`), plus 2,031 lines of WGSL in 20 shaders | `git ls-files` + `wc -l` |
| Rust on `m7-pseudopotentials` | 91,821 lines | same, in that worktree |
| Crates | 10: `units`, `qm`, `gpu`, `qm-gpu`, `render`, `sim`, `surrogate`, `providers`, `validate`, `app` | `crates/` |
| Largest crates (lines) | `app` 23,460 · `sim` 12,655 · `providers` 12,113 · `qm-gpu` 9,949 · `surrogate` 8,899 | same |
| Markdown | about 15,000 lines (specs, plans, notes, README, `references.md` at 1,864 lines) | `git ls-files '*.md'` |
| `#[test]` functions on `main` | 803 (the oracles are release-only by design) | `git grep` |
| Suite sizes at the orbital-viewer merge | app 210 (2 ignored), providers 106, render 35, sim 102 | orbital ledger |
| Milestones merged to `main` | M1, M2, M3, M4, M5, M5b (viewer extras), UI redesign, M6, orbital viewer, Activity tab, M7, custom window frame, app icon + dist build, File/Save/Profiles windows, Camera view | git |
| In flight, not merged | none (next: M8) | branches |
| Commit trailers on `main` | Opus 5.5: 167 · Fable 5.1: 110 · Sonnet 5.5: 1 | `git log` |

Notes on the table:

- The trailer counts show who the commit message was attributed to, not how many agent runs happened.
  The ledgers say some UI tasks (orbital Tasks 6 and 7, Activity Tasks 6 to 9) were run on Sonnet, but
  their commits carry the Opus trailer. The last Fable trailer on `main` is Oct 14 12:56.
- I did not count agent dispatches, tokens or cost. The ledgers do not record them consistently, so I
  have no honest number to print.
- **Reviewer catch rate.** The ledgers record findings per review in the form "0C/2I/6M" (Critical,
  Important, Minor) and "MUST/SHOULD" for design reviews, but I did not tally a single overall rate, and I
  will not invent one. What they do support:
  - Every final whole-branch review I read found 0 Critical issues (M4, M5, M6, the orbital viewer and the
    Activity tab; the M2 and M3 handoffs say "no Critical/Important"). Important findings where counted: M6
    core half 2, M6 app half 5, orbital viewer 5 (plus 9 Minor), Activity tab 3 (plus 11 Minor).
  - Reviews of written designs, before any code existed, found 6 MUSTs in the pseudopotential spec, 6 in
    the M7 plan, 4 in the orbital-viewer spec, 4 in the precision spec, 6 in the user-data spec and 6 in
    the Activity plan: 32 in total from those six reviews.
  - Several task-level reviews turned an implementer's "done" into "request changes" (examples in the
    failures section), and a few reviews found real bugs that tests had missed (an inverted box rule, a
    vacuous test, a NaN that passed a trust check).

---

## Timeline

Durations are wall-clock spans between commits (design or plan commit to the last commit or the merge).
Several tracks ran in parallel from Oct 14, so the durations overlap and do not add up to the 4-day span.

### Day 0: the design (Oct 12, 21:24)

- **21:24** The first commit adds the design spec, `CLAUDE.md` (the rules) and the agent definitions.
  The rules were set at this point and stayed: only Z and mass per element, hand-written physics, atomic
  units inside, f32 on the GPU and f64 for nuclei, `ForceProvider` as the boundary, validation gates that
  may not be loosened, honesty about approximation error, and "time speed is not timestep".
- **21:50** The M1 plan is committed, with each task tagged Fable (physics) or Opus (engineering).

### M1: CPU reference solver (Oct 12 21:24 to Oct 13 04:26, about 7 h)

The scaffold went in at 22:07 and the last M1 code at 04:01, so the coding took about 6 hours.

- **Delivered:** an f64 spin-polarised LSDA Kohn-Sham solver on a uniform real-space grid: 8th-order
  Laplacian, Jacobi-preconditioned CG Poisson solver with a multipole boundary, PW92 exchange-correlation,
  Gaussian-smeared nuclei, LOBPCG eigensolver, DIIS-mixed SCF and Hellmann-Feynman forces checked against
  finite differences of the energy.
- **Pace:** the core tasks (grid through forces) landed between 22:07 and 00:38, then about 3.5 hours
  (00:38 to 04:01) went on validation: the oracle suite, the He oracle at a finer grid, and gating the
  oracles to release builds.
- **Plan corrections made on the way** (each recorded as a plan commit): the Poisson boundary needed a
  solid-harmonic multipole expansion to l = 6 (monopole plus dipole was not enough); the smearing
  coefficient was 2π, not 2π/3; the preconditioner became multi-sweep damped Jacobi.
- **Results (M1 handoff):**

  | Oracle | Computed | Reference | Status |
  |---|---|---|---|
  | H atom, LSD | -0.477710 Ha | -0.478671 (NIST) | pass, +0.96 mHa |
  | He atom, LSD (h = 0.10) | -2.822561 Ha | -2.834836 (NIST) | pass, +12.3 mHa (bound 15) |
  | Li atom, LSD (h = 0.15) | -7.004270 Ha | -7.335195 | **ignored**, +331 mHa |
  | H2 bond length | 1.4511 Bohr | 1.446 (CCCBDB LSDA) | pass |
  | H2 force at R_e | +0.0010 Ha/Bohr | 0 | pass |

- **What did not work:** Li. The Gaussian smearing error grows like ρ(0)Z⁴σ⁵ and the Li oracle misses by
  331 mHa. (The reference value in that row later turned out to be NIST's LDA figure, not its LSD one; see
  the citation entry below. The conclusion does not change.) It was left as an `#[ignore]`d test with the reason written down, and became one of the
  motivations for pseudopotentials.
- **First pictures** (README's table names this "First pictures"): at 04:01 an H2 density image and an E(R)
  curve. The repository has no separately labelled "spike" between M1 and M2, so this is what I have
  filed under that heading.

### M2: 3D viewer (Oct 13 04:24 to 05:42, about 1 h 16 m)

- **Delivered:** the `gpu`, `render` and `app` crates: winit and egui window, orbit camera, nuclei as
  impostor spheres with true depth, density as a raymarched volume or isosurface, place/select/drag/delete
  tools, presets (H, H2, He, LiH, H2O), scene files, and a worker thread around the CPU solver.
- **Honest speed at this point:** H2 at 49³ took 5 to 7 s per solve on the CPU, and H2O at 57³ about 116 s,
  so the cloud only updated after a drag paused. That was M3's job.

### M3: GPU solver (Oct 13 05:42 to 12:54, about 7 h)

- **Delivered:** `qm-gpu`, a wgpu/WGSL port of the whole solver (padded f32 fields, two-level f32-to-f64
  reductions, analytic erf nuclear potential, Jacobi-CG Poisson with multigrid, GPU LOBPCG, SCF with DIIS,
  forces), plus a GPU backend in the viewer with the density texture handed straight to the renderer.
- **Agreement with the CPU reference at fixed smearing (49³):** |dE| of 1.5e-7 (H), 1.0e-6 (He), 5.9e-7
  (H2), 3.1e-7 (LiH) and 2.35e-7 Ha (H2O); forces within 2e-5 Ha/Bohr.
- **Speed:** H2 49³ went from 5 to 7 s to 206 ms cold (108 ms warm). H2O 57³ went from about 116 s to
  1.8 s. That is roughly 30× for H2 and 64× for H2O (my arithmetic on the handoff figures).
- **Re-aimed mid-milestone:** the Task 8 verdict showed the cost was latency (about 70 dispatches per
  LOBPCG iteration), not arithmetic, so Task 10 became "one readback per iteration, a per-batch uniform
  ring, multigrid, erf to 6e-9".
- **A forced change of method:** the M1 handoff had warned that the sigma-to-zero extrapolated oracles are
  hostile to f32 (intercept weights near 88). So the GPU is validated against the CPU at fixed sigma, and
  the extrapolated oracles stay as the CPU gate.

### M4: dynamics (Oct 13 12:59 to 20:43, about 7 h 43 m)

- **Delivered:** the `sim` crate: `ForceProvider` and its CPU/GPU Kohn-Sham adapters, velocity Verlet
  (reversible to rounding), a Bussi CSVR thermostat, the dynamics driver (play, pause, step, rewind, drag
  springs, pins, one-surface energy bookkeeping), and MD in the viewer.
- **Validation (`--test dynamics`, about 6 min):** H2 vibration within -5.6e-4 relative of its own
  surface's curvature; NVE drift -4.0e-6 Ha/ps over 0.5 ps; GPU rewind to 5.8e-6 Bohr.
- **Numbers that stayed uncomfortable:** 49³ H2 ran at about 3 fs of simulated time per wall second.
  One H2 vibration took about 4 s to watch (M4 handoff), and the M3 handoff had already worked out that
  1 ps of H2 would take about 7 minutes.
- **First 23 hours:** M1 to M4, four milestones, between Oct 12 21:24 and Oct 13 20:43.

### M5: the surrogate ("estimated") force provider (Oct 13 19:39 to Oct 14 08:15, about 12 h 36 m)

- **Trigger:** the user tried M4 and decided there were "no usable frame rates". At 19:39 they chose
  option 1: a learned surrogate that is taught by the project's own solver. The spec followed within two
  minutes, the plan at 19:54, and the first code at 21:01.
- **Delivered:** the `surrogate` crate (Chebyshev radial basis, real solid harmonics, body-2 and body-3
  descriptors, ridge fit with a Bayesian posterior, a bit-exact `.model` sidecar) and `sim::SurrogateKs`:
  the model answers when it trusts itself, falls back to DFT when it does not, learns from the fallbacks,
  and is audited in the background. The viewer got an "estimated" solver, a sampling pass, uncertainty
  readouts and autosave.
- **Validation (`--test surrogate`, about 4 min):** H2 surface reproduced to 2.5e-6 Ha rms; surrogate MD
  frequency within 7e-6 relative of the DFT-MD pin; 10 ps NVE drift of -2.6e-11 Ha/ps; H3+ at 49³ within
  3.6e-6 Ha; out-of-distribution geometries trigger fallback; active learning from nothing needed 38 DFT
  calls in 300 steps.
- **Speed:** about 1.6e5 steps/s (about 40,000 fs/s) in the worker test, against DFT's roughly 3 fs/s.
  That is about 10⁴× (my arithmetic).
- **User-driven fix rounds:** the M5 handoff says eight fix rounds on Task 8 were driven by the user's
  play sessions, with their window logs as the bug reports. (The labelled commits in git go up to "round 4"
  for Task 8, plus further rounds on Task 9; I report the handoff's count as written.)
- **Merged at 08:15** on Oct 14.

### M5b, the UI redesign and the small-requests wave (Oct 14, morning to early afternoon)

- **M5b viewer extras** (07:45 to 10:32, about 2 h 47 m, all user requests): container walls with an
  adjustable box (a "piston"), density crossfade, a live cloud in estimated mode (the user had reported
  that "cloud and isosurface don't work in estimated mode"), and draft mode (the model answers every step
  with a visible honesty banner).
- **UI redesign** (08:27 to 11:02, about 2 h 35 m): the user asked for a mockup first, then approved it as
  the spec. The result: top menus, bottom toolbar, right dock, overlays, Ctrl-K search, and every control
  routed through one `Action` layer. The control-mapping checklist reached 148 of 148.
### M6: tools (Oct 14 08:57 to Oct 15 07:14, about 22 h 17 m)

The wall-clock figure includes the overnight Task 11 saga below and several side branches.

- **Delivered:**
  - A `providers` crate behind a `Trainable` trait (Task 1, Oct 14 11:24 to 11:31), splitting the two
    biggest files that the M5 final review had flagged (about 3.7k and 2.8k lines). The surrogate's
    frequency check came out identical before and after, which is the evidence that nothing moved.
  - Bonds read off the solved density, so no distance tables: a saddle or deformation-maximum test against
    a promolecule reference, a Hirshfeld-share attribution rule and a multi-centre march for ring bonds.
  - Molecule formulas (Hill, IUPAC Red Book) and names, including a user-requested name style switch
    (common, IR-5 stoichiometric "dihydrogen monoxide", IR-6 "oxidane", formula only).
  - A composition badge, per-nucleus isotopes (AME2020 masses), a FIRE Relax tool, and a 118-element
    periodic-table picker with the heavier elements greyed out until pseudopotentials exist.
  - An input-replay harness (`--record`, `--replay`, ten checklist scripts).
  - Two user-requested additions folded in at the end: sampling-plan presets, and a bond-strength display
    (line width and style from the density and Laplacian at the bond point).
- **Validation (tools suite, 9/9 at the first run):** H2 at R_e bonded (q 1.703); H2 at 6 Bohr not bonded;
  He2 at 3.0 Bohr not bonded (q 0.9705); H3+ three multi-centre bonds; water exactly two O-H bonds at three
  orientations, with no H-H bond; LiH bonded with ρ_b 0.0423 (literature about 0.04); D2/H2 frequency
  ratio within -4.6e-5 of the mass-ratio prediction.
- **Merged** on Oct 15 at about 07:14 (fast-forward).

#### Task 11: the surrogate safety fixes (Oct 14 18:18 to Oct 15 02:34, about 8 h 16 m, 7 rounds)

This deserves its own entry. See [the stuck-then-escape story](#the-stuck-then-escape-bug-and-the-safety-rule-that-took-seven-rounds)
under failures.

### Small user-requested items on `main` (Oct 14, afternoon to evening)

These landed outside the milestone plans: the scroll wheel setting the drag spring's force cap (merged
16:55); a fixed-width bottom toolbar, because live numbers made it jitter (18:33); an **empty** preset,
**container walls on by default**, **Quality presets** with Accurate (h = σ = 0.15 Bohr) as the new default
grid (see the smeared-nucleus story below), temperatures in K, °C or °F, and a GPU memory guard (all merged
together at 19:46); and `start.bat` / `build.bat` (19:48).

Two further designs were written the same afternoon and not built: **user data** (a model library and
settings profiles, spec and plan at 17:32) and the **precision milestone** (precision renders, exact and
DMC teachers; spec at 19:37).

### Orbital viewer (spec Oct 14 17:59, merged Oct 16 12:23, about 42 h 24 m wall clock)

- **Delivered:** Kohn-Sham orbitals on screen: one orbital as a signed (+/-) surface or volume, or its
  square; all orbitals coloured; probability dots; a spin-density view; unoccupied orbitals from a separate
  display eigensolve; GPU display products with one scalar readback per update (about 1 to 2 ms at 49³ to
  97³); a tab with honest labels.
- **Pace:** a spike (Task 0) and two tasks on Oct 14 evening, then the work waited for M6 to merge and for
  the shared heavy-job slot (see the process section). Tasks 3 to 5 ran on Oct 15, Tasks 6 and 7 on the
  morning of Oct 16.
- **Preview first:** after the user asked about the orbital views, a throwaway `orbital-preview` branch
  was built so they could try dots and the volume knee. Their feedback (dots coloured by the field, and
  dots sampled exactly from the trilinear field, which removed a cube-shaped grid artefact) was ported to
  `main` as a separate dots view before the milestone merged.
- **Final review:** ready after fixes, 0 Critical, 5 Important (for example: paused estimated orbitals
  never arrived at an idle worker; stale unoccupied rows after play then pause; a silent orbital switch
  on solver change). Fixed and merged at 12:23 on Oct 16.
- **Not done:** Task 8 (reconstructing core shells for display) waits on pseudopotentials.
- **After the merge:** the user reported a spin-density bug (see failures).

### Activity tab (Oct 15 07:42 to Oct 16 23:15, about 39 h 33 m, not merged)

- **Origin:** a user request on Oct 15 for obvious processing feedback with something to watch, followed by
  a request from the Model tab for a live log of new solves added to the model, with the total solve count
  updating on its own.
- **Delivered on the branch:** activity records and a live probe in the solver and surrogate; a trust
  checklist taken from the policy's own predicate; a read-only `would_answer`; training views (parity and
  learning curves, a timeline); an Activity dock tab with jobs, live counters and the solve log; a status
  headline and toolbar progress bar; plain-language refusals.
- **Cost of the instrumentation, measured:** 3.7 ns per SCF step for the probe; 0.0084 % of teacher time
  and 0.78 ms per frame for the whole feature.
- **State:** the final fix wave committed at 23:15 on Oct 16. Before it merges, the surrogate, dynamics and
  gpu_vs_cpu suites still have to run on the merged tree.

### M7: self-generated pseudopotentials (spec Oct 14 08:49, plan Oct 15 04:23, merged Oct 19 ~10:20 local)

**Merged 2026-10-19.** Tasks 1–13 plus the preview branch's work landed together: Task 10's
oracles (H₂O under *all species* 104.97° against SVWN 104.99°, O–H +0.0097 Å; NH₃ 107.6°, CH₄
tetrahedral), Task 11's hydrogen dimer gate and the new default *accepted species* (H₂ Richardson
ω 4024 cm⁻¹, −3.7 % against the all-electron curve, bound 5 %, noise ≈ 0.55 %), a checksummed
disk cache of gate verdicts (first launch ≈ 80 s, then ≈ 25 s), the level-gated SCF stall stop,
the estimated mode's trust fixes (teach-around passes, shell checks and structural shell fences,
the "stuck on DFT" prompt), and the user's preview requests (left dock, model file dialogs,
load-settings card, bond lengths, steady labels, follow, audit switch + auto cadence, the
training ghost with its last three samples). The final scoped physics review found one critical
(a shell fence lifted by a refit) and three important issues, all fixed with pinning tests
before the single final run: units+qm 212, qm-gpu+render 162, sim+surrogate 192, providers
239, validate all binaries green (oracles 1414 s, pseudo 3016 s, pseudo_oracles 5294 s,
surrogate 620 s), app 345 — 3 h 43 m serial. Testing split agreed with the user: Claude runs
the physics suites once; the user hand-tests UI in previews (they found most of the workflow
bugs this way). Failures worth remembering: a helper killed a test process it had not verified
as its own; an interim report was misread as "your model is trusted"; the user's playing
window was closed once by a relaunch.

The history below is the in-progress record as it stood before the merge.

About 42 h 39 m from the plan commit so far, with 25 commits on the branch.

- **Why it exists:** at the default grid the smeared all-electron oxygen is not oxygen (see below), so
  anything with O, N or C is dishonest until the grid can resolve less. The route is pseudopotentials
  generated from the project's own all-electron atom, with one element-independent recipe (rule 1).
- **Done so far (the branch ledger's tasks):**
  - Task 1: a radial all-electron LSDA atom that matches NIST's LSD tables for H to Ne.
  - Task 2: the core/valence partition and Troullier-Martins channels.
  - Task 3: the Kleinman-Bylander form and ghost-state gates.
  - Task 4: non-linear core correction (NLCC), transferability gates T1 to T7, a generation record.
  - Task 5: the spherical-Bessel filter and the short-range table.
  - Task 6: `qm::ionic` and SCF integration.
  - Task 7c: the partial-core alias fix.
  - Task 8a: GPU energy kernels.
- **Still to do:** Tasks 7 and 8b to 13 (see What's next).


### 2D molecule builder (spec Oct 20 07:43, last commit 08:46, about 1 h 3 m)

File ▸ **Builder…**: a window where a molecule is sketched as a 2D structural formula and handed to the
scene as a 3D starting geometry, which the solver then relaxes. One Opus session, no subagents, four parts
with the user hand-testing each one in the app before the next (the testing split: Claude ran only the
builder's CPU unit tests, seconds; `cargo test -p app --release --bin app` once on the merged tree).

- **Rule 1, signed off.** The builder needs a valence table and covalent radii, which is per-element data
  beyond Z and mass. The user signed off on them in the spec (§3) under conditions: they live only in
  `crates/app/src/builder/chem.rs`, they only decide where atoms are first put and which atoms the window
  marks, and nothing from them reaches `sim`, `qm`, `providers`, the forces, the bond detector or the
  names. The Relax after the hand-off (on by default) moves the guess to the solver's own geometry.
- **Parts:** the canvas (atoms, bond orders, charges, undo); the assist (Fill hydrogens, valence warnings,
  group stamps; a charged atom's valence is the octet count of its v − q electrons, which gives NH₄⁺,
  H₃O⁺, OH⁻ and BH₄⁻ without a second table); the 2D → 3D embedding (bond and VSEPR 1–3 springs, a soft
  repulsion, a triple-product planarity term, gradient descent); the hand-off (Replace / Insert, a fit
  check with an "enlarge the box" offer that shows the GPU memory need first, Relax after).
- **The user's hand-tests changed the design three times, all within the hour:** the Atom tool became
  select-then-click (drag moves an atom; a click near the selected atom places a bonded one, with a ghost
  preview; right-click deselects); bonded atoms then snapped to a fixed length and 30° steps; and then,
  on the user's next two messages, to rough real geometry instead — lengths from the radii, VSEPR angles
  imposed only once there is a bond to measure from. A faded controls cheat sheet went into the canvas
  corner on request.
- **`names` gained display-only entries** for ethanol, acetic acid and benzene, with C–C signatures so
  dimethyl ether and methyl formate are told apart: the spec's success test is that a drawn ethanol is
  named ethanol by the bond detector after a Relax.
- **Failures worth remembering:** a PowerShell `Get-Content | Set-Content` edit read a UTF-8 file as ANSI
  and wrote it back garbled (`·` became `Â·`) plus a BOM, and the garbled file was committed before the
  next edit noticed it; the repair reversed the mis-decoding run by run so later, correct text was not
  touched. Edits since go through Python or the editor tool, never a PowerShell round trip. The first
  benzene embedding failed its equal-C–C test honestly: a Kekulé ring has alternating single and double
  bond lengths, so a six-ring drawn alternating is now treated as aromatic (the mean of the two factors).
  The first build waited behind other sessions' suites; the user chose a low-priority `-j 4` build
  instead, which took 30 s.

---

## Failures and dead ends

These are the places where the first idea was wrong, or an approach was tried and abandoned. They are
roughly in the order they happened.

### The smeared nucleus gives wrong bonds (the oldest problem, still being worked on)

A point nucleus has a cusp the grid cannot represent, so each nucleus is smeared into a Gaussian of width
σ, and σ must be at least the grid spacing h. That smearing adds a repulsive term and moves everything.

- **Measured by the user:** at the original default (49³, h = σ = 0.25), H2 came out at 1.78 Bohr, which is
  0.944 Å. Experiment is 0.7414 Å. The user filed this as a request to make the more accurate settings
  the default.
- **What the solver itself says (README, from `references.md`):** H2 R_e at Fast 1.002 Å, Balanced 0.947,
  Accurate 0.847, Precise 0.807, against the sigma-to-zero extrapolation at 0.768 Å, whose gap to experiment
  is the LSDA functional's own error. Accurate became the default grid on Oct 14. Even there the bond
  is 0.106 Å too long.
- **Water goes straight.** At h = 0.25 the relaxed water on this surface is near-linear (O-H 2.34 Bohr,
  177°), 0.160 Ha (4.4 eV) below the physical geometry that the `h2o` preset loads. Pressing Play heats it
  past 10,000 K within about 2.5 fs as it opens. The M4 dynamics test for water was left `#[ignore]`d as a
  documented resolution limit.
- **Oxygen is worse.** The M7 spec computes that at σ = 0.25 the oxygen atom comes out near -36.6 Ha
  against about -74.5 Ha resolved, and (at 33³) the singlet lies 0.227 Ha *below* the triplet, so Hund's rule does
  not emerge. The orbital-viewer spike found that Ne's 2s orbital sits above its 2p (2s at -0.298 Ha, 2p at
  -0.845 Ha), which is backwards. The UI now flags this as "shell order inverted by the smeared nucleus".
- **A first explanation was wrong.** The implementer of orbital Task 1 explained the inversion as the
  smeared nucleus acting like a harmonic well past a threshold. The physics review corrected the mechanism to
  cusp removal.
- The README, the UI and `references.md` all say these things out loud (rule 7).

### The Li atom and the extrapolation oracles

Li misses its oracle by 331 mHa and was left ignored (above). The sigma-to-zero oracles also turned out to
be hostile to f32, which forced the "compare GPU to CPU at fixed sigma" design in M3.

### Dynamics too slow to enjoy (Oct 13)

M4 worked and was correct, and the user judged it unusable as a toy: about 3 fs per wall second. Instead
of tuning, they chose a different architecture (the surrogate). It bought roughly a 10⁴× speed-up. A large
part of the cost shows up below, in the safety problems of Task 11.

### Smaller M3 and M4 stumbles

- **A crash at exit.** After GPU tests the test process exited with 0xc0000409. The cause, found from a crash
  dump (commit 469682c): wgpu's default instance also initialised the GL backend, which loaded AMD's
  OpenGL driver, which then fast-failed during process detach. The fix was to ask for the primary backends
  only.
- **A panic on window resize** when the worker thread was submitting to the shared GPU device, because
  wgpu 30's surface reconfigure raised a validation error that the default handler turned into a panic
  (d40dbb9). It now returns an error and the viewer retries.
- **A drag that injected 1.1 Ha** and flew an H2 out of the box, in the M4 handoff. The cure is a
  critically damped spring with a force cap of 0.1 Ha/Bohr and a cell-margin guard.
- **A warm-start stall on water.** The 57³ water MD stalls on step 2 (warm SCF does not converge while cold
  solves do). The trigger was never found; the "fix" was to retry cold once, which the handoff calls
  unresolved.
- **The first wall potential was wrong.** A quadratic wall made the kinetic energy random-walk by about 1 %
  per bounce (20 % after 300 bounces) because of a curvature jump a Verlet integrator does not conserve
  across. The replacement is a C² wall, measured at max |K/E - 1| of 1.05e-3 over 306 bounces.
- **Plan numbers that were wrong:** the M4 plan's eigen-tolerance factor F of about 0.1 left out a factor of
  about 10 (the measured value is 0.0015); and the plan's NVE swing bound of 2e-4 was unattainable, so
  ruling R6 replaced it with a derived h²/4 argument.

### Wrong-looking-right numbers in M5 and M6

- **The box rule was inverted.** In M5 the controller's own first expression for the h-scaled box rule had
  the scaling backwards. A re-review caught it (ruling S17), and the correct form was 2e-5 times (h/0.25)².
- **Guessed bounds, replaced by derived or measured ones.** In M6 Task 6 the brief's Morse tolerance lay
  inside the stop criterion's slack, so it was re-pinned to the derived |F_final|/k; the brief's 2× pin for
  the FIRE-versus-quench ratio was a guess (measured 1.15 and 1.63). In Task 7 the spec's claim that FIRE
  needs half the evaluations of a quench turned out false on small molecules (25 versus 16 on H3+), and
  the spec was to be corrected. In Task 3 a ceiling had been asserted against the wrong statistic (review
  finding I1).
- **A real bug found during a fix round:** Task 3's detector asked the surrogate for its teacher on every
  run, which switched off the surrogate's density sampling after each fallback, so no bond graph was ever
  published under the surrogate.
- **A false "bond".** The Task 7 review found that "H2 stays bonded at 5 Bohr" was not physics but the
  spin-symmetric SCF start (n↑ = n↓). The wording and README were changed, and a spin-asymmetric start was
  queued as a solver follow-up.

### The stuck-then-escape bug, and the safety rule that took seven rounds

This is the biggest dead end in the log.

**The report (Oct 14).** The user was in estimated mode on a trained H2 and, after an atom edit, saw the
estimated solver get stuck in a storm of fallbacks, then atoms escape, with the status still saying
"Surrogate, trusted". The diagnosis reproduced it by heating a trained H2 and adding one hydrogen while
playing. About 140 fallbacks to DFT happened and none were learned from, and E_total read +578 Ha on
fallback steps.

**The diagnosis (git: `a031a59`, 18:18).** The cause was several things together:

1. The surrogate's "envelope" check (is this geometry like the training data?) relied on a vanishing feature
   firing for an isolated atom. One kept training sample of a stretched pair at r = 5.125 Bohr was enough to
   break that, so an isolated atom looked in-range and was answered with zero force. A hot H2 could then
   dissociate and fly off at 287 Bohr, with no edit needed.
2. The background audits that would catch it were blind: a dissociated pair has its density on the cell
   faces (so the audit is marked box-affected and not judged), and once an atom is outside the cell the
   teacher refuses the geometry. Measured: over 1.5 million answered steps the last audit stayed empty.
3. The fallbacks were never learned from because the box rule (correctly) rejected every one of them: in a
   12 Bohr cell nearly any H3 is box-affected.
4. Some of the strange energies were bookkeeping, but not only: the energy accounting restarted from the
   model's extrapolated number and carried it forever.

**The cure was planned as Task 11 and took six committed rounds, plus two parked ones:**

| Round | Commit time | What it did |
|---|---|---|
| 1 | Oct 14 20:27 | the six requirements, including a cell check and a lone-atom rule. Its own finding: an H2-only model still answered H2 plus one H inside its envelope with forces 4 to 22 times off |
| 2 | 21:11 | the fix for that: a novelty rule (rule 5d: a direction the training data never spanned), no verdict reuse across a coordination change, and audits that judge forces |
| 3 | 22:06 | a compensated Gram matrix (fixed a pre-existing singular fit at 30,000 dimers). The implementer `#[ignore]`d the hot repro test; **the controller rejected that** as a rule-6 violation |
| 4 | 22:48 | an extrapolation grade by D-optimality (MaxVol), gamma_max = 2, with the hot repro un-ignored and green |
| 5 | Oct 15 00:07 | citations and the convergence flag, but the strict hot run (every answer judged) **failed**: 12 answers at 3.1 to 6.2 times the force bound |
| 6 | 01:05 | class-wise exact leave-one-out trust (isolated, pair, many-body); the strict hot run passes (58 judged, worst 2.69× the bound, then refused by the class rule) |
| 7 | parked | per-class conformal calibration (the physics review's demand): fixed the over-cautious fallback loops, but **six answers at up to 8.09× the force bound** in an unprobed three-body region |
| 7b | parked | the same with gamma_max = 1: no wrong answers (worst 0.63×), but **too cautious**: the symmetric H3+ stretch fell back, active learning needed 10 % fallbacks versus a 5 % bound |

The decision, recorded in the ledger: ship round 6 (merged at 02:34), park rounds 7 and 7b as patches, and
stop research on this ("a research cap on Task 11, told to the user"). The cost is documented: one badly
learned sample can leave a whole coordination class untrusted, so estimated mode falls back more than it
needs to. The calibrated replacement is still parked with its measurements in `.superpowers/parked/`.

The rule that held throughout: when a safety test failed, nobody loosened the bound. Round 3's `#[ignore]`
was refused, round 7b's cautious result was weighed against "no wrong answers", and the choice was
between two honest costs.

### Three hydrogens will not reach trust (Oct 15)

Another user report: the surrogate could not fit its own training data for 3 H. The diagnosis
(`diag-3h-report.md`) found two independent causes. First, box confinement (an H atom 2 Bohr off centre
already gains 1.6 mHa; 130 of 145 samples were box-affected; in an 18 Bohr cell the same model was
trusted). Second, open-shell LSDA has several self-consistent broken-symmetry solutions, up to 27 mHa
apart, so the **teacher's labels are multi-valued** and the model correctly refuses. The first was fixed in
M6 (a cell-aware refusal message); the second is queued for "lowest of several spin starts".

### Process failures

- **A merge piped through `tail` hid a failure.** Writing `git merge ... | tail` inside an `&&` chain makes
  the chain's exit status come from `tail`, so a failed merge did not stop later steps. It happened twice:
  on Oct 14 a failed fast-forward merge let the chain go on to delete the remote branch and the build
  workspace; on Oct 15 a conflicted merge let the chain remove a branch's worktree. The commits survived
  both times. The rule now (in the assistant's memory notes): run gating git steps alone, never piped.
- **A commit swept up someone else's files.** On Oct 13 (`657eeea`, M5 Task 4) a controller commit of a
  docs note also committed ten staged files that an implementer was working on, mislabelling the history.
  It was not rewritten; a marker was added and the rule became `git commit -o -- <paths>`.
- **Three parallel test tracks overloaded the user's PC (Oct 15, about 14:30).** M7, the orbital viewer
  and the Activity tab each ran heavy tests at once: a 4 GB test binary on most cores, the 31-minute
  oracles and the app suite. Free RAM fell to about 2 GB of 31 and the user said they literally could not
  use their computer. All agents were paused, then resumed under a new rule: any number of agents may edit,
  but only one may run a heavy build or suite at a time, and the controller hands out that slot. A later
  note raised the free-RAM threshold to 4 GB after a 6 GB margin stalled a study for an hour.
- **A six-hour CPU test run.** The M7 Task 6 round 1 pseudo suite took 22,041 s (6.1 hours) on the CPU and
  ended with 7 passes and 1 failure. The user's question afterwards produced the rule that stuck: the CPU is
  only the f64 answer key, so CPU tests stay small (minutes) and heavy 3D studies run on the GPU. On the
  same oxygen case the GPU took 3.1 s against 235 s on the CPU (see the wins).
- **Fable hit its usage limit mid-task (Oct 14).** In M6 Task 3's fix round the Fable implementer got an
  HTTP 429 with about 440 uncommitted lines in 13 files. A fresh Opus agent continued from the partial
  patch, and at the user's word ("continue with opus") the rest of M6, M7, the orbital viewer and the
  Activity tab went to Opus, with a mandatory physics review on physics tasks.
- **Weekly usage.** The user asked on Oct 15 for a stop at 60 % of the weekly plan and a cost cut: cheaper
  models for engineering-only tasks, Opus or Fable only for physics. The ledgers record readings from 35 %
  on Oct 15 up to 53 %; on Oct 16 at 49 % the user raised the cap to 75 %.

### M7: four things that were tried and turned out wrong

These are the best illustration of "plausible but wrong" in the project, so they get detail.

**1. The T5 window had to be revised.** Task 4 added a transferability gate (T5) that checks, over an
energy window, that the pseudo-atom's eigenvalue errors stay small. The first window ran up to the top of
the valence band plus 1 Ha, which includes the positive continuum, and it refused Be, B and Al to Cl at
every setting. The reviewer derived a per-channel rule (each angular channel's eigenvalue ±0.5 Ha, tested at
5 mHa) by a confined-state check that matched to 4 digits. Result: Be and Al refused, B, Si, P, S and Cl
accepted. This is recorded as a derived revision, not a tuned one.

**2. The 1e-8 lattice-sum judge was never derived.** Task 5 stopped, blocked: the filtered tables missed
a 1e-8 bound that the spec had stated. The ruling says plainly the bound was never derived. It was
withdrawn and replaced with a gate derived from what the tables are for (their egg-box force on an atom,
under 2e-6 Ha/Bohr per atom for the tables against a 3e-5 total gate). Numbers followed: the worst
table force came out at 1.72e-6 (nitrogen).

**3. The O₂ egg-box got *worse* on finer grids.** The "egg-box" effect is the grid-position dependence of
the force. The M7 gate is a sum of forces below 3e-5 Ha/Bohr. For O₂ the measured miss was
**6.07e-5 at h = 0.20, 7.51e-5 at 0.175 and 9.01e-5 at 0.15**: finer made it worse, the opposite of
what a discretisation error should do.
  - Several remedies were measured and did not solve it: doubling the grid; a softer partial-core filter
    (it brought 0.15 down to 2.3e-6 but did nothing at 0.20, and the ruling rejected it as not
    principled); a cubic prolongation of XC (3.5e-4 at 0.20, worse). A first suspect, the 12 Bohr cell
    wall, was ruled out by an 18 Bohr run with the same result.
  - GPU f32 energies failed to qualify as the measuring instrument (their noise was 5.5e-5, the same size
    as the effect), so the study re-evaluated the GPU's converged states in f64, which agreed with the
    CPU to 1.15e-7.
  - The mechanism, found by the ruling: the filtered partial core's q⁻⁴ tail aliases inside the XC
    functional, and the earlier egg-box test filter cut off exactly where the effect lives, so it was blind.
  - **Smoother partial cores were refuted.** The tempting fix was a smoother core form. The reviewer's
    argument (a Bernstein-inequality bound) showed that matching more derivatives of an exponential tail
    requires either content at the aliasing frequency or a larger central amplitude, so smoothness cannot
    help, and a numpy model confirmed that at equal cost a plain linear core at a smaller crossover beats
    every smoother form. Option A was rejected on paper before more GPU time was spent.
  - The option that worked is in the wins section.

**4. The "H floor" was the box.** Task 6 reported a hydrogen egg-box floor of 2.90e-5 Ha/Bohr, next to the
3e-5 gate. The ruling found that the 12 Bohr Dirichlet wall alone supplied 2.86e-5; the true egg-box was
about 4e-7. The gate was not changed. Every later probe subtracts the wall.

Two smaller M7 reversals: the open-shell oxygen SCF would not converge (singlet: the hole orientation
flipped; triplet: orientation nearly free and stalled at 7e-5) until a shared-degenerate-shell occupation
rule was added, and the first version of that rule depended on the SCF path and was replaced by a
state-defined two-phase rule; and Task 2's V_ion tail test missed its bound (4.46e-8 against 1e-8), which
turned out to be the core's own Hartree tail rather than a bug; the physics ruling re-posed the test to the
region beyond the compression radius, where the identity holds, and asserted it over 52 element and f
cases.

### A citation error found by reading the source

When the M7 radial atom was compared with NIST, the project's own `references.md` and its ignored Li oracle
cited -7.335195 Ha as lithium's LSD energy. On NIST's page that figure is the LDA column. The LSD value is
**-7.343957 Ha**, 8.8 mHa lower. It was inside the oracle's bound either way, but it was recorded and
corrected.

### The spin-view noise bug the user just found

After the orbital viewer merged (b40c12d, Oct 16 12:23, "ready after fixes" and reviewed), the user
reported around 23:00 that H2's spin-density view shows a full-scale flickering amber and cyan dot cloud.
H2 is a closed-shell singlet, so its net spin density m = n↑ - n↓ is zero, and the view scales to the
maximum |m|, which is f32 round-off, so it magnifies noise to full brightness. A second report: LiH
(also closed-shell) shows large smooth blobs that flip sign between steps, which is the SCF's
spin-channel residual at its tolerance. They also noticed that the noise appears less at finer quality
presets. The fix is queued, with the requirement that a noise floor must cover both the SCF tolerance
residual and the f32 floor, scale with the grid, and still draw real spin (Li, an O triplet, a stretched
H2). It has not landed yet. The ledger does not say why the orbital review and tests did not catch it.

---

## Successes and big wins

### The radial atom matches NIST to the printed digit

M7 Task 1 wrote a radial all-electron LSDA atom (an independent solver for the spherically symmetric
problem, used as the generator of pseudopotentials and as a check on the 3D code). Compared with NIST
Standard Reference Database 141 for H through Ne, every total energy, every energy component and every
occupied eigenvalue agrees with NIST "to its printed rounding": the worst absolute difference is 4.9e-7 Ha.
(The test-only comparison uses VWN, which is what NIST tabulates; the 3D code uses PW92, and the measured
gap between the two functionals is small and recorded: H -0.04, He +0.38, Li +0.67 mHa.) Other checks
in the same task: the scaling identity E(Z, σ) = Z²E(1, Zσ) holds to 6.4e-11, and the error ratio under
grid refinement is 16.05 where theory says 16.

### The GPU solver (M3)

A full Kohn-Sham solver in f32 WGSL on an AMD card (no CUDA), agreeing with the f64 CPU reference to
between 1.5e-7 and 1.0e-6 Ha across H, He, H2, LiH and H2O at fixed smearing, bit-deterministic, and
about 30× to 64× faster. The default
81³ H2 solve now takes 0.53 s cold and 0.28 s warm; H2O at 97³ about 3.6 s cold. A frame time of about
6 ms (p95) is kept during a 97³ solve, because the render thread never waits for the solver.

### The surrogate reproduces the DFT it was taught, and says when it does not (M5)

Surface error of 2.5e-6 Ha rms on fresh points, an MD frequency within 7e-6 of the DFT run, and 10 ps NVE
with a drift of -2.6e-11 Ha/ps, at roughly 10⁴× the DFT's simulated time per second. Uncertainty is shown,
fallbacks are visible, and out-of-distribution geometries refuse. It is not a general force field and the
UI and README say it is valid only where taught.

### Physics reviews caught maths that looked right

A representative list from the ledgers (none of these would have failed a casual test):
- the h-scaled box rule written upside down (M5);
- a Morse tolerance that lay inside its own stop criterion's slack (M6);
- the "H2 still bonded at 5 Bohr" artefact of a spin-symmetric start (M6);
- the 1e-8 judge that was never derived (M7 Task 5);
- the "H floor" that was really the cell wall (M7 Task 6);
- an LDA number labelled LSD in the references (M7 Task 1);
- the wrong mechanism for the shell inversion (orbital Task 1);
- a trust check that a NaN would pass (Activity Task 3b: "NaN must refuse");
- the T5 window that refused elements it should have accepted (M7 Task 4);
- a mid-task spec correction on the radial atom's hydrogenic error bound: the error is relative and scales
  like δ⁴, so the absolute bound scales like Z² (sent to the implementer while it worked).

### Bonds read from the density (M6)

No distance table and no per-element radius: the viewer solves the density, compares it with the sum of
free atoms, and decides bond or no bond from where it piles up. It gets H2 and H3+ (a ring bond) right,
leaves He2 unbonded, finds exactly two O-H bonds in water in three orientations, and measures LiH's bond
critical density at 0.0423 against a literature value of about 0.04. Isotopes (D2 versus H2) come out
within -4.6e-5 of the mass-ratio prediction. It also exposed honest failures that are written down: HeH+
comes out bonded, a stretched H2 stays bonded until 5 Bohr, and water's O-H bonds flicker between solid
and dotted at the noise band.

### The O₂ egg-box is fixed at the default grid (M7 Task 7c)

After the Bernstein argument ruled out smoother cores, the ruling chose a simpler fix: a global
partial-core crossover constant c = 1/4, chosen by a minimax rule over a ladder {1, 1/2, 1/4, 1/8}, with a
bounded 3D confirmation and an automatic fallback if it failed. Measured result (recipe version
`m7-tm-kb-nlcc-4`, 83 of 83 rows):

| h (Bohr) | O₂ force sum before | after | gate 3e-5 |
|---|---|---|---|
| 0.20 | 6.07e-5 | 4.26e-5 | still misses, recorded as a ratchet (8× better than all-electron) |
| 0.175 | 7.51e-5 | 9.2e-6 | pass |
| 0.15 (the default grid) | 9.01e-5 | 6.75e-6 | pass |

The inversion (worse on finer grids) is gone. The lone-oxygen frozen force at 0.15 fell from 7.07e-5 to
4.2e-6. H2O with the core only (frozen core shells) passes at 0.15 (4.14e-6) and misses at 0.20 by 1 %. The
fix passes the project's own rules: one constant, element-independent, chosen by a declared rule and
published with the cost (it makes some transferability gates harder to pass; nitrogen's recipe changed to
f = 1.2 with a p-local channel).

### GPU pseudopotentials are about 63× to 76× faster than the CPU

From the Task 8a report: pseudo-O (61³, h = 0.20) takes 3.1 s on the GPU versus 235 s on the CPU (76×);
CH₄ 1.7 s versus 108 s (64×); H₂O 2.3 s versus 145 s (63×). The same GPU-versus-CPU agreement holds: energy
within 6.1e-7 Ha. Because of this, the O₂ egg-box gate that took about 3 hours on the CPU took 143 seconds on
the GPU, which is what made the egg-box study practical at all (with the f32 instrument caveat above).

### The orbital viewer

Orbitals as signed surfaces and volumes, probability dots drawn from the exact trilinear field, and a
tab that labels what is and is not physical. It costs about 1 to 2.3 ms per update at 97³ with a single
scalar readback, plays at the same trajectory and rate with the view on (2.2 to 2.6 fs/s either way, E
conserved spread 3.10e-5 Ha), and its unoccupied-orbital solve has a latency cap (0.25 s) so it cannot
stall play. The display eigensolve agrees with the SCF's occupied eigenvalues to 1.9e-7 on the CPU.

### Process wins

- An input-replay harness (M6 Task 8) that found a real app bug on its first scripts: a running relax
  stopped after one FIRE step in the window, because the worker did not count it as work.
- Zero Critical findings in the final whole-branch review of M4, M5, M6, the orbital viewer or the
  Activity tab, and none in M2 or M3 either.
- Rule 6 held: no tolerance was loosened to make something pass. The ledgers record each time a pin
  changed and show a derivation or a measurement beside it.

---

## How the process works

**The rules** (from `CLAUDE.md`, unchanged since Oct 12):

1. Per-element data is Z and isotope mass only. A fitted per-element or per-pair parameter is a design
   violation. Display colours are not physics.
2. Hand-written physics. Crates are for plumbing (wgpu, winit, egui, math types).
3. Atomic units inside; conversion only at the display boundary.
4. f32 for grid quantities on the GPU; f64 for nuclei, integrators and small dense algebra; compensated
   summation for energies.
5. `ForceProvider` is the boundary: `sim` does not know what computes forces; `qm` does not know about
   rendering.
6. Validation gates. Solver changes pass the unit and oracle suites; dynamics changes pass the dynamics
   suite. Never loosen a tolerance to make a test pass; investigate.
7. Physics honesty: if the approximation disagrees with exact or experimental values, document the gap.
8. UI speed changes steps per frame, never the timestep.

**The loop for every milestone:** brainstorm, then a spec, then a physics review of the spec (the six design
reviews above found 32 MUSTs before any code), then a plan with each task tagged Fable or Opus, then each
task is implemented test-first, reviewed, fixed, and re-reviewed on only the changed parts; then a final
whole-branch review (often split into a physics half and an app half), a fix wave, the suites run on the
merge tree, a merge, a handoff note and a README update. The ledger for each milestone records each
decision as a "ruling" with its reason and "cost if wrong".

**Models.** Fable 5.1 was assigned solver maths, integrators, force derivations, oracles and physics
reviews (any task where a wrong answer still looks plausible). Opus 5.5 took GPU plumbing, UI, rendering,
tests with pinned values and refactors. When Fable hit its limit on Oct 14, Opus took everything, and every
physics-tagged task got a mandatory physics review. Later the user cut cost further: Sonnet for batched
UI work.

**Parallelism, and its limit.** From Oct 14 several milestones ran at once, each in its own git worktree
(`Physics-m7`, `Physics-orbv`, `Physics-activity` and others), merging `main` into their branch as needed.
The rule after the Oct 15 overload: many agents may edit and read at once, but at most one runs a heavy
build or suite, and the controller assigns that slot. Waiting for the slot is a real cost; the orbital
viewer's Task 5 and M7's Task 6 round 2 are examples of work that queued behind each other.

**The user's role.** They did not write code. They asked for features during play-testing (many appear as
dated entries in `docs/superpowers/notes/2026-10-13-m5-user-requests.md`), reported bugs from their window
logs, rejected one milestone shape ("no usable frame rates"), asked for a mockup before the UI redesign,
set the standing order "merge and keep going" (hands-off execution, Oct 13), and paused the project once
when their computer was unusable. They also reordered work: after asking about orbital views on Oct 15, the
orbital viewer moved ahead of the user-data milestone; on Oct 16 they set "when M7 is complete, move onto
M8".

**Rulings the controller made** are listed at each merge. Examples that stand: a guessed bound may be
replaced by a derived or measured one (and the measurement is recorded), but not by one fitted to pass; a
failing safety test is not `#[ignore]`d (Task 11, round 3); research on one task is capped and the cap is
told to the user (Task 11, rounds 7 and 7b).

---

## What's next

From the ledgers, the plans and the standing order of work:

1. **Finish M7.** Task 7 (forces for the local, nonlocal and partial-core pieces), 8b (GPU force kernels),
   9 (plumbing through `sim`, `providers`, `surrogate` and `app`), 10 (oracles and dynamics re-pins), 11
   (pseudo-H and He, the dimer gate, the H₂O egg-box gate, making pseudopotentials the default), 12 (enable
   Z = 11 to 18) and 13 (bond-angle overlay). Before the merge the full oracle, dynamics and surrogate suites
   are owed on the merged tree. The target is honest C, N and O. The user-requests note sets the oracle:
   VSEPR shapes must emerge (relaxed H2O 104.5°, NH3 106.7°, CH4 109.5° within a few degrees) rather than
   be put in.
2. **Merge the Activity tab** once its suites pass on the merged tree.
3. **The spin-view fix** for the closed-shell noise (queued, Sonnet).
4. **Orbital Task 8:** core shells reconstructed for display, which waits for M7.
5. **M8: locality and transfer.** Per-atom models that carry between scenes, fragment training, a shared
   growing model. The user set M8 to start as soon as M7 is complete.
6. **After M8, unless reordered:** user data (model library, settings profiles), the small queued items
   (per-atom charge with a Hirshfeld readout, bond-angle follow-ups, a refined bond point, auto-sizing the
   cell, a lowest-of-spin-starts teacher that would also help the 3-hydrogen case), and the precision
   milestone.
7. **Later roadmap:** long-range electrostatics (M9), periodic cells with pressure (M10, for crystals and
   "carbon under pressure"), and a nuclear-events layer (stochastic decay on top of point nuclei; explicit
   nucleons are ruled out for scale and for rule 1).

---

## Caveats about this document

- Timestamps are commit times. Agents often worked before they committed, so a "duration" is the span
  between commits and not effort.
- Parallel tracks overlap. The 4-day span is real; adding up milestone durations would double count.
- Branches are still moving. M7 and the Activity tab are unmerged at the time of writing, and `main` is
  at the orbital-viewer merge. Numbers for them come from branch ledgers and `references.md` on those
  branches.
- I did not verify the ledgers by re-running anything; no builds or tests were run for this document.
