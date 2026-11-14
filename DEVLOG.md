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

### Orbital colour follow-ups (Oct 20, about 3 h including two waits for other sessions' GPU suites)

The user compared All orbitals coloured with a textbook viewer (three clean p dumbbells) and found ours
cluttered. One Opus session, no subagents; the user hand-tested the build.

- **Root cause of "two colours per orbital"**: colour slots were already shared across spins, but only
  when a *new* ↓ track overlapped an ↑ row by ≥ 0.9. A pseudised O's ↓ p set was rotated independently of
  its ↑ set, so one ↓ p had no partner above 0.9 and got a fifth colour (legend `#1 #2 #3 #4 #3`). A
  degenerate ↓ set now rotates onto the ↑ rows it lies within, and ↓ rows re-pair with their ↑ partner at
  every update.
- **Pseudised atoms are now aligned to the cell axes** like labelled ones (their rows stay `#k`), and every
  row carries its angular character `l`, which the new **p only** preset reads (off for molecules).
- **Presets** (valence only, one spin, p only, everything), **ticked unoccupied orbitals** drawn as boundary
  surfaces, and a **box warning**.
- **Failure worth remembering**: the box warning was first specified as "the 90 % surface reaches the
  outermost grid plane". It never fired: ψ = 0 on the Dirichlet walls keeps the outermost plane tiny even
  in a 33³ × 0.15 box that pushed O's HOMO from −0.34 to −0.10 Ha. A screenshot of exactly that case caught
  it; the test now uses the box rule's outer-L/6 zone, verified to fire there and to stay quiet in the
  default boxes (O atom, water).

### Viewer capture (Oct 20, first commit 08:44, last 09:33, about 2 h with the build)

Screenshot and Record buttons at the top right of the 3D view (`crates/app/src/capture/`,
`ui/capture.rs`). One Opus session, no subagents, the user hand-testing each commit in the app.

- **Recording by simulated time** (the user's request that morning): a DFT run advances a few fs per wall
  second, so wall-clock frames would give minutes of stillness. Frames are taken per slice of simulated
  time (0.5 fs by default) with positions interpolated between snapshots and played at 30 fps; nothing is
  recorded while paused. When the snapshots are too far apart to interpolate (the estimated solver at full
  speed) the frame is taken at the snapshot and the button says so; when the capture falls behind it
  doubles the slice rather than skipping, so the video stays evenly spaced. An "as seen" mode keeps 30
  wall fps. Nothing waits on the GPU or the disk: a readback pool, PNG writer threads behind a bounded
  queue, and a finish thread that fills skipped slots and encodes with ffmpeg when it is on PATH.
- **The user's hand-tests reshaped it five times in under an hour:** the notes moved out of the button card
  (a long path pushed the buttons aside); hidden-UI captures dropped the cell box and the other editor
  geometry; aspect ratios with a guide frame over the view, an overlay of labels, a molecule title, the
  time and a scale bar (drawn by a second egui context at the frame's size, its labels smoothed on the
  video's clock), and a fixed or orbiting capture camera with dials and a gizmo in the view; the rig was
  then anchored to the cell's centre (it followed the window's pan); and an optional stop-at length.
- **Failures worth remembering:** the first frame-size code let a 4K 21:9 capture in the Camera view ask
  the supersampled glow pass for a 10080 px texture (limit 8192) and the app panicked; frames now fit the
  limit and the glow pass refuses an oversized target. The same PowerShell UTF-8 round trip as the
  builder's garbled two files again before the memory note about it was written.

### Camera pop-out view (Oct 20, about 1 h to the first build, one commit)

A second, real OS window that shows exactly what Screenshot / Record take (`crates/app/src/capture/popout.rs`),
opened from the capture settings' first button or Ctrl K. One Opus session, no subagents; the user
hand-tested the first build and asked for the merge.

- **What you see is what you get by construction:** the pop-out calls the capture's own `render_capture`
  into its own offscreen frame at the capture's size (a 4K capture is rendered at 4K and shown scaled), or
  copies the presented window when the UI is shown, so it cannot drift from the files. It is drawn in the
  main window's redraw, so it follows the camera live; a recording shows its own fixed size, a red dot and
  the frame count.
- **Two traps found by reading before writing:** the capture overlay's egui renderer was shared state; a
  second egui context through the same renderer would have overwritten its font atlas (both use texture
  `Managed(0)`), so the painter now keeps one bloom / glow / egui set per offscreen user (`gfx::Fx`).
  egui's shader expects gamma-encoded textures, so the frame is sampled through a non-sRGB view of the
  sRGB target (sampling the sRGB view would have darkened it).
- **Shared-device care:** the second surface's configure runs in a validation error scope (the solver
  may be submitting; wgpu would otherwise panic), and the pop-out presents in Mailbox when it can, so its
  vsync never holds the main window's frames.

### Training-pass robustness and resumable training (Oct 20, about 3 h, four commits + docs)

The user's report: a hot-burst solve hit the SCF cap after 149 s "and the entire training failed", and could
the active training be saved to disk and resumed after a crash or close. One Opus session; one physics-reviewer
pass on the sampler changes; the user hand-tested and asked for the merge.

- **The diagnosis was in the code, not the logs:** a failed burst step had no forces to continue with, so the
  burst ended — and the burst is the pass's last stage. Now a failed step restarts the burst from the last solved
  geometry with fresh Maxwell–Boltzmann velocities and a cold teacher (a sampler, not a trajectory); three failures
  in a row end it.
- **A per-sample time limit** (5× the pass's median solve, at least 15 s, after three solves) gives a hopeless
  geometry up in seconds; it is logged as *time limit* and treated like a stall. The review caught a real bug
  before it shipped: an abandoned GPU solve parks its half-converged state as the next warm start, so the solve
  after a time-out has to be cold — without that, one time-out could cascade. It also asked for the limit to
  double per time-out in a row (a GPU shared with another job), and for a time-out never to skip a pair's
  shell check. Still open: measuring the restart's cold solve against a warm-dominated median on a big grid.
- **Resumable passes:** the pass state (queued jobs with their geometries, counters, the burst's positions,
  velocities and forces, solved shell checks) is a line-based text checkpoint written with the samples file to
  `%LOCALAPPDATA%\atomic-sandbox\training` at most every 2 s; the Model tab offers Resume / Discard at the next
  start. The one thing that cannot be saved is the thermostat's random stream (`StdRng` has no stable state), so
  a resumed burst reseeds it. A test resumes after every solve of a pass and checks the teacher is asked about
  the same remaining geometries.
- **Process:** the session's own tool trouble was bash heredocs with apostrophes failing to parse; edits went
  through small Python scripts that keep each file's CRLF line endings.

### M7 leftovers (Oct 20, about 7 h wall clock including a pause and three waits for other sessions' GPU suites)

The last items before M8: the four minors the M7 final review parked, and the plan's unfinished visual check
(Task 13 Step 5). One Opus session; one physics-reviewer pass on the shell-fence and shell-check changes
(verdict: ship); the user gave the visual verdict on the screenshots and asked for the merge.

- **Fixed with tests:** the Activity tab's pair curves now refuse a fenced shell band as strict mode does (and
  show an audit fence with nothing quarantined, which they had skipped too); a pass's shell checks are judged only
  against that pass's own refit, otherwise said "stays unmeasured" (before: a silent drop, or an earlier fit's
  verdict); a run without a GPU says the CPU dimer check's ≈ 40 min in the status line and on the model offer's
  row before Apply (it said ≈ 1 min).
- **Ruled, not changed:** the provenance tag keeps hashing the dimer gate's procedure rather than its verdict.
  Every existing model file loads or is refused as before. The premise (each verdict outside the GPU-vs-CPU
  spread: nearest today 0.0037 Bohr against 0.0025) is now asserted by the GPU dimer-gate test.
- **The visual check:** relaxed water under `All` (a new screenshot key `wait=relax`) at 104.36°, the arc and
  label at O. The lone-pair oracle as written ("two lobes on the O isosurface at the published threshold") could
  not pass: design §8 publishes no threshold, and the total density has no lobes at any of 0.05–0.4 × max. The
  lobes are in the HOMO 1b₁, recorded that way and accepted by the user. The shots also caught a display bug: the
  Orbitals tab told the pseudo-O its core was smeared (Zσ = 1.2).
- **Process:** one honesty catch before it landed: a first draft attributed the 0.45° gap to the oracle's 104.8°
  to "the stopping rule"; the oracle was at a different grid too, so the record now says it is not attributed by
  a run. The merge of main met main's resumable passes in the same function (`finish_pass`); a cancelled pass
  deletes its checkpoint, so "the pass ended before its shell check" stays true.

### M8a: one shared model per solver surface (plan Oct 21; first commit Oct 20 16:01, last Nov 12 02:12, merged Nov 12)

About 23 days of calendar time, but commits on only five of them (Oct 20: 4, Oct 21: 12, Oct 22: 11, Nov 2: 11,
Nov 12: 3; 39 non-merge commits, about 22,600 lines added across 111 files). The gaps are the user's weekly usage
cap (the controller stopped at 91 % on Oct 22 and resumed after the reset on Nov 2) and a ten-day pause before the
last two tasks. One controller session; Fable on the first physics tasks and reviews until the user asked on Oct 21
to stop using it for tokens, Opus after that, some re-reviews on Sonnet. The user tested builds between tasks, and four of their
findings became model fixes.

- **What shipped.** The estimated solver learns into one append-only **collection** per solver surface (functional,
  h, σ, teacher, tolerance, ionic surface; not the box) shared by every neutral scene, with isolated-atom references
  per box. Trust is a **calibrated, hysteretic verdict per environment** (class, central element, neighbour
  species) in place of the old leave-one-out knife edge. The basis gained **four-body** invariants, refits became
  interactive at the fit (≈ 0.17 s per added sample at 10,000 samples; the leave-one-out trust checks around it still
  take ≈ 20 min at that size, a follow-up), and a collection can be cut into a **specialised model
  file**. The **knowledge map** (a dock tab, its pop-out and a 3D overlay) shows what the model knows. A **blur**
  slider for the density dots came in on the side.
- **User-test fixes.** A shared H collection stuck untrusted: a recovery degree cap set once was never revisited,
  and one stretched H₂ label spread its misfit over every sample (fixed: the cap is re-judged). Water fell back on
  every step because one hot-H₃ point blocked the whole many-body class (fixed: the verdict is kept per environment
  key and re-judged as evidence arrives). A Standard pass on a single H atom made a fresh collection untrusted (fixed:
  **a lone atom is never trained on**; a pass on one is refused, and stored lone samples are held out at load).
  Training N₂ into an H/O collection grew the model from 452 to 1,590 columns and every key, water's included, read
  "not learned yet" (fixed: **each key's sample floor is the columns it touches**, not the whole basis).
- **An honest negative: fragments do not transfer.** The transfer test taught a collection on CH₄ and C₂H₆ (5.0 h of
  GPU) and asked it about propane. It refused all 220 propane atoms at both a 6 and a 9 Bohr cutoff, and the refusals
  were right: its answers there would have been 3 to 14 Ha/Bohr off. Each carbon of propane sees three carbons, the
  fragments never more than two. The criterion therefore "passed" with nothing trusted. The ruling was to print
  VACUOUS and write "transfer not demonstrated", not to add an assertion that would just fail. The water half
  (6 to 10 h estimated) was not run.
- **The capping gate failed, 12×.** "Train this segment" cuts part of a large molecule out and caps the cut bonds
  with hydrogens. Its gate compared butane taught on capped segments with butane taught whole, on held-out
  whole-molecule forces: 11.96× worse against a 1.5× limit, and a control showed the interior rows alone lose
  nothing (1.00×), so the capping is the whole error. The controller's diagnosis: an atom's force sums the energy
  derivatives of every centre within r_cut of it, and each of those centres sees r_cut further, so a force row is
  only right when everything within **2·r_cut** is the real molecule. The cut had used r_cut. Corrected (2·r_cut plus
  the pass's largest planned move, ≈ 14 Bohr), butane has no interior atom in any proper segment. The user chose to
  defer the feature and merge: the code and its tests stay, the buttons and the Ctrl K entry are hidden behind one
  constant, and a gate on a longer chain (decane or longer) is the follow-up.
- **Process.** The gate's first run died with an old session after about 2.2 h of solves; the stored teacher answers
  let the Nov 12 run finish in minutes (`CAPPING_REUSE=1`). The knowledge map's UI waited 13 days for the user's
  mockup approval, which came on Nov 2 with no changes. A deferred-before-merge honesty fix from Task 2 (the
  calibration's gap is bounded by the audit cadence only for synchronous audits, not at the app's background
  cadence of 2,000 steps) was carried through three stops and landed in the docs at the end.

### Docking UI and orbitals by atom (Nov 3 and Nov 11–12; branch `docking`, cut from M8a's tip)

The View menu had become "such a mess" (user, Oct 21), and the two fixed side docks could only hold tabs on the
left or right. The user picked a full docking workspace from a brainstorm with mockups: `egui_dock` for the
tree, the View menu regrouped into one View panel, duotone icons, pins that auto-hide (Visual Studio style),
layouts stored in the profile, and four presets (One monitor, Two monitors, Training, Presenting). Spec and
plan landed Nov 3 12:25. That afternoon one Opus implementer per task built D1 and D2: the layout grammar, the
dock tree, the icons, the View panel, auto-hide and the presets. D3 (detaching panels into their own OS
windows) was committed at 17:14 when the user asked to stop, unreviewed. Eight days later the user resumed:
- **Detach:** the review found two bugs that bite on the user's three monitors. A window saved maximised on the
  right-hand monitor overlaps the primary by 8 px, so the "on no monitor" check let it open off-screen when that
  monitor was unplugged (now: 64 × 8 px of its title bar must be on a monitor). "Reset layout" from a tab's menu
  closed Activity and Graph, because it mixed deferred and immediate changes while the tree was lent.
- **Drag tabs between windows (user decision Nov 12):** floating panels inside the window went. A tab dragged
  out of every window now opens a new one there, and a drag over another window shows the same drop targets. The
  review caught one invisible bug: when windows overlapped, a drop on nothing still let the covered window apply
  its own drop underneath. The source tree is now snapshotted during a drag and restored. The user's hand-test:
  "Works nicely".
- **A press the UI takes never moves the camera** (user report: resizing a panel near its edge also orbited).
  egui-winit's `consumed` is last frame's `wants_pointer_input`, which misses the few-pixel grab margin outside a
  widget's edge; the fix asks egui's own hit test.
- **Small user requests on the way:** the Activity job list stopped jumping between iterations and steps; a
  closed shell's identical ↑ and ↓ orbital lists fold into one ↑↓ list (and, after the user found He still
  showing two, also when one spin shows an extra unoccupied row).
- **Orbitals by atom (spec and plan Nov 12, five tasks, about 4 h).** The user wanted to see "which orbitals are
  mostly on that atom". A molecular orbital belongs to no atom, so the design uses the split the partial charges
  already use (Hirshfeld weights from this solver's own free atoms) and says so in the tooltip. The rows group
  under atoms at ≥ 55 % (or "shared · …"), and selecting atoms dims the rest. `sim` hands each nucleus's radial
  table to the provider as data (rule 5); the CPU path sums over `CpuView`'s ψ̃, a new WGSL kernel does the same
  on the GPU (agreement 5e-7; 0.5 ms at H₂O 95³). Degenerate sets show their average share, since a single
  rotated member's share depends on the display's arbitrary rotation. The oracles cross-check against the
  literature: water's q_O from the orbital shares is −0.298 (Hirshfeld ≈ −0.3), LiH's q_Li +0.397 against the
  charges suite's +0.4025.
- **Gate (Nov 12, 8f54423):** core 480, providers 308, validate orbitals 12, app 536 passed, nothing failed.

---

### Refit cost at scale (Nov 12, first commit 14:16, fix 4 at 16:03; four fixes)

The user asked: "what happens when we end up with 50,000 samples, are we going to be waiting hours?"

**The measurement.** A timing harness ran on a copy of the user's 1,805-sample H/N/O collection (Q′ 1,032), on species subsets and on jittered replicas up to 14,440 samples. The hyperparameter search, the solve and the collection's own writes do not grow with P.

The per-refit trust pass grows linearly: an exact leave-one-out over every sample, 4.6 ms each at Q′ 1,032 and growing as Q′². The trust recovery then repeated the whole fit seven times per refit. The user's "Refitting · 1:34" was 90 % recovery. Extrapolated, a refit at 50,000 samples would take about 35 minutes, and the simulation stops for all of it.

**The fixes.**
1. The trust passes run once per refit, for the fit that stands. They are bit for bit those of a pass after every fit, compared line by line.
2. The leave-one-out runs on threads, bit for bit.
3. A rebuild's accumulate is batched on threads, bit for bit a loop of `add`.
   - The downdate first proposed was dropped. Subtracting samples from compensated sums is not bit-exact, and the code promises that a reopened collection's first fit equals the session's last search bit for bit.
   - The surprise was that the old rebuild spent most of its 0.8 ms per sample spawning threads.
4. While playing, a refit runs on a copy of the provider in the background. The install keeps whatever the live provider appended meanwhile. Anything else discards the result, which is due again.

**The numbers.** The user's refit went 100 s → 25 s → 21 s. A cold rebuild of 13,043 samples went 10.8 s → 0.45 s. At 50,000 samples a refit is extrapolated at about 1–1.5 min, and while playing it no longer stops the simulation.

**What was wrong along the way.** An independent review of fix 4's merge found two real holes:
- A box reference landing mid-refit would have put stale energies back under new references.
- A key that left trust on live evidence could have re-entered on the install's one re-judge.

Both are fixed, with tests. The second test fails without its fix. The merged tree's app suite then caught a third: a background refit in flight when the user paused waited for the next command, because the idle worker blocked on its inbox. A pause now keeps polling until the new model stands.
- **Gate:** unit crates, `providers`, `app` and validate `surrogate` on the merged tree before the merge.

### One bad sample, and a Forget button (Nov 12–13; branch `sample-recovery`, two tasks)

The user's shared collection went untrusted after a session of play. A debug agent replayed the collection's own
history from snapshots and found the cause:
- **The cause.** One sample, #1804, a lone OH radical squeezed to O–H 1.31 Bohr with 0.72 Ha/Bohr on each atom.
  The fit reproduced it within its own (large) bound, but it bent the shared O–H pair term, so 34 water and H₃O
  samples failed: the worst ratio went from 1.00 to 2.36.
- **Why recovery missed it.** Recovery ranked samples by their own training residual, so it never named #1804. It
  quarantined the victims instead, round after round, and ran out of rounds at exactly the user's 3.7e-3. The
  leave-one-out residual, which the class rule already computes every fit, put #1804 first at 38× its bound.

**What landed:**
- **Task 1:** recovery rounds rank by the leave-one-out ratio.
- **Task 2:** a Model-tab list of the worst-fitting samples, with Forget and Restore (code 7, written to the
  collection at once).

**It took more rounds than planned:**
- **Stuck-H regression.** The leave-one-out ranking alone made the stuck-H test (the user's earlier H collection)
  worse: trusted at 0.961 became untrusted at 1.60, because after the right sample it kept holding samples that
  fit fine. The ruling: leave-one-out rounds first; if they do not recover, the old rounds from the same entry
  state, kept only if they recover. The recovery is never worse than before.
- **A plan expectation that was wrong.** The plan expected the user's pre-1805 snapshot to end trusted with under
  110 held. The debug number had skipped the release step that runs first. Measured: the first refit ends just
  untrusted (1.0185×, 128 held; the old ranking 1.46×, 165 held), and the next refit is trusted (0.995, 122
  held). Both are pinned in `references.md`.
- **Reviews caught three safety holes before merge:**
  - forgetting a failed audit's own sample let the next refit drop the audit unjudged, so one click could make
    the model trusted;
  - a step could be skipped when the leave-one-out result passed its residuals but lacked samples;
  - next to a forgotten sample every agreeing DFT answer was kept unthinned, for as long as it stayed forgotten.

  Each has a test that fails without its fix.
- **Honest wording, more than once.** Notes that said "the model cannot represent them" or "breaks the fit of
  their neighbours" claimed more than a leave-one-out ratio shows; they now say what was measured.

**Gate (Nov 13, before merge):**

| Suite | Passed | Failed |
|---|---|---|
| providers | 358 | 0 |
| validate `surrogate` | 7 | 0 |
| app | 554 | 0 |
| surrogate | 95 | 0 |
| sim | 142 | 0 |

Parked: one diagnostic count (`box_blocks_trust`) double-subtracts a forgotten box-affected sample.

### Bonds between solves, from experience (Nov 12–13; branches `stale-bonds`, `bond-experience`)

In estimated mode the density-based bond graph is only as fresh as the last DFT solve, so a stretched H₃ kept
showing its bonds after it came apart. First, stale bonds were drawn dashed with a label, and a bond refresh
ran about 1 s after a drag. Then the user asked for bonds "guessed from experience":
- **How it works.** Every fresh DFT graph is recorded pair by pair: the two Z, each end's crowding (the number of
  neighbours within 8 Bohr, capped at 6), the distance and the verdict. The examples go into
  `bond_experience.txt` in the collection's folder. Over a stale graph, a pair is shown bonded or not bonded only
  when at least 3 examples of the same key within 0.1 Bohr all agree. Otherwise it stays dashed.
- **Within rule 1.** It is learned from the app's own DFT, not typed-in chemistry. It is also display only:
  names, charges and the Selection panel keep the DFT graph.
- **The review found the speed problem.** The first version scanned every stored example near each pair, on the
  render thread, even with the setting off: 8 ms per frame at 300 atoms. With per-key running counts each lookup
  became two binary searches (0.29 ms for all the lookups at 300 atoms on a full store). Nothing is computed when
  the setting is off, and the result is reused while nothing moves. The timing test is a measurement
  (`#[ignore]`), not a gate.

### The Windows installer and the in-app updater (Nov 12–13; branch `installer`, six tasks)

The user asked for a real installer: download it, click Next, and it opens on H₂. They also asked for in-app
updates, with a way to pick older versions and betas, all over GitHub Releases. Linux was dropped because the
user can't test it.
- **What landed:**
  - **Installer.** A per-user Inno Setup installer.
  - **Data folders.** Three modes: dev, portable and installed. Data lives in `%LOCALAPPDATA%\atomic-sandbox`,
    seeded from `bundled\`, and re-seeded add-missing-only at the first launch of each version.
  - **Version panel.** A chip and panel showing stable and beta builds, with downgrade warnings driven by a
    format manifest.
  - **Downloads.** Fetched from GitHub hosts only (an exact four-host allowlist, manual redirects), then checked
    against SHA-256 and a minisign signature whose trusted comment names the tag.
  - **Restart now.** It never interrupts a solve, refit, pass, relax or audit without the user's click. It
    resumes the session after the update.
  - **CI.** GitHub Actions builds every tag, signs in a tag-only job behind the `release` Environment, and opens
    a draft release.
- **Security review changed the CI design.** The first workflow could sign from a branch build. The signing
  key now lives only in a tag-only job, and that job downloads nothing but `SHA256SUMS.txt`.
- **CI's first three runs failed on one line.** Inno Setup's compiler has no version resource, its banner shows
  only the major version, and `ISCC /?` exits non-zero. Run 4 compiled the `.iss` for the first time.
- **The final review caught a restart that didn't wait.** Restart now didn't wait for a paused solve. The
  re-review then found three more solves (Bonds now, the drag refresh, the estimated graph solve) that run while
  the worker reports idle; the guard now also reads the probe's Main slot. It also corrected the record: the
  app exits through `process::exit`, so a CPU solve can never hold the installer's 120 s wait.
- **Not yet run:** the sign job, the release job and the uninstaller's `[Code]`. No key is committed and no tag
  is pushed. These wait for the user to set up the `release` Environment and to hand-test the first beta draft.

**Gate (Nov 13, merged tree):** app 673 passed, 0 failed, 4 ignored; `surface_reconfigure` 1 passed.

### One model for radicals and closed shells (Nov 12–14; branch `claude/charming-keller-0bc26b`, nine tasks plus two added)

Diagnosis Nov 12 20:58, spec Nov 13 00:40, last task commit Nov 14 00:23. 22 commits on the branch before the main merge, about 4,200 lines added across 31 files against main.

The user's collection failed its training residuals at every refit. The diagnosis found two causes:
- **A basis that was too small.** At body-4 degree 2 the shared O–H terms could not fit H₂O and H₃O at once (H₃O worst at 1.46× its bound).
- **A recovery that ratcheted.** Each refit's recovery held more samples, failed anyway and kept them, reaching 180 held.

Measured before the spec: degree 3 fits the collection and releases 68 holds; degree 4 was 46× slower and no better.

**What landed:**
- **G1:** the four-body degree is recorded per basis and grows from 2 to 3 with the store. The user accepted the cost: 1.1–1.4× a degree-2 refit at three species, ≈ 18 min for a cold attach at four species and 10,000 samples, over the spec's 5-minute stop line.
- **G2:** a recovery that does not restore the training residuals holds nothing (one that restores them short of samples keeps its holds, untrusted).
- **G3:** the samples the model still cannot reproduce (M, at most 10 %) are refused locally (`NearMisfit`) instead of untrusting the whole model.

**What the reviews changed:**
- **The physics review found a hole in G3.** The spec left M out of the class rule entirely. The near-misfit fence covers M's sampled points, not the band between them, so a geometry in the band whose environment key M shares would be answered as trusted. Estimated mode's audits catch that; Sandbox (M8b) has none.
- **The user chose C1.** M stays in the class rule's average (RMS) terms, so a key with far-off misfits is refused. A wider fence (C2) was measured first and rejected: at 0.75 spreads it flagged 67 % of H₂ atoms on a variant with M populated.
- **The C1 review found that the energy RMS decided nothing.** The physics review had assumed it did. The user then added the energy half: a key is refused when its energy RMS, misfits included, exceeds 1 — only when a misfit contributes, so with M empty nothing changes.
- **Task reviews caught:** a loosened column floor, a recovery note that contradicted the trust badge, and answers near M kept on every step.

**Acceptance on the user's collection (a copy):** trusted at degree 3, held 180 → 112 at the attach, then 88, 71, 54, 45, 44, 43, 42, 42. The fixed point comes at refit 8, not within the spec's 5; the user accepted that. Holds never rise, and |M| is 0 throughout, so G3 and C1 are not what make this collection trusted. Force errors are at or below degree 2 in every scene on both comparisons. The exception is HO's leave-one-out on its own fitted set: one sample, refused at both degrees for having one sample; the user accepted it.

**The main merge** composed main's leave-one-out recovery ranking and Forget (code 7) with the no-ratchet restore. One change was made to main's rounds so that "never worse than before" still holds under the restore. Main's pre-1805 snapshot numbers in `references.md` (128 and 122 held) were measured before the merge and not re-run.

**Gate (Nov 14, merged tree, each suite alone):**

| Suite | Passed | Failed |
|---|---|---|
| units, qm, sim, surrogate | 486 | 0 |
| providers | 384 | 0 |
| validate `--test surrogate` | 7 | 0 |
| `trust_baseline_report`, cadences 50 and 10 | identical to main at 50; met at 10 | — |
| app (incl. replays) | 675 + 1 | 0 |

The trust baseline's cadence-50 hot run had risen against its Oct 21 record (worst 5.15× against 2.87×). Main gave the identical report on the same machine, so the rise is main's, not this branch's. It is an open item for main. The app suite first failed to compile: the merge had added a field that one of this branch's tests did not set.


### The UI redesign, purge and a box that fits the cloud (Nov 14; branches `thermostat-bar`, `model-tab`, `orbitals-tab`, `purge-forgotten`, `estimated-box`)

All of this came from the user's play-testing in one day. They saw the mockup of each panel inline in the chat and approved it before it was built.

**The panels:**
- **Model:** a verdict card, Teach next, Health, and folds for the rest. The bookkeeping (every old row) moved into Details.
- **Activity:** a job card with a convergence plot. The plot has a fixed log scale and keeps the previous three solves as ghosts, so it no longer flashes. The card also has an iterations-per-sample strip, a session timeline with a failures lane, and a solve log that folds repeated rows.
- **Orbitals:** an energy-level diagram, compact rows, and folds for box states and the honesty texts.
- **Knowledge:** a summary chip, scene pair chips, and a pair card with a "now" marker and Teach this.
- **View:** four looks, called Clean, Glow, Scientific and Present. Glow is the user's own style, saved.
- **Readouts:** it absorbed the Graph tab. Its ΔE axis is in µHa, because the old graph scaled every curve to its own range and drew 1e-6 Ha noise as a full swing.
- **Selection:** an atom card and one neighbours list that merges distances and bonds.
- **Also new:** a Thermostat group in the bottom bar, keys 1–4 for the tools, and zoom and pan on the knowledge map.

**What the user caught by eye:**
- There was no right-hand padding.
- Inside the new cards, slider rails and preset buttons were invisible, because the cards' fill matched the widget fill. The fix went into the shared card helper, with a contrast test.

**What the reviews caught:**
- A resize estimate was 31 pt off, because it didn't know the toolbar's new compact height.
- The full toolbar drew its bottom 3 pt off-screen.

**Purge and the fence lift:**
- **The bug.** Forgotten samples fenced their geometries forever. Agreeing DFT answers there were thinned as "similar", so H₂ fell back on every step once the thermostat or a drag reached the fenced band, whatever the user trained.
- **New rule.** A forgotten sample's fence now lifts after three consecutive fresh answers agree with a model fitted without it.
- **Purge.** It deletes forgotten samples for good, with a one-file backup, under the collection lock, and the swap-in is crash-safe.
- **The review found a Critical.** A lift could be voted by a model still trained on the forgotten sample, in the window before the refit. It also found:
  - the lift used the union of the votes' fenced sets instead of their intersection;
  - a post-rename error that left memory and disk out of step;
  - a replay that treated purge as harmless.

  All were fixed and re-reviewed.
- **The user chose "purge lifts, warn".**
- **Purging their 45 forgotten H₂ samples ended the fallbacks.** The user: "after purging it works amazing".

**The estimated box:** H₂ in the 12 Bohr preset box was box-clean only when centred, because the H tail falls below the face rule only about 5 Bohr out. In Estimated mode the box is now extent + 2 × (cloud reach + 2 Bohr), with the reach taken from the app's own isolated-atom solve. H₂ gets 109³ instead of 81³, and a DFT step costs about 2.4× more.

**Gate (Nov 14, merged tree):**
- app: 736 passed out of 737.
- One test hit its 120 s snapshot wait while the user's app shared the GPU. It passed alone.
- `pause_and_cancel_abort_a_sampling_solve_promptly` failed twice under GPU sharing (Pause took 1.7 s against its bound). Alone it took 681 ms and passed.
- purge: surrogate 99, providers 399, validate `--test surrogate` 7, all passing.

**Paused once.** The user suspected their GPU was overheating. Everything was stopped, including a measurement script that kept relaunching itself. Since then GPU work has run one job at a time, in batches of 15 min at most, with cool-downs between them.

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
- **Docking (Nov 11–12): tests and agents that tripped over each other.** On Windows a running test exe can't
  be relinked: a focused test run failed with LNK1104 while the full suite still held the binary, and cargo
  does not hold its lock while tests run. Every app test also links the app binary (an integration test needs
  it), so a fix round waited about 48 minutes for the user to close the app. A lag report sent a debugging
  agent to build two versions side by side; the user then found every window on the PC lagged, so the agent
  was stopped before measuring anything, and its 1.2 GB build folder deleted. A controller probe also added a
  blank tab to the user's open Notepad (Windows 11 Notepad is single-process); nothing was lost.

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
   growing model. M8a (the shared model) merged on Nov 12, with fragment transfer measured as not working and
   Train this segment deferred (see the timeline); a long-chain capping gate is its follow-up.
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
