# User requests captured during M3 play-testing (2026-10-13) — input to the M5 (tools) plan

Landed immediately on `m4-dynamics` (small, app-only): middle-drag orbit / right-drag pan
(00ebded); distance guides while dragging or placing (7e263e3); display aids — cell box, floor
grid with axes, placement-plane grid, 1 Bohr / 1 Å spacing (this commit).

## For M5

1. **Bond graph from the density, not from distance tables.** Two nuclei are bonded when the
   electron density along their internuclear segment never drops below a threshold (a bond
   critical point in Bader/QTAIM terms). Element-agnostic (rule 1), computed on the sim side after
   each solve (small readback or a GPU line-sampling kernel), published in the snapshot. Draw thin
   bond lines; connected components are molecules.
2. **Molecule labels.** Hill formula with charge (H₂, H₂O, LiH, H₃⁺) next to each component.
   Programmatic structural names where the bond graph defines them (binary hydrides/oxides/
   halides, *peroxide* = O–O single bond with both O further bonded, simple hydrocarbons), plus a
   small trivial-name table (water, ammonia, methane, hydrogen peroxide, …). The name table is
   display data like element colours — allowed; it never feeds physics.
3. **Isotopes** came up ("tritium"): currently every element uses its most abundant isotope's
   mass (`units::isotope_mass_amu`). A per-nucleus mass number is a legitimate M5 addition (mass
   is one of the two allowed per-element inputs); it only matters once M4 dynamics exist (isotope
   effect on vibration frequencies — a nice validation: H₂ vs D₂ ω ratio √2).
4. Honest expectations to put in the UI copy: H₃ is not a stable molecule (H₃⁺ is); bonds and
   names are *read off* the density, so they appear only after a solve converges.

## Beyond M5 (user ask 2026-10-13): pressure, crystals, "carbon under pressure → diamond"

Needs a periodic-cell milestone (candidate M6), in this order:
1. **Periodic boundary conditions** in `qm`/`qm-gpu`: Γ-point Bloch orbitals, periodic Poisson
   (FFT or periodic multigrid), Ewald-style nuclear interaction. Multipole boundary stays for
   molecules. Most kernels carry over; boundaries change.
2. **Stress / pressure**: P = −dE/dV from the stress tensor or an E(V) scan; egg-box care when
   the cell strains (grid moves with the cell).
3. **Barostat** (NPT: Parrinello–Rahman / Bussi isotropic) — the pressure counterpart of CSVR.
4. **Pseudopotentials** (already the roadmap route past Z = 10) — all-electron C at h = 0.25 is as
   under-resolved as O; carbon crystals are not honest without them.
Validation oracle: graphite vs diamond E(V), bulk moduli, and the enthalpy crossover
H = E + PV at a few GPa (LSDA gets the order of magnitude right; cite the literature value).
Honest limit to state in the UI: spontaneous graphite → diamond is nucleation-limited and does
not occur in ps-scale MD of small cells at any time-speed setting; offer seeded cells or a bias
(metadynamics-style) tool later, labelled as such. Cheap interim: a repulsive confining wall
("piston") in the finite box as a cluster-pressure proxy, labelled as not-a-crystal.

## Decision 2026-10-13: next milestone after M4 = the surrogate ("estimated") provider

User ruling after testing M4 ("no usable frame rates"; option 1 chosen): **M5 = ML surrogate force provider**, the
tools list above moves to **M6**. Shape agreed in chat:
- a sampling pass drives the real solver (GpuKs) over geometries near what the user built (and random
  perturbations / short hot MD), collecting energies + forces as training data — Z-only physics, nothing
  fitted to experiment (rule 1 holds);
- an in-Rust fit of a body-ordered, linear-in-descriptors model (ACE-style: least squares, no autodiff,
  no ML framework — rule 2) with an ensemble for uncertainty;
- `SurrogateKs: ForceProvider` with a live uncertainty readout; when uncertainty is high the step falls
  back to DFT and the result is added to the training set (active learning), visibly;
- an "estimated" toggle in the panel; honest labelling that the surrogate is valid only where trained;
- validation: the surrogate reproduces the DFT H₂ frequency / energy surface it was trained on, the NVE
  drift suite passes on it, and out-of-distribution geometries trigger fallback.

## Decision 2026-10-13: nuclear physics — a "nuclear events" layer, never explicit nucleons

User asked whether protons and neutrons should be simulated explicitly (for nuclear physics and decay
later). Ruled, with the user's agreement:

**Permanently out:** explicit nucleons. (1) Scale: the nucleus is ~1e-5 of the atom; resolving it needs
~1 fm grids (1e15× more points) and 1e-22 s steps — not computable. (2) Theory: no first-principles
route from QCD; real nuclear-structure codes use effective interactions fitted to nuclear data, the kind
of per-system fitted input rule 1 forbids — the emergence-from-fundamentals story honestly stops at the
nucleus. (3) Decay spans 1e-24 s to 1e24 yr and is a stochastic rate process, not something to time-step.

**In, as a later milestone ("nuclear events", after the surrogate (M5), tools (M6) and the pseudopotential
/ Z > 10 work it depends on):** each nucleus carries a nuclear state (Z, N, excitation) and a stochastic
event process on top of the point-nucleus electrons:
- Inputs: isotope masses (AME2020 — already loaded) give Q-values and allowed decays as mass
  differences; measured half-lives and branching ratios (NUBASE) are a documented extension of rule 1
  from "Z and mass" to "Z, mass, and measured nuclear decay data" — measured properties, not fitted
  behaviour.
- Events: α (nucleus → Z−2, A−4; a new He nucleus with the Q-value shared by momentum conservation —
  MeV, leaves the cell in a fs); β⁻/β⁺/EC (Z ± 1, electron count changes → ion; the emitted lepton is not
  tracked); γ/isomeric (state only). The electrons respond through the existing DFT layer — live
  transmutation with nothing per-element added; decay chains once Z > 10 exists.
- Time: a separate, labelled "nuclear clock" multiplier (decay rates ×10ⁿ) — never folded into the MD
  time speed (same principle as rule 8).
- Cost: negligible — rare events checked per step; the post-event chemistry is a cold solve the solver
  already handles.
- Pairs with the M6 isotope item (per-nucleus mass number makes decay meaningful).
- Display (ruled 2026-10-13): nuclei stay single spheres (the marble-cluster nucleus is a counting
  cartoon; nucleons are overlapping wavefunctions and the nucleus is 1e-5 of the atom — our sphere is
  already a labelled legibility choice). Show composition as DATA: a hover/select badge "⁷Li · 3 p · 4 n"
  (+ half-life/decay mode with the events layer), mass number text, rim colour by Z. An explicitly
  labelled "schematic" marble toggle is acceptable for teaching, never the default. → M6 / nuclear events.
- VSEPR (user question 2026-10-13): not implemented — it must EMERGE. Oracle for the pseudopotential /
  Z > 10 milestone: relaxed H₂O 104.5°, NH₃ 106.7°, CH₄ 109.5° within a few degrees (LSDA literature
  values cited), lone-pair lobes visible in the isosurface. At h = 0.25 all-electron O/N/C are too
  smeared (water ~177°). M6 tool: a one-click **relax** (FIRE / damped MD on the forces) so shapes can
  be found without a thermostat run.
- Smooth motion (user request 2026-10-14, landed in M5 Task 8's fix round): render-thread interpolation
  between the last two snapshots (one publish interval of latency; snaps on count changes; off above a
  0.5 s gap) with a "smooth" toggle; plus the honest aliasing note — above ~30 fs/s a 10 fs vibration is
  faster than the frame rate (shimmer, not swing) — and a "watchable" speed preset (one vibration ≈ 0.5 s
  wall). The surrogate's speed is for slow processes, not for watching one bond.

## Roadmap to "multiple large molecules interacting" (user question 2026-10-14)

The surrogate removed the dynamics bottleneck (µs/pair, thousands of steps/s at 200 atoms); the
remaining bottleneck is TEACHING (DFT cost ∝ grid × electrons; a 50-atom organic ≈ 1e3–1e4× H₂ per
solve) plus two missing pieces of physics. Order after M6 (tools):
1. **Pseudopotentials** — generated from our own all-electron atoms (Z in, potential out; rule 1
   holds) so C/N/O are honest at an affordable grid; opens Z > 10. Biggest single job left.
2. **Locality and transfer** — the ACE model is local by construction: teach on small fragments
   (dimers, motifs), apply to large molecules; needs fragment sampling, the S = 4 performance
   fixes (spectral Σ, warm evidence), per-atom "new environment → fallback" (already there).
3. **Long-range electrostatics** — learned per-atom charges + explicit Coulomb/Ewald term; the
   6 Bohr cutoff cannot see dipoles across molecules; also the route to intermolecular charge
   transfer.
4. **Periodic cells** (the pressure/crystal milestone) for condensed phases.
Expected experience: after 1 — honest small molecules reacting live; after 2+3 — hundreds of
atoms on the surrogate with DFT fallbacks only at genuinely new environments; after 4 — solvent,
crystals, pressure.

## "Draft" mode — estimated without fallback (user request 2026-10-14; first small task after the M5 merge)

`FallbackPolicy.never_fallback` (UI toggle "guess instead of falling back (unsafe)"): the model
answers every step (extrapolating beyond its envelope / trust); background audits keep running and
the model keeps learning from them (samples appended, refits rate-limited) but never blocks on them.
Honesty surface: a red banner while any atom is outside the envelope ("extrapolating: N atoms beyond
what the model has learned — forces here are an educated guess"), atoms tinted by their per-atom
uncertainty, "last audit: model off by X Ha" with its step, and the same one-surface bookkeeping
(drift/work rows stay meaningful). Provenance/species/charge mismatches still refuse (there is no
model to guess with); the toggle is per session, off by default, and the README states what the
guess is (beyond r_cut a bond contributes nothing; new species/charges are confidently wrong).
- As built (branch m5b-viewer-extras): draft answers past rule 4 (untrusted), 5 (envelope), 6–7 (variance) and
  the quarantine fence; rules 1–3 still go to the teacher. `EvalStats::extrapolation` (atoms outside, the
  trust reason, the rule that would have fallen back, per-atom envelope distance) and `atom_uncertainty`
  (per-atom force std ÷ bound) feed the banner and the tint (the tint uses the larger of the two: at R = 4
  Bohr the variance ratio was 0.07 — a linear model's variance is small where its features vanish — while
  the envelope distance was 6.8× the limit). Audits only in the background in draft mode (an in-step one
  would be a teacher step); a failed audit untrusts the model and the flip is booked when draft is switched
  off — after a failed audit only; switched off while untrusted for another reason (training residuals, too
  few samples), or switched on from an untrusted (teacher-surface) state, the surface changes without a
  booked jump and the conserved-energy history restarts with a note (the pre-existing untrusted-refit path).
  Fix round 1 (review I1): a refit while untrusted in draft mode books its jump (the next answer is the
  model's). "last audit: model off by X Ha at step N", N = the verdict's snapshot step less its audit lag.
  The history marks guessed steps (`history_guess`): the sparkline draws them red.

## Container walls (user request 2026-10-14; queued with draft mode after the M5 merge)

`Scene`/`Dynamics` option "container walls": a smooth repulsive wall potential U = ½ k_wall (d₀ − d)²
for a nucleus closer than d₀ to any cell face (d₀ = the providers' 3h margin + ½h, so the density never
reaches the wall), force into the box, booked into E_conserved as a potential (like the drag spring)
so drift/work stay meaningful and rewind stays exact; k_wall a fixed constant (not per-element). Off by
default; when on, the "left the simulation cell" pause never fires and the wall is drawn as a faint inner
box. Not a hard velocity flip (would break reversibility and the bookkeeping). This is also the piston
pressure proxy: a later "shrink the box" control compresses clusters honestly (labelled not-a-crystal).
- Adjustable container (same request): the wall box has its own half-width slider, ≤ the DFT cell
  (the cell is the physics limit; growing it stays "Apply grid"); changing it while playing moves the
  wall, whose work on the atoms is booked like a moving drag target (W += Σ F_wall·Δwall), so
  shrinking the box is an honest piston for clusters.
- As built (branch m5b-viewer-extras): the slider sets the wall's *face* L (≤ the cell less 3h + h/2);
  the soft layer is the 1 Bohr inside it, and the wall is C², not ½ k d² — u'' ramps 0 → k over 0.5 Bohr,
  k = 3.5 Ha/Bohr² — because the C¹ quadratic random-walked the kinetic energy ≈ 1 % per bounce (20 % after
  300 bounces) for want of a Verlet shadow Hamiltonian across the curvature jump; the C² wall: measured
  max |K/E − 1| = 1.05e-3 over 306 bounces (test bound 2e-3; `validate/references.md` "Viewer extras (M5b)").
  Stopping bound: ≤ 1 Ha per axis turns before the face; the margin guard fires above 1.38 Ha (h = 0.25).
  `sim::dynamics` module doc, "Container walls", has the derivation and the measured overshoots.

## Density in estimated mode (user report 2026-10-14: "cloud and isosurface don't work in estimated mode")

The surrogate has no density; the viewer shows the last DFT density (faded, labelled), refreshed only
on fallbacks/audits — frozen at a stale geometry while the atoms move. Fix, queued with draft mode:
1. publish the background auditor's density (a real solve at the current geometry, currently discarded)
   labelled "DFT density from the audit at t = …", plus a "refresh cloud every N s" control that
   triggers a display-only background solve (≈ 0.1 s each; a few % overhead at 1–2 s);
2. between refreshes, a moving "atomic superposition" cloud: the isolated-atom densities already
   computed for the references, placed at the current nuclei, blended with the last real density,
   labelled "atomic superposition — not solved" (shows where atoms are, not bonds);
3. both apply in draft mode.
- As built (branch m5b-viewer-extras): `ForceProvider::side_jobs`/`take_side_display` carry the auditor's
  densities to `DynSnapshot::side_display`, stamped by the driver with the dispatch geometry's time; the
  auditor's GpuKs now has display on; refreshes and isolated-atom solves are queued on the auditor ahead of
  the next audit (at full speed it never idles). The superposition profiles come from one isolated-atom solve
  per species and grid on the auditor (the reference cache may hold no solve at all after a model load), read
  back once on the worker; the viewer builds the superposition at ≤ 30 Hz and blends it in the shader with
  the last real density (the crossfade's two-texture binding), weight = max displacement / 0.5 Bohr. CPU
  teacher: neither exists (no background solver) — its audits are in the step and show their own density.
- Smooth the density too (user request 2026-10-14, queued with the post-merge viewer task): crossfade
  the previous and current density textures in the raymarch/isosurface with the same α as the nucleus
  interpolation (bind two volumes, lerp per sample), so cloud and spheres move together; snap on grid
  changes or dims mismatch; off when "smooth motion" is off. Display-only (a blend of two solved
  densities, not a transported one) — no label needed beyond the existing smooth-motion tooltip.

## UI redesign (user request + approval 2026-10-14)

"Clean, modern, user-friendly but super in-depth": bottom toolbar (transport, speed + fs/s +
watchable, dt, tool modes with the element picker, selection actions, solver chooser with the source
badge/trust light), top menu bar (File / Scene / View / Physics / Help), collapsible right dock with
tabs (Readouts, Sparkline, Model, Selection, Log), in-view overlays (status chip, atom card, banners).
Hard rule: no control or readout is lost — the mockup's "current control → new location" table is the
acceptance checklist. Mockup published as an artifact first (agent-built); implementation approved by
the user in advance; lands on its own branch after the viewer extras (same files).

## M6 notes — solver follow-up from the tools suite (Task 7 review I1, 2026-10-14)

**Stretched H₂ stays on the spin-symmetric SCF solution.** The solver is spin-unrestricted LSDA, but
`multiplicity: None` gives n↑ = n↓ and the cold start sets ρ↑ = ρ↓ = guess/2 (`qm/src/scf.rs`,
`qm-gpu/src/scf.rs`), so nothing breaks the spin symmetry: past the Coulson–Fischer point (≈ 3 Bohr)
H₂ keeps a doubly occupied bonding orbital, its midpoint density stays ≈ 2× the promolecule's (a
bond line at 5 Bohr in the viewer) and the separated pair sits +4.6e-2 Ha above two H atoms, so the
stretched-H₂ energies and forces are wrong too. Unrestricted LSDA has a lower broken-symmetry
solution there (q → 1, no bond). **Follow-up (solver, Fable):** for an even-N lowest multiplicity
also try a spin-asymmetric start and keep the lower energy (and say so in the solve's notes). Numbers:
`crates/validate/references.md`, "Tools (M6 Task 7)", Findings.

## Drag force on the wheel (user request 2026-10-14)

"When dragging atoms around with left click on select mode, can you make scroll wheel change how much
force you're applying." Ruling: the wheel during a Select drag scales the drag spring's force cap
geometrically (×1.25 per line), clamped to [0.01, 1.0] Ha/Bohr, default 0.1 (`F_DRAG_MAX`, R12, stays
the default); k and the critical damping are unchanged so the integrator's stability is unchanged; a
cap change mid-drag is booked into W at fixed positions and never aborts a solve; the value persists
for the session and shows next to the cursor. Branch `drag-force-wheel` off main.

## Sampling-pass ETA during the atom references (user report 2026-10-14)

Water at 57³ showed ≈ 70 min. Cause: while the pass is in its isolated-atom reference stage
(O cold, three multiplicities, 13–15 s each) the live ETA multiplies that rate by all ~300 remaining
solves; the geometry solves cost ≈ 0.5–1.5 s, so the real pass is ≈ 8–12 min and the ETA corrects
itself once the references finish. Fix (small, offered to the user): remaining references × the
reference rate + remaining geometry solves × the scene's geometry estimate. Applies to main and to
the M6 branch (Task 9's sizing has the same reference-dominated median for water).

## Model library: automatic per-scene models (user request 2026-10-14)

"Rather than me saving/loading models every time the scene changes, it will create a new model for a
vastly different scene and I can begin sampling, then automatically save that; if I switch the scene
to something that has been previously sampled it can switch back to that model and continue sampling
when told to. Ideally in the model page get some information on the current model like what it's
trained on and how much it's been trained."

Design ruling (to be specced after M6): a model is valid for any scene with the same **provenance
key** — species set, charge, multiplicity, grid (n, h), σ, teacher and teacher tolerance — not the
geometry or atom count (an H₂ scene and a six-H scene share and pool one model; water gets its own;
isotopes are recorded, not keyed — ruling P2). An on-disk library (one `.model` per key plus an index
with a human label, the scenes it was used on, created/last-trained times) replaces the per-session
autosave: entering estimated mode or editing the scene looks the key up — found → load and keep
teaching; not found → a fresh model and the pass offer. Saves are automatic (after refits, debounced,
and on switch/exit); explicit Save/Load stay as export/import. A key change caused by the user (grid,
σ, charge, multiplicity, a new element) says which model it switched to and why, so trained data
never silently disappears. Model tab info: provenance in words (`Provenance::describe_recorded`),
samples by origin (pass / fallback / audit) and quarantined, passes run, teacher time spent, epoch,
last fit rmse E/F, trust state and reason, Q′ and residual dofs, sampled coverage (pair-distance ranges
per species pair, burst temperatures), file path and dates. Opus; after M6 (it uses the providers
crate, `Trainable`, the Model dock tab and Task 5's mass records).

## Settings profiles (user request 2026-10-14)

"Would it be possible to have settings profiles too? To be able to set a default profile, as well as
reset to out of the box settings." Ruling: specced together with the model library (same per-user data
folder, atomic writes, lock): named profiles, a default applied at launch, "Reset to factory settings"
(profiles and models untouched), every user-adjustable setting except the scene's atoms, an appendix
table of every setting and its default as the acceptance list; applying a profile that changes physics
settings pauses like a scene edit and logs what changed; "reopen last scene on launch" is a profile
option, off by default.

## Empty preset (user request 2026-10-14)

"Lets add empty as a preset." Ruling: `empty` = zero nuclei, the default cell; listed first; no solve
until an atom is placed; every layer (driver, solvers, estimated mode, render, readouts) handles zero
electrons without errors or NaNs. Branch `empty-preset` off main.

## Periodic table + isotopes picker (user request 2026-10-14)

"Can we turn this [the H–Ne element strip] into the entire periodic table + isotopes?" Ruling: a
periodic-table picker (all 118 in the standard layout, searchable) with an isotope choice per element
(most abundant by default); Z ≤ 10 enabled, Z > 10 shown greyed "needs pseudopotentials (planned)" and
unlocked by the pseudopotential milestone (the all-electron smeared-nucleus solver cannot resolve their
cores at h = 0.25 — rule 7: no silently wrong physics). The isotope table in `units` is extended to every
element's stable and long-lived isotopes (AME2020 masses; IUPAC/CIAAW isotopic compositions used only to
pick the default isotope — measured data, rule 1). Lands in M6 Task 10 (the chrome wiring of M6's
isotopes and badge), not a separate branch.

## Activity view: obvious processing feedback with something to watch (user request 2026-10-15)

"When the systems are solving equations / training the models, could we make the processing
information a little more obvious so you know it's processing stuff, and maybe a little more action
going on, maybe something to see (even display each step? unless that significantly slows things
down)." Ruling (built right after M6 merges, before the user-data milestone): (1) an animated activity
badge in the status chip naming what runs and its progress ("Solving · SCF iteration 7 · residual 3e-5",
"Sampling 82/273 · bond scan · 1:40 left", "Refitting model · 250 samples", "Bond references: O 2/3",
"Audit running"), a thin progress bar under the toolbar while anything runs, and an **Activity** dock tab
listing every background job (progress, ETA, cancel); (2) something to watch: during a sampling pass the
geometry being solved is drawn as translucent ghost atoms with its density filling in (GPU-to-GPU copy,
no CPU readback); during any DFT solve a small live log-scale SCF-convergence chart; during refits a
"model vs DFT" parity plot of the training samples; (3) display updates rate-limited (~10 Hz), measured
to cost < 1 % of solve time before it ships (rule: timings reported, never asserted). Follow-up (same day): "would be good to see more live details on the training too" — the training
view adds: samples arriving coloured by origin (pass / fallback / audit) with their teacher cost; a live
trust checklist naming the unmet condition (n ≥ Q′, residual dofs ≥ 30, max training residual vs claimed
accuracy, σ_f once Task 11 lands); energy/force rmse per refit epoch; the parity plot; per species-pair
model energy curves vs distance with the ±σ uncertainty band and the DFT samples as dots (thin band =
learned, wide = will fall back); coverage histograms (pair distances, burst temperatures); and an event
timeline of fallbacks, refits, audits and quarantines. All read from the store and the posterior already
held in memory — no extra teacher solves.
Follow-up (2026-10-15, from the Model tab): "Would I be able to get an activity log of new solves being added to the model and the total number of solves for the model automatically update?" — the Activity tab's sample log is a scrolling list, newest first, one row per teacher solve: time, sim step, origin (pass / fallback / audit), rule that triggered it (e.g. 5e grade, lone atom), whether it was kept, quarantined or box-excluded, |ΔE| vs the model where known, and its solve time; above it two live counters that tick per solve (not per refit): samples in the model and total teacher solves ever spent on it (kept + rejected), plus solves this session. The Model tab's "samples" row shows the same live count. Read from the store's existing records; no extra solves.
The job list is the
seed of the precision milestone's job queue and the user-data model manager — one component, not three.

## Quality presets (user request 2026-10-14)

"Let's make the more accurate measurements default, but in scene there should be a slider/presets
from faster but lower accuracy, or slower but more accurate." (The user measured H₂ at the old
default 49³ × 0.25, σ = h: 1.78 Bohr = 0.944 Å against 0.7414 Å.)

Ruling, as built (branch `empty-preset`):
1. **Scene ▸ Quality** (+ Ctrl K): **Fast** h = 0.30, **Balanced** 0.25 (the old default),
   **Accurate** 0.15 (**the new default**), **Precise** 0.10, **Custom** h ∈ [0.08, 0.35] snapped to
   3 decimals (model provenance keys stay canonical). σ = h unless σ was set explicitly. The box's
   extent is kept: n = the smallest odd integer with (n − 1)·h ≥ the box (12 Bohr for the small
   presets, 14 for lih/h2o), measured from the preset's / file's box so repeated changes never creep;
   the resulting n³ and the box in Bohr/Å are shown.
2. Each level shows its cost (≈, rough: the scene's last solve × (n′/n)³ — it overestimates finer levels
   from a coarse solve because the GPU is latency-bound below ~10⁶ points; review I1) and H₂'s bond length at that level,
   measured (`crates/validate/references.md` "Quality presets": 1.002 / 0.947 / 0.847 / 0.807 Å for
   Fast / Balanced / Accurate / Precise; experiment 0.741 Å).
3. A change is a scene edit (`Command::SetGrid`: pauses, re-solves, provider reset; estimated mode's
   provenance path unchanged). Scene files keep their grid; presets (incl. empty) and a fresh launch
   take the panel's quality (Accurate unless changed); `--quality fast|balanced|accurate|precise`;
   `--scene` beats it (there are no other grid flags).
4. Grid limits raised to n ≤ 201, h ≥ 0.08 (Precise is 121³, Custom 0.08 on a 14 Bohr box 177³). The
   validate suites build their own grids (no `app` dependency); the app's viewer tests that solve are
   pinned to the preset files' grid; the worker tests use `Scene::preset` (the file as written, 49³).
5. The `h2` preset moved from R = 1.6 Bohr (which sits at the σ = 0.15 minimum, 1.601, and would not
   vibrate on the new default) to **1.45 Bohr**, this solver's σ → 0 LSDA bond length: it starts at the
   physical geometry and vibrates because the smearing moves the minimum (fix round 1; k(0.15) ≈ 0.31
   Ha/Bohr² → ≈ 4e-3 Ha of mode energy, like the old preset's). Tests that need 1.6 use
   `scene::h2_r16`. `--backend cpu` without `--quality`/`--scene` loads Balanced (an 81³ CPU step is
   ≈ 40 s, uncancellable). A GPU memory guard refuses scenes over the budget (12 GB default,
   `ATOMIC_SANDBOX_VRAM_GB`; review I2).

## Container walls on by default (user request 2026-10-14)

"Lets have container walls enabled by default." Ruling, as built (branch `empty-preset`, b6a7ce9):
the app's panel starts with walls on at the slider's default half-width (1.5 Bohr inside the cell's
face: 4.5 Bohr for the 12 Bohr presets, 5.55 for lih/h2o at Accurate), on every launch, preset and
backend (the walls act in the integrator's tool forces, so estimated mode's model steps are confined
too); a preset or scene switch moves them to the new cell's default (fix round 1). `sim::Dynamics`
keeps walls off by default, so the validate suites stay unconfined. `--walls <Bohr>` sets the
half-width, `--no-walls` starts without them.

## Temperature in °C or °F (user request 2026-10-14)

"Would it be possible to have in the thermal area to have the option of temp in C or F as well as K."
Ruling, as built (branch `empty-preset`): a K / °C / °F toggle beside the thermostat (Physics ▸
Thermostat, styled like the Ha / eV / both toggle; Ctrl K "celsius" / "fahrenheit"), display only —
rule 3: the driver, the thermostat and the sampling plan stay in kelvin, and the conversions live in
`units` (`kelvin_to_celsius`, `kelvin_to_fahrenheit` and back, tested). It applies to the T readout,
the status chip, the sparkline's T range, the thermostat target (typed in the chosen scale, clamped at
absolute zero: 0 K / −273.15 °C / −459.67 °F) and the sampling plan's hot-burst T. Kelvin by default.

## Sampling-plan presets (user request 2026-10-15)

"In the sampling area can we get some presets with descriptions for use cases etc?" (after a pass over three H atoms
ended "max training energy residual 4.1e-3 > 1.0e-3 Ha/atom": the 2000 K burst throws three atoms into close
encounters the linear model cannot fit to 1 mHa/atom). Built in M6 (branch `m6-sampling-presets`): the Model tab's
Sampling plan has a preset row — Quick look, Gentle, Standard (= `SamplingPlan::default()`, unchanged), Hot / reactive —
with each use case shown under it, "Custom" when the fields match none; UI data only (`ui/sampling_presets.rs`), no physics change.

## Precision milestone (user request 2026-10-15)

After a discussion of why LSDA's bond energies are ~30 kcal/mol off and that the estimator could be
trained on a better teacher: "perhaps we go ahead and add that to our plan? a different model to run
that is highly precise using the real equations for almost perfect precision, but cannot be played
live, you can either prep a scene then have it solve each step that you could then play back as a
video whilst it solves each step. Or train a highly precise model. Again we need to have a super easy
to use UI to configure, manage and understand each model on the system."

Controller's commitments (binding): (1) **precision render** — background solve of each MD step,
progressive playback (scrub/pause/resume) while it computes, checkpointed across restarts, usable with
any slow solver (DFT at Precise first, so it ships before the precise method); (2) **precision models**
— the estimator trained on the precise method, mainly by Δ-learning (E_precise − E_LSDA), then played
live; (3) a **model manager** — every model in one place (teacher + accuracy tier, training data,
coverage, cost, comparison, which model answers), long jobs as a queue with progress/ETA, plain-language
explanations. Precise method: real-space QMC from the sandbox's own orbitals (Z-only), essentially exact
methods for 1–3 electrons.

Specced as `docs/superpowers/specs/2026-10-15-precision-design.md`, planned in
`docs/superpowers/plans/2026-10-15-precision.md` (five stages). Decision in one line: explicitly
correlated Gaussians ("exact", ≤ 3 electrons) and single-determinant fixed-node DMC from our own LSDA
orbitals (point-nucleus H/He, our M7 pseudopotentials with T-moves for Z ≥ 3), made live by Δ-learning
on the LSDA surrogate; DMC renders integrate the learned surface refined on the fly (noisy-force
Langevin rejected: grid orbitals have no cheap ∂Ψ/∂R). Gaussian-basis CCSD(T) is rejected under rule 1
(per-element fitted basis sets). Placement: stages 1–2 (render, manager) right after the user-data
milestone; stage 3 (exact tier) independent of M7; stage 4 (DMC) after M7; stage 5 (Δ-models) after
3 and 4. Honest headline expected for H₂O's atomization energy (233.0 kcal/mol non-relativistic BO),
revised after the physics review (2026-10-15): underbound by ~3 ± 1.5 kcal/mol at the end of stage 5
(~60–65 % chance of ≤ 4), against LSDA's tens; this requires the QMC pseudopotential variant (HF
reference, no NLCC, semilocal in DMC) — with M7's potentials as they are it would be unpredictable.

## Bonds that "aren't true" (user report 2026-10-15) — strength display built in M6

Screenshot: three H in a near-line, bonds drawn on both sides, labelled H₃. Causes: (1) LSDA's
delocalisation (self-interaction) error spreads density across stretched contacts and makes H + H₂ far
too attractive, so the density really does show a bond the exact physics would not; (2) the detector
was binary — a stretched, weak contact past its thresholds looked the same as a real bond.

**Built in M6 (branch `m6-bond-strength`): the strength display.** Each bond carries ∇²ρ at its bond
point and the QTAIM kind by its sign (Bader & Essén 1984): shared (covalent), closed-shell (weak /
stretched) or indeterminate (within 3× its estimated grid noise). The line's width and opacity follow
log ρ_b; solid shared, dashed closed-shell, dotted indeterminate; Dock ▸ Selection shows ρ_b, ∇²ρ_b,
the kind in words and "LSDA overbinds stretched contacts (delocalisation error): a weak contact here may
be weaker or absent in reality". Measured on the user's case (linear H₃, H–H 1.45 + H…H R₂, default
grid): the long contact is bonded up to R₂ = 2.55 and shared-shell while clearly bonded (R₂ ≤ 2.25,
ρ_b 0.08–0.10 against the short bond's 0.23 — drawn thinner and fainter), indeterminate (dotted) at
2.5–2.55, and turns closed-shell only after the detector has dropped it; so the sign alone does not
expose the LSDA overbinding, and ρ_b only modestly (it is LSDA's own density, inflated by the same
error: at R₂ = 2.0 the contact is drawn ≈ 16 % narrower and ≈ 15 % fainter than the short bond)
(`crates/validate/references.md`, "Bond strength"). The molecule grouping and names are unchanged (a
weak contact still joins a component). Known follow-up, not done in M6: a refined bond point (the
parabolic vertex of q or ρ̂ through the chord samples k − 1, k, k + 1, L evaluated there) to cut the
location noise that makes water's O–H kind sit at its band (intrinsic to hydrogen's density slope,
not the smearing — M7 will not change it). Still open: exact-teacher densities (precision milestone)
for few-electron checks.

## Per-atom charge in the atom editor (user request 2026-10-15)

"I think we should be able to change the charge of each atom in the atom editor with the different
isotopes." Design ruling (queued until after the post-M6 pause): (1) a per-atom **intended charge**
row in the Selection tab beside the isotope choice; the scene's total charge becomes the sum of the
atoms' intended charges (one source of truth: the scene charge is derived, the scene file stores per-atom
`q` only when nonzero, old files keep their scene-level `charge` line and read as all-neutral atoms plus a
residual); where the electrons actually go remains the self-consistent solution — the UI never claims an
atom *holds* its intended charge; (2) after every DFT solve, show each atom's **computed charge** (Hirshfeld
stockholder partition of ρ against the solver's own promolecule — the same machinery the bond detector
already uses; rule 1 clean) next to the intended one, so LSDA's delocalisation error is visible
("asked O⁻, solver put −0.72 e"); (3) optional later: **constrained DFT** (Wu & Van Voorhis, Phys. Rev. A
72, 024502 (2005)) to enforce electron counts on atoms/fragments (e.g. separated Na⁺ + Cl⁻), a solver
milestone of its own. Honest caveats in the tooltip: negative ions in LSDA may be unbound (self-interaction),
the box can fake binding; charge under the surrogate keys the model (provenance includes total charge).

## Bond angles on the drawn bonds (user request 2026-10-15)

"With M7, you should display the bond angles on the bond chains that are drawn between atoms." Added to the
M7 spec as Task 13 (Opus, display only): an arc + degree label at each atom joining two drawn bonds, from
nuclear positions alone; weak contacts' angles in their faint/dashed style; decluttered to the selected atom
where many bonds meet; a View toggle; listed in the Selection tab. It is how M7's geometry gates (water
104.5°, NH₃ 106.7°, CH₄ 109.47°) become visible in the app.

## Three hydrogens will not reach trust (user report 2026-10-15; diagnosed)

The user's run: three H, Accurate grid (81³ × 0.15, a 12 Bohr cell), "Gentle" plan: refused at max training
residual 2.2e-3 Ha/atom. Diagnosis (headless reproduction, report kept at `.superpowers/parked/diag-3h-report.md`):
(1) **H₂ + a far H: the box.** 130 of 145 samples box-affected (box energy 1.6 mHa at 2 Bohr off-centre, net
label force up to 0.23 Ha/Bohr); the same scene in an 18 Bohr cell (121³ × 0.15) is trusted at 7.6e-5 Ha/atom.
Fixed in M6's final fix wave: the refusal names the box and suggests the cell to use. Queued: size the cell to the
scene automatically (≈ 6 Bohr from any atom to every face) — a design change, because models are keyed on the grid.
(2) **Triangles / three separated H: the teacher is multi-valued.** LSDA with 2↑1↓ has several converged spin
solutions (which atom carries the ↓, which pair is singlet-coupled), up to 27 mHa apart, and the lowest one switches
branch near the equilateral geometry (a cone-shaped kink). No box or basis fixes it; refusing is correct. Queued
(physics, needs review): a lowest-of-several-spin-starts teacher so labels are single-valued (costs several solves per
label), and/or a per-sample spin-pattern record so the model can be told which branch a label is on.

## Show what training is solving (user request 2026-10-18)

While a sampling pass runs, the 3D view should replace the user's scene with the geometry the teacher is solving at
that moment — the reference atom, a random nudge, a scan point, a hot-burst step — drawn as a wireframe ghost with a
label (`reference: isolated O`, `O–H · 1.50 × r₀ · 2.71 Bohr`), flashing green / red / grey as each sample is kept /
fails / is skipped, and give the scene back when the pass ends, is cancelled or fails. **Built** on `training-ghost`
(checklist row 208, README "Seeing what training solves"): `sim::PassGhost` (positions + Z, stage, job, base
positions, a burst's trail, the last finished job's serial / outcome / geometry) comes from the pass's own queue
(`providers::sampling::Sampler::ghost` → `Trainable::pass_ghost` → `DynSnapshot::ghost`, refreshed wherever the
progress is; no density, no GPU readback), `app::ghost` draws it with `render::lines`. Display only; View ▸ Display aids
▸ show what training is solving (default on), `--ui ghost=on|off`. Deviation from the request: the thin links join
atoms closer than 2.2 × the sum of the display radii, not 1.3 × — the display radii are legibility choices (H 0.35
Bohr), and H₂'s own bond (1.45 Bohr) is 2.07 × their sum, so 1.3 × would draw no bond on the molecule it is meant for.
The scanned pair always has its link. A cell-guard skip row is accounted for without a solve and never has a ghost of
its own (its flash is the base geometry in grey, and is overwritten by the next solve's).

## Robustness bugs seen 2026-10-17 (fixed on `robustness-fixes`, 2026-10-17)

All three are fixed (commits `fix(gpu)`, `fix(orbitals)`, `fix(app)`): readbacks return `gpu::ReadbackError`
(the solver latches it and ends the solve with `QmError::Device`; `GpuKs` retries twice, then the worker shows "GPU
busy …"), the spin view draws nothing inside `SpinSummary::noise_l1`, and waits fail on a stall (`app::progress`), not a
wall clock. The bare `Device::poll(...).expect("poll")` calls the review listed (`qm-gpu` `batch`, `reduce`, `poisson`,
`stencil`, `render::volume`) are all inside `#[cfg(test)]` timing benches, not the SCF path: there is no bare poll
left in production code (the production waits are `gpu::read_buffer`'s, which return `ReadbackError`); the benches now
use `gpu::wait_idle`. The original report:

- **GPU readback panics under device contention.** The user's window (main b40c12d) ended with
  `qm-gpu/src/grid.rs:131 map callback: RecvError` after repeated "Failed to wait for GPU to come idle before
  reconfiguring the Surface", while M7 GPU test suites ran on the same device. Every `rx.recv().expect("map callback")`
  (gpu/volume.rs, qm-gpu/grid.rs, render/dots.rs, render/headless.rs) should return an error the worker's provider-failure
  path already handles (retry / show "GPU busy") instead of panicking.
- **Spin-density view amplifies noise for closed shells** (H₂, LiH): see the orbital-viewer ledger; suppress below a
  grid-aware noise floor and say "no net spin — closed shell".
- **Load-sensitive wall-clock timeouts in tests**: worker `pause_and_cancel_abort_a_sampling_solve_promptly` ("no
  snapshot within 120s" under full-suite load; passes alone in 97 s) and the replay `wait idle` 600 s.

## Atom trails (user request 2026-10-17)

"another view option, like a persistents display? so I can see a bit of a ghost path behind each atom, track where its
been, I should be able to set how long its stays around for (decay) and how transparent it is" / "a single line would be
good, not the entire atom size". Built as View ▸ Display aids ▸ Trails (`crates/app/src/trails.rs`, checklist row 209):
display only, off by default; a single fading line per atom in its element colour; **length** in simulated time (5 fs –
10 ps, logarithmic, default 100 fs — simulated, not wall time, so it covers the same motion at any play speed) and
**opacity** (0–100 %, default 70 %, fading linearly to nothing at age = length). The viewer keeps the history from the
snapshots it receives, only while the toggle is on.

## Sampling: failing samples drag on; sample vs iteration names; per-iteration detail; per-atom model storage (user request 2026-10-17)

Watching the H₂O sampling pass in the M7 preview (CoresOnly, 95³ × 0.15): "the only steps that seem to pass are the ones passing before around step 20 - 50 … everything over that ends up failing at step 400". The failures are the outermost pair-scan points (1.8 r₀; O–H ≈ 3.3 Bohr), a half-broken bond whose near-degenerate occupations flip between iterations; the SCF only gives up at its iteration cap.
- **A (queued, M7 branch after Task 9's fix round):** (1) a stall stop — end a solve whose best Δρ has not improved by ≥ 2× over the last ~50 iterations (measure the slowest converging solves first so it never ends a slow good one; converged results unchanged); (2) after a scan point fails, skip the further-out points of that scan with a note. (3) Converging stretched geometries (smeared/fractional occupations for broken bonds) → the precision milestone.
- **B (queued with A):** "sample" = one full DFT calculation at one geometry (one training example); "iteration" = one SCF refinement round inside it. Activity reads "Sample 42 of 158 · iteration 37 / 400".
- **C (queued with A):** the running sample shows Δρ vs target, ΔE per iteration, time per iteration, a Δρ sparkline and a "stalled" tag; finished samples keep their iteration count and end reason (converged / stalled / cap).
- **D (M8 + user-data milestone):** per-atom (per-element, per-element-pair) model parts stored in a named **collection** folder (element parts, the pairs/triples learned, the samples), extended in place when a new element joins; a collection only serves scenes with the same solver provenance (grid, pseudopotential recipe, tolerance).

## Orbitals and density in estimated mode (user request 2026-10-17)

"Will it ever be possible to display the different orbital fields and all that in estimated mode?" → "yeah it be good to have at a later milestone". The surrogate maps positions to energy/forces only; it holds no electrons. Agreed staging for a later milestone:
1. **On-demand DFT snapshot** — pause → "show orbitals" runs one DFT solve at the current geometry and feeds the existing density/orbital views (periodic refresh while playing; for a scene larger than the grid, a cut-out patch around the selection). Plumbing over the existing solver and orbital viewer.
2. **Learned electron density** — a second local model predicting ρ from positions, trained on the densities every teacher solve already produces (research item; design after M8).
3. **Learned orbitals** — a learned atom-centred Hamiltonian diagonalised per frame (orbitals are not unique, so predict H, not orbitals); orbital views on large estimated scenes (research item; design after M8).

## Photons and light–matter interaction (user request 2026-10-18) — after M10

"at some stage would it be possible to simulate things such as photons and their interactions with atoms?" → "yeah note it for after M10". Staging (semiclassical light: a classical field, quantum electrons; no per-element parameters):
1. **IR spectra** from the BOMD dipole autocorrelation (the dipole is already available from the density) — the absorption peaks emerge from the dynamics. Small; may come earlier as a small item.
2. **Real-time TDDFT** — propagate the Kohn–Sham orbitals under an oscillating field on the existing GPU grid (time steps ~1000× shorter than BOMD; small molecules over femtoseconds): density sloshing in the view, UV/visible absorption spectra, strong-field ionisation (needs absorbing boundaries), photodissociation with Ehrenfest nuclei. The "excited states and light" milestone after M10.
3. **Quantised light** (spontaneous emission, cavity QED / QEDFT) — research-grade, much later.

## Fusion and fission in the nuclear-events milestone (user request 2026-10-18)

"Is Fusion and Fission planned in this project?" → "yeah note it for the nuclear milestone". Additions to the nuclear-events layer (spec §11.x: point nuclei, stochastic events from measured data, a separate nuclear clock; the same documented rule-1 extension to measured nuclear data):
1. **Spontaneous fission** — another decay mode with measured branching ratios (NUBASE); fragments' Z/N from measured yields, Q from the isotope mass table. In scope with α/β/γ.
2. **Neutron-induced fission** (stretch) — free neutrons as a new particle type, measured cross-sections (ENDF/B), chain reactions as events.
3. **Fusion plasma mode** (stretch) — at keV–MeV temperatures the electrons are fully stripped, so the DFT chemistry does not apply; a separate hot-plasma mode where bare nuclei collide with measured fusion cross-sections (e.g. p–p, D–T, the CNO steps) and release Q = Δm c² from the mass table — a "star-core box". Not first-principles nuclear physics; the energy accounting is exact.

## Show what training is solving (user request 2026-10-18) — queued after the surrogate trust fix

"can we show something to look at during training? … some super simplified low cost graphic to see what exactly its simulating? wireframe, etc?" → "I dont think we need to see the real scene whilst training". Design agreed in chat:
- During a sampling pass the 3D view **replaces** the scene with the geometry being solved (the real scene is paused and hidden; the cell outline/walls stay for reference): wireframe/outlined spheres in element colours, thin lines between atoms. The scene returns when the pass ends or is cancelled.
- Per stage: references — one lone atom, "reference: isolated O"; perturbations — the jittered molecule with small arrows from the unperturbed positions; bond scans — the scanned pair highlighted with a live "O–H · 1.50 × r₀ · 2.71 Bohr" label; hot burst — the moving geometry with a short trail.
- Each finished sample flashes green (kept) / red (failed, stalled) / grey (skipped), matching the solve log.
- View ▸ Display aids toggle "show what training is solving" (default on during a pass); display-only; the worker ships the job's positions (a few dozen numbers per sample), no density readback. A low-res density thumbnail is a possible later add-on.

## Custom window frame (user request 2026-10-18) — done 2026-10-19 (`crates/app/src/frame.rs`, user hand-tested: "works great")

"Would you be able to create a custom window for the app? Like windows window?" → chose **option B**, "leave it till after M7 is complete". A borderless window with the app's own title bar in the theme: the menu bar (File / Scene / View / Physics / Help, scene chip, Ctrl-K search) moves into the title bar; custom minimise / maximise / close buttons; drag by the empty bar (double-click maximises); edge/corner resize; Windows 11 rounded corners. Keep native behaviour: Snap Layouts on maximise-button hover (WM_NCHITTEST → HTMAXBUTTON), Win+arrow snapping, Aero Snap by drag, correct maximised insets, multi-monitor DPI. Display/plumbing only; replay/screenshot harness unaffected (headless). Option A (dark native caption via DWM attributes) is the fallback if B's native behaviours can't all be kept.

## Load a model at other settings (user request 2026-10-18)

"When loading models, it should give an option to load the required scene settings to run it, instead of only refusing to load it." Built as a card (UI checklist row 211): picking a `.model` reads its provenance header first; if grid, σ, charge, multiplicity, teacher, teacher tolerance or ionic surface differ from the scene's, a modal card lists each as "model: X · scene: Y" and offers **Apply the model's settings and load** / **Cancel**. Apply changes the scene's grid, σ, charge, multiplicity, ionic policy and (single-isotope) isotopes through the ordinary commands and then loads the model; the nuclei stay where they are. What the app does not let you set — the teacher tolerance (1e-6, fixed), the teacher (the GPU solver on a machine that has one), another build's pseudopotential recipe, a grid over the GPU memory budget, a charge or multiplicity the electrons cannot realise — is named with the reason and disables Apply. The recorded isotopes are shown but never gate the card (they are not compared; an H₂ model serves D₂).

## Distributed training and a headless Linux server (user idea 2026-10-18) — future planning, not scheduled

User: "be able to have the program run on multiple computers, one of them 'host' a session, then other clients could connect to distribute training load … a linux version that could be a headless server that would access the systems gpu to run distributed training … settings to be able to choose who does the work … 2 normal clients 1 being host could be 50/50 or all on the client and not host … with the server … put all load onto the server (chosen from the client) or again 50/50". Captured only ("no need to think about it"):
- Host/client sessions: one app hosts a session (scene, model, sample store); other apps join and take a share of the teacher solves (sampling passes, fallbacks, audits).
- A headless Linux server build (no window; the GPU solver over wgpu/Vulkan) that joins sessions as a worker.
- A work-share setting chosen by the user: e.g. host 50 / client 50, all on the client, all on the server — per participant.
- Notes for when it is designed: provenance must match across machines (same build/recipe, grid, tolerance; GPU vendor bit-differences stay within tolerance), samples are the shared state (the fit stays on the host), and the transport/security model is a design question (spec first).

## Model sharing hub on a self-hosted server (user idea 2026-10-19) — future planning, not scheduled

User: "Model sharing, I will be able to host a server on my linux dedi that could store a lot of trained models … a nice UI where people could easily browse already trained models, search from precision, and how much it has actually been trained etc, and they can all have some scoring system on each part of the model … some secure way of actually sharing things without allowing skewed data, whether intentional or not." Captured only:
- A server (the user's Linux dedicated host) storing many trained models/collections; an in-app browser (and possibly web) to search and download them.
- Search/filter by precision tier (grid h, tolerance, pseudopotential recipe — the provenance), training extent (samples, passes, elements/pairs covered, trust status), and a score per part of the model (per element/pair/class — fits M8's per-environment parts).
- Integrity against skewed data, accidental or deliberate: uploads verified rather than trusted — e.g. the server re-solves a random audit subset of each upload's samples with the reference solver before accepting (same provenance/build), signed provenance and hashes, per-sample origin kept, quarantine of contributions that fail audits, reputation per uploader. Design question for its spec (pairs with the distributed-training idea and the user-data milestone's model library).

## Placement assist and rewind-to-Play snapshot (user request 2026-10-19) — after M7, "sooner rather than later"

User: "some quick calculation assist with creating your own molecules, when I goto place molecules next to each other when things are paused, it should somewhat guide me on where it can be placed … otherwise when hitting play things can be extremely slow and take a long time to settle … (optional) … a check box … it could also prevent the user from placing the atoms in places that will break the simulation." / "make the rewind button take you to the last best 'snapshot'. Snapshot can be taken as you press play, if the simulation breaks, hit rewind to go back to when you hit play."
- **Placement assist (opt-in checkbox on the Place tool).** Guidance must come from the physics, never a per-element radius table (CLAUDE.md rule 1): while placing (paused), preview where the new atom would settle — a short constrained relax of the placed atom alone (others fixed) on the trusted estimated model when one covers the scene, else a coarse-grid DFT probe — and draw the suggested spot / an energy hint at the cursor; optional "snap to the suggestion". Refuse placements that break the simulation, with the reason: coincident/overlapping cores (the frozen-core overlap the M7 core-overlap display already detects), inside the wall or box-rule zone, an h too coarse for the species (h_req), a species the recipe refuses.
- **Rewind to the Play snapshot.** Pressing Play stores the scene + velocities (+ model epoch); Rewind restores that snapshot ("back to when you pressed Play"), separate from the existing reset; keep the last few with timestamps in the Activity timeline.

## 2D molecule builder (user request 2026-10-19) — next, after the weekly reset

"A 2D blueprint molecule builder which will allow you to quickly put together molecules, and it should assist you too using known lookup tables" — under File, in a pop-up window like Open/Save. Proposed: a canvas (element palette or the periodic table; click to place, drag atom to atom for a bond, click a bond to cycle single/double/triple), assistance (fill missing hydrogens to the usual valence, flag over/under-bonded atoms, common groups OH / NH₂ / CH₃ / rings, live formula and name via `names`), 2D → 3D (bond lengths from covalent radii, angles from VSEPR shapes), centring and a cell-fit check, Insert into the scene or Replace it, then an optional Relax so the DFT corrects the guess. **Design rule 1:** the radii and valence tables are per-element data beyond Z and mass — allowed here only as a starting-geometry aid that never enters forces or energies and is relaxed by the solver (needs the user's sign-off in the spec). Supersedes the queued "placement assist". Needs a short spec first (canvas interactions, bond orders and charges → the 3D scene, radicals and ions).

## Also queued (2026-10-19, after the reset)

- Viewer capture: Screenshot and Record buttons at the top right of the 3D view → a `captures/` folder; quick settings hide-UI and resolution (offscreen render); Record as a PNG sequence unless an encoder is added. The atom card (top right) moves down by the buttons' height.
- Camera view: a smooth-union solid surface (smooth-min of the atoms' distance fields, ray-marched) with a smoothness slider — the soft neck between bonded atoms instead of a crease, and no rim artefacts.
- ~~Raise the speed cap (`sim::dynamics::MAX_SPEED` = 1000 steps per tick) to ~10 000: the 16 ms tick budget already stops a tick early.~~ Done 2026-10-20 (branch `small-items-speed-eigensolve-vram`; `2026-10-20-small-items-how-to-test.md`).

- ~~Activity tab: show when the orbitals are being computed (the display eigensolve after a DFT step) as a job line with its time in the solve log; display only (user request 2026-10-19).~~ Done 2026-10-20 (same branch). Also done there: the GPU memory budget derived from the card (half its device-local memory; `ATOMIC_SANDBOX_VRAM_GB` overrides, 12 GB fallback), shown in the refusal.
- Dots mode, "glowing cloud" look (user reference: a Quantum-Atom simulator screenshot): additive blending (dense regions burn to white), many more soft sprite points with varied size/brightness, a bloom pass (downsample, blur, add back; shareable with the Camera view), a hot colour ramp with a filmic curve, optional decorative background stars; View sliders for glow, point count and size. Display only: the points still sample the real density/orbitals.
  Reference to read first: https://nazat02.github.io/Quantum-Atom/simulator_dark.html (its GitHub project: the point sampling, blending, glow and colour ramp); check the licence before reusing code (ideas only if not permissive).
  The "floating" motion the user likes: persistent points that wander rather than a fresh sample per frame (our shimmer flickers) — a Metropolis random walk on the density (each point steps randomly, accepted with min(1, rho_new/rho_old)): the cloud stays a correct sample of the density/orbital at every frame while each point drifts smoothly; a drift-speed slider. On the GPU: one state buffer of points, one compute step per frame.

## Boundary-surface orbitals (user request 2026-10-19) — next session, start here

User: make the dots view read like a textbook orbital picture (reference: separate coloured lobes, hollow inside). Plan agreed in chat:
1. Per orbital, the **90 % enclosure level** c_k: the density at which the grid cells with |ψ_k|² > c_k hold 90 % of the orbital's probability (sort/histogram once per solve, GPU or CPU).
2. Dots only inside each orbital's boundary, in that orbital's colour (filled); a **shell** option keeps only a thin band just inside the boundary (the hollow look).
3. Controls: enclosed share 50–99 % (default 90), **shell / filled — shell is the default** (user, 2026-10-19; filled stays an option), **hide core orbitals** (valence only, e.g. Al's 3s/3p).
4. The catch: in "All orbitals coloured" the GPU holds only the orbitals' mix (colour volume: rgb = Σ colour_k w_k, a = Σ w_k) — per-orbital fields (or per-orbital sampling) are needed for step 1; that is the bulk of the work (orbital display code: `qm-gpu` orbital display products, `render::dots`, `app::gfx` dots, `ui::dock::orbitals`). No solver change.
Shipped already on main (2026-10-19): floating dots (underdamped Langevin on −ln ρ, `render/src/shaders/dots_compute.wgsl`), glow (additive soft world-size sprites), density/drift/feather/size sliders beside the view switch, per-orbital feather (`fade_at`). Still queued: bloom pass, molecule builder, capture buttons, Camera smooth-union surface, Activity orbital-solve line, MAX_SPEED raise.
Implementation design (read in the code 2026-10-19):
- The coloured view's field is built on the GPU by `qm-gpu/src/shaders/colour_accumulate.wgsl` (`ColourParams.w[i]` = (f c / P, f / P) per orbital; ≤ 6 orbitals per chunk), called from `OrbitalView::update_with`, `Product::Colour { include, physical, out }` (qm-gpu/src/orbitals.rs ≈ l. 609); the per-orbital fields are `self.prev[ch][r.index]` (qm-gpu `Field`), peaks `out.scale[ch][r.index]`.
- 90 % level: a new histogram kernel per orbital — bins of log10(ψ²/P) (e.g. 256 over 8 decades), each point adds its probability ψ²·h³ as u32 fixed point (×2^30 fits: Σ = 1) with atomicAdd; read back 256 u32 per orbital (one small readback), CPU finds the bin where the cumulative from the top reaches the share → c_k = P·10^edge.
- Mask in `colour_accumulate`: add per-term thresholds (lo, hi) and a mode to `ColourParams`; filled: m = smoothstep(0.8 lo, lo, ψ²); shell: m = that × (1 − smoothstep(hi, 1.25 hi, ψ²)) with hi ≈ 3 lo (a band just inside the boundary); v += (c·occ, occ) × m. The dots then sample a = Σ masks (uniform inside each orbital's shell/region), colour = rgb / a.
- Plumbing: `sim::OrbitalRequest::AllColoured { include, physical }` gains `boundary: Option<Boundary { share: u8, shell: bool }>` (15 uses: sim dynamics.rs 2933 pattern, providers ks.rs 891/2113 + orbitals.rs 451/493/502 + tests 1009–1017, app ui/mod.rs 1056, gfx.rs 129/1061, overlays.rs 564, screenshot.rs 653); `qm-gpu Product::Colour` gains it; the CPU path (`qm::orbitals::colour_volume`) may ignore it at first (GPU default). UI: View ▸ Density / Orbitals tab — boundary on/off, share (50–99, default 90), shell (default) / filled, hide core orbitals (exclude tracks below the valence shell via `include`).

## Edit tool: XYZ move handles and group moves (user request 2026-10-20) — next session

User: "a new edit tool … select a single atom like the select tool then give you a XYZ move mode which will move it without changing its velocity (the simulation will be paused whilst its moving — holding left click on the axis, release resumes). … highlight-select a group of atoms, then move them all in the exact same way."
Design agreed:
- Toolbar: Inspect · Place · Grab · **Edit** (`ui::ToolMode::Edit`, its own icon; Ctrl-K entry).
- Selection: click = one atom; Shift-click adds/removes; drag a box on empty space = select the group (screen-space rectangle over projected nuclei); Esc clears. Highlight every selected sphere (`SphereInstance.selected`); the Selection tab lists the group.
- Gizmo at the selection's centroid: three axis arrows (X red, Y green, Z blue) and three plane squares (XY/YZ/XZ), drawn via `render::lines` + screen-space hit test; hover highlights.
- Drag a handle: rigid translation of the whole selection along the axis/plane (`tools::drag_point` on the axis line / plane). If playing: Pause on press, resume Play on release. A label shows the offset (Å). One `Action` per release (recordable, undoable later).
- **sim:** new `DynCommand::TranslateNuclei { indices: Vec<usize>, delta: [f64; 3] }` — paused only, keeps velocities (unlike `MoveNucleusDiscontinuous`, which zeroes them), clamps into the cell/walls, respects pins, then `after_edit(true)` (forces stale → re-evaluated; E_conserved history restarts). Worker/viewer mirror via `Command::apply`. Dynamics changes → run `cargo test -p validate --release --test dynamics` + `-p sim` before merging.

## Queued 2026-10-20 (after the Edit tool merge)

- **Reference comparison (PySCF, out of process)**: an optional helper process (Python + PySCF, only when installed) answers the current geometry at LDA / PBE / B3LYP / CCSD(T); a Compare panel shows ours vs theirs on comparable quantities only (forces, dipole, relaxed bond lengths/angles, frequencies, binding energies — never raw total energies under pseudopotentials). Spec rule "reference codes as test oracles only, never linked" holds: nothing links, nothing drives Play. A "reference" Play backend is a rule change: only if the user asks.
- **Idea only, NOT queued (user 2026-10-20: "lets not do this yet as a task") — PBE as a second functional (option, LSDA stays default)**: Physics ▸ Functional LSDA | PBE (a scene replacement like Ionic potential); no per-element parameters (rule 1 holds); CPU f64 reference then WGSL; low-density thresholds; pseudopotentials generated per functional (cache keyed by it, gates re-run); `Provenance` carries the functional (LSDA models stay valid under LSDA); oracles against published PBE numbers; suites under both. Expected: bond-strength error roughly halved, per-solve cost +5–15 % (unmeasured), estimated-mode run speed unchanged. Physics milestone: spec → plan → Fable implementation + physics review.
- **Ion buttons + measured partial charges**: Selection tab ±1 e⁻ changes the system's charge (with the note that the electron goes where the physics puts it); per-atom charge by Hirshfeld partition with the bond detector's free-atom densities (display only).
- **Profiling pass**: time each kernel of an aspirin SCF iteration; fix the largest (expected: the dense subspace work).
