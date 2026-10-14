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
3 and 4. Honest headline expected for H₂O's atomization energy (233.0 kcal/mol non-relativistic BO):
within ~2–4 kcal/mol at single-determinant DMC, against LSDA's tens.
