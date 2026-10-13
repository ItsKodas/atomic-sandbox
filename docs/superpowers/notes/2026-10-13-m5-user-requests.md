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
