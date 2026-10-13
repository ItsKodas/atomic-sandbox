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
