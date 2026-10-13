# Atomic Sandbox

An interactive 3D sandbox that simulates matter from the atomic level up with real physics, written from scratch in Rust + wgpu. Nuclei are classical point particles; electrons are solved with real-space Kohn-Sham density-functional theory on a GPU grid. Nothing chemical is hardcoded — every element is described by its atomic number and isotope mass only, and bonds, shapes and energies have to *emerge* from the equations.

![Water (57³ grid) with the electron density cloud and the display aids](docs/images/viewer_h2o_display_aids.png)

## What it does today

| Milestone | Status | What you get |
|---|---|---|
| M1 — CPU reference solver (`qm`) | done | f64 spin-polarised LSDA Kohn-Sham solver on a uniform real-space grid, validated against NIST/literature atoms and the H₂ bond length |
| First pictures | done | H₂ density slice and energy curve (`docs/images/h2_*.png`) |
| M2 — 3D viewer (`app`, `render`) | done | place / select / drag nuclei, orbit camera, volume or isosurface density, energy readouts, scene save/load |
| M3 — GPU solver (`qm-gpu`) | done | the same solver in f32 WGSL compute, 30–100× faster, agrees with the CPU reference to ≤ 1e-6 Ha; the viewer re-solves ~0.1–0.2 s after you move an atom |
| M4 — Dynamics (`sim`) | **in progress** | Born-Oppenheimer molecular dynamics: play / pause / step / rewind, temperature from nuclear motion, CSVR thermostat, energy-drift health readout |
| M5 — Tools | planned | bond graph read off the density, molecule formulas and names, isotopes, input-replay harness, more presets |

Right now the sandbox is a **static quantum-chemistry playground**: you move nuclei by hand and watch the electrons respond, and the energy tells you whether a geometry is bound. Atoms start moving on their own when M4 lands.

Measured on an AMD RX 7900 XTX: H₂ on a 49³ grid 0.2 s cold / 0.1 s warm; H₂O on 97³ 3.6 s. Energies vs the f64 CPU reference at fixed smearing: H 1.5e-7, He 1.0e-6, H₂ 5.9e-7, LiH 3.1e-7, H₂O 2.4e-7 Ha.

## Running it

Requirements: Rust (stable, 1.99 or newer), a Vulkan/DX12/Metal-capable GPU for the GPU backend (the CPU backend works anywhere, slowly).

```bash
cargo run -p app --release
```

Options: `-- --preset h2o` (presets: `h`, `h2`, `he`, `lih`, `h2o`), `-- --backend cpu`. Always build `--release`; debug builds are far too slow for the solver.

### Controls

| Action | Input |
|---|---|
| Orbit | middle-drag, or left-drag on empty space |
| Pan | right-drag |
| Zoom | scroll wheel |
| Place an atom | tool **Place**, pick an element (H–Ne), left-click; it lands on the camera's focus plane (shown as the blue placement grid) |
| Move an atom | tool **Select**, click a nucleus, drag; faint lines show its distance to nearby atoms in Bohr and Å |
| Remove an atom | select it, press **Delete** |
| Solver | CPU / GPU radio; charge, spin multiplicity, nuclear smearing σ, grid size and spacing |
| View | volume or isosurface density, gamma / opacity / iso sliders; cell box, floor grid, axes, placement plane, 1 Bohr or 1 Å spacing |

Things to try: stretch the H₂ bond and watch the energy curve and the density neck split; compare the symmetric H₂ cloud with lopsided LiH (charge transfer with nothing told about electronegativity); build water from an O and two H and see that bent beats linear; push two He atoms together and watch them refuse to bond; make HeH⁺ (charge +1) and watch them bond.

## Physics, honestly

- **Model:** Born-Oppenheimer separation; electrons in spin-polarised Kohn-Sham DFT with the LSDA exchange-correlation functional (Dirac exchange + Perdew–Wang 92 correlation); 8th-order finite-difference Laplacian; nuclei represented as narrow Gaussians (σ = grid spacing by default) with an analytic potential; Hartree potential from a multipole-bounded Poisson solve; Hellmann-Feynman forces.
- **Units:** atomic units everywhere inside the solver (Hartree, Bohr, mₑ, ħ = e = 1); the UI converts.
- **Precision:** f32 on the GPU grid with two-level compensated reductions; f64 on the CPU for nuclei, small dense algebra and the reference solver.
- **Known gaps, documented rather than hidden:** LSDA underbinds the hydrogen atom (−0.4777 Ha here vs −0.5 exact; the LSDA reference value is −0.4787); absolute energies of anything containing oxygen or heavier are not physical at the default 0.25 Bohr spacing because the nucleus is heavily smeared — the density *shapes* are right, the numbers are not; the f32 GPU path carries ~1e-6 Ha of scatter between solve paths; the lithium atom oracle misses its reference by 0.33 Ha at the tested grid and is kept as an ignored test. See `crates/validate/references.md` for every reference number and tolerance.
- **No per-element parameters.** Allowed inputs are Z and the isotope mass (plus display colours and radii, which are not physics). No force fields, no fitted pair potentials, no chemistry libraries.

## Repository layout

```
crates/units      unit conversions and isotope masses (the only per-element data)
crates/qm         CPU f64 reference Kohn-Sham solver: grid, Laplacian, Poisson, LOBPCG, LSDA, SCF, forces
crates/qm-gpu     the same solver in WGSL compute (wgpu), bit-deterministic, cancellable
crates/gpu        wgpu plumbing: context, surface, 3D volume textures
crates/render     camera, impostor spheres, volume raymarch, line overlays
crates/app        the viewer: winit + egui, solver worker thread, tools, scene files
crates/validate   physics oracles and GPU-vs-CPU suites with cited references
docs/superpowers  design spec, per-milestone implementation plans, handoff notes
```

Tests: `cargo test -p units -p qm --release` (~75 s), `cargo test -p qm-gpu -p render -p app --release`, `cargo test -p validate --release --test gpu_vs_cpu` (~3.5 min, needs a GPU), `cargo test -p validate --release` (the full oracle suite, ~30 min). Validation gates are never loosened to make a test pass.

## Design and process

The design is in [`docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md`](docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md); each milestone has a plan in `docs/superpowers/plans/` and a handoff note in `docs/superpowers/notes/`. The code is developed with Claude Code (Fable 5.1 for physics and numerics, Opus 5.5 for engineering), task by task with independent review; every decision made along the way is recorded in the plans and notes.

## Roadmap after M5

Scaling up the "ladder": more elements (pseudopotentials for Z > 10), larger systems through ML potentials trained on this solver's own data and loaded as data behind the same `ForceProvider` boundary, then coarse-grained and continuum stages. Gravity exists as an external-field hook but is invisible at atomic scale, as it should be.
