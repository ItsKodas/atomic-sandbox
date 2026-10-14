# Atomic Sandbox — Design Spec

**Date:** 2026-10-12
**Status:** Approved in conversation; awaiting written review
**Author:** Claude (Fable 5.1) with Dakoda

## 1. Purpose

An interactive 3D sandbox that simulates matter from the atomic level up using real physics. The user places nuclei in a 3D scene, electrons are solved quantum-mechanically, forces from the electrons drive the nuclei, and chemistry, bonding, temperature, and the behaviour of every element **emerge** from the equations rather than from per-element rules.

This is primarily the author's own research/exploration tool. v1 has no installer, onboarding, or polish requirements. Simulation accuracy and efficiency are the priorities; graphics are not.

### What success looks like (v1)

Place two hydrogen nuclei, drag them together, and watch the electron density form a bond with the correct bond length and vibration frequency, at interactive frame rates, with total energy conserved. Then the same for helium, LiH, and water — none of which have any element-specific code.

## 2. Physics reality check (why the design is shaped this way)

- **Electrons are wavefunctions, not points.** A classical point-electron model is wrong, not approximate: it radiates and collapses in ~10⁻¹¹ s. Atoms, bonds, and the periodic table exist only because of quantum mechanics plus Pauli exclusion. So electrons must be solved quantum-mechanically.
- **The periodic table emerges from Z alone.** Solving the electronic Schrödinger equation (approximated via Kohn-Sham DFT) for a nucleus of charge Z and Z electrons reproduces every element's chemistry. No per-element behaviour is hardcoded.
- **Nuclei are point particles.** Simulating protons and neutrons from quarks (lattice QCD) costs a supercomputer per proton and does not affect chemistry or temperature. Every chemistry code treats nuclei as points with charge, mass, and spin. We do the same. Nuclear physics (fusion, decay) is out of scope; it would be a separate module.
- **Temperature is nuclear motion.** Born-Oppenheimer molecular dynamics — nuclei move classically on forces from the quantum electrons — gives emergent temperature, vibration, and reactions.
- **Scale is a ladder, not one algorithm.** Quantum electrons (dozens of atoms) → classical atomistic with quantum-derived or ML forces (millions) → coarse-grained / continuum (household objects). Each rung has its own math and explicit handoffs (QM/MM, coarse-graining). v1 builds the first rung and the interface the others plug into.
- **"Accurate" means:** as accurate as the best method that runs interactively at the current atom count, with the sandbox always showing which rung and which approximations are in effect.

## 3. Decisions made

| Decision | Choice | Rejected alternatives |
|---|---|---|
| First rung | **Hybrid (C):** classical nuclei + quantum electrons (Born-Oppenheimer MD) | Quantum-only (slow to demo); classical-only (atoms unstable, no chemistry) |
| Electron representation | **Real-space grid Kohn-Sham DFT** | Gaussian-basis HF/DFT (per-element basis tables, CPU-bound); tight-binding (fitted per-pair parameters = hardcoding) |
| Language / GPU | **Rust + wgpu** (Vulkan on AMD; no CUDA available) | C++ + Vulkan; Python-first |
| Dependency policy | **Hand-write all physics kernels.** Crates only for plumbing (windowing, GPU, math, linear algebra). Door left open for pretrained ML potentials as a *data* dependency behind `ForceProvider` later. | Linking PySCF/xTB/OpenMM (defeats the purpose) |
| Reference codes | Used as **test oracles only** (published numbers), never linked | — |
| Per-element data allowed | **Z and isotope masses** (measured constants). Element colours/radii for display only. | Any fitted behavioural parameters |

## 4. Target hardware

- GPU: AMD Radeon RX 7900 XTX, 24 GB VRAM → wgpu/Vulkan compute. WGSL has no f64.
- CPU: AMD Ryzen 7 9800X3D, 8 cores. RAM: 32 GB.
- OS: Windows 11.

## 5. Architecture

Cargo workspace, one repo.

| Crate | Responsibility | Depends on |
|---|---|---|
| `units` | Atomic units internally (ħ = mₑ = e = 1; energy in Hartree, length in Bohr, time unit ≈ 0.0242 fs). Typed conversions to Å / eV / fs / K. CODATA constants. Isotope mass table. | — |
| `gpu` | wgpu device/queue, typed buffers, compute dispatch helpers, grid descriptors. Shared by solver and renderer. | wgpu |
| `qm` | Real-space Kohn-Sham solver. Input: nuclei (positions, Z) + net charge + multiplicity. Output: density, orbitals, total energy, forces. Pure function of nuclear positions plus a warm-start cache. | `gpu`, `units` |
| `sim` | Scene state, `ForceProvider` trait (and `Trainable`, for a provider that learns), velocity-Verlet integrator, thermostat, time control, energy bookkeeping. No rendering knowledge; `gpu` only as plumbing — the display payload holds a texture handle (M6; the alternative would be a type-erased payload). | `units`, `gpu` |
| `providers` | The force providers behind `ForceProvider`: the Kohn-Sham adapters (`CpuKs` on `qm`, `GpuKs` on `qm-gpu`), the surrogate (`SurrogateKs`, `Trainable`) and its sampling pass. `sim` never names them (M6). | `sim`, `qm`, `qm-gpu`, `gpu`, `surrogate`, `units` |
| `render` | Nuclei as spheres, density as volume raymarch, isosurfaces, orbital/spin views, camera. | `gpu` |
| `app` | winit window, egui UI, tools, time controls, readouts, scene I/O. Wires everything. | all |
| `validate` | Headless tests against reference values. | `qm`, `sim` |

### 5.1 Core types (`sim`)

```rust
struct Nucleus { pos: DVec3, vel: DVec3, z: u8, mass: f64 }          // f64, atomic units
struct Scene   { nuclei: Vec<Nucleus>, cell: Cell, net_charge: i32,
                 multiplicity: Option<u32>, thermostat: Option<Thermostat>, time: f64 }

struct ForceResult { energy: f64, forces: Vec<DVec3>, display: Option<DisplayHandle> }

trait ForceProvider {
    fn evaluate(&mut self, scene: &Scene) -> ForceResult;
}
```

`ForceProvider` is the extension point for every later rung: ML potentials, classical force fields, QM/MM composites. The dynamics loop never knows which it is talking to.

### 5.2 Precision

- Grid quantities (orbitals, density, potentials): **f32 on GPU**, as production GPU-DFT codes do. Energy reductions use pairwise/compensated summation.
- Nuclear positions, velocities, integrator, small dense linear algebra (Gram/Rayleigh-Ritz): **f64 on CPU**.
- If validation shows f32 limits energy conservation, mixed precision is a contained change inside `qm`.

## 6. Quantum solver (`qm`)

Spin-polarized Kohn-Sham DFT on a real-space grid.

**Grid.** Uniform cubic, spacing *h* (default 0.25 Bohr, user-adjustable). Isolated (Dirichlet) boundaries for v1; periodic cells later. Convergence with *h* is a test, never assumed.

**Hamiltonian** H = −½∇² + V_nuc + V_H + V_xc

- **Kinetic:** 8th-order finite-difference Laplacian stencil (PARSEC/Octopus style).
- **Nuclear potential:** each nucleus deposited as a narrow Gaussian charge (width tied to *h*) rather than a bare 1/r point, regularizing the singularity with *h*-controlled error. V_nuc + V_H come from **one Poisson solve** of total charge (electrons − nuclei). Behind a `NuclearPotential` trait so self-generated pseudopotentials can replace it for heavy elements later.
- **Poisson:** conjugate gradient with multigrid preconditioner; boundary values from a multipole expansion of the charge. FFT solver is a possible later optimization.
- **Exchange-correlation:** LSDA, Perdew-Wang 92 parameterization. The one fitted ingredient — a universal fit to the uniform electron gas, not per-element. GGA (PBE) later behind an `Xc` trait.
- **Spin:** two channels from day one (the hydrogen atom is open-shell).

**Eigensolver.** LOBPCG, block, preconditioned, warm-started from the previous frame's orbitals. Gram matrices reduced on GPU; the N×N Rayleigh-Ritz problem solved on CPU in f64. Chebyshev-filtered subspace iteration is the known upgrade if needed.

**SCF loop.** Density from occupied orbitals → Poisson → XC → H → eigensolve → Pulay/DIIS density mixing → repeat until density change < tolerance. Integer aufbau occupations in v1; Fermi smearing later for metals/degenerate states.

**Forces.** Hellmann-Feynman: F_I = −∫ ρ ∇V_I dr + Σ_J Z_I Z_J / R_IJ. No Pulay terms (grid basis doesn't move with nuclei). The egg-box effect (grid-phase energy ripple) is measured in validation and must be below threshold.

**Warm start.** Previous orbitals + extrapolated density → 2–5 SCF iterations per MD step instead of ~30 cold. Adding/removing a nucleus triggers a cold SCF seeded with the old subspace.

**Kernel inventory (WGSL):** Laplacian stencil, Poisson CG step (stencil + reductions), multigrid restrict/prolong/smooth, nuclear-charge deposition, density accumulation, XC evaluation, block axpy/dot reductions, force integration. ~8–10 kernels; orchestration in Rust.

**Capability envelope:** H through ~Ne all-electron accurately; heavier elements work but need fine grids until pseudopotentials land. Interactive range ~2–30 atoms. This is the "truth" rung.

## 7. Dynamics loop (`sim`)

- **Integrator:** velocity Verlet, fixed dt, default 0.25 fs (must resolve H–X stretch, ~10 fs period). User-adjustable with a warning above safe values.
- **Time speed ≠ timestep.** Speed changes steps attempted per rendered frame, never dt. If the solver can't keep up, sim lags wall-clock and the UI shows the true ratio. Pause, single-step, and rewind (flip velocities; Verlet is time-reversible) are supported.
- **Temperature:** from nuclear KE via equipartition, COM motion removed, DOF = 3N − 3 for free systems. Always displayed.
- **Thermostat:** optional Bussi stochastic velocity rescaling (CSVR). Berendsen rejected (wrong fluctuations). Nosé–Hoover chains later. Default off = NVE.
- **Energy bookkeeping:** total = nuclear KE + Kohn-Sham energy. NVE drift per ps displayed as the primary health indicator; validation pins acceptable drift.
- **Electrons:** N = ΣZ − net_charge. Multiplicity user-settable; default lowest-energy guess (paired for even N, one unpaired for odd).
- **Gravity / external fields:** `sim` supports an optional uniform external field (e.g. gravity *g*, a uniform electric field) applied to every nucleus in the integrator. Inter-particle gravity is **not** computed: at atomic scales it is ~10⁻³⁶–10⁻⁴⁰ of the Coulomb force, far below f32 resolution, and no atomistic code includes it. Uniform *g* only becomes physically visible on the coarse-grained/continuum rung (objects falling), where the same `ExternalField` hook applies.
- **Tool interactions enter as forces:** drag = stiff spring to cursor target; place = nucleus at rest; delete = remove + cold re-solve; freeze = zero velocity each step.
- **Threading:** simulation thread owns the scene and the `ForceProvider`; UI sends commands through a queue; sim publishes immutable snapshots (positions, density grid handle, energies, T, SCF stats) into a ring buffer the render thread reads. Render never blocks on an SCF.
- **Per-step sequence:** apply commands → half-kick → drift → `evaluate` → half-kick → thermostat → bookkeeping → publish.

## 8. Sandbox tooling & rendering (`app`, `render`)

**Display.** Nuclei as spheres (radius ∝ Z^⅓, conventional element colours — display only). Electron density as a volume raymarch of the GPU grid (no copy). Toggles: isosurface at chosen density, per-orbital view, spin density (↑ − ↓). No bond sticks — the density shows bonds.

**Tools** (egui side panel + 3D interaction):
- Place: periodic-table picker; click on a plane through the scene focus. Shift-click offsets the plane.
- Drag (spring), pin/freeze, delete, set net charge, set multiplicity.
- Time: play/pause, step, speed slider, rewind, dt field with warning.
- Thermostat: on/off, target T, coupling time.
- Grid: spacing h, box size, live cost estimate (points × orbitals).
- Readouts: sim time, wall ratio, T, total energy + drift, SCF iterations, per-nucleus force on hover.
- Scene I/O: small text format (positions, Z, velocities, settings). Presets: H, H₂, He, LiH, H₂O.

**Camera.** Orbit/pan/zoom, focus on selection.

**Non-goals (v1):** shadows, materials, VR, screenshot/video export.

## 9. Testing & validation (`validate`)

All headless.

1. **Unit:** each kernel vs a CPU f64 reference on tiny grids (Laplacian of a Gaussian, Poisson of a point charge, XC of a known density).
2. **Physics oracles** (each with pinned tolerance and cited source):
   - H atom: E_LSDA ≈ −0.479 Ha (exact −0.5; gap is the functional, documented). Spherical density.
   - H₂: bond length 0.74–0.77 Å; dissociation curve shape.
   - He: E_LSDA ≈ −2.83 Ha; closed shell.
   - LiH; H₂O geometry (angle ≈ 104°, O–H ≈ 0.97 Å).
   - Grid convergence: energy vs h monotone; egg-box ripple amplitude below threshold.
3. **Dynamics:** H₂ vibration within ~10% of 4400 cm⁻¹ (LSDA shift documented); NVE drift below threshold over 1 ps; thermostat reaches target T.

Solver changes must pass 1–2; dynamics changes must pass 3. Reference numbers are sourced in `validate/references.md` (created during implementation).

## 10. Model assignment policy

Every implementation task is tagged with the model to execute it:

- **Fable 5.1:** solver math, integrators, force derivations, numerical-stability debugging, validation oracles, physics review — anything where a wrong answer still looks plausible.
- **Opus 5.5:** GPU plumbing, UI, rendering, tooling, tests with pinned expected values, refactors.

Subagents are dispatched on the tagged model. See `CLAUDE.md` and `.claude/agents/`.

## 11. Roadmap beyond v1 (not in this spec's scope)

1. Self-generated pseudopotentials (from our own all-electron atomic solves) for Z > 10.
2. Periodic cells; Fermi smearing; PBE.
3. Second rung: classical atomistic `ForceProvider` — hand-written force field and/or pretrained ML potential (MACE/UMA-class via ONNX) for thousands–millions of atoms.
4. QM/MM composite provider (quantum region embedded in classical surroundings).
5. Coarse-grained / continuum rung for household-scale objects.
6. Nuclear physics module, if ever wanted.

### 11.x Nuclear physics (ruled 2026-10-13)

Explicit protons and neutrons are permanently out of scope: the nucleus is 1e-5 of the atom (1 fm grids,
1e-22 s steps), there is no first-principles route from QCD (nuclear-structure codes rely on fitted
effective interactions, which rule 1 forbids), and decay is a stochastic rate process spanning 1e-24 s
to 1e24 yr. The point-nucleus description is the *correct* one for everything electrons can feel.

Nuclear physics enters instead as a **nuclear-events layer** (milestone after the surrogate, tools and
Z > 10): per-nucleus state (Z, N, excitation) with stochastic α/β/γ events whose Q-values come from the
isotope mass table already in `units` and whose rates come from measured half-lives/branching ratios
(NUBASE) — an explicit, documented extension of rule 1 to measured nuclear decay data. Transmutation is
live: the electrons see the new Z through the existing solver. Decay time runs on a separate, labelled
"nuclear clock" multiplier, never through the MD time speed. Details: `docs/superpowers/notes/2026-10-13-m5-user-requests.md`.

## 12. Prior art (for research prep)

Interactive MD: Narupa/NanoVer, SAMSON, Atomify. Builders: Avogadro, ASE GUI. Education: Molecular Workbench, Falstad quantum applets. Reference solvers (oracles only): PARSEC, Octopus (real-space DFT); PySCF, Psi4 (Gaussian basis); OpenMM, LAMMPS, GROMACS (classical MD). Multiscale: QM/MM (Karplus, Levitt, Warshel). Universal ML potentials: MACE-MP-0, Orb, MatterSim, UMA. Nothing combines interactive sandbox + emergent chemistry + a scale ladder in one tool.
