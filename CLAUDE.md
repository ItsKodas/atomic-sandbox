# Atomic Sandbox

Interactive 3D sandbox simulating matter from the atomic level up with real physics: classical nuclei + Kohn-Sham DFT electrons on a real-space GPU grid (Born-Oppenheimer MD), built in Rust + wgpu. Chemistry and the periodic table must *emerge*; nothing per-element is hardcoded.

**Read first:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (the design) and the current plan in `docs/superpowers/plans/`.

## Non-negotiable rules

1. **Per-element data allowed: Z and isotope mass only.** Display colours/radii are fine (they are not physics). Any fitted per-element or per-pair behavioural parameter is a design violation — raise it, don't add it.
2. **Hand-written physics.** Crates for plumbing only (wgpu, winit, egui, nalgebra/glam, rand). No chemistry/MD libraries. Pretrained ML potentials may later be *loaded as data* behind `ForceProvider`, never as the foundation.
3. **Atomic units internally** (ħ = mₑ = e = 1; Hartree, Bohr). Convert only at the UI/display boundary via the `units` crate. Never mix unit systems inside `qm` or `sim`.
4. **Precision:** grid quantities f32 on GPU; nuclear state, integrator, and small dense linear algebra f64 on CPU. Energy reductions use compensated/pairwise summation.
5. **`ForceProvider` is the boundary.** `sim` never knows what computes forces. `qm` never knows about rendering or UI.
6. **Validation gates:** solver changes must pass the unit + physics-oracle suites in `validate`; dynamics changes must also pass the dynamics suite. Reference numbers are cited in `validate/references.md`. Never loosen a tolerance to make a test pass — investigate.
7. **Physics honesty.** When a result differs from exact/experimental values because of the approximation (e.g. LSDA H atom at −0.479 Ha vs exact −0.5), document the gap; don't hide it.
8. **Time speed ≠ timestep.** UI speed changes steps per frame, never dt.

## Model assignment

Tag every plan task and dispatch subagents accordingly:

- **Fable 5.1 (`claude-fable-5-1`):** solver math, integrators, force derivations, numerical-stability debugging, validation oracles, physics review — anything where a wrong answer still looks plausible.
- **Opus 5.5 (`claude-opus-5-5`):** GPU plumbing, UI, rendering, tooling, tests with pinned expected values, refactors.

Agents in `.claude/agents/`:
- `physics-reviewer` (Fable, read-only) — reviews derivations, units, signs, conservation before solver/dynamics code lands.
- `solver-implementer` (Fable) — implements numerical algorithms TDD-style against CPU f64 references.
- `gpu-kernel-engineer` (Opus) — WGSL kernels + Rust dispatch against a given spec and CPU reference.
- `validation-runner` (Opus) — runs the `validate` suites, reports numbers, never edits physics.

## Workflow

- Superpowers skills apply: brainstorm → spec → plan → TDD implementation → review → verify before claiming done.
- Commit the spec/plan changes alongside code. Commit messages end with the Claude attribution line.
- Build: `cargo build --workspace`. App: `cargo run -p app` (the `app` crate does not exist yet in M1).
- Unit tests: `cargo test -p units -p qm --release` (~75 s). The `qm` SCF/forces tests are release-only in practice (debug is far too slow).
- Oracle suite: `cargo test -p validate --release` (~31 min; H, He and H₂ oracles). The Li oracle is `#[ignore]`d as a documented limitation (see `li_atom_lsda` in `crates/validate/tests/oracles.rs`).
- `cargo test --workspace` (debug) skips the oracles by design: they are `#[cfg_attr(debug_assertions, ignore)]` and run only under `--release`.

## Hardware

AMD RX 7900 XTX (24 GB, Vulkan via wgpu — **no CUDA**), Ryzen 7 9800X3D, 32 GB RAM, Windows 11. WGSL has no f64.
