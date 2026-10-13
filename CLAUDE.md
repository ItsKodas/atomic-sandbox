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
- **Keep `README.md` current** (user instruction): whenever user-facing behaviour, controls, presets, performance numbers, milestone status or the crate layout change, update the README in the same commit or milestone merge. It is the public face of the repo; the spec/plans/notes stay the engineering record.
- Commit the spec/plan changes alongside code. Commit messages end with the Claude attribution line.
- Build: `cargo build --workspace`. App: `cargo run -p app --release [-- --backend gpu|cpu] [--preset h2o]` (always release). The default solver backend is the GPU (`qm-gpu` on the window's device; the density texture is bound by the renderer without a CPU copy, and a newer edit abandons the solve in flight at the next SCF step); the panel's "solver" radio switches to the CPU reference (`qm::scf`, uncancellable) and back by re-spawning the worker with the current scene. Since M4 the worker runs `sim::Dynamics`: the app starts paused on H₂ at 1.6 Bohr (compressed from the smeared minimum so it vibrates); Space / the Play button starts Born-Oppenheimer MD, speed = steps per tick (never dt — rule 8), measured fs/s shown; Pause and scene edits (teleport/add/remove/ReplaceScene/backend/grid) abort the solve in flight, drag moves never do (they only move the spring target). Hidden `--screenshot <png> [--preset p] [--backend b] [--play-ms n]` plays for n ms before capturing. One-shot solve times, RX 7900 XTX (after M3 Task 10): H₂ 49³ ~0.2 s (warm ~0.1 s), H₂O 57³ ~1.8 s (warm ~1.0 s), H₂O 97³ ~3.6 s (warm ~1.8 s) at the validate suite's tol 1e-6 (CPU: H₂ ~6 s, H₂O 57³ ~2 min). The render thread is not blocked by a running solve (frame p95 ≈ 6 ms during a 97³ solve). Since M5 the solver radio has a third option, **estimated**: `sim::SurrogateKs` (crate `surrogate` — a linear ACE-style potential fitted to this solver's own energies/forces, Bayesian uncertainty, DFT fallback + active learning, background audits) wraps the current teacher; with no model the panel runs a sampling pass (H₂ 49³ ≈ 30 s, 273 solves; passes over ~2 min ask first); models autosave per session and save/load as `<scene>.model`. Rates: H₂ ≈ 40 000 fs/s (2.4e5 steps/s) with audits every 2000, ≈ 1.2e5 steps/s with audits off, vs ~3 fs/s DFT; at speed 1000 the background auditor saturates (one audit per teacher solve). Flags: `--backend estimated` / `--estimated`, `--sample` (run the default pass headlessly first), `--speed n`, `--audit n`. The policy constants and their justifications: `crates/validate/references.md` "Surrogate (M5)" table; the decision list: `crates/sim/src/surrogate_ks.rs` module doc. `.claude/launch.json` has the plain run command as `atomic-sandbox`.
- Viewer manual checklist (run after UI/render changes): Task 7, Step 5 of `docs/superpowers/plans/2026-10-13-m2-viewer.md`. Which items were verified in a window and which only by code reading: `docs/superpowers/notes/2026-10-13-m2-viewer-checklist.md` (the manual record until M5's input-replay harness).
- Viewer tests: `cargo test -p app --release` (~3.5 min since M5; the GPU tests are serialised) runs the `sim` worker tests (dynamics driver through the worker: play/pause/step, drag flood, provider failure, warm-start witness) (real SCF solves, both backends; the GPU ones skip without an adapter). Like the oracles, they are `#[cfg_attr(debug_assertions, ignore)]`, so a debug `cargo test -p app` shows them as ignored.
- Unit tests: `cargo test -p units -p qm --release` (~75 s); `cargo test -p sim --release` (~10 s, the dynamics crate: integrator, CSVR, driver on analytic providers plus a few GPU provider tests). The `qm` SCF/forces tests are release-only in practice (debug is far too slow).
- Oracle suite: `cargo test -p validate --release` runs ALL validate binaries (oracles ~31 min + gpu_vs_cpu + dynamics); the σ → 0 oracles alone are `--test oracles` (H, He and H₂). The Li oracle is `#[ignore]`d as a documented limitation (see `li_atom_lsda` in `crates/validate/tests/oracles.rs`).
- Surrogate suite (M5): `cargo test -p validate --release --test surrogate` (~4 min on the GPU; H₂ frequency vs the M4 DFT-MD pin, fresh-point surface reproduction, 10 ps NVE on the model, OOD fallback rules, H₃⁺ tier at 49³, active learning from no model; `SURROGATE_SEED` override). Surrogate/provider/policy changes must pass it (rule 6).
- Dynamics suite (M4): `cargo test -p validate --release --test dynamics` (~6 min solo on the GPU; H₂ vibration vs the curvature of its own σ = 0.25 surface, 0.5 ps NVE drift, CSVR on H₃⁺, GPU rewind; the H₂O-preset test is `#[ignore]`d as a documented resolution limit; `DYNAMICS_SEED` overrides the thermostat seed). Dynamics changes must pass it (rule 6).
- GPU-vs-CPU suite (M3): `cargo test -p validate --release --test gpu_vs_cpu` (~3.5 min; energies, forces and the H₂ curve shape at **fixed σ = h = 0.25**, 49³, plus a timing table written to the gitignored `crates/validate/last-run-gpu.md`). Skips cleanly without a GPU adapter. Why fixed σ and not the σ → 0 oracles: `crates/validate/tests/gpu_vs_cpu.rs` module doc / `references.md` "GPU vs CPU at fixed σ".
- `cargo test --workspace` (debug) skips the oracles by design: they are `#[cfg_attr(debug_assertions, ignore)]` and run only under `--release`.

## Hardware

AMD RX 7900 XTX (24 GB, Vulkan via wgpu — **no CUDA**), Ryzen 7 9800X3D, 32 GB RAM, Windows 11. WGSL has no f64.
