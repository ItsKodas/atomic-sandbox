# Atomic Sandbox

Interactive 3D sandbox simulating matter from the atomic level up with real physics: classical nuclei + Kohn-Sham DFT electrons on a real-space GPU grid (Born-Oppenheimer MD), built in Rust + wgpu. The solver engine is called **LIRA** (live interactive real-space ab initio). Chemistry and the periodic table must *emerge*; nothing per-element is hardcoded.

**Read first:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (the design) and the current plan in `docs/superpowers/plans/`. **Detailed reference** (every flag and `--ui` screenshot key, the viewer layout, pseudopotential plumbing, the replay format, each suite's contents and timings): `docs/dev-reference.md` — grep for the section you need instead of loading the whole file.

## Non-negotiable rules

1. **Per-element data allowed: Z and isotope mass only.** Display colours/radii are fine (they are not physics). Any fitted per-element or per-pair behavioural parameter is a design violation — raise it, don't add it. (The one signed-off exception: the 2D molecule builder's valence/covalent-radius tables in `crates/app/src/builder/chem.rs`, a starting-geometry aid that never reaches `sim`/`qm`/`providers`.)
2. **Use existing code when it is better** (user, 2026-11-16, replacing "hand-written physics"): adopt a library when it beats what we would write and saves significant work; write our own where ours would be better. Hard constraints: runs on the AMD/Vulkan GPU path (no CUDA-only), ships in the single Windows installer (avoid Python/Fortran/MPI runtimes), licence compatible with PolyForm Noncommercial (MIT/Apache/MPL yes, GPL no), and keeps interactivity (cancel, warm start). Whatever is adopted is still validated by our oracles. Pretrained ML potentials may be *loaded as data* behind `ForceProvider`; whether they may be a foundation is rule 1's question.
3. **Atomic units internally** (ħ = mₑ = e = 1; Hartree, Bohr). Convert only at the UI/display boundary via the `units` crate. Never mix unit systems inside `qm` or `sim`.
4. **Precision:** grid quantities f32 on GPU; nuclear state, integrator, and small dense linear algebra f64 on CPU. Energy reductions use compensated/pairwise summation.
5. **`ForceProvider` is the boundary.** `sim` never knows what computes forces; it keeps the `ForceProvider`/`Trainable` traits and depends on `units` and `gpu` only. The providers (`CpuKs`, `GpuKs`, `SurrogateKs`, the sampling pass) live in `providers`; a provider that learns is reached through `ForceProvider::as_trainable`, and nothing downcasts a provider. `qm` never knows about rendering or UI.
6. **Validation gates:** solver changes must pass the unit + physics-oracle suites in `validate`; dynamics changes must also pass the dynamics suite. Reference numbers are cited in `crates/validate/references.md`. Never loosen a tolerance to make a test pass — investigate.
7. **Physics honesty.** When a result differs from exact/experimental values because of the approximation (e.g. LSDA H atom at −0.479 Ha vs exact −0.5), document the gap; don't hide it.
8. **Time speed ≠ timestep.** UI speed changes steps per frame, never dt.

## Model assignment

Tag every plan task and dispatch subagents accordingly: **Fable 5.1** (`claude-fable-5-1`) for solver math, integrators, force derivations, numerical-stability debugging, validation oracles, physics review — anything where a wrong answer still looks plausible; **Opus 5.5** (`claude-opus-5-5`) for GPU plumbing, UI, rendering, tooling, tests with pinned values, refactors. Agents in `.claude/agents/`: `physics-reviewer` (Fable, read-only; reviews derivations, units, signs, conservation before solver/dynamics code lands), `solver-implementer` (Fable, TDD against CPU f64 references), `gpu-kernel-engineer` (Opus, WGSL kernels + Rust dispatch against a given spec and CPU reference), `validation-runner` (Opus, runs suites, never edits physics).

## Workflow

- Superpowers skills apply: brainstorm → spec → plan → TDD implementation → review → verify before claiming done.
- **Keep `README.md` current** (user instruction): whenever user-facing behaviour, controls, presets, performance numbers, milestone status or the crate layout change, update the README in the same commit or milestone merge. It is the public face of the repo; the spec/plans/notes stay the engineering record.
- Commit the spec/plan changes alongside code. Commit messages end with the Claude attribution line.
- When a fact here or in `docs/dev-reference.md` changes, update it there; keep this file short (it is loaded into every session and every subagent).

## Crates

`units` · `qm` (CPU f64 reference solver) · `qm-gpu` (the GPU solver) · `gpu` · `render` · `sim` (scene, integrator, thermostat, the `Dynamics` driver, relax, bonds, charges; the `ForceProvider`/`Trainable` boundary) · `providers` (Kohn-Sham adapters `ks.rs`; the surrogate provider `surrogate/` — its decision list is `mod.rs`'s module doc; the sampling pass `sampling/`; the pseudopotential cache `pseudo`) · `surrogate` (the model: descriptors, basis, Bayesian fit, sidecar, the shared collections) · `app` (the viewer; worker in `src/worker/`, UI in `src/ui/` — the docking workspace in `src/ui/workspace/`: tree, auto-hide, layouts, detached windows, drag — builder in `src/builder/`, replay in `src/replay.rs`) · `validate` (the suites).

## Build and run

- `cargo build --workspace`. App, always release: `cargo run -p app --release [-- --backend gpu|cpu|estimated] [--preset <name>] [--quality fast|balanced|accurate|precise] [--ionic none|cores|accepted|all] [--scene <file>]`; `start.bat` / `build.bat` build incrementally and launch / package.
- App defaults: GPU backend, **Accurate** grid (h = σ = 0.15 Bohr), `Accepted` ionic policy (pseudopotentials for every species the recipe accepts), container walls on, starts paused on H₂. `--backend cpu` without `--quality` loads Balanced (an 81³ CPU step ≈ 40 s, uncancellable).
- `Scene::preset`, a `Scene` literal and a scene file without an `ionic` line keep the file's own grid (49³/57³ × 0.25) and `ionic none`, which the tests rely on. Replay scripts (`crates/app/replays`) must pin `action quality`, `action walls` and `action ionic`.
- Every control that reaches the worker or changes the scene goes through `ui::Action` (so `--record` sees it); display-only settings write `UiState` directly and are not recorded.
- Docs screenshots: hidden `--screenshot <png> [--ui "<k=v;…>"]` — keys in `docs/dev-reference.md` "Build, run and the app's defaults".
- Releases (tags, the workflow, the signing key, the checklist; ask before any tag push or publish): `docs/dev-reference.md` "Releasing".

## Which suite gates which change

All heavy tests are release-only (`#[cfg_attr(debug_assertions, ignore)]`); run one heavy GPU job at a time.

| Change | Command | Time |
|---|---|---|
| any | `cargo test -p units -p qm -p sim -p surrogate --release`, `cargo test -p providers --release` | ~9.5 min + ~10–11.5 min |
| solver (`qm`, `qm-gpu`) | `cargo test -p validate --release --test oracles` and `--test gpu_vs_cpu` | ~51–54 min, ~8 min |
| dynamics | `cargo test -p validate --release --test dynamics` | ~42–43 min (V5's CO₂⁺ NVE and CPU work integral; the stray-atom NVE re-entry at 101³, ~14 min) |
| surrogate / provider / policy | `cargo test -p validate --release --test surrogate` | ~20–40 min (with the stray-atom and trust-tightening tests) |
| shared-model measurements (not gates) | `--test transfer`, `--test capping` (`#[ignore]`d; commands in `references.md`) | hours |
| collection migration, V11 (measurement, not a gate) | `FD_DRY_RUN_COPY=<copy of a pre-FD collection> cargo test -p validate --release --test migration dry_run -- --ignored --nocapture` (on a copy only, never `models/collections`) | ~17 h GPU for a 1 657-sample Accurate collection (V11, 37 s/sample) |
| bonds / relax / isotopes | `cargo test -p validate --release --test tools` | ~3–10 min |
| charges | `cargo test -p validate --release --test charges` (CPU) | ~5.5–11 min |
| pseudopotentials | `--test pseudo`, `--test pseudo_gen`, `--test pseudo_oracles` | ~39 min–1 h, ~4.5–9 min, ~65 min–2 h |
| orbital display | `cargo test -p validate --release --test orbitals` | ~4–10 min |
| live orbitals (Harris display solves) | `cargo test -p validate --release --test harris` (V-A4 H₂O alone ~15 min) | ~11–18 min |
| Harris display orbitals (`qm_gpu::harris`, `providers::harris`) | `cargo test -p validate --release --test harris` | ~11–18 min |
| viewer / UI / worker | `cargo test -p app --release` (incl. the replay scripts) | ~10 min |
| molecule builder | `cargo test -p app --release --bin app -- builder::` | seconds |

Times as of 2026-11-20 at `-j 4`. Measured that day beside another agent's GPU suite: oracles 54 min, gpu_vs_cpu 8 min, dynamics 42 min, charges 5.5 min, orbitals 10 min, harris 18 min, and the lower ends of the pseudo rows (39 min, 4.5 min, 65 min); contention inflates GPU times. Per-suite numbers: `docs/dev-reference.md` "Test suites". `cargo test -p validate --release` runs every validate binary. What each suite checks and its measured numbers: `docs/dev-reference.md` "Test suites" and `crates/validate/references.md`.

## Hardware

AMD RX 7900 XTX (24 GB, Vulkan via wgpu — **no CUDA**), Ryzen 7 9800X3D, 32 GB RAM, Windows 11. WGSL has no f64.
