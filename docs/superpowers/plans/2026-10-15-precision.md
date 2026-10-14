# Precision — precision renders, exact and DMC teachers, Δ-learned precision models, the model manager: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Near-exact electronic structure that cannot run live, made useful anyway:
- **Precision renders** — a prepared scene is solved step by step in the background (any slow solver: DFT at the Precise quality first, later the exact and DMC tiers), checkpointed, and played back as it fills in.
- **Precise teachers** — explicitly correlated Gaussians (ECG, "exact") for ≤ 3 electrons; fixed-node diffusion Monte Carlo (DMC) from this solver's own orbitals for everything else.
- **Precision models** — a Δ-model of E_precise − E_LSDA on top of the LSDA surrogate, played live.
- **The model manager** — every model and long job in one place, compared, chosen and explained in plain language.

**Architecture:**
- **New crate `jobs`** (std + `userdata`): the job record, the queue and runner thread, `frames.asfr`, checkpoints, density keyframes. It drives the unchanged `sim::Dynamics` with whatever provider the job names.
- **New crate `ecg`** (CPU f64, `nalgebra` for the dense generalised eigenproblem): integrals, antisymmetrisation, the stochastic variational method, Hellmann–Feynman forces.
- **New crates `qmc`** (CPU f64 reference: B-spline orbitals, Slater–Jastrow, VMC, linear-method optimisation, DMC, pseudopotentials with the locality approximation and T-moves, blocking) and **`qmc-gpu`** (WGSL walkers against it).
- **`providers`** gains `precise::{EcgKs, DmcTeacher}` and `delta::DeltaKs`; `providers::ks` gains a crate-internal, read-only orbital accessor.
- **`surrogate`** gains the heteroscedastic Δ fit and the precise label store / Δ files (`library::precise`).
- **`sim`** gains plain records only (`DeltaStats`, a tier string): rule 5.
- **`app`** gains the playback mode and timeline, the render form, the jobs chip, the Models & jobs page (Models / Jobs / Compare), the chooser.
- **`validate`** gains `--test precision_ecg`, `--test precision_qmc`, `--test precision_delta`, `--test precision_render` and two tripwires.

**Tech Stack:** Rust; std, `userdata` (the user-data milestone's crate), wgpu/WGSL, egui, `nalgebra` (plumbing: dense f64 linear algebra). No chemistry, QMC or ML libraries (rule 2).

**Spec:** `docs/superpowers/specs/2026-10-15-precision-design.md` (section numbers below are the spec's). Inputs: the user's request of 2026-10-15 and the controller's commitments (spec §1; `docs/superpowers/notes/2026-10-13-m5-user-requests.md` § "Precision milestone"); the M5, M6, user-data, M7 and orbital-viewer specs.

**Model tags** (CLAUDE.md "Model assignment"):
- **[Fable, Opus fallback + mandatory physics review]:** 3.1, 3.2, 4.0 (physics half), 4.1, 4.2, 4.3, 4.4, 4.6, 4.7, 5.2, 5.3, 5.4, 5.6 — solver math, estimators, statistics, the Δ fit and its trust, every oracle.
- **[Opus]:** 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 3.3, 4.0 (GPU half), 4.5 (`gpu-kernel-engineer`), 4.8, 5.1, 5.5 (with a Fable review of the R2 bookkeeping), 5.7.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, or the Fable 5.1 line when Fable ran the task.

**Stages, each shipping something usable** (spec §10):

| stage | tasks | ships | starts after |
|---|---|---|---|
| 1 Precision render | 1.1–1.5 | background step-by-step renders with today's solvers (DFT at any Quality, incl. Precise), checkpointed, with progressive playback | the user-data milestone is merged |
| 2 Model manager | 2.1–2.4 | the Models & jobs page, Compare, the chooser, plain-language copy | stage 1 |
| 3 Exact tier | 3.1–3.3 | exact (ECG) renders and labels for ≤ 3 electrons | stage 2 (independent of M7) |
| 4 DMC tier | 4.0–4.8 | DMC energies along trajectories and as labels | M7 (pseudopotentials) and stage 2 |
| 5 Precision models | 5.1–5.7 | Δ-models played live; R2 renders | 5.1–5.3 after stage 3; 5.4–5.7 after stage 4 |

**Branches:** one per stage (`precision-1-render`, …), each from the main of its time, merged after its review (memory: hands-off execution of reviewed milestones). Before each stage, re-read the post-merge versions of the files its tasks touch (`crates/sim/src/{provider.rs, dynamics.rs}`, `crates/providers/src/{ks.rs, surrogate/*}`, `crates/surrogate/src/{library/*, sidecar.rs, fit.rs}`, `crates/app/src/{worker/*, ui/*}`, and for stage 4 `crates/qm/src/{radial/*, ionic.rs}`) and adjust names; record each change in the task report.

## Global Constraints

- **Rule 1.** No per-element data. ECG nonlinear parameters come from the universal draw of spec §14 scaled by the scene's own Z and distances; Jastrow coefficients are per-scene variational outputs; trial orbitals are this solver's; pseudopotentials are M7's. A grep-style test per crate (`no_element_tables`) asserts that no `match z`/Z-indexed constant table exists outside `units` and display code.
- **Rule 2.** No QMC, quantum-chemistry or ML crates. `nalgebra` only for dense f64 linear algebra (`ecg`, the Δ fit already uses its own Cholesky).
- **Rule 3.** Atomic units in every file, solver and record; kcal/mol and eV appear only in the UI (`units`). 1 kcal/mol = 1.5936 mHa.
- **Rule 4.** ECG entirely f64 on the CPU. DMC: walkers f32 on the GPU with positions relative to the cell centre; determinant inverses refreshed in f64 on the CPU every `DET_REFRESH` sweeps; per-walker energies reduced as f32 partials → Neumaier f64 on the CPU; branching and population in f64.
- **Rule 5.** `sim` never sees orbitals, walkers, ECG bases or file formats: `jobs` and `providers` hold them; `sim` gains plain records only. Orbitals cross only inside `providers` (the accessor is `pub(crate)`).
- **Rule 6.** New methods are gated by new suites (`validate --test precision_*`); the LSDA suites must stay green and unchanged (no LSDA solver code changes in this plan; the orbital accessor is read-only and pinned bitwise, Task 4.6). Each precise method gets a `method_rev` and a tripwire (Tasks 3.3, 4.6). Never loosen a bound — investigate.
- **Rule 7.** Every tier states its systematic error in words (spec §8.6); per-frame bars are statistical/posterior only and say so; R3 renders say "motion: LSDA"; every render says "classical nuclei".
- **Rule 8.** A job's dt is fixed at submission; playback speed is frames per wall second; the live speed setting is untouched.
- **Pre-measurement rule.** A bound marked "measured" is pinned by the task that first runs it, at the measured value × the stated margin, with the run recorded in `references.md` § "Precision"; it is never loosened later.
- **Statistics.** Every DMC number published with an error bar has passed the blocking plateau test (Flyvbjerg–Petersen); otherwise it is shown as "error bar not yet reliable".
- **User data.** Jobs live under `<root>/jobs/`, precise models under `<root>/models/precise/`; every write through `userdata::atomic_write`; tests and automation never touch the real root (user-data §4.1, L2).
- **The app never destroys data the user did not delete**: deleting a job or a model renames it (`.deleted-<unix>` / `.forgotten-<unix>`); freeing space is an explicit two-click action.
- **Keep README.md current** (CLAUDE.md): each stage's last task updates it in the same commit (controls, flags, job files, tiers, measured costs).
- **Commit with `git commit -o -- <paths>`** (memory rule).

## Review Focus

Each item names the failure and the test that pins it.

1. **A render that is not the live physics** (a second integrator, a different bookkeeping). Task 1.2 `a_dft_render_equals_live_dynamics_bitwise`.
2. **A resume that silently changes the run.** Task 1.2 `resume_after_a_killed_process_is_bitwise_for_deterministic_teachers` (Morse; ECG in Task 3.3); `a_torn_frame_is_truncated_and_recomputed`.
3. **A render starving the live sandbox.** Task 1.3 `share_mode_yields_between_chunks` and the measured latency row.
4. **Wrong exact numbers.** Task 3.1 `h2_at_1_4011_matches_pachucki`, `h3plus_matches_the_ecg_record`, `heh_plus_matches_pachucki`, `helium_atom`; each also asserts the variational bound (E ≥ E_exact − 1e-10).
5. **ECG forces inconsistent with the energy the trajectory sees.** Task 3.2 `hf_force_is_the_frozen_basis_derivative`, `ecg_render_conserves_energy_across_refreshes`.
6. **A biased or overconfident DMC.** Task 4.3 `h2_nodeless_dmc_matches_ecg`, `error_bars_are_honest` (40 seeds); Task 4.5 `gpu_dmc_matches_cpu_reference`; Task 4.4 `time_step_bias_is_measured_and_extrapolated`.
7. **Pseudopotential handling in DMC** (sign of the nonlocal term, quadrature normalisation, T-move weights). Task 4.4 `nonlocal_quadrature_reproduces_the_kb_operator`, `t_moves_and_locality_are_both_reported`, `size_consistency_of_t_moves`.
8. **The headline oracle.** Task 4.6 `water_atomization_within_4_kcal`; the decomposition `pseudopotential_error_offline`.
9. **The orbital accessor perturbing the LSDA solve.** Task 4.6 `reading_orbitals_leaves_the_solve_bitwise_unchanged`.
10. **A Δ fit that is overconfident with noisy labels.** Task 5.2 `chi_square_trust_rejects_overconfidence`, `predictive_sigma_is_calibrated_on_synthetic_noise`; Task 5.6 `delta_uncertainty_is_calibrated_on_dmc`.
11. **A silent surface change** when the Δ-model starts or stops answering, or is refitted. Task 5.3 `losing_precision_books_a_surface_jump_and_says_so`.
12. **Keys that mix teachers.** Task 3.3 `precise_keys_are_pinned`; Task 5.1 `delta_key_includes_base_and_precise_keys`; Task 5.1 `a_dmc_label_never_enters_an_ecg_model`.
13. **Costs that lie.** Every form's estimate comes from measured medians (Tasks 1.2, 4.8); Task 4.0 replaces the spec's [est] table with measurements before stage 4 builds anything else.

## File Structure

```
Cargo.toml                                            # members: jobs (1.1), ecg (3.1), qmc (4.1), qmc-gpu (4.5)
crates/jobs/src/{lib.rs, job.rs, frames.rs, density.rs, runner.rs, eta.rs, files.rs, tests.rs}   # 1.1–1.3
crates/sim/src/provider.rs                            # 5.3: DeltaStats in EvalStats; tier text
crates/app/src/ui/playback/{mod.rs, timeline.rs}      # 1.4
crates/app/src/ui/render_form.rs                      # 1.4 (DFT), 3.3 (exact), 4.8 (DMC R3), 5.5 (R2)
crates/app/src/ui/manager/{mod.rs, models.rs, jobs.rs, compare.rs, copy.rs}   # 2.1–2.4, 5.7
crates/app/src/ui/dock/model.rs                       # 2.4 chooser, 5.7 Δ rows
crates/app/src/ui/{menus.rs, search.rs, toolbar.rs}   # 1.4, 2.1
crates/app/src/{main.rs, viewer.rs}                   # 1.5 (runner lifecycle, resume at launch)
crates/app/src/settings.rs                            # 1.5 (jobs.* profile rows)
crates/ecg/src/{lib.rs, gaussian.rs, integrals.rs, boys.rs, spin.rs, solve.rs, svm.rs, forces.rs, tests.rs}   # 3.1–3.2
crates/qmc/src/{lib.rs, bspline.rs, slater.rs, jastrow.rs, wavefunction.rs, local_energy.rs, vmc.rs, optimise.rs, dmc.rs, pseudo.rs, blocking.rs, rng.rs, tests.rs}   # 4.1–4.4
crates/qmc-gpu/src/{lib.rs, walkers.rs, shaders/*.wgsl, tests.rs}   # 4.5
crates/providers/src/ks.rs                            # 4.6: pub(crate) occupied_orbitals()
crates/providers/src/precise/{mod.rs, ecg.rs, dmc.rs, key.rs}       # 3.2, 3.3, 4.6
crates/providers/src/delta/{mod.rs, policy.rs, tests.rs}           # 5.3
crates/surrogate/src/fit.rs                           # 5.2: heteroscedastic rows (σ_i), energy-only rows
crates/surrogate/src/library/precise.rs               # 5.1: PreciseKey store, label files, Δ files
crates/qm/src/radial/{atom.rs, recipe.rs}             # 4.7 (conditional): Hartree–Fock reference atom
crates/validate/tests/{precision_ecg.rs, precision_qmc.rs, precision_delta.rs, precision_render.rs, tripwire_precise.rs}
crates/validate/references.md                         # § "Precision" (every number of record)
README.md, CLAUDE.md                                  # 1.5, 2.4, 3.3, 4.8, 5.7
```

---

## Stage 1 — Precision render with the existing solvers

### Task 1.1: the `jobs` crate — job record, frame log, density keyframes **[Opus]**

**Files:** create `crates/jobs/src/{lib.rs, job.rs, frames.rs, density.rs, files.rs, tests.rs}`; `Cargo.toml` member.

**Interfaces (produces):**
```rust
pub struct JobId(String);                       // "<UTC yyyy-mm-ddThhmm>-<slug>-<4 hex of FNV-1a>"
pub enum JobKind { Render(RenderSpec), Label(LabelSpec), Rebuild(RebuildSpec) }   // Label/Rebuild bodies land in 4.8/5.1
pub struct RenderSpec { pub render: RenderKind /* R1 | R2 | R3 */, pub steps: u64, pub dt_fs: f64, pub thermostat: Option<sim::Thermostat>, pub field: Option<sim::ExternalField>, pub walls: Option<f64>, pub density_every: u32, pub check_every: Option<u32> }
pub enum TeacherSpec { Dft { quality: Quality /* or explicit n, h, σ */, backend: Backend }, Ecg, Dmc(PreciseKeyFields) }
pub enum JobState { Queued, Running, Paused { why: String }, Done, Failed { error: String }, Cancelled }
pub struct Job { pub id: JobId, pub name: String, pub kind: JobKind, pub scene: sim::Scene, pub teacher: TeacherSpec, pub base: Option<String /* library key text */>, pub seed: u64, pub created: u64, pub state: JobState, pub progress: Progress, pub cost: CostHistory }
pub struct FrameRecord { pub step: u64, pub t: f64, pub pos: Vec<[f64;3]>, pub vel: Vec<[f64;3]>, pub forces: Vec<[f64;3]>, pub rng: [u64; 4], pub e_pot: f64, pub ke: f64, pub e_conserved: f64, pub work: WorkRows, pub temperature: f64, pub sigma_model: Option<f64>, pub label: Option<(f64, f64) /* E, σ_stat */>, pub source: FrameSource, pub solve_ms: u64, pub density_key: Option<u32>, pub notes: Vec<String> }
pub struct FrameWriter; impl FrameWriter { pub fn create(dir: &Path, header: &FrameHeader) -> io::Result<Self>; pub fn append(&mut self, f: &FrameRecord) -> io::Result<()>; }
pub struct FrameLog;  impl FrameLog { pub fn open(dir: &Path) -> io::Result<(FrameLog, Option<String> /* truncation note */)>; pub fn len(&self) -> u64; pub fn get(&self, i: u64) -> io::Result<FrameRecord>; pub fn last(&self) -> Option<FrameRecord>; }
pub fn write_density(dir: &Path, frame: u32, n: usize, rho: &[f32]) -> io::Result<u64 /* bytes */>;   // f16, little-endian, header with n, h, origin
pub fn read_density(dir: &Path, frame: u32) -> io::Result<(DensityHeader, Vec<f32>)>;
pub fn delete_job(root: &Path, id: &JobId) -> io::Result<PathBuf>;         // rename to <id>.deleted-<unix>
pub fn free_deleted(root: &Path) -> io::Result<u64>;                        // the explicit "free X MB"
```

- [ ] **Step 1: Failing tests** (`jobs/src/tests.rs`)
```rust
#[test] fn job_json_round_trips() { /* every field incl. Scene with velocities and masses; userdata::json; unknown keys kept on rewrite (the profile rule) */ }
#[test] fn frames_round_trip_bitwise() { /* 100 records, f64 fields compared with to_bits */ }
#[test] fn a_torn_last_record_is_truncated_with_a_note() { /* cut the file 7 bytes into record 100 → open: len 99, note "frame 100 was incomplete (crash during write); it will be recomputed" */ }
#[test] fn a_flipped_bit_fails_the_record_checksum() { /* FNV-1a 64 per record; a corrupt middle record → Err(InvalidData) naming the frame; never silently skipped */ }
#[test] fn f16_density_round_trip_within_half_precision() { /* |x − f16(x)| ≤ 2^-11 |x| for normal values; 0 and denormals exact or flushed as documented */ }
#[test] fn job_ids_are_unique_and_sortable() { /* two jobs created in the same minute differ; lexical order = creation order */ }
#[test] fn delete_renames_and_free_removes_only_deleted() { /* delete → <id>.deleted-<unix> exists with the same bytes; free_deleted removes it and nothing else; returns its size */ }
```
- [ ] **Step 2:** Run `cargo test -p jobs` → fail (missing crate). **Step 3:** implement. **Step 4:** pass. **Step 5:** commit `feat(jobs): job record, frame log with per-record checksums, f16 density keyframes`.

### Task 1.2: the runner — queue, one tick per frame, resume, ETA **[Opus]**

**Files:** `crates/jobs/src/{runner.rs, eta.rs}`, tests; `crates/sim` unchanged.

**Interfaces:**
```rust
pub struct Runner;  // owns a thread; the app holds a handle
impl Runner {
    pub fn spawn(root: PathBuf, factory: Box<dyn TeacherFactory>, share: Arc<ShareGate>) -> Runner;
    pub fn submit(&self, job: Job) -> JobId;  pub fn pause(&self, id: &JobId); pub fn resume(&self, id: &JobId);
    pub fn cancel(&self, id: &JobId);  pub fn reorder(&self, id: &JobId, to: usize);
    pub fn snapshot(&self) -> Arc<QueueView>;          // republished on change; the UI reads it per frame
    pub fn take_frames(&self, id: &JobId) -> Vec<FrameRecord>;   // new frames since the last call (for live playback)
    pub fn shutdown(&self, wait: Duration) -> bool;    // running jobs → Paused{"interrupted by exit"}
}
pub trait TeacherFactory: Send { fn provider(&mut self, job: &Job) -> Result<Box<dyn sim::ForceProvider>, String>; }   // app/providers implement; jobs never names a solver (rule 5)
pub struct Eta { /* rolling median of the last 15 unit costs; "≈" until 3 units */ }
```
The render loop: `Dynamics::new(scene, provider, dt_fs, seed)` (+ thermostat/field/walls as the live run applies them), then per frame `tick(budget, should_abort)` with the speed fixed at one step per tick, append the frame, every `density_every` frames write the density from the snapshot's display payload (CPU path: values; GPU path: one readback of the texture into f32 — the only readback, off the live path).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn a_dft_render_equals_live_dynamics_bitwise() { /* CpuKs, H₂ at 33³ h 0.25 (the dynamics suite's small CPU setup), 20 steps: frames' positions/energies == a Dynamics driven directly with the same seed (f64 ==) */ }
#[test] fn a_morse_render_equals_live_dynamics_bitwise() { /* sim::test_providers Morse dimer, 2000 steps, CSVR on: bitwise incl. the thermostat RNG */ }
#[test] fn resume_after_a_killed_process_is_bitwise_for_deterministic_teachers() { /* crates/jobs/tests/kill_resume.rs: a child process runs a Morse render; Child::kill at ~frame 700; relaunch → resumes from the last complete record; frames 0..2000 == the uninterrupted run's, bitwise */ }
#[test] fn a_torn_frame_is_truncated_and_recomputed() { /* the 1.1 torn file → resume recomputes that frame; final log == uninterrupted */ }
#[test] fn a_dft_resume_continues_the_same_physics() { /* CpuKs H₂, kill at frame 10, resume cold: frames 11–15 within 1e-6 Ha (E) and 1e-5 Ha/Bohr (F) of the uninterrupted run; the job log says "resumed cold: continues the same physics, not bitwise" */ }
#[test] fn cancel_keeps_every_frame_and_the_job_stays_playable() { /* cancel at 37 → state Cancelled, FrameLog len 37 */ }
#[test] fn pause_resume_and_reorder() { /* two queued Morse jobs; reorder; the second runs first; pause mid-run → no frames appended while paused; resume continues */ }
#[test] fn a_provider_error_fails_the_job_and_keeps_frames() { /* a provider that errs at step 50 → Failed{error}, 49 frames, "Retry from frame 49" possible (resubmits from the last record) */ }
#[test] fn eta_is_a_rolling_median_and_says_approximately_until_three_units() { /* costs [10, 10, 100, 10] s → median 10; "≈" flag for n < 3 */ }
#[test] fn edits_of_the_live_scene_never_reach_a_job() { /* submit, then mutate the caller's Scene → the job's scene unchanged (it is a snapshot) */ }
```
- [ ] Steps 2–5 as above. Commit `feat(jobs): runner — renders drive sim::Dynamics one tick per frame; resume from the last record`.

### Task 1.3: sharing the GPU with the live sandbox **[Opus]**

**Files:** `crates/jobs/src/runner.rs` (`ShareGate`), `crates/app/src/worker/mod.rs` (signals a pending live solve), `crates/providers/src/ks.rs` (no change: the existing `should_abort` granularity is the chunk).

**Behaviour (spec §6.3):** `ShareMode::{Pause, Share, Full}`; in `Share` the runner's `should_abort`/between-step hook yields while `ShareGate::live_pending()` is set and resumes when cleared; in `Pause` jobs on the GPU pause while the live run plays; VRAM: a job whose estimated footprint exceeds the free budget waits with the reason.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn share_mode_yields_between_chunks() { /* a fake GPU job of 100 chunks and a gate toggled by a test thread: no chunk starts while live_pending; every chunk completes once started */ }
#[test] fn pause_mode_pauses_gpu_jobs_while_playing_but_not_cpu_jobs() { }
#[test] fn a_job_waits_for_vram_instead_of_failing() { /* budget 1 GB, job needs 3 GB → state Paused{why contains "waiting for GPU memory"}; budget raised → runs */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn live_latency_during_a_precise_render_is_reported() {
    /* GPU: the app worker plays H₂ 49³ while the runner renders H₂O at Precise; report the live worker's added latency p50/p95 (ms) and the render's slowdown — measured, recorded in references.md "Precision: GPU sharing", not asserted */ }
```
- [ ] Steps 2–5. Commit `feat(jobs): share the GPU with the live run (pause / share / full), VRAM waits`.

### Task 1.4: playback mode, the timeline, the render form, the jobs chip **[Opus]**

**Files:** `crates/app/src/ui/playback/{mod.rs, timeline.rs}`, `ui/render_form.rs`, `ui/{menus.rs, search.rs, toolbar.rs}`, `viewer.rs` (a playback snapshot source next to the live one).

**Behaviour (spec §6.4, §8.4):** Simulation ▸ Precision render… (Ctrl-K "precision render", "render"); the form with solver = DFT at a Quality level (stage 1), length, dt (from the live run), thermostat, density cadence with the disk estimate, the measured-cost estimate (the live scene's last solve × the Quality cost rule, then replaced by the job's own median); the toolbar chip "⚙ n jobs · p % · ETA"; Playback replaces the transport with the timeline: play/pause/step, scrub up to the computed frontier, Follow live, frames/s 1–120 (rule 8), the frontier shading, labelled-frame ticks, the sparkline with E_conserved and error bands; density from the nearest keyframe with the "density from frame k" label; Re-solve this frame; Fork live scene from this frame (`Command::SetScene`); Re-render from this frame…; Exit.

- [ ] **Step 1: Failing tests** (app, no window: `ui::actions` and the replay harness of M6)
```rust
#[test] fn the_form_prefills_from_the_live_run() { /* dt, thermostat, walls, field, the scene at submission */ }
#[test] fn the_form_shows_disk_and_time_estimates() { /* 1000 steps, density every 10 at 141³ → "≈ 560 MB" (100 × 141³ × 2 B + frames), time from the given per-step cost */ }
#[test] fn scrubbing_never_passes_the_frontier() { /* 412 computed of 1000: seek(900) → 411 */ }
#[test] fn follow_live_tracks_new_frames() { }
#[test] fn playback_speed_is_frames_per_second_and_never_touches_dt() { /* rule 8: the job's dt unchanged; the live run's speed unchanged */ }
#[test] fn the_density_label_names_its_keyframe() { /* frame 388 with keyframes every 10 → "density from frame 380 (t = 95.00 fs)" */ }
#[test] fn fork_sends_one_set_scene_with_the_frame_state() { }
#[test] fn editing_in_playback_is_refused_with_the_reason() { /* "this is a recording — Fork to edit" */ }
#[test] fn exit_restores_the_live_view_unchanged() { }
```
- [ ] **Step 2:** a replay script `crates/app/tests/replays/precision_render.replay` (M6 harness): queue a 30-step CPU render of H₂, open playback, scrub, follow live, exit; asserts on the snapshot.
- [ ] Steps 3–5. Commit `feat(app): precision render form, jobs chip, playback mode with timeline`.

### Task 1.5: lifecycle, settings, README — stage 1 ships **[Opus]**

**Files:** `crates/app/src/{main.rs, viewer.rs, settings.rs}`, `README.md`, `CLAUDE.md`, `crates/validate/references.md`.

- Runner spawned in `main` with the user-data root (tests and automation: temp roots, user-data L2); `Shutdown` waits ≤ `EXIT_SAVE_WAIT` for the current frame append; jobs → Paused{"interrupted by exit"}.
- Profile rows (user-data Appendix A, class "run"): `jobs.resume_at_launch` (true, P5), `jobs.share_mode` ("share", P6).
- README: the render form, playback controls, job files and where they live, the honesty lines ("DFT at Precise is still LSDA"), the measured costs from Task 1.3 and the job medians; CLAUDE.md: the `jobs` crate and its tests.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn interrupted_jobs_resume_at_launch_when_enabled() { /* a temp root holding a Paused{"interrupted by exit"} job; launch → Running; with the profile option off → stays Paused */ }
#[test] fn factory_settings_include_the_job_rows() { /* user-data's factory_settings_match_the_shipped_constants extended */ }
#[test] fn automation_never_touches_the_real_jobs_folder() { /* the user-data grep step extended to jobs:: */ }
```
- [ ] Manual checklist (viewer checklist file, a new "Precision render" block): queue a Precise H₂ render, quit mid-run, relaunch, watch it resume and fill in; scrub; fork.
- [ ] Commit `feat(app): jobs survive restarts; README for precision renders` — **stage 1 merge** after review.

---

## Stage 2 — The model manager

### Task 2.1: the Models & jobs page; the Models list and detail **[Opus]**

**Files:** `crates/app/src/ui/manager/{mod.rs, models.rs}`, `ui/{menus.rs, search.rs}`, `ui/dock/model.rs` (the Library section re-uses `models.rs`'s list in compact form — one component, two sizes; spec §8.1).

**Behaviour (spec §8.1–§8.2):** View ▸ Models & jobs (Ctrl-K "models", "model manager", "jobs"); Esc / "Back to the scene"; the list from `Library::entries()` (+ precise entries from stage 5 onward), the ● / ◐ / ○ markers, search, filters, totals; the detail panel built from the user-data `ModelRecord` rows (§12 of that spec) plus "What it is" / "How good" lines from `copy.rs` (Task 2.4).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn every_library_entry_is_listed_once_with_its_marker() { /* 3 entries, one active → markers ●, ○, ○; filters hide by kind */ }
#[test] fn the_library_section_and_the_page_render_the_same_rows() { /* compact vs full: same entries, same order, same actions */ }
#[test] fn use_for_this_scene_is_disabled_with_the_key_diff_reason() { /* "this model is for species H O; the scene is H" */ }
#[test] fn delete_renames_and_needs_two_clicks() { /* user-data §5.5 semantics reused */ }
```
- [ ] Steps 2–5. Commit `feat(app): Models & jobs page — models list and detail`.

### Task 2.2: the Jobs tab **[Opus]**

**Files:** `crates/app/src/ui/manager/jobs.rs`.

**Behaviour (spec §8.4):** the queue table from `Runner::snapshot()`, sub-rows for a job's current stage, Watch (opens playback), Pause, Resume, Cancel… (two clicks, frames kept), ▲▼ reorder, Open log, Delete… (rename), the finished-jobs footer with "free X MB"; the live sampling pass as a read-only row with its own Cancel (P7); toasts on finish.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn actions_map_to_runner_calls() { /* each button → exactly one Runner call */ }
#[test] fn the_live_pass_row_is_read_only_and_cancels_through_the_worker() { }
#[test] fn eta_shows_approximately_until_three_units() { }
#[test] fn a_finished_job_raises_one_toast_with_watch() { }
```
- [ ] Steps 2–5. Commit `feat(app): Jobs tab — queue, progress, ETA, pause/resume/cancel, reorder`.

### Task 2.3: Compare **[Opus]**

**Files:** `crates/app/src/ui/manager/compare.rs`; `crates/sim/src/scan.rs` (new, plain geometry generation, no solver knowledge).

**Behaviour (spec §8.3):** 2–4 models of the same species/charge/multiplicity; scans derived from the scene (pair stretch from 0.75× to 1.7× the current distance, a bend ±30°, rigid approach of two fragments — fractions of the scene's own geometry, never an element radius: rule 1); per model: energy here, the scan minimum, stiffness (a 5-point fit at the minimum), the scan-end energy, cost per step, trusted here; the curves with bands. Models evaluate in µs; a DFT column runs as a job with its cost shown first.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn scans_are_derived_from_the_scene() { /* H₂O: O–H stretch 1.36–3.08 Bohr from R = 1.81; no constant per element anywhere (grep) */ }
#[test] fn compare_values_equal_direct_evaluation() { /* each cell == the model's own evaluate at that geometry (f64 ==) */ }
#[test] fn models_of_different_species_cannot_be_compared() { }
#[test] fn the_dft_column_runs_as_a_job_with_its_cost_shown_first() { }
```
- [ ] Steps 2–5. Commit `feat(app): Compare — models side by side on scene-derived scans`.

### Task 2.4: the chooser, tier badges and the plain-language copy — stage 2 ships **[Opus]**

**Files:** `crates/app/src/ui/manager/copy.rs`, `ui/dock/model.rs`, `ui/toolbar.rs`, `crates/surrogate/src/library/index.rs` (`chosen` per key), `README.md`.

**Behaviour (spec §8.5–§8.6):** "Answering this scene" radios (stage 2: the LSDA model of the key only; stage 5 adds Δ-models), the Estimated button's tier badge, the badges `LSDA`, `LSDA (Precise grid)`, `exact`, `DMC`, `LSDA + Δ(exact)`, `LSDA + Δ(DMC)`, and every tooltip. **The copy of spec §8.6 is pinned verbatim** in `copy.rs` and tested, so a later edit is a reviewed change.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn tier_copy_is_pinned() { /* each badge's sentence == spec §8.6 text */ }
#[test] fn every_manager_number_has_a_tooltip() { /* walk the rendered rows: each numeric cell has a non-empty source/meaning string */ }
#[test] fn the_choice_is_remembered_per_key() { /* choose, switch scene, come back → same choice */ }
#[test] fn kcal_per_mol_is_shown_beside_bond_energy_like_differences() { /* 0.0366 Ha → "23.0 kcal/mol"; conversion only in units (rule 3) */ }
```
- [ ] README: the manager, the badges' meanings. Commit; **stage 2 merge** after review.

---

## Stage 3 — The exact tier (explicitly correlated Gaussians)

### Task 3.1: `ecg` — integrals, antisymmetrisation, SVM, the exact oracles **[Fable, Opus fallback + physics review]**

**Files:** `crates/ecg/src/{lib.rs, gaussian.rs, integrals.rs, boys.rs, spin.rs, solve.rs, svm.rs, tests.rs}`; `crates/validate/tests/precision_ecg.rs`.

**Interfaces:**
```rust
pub struct System { pub nuclei: Vec<([f64; 3], f64 /* Z */)>, pub n_up: usize, pub n_dn: usize }   // N = n_up + n_dn ≤ 3
pub struct Basis { pub fns: Vec<Ecg> }  pub struct Ecg { pub a: Mat /* N×N SPD */, pub s: Vec<[f64; 3]> /* centres */ }
pub fn matrices(sys: &System, b: &Basis) -> (DMat /* H */, DMat /* S */);         // antisymmetrised, spin-projected
pub fn ground_state(h: &DMat, s: &DMat) -> Result<(f64, DVec), EcgError>;          // generalised symmetric eigenproblem, f64; S conditioning checked
pub fn svm(sys: &System, start: Option<Basis>, target_k: usize, rng: &mut Rng, abort: &mut dyn FnMut() -> bool) -> Result<(Basis, f64), EcgError>;
pub fn boys_f0(t: f64) -> f64;  pub fn boys_f1(t: f64) -> f64;
```

- [ ] **Step 1: Failing tests** (unit, `ecg/src/tests.rs`)
```rust
#[test] fn boys_function_matches_its_series_and_asymptote() { /* F0(0) = 1, F0(t) = ½√(π/t) erf(√t) to 1e-14 rel over t ∈ [1e-8, 50] */ }
#[test] fn one_gaussian_hydrogen_matches_the_closed_form() { /* single s Gaussian, exponent α: E(α) = 3α/2 − 2√(2α/π); optimum α = 8/(9π), E = −4/(3π) = −0.4244131815783876 to 1e-13 */ }
#[test] fn matrix_elements_match_independent_routes() { /* diagonal A (uncorrelated product Gaussians): every element vs the textbook one-particle Gaussian formulas (Gaussian product theorem; nuclear attraction and e–e via erf(√p R)/R) to 1e-12; correlated A: vs a 1-D radial quadrature in the relevant relative coordinate after the analytic Gaussian reduction, to 1e-10 rel */ }
#[test] fn antisymmetry_and_spin() { /* triplet H₂ (n_up = 2): Ψ changes sign under exchange; the 3-e doublet's spin projection is an eigenfunction of S² */ }
#[test] fn h2_energy_is_rotation_and_translation_invariant() { /* same basis rotated with the molecule: E equal to 1e-12 */ }
#[test] fn no_element_tables() { /* grep: no Z-indexed constants in crates/ecg */ }
```
- [ ] **Step 2: Failing oracles** (`validate/tests/precision_ecg.rs`, release)
```rust
#[test] fn h2_at_1_4011_matches_pachucki() { /* E − (−1.1744759314002167) ∈ [−1e-10, +1e-6] */ }
#[test] fn h2_well_depth() { /* E(R = 40) − E(1.4011) within 2e-6 of 0.17447593 */ }
#[test] fn h3plus_matches_the_ecg_record() { /* equilateral R = 1.65: E − (−1.34383562502) ∈ [−1e-10, +1e-5] */ }
#[test] fn heh_plus_matches_pachucki() { /* R = 1.463283: E − (−2.978708310771) ∈ [−1e-10, +1e-5] */ }
#[test] fn helium_atom() { /* E − (−2.903724) ∈ [−1e-6, +1e-6] — the digits checked to 7 places */ }
#[test] fn convergence_and_cost_are_recorded() { /* K = 25, 50, 100, 200, 400 for H₂ and H₃⁺: energy error and wall time per energy and per SVM sweep → references.md "Precision: ECG"; sets ECG_K defaults (measured) */ }
```
- [ ] **Step 3:** implement; the SVM candidate draw is spec §14's (re-decided here from the convergence run, the decision recorded in `references.md` and the spec's ledger). **Step 4:** pass. **Step 5:** physics review; commit `feat(ecg): explicitly correlated Gaussians — exact tier for ≤ 3 electrons`.

### Task 3.2: ECG forces and frozen-basis segments; `EcgKs` **[Fable, Opus fallback + physics review]**

**Files:** `crates/ecg/src/forces.rs`; `crates/providers/src/precise/{mod.rs, ecg.rs}`.

**Interfaces:** `pub fn hf_forces(sys, basis, c) -> Vec<[f64; 3]>` (closed-form ∂V_en/∂R_I over Gaussians, F₁); `EcgKs: ForceProvider` with `refresh_policy: (ECG_REFRESH_STEPS, ECG_REFRESH_DIST)`; a refresh is reported through `EvalStats::refit_jump` (booked by the driver) and `surface_epoch()`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn hf_force_is_the_frozen_basis_derivative() { /* H₂, H₃⁺, HeH⁺: analytic vs central FD of E at a frozen basis (δ = 1e-4) ≤ 1e-7 Ha/Bohr; ΣF = 0 to 1e-10 */ }
#[test] fn frozen_basis_error_grows_with_displacement_and_is_recorded() { /* H₂: E_frozen(R + d) − E_reoptimised(R + d) for d = 0.01…0.2 Bohr → sets ECG_REFRESH_DIST (measured) */ }
#[test] fn a_refresh_books_its_jump() { /* refit_jump == E_after − E_before at the same positions (f64 ==); surface_epoch increments */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn ecg_render_conserves_energy_across_refreshes() { /* H₂ 300 K start, NVE, 2000 steps, dt 0.25 fs: |ΔE_conserved| drift ≤ measured × 3 (recorded); continuity at every refresh 1e-10 */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn exact_h2_vibrates_at_the_exact_surface_frequency() { /* small-amplitude NVE: ω from the trajectory vs ω from the ECG curvature at R_e (5-point) within 0.5 %; both reported against ω_e = 4401.21 cm⁻¹ (experiment includes anharmonicity and is not the harmonic BO value — reported, not asserted) */ }
```
- [ ] Steps 2–5; physics review. Commit `feat(providers): EcgKs — exact forces in frozen-basis segments, booked refreshes`.

### Task 3.3: the exact tier in renders; `PreciseKey`; the ECG tripwire — stage 3 ships **[Opus]**

**Files:** `crates/providers/src/precise/key.rs`, `crates/jobs` (ECG warm state in `state.ckpt`), `crates/app/src/ui/render_form.rs`, `crates/validate/tests/tripwire_precise.rs`, README.

**Interfaces:** `PreciseKey { species, charge, multiplicity, method: Method /* Ecg | Dmc | Dft */, tier: String, method_rev: u32, trial: Option<String /* the user-data key text of the orbital solve */>, pseudo: Option<(String /* recipe_hash */, IonicPolicy)>, tau: Option<f64> }` with `canonical()`, `stem()` (FNV-1a 64), `label()`, `diff()` — the user-data §3 rules; `providers::precise::ECG_REV: u32 = 1`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn precise_keys_are_pinned() { /* the ECG key of H₂ (charge 0, lowest): canonical text pinned verbatim; stem pinned after computing it once with an independent FNV-1a (the user-data Task 3 procedure); a DMC key of H₂O with trial/pseudo/tau lines likewise */ }
#[test] fn the_form_refuses_exact_above_three_electrons_with_the_reason() { }
#[test] fn an_ecg_render_resumes_bitwise_from_its_checkpointed_basis() { /* kill/resume child test as Task 1.2, ECG H₂, 200 steps */ }
// validate/tests/tripwire_precise.rs
#[test] fn ecg_rev_tripwire() { /* H₂ at R = 1.4011 with the pinned seed and K: E pinned to 1e-9 (measured once) — "ECG output changed: bump ECG_REV and add a references.md row" */ }
```
- [ ] README: the exact tier (what it is, ≤ 3 electrons, costs measured in 3.1). Commit; **stage 3 merge**.

---

## Stage 4 — The DMC tier (after M7)

### Task 4.0: measurement spike — variance, walker throughput, the go/no-go **[Fable + Opus]**

**Files:** `crates/qmc/src/spike.rs` (`#[cfg(test)]`, deleted or kept as a bench), `crates/qmc-gpu/src/spike.rs`; `references.md` "Precision: DMC cost".

- [ ] **Step 1 (Fable):** a minimal CPU VMC of pseudo-H₂O (M7 tables, CoresOnly; trial orbitals from a Precise-quality `CpuKs` solve; Jastrow e–e cusp only) and of H₂: local-energy variance, autocorrelation time, cost per sweep. **Report:** variance (Ha²), τ_corr, sweeps/s/core.
- [ ] **Step 2 (Opus, `gpu-kernel-engineer`):** a throwaway WGSL kernel: B-spline orbital gather (value, gradient, Laplacian) + determinant ratio for 4096 walkers × 8 electrons; walker-steps/s on the RX 7900 XTX.
- [ ] **Step 3:** compute the cost per geometry at ±1 mHa for H₂, H₃⁺, H₂O and the glycine extrapolation; **replace spec §3.4's [est] rows with the measured ones** (spec edit in the same commit). **Go/no-go:** if pseudo-H₂O at ±1 mHa exceeds 30 min per geometry on the GPU, stop stage 4 and report to the controller before Task 4.1 (Δ-learning on ECG still ships).
- [ ] Commit `docs(spec): measured DMC cost (Task 4.0 spike)`.

### Task 4.1: `qmc` CPU reference — orbitals, Slater–Jastrow, local energy, VMC **[Fable, Opus fallback + physics review]**

**Files:** `crates/qmc/src/{lib.rs, bspline.rs, slater.rs, jastrow.rs, wavefunction.rs, local_energy.rs, vmc.rs, rng.rs, tests.rs}`.

**Interfaces:**
```rust
pub struct Orbitals { /* cubic B-spline coefficients (f64 here) per orbital on the trial grid, origin, h */ }
impl Orbitals { pub fn from_grid(n: usize, h: f64, origin: [f64; 3], values: &[Vec<f32>]) -> Orbitals; pub fn eval(&self, r: [f64; 3], out: &mut [OrbVal /* v, ∇, ∇² */]); }
pub struct Jastrow { /* DTN form: u (↑↑, ↑↓), χ_Z, f_Z; cutoffs per spec §14; cusp constraints as linear relations */ }
pub struct SlaterJastrow { pub orbitals: Orbitals, pub n_up: usize, pub n_dn: usize, pub jastrow: Jastrow, pub nuclei: Vec<Ion> /* point Z or pseudo table */ }
pub fn local_energy(wf: &SlaterJastrow, walker: &Walker) -> LocalEnergy { /* kinetic, e–e, e–n local, nonlocal (4.4), ion–ion */ }
pub fn vmc(wf: &SlaterJastrow, cfg: &VmcConfig, rng: &mut Rng) -> VmcResult;   // Metropolis, electron-by-electron, Sherman–Morrison updates
```

- [ ] **Step 1: Failing tests**
```rust
#[test] fn bspline_reproduces_cubics_exactly_and_converges_as_h4() { /* a cubic polynomial → exact (1e-12); a Gaussian: error ratio h → h/2 ≈ 16 (±20 %) */ }
#[test] fn gradients_and_laplacians_match_finite_differences() { /* orbitals, Jastrow, the full ln Ψ_T: 1e-6 rel */ }
#[test] fn the_jastrow_satisfies_the_kato_cusps() { /* d u↑↓/dr(0) = ½, d u↑↑/dr(0) = ¼; dχ_Z/dr(0) = −Z for point nuclei, 0 for pseudo ions — for random coefficients */ }
#[test] fn sherman_morrison_matches_a_fresh_inverse() { /* 1000 accepted moves → ‖A⁻¹ − inv(A)‖ ≤ 1e-10 in f64 */ }
#[test] fn the_local_energy_of_an_eigenfunction_is_constant() { /* H atom with the exact 1s supplied analytically (an Orbitals adapter) and J = 0: E_L = −0.5 at 1000 random points to 1e-9 */ }
#[test] fn vmc_hydrogen_with_grid_orbitals() { /* the grid 1s of the solver at σ = h = 0.15 + χ cusp term: ⟨E⟩ ≥ −0.5 − 3σ and variance reported */ }
#[test] fn antisymmetry_of_the_determinant() { /* swapping two ↑ electrons flips the sign of Ψ_T */ }
#[test] fn no_element_tables() { }
```
- [ ] Steps 2–5; physics review. Commit `feat(qmc): Slater–Jastrow from the solver's orbitals, local energy, VMC (CPU f64 reference)`.

### Task 4.2: VMC optimisation by the linear method **[Fable, Opus fallback + physics review]**

**Files:** `crates/qmc/src/optimise.rs`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn linear_method_lowers_the_energy_and_the_variance_on_helium() { /* He, grid orbital, Jastrow from zero: final VMC E < initial − 10 mHa and ≥ −2.903724 − 3σ; variance decreases */ }
#[test] fn the_optimum_is_stationary() { /* energy gradient w.r.t. every parameter |g_i| ≤ 3σ_g at the end */ }
#[test] fn optimisation_is_reproducible_for_a_seed() { }
#[test] fn warm_start_from_a_nearby_geometry_needs_fewer_iterations() { /* H₂ at 1.40 → 1.45: iterations recorded; ≤ the cold count */ }
```
- [ ] Steps 2–5; review. Commit `feat(qmc): linear-method optimisation of the Jastrow`.

### Task 4.3: DMC (UNR), population control, blocking; the nodeless oracles **[Fable, Opus fallback + physics review]**

**Files:** `crates/qmc/src/{dmc.rs, blocking.rs}`; `crates/validate/tests/precision_qmc.rs`.

- [ ] **Step 1: Failing tests** (unit)
```rust
#[test] fn blocking_recovers_the_error_of_a_known_ar1_series() { /* AR(1) with φ = 0.9: blocked σ within 10 % of the analytic √((1+φ)/(1−φ)) σ/√n; the plateau detector fires; a too-short series reports "not reliable" */ }
#[test] fn dmc_of_the_exact_wavefunction_has_zero_variance() { /* H atom with the exact 1s: E_L ≡ −0.5, every generation */ }
#[test] fn node_crossings_are_rejected() { /* a 2-e triplet: no accepted move changes the sign of Ψ_T */ }
#[test] fn population_control_bias_is_negligible_at_the_default_population() { /* H₂: E(N_w = 512) vs E(4096) within 2σ; recorded */ }
```
- [ ] **Step 2: Failing oracles** (release, CPU reference; GPU in 4.5)
```rust
#[test] fn hydrogen_atom_dmc() { /* |E + 0.5| ≤ 3σ + the measured box bias (box_bias below) */ }
#[test] fn h2_nodeless_dmc_matches_ecg() { /* R = 1.4011, τ ladder {0.02, 0.01, 0.005} → τ → 0 (linear fit): |E − E_ECG| ≤ 3σ with σ ≤ 0.1 mHa */ }
#[test] fn h3plus_nodeless_dmc_matches_ecg() { /* R = 1.65, same */ }
#[test] fn error_bars_are_honest() { /* 40 seeds, H₂ at the tier τ: fraction within 1σ of E_ECG + measured τ-bias ∈ [0.49, 0.87]; within 2σ ∈ [0.87, 1.0]; χ²/40 ∈ [0.52, 1.67] */ }
#[test] fn box_bias() { /* H atom in 12, 14, 16 Bohr boxes: bias recorded, monotone decreasing */ }
```
- [ ] Steps 3–5; review. Commit `feat(qmc): fixed-node DMC with blocked error bars; nodeless oracles against the exact tier`.

### Task 4.4: pseudopotentials in QMC — locality approximation and size-consistent T-moves; the time step **[Fable, Opus fallback + physics review]**

**Files:** `crates/qmc/src/pseudo.rs` (reads `qm::radial` tables; unfiltered), `dmc.rs`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn nonlocal_quadrature_reproduces_the_kb_operator() { /* for a test function f = R(r) Y_lm, l ≤ 2, the 12-point rule's ⟨V_NL⟩ equals the radial KB integral E_KB ⟨β|f⟩² to 1e-10 (the integrand has degree ≤ 4; the rule is exact through 5); random rotations leave the mean unchanged; an l = 3 projector is refused with "needs a higher-degree quadrature" */ }
#[test] fn vmc_pseudo_atom_matches_the_radial_kb_atom_within_the_jastrow_free_limit() { /* pseudo-O with J = 0 and the radial pseudo-orbitals as orbitals: VMC E = the radial KB energy's non-interacting check — the one-body parts to 3σ */ }
#[test] fn t_moves_and_locality_are_both_reported() { /* pseudo-O atom and H₂O: E_TM and E_LA reported with their difference (the tier's localisation uncertainty); T-moves keep the upper-bound property along the τ ladder: E_TM(τ) ≥ E_TM(τ → 0) − 3σ at every τ */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn time_step_bias_is_measured_and_extrapolated() { /* H₂O, τ ∈ {0.02, 0.01, 0.005, 0.0025}: bias at the tier τ recorded; linear-in-τ fit residuals ≤ 2σ; sets τ (spec §14) if the 0.5 mHa bias target needs a smaller one */ }
#[test] fn size_consistency_of_t_moves() { /* two pseudo-O atoms 30 Bohr apart vs 2 × one atom: within 3σ at the tier τ (the property Casula et al. 2010 restore) */ }
```
- [ ] Steps 2–5; review. Commit `feat(qmc): nonlocal pseudopotentials — locality approximation and size-consistent T-moves`.

### Task 4.5: `qmc-gpu` — walkers in WGSL against the CPU reference **[Opus, gpu-kernel-engineer]**

**Files:** `crates/qmc-gpu/src/{lib.rs, walkers.rs, shaders/{bspline.wgsl, jastrow.wgsl, slater.wgsl, move.wgsl, local_energy.wgsl, nonlocal.wgsl}, tests.rs}`.

**Spec:** spec §5.2.4–§5.2.6 and rule 4: positions f32 relative to the cell centre; orbitals as f32 B-spline coefficients in storage buffers; one invocation group per walker; per-walker f32 energies → workgroup partials → Neumaier f64 on the CPU; determinant inverses refreshed on the CPU in f64 every `DET_REFRESH` sweeps; branching/reconfiguration on the CPU; chunked for sharing (Task 1.3's gate).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn kernels_match_the_cpu_reference_on_fixed_configurations() { /* 256 walker configurations of H₂O: orbital values/gradients/Laplacians, ln Ψ_T, drift, E_L (incl. nonlocal with a fixed rotation) vs qmc f64 within 1e-5 rel (E_L 1e-5 Ha abs) */ }
#[test] fn the_f64_refresh_bounds_the_inverse_drift() { /* 10⁴ sweeps: ‖A⁻¹_gpu − inv(A)‖ ≤ 1e-4 before each refresh, recorded */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn gpu_dmc_matches_cpu_reference() { /* H₂O, same Ψ_T and τ: |ΔE| ≤ 3√(σ_g² + σ_c²) with σ ≤ 0.5 mHa each */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn gpu_throughput_is_reported() { /* walker-steps/s for H₂, H₂O at 1024/4096/16384 walkers → references.md */ }
```
- [ ] Steps 2–5. Commit `feat(qmc-gpu): DMC walkers on the GPU, gated statistically against the CPU reference`.

### Task 4.6: the DMC teacher; the water oracle; the pseudopotential error **[Fable, Opus fallback + physics review]**

**Files:** `crates/providers/src/ks.rs` (`pub(crate) fn occupied_orbitals(&self) -> Option<OrbitalSet>` — a read of `GpuState::orbitals_up/dn` / `KsState`, no state change), `crates/providers/src/precise/dmc.rs`, `crates/validate/tests/{precision_qmc.rs, tripwire_precise.rs}`.

**Interfaces:**
```rust
pub trait EnergyTeacher: Send {   // providers::precise — never in sim
    fn label(&mut self, scene: &Scene, target_sigma: f64, should_abort: &mut dyn FnMut() -> bool) -> Result<Label, ProviderError>;
    fn key(&self) -> PreciseKey;
    fn warm_state(&self) -> Vec<u8>;  fn restore(&mut self, bytes: &[u8]) -> Result<(), String>;
}
pub struct Label { pub energy: f64, pub sigma: f64, pub reliable: bool, pub face_density: Option<f64>, pub cost_s: f64, pub notes: Vec<String> }
pub struct DmcTeacher { /* trial GpuKs at the trial grid (CoresOnly), Orbitals, Jastrow (warm), walkers (continuous, spec §5.2.5), qmc-gpu engine */ }
pub const DMC_REV: u32 = 1;
```

- [ ] **Step 1: Failing tests**
```rust
#[test] fn reading_orbitals_leaves_the_solve_bitwise_unchanged() { /* 20 MD steps of H₂ with occupied_orbitals() read every step: energies and forces == without (f64 ==) — the orbital viewer's §4.4 pin */ }
#[test] fn a_label_reaches_its_target_sigma_and_reports_reliability() { /* H₂: target 0.2 mHa → σ ≤ 0.2 mHa, reliable = plateau detected */ }
#[test] fn continuous_walkers_cost_less_than_cold_labels() { /* H₂ along 10 MD frames: cost per label after the first ≤ 0.5 × the first (recorded) */ }
#[test] fn box_affected_labels_are_flagged() { /* face density above the M5 rule → Label.notes says so; Task 5.1 never trains on it */ }
// validate/tests/precision_qmc.rs (release, GPU)
#[test] fn water_atomization_within_4_kcal() { /* TAE_e = E(O, lowest multiplicity by the S9 scan) + 2E(H) − E(H₂O) at the experimental geometry (pinned from the CCCBDB entry at task time), pseudo-O + AE H, T-moves, τ → 0: |TAE_e − 233.0 kcal/mol| ≤ 4 kcal/mol with σ ≤ 0.3 kcal/mol; the measured value and its decomposition (pseudopotential row below, the 0.38 kcal/mol frozen-core term) recorded in references.md */ }
#[test] #[ignore = "offline: hours"] fn pseudopotential_error_offline() { /* all-electron point-nucleus DMC of O, OH, H₂O (trial orbitals from an h = 0.05 all-electron grid solve + Ma et al. cusp correction) vs the pseudo-O DMC: ΔTAE reported; > 1 kcal/mol → Task 4.7 runs */ }
#[test] fn locality_vs_t_moves_on_water() { /* reported: the localisation uncertainty of the tier */ }
// validate/tests/tripwire_precise.rs
#[test] fn dmc_rev_tripwire() { /* CPU reference, H₂ at 1.4011, fixed seed, 2000 generations: E bitwise-pinned (measured once) — "DMC output changed: bump DMC_REV and add a references.md row" */ }
```
- [ ] Steps 2–5; review. Commit `feat(providers): DmcTeacher — DMC labels from the solver's own orbitals; the water oracle`.

### Task 4.7 (conditional on 4.6): a Hartree–Fock reference atom for the pseudopotential generator **[Fable, Opus fallback + physics review]**

Runs only if `pseudopotential_error_offline` measured > 1 kcal/mol on water's atomization.

**Files:** `crates/qm/src/radial/{atom.rs, recipe.rs}` — a spherical, spin-restricted radial Hartree–Fock atom (exchange by the radial Slater integrals) as a recipe option `xc = HF` (part of `recipe_hash`, a universal switch: Z in, potential out — rule 1); the M7 gates re-run with HF references; the DMC tier's `pseudo` key line carries the hash.

- [ ] **Step 1: Failing tests:** `radial_hf_hydrogen_is_exact` (−0.5 to 1e-10: one electron, no self-interaction in HF); `radial_hf_helium_matches_the_hf_limit` (the Clementi–Roetti-class HF limit pinned from the source at task time); the M7 gate suite under `xc = HF`; `water_atomization_within_4_kcal` re-run with HF potentials, both numbers recorded.
- [ ] Commit `feat(qm): Hartree–Fock reference atom for QMC pseudopotentials (recipe option)`.

### Task 4.8: DMC in the app — R3 renders, Label jobs, DMC progress — stage 4 ships **[Opus]**

**Files:** `crates/jobs` (`JobKind::Label` body: explicit geometries + target σ; R3 render kind: base provider + side labeller at `check_every`), `crates/app/src/ui/{render_form.rs, manager/jobs.rs}`, README.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn an_r3_render_labels_every_kth_frame_and_says_motion_lsda() { /* Morse base + a fake EnergyTeacher: labels at k, 2k…; frames carry label (E, σ); the header text "motion: LSDA" */ }
#[test] fn a_label_job_checkpoints_walkers_and_resumes() { /* kill/resume with a fake teacher whose warm state is bytes: restored exactly */ }
#[test] fn dmc_progress_shows_sigma_shrinking() { /* Jobs row "σ 0.8 → 0.6 mHa" from the teacher's progress */ }
#[test] fn the_dmc_cost_estimate_comes_from_measured_medians() { /* form estimate == the label store's median cost × count */ }
```
- [ ] README: the DMC tier (what it is, what its ± means, the fixed-node statement, measured costs from 4.0/4.5, the water oracle's measured number). Commit; **stage 4 merge**.

---

## Stage 5 — Δ-learned precision models

### Task 5.1: label stores, Δ files, keys, the Rebuild job **[Opus]**

**Files:** `crates/surrogate/src/library/precise.rs`; `crates/jobs` (`JobKind::Rebuild`); tests.

**Interfaces:**
```rust
pub struct LabelStore { pub key: PreciseKey, pub labels: Vec<StoredLabel> }   // <root>/models/precise/<stem>.labels — magic ASLABEL1, JSON header, f64 arrays
pub struct StoredLabel { pub z: Vec<u8>, pub pos: Vec<[f64; 3]>, pub energy: f64, pub sigma: f64, pub forces: Option<Vec<[f64; 3]>> /* ECG */, pub face_density: Option<f64>, pub origin: LabelOrigin /* Pass | Play | Audit | Render */, pub cost_s: f64, pub at: u64 }
pub struct DeltaKey { pub base: LibraryKey, pub precise: PreciseKey }   // canonical = base.canonical() + "--\n" + precise.canonical(); stem = FNV-1a 64
pub struct DeltaFile { pub key: DeltaKey, pub base_energies: Vec<f64> /* per label, the base surface's */, pub model: Option<Model>, pub fit: Option<FitReport>, pub record: DeltaRecord }   // <stem>.delta, magic ASDELTA1
```
- [ ] **Step 1: Failing tests**
```rust
#[test] fn delta_key_includes_base_and_precise_keys() { /* canonical text pinned; changing any base field or any precise field changes the stem */ }
#[test] fn a_dmc_label_never_enters_an_ecg_model() { /* stores are per PreciseKey; appending a label of another key → Err */ }
#[test] fn label_stores_round_trip_and_append_atomically() { }
#[test] fn a_rebuild_evaluates_the_base_at_every_label_and_refits() { /* Morse base A → B: base_energies recomputed (one evaluate each), the Δ refitted; the label store untouched */ }
#[test] fn older_builds_never_see_precise_files() { /* the user-data Library::open on a root with models/precise/ lists only .model entries and never touches precise/ */ }
#[test] fn box_affected_labels_are_kept_but_never_fitted() { /* the S17 rule */ }
```
- [ ] Steps 2–5. Commit `feat(surrogate): precise label stores and derived Δ-model files`.

### Task 5.2: the Δ fit with noisy labels; trust; the learning curve **[Fable, Opus fallback + physics review]**

**Files:** `crates/surrogate/src/fit.rs` (rows carry an optional known σ_i; energy-only rows), `crates/surrogate/src/delta.rs` (trust: χ², LOO).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn zero_sigma_rows_reproduce_the_m5_fit_bitwise() { /* σ_i = 0 everywhere: coefficients, evidence and Σ == the existing fit (f64 ==) — the surrogate suite stays green */ }
#[test] fn predictive_sigma_is_calibrated_on_synthetic_noise() { /* a known smooth Δ(R) on H₃⁺ geometries + Gaussian label noise with known σ_i (0.2–1 mHa): held-out standardised residual rms ∈ [0.85, 1.15] over 2000 points */ }
#[test] fn chi_square_trust_rejects_overconfidence() { /* labels whose stated σ_i are 3× too small → reduced χ² outside the 99 % interval → untrusted with the reason "label errors understated or the model underfits" */ }
#[test] fn chi_square_trust_rejects_overstated_noise() { /* σ_i 3× too large → χ² far below → untrusted ("label errors overstated") */ }
#[test] fn loo_residuals_match_explicit_refits() { /* closed-form r_i/(1 − h_ii) == refit without i, to 1e-10 */ }
#[test] fn energy_only_fit_converges_with_enough_labels() { /* energy-only rows on the Morse dimer: error falls as the learning curve predicts */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn learning_curve_on_the_proxy_delta() { /* Δ = E_DFT(Precise) − E_DFT(Balanced) for H₂O (labels cost seconds): held-out error vs n ∈ {10, 20, 40, 80, 160} at D = 2 and D = 3; recorded; DELTA_D and DELTA_ANSWER_STD re-decided (spec P9, the decision recorded in the spec's ledger) */ }
```
- [ ] Steps 2–5; review. Commit `feat(surrogate): Δ fit with per-label noise, χ² and leave-one-out trust`.

### Task 5.3: `DeltaKs` — live precision; losing precision honestly **[Fable, Opus fallback + physics review]**

**Files:** `crates/providers/src/delta/{mod.rs, policy.rs, tests.rs}`; `crates/sim/src/provider.rs` (`DeltaStats { value, std, answered, tier }` in `EvalStats`); `crates/validate/tests/precision_delta.rs`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn delta_forces_are_the_gradient_of_delta_energy() { /* FD vs analytic 1e-8 rel; ΣF = 0, Σr×F = 0 to 1e-12 */ }
#[test] fn losing_precision_books_a_surface_jump_and_says_so() { /* drive the geometry outside the Δ envelope: the step answers with the base, refit_jump == −E_Δ at those positions, the banner text pinned, history marks the step; coming back books +E_Δ; E_conserved continuous to 1e-10 */ }
#[test] fn a_delta_refit_is_booked_like_a_surrogate_refit() { }
#[test] fn the_base_keeps_learning_from_dft_unchanged() { /* as_trainable() is the base's Trainable; the base's samples/fits identical with and without a Δ on top */ }
#[test] fn audits_judge_with_combined_error_bars() { /* gap 2.9 × √(σ_stat² + σ_Δ²) passes, 3.1 × fails and untrusts */ }
// validate/tests/precision_delta.rs (release)
#[test] fn h3plus_from_ecg_labels() { /* Δ(ECG) on Accurate-LSDA H₃⁺ with active learning (5.4) or a fixed design: |E − E_ECG| ≤ 3σ_Δ on ≥ 95 % of 100 held-out geometries; bend/stretch frequencies on the corrected surface within 2 % of the ECG surface's */ }
#[test] fn live_h2_on_the_corrected_surface_conserves_energy() { /* LSDA surrogate + Δ(ECG), 10 ps NVE: drift < 1e-6 Ha/ps (the M5 surrogate bound) */ }
```
- [ ] Steps 2–5; review. Commit `feat(providers): DeltaKs — precision played live, honest fallback to the base`.

### Task 5.4: active learning against the teacher's cost; difference labels **[Fable, Opus fallback + physics review]**

**Files:** `crates/providers/src/delta/select.rs`; `crates/qmc/src/dmc.rs` (correlated pair sampling with the space-warp transformation, Filippi & Umrigar 2000).

- [ ] **Step 1: Failing tests**
```rust
#[test] fn greedy_selection_maximises_the_posterior_variance_reduction() { /* on a synthetic pool: the greedy batch's log det reduction ≥ 0.63 × the brute-force optimum (submodularity bound) for batch 5 of 30 */ }
#[test] fn the_pool_comes_from_the_base_surface_and_skips_box_affected_geometries() { }
#[test] fn the_budget_is_shown_before_anything_runs() { /* LabelSpec → "≈ n solves × median cost" */ }
#[test] fn active_learning_needs_fewer_labels_than_random() { /* proxy Δ on H₂O: labels to reach σ_Δ ≤ 1 mHa on the pool, greedy vs random (5 seeds) — recorded; greedy ≤ random */ }
#[test] #[cfg_attr(debug_assertions, ignore)] fn difference_labels_are_measured() { /* H₂ pairs (R, R + 0.05): σ of the correlated difference vs two independent labels at equal cost; adopted as fit rows only if ≥ 4× cheaper per unit variance (recorded; spec §7.3 updated either way) */ }
```
- [ ] Steps 2–5; review. Commit `feat(providers): cost-aware active learning for precise labels`.

### Task 5.5: R2 renders — the learned surface refined on the fly **[Opus; Fable review of the bookkeeping]**

**Files:** `crates/jobs` (R2: `DeltaKs` over the base + an `EnergyTeacher`; pause the trajectory for a label when σ_Δ exceeds the answer bound or every `check_every` frames; refit; booked jump), `crates/app/src/ui/{render_form.rs, playback/timeline.rs}`.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn r2_labels_when_uncertain_and_every_kth_frame() { /* Morse base + analytic "precise" teacher (Morse + a smooth Δ): labels exactly at uncertain frames and multiples of k */ }
#[test] fn r2_refits_are_booked_and_e_conserved_is_continuous() { }
#[test] fn r2_frames_carry_model_sigma_and_labels() { /* every frame σ_model; labelled frames (E, σ_stat); the timeline marks them */ }
#[test] fn r2_converges_to_the_precise_surface_trajectory() { /* with the analytic precise teacher: after the labels, positions track a Dynamics on the precise surface within 1e-3 Bohr over 500 steps */ }
```
- [ ] Steps 2–5; Fable review. Commit `feat(jobs): R2 precision renders on a learned, refined surface`.

### Task 5.6: DMC Δ for water — the calibration oracles **[Fable, Opus fallback + physics review]**

**Files:** `crates/validate/tests/precision_delta.rs`, `references.md`.

- [ ] **Step 1: Failing oracles** (release, GPU; hours — `#[ignore]` by default with the run of record in `references.md`, like the long M4 tests)
```rust
#[test] fn delta_uncertainty_is_calibrated_on_dmc() { /* H₂O, Accurate base, DMC labels by active learning to σ_Δ ≤ 1 mHa on a 300 K pool; 30 held-out DMC labels: standardised residual rms ∈ [0.7, 1.4] */ }
#[test] fn water_live_reproduces_dmc_atomization() { /* the live LSDA+Δ(DMC) TAE at the oracle geometry within 3σ_Δ of water_atomization_within_4_kcal's DMC value */ }
#[test] fn water_corrected_geometry_is_reported() { /* relax (M6 FIRE) on the corrected surface: r(O–H), ∠HOH, reported with σ against experiment — reported, not asserted (classical nuclei, fixed node) */ }
```
- [ ] Steps 2–5; review. Commit `test(validate): DMC Δ-model of water — calibration and the atomization check`.

### Task 5.7: precision models in the UI — stage 5 ships **[Opus]**

**Files:** `crates/app/src/ui/{dock/model.rs, manager/{models.rs, compare.rs}}`, `worker/*` (Learn while I play → Label jobs), README, CLAUDE.md.

**Behaviour (spec §8.2, §8.5):** precise entries (label stores, Δ-models) in the Models list and detail; the chooser lists Δ-models; switching the answering model books the surface jump and logs it; the This-model group gains the Δ rows; **Label more…** (the selection of 5.4 with the budget first); **Learn while I play** (queues Label jobs for geometries where precision was lost, never blocking); Compare shows Δ bands and labels with error bars; the solver chooser's Estimated badge shows the tier.

- [ ] **Step 1: Failing tests**
```rust
#[test] fn switching_the_answering_model_books_and_logs_the_jump() { /* "model switched: LSDA → LSDA + Δ(DMC) — the energy at these positions changed by …" */ }
#[test] fn learn_while_i_play_queues_labels_without_blocking() { /* the live run never waits on a Label job */ }
#[test] fn the_delta_rows_say_where_numbers_come_from() { /* tooltips non-empty; σ rows say "statistical / fit, not the method's own error" */ }
#[test] fn deleting_a_label_store_names_its_derived_models() { }
```
- [ ] README: precision models end to end (label → Δ → play live; what the badge and ± mean; measured water numbers); CLAUDE.md: the new crates, suites and commands. Commit; **stage 5 merge**.

---

## Self-review

- **Spec coverage:** §6 → Tasks 1.1–1.5, 4.8, 5.5; §7 → 5.1–5.7; §8 → 2.1–2.4, 5.7; §5.1 → 3.1–3.3; §5.2 → 4.0–4.7; §9's every row has a named test (ECG oracles 3.1/3.2, QMC 4.3–4.6, Δ 5.2/5.3/5.6, render 1.2/3.3); §10's stages are the plan's stages; §13's provisional rulings are each exercised by a test (P1 2.1, P2 4.3, P3 by absence, P4 5.5, P5 1.5, P6 1.3, P7 2.2, P8 5.1, P9 5.2, P10 4.4, P11 4.6, P12 4.6).
- **Every [est] has a measuring task:** spec §3.4 → 3.1 (ECG), 4.0 (DMC; go/no-go), 1.3 (sharing), 4.5 (throughput); every "measured" bound names its task.
- **No placeholder values:** pinned numbers are the verified literature values of spec §2 or are "measured first" with the procedure stated (stems computed once independently, tripwire energies, bias rows).
- **Consistency:** `PreciseKey` (3.3) is used by 4.6 and 5.1; `EnergyTeacher` (4.6) by 4.8, 5.4, 5.5; `DeltaKs` (5.3) by 5.5 and 5.7; `Runner`/`TeacherFactory` (1.2) by every later job kind; `refit_jump` booking is the M5 mechanism throughout.
- **Tags:** physics that can be plausibly wrong is Fable with a mandatory review on fallback; plumbing, UI and kernels are Opus.
