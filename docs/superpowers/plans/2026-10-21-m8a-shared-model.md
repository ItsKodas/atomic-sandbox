# M8a — The Shared Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One growing machine-learned model per set of solver settings that learns from everything trained, transfers to scenes it never saw whole, shows what it knows, and offers "Train this segment" — the foundation the Sandbox solver (M8b) runs on.

**Architecture:** The linear ACE surrogate (`crates/surrogate`) gains a *collection*: an on-disk, append-only sample store keyed by the solver surface (functional, h, σ, tolerance, teacher, ionic tag), with the incremental normal equations (`Accumulator`) rebuilt on species growth. The provider (`providers::surrogate::SurrogateKs`) attaches a collection instead of a per-scene store; the class-wise trust rule is replaced by a verdict calibrated on a persisted audit history. Display-side, the worker ships a `KnowledgeMap` for a new dock tab, and the app cuts capped training segments.

**Tech Stack:** Rust workspace (surrogate, providers, sim, app, validate), wgpu GPU teacher, egui UI.

**Spec:** `docs/superpowers/specs/2026-10-20-m8-shared-model-and-sandbox-design.md` (§1–4, §6, §7; M8b and the PBE milestone are separate plans). Interface map used to write this plan: the sections below cite file:line as of main `62f5d4a` (2026-10-20).

## Global Constraints

- CLAUDE.md rule 1: per-element data = Z and isotope mass only; the **one** exception M8a uses is the molecule builder's tables (`crates/app/src/builder/chem.rs`: `covalent_radius_bohr`, `bond_length`, `usual_valence`) to **place capping hydrogens** in training segments (user sign-off 2026-10-20, spec §4.5) — never in forces, energies, the fit or the trust rules.
- Rule 3: atomic units inside `surrogate`, `providers`, `sim`; Å/eV only at the UI.
- Rule 5: `sim` never knows what computes forces; `KnowledgeMap` and calibration types that cross into `sim` are plain data.
- Rule 6: every task touching trust, fitting, collections or sampling passes `cargo test -p surrogate --release`, `cargo test -p providers --release` and, at the task's end, `cargo test -p validate --release --test surrogate` (~15 min, GPU). Never loosen a tolerance.
- Rule 7: limitations documented (README, `crates/validate/references.md`).
- One heavy GPU job at a time on the machine (other sessions share it); builds may run at low priority.
- Collections are **neutral only** (charge 0); box-affected samples (quarantine `Q_BOX`) never enter a collection; samples whose scene forced a multiplicity never enter a collection (spec §4.2, physics review may relax the spin rule).
- Surface key = functional + h + σ + SCF tolerance + teacher kind + ionic tag; **not** grid n, charge or multiplicity (spec §4.2). The functional is `"lsda"` until the PBE milestone.
- Existing `.model` sidecars keep loading as per-scene ("specialised") models exactly as today; importing into a collection is explicit and reports what it refused.
- The user hand-tests visible work; implementers build (`cargo build -p app --release`; rename a locked `target\release\app.exe`, never kill the user's app), launch, list what to test, and continue (memory `keep-working-note-tests`).
- Model assignment (CLAUDE.md): tasks tagged **[Fable]** go to `claude-fable-5-1` / the `solver-implementer` agent with a `physics-reviewer` pass; **[Opus]** to `claude-opus-5-5`.

## Review Focus

1. **A collection folder left mid-write by a crash** (app killed during an append) → the next load keeps every complete sample, drops the torn tail, and says how many were dropped — never refuses the whole collection. *Test in Task 3.*
2. **Two app windows on the same collection** → the second opens it read-only (a lock file) and says so; it never interleaves writes. *Test in Task 3.*
3. **A scene with an element the collection has never seen** → Estimated falls back to DFT and teaches as today; the refit after the first sample of the new element rebuilds the basis and succeeds; the knowledge map shows the element grey until trusted. *Test in Task 4.*
4. **Importing a charged, forced-spin or other-surface `.model`** → nothing is imported, the reason is named ("charge +1: collections are neutral until M9"), the file still opens as a specialised model. *Test in Task 3.*
5. **Train this segment with an empty selection, a lone atom, or a segment too big for the GPU budget** → a clear refusal naming the reason, no scene change. *Test in Task 10.*

---

### Task 0: Branch and pre-flight [Opus]

**Files:** none (git, ledger).

- [ ] **Step 1:** Wait until `m7-leftovers` and the training-robustness branch (`docs/superpowers/notes/2026-10-13-m5-user-requests.md` "Training pass robustness") are merged into `origin/main` — both touch `providers::surrogate` / `sampling`. Then: `git fetch && git worktree add <dir> -b m8a origin/main`.
- [ ] **Step 2:** Run `cargo test -p surrogate --release && cargo test -p providers --release` on the fresh branch. Expected: all pass (the baseline every task compares against).

### Task 1: Audit history and the trust baseline [Fable]

The calibration needs, per audit, what the model predicted *and* what the teacher measured; today only the last `AuditRecord` (sim/src/provider.rs:1196) is kept and the predicted std is not recorded (`judge_audit`, providers/src/surrogate/audit.rs:563, predicts with `with_variance = false`).

**Files:**
- Create: `crates/surrogate/src/history.rs` (the record type and its ring buffer, sidecar-serialisable)
- Modify: `crates/providers/src/surrogate/audit.rs` (record every judged audit), `crates/surrogate/src/sidecar.rs` (persist the history; new magic `ASMODEL7`, V5/V6 still read)
- Test: `crates/surrogate/src/history.rs` (unit), `crates/validate/tests/surrogate.rs` (an `#[ignore]` measurement)

**Interfaces:**
- Produces: `pub struct AuditPoint { pub classes: Vec<u8> /* per atom: 0 isolated, 1 pair, 2 many-body (trust.rs atom_class) */, pub pred_force_std: Vec<f64> /* per atom, Ha/Bohr */, pub pred_energy_std: f64 /* Ha */, pub err_force: Vec<f64> /* |F_teacher − F_model| per atom */, pub err_energy: f64, pub n_atoms: usize, pub model_gen: u64 }`; `pub struct AuditHistory { /* ring, capacity HISTORY_CAP = 4096 */ }` with `push(AuditPoint)`, `iter()`, `len()`; `Sidecar::audit_history: AuditHistory`.

- [ ] **Step 1: Failing tests** — `history_ring_keeps_the_newest` (push HISTORY_CAP + 10 points; `len() == HISTORY_CAP`, the oldest 10 gone) and `sidecar_round_trips_the_history` (write V7, read back equal; a V5 file reads with an empty history).
- [ ] **Step 2:** Run `cargo test -p surrogate --release history` → FAIL (module missing).
- [ ] **Step 3:** Implement `history.rs`; record an `AuditPoint` in `judge_audit` (predict **with** variance there; the extra cost is one variance evaluation per audit, already budgeted by `variance_cadence`); persist in the sidecar.
- [ ] **Step 4:** Run the unit tests → PASS; `cargo test -p providers --release` → PASS.
- [ ] **Step 5: Baseline measurement** — add `#[ignore] fn trust_baseline_report()` to `validate/tests/surrogate.rs`: run `stuck_escape_repro` and `active_learning_run_converges`' scenario with audits every 50 steps, and report per class: audits, the share of **missed large errors** (an audited atom whose `err_force` exceeds the fit's 99th-percentile in-domain LOO force error while the verdict was *trusted*), and how often the class verdict flipped between consecutive steps. Record the numbers in `references.md` under a new heading "M8a trust baseline".
- [ ] **Step 6:** Commit `feat(surrogate): the audit history (predicted vs measured, persisted); the M8a trust baseline`.

### Task 2: The calibrated class verdict [Fable]

Replace rule 4b's knife-edge (trust.rs `class_refusal` :82, `compute_class_loo_full` :334) with a calibrated verdict + hysteresis (spec §4.1). The method is the implementer's, justified in `references.md` and signed off by the physics reviewer; the plan fixes the interface and the acceptance.

**Files:**
- Create: `crates/providers/src/surrogate/calibrate.rs`
- Modify: `crates/providers/src/surrogate/trust.rs` (rule 4b uses the calibrated verdict), `crates/providers/src/surrogate/mod.rs` (module doc's decision list, "Class-wise trust"), `crates/sim/src/provider.rs` (`ClassLoo` gains `calibration: f64` and `trusted_since: Option<u64>`)
- Test: `crates/providers/src/surrogate/tests.rs`, `crates/validate/tests/surrogate.rs`

**Interfaces:**
- Consumes: `AuditHistory` (Task 1), `ClassLoo` per class.
- Produces: `pub struct Calibration { pub per_class: [ClassCalibration; 2] }`, `pub struct ClassCalibration { pub factor: f64 /* measured / predicted, ≥ 1 */, pub audits: usize }`, `pub fn calibrate(history: &AuditHistory, loo: &[ClassLoo]) -> Calibration`, and `fn class_verdict(&self, class: usize) -> ClassVerdict` with `pub enum ClassVerdict { Trusted, Untrusted { reason: String } }`; the verdict carries hysteresis state in `SurrogateKs` (enter and leave thresholds differ; the implementer sets them).

- [ ] **Step 1: Failing tests** — in providers tests: `class_verdict_does_not_flip_at_the_cutoff` (an analytic teacher, a pair oscillating across `0.999·r_cut … 1.001·r_cut` for 200 steps: the verdict changes at most once) and `calibration_widens_when_audits_disagree` (a history whose measured errors are 3× the predicted ones gives `factor ≥ 2.5` and makes a previously trusted class untrusted). *As built (the physics review's M6, accepted): the cutoff test became `a_class_refused_for_one_outlier_earns_the_calibrated_verdict_and_holds` — a pair crossing r_cut is the shell check's structural case (f_c → 0 while the teacher's interaction does not) and under the old rule the class verdict there trips once and stays, so the brief's test would not have been RED; the knife-edge the class verdict owns is the lone outlier.*
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement `calibrate` and `class_verdict`; rule 4b consults it; the note text names the calibration ("pair class: calibrated error 2.1× the bound").
- [ ] **Step 4:** Unit tests PASS; `cargo test -p providers --release` PASS.
- [ ] **Step 5: Acceptance** — run `trust_baseline_report` again: missed large errors ≤ the Task 1 baseline per class, flips per 1000 steps strictly fewer; `cargo test -p validate --release --test surrogate` PASS (incl. `the_stuck_escape_repro_is_answered_honestly`). Record the numbers in `references.md` "M8a calibrated verdict".
- [ ] **Step 6:** Physics review (`physics-reviewer`), fix round(s), then commit `feat(providers): the calibrated, hysteretic class verdict replaces rule 4b's knife-edge`.

### Task 3: The collection on disk [Fable for the sharing rules, Opus for storage]

**Files:**
- Create: `crates/surrogate/src/collection.rs`
- Modify: `crates/surrogate/src/model.rs` (`SurfaceKey`, `Provenance::surface_key()`), `crates/surrogate/src/lib.rs`
- Test: `crates/surrogate/src/collection.rs` (unit)

**Interfaces:**
- Produces:
  - `pub struct SurfaceKey { pub functional: String, pub h: f64, pub sigma: f64, pub tol: f64, pub teacher: String, pub ionic: String }` with `fn slug(&self) -> String` (a filesystem-safe folder name, stable across runs) and `Provenance::surface_key(&self) -> SurfaceKey` (functional `"lsda"`).
  - `pub struct CollectionSample { pub sample: Sample, pub grid_n: usize, pub multiplicity_forced: bool, pub scene: String /* the scene's name, for the knowledge map */ }`
  - `pub struct Collection { pub key: SurfaceKey, pub samples: Vec<CollectionSample>, pub references: BTreeMap<u8, f64>, pub history: AuditHistory, pub read_only: bool }`
  - `impl Collection { pub fn open(dir: &Path, key: &SurfaceKey) -> Result<(Collection, OpenReport), String>; pub fn admit(&self, s: &Sample, prov: &Provenance, quarantine: u8) -> Result<(), String>; pub fn append(&mut self, s: CollectionSample) -> Result<(), String>; pub fn import_sidecar(&mut self, path: &Path) -> Result<ImportReport, String>; pub fn write_readme(&self) -> Result<(), String> }`
  - `pub struct OpenReport { pub samples: usize, pub dropped_tail: usize, pub locked_by_other: bool }`, `pub struct ImportReport { pub imported: usize, pub refused: Vec<(usize, String)> }`
  - Layout under `models/collections/<slug>/`: `key.txt` (the key, human-readable), `samples.bin` (magic `ASCOLL1`, append-only length-prefixed records, each with a checksum), `references.txt`, `history.bin`, `README.txt` (what it knows: species, pairs, sample counts), `.lock` (process id).
- Rules (`admit`): charge ≠ 0 → `Err("charge {q}: collections are neutral until M9")`; `quarantine == Q_BOX` → `Err("box-affected sample")`; `multiplicity_forced` → `Err("the scene forced multiplicity {m}")`; `prov.surface_key() != self.key` → `Err` naming the differing field.

- [ ] **Step 1: Failing tests** — `admit_refuses_charged_box_and_forced_spin_samples`; `append_survives_a_torn_tail` (append 10, truncate the file mid-record, reopen: 9 samples, `dropped_tail == 1`); `a_second_open_is_read_only` (lock held by a live pid → `read_only` and `locked_by_other`); `import_reports_what_it_refused` (a V5 sidecar with charge +1 → `imported == 0`, one refusal naming the charge); `surface_key_ignores_grid_n_charge_and_multiplicity`.
- [ ] **Step 2:** Run `cargo test -p surrogate --release collection` → FAIL.
- [ ] **Step 3:** Implement. Appends are flushed per record (crash-safe); the lock is released on drop.
- [ ] **Step 4:** PASS; commit `feat(surrogate): collections — one append-only sample store per solver surface`.

### Task 4: The provider and the worker use the collection [Fable for provider, Opus for worker/UI]

**Files:**
- Modify: `crates/providers/src/surrogate/{mod.rs,store.rs,trust.rs}` (a collection-backed store), `crates/sim/src/provider.rs` (`Trainable::attach_collection`, `ModelSource`), `crates/app/src/worker/{model.rs,commands.rs,autosave.rs}`, `crates/app/src/ui/dock/model.rs`, `crates/app/src/scene.rs` (`collections_dir()`)
- Test: `crates/providers/src/surrogate/tests.rs`, `crates/app/src/worker/tests.rs`

**Interfaces:**
- Consumes: `Collection`, `SurfaceKey` (Task 3).
- Produces: `Trainable::attach_collection(&mut self, c: Collection) -> Result<(), String>`; `pub enum ModelSource { Collection { slug: String, samples: usize, read_only: bool }, File(PathBuf), None }` in `ModelStatus` (worker/model.rs:23) and on the Model tab; `Command::UseModel(ModelChoice)` with `pub enum ModelChoice { Collection, File(PathBuf) }`; `scene::collections_dir() -> Option<PathBuf>` (`models/collections`, created on first use).
- Behaviour: with a collection attached, every kept teacher answer is `admit`ted and `append`ed (refusals keep today's per-scene path for that sample and add a note); `push_sample`'s provenance check becomes the surface-key check; species growth rebuilds the accumulator from the collection (today's `species_dirty` path); `EnsureModel` attaches the collection for the scene's surface key by default (the user's "one big model"), and `LoadModel` of a file switches to it as a specialised model. The autosave keeps writing the per-scene sidecar only in File mode.

- [ ] **Step 1: Failing tests** — providers: `collection_mode_learns_across_scenes` (analytic teacher: teach H₂, then a scene with H₂ + He: the He samples append, the basis grows to {1, 2}, the refit succeeds, the H₂ samples are still used); `a_charged_scene_keeps_a_per_scene_store` (charge +1 → nothing appended, a note). Worker: `ensure_model_attaches_the_scene_surfaces_collection` (two worker runs on different scenes with the same h/σ/ionic share one `models/collections/<slug>` under a temp data dir).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** PASS + `cargo test -p providers --release` + `cargo test -p app --release --bin app -- worker::` PASS.
- [ ] **Step 5:** Model tab: a line "model: shared — <slug> · N samples" (or "file: <name>"), a switch, and "Import a model file…" (calls `import_sidecar`, shows the `ImportReport`). Build, launch; **user test list:** the Model tab line; train H₂ then open water — the same collection grows.
- [ ] **Step 6:** `cargo test -p validate --release --test surrogate` PASS; commit `feat: Estimated learns into the shared collection of its solver surface`.

### Task 5: Model capacity — four-body terms and an interactive refit [Fable]

**Files:**
- Modify: `crates/surrogate/src/{descriptor.rs,basis.rs,fit.rs}` (`Basis.body4`, `body4_terms`, the spectral-Σ / warm-evidence fit path), `crates/surrogate/src/{loo.rs,novelty.rs,grade.rs,envelope.rs}` (new columns flow through)
- Test: `crates/surrogate/src/descriptor.rs`, `crates/surrogate/src/fit.rs`, `crates/validate/tests/surrogate_perf.rs`

**Interfaces:**
- Produces: `Basis { …, body4: bool }` (default **true** for collections, false for per-scene models so existing sidecars keep their basis), `Basis::body4_terms(&self) -> &[Body4Term]`; `fit` unchanged in signature.

- [ ] **Step 1: Failing tests** — `body4_features_are_rotation_invariant` (a random 6-atom cluster rotated by a random rotation: features equal to 1e-10 relative); `body4_forces_match_finite_differences` (central differences, step 1e-5 Bohr, agreement 1e-6 relative); `refit_is_interactive_at_ten_thousand_samples` (`#[ignore]` measurement in `surrogate_perf.rs`, synthetic CPU samples: a refit after one added sample < 1 s; record the time in `references.md`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (the four-body contraction of the real-harmonic A basis under the degree cap; the spectral Σ / warm-evidence changes keep the Bayesian evidence step from refactoring the whole system on every add). **Step 4:** PASS; `cargo test -p surrogate -p providers --release` PASS; the surrogate suite PASS.
- [ ] **Step 5:** Physics review; commit `feat(surrogate): four-body invariants and an interactive refit for growing collections`.

### Task 6: The transfer test and the reach comparison [Fable]

**Files:**
- Create: `crates/validate/tests/transfer.rs`
- Modify: `crates/surrogate/src/descriptor.rs` (`COLLECTION_R_CUT` const, chosen in Step 4), `crates/validate/references.md`

**Interfaces:**
- Consumes: Tasks 3–5; validate helpers `common::Locked`, `student`, `sample_and_fit`, `model_vs_dft` (validate/tests/surrogate.rs:293/324/375) — move the three into `tests/common/surrogate.rs` so both suites use them.

- [ ] **Step 1: The test** — `fragments_transfer_to_propane_and_the_water_trimer`, for r_cut ∈ {6.0, 9.0} Bohr: a collection taught only on CH₄ and C₂H₆ (default Accurate grid, `ionic accepted`, the Standard sampling preset) predicts 20 held-out geometries of propane (perturbation σ = 0.15 Bohr from the shipped `scenes/c3h8.scene`) with force RMSE and |ΔE|/atom **≤ 2×** the collection's own LOO RMSE on the atoms it calls trusted, and refuses (rule 4b/5) the rest; likewise H₂O + (H₂O)₂ → a water trimer. Print both reaches' errors, evaluation times and teacher time.
- [ ] **Step 2:** Run `cargo test -p validate --release --test transfer -- --nocapture` (GPU, long: run alone). Expected: PASS at at least one reach.
- [ ] **Step 3:** Record the table in `references.md` "M8a transfer and reach"; set `COLLECTION_R_CUT` to the reach that passes with the better accuracy-per-cost (if both pass and 9 Bohr costs > 3× evaluation time, keep 6; the plan's rule, revisable by the physics reviewer).
- [ ] **Step 4:** Commit `test(validate): fragments transfer to larger molecules; the collection's reach chosen from it`.

### Task 7: Specialised models [Opus]

**Files:** Modify `crates/surrogate/src/collection.rs` (`specialise`), `crates/app/src/ui/dock/model.rs`, `crates/app/src/ui/actions.rs`.

**Interfaces:** `Collection::specialise(&self, species: &[u8]) -> SampleSet` (the samples whose atoms are all in `species`, with their references) → written with `sidecar::write_samples` then fitted on load as today; `Action::SaveSpecialised(PathBuf)`.

- [ ] **Step 1: Failing test** — `specialise_keeps_only_samples_within_the_species` (a collection with H, C, O samples; `specialise(&[1, 6])` has no O atom in any sample).
- [ ] **Step 2–4:** FAIL → implement → PASS. Model tab: "Save a specialised model…" (scene's species preselected). Build, launch; user test list: save a hydrocarbon-only model from a mixed collection and load it.
- [ ] **Step 5:** Commit `feat: save a specialised model from the shared collection`.

### Task 8: The mockup page [Opus] — user approval gate for Task 9's UI

**Files:** Create `docs/superpowers/notes/2026-10-21-m8-mockup.html` (self-contained HTML in the style of `docs/superpowers/notes/2026-10-14-ui-mockup.html`).

- [ ] **Step 1:** Mock, with realistic numbers: (a) the knowledge map — elements as nodes sized by samples, pairs as edges coloured trusted / thin / unseen, a pair's distance histogram with the "asked now" marker, an element's coordination and angle coverage; (b) the Model tab with the collection line and the specialise/import buttons; (c) the 3D overlay tint and its legend; (d) Train this segment — the selection, the segment preview with caps, the confirmation, the "back to the scene" chip; (e) for M8b, the Sandbox solver panel and the frozen-atom banner.
- [ ] **Step 2:** Commit, publish/show to the user, record their verdict in the ledger. Do not block: continue with Task 9's data part and Task 10 while waiting; Task 9's UI steps wait for the approval.

### Task 9: The knowledge map [Opus; Fable review of the confidence rule]

**Files:**
- Create: `crates/sim/src/knowledge.rs` (plain data), `crates/providers/src/surrogate/knowledge.rs` (builder), `crates/app/src/ui/dock/knowledge.rs` (tab), `crates/app/src/ui/knowledge_window.rs` (pop-out)
- Modify: `crates/sim/src/lib.rs`, `crates/app/src/ui/dock/mod.rs` (`DockTab::Knowledge`, right dock), `crates/app/src/worker/mod.rs` (built per `model_gen`, shipped in the snapshot like `TrainingViews`), `crates/app/src/viewer.rs` + `crates/app/src/gfx.rs` (per-atom tint overlay)

**Interfaces:**
- Produces (sim): `pub struct KnowledgeMap { pub elements: Vec<ElementKnowledge>, pub pairs: Vec<PairKnowledge>, pub slug: String, pub samples: usize }`; `pub struct ElementKnowledge { pub z: u8, pub atoms_seen: usize, pub coordination: Vec<u32> /* histogram, index = neighbour count within r_cut */, pub angles: Vec<u32> /* 5° bins of bond-pair angles, pairs within the reference bonded distance — the pair histogram's first peak */ }`; `pub struct PairKnowledge { pub z: (u8, u8), pub samples: usize, pub hist: PairHistogram /* reuse sim::activity, BIN 0.1 Bohr */, pub confidence: Confidence }`; `pub enum Confidence { Trusted, Thin, Unseen }`; per-atom `pub fn atom_coverage(map: &KnowledgeMap, z: &[u8], pos: &[[f64;3]]) -> Vec<f32>` (0 unknown … 1 well known), used by the overlay.
- Confidence rule: `Trusted` iff the pair has ≥ Q′ samples (the fit's `q_prime`) **and** the calibrated class verdict (Task 2) of the classes its atoms fall in is trusted; `Thin` if it has samples but fails that; `Unseen` with none.

- [ ] **Step 1: Failing tests** — `knowledge_counts_pairs_and_marks_unseen` (a synthetic collection of H₂ and CH₄ samples: pairs H–H, C–H present with counts; C–C `Unseen`; C coordination histogram peaks at 4); `atom_coverage_is_low_for_an_unseen_pair`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5 (after the mockup is approved):** the dock tab, the pop-out window, the overlay (View toggle "knowledge overlay", display state, profile key, Ctrl-K). Build, launch; user test list: the graph on a water+methane collection, a pair's histogram with the "asked now" marker, the overlay on propane.
- [ ] **Step 6:** Commit `feat: the knowledge map — what the shared model knows, as a graph, histograms and a 3D overlay`.

### Task 10: Train this segment, with capping [Opus for the cut, Fable for the masked fit + review]

**Files:**
- Create: `crates/app/src/segment.rs` (the cut), `crates/validate/tests/capping.rs`
- Modify: `crates/surrogate/src/fit.rs` (`Sample.force_mask: Option<Vec<bool>>`, `Sample.use_energy: bool`; `Accumulator::add` honours both), `crates/surrogate/src/sidecar.rs` + `collection.rs` (persist them), `crates/app/src/ui/{actions.rs,dock/selection.rs,dock/model.rs}`, `crates/app/src/viewer.rs` ("back to the scene")

**Interfaces:**
- Consumes: `ToolState::set` (tools.rs:37), the builder's `bond_length(za, zb, order) -> f64` and `usual_valence(z)` (builder/chem.rs:80/:17), the snapshot's density bond graph when present.
- Produces: `pub struct Segment { pub scene: Scene, pub interior: Vec<bool> /* per segment atom: its whole r_cut neighbourhood matches the source */, pub caps: Vec<usize>, pub source_indices: Vec<Option<usize>> }`; `pub fn cut_segment(scene: &Scene, atoms: &[usize], r_cut: f64, bonds: Option<&BondGraph>) -> Result<Segment, String>`; `Action::TrainSegment`.
- The cut: selected atoms + every atom within `r_cut` of one of them; a bond is "cut" when one end is kept and the other is not — from the density bond graph when the snapshot has one, else geometrically (distance ≤ 1.25 × `bond_length(za, zb, 1)`); each cut bond gets an H at `bond_length(z_kept, 1, 1)` along the bond; the segment's samples carry `force_mask = interior` and `use_energy = false` (the total energy includes the caps) — the masked fit is Fable's, reviewed.
- Refusals: empty selection; a lone atom with no neighbour; a segment over the GPU memory budget (`providers::gpu_footprint_bytes` vs the budget) — each a named reason, no scene change.

- [ ] **Step 1: Failing tests** — `cut_caps_every_broken_bond` (butane, select C2: the segment has the caps on the cut C–C bonds and `interior` true only for atoms ≥ r_cut from every cut); `masked_samples_add_only_interior_force_rows` (an `Accumulator` with a masked sample: `n_f` grows by 3 × interior atoms, `n_e` unchanged); `segment_refusals_name_the_reason` (empty, lone atom, over budget). Capping gate `capped_segments_train_like_the_whole_molecule` in `validate/tests/capping.rs`: butane (and a water trimer) — collection A taught on the whole molecule's samples, collection B only on capped segments of it; on 20 held-out geometries of the whole molecule, B's force RMSE on interior-class atoms is within **1.5×** of A's (spec §4.5; if it fails, implement the uncapped two-shell cut instead and rerun).
- [ ] **Step 2–4:** FAIL → implement → PASS (`cargo test -p validate --release --test capping`, GPU, alone).
- [ ] **Step 5:** UI: "Train this segment" in the Selection tab and the Model tab; it saves the current scene to a temp file, opens the segment in Estimated with the Standard plan ready, shows a "back to the scene" chip. Build, launch; user test list: select part of propane, train the segment, go back.
- [ ] **Step 6:** Physics review of the masked fit; commit `feat: Train this segment — capped cut-outs teach the shared model`.

### Task 11: Integration, docs and the merge gates [Opus; Fable final physics review]

- [ ] **Step 1:** README (the shared model, the knowledge map, Train this segment, specialised models, the neutral-only limit), CLAUDE.md (collections, the new suites and their run times), DEVLOG (M8a entry), `references.md` complete.
- [ ] **Step 2:** Final gates on the merged tree (`git merge origin/main` first), one at a time: `cargo test -p surrogate -p providers -p sim --release`; `cargo test -p validate --release --test surrogate`; `--test transfer`; `--test capping`; `--test dynamics` if `sim` changed; `cargo test -p app --release`.
- [ ] **Step 3:** Whole-branch review (most capable model), one fix wave, merge to main, push, remove the worktree; hand-back: what changed, the user's test list, "Rulings I made".
