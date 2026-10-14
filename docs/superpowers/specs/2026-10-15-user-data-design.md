# User data: model library and settings profiles — Design Spec

**Date:** 2026-10-15 (written 2026-10-14, for execution after M6)
**Status:** Specified from the user's requests and the controller's binding rulings: the model library first, then, the same day, settings profiles in the same per-user data folder. The decisions this spec had to make on its own are marked **[provisional]** and listed in §18.
**Parts:** A — the model library (§1–§3, §5–§14); B — settings profiles (§15, with the settings inventory in Appendix A); C — the shared user-data machinery (§4: folder, atomic writes, locks), which both use. Review focus §16, validation §17.
**Author:** Claude (Opus 5.5) with Dakoda
**Parent:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (§5 architecture, §8 tooling); `docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md` (provenance and the refusal rule, S9 references, S15 quarantine, S18 sample tolerance); `docs/superpowers/specs/2026-10-14-m6-tools-design.md` (the `providers` crate, `Trainable`, the worker split, the sidecar v3 mass record — ruling P2).
**Inputs:** `docs/superpowers/notes/2026-10-13-m5-user-requests.md` § "Model library: automatic per-scene models" (the binding ruling); the user's follow-up (2026-10-14): "those models will auto save to files? then auto load when the app is reopened?" (answered by §5.4 and pinned in plan Task 7); the user's request (2026-10-14) "would it be possible to have settings profiles too? to be able to set a default profile, as well as reset to out of the box settings" and the controller's ruling on it (Part B, §15); `docs/superpowers/notes/2026-10-14-m5-handoff.md` (session autosave with ownership); `docs/superpowers/notes/2026-10-14-ui-mockup.html` (the Model tab).

**Lands after M6.** It is written against the M6 branch (`m6-tools`) as it stands: `providers::surrogate::{mod, store, trust, audit, quarantine}`, `sim::provider::{Trainable, TrainableStatus}`, `surrogate::sidecar` v3, `app::worker::{mod, model, autosave, commands}`, and the redesigned Model dock tab (`app::ui::dock::model`, on main, merged into M6 by its Task 10). Every name below that does not exist there yet is introduced by this spec.

## 1. Purpose (Part A: the model library; Part B, settings profiles, is §15)

Today a model lives for one session: the estimated worker writes it to a temp file (`%TEMP%\atomic-sandbox-<pid>-<n>.model`) when it retires, the next estimated worker of the same session reloads it, and the app deletes it on exit. A model survives the session only if the user presses **Save model**, which writes `<scene>.model` beside the scene file, and comes back only through **Load model**, which refuses a model taught on other settings. A user who moves between scenes either retrains or keeps track of model files by hand.

The library makes models persistent and automatic:

- **One model per Kohn-Sham surface and species set (the key, §3).** An H₂ scene and a six-H scene on the same grid share one model and add to the same samples. Water gets its own model.
- **Resolved from the scene.** Entering estimated mode, editing the scene, changing grid / σ / charge / multiplicity, or loading a scene looks the key up. If the key is found, that model is loaded and keeps learning. If not, a fresh model starts and the pass is offered.
- **Saved without asking.** The model is saved after refits (debounced), when a pass ends, on a switch and on exit. Nothing the teacher computed is lost silently: a save that fails is reported and retried, and the model stays in memory until it lands.
- **Switches are announced.** When an edit changes the key, the Log and a toast name the old model, the new one and the field that changed.
- **The Model tab says what the model is:** what it was trained on, how much (samples by origin, passes, teacher time), how well (fit, trust, dofs), where (pair-distance coverage, burst temperatures) and when.

What success looks like: the user opens the app, picks Estimated on H₂, samples once and plays. Then they load water (a new model and a pass offer), come back to H₂ (the H₂ model returns with every sample, untouched) and quit. The next day, Estimated on H₂ shows "loaded from the library (last trained 2026-10-14 17:46 UTC)". Its trust state is the one they left, and no refit or pass runs.

## 2. Non-goals

- **No transfer or merging across keys.** An H₂O model does not seed an H₂ model, a 49³ model does not seed a 57³ one, and a GPU-taught model does not seed a CPU-taught one. That belongs to the later locality/transfer milestone, which needs physics this spec does not have. Within a key, samples *are* pooled (that is what the key means, §3.1). Pooling by import (§9) and by a freed lease (§4.4) is the same operation.
- **No cloud sync**, no sharing service, no network. Export (§9) writes a file. Moving it is up to the user.
- **No automatic eviction.** The library grows until the user deletes entries in the browser (§11), which shows the total size.
- **Profiles are local files.** There is no profile sharing, sync or import UI: a profile is a JSON file in `<root>/profiles/` that the user can copy. Profiles never hold the scene's atoms (§15.1).
- **No change to the physics.** The fit, trust rule, fallback policy, quarantine and tolerances are untouched. The new sample tag (`Audit`) is bookkeeping, and the fit does not read tags (§7.2; pinned bit-exact in plan Task 4).
- **The app does not reopen the last scene on launch** unless the user opts in. It starts on H₂ (or `--preset`). So "auto load when reopened" means: when the user returns to a scene they sampled, by loading it, choosing its preset or building it, its model comes back. A scene does not come back by itself. Reopening the last scene is an optional small task (plan Task 10): a profile setting `launch.reopen_last_scene`, **off by default** (§15.5, Appendix A).

## 3. The key

### 3.1 Fields

A model is valid for any scene whose **library key** equals its own. The key is the M5 provenance gate (`surrogate::Provenance` equality, which the loader enforces) plus the scene's species set:

| field | source | canonical text |
|---|---|---|
| species set | the scene's nuclei: sorted unique Z | `species 1,8` |
| charge | `Scene::charge` | `charge 0` (`charge -1`, `charge 1`) |
| multiplicity | `Scene::multiplicity` | `multiplicity lowest` for `None`, else `multiplicity 3` |
| grid n | `Cell::n` | `grid 49` |
| h | `Cell::h` (Bohr) | `h 0.25` |
| σ | `Scene::sigma.unwrap_or(h)`, as `scene_provenance` takes it | `sigma 0.25` |
| teacher | `ForceProvider::name()` of the surrogate's teacher | `teacher Kohn-Sham (GPU)` |
| teacher tolerance | the surrogate's `teacher_tol` (`SAMPLE_TOL` = 1e-6) | `tol 1e-6` |

The key does not depend on geometry, atom count, velocities, the external field, the walls or the thermostat. The field and walls are driver-side potentials outside the Kohn-Sham surface (`sim::Dynamics::field_energy`/`walls_energy`). **Isotopes are recorded, not keyed** (ruling P2: the Born-Oppenheimer surface is mass-independent). The mass set goes into the file's provenance record (`Provenance::record_masses`), so H₂ and D₂ share one model, and the Model tab shows "masses ¹H ²H (recorded, not compared)".

**Why the species set is in the key when the provenance gate leaves it out.** The binding ruling puts it there. The gate lets a model taught on H answer nothing on water: rule 2 of the decision list sends every step with a new species to the teacher and marks the basis for a rebuild. That pooling is legal, but it is the wrong default for a library, because a user who builds water from H₂ would grow the H₂ model into a water model and lose the clean H₂ one. Separate keys keep each species set's model whole. A scene that later loses its O (water → remove O → H₂) switches back to the H model.

**The teacher name must change whenever the teacher's Hamiltonian does.** This is a requirement on future milestones and is stated in the plan's Global Constraints. The key cannot see a code change. If the pseudopotential milestone ships a new surface under the same name `Kohn-Sham (GPU)`, old all-electron models would load on it. That milestone must name its teacher distinctly, for example `Kohn-Sham (GPU, PP)`. Bug fixes that move energies by less than the trust bounds are not keyed; the file records `created_by` (the build) so such a gap can be traced. This gap is documented, not hidden (rule 7).

### 3.2 Canonical form

The canonical key is UTF-8 text, one field per line, in the order of the table, under a version line:

```text
atomic-sandbox model key v1
species 1
charge 0
multiplicity lowest
grid 49
h 0.25
sigma 0.25
teacher Kohn-Sham (GPU)
tol 1e-6
```

(each line ends with `\n`, including the last).

- **Floats (h, σ, tol)** are written with Rust's `{:?}`, the shortest decimal that round-trips to the same f64. The text is therefore **bit-exact**: two keys are equal exactly when the f64s are equal, which is exactly when the provenance gate (`Provenance::eq`, f64 `==`) accepts. `0.25` typed as "0.25", "0.250" or "2.5e-1" parses to one f64 and one key. `0.25 + 1 ulp` gives `h 0.25000000000000006`, a different key, which the gate would also refuse. A rounded key (for example `{:.4}`) would map two surfaces the gate refuses to the same file, and every load would fail. **Key and gate agree by construction, and plan Task 3 pins it** (§16, F2).
- `LibraryKey::new` refuses a non-finite or non-positive h, σ or tol, a teacher name containing `\n`, an empty species set, and Z = 0. The scene validator already rules these out, so a refusal is a bug, reported as an error and not keyed.
- −0.0 cannot occur: h, σ and tol are positive.

### 3.3 Hashing and file names

The file name is `<h>.model`, where `<h>` is the 64-bit **FNV-1a** of the canonical text as 16 lower-case hex digits. The function is hand-written (offset basis 0xcbf29ce484222325, prime 0x100000001b3), so it is the same on every build and platform. **`std::collections::hash_map::DefaultHasher` is forbidden here**: its output is not stable across Rust releases. Pinned values (computed independently, plan Task 3):

| key | FNV-1a |
|---|---|
| H, charge 0, lowest, 49³, h 0.25, σ 0.25, `Kohn-Sham (GPU)`, 1e-6 | `2d97a5e0204aa194` |
| H O, 0, lowest, 57³, 0.25, 0.25, `Kohn-Sham (GPU)`, 1e-6 | `6d1e5a66779f2e3b` |
| H, 0, lowest, 49³, 0.25, 0.25, `Kohn-Sham (CPU)`, 1e-6 | `668ae6928f73de90` |
| H, 0, lowest, 49³, h = σ = 0.25000000000000006, GPU, 1e-6 | `69a816e5564c635c` |
| H, 0, lowest, 41³, 0.25, 0.25, `Kohn-Sham (GPU)`, 1e-6 | `7f20eb58eb88f7fc` |
| H, 0, lowest, 33³, 0.25, 0.25, `morse dimer`, 1e-6 (the worker tests' key) | `3ec9d60e924dcf31` |
| …the same at 35³ | `809f73395b6ffd6b` |
| FNV-1a of the empty string / of `a` (the published test vectors) | `cbf29ce484222325` / `af63dc4c8601ec8c` |

**The hash only names files. It never identifies a model.** Each v4 file stores its full canonical key in its header (§7.1), and every lookup compares that text. If a file named `<h>.model` holds another key (a hash collision, or a file copied in by hand), the new entry is written as `<h>-2.model` (then `-3`, …) and the index maps the key to that name.

### 3.4 Human label and the reasons for a switch

`LibraryKey::label()` is display text. It is never parsed and never keyed:

```text
{species symbols, by Z}[ · charge {+c}][ · mult {m}] · {n}³ · {h} Bohr[ σ {σ}] · {teacher} {tol:e}
```

The σ part appears only when σ ≠ h, the charge part only when the charge is not 0, and the multiplicity part only when it is set. Teacher `Kohn-Sham (X)` shows as `X`; any other name shows as itself. Symbols come from `units::isotopes::symbol` (display data). Pinned examples: `H · 49³ · 0.25 Bohr · GPU 1e-6`, `H O · 57³ · 0.25 Bohr · GPU 1e-6`, `H · 33³ · 0.25 Bohr σ 0.3 · morse dimer 1e-6`, `H · charge +1 · mult 3 · 33³ · 0.25 Bohr · morse dimer 1e-6`.

`LibraryKey::diff(&old, &new) -> Vec<String>` names the fields that changed, in table order: `species H → H O`, `charge 0 → +1`, `multiplicity lowest → 3`, `grid 49³ → 57³`, `h 0.25 → 0.3 Bohr`, `σ 0.25 → 0.3 Bohr`, `teacher GPU → CPU`, `tolerance 1e-5 → 1e-6`. The switch announcement (§5.3) uses it.

## 4. The user-data folder (Part C: shared by the library and the profiles)

### 4.1 Directory

One per-user data root holds both parts: `<root>/models/` (the library) and `<root>/profiles/` plus `<root>/user.json` (§15). The machinery (resolution, atomic writes, locks, the JSON reader/writer, FNV-1a, `utc_label`) is a small new std-only crate, **`userdata`**. The `surrogate` crate (the library, the sidecar's JSON) and `app` (the profiles) both use it. The root is resolved once, in `main`, in this order:

1. `--data-dir <path>` on the command line;
2. the environment variable `ATOMIC_SANDBOX_DATA_DIR`;
3. the platform default: **`%LOCALAPPDATA%\atomic-sandbox\`** on Windows **[provisional, L1]**; `$XDG_DATA_HOME/atomic-sandbox` or `$HOME/.local/share/atomic-sandbox` elsewhere. No `dirs` crate: these are three environment reads (`userdata::default_root`).

The library's directory is `<root>/models/`, written `<dir>` below.

**L1: Local, not Roaming (`%APPDATA%`).** The library is unbounded in size. A body-3 model's Gram matrices are 2·Q′²·8 bytes, several MB at Q′ ≈ 500. Windows guidance keeps large, machine-local data out of the roaming profile, where domain logons copy it at every sign-in. The models are not machine-specific (the teacher name and tolerance are in the key), so Export (§9) moves one by hand.

If `<dir>` cannot be created or written (§14), the library is **unavailable** for the session. Each key's model is then kept in memory for the session only, as today's session autosave did. The Model tab and the Log say "model library unavailable: <error> — models are kept for this session only; Export to keep one".

**Tests never touch the real user data.** `userdata::default_root()` is called only from `main.rs`. Workers, viewers and the profile store take an explicit directory, and every test passes its own temp directory. A `grep` step in plan Task 5 enforces this. `--screenshot` runs use a fresh temp root unless `--data-dir` is given **[provisional, L2]**. They therefore start from factory settings with an empty library: README screenshots do not depend on, or write into, the user's data.

### 4.2 Layout

```text
<dir>/
  index.json               # the cache of entries (below); rebuildable from the .model headers
  index.lock               # held (exclusive) only while index.json is read-modified-written
  2d97a5e0204aa194.model   # one sidecar (v4, §7) per key
  2d97a5e0204aa194.lock    # the entry's lease (§4.4); never deleted
  6d1e5a66779f2e3b.model
  …
  3ec9d60e924dcf31.model.corrupt   # an unreadable file, set aside (§14), never deleted by the app
```

**The `.model` files are the truth and `index.json` is a cache.** Every v4 header carries the canonical key and the record the index shows (§7.1). If the index is missing or unreadable, or disagrees with the directory, it is rebuilt from the headers, which is a header-only read (`sidecar::read_header`). On open, the library compares the index with the directory listing (names, sizes, mtimes) and re-reads only the files that differ. Only the "scenes used on" lists live in the index alone, so a rebuild loses them and says so in the Log.

`index.json` (hand-written JSON through the sidecar's own reader and writer, moved to `userdata::json`):

```json
{"format": "atomic-sandbox model library 1",
 "entries": [
  {"file": "2d97a5e0204aa194.model",
   "key": "atomic-sandbox model key v1\nspecies 1\n…tol 1e-6\n",
   "label": "H · 49³ · 0.25 Bohr · GPU 1e-6",
   "created": 1792000000, "last_trained": 1792003600,
   "samples": 412, "quarantined": 2, "fitted": true, "passes": 3,
   "size": 183552, "modified": 1792003605,
   "scenes": [{"name": "preset h2", "last_used": 1792003600},
              {"name": "D:\\runs\\h2-stretch.scene", "last_used": 1791990000}],
   "state": "ok"}],
 "imported": [{"path": "D:\\runs\\h2.model", "size": 183552, "modified": 1791900000,
               "into": "2d97a5e0204aa194.model", "at": 1792000000}]}
```

- Times are Unix seconds, UTC. They are displayed as `2026-10-14 17:46 UTC` by a hand-written civil-from-days conversion (`userdata::time::utc_label`, pinned: 0 → `1970-01-01 00:00 UTC`, 951782400 → `2000-02-29 00:00 UTC`, 1792000000 → `2026-10-14 17:46 UTC`) and as a relative age ("3 h ago").
- `scenes` holds the ten most recently used, newest first. A name is `preset <name>`, the scene file's path, or `unsaved (<formula>)` using the M6 `app::names` formula.
- `state` is `ok`, `corrupt` (the `.corrupt` file's path is kept in `"corrupt_file"`) or `newer` (a format this build cannot read, §14).
- `imported` records each migrated or imported file by (path, size, mtime), so it is imported once (§8).

### 4.3 Atomic writes

Every write (a `.model`, `index.json`, a profile or `user.json`) goes through these steps (`userdata::atomic_write`):

1. Write the bytes to `<name>.<pid>-<seq>.tmp` in the same directory. The pid suffix means two instances never share a temp file.
2. `File::sync_all` the temp file.
3. `std::fs::rename` it onto `<name>`. On Windows this is `MoveFileExW(REPLACE_EXISTING)`. A destination held open without delete-sharing, typically by an antivirus scanner, makes the rename fail transiently, so it is retried 5 times over 500 ms before it counts as a failure.
4. On failure, remove the temp file. The previous file is intact, because the old bytes are never truncated in place.

At open, the library removes `*.tmp` files left more than 24 h ago by a crashed write. They match the pattern above and are never a user's file.

### 4.4 Two instances: the index lock and the entry lease

Two app instances, or two workers of one instance during a backend switch (the old one retires for up to `RETIRE_WAIT` = 60 s), can touch the same library. They coordinate through `std::fs::File` locks (`try_lock`, stable since Rust 1.89; this toolchain is 1.99). On Windows these are `LockFileEx` locks, released by the OS when the process dies, so a crash never leaves a stale lock.

- **`index.lock`** is held only while the index is read, changed and written back (milliseconds). The writer re-reads `index.json` under the lock and changes only its own entries before writing, so two instances' updates both survive. Acquisition is `try_lock` polled for up to 2 s. If that fails, the index update is skipped and logged. The index is only a cache (§4.2), so the next update or open repairs it.
- **`<h>.lock`, the entry lease,** is taken (`try_lock`, exclusive) when an entry becomes the worker's active model and held until it switches away or retires. **Only the lease holder writes `<h>.model`.** A worker that cannot get the lease opens the entry **read-only**: it loads a copy, trains in memory, and never writes. This is announced: "model 'H · 49³ …' is in use by another instance or worker: this session trains a copy that is not saved — Export to keep it". It retries the lease every `LEASE_RETRY` = 30 s. Once it holds the lease, it **merges** the file's samples that it does not hold into its own (the file-level merge of §8.3), writes the union, reloads it, and owns the entry from then on. A retiring worker that times out and finishes later therefore hands its samples over, with nothing lost and nothing duplicated.
- Deleting an entry (Forget, the browser) needs its lease. An entry leased elsewhere cannot be deleted, and the UI says why.
- **Profiles** use one short lock, `<root>/profiles/profiles.lock`, held while a profile file or `user.json` is written or deleted (the same `try_lock` loop, 2 s). There are no leases: a profile is written only by an explicit user action, so two instances saving the same profile resolve to last-writer-wins. Each write is atomic, so neither leaves a torn file (§15.6).

## 5. Lifecycle

### 5.1 When the key is resolved

The worker owns resolution. It computes the scene's key after any applied command that can change it:

- `EnsureModel`, which is sent when estimated mode is entered;
- `SetScene` / `ReplaceScene` (a scene or preset load, charge, multiplicity, grid, σ);
- `AddNucleus` with a new Z;
- `RemoveNucleus` that removes the last nucleus of a Z.

A teacher change cannot happen within a worker: a backend switch spawns a new worker, which resolves at its own `EnsureModel`.

The worker also keeps the scene's name for the index: `Command::SetSceneName(String)`, sent by the viewer on Load / Save scene, on preset choice, and with `unsaved (<formula>)` after the first edit of a named scene.

### 5.2 Resolve → load or create

`resolve(key)`:

1. **Index lookup** by canonical key, falling back to a directory scan if the index is stale.
2. **Found:**
   - Take the lease. If that fails, open read-only (§4.4).
   - Read the file. If it cannot be read, follow §14 (corrupt → set aside, fresh start).
   - Check the header's key equals ours (§3.3) and install it through `DynCommand::LoadModel(path)`. That goes through the unchanged provenance gate, so a file whose key and provenance disagree is treated as corrupt (§14).
   - Samples-only files (§7.1, §13) install their samples and references without a model.
   - **No refit runs on load.** The file holds the fit, its accumulator and its report, and the trust state is recomputed exactly as `install_sidecar` does today (training residuals from the stored samples). Samples taken after the last refit come back pending and are refitted at the next opportunity, as today.
3. **Not found:** a fresh, empty model.
   - **No file is created until the first sample is kept**, so browsing scenes leaves no empty entries.
   - The index entry is created with the file.
   - The lease is taken at creation.

### 5.3 Switching

A switch happens when the resolved key differs from the active entry's. The sequence, all on the worker thread and between ticks:

1. **A running pass is ended first.** If the edit itself is a `ReplaceScene`, the driver has already cancelled the pass, because an accepted `ReplaceScene` cancels it today. Otherwise the worker applies `CancelSampling` before the edit. The pass's samples belong to the **old** key. Cancel keeps and refits them, or keeps them without a refit below Q′ (M6 Task 3 round 2). An edit that does **not** change the key (moving, adding or removing a nucleus of an existing species) does not touch the pass, which is today's behaviour.
2. **Save the old entry** (§6, an immediate save). If the save fails, the encoded bytes go to the worker's `unsaved` list with the error. They are retried every `SAVE_RETRY` = 30 s and at retire, and shown in red in the Model tab until they land (§6.4). **Samples are never dropped**: the in-memory model is released only after its bytes are either on disk or held in `unsaved`.
3. Release the old lease. Then `ForgetModel` on the provider, quietly (no "model forgotten" note; the switch note replaces it). The reference cache survives a forget, as today. If the old model was the reported surface, the driver restarts its conserved-energy history: the surface changed, and that is the honest outcome.
4. `resolve(new key)` (§5.2).
5. **Announce** (Log entry + toast, `Event::Library(LibraryNote::Switched{…})`):

   > model switched — 'H · 49³ · 0.25 Bohr · GPU 1e-6' (412 samples, saved) → 'H O · 49³ · 0.25 Bohr · GPU 1e-6' (new, no samples yet) — species H → H O

   For a found entry the right-hand side reads "(380 samples, trusted)". A read-only one reads "(…, read-only: in use elsewhere)", and a failed save of the old entry reads "(412 samples, NOT SAVED: <error>)" in red.
6. **What a fresh or unfitted entry gets.** On **entering estimated mode** (`EnsureModel`), the behaviour is today's automatic first pass, re-sized from the median of its first solves or offered when long (`first_pass`, `PassOffer`). **After a switch caused by an edit**, nothing starts automatically **[provisional, L3]**. The Model tab shows "no model for this scene yet" with the Start button highlighted, and the toast ends "— Start sampling in the Model tab". Building a molecule atom by atom would otherwise start a pass for every intermediate species set.

Sequences that switch back and forth (add O, remove O) cost a save and a load each: milliseconds for H₂-sized files, measured and reported in plan Task 5.

### 5.4 Restart: the model comes back when the app is reopened

This is the user's follow-up, stated explicitly:

- **After a clean exit**, the model is on disk. Closing the window shuts the estimated worker down through `retire` (§6.3), which saves the active entry, and the app waits up to `EXIT_SAVE_WAIT` = 10 s for it.
- **After a crash**, the last debounced save is on disk. Saves happen at most `SAVE_EVERY` = 5 s after any change, and immediately at pass end (§6.1). Only the samples kept since that save, plus the solve that was in flight, are lost.
- **Relaunching and entering estimated mode on a scene with a known key loads that model**:
  - every stored sample, with its tag, quarantine flag and face density;
  - its fit (the model, accumulator and `FitReport`, so the same rmse E/F, Q′ and dofs);
  - its trust state (trusted or not, and the same reason), recomputed from the stored samples exactly as `install_sidecar` does;
  - its history (passes, teacher time, dates, lifetime refits).
- **Without a refit and without a pass.** The worker's refit counter stays 0 and the surface epoch is the load's. The Model tab's state line reads **"loaded from the library (last trained 2026-10-14 17:46 UTC)"**: the record's `last_trained` through `utc_label`.
- Samples pending at the save (kept after the last refit) come back pending and are refitted at the next opportunity. That is the same model continuing, not a retraining.
- Pinned by plan Task 7: `a_relaunched_viewer_restores_the_model_without_a_refit` covers the clean exit, and `a_crash_restores_the_last_debounced_save` covers the crash.
- Reopening the **scene** itself on launch is the optional plan Task 10 (§2).

### 5.5 Forget

**Forget model…** keeps its two-click confirmation (4 s). It now drops the in-memory model **and deletes the library entry**: the file is removed, the index entry dropped, and the lease released. The second click's label says so: "Click again to delete this model and its 412 samples". After a forget the entry is gone. The next kept sample creates a fresh one under the same key.

## 6. Saving

### 6.1 When

The provider exposes `Trainable::revision() -> u64`. It changes whenever anything the file holds changes: a kept sample, a refit, a quarantine flag, a pass record, the references, or the mass record. The worker saves the active (leased) entry when `revision != saved_revision` and one of these holds:

- **debounced:** at least `SAVE_EVERY` = 5 s since the last save. This covers refits (at most one per `REFIT_INTERVAL` = 500 ms while playing), kept fallback and audit samples, the samples a pass adds while it runs, and the < Q′ samples that have no model;
- **immediately:**
  - when a pass finishes or is cancelled;
  - on a switch (§5.3);
  - on retire (backend switch, exit);
  - before a panic recovery rebuilds the driver (replacing today's carry-over through the autosave path);
  - on Export, because the export writes the same bytes.

### 6.2 How: encode on the worker thread, then write

`Trainable::encode(key) -> Result<Option<Vec<u8>>, String>` returns the full v4 file as bytes, or `None` when nothing is held. The worker calls it **between ticks, on the worker thread**: the same thread that runs refits, pass steps and installs. A save therefore always holds a consistent state and **cannot race a refit**. A refit happens before the encode or after it, never during. The worker then writes the bytes (§4.3), records `saved_revision` (the revision it encoded, not the revision at write end) and updates the index (§4.4).

The save cost (encode + write + fsync) is measured and reported (plan Task 5, timing reported, not asserted). It is expected to be milliseconds for H₂ sizes and tens of ms for a large body-3 model. If a measurement shows a tick-visible cost, the write moves to a writer thread fed with `(revision, bytes)`. The writer drops an older revision queued behind a newer one and never writes a revision older than the one on disk. The encode stays on the worker thread either way.

### 6.3 Exit

Today `main` calls `process::exit` without joining the worker, and the session autosave is deleted. With the library:

- On close, the viewer sends `Shutdown` to an estimated worker. It keeps the window up with a "saving the model…" card for up to `EXIT_SAVE_WAIT` = 10 s, waiting for `Event::Retired`, which `retire` sends after its save. A GPU solve in flight stops within one SCF step, so this normally takes well under a second.
- A CPU solve cannot be interrupted. If the wait times out, the app exits anyway and prints to stderr: "the model library was last saved at <time> (<n> samples); samples kept since then were not saved".
- The temp-file autosave and its removal at exit are deleted from `main.rs` and `viewer.rs`.

### 6.4 Failure is visible, never silent

A failed write (disk full, permissions, the rename retries exhausted) is handled as follows:

- an `Event::Library(LibraryNote::Error)` goes to the Log, with a toast in red;
- the Model tab's file line turns red: "not saved since 17:46:12: <OS error> — retrying every 30 s; Export to save elsewhere";
- the bytes stay in memory (`unsaved`) and are retried. A newer save of the same key replaces them in the list, because the newer revision contains the older one;
- at exit, a still-unsaved model is named on stderr;
- **no failure path drops samples without a message**.

## 7. The model file: sidecar v4

### 7.1 Format changes

Everything in the v3 format stays (`surrogate::sidecar` module doc). Version 4 (`ASMODEL4`, the only version this build writes) adds four things:

1. **Samples-only files.** The header's `model`, `basis`, `accumulator`, `fit` and `n_samples` are all `null` together. The model arrays (`coeffs` … `env_sumsq`, `body3_terms`) are then absent, and `sample_*` are present (≥ 1 sample). `species` is the sorted union of the samples' Z. `provenance` is required (the store's). `sample_quarantine` is absent, because quarantine only marks the fitted prefix. `sample_face_density` is optional as before. A new header key `references` carries `[[Z, E], …]`: the isolated-atom energies the provider holds, bit-exact via `{:?}` with `reference_rule`. A samples-only file needs this key. A fitted file carries its energies in `ref_energy` as before. **This is how the < Q′ rule's samples (§13), and the "no model yet" samples, survive a restart** without re-solving the references (a cold O reference is 13–15 s per multiplicity).
2. **Tag code 6 = `Audit`.** An admitted background or in-step audit answer is kept as `SampleTag::Audit`; it was tagged `Fallback` before v4. Earlier files keep their codes, so their audits count as fallbacks, and the Model tab says so (§12).
3. **The `library` header object**, which is the record the Model tab and the index show:

```json
"library": {"key": "atomic-sandbox model key v1\n…", "created": 1792000000, "last_trained": 1792003600,
            "refits": 37, "revision": 1180,
            "teacher_s": {"pass": 61.4, "fallback": 12.9, "audit": 30.2},
            "passes": [{"at": 1792000000, "perturbations": 60, "sigma_r": 0.15, "burst_steps": 200,
                        "burst_t": 2000.0, "scan_points": 12, "seed": 7, "solves": 273, "kept": 271,
                        "cancelled": false, "teacher_s": 41.0}],
            "history_since": null}
```

   - `created` is the first sample's time. `last_trained` is the last refit's time, or the last kept sample's for a samples-only file.
   - `refits` counts lifetime refits, the pass's included.
   - `teacher_s` sums `EvalStats::solve_ms` over every teacher evaluation the surrogate made for this model: pass solves (references included), fallback steps and audits, kept or not.
   - `passes` holds one record per pass, ended or cancelled.
   - `history_since` is the import time of a pre-v4 file, whose earlier history is unrecorded.
4. **`created_by` is now also written for samples-only files.** (In v3 it was always written; nothing changes for fitted files.)

**v1–v3 files are still read** (`read` unchanged for them). They have no `library` object, so the record says "unrecorded" for dates, passes, teacher time and lifetime refits. They also have no audit tag. `created` and `last_trained` then fall back to the file's mtime, labelled "≈ (file time)". `read_header(path)` reads magic and header only, for the index, collisions and import keys. An `ASMODEL5`-or-later magic reads as `Err(kind Unsupported)`, never as corrupt (§14).

### 7.2 What the store must start recording

The provider (`providers::surrogate`, plan Task 4) gains the following. None of it enters the fit, the trust rule or the fallback policy.

- `SampleTag::Audit` for kept audit answers. The `keep_sample` call site in `audit.rs` passes the tag. **The fit does not read tags**: plan Task 4 pins that a store with `Audit` tags fits bit-identically to the same store tagged `Fallback`.
- Teacher time per origin, from each teacher answer's `EvalStats::solve_ms`.
- Pass records: the `SamplingPlan` fields, solves done, samples kept, cancelled or not, and teacher time. They are written when the pass ends or is cancelled.
- `created` / `last_trained` (Unix seconds from `SystemTime`) and lifetime refits.
- `revision` (§6.1).
- **Coverage, derived from the samples (nothing stored).** Each species pair gets the range [r_min, r_max] of the pair distances of the samples the fit uses (quarantine code 0), over pairs closer than the basis's r_cut, the ones the model learns from. Each also gets the count of samples contributing. The burst temperatures come from the pass records. Coverage is recomputed at a refit, an install or a merge, and cached. It is never computed per tick.

## 8. Migration and import

### 8.1 Leftover session autosaves

The temp files `atomic-sandbox-<pid>-<n>.model` in `std::env::temp_dir()` normally disappear at exit. A crash leaves them behind, and they may hold the only copy of a model.

- On the first library open of a session, each one not already in `imported` (by path, size and mtime) is imported (§8.3) into the entry its key names.
- The key comes from the file's provenance and its header `species`.
- **The original is kept**, and recorded in `imported` so it is not imported again.
- A file whose pid is a live instance of an older build is harmless to import: it is copied, never moved.
- The Log says "imported 1 earlier session model (H · 41³ · 0.25 Bohr · GPU 1e-6, 83 samples) from %TEMP%\…".

### 8.2 A `<scene>.model` beside a scene file

When a scene is loaded (Load scene, in estimated mode) and `model_path(scene_path)` exists and is not in `imported` (by path, size and mtime), it is imported (§8.3) on first sight. **The original stays where it is.** If its key is the scene's, the import lands in the active entry. If not, it lands in its own entry and the Log says which, and why it is not the active one ("imported my.model as 'H · 57³ …' — not this scene's model: grid 57³ → 49³"). A later Load of the same unchanged file does nothing. If the file was rewritten by an Export or by another tool, its new size or mtime makes it a fresh sighting, and the bitwise de-duplication of §8.3 keeps the result free of duplicates.

### 8.3 The import rule (same key only)

`surrogate::sidecar::merge(base, other) -> (Sidecar, appended)` is file-level and needs no provider. **[Fable]** owns it in plan Task 4, because it decides what the fit sees.

- **The keys must be equal.** Different keys never merge (§2). Import routes a file to its own key's entry instead.
- **No entry yet:** the file is copied in as is, re-encoded as v4 with `history_since` = now.
- **Entry exists:** the result keeps the **fitted** side's model, accumulator, report and samples, in order. If both are fitted, the library entry's are kept. If neither is fitted, the entry's are kept. The fitted prefix must stay a prefix (`install_sidecar` derives the accumulated prefix from `accumulator.n_samples` and the quarantine flags). The other side's samples that are not already present are then appended **as pending**:
  - "present" means bitwise equality of (z, pos, energy, forces);
  - appended samples carry their **tag, weight and face density** (so the box rule still applies to them);
  - their **quarantine flags are dropped**. Quarantine only marks samples inside the fitted prefix, and the next refit's trust recovery re-judges them (ruling S15, unchanged);
  - the reference energies must agree bit-exactly on shared species. A disagreement refuses the merge with a message naming the species. Both were solved by the same teacher at the same tolerance, so a mismatch means a different reference rule (pre-S9), and mixing those is refused, never averaged.
- The merged entry's next activation finds samples pending and refits. A merge into the **active** entry is applied as: encode the provider's state → merge with the import → write → `LoadModel(entry)`. It is refused while a pass runs: "cancel the running sampling pass first", the existing rule.
- The history records merge: passes concatenate, teacher times add, `created` = min, `last_trained` = max, refits add.

## 9. Export and import (Save / Load renamed)

The explicit file buttons stay, renamed to say what they now are:

- **Export model** (was Save model) writes the active entry's current bytes to `model_path(scene_path)` beside the scene file, through the atomic write of §4.3. It is enabled with **any samples** (it used to need a fitted model), because a samples-only file is now a valid file. The message reads "exported the model 'H · 49³ …' (412 samples) to D:\runs\h2.model".
- **Import model** (was Load model) reads `model_path(scene_path)` and imports it (§8.3).
- **The refusal rules keep their meaning.** A model is **never installed on a scene whose provenance differs**. The gate is unchanged, and `Trainable::load` is still the only installer. A different-key import is stored in its own entry, not refused-and-lost, and the message explains it with `Provenance::describe` on both sides, as today's refusal does. A corrupt file is refused with the reader's error and is not set aside, because it is the user's file and not the library's. Import is refused while a pass runs (same-key case, §8.3).
- The File ▸ Estimated model menu, the Model tab buttons and the Ctrl-K entries are renamed. The search entries keep "save model" / "load model" as keywords. The scene Load's automatic sidecar load is replaced by §8.2.

## 10. Interaction with M6 and the worker

- **The session autosave (`worker/autosave.rs`, `owns_autosave`, `Viewer::autosave`) is removed.** It is replaced by the lease. The ownership rule it implemented (a worker that never resolved must not overwrite or delete a model) becomes: a worker writes only an entry whose lease it holds, and it resolves (takes a lease) only at `EnsureModel`. A worker shut down before `EnsureModel` holds no lease and writes nothing, which keeps today's I1 guarantee. A refused autoload no longer exists: a key's file always matches its key, otherwise §14 applies. So `autoload_refused` and "Forget it and sample this scene" go away. The provenance-mismatch warning stays for a model installed on a scene whose provenance later changed. That case cannot arise with the library resolving every change, but the warning is kept as a guard.
- **`Command::EnsureModel { autoload, plan }`** becomes `EnsureModel { plan }`. The worker resolves the key itself.
- **The spawn signature** changes from `SimWorker::spawn_with(…, autosave: Option<PathBuf>)` to `…, library: Option<LibraryConfig>`. `LibraryConfig { dir, scan_legacy: bool }`, where `scan_legacy` is false in tests unless a test sets it. `spawn_test` takes one too.
- **Panic recovery** saves through the library (inside `catch_unwind`), rebuilds the driver, and re-resolves.
- **New snapshot fields:** `ModelStatus` gains `record: sim::ModelRecord` (from `TrainableStatus::record`) and `entry: Option<EntryStatus>`. `EntryStatus` holds the label, file, read-only reason, `loaded_from_library: Option<u64>` (last_trained, for §5.4's line), last save time, save error and the count of other unsaved models. `Snapshot` gains `library: Option<Arc<LibraryView>>` (the index entries, total size, the directory and its availability), republished only when it changes.
- **`Trainable` gains** `teacher_tol()`, `revision()`, `encode(key)` and `record()`. `save` writes samples-only files, and `load` reads them. These are listed in the plan's Task 4 interfaces. Nothing else in `sim` changes, and `sim` still never sees a file format (rule 5): bytes and plain records cross the trait.

## 11. The library browser

The Model tab gains a collapsible **Library** section (search id `d-library`; File ▸ Estimated model ▸ "Model library…" and Ctrl-K "model library" open it). It sits in the dock rather than in a new window, which keeps the redesign's no-floating-windows layout. It lists every entry, newest `last_trained` first:

| column | content |
|---|---|
| ● | filled for the active entry; amber for read-only (leased elsewhere); red for `corrupt` / `newer` |
| model | the label |
| samples | total (· quarantined n) |
| last trained | relative age; hover shows the UTC date |
| size | kB / MB |

The header line reads "12 models · 38.2 MB · <directory>", with the path selectable as text.

**Selecting an entry** shows its scenes used on, created date, passes and fitted/trusted-at-last-save state (from the header). Then:

- **If it is the active entry,** the section says "this scene's model".
- **Otherwise switching to it is impossible, and the section says why** with `LibraryKey::diff`: "this model is for 'H O · 57³ · 0.25 Bohr · GPU 1e-6'; the scene is 'H · 49³ · 0.25 Bohr · GPU 1e-6' — species H → H O, grid 49³ → 57³. A model only answers on the Kohn-Sham surface and species it was taught on; load a scene with these settings to use it."
- If one of its recorded scenes is a preset or an existing scene file, an **Open that scene** button loads it. That uses the existing LoadPreset / LoadScene actions, which resolve to this entry.
- **Delete…** uses the two-click confirmation (4 s, as Forget): "Click again to delete 'H O · 57³ …' (380 samples)". It needs the lease (§4.4). Deleting the active entry is Forget (§5.5). A `corrupt` entry's Delete removes only the index entry; the `.corrupt` file stays for the user (the path is shown).

## 12. The Model tab's information

A new **This model** group heads the Model tab's readouts (search id `d-model-info`). The "exists" column says whether today's M6 code already has the value, where, and what this spec adds.

| row | content (example) | exists in M6? |
|---|---|---|
| model | `H · 49³ · 0.25 Bohr · GPU 1e-6`; "loaded from the library (last trained 2026-10-14 17:46 UTC)" / "new — no samples yet" / "read-only: in use elsewhere" | **new** (key, entry status) |
| trained on | `Provenance::describe_recorded()` of the store: "49³ h 0.25 σ 0.25, charge 0, multiplicity lowest, teacher 'Kohn-Sham (GPU)' at tol 1e-6, masses ¹H ²H (recorded, not compared)" | data exists (`SampleStore::provenance`); **not exposed** — `ModelRecord::provenance` |
| samples | "412 — pass 343 (60 perturbations · 36 scans · 247 burst) · fallback 51 · audit 18" (+ "live n" when > 0); pre-v4 files: "fallback (audits before v4 counted here)" | total exists (`samples`); **by origin new** (tags exist except `Audit`, §7.2) |
| excluded | "quarantined 2 · superseded 0 · box-affected 1" | quarantined count exists; **the split is new** (from the flags) |
| passes | "3 (last 2026-10-14 17:40 UTC, 273 solves, finished)"; expandable: one line per pass with plan, T and teacher time | **new** (pass records) |
| teacher time | "1 min 44 s — pass 61 s · fallback 13 s · audit 30 s" | **new** |
| fit | "epoch 14 (this session) · refits 37 (lifetime) · last fit 412 samples, rmse E 2.5e-6 Ha/atom, F 3.1e-4 Ha/Bohr" | epoch, fit tuple exist; **lifetime refits new** |
| dofs | "Q′ 16 · dof_eff 14.2 · residual dofs E 396 / F 2461" | Q′ exists (fit tuple); **`dof_eff`, `residual_dof_e/f` exist in `FitReport` but are not exposed** — `ModelRecord::fit_dof` |
| trust | "trusted" / "no — n_samples 12 < q_prime 16" | exists (`trusted`, `untrusted_reason`) |
| coverage | per species pair: "H–H 1.12–3.80 Bohr (412 samples)"; a current-scene pair distance outside its range is amber: "now 4.10: outside what was sampled"; "burst temperatures 2000 K ×2, 500 K" | **new** (derived, §7.2); burst T from pass records |
| file | path · size · created · last trained · last saved (red with the error, §6.4) | **new** |
| scenes | the index's "scenes used on" (up to 10) | **new** |

Units: distances in Bohr with Å on hover through `units`, and energies as in today's readouts (rule 3, conversion at the UI edge). Every new row has a tooltip that says where the number comes from. Rows a pre-v4 file cannot fill say "unrecorded (file before v4)", never 0.

## 13. The < Q′ rule

M6 (Task 3 fix round 2) makes a pass cancelled below Q′ keep its samples **without** a refit and without a model. Playing with no model also keeps up to 4 Q′ DFT answers, again without a refit. Today `save` refuses with `InvalidInput` when nothing is fitted, so those samples die with the session. Under the library:

- `encode` writes a **samples-only v4 file** (§7.1) whenever samples exist. The entry is created at the first kept sample (§5.2).
- Such an entry restores as "samples, no model". The Model tab reads "412… no — 11 samples kept, not enough to fit (Q′ = 16); sample more to train", and EnsureModel's automatic first pass (or the edit-switch offer, L3) continues from them. **The next pass's refit covers all of them**, which is the pass's existing behaviour ("refits over everything it holds").
- The references the cancelled pass solved are kept in the file's `references` (§7.1). The next pass finds them cached and does not re-solve them.
- Pinned: plan Task 5 `a_cancelled_pass_below_q_prime_is_persisted_and_restored`.

## 14. Errors

| case | detection | what happens | the user sees |
|---|---|---|---|
| corrupt entry file | `sidecar::read` → `InvalidData`; or the header key ≠ the entry's key and the provenance disagrees with it | renamed to `<h>.model.corrupt` (`.corrupt-<unix>` if that exists); index `state: corrupt`; the key starts fresh (§5.2) | Log error + red toast: "model file <path> was unreadable (<reader error>); kept as <path>.corrupt; starting a fresh model for this scene" |
| newer format (`ASMODEL5`+) | `read_header` → `Unsupported` | never overwritten, never moved; the key is **read-only for the session**: a fresh in-memory model that is not saved | "model <label> was written by a newer version of the app; it is left untouched — this session's model for the scene is not saved (Export to keep it)" |
| corrupt `index.json` | the JSON reader fails | renamed to `index.json.corrupt`; rebuilt from the headers | Log note: "the library index was unreadable; rebuilt from 12 model files (the 'scenes used on' lists were lost)" |
| disk full / write error | the temp write, `sync_all` or the rename fails | §6.4: the original is intact, the bytes are kept, retried every 30 s | red toast + red file line; stderr at exit if still unsaved |
| read-only or uncreatable directory | `create_dir_all` or the first temp write fails at open | library **unavailable** for the session (§4.1); models are per key in memory | "model library unavailable: <error> — models are kept for this session only; Export to keep one" |
| lease held elsewhere | `try_lock` fails | read-only copy, lease retried every 30 s, merged on acquisition (§4.4) | amber line + toast at the switch |
| index lock busy > 2 s | `try_lock` loop | index update skipped; the files are intact | Log note only |
| import of a corrupt file | `read` fails | refused, the file is left in place (§9) | the reader's error, as today |

## 15. Settings profiles (Part B)

The user's request (2026-10-14): "would it be possible to have settings profiles too? to be able to set a default profile, as well as reset to out of the box settings". The controller's ruling sets the scope:

- profiles live in the same per-user folder and share the machinery of §4;
- named profiles: save as, apply ("switch"), rename, delete with confirmation;
- one profile is the **default**, applied at launch;
- **Reset to factory settings** with a confirmation;
- a profile holds every user-adjustable setting **except the scene's atoms**;
- versioned JSON, forward and backward compatible;
- corrupt profiles are quarantined;
- a File ▸ Profiles submenu and Ctrl-K entries.

### 15.1 What a profile holds

**Appendix A is the inventory.** Every field of `UiState` (main's `crates/app/src/ui/mod.rs`, the redesign) and every worker-side setting (`Command::SetAudit`, `SetDraft`, `SetCloud`, the `run_settings` commands, the backend) is listed with its profile key, its control, its factory default and how it is applied. Each setting falls into exactly one of three classes:

- **Profile settings:** view options (density display, display aids, smooth motion), units, layout, run settings (speed, dt, thermostat, COM correction, gravity, walls, drag pull), the solver, the estimated-mode options and the sampling-plan defaults. A profile stores a value for each.
- **Scene-carried settings:** grid n and h, σ, charge, multiplicity. These belong to the scene: scene files and presets carry them, and every scene load keeps doing so. A profile **may** set them. Its value is either `null` ("as the scene says", the factory value) or a concrete value applied to the current scene when the profile is applied, and at launch to the starting scene **[provisional, U1]**. A later scene or preset load uses the file's own values. A profile that pinned a grid would otherwise silently re-grid every scene the user opens.
- **Session-only, never in a profile:** the atoms (positions, velocities, isotopes: the scene); the scene path and preset name; the tool mode and the element on the Place tool (tool state); the camera; the log, the status line, the search state, Help and the armed confirmations (transient); `gpu_available` (hardware); the sampling seed, which advances after each pass so that "sample more" draws new geometries; and **draft mode** **[provisional, U2]**. The ruling's list names draft mode, but the M5b ruling made it "session only, off by default" (Model tab and Physics menu copy: the model answers beyond what it has learned). A profile that switched it on at launch would make a fresh session extrapolate without the user having asked in that session. Draft mode stays session-only and the conflict is flagged for the user. Adding it is one Appendix A row and one key.

**Nothing is missed.** `app::settings::classify(&UiState)` destructures `UiState` **exhaustively**, with no `..`. Every field must be named and routed to a profile key, a scene-carried key or the session list, so a field added later (M6 Task 10's chrome, or any future control) breaks the build until it is classified. Plan Task 8 pins this, the same acceptance-table approach the UI redesign used (mockup Appendix A).

### 15.2 Files

```text
<root>/
  user.json                          # {"format": "atomic-sandbox user 1", "default_profile": "teaching.json" | null, "last_scene": …}
  profiles/
    profiles.lock
    teaching.json                    # one file per profile; the name lives inside, the file name is a slug
    big-grid-cpu.json
    old-one.json.corrupt             # quarantined (§15.6), never deleted by the app
```

The file name is the slug of the profile name: lower-case ASCII letters and digits, other runs replaced by `-`, at most 40 characters, `-2`, `-3`, … on a collision. A name with no ASCII letters or digits becomes `profile`, then `profile-2`, and so on. Names are free text of 1–40 characters, unique case-insensitively.

A profile file is versioned JSON with flat dotted keys, through `userdata::json`:

```json
{"format": "atomic-sandbox profile", "version": 1, "name": "Teaching",
 "saved": 1792000000, "created_by": "atomic-sandbox 0.6.0",
 "settings": {"view.density_mode": "volume", "view.gamma": 0.5, "run.dt_fs": 0.25,
              "run.thermostat.on": true, "run.thermostat.t_k": 300.0, "scene.grid": null,
              "solver.backend": "estimated", "estimated.audit_every": 2000, "…": "…"}}
```

**Compatibility rules (forward and backward):**

- **A missing key** (an older file, or one edited by hand) takes the factory default. A note lists the keys that defaulted, once per apply.
- **An unknown key** (a newer version's setting, or a typo) is ignored with a warning naming it. It is **kept verbatim** when this build saves over that profile (Update), so a profile written by a newer build survives a round trip through an older one. Pinned: Review Focus P1.
- **A value of the wrong type or out of range** for a known key takes that key's factory default, with a warning naming the key, the value and the accepted range. The ranges are the controls' own (Appendix A). Only that key is affected; the rest of the profile still applies.
- **`version` greater than 1** is read under the same rules: known keys apply, unknown keys are ignored and kept. A future version that changes a known key's *meaning* must rename the key. That requirement is written into the module doc.
- **`version` missing or not a positive integer, `format` not the profile format, or JSON that does not parse:** the file is corrupt (§15.6).

### 15.3 Operations

| operation | effect | confirmation |
|---|---|---|
| **Save current settings as…** | a new profile from the current settings (`Settings::from_ui`) under a typed name. A checkbox, "include this scene's grid, σ, charge and multiplicity", is off by default: the scene-carried keys are saved as `null` unless it is ticked (§15.1). | an existing name asks "replace 'X'?" (two clicks, 4 s) |
| **Update 'X'** | rewrites the applied profile with the current settings (unknown keys kept) | none (the label names the profile) |
| **Apply** (switch to) 'X' | applies it (§15.4); the submenu marks it applied | only mid-pass when it would cancel the pass (§15.4) |
| **Rename…** | rewrites `name`, renames the file to the new slug, and moves the default pointer with it | none |
| **Set as default / No default** | writes `user.json`'s `default_profile` | none |
| **Delete…** | removes the file; if it was the default, also clears the pointer | two clicks (4 s): "Click again to delete 'X'"; for the default: "Click again to delete 'X' — the default profile; launch will use factory settings" |
| **Reset to factory settings…** | applies `Settings::factory()` like a profile (§15.4). **Profiles, the default pointer and the model library are untouched.** | two clicks (4 s): "Click again to reset every setting to factory defaults (profiles and models are kept; the default profile 'X' still applies at the next launch)" |

The submenu shows the applied profile with "(modified)" once any profile setting differs from it. The comparison is `Settings::diff(from_ui, profile)` and is computed when the menu opens, not per frame.

### 15.4 Applying a profile

`Settings::diff(current, target) -> Vec<Change>` lists every setting that differs. Each change records its key, old and new display values, and a class: **view** (applied at once, no pause), **run** (a `DynCommand`), **scene** (one `ReplaceScene` through the worker's `replace`), or **solver** (a worker re-spawn). Applying runs in this order:

1. **Nothing to do.** If the diff is empty, the result is "profile 'X' applied — no setting changed".
2. **Mid-pass check.** A pass is running and the diff contains a scene or solver change, both of which cancel a pass (§5.3). Without a confirmation the apply is refused: "applying 'X' changes the grid 49³ → 41³: the running sampling pass will be cancelled (its samples are kept) — click Apply again to confirm" (two clicks, 4 s). With it, the apply proceeds and the pass is cancelled into the old model entry (§5.3). Run-class and view-class changes never cancel a pass: dt, thermostat, walls and the rest affect the MD run, and a pass runs its own teacher.
3. **Pause.** If any run, scene or solver change is present, the run pauses first, as a scene edit does (rule 8 is untouched: `run.speed` is steps per tick and never changes dt).
4. **View changes** are applied to `UiState`.
5. **Run changes** go out as the existing commands, in `run_settings` order: `SetSpeed`, `SetComCorrection`, `SetDt`, `SetExternalField`, `SetThermostat`, `SetWalls`, `SetDragForce`, `SetAudit`, `SetCloud`. The walls' half-width `null` means the factory value derived from the box (`default_wall_half_width`). A concrete half-width is clamped to the current box's valid range, and the Log notes any clamp.
6. **Scene changes** are applied as one `ReplaceScene` after validation, exactly as **Apply grid** does: `validate_grid` (odd n in 17..=97, h in the panel's range) and the cell guard (every nucleus inside the new box by the R12 margin). σ, charge and multiplicity are checked as their own controls check them. **A refused scene change is not applied, and everything else in the profile still is.** The Log names it: "grid 99³ from profile 'X' not applied: n must be odd and 17..=97". Pinned: P2.
7. **The solver change comes last** (`Viewer::start`, the existing re-spawn with the current scene and run settings). The estimated worker resolves its model (§5.1).
8. **One Log entry and a toast** list exactly what changed, in Appendix A order: "profile 'Teaching' applied — dt 0.25 → 0.5 fs; thermostat off → on (300 K, τ 10 fs); grid 49³ → 41³; solver GPU → estimated — paused (the run's settings changed)". View-only changes are listed but say "not paused".

**A GPU that cannot fit the grid.** `validate_grid` bounds n at 97, and 97³ fits the RX 7900 XTX (24 GB) with room to spare. On a smaller adapter, a grid that passes validation can still fail to allocate. That failure takes the existing path: the provider error pauses the run and lands in `last_error`, as it does for a manual Apply grid. The Log line names the profile as the cause, and Undo (below) restores the previous grid in one click. Pinned: P2.

**Undo.** The toast of an apply that changed run, scene or solver settings offers **Undo** for 10 s. Undo applies the previous `Settings` through the same path, including the mid-pass check. It is one entry deep.

### 15.5 Launch

`main` builds the starting settings in this order:

1. `Settings::factory()`, the constants in code. `UiState::new` is built **from** `Settings::factory()`, so there is one source of truth (§15.8).
2. **The default profile.** `user.json`'s `default_profile`, if set and readable, is applied over the factory settings. Its scene-carried keys apply to the starting scene.
3. **The command line.** `--backend`, `--preset`, `--speed` and `--audit` win over the profile. The Log notes it: "--backend cpu overrides profile 'Teaching' (solver: estimated)".
4. **Optional, plan Task 10:** `launch.reopen_last_scene` (factory **off**). When on and no `--preset` is given, the app opens `user.json`'s `last_scene` instead of H₂: a scene file path, or `preset <name>`. `last_scene` is recorded on every Load scene, Save scene and preset choice. An unsaved edited scene is not reopened (there is no file to reopen). If the file is gone, the app starts on H₂ with a Log note.

The Log's first lines say what was applied: "settings: profile 'Teaching' (default)" / "settings: factory (no default profile)" / "settings: factory — the default profile 'teaching.json' was not found".

### 15.6 Errors

| case | what happens | the user sees |
|---|---|---|
| corrupt profile (unparsable, wrong `format`, bad `version`) | renamed to `<slug>.json.corrupt` (`.corrupt-<unix>` if taken); not applied: the current settings stay. At launch the factory settings are used | Log error + toast: "profile <path> is unreadable (<error>); kept as <path>.corrupt; factory settings used" |
| a profile from a newer version | known keys applied, unknown ignored and kept on Update (§15.2) | warning: "profile 'X' was saved by a newer version: ignored settings view.foo, run.bar (kept in the file)" |
| a bad value | that key takes its factory default | warning naming the key, the value and the range |
| the default profile missing (deleted by hand, or by another instance) | factory settings at launch; the pointer is left as is (the user may restore the file) | "settings: factory — the default profile 'teaching.json' was not found" (P4) |
| `user.json` corrupt | quarantined as `user.json.corrupt`; no default; no last scene | Log error naming it |
| a write fails (disk full, permissions) | the atomic write leaves the previous file intact | error toast naming the profile and the OS error; nothing changes in the menu |
| user data root unavailable | profiles disabled for the session (the submenu greys out with the reason); factory settings | as §14's library row |
| two instances save the same profile | last writer wins; each write is atomic | — (documented; profiles are explicit user actions) |

### 15.7 UI

The File menu gains **Profiles ▸**, in the mockup's menu style (`mh` section header, `mi` rows with a `faint` hint on the right):

```text
Profiles                                  <root>/profiles
  ● Teaching            ★ default          applied · Apply
    Big grid (CPU)                         Apply
  ─────────
  Save current settings as… [name____] [Save]
  Update 'Teaching'                        (modified)
  Rename 'Teaching'…  [name____]
  Set 'Teaching' as default | No default profile
  Delete 'Teaching'…                       two clicks within 4 s
  ─────────
  Reset to factory settings…               two clicks · profiles and models kept
```

Ctrl-K entries (`ui/search.rs`, static): "Profiles" (opens the submenu), "Save settings as profile", "Update profile", "Set default profile", "Reset to factory settings". Their keywords include "settings", "preferences", "defaults" and "factory". The Help ▸ Controls text gains one paragraph. The README's controls section gains the Profiles submenu and the `--data-dir` flag.

### 15.8 Code

- **`app::settings`:**
  - `Settings` holds every profile key as a typed field, with scene-carried keys as `Option`;
  - `Settings::factory()` reads the constants already in code: `DEFAULT_DT_FS`, `F_DRAG_MAX`, `SamplingPlan::default()`, `FallbackPolicy::default().audit_every`, and the literals now in `UiState::new`, which move here;
  - `Settings::from_ui(&UiState, &Scene, Backend)`;
  - `Settings::diff`;
  - `Settings::to_json` / `from_json(…) -> (Settings, Vec<Warning>)`;
  - `classify(&UiState)` (§15.1).
- **`app::profiles`:** `ProfileStore { dir }` with `list`, `read`, `save_as`, `update`, `rename`, `delete`, `default`, `set_default`, `last_scene`, `set_last_scene`, over `userdata::{atomic_write, Lock}`.
- **`app::ui::actions`:** `Action::{ApplyProfile, SaveProfileAs, UpdateProfile, RenameProfile, DeleteProfile, SetDefaultProfile, ResetFactory, UndoSettings}`. The apply path in §15.4 is `apply_settings(st, input, &Settings) -> PanelOutput`: the same commands the controls send, so the worker sees nothing new.
- **Physics: none.** Applying a profile is exactly the sequence of control changes a user could make by hand. Rule 8 holds: speed is steps per tick, never dt.

## 16. Review Focus: failure modes

Each mode is pinned to a test in the plan (plan § Review Focus).

- **F1 Two instances.** Lost samples or a torn index when two app instances, or two workers during a backend switch, use one library. The guards are leases, the index lock with read-merge-write, and merge on late lease acquisition.
- **F2 Key drift from float formatting.** The same scene keys differently across sessions ("0.25" vs "0.250", σ `None` vs `Some(h)`), or the key equates two surfaces the gate refuses. The guards are bit-exact `{:?}` text, the pinned canonical string and hash, and a property test of key equality ⇔ gate acceptance.
- **F3 A scene switch mid-pass.** The pass's samples land in the wrong entry, or the pass keeps running on the new key. The guard: the key-changing edit cancels the pass into the old entry, and a non-key edit leaves the pass running.
- **F4 An autosave racing a refit.** A file holds a half-applied refit, or a save writes an older state over a newer one. The guards: encode and refit share the worker thread, `saved_revision` records what was encoded, and the debounce.
- **F5 A renamed scene file.** The model is lost because the path changed, or a sidecar is imported twice. The guards: the key is not the path, imports are recorded by (path, size, mtime), and the merge de-duplicates bitwise.
- **F6 (added) The restart.** Reopening the app loses samples, refits, or changes the trust state (§5.4).
- **P1 A profile from a newer version.** It is refused as corrupt, its unknown settings are lost when it is updated, or a known key with a changed meaning is misread. The guards: version > 1 is read under the same rules, unknown keys are ignored with a warning and kept verbatim on Update, and a meaning change requires a new key name (§15.2).
- **P2 A profile naming a grid the GPU cannot fit**, or an invalid or too-small grid for the scene. The failure modes are a panic, a silent half-applied profile, or nuclei outside the box. The guards: the scene-carried keys are validated as Apply grid is (`validate_grid` and the cell guard), a refused key is reported while the rest still applies, an allocation failure takes the existing provider-error path naming the profile, and Undo restores the previous grid (§15.4).
- **P3 Switching profile mid-pass.** A grid or solver change silently cancels a pass, or its samples land in the wrong model entry. The guards: the two-click mid-pass confirmation, and the cancel into the old entry (§5.3, §15.4). Run-class changes leave the pass running.
- **P4 The default profile deleted** (in the UI, by hand, or by another instance). The failure modes are a launch failure, a stale pointer applied, or silent factory settings. The guards: deleting the default in the UI clears the pointer and says so, and a missing file at launch gives factory settings with a Log line naming it (§15.6).

## 17. Validation

This milestone does not touch solver or dynamics code. It touches the surrogate's store (a tag and its bookkeeping) and the sidecar, so **rule 6 applies to the surrogate suite**: `cargo test -p validate --release --test surrogate` must reproduce the M6 numbers to the printed digits (plan Task 4). The audit tag is bookkeeping, and the bit-identical-fit unit test is its guard. Everything else is unit- and worker-tested:

- `cargo test -p surrogate` (format, key, library);
- `cargo test -p providers --release` (records, merge);
- `cargo test -p app --release` (worker, viewer, UI).

Timings (save cost, switch cost) are reported, never asserted.

Part B is pure app code. Its tests are `cargo test -p userdata` (atomic write, lock, JSON, time) and `cargo test -p app --release` (`settings`, `profiles`, the apply path through `ui::actions`, launch). Applying a profile sends the same commands as the controls, so the dynamics suite is not affected (rule 6 has nothing new to gate). A test (plan Task 9) pins that a profile changing dt and the thermostat produces exactly the `DynCommand`s the controls produce.

## 18. Provisional rulings (made here, flagged for the user)

- **L1** The user-data root defaults to `%LOCALAPPDATA%\atomic-sandbox\`, not `%APPDATA%` (§4.1): unbounded size, machine-local, not roamed.
- **L2** `--screenshot` runs use a fresh temp root (factory settings, empty library) unless `--data-dir` is given (§4.1).
- **L3** A key switch caused by an edit offers the pass and never starts it. Entering estimated mode keeps today's automatic first pass (§5.3).
- **L4** Import into an existing same-key entry pools samples: the fitted side's prefix is kept, the rest is appended as pending, quarantine flags are dropped for re-judgement, and references must agree bit-exactly (§8.3). The alternative, asking the user to choose replace or add, was rejected: pooling is what a key means.
- **L5** The provider stamps dates with `SystemTime::now()` (display data, never physics). Tests assert order and presence, not values.
- **L6** The Library browser lives in the Model tab, not a new window (§11).
- **U1** Scene-carried settings (grid, σ, charge, multiplicity) default to `null`, meaning "as the scene says". A profile value applies to the current scene when the profile is applied and to the starting scene at launch. Later scene loads keep their files' values, and Reset to factory settings does not re-grid the scene (§15.1).
- **U2** Draft mode is **not** in profiles. The ruling's list names it, but M5b made it session-only and off by default. Flagged for the user; adding it is one row (§15.1).
- **U3** Applying a profile offers a one-deep Undo for 10 s (§15.4).
- **U4** Command-line flags win over the default profile, and the Log says so (§15.5).
- **U5** Profiles have no leases. Two instances saving the same profile resolve to last-writer-wins, with atomic writes (§4.4, §15.6).

## 19. References

- Ruling and user words: `docs/superpowers/notes/2026-10-13-m5-user-requests.md` § "Model library: automatic per-scene models".
- Provenance and refusal: `crates/surrogate/src/model.rs` (`Provenance`, `describe`, `describe_recorded`, `record_masses`); `crates/providers/src/surrogate/mod.rs` (`Trainable::load`); `crates/sim/src/dynamics.rs` module doc ("SaveModel… LoadModel").
- Sidecar v3: `crates/surrogate/src/sidecar.rs` module doc (format, validation, atomic write).
- Today's session autosave: `crates/app/src/worker/{mod.rs ("Autosave"), autosave.rs, model.rs (ensure_model)}`; `crates/app/src/viewer.rs` (`autosave_path`, `Viewer::start`, `RETIRE_WAIT`); `crates/app/src/main.rs` (exit).
- The Model tab: `crates/app/src/ui/dock/model.rs`; the mockup's Model tab and Appendix A rows 10–45 (`docs/superpowers/notes/2026-10-14-ui-mockup.html`).
- FNV-1a: Fowler–Noll–Vo, 64-bit parameters (offset basis 0xcbf29ce484222325, prime 0x100000001b3); the published vector FNV-1a("a") = 0xaf63dc4c8601ec8c.
- File locks: `std::fs::File::{lock, try_lock}` (Rust 1.89); Windows `LockFileEx`, `MoveFileExW`.
- Settings sources: `crates/app/src/ui/{mod.rs (UiState, UiState::new), menus.rs, toolbar.rs, dock/*.rs}` (main); `crates/app/src/worker/commands.rs` (m6-tools); the mockup's menu style and Appendix A acceptance-table method (`docs/superpowers/notes/2026-10-14-ui-mockup.html`).

## Appendix A — The settings inventory (every user-adjustable setting, with its factory default)

**Sources:** `crates/app/src/ui/mod.rs` (`UiState`, `UiState::new`), `ui/menus.rs`, `ui/toolbar.rs`, `ui/dock/{model,selection,graph,log}.rs` on main (the redesign); `crates/app/src/worker/{commands.rs, mod.rs}` on `m6-tools` (`Command::SetAudit/SetDraft/SetCloud`, `run_settings`, the backend); `crates/sim/src/{dynamics.rs, sampling.rs}` (`DEFAULT_DT_FS`, `F_DRAG_MAX`, `MAX_SPEED`, `SamplingPlan::default`).

**Acceptance:** every row exists in `Settings` with this key, this factory value and this range. Plan Task 8's `factory_settings_match_the_shipped_constants` pins the values. `classify` destructures `UiState` exhaustively (§15.1). The controls M6 Task 10 adds (bond lines, molecule labels, the isotope picker) are classified when this milestone starts: a new view toggle becomes a `view.*` row here, and anything else goes to the session list. Plan Task 8 Step 0 re-reads `UiState` on the post-M6 main and amends this table before writing the test.

### A.1 Profile settings

| # | key | `UiState` field / where | control | factory | range / values | class |
|---|---|---|---|---|---|---|
| 1 | `view.density_mode` | `volume_mode` (0 / 1) | View ▸ density: volume / isosurface | `"volume"` | `"volume"`, `"isosurface"` | view |
| 2 | `view.gamma` | `gamma` | View ▸ gamma slider | 0.5 | 0.2–1.0 | view |
| 3 | `view.opacity` | `opacity` | View ▸ opacity slider (log) | 2.0 | 0.1–20 | view |
| 4 | `view.iso` | `iso` | View ▸ iso (× ρ max) | 0.1 | 0.01–0.9 | view |
| 5 | `view.show_cell` | `show_cell` | View ▸ display aids: cell | true | bool | view |
| 6 | `view.show_floor` | `show_floor` | … floor grid | true | bool | view |
| 7 | `view.show_axes` | `show_axes` | … axes | true | bool | view |
| 8 | `view.show_place_grid` | `show_place_grid` | … placement plane | true | bool | view |
| 9 | `view.grid_spacing` | `grid_in_angstrom` | … 1 Bohr / 1 Å | `"bohr"` | `"bohr"`, `"angstrom"` | view |
| 10 | `view.smooth_motion` | `smooth_motion` | View ▸ smooth motion | true | bool | view |
| 11 | `view.show_atom_card` | `show_atom_card` | Selection tab: show the card | true | bool | view |
| 12 | `view.energy_units` | `energy_units` | Readouts: Ha / eV / both | `"both"` | `"ha"`, `"ev"`, `"both"` | view |
| 13 | `view.graph_series` | `graph_series` | Graph tab: curves | `"both"` | `"both"`, `"energy"`, `"temperature"` | view |
| 14 | `view.graph_all` | `graph_all` | Graph tab: all points | false | bool | view |
| 15 | `view.log_filter` | `log_filter` | Log tab filter | `"all"` | `"all"`, `"errors"`, `"refused"`, `"warnings"`, `"notes"`, `"status"` | view |
| 16 | `layout.dock_tab` | `dock_tab` | the dock's tab | `"readouts"` | `"readouts"`, `"graph"`, `"model"`, `"selection"`, `"log"` | view |
| 17 | `layout.dock_open` | `dock_open` | dock open / icon strip | true | bool | view |
| 18 | `run.speed` | `speed` | toolbar speed slider / field / watchable preset (steps per tick, never dt — rule 8) | 1 | 1–`MAX_SPEED` (1000) | run |
| 19 | `run.dt_fs` | `dt_fs` | Simulation ▸ dt | 0.25 (`DEFAULT_DT_FS`) | 0.01–2.0 fs | run |
| 20 | `run.thermostat.on` | `thermostat_on` | Simulation ▸ thermostat (Bussi CSVR) | false | bool | run |
| 21 | `run.thermostat.t_k` | `thermostat_t` | … target | 300 | 0–5000 K | run |
| 22 | `run.thermostat.tau_fs` | `thermostat_tau` | … τ | 10 | 1–1000 fs | run |
| 23 | `run.com_correction` | `com_correction` | Simulation ▸ COM correction | true | bool | run |
| 24 | `run.gravity` | `gravity` | Simulation ▸ gravity (the field hook, g along −y) | false | bool | run |
| 25 | `run.walls.on` | `walls_on` | Simulation ▸ container walls | false | bool | run |
| 26 | `run.walls.half_width_bohr` | `wall_half_width` | … half-width slider | `null` = derived from the box (`default_wall_half_width`: (n−1)h/2 − 1.5 Bohr, clamped) | `null` or the box's valid range (clamped on apply, noted) | run |
| 27 | `run.drag_pull` | `drag_force` | Simulation ▸ drag pull; the wheel while dragging | 0.1 Ha/Bohr (`F_DRAG_MAX`) | `F_DRAG_LO`–`F_DRAG_HI` (0.01–1.0) | run |
| 28 | `solver.backend` | `backend` | toolbar Solver: CPU / GPU / Estimated | `"gpu"` (`main`'s default; CPU without an adapter, as today) | `"cpu"`, `"gpu"`, `"estimated"` | solver |
| 29 | `estimated.audit_every` | `audit_every` | Model ▸ Audits: DFT audit every n | 2000 (`FallbackPolicy::default().audit_every`) | 0–1 000 000 (0 warns "unsafe" on apply, as the control does) | run |
| 30 | `estimated.cloud_superposition` | `cloud_superposition` | Physics ▸ Estimated: atomic superposition | true | bool | run |
| 31 | `estimated.cloud_refresh` | `cloud_refresh` | … refresh cloud every | false | bool | run |
| 32 | `estimated.cloud_refresh_s` | `cloud_refresh_s` | … every N s | 2.0 | 1–10 s | run |
| 33 | `sampling.perturbations` | `plan.perturbations` | Model ▸ Sampling plan | 60 | 0–2000 | view (the next pass's plan) |
| 34 | `sampling.sigma_r_bohr` | `plan.perturbation_sigma` | … σ_r | 0.15 | 0–0.5 Bohr | view |
| 35 | `sampling.burst_steps` | `plan.hot_burst_steps` | … hot burst steps | 200 | 0–5000 | view |
| 36 | `sampling.burst_t_k` | `plan.hot_burst_t` | … hot burst T | 2000 | 0–10 000 K | view |
| 37 | `sampling.scan_points` | `plan.scan_points_per_pair` | … scan points / pair | 12 | 0–200 | view |
| 38 | `launch.reopen_last_scene` | — (new, plan Task 10) | File ▸ Profiles: reopen the last scene on launch | **false** | bool | launch only |

### A.2 Scene-carried settings (§15.1: `null` = the scene's own; applied to the current scene, validated as their controls)

| # | key | where | control | factory | values |
|---|---|---|---|---|---|
| 39 | `scene.grid` | `Scene::{grid_n, grid_h}` (`UiState::{grid_n, grid_h}` is the pending edit) | Physics ▸ grid n, h, Apply grid | `null` | `null` or `{"n": odd 17–97, "h": 0.1–0.5}` (`validate_grid`) + the cell guard |
| 40 | `scene.sigma` | `Scene::sigma` | Physics ▸ σ (auto = h) | `null` | `null`, `"auto"` (= h), or 0.05–1.0 Bohr |
| 41 | `scene.charge` | `Scene::charge` | Physics ▸ charge | `null` | `null` or an integer in −4 … (protons − 1) of the current scene (checked on apply) |
| 42 | `scene.multiplicity` | `Scene::multiplicity` | Physics ▸ multiplicity (auto) | `null` | `null`, `"auto"`, or 1–11 (`allowed_multiplicities` checked on apply) |

### A.3 Session-only (never in a profile)

| `UiState` field / setting | why |
|---|---|
| the atoms (`Scene::nuclei`: positions, velocities, isotopes) | the scene, by the ruling |
| `scene_path`, `preset` | the scene's identity (`last_scene` is in `user.json`, §15.5) |
| `element`, `mode` | tool state |
| `grid_n`, `grid_h` | the pending Apply-grid edit (the profile's grid is row 39) |
| `plan.seed`, `plan.scan_range` | the seed advances after each pass; the scan range has no control |
| `draft` | M5b: session only, off by default **[provisional, U2]** |
| `status`, `status_seq`, `log`, `search`, `help`, `queued`, `forget_armed`, `start_armed` | transient |
| `gpu_available` | hardware |
| the camera (`Viewer::camera`), the viewport | per view |
| the estimated solver's teacher | not a setting: the GPU when an adapter exists, else the CPU (it is part of the model key, §3.1) |
