# User data: model library and settings profiles — Design Spec

**Date:** 2026-10-15 (written 2026-10-14, for execution after M6)
**Status:** Specified from the user's requests and the controller's binding rulings: the model library first, then, the same day, settings profiles in the same per-user data folder. The decisions this spec had to make on its own are marked **[provisional]** and listed in §18.
**Parts:** A — the model library (§1–§3, §5–§14); B — settings profiles (§15, with the settings inventory in Appendix A); C — the shared user-data machinery (§4: folder, atomic writes, locks), which both use. Review focus §16, validation §17.
**Author:** Claude (Opus 5.5) with Dakoda
**Parent:** `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` (§5 architecture, §8 tooling); `docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md` (provenance and the refusal rule, S9 references, S15 quarantine, S18 sample tolerance); `docs/superpowers/specs/2026-10-14-m6-tools-design.md` (the `providers` crate, `Trainable`, the worker split, the sidecar v3 mass record — ruling P2).
**Inputs:** `docs/superpowers/notes/2026-10-13-m5-user-requests.md` § "Model library: automatic per-scene models" (the binding ruling); the user's follow-up (2026-10-14): "those models will auto save to files? then auto load when the app is reopened?" (answered by §5.4 and pinned in plan Task 7a); the user's request (2026-10-14) "would it be possible to have settings profiles too? to be able to set a default profile, as well as reset to out of the box settings" and the controller's ruling on it (Part B, §15); `docs/superpowers/notes/2026-10-14-m5-handoff.md` (session autosave with ownership); `docs/superpowers/notes/2026-10-14-ui-mockup.html` (the Model tab).

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
- **No change to the physics.** The fit, trust rule, fallback policy, quarantine and tolerances are untouched. The new sample tag (`Audit`) is bookkeeping, and the fit does not read tags (§7.2; pinned bit-exact in plan Task 4a).
- **The app does not reopen the last scene on launch by default.** It starts on H₂ (or `--preset`/`--scene`). So "auto load when reopened" means: when the user returns to a scene they sampled, by loading it, choosing its preset or building it, its model comes back. Reopening the last scene is a **required** task (plan Task 10, review M4): a profile option `launch.reopen_last_scene`, **off by default** (§15.5, Appendix A).

## 3. The key

### 3.1 Fields

A model is valid for any scene whose **library key** equals its own. The key is the M5 provenance gate plus two additions. The gate is `surrogate::Provenance` equality, which the loader enforces; this milestone extends it with `teacher_rev` (review M3). The second addition is the scene's species set.

| field | source | canonical text |
|---|---|---|
| species set | the scene's nuclei: sorted unique Z | `species 1,8` |
| charge | `Scene::charge` | `charge 0` (`charge -1`, `charge 1`) |
| multiplicity | `Scene::multiplicity` | `multiplicity lowest` for `None`, else `multiplicity 3` |
| grid n | `Cell::n` | `grid 49` |
| h | `Cell::h` (Bohr), snapped (§3.2) | `h 0.25` |
| σ | `Scene::sigma.unwrap_or(h)`, as `scene_provenance` takes it, snapped | `sigma 0.25` |
| teacher | `ForceProvider::name()` of the surrogate's teacher | `teacher Kohn-Sham (GPU)` |
| teacher revision | `providers::ks::TEACHER_REV` (a `u32`, 1 at this milestone), carried in `Provenance` and compared by the gate | `teacher_rev 1` |
| teacher tolerance | the surrogate's `teacher_tol` (`SAMPLE_TOL` = 1e-6) | `tol 1e-6` |

**What the key leaves out.** The key does not depend on geometry, atom count, velocities, the external field, the walls or the thermostat. The field and the walls are driver-side potentials outside the Kohn-Sham surface (`sim::Dynamics::field_energy`/`walls_energy`).

**Isotopes are recorded, not keyed** (ruling P2: the Born-Oppenheimer surface is mass-independent). The mass set goes into the file's provenance record (`Provenance::record_masses`). So H₂ and D₂ share one model, and the Model tab shows "masses ¹H ²H (recorded, not compared)".

**Multiplicity `None` and `Some(m)` are different keys, even when m is the lowest multiplicity** (review N1). This is correct. `None` means "the lowest found by the solver's scan", and that choice can differ from a pinned m at another geometry. The gate already distinguishes them.

**Why the species set is in the key when the provenance gate leaves it out.** The binding ruling puts it there. Without it, a model taught on H would also be the water model. It would answer nothing on water: rule 2 of the decision list sends every step with a new species to the teacher and marks the basis for a rebuild. That pooling is legal, but it is the wrong default for a library: a user who builds water from H₂ would grow the H₂ model into a water model and lose the clean H₂ one. Separate keys keep each species set's model whole. A scene that later loses its O (water → remove O → H₂) switches back to the H model.

**The teacher revision (review M3, adopted).** The key cannot see a code change. If a solver change moves the teacher's energies under an unchanged name, old models would load on a new surface. Three guards close that gap:

1. **`TEACHER_REV`.** `providers::ks::TEACHER_REV: u32` is part of `Provenance` (gate) and of the key. Files of v1–v3 read as revision 1.
2. **A tripwire in `validate`.** `teacher_rev_tripwire`: the CPU f64 teacher on H₂ at 33³, h 0.25, R = 1.4 Bohr, tol 1e-10. The energy is pinned to about 1e-10 Ha, measured once and recorded in `references.md`. Any change to the solver's output fails it with "solver output changed: bump TEACHER_REV and add a references.md row". The GPU teacher shares the revision, and its numerics are gated by the `gpu_vs_cpu` suite.
3. **The pseudopotential milestone will bump it to 2.** Old models then stay in the library, inert, under their own keys. The browser shows why: "teacher rev 1 → 2".

The file also records `created_by` (the build).

### 3.2 Canonical form

The canonical key is UTF-8 text with one field per line, in the order of the table, under a version line:

```text
atomic-sandbox model key v1
species 1
charge 0
multiplicity lowest
grid 49
h 0.25
sigma 0.25
teacher Kohn-Sham (GPU)
teacher_rev 1
tol 1e-6
```

Each line ends with `\n`, including the last.

- **Floats (h, σ, tol) are written with Rust's `{:?}`**, the shortest decimal that round-trips to the same f64. The text is therefore **bit-exact**: two keys are equal exactly when the f64s are equal, which is exactly when the provenance gate (`Provenance::eq`, f64 `==`) accepts. A rounded key (`{:.4}`) would map two surfaces the gate refuses to one file, and every load would then fail. **Key and gate agree by construction; plan Task 3 pins it** (§16, F2).
- **The inputs are canonicalised at the source (review M2).**
  - **The problem.** Bit-exactness alone would turn input noise into different keys. The σ DragValue (`menus.rs:262`, no `fixed_decimals`) can produce 0.30000000000000004 through egui's `smart_aim`, while a scene file holds an exact 0.3. That gives two keys, and a gate refusal between them.
  - **Snapping.** Every h and σ that enters a scene is passed through `app::scene::snap_length(x) = format!("{:.6}", x).parse::<f64>()`. The entry points are `SetGrid`, `SetSigma`, `SetSceneSettings` (§5.1), the scene-file parser, presets, and profile apply (§15.4). Snapping happens before the value reaches `sim::Scene`, so the gate, the key and the solver all see the same f64.
  - **The σ control** additionally gets `fixed_decimals(3)`, like the h control.
  - **Pinned:** σ = `3.0 * 0.1` (0.30000000000000004) snapped keys the same as σ = 0.3. On H, 49³, h 0.25, GPU, the stem is `da3f9c7f1eb4da29`.
  - **Old files.** A model file from before this milestone whose h or σ was not snapped keeps its exact value. It lands under its own key and is inert, which is honest because it was taught on that exact surface. The Library section shows its full-precision label.
- `LibraryKey::new` refuses a non-finite or non-positive h, σ or tol, a teacher name containing `\n`, an empty species set and Z = 0. The scene validator already rules these out, so a refusal is a bug, reported as an error and never keyed.
- −0.0 cannot occur: h, σ and tol are positive.

### 3.3 Hashing and file names

The file name is `<h>.model`, where `<h>` is the 64-bit **FNV-1a** of the canonical text, written as 16 lower-case hex digits. The function is hand-written in `userdata::fnv1a_64` (offset basis 0xcbf29ce484222325, prime 0x100000001b3), so it is the same on every build and platform. **`std::collections::hash_map::DefaultHasher` is forbidden here**, because its output is not stable across Rust releases. Pinned values, computed independently and checked in plan Task 3:

| key | FNV-1a |
|---|---|
| H, charge 0, lowest, 49³, h 0.25, σ 0.25, `Kohn-Sham (GPU)`, rev 1, 1e-6 | `57c4698cfdbc5bed` |
| H O, 0, lowest, 57³, 0.25, 0.25, `Kohn-Sham (GPU)`, 1, 1e-6 | `1ac0fa723264d7b6` |
| H, 0, lowest, 49³, 0.25, 0.25, `Kohn-Sham (CPU)`, 1, 1e-6 | `11c34fbecf89d989` |
| H, 0, lowest, 49³, h = σ = 0.25000000000000006, GPU, 1, 1e-6 (a hand-built provenance: snapping keeps it out of the UI) | `9bb347cfebec92e5` |
| H, 0, lowest, 49³, h 0.25, σ 0.3 (or `3.0 * 0.1` snapped), GPU, 1, 1e-6 | `da3f9c7f1eb4da29` |
| H, 0, lowest, 49³, 0.25, 0.25, GPU, **rev 2**, 1e-6 | `246d60367e8caed8` |
| H, 0, lowest, 41³, 0.25, 0.25, `Kohn-Sham (GPU)`, 1, 1e-6 | `c26f754910eebd85` |
| H, 0, lowest, 33³, 0.25, 0.25, `morse dimer`, 1, 1e-6 (the worker tests' key) | `182776be1d591cdc` |
| …the same at 35³ | `0b5242d8bb730026` |
| FNV-1a of the empty string / of `a` (the published test vectors) | `cbf29ce484222325` / `af63dc4c8601ec8c` |

**The hash only names files. It never identifies a model.** Every v4 file stores its full canonical key in its header (§7.1), and every lookup compares that text. If a file named `<h>.model` holds another key (a hash collision, or a file copied in by hand), the new entry is written as `<h>-2.model` (then `-3`, …), and the index maps the key to that name. **Every entry's lease file is named after its model file:** `<h>-2.model` has the lease `<h>-2.lock` (review S7).

### 3.4 Human label and the reasons for a switch

`LibraryKey::label()` is display text. It is never parsed and never keyed:

```text
{species symbols, by Z}[ · charge {+c}][ · mult {m}] · {n}³ · {h} Bohr[ σ {σ}] · {teacher}[ rev {r}] {tol:e}
```

- σ is shown only when σ ≠ h, the charge only when it is not 0, the multiplicity only when it is set, and the revision only when it is not 1.
- Teacher `Kohn-Sham (X)` shows as `X`; any other teacher shows its full name.
- **h and σ print with `{}`**, which for an f64 is the shortest round-trip decimal. Two different keys therefore never share a label (review M2): a snapped value prints as typed (`0.25`, `0.3`, `0.123457`), and an unsnapped old file prints its full digits (`0.30000000000000004`).
- Symbols come from `units::isotopes::symbol` (display data).

Pinned examples:
- `H · 49³ · 0.25 Bohr · GPU 1e-6`
- `H O · 57³ · 0.25 Bohr · GPU 1e-6`
- `H · 33³ · 0.25 Bohr σ 0.3 · morse dimer 1e-6`
- `H · charge +1 · mult 3 · 33³ · 0.25 Bohr · morse dimer 1e-6`
- `H · 49³ · 0.25 Bohr · GPU rev 2 1e-6`

`LibraryKey::diff(&old, &new) -> Vec<String>` names the fields that changed, in table order. Its phrases are:
- `species H → H O`
- `charge 0 → +1`
- `multiplicity lowest → 3`
- `grid 49³ → 57³`
- `h 0.25 → 0.3 Bohr`
- `σ 0.25 → 0.3 Bohr`
- `teacher GPU → CPU`
- `teacher rev 1 → 2`
- `tolerance 1e-5 → 1e-6`

The switch announcement (§5.3) and the browser's explanation (§11) use it.

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
  57c4698cfdbc5bed.model   # one sidecar (v4, §7) per key
  57c4698cfdbc5bed.lock    # the entry's lease (§4.4); never deleted
  1ac0fa723264d7b6.model
  …
  182776be1d591cdc.model.corrupt   # an unreadable file, set aside (§14), never deleted by the app
```

**The `.model` files are the truth and `index.json` is a cache.** Every v4 header carries the canonical key and the record the index shows (§7.1). If the index is missing or unreadable, or disagrees with the directory, it is rebuilt from the headers, which is a header-only read (`sidecar::read_header`). On open, the library compares the index with the directory listing (names, sizes, mtimes) and re-reads only the files that differ. Only the "scenes used on" lists live in the index alone, so a rebuild loses them and says so in the Log.

`index.json` (hand-written JSON through the sidecar's own reader and writer, moved to `userdata::json`):

```json
{"format": "atomic-sandbox model library 1",
 "entries": [
  {"file": "57c4698cfdbc5bed.model",
   "key": "atomic-sandbox model key v1\nspecies 1\n…tol 1e-6\n",
   "label": "H · 49³ · 0.25 Bohr · GPU 1e-6",
   "created": 1792000000, "last_trained": 1792003600,
   "samples": 412, "quarantined": 2, "fitted": true, "passes": 3,
   "size": 183552, "modified": 1792003605,
   "scenes": [{"name": "preset h2", "last_used": 1792003600},
              {"name": "D:\\runs\\h2-stretch.scene", "last_used": 1791990000}],
   "state": "ok"}],
 "imported": [{"path": "D:\\runs\\h2.model", "size": 183552, "modified": 1791900000,
               "into": "57c4698cfdbc5bed.model", "at": 1792000000}]}
```

- Times are Unix seconds, UTC. They are displayed as `2026-10-14 17:46 UTC` by a hand-written civil-from-days conversion (`userdata::time::utc_label`, pinned: 0 → `1970-01-01 00:00 UTC`, 951782400 → `2000-02-29 00:00 UTC`, 1792000000 → `2026-10-14 17:46 UTC`) and as a relative age ("3 h ago").
- `scenes` holds the ten most recently used, newest first. A name is `preset <name>`, the scene file's path, or `unsaved (<formula>)` using the M6 `app::names` formula.
- `state` is `ok`, `corrupt` (the `.corrupt` file's path is kept in `"corrupt_file"`) or `newer` (a format this build cannot read, §14).
- `imported` records each migrated or imported file by (path, size, mtime), so it is imported once (§8).

### 4.3 Atomic writes

Every write goes through `userdata::atomic_write`. That covers a `.model`, `index.json`, a profile and `user.json`. The steps:

1. **Write a temp file.** The bytes go to `<name>.<pid>-<seq>.tmp` in the same directory. Two instances never share a temp file.
2. **`File::sync_all` the temp file.**
3. **`std::fs::rename` it onto `<name>`.**
   - **On Windows,** Rust's `fs::rename` first tries `SetFileInformationByHandle(FileRenameInfoEx, FILE_RENAME_FLAG_POSIX_SEMANTICS | FILE_RENAME_FLAG_REPLACE_IF_EXISTS)`. That replaces the file even while another process has it open with delete-sharing. If the volume does not support it, Rust falls back to `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`.
   - **Retry rule.** A rename that fails with **OS error 5 (access denied) or 32 (sharing violation)** is retried 5 times over 500 ms. These are typically an antivirus scanner or the search indexer holding the destination (review S5). Any other error fails at once.
   - **On Unix,** the directory is `fsync`ed after the rename, so the rename itself survives a power loss.
4. **On failure, remove the temp file.** The previous file is intact; old bytes are never truncated in place.

At open, the library removes `*.tmp` files more than 24 h old. They are left by crashed writes, match the pattern above, and are never a user's file.

### 4.4 Two instances: the index lock, the entry lease, and read errors

Two app instances can touch the same library at once. So can two workers of one instance during a backend switch: the old worker retires for up to `RETIRE_WAIT` = 60 s while the new one starts. They coordinate through `std::fs::File` locks (`try_lock`, stable since Rust 1.89; this toolchain is 1.99). On Windows these are `LockFileEx` locks, which the OS releases when the process dies, so **a crash never leaves a stale lock**. A killed child process pins this (§16, F7).

- **`index.lock`** is held only while the index is read, changed and written back (milliseconds).
  - The writer re-reads `index.json` under the lock and changes only its own entries before writing, so two instances' updates both survive.
  - Acquisition polls `try_lock` for up to 2 s. If that fails, the index update is skipped and logged. The index is a cache (§4.2), so the next update or open repairs it.
- **The entry lease, `<file stem>.lock`,** is taken (`try_lock`, exclusive) when an entry becomes the worker's active model. It is held until the worker switches away or retires.
  - **Only the lease holder writes, renames or quarantines that entry's files.** Everyone else only reads them.
  - A worker that cannot get the lease opens the entry **read-only**: it loads a copy, trains in memory and never writes. This is announced: "model 'H · 49³ …' is in use by another instance or worker: this session trains a copy that is not saved — Export to keep it".
  - It retries the lease every `LEASE_RETRY` = 30 s. Once it holds the lease, it **merges** the file's samples it does not hold into its own (the file-level merge of §8.3), writes the union, reloads it and owns the entry. A retiring worker that times out and finishes later therefore hands its samples over, with nothing lost and nothing duplicated.
  - Within one viewer, the next estimated worker's `EnsureModel` is held until the retiring workers have finished (`pending_teach`, kept from today; review S3). A quick solver toggle therefore does not sit read-only for 30 s.
- **Read errors are not corruption (review M1).** Reading an entry can fail in two ways:
  - **`InvalidData`** (a bad magic, header or body, as the sidecar reader reports it) is corruption. The **lease holder alone** sets the file aside (§14).
  - **Any other error** is not: a sharing violation (OS error 32), `PermissionDenied` (5, typically from an antivirus scanner or the indexer), or a transient I/O error. The entry opens **read-only, empty**, with the note "model file <path> could not be read (<error>); retrying — nothing is written until it is read", and the read is retried every `LEASE_RETRY`. **It never becomes `Fresh` and the file is never overwritten.** Once the read succeeds, the worker merges any samples it gathered in the meantime, exactly as for a late lease.
- **Deleting needs the lease.** Forget and the browser's Delete rename the entry to `.forgotten-<unix>` (§5.5). An entry leased elsewhere cannot be forgotten, and the UI says why.
- **Profiles** use one short lock, `<root>/profiles/profiles.lock`, held while a profile file or `user.json` is written or deleted (the same 2 s `try_lock` loop). There are no leases: a profile is written only by an explicit user action, so two instances saving the same profile resolve to last-writer-wins. Each write is atomic, so neither leaves a torn file (§15.6).

## 5. Lifecycle

### 5.1 When the key is resolved

The worker owns resolution. It recomputes the scene's key after every applied command that can change it. Each command below is a worker `Command` that exists today (`app::worker::commands`):

- `EnsureModel` (sent when estimated mode is entered);
- `SetScene` (a scene file or preset load);
- `SetCharge`, `SetMultiplicity`, `SetGrid`, `SetSigma`, each of which goes through `Worker::replace` → `DynCommand::ReplaceScene`;
- `AddNucleus` with a new Z;
- `RemoveNucleus` that removes the last nucleus of a Z;
- **`SetSceneSettings { n, h, sigma, charge, multiplicity }`**, new with this milestone (review M6).

**Why `SetSceneSettings`.** It is **one** `ReplaceScene`, validated jointly and in dependency order:

1. the grid (`validate_grid`, then the cell guard: every nucleus inside the new box by the R12 margin);
2. σ;
3. the charge (its range depends on the proton count);
4. the multiplicity (`allowed_multiplicities` for the electron count after the charge).

A refusal names the first failing field and changes nothing. The UI's Physics controls keep their single-field commands. Profile apply (§15.4) and Undo use `SetSceneSettings`, so a profile that changes grid, σ and charge causes **one** key switch, one toast and one resolve, not four. Every field is `Option` (`None` = unchanged). h and σ are snapped (§3.2).

A teacher change cannot happen within a worker: a backend switch spawns a new worker, which resolves at its own `EnsureModel`.

**Scene names.** The worker also keeps the scene's name for the index through `Command::SetSceneName(String)`. The viewer sends it on Load / Save scene, on a preset choice, and with `unsaved (<formula>)` after the first edit of a named scene.

**Scene sidecars.** When the scene has a path, `EnsureModel` also imports `model_path(scene_path)` on first sight (§8.2; review S6).

### 5.2 Resolve → load or create

`resolve(key)` runs these steps:

1. **Index lookup** by canonical key. If the index is stale, the lookup falls back to a directory scan.
2. **Found.**
   1. Take the lease. If that fails, open read-only (§4.4).
   2. Read the file:
      - **`InvalidData`** → §14: the lease holder sets the file aside and starts fresh.
      - **`Unsupported`** (a newer format) → §14: read-only, untouched.
      - **Any other read error** → read-only, empty, retried (§4.4, review M1). **Never `Fresh`.**
   3. Check that the header's key equals ours (§3.3).
   4. Install through `DynCommand::LoadModel(path)`. This goes through the unchanged provenance gate (now with `teacher_rev`), so a file whose key and provenance disagree is treated as `InvalidData`.
   - **What may be installed.** Samples-only and references-only files (§7.1, §13) install their samples and references without a model.
   - **No refit runs on load.** The file holds the fit, its accumulator and its report, and the trust state is recomputed exactly as `install_sidecar` does today (training residuals from the stored samples). Samples kept after the last refit come back pending and are refitted at the next opportunity, as today.
   - **The worker's `saved_revision` is set to the file's revision** (review S7), so a load is never followed by a pointless re-save.
3. **Not found:** a fresh, empty model.
   - **No file is created until the first sample or reference is kept**, so browsing scenes leaves no empty entries.
   - The index entry is created with the file.
   - The lease is taken at resolution, so an entry that is created later is already owned.

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
- Pinned by plan Task 7a: `a_relaunched_viewer_restores_the_model_without_a_refit` covers the clean exit, and `a_killed_process_leaves_no_stale_lock_and_its_last_save` (`crates/app/tests/crash_restore.rs`) covers the crash.
- **A crash** is tested with a real killed process (`crates/app/tests/crash_restore.rs`, review S2). A child `app` runs with `--data-dir` and is killed with `Child::kill` after a save. The test then checks two things: the relaunched viewer loads the last save, and it acquires the lease (no stale lock).
- Reopening the **scene** itself on launch is plan Task 10, which is required. It is a profile option, off by default (§15.5).

### 5.5 Forget

**Forget model…** keeps its two-click confirmation (4 s). It drops the in-memory model and **removes the entry from the library** (review N4, adopted):

- The file is **renamed** to `<file>.forgotten-<unix>`, never deleted.
- The index entry is dropped and the lease released.
- The second click's label says so: "Click again to forget this model and its 412 samples".
- The Model tab then shows "forgotten — kept as <file>.forgotten-<unix> in <dir>; rename it back to <file> (with the app closed or on another scene) to restore it".
- The next kept sample creates a fresh entry under the same key.

**Every forgotten file is kept.** The browser's footer shows how many there are and their size. Removing them is up to the user, who can delete them in the folder. The app never deletes model data.

## 6. Saving

### 6.1 When

The provider exposes `Trainable::revision() -> u64`. It changes whenever anything the file holds changes: a kept sample, a refit, a quarantine flag, a pass record, the references or the mass record. The worker saves the active (leased) entry when `revision != saved_revision` and one of the following holds:

- **Debounced:** at least `save_every` since the last save. `save_every` = max(`SAVE_EVERY` = 5 s, 50 × the last write's wall time; review S4). This covers:
  - refits, which happen at most once per `REFIT_INTERVAL` = 500 ms while playing;
  - kept fallback and audit samples;
  - the samples a pass adds while it runs;
  - the < Q′ samples that have no model.

  A 50 ms write keeps the 5 s debounce. A 400 ms write (a large body-3 file on a slow disk with an antivirus rescan) stretches it to 20 s. The write volume is therefore bounded at about 2 % of wall time, whatever the file size.
- **Immediately** (each still goes through the writer, §6.2):
  - a pass finishes or is cancelled;
  - a switch (§5.3);
  - retire (backend switch, exit);
  - before a panic recovery rebuilds the driver (this replaces today's carry-over through the autosave path);
  - Export (the export writes the same bytes).

### 6.2 How: encode on the worker thread, write on the writer thread

**The encode.** `Trainable::encode(key) -> Result<Option<Vec<u8>>, String>` returns the full v4 file as bytes, or `None` when nothing is held. The worker calls it **between ticks, on the worker thread**: the same thread that runs refits, pass steps and installs. A saved state is therefore always consistent and **cannot race a refit**: a refit runs before the encode or after it, never during (§16, F4).

**The writer.** The worker hands `(key, revision, bytes)` to the library's **writer thread** (review S4: from the start, not as a later optimisation). The writer:

1. performs the atomic write and the fsync (§4.3) and updates the index;
2. reports `Saved { revision }` or `Error`;
3. **drops a queued older revision** when a newer one of the same key arrives;
4. **never writes a revision older than the one on disk.** The file's header carries the revision.

The worker records `saved_revision` from the `Saved` report. The write cost is measured per write and feeds the debounce (§6.1). An immediate save at retire waits for its write to complete before `Retired`.

### 6.3 Exit

Today `main` calls `process::exit` without joining the worker, and the session autosave is deleted. With the library, exit works as follows:

- **On close,** the viewer sends `Shutdown` to the estimated worker **and waits for any worker still retiring on a reaper thread (`Viewer::retiring`; review S3)**. It keeps the window up with a "saving the model…" card for up to `EXIT_SAVE_WAIT` = 10 s in total, waiting for each `Event::Retired`. `retire` sends that after its save has been written.
- **A GPU solve** in flight stops within one SCF step, so the wait normally takes well under a second.
- **A CPU solve** cannot be interrupted. If the wait times out, the app exits anyway and prints to stderr: "the model library was last saved at <time> (<n> samples); samples kept since then were not saved".
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

**References-only files (review N2, adopted).** A pass cancelled before its first kept sample has still solved its isolated-atom references, which cost 13–15 s per O multiplicity. A v4 file may therefore hold **zero samples and at least one reference**: the samples-only layout with empty `sample_*` arrays. `species` is then the references' Z. Such an entry restores its references into the cache, and the next pass does not re-solve them. A file with neither samples nor references is never written (`encode` returns `None`).

**`teacher_rev` in the provenance (review M3).** v4's `provenance` object carries `teacher_rev` (an integer ≥ 1). v1–v3 files read as `teacher_rev` 1.

**Older builds.** Builds before this milestone refuse `ASMODEL4` files with "unknown magic" (review N6). The README says so. Export does not write older formats.

### 7.2 What the store must start recording

The provider (`providers::surrogate`, plan Task 4b) gains the following. None of it enters the fit, the trust rule or the fallback policy.

- `SampleTag::Audit` for kept audit answers. The `keep_sample` call site in `audit.rs` passes the tag. **The fit does not read tags**: plan Task 4a pins that a store with `Audit` tags fits bit-identically to the same store tagged `Fallback`.
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

`surrogate::sidecar::merge(base, other) -> (Sidecar, appended)` is file-level and needs no provider. **[Fable]** owns it in plan Task 4a, because it decides what the fit sees.

- **The keys must be equal.** Different keys never merge (§2). Import routes a file to its own key's entry instead.
- **No entry yet:** the file is copied in as is, re-encoded as v4 with `history_since` = now.
- **Entry exists:** the result keeps the **fitted** side's model, accumulator, report and samples, in order. If both are fitted, the library entry's are kept. If neither is fitted, the entry's are kept. The fitted prefix must stay a prefix (`install_sidecar` derives the accumulated prefix from `accumulator.n_samples` and the quarantine flags). The other side's samples that are not already present are then appended **as pending**:
  - "present" means bitwise equality of (z, pos, energy, forces);
  - appended samples carry their **tag, weight and face density** (so the box rule still applies to them);
  - their **quarantine flags are dropped**. Quarantine only marks samples inside the fitted prefix, and the next refit's trust recovery re-judges them (ruling S15, unchanged);
  - the reference energies must agree bit-exactly on shared species. A disagreement refuses the merge with a message naming the species. Both were solved by the same teacher at the same tolerance, so a mismatch means a different reference rule (pre-S9), and mixing those is refused, never averaged.
- The merged entry's next activation finds samples pending and refits. A merge into the **active** entry is applied as: encode the provider's state → merge with the import → write → `LoadModel(entry)`. It is refused while a pass runs: "cancel the running sampling pass first", the existing rule.
- The history records merge: passes concatenate, teacher times add, `created` = min, `last_trained` = max, refits add. The merged file's `revision` is max(base, other) + 1 (review S7), so a writer never treats the merge as older than either input.
- **The first refit after a merge runs the trust recovery before the model can be trusted** (review N3). The appended samples' quarantine flags were dropped, so they are re-judged before they can earn trust. Pinned in plan Task 4a.

### 8.4 The key of an imported file (review S6)

An imported file's key comes from its provenance plus a species set. The species set is **the sorted union of its samples' Z (`sample_z`) and its basis species**. A v1–v3 header's `species` is the basis's alone, so it can miss a species whose samples arrived after the last refit. For a references-only file, the set is the references' Z.

## 9. Export and import (Save / Load renamed)

The explicit file buttons stay, renamed to say what they now are:

- **Export model** (was Save model) writes the active entry's current bytes to `model_path(scene_path)` beside the scene file, through the atomic write of §4.3. It is enabled with **any samples** (it used to need a fitted model), because a samples-only file is now a valid file. The message reads "exported the model 'H · 49³ …' (412 samples) to D:\runs\h2.model".
- **Import model** (was Load model) reads `model_path(scene_path)` and imports it (§8.3).
- **The refusal rules keep their meaning.** A model is **never installed on a scene whose provenance differs**. The gate is unchanged, and `Trainable::load` is still the only installer. A different-key import is stored in its own entry, not refused-and-lost, and the message explains it with `Provenance::describe` on both sides, as today's refusal does. A corrupt file is refused with the reader's error and is not set aside, because it is the user's file and not the library's. Import is refused while a pass runs (same-key case, §8.3).
- The File ▸ Estimated model menu, the Model tab buttons and the Ctrl-K entries are renamed. The search entries keep "save model" / "load model" as keywords. The scene Load's automatic sidecar load is replaced by §8.2.

## 10. Interaction with M6 and the worker

- **The session autosave (`worker/autosave.rs`, `owns_autosave`, `Viewer::autosave`) is removed.** It is replaced by the lease. The ownership rule it implemented (a worker that never resolved must not overwrite or delete a model) becomes: a worker writes only an entry whose lease it holds, and it resolves (takes a lease) only at `EnsureModel`. A worker shut down before `EnsureModel` holds no lease and writes nothing, which keeps today's I1 guarantee. A refused autoload no longer exists: a key's file always matches its key, otherwise §14 applies. So `autoload_refused` and "Forget it and sample this scene" go away. The provenance-mismatch warning stays for a model installed on a scene whose provenance later changed. That case cannot arise with the library resolving every change, but the warning is kept as a guard.
- **`Command::EnsureModel { autoload, plan }`** becomes `EnsureModel { plan }`. The worker resolves the key itself.
- **The spawn signature** changes from `SimWorker::spawn_with(…, autosave: Option<PathBuf>)` to `…, library: Option<LibraryConfig>`. `LibraryConfig { dir, scan_legacy: Option<PathBuf> }`, where `scan_legacy` is `None` in tests unless a test sets it (production: `Some(std::env::temp_dir())`). `spawn_test` takes one too.
- **Panic recovery** saves through the library (inside `catch_unwind`), rebuilds the driver, and re-resolves.
- **New snapshot fields:** `ModelStatus` gains `record: sim::ModelRecord` (from `TrainableStatus::record`) and `entry: Option<EntryStatus>`. `EntryStatus` holds the label, file, read-only reason, `loaded_from_library: Option<u64>` (last_trained, for §5.4's line), last save time, save error and the count of other unsaved models. `Snapshot` gains `library: Option<Arc<LibraryView>>` (the index entries, total size, the directory and its availability), republished only when it changes.
- **`Trainable` gains** `teacher_tol()`, `revision()`, `encode(key)` and `record()`. `save` writes samples-only files, and `load` reads them. These are listed in the plan's Task 4a/4b interfaces. Nothing else in `sim` changes, and `sim` still never sees a file format (rule 5): bytes and plain records cross the trait.

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

**Freshness (review N5).** The section re-reads `index.json` (through `Library::refresh`, under the index lock) each time it is opened, and when its Refresh button is pressed. Another instance's new entries therefore appear without a restart. **Delete** is `Command::DeleteModel(file)`: it renames the entry to `.forgotten-<unix>` like Forget (§5.5) and needs the lease. The footer reads "N forgotten models (X MB) kept in <dir>".

## 12. The Model tab's information

A new **This model** group heads the Model tab's readouts (search id `d-model-info`). The "exists" column says whether today's M6 code already has the value, where, and what this spec adds.

| row | content (example) | exists in M6? |
|---|---|---|
| model | `H · 49³ · 0.25 Bohr · GPU 1e-6`; "loaded from the library (last trained 2026-10-14 17:46 UTC)" / "new — no samples yet" / "read-only: in use elsewhere" | **new** (key, entry status) |
| trained on | `Provenance::describe_recorded()` of the store: "49³ h 0.25 σ 0.25, charge 0, multiplicity lowest, teacher 'Kohn-Sham (GPU)' rev 1 at tol 1e-6, masses ¹H ²H (recorded, not compared)" | data exists (`SampleStore::provenance`); **not exposed** — `ModelRecord::provenance` |
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
| an entry that exists but cannot be read now (OS error 5 / 32, other I/O; review M1) | any `read` error other than `InvalidData`/`Unsupported` | **read-only, empty, retried every 30 s; never `Fresh`, never overwritten**; merged on success (§4.4) | amber line: "model file <path> could not be read (<error>); retrying — nothing is written until it is read" |
| a forgotten or deleted entry | Forget / Delete | renamed to `.forgotten-<unix>` (§5.5) | the restore line in the Model tab |

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
6. **Scene changes** go out as **one `Command::SetSceneSettings`** (§5.1, review M6). The worker validates them jointly, in dependency order (grid and cell guard, σ, charge, multiplicity). h and σ are snapped (§3.2). If a field is refused, the scene fields are not applied, the refusal names that field, and **everything else in the profile still is applied**. The Log names it, for example "grid 99³ from profile 'X' not applied: n must be odd and 17..=97". The result is one key switch at most. Pinned: P2.
7. **The solver change comes last** (`Viewer::start`, the existing re-spawn with the current scene and run settings). The estimated worker resolves its model (§5.1).
8. **One Log entry and a toast** list exactly what changed, in Appendix A order: "profile 'Teaching' applied — dt 0.25 → 0.5 fs; thermostat off → on (300 K, τ 10 fs); grid 49³ → 41³; solver GPU → estimated — paused (the run's settings changed)". View-only changes are listed but say "not paused".

**A GPU that cannot fit the grid.** `validate_grid` bounds n at 97, and 97³ fits the RX 7900 XTX (24 GB) with room to spare. On a smaller adapter, a grid that passes validation can still fail to allocate. That failure takes the existing path: the provider error pauses the run and lands in `last_error`, as it does for a manual Apply grid. The Log line names the profile as the cause, and Undo (below) restores the previous grid in one click. Pinned: P2.

**Undo.** The toast of an apply that changed run, scene or solver settings offers **Undo** for 10 s. Undo applies the previous `Settings` through the same path, including the mid-pass check. It is one entry deep.

### 15.5 Launch

`main` builds the starting settings in this order:

1. **`Settings::factory()`**: the constants in code. `UiState::new` is built **from** `Settings::factory()`, so there is one source of truth (§15.8).
2. **The default profile.** `user.json`'s `default_profile`, if set and readable, is applied over the factory settings. Its scene-carried keys apply to the starting scene.
3. **The command line wins over the profile (review M5).** The flags that beat it are `--preset`, `--backend`, `--scene`, `--walls`, `--no-walls`, `--draft`, `--cloud-refresh`, `--estimated`, `--sample`, `--speed` and `--audit`. Each override is logged, for example "--backend cpu overrides profile 'Teaching' (solver: estimated)".
4. **The starting scene.** `launch.reopen_last_scene` is factory **off** (plan Task 10, required; review M4).
   - When it is on and neither `--preset` nor `--scene` is given, the app opens `user.json`'s `last_scene` instead of H₂: a scene-file path, or `preset <name>`.
   - **`--scene` and `--preset` beat it.**
   - `last_scene` is recorded on every Load scene, Save scene and preset choice. An unsaved edited scene is not reopened (there is no file).
   - If the file is gone, the app starts on H₂ with a Log note.

**Automation runs never touch the user's data (review M5, extending L2).** `--screenshot`, `--record` and `--replay` run with a **fresh temp data root**, so factory settings and an empty library, unless `--data-dir` is given explicitly. The user's default profile, last scene and models are therefore neither read nor written by automation.

The Log's first lines say what was applied, for example:
- "settings: profile 'Teaching' (default)"
- "settings: factory (no default profile)"
- "settings: factory — the default profile 'teaching.json' was not found"

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
- **F7 A crash leaves a stale lock or loses the last save** (review S2). The guards are OS-released `LockFileEx` locks and the debounced writer. The test kills a real child process.
- **F8 An unreadable but valid entry is overwritten** (review M1), for example while an antivirus scanner holds it. The guards: only `InvalidData` is corruption, only the lease holder quarantines, and any other error opens read-only and retries.
- **F9 Input noise splits a key** (review M2), for example σ from a DragValue. The guards are snapping at every entry point, `fixed_decimals(3)`, and the pinned `3.0 * 0.1` test.
- **F10 A solver change under an unchanged key** (review M3). The guards are `TEACHER_REV` in the gate and the key, and the `teacher_rev_tripwire` in `validate`.
- **P1 A profile from a newer version.** It is refused as corrupt, its unknown settings are lost when it is updated, or a known key with a changed meaning is misread. The guards: version > 1 is read under the same rules, unknown keys are ignored with a warning and kept verbatim on Update, and a meaning change requires a new key name (§15.2).
- **P2 A profile naming a grid the GPU cannot fit**, or an invalid or too-small grid for the scene. The failure modes are a panic, a silent half-applied profile, or nuclei outside the box. The guards: the scene-carried keys are validated as Apply grid is (`validate_grid` and the cell guard), a refused key is reported while the rest still applies, an allocation failure takes the existing provider-error path naming the profile, and Undo restores the previous grid (§15.4).
- **P3 Switching profile mid-pass.** A grid or solver change silently cancels a pass, or its samples land in the wrong model entry. The guards: the two-click mid-pass confirmation, and the cancel into the old entry (§5.3, §15.4). Run-class changes leave the pass running.
- **P4 The default profile deleted** (in the UI, by hand, or by another instance). The failure modes are a launch failure, a stale pointer applied, or silent factory settings. The guards: deleting the default in the UI clears the pointer and says so, and a missing file at launch gives factory settings with a Log line naming it (§15.6).

## 17. Validation

This milestone does not touch solver or dynamics code. It touches the surrogate's store (a tag and its bookkeeping) and the sidecar, so **rule 6 applies to the surrogate suite**: `cargo test -p validate --release --test surrogate` must reproduce the M6 numbers to the printed digits (plan Task 4a). The audit tag is bookkeeping, and the bit-identical-fit unit test is its guard. Everything else is unit- and worker-tested:

- `cargo test -p surrogate` (format, key, library);
- `cargo test -p providers --release` (records, merge);
- `cargo test -p app --release` (worker, viewer, UI).

Timings (save cost, switch cost) are reported, never asserted.

**The teacher-revision tripwire (review M3).** `cargo test -p validate --release --test tripwire` runs `teacher_rev_tripwire`. It is release-only and takes seconds on the CPU. It runs the CPU f64 teacher on H₂ at 33³, h 0.25, R = 1.4 Bohr, tol 1e-10, with the energy pinned to 1e-10 Ha. The pin is measured once (pre-measurement rule) and recorded in `references.md` with `TEACHER_REV` = 1. It fails with "solver output changed: bump TEACHER_REV and add a references.md row". It joins the solver gate: every solver change must pass it, or bump the revision.

Part B is pure app code. Its tests are `cargo test -p userdata` (atomic write, lock, JSON, time) and `cargo test -p app --release` (`settings`, `profiles`, the apply path through `ui::actions`, launch). Applying a profile sends the same commands as the controls, so the dynamics suite is not affected (rule 6 has nothing new to gate). A test (plan Task 9) pins that a profile changing dt and the thermostat produces exactly the `DynCommand`s the controls produce.

## 18. Provisional rulings (made here, flagged for the user)

- **L1** The user-data root defaults to `%LOCALAPPDATA%\atomic-sandbox\`, not `%APPDATA%` (§4.1): unbounded size, machine-local, not roamed.
- **L2** `--screenshot`, `--record` and `--replay` use a fresh temp root (factory settings, empty library) unless `--data-dir` is given (§4.1, §15.5; review M5).
- **L3** A key switch caused by an edit offers the pass and never starts it. Entering estimated mode keeps today's automatic first pass (§5.3).
- **L4** Import into an existing same-key entry pools samples: the fitted side's prefix is kept, the rest is appended as pending, quarantine flags are dropped for re-judgement, and references must agree bit-exactly (§8.3). The alternative, asking the user to choose replace or add, was rejected: pooling is what a key means.
- **L5** The provider stamps dates with `SystemTime::now()` (display data, never physics). Tests assert order and presence, not values.
- **L6** The Library browser lives in the Model tab, not a new window (§11).
- **U1** Scene-carried settings (grid, σ, charge, multiplicity) default to `null`, meaning "as the scene says". A profile value applies to the current scene when the profile is applied and to the starting scene at launch. Later scene loads keep their files' values, and Reset to factory settings does not re-grid the scene (§15.1).
- **U1a (controller, 2026-10-15, after the Quality presets request):** the user-level **Quality** setting (Fast / Balanced / Accurate / Precise / Custom h; factory default Accurate, h = 0.15 with σ = h) is a profile setting like any other (Appendix A gains a `run.quality` row); it applies to new scenes, presets and the launch scene, exactly as the app's own default does. Scene files that carry an explicit grid keep it (U1 unchanged for them). A profile's explicit grid/σ values (U1) still win over its quality level when both are set. Tasks 8–9 add the row and a test that applying a profile with quality Balanced to a preset re-grids it to h = 0.25 with the box preserved.
- **U2** Draft mode is **not** in profiles. The ruling's list names it, but M5b made it session-only and off by default. Flagged for the user; adding it is one row (§15.1).
- **U3** Applying a profile offers a one-deep Undo for 10 s (§15.4).
- **U4** Command-line flags win over the default profile, and the Log says so. The full list is in §15.5 (review M5). `--scene`/`--preset` beat `launch.reopen_last_scene`.
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
| 17b | `layout.ui_zoom` | `ui_zoom` (new; egui `Context::zoom_factor`) | Ctrl + / Ctrl − / Ctrl 0 (egui's keyboard zoom, on by default) | 1.0 | 0.5–2.0 | view — *if plan Task 8 Step 0 finds egui's keyboard zoom conflicting with an app shortcut, `zoom_with_keyboard` is disabled instead and this row moves to A.3 (review S1)* |
| 18 | `run.speed` | `speed` | toolbar speed slider / field / watchable preset (steps per tick, never dt — rule 8) | 1 | 1–`MAX_SPEED` (1000) | run |
| 19 | `run.dt_fs` | `dt_fs` | Simulation ▸ dt | 0.25 (`DEFAULT_DT_FS`) | 0.01–2.0 fs | run |
| 20 | `run.thermostat.on` | `thermostat_on` | Simulation ▸ thermostat (Bussi CSVR) | false | bool | run |
| 21 | `run.thermostat.t_k` | `thermostat_t` | … target | 300 | 0–5000 K | run |
| 22 | `run.thermostat.tau_fs` | `thermostat_tau` | … τ | 10 | 1–1000 fs | run |
| 23 | `run.com_correction` | `com_correction` | Simulation ▸ COM correction | true | bool | run |
| 24 | `run.gravity` | `gravity` | Simulation ▸ gravity (the field hook, g along −y) | false | bool | run |
| 25 | `run.walls.on` | `walls_on` | Simulation ▸ container walls | false | bool | run |
| 26 | `run.walls.half_width_bohr` | `wall_half_width` | … half-width slider | `null` = derived from the box (`default_wall_half_width`: (n−1)h/2 − 1.5 Bohr, clamped). **`from_ui` saves `null` when the value equals the scene's default** (review S1), so a profile does not pin one box's width | `null` or the box's valid range (clamped on apply, noted) | run |
| 27 | `run.drag_pull` | `drag_force` | Simulation ▸ drag pull; the wheel while dragging | 0.1 Ha/Bohr (`F_DRAG_MAX`) | `F_DRAG_LO`–`F_DRAG_HI` (0.01–1.0) | run |
| 28 | `solver.backend` | `requested_backend` (new: the user's choice; `backend` stays the running one) | toolbar Solver: CPU / GPU / Estimated | `"gpu"` (`main`'s default). Without an adapter it **runs** the CPU, as today, but the **requested** value stays `gpu`, so a fresh machine without a GPU never shows "(modified)" (review S1). `UiState::new` takes it from the factory settings, not `Cpu` | `"cpu"`, `"gpu"`, `"estimated"` | solver |
| 29 | `estimated.audit_every` | `audit_every` | Model ▸ Audits: DFT audit every n | 2000 (`FallbackPolicy::default().audit_every`) | 0–1 000 000 (0 warns "unsafe" on apply, as the control does) | run |
| 30 | `estimated.cloud_superposition` | `cloud_superposition` | Physics ▸ Estimated: atomic superposition | true | bool | run |
| 31 | `estimated.cloud_refresh` | `cloud_refresh` | … refresh cloud every | false | bool | run |
| 32 | `estimated.cloud_refresh_s` | `cloud_refresh_s` | … every N s | 2.0 | 1–10 s | run |
| 33 | `sampling.perturbations` | `plan.perturbations` | Model ▸ Sampling plan | 60 | 0–2000 | view (the next pass's plan) |
| 34 | `sampling.sigma_r_bohr` | `plan.perturbation_sigma` | … σ_r | 0.15 | 0–0.5 Bohr | view |
| 35 | `sampling.burst_steps` | `plan.hot_burst_steps` | … hot burst steps | 200 | 0–5000 | view |
| 36 | `sampling.burst_t_k` | `plan.hot_burst_t` | … hot burst T | 2000 | 0–10 000 K | view |
| 37 | `sampling.scan_points` | `plan.scan_points_per_pair` | … scan points / pair | 12 | 0–200 | view |
| 38 | `launch.reopen_last_scene` | — (new, plan Task 10, required) | File ▸ Profiles: reopen the last scene on launch | **false** | bool | launch only |

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
| `plan.seed` (it has a control, Model ▸ Sampling plan ▸ seed), `plan.scan_range` | session-only: the seed advances after each pass, so a pinned seed would repeat geometries; the scan range has no control (review S1) |
| `draft` | M5b: session only, off by default **[provisional, U2]** |
| `status`, `status_seq`, `log`, `search`, `help`, `queued`, `forget_armed`, `start_armed` | transient |
| `gpu_available` | hardware |
| the camera (`Viewer::camera`), the viewport | per view |
| the estimated solver's teacher | not a setting: the GPU when an adapter exists, else the CPU (it is part of the model key, §3.1) |
| `backend` (the running solver) | follows `requested_backend` and the adapter (row 28) |
| the dock's width (egui memory), the window size and position, collapsing-section open states | window/egui layout state, not settings (review S1) |
| the selection, pins, play/pause | run state of the current scene |
| the Place tool's per-element isotope choice (M6 Task 10's picker) | tool state, like `element` (review S1) |
| this plan's own UI fields: the Library selection and its delete-armed time (Task 7b); `profile_armed`, `applied_profile`, `undo` (Task 9) | transient |

## 20. Review (2026-10-15, `docs/superpowers/notes/2026-10-15-user-data-spec-review.md`): each MUST and the change

The verdict was "ready with fixes", with every provisional ruling (L1–L6, U1–U5) agreed. The controller ruled M3 adopt, M4 required, N2 adopt and N4 adopt. Every SHOULD and NOTE is applied, as listed below.

| item | change |
|---|---|
| **M1** a non-`InvalidData` read failure became `Fresh` and could overwrite a valid entry | §4.4, §5.2, §14: only `InvalidData`/a bad header is corruption, and only the lease holder quarantines. Any other read error opens read-only and empty, retries, and never overwrites. Plan Task 3 `an_unreadable_but_valid_entry_is_never_overwritten`; Review Focus F8. |
| **M2** non-canonical inputs (σ DragValue noise) split keys | §3.2: `snap_length` (`{:.6}`) at every entry point (`SetGrid`, `SetSigma`, `SetSceneSettings`, scene parse, presets, profile apply), plus `fixed_decimals(3)` on σ. §3.4: labels print the shortest round-trip, so distinct keys get distinct labels. Pin: `3.0 * 0.1` keys as 0.3 (`da3f9c7f1eb4da29`). Plan Tasks 3 and 5; F9. |
| **M3** a teacher rename by hand is unsafe | §3.1: `TEACHER_REV: u32` in `providers::ks`, carried in `Provenance` (gate) and the key (`teacher_rev 1`; v1–v3 read as 1). §17: the `teacher_rev_tripwire`. §3.4: the diff shows "teacher rev 1 → 2". All §3.3 pins recomputed. The pseudopotential milestone bumps it. Plan Tasks 2, 3, 4a; F10. |
| **M4** reopen-last-scene was optional | §2, §15.5: a required task (plan Task 10), a profile option, OFF by default. |
| **M5** the override flag list was incomplete | §15.5: `--preset --backend --scene --walls --no-walls --draft --cloud-refresh --estimated --sample --speed --audit` beat the profile, and `--scene`/`--preset` beat `reopen_last_scene`. `--screenshot --record --replay` use a temp root (L2 extended). Plan Tasks 7a and 9. |
| **M6** a profile's scene fields would cause up to four replaces, switches and toasts, validated in the wrong order | §5.1: `Command::SetSceneSettings { n, h, sigma, charge, multiplicity }`, one replace validated jointly in dependency order. §15.4 step 6 uses it. The existing commands are named in §5.1. Plan Task 5. |

SHOULD and NOTE items:

- **S1** Appendix A rows `ui_zoom`, walls `null`, `requested_backend`, and A.3 additions.
- **S2** The killed-process crash test (§5.4, F7).
- **S3** Exit waits for retiring workers; `pending_teach` is kept (§4.4, §6.3).
- **S4** The writer thread and the cost-scaled debounce (§6.1–6.2).
- **S5** The rename semantics and retry codes, and the Unix directory fsync (§4.3).
- **S6** Import keys and EnsureModel sidecar import (§8.4, §5.1).
- **S7** Consistency: `scan_legacy: Option<PathBuf>`, the `&Lock` proof, `Command::DeleteModel`, `<stem>.lock` for suffixed entries, revision after a merge = max + 1, and `saved_revision` on load (§3.3, §5.2, §8.3, §11). The plan's interfaces match.
- **S8** The plan splits (4a/4b, 7a/7b) and pins.
- **N1** §3.1.
- **N2** References-only files (§7.1).
- **N3** Plan Task 4a pins recovery before trust after a merge.
- **N4** Forget renames to `.forgotten-<unix>` (§5.5).
- **N5** The browser re-reads the index (§11).
- **N6** The README line (§7.1, plan Task 7b).
