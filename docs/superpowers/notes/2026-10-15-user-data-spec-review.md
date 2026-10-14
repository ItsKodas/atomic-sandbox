# Review — User data spec and plan (2026-10-15)

**Reviewed:** `docs/superpowers/specs/2026-10-15-user-data-design.md` and `docs/superpowers/plans/2026-10-15-user-data.md` at commit 0659385.

**Verdict:** ready with fixes. 6 MUST, 8 SHOULD, 6 NOTE. The reviewer agrees with every provisional ruling (L1–L6, U1–U5).

The controller's rulings are folded in below: M3 adopt; M4 required task; N2 adopt; N4 adopt. Every other item was to be applied as written. Spec §20 records how each MUST was applied.

## What checks out

- The design is sound and complete against the ruling.
- The Model tab list matches the ruling item for item.
- Restart and crash recovery are covered (§5.4, Task 7).
- The < Q′ samples are covered (§13, Tasks 4 and 5).
- All 9 FNV-1a pins and the 3 `utc_label` pins were re-checked in Python.
- `Provenance::eq` ignores masses, so the 12-variant property test is consistent.

## MUST

- **M1** (§4.3, §5.2, §14; Task 3): read errors.
  - A read failure that is not `InvalidData` must never become `Fresh`. Examples: a sharing violation, `PermissionDenied` from an antivirus scanner or indexer, a transient I/O error.
  - Open the entry read-only and retry instead.
  - Only `InvalidData` or a bad header leads to quarantine, and only the lease holder renames a file to `.corrupt` or deletes it.
  - Test: `an_unreadable_but_valid_entry_is_never_overwritten`.
- **M2** (§3.2, §3.4): the inputs to the key are not canonical.
  - The `{:?}` formatting is right, but the σ DragValue (`menus.rs:262`, no `fixed_decimals`) can reach 0.30000000000000004 via `smart_aim`, while a scene file holds an exact 0.3. That gives two keys and a gate refusal.
  - Snap h and σ in `SetGrid`, `SetSigma`, scene parsing and profile apply, or set `fixed_decimals(3)` on the control.
  - Test that `3.0*0.1` keys the same as `0.3`.
  - Labels must show enough digits to tell keys apart.
  - **Ruling:** do both. Snap with `format!("{:.6}").parse()`, and set `fixed_decimals(3)` on the σ control.
- **M3** (§3.1, Global Constraints): renaming the teacher by hand is unsafe.
  - Add `TEACHER_REV: u32` in `providers::ks` and carry it in `Provenance`, so that gate and key agree.
  - The key gets the line `teacher_rev 1`; v1–v3 files read as 1.
  - Add a validate tripwire: CPU f64 H₂ at 33³ and a fixed R, energy pinned to ~1e-10 Ha.
  - The browser diff shows "teacher rev 1 → 2".
  - **Ruling:** adopt. The tripwire fails with "solver output changed: bump TEACHER_REV and add a references.md row". The pseudopotential milestone will bump the revision.
- **M4** (§2, Task 10): the ruling makes reopen-last-scene a profile option that is off by default. That makes it a required task, not an optional one.
  - **Ruling:** required, implemented as a profile option, OFF by default.
- **M5** (§15.5, U4, L2): the override flag list is incomplete.
  - Missing: `--scene`, `--walls`, `--draft`, `--cloud-refresh`, `--estimated`.
  - `--scene` beats `reopen_last_scene`.
  - `--record` and `--replay` need a temp data root, like `--screenshot`.
  - **Ruling:** the full list is `--preset --backend --scene --walls --no-walls --draft --cloud-refresh --estimated --sample --speed --audit`.
- **M6** (§5.1, §15.4): there is no `ReplaceScene` command at the worker level.
  - `SetCharge`, `SetMultiplicity`, `SetGrid` and `SetSigma` each go through `Worker::replace`.
  - A profile apply would therefore cause up to 4 switches, 4 toasts and 4 resolves.
  - The validation order is also wrong: the charge range depends on the proton count, and the multiplicity on the electron count after the charge.
  - Add `Command::SetSceneSettings { n, h, sigma, charge, multiplicity }`.
  - **Ruling:** one replace, validated jointly in the right order, one key switch, used by profile apply. Name the existing commands in §5.1.

## SHOULD

- **S1** (Appendix A): missing and mis-filed settings.
  - Missing: egui's keyboard UI zoom (`zoom_with_keyboard`, on by default). Add a `layout.ui_zoom` row, or disable keyboard zoom. **Ruling:** disable it only if it conflicts with the app's shortcuts; otherwise add a profile row.
  - The dock's resizable width (kept in egui memory): add `layout.dock_width`, or list it in A.3.
  - Window size and position, collapsing-section states, selection, pins and play state belong in A.3.
  - The sampling seed has a control (`dock/model.rs:255`). It is session-only; say so.
  - M6 Task 10's per-element isotope choice needs a row.
  - Fields the plan adds belong in A.3: Task 7's library selection and delete-armed state; Task 9's `profile_armed`, `applied_profile` and `undo`.
  - `from_ui` should save `run.walls.half_width_bohr` as null when it equals `default_wall_half_width(scene)`.
  - The factory backend is `"gpu"`, but `UiState::new` uses Cpu, so a machine without a GPU shows "(modified)". Fix the mismatch.
- **S2** (Task 7 crash test): detaching the worker without a retire keeps the worker and its lease alive in-process, so v2 opens read-only.
  - That proves neither crash recovery nor that the OS releases the lock.
  - Kill a child process instead (`Child::kill` on a `--data-dir` root).
  - Add to Review Focus: "a crash leaves no stale lock".
- **S3** (§6.3, Task 5): exit and retiring workers.
  - At exit, also wait for the reaper-thread workers (`Viewer::retiring`) within `EXIT_SAVE_WAIT`.
  - Keep `pending_teach` (EnsureModel held back until retirements finish) when removing autoload. Otherwise a quick solver toggle sits read-only for `LEASE_RETRY` (30 s).
- **S4** (§6.1–6.2): write volume.
  - A multi-MB body-3 file rewritten every 5 s while playing, with a refit every 500 ms, comes to about 3 GB/h with fsync, plus antivirus rescans.
  - Put write + fsync on the writer thread from the start. Encoding stays on the worker, so F4 still holds.
  - Scale the debounce with the last write's cost.
- **S5** (§4.3): rename semantics.
  - On Windows, Rust's `fs::rename` tries `SetFileInformationByHandle(FileRenameInfoEx, POSIX | REPLACE_IF_EXISTS)` first, then `MoveFileExW`. Document this.
  - Retry only on OS errors 5 and 32.
  - On Unix, fsync the directory after the rename.
- **S6** (§8.1–8.3): import keys.
  - The v1–v3 header `species` is the basis's. Key imports by sorted unique `sample_z` ∪ basis species.
  - Import the scene's sidecar at EnsureModel when the scene has a path.
- **S7**: consistency fixes.
  - Spec §10 says `scan_legacy: bool`; plan Task 5 says `Option<PathBuf>`.
  - Task 3's `write_entry` is missing its `&Lock` proof parameter.
  - `Command::DeleteModel`, used in Task 7, is never declared.
  - The lease file for a collision suffix (`<h>-2.lock`) is not specified.
  - The revision after a merge is unspecified; make it max + 1.
  - A load should restore `saved_revision`.
- **S8**: task splits and pins.
  - Split Task 4 into 4a [Fable] (merge, samples-only install, audit-tag inertness, the surrogate gate) and 4b [Opus] (records, coverage, revision, teacher time).
  - Split Task 7 into 7a (exit, `--data-dir`, the F6 restart tests) and 7b (info rows, Library section).
  - Task 4's "8 − skipped_p" is undefined.
  - Task 7's info test covers 4 of 12 rows. Give exact strings for trained on, fit, trust, passes, file and scenes too.
  - Task 1's sidecar-bytes pin should be an `fnv1a_64` recorded before the move.

## NOTE

- **N1** multiplicity `None` and `Some(lowest)` are separate keys. That is correct; say so in §3.
- **N2** A pass cancelled before its first kept sample loses its solved references, at 13–15 s per O multiplicity. Allow references-only files. **Ruling:** adopt.
- **N3** The merge drops quarantine flags. Pin that the first refit after a merge runs trust recovery before the model can be trusted.
- **N4** Forget deletes the file. Rename it to `.forgotten-<unix>` instead. **Ruling:** adopt, with a line in the Model tab on how to restore.
- **N5** `LibraryView` should re-read the index when the browser opens.
- **N6** Older builds reject `ASMODEL4` files. Add a README line.
