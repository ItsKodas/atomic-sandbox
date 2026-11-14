# Fermi–Dirac Smearing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `DegenerateShells` and all-electron aufbau with one continuous rule, Fermi–Dirac occupations at kT = 5e-3 Ha per spin channel (the user's 2026-11-14 ruling after V0; 1e-3 before), so the SCF converges on hand-built clusters; report and learn the Mermin free energy F = E − TS; carry the user's collections onto the new surface key by a user-started re-solve pass that never writes the old folder.

**Architecture:**
- One host f64 function in `qm` (`fermi_dirac`: μ_σ by two tie-directed bisections, f, S) is called by the CPU loop and imported by `qm-gpu`. `Energy.total` becomes F, so every consumer of `total` gets F unchanged.
- `providers` switches every scene to FD under new provenance tags (`none/fd5e-3/pin1e-5`, `…/fd5e-3/stop2x…/pin1e-5`) and replaces the occupation-pattern watch with a work-consistency guard, `branch_jump`.
- The collection migration is a pass in `providers::sampling`. It reads the predecessor folder through a strictly read-only open, re-solves every sample cold and classifies it U/T/S. It appends to the new folder, holds S-class samples at status code 9 (`Q_RESOLVED`; code 8 is `purge-forgotten`'s "forgotten, fence lifted") and reviews them by leave-one-out inside a refit.

**Tech Stack:** Rust (workspace crates `qm`, `qm-gpu`, `sim`, `providers`, `surrogate`, `app`, `validate`), wgpu/WGSL (f32 grid), egui.

**Spec:** `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md` (commit ad774a9; "the spec", sections cited as §n). Also read its physics review `docs/superpowers/specs/.fermi-smearing-physics-review.md` (untracked; findings cited as [MUST-n] etc.) and the diagnosis `docs/superpowers/notes/2026-11-13-scf-stall-diagnosis.md` (cited as D§n). Project rules: `CLAUDE.md`.

## Global Constraints

- **Constants (spec §4, §5, §6.3, §7.4; all global, none per element, rule 1):**
  - `ELECTRON_KT = 5e-3` Ha (T_el = 1 578.9 K, shown as 1579 K; the user's 2026-11-14 ruling, spec §2 item 3). Not exposed in the app, no dev flag. Harnesses set `kt` through `ScfSettings`.
  - `GPU_PARTIAL_TOL = 1e-5`, defined once in `qm::occupations` (Task 4), re-exported by `qm-gpu`; the tags (Task 9) and the dimer ledger (Task 10) read this one constant (preflight F11). The GPU SCF stops at max(tol_eff, 1e-5) on a step whose occupations have a partial level; gapped GPU solves and the CPU keep their tolerance (spec §5.1).
  - `WITNESS_GAP = 0.1` Ha (the renamed `CANDIDATE_TAU`); tail arm 36 kT; `MAX_WITNESS_GROWTH = 16`.
  - μ bracket [ε_min − 50 kT, ε_max + 50 kT]; backstop 2 048 halvings.
  - `PARTIAL_OCCUPATION_MIN = 1e-6`.
  - `branch_jump`: factor 3, window 8, reach 0.1 Bohr, κ_max = 1 Ha/Bohr³.
  - Migration: ε_E = 1e-5 Ha per sample, ε_F = 1.5e-4 Ha/Bohr per component.
  - `Q_RESOLVED = 9` (preflight F1): code 8 is `Q_FORGOTTEN_UNFENCED` ("forgotten, fence lifted"), from branch `purge-forgotten`, which merges to `main` first.
  - **Formats are symbolic** (F1): collection `RECORD_VERSION` = R + 1 and one new sidecar magic `ASMODEL<K + 1>`, where R and K are the merged `main`'s `RECORD_VERSION` and `MAGICS.len()`, read in Task 2 step 4. Never assume today's values; no task hardcodes them.
    - **Read in Task 2 (merge 97da009, `main` fe30578): R = 1, K = 8.** `surrogate::collection::RECORD_VERSION = 1` (`crates/surrogate/src/collection.rs` l. 273; the reader, l. 708, takes exactly `RECORD_VERSION`; neither `purge-forgotten` nor `forget-batch` bumped it). `surrogate::sidecar::MAGICS` (`crates/surrogate/src/sidecar.rs` l. 267) = `ASMODEL1` … `ASMODEL8`; `purge-forgotten` added no body (code 8 rides the existing `sample_quarantine` array). R11's old-body set (bodies from 6 up) = **{6, 7, 8}**: 6 = samples without a fit (`write_samples`, l. ~993/1013), 7 = fitted without body-4, 8 = fitted with body-4 (`encode`, l. 592). So the new collection version is **2** and the new magic **`ASMODEL9`** (`MAGICS.len()` = 9); the app's manifest reads both through `app::update::formats` l. 55/57.
    - Quarantine codes persisted on the merged tree: 0–3, 7 (`Q_FORGOTTEN`), **8 (`Q_FORGOTTEN_UNFENCED`)**; 4 is tag-carried, 5 (`Q_REF_WAIT`) and 6 (`Q_LONE`) are the provider's own (`providers/src/surrogate/quarantine.rs` l. 53/69). **No code 9 exists anywhere**: `Q_RESOLVED = 9` is free. `is_persisted_code` (`crates/surrogate/src/quarantine.rs` l. 89) = {0, 1, 2, 3, 7, 8}, checked by the sidecar reader and writer (`sidecar.rs` l. 860–862); the collection reader keeps any code (no check), so `is_collection_code` is new (Task 13).
- **Tags:** pseudised `<word>/<surface_hash:016x>/fd5e-3/stop2x<floor>/pin1e-5` (`fd{kT:e}`, `pin{GPU_PARTIAL_TOL:e}`); all-electron `none/fd5e-3/pin1e-5`. Every "is this the all-electron surface" comparison goes through `providers::pseudo::is_all_electron_tag`.
- **Unchanged (anti-goals, spec §3):**
  - the mixer (Pulay/DIIS, α = 0.3, m = 6), the stall stop (W = 50, 1e-3 per electron), the convergence criterion, the cold guess, dt;
  - the force code (only its docs change);
  - the radial generator's occupations;
  - `history.bin`, `pseudo_cache` and `checkpoint` formats.
- **Never loosen a tolerance or bound to pass** (rule 6). A moved pin gets its TS term or its re-pin reason written beside it ("all-electron now solves witness vectors (FD, spec 2026-11-13)"). Bitwise pins are re-pinned, never deleted.
- **E is never reported with F's forces, or F with E's.** Old-rule and new-rule labels never share a collection or a fit.
- **The old collection folder is never written, cut, re-locked or deleted.** The migration starts only from the user's click. Measurements run on scratchpad copies. Never touch `D:\Projects\Physics\models` or the main checkout's untracked files (`models/bh3.model`, `models/nh4+.model`, `models/ph3.model`, `profiles/`, `research_notes/`, `scenes/h3+.scene`).
- **Agents and tags:** every subagent runs on Opus (memory "No Fable"). Each task is tagged **physics** or **engineering**. Physics tasks, and every task whose step says so, get a `physics-reviewer` pass on the diff before their commit.
- **Heavy jobs one at a time.** Never kill the user's cargo or `app.exe`. With another cargo job running, build at BelowNormal priority with `-j 4`. Heavy tests stay release-only (`#[cfg_attr(debug_assertions, ignore)]`).
- **Commits:** `git commit -o -m "<message>" -- <paths>` (options before `--`; a new file needs `git add <path>` first). Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never pipe a gating git command (`git merge … | tail`). Edit files with Edit/Write or Python, never PowerShell `Set-Content`.
  - **No bare `git stash`** (F6; the stash stack is shared by every worktree). Set work aside with a WIP commit, or `git stash push -u -m "<unique-tag>"` applied by its SHA (`git stash apply <sha>`, never `pop`).
  - **No diagnostic code is ever committed on `scf-stall`** (F5). The V0/A1 prototype (`SCF_SMEAR`/`SCF_TRACE2` knobs in `crates/qm/src/scf.rs` and `crates/qm-gpu/src/scf.rs`, `crates/providers/src/scf_stall_diag.rs` and its `lib.rs` mod line) is committed on the side branch `smearing-prototype` at the end of Task 3 and leaves the tree there. Implementers may read it (`git show smearing-prototype:<path>`); nobody merges it. `git commit -o` commits whole files, so before every commit that touches one of those files, `rg "SCF_SMEAR|SCF_TRACE2|DIAG_|scf_stall_diag" crates` returns nothing.
- **Task order** (preflight ruling): Task 1 → **Task 3 → Task 2** → Tasks 4–18. Task 3 (GPU measurements on the prototype, independent of `main`) runs before Task 2. Task 2 waits until `purge-forgotten` (and the trust rework) have merged to `main`.
- **Docs move with the code:** README (user rule), `crates/validate/references.md`, `docs/dev-reference.md`, and the spec sections each task changes are updated in that task.
- **It ships as one merge** (spec §11): the solver change and the migration together.

## Review Focus

1. **A warm state from another occupation scheme or another kT** (a harness state at kT = 2e-3, an `Aufbau` state, a state restored from before the change). It must never seed the solve: the result must be bitwise the cold solve. Tests: Task 5 (CPU), Task 7 (GPU).
2. **An edit between two evaluations that adds, removes or reorders atoms.** The branch-jump guard must restart its history, judge nothing and not panic on mismatched ΔR. Test: Task 8.
3. **A refit while the migration pass is still running.** No code-9 sample may be reviewed or released before the pass completes (spec §7.6). Test: Task 15.
4. **A crash between a migrated sample's append and the cursor write, or an older build still holding the old folder's lock.** Resume reproduces the counts of an uninterrupted run, and the duplicate is counted as done. A live lock refuses with the reason. Tests: Task 14.
5. **A predecessor `samples.bin` with a damaged tail.** The read-only open reads to the last whole record, reports the damage and leaves every byte of the folder identical. Test: Task 13.
6. **A purge of forgotten samples around a migration** (F2, F3; spec §8 amendment). A purge is refused while the pass runs; a purge of the new folder afterwards remaps `migration.txt`'s review log through `PurgeOutcome::map`; a source `samples.bin` whose length or hash changed restarts the pass from 0; the read-only open never cleans purge's `samples.bin.new`/`samples.bin.bak.new`; a forgotten-unfenced (code 8) sample in class S returns to 7. Tests: Tasks 13–15.

## Touch-point map (spec §8; filled in by Task 2 against merged `main`)

| # | role (spec §8) | name on `claude/charming-keller-0bc26b` (pre-merge) | name on `main` after merge (Task 2) |
|---|---|---|---|
| 1 | codes excluded from the fit | `Quarantine::excluded` (`providers/src/surrogate/quarantine.rs`) | `SurrogateKs::excluded(i)` = "code ≠ 0", `providers/src/surrogate/quarantine.rs` l. 606: code 9 is excluded **with no change**. Same reading in `fit_member_indices` (l. ~1132, code 0 only) and `worst_offenders_bound`'s filter (l. ~1114). `Collection::admit` refuses any non-zero code (`surrogate/src/collection.rs`, `refuse_quarantined` l. 249). |
| 2 | fence / near-misfit refusal | `Quarantine::fences`, `misfit_fence_stands`, `TeachRule::NearMisfit` | Quarantine fence: `SurrogateKs::fences(i)` l. 454 (today `Q_HELD \| Q_FORGOTTEN`), read by `quarantine_distances_cached` l. 972–1002 (filter l. 981) → `near_quarantined[_cached]` l. 949/1006 → `admit`'s fence branch l. ~1340 and `fence_refusal_text` l. 473 (refusal words; a failed 9 would read "held out of the fit (quarantined…)"). Misfit fence (M, untouched by code 9): `misfit_fence()` `trust.rs` l. 485, `misfit_fence_stands` `quarantine.rs` l. 1022, `near_misfit_atoms` l. 1039, `TeachRule::NearMisfit` refused per atom in `policy.rs` `atom_verdicts` l. 633 and in `mod.rs` `evaluate` l. ~2371. M is built from **fitted** samples only (`trust.rs` `Misfit` l. 169, `compute_residuals` l. 692), so an excluded code 9 never enters M with no change. **See the Task 2 finding on `fencing()` below.** |
| 3 | the recovery's held-cap count | `QUARANTINE_MAX_FRACTION` check in the recovery rounds | `rounds_by` `quarantine.rs` l. 830–831: `quarantined() + forgotten() + worst.len() > QUARANTINE_MAX_FRACTION × n_store` (`QUARANTINE_MAX_FRACTION` l. 131). The same sum warns in `forget_sample` l. 1638 and `forget_samples` l. 1727. `quarantined()` (l. 413) counts code 1 only and `forgotten()` (l. 430) codes 7 and 8: **code 9 is not counted, with no change** (confirm with a test). M's own limit (`trust.rs` `misfit_limit` l. 93) counts fitted samples only. |
| 4 | refit pipeline + leave-one-out ratio | `quarantine::loo_ratio`, `loo_ratios` | Ratio: `loo_ratio(p, per_atom, s, l)` `quarantine.rs` l. 205 (returns (ρ, ρ_σ, Channel); the review reads ρ), on `surrogate::loo::leave_one_out(model, smp)` (as `trust.rs` `judge_case` l. 1352 and the class pass l. 1118/1141 do). Cache: `LooRatios` l. 243, `fresh_loo_ratios` l. ~1142, `ensure_loo_ratios` l. ~1150, filled by `refresh_class_loo` (`trust.rs` l. 737). Pipeline: `refit` (`trust.rs` l. 1244; `sync_collection_status` + `flush_collection` after it unless `refit_fork`) → `refit_inner` l. 1272 → `refit_fits` l. 1294: `fit_store_deferred` (l. 1779) → `judge_votes` → `release_reproduced` (touches code 1 only) → `recover_trust` → `judge_held_out` (l. 1650). The review step goes in `refit_fits` after the first `fit_store_deferred` and before `judge_votes`/`recover_trust` (the trust judgement). A background refit (`background.rs`) runs `refit` on a fork: codes it moves (9 → 0) are the copy's and are installed; a live change of a code < P during it discards the result (`background` module doc). |
| 5 | lone-atom "ends at a gap" | `lone::closed_spin_channels` | Unchanged: `closed_spin_channels(z, all_electron, fractional)` `providers/src/surrogate/lone.rs` l. 181, its one caller `lone_reference` l. 289. |
| 6 | shared-collection attach | `SurrogateKs::attach_inner`, `attach_collection_at` (`shared.rs`) | Unchanged names: `attach` l. 246 → `attach_inner(c, restoring)` `providers/src/surrogate/shared.rs` l. 252 (installs `c.quarantine` as the codes, `hold_out_lone`, refit when due); `attach_collection_at(root, scene)` l. 340 (opens by the scene's key). `predecessor_of` does not exist yet (Task 15). Status writes: `sync_collection_status` l. 550, `write_status_now` l. 592; `released_code` l. 766. |
| 7 | label-noise prior | `policy::force_noise_prior` | Unchanged: `force_noise_prior(ionic)` `providers/src/surrogate/policy.rs` l. 130 (`ionic == "none"` is the all-electron test); callers `trust.rs` `fit_store_with` l. 1802 and, through `audit_force_floor` (l. 139), `audit.rs` l. 831, `mod.rs` l. 1980, `teach.rs` l. 817/967, `trust.rs` l. 1373. `is_all_electron_tag` does not exist yet. |
| 8 | occupation-change training refusal | `EvalStats::untrainable` (`sim/src/provider.rs`) | Unchanged: `EvalStats::untrainable` `crates/sim/src/provider.rs` l. 651 (`occupation_changed` field l. 562, `OCCUPATION_CHANGE_WARNING` l. 570); set in `providers/src/ks.rs` `eval_stats` l. 470 (`OccupationWatch`). Consumers: `surrogate/mod.rs` l. 1884, `audit.rs` l. 314, `sampling/mod.rs` l. 1321/1370/1590. |
| 9 | purge's index remap (F3) | `PurgeOutcome::map` (`surrogate/src/collection.rs`, branch `purge-forgotten`) | `PurgeOutcome { purged, map: Vec<Option<usize>>, backup, warning }` `crates/surrogate/src/collection.rs` l. 210 (`map[old] = Some(new)`, `None` = purged); `Collection::purge(indices)` l. 1260, `purge_forgotten` l. 1382, `purge_refusal` l. 1248; the rewrite `rewrite_without` re-encodes moved status records with `status_record` (so at the **current** `RECORD_VERSION`), temp files `samples.bin.new` / `samples.bin.bak.new`, backup `PURGE_BACKUP` = `samples.bin.bak`. Provider side: `providers/src/surrogate/purge.rs` — `purge_refusal` l. 79 (add "a migration pass is running" here, F3), `purge_forgotten` l. 106, `remap_after_purge(keep, coll)` l. 167 (remap `migration.txt`'s review log with the collection's `map`, F3). |
| 10 | the forgotten fence's lift (F4) | `lift_votes`, `fencing()` (`providers/src/surrogate/quarantine.rs`, branch `purge-forgotten`) | `lift_votes: Vec<FenceVote>` (`mod.rs` l. 1472), voted in `admit` `quarantine.rs` l. 1425–1471: the samples lifted are `common`, filtered to `== Q_FORGOTTEN` (l. 1439), so **a code 9 is never lifted, with no change**. But `admit`'s `movable` (l. 1396: fencing samples near the answer that are not forgotten) is **released or superseded** by one agreeing fresh answer (l. 1473–1483: `released_code` or `Q_SUPERSEDED`) — a fencing code 9 would be, unless excluded there (Task 2 finding B). `fencing()` l. 483 = count of `Q_HELD \| Q_FORGOTTEN`; see finding A. |

Later tasks name these by role. The implementer uses the right-hand column.

Rows 2, 3 and 10 are settled by preflight F4 (spec §8 amendment): a reviewed-and-failed code 9 is in `fences()`, not in `fencing()` or the held cap, purge's lift votes never lift it, and it never joins the misfit set M. Task 2 fills in the names; it does not re-decide this.

**Task 2 findings on the merged tree (for the controller; spec §8 "Amendment (plan Task 2)").**
- **A. `fencing()` is the fence's on-switch, not a cap.** On `main` (and on both branches before it) `fencing()` (`quarantine.rs` l. 483) has no meaning beyond "how many samples fence": its only readers are the early returns `if self.fencing() == 0 { return Vec::new() }` of `quarantine_distances` (l. 958) and `quarantine_distances_cached` (l. 973), and `views.rs` `any_fence` (l. 225) and `training_views`' descriptor-cache fill (l. 241). A failed code 9 that is in `fences()` but not counted by `fencing()` would fence **nothing** whenever no code 1 or 7 sample exists (the distance scan returns empty before it reads `fences()`), and the views would not re-judge their cached points. So F4's "not counted by `fencing()`" cannot hold together with "joins the quarantine fence". Nothing in `fencing()` feeds the held cap (row 3: `quarantined() + forgotten()`), so counting a failed 9 there costs nothing F4 protects. Needs a ruling before Task 15: (a) `fencing()` stays "the count of `fences(i)`" and so counts a failed 9 (smallest change; F4's intent — out of the cap, the lift and M — holds by rows 3 and 10), or (b) `fencing()` keeps its codes and the four gates read a separate "any sample fences" predicate. Behaviour is identical.
- **B. `admit` moves a fencing non-forgotten sample on one agreeing answer.** Spec §7.6 says a failed 9 "fences like a held sample" and "a later review can still release it", and is silent on `admit`'s release/supersede of `movable` (l. 1396, 1473–1483), which today applies to every fencing sample that is not forgotten. If a failed 9 stays in `movable`, one fresh answer near it that agrees with the model releases it to 0 (or `Q_REF_WAIT`) when the model reproduces it, else supersedes it to 2 — outside the review. Needs a ruling before Task 15: keep that (held-sample behaviour) or exclude code 9 from `movable` like a forgotten sample (only the review moves it). `release_reproduced` (l. 932) and `judge_votes` touch code 1 only, so the refit does not move a 9 outside the review either way.
- **C. Debounced refit and `user_changes_pending`** (`mod.rs` l. 1814): derived from forgotten membership only (codes 7/8 vs the running refit's `snap_quarantine` or `fit_members.1`). Code 9 is not the user's, so a review's 9 → 0 never counts as a pending user change and never starts the debounce (`app` `worker::model::UserRefit`); the review runs at whatever refit comes next. `purge_refusal` already waits on `pending_refit`-style states (`PURGE_WAIT_REFIT` / `PURGE_WAIT_CHANGES`, `purge.rs` l. 57/61); the migration's refusal (F3) is one more check there.
- **D. The code-9 status record** goes through `sync_collection_status` (`shared.rs` l. 550), which writes every changed code except `Q_LONE` and `Q_REF_WAIT` → a 9 → 0 release is written as an ordinary status record at the end of `refit` (the background path: after the install). `forgettable` (l. 1656) allows code 0 only, so Forget never applies to a 9 with no change.
- **E. Files the plan names that `main` changed.** `crates/app/src/ui/dock/graph.rs` is **gone** (merged into `readouts.rs`, `17ba02c`): Task 12's two "E total = KE + E_KS" sites are now `readouts.rs` `CHART_TIP` l. 112, the info text l. 222 and the row hover l. 241; its warning list is l. 461. `activity/jobs.rs`' iteration hover no longer says "(Harris–Foulkes) energy": the running-solve line is l. 72–79 ("ΔE … Ha") and its hover l. 523. `activity/log.rs`: `solve_cell` l. 146, `solve_hover` l. 160, the lone-answer "E = Σ E_ref" l. 41. `activity/plots.rs` sites unchanged in text (l. 6, 238, 285, 312, 398). `dock/orbitals.rs` partial-occupation wording now l. 165–167. Other plan-named files `main` touched (content shifted, roles unchanged): `providers/src/surrogate/{mod,policy,tests}.rs`, `sim/src/{lib,provider}.rs`, `surrogate/src/{collection,quarantine,sidecar,lib}.rs`, `app/src/{replay.rs, ui/actions.rs, worker/commands.rs, ui/dock/compare.rs}`, `validate/references.md`. Re-read line numbers at each task; the "~l." numbers in Tasks 4–17 predate the merge.

---

### Task 1: V0, open-shell stability (entry gate, pre-registered stop) — **physics**

**Status (2026-11-14).** Run at kT = 1e-3 on 2026-11-13: **FAIL** (the V0 note). The user ruled on 2026-11-14:
kT = 5 mHa, with the lone-atom reference scan's stop at 1e-5 (spec §2 item 3; where the stop applies: spec §5.1).
V0 was re-run under that ruling on 2026-11-14, with Si, S, Cl (and P as N's analogue) and the spec's molecular cases
added (the V0 note's 2026-11-14 section). Steps 1–3 below are done for both runs; Step 4 (the physics review of the
verdict) and Step 6 are what remain. The steps are kept as written, with kT = 5 mHa and the 1e-5 stop where they
name a kT or a stop.

**Files:**
- Read, then append to (do not rewrite): `docs/superpowers/notes/2026-11-13-v0-open-shell-stability.md`. Another agent may already have written it. Create it if it is absent.
- Uncommitted measurement code only: `crates/providers/src/scf_stall_diag.rs` and the `SCF_SMEAR`/`SCF_TRACE2` knobs in `crates/qm-gpu/src/scf.rs`. For the CPU cross-check, add a throwaway `SCF_SMEAR` knob in `crates/qm/src/scf.rs`, also uncommitted. All of it moves to the side branch `smearing-prototype` at the end of Task 3 (F5); Task 4's `fermi_dirac` replaces the knob.
- Modify: `crates/validate/references.md`. New section "Fermi–Dirac migration", subsection "V0".

**Interfaces:**
- Consumes: the diagnosis prototype. Env `SCF_SMEAR` (kT), `SCF_TRACE2`, `DIAG_*` (header of `scf_stall_diag.rs`). Run with `cargo test -p providers --release scf_stall_diag -- --ignored --nocapture`.
- Produces:
  - the verdict per species × M: PASS or FAIL;
  - **the M V0 settled on for pseudo-O** (Task 7's V4 cases);
  - σ_path per species;
  - the GPU jitter floors;
  - U↑↑, U_c and U_s;
  - the lopsided-tetrahedron seeding result.

- [ ] **Step 1: Inventory.** If the note exists, add a section "Coverage (plan Task 1)". It holds one row per item below, marked measured / missing:
  - **Cases:**
    - pseudo C, O and F, B if the recipe accepts it, and N as the gapped control;
    - pseudo Si, S and Cl (accepted 3p open shells), and P as N's 3p analogue (gapped at M = 4, open at M = 2). Al,
      Na and Mg are refused by the recipe (all-electron under `Cores only`), Li and Be too;
    - every M the reference scan solves;
    - four sites: the node, the cube centre, (0, h/2, h/2) and a generic site;
    - both grids: h_req and Accurate;
    - all-electron O at M = 1 and 3, Balanced and Accurate;
    - OH bent and linear, O₂ singlet and triplet, H₂O, NH at auto.
  - **Per case:**
    - warm and cold solves (σ_path);
    - the DegenerateShells site spread of E, measured the same way;
    - the reached state: symmetric (TS ≈ closed form, spec §4.3) or polarised (TS ≈ 0);
    - U↑↑, U_c and U_s from the integer splittings, against the §4.7(d) predictions;
    - the per-iteration GPU eigenvalue jitter δε near convergence, and its Δρ floor ≈ 2δε/(4kT)·∫||ψ_i|² − |ψ_j|²|;
    - the lopsided tetrahedron seeded from its α = 0.1 and α = 0.05 broken-symmetry densities: each stays at TS ≲ 1e-6, with F ≈ E within the path scatter;
    - the CPU f64 cross-check of each species' worst case.
- [ ] **Step 2: Measure every item marked missing.** Run one GPU job at a time. Append the raw numbers to the note.
- [ ] **Step 3: Apply the pre-registered rules per species and M** (spec §9, "V0 in detail"). Write each verdict in the note:
  - **convergence:** ≤ 60 iterations without the stall stop;
  - **site spread:** spread(F) − spread(E_DS) ≤ 6σ_path + 1e-6 Ha;
  - **same winner:** the scan's winning M is the same at every site;
  - **jitter:** the Δρ floor sits ≥ 3× below the GPU stop rule max(tol, 2·floor·N), which for a solve with a partial level is max(that, 1e-5) (spec §5.1) [kT-5].
- [ ] **Step 4: Physics review of the verdict.** Brief the `physics-reviewer` with the note and spec §4.7(d) and §9 V0, not the code. It checks that the rules were applied exactly as pre-registered, that the U extraction is right, and that the M = 1 outcome is read against kT*_s ≳ 5 mHa.
- [ ] **Step 5: If any species or M fails, STOP.**
  - Report to the user: the failing cases with their numbers; the measured smallest passing kT, if measured; whether option (ii)'s convergence precondition holds.
  - Give options (i), (ii) and (iii) verbatim from spec §9. Do not choose.
  - Do not start Task 2.
  - After the user's ruling: amend the spec, and the "1e-3"/"fd1e-3"/"316 K" constants of this plan if kT moves. Re-run V0 under the ruling, then continue.
  - **Done 2026-11-14:** the ruling (kT = 5 mHa, 1e-5 scan stop) is applied to the spec and this plan. The re-run passes states, spread and winner everywhere, and still misses: the jitter rule for the non-winning M = 1-type states and the O₂ singlet / NH at auto, and convergence for S at M = 1 on Accurate (61 at one site) and all-electron O at M = 5 (spec §9 V0 "Outcome"). **The user accepted these as documented on 2026-11-14 ("Accept and build")**: the 1e-5 stop applies to every GPU solve with a partly occupied level, gapped GPU solves and the CPU keep their tolerance, the witness is unchanged, and a CPU f64 cross-check at 5 mHa of S at M = 1 and all-electron O at M = 5 joins Task 11's physics review.
- [ ] **Step 6: On PASS, or on the user's acceptance of the documented failures (2026-11-14, which applies),** write the V0 subsection of references.md: the study, U↑↑/U_c/U_s, the jitter floors, the seeding. Commit `git commit -o -m "docs(v0): open-shell stability at kT = 5e-3 …" -- docs/superpowers/notes/2026-11-13-v0-open-shell-stability.md crates/validate/references.md`. The diagnostic code stays uncommitted on `scf-stall` (Task 3's last step moves it to `smearing-prototype`).

**Gates:** none (a measurement). **Time:** about 30 min if the note covers everything, else about 1–2 h of serial GPU solves.

---

### Task 2: Merge `main` into `scf-stall`; re-read the touch points — **engineering**

**Order:** runs **after Task 3**, and only once `purge-forgotten` has merged to `main` (preflight order ruling).

**Files:**
- Modify: this plan (the Touch-point map).
- Modify: `docs/superpowers/specs/2026-11-13-fermi-smearing-design.md` §8, only if a touch point changed meaning, as an "Amendment (plan Task 2)" block.

**Interfaces:**
- Consumes: Task 1's verdict as the user accepted it on 2026-11-14 ("Accept and build": the documented non-winner jitter/convergence failures stand); Task 3's committed measurements and the `smearing-prototype` branch.
- Produces: the filled Touch-point map, used by Tasks 8–10 and 13–16; R and K (the merged `RECORD_VERSION` and `MAGICS.len()`), used by Tasks 13–17.

- [ ] **Step 1: Check the preconditions.**
  - `git merge-base --is-ancestor claude/charming-keller-0bc26b main` and `git merge-base --is-ancestor purge-forgotten main` both succeed (F1). If either fails, that branch has not merged: wait, and never merge it yourself.
  - Task 3 is committed, and `git show --stat smearing-prototype` lists the prototype's four files.
- [ ] **Step 2: Make sure the tree is clean of the prototype** (F5, F6). Task 3's last step already removed it; `git status --short` shows no change to `crates/qm/src/scf.rs`, `crates/qm-gpu/src/scf.rs` or `crates/providers/src/lib.rs`, and no `crates/providers/src/scf_stall_diag.rs`. If any remains, compare it with `smearing-prototype` and stop if they differ; never use a bare `git stash`. Untracked docs (the review file) stay as they are.
- [ ] **Step 3: Merge.** Run `git merge main` on its own. Resolve conflicts; only docs are expected.
- [ ] **Step 4: Re-check the ancestry and read the format constants** (F1).
  - `git merge-base --is-ancestor purge-forgotten HEAD` and `git merge-base --is-ancestor claude/charming-keller-0bc26b HEAD` both succeed.
  - Read R = `surrogate::collection::RECORD_VERSION` and K = `surrogate::sidecar::MAGICS.len()` on the merged tree, and the set of sidecar bodies the merged `MAGICS` reads (R11's old-body set, including any body `purge-forgotten` added). Write them into this plan beside the Global Constraints' format bullet.
  - Confirm that code 8 is `Q_FORGOTTEN_UNFENCED` and that no persisted code 9 exists in `surrogate::quarantine`. If 9 is taken, STOP and report: `Q_RESOLVED` needs a ruling.
- [ ] **Step 5: Verify the merged tree.** Run `cargo build --workspace --release`, then `cargo test -p units -p qm -p sim -p surrogate --release` and `cargo test -p providers --release`. Expected: PASS, i.e. main's state (the prototype is off the tree).
- [ ] **Step 6: Re-read the ten touch points on the merged tree.** Fill in the right-hand column of the map with names, files and lines.
  - The mechanism for a failed code-9 review is settled (F4: the quarantine fence `fences()`, not the misfit set M; outside `fencing()`, the held cap and purge's lift votes). Record the names it uses on `main`.
  - Where a role's meaning changed, write the mapping into spec §8 as an amendment, never a silent reinterpretation.
- [ ] **Step 7: Commit.** Run `git commit -o -- docs/superpowers/plans/2026-11-13-fermi-smearing.md docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`.

**Gates:** the build plus the "any" row. **Time:** about 12 min of tests plus the build.

---

### Task 3: The remaining A1 measurements — **physics**

**Order:** runs **before Task 2**, on the pre-merge tree (it needs only the prototype, not `main`).

**Files:**
- Create: `docs/superpowers/notes/2026-11-13-a1-measurements.md`.
- Modify: `crates/validate/references.md` ("Fermi–Dirac migration": witness cost, E‴, the V3 and V5 choices).
- Modify: the spec's §9 V2 table (an amendment, Step 0).
- The measurement code stays uncommitted on `scf-stall` (the diag harness); Step 6 moves it to `smearing-prototype`.

**Interfaces:**
- Consumes: the uncommitted prototype in the working tree (Task 1's knobs and harness).
- Produces:
  - **V3 geometry:** a near-square H₄ rectangle (a × b), the displaced atom and its in-plane axis; c = dΔ/dx; |d(TS)/dx|.
  - **V5 NVE case:** the molecule, box n × h, starting bend, dt, step count, h²/4·E_mode.
  - **Witness cost:** all-electron per-solve time.
  - **E‴ of the stiffest bonds.**

- [ ] **Step 0: Re-measure the V2 pins at kT = 5 mHa** [kT-5]. Spec §9's V2 table and D§6's F/TS values were measured at 1e-3, so they cannot pin a 5 mHa solve. Run the diagnosis harness (`scf_stall_diag`, `SCF_SMEAR=5e-3`, the D§2 cases, 81³ × 0.15, the app tolerance raised by `effective_gpu_tol`, and the partial-level stop of spec §5.1 emulated by `DIAG_TOL`=1e-5 on the cases that keep a partial level) and record per case: converged or not, iterations, F, TS. Each case must converge without the stall stop or the cap; a case that does not is a STOP (report, do not re-budget). The V2 budgets become ≈ 1.5× these counts (the spec's rule, unchanged) and the F/TS pins these values; write both into the spec's V2 table as an amendment and into the note.
- [ ] **Step 1: V3 aspect.** Use the GPU prototype on all-electron H₄ at 41³ × 0.30, σ = h (`DIAG_DS=1 SCF_SMEAR=5e-3`, `DIAG_N=41`, `DIAG_H=0.30`). Scan the aspect until:
  - c ∈ [5e-3, 1e-1] Ha/Bohr (at kT = 5 mHa: |d(TS)/dx| ≤ 0.45·c needs c ≥ 4.4e-3, and F‴ ≈ 0.2·c³/(kT)² keeps the plain |FD(δ) − FD(2δ)| = F‴δ²/2 ≤ 1e-3 up to c ≈ 0.13; the 1e-3 window was [5e-3, 5e-2]) [kT-5];
  - |d(TS)/dx| ≥ 2 mHa/Bohr;
  - the whole 4δ span (δ = 0.01 Bohr) lies in that regime;
  - the plain |FD(δ) − FD(2δ)| ≤ 1e-3 Ha/Bohr.

  Record the candidate. Task 6 confirms it on the CPU, since no CPU prototype exists.
- [ ] **Step 2: V5 NVE case (N₃).** On the pseudo-N surface at h_req:
  - Confirm that LSDA keeps N₃ linear (the bend force is restoring at small angles), with a ²Π_g ground state (TS at linear = 2 kT ln 2 = 6.93 mHa at 5 mHa, within 1e-5 Ha).
  - Find a start bend that splits the π pair by about 10 kT (50 mHa at 5 mHa [kT-5]), with h²/4·E_mode ≤ 1e-4 Ha. The larger split needs a larger bend than at 1e-3; if no bend meets both, STOP and report (the spec's NVE case needs an amendment).
  - Choose a box that keeps the face density within `box_affected_face_density(h)`.
  - If N₃ fails, substitute another linear ²Π radical (the review suggests CO₂⁺) with the same properties.
  - If none qualifies, STOP and report: the spec needs an amendment.
- [ ] **Step 3: All-electron witness cost.** H₂ and H₃⁺ at 49³ × 0.25: per-solve time, cold and warm, aufbau against the witness configuration (`DIAG_DS=1`, `SCF_SMEAR=5e-3`, so both witness arms run). Median of 5. For the README and references.md. At 5 mHa the 36 kT arm (0.18 Ha above μ) binds in every channel, not only pinned ones (spec §4.2), so also record the extra vectors per channel and the pseudised H₂O cold/warm time against DS at 57³ and 81³; V0 measured cold solves 3–9× DS for the 3p atoms at 14 Bohr (the V0 note, 2026-11-14).
- [ ] **Step 4: E‴ of the stiffest bonds.**
  - O–H (in H₂O) and N–H (in NH₃) on the default pseudo surface at h = 0.15, 7 points from 0.8 R_e to 1.0 R_e; fit (1/12)|E‴|.
  - Expected: below κ_max = 1 Ha/Bohr³ ("M7 forces" OH gives about 0.18).
  - If any value ≥ 1, STOP: κ_max is a spec constant, so it is the user's call.
- [ ] **Step 5: Commit** the note, references.md and the spec (`git commit -o -- docs/superpowers/notes/2026-11-13-a1-measurements.md crates/validate/references.md docs/superpowers/specs/2026-11-13-fermi-smearing-design.md`), on `scf-stall`, with no code.
- [ ] **Step 6: Preserve the prototype on `smearing-prototype` and take it off the tree** (F5, F6).
  - `git diff > <scratchpad>/scf-stall-diag.patch` (a second copy).
  - `git switch -c smearing-prototype`, then `git add crates/providers/src/scf_stall_diag.rs`, then `git commit -o -m "wip(prototype): the V0/A1 smearing prototype (never merge)" -- crates/qm/src/scf.rs crates/qm-gpu/src/scf.rs crates/providers/src/lib.rs crates/providers/src/scf_stall_diag.rs`.
  - `git switch scf-stall`. The four files return to `scf-stall`'s committed state; untracked docs stay. `git status --short` shows no code change.
  - Never a bare `git stash`. Never merge `smearing-prototype`.

**Gates:** none (measurement). **Time:** about 1 h GPU, serial.

---

### Task 4: `qm` Fermi–Dirac occupations (host f64) — **physics**

**Files:**
- Create: `crates/qm/src/occupations.rs` (the rule; tests in its `mod tests`).
- Modify:
  - `crates/qm/src/lib.rs`: the module, and re-exports of the items below at the crate root as `qm::…`;
  - `crates/qm/src/scf.rs`: `pub use crate::occupations::*` so `qm::scf::fermi_dirac` resolves (spec §5).
- Nothing to remove: Task 1's `SCF_SMEAR` knob left the tree in Task 3 step 6 (`smearing-prototype`). Before committing, `rg "SCF_SMEAR|SCF_TRACE2|DIAG_|scf_stall_diag" crates` returns nothing (F5).

**Interfaces:**
- Produces:

```rust
pub const ELECTRON_KT: f64 = 5e-3;          // Ha (spec §2 item 3, 2026-11-14)
pub const WITNESS_GAP: f64 = 0.1;           // Ha, from ε_{N_σ}
pub const WITNESS_TAIL_KT: f64 = 36.0;      // ε_k − μ ≥ 36 kT
pub const PARTIAL_OCCUPATION_MIN: f64 = 1e-6;
pub const GPU_PARTIAL_TOL: f64 = 1e-5;      // the GPU stop with a partial level (spec §5.1); the one definition (F11)
pub const MU_BRACKET_KT: f64 = 50.0;
pub const MAX_MU_HALVINGS: usize = 2048;
pub struct FdOccupations { pub f: Vec<f64>, pub mu: Option<f64>, pub entropy: f64, pub halvings: usize }
pub fn fermi_dirac(evals: &[f64], n: usize, kt: f64) -> Result<FdOccupations, QmError>;
pub fn fd_entropy(x: f64) -> f64;           // s at x = (ε − μ)/kT
pub fn witness_reached(evals: &[f64], n: usize, mu: f64, kt: f64) -> bool;
#[derive(Clone, Debug, PartialEq)]
pub struct PartialLevel { pub spin: u8, pub index: usize, pub eps: f64, pub f: f64 }
pub fn partial_levels(evals: &[f64], f: &[f64], spin: u8) -> Vec<PartialLevel>;
```

**Semantics:**
- `f` is in `evals`' order (any order).
- n = 0 returns all-zero f, `mu: None`, entropy 0, and needs no extra level.
- For n > 0, any of these is `QmError::Numerical`, raised before bisecting:
  - `evals.len() < n + 1`;
  - a non-finite eigenvalue;
  - a `kt` that is not finite and > 0.
- μ is the midpoint of two bisections over the spec §4.1 bracket, each halving until the midpoint equals an endpoint in f64:
  - one moving the upper end on Σf − n ≥ 0;
  - one moving the lower end on Σf − n ≤ 0.
- `halvings` is the larger count of the two. Either loop stops at `MAX_MU_HALVINGS`.
- f uses the overflow-free form (spec §4.1). It is never clipped to 0.0.
- `fd_entropy(x) = ln_1p(e^{−|x|}) + |x|/(1 + e^{|x|})`. `entropy` sums it with Neumaier compensation.
- `witness_reached` means both arms hold: ε_max ≥ ε_(n) + `WITNESS_GAP`, where ε_(n) is the n-th lowest level, and ε_max − μ ≥ 36 kT.
- `PartialLevel.index` is the level's rank in ascending order within its channel. A level is partial when min(f, 1−f) ≥ 1e-6.

- [ ] **Step 1: Write the failing tests** in `occupations.rs`:
  - `fd_degenerate_pair_half_filled`: evals [−0.3, −0.3], n = 1, kT = 1e-3 (any kT; the unit tests use explicit values). f == [0.5, 0.5] exactly; |μ + 0.3| ≤ 1e-15; |entropy − 2 ln 2| ≤ 1e-14.
  - `fd_gap_symmetric_mu`: evals [−0.3, −0.3 + 5kT], n = 1. f₀ = 1/(1 + e^{−2.5}) and f₁ = 1/(1 + e^{2.5}) within 1e-14; |μ − (−0.3 + 2.5kT)| ≤ 1e-12.
  - `fd_gapped_plateau_midpoint` [MUST-1, SHOULD-5]: evals [−0.5, −0.5, 0.1], n = 2.
    - f[0] == f[1] == 1.0;
    - 0.0 < f[2] ≤ e^{−36};
    - entropy ≤ 1e-14;
    - μ ∈ [−0.5 + 30kT, 0.1 − 30kT] and |μ − (−0.2 + 0.5kT)| ≤ 2kT.
  - `fd_empty_channel`: n = 0 → f all 0.0, `mu == None`, entropy == 0.0.
  - `fd_sums_to_n`: 200 deterministic pseudo-random level sets (k ≤ 20, 0 < n < k, spreads from 1e-6 to 20 Ha) → |Σf − n| ≤ 1e-12 and `halvings < MAX_MU_HALVINGS`.
  - `fd_non_finite_is_numerical` [SHOULD-1]: [−0.3, NaN], [−0.3, +∞] and [−∞, 0.1] each give `Err(QmError::Numerical(_))`.
  - `fd_halvings_near_zero_mu` [NOTE-1]: evals [−1e-300, 1e-300], n = 1 → `halvings < MAX_MU_HALVINGS`.
  - `fd_is_deterministic`: the same inputs twice give identical bits for f, μ and entropy.
  - `fd_entropy_matches_direct_formula`:
    - over 1 000 f in [1e-12, 1 − 1e-12], with x = ln((1−f)/f): |fd_entropy(x) − (−f ln f − (1−f) ln(1−f))| ≤ 1e-14;
    - `fd_entropy(±800.0)` is finite and ≤ 1e-300;
    - s(⅓) = 0.636514 ± 1e-6;
    - s(½) = ln 2 ± 1e-15.
  - `witness_both_arms`:
    - arm 1 fails when ε_max = ε_(n) + 0.099;
    - with arm 1 met, arm 2 fails at kt = 3e-3 when ε_max − μ = 0.105 (< 36 kT = 0.108);
    - both met → true.
  - `partial_levels_threshold`: f = 1e-6 is listed, f = 9.9e-7 is not; `index` is the ascending rank.
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p qm --release occupations`. Expected: FAIL to compile (`fermi_dirac` undefined).
- [ ] **Step 3: Implement** the items above in `occupations.rs`.
- [ ] **Step 4: Run them to check they pass.** Run `cargo test -p qm --release occupations`. Expected: all PASS.
- [ ] **Step 5: Commit** with `git commit -o -- crates/qm/src/occupations.rs crates/qm/src/lib.rs crates/qm/src/scf.rs`. Message: `feat(qm): Fermi–Dirac occupations, plateau-midpoint μ and the entropy (spec 2026-11-13 §4.1–4.3)`.

**Gates:** `cargo test -p qm --release occupations` (seconds). The crate suites run in Task 5. **Physics review:** in Task 5, together with this diff.

---

### Task 5: `qm` CPU SCF on Fermi–Dirac: the witness loop, energies, stats — **physics**

**Files:**
- Modify `crates/qm/src/scf.rs`:
  - `Occupations`;
  - `solve_shared` → the witness loop;
  - `Energy`, `ScfStats`, `ScfProgress::energy`;
  - the final energy;
  - `counts_match`;
  - the module doc "Occupations";
  - `KsState` doc.
- Modify `crates/qm/src/forces.rs`: module doc only [SHOULD-8]. The forces are −dF/dR of the Mermin functional, stationary in ψ and f, to today's floors. Rewrite the "Janak … fixed occupations" grounding and the "spec §5.7" citation.
- Modify `crates/qm-gpu/src/scf.rs`: compile shim only.
  - `Energy` and `ScfStats` get the new fields (`internal = total`, `ts = 0.0`, `entropy = 0.0`, `mu = [None; 2]`, `partial = vec![]`).
  - An `Occupations::FermiDirac` arm returns `Err(QmError::Numerical("FermiDirac: GPU support lands in plan Task 7"))`.
- Modify `docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md` §6 "SCF loop" and §11 item 2 (spec §10 wording).
- Modify `docs/superpowers/specs/2026-10-14-pseudopotential-design.md`: an amendment block at the top of §5.7, "superseded 2026-11-13 by `2026-11-13-fermi-smearing-design.md`". The why-not-integer evidence stays.

**Interfaces:**
- Consumes: Task 4's items.
- Produces:
  - `pub enum Occupations { Aufbau, DegenerateShells { tau: f64 }, FermiDirac { kt: f64 } }`. `DegenerateShells` is deleted in Task 10.
  - `pub struct Energy { kinetic, external, hartree, xc, nuclear, internal, ts, total: f64 }`, with `internal` = the sum of the five and `total = internal − ts`, bitwise.
  - `ScfStats` gains `entropy: f64`, `mu: [Option<f64>; 2]`, `partial: Vec<PartialLevel>`. `fractional`/`released` stay, empty under FD, until Task 10.
  - `ScfProgress::energy` is F_HF = Σfε + double counting − kT·S under FD.
  - `ScfSettings::for_grid` is **unchanged here** (still `Aufbau`). Task 7 flips it.

- [ ] **Step 1: Write the failing tests** in `scf.rs` `mod tests`. The heavy ones are release-only. Every system is all-electron, σ = h (ruling R3):
  - `fd_energy_invariant`: one FD solve (H₂, 33³ × 0.30). `total == internal − ts` bitwise; the five components sum to `internal` within 1e-12·|internal|.
  - `fd_h4_square_closed_form` [SHOULD-6]: H₄ square, a = 1.8, axes-aligned and centred, 41³ × 0.30, tol = eig_tol = 1e-9.
    - |ts − 4·ELECTRON_KT·ln 2| ≤ 1e-8;
    - |internal − E_DS| ≤ 1e-8, where E_DS is the same solve under `DegenerateShells { tau: DEGENERACY_TAU }` run in the test;
    - print E_DS; Task 10 replaces the DS run with this literal.
  - `fd_matches_aufbau_on_gapped` (§5): H₂ (R = 1.4) and H₂O (the `gpu_vs_cpu` geometry), 41³ × 0.30, tol = eig_tol = 1e-9, at an explicit `FermiDirac { kt: 1e-3 }` (spec ruling 46 [kT-5]): |total_FD − total_Aufbau| ≤ 1e-8 and ts ≤ 1e-12. Also print, not assert, each one's frontier gap and TS at `ELECTRON_KT` for references.md.
  - `fd_all_electron_grows_witnesses`: H₂ under FD stores ≥ n_σ + 1 orbitals per occupied channel, and `witness_reached` holds for each.
  - `fd_empty_channel_has_no_mu`: H atom → `mu[0].is_some()`, `mu[1].is_none()`, `partial` empty, ts ≤ 1e-12.
  - `fd_warm_state_from_other_scheme_is_cold` (Review Focus 1): warm states converged at `FermiDirac { kt: 2e-3 }` and at `Aufbau`, given to an FD `ELECTRON_KT` solve of the same H₂, give results bitwise equal to the cold solve.
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p qm --release fd_`. Expected: compile FAIL (no `FermiDirac`, `internal`, `partial`).
- [ ] **Step 3: Implement.**
  - `solve_shared` under FD: grow one vector at a time until `witness_reached` on the re-solved levels, capped by `MAX_WITNESS_GROWTH` (same error, its message citing `WITNESS_GAP`).
  - Occupations come from `fermi_dirac` on each step's levels; they are never mixed.
  - The final energy adds −kT·S of the final f.
  - The stall stop, mixer and convergence test stay untouched.
  - `counts_match` already compares `Occupations` with `==`, so kT is part of the warm guard. Verify that, don't add logic.
- [ ] **Step 4: Run them to check they pass.** Run `cargo test -p qm --release fd_`. Expected: PASS.
- [ ] **Step 5: Physics review of Tasks 4 and 5.** Brief the `physics-reviewer` with the diff and spec §4.1–4.4 and §5. It checks:
  - the plateau rule and both witness arms;
  - the entropy form and its 0·ln 0 limit;
  - F = E − kT·S in the final energy and in F_HF;
  - that the forces doc's −dF/dR claim matches the code's (ρ_out, f_out).

  Fix every MUST before continuing.
- [ ] **Step 6: Run the gates,** one at a time: `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min), then `cargo test -p providers --release` (~5.5 min). Expected: PASS with no pin moved; providers still select `Aufbau`/DS explicitly or by default.
- [ ] **Step 7: Commit** with `git commit -o --` the files above. First check that `rg "SCF_SMEAR|SCF_TRACE2|DIAG_|scf_stall_diag" crates` returns nothing: `-o` commits whole files, and `crates/qm/src/scf.rs` and `crates/qm-gpu/src/scf.rs` carried the prototype's knobs (now only on `smearing-prototype`, F5). Message: `feat(qm): SCF on Fermi–Dirac occupations; Energy.total = F (spec 2026-11-13 §4–5)`.

**Time:** about 12 min of gates plus the release-only tests (a few min).

---

### Task 6: V3, the FD-consistency force oracle (CPU f64) — **physics**

**Files:**
- Modify: `crates/validate/tests/oracles.rs` (new test).
- Modify: `crates/validate/references.md` ("Fermi–Dirac migration": V3 numbers).

**Interfaces:**
- Consumes: Task 3's V3 candidate; Task 5's CPU FD path. Settings are set explicitly to `FermiDirac { kt: ELECTRON_KT }`.
- Produces: the V3 geometry, pinned. Task 11's work integral reuses it, through a `fn v3_h4() -> (Grid, System, usize /*atom*/, usize /*axis*/)` helper in `crates/validate/tests/common/`.

- [ ] **Step 1: Confirm the aspect on the CPU.**
  - Run a scratch `#[ignore]`d scan at 41³ × 0.30, tol 1e-9, around Task 3's candidate. Verify:
    - |d(TS)/dx| ≥ 2 mHa/Bohr;
    - the plain |FD(δ) − FD(2δ)| ≤ 1e-3 Ha/Bohr;
    - the 4δ span in regime.
  - If it fails, re-scan within c ∈ [5e-3, 1e-1] Ha/Bohr (the 5 mHa window, spec §9 V3; F15).
  - Record the result, then delete the scratch scan.
- [ ] **Step 2: Write the test** `fd_forces_are_minus_df_dx_near_a_pinned_pair` [MUST-4] in `oracles.rs`.
  - Definitions: FD_X(d) = −[X(x+d) − X(x−d)]/(2d), a force, for X = F (`total`) and X = E (`internal`). R_X(a, 2a) = (4·FD_X(a) − FD_X(2a))/3. δ = 0.01 Bohr.
  - Assertions:
    - `(R_F(δ/2, δ) − R_F(δ, 2δ)).abs() <= 2.5e-5` (certifies the δ⁴ remainder);
    - if `(FD_F(δ) − HF).abs() >= 1e-5`: `(3.0..=5.0).contains(&((FD_F(2δ) − HF)/(FD_F(δ) − HF)))`;
    - `(HF − R_F(δ/2, δ)).abs() <= 1e-4` (the "M7 forces" Richardson bound);
    - `(HF − R_E(δ/2, δ)).abs() >= 5e-4` (it tells F's derivative from E's).
  - HF is the `qm::forces` component of the chosen atom at the unperturbed geometry.
  - **Pre-registered (review fix round 1, spec §9 V3).** On the A1 candidate the plain truncation e = F‴δ²/6 ≈ 1e-5 (F‴ ≈ 0.58 measured on the GPU), so the δ-scaling check sits at its own 1e-5 gate. Print and record e = FD_F(δ) − R_F(δ/2, δ) and b = HF − R_F(δ/2, δ) separately; the ratio is (4e − b)/(e − b) and stays in [3, 5] exactly when −e/2 ≤ b ≤ e/4. **If the ratio check is asserted and b lies outside [−e/2, e/4], stop and report: it is a spec decision, never a re-pin.**
- [ ] **Step 3: Run it.** Run `cargo test -p validate --release --test oracles fd_forces_are_minus_df_dx -- --nocapture`. Expected: PASS. A failure is investigated (systematic-debugging), never re-thresholded.
- [ ] **Step 4: Physics review of the test.** Check the stencil, the Richardson combination, the sign convention and that the thresholds are verbatim from spec §9 "V3 force oracle".
- [ ] **Step 5: Record and commit.** Write R(δ/2, δ), R(δ, 2δ), the ratio and |HF − R_E| into references.md. Commit `oracles.rs`, `common/`, `references.md`.

**Gates:** the single new test. **Time:** about 7 CPU solves at 41³ and 1e-9, about 5–15 min. The full oracle suite runs in Task 7.

---

### Task 7: `qm-gpu` on the shared host rule; FD becomes the default — **engineering**

**Files:**
- Modify `crates/qm-gpu/src/scf.rs`:
  - `Channel::solve_shared`: the witness loop + `qm::fermi_dirac`;
  - the energy (`internal`, `ts`, `total`);
  - F_HF in the progress energy;
  - `ScfStats`;
  - the module paragraph;
  - **the partial-level stop** (spec §5.1) [kT-5]: `pub use qm::GPU_PARTIAL_TOL` (defined once in Task 4; F11); the convergence test compares Δρ with `max(settings.tol, GPU_PARTIAL_TOL)` on a step whose occupations have a partial level (`qm::partial_levels` non-empty in either channel), else with `settings.tol`; it fills `ScfStats.tol_used` with the tolerance the converged step was tested against. The stall watch and the cap are unchanged;
  - the `SCF_SMEAR` and `SCF_TRACE2` knobs are already gone (they left the tree in Task 3 step 6); `rg "SCF_SMEAR|SCF_TRACE2" crates` returns nothing.
- Modify `crates/qm-gpu/src/forces.rs`: module doc (−dF/dR, as in Task 5) and the test settings at ~l. 734.
- Modify `crates/qm/src/scf.rs`:
  - `ScfSettings::for_grid` → `occupations: Occupations::FermiDirac { kt: ELECTRON_KT }`, and its doc;
  - `ScfStats` gains `pub tol_used: f64` (F9): the tolerance the converged step was tested against. On the CPU it is always `settings.tol`.
- **The hold** (F8): every caller that must keep today's rule sets `qm::Occupations::Aufbau` explicitly, with the comment "held until plan Task <n> switches it":
  - `crates/providers/src/ks.rs`, `problem`: the all-electron branch, held until Task 9 switches the surface and its tag together;
  - `crates/providers/src/ks.rs`, test `cpu_provider_matches_qm_directly`: held until Task 9;
  - `crates/providers/src/pseudo.rs`, `BondForce::settings`: held until Task 10 (the dimer procedure moves with its hash there).
  - Grep every `ScfSettings::for_grid` caller (`rg "for_grid\(" crates`, about 100 hits) and classify each as: takes FD now; sets its own occupations already; or needs the hold. Record the counts in the commit body.
- Modify `crates/validate/tests/gpu_vs_cpu.rs` (V4 cases) and `crates/validate/tests/oracles.rs` (the TS assertions).
- Modify `crates/validate/references.md`, "GPU vs CPU at fixed σ".

**Interfaces:**
- Consumes: `qm::{fermi_dirac, witness_reached, partial_levels, ELECTRON_KT, GPU_PARTIAL_TOL, WITNESS_GAP, MAX_WITNESS_GROWTH}`.
- Produces: `GpuSolver` accepts `FermiDirac`, and its `ScfStats`/`Energy` carry the same fields with the same meaning as the CPU's, including `ScfStats.tol_used` (Task 9 wires it into `EvalStats.scf_tol_used`).

- [ ] **Step 1: Write the failing tests.**
  - qm-gpu `diis_reset_mixes_only_new_map_residuals` (spec §5): fill the DIIS history with entries from map A; apply the frozen-orbital reset; push one entry measured on map B. Assert the next mix equals plain α-mixing of B's residual: zero weight on any A entry.
  - qm-gpu `gpu_fd_warm_state_from_other_kt_is_cold` (Review Focus 1): bitwise equal to the cold solve, as in Task 5.
  - qm-gpu `gpu_partial_stop_only_with_a_partial_level` (spec §5.1) [kT-5]: (a) all-electron H₂ at tol 1e-6 (gapped) converges with `tol_used == settings.tol` (asserted exactly, F14) and Δρ < 1e-6, bitwise the same stop as before the rule; (b) a lone pseudo-O at M = 3, tol 1e-6, converges with `tol_used == GPU_PARTIAL_TOL` within 60 iterations; (c) the same O on the CPU reports `tol_used == 1e-6`.
  - gpu_vs_cpu `gpu_fd_h4_square_matches_cpu` and `gpu_fd_pseudo_o_m<M>_matches_cpu`, one per M Task 1 settled on [MUST-5]: **M = 3 (the winner) and M = 1** (both symmetric at every site at 5 mHa) [kT-5]. M = 5 is gapped (no partial level, FD ≡ Aufbau there) and needs no V4 case. `pseudo_o_triplet_matches_cpu` (qm-gpu) is M = 3 with `DegenerateShells` set explicitly, so it does not move here; Task 10 converts it to FD and re-pins it (F12). **Pre-registered (F15):** if the M = 1 case stalls at the f32 floor (the stall stop or the cap, not a convergence at 1e-5), STOP and report; do not re-budget or drop the case. σ = h = 0.25, 49³. The GPU side stops at the partial-level 1e-5 (spec §5.1) and the CPU at 1e-6; V0 measured the stop's own error on the pseudo surface at ≤ 1.7 µHa in F and ≤ 3.2e-6 Ha/Bohr per force component, inside `E_BOUND_PSEUDO` = 6e-6 and `F_BOUND_PSEUDO` = 6e-5, so the bounds stay. Each asserts:
    - |F_gpu − F_cpu| and |TS_gpu − TS_cpu| within the suite's existing energy bounds (`E_BOUND`/`E_BOUND_PSEUDO`);
    - where a partial level exists, |μ_gpu − μ_cpu| ≤ `EPS_BOUND_PSEUDO`;
    - per level, |Δf| ≤ `EPS_BOUND_PSEUDO / (2.0 * ELECTRON_KT)` (= 1e-3 at kT = 5 mHa; 5e-3 at 1e-3) [kT-5];
    - Σf over each degenerate cluster (CPU levels within 1e-6 Ha of each other) agrees within 1e-9;
    - the same state class on both solvers: Symmetric if |TS − TS_cf| ≤ 0.05·TS_cf, Polarised if TS ≤ 0.05·TS_cf, anything else fails (ruling R9);
    - forces within the existing force bounds.
  - oracles: in each σ → 0 oracle (H, He, H₂), assert `stats.energy.ts <= 1e-12`.
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p qm-gpu --release diis_reset gpu_fd` and `cargo test -p validate --release --test gpu_vs_cpu gpu_fd`. Expected: FAIL (the GPU shim returns `Err`).
- [ ] **Step 3: Implement** the GPU FD path, `ScfStats.tol_used` on both solvers, the `for_grid` flip and the holds.
- [ ] **Step 4: Re-pin the moved pins.** `gpu_none_policy_is_bitwise_today` and any other pin whose settings come from `for_grid` without a hold each get the one-line reason "all-electron now solves witness vectors (FD, spec 2026-11-13)" or the measured −TS. Never delete one.
- [ ] **Step 5: Run the gates,** serially:
  - `cargo test -p qm-gpu --release` (not in CLAUDE.md's table; record its time in dev-reference "Test suites");
  - `cargo test -p validate --release --test gpu_vs_cpu` (~3.5 min);
  - `cargo test -p validate --release --test oracles` (~31 min);
  - `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min);
  - `cargo test -p providers --release` (~5.5 min; the hold keeps its pins).

  Expected: PASS, existing numbers within the existing tolerances.
- [ ] **Step 6: Commit** the files above (`git commit -o --`), after `rg "SCF_SMEAR|SCF_TRACE2|DIAG_|scf_stall_diag" crates` returns nothing (F5). Message: `feat(qm-gpu): Fermi–Dirac on the shared host rule; FD is ScfSettings' default (spec 2026-11-13 §5, V3/V4)`.

**Time:** about 50 min of gates plus qm-gpu's own suite.

---

### Task 8: The `branch_jump` work-consistency guard — **physics**

**Files:**
- Create: `crates/providers/src/branch_jump.rs` (the guard and its unit tests).
- Modify:
  - `crates/providers/src/lib.rs`;
  - `crates/providers/src/ks.rs`: `CpuKs`/`GpuKs` hold a guard; `reset` and every scene replace forget it; `eval_stats` fills the fields; module doc;
  - `crates/sim/src/provider.rs` and `crates/sim/src/lib.rs`: the `EvalStats` fields, `BRANCH_JUMP_WARNING`, `untrainable`;
  - `crates/validate/tests/surrogate.rs` (V7 refusal);
  - `crates/validate/references.md` (constants, derivation, tiers);
  - `docs/dev-reference.md` (the guard, in a new "Occupations (Fermi–Dirac)" section).

**Interfaces:**
- Produces:

```rust
pub const BRANCH_JUMP_SIGMAS: f64 = 3.0;
pub const BRANCH_JUMP_WINDOW: usize = 8;
pub const BRANCH_JUMP_REACH: f64 = 0.1;   // Bohr, largest single-atom displacement judged
pub const KAPPA_MAX: f64 = 1.0;           // Ha/Bohr³
#[derive(Clone, Copy)] pub struct NoiseTier { pub sigma_e: f64, pub sigma_f: f64 }
pub fn noise_tier(all_electron: bool, scf_tol_used: f64) -> NoiseTier;
pub struct Evaluation<'a> { pub z: &'a [u8], pub positions: &'a [[f64; 3]], pub forces: &'a [[f64; 3]],
    pub energy: f64, pub cell: sim::Cell, pub origin: [f64; 3], pub spin_abs: Option<f64>, pub tier: NoiseTier }
pub struct Judgement { pub judged: bool, pub mismatch: f64, pub bound: f64, pub flagged: bool, pub spin_change: Option<f64> }
pub struct BranchJumpGuard { /* previous evaluation, κ window */ }
impl BranchJumpGuard { pub fn new() -> Self; pub fn reset(&mut self); pub fn judge(&mut self, e: &Evaluation) -> Judgement; }
// sim:
pub const BRANCH_JUMP_WARNING: &str = "F changed by more than the forces' work";   // the short reason `untrainable` returns
pub fn branch_jump_warning(m: f64) -> String;   // the EvalStats warning (spec §6.6), m in Ha shown in mHa, one decimal
// EvalStats gains: pub branch_jump: Option<f64> /* Some(m) when flagged */, pub spin_change: Option<f64>
```

`branch_jump_warning(0.012)` is exactly "F changed by 12.0 mHa more than the forces' work: the SCF moved to another stationary state (not trained on)" (F14; Task 12 asserts this text).

**Algorithm** (spec §6.3, exactly):
- ΔF = E₁ − E₀ and W = −½ Σ_I (F_I,0 + F_I,1)·ΔR_I. So m = |ΔF − W|, which is 0 on a smooth surface.
- The noise term is n = 3·(√2·σ_E + ‖ΔR‖·σ_F/√2). The bound is b = n + 3κ̂‖ΔR‖³, with κ̂ = min(median(window), κ_max). It is κ_max alone until the window holds 8 entries.
- **Judged** only when z, the cell, the origin and the atom count all match the previous evaluation, and max_I |ΔR_I| ≤ 0.1. Any mismatch restarts: window emptied, not judged.
- **Window admission:** a judged step enters the window (m/‖ΔR‖³) only if it is unflagged and 3κ_max‖ΔR‖³ ≥ n.
- **`spin_change`** is |Δ∫|ρ↑ − ρ↓|| when N↑ = N↓. Take it from the adapter's existing spin reduction (`SpinSummary::abs_integral`'s source); `None` otherwise.
- **Tiers** (ruling R8):
  - Sample tier when `scf_tol_used ≤ SAMPLE_TOL`: pseudo (1.87e-6, 5e-5 = `PSEUDO_FORCE_NOISE`); all-electron (5.4e-6, the all-electron force path scatter from references.md, measured here as warm vs cold ×5 if none is on record).
  - App tier otherwise: measured in Step 1. Every GPU solve with a partial level stops at 1e-5 (spec §5.1) and reports `scf_tol_used` = 1e-5, so it is judged on this tier even when the provider's tolerance is the sample one [kT-5]. (`EvalStats.scf_tol_used` carries the requested tolerance until Task 9 sets it from `ScfStats.tol_used`, F9; this task's tests set the tier directly.)

- [ ] **Step 1: Measure the app tier.** The providers still run the old rule here, so drive `GpuSolver` directly with explicit `FermiDirac { kt: ELECTRON_KT }` settings (F10), on the scenes `providers::ks::problem` builds. Pseudo and all-electron H₂O at `effective_gpu_tol(INTERACTIVE_TOL)`, warm vs cold, 5 repeats: σ_E and the largest per-component σ_F. Add a pinned case, linear OH at the sample tolerance (it stops at the partial-level 1e-5), because that tier now also holds pinned teacher solves; the tier is the larger of the two cases' scatters. Record them in references.md as the constants `noise_tier` returns [kT-5].
- [ ] **Step 2: Write the failing tests.**
  - `branch_jump.rs`, on an analytic Morse dimer (D = 0.2 Ha, a = 1.5 Bohr⁻¹, R_e = 1.8), pseudo sample tier:
    - `synthetic_jump_is_flagged`: 0.02 Bohr steps; from step 10 on, E shifted by +12 mHa. Flagged at step 10 only.
    - `smooth_morse_never_flagged`: steps of 0.005, 0.02 and 0.05 Bohr across R ∈ [0.8, 2.0]·R_e, plus one 0.1 Bohr step with an empty window. Zero flags.
    - `over_reach_not_judged`: one atom moves 0.11 Bohr → `judged == false`.
    - `relax_then_jump_flagged`: 50 steps of 1e-3 Bohr (the window stays empty), then a 0.02 Bohr step carrying +12 mHa → flagged.
    - `alternating_jumps_stay_flagged`: ±12 mHa on every other 0.02 Bohr step for 40 steps → every jump flagged, and no flagged step ever enters the window.
    - `cell_or_origin_change_restarts`: the cell n, h or the origin changes → not judged, window emptied.
    - `atom_set_change_restarts` (Review Focus 2): add an atom, remove one, swap two species → not judged, no panic.
  - sim: `untrainable_reports_branch_jump` → `Some(BRANCH_JUMP_WARNING)`; `untrainable_branch_jump_warning_text`: `branch_jump_warning(0.012)` equals the text above.
  - validate surrogate (V7) `branch_jump_step_is_not_trained_on`: a synthetic teacher (the analytic Morse dimer wrapped in `BranchJumpGuard`) streams a 12 mHa switch. That step is warned and absent from the trained samples; the steps before and after are trained on.
- [ ] **Step 3: Run them to check they fail.** Run `cargo test -p providers --release branch_jump` and `cargo test -p sim --release untrainable`. Expected: compile FAIL.
- [ ] **Step 4: Implement.** `untrainable()` order: core overlaps, then `branch_jump`, then the old `occupation_changed`. Task 9 removes the last. The warning in `EvalStats` is `branch_jump_warning(m)`, spec §6.6's text with m in mHa.
- [ ] **Step 5: Run them to check they pass.** Run the commands from Step 3. Expected: PASS.
- [ ] **Step 6: Physics review.** The reviewer checks:
  - the factor 3, the window of 8, the 0.1 Bohr reach;
  - κ_max = 1 against Task 3's measured (1/12)|E‴| of O–H and N–H;
  - the noise term's √2 factors;
  - the trapezoid W and its sign;
  - the window admission rule;
  - the coverage statement: sampling-pass solves and sparse audits are not judged [R-NOTE-b].

  None of them may be loosened.
- [ ] **Step 7: Run the gates,** serially: `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min), `cargo test -p providers --release` (~5.5 min), `cargo test -p validate --release --test surrogate` (~12–17 min).
- [ ] **Step 8: Commit** the files above, after `rg "SCF_SMEAR|SCF_TRACE2|DIAG_|scf_stall_diag" crates` returns nothing (`crates/providers/src/lib.rs` carried the prototype's mod line; F5).

**Time:** about 35 min of gates plus about 10 min to measure the tier.

---

### Task 9: The providers' surface switch: FD everywhere, new tags, `partial` — **engineering**

**Files:**
- Modify `crates/providers/src/ks.rs`:
  - `problem`: FD for every scene; the Task 7 holds in `problem` and in the test `cpu_provider_matches_qm_directly` removed (`BondForce::settings` keeps its hold until Task 10);
  - the module doc: "nothing pseudised is exactly today's problem" rewritten;
  - `OccupationWatch` deleted;
  - `eval_stats` maps `partial`, `internal`, `ts`, and sets `EvalStats.scf_tol_used = Some(stats.tol_used)` (F9), so a partial-level GPU solve reports 1e-5 and Task 8's guard judges it on the app tier.
- Modify `crates/providers/src/pseudo.rs`: the new tag format; `is_all_electron_tag`; `legacy_tag_of`; module doc.
- Modify `crates/providers/src/surrogate/policy.rs`: the label-noise prior (touch point 7) reads `is_all_electron_tag`.
- Modify every other `"none"` literal that means the all-electron surface: `rg '"none"' crates --type rust`, each hit classified as either the scene's policy word (unchanged) or the surface (→ predicate). Record the count in the commit body.
- Modify `crates/sim/src/provider.rs` and `crates/sim/src/lib.rs`:
  - `SharedShell` → `PartialLevel { spin, index, eps, f }`;
  - `EvalStats.fractional` → `partial`;
  - `EnergyBreakdown` gains `internal` and `ts`;
  - `occupation_changed` and `OCCUPATION_CHANGE_WARNING` removed.
- Modify every consumer that rebuilds a total from components (re-grep `kinetic +`, `.xc +`, `components(`): `crates/qm-gpu/src/scf.rs` (energy assembly, done in Task 7; verify), `crates/sim/src/dynamics.rs` (`energy_ks` doc), `crates/validate/tests/gpu_vs_cpu.rs` `components()`, the app breakdown, the sidecar, `sim::references`.
- Modify `crates/app/src/ui/pseudo.rs`, `occupation_lines`: compile-level change to per-level rows. The wording comes in Task 12.
- Modify the synthetic teachers in `crates/providers/src/sampling/tests.rs` and `crates/providers/src/surrogate/tests.rs`: they set `branch_jump` instead of `occupation_changed`.
- Create `crates/providers/src/fd_regression.rs` (V2, `#[cfg(test)]`). The diag harness is not on the tree (it left in Task 3 step 6); read its generators with `git show smearing-prototype:crates/providers/src/scf_stall_diag.rs` and never merge that branch.
- Modify: references.md and `docs/dev-reference.md` ("Occupations (Fermi–Dirac)": tags, `legacy_tag_of`, re-pins; "SCF stall stop": no release restart).

**Interfaces:**
- Consumes: Task 8's guard; Tasks 5 and 7's `ScfStats`.
- Produces:
  - `pub fn is_all_electron_tag(tag: &str) -> bool`: `"none"` or a `"none/"` prefix.
  - `pub fn legacy_tag_of(policy: IonicPolicy) -> String`: the tag the last DS build computed for `policy` under the current dimer backend. The all-electron predecessor is `"none"`, chosen by the caller through `is_all_electron_tag`.
  - `sim::PartialLevel`.
  - `EnergyBreakdown { …, internal, ts, total }`.

- [ ] **Step 1: Record the legacy strings before any change.** On the unmodified tree, print `provenance_tag` for every `IonicPolicy`, under both `set_dimer_backend(Cpu)` and `Gpu`. Save them into the new test as literals. These are the only source of truth for `legacy_tag_of`.
- [ ] **Step 2: Write the failing tests.**
  - pseudo.rs:
    - `tag_strings_are_fd`: the all-electron tag is `"none/fd5e-3/pin1e-5"`; a pseudised tag matches `^<word>/[0-9a-f]{16}/fd5e-3/stop2x5e-8/pin1e-5$`.
    - `legacy_tag_of_matches_recorded_ds_strings`: every policy × backend equals the Step 1 literal.
    - `is_all_electron_tag_cases`: `"none"` and `"none/fd5e-3/pin1e-5"` are true; `"cores/0123456789abcdef/fd5e-3/stop2x5e-8/pin1e-5"` and `""` are false.
  - ks.rs: the Aufbau assert (~l. 3179) becomes `FermiDirac { kt: ELECTRON_KT }`, with the re-pin reason. The stall-stop bitwise pins are re-pinned with the reason.
  - ks.rs `scf_tol_used_reports_the_partial_stop` (F9): a lone pseudo-O at M = 3 through the GPU provider at the sample tolerance reports `scf_tol_used == Some(GPU_PARTIAL_TOL)`; gapped H₂ reports the requested tolerance.
  - `fd_regression.rs` (V2, GPU, release-only). Scenes go through `providers::ks::problem`, cold, at 81³ × 0.15, at the app tolerance raised by `effective_gpu_tol`, with the stall stop on. **Budgets and pins come from spec §9's V2 table as amended by Task 3 step 0 (the 5 mHa re-measurement; F7).** The 1e-3 counts and D§6's F/TS values are stale and are not used.
    - `fd_converges_default_policy_stallers`: H₄ rhombus, lopsided tetra, H₄ seed 1, H₅ seed 3, H₈ seed 8, each within its amended budget.
    - `fd_converges_all_electron_failures`: H₅ ring (radius 1.531, xy plane), H₆ seed 5, lopsided tetra, each within its amended budget.
    - `fd_converges_all_electron_h2o_plus_2h`: H₂O + 2 H alone (434 s of GPU at 5 mHa in A1), run as its own GPU batch (never in the same `cargo test` invocation as the other GPU cases, with the cool-down between); it logs the extra (witness) vectors each channel grows. Same budget and pins as the table.
    - Each case: converged, no stall stop and no cap; |F − F_pin| ≤ 1e-5 and |TS − TS_pin| ≤ 1e-5 against the amended table. The seeded geometries are reduced from the prototype's generators (`smearing-prototype`).
    - `fd_controls_keep_their_energies`: H₂ within 2e-6 and H₂O within 6e-6 of their amended-table values.
    - A mismatch is investigated, never re-pinned blindly.
  - `all_electron_h_minus_witness_cap` [NOTE-6]: all-electron H⁻ in the largest box the app offers (Accurate, h = 0.15; name the box in the test). Pin the measured outcome: either it converges, or it returns `Err(QmError::Numerical)` whose message names `MAX_WITNESS_GROWTH`.
- [ ] **Step 3: Run them to check they fail.** Run `cargo test -p providers --release tag_ legacy_tag is_all_electron fd_regression all_electron_h_minus scf_tol_used`. Expected: FAIL.
- [ ] **Step 4: Implement** the switch, the tags, the predicate, the `sim` types, the consumers and the app's compile change. Delete the occupation watch. Run `cargo build --workspace --release`.
- [ ] **Step 5: Run them to check they pass.** Run the command from Step 3. Expected: PASS.
- [ ] **Step 6: Run the gates,** serially:
  - `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min);
  - `cargo test -p providers --release` (~5.5 min, plus V2's 9 GPU cases, about 10 min);
  - `cargo test -p validate --release --test surrogate` (~12–17 min);
  - `cargo test -p validate --release --test gpu_vs_cpu` (~3.5 min).
- [ ] **Step 7: Commit** the files above, after `rg "SCF_SMEAR|SCF_TRACE2|DIAG_|scf_stall_diag" crates` returns nothing (F5).

**Time:** about 40 min of gates.

---

### Task 10: The dimer procedure, deleting `DegenerateShells`, the lone-atom rule, the reference scan — **physics**

**Files:**
- Modify `crates/qm/src/radial/ledger.rs`:
  - `DIMER_PROCEDURE` gains the occupation rule and kT among its hashed constants;
  - `DIMER_PROCEDURE_DS`, a frozen copy of the pre-change procedure, is used only by `legacy_tag_of`.
- Modify `crates/providers/src/pseudo.rs`: the dimer gate's 3-D solves use FD (Task 7's `Aufbau` hold in `BondForce::settings` removed; F8); the legacy hash uses `DIMER_PROCEDURE_DS`. `DIMER_PROCEDURE`'s stop reads `qm::GPU_PARTIAL_TOL`, never a literal (F11).
- Modify `crates/qm-gpu/src/scf.rs` tests (F12): `pseudo_o_triplet_matches_cpu` and every other test that sets `DegenerateShells` explicitly are converted to `FermiDirac { kt: ELECTRON_KT }` and re-pinned, each moved bound with its measured −TS or re-pin reason beside it.
- Delete everywhere:
  - `Occupations::DegenerateShells`, `degenerate_shell_occupations`, `released_occupations`, `decide`, `shared_runs`, `FractionalShell`, `DEGENERACY_TAU`, `CANDIDATE_TAU`;
  - `ScfStats.fractional` and `released`;
  - the phase-2 stall-watch restart and the "frozen by rank" state, in both loops.

  Files: `crates/qm/src/{scf,lib}.rs`, `crates/qm-gpu/src/{scf,forces}.rs`, `crates/providers/src/orbitals.rs`, `crates/validate/tests/{eggbox/mod.rs,gpu_vs_cpu.rs,pseudo.rs}`.
- Modify `crates/qm/src/scf.rs` tests: `fd_h4_square_closed_form`'s DS run becomes the literal E_DS printed in Task 5, with that reason.
- Modify `crates/providers/src/surrogate/lone.rs` and `crates/providers/src/surrogate/tests/lone.rs` (touch point 5).
- Modify `crates/sim/src/references.rs`: docs only. The scan keeps the lowest F; `AtomReference::fractional` means "has a partial level".
- Modify `crates/validate/tests/pseudo.rs`: `degenerate_spread_at_low_symmetry_sites` is repurposed as V0's site study under FD.
- Create `crates/validate/tests/fixtures/ds_era_labels.txt` (Step 1).
- Modify references.md: every moved reference with its TS and closed form, and the approximation gaps of spec §10.
- Modify the pseudopotential spec §5.5: the tag text.
- Modify `docs/dev-reference.md`.

**Interfaces:**
- Produces:
  - `pub(super) fn closed_spin_channels(z: u8, partial: Option<bool>) -> bool`: true iff z == 1, or `partial == Some(false)`.
  - The V7 fixture file: the policy, the grid and the box, then per case the geometry, E_DS and the DS forces. Task 14's V7 test reads the policy, grid and box from it.

- [ ] **Step 1: Capture the DS-era labels for V7 before deleting anything** (F13). H₂ at R = 1.4, linear OH and a near-square H₄, under the default pseudised policy (`Accepted`) on one fixed grid and box chosen here and written into the fixture, solved under `DegenerateShells` at the sample tolerance: E and forces at full f64 precision. Choose the H₄ aspect so that its internal energy moves by |ΔE| ≥ 3ε_E = 3e-5 Ha between the DS solve and an FD solve at `ELECTRON_KT` (so it is class S outside the false-S band); record that |ΔE| in the fixture. Write the labels to the fixture.
- [ ] **Step 2: Write the failing tests.**
  - lone.rs: `closed_spin_channels(1, None)` is true; `(2, Some(false))` true; `(8, Some(true))` false; `(2, None)` false.
  - `polarised_or_partial_o_is_declined` (V8): an O reference whose winning solve has partial levels → declined.
  - pseudo.rs (providers):
    - `dimer_procedure_hashes_fd`: `DIMER_PROCEDURE`'s debug text names `FermiDirac`, `5e-3` and the GPU partial stop `qm::GPU_PARTIAL_TOL` (1e-5; the gate's 3-D solves run on the GPU);
    - `core_recipe_hash_unchanged`: equal to the literal recorded on the tree before this task.
- [ ] **Step 3: Run them to check they fail.** Expected: FAIL.
- [ ] **Step 4: Implement, then delete DS.** `rg "DegenerateShells|degenerate_shell_occupations|released_occupations|DEGENERACY_TAU|CANDIDATE_TAU|FractionalShell|shared_runs|fn decide" crates` must return nothing. Docs may cite them as history.
- [ ] **Step 5: Run them to check they pass**, then `cargo build --workspace --release`.
- [ ] **Step 6: Physics review.** The reviewer checks:
  - the lone rule's "no partial level" argument (spec §6.4): 27.6 kT = 138 mHa at 5 mHa [kT-5];
  - each moved reference against its closed-form TS (spec §4.3 table);
  - that the scan's winning M is unchanged;
  - that the generation records are bitwise identical with only `recipe_hash` moved.
- [ ] **Step 7: Run the gates,** strictly serial:
  - `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min);
  - `cargo test -p providers --release` (~5.5 min plus V2);
  - `cargo test -p qm-gpu --release` (F12): `pseudo_o_triplet_matches_cpu` and the other converted tests at their new pins;
  - `cargo test -p validate --release --test gpu_vs_cpu` (~3.5 min);
  - `--test pseudo_gen` (~9 min): `generation_record_table_matches_references_md` stays bitwise; H and He regenerate once (about 1 min);
  - `--test pseudo` (~1 h);
  - `--test pseudo_oracles` (~1.5–2 h): the dimer verdicts and R_e/ω reproduce within the backend margin. The O-reference pins (`gpu_oxygen_reference_multiplicity` in `crates/providers/src/sampling/tests.rs`, run by the providers gate, and qm-gpu's `pseudo_o_triplet_matches_cpu`) move by the measured −TS, recorded with the closed form (O at M = 3: 3 kT s(⅓) = 9.5477e-3 Ha at 5 mHa; V0 measured TS within 1e-9 of it at every site) [kT-5]. The scan's winner stays M = 3, and it now converges through the partial-level 1e-5 stop.
- [ ] **Step 8: Commit** the files above.

**Time:** about 3–3.5 h of gates.

---

### Task 11: V5 dynamics, and a physics review of the whole solver half — **physics**

**Files:**
- Modify: `crates/validate/tests/dynamics.rs` (two tests); `crates/validate/references.md` ("Dynamics" and "Fermi–Dirac migration": the V5 bounds with their derivation).

**Interfaces:**
- Consumes: Task 6's `v3_h4()`; Task 3's NVE case.

- [ ] **Step 1: Time the case and pre-register the NVE bounds.** Task 3's case is CO₂⁺ (spec §9 V5 2, amended): 71³ × 0.20, C on a grid node, β = 70° at R = 2.226, at rest, dt = 0.25 fs.
  - First time a few warm MD steps (the per-step cost at 71³ was not measured in A1) and measure the path scatter σ_path of F for this system under the 1e-5 partial-level stop (warm vs cold at the start geometry).
  - Steps: ≥ 2 bend periods (≈ 600 steps; 300 steps = 0.075 ps cannot resolve drift at `DRIFT_BOUND` 2e-4 Ha/ps against the Verlet swing). If 600 steps exceed one 15-min GPU batch, split the run into checkpointed batches.
  - Then write the drift and fluctuation bounds (fluctuation 3 × (h²/4·E_mode + 2σ_path), h²/4·E_mode = 1.4e-5 Ha) into references.md before the test exists, derived by `nve_energy_drift_is_bounded`'s own method. They are never scaled per atom.
- [ ] **Step 2: Write the tests.**
  - `work_integral_matches_free_energy_change` [MUST-6]. CPU f64, `v3_h4()`, tol = eig_tol = 1e-9.
    - Move the chosen atom 0.2 Bohr along its axis across the TS swing: 41 points.
    - W₄₀ = Simpson over 41 points of F_HF·dx, the work the forces do (ruling R10). W₂₀ is the same over every other point.
    - σ_E and σ_F come from warm vs cold solves at the start point.
    - bound = |W₄₀ − W₂₀|/15 + 3·(0.2·σ_F + √2·σ_E).
    - Assert `(ΔF + W₄₀).abs() <= bound` and `(ΔE + W₄₀).abs() >= 5.0 * bound`.
  - `fd_nve_conserves_f_plus_k_through_the_ts_swing`: Task 3's case (CO₂⁺, box, bend, dt) with Step 1's step count (≥ 2 bend periods).
    - F + K stays within the Step 1 bounds;
    - max TS − min TS ≥ 1e-3 Ha along the run;
    - the face density stays ≤ `box_affected_face_density(h)` throughout.
- [ ] **Step 3: Run them.** Run `cargo test -p validate --release --test dynamics -- --nocapture`. Expected: PASS, and the existing gapped tests unchanged. A failure is debugged, never re-bounded.
- [ ] **Step 4: Physics review of the whole solver half** (Tasks 4–11 diff):
  - FD and the energies;
  - GPU parity;
  - `branch_jump`;
  - the V5 derivations;
  - that no tolerance moved without its TS reason;
  - the CPU f64 cross-check at 5 mHa of pseudo-S at M = 1 and all-electron O at M = 5, V0's two accepted convergence misses (the user's 2026-11-14 ruling; Task 1 step 5): run both on the CPU FD path and record converged or not, iterations and F beside the GPU numbers in references.md.
- [ ] **Step 5: Commit** `dynamics.rs` and `references.md`.

**Gates:** `--test dynamics` (~6 min, plus the work integral's 41 + 2 CPU solves at 41³, about 30 min, plus the NVE run). **Time:** about 45–60 min.

---

### Task 12: The UI: readouts, breakdown, Activity, orbitals, Compare — **engineering**

**Files** (F14: the sites are named now; line numbers are from the pre-merge tree of 2026-11-14, so find each by name after Task 2's merge):
- Modify:
  - `crates/app/src/ui/pseudo.rs`: `energy_label` (~l. 111), a new `ENERGY_TOOLTIP`, `occupation_lines` (~l. 118);
  - `crates/app/src/ui/dock/readouts.rs`: the breakdown rows; the "E total" row (~l. 106, "KE + E_KS: …") and its info text (~l. 88); the warning list (`d.eval.warnings`, ~l. 215);
  - `crates/app/src/ui/dock/graph.rs`: the "E total = KE + E_KS" texts (~l. 33 and ~l. 123);
  - `crates/app/src/ui/activity/jobs.rs`: the iteration-energy hover (~l. 240, "(Harris–Foulkes) energy") → F_HF;
  - the solve-log line: `crates/app/src/ui/activity/log.rs`, `solve_cell` (~l. 130) and `solve_hover` (~l. 144);
  - `crates/app/src/ui/dock/orbitals.rs`: per-level f, with the `DegenerateShells` wording (~l. 161) rewritten;
  - `crates/app/src/ui/dock/compare.rs`: the T_el note;
  - the reference-energy sites (spec §4.7(i)): `crates/app/src/ui/activity/plots.rs` (the module doc ~l. 6, the parity toggle ~l. 238, the axis label ~l. 285, the parity hover ~l. 312: "(E − Σ E_ref)/N"; the dimer-curve label ~l. 398: "E − E_ref(a) − E_ref(b)") and `crates/app/src/ui/activity/log.rs` (the lone-answer text ~l. 53: "E = Σ E_ref").
- Modify `README.md`:
  - the readouts;
  - "SCF converges on loose hand-built clusters";
  - the all-electron performance numbers from Task 3, if they moved.
- Modify `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md` item 4: a note that the shared-shell rows are now per-level f.
- Modify `docs/dev-reference.md` (viewer).

**Interfaces:**
- Consumes: `EvalStats.{partial, energy.internal, energy.ts, branch_jump}`, `qm::ELECTRON_KT`, and `units::hartree_to_kelvin` (rule 3: the K value is converted only here).

- [ ] **Step 1: Write the failing tests** (app unit tests). Strings are verbatim from spec §6.6:
  - `energy_label_reads_f`: no pseudised species → `"F (electronic free energy, T_el 1579 K)"`; any pseudised species → `"F (electronic free energy, valence, pseudopotential, T_el 1579 K)"`. The 1579 comes from `units::hartree_to_kelvin(qm::ELECTRON_KT)` rounded.
  - `energy_tooltip_text`: equals the §6.6 tooltip verbatim.
  - `breakdown_rows`: the labels are kinetic, external, Hartree, xc, nuclear, `E (internal)`, `−TS (electronic entropy, T_el 1579 K)`, `F`.
  - `e_total_reads_ke_plus_f`: the row tip is `"KE + F: conserved in NVE without tools"`.
  - `solve_log_ts_line`: ts = 9.5477e-3 (a lone O at M = 3 at 5 mHa) → contains `"−TS = 9.5 mHa"`; ts = 9e-7 → no −TS text.
  - `compare_note_when_partial`: ts ≥ 1e-6 → `"ours is at T_el = 1579 K with partly occupied levels; PySCF's is integer-occupied"`; otherwise absent.
  - `orbital_rows_per_level_f`: f < 1e-6 shows as empty; f = ⅓ shows its f.
  - `branch_jump_warning_shown`: an `EvalStats` carrying `sim::branch_jump_warning(0.012)` puts exactly "F changed by 12.0 mHa more than the forces' work: the SCF moved to another stationary state (not trained on)" into the readouts' warning list (the formatted text, not the `BRANCH_JUMP_WARNING` key; F14).
  - `reference_energies_say_free_energy` (spec §4.7(i)): each reference-energy site named under Files says the references are free energies at T_el.
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p app --release --bin app -- ui::`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to check they pass**, then run the gate `cargo test -p app --release` (~10 min, replays included; their energy expectations are gapped).
- [ ] **Step 5: List the user's hand-tests in the commit body** (memory: testing split). They are: the readout row and tooltip, the breakdown, an open-shell O's orbital rows, the Activity F_HF, and the Compare note on linear OH. Commit.

**Time:** about 10 min of gates.

---

### Task 13: The collection format: read-only open, record R + 1, code 9, `migration.txt`, the sidecar bump — **engineering**

**Files:**
- Modify `crates/surrogate/src/collection.rs`:
  - `RECORD_VERSION = R + 1` (R = the merged value, Task 2 step 4; F1); the reader takes 1 ..= R + 1;
  - `open_read_only`, which never cleans purge's `samples.bin.new`/`samples.bin.bak.new` (F3: it leaves them byte-identical);
  - the layout table gains `migration.txt`;
  - module doc.
- Modify `crates/surrogate/src/quarantine.rs`:
  - `Q_RESOLVED = 9` (code 8 is `purge-forgotten`'s `Q_FORGOTTEN_UNFENCED`, untouched; F1);
  - `is_collection_code` (ruling R12);
  - the docs paragraph "An older build meeting code 9", in the style of code 7's.
- Modify `crates/surrogate/src/sidecar.rs`: one new magic `ASMODEL<K + 1>` with K = Task 2's `MAGICS.len()` (F1); its reader; writers emit it and leave code-9 samples out (ruling R11).
- Create `crates/surrogate/src/migration.rs` (the `migration.txt` model).
- Modify `crates/surrogate/src/lib.rs`.
- Modify `crates/app/src/update/formats.rs`: the table doc. The versions follow the constants.

**Interfaces:**
- Produces:

```rust
pub const Q_RESOLVED: u8 = 9;
pub fn is_collection_code(code: u8) -> bool;   // is_persisted_code(code) || code == Q_RESOLVED
pub struct ReadOnlyCollection { pub key: SurfaceKey, pub samples: Vec<StoredSample>, pub codes: Vec<u8>,
    pub references: Vec<(u8, usize, CollectionReference)>, pub damaged_tail: Option<String> }
pub enum ReadOnlyError { LiveWriter { pid: u32 }, Unreadable(String), NewerFormat(u8) }
impl Collection { pub fn open_read_only(dir: &Path) -> Result<ReadOnlyCollection, ReadOnlyError>; }
pub enum MigrationState { Running, Complete, Reviewed }
pub struct MigrationLog { pub source_slug: String, pub source_key: String, pub build: String, pub date: String,
    pub source_len: u64, pub source_hash: u64 /* the source samples.bin at the pass's start (F3) */,
    pub cursor: usize, pub state: MigrationState, pub counts: [usize; 4] /* U, T, S, not migrated */,
    pub not_migrated: Vec<(usize, String)>, pub references: Vec<(u8, usize, Option<f64>, Option<f64>)>,
    pub class_u_max: (f64, f64), pub review: Vec<(usize, f64, bool)> /* new index, ratio, released */ }
pub fn read_migration(dir: &Path) -> Result<Option<MigrationLog>, String>;
pub fn write_migration(dir: &Path, log: &MigrationLog) -> Result<(), String>;  // temp file + rename
```

`StoredSample` and `CollectionReference` are the existing record types; use whatever names `collection.rs` has.

- [ ] **Step 1: Write the failing tests** (surrogate crate):
  - `read_only_open_leaves_every_byte`: hash every file in a fixture folder; `open_read_only`; the hashes are identical; no `.lock` is created.
  - `read_only_open_damaged_tail` (Review Focus 5): truncate the last record mid-payload. Every whole record is read; `damaged_tail.is_some()`; the bytes are identical afterwards.
  - `read_only_open_refuses_live_writer`: a `.lock` holding the test's own pid → `Err(LiveWriter { pid })`. A stale pid → it reads, and the lock file is untouched.
  - `read_only_open_leaves_purge_temporaries` (F3): a fixture folder holding `samples.bin.new` and `samples.bin.bak.new` → `open_read_only` reads `samples.bin`; both temporaries are byte-identical afterwards and still present.
  - `records_are_new_version_and_older_read`: new appends carry `RECORD_VERSION` (R + 1); a version-R fixture and a version-1 fixture read; a version R + 2 record is refused as "written by a newer build" with the file untouched. The test names the versions through the constant, never as literals.
  - `code_9_persists_in_collections_not_sidecars`: `is_collection_code(9)` and `!is_persisted_code(9)`; `is_persisted_code(8)` still holds (purge's code); a sidecar written from a store holding a code-9 sample omits it.
  - `sidecar_new_magic_round_trips` for each old body in R11's set (the bodies the merged `MAGICS` reads from 6 up, including any `purge-forgotten` added; Task 2 step 4). `MAGICS.len()` is K + 1.
  - `migration_log_round_trips` (including `source_len`/`source_hash`) and `migration_log_write_is_atomic` (a crash leaves the previous file whole).
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p surrogate --release read_only records_are code_9 sidecar_new migration_log`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to check they pass.**
- [ ] **Step 5: Run the gates:** `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min), `cargo test -p providers --release` (~5.5 min), `cargo test -p app --release` (~10 min; the formats tests now read collection R + 1 and the new sidecar version).
- [ ] **Step 6: Commit** the files above.

**Time:** about 22 min of gates.

---

### Task 14: The migration engine: predecessor, re-solve pass, classification, writes, resume — **engineering** (physics-reviewed)

**Files:**
- Create `crates/providers/src/sampling/migrate.rs` (the pass and `classify`).
- Modify `crates/providers/src/sampling/mod.rs`: the pass kind, sharing the GPU with the live scene, Pause/Resume/Cancel, the Activity line.
- Modify `crates/providers/src/surrogate/shared.rs` (touch point 6): `predecessor_of` and the offer state on attach.
- Modify `crates/sim/src/provider.rs`: the `Trainable` methods below.
- Tests: `crates/providers/src/sampling/tests.rs` and `crates/validate/tests/surrogate.rs` (V7).

**Interfaces:**
- Consumes:
  - Task 13's `open_read_only`, `MigrationLog` (with `source_len`/`source_hash`), `Q_RESOLVED`;
  - `purge-forgotten`'s purge entry point and `Q_FORGOTTEN_UNFENCED` (Touch-point map rows 9–10);
  - Task 9's `legacy_tag_of` and `is_all_electron_tag`;
  - Task 10's fixture;
  - `PSEUDO_FORCE_NOISE`, `box_affected_face_density`, `Collection::offer_reference`, the multiplicity scan.
- Produces:

```rust
pub const MIGRATION_EPS_E: f64 = 1e-5;
pub const MIGRATION_EPS_F: f64 = 3.0 * PSEUDO_FORCE_NOISE;   // 1.5e-4
pub enum MigrationClass { Unchanged, EntropyOnly, StateChanged }
pub fn classify(old_e: f64, old_f: &[[f64; 3]], new_free: f64, new_internal: f64, new_f: &[[f64; 3]]) -> MigrationClass;
pub fn migrated_code(old: u8, class: MigrationClass) -> u8;   // U, T: old (8 stays 8); S: Q_RESOLVED (9), except 7 → 7, 8 → 7 and 2 → 2 (1 → 9)
pub fn predecessor_of(root: &Path, key: &SurfaceKey) -> Option<PathBuf>;   // exact key.txt match, legacy ionic tag
// sim::Trainable
fn migration(&self) -> Option<sim::MigrationStatus>;   // offer / running / complete, with counts and the estimate
fn start_migration(&mut self) -> Result<(), String>;
pub enum MigrationPhase { Offered, Running, Paused, Complete, Reviewed }
pub struct MigrationStatus { pub source: String, pub samples: usize, pub done: usize, pub counts: [usize; 4],
    pub eta_s: Option<f64>, pub state: sim::MigrationPhase }
```

- **Per sample, in file order** (spec §7.3):
  - The cold solve uses the key's h, σ and tolerance, the sample's own `grid_n` and its stored positions bit for bit, and multiplicity auto.
  - The checks: converged; no solver warning; face density ≤ `box_affected_face_density(h)`; the recorded multiplicity reproduced.
  - Classify, then write by spec §7.5:
    - U appends the old record bit for bit, plus a status record if its code ≠ 0;
    - T appends the new label with the old metadata and the old code;
    - S appends the new label with `migrated_code`.
  - Masked samples are compared on all atoms; mask and flag are carried.
  - Then rewrite the cursor.
- **References:** re-solved by the scan (lowest F) and offered through `offer_reference`. Old and new values go into the log.
- **Copied or dropped:** `bond_experience.txt` is copied; `history.bin` is not carried; `README.txt` is regenerated.
- **Forgotten-unfenced (F2):** a code-8 sample (`Q_FORGOTTEN_UNFENCED`, purge-forgotten) keeps 8 under U and T; under S it goes to 7: the user's Forget survives, and the fence returns, because the lift judged the old label.
- **Purge interplay (F3):**
  - the pass records the source `samples.bin`'s length and hash in `migration.txt` at its start; on resume, a mismatch (the old folder was purged or appended to by another build) restarts the pass from cursor 0 with the counts reset, the already-appended records counted as done by the duplicate rule of Review Focus 4;
  - a purge of forgotten samples is refused on the new folder while the pass is Running, with the reason "a re-solve pass is running".
- **Lock:** a live lock on the old folder refuses with the spec §7.1 reason.
- **Estimate:** a count and an estimate before the start, updated from the median after 20 solves.

- [ ] **Step 1: Write the failing tests** (providers, synthetic teacher unless noted):
  - `classify_thresholds`:
    - (ΔE = 0.99e-5, forces within) → Unchanged;
    - F moved by −2·ELECTRON_KT·ln 2 with internal equal → EntropyOnly;
    - internal moved by 2e-5 → StateChanged;
    - one force component off by 1.6e-4 → StateChanged;
    - `(MIGRATION_EPS_F - 1.5e-4).abs() <= 1e-18` (it is 3.0 × `PSEUDO_FORCE_NOISE`, so not compared with `==`; F14).
  - `migrated_code_table`: (0,S)→9, (1,S)→9, (2,S)→2, (7,S)→7, (8,S)→7, (8,U)→8, (8,T)→8, (1,U)→1, (1,T)→1, (0,T)→0 (F2).
  - `predecessor_by_exact_key`: found when key.txt matches with the legacy tag; not found when any other field differs (e.g. h).
  - `old_folder_bytes_identical_after_pass`: hash every file of the old folder before and after.
  - `refuses_while_old_folder_has_a_live_writer` (Review Focus 4): the message says it would be reading a store still being appended to.
  - `resume_after_crash_between_append_and_cursor` (Review Focus 4): append sample k without writing the cursor, then resume. The duplicate is refused and counted as done; the final counts and the new samples.bin equal an uninterrupted run's.
  - `failed_resolve_is_listed_not_migrated`: a stalling teacher → listed with its reason; no record appended.
  - `masked_sample_compared_on_all_atoms_and_carried`.
  - `references_resolved_not_carried`: the new references.txt holds only re-solved values; the log holds the old and new values.
  - `not_offered_when_complete`: a complete log → `migration()` reports Complete and no offer.
  - `source_change_restarts_from_zero` (F3): change the old `samples.bin`'s length (or one byte) between a partial pass and its resume → the pass restarts at cursor 0, and the final counts equal an uninterrupted run on the changed source.
  - `purge_refused_while_migration_runs` (F3): with the log at Running, a purge of the new folder is refused with its reason and changes no byte.
  - `old_pass_checkpoint_resume_refused` (spec §7.7): a `training/pass.ckpt` carrying the legacy provenance is refused with the reason "started with the previous solver". The file is left byte-identical.
  - validate surrogate (V7, GPU, small) `migration_classifies_h2_oh_h4`: the old folder is built from Task 10's DS-era fixture labels under the legacy tag. The policy, grid and box are the fixture's (F13). Expected: H₂ → U; linear OH → T (ΔF = −2·ELECTRON_KT·ln 2 within 1e-5, E and forces unchanged within ε); near-square H₄ → S at code 9 (its |ΔE| ≥ 3ε_E by the fixture's choice). The old folder hashes identical. Task 15 adds the review assertions.
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p providers --release migrat classify predecessor source_change purge_refused`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to check they pass.**
- [ ] **Step 5: Physics review.** The reviewer checks:
  - the U/T/S conditions against spec §7.4, ε_E's derivation from the 1.87e-6 pseudo H₂O scatter;
  - the T test as the review of T;
  - that the cold-solve checks exclude teacher pathologies;
  - that no old label is ever carried into class S.
- [ ] **Step 6: Run the gates,** serially: `cargo test -p providers --release` (~5.5 min), `cargo test -p validate --release --test surrogate` (~12–17 min).
- [ ] **Step 7: Commit.**

**Time:** about 20 min of gates.

---

### Task 15: Code 9 in the trust machinery, and the leave-one-out review — **engineering** (physics-reviewed)

**Files:**
- Modify the providers' surrogate files at touch points 1–4 and 9–10 (the Task 2 map): the excluded predicate, `fences()` (F4), the held-cap count and `fencing()` (code 9 stays out of both), the refit pipeline, the lift votes (they never lift code 9), and purge's remap of `migration.txt`'s review log.
- Modify the decision list in `crates/providers/src/surrogate/mod.rs`'s module doc.
- Modify the Forget/Restore paths: they refuse code 9.
- Modify `crates/surrogate/src/quarantine.rs` docs.
- Tests: providers surrogate tests and `crates/validate/tests/surrogate.rs` (V7's review part).

**Interfaces:**
- Consumes: the leave-one-out ratio at touch point 4; `MigrationLog.review`.
- Produces: `fn review_resolved(&mut self) -> Option<ReviewOutcome>`, where `ReviewOutcome { released: Vec<usize>, kept: Vec<(usize, f64)> }`. It runs inside a refit only when all of the following hold:
  - the migration state is Complete or Reviewed;
  - a code-9 sample exists;
  - the store changed since the last review.

  The procedure:
  1. Fit a candidate on the fit's members plus every code-9 sample.
  2. A sample whose ratio ≤ 1 goes 9 → 0.
  3. A sample with ratio > 1 stays at 9. Its verdict is written to `migration.txt`'s review log and read back on attach (ruling R13).
  4. A reviewed-and-failed code-9 sample joins the quarantine fence: `fences()` is true for it (F4; spec §7.6 step 4 and §8). It is not counted by `fencing()` or the held cap, purge's lift votes never lift it, it never joins the misfit set M, and it is never converted to code 1.
  5. A purge of the new folder after the pass remaps the review log's indices through `PurgeOutcome::map` and rewrites `migration.txt` (F3).

- [ ] **Step 1: Write the failing tests:**
  - `code9_out_of_fit_envelope_grade_loo_and_stats`;
  - `code9_not_counted_in_the_held_cap_or_fencing`;
  - `code9_before_review_does_not_fence`;
  - `review_releases_siblings_its_neighbours_predict`: 10 code-9 samples on a smooth analytic surface, all shifted consistently → all released;
  - `review_keeps_an_inconsistent_sample_and_it_is_refused_near`: one sample offset by 50 mHa → stays 9; `fences()` holds for it and a geometry near it is refused; its code is never 1; a later review can still release it;
  - `failed_code9_is_not_lifted_by_purge_votes` (F4): three agreeing answers near a failed code-9 sample leave it at 9 and fencing;
  - `purge_after_review_remaps_the_review_log` (F3): purge forgotten samples that precede reviewed ones → every review-log entry names the same sample at its new index;
  - `no_review_while_the_pass_is_running` (Review Focus 3): a refit with the log at Running leaves every code 9 unchanged;
  - `review_repeats_only_when_the_store_changed`;
  - `forget_and_restore_refuse_code9`;
  - V7: extend `migration_classifies_h2_oh_h4` with the review. Assert that the H₄ sample is **released** (ratio ≤ 1; F14), and record its ratio. A kept outcome is a failure to investigate, never a re-pin.
- [ ] **Step 2: Run them to check they fail.**
- [ ] **Step 3: Implement** at the mapped touch points.
- [ ] **Step 4: Run them to check they pass.**
- [ ] **Step 5: Physics review.** The reviewer checks the candidate-with-siblings rule against spec §7.6, and the no-ratchet invariant of the trust rework (G2 of its spec).
- [ ] **Step 6: Run the gates,** serially: `cargo test -p providers --release` (~5.5 min), `cargo test -p validate --release --test surrogate` (~12–17 min).
- [ ] **Step 7: Commit.**

**Time:** about 20 min of gates.

---

### Task 16: The app: offer, progress, summary, downgrade line, Load-model refusal — **engineering**

**Files:**
- Modify:
  - the Model tab's collection section (`crates/app/src/ui/…` where the collection card lives);
  - `crates/app/src/ui/actions.rs`: `Action::StartMigration` and `Action::MigrationLater`;
  - `crates/app/src/worker/commands.rs`;
  - `crates/app/src/ui/activity/`: the progress line;
  - `crates/app/src/replay.rs`: the action is recorded, but a replay never starts a migration (ruling R16);
  - `crates/app/src/update/formats.rs`: the downgrade line;
  - the Load-model refusal path (the sidecar tag check in `providers::surrogate`) and the Load-model card's "apply the model's settings" offer.
- Modify `README.md` ("Upgrading: collections are re-solved once", with the time estimate and "the old folder is kept").
- Modify `docs/dev-reference.md` (new section "Collection migration"; "Shared collections": record R + 1 and code 9, beside purge's code 8).

**Interfaces:**
- Consumes: `Trainable::migration`, `start_migration`, `MigrationLog`.

- [ ] **Step 1: Write the failing tests.** The copy is verbatim from spec §7.8 and §7.9, with numbers from a fixture predecessor.
  - `migration_offer_text`: the full §7.8 offer, with the species list, the sample count, the folder path and the "About 1 h 30 min" estimate format, plus the buttons [Re-solve now] [Later].
  - `migration_progress_line`: the §7.8 Activity line.
  - `migration_summary_text`: the §7.8 summary, including "identical within solver noise", "kept out" and the old slug.
  - `no_offer_without_samples_to_migrate`: a predecessor with 0 samples, or a complete log → no offer.
  - `downgrade_line_names_the_pre_migration_folder`: present only while the log's source folder exists. The text follows §7.7.
  - `legacy_tag_model_refused_with_reason`: "taught with the solver before <version> (integer / shared occupations); this version uses Fermi–Dirac smearing: re-teach it, or open it in <version>", where `<version>` is this build's `CARGO_PKG_VERSION` (ruling R17).
  - `apply_settings_offer_omits_occupations`.
  - `start_migration_is_an_action`: recorded with `--record`; skipped in replay.
- [ ] **Step 2: Run them to check they fail.** Run `cargo test -p app --release --bin app -- migration downgrade legacy_tag apply_settings`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to check they pass.** Then run the gate `cargo test -p app --release` (~10 min).
- [ ] **Step 5: Commit.** List the user's hand-tests in the body: the offer, Later, a pause and resume, the summary.

**Time:** about 10 min of gates.

---

### Task 17: Re-make the shipped models; V11 dry run on a copy — **engineering** (measurement)

**Files:**
- Create `crates/validate/tests/migration.rs`, `#[ignore]`d:
  - `remake_shipped_models`;
  - `dry_run_on_collection_copy`, taking the copy's path from env `FD_DRY_RUN_COPY`.
- Modify in the worktree: `models/hydrogen.model`, `models/h2o.model`, `models/h3+.model`, `models/SiH4.model`, `models/README.md`.
- Modify `README.md` (the model list and its status) and `crates/validate/references.md` ("Fermi–Dirac migration": the re-make and V11).
- Modify `CLAUDE.md`: one row under "shared-model measurements (not gates)" for `--test migration`.

**Interfaces:**
- Consumes: Task 14's engine (`classify`, the cold solve, the writes) and Task 15's review. The harness reads each old `.model`'s samples through the sidecar reader, which takes any tag; the provider-level tag refusal does not apply in the harness.

- [ ] **Step 1: Re-make the shipped models.** For each of the four tracked models:
  1. re-solve, classify U/T/S, review in a refit, and write the new magic with the new tag;
  2. assert the re-made model passes the trust rule it passed before;
  3. if it does not, re-teach it from scratch with the sampling pass, and say which in the README model list.

  Run `cargo test -p validate --release --test migration remake_shipped_models -- --ignored --nocapture`. Record the counts per class and the trust verdicts in references.md.
- [ ] **Step 2: V11 dry run.**
  - Copy the user's collection folder from `D:\Projects\Physics\models\collections\` into the scratchpad. Never run on the original.
  - Hash every file of the original, and again after the run.
  - Run `FD_DRY_RUN_COPY=<copy> cargo test -p validate --release --test migration dry_run -- --ignored --nocapture`.
  - Record in references.md:
    - the counts per class;
    - class U's largest |ΔE| and |ΔF|;
    - the time per sample;
    - the review outcome (released / kept out);
    - the refit's trust verdict against the pre-migration one;
    - the false-S estimate: the fraction of S whose only failing test is |ΔE| ∈ (ε_E, 3ε_E].
  - **Must:**
    - the original and the source copy are byte-identical afterwards;
    - no sample changes code without a logged reason.
  - **Reported, not gated:** if the migrated collection is not trusted as before, investigate the reason before Task 18.
- [ ] **Step 3: Commit** the models, `models/README.md`, `README.md`, `references.md`, `migration.rs` and `CLAUDE.md`.

**Time:** about 30–60 min for the re-make, plus about 1–2 h of GPU for the dry run.

---

### Task 18: The docs sweep, the full serial gate run, DEVLOG, the merge — **engineering**

**Files:**
- Modify `DEVLOG.md`: timeline, durations, failures, wins; the diagnosis note linked forward to the spec.
- Modify `docs/superpowers/notes/2026-11-13-scf-stall-diagnosis.md`: a forward link to the spec.
- Fix any item Step 1 finds missing.

- [ ] **Step 1: Check spec §10 item by item.** Each item points at the commit that did it:
  - the `qm::scf`/`qm_gpu::scf` docs;
  - the pseudopotential spec §5.7 and §5.5;
  - the main spec §6 and §11;
  - the orbital-viewer spec item 4;
  - every references.md entry;
  - every dev-reference section;
  - the README items;
  - the code docs: collection, quarantine, pseudo, ks, both forces modules, lone, formats.

  Fix any gap.
- [ ] **Step 2: Run the full serial gate,** one job at a time. Total about 4.5–5 h:
  1. `cargo test -p units -p qm -p sim -p surrogate --release` (~6.5 min)
  2. `cargo test -p providers --release` (~5.5 min plus V2)
  3. `cargo test -p qm-gpu --release`
  4. `cargo test -p validate --release --test oracles` (~31 min)
  5. `--test gpu_vs_cpu` (~3.5 min)
  6. `--test dynamics` (~6 min plus V5's tests)
  7. `--test surrogate` (~12–17 min)
  8. `--test tools` (~3–10 min)
  9. `--test charges` (~11 min)
  10. `--test orbitals` (~4 min)
  11. `--test pseudo_gen` (~9 min)
  12. `--test pseudo` (~1 h)
  13. `--test pseudo_oracles` (~1.5–2 h)
  14. `cargo test -p app --release` (~10 min)
  15. `cargo test -p app --release --bin app -- builder::` (seconds)

  Expected: all PASS. A failure is debugged, never loosened. Record the times in dev-reference "Test suites".
- [ ] **Step 3: Whole-branch review** (superpowers:requesting-code-review).
- [ ] **Step 4: Commit the docs, then merge.** Under the standing hands-off approval, merge in the main checkout `D:\Projects\Physics` (F6), never by `git checkout main` in this worktree: confirm `git -C D:\Projects\Physics branch --show-current` prints `main` and its tree is clean of tracked changes, then run `git -C D:\Projects\Physics merge scf-stall` on its own, then push. Never merge `smearing-prototype`. The README and DEVLOG are already in the branch.

**Time:** about 5 h.

---

## Rulings made in this plan (spec gaps)

- **R1.** `scf-stall` **merges** `main` (the user's instruction) instead of the spec §11 rebase.
- **R2.** A1 is split. V0 is Task 1, before the merge, consuming the other agent's note. The rest of A1 is Task 3, also before the merge (preflight order ruling: Task 3 runs before Task 2). V3's aspect is chosen on the GPU prototype and confirmed on the CPU in Task 6, because no CPU FD prototype exists.
- **R3.** The CPU oracles (V1's H₄ square and FD ≡ Aufbau, V3, V5's work integral) run on the **all-electron surface, σ = h**. Spec §9 names the grid but not the surface. `qm` unit tests have no table cache, and FD-consistency does not depend on the surface.
- **R4.** `fermi_dirac` returns `Result<FdOccupations, QmError>` and carries `halvings`. With n > 0, a level count ≤ n or a kT that is not finite and > 0 is also `QmError::Numerical`.
- **R5.** `PartialLevel.index` is the level's ascending rank within its channel.
- **R6.** `DegenerateShells` is deleted in Task 10, not A2, so the workspace builds at every commit. `for_grid` flips to FD in Task 7, with a one-line `Aufbau` hold on the providers' all-electron path until Task 9. No surface moves under its old tag.
- **R7.** The harness is reduced into V2 in Task 9, since V2 can only pass once the providers run FD. Spec §11 put both at A2. (Superseded in part by R21: the env knobs and the harness leave the tree at the end of Task 3, preserved on `smearing-prototype`.)
- **R8.** `branch_jump` is built before the surface switch (Task 8), so the occupation watch is replaced, never just removed.
  - The tier is the sample tier iff `scf_tol_used ≤ SAMPLE_TOL`, else the app tier; the surface comes from `is_all_electron_tag`.
  - The history also restarts on a change of atom count, species order or origin.
- **R9.** V4's "same state class": Symmetric if |TS − TS_cf| ≤ 0.05·TS_cf, Polarised if TS ≤ 0.05·TS_cf. Anything else fails.
- **R10.** V5's sign. W is the work the forces do, ∫F_HF·dx, so ΔF + W = 0. Spec §9 writes "W = −∫F_HF·dx" together with "|W + ΔF|", which is inconsistent; §6.3's `branch_jump` convention, |ΔF − (−∫F·dR)|, is the same physics.
- **R11.** The sidecar gets one new magic. Its header names the old body it carries (6, 7 or 8 before the merge, plus any body `purge-forgotten` added), which keeps "one more entry in `MAGICS`" for all of them. Its number is K + 1, set after the merge (F1).
- **R12.** `is_persisted_code` stays the sidecar's predicate (it keeps purge's 8). The new `is_collection_code` admits 9.
- **R13.** A failed code-9 review is persisted in `migration.txt`'s review log; the code stays 9. `Collection::open_read_only` returns a separate read-only type with no write methods.
- **R14.** V7's old-folder fixture uses DS-era labels captured in Task 10, before DS is deleted.
- **R15.** `legacy_tag_of` hashes with a frozen `DIMER_PROCEDURE_DS` and is pinned to strings recorded in Task 9 before any change.
- **R16.** Starting a migration is a recorded `ui::Action`, but a replay never starts one, as with the PySCF setup.
- **R17.** "<version>" in the legacy-model refusal is this build's `CARGO_PKG_VERSION`.
- **R18.** V5 dynamics (Task 11) comes before the UI (Task 12), swapping spec §11's A5/A6. This closes the whole-solver-half review before the UI copy is written.
- **R19 (2026-11-14, the user's ruling after V0).** kT = 5 mHa, and the reference scan's stop at 1e-5. Spec §2 item 3 and rulings 42–46 apply it: every GPU solve with a partial level stops at max(tol_eff, 1e-5) (Task 7 implements `GPU_PARTIAL_TOL`; Task 9's tags carry `fd5e-3` and `pin1e-5`); the witness stays 36 kT / 16; V2's pins are re-measured at 5 mHa (Task 3, step 0); the FD ≡ Aufbau test runs at an explicit kT = 1e-3 (Task 5). Every "316 K" in Tasks 12 and 16 is 1579 K (`units::hartree_to_kelvin(ELECTRON_KT)`, rounded).
- **R20.** The V0 re-run at 5 mHa (Task 1) covers B, C, N, O, F, Si, P, S, Cl, all-electron O and the spec's molecules (OH along an axis and off-axis, the O₂ singlet at auto and triplet, H₂O, NH at auto and at M = 3). Its numbers are in the V0 note's 2026-11-14 section; Task 1 step 6 copies the summary into references.md.
- **R21 (2026-11-14, the controller's preflight rulings F1–F15, ledger "Preflight scan").** `Q_RESOLVED` = 9 (code 8 is `purge-forgotten`'s); the formats are symbolic (R + 1, K + 1); purge's code 8 maps to 7 in class S (F2); purge and the migration exclude each other and the review log follows `PurgeOutcome::map` (F3); a failed code 9 fences but is outside `fencing()`, the held cap and the lift votes, never in M (F4); the prototype lives only on `smearing-prototype` (F5); no bare stash, the final merge in `D:\Projects\Physics` (F6); and the per-task fixes F7–F15. Order: Task 1 → Task 3 → Task 2 (after `purge-forgotten` merges) → Tasks 4–18.
