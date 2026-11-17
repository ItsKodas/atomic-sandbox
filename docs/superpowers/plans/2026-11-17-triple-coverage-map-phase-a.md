# Three-Atom Coverage Map, Phase A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Knowledge panel gains a "Three-atom arrangements" section: a per-triple-type coverage map of the fitted three-atom arrangements over (r₁, r₂, cos θ), drawn as a 2D slice by default and as a rotatable 3D voxel cloud on a switch, with a picker ordered by gaps, cell hover and readouts, and live "new, waiting for refit" outlines during training. There is no Teach in this phase.

**Architecture:**
- `sim::knowledge::triples` (new, plain data and geometry, no physics) holds the grid, the one enumeration `each_triple`, a counting `TripleTally`, the shipped `TripleMap`, and the pure rules `reach`, `cell_class`, `summary`.
- `providers::surrogate::triples` (new) builds a `TripleMap` from the fitted and pending samples. The fitted block is cached against the fitted set's identity, so a pending-only rebuild recounts only the tail.
- The worker builds the map only while the UI says the section is shown (`Command::SetTripleView`), on the knowledge map's schedule, and ships it as `ActivityDelta::triples`.
- The app draws it with `egui::Painter` only: `ui/knowledge/triples.rs` (picker, card, 2D slice, cell card) and `ui/knowledge/cloud.rs` (3D projection and hit test).

**Tech Stack:** Rust workspace (`sim`, `providers`, `app`), egui. No new dependency, no GPU pipeline.

**Spec:** `docs/superpowers/specs/2026-11-17-triple-coverage-map-design.md` ("the spec", sections cited as §n), with its "Phasing (user, 2026-11-17)" paragraph in §8. Project rules: `CLAUDE.md`. Base: branch `triple-map` at 8c6f52a (`main` merged in, live-knowledge included). Targeted teach is **not** merged and nothing here depends on it.

## Global Constraints

- **Units and rules.** Bohr inside `sim` and `providers` (rule 3). Degrees appear only at the display boundary; the angle is binned in cos θ. Nothing per element beyond Z (rule 1): every range comes from the grid constants, r_cut, or the map's own fitted pair histograms. `sim` gets plain data; `providers` alone reads the store, `is_pending`, the quarantine codes and the misfit set; the app draws (rule 5). No tolerance, trust rule, fence or stored sample changes.
- **Thresholds (decided for Phase A).** `TAUGHT` = 0.9 moves now to `sim::knowledge`, beside `COVERAGE_COUNT` = 4.0, with targeted teach's exact names and signatures: `pub const TAUGHT: f64`, `pub fn taught_count() -> u32` (N_T = ⌈−COVERAGE_COUNT·ln(1 − TAUGHT)⌉ = **10**, derived, never a literal), `pub fn reads_taught(window: u32) -> bool` (1 − e^(−n/COVERAGE_COUNT) ≥ TAUGHT). The app re-exports `TAUGHT` and its `Taught::of` reads `reads_taught`. `window_count` stays targeted teach's to add. When targeted teach rebases, its Task 1 finds these three items already on `main`, identical, and keeps them.
- **What is counted (§6.1), verbatim.**
  - A triple type t = (a, c, b): centre species c, neighbour species a ≤ b. Stored as `TripleType { centre, a, b }`.
  - A triple (i, {j, k}) counts when centre i trains (`trains_atom`), j ≠ k are both within r_cut of i (0 < r_ij, r_ik < r_cut), and their species are {a, b}. The j–k distance may be anything, including beyond r_cut. Each unordered {j, k} counts once. j = k is never counted.
  - The neighbour set is **every neighbour within r_cut**, not the element card's first shell, which stays as it is.
  - Axes: for a ≠ b, r₁ = r_ij to the species-a neighbour and r₂ = r_ik to the species-b one. For a = b the type is **folded**: r₁ = min, r₂ = max, and only cells with k₁ ≤ k₂ are stored. u = cos θ_jik.
  - Fitted triples come from exactly the knowledge map's sample rule (the accumulated prefix, not excluded, not in the misfit set). Pending triples come from the samples `is_pending` admits. They are counted apart, enter no class, and never move a cell's colour.
  - Per type: the fitted samples holding at least one triple of it, the fitted triple total, the pending samples, and the top three stored compositions holding it by fitted count.
- **Cells (§6.2), verbatim.** `TRIPLE_BIN` = 3·`BIN` = **0.3 Bohr** from 0, n_r = ⌈r_cut/0.3⌉ (20 at r_cut 6, 27 at 8, 34 at 10); only r < r_cut is counted. `COS_BINS` = **10** bins of Δu = 0.2; q = min(⌊(1 − u)/0.2⌋, 9), so θ increases left to right; edges at 0, 37, 53, 66, 78, 90, 102, 114, 127, 143 and 180°. Dense storage per type present: n_r² × 10 u32, fitted and pending.
- **Classes and reach (§6.3), verbatim.**
  - Taught: the cell's fitted count n satisfies `reads_taught(n)`, so n ≥ N_T = `taught_count()` = 10. Thin: 1 ≤ n < N_T. The window is the cell itself; no new constant.
  - Missing within reach: n = 0 and the cell, judged at its centre (r₁, r₂, u), is in the type's reach:
    1. r₁ lies in the c–a pair's taught range (`taught_range`: from the first filled bin to the last) and r₁ ≤ 5/6·r_cut;
    2. r₂ lies in the c–b pair's taught range and r₂ ≤ 5/6·r_cut;
    3. r_ab = √(r₁² + r₂² − 2r₁r₂u) is at least the a–b pair's shortest fitted distance (the lower edge of its first filled bin); with no fitted a–b distance within r_cut, r_ab ≥ r_cut.
  - Cells outside reach with n = 0 are not drawn; with n > 0 they are drawn in their class.
  - New (dashed outline): pending > 0. It never changes the class.
  - Gaps of a type: the cells in reach that are not taught (missing or thin).
- **Cost budget (§6.4).** Cold build ≤ 100 ms at the user's BEST collection; pending-only rebuild ≤ 2 ms. The plan measures both and records them in `references.md`, whatever they are, and tunes nothing away.
- **Copy, verbatim (§6.5, §6.7). Tests pin these.** `{…}` is formatted; "10" comes from `taught_count()`, "0.3" from `TRIPLE_BIN`, "0.2" from the cos bin width.
  - section title: `Three-atom arrangements`
  - picker row: `{name} · {gaps} gaps · {triples} arrangements in {samples} samples · +{k} new` (the last part only when k > 0)
  - card sentence: `{triples} arrangements of {name} in {samples} samples ({taught} cells taught, {thin} thin, {missing} missing within reach).`
  - verdict line: `the model's many-body key {key words}: trusted`, or `… : {n} of {q} samples — the per-environment sample floor, not this map`
  - 2D caption: `angle at {c} (cos θ bins)`; slice label `r₂ ({c}–{b}) {lo:.1}–{hi:.1} Bohr`; marker `now`
  - hover: `r₁ {lo:.1}–{hi:.1} · r₂ {lo:.1}–{hi:.1} Bohr · θ {θa}–{θb}° · n {n} (+{k} new) · {class}` (the parenthesis only when k > 0; θ edges `{:.0}`; class `taught`, `thin` or `missing within reach`)
  - cell card: `{name} · r₁ {lo:.1}–{hi:.1}, r₂ {lo:.1}–{hi:.1} Bohr, θ {θa}–{θb}° · {n} arrangements · {class} · +{k} new`
  - band, shown while a cell is picked (§6.7 items 1–3 only):
    1. `Green counts arrangements in the cell (10 within 0.3 Bohr in each distance and 0.2 in cos θ); it is not the model's trust or accuracy.`
    2. `Counts every pair of neighbours within {r_cut:.1} Bohr of the centre, as the model's three-atom terms do.` + ` Four-atom arrangements (body-4) are learned too and not shown.` only when the basis has body-4. With `body3` false the line is instead `this model's basis has no three-atom terms: it does not learn the angle`.
    3. `The model's three-atom terms are smooth in the distances and in cos θ: it may interpolate across a missing cell, and a green cell can still be wrong.`
- **Phase A has no Teach.** No Teach button, no drag selection, no estimate, no band items 4–7, no `teach-cells` word, no `region=` key. Nothing in Phase A emits a `ui::Action`.
- **Recording and display state (§4, §6.8).** The pick, the 2D/3D mode, the slice, the rotation and zoom, the "missing" checkbox and the picked cell are `KnowledgeView` display state, not recorded. `Command::SetTripleView(bool)` is a display-visibility command derived from what is shown, as `SetActivityViews` is (`ui/mod.rs:1446-1450`).
- **egui.** New widgets go through `ui::keyed` (`ui/mod.rs:1601`) where ids could shift. Probes are named `knowledge-triple-*`. Drawing uses `egui::Painter` shapes only.
- **Process.**
  - Every task is tagged **Opus**, and every subagent runs on Opus (memory "No Fable"). Tasks whose geometry or angle computation is a matter of physics correctness say so; the `physics-reviewer` (Opus) reads §6.1–§6.3 and the diff of Tasks 1–3 before Task 4 lands (spec §7 "Review").
  - Heavy jobs run one at a time. With another cargo job running, build at BelowNormal with `-j 4`. Never kill the user's cargo or `app.exe`.
  - Commit with `git commit -o -m "<msg>" -- <paths>` (`git add` a new file first). Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never pipe a gating git command. Edit with Edit/Write or Python, never PowerShell `Set-Content`. No bare `git stash`.
  - Never write to `D:\Projects\Physics\models`; the measurement reads a copy.
  - UI tasks list the user's hand-tests in the commit body (memory "Testing split").

## Review Focus

1. **The map changes under the view.** A Forget removes the picked type, a refit drops a type, or the model's reach changes r_cut (6 ↔ 10), so the stored slice and cell indices exceed or misname the new grid. Expected: a plain note for a vanished type and a fallback to the default type; the slice clamped into the new extent; a picked cell dropped when it is out of range or no longer drawn; never a panic or an out-of-bounds index. Test: Task 7 `stale_view_state_is_clamped_after_the_map_changes`.
2. **Floating-point bin edges.** u exactly 0 or ±1, |u| a hair over 1 from a rounded dot product, r_cut an exact multiple of 0.3, r a hair under r_cut, NaN or infinite inputs. Expected: the bins of the spec's formula, the last bin for r just under r_cut, `None` for r ≤ 0, r past the grid or a non-finite value; never an index past the grid. Test: Task 1 `the_grid_spans_r_cut_in_0_3_bohr_bins`, `cos_bins_are_uniform_in_cos_theta`.
3. **Coincident or overlapping atoms** (r = 0) in a stored sample or in the open scene. Expected: a zero-distance neighbour is skipped, no NaN reaches a bin, and nothing panics; the UI's markers go through the same `each_triple`. Test: Task 2 `coincident_atoms_make_no_triple_and_no_nan`.
4. **Nothing to show.** An H₂-only collection holds no triple type, a basis has `body3` off, or the section is open before any map arrives. Expected: the section says so in a plain sentence, the picker is empty without panicking, and `SetTripleView` still follows visibility. Test: Task 6 `the_section_explains_an_empty_or_three_body_free_map`.
5. **A worker respawn while the section is open** (a backend switch, a panic recovery). Expected: the viewer resets `triples_sent` and drops the old map, the next frame sends `SetTripleView(true)` to the new worker, and its first publish builds the map. Test: Task 6 `the_triple_view_command_follows_the_fold_and_a_respawn`.

## File map

| File | Change |
|---|---|
| `crates/sim/src/knowledge.rs` | `TAUGHT`, `taught_count`, `reads_taught`, `taught_range` (moved from the app); `pub mod triples; pub use triples::*;`; module doc "Three-atom arrangements" |
| `crates/sim/src/knowledge/triples.rs` | **new**: grid, `each_triple`, `TripleTally`, `TripleMap`, `reach`, `cell_class`, `summary` |
| `crates/sim/src/provider.rs` | `Trainable::triple_map` (default `None`) |
| `crates/providers/src/surrogate/triples.rs` | **new**: `SurrogateKs::triple_map`, the fitted-block cache |
| `crates/providers/src/surrogate/knowledge.rs` | `each_pair` becomes `pub(super)` |
| `crates/providers/src/surrogate/mod.rs` | `mod triples;`, the cache field, the `Trainable` forwarder |
| `crates/providers/src/surrogate/tests/triples.rs`, `tests/triple_cost.rs` | **new**: B1, B2; the `#[ignore]`d cost measurement |
| `crates/app/src/worker/{commands,activity}.rs`, `worker/tests.rs` | `Command::SetTripleView`, `ActivityDelta::triples`, the schedule |
| `crates/app/src/ui/activity/mod.rs`, `crates/app/src/viewer.rs` | `ActivityState::{triples, triples_sent}`, absorb, respawn reset |
| `crates/app/src/ui/knowledge.rs` | `Pick::Triple`, `Sections::triples`, `KnowledgeView::triple`, entry links, `fixture_triples()` |
| `crates/app/src/ui/knowledge/triples.rs` | **new**: the section, picker, card, 2D slice, cell card, band |
| `crates/app/src/ui/knowledge/cloud.rs` | **new**: 3D projection, faces, depth order, hit test |
| `crates/app/src/ui/mod.rs` | `SetTripleView` emission |
| `crates/app/src/screenshot.rs` | `--ui` keys `triple`, `triples`, `slice`, `cell`, `rotate`; the wait |
| `README.md`, `docs/dev-reference.md`, `crates/validate/references.md`, `DEVLOG.md`, `docs/images/knowledge_triples_{2d,3d}.png` | docs |

---

### Task 1: Thresholds and the triple grid (`sim::knowledge`) — **Opus**

Physics correctness: the cos θ binning and its degree edges. Equal widths in u are the spec's argument (§6.2); a bin formula that drifts at an edge silently moves arrangements between cells.

**Files:**
- Modify: `crates/sim/src/knowledge.rs` (constants and helpers after `COVERAGE_COUNT`, line 110; `pub mod triples; pub use triples::*;`)
- Create: `crates/sim/src/knowledge/triples.rs` (grid half only in this task)
- Modify: `crates/app/src/ui/knowledge.rs:62-63` (`pub use sim::knowledge::TAUGHT;`), `:129-137` (`Taught::of` reads `reads_taught`), `:1011-1016` (`taught_range(p)` calls `sim::knowledge::taught_range(&p.hist)`)

**Interfaces:**
- Produces (`sim::knowledge`):
  ```rust
  pub const TAUGHT: f64 = 0.9;
  pub fn taught_count() -> u32;
  pub fn reads_taught(window: u32) -> bool;
  pub fn taught_range(h: &PairHistogram) -> Option<(f64, f64)>; // (first filled bin's lower edge, last filled bin's upper edge)
  pub const TRIPLE_BIN: f64 = 3.0 * BIN;
  pub const COS_BINS: usize = 10;
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub struct TripleType { pub centre: u8, pub a: u8, pub b: u8 } // a ≤ b
  impl TripleType { pub fn new(centre: u8, x: u8, y: u8) -> TripleType; pub fn folded(&self) -> bool; }
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct TripleGrid { pub r_bin: f64, pub n_r: usize, pub cos_bins: usize }
  impl TripleGrid {
      pub fn new(r_cut: f64) -> TripleGrid;
      pub fn cells(&self) -> usize;                                        // n_r² · cos_bins
      pub fn r_bin_of(&self, r: f64) -> Option<usize>;                     // None: non-finite, r ≤ 0, or past the grid
      pub fn index(&self, folded: bool, k1: usize, k2: usize, q: usize) -> usize; // (k1·n_r + k2)·cos_bins + q, k1 ↔ k2 swapped first when folded and k1 > k2
      pub fn cell(&self, folded: bool, r1: f64, r2: f64, u: f64) -> Option<usize>; // None for a bad r or a non-finite u
      pub fn coords(&self, cell: usize) -> (usize, usize, usize);         // (k1, k2, q)
      pub fn centre(&self, cell: usize) -> (f64, f64, f64);               // (r1, r2, u) at the cell centre
      pub fn r_edges(&self, k: usize) -> (f64, f64);
  }
  pub fn cos_bin(u: f64) -> usize;           // clamps u to [−1, 1] first
  pub fn cos_edges_deg(q: usize) -> (f64, f64); // (θ at u = 1 − 0.2q, θ at u = 1 − 0.2(q + 1)), degrees
  ```

- [ ] **Step 1: Write the failing tests** (in `knowledge.rs`'s `tests` and a new `tests` module of `triples.rs`)

```rust
#[test]
fn taught_count_is_derived() {
    assert_eq!(taught_count(), 10);
    assert!(!reads_taught(taught_count() - 1) && reads_taught(taught_count()) && !reads_taught(0));
    let (lo, hi) = taught_range(&PairHistogram { z: (1, 8), bin: BIN, counts: vec![0, 0, 3, 0, 5, 0] }).unwrap();
    assert!((lo - 0.2).abs() < 1e-12 && (hi - 0.5).abs() < 1e-12);
    assert_eq!(taught_range(&PairHistogram { z: (1, 1), bin: BIN, counts: vec![0; 6] }), None);
}

#[test]
fn the_grid_spans_r_cut_in_0_3_bohr_bins() {
    assert!((TRIPLE_BIN - 0.3).abs() < 1e-12);
    for (rc, n) in [(6.0, 20), (8.0, 27), (10.0, 34)] {
        let g = TripleGrid::new(rc);
        assert_eq!((g.n_r, g.cos_bins, g.cells()), (n, COS_BINS, n * n * COS_BINS), "r_cut {rc}");
    }
    let g = TripleGrid::new(6.0);
    assert_eq!(g.r_bin_of(1.81), Some(6));
    assert_eq!(g.r_bin_of(5.999_999), Some(19), "a hair under r_cut: the last bin");
    for bad in [0.0, -1.0, 6.3, f64::NAN, f64::INFINITY] {
        assert_eq!(g.r_bin_of(bad), None, "{bad}");
    }
    let (lo, hi) = g.r_edges(6);
    assert!((lo - 1.8).abs() < 1e-12 && (hi - 2.1).abs() < 1e-12);
}

/// Spec M2.
#[test]
fn cos_bins_are_uniform_in_cos_theta() {
    assert_eq!(cos_bin(1.0), 0, "0°");
    assert_eq!(cos_bin(0.0), 5, "90° on the edge: the larger-θ bin");
    assert_eq!(cos_bin(-1.0), 9, "180°");
    assert_eq!((cos_bin(1.000_000_1), cos_bin(-1.000_000_1)), (0, 9), "clamped");
    assert_eq!(cos_bin(104.5f64.to_radians().cos()), 6);
    let mut edges: Vec<i64> = (0..COS_BINS).map(|q| cos_edges_deg(q).0.round() as i64).collect();
    edges.push(cos_edges_deg(COS_BINS - 1).1.round() as i64);
    assert_eq!(edges, vec![0, 37, 53, 66, 78, 90, 102, 114, 127, 143, 180]);
}

/// Spec M3.
#[test]
fn a_symmetric_triple_folds() {
    let g = TripleGrid::new(6.0);
    let a = g.cell(true, 1.85, 2.95, -0.25).unwrap();
    assert_eq!(Some(a), g.cell(true, 2.95, 1.85, -0.25), "(r₁, r₂) and (r₂, r₁): one cell");
    assert_eq!(g.coords(a), (6, 9, 6));
    assert_ne!(g.cell(false, 1.85, 2.95, -0.25), g.cell(false, 2.95, 1.85, -0.25), "unfolded keeps the order");
    assert_eq!((g.index(true, 9, 6, 6), g.index(true, 6, 9, 6)), (a, a), "a folded slice at s reads {min(k₁, s), max(k₁, s)}");
    let (r1, r2, u) = g.centre(a);
    assert!((r1 - 1.95).abs() < 1e-12 && (r2 - 2.85).abs() < 1e-12 && (u + 0.3).abs() < 1e-12);
    assert_eq!(g.cell(false, 1.8, 1.9, f64::NAN), None);
    assert_eq!(TripleType::new(1, 8, 1), TripleType { centre: 1, a: 1, b: 8 });
    assert!(TripleType::new(8, 1, 1).folded() && !TripleType::new(1, 8, 1).folded());
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sim --release --lib knowledge::`
Expected: FAIL to compile, "cannot find function `taught_count`" (and the triple items).

- [ ] **Step 3: Implement the items above.** `cos_bin(u) = ⌊(1 − clamp(u, −1, 1))·COS_BINS/2⌋` capped at `COS_BINS − 1` (the spec's ⌊(1 − u)/0.2⌋, written as a multiplication so u = 0 lands in bin 5 exactly). `n_r = ⌈r_cut/TRIPLE_BIN⌉`. In the app, delete the local `TAUGHT` const, re-export the sim one, and make `Taught::of(n)` read `n == 0 → Never; reads_taught(n) → Taught; else Thin`.

- [ ] **Step 4: Run them to see them pass, and the app's knowledge tests unchanged**

Run: `cargo test -p sim --release --lib knowledge::` — Expected: PASS.
Run: `cargo test -p app --release --bin app -- knowledge` — Expected: PASS, no test changed.

- [ ] **Step 5: Commit**

```bash
git add crates/sim/src/knowledge/triples.rs
git commit -o -m "feat(knowledge): TAUGHT and taught_count in sim; the triple grid, binned in cos θ" -- crates/sim/src/knowledge.rs crates/sim/src/knowledge/triples.rs crates/app/src/ui/knowledge.rs
```

---

### Task 2: One enumeration and the tally (`each_triple`, `TripleTally`) — **Opus**

Physics correctness: this is what the model's body-3 terms sum (§2, `descriptor.rs:56-75`): every neighbour pair of a training centre within r_cut, the angle at the centre from the two bond vectors, j–k unrestricted, the fold for a = b. A wrong neighbour rule or angle still produces a plausible map.

**Files:**
- Modify: `crates/sim/src/knowledge/triples.rs`

**Interfaces:**
- Consumes: Task 1's `TripleType`, `TripleGrid`, `cos_bin`; `sim::knowledge::Composition`.
- Produces:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct Triple { pub t: TripleType, pub centre: usize, pub j: usize, pub k: usize, pub r1: f64, pub r2: f64, pub u: f64 }
  // j: the species-a neighbour (folded: the nearer; a tie goes to the lower index), r1 = |R_j − R_i|; k likewise for b, r2.
  pub fn each_triple(z: &[u8], pos: &[[f64; 3]], r_cut: f64, trains: impl Fn(usize) -> bool, f: impl FnMut(Triple));
  pub struct TripleTally { /* grid, r_cut, per type: counts, samples, triples, compositions */ }
  impl TripleTally {
      pub fn new(r_cut: f64) -> TripleTally;
      pub fn add(&mut self, z: &[u8], pos: &[[f64; 3]], trains: impl Fn(usize) -> bool); // one sample
      pub fn finish(self) -> Vec<TripleBlock>; // ascending t
  }
  #[derive(Clone, Debug, PartialEq)]
  pub struct TripleBlock { pub t: TripleType, pub counts: std::sync::Arc<[u32]>, pub samples: usize, pub triples: u64, pub compositions: Vec<(Composition, usize)> } // top three, most first, then by composition
  ```

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec M1: three bent H₂O (104.5°) and one linear; in the fourth, O does not train.
#[test]
fn each_triple_counts_every_neighbour_pair_of_a_training_centre() {
    let bent = |r: f64| {
        let t = 104.5f64.to_radians() / 2.0;
        vec![[0.0; 3], [r * t.sin(), r * t.cos(), 0.0], [-r * t.sin(), r * t.cos(), 0.0]]
    };
    let lin = |r: f64| vec![[0.0; 3], [r, 0.0, 0.0], [-r, 0.0, 0.0]];
    let z = [8u8, 1, 1];
    let mut tally = TripleTally::new(6.0);
    for g in [bent(1.81), bent(1.85), lin(1.81)] {
        tally.add(&z, &g, |_| true);
    }
    tally.add(&z, &bent(1.83), |a| a != 0);
    let blocks = tally.finish();
    let g = TripleGrid::new(6.0);
    let (hoh, ohh) = (TripleType::new(8, 1, 1), TripleType::new(1, 8, 1));
    assert_eq!(blocks.iter().map(|b| b.t).collect::<Vec<_>>(), vec![ohh, hoh]);
    let w = &blocks[1];
    assert_eq!((w.samples, w.triples), (3, 3), "a non-training centre adds nothing");
    assert_eq!((w.counts[g.index(true, 6, 6, 6)], w.counts[g.index(true, 6, 6, 9)]), (2, 1), "bent at cos −0.25 (bin 6), linear (bin 9)");
    assert_eq!(w.counts.iter().sum::<u32>(), 3);
    let h = &blocks[0];
    assert_eq!((h.samples, h.triples), (4, 8), "two H centres per water");
    assert_eq!(h.counts[g.index(false, 9, 6, 1)], 6, "r₁ to H 2.86–2.93 (bin 9), r₂ to O (bin 6), 37.75° (bin 1)");
    assert_eq!(h.counts[g.index(false, 12, 6, 0)], 2, "linear: r₁ 3.62 (bin 12), 0°");
    assert_eq!(w.compositions, vec![(Composition::of(&z), 3)]);
    // j–k beyond r_cut still counts; j = k never; a dimer has none
    let mut t3 = TripleTally::new(3.0);
    t3.add(&z, &lin(1.81), |_| true);
    t3.add(&[1, 1], &[[0.0; 3], [1.4, 0.0, 0.0]], |_| true);
    let b3 = t3.finish();
    assert_eq!(b3.iter().map(|b| (b.t, b.triples)).collect::<Vec<_>>(), vec![(hoh, 1)], "H–H 3.62 > r_cut: no H-centred triple, H–O–H still counted");
}

/// Review Focus 3.
#[test]
fn coincident_atoms_make_no_triple_and_no_nan() {
    let (z, pos) = ([8u8, 1, 1], [[0.0; 3], [0.0; 3], [1.8, 0.0, 0.0]]);
    let mut seen = Vec::new();
    each_triple(&z, &pos, 6.0, |_| true, |t| seen.push(t));
    assert_eq!(seen.len(), 1, "only at the far H, whose two neighbours coincide: {seen:?}");
    assert!(seen.iter().all(|t| t.r1 > 0.0 && t.r2 > 0.0 && (-1.0..=1.0).contains(&t.u)));
    let mut tally = TripleTally::new(6.0);
    tally.add(&z, &pos, |_| true);
    assert_eq!(tally.finish().iter().map(|b| b.triples).sum::<u64>(), 1);
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sim --release --lib knowledge::triples` — Expected: FAIL to compile (`each_triple`, `TripleTally` not found).

- [ ] **Step 3: Implement.** `each_triple`: for each training centre i, collect the neighbours with 0 < r < r_cut (unit vector, r, index), then every unordered pair; u = clamp(û_j·û_k, −1, 1). Atoms past min(z.len(), pos.len()) are ignored. `TripleTally::add` bins each triple with `grid.cell(t.folded(), r1, r2, u)` and counts the sample once per type it holds, under `Composition::of(z)`.

- [ ] **Step 4: Run them to see them pass**

Run: `cargo test -p sim --release --lib knowledge::triples` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(knowledge): each_triple and the triple tally (every neighbour pair within r_cut)" -- crates/sim/src/knowledge/triples.rs
```

---

### Task 3: The map, the reach and the classes — **Opus**

Physics correctness: the reach's three conditions and the law-of-cosines third side (§6.3), and the rule that pending counts move no class. Then the physics review.

**Files:**
- Modify: `crates/sim/src/knowledge/triples.rs`
- Modify: `crates/sim/src/knowledge.rs` (module doc: a "Three-atom arrangements" section — what is counted, the cos θ argument, the reach, the fold, fitted versus pending, and that it says nothing about accuracy)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct PairSpan { pub z: (u8, u8), pub lo: f64, pub hi: f64 } // a fitted pair's taught_range
  impl PairSpan { pub fn of(h: &PairHistogram) -> Option<PairSpan>; }
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub struct TripleBasis { pub body3: bool, pub body4: bool, pub degree_cap: usize, pub l_max: usize }
  #[derive(Clone, Debug, PartialEq)]
  pub struct TripleKnowledge { pub t: TripleType, pub samples: usize, pub triples: u64, pub counts: Arc<[u32]>, pub pending: Vec<u32>, pub pending_samples: usize, pub compositions: Vec<(Composition, usize)> }
  #[derive(Clone, Debug, PartialEq)]
  pub struct TripleMap { pub grid: TripleGrid, pub r_cut: f64, pub reach_limit: f64, pub basis: TripleBasis, pub spans: Vec<PairSpan>, pub types: Vec<TripleKnowledge> }
  impl TripleMap {
      pub fn assemble(r_cut: f64, reach_limit: f64, basis: TripleBasis, spans: Vec<PairSpan>, fitted: &[TripleBlock], pending: &[TripleBlock]) -> TripleMap;
      pub fn get(&self, t: TripleType) -> Option<&TripleKnowledge>;
      pub fn span(&self, a: u8, b: u8) -> Option<&PairSpan>;
  }
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum CellClass { Taught, Thin, Missing, OutOfReach }
  pub fn reach(map: &TripleMap, t: TripleType, cell: usize) -> bool;
  pub fn cell_class(map: &TripleMap, tk: &TripleKnowledge, cell: usize) -> CellClass;
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
  pub struct TripleSummary { pub taught: usize, pub thin: usize, pub missing: usize, pub gaps: usize }
  pub fn summary(map: &TripleMap, tk: &TripleKnowledge) -> TripleSummary;
  ```
  - `assemble`: one `TripleKnowledge` per type of `fitted` ∪ `pending`, ascending. `counts` is the fitted block's `Arc` **cloned** (pointer-equal), or zeros for a pending-only type. `pending` is the pending block's counts, or zeros.
  - `reach_limit` is 5/6·r_cut, supplied by the builder from `providers::SCAN_CUTOFF_FRACTION` (no copy of that constant in `sim`).
  - `cell_class`: n > 0 gives Taught or Thin, in reach or not; n = 0 gives Missing in reach and OutOfReach otherwise; a folded type's k₁ > k₂ cell is OutOfReach.
  - `summary`: taught and thin over every filled cell; missing over in-reach empty cells; gaps = in-reach thin + missing.

- [ ] **Step 1: Write the failing tests** (test helpers: `span(z, lo, hi)` builds a `PairSpan`; `block(t, &[(cell, n)])` builds a `TripleBlock` on the r_cut-6 grid with those counts, `samples` 1, `triples` their sum and no compositions; `BASIS = TripleBasis { body3: true, body4: false, degree_cap: 4, l_max: 3 }`)

```rust
/// Spec M4.
#[test]
fn a_cell_is_taught_at_taught_count_and_pending_moves_nothing() {
    let g = TripleGrid::new(6.0);
    let (t, only_new) = (TripleType::new(8, 1, 1), TripleType::new(8, 1, 8));
    let (thin, taught, empty) = (g.index(true, 6, 6, 6), g.index(true, 6, 6, 5), g.index(true, 6, 6, 7));
    let spans = vec![span((1, 1), 2.6, 3.2), span((1, 8), 1.6, 2.2)];
    let fitted = block(t, &[(thin, taught_count() - 1), (taught, taught_count())]);
    let m0 = TripleMap::assemble(6.0, 5.0, BASIS, spans.clone(), &[fitted.clone()], &[]);
    let tk = m0.get(t).unwrap();
    assert_eq!([thin, taught, empty].map(|c| cell_class(&m0, tk, c)), [CellClass::Thin, CellClass::Taught, CellClass::Missing]);
    assert_eq!(cell_class(&m0, tk, g.index(false, 7, 6, 6)), CellClass::OutOfReach, "a folded type's k₁ > k₂ half is never a cell");
    let m1 = TripleMap::assemble(6.0, 5.0, BASIS, spans, &[fitted], &[block(t, &[(thin, 50), (empty, 50)]), block(only_new, &[(thin, 3)])]);
    let tk1 = m1.get(t).unwrap();
    assert_eq!([thin, taught, empty].map(|c| cell_class(&m1, tk1, c)), [CellClass::Thin, CellClass::Taught, CellClass::Missing], "pending never moves a class");
    assert!(Arc::ptr_eq(&tk.counts, &tk1.counts) && tk1.pending[empty] == 50);
    let n = m1.get(only_new).unwrap();
    assert_eq!((n.samples, n.triples, n.counts.iter().sum::<u32>(), n.pending_samples), (0, 0, 0, 1), "a type only pending samples hold");
    let s = summary(&m0, tk);
    let brute = (0..g.cells()).filter(|&c| cell_class(&m0, tk, c) == CellClass::Missing).count();
    assert_eq!(s, TripleSummary { taught: 1, thin: 1, missing: brute, gaps: brute + 1 });
}

/// Spec M5.
#[test]
fn reach_follows_the_three_pair_spans() {
    let g = TripleGrid::new(6.0);
    let hoh = TripleType::new(8, 1, 1);
    let m = TripleMap::assemble(6.0, 5.0, BASIS, vec![span((1, 1), 2.8, 4.0), span((1, 8), 1.5, 2.4)], &[], &[]);
    // 1 and 2: both distances in O–H's taught range, judged at the bin centre (1.65 in, 1.35 and 2.55 out)
    assert!(reach(&m, hoh, g.index(true, 5, 7, 9)));
    assert!(!reach(&m, hoh, g.index(true, 4, 6, 9)) && !reach(&m, hoh, g.index(true, 6, 8, 9)));
    // 3: r_ab ≥ H–H's shortest fitted distance 2.8 — at r₁ = r₂ = 1.95, cos −0.1 gives 2.89 (in), cos 0.1 gives 2.62 (out)
    assert!(reach(&m, hoh, g.index(true, 6, 6, 5)) && !reach(&m, hoh, g.index(true, 6, 6, 4)));
    // a–b never fitted within r_cut: the bound is r_cut, so no water-sized cell is in reach
    let no_hh = TripleMap::assemble(6.0, 5.0, BASIS, vec![span((1, 8), 1.5, 2.4)], &[], &[]);
    assert!((0..g.cells()).all(|c| !reach(&no_hh, hoh, c)));
    // the scan's limit: r ≤ reach_limit even inside the taught range (4.95 in, 5.25 out)
    let far = TripleMap::assemble(6.0, 5.0, BASIS, vec![span((1, 1), 0.5, 6.0), span((1, 8), 1.5, 5.8)], &[], &[]);
    assert!(reach(&far, hoh, g.index(true, 16, 16, 9)) && !reach(&far, hoh, g.index(true, 16, 17, 9)));
    // no c–a span: nothing
    assert!((0..g.cells()).all(|c| !reach(&m, TripleType::new(6, 1, 1), c)));
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p sim --release --lib knowledge::triples` — Expected: FAIL to compile (`TripleMap::assemble`, `reach` not found).

- [ ] **Step 3: Implement the interfaces above and the module doc.**

- [ ] **Step 4: Run the sim gate**

Run: `cargo test -p units -p qm -p sim -p surrogate --release` — Expected: PASS (~6.5 min).

- [ ] **Step 5: Physics review.** Dispatch `physics-reviewer` (Opus) on spec §6.1–§6.3 and the diff of Tasks 1–3: the neighbour rule against body-3's sum, the angle at the centre, the fold, the cos θ bins and edges, the reach's three conditions, pending moving no class. Fix its findings in this task, rerun Step 4, and record the review's verdict in the commit body.

- [ ] **Step 6: Commit**

```bash
git commit -o -m "feat(knowledge): the triple map, its reach from the pair spans, and the cell classes" -- crates/sim/src/knowledge.rs crates/sim/src/knowledge/triples.rs
```

---

### Task 4: The provider builds the map; the fitted block is cached — **Opus**

Physics correctness: which samples count. Fitted means exactly the knowledge map's rule (`providers/.../knowledge.rs:72-77`); pending means `is_pending`; a capped segment's caps and edge atoms are never centres. The reach's spans must equal the knowledge map's own taught ranges.

**Files:**
- Create: `crates/providers/src/surrogate/triples.rs`
- Modify: `crates/providers/src/surrogate/knowledge.rs:52` (`each_pair` → `pub(super)`)
- Modify: `crates/providers/src/surrogate/mod.rs` (`mod triples;`; a field `triple_cache: std::sync::Mutex<Option<triples::TripleCache>>` beside `fit_floor_cache`, initialised wherever that one is; the `Trainable` forwarder beside `knowledge_map`, line 3499)
- Modify: `crates/sim/src/provider.rs:1316-1320` (the trait method below `knowledge_map`)
- Create: `crates/providers/src/surrogate/tests/triples.rs`; Modify: `crates/providers/src/surrogate/tests.rs:11336` (`mod triples;`)

**Interfaces:**
- Consumes: Task 3's `TripleTally`, `TripleMap::assemble`, `PairSpan::of`, `TripleBasis`; `providers::SCAN_CUTOFF_FRACTION`.
- Produces:
  ```rust
  // sim::provider::Trainable
  fn triple_map(&self) -> Option<crate::knowledge::TripleMap> { None }
  // providers::surrogate
  impl SurrogateKs { pub fn triple_map(&self) -> Option<sim::knowledge::TripleMap>; } // None without a model or a store
  pub(super) struct TripleCache { key: (u64, u64, Vec<usize>), spans: Vec<PairSpan>, blocks: Vec<TripleBlock> }
  ```
  - The cache key is (`model_generation()`, `cutoff().to_bits()`, the fitted store indices). A verdict re-judge or new pending samples leave it equal; a refit, Forget, Restore or basis change does not.
  - The fitted pass runs `TripleTally::add(&s.z, &s.pos, |a| s.trains_atom(a))` and the pair histograms by `each_pair` (bins of `BIN` to r_cut), then `PairSpan::of` for each. The pending pass tallies only `is_pending` samples, on every call.
  - `reach_limit` = `SCAN_CUTOFF_FRACTION * cutoff()`. `basis` comes from `self.basis()` (`body3`, `body4`, `degree_cap`, `l_max`), all false or 0 without one.

- [ ] **Step 1: Write the failing tests** (`h2_and_ch4` and `random_ch4` from `tests/knowledge.rs`, made `pub(super)`; `brute(s, indices)` is a test helper that counts by plain triple loops over each sample's atoms with its own distance and dot-product code, bins with `TripleGrid::cell`, and returns per type (counts, samples))

```rust
/// Spec B1.
#[test]
fn the_triple_map_counts_fitted_triples_and_pending_apart() {
    let mut s = h2_and_ch4(120, 200);
    let fitted: Vec<usize> = (0..s.accumulated).filter(|&i| !s.excluded(i) && !s.in_misfit(i)).collect();
    let ch4_fitted: Vec<usize> = fitted.iter().copied().filter(|&i| s.store.as_ref().unwrap().samples[i].z.len() == 5).collect();
    // one fitted CH₄ whose C does not train (a capped segment's mask), set before the first build
    let masked = *ch4_fitted.last().unwrap();
    assert_eq!(s.store.as_ref().unwrap().samples[masked].z[0], 6, "premise: C first");
    s.store.as_mut().unwrap().samples[masked].force_mask = Some(vec![false, true, true, true, true]);
    // six CH₄ kept since the refit; one forgotten, one box-affected
    let mut rng = StdRng::seed_from_u64(0x3b1);
    for _ in 0..6 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    let n = s.samples();
    s.quarantine.resize(n, Q_NONE);
    s.quarantine[n - 1] = Q_FORGOTTEN;
    s.face.resize(n, f64::NAN);
    s.face[n - 2] = 1.0;
    let m = s.triple_map().expect("a fitted model");
    let pending: Vec<usize> = (0..n).filter(|&i| s.is_pending(i)).collect();
    assert_eq!(pending.len(), 4);
    let (bf, bp) = (brute(&s, &fitted), brute(&s, &pending));
    assert_eq!(m.types.iter().map(|k| k.t).collect::<BTreeSet<_>>(), bf.keys().chain(bp.keys()).copied().collect());
    let zero = || (vec![0u32; m.grid.cells()], 0usize);
    for k in &m.types {
        let (fc, fs) = bf.get(&k.t).cloned().unwrap_or_else(zero);
        let (pc, ps) = bp.get(&k.t).cloned().unwrap_or_else(zero);
        assert_eq!((&k.counts[..], k.samples), (&fc[..], fs), "fitted {:?}", k.t);
        assert_eq!((&k.pending[..], k.pending_samples), (&pc[..], ps), "pending {:?}", k.t);
    }
    let hch = m.get(TripleType::new(6, 1, 1)).unwrap();
    let held = ch4_fitted.len() - 1;
    assert_eq!((hch.samples, hch.triples), (held, 6 * held as u64), "six H–C–H per CH₄, the masked C adds none");
    assert_eq!(hch.compositions, vec![(Composition::of(&[6, 1, 1, 1, 1]), held)]);
    assert_eq!(hch.pending_samples, 4, "the forgotten and the box-affected count nowhere");
    // the reach's spans are the knowledge map's taught ranges, from the same fitted samples
    let km = s.knowledge_map().unwrap();
    for p in km.pairs.iter() {
        assert_eq!(m.span(p.z.0, p.z.1).copied(), PairSpan::of(&p.hist), "{:?}", p.z);
    }
    assert_eq!((m.r_cut, m.reach_limit), (s.cutoff(), SCAN_CUTOFF_FRACTION * s.cutoff()));
    assert_eq!(m.basis.body3, s.basis().unwrap().body3);
}

/// Spec B2.
#[test]
fn a_pending_only_rebuild_reuses_the_fitted_block() {
    let mut s = h2_and_ch4(120, 200);
    let t = TripleType::new(6, 1, 1);
    let m0 = s.triple_map().unwrap();
    assert!(Arc::ptr_eq(&m0.get(t).unwrap().counts, &s.triple_map().unwrap().get(t).unwrap().counts), "nothing changed: cached");
    let mut rng = StdRng::seed_from_u64(0x5e7);
    for _ in 0..4 {
        let sc = random_ch4(&mut rng);
        s.add_sample(pair_morse_sample(&sc), &sc).unwrap();
    }
    let m1 = s.triple_map().unwrap();
    let (k0, k1) = (m0.get(t).unwrap(), m1.get(t).unwrap());
    assert!(Arc::ptr_eq(&k0.counts, &k1.counts), "pending-only: the fitted block is the cached one");
    assert_eq!((k1.pending_samples, k1.pending.iter().sum::<u32>()), (4, 24));
    s.refit().unwrap();
    let m2 = s.triple_map().unwrap();
    let k2 = m2.get(t).unwrap();
    assert!(!Arc::ptr_eq(&k0.counts, &k2.counts) && k2.triples > k0.triples, "a refit recounts");
    assert!(m2.types.iter().all(|k| k.pending_samples == 0));
    assert!(SurrogateKs::new(Box::new(PairMorse(MorseDimer::h2())), FallbackPolicy::default()).triple_map().is_none(), "no model, no map");
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p providers --release --lib triples` — Expected: FAIL to compile (`triple_map` not found).

- [ ] **Step 3: Implement the interfaces above.**

- [ ] **Step 4: Run them, then the providers gate**

Run: `cargo test -p providers --release --lib triples` — Expected: PASS.
Run: `cargo test -p providers --release` — Expected: PASS (~5.5 min).

- [ ] **Step 5: Commit**

```bash
git add crates/providers/src/surrogate/triples.rs crates/providers/src/surrogate/tests/triples.rs
git commit -o -m "feat(surrogate): the triple map builder, its fitted block cached per fit" -- crates/providers/src/surrogate/triples.rs crates/providers/src/surrogate/knowledge.rs crates/providers/src/surrogate/mod.rs crates/providers/src/surrogate/tests.rs crates/providers/src/surrogate/tests/triples.rs crates/providers/src/surrogate/tests/knowledge.rs crates/sim/src/provider.rs
```

---

### Task 5: The worker builds it only while shown, on the knowledge map's schedule — **Opus**

**Files:**
- Modify: `crates/app/src/worker/commands.rs` (`SetTripleView(bool)` after `SetActivityViews`, line 138, in the same four classifier arms and in `apply`, line 525)
- Modify: `crates/app/src/worker/activity.rs` (module doc paragraph "The three-atom map"; `ActivityDelta::triples`; `ActivityWork` fields and `set_triples`; `KnowledgeKey::triple_part`; `activity_delta`; `views_wake`)
- Modify: `crates/app/src/ui/activity/mod.rs` (`triples`, `triples_sent`; `absorb` keeps and drops `triples` exactly as `knowledge`)
- Modify: `crates/app/src/viewer.rs:676-679` (`triples_sent = false; triples = None;`)
- Test: `crates/app/src/worker/activity.rs` (`schedule_tests`), `crates/app/src/worker/tests.rs`, `crates/app/src/ui/tests.rs`

**Interfaces:**
- Consumes: Task 4's `Trainable::triple_map`.
- Produces:
  ```rust
  Command::SetTripleView(bool)                       // not an edit, never pauses, never reaches the driver, never aborts
  ActivityDelta { pub triples: Option<Arc<sim::knowledge::TripleMap>>, .. } // None = no new build
  impl ActivityWork { pub(super) fn set_triples(&mut self, on: bool) } // on after off: the next publish builds; off: no deferred wake
  impl KnowledgeKey { fn triple_part(&self) -> KnowledgeKey }  // verdicts emptied, trusted false: a re-judge rebuilds nothing
  ActivityState { pub triples: Option<Arc<sim::knowledge::TripleMap>>, pub triples_sent: bool, .. }
  ```
  - Schedule: while `triples_on` and the model is fitted, `knowledge_build(triples_key, triples_built, &kkey.triple_part(), now)`: `Now` builds, `At(due)` sets `triples_due_at` (woken by `views_wake`), `Skip` does nothing. When both maps build in one publish, the triple build's ms is added to `knowledge_built`'s ms (§6.4: the duty backstop covers both).

- [ ] **Step 1: Write the failing tests**

```rust
// worker/activity.rs, schedule_tests
#[test]
fn the_triple_key_ignores_verdicts_and_trust() {
    let m = crate::ui::tests::model();
    let mut judged = m.clone();
    judged.trusted = !m.trusted;
    if let Some(c) = judged.class_loo.first_mut() {
        c.trusted = !c.trusted;
    }
    assert_ne!(KnowledgeKey::of(&m), KnowledgeKey::of(&judged), "premise: the knowledge map rebuilds");
    assert_eq!(KnowledgeKey::of(&m).triple_part(), KnowledgeKey::of(&judged).triple_part(), "a re-judge changes no count");
    let new = KnowledgeKey::of(&ModelStatus { pending: 3, ..m.clone() }).triple_part();
    assert!(new != KnowledgeKey::of(&m).triple_part() && new.same_fit(&KnowledgeKey::of(&m).triple_part()), "pending-only: rate-limited");
}

// worker/tests.rs
#[test]
fn the_triple_map_ships_only_while_shown() {
    assert!(![Command::SetTripleView(true).is_edit(), Command::SetTripleView(true).pauses(), Command::SetTripleView(true).reaches_driver(), Command::SetTripleView(true).aborts()].contains(&true));
    let w = morse_estimated(h2_33());
    idle(&w);
    w.send(Command::EnsureModel { autoload: None, plan: SamplingPlan::default(), auto_pass: true });
    let mut early = false;
    snapshots_until(&w, "a fitted model", |s| early |= s.activity.triples.is_some(), |s| s.model.as_ref().is_some_and(|m| m.fitted) && !s.working);
    assert!(!early, "not built while hidden");
    w.send(Command::SetTripleView(true));
    let s = snapshot_until(&w, "the triple map", |s| s.activity.triples.is_some());
    let t = s.activity.triples.as_ref().unwrap();
    assert!(t.types.is_empty() && t.span(1, 1).is_some(), "H₂ holds no triple, but its H–H span is there");
    w.send(Command::Dyn(DynCommand::SetSpeed(3)));
    assert!(idle(&w).activity.triples.is_none(), "nothing changed: no rebuild");
    w.send(Command::SetAudit(7));
    snapshot_until(&w, "a rebuild after a new model generation", |s| s.activity.triples.is_some());
    w.send(Command::SetTripleView(false));
    w.send(Command::SetAudit(5));
    let mut built = false;
    snapshots_until(&w, "the knowledge map after the policy change", |s| built |= s.activity.triples.is_some(), |s| s.activity.knowledge.is_some());
    assert!(!built, "hidden again: not built");
}

// ui/tests.rs
#[test]
fn the_triple_map_is_kept_like_the_knowledge_map() {
    // absorb: Some is kept, a later None keeps it, a snapshot with no fitted model drops it
}
```
The third test's body follows `ActivityState::absorb`'s knowledge tests: three snapshots, `fixture_triples()` (Task 6) or `TripleMap::assemble(6.0, 5.0, …, &[], &[])`, assertions on `st.activity.triples.is_some()` after each.

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p app --release --bin app -- triple` — Expected: FAIL to compile (`SetTripleView`, `triple_part`, `triples` not found).

- [ ] **Step 3: Implement the interfaces above.**

- [ ] **Step 4: Run them to see them pass**

Run: `cargo test -p app --release --bin app -- triple knowledge views` — Expected: PASS, the existing knowledge and views schedule tests unchanged.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(worker): build the triple map only while shown, on the knowledge map's schedule" -- crates/app/src/worker/commands.rs crates/app/src/worker/activity.rs crates/app/src/worker/tests.rs crates/app/src/ui/activity/mod.rs crates/app/src/viewer.rs crates/app/src/ui/tests.rs
```

---

### Task 6: The section, the picker, the card head and the entry links — **Opus**

**Files:**
- Create: `crates/app/src/ui/knowledge/triples.rs` (declared `mod triples;` in `ui/knowledge.rs`)
- Modify: `crates/app/src/ui/knowledge.rs` (`Pick::Triple`; `Sections { why, triples }` with encode `"why,triples"`; `KnowledgeView::triple`, kept by `reset_view`; `body` places the section under `detail` in both layouts; `detail`'s `Pick::Triple` arm; the pair card's link under its sentence; the element card's "Arrangements around {c}" line; `#[cfg(test)] fixture_triples()`)
- Modify: `crates/app/src/ui/mod.rs:1446-1450` (after `SetActivityViews`)
- Modify: `crates/app/src/ui/profile.rs:10` (the doc names `knowledge_sections = why,triples`)
- Test: `crates/app/src/ui/tests.rs` (new `mod triple_map`)

**Interfaces:**
- Consumes: Task 3's map, `summary`, `TripleType`; Task 5's `ActivityState::{triples, triples_sent}`, `Command::SetTripleView`; `activity::words::thousands`.
- Produces (`ui::knowledge::triples` unless noted):
  ```rust
  // ui::knowledge
  pub enum Pick { .., Triple(TripleType) }
  pub struct Sections { pub why: bool, pub triples: bool }
  #[derive(Clone, Debug, PartialEq)]
  pub struct TripleView { pub mode: TripleMode, pub slice: Option<usize> /* k₂ */, pub cell: Option<(usize, usize, usize)> /* (k₁, k₂, q): bins are 0.3 Bohr at every r_cut, so a pick survives a reach change */, pub yaw_deg: f32, pub pitch_deg: f32, pub zoom: f32, pub show_missing: bool, pub want_centre: Option<u8>, pub pose: Option<ShotPose> /* Task 9 */, pub cache: Option<ClassCache> /* Task 7 */ }
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
  pub enum TripleMode { #[default] Flat, Cloud }
  // ui::knowledge::triples
  pub fn triple_name(t: TripleType) -> String;       // a–c–b with pair_name's rule (H last): "H–O–H", "O–H–H" for centre H with H and O
  pub fn triple_probe(t: TripleType) -> String;      // "knowledge-triple-row-{a}-{c}-{b}" (Z numbers)
  pub fn scene_types(nuclei: &[SceneNucleus], r_cut: f64) -> BTreeSet<TripleType>; // each_triple, every atom a centre
  #[derive(Clone, Debug, PartialEq)]
  pub struct PickerRow { pub t: TripleType, pub gaps: usize, pub triples: u64, pub samples: usize, pub pending: usize, pub in_scene: bool }
  pub fn picker_rows(map: &TripleMap, scene: &BTreeSet<TripleType>) -> Vec<PickerRow>; // types with samples or pending only, in `order`
  pub fn order(rows: &mut [PickerRow]);              // gaps desc, then triples asc, then triple_name
  pub fn row_text(r: &PickerRow) -> String;
  pub fn shown_type(view: &KnowledgeView, map: &TripleMap, scene: &BTreeSet<TripleType>) -> Option<TripleType>; // Pick::Triple if held; else want_centre's most-gaps type; else the scene's most-gaps type; else the first row
  pub fn head_sentence(map: &TripleMap, tk: &TripleKnowledge) -> String;
  pub fn verdict_line(keys: &[(&str, &sim::SpeciesVerdict)]) -> Option<String>;
  pub fn section(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, r_cut: f64);
  pub fn card_head(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, map: &TripleMap, tk: &TripleKnowledge);
  ```
  - `impl Default for TripleView`: `Flat`, yaw 35°, pitch 20°, zoom 1, `show_missing` true, the rest `None` (Task 8 points the two angles at `cloud::DEFAULT_*`). `KnowledgeView::reset_view` keeps `triple` and clears only its `cache`.
  - `SetTripleView`'s want = `st.tab_shown(DockTab::Knowledge) && st.knowledge_sections.triples`, sent on a change against `st.activity.triples_sent`.
  - `verdict_line`'s keys are the Model status's verdicts with `z == t.centre` whose neighbours contain a and b (the `keys_of` pattern), unlearned first; the first is worded and " (+ {k} more)" follows. A click reveals the Model tab's Learned section (`Section::Learned`, anchor `d-model-learned`), as `molecule_detail` reveals Suspects.
  - Entry links: the element card lists one link per held type centred on that element (probe `knowledge-triple-link-{a}-{c}-{b}`: pick it and open the fold); with no triple map yet, one link `Show three-atom arrangements` (probe `knowledge-triple-link-element`) sets `want_centre` and opens the fold. The pair card, when its sentence names a short many-body key, adds `Show arrangements around {sym}` (probe `knowledge-triple-link-pair`): `want_centre` = that key's centre, fold open.
  - Empty states (Review Focus 4), copy pinned by the test: map not yet shipped: `building the three-atom map…`; no types: `No three-atom arrangements in the fitted samples: no atom has two neighbours within {r_cut:.1} Bohr.`; `body3` false: `this model's basis has no three-atom terms: it does not learn the angle` (the head shows it in place of the classes); a picked type the map no longer holds: `{name} is no longer in the model's fitted or new samples`.

- [ ] **Step 1: Write the failing tests** (`fixture_triples()`, built with `TripleMap::assemble`: r_cut 6, reach_limit 5, spans H–H 2.6–3.2 and O–H 1.6–2.2, body3 on and body4 off; H–O–H with fitted 342 at (6, 6, 6), 9 at (5, 6, 6), 10 at (6, 6, 5), samples 610, triples 2140, and a pending block of 12 at (6, 6, 6) with 12 samples; O–H–H (centre H) with 120 at (9, 6, 1), samples 140, triples 120; a pending-only H–O–O (`TripleType::new(8, 1, 8)`) with 3 at (6, 6, 6) in 1 sample)

```rust
#[test]
fn the_picker_lists_held_types_by_gaps() {
    let row = |t: TripleType, gaps, triples, pending| PickerRow { t, gaps, triples, samples: 610, pending, in_scene: false };
    let (hoh, ohh, hhh) = (TripleType::new(8, 1, 1), TripleType::new(1, 8, 1), TripleType::new(1, 1, 1));
    let mut rows = vec![row(hhh, 3, 50, 0), row(hoh, 14, 2140, 12), row(ohh, 14, 900, 0)];
    triples::order(&mut rows);
    assert_eq!(rows.iter().map(|r| r.t).collect::<Vec<_>>(), vec![ohh, hoh, hhh], "gaps desc, then fewer arrangements first");
    assert_eq!(triples::row_text(&rows[1]), "H–O–H · 14 gaps · 2,140 arrangements in 610 samples · +12 new");
    assert_eq!(triples::row_text(&rows[0]), "O–H–H · 14 gaps · 900 arrangements in 610 samples");
    let map = knowledge::fixture_triples();
    let listed: Vec<TripleType> = triples::picker_rows(&map, &BTreeSet::new()).iter().map(|r| r.t).collect();
    assert_eq!(listed.iter().copied().collect::<BTreeSet<_>>(), map.types.iter().map(|k| k.t).collect(), "only the held types, the pending-only one included");
    assert_eq!(triples::triple_name(TripleType::new(1, 1, 8)), "O–H–H");
}

#[test]
fn the_triple_view_command_follows_the_fold_and_a_respawn() {
    // the_views_command_is_resent_after_a_worker_respawn's shape: Knowledge open, fold closed → no command;
    // st.knowledge_sections.triples = true → [SetTripleView(true)] once, then nothing;
    // st.activity.triples_sent = false (Viewer::start) → [SetTripleView(true)] again;
    // st.open_tab(DockTab::Log) in the same leaf → [SetTripleView(false)]; back → [SetTripleView(true)];
    // a picker row click, a mode change and a cell pick → out.actions empty and no further command.
}

#[test]
fn the_section_explains_an_empty_or_three_body_free_map() {
    // fold open, st.activity.triples None → "building the three-atom map…";
    // TripleMap::assemble(6.0, 5.0, BASIS, spans, &[], &[]) → the "No three-atom arrangements …" sentence, no picker row probe;
    // fixture_triples() with basis.body3 = false → "this model's basis has no three-atom terms: it does not learn the angle";
    // st.knowledge.pick = Pick::Triple(TripleType::new(6, 1, 1)) → "H–C–H is no longer in the model's fitted or new samples",
    // and the section shows the default type; no frame panics.
}

#[test]
fn the_card_head_reads_the_type_and_its_verdict() {
    let map = knowledge::fixture_triples();
    let tk = map.get(TripleType::new(8, 1, 1)).unwrap();
    let s = sim::knowledge::summary(&map, tk);
    assert_eq!(triples::head_sentence(&map, tk), format!("2,140 arrangements of H–O–H in 610 samples ({} cells taught, {} thin, {} missing within reach) · +12 new since the last refit.", s.taught, s.thin, s.missing));
    let ok = key(8, vec![1], None, 967, 38);
    let short = key(8, vec![1, 6], Some((sim::ClassKey::Samples, 276.0, 322.0)), 276, 322);
    assert_eq!(triples::verdict_line(&[("many-body", &ok)]).unwrap(), "the model's many-body key around O, with H: trusted");
    assert_eq!(triples::verdict_line(&[("many-body", &short), ("many-body", &ok)]).unwrap(), "the model's many-body key around O, with H and C: 276 of 322 samples — the per-environment sample floor, not this map (+ 1 more)");
    assert_eq!(triples::verdict_line(&[]), None);
}
```
`key(...)` is the existing `SpeciesVerdict` helper shape at `ui/tests.rs:11395`. The two sketched tests are written as full tests from their comments, with the copy quoted there pinned verbatim. The entry links get one test, `the_entry_links_open_the_section`: a click on `knowledge-triple-link-pair` (the `the_pair_detail_lists_the_keys_behind_its_verdict` setup, H–H thin on the key around H) sets `want_centre == Some(1)` and opens the fold, and once the fixture map is present `shown_type` is the H-centred type with the most gaps.

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p app --release --bin app -- triple` — Expected: FAIL to compile (`ui::knowledge::triples` not found).

- [ ] **Step 3: Implement the interfaces above.** The section is a `theme::fold_header` "Three-atom arrangements" (probe `knowledge-triple-fold`); the picker is a list of rows (probes `triple_probe(t)`, a dot for `in_scene`); the head is a card (probe `knowledge-triple-head`, verdict line probe `knowledge-triple-verdict`). The view area below the picker is a placeholder `knowledge-triple-view` until Task 7.

- [ ] **Step 4: Run them, then the knowledge tests**

Run: `cargo test -p app --release --bin app -- triple knowledge profile` — Expected: PASS; `the_knowledge_tab_draws_the_map_and_picks_by_click` and the other knowledge tests unchanged (the fold is closed by default, so their "display only" assertion still holds).

- [ ] **Step 5: Commit** (hand-tests in the body: open the fold, pick rows, follow the element and pair links, close the panel and watch the map stop updating)

```bash
git add crates/app/src/ui/knowledge/triples.rs
git commit -o -m "feat(knowledge): the Three-atom arrangements section, its picker, card and links" -- crates/app/src/ui/knowledge.rs crates/app/src/ui/knowledge/triples.rs crates/app/src/ui/mod.rs crates/app/src/ui/profile.rs crates/app/src/ui/tests.rs
```

---

### Task 7: The 2D slice, hover, the cell card and the band — **Opus**

**Files:**
- Modify: `crates/app/src/ui/knowledge/triples.rs`
- Test: `crates/app/src/ui/tests.rs` (`mod triple_map`)

**Interfaces:**
- Consumes: Task 6's `TripleView`, `shown_type`; Task 3's `cell_class`, `TripleGrid`; `sim::knowledge::{each_triple, Triple, cos_edges_deg}`.
- Produces:
  ```rust
  #[derive(Clone, Debug, PartialEq)]
  pub struct ClassCache { pub map: usize /* Arc::as_ptr as usize */, pub t: TripleType, pub classes: Vec<CellClass>, pub extent: Option<(Range<usize>, Range<usize>)> }
  pub fn class_table(map: &TripleMap, tk: &TripleKnowledge) -> Vec<CellClass>;
  pub fn drawn(tk: &TripleKnowledge, class: CellClass, cell: usize) -> bool; // fitted > 0, pending > 0, or Missing
  pub fn extent(map: &TripleMap, tk: &TripleKnowledge, classes: &[CellClass]) -> Option<(Range<usize>, Range<usize>)>; // (r₁ rows, r₂ slices), contiguous, over drawn cells; folded: both ranges are the union of k₁ and k₂
  pub fn slice_cells(grid: &TripleGrid, folded: bool, rows: Range<usize>, s: usize) -> Vec<[usize; COS_BINS]>; // grid.index(folded, k₁, s, q)
  pub fn scene_instances(nuclei: &[SceneNucleus], t: TripleType, r_cut: f64) -> Vec<Triple>; // each_triple, every atom a centre
  pub fn chosen_instance(instances: &[Triple], selected: Option<usize>) -> Option<Triple>; // centre == selected: least r₁ + r₂; else the lowest (centre, j, k)
  pub fn default_slice(map: &TripleMap, tk: &TripleKnowledge, chosen: Option<&Triple>) -> usize; // chosen's k₂, else the most-filled k₂
  pub fn on_slice(grid: &TripleGrid, inst: &Triple, s: usize) -> Option<usize>; // the row it sits on in slice s (folded: k₂ when k₁ = s, k₁ when k₂ = s)
  pub fn angle_labels() -> Vec<String>;                                   // "0", "37", …, "180"
  pub fn cell_hover(map: &TripleMap, tk: &TripleKnowledge, cell: usize) -> String;
  pub fn cell_card_line(map: &TripleMap, tk: &TripleKnowledge, cell: usize) -> String;
  pub fn band_lines(map: &TripleMap) -> Vec<String>;
  ```
  - The class table and extent are computed once per (map `Arc` pointer, type) into `TripleView::cache` (§6.4 UI cost).
  - Drawing: x is the angle (10 columns, `angle_labels` at the edges, caption `angle at {c} (cos θ bins)`), y is r₁ over `extent`'s rows, labelled `r₁ ({pair_name(c, a)}) Bohr`; a slider over `extent`'s slices (probe `knowledge-triple-slice`) labelled `r₂ ({pair_name(c, b)}) {lo:.1}–{hi:.1} Bohr`. Fill GREEN or AMBER by class with opacity rising with ln(1 + n); Missing an `INK_FAINT` outline; pending > 0 a dashed `NEW_INK` outline; styling per §10. Each drawn cell rect is probed `knowledge-triple-cell-{index}`. Markers: a white ring and `now` at each instance on the slice, a faint ring for the others with "(+k in other slices)" in the hover.
  - Hover shows `cell_hover`; a click sets `TripleView::cell` to the cell's (k₁, k₂, q). The cell card (probe `knowledge-triple-cell-card`) shows `cell_card_line` and the band (probe `knowledge-triple-band`, `band_lines`, one label per line). No Teach control.
  - Stale state (Review Focus 1): each frame, a `slice` outside `extent`'s slices is clamped into it, and a `cell` with k₁ or k₂ ≥ n_r, or no longer drawn, is dropped.

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec U1: rows cropped to reach ∪ filled, the fold read, markers in the right cell.
#[test]
fn the_slice_model_crops_folds_and_marks() {
    let map = knowledge::fixture_triples();
    let t = TripleType::new(8, 1, 1);
    let tk = map.get(t).unwrap();
    let classes = triples::class_table(&map, tk);
    let (rows, slices) = triples::extent(&map, tk, &classes).unwrap();
    let want: Vec<usize> = (0..map.grid.cells()).filter(|&c| triples::drawn(tk, classes[c], c)).flat_map(|c| { let (a, b, _) = map.grid.coords(c); [a, b] }).collect();
    assert_eq!((rows.start, rows.end), (*want.iter().min().unwrap(), want.iter().max().unwrap() + 1));
    assert_eq!(rows, slices, "folded: one range for both");
    assert!(rows.contains(&5) && rows.contains(&6), "the fixture's filled cells");
    let cells = triples::slice_cells(&map.grid, true, rows.clone(), 5);
    assert_eq!(cells[6 - rows.start][3], map.grid.index(false, 5, 6, 3), "slice 5, row 6 reads the stored (5, 6) cell");
    let water = Scene::preset("h2o").unwrap();
    let inst = triples::scene_instances(&water.nuclei, t, map.r_cut);
    assert_eq!(inst.len(), 1);
    let (k1, k2, _) = map.grid.coords(map.grid.cell(true, inst[0].r1, inst[0].r2, inst[0].u).unwrap());
    assert_eq!(triples::on_slice(&map.grid, &inst[0], k2), Some(k1));
    assert_eq!(triples::default_slice(&map, tk, inst.first()), k2);
    assert_eq!(triples::chosen_instance(&inst, None), inst.first().copied());
}

#[test]
fn hover_card_and_band_read_the_spec_copy() {
    let map = knowledge::fixture_triples();
    let tk = map.get(TripleType::new(8, 1, 1)).unwrap();
    let c = map.grid.index(true, 6, 6, 6);
    assert_eq!(triples::angle_labels(), ["0", "37", "53", "66", "78", "90", "102", "114", "127", "143", "180"]);
    assert_eq!(triples::cell_hover(&map, tk, c), "r₁ 1.8–2.1 · r₂ 1.8–2.1 Bohr · θ 102–114° · n 342 (+12 new) · taught");
    assert_eq!(triples::cell_card_line(&map, tk, c), "H–O–H · r₁ 1.8–2.1, r₂ 1.8–2.1 Bohr, θ 102–114° · 342 arrangements · taught · +12 new");
    assert!(triples::cell_hover(&map, tk, map.grid.index(true, 5, 6, 6)).ends_with("n 9 · thin"));
    let band = triples::band_lines(&map);
    assert_eq!(band[0], "Green counts arrangements in the cell (10 within 0.3 Bohr in each distance and 0.2 in cos θ); it is not the model's trust or accuracy.");
    assert_eq!(band[1], "Counts every pair of neighbours within 6.0 Bohr of the centre, as the model's three-atom terms do.");
    assert_eq!(band[2], "The model's three-atom terms are smooth in the distances and in cos θ: it may interpolate across a missing cell, and a green cell can still be wrong.");
    let mut b4 = map.clone();
    b4.basis.body4 = true;
    assert!(triples::band_lines(&b4)[1].ends_with(" Four-atom arrangements (body-4) are learned too and not shown."));
    b4.basis.body3 = false;
    assert_eq!(triples::band_lines(&b4)[1], "this model's basis has no three-atom terms: it does not learn the angle");
    assert!(band.iter().all(|l| !l.contains("Teach")), "Phase A: the band says nothing about teaching");
}

/// Review Focus 1.
#[test]
fn stale_view_state_is_clamped_after_the_map_changes() {
    // fold open, the fixture map, Pick::Triple(H–O–H); st.knowledge.triple.slice = Some(30), cell = Some((25, 25, 3));
    // one frame → slice ∈ extent's slices, cell None (past n_r 20), no panic;
    // click_widget("knowledge-triple-cell-{index of (6, 6, 6)}") → cell == Some((6, 6, 6)) and the card shows;
    // a map assembled at r_cut 10 (n_r 34) with H–O–H at its own (6, 6, 6) shipped in its place → the pick is kept and
    // the card reads the new map's count; that map with (6, 6, 6) emptied and out of reach → the pick is dropped.
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p app --release --bin app -- triple` — Expected: FAIL to compile (`class_table`, `extent` not found).

- [ ] **Step 3: Implement the interfaces and the drawing above.** The plot takes clicks with `Sense::click()` through `ui::keyed("knowledge-triple-plot", …)` (probe `knowledge-triple-plot`).

- [ ] **Step 4: Run them to see them pass**

Run: `cargo test -p app --release --bin app -- triple` — Expected: PASS.

- [ ] **Step 5: Commit** (hand-tests: hover cells, move the slice, pick a cell, watch the dashed outlines appear during a pass and turn solid at the refit, Forget the samples of a type while it is shown)

```bash
git commit -o -m "feat(knowledge): the 2D triple slice, its hover, the cell card and the band" -- crates/app/src/ui/knowledge/triples.rs crates/app/src/ui/tests.rs
```

---

### Task 8: The 3D voxel cloud and the 2D/3D switch — **Opus**

Rendering geometry, not physics: an orthographic projection of the same cells.

**Files:**
- Create: `crates/app/src/ui/knowledge/cloud.rs` (declared in `ui/knowledge.rs`)
- Modify: `crates/app/src/ui/knowledge/triples.rs` (the `2D | 3D` switch, probes `knowledge-triple-mode-2d` and `knowledge-triple-mode-3d`; the 3D branch)
- Test: `crates/app/src/ui/knowledge/cloud.rs` (pure tests), `crates/app/src/ui/tests.rs`

**Interfaces:**
- Consumes: Task 7's `ClassCache`, `extent`, `drawn`, `scene_instances`.
- Produces (`ui::knowledge::cloud`):
  ```rust
  pub const DEFAULT_YAW_DEG: f32 = 35.0;
  pub const DEFAULT_PITCH_DEG: f32 = 20.0;
  pub const PITCH_LIMIT_DEG: f32 = 89.0;
  pub const CUBE_FILL: f32 = 0.8;                 // a cube is 0.8 of its cell
  pub const SHADES: [f32; 3] = [1.0, 0.85, 0.7];  // the visible faces
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct Voxel { pub cell: usize, pub at: [f32; 3], pub cube: bool } // at in [−½, ½]³ over the extent: (r₁, θ, r₂); cube false = a missing cell's square
  pub struct Drawn { pub cell: usize, pub depth: f32, pub faces: Vec<([Pos2; 4], f32)>, pub hull: Vec<Pos2> }
  pub fn voxels(grid: &TripleGrid, cells: &[usize], extent: &(Range<usize>, Range<usize>), cube: impl Fn(usize) -> bool) -> Vec<Voxel>;
  pub fn project(voxels: &[Voxel], yaw_deg: f32, pitch_deg: f32, zoom: f32, rect: Rect) -> Vec<Drawn>; // back to front; equal depth by ascending cell
  pub fn hit(drawn: &[Drawn], p: Pos2) -> Option<usize>;  // the front-most hull holding p
  pub fn rotate(view: &mut TripleView, drag: Vec2);       // yaw and pitch, pitch clamped to ±PITCH_LIMIT_DEG
  pub fn reset(view: &mut TripleView);                    // the defaults, zoom 1
  ```
  - Convention: at yaw 0°, pitch 0° the screen's x is r₁, its up is θ, and the viewer looks along +r₂, so of two cells differing only in r₂ the smaller r₂ is in front.
  - The UI: drag rotates, the wheel zooms, a double-click or Reset (probe `knowledge-triple-reset`) calls `reset`. A `missing` checkbox (probe `knowledge-triple-missing`, on by default) shows the missing cells as small `INK_FAINT` outlined squares, which the hit test also picks. Pending cells get a dashed square. Scene instances are white crosses. A click opens the same cell card as in 2D. The cloud's rect is probed `knowledge-triple-cloud`.

- [ ] **Step 1: Write the failing tests**

```rust
/// Spec U3.
#[test]
fn the_cloud_hit_test_picks_the_front_cube() {
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 400.0));
    let v = [Voxel { cell: 7, at: [0.0, 0.0, -0.3], cube: true }, Voxel { cell: 3, at: [0.0, 0.0, 0.3], cube: true }];
    let d = project(&v, 0.0, 0.0, 1.0, rect);
    assert_eq!(d.iter().map(|x| x.cell).collect::<Vec<_>>(), vec![3, 7], "back to front");
    assert_eq!(hit(&d, rect.center()), Some(7), "the front-most");
    assert_eq!(hit(&project(&v, 180.0, 0.0, 1.0, rect), rect.center()), Some(3));
    assert_eq!(hit(&d, pos2(1.0, 1.0)), None);
    let side = [Voxel { cell: 9, at: [-0.3, 0.0, 0.0], cube: true }, Voxel { cell: 2, at: [0.3, 0.0, 0.0], cube: true }];
    assert_eq!(project(&side, 0.0, 0.0, 1.0, rect).iter().map(|x| x.cell).collect::<Vec<_>>(), vec![2, 9], "equal depth: by cell");
    let a = project(&v, 35.0, 20.0, 1.0, rect);
    assert!(a.iter().all(|x| x.faces.len() <= 3));
    assert_eq!(a.iter().map(|x| x.cell).collect::<Vec<_>>(), project(&v, 35.0, 20.0, 1.0, rect).iter().map(|x| x.cell).collect::<Vec<_>>(), "stable");
    let mut view = TripleView::default();
    rotate(&mut view, vec2(0.0, -10_000.0));
    assert_eq!(view.pitch_deg.abs(), PITCH_LIMIT_DEG);
    reset(&mut view);
    assert_eq!((view.yaw_deg, view.pitch_deg, view.zoom), (DEFAULT_YAW_DEG, DEFAULT_PITCH_DEG, 1.0));
}

// ui/tests.rs
#[test]
fn the_3d_switch_draws_the_cloud_and_emits_nothing() {
    // fold open, fixture map, click "knowledge-triple-mode-3d" → st.knowledge.triple.mode == TripleMode::Cloud,
    // probed_rect("knowledge-triple-cloud").width() > 0, no Action; click "knowledge-triple-missing" → show_missing false;
    // click "knowledge-triple-reset" after a drag → the default angles.
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test -p app --release --bin app -- cloud triple` — Expected: FAIL to compile (`cloud` not found).

- [ ] **Step 3: Implement the interfaces above.** A cube's ≤ 3 visible faces are the ones whose normals face the viewer, shaded by `SHADES` in the order θ-face, r₁-face, r₂-face. Depth is the rotated centre's coordinate along the view direction.

- [ ] **Step 4: Run them to see them pass**

Run: `cargo test -p app --release --bin app -- cloud triple` — Expected: PASS.

- [ ] **Step 5: Commit** (hand-tests: switch 2D/3D, rotate, zoom, reset, pick a cube and a missing square, hide the missing cells)

```bash
git add crates/app/src/ui/knowledge/cloud.rs
git commit -o -m "feat(knowledge): the 3D triple cloud with rotate, zoom and a front-most hit test" -- crates/app/src/ui/knowledge.rs crates/app/src/ui/knowledge/cloud.rs crates/app/src/ui/knowledge/triples.rs crates/app/src/ui/tests.rs
```

---

### Task 9: The `--ui` keys and the shot's wait — **Opus**

**Files:**
- Modify: `crates/app/src/screenshot.rs` (doc paragraph beside the knowledge keys, lines 66-68; `ShotUi` fields; `parse`; `apply`; `wait_for_triples` beside `wait_for_views`, line 915; tests)
- Modify: `crates/app/src/ui/knowledge/triples.rs` (`ShotPose` resolved once the map arrives)
- Modify: `docs/dev-reference.md` ("Build, run and the app's defaults": the five keys; the `knowledge-triple-*` probes)

**Interfaces:**
- Consumes: Task 6's `TripleView`, `Pick::Triple`, `Sections::triples`; Task 8's `TripleMode`.
- Produces:
  ```rust
  // screenshot::ShotUi
  pub triple: Option<TripleType>, pub triples_3d: Option<bool>, pub slice: Option<f64>, pub cell: Option<(f64, f64, f64)>, pub rotate: Option<(f32, f32)>,
  // ui::knowledge::triples
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct ShotPose { pub slice_r: Option<f64>, pub cell: Option<(f64, f64, f64)> } // Bohr, Bohr, degrees
  ```
  - Keys (§6.8): `triple=<Za>-<Zc>-<Zb>` picks the type (either neighbour order) and opens the fold; `triples=2d|3d`; `slice=<r₂ Bohr>`; `cell=<r₁>,<r₂>,<deg>`; `rotate=<yaw>,<pitch>` (degrees). Z is 1–118, distances finite and > 0, the angle 0–180.
  - With `triple` set, the shot waits for `viewer.ui.activity.triples` (300 s, "no three-atom map within 300 s (is the model fitted?)"). The pose is resolved once against that map's grid: `slice = r_bin_of(slice_r)`, `cell = coords(grid.cell(folded, r1, r2, cos(deg)))`.

- [ ] **Step 1: Write the failing test**

```rust
/// Spec U8 (Phase A's keys).
#[test]
fn the_triple_keys_parse() {
    use sim::knowledge::TripleType;
    let s = ShotUi::parse("tab=knowledge;triple=1-8-1;triples=3d;rotate=35,20").unwrap();
    assert_eq!((s.triple, s.triples_3d, s.rotate), (Some(TripleType::new(8, 1, 1)), Some(true), Some((35.0, 20.0))));
    assert_eq!(ShotUi::parse("triple=8-1-1").unwrap().triple, Some(TripleType { centre: 1, a: 1, b: 8 }));
    let p = ShotUi::parse("triple=1-8-1;triples=2d;slice=1.9;cell=1.9,1.9,108").unwrap();
    assert_eq!((p.triples_3d, p.slice, p.cell), (Some(false), Some(1.9), Some((1.9, 1.9, 108.0))));
    for bad in ["triple=1-8", "triple=1-0-1", "triple=a-b-c", "triples=4d", "slice=-1", "slice=x", "cell=1.9,1.9", "cell=1.9,1.9,200", "rotate=35", "region=1-2,90-120"] {
        assert!(ShotUi::parse(bad).is_err(), "{bad}");
    }
    assert_eq!(ShotUi::parse("tab=knowledge").unwrap().triple, None);
}
```

- [ ] **Step 2: Run it to see it fail**

Run: `cargo test -p app --release --bin app -- the_triple_keys_parse` — Expected: FAIL to compile (no field `triple`).

- [ ] **Step 3: Implement the interfaces above and the dev-reference lines.**

- [ ] **Step 4: Run it to see it pass, then a real shot**

Run: `cargo test -p app --release --bin app -- the_triple_keys_parse` — Expected: PASS.
Run (no other app open on the shared collection; working directory `D:\Projects\Physics`, so the BEST collection is the model): `D:\Projects\Physics\.claude\worktrees\triple-map\target\release\app.exe --screenshot <scratchpad>\t2d.png --preset h2o --estimated --ui "wait=model;tab=knowledge;triple=1-8-1;triples=2d;cell=1.9,1.9,108"` — Expected: exit 0, the PNG shows the slice and the cell card.

- [ ] **Step 5: Commit**

```bash
git commit -o -m "feat(screenshot): --ui triple, triples, slice, cell and rotate; wait for the triple map" -- crates/app/src/screenshot.rs crates/app/src/ui/knowledge/triples.rs docs/dev-reference.md
```

---

### Task 10: Measure the build cost on the BEST collection — **Opus**

**Files:**
- Create: `crates/providers/src/surrogate/tests/triple_cost.rs` (`#[ignore]`d measurement); Modify: `crates/providers/src/surrogate/tests.rs` (`mod triple_cost;` beside `mod refit_profile;`), `tests/refit_profile.rs` (`open_copy`, `attach_timed`, `provider` → `pub(super)`)
- Modify: `crates/validate/references.md` (a new section "Three-atom coverage map build cost (2026-11-17)")

**Interfaces:**
- Consumes: Task 4's `SurrogateKs::triple_map`; `refit_profile`'s helpers and its `append` recipe (the collection less its last `APPEND_N` = 20, attached, then those pushed with `push_shared` and **no** refit).

- [ ] **Step 1: Write the measurement** `triple_cost` (reported, never asserted), printing one table: the fitted samples, the types, the fitted triples, and the wall ms of (a) `knowledge_map()`, (b) `triple_map()` cold (an empty cache), (c) `triple_map()` again with nothing new, (d) `triple_map()` after the 20 appended samples (pending-only), each the median of 5 calls (a fresh provider per cold call).

- [ ] **Step 2: Run it on a copy of the collection**

Copy `D:\Projects\Physics\models\collections\lsda_h0.15_s0.15_tol1e-6_kohn-sham-gpu_all-54eedc40ddc5` to the scratchpad (never write to `models`), then:
Run: `$env:COLLECTION_DIR = '<copy>'; cargo test -p providers --release --lib triple_cost -- --ignored --nocapture --test-threads 1`
Expected: PASS with the table printed.

- [ ] **Step 3: Record it.** Write the table, the machine and the load in `references.md`, against the budget (cold ≤ 100 ms, pending-only ≤ 2 ms), with the verdict either way. Nothing is tuned here: an over-budget cold build is throttled by the duty backstop, as the spec says.

- [ ] **Step 4: Commit**

```bash
git add crates/providers/src/surrogate/tests/triple_cost.rs
git commit -o -m "test(surrogate): measure the triple map's cold and pending-only build on the BEST collection" -- crates/providers/src/surrogate/tests/triple_cost.rs crates/providers/src/surrogate/tests.rs crates/providers/src/surrogate/tests/refit_profile.rs crates/validate/references.md
```

---

### Task 11: README, screenshots, the gates, the merge — **Opus**

**Files:**
- Modify: `README.md` (the Knowledge map row, line 203: the "Three-atom arrangements" section — what a cell counts, the classes, 2D and 3D, the picker, hover, live outlines, that green is not accuracy and body-4 is not shown; the `--screenshot --ui` list at line 154 gains the five keys)
- Create: `docs/images/knowledge_triples_2d.png`, `docs/images/knowledge_triples_3d.png`
- Modify: `DEVLOG.md` (at the merge)

- [ ] **Step 1: Take the screenshots** with the hidden flag (docs/dev-reference.md, "Build, run and the app's defaults"), one GPU job at a time, working directory `D:\Projects\Physics`, no other app open on the collection:
  - `target\release\app.exe --screenshot <worktree>\docs\images\knowledge_triples_2d.png --preset h2o --estimated --ui "wait=model;layout=one;tab=knowledge;triple=1-8-1;triples=2d;cell=1.9,1.9,108"`
  - `target\release\app.exe --screenshot <worktree>\docs\images\knowledge_triples_3d.png --preset h2o --estimated --ui "wait=model;layout=one;tab=knowledge;triple=1-8-1;triples=3d;rotate=35,20"`

  Expected: both exit 0; read each PNG and check the section, the slice or cloud and the card are visible and legible.

- [ ] **Step 2: Write the README passage** with the two images in the Knowledge row (or a two-image table below it, as the Activity images are).

- [ ] **Step 3: Run the gates, one at a time**

Run: `cargo test -p units -p qm -p sim -p surrogate --release` — Expected: PASS (~6.5 min).
Run: `cargo test -p providers --release` — Expected: PASS (~5.5 min).
Run: `cargo test -p app --release` — Expected: PASS (~10 min, the replay scripts included).

- [ ] **Step 4: Commit**

```bash
git add docs/images/knowledge_triples_2d.png docs/images/knowledge_triples_3d.png
git commit -o -m "docs: the three-atom coverage map in the README, with its 2D and 3D screenshots" -- README.md docs/images/knowledge_triples_2d.png docs/images/knowledge_triples_3d.png
```

- [ ] **Step 5: Review and merge.** A whole-branch review (`superpowers:requesting-code-review`), then merge `triple-map` to `main` under the standing approval (memory "Hands-off execution"; run the merge alone, never piped). At the merge, add the DEVLOG entry (timeline, durations, failures, wins) and commit it with the merge. Phase B stays on its own plan, after targeted teach.

---

## Spec questions settled in this plan

1. **Thresholds.** `TAUGHT`, `taught_count()` and `reads_taught()` move to `sim::knowledge` now, with targeted teach's exact names and signatures. `window_count` and the overlay's rewiring stay targeted teach's.
2. **`SCAN_CUTOFF_FRACTION`** stays in `providers` (targeted teach moves it later, its R4). The builder writes `reach_limit` = 5/6·r_cut into the `TripleMap`, so `sim` holds no copy.
3. **The reach's pair ranges travel with the triple map** (`TripleMap::spans`, the knowledge map's own taught ranges from the same cached fitted pass). The two maps ship on different schedules, so reach never mixes fits. `reach(map, t, cell)` takes the `TripleMap` alone; B1 pins the spans to the knowledge map's.
4. **`TripleKnowledge::counts` is `Arc<[u32]>`**, as §6.4's cache paragraph needs; its field list said `Vec<u32>`. The basis fields are a `TripleBasis` and include `body4`, which band line 2 needs.
5. **The cell card has no per-cell sample count.** §6.5's "342 arrangements in 120 samples" would need a second dense array with per-sample deduplication; the card reads "342 arrangements", and the head keeps the type's sample count.
6. **The section is a fold, closed by default**, kept in `knowledge_sections` as `triples`. Its visibility (the Knowledge tab shown and the fold open) drives `SetTripleView`.
7. **Type names.** Stored with a ≤ b by Z, and r₁ is always to a. Shown with `pair_name`'s "H last" rule, so centre H with H and O reads `O–H–H` as in the spec. The axis labels name their pairs. The `--ui` key accepts either neighbour order.
8. **Numbers** use the app's `words::thousands` ("2,140"), not the spec's "2 140".
9. **No drag selection in Phase A** (it only feeds Teach); a click picks a cell. The 3D hit test also picks a missing cell's square, so a missing cell can be read in 3D. The picked cell is kept as its bins (k₁, k₂, q), not a flat index: the bins are 0.3 Bohr at every r_cut, so a pick survives a reach change.
10. **The scene instance** for the slice default and the `now` marker is the selected atom's least-stretched instance, else the lowest indices. §6.6's full resolution (the Edit group, nearest to a selection) belongs to Teach in Phase B.
11. **The triple map's schedule** uses its own key, the `KnowledgeKey` without the verdicts and the trust flag, through the same `knowledge_build` rule. Its build ms is added to the knowledge build's when both run in one publish.
12. **Entry links** made before the map arrives set `want_centre`, which resolves to that centre's type with the most gaps.
13. **`references.md` has no knowledge-map build time to stand beside.** The measurement records the knowledge map's time too.
14. **M1's "one of them held"** is realised in `sim` as a sample whose centre does not train. The fitted/held split itself is the provider's, which B1 tests with real quarantine codes.

## Deferred to Phase B (after targeted teach)

Teach this cell and the 2D drag-select region (§6.5's drag, §6.6), `CellRequest`, `plan_cells`, the `RangePlan` → `TargetRuns` refactor, `PassJob::Cells`, the estimate and arming, band items 4–7, `Action::TeachCells` and its script word, the `region=` key, and the tests U2, U5–U7, C1–C9, B3–B7 and V-T1 with their gates (`validate --test surrogate`, `--test dynamics`). Also the "Cell scans" paragraph of the `sampling` module doc.
