# Three-atom coverage map: where the model's three-atom arrangements are taught, and Teach a cell (2026-11-17)

Status: approved by the user 2026-11-17; plan after targeted teach. Branch `triple-map` off `main` 5613bdc.
No product code. Lands **after targeted teach** (§11). Atomic units throughout (Bohr, Hartree); angles are shown in
degrees and binned in cos θ (§5.2).

## 1. Intent, and why

The Knowledge panel's pair card shows a log histogram of one species pair's distances. That is a 1-D shadow of what
the model learns from. Its body-3 terms learn from **arrangements of three atoms**: a centre, two of its neighbours,
the two centre–neighbour distances and the angle between them. On a collection basis, its body-4 terms also learn
from four-atom arrangements. The card already says that the shadow is not enough: "Still missing: three-or-more-atom
arrangements around H, with H and O, 276 of 322 samples". Nothing shows *which* arrangements those are.

The user wants:
- a **coverage map of three-atom arrangements** (triples such as H–O–H around O, or O–H–H around H), with the two
  centre–neighbour distances and the angle as its axes;
- the map in **2D by default, with a switch for a 3D view**;
- **Teach this cell**, and a drag-selected region in 2D, which runs a targeted scan that places atoms to cover the
  selected cells. This extends targeted teach from pair-distance ranges to triple cells.

## 2. What the code does today (main 5613bdc, plus the branches this builds on)

- **The knowledge map** (`crates/sim/src/knowledge.rs`, built by `crates/providers/src/surrogate/knowledge.rs:37`)
  counts over the **fitted** samples only: the accumulated prefix, without excluded samples or the misfit set
  (`providers/.../knowledge.rs:42-48`). It holds:
  - per pair, a histogram of every atom pair closer than r_cut in bins of `BIN` = 0.1 Bohr (`sim/.../knowledge.rs:79`),
    counted when at least one of the two atoms trains (`trains_atom`, `providers/.../knowledge.rs:61`);
  - per element, neighbour counts and a **first-shell** angle histogram: 36 bins of 5° in θ (`ANGLE_BIN_DEG`,
    `ANGLE_BINS`, `sim/.../knowledge.rs:82-85`). It counts only the neighbours inside the pair's measured first shell
    (`first_shell`, `:497`), with the centre training (`providers/.../knowledge.rs:95-115`). It has no distances, so
    it cannot say *which* arrangement is missing.
- **Taught / thin** for a pair bin (`app/src/ui/knowledge.rs:56`, `:108-116`, `:178`): n is the count of bin k plus
  its two neighbours. The bin reads taught when 1 − e^(−n/`COVERAGE_COUNT`) ≥ `TAUGHT` = 0.9 (`COVERAGE_COUNT` = 4, so
  n ≥ 10), thin for 1 ≤ n < 10, and never for n = 0. Targeted teach moves `TAUGHT` into `sim::knowledge` and derives
  N_T = ⌈−COVERAGE_COUNT·ln(1 − TAUGHT)⌉ = 10 (`taught_count()`, `reads_taught`, `window_count`; its plan ee44aab,
  Global Constraints). These counts are display rules. They are not the trust verdict.
- **"three-or-more-atom arrangements … 276 of 322 samples"** (`pair_sentence`, `app/src/ui/knowledge.rs:989-1011`)
  is **not** a geometric count. It is the per-key verdict's sample floor:
  - an environment key is (class pair/many-body, centre Z, the sorted set of its neighbours' species within r_cut)
    (`surrogate::EnvKey`, `surrogate/src/history.rs:197`);
  - the class comes from the atom's neighbour count within r_cut: 1 is pair, ≥ 2 is many-body
    (`providers/src/surrogate/trust.rs:239`);
  - n_k counts the fitted samples holding an atom of exactly key k, against q_k, the basis columns its atoms touch
    (`calibrate.rs:20-50`, `trust::key_column_counts`, `:276`);
  - `ClassKey::Samples (n, q)` is the "276 of 322".

  The new map is a different measure, and the card says so (§6.7).
- **What body-3 encodes** (`crates/surrogate/src/descriptor.rs:56-75`, `basis.rs:16-24`). For centre i it is a sum over
  pairs {j, k} of **all** its neighbours within r_cut (not only the first shell):

    B_{i,(c₁,c₂,l)} = Σ_{j∈t₁} Σ_{k∈t₂} R_{n₁}f_c(r_ij) R_{n₂}f_c(r_ik) (r_ij r_ik / r_cut²)^l P_l(cos θ_jik).

  - f_c = ½(1 + cos πr/r_cut) is zero at r_cut (6 Bohr by default, `Basis::new`, `descriptor.rs:400-412`).
  - R_n = T_n(1 − 2r/r_cut) are Chebyshev polynomials.
  - `l ≤ l_max` = 3, and the total-degree cap is n₁ + n₂ + l ≤ D = `degree_cap` (4 by default; 6 is the recovery
    setting).
  - The terms with j = k are body-2-like functions, not arrangements.
  - The `body3` flag can switch the block off, and `body4` is on for a collection basis only.
- **The live graph** (branch `live-knowledge`: spec `2026-11-16-live-knowledge-graph-design.md`, commits d4dae8a,
  cff891d, 3bfe872). Samples kept since the last refit are counted per bin apart from the fitted counts.
  - One predicate decides it: `SurrogateKs::is_pending(i)`, true when the sample is past the fitted prefix, has no
    quarantine code, is not box-affected, and has one Z per position (cff891d).
  - They are drawn as an outline, and they never change a class.
  - The worker rebuilds on `KnowledgeKey`. A pending-only change waits for `views_interval` of the last build
    (2 s floor, 50× duty backstop; `app/src/worker/activity.rs` on that branch, `knowledge_build`). Any other change
    rebuilds at once.
- **Targeted teach** (branch `targeted-teach`: spec `2026-11-16-targeted-teach-design.md`, plan ee44aab) supplies:
  - the request path `Action::TeachRange` → `DynCommand::StartRange` → `Trainable::start_range`;
  - a pure planner in `sim` (`plan_range`) with slots, a greedy fill to green, feasibility skips (cell guard, cores,
    forgotten), and legs that keep warm starts;
  - a pass kind (`Sampler::range`, `PassJob::Range`) with no checkpoint and its own ETA, arming, honest band, keys
    and replay.

  Its band already names this feature as a non-goal: "It does not cover arrangements of three or more atoms at these
  distances" (its §4 item 1, "a later 'place a third atom at these distances' mode").

## 3. Goals, non-goals, anti-goals

**Goals.**
1. A per-triple-type coverage map of fitted three-atom arrangements, over (r₁, r₂, θ), with the pair graph's classes:
   taught, thin, missing within reach, and new (waiting for refit). The scene's current arrangements are markers.
2. 2D (default) and 3D views of the same cells, switched by a display toggle.
3. A triple picker listing only the triple types the fitted samples hold, ordered by gaps.
4. Teach this cell (both views) and Teach a drag-selected region (2D), which fill each selected cell to green by moving
   atoms of the open scene. This reuses targeted teach's request, planner shape, pass kind and wording.
5. Honest text: the map is geometric coverage of triples by fitted samples. It says nothing about the model's accuracy
   there (rule 7).

**Non-goals.**
- **Body-4** (four-atom arrangements). Three neighbours and a centre have six internal coordinates, which no 2D or 3D
  map shows without projecting away what matters. Body-4 is on only for collection bases, at degree 2–3
  (`descriptor.rs`, "Body-4 features"), which is smoother still. The card says the model also learns from four-atom
  arrangements and that this map does not show them (§6.7).
- Arrangements beyond r_cut. Body-3 does not see them, and long reach's beyond-reach band is pair-only.
- A triple-aware 3D-overlay tint (`sim::atom_coverage` is unchanged).
- Fill gaps for a whole triple map, which would be hundreds to thousands of solves, or for one slice (§11.3: a
  possible later addition).
- Adding atoms to the scene to make a triple it lacks (§11.2).

**Anti-goals.**
- No per-element or per-pair bins, ranges or bond lengths (rule 1). Every range comes from the grid constants, r_cut,
  or the map's own measured pair histograms.
- No change to training, fitting, trust, fences or the stored samples from the map itself. The only thing that solves
  is Teach, which is a pass.
- Green never means "accurate" or "trusted".
- No new GPU pipeline.

## 4. Constraints

- **Rule 5.** `sim` gets plain data (the triple grid, counts, reach and the planner's geometry). `providers` alone reads
  the store, `is_pending`, the quarantine codes and the forgotten set. The app draws. `sim` holds no physics, only
  geometry and counts.
- **Rule 3.** Bohr internally. Degrees only at the display boundary. The cos θ bin edges are shown as degrees.
- **Recording.** Teach reaches the worker, so it is a `ui::Action` (recorded and replayable). The 2D/3D switch, the
  picker, the slice, the rotation and the selection are `KnowledgeView` / `UiState` display state and are not
  recorded. Building the map only while it is shown uses a display-visibility `Command`, as the training views do
  (`Command::SetActivityViews`, `ui/mod.rs:1446-1450`). It changes nothing a replay must reproduce.
- **egui ids.** New widgets go through `ui::keyed` where ids could shift. Probes are named `knowledge-triple-*`.
- **Gentle GPU.** Teach solves are a pass, under the pass's rules: time limit, arming past `PASS_TARGET_S`, cancel and
  pause.

## 5. Approaches considered

**A. Binned triple counts per triple type, drawn in egui as a 2D heat-map slice and a projected voxel cloud
(chosen).**
- The counts use the same fitted/pending split as the pair histograms, and the same N_T. The cells can be taught.
- It is cheap: O(Σ_i k_i²) per sample, cached per fit.
- egui draws ≤ 10⁴ cells comfortably.

**B. Descriptor-space coverage**: the leverage, or the distance to the nearest fitted sample in body-3 feature space,
the fence's own `max_descriptor_distance` metric. It is closer to "what the model knows", but:
- it has no geometric axes to draw or teach;
- it moves with every refit's standardisation;
- it duplicates what the verdict and the fences already judge, and would read as a trust measure.

Rejected for this map. Rule 7 is better served by saying plainly that the map is geometry.

**C. Raw triples as a point cloud, unbinned.** Rejected for the reason the live graph gave for raw ticks: there are no
classes and no cell to teach, and tens of thousands of points hide the gaps.

**D. A wgpu volume pipeline for 3D** (the `render` crate's path). Rejected. The cell count is ≤ 10⁴, which the egui
painter tessellates in well under a millisecond. A paint callback would also have to survive docking and detached
windows. Revisit only if a measured frame cost says so.

## 6. Design

### 6.1 What is counted

- **A triple type** t = (a, c, b), where c is the centre species and a ≤ b are the two neighbour species, written
  `a–c–b` (centre in the middle: `H–O–H`, `O–H–H`, `H–H–H`). It is the body-3 term's (centre, t₁, t₂).
- **A triple** is (i, {j, k}), counted in a fitted sample when:
  - centre i trains (`trains_atom`: a capped segment's caps and edge atoms add nothing, as for the element card's
    angles);
  - j ≠ k are both neighbours of i within r_cut (0 < r_ij, r_ik < r_cut);
  - their species are {a, b}.

  The j–k distance may be anything, including beyond r_cut: body-3 sums every neighbour pair of the centre. Each
  unordered {j, k} is counted once. The j = k terms are not arrangements and are not counted.
- **The neighbour set is every neighbour within r_cut**, as body-3 sums. It is not the element card's first shell.
  That card stays as it is (a chemistry-like bond-angle view, documented as first-shell). The two differ on purpose
  (the user's decision, §11.1).
- **Axes.**
  - For a ≠ b: r₁ = r_ij to the species-a neighbour and r₂ = r_ik to the species-b one.
  - For a = b the triple is symmetric and is **folded**: r₁ = min, r₂ = max, and only cells with k₁ ≤ k₂ are stored.
  - u = cos θ_jik.
- **Fitted versus pending.** Fitted triples come from the fit that stands, under exactly the knowledge map's sample rule
  (`providers/.../knowledge.rs:42-48`). Pending triples come from the samples `is_pending` admits. They are counted
  apart, enter no class, and never move a cell's colour (the live graph's rule).
- **Per type** the map also records:
  - the fitted samples holding at least one triple of the type;
  - the fitted triple total;
  - the pending samples;
  - the top three stored compositions holding it (`Composition`, by fitted count), for the tip when the open scene
    lacks the type.
- **One definition.** A pure `sim::knowledge::each_triple(z, pos, r_cut, trains, f)` enumerates triples. The builder,
  the planner's projection and the UI's scene markers all use it, so a marker and the cell it sits in cannot disagree.

### 6.2 Cells: ranges, resolution, the angular variable

- **Distance bins** are `TRIPLE_BIN` = 3·`BIN` = **0.3 Bohr**, from 0, with n_r = ⌈r_cut/0.3⌉ bins (20 at r_cut 6, 27
  at 8, 34 at 10). Only r < r_cut is counted, so the last bin may be partial.
  - 0.3 Bohr is the pair rule's window: a pair bin and its two neighbours. So "10 like it" means the same distance
    tolerance on both maps, along each distance.
  - This is a display resolution. It is **finer** than what body-3 resolves at D = 4. In one distance its radial
    polynomial degree is at most D, so its structure is on the scale of the Chebyshev node spacing,
    ≈ r_cut·π/(2D) ≈ 2.4 Bohr mid-range at D = 4 (an estimate from the node spacing, not a measurement; finer near
    r = 0 and r_cut).
  - Finer than the descriptor errs toward showing gaps that the model may interpolate across. That is the
    conservative side, and the band says it (§6.7).
- **The angular variable is cos θ**, in `COS_BINS` = **10** bins of Δu = 0.2. Bin q = min(⌊(1 − u)/0.2⌋, 9), so θ
  increases left to right, with edges at 0, 37, 53, 66, 78, 90, 102, 114, 127, 143 and 180°.
  - Why cos θ: body-3 depends on the angle only through P_l(cos θ), l ≤ l_max = 3, a cubic polynomial in u. Equal
    widths in u are therefore equal spans of what the descriptor sees.
  - Near linear arrangements, equal θ bins would split arrangements the descriptor cannot tell apart:
    cos 170° = −0.985 against cos 180° = −1, and dB/dθ = −sin θ · dB/du → 0.
  - Isotropic arrangements are also uniform in u.
  - 10 bins resolve a cubic's structure about three times over. The count is a display choice, not derived.
  - Bins and their edges fall where they fall, never placed to suit any molecule (rule 1). An equilibrium cloud lying
    on an edge splits across two cells, both green when sampled.
- **Size.** Dense, per type present: 2 × n_r² × 10 u32 (fitted and pending) = 32 KB at r_cut 6 and 92 KB at r_cut 10.
  A four-species collection has at most 4·10 = 40 types: ≤ 1.3 MB at reach 6 and ≤ 3.7 MB at reach 10. Only the types
  the fitted or pending samples hold are stored. A dense layout beats a sparse map here: it is smaller in the worst
  case and simpler to slice.

### 6.3 Classes, the reach, and the thresholds

- **Taught**: the cell's fitted count n satisfies `reads_taught(n)`, the pair rule's 1 − e^(−n/4) ≥ 0.9, so
  n ≥ N_T = `taught_count()` = **10**.
- **Thin**: 1 ≤ n < N_T.
- **The window is the cell itself.** The pair rule's 3-bin window is 0.3 Bohr wide, and the cell already is that wide
  in each distance. A further ±1-cell window would be 27 cells, 0.9 × 0.9 Bohr × 0.6 in u. It would let one dense
  cell paint 26 empty neighbours green, which is exactly what targeted teach's "a zero-count bin is a gap" decision
  refused.
  - With the cell as its own window, "covered" = taught. The pair planner's extra "count ≥ 1" holds automatically.
  - The same N_T, `reads_taught` and `taught_count` are reused. There is no new constant.
- **Missing within reach** (grey outline): n = 0, and the cell lies in the type's **reach**. The reach is derived from
  the map's own fitted pair histograms; no table, no constant beyond r_cut and `SCAN_CUTOFF_FRACTION`. It reads a
  span for every pair met with a training atom (each pair the fitted samples hold within r_cut where at least one of
  the two atoms trains), a superset of the pair graph's rows, equal on those: so a neighbour species that only ever
  appears capped can be in reach without having a pair-graph row. A cell (r₁, r₂, u), judged at its centre, is in
  reach when:
  1. r₁ lies in the c–a pair's taught range (`taught_range`: from the first filled bin to the last) and
     r₁ ≤ 5/6·r_cut;
  2. r₂ lies in the c–b pair's taught range and r₂ ≤ 5/6·r_cut;
  3. the third side r_ab = √(r₁² + r₂² − 2r₁r₂u) is at least the a–b pair's shortest fitted distance (the lower edge of
     its first filled bin). When the a–b pair has no fitted distance within r_cut, the bound is r_ab ≥ r_cut: a and b
     were never taught closer than that.

  Reach means: the 1-D shadows say every edge length here was taught, but this arrangement never was. That is exactly
  the gap the pair graph cannot show. Teach only places points where the pass would (≤ 5/6·r_cut).
- Cells outside the reach with n = 0 are not drawn. Cells outside it with n > 0 are drawn in their class: fitted data
  is always shown.
- **New, waiting for refit** (dashed outline): the cell's pending count is > 0. A dashed outline over the fill, or in
  place of the grey outline for a missing cell. It never changes the class. "+k new" appears in the cell card and the
  picker row, as on the pair card.
- **Gaps of a type** (the picker's order): the cells in reach that are not taught, missing or thin.
- **Teachable**: Teach accepts only cells in reach. A cell is out of reach because a pair distance or the a–b contact
  was never taught, and the band says so: "outside what the pair graphs have taught — teach the pair first" (§6.7).
  The pair graph's Teach is the tool for that. This keeps triple teach from silently becoming an untargeted pair scan.

### 6.4 Data, builder, cost, cadence

- **`sim::knowledge` (new).**
  - `TripleType { centre, a, b }`;
  - `TripleGrid { r_bin, n_r, cos_bins }` and `TripleGrid::cell(r₁, r₂, u) -> Option<usize>`, with the fold;
  - `TripleKnowledge { t, samples, triples, counts: Vec<u32>, pending: Vec<u32>, pending_samples,
    compositions: Vec<(Composition, usize)> }`;
  - `TripleMap { grid, r_cut, body3: bool, degree_cap, l_max, types: Vec<TripleKnowledge> }`;
  - pure helpers `each_triple`, `cell_class(map, triples, t, cell) -> CellClass { Taught, Thin, Missing, OutOfReach }`,
    `reach(map, t, cell)` and `gaps(map, triples, t)`.

  Basis fields are carried so the card can say "this model's basis has no three-atom terms" when `body3` is false.
  The `KnowledgeMap` stays as it is: the triple map is a separate value.
- **`providers` (new).** `SurrogateKs::triple_map(&self, fitted: Option<&TripleCountsCache>) -> Option<TripleMap>`:
  - the fitted counts come from one pass over the knowledge map's fitted samples, with the neighbour lists within
    r_cut, then `each_triple`;
  - the pending counts come from one pass over the samples `is_pending` admits.

  The fitted block is cached in the provider against the fitted set's identity: the `KnowledgeKey` fields other than
  pending and verdicts (generation, forgotten, quarantined, superseded, box-excluded). A pending-only rebuild therefore
  recounts only the pending tail and clones the fitted `Arc<[u32]>` blocks. A verdict re-judge changes no count and
  does not rebuild the triple map.
- **Shipped** as `ActivityDelta::triples: Option<Arc<TripleMap>>`, beside `knowledge`, where `None` means no new
  build.
  - It is built only while the arrangements are shown: the Knowledge panel is shown and the section is open or a
    picked type's card (`Pick::Triple`) is, so a picked card stays live with the fold closed.
    `Command::SetTripleView(bool)` is sent on a change, as `SetActivityViews` is (`ui::knowledge::triples::wanted`,
    `ui/mod.rs`).
  - It follows the knowledge map's schedule: `knowledge_build` on the `KnowledgeKey` without the verdicts and trust
    (`triple_part`), at once on a fit change, and at most once per `views_interval` on a pending-only change, with
    the deferred wake. A pending-only rebuild that would wait builds anyway beside a knowledge build in the same
    publish, so the two cadences fall back into step.
  - When both maps build in one publish, its wall time adds to the knowledge build's ms, so the duty backstop covers
    both; a triple build alone records its own ms.
  - Turning the view on builds once on the next publish.
- **Cost.**
  - Per sample, Σ over training centres of C(k_i, 2), where k_i is the neighbours within r_cut, and each triple costs
    one dot product and one bin.
  - Water is 3 triples per sample. A 20-atom cluster fully inside r_cut is 20·C(19, 2) = 3 420.
  - 2 000 such clusters (≈ 7 M triples) are an estimated ≈ 0.1 s cold. The user's H/O collections of small molecules
    are ~ms.
  - The pending tail during a pass is tens of samples: ≪ 1 ms.
  - The plan measures the user's BEST collection cold and pending-only and records both in `references.md` beside
    the knowledge map's.
  - Budget: a cold build ≤ 100 ms at that collection, and a pending-only rebuild ≤ 2 ms. If the cold build exceeds its
    budget, the duty backstop already throttles it. The plan records the number either way and does not tune anything
    away.
- **UI cost.** A slice is ≤ n_r × 10 = 340 cells. The 3D cloud has ≤ n_r² × 10 cells, at most a few thousand drawn,
  since reach and the fold trim most. Reach and classes are computed once per (map `Arc` pointer, type) and cached in
  `KnowledgeView`. Projection and depth sort are recomputed per frame only while rotating; that is ≈ 10⁴ operations.

### 6.5 The panel

- **Where.**
  - A section of the Knowledge panel, **"Three-atom arrangements"**, below the picked card. It holds the picker, the
    view and the cell card.
  - A new `Pick::Triple(TripleType)` shows the triple card in the detail area.
  - Entry points: the picker; the element card's new "Arrangements around O" line (one link per type centred on that
    element); and the pair card's "Still missing: three-or-more-atom arrangements around H …" sentence, which gains a
    "Show arrangements around H" link to the type centred on H with the most gaps.
- **The picker.** It lists only the types the fitted or pending samples hold, ordered by gaps (descending), then by
  fitted triples (ascending), then by name. A row reads `H–O–H · 14 gaps · 2 140 arrangements in 610 samples · +12 new`.
  A dot marks types the open scene has. With nothing picked, the default is the type the open scene has with the most
  gaps, else the first row.
- **The card head.**
  - The type, its sentence and the verdict link. Sentence: "2 140 arrangements of H–O–H in 610 samples (13 cells
    taught, 9 thin, 14 missing within reach)."
  - The verdict link: the Model status's per-key verdicts for keys with centre c whose neighbours include a and b
    (`keys_of`'s pattern). For example, "the model's many-body key around O, with H: trusted", or "… around H, with
    H and O: 276 of 322 samples — the per-environment sample floor, not this map".
- **The 2D view (default).**
  - x is the angle: 10 columns uniform in cos θ, labelled in degrees at the bin edges, with the caption
    "angle at O (cos θ bins)". y is r₁.
  - A slice control for r₂: a slider over the r₂ bins plus the label "r₂ (O–H) 1.8–2.1 Bohr". Its default is the
    chosen scene instance's r₂ bin, else the type's most-filled r₂ bin.
  - For a folded type, the slice at r₂ bin s shows every r₁ and reads the folded cell {k₁, s}, so no half of the
    slice is blank. Diagonal cells are shared.
  - The rows are cropped to the union of reach and filled bins.
  - The fill colour is the class (`GREEN` / `AMBER`), with opacity rising with log n so density shows. Missing cells
    have a grey outline (`theme::INK_FAINT`). New cells have a dashed outline.
  - Markers: a white ring at each scene instance whose r₂ falls in the slice, labelled "now" as on the pair graph,
    and a faint ring for instances in other slices, noted in the hover.
  - Hover reads "r₁ 1.8–2.1 · r₂ 1.8–2.1 Bohr · θ 102–114° · n 342 (+12 new) · taught".
  - A click picks a cell (the cell card). A drag selects a rectangle of cells on the slice, snapped to cells.
    Shift-drag adds to the selection. A click without a drag, or ×, clears it. This is targeted teach's interaction,
    and the plot takes the drag from the ScrollArea in the same way.
- **The 3D view** (switch: `2D | 3D`, `UiState`, not recorded).
  - Axes r₁, r₂ and cos θ, normalised to a cube over the cropped extent, in an orthographic projection.
  - Drag rotates (yaw and pitch), the wheel zooms, and a double-click or the Reset control returns to the default
    angle.
  - Each filled cell is a projected cube at 0.8 of the cell size. Its ≤ 3 visible faces are convex polygons, shaded
    ×1 / 0.85 / 0.7, and they are drawn back to front (painter's sort by depth).
  - Missing cells in reach are small grey outlined squares at their centres; a "missing" checkbox hides them, on by
    default. New cells get a dashed square.
  - Scene instances are white cross markers.
  - A click hit-tests front to back on each cube's projected hull and picks the front-most cell, which opens the cell
    card with its count and **Teach this cell**: the mockup the user saw.
  - The cloud is drawn with `egui::Painter` shapes only (§5 D).
- **The cell card** reads, for example: "H–O–H · r₁ 1.8–2.1, r₂ 1.8–2.1 Bohr, θ 102–114° · 342 arrangements ·
  taught · +12 new" — no per-cell sample count (that would need a second dense array with per-sample deduplication); the
  section's head keeps the type's sample count (plan, "Spec questions settled" 5). It then shows Teach this cell, or "Teach 6 cells …" when a 2D selection exists, and the
  band (§6.7).

### 6.6 Teach: what is reused from targeted teach, and what is new

**Reused unchanged:**
- the request → command → trait path;
- the scene-changed refusal;
- the pass's references, its warm-start rule (the first point of a run is fresh), its time limit, cancel (refit),
  pause, collection append and `held_by_collection` skip, the box's `KeptBoxAffected`, and the end refit;
- no checkpoint and no resume: pressing Teach again replans against the new map;
- `sample_button`'s ETA rate and two-click arming past `PASS_TARGET_S`;
- the feasibility tests: the cell guard, core overlap against every atom with r_core from `EvalStats::pseudised`, and
  `refuse` (points near forgotten samples, codes 7 and 8, via `near_forgotten_cached`);
- the decision **fill to green**, and the decision **skip forgotten ranges**.

The decision **take zero-sample bins** is automatic here: with the cell as its own window, a zero-count cell is never
green.

**New:**
- **The request.**
  `sim::sampling::CellRequest { t: TripleType, i: usize, j: usize, k: usize, cells: Vec<(f64, f64, f64)> }`:
  - i is the centre, j the species-a neighbour, k the species-b neighbour;
  - each cell is given by its centre (r₁ Bohr, r₂ Bohr, cos θ), so a replay line is readable, and the worker re-bins
    it with its own grid.

  It travels as `Action::TeachCells { req, long }` → `DynCommand::StartCells(req)` →
  `Trainable::start_cells(scene, &req, refs)`. The default implementation refuses. `CellRequest::check` refuses
  changed species or indices with targeted teach's `the scene changed — select the range again` (here "… select the
  cells again").
- **The triple in the scene**, resolved by the UI and named on the card ("atoms 0 (O), 1 and 2 (H)"):
  1. the Edit group (`UiState::edit_group`) when it holds exactly three atoms forming the type. The centre is the
     selected atom when it qualifies, else the lowest index whose two others are both within r_cut;
  2. else the selected atom, when it is of species c, as the centre with its nearest a and nearest b neighbours
     (distinct atoms);
  3. else the scene instance nearest the selection, in cell steps (ties to the lowest indices).

  With no instance, Teach is disabled. The tip names the stored compositions that hold the type:
  "the open scene has no H–O–H within the cutoff — H₂O (610 samples) has it" (§11.2).
- **The planner.** `sim::sampling::plan_cells(scene, &req, triples: &TripleKnowledge, grid, r_cut, r_core, refuse) ->
  CellPlan`. It is pure and deterministic, has no physics, and is shared by the UI's estimate and the provider's start,
  as `plan_range` is.
  1. **Clamp.** Cells outside reach, or with a centre distance > 5/6·r_cut, are dropped, with a note. There is no
     shell check: a shell check measures one pair's band, and a triple has no band of its own.
  2. **Slots.** Each cell has 3 × 3 × 3 = 27 slots at offsets (s + ½)/3 of the cell along r₁, r₂ and u. For a folded
     cell on the diagonal, only the 18 slots with r₁ ≤ r₂ are used. The slots are ordered farthest-point first: each
     next slot maximises its minimum distance to those already chosen, in cell-normalised coordinates, with ties by
     index. The minimum spacing is 0.1 Bohr and 0.067 in u. That is coarser than targeted teach's 0.025 Bohr, so its
     leave-one-out argument (no denser than a burst) holds a fortiori.
  3. **Geometry of a slot** (one rule, the pass's "lighter moves" extended to two atoms; the rest of the scene is
     fixed). The centre i stays put. Of j and k, the heavier slides and the lighter swings (isotope mass; on a tie the
     higher index swings). Call the slider s and the swinger w.
     - s is placed at R_i + r_s·û_is, a radial move along its current axis: the pair scan's move.
     - w is placed at R_i + r_w·(u·û_is + √(1 − u²)·ŵ), where ŵ is the unit vector ⟂ û_is in the plane (i, s, w) on
       w's current side. When w lies on the i–s line, ŵ is a fixed deterministic perpendicular.
     - For a folded type, whichever of j and k is currently nearer i takes r₁ (the shorter), which keeps the motion
       least.
  4. **Feasibility.** A slot is refused, and counted by reason as a `Job::Skip`, when it fails the cell guard, the
     cores (each moved atom against every atom), or `refuse` (forgotten samples).

     **New:** it is also refused when its j–k distance falls in the shell band [5/6·r_cut, r_cut) (`shell_skipped`).
     The scan drives that distance, so R3-1's reason holds for it: no fitted scan point in the band. Distances to
     bystanders are not checked, as in the pair scan. The physics reviewer rules on this (§8).
  5. **Greedy fill to green.**
     - The projected counts c′ start at the map's **fitted** counts, as `plan_range` does.
     - Each planned geometry adds **all** of its triples of type t via `each_triple`: every centre of species c and
       every neighbour pair. That counts the collateral exactly; for example, both H centres of an O–H–H scan.
     - A cell is covered when `reads_taught(c′)`.
     - Each next point goes to the uncovered selected cell with the lowest c′ (ties to the cell nearest the scene
       instance, then to the index), at its next untried slot. No point goes to a covered cell or outside the
       selection.
     - A cell whose slots run out goes into `still_thin`.
     - The cost is about 10 solves per empty cell, against 3–4 per pair bin: one geometry adds one triple to a
       centre-i cell, and collateral rarely lands in the same cell. The estimate states this up front (§6.7).
  6. **Order and warm starts.** The points form a greedy nearest-neighbour path in cell-normalised (r₁, r₂, u) space,
     starting from the scene instance's own coordinates. A run breaks wherever a step exceeds one cell on any axis.
     Each run is one scan run, so its first point is fresh by the existing `k == 0` rule and gets `discontinuity`.
  7. **Tail rule, generalised.** A Stalled or Cap failure at a point stretched relative to the scene instance
     (r₁ > r₁⁰ or r₂ > r₂⁰) skips the rest of its cell and every later point at least as stretched on both distances
     (`tail_skipped`). A compressed failure skips nothing. It is one shared function with targeted teach's
     `skip_scan_tail`, extended. Whatever TimeLimit ruling `main` holds when this lands applies to both.
- **The pass.**
  - Targeted teach's `Sampler::range` takes a `RangePlan` built from distances. This feature lowers both plans to one
    shape, `TargetRuns` (runs of full geometries plus the skip counts and `still_thin`), and the sampler entry takes
    that.
  - This is a small refactor of targeted teach's entry, made in this plan's first task. Its S3 byte-identity fixture
    for the default pass must stay byte-identical.
  - New: `PassJob::Cells { t }`, in memory only. Samples stay `SampleTag::Scan` with history code 20, so there is **no
    format change**.
  - Activity words: `cell scan H–O–H 6 cells`. The stage reads "cell scan".
- **The estimate.**
  - "Teach this cell (≈ 10 solves, ~25 s)" or "Teach 6 cells (≈ 58 solves, ~2 min)", with "(≈ n solves)" when there
    is no rate yet.
  - Rule line: `fills each cell to green — 10 arrangements in the cell, about 10 solves per empty cell`.
  - Past `PASS_TARGET_S`: "Start anyway (≈ …)".
  - Disabled when every selected cell is already covered: `nothing to teach: every selected cell reads taught`.

### 6.7 Honest limits (the card's band, shown while a cell or selection exists)

1. `Green counts arrangements in the cell (10 within 0.3 Bohr in each distance and 0.2 in cos θ); it is not the
   model's trust or accuracy.`
2. `Counts every pair of neighbours within {r_cut} Bohr of the centre, as the model's three-atom terms do. Four-atom
   arrangements (body-4) are learned too and not shown.` Body-4 is named only when the basis has it. When `body3` is
   false: `this model's basis has no three-atom terms: it does not learn the angle`.
3. `The model's three-atom terms are smooth in the distances and in cos θ: it may interpolate across a missing cell,
   and a green cell can still be wrong.`
4. `Teaches {type} in this scene's surroundings (atom {i} stays, {s} slides, {w} swings; the rest is fixed).`
5. `Skipped: {k} points ({a} box margin, {b} cores overlap, {c} j–k in the cutoff's shell, {d} near forgotten
   samples); {t} cells stay thin.` Zero parts are dropped.
6. `Outside what the pair graphs have taught: {n} cells — teach the pair first.` This appears when a selection
   touches out-of-reach cells.
7. Box: targeted teach's `{k} of {n} box-affected — their cells may stay thin`.

### 6.8 Actions, display state, keys, replay

- **Recorded:** `Action::TeachCells` only. It is replayable. Its script line is
  `teach-cells {Za}-{Zc}-{Zb} {i} {j} {k} {r1}:{r2}:{cos}[,…]`, which parses to `long: false`. A script using it
  needs the estimated backend with a model, and pins quality, walls and ionic as every solving script does.
- **Not recorded (`KnowledgeView` / `UiState`):** the triple pick, the 2D/3D mode, the slice, the rotation and zoom,
  the "missing" checkbox and the cell selection. A new pick or a new scene clears the selection (`adopt`).
  `Command::SetTripleView` is derived from what is shown.
- **`--ui` keys** (parsed in `screenshot.rs` beside `knowledge`, and listed in `docs/dev-reference.md`):
  - `triple=<Za>-<Zc>-<Zb>` picks the type and opens the section;
  - `triples=2d|3d` sets the view;
  - `slice=<r₂ Bohr>`;
  - `cell=<r₁>,<r₂>,<deg>` picks a cell and opens its card;
  - `region=<r₁a>-<r₁b>,<θa>-<θb>` makes a 2D drag selection on the slice;
  - `rotate=<yaw>,<pitch>` (degrees).

  Example: `--preset h2o --estimated --sample --ui "tab=knowledge;triple=1-8-1;triples=3d;rotate=35,20"`.

## 7. Tests and gates

**Gates** (CLAUDE.md table, one heavy job at a time, with the gentle-GPU rules):
- `-p units -p qm -p sim -p surrogate --release` and `-p providers --release` (any change);
- `-p validate --release --test surrogate` (provider and policy: a pass kind);
- `--test dynamics` (`Dynamics` gains a command);
- `-p app --release` (viewer, worker, replay).

There are no oracles, because there is no solver change.

**New tests.**
- **sim, map (seconds).**
  - (M1) `each_triple` on a hand-built set: three bent H₂O (104.5°) and one linear, one of them held. It gives exact
    cells for H–O–H (folded) and O–H–H. j = k is never counted, j–k beyond r_cut is counted, and a non-training
    centre adds nothing.
  - (M2) The cos θ bins: 0° → 0, 90° on the edge → bin 5, 180° → 9, clamped.
  - (M3) The fold: (r₁, r₂) and (r₂, r₁) give the same cell. The slice of a folded type reads {k₁, s}.
  - (M4) `cell_class`: 9 is thin and 10 is taught, from `taught_count()` and never a literal. Pending never changes a
    class.
  - (M5) Reach: from hand-built pair histograms, all three conditions, including the a–b-never-seen case (r_ab ≥ r_cut).
- **sim, planner** (the P1–P8 analogs).
  - (C1) Every planned (r₁, r₂, u) lies in a selected cell.
  - (C2) No point goes to a covered cell, counting collateral (an O–H–H scan whose second H centre also lands a
    triple).
  - (C3) Spacing and the slot cap (27, or 18 on a folded diagonal).
  - (C4) An empty-map selection covers every cell, and a lone cell whose slots are refused lands in `still_thin`.
  - (C5) The clamp and reach drop, with no shell check.
  - (C6) Margin, cores, the j–k shell band and `refuse` become skips counted by reason.
  - (C7) The geometry: i is fixed, s is radial, w is in-plane on its own side, the collinear case is deterministic,
    and every other atom is bit-identical.
  - (C8) Runs break at > 1 cell steps.
  - (C9) The same inputs give the same plan.
- **providers.**
  - (B1) The builder against a brute-force count over a stub store: fitted only, pending apart (`is_pending`;
    forgotten and box-affected samples count nowhere), masked centres, and compositions.
  - (B2) A pending-only rebuild reuses the fitted block (pointer-equal `Arc`) and recounts the tail.
  - (B3) The sampling pass: exactly the planned geometries, `PassJob::Cells`, skip counts, and the generalised tail
    rule.
  - (B4) The S3 fixture is byte-identical.
  - (B5) `pass_checkpoint()` is `None`.
  - (B6) Collection append, and a bit-identical held geometry is skipped.
  - (B7) Forgotten samples (codes 7 and 8) are skipped, and Restore lifts the skip.
- **app.**
  - (U1) The slice model: the rows cropped to reach ∪ filled, the fold read, the markers in the right cell (via
    `each_triple`).
  - (U2) A drag maps to snapped cells, shift adds, and a click clears.
  - (U3) The 3D hit test picks the front-most cube at a fixed rotation, and back-to-front order is stable.
  - (U4) The picker's order by gaps, and only held types listed.
  - (U5) The Teach label, arming and the disabled reasons.
  - (U6) The `describe` → parse round trip, and the recorder writes `action teach-cells …`.
  - (U7) `apply` emits `StartCells` and is refused while a pass runs.
  - (U8) The `--ui` keys parse.
  - (U9) The display toggles emit no Action, and `SetTripleView` follows visibility (the `SetActivityViews` test's
    pattern).
- **Physics** (`validate --test surrogate`, ≈ 5 min, Fast): (V-T1) H₂O after a short pass, then a 2D region of H–O–H
  cells: r₁ 1.8–2.4, slice r₂ 1.8–2.1, θ 90–127°.
  - After the refit, every selected cell reads taught unless the band names it (box-affected, failed or skipped).
  - E along each run has no step beyond the teacher's tolerance-scaled noise.
  - The scene's other atoms are untouched in every stored sample.
- **Review.** The physics-reviewer reads §6.1–§6.3 (what is counted, the cos θ argument, reach) and §6.6 steps 3–5 and
  7 (the geometry, the j–k shell refusal, the tail rule) before the provider code lands.
- **Docs.**
  - The README's Knowledge row and controls.
  - `dev-reference.md`: probes `knowledge-triple-*`, the `--ui` keys, and the script word.
  - A "Cell scans" paragraph in the `sampling` module doc.
  - The `sim::knowledge` module doc, "Three-atom arrangements".
  - The measured build cost in `references.md`.
  - DEVLOG at the merge.

## 8. Order

**Phasing (user, 2026-11-17).** The map is built in two phases, and the order below applies to Phase B only.
- **Phase A is built now, ahead of targeted teach.** It holds everything that does not need targeted teach: the triple
  cell counts (§6.1–§6.4), the 2D map with the 3D switch, the picker ordered by gaps, cell hover and the readouts, the
  live update during training, the docs and the README. It has no Teach controls at all (absent, not disabled), and
  its band holds only items 1–3 of §6.7. Its plan is `docs/superpowers/plans/2026-11-17-triple-coverage-map-phase-a.md`.
- **Phase B (Teach) comes after targeted teach**: Teach this cell, the 2D drag-select region and its Teach, and the
  `RangePlan` → `TargetRuns` refactor with the cell scan (§6.6, §6.7 items 4–7, §6.8's `teach-cells` and `region=`).

- **After targeted teach**, which itself follows smearing and long-reach phase 1. This feature extends its request,
  planner, pass kind, band, keys and replay, and refactors its sampler entry (§6.6). Building first would mean
  inventing all of that twice.
- **After live-knowledge merges.** `is_pending`, the pending split and the `KnowledgeKey` schedule are on that branch.
- **Independent of charged collections.** Cell samples take the pass's append path, as range samples do.
- **Broken symmetry** affects stretched cells exactly as it affects stretched pair ranges. Targeted teach's band item
  5 (the spin-symmetric stretched bond) applies to a cell scan whose r₁ or r₂ exceeds 2× the scene's. It is reused,
  and broken symmetry's plan deletes it from both.

## 9. Decisions the user made (2026-11-16/17)

- The map is "a coverage map for three-atom arrangements … Its axes are the two centre–neighbour distances and the
  angle."
- "2D is the default. A switch turns on a 3D view": "2D: a heat-map of angle against one distance, with a slider or
  selector for the other distance. Symmetric triples may fold r₁ ↔ r₂. 3D: a rotatable voxel cloud with all three
  axes."
- Cells are coloured like the pair graph: "taught (green), thin (amber), missing within reach (grey outline), and 'new,
  waiting for refit' (dashed outline; the same live pending rule as the live knowledge graph). The scene's current
  geometry is a marker."
- "A triple picker lists only the triples the collection has, ordered by gaps."
- "'Teach this cell' (and a drag-selected region in 2D) runs a targeted scan that places atoms to cover the selected
  cells", extending targeted teach. That design's decisions carry over: "fill to green", take zero-sample bins, and
  skip forgotten ranges.
- The mockup: "8×8×8-ish cells, drag to rotate, click a cell for its count plus a 'Teach this cell' button."

Decided here, within the user's brief:
- cos θ as the angular variable;
- 0.3 Bohr × 0.2 cells, with the cell as its own window and N_T = 10;
- reach derived from the pair histograms, and teach limited to reach;
- body-4 out of scope;
- the lighter neighbour swings and the heavier slides;
- the j–k shell-band refusal;
- built only while shown, with the fitted block cached per fit.

## 10. Left to the builder

- The exact styling: the opacity ramp, the dash pattern, the cube shading and the marker glyphs (follow `theme`).
- The slider widget's form.
- Where the "Three-atom arrangements" section sits in a wide two-column panel.
- The exact `TargetRuns` field names, and whether the lowering lives in `sim` or `providers`.
- The fitted-block cache's container.
- The farthest-point slot order's tie detail, provided the order is deterministic (C9) and pinned by a test.
- Whether the 3D default rotation is fixed or remembered per type in `KnowledgeView`.

## 11. Decisions recorded (the user, 2026-11-17; formerly the open questions)

1. **A triple is every pair of neighbours within r_cut**, exactly what the model's three-atom terms sum (§6.1). It is
   not the first shell. The element card's first-shell bond angles stay as they are. The reach keeps the drawn gaps to
   distances the pair graphs have taught (§6.3).
2. **When the open scene lacks the picked triple, Teach is disabled.** The tip names the stored molecules that have it
   (`TripleKnowledge::compositions`, §6.1, §6.6). No atom is added. A "place a third atom" mode is a separate, later
   design.
3. **No "Fill gaps on this slice" in this feature.** Drag-select covers it, at a cost the user sees. A slice-only Fill
   gaps (the pair graph's, limited to one r₂ slice) is a possible later addition.
