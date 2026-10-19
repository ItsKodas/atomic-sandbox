# 2D molecule builder — implementation plan

> Executed inline by the session that wrote it (no subagents, user instruction 2026-10-20), one part per commit, the user hand-testing each part in the app before the next. Model tag: **Opus 5.5** for all four parts (UI, tooling, an editor's embedding — none of it reaches `qm`/`sim`/`providers`; rule 1's sign-off is in the spec §3).

**Goal:** File ▸ Builder… — sketch a molecule as a 2D structural formula and hand a sensible 3D starting geometry to the scene (Replace / Insert, Relax after).

**Spec:** `docs/superpowers/specs/2026-10-20-molecule-builder-design.md` (approved 2026-10-20, with the rule-1 sign-off of its §3).

**Architecture:** a new module `crates/app/src/builder/` (`mod.rs` the drawing + undo + hand-off, `chem.rs` the two tables and their arithmetic — their only home —, `embed.rs` the pure 2D → 3D function, `ui.rs` the window). Its state lives in `UiState::builder` (display state, not recorded). The window emits one new `ui::Action::Builder(Handoff)`; `ui::actions::apply` turns it into the ordinary `Command`s (`SetScene`, or `AddNucleus` + `SetCharge`; `SetGrid` + `SetWalls` when the box is enlarged; `DynCommand::Relax`). The recorder writes it as `# not replayable: builder`.

## Global constraints

- Rule 1: the valence and covalent-radius tables exist only in `builder/chem.rs`; they produce 2D/3D positions and warnings, nothing else. Nothing under `crates/{sim,qm,qm-gpu,providers,surrogate}` changes.
- Atomic units: the embedding works in Bohr (`units::angstrom_to_bohr` on Cordero's Å once, in `chem`).
- Elements H–Ar only (`ui::ptable::enabled`).
- Tests: `cargo test -p app --release --bin app -- builder::` only (CPU, seconds); no GPU suites.
- The window says: "bond lengths and valences here only place the atoms; the solver decides the real geometry and bonds".

## Review focus (inputs the spec implies but does not name)

1. Keys while the window is open: Space / Delete / Esc must not reach the 3D view (Space would play, Delete would remove the selected nucleus) → `Viewer::input` treats every event as consumed while `builder.open`; `ui::keys` skips Deselect; the molecule labels are not drawn over the window.
2. An empty drawing, a lone atom, several disconnected fragments → the embedding returns finite positions for each (a lone atom at the origin; fragments kept ≥ the repulsion distance apart); Replace with nothing drawn is disabled.
3. Erasing an atom shifts the later indices → bonds are re-indexed, the selection and any drag in progress are dropped; undo restores the exact drawing (`Drawing: PartialEq` test).
4. A drag released on its own start atom or onto an existing bond partner → no self-bond, no duplicate bond (an existing bond cycles its order instead).
5. Insert into a scene whose total charge plus the formal charges is not the scene's → exactly one `SetCharge` with the sum; Replace into a cell too small → refused with the enlarge offer, never sent unfitted.

## Part 1 — the window and the canvas (commit 1)

**Files:** create `crates/app/src/builder/{mod.rs,ui.rs}`; modify `main.rs` (`mod builder;`), `ui/mod.rs` (`UiState::builder`, `show` calls `builder::ui::show`, `keys` guard), `ui/menus.rs` (File ▸ Builder…), `ui/ptable.rs` (the table grid factored out as `pub fn grid(ui, query: &mut String, selected: u8, id: &str) -> Option<u8>` so the builder's "more…" popover picks its own element, not the Place tool's), `viewer.rs` (the input/label guards).

- `builder::Drawing { atoms: Vec<Atom>, bonds: Vec<Bond> }` (`Clone, Default, PartialEq`), `Atom { z: u8, pos: [f32; 2], charge: i8 }` (canvas units: one drawn bond ≈ 1.0), `Bond { a: usize, b: usize, order: u8 }`.
- Operations on `Drawing`: `add_atom(z, pos) -> usize`, `set_z(i, z)`, `bond(a, b) -> Option<usize>` (none for a == b; an existing bond cycles), `cycle_order(k)` (1 → 2 → 3 → 1), `cycle_charge(i)` (0 → +1 → −1 → 0), `remove_atom(i)` (re-indexes bonds), `remove_bond(k)`, `bond_between(a, b)`, `bond_sum(i)`, `components() -> Vec<Vec<usize>>`, `charge() -> i32`.
- `builder::Builder` (in `UiState`): `open`, `drawing`, undo/redo stacks of `Drawing` (depth 100; `checkpoint()` before every edit, once per drag), `tool: Tool { Atom, Bond, Charge, Erase, Select }` (default Atom), `element` (default 6), `selection: BTreeSet<usize>`, view (`pan`, `zoom`), the gesture in progress, `relax_after` (default true).
- Window: an `egui::Modal` sized like the File window; closes on Close or Esc (not on a click outside — a stray click must not end a drawing). Palette (elements H C N O F P S Cl + "more…", tools), canvas, footer (formula + name via `names::label_components` on the drawn graph with Σ formal charges, `ui::rich`, the user's `name_style`; the rule-1 line; Clear). Canvas gestures exactly as spec §4 (click / drag-from-atom chain / bond cycling / charge cycling / erase incl. right-click and Delete / select-move with box select / wheel zoom about the cursor / middle- or right-drag pan / Ctrl-Z, Ctrl-Y, Ctrl-Shift-Z).
- Tests (`builder::tests`): `erase_reindexes_bonds_and_undo_restores`, `drag_onto_partner_cycles_not_duplicates`, `no_self_bond`, `formula_and_name_of_a_drawn_ethanol` (C₂H₆O — ethanol via `names`), `undo_depth_is_capped`.

## Part 2 — chem helpers (commit 2)

**Files:** create `crates/app/src/builder/chem.rs`; modify `builder/{mod.rs,ui.rs}`.

- `usual_valence(z) -> u8` (the spec §3 list), `allowed(z, q) -> &[u8]`-like `Vec<u8>` (P 3/5, S 2/4/6 when neutral; a charged atom: the octet count of its v − q electrons — reproduces NH₄⁺, H₃O⁺, OH⁻, BH₄⁻, CH₃⁺/CH₃⁻), `valence_electrons(z)` (from Z's period, Z ≤ 18), `lone_pairs(z, q, bond_sum)`, `covalent_radius_bohr(z)` (Cordero 2008 single-bond radii, Å → Bohr), `order_factor(order)` (1, 0.87, 0.78).
- `warnings(&Drawing) -> Vec<Warning { atom, kind: Over | Under }>`; red ring for Over always, amber for Under after Fill was used on this drawing; the footer counts them.
- `fill_hydrogens(&mut Drawing)`: each non-H atom to the smallest allowed valence ≥ its bond sum, new H's in the largest angular gaps around it.
- Stamps CH₃ · OH · NH₂ · COOH (with their H's) and benzene (C₆, alternating 1/2, regular hexagon, no H's): bonded to the one selected atom along its largest gap, else free at the view's centre.
- Tests: `fill_h_on_c_n_o` (CH₄, NH₃, H₂O from lone C, N, O), `fill_h_on_charged_n_and_o` (NH₄⁺, OH⁻; and H₃O⁺, BH₄⁻), `warnings_flag_over_and_under`, `neutral_valences_match_the_spec_table`, `benzene_stamp_is_a_regular_alternating_ring`.

## Part 3 — the 2D → 3D embedding (commit 3)

**Files:** create `crates/app/src/builder/embed.rs`.

- `pub fn embed(d: &Drawing) -> Vec<[f64; 3]>` (Bohr, centred at the origin), per spec §5: r₀ = radius sum × order factor; ideal angles from SN = σ count + lone pairs (180 / 120 / 109.47°, 107° with one lone pair and 104.5° with two at SN 4; SN ≥ 5: neighbour pairs only kept apart, ≥ the 90° distance); start = the drawing scaled to the mean r₀ with deterministic z offsets ±0.3 Bohr (an index hash); energy = bond springs + 1–3 springs + a soft repulsion between pairs beyond 1–3 closer than max(1.2 r₀, 3 Bohr) + planarity (the scalar triple product, for SN 3 centres with three σ neighbours and across each double bond); gradient descent with backtracking and a 0.2 Bohr step cap, ≤ 2000 iterations, converged on max |∇|.
- `pub fn place_beside(existing: &[[f64; 3]], new: &mut [[f64; 3]])` (along +x in 0.25 Bohr steps until the closest pair ≥ 3 Bohr).
- Tests (spec §8): `ch4_is_tetrahedral` (H–C–H 109.5° ± 3°), `ethylene_is_planar_and_trigonal` (all six atoms within 0.05 Bohr of one plane, angles ≈ 120° ± 3°), `water_is_bent` (90°–115°), `benzene_is_planar_with_equal_cc` (C–C within ± 2 %, planar within 0.05 Bohr), `no_nonbonded_pair_closer_than_1_5_bohr` (ethanol, acetic acid, benzene), `embedding_is_deterministic`, `place_beside_keeps_3_bohr`.

## Part 4 — the hand-off (commit 4)

**Files:** modify `builder/{mod.rs,ui.rs}`, `ui/actions.rs` (`Action::Builder(Handoff)`, `describe` → `None`), `replay.rs` (`unreplayable_kind` → `"builder"`), `ui/mod.rs` (`SceneSource::Built(String)`; label; Reset re-sends the last Replace without Relax), `ui/menus.rs`, `ui/files.rs` (the new source variant), `ui/search.rs` (Ctrl-K "Molecule builder"), `README.md` (Controls row).

- `Handoff { replace: bool, nuclei: Vec<(u8, [f64; 3])>, charge: i32, relax: bool, grid: Option<(usize, f64)>, label: String }`.
- Fit: `fit_limit(scene, walls: Option<f64>) -> f64` = min(the `tools::clamp_into_cell` limit, half-width − `sim::dynamics::WALL_LAYER` when the walls are on); `enlarge_n(max_abs, h) -> usize` = the smallest odd n with (n − 1) h / 2 ≥ max_abs + 4 Bohr. The footer shows the misfit and an "Enlarge the box to n³" button with `quality::vram_check`'s verdict and the footprint vs the budget; accepting it puts the grid into the hand-off (walls follow: `default_wall_half_width` of the new cell).
- Replace: a new `Scene` (nuclei with default isotopes, `charge` = Σ formal charges, multiplicity auto, σ / ionic policy / grid kept unless enlarged) → `vram_check` → `SetScene` (+ `SetWalls` when on). Insert: `place_beside` the current nuclei, `SetGrid` first if enlarged, one `AddNucleus` per atom, `SetCharge(scene + Σ)` only when the sum is non-zero. Relax after: `DynCommand::Relax` last (the `RELAX_REFUSED_SAMPLING` status during a pass). The window closes. A drawing with warnings needs a second press, the confirm line naming them.
- Tests: `replace_scene_charge_is_the_formal_sum`, `insert_sends_one_set_charge_with_the_new_total`, `fit_check_and_enlarge_n` (an odd n, ≥ 4 Bohr of margin), `handoff_is_not_replayable`.
- README: the Controls row for File ▸ Builder…; CLAUDE.md's layout line gains `builder`.

## Verification per part

`cargo build -p app --release` (read the output), `cargo test -p app --release --bin app -- builder::`, launch `target\release\app.exe` for the user's hand-test, wait for their feedback. Merge only on the user's word, after one `cargo test -p app --release --bin app` on the merged tree.
