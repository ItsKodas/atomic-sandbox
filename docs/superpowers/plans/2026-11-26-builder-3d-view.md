# Builder 3D View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only 2D | 3D toggle in the molecule builder that shows the shape the hand-off would place as a rotatable ball-and-stick, with bond-length and angle readouts on hover.

**Architecture:**
- A new `crates/app/src/builder/view3d.rs` holds the view state (rotation, zoom), the projection, depth sorting and hover hit-testing, as pure functions on `Shaped` positions and the drawing's bonds.
- `ui.rs` draws it with the egui painter in place of the canvas when the toggle reads 3D.
- The toggle is display-only `UiState`.

**Tech Stack:** Rust, egui 0.36 painter, glam (already a dependency), the `units` crate for Å, `render::elements` for colours.

**Spec:** `docs/superpowers/specs/2026-11-26-builder-3d-view-design.md`

## Global Constraints

- Display only. The toggle is not a `ui::Action`, is not recorded and is not saved in the profile. Nothing handed off changes (spec §1, §5).
- What is drawn is exactly `Builder::shaped()`'s positions, the ones Replace scene would place, with the drawing's bonds and their orders (spec §2).
- No wgpu pass: everything is drawn with the egui painter on the CPU (spec §3).
- Lengths in Å through `units`, angles in degrees. Positions are Bohr internally (CLAUDE.md rule 3).
- An approximate (not-in-database) fragment gets a muted outline (spec §6).
- Rule 1: no per-element behavioural data. Colours and radii are display only.

## Review Focus

- **Empty, one-atom and two-atom drawings:** the 3D view shows them without a NaN or a panic. A single atom has no bonds and no angles, and the reset view of a diatomic is well defined.
- **The shape changes while 3D is open** (a PubChem answer arrives, or a stamp, an erase or Clear happens): the view updates and the hover never indexes an atom that no longer exists.
- **Very large hand drawings** (over 100 atoms, past the database's 30): drawing stays cheap per frame. Sort once per frame; no O(n²) work beyond the bonds' and angles' natural size.
- **Coincident or overlapping atoms in an approximate embedding:** depth sort and hover stay stable, with no division by zero in the projection.
- **The builder closed and reopened in 3D:** it reopens in 3D, or 2D (the builder's choice), without a stale rotation for a different molecule. Double-click always resets.

---

### Task 1: The 3D view, its toggle, readouts, screenshot key and docs (Opus)

**Files:**
- Create: `crates/app/src/builder/view3d.rs`
- Modify:
  - `crates/app/src/builder/mod.rs`: the view state on `Builder`, display-only;
  - `crates/app/src/builder/ui.rs`: the toggle, and drawing 3D in place of `canvas_area`;
  - `crates/app/src/screenshot.rs`: the `builder_view=3d` key;
  - `README.md` and `docs/dev-reference.md`: the builder row and the builder bullet.
- Test: `view3d.rs` `#[cfg(test)] mod tests`; UI tests beside the builder's existing ones (`crates/app/src/builder/` test modules or `crates/app/src/ui/tests.rs`, following where the builder UI tests live today).

**Interfaces:**
- Consumes:
  - `Builder::shaped() -> &shape::Shaped` (`mod.rs`). Positions are in Bohr, indexed like `Drawing::atoms`.
  - `Drawing::{atoms, bonds}`, bond order and formal charge.
  - The per-fragment source/label in `Shaped`, to find approximate fragments.
  - `render::elements::colour_srgb(z)`.
  - `units` for Bohr→Å.
- Produces:
  - `pub(crate) struct View3d { rot: glam::DQuat, zoom: f64 }` with `Default` and `fn reset(&mut self, pos: &[[f64; 3]])`, the principal-axes frame.
  - `fn project(&self, pos: &[[f64; 3]], rect: egui::Rect) -> Vec<(egui::Pos2, f64 /*depth*/, f32 /*radius px*/)>`.
  - `fn hit(&self, …) -> Option<Hover>`, where `enum Hover { Atom(usize), Bond(usize) }`.
  - `fn bond_length_angstrom(pos, bond) -> f64` and `fn angles_at(pos, bonds, atom) -> Vec<(usize, usize, f64 /*deg*/)>`, the latter capped at 6.
  - Exact names are the implementer's call; keep them `pub(crate)` inside `builder`.

- [ ] **Step 1: Write the failing unit tests** in `view3d.rs`:
  - `a_rotation_keeps_distances`: project with a rotation and with none; the 3D distances between rotated points equal the originals to 1e-12.
  - `the_reset_view_of_a_linear_molecule_lies_along_x`: CO₂ positions; after `reset`, the projected O–C–O screen points share a y within 1e-6 px.
  - `the_nearer_atom_is_drawn_last`: two atoms at different depth; the draw order ends with the nearer one.
  - `water_bond_and_angle_read_out_from_the_record`: water's shipped PubChem record via `shape()`. `bond_length_angstrom` equals the record's own O–H distance converted with `units`, to 1e-9. `angles_at(O)` gives one angle equal to the record's H–O–H, to 1e-9 degrees.
  - `degenerate_drawings_do_not_panic`: empty, one atom, and two atoms at the same position. `reset` and `project` give finite values (no NaN), and `angles_at` is empty.
- [ ] **Step 2: Run them and watch them fail.** `cargo test -p app --release --bin app -- builder::view3d` (BelowNormal, `-j 4`).
- [ ] **Step 3: Implement `view3d.rs`.** One line on the approach: an orthographic projection after the rotation, with the scale fitted to the rect from the shape's extent times zoom; a perspective-free depth shade.
- [ ] **Step 4: Write the failing UI tests** (real egui frames):
  - `the_3d_view_changes_nothing_handed_off`: the same drawing, with Insert pressed in 2D and in 3D, gives identical `Action::Builder(Handoff)` payloads. Check both a matched record (water) and an approximate fragment.
  - `a_click_in_3d_adds_no_atom`.
  - `a_drag_in_3d_turns_only_the_view`: the drawing and the shaped positions are unchanged, and the rotation is changed.
  - `a_stamp_in_3d_shows_the_new_shape`: stamp "water" from the search while in 3D; the view's atom count is 3.
- [ ] **Step 5: Wire the toggle and the drawing into `ui.rs`.**
  - The muted outline for approximate fragments.
  - The hover readout as a tooltip near the pointer.
  - The one-line controls hint under the view.
  - Double-click resets the view.
  - Read-only: the 2D drawing tools are not applied while 3D shows.
- [ ] **Step 6: Run the builder and UI tests and watch them pass.** `cargo test -p app --release --bin app -- builder:: ui::`: all pass.
- [ ] **Step 7: Add the screenshot key** `builder_view=3d`, with a parse test beside `builder=`'s, and render `--ui "builder=caffeine;builder_view=3d"` (or another shipped molecule with depth). Look at the PNG.
- [ ] **Step 8: Docs.** Add the toggle and the readouts to the README builder row. Add the module, the screenshot key and "display only, not recorded" to the dev-reference builder bullet.
- [ ] **Step 9: Commit.** `feat(builder): a read-only 3D view of the shape the hand-off places (builder 3D view, Task 1)`
