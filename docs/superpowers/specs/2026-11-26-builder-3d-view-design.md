# A 3D view in the molecule builder: design (short)

**Date:** 2026-11-26 · **Status:** approved by the user ("yeah go with C") · **Requested by:** the user. They asked to insert molecules that don't fit the box, and gave the reason: "maybe you want to look at the molecules structure".

Builds on the builder database (`2026-11-25-builder-database-design.md`; code `crates/app/src/builder/`).

## Intent

A molecule the box can't hold (the enlarge offer is over the GPU memory budget) can't be handed to the scene today, so its 3D shape can't be seen at all. What the user wants is to look at the structure. A scene in which they could solve it is not needed for that.

## Approaches

- **C (chosen). A read-only 3D view inside the builder.** A 2D | 3D toggle swaps the sketch for a ball-and-stick of the shape Insert/Replace would place. It works for any size, never touches the scene or the GPU memory guard, and needs about half a day.
- **A. "Enlarge anyway".** Insert into an over-budget box and hold the solver until the user raises GPU memory or lowers quality. Rejected for this purpose:
  - it needs a new "loaded but not solvable" scene state through the worker, because over-budget scene changes are refused before they are sent, or wgpu panics on the allocation;
  - the main view draws bonds from the solved density, so an unsolved scene would show bare atoms.
  - It can come later, for a different reason: wanting the atoms in the scene.
- **B. Insert into the current box regardless.** Rejected: atoms past the walls or the grid give a refused or wrong solve.

## Design

1. **The toggle.** A small segmented **2D | 3D** control at the canvas's top edge. 2D is the default and today's canvas. The choice is display-only `UiState`: not a `ui::Action`, not recorded, not saved in the profile.
2. **What 3D shows.** The positions the hand-off would use for **Replace scene**: the `Shaped` positions from `Builder::shaped()`, so a database shape, a LIRA shape or the approximate embedding, exactly as placed. The bonds are the drawing's bonds, with their order drawn as single, double or triple sticks. It updates live when the drawing changes or a PubChem lookup answers.
3. **Look.** Ball-and-stick on the canvas's own background:
   - atoms are discs in the render crate's element colours (`render::elements::colour_srgb`), sized by the builder's existing display radii scaled down, depth-sorted back to front, with a simple shade by depth;
   - bonds are drawn between atom rims;
   - element letters stay on the atoms as in 2D;
   - a formal charge shows as on the 2D canvas.
   
   It is drawn with the egui painter: a CPU projection with no wgpu pass, so the GPU memory budget plays no part.
4. **Controls.**
   - Drag to rotate (trackball-style about the shape's centroid); the wheel zooms.
   - Double-click resets the view: the principal-axes view, the same frame as the 2D layout's projection.
   - **Hover a bond:** its length in Å, through the `units` crate.
   - **Hover an atom:** its element and its bond angles to bonded neighbours, at most 6, in degrees.
   - A one-line hint under the view gives these controls.
5. **Read-only.**
   - In 3D the drawing tools don't act on the view: clicks don't place or erase.
   - The palette and the search stay live; a stamp switches the view to show the new shape.
   - Insert, Replace, Clear, Fill hydrogens and the footer work as in 2D. The toggle changes nothing that is handed off.
6. **Honesty.** The footer's shape line already says where the shape comes from. In 3D, an approximate (not-in-database) fragment is also drawn with a muted outline, so a guessed shape is not mistaken for a real one.
7. **Rule 1:** display only; no fitted parameters; nothing reaches `sim`, `qm` or `providers`.
8. **Screenshot key:** `--ui "builder=<query>;builder_view=3d"` renders the 3D view, for the docs.

## Tests

- **Projection and rotation maths** (unit tests): a rotation keeps distances; the reset view of a linear molecule lies along x; depth sorting puts the nearer atom last.
- **Hover:**
  - the bond under the pointer is found and its length is reported in Å, e.g. water's O–H from the PubChem record, compared with the record's own distance;
  - an atom's angles are reported in degrees, e.g. water's H–O–H against the record.
- **No hand-off change:** the `Handoff` is identical with the view in 2D or 3D, for a matched record and for an approximate fragment.
- **UI test** (real egui frames, as the builder's existing UI tests do):
  - toggling to 3D and dragging changes only the view;
  - a click in 3D adds no atom;
  - a stamp from the search while in 3D shows the new shape.
- **Gate:** `cargo test -p app --release --bin app -- builder:: ui::`, then `cargo test -p app --release` once on the merged code.
- **Docs:** the README builder row and the dev-reference builder bullet (the toggle, the hover readouts, the screenshot key).

## Left to the builder

The exact sizes, colours and shading; the rotation scheme (trackball or orbit); where the toggle sits; and how the hover readout is laid out.
