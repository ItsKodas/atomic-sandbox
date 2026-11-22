# View overlays in zones: a responsive 3D view — design

**Approved approach (user, 2026-11-22): 1, zones that stack.** The user asked to overhaul the UI in the viewing window: "it's not the most responsive, when it gets too small, a lot of things overlap". Build after the weekly usage reset (24 Nov), after the refit visualisation and the faster trust recovery (they share code with neither).

## Why

Every overlay of the 3D view is its own `egui::Area`, pinned to a fixed point with a fixed pivot, and none of them knows the others exist. On a narrow or short view they collide. The user's screenshot (1186 px wide) shows:
- the sampling-pass card covering the orbital legend;
- the "orbital display solves paused" note covering the dots sliders.

A few local fixes exist: the bonds pair returns its top edge for the stale line (`Corner`), and the toasts stack. There is no shared layout.

## Scope

**Only the UI inside the 3D viewport** (the user, 2026-11-22: "only changing the UI within the viewing port"). The dock panels (Model, Activity, Orbitals …), the menus, the docking workspace, detached windows and the control search are untouched. Their code is not changed except where an overlay inside the view reads them.

## Goals

- **Nothing overlaps**, at any view size down to a stated minimum (below).
- **Everything stays in the view.** The user: the important things are already small and fine, and the rest is needed as much. Nothing moves to another panel. Long prose takes less room rather than disappearing.
- Each overlay keeps its look and contents. This is about placement and compact forms.

## Non-goals and anti-goals

- No new panel, no status bar outside the view (approach 2, not chosen).
- No priority system that hides overlays.
- Not a restyle of the overlays' contents. The bonds pair and its card stay as built 2026-11-22.
- Anti-goal: anything that jumps around frame to frame as sizes change (approach 3's failure). A zone's order is fixed, and its items move only when one appears, disappears or changes form.

## The overlays today (`egui::Area`, by file and anchor)

| Id | File | Anchor today | Zone |
|---|---|---|---|
| `status-chip` (the top bar: sample / iteration / badges / t / T, its help lines, the paused line) | overlays.rs | view top-left + 12 | **top** |
| `draft-banner` | overlays.rs | under the status chip | **top** (stacks under it) |
| `capture-buttons`, `capture-notes`, `capture-settings` | capture.rs | top-right | **top-right** |
| `segment-card` | overlays.rs | right, top + 56 | **top-right** (under capture) |
| `offer-card`, `teach-prompt-card` | overlays.rs | right edge, `right_y` | **right** (stack) |
| `model-offer-card` | overlays.rs | view centre | **centre** (modal-like; stays centred, see below) |
| the centre tag "training is solving · random nudge 3 of 60" | overlays.rs (`density-label`'s neighbour / sampling) | top-centre | **top** (centre slot) |
| `density-label` | overlays.rs | bottom-centre | **bottom-centre** |
| `sampling-card`, `relax-card` | overlays.rs | bottom-centre | **bottom-centre** |
| `toast` | overlays.rs | pivot / stack | **bottom-centre** (above the cards) |
| `view-switch` (Volume / Isosurface / Dots / Camera, and the dots/density sliders row, the orbital paused note) | overlays.rs | bottom-left | **bottom-left** |
| `orbital-legend` | overlays.rs | bottom-left, lifted by a fixed amount | **bottom-left** |
| `knowledge-legend` | knowledge.rs | bottom-left, lifted | **bottom-left** |
| `bonds-now`, `bonds-stale-line`, the Orbitals card | overlays.rs | bottom-right | **bottom-right** |

The builder confirms this list against the code (`Area::new` in `ui/`) and places anything missed by the same rules.

## The design

**Zones.** The view rect is divided into six zones, each a stack with a fixed order:
- **top:** the status bar full width at the top-left; the centre tag in its centre slot;
- **top-right:** capture, then segment;
- **right:** offer cards;
- **bottom-left:** view switch at the bottom, then the sliders row, the paused note, then the legends above;
- **bottom-centre:** cards at the bottom, then the density label, then toasts above;
- **bottom-right:** the bonds pair, the stale line and the orbitals card above it.

**One layout pass per frame.** Each overlay first reports the size it wants in its full form and in its compact form. The pass then:
1. gives the bottom zones the bottom band;
2. splits the band's width between bottom-left, bottom-centre and bottom-right. Each keeps its stack's widest item; bottom-centre gets the rest, centred, and never narrower than its compact width;
3. if the three do not fit side by side, switches the widest items to compact forms first (below). If they still do not fit, bottom-centre moves up above bottom-left and bottom-right: a second row, never an overlap;
4. stacks each zone upward from its edge, with a gap of 6 points;
5. gives the top zone the top band, and the right and top-right zones the right edge between the two bands.

Each overlay is then drawn at its rect with `fixed_pos` and `pivot`, as now, from a shared `ViewLayout` the pass returns, instead of its own constants. The pass is pure (rects in, rects out), so it is unit-tested without egui.

**Compact forms (only these).**
- **Long help lines** (the status bar's second and third lines, the legend's caption, the orbital paused note) fold to one line, cut with "…", plus an ⓘ that shows the full text on hover. Triggered when the line would wrap past 2 lines at the zone's width.
- **The sampling, relax and segment cards** shrink to a single row: title · stage n/N · Pause · Cancel. The details go into the hover.
- **The sliders row** wraps onto two rows when narrow. If the zone still lacks room, the sliders go into a popover behind a sliders button next to the view switch.
- **The status bar** keeps its first line (the important small items) always, and wraps it onto two lines rather than clipping.

**The minimum view.** Below about 560 × 360 points, the view's overlays use every compact form and the bottom-centre second row. Below that, nothing is promised except no overlap: the stacks may run past the top band, and then the lowest-priority prose (help lines) is dropped to its ⓘ alone.

**Modal-like cards** (`model-offer-card`) stay centred over everything and are not part of the zones. They ask for a decision and are dismissed.

## Approaches considered

1. **Zones that stack (chosen):** everything stays where the eye is; this fixes the cause.
2. **A status bar under the view:** the tidiest, but the eye leaves the view and it is a larger docking change.
3. **Collision nudging:** the cheapest, but things jump and a small view still runs out of room.

## Left to the builder

- The exact widths that trigger compact forms. Measure from the content; make them constants with their reason.
- How each overlay reports its two sizes. Last frame's measured rect (`egui` memory) is fine, with the compact size measured once.
- Animations, if any. A short ease on a zone's movement is allowed; no animation is required.

## Tests and gates

- **Pure layout tests:** no two rects overlap; each stays inside the view. Checked at 1920 × 1080, 1186 × 1280 (the user's screenshot), 900 × 600 and 560 × 360, with every overlay present at once and in common subsets (training, relax, orbitals with legend, toasts).
- **Existing probe tests** that pin positions (e.g. `the_stale_bonds_line_sits_over_bonds_now`, the toast and legend tests) are updated to the zone rules. Each updated pin says why.
- **A rendered-frame check** (`--screenshot` at the four sizes) before handing over a preview, as learned on 2026-11-22.
- `cargo test -p app --release`, including the replays.
