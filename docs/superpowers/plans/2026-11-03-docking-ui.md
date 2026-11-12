# Docking UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two fixed docks and the View menu with a docking workspace (drag / split / tab / resize / float, auto-hide pins, saved layouts with four presets, panels detachable to other monitors) and a duotone icon set.

**Architecture:** `egui_dock` 0.21 holds the main window's tree of tiles (`Tile::View3d` + one `Tile::Panel(DockTab)` per panel); our own `ui/workspace/` module owns the layout grammar the profile saves, the auto-hide strips and flyouts, the presets, and (D3) detached OS windows built on `capture/popout.rs`'s pattern. Panel bodies (`ui/dock/*.rs`) move across unchanged.

**Tech Stack:** Rust, egui 0.36, egui-winit 0.36, egui-wgpu 0.36, egui_dock 0.21.1, winit, wgpu.

**Spec:** `docs/superpowers/specs/2026-11-03-docking-ui-design.md` (decisions D-1 … D-14 are binding).

## Global Constraints

- All tasks tagged **Opus** (`claude-opus-5-5`); no physics changes (spec D-14).
- Display state only: nothing in this plan becomes a `ui::Action` or reaches the worker (spec D-13).
- The profile stays hand-readable `key = value` text; the layout uses the spec D-9 grammar.
- Never kill the user's running `app.exe`; build in this worktree (`D:\Projects\Physics\.claude\worktrees\docking`), one heavy cargo job at a time; when another suite is running build at BelowNormal with `-j 4`.
- Gate per task: `cargo test -p app --release --bin app -- ui::` (fast) plus the task's own tests; the full `cargo test -p app --release` (~10 min) at the end of each milestone.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Edit files with Edit/Write or Python, never PowerShell Set-Content.
- Keep `README.md` (controls, screenshots list) and `docs/dev-reference.md` ("Viewer layout", screenshot `--ui` keys) current in the milestone's last task.

## Review Focus

1. An old profile (`left_tabs = …`, `right_tabs = …`, no `layout`) and a corrupt `layout` line must open a sensible layout, never panic and never lose the 3D view.
2. The 3D view must keep receiving mouse input (orbit, place, grab, the edit rings) wherever it sits in the tree; panels over it (flyouts, floats) must take their own clicks.
3. Ctrl K must still reveal every control that used to be in the View menu.
4. A narrow window (< `COMPACT_WIDTH`) and a very small window must not crash the tree (egui_dock with tiny rects).
5. D3: closing a detached window, or loading a profile whose window was on an unplugged monitor, must bring its panels back.

---

## Milestone D1 — the tree, the View panel, the icons

### Task 1: Layout grammar (Opus)

**Files:** Create `crates/app/src/ui/workspace/layout.rs`, `crates/app/src/ui/workspace/mod.rs` (just `pub mod layout;` for now); modify `crates/app/src/ui/mod.rs` (`mod workspace;`), `crates/app/src/ui/mod.rs` `DockTab` (add `View`, title "View", into `ALL`).

**Interfaces — Produces:**
```rust
pub enum Node { View3d, Leaf { tabs: Vec<DockTab>, active: usize }, Split { horizontal: bool, frac: f32, a: Box<Node>, b: Box<Node> } }
pub fn parse(s: &str) -> Option<Node>          // None only when there is no 3D view after cleaning
pub fn encode(n: &Node) -> String
pub fn default_layout() -> Node                // spec D-9 example, exactly
pub fn from_sides(left: &[DockTab], right: &[DockTab]) -> Node  // the old profile's migration
pub fn clean(n: Node) -> Option<Node>          // drop unknown/duplicate/empty, clamp frac
pub fn panels(n: &Node) -> Vec<DockTab>        // in order
```
Tab names are `DockTab::title()` (case-insensitive on read). `from_sides`: the left column holds the old left tabs in order (Activity, Readouts, Graph by default), the right column holds the old right tabs; the View panel joins the right column's first leaf; the result has the default's fractions.

- [ ] Write tests in `layout.rs` (`#[cfg(test)]`): round trip of `default_layout()` through `encode`/`parse` is identity and the encoded text equals the spec D-9 string; `parse` of garbage → None; a tree with an unknown name drops it; a duplicate keeps the first; an empty leaf is removed and its split collapses; `frac` 2.0 clamps to 0.95; a tree without `3D` → None; `from_sides(&DockTab::LEFT, &DockTab::RIGHT)` contains all panels exactly once plus View; whitespace is tolerated.
- [ ] Run them, see them fail, implement, see them pass (`cargo test -p app --release --bin app -- workspace::`).
- [ ] Commit `feat(workspace): the layout grammar the profile saves (spec D-9)`.

### Task 2: The dock tree replaces the two docks (Opus)

**Files:** Add `egui_dock = "0.21.1"` to the workspace `Cargo.toml` and `crates/app/Cargo.toml`. Create `crates/app/src/ui/workspace/tree.rs`; modify `ui/mod.rs` (`UiState`, `show`), `ui/dock/mod.rs` (keep `body` per tab, drop `side`/`strip`/`drag_tabs`), `ui/profile.rs`, `ui/knowledge.rs` (remove `window` and its toggle per spec D-5), `screenshot.rs` (`tab=`, `dock=closed`), `ui/tests.rs`.

**Interfaces — Consumes:** Task 1's `Node`, `parse`, `encode`, `default_layout`, `from_sides`. **Produces:** `UiState::workspace: workspace::Workspace` with `fn to_node(&self) -> layout::Node`, `fn set_node(&mut self, n: layout::Node)`, `fn visible(&self, t: DockTab) -> bool`, `fn show_panel(&mut self, t: DockTab)` (make active in its leaf; if closed, add it to the right column's first leaf, or a new split right of the 3D view if none), `fn close_panel(&mut self, t: DockTab)`; `UiState::open_tab`/`tab_shown`/`reset_layout` delegate to these.

- [ ] First read `egui_dock` 0.21.1's docs/source (`~/.cargo/registry/src/*/egui_dock-0.21.1`) and record in the report: how to keep one tab non-closable / non-floating / without tab bar, how a tab gets its body rect, how to paint into a tab's title area (icon + status dot), how to add buttons at the end of a leaf's tab bar, and whether a drop into a given leaf can be refused.
- [ ] `Tile::View3d` per spec D-3: `clear_background` false, not closable, not allowed in windows, no tab bar for its leaf (or undo drops into it on the same frame); its body rect becomes `PanelOutput::view` (replacing `root.available_rect_before_wrap()`). Panels: closable, may float (spec D-5).
- [ ] Tabs per spec D-4: duotone icon (Task 3 provides the fill; use `icons::paint` now), name, status dot from the existing strip code (`dock/mod.rs`'s strip dot logic moves into a `fn status(tab, st, input) -> Option<(Color32, Option<String>)>`), hover peek. Leaf tab-bar end: a close button now; pin / pop-out arrive in D2 / D3.
- [ ] Profile: write `layout = …`; read it (else `from_sides` from `left_tabs`/`right_tabs`, else default). Drop `left_tab`, `right_tab`, `left_tabs`, `right_tabs`, `left_width`, `right_width`, `dock_open`, `left_open` from the writer; keep reading the tab lists for migration. Update profile tests.
- [ ] The Activity tab's `SetActivityViews` gating and the Log unread reset use `tab_shown`, now meaning "visible in the tree".
- [ ] Screenshot harness: `tab=<name>` calls `show_panel`; `dock=closed` closes every panel (until D2's Presenting preset replaces it). Existing shots must still parse.
- [ ] Tests (headless, `ui/tests.rs` style with PROBES): the default tree shows Model, Selection, View, Log; `open_tab(Graph)` makes it visible; closing then `open_tab` brings it back; the 3D view rect is non-empty and lies between the left and right columns; a profile with only `left_tabs`/`right_tabs` loads with every panel present once; a profile with `layout = nonsense` loads the default. Fix every existing test that used `DockSide`/`left_open`/`dock_open` to the new API without weakening what it checks.
- [ ] `cargo test -p app --release --bin app -- ui::` passes; commit `feat(workspace): one dock tree (egui_dock) replaces the two fixed docks`.

### Task 3: Duotone icons (Opus)

**Files:** `crates/app/src/ui/icons.rs`, `crates/app/src/ui/theme.rs` (an `ICON_FILL` = accent at alpha ≈ 82).

**Produces:** `Icon::{View, Knowledge, Pin, Unpin, DockBack, Split, Layouts, Close, Drag}` and a fill pass in `paint` (fill first, then the existing strokes). The shapes are the brainstorm's (24 × 24 box): Model's three atoms filled; Knowledge a folded map, middle panel filled; Selection the inner circle filled, outer ring dashed; Readouts and Activity a soft rounded-square backdrop; View an eye, the eye shape filled, pupil stroked; Orbitals both lobes filled; Log the page filled; Graph the area under the line filled; Pin a push-pin with its head filled (Unpin the same tilted 45°); DockBack a window with an arrow entering; Split a box halved, left half filled; Layouts a 2 × 2 grid, top-left filled; Close an X; Drag six dots. The fill uses `theme::ICON_FILL` whatever the stroke colour; icons drawn disabled use the fill at half alpha.

- [ ] Test: `paint` of every `Icon` into a 24 px rect on a headless painter adds ≥ 1 shape and no NaN positions (iterate an `Icon::ALL` you add).
- [ ] Use the new icons on the tabs (View, Knowledge) and leaf buttons (Close); commit `feat(icons): the duotone set (user choice 2026-11-03)`.

### Task 4: The View panel and the Window menu (Opus)

**Files:** Create `crates/app/src/ui/dock/view.rs`; modify `ui/menus.rs` (move `view_menu`, `trail_controls`, `cloud_controls` contents out; `MenuId::View` → `MenuId::Window`), `ui/search.rs`, `ui/knowledge.rs` (`overlay_toggle` called from the panel), `screenshot.rs` (`menu=view` maps to showing the View panel; `menu=window` opens the menu), `ui/tests.rs`.

- [ ] Sections and their contents exactly per spec D-6, as collapsing headers (ids `view-what`, `view-looks`, `view-camera`, `view-bonds`, `view-aids`), open by default: What you see, How it looks; the open state persists in the profile (`view_sections = what,looks`). Hide what does not apply (Dots settings only in Dots mode; Camera settings only while the Camera view is on; orbital colouring only while orbitals are shown). Every widget keeps its existing `anchored(...)` anchor name and probe name.
- [ ] Window menu (spec D-7): one row per panel with its icon (click: `show_panel`; a check when visible), a separator, Layouts ▸ (placeholder entry "One monitor" applying the default until Task 7), Reset layout.
- [ ] Ctrl K: every entry with `Menu(M::View)` becomes `Dock(D::View)`; revealing one shows the View panel and forces its section open for that frame. Add a test that every former View-menu entry resolves to a probe inside the View panel after reveal.
- [ ] Remove the View menu; existing tests that opened it open the panel instead.
- [ ] Commit `feat(view-panel): the View menu's settings as one organised panel; Window menu`.

### Task 5: D1 docs and gate (Opus)

- [ ] README: controls (the workspace: drag, split, float, Window menu, the View panel), screenshot list if any shot changes; `docs/dev-reference.md` "Viewer layout" rewritten for the workspace, `--ui` keys updated; CLAUDE.md crate map line for `app` mentions `src/ui/workspace/`.
- [ ] Run the full `cargo test -p app --release`; regenerate any docs screenshot the README embeds that shows the old docks (`--screenshot` with the same `--ui` keys).
- [ ] Commit `docs: the docking workspace (D1)`.

---

## Milestone D2 — auto-hide pins, layouts, presets

### Task 6: Auto-hide (Opus)

**Files:** Create `crates/app/src/ui/workspace/autohide.rs`; modify `workspace/tree.rs`, `workspace/layout.rs` (the `hidden` line), `ui/profile.rs`, `ui/icons.rs` use, `ui/tests.rs`.

**Produces:** `Workspace::pin(&mut self, t: DockTab, pinned: bool)`, `Workspace::hidden(&self) -> &[Hidden]` with `struct Hidden { tab: DockTab, edge: Edge, size: f32 }`, `enum Edge { Left, Right, Bottom }`; `layout::encode_hidden`/`parse_hidden` for `hidden = Graph:left:340, Log:right:300` (names case-insensitive, sizes clamped 160..=900, unknown dropped).

- [ ] Behaviour exactly per spec D-10 (the edge is where the panel's leaf centre lay relative to the 3D view's rect; size = its leaf's width, or height for Bottom). Strip buttons reuse `icons::strip_button` and Task 2's `status` dots. The flyout is an `egui::Area` anchored to the edge over the 3D view with the panel's body and a title bar (icon, name, pin, close); it consumes the pointer so the 3D view does not orbit through it.
- [ ] `open_tab(t)` on a hidden panel opens its flyout (stuck open); `tab_shown(t)` is true while its flyout is open.
- [ ] Leaf tab-bar pin button (Pin icon) unpins the active tab; the flyout's pin re-docks it.
- [ ] Tests: unpin Graph → not in the tree, in `hidden()` with edge Left; `open_tab(Graph)` → shown; pin → back in the tree; profile round trip with a hidden panel; a hover over the strip button for 300 ms of simulated frames opens the flyout and moving away for 600 ms closes it.
- [ ] Commit `feat(workspace): auto-hide pins (Visual Studio style)`.

### Task 7: Presets and Window ▸ Layouts (Opus)

**Files:** Create `crates/app/src/ui/workspace/presets.rs`; modify `ui/menus.rs` (Window ▸ Layouts), `screenshot.rs` (`layout=<one|two|training|presenting>`), `ui/search.rs` (the four presets and Reset layout searchable), `ui/tests.rs`.

- [ ] `enum Preset { OneMonitor, TwoMonitors, Training, Presenting }` with `fn tree(self) -> (Node, Vec<Hidden>)` exactly per spec D-11 (TwoMonitors = OneMonitor until D3). Applying one replaces the current arrangement and the profile autosaves it. `reset_layout` applies OneMonitor.
- [ ] Tests: every preset contains every panel exactly once (tree ∪ hidden); Presenting's tree is just `3D`; applying then encoding then parsing returns the same arrangement.
- [ ] Commit `feat(workspace): the four built-in layouts`.

### Task 8: D2 docs and gate (Opus) — README, dev-reference (`layout=` key, auto-hide), full `cargo test -p app --release`; commit `docs: auto-hide and layouts (D2)`.

---

## Milestone D3 — detach to other monitors

### Task 9: Detached OS windows (Opus)

**Files:** Create `crates/app/src/ui/workspace/detached.rs` (UI side: per-window tree, tab menu "Move to …"), `crates/app/src/detached.rs` (window side: winit window, surface, egui `Context`, egui-winit `State`, egui-wgpu renderer, on `capture/popout.rs`'s pattern); modify `main.rs` (route `WindowEvent`s by `WindowId`; create/destroy windows on request; redraw them after the main window), `viewer.rs` (draw detached windows each frame with the same `UiState` and `PanelInput`, append their actions to the frame's), `workspace/layout.rs` (`window = x,y,w,h <tree>` lines), `ui/profile.rs`, `ui/tests.rs`.

- [ ] Behaviour exactly per spec D-12. Requests flow UI → viewer through `PanelOutput` (a `Vec<WindowRequest>`: `Open { id, panels, rect }`, `Close { id }`, `Focus { id }`), because the UI cannot create winit windows.
- [ ] The leaf tab bar gains the PopOut button; tabs get a context menu with "Move to new window", "Move to main window" and "Move to window N".
- [ ] Closing the OS window → its panels go back (to the leaf each left, else right of the 3D view). Profile load: a window whose rectangle intersects no monitor of `event_loop.available_monitors()` docks back.
- [ ] Two monitors preset: Activity and Graph in a detached window filling the second monitor's work area; with one monitor it is One monitor with a status note "Only one monitor found".
- [ ] Ctrl K reveal of a panel in a detached window sends `Focus`.
- [ ] Tests (UI side, headless): popping out Activity removes it from the main tree and lists it in a detached window; closing that window returns it to its old leaf; profile round trip with a window line; a window off every monitor docks back (monitor list injected).
- [ ] Hand-test checklist for the user in the report (two monitors).
- [ ] Commit `feat(workspace): detach panels to their own windows (other monitors)`.

### Task 11: Drag tabs between windows; no in-window floating; the 3D view never shares a drag (Opus)

User decision 2026-11-12 (spec D-5, D-15, D-16); runs after Task 9, before Task 10.

**Files:** modify `crates/app/src/ui/workspace/tree.rs` (tab drag latch from `TabViewer::on_tab_button`'s `Response`, `allowed_in_windows` false, drop-target overlay and the move), `ui/workspace/detached.rs` (window rects in desktop pixels for hit-testing, the move into/out of a detached tree, a window whose last tab leaves closes), `crates/app/src/main.rs` and `crates/app/src/detached.rs` (desktop cursor position during a drag — winit on Windows keeps sending the pressed window `CursorMoved` outside itself; convert with each window's inner position and scale factor; redraw the target window while a drag is over it), `crates/app/src/viewer.rs` / `main.rs` (D-16 input gate), `ui/tests.rs`, README, dev-reference.

- [ ] No floating: `allowed_in_windows` false everywhere; any code and docs for floats (incl. `present()` for a floating panel, the "floats are lost on restart" minor) go.
- [ ] Drag out: a tab dragged with the pointer released outside every app window → `WindowRequest::Open` with that tab, rect at the release point (desktop px), the panel's current size, clamped into the work area of the monitor under the cursor; the tab leaves its tree (a detached window left empty closes).
- [ ] Drag over another app window → that window draws drop targets (leaf centre = tab in, four edges = split, tab bar = insert; the main 3D tile: edges only) at the cursor; release on a target applies the move through the existing `Change`s; no target / Esc cancels. Main ↔ detached and detached ↔ detached.
- [ ] D-16: a press the UI takes (widgets, resize grips, separators and their grab margins just outside the drawn edge) never moves the camera or starts a tool; a drag the UI holds never reaches the 3D view. Find why `consumed` misses these presses (egui-winit's `wants_pointer_input` is last frame's and excludes grab margins) and fix it at the source; test with a press in a separator's / grip's margin over the 3D view.
- [ ] Display state only (D-13): nothing becomes an `Action`.
- [ ] Tests (headless, monitors and window rects injected): release outside → one `Open` with the tab and the release rect; drop on a detached window's leaf centre / edge / tab bar moves the tab there; drop on the 3D tile's centre is refused, its edge splits; detached → main; last tab out closes the window; no float is ever created by a drag-release inside the window; D-16 margin press leaves the camera unchanged.
- [ ] Hand-test checklist in the report (three monitors, mixed DPI if available).
- [ ] Commit `feat(workspace): drag tabs between windows (out of the main window opens a new one)`.

### Task 10: D3 docs and final gate (Opus) — README (detaching, the Two monitors layout), dev-reference, DEVLOG entry, full `cargo test -p app --release`; commit `docs: detached windows (D3)`.
