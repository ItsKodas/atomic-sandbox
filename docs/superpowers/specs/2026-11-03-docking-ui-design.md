# Docking UI — design

Status: approved by the user 2026-11-03 ("approach 1 with that split", then "just implement the new ui").
Brainstorm screens: `.superpowers/brainstorm/` (layout directions, default layout, icon styles).

## Why

The View menu is "such a mess" (user, 2026-10-21) and the two fixed docks can only hold tabs on
the left or right. The user wants a docking workspace like most pro apps: every panel its own
window that can be docked anywhere, tabbed, split, resized, auto-hidden ("pinned") and detached to
another monitor, with saved layouts and one consistent icon set.

## Decisions (user)

1. **Shape:** a full docking workspace (A) with the View-menu settings regrouped into one organised
   **View panel** with folding sections (B's inspector).
2. **Default layout:** main window = Model / Knowledge over Selection / Readouts on the left, the
   3D view, View / Orbitals over Log on the right; second monitor = Activity over Graph.
3. **Icons:** duotone — today's 1.7 px line icons plus a soft violet fill (accent at ≈ 32 % alpha)
   on each icon's main shape. Painted by `ui/icons.rs`, no icon font, no image files.
4. **Pin = auto-hide (Visual Studio):** an unpinned panel folds to an icon on its window edge and
   slides out over the 3D view on hover or click; pinned panels stay docked and take space.
5. **Layouts live in the profile:** the profile stores the arrangement. Four built-in layouts ship
   — One monitor, Two monitors, Training, Presenting — and choosing one rearranges the current
   profile (which then remembers it).
6. **Approach 1:** `egui_dock` (UI plumbing, allowed by rule 2) for the in-window tree; our own
   auto-hide; our own OS windows for detaching, on the capture pop-out's pattern
   (`capture/popout.rs`: a winit window on the main device with its own surface and egui renderer).
7. **Milestones:** D1 tree + View panel + icons; D2 auto-hide + layouts + presets; D3 detach.

## Decisions (Claude, recorded for review)

- **D-1 Base:** the `docking` branch starts from M8a's tip (it needs the Knowledge tab and the
  knowledge overlay). It merges to main after M8a does; main is merged into it before that.
- **D-2 Names:** the existing `DockTab` enum stays (it is used across ~5 k lines of tests) and
  gains `View`. "Dock tab" now means "panel". `DockSide`, `left_tabs`/`right_tabs`, `dock_open`,
  `left_open`, `dock_w` and the strip go; `UiState::open_tab`, `tab_shown` and `reset_layout` keep
  their meaning over the workspace so callers and tests keep working.
- **D-3 The 3D view is a tile, not a panel:** `Tile::View3d` is never closable, never floats or
  detaches, draws no background (the scene shows through) and shows no tab bar of its own. Its
  body rect is `PanelOutput::view`. Panels may split around it; if the dock library cannot stop a
  panel being tabbed into the 3D view's leaf, such a drop is undone on the same frame (the panel
  goes to a new split beside the view).
- **D-4 Tabs:** each tab shows its duotone icon, its name and its status dot (Readouts: drift /
  box health; Activity: work running; Model: trust; Log: unread count — today's strip dots).
  Hovering a tab shows today's peek. A leaf's tab bar ends with its controls: pin (auto-hide, D2),
  pop out (D3), close.
- **D-5 No floating inside the window** (user decision 2026-11-12, replacing D1's floating): `egui_dock`'s
  undock-to-floating-window is off (`allowed_in_windows` false). Dragging a tab out of the main window
  detaches it instead (D-15); dropping it on an empty spot inside a window cancels the drag. The
  Knowledge tab's hand-made in-app window (`knowledge::window`) stays removed; detaching does its job.
- **D-6 The View panel** (`ui/dock/view.rs`) holds every control of today's View menu, regrouped:
  - *What you see:* Density / Orbitals / All orbitals coloured / Spin density / None; the
    estimated solver's cloud (draft) controls.
  - *How it looks:* Volume / Isosurface / Dots; the volume's gamma, opacity, glow, partial core;
    the dots' density, drift, feather, size, blur, shimmer, colour by; bloom.
  - *Camera:* camera view and its settings (shown only while the Camera view is on).
  - *Bonds & labels:* bonds, molecule names (style), composition badge, angles, lengths, partial
    charges, label opacity.
  - *Display aids:* cell box, floor grid (Bohr / Å), axes, placement plane, trails (length,
    opacity), knowledge overlay, show what training is solving, smooth motion, nucleus size.
  - *Reset view settings.*
  Controls that do not apply are hidden, not greyed. Each section is a collapsing header whose
  open state is display state (saved in the profile). Every widget keeps its search anchor.
- **D-7 Menus:** the top bar becomes File · Scene · Physics · Window · Help. `MenuId::View` becomes
  `MenuId::Window`: one entry per panel (shows / focuses it; a check when it is visible), Layouts ▸
  (D2), Reset layout. Ctrl K results that lived in the View menu now live in
  `Place::Dock(DockTab::View)`; revealing one opens the View panel and unfolds its section.
- **D-8 Profile format:** a new line `layout = <tree>` in our own small text grammar (below), not
  the library's serde (the profile is hand-readable `key = value` text and must survive a library
  upgrade). Old profiles with `left_tabs`/`right_tabs` load into the D1 default with those tabs in
  the matching columns. D2 adds `hidden = …` (auto-hidden panels with their edge and size); D3
  adds `window = …` lines (detached windows).
- **D-9 Layout grammar:**
  ```
  node  := "3D" | leaf | split
  leaf  := "[" tab ("," tab)* "]"        tab := Name ["*"]   (* = the active tab)
  split := ("H" | "V") "(" frac "," node "," node ")"   frac = the first child's share, 0.05..0.95
  ```
  `H` puts children side by side, `V` stacks them. Example (the D1 default):
  `H(0.24,V(0.55,[Model*,Knowledge],[Selection*,Readouts]),H(0.70,3D,V(0.62,[View*,Orbitals],[Log*,Activity,Graph])))`.
  Opening the app for the first time, Activity and Graph are tabbed with Log on the right (ruling
  2026-11-03, docking Task 4: in the lower-left leaf beside Selection and Readouts the Graph tab was
  scrolled off its bar at 1400 pt); the Two monitors preset (D-11) moves them to the second monitor.
  Parsing is total: unknown names are dropped, empty leaves removed, a duplicated panel keeps its
  first place, a tree without `3D` (or that fails to parse) falls back to the default.
- **D-10 Auto-hide (D2):** unpinning a panel takes it out of the tree onto the strip of the edge it
  was nearest to (left, right or bottom of the 3D view), remembering its size. The strip is today's
  44 px icon strip with status dots. Hover (≥ 250 ms) or click opens a flyout over the 3D view at
  that edge, at the panel's remembered size, with a title bar (pin, close); leaving the flyout and
  strip for 500 ms, Esc, or clicking elsewhere hides it; a click on the strip icon keeps it open
  until the next click elsewhere. Pinning a flyout puts the panel back into the tree beside the 3D
  view on that edge. An edge with no hidden panels shows no strip.
- **D-11 Presets (D2):** One monitor (the D1 default); Two monitors (the default minus Activity and
  Graph, which go to a detached window on the second monitor — until D3 ships, or with one
  monitor, they stay tabbed with Log on the right, as in the default); Training (left column Model / Knowledge at 34 %, right
  column Activity over Graph at 30 %, View / Orbitals / Selection / Readouts / Log auto-hidden on the
  right); Presenting (every panel auto-hidden, the 3D view fills the window). Window ▸ Layouts lists
  them; Reset layout applies One monitor (or Two monitors when a second monitor is present, D3).
- **D-12 Detach (D3):** a leaf's pop-out button (or a tab's context menu "Move to new window")
  moves the panel(s) into a new OS window: a winit window on the main GPU device with its own
  surface, egui `Context`, egui-winit `State` and egui-wgpu renderer, holding its own `egui_dock`
  tree. It is drawn each time the main window draws; its panels read the same `UiState` and
  `PanelInput`, and their `Action`s are appended to the main frame's (so `--record` sees them).
  Tabs move between OS windows by the tab menu ("Move to main window", "Move to window N"), and by
  dragging since D-15. Closing a detached window docks its panels back into the main tree
  (each to the leaf it left, else beside the 3D view on the right; a panel with no recorded leaf,
  e.g. from a window read from a profile, docks beside its default-layout leaf-mates). The
  window's rectangle — absolute desktop pixels, its outer position and inner size — whether it
  is maximised, and its sub-tree are saved: `window = x,y,w,h[ max] <tree-without-3D>` (a
  maximised one reopens maximised on the monitor holding its centre); on load a window not
  usable on any connected monitor (less than 64 × 8 px of its title-bar strip, the top 32 px
  inset 8 px each side, on one) docks back instead. Ctrl K revealing a panel in a detached window
  focuses that window.
- **D-15 Dragging tabs between windows** (user request 2026-11-12): a tab dragged and released
  outside every one of the app's windows opens a new detached window (D-12) holding that tab, its
  top-left at the release point, at the panel's current size, kept on the monitor under the cursor.
  While a tab is dragged over another of the app's windows (main or detached), that window shows the
  same drop targets as a drag inside it: over a leaf, its centre tabs the panel in and its four edges
  split; over a tab bar, the tab is inserted. In the main window the 3D view's tile takes edge splits
  only (D-3). Releasing on a target moves the tab; releasing on no target, or Esc, cancels (the tab
  stays). This works main → detached, detached → main and detached → detached; a detached window
  whose last tab leaves closes. Inside one window the drag is `egui_dock`'s own, as before.
- **D-16 The 3D view never shares a drag with the UI** (user report 2026-11-12: resizing a panel near
  its edge also orbited the camera): a press that the UI takes — a widget, a resize grip or a
  separator, including the grab margin just outside their drawn edge — never moves the camera or
  starts a tool, and a drag the UI holds never reaches the 3D view.
- **D-13 Recording:** all of this is display state — nothing becomes an `Action`, nothing reaches
  the worker. The screenshot harness gains `--ui layout=<preset>` and keeps `tab=` (shows that
  panel) and `dock=closed` (Presenting).
- **D-14 Physics:** untouched. No `sim`/`qm`/`providers` change; the gating suite is
  `cargo test -p app --release`.

## Out of scope

Theming beyond the icon fill; touch input; per-scene layouts.

## Hand-testing (user)

D1: drag tabs between leaves, split by dropping on an edge, resize, close and reopen
via Window, every View setting still works and is found by Ctrl K, old profile opens sensibly.
D2: unpin / hover / pin back, each preset, layout survives restart. D3: pop out Activity + Graph to
the second monitor, restart (it comes back there), unplug the monitor (panels dock back); drag a
tab out of the main window (a new window opens there), drag tabs into it and back, onto an edge
(split) and its tab bar; resize panels right at their edges (the camera never moves).
