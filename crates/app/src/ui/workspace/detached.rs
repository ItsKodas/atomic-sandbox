//! Detached windows, the UI side (docking spec D-12, D3; user decision 6: "our own OS windows for
//! detaching, on the capture pop-out's pattern"; user request 2026-10-21: panels detachable to
//! another monitor, the second one holding Activity and Graph — the Two monitors layout, D-11).
//! Everything here is display state (spec D-13): nothing is an `Action`, nothing reaches the
//! worker. The OS windows themselves (winit window, surface, egui context and renderer) are
//! `crate::detached`'s; the UI cannot create them, so it asks through [`WindowRequest`]s that
//! `PanelOutput::windows` hands to the viewer and `main.rs` acts on.
//!
//! - **A detached window** holds its own `egui_dock` tree of panels (no 3D view) under an id
//!   (1, 2, … — "window N" in the tab menus). A leaf's pop-out button, or a tab's "Move to new
//!   window", moves the panel(s) into a new one ([`Workspace::pop_out`]); "Move to main window"
//!   and "Move to window N" move one tab between windows ([`Workspace::move_to_main`],
//!   [`Workspace::move_to_window`]), and so does dragging it onto another window or out of every
//!   window (spec D-15, user request 2026-11-12: `super::drag`).
//! - **Docking back** ([`Workspace::window_closed`], "Move to main window"): each panel goes to the
//!   leaf it left — the main-tree leaf holding any of the panels it shared its leaf with when it
//!   left ([`Windows::origins`]; a panel with no recorded origin, e.g. one from a window read from
//!   the profile, has the panels it shares its leaf with in the default layout stand in) — else (it
//!   was alone in its leaf, or that leaf is gone), together, into a new leaf right of the 3D view.
//! - **Where it is** ([`WinRect`]): the window's outer position and inner size in physical pixels
//!   on the desktop (winit's units, the monitors' too) and whether it is maximised, reported by the
//!   window side as it moves. The profile keeps one line per window, `window = x,y,w,h[ max]
//!   <tree>` ([`encode_window`]; the rect is left out while the window side has not reported one,
//!   e.g. headless); a window saved maximised reopens maximised on the monitor holding its centre.
//!   A window read back that is usable on no connected monitor ([`WinRect::usable_on`]: too little
//!   of its title-bar strip on any one; [`Workspace::set_monitors`], injected by the window side
//!   from `available_monitors`) docks back instead of opening, with a status note. The monitors'
//!   work areas ([`Workspace::set_work_areas`]) keep a tab torn out of every window on its monitor.
//! - **Requests** ([`Workspace::take_requests`]): each window not yet asked for is opened
//!   (`Open`), closing one asks `Close`, and showing a panel that lives in one (the Window menu,
//!   Ctrl K's reveal: `UiState::open_tab`) asks `Focus`.
use super::layout::{self, Node};
use super::tree::{build, first_leaf, node_of, Change, Tile, Workspace, VIEW_SHARE};
use crate::ui::DockTab;
use egui_dock::{DockState, NodeIndex, Split};
use std::collections::HashMap;

/// A window's place on the desktop: its outer position and inner size, physical pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WinRect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

/// The title-bar strip [`WinRect::usable_on`] looks at: the window's top `STRIP_H` px, `STRIP_INSET`
/// px in from each side (Windows reports a maximised window's outer position 8 px outside its
/// monitor, so a window maximised on the monitor to the right overhangs its left neighbour by 8 px:
/// review I1), of which at least `MIN_ON` × `MIN_ON_H` px must be on one monitor.
const STRIP_H: i64 = 32;
const STRIP_INSET: i64 = 8;
const MIN_ON: i64 = 64;
const MIN_ON_H: i64 = 8;

impl WinRect {
    /// Whether a window at `self` is usable on monitor `m`: at least 64 × 8 px of its title-bar
    /// strip (its top 32 px, 8 px in from each side) are on it — enough to see it and drag it.
    pub fn usable_on(&self, m: &WinRect) -> bool {
        let overlap = |a0: i64, a1: i64, b0: i64, b1: i64| (a1.min(b1) - a0.max(b0)).max(0);
        let (x, y, w, h) = (self.x as i64, self.y as i64, self.w as i64, self.h as i64);
        let (mx, my) = (m.x as i64, m.y as i64);
        overlap(x + STRIP_INSET, x + w - STRIP_INSET, mx, mx + m.w as i64) >= MIN_ON && overlap(y, y + STRIP_H.min(h), my, my + m.h as i64) >= MIN_ON_H
    }

    /// Whether `m` holds this rect's centre.
    fn centre_on(&self, m: &WinRect) -> bool {
        let (cx, cy) = (self.x as i64 + self.w as i64 / 2, self.y as i64 + self.h as i64 / 2);
        (m.x as i64..m.x as i64 + m.w as i64).contains(&cx) && (m.y as i64..m.y as i64 + m.h as i64).contains(&cy)
    }
}

/// What the UI asks of the window side (`main.rs`), in the order asked.
#[derive(Clone, Debug, PartialEq)]
pub enum WindowRequest {
    /// Open window `id` holding `panels` (its title): at `rect` when known (else where the OS puts
    /// it, `size` points large when given); `fill`: maximised on the monitor `rect` names (its
    /// work area, not full screen — the Two monitors layout).
    Open { id: u32, panels: Vec<DockTab>, rect: Option<WinRect>, size: Option<[f32; 2]>, fill: bool },
    /// Close window `id` (its panels have already gone back or away).
    Close { id: u32 },
    /// Bring window `id` to the front (a panel in it was asked for).
    Focus { id: u32 },
}

/// One detached window.
pub(super) struct Window {
    pub(super) id: u32,
    pub(super) dock: DockState<Tile>,
    pub(super) rect: Option<WinRect>,
    /// It is maximised (as the window side last reported it; the profile's `max`).
    max: bool,
    size: Option<[f32; 2]>,
    fill: bool,
    /// Its `Open` has been handed out.
    asked: bool,
}

/// The detached windows and their bookkeeping ([`Workspace::det`]).
#[derive(Default)]
pub(super) struct Windows {
    pub(super) list: Vec<Window>,
    /// The last id handed out (ids are never reused within a session).
    last: u32,
    requests: Vec<WindowRequest>,
    /// The connected monitors, the main window's first (`None`: not known — headless, or the OS
    /// listed none), and their work areas (the monitor less the taskbar), in the same order.
    monitors: Option<Vec<WinRect>>,
    work: Vec<WinRect>,
    /// The panels each detached panel shared its main-tree leaf with when it left.
    pub(super) origins: HashMap<DockTab, Vec<DockTab>>,
    /// Status lines for the user (a window docked back for want of a monitor).
    notes: Vec<String>,
}

/// A `window = …` line: where the window is (`None`: not reported), whether it is maximised, and
/// its tree without the 3D view.
pub type SavedWindow = (Option<WinRect>, bool, Node);

/// The `window = …` line's value (spec D-12): `x,y,w,h <tree>`, `x,y,w,h max <tree>` for a
/// maximised window, or just the tree when the window side has not reported where it is.
pub fn encode_window(rect: Option<WinRect>, max: bool, tree: &Node) -> String {
    match rect {
        Some(r) => format!("{},{},{},{}{} {}", r.x, r.y, r.w, r.h, if max { " max" } else { "" }, layout::encode(tree)),
        None => layout::encode(tree),
    }
}

/// Reads a `window = …` value: an optional `x,y,w,h` (a size of 0 reads as unknown), `max` after
/// it for a maximised window, then the window's tree (`layout::parse_window`). `None` when no
/// panel is left. A line without `max` (as written before the flag was saved) is not maximised.
pub fn parse_window(s: &str) -> Option<SavedWindow> {
    let s = s.trim();
    let (head, tail) = s.split_once(char::is_whitespace).unwrap_or((s, ""));
    let parts: Vec<&str> = head.split(',').map(str::trim).collect();
    if parts.len() == 4 && !head.contains(['[', '(']) {
        let rect = (|| {
            let r = WinRect { x: parts[0].parse().ok()?, y: parts[1].parse().ok()?, w: parts[2].parse().ok()?, h: parts[3].parse().ok()? };
            (r.w > 0 && r.h > 0).then_some(r)
        })();
        let tail = tail.trim_start();
        let (max, tail) = match tail.strip_prefix("max") {
            Some(rest) if rest.starts_with(char::is_whitespace) => (true, rest),
            _ => (false, tail),
        };
        return layout::parse_window(tail).map(|n| (rect, max && rect.is_some(), n));
    }
    layout::parse_window(s).map(|n| (None, false, n))
}

/// The panels `t` shares its leaf with in the default layout (the stand-in origin).
fn default_siblings(t: DockTab) -> Vec<DockTab> {
    fn walk(n: &Node, t: DockTab) -> Option<Vec<DockTab>> {
        match n {
            Node::View3d => None,
            Node::Leaf { tabs, .. } => tabs.contains(&t).then(|| tabs.iter().copied().filter(|&x| x != t).collect()),
            Node::Split { a, b, .. } => walk(a, t).or_else(|| walk(b, t)),
        }
    }
    walk(&layout::default_layout(), t).unwrap_or_default()
}

/// The panels among a leaf's tiles.
fn panels_of(tabs: &[Tile]) -> Vec<DockTab> {
    tabs.iter()
        .filter_map(|t| match t {
            Tile::Panel(p) => Some(*p),
            Tile::View3d => None,
        })
        .collect()
}

impl Workspace {
    /// The detached windows' ids, in the order they were opened.
    pub fn windows(&self) -> Vec<u32> {
        self.det.list.iter().map(|w| w.id).collect()
    }

    /// Window `id`'s tree (`None`: no such window, or it is empty).
    pub fn window_node(&self, id: u32) -> Option<Node> {
        let w = self.det.list.iter().find(|w| w.id == id)?;
        node_of(w.dock.main_surface(), NodeIndex::root()).and_then(layout::clean)
    }

    /// Window `id`'s panels in reading order.
    pub fn window_panels(&self, id: u32) -> Vec<DockTab> {
        self.window_node(id).map(|n| layout::panels(&n)).unwrap_or_default()
    }

    /// The window `t` is in.
    pub fn window_of(&self, t: DockTab) -> Option<u32> {
        self.det.list.iter().find(|w| w.dock.find_tab(&Tile::Panel(t)).is_some()).map(|w| w.id)
    }

    /// Window `id`'s title: its panels' names ("Activity · Graph — Atomic Sandbox").
    pub fn window_title(&self, id: u32) -> String {
        let names: Vec<&str> = self.window_panels(id).iter().map(|t| t.name()).collect();
        format!("{} — Atomic Sandbox", if names.is_empty() { "Panels".to_string() } else { names.join(" · ") })
    }

    /// Where window `id` is now and whether it is maximised (the window side reports it as the
    /// window moves; the profile saves it).
    pub fn set_window_rect(&mut self, id: u32, rect: Option<WinRect>, max: bool) {
        if let Some(w) = self.det.list.iter_mut().find(|w| w.id == id) {
            if rect.is_some() {
                w.rect = rect;
                w.max = max;
            }
        }
    }

    /// The connected monitors (physical pixels, the main window's first): windows read from a
    /// profile that are usable on none of them dock back instead of opening, and the Two monitors
    /// layout fills the second. An empty list is "not known" (as headless), not "no monitors":
    /// nothing docks back for it.
    pub fn set_monitors(&mut self, monitors: Vec<WinRect>) {
        self.det.monitors = (!monitors.is_empty()).then_some(monitors);
    }

    /// The monitors' work areas (physical pixels), in [`Workspace::set_monitors`]' order: a tab
    /// dragged out of every window opens its window inside the work area under the cursor.
    pub fn set_work_areas(&mut self, work: Vec<WinRect>) {
        self.det.work = work;
    }

    /// The work area of the monitor holding desktop point `(x, y)` (its whole rect when the work
    /// areas are not known; `None` when no monitor is, or none holds it).
    pub(super) fn work_area_at(&self, x: i32, y: i32) -> Option<WinRect> {
        let m = self.det.monitors.as_ref()?;
        let at = |r: &WinRect| (r.x as i64..r.x as i64 + r.w as i64).contains(&(x as i64)) && (r.y as i64..r.y as i64 + r.h as i64).contains(&(y as i64));
        let i = m.iter().position(at)?;
        Some(if self.det.work.len() == m.len() { self.det.work[i] } else { m[i] })
    }

    /// The number of connected monitors (one while unknown).
    pub fn monitor_count(&self) -> usize {
        self.det.monitors.as_ref().map_or(1, |m| m.len().max(1))
    }

    /// The second monitor (the first that is not the main window's).
    pub fn second_monitor(&self) -> Option<WinRect> {
        self.det.monitors.as_ref().and_then(|m| m.get(1)).copied()
    }

    /// The requests for the window side since the last call: an `Open` for each window not yet
    /// asked for (one read from a profile that is usable on no known monitor docks back instead,
    /// with a note; one saved maximised opens maximised on its monitor), then the queued `Close`s
    /// and `Focus`es.
    pub fn take_requests(&mut self) -> Vec<WindowRequest> {
        if self.lent.is_some() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut off = Vec::new();
        for w in self.det.list.iter_mut().filter(|w| !w.asked) {
            let on_none = match (&w.rect, &self.det.monitors) {
                (Some(r), Some(m)) => !w.fill && !m.iter().any(|m| r.usable_on(m)),
                _ => false,
            };
            if on_none {
                off.push(w.id);
                continue;
            }
            w.asked = true;
            // a window saved maximised reopens maximised on its monitor: the one holding its
            // centre, else one it is usable on (monitors unknown: its own rect, which the window
            // side maximises on whichever monitor that is)
            let (rect, fill) = match w.rect {
                Some(r) if w.max && !w.fill => {
                    let on = self.det.monitors.as_ref().and_then(|m| m.iter().find(|m| r.centre_on(m)).or_else(|| m.iter().find(|m| r.usable_on(m)))).copied();
                    (Some(on.unwrap_or(r)), true)
                }
                _ => (w.rect, w.fill),
            };
            let panels = node_of(w.dock.main_surface(), NodeIndex::root()).map(|n| layout::panels(&n)).unwrap_or_default();
            out.push(WindowRequest::Open { id: w.id, panels, rect, size: w.size, fill });
        }
        for id in off {
            let names = self.window_panels(id).iter().map(|t| t.name()).collect::<Vec<_>>().join(", ");
            self.dock_window_back(id);
            self.det.notes.push(format!("a detached window ({names}) was on no connected monitor: its panels are back in the main window"));
        }
        out.append(&mut self.det.requests);
        out
    }

    /// The status lines since the last call.
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.det.notes)
    }

    /// Moves `tabs` (where they are: the main tree, another detached window or a strip) into a
    /// new detached window, one leaf with `active` shown; returns its id (`None`
    /// while the tree is lent out — done when it is back — or when none of them is anywhere).
    pub fn pop_out(&mut self, tabs: &[DockTab], active: DockTab) -> Option<u32> {
        if self.lent.is_some() {
            self.pending.push(Change::PopOut(tabs.to_vec(), active));
            return None;
        }
        // the new window's size: the leaf's, when it has been drawn
        let size = tabs.first().and_then(|&t| {
            let r = self.leaf_rect(t)?;
            r.is_positive().then_some([r.width(), r.height()])
        });
        let moved: Vec<DockTab> = tabs.iter().copied().filter(|&t| self.take_out(t)).collect();
        if moved.is_empty() {
            return None;
        }
        let active = moved.iter().position(|&t| t == active).unwrap_or(0);
        Some(self.new_window(Node::Leaf { tabs: moved, active }, None, false, size, false))
    }

    /// Moves `tabs` from their detached windows back into the main tree, together (each to the
    /// leaf it left, else — the ones whose leaf is gone — into one new leaf beside the 3D view on
    /// the right), the last one shown; a window closes when that was its last panel.
    pub fn move_to_main(&mut self, tabs: &[DockTab]) {
        if self.lent.is_some() {
            self.pending.push(Change::ToMain(tabs.to_vec()));
            return;
        }
        let moved: Vec<DockTab> = tabs.iter().copied().filter(|&t| self.remove_from_window(t)).collect();
        if let Some(&last) = moved.last() {
            self.dock_back(moved);
            self.show_panel(last);
        }
    }

    /// Moves `t` (from the main tree, a strip or another window) into window `id`'s first leaf,
    /// shown there.
    pub fn move_to_window(&mut self, t: DockTab, id: u32) {
        if self.lent.is_some() {
            self.pending.push(Change::ToWindow(t, id));
            return;
        }
        if self.window_of(t) == Some(id) || !self.det.list.iter().any(|w| w.id == id) || !self.take_out(t) {
            return;
        }
        if let Some(w) = self.det.list.iter_mut().find(|w| w.id == id) {
            let tree = w.dock.main_surface_mut();
            match first_leaf(tree, NodeIndex::root()) {
                Some(leaf) => tree[leaf].append_tab(Tile::Panel(t)),
                None => w.dock = DockState::new(vec![Tile::Panel(t)]),
            }
        }
    }

    /// The OS window `id` was closed (spec D-12): its panels dock back into the main tree.
    pub fn window_closed(&mut self, id: u32) {
        if self.lent.is_some() {
            return; // never: the window side reports between frames
        }
        self.dock_window_back(id);
    }

    /// Closes every detached window, its panels dropped (a layout that places every panel is
    /// being applied). While a tree is lent out (Reset from a tab's menu) it is done when the tree
    /// is back, in order with the rest of the layout (review I2).
    pub fn close_windows(&mut self) {
        if self.lent.is_some() {
            self.pending.push(Change::CloseWindows);
            return;
        }
        for w in std::mem::take(&mut self.det.list) {
            if w.asked {
                self.det.requests.push(WindowRequest::Close { id: w.id });
            }
        }
        self.det.origins.clear();
    }

    /// Opens a new detached window holding `tree` (the Two monitors layout): filling the monitor
    /// `rect` when `fill`. Panels already elsewhere in the workspace keep their place. `None` when
    /// every panel is placed, or while a tree is lent out (done when it is back).
    pub fn open_window(&mut self, tree: Node, rect: Option<WinRect>, fill: bool) -> Option<u32> {
        if self.lent.is_some() {
            self.pending.push(Change::OpenWindow(tree, rect, fill));
            return None;
        }
        let tree = self.unplaced(tree)?;
        Some(self.new_window(tree, rect, false, None, fill))
    }

    /// Replaces the detached windows with the profile's (`window = …` lines, in order): the old
    /// ones close, a panel already in the main tree, on a strip or in an earlier window keeps that
    /// place, a window left empty is dropped.
    pub fn set_windows(&mut self, list: Vec<SavedWindow>) {
        if self.lent.is_some() {
            self.pending.push(Change::SetWindows(list));
            return;
        }
        self.close_windows();
        for (rect, max, tree) in list {
            if let Some(tree) = self.unplaced(tree) {
                self.new_window(tree, rect, max, None, false);
            }
        }
    }

    /// The windows as the profile saves them: each one's rect, maximised flag and tree, in order.
    pub fn saved_windows(&self) -> Vec<SavedWindow> {
        self.det.list.iter().filter_map(|w| Some((w.rect, w.max, self.window_node(w.id)?))).collect()
    }

    /// Takes the panels of `n` already placed (the main tree, a strip, a window) out of it.
    fn unplaced(&self, n: Node) -> Option<Node> {
        fn keep(n: Node, placed: &dyn Fn(DockTab) -> bool) -> Option<Node> {
            match n {
                Node::View3d => None,
                Node::Leaf { tabs, active } => {
                    let shown = tabs.get(active).copied();
                    let kept: Vec<DockTab> = tabs.into_iter().filter(|&t| !placed(t)).collect();
                    let active = shown.and_then(|s| kept.iter().position(|&t| t == s)).unwrap_or(0);
                    (!kept.is_empty()).then_some(Node::Leaf { tabs: kept, active })
                }
                Node::Split { horizontal, frac, a, b } => match (keep(*a, placed), keep(*b, placed)) {
                    (Some(a), Some(b)) => Some(Node::Split { horizontal, frac, a: Box::new(a), b: Box::new(b) }),
                    (Some(one), None) | (None, Some(one)) => Some(one),
                    (None, None) => None,
                },
            }
        }
        keep(layout::clean(n)?, &|t| self.present(t))
    }

    pub(super) fn new_window(&mut self, tree: Node, rect: Option<WinRect>, max: bool, size: Option<[f32; 2]>, fill: bool) -> u32 {
        self.det.last += 1;
        let id = self.det.last;
        self.det.list.push(Window { id, dock: build(&tree), rect, max, size, fill, asked: false });
        id
    }

    /// The rect of the leaf `t` is drawn in (main tree or a detached window), as last drawn.
    pub(crate) fn leaf_rect(&self, t: DockTab) -> Option<egui::Rect> {
        let find = |d: &DockState<Tile>| d.find_tab(&Tile::Panel(t)).and_then(|p| d.leaf(p.node_path()).ok().map(|l| l.rect));
        find(&self.dock).or_else(|| self.det.list.iter().find_map(|w| find(&w.dock)))
    }

    /// Takes `t` out of wherever it is (the main tree — remembering the panels it shared its leaf
    /// with —, a detached window, a strip). False when it was closed.
    pub(super) fn take_out(&mut self, t: DockTab) -> bool {
        if let Some(p) = self.dock.find_tab(&Tile::Panel(t)) {
            let mates = self.dock.leaf(p.node_path()).map(|l| panels_of(&l.tabs)).unwrap_or_default();
            self.det.origins.insert(t, mates.into_iter().filter(|&x| x != t).collect());
            self.dock.remove_tab(p);
            return true;
        }
        if self.remove_from_window(t) {
            return true;
        }
        if self.is_hidden(t) {
            self.hidden.retain(|h| h.tab != t);
            if self.fly.open == Some(t) {
                self.fly.close();
            }
            return true;
        }
        false
    }

    /// Takes `t` out of its detached window (which closes when it was its last panel).
    pub(super) fn remove_from_window(&mut self, t: DockTab) -> bool {
        let Some(i) = self.det.list.iter().position(|w| w.dock.find_tab(&Tile::Panel(t)).is_some()) else { return false };
        let w = &mut self.det.list[i];
        if let Some(p) = w.dock.find_tab(&Tile::Panel(t)) {
            w.dock.remove_tab(p);
        }
        self.close_if_empty(i);
        true
    }

    /// Takes every panel of `panels` out of the detached windows (a new main tree holds them).
    pub(super) fn drop_from_windows(&mut self, panels: &[DockTab]) {
        for &t in panels {
            self.remove_from_window(t);
        }
    }

    /// Closes the window at `i` when it holds no panel.
    fn close_if_empty(&mut self, i: usize) {
        if self.det.list[i].dock.iter_all_tabs().next().is_none() {
            let w = self.det.list.remove(i);
            if w.asked {
                self.det.requests.push(WindowRequest::Close { id: w.id });
            }
        }
    }

    /// Closes every empty detached window (after one was drawn: its last tab may have gone).
    pub(super) fn close_empty_windows(&mut self) {
        let mut i = 0;
        while i < self.det.list.len() {
            let before = self.det.list.len();
            self.close_if_empty(i);
            if self.det.list.len() == before {
                i += 1;
            }
        }
    }

    /// Window `id` goes and its panels dock back (no `Close`: the window side closed it or never
    /// opened it).
    fn dock_window_back(&mut self, id: u32) {
        let Some(i) = self.det.list.iter().position(|w| w.id == id) else { return };
        let panels = self.window_panels(id);
        self.det.list.remove(i);
        self.dock_back(panels);
    }

    /// Docks `tabs` back into the main tree, each into the leaf holding any panel it left beside
    /// (or, with no recorded origin, shares a leaf with in the default layout); the rest — a panel
    /// that was alone in its leaf, or whose leaf-mates are all gone — together into a new leaf
    /// right of the 3D view (spec D-12).
    fn dock_back(&mut self, tabs: Vec<DockTab>) {
        let mut rest = Vec::new();
        for t in tabs {
            if self.present(t) {
                continue;
            }
            // a known origin, even an empty one (it was alone in its leaf), is never replaced by
            // the default layout's leaf-mates (review M1)
            let mates = match self.det.origins.remove(&t) {
                Some(m) => m,
                None => default_siblings(t),
            };
            let leaf = self
                .dock
                .iter_leaves()
                .find(|(_, l)| l.tabs.iter().any(|x| matches!(x, Tile::Panel(p) if mates.contains(p))))
                .map(|(p, _)| p);
            match leaf.and_then(|p| self.dock.leaf_mut(p).ok()) {
                Some(l) => l.append_tab(Tile::Panel(t)),
                None => rest.push(t),
            }
        }
        if rest.is_empty() {
            return;
        }
        let main = self.dock.main_surface_mut();
        let Some((view, _)) = main.find_tab(&Tile::View3d) else { return };
        let tiles = rest.into_iter().map(Tile::Panel).collect();
        main.split(view, Split::Right, VIEW_SHARE, egui_dock::Node::leaf_with(tiles));
    }

    /// Shows `t` in its detached window (its leaf's active tab) and asks for the window to come
    /// to the front. False when it is in none.
    pub(super) fn show_in_window(&mut self, t: DockTab) -> bool {
        let Some(w) = self.det.list.iter_mut().find(|w| w.dock.find_tab(&Tile::Panel(t)).is_some()) else { return false };
        if let Some(p) = w.dock.find_tab(&Tile::Panel(t)) {
            let _ = w.dock.set_active_tab(p);
            w.dock.set_focused_node_and_surface(p.node_path());
        }
        let id = w.id;
        self.det.requests.push(WindowRequest::Focus { id });
        true
    }

    /// Whether `t` is the shown tab of its leaf in a detached window.
    pub(super) fn visible_in_window(&self, t: DockTab) -> bool {
        self.det.list.iter().any(|w| w.dock.find_tab(&Tile::Panel(t)).is_some_and(|p| w.dock.leaf(p.node_path()).is_ok_and(|l| l.active == p.tab)))
    }

    /// Lends window `id`'s tree out for its frame (`None`: no such window, or a tree is out).
    pub(super) fn lend_window(&mut self, id: u32) -> Option<DockState<Tile>> {
        if self.lent.is_some() {
            return None;
        }
        let i = self.det.list.iter().position(|w| w.id == id)?;
        self.lent = Some(self.snapshot());
        Some(std::mem::replace(&mut self.det.list[i].dock, DockState::new(Vec::new())))
    }

    /// Window `id`'s tree back from its frame, then what was asked meanwhile done; a window left
    /// empty closes.
    pub(super) fn give_back_window(&mut self, id: u32, dock: DockState<Tile>, asked: Vec<Change>) {
        if let Some(w) = self.det.list.iter_mut().find(|w| w.id == id) {
            w.dock = dock;
        }
        self.lent = None;
        let changes: Vec<Change> = std::mem::take(&mut self.pending).into_iter().chain(asked).collect();
        self.apply_changes(changes);
        self.close_empty_windows();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::workspace::presets::Preset;
    use DockTab::*;

    fn opens(r: &[WindowRequest]) -> Vec<(u32, Vec<DockTab>)> {
        r.iter()
            .filter_map(|r| match r {
                WindowRequest::Open { id, panels, .. } => Some((*id, panels.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn popping_out_activity_moves_it_to_a_window_and_closing_returns_it() {
        let mut w = Workspace::default();
        let before = layout::encode(&w.to_node());
        let id = w.pop_out(&[Activity], Activity).expect("a window");
        assert!(!layout::panels(&w.to_node()).contains(&Activity), "out of the main tree");
        assert_eq!(w.window_panels(id), vec![Activity]);
        assert_eq!((w.window_of(Activity), w.windows()), (Some(id), vec![id]));
        assert!(w.present(Activity) && w.visible(Activity), "in the workspace and shown");
        assert_eq!(opens(&w.take_requests()), vec![(id, vec![Activity])]);
        assert!(w.take_requests().is_empty(), "asked once");
        assert_eq!(w.window_title(id), "Activity — Atomic Sandbox");
        // the OS window closed: back to the leaf it left (beside Log and Graph), shown
        w.window_closed(id);
        assert!(w.windows().is_empty());
        assert_eq!(layout::encode(&w.to_node()), before.replace("[Log*,Activity,Graph]", "[Log,Graph,Activity*]"));
        assert!(w.take_requests().is_empty(), "no Close: the window side closed it");
    }

    #[test]
    fn a_whole_leaf_pops_out_and_docks_back_beside_the_view() {
        let mut w = Workspace::default();
        let id = w.pop_out(&[View, Orbitals], Orbitals).unwrap();
        assert_eq!(layout::encode(&w.window_node(id).unwrap()), "[View,Orbitals*]");
        assert!(w.visible(Orbitals) && !w.visible(View));
        // its leaf is gone: both come back together right of the 3D view
        w.window_closed(id);
        assert_eq!(layout::encode(&w.to_node()), "H(0.24,V(0.55,[Model*,Knowledge],[Selection*,Readouts]),H(0.70,H(0.70,3D,[View*,Orbitals]),[Log*,Activity,Graph]))");
    }

    #[test]
    fn tabs_move_between_windows_and_back_by_the_menu() {
        let mut w = Workspace::default();
        let a = w.pop_out(&[Activity], Activity).unwrap();
        let b = w.pop_out(&[Graph], Graph).unwrap();
        assert_eq!(opens(&w.take_requests()).len(), 2);
        w.move_to_window(Graph, a);
        assert_eq!(w.window_panels(a), vec![Activity, Graph]);
        assert_eq!(w.windows(), vec![a], "window {b} emptied and closed");
        assert_eq!(w.take_requests(), vec![WindowRequest::Close { id: b }]);
        w.move_to_window(Model, a);
        assert_eq!(w.window_panels(a), vec![Activity, Graph, Model]);
        w.move_to_main(&[Model]);
        assert_eq!(w.window_of(Model), None);
        assert!(layout::encode(&w.to_node()).contains("[Knowledge,Model*]"), "back beside Knowledge, shown: {}", layout::encode(&w.to_node()));
        w.move_to_main(&[Activity, Graph]);
        assert!(w.windows().is_empty());
        assert_eq!(w.take_requests(), vec![WindowRequest::Close { id: a }]);
        assert_eq!(layout::panels(&w.to_node()).len(), DockTab::PANELS.len(), "every panel back");
    }

    #[test]
    fn showing_a_detached_panel_focuses_its_window_and_closing_its_last_panel_closes_it() {
        let mut w = Workspace::default();
        let id = w.pop_out(&[Activity, Graph], Activity).unwrap();
        w.take_requests();
        w.show_panel(Graph);
        assert!(w.visible(Graph) && !w.visible(Activity));
        assert_eq!(w.take_requests(), vec![WindowRequest::Focus { id }]);
        w.close_panel(Graph);
        w.close_panel(Activity);
        assert!(!w.present(Activity) && w.windows().is_empty());
        assert_eq!(w.take_requests(), vec![WindowRequest::Close { id }]);
        // unpinning a detached panel puts it on the main window's right strip
        let id = w.pop_out(&[Log], Log).unwrap();
        w.pin(Log, false);
        assert!(w.windows().is_empty() && w.hidden().iter().any(|h| h.tab == Log && h.edge == layout::Edge::Right));
        assert!(!w.take_requests().iter().any(|r| *r == WindowRequest::Close { id }), "never opened: nothing to close");
    }

    #[test]
    fn a_new_main_tree_takes_its_panels_out_of_the_windows() {
        let mut w = Workspace::default();
        let id = w.pop_out(&[Activity], Activity).unwrap();
        w.take_requests();
        w.set_node(layout::default_layout());
        assert!(w.windows().is_empty() && w.window_of(Activity).is_none());
        assert_eq!(w.take_requests(), vec![WindowRequest::Close { id }]);
        // a hidden list never takes a detached panel
        let id = w.pop_out(&[Graph], Graph).unwrap();
        w.set_hidden(vec![layout::Hidden { tab: Graph, edge: layout::Edge::Left, size: 300.0 }]);
        assert_eq!((w.window_of(Graph), w.hidden().len()), (Some(id), 0));
    }

    #[test]
    fn the_window_line_round_trips() {
        let tree = layout::parse_window("V(0.50,[Activity*],[Graph*])").unwrap();
        let r = WinRect { x: 1920, y: 0, w: 1920, h: 1040 };
        let line = encode_window(Some(r), false, &tree);
        assert_eq!(line, "1920,0,1920,1040 V(0.50,[Activity*],[Graph*])");
        assert_eq!(parse_window(&line), Some((Some(r), false, tree.clone())));
        // maximised: `max` after the rect (review M4)
        let max = WinRect { x: 2552, y: -8, w: 1920, h: 1009 };
        let line = encode_window(Some(max), true, &tree);
        assert_eq!(line, "2552,-8,1920,1009 max V(0.50,[Activity*],[Graph*])");
        assert_eq!(parse_window(&line), Some((Some(max), true, tree.clone())));
        assert_eq!(parse_window("2552,-8,1920,1009   max   [Log*]"), Some((Some(max), true, layout::parse_window("[Log*]").unwrap())));
        assert_eq!(parse_window("-1280,100,1280,1000   [Log*]"), Some((Some(WinRect { x: -1280, y: 100, w: 1280, h: 1000 }), false, layout::parse_window("[Log*]").unwrap())));
        assert_eq!(encode_window(None, true, &tree), "V(0.50,[Activity*],[Graph*])", "no rect: no max");
        assert_eq!(parse_window("V(0.50,[Activity*],[Graph*])"), Some((None, false, tree)));
        assert_eq!(parse_window("0,0,0,0 [Log]").map(|(r, m, _)| (r, m)), Some((None, false)), "a zero size reads as unknown");
        assert_eq!(parse_window("0,0,0,0 max [Log]").map(|(r, m, _)| (r, m)), Some((None, false)), "max means nothing without a rect");
        for bad in ["", "1,2,3,4", "1,2,3,4 3D", "1,2,3,4 max", "1,2,3,4 maximal [Log]", "x,y,w,h [Bogus]", "nonsense"] {
            assert_eq!(parse_window(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_window_off_every_monitor_docks_back_and_one_on_a_monitor_opens() {
        let primary = WinRect { x: 0, y: 0, w: 2560, h: 1440 };
        let second = WinRect { x: 2560, y: 0, w: 1920, h: 1080 };
        let mut w = Workspace::default();
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(vec![
            (Some(WinRect { x: 2600, y: 40, w: 800, h: 900 }), false, layout::parse_window("[Activity*]").unwrap()),
            (Some(WinRect { x: 5000, y: 40, w: 800, h: 900 }), false, layout::parse_window("[Graph*]").unwrap()),
        ]);
        assert_eq!(w.windows().len(), 2);
        w.set_monitors(vec![primary, second]);
        let r = w.take_requests();
        assert_eq!(opens(&r).iter().map(|o| o.1.clone()).collect::<Vec<_>>(), vec![vec![Activity]], "only the one on a monitor opens");
        assert_eq!(w.window_of(Graph), None);
        assert!(layout::encode(&w.to_node()).contains("[Log,Graph*]"), "Graph docked back beside Log, its default leaf-mate: {}", layout::encode(&w.to_node()));
        assert_eq!(w.take_notes().len(), 1);
        // unplug the second monitor: the next start docks Activity back too
        let saved = w.saved_windows();
        let mut next = Workspace::default();
        next.set_node(w.to_node());
        next.set_windows(saved);
        next.set_monitors(vec![primary]);
        assert!(opens(&next.take_requests()).is_empty() && next.windows().is_empty());
        assert!(layout::panels(&next.to_node()).contains(&Activity));
    }

    /// A window saved maximised (`max`; the Two monitors window: its rect as Windows reports a
    /// maximised window on a 1920 × 1080 monitor, measured 2026-11-03) reopens maximised on the
    /// monitor holding its centre; one not maximised at its own rect, even when it spans the
    /// monitor (review M4: the flag, not a guess from the rect — an old line without `max` reads
    /// as not maximised).
    #[test]
    fn a_window_saved_maximised_reopens_maximised() {
        let second = WinRect { x: 2560, y: 0, w: 1920, h: 1080 };
        let mut w = Workspace::default();
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(vec![
            (Some(WinRect { x: 2552, y: -8, w: 1920, h: 1009 }), true, layout::parse_window("[Activity*]").unwrap()),
            (Some(WinRect { x: 2600, y: 40, w: 800, h: 900 }), false, layout::parse_window("[Graph*]").unwrap()),
            (Some(WinRect { x: 2552, y: -8, w: 1920, h: 1009 }), false, layout::parse_window("[Model*]").unwrap()),
        ]);
        w.set_monitors(vec![WinRect { x: 0, y: 0, w: 2560, h: 1440 }, second]);
        let r = w.take_requests();
        assert!(matches!(r[0], WindowRequest::Open { rect: Some(m), fill: true, .. } if m == second), "{r:?}");
        assert!(matches!(r[1], WindowRequest::Open { rect: Some(WinRect { x: 2600, .. }), fill: false, .. }), "{r:?}");
        assert!(matches!(r[2], WindowRequest::Open { rect: Some(WinRect { x: 2552, .. }), fill: false, .. }), "{r:?}");
        // the window side reports it restored: saved without `max`
        let id = w.windows()[0];
        w.set_window_rect(id, Some(WinRect { x: 2700, y: 100, w: 900, h: 700 }), false);
        assert!(!w.saved_windows()[0].1);
        w.set_window_rect(id, None, true);
        assert!(!w.saved_windows()[0].1, "an unknown rect changes nothing");
    }

    /// Review I1: a window maximised on the monitor right of the primary (`2552,-8,1920,1009`:
    /// 8 px of its frame overhang the primary) with that monitor unplugged docks back — its
    /// title bar is not on the primary; with both monitors it opens, maximised on the second.
    #[test]
    fn a_window_maximised_on_an_unplugged_monitor_docks_back() {
        let primary = WinRect { x: 0, y: 0, w: 2560, h: 1440 };
        let second = WinRect { x: 2560, y: 0, w: 1920, h: 1080 };
        let saved = || vec![(Some(WinRect { x: 2552, y: -8, w: 1920, h: 1009 }), true, layout::parse_window("V(0.50,[Activity*],[Graph*])").unwrap())];
        let mut w = Workspace::default();
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(saved());
        w.set_monitors(vec![primary]);
        assert!(opens(&w.take_requests()).is_empty() && w.windows().is_empty(), "docked back");
        assert!(w.present(Activity) && w.present(Graph) && w.window_of(Activity).is_none());
        assert_eq!(w.take_notes().len(), 1);
        // both monitors connected: it stays detached
        let mut both = Workspace::default();
        both.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        both.set_windows(saved());
        both.set_monitors(vec![primary, second]);
        let r = both.take_requests();
        assert!(matches!(r.as_slice(), [WindowRequest::Open { rect: Some(m), fill: true, .. }] if *m == second), "{r:?}");
        assert!(both.window_of(Activity).is_some() && both.take_notes().is_empty());
    }

    /// Review M2: an empty monitor list (the OS listed none) is "not known", as headless: a saved
    /// window opens instead of docking back, and there is one monitor and no second.
    #[test]
    fn an_empty_monitor_list_is_unknown() {
        let mut w = Workspace::default();
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(vec![(Some(WinRect { x: 9000, y: 40, w: 800, h: 900 }), false, layout::parse_window("[Graph*]").unwrap())]);
        w.set_monitors(vec![]);
        assert_eq!(opens(&w.take_requests()).len(), 1);
        assert!(w.take_notes().is_empty());
        assert_eq!((w.monitor_count(), w.second_monitor()), (1, None));
    }

    /// Review M1: a panel that was alone in its leaf docks back beside the 3D view on the right
    /// (spec D-12), not next to its default-layout leaf-mates; with no recorded origin (a window
    /// from a profile) the default-layout leaf-mates stand in.
    #[test]
    fn a_panel_alone_in_its_leaf_docks_back_beside_the_view() {
        let mut w = Workspace::default();
        w.set_node(layout::parse("H(0.70,3D,V(0.50,[Log*],[Graph*]))").unwrap());
        let id = w.pop_out(&[Graph], Graph).unwrap();
        w.window_closed(id);
        assert_eq!(layout::encode(&w.to_node()), "H(0.70,H(0.70,3D,[Graph*]),[Log*])");
        // the same window read from a profile: Graph has no origin, so beside Log (its default leaf-mate)
        let mut p = Workspace::default();
        p.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        p.set_windows(vec![(None, false, layout::parse_window("[Graph*]").unwrap())]);
        let id = p.windows()[0];
        p.window_closed(id);
        assert_eq!(layout::encode(&p.to_node()), "H(0.70,3D,[Log,Graph*])");
    }

    #[test]
    fn set_windows_keeps_each_panel_once() {
        let mut w = Workspace::default(); // every panel in the main tree
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(vec![
            (None, false, layout::parse_window("V(0.5,[Activity*,Log],[Graph])").unwrap()),
            (None, false, layout::parse_window("[Graph*,Model]").unwrap()),
            (None, false, layout::parse_window("[Log*]").unwrap()),
        ]);
        let ids = w.windows();
        assert_eq!(ids.len(), 2, "the third held only Log, already placed");
        assert_eq!(layout::encode(&w.window_node(ids[0]).unwrap()), "V(0.50,[Activity*],[Graph*])");
        assert_eq!(layout::encode(&w.window_node(ids[1]).unwrap()), "[Model*]");
    }

    #[test]
    fn the_two_monitors_window_holds_activity_over_graph() {
        let (tree, hidden) = Preset::TwoMonitors.tree();
        assert!(hidden.is_empty());
        let win = Preset::TwoMonitors.window().unwrap();
        assert_eq!(layout::encode(&win), "V(0.50,[Activity*],[Graph*])");
        let mut all = layout::panels(&tree);
        all.extend(layout::panels(&win));
        assert_eq!(all.len(), DockTab::PANELS.len());
        assert!(DockTab::PANELS.iter().all(|t| all.contains(t)));
    }

    /// A window is usable on a monitor when 64 × 8 px of its title-bar strip (top 32 px, 8 px in
    /// from each side) are on it.
    #[test]
    fn a_window_is_usable_where_its_title_bar_is() {
        let m = WinRect { x: 0, y: 0, w: 1000, h: 800 };
        let at = |x, y| WinRect { x, y, w: 400, h: 300 };
        assert!(at(100, 100).usable_on(&m));
        assert!(at(-8, -8).usable_on(&m), "maximised-style overhang");
        assert!(at(1000 - 8 - 64, 0).usable_on(&m) && !at(1000 - 8 - 63, 0).usable_on(&m), "64 px of the strip, inset 8");
        assert!(at(-400 + 8 + 64, 0).usable_on(&m) && !at(-400 + 8 + 63, 0).usable_on(&m));
        assert!(at(100, 800 - 8).usable_on(&m) && !at(100, 800 - 7).usable_on(&m), "8 px of its height");
        assert!(!at(100, -32 + 7).usable_on(&m), "its title bar above the monitor");
        assert!(!at(1000, 0).usable_on(&m) && !at(-400, 0).usable_on(&m), "beside it");
    }
}
