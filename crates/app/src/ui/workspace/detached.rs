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
//!   [`Workspace::move_to_window`]) — tabs never cross windows by dragging (spec D-12).
//! - **Docking back** ([`Workspace::window_closed`], "Move to main window"): each panel goes to the
//!   leaf it left — the main-tree leaf holding any of the panels it shared its leaf with when it
//!   left ([`Windows::origins`]; a window read from the profile has none, so the panels it shared
//!   its leaf with in the default layout stand in, recorded for review) — else, together, into a
//!   new leaf right of the 3D view.
//! - **Where it is** ([`WinRect`]): the window's outer position and inner size in physical pixels
//!   on the desktop (winit's units, the monitors' too), reported by the window side as it moves.
//!   The profile keeps one line per window, `window = x,y,w,h <tree>` ([`encode_window`]; the rect
//!   is left out while the window side has not reported one, e.g. headless). A window read back
//!   whose rect is on no connected monitor ([`Workspace::set_monitors`], injected by the window side
//!   from `available_monitors`) docks back instead of opening, with a status note.
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

impl WinRect {
    /// Whether the two overlap by at least a pixel.
    pub fn intersects(&self, o: &WinRect) -> bool {
        let (ax, ay, bx, by) = (self.x as i64, self.y as i64, o.x as i64, o.y as i64);
        ax < bx + o.w as i64 && bx < ax + self.w as i64 && ay < by + o.h as i64 && by < ay + self.h as i64
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
    /// The connected monitors, the main window's first (`None`: not known — headless).
    monitors: Option<Vec<WinRect>>,
    /// The panels each detached panel shared its main-tree leaf with when it left.
    origins: HashMap<DockTab, Vec<DockTab>>,
    /// Status lines for the user (a window docked back for want of a monitor).
    notes: Vec<String>,
}

/// The `window = …` line's value (spec D-12): `x,y,w,h <tree>`, or just the tree when the window
/// side has not reported where it is.
pub fn encode_window(rect: Option<WinRect>, tree: &Node) -> String {
    match rect {
        Some(r) => format!("{},{},{},{} {}", r.x, r.y, r.w, r.h, layout::encode(tree)),
        None => layout::encode(tree),
    }
}

/// Reads a `window = …` value: an optional `x,y,w,h` (a size of 0 reads as unknown) and the
/// window's tree (`layout::parse_window`). `None` when no panel is left.
pub fn parse_window(s: &str) -> Option<(Option<WinRect>, Node)> {
    let s = s.trim();
    let (head, tail) = s.split_once(char::is_whitespace).unwrap_or((s, ""));
    let parts: Vec<&str> = head.split(',').map(str::trim).collect();
    if parts.len() == 4 && !head.contains(['[', '(']) {
        let rect = (|| {
            let r = WinRect { x: parts[0].parse().ok()?, y: parts[1].parse().ok()?, w: parts[2].parse().ok()?, h: parts[3].parse().ok()? };
            (r.w > 0 && r.h > 0).then_some(r)
        })();
        return layout::parse_window(tail).map(|n| (rect, n));
    }
    layout::parse_window(s).map(|n| (None, n))
}

/// Whether a window at `r` was maximised on monitor `m`: it spans the monitor's width (a
/// maximised window's frame overhangs each side by a few pixels, so its outer position is a little
/// left of the monitor's) and at least 85 % of its height (the taskbar and the title bar take the
/// rest), its centre on it.
fn fills(r: &WinRect, m: &WinRect) -> bool {
    let (cx, cy) = (r.x as i64 + r.w as i64 / 2, r.y as i64 + r.h as i64 / 2);
    let inside = (m.x as i64..m.x as i64 + m.w as i64).contains(&cx) && (m.y as i64..m.y as i64 + m.h as i64).contains(&cy);
    inside && r.w >= m.w && r.h * 100 >= m.h * 85
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

    /// Where window `id` is now (the window side reports it as the window moves; the profile
    /// saves it).
    pub fn set_window_rect(&mut self, id: u32, rect: Option<WinRect>) {
        if let Some(w) = self.det.list.iter_mut().find(|w| w.id == id) {
            if rect.is_some() {
                w.rect = rect;
            }
        }
    }

    /// The connected monitors (physical pixels, the main window's first): windows read from a
    /// profile that are on none of them dock back instead of opening, and the Two monitors layout
    /// fills the second.
    pub fn set_monitors(&mut self, monitors: Vec<WinRect>) {
        self.det.monitors = Some(monitors);
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
    /// asked for (one read from a profile whose rect is on no known monitor docks back instead,
    /// with a note), then the queued `Close`s and `Focus`es.
    pub fn take_requests(&mut self) -> Vec<WindowRequest> {
        if self.lent.is_some() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut off = Vec::new();
        for w in self.det.list.iter_mut().filter(|w| !w.asked) {
            let on_none = match (&w.rect, &self.det.monitors) {
                (Some(r), Some(m)) => !w.fill && !m.iter().any(|m| r.intersects(m)),
                _ => false,
            };
            if on_none {
                off.push(w.id);
                continue;
            }
            w.asked = true;
            // a window saved maximised (its rect fills a monitor's width) reopens maximised there
            let filled = match (&w.rect, &self.det.monitors) {
                (Some(r), Some(m)) if !w.fill => m.iter().copied().find(|m| fills(r, m)),
                _ => None,
            };
            let (rect, fill) = match filled {
                Some(m) => (Some(m), true),
                None => (w.rect, w.fill),
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

    /// Moves `tabs` (where they are: the main tree, a floating window, another detached window or
    /// a strip) into a new detached window, one leaf with `active` shown; returns its id (`None`
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
        Some(self.new_window(Node::Leaf { tabs: moved, active }, None, size, false))
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
    /// being applied).
    pub fn close_windows(&mut self) {
        for w in std::mem::take(&mut self.det.list) {
            if w.asked {
                self.det.requests.push(WindowRequest::Close { id: w.id });
            }
        }
        self.det.origins.clear();
    }

    /// Opens a new detached window holding `tree` (the Two monitors layout): filling the monitor
    /// `rect` when `fill`. Panels already elsewhere in the workspace keep their place.
    pub fn open_window(&mut self, tree: Node, rect: Option<WinRect>, fill: bool) -> Option<u32> {
        let tree = self.unplaced(tree)?;
        Some(self.new_window(tree, rect, None, fill))
    }

    /// Replaces the detached windows with the profile's (`window = …` lines, in order): the old
    /// ones close, a panel already in the main tree, on a strip or in an earlier window keeps that
    /// place, a window left empty is dropped.
    pub fn set_windows(&mut self, list: Vec<(Option<WinRect>, Node)>) {
        self.close_windows();
        for (rect, tree) in list {
            if let Some(tree) = self.unplaced(tree) {
                self.new_window(tree, rect, None, false);
            }
        }
    }

    /// The windows as the profile saves them: each one's rect and tree, in order.
    pub fn saved_windows(&self) -> Vec<(Option<WinRect>, Node)> {
        self.det.list.iter().filter_map(|w| Some((w.rect, self.window_node(w.id)?))).collect()
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

    fn new_window(&mut self, tree: Node, rect: Option<WinRect>, size: Option<[f32; 2]>, fill: bool) -> u32 {
        self.det.last += 1;
        let id = self.det.last;
        self.det.list.push(Window { id, dock: build(&tree), rect, size, fill, asked: false });
        id
    }

    /// The rect of the leaf `t` is drawn in (main tree or a detached window), as last drawn.
    fn leaf_rect(&self, t: DockTab) -> Option<egui::Rect> {
        let find = |d: &DockState<Tile>| d.find_tab(&Tile::Panel(t)).and_then(|p| d.leaf(p.node_path()).ok().map(|l| l.rect));
        find(&self.dock).or_else(|| self.det.list.iter().find_map(|w| find(&w.dock)))
    }

    /// Takes `t` out of wherever it is (the main tree or a floating window — remembering the
    /// panels it shared its leaf with —, a detached window, a strip). False when it was closed.
    fn take_out(&mut self, t: DockTab) -> bool {
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
    /// (or, unknown, shares a leaf with in the default layout); the rest together into a new leaf
    /// right of the 3D view.
    fn dock_back(&mut self, tabs: Vec<DockTab>) {
        let mut rest = Vec::new();
        for t in tabs {
            if self.present(t) {
                continue;
            }
            let mates = self.det.origins.remove(&t).filter(|m| !m.is_empty()).unwrap_or_else(|| default_siblings(t));
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
        let line = encode_window(Some(r), &tree);
        assert_eq!(line, "1920,0,1920,1040 V(0.50,[Activity*],[Graph*])");
        assert_eq!(parse_window(&line), Some((Some(r), tree.clone())));
        assert_eq!(parse_window("-1280,100,1280,1000   [Log*]"), Some((Some(WinRect { x: -1280, y: 100, w: 1280, h: 1000 }), layout::parse_window("[Log*]").unwrap())));
        assert_eq!(encode_window(None, &tree), "V(0.50,[Activity*],[Graph*])");
        assert_eq!(parse_window("V(0.50,[Activity*],[Graph*])"), Some((None, tree)));
        assert_eq!(parse_window("0,0,0,0 [Log]").map(|(r, _)| r), Some(None), "a zero size reads as unknown");
        for bad in ["", "1,2,3,4", "1,2,3,4 3D", "x,y,w,h [Bogus]", "nonsense"] {
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
            (Some(WinRect { x: 2600, y: 40, w: 800, h: 900 }), layout::parse_window("[Activity*]").unwrap()),
            (Some(WinRect { x: 5000, y: 40, w: 800, h: 900 }), layout::parse_window("[Graph*]").unwrap()),
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

    /// A window saved maximised (the Two monitors window: its rect as Windows reports a maximised
    /// window on a 1920 × 1080 monitor, measured 2026-11-03) reopens maximised on that monitor; a
    /// smaller one on it at its own rect.
    #[test]
    fn a_window_saved_maximised_reopens_maximised() {
        let second = WinRect { x: 2560, y: 0, w: 1920, h: 1080 };
        let mut w = Workspace::default();
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(vec![
            (Some(WinRect { x: 2552, y: -8, w: 1920, h: 1009 }), layout::parse_window("[Activity*]").unwrap()),
            (Some(WinRect { x: 2600, y: 40, w: 800, h: 900 }), layout::parse_window("[Graph*]").unwrap()),
        ]);
        w.set_monitors(vec![WinRect { x: 0, y: 0, w: 2560, h: 1440 }, second]);
        let r = w.take_requests();
        assert!(matches!(r[0], WindowRequest::Open { rect: Some(m), fill: true, .. } if m == second), "{r:?}");
        assert!(matches!(r[1], WindowRequest::Open { rect: Some(WinRect { x: 2600, .. }), fill: false, .. }), "{r:?}");
    }

    #[test]
    fn set_windows_keeps_each_panel_once() {
        let mut w = Workspace::default(); // every panel in the main tree
        w.set_node(layout::parse("H(0.70,3D,[Log*])").unwrap());
        w.set_windows(vec![
            (None, layout::parse_window("V(0.5,[Activity*,Log],[Graph])").unwrap()),
            (None, layout::parse_window("[Graph*,Model]").unwrap()),
            (None, layout::parse_window("[Log*]").unwrap()),
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

    #[test]
    fn rects_intersect() {
        let a = WinRect { x: 0, y: 0, w: 100, h: 100 };
        assert!(a.intersects(&WinRect { x: 99, y: 99, w: 10, h: 10 }));
        assert!(!a.intersects(&WinRect { x: 100, y: 0, w: 10, h: 10 }), "touching is not overlapping");
        assert!(!a.intersects(&WinRect { x: -10, y: 0, w: 10, h: 10 }));
        assert!(a.intersects(&WinRect { x: -10, y: -10, w: 200, h: 200 }));
    }
}
