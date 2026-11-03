//! The workspace on screen (docking spec D-2..D-5; user decision 6, "approach 1": `egui_dock` for
//! the in-window tree): one tree of splits and tabbed leaves holding the 3D view and every panel,
//! replacing the two fixed docks. Drag a tab onto another leaf's tab bar to tab it there, onto an
//! edge to split, out of the tree to float it in a window inside the app (spec D-5); the
//! separators resize. All of it is display state (spec D-13): nothing here is an `Action`.
//!
//! - **The 3D view is a tile, not a panel** (spec D-3, [`Tile::View3d`]): never closable, never in
//!   a window, no background (the scene shows through) and no tab bar of its own (its leaf's bar
//!   is hidden). `egui_dock` cannot refuse a drop into a given leaf, so a panel dropped into the 3D
//!   view's leaf is moved out on the same frame — to a new split right of the view — and the pass
//!   is discarded so nothing is drawn with the panel over the scene ([`Workspace`]'s `normalize`).
//!   Its body rect is `PanelOutput::view`, and the root `Ui` is left holding exactly that rect
//!   (`leave_only`): egui counts the pointer as over egui outside the root's remaining rect, so the
//!   3D view keeps its orbit / place / grab / edit input wherever its tile sits, as the space the
//!   old docks left did.
//! - **Tabs** (spec D-4): the panel's icon, its name and its status dot (`dock::status`: drift and
//!   box health, work running, trust, the Log's unread count) are painted into the tab, whose
//!   title reserves the room; hovering a tab peeks its key numbers (`dock::peek`); a middle click
//!   or the context menu closes it, the menu also floats it or resets the layout. A leaf's tab bar
//!   ends with its pin (auto-hides its active tab, D2: `egui_dock`'s add button, its plus left
//!   transparent under the duotone Pin icon — the library has no custom bar buttons, and the add
//!   button is a real widget that reports its leaf, so no private geometry is clicked) and its
//!   close button (closes the leaf's panels: `egui_dock`'s close-all, painted with the Close
//!   icon); pop-out comes with D3.
//! - **Auto-hidden panels** (spec D-10, `super::autohide`): [`Workspace::pin`] takes a panel out of
//!   the tree onto the strip of the 3D view's edge its leaf lay beyond, remembering its size, and
//!   docks it back beside the view on that edge; their strips and flyout are drawn round the 3D
//!   view's tile after the tree.
//! - **Panels stay where the user put them**: [`Workspace::show_panel`] makes a panel the active
//!   tab of its leaf, or — when it was closed — adds it to the right column's first leaf (the one
//!   right of the 3D view), else to a new split right of the view.
//!
//! The panels draw with `&mut UiState`, which also holds the tree, so the tree is lent to the dock
//! area for the frame; what the panels ask of the workspace meanwhile (the Readouts' "open the
//! Graph", say) is done when it comes back, and `visible` / `present` answer from what was
//! visible / present when it left. Panels floated in a window are not part of
//! [`Workspace::to_node`] (the profile's `layout = …`): [`Workspace::saved_hidden`] writes each as auto-hidden on the edge nearest its
//! window (`hidden = …`), so it comes back on that strip (recorded for review: a float is a
//! moment's arrangement, and D3's `window = …` lines will keep detached windows).
//! - **Saved fractions hold in a narrow window**: `egui_dock` clamps each split's fraction every
//!   frame to leave `separator.extra` either side; a fraction it only clamped — any split but the
//!   one whose separator the pointer was pressed on — is put back after the frame
//!   (`keep_fractions`), so the profile keeps the user's and a drag still moves its separator.
use super::autohide::{self, Flyout};
use super::layout::{self, Edge, Hidden, Node};
use crate::ui::{dock, icons, theme, Action, DockTab, PanelInput, UiState};
use egui::{pos2, vec2, Align, Align2, Color32, FontId, Frame, Id, LayerId, Layout, Margin, Painter, Pos2, Rect, Response, Stroke, Ui, UiBuilder, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, NodePath, Split, TabBodyStyle, TabIndex, TabPath, TabStyle, TabViewer, Tree};
use std::collections::HashMap;

/// One tile of the tree: the 3D view, or a panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tile {
    View3d,
    Panel(DockTab),
}

/// The 3D view's share of a split made beside it ([`Workspace::show_panel`] with no right column,
/// a panel dropped into its leaf): the default layout's `H(0.70,3D,…)`.
pub(super) const VIEW_SHARE: f32 = 0.70;
/// A tab's padding either side of its title (`egui_dock`'s `tab_title`: the text sits 8 pt in).
const PAD: f32 = 8.0;
/// The icon drawn at the start of a tab, and the room its title leaves for it.
const ICON: f32 = 14.0;
const ICON_GAP: f32 = ICON + 6.0;
/// The room a tab's title leaves at its end for the status dot, and for the Log's count (kept
/// small: the default layout's lower-left leaf already scrolls its four tabs at 1400 pt).
const DOT_GAP: f32 = 12.0;
const COUNT_GAP: f32 = 18.0;
/// A tab bar's height.
const TAB_BAR: f32 = 28.0;
/// A floated panel's first size.
const FLOAT_SIZE: egui::Vec2 = vec2(380.0, 480.0);

/// A change the panels asked for while a tree was lent out (done when it is back).
pub(super) enum Change {
    Show(DockTab),
    Close(DockTab),
    Float(DockTab),
    Set(Node),
    Pin(DockTab, bool),
    SetHidden(Vec<Hidden>),
    /// Into a new detached window, the second panel shown (spec D-12).
    PopOut(Vec<DockTab>, DockTab),
    /// Back into the main tree, together (spec D-12: Move to main window, Dock back).
    ToMain(Vec<DockTab>),
    ToWindow(DockTab, u32),
}

/// The workspace: the dock tree, the auto-hidden panels and their flyout, the detached windows,
/// and what the panels asked of it while a tree was lent out.
pub struct Workspace {
    pub(super) dock: DockState<Tile>,
    /// The auto-hidden panels, in the order they were unpinned (each strip's order).
    pub(super) hidden: Vec<Hidden>,
    pub(super) fly: Flyout,
    /// The pointer's press origin while a button is held, and the split whose separator it was
    /// pressed on (`keep_fractions` leaves a dragged split alone).
    drag: Option<(Pos2, Option<NodePath>)>,
    /// While a dock area draws: the panels visible and the panels present when its tree left
    /// (what [`Workspace::visible`] and [`Workspace::present`] answer meanwhile).
    pub(super) lent: Option<(Vec<DockTab>, Vec<DockTab>)>,
    pub(super) pending: Vec<Change>,
    /// The detached windows (spec D-12, `super::detached`).
    pub(super) det: super::detached::Windows,
}

impl Default for Workspace {
    /// The D1 default ([`layout::default_layout`]).
    fn default() -> Self {
        Workspace::from_node(&layout::default_layout())
    }
}

impl Workspace {
    fn from_node(n: &Node) -> Workspace {
        let mut w = Workspace { dock: build(n), hidden: Vec::new(), fly: Flyout::default(), drag: None, lent: None, pending: Vec::new(), det: Default::default() };
        w.normalize();
        w
    }

    /// The main tree as a layout (the profile's `layout = …`), cleaned; panels floating in a window
    /// are left out.
    pub fn to_node(&self) -> Node {
        let main = self.dock.main_surface();
        node_of(main, NodeIndex::root()).and_then(layout::clean).filter(layout::has_view).unwrap_or_else(layout::default_layout)
    }

    /// Replaces the arrangement with `n` (a tree without the 3D view gives the default); panels
    /// floating in a window and the auto-hidden ones go ([`Workspace::set_hidden`] adds those), and
    /// the panels `n` holds leave their detached windows (an emptied one closes).
    pub fn set_node(&mut self, n: Node) {
        if self.lent.is_some() {
            self.pending.push(Change::Set(n));
            return;
        }
        let n = layout::clean(n).filter(layout::has_view).unwrap_or_else(layout::default_layout);
        self.dock = build(&n);
        self.normalize();
        self.hidden.clear();
        self.fly.close();
        self.drop_from_windows(&layout::panels(&n));
    }

    /// The auto-hidden panels (spec D-10), in each strip's order.
    pub fn hidden(&self) -> &[Hidden] {
        &self.hidden
    }

    /// Replaces the auto-hidden panels (the profile's `hidden = …`): a panel already in the tree
    /// or a detached window keeps its place there (as in the grammar, its first place wins), a
    /// duplicate its first entry.
    pub fn set_hidden(&mut self, list: Vec<Hidden>) {
        if self.lent.is_some() {
            self.pending.push(Change::SetHidden(list));
            return;
        }
        self.hidden.clear();
        for h in list {
            if self.dock.find_tab(&Tile::Panel(h.tab)).is_none() && !self.is_hidden(h.tab) && self.window_of(h.tab).is_none() {
                self.hidden.push(Hidden { size: h.size.clamp(layout::HIDDEN_MIN, layout::HIDDEN_MAX), ..h });
            }
        }
        if self.fly.open.is_some_and(|t| !self.is_hidden(t)) {
            self.fly.close();
        }
    }

    /// The auto-hidden panels as the profile saves them: [`Workspace::hidden`], then each panel
    /// floating in a window as hidden on the 3D view's edge nearest its window, at its width (its
    /// height on the bottom edge).
    pub fn saved_hidden(&self) -> Vec<Hidden> {
        let mut out = self.hidden.clone();
        for (path, leaf) in self.dock.iter_leaves() {
            if path.surface.is_main() {
                continue;
            }
            let (edge, size) = self.fold(leaf.rect);
            for tile in &leaf.tabs {
                if let Tile::Panel(t) = *tile {
                    out.push(Hidden { tab: t, edge, size: size.clamp(layout::HIDDEN_MIN, layout::HIDDEN_MAX) });
                }
            }
        }
        out
    }

    /// The panel whose flyout is out.
    pub fn flyout(&self) -> Option<DockTab> {
        self.fly.open
    }

    pub(super) fn is_hidden(&self, t: DockTab) -> bool {
        self.hidden.iter().any(|h| h.tab == t)
    }

    /// Pins or unpins `t` (spec D-10). Unpinning a panel in the tree (or a floating window) takes
    /// it out onto the strip of the 3D view's edge its leaf lay beyond, remembering the leaf's
    /// width (height, on the bottom edge) — before the tree is drawn, the side of the view it sits
    /// on in the tree at [`autohide::DEFAULT_SIZE`]; a closed panel, or one in a detached window
    /// (whose window closes when it was its last), goes to the main window's right edge. Pinning a hidden
    /// panel docks it back beside the 3D view on its edge at its remembered size (a closed one is
    /// shown, [`Workspace::show_panel`]); either is a no-op when the panel already is so.
    pub fn pin(&mut self, t: DockTab, pinned: bool) {
        if self.lent.is_some() {
            self.pending.push(Change::Pin(t, pinned));
            return;
        }
        if pinned {
            match self.hidden.iter().position(|h| h.tab == t) {
                Some(i) => {
                    let h = self.hidden.remove(i);
                    if self.fly.open == Some(t) {
                        self.fly.close();
                    }
                    self.dock_beside_view(h);
                }
                None => self.show_panel(t),
            }
            return;
        }
        if self.is_hidden(t) {
            return;
        }
        let (edge, size) = match self.dock.find_tab(&Tile::Panel(t)) {
            Some(p) => {
                let folded = self.fold_tab(p);
                self.dock.remove_tab(p);
                folded
            }
            None => {
                self.remove_from_window(t);
                (Edge::Right, autohide::DEFAULT_SIZE)
            }
        };
        self.hidden.push(Hidden { tab: t, edge, size: size.clamp(layout::HIDDEN_MIN, layout::HIDDEN_MAX) });
    }

    /// The edge and size the tab at `p` folds to: its leaf's ([`Workspace::fold`]).
    fn fold_tab(&self, p: TabPath) -> (Edge, f32) {
        let r = self.dock.leaf(p.node_path()).map_or(Rect::NOTHING, |l| l.rect);
        if !(r.is_positive() && self.view_rect().is_some_and(|v| v.is_positive())) && p.surface.is_main() {
            // not drawn yet (a layout just set): the side of the view it sits on in the tree
            let main = self.dock.main_surface();
            if let Some((view, _)) = main.find_tab(&Tile::View3d) {
                return (side_in_tree(main, p.node, view), autohide::DEFAULT_SIZE);
            }
        }
        self.fold(r)
    }

    /// The edge a leaf drawn at `r` folds to ([`autohide::edge_of`]: where its centre lies
    /// relative to the 3D view's tile) and its size (width; height for the bottom edge). Right at
    /// [`autohide::DEFAULT_SIZE`] when the tree has not been drawn yet.
    fn fold(&self, r: Rect) -> (Edge, f32) {
        match (Some(r).filter(|r| r.is_positive()), self.view_rect().filter(|v| v.is_positive())) {
            (Some(r), Some(v)) => {
                let e = autohide::edge_of(r.center(), v);
                (e, if e == Edge::Bottom { r.height() } else { r.width() })
            }
            _ => (Edge::Right, autohide::DEFAULT_SIZE),
        }
    }

    /// Docks a hidden panel back: a new split beside the 3D view on its edge, giving it its size
    /// of the view's tile (10..90 %; 30 % before the tree is drawn).
    fn dock_beside_view(&mut self, h: Hidden) {
        let extent = self.view_rect().filter(|v| v.is_positive()).map(|v| if h.edge == Edge::Bottom { v.height() } else { v.width() });
        let share = extent.map_or(1.0 - VIEW_SHARE, |w| (h.size / w).clamp(0.1, 0.9));
        let main = self.dock.main_surface_mut();
        let Some((view, _)) = main.find_tab(&Tile::View3d) else { return };
        // `frac` is the first child's share: the new leaf's on the left, the view's otherwise
        let (split, frac) = match h.edge {
            Edge::Left => (Split::Left, share),
            Edge::Right => (Split::Right, 1.0 - share),
            Edge::Bottom => (Split::Below, 1.0 - share),
        };
        let [_, new] = main.split(view, split, frac, egui_dock::Node::leaf(Tile::Panel(h.tab)));
        main.set_focused_node(new);
    }

    /// Whether `t` is on screen: in the tree (a floating window, a detached window) and the
    /// active tab of its leaf, or out in the flyout.
    pub fn visible(&self, t: DockTab) -> bool {
        if self.fly.open == Some(t) {
            return true;
        }
        if let Some((shown, _)) = &self.lent {
            return shown.contains(&t);
        }
        self.dock
            .find_tab(&Tile::Panel(t))
            .is_some_and(|p| self.dock.leaf(p.node_path()).is_ok_and(|l| l.active == p.tab && !l.collapsed))
            || self.visible_in_window(t)
    }

    /// Whether `t` is in the workspace at all — a tab in the tree, a floating or a detached window
    /// (shown or behind another tab), or auto-hidden — rather than closed: the Window menu's check.
    pub fn present(&self, t: DockTab) -> bool {
        if let Some((_, present)) = &self.lent {
            return present.contains(&t);
        }
        self.is_hidden(t) || self.dock.find_tab(&Tile::Panel(t)).is_some() || self.window_of(t).is_some()
    }

    /// Shows `t`: makes it the active tab of its leaf (and focuses the leaf); an auto-hidden panel
    /// opens its flyout, stuck open (spec D-10); one in a detached window also brings that window
    /// to the front (spec D-12: Ctrl K's reveal, the Window menu); a closed panel joins the right
    /// column's first leaf, or a new split right of the 3D view when there is none.
    pub fn show_panel(&mut self, t: DockTab) {
        if self.lent.is_some() {
            self.pending.push(Change::Show(t));
            return;
        }
        if self.is_hidden(t) {
            self.fly.open(t, true);
            return;
        }
        let tile = Tile::Panel(t);
        if let Some(p) = self.dock.find_tab(&tile) {
            let _ = self.dock.set_active_tab(p);
            self.dock.set_focused_node_and_surface(p.node_path());
            return;
        }
        if self.show_in_window(t) {
            return;
        }
        let main = self.dock.main_surface_mut();
        let Some((view, _)) = main.find_tab(&Tile::View3d) else { return };
        match right_column(main, view).and_then(|c| first_leaf(main, c)) {
            Some(leaf) => {
                main[leaf].append_tab(tile);
                main.set_focused_node(leaf);
            }
            None => {
                main.split(view, Split::Right, VIEW_SHARE, egui_dock::Node::leaf(tile));
            }
        }
    }

    /// Closes `t` (its leaf goes when it was the last tab there; an auto-hidden panel leaves its
    /// strip; a detached window closes with its last panel).
    pub fn close_panel(&mut self, t: DockTab) {
        if self.lent.is_some() {
            self.pending.push(Change::Close(t));
            return;
        }
        if self.is_hidden(t) {
            self.hidden.retain(|h| h.tab != t);
            if self.fly.open == Some(t) {
                self.fly.close();
            }
            return;
        }
        match self.dock.find_tab(&Tile::Panel(t)) {
            Some(p) => {
                self.dock.remove_tab(p);
            }
            None => {
                self.remove_from_window(t);
            }
        }
    }

    /// Floats `t` in a window over the workspace, near where it was (a no-op when it already
    /// floats, or is closed): the tab's context menu.
    pub fn float_panel(&mut self, t: DockTab) {
        if self.lent.is_some() {
            self.pending.push(Change::Float(t));
            return;
        }
        let Some(p) = self.dock.find_tab(&Tile::Panel(t)).filter(|p| p.surface.is_main()) else { return };
        // a leaf not drawn yet has no rect
        let at = self.dock.leaf(p.node_path()).ok().map(|l| l.rect.min).filter(|m| m.is_finite()).map_or(pos2(80.0, 80.0), |m| m + vec2(24.0, 32.0));
        self.dock.detach_tab(p, Rect::from_min_size(at, FLOAT_SIZE));
    }

    /// Test hook: what `egui_dock` does when a dragged tab is dropped on the middle of the 3D
    /// view's leaf — `t` joins that leaf as its active tab — without the clean-up that follows.
    #[cfg(test)]
    pub fn drop_into_view_for_test(&mut self, t: DockTab) {
        if let Some(p) = self.dock.find_tab(&Tile::Panel(t)) {
            self.dock.remove_tab(p);
        }
        let p = self.dock.find_tab(&Tile::View3d).expect("the 3D view");
        if let Ok(leaf) = self.dock.leaf_mut(p.node_path()) {
            leaf.append_tab(Tile::Panel(t));
        }
    }

    /// The 3D view's tile as last drawn (its strips included).
    fn view_rect(&self) -> Option<Rect> {
        let p = self.dock.find_tab(&Tile::View3d)?;
        self.dock.leaf(p.node_path()).ok().map(|l| l.viewport)
    }

    /// Keeps the tree's rules after the dock area moved things: the 3D view is in the main tree,
    /// alone in its leaf, whose tab bar is hidden; every other leaf shows its tab bar. A panel found
    /// in the 3D view's leaf (a drop `egui_dock` could not refuse) goes to a new split right of the
    /// view, still active if it was. True when it moved something (the pass is redrawn).
    fn normalize(&mut self) -> bool {
        let Some(p) = self.dock.find_tab(&Tile::View3d).filter(|p| p.surface.is_main()) else {
            // cannot happen (the view is neither closable nor floatable); never leave it missing
            self.dock = build(&layout::default_layout());
            return true;
        };
        let mut moved = false;
        if let Ok(leaf) = self.dock.leaf_mut(p.node_path()) {
            leaf.tab_bar_hidden = true;
            if leaf.tabs.len() > 1 {
                let active = leaf.tabs.get(leaf.active.0).copied().filter(|t| *t != Tile::View3d);
                let extras: Vec<Tile> = leaf.tabs.iter().copied().filter(|t| *t != Tile::View3d).collect();
                leaf.tabs = vec![Tile::View3d];
                leaf.active = TabIndex(0);
                let at = active.and_then(|a| extras.iter().position(|t| *t == a)).unwrap_or(0);
                let [_, new] = self.dock.split(p.node_path(), Split::Right, VIEW_SHARE, egui_dock::Node::leaf_with(extras));
                if let Ok(l) = self.dock.main_surface_mut().leaf_mut(new) {
                    l.active = TabIndex(at);
                }
                moved = true;
            }
        }
        for (_, leaf) in self.dock.iter_leaves_mut() {
            if !leaf.tabs.contains(&Tile::View3d) {
                leaf.tab_bar_hidden = false;
            }
        }
        moved
    }

    /// Lends the tree to the dock area for the frame (a placeholder stays behind).
    fn lend(&mut self) -> DockState<Tile> {
        self.lent = Some(self.snapshot());
        std::mem::replace(&mut self.dock, DockState::new(Vec::new()))
    }

    /// The panels visible and the panels present now (what answers while a tree is lent out).
    pub(super) fn snapshot(&self) -> (Vec<DockTab>, Vec<DockTab>) {
        let shown = DockTab::PANELS.into_iter().filter(|&t| self.visible(t)).collect();
        let present = DockTab::PANELS.into_iter().filter(|&t| self.present(t)).collect();
        (shown, present)
    }

    /// The tree back from the dock area: its rules kept, then what the panels and the tabs asked
    /// for meanwhile done. True when a drop had to be undone (the pass is redrawn).
    fn give_back(&mut self, dock: DockState<Tile>, asked: Vec<Change>) -> bool {
        self.dock = dock;
        self.lent = None;
        let moved = self.normalize();
        let changes: Vec<Change> = std::mem::take(&mut self.pending).into_iter().chain(asked).collect();
        self.apply_changes(changes);
        moved
    }

    /// Does what was asked while a tree was lent out, in order.
    pub(super) fn apply_changes(&mut self, changes: Vec<Change>) {
        for c in changes {
            match c {
                Change::Show(t) => self.show_panel(t),
                Change::Close(t) => self.close_panel(t),
                Change::Float(t) => self.float_panel(t),
                Change::Set(n) => self.set_node(n),
                Change::Pin(t, pinned) => self.pin(t, pinned),
                Change::SetHidden(h) => self.set_hidden(h),
                Change::PopOut(tabs, active) => {
                    self.pop_out(&tabs, active);
                }
                Change::ToMain(tabs) => self.move_to_main(&tabs),
                Change::ToWindow(t, id) => self.move_to_window(t, id),
            }
        }
    }
}

/// The dock tree of a layout: the root leaf holds the first leaf's tabs, and each split splits the
/// leaf holding its first child's (`Split::Right` / `Split::Below` keep the old leaf first, so
/// `frac` is the first child's share, as in the grammar).
pub(super) fn build(n: &Node) -> DockState<Tile> {
    let mut d = DockState::new(first_tiles(n));
    grow(d.main_surface_mut(), NodeIndex::root(), n);
    d
}

fn first_tiles(n: &Node) -> Vec<Tile> {
    match n {
        Node::View3d => vec![Tile::View3d],
        Node::Leaf { tabs, .. } => tabs.iter().map(|&t| Tile::Panel(t)).collect(),
        Node::Split { a, .. } => first_tiles(a),
    }
}

/// Grows the leaf at `at` (which holds `first_tiles(n)`) into `n`.
fn grow(tree: &mut Tree<Tile>, at: NodeIndex, n: &Node) {
    match n {
        Node::View3d => {}
        Node::Leaf { tabs, active } => {
            if let Ok(l) = tree.leaf_mut(at) {
                l.active = TabIndex(if *active < tabs.len() { *active } else { 0 });
            }
        }
        Node::Split { horizontal, frac, a, b } => {
            let split = if *horizontal { Split::Right } else { Split::Below };
            let [old, new] = tree.split(at, split, frac.clamp(0.0, 1.0), egui_dock::Node::leaf_with(first_tiles(b)));
            grow(tree, old, a);
            grow(tree, new, b);
        }
    }
}

/// The layout of the subtree at `at` (`None`: empty). A leaf holding the 3D view is the view.
pub(super) fn node_of(tree: &Tree<Tile>, at: NodeIndex) -> Option<Node> {
    if at.0 >= tree.len() {
        return None;
    }
    match &tree[at] {
        egui_dock::Node::Empty => None,
        egui_dock::Node::Leaf(l) => {
            if l.tabs.contains(&Tile::View3d) {
                return Some(Node::View3d);
            }
            let tabs: Vec<DockTab> = l
                .tabs
                .iter()
                .filter_map(|t| match t {
                    Tile::Panel(p) => Some(*p),
                    Tile::View3d => None,
                })
                .collect();
            (!tabs.is_empty()).then_some(Node::Leaf { tabs, active: l.active.0 })
        }
        egui_dock::Node::Horizontal(s) | egui_dock::Node::Vertical(s) => {
            let horizontal = tree[at].is_horizontal();
            match (node_of(tree, at.left()), node_of(tree, at.right())) {
                (Some(a), Some(b)) => Some(Node::Split { horizontal, frac: s.fraction, a: Box::new(a), b: Box::new(b) }),
                (one, None) | (None, one) => one,
            }
        }
    }
}

/// The side of the 3D view (at `view`) the leaf at `leaf` sits on in the tree: at the lowest split
/// holding both, left or right of it side by side, below it stacked (above it — there is no top
/// strip — right).
fn side_in_tree(tree: &Tree<Tile>, leaf: NodeIndex, view: NodeIndex) -> Edge {
    let mut ancestors = vec![leaf];
    while let Some(p) = ancestors.last().and_then(|n| n.parent()) {
        ancestors.push(p);
    }
    let mut n = view;
    while let Some(parent) = n.parent() {
        if let Some(i) = ancestors.iter().position(|&a| a == parent) {
            let leaf_first = i > 0 && ancestors[i - 1].is_left();
            return match (tree[parent].is_horizontal(), leaf_first) {
                (true, true) => Edge::Left,
                (false, false) => Edge::Bottom,
                _ => Edge::Right,
            };
        }
        n = parent;
    }
    Edge::Right
}

/// The right column: the subtree right of the 3D view in the nearest side-by-side split that has
/// the view on its left.
fn right_column(tree: &Tree<Tile>, view: NodeIndex) -> Option<NodeIndex> {
    let mut n = view;
    while let Some(parent) = n.parent() {
        if tree[parent].is_horizontal() && n.is_left() {
            return Some(parent.right());
        }
        n = parent;
    }
    None
}

/// The first leaf of the subtree at `at` (top-left first).
pub(super) fn first_leaf(tree: &Tree<Tile>, mut at: NodeIndex) -> Option<NodeIndex> {
    while at.0 < tree.len() {
        match &tree[at] {
            egui_dock::Node::Leaf(_) => return Some(at),
            egui_dock::Node::Empty => return None,
            _ => at = at.left(),
        }
    }
    None
}

/// What draws each tile, and the changes its tabs ask for.
struct Tiles<'a, 'i> {
    st: &'a mut UiState,
    input: &'a PanelInput<'i>,
    acts: &'a mut Vec<Action>,
    asked: Vec<Change>,
    /// The layer each tab drawn this frame was drawn on (its leaf's: the main area's, or a
    /// floating window's), where its leaf's pin and close icons are painted.
    layers: HashMap<Tile, LayerId>,
    /// The leaves whose pin (`egui_dock`'s add button) was clicked: their active tabs unpin.
    pins: Vec<NodePath>,
    /// The detached window drawn (`None`: the main window), the detached windows' ids (the tab
    /// menu's "Move to window N"), and how many panels the drawn window holds.
    window: Option<u32>,
    windows: Vec<u32>,
    window_tabs: usize,
}

/// A panel's tab title: its name, with room before it for the icon and, on a tab that can carry
/// one (`dock::can_flag`), after it for the status dot whether it shows or not, so the tab's width
/// holds still (`on_tab_button` paints both). The colour is left to the tab's style.
fn title(t: DockTab) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let format = TextFormat { font_id: FontId::proportional(13.0), color: Color32::PLACEHOLDER, ..Default::default() };
    let mut job = LayoutJob::default();
    job.append(t.name(), ICON_GAP, format.clone());
    if let Some(room) = dot_room(t) {
        job.append(" ", room - 3.0, format);
    }
    job
}

/// The room `t`'s title keeps for its status (`None`: it never has one).
fn dot_room(t: DockTab) -> Option<f32> {
    dock::can_flag(t).then_some(if t == DockTab::Log { COUNT_GAP } else { DOT_GAP })
}

impl TabViewer for Tiles<'_, '_> {
    type Tab = Tile;

    fn id(&mut self, tab: &mut Tile) -> Id {
        Id::new(("workspace-tile", *tab))
    }

    fn title(&mut self, tab: &mut Tile) -> WidgetText {
        match *tab {
            Tile::View3d => "3D view".into(),
            Tile::Panel(t) => title(t).into(),
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Tile) {
        // the 3D view draws nothing: the scene shows through its tile (its rect: the leaf's viewport)
        if let Tile::Panel(t) = *tab {
            dock::body(ui, self.st, self.input, self.acts, t);
        }
    }

    fn on_tab_button(&mut self, tab: &mut Tile, r: &Response) {
        let Tile::Panel(t) = *tab else { return };
        // the part that can be clicked (a tab scrolled past its bar's end is clipped)
        crate::ui::probe_rect(&format!("dock-tab-{t:?}"), r.interact_rect);
        self.layers.insert(*tab, r.layer_id);
        let p = r.ctx.layer_painter(r.layer_id).with_clip_rect(r.interact_rect);
        let ink = if self.st.tab_shown(t) || r.hovered() { theme::INK } else { theme::INK_DIM };
        let icon = Rect::from_center_size(pos2(r.rect.left() + PAD + ICON / 2.0, r.rect.center().y), egui::Vec2::splat(ICON));
        icons::paint(&p, icon, dock::icon(t), ink);
        // the status is asked once per tab and frame: here (the title only reserves its room)
        if let Some(s) = dock::status(t, self.st, self.input) {
            debug_assert!(dock::can_flag(t), "{t:?} has a status but its title keeps no room for it");
            let at = pos2(r.rect.right() - PAD - dot_room(t).unwrap_or(DOT_GAP) / 2.0 + 2.0, r.rect.center().y);
            paint_status(&p, at, s, t, r.ctx.input(|i| i.time));
        }
        if r.middle_clicked() {
            self.asked.push(Change::Close(t));
        }
        if r.hovered() {
            let tip = dock::peek(self.st, self.input, t);
            let _ = r.clone().on_hover_text(format!("{tip}\n(drag the tab to another tab bar, to an edge to split, or out to float it; right-click for more)"));
        }
        // float (main window only), the detached windows (spec D-12: tabs cross OS windows by
        // this menu, never by dragging), close, reset
        r.context_menu(|ui| {
            let ask = |ui: &mut Ui, label: &str, probe: &str, tip: &str, c: Change, asked: &mut Vec<Change>| {
                if crate::ui::probed(probe, ui.button(label)).on_hover_text(tip).clicked() {
                    asked.push(c);
                    ui.close();
                }
            };
            match self.window {
                None => {
                    ask(ui, "Float in a window", "tab-menu-float", "a window inside the app, over the workspace", Change::Float(t), &mut self.asked);
                    ask(ui, "Move to new window", "tab-menu-new-window", "its own OS window: move it to another monitor", Change::PopOut(vec![t], t), &mut self.asked);
                }
                Some(_) => {
                    ask(ui, "Move to main window", "tab-menu-main", "back into the main window, where it was", Change::ToMain(vec![t]), &mut self.asked);
                    if self.window_tabs > 1 {
                        ask(ui, "Move to new window", "tab-menu-new-window", "its own OS window", Change::PopOut(vec![t], t), &mut self.asked);
                    }
                }
            }
            for &id in self.windows.iter().filter(|&&id| Some(id) != self.window) {
                ask(ui, &format!("Move to window {id}"), &format!("tab-menu-window-{id}"), "into that detached window", Change::ToWindow(t, id), &mut self.asked);
            }
            ask(ui, "Close", "tab-menu-close", "close the panel (Window brings it back)", Change::Close(t), &mut self.asked);
            if self.window.is_none() {
                ui.separator();
                if ui.button("Reset the layout").clicked() {
                    self.st.reset_layout(); // done when the tree is back (`Workspace::set_node`)
                    ui.close();
                }
            }
        });
    }

    /// The leaf's add button is its pin (spec D-4, D-10): the active tab unpins once the tree is
    /// back (`show`).
    fn on_add(&mut self, path: NodePath) {
        self.pins.push(path);
    }

    fn is_closeable(&self, tab: &Tile) -> bool {
        *tab != Tile::View3d
    }

    /// Floating inside the main window only (a detached window's tabs stay in its tree).
    fn allowed_in_windows(&self, tab: &mut Tile) -> bool {
        *tab != Tile::View3d && self.window.is_none()
    }

    fn clear_background(&self, tab: &Tile) -> bool {
        *tab != Tile::View3d
    }

    /// No scroll area of the dock's: each panel's body scrolls itself (`dock::body`), the 3D view
    /// not at all.
    fn scroll_bars(&self, _tab: &Tile) -> [bool; 2] {
        [false, false]
    }

    /// The 3D view's tile: no margin, no border, no fill.
    fn tab_style_override(&self, tab: &Tile, global: &TabStyle) -> Option<TabStyle> {
        (*tab == Tile::View3d).then(|| TabStyle {
            tab_body: TabBodyStyle { inner_margin: Margin::ZERO, stroke: Stroke::NONE, bg_fill: Color32::TRANSPARENT, ..global.tab_body.clone() },
            ..global.clone()
        })
    }
}

/// The dock area's look, in the theme's colours: tabs on the app's dark ground, the active one in
/// the panel colour it opens onto; the hidden tab bar of the 3D view's leaf left inert (no drag
/// handle, no show-tab-bar button).
fn style(egui_style: &egui::Style) -> egui_dock::Style {
    let mut s = egui_dock::Style::from_egui(egui_style);
    s.main_surface_border_stroke = Stroke::NONE;
    s.tab_bar.bg_fill = theme::APP;
    s.tab_bar.height = TAB_BAR;
    s.tab_bar.hline_color = theme::LINE;
    s.tab_bar.fill_tab_bar = false;
    s.tab.minimum_width = None; // a tab is as wide as its title: the icon and the dot sit at its ends
    let look = |bg: Color32, text: Color32, outline: Color32, base: &egui_dock::TabInteractionStyle| egui_dock::TabInteractionStyle { bg_fill: bg, text_color: text, outline_color: outline, ..base.clone() };
    s.tab.active = look(theme::PANEL, theme::INK, theme::LINE, &s.tab.active);
    s.tab.active_with_kb_focus = look(theme::PANEL, theme::INK, theme::ACCENT, &s.tab.active_with_kb_focus);
    s.tab.focused = look(theme::PANEL, theme::ACCENT_INK, theme::LINE, &s.tab.focused);
    s.tab.focused_with_kb_focus = look(theme::PANEL, theme::ACCENT_INK, theme::ACCENT, &s.tab.focused_with_kb_focus);
    s.tab.inactive = look(theme::APP, theme::INK_DIM, theme::APP, &s.tab.inactive);
    s.tab.inactive_with_kb_focus = look(theme::APP, theme::INK_DIM, theme::ACCENT, &s.tab.inactive_with_kb_focus);
    s.tab.hovered = look(theme::PANEL_2, theme::INK, theme::LINE, &s.tab.hovered);
    s.tab.tab_body.bg_fill = theme::PANEL;
    s.tab.tab_body.stroke = Stroke::new(1.0, theme::LINE);
    s.tab.tab_body.inner_margin = Margin::same(8);
    s.tab.tab_body.hidden_tab_bar_drag_height = Some(0.0);
    s.buttons.show_tab_bar_size = 0.0;
    s.buttons.show_tab_bar_hover_expand = 0.0;
    // the bar's last 48 pt hold our own pop-out (dock-back in a detached window) and close
    // buttons (`leaf_buttons`): egui_dock lays its tabs and its add button out inside this margin
    s.tab_bar.inner_margin = Margin { left: 0, right: (POP_BUTTON + CLOSE_BUTTON) as i8, top: 0, bottom: 0 };
    // the leaf's pin: egui_dock's add button, its plus left transparent under the Pin icon
    s.buttons.add_tab_align = egui_dock::TabAddAlign::Right;
    s.buttons.add_tab_color = Color32::TRANSPARENT;
    s.buttons.add_tab_active_color = Color32::TRANSPARENT;
    s.buttons.add_tab_bg_fill = theme::PANEL_2;
    s.buttons.add_tab_border_color = theme::LINE_SOFT;
    s.separator.color_idle = theme::LINE;
    s.separator.color_hovered = theme::ACCENT;
    s.separator.color_dragged = theme::GLOW;
    // egui_dock clamps each split's fraction every frame to leave this much either side, and the
    // profile saves what it left: 175 pt (its default) rewrote the default layout's 24 % column in
    // a 700-pt window; 80 still leaves a panel room for its title and a control
    s.separator.extra = SEPARATOR_EXTRA;
    s
}

/// The least a split leaves either side of its separator (`egui_dock`'s `separator.extra`).
const SEPARATOR_EXTRA: f32 = 80.0;
/// A tab bar's end, right to left: our close and pop-out buttons in the bar's right margin
/// (`style`), then `egui_dock`'s add button — the pin — just inside it (its `TAB_ADD_BUTTON_SIZE`,
/// right-aligned); each 24 pt wide, the bar's height less 2, as egui_dock's own; and the icons
/// painted in them.
const CLOSE_BUTTON: f32 = 24.0;
const POP_BUTTON: f32 = 24.0;
const PIN_BUTTON: f32 = 24.0;
const CLOSE_ICON: f32 = 13.0;

/// Each shown tab bar's end (spec D-4: pin, pop out, close): paints the duotone Pin icon in
/// `egui_dock`'s add button (its plus left transparent, `style`) and draws our pop-out and close
/// buttons in the bar's right margin, on the layer its leaf was drawn on (the main area's or a
/// floating window's: they are real widgets there, over the bar). In a detached window (`window`)
/// there is no pin and the pop-out button is Dock back (its panels to the main window). Returns
/// what was clicked: the leaf popped out (all its panels, its shown one shown), docked back, or
/// closed.
fn leaf_buttons(ctx: &egui::Context, dock: &DockState<Tile>, layers: &HashMap<Tile, LayerId>, bar: f32, window: Option<u32>) -> Vec<Change> {
    let mut asked = Vec::new();
    let px = 1.0 / ctx.pixels_per_point();
    for (_, leaf) in dock.iter_leaves() {
        if leaf.tab_bar_hidden || !leaf.rect.is_positive() {
            continue;
        }
        let tabs: Vec<DockTab> = leaf.tabs.iter().filter_map(|t| if let Tile::Panel(p) = t { Some(*p) } else { None }).collect();
        let Some(&first) = tabs.first() else { continue };
        let shown = leaf.tabs.get(leaf.active.0).and_then(|t| if let Tile::Panel(p) = t { Some(*p) } else { None }).unwrap_or(first);
        // any tab of the leaf drawn this frame names its layer (the first may not have been)
        let Some(&layer) = leaf.tabs.iter().find_map(|t| layers.get(t)) else { continue };
        let r = leaf.rect;
        let close = Rect::from_min_max(pos2(r.right() - CLOSE_BUTTON, r.top()), pos2(r.right(), r.top() + bar - 2.0));
        let pop = close.translate(vec2(-POP_BUTTON, 0.0));
        let pin = pop.translate(vec2(-PIN_BUTTON, 0.0));
        crate::ui::probe_rect(&format!("dock-close-{first:?}"), close);
        crate::ui::probe_rect(&format!("dock-pop-{first:?}"), pop);
        let p = ctx.layer_painter(layer).with_clip_rect(r);
        // egui_dock's line under the bar stops at its margin: carried on under our buttons
        p.hline(pop.left()..=r.right(), r.top() + bar - px, Stroke::new(px, theme::LINE));
        if window.is_none() {
            crate::ui::probe_rect(&format!("dock-pin-{first:?}"), pin);
            let ink = if ctx.rect_contains_pointer(layer, pin) { theme::INK } else { theme::INK_DIM };
            icons::paint(&p, Rect::from_center_size(pin.center(), egui::Vec2::splat(CLOSE_ICON)), icons::Icon::Pin, ink);
        }
        let ui = Ui::new(ctx.clone(), Id::new(("leaf-buttons", window, first)), UiBuilder::new().layer_id(layer).max_rect(pop.union(close)));
        let (pop_icon, pop_tip) = match window {
            None => (icons::Icon::PopOut, "pop out: this leaf's panels in their own window (move it to another monitor)"),
            Some(_) => (icons::Icon::DockBack, "dock back: this leaf's panels into the main window, where they were"),
        };
        for (rect, icon, tip, which) in [(pop, pop_icon, pop_tip, 0), (close, icons::Icon::Close, "close this leaf's panels (Window brings them back)", 1)] {
            let resp = ui.interact(rect, ui.id().with(which), egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tip);
            if resp.hovered() {
                p.rect_filled(rect, 0.0, theme::PANEL_2);
            }
            p.vline(rect.left(), rect.y_range(), Stroke::new(px, theme::LINE_SOFT));
            let ink = if resp.hovered() { theme::INK } else { theme::INK_DIM };
            icons::paint(&p, Rect::from_center_size(rect.center(), egui::Vec2::splat(CLOSE_ICON)), icon, ink);
            if resp.clicked() {
                match (which, window) {
                    (0, None) => asked.push(Change::PopOut(tabs.clone(), shown)),
                    (0, Some(_)) => asked.extend([Change::ToMain(tabs.clone()), Change::Show(shown)]),
                    _ => asked.extend(tabs.iter().map(|&t| Change::Close(t))),
                }
            }
        }
    }
    asked
}

/// Every split's fraction, by node, in the dock's own order.
fn fractions(dock: &DockState<Tile>) -> Vec<(NodePath, f32)> {
    dock.iter_all_nodes()
        .filter_map(|(p, n)| match n {
            egui_dock::Node::Horizontal(s) | egui_dock::Node::Vertical(s) => Some((p, s.fraction)),
            _ => None,
        })
        .collect()
}

/// The split whose separator (its interact band, `half` either side of the line) is at `p`.
fn separator_at(dock: &DockState<Tile>, p: Pos2, half: f32) -> Option<NodePath> {
    dock.iter_all_nodes().find_map(|(path, n)| match n {
        egui_dock::Node::Horizontal(s) if s.rect.is_positive() => {
            let x = s.rect.left() + s.rect.width() * s.fraction;
            ((p.x - x).abs() <= half && s.rect.y_range().contains(p.y)).then_some(path)
        }
        egui_dock::Node::Vertical(s) if s.rect.is_positive() => {
            let y = s.rect.top() + s.rect.height() * s.fraction;
            ((p.y - y).abs() <= half && s.rect.x_range().contains(p.x)).then_some(path)
        }
        _ => None,
    })
}

/// Puts back each split fraction `egui_dock` only clamped this frame (`before`: the fractions
/// when the tree was lent): it clamps a split's fraction every frame to leave `separator.extra`
/// either side, which in a narrow window rewrote the user's (and the profile saved it). The split
/// whose separator is being dragged (`dragged`) keeps what egui_dock made of it — a fraction
/// outside the bounds jumps to the bound and moves on from there; any other split whose fraction
/// is exactly its old value clamped to that range was only clamped, and the old value goes back.
/// Nothing is put back when the tree changed shape this frame.
fn keep_fractions(dock: &mut DockState<Tile>, before: &[(NodePath, f32)], dragged: Option<NodePath>) {
    let after = fractions(dock);
    if after.len() != before.len() || after.iter().zip(before).any(|(a, b)| a.0 != b.0) {
        return;
    }
    for ((path, node), &(_, old)) in dock.iter_all_nodes_mut().filter(|(_, n)| n.is_parent()).zip(before) {
        if Some(path) == dragged {
            continue;
        }
        let horizontal = node.is_horizontal();
        if let egui_dock::Node::Horizontal(s) | egui_dock::Node::Vertical(s) = node {
            let range = if horizontal { s.rect.width() } else { s.rect.height() };
            if s.fraction == old || !(range > 0.0) {
                continue;
            }
            // egui_dock's own bounds (`show_separator`)
            let min = (SEPARATOR_EXTRA / range).min(1.0);
            let max = 1.0 - min;
            let (min, max) = (min.min(max), max.max(min));
            if (s.fraction - old.clamp(min, max)).abs() < 1e-5 {
                s.fraction = old;
            }
        }
    }
}

/// Paints a status (`dock::status`: its colour, and its count when it counts) centred at `at`:
/// a tab's and a strip icon's. The Activity dot pulses while work runs (`time`: egui's input time).
pub(super) fn paint_status(p: &Painter, at: Pos2, (c, count): (Color32, Option<String>), t: DockTab, time: f64) {
    match count {
        Some(n) => {
            p.circle_filled(at, 7.5, c);
            p.text(at, Align2::CENTER_CENTER, n, FontId::monospace(9.0), Color32::WHITE);
        }
        None => {
            let pulse = if t == DockTab::Activity { 0.85 + 0.15 * (time * 7.0).sin() as f32 } else { 1.0 };
            p.circle_filled(at, 4.5, theme::PANEL);
            p.circle_filled(at, 3.5 * pulse, c);
        }
    }
}

/// Draws the workspace into what the top bar and the toolbar leave of `root`, then the auto-hide
/// strips and flyout round the 3D view's tile (`autohide::show`); returns the 3D view the strips
/// leave, which is then all the root `Ui` has left (so egui counts the pointer over it as not over
/// egui).
pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Rect {
    let area = root.available_rect_before_wrap();
    let style = style(root.style());
    let mut dock = st.workspace.lend();
    let before = fractions(&dock);
    // the separator being dragged, if any: latched at its press, kept through the release frame
    // (whose movement is the drag's last)
    let origin = root.ctx().input(|i| i.pointer.press_origin());
    let half = style.separator.width / 2.0 + style.separator.extra_interact_width / 2.0 + 1.0;
    let dragged = match (origin, st.workspace.drag) {
        (Some(o), Some((from, split))) if o == from => split,
        (Some(o), _) => separator_at(&dock, o, half),
        (None, latched) => latched.and_then(|(_, split)| split),
    };
    let windows = st.workspace.windows();
    let asked = {
        let mut tiles = Tiles { st, input, acts, asked: Vec::new(), layers: HashMap::new(), pins: Vec::new(), window: None, windows, window_tabs: 0 };
        let mut ui = root.new_child(UiBuilder::new().max_rect(area).layout(Layout::top_down(Align::Min)).id_salt("workspace"));
        DockArea::new(&mut dock)
            .id(Id::new("workspace"))
            .style(style)
            .show_close_buttons(false) // the leaf's close button ends its tab bar (spec D-4, `leaf_buttons`)
            .show_leaf_close_all_buttons(false)
            .show_add_buttons(true) // the leaf's pin (`on_add`)
            .show_leaf_collapse_buttons(false)
            .hidable_tab_bars(true) // only the 3D view's leaf hides its bar (`normalize`)
            .tab_context_menus(false) // our own (`on_tab_button`): float, the windows, close, reset
            .show_inside(&mut ui, &mut tiles);
        let buttons = leaf_buttons(root.ctx(), &dock, &tiles.layers, TAB_BAR, None);
        keep_fractions(&mut dock, &before, dragged);
        let mut asked = tiles.asked;
        asked.extend(buttons);
        for path in tiles.pins {
            if let Some(Tile::Panel(t)) = dock.leaf(path).ok().and_then(|l| l.tabs.get(l.active.0)) {
                asked.push(Change::Pin(*t, false));
            }
        }
        asked
    };
    st.workspace.drag = origin.map(|o| (o, dragged));
    if st.workspace.give_back(dock, asked) {
        root.ctx().request_discard("a panel dropped into the 3D view's leaf went beside it");
    }
    let tile = st.workspace.view_rect().filter(|r| r.is_positive() && area.expand(1.0).contains_rect(*r)).unwrap_or(area);
    let view = autohide::show(root, st, input, acts, tile);
    leave_only(root, view);
    view
}

/// Draws detached window `id`'s tree (spec D-12) into all of `root` (the window's own egui
/// context, the same frame as the main window's): its panels with the same `UiState` and
/// `PanelInput`, their actions into `acts`; no 3D view, no strips, no floating; its tab bars end
/// with Dock back and Close. False when there is no such window.
pub fn show_window(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, id: u32) -> bool {
    let tabs = st.workspace.window_panels(id).len();
    let windows = st.workspace.windows();
    let Some(mut dock) = st.workspace.lend_window(id) else { return false };
    let area = root.available_rect_before_wrap();
    let style = style(root.style());
    let asked = {
        let mut tiles = Tiles { st, input, acts, asked: Vec::new(), layers: HashMap::new(), pins: Vec::new(), window: Some(id), windows, window_tabs: tabs };
        let mut ui = root.new_child(UiBuilder::new().max_rect(area).layout(Layout::top_down(Align::Min)).id_salt(("detached", id)));
        DockArea::new(&mut dock)
            .id(Id::new(("detached-workspace", id)))
            .style(style)
            .show_close_buttons(false)
            .show_leaf_close_all_buttons(false)
            .show_add_buttons(false) // no pin: a detached window has no strips
            .show_leaf_collapse_buttons(false)
            .tab_context_menus(false)
            .show_inside(&mut ui, &mut tiles);
        let buttons = leaf_buttons(root.ctx(), &dock, &tiles.layers, TAB_BAR, Some(id));
        let mut asked = tiles.asked;
        asked.extend(buttons);
        asked
    };
    st.workspace.give_back_window(id, dock, asked);
    true
}

/// Shrinks what `root` has left to `view` with frameless, fixed panels round it (egui's own way of
/// taking space from a `Ui`'s remaining rect, as the old docks did).
fn leave_only(root: &mut Ui, view: Rect) {
    let a = root.available_rect_before_wrap();
    let v = view.intersect(a);
    let gaps = [
        (egui::Panel::left("workspace-gap-left"), v.left() - a.left()),
        (egui::Panel::right("workspace-gap-right"), a.right() - v.right()),
        (egui::Panel::top("workspace-gap-top"), v.top() - a.top()),
        (egui::Panel::bottom("workspace-gap-bottom"), a.bottom() - v.bottom()),
    ];
    for (panel, size) in gaps {
        if size > 0.0 {
            panel.exact_size(size).resizable(false).show_separator_line(false).frame(Frame::NONE).show(root, |_| {});
        }
    }
}
