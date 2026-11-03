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
//!   ends with its close button (closes the leaf's panels: `egui_dock`'s, painted with the
//!   duotone Close icon); pin and pop-out come with D2 and D3.
//! - **Panels stay where the user put them**: [`Workspace::show_panel`] makes a panel the active
//!   tab of its leaf, or — when it was closed — adds it to the right column's first leaf (the one
//!   right of the 3D view), else to a new split right of the view.
//!
//! The panels draw with `&mut UiState`, which also holds the tree, so the tree is lent to the dock
//! area for the frame; what the panels ask of the workspace meanwhile (the Readouts' "open the
//! Graph", say) is done when it comes back, and `visible` answers from what was visible when it
//! left. Panels floated in a window are not part of [`Workspace::to_node`] (the profile's
//! `layout = …`); D3's `window = …` lines will keep detached ones.
use super::layout::{self, Node};
use crate::ui::{dock, icons, theme, Action, DockTab, PanelInput, UiState};
use egui::{pos2, vec2, Align, Align2, Color32, FontId, Frame, Id, LayerId, Layout, Margin, Rect, Response, Stroke, Ui, UiBuilder, WidgetText};
use egui_dock::{DockArea, DockState, NodeIndex, Split, TabBodyStyle, TabIndex, TabStyle, TabViewer, Tree};
use std::collections::HashMap;

/// One tile of the tree: the 3D view, or a panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tile {
    View3d,
    Panel(DockTab),
}

/// The 3D view's share of a split made beside it ([`Workspace::show_panel`] with no right column,
/// a panel dropped into its leaf): the default layout's `H(0.70,3D,…)`.
const VIEW_SHARE: f32 = 0.70;
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

/// A change the panels asked for while the tree was lent out (done when it is back).
enum Change {
    Show(DockTab),
    Close(DockTab),
    Float(DockTab),
    Set(Node),
}

/// The workspace: the dock tree, and what the panels asked of it while it was lent out.
pub struct Workspace {
    dock: DockState<Tile>,
    /// While the dock area draws: the panels visible when the tree left (what [`Workspace::visible`]
    /// answers meanwhile).
    lent: Option<Vec<DockTab>>,
    pending: Vec<Change>,
}

impl Default for Workspace {
    /// The D1 default ([`layout::default_layout`]).
    fn default() -> Self {
        Workspace::from_node(&layout::default_layout())
    }
}

impl Workspace {
    fn from_node(n: &Node) -> Workspace {
        let mut w = Workspace { dock: build(n), lent: None, pending: Vec::new() };
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
    /// floating in a window go.
    pub fn set_node(&mut self, n: Node) {
        if self.lent.is_some() {
            self.pending.push(Change::Set(n));
            return;
        }
        let n = layout::clean(n).filter(layout::has_view).unwrap_or_else(layout::default_layout);
        self.dock = build(&n);
        self.normalize();
    }

    /// Whether `t` is on screen: in the tree (or a floating window) and the active tab of its leaf.
    pub fn visible(&self, t: DockTab) -> bool {
        if let Some(shown) = &self.lent {
            return shown.contains(&t);
        }
        self.dock
            .find_tab(&Tile::Panel(t))
            .is_some_and(|p| self.dock.leaf(p.node_path()).is_ok_and(|l| l.active == p.tab && !l.collapsed))
    }

    /// Shows `t`: makes it the active tab of its leaf (and focuses the leaf); a closed panel joins
    /// the right column's first leaf, or a new split right of the 3D view when there is none.
    pub fn show_panel(&mut self, t: DockTab) {
        if self.lent.is_some() {
            self.pending.push(Change::Show(t));
            return;
        }
        let tile = Tile::Panel(t);
        if let Some(p) = self.dock.find_tab(&tile) {
            let _ = self.dock.set_active_tab(p);
            self.dock.set_focused_node_and_surface(p.node_path());
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

    /// Closes `t` (its leaf goes when it was the last tab there).
    pub fn close_panel(&mut self, t: DockTab) {
        if self.lent.is_some() {
            self.pending.push(Change::Close(t));
            return;
        }
        if let Some(p) = self.dock.find_tab(&Tile::Panel(t)) {
            self.dock.remove_tab(p);
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

    /// The 3D view's body rect as last drawn.
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
        self.lent = Some(DockTab::PANELS.into_iter().filter(|&t| self.visible(t)).collect());
        std::mem::replace(&mut self.dock, DockState::new(Vec::new()))
    }

    /// The tree back from the dock area: its rules kept, then what the panels and the tabs asked
    /// for meanwhile done. True when a drop had to be undone (the pass is redrawn).
    fn give_back(&mut self, dock: DockState<Tile>, asked: Vec<Change>) -> bool {
        self.dock = dock;
        self.lent = None;
        let moved = self.normalize();
        let changes: Vec<Change> = std::mem::take(&mut self.pending).into_iter().chain(asked).collect();
        for c in changes {
            match c {
                Change::Show(t) => self.show_panel(t),
                Change::Close(t) => self.close_panel(t),
                Change::Float(t) => self.float_panel(t),
                Change::Set(n) => self.set_node(n),
            }
        }
        moved
    }
}

/// The dock tree of a layout: the root leaf holds the first leaf's tabs, and each split splits the
/// leaf holding its first child's (`Split::Right` / `Split::Below` keep the old leaf first, so
/// `frac` is the first child's share, as in the grammar).
fn build(n: &Node) -> DockState<Tile> {
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
fn node_of(tree: &Tree<Tile>, at: NodeIndex) -> Option<Node> {
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
fn first_leaf(tree: &Tree<Tile>, mut at: NodeIndex) -> Option<NodeIndex> {
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
    /// floating window's), where its leaf's close icon is painted.
    layers: HashMap<Tile, LayerId>,
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
        if let Some((c, count)) = dock::status(t, self.st, self.input) {
            let at = pos2(r.rect.right() - PAD - dot_room(t).unwrap_or(DOT_GAP) / 2.0 + 2.0, r.rect.center().y);
            match count {
                Some(n) => {
                    p.circle_filled(at, 7.5, c);
                    p.text(at, Align2::CENTER_CENTER, n, FontId::monospace(9.0), Color32::WHITE);
                }
                None => {
                    // the Activity dot pulses while work runs
                    let pulse = if t == DockTab::Activity { 0.85 + 0.15 * (r.ctx.input(|i| i.time) * 7.0).sin() as f32 } else { 1.0 };
                    p.circle_filled(at, 4.5, theme::PANEL);
                    p.circle_filled(at, 3.5 * pulse, c);
                }
            }
        }
        if r.middle_clicked() {
            self.asked.push(Change::Close(t));
        }
        if r.hovered() {
            let tip = dock::peek(self.st, self.input, t);
            let _ = r.clone().on_hover_text(format!("{tip}\n(drag the tab to another tab bar, to an edge to split, or out to float it; right-click for more)"));
        }
        r.context_menu(|ui| {
            if ui.button("Float in a window").clicked() {
                self.asked.push(Change::Float(t));
                ui.close();
            }
            if ui.button("Close").clicked() {
                self.asked.push(Change::Close(t));
                ui.close();
            }
            ui.separator();
            if ui.button("Reset the layout").clicked() {
                self.st.reset_layout(); // done when the tree is back (`Workspace::set_node`)
                ui.close();
            }
        });
    }

    fn is_closeable(&self, tab: &Tile) -> bool {
        *tab != Tile::View3d
    }

    fn allowed_in_windows(&self, tab: &mut Tile) -> bool {
        *tab != Tile::View3d
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
    // the leaf's close button: egui_dock's, its X left transparent under the duotone Close icon
    // (`paint_close_icons`)
    s.buttons.close_all_tabs_color = Color32::TRANSPARENT;
    s.buttons.close_all_tabs_active_color = Color32::TRANSPARENT;
    s.buttons.close_all_tabs_disabled_color = Color32::TRANSPARENT;
    s.buttons.close_all_tabs_bg_fill = theme::PANEL_2;
    s.buttons.close_all_tabs_border_color = theme::LINE_SOFT;
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
/// `egui_dock`'s close-all button at a tab bar's end (its `TAB_CLOSE_ALL_BUTTON_SIZE`: 24 pt wide,
/// the bar's height less 2), and the Close icon painted in it.
const CLOSE_BUTTON: f32 = 24.0;
const CLOSE_ICON: f32 = 13.0;

/// Paints the duotone Close icon in each shown tab bar's close button (whose own X is left
/// transparent, `style`), on the layer its leaf was drawn on; brighter while hovered.
fn paint_close_icons(ctx: &egui::Context, dock: &DockState<Tile>, layers: &HashMap<Tile, LayerId>, bar: f32) {
    for (_, leaf) in dock.iter_leaves() {
        if leaf.tab_bar_hidden || !leaf.rect.is_positive() {
            continue;
        }
        let Some(&layer) = leaf.tabs.first().and_then(|t| layers.get(t)) else { continue };
        let r = leaf.rect;
        let button = Rect::from_min_max(pos2(r.right() - CLOSE_BUTTON, r.top()), pos2(r.right(), r.top() + bar - 2.0));
        if let Some(Tile::Panel(t)) = leaf.tabs.first() {
            crate::ui::probe_rect(&format!("dock-close-{t:?}"), button);
        }
        let ink = if ctx.rect_contains_pointer(layer, button) { theme::INK } else { theme::INK_DIM };
        let p = ctx.layer_painter(layer).with_clip_rect(r);
        icons::paint(&p, Rect::from_center_size(button.center(), egui::Vec2::splat(CLOSE_ICON)), icons::Icon::Close, ink);
    }
}

/// Draws the workspace into what the top bar and the toolbar leave of `root`; returns the 3D
/// view's rect, which is then all the root `Ui` has left (so egui counts the pointer over it as
/// not over egui).
pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Rect {
    let area = root.available_rect_before_wrap();
    let style = style(root.style());
    let mut dock = st.workspace.lend();
    let asked = {
        let mut tiles = Tiles { st, input, acts, asked: Vec::new(), layers: HashMap::new() };
        let mut ui = root.new_child(UiBuilder::new().max_rect(area).layout(Layout::top_down(Align::Min)).id_salt("workspace"));
        DockArea::new(&mut dock)
            .id(Id::new("workspace"))
            .style(style)
            .show_close_buttons(false) // the leaf's close button ends its tab bar (spec D-4)
            .show_leaf_close_all_buttons(true)
            .show_leaf_collapse_buttons(false)
            .hidable_tab_bars(true) // only the 3D view's leaf hides its bar (`normalize`)
            .tab_context_menus(false) // our own (`on_tab_button`): float, close, reset
            .show_inside(&mut ui, &mut tiles);
        paint_close_icons(root.ctx(), &dock, &tiles.layers, TAB_BAR);
        tiles.asked
    };
    if st.workspace.give_back(dock, asked) {
        root.ctx().request_discard("a panel dropped into the 3D view's leaf went beside it");
    }
    let view = st.workspace.view_rect().filter(|r| r.is_positive() && area.expand(1.0).contains_rect(*r)).unwrap_or(area);
    leave_only(root, view);
    view
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
