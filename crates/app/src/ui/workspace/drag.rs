//! Dragging tabs between windows (docking spec D-15, user request 2026-11-12; spec D-5: nothing
//! floats inside a window any more). Inside one window a tab's drag is `egui_dock`'s own, as
//! before; this follows a drag past its window's edge. Display state only (spec D-13): nothing here
//! is an `Action`, nothing reaches the worker.
//!
//! - **The latch:** the tab being dragged is read from its tab button's response each frame
//!   (`Tiles::on_tab_button`: `egui_dock` draws a dragged tab as its own widget,
//!   `id.with("dragged")`, so `ctx.dragged_id()` names it — no private state of the library),
//!   with its leaf's size and, from the main window, the panels it shares that leaf with (where it
//!   docks back to, spec D-12). A button pressed in a window keeps reporting the pointer to that
//!   window while it is held, outside it too (winit on Windows captures the mouse), so the source
//!   window's own frames see the whole drag; its pointer goes to desktop pixels through that
//!   window's client origin and its context's pixels per point.
//! - **Where the windows are** ([`Place`], [`Workspace::set_places`]): each window's client area
//!   and outer frame in desktop pixels, top-most first (the window side keeps them in focus
//!   order); each window's pixels per point are read from its own context as it draws, so with
//!   mixed DPI each converts with its own. Headless, with no places given, only the source's own
//!   screen is known (at the desktop origin).
//! - **Over another window:** that window draws, at the cursor, the drop targets a drag inside it
//!   would show ([`over`]: `egui_dock`'s widget overlay geometry and look — a leaf's centre tabs
//!   the panel in, its four edge buttons split, its tab bar inserts before the tab under the
//!   pointer or at the end; the main window's 3D view tile has the edge buttons only, spec D-3).
//!   Every window is drawn in each of the main window's frames, so it keeps drawing them while the
//!   drag is over it (one frame behind when it draws before the source).
//! - **The release**, seen by the source window: over another window, on a target — the tab moves
//!   there ([`Change::DropTab`]); on no target — nothing; over none of the app's windows — a new
//!   detached window holding the tab, its top-left at the release point, the panel's current size,
//!   clamped into the work area of the monitor under the cursor ([`Change::TearOut`]); over its own
//!   window — `egui_dock`'s (a drop, or nothing: with `allowed_in_windows` false it never floats).
//!   Esc cancels (and is the drag's alone that frame: [`Workspace::drag_escape`]). A detached
//!   window whose last tab leaves closes, and a drag whose source window closes is forgotten.
//! - **Overlapping windows** (Task 11 review I1): the source's `egui_dock` knows nothing of a
//!   window covering it, and resolves a drop for whatever of its own leaves lies under the
//!   pointer. While a drag from it is on, its tree is kept as it was before its dock area drew,
//!   and a release over another window (or over none) puts that back first — so a release on no
//!   target there changes nothing, and one on a target moves the tab once, ours.
//! - **The capture pop-out view** is one of the app's windows too, with no drop targets
//!   ([`Place::targets`] false): a release over it cancels.
use super::detached::{panels_of, WinRect, MIN_WINDOW};
use super::layout::Node;
use super::tree::{Change, Tile, Workspace, TAB_BAR};
use crate::ui::DockTab;
use egui::{vec2, Context, Id, LayerId, Order, Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};
use egui_dock::{DockState, NodeIndex, OverlayStyle, Split, TabIndex};
use std::collections::HashMap;

/// Where one of the app's windows is on the desktop (physical pixels, winit's units): its client
/// area (inner position and size) and its outer frame (title bar and borders included). `win`:
/// detached window N, `None` for the main window; `targets`: it has a tree to drop a tab into
/// (false for the capture pop-out view, whose `win` means nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    pub win: Option<u32>,
    pub inner: WinRect,
    pub outer: WinRect,
    pub targets: bool,
}

/// A drop target in a window's tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Into leaf `leaf`'s tabs: before tab `at` (on the tab bar), else at the end (its centre
    /// button, or the bar past its last tab).
    Tab { leaf: NodeIndex, at: Option<usize> },
    /// A new leaf on `side` of leaf `leaf` (its edge buttons), half of it, as `egui_dock` splits.
    Split { leaf: NodeIndex, side: Split },
}

/// The tab being dragged.
struct TabDrag {
    tab: DockTab,
    /// The window it is dragged from (`None`: the main one).
    from: Option<u32>,
    /// Its leaf's size when the drag began, physical pixels (a new window's size).
    size: [u32; 2],
    /// The panels it shared its main-tree leaf with (`None`: dragged from a detached window).
    mates: Option<Vec<DockTab>>,
    /// The pointer, desktop pixels, as last seen.
    at: Option<[f64; 2]>,
    /// Esc was pressed: the release does nothing.
    cancelled: bool,
}

/// The cross-window drag and what it needs to know of the windows ([`Workspace::xd`]).
#[derive(Default)]
pub(super) struct Drags {
    drag: Option<TabDrag>,
    places: Vec<Place>,
    /// Each window's pixels per point, its screen (points) and its tab buttons' rects, as it last drew.
    ppp: HashMap<Option<u32>, f32>,
    screens: HashMap<Option<u32>, Rect>,
    bars: HashMap<Option<u32>, Vec<(Tile, Rect)>>,
}

/// A new window's least size (points; the window side's `with_min_inner_size`).
const MIN_SIZE: Vec2 = vec2(MIN_WINDOW[0], MIN_WINDOW[1]);

impl Workspace {
    /// Where the app's windows are (physical desktop pixels), top-most first: the window side
    /// reports them every frame.
    pub fn set_places(&mut self, places: Vec<Place>) {
        self.xd.places = places;
    }

    /// The panel whose tab is being dragged between windows, if any (the 3D view takes no keys
    /// meanwhile: Esc is the drag's).
    pub fn dragging(&self) -> Option<DockTab> {
        self.xd.drag.as_ref().map(|d| d.tab)
    }

    /// Whether a drag from window `win` is on (its tree is then kept from before its dock area
    /// draws, for a release over another window: see the module doc).
    pub(super) fn drag_from(&self, win: Option<u32>) -> bool {
        self.xd.drag.as_ref().is_some_and(|d| d.from == win)
    }

    /// Esc during a drag from window `win`, before anything else in its frame reads the keys: it
    /// cancels the drag (and `egui_dock`'s own drop with it) and is consumed, so it neither
    /// deselects nor hides a flyout (Task 11 review M4).
    pub fn drag_escape(&mut self, ctx: &Context, win: Option<u32>) {
        let Some(d) = self.xd.drag.as_mut().filter(|d| d.from == win && !d.cancelled) else { return };
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            d.cancelled = true;
            ctx.stop_dragging();
        }
    }

    /// Forgets a drag from detached window `id` (it closed under the drag: the other windows stop
    /// drawing its targets).
    pub(super) fn forget_drag_from(&mut self, id: u32) {
        if self.drag_from(Some(id)) {
            self.xd.drag = None;
        }
    }

    fn ppp(&self, win: Option<u32>) -> f32 {
        self.xd.ppp.get(&win).copied().filter(|k| *k > 0.0).unwrap_or(1.0)
    }

    /// Where window `win` is: as reported, else — headless, nothing reported — its own screen at
    /// the desktop origin.
    fn place(&self, win: Option<u32>) -> Option<Place> {
        if let Some(p) = self.xd.places.iter().find(|p| p.win == win && p.targets) {
            return Some(*p);
        }
        if !self.xd.places.is_empty() {
            return None;
        }
        let (r, k) = (self.xd.screens.get(&win)?, self.ppp(win));
        let px = WinRect { x: 0, y: 0, w: (r.right() * k).round().max(0.0) as u32, h: (r.bottom() * k).round().max(0.0) as u32 };
        Some(Place { win, inner: px, outer: px, targets: true })
    }

    /// A point of window `win` (points) in desktop pixels.
    fn desk(&self, win: Option<u32>, p: Pos2) -> [f64; 2] {
        let k = self.ppp(win) as f64;
        let (x, y) = self.place(win).map_or((0, 0), |p| (p.inner.x, p.inner.y));
        [x as f64 + p.x as f64 * k, y as f64 + p.y as f64 * k]
    }

    /// The app's window at desktop point `d`, the top-most first (`None`: none of them), its
    /// outer frame counting; `from` is the drag's source (headless, the only one known).
    fn window_at(&self, d: [f64; 2], from: Option<u32>) -> Option<Place> {
        if self.xd.places.is_empty() {
            return self.place(from).filter(|p| contains(&p.outer, d));
        }
        self.xd.places.iter().find(|p| contains(&p.outer, d)).copied()
    }

    /// The window with drop targets at desktop point `d` (`None` for none, or the pop-out view).
    fn target_window_at(&self, d: [f64; 2], from: Option<u32>) -> Option<Option<u32>> {
        self.window_at(d, from).filter(|p| p.targets).map(|p| p.win)
    }

    /// Desktop point `d` in window `win`'s client area, points (`None`: not in it).
    fn local(&self, win: Option<u32>, d: [f64; 2]) -> Option<Pos2> {
        let p = self.place(win)?;
        let k = self.ppp(win) as f64;
        contains(&p.inner, d).then(|| Pos2::new(((d[0] - p.inner.x as f64) / k) as f32, ((d[1] - p.inner.y as f64) / k) as f32))
    }

    /// Window `win`'s tree (`None`: the main one; a detached window that is gone has none).
    fn tree(&self, win: Option<u32>) -> Option<&DockState<Tile>> {
        match win {
            None => Some(&self.dock),
            Some(id) => self.det.list.iter().find(|w| w.id == id).map(|w| &w.dock),
        }
    }

    fn tree_mut(&mut self, win: Option<u32>) -> Option<&mut DockState<Tile>> {
        match win {
            None => Some(&mut self.dock),
            Some(id) => self.det.list.iter_mut().find(|w| w.id == id).map(|w| &mut w.dock),
        }
    }

    /// One window's part of a cross-window drag, after its dock area drew (its tree still lent:
    /// `dock`, with its tab buttons `bars` and the tab being dragged in it, `dragged`; `before`:
    /// the tree before the dock area drew, kept while a drag from this window is on): follows a
    /// drag from this window, acts on its release (the changes returned, done when the tree is
    /// back; a release over another window, or none, first puts `before` back, undoing a drop
    /// `egui_dock` made under the window covering it), and draws the drop targets of another
    /// window's drag over this one.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn drag_frame(&mut self, ctx: &Context, win: Option<u32>, dock: &mut DockState<Tile>, before: Option<DockState<Tile>>, bars: &[(Tile, Rect)], dragged: Option<Tile>, overlay: &OverlayStyle) -> Vec<Change> {
        self.xd.ppp.insert(win, ctx.pixels_per_point());
        self.xd.screens.insert(win, ctx.content_rect());
        self.xd.bars.insert(win, bars.to_vec());
        let pointer = ctx.input(|i| i.pointer.latest_pos().or(i.pointer.interact_pos()));
        let mut out = Vec::new();
        if let Some(Tile::Panel(t)) = dragged {
            if !self.xd.drag.as_ref().is_some_and(|d| d.tab == t && d.from == win) {
                let k = ctx.pixels_per_point();
                let leaf = dock.find_tab(&Tile::Panel(t)).and_then(|p| dock.leaf(p.node_path()).ok());
                let size = leaf.map(|l| l.rect.size()).filter(|s| s.x > 0.0 && s.y > 0.0).unwrap_or(MIN_SIZE).max(MIN_SIZE);
                let mates = win.is_none().then(|| leaf.map(|l| panels_of(&l.tabs).into_iter().filter(|&p| p != t).collect()).unwrap_or_default());
                self.xd.drag = Some(TabDrag { tab: t, from: win, size: [(size.x * k).round() as u32, (size.y * k).round() as u32], mates, at: None, cancelled: false });
            }
            let at = pointer.map(|p| self.desk(win, p));
            if let Some(d) = self.xd.drag.as_mut() {
                d.at = at.or(d.at);
            }
            ctx.request_repaint();
        } else if self.xd.drag.as_ref().is_some_and(|d| d.from == win) {
            // the drag ended in this frame: a release (anything else — Esc — does nothing)
            let d = self.xd.drag.take().expect("a drag");
            let released = ctx.input(|i| i.pointer.primary_released());
            if let (true, Some(at)) = (released, pointer.map(|p| self.desk(win, p)).or(d.at)) {
                let over = self.window_at(at, win);
                let own = over.is_some_and(|p| p.targets && p.win == win);
                // released over another window, or none: `egui_dock` here may have dropped the
                // tab on a leaf of this window that lies under it — undone (review I1)
                if let Some(b) = before.filter(|_| !own) {
                    if shape(&b) != shape(dock) {
                        ctx.request_discard("a drop under another window was undone");
                    }
                    *dock = b;
                }
                match over {
                    _ if d.cancelled || own => {} // cancelled, or egui_dock's own: a drop, or nothing
                    Some(p) if !p.targets => {}   // the pop-out view: nothing to drop into
                    Some(p) => {
                        if let Some(target) = self.target_in(p.win, at, overlay) {
                            out.push(Change::DropTab(d.tab, p.win, target, d.mates));
                        }
                    }
                    None => out.push(Change::TearOut(d.tab, self.tear_rect(at, d.size), d.mates)),
                }
            }
        }
        // another window's drag over this one: its drop targets at the cursor
        let shown = match &self.xd.drag {
            Some(d) if d.from != win && !d.cancelled => d.at.filter(|&at| self.target_window_at(at, d.from) == Some(win)).and_then(|at| self.local(win, at)),
            _ => None,
        };
        if let Some(over) = shown.and_then(|p| over(dock, bars, p, win.is_none(), overlay)) {
            paint(ctx, &over, overlay);
            ctx.request_repaint();
        }
        out
    }

    /// The drop target at desktop point `d` in window `w` (as it last drew).
    fn target_in(&self, w: Option<u32>, d: [f64; 2], overlay: &OverlayStyle) -> Option<Target> {
        let p = self.local(w, d)?;
        let bars = self.xd.bars.get(&w).map_or(&[][..], |b| &b[..]);
        over(self.tree(w)?, bars, p, w.is_none(), overlay)?.target()
    }

    /// A torn-out tab's window: its top-left at `at`, `size` large, kept inside the work area of
    /// the monitor under `at` (none known: where it is).
    fn tear_rect(&self, at: [f64; 2], size: [u32; 2]) -> WinRect {
        let (x, y) = (at[0].round() as i32, at[1].round() as i32);
        let r = WinRect { x, y, w: size[0].max(1), h: size[1].max(1) };
        match self.work_area_at(x, y) {
            Some(a) => {
                let (w, h) = (r.w.min(a.w), r.h.min(a.h));
                WinRect { x: x.clamp(a.x, a.x + (a.w - w) as i32), y: y.clamp(a.y, a.y + (a.h - h) as i32), w, h }
            }
            None => r,
        }
    }

    /// Moves `t` onto `at` in window `to` (spec D-15): out of wherever it is (an emptied detached
    /// window closes), then into the target's leaf or a new split beside it. Nothing happens when
    /// the target is gone, or `t` is already in that window.
    pub(super) fn drop_tab(&mut self, t: DockTab, to: Option<u32>, at: Target, mates: Option<Vec<DockTab>>) {
        if self.lent.is_some() {
            self.pending.push(Change::DropTab(t, to, at, mates));
            return;
        }
        let leaf = match at {
            Target::Tab { leaf, .. } | Target::Split { leaf, .. } => leaf,
        };
        let there = self.tree(to).is_some_and(|d| {
            let tree = d.main_surface();
            d.find_tab(&Tile::Panel(t)).is_none() && leaf.0 < tree.len() && tree[leaf].is_leaf() && !(matches!(at, Target::Tab { .. }) && tree[leaf].iter_tabs().any(|x| *x == Tile::View3d))
        });
        if !there || !self.take_out(t) {
            return;
        }
        match (to, mates) {
            (None, _) => {
                self.det.origins.remove(&t);
            }
            (Some(_), Some(m)) => {
                self.det.origins.insert(t, m);
            }
            (Some(_), None) => {}
        }
        let tile = Tile::Panel(t);
        let Some(dock) = self.tree_mut(to) else { return };
        let tree = dock.main_surface_mut();
        let node = match at {
            Target::Tab { leaf, at } => {
                match at {
                    Some(i) => {
                        let i = i.min(tree[leaf].tabs_count());
                        tree[leaf].insert_tab(TabIndex(i), tile);
                    }
                    None => tree[leaf].append_tab(tile),
                }
                leaf
            }
            Target::Split { leaf, side } => tree.split(leaf, side, 0.5, egui_dock::Node::leaf(tile))[1],
        };
        tree.set_focused_node(node);
        if to.is_none() {
            self.normalize();
        }
    }

    /// Moves `t` into a new detached window at `rect` (spec D-15: dragged out of every window).
    pub(super) fn tear_out(&mut self, t: DockTab, rect: WinRect, mates: Option<Vec<DockTab>>) {
        if self.lent.is_some() {
            self.pending.push(Change::TearOut(t, rect, mates));
            return;
        }
        if !self.take_out(t) {
            return;
        }
        if let Some(m) = mates {
            self.det.origins.insert(t, m);
        }
        self.new_window(Node::Leaf { tabs: vec![t], active: 0 }, Some(rect), false, None, false);
    }
}

/// Where every tab of a tree is (each leaf's index and tabs): whether `egui_dock` moved one.
fn shape(d: &DockState<Tile>) -> Vec<(usize, Vec<Tile>)> {
    d.iter_all_nodes().filter_map(|(p, n)| match n {
        egui_dock::Node::Leaf(l) => Some((p.node.0, l.tabs.clone())),
        _ => None,
    }).collect()
}

/// Whether desktop point `d` is in `r`.
fn contains(r: &WinRect, d: [f64; 2]) -> bool {
    (r.x as f64..r.x as f64 + r.w as f64).contains(&d[0]) && (r.y as f64..r.y as f64 + r.h as f64).contains(&d[1])
}

/// What the pointer is over in a window's tree during another window's drag.
enum Over {
    /// A tab bar: the tab the panel would go before (`None`: the end), and the rect lit.
    Bar { leaf: NodeIndex, at: Option<usize>, lit: Rect },
    /// A leaf's body: its rect, its drop buttons (the centre — not on the 3D view's tile — then
    /// below, right, above, left: `egui_dock`'s order, the last one hit wins) and the one hit.
    Body { leaf: NodeIndex, rect: Rect, buttons: Vec<(Option<Split>, Rect)>, hot: Option<usize> },
}

impl Over {
    fn target(&self) -> Option<Target> {
        match *self {
            Over::Bar { leaf, at, .. } => Some(Target::Tab { leaf, at }),
            Over::Body { leaf, ref buttons, hot, .. } => hot.map(|i| match buttons[i].0 {
                None => Target::Tab { leaf, at: None },
                Some(side) => Target::Split { leaf, side },
            }),
        }
    }
}

/// What point `p` (points) is over in `dock` as last drawn (`bars`: its tab buttons' rects; `main`:
/// the main window, whose 3D view tile takes edge splits only, spec D-3) — `egui_dock`'s
/// geometry: over a shown tab bar, the tab under the pointer or the bar's end; over a leaf's body,
/// its five buttons (`resolve_icon_based`), hit within `interact_expansion` of each.
fn over(dock: &DockState<Tile>, bars: &[(Tile, Rect)], p: Pos2, main: bool, overlay: &OverlayStyle) -> Option<Over> {
    let (path, leaf) = dock.iter_leaves().find(|(path, l)| path.surface.is_main() && l.rect.is_positive() && l.rect.contains(p))?;
    let view = leaf.tabs.contains(&Tile::View3d);
    if !leaf.tab_bar_hidden && !view {
        let bar = Rect::from_min_size(leaf.rect.min, vec2(leaf.rect.width(), TAB_BAR));
        if bar.contains(p) {
            let on = leaf.tabs.iter().enumerate().find_map(|(i, t)| bars.iter().find(|(b, r)| b == t && r.contains(p)).map(|(_, r)| (i, *r)));
            return Some(match on {
                Some((i, r)) => Over::Bar { leaf: path.node, at: Some(i), lit: r },
                None => Over::Bar { leaf: path.node, at: None, lit: bar },
            });
        }
    }
    let buttons = buttons(leaf.rect, overlay, !(view && main));
    let hot = buttons.iter().rposition(|(_, r)| r.shrink(r.width() * 0.1).expand(overlay.feel.interact_expansion).contains(p));
    Some(Over::Body { leaf: path.node, rect: leaf.rect, buttons, hot })
}

/// A leaf's drop buttons (`egui_dock`'s `resolve_icon_based`): squares of the leaf's least third
/// (at most `max_button_size`), `button_spacing` apart round its centre.
fn buttons(rect: Rect, overlay: &OverlayStyle, centre: bool) -> Vec<(Option<Split>, Rect)> {
    let s = overlay.button_spacing;
    let r = rect.shrink(s);
    let side = ((r.width() - 2.0 * s) / 3.0).min((r.height() - 2.0 * s) / 3.0).min(overlay.max_button_size);
    let (c, off) = (r.center(), side + s);
    let mut out = Vec::new();
    if centre {
        out.push((None, Rect::from_center_size(c, Vec2::splat(side))));
    }
    for (split, o) in [(Split::Below, vec2(0.0, off)), (Split::Right, vec2(off, 0.0)), (Split::Above, vec2(0.0, -off)), (Split::Left, vec2(-off, 0.0))] {
        out.push((Some(split), Rect::from_center_size(c + o, Vec2::splat(side))));
    }
    out
}

/// Draws what the pointer is over as `egui_dock` draws a drag's overlay inside its window: the
/// lit tab or tab bar, or the leaf's buttons with the hit one's half (all of the centre) lit.
fn paint(ctx: &Context, over: &Over, overlay: &OverlayStyle) {
    let p = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("workspace-drop-targets")));
    match over {
        Over::Bar { lit, .. } => {
            crate::ui::probe_rect("drop-target", *lit);
            p.rect_filled(*lit, 0.0, overlay.selection_color);
        }
        Over::Body { rect, buttons, hot, .. } => {
            let h = &overlay.hovered_leaf_highlight;
            p.rect(rect.expand(h.expansion), h.corner_radius, h.color, h.stroke, StrokeKind::Inside);
            for (i, &(split, r)) in buttons.iter().enumerate() {
                if Some(i) == *hot {
                    crate::ui::probe_rect("drop-target", r);
                }
                button(&p, r, split, Some(i) == *hot, overlay);
            }
        }
    }
}

/// The dashes across a split button (`egui_dock`'s `DASHED_LINE_ALPHAS`).
const DASHES: [f32; 8] = [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375];

/// One drop button, as `egui_dock`'s `button_ui` draws it.
fn button(p: &Painter, rect: Rect, split: Option<Split>, hot: bool, overlay: &OverlayStyle) {
    let stroke = Stroke::new(1.0, overlay.button_color);
    p.rect_stroke(rect, 0.0, overlay.button_border_stroke, StrokeKind::Inside);
    let rect = rect.shrink(rect.width() * 0.1);
    p.rect_stroke(rect, 0.0, stroke, StrokeKind::Inside);
    p.rect_filled(Rect::from_two_pos(rect.min, rect.lerp_inside(vec2(1.0, 0.1))), 0.0, overlay.button_color);
    let along = |s: Split, a: f32| if matches!(s, Split::Above | Split::Below) { vec2(a, 0.5) } else { vec2(0.5, a) };
    if let Some(s) = split {
        for d in DASHES.chunks(2) {
            p.line_segment([rect.lerp_inside(along(s, d[0])), rect.lerp_inside(along(s, d[1]))], stroke);
        }
    }
    if hot {
        let lit = match split {
            Some(Split::Above) => Rect::from_min_size(rect.min, rect.size() * vec2(1.0, 0.5)),
            Some(Split::Left) => Rect::from_min_size(rect.min, rect.size() * vec2(0.5, 1.0)),
            Some(Split::Below) => Rect::from_min_size(rect.lerp_inside(along(Split::Below, 0.0)), rect.size() * vec2(1.0, 0.5)),
            Some(Split::Right) => Rect::from_min_size(rect.lerp_inside(along(Split::Right, 0.0)), rect.size() * vec2(0.5, 1.0)),
            None => rect,
        };
        p.rect_filled(lit, 0.0, overlay.selection_color);
    }
}
