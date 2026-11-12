//! Auto-hide (docking spec D-10, D2; user decision 4, "Pin = auto-hide (Visual Studio)"): an
//! unpinned panel leaves the tree for an icon on the strip of the 3D view's edge it was nearest
//! to, and slides out over the 3D view from there on hover or click; pinned panels stay docked
//! and take space. Display state only (spec D-13): nothing here is an `Action`.
//!
//! - **Strips**: an edge with hidden panels (left, right or bottom of the 3D view's tile) shows a
//!   [`STRIP`]-wide band of [`icons::strip_button`]s with their status dots (`dock::status`, as on
//!   the tabs); an edge without shows none. The strips are carved from the 3D view's tile, so the
//!   view the viewer renders (`PanelOutput::view`) is what they leave.
//! - **The flyout**: an [`egui::Area`] in the foreground (over the 3D view's own overlays) against
//!   its strip, sliding out over the 3D view at the panel's remembered size, with a title bar (icon, name, pin, close) over the panel's body. Being an
//!   area it takes the pointer over itself (egui counts it as over egui, so the 3D view does not
//!   orbit through it), and only there: elsewhere the 3D view keeps its input.
//! - **Timings** (egui's input time, so the tests drive simulated frames): hovering a strip icon
//!   for [`HOVER_OPEN`] opens its flyout; leaving the flyout and the strips for [`LEAVE_CLOSE`],
//!   Esc, or a press anywhere else hides it. A click on a strip icon — or a press inside the
//!   flyout, which is working in it (recorded for review) — keeps it open until the next press
//!   elsewhere or Esc. `UiState::open_tab` on a hidden panel opens its flyout the same way.
//! - **Pin / unpin** ([`super::Workspace::pin`]): each leaf's tab bar ends with a Pin button that
//!   unpins its active tab; the flyout's (tilted) pin docks the panel back beside the 3D view on
//!   its edge.
use super::layout::{Edge, Hidden};
use crate::ui::{dock, icons, probe_rect, theme, Action, DockTab, PanelInput, UiState};
use egui::{pos2, vec2, Align, Color32, Id, Key, LayerId, Layout, Margin, Order, Pos2, Rect, Response, RichText, Sense, Stroke, StrokeKind, Ui, UiBuilder};

/// A strip's width (or the bottom strip's height): the old docks' icon strip.
pub const STRIP: f32 = 44.0;
/// Hovering a strip icon this long opens its flyout (s).
pub const HOVER_OPEN: f64 = 0.25;
/// Leaving the flyout and the strips this long hides it (s).
pub const LEAVE_CLOSE: f64 = 0.5;
/// The flyout's slide out (s).
const SLIDE: f64 = 0.12;
/// A hidden panel's size when it had no rect to measure (unpinned before the tree was drawn).
pub const DEFAULT_SIZE: f32 = 320.0;

/// The flyout's state: which hidden panel is out, and the timers that open and hide it.
#[derive(Default)]
pub struct Flyout {
    /// The panel whose flyout is out.
    pub(super) open: Option<DockTab>,
    /// Opened by a click (or `open_tab`, or worked in): only a press elsewhere or Esc hides it.
    stuck: bool,
    /// The strip icon under the pointer, and since when (input time, s).
    hover: Option<(DockTab, f64)>,
    /// The strip icon the pointer rested on when the flyout was hidden (Esc, say): resting there
    /// does not reopen it; leaving the icon re-arms its hover.
    quiet: Option<DockTab>,
    /// Since when the pointer has been off the flyout and the strips.
    left_at: Option<f64>,
    /// When the open flyout was first drawn (its slide).
    opened_at: Option<f64>,
}

impl Flyout {
    /// Opens `t`'s flyout (stuck: until a press elsewhere). Opening it again keeps it out, stuck
    /// if either opening was.
    pub(super) fn open(&mut self, t: DockTab, stuck: bool) {
        if self.open == Some(t) {
            self.stuck |= stuck;
        } else {
            *self = Flyout { open: Some(t), stuck, hover: self.hover, quiet: None, left_at: None, opened_at: None };
        }
        self.left_at = None;
    }

    /// Hides the flyout; the strip icon under the pointer, if any, stays quiet until the pointer
    /// leaves it (its hover timer restarts then, rather than reopening at once).
    pub(super) fn close(&mut self) {
        *self = Flyout { quiet: self.hover.map(|(t, _)| t), ..Flyout::default() };
    }
}

/// The edge a panel whose rect is centred at `c` folds to: the side of the 3D view's rect `view`
/// it lies beyond (the farthest beyond, at a corner), or — over the view, or above it (there is no
/// top strip) — the nearest of the three.
pub fn edge_of(c: Pos2, view: Rect) -> Edge {
    let beyond = [(Edge::Left, view.left() - c.x), (Edge::Right, c.x - view.right()), (Edge::Bottom, c.y - view.bottom())];
    beyond.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(Edge::Right, |(e, _)| e)
}

/// Where the strips and the flyout go in the 3D view's tile.
struct Geometry {
    /// The 3D view the strips leave.
    view: Rect,
    /// Each edge's strip ([`Edge`] order), when it has hidden panels.
    strips: [Option<Rect>; 3],
}

impl Geometry {
    /// The strips of the edges in `used` ([`Edge::ALL`] order) carved from `tile`: the side strips
    /// run down to the bottom strip, which spans the tile. A tile too small to keep a view beside
    /// its strips shows none (the panels stay reachable from the Window menu).
    fn new(tile: Rect, used: [bool; 3]) -> Geometry {
        let [left, right, bottom] = used;
        let (w, h) = (STRIP * (left as u8 + right as u8) as f32, STRIP * bottom as u8 as f32);
        if !used.contains(&true) || tile.width() < w + STRIP || tile.height() < h + STRIP {
            return Geometry { view: tile, strips: [None; 3] };
        }
        let mut v = tile;
        if left {
            v.min.x += STRIP;
        }
        if right {
            v.max.x -= STRIP;
        }
        if bottom {
            v.max.y -= STRIP;
        }
        let strips = [
            left.then(|| Rect::from_min_max(tile.min, pos2(v.left(), v.bottom()))),
            right.then(|| Rect::from_min_max(pos2(v.right(), tile.top()), pos2(tile.right(), v.bottom()))),
            bottom.then(|| Rect::from_min_max(pos2(tile.left(), v.bottom()), tile.max)),
        ];
        Geometry { view: v, strips }
    }

    /// The flyout of a panel hidden on `edge` with `size`: against that edge of the view, at most
    /// the view's width (height).
    fn flyout(&self, edge: Edge, size: f32) -> Rect {
        let v = self.view;
        match edge {
            Edge::Left => Rect::from_min_max(v.min, pos2(v.left() + size.min(v.width()), v.bottom())),
            Edge::Right => Rect::from_min_max(pos2(v.right() - size.min(v.width()), v.top()), v.max),
            Edge::Bottom => Rect::from_min_max(pos2(v.left(), v.bottom() - size.min(v.height())), v.max),
        }
    }

    fn on_strip(&self, p: Pos2) -> bool {
        self.strips.iter().flatten().any(|s| s.contains(p))
    }
}

/// Whether `p` is on an open popup or menu (a combo box opened from the flyout spills out of it: a
/// press there is not "elsewhere"; the press that closes the popup leaves the flyout out too).
fn on_popup(ctx: &egui::Context, p: Pos2) -> bool {
    egui::Popup::is_any_open(ctx) && ctx.layer_id_at(p).is_some_and(|l| matches!(l.order, Order::Foreground | Order::Tooltip | Order::Debug))
}

/// The flyout's area: in the foreground, over the 3D view's own overlays (the status chip, the
/// capture buttons).
fn flyout_layer() -> LayerId {
    LayerId::new(Order::Foreground, Id::new("autohide-flyout"))
}

/// Draws the strips round the 3D view's `tile`, runs the flyout's timers and draws it; returns the
/// 3D view the strips leave.
pub fn show(root: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, tile: Rect) -> Rect {
    let ctx = root.ctx().clone();
    let hidden: Vec<Hidden> = st.workspace.hidden().to_vec();
    let g = Geometry::new(tile, Edge::ALL.map(|e| hidden.iter().any(|h| h.edge == e)));
    // a flyout whose panel is no longer hidden (pinned, closed, a new layout) goes
    if st.workspace.fly.open.is_some_and(|t| !hidden.iter().any(|h| h.tab == t)) {
        st.workspace.fly.close();
    }
    // the strips
    let (mut hovered, mut clicked) = (None, None);
    for (edge, strip) in Edge::ALL.into_iter().zip(g.strips) {
        let Some(strip) = strip else { continue };
        probe_rect(&format!("strip-{edge:?}"), strip);
        let p = root.painter();
        p.rect_filled(strip, 0.0, theme::APP);
        let line = match edge {
            Edge::Left => [strip.right_top(), strip.right_bottom()],
            Edge::Right => [strip.left_top(), strip.left_bottom()],
            Edge::Bottom => [strip.left_top(), strip.right_top()],
        };
        p.line_segment(line, Stroke::new(1.0, theme::LINE));
        let layout = if edge == Edge::Bottom { Layout::left_to_right(Align::Center) } else { Layout::top_down(Align::Center) };
        let mut ui = root.new_child(UiBuilder::new().max_rect(strip.shrink(5.0)).layout(layout).id_salt(("autohide-strip", edge)));
        ui.set_clip_rect(strip);
        ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
        for h in hidden.iter().filter(|h| h.edge == edge) {
            let r = icons::strip_button(&mut ui, dock::icon(h.tab), st.workspace.fly.open == Some(h.tab));
            probe_rect(&format!("strip-icon-{:?}", h.tab), r.rect);
            if let Some(s) = dock::status(h.tab, st, input) {
                super::tree::paint_status(ui.painter(), pos2(r.rect.right() - 7.0, r.rect.top() + 7.0), s, h.tab, ctx.input(|i| i.time));
            }
            if r.hovered() {
                hovered = Some(h.tab);
            }
            if r.clicked() {
                clicked = Some(h.tab);
            }
        }
    }
    // the timers
    let (now, at, pressed, down, esc) = ctx.input(|i| {
        (i.time, i.pointer.hover_pos(), i.pointer.any_pressed().then(|| i.pointer.press_origin()).flatten(), i.pointer.any_down(), i.key_pressed(Key::Escape))
    });
    let fly = &mut st.workspace.fly;
    if fly.quiet.is_some() && fly.quiet != hovered {
        fly.quiet = None; // the pointer left the icon it rested on
    }
    match hovered {
        Some(t) if fly.quiet == Some(t) => {}
        Some(t) => {
            let since = match fly.hover {
                Some((h, since)) if h == t => since,
                _ => now,
            };
            fly.hover = Some((t, since));
            if fly.open != Some(t) {
                if now - since >= HOVER_OPEN {
                    fly.open(t, false);
                } else {
                    ctx.request_repaint_after_secs((HOVER_OPEN - (now - since)) as f32);
                }
            }
        }
        None => fly.hover = None,
    }
    if let Some(t) = clicked {
        fly.open(t, true);
    }
    let out = fly.open.and_then(|t| hidden.iter().find(|h| h.tab == t).copied());
    let rect = out.map(|h| g.flyout(h.edge, h.size));
    let inside = |p: Pos2| rect.is_some_and(|r| r.contains(p)) || g.on_strip(p);
    if fly.open.is_some() {
        if esc && !ctx.egui_wants_keyboard_input() {
            fly.close();
        } else if let Some(p) = pressed {
            if rect.is_some_and(|r| r.contains(p)) {
                fly.stuck = true; // working in it
            } else if !inside(p) && !on_popup(&ctx, p) {
                fly.close();
            }
        } else if !fly.stuck {
            if down || at.is_some_and(inside) {
                fly.left_at = None;
            } else {
                let since = *fly.left_at.get_or_insert(now);
                if now - since >= LEAVE_CLOSE {
                    fly.close();
                } else {
                    ctx.request_repaint_after_secs((LEAVE_CLOSE - (now - since)) as f32);
                }
            }
        }
    }
    // the flyout
    let (Some(h), Some(rect)) = (out.filter(|h| fly.open == Some(h.tab)), rect) else { return g.view };
    if fly.opened_at.is_none() {
        ctx.move_to_top(flyout_layer()); // over whatever came to the top since it was last out
    }
    let opened = *fly.opened_at.get_or_insert(now);
    let k = ((now - opened) / SLIDE).clamp(0.0, 1.0) as f32;
    if k < 1.0 {
        ctx.request_repaint();
    }
    let hide = (1.0 - k) * (1.0 - k); // eased out: the share still to slide
    let shift = match h.edge {
        Edge::Left => vec2(-rect.width() * hide, 0.0),
        Edge::Right => vec2(rect.width() * hide, 0.0),
        Edge::Bottom => vec2(0.0, rect.height() * hide),
    };
    let (pin, close) = flyout(&ctx, st, input, acts, h.tab, rect.translate(shift), g.view);
    if pin {
        st.workspace.pin(h.tab, true);
    } else if close {
        st.workspace.close_panel(h.tab);
    }
    if pin || close {
        ctx.request_repaint(); // the tree drawn this frame did not have it yet
    }
    g.view
}

/// A small icon button (the flyout's title bar).
fn icon_button(ui: &mut Ui, icon: icons::Icon, tip: &str) -> Response {
    let (rect, r) = ui.allocate_exact_size(egui::Vec2::splat(24.0), Sense::click());
    if r.hovered() {
        ui.painter().rect_filled(rect, 5.0, theme::PANEL_3);
    }
    icons::paint(ui.painter(), Rect::from_center_size(rect.center(), egui::Vec2::splat(14.0)), icon, if r.hovered() { theme::INK } else { theme::INK_DIM });
    r.on_hover_text(tip)
}

/// The flyout of `t` at `rect`, clipped to the 3D view: its title bar (icon, name, pin, close)
/// over the panel's body. Returns whether its pin, its close button was clicked.
fn flyout(ctx: &egui::Context, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, t: DockTab, rect: Rect, view: Rect) -> (bool, bool) {
    let (mut pin, mut close) = (false, false);
    // the area is only the part over the view: while it slides out, the part still clipped away
    // must not take the pointer over the strip it slides from
    let shown = rect.intersect(view);
    egui::Area::new(flyout_layer().id)
        .order(flyout_layer().order)
        .fixed_pos(shown.min)
        .constrain(false)
        .movable(false)
        .show(ctx, |ui| {
            ui.set_clip_rect(view);
            probe_rect(&format!("flyout-{t:?}"), rect);
            // the whole of it on screen is the flyout's (the pointer anywhere on it is over egui)
            ui.allocate_rect(shown, Sense::hover());
            let p = ui.painter();
            p.add(egui::Shadow { offset: [0, 4], blur: 18, spread: 0, color: Color32::from_black_alpha(120) }.as_shape(rect, 6.0));
            p.rect(rect, 6.0, theme::PANEL, Stroke::new(1.0, theme::LINE), StrokeKind::Inside);
            let mut ui = ui.new_child(UiBuilder::new().max_rect(rect - Margin::same(8)).layout(Layout::top_down(Align::Min)).id_salt("autohide-flyout-body"));
            ui.set_clip_rect(rect.intersect(view));
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::Vec2::splat(16.0), Sense::hover());
                icons::paint(ui.painter(), r, dock::icon(t), theme::INK);
                ui.label(RichText::new(t.name()).strong().size(13.5).color(theme::INK));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let c = icon_button(ui, icons::Icon::Close, "close this panel (the Window menu brings it back)");
                    probe_rect(&format!("flyout-close-{t:?}"), c.rect);
                    let p = icon_button(ui, icons::Icon::Unpin, "pin: dock this panel back beside the 3D view");
                    probe_rect(&format!("flyout-pin-{t:?}"), p.rect);
                    (pin, close) = (p.clicked(), c.clicked());
                });
            });
            ui.add(egui::Separator::default().spacing(6.0));
            dock::body(&mut ui, st, input, acts, t);
        });
    (pin, close)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_edge_is_where_the_panel_lay() {
        let v = Rect::from_min_max(pos2(300.0, 50.0), pos2(1000.0, 800.0));
        assert_eq!(edge_of(pos2(150.0, 400.0), v), Edge::Left);
        assert_eq!(edge_of(pos2(1200.0, 400.0), v), Edge::Right);
        assert_eq!(edge_of(pos2(600.0, 850.0), v), Edge::Bottom);
        assert_eq!(edge_of(pos2(100.0, 900.0), v), Edge::Left, "below-left, farther left than below");
        assert_eq!(edge_of(pos2(600.0, 10.0), v), Edge::Left, "above the view: the nearer side");
        assert_eq!(edge_of(pos2(900.0, 10.0), v), Edge::Right);
        assert_eq!(edge_of(pos2(950.0, 400.0), v), Edge::Right, "over the view: the nearest edge");
        assert_eq!(edge_of(pos2(600.0, 780.0), v), Edge::Bottom);
    }

    #[test]
    fn the_strips_are_carved_from_the_tile() {
        let tile = Rect::from_min_max(pos2(0.0, 60.0), pos2(1400.0, 900.0));
        let g = Geometry::new(tile, [true, false, true]);
        assert_eq!(g.view, Rect::from_min_max(pos2(STRIP, 60.0), pos2(1400.0, 900.0 - STRIP)));
        assert_eq!(g.strips[0], Some(Rect::from_min_max(pos2(0.0, 60.0), pos2(STRIP, 900.0 - STRIP))));
        assert_eq!(g.strips[1], None, "no hidden panel on the right: no strip");
        assert_eq!(g.strips[2], Some(Rect::from_min_max(pos2(0.0, 900.0 - STRIP), pos2(1400.0, 900.0))));
        assert_eq!(g.flyout(Edge::Left, 340.0), Rect::from_min_max(pos2(STRIP, 60.0), pos2(STRIP + 340.0, 900.0 - STRIP)));
        assert_eq!(g.flyout(Edge::Bottom, 5000.0), g.view, "at most the view");
        let none = Geometry::new(tile, [false; 3]);
        assert_eq!((none.view, none.strips), (tile, [None; 3]));
        let tiny = Geometry::new(Rect::from_min_size(Pos2::ZERO, vec2(60.0, 60.0)), [true, true, true]);
        assert_eq!(tiny.strips, [None; 3], "no room: no strips");
    }
}
