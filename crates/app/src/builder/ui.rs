//! The builder's window (spec §4): a palette column (elements, tools), the canvas, and a footer
//! with the live formula and name. An `egui::Modal` like the File window, sized from the app's
//! window every frame; it closes on Close or Esc, never on a click outside (a stray click must not
//! end a drawing — the drawing is kept anyway).
//!
//! Canvas gestures (the default tool is Atom with carbon; the Atom tool and right-click revised by
//! the user's hand-test, 2026-10-20 — spec §4):
//! - Atom tool: click an atom to select it; with an atom selected, a click on empty space places a
//!   new atom of the palette's element bonded to it (a ghost bond and atom follow the cursor) and
//!   selects the new one, so clicks draw a chain — snapped to the fixed drawn bond length and the
//!   cursor's direction rounded to 30° ([`SNAP_DEG`]; Alt places freely; a snapped spot on an
//!   existing atom bonds to it, closing a ring); with nothing selected a click places a lone atom.
//!   Click the selected atom again: it becomes the palette's element. Shift-click another atom: a
//!   bond to it (a ring closure). Drag an atom: it moves (the selection with it);
//! - right-click deselects; Esc deselects, and closes the window when nothing is selected;
//! - click a bond (Bond tool, or the Atom tool): its order cycles 1 → 2 → 3 → 1; Bond tool: drag
//!   from atom to atom for a bond;
//! - Charge tool: click an atom to cycle 0 → +1 → −1 → 0;
//! - Erase tool, Delete: removes the atom (and its bonds) or the bond under the cursor (Delete
//!   with nothing under it: the selection);
//! - Select/Move: drag an atom (the whole selection when it is in it), drag on empty space to
//!   box-select, Shift adds;
//! - wheel zooms about the cursor, middle- or right-drag pans, Ctrl-Z / Ctrl-Y (Ctrl-Shift-Z) undo /
//!   redo.
use super::{Builder, Tool};
use crate::ui::{theme, Action, PanelInput, UiState};
use egui::{Align2, Color32, FontId, Id, PointerButton, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};
use render::elements::symbol;

/// Points per canvas unit at zoom 1 (a drawn bond is about one unit).
pub const UNIT: f32 = 46.0;
/// The drawn atom's radius at zoom 1 (points).
const ATOM_R: f32 = 13.0;
/// The palette's elements (spec §4); "more…" opens the periodic table.
const PALETTE: [u8; 8] = [1, 6, 7, 8, 9, 15, 16, 17];
/// The rule-1 line (spec §3: the window states it).
pub const RULE_1_LINE: &str = "bond lengths and valences here only place the atoms; the solver decides the real geometry and bonds";

/// What a press on the canvas started, while the button is held.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Gesture {
    #[default]
    None,
    /// A bond being drawn from this atom to the cursor.
    Bond { from: usize },
    /// These atoms follow the cursor.
    Move { atoms: Vec<usize> },
    /// A selection rectangle from this screen point.
    Box { from: Pos2 },
}

/// Opens the window (File ▸ Builder…); the last drawing is still there.
pub fn open(st: &mut UiState) {
    st.builder.open = true;
}

/// The canvas's mapping between canvas units and screen points.
#[derive(Clone, Copy)]
struct View {
    centre: Pos2,
    pan: [f32; 2],
    scale: f32,
}

impl View {
    fn screen(&self, p: [f32; 2]) -> Pos2 {
        self.centre + Vec2::new(p[0] - self.pan[0], p[1] - self.pan[1]) * self.scale
    }

    fn canvas(&self, s: Pos2) -> [f32; 2] {
        let d = (s - self.centre) / self.scale;
        [self.pan[0] + d.x, self.pan[1] + d.y]
    }
}

/// What the cursor is over.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Hit {
    Atom(usize),
    Bond(usize),
}

fn atom_radius(zoom: f32) -> f32 {
    ATOM_R * zoom.clamp(0.5, 2.0)
}

/// The atom under `p` (the nearest within its radius and a few points), else the bond.
fn hit(b: &Builder, view: &View, p: Pos2, skip: Option<usize>) -> Option<Hit> {
    let r = atom_radius(b.zoom) + 4.0;
    let atom = b
        .drawing
        .atoms
        .iter()
        .enumerate()
        .filter(|&(i, _)| Some(i) != skip)
        .map(|(i, a)| (i, view.screen(a.pos).distance(p)))
        .filter(|&(_, d)| d <= r)
        .min_by(|x, y| x.1.total_cmp(&y.1));
    if let Some((i, _)) = atom {
        return Some(Hit::Atom(i));
    }
    if skip.is_some() {
        return None;
    }
    b.drawing
        .bonds
        .iter()
        .enumerate()
        .map(|(k, bd)| (k, segment_distance(p, view.screen(b.drawing.atoms[bd.a].pos), view.screen(b.drawing.atoms[bd.b].pos))))
        .filter(|&(_, d)| d <= 7.0)
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map(|(k, _)| Hit::Bond(k))
}

fn segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

/// The window, while open.
pub fn show(ctx: &egui::Context, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if !st.builder.open {
        return;
    }
    let screen = ctx.content_rect();
    let size = Vec2::new((screen.width() * 0.82).min(1500.0), screen.height() * 0.86).max(Vec2::new(560.0, 420.0));
    let inner = size - Vec2::splat(28.0);
    let mut close = false;
    egui::Modal::new(Id::new("builder-window"))
        .frame(
            egui::Frame::new()
                .fill(theme::PANEL)
                .stroke(Stroke::new(1.0, theme::LINE))
                .corner_radius(10)
                .inner_margin(14)
                .shadow(egui::Shadow { offset: [0, 8], blur: 24, spread: 0, color: Color32::from_black_alpha(140) }),
        )
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(inner, Sense::hover());
            let header = Rect::from_min_size(rect.min, Vec2::new(rect.width(), 30.0));
            let footer_h = 150.0;
            let footer = Rect::from_min_max(egui::pos2(rect.left(), rect.bottom() - footer_h), rect.max);
            let body = Rect::from_min_max(egui::pos2(rect.left(), header.bottom() + 6.0), egui::pos2(rect.right(), footer.top() - 8.0));
            let palette_w = 150.0;
            let palette = Rect::from_min_size(body.min, Vec2::new(palette_w, body.height()));
            let canvas = Rect::from_min_max(egui::pos2(palette.right() + 10.0, body.top()), body.max);
            child(ui, header, egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.label(RichText::new("Molecule builder").size(17.0).strong());
                ui.label(RichText::new("sketch the structural formula; the scene gets the atoms").size(12.0).color(theme::INK_FAINT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").on_hover_text("Esc (once nothing is selected) — the drawing is kept for next time").clicked() {
                        close = true;
                    }
                });
            });
            child(ui, palette, egui::Layout::top_down(egui::Align::Min), |ui| palette_column(ui, st));
            canvas_area(ui, canvas, &mut st.builder);
            child(ui, footer, egui::Layout::top_down(egui::Align::Min), |ui| {
                footer_rows(ui, st);
                ui.add_space(4.0);
                if handoff_rows(ui, st, input, acts) {
                    close = true;
                }
            });
        });
    // Esc deselects, or closes when nothing is selected — unless a popover (the periodic table) or
    // a text field has it
    if !egui::Popup::is_any_open(ctx) && !ctx.egui_wants_keyboard_input() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
        if st.builder.selection.is_empty() {
            close = true;
        } else {
            st.builder.selection.clear();
        }
    }
    if close {
        st.builder.open = false;
        st.builder.gesture = Gesture::None;
    }
}

/// A child `Ui` filling exactly `rect`.
fn child(ui: &mut Ui, rect: Rect, layout: egui::Layout, f: impl FnOnce(&mut Ui)) {
    let mut c = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(layout));
    c.set_clip_rect(rect.expand(2.0));
    f(&mut c);
}

fn palette_column(ui: &mut Ui, st: &mut UiState) {
    let b = &mut st.builder;
    theme::section(ui, "Element");
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(4.0);
        for z in PALETTE {
            let on = b.element == z;
            let [r, g, bl] = render::elements::colour_srgb(z);
            let text = RichText::new(symbol(z)).size(14.0).strong().color(if on { Color32::WHITE } else { theme::INK });
            let btn = egui::Button::new(text).min_size(Vec2::new(32.0, 28.0)).fill(if on { theme::ACCENT_BG } else { theme::PANEL_2 }).stroke(Stroke::new(1.0, if on { theme::ACCENT } else { Color32::from_rgb(r, g, bl).gamma_multiply(0.6) }));
            if ui.add(btn).on_hover_text(format!("{} — click on the canvas to place it", crate::names::element_name(z))).clicked() {
                pick_element(b, z);
            }
        }
        let more = ui.add(egui::Button::new(RichText::new("more…").size(12.5)).min_size(Vec2::new(68.0, 28.0)));
        let more = more.on_hover_text(format!("the periodic table (Z 1–{})", crate::ui::ptable::Z_ENABLED));
        egui::Popup::from_toggle_button_response(&more)
            .id(Id::new("builder-ptable"))
            .align(egui::RectAlign::RIGHT_START)
            .gap(6.0)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Periodic table").strong());
                    let r = ui.add(egui::TextEdit::singleline(&mut b.ptable_query).hint_text("symbol, name or Z").desired_width(150.0));
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Some(z) = crate::ui::ptable::search(&b.ptable_query).into_iter().find(|&z| crate::ui::ptable::enabled(z)) {
                            pick_element(b, z);
                            ui.close();
                        }
                    }
                });
                let q = b.ptable_query.clone();
                if let Some(z) = crate::ui::ptable::grid(ui, &q, Some(b.element), "builder-ptable") {
                    pick_element(b, z);
                    ui.close();
                }
            });
    });
    if !PALETTE.contains(&b.element) {
        ui.label(RichText::new(format!("placing {} ({})", crate::names::element_name(b.element), symbol(b.element))).size(12.0).color(theme::ACCENT_INK));
    }
    ui.add_space(10.0);
    theme::section(ui, "Tool");
    for (tool, name, tip) in [
        (Tool::Atom, "Atom", "click an atom to select it, then click nearby: a new atom bonded to it at the fixed bond length and the nearest 30° direction (Alt: exactly where you click), and so on along a chain — a click that lands on an existing atom bonds to it; with nothing selected a click places a lone atom; right-click deselects; click the selected atom again to change its element; Shift-click another atom to bond to it; drag an atom to move it"),
        (Tool::Bond, "Bond", "click a bond: single → double → triple; drag atom to atom: a bond"),
        (Tool::Charge, "Charge", "click an atom: 0 → +1 → −1 → 0"),
        (Tool::Erase, "Erase", "click an atom or a bond to remove it (Delete does the same with any tool)"),
        (Tool::Select, "Select / Move", "drag an atom (the selection with it), drag on empty space to select a group; Shift adds"),
    ] {
        let on = b.tool == tool;
        let btn = egui::Button::new(RichText::new(name).size(13.0).color(if on { Color32::WHITE } else { theme::INK }))
            .min_size(Vec2::new(ui.available_width(), 26.0))
            .fill(if on { theme::ACCENT_BG } else { theme::PANEL_2 })
            .stroke(Stroke::new(1.0, if on { theme::ACCENT } else { theme::LINE }));
        if ui.add(btn).on_hover_text(tip).clicked() {
            b.tool = tool;
            b.gesture = Gesture::None;
        }
    }
    ui.add_space(10.0);
    theme::section(ui, "Groups");
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(4.0);
        for s in super::chem::Stamp::ALL {
            let btn = egui::Button::new(RichText::new(s.label()).size(13.0)).min_size(Vec2::new(44.0, 26.0)).fill(theme::PANEL_2);
            if ui.add(btn).on_hover_text(s.tip()).clicked() {
                let at = b.pan;
                b.stamp(s, at);
            }
        }
    });
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        if ui.add_enabled(b.can_undo(), egui::Button::new("↶ Undo")).on_hover_text("Ctrl-Z").clicked() {
            b.undo();
        }
        if ui.add_enabled(b.can_redo(), egui::Button::new("↷ Redo")).on_hover_text("Ctrl-Y").clicked() {
            b.redo();
        }
    });
}

/// Picking an element switches to the Atom tool (as the toolbar's chips switch to Place).
fn pick_element(b: &mut Builder, z: u8) {
    b.element = z;
    b.tool = Tool::Atom;
}

fn canvas_area(ui: &mut Ui, rect: Rect, b: &mut Builder) {
    let resp = ui.interact(rect, Id::new("builder-canvas"), Sense::click_and_drag());
    let view = View { centre: rect.center(), pan: b.pan, scale: UNIT * b.zoom };
    let pointer = ui.input(|i| i.pointer.latest_pos());
    let shift = ui.input(|i| i.modifiers.shift);
    let over = pointer.filter(|p| rect.contains(*p));
    let hovered = if b.gesture == Gesture::None { over.and_then(|p| hit(b, &view, p, None)) } else { None };

    // ── zoom and pan ──
    if resp.hovered() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let f = (scroll / 300.0).exp() * pinch;
        if (f - 1.0).abs() > 1e-4 {
            if let Some(p) = over {
                let c = view.canvas(p);
                b.zoom = (b.zoom * f).clamp(0.25, 4.0);
                let s = UNIT * b.zoom;
                b.pan = [c[0] - (p.x - rect.center().x) / s, c[1] - (p.y - rect.center().y) / s];
            }
        }
    }
    if resp.dragged_by(PointerButton::Middle) || resp.dragged_by(PointerButton::Secondary) {
        let d = resp.drag_delta() / view.scale;
        b.pan = [b.pan[0] - d.x, b.pan[1] - d.y];
    }

    // ── clicks ──
    if resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            click(b, &view, p, shift, ui.input(|i| i.modifiers.alt));
        }
    }
    // right-click deselects (a right-drag pans)
    if resp.secondary_clicked() {
        b.selection.clear();
    }

    // ── primary drags ──
    if resp.drag_started_by(PointerButton::Primary) {
        let origin = ui.input(|i| i.pointer.press_origin()).unwrap_or(rect.center());
        let at = hit(b, &view, origin, None);
        b.gesture = match (b.tool, at) {
            (Tool::Bond, Some(Hit::Atom(i))) => Gesture::Bond { from: i },
            (Tool::Atom | Tool::Select, Some(Hit::Atom(i))) => {
                b.checkpoint();
                if !b.selection.contains(&i) {
                    if !shift {
                        b.selection.clear();
                    }
                    b.selection.insert(i);
                }
                Gesture::Move { atoms: b.selection.iter().copied().collect() }
            }
            (Tool::Select, _) => Gesture::Box { from: origin },
            _ => Gesture::None,
        };
    }
    if resp.dragged_by(PointerButton::Primary) {
        if let Gesture::Move { atoms } = &b.gesture {
            let d = resp.drag_delta() / view.scale;
            for &i in atoms {
                if let Some(a) = b.drawing.atoms.get_mut(i) {
                    a.pos = [a.pos[0] + d.x, a.pos[1] + d.y];
                }
            }
        }
    }
    if resp.drag_stopped_by(PointerButton::Primary) {
        let end = pointer.unwrap_or(rect.center());
        match std::mem::take(&mut b.gesture) {
            Gesture::Bond { from } if from < b.drawing.atoms.len() => match hit(b, &view, end, Some(from)) {
                Some(Hit::Atom(j)) => {
                    b.checkpoint();
                    b.drawing.bond(from, j);
                }
                _ => {
                    let to = view.canvas(end);
                    let a = b.drawing.atoms[from].pos;
                    // a short wiggle is no bond (the press already made its atom, if any)
                    if ((to[0] - a[0]).powi(2) + (to[1] - a[1]).powi(2)).sqrt() > 0.35 && rect.contains(end) {
                        b.checkpoint();
                        let j = b.drawing.add_atom(b.element, to);
                        b.drawing.bond(from, j);
                        b.selection = [j].into();
                    }
                }
            },
            Gesture::Box { from } => {
                let r = Rect::from_two_pos(from, end);
                if !shift {
                    b.selection.clear();
                }
                for (i, a) in b.drawing.atoms.iter().enumerate() {
                    if r.contains(view.screen(a.pos)) {
                        b.selection.insert(i);
                    }
                }
            }
            _ => {}
        }
    }

    // ── keys (the window has them: the 3D view ignores input while it is open) ──
    if !ui.ctx().egui_wants_keyboard_input() {
        let (redo, undo, delete) = ui.input_mut(|i| {
            let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z) || i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y);
            let undo = i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z);
            let delete = i.consume_key(egui::Modifiers::NONE, egui::Key::Delete) || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace);
            (redo, undo, delete)
        });
        if redo {
            b.redo();
        }
        if undo {
            b.undo();
        }
        if delete {
            match hovered {
                Some(Hit::Atom(i)) => b.erase_atom(i),
                Some(Hit::Bond(k)) => b.erase_bond(k),
                None => b.erase_selection(),
            }
        }
    }

    // the Atom tool's ghost: where a click would put the next atom (bonded to the selection), or the
    // atom a Shift-click would bond the selection to
    let anchor = b.selected_one();
    let free = ui.input(|i| i.modifiers.alt);
    let ghost = over.filter(|_| b.tool == Tool::Atom && b.gesture == Gesture::None && resp.hovered()).and_then(|p| match hovered {
        None => {
            let (at, onto) = placement(b, view.canvas(p), free);
            Some((anchor, view.screen(at), onto))
        }
        Some(Hit::Atom(j)) if shift && anchor.is_some_and(|a| a != j) => Some((anchor, p, Some(j))),
        _ => None,
    });
    let marks = super::chem::warnings(&b.drawing);
    paint(ui, rect, b, &view, hovered, pointer, &marks, ghost);
    if hovered.is_some() || b.gesture != Gesture::None {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
}

/// The Atom tool's snap (user request 2026-10-20): a bond from the selected atom is drawn at a
/// fixed length ([`chem::drawn_length`](super::chem::drawn_length)) in the cursor's direction
/// rounded to this many degrees — 120° zigzags, 90° crosses, straight chains.
pub const SNAP_DEG: f32 = 30.0;

/// An existing atom this close (canvas units) to a snapped spot is bonded to instead of stacking a
/// new atom on it (the sixth click round a hexagon closes the ring).
const JOIN_WITHIN: f32 = 0.3;

/// Where an Atom-tool click at canvas point `to` puts its atom: bonded to the one selected atom,
/// at the fixed drawn length and the snapped angle — or at the cursor when `free` (Alt) or when
/// nothing is selected. Also the existing atom that spot falls on, if any (bonded to instead).
fn placement(b: &Builder, to: [f32; 2], free: bool) -> ([f32; 2], Option<usize>) {
    let Some(a) = b.selected_one().filter(|_| !free) else { return (to, None) };
    let from = b.drawing.atoms[a].pos;
    let step = SNAP_DEG.to_radians();
    let angle = ((to[1] - from[1]).atan2(to[0] - from[0]) / step).round() * step;
    let len = super::chem::drawn_length(b.drawing.atoms[a].z, b.element);
    let at = [from[0] + len * angle.cos(), from[1] + len * angle.sin()];
    let onto = b
        .drawing
        .atoms
        .iter()
        .enumerate()
        .filter(|&(j, x)| j != a && ((x.pos[0] - at[0]).powi(2) + (x.pos[1] - at[1]).powi(2)).sqrt() < JOIN_WITHIN)
        .map(|(j, _)| j)
        .next();
    (at, onto)
}

/// A primary click on the canvas at `p`, by tool (`free`: Alt, the Atom tool places without the
/// snap).
fn click(b: &mut Builder, view: &View, p: Pos2, shift: bool, free: bool) {
    let at = hit(b, view, p, None);
    match (b.tool, at) {
        (Tool::Atom, Some(Hit::Atom(i))) => match b.selected_one() {
            // Shift-click another atom: bond them (a ring closure), and go on from it
            Some(a) if shift && a != i => {
                b.checkpoint();
                b.drawing.bond(a, i);
                b.selection = [i].into();
            }
            // the selected atom again: it becomes the palette's element
            Some(a) if a == i => {
                if b.drawing.atoms[i].z != b.element {
                    b.checkpoint();
                    b.drawing.set_z(i, b.element);
                }
            }
            _ => b.selection = [i].into(),
        },
        (Tool::Atom | Tool::Bond, Some(Hit::Bond(k))) => {
            b.checkpoint();
            b.drawing.cycle_order(k);
        }
        // a new atom, bonded to the selected one if there is one (at the snapped spot, or to the
        // atom already there); it becomes the selection
        (Tool::Atom, None) => {
            let from = b.selected_one();
            let (at, onto) = placement(b, view.canvas(p), free);
            b.checkpoint();
            let i = match onto {
                Some(j) => j,
                None => b.drawing.add_atom(b.element, at),
            };
            if let Some(a) = from {
                if b.drawing.bond_between(a, i).is_none() {
                    b.drawing.bond(a, i);
                }
            }
            b.selection = [i].into();
        }
        (Tool::Charge, Some(Hit::Atom(i))) => {
            b.checkpoint();
            b.drawing.cycle_charge(i);
        }
        (Tool::Erase, Some(Hit::Atom(i))) => b.erase_atom(i),
        (Tool::Erase, Some(Hit::Bond(k))) => b.erase_bond(k),
        (Tool::Select, Some(Hit::Atom(i))) => {
            if shift {
                if !b.selection.remove(&i) {
                    b.selection.insert(i);
                }
            } else {
                b.selection = [i].into();
            }
        }
        (Tool::Select, None) if !shift => b.selection.clear(),
        _ => {}
    }
}

/// The warning rings (spec §4): over the allowed valence red, under it amber.
const OVER: Color32 = Color32::from_rgb(0xff, 0x6b, 0x6b);
const UNDER: Color32 = Color32::from_rgb(0xf0, 0xb4, 0x5a);

/// The Atom tool's ghost: the anchor (the selected atom, if any), the cursor, and the atom a
/// Shift-click would bond to instead of placing a new one.
type Ghost = (Option<usize>, Pos2, Option<usize>);

#[allow(clippy::too_many_arguments)]
fn paint(ui: &Ui, rect: Rect, b: &Builder, view: &View, hovered: Option<Hit>, pointer: Option<Pos2>, marks: &[(usize, super::chem::Warning)], ghost: Option<Ghost>) {
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 8.0, theme::APP);
    p.rect_stroke(rect, 8.0, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
    // a dot every canvas unit (when they are far enough apart to read as a grid)
    if view.scale >= 18.0 {
        let a = view.canvas(rect.min);
        let z = view.canvas(rect.max);
        let mut x = a[0].floor();
        while x <= z[0] {
            let mut y = a[1].floor();
            while y <= z[1] {
                p.circle_filled(view.screen([x, y]), 1.0, theme::LINE_SOFT);
                y += 1.0;
            }
            x += 1.0;
        }
    }
    let d = &b.drawing;
    let r = atom_radius(b.zoom);
    let ink = Stroke::new(2.0 * b.zoom.clamp(0.7, 1.6), theme::INK_DIM);
    for (k, bd) in d.bonds.iter().enumerate() {
        let (s, e) = (view.screen(d.atoms[bd.a].pos), view.screen(d.atoms[bd.b].pos));
        let dir = (e - s).normalized();
        if !dir.is_finite() || (e - s).length() < 2.0 * r * 0.9 {
            continue;
        }
        let (s, e) = (s + dir * r * 0.95, e - dir * r * 0.95);
        let stroke = if hovered == Some(Hit::Bond(k)) { Stroke::new(ink.width + 1.0, theme::ACCENT) } else { ink };
        let n = dir.rot90() * 3.5 * b.zoom.clamp(0.7, 1.6);
        let offsets: &[f32] = match bd.order {
            2 => &[-1.0, 1.0],
            3 => &[-1.6, 0.0, 1.6],
            _ => &[0.0],
        };
        for &o in offsets {
            p.line_segment([s + n * o, e + n * o], stroke);
        }
    }
    // the bond being drawn
    if let (Gesture::Bond { from }, Some(ptr)) = (&b.gesture, pointer) {
        if let Some(a) = d.atoms.get(*from) {
            let s = view.screen(a.pos);
            let snap = hit(b, view, ptr, Some(*from));
            let end = match snap {
                Some(Hit::Atom(j)) => view.screen(d.atoms[j].pos),
                _ => ptr,
            };
            p.add(egui::Shape::dashed_line(&[s, end], Stroke::new(2.0, theme::ACCENT), 6.0, 4.0));
            if snap.is_none() {
                p.circle_stroke(end, r, Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.6)));
                p.text(end, Align2::CENTER_CENTER, symbol(b.element), FontId::proportional(13.0 * b.zoom.clamp(0.6, 1.8)), theme::ACCENT_INK);
            }
        }
    }
    if let Some((anchor, at, onto)) = ghost {
        let end = onto.map_or(at, |j| view.screen(d.atoms[j].pos));
        if let Some(a) = anchor.and_then(|a| d.atoms.get(a)) {
            p.add(egui::Shape::dashed_line(&[view.screen(a.pos), end], Stroke::new(2.0, theme::ACCENT.gamma_multiply(0.7)), 6.0, 4.0));
        }
        if onto.is_none() {
            p.circle_stroke(end, r, Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.6)));
            p.text(end, Align2::CENTER_CENTER, symbol(b.element), FontId::proportional(13.0 * b.zoom.clamp(0.6, 1.8)), theme::ACCENT_INK.gamma_multiply(0.7));
        }
    }
    for (i, a) in d.atoms.iter().enumerate() {
        let c = view.screen(a.pos);
        if !rect.expand(r).contains(c) {
            continue;
        }
        let [cr, cg, cb] = render::elements::colour_srgb(a.z);
        let col = Color32::from_rgb(cr, cg, cb);
        p.circle_filled(c, r, blend(theme::PANEL_2, col, 0.45));
        p.circle_stroke(c, r, Stroke::new(1.5, col));
        match marks.iter().find(|m| m.0 == i).map(|m| m.1) {
            Some(super::chem::Warning::Over) => {
                p.circle_stroke(c, r + 6.5, Stroke::new(2.5, OVER));
            }
            Some(super::chem::Warning::Under) if b.filled => {
                p.circle_stroke(c, r + 6.5, Stroke::new(2.0, UNDER));
            }
            _ => {}
        }
        if b.selection.contains(&i) {
            p.circle_stroke(c, r + 3.5, Stroke::new(2.0, theme::ACCENT));
        } else if hovered == Some(Hit::Atom(i)) {
            p.circle_stroke(c, r + 3.0, Stroke::new(1.5, theme::INK_DIM));
        }
        let fs = 13.0 * b.zoom.clamp(0.6, 1.8);
        p.text(c, Align2::CENTER_CENTER, symbol(a.z), FontId::proportional(fs), Color32::WHITE);
        if a.charge != 0 {
            let at = c + Vec2::new(r * 0.85, -r * 0.85);
            p.circle_filled(at, fs * 0.48, theme::PANEL);
            p.circle_stroke(at, fs * 0.48, Stroke::new(1.0, theme::INK_DIM));
            p.text(at, Align2::CENTER_CENTER, if a.charge > 0 { "+" } else { "−" }, FontId::proportional(fs * 0.8), theme::INK);
        }
    }
    if let (Gesture::Box { from }, Some(ptr)) = (&b.gesture, pointer) {
        let r = Rect::from_two_pos(*from, ptr);
        p.rect_filled(r, 2.0, theme::ACCENT.gamma_multiply(0.08));
        p.rect_stroke(r, 2.0, Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
    }
    if d.atoms.is_empty() {
        p.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "click to place an atom · click nearby for one bonded to it · right-click deselects · wheel zooms, right-drag pans",
            FontId::proportional(14.0),
            theme::INK_FAINT,
        );
    }
}

/// `a` → `b` by `t` in sRGB (a display tint).
fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}

fn footer_rows(ui: &mut Ui, st: &mut UiState) {
    let style = st.name_style;
    let b = &mut st.builder;
    ui.separator();
    ui.horizontal(|ui| {
        let label = b.drawing.label(style);
        if label.is_empty() {
            ui.label(RichText::new("nothing drawn yet").size(15.0).color(theme::INK_FAINT));
        } else {
            ui.label(crate::ui::rich(&label, 16.0, theme::INK));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_enabled(!b.drawing.atoms.is_empty(), egui::Button::new(RichText::new("Fill hydrogens").strong()))
                .on_hover_text("every atom gets hydrogens up to its usual valence (with its formal charge); the hydrogens become real atoms")
                .clicked()
            {
                b.fill_hydrogens();
            }
            if ui.add_enabled(!b.drawing.atoms.is_empty(), egui::Button::new("Clear")).on_hover_text("an empty canvas (Ctrl-Z brings it back)").clicked() {
                b.clear();
            }
        });
    });
    // the warnings (amber ones only once Fill hydrogens was used)
    let marks: Vec<_> = super::chem::warnings(&b.drawing).into_iter().filter(|m| b.filled || m.1 == super::chem::Warning::Over).collect();
    if !marks.is_empty() {
        let over = marks.iter().filter(|m| m.1 == super::chem::Warning::Over).count();
        let under = marks.len() - over;
        let mut words = Vec::new();
        if over > 0 {
            words.push(format!("{over} over its valence"));
        }
        if under > 0 {
            words.push(format!("{under} under it (a radical, or a missing bond)"));
        }
        ui.label(RichText::new(format!("⚠ {} atom{}: {}", marks.len(), if marks.len() == 1 { "" } else { "s" }, words.join(", "))).size(12.5).color(if over > 0 { OVER } else { UNDER }))
            .on_hover_text(super::chem::describe(&b.drawing, &marks));
    }
    ui.label(RichText::new(RULE_1_LINE).size(11.5).color(theme::INK_FAINT));
}

/// Whether a hand-off fits the cell (spec §6): as it is, in the accepted enlarged grid, or not.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fit {
    Now,
    Enlarged((usize, f64)),
    /// Does not fit: the farthest coordinate and the limit of the grid now.
    No { reach: f64, limit: f64 },
}

/// The hand-off row: the fit check and the enlarge offer, the warnings' confirm line, Relax
/// after, Insert and Replace. Returns true when a hand-off was sent (the window closes).
fn handoff_rows(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> bool {
    use super::{embed, enlarge_n, fit_limit, max_abs};
    let walls_on = st.walls_on;
    let wall_now = st.wall_half_width;
    let (backend, gpu) = (st.backend, st.gpu_available);
    let b = &mut st.builder;
    if b.drawing.atoms.is_empty() {
        ui.label(RichText::new("draw a molecule to put it into the scene").size(12.5).color(theme::INK_FAINT));
        return false;
    }
    let scene = input.scene;
    let (n0, h0) = (scene.grid_n, scene.grid_h);
    let new = b.embedding().to_vec();
    let existing: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
    let mut beside = new.clone();
    embed::place_beside(&existing, &mut beside);
    let reach_replace = max_abs(&new);
    let reach_insert = max_abs(&beside).max(max_abs(&existing));
    // the walls of a grid: today's on the grid now, else the default for that cell
    let walls_of = |n: usize, h: f64| -> Option<f64> {
        walls_on.then(|| if (n, h) == (n0, h0) { wall_now } else { crate::ui::default_wall_half_width(&crate::scene::Scene { grid_n: n, grid_h: h, ..scene.clone() }) })
    };
    let accepted = b.enlarge.filter(|&(n, h)| h == h0 && n > n0);
    let fit = |reach: f64| {
        let limit = fit_limit(n0, h0, walls_of(n0, h0));
        if reach <= limit {
            Fit::Now
        } else if let Some(g) = accepted.filter(|&(n, h)| reach <= fit_limit(n, h, walls_of(n, h))) {
            Fit::Enlarged(g)
        } else {
            Fit::No { reach, limit }
        }
    };
    let (fit_r, fit_i) = (fit(reach_replace), fit(reach_insert));

    // ── the enlarge offer (spec §6), with the memory guard's verdict before it is accepted ──
    let misfit = [fit_r, fit_i].into_iter().filter_map(|f| if let Fit::No { reach, limit } = f { Some((reach, limit)) } else { None }).fold(None, |m: Option<(f64, f64)>, (r, l)| Some(m.map_or((r, l), |(mr, _)| (mr.max(r), l))));
    if let Some((reach, limit)) = misfit {
        let n = enlarge_n(reach, h0, n0);
        let mut probe = scene.clone();
        probe.grid_n = n;
        probe.nuclei.extend(new.iter().zip(&b.drawing.atoms).map(|(&pos, a)| crate::scene::SceneNucleus { z: a.z, a: None, pos, vel: [0.0; 3] }));
        let solvers = crate::quality::gpu_solvers(backend, gpu);
        let verdict = crate::quality::vram_check(&probe, solvers, crate::quality::vram_budget());
        let memory = if solvers == 0 {
            "CPU solver: no GPU memory limit, but a larger grid is slower".to_string()
        } else {
            let need = probe.to_sim().map(|s| solvers * providers::gpu_footprint_bytes(&s)).unwrap_or(0);
            format!("≈ {:.1} GB of the {} GPU budget", need as f64 / 1e9, crate::quality::vram_budget().gb())
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!("does not fit the {n0}³ × {h0:.3} Bohr box: the molecule reaches ±{reach:.1} Bohr, the box{} holds ±{limit:.1}", if walls_on { " with its walls" } else { "" }))
                    .size(12.5)
                    .color(UNDER),
            );
            match &verdict {
                Ok(()) => {
                    let r = ui.button(format!("Enlarge the box to {n}³")).on_hover_text(format!(
                        "{n}³ × {h0:.3} Bohr: ±{:.1} Bohr, {ENLARGE_MARGIN_TEXT}; {memory}{}",
                        (n as f64 - 1.0) * h0 / 2.0,
                        if walls_on { "; the walls move out with it" } else { "" }
                    ));
                    ui.label(RichText::new(memory.clone()).size(11.5).color(theme::INK_FAINT));
                    if r.clicked() {
                        b.enlarge = Some((n, h0));
                    }
                }
                Err(e) => {
                    ui.add_enabled(false, egui::Button::new(format!("Enlarge the box to {n}³"))).on_disabled_hover_text(e.clone());
                    ui.label(RichText::new("over the GPU memory budget").size(11.5).color(OVER)).on_hover_text(e.clone());
                }
            }
        });
    } else if let Some((n, _)) = [fit_r, fit_i].into_iter().find_map(|f| if let Fit::Enlarged(g) = f { Some(g) } else { None }) {
        ui.label(RichText::new(format!("the box will be enlarged to {n}³ (from {n0}³) on hand-off")).size(12.5).color(theme::ACCENT_INK));
    }

    // ── the warnings' confirm line ──
    let marks = super::chem::warnings(&b.drawing);
    let armed = b.armed.as_ref().filter(|(_, d)| *d == b.drawing).map(|(r, _)| *r);
    if let (Some(replace), false) = (armed, marks.is_empty()) {
        ui.label(
            RichText::new(format!("{} — press {} again to hand it over anyway", super::chem::describe(&b.drawing, &marks), if replace { "Replace scene" } else { "Insert into scene" }))
                .size(12.5)
                .color(UNDER),
        );
    }

    // ── the buttons ──
    let mut sent = false;
    ui.horizontal(|ui| {
        ui.checkbox(&mut b.relax_after, "Relax after").on_hover_text("relax the geometry on the solver's own surface once the atoms are in (FIRE); the guess here is only a start");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            for (replace, fit, label, tip) in [
                (true, fit_r, "Replace scene", "a new scene of these atoms (the grid, walls and ionic potential kept; time restarts)"),
                (false, fit_i, "Insert into scene", "add these atoms beside the scene's (3 Bohr apart at least)"),
            ] {
                let can = !matches!(fit, Fit::No { .. });
                let r = ui.add_enabled(can, egui::Button::new(RichText::new(label).strong()).min_size(Vec2::new(130.0, 30.0)));
                let r = if can { r.on_hover_text(tip) } else { r.on_disabled_hover_text("does not fit the box: enlarge it first") };
                if r.clicked() {
                    if !marks.is_empty() && armed != Some(replace) {
                        b.armed = Some((replace, b.drawing.clone()));
                        continue;
                    }
                    let atoms = if replace { &new } else { &beside };
                    let z: Vec<u8> = b.drawing.atoms.iter().map(|a| a.z).collect();
                    acts.push(Action::Builder(super::Handoff {
                        replace,
                        nuclei: z.iter().copied().zip(atoms.iter().copied()).collect(),
                        charge: b.drawing.charge(),
                        relax: b.relax_after,
                        grid: if let Fit::Enlarged(g) = fit { Some(g) } else { None },
                        label: crate::names::formula(&z, None),
                    }));
                    b.armed = None;
                    b.enlarge = None;
                    sent = true;
                }
            }
        });
    });
    sent
}

/// The enlarge offer's margin, in words.
const ENLARGE_MARGIN_TEXT: &str = "4 Bohr to spare from the farthest atom to the box";

#[cfg(test)]
mod tests {
    use super::*;

    /// The Atom tool as the user asked for it (2026-10-20): click places a lone atom and selects it,
    /// the next click elsewhere places one bonded to it, deselect, then a lone one again; clicking the
    /// selected atom changes its element; Shift-click bonds two existing atoms.
    #[test]
    fn atom_tool_clicks_chain_and_deselect_starts_a_lone_atom() {
        let mut b = Builder::default();
        let view = View { centre: Pos2::ZERO, pan: [0.0, 0.0], scale: UNIT };
        let at = |x: f32, y: f32| Pos2::new(x * UNIT, y * UNIT);
        click(&mut b, &view, at(0.0, 0.0), false, false);
        click(&mut b, &view, at(1.0, 0.0), false, false);
        click(&mut b, &view, at(2.0, 0.5), false, false);
        assert_eq!(b.drawing.atoms.len(), 3);
        assert_eq!(b.drawing.bonds, vec![super::super::Bond { a: 0, b: 1, order: 1 }, super::super::Bond { a: 1, b: 2, order: 1 }]);
        assert_eq!(b.selected_one(), Some(2));
        // right-click's deselect, then a lone atom
        b.selection.clear();
        click(&mut b, &view, at(0.0, 3.0), false, false);
        assert_eq!(b.drawing.atoms.len(), 4);
        assert_eq!(b.drawing.bonds.len(), 2, "no bond to a lone atom");
        // clicking an unselected atom selects it; clicking it again changes it to the palette's element
        b.element = 8;
        click(&mut b, &view, at(1.0, 0.0), false, false);
        assert_eq!((b.selected_one(), b.drawing.atoms[1].z), (Some(1), 6));
        click(&mut b, &view, at(1.0, 0.0), false, false);
        assert_eq!(b.drawing.atoms[1].z, 8);
        // Shift-click another atom bonds it to the selection (a ring closure)
        click(&mut b, &view, at(0.0, 3.0), true, false);
        assert!(b.drawing.bond_between(1, 3).is_some());
        assert_eq!(b.selected_one(), Some(3));
        // everything is undoable, one click at a time
        let mut steps = 0;
        while b.can_undo() {
            b.undo();
            steps += 1;
        }
        assert_eq!((steps, b.drawing.atoms.len()), (6, 0));
    }

    /// The snap (user request 2026-10-20): a bonded atom lands at the fixed drawn length, in the
    /// cursor's direction rounded to 30°; to a hydrogen it is shorter; Alt places at the cursor;
    /// six clicks round a hexagon close the ring onto the first atom.
    #[test]
    fn bonded_atoms_snap_to_fixed_length_and_30_degree_angles() {
        let mut b = Builder::default();
        let view = View { centre: Pos2::ZERO, pan: [0.0, 0.0], scale: UNIT };
        let at = |x: f32, y: f32| Pos2::new(x * UNIT, y * UNIT);
        let close = |p: [f32; 2], q: [f32; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt() < 1e-4;
        click(&mut b, &view, at(0.0, 0.0), false, false);
        // 1.7 units away at 37°: one unit away at 30°
        click(&mut b, &view, at(1.7 * 37f32.to_radians().cos(), 1.7 * 37f32.to_radians().sin()), false, false);
        let p = b.drawing.atoms[1].pos;
        assert!(close(p, [30f32.to_radians().cos(), 30f32.to_radians().sin()]), "{p:?}");
        // a hydrogen from it: 0.8 long, straight down (−87° rounds to −90°)
        b.element = 1;
        click(&mut b, &view, at(p[0] + 0.05, p[1] + 2.0), false, false);
        assert!(close(b.drawing.atoms[2].pos, [p[0], p[1] + 0.8]), "{:?}", b.drawing.atoms[2].pos);
        // Alt: exactly where clicked
        click(&mut b, &view, at(3.3, 2.2), false, true);
        assert!(close(b.drawing.atoms[3].pos, [3.3, 2.2]));
        // a hexagon by six clicks: the sixth snaps onto the first atom and closes the ring
        let mut h = Builder::default();
        click(&mut h, &view, at(0.0, 0.0), false, false);
        for k in 0..6 {
            let a = (k as f32 * 60.0).to_radians();
            let from = h.drawing.atoms[h.selected_one().unwrap()].pos;
            click(&mut h, &view, at(from[0] + 1.5 * a.cos(), from[1] + 1.5 * a.sin()), false, false);
        }
        assert_eq!((h.drawing.atoms.len(), h.drawing.bonds.len()), (6, 6), "a closed ring, no stacked atom");
        assert_eq!(h.selected_one(), Some(0));
        for bd in &h.drawing.bonds {
            let (p, q) = (h.drawing.atoms[bd.a].pos, h.drawing.atoms[bd.b].pos);
            assert!((((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt() - 1.0).abs() < 1e-4);
        }
    }
}
