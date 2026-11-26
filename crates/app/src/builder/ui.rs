//! The builder's window (spec §4): a palette column (elements, tools), the canvas, and a footer
//! with the live formula and name. An `egui::Modal` like the File window, sized from the app's
//! window every frame; it closes on Close or Esc, never on a click outside (a stray click must not
//! end a drawing — the drawing is kept anyway).
//!
//! Canvas gestures (the default tool is Atom with carbon; the Atom tool and right-click revised by
//! the user's hand-test, 2026-10-20 — spec §4):
//! - Atom tool: click an atom to select it; with an atom selected, a click on empty space places a
//!   new atom of the palette's element bonded to it (a ghost bond and atom follow the cursor) and
//!   selects the new one, so clicks draw a chain — at the bond's real length relative to C–C and,
//!   once the selected atom has a bond to measure from, at its VSEPR angle on the cursor's side
//!   (109.5° at a saturated C, 120° by a double bond, 104.5° at O; Alt places freely; a spot on an
//!   existing atom bonds to it, closing a ring); with nothing selected a click places a lone atom.
//!   Click the selected atom again: it becomes the palette's element. Shift-click another atom: a
//!   bond to it (a ring closure). Drag an atom: it moves (the selection with it);
//! - right-click deselects; Esc deselects, and closes the window when nothing is selected (Esc in
//!   the search box only leaves the box);
//! - click a bond (Bond tool, or the Atom tool): its order cycles 1 → 2 → 3 → 1; Bond tool: drag
//!   from atom to atom for a bond;
//! - Charge tool: click an atom to cycle 0 → +1 → −1 → 0;
//! - Erase tool, Delete: removes the atom (and its bonds) or the bond under the cursor (Delete
//!   with nothing under it: the selection);
//! - Select/Move: drag an atom (the whole selection when it is in it), drag on empty space to
//!   box-select, Shift adds;
//! - wheel zooms about the cursor, middle- or right-drag pans, Ctrl-Z / Ctrl-Y (Ctrl-Shift-Z) undo /
//!   redo.
//!
//! The canvas's **2D | 3D** toggle (top-right; spec `docs/superpowers/specs/2026-11-26-builder-3d-view-design.md`)
//! swaps the canvas for a read-only ball-and-stick of the shape the hand-off would place
//! ([`super::view3d`]): drag turns it, the wheel zooms, a double-click resets it, and hovering a
//! bond or an atom reads out its length (Å) or bond angles. No tool acts there; the palette, the
//! search and the footer stay live. Display state only (`Builder::view_3d`), never an `Action`.
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
pub const RULE_1_LINE: &str = "bond lengths, valences and database shapes here only place the atoms; the solver decides the real geometry and bonds";

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
            if st.builder.view_3d {
                view3d_area(ui, canvas, &mut st.builder);
            } else {
                canvas_area(ui, canvas, &mut st.builder);
            }
            // after the canvas, so it sits on top of it (egui hit-tests the last widget first)
            view_toggle(ui, canvas, &mut st.builder);
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
    } else if st.builder.lookup.busy() || st.builder.search_waiting() {
        // the lookups' answers and timeouts arrive without input: keep the frames coming
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    } else if st.builder.lookup.awaiting() || st.builder.search_awaiting() {
        // a timed-out lookup's late answer relabels its fragment when it comes
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
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
    search_box(ui, b);
    ui.add_space(10.0);
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
        (Tool::Atom, "Atom", "click an atom to select it, then click nearby: a new atom bonded to it at the bond's real length and, once the selected atom has a bond, its real angle (109.5° at a saturated carbon, 120° by a double bond, 104.5° at O) on the side you click (Alt: exactly where you click), and so on along a chain — a click that lands on an existing atom bonds to it; with nothing selected a click places a lone atom; right-click deselects; click the selected atom again to change its element; Shift-click another atom to bond to it; drag an atom to move it"),
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
    ui.add_space(10.0);
    if ui.checkbox(&mut b.online, RichText::new(ONLINE_LABEL).size(12.5)).on_hover_text(ONLINE_TIP).changed() && !b.online {
        // off: a search in flight stops now, not on the next frame (Task 5 review fix 1)
        b.poll_search();
    }
}

/// The search box (spec §8, `search.rs`) and its results: the local ones (the shipped list and the
/// cache), then — with the online setting on and PubChem reachable — "Search PubChem for '…'",
/// whose answer is listed below it. A row stamps its molecule beside the drawing; Enter stamps the
/// first local result, or asks PubChem when there is none.
fn search_box(ui: &mut Ui, b: &mut Builder) {
    use super::search::{parse_query, Query};
    let r = ui.add(egui::TextEdit::singleline(&mut b.search_query).hint_text(SEARCH_HINT).char_limit(SEARCH_CHAR_LIMIT).desired_width(ui.available_width()));
    let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
    // Esc in the box only leaves the box (final review M1): egui drops the focus at the start of the
    // frame, so the window's Esc (deselect, then close) would see no text field and take it too
    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
    }
    b.poll_search();
    if parse_query(&b.search_query) == Query::Empty {
        return;
    }
    let local = b.search_hits().to_vec();
    let online_row = b.can_search_online();
    let mut pick = if enter { local.first().and_then(|h| h.record.clone()) } else { None };
    let mut ask = enter && local.is_empty() && online_row && !b.search_waiting();
    let small = |ui: &mut Ui, text: &str| {
        ui.label(RichText::new(text).size(11.5).color(theme::INK_FAINT));
    };
    // PubChem's answer, once it came: its new rows take the lowest-ranked local rows' places (8 in all)
    let answered: Option<Vec<super::search::Hit>> = if online_row { b.online_hits().map(<[_]>::to_vec) } else { None };
    let (shown, fresh) = match &answered {
        Some(hits) => super::search::combined_rows(&local, hits),
        None => (local.iter().collect(), Vec::new()),
    };
    egui::ScrollArea::vertical().id_salt("builder-search-results").max_height(240.0).auto_shrink([false, true]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        for h in shown.iter().chain(&fresh) {
            if result_row(ui, h).clicked() {
                pick = h.record.clone();
            }
        }
        if local.is_empty() && fresh.is_empty() {
            small(ui, NO_LOCAL);
        }
        if !online_row {
            return;
        }
        if let Some(hits) = &answered {
            if fresh.is_empty() {
                small(ui, if hits.is_empty() { "no results on PubChem" } else { "nothing more on PubChem" });
            }
        } else if b.search_waiting() {
            small(ui, "searching PubChem…");
        } else {
            if let Some(e) = b.online_failed() {
                small(ui, &failed_line(e));
            }
            let text = format!("Search PubChem for '{}'", b.search_query.trim());
            if ui.add(egui::Button::new(RichText::new(text).size(12.0)).wrap().min_size(Vec2::new(ui.available_width(), 0.0))).on_hover_text(ONLINE_TIP).clicked() {
                ask = true;
            }
        }
    });
    if ask {
        b.search_online();
    }
    if let Some(r) = pick {
        b.stamp_record(&r);
    }
}

/// One result row: the name, then the formula and where the record comes from.
fn result_row(ui: &mut Ui, h: &super::search::Hit) -> egui::Response {
    let formula = h.record.as_ref().map_or_else(|| h.formula.clone(), |r| crate::names::formula(&r.z, Some(r.charge.iter().map(|&q| q as i32).sum())));
    let mut job = egui::text::LayoutJob::default();
    job.append(&h.name, 0.0, egui::TextFormat { font_id: FontId::proportional(13.0), color: theme::INK, ..Default::default() });
    job.append(&format!("\n{formula} · {}", h.source), 0.0, egui::TextFormat { font_id: FontId::proportional(11.0), color: theme::INK_FAINT, ..Default::default() });
    let cid = h.cid().map_or(String::new(), |c| format!(" (PubChem CID {c})"));
    let r = ui.add(egui::Button::new(job).wrap().right_text("").min_size(Vec2::new(ui.available_width(), 0.0)).fill(theme::PANEL_2));
    r.on_hover_text(format!("{}{cid} — click to put it on the canvas beside the drawing (Ctrl-Z takes it back)", h.name))
}

/// The online search's failure, in one short line.
fn failed_line(e: super::online::NetError) -> String {
    use super::online::NetError;
    match e {
        NetError::Offline => "PubChem not reachable (offline)".into(),
        NetError::Timeout => "PubChem did not answer in time".into(),
        NetError::Status(s) => format!("PubChem answered {s}: try again later"),
        NetError::Malformed => "PubChem's answer was not readable".into(),
    }
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
    // only while the canvas itself is under the pointer: not through the 2D | 3D toggle on top of
    // it (review: Delete with the pointer on the toggle erased the atom beneath)
    let hovered = if b.gesture == Gesture::None && resp.hovered() { over.and_then(|p| hit(b, &view, p, None)) } else { None };

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

/// An existing atom this close (canvas units) to the placed spot is bonded to instead of stacking a
/// new atom on it (a ring closes).
const JOIN_WITHIN: f32 = 0.3;

/// Where an Atom-tool click at canvas point `to` puts its atom (user requests 2026-10-20: rough
/// real distances and angles, set only once there is a bond to measure from). Bonded to the one
/// selected atom: at the bond's real length relative to C–C ([`chem::drawn_length`](super::chem::drawn_length));
/// the direction follows the cursor while the selected atom has no bond yet, else the VSEPR angle
/// from one of its bonds on the cursor's side ([`chem::next_direction`](super::chem::next_direction)).
/// At the cursor when `free` (Alt) or when nothing is selected. Also the existing atom that spot
/// falls on, if any (bonded to instead).
fn placement(b: &Builder, to: [f32; 2], free: bool) -> ([f32; 2], Option<usize>) {
    let Some(a) = b.selected_one().filter(|_| !free) else { return (to, None) };
    let from = b.drawing.atoms[a].pos;
    let cursor = (to[1] - from[1]).atan2(to[0] - from[0]);
    let angle = super::chem::next_direction(&b.drawing, a, 1, Some(cursor));
    let len = super::chem::drawn_length(b.drawing.atoms[a].z, b.element, 1);
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
        p.text(rect.center(), Align2::CENTER_CENTER, "click to place an atom", FontId::proportional(15.0), theme::INK_FAINT);
    }
    // the basic controls, faded in the bottom-left corner (user request 2026-10-20): the tool's own
    // lines first, then the ones every tool shares
    let mut y = rect.bottom() - 10.0;
    for line in controls(b.tool).iter().rev() {
        p.text(egui::pos2(rect.left() + 12.0, y), Align2::LEFT_BOTTOM, *line, FontId::proportional(11.5), theme::INK_FAINT.gamma_multiply(0.75));
        y -= 15.0;
    }
}

/// The corner cheat sheet: the tool's controls, then the shared ones.
fn controls(tool: Tool) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = match tool {
        Tool::Atom => vec![
            "click: place an atom · click an atom: select it",
            "selected + click nearby: a bonded atom at its real length and angle",
            "click the selected atom: change its element · Shift-click an atom: bond to it",
            "drag an atom: move · Alt-click: place exactly · click a bond: single → double → triple",
        ],
        Tool::Bond => vec!["click a bond: single → double → triple", "drag from an atom to an atom: a bond"],
        Tool::Charge => vec!["click an atom: charge 0 → +1 → −1"],
        Tool::Erase => vec!["click an atom or a bond: remove it"],
        Tool::Select => vec!["click: select · Shift-click: add or remove", "drag an atom: move the selection · drag on empty space: box select"],
    };
    v.extend(["right-click: deselect · Delete: remove · Ctrl-Z / Ctrl-Y: undo / redo", "wheel: zoom · right- or middle-drag: pan · Esc: deselect, then close"]);
    v
}

/// The canvas's 2D | 3D toggle (3D view spec §1), at the canvas's top-right corner: display state
/// on the builder, not an `Action`, not recorded, not in the profile.
fn view_toggle(ui: &mut Ui, canvas: Rect, b: &mut Builder) {
    let r = Rect::from_min_size(egui::pos2(canvas.right() - 94.0, canvas.top() + 8.0), Vec2::new(86.0, 24.0));
    child(ui, r, egui::Layout::left_to_right(egui::Align::Center), |ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        for (three, name, tip) in [(false, "2D", TIP_2D), (true, "3D", TIP_3D)] {
            let on = b.view_3d == three;
            let btn = egui::Button::new(RichText::new(name).size(12.5).color(if on { Color32::WHITE } else { theme::INK_DIM }))
                .min_size(Vec2::new(43.0, 24.0))
                .fill(if on { theme::ACCENT_BG } else { theme::PANEL_2 })
                .stroke(Stroke::new(1.0, if on { theme::ACCENT } else { theme::LINE }));
            if ui.add(btn).on_hover_text(tip).clicked() && !on {
                b.view_3d = three;
                b.gesture = Gesture::None;
            }
        }
    });
}

/// The toggle's tooltips.
const TIP_2D: &str = "the sketch: draw and edit the structural formula";
const TIP_3D: &str = "the shape Insert / Replace would place, as a ball-and-stick: drag to turn it, hover a bond or an atom for lengths and angles (read-only: switch back to 2D to edit)";
/// The 3D view's one-line controls hint (spec §4), and its legend for a guessed shape (spec §6).
pub const HINT_3D: &str = "drag: turn · wheel: zoom · double-click: reset the view · hover a bond: its length · hover an atom: its angles · read-only (2D to edit)";
pub const APPROXIMATE_3D: &str = "dashed outline: an approximate shape (not from the database)";

/// The 3D view in place of the canvas (3D view spec §2–§6): the shape [`Builder::shaped`] gives
/// (what Replace scene would place), turned by a drag, zoomed by the wheel, reset by a
/// double-click, with the hovered bond's length or atom's angles beside the pointer. Read-only: no
/// tool acts here (Ctrl-Z / Ctrl-Y still undo and redo, as the palette's buttons do).
fn view3d_area(ui: &mut Ui, rect: Rect, b: &mut Builder) {
    let resp = ui.interact(rect, Id::new("builder-view3d"), Sense::click_and_drag());
    // the keys first, so the shape below is the drawing's as this frame leaves it
    if !ui.ctx().egui_wants_keyboard_input() {
        let (redo, undo) = ui.input_mut(|i| {
            let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z) || i.consume_key(egui::Modifiers::COMMAND, egui::Key::Y);
            (redo, i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z))
        });
        if redo {
            b.redo();
        }
        if undo {
            b.undo();
        }
    }
    let shaped = b.shaped().clone();
    b.view3d.follow(&b.drawing, &shaped);
    if resp.double_clicked() {
        b.view3d.reset(&shaped.positions);
    } else if resp.dragged_by(PointerButton::Primary) {
        b.view3d.turn(resp.drag_delta());
    }
    if resp.hovered() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let f = (scroll / 300.0).exp() * pinch;
        if (f - 1.0).abs() > 1e-4 {
            b.view3d.zoom_by(f as f64);
        }
    }
    let z: Vec<u8> = b.drawing.atoms.iter().map(|a| a.z).collect();
    let proj = b.view3d.project(&shaped.positions, &z, rect);
    b.view3d.shown = proj.len();
    let scale = b.view3d.scale(&shaped.positions, rect);
    let pointer = ui.input(|i| i.pointer.latest_pos()).filter(|p| rect.contains(*p));
    let hovered = pointer.filter(|_| resp.hovered() && !resp.dragged()).and_then(|p| super::view3d::View3d::hit(&proj, &b.drawing.bonds, p));
    paint3d(ui, rect, &b.drawing, &b.selection, &shaped, &proj, scale, hovered, pointer);
    if resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if resp.hovered() && !b.drawing.atoms.is_empty() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
}

/// Paints the 3D view: back to front ([`super::view3d::draw_order`]), farther parts dimmer;
/// bonds as single, double or triple sticks between the balls' rims; element letters and formal
/// charges as on the canvas; a dashed outline on a guessed (approximate) fragment's atoms; the
/// hovered part's readout in a box by the pointer; the controls hint along the bottom.
#[allow(clippy::too_many_arguments)]
fn paint3d(ui: &Ui, rect: Rect, d: &super::Drawing, selection: &std::collections::BTreeSet<usize>, s: &super::shape::Shaped, proj: &[(Pos2, f64, f32)], scale: f64, hovered: Option<super::view3d::Part>, pointer: Option<Pos2>) {
    use super::view3d::{self as v3, Part};
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 8.0, theme::APP);
    p.rect_stroke(rect, 8.0, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
    let approx = v3::approximate_atoms(d.atoms.len(), s);
    let mut hints = vec![HINT_3D];
    if approx.iter().any(|&a| a) {
        hints.push(APPROXIMATE_3D);
    }
    let mut y = rect.bottom() - 10.0;
    for line in hints.iter().rev() {
        p.text(egui::pos2(rect.left() + 12.0, y), Align2::LEFT_BOTTOM, *line, FontId::proportional(11.5), theme::INK_FAINT.gamma_multiply(0.75));
        y -= 15.0;
    }
    if d.atoms.is_empty() || proj.len() != d.atoms.len() {
        p.text(rect.center(), Align2::CENTER_CENTER, "nothing drawn yet: switch to 2D to draw, or pick a molecule from the search", FontId::proportional(15.0), theme::INK_FAINT);
        return;
    }
    let (lo, hi) = proj.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| (lo.min(q.1), hi.max(q.1)));
    // 0 at the front, 1 at the back
    let far = |depth: f64| if hi - lo > 1e-9 { ((depth - lo) / (hi - lo)) as f32 } else { 0.0 };
    for part in v3::draw_order(proj, &d.bonds) {
        match part {
            Part::Bond(k) => {
                let bd = d.bonds[k];
                let ((pa, da, ra), (pb, db, rb)) = (proj[bd.a], proj[bd.b]);
                // the bond in 3D (points): from one ball's surface to the other's
                let v = pb - pa;
                let dz = ((db - da) * scale) as f32;
                let len = (v.length_sq() + dz * dz).sqrt();
                if !len.is_finite() || len <= ra + rb {
                    continue;
                }
                let u = v / len;
                let (s0, s1) = (pa + u * ra, pb - u * rb);
                let t = far(0.5 * (da + db));
                let w = (0.24 * ra.min(rb)).clamp(1.2, 5.0);
                let colour = if hovered == Some(Part::Bond(k)) { theme::ACCENT } else { blend(theme::APP, theme::INK_DIM, 1.0 - 0.6 * t) };
                let n = if v.length() > 1.0 { v.normalized().rot90() } else { Vec2::ZERO };
                let (width, offsets): (f32, &[f32]) = match bd.order {
                    2 => (0.7 * w, &[-1.1, 1.1]),
                    3 => (0.6 * w, &[-1.9, 0.0, 1.9]),
                    _ => (w, &[0.0]),
                };
                for &o in offsets {
                    p.line_segment([s0 + n * o * w, s1 + n * o * w], Stroke::new(width, colour));
                }
            }
            Part::Atom(i) => {
                let (c, depth, r) = proj[i];
                let t = far(depth);
                let a = d.atoms[i];
                let [cr, cg, cb] = render::elements::colour_srgb(a.z);
                let col = Color32::from_rgb(cr, cg, cb);
                let fill = blend(theme::APP, col, 0.92 - 0.5 * t);
                p.circle_filled(c, r, fill);
                p.circle_stroke(c, r, Stroke::new(1.0, blend(theme::APP, fill, 0.55)));
                // a soft highlight up and to the left: a ball, not a disc
                p.circle_filled(c + Vec2::new(-0.35 * r, -0.35 * r), 0.3 * r, Color32::from_white_alpha((60.0 * (1.0 - 0.6 * t)) as u8));
                if approx[i] && r >= 3.0 {
                    let ring: Vec<Pos2> = (0..=32).map(|k| c + Vec2::angled(k as f32 / 32.0 * std::f32::consts::TAU) * (r + 2.5)).collect();
                    p.extend(egui::Shape::dashed_line(&ring, Stroke::new(1.2, theme::INK_DIM.gamma_multiply(0.9)), 3.0, 3.0));
                }
                // the 2D selection shows here too (Esc, Delete in 2D and the groups act on it)
                if selection.contains(&i) {
                    p.circle_stroke(c, r + 3.5, Stroke::new(2.0, theme::ACCENT));
                } else if hovered == Some(Part::Atom(i)) {
                    p.circle_stroke(c, r + 3.0, Stroke::new(1.5, theme::INK_DIM));
                }
                if r >= 7.0 {
                    let fs = (0.95 * r).clamp(8.0, 18.0);
                    let light = 0.299 * fill.r() as f32 + 0.587 * fill.g() as f32 + 0.114 * fill.b() as f32 > 150.0;
                    p.text(c, Align2::CENTER_CENTER, symbol(a.z), FontId::proportional(fs), if light { Color32::from_gray(25) } else { Color32::WHITE });
                    if a.charge != 0 {
                        let at = c + Vec2::new(r * 0.85, -r * 0.85);
                        p.circle_filled(at, fs * 0.48, theme::PANEL);
                        p.circle_stroke(at, fs * 0.48, Stroke::new(1.0, theme::INK_DIM));
                        p.text(at, Align2::CENTER_CENTER, if a.charge > 0 { "+" } else { "−" }, FontId::proportional(fs * 0.8), theme::INK);
                    }
                }
            }
        }
    }
    // the readout (spec §4): a bond's length in Å, an atom's element and bond angles
    let text = match hovered {
        Some(Part::Bond(k)) => v3::bond_readout(d, &s.positions, k),
        Some(Part::Atom(i)) => v3::atom_readout(d, &s.positions, i),
        None => None,
    };
    if let (Some(text), Some(at)) = (text, pointer) {
        let galley = p.layout_no_wrap(text, FontId::proportional(12.5), theme::INK);
        let size = galley.size() + Vec2::splat(16.0);
        let mut min = at + Vec2::new(16.0, 16.0);
        if min.x + size.x > rect.right() - 4.0 {
            min.x = at.x - 16.0 - size.x;
        }
        if min.y + size.y > rect.bottom() - 4.0 {
            min.y = at.y - 16.0 - size.y;
        }
        // flipped, it may now cross the other edge: keep it inside the view (its top-left first)
        min.x = min.x.min(rect.right() - 4.0 - size.x).max(rect.left() + 4.0);
        min.y = min.y.min(rect.bottom() - 4.0 - size.y).max(rect.top() + 4.0);
        let r = Rect::from_min_size(min, size);
        p.rect_filled(r, 6.0, theme::PANEL);
        p.rect_stroke(r, 6.0, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
        p.galley(min + Vec2::splat(8.0), galley, theme::INK);
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
    // where the hand-off's shape comes from (spec §8): one line, each fragment's in the tooltip
    if !b.drawing.atoms.is_empty() {
        let s = b.shaped().clone();
        let line = super::shape::footer_label(&s);
        if !line.is_empty() {
            let approximate = s.sources.iter().any(|(_, f)| matches!(f, super::shape::FragmentSource::Approximate(_)));
            let r = ui.label(RichText::new(line).size(12.5).color(if approximate { theme::INK_DIM } else { theme::ACCENT_INK }));
            if s.sources.len() > 1 {
                let tip = s
                    .sources
                    .iter()
                    .map(|(frag, f)| {
                        let z: Vec<u8> = frag.iter().map(|&i| b.drawing.atoms[i].z).collect();
                        let q: i32 = frag.iter().map(|&i| b.drawing.atoms[i].charge as i32).sum();
                        format!("{}: {}", crate::names::formula(&z, Some(q)), super::shape::fragment_label(f))
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                r.on_hover_text(tip);
            }
        }
        // the fallback's hint (spec §6.2): a drawing without its hydrogens is a radical
        if let Some(hint) = b.fill_hint() {
            ui.label(RichText::new(hint).size(11.5).color(theme::ACCENT_INK))
                .on_hover_text("drawn atoms are real atoms: without its hydrogens the drawing is a radical, which no closed-shell record matches — Fill hydrogens adds them");
        }
    }
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

/// What a hand-off puts in (spec §6, §8): the shape Replace sends, Insert's copy of it beside the
/// scene's atoms, and the reaches the fit check and the enlarge offer use — all from the shape the
/// hand-off sends, the database's where it knows a fragment.
struct HandoffGeometry {
    new: Vec<[f64; 3]>,
    beside: Vec<[f64; 3]>,
    reach_replace: f64,
    reach_insert: f64,
}

/// [`geometry_of`] the shape `src` gives (tests).
#[cfg(test)]
fn handoff_geometry(b: &mut Builder, src: &dyn super::shape::RecordSource, existing: &[[f64; 3]]) -> HandoffGeometry {
    geometry_of(b.shaped_from(src).positions.clone(), existing)
}

/// The hand-off's geometry from the shape it sends, `new`.
fn geometry_of(new: Vec<[f64; 3]>, existing: &[[f64; 3]]) -> HandoffGeometry {
    use super::{embed, max_abs};
    let mut beside = new.clone();
    embed::place_beside(existing, &mut beside);
    HandoffGeometry { reach_replace: max_abs(&new), reach_insert: max_abs(&beside).max(max_abs(existing)), new, beside }
}

/// The hand-off row: the fit check and the enlarge offer, the warnings' confirm line, Relax
/// after, Insert and Replace. Returns true when a hand-off was sent (the window closes).
fn handoff_rows(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> bool {
    use super::{enlarge_n, fit_limit};
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
    let existing: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
    let shaped = b.shaped();
    let looking = super::shape::looking_up(shaped);
    let HandoffGeometry { new, beside, reach_replace, reach_insert } = geometry_of(shaped.positions.clone(), &existing);
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
                let (can, label) = handoff_button(replace, label, !matches!(fit, Fit::No { .. }), looking);
                let r = ui.add_enabled(can, egui::Button::new(RichText::new(label).strong()).min_size(Vec2::new(130.0, 30.0)));
                let r = if can {
                    r.on_hover_text(tip)
                } else if looking {
                    r.on_disabled_hover_text("waiting for PubChem's shape (at most 5 s); the drawing stays editable")
                } else {
                    r.on_disabled_hover_text("does not fit the box: enlarge it first")
                };
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

/// Insert's or Replace's button (spec §7): pressable when the shape fits the box and no fragment is
/// still being looked up online; while one is, it reads "Insert (looking up…)" / "Replace (looking
/// up…)" until the answer comes or the lookup times out. `label` is the button's usual text.
fn handoff_button(replace: bool, label: &'static str, fits: bool, looking: bool) -> (bool, &'static str) {
    let text = match (looking, replace) {
        (true, true) => "Replace (looking up…)",
        (true, false) => "Insert (looking up…)",
        (false, _) => label,
    };
    (fits && !looking, text)
}

/// The online setting's label and tooltip (spec §7).
pub const ONLINE_LABEL: &str = "Look up molecules online (PubChem)";
pub const ONLINE_TIP: &str = "sends the molecule's name or structure string to NCBI's PubChem; off: no network";

/// The search box's hint (spec §8) and its "nothing here" line.
pub const SEARCH_HINT: &str = "molecule name, formula or CID";
pub const NO_LOCAL: &str = "no match in the shipped list or the cache";
/// The search box's length limit (characters): a few times what the online search sends.
const SEARCH_CHAR_LIMIT: usize = 4 * super::search::MAX_NAME;

/// The enlarge offer's margin, in words.
const ENLARGE_MARGIN_TEXT: &str = "4 Bohr to spare from the farthest atom to the box";

#[cfg(test)]
mod tests {
    use super::*;

    /// Review focus 4: the fit check's reach is the shape the hand-off sends, not the embedding's.
    #[test]
    fn the_fit_check_uses_the_database_shape() {
        use super::super::shape::{shape, tests::draw_record, Lookup, RecordSource};
        use super::super::{embed, fit_limit, iso, max_abs, records::Library};
        // a source answering ethanol's record stretched ×3
        struct Stretched(super::super::records::Record);
        impl RecordSource for Stretched {
            fn find(&self, d: &super::super::Drawing, frag: &[usize]) -> Lookup {
                match iso::map_fragment(d, frag, &self.0) {
                    iso::MapResult::Mapped(m) => Lookup::Found(self.0.clone(), m),
                    _ => Lookup::NotFound,
                }
            }
        }
        let mut eth = Library::shipped().records().iter().find(|r| r.name == "ethanol").unwrap().clone();
        for p in &mut eth.xyz {
            *p = p.map(|c| 3.0 * c);
        }
        let src = Stretched(eth.clone());
        let mut b = Builder::default();
        draw_record(&mut b.drawing, &eth, [0.0, 0.0], &(0..eth.z.len()).collect::<Vec<_>>());
        let shaped = shape(&b.drawing, &src).positions;
        let embedded = embed::embed(&b.drawing);
        let existing = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let g = handoff_geometry(&mut b, &src, &existing);
        assert_eq!(g.new, shaped, "Replace sends the database's shape");
        assert_eq!(g.reach_replace, max_abs(&shaped));
        assert!(max_abs(&embedded) * 2.0 < g.reach_replace, "the stretched shape reaches further: {} vs {}", g.reach_replace, max_abs(&embedded));
        // Insert's copy is the same shape moved beside the scene's atoms
        let mut beside = shaped.clone();
        embed::place_beside(&existing, &mut beside);
        assert_eq!(g.beside, beside);
        assert_eq!(g.reach_insert, max_abs(&beside).max(max_abs(&existing)));
        // so an 81³ × 0.15 box that holds the embedding does not hold it
        let lim = fit_limit(81, 0.15, None);
        assert!(max_abs(&embedded) <= lim && g.reach_replace > lim, "{} / {} / {lim}", max_abs(&embedded), g.reach_replace);
    }

    /// Spec §7: while a fragment is being looked up, Insert and Replace wait, saying so; the lookup
    /// timing out (the clock moved by hand past [`super::super::online::TIMEOUT`]) enables them with
    /// the offline label.
    #[test]
    fn insert_waits_while_looking_up() {
        use super::super::online::{tests as net, Cache, Lookups, DEBOUNCE, TIMEOUT};
        use super::super::shape::{footer_label, looking_up};
        use std::sync::{Arc, Condvar, Mutex};
        use std::time::Duration;
        let clock = net::ManualClock::new();
        // PubChem never answers: the gate stays shut until the end of the test
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let canned = net::Canned::new(net::ethane_answers("CC"));
        let lookups = Lookups::with_clock(Box::new(net::Gate { inner: canned.clone(), open: gate.clone() }), Cache::none(), clock.clone());
        let mut b = Builder { lookup: net::ethane_session(lookups, Cache::none(), clock.clone()), drawing: net::ethane(false), ..Default::default() };
        let buttons = |s: &super::super::shape::Shaped| [handoff_button(false, "Insert into scene", true, looking_up(s)), handoff_button(true, "Replace scene", true, looking_up(s))];
        // waiting out the debounce, then asked and waiting for the answer
        for step in [Duration::ZERO, DEBOUNCE, TIMEOUT - Duration::from_millis(1)] {
            clock.advance(step);
            let s = b.shaped().clone();
            assert_eq!(footer_label(&s), "looking up online…");
            assert_eq!(buttons(&s), [(false, "Insert (looking up…)"), (false, "Replace (looking up…)")]);
            assert!(b.lookup.busy(), "the window keeps repainting");
        }
        // five seconds after it was asked: offline, and the buttons are back
        clock.advance(Duration::from_millis(1));
        let s = b.shaped().clone();
        assert_eq!(footer_label(&s), "shape: approximate (offline)");
        assert_eq!(buttons(&s), [(true, "Insert into scene"), (true, "Replace scene")]);
        assert!(!b.lookup.busy());
        // a shape that does not fit still disables them, with their usual text
        assert_eq!(handoff_button(false, "Insert into scene", false, false), (false, "Insert into scene"));
        net::open_gate(&gate);
    }

    /// Every text the palette column paints, after two frames in a 150 × 900 column.
    fn palette_texts(st: &mut UiState) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => {
                    *out += t.galley.text();
                    out.push('\n');
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let ctx = egui::Context::default();
        let mut out = String::new();
        for _ in 0..2 {
            out.clear();
            let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(150.0, 900.0))), ..Default::default() };
            let full = ctx.run_ui(raw, |ui| palette_column(ui, st));
            full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        }
        out
    }

    /// Spec §8: the search box tops the palette column with its hint; each result row shows the
    /// name, the formula and the source; the PubChem row appears only with the setting on and
    /// PubChem reachable (never in a test build's own session), above the online setting.
    #[test]
    fn the_search_box_lists_results_and_the_pubchem_row() {
        use super::super::online::{tests as net, Cache, Lookups, Session};
        let mut st = UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap());
        let t = palette_texts(&mut st);
        assert!(t.contains(SEARCH_HINT), "{t}");
        let at = |s: &str| t.to_lowercase().find(&s.to_lowercase()).unwrap_or_else(|| panic!("{s} missing: {t}"));
        assert!(at(SEARCH_HINT) < at("element") && at("element") < at(ONLINE_LABEL), "the box first, the setting last: {t}");
        // the search tests' own list, whatever the shipped one holds (review M8)
        st.builder.search.library = Some(super::super::search::tests::test_library());
        st.builder.search_query = "eth".into();
        let t = palette_texts(&mut st);
        assert!(t.contains("ethanol\nC₂H₆O · shipped"), "{t}");
        assert!(!t.contains("Search PubChem"), "no network in a test build: {t}");
        st.builder.search_query = "xyzzy".into();
        assert!(palette_texts(&mut st).contains(NO_LOCAL));
        // a session whose PubChem (canned) is reachable: the row, unless the setting is off
        let clock = net::ManualClock::new();
        st.builder.lookup = Session::with(Lookups::with_clock(Box::new(net::Canned::new(Default::default())), Cache::none(), clock.clone()), Cache::none(), clock);
        assert!(palette_texts(&mut st).contains("Search PubChem for 'xyzzy'"));
        // asked: PubChem knows no such name (its 404) — one short line, the row gone
        st.builder.search_online();
        assert!(st.builder.lookup.wait_search(std::time::Duration::from_secs(30)));
        let t = palette_texts(&mut st);
        assert!(t.contains("no results on PubChem") && !t.contains("Search PubChem"), "{t}");
        // a name it knows: its row, with the source
        let mut answers = net::ethane_answers("CC");
        answers.insert(super::super::online::name_url("ethane"), Ok("6324\n".into()));
        let clock = net::ManualClock::new();
        st.builder.lookup = Session::with(Lookups::with_clock(Box::new(net::Canned::new(answers)), Cache::none(), clock.clone()), Cache::none(), clock);
        st.builder.search_query = "ethane".into();
        st.builder.search_online();
        assert!(st.builder.lookup.wait_search(std::time::Duration::from_secs(30)));
        assert!(palette_texts(&mut st).contains("Ethane\nC₂H₆ · PubChem"));
        st.builder.online = false;
        assert!(!palette_texts(&mut st).contains("Search PubChem"));
    }

    /// Every text the footer rows paint, after two frames in a 900 × 300 area.
    fn footer_texts(st: &mut UiState) -> String {
        fn walk(s: &egui::epaint::Shape, out: &mut String) {
            match s {
                egui::epaint::Shape::Text(t) => {
                    *out += t.galley.text();
                    out.push('\n');
                }
                egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let ctx = egui::Context::default();
        let mut out = String::new();
        for _ in 0..2 {
            out.clear();
            let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 300.0))), ..Default::default() };
            let full = ctx.run_ui(raw, |ui| footer_rows(ui, st));
            full.shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        }
        out
    }

    /// Final review I2 (spec §6.2): a drawing without its hydrogens is a radical and matches no
    /// closed-shell record; the fallback label stays as the spec words it, and a second line says
    /// Fill hydrogens would match the record the filled drawing maps to. Once filled, the database's
    /// shape and no hint.
    #[test]
    fn the_fallback_suggests_fill_hydrogens() {
        let mut st = UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap());
        // C–C–O, no hydrogens (a test build has no network: offline)
        let d = &mut st.builder.drawing;
        for (i, z) in [6u8, 6, 8].into_iter().enumerate() {
            d.add_atom(z, [i as f32, 0.0]);
        }
        d.bond(0, 1);
        d.bond(1, 2);
        let t = footer_texts(&mut st);
        assert!(t.contains("shape: approximate (offline)\n"), "{t}");
        assert!(t.contains("Fill hydrogens would match ethanol\n"), "{t}");
        st.builder.fill_hydrogens();
        let t = footer_texts(&mut st);
        assert!(t.contains("shape: PubChem \"ethanol\" (CID 702)") && !t.contains("would match"), "{t}");
        // a drawing whose filled form is in no database either: no hint
        let mut he = super::super::Drawing::default();
        he.add_atom(2, [0.0, 0.0]);
        he.add_atom(2, [1.0, 0.0]);
        he.bond(0, 1);
        st.builder.drawing = he;
        let t = footer_texts(&mut st);
        assert!(t.contains("shape: approximate (offline)") && !t.contains("would match"), "{t}");
    }

    /// One frame of the palette column with `events`; whether an Esc was still there for the
    /// builder's own Esc (deselect, then close) after it.
    fn palette_frame(ctx: &egui::Context, st: &mut UiState, events: Vec<egui::Event>) -> bool {
        let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(150.0, 900.0))), events, ..Default::default() };
        let mut esc_left = false;
        let _ = ctx.run_ui(raw, |ui| {
            palette_column(ui, st);
            esc_left = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        });
        esc_left
    }

    /// Final review M1: Esc in the search box only leaves the box (egui drops the focus at the start
    /// of the frame, so the window's Esc would otherwise see no text field and close the builder or
    /// clear the selection); with the box not focused, Esc is the builder's.
    #[test]
    fn esc_in_the_search_box_stays_in_the_box() {
        let mut st = UiState::new("h2", &crate::scene::Scene::preset("h2").unwrap());
        let ctx = egui::Context::default();
        let esc = egui::Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE };
        let press = |pressed: bool| egui::Event::PointerButton { pos: Pos2::new(40.0, 10.0), button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        palette_frame(&ctx, &mut st, vec![]);
        assert!(palette_frame(&ctx, &mut st, vec![esc.clone()]), "the box not focused: the builder's Esc");
        palette_frame(&ctx, &mut st, vec![]);
        palette_frame(&ctx, &mut st, vec![egui::Event::PointerMoved(Pos2::new(40.0, 10.0)), press(true)]);
        palette_frame(&ctx, &mut st, vec![press(false)]);
        assert!(ctx.text_edit_focused(), "the click put the search box in focus");
        assert!(!palette_frame(&ctx, &mut st, vec![esc.clone()]), "Esc in the box is the box's");
        assert!(!ctx.text_edit_focused(), "and the box let go of the focus");
        palette_frame(&ctx, &mut st, vec![]);
        palette_frame(&ctx, &mut st, vec![]);
        assert!(palette_frame(&ctx, &mut st, vec![esc]), "the next Esc is the builder's again");
    }

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

    /// The drawing's geometry (user requests 2026-10-20): the first bond from a lone atom follows the
    /// cursor (nothing to measure from) at the real length; the next takes the VSEPR angle from it on
    /// the cursor's side; a bond to H is shorter; Alt places at the cursor; a placed spot on an
    /// existing atom bonds to it.
    #[test]
    fn bonded_atoms_take_real_lengths_and_vsepr_angles() {
        let mut b = Builder::default();
        let view = View { centre: Pos2::ZERO, pan: [0.0, 0.0], scale: UNIT };
        let at = |x: f32, y: f32| Pos2::new(x * UNIT, y * UNIT);
        let dist = |p: [f32; 2], q: [f32; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
        let angle = |d: &super::super::Drawing, a: usize, c: usize, e: usize| {
            let (pa, pc, pe) = (d.atoms[a].pos, d.atoms[c].pos, d.atoms[e].pos);
            let (u, v) = ([pa[0] - pc[0], pa[1] - pc[1]], [pe[0] - pc[0], pe[1] - pc[1]]);
            ((u[0] * v[0] + u[1] * v[1]) / (dist(pa, pc) * dist(pe, pc))).acos().to_degrees()
        };
        click(&mut b, &view, at(0.0, 0.0), false, false);
        // the first bond: exactly the cursor's direction (37°), one C–C length
        let c37 = [37f32.to_radians().cos(), 37f32.to_radians().sin()];
        click(&mut b, &view, at(1.7 * c37[0], 1.7 * c37[1]), false, false);
        assert!(dist(b.drawing.atoms[1].pos, c37) < 1e-4, "{:?}", b.drawing.atoms[1].pos);
        // the second: 109.47° at the saturated carbon, on the cursor's side
        let p1 = b.drawing.atoms[1].pos;
        click(&mut b, &view, at(p1[0] + 1.0, p1[1] - 0.3), false, false);
        assert!((angle(&b.drawing, 0, 1, 2) - 109.47).abs() < 0.05, "{}", angle(&b.drawing, 0, 1, 2));
        assert!(b.drawing.atoms[2].pos[0] > p1[0], "the cursor's side");
        // a hydrogen from it: the C–H length, ≈ 0.70 of C–C
        b.element = 1;
        let p2 = b.drawing.atoms[2].pos;
        click(&mut b, &view, at(p2[0], p2[1] + 2.0), false, false);
        let ch = dist(b.drawing.atoms[3].pos, p2);
        assert!((ch - super::super::chem::drawn_length(6, 1, 1)).abs() < 1e-4 && (ch - 0.704).abs() < 0.01, "C–H {ch}");
        // Alt: exactly where clicked
        b.selection = [2].into();
        click(&mut b, &view, at(3.3, 2.2), false, true);
        assert!(dist(b.drawing.atoms[4].pos, [3.3, 2.2]) < 1e-4);
        // water by clicks: O, then two H — 104.5°
        let mut w = Builder { element: 8, ..Default::default() };
        click(&mut w, &view, at(0.0, 0.0), false, false);
        w.element = 1;
        click(&mut w, &view, at(1.0, 0.0), false, false);
        w.selection = [0].into();
        click(&mut w, &view, at(-0.5, 1.0), false, false);
        assert!((angle(&w.drawing, 1, 0, 2) - 104.5).abs() < 0.05, "H–O–H {}", angle(&w.drawing, 1, 0, 2));
        // a spot on an existing atom bonds to it instead of stacking a new one
        let mut j = Builder::default();
        click(&mut j, &view, at(0.0, 0.0), false, false);
        click(&mut j, &view, at(1.0, 0.0), false, false);
        let spot = placement(&j, [2.0, 1.0], false).0;
        j.drawing.add_atom(6, spot);
        click(&mut j, &view, at(2.0, 1.0), false, false);
        assert_eq!((j.drawing.atoms.len(), j.drawing.bonds.len(), j.selected_one()), (3, 2, Some(2)));
    }

    // ── the 3D view (builder 3D view spec §5, Task 1 step 4): real egui frames of the whole window ──

    /// The builder window's frames (1400 × 900), 1/60 s apart.
    struct Win {
        ctx: egui::Context,
        t: f64,
    }

    impl Win {
        fn new() -> Win {
            Win { ctx: egui::Context::default(), t: 0.0 }
        }

        /// One frame with `events`: the actions sent and every text painted, with its rect.
        fn frame(&mut self, st: &mut UiState, inp: &PanelInput<'_>, events: Vec<egui::Event>) -> (Vec<Action>, Vec<(String, Rect)>) {
            fn walk(s: &egui::epaint::Shape, out: &mut Vec<(String, Rect)>) {
                match s {
                    egui::epaint::Shape::Text(t) => out.push((t.galley.text().to_string(), t.visual_bounding_rect())),
                    egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                    _ => {}
                }
            }
            self.t += 1.0 / 60.0;
            let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1400.0, 900.0))), events, time: Some(self.t), ..Default::default() };
            let mut acts = Vec::new();
            let full = self.ctx.run_ui(raw, |root| show(root.ctx(), st, inp, &mut acts));
            let mut texts = Vec::new();
            full.shapes.iter().for_each(|c| walk(&c.shape, &mut texts));
            (acts, texts)
        }

        /// A few frames with no input (the modal lays itself out).
        fn settle(&mut self, st: &mut UiState, inp: &PanelInput<'_>) -> Vec<(String, Rect)> {
            let mut texts = Vec::new();
            for _ in 0..4 {
                texts = self.frame(st, inp, vec![]).1;
            }
            texts
        }

        /// A primary click at `p` (move, press, release, a quiet frame); the actions it sent.
        fn click(&mut self, st: &mut UiState, inp: &PanelInput<'_>, p: Pos2) -> Vec<Action> {
            let button = |pressed| egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            let mut acts = self.frame(st, inp, vec![egui::Event::PointerMoved(p)]).0;
            acts.extend(self.frame(st, inp, vec![button(true)]).0);
            acts.extend(self.frame(st, inp, vec![button(false)]).0);
            acts.extend(self.frame(st, inp, vec![]).0);
            acts
        }

        /// A primary drag from `a` to `b` in ten steps.
        fn drag(&mut self, st: &mut UiState, inp: &PanelInput<'_>, a: Pos2, b: Pos2) {
            let button = |pos, pressed| egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            self.frame(st, inp, vec![egui::Event::PointerMoved(a)]);
            self.frame(st, inp, vec![button(a, true)]);
            for k in 1..=10 {
                self.frame(st, inp, vec![egui::Event::PointerMoved(a + (b - a) * (k as f32 / 10.0))]);
            }
            self.frame(st, inp, vec![button(b, false)]);
            self.frame(st, inp, vec![]);
        }

        /// The canvas's rect (the 2D canvas's or the 3D view's) as last laid out.
        fn canvas(&self) -> Rect {
            self.ctx.read_response(Id::new("builder-view3d")).or_else(|| self.ctx.read_response(Id::new("builder-canvas"))).expect("the canvas is laid out").rect
        }
    }

    /// The centre of the painted text that is exactly `text`.
    fn text_at(texts: &[(String, Rect)], text: &str) -> Pos2 {
        texts.iter().find(|(t, _)| t == text).unwrap_or_else(|| panic!("no {text:?} in {:?}", texts.iter().map(|t| &t.0).collect::<Vec<_>>())).1.center()
    }

    /// A scene whose box holds the test molecules beside its H₂ (121³ × 0.25 Bohr).
    fn roomy() -> crate::scene::Scene {
        crate::scene::Scene { grid_n: 121, ..crate::scene::Scene::preset("h2").unwrap() }
    }

    fn water_drawing() -> super::super::Drawing {
        let mut d = super::super::Drawing::default();
        super::super::shape::tests::draw_record(&mut d, super::super::shape::tests::rec("water"), [0.0, 0.0], &[0, 1, 2]);
        d
    }

    /// He–He: in no database, so its shape is the embedding's guess (and it is over He's valence:
    /// the first press only arms the hand-off).
    fn helium_pair() -> super::super::Drawing {
        let mut d = super::super::Drawing::default();
        d.add_atom(2, [0.0, 0.0]);
        d.add_atom(2, [1.0, 0.0]);
        d.bond(0, 1);
        d
    }

    /// Spec §5: the toggle changes nothing handed off — Insert in 2D and in 3D (after turning the
    /// view) sends the same `Handoff`, for a matched record and for an approximate fragment.
    #[test]
    fn the_3d_view_changes_nothing_handed_off() {
        let scene = roomy();
        let cam = crate::ui::tests::camera();
        let inp = crate::ui::tests::input(&scene, None, &cam);
        for (name, drawing) in [("water", water_drawing()), ("He–He", helium_pair())] {
            let mut sent = Vec::new();
            for three in [false, true] {
                let mut st = UiState::new("h2", &scene);
                st.builder.open = true;
                st.builder.drawing = drawing.clone();
                st.builder.view_3d = three;
                let mut w = Win::new();
                let mut texts = w.settle(&mut st, &inp);
                if three {
                    let before = st.builder.view3d.rot;
                    let c = w.canvas().center();
                    w.drag(&mut st, &inp, c, c + Vec2::new(120.0, 40.0));
                    assert_ne!(st.builder.view3d.rot, before, "{name}: the drag turned the view");
                    texts = w.settle(&mut st, &inp);
                }
                let shape_line = texts.iter().find(|t| t.0.starts_with("shape: ")).map(|t| t.0.clone()).unwrap_or_default();
                assert_eq!(shape_line.contains("approximate"), name == "He–He", "{name}: {shape_line}");
                let mut handoff = None;
                for _ in 0..2 {
                    let at = text_at(&texts, "Insert into scene");
                    for a in w.click(&mut st, &inp, at) {
                        if let Action::Builder(h) = a {
                            handoff = Some(h);
                        }
                    }
                    if handoff.is_some() {
                        break;
                    }
                    texts = w.settle(&mut st, &inp);
                }
                let h = handoff.unwrap_or_else(|| panic!("{name}, 3D {three}: Insert sent nothing"));
                assert!(!h.replace && h.nuclei.len() == drawing.atoms.len(), "{name}: {h:?}");
                sent.push(h);
            }
            assert_eq!(sent[0], sent[1], "{name}: the same hand-off in 2D and 3D");
        }
    }

    /// Spec §5: in 3D a click places nothing (the same click in 2D places an atom).
    #[test]
    fn a_click_in_3d_adds_no_atom() {
        let scene = roomy();
        let cam = crate::ui::tests::camera();
        let inp = crate::ui::tests::input(&scene, None, &cam);
        for three in [true, false] {
            let mut st = UiState::new("h2", &scene);
            st.builder.open = true;
            st.builder.drawing = water_drawing();
            st.builder.view_3d = three;
            let mut w = Win::new();
            w.settle(&mut st, &inp);
            let r = w.canvas();
            // on empty space and on the shape's centre
            for p in [r.left_top() + Vec2::new(60.0, 80.0), r.center()] {
                w.click(&mut st, &inp, p);
                // apart in time: not a double-click
                for _ in 0..40 {
                    w.frame(&mut st, &inp, vec![]);
                }
            }
            if three {
                assert_eq!(st.builder.drawing, water_drawing(), "3D: the drawing is untouched");
                assert!(!st.builder.can_undo(), "3D: no edit was made");
            } else {
                assert!(st.builder.drawing.atoms.len() > 3, "2D: the same clicks place atoms (the clicks land)");
            }
        }
    }

    /// Spec §5: a drag in 3D turns the view only: the drawing and the shape handed off are kept.
    #[test]
    fn a_drag_in_3d_turns_only_the_view() {
        let scene = roomy();
        let cam = crate::ui::tests::camera();
        let inp = crate::ui::tests::input(&scene, None, &cam);
        let mut st = UiState::new("h2", &scene);
        st.builder.open = true;
        st.builder.drawing = water_drawing();
        st.builder.view_3d = true;
        let mut w = Win::new();
        w.settle(&mut st, &inp);
        let (drawing, shaped, rot) = (st.builder.drawing.clone(), st.builder.shaped().positions.clone(), st.builder.view3d.rot);
        let r = w.canvas();
        // across an atom (the centre) and across empty space
        w.drag(&mut st, &inp, r.center(), r.center() + Vec2::new(-90.0, 130.0));
        w.drag(&mut st, &inp, r.left_top() + Vec2::new(60.0, 80.0), r.left_top() + Vec2::new(200.0, 90.0));
        assert_eq!(st.builder.drawing, drawing);
        assert_eq!(st.builder.shaped().positions, shaped);
        assert!(st.builder.view3d.rot.angle_between(rot) > 0.5, "turned: {:?} → {:?}", rot, st.builder.view3d.rot);
        assert!(st.builder.selection.is_empty() && !st.builder.can_undo());
        // a double-click turns it back to the reset view
        let mut reset = super::super::view3d::View3d::default();
        reset.reset(&shaped);
        let c = r.center();
        let button = |pressed| egui::Event::PointerButton { pos: c, button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        w.frame(&mut st, &inp, vec![egui::Event::PointerMoved(c)]);
        w.frame(&mut st, &inp, vec![button(true)]);
        w.frame(&mut st, &inp, vec![button(false)]);
        w.frame(&mut st, &inp, vec![button(true)]);
        w.frame(&mut st, &inp, vec![button(false)]);
        w.frame(&mut st, &inp, vec![]);
        assert!(st.builder.view3d.rot.angle_between(reset.rot) < 1e-9, "double-click resets");
        assert_eq!(st.builder.drawing, drawing);
    }

    /// Spec §5: the search stays live in 3D, and a stamp shows its shape there.
    #[test]
    fn a_stamp_in_3d_shows_the_new_shape() {
        let scene = roomy();
        let cam = crate::ui::tests::camera();
        let inp = crate::ui::tests::input(&scene, None, &cam);
        let mut st = UiState::new("h2", &scene);
        st.builder.open = true;
        st.builder.view_3d = true;
        let mut w = Win::new();
        w.settle(&mut st, &inp);
        assert_eq!(st.builder.view3d.shown, 0);
        st.builder.search_query = "water".into();
        let texts = w.settle(&mut st, &inp);
        // the result row (its name, then the formula and source), not the search box's own text
        let row = texts.iter().find(|t| t.0.starts_with("water\n")).unwrap_or_else(|| panic!("no water row: {:?}", texts.iter().map(|t| &t.0).collect::<Vec<_>>())).1.center();
        w.click(&mut st, &inp, row);
        w.settle(&mut st, &inp);
        assert_eq!(st.builder.drawing.atoms.len(), 3, "the stamp went in");
        assert!(st.builder.view_3d, "still in 3D");
        assert_eq!(st.builder.view3d.shown, 3, "the view shows the stamped water");
        // the view was reset for it: the principal-axes frame of its shape
        let mut want = super::super::view3d::View3d::default();
        want.reset(&st.builder.shaped().positions.clone());
        assert!(st.builder.view3d.rot.angle_between(want.rot) < 1e-9);
    }

    /// Spec §5 (review): in 3D, Delete removes nothing (not the hovered atom, not the selection),
    /// while Ctrl-Z still undoes, as the palette's Undo does.
    #[test]
    fn in_3d_delete_does_nothing_and_ctrl_z_undoes() {
        let scene = roomy();
        let cam = crate::ui::tests::camera();
        let inp = crate::ui::tests::input(&scene, None, &cam);
        let mut st = UiState::new("h2", &scene);
        st.builder.open = true;
        st.builder.drawing = water_drawing();
        st.builder.checkpoint();
        st.builder.drawing.add_atom(6, [6.0, 0.0]);
        st.builder.selection = [0].into();
        st.builder.view_3d = true;
        let mut w = Win::new();
        w.settle(&mut st, &inp);
        let key = |key, modifiers| egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers };
        // the pointer resting on an atom (the view's centre is the shape's centroid: find a ball)
        let r = w.canvas();
        let z: Vec<u8> = st.builder.drawing.atoms.iter().map(|a| a.z).collect();
        let pos = st.builder.shaped().positions.clone();
        let on_atom = st.builder.view3d.project(&pos, &z, r)[0].0;
        w.frame(&mut st, &inp, vec![egui::Event::PointerMoved(on_atom)]);
        for k in [egui::Key::Delete, egui::Key::Backspace] {
            w.frame(&mut st, &inp, vec![key(k, egui::Modifiers::NONE)]);
            w.frame(&mut st, &inp, vec![]);
        }
        assert_eq!(st.builder.drawing.atoms.len(), 4, "Delete removed nothing in 3D");
        assert_eq!(st.builder.selection, [0].into());
        w.frame(&mut st, &inp, vec![key(egui::Key::Z, egui::Modifiers::COMMAND)]);
        w.frame(&mut st, &inp, vec![]);
        assert_eq!(st.builder.drawing, water_drawing(), "Ctrl-Z undid the added atom");
        assert_eq!(st.builder.view3d.shown, 3, "and the view shows the drawing as undone");
    }
}
