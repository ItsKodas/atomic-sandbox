//! The chrome's look (the mockup's palette: egui dark nudged toward violet, the density glow as
//! the accent) and the small building blocks every part of it shares: toolbar groups, section
//! headers, pills, the info "i", the queued-item placeholder. Display only: nothing here is
//! physics.
use egui::{Color32, CornerRadius, FontId, Margin, RichText, Stroke, Theme, Vec2};

const fn rgb(c: u32) -> Color32 {
    Color32::from_rgb((c >> 16) as u8, (c >> 8) as u8, c as u8)
}

/// Surfaces: the app background, panels and their raised levels, and the separators.
pub const APP: Color32 = rgb(0x15141b);
pub const PANEL: Color32 = rgb(0x1c1b23);
pub const PANEL_2: Color32 = rgb(0x24222d);
pub const PANEL_3: Color32 = rgb(0x2e2b39);
pub const LINE: Color32 = rgb(0x35323f);
pub const LINE_SOFT: Color32 = rgb(0x2a2833);
/// Text: normal, dim, faint.
pub const INK: Color32 = rgb(0xd6d2e0);
pub const INK_DIM: Color32 = rgb(0x9792a6);
pub const INK_FAINT: Color32 = rgb(0x6b6779);
/// The accent (the density glow), its text colour on dark, and its selected background.
pub const ACCENT: Color32 = rgb(0xa78bfa);
pub const ACCENT_INK: Color32 = rgb(0xcbbcff);
pub const ACCENT_BG: Color32 = rgb(0x2d2547);
pub const GLOW: Color32 = rgb(0x8b5cf6);
/// The duotone icons' fill (`icons::paint`): the accent at 32 % (alpha 82), whatever the stroke
/// colour; half that on a disabled control.
pub const ICON_FILL: Color32 = Color32::from_rgba_unmultiplied_const(0xa7, 0x8b, 0xfa, 82);
/// A danger button's text.
pub const DANGER_INK: Color32 = rgb(0xffb4b4);
/// The overlays' translucent card fill.
pub const CARD: Color32 = Color32::from_rgba_premultiplied(20, 19, 27, 236);

/// Installs the dark theme with the accent (called once per egui context).
pub fn apply(ctx: &egui::Context) {
    // The proportional face has no arrows, and the orbital viewer's spin labels and honesty text
    // are full of them (↑ ↓): fall back to the monospace face (Hack), which has them, for any glyph
    // the proportional one lacks — else they draw as missing-glyph boxes.
    let mut fonts = egui::FontDefinitions::default();
    if let Some(fallback) = fonts.families.get(&egui::FontFamily::Monospace).and_then(|f| f.first().cloned()) {
        fonts.families.entry(egui::FontFamily::Proportional).or_default().push(fallback);
    }
    ctx.set_fonts(fonts);
    ctx.set_theme(Theme::Dark);
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = APP;
    v.faint_bg_color = rgb(0x201f28);
    v.code_bg_color = PANEL_2;
    v.hyperlink_color = ACCENT_INK;
    v.window_stroke = Stroke::new(1.0, rgb(0x3e3a4b));
    v.window_corner_radius = CornerRadius::same(9);
    v.menu_corner_radius = CornerRadius::same(9);
    v.selection.bg_fill = ACCENT_BG;
    v.selection.stroke = Stroke::new(1.0, ACCENT_INK);
    v.override_text_color = None;
    let w = &mut v.widgets;
    w.noninteractive.bg_stroke = Stroke::new(1.0, LINE_SOFT);
    w.noninteractive.fg_stroke = Stroke::new(1.0, INK);
    w.inactive.weak_bg_fill = PANEL_2;
    w.inactive.bg_fill = PANEL_2;
    w.inactive.bg_stroke = Stroke::new(1.0, LINE);
    w.inactive.fg_stroke = Stroke::new(1.0, INK);
    w.hovered.weak_bg_fill = PANEL_3;
    w.hovered.bg_fill = PANEL_3;
    w.hovered.bg_stroke = Stroke::new(1.0, rgb(0x4b3d78));
    w.active.weak_bg_fill = ACCENT_BG;
    w.active.bg_fill = ACCENT_BG;
    w.active.bg_stroke = Stroke::new(1.0, ACCENT);
    w.open.weak_bg_fill = ACCENT_BG;
    w.open.bg_fill = ACCENT_BG;
    for s in [&mut w.noninteractive, &mut w.inactive, &mut w.hovered, &mut w.active, &mut w.open] {
        s.corner_radius = CornerRadius::same(6);
    }
    ctx.set_visuals_of(Theme::Dark, v);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = Vec2::new(6.0, 5.0);
        s.spacing.button_padding = Vec2::new(8.0, 3.0);
        s.spacing.menu_margin = Margin::same(6);
        s.interaction.tooltip_delay = 0.25;
        s.spacing.tooltip_width = 420.0;
    });
}

/// A toolbar group's caps label ("TIME", "SPEED · STEPS PER TICK").
pub fn caps(text: &str) -> RichText {
    RichText::new(text.to_uppercase()).font(FontId::monospace(9.5)).color(INK_FAINT)
}

/// A dock section header (caps, faint).
pub fn section(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add_space(2.0);
    ui.label(caps(text))
}

/// A small dim note line.
pub fn note(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    ui.label(RichText::new(text.into()).size(11.0).color(INK_DIM))
}

/// The "i" whose tooltip carries the long text (the tooltip texts of the old side panel).
pub fn info(ui: &mut egui::Ui, tip: impl Into<egui::WidgetText>) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(14.0), egui::Sense::hover());
    let p = ui.painter();
    p.circle_stroke(rect.center(), 6.5, Stroke::new(1.0, INK_FAINT));
    p.text(rect.center(), egui::Align2::CENTER_CENTER, "i", FontId::monospace(9.0), INK_DIM);
    r.on_hover_text(tip)
}

/// A coloured pill (the source badge: surrogate / fallback / DFT).
pub fn pill(ui: &mut egui::Ui, text: &str, colour: Color32) -> egui::Response {
    egui::Frame::new()
        .fill(colour.gamma_multiply(0.12))
        .stroke(Stroke::new(1.0, colour))
        .corner_radius(CornerRadius::same(255))
        .inner_margin(Margin::symmetric(7, 1))
        .show(ui, |ui| ui.label(RichText::new(text).monospace().size(11.0).color(colour)))
        .response
}

/// A status light (the trust light, the log's error marker).
pub fn light(ui: &mut egui::Ui, colour: Color32) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 6.0, colour.gamma_multiply(0.25));
    ui.painter().circle_filled(rect.center(), 4.0, colour);
    r
}

/// A card frame for the overlays in the 3D view.
pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 7))
}

/// A segmented control: one selectable button per option, returns the clicked option.
pub fn segmented<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    current: T,
    options: &[(T, &str, &str)],
) -> Option<T> {
    segmented_in(ui, current, options, false)
}

/// [`segmented`] whose options wrap onto more rows when the space is narrow (a panel).
pub fn segmented_wrapped<T: Copy + PartialEq>(ui: &mut egui::Ui, current: T, options: &[(T, &str, &str)]) -> Option<T> {
    segmented_in(ui, current, options, true)
}

fn segmented_in<T: Copy + PartialEq>(ui: &mut egui::Ui, current: T, options: &[(T, &str, &str)], wrap: bool) -> Option<T> {
    let mut picked = None;
    egui::Frame::new()
        .fill(APP)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            let add = |ui: &mut egui::Ui| {
                for &(v, label, tip) in options {
                    let r = ui.add(egui::Button::selectable(current == v, label));
                    let r = if tip.is_empty() { r } else { r.on_hover_text(tip) };
                    if r.clicked() {
                        picked = Some(v);
                    }
                }
            };
            if wrap {
                ui.horizontal_wrapped(add);
            } else {
                ui.horizontal(add);
            }
        });
    picked
}

/// Inside a panel card ([`panel_card`]): the inactive widgets' fill (a slider's rail, a checkbox's
/// box, a button's face) — lighter than the card's [`PANEL_2`] so it reads against it (user report
/// 2026-11-14: in the cards the rails were the card's own colour and only the handles showed).
pub const CARD_WIDGET: Color32 = rgb(0x3a3648);
/// …hovered.
pub const CARD_WIDGET_HOVER: Color32 = rgb(0x48425a);
/// …and the inactive widgets' outline.
pub const CARD_EDGE: Color32 = rgb(0x575170);

/// A panel card's frame: [`PANEL_2`] fill, [`LINE`] outline, the redesigned panels' radius and
/// margins (the Model tab's verdict card, the Activity job card, the View panel's LOOK, …).
pub fn panel_card_frame() -> egui::Frame {
    egui::Frame::new().fill(PANEL_2).stroke(Stroke::new(1.0, LINE)).corner_radius(8).inner_margin(Margin::symmetric(10, 8))
}

/// The widget visuals inside a card: inactive widgets on [`CARD_WIDGET`] with a [`CARD_EDGE`]
/// outline, hovered lighter, active and open in the accent — so a slider's rail, a checkbox, a
/// button and a text field keep their contrast against the card's fill.
pub fn card_widgets(v: &mut egui::Visuals) {
    let w = &mut v.widgets;
    w.inactive.bg_fill = CARD_WIDGET;
    w.inactive.weak_bg_fill = CARD_WIDGET;
    w.inactive.bg_stroke = Stroke::new(1.0, CARD_EDGE);
    w.hovered.bg_fill = CARD_WIDGET_HOVER;
    w.hovered.weak_bg_fill = CARD_WIDGET_HOVER;
    w.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.7));
    w.active.bg_stroke = Stroke::new(1.0, ACCENT);
    w.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
    v.extreme_bg_color = APP;
}

/// A card: `frame` (usually [`panel_card_frame`]) holding `add`, its widgets drawn with
/// [`card_widgets`]. Every redesigned panel's cards go through this one helper.
pub fn panel_card<R>(ui: &mut egui::Ui, frame: egui::Frame, add: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
    frame.show(ui, |ui| {
        card_widgets(ui.visuals_mut());
        add(ui)
    })
}

/// A framed choice button (the View panel's look presets, the quick presets, the log's filter
/// chips): selected — the accent fill and outline; else a neutral face with a visible outline
/// (`Button::selectable` draws no frame until hovered, which read as bare text in a card).
pub fn choice_button(selected: bool, text: impl Into<egui::WidgetText>) -> egui::Button<'static> {
    let b = egui::Button::new(text).corner_radius(6);
    if selected {
        b.fill(ACCENT_BG).stroke(Stroke::new(1.0, ACCENT))
    } else {
        b.fill(CARD_WIDGET).stroke(Stroke::new(1.0, CARD_EDGE))
    }
}

/// A panel's folding header (the View panel's look: a caret, the title; the Model and Activity
/// panels' folds): `open` says which caret to draw, `summary` is cut with "…" to the room left of
/// the title, and `link` is a clickable word at the right end (its own response, drawn over the
/// header so a click on it does not fold). `sub`: a smaller header for a fold inside a fold. The
/// header's response is clicked when the fold should toggle; the caller keeps the state.
pub fn fold_header(
    ui: &mut egui::Ui,
    title: &str,
    open: bool,
    summary: Option<(String, Color32)>,
    link: Option<&str>,
    sub: bool,
) -> (egui::Response, Option<egui::Response>) {
    use egui::{pos2, vec2};
    let (h, size) = if sub { (22.0, 12.5) } else { (26.0, 13.5) };
    let (rect, r) = ui.allocate_exact_size(vec2(ui.available_width(), h), egui::Sense::click());
    let p = ui.painter().clone();
    if r.hovered() {
        p.rect_filled(rect, 5.0, PANEL_2);
    }
    let ink = if r.hovered() { INK } else { INK_DIM };
    let c = pos2(rect.left() + 10.0, rect.center().y);
    let tri = if open { vec![c + vec2(-4.0, -2.0), c + vec2(4.0, -2.0), c + vec2(0.0, 3.0)] } else { vec![c + vec2(-2.0, -4.0), c + vec2(3.0, 0.0), c + vec2(-2.0, 4.0)] };
    p.add(egui::Shape::convex_polygon(tri, ink, Stroke::NONE));
    let title_ink = if sub { INK_DIM } else { INK };
    let g = p.layout_no_wrap(title.to_string(), FontId::proportional(size), title_ink);
    let tw = g.size().x;
    p.galley(pos2(rect.left() + 22.0, rect.center().y - g.size().y / 2.0), g, title_ink);
    let mut right = rect.right() - 6.0;
    let link_r = link.map(|w| {
        let g = p.layout_no_wrap(w.to_string(), FontId::proportional(11.5), ACCENT_INK);
        let lr = egui::Rect::from_min_size(pos2(right - g.size().x, rect.center().y - g.size().y / 2.0), g.size());
        p.galley(lr.min, g, ACCENT_INK);
        p.line_segment([pos2(lr.left(), lr.bottom()), pos2(lr.right(), lr.bottom())], Stroke::new(1.0, ACCENT_INK.gamma_multiply(0.6)));
        right = lr.left() - 8.0;
        ui.interact(lr.expand(2.0), r.id.with("link"), egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand)
    });
    if let Some((text, colour)) = summary {
        let room = right - (rect.left() + 22.0 + tw + 12.0);
        if room > 24.0 {
            let mut job = egui::text::LayoutJob::single_section(text, egui::TextFormat { font_id: FontId::proportional(11.5), color: colour, ..Default::default() });
            job.wrap = egui::text::TextWrapping { max_width: room, max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
            let g = p.layout_job(job);
            p.galley(pos2(right - g.size().x, rect.center().y - g.size().y / 2.0), g, colour);
        }
    }
    let r = r.on_hover_text(if open { "fold this section" } else { "unfold this section" });
    (r, link_r)
}
