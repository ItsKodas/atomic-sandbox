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
    let mut picked = None;
    egui::Frame::new()
        .fill(APP)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.horizontal(|ui| {
                for &(v, label, tip) in options {
                    let r = ui.add(egui::Button::selectable(current == v, label));
                    let r = if tip.is_empty() { r } else { r.on_hover_text(tip) };
                    if r.clicked() {
                        picked = Some(v);
                    }
                }
            });
        });
    picked
}
