//! Line icons painted with egui's painter (no icon font): the toolbar's time and tool buttons and
//! the dock strip's tab buttons. Each icon is drawn in a 24 × 24 design box scaled into `rect`.
use super::theme;
use egui::{epaint::PathShape, pos2, Color32, Painter, Pos2, Rect, Response, Sense, Stroke, Ui, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Play,
    Pause,
    Step,
    Rewind,
    Orbit,
    Place,
    Select,
    Readouts,
    Graph,
    Model,
    Orbitals,
    Selection,
    Log,
    ChevronLeft,
    ChevronRight,
    Search,
}

/// Paints `icon` into `rect` in `colour`.
pub fn paint(p: &Painter, rect: Rect, icon: Icon, colour: Color32) {
    let s = rect.width().min(rect.height()) / 24.0;
    let o = rect.center() - Vec2::splat(12.0 * s);
    let at = |x: f32, y: f32| -> Pos2 { o + Vec2::new(x * s, y * s) };
    let stroke = Stroke::new(1.7 * s.max(0.6), colour);
    let line = |pts: &[(f32, f32)]| {
        p.add(PathShape::line(pts.iter().map(|&(x, y)| at(x, y)).collect(), stroke));
    };
    let closed = |pts: &[(f32, f32)]| {
        p.add(PathShape::closed_line(pts.iter().map(|&(x, y)| at(x, y)).collect(), stroke));
    };
    let filled = |pts: &[(f32, f32)]| {
        p.add(PathShape::convex_polygon(pts.iter().map(|&(x, y)| at(x, y)).collect(), colour, Stroke::NONE));
    };
    match icon {
        Icon::Play => filled(&[(7.0, 5.0), (19.0, 12.0), (7.0, 19.0)]),
        Icon::Pause => {
            p.rect_filled(Rect::from_min_max(at(6.0, 5.0), at(10.0, 19.0)), 1.0, colour);
            p.rect_filled(Rect::from_min_max(at(14.0, 5.0), at(18.0, 19.0)), 1.0, colour);
        }
        Icon::Step => {
            closed(&[(6.0, 5.0), (15.0, 12.0), (6.0, 19.0)]);
            line(&[(18.0, 5.0), (18.0, 19.0)]);
        }
        Icon::Rewind => {
            // a counter-clockwise arc with its arrow head
            let c = at(12.0, 12.0);
            let r = 8.0 * s;
            let pts: Vec<Pos2> = (0..=20)
                .map(|k| {
                    let a = std::f32::consts::PI * (1.0 + 1.6 * k as f32 / 20.0);
                    c + Vec2::new(a.cos(), a.sin()) * r
                })
                .collect();
            p.add(PathShape::line(pts, stroke));
            line(&[(4.0, 4.5), (4.0, 12.0), (11.0, 12.0)]);
        }
        Icon::Orbit => {
            let c = at(12.0, 12.0);
            let pts: Vec<Pos2> = (0..32)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / 32.0;
                    c + Vec2::new(9.0 * a.cos(), 4.0 * a.sin()) * s
                })
                .collect();
            p.add(PathShape::closed_line(pts, stroke));
            p.circle_filled(c, 2.0 * s, colour);
        }
        Icon::Place => {
            line(&[(12.0, 5.0), (12.0, 19.0)]);
            line(&[(5.0, 12.0), (19.0, 12.0)]);
        }
        Icon::Select => closed(&[(6.0, 3.0), (18.0, 11.0), (13.0, 12.5), (10.5, 18.0)]),
        Icon::Readouts => {
            line(&[(4.0, 6.0), (20.0, 6.0)]);
            line(&[(4.0, 12.0), (14.0, 12.0)]);
            line(&[(4.0, 18.0), (17.0, 18.0)]);
        }
        Icon::Graph => line(&[(3.0, 17.0), (8.0, 11.0), (12.0, 14.0), (17.0, 6.0), (21.0, 11.0)]),
        Icon::Model => {
            for (x, y) in [(6.0, 6.0), (18.0, 6.0), (12.0, 18.0)] {
                p.circle_stroke(at(x, y), 2.0 * s, stroke);
            }
            line(&[(7.5, 7.5), (11.0, 16.5)]);
            line(&[(16.5, 7.5), (13.0, 16.5)]);
            line(&[(8.0, 6.0), (16.0, 6.0)]);
        }
        Icon::Orbitals => {
            // a p orbital: two lobes about the nucleus
            p.circle_stroke(at(7.0, 12.0), 4.4 * s, stroke);
            p.circle_stroke(at(17.0, 12.0), 4.4 * s, stroke);
            p.circle_filled(at(12.0, 12.0), 1.4 * s, colour);
        }
        Icon::Selection => {
            p.circle_stroke(at(12.0, 12.0), 4.0 * s, stroke);
            for k in 0..8 {
                let a0 = std::f32::consts::TAU * k as f32 / 8.0;
                let a1 = a0 + 0.45;
                let c = at(12.0, 12.0);
                p.line_segment(
                    [c + Vec2::new(a0.cos(), a0.sin()) * 9.0 * s, c + Vec2::new(a1.cos(), a1.sin()) * 9.0 * s],
                    stroke,
                );
            }
        }
        Icon::Log => {
            closed(&[(6.0, 4.0), (15.0, 4.0), (18.0, 7.0), (18.0, 20.0), (6.0, 20.0)]);
            line(&[(9.0, 10.0), (15.0, 10.0)]);
            line(&[(9.0, 14.0), (15.0, 14.0)]);
            line(&[(9.0, 18.0), (13.0, 18.0)]);
        }
        Icon::ChevronRight => line(&[(10.0, 6.0), (16.0, 12.0), (10.0, 18.0)]),
        Icon::ChevronLeft => line(&[(14.0, 6.0), (8.0, 12.0), (14.0, 18.0)]),
        Icon::Search => {
            p.circle_stroke(at(11.0, 11.0), 6.0 * s, stroke);
            line(&[(15.5, 15.5), (20.0, 20.0)]);
        }
    }
}

/// A square icon button (`selected` draws the accent background). Disabled buttons paint dim.
pub fn button(ui: &mut Ui, icon: Icon, size: f32, selected: bool, enabled: bool) -> Response {
    let r = ui.add_enabled(
        enabled,
        egui::Button::selectable(selected, "").min_size(Vec2::splat(size)),
    );
    let colour = if !enabled {
        theme::INK_FAINT
    } else if selected {
        theme::ACCENT_INK
    } else if r.hovered() {
        theme::INK
    } else {
        theme::INK_DIM
    };
    let inner = Rect::from_center_size(r.rect.center(), Vec2::splat(size * 0.62));
    paint(ui.painter(), inner, icon, colour);
    r
}

/// A button with an icon on the left of its text (`primary` fills it with the accent).
pub fn text_button(ui: &mut Ui, icon: Icon, text: &str, width: f32, primary: bool, enabled: bool) -> Response {
    let fg = if primary { Color32::from_rgb(0x17, 0x11, 0x2a) } else { theme::INK };
    let mut b = egui::Button::new(egui::RichText::new(format!("      {text}")).strong().color(fg))
        .min_size(Vec2::new(width, 28.0));
    if primary {
        b = b.fill(theme::ACCENT).stroke(Stroke::new(1.0, theme::ACCENT));
    }
    let r = ui.add_enabled(enabled, b);
    let icon_rect = Rect::from_center_size(pos2(r.rect.left() + 17.0, r.rect.center().y), Vec2::splat(15.0));
    paint(ui.painter(), icon_rect, icon, if enabled { fg } else { fg.gamma_multiply(0.5) });
    r
}

/// A selectable segment with an icon left of its text (or the icon alone when `text` is empty).
pub fn seg_button(ui: &mut Ui, icon: Icon, text: &str, selected: bool) -> Response {
    let label = if text.is_empty() { String::new() } else { format!("      {text}") };
    let min = if text.is_empty() { Vec2::new(30.0, 24.0) } else { Vec2::new(0.0, 24.0) };
    let r = ui.add(egui::Button::selectable(selected, label).min_size(min));
    let fg = if selected { theme::ACCENT_INK } else if r.hovered() { theme::INK } else { theme::INK_DIM };
    let x = if text.is_empty() { r.rect.center().x } else { r.rect.left() + 14.0 };
    paint(ui.painter(), Rect::from_center_size(pos2(x, r.rect.center().y), Vec2::splat(14.0)), icon, fg);
    r
}

/// An unframed strip button for the dock (34 × 34), with the selected state.
pub fn strip_button(ui: &mut Ui, icon: Icon, selected: bool) -> Response {
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(34.0), Sense::click());
    let bg = if selected {
        theme::ACCENT_BG
    } else if r.hovered() {
        theme::PANEL_3
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 7.0, bg);
    let fg = if selected {
        theme::ACCENT_INK
    } else if r.hovered() {
        theme::INK
    } else {
        theme::INK_DIM
    };
    paint(ui.painter(), Rect::from_center_size(rect.center(), Vec2::splat(18.0)), icon, fg);
    r
}
