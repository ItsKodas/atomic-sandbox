//! Duotone icons painted with egui's painter (no icon font, no image files): the toolbar's time
//! and tool buttons, the workspace's tabs and leaf buttons. Each icon is drawn in a 24 × 24 design
//! box scaled into `rect`, with 1.7 px line strokes in the caller's colour; the panels' and the
//! window controls' icons first lay a soft fill on their main shape in [`theme::ICON_FILL`], the
//! accent at 32 %, whatever the stroke colour (the duotone set, the user's choice of three styles
//! on 2026-11-03, docking spec decision 3). Display only.
use super::theme;
use egui::{epaint::PathShape, pos2, Color32, Painter, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    /// A filled triangle (Play).
    Play,
    /// Two filled bars (Pause).
    Pause,
    /// A triangle and a bar (Step).
    Step,
    /// A counter-clockwise arc with its arrow head (the Play snapshot's Rewind).
    Rewind,
    /// An orbit round a dot (the Orbit tool).
    Orbit,
    /// A plus (the Place tool).
    Place,
    /// An open hand (the Grab tool).
    Grab,
    /// Four arrows from a centre (the Edit tool's move).
    Move,
    /// Three lines of text on a soft rounded-square backdrop (the Readouts tab).
    Readouts,
    /// A line chart, the area under the line filled (the Graph tab).
    Graph,
    /// Three bonded atoms, the atoms filled (the Model tab; the empty scene's placeholder).
    Model,
    /// A pulse line on a soft rounded-square backdrop (the Activity tab).
    Activity,
    /// A p orbital: two lobes, both filled, about the nucleus (the Orbitals tab).
    Orbitals,
    /// A filled circle in a dashed ring (the Selection tab).
    Selection,
    /// A page with a folded corner, the page filled, and its lines (the Log tab).
    Log,
    /// The old docks' collapse chevrons; unused until the auto-hide strip (docking D2).
    #[allow(dead_code)]
    ChevronLeft,
    #[allow(dead_code)]
    ChevronRight,
    /// A magnifier (the control search).
    Search,
    /// A camera (the capture's Screenshot).
    Camera,
    /// A filled dot (Record) and a filled square (Stop).
    Record,
    Stop,
    /// Three sliders (the capture's quick settings).
    Sliders,
    /// A window with an arrow leaving it, the window's corner filled (the capture's pop-out view;
    /// D3's detach a panel).
    PopOut,
    /// A folded map, its middle panel filled (the Knowledge tab).
    Knowledge,
    /// An eye: the eye shape filled, the pupil stroked (the View tab).
    View,
    /// A push-pin, its head filled (a docked leaf's pin: auto-hide its active tab, D2).
    Pin,
    /// The push-pin tilted 45° (an auto-hidden panel's flyout: pin it back, D2).
    Unpin,
    /// A window, filled, with an arrow entering it (D3's dock a detached panel back).
    #[allow(dead_code)]
    DockBack,
    /// A box halved, its left half filled (D2's split a leaf).
    #[allow(dead_code)]
    Split,
    /// A 2 × 2 grid, its top-left cell filled (D2's layouts menu).
    #[allow(dead_code)]
    Layouts,
    /// An X (a leaf's close button, closing its panels).
    Close,
    /// Six dots (D2's drag handle).
    #[allow(dead_code)]
    Drag,
}

impl Icon {
    /// Every icon (the painter's tests draw each).
    #[cfg(test)]
    pub const ALL: [Icon; 32] = [
        Icon::Play,
        Icon::Pause,
        Icon::Step,
        Icon::Rewind,
        Icon::Orbit,
        Icon::Place,
        Icon::Grab,
        Icon::Move,
        Icon::Readouts,
        Icon::Graph,
        Icon::Model,
        Icon::Activity,
        Icon::Orbitals,
        Icon::Selection,
        Icon::Log,
        Icon::ChevronLeft,
        Icon::ChevronRight,
        Icon::Search,
        Icon::Camera,
        Icon::Record,
        Icon::Stop,
        Icon::Sliders,
        Icon::PopOut,
        Icon::Knowledge,
        Icon::View,
        Icon::Pin,
        Icon::Unpin,
        Icon::DockBack,
        Icon::Split,
        Icon::Layouts,
        Icon::Close,
        Icon::Drag,
    ];
}

/// Paints `icon` into `rect` in `colour` (its fill, when it has one, in [`theme::ICON_FILL`]).
pub fn paint(p: &Painter, rect: Rect, icon: Icon, colour: Color32) {
    draw(p, rect, icon, colour, theme::ICON_FILL);
}

/// Paints `icon` as a disabled control's: the fill at half its alpha (the caller dims `colour`).
pub fn paint_disabled(p: &Painter, rect: Rect, icon: Icon, colour: Color32) {
    draw(p, rect, icon, colour, theme::ICON_FILL.gamma_multiply(0.5));
}

/// The points of the cubic Bézier from `a` (left out) to `d` with controls `b` and `c`, `n` steps.
fn bezier(out: &mut Vec<(f32, f32)>, [a, b, c, d]: [(f32, f32); 4], n: usize) {
    for k in 1..=n {
        let t = k as f32 / n as f32;
        let u = 1.0 - t;
        let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
        out.push((w[0] * a.0 + w[1] * b.0 + w[2] * c.0 + w[3] * d.0, w[0] * a.1 + w[1] * b.1 + w[2] * c.1 + w[3] * d.1));
    }
}

/// The eye's outline (the View icon): two curves meeting at the corners (2, 12) and (22, 12).
fn eye() -> Vec<(f32, f32)> {
    let mut pts = vec![(2.0, 12.0)];
    bezier(&mut pts, [(2.0, 12.0), (6.0, 5.5), (18.0, 5.5), (22.0, 12.0)], 12);
    bezier(&mut pts, [(22.0, 12.0), (18.0, 18.5), (6.0, 18.5), (2.0, 12.0)], 12);
    pts.pop(); // back at the start
    pts
}

/// Paints `icon`: its fill in `fill` first (a fill must be convex, so a concave shape is filled in
/// convex pieces), then its strokes in `colour`.
fn draw(p: &Painter, rect: Rect, icon: Icon, colour: Color32, fill: Color32) {
    let s = rect.width().min(rect.height()) / 24.0;
    let c = rect.center();
    // the unpinned pin is the pin tilted 45° (clockwise on screen) about the box's centre
    let (sin, cos) = if icon == Icon::Unpin { std::f32::consts::FRAC_PI_4.sin_cos() } else { (0.0, 1.0) };
    let at = |x: f32, y: f32| -> Pos2 {
        let (dx, dy) = (x - 12.0, y - 12.0);
        c + Vec2::new(dx * cos - dy * sin, dx * sin + dy * cos) * s
    };
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
    // the duotone fill (painted before the strokes)
    let fill_poly = |pts: &[(f32, f32)]| {
        p.add(PathShape::convex_polygon(pts.iter().map(|&(x, y)| at(x, y)).collect(), fill, Stroke::NONE));
    };
    let fill_rect = |x0: f32, y0: f32, x1: f32, y1: f32, r: f32| p.rect_filled(Rect::from_min_max(at(x0, y0), at(x1, y1)), r * s, fill);
    let fill_circle = |x: f32, y: f32, r: f32| p.circle_filled(at(x, y), r * s, fill);
    let rect_line = |x0: f32, y0: f32, x1: f32, y1: f32, r: f32| {
        p.rect_stroke(Rect::from_min_max(at(x0, y0), at(x1, y1)), r * s, stroke, StrokeKind::Middle);
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
        Icon::Grab => {
            // an open hand: four fingers, a thumb and the palm
            for (x, top) in [(8.0, 6.0), (11.0, 4.5), (14.0, 5.0), (17.0, 7.0)] {
                line(&[(x, 13.0), (x, top)]);
            }
            line(&[(8.0, 13.0), (5.5, 11.0), (4.5, 12.0), (7.5, 17.0), (10.0, 20.0), (15.5, 20.0), (17.0, 17.0), (17.0, 13.0)]);
        }
        Icon::Move => {
            // a cross with an arrow head on each end
            line(&[(12.0, 3.5), (12.0, 20.5)]);
            line(&[(3.5, 12.0), (20.5, 12.0)]);
            line(&[(9.5, 6.0), (12.0, 3.5), (14.5, 6.0)]);
            line(&[(9.5, 18.0), (12.0, 20.5), (14.5, 18.0)]);
            line(&[(6.0, 9.5), (3.5, 12.0), (6.0, 14.5)]);
            line(&[(18.0, 9.5), (20.5, 12.0), (18.0, 14.5)]);
        }
        Icon::Readouts => {
            fill_rect(2.5, 3.0, 21.5, 21.0, 3.0);
            line(&[(4.0, 6.0), (20.0, 6.0)]);
            line(&[(4.0, 12.0), (14.0, 12.0)]);
            line(&[(4.0, 18.0), (17.0, 18.0)]);
        }
        Icon::Graph => {
            // the area under the line down to y = 20, a trapezoid under each segment
            let pts = [(3.0, 17.0), (8.0, 11.0), (12.0, 14.0), (17.0, 6.0), (21.0, 11.0)];
            for w in pts.windows(2) {
                fill_poly(&[w[0], w[1], (w[1].0, 20.0), (w[0].0, 20.0)]);
            }
            line(&pts);
        }
        Icon::Model => {
            for (x, y) in [(6.0, 6.0), (18.0, 6.0), (12.0, 18.0)] {
                fill_circle(x, y, 2.0);
            }
            for (x, y) in [(6.0, 6.0), (18.0, 6.0), (12.0, 18.0)] {
                p.circle_stroke(at(x, y), 2.0 * s, stroke);
            }
            line(&[(7.5, 7.5), (11.0, 16.5)]);
            line(&[(16.5, 7.5), (13.0, 16.5)]);
            line(&[(8.0, 6.0), (16.0, 6.0)]);
        }
        Icon::Activity => {
            fill_rect(2.5, 3.0, 21.5, 21.0, 3.0);
            line(&[(3.0, 12.0), (7.0, 12.0), (9.0, 7.0), (13.0, 17.0), (15.0, 12.0), (21.0, 12.0)]);
        }
        Icon::Orbitals => {
            // a p orbital: two lobes about the nucleus
            fill_circle(7.0, 12.0, 4.4);
            fill_circle(17.0, 12.0, 4.4);
            p.circle_stroke(at(7.0, 12.0), 4.4 * s, stroke);
            p.circle_stroke(at(17.0, 12.0), 4.4 * s, stroke);
            p.circle_filled(at(12.0, 12.0), 1.4 * s, colour);
        }
        Icon::Selection => {
            fill_circle(12.0, 12.0, 4.0);
            p.circle_stroke(at(12.0, 12.0), 4.0 * s, stroke);
            for k in 0..8 {
                let a0 = std::f32::consts::TAU * k as f32 / 8.0;
                let a1 = a0 + 0.45;
                p.line_segment([c + Vec2::new(a0.cos(), a0.sin()) * 9.0 * s, c + Vec2::new(a1.cos(), a1.sin()) * 9.0 * s], stroke);
            }
        }
        Icon::Log => {
            let page = [(6.0, 4.0), (15.0, 4.0), (18.0, 7.0), (18.0, 20.0), (6.0, 20.0)];
            fill_poly(&page);
            closed(&page);
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
        Icon::Camera => {
            closed(&[(3.5, 8.0), (8.0, 8.0), (9.5, 5.5), (14.5, 5.5), (16.0, 8.0), (20.5, 8.0), (20.5, 19.0), (3.5, 19.0)]);
            p.circle_stroke(at(12.0, 13.2), 3.6 * s, stroke);
        }
        Icon::Record => {
            p.circle_filled(at(12.0, 12.0), 6.5 * s, colour);
        }
        Icon::Stop => {
            p.rect_filled(Rect::from_min_max(at(6.5, 6.5), at(17.5, 17.5)), 1.5 * s, colour);
        }
        Icon::Sliders => {
            for (y, knob) in [(6.0, 15.0), (12.0, 8.0), (18.0, 13.0)] {
                line(&[(4.0, y), (20.0, y)]);
                p.circle_filled(at(knob, y), 2.4 * s, colour);
            }
        }
        Icon::PopOut => {
            fill_rect(5.0, 9.0, 15.0, 19.0, 1.0);
            line(&[(10.5, 5.0), (5.0, 5.0), (5.0, 19.0), (19.0, 19.0), (19.0, 13.5)]);
            line(&[(11.0, 13.0), (19.5, 4.5)]);
            line(&[(13.5, 4.5), (19.5, 4.5), (19.5, 10.5)]);
        }
        Icon::Knowledge => {
            fill_poly(&[(9.0, 4.0), (15.0, 6.5), (15.0, 20.0), (9.0, 17.5)]);
            closed(&[(3.0, 6.5), (9.0, 4.0), (15.0, 6.5), (21.0, 4.0), (21.0, 17.5), (15.0, 20.0), (9.0, 17.5), (3.0, 20.0)]);
            line(&[(9.0, 4.0), (9.0, 17.5)]);
            line(&[(15.0, 6.5), (15.0, 20.0)]);
        }
        Icon::View => {
            let eye = eye();
            fill_poly(&eye);
            closed(&eye);
            p.circle_stroke(at(12.0, 12.0), 3.0 * s, stroke);
        }
        Icon::Pin | Icon::Unpin => {
            // the head and the collar (two convex pieces of one outline), then the needle
            fill_poly(&[(9.0, 3.0), (15.0, 3.0), (14.0, 9.0), (10.0, 9.0)]);
            fill_poly(&[(10.0, 9.0), (14.0, 9.0), (17.5, 13.0), (6.5, 13.0)]);
            closed(&[(9.0, 3.0), (15.0, 3.0), (14.0, 9.0), (17.5, 13.0), (6.5, 13.0), (10.0, 9.0)]);
            line(&[(12.0, 13.0), (12.0, 21.0)]);
        }
        Icon::DockBack => {
            fill_rect(4.0, 4.0, 20.0, 20.0, 2.0);
            rect_line(4.0, 4.0, 20.0, 20.0, 2.0);
            line(&[(17.0, 7.0), (10.0, 14.0)]);
            line(&[(10.0, 9.0), (10.0, 14.0), (15.0, 14.0)]);
        }
        Icon::Split => {
            fill_rect(4.0, 5.0, 12.0, 19.0, 0.0);
            rect_line(3.5, 4.5, 20.5, 19.5, 2.0);
            line(&[(12.0, 4.5), (12.0, 19.5)]);
        }
        Icon::Layouts => {
            fill_rect(4.0, 4.0, 11.0, 11.0, 1.2);
            for (x, y) in [(4.0, 4.0), (13.0, 4.0), (4.0, 13.0), (13.0, 13.0)] {
                rect_line(x, y, x + 7.0, y + 7.0, 1.2);
            }
        }
        Icon::Close => {
            line(&[(6.0, 6.0), (18.0, 18.0)]);
            line(&[(18.0, 6.0), (6.0, 18.0)]);
        }
        Icon::Drag => {
            for y in [6.0, 12.0, 18.0] {
                for x in [9.0, 15.0] {
                    p.circle_filled(at(x, y), 1.3 * s, colour);
                }
            }
        }
    }
}

/// A square icon button (`selected` draws the accent background). Disabled buttons paint dim,
/// their fill at half alpha.
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
    if enabled {
        paint(ui.painter(), inner, icon, colour);
    } else {
        paint_disabled(ui.painter(), inner, icon, colour);
    }
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
    if enabled {
        paint(ui.painter(), icon_rect, icon, fg);
    } else {
        paint_disabled(ui.painter(), icon_rect, icon, fg.gamma_multiply(0.5));
    }
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

/// An unframed strip button (34 × 34), with the selected state: the auto-hide strips' (docking
/// spec D-10, D2; the old docks' icon strip).
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

#[cfg(test)]
mod tests {
    use super::*;
    use egui::epaint::Shape;

    /// The icons with a duotone fill on their main shape (the user's set, 2026-11-03).
    const DUOTONE: [Icon; 15] = [
        Icon::Model,
        Icon::Knowledge,
        Icon::Selection,
        Icon::Readouts,
        Icon::View,
        Icon::Orbitals,
        Icon::Log,
        Icon::Activity,
        Icon::Graph,
        Icon::Pin,
        Icon::Unpin,
        Icon::PopOut,
        Icon::DockBack,
        Icon::Split,
        Icon::Layouts,
    ];

    /// The shapes `icon` paints into a 24 px rect at (10, 10) on a headless context, in
    /// `colour`, enabled or not.
    fn painted(icon: Icon, colour: Color32, enabled: bool) -> Vec<Shape> {
        let ctx = egui::Context::default();
        let rect = Rect::from_min_size(pos2(10.0, 10.0), Vec2::splat(24.0));
        let clip = rect.expand(3.0); // marks this icon's shapes among whatever else the pass adds
        let raw = egui::RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(64.0))), ..Default::default() };
        let full = ctx.run_ui(raw, |ui| {
            let p = ui.painter().with_clip_rect(clip);
            if enabled {
                paint(&p, rect, icon, colour);
            } else {
                paint_disabled(&p, rect, icon, colour);
            }
        });
        full.shapes.into_iter().filter(|c| c.clip_rect == clip).map(|c| c.shape).collect()
    }

    fn points(s: &Shape, out: &mut Vec<Pos2>) {
        match s {
            Shape::Vec(v) => v.iter().for_each(|s| points(s, out)),
            Shape::Circle(c) => out.extend([c.center - Vec2::splat(c.radius), c.center + Vec2::splat(c.radius)]),
            Shape::LineSegment { points: p, .. } => out.extend(p),
            Shape::Path(p) => out.extend(&p.points),
            Shape::Rect(r) => out.extend([r.rect.min, r.rect.max]),
            Shape::Mesh(m) => out.extend(m.vertices.iter().map(|v| v.pos)),
            _ => {}
        }
    }

    fn fill_of(s: &Shape) -> Option<Color32> {
        match s {
            Shape::Path(p) => Some(p.fill),
            Shape::Circle(c) => Some(c.fill),
            Shape::Rect(r) => Some(r.fill),
            _ => None,
        }
    }

    /// Every icon paints at least one shape, every point finite and inside its 24 px box.
    #[test]
    fn every_icon_paints_finite_shapes_inside_its_box() {
        let rect = Rect::from_min_size(pos2(10.0, 10.0), Vec2::splat(24.0));
        for icon in Icon::ALL {
            let shapes = painted(icon, theme::INK, true);
            assert!(!shapes.is_empty(), "{icon:?} paints nothing");
            let mut pts = Vec::new();
            shapes.iter().for_each(|s| points(s, &mut pts));
            assert!(!pts.is_empty(), "{icon:?}");
            for q in pts {
                assert!(q.x.is_finite() && q.y.is_finite(), "{icon:?}: {q:?}");
                assert!(rect.expand(0.5).contains(q), "{icon:?}: {q:?} outside {rect:?}");
            }
        }
    }

    /// Duotone (docking spec decision 3): each of the set paints its fill first, in
    /// `theme::ICON_FILL` whatever the stroke colour, at half alpha when disabled; the others
    /// paint no accent fill.
    #[test]
    fn the_duotone_set_paints_its_fill_first() {
        for icon in Icon::ALL {
            for colour in [theme::INK, theme::INK_DIM, super::super::RED] {
                let first = painted(icon, colour, true).first().and_then(fill_of);
                let dim = painted(icon, colour, false).first().and_then(fill_of);
                if DUOTONE.contains(&icon) {
                    assert_eq!(first, Some(theme::ICON_FILL), "{icon:?} in {colour:?}");
                    assert_eq!(dim, Some(theme::ICON_FILL.gamma_multiply(0.5)), "{icon:?} disabled");
                } else {
                    let all: Vec<Color32> = painted(icon, colour, true).iter().filter_map(fill_of).collect();
                    assert!(!all.contains(&theme::ICON_FILL), "{icon:?} has no fill");
                }
            }
        }
        assert_eq!(theme::ICON_FILL.a(), 82, "the accent at 32 %");
    }
}
