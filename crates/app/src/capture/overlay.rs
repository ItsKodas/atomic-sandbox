//! What a hidden-UI capture may draw over the 3D view (`Settings::overlay`): the window's labels
//! (`Viewer::label_overlay` with the capture's camera and its own smoothing state), a title in the
//! top-left corner listing the molecules, the time readout under it and a distance scale bottom
//! left. Drawn by a second egui context sized to the frame (its own font atlas, painted by the
//! painter's capture egui renderer), so the window's UI state is never touched.
use crate::viewer::LabelSmoother;
use egui::{pos2, vec2, Align2, Color32, FontId, LayerId, Order};
use std::cell::RefCell;
use std::collections::HashMap;

/// The capture's egui context (made on first use) and its labels' smoothing state, kept between
/// frames: a recording's labels are smoothed on the video's clock.
#[derive(Default)]
pub struct LabelState {
    ctx: Option<egui::Context>,
    smooth: Smooth,
    sides: Sides,
}

/// The labels' smoothing state and their bond-length sides (`Viewer::label_overlay`).
pub type Smooth = RefCell<LabelSmoother>;
pub type Sides = RefCell<HashMap<(u8, usize, usize, usize), f32>>;

/// The corner text of one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hud {
    /// The title's lines (the molecules: "2 × H₂O — water"); empty: no title.
    pub title: Vec<String>,
    /// Under the title, dim: the scene and the solver.
    pub subtitle: Option<String>,
    /// The time readout's lines.
    pub time: Vec<String>,
    /// The scale bar: its length in Å.
    pub scale: bool,
}

impl LabelState {
    /// Runs one pass of the capture's egui context for a `size`-pixel frame at `ppp` pixels per
    /// point and clock `time` (s), drawing with `draw`. The first pass of a new context is run
    /// twice (egui lays a new layer out over a sizing pass); the texture updates of both are kept.
    pub fn run(&mut self, size: (u32, u32), ppp: f32, time: f64, mut draw: impl FnMut(&egui::Context, &Smooth, &Sides)) -> (egui::Context, egui::FullOutput) {
        let LabelState { ctx, smooth, sides } = self;
        let fresh = ctx.is_none();
        let ctx = ctx.get_or_insert_with(|| {
            let c = egui::Context::default();
            crate::ui::apply_theme(&c);
            c
        });
        let raw = || {
            let mut raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(size.0 as f32 / ppp, size.1 as f32 / ppp))),
                time: Some(time),
                ..Default::default()
            };
            raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
            raw
        };
        let mut textures = egui::TexturesDelta::default();
        if fresh {
            let first = ctx.run_ui(raw(), |root| draw(root.ctx(), smooth, sides));
            textures.append(first.textures_delta);
        }
        let mut out = ctx.run_ui(raw(), |root| draw(root.ctx(), smooth, sides));
        textures.append(std::mem::take(&mut out.textures_delta));
        out.textures_delta = textures;
        (ctx.clone(), out)
    }
}

/// The title from the snapshot's molecules (two or more atoms, as the view labels them), counted:
/// "2 × H₂O — water".
pub fn title_lines(names: &[String]) -> Vec<String> {
    let mut seen: Vec<(String, usize)> = Vec::new();
    for n in names {
        match seen.iter_mut().find(|(m, _)| m == n) {
            Some(e) => e.1 += 1,
            None => seen.push((n.clone(), 1)),
        }
    }
    seen.into_iter().map(|(n, k)| if k > 1 { format!("{k} × {n}") } else { n }).collect()
}

/// Draws the title, the time readout and the scale bar (`scale`: length in Å and in points).
pub fn draw_hud(ctx: &egui::Context, hud: &Hud, scale: Option<(f64, f32)>) {
    let screen = ctx.content_rect();
    let painter = ctx.layer_painter(LayerId::new(Order::Foreground, egui::Id::new("capture-hud")));
    let margin = 16.0;
    let mut y = margin;
    let mut block: Vec<(egui::text::LayoutJob, f32)> = Vec::new();
    for (k, line) in hud.title.iter().enumerate() {
        let size = if k == 0 { 20.0 } else { 16.0 };
        block.push((crate::ui::rich(line, size, Color32::WHITE), 2.0));
    }
    if let Some(s) = &hud.subtitle {
        block.push((crate::ui::rich(s, 12.5, Color32::from_gray(185)), 6.0));
    }
    if !hud.time.is_empty() && !block.is_empty() {
        if let Some(last) = block.last_mut() {
            last.1 = 10.0;
        }
    }
    for line in &hud.time {
        block.push((crate::ui::rich(line, 13.5, Color32::from_rgb(205, 225, 245)), 2.0));
    }
    if !block.is_empty() {
        let galleys: Vec<_> = block.into_iter().map(|(job, gap)| (painter.layout_job(job), gap)).collect();
        let w = galleys.iter().map(|(g, _)| g.size().x).fold(0.0, f32::max);
        let h: f32 = galleys.iter().map(|(g, gap)| g.size().y + gap).sum();
        let rect = egui::Rect::from_min_size(pos2(margin, y), vec2(w, h)).expand(8.0);
        painter.rect_filled(rect, 6.0, Color32::from_black_alpha(150));
        for (g, gap) in galleys {
            let gh = g.size().y;
            painter.galley(pos2(margin, y), g, Color32::WHITE);
            y += gh + gap;
        }
    }
    if let Some((angstrom, len)) = scale.filter(|_| hud.scale) {
        let (x0, yb) = (margin + 4.0, screen.bottom() - margin - 4.0);
        let text = if angstrom >= 1.0 { format!("{angstrom:.0} Å") } else { format!("{angstrom} Å") };
        let galley = painter.layout_no_wrap(text, FontId::proportional(13.0), Color32::WHITE);
        let rect = egui::Rect::from_min_max(pos2(x0 - 8.0, yb - galley.size().y - 14.0), pos2(x0 + len.max(galley.size().x) + 8.0, yb + 6.0));
        painter.rect_filled(rect, 5.0, Color32::from_black_alpha(140));
        let stroke = egui::Stroke::new(2.0, Color32::WHITE);
        painter.line_segment([pos2(x0, yb), pos2(x0 + len, yb)], stroke);
        painter.line_segment([pos2(x0, yb - 5.0), pos2(x0, yb + 3.0)], stroke);
        painter.line_segment([pos2(x0 + len, yb - 5.0), pos2(x0 + len, yb + 3.0)], stroke);
        painter.galley(Align2::LEFT_BOTTOM.anchor_size(pos2(x0, yb - 6.0), galley.size()).min, galley, Color32::WHITE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_title_counts_repeated_molecules() {
        let n = |s: &str| s.to_string();
        assert_eq!(title_lines(&[n("H₂O — water"), n("H₂ — hydrogen"), n("H₂O — water")]), vec![n("2 × H₂O — water"), n("H₂ — hydrogen")]);
        assert!(title_lines(&[]).is_empty());
    }
}
