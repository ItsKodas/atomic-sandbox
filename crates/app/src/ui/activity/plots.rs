//! The training plots and the event timeline (activity spec §3.3.5, §3.3.6, §9): hand-painted like
//! the Graph tab's sparkline (egui `Painter`, no plot crate), from the views the worker built and
//! the app-side histories. Pure painters over cached vectors: nothing here asks for a solve.
//!
//! What the plots say, and do not say (physics honesty, spec §9):
//! - parity axes are (E − Σ E_ref)/N, the quantity the fit models and the energy tier bounds;
//! - the pair curve is the model's energy of an *isolated* dimer, not a pair potential; where the
//!   policy would refuse that dimer (`PairCurve::refused`, from `would_answer`) the curve is dashed
//!   and labelled with the rule — the dominant encoding, never read from the band;
//! - the band is ±1σ model (coefficient) uncertainty, what rule 7 judges, **not an error bar**
//!   against DFT; the grey floor is the fit's residual scatter s_E (teacher noise plus what this
//!   basis cannot fit; more samples of the same kind do not remove it, Task 4 review I1);
//! - rule 7 compares a width with the tier, so the comparison is drawn where widths live: a log
//!   lane of σ_epi/N ÷ tier and rule 6's ratio, both against 1.
use super::words;
use super::{ActivityState, Mark, BURST_BIN_K};
use crate::ui::logbook::clock;
use crate::ui::{theme, AMBER, GREEN, RED, SPARK_E, SPARK_T};
use egui::{pos2, vec2, Align2, Color32, FontId, Mesh, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, Ui};
use sim::activity::{PairCurve, SampleKind, SampleState, TeachRule, TrainingViews};

// ---- pure helpers ----

/// (log10 min, log10 max) over the positive finite values; `None` without one.
pub fn log_range(xs: &[f64]) -> Option<(f64, f64)> {
    let mut it = xs.iter().copied().filter(|x| x.is_finite() && *x > 0.0).map(f64::log10);
    let first = it.next()?;
    Some(it.fold((first, first), |(a, b), x| (a.min(x), b.max(x))))
}

/// Data (x, y) in ranges `xr`, `yr` to a point of `r` (y up).
pub fn to_screen(r: Rect, xr: (f64, f64), yr: (f64, f64), x: f64, y: f64) -> Pos2 {
    let fx = if xr.1 > xr.0 { (x - xr.0) / (xr.1 - xr.0) } else { 0.5 };
    let fy = if yr.1 > yr.0 { (y - yr.0) / (yr.1 - yr.0) } else { 0.5 };
    pos2(r.left() + fx as f32 * r.width(), r.bottom() - fy as f32 * r.height())
}

/// The ±σ band as a closed polygon: y + σ forward, y − σ backward.
pub fn band_polygon(r: Rect, xr: (f64, f64), yr: (f64, f64), x: &[f64], y: &[f64], s: &[f64]) -> Vec<Pos2> {
    let up = x.iter().zip(y).zip(s).map(|((&x, &y), &s)| to_screen(r, xr, yr, x, y + s));
    let down: Vec<Pos2> = x.iter().zip(y).zip(s).map(|((&x, &y), &s)| to_screen(r, xr, yr, x, y - s)).collect();
    up.chain(down.into_iter().rev()).collect()
}

/// The runs of points where the policy would refuse the dimer: half-open index ranges
/// [start, end) of one rule each (the curve's dashed, labelled stretches; plan review M5).
pub fn refused_stretches(refused: &[Option<TeachRule>]) -> Vec<(usize, usize, TeachRule)> {
    let mut v: Vec<(usize, usize, TeachRule)> = Vec::new();
    for (k, r) in refused.iter().enumerate() {
        match (r, v.last_mut()) {
            (Some(rule), Some(last)) if last.1 == k && last.2 == *rule => last.1 = k + 1,
            (Some(rule), _) => v.push((k, k + 1, *rule)),
            (None, _) => {}
        }
    }
    v
}

/// The curve's y range (Ha): the answered points and the dots, so a refused extrapolation (r
/// toward 0.5 Bohr) does not set the scale (plan review M5); every point when none of those
/// exists; `None` for an empty curve.
pub fn curve_y_range(c: &PairCurve) -> Option<(f64, f64)> {
    let answered = c.energy.iter().zip(&c.refused).filter(|(_, r)| r.is_none()).map(|(e, _)| *e);
    let mut ys: Vec<f64> = answered.chain(c.dots.iter().map(|d| d.1)).filter(|y| y.is_finite()).collect();
    if ys.is_empty() {
        ys = c.energy.iter().copied().filter(|y| y.is_finite()).collect();
    }
    let first = *ys.first()?;
    Some(ys.iter().fold((first, first), |(a, b), &y| (a.min(y), b.max(y))))
}

/// The uncertainty lane's energy series and floor as ratios to the rule-7 tier, for a dimer
/// (N = 2): σ_epi/N ÷ `energy_tier` per r, and (`dimer_floor`/N) ÷ tier, the fit's residual
/// scatter s_E on the same scale. Rule 6's series is `PairCurve::force_ratio` as it is. Both rules
/// refuse above 1.
pub fn lane_ratios(c: &PairCurve, dimer_floor: f64, energy_tier: f64) -> (Vec<f64>, f64) {
    const N: f64 = 2.0;
    (c.sigma.iter().map(|s| s / N / energy_tier).collect(), dimer_floor / N / energy_tier)
}

/// The index of the entry of `n` evenly spread points nearest to x-fraction `fx` (0..1).
fn nearest_index(n: usize, fx: f32) -> usize {
    if n <= 1 {
        0
    } else {
        ((fx.clamp(0.0, 1.0) * (n - 1) as f32).round() as usize).min(n - 1)
    }
}

/// A short number for an axis end: "2.1e-3", "12.5", "0.045".
fn axis(x: f64) -> String {
    let a = x.abs();
    if a == 0.0 {
        "0".into()
    } else if !(1e-2..1e4).contains(&a) {
        format!("{x:.1e}")
    } else if a >= 100.0 {
        format!("{x:.0}")
    } else if a >= 1.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.3}")
    }
}

// ---- painting kit ----

/// A plot frame: the allocated rect and its inner data rect (room for axis labels), the hover
/// response and a painter clipped to the frame.
fn plot_frame(ui: &mut Ui, h: f32) -> (Rect, Rect, Response, Painter) {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_black_alpha(70));
    let inner = Rect::from_min_max(rect.min + vec2(42.0, 6.0), rect.max - vec2(8.0, 14.0));
    (rect, inner, resp, p)
}

fn small(p: &Painter, at: Pos2, align: Align2, text: &str, colour: Color32) {
    p.text(at, align, text, FontId::monospace(9.0), colour);
}

fn marker(p: &Painter, at: Pos2, r: f32, colour: Color32, filled: bool) {
    if filled {
        p.circle_filled(at, r, colour);
    } else {
        p.circle_stroke(at, r, Stroke::new(1.0, colour));
    }
}

/// Fills the strip between two polylines (`up` and `down`, same length), segment by segment in
/// `colour_of(segment)`: a triangle mesh, since the strip of a curved band is not convex.
fn fill_strip(p: &Painter, up: &[Pos2], down: &[Pos2], colour_of: impl Fn(usize) -> Color32) {
    let mut mesh = Mesh::default();
    for k in 0..up.len().saturating_sub(1) {
        let c = colour_of(k);
        let i = mesh.vertices.len() as u32;
        for q in [up[k], up[k + 1], down[k + 1], down[k]] {
            mesh.colored_vertex(q, c);
        }
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i, i + 2, i + 3);
    }
    p.add(Shape::mesh(mesh));
}

fn kind_colour(k: SampleKind) -> Color32 {
    match k {
        SampleKind::Pass => theme::ACCENT,
        SampleKind::Fallback => AMBER,
        SampleKind::Live => SPARK_E,
    }
}

fn kind_word(k: SampleKind) -> &'static str {
    match k {
        SampleKind::Pass => "sampling pass",
        SampleKind::Fallback => "fallback (or audit)",
        SampleKind::Live => "live step",
    }
}

fn state_word(s: SampleState) -> &'static str {
    match s {
        SampleState::Fitted => "in the fit",
        SampleState::Unfitted => "newer than the fit",
        SampleState::Quarantined => "quarantined",
        SampleState::Superseded => "superseded",
        SampleState::BoxExcluded => "box-excluded",
        SampleState::HeldOut => "held out (draft guess)",
    }
}

fn caption(ui: &mut Ui, text: impl Into<String>) {
    ui.add(egui::Label::new(egui::RichText::new(text.into()).size(10.5).color(theme::INK_FAINT)).wrap());
}

// ---- rmse per refit ----

/// rmse per refit (spec §3.3.5): two series on a log y, one point per refit of this session.
pub fn rmse(ui: &mut Ui, st: &ActivityState) {
    theme::section(ui, "rmse per refit");
    let pts = &st.fit_points;
    let (_, inner, resp, p) = plot_frame(ui, 110.0);
    let es: Vec<f64> = pts.iter().map(|f| f.rmse_e * 1e3).collect();
    let fs: Vec<f64> = pts.iter().map(|f| f.rmse_f * 1e3).collect();
    let all: Vec<f64> = es.iter().chain(&fs).copied().collect();
    let Some((lo, hi)) = log_range(&all) else {
        p.text(inner.center(), Align2::CENTER_CENTER, "a point appears here after each refit", FontId::proportional(11.0), theme::INK_FAINT);
        caption(ui, "one point per refit this session; filled: trusted at the refit (a later failed audit does not change it); a loaded model starts with its file's fit");
        return;
    };
    let yr = (lo - 0.1, hi + 0.1);
    let xr = (0.0, (pts.len().max(2) - 1) as f64);
    let at = |k: usize, v: f64| to_screen(inner, xr, yr, k as f64, v.max(1e-300).log10());
    let data = p.with_clip_rect(inner);
    for (series, colour) in [(&es, SPARK_E), (&fs, SPARK_T)] {
        let line: Vec<Pos2> = series.iter().enumerate().filter(|(_, v)| **v > 0.0 && v.is_finite()).map(|(k, v)| at(k, *v)).collect();
        if line.len() >= 2 {
            data.add(Shape::line(line, Stroke::new(1.3, colour)));
        }
        for (k, v) in series.iter().enumerate().filter(|(_, v)| **v > 0.0 && v.is_finite()) {
            marker(&data, at(k, *v), 3.0, colour, pts[k].trusted);
        }
    }
    small(&p, pos2(resp.rect.left() + 3.0, inner.top()), Align2::LEFT_TOP, &axis(10f64.powf(yr.1)), theme::INK_FAINT);
    small(&p, pos2(resp.rect.left() + 3.0, inner.bottom()), Align2::LEFT_BOTTOM, &axis(10f64.powf(yr.0)), theme::INK_FAINT);
    small(&p, pos2(inner.left(), resp.rect.bottom() - 1.0), Align2::LEFT_BOTTOM, "refit 1", theme::INK_FAINT);
    small(&p, pos2(inner.right(), resp.rect.bottom() - 1.0), Align2::RIGHT_BOTTOM, &format!("{}", pts.len()), theme::INK_FAINT);
    if let Some(pos) = resp.hover_pos().filter(|q| inner.expand2(vec2(4.0, 6.0)).contains(*q)) {
        let k = nearest_index(pts.len(), (pos.x - inner.left()) / inner.width());
        let f = &pts[k];
        resp.on_hover_ui_at_pointer(|ui| {
            ui.label(format!(
                "epoch {} · {} samples · E {:.3} mHa/atom · F {:.3} mHa/Bohr · {} at the refit",
                f.epoch,
                f.samples,
                f.rmse_e * 1e3,
                f.rmse_f * 1e3,
                if f.trusted { "trusted" } else { "not trusted" }
            ));
        });
    }
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("● E rmse, mHa/atom").size(10.5).color(SPARK_E));
        ui.label(egui::RichText::new("● F rmse, mHa/Bohr").size(10.5).color(SPARK_T));
    });
    caption(ui, "one point per refit this session; filled: trusted at the refit (a later failed audit does not change it); a loaded model starts with its file's fit");
}

// ---- parity ----

/// Model vs DFT for the training samples (spec §3.3.5, §9.1): energies per atom or force components.
pub fn parity(ui: &mut Ui, st: &mut ActivityState, v: &TrainingViews) {
    ui.horizontal(|ui| {
        theme::section(ui, "model vs DFT");
        if let Some(f) = theme::segmented(ui, st.parity_forces, &[(false, "E", "energy per atom: (E − Σ E_ref)/N"), (true, "F", "force components of the fitted samples")]) {
            st.parity_forces = f;
        }
    });
    let (_, inner, resp, p) = plot_frame(ui, 190.0);
    let data = p.with_clip_rect(inner);
    if st.parity_forces {
        let pts = &v.parity_force;
        let m = pts.iter().flat_map(|q| q.iter()).map(|x| (*x as f64).abs() * 1e3).fold(0.0f64, f64::max).max(1e-6);
        let r = (-m * 1.05, m * 1.05);
        data.line_segment([to_screen(inner, r, r, r.0, r.0), to_screen(inner, r, r, r.1, r.1)], Stroke::new(1.0, theme::INK_FAINT));
        // one dot per pixel cell at most: the 20 000-point cap collapses onto the plot's area
        let (w, h) = (inner.width().ceil().max(1.0) as usize, inner.height().ceil().max(1.0) as usize);
        let mut seen = vec![false; w * h];
        for q in pts {
            let at = to_screen(inner, r, r, q[0] as f64 * 1e3, q[1] as f64 * 1e3);
            let (cx, cy) = (((at.x - inner.left()) as isize).clamp(0, w as isize - 1) as usize, ((at.y - inner.top()) as isize).clamp(0, h as isize - 1) as usize);
            if !std::mem::replace(&mut seen[cy * w + cx], true) {
                data.rect_filled(Rect::from_center_size(at, vec2(1.8, 1.8)), 0.0, theme::ACCENT_INK);
            }
        }
        small(&p, pos2(resp.rect.left() + 3.0, inner.top()), Align2::LEFT_TOP, &axis(r.1), theme::INK_FAINT);
        small(&p, pos2(resp.rect.left() + 3.0, inner.bottom()), Align2::LEFT_BOTTOM, &axis(r.0), theme::INK_FAINT);
        small(&p, pos2(inner.center().x, resp.rect.bottom() - 1.0), Align2::CENTER_BOTTOM, "DFT, mHa/Bohr →", theme::INK_FAINT);
        if v.force_points_total > pts.len() {
            theme::note(ui, format!("showing {} of {} components (whole atoms, thinned evenly)", pts.len(), v.force_points_total));
        }
    } else {
        let pts = &v.parity_energy;
        let vals: Vec<f64> = pts.iter().flat_map(|q| [q.dft * 1e3, q.model * 1e3]).filter(|x| x.is_finite()).collect();
        if vals.is_empty() {
            p.text(inner.center(), Align2::CENTER_CENTER, "no samples with a prediction yet", FontId::proportional(11.0), theme::INK_FAINT);
        } else {
            let (lo, hi) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
            let pad = ((hi - lo) * 0.06).max(1e-3);
            let r = (lo - pad, hi + pad);
            let tier = v.energy_tier * 1e3;
            // the energy tier around y = x: the band rule 7's per-atom limit draws
            let xs = [r.0, r.1];
            let poly = band_polygon(inner, r, r, &xs, &xs, &[tier, tier]);
            fill_strip(&data, &poly[..2], &[poly[3], poly[2]], |_| theme::ACCENT.gamma_multiply(0.14));
            data.line_segment([to_screen(inner, r, r, r.0, r.0), to_screen(inner, r, r, r.1, r.1)], Stroke::new(1.0, theme::INK_FAINT));
            for q in pts.iter().filter(|q| q.dft.is_finite() && q.model.is_finite()) {
                marker(&data, to_screen(inner, r, r, q.dft * 1e3, q.model * 1e3), 2.6, kind_colour(q.kind), q.state == SampleState::Fitted);
            }
            small(&p, pos2(resp.rect.left() + 3.0, inner.top()), Align2::LEFT_TOP, &axis(r.1), theme::INK_FAINT);
            small(&p, pos2(resp.rect.left() + 3.0, inner.bottom()), Align2::LEFT_BOTTOM, &axis(r.0), theme::INK_FAINT);
            small(&p, pos2(inner.center().x, resp.rect.bottom() - 1.0), Align2::CENTER_BOTTOM, "DFT (E − Σ E_ref)/N, mHa/atom →", theme::INK_FAINT);
            if let Some(pos) = resp.hover_pos().filter(|q| inner.contains(*q)) {
                let best = pts
                    .iter()
                    .filter(|q| q.dft.is_finite() && q.model.is_finite())
                    .map(|q| (q, to_screen(inner, r, r, q.dft * 1e3, q.model * 1e3).distance(pos)))
                    .filter(|(_, d)| *d < 10.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((q, _)) = best {
                    resp.on_hover_ui_at_pointer(|ui| {
                        ui.label(format!("sample {} · {} · {} · DFT {:.3} · model {:.3} mHa/atom", q.index, kind_word(q.kind), state_word(q.state), q.dft * 1e3, q.model * 1e3));
                    });
                }
            }
        }
    }
    if !st.parity_forces {
        ui.horizontal_wrapped(|ui| {
            for k in [SampleKind::Pass, SampleKind::Fallback, SampleKind::Live] {
                ui.label(egui::RichText::new(format!("● {}", kind_word(k))).size(10.5).color(kind_colour(k)));
            }
        });
        caption(ui, "filled: in the fit; hollow: quarantined, superseded, box-excluded or newer than the fit; the violet strip is the energy tier around y = x");
    }
    let omitted = if v.parity_omitted > 0 { format!("; {} samples omitted (no reference or no prediction)", v.parity_omitted) } else { String::new() };
    caption(
        ui,
        format!("model vs DFT for the training samples, (E − Σ E_ref)/N with the fit's own reference energies; audits are counted as fallbacks until the model library (sidecar v4) tags them{omitted}"),
    );
}

// ---- pair curves ----

fn pair_title(z: (u8, u8)) -> String {
    format!("{}–{}", units::isotopes::symbol(z.0), units::isotopes::symbol(z.1))
}

/// One species pair: the model's energy of an isolated dimer with its ±σ band, the dots of the
/// stored two-atom samples and the rug of the pair distances, dashed and labelled where the policy
/// would refuse the dimer; under it the uncertainty lane.
pub fn pair_curve(ui: &mut Ui, c: &PairCurve, v: &TrainingViews, hist: Option<&sim::activity::PairHistogram>, draft: bool, mismatch: bool) {
    theme::section(ui, &format!("{} curve", pair_title(c.z)));
    let n = c.r.len().min(c.energy.len());
    let (_, inner, resp, p) = plot_frame(ui, 150.0);
    let (Some((ylo, yhi)), true) = (curve_y_range(c), n >= 2) else {
        p.text(inner.center(), Align2::CENTER_CENTER, "no curve", FontId::proportional(11.0), theme::INK_FAINT);
        return;
    };
    let (ylo, yhi) = (ylo * 1e3, yhi * 1e3);
    let pad = ((yhi - ylo) * 0.10).max(1e-3);
    let yr = (ylo - pad, yhi + pad);
    let xr = (c.r[0], c.r[n - 1]);
    let pt = |k: usize, e: f64| to_screen(inner, xr, yr, c.r[k], e * 1e3);
    let data = p.with_clip_rect(inner);
    let refused = |k: usize| c.refused.get(k).copied().flatten();
    let has_band = c.sigma.len() >= n;
    // the energy origin: the dissociated pair
    data.line_segment([to_screen(inner, xr, yr, xr.0, 0.0), to_screen(inner, xr, yr, xr.1, 0.0)], Stroke::new(1.0, Color32::from_white_alpha(18)));
    // the fit's residual scatter around the curve (grey), then ±σ_epi (accent, a third where refused)
    if v.dimer_floor > 0.0 {
        let up: Vec<Pos2> = (0..n).map(|k| pt(k, c.energy[k] + v.dimer_floor)).collect();
        let dn: Vec<Pos2> = (0..n).map(|k| pt(k, c.energy[k] - v.dimer_floor)).collect();
        fill_strip(&data, &up, &dn, |_| Color32::from_white_alpha(22));
    }
    if has_band {
        let up: Vec<Pos2> = (0..n).map(|k| pt(k, c.energy[k] + c.sigma[k])).collect();
        let dn: Vec<Pos2> = (0..n).map(|k| pt(k, c.energy[k] - c.sigma[k])).collect();
        fill_strip(&data, &up, &dn, |k| theme::ACCENT.gamma_multiply(if refused(k).is_some() || refused(k + 1).is_some() { 0.25 / 3.0 } else { 0.25 }));
    }
    // the curve: solid where the model would answer, dashed where it would not
    let stretches = refused_stretches(&c.refused[..n.min(c.refused.len())]);
    let stroke = Stroke::new(1.5, theme::ACCENT);
    let mut from = 0;
    for &(a, b, _) in &stretches {
        // a run touches the next stretch's first point so the line has no gap
        if a > from {
            data.add(Shape::line((from..=a).map(|k| pt(k, c.energy[k])).collect::<Vec<_>>(), stroke));
        }
        let dashed: Vec<Pos2> = (a.saturating_sub(1)..b.min(n)).map(|k| pt(k, c.energy[k])).collect();
        data.extend(Shape::dashed_line(&dashed, Stroke::new(1.5, theme::ACCENT.gamma_multiply(0.7)), 4.0, 3.0));
        from = b.min(n).saturating_sub(1);
    }
    if from < n - 1 || stretches.is_empty() {
        data.add(Shape::line((from..n).map(|k| pt(k, c.energy[k])).collect::<Vec<_>>(), stroke));
    }
    // labels start right of the axis gutter and a label that would run into the previous one is dropped
    let mut label_end = inner.left();
    for &(a, b, rule) in &stretches {
        let label = words::rule_label(rule, None);
        let w = 5.6 * label.chars().count() as f32;
        let x = ((pt(a, 0.0).x + pt(b.min(n) - 1, 0.0).x) / 2.0).max(inner.left() + w / 2.0 + 2.0);
        if b - a >= 3 && x - w / 2.0 > label_end && x + w / 2.0 < inner.right() {
            small(&data, pos2(x, inner.top() + 2.0), Align2::CENTER_TOP, &label, theme::INK_DIM);
            label_end = x + w / 2.0 + 4.0;
        }
    }
    // the dots (stored two-atom samples) and the rug (pair distances in the fit)
    for &(r, e, s) in &c.dots {
        marker(&data, to_screen(inner, xr, yr, r, e * 1e3), 3.0, GREEN, s == SampleState::Fitted);
    }
    if let Some(h) = hist {
        for (k, &cnt) in h.counts.iter().enumerate().filter(|(_, c)| **c > 0) {
            let r = (k as f64 + 0.5) * h.bin;
            if r >= xr.0 && r <= xr.1 {
                let x = to_screen(inner, xr, yr, r, 0.0).x;
                let len = 2.0 + (cnt as f32).log2().max(0.0).min(2.0);
                data.line_segment([pos2(x, inner.bottom()), pos2(x, inner.bottom() - len)], Stroke::new(1.0, theme::INK_DIM));
            }
        }
    }
    small(&p, pos2(resp.rect.left() + 3.0, inner.top()), Align2::LEFT_TOP, &axis(yr.1), theme::INK_FAINT);
    small(&p, pos2(resp.rect.left() + 3.0, inner.bottom()), Align2::LEFT_BOTTOM, &axis(yr.0), theme::INK_FAINT);
    small(&p, pos2(inner.left(), resp.rect.bottom() - 1.0), Align2::LEFT_BOTTOM, &format!("{:.1} Bohr", xr.0), theme::INK_FAINT);
    small(&p, pos2(inner.right(), resp.rect.bottom() - 1.0), Align2::RIGHT_BOTTOM, &format!("{:.1} Bohr   E − E_ref(a) − E_ref(b), mHa", xr.1), theme::INK_FAINT);
    let k_at = |pos: Pos2| nearest_index(n, (pos.x - inner.left()) / inner.width());
    if let Some(pos) = resp.hover_pos().filter(|q| inner.contains(*q)) {
        let k = k_at(pos);
        let rule = refused(k);
        resp.on_hover_ui_at_pointer(|ui| {
            ui.set_max_width(360.0);
            ui.label(format!("r = {:.2} Bohr ({:.2} Å) · E {:+.2} mHa", c.r[k], units::bohr_to_angstrom(c.r[k]), c.energy[k] * 1e3));
            if has_band {
                ui.label(format!("±{:.2e} mHa: model (coefficient) uncertainty, what rule 7 judges; not an error bar against DFT", c.sigma[k] * 1e3));
            }
            ui.label(format!("grey ±{:.2e} mHa: this dimer's floor (2·s_E for a model fitted per atom, s_E per sample), s_E being the fit's residual scatter (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it)", v.dimer_floor * 1e3));
            if let Some(rule) = rule {
                ui.label(format!("the model won't answer this dimer here: {}", words::rule_sentence(rule, None, words::RowCtx::default())));
            }
        });
    }
    // the uncertainty lane
    if has_band && c.force_ratio.len() >= n && v.energy_tier > 0.0 {
        let (rule7, floor) = lane_ratios(c, v.dimer_floor, v.energy_tier);
        let (_, li, lresp, lp) = plot_frame(ui, 64.0);
        let ld = lp.with_clip_rect(li);
        let mut all: Vec<f64> = rule7.iter().chain(c.force_ratio.iter()).copied().collect();
        all.extend([1.0, floor]);
        let (lo, hi) = log_range(&all).unwrap_or((-1.0, 1.0));
        let yr = (lo.min(-0.3) - 0.1, hi.max(0.3) + 0.1);
        let at = |k: usize, v: f64| to_screen(li, xr, yr, c.r[k], v.max(1e-300).log10());
        let one = to_screen(li, xr, yr, xr.0, 0.0).y;
        ld.extend(Shape::dashed_line(&[pos2(li.left(), one), pos2(li.right(), one)], Stroke::new(1.0, Color32::from_white_alpha(120)), 3.0, 3.0));
        if floor > 0.0 {
            let fy = to_screen(li, xr, yr, xr.0, floor.log10()).y;
            ld.extend(Shape::dashed_line(&[pos2(li.left(), fy), pos2(li.right(), fy)], Stroke::new(1.0, theme::INK_FAINT), 2.0, 3.0));
        }
        let series = |vals: &[f64], colour: Color32| {
            let line: Vec<Pos2> = (0..n).filter(|&k| vals[k] > 0.0 && vals[k].is_finite()).map(|k| at(k, vals[k])).collect();
            if line.len() >= 2 {
                ld.add(Shape::line(line, Stroke::new(1.3, colour)));
            }
        };
        series(&rule7, theme::ACCENT);
        series(&c.force_ratio[..n], AMBER);
        small(&lp, pos2(lresp.rect.left() + 3.0, li.top()), Align2::LEFT_TOP, &axis(10f64.powf(yr.1)), theme::INK_FAINT);
        small(&lp, pos2(lresp.rect.left() + 3.0, li.bottom()), Align2::LEFT_BOTTOM, &axis(10f64.powf(yr.0)), theme::INK_FAINT);
        if let Some(pos) = lresp.hover_pos().filter(|q| li.contains(*q)) {
            let k = k_at(pos).min(n - 1);
            lresp.on_hover_ui_at_pointer(|ui| {
                ui.label(format!("r = {:.2} Bohr · σ_epi/N = {:.2e} mHa/atom (rule 7: {:.2}× its tier) · rule 6 ratio {:.2}", c.r[k], c.sigma[k] / 2.0 * 1e3, rule7[k], c.force_ratio[k]));
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("— rule 7 (energy)").size(10.5).color(theme::ACCENT));
            ui.label(egui::RichText::new("— rule 6 (forces)").size(10.5).color(AMBER));
            ui.label(egui::RichText::new("- - dimer floor per atom").size(10.5).color(theme::INK_DIM)).on_hover_text("this dimer's floor (2·s_E for a model fitted per atom, s_E per sample) divided by its two atoms, as a share of the rule-7 tier; s_E is the fit's residual scatter (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it)");
            ui.label(egui::RichText::new("- - 1: a rule refuses above it").size(10.5).color(Color32::from_white_alpha(150)));
            ui.label(egui::RichText::new("σ_epi/N ÷ tier and rule 6's ratio, log scale").size(10.5).color(theme::INK_FAINT));
        });
    }
    let mut cap = String::from("The model's energy of an isolated A–B pair. It is not a pair potential: in a molecule the model has no per-pair energy, and inside a molecule it may answer this pair where the curve is dashed and refuse it where it is solid. The band, the lane and the dashed stretches describe the isolated pair. Band: ±1σ model (coefficient) uncertainty, what rule 7 judges — not an error bar against DFT. Grey: this dimer's floor (2·s_E for a model fitted per atom, s_E per sample), s_E being the fit's residual scatter (teacher noise plus what this basis cannot fit; more samples of the same kind do not remove it). Dashed: where the model would not answer this dimer (rules 4–7, the non-finite check and the fence, with a fresh uncertainty check; rule 0, audits and the check's reuse between steps aside).");
    if !has_band {
        cap += " This model has no posterior, so there is no band and no lane.";
    }
    if mismatch {
        cap += " The model was taught at other settings; in this scene it answers nothing (rule 3).";
    }
    if draft {
        cap += " Draft mode is on: the model answers wherever rules 1–3 pass; dashed shows where it would fall back outside draft mode.";
    }
    caption(ui, cap);
}

// ---- coverage ----

/// A row of bars over `counts` (bin `bin` wide, starting at 0), with an optional dashed marker at
/// `mark` (same unit as the bins' edges) and the bar under the pointer in a tooltip.
fn bars(ui: &mut Ui, counts: &[u32], bin: f64, mark: Option<f64>, colour: Color32, unit: &str) {
    let (_, inner, resp, p) = plot_frame(ui, 70.0);
    let max = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let nb = counts.len().max(1);
    let xmax = (nb as f64 * bin).max(mark.map_or(0.0, |m| m + bin));
    let w = inner.width() * (nb as f64 * bin / xmax) as f32 / nb as f32;
    for (k, &c) in counts.iter().enumerate().filter(|(_, c)| **c > 0) {
        let x = inner.left() + inner.width() * (k as f64 * bin / xmax) as f32;
        let h = inner.height() * c as f32 / max;
        p.rect_filled(Rect::from_min_max(pos2(x, inner.bottom() - h), pos2(x + (w - 0.5).max(0.5), inner.bottom())), 0.0, colour.gamma_multiply(0.8));
    }
    if let Some(m) = mark {
        let x = inner.left() + inner.width() * (m / xmax) as f32;
        p.extend(Shape::dashed_line(&[pos2(x, inner.top()), pos2(x, inner.bottom())], Stroke::new(1.0, Color32::from_white_alpha(170)), 3.0, 3.0));
    }
    small(&p, pos2(inner.left(), resp.rect.bottom() - 1.0), Align2::LEFT_BOTTOM, &format!("0 {unit}"), theme::INK_FAINT);
    small(&p, pos2(inner.right(), resp.rect.bottom() - 1.0), Align2::RIGHT_BOTTOM, &format!("{xmax:.0} {unit}"), theme::INK_FAINT);
    small(&p, pos2(resp.rect.left() + 3.0, inner.top()), Align2::LEFT_TOP, &format!("{}", max as u32), theme::INK_FAINT);
    if let Some(pos) = resp.hover_pos().filter(|q| inner.contains(*q)) {
        let k = (((pos.x - inner.left()) / inner.width()) as f64 * xmax / bin) as usize;
        if let Some(c) = counts.get(k) {
            resp.on_hover_text(format!("{:.1}–{:.1} {unit}: {c}", k as f64 * bin, (k + 1) as f64 * bin));
        }
    }
}

// ---- the training group ----

/// The Training section's content (spec §3.3.5): rmse per refit, the parity plot, one curve per
/// species pair, the coverage histograms and the age line. `target_t` is the plan's hot-burst
/// temperature (K) for the burst histogram, `draft` is draft mode (the pair-curve caption).
pub fn training(ui: &mut Ui, st: &mut ActivityState, target_t: Option<f64>, draft: bool) {
    rmse(ui, st);
    let Some(views) = st.views.clone() else {
        theme::note(ui, "no training views yet: they are built from a fitted model a moment after this section opens");
        return;
    };
    ui.add_space(4.0);
    parity(ui, st, &views);
    let mismatch = st.provenance_mismatch;
    for c in &views.curves {
        ui.add_space(4.0);
        let hist = views.pairs.iter().find(|h| h.z == c.z);
        pair_curve(ui, c, &views, hist, draft, mismatch);
    }
    if views.curves_skipped > 0 {
        theme::note(ui, format!("{} species pairs skipped: a prediction on their grid failed", views.curves_skipped));
    }
    ui.add_space(4.0);
    theme::section(ui, "coverage");
    for h in &views.pairs {
        theme::note(ui, format!("{} pair distances", pair_title(h.z)));
        bars(ui, &h.counts, h.bin, None, theme::ACCENT, "Bohr");
    }
    caption(ui, "pair distances in the current fit");
    theme::note(ui, "hot-burst temperatures solved this session");
    if st.burst_hist.iter().all(|&c| c == 0) {
        theme::note(ui, "no hot-burst geometry has been solved this session");
    } else {
        bars(ui, &st.burst_hist, BURST_BIN_K, target_t, AMBER, "K");
    }
    caption(
        ui,
        format!(
            "instantaneous kinetic temperature (3N − 3 dof) of each hot-burst geometry solved this session, from its Verlet velocities before the thermostat. Consecutive points of one burst are correlated (CSVR τ = {tau} fs at dt = {dt} fs), and for two atoms the instantaneous T alone spreads by about 82 % (√(2/3)). A loaded model's earlier bursts are unrecorded.",
            tau = providers::sampling::HOT_BURST_TAU_FS,
            dt = providers::sampling::HOT_BURST_DT_FS
        ),
    );
    // the age of what is on screen, and which model it describes (Task 4 review I2)
    let age = st.views_at.map_or(String::new(), |t| format!("{:.0} s ago", t.elapsed().as_secs_f64()));
    caption(ui, format!("built {age} for model generation {} (surface epoch {}), {:.1} ms", views.model_gen, views.epoch, views.build_ms));
}

// ---- the timeline ----

/// The event timeline (spec §3.3.6): five lanes over session time.
pub fn timeline(ui: &mut Ui, st: &ActivityState) {
    const LANES: [&str; 5] = ["fallbacks", "audits", "refits", "quarantines", "passes"];
    let lane_h = 15.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), lane_h * 5.0 + 14.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_black_alpha(70));
    let inner = Rect::from_min_max(rect.min + vec2(70.0, 3.0), rect.max - vec2(8.0, 14.0));
    for (i, name) in LANES.iter().enumerate() {
        let y = inner.top() + lane_h * i as f32;
        small(&p, pos2(rect.left() + 4.0, y + lane_h / 2.0), Align2::LEFT_CENTER, name, theme::INK_FAINT);
        p.line_segment([pos2(inner.left(), y + lane_h), pos2(inner.right(), y + lane_h)], Stroke::new(1.0, theme::LINE_SOFT));
    }
    let start_of = |m: &Mark| match *m {
        Mark::Fallback(t) | Mark::Audit(t, _) | Mark::Refit(t, _) | Mark::Quarantine(t, _) | Mark::Pass(t, _, _) => t,
    };
    let Some(t0) = st.timeline.iter().map(start_of).min() else {
        p.text(inner.center(), Align2::CENTER_CENTER, "events appear here as they happen", FontId::proportional(11.0), theme::INK_FAINT);
        return;
    };
    let now = st.clock_now();
    let t1 = now.max(t0 + std::time::Duration::from_secs(1));
    let span = (t1 - t0).as_secs_f32();
    let x_of = |t: std::time::Duration| inner.left() + inner.width() * (t.saturating_sub(t0).as_secs_f32() / span).clamp(0.0, 1.0);
    let lane_mid = |i: usize| inner.top() + lane_h * i as f32 + lane_h / 2.0;
    // fallbacks: one tick per occupied pixel column, taller with the count
    let cols = inner.width().ceil().max(1.0) as usize;
    let mut fall = vec![0u32; cols];
    for m in &st.timeline {
        if let Mark::Fallback(t) = m {
            let c = (((x_of(*t) - inner.left()) as usize).min(cols - 1)).min(cols - 1);
            fall[c] += 1;
        }
    }
    for (c, &n) in fall.iter().enumerate().filter(|(_, n)| **n > 0) {
        let h = (3.0 + 2.0 * (n as f32).log2()).min(lane_h - 2.0);
        let x = inner.left() + c as f32 + 0.5;
        p.line_segment([pos2(x, lane_mid(0) + h / 2.0), pos2(x, lane_mid(0) - h / 2.0)], Stroke::new(1.0, AMBER));
    }
    let mut spots: Vec<(Pos2, String)> = Vec::new();
    for m in &st.timeline {
        match *m {
            Mark::Fallback(_) => {}
            Mark::Audit(t, v) => {
                let (c, w) = match v {
                    sim::AuditVerdict::Passed => (GREEN, "passed"),
                    sim::AuditVerdict::Failed => (RED, "failed"),
                    sim::AuditVerdict::BoxAffected => (theme::INK_DIM, "box-affected, not judged"),
                };
                let at = pos2(x_of(t), lane_mid(1));
                p.circle_filled(at, 3.0, c);
                spots.push((at, format!("audit {w} at {}", clock(t))));
            }
            Mark::Refit(t, trusted) => {
                let at = pos2(x_of(t), lane_mid(2));
                let d = [pos2(at.x, at.y - 4.0), pos2(at.x + 4.0, at.y), pos2(at.x, at.y + 4.0), pos2(at.x - 4.0, at.y)];
                if trusted {
                    p.add(Shape::convex_polygon(d.to_vec(), SPARK_E, Stroke::NONE));
                } else {
                    p.add(Shape::closed_line(d.to_vec(), Stroke::new(1.0, SPARK_E)));
                }
                spots.push((at, format!("refit at {}: {} afterwards", clock(t), if trusted { "trusted" } else { "not trusted" })));
            }
            Mark::Quarantine(t, n) => {
                let at = pos2(x_of(t), lane_mid(3));
                p.rect_filled(Rect::from_center_size(at, vec2(7.0, 7.0)), 1.0, Color32::from_rgb(190, 130, 240));
                spots.push((at, format!("{n} samples quarantined at {}", clock(t))));
            }
            Mark::Pass(s, e, cancelled) => {
                let (x0, x1) = (x_of(s), x_of(e.unwrap_or(now)).max(x_of(s) + 2.0));
                let r = Rect::from_min_max(pos2(x0, lane_mid(4) - 4.0), pos2(x1, lane_mid(4) + 4.0));
                if cancelled {
                    p.rect_stroke(r, 1.0, Stroke::new(1.0, theme::ACCENT), egui::StrokeKind::Inside);
                    let mut x = r.left();
                    while x < r.right() {
                        p.line_segment([pos2(x, r.bottom()), pos2((x + 4.0).min(r.right()), r.top())], Stroke::new(1.0, theme::ACCENT));
                        x += 4.0;
                    }
                } else {
                    p.rect_filled(r, 1.0, theme::ACCENT);
                }
                let what = match e {
                    None => "running".to_string(),
                    Some(e) => format!("to {}{}", clock(e), if cancelled { ", cancelled" } else { "" }),
                };
                spots.push((r.center(), format!("sampling pass from {} {what}", clock(s))));
            }
        }
    }
    small(&p, pos2(inner.left(), rect.bottom() - 1.0), Align2::LEFT_BOTTOM, &clock(t0), theme::INK_FAINT);
    small(&p, pos2(inner.right(), rect.bottom() - 1.0), Align2::RIGHT_BOTTOM, &clock(t1), theme::INK_FAINT);
    if let Some(pos) = resp.hover_pos().filter(|q| inner.expand(3.0).contains(*q)) {
        let lane = (((pos.y - inner.top()) / lane_h) as usize).min(4);
        if lane == 0 {
            let c = ((pos.x - inner.left()) as usize).min(cols - 1);
            if fall[c] > 0 {
                resp.on_hover_text(format!("{} fallback{} here", fall[c], if fall[c] == 1 { "" } else { "s" }));
            }
        } else if let Some((_, text)) = spots.iter().filter(|(at, _)| (at.x - pos.x).abs() < 7.0 && (at.y - pos.y).abs() < lane_h / 2.0 + 1.0).min_by(|a, b| (a.0.x - pos.x).abs().total_cmp(&(b.0.x - pos.x).abs())) {
            resp.on_hover_text(text.clone());
        }
    }
}
