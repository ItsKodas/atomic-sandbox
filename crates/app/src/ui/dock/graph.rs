//! The Graph tab: E total (coloured by who answered each step) and T over the history, large,
//! with their ranges, the time span and the source legend (rows 68–70 of the mapping); the
//! Readouts tab draws the same sparkline small.
use super::super::{source_colour, source_label, theme, GraphSeries, PanelInput, UiState, AMBER, RED, SPARK_E, SPARK_T, SPARK_WINDOW};
use crate::sim::Backend;
use egui::{Color32, Ui};
use sim::Source;

pub fn show(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    super::group(ui, st, "d-graph", |ui, st| {
        ui.horizontal_wrapped(|ui| {
            if let Some(s) = theme::segmented(
                ui,
                st.graph_series,
                &[(GraphSeries::Both, "E total + T", ""), (GraphSeries::Energy, "E total", ""), (GraphSeries::Temperature, "T", "")],
            ) {
                st.graph_series = s;
            }
            if let Some(a) = theme::segmented(ui, st.graph_all, &[(false, "last 400 points", ""), (true, "all ≤ 2000", "")]) {
                st.graph_all = a;
            }
        });
        let Some(s) = input.snapshot else {
            theme::note(ui, "no solution yet");
            return;
        };
        let d = &s.dynamics;
        let window = if st.graph_all { usize::MAX } else { SPARK_WINDOW };
        let estimated = s.backend == Backend::Estimated;
        sparkline(ui, &d.history, &d.history_source, &d.history_guess, 170.0, window, st.graph_series, true, estimated, st.temp_units);
        theme::note(
            ui,
            "Each curve is scaled to its own range over the window shown. E total = KE + E_KS: it moves when tools or the thermostat act. The drift readout uses E conserved.",
        );
    });
}

/// E_total and T over the newest `window` history points, each scaled to its own range:
/// `Painter::line_segment` polylines, no plot crate. Each E_total segment takes the colour of the
/// source of the step it ends on (`source_colour`: surrogate green, fallback amber, DFT blue), and
/// fallback steps get a dot, so a single DFT step among a thousand surrogate ones is visible; a
/// draft-mode guess (`guesses`, parallel to `history`) is drawn red whoever answered.
/// `details` adds the ranges and the time span below, `legend` the source key.
#[allow(clippy::too_many_arguments)]
pub fn sparkline(
    ui: &mut Ui,
    history: &[(f64, f64, f64)],
    sources: &[Source],
    guesses: &[bool],
    height: f32,
    window: usize,
    series: GraphSeries,
    details: bool,
    legend: bool,
    temp_units: super::super::TempUnits,
) {
    let width = ui.available_width().max(120.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_black_alpha(70));
    if history.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "E total and T appear here once time runs",
            egui::FontId::proportional(11.0),
            ui.visuals().weak_text_color(),
        );
        return;
    }
    let start = history.len().saturating_sub(window);
    let pts: Vec<&(f64, f64, f64)> = history[start..].iter().collect();
    // parallel to `history` (a snapshot without sources draws everything as DFT)
    let src = |i: usize| sources.get(start + i).copied().unwrap_or(Source::Dft);
    let guess = |i: usize| guesses.get(start + i).copied().unwrap_or(false);
    let any_guess = (0..pts.len()).any(guess);
    let (t0, t1) = (pts[0].0, pts[pts.len() - 1].0);
    let range = |f: &dyn Fn(&(f64, f64, f64)) -> f64| {
        pts.iter()
            .map(|p| f(p))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(x), hi.max(x)))
    };
    let inner = rect.shrink(4.0);
    if height > 100.0 {
        // quarter grid lines on the big graph
        for k in 1..4 {
            let y = inner.top() + inner.height() * k as f32 / 4.0;
            let x = inner.left() + inner.width() * k as f32 / 4.0;
            let g = egui::Stroke::new(1.0, Color32::from_white_alpha(10));
            painter.line_segment([egui::pos2(inner.left(), y), egui::pos2(inner.right(), y)], g);
            painter.line_segment([egui::pos2(x, inner.top()), egui::pos2(x, inner.bottom())], g);
        }
    }
    let series_of = |f: &dyn Fn(&(f64, f64, f64)) -> f64, colour: Option<Color32>, draw: bool| -> (f64, f64) {
        let (lo, hi) = range(f);
        if !draw {
            return (lo, hi);
        }
        let span = if hi > lo { hi - lo } else { 1.0 };
        let to = |p: &(f64, f64, f64)| {
            let x = if t1 > t0 { (p.0 - t0) / (t1 - t0) } else { 0.0 };
            let y = if hi > lo { (f(p) - lo) / span } else { 0.5 };
            egui::pos2(inner.left() + x as f32 * inner.width(), inner.bottom() - y as f32 * inner.height())
        };
        let w = if height > 100.0 { 1.6 } else { 1.2 };
        for (i, seg) in pts.windows(2).enumerate() {
            let c = colour.unwrap_or_else(|| if guess(i + 1) { RED } else { source_colour(src(i + 1)) });
            painter.line_segment([to(seg[0]), to(seg[1])], egui::Stroke::new(w, c));
        }
        if colour.is_none() {
            for (i, p) in pts.iter().enumerate() {
                if src(i) == Source::Fallback {
                    painter.circle_filled(to(p), 2.2, AMBER);
                }
            }
        }
        (lo, hi)
    };
    // T under E, so the source colours stay on top
    let (tlo, thi) = series_of(&|p| p.2, Some(SPARK_T), series != GraphSeries::Energy);
    let (elo, ehi) = series_of(&|p| p.1, None, series != GraphSeries::Temperature);
    response.on_hover_text(
        "each curve is scaled to its own range over the window shown; E total = KE + E_KS (it moves when tools or the thermostat act — the drift readout uses E conserved)",
    );
    if !details {
        return;
    }
    let span_fs = units::au_time_to_fs(t1 - t0);
    ui.horizontal_wrapped(|ui| {
        ui.colored_label(SPARK_E, format!("E total {:+.5} … {:+.5} Ha", elo, ehi));
        ui.colored_label(SPARK_T, format!("T {:.0} … {}", temp_units.from_k(tlo), temp_units.show(thi, 0)));
        ui.weak(format!("last {span_fs:.1} fs"));
    });
    if legend {
        ui.horizontal_wrapped(|ui| {
            ui.weak("E total by step:");
            for s in [Source::Surrogate, Source::Fallback, Source::Dft] {
                ui.colored_label(source_colour(s), source_label(s));
            }
            if any_guess {
                ui.colored_label(RED, "draft guess");
            }
        });
    }
}
