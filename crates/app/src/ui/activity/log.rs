//! The counters and the live solve log of the Activity tab (activity spec §3.3.2, §3.3.3): one
//! row per teaching solve, newest first, with the rule that asked for it in words and what became
//! of the answer. Drawn with `show_rows`, so only the visible rows are laid out.
use super::words::{self, RowCtx};
use super::{ActivityState, LogFilter, LogRow};
use crate::ui::logbook::clock;
use crate::ui::{theme, AMBER, GREEN, RED};
use egui::text::{LayoutJob, TextFormat, TextWrapping};
use egui::{pos2, vec2, Color32, FontId, Rect, RichText, Sense, Ui};
use sim::activity::{TeachOrigin, TeachOutcome};

/// The numbers the three counters show (a replay's `counters` expectation reads them here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterValues {
    /// Samples in the model: the store's length (`ModelStatus::samples`).
    pub samples: Option<usize>,
    /// Teacher solves for this model (a lower bound when `unrecorded_before`).
    pub teacher_solves: Option<u64>,
    /// Teaching solves this session.
    pub session: u64,
}

pub fn counter_values(st: &ActivityState, m: Option<&crate::sim::ModelStatus>) -> CounterValues {
    CounterValues {
        samples: m.map(|m| m.samples),
        teacher_solves: m.map(|m| if m.solves.unrecorded_before { m.solves.total().max(m.samples as u64) } else { m.solves.total() }),
        session: st.session_solves,
    }
}

/// The three counters (spec §3.3.2), ticking per solve: a dim name, then the number.
pub fn counters(ui: &mut Ui, st: &ActivityState, m: Option<&crate::sim::ModelStatus>) {
    let line = |ui: &mut Ui, name: &str, value: String, extra: Option<String>, tip: String| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(name).color(theme::INK_DIM)).on_hover_text(tip.as_str());
            ui.label(RichText::new(value).strong().monospace()).on_hover_text(tip.as_str());
            if let Some(x) = extra {
                ui.label(RichText::new(x).size(11.0).color(theme::INK_DIM)).on_hover_text(tip.as_str());
            }
        });
    };
    if let Some(m) = m {
        // held vs fitted (plan review S9): the trust check counts the fit's samples
        let fitted = m.fit.map_or(String::new(), |f| format!("; {} in the current fit", f.0));
        line(ui, "samples in the model", m.samples.to_string(), None, format!("held{fitted} (the store; the Model tab shows the same number)"));
        let c = &m.solves;
        let total = if c.unrecorded_before { format!("≥ {} (earlier solves unrecorded)", (c.total() as usize).max(m.samples)) } else { c.total().to_string() };
        line(
            ui,
            "teacher solves for this model",
            total,
            Some(format!("(pass {} · fallback {} · audit {})", c.pass, c.fallback, c.audit)),
            format!(
                "pass {} · fallback {} · audit {} · relax checks {} — kept {}, not kept {} (references, similar answers, box-affected, failed, discarded audits, …). Completed or failed teaching solves; a cancelled solve is not counted",
                c.pass,
                c.fallback,
                c.audit,
                c.verification,
                c.kept,
                c.not_kept()
            ),
        );
    }
    // teaching solves only: a plain DFT step is not one (plan review S16)
    line(
        ui,
        "teaching solves this session",
        st.session_solves.to_string(),
        None,
        "teacher solves offered to a model since the app started, over every solver switch; plain DFT steps are not counted".into(),
    );
}

/// The step cell of a log row and its hover (plan review M6): `pass` for a pass solve, else the
/// driver step whose evaluation produced the event; an audit's hover says how many answered
/// steps after its dispatch it was judged (fallback steps in between are not counted in that lag).
pub fn step_cell(e: &sim::TeachEvent) -> (String, Option<String>) {
    let text = match (e.origin, e.step) {
        (TeachOrigin::Pass, _) => "pass".to_string(),
        (_, None) => "—".to_string(),
        (_, Some(s)) => s.to_string(),
    };
    let hover = e.audit_lag.map(|k| format!("judged {k} answered steps after its dispatch"));
    (text, hover)
}

/// The origin's word and colour.
pub fn origin_style(o: TeachOrigin) -> (&'static str, Color32) {
    match o {
        TeachOrigin::Pass => ("pass", theme::ACCENT),
        TeachOrigin::Fallback => ("fallback", AMBER),
        TeachOrigin::Audit => ("audit", GREEN),
        TeachOrigin::Verification => ("check", theme::INK_DIM),
    }
}

/// Whether the chips admit this row.
pub fn admits(f: &LogFilter, e: &sim::TeachEvent) -> bool {
    let origin = match e.origin {
        TeachOrigin::Pass => f.pass,
        TeachOrigin::Fallback => f.fallback,
        TeachOrigin::Audit => f.audit,
        TeachOrigin::Verification => f.check,
    };
    origin && if e.outcome.is_kept() { f.kept } else { f.not_kept }
}

/// The rule cell: the code, with an audit's verdict (`8 audit (passed)`).
pub fn rule_cell(e: &sim::TeachEvent) -> String {
    let base = words::rule_label(e.rule, e.burst_t);
    match (e.rule, e.audit) {
        (sim::activity::TeachRule::Audit, Some(v)) => {
            let w = match v {
                sim::AuditVerdict::Passed => "passed",
                sim::AuditVerdict::Failed => "failed",
                sim::AuditVerdict::BoxAffected => "box",
            };
            format!("{base} ({w})")
        }
        _ => base,
    }
}

/// The |ΔE| cell and its hover.
pub fn gap_cell(e: &sim::TeachEvent) -> (String, Option<String>) {
    match e.gap {
        Some(g) if g.is_finite() => {
            let per_atom = if e.n_atoms > 0 { format!("{:.2} mHa per atom over {} atoms; ", 1e3 * g.abs() / e.n_atoms as f64, e.n_atoms) } else { String::new() };
            (format!("{:.2} mHa", g.abs() * 1e3), Some(format!("{per_atom}signed {g:+.3e} Ha (teacher − model, where the model was consulted)")))
        }
        _ => ("—".into(), Some("the model was not consulted for this solve".into())),
    }
}

/// The solve-time cell: ms, or s above 10 s; "—" for a row with no solve.
pub fn solve_cell(e: &sim::TeachEvent) -> String {
    if !e.is_solve() {
        "—".into()
    } else if e.solve_ms >= 10_000.0 {
        format!("{:.1} s", e.solve_ms / 1e3)
    } else {
        format!("{:.0} ms", e.solve_ms)
    }
}

/// One line of the displayed list (oldest-first positions are indices into `rows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Row(usize),
    /// A new run began between the neighbouring rows: steps restart at 0.
    Divider,
}

/// The displayed items, oldest first: the rows the chips admit, with a divider before the first
/// row of each new run (by event time; none before the first row of the session).
pub fn items(st: &ActivityState) -> Vec<Item> {
    let mut v = Vec::new();
    let mut prev: Option<std::time::Duration> = None;
    for (i, r) in st.rows.iter().enumerate() {
        if !admits(&st.filter, &r.event) {
            continue;
        }
        if let Some(p) = prev {
            let k = st.runs.partition_point(|&t| t <= p);
            if st.runs.get(k).is_some_and(|&t| t <= r.at) {
                v.push(Item::Divider);
            }
        }
        prev = Some(r.at);
        v.push(Item::Row(i));
    }
    v
}

const ROW_H: f32 = 17.0;
/// Column widths: time, step, origin, rule, outcome, |ΔE|, solve (rule and outcome grow with the
/// dock).
const FIXED: [f32; 5] = [54.0, 52.0, 52.0, 56.0, 46.0];
const MIN_W: f32 = 452.0;

fn cell(ui: &Ui, rect: Rect, text: &str, font: FontId, colour: Color32) {
    let mut job = LayoutJob::single_section(text.to_string(), TextFormat { font_id: font, color: colour, ..Default::default() });
    job.wrap = TextWrapping::truncate_at_width((rect.width() - 4.0).max(1.0));
    let galley = ui.painter().layout_job(job);
    let y = rect.center().y - galley.size().y / 2.0;
    ui.painter_at(rect).galley(pos2(rect.left() + 2.0, y), galley, colour);
}

/// The chips and the table.
pub fn show_log(ui: &mut Ui, st: &mut ActivityState, estimated: bool) {
    ui.horizontal_wrapped(|ui| {
        let f = &mut st.filter;
        for (on, label, tip) in [
            (&mut f.pass, "pass", "solves of a sampling pass"),
            (&mut f.fallback, "fallback", "steps the model did not answer"),
            (&mut f.audit, "audit", "routine checks of the model against DFT"),
            (&mut f.check, "check", "the DFT check of a relaxed geometry"),
        ] {
            if ui.add(egui::Button::selectable(*on, label)).on_hover_text(tip).clicked() {
                *on = !*on;
            }
        }
        ui.separator();
        for (on, label, tip) in [
            (&mut f.kept, "kept", "answers that went into the store"),
            (&mut f.not_kept, "not kept", "answers used for the step but not kept (similar, box-affected, other settings, failed, …) and rows that were not solved"),
        ] {
            if ui.add(egui::Button::selectable(*on, label)).on_hover_text(tip).clicked() {
                *on = !*on;
            }
        }
    });
    if st.rows.is_empty() {
        theme::note(
            ui,
            if estimated {
                "no teacher solves yet — they appear here as they land"
            } else {
                "no teacher solves yet — they appear here as they land (estimated solver); a plain DFT step is not a teaching solve"
            },
        );
        return;
    }
    let items = items(st);
    if items.is_empty() {
        theme::note(ui, "no row passes the filter chips");
        return;
    }
    if st.dropped > 0 {
        ui.label(RichText::new(format!("{} older events were dropped before they reached this view (a fallback storm); the model's solve counter includes them, \"this session\" does not", st.dropped)).size(11.0).color(AMBER));
    }
    let total_w = ui.available_width().max(MIN_W);
    let flex = ((total_w - FIXED.iter().sum::<f32>() - 4.0) / 2.0).max(60.0);
    let widths = [FIXED[0], FIXED[1], FIXED[2], flex + 14.0, flex - 14.0, FIXED[3], FIXED[4]];
    let mono = FontId::monospace(10.5);
    let prop = FontId::proportional(11.0);
    // rows are packed (the dock's 6 pt row gap would double their pitch)
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        egui::ScrollArea::both().id_salt("solve-log").max_height(260.0).auto_shrink([false, true]).show_rows(ui, ROW_H, items.len() + 1, |ui, range| {
        for j in range {
            let (rect, _) = ui.allocate_exact_size(vec2(total_w, ROW_H), Sense::hover());
            if j == 0 {
                // the header
                let mut x = rect.left();
                for (w, h) in widths.iter().zip(["time", "step", "origin", "rule", "outcome", "|ΔE|", "solve"]) {
                    cell(ui, Rect::from_min_size(pos2(x, rect.top()), vec2(*w, ROW_H)), h, FontId::monospace(9.5), theme::INK_FAINT);
                    x += w;
                }
                continue;
            }
            // newest first
            let item = items[items.len() - j];
            match item {
                Item::Divider => {
                    ui.painter().line_segment([pos2(rect.left(), rect.center().y), pos2(rect.left() + total_w - 10.0, rect.center().y)], egui::Stroke::new(1.0, theme::LINE));
                    cell(ui, Rect::from_min_size(rect.min, vec2(total_w, ROW_H)), "  new run — steps restart at 0  ", prop.clone(), theme::ACCENT_INK);
                }
                Item::Row(i) => {
                    let id = ui.id().with(("solve-row", i));
                    row(ui, id, rect, &st.rows[i], &widths, &mono, &prop);
                }
            }
        }
        });
    });
}

fn row(ui: &mut Ui, id: egui::Id, rect: Rect, r: &LogRow, widths: &[f32; 7], mono: &FontId, prop: &FontId) {
    let e = &r.event;
    let (step, step_hover) = step_cell(e);
    let (origin, origin_colour) = origin_style(e.origin);
    let cx = RowCtx { box_affected: e.box_affected, near_face: r.near_face, kept: e.outcome.is_kept() };
    let outcome_colour = match e.outcome {
        o if o.is_kept() => theme::INK,
        TeachOutcome::Failed | TeachOutcome::StoreRefused => RED,
        TeachOutcome::Reference => theme::INK_DIM,
        _ => AMBER,
    };
    let (gap, gap_hover) = gap_cell(e);
    let mut x = rect.left();
    let mut col = |ui: &mut Ui, k: usize, text: &str, font: &FontId, colour: Color32, hover: Option<String>| {
        let cr = Rect::from_min_size(pos2(x, rect.top()), vec2(widths[k], ROW_H));
        x += widths[k];
        cell(ui, cr, text, font.clone(), colour);
        let resp = ui.interact(cr, id.with(k), Sense::hover());
        if let Some(h) = hover {
            resp.on_hover_ui(|ui| {
                ui.set_max_width(380.0);
                ui.label(h);
            });
        }
    };
    col(ui, 0, &clock(r.at), mono, theme::INK_DIM, Some(format!("{:.1} s into the session", r.at.as_secs_f64())));
    col(ui, 1, &step, mono, theme::INK_DIM, step_hover);
    col(ui, 2, origin, prop, origin_colour, None);
    let sentence = words::rule_sentence(e.rule, e.atom, cx);
    let rule_hover = if e.detail.is_empty() { sentence } else { format!("{sentence}\n\n{}", e.detail) };
    col(ui, 3, &rule_cell(e), prop, theme::INK, Some(rule_hover));
    let mut outcome_hover = words::outcome_sentence(e.outcome);
    let says_box = matches!(e.outcome, TeachOutcome::KeptBoxAffected | TeachOutcome::BoxNotKept);
    if e.box_affected && !says_box {
        // a vote can keep a box-affected answer (plan review N1)
        outcome_hover += &format!("; the answer is box-affected (face density {:.1e} e/Bohr³)", e.face_density.unwrap_or(f64::NAN));
    }
    col(ui, 4, &words::outcome_label(e.outcome), prop, outcome_colour, Some(outcome_hover));
    col(ui, 5, &gap, mono, theme::INK, gap_hover);
    col(ui, 6, &solve_cell(e), mono, theme::INK_DIM, e.scf_steps.map(|n| format!("{n} SCF steps")));
}
