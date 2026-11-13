//! The counters and the live solve log of the Activity tab (activity spec §3.3.2, §3.3.3): one
//! row per teaching solve, newest first, with the rule that asked for it in words and what became
//! of the answer. Drawn with `show_rows`, so only the visible rows are laid out.
use super::words::{self, RowCtx};
use super::{ActivityState, DisplayRow, LogFilter, LogRow};
use crate::ui::logbook::clock;
use crate::ui::{theme, AMBER, GREEN, RED};
use egui::text::{LayoutJob, TextFormat, TextWrapping};
use egui::{pos2, vec2, Color32, FontId, Rect, RichText, Sense, Ui};
use sim::activity::{TeachOrigin, TeachOutcome};

/// The counters in one line under the timeline (the Activity redesign, 2026-11-14; spec §3.3.2):
/// "1589 samples · 209 teacher solves" on the left, "pass 182 · fallback 26 · audit 0" on the
/// right; without a model, the teaching solves this session. The full counter texts are the hover.
pub fn counters(ui: &mut Ui, st: &ActivityState, m: Option<&crate::sim::ModelStatus>) {
    let mut tips = Vec::new();
    let (left, right) = match m {
        Some(m) => {
            let c = &m.solves;
            let total = if c.unrecorded_before { format!("≥ {}", (c.total() as usize).max(m.samples)) } else { c.total().to_string() };
            // held vs fitted (plan review S9): the trust check counts the fit's samples
            let fitted = m.fit.map_or(String::new(), |f| format!("; {} in the current fit", f.0));
            tips.push(format!("samples in the model: {} held{fitted} (the store; the Model tab shows the same number)", m.samples));
            let earlier = if c.unrecorded_before { " (earlier solves unrecorded)" } else { "" };
            tips.push(format!(
                "teacher solves for this model: {total}{earlier} — pass {} · fallback {} · audit {} · relax checks {} — kept {}, not kept {} (references, similar answers, box-affected, failed, discarded audits, …). Completed or failed teaching solves; a cancelled solve is not counted",
                c.pass,
                c.fallback,
                c.audit,
                c.verification,
                c.kept,
                c.not_kept()
            ));
            (format!("{} samples · {total} teacher solves", m.samples), format!("pass {} · fallback {} · audit {}", c.pass, c.fallback, c.audit))
        }
        None => (format!("{} teaching solves this session", st.session_solves), String::new()),
    };
    // lone-atom steps answered from the references (rule 0b): their own kind — no solve, not a
    // fallback, not a model answer, nothing learned
    if let Some(n) = m.map(|m| m.solves.reference).filter(|&n| n > 0) {
        tips.push(format!("lone-atom steps from the references: {n} — steps whose atoms had no neighbour within the model's cutoff, answered from the isolated-atom references (E = Σ E_ref, zero forces: the model's own answer there): no DFT solve, not a fallback, not a model answer, nothing learned, never audited. Counted since the solver was built or the model forgotten"));
    }
    // teaching solves only: a plain DFT step is not one (plan review S16)
    tips.push(format!("teaching solves this session: {} — teacher solves offered to a model since the app started, over every solver switch; plain DFT steps are not counted", st.session_solves));
    super::jobs::two_sides(ui, &left, &right, FontId::monospace(11.0), theme::INK_DIM).on_hover_text(tips.join("\n\n"));
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

/// The solve log's view (the chips "all / fallback / audit / failed", 2026-11-14): every row the
/// filters admit, or only fallbacks, audits or failures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogView {
    #[default]
    All,
    Fallback,
    Audit,
    Failed,
}

/// A solve that failed: the teacher's error, a refused store, or an SCF that did not converge
/// (stalled, the cap, a non-finite Δρ, the pass's time limit).
pub fn is_failure(e: &sim::TeachEvent) -> bool {
    matches!(e.outcome, TeachOutcome::Failed | TeachOutcome::StoreRefused)
        || matches!(e.scf_end, Some(sim::ScfEnd::Stalled | sim::ScfEnd::Cap | sim::ScfEnd::NonFinite | sim::ScfEnd::TimeLimit))
}

/// A row that may fold into a run of identical ones: kept answers, references and held-out
/// checks — never a failure or an answer that was not kept.
pub fn collapsible(e: &sim::TeachEvent) -> bool {
    !is_failure(e) && (e.outcome.is_kept() || matches!(e.outcome, TeachOutcome::Reference | TeachOutcome::HeldOut))
}

/// Whether the view chip admits this row.
pub fn in_view(v: LogView, e: &sim::TeachEvent) -> bool {
    match v {
        LogView::All => true,
        LogView::Fallback => e.origin == TeachOrigin::Fallback,
        LogView::Audit => e.origin == TeachOrigin::Audit,
        LogView::Failed => is_failure(e),
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

/// The solve cell: the time (ms, or s above 10 s) and the SCF iterations with how they ended —
/// "840 ms · 37 it" converged, "6.1 s · 61 stalled" (the stall stop), "9.8 s · 200 cap"; "—" for a
/// row with no solve.
pub fn solve_cell(e: &sim::TeachEvent) -> String {
    if !e.is_solve() {
        return "—".into();
    }
    let t = if e.solve_ms >= 10_000.0 { format!("{:.1} s", e.solve_ms / 1e3) } else { format!("{:.0} ms", e.solve_ms) };
    match (e.scf_steps.filter(|&n| n > 0), e.scf_end) {
        (Some(n), None | Some(sim::ScfEnd::Converged)) => format!("{t} · {n} it"),
        (Some(n), Some(end)) => format!("{t} · {n} {}", end.word()),
        (None, Some(sim::ScfEnd::TimeLimit)) => format!("{t} · {}", sim::ScfEnd::TimeLimit.word()),
        (None, _) => t,
    }
}

/// The solve cell's hover: the iterations and the end in words.
pub fn solve_hover(e: &sim::TeachEvent) -> Option<String> {
    if e.scf_end == Some(sim::ScfEnd::TimeLimit) {
        return Some("given up at the sampling pass's per-sample time limit (several times the pass's median solve): not kept, like a stalled solve".into());
    }
    let n = e.scf_steps?;
    let why = match e.scf_end {
        Some(sim::ScfEnd::Converged) => " — converged",
        Some(sim::ScfEnd::Stalled) => " — stalled: the best Δρ did not halve within the stall window, so the solve stopped before the cap",
        Some(sim::ScfEnd::Cap) => " — stopped at the iteration cap without converging",
        Some(sim::ScfEnd::NonFinite) => " — stopped: Δρ was not finite",
        Some(sim::ScfEnd::TimeLimit) | None => "",
    };
    Some(format!("{n} SCF iterations{why}"))
}

/// One line of the displayed list (oldest-first positions are indices into `rows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Row(usize),
    /// A display eigensolve: an index into `display_rows`.
    Display(usize),
    /// A new run began between the neighbouring rows: steps restart at 0.
    Divider,
    /// A run of identical rows (same step, origin, rule and outcome; [`collapsible`]): the oldest
    /// and newest row indices and how many; a click expands it into its [`Item::Member`]s.
    Group { first: usize, last: usize, n: usize },
    /// A row of an expanded group.
    Member(usize),
}

/// The key two rows must share to fold together.
fn same(a: &sim::TeachEvent, b: &sim::TeachEvent) -> bool {
    (a.step, a.origin, a.rule, a.outcome) == (b.step, b.origin, b.rule, b.outcome)
}

/// The group's words: "12 × nudge — kept · 21–28 s · 17–24 it".
pub fn group_text(rows: &[&sim::TeachEvent]) -> String {
    let Some(e) = rows.first() else { return String::new() };
    let ms: Vec<f64> = rows.iter().filter(|e| e.is_solve()).map(|e| e.solve_ms).collect();
    let time = if ms.is_empty() {
        String::new()
    } else {
        let (lo, hi) = (ms.iter().copied().fold(f64::INFINITY, f64::min), ms.iter().copied().fold(0.0, f64::max));
        if hi >= 1000.0 {
            format!(" · {:.0}–{:.0} s", lo / 1e3, hi / 1e3)
        } else {
            format!(" · {lo:.0}–{hi:.0} ms")
        }
    };
    let its: Vec<usize> = rows.iter().filter_map(|e| e.scf_steps).collect();
    let it = match (its.iter().min(), its.iter().max()) {
        (Some(a), Some(b)) => format!(" · {a}–{b} it"),
        _ => String::new(),
    };
    format!("{} × {} — {}{time}{it}", rows.len(), rule_cell(e), words::outcome_label(e.outcome))
}

/// The displayed items, oldest first: the rows the chips admit and the display eigensolves
/// (the "orbitals" chip), merged by time, with a divider before the first row of each new run
/// (by event time; none before the first row of the session).
pub fn items(st: &ActivityState) -> Vec<Item> {
    let teach = st.rows.iter().enumerate().filter(|(_, r)| admits(&st.filter, &r.event) && in_view(st.log_view, &r.event)).map(|(i, r)| (r.at, Item::Row(i)));
    let display = st.display_rows.iter().enumerate().filter(|_| st.filter.orbitals && st.log_view == LogView::All).map(|(i, r)| (r.at, Item::Display(i)));
    // both are oldest first: a stable sort by time keeps each one's order
    let mut merged: Vec<(std::time::Duration, Item)> = teach.chain(display).collect();
    merged.sort_by_key(|&(at, _)| at);
    let mut v = Vec::new();
    let mut prev: Option<std::time::Duration> = None;
    for (at, item) in merged {
        if let Some(p) = prev {
            let k = st.runs.partition_point(|&t| t <= p);
            if st.runs.get(k).is_some_and(|&t| t <= at) {
                v.push(Item::Divider);
            }
        }
        prev = Some(at);
        v.push(item);
    }
    collapse(st, v)
}

/// Folds runs of two or more identical [`collapsible`] rows into a [`Item::Group`] (followed by its
/// members when expanded, `ActivityState::expanded`, keyed by the oldest row's time).
fn collapse(st: &ActivityState, v: Vec<Item>) -> Vec<Item> {
    let mut out = Vec::with_capacity(v.len());
    let mut run: Vec<usize> = Vec::new();
    let flush = |run: &mut Vec<usize>, out: &mut Vec<Item>| {
        match run.len() {
            0 => {}
            1 => out.push(Item::Row(run[0])),
            n => {
                out.push(Item::Group { first: run[0], last: run[n - 1], n });
                if st.expanded.contains(&st.rows[run[0]].at) {
                    out.extend(run.iter().map(|&i| Item::Member(i)));
                }
            }
        }
        run.clear();
    };
    for item in v {
        match item {
            Item::Row(i) if collapsible(&st.rows[i].event) => {
                if run.last().is_some_and(|&j| !same(&st.rows[j].event, &st.rows[i].event)) {
                    flush(&mut run, &mut out);
                }
                run.push(i);
            }
            other => {
                flush(&mut run, &mut out);
                out.push(other);
            }
        }
    }
    flush(&mut run, &mut out);
    out
}

/// The newest failure of the pass running now (pinned over the log while the pass runs).
pub fn pinned_failure(st: &ActivityState) -> Option<usize> {
    let since = st.pass_started()?;
    st.rows.iter().enumerate().rev().take_while(|(_, r)| r.at >= since).find(|(_, r)| is_failure(&r.event)).map(|(i, _)| i)
}

/// A display eigensolve's solve cell: its wall time, and "stopped" when a command interrupted it.
pub fn display_solve_cell(d: &sim::DisplaySolve) -> String {
    let t = if d.ms >= 10_000.0 { format!("{:.1} s", d.ms / 1e3) } else { format!("{:.0} ms", d.ms) };
    if d.interrupted {
        format!("{t} · stopped")
    } else {
        t
    }
}

const ROW_H: f32 = 17.0;
/// Column widths: time, step, origin, rule, outcome, |ΔE|, solve (rule and outcome grow with the
/// dock; the solve cell holds the time and the iterations, "6.1 s · 61 stalled").
const FIXED: [f32; 5] = [54.0, 52.0, 52.0, 56.0, 104.0];
const MIN_W: f32 = 510.0;

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
        for (v, label, tip) in [
            (LogView::All, "all", "every row the filters below admit"),
            (LogView::Fallback, "fallback", "steps the model did not answer"),
            (LogView::Audit, "audit", "routine checks of the model against DFT"),
            (LogView::Failed, "failed", "solves that failed or did not converge (stalled, the cap, the pass's time limit)"),
        ] {
            if crate::ui::probed(&format!("log-chip-{label}"), ui.add(egui::Button::selectable(st.log_view == v, label))).on_hover_text(tip).clicked() {
                st.log_view = v;
            }
        }
    });
    // the finer filters of before, folded away
    egui::CollapsingHeader::new(RichText::new("more filters").size(11.5).color(theme::INK_DIM)).id_salt("log-more-filters").show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            let f = &mut st.filter;
            for (on, label, tip) in [
                (&mut f.pass, "pass", "solves of a sampling pass"),
                (&mut f.fallback, "fallback", "steps the model did not answer"),
                (&mut f.audit, "audit", "routine checks of the model against DFT"),
                (&mut f.check, "check", "the DFT check of a relaxed geometry"),
                (&mut f.kept, "kept", "answers that went into the store"),
                (&mut f.not_kept, "not kept", "answers used for the step but not kept (similar, box-affected, other settings, failed, …) and rows that were not solved"),
            ] {
                if ui.add(egui::Button::selectable(*on, label)).on_hover_text(tip).clicked() {
                    *on = !*on;
                }
            }
            let f = &mut st.filter.orbitals;
            if ui.add(egui::Button::selectable(*f, "orbitals")).on_hover_text("the display eigensolve of unoccupied orbitals after a DFT step (Orbitals tab): display only, not a teaching solve").clicked() {
                *f = !*f;
            }
        });
    });
    if st.rows.is_empty() && st.display_rows.is_empty() {
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
    let pinned = pinned_failure(st);
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
        let extra = 1 + pinned.is_some() as usize;
        let mut toggle: Option<std::time::Duration> = None;
        egui::ScrollArea::both().id_salt("solve-log").max_height(260.0).auto_shrink([false, true]).show_rows(ui, ROW_H, items.len() + extra, |ui, range| {
        for j in range {
            let (rect, resp) = ui.allocate_exact_size(vec2(total_w, ROW_H), Sense::click());
            if j == 0 {
                // the header
                let mut x = rect.left();
                for (w, h) in widths.iter().zip(["time", "step", "origin", "rule", "outcome", "|ΔE|", "solve"]) {
                    cell(ui, Rect::from_min_size(pos2(x, rect.top()), vec2(*w, ROW_H)), h, FontId::monospace(9.5), theme::INK_FAINT);
                    x += w;
                }
                continue;
            }
            // the newest failure of the running pass, pinned at the top
            if let (1, Some(i)) = (j, pinned) {
                ui.painter().rect_filled(rect, 2.0, RED.gamma_multiply(0.16));
                let id = ui.id().with(("pinned-row", i));
                row(ui, id, rect, &st.rows[i], &widths, &mono, &prop);
                crate::ui::probe_rect("log-pinned", rect);
                continue;
            }
            // newest first
            let item = items[items.len() - (j + 1 - extra)];
            match item {
                Item::Divider => {
                    ui.painter().line_segment([pos2(rect.left(), rect.center().y), pos2(rect.left() + total_w - 10.0, rect.center().y)], egui::Stroke::new(1.0, theme::LINE));
                    cell(ui, Rect::from_min_size(rect.min, vec2(total_w, ROW_H)), "  new run — steps restart at 0  ", prop.clone(), theme::ACCENT_INK);
                }
                Item::Row(i) | Item::Member(i) => {
                    let e = &st.rows[i].event;
                    if is_failure(e) {
                        ui.painter().rect_filled(rect, 2.0, RED.gamma_multiply(0.10));
                    } else if !collapsible(e) && e.outcome != TeachOutcome::NotSolved {
                        ui.painter().rect_filled(rect, 2.0, AMBER.gamma_multiply(0.06));
                    }
                    if matches!(item, Item::Member(_)) {
                        ui.painter().rect_filled(Rect::from_min_size(rect.min, vec2(2.0, ROW_H)), 0.0, theme::ACCENT.gamma_multiply(0.6));
                    }
                    let id = ui.id().with(("solve-row", i));
                    row(ui, id, rect, &st.rows[i], &widths, &mono, &prop);
                }
                Item::Group { first, last, n: _ } => {
                    let open = st.expanded.contains(&st.rows[first].at);
                    let members: Vec<&sim::TeachEvent> = (first..=last).map(|i| &st.rows[i].event).filter(|e| same(e, &st.rows[first].event)).collect();
                    let newest = &st.rows[last];
                    let (step, _) = step_cell(&newest.event);
                    let (origin, origin_colour) = origin_style(newest.event.origin);
                    let mut x = rect.left();
                    for (k, (text, font, colour)) in [(clock(newest.at), &mono, theme::INK_DIM), (step, &mono, theme::INK_DIM), (origin.to_string(), &prop, origin_colour)].into_iter().enumerate() {
                        cell(ui, Rect::from_min_size(pos2(x, rect.top()), vec2(widths[k], ROW_H)), &text, font.clone(), colour);
                        x += widths[k];
                    }
                    let caret = if open { "▾ " } else { "▸ " };
                    let rest = Rect::from_min_max(pos2(x, rect.top()), pos2(rect.left() + total_w, rect.bottom()));
                    cell(ui, rest, &format!("{caret}{}", group_text(&members)), prop.clone(), theme::INK);
                    crate::ui::probe_rect(&format!("log-group-{first}"), rect);
                    if resp.on_hover_text(if open { "fold these rows" } else { "show each of these rows" }).clicked() {
                        toggle = Some(st.rows[first].at);
                    }
                }
                Item::Display(i) => {
                    let id = ui.id().with(("display-row", i));
                    display_row(ui, id, rect, &st.display_rows[i], &widths, &mono, &prop);
                }
            }
        }
        });
        if let Some(at) = toggle {
            if !st.expanded.remove(&at) {
                st.expanded.insert(at);
            }
        }
    });
}

/// A display eigensolve's row: "orbitals" in the origin column, the orbitals it solved for, and
/// its wall time in the solve column; no step, outcome or |ΔE| (it is not a teaching solve).
fn display_row(ui: &mut Ui, id: egui::Id, rect: Rect, r: &DisplayRow, widths: &[f32; 7], mono: &FontId, prop: &FontId) {
    let d = &r.solve;
    let mut x = rect.left();
    let mut col = |ui: &mut Ui, k: usize, text: &str, font: &FontId, colour: Color32, hover: Option<String>| {
        let cr = Rect::from_min_size(pos2(x, rect.top()), vec2(widths[k], ROW_H));
        x += widths[k];
        cell(ui, cr, text, font.clone(), colour);
        if let Some(h) = hover {
            ui.interact(cr, id.with(k), Sense::hover()).on_hover_ui(|ui| {
                ui.set_max_width(380.0);
                ui.label(h);
            });
        }
    };
    col(ui, 0, &clock(r.at), mono, theme::INK_DIM, Some(format!("{:.1} s into the session (when it ended)", r.at.as_secs_f64())));
    col(ui, 1, "—", mono, theme::INK_DIM, Some("display work after a DFT step, not a step's solve".into()));
    col(ui, 2, "orbitals", prop, theme::INK_DIM, None);
    let what = format!("{} unoccupied per spin", d.unoccupied);
    col(ui, 3, &what, prop, theme::INK, Some("the display eigensolve: the lowest unoccupied orbitals on the converged state of the last DFT step, for the Orbitals view only — it never feeds the solve, so it is not a teaching solve and no counter counts it".into()));
    let (outcome, colour, why) = if d.interrupted {
        ("stopped", AMBER, "a command arrived (an edit, Play, a slider) and the eigensolve yielded to it; it runs again later")
    } else {
        ("shown", theme::INK_DIM, "the orbitals went to the Orbitals view")
    };
    col(ui, 4, outcome, prop, colour, Some(why.into()));
    col(ui, 5, "—", mono, theme::INK_DIM, None);
    col(ui, 6, &display_solve_cell(d), mono, theme::INK_DIM, Some(format!("{:.0} ms of wall time", d.ms)));
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
    let solve_colour = match e.scf_end {
        Some(sim::ScfEnd::Stalled | sim::ScfEnd::Cap | sim::ScfEnd::NonFinite | sim::ScfEnd::TimeLimit) => AMBER,
        _ => theme::INK_DIM,
    };
    col(ui, 6, &solve_cell(e), mono, solve_colour, solve_hover(e));
}
