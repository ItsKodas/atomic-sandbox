//! The refit picture (spec `docs/superpowers/specs/2026-11-22-refit-visualisation-design.md`, plan
//! `docs/superpowers/plans/2026-11-25-refit-visualisation.md`): what a refit is actually doing, from
//! its own stage counters (`sim::refit_live`). Display only.
//!
//! - **The view** ([`show`]) has the full picture while nothing else needs it: a refit runs and no
//!   model stands (the first fit at launch, any refit while there is no model; Play is refused then
//!   anyway), and no decision card (the model offer) is up. It fades out when the fit lands. It is
//!   not one of the zones' overlays: it takes the room the zones leave free ([`free_rect`]), never
//!   input (the camera still turns under it).
//! - **Activity's refit card** ([`strip`]) has the compact strip of the same stages for every
//!   refit, so the view stays on the simulation.
//!
//! The picture: one block per sample ([`blocks`]: one per k samples when they do not fit), filling
//! as the samples are gathered and lighting up as each is checked — green within its bound, amber
//! beyond it (the Worst-fitting list's ratios); the Q′ × Q′ matrix, its current decomposition's
//! square filling over the mean time of one, a tick per decomposition finished, the search's round;
//! the stage list with each stage's time spent, the running one's counter, and the trust recovery's
//! round. A stage that has not started has no number.
use super::zones::{ViewLayout, GAP};
use super::{activity::words, theme, PanelInput, UiState, AMBER, GREEN};
use egui::{pos2, vec2, Align2, Color32, FontId, Id, Order, Painter, Rect, Sense, Stroke, Ui, Vec2};
use sim::{LiveStage, RefitLive, RefitView};

/// The fade-out once the fit lands (seconds).
pub const FADE_S: f32 = 0.6;
/// The picture's largest and smallest size (points): below the smallest the view has no room
/// for it (the Activity strip still says everything).
pub const MAX_SIZE: Vec2 = Vec2::new(760.0, 430.0);
pub const MIN_SIZE: Vec2 = Vec2::new(340.0, 190.0);
/// The picture's inset from the view's edges.
const MARGIN: f32 = 16.0;
/// A sample block's side (points) at most and at least, and the gap between blocks.
pub const BLOCK_MAX: f32 = 12.0;
pub const BLOCK_MIN: f32 = 3.0;
pub const BLOCK_GAP: f32 = 1.0;

/// The picture's last frame, for its fade-out (`UiState::refit_picture`).
#[derive(Default)]
pub struct Fade {
    pub last: Option<(RefitView, Vec<f32>)>,
}

/// Whether the view has the full picture now (module doc).
pub fn in_view(input: &PanelInput<'_>) -> bool {
    input.probe.and_then(|p| p.refit).is_some() && !input.snapshot.and_then(|s| s.model.as_ref()).is_some_and(|m| m.fitted)
}

/// The largest rect of `view` (less [`MARGIN`]) clear of every rect in `taken` (with [`GAP`]):
/// each overlap is cut away on the side that keeps the most area.
pub fn free_rect(view: Rect, taken: &[Rect]) -> Rect {
    let mut r = view.shrink(MARGIN);
    for t in taken {
        let t = t.expand(GAP);
        if !r.intersects(t) || r.width() <= 0.0 || r.height() <= 0.0 {
            continue;
        }
        let cuts = [
            Rect::from_min_max(pos2(r.left(), t.bottom()), r.max),
            Rect::from_min_max(r.min, pos2(r.right(), t.top())),
            Rect::from_min_max(pos2(t.right(), r.top()), r.max),
            Rect::from_min_max(r.min, pos2(t.left(), r.bottom())),
        ];
        let area = |c: &Rect| if c.width() > 0.0 && c.height() > 0.0 { c.area() } else { 0.0 };
        r = cuts.into_iter().max_by(|a, b| area(a).total_cmp(&area(b))).filter(|c| area(c) > 0.0).unwrap_or(Rect::from_min_size(r.min, Vec2::ZERO));
    }
    r
}

/// The picture's rect: centred in [`free_rect`], at most [`MAX_SIZE`]; `None` when the free room
/// is under [`MIN_SIZE`].
pub fn picture_rect(view: Rect, taken: &[Rect]) -> Option<Rect> {
    let free = free_rect(view, taken);
    (free.width() >= MIN_SIZE.x && free.height() >= MIN_SIZE.y).then(|| Rect::from_center_size(free.center(), free.size().min(MAX_SIZE)))
}

/// How `n` samples are drawn in `area` ([`blocks`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blocks {
    /// Samples per block (1 unless they do not fit at [`BLOCK_MIN`]).
    pub per: usize,
    /// Blocks drawn.
    pub count: usize,
    /// Blocks per row.
    pub cols: usize,
    /// A block's side (points).
    pub side: f32,
}

/// The largest block side in [[`BLOCK_MIN`], [`BLOCK_MAX`]] at which `n` blocks fit `area`,
/// wrapping to its width; past the room at [`BLOCK_MIN`], one block per k samples.
pub fn blocks(n: usize, area: Vec2) -> Blocks {
    let fit = |s: f32| {
        let c = (((area.x + BLOCK_GAP) / (s + BLOCK_GAP)).floor() as usize).max(1);
        let r = (((area.y + BLOCK_GAP) / (s + BLOCK_GAP)).floor() as usize).max(1);
        (c, r)
    };
    if n == 0 {
        return Blocks { per: 1, count: 0, cols: fit(BLOCK_MAX).0, side: BLOCK_MAX };
    }
    let mut side = ((area.x.max(1.0) * area.y.max(1.0) / n as f32).sqrt() - BLOCK_GAP).clamp(BLOCK_MIN, BLOCK_MAX).floor();
    while side > BLOCK_MIN {
        let (c, r) = fit(side);
        if c * r >= n {
            break;
        }
        side -= 1.0;
    }
    let side = side.max(BLOCK_MIN);
    let (cols, rows) = fit(side);
    let per = n.div_ceil(cols * rows).max(1);
    Blocks { per, count: n.div_ceil(per), cols, side }
}

/// A block's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// Not gathered, or not checked yet.
    Waiting,
    /// Gathered into the fit's statistics.
    Done,
    /// Checked, every sample within its bound.
    Within,
    /// Checked, some sample beyond its bound (or undefined).
    Beyond,
}

/// What the blocks show: a counted pass's progress, or the checked samples' ratios.
#[derive(Clone, Copy, Debug)]
pub enum Fill<'a> {
    Counted { done: usize },
    Ratios(&'a [f32]),
}

/// The blocks' sample count and fill for a refit now: gathering its pass's count; checking (and
/// after it) the ratios of the last check; before any check, every sample gathered once the
/// gathering is past.
pub fn fill_of<'a>(v: &RefitView, ratios: &'a [f32]) -> (usize, Fill<'a>) {
    let l = &v.live;
    match l.stage {
        LiveStage::Gather => (l.total as usize, Fill::Counted { done: l.done as usize }),
        _ if !ratios.is_empty() => (ratios.len(), Fill::Ratios(ratios)),
        LiveStage::Starting => (v.samples as usize, Fill::Counted { done: 0 }),
        _ => (v.samples as usize, Fill::Counted { done: v.samples as usize }),
    }
}

/// Block `b`'s state (of `per` samples, `n` in all).
pub fn cell(b: usize, per: usize, n: usize, fill: Fill<'_>) -> Cell {
    let (lo, hi) = (b * per, ((b + 1) * per).min(n));
    match fill {
        Fill::Counted { done } => {
            if hi <= done {
                Cell::Done
            } else {
                Cell::Waiting
            }
        }
        Fill::Ratios(r) => {
            let r = &r[lo.min(r.len())..hi.min(r.len())];
            if r.iter().any(|x| *x > 1.0) {
                Cell::Beyond
            } else if r.iter().all(|x| x.is_nan()) {
                Cell::Waiting
            } else {
                Cell::Within
            }
        }
    }
}

fn cell_colour(c: Cell) -> Color32 {
    match c {
        Cell::Waiting => theme::PANEL_3,
        Cell::Done => theme::ACCENT.gamma_multiply(0.7),
        Cell::Within => GREEN.gamma_multiply(0.85),
        Cell::Beyond => AMBER,
    }
}

/// "1:05" for 65 s.
fn mmss(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// The stage rows of the list and the strip: (short name, long name, slot in `spent_s`, running
/// now, its counter when running). The mix row only when the refit learnt W; the trust row only in
/// a recovery.
fn stage_rows(l: &RefitLive) -> Vec<(&'static str, &'static str, Option<usize>, bool, String)> {
    let running = |s: LiveStage| !l.mix && l.stage.slot() == s.slot();
    let counter = |on: bool| if on { words::refit_counter_words(l) } else { String::new() };
    let mut v = vec![("Gather", "Gathering samples", Some(LiveStage::Gather.slot()), running(LiveStage::Gather), counter(running(LiveStage::Gather)))];
    if l.mix || l.spent_s[sim::refit_live::MIX_SLOT] > 0.0 {
        v.push(("Mix W", "Learning the mix W", Some(sim::refit_live::MIX_SLOT), l.mix, if l.mix { format!("{} · {}", words::refit_stage_name(l).trim_start_matches("learning the mix W · "), words::refit_counter_words(l)) } else { String::new() }));
    }
    v.push(("Solve", "Solving the model", Some(LiveStage::Solve.slot()), running(LiveStage::Solve), counter(running(LiveStage::Solve))));
    v.push(("Check", "Checking each sample", Some(LiveStage::Check.slot()), running(LiveStage::Check), counter(running(LiveStage::Check))));
    let other = matches!(l.stage, LiveStage::Other(_)) && !l.mix;
    v.push(("Other", "Other checks", Some(LiveStage::Other(sim::OtherPass::Counts).slot()), other, if other { words::refit_stage_name(l) } else { String::new() }));
    if let Some(t) = l.trust {
        let w = words::refit_trust_words(l).unwrap_or_default();
        v.push(("Trust", "Trust recovery", None, true, if t.round == 0 { "basis retry".into() } else { w.trim_start_matches("trust ").to_string() }));
    }
    v
}

/// The view's full picture (module doc), drawn after the zones' layout (`overlays::show`).
pub fn show(ctx: &egui::Context, view: Rect, lay: &ViewLayout, st: &mut UiState, input: &PanelInput<'_>) {
    let live = in_view(input) && st.model_offer.is_none();
    if live {
        if let Some(v) = input.probe.and_then(|p| p.refit) {
            let last = st.refit_picture.last.get_or_insert_with(|| (v, Vec::new()));
            last.0 = v;
            last.1.clear();
            last.1.extend_from_slice(&input.refit_ratios);
        }
    }
    let alpha = ctx.animate_bool_with_time(Id::new("refit-picture-fade"), live, FADE_S);
    if alpha <= 0.0 {
        st.refit_picture.last = None;
        return;
    }
    let Some((v, ratios)) = st.refit_picture.last.as_ref() else { return };
    let taken: Vec<Rect> = lay.rects().map(|(_, r)| r).collect();
    let Some(rect) = picture_rect(view, &taken) else { return };
    let time = ctx.input(|i| i.time);
    egui::Area::new(Id::new("refit-picture"))
        .order(Order::Middle)
        .fixed_pos(rect.min)
        .interactable(false)
        .fade_in(false)
        .show(ctx, |ui| {
            ui.set_opacity(alpha);
            let (r, _) = ui.allocate_exact_size(rect.size(), Sense::hover());
            super::probe_rect("refit-picture", r);
            draw(ui.painter(), r, v, ratios, time);
        });
    ctx.request_repaint_after(std::time::Duration::from_millis(if live { 100 } else { 16 }));
}

/// The full picture in `rect`.
fn draw(p: &Painter, rect: Rect, v: &RefitView, ratios: &[f32], time: f64) {
    let l = &v.live;
    p.rect_filled(rect, 8.0, theme::CARD);
    p.rect_stroke(rect, 8.0, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
    let inner = rect.shrink(14.0);
    // the header: what, how many, how long; then the stage now
    p.text(inner.left_top(), Align2::LEFT_TOP, "Fitting the model", FontId::proportional(15.0), theme::INK);
    p.text(inner.right_top(), Align2::RIGHT_TOP, format!("{} samples · {}", words::thousands(v.samples as u64), mmss(v.elapsed_s)), FontId::proportional(12.5), theme::INK_DIM);
    // "checking each sample · 1,700 of 2,655 · 8 beyond their bound · about 0:02 left in this stage"
    let mut parts: Vec<String> = words::refit_trust_words(l).into_iter().collect();
    parts.push(words::refit_stage_name(l));
    parts.push(words::refit_counter_words(l));
    parts.retain(|p| !p.is_empty());
    parts.extend(words::refit_left_words(l));
    let line = parts.join(" · ");
    p.text(inner.left_top() + vec2(0.0, 22.0), Align2::LEFT_TOP, line, FontId::proportional(12.0), theme::ACCENT_INK);
    let body = Rect::from_min_max(inner.left_top() + vec2(0.0, 46.0), inner.max);
    // the stage list on the left when there is room
    let list_w = if body.width() >= 520.0 { 200.0 } else { 0.0 };
    if list_w > 0.0 {
        let mut y = body.top();
        for (_, name, slot, on, count) in stage_rows(l) {
            let colour = if on { theme::ACCENT } else if slot.is_some_and(|s| l.spent_s[s] > 0.0) { theme::INK_DIM } else { theme::INK_FAINT };
            p.circle_filled(pos2(body.left() + 4.0, y + 7.0), 3.5, colour);
            p.text(pos2(body.left() + 14.0, y), Align2::LEFT_TOP, name, FontId::proportional(12.0), if on { theme::INK } else { colour });
            // a stage that has not started has no number
            let spent = slot.map(|s| l.spent_s[s]).filter(|&s| s > 0.0).map_or("—".to_string(), mmss);
            p.text(pos2(body.left() + list_w - 10.0, y), Align2::RIGHT_TOP, spent, FontId::monospace(11.0), colour);
            if on && !count.is_empty() {
                y += 16.0;
                let g = p.layout(count, FontId::proportional(10.5), theme::INK_DIM, list_w - 18.0);
                let h = g.size().y;
                p.galley(pos2(body.left() + 14.0, y), g, theme::INK_DIM);
                y += h - 16.0;
            }
            y += 22.0;
        }
    }
    // the matrix on the right, the sample blocks between
    let side = (body.height() * 0.55).min(130.0).min(body.width() * 0.25);
    let matrix = Rect::from_min_size(pos2(body.right() - side, body.top() + 16.0), Vec2::splat(side));
    draw_matrix(p, matrix, l, time);
    let grid = Rect::from_min_max(pos2(body.left() + list_w, body.top()), pos2(matrix.left() - 18.0, body.bottom()));
    draw_blocks(p, grid, v, ratios);
}

/// The Q′ × Q′ matrix (module doc).
fn draw_matrix(p: &Painter, m: Rect, l: &RefitLive, time: f64) {
    let solving = l.stage == LiveStage::Solve;
    let ink = if solving { theme::ACCENT } else { theme::INK_FAINT };
    let label = if l.q > 0 { format!("Q′ {} × {}", words::thousands(l.q as u64), words::thousands(l.q as u64)) } else { "the model's matrix".into() };
    p.text(m.left_top() - vec2(0.0, 3.0), Align2::LEFT_BOTTOM, label, FontId::proportional(10.5), theme::INK_DIM);
    p.rect_filled(m, 2.0, theme::PANEL_2);
    if solving {
        match l.decomposition_fill() {
            // the current decomposition, over the mean time of one: a sweep down the matrix
            Some(f) => {
                let filled = Rect::from_min_max(m.min, pos2(m.right(), m.top() + m.height() * f as f32));
                p.rect_filled(filled, 2.0, theme::ACCENT.gamma_multiply(0.45));
            }
            // the first: no measure yet, a pulse
            None => {
                let a = 0.2 + 0.2 * (time * 3.0).sin().abs() as f32;
                p.rect_filled(m, 2.0, theme::ACCENT.gamma_multiply(a));
            }
        }
    }
    // the grid lines of the matrix's blocks
    for k in 1..8 {
        let t = k as f32 / 8.0;
        let s = Stroke::new(0.5, theme::LINE_SOFT);
        p.line_segment([pos2(m.left() + m.width() * t, m.top()), pos2(m.left() + m.width() * t, m.bottom())], s);
        p.line_segment([pos2(m.left(), m.top() + m.height() * t), pos2(m.right(), m.top() + m.height() * t)], s);
    }
    p.rect_stroke(m, 2.0, Stroke::new(1.0, ink), egui::StrokeKind::Inside);
    // a tick per decomposition finished in this search
    let mut y = m.bottom() + 8.0;
    if l.round > 0 {
        let per_row = ((m.width() + 2.0) / 5.0).floor().max(1.0) as usize;
        let shown = (l.decompositions as usize).min(per_row * 3);
        for k in 0..shown {
            let (row, col) = (k / per_row, k % per_row);
            let c = pos2(m.left() + 2.0 + col as f32 * 5.0, y + 2.0 + row as f32 * 5.0);
            p.rect_filled(Rect::from_center_size(c, Vec2::splat(3.0)), 0.5, ink);
        }
        y += 6.0 + 5.0 * shown.div_ceil(per_row) as f32;
        let words = match l.decompositions {
            0 => format!("round {}", l.round),
            1 => format!("round {} · 1 decomposition", l.round),
            k => format!("round {} · {k} decompositions", l.round),
        };
        p.text(pos2(m.left(), y), Align2::LEFT_TOP, words, FontId::proportional(10.5), theme::INK_DIM);
    }
}

/// The sample blocks (module doc).
fn draw_blocks(p: &Painter, grid: Rect, v: &RefitView, ratios: &[f32]) {
    if grid.width() < 20.0 || grid.height() < 20.0 {
        return;
    }
    let (n, fill) = fill_of(v, ratios);
    let label = match (fill, v.live.stage) {
        (_, LiveStage::Gather) if v.live.mix => "the samples W is learnt from",
        (_, LiveStage::Gather) => "the samples, gathered into the fit",
        (Fill::Ratios(_), _) => "each sample left out of the fit: green within its bound, amber beyond",
        _ => "the samples",
    };
    p.text(grid.left_top(), Align2::LEFT_TOP, label, FontId::proportional(10.5), theme::INK_DIM);
    let area = Rect::from_min_max(grid.left_top() + vec2(0.0, 16.0), grid.max);
    let b = blocks(n, area.size());
    let step = b.side + BLOCK_GAP;
    for k in 0..b.count {
        let (row, col) = (k / b.cols, k % b.cols);
        let r = Rect::from_min_size(area.left_top() + vec2(col as f32 * step, row as f32 * step), Vec2::splat(b.side));
        p.rect_filled(r, if b.side >= 6.0 { 1.5 } else { 0.0 }, cell_colour(cell(k, b.per, n, fill)));
    }
    if b.per > 1 {
        let rows = b.count.div_ceil(b.cols);
        p.text(area.left_top() + vec2(0.0, rows as f32 * step + 4.0), Align2::LEFT_TOP, format!("one block per {} samples", b.per), FontId::proportional(10.0), theme::INK_FAINT);
    }
}

/// The strip's running cell's counter, short enough for a narrow cell: "1,700/2,655", "r4 · 131"
/// (search round 4, 131 decompositions), "r2/5 · 41 out" (a trust round, samples held out), the
/// pass's name otherwise.
pub fn short_counter(l: &RefitLive, trust_row: bool) -> String {
    if trust_row {
        return match l.trust {
            Some(t) if t.round > 0 => format!("r{}/{} · {} out", t.round, t.max, words::thousands(t.held as u64)),
            Some(_) => "basis retry".into(),
            None => String::new(),
        };
    }
    match l.stage {
        LiveStage::Gather | LiveStage::Check => format!("{}/{}", words::thousands(l.done as u64), words::thousands(l.total as u64)),
        LiveStage::Solve => format!("r{} · {}", l.round.max(1), l.decompositions),
        LiveStage::Starting => String::new(),
        LiveStage::Other(_) => words::refit_stage_name(l),
    }
}

/// Activity's compact strip for a running refit (module doc): a cell per stage — its time spent,
/// the running one lit with its short counter ([`short_counter`]; the card's count line has the
/// words) — and the checked samples' ratios as a thin line.
pub fn strip(ui: &mut Ui, v: &RefitView, ratios: &[f32]) {
    let l = &v.live;
    let rows: Vec<_> = stage_rows(l).into_iter().map(|(short, long, slot, on, _)| (short, long, slot, on, if on { short_counter(l, short == "Trust") } else { String::new() })).collect();
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 34.0), Sense::hover());
    super::probe_rect("refit-strip", rect);
    let p = ui.painter_at(rect);
    let w = rect.width() / rows.len().max(1) as f32;
    for (k, (short, _, slot, on, count)) in rows.iter().enumerate() {
        let cellr = Rect::from_min_size(pos2(rect.left() + k as f32 * w, rect.top()), vec2(w - 3.0, 26.0));
        let spent = slot.map(|s| l.spent_s[s]).filter(|&s| s > 0.0);
        p.rect_filled(cellr, 3.0, if *on { theme::ACCENT_BG } else { theme::PANEL_2 });
        let ink = if *on { theme::ACCENT_INK } else if spent.is_some() { theme::INK_DIM } else { theme::INK_FAINT };
        p.text(cellr.left_top() + vec2(5.0, 2.0), Align2::LEFT_TOP, *short, FontId::proportional(10.5), ink);
        let right = if *on && !count.is_empty() { count.clone() } else { spent.map_or("—".to_string(), mmss) };
        let g = p.layout_job({
            let mut job = egui::text::LayoutJob::single_section(right, egui::TextFormat { font_id: FontId::monospace(9.5), color: ink, ..Default::default() });
            job.wrap = egui::text::TextWrapping { max_width: (w - 12.0).max(10.0), max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
            job
        });
        p.galley(cellr.left_top() + vec2(5.0, 14.0), g, ink);
    }
    // the last check's ratios, a pixel column per block
    if !ratios.is_empty() {
        let line = Rect::from_min_max(pos2(rect.left(), rect.bottom() - 5.0), pos2(rect.right() - 3.0, rect.bottom() - 1.0));
        let cols = (line.width().floor() as usize).max(1);
        let per = ratios.len().div_ceil(cols).max(1);
        let n = ratios.len().div_ceil(per);
        let dx = line.width() / n as f32;
        for b in 0..n {
            let c = cell_colour(cell(b, per, ratios.len(), Fill::Ratios(ratios)));
            p.rect_filled(Rect::from_min_size(pos2(line.left() + b as f32 * dx, line.top()), vec2(dx.max(1.0), line.height())), 0.0, c);
        }
    }
    resp.on_hover_text(words::REFIT_ETA_TIP);
}

/// A made-up refit at `stage` (`gather|mix|solve|check|trust`) for the hidden `--screenshot
/// --ui "refit=<stage>"` key and the tests: 2,655 samples of a channel collection (Q′ 4,140), 40 s
/// in, the times of the measured channel refit (plan table). Display only.
pub fn demo(stage: &str) -> Option<(RefitView, Vec<f32>)> {
    use sim::refit_live::{MIX_SLOT, STAGES};
    let n = 2655u32;
    let mut live = RefitLive { q: 4140, decomposition_s: f64::NAN, ..Default::default() };
    let mut spent = [0.0; STAGES];
    let mut ratios: Vec<f32> = Vec::new();
    let ratio = |k: usize| -> f32 {
        // a deterministic spread: most well within, a few beyond
        let x = ((k * 7919) % 1000) as f32 / 1000.0;
        if k % 211 == 17 { 1.6 + x } else { 0.05 + 0.7 * x * x }
    };
    match stage {
        "gather" => {
            live.stage = LiveStage::Gather;
            (live.done, live.total, live.stage_s) = (1536, n, 0.3);
            spent[0] = 0.1;
        }
        "mix" => {
            live.stage = LiveStage::Solve;
            live.mix = true;
            (live.round, live.decompositions, live.q, live.decomposition_s, live.since_decomposition_s, live.stage_s) = (4, 131, 1230, 0.24, 0.1, 31.0);
            spent[0] = 0.1;
            spent[MIX_SLOT] = 31.5;
        }
        "solve" => {
            live.stage = LiveStage::Solve;
            (live.round, live.decompositions, live.decomposition_s, live.since_decomposition_s, live.stage_s) = (3, 5, 0.64, 0.4, 3.2);
            spent = [0.1, 0.6, 3.2, 0.0, 1.1, 53.2];
        }
        "check" | "trust" => {
            live.stage = LiveStage::Check;
            (live.done, live.total, live.stage_s) = (1700, n, 2.7);
            ratios = (0..n as usize).map(|k| if k < 1700 { ratio(k) } else { f32::NAN }).collect();
            live.beyond = ratios.iter().filter(|r| **r > 1.0).count() as u32;
            spent = [0.1, 0.6, 16.7, 2.7, 3.4, 53.2];
            if stage == "trust" {
                live.trust = Some(sim::TrustRound { round: 2, max: 5, held: 41 });
                spent[2] += 21.0;
            }
        }
        _ => return None,
    }
    live.spent_s = spent;
    let elapsed_s = spent.iter().sum::<f64>();
    let v = RefitView { elapsed_s, samples: n, species: 5, background: true, stages: sim::RefitStages::default(), estimate: None, eta: sim::RefitEta::Estimating, live };
    Some((v, ratios))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect::from_min_max(pos2(x0, y0), pos2(x1, y1))
    }

    #[test]
    fn the_free_rect_keeps_clear_of_the_zones_overlays() {
        let view = r(0.0, 0.0, 1000.0, 700.0);
        assert_eq!(free_rect(view, &[]), view.shrink(MARGIN));
        // a chip top-left, the capture buttons top-right, the view switch bottom-left, a tall atom
        // card on the right: cut on the sides that keep the most room
        let chip = r(12.0, 12.0, 300.0, 70.0);
        let capture = r(900.0, 12.0, 988.0, 40.0);
        let switch = r(14.0, 640.0, 260.0, 686.0);
        let card = r(760.0, 60.0, 988.0, 600.0);
        let taken = [chip, capture, switch, card];
        let free = free_rect(view, &taken);
        for t in taken {
            assert!(!free.intersects(t.expand(GAP - 0.01)), "{free:?} overlaps {t:?}");
        }
        assert!(free.width() > 500.0 && free.height() > 500.0, "{free:?}");
        let pic = picture_rect(view, &taken).unwrap();
        assert!(free.contains_rect(pic) && pic.width() <= MAX_SIZE.x && pic.height() <= MAX_SIZE.y, "{pic:?} in {free:?}");
        // no room: no picture
        assert_eq!(picture_rect(r(0.0, 0.0, 300.0, 200.0), &[]), None);
        let crowded = [r(0.0, 0.0, 1000.0, 300.0), r(0.0, 420.0, 1000.0, 700.0)];
        assert_eq!(picture_rect(view, &crowded), None, "a 120 pt band is too low");
    }

    #[test]
    fn blocks_wrap_to_the_room_and_group_samples_past_it() {
        let a = vec2(400.0, 200.0);
        let b = blocks(100, a);
        assert_eq!((b.per, b.count, b.side), (1, 100, BLOCK_MAX), "a hundred fit at the largest side");
        let b = blocks(2655, a);
        assert_eq!(b.per, 1, "{b:?}");
        assert!(b.side >= BLOCK_MIN && b.side < BLOCK_MAX);
        let rows = b.count.div_ceil(b.cols);
        assert!(b.cols as f32 * (b.side + BLOCK_GAP) - BLOCK_GAP <= a.x && rows as f32 * (b.side + BLOCK_GAP) - BLOCK_GAP <= a.y, "{b:?}");
        // more than fit at the smallest side: one block per k
        let b = blocks(50_000, a);
        assert_eq!(b.side, BLOCK_MIN);
        assert!(b.per > 1 && b.count * b.per >= 50_000 && (b.count - 1) * b.per < 50_000, "{b:?}");
        assert!(b.count <= b.cols * ((a.y + BLOCK_GAP) / (BLOCK_MIN + BLOCK_GAP)).floor() as usize);
        assert_eq!(blocks(0, a).count, 0);
    }

    #[test]
    fn a_block_is_gathered_checked_within_or_beyond() {
        let counted = Fill::Counted { done: 5 };
        assert_eq!((cell(1, 2, 10, counted), cell(2, 2, 10, counted), cell(3, 2, 10, counted)), (Cell::Done, Cell::Waiting, Cell::Waiting), "a block is done once all its samples are");
        let ratios = [0.2, f32::NAN, 1.5, 0.9, f32::INFINITY, f32::NAN, f32::NAN];
        let f = Fill::Ratios(&ratios);
        assert_eq!(cell(0, 2, 7, f), Cell::Within, "one checked, within; one not yet");
        assert_eq!(cell(1, 2, 7, f), Cell::Beyond);
        assert_eq!(cell(2, 2, 7, f), Cell::Beyond, "undefined counts as beyond");
        assert_eq!(cell(3, 2, 7, f), Cell::Waiting, "the last, short block: not checked");
        assert_eq!(cell(0, 1, 7, f), Cell::Within);
        assert_eq!(cell(1, 1, 7, f), Cell::Waiting);
    }

    #[test]
    fn the_blocks_follow_the_stage() {
        let (v, ratios) = demo("gather").unwrap();
        assert!(matches!(fill_of(&v, &ratios), (2655, Fill::Counted { done: 1536 })));
        let (v, ratios) = demo("check").unwrap();
        assert!(matches!(fill_of(&v, &ratios), (2655, Fill::Ratios(r)) if r.len() == 2655));
        let (v, _) = demo("solve").unwrap();
        assert!(matches!(fill_of(&v, &[]), (2655, Fill::Counted { done: 2655 })), "past the gathering: every sample in");
        let starting = RefitView { live: RefitLive::default(), ..v };
        assert!(matches!(fill_of(&starting, &[]), (2655, Fill::Counted { done: 0 })));
        // the last check's ratios stay up through the passes after it
        let other = RefitView { live: RefitLive { stage: LiveStage::Other(sim::OtherPass::ActiveSet), ..v.live }, ..v };
        assert!(matches!(fill_of(&other, &[0.5, 2.0]), (2, Fill::Ratios(_))));
        assert!(demo("nonsense").is_none());
    }

    #[test]
    fn the_stage_rows_show_the_mix_and_trust_only_when_they_ran() {
        let names = |s: &str| stage_rows(&demo(s).unwrap().0.live).iter().map(|r| (r.0, r.3)).collect::<Vec<_>>();
        assert_eq!(names("gather"), vec![("Gather", true), ("Solve", false), ("Check", false), ("Other", false)]);
        assert_eq!(names("mix"), vec![("Gather", false), ("Mix W", true), ("Solve", false), ("Check", false), ("Other", false)]);
        assert_eq!(names("solve"), vec![("Gather", false), ("Mix W", false), ("Solve", true), ("Check", false), ("Other", false)]);
        assert_eq!(names("trust"), vec![("Gather", false), ("Mix W", false), ("Solve", false), ("Check", true), ("Other", false), ("Trust", true)]);
        let rows = stage_rows(&demo("mix").unwrap().0.live);
        assert_eq!(rows[1].4, "solving · round 4 · 131 decompositions of 1,230 × 1,230 · 0.24 s each");
        // the strip's short counters
        assert_eq!(short_counter(&demo("mix").unwrap().0.live, false), "r4 · 131");
        assert_eq!(short_counter(&demo("check").unwrap().0.live, false), "1,700/2,655");
        assert_eq!(short_counter(&demo("trust").unwrap().0.live, true), "r2/5 · 41 out");
        assert_eq!(short_counter(&demo("check").unwrap().0.live, true), "");
    }
}
