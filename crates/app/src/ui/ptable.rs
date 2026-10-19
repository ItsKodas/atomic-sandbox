//! The periodic-table picker (M6 Task 10, user request 2026-10-14): a popover from the toolbar's
//! element group with the standard 18-column layout and the two f-block rows below it, searchable
//! by symbol, name (where `units::elements::name` has one) or Z. Z ≤ 18 can be placed (M7 Task 12: Z ≤ 10
//! all-electron or pseudised, Z 11–18 on a pseudopotential generated on first use — or refused with its
//! reason, see [`tip`]); heavier elements are shown greyed, "needs pseudopotentials (planned)". Picking an enabled element places its most abundant isotope and offers the listed
//! isotopes of that Z (`units::isotopes`) with their masses.
//!
//! Display data only (rule 1): the cell positions are computed from Z and the period boundaries
//! (the noble gases' Z), the symbols and names come from `units::elements`.
use super::{theme, Action, UiState};
use egui::{Align2, Color32, FontId, Id, Rect, RichText, Sense, Stroke, Ui, Vec2};
use units::elements::{name, symbol, Z_MAX};

/// The last Z of each period (the noble gases): the only input of the layout.
pub const PERIOD_ENDS: [u8; 7] = [2, 10, 18, 36, 54, 86, 118];
/// The heaviest element that can be placed today: the isotope table's range (`units::isotopes`), whose
/// pseudopotentials the generator makes on first use (a refused species is placeable but runs all-electron
/// under Cores only and is an error under All — [`tip`] says so).
pub const Z_ENABLED: u8 = 18;
/// Why the rest are greyed.
pub const GREYED_TIP: &str = "needs pseudopotentials (planned)";
/// Rows of the two f-block rows (row 8 is the gap between them and the main table).
pub const F_ROWS: [u8; 2] = [9, 10];

/// Whether Z can be placed now.
pub fn enabled(z: u8) -> bool {
    (1..=Z_ENABLED).contains(&z)
}

/// The cell of Z in the standard layout, `(row, column)`, 1-based: rows 1–7 are the periods,
/// columns 1–18 the groups; the lanthanides (Z 57–71) and actinides (89–103) sit in rows 9 and 10,
/// columns 3–17. Computed from the period boundaries: a 2-element period fills columns 1 and 18,
/// an 8-element one columns 1–2 and 13–18, an 18-element one all 18, a 32-element one columns 1–2,
/// its f-block row, then columns 4–18. `None` outside 1..=118.
pub fn position(z: u8) -> Option<(u8, u8)> {
    if z == 0 || z > Z_MAX {
        return None;
    }
    let p = PERIOD_ENDS.iter().position(|&e| z <= e)?;
    let start = if p == 0 { 1 } else { PERIOD_ENDS[p - 1] + 1 };
    let len = PERIOD_ENDS[p] - start + 1;
    let k = z - start;
    let row = p as u8 + 1;
    Some(match len {
        2 => (row, if k == 0 { 1 } else { 18 }),
        8 => (row, if k < 2 { k + 1 } else { k + 11 }),
        18 => (row, k + 1),
        // 32: s block, the 15-wide f row (row 6 → 9, row 7 → 10), then d and p from column 4
        _ if k < 2 => (row, k + 1),
        _ if k < 17 => (F_ROWS[(row - 6) as usize], k + 1),
        _ => (row, k - 13),
    })
}

/// Whether Z answers the search: a number is Z itself; text is a case-insensitive prefix of the
/// symbol or of the name (where one is listed). An empty query matches everything.
pub fn matches(query: &str, z: u8) -> bool {
    let q = query.trim();
    if q.is_empty() {
        return true;
    }
    if q.chars().all(|c| c.is_ascii_digit()) {
        return q.parse::<u16>().is_ok_and(|n| n == z as u16);
    }
    let q = q.to_lowercase();
    symbol(z).to_lowercase().starts_with(&q) || name(z).is_some_and(|n| n.starts_with(&q))
}

/// The matches of a query in Z order (the Enter key picks the first enabled one).
pub fn search(query: &str) -> Vec<u8> {
    (1..=Z_MAX).filter(|&z| matches(query, z)).collect()
}

/// The partition rule's verdict on Z (`qm::radial::has_core`: the radial all-electron atom and §4.2, a few
/// tens of milliseconds), memoised per Z. Refusals that only the full generation can reach (Al's T5) are read from
/// the providers' cache instead, once something has generated them.
fn partition_memo() -> &'static std::sync::Mutex<std::collections::HashMap<u8, Result<bool, qm::radial::Refusal>>> {
    static MEMO: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u8, Result<bool, qm::radial::Refusal>>>> = std::sync::OnceLock::new();
    MEMO.get_or_init(Default::default)
}

/// Computes the partition verdicts of Z 11..=Z_ENABLED into the memo (idempotent). The popover primes it on a
/// thread when it opens ([`prime`]), so the render thread never waits for it; the tests call it directly.
pub fn prime_partition() {
    for z in 11..=Z_ENABLED {
        if partition_memo().lock().is_ok_and(|m| m.contains_key(&z)) {
            continue;
        }
        let verdict = qm::radial::has_core(z);
        if let Ok(mut m) = partition_memo().lock() {
            m.insert(z, verdict);
        }
    }
}

/// Starts [`prime_partition`] on a background thread, once per process.
fn prime() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("ptable-partition".into()).spawn(prime_partition);
    });
}

/// The recipe's refusal of Z when one is known without generating anything: the providers' cache (a table the
/// process generated, or its refusal) first, then the partition memo.
fn known_refusal(z: u8) -> Option<qm::radial::Refusal> {
    if let Some(Err(r)) = providers::pseudo::cached_outcome(z) {
        return Some(r);
    }
    match partition_memo().try_lock().ok()?.get(&z)? {
        Err(r) => Some(r.clone()),
        Ok(_) => None,
    }
}

/// The refusal in one clause: the reason, then where it leaves the user (display words only).
fn refusal_reason(r: &qm::radial::Refusal) -> String {
    use qm::radial::Refusal;
    match r {
        Refusal::TwoStatesSameL { l, .. } => {
            format!("two valence {} states — needs the ONCV stage", ["s", "p", "d", "f"].get(*l as usize).copied().unwrap_or("?"))
        }
        Refusal::NoLadderEntryPasses { .. } => {
            "no entry of the recipe's softness ladder passes its transferability gates (T5) — needs the ONCV stage".to_string()
        }
        other => other.to_string(),
    }
}

/// Zσ at the default grid (h = σ = [`crate::quality::DEFAULT_H`]) as "1.7": computed as z·h in hundredths of a
/// Bohr, rounded half up, so 11 × 0.15 reads 1.7 and not the float's 1.6.
fn z_sigma(z: u8) -> String {
    let hundredths = (crate::quality::DEFAULT_H * 100.0).round() as u32;
    let tenths = (z as u32 * hundredths + 5) / 10;
    format!("{}.{}", tenths / 10, tenths % 10)
}

/// A cell's tooltip: what clicking does, or why it is greyed. For Z > 10 it states what the species gets from
/// the pseudopotential recipe — a refusal (known from the partition rule or from a generation the process
/// already ran) with what the all-electron fallback costs, else "pseudopotential (generated on first use)".
pub fn tip(z: u8) -> String {
    let who = match name(z) {
        Some(n) => format!("{n} ({}, Z = {z})", symbol(z)),
        None => format!("{} (Z = {z})", symbol(z)),
    };
    if !enabled(z) {
        return format!("{who}: {GREYED_TIP}");
    }
    let place = "click to place it (its most abundant isotope; the isotopes are listed below)";
    if z <= 10 {
        return format!("{who}: {place}");
    }
    match known_refusal(z) {
        Some(r) => format!(
            "{who}: {place}. refused: {}; under Cores only it falls back to all-electron at σ = h, where Zσ ≈ {} even at h = {}: qualitatively wrong, as O was before M7; under All species it is an error",
            refusal_reason(&r),
            z_sigma(z),
            crate::quality::DEFAULT_H
        ),
        None => format!("{who}: {place}. pseudopotential (generated on first use)"),
    }
}

/// The popover's id (`ui::open_ptable` opens it).
pub fn popup_id() -> Id {
    Id::new("ptable-popup")
}

/// Cell size and gap (points).
const CELL: Vec2 = Vec2::new(28.0, 30.0);
const GAP: f32 = 2.0;
/// Height of the gap row between the main table and the f-block rows.
const ROW_GAP: f32 = 8.0;

/// The popover's contents: search, table, the chosen element's isotopes, the note.
pub fn contents(ui: &mut Ui, st: &mut UiState, acts: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Periodic table").strong());
        let r = ui.add(egui::TextEdit::singleline(&mut st.ptable_query).hint_text("symbol, name or Z").desired_width(150.0));
        super::probe("ptable-search", &r);
        if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let Some(z) = search(&st.ptable_query).into_iter().find(|&z| enabled(z)) {
                acts.push(Action::Element(z));
            }
        }
    });
    let query = st.ptable_query.clone();
    let selected = st.element;
    if let Some(z) = grid(ui, &query, Some(selected), "ptable") {
        acts.push(Action::Element(z));
    }
    isotope_row(ui, st, acts);
    theme::note(ui, format!("Z 1–{Z_ENABLED} can be placed; heavier elements need the next pseudopotential stage (ONCV, semicore). Greyed (Z > {Z_ENABLED}): {GREYED_TIP}."));
}

/// The table's cells (no search row, no isotopes): the enabled element clicked, if any. `salt`
/// keeps two tables apart (the toolbar's `"ptable"`, the molecule builder's); `selected` is
/// highlighted, `query` dims the elements it does not match.
pub fn grid(ui: &mut Ui, query: &str, selected: Option<u8>, salt: &str) -> Option<u8> {
    prime();
    let cols = 18.0;
    let size = Vec2::new(cols * CELL.x + (cols - 1.0) * GAP, 9.0 * (CELL.y + GAP) + ROW_GAP);
    let (area, _) = ui.allocate_exact_size(size, Sense::hover());
    let at = |row: u8, col: u8| {
        let y = if row >= F_ROWS[0] { (row as f32 - 2.0) * (CELL.y + GAP) + ROW_GAP } else { (row as f32 - 1.0) * (CELL.y + GAP) };
        Rect::from_min_size(area.min + Vec2::new((col as f32 - 1.0) * (CELL.x + GAP), y), CELL)
    };
    // the f-block's place in periods 6 and 7
    for (row, text) in [(6, "57–71"), (7, "89–103")] {
        let r = at(row, 3);
        ui.painter().rect_stroke(r, 4.0, Stroke::new(1.0, theme::LINE_SOFT), egui::StrokeKind::Inside);
        ui.painter().text(r.center(), Align2::CENTER_CENTER, text, FontId::monospace(7.0), theme::INK_FAINT);
    }
    let mut picked = None;
    for z in 1..=Z_MAX {
        let Some((row, col)) = position(z) else { continue };
        let rect = at(row, col);
        let r = ui.interact(rect, Id::new((salt, z)), if enabled(z) { Sense::click() } else { Sense::hover() });
        let hit = matches(query, z);
        let is_selected = enabled(z) && selected == Some(z);
        cell(ui, rect, z, is_selected, r.hovered(), hit && !query.trim().is_empty(), hit);
        super::probe(&format!("{salt}-{z}"), &r);
        if r.on_hover_text(tip(z)).clicked() && enabled(z) {
            picked = Some(z);
        }
    }
    picked
}

/// One painted cell: symbol, Z, the display colour as a stripe for the enabled ones.
fn cell(ui: &Ui, rect: Rect, z: u8, selected: bool, hovered: bool, highlit: bool, hit: bool) {
    let p = ui.painter();
    let on = enabled(z);
    let dim: f32 = if hit { 1.0 } else { 0.3 };
    let (fill, stroke) = if selected {
        (theme::ACCENT_BG, Stroke::new(1.0, theme::ACCENT))
    } else if highlit {
        (theme::PANEL_3, Stroke::new(1.0, theme::ACCENT_INK))
    } else if hovered && on {
        (theme::PANEL_3, Stroke::new(1.0, theme::LINE))
    } else if on {
        (theme::PANEL_2, Stroke::new(1.0, theme::LINE))
    } else {
        (theme::APP, Stroke::new(1.0, theme::LINE_SOFT))
    };
    if selected {
        p.rect_filled(rect.expand(2.0), 6.0, theme::GLOW.gamma_multiply(0.18));
    }
    p.rect(rect, 4.0, fill.gamma_multiply(dim.max(0.6)), Stroke::new(stroke.width, stroke.color.gamma_multiply(dim)), egui::StrokeKind::Inside);
    let ink = if on { if selected { Color32::WHITE } else { theme::INK } } else { theme::INK_FAINT };
    p.text(rect.left_top() + Vec2::new(2.5, 1.0), Align2::LEFT_TOP, z.to_string(), FontId::monospace(6.5), theme::INK_FAINT.gamma_multiply(dim));
    p.text(rect.center() + Vec2::new(0.0, 2.5), Align2::CENTER_CENTER, symbol(z), FontId::monospace(11.0), ink.gamma_multiply(dim));
    if on {
        let [r, g, b] = render::elements::colour_srgb(z);
        let c = Color32::from_rgb(r, g, b);
        let bar = Rect::from_min_max(rect.left_bottom() + Vec2::new(5.0, -3.0), rect.right_bottom() + Vec2::new(-5.0, -2.0));
        p.rect_filled(bar, 1.0, c.gamma_multiply(0.8 * dim));
    }
}

/// The chosen element's listed isotopes with their masses: the most abundant is the default.
fn isotope_row(ui: &mut Ui, st: &UiState, acts: &mut Vec<Action>) {
    let z = st.element;
    if units::isotopes_of(z).is_empty() {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(format!("isotope of {}:", name(z).unwrap_or(symbol(z)))).color(theme::INK_DIM));
        if let Some(a) = super::isotope_buttons(ui, z, st.isotope.or(units::most_abundant_a(z)), "ptable-isotope") {
            acts.push(Action::ElementIsotope(z, a));
        }
    });
}

/// The toolbar's opener and the popover anchored to it (`label` is the button text).
pub fn opener(ui: &mut Ui, st: &mut UiState, label: impl Into<egui::WidgetText>, min: Vec2, acts: &mut Vec<Action>) -> egui::Response {
    let r = ui
        .add(egui::Button::new(label).min_size(min).frame(false))
        .on_hover_text(format!("the periodic table: pick the element (Z 1–{Z_ENABLED}) and its isotope; search by symbol, name or Z"));
    super::probe("ptable-open", &r);
    egui::Popup::from_toggle_button_response(&r)
        .id(popup_id())
        .align(egui::RectAlign::TOP_START)
        .gap(6.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| contents(ui, st, acts));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard positions from the period boundaries alone: the noble gases in column 18, the
    /// p block of periods 2–3 from column 13, the f rows below, Hf back in column 4.
    #[test]
    fn positions_follow_the_period_boundaries() {
        for (z, cell) in [
            (1, (1, 1)),
            (2, (1, 18)),
            (3, (2, 1)),
            (5, (2, 13)),
            (10, (2, 18)),
            (11, (3, 1)),
            (18, (3, 18)),
            (21, (4, 3)),
            (36, (4, 18)),
            (54, (5, 18)),
            (55, (6, 1)),
            (57, (9, 3)),
            (71, (9, 17)),
            (72, (6, 4)),
            (86, (6, 18)),
            (89, (10, 3)),
            (103, (10, 17)),
            (104, (7, 4)),
            (118, (7, 18)),
        ] {
            assert_eq!(position(z), Some(cell), "Z = {z}");
        }
        assert_eq!(position(0), None);
        assert_eq!(position(119), None);
        // every cell is taken once, inside the 18 × 10 frame, row 8 empty
        let mut seen = std::collections::BTreeSet::new();
        for z in 1..=Z_MAX {
            let (r, c) = position(z).unwrap();
            assert!((1..=18).contains(&c) && (1..=10).contains(&r) && r != 8, "Z = {z} at {r},{c}");
            assert!(seen.insert((r, c)), "Z = {z} shares {r},{c}");
        }
    }

    /// Z ≤ 18 enabled; the rest greyed with the planned-pseudopotentials tooltip (M7 Task 12). Na and Mg tell their
    /// refusal and what the all-electron fallback costs; the species the recipe may accept say they are generated
    /// on first use.
    #[test]
    fn z_up_to_18_are_enabled_and_the_rest_greyed() {
        prime_partition();
        let on: Vec<u8> = (1..=Z_MAX).filter(|&z| enabled(z)).collect();
        assert_eq!(on, (1..=18).collect::<Vec<u8>>());
        assert!(enabled(18) && !enabled(0) && !enabled(19) && !enabled(118));
        assert!(tip(19).ends_with("needs pseudopotentials (planned)"), "{}", tip(19));
        assert!(tip(92).ends_with("needs pseudopotentials (planned)"), "{}", tip(92));
        assert!(tip(8).starts_with("oxygen (O, Z = 8): click to place it"), "{}", tip(8));
        assert!(!tip(8).contains("pseudopotentials"));
        // beyond the name table: the symbol and Z
        assert!(tip(79).starts_with("Au (Z = 79)"), "{}", tip(79));
        // Na and Mg: refused by the partition (two valence s states), the fallback's Zσ at h = 0.15 stated
        let na = tip(11);
        assert!(na.starts_with("sodium (Na, Z = 11): click to place it"), "{na}");
        assert!(na.contains("refused: two valence s states — needs the ONCV stage"), "{na}");
        assert!(na.contains("falls back to all-electron at σ = h, where Zσ ≈ 1.7 even at h = 0.15: qualitatively wrong, as O was before M7"), "{na}");
        let mg = tip(12);
        assert!(mg.contains("refused: two valence s states") && mg.contains("Zσ ≈ 1.8"), "{mg}");
        // the species whose refusal (Al: T5) or acceptance only a generation can tell
        for z in 14..=18 {
            assert!(tip(z).ends_with("pseudopotential (generated on first use)"), "{}", tip(z));
        }
        assert!(tip(16).starts_with("sulfur (S, Z = 16): click to place it"), "{}", tip(16));
    }

    /// Zσ at h = 0.15 is computed as z · h and rounded half up (not tabulated; a float would print 11 × 0.15 as 1.6).
    #[test]
    fn z_sigma_rounds_half_up_from_z_times_h() {
        assert_eq!([11, 12, 13, 14, 18].map(z_sigma), ["1.7", "1.8", "2.0", "2.1", "2.7"]);
    }

    /// A refusal the process already generated (the providers' cache) is told by the tip, not "generated on first use".
    #[test]
    fn a_cached_refusal_is_read_from_the_providers() {
        let reason = refusal_reason(&qm::radial::Refusal::NoLadderEntryPasses { z: 13, tried: vec![] });
        assert!(reason.contains("T5") && reason.contains("ONCV"), "{reason}");
        let two = refusal_reason(&qm::radial::Refusal::TwoStatesSameL { z: 11, l: 0, states: vec![] });
        assert_eq!(two, "two valence s states — needs the ONCV stage");
    }

    /// Search by symbol, name prefix or Z, case-insensitively.
    #[test]
    fn the_search_finds_symbols_names_and_atomic_numbers() {
        assert_eq!(search("8"), vec![8]);
        assert_eq!(search("118"), vec![118]);
        assert!(search("300").is_empty());
        assert_eq!(search("oxy"), vec![8]);
        assert_eq!(search("NE"), vec![10]);
        assert_eq!(search(" neon "), vec![10]);
        assert_eq!(search("Og"), vec![118]);
        assert_eq!(search("hel"), vec![2]);
        assert!(search("c").contains(&6) && search("c").contains(&17) && search("c").contains(&20));
        assert!(search("xyz").is_empty());
        assert_eq!(search("").len(), Z_MAX as usize);
        // the first enabled match is what Enter picks
        assert_eq!(search("h").into_iter().find(|&z| enabled(z)), Some(1));
    }
}
