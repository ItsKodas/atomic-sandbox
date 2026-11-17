//! The Knowledge panel's **"Three-atom arrangements"** section (spec 2026-11-17 §6.5; Phase A
//! Task 6): which angles the model was taught, per triple type a–c–b (`sim::knowledge::TripleMap`,
//! shipped in `ActivityDelta::triples` and kept in `ActivityState::triples`).
//!
//! - **The fold** sits under the card of the pick, closed by default (`Sections::triples`, kept in
//!   the profile). With the panel shown and the fold open or a type picked (its card is live), the
//!   UI sends `Command::SetTripleView(true)` and the worker builds the map on the knowledge map's
//!   schedule; otherwise `false`, and the map stops updating ([`wanted`], sent by `ui::show`
//!   beside `SetActivityViews`).
//! - **The picker** lists the types the fitted or new samples hold ([`picker_rows`]): most gaps
//!   first, then fewest arrangements, then by name ([`order`]); a dot marks the types the open
//!   scene has ([`scene_types`]). A click picks the type ([`Pick::Triple`]); with nothing picked
//!   the section shows [`shown_type`].
//! - **The card head** ([`card_head`]): the type, its sentence ([`head_sentence`]) and the model's
//!   per-key verdict behind it ([`verdict_line`]), whose click reveals the Model tab's Learned
//!   fold. A picked type's head is the detail area's card ([`detail`]); otherwise it heads the
//!   section.
//! - **Entry links**: the element card's "Arrangements around O" line ([`element_links`]) and the
//!   pair card's "Show arrangements around H" ([`pair_link`]) when its sentence names a short
//!   many-body key.
//! - **Empty states** (Review Focus 4): no map yet, no type, no three-atom terms in the basis, and
//!   a picked type the map no longer holds each read as one plain sentence.
//!
//! Display state only (`KnowledgeView::triple`, the pick, the fold): nothing here emits an
//! `Action`. Distances in Bohr.
use super::{new_since, sym, KnowledgeView, Pick};
use crate::scene::SceneNucleus;
use crate::sim::Backend;
use crate::ui::{activity::words::thousands, dock, keyed, probe, probe_rect, theme, DockTab, PanelInput, UiState, AMBER, GREEN};
use egui::{pos2, vec2, FontId, RichText, Sense, Stroke, Ui};
use sim::knowledge::{each_triple, summary, TripleKnowledge, TripleMap, TripleType};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Whether the worker should build the three-atom map: the Knowledge panel is shown and the
/// section is open or a picked type's card is (review I1: a picked card closed under its fold
/// must not freeze).
pub fn wanted(st: &UiState) -> bool {
    st.tab_shown(DockTab::Knowledge) && (st.knowledge_sections.triples || matches!(st.knowledge.pick, Pick::Triple(_)))
}

/// The section's title.
pub const TITLE: &str = "Three-atom arrangements";

/// The fold is open and no map has arrived yet.
pub const BUILDING: &str = "building the three-atom map…";

/// The model's basis has `body3` off: it shows in place of the classes.
pub const NO_BODY3: &str = "this model's basis has no three-atom terms: it does not learn the angle";

/// The map holds no type: no fitted or new sample has an atom with two neighbours within r_cut.
pub fn no_types(r_cut: f64) -> String {
    format!("No three-atom arrangements in the fitted samples: no atom has two neighbours within {r_cut:.1} Bohr.")
}

/// A picked type the map no longer holds (a Forget, a refit, a reach change).
pub fn stale(t: TripleType) -> String {
    format!("{} is no longer in the model's fitted or new samples", triple_name(t))
}

/// The map shown: the last one shipped, with the estimated solver only (the knowledge map's rule).
fn map_of(st: &UiState, input: &PanelInput<'_>) -> Option<Arc<TripleMap>> {
    st.activity.triples.clone().filter(|_| input.snapshot.is_none_or(|s| s.backend == Backend::Estimated))
}

/// a–c–b with `pair_name`'s rule for the two neighbours (H last, else ascending Z): "H–O–H",
/// "O–H–H" (centre H with H and O), "O–C–H".
pub fn triple_name(t: TripleType) -> String {
    let (x, y) = if t.a == 1 && t.b != 1 { (t.b, t.a) } else { (t.a, t.b) };
    format!("{}–{}–{}", sym(x), sym(t.centre), sym(y))
}

/// The probe of a picker row: `knowledge-triple-row-{a}-{c}-{b}` (Z numbers, a ≤ b).
pub fn triple_probe(t: TripleType) -> String {
    format!("knowledge-triple-row-{}-{}-{}", t.a, t.centre, t.b)
}

/// The probe of an element card's link to a type: `knowledge-triple-link-{a}-{c}-{b}`.
fn link_probe(t: TripleType) -> String {
    format!("knowledge-triple-link-{}-{}-{}", t.a, t.centre, t.b)
}

/// The types the open scene has: every atom a centre, every pair of its neighbours within `r_cut`
/// (`each_triple`, the map's own definition).
pub fn scene_types(nuclei: &[SceneNucleus], r_cut: f64) -> BTreeSet<TripleType> {
    let z: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
    let pos: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
    let mut out = BTreeSet::new();
    each_triple(&z, &pos, r_cut, |_| true, |t| {
        out.insert(t.t);
    });
    out
}

/// One row of the picker.
#[derive(Clone, Debug, PartialEq)]
pub struct PickerRow {
    pub t: TripleType,
    /// The cells in reach that are not taught (`sim::knowledge::summary`).
    pub gaps: usize,
    /// Fitted arrangements.
    pub triples: u64,
    /// Fitted samples holding it.
    pub samples: usize,
    /// New samples (kept since the last refit) holding it.
    pub pending: usize,
    /// The open scene has it (the row's dot).
    pub in_scene: bool,
}

/// The picker's rows: the types with fitted or new samples, in [`order`].
pub fn picker_rows(map: &TripleMap, scene: &BTreeSet<TripleType>) -> Vec<PickerRow> {
    let mut rows: Vec<PickerRow> = map
        .types
        .iter()
        .filter(|k| k.samples > 0 || k.pending_samples > 0)
        .map(|k| PickerRow { t: k.t, gaps: summary(map, k).gaps, triples: k.triples, samples: k.samples, pending: k.pending_samples, in_scene: scene.contains(&k.t) })
        .collect();
    order(&mut rows);
    rows
}

/// Most gaps first, then fewest fitted arrangements, then by name.
pub fn order(rows: &mut [PickerRow]) {
    rows.sort_by(|a, b| b.gaps.cmp(&a.gaps).then(a.triples.cmp(&b.triples)).then_with(|| triple_name(a.t).cmp(&triple_name(b.t))));
}

/// "H–O–H · 14 gaps · 2,140 arrangements in 610 samples · +12 new" (the last part only with new
/// samples).
pub fn row_text(r: &PickerRow) -> String {
    let new = if r.pending > 0 { format!(" · +{} new", r.pending) } else { String::new() };
    format!("{} · {} gaps · {} arrangements in {} samples{new}", triple_name(r.t), r.gaps, thousands(r.triples), thousands(r.samples as u64))
}

/// The type the section shows: the picked one while the map holds it; else the most-gaps type
/// centred on an entry link's element (`want_centre`); else the open scene's most-gaps type; else
/// the picker's first row. `None` only for a map without a type.
pub fn shown_type(view: &KnowledgeView, map: &TripleMap, scene: &BTreeSet<TripleType>) -> Option<TripleType> {
    if let Pick::Triple(t) = view.pick {
        if map.get(t).is_some_and(|k| k.samples > 0 || k.pending_samples > 0) {
            return Some(t);
        }
    }
    let rows = picker_rows(map, scene);
    let first = |f: &dyn Fn(&PickerRow) -> bool| rows.iter().find(|r| f(r)).map(|r| r.t);
    view.triple.want_centre.and_then(|c| first(&|r| r.t.centre == c)).or_else(|| first(&|r| r.in_scene)).or_else(|| rows.first().map(|r| r.t))
}

/// The card's sentence: "2,140 arrangements of H–O–H in 610 samples (13 cells taught, 9 thin, 14
/// missing within reach) · +12 new since the last refit." Without three-atom terms in the basis
/// the classes are left out ([`NO_BODY3`] says why).
pub fn head_sentence(map: &TripleMap, tk: &TripleKnowledge) -> String {
    let new = new_since(tk.pending_samples);
    let head = format!("{} arrangements of {} in {} samples", thousands(tk.triples), triple_name(tk.t), thousands(tk.samples as u64));
    if !map.basis.body3 {
        return format!("{head}{new}.");
    }
    let s = summary(map, tk);
    format!("{head} ({} cells taught, {} thin, {} missing within reach){new}.", s.taught, s.thin, s.missing)
}

/// The Model status's per-key verdicts behind a type: the keys centred on its centre whose
/// neighbours include both its neighbour species (the pair card's `keys_of` pattern), unlearned
/// first, then the many-body class's before the pair class's.
fn keys_of<'a>(input: &'a PanelInput<'_>, t: TripleType) -> Vec<(&'a str, &'a sim::SpeciesVerdict)> {
    let Some(m) = input.model() else { return Vec::new() };
    let mut v: Vec<(&str, &sim::SpeciesVerdict)> = m
        .class_loo
        .iter()
        .flat_map(|c| c.species.iter().map(move |s| (c.class, s)))
        .filter(|(_, s)| s.z == t.centre && s.neighbours.contains(&t.a) && s.neighbours.contains(&t.b))
        .collect();
    v.sort_by_key(|(class, s)| (s.trusted, *class == "pair"));
    v
}

/// The verdict line: the first key (unlearned first) worded — "the model's many-body key around O,
/// with H: trusted", or "…: 276 of 322 samples — the per-environment sample floor, not this map" —
/// and " (+ k more)" for the rest. `None` without a key.
pub fn verdict_line(keys: &[(&str, &sim::SpeciesVerdict)]) -> Option<String> {
    let mut keys = keys.to_vec();
    keys.sort_by_key(|(_, s)| s.trusted);
    let (class, s) = keys.first()?;
    let state = match s.refusal {
        _ if s.trusted => "trusted".to_string(),
        Some((sim::ClassKey::Samples, n, q)) => format!("{n:.0} of {q:.0} samples — the per-environment sample floor, not this map"),
        _ => "not yet trusted — its error checks, not this map".to_string(),
    };
    let more = if keys.len() > 1 { format!(" (+ {} more)", keys.len() - 1) } else { String::new() };
    Some(format!("the model's {class} key {}: {state}{more}", s.words()))
}

/// The card head: the type, the sentence, the basis note when it has no three-atom terms, and the
/// verdict line (a click shows the Model tab's Learned fold, as "Forget these…" reveals Suspects).
pub fn card_head(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, map: &TripleMap, tk: &TripleKnowledge) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(triple_name(tk.t)).strong().size(14.0));
        theme::pill(ui, "three-atom", theme::INK_DIM);
        ui.label(RichText::new(format!("arrangements around {}", sym(tk.t.centre))).size(10.5).color(theme::INK_FAINT));
    });
    let s = ui.add(egui::Label::new(RichText::new(head_sentence(map, tk)).size(12.0).color(theme::INK)).wrap());
    probe("knowledge-triple-sentence", &s);
    if !map.basis.body3 {
        ui.add(egui::Label::new(RichText::new(NO_BODY3).size(11.5).color(AMBER)).wrap());
    }
    let keys = keys_of(input, tk.t);
    if let Some(line) = verdict_line(&keys) {
        let colour = if keys.first().is_some_and(|(_, s)| s.trusted) { GREEN } else { AMBER };
        let r = ui
            .add(egui::Label::new(RichText::new(line).size(11.5).color(colour).underline()).sense(Sense::click()).wrap())
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("the model's own per-environment verdict for atoms of this centre with both these neighbours: its trust rule, not these counts. Click for the Model tab's learned classes");
        probe("knowledge-triple-verdict", &r);
        if r.clicked() {
            st.open_tab(DockTab::Model);
            st.model_sections.set(dock::model::Section::Learned, true);
            st.search.reveal = Some(("d-model-learned", std::time::Instant::now(), false));
        }
    }
}

/// The detail area's card for a picked type ([`Pick::Triple`]): its head, or a plain note.
pub fn detail(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, t: TripleType) {
    let Some(map) = map_of(st, input) else {
        theme::note(ui, BUILDING);
        return;
    };
    match map.get(t).filter(|k| k.samples > 0 || k.pending_samples > 0) {
        Some(tk) => super::card(ui, "knowledge-triple-head", |ui| card_head(ui, st, input, &map, tk)),
        None => {
            ui.add(egui::Label::new(RichText::new(stale(t)).size(11.0).color(theme::INK_DIM)).wrap());
        }
    }
}

/// The section: the fold, then (open) the picker, the shown type's head when it is not already the
/// detail's card, and the view (Task 7).
pub fn section(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, r_cut: f64) {
    ui.add_space(8.0);
    let open = st.knowledge_sections.triples;
    let (h, _) = keyed(ui, "knowledge-triple-fold", |ui| theme::fold_header(ui, TITLE, open, Some((format!("neighbours within {r_cut:.1} Bohr"), theme::INK_FAINT)), None, false));
    probe("knowledge-triple-fold", &h);
    if h.clicked() {
        st.knowledge_sections.triples = !open;
    }
    if !st.knowledge_sections.triples {
        return;
    }
    let Some(map) = map_of(st, input) else {
        theme::note(ui, BUILDING);
        return;
    };
    let scene = scene_types(&input.scene.nuclei, map.r_cut);
    let rows = picker_rows(&map, &scene);
    if rows.is_empty() {
        ui.add(egui::Label::new(RichText::new(no_types(map.r_cut)).size(11.0).color(theme::INK_DIM)).wrap());
        return;
    }
    let shown = shown_type(&st.knowledge, &map, &scene);
    picker(ui, st, &rows, shown);
    let Some(tk) = shown.and_then(|t| map.get(t)) else { return };
    if st.knowledge.pick != Pick::Triple(tk.t) {
        ui.add_space(4.0);
        super::card(ui, "knowledge-triple-head", |ui| card_head(ui, st, input, &map, tk));
    }
    // the slice and the cloud (Tasks 7–8) draw here
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 1.0), Sense::hover());
    probe_rect("knowledge-triple-view", rect);
}

/// The picker: one row per type, the shown one outlined, a dot for the scene's types.
fn picker(ui: &mut Ui, st: &mut UiState, rows: &[PickerRow], shown: Option<TripleType>) {
    ui.add_space(2.0);
    for r in rows {
        let name = triple_probe(r.t);
        let resp = keyed(ui, &name, |ui| picker_row(ui, r, shown == Some(r.t)));
        probe(&name, &resp);
        if resp.clicked() {
            st.knowledge.set_pick(Pick::Triple(r.t));
        }
    }
}

fn picker_row(ui: &mut Ui, r: &PickerRow, on: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 20.0), Sense::click());
    let p = ui.painter();
    if on {
        p.rect_filled(rect, 4.0, theme::ACCENT.gamma_multiply(0.18));
        p.rect_stroke(rect, 4.0, Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.7)), egui::StrokeKind::Inside);
    } else if resp.hovered() {
        p.rect_filled(rect, 4.0, theme::PANEL_2);
    }
    if r.in_scene {
        p.circle_filled(pos2(rect.left() + 8.0, rect.center().y), 3.0, theme::INK);
    }
    let text = row_text(r);
    let name = triple_name(r.t);
    let mut job = egui::text::LayoutJob::default();
    job.append(&name, 0.0, egui::TextFormat { font_id: FontId::monospace(11.5), color: theme::INK, ..Default::default() });
    job.append(&text[name.len()..], 0.0, egui::TextFormat { font_id: FontId::proportional(11.0), color: theme::INK_DIM, ..Default::default() });
    job.wrap = egui::text::TextWrapping { max_width: (rect.width() - 22.0).max(10.0), max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
    let g = p.layout_job(job);
    p.galley(pos2(rect.left() + 16.0, rect.center().y - g.size().y / 2.0), g, theme::INK);
    let scene = if r.in_scene { " · the open scene has it" } else { "" };
    resp.on_hover_text(format!("{text}{scene} — click for its card"))
}

/// The pair card's link to the arrangements around `centre` (its sentence names a short
/// many-body key there): the section opens on that centre's type with the most gaps.
pub fn pair_link(ui: &mut Ui, st: &mut UiState, centre: u8) {
    let r = keyed(ui, "knowledge-triple-link-pair", |ui| ui.link(RichText::new(format!("Show arrangements around {}", sym(centre))).size(11.5)))
        .on_hover_text(format!("open Three-atom arrangements below on the type centred on {} with the most gaps", sym(centre)));
    probe("knowledge-triple-link-pair", &r);
    if r.clicked() {
        st.knowledge.triple.want_centre = Some(centre);
        st.knowledge_sections.triples = true;
    }
}

/// The element card's "Arrangements around O" line: one link per type centred on `z` the map holds
/// (picks it and opens the section), or before any map one link that opens the section on that
/// centre.
pub fn element_links(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, z: u8) {
    let map = map_of(st, input);
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        ui.label(RichText::new(format!("Arrangements around {}:", sym(z))).size(11.5).color(theme::INK_DIM));
        let Some(map) = map else {
            let r = keyed(ui, "knowledge-triple-link-element", |ui| ui.link(RichText::new("Show three-atom arrangements").size(11.5)))
                .on_hover_text(format!("open Three-atom arrangements below (the worker builds the map) on the type centred on {} with the most gaps", sym(z)));
            probe("knowledge-triple-link-element", &r);
            if r.clicked() {
                st.knowledge.triple.want_centre = Some(z);
                st.knowledge_sections.triples = true;
            }
            return;
        };
        let mine: Vec<TripleType> = map.types.iter().filter(|k| k.t.centre == z && (k.samples > 0 || k.pending_samples > 0)).map(|k| k.t).collect();
        if mine.is_empty() {
            ui.label(RichText::new("none in the model's samples").size(11.0).color(theme::INK_FAINT));
        }
        for t in mine {
            let name = link_probe(t);
            let r = keyed(ui, &name, |ui| ui.link(RichText::new(triple_name(t)).monospace().size(11.5))).on_hover_text("its card, and Three-atom arrangements below");
            probe(&name, &r);
            if r.clicked() {
                st.knowledge.set_pick(Pick::Triple(t));
                st.knowledge_sections.triples = true;
            }
        }
    });
}
