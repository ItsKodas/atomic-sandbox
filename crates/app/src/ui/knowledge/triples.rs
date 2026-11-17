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
//! - **The 2D slice** (Task 7, spec §6.5 and U1; [`flat`]): x the angle at the centre in the 10
//!   cos θ bins ([`angle_labels`] at their edges), y r₁ over the rows [`extent`] crops to the
//!   drawn cells ([`drawn`]: filled, new, or missing within reach), one r₂ bin per slice (a
//!   slider; a folded type's slice s reads the stored cell {k₁, s}, [`slice_cells`]). Fill
//!   [`GREEN`] taught / [`AMBER`] thin, its opacity rising with ln(1 + n) over the type's fullest
//!   cell; missing an `INK_FAINT` outline; new (pending > 0) a dashed [`NEW_INK`] outline; the
//!   picked cell an accent outline. The open scene's arrangements of the type
//!   ([`scene_instances`], the map's own `each_triple`) are white rings on their slice, "now" by
//!   the chosen one ([`chosen_instance`]: the selected atom's, else the first), faint rings for
//!   those in other slices. Hover reads [`cell_hover`]; a click picks a drawn cell (empty space
//!   clears it) and its card shows [`cell_card_line`] and the honest-limits band
//!   ([`band_lines`], §6.7 items 1–3).
//! - **The `2D | 3D` switch** (Task 8, [`mode_switch`]; `TripleView::mode`, 2D by default): 3D
//!   draws the same cells as a rotatable voxel cloud ([`super::cloud`]) with the same hover, cell
//!   pick and card ([`cell_card`]).
//! - **Stale state** (Review Focus 1, [`sync`]): the classes and extent are computed once per
//!   (map `Arc`, type) into `TripleView::cache`; each frame a slice outside the extent is clamped
//!   into it and a picked cell past the grid or no longer drawn is dropped. Showing another type
//!   resets the slice and the cell (they named the old type's bins).
//!
//! Display state only (`KnowledgeView::triple`, the pick, the fold): nothing here emits an
//! `Action`. Distances in Bohr; degrees only in the labels.
use super::{cloud, new_since, pair_name, sym, KnowledgeView, Pick, TripleMode, TripleView, NEW_INK};
use crate::scene::SceneNucleus;
use crate::sim::Backend;
use crate::ui::{activity::words::thousands, dock, keyed, probe, probe_rect, theme, DockTab, PanelInput, UiState, AMBER, GREEN};
use egui::{pos2, vec2, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, Ui};
use sim::knowledge::{cell_class, cos_bin, cos_edges_deg, each_triple, summary, taught_count, CellClass, Triple, TripleGrid, TripleKnowledge, TripleMap, TripleSummary, TripleType, COS_BINS, TRIPLE_BIN};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
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

#[cfg(test)]
thread_local! {
    /// Tests only: the [`type_summaries`] passes and [`SceneTriples::of`] enumerations this thread
    /// ran (final review Minor 1: a frame on the same map and scene runs neither).
    pub static SUMMARY_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub static SCENE_PASSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The open scene's arrangements, by type: every atom a centre, every pair of its neighbours
/// within `r_cut` (`each_triple`, the map's own definition, so a marker sits in the cell its
/// arrangement would be counted in). Kept in `TripleView::scene` and enumerated again only when
/// the geometry (each nucleus's Z and position, bit for bit) or `r_cut` changes ([`scene_memo`]):
/// a scene is Σ C(k_i, 2) arrangements, too many to redo every frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneTriples {
    r_cut: u64,
    geometry: Vec<(u8, [u64; 3])>,
    by_type: BTreeMap<TripleType, Vec<Triple>>,
}

impl SceneTriples {
    pub fn of(nuclei: &[SceneNucleus], r_cut: f64) -> SceneTriples {
        #[cfg(test)]
        SCENE_PASSES.with(|n| n.set(n.get() + 1));
        let z: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
        let pos: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        let mut by_type: BTreeMap<TripleType, Vec<Triple>> = BTreeMap::new();
        each_triple(&z, &pos, r_cut, |_| true, |x| by_type.entry(x.t).or_default().push(x));
        let geometry = nuclei.iter().map(|n| (n.z, n.pos.map(f64::to_bits))).collect();
        SceneTriples { r_cut: r_cut.to_bits(), geometry, by_type }
    }

    /// Enumerated from this very geometry and cutoff.
    pub fn holds(&self, nuclei: &[SceneNucleus], r_cut: f64) -> bool {
        self.r_cut == r_cut.to_bits() && self.geometry.len() == nuclei.len() && self.geometry.iter().zip(nuclei).all(|(g, n)| g.0 == n.z && g.1 == n.pos.map(f64::to_bits))
    }

    /// The types the scene has.
    pub fn types(&self) -> BTreeSet<TripleType> {
        self.by_type.keys().copied().collect()
    }

    /// The scene's arrangements of type `t`.
    pub fn instances(&self, t: TripleType) -> &[Triple] {
        self.by_type.get(&t).map_or(&[], Vec::as_slice)
    }
}

/// The view's [`SceneTriples`] of this geometry and cutoff, enumerated again only when either
/// changed.
pub fn scene_memo<'a>(view: &'a mut TripleView, nuclei: &[SceneNucleus], r_cut: f64) -> &'a SceneTriples {
    if !view.scene.as_ref().is_some_and(|s| s.holds(nuclei, r_cut)) {
        view.scene = Some(SceneTriples::of(nuclei, r_cut));
    }
    view.scene.as_ref().expect("filled above")
}

/// Each type with fitted or new samples and its [`summary`], in the map's order: every cell's
/// class and reach, so computed once per map ([`SummaryCache`], spec §6.4 "UI cost").
pub fn type_summaries(map: &TripleMap) -> Vec<(TripleType, TripleSummary)> {
    #[cfg(test)]
    SUMMARY_PASSES.with(|n| n.set(n.get() + 1));
    map.types.iter().filter(|k| k.samples > 0 || k.pending_samples > 0).map(|k| (k.t, summary(map, k))).collect()
}

/// [`type_summaries`] of one map, held as `ClassCache` holds its map (`Arc::ptr_eq` cannot match
/// a later map at a reused address): the picker's gaps and the card head's counts read it, so a
/// repaint with the same map summarises nothing (final review Minor 1).
#[derive(Clone)]
pub struct SummaryCache {
    pub map: Arc<TripleMap>,
    pub sums: Vec<(TripleType, TripleSummary)>,
}

impl SummaryCache {
    /// Computed from this very map.
    pub fn holds(&self, map: &Arc<TripleMap>) -> bool {
        Arc::ptr_eq(&self.map, map)
    }

    /// Type `t`'s summary (`None`: no fitted or new samples hold it).
    pub fn get(&self, t: TripleType) -> Option<TripleSummary> {
        self.sums.iter().find(|(x, _)| *x == t).map(|(_, s)| *s)
    }
}

impl PartialEq for SummaryCache {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.map, &o.map) && self.sums == o.sums
    }
}

/// The map shown as its pointer.
impl std::fmt::Debug for SummaryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummaryCache").field("map", &Arc::as_ptr(&self.map)).field("sums", &self.sums).finish()
    }
}

/// The view's [`SummaryCache`] of `map`, summarised again only for another map.
pub fn summaries<'a>(view: &'a mut TripleView, map: &Arc<TripleMap>) -> &'a SummaryCache {
    if !view.summaries.as_ref().is_some_and(|c| c.holds(map)) {
        view.summaries = Some(SummaryCache { map: map.clone(), sums: type_summaries(map) });
    }
    view.summaries.as_ref().expect("filled above")
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

/// The picker's rows: the types with fitted or new samples (`sums`, [`type_summaries`] of `map`),
/// in [`order`].
pub fn picker_rows(map: &TripleMap, sums: &[(TripleType, TripleSummary)], scene: &BTreeSet<TripleType>) -> Vec<PickerRow> {
    let mut rows: Vec<PickerRow> = sums
        .iter()
        .filter_map(|(t, s)| map.get(*t).map(|k| PickerRow { t: k.t, gaps: s.gaps, triples: k.triples, samples: k.samples, pending: k.pending_samples, in_scene: scene.contains(&k.t) }))
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
/// the picker's first row (`rows`: [`picker_rows`] of `map`). `None` only for a map without a
/// type.
pub fn shown_type(view: &KnowledgeView, map: &TripleMap, rows: &[PickerRow]) -> Option<TripleType> {
    if let Pick::Triple(t) = view.pick {
        if map.get(t).is_some_and(|k| k.samples > 0 || k.pending_samples > 0) {
            return Some(t);
        }
    }
    let first = |f: &dyn Fn(&PickerRow) -> bool| rows.iter().find(|r| f(r)).map(|r| r.t);
    view.triple.want_centre.and_then(|c| first(&|r| r.t.centre == c)).or_else(|| first(&|r| r.in_scene)).or_else(|| rows.first().map(|r| r.t))
}

/// The card's sentence: "2,140 arrangements of H–O–H in 610 samples (13 cells taught, 9 thin, 14
/// missing within reach) · +12 new since the last refit." Without three-atom terms in the basis
/// the classes are left out ([`NO_BODY3`] says why). `s` is the type's [`summary`] (the view's
/// [`SummaryCache`]).
pub fn head_sentence(map: &TripleMap, tk: &TripleKnowledge, s: TripleSummary) -> String {
    let new = new_since(tk.pending_samples);
    let head = format!("{} arrangements of {} in {} samples", thousands(tk.triples), triple_name(tk.t), thousands(tk.samples as u64));
    if !map.basis.body3 {
        return format!("{head}{new}.");
    }
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
pub fn card_head(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, map: &Arc<TripleMap>, tk: &TripleKnowledge) {
    let s = summaries(&mut st.knowledge.triple, map).get(tk.t).unwrap_or_default();
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(triple_name(tk.t)).strong().size(14.0));
        theme::pill(ui, "three-atom", theme::INK_DIM);
        ui.label(RichText::new(format!("arrangements around {}", sym(tk.t.centre))).size(10.5).color(theme::INK_FAINT));
    });
    let s = ui.add(egui::Label::new(RichText::new(head_sentence(map, tk, s)).size(12.0).color(theme::INK)).wrap());
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
    // once per frame, from the view's memos: the scene's types and the map's summaries (Minor 1)
    let scene = scene_memo(&mut st.knowledge.triple, &input.scene.nuclei, map.r_cut).types();
    let sums = summaries(&mut st.knowledge.triple, &map).sums.clone();
    let rows = picker_rows(&map, &sums, &scene);
    if rows.is_empty() {
        ui.add(egui::Label::new(RichText::new(no_types(map.r_cut)).size(11.0).color(theme::INK_DIM)).wrap());
        return;
    }
    let shown = shown_type(&st.knowledge, &map, &rows);
    picker(ui, st, &rows, shown);
    let Some(tk) = shown.and_then(|t| map.get(t)) else { return };
    if st.knowledge.pick != Pick::Triple(tk.t) {
        ui.add_space(4.0);
        super::card(ui, "knowledge-triple-head", |ui| card_head(ui, st, input, &map, tk));
    }
    // the view: the slice, or the 3D cloud (Task 8)
    ui.add_space(6.0);
    let picked = st.knowledge.pick == Pick::Triple(tk.t);
    let view = &mut st.knowledge.triple;
    mode_switch(ui, view);
    sync(view, &map, tk);
    // a screenshot's pose (`--ui slice=`, `cell=`) belongs to the picked type
    if picked {
        apply_pose(view, &map, tk);
    }
    let mut cache = view.cache.take().expect("sync fills it");
    let r = ui
        .scope(|ui| match view.mode {
            TripleMode::Flat => flat(ui, view, &cache, input, &map, tk),
            TripleMode::Cloud => cloud::show(ui, view, &mut cache, input, &map, tk),
        })
        .response;
    view.cache = Some(cache);
    probe("knowledge-triple-view", &r);
}

/// The `2D | 3D` switch (display state, not recorded).
fn mode_switch(ui: &mut Ui, view: &mut TripleView) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let modes = [
            (TripleMode::Flat, "2D", "knowledge-triple-mode-2d", "one slice: the angle against r₁ at one r₂"),
            (TripleMode::Cloud, "3D", "knowledge-triple-mode-3d", "every cell at once: r₁, r₂ and the angle as a cloud you can turn"),
        ];
        for (mode, label, name, tip) in modes {
            let r = keyed(ui, name, |ui| ui.add(theme::choice_button(view.mode == mode, RichText::new(label).size(11.0)))).on_hover_text(tip);
            probe(name, &r);
            if r.clicked() {
                view.mode = mode;
            }
        }
    });
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

// ---- the 2D slice (Task 7) ----

/// A type's classes and drawn extent for one map (spec §6.4 "UI cost": computed once per (map
/// `Arc`, type), kept in `TripleView::cache`), with the two per-type scans the slice would
/// otherwise repeat every frame.
#[derive(Clone)]
pub struct ClassCache {
    /// The map they were computed from, held (fix round 1, I1): a held `Arc` keeps its address
    /// from being reused by a later map, so `Arc::ptr_eq` cannot match a different map.
    pub map: Arc<TripleMap>,
    pub t: TripleType,
    /// Per cell of the map's grid ([`class_table`]).
    pub classes: Vec<CellClass>,
    /// [`extent`].
    pub extent: Option<(Range<usize>, Range<usize>)>,
    /// The type's fullest slice ([`fullest_slice`]): the default slice without a scene instance.
    pub fullest: usize,
    /// The type's fullest fitted cell's count: the fill opacity's scale.
    pub top: u32,
    /// The 3D view's voxels and last projection (Task 8), made on its first frame: a memo of the
    /// fields above and the camera, so not compared.
    pub cloud: Option<cloud::Cloud>,
}

impl ClassCache {
    /// The cache of `tk` (a type of `map`).
    pub fn of(map: &Arc<TripleMap>, tk: &TripleKnowledge) -> ClassCache {
        let classes = class_table(map, tk);
        let extent = extent(map, tk, &classes);
        ClassCache { map: map.clone(), t: tk.t, classes, extent, fullest: fullest_slice(map, tk), top: tk.counts.iter().copied().max().unwrap_or(0), cloud: None }
    }

    /// Computed from this very map (`Arc::ptr_eq`) for type `t`.
    pub fn holds(&self, map: &Arc<TripleMap>, t: TripleType) -> bool {
        Arc::ptr_eq(&self.map, map) && self.t == t
    }
}

/// The map compared by pointer (it is a key, not a value); the cloud, a memo, is not compared.
impl PartialEq for ClassCache {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.map, &o.map) && self.t == o.t && self.classes == o.classes && self.extent == o.extent && self.fullest == o.fullest && self.top == o.top
    }
}

/// The map shown as its pointer.
impl std::fmt::Debug for ClassCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassCache")
            .field("map", &Arc::as_ptr(&self.map))
            .field("t", &self.t)
            .field("classes", &self.classes.len())
            .field("extent", &self.extent)
            .field("fullest", &self.fullest)
            .field("top", &self.top)
            .field("cloud", &self.cloud.as_ref().map(|c| (c.voxels.len(), c.drawn.len())))
            .finish()
    }
}

/// Every cell's class (`sim::knowledge::cell_class`), indexed by cell.
pub fn class_table(map: &TripleMap, tk: &TripleKnowledge) -> Vec<CellClass> {
    (0..map.grid.cells()).map(|c| cell_class(map, tk, c)).collect()
}

/// A cell the views draw: fitted > 0, new > 0, or missing within reach.
pub fn drawn(tk: &TripleKnowledge, class: CellClass, cell: usize) -> bool {
    class == CellClass::Missing || tk.counts.get(cell).is_some_and(|&n| n > 0) || tk.pending.get(cell).is_some_and(|&n| n > 0)
}

/// The drawn cells' extent: (the r₁ rows k₁, the r₂ slices k₂), each the contiguous range from the
/// lowest drawn bin to the highest. A folded type's two ranges are one, the union of its k₁ and
/// k₂ (its slice s shows every r₁). `None` with nothing drawn.
pub fn extent(map: &TripleMap, tk: &TripleKnowledge, classes: &[CellClass]) -> Option<(Range<usize>, Range<usize>)> {
    let g = &map.grid;
    let grow = |span: &mut Option<(usize, usize)>, k: usize| *span = Some(span.map_or((k, k), |(lo, hi)| (lo.min(k), hi.max(k))));
    let (mut r, mut s) = (None, None);
    for (c, &class) in classes.iter().enumerate().take(g.cells()) {
        if drawn(tk, class, c) {
            let (k1, k2, _) = g.coords(c);
            grow(&mut r, k1);
            grow(&mut s, k2);
        }
    }
    let ((r0, r1), (s0, s1)) = (r?, s?);
    if tk.t.folded() {
        let (lo, hi) = (r0.min(s0), r1.max(s1));
        return Some((lo..hi + 1, lo..hi + 1));
    }
    Some((r0..r1 + 1, s0..s1 + 1))
}

/// Slice s's cells: per row k₁ in `rows`, the [`COS_BINS`] cell indices `grid.index(folded, k₁, s,
/// q)` (a folded type reads the stored {k₁, s}).
pub fn slice_cells(grid: &TripleGrid, folded: bool, rows: Range<usize>, s: usize) -> Vec<[usize; COS_BINS]> {
    rows.map(|k1| std::array::from_fn(|q| grid.index(folded, k1, s, q))).collect()
}

/// The instance the default slice follows: with a selected atom that centres one, its least
/// r₁ + r₂ (a tie: the lowest (centre, j, k)); else the lowest (centre, j, k).
pub fn chosen_instance(instances: &[Triple], selected: Option<usize>) -> Option<Triple> {
    let key = |x: &Triple| (x.centre, x.j, x.k);
    selected
        .and_then(|i| instances.iter().filter(|x| x.centre == i).min_by(|a, b| (a.r1 + a.r2).total_cmp(&(b.r1 + b.r2)).then_with(|| key(a).cmp(&key(b)))))
        .or_else(|| instances.iter().min_by_key(|x| key(x)))
        .copied()
}

/// The slice shown when none was chosen: the chosen instance's r₂ bin, else `fullest`, the
/// type's fullest slice ([`fullest_slice`], cached as `ClassCache::fullest`).
pub fn default_slice(grid: &TripleGrid, chosen: Option<&Triple>, fullest: usize) -> usize {
    chosen.and_then(|x| grid.r_bin_of(x.r2)).unwrap_or(fullest)
}

/// The type's fullest slice: fitted + new, as the slice draws them (a folded cell {k₁, k₂} in
/// both slices), the lowest on a tie; 0 with nothing filled.
pub fn fullest_slice(map: &TripleMap, tk: &TripleKnowledge) -> usize {
    let g = &map.grid;
    let mut fill = vec![0u64; g.n_r];
    for c in 0..g.cells() {
        let n = tk.counts.get(c).copied().unwrap_or(0) as u64 + tk.pending.get(c).copied().unwrap_or(0) as u64;
        if n == 0 {
            continue;
        }
        let (k1, k2, _) = g.coords(c);
        fill[k2] += n;
        // a folded cell {k₁, k₂} shows in slice k₁ too
        if tk.t.folded() && k1 != k2 {
            fill[k1] += n;
        }
    }
    let best = fill.iter().copied().max().unwrap_or(0);
    fill.iter().position(|&n| n == best && n > 0).unwrap_or(0)
}

/// The row an instance sits on in slice s: its k₁ when its k₂ = s; for a folded type also its k₂
/// when its k₁ = s (the slice reads {row, s}). `None` off the slice or for a distance without a
/// bin.
pub fn on_slice(grid: &TripleGrid, inst: &Triple, s: usize) -> Option<usize> {
    let (k1, k2) = (grid.r_bin_of(inst.r1)?, grid.r_bin_of(inst.r2)?);
    if k2 == s {
        Some(k1)
    } else if inst.t.folded() && k1 == s {
        Some(k2)
    } else {
        None
    }
}

/// The angle axis' labels at the [`COS_BINS`] + 1 bin edges, in degrees: "0", "37", …, "180".
pub fn angle_labels() -> Vec<String> {
    (0..=COS_BINS).map(|q| if q < COS_BINS { cos_edges_deg(q).0 } else { cos_edges_deg(q - 1).1 }).map(|d| format!("{d:.0}")).collect()
}

/// A cell's class in words: "taught", "thin", "missing within reach"; a new-only cell outside the
/// reach "outside reach".
fn class_word(class: CellClass) -> &'static str {
    match class {
        CellClass::Taught => "taught",
        CellClass::Thin => "thin",
        CellClass::Missing => "missing within reach",
        CellClass::OutOfReach => "outside reach",
    }
}

/// (r₁ edges, r₂ edges, θ edges in degrees) of a cell, for its words.
fn cell_edges(g: &TripleGrid, cell: usize) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let (k1, k2, q) = g.coords(cell);
    (g.r_edges(k1), g.r_edges(k2), cos_edges_deg(q))
}

/// The hover: "r₁ 1.8–2.1 · r₂ 1.8–2.1 Bohr · θ 102–114° · n 342 (+12 new) · taught" (the
/// parenthesis only with new arrangements).
pub fn cell_hover(map: &TripleMap, tk: &TripleKnowledge, cell: usize) -> String {
    let ((a1, b1), (a2, b2), (ta, tb)) = cell_edges(&map.grid, cell);
    let n = tk.counts.get(cell).copied().unwrap_or(0);
    let k = tk.pending.get(cell).copied().unwrap_or(0);
    let new = if k > 0 { format!(" (+{k} new)") } else { String::new() };
    format!("r₁ {a1:.1}–{b1:.1} · r₂ {a2:.1}–{b2:.1} Bohr · θ {ta:.0}–{tb:.0}° · n {}{new} · {}", thousands(n as u64), class_word(cell_class(map, tk, cell)))
}

/// The cell card's line: "H–O–H · r₁ 1.8–2.1, r₂ 1.8–2.1 Bohr, θ 102–114° · 342 arrangements ·
/// taught · +12 new" (the last part only with new arrangements).
pub fn cell_card_line(map: &TripleMap, tk: &TripleKnowledge, cell: usize) -> String {
    let ((a1, b1), (a2, b2), (ta, tb)) = cell_edges(&map.grid, cell);
    let n = tk.counts.get(cell).copied().unwrap_or(0);
    let k = tk.pending.get(cell).copied().unwrap_or(0);
    let new = if k > 0 { format!(" · +{k} new") } else { String::new() };
    format!(
        "{} · r₁ {a1:.1}–{b1:.1}, r₂ {a2:.1}–{b2:.1} Bohr, θ {ta:.0}–{tb:.0}° · {} arrangements · {}{new}",
        triple_name(tk.t),
        thousands(n as u64),
        class_word(cell_class(map, tk, cell))
    )
}

/// The cell card's honest-limits band (spec §6.7 items 1–3; Phase A has no Teach), one line each.
pub fn band_lines(map: &TripleMap) -> Vec<String> {
    let first = format!(
        "Green counts arrangements in the cell ({} within {:.1} Bohr in each distance and {:.1} in cos θ); it is not the model's trust or accuracy.",
        taught_count(),
        TRIPLE_BIN,
        2.0 / COS_BINS as f64
    );
    let second = if !map.basis.body3 {
        NO_BODY3.to_string()
    } else {
        let four = if map.basis.body4 { " Four-atom arrangements (body-4) are learned too and not shown." } else { "" };
        format!("Counts every pair of neighbours within {:.1} Bohr of the centre, as the model's three-atom terms do.{four}", map.r_cut)
    };
    let third = "The model's three-atom terms are smooth in the distances and in cos θ: it may interpolate across a missing cell, and a green cell can still be wrong.".to_string();
    vec![first, second, third]
}

/// Brings the view up to date with the map shown (Review Focus 1): the class cache recomputed
/// when the map (`Arc::ptr_eq`) or the type changed — another type resets the slice and the
/// cell —, the slice clamped into the extent's slices, and a picked cell past the grid or no
/// longer drawn dropped (a kept one is normalised to its stored (k₁, k₂, q)).
pub fn sync(view: &mut TripleView, map: &Arc<TripleMap>, tk: &TripleKnowledge) {
    let g = &map.grid;
    if !view.cache.as_ref().is_some_and(|c| c.holds(map, tk.t)) {
        if view.cache.as_ref().is_some_and(|c| c.t != tk.t) {
            view.slice = None;
            view.cell = None;
        }
        view.cache = Some(ClassCache::of(map, tk));
    }
    let cache = view.cache.as_ref().expect("filled above");
    if let (Some(s), Some((_, slices))) = (view.slice, &cache.extent) {
        if !slices.contains(&s) {
            view.slice = Some(s.clamp(slices.start, slices.end - 1));
        }
    }
    view.cell = view.cell.and_then(|(k1, k2, q)| {
        if k1 >= g.n_r || k2 >= g.n_r || q >= g.cos_bins {
            return None;
        }
        let c = g.index(tk.t.folded(), k1, k2, q);
        drawn(tk, cache.classes[c], c).then(|| g.coords(c))
    });
}

/// A screenshot's pose of the view (`--ui slice=`, `cell=`; spec §6.8, Phase A Task 9), kept in
/// `TripleView::pose` until the picked type is shown with a map, then resolved once against that
/// map's grid ([`apply_pose`]): in Bohr, Bohr and degrees, so it does not depend on r_cut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShotPose {
    /// The r₂ (Bohr) whose bin is the 2D slice.
    pub slice_r: Option<f64>,
    /// The cell to pick: (r₁, r₂, θ) in Bohr, Bohr, degrees.
    pub cell: Option<(f64, f64, f64)>,
}

impl ShotPose {
    /// (the slice, the cell's stored (k₁, k₂, q)) on `grid`: the slice is the bin of `slice_r`,
    /// else of the cell's r₂; the cell is `grid.cell(folded, r₁, r₂, u)` with u = cos θ computed
    /// as sin(90° − θ), exact at 0°, 90° and 180° (cos 90° would be 6e-17 and land a bin short
    /// of u = 0's bin 5). A distance past the grid gives `None`.
    pub fn resolve(&self, grid: &TripleGrid, folded: bool) -> (Option<usize>, Option<(usize, usize, usize)>) {
        let cell = self.cell.and_then(|(r1, r2, deg)| grid.cell(folded, r1, r2, (90.0 - deg).to_radians().sin())).map(|c| grid.coords(c));
        let slice = self.slice_r.or(self.cell.map(|c| c.1)).and_then(|r| grid.r_bin_of(r));
        (slice, cell)
    }
}

/// Resolves a pending [`ShotPose`] once the picked type `tk` is shown with its map: the slice and
/// the cell are set from it, then [`sync`] clamps the slice and drops a cell that is not drawn,
/// as for any stale pick. The pose is spent either way.
pub fn apply_pose(view: &mut TripleView, map: &Arc<TripleMap>, tk: &TripleKnowledge) {
    let Some(pose) = view.pose.take() else { return };
    let (slice, cell) = pose.resolve(&map.grid, tk.t.folded());
    if slice.is_some() {
        view.slice = slice;
    }
    if pose.cell.is_some() {
        view.cell = cell;
    }
    sync(view, map, tk);
}

/// The rows' height (points): together ≈ 180 pt, each 8–18.
fn row_height(rows: usize) -> f32 {
    (180.0 / rows.max(1) as f32).clamp(8.0, 18.0)
}

/// The plot's left gutter (the r₁ labels) and right margin (half the last angle label).
const GUTTER: f32 = 26.0;
const MARGIN: f32 = 10.0;
/// Room above the slice for the top r₁ label, drawn centred on the plot's top edge (half of it
/// was clipped by the plot's painter without it).
const TOP_PAD: f32 = 6.0;

/// A rectangle's outline, dashed (a new cell).
fn dashed_rect(p: &egui::Painter, r: Rect, stroke: Stroke) {
    p.extend(Shape::dashed_line(&[r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()], stroke, 2.0, 2.0));
}

/// One marker of the open scene: its (row, q) on the plot, where it is drawn, the arrangement.
type Marker = (usize, usize, Pos2, Triple);

/// The 2D slice of the shown type (module doc): the slice control, the plot with its markers,
/// hover and click, the caption, and the picked cell's card.
fn flat(ui: &mut Ui, view: &mut TripleView, cache: &ClassCache, input: &PanelInput<'_>, map: &TripleMap, tk: &TripleKnowledge) {
    let t = tk.t;
    let Some((rows, slices)) = cache.extent.clone() else {
        nothing_drawn(ui, t);
        return;
    };
    let (g, folded) = (&map.grid, t.folded());
    let instances = scene_memo(view, &input.scene.nuclei, map.r_cut).instances(t).to_vec();
    let chosen = chosen_instance(&instances, input.selected);
    let mut s = view.slice.unwrap_or_else(|| default_slice(g, chosen.as_ref(), cache.fullest)).clamp(slices.start, slices.end - 1);
    // the slice control: a slider over the extent's r₂ bins, and the bin's range
    ui.horizontal_wrapped(|ui| {
        let mut k = s;
        let r = keyed(ui, "knowledge-triple-slice", |ui| ui.add_enabled(slices.len() > 1, egui::Slider::new(&mut k, slices.start..=slices.end - 1).show_value(false)));
        probe("knowledge-triple-slice", &r);
        if r.changed() && k != s {
            view.slice = Some(k);
            s = k;
        }
        let (lo, hi) = g.r_edges(s);
        ui.label(RichText::new(format!("r₂ ({}) {lo:.1}–{hi:.1} Bohr", pair_name(t.centre, t.b))).size(11.0).color(theme::INK_DIM));
    });
    ui.label(RichText::new(format!("r₁ ({}) Bohr", pair_name(t.centre, t.a))).size(10.0).color(theme::INK_FAINT));
    // the plot
    let rh = row_height(rows.len());
    let size = vec2(ui.available_width(), TOP_PAD + rh * rows.len() as f32 + 14.0);
    let resp = keyed(ui, "knowledge-triple-plot", |ui| ui.allocate_exact_size(size, Sense::click()).1);
    probe("knowledge-triple-plot", &resp);
    let rect = resp.rect;
    let plot = Rect::from_min_max(rect.min + vec2(GUTTER, TOP_PAD), pos2(rect.right() - MARGIN, rect.top() + TOP_PAD + rh * rows.len() as f32));
    let cw = plot.width() / COS_BINS as f32;
    let p = ui.painter_at(rect.expand(2.0));
    p.rect_filled(plot, 0.0, Color32::from_black_alpha(64));
    let cells = slice_cells(g, folded, rows.clone(), s);
    let cell_rect = |row: usize, q: usize| Rect::from_min_size(pos2(plot.left() + q as f32 * cw, plot.bottom() - (row - rows.start + 1) as f32 * rh), vec2(cw, rh));
    // (row, q) under a point of the plot
    let at = |pos: Pos2| -> Option<(usize, usize)> {
        plot.contains(pos).then(|| (rows.start + (((plot.bottom() - pos.y) / rh) as usize).min(rows.len() - 1), (((pos.x - plot.left()) / cw) as usize).min(COS_BINS - 1)))
    };
    let shown = |row: usize, q: usize| Some(cells[row - rows.start][q]).filter(|&c| drawn(tk, cache.classes[c], c));
    let top = cache.top;
    let opacity = |n: u32| 0.3 + 0.7 * ((1.0 + n as f32).ln() / (1.0 + top as f32).ln().max(1e-6)).min(1.0);
    for row in rows.clone() {
        for q in 0..COS_BINS {
            let Some(c) = shown(row, q) else { continue };
            let r = cell_rect(row, q).shrink(0.5);
            let n = tk.counts.get(c).copied().unwrap_or(0);
            match cache.classes[c] {
                CellClass::Taught => {
                    p.rect_filled(r, 0.0, GREEN.gamma_multiply(opacity(n)));
                }
                CellClass::Thin => {
                    p.rect_filled(r, 0.0, AMBER.gamma_multiply(opacity(n)));
                }
                CellClass::Missing => {
                    p.rect_stroke(r.shrink(0.5), 0.0, Stroke::new(1.0, theme::INK_FAINT), egui::StrokeKind::Inside);
                }
                CellClass::OutOfReach => {}
            }
            if tk.pending.get(c).is_some_and(|&k| k > 0) {
                dashed_rect(&p, r.shrink(1.5), Stroke::new(1.0, NEW_INK));
            }
            if view.cell == Some(g.coords(c)) {
                p.rect_stroke(r, 0.0, Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Inside);
            }
            if cfg!(test) {
                probe_rect(&format!("knowledge-triple-cell-{c}"), r);
            }
        }
    }
    // the axes' labels: r₁ bin edges in the gutter (≥ 11 pt apart), the angle edges under the plot
    let small = FontId::monospace(8.5);
    let step = (11.0 / rh).ceil().max(1.0) as usize;
    for k in (rows.start..=rows.end).filter(|k| (k - rows.start) % step == 0) {
        let y = plot.bottom() - (k - rows.start) as f32 * rh;
        p.text(pos2(plot.left() - 3.0, y), Align2::RIGHT_CENTER, format!("{:.1}", g.r_edges(k).0), small.clone(), theme::INK_FAINT);
    }
    for (q, l) in angle_labels().into_iter().enumerate() {
        p.text(pos2(plot.left() + q as f32 * cw, plot.bottom() + 2.0), Align2::CENTER_TOP, l, small.clone(), theme::INK_FAINT);
    }
    // the open scene's arrangements: white rings on this slice ("now" by the chosen one), faint
    // rings at their r₁ for those in other slices
    let x_of = |u: f64| plot.left() + ((1.0 - u.clamp(-1.0, 1.0)) / 2.0) as f32 * plot.width();
    let y_of = |r: f64| (plot.bottom() - ((r / g.r_bin) as f32 - rows.start as f32) * rh).clamp(plot.top(), plot.bottom());
    let (mut here, mut elsewhere): (Vec<Marker>, Vec<Marker>) = (Vec::new(), Vec::new());
    for inst in &instances {
        let q = cos_bin(inst.u);
        match on_slice(g, inst, s) {
            Some(row) if rows.contains(&row) => {
                // the row's own distance: r₁, or a folded type's r₂ when it is its r₁ that is on s
                let r = if folded && g.r_bin_of(inst.r2) != Some(s) { inst.r2 } else { inst.r1 };
                here.push((row, q, pos2(x_of(inst.u), y_of(r)), *inst));
            }
            Some(_) => {}
            None => {
                if let Some(row) = g.r_bin_of(inst.r1).filter(|k| rows.contains(k)) {
                    elsewhere.push((row, q, pos2(x_of(inst.u), y_of(inst.r1)), *inst));
                }
            }
        }
    }
    for &(_, _, c, _) in &elsewhere {
        p.circle_stroke(c, 3.5, Stroke::new(1.0, Color32::from_white_alpha(90)));
    }
    for &(_, _, c, _) in &here {
        p.circle_stroke(c, 4.0, Stroke::new(1.5, Color32::WHITE));
    }
    if let Some(&(_, _, c, _)) = here.iter().find(|h| Some(h.3) == chosen).or(here.first()) {
        probe_rect("knowledge-triple-ring", Rect::from_center_size(c, vec2(8.0, 8.0)));
        let galley = p.layout_no_wrap("now".into(), FontId::monospace(9.0), Color32::WHITE);
        let left = if c.x + 6.0 + galley.size().x <= plot.right() { c.x + 6.0 } else { c.x - 6.0 - galley.size().x };
        let at = pos2(left, (c.y - galley.size().y / 2.0).clamp(rect.top(), (plot.bottom() - galley.size().y).max(rect.top())));
        probe_rect("knowledge-triple-now", Rect::from_min_size(at, galley.size()));
        p.galley(at, galley, Color32::WHITE);
    }
    // hover and click
    let hovered = resp.hover_pos().and_then(at).and_then(|(row, q)| shown(row, q).map(|c| (row, q, c)));
    let resp = match hovered {
        Some((row, q, c)) => {
            p.rect_stroke(cell_rect(row, q), 0.0, Stroke::new(1.0, Color32::from_white_alpha(200)), egui::StrokeKind::Inside);
            let off = instances.iter().filter(|i| on_slice(g, i, s).is_none()).count();
            let mut text = cell_hover(map, tk, c);
            let mine = here.iter().filter(|h| (h.0, h.1) == (row, q)).count();
            if mine > 0 {
                let what = if mine == 1 { "arrangement".to_string() } else { format!("{mine} arrangements") };
                let more = if off > 0 { format!(" (+{off} in other slices)") } else { String::new() };
                text.push_str(&format!("\nnow: the open scene's {what} here{more}"));
            }
            for e in elsewhere.iter().filter(|e| (e.0, e.1) == (row, q)) {
                text.push_str(&format!("\nfaint ring: the open scene's arrangement at r₂ {:.2} Bohr, in another slice", e.3.r2));
            }
            resp.on_hover_ui_at_pointer(|ui| {
                ui.label(RichText::new(text).size(11.0));
            })
        }
        None => resp,
    };
    if resp.clicked() {
        view.cell = resp.interact_pointer_pos().and_then(at).and_then(|(row, q)| shown(row, q)).map(|c| g.coords(c));
    }
    ui.label(RichText::new(format!("angle at {} (cos θ bins)", sym(t.centre))).size(10.0).color(theme::INK_FAINT));
    cell_card(ui, view, map, tk);
}

/// A type without a drawn cell: one plain line in place of the view.
pub fn nothing_drawn(ui: &mut Ui, t: TripleType) {
    ui.add(egui::Label::new(RichText::new(format!("nothing to draw: no cell of {} is filled or within reach", triple_name(t))).size(11.0).color(theme::INK_DIM)).wrap());
}

/// The picked cell's card, under either view (no Teach in Phase A): its line and the band.
pub fn cell_card(ui: &mut Ui, view: &TripleView, map: &TripleMap, tk: &TripleKnowledge) {
    if let Some((k1, k2, q)) = view.cell {
        let c = map.grid.index(tk.t.folded(), k1, k2, q);
        ui.add_space(4.0);
        super::card(ui, "knowledge-triple-cell-card", |ui| {
            ui.add(egui::Label::new(RichText::new(cell_card_line(map, tk, c)).size(11.5).color(theme::INK)).wrap());
            let b = ui
                .vertical(|ui| {
                    for l in band_lines(map) {
                        ui.add(egui::Label::new(RichText::new(l).size(10.5).color(theme::INK_DIM)).wrap());
                    }
                })
                .response;
            probe("knowledge-triple-band", &b);
        });
    }
}
