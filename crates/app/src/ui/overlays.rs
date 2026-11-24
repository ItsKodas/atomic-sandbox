//! What the chrome draws over the 3D view (the rect the panels leave), laid out in stacking zones
//! by [`super::zones`] so nothing overlaps at any view size (spec 2026-11-22, view overlay zones):
//! - top-left: the status chip (state, who answered, time, step, T, thermostat; red with the error
//!   when there is one) and the draft banner under it;
//! - the right edge: the capture buttons, the segment card, the long first-pass offer and the
//!   teach-around prompt, then the atom card;
//! - bottom-left: the Volume / Isosurface / Dots switch with the axes gizmo, the Camera and Dots
//!   sliders over it, the orbital and knowledge legends above;
//! - bottom-centre: the sampling and relax cards, the faded-density label, the toasts;
//! - bottom-right: the estimated view's bonds / orbitals pair, the stale bonds' line over it and
//!   the Orbitals card above;
//! - the model offer stays centred over everything (a decision card, not in the zones).
//!
//! Each overlay reports the size it took per form (`UiState::zones`), the pass lays them out for
//! the next frame, and each draws at its rect. The distance guides, drag spring and pin squares
//! stay with the viewer (`viewer.rs`), which projects them from the camera; the cell, floor, axes
//! and placement plane are 3D lines (`overlay.rs`).
use super::dock;
use super::activity;
use super::icons::Icon;
use super::zones::{Item, Slot, ViewLayout};
use super::{keyed, source_colour, source_label, theme, Action, PanelInput, UiState, GREEN, RED, SPARK_E};
use crate::sim::Backend;
use egui::{vec2, Align2, Color32, Id, Order, Pos2, Rect, RichText, Stroke, Ui, Vec2};

pub fn show(ctx: &egui::Context, view: Rect, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    if let Some(s) = input.snapshot {
        st.temp_avg.observe(ctx.input(|i| i.time), &s.dynamics);
    }
    if view.width() < 50.0 || view.height() < 50.0 {
        return;
    }
    // the bottom-right corner's state first (it says whether its overlays show), then the layout
    // from what each overlay measured last frame
    let corner = corner(ctx, st, input);
    let label = layer_label(st, input, corner.as_ref());
    remember_content(st, input, label.as_deref());
    let items = requests(st, input, corner.as_ref(), label.as_deref());
    let lay = super::zones::layout_after(view, &items, Some(&st.zones.layout));
    st.zones.layout = lay.clone();
    let chip = status_chip(ctx, view, st, input, acts, &lay);
    st.zones.measured(Slot::Chip, lay.form(Slot::Chip), chip.size());
    if let Some(r) = draft_banner(ctx, view, &lay, input) {
        st.zones.measured(Slot::Banner, 0, r.size());
    }
    // the right edge: the capture buttons, the segment card, the offers, the atom card
    let capture = super::capture::show(ctx, view, &lay, st, input.camera, acts);
    st.zones.measured(Slot::Capture, 0, capture.size());
    if let Some(r) = segment_card(ctx, view, &lay, st, input, acts) {
        st.zones.measured(Slot::Segment, lay.form(Slot::Segment), r.size());
    }
    if let Some(o) = input.model().and_then(|m| m.pass_offer.as_ref()).filter(|_| !input.sampling()) {
        let r = area("offer-card", Slot::Offer, &lay).show(ctx, |ui| {
            theme::card().stroke(Stroke::new(1.0, super::AMBER.gamma_multiply(0.5))).show(ui, |ui| {
                ui.set_max_width(300.0);
                dock::offer(ui, o, acts);
            });
        });
        st.zones.measured(Slot::Offer, 0, r.response.rect.size());
    }
    // the teach-around prompt: one click, never run by itself
    if let Some(m) = input.model().filter(|_| teach_prompt(input).is_some()) {
        let r = area("teach-prompt-card", Slot::TeachPrompt, &lay).show(ctx, |ui| {
            theme::card().stroke(Stroke::new(1.0, super::AMBER.gamma_multiply(0.5))).show(ui, |ui| {
                ui.set_max_width(300.0);
                dock::teach_row(ui, m, input.sampling(), acts);
            });
        });
        st.zones.measured(Slot::TeachPrompt, 0, r.response.rect.size());
    }
    if st.show_atom_card {
        if let (Some((i, n)), Some(p)) = (input.selection(), lay.get(Slot::AtomCard)) {
            let pseudo = input.snapshot.and_then(|s| super::pseudo::of_species(&s.dynamics.eval.pseudised, n.z));
            // the layout's height, or the whole card on its first frame (measured, then laid out)
            let max_h = if p.rect.height() > 0.5 { p.rect.height() } else { f32::INFINITY };
            let natural = crate::atom_card::show(ctx, p.rect.right_top(), n, i, pseudo, max_h);
            st.zones.measured(Slot::AtomCard, 0, natural);
        }
    }
    model_offer_card(ctx, view, st, input, acts);
    // bottom-centre: the cards, the density label, the toast
    if let Some(r) = sampling_card(ctx, view, &lay, input, acts) {
        st.zones.measured(Slot::Sampling, lay.form(Slot::Sampling), r.size());
    }
    if let Some(r) = relax_card(ctx, view, &lay, input, acts) {
        st.zones.measured(Slot::Relax, lay.form(Slot::Relax), r.size());
    }
    if let Some(label) = &label {
        let r = density_label(ctx, view, &lay, label);
        st.zones.measured(Slot::DensityLabel, lay.form(Slot::DensityLabel), r.size());
    }
    if let Some(r) = toast(ctx, &lay, st) {
        st.zones.measured(Slot::Toast, 0, r.size());
    }
    // bottom-left: the switch, the sliders, the legends
    view_switch(ctx, &lay, st, input, acts);
    if let Some(r) = legend(ctx, view, &lay, st, input) {
        st.zones.measured(Slot::OrbitalLegend, lay.form(Slot::OrbitalLegend), r.size());
    }
    if let Some(r) = super::knowledge::legend(ctx, &lay, st, input) {
        st.zones.measured(Slot::KnowledgeLegend, lay.form(Slot::KnowledgeLegend), r.size());
    }
    // bottom-right: the pair, the stale line, the Orbitals card
    if let Some(c) = &corner {
        let r = bonds_pair(ctx, &lay, st, input, c, acts);
        st.zones.measured(Slot::BondsPair, 0, r.size());
        if let Some(r) = orbitals_card(ctx, &lay, c) {
            st.zones.measured(Slot::OrbitalsCard, 0, r.size());
        }
        let switch_w = st.zones.size(Slot::ViewSwitch, 0).map_or(0.0, |s| s.x);
        if let Some(r) = c.line.as_deref().map(|t| stale_line(ctx, view, &lay, switch_w, t)) {
            st.zones.measured(Slot::StaleLine, 0, r.size());
        }
    }
    // an overlay that came up, changed form or grew this pass: redo the pass with its new size,
    // so the frame shown never has it overlapping or off-centre (egui's second pass takes no input)
    if st.zones.moved(&items, &lay) && !ctx.will_discard() {
        ctx.request_discard("a 3D-view overlay changed size");
    }
    // Q3 of Appendix B (draft mode's per-atom uncertainty tint) is the sphere colour itself:
    // `gfx::risk_tint` of `viewer::Viewer::atom_risk`.
}

/// Tells the zones what decides each overlay's form sizes now (`zones::Zones::content`) and which
/// scene is up (`Zones::scene`: a new one forgets every size).
fn remember_content(st: &mut UiState, input: &PanelInput<'_>, label: Option<&str>) {
    use std::hash::{Hash, Hasher};
    let key = |f: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        f(&mut h);
        h.finish()
    };
    let scene = key(&|h| st.source.label().hash(h));
    st.zones.scene(scene);
    let notes = key(&|h| chip_notes(st, input).iter().for_each(|(_, n)| n.hash(h)));
    let label_key = key(&|h| label.hash(h));
    let legend = key(&|h| legend_of(st, input).map(|l| (l.text, l.warn)).hash(h));
    let segment = key(&|h| st.segment_preview.as_ref().map(|s| s.scene.nuclei.len()).hash(h));
    for (slot, k) in [(Slot::Chip, notes), (Slot::DensityLabel, label_key), (Slot::OrbitalLegend, legend), (Slot::Segment, segment)] {
        let drawn = st.zones.layout.form(slot);
        st.zones.content(slot, drawn, k);
    }
}

/// The `Area` of an overlay at its place in the zones: pinned at its rect's corner by its zone's
/// pivot (an overlay that is not laid out yet — it reports its size after its first frame — at
/// its zone's corner of the view).
fn area(id: &'static str, slot: Slot, lay: &ViewLayout) -> egui::Area {
    let (pivot, at) = lay.pin(slot).unwrap_or_else(|| (slot.pivot(), slot.pivot().pos_in_rect(&lay.view.shrink(super::zones::INSET_BOTTOM))));
    egui::Area::new(Id::new(id)).pivot(pivot).fixed_pos(at).order(Order::Middle).fade_in(false)
}

/// What each overlay that shows this frame asks of the layout: its forms' sizes as measured (the
/// compact forms guessed from the full one until drawn once).
fn requests(st: &UiState, input: &PanelInput<'_>, corner: Option<&Corner>, label: Option<&str>) -> Vec<Item> {
    let z = &st.zones;
    let full = |s: Slot| z.size(s, 0).unwrap_or(Vec2::ZERO);
    let item = |s: Slot, guesses: &[Vec2]| Item::new(s, z.forms(s, guesses));
    let mut v = Vec::new();
    // the chip: its help lines folded to one line each, then behind one ⓘ
    let c = full(Slot::Chip);
    let notes = chip_notes(st, input).len() as f32;
    v.push(item(Slot::Chip, &[Vec2::ZERO, vec2(c.x, c.y.min(CHIP_HEAD + NOTE_ROW * notes)), vec2(c.x, c.y.min(CHIP_HEAD + 4.0))]));
    if input.draft_banner.is_some() {
        v.push(item(Slot::Banner, &[Vec2::ZERO]));
    }
    if let Some(t) = st.zones.tag.get() {
        v.push(Item::new(Slot::CentreTag, vec![t]));
    }
    v.push(item(Slot::Capture, &[Vec2::ZERO]));
    if st.segment_preview.is_some() {
        v.push(item(Slot::Segment, &[Vec2::ZERO, vec2(SEGMENT_ROW_W + CARD_FRAME_W, 40.0)]));
    }
    if input.model().and_then(|m| m.pass_offer.as_ref()).is_some() && !input.sampling() {
        v.push(item(Slot::Offer, &[Vec2::ZERO]));
    }
    if teach_prompt(input).is_some() && input.model().is_some() {
        v.push(item(Slot::TeachPrompt, &[Vec2::ZERO]));
    }
    if st.show_atom_card && input.selection().is_some() {
        // its whole height (shrunk to the room above the bottom-right stack), then its least
        let f = full(Slot::AtomCard);
        v.push(item(Slot::AtomCard, &[Vec2::ZERO, vec2(f.x, ATOM_CARD_MIN_H.min(f.y))]).shrinkable(ATOM_CARD_MIN_H));
    }
    v.push(item(Slot::ViewSwitch, &[Vec2::ZERO]));
    // the sliders: one row, two rows, then in a popover behind the switch's sliders button
    if st.camera_view {
        let f = full(Slot::CameraSliders);
        v.push(item(Slot::CameraSliders, &[Vec2::ZERO, vec2(f.x * 0.62, f.y * 2.0), Vec2::ZERO]));
    }
    if st.volume_mode == super::VOLUME_DOTS {
        let f = full(Slot::DotsSliders);
        v.push(item(Slot::DotsSliders, &[Vec2::ZERO, vec2(f.x * 0.58, f.y * 2.0), Vec2::ZERO]));
    }
    if legend_of(st, input).is_some() {
        let f = full(Slot::OrbitalLegend);
        v.push(item(Slot::OrbitalLegend, &[Vec2::ZERO, vec2(f.x, f.y.min(56.0))]));
    }
    if st.show_knowledge {
        let f = full(Slot::KnowledgeLegend);
        v.push(item(Slot::KnowledgeLegend, &[Vec2::ZERO, vec2(f.x, f.y.min(84.0))]));
    }
    if input.snapshot.and_then(|s| s.dynamics.sampling.as_ref()).is_some() && input.sampling() {
        v.push(item(Slot::Sampling, &[Vec2::ZERO, vec2(SAMPLING_ROW_W + CARD_FRAME_W, 40.0)]));
    }
    if input.snapshot.and_then(|s| s.dynamics.relax.as_ref()).is_some_and(|r| r.converged.is_none()) {
        v.push(item(Slot::Relax, &[Vec2::ZERO, vec2(RELAX_ROW_W + CARD_FRAME_W, 40.0)]));
    }
    if let Some(l) = label {
        let f = full(Slot::DensityLabel);
        v.push(item(Slot::DensityLabel, &[Vec2::ZERO, vec2(f.x, f.y.min(12.0 + NOTE_ROW * l.lines().count() as f32))]));
    }
    if st.log.current_toast().is_some() {
        v.push(item(Slot::Toast, &[Vec2::ZERO]));
    }
    if let Some(c) = corner {
        v.push(item(Slot::BondsPair, &[Vec2::ZERO]));
        if c.line.is_some() {
            v.push(item(Slot::StaleLine, &[Vec2::ZERO]));
        }
        if c.card.is_some() {
            v.push(item(Slot::OrbitalsCard, &[Vec2::ZERO]));
        }
    }
    v
}

/// The chip's first line with its frame (points): the guess for its compact forms' height.
const CHIP_HEAD: f32 = 38.0;
/// A folded help line's height (points).
const NOTE_ROW: f32 = 17.0;
/// The least the atom card shrinks to (it scrolls below its title and first lines).
const ATOM_CARD_MIN_H: f32 = 96.0;

/// A help line (spec "Compact forms"): wrapped while it takes at most `rows` lines at the width
/// there is, else folded to one line cut with "…" and an ⓘ that shows the whole text on hover.
pub(super) fn help_line(ui: &mut Ui, text: &str, size: f32, colour: Color32, rows: usize) {
    if folds(ui, text, size, ui.available_width(), rows) {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                theme::info(ui, text);
                ui.add(egui::Label::new(RichText::new(text).size(size).color(colour)).truncate());
            });
        });
    } else {
        ui.add(egui::Label::new(RichText::new(text).size(size).color(colour)).wrap());
    }
}

/// Whether `text` at `size` takes more than `rows` lines at `width`.
fn folds(ui: &Ui, text: &str, size: f32, width: f32, rows: usize) -> bool {
    text.lines().count() > rows || ui.painter().layout(text.to_owned(), egui::FontId::proportional(size), Color32::WHITE, width.max(1.0)).rows.len() > rows
}

/// Draft mode's red "extrapolating" banner (Q2 of Appendix B): the model is answering beyond what
/// it has learned. It stacks directly under the status chip (top-left zone) so it never covers the
/// chip's time, step and temperature.
fn draft_banner(ctx: &egui::Context, view: Rect, lay: &ViewLayout, input: &PanelInput<'_>) -> Option<Rect> {
    let text = input.draft_banner.as_ref()?;
    let room = top_left_room(view, lay);
    let r = area("draft-banner", Slot::Banner, lay).interactable(false).show(ctx, |ui| {
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(150, 20, 20, 220))
            .stroke(Stroke::new(1.0, RED))
            .corner_radius(6.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.set_max_width((view.width() - 120.0).clamp(240.0, 560.0).min(room - 18.0).max(160.0));
                ui.label(RichText::new(text).color(Color32::WHITE).strong());
            });
    });
    Some(r.response.rect)
}

/// The top-left stack's width at most: up to the right edge's overlays beside it (the layout's
/// room; the view's width less the corners before the first layout).
fn top_left_room(view: Rect, lay: &ViewLayout) -> f32 {
    if lay.top_left_room > 0.0 {
        lay.top_left_room
    } else {
        view.width() - 2.0 * super::zones::INSET_TOP
    }
}

/// The state line of the old panel (row 2: every state keeps its text).
pub(super) fn state(input: &PanelInput<'_>) -> (bool, String) {
    let snap = input.snapshot;
    let sampling = snap.and_then(|s| s.dynamics.sampling.as_ref().filter(|_| crate::sim::sampling_active(&s.dynamics)));
    if let Some(p) = sampling {
        (true, format!("sampling the teacher: {} {}/{}", crate::ui::activity::words::pass_stage(p), p.done, p.total))
    } else if input.scene.nuclei.is_empty() && !input.busy {
        // playing or not: nothing moves until an atom is placed
        (false, super::EMPTY_STATUS.into())
    } else if input.playing {
        (true, "playing".into())
    } else if input.busy {
        (true, "solving…".into())
    } else if let Some(s) = snap {
        if s.dynamics.forces_stale {
            (false, "paused — not evaluated (see the error)".into())
        } else if s.scene == *input.scene {
            (false, "paused — converged".into())
        } else {
            (false, "paused — density is from an earlier geometry".into())
        }
    } else {
        (false, "waiting for the first solve".into())
    }
}

fn sep(ui: &mut egui::Ui) {
    ui.add(egui::Separator::default().vertical().spacing(6.0));
}

/// The chip's lines under the state: the bond references being solved in the background (dim,
/// M6 Task 3 fix round 2 — background work, never "solving…"), the status line (amber while it
/// asks for something, like "paused: … press Play to continue", dim otherwise) until it is
/// replaced or cleared, and the driver's refusal (amber) until the next accepted command — in
/// place, as the old panel kept them.
pub fn chip_notes(st: &UiState, input: &PanelInput<'_>) -> Vec<(Color32, String)> {
    let mut v = Vec::new();
    if st.volume_mode == super::VOLUME_DOTS {
        v.push((theme::INK_DIM, super::dock::orbitals::dots_note(st)));
    }
    if let Some(r) = input.snapshot.and_then(|s| s.dynamics.bond_reference) {
        v.push((theme::INK_DIM, bond_reference_line(&r)));
    }
    if let Some(g) = input.pseudo_generating {
        v.push((theme::INK_DIM, generating_line(g)));
    }
    if !st.status.is_empty() {
        let asks = st.status.starts_with("paused:") || st.status.contains("press ");
        v.push((if asks { super::AMBER } else { theme::INK_DIM }, st.status.clone()));
    }
    if let Some(r) = input.snapshot.and_then(|s| s.dynamics.last_rejection.as_ref()) {
        v.push((super::AMBER, format!("refused: {r}")));
    }
    if let Some(p) = teach_prompt(input) {
        v.push((super::AMBER, format!("stuck on DFT: {p}")));
    }
    v
}

/// The teach-around prompt (`sim::TeachOffer::prompt`): consecutive class-rule fallbacks blocked by
/// a sample far from here; `None` while a pass runs.
pub fn teach_prompt<'a>(input: &'a PanelInput<'_>) -> Option<&'a str> {
    input.model().and_then(|m| m.teach_offer.as_ref()).and_then(|o| o.prompt.as_deref()).filter(|_| !input.sampling())
}

/// "generating the H pseudopotential (≈ 1 min: its dimer check)… 12 s" — the first solve of a scene waits for its tables
/// (M7 Task 11; controller's cost condition: the minute is said, never silent). Without the gate, or with its answers on
/// disk: "(seconds)"; the gate on the CPU: "(≈ 40 min on the CPU: its dimer check (no GPU))" (final review M6).
pub fn generating_line((z, secs, cost): (u8, f64, providers::pseudo::GenerationCost)) -> String {
    format!("generating the {} pseudopotential ({})… {secs:.0} s", units::isotopes::symbol(z), cost.words())
}

/// "bond references: O 2/3… (14 s)": the species, the multiplicity being solved of the scan's
/// total, and the time spent (the old panel's status line, M6 Task 3).
pub fn bond_reference_line(r: &sim::ReferenceProgress) -> String {
    format!(
        "bond references: {} {}/{}… ({:.0} s)",
        units::isotopes::symbol(r.z),
        (r.done + 1).min(r.total),
        r.total,
        r.elapsed_s
    )
}

/// The status chip's ⚠ Model only pill (user request 2026-11-12), while it is on.
pub fn model_only_chip(st: &UiState) -> Option<&'static str> {
    st.model_only.is_some().then_some("model only — no DFT checks")
}

/// Draws the chip and returns its rect (the draft banner stacks under it). Its first line (the
/// state, the badges, t, step, T) always shows, wrapped rather than clipped; its help lines fold
/// to one line past two (form 0), to one line each (form 1), or behind one ⓘ (form 2, the last
/// resort below the minimum view).
fn status_chip(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>, lay: &ViewLayout) -> Rect {
    let form = lay.form(Slot::Chip);
    let notes = chip_notes(st, input);
    let errors: Vec<String> = input
        .last_error
        .map(str::to_string)
        .into_iter()
        .chain(input.snapshot.and_then(|s| s.dynamics.last_error.as_ref()).map(|e| format!("paused by a solver failure: {e}")))
        .collect();
    // the activity headline names what runs (spec §3.1); idle states keep their old texts. An
    // empty headline text is background work with no state of its own (bond references: the dim
    // note line below), so the old state text stays.
    let headline = st.activity.headline.peek().clone();
    let (busy, text) = match headline.as_ref().filter(|h| !h.text.is_empty()) {
        Some(h) => (true, h.text.clone()),
        None => state(input),
    };
    if headline.is_some() {
        ctx.request_repaint_after(activity::headline::REFRESH);
    }
    let room = top_left_room(view, lay);
    area("status-chip", Slot::Chip, lay)
        .constrain_to(view)
        .show(ctx, |ui| {
            let mut frame = theme::card();
            if !errors.is_empty() {
                frame = frame.stroke(Stroke::new(1.0, RED)).fill(Color32::from_rgba_premultiplied(70, 14, 20, 225));
            }
            frame.show(ui, |ui| {
                // within the 3D view and clear of the right edge's overlays beside it (the capture
                // buttons, their notes, the cards under them: the zones' room): one line while it
                // fits, wrapped onto more lines when it does not (user request 2026-10-21)
                ui.set_max_width((room - 24.0).max(200.0));
                ui.horizontal_wrapped(|ui| {
                    if busy {
                        ui.add(egui::Spinner::new().size(12.0).color(theme::ACCENT));
                    } else {
                        theme::light(ui, if errors.is_empty() { theme::INK_FAINT } else { RED });
                    }
                    ui.label(RichText::new(text).strong());
                    if let Some(a) = headline.as_ref().and_then(|h| h.audit.as_ref()) {
                        theme::pill(ui, a, GREEN).on_hover_text("a routine check of the model against DFT runs on the background teacher; the model keeps answering meanwhile");
                    }
                    if let Some(o) = headline.as_ref().and_then(|h| h.orbitals.as_ref()) {
                        theme::pill(ui, o, theme::ACCENT).on_hover_text("the orbital view's display eigensolve (Orbitals tab); it shows when it lands");
                    }
                    if let Some(s) = input.snapshot {
                        let d = &s.dynamics;
                        if s.backend == Backend::Estimated {
                            // lone atoms say so in the chip (rule 0b): the label alone could read as a model answer
                            let text = if d.source == sim::Source::Reference { "lone atoms: atom references" } else { source_label(d.source) };
                            theme::pill(ui, text, source_colour(d.source)).on_hover_text(super::source_tip(d.source));
                            if d.history_guess.last() == Some(&true) {
                                theme::pill(ui, "draft guess", RED).on_hover_text("draft mode: this step is the model's guess, also where it would have fallen back to DFT");
                            }
                            // clickable: the way off where the compact toolbar has no room for the
                            // toggle (a window narrower than about 1090 pt)
                            if let Some(t) = model_only_chip(st) {
                                let pill = theme::pill(ui, t, super::AMBER).rect;
                                // a click area over the whole pill, its text included (registered last: on top)
                                let r = ui.interact(pill, Id::new("model-only-chip"), egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
                                if super::probed("model-only-chip", r).on_hover_text(format!("click to turn Model only off (draft mode and the audits as they were before).\n\n{}", super::toolbar::MODEL_ONLY_TIP)).clicked() {
                                    acts.push(Action::ModelOnly(false));
                                }
                            }
                        } else {
                            theme::pill(ui, &format!("DFT · {}", s.engine.label()), SPARK_E);
                        }
                        sep(ui);
                        ui.label(RichText::new(format!("t {:.2} fs", d.sim_time_fs())).monospace());
                        ui.label(RichText::new(format!("step {}", d.step)).monospace().color(theme::INK_DIM));
                        sep(ui);
                        ui.label(RichText::new(format!("T {}", super::averaged_temperature_text(d, &st.temp_avg, 0, st.temp_units))).monospace())
                            .on_hover_text(format!("kinetic temperature averaged over the last second; this step: {}", super::temperature_text(d, 1, st.temp_units)));
                        if let Some((t, tau)) = d.thermostat {
                            ui.label(RichText::new(format!("CSVR {}", st.temp_units.show(t, 0))).color(theme::INK_DIM)).on_hover_text(format!("thermostat on: target {t} K, τ {tau} fs (Physics menu)"));
                        }
                    }
                    // the last resort: every help line behind one ⓘ at the end of the first line
                    if form >= 2 && !notes.is_empty() {
                        theme::info(ui, notes.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>().join("\n"));
                    }
                });
                for e in &errors {
                    ui.add(egui::Label::new(RichText::new(e).color(Color32::from_rgb(255, 210, 210))).wrap());
                }
                if form < 2 {
                    for (c, n) in &notes {
                        help_line(ui, n, 11.5, *c, if form == 0 { 2 } else { 1 });
                    }
                }
            });
        })
        .response
        .rect
}

/// The title, the Apply button and the Cancel button of [`model_offer_card`] (the tests and the docs quote them).
pub const OFFER_TITLE: &str = "This model was taught at other settings than this scene's";
pub const OFFER_APPLY: &str = "Apply the model's settings and load";
pub const OFFER_CANCEL: &str = "Cancel";
pub const OFFER_NOTE: &str = "Apply changes only the settings listed (the nuclei stay where they are), then loads the model. Cancel leaves everything as it is.";

/// The card for a picked model file whose settings differ from the scene's (`ui::model_offer`): each
/// differing setting as "model: X · scene: Y", what cannot be applied and why, and the two buttons.
/// A decision card: centred over everything, not in the zones.
fn model_offer_card(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let Some(o) = st.model_offer.as_ref() else { return };
    let sampling = input.sampling();
    let blocked = o.blocked();
    egui::Area::new(Id::new("model-offer-card"))
        .pivot(Align2::CENTER_CENTER)
        .fixed_pos(view.center())
        .order(Order::Foreground)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().stroke(Stroke::new(1.0, super::AMBER.gamma_multiply(0.6))).show(ui, |ui| {
                ui.set_width(460.0f32.min(view.width() - 40.0));
                ui.label(RichText::new(OFFER_TITLE).strong());
                ui.label(RichText::new(o.path.display().to_string()).monospace().size(11.0).color(theme::INK_DIM));
                ui.add_space(4.0);
                for r in &o.rows {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(r.name).strong());
                        ui.label(RichText::new(r.versus()).monospace().size(11.5));
                    });
                    if let Some(b) = &r.blocked {
                        ui.add(egui::Label::new(RichText::new(format!("cannot be applied: {b}")).size(11.0).color(super::AMBER)).wrap());
                    }
                    if let Some(n) = &r.note {
                        ui.add(egui::Label::new(RichText::new(n).size(11.0).color(theme::INK_DIM)).wrap());
                    }
                }
                ui.add_space(4.0);
                theme::note(ui, OFFER_NOTE);
                ui.horizontal(|ui| {
                    let apply = egui::Button::new(RichText::new(OFFER_APPLY).strong());
                    let why = if sampling {
                        Some("a sampling pass is running: cancel it first")
                    } else {
                        blocked.map(|_| "a setting listed above cannot be applied")
                    };
                    let r = ui.add_enabled(why.is_none(), apply);
                    let r = match why {
                        Some(w) => r.on_disabled_hover_text(w),
                        None => r.on_hover_text("set the grid, σ, charge, multiplicity, ionic potential and isotopes listed to the model's, then load it"),
                    };
                    if r.clicked() {
                        acts.push(Action::ApplyModelSettings);
                    }
                    if ui.button(OFFER_CANCEL).on_hover_text("close this card; nothing is changed or loaded").clicked() {
                        acts.push(Action::CancelModelOffer);
                    }
                });
            });
        });
}

/// What the segment confirmation's note says (mockup (d) 2.).
pub const SEGMENT_NOTE: &str = "The segment is the selection plus every atom within the reach (twice the model's cutoff, plus the Standard pass's largest planned move, so every force it teaches is an equation of this scene's own); a hydrogen goes with its partner. Each cut bond gets a capping H placed with the molecule builder's tables (placement only). Only the green-ringed atoms are taught; the caps (teal) and the edge atoms are context. The cut is for the Standard plan: another plan may move a cap closer than it allows.";

/// Train this segment's preview card (mockup (d) 2.): the counts, the reach, the box, the GPU
/// memory and the plan, with "Open the segment" / "Cancel" (`Action::OpenSegment` / `CancelSegment`).
/// Compact (form 1): one row — the title, the atoms, Open and Cancel — the rest on hover.
fn segment_card(ctx: &egui::Context, view: Rect, lay: &ViewLayout, st: &UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Option<Rect> {
    let seg = st.segment_preview.as_ref()?;
    let kept = seg.source_indices.iter().flatten().count();
    let s = &seg.scene;
    let sym = |i: usize| input.scene.nuclei.get(i).map_or("?", |n| render::elements::symbol(n.z));
    let solvers = crate::quality::gpu_solvers(Backend::Estimated, st.gpu_available);
    let gpu = if solvers == 0 {
        "the CPU teacher: no GPU budget".to_string()
    } else {
        let need = solvers * s.to_sim().map(|x| providers::gpu_footprint_bytes(&x)).unwrap_or(0);
        format!("≈ {:.1} GB of {} ({solvers} solvers: teacher + auditor)", need as f64 / 1e9, crate::quality::vram_budget().gb())
    };
    let bonds: Vec<String> = seg.cut_bonds.iter().map(|&(k, o)| format!("{}#{k}–{}#{o}", sym(k), sym(o))).collect();
    let rows: Vec<(&str, String)> = vec![
        ("atoms", format!("{} = {kept} from {} + {} caps", s.nuclei.len(), st.source.label(), seg.caps.len())),
        ("caps", if bonds.is_empty() { "none: no bond is cut".into() } else { format!("{} H on {}", seg.caps.len(), bonds.join(", ")) }),
        ("trained on", format!("{} atoms ({} selected)", seg.n_interior(), seg.selected.len())),
        ("not trained", format!("{} edge atoms + {} caps", kept - seg.n_interior(), seg.caps.len())),
        ("reach", format!("{:.2} Bohr = 2 × r_cut {:.1} + δ {:.2}", seg.reach, seg.r_cut, seg.delta)),
        ("solver", format!("estimated · ionic {} (the scene's key)", s.ionic.word())),
        ("box", format!("{:.1} Bohr · {}³ × {} Bohr", crate::quality::box_len(s), s.grid_n, s.grid_h)),
        ("GPU memory", gpu),
        ("plan", "Standard, ready — not started".into()),
    ];
    let details = |ui: &mut Ui| {
        egui::Grid::new("segment-rows").num_columns(2).spacing([12.0, 3.0]).show(ui, |ui| {
            for (k, v) in &rows {
                ui.label(RichText::new(*k).color(theme::INK_DIM));
                ui.add(egui::Label::new(RichText::new(v).monospace().size(11.5)).wrap());
                ui.end_row();
            }
        });
        ui.add_space(4.0);
        theme::note(ui, SEGMENT_NOTE);
    };
    let open = |ui: &mut Ui, acts: &mut Vec<Action>| {
        let open = egui::Button::new(RichText::new("Open the segment").strong().color(Color32::from_rgb(0x17, 0x11, 0x2a))).fill(theme::ACCENT);
        if super::probed("segment-open", ui.add(open)).on_hover_text("save this scene to a temp file and open the segment in the estimated solver, the Standard plan ready (press Start sampling to teach it); the chip by the scene chip brings you back").clicked() {
            acts.push(Action::OpenSegment);
        }
    };
    let cancel = |ui: &mut Ui, acts: &mut Vec<Action>| {
        if super::probed("segment-cancel", ui.button("Cancel")).on_hover_text("close the preview; nothing changes").clicked() {
            acts.push(Action::CancelSegment);
        }
    };
    let compact = lay.form(Slot::Segment) > 0;
    let r = area("segment-card", Slot::Segment, lay).order(Order::Foreground).show(ctx, |ui| {
        theme::card().stroke(Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.6))).show(ui, |ui| {
            if compact {
                ui.set_width(SEGMENT_ROW_W.min(view.width() - 40.0));
                one_row(ui, |ui| {
                    // right to left: Cancel at the end, Open before it
                    cancel(ui, acts);
                    open(ui, acts);
                }, |ui| {
                    let text = format!("Train this segment · {} atoms, {} trained", s.nuclei.len(), seg.n_interior());
                    ui.add(egui::Label::new(RichText::new(text).strong()).truncate()).on_hover_ui(|ui| {
                        ui.set_max_width(400.0);
                        details(ui);
                    });
                });
            } else {
                ui.set_width(400.0f32.min(view.width() - 40.0));
                ui.label(RichText::new("Train this segment").strong().size(14.0));
                details(ui);
                ui.horizontal(|ui| {
                    open(ui, acts);
                    cancel(ui, acts);
                });
            }
        });
    });
    Some(r.response.rect)
}

/// A pass's Pause, or Resume while it is paused (user request 2026-11-16), probed as
/// `<probe>-pause` / `<probe>-resume`.
pub fn pause_button(ui: &mut egui::Ui, paused: bool, probe: &str, acts: &mut Vec<Action>) {
    let (label, action) = crate::ui::activity::jobs::pause_control(paused);
    let tip = if paused { crate::ui::activity::jobs::RESUME_PASS_TIP } else { crate::ui::activity::jobs::PAUSE_PASS_TIP };
    if super::probed(&format!("{probe}-{}", label.to_lowercase()), ui.button(label)).on_hover_text(tip).clicked() {
        acts.push(action);
    }
}

/// What the sampling card's Cancel says on hover.
const CANCEL_PASS_TIP: &str = "end the pass now: what it sampled is kept; it is refitted only once there are enough samples to fit (Q′, the model's number of directions), else the Log says so and Sample more adds to them";
/// The sampling card's footnote.
const PASS_STAGES: &str = "Stages: isolated-atom references, perturbations, pair scans, hot MD burst.";

/// The running pass's card (bottom centre). Compact (form 1): one row — the title, the stage n/N,
/// Pause and Cancel — the progress and the stages on hover.
fn sampling_card(ctx: &egui::Context, view: Rect, lay: &ViewLayout, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Option<Rect> {
    let p = input.snapshot.and_then(|s| s.dynamics.sampling.as_ref()).filter(|_| input.sampling())?;
    let compact = lay.form(Slot::Sampling) > 0;
    let title = if p.paused { "Sampling pass · paused" } else { "Sampling pass" };
    let r = area("sampling-card", Slot::Sampling, lay).show(ctx, |ui| {
        theme::card().show(ui, |ui| {
            if compact {
                // one width whatever the stage, the counts or Pause ↔ Resume say (the text is cut
                // with "…" past it): the zones never move for a digit
                ui.set_width(SAMPLING_ROW_W.min(view.width() - 40.0));
                one_row(ui, |ui| {
                    if ui.button("Cancel").on_hover_text(CANCEL_PASS_TIP).clicked() {
                        acts.push(Action::CancelSampling);
                    }
                    pause_button(ui, p.paused, "sampling-card", acts);
                }, |ui| {
                    let stage = format!("{title} · {} {}/{}", crate::ui::activity::words::pass_stage(p), p.done, p.total);
                    ui.add(egui::Label::new(RichText::new(stage).strong()).truncate()).on_hover_ui(|ui| {
                        ui.set_max_width(420.0);
                        dock::progress(ui, p, input.model());
                        theme::note(ui, PASS_STAGES);
                    });
                });
                return;
            }
            ui.set_width(520.0f32.min(view.width() - 40.0));
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Cancel").on_hover_text(CANCEL_PASS_TIP).clicked() {
                        acts.push(Action::CancelSampling);
                    }
                    pause_button(ui, p.paused, "sampling-card", acts);
                });
            });
            dock::progress(ui, p, input.model());
            theme::note(ui, PASS_STAGES);
        });
    });
    Some(r.response.rect)
}

/// The one-row cards' fixed widths (points, inside the frame; the frame adds 22): room for the
/// common texts with their buttons ("Sampling pass · random nudges 8/60", Pause, Cancel — the
/// width it measured, so the 560 × 360 view keeps it beside the corners); a longer one ("· paused ·
/// teaching around #345 …") is cut with "…", its whole on hover. Fixed,
/// so a digit, a stage word or Pause ↔ Resume never changes the card's size.
const SAMPLING_ROW_W: f32 = 310.0;
const RELAX_ROW_W: f32 = 280.0;
const SEGMENT_ROW_W: f32 = 400.0;
/// A card frame's horizontal margins and stroke (`theme::card`).
const CARD_FRAME_W: f32 = 22.0;

/// A one-row card's row: the buttons at its right end (`right`, added right to left), the text
/// filling the rest from the left (`left`, cut to fit).
fn one_row(ui: &mut Ui, right: impl FnOnce(&mut Ui), left: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            right(ui);
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), left);
        });
    });
}

/// The FIRE note under the relax card.
const RELAX_NOTE: &str = "FIRE steps are not time: t and the step count stay put; the energy drained is booked as work done by tools.";

/// The running relax's card (bottom centre): its readout and Stop. Compact (form 1): one row — the
/// title, the step and Stop — the readout on hover.
fn relax_card(ctx: &egui::Context, view: Rect, lay: &ViewLayout, input: &PanelInput<'_>, acts: &mut Vec<Action>) -> Option<Rect> {
    let r = input.snapshot.and_then(|s| s.dynamics.relax.as_ref()).filter(|r| r.converged.is_none())?;
    let compact = lay.form(Slot::Relax) > 0;
    let resp = area("relax-card", Slot::Relax, lay).show(ctx, |ui| {
        theme::card().show(ui, |ui| {
            if compact {
                ui.set_width(RELAX_ROW_W.min(view.width() - 40.0));
                one_row(ui, |ui| {
                    if ui.button("Stop").on_hover_text(super::STOP_RELAX_TIP).clicked() {
                        acts.push(Action::StopRelax);
                    }
                }, |ui| {
                    ui.add(egui::Label::new(RichText::new(format!("Relaxing (FIRE) · step {}", r.steps)).strong()).truncate()).on_hover_ui(|ui| {
                        ui.set_max_width(460.0);
                        ui.label(super::RELAX_TIP);
                        for l in super::relax_lines(r) {
                            ui.add(egui::Label::new(RichText::new(l).monospace().size(11.5)).wrap());
                        }
                        theme::note(ui, RELAX_NOTE);
                    });
                });
                return;
            }
            ui.set_width(520.0f32.min(view.width() - 40.0));
            ui.horizontal(|ui| {
                ui.label(RichText::new("Relaxing (FIRE)").strong()).on_hover_text(super::RELAX_TIP);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Stop").on_hover_text(super::STOP_RELAX_TIP).clicked() {
                        acts.push(Action::StopRelax);
                    }
                });
            });
            for l in super::relax_lines(r) {
                ui.add(egui::Label::new(RichText::new(l).monospace().size(11.5)).wrap());
            }
            theme::note(ui, RELAX_NOTE);
        });
    });
    Some(resp.response.rect)
}

/// The label under the 3D view's layer: the density's (or, in an orbital mode, the orbitals' own
/// label or note) and the bonds' (less the stale line the bottom-right corner already shows).
fn layer_label(st: &UiState, input: &PanelInput<'_>, corner: Option<&Corner>) -> Option<String> {
    // in an orbital mode the orbitals' own label or note replaces the density's
    let layer_label = if st.show == super::ShowMode::Density { &input.density_label } else { &input.orbital_label };
    // the stale bonds' line sits over "Bonds now" when that is drawn: not repeated here
    let bonds_label = match (&input.bonds_label, corner.and_then(|c| c.line.as_ref())) {
        (Some(b), Some(l)) => Some(b.lines().filter(|x| x != l).collect::<Vec<_>>().join("\n")).filter(|x| !x.is_empty()),
        (b, _) => b.clone(),
    };
    match (layer_label, &bonds_label) {
        (Some(d), Some(b)) => Some(format!("{d}\n{b}")),
        (d, b) => d.clone().or_else(|| b.clone()),
    }
}

/// The faded-density label (bottom centre, over the cards): each line wrapped up to two lines, else
/// folded to one with its ⓘ (form 0); every line folded to one (form 1). Not interactable — it
/// never takes a click from the view — unless a line is folded (its ⓘ needs the hover).
fn density_label(ctx: &egui::Context, view: Rect, lay: &ViewLayout, label: &str) -> Rect {
    let rows = if lay.form(Slot::DensityLabel) == 0 { 2 } else { 1 };
    let width = (view.width() - 2.0 * 280.0).max(260.0);
    let folded = ctx.fonts_mut(|f| label.lines().any(|l| f.layout(l.to_owned(), egui::FontId::proportional(14.0), Color32::WHITE, width).rows.len() > rows));
    area("density-label", Slot::DensityLabel, lay)
        .interactable(folded)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(Color32::from_black_alpha(150))
                .corner_radius(4.0)
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.set_max_width(width);
                    ui.spacing_mut().item_spacing.y = 2.0;
                    for l in label.lines() {
                        help_line(ui, l, 14.0, Color32::from_gray(210), rows);
                    }
                });
        })
        .response
        .rect
}

/// The current status line as a toast (bottom centre, above the cards and the density label;
/// a refusal in amber).
fn toast(ctx: &egui::Context, lay: &ViewLayout, st: &mut UiState) -> Option<Rect> {
    let (kind, text) = st.log.current_toast()?;
    let text = text.to_string();
    let mut open_log = false;
    let r = area("toast", Slot::Toast, lay).show(ctx, |ui| {
        theme::card().show(ui, |ui| {
            // at most half the view: on a small view it wraps rather than spanning the corners
            ui.set_max_width(420.0f32.min((lay.view.width() - 2.0 * super::zones::INSET_BOTTOM) * 0.5).max(200.0));
            ui.horizontal(|ui| {
                theme::light(ui, kind.colour());
                let t = RichText::new(&text);
                let t = if kind == super::logbook::LogKind::Refused { t.color(super::AMBER) } else { t };
                ui.add(egui::Label::new(t).wrap());
                if ui.add(egui::Button::new(RichText::new("· Log").color(theme::INK_FAINT)).frame(false)).clicked() {
                    open_log = true;
                }
            });
        });
    });
    if open_log {
        st.open_tab(super::DockTab::Log);
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(500));
    Some(r.response.rect)
}

/// Bottom-left: the axes gizmo (with the axes toggle) and the quick Volume / Isosurface switch,
/// with the Camera view's and the Dots' sliders stacked above it. Camera (the fourth choice) is the
/// glow view (`UiState::camera_view`). The sliders wrap onto two rows (form 1) and, as the last
/// resort, go into a popover behind a sliders button beside the switch (form 2).
fn view_switch(ctx: &egui::Context, lay: &ViewLayout, st: &mut UiState, input: &PanelInput<'_>, acts: &mut Vec<Action>) {
    let dots = st.volume_mode == super::VOLUME_DOTS;
    let camera = st.camera_view;
    let dots_form = lay.form(Slot::DotsSliders);
    let camera_form = lay.form(Slot::CameraSliders);
    if camera && camera_form < 2 {
        let r = area("camera-sliders", Slot::CameraSliders, lay).show(ctx, |ui| slider_card(ui, |ui| camera_sliders(ui, st, camera_form + 1)));
        st.zones.measured(Slot::CameraSliders, camera_form, r.response.rect.size());
    }
    if dots && dots_form < 2 {
        let r = area("dots-sliders", Slot::DotsSliders, lay).show(ctx, |ui| slider_card(ui, |ui| dots_sliders(ui, st, dots_form + 1)));
        st.zones.measured(Slot::DotsSliders, dots_form, r.response.rect.size());
    }
    let popover = (dots && dots_form >= 2) || (camera && camera_form >= 2);
    let dots_tip = format!("random samples of the field drawn, resampled every frame (View › How it looks › shimmer). {}", super::dock::orbitals::dots_note(st));
    let r = area("view-switch", Slot::ViewSwitch, lay).show(ctx, |ui| {
        ui.horizontal(|ui| {
            if st.show_axes {
                gizmo(ui, input);
            }
            egui::Frame::new().fill(theme::CARD).corner_radius(7).show(ui, |ui| {
                if let Some(m) = theme::segmented(
                    ui,
                    st.volume_mode,
                    &[
                        (0, "Volume", "density (or an orbital, or the spin density) as a glowing volume (View › How it looks for gamma and opacity)"),
                        (1, "Isosurface", "density as a surface at iso × ρ max; an orbital or the spin density as a ± isosurface (View › How it looks, and the iso slider in the Orbitals tab)"),
                        (2, "Dots", dots_tip.as_str()),
                    ],
                ) {
                    // the Camera view draws no isosurface: picking it leaves the Camera view
                    if m == super::VOLUME_ISO {
                        st.camera_view = false;
                    }
                    acts.push(Action::VolumeMode(m));
                }
            });
            // Camera: a toggle over Volume or Dots (the atoms as they look; Isosurface turns it off)
            egui::Frame::new().fill(theme::CARD).corner_radius(7).show(ui, |ui| {
                let on = st.camera_view;
                let r = ui.add(egui::Button::selectable(on, "Camera")).on_hover_text(
                    "what an atom looks like: no surface, a soft electron glow densest at the centre (or solid balls); a moving atom streaks over the shutter — over the Volume or the Dots (not the Isosurface)",
                );
                if r.clicked() {
                    st.camera_view = !on;
                    if st.camera_view && st.volume_mode == super::VOLUME_ISO {
                        acts.push(Action::VolumeMode(super::VOLUME_VOLUME));
                    }
                }
            });
            // a view too small for the slider rows: they open from here
            if popover {
                let open = egui::Popup::is_id_open(ui.ctx(), Id::new(SLIDERS_POPUP));
                let b = egui::Frame::new().fill(theme::CARD).corner_radius(7).show(ui, |ui| super::icons::button(ui, Icon::Sliders, 26.0, open, true)).inner;
                let b = super::probed("view-sliders", b).on_hover_text("the sliders of the Dots and the Camera view (the view is too small to show them over the switch)");
                egui::Popup::from_toggle_button_response(&b)
                    .id(Id::new(SLIDERS_POPUP))
                    .align(egui::RectAlign::TOP_START)
                    .gap(6.0)
                    .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                    .show(|ui| {
                        if dots && dots_form >= 2 {
                            dots_sliders(ui, st, 2);
                        }
                        if camera && camera_form >= 2 {
                            camera_sliders(ui, st, 2);
                        }
                    });
            }
        });
    });
    st.zones.measured(Slot::ViewSwitch, 0, r.response.rect.size());
}

/// The sliders' popover (the view switch's last resort).
const SLIDERS_POPUP: &str = "view-sliders-popup";

/// A slider row's card over the view.
fn slider_card(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    egui::Frame::new().fill(theme::CARD).corner_radius(7).inner_margin(egui::Margin::symmetric(10, 4)).show(ui, add);
}

/// The dots' density, drift speed, feather, size and blur, on one row or two.
fn dots_sliders(ui: &mut Ui, st: &mut UiState, rows: usize) {
    ui.spacing_mut().slider_width = 90.0;
    let first = |ui: &mut Ui, st: &mut UiState| {
        ui.label("density");
        keyed(ui, "ov-dots-amount", |ui| ui.add(egui::Slider::new(&mut st.dots_amount, 0.5..=6.0).show_value(false))).on_hover_text("how many dots: more fill the same space more densely");
        ui.label(format!("{:.1}×", st.dots_amount));
        ui.add_space(8.0);
        ui.label("drift");
        keyed(ui, "ov-dots-drift", |ui| ui.add_enabled(st.dots_float, egui::Slider::new(&mut st.dots_drift, 0.002..=1.5).logarithmic(true).show_value(false)))
            .on_hover_text("how fast the dots drift (View ▸ How it looks ▸ floating)");
        ui.add_space(8.0);
        ui.label("feather");
        keyed(ui, "ov-dots-feather", |ui| ui.add(egui::Slider::new(&mut st.dots_feather, 0.0..=0.8).show_value(false)))
            .on_hover_text("fade out the faint outskirts: the dots where the field is low disappear softly, leaving the main lobes (0 = every dot)");
    };
    let second = |ui: &mut Ui, st: &mut UiState| {
        ui.label("size");
        keyed(ui, "ov-dots-size", |ui| ui.add(egui::Slider::new(&mut st.dots_size, 0.05..=3.0).logarithmic(true).show_value(false))).on_hover_text("the dots' size");
        ui.add_space(8.0);
        ui.label("blur");
        keyed(ui, "ov-dots-blur", |ui| ui.add(egui::Slider::new(&mut st.dots_blur, 0.0..=1.0).show_value(false))).on_hover_text("how soft each dot's edge is: left crisp, right blurry (the middle is the default look)");
    };
    if rows <= 1 {
        ui.horizontal(|ui| {
            first(ui, st);
            ui.add_space(8.0);
            second(ui, st);
        });
    } else {
        ui.vertical(|ui| {
            ui.horizontal(|ui| first(ui, st));
            ui.horizontal(|ui| second(ui, st));
        });
    }
}

/// The Camera view's shutter, exposure and smoothness, on one row or two.
fn camera_sliders(ui: &mut Ui, st: &mut UiState, rows: usize) {
    ui.spacing_mut().slider_width = 110.0;
    let first = |ui: &mut Ui, st: &mut UiState| {
        ui.label("shutter");
        keyed(ui, "ov-shutter", |ui| ui.add(egui::Slider::new(&mut st.shutter, 0.0..=1.0).show_value(false)))
            .on_hover_text("how much of the motion between frames one picture collects: longer streaks to the right, a sharp still at 0");
        ui.label(format!("{:.0} %", st.shutter * 100.0));
        ui.add_space(8.0);
        ui.label("exposure");
        keyed(ui, "ov-exposure", |ui| ui.add(egui::Slider::new(&mut st.exposure_ev, -3.0..=3.0).step_by(0.1).show_value(false))).on_hover_text("brightness of the glow, in stops (EV)");
        ui.label(format!("{:+.1} EV", st.exposure_ev));
    };
    let second = |ui: &mut Ui, st: &mut UiState| {
        ui.label("smooth");
        keyed(ui, "ov-camera-smooth", |ui| ui.add(egui::Slider::new(&mut st.camera_smooth, 0.0..=2.0).show_value(false)))
            .on_hover_text("with any solidity: one smooth surface — neighbouring atoms flow into each other through a soft neck (0 = separate balls)");
    };
    if rows <= 1 {
        ui.horizontal(|ui| {
            first(ui, st);
            ui.add_space(8.0);
            second(ui, st);
        });
    } else {
        ui.vertical(|ui| {
            ui.horizontal(|ui| first(ui, st));
            ui.horizontal(|ui| second(ui, st));
        });
    }
}

/// What the legend chip shows (orbital-viewer spec §3): the colour pair or the palette in use.
pub struct Legend {
    /// Swatches (linear display colour, name) and the line under them.
    pub swatches: Vec<([f32; 3], String)>,
    pub text: String,
    /// The tooltip: the facts of spec §2 that apply to what is drawn.
    pub tip: String,
    /// A warning under the text (amber): a boundary surface that reaches the box walls.
    pub warn: Option<String>,
}

/// The legend of the orbital layer for the panel's `show` mode and the payload on screen (`None`
/// for the density, or with nothing of that mode to draw yet): phase ±, spin ↑/↓, or the palette
/// slots of the included orbitals with the scale in use.
pub fn legend_of(st: &UiState, input: &PanelInput<'_>) -> Option<Legend> {
    use super::dock::orbitals as ot;
    use super::ShowMode;
    let o = input.snapshot?.dynamics.orbitals.as_ref().filter(|o| o.field.is_some())?;
    let signed = matches!(o.field, Some(sim::OrbitalField::Gpu { signed: true, .. } | sim::OrbitalField::Cpu { signed: true, .. }));
    Some(match (st.show, &o.what) {
        (ShowMode::Orbitals, sim::OrbitalRequest::Orbital { .. }) if signed => Legend {
            swatches: vec![(render::volume::PHASE_POS, "+".into()), (render::volume::PHASE_NEG, "−".into())],
            text: if st.volume_mode == super::VOLUME_ISO {
                format!("sign of ψ̃ · iso ±{:.2} of max|ψ| · only the relative sign of its lobes means anything", st.phase_iso)
            } else {
                "sign of ψ̃ · only the relative sign of its lobes means anything".into()
            },
            tip: format!("{} {}.", ot::KS_NOTE, ot::SIGN_TIP),
            warn: None,
        },
        (ShowMode::Orbitals, sim::OrbitalRequest::Orbital { .. }) => Legend {
            swatches: Vec::new(),
            text: "ψ̃²: where an electron in this orbital is likely to be (the density's colour ramp)".into(),
            tip: ot::KS_NOTE.into(),
            warn: None,
        },
        (ShowMode::Spin, sim::OrbitalRequest::SpinDensity) => Legend {
            swatches: vec![(render::volume::SPIN_UP, "↑ excess".into()), (render::volume::SPIN_DOWN, "↓ excess".into())],
            text: if st.volume_mode == super::VOLUME_ISO { format!("ρ↑ − ρ↓ · iso ±{:.2} of its scale", st.spin_iso) } else { "ρ↑ − ρ↓".into() },
            tip: format!("{} {}", ot::SPIN_NOTE, ot::KS_NOTE),
            warn: None,
        },
        (ShowMode::Coloured, sim::OrbitalRequest::AllColoured { include, physical, look }) => {
            let drawn = ot::drawn_rows(&o.rows, include.as_deref(), look);
            // one chip per colour slot: a ↓ orbital shares its ↑ partner's (one colour per spatial
            // orbital); a slot drawn in one channel only names it
            let mut chips: Vec<(u8, String, [bool; 2])> = Vec::new();
            for r in &drawn {
                let ch = usize::from(r.spin == sim::Spin::Down);
                match chips.iter_mut().find(|c| c.0 == r.slot) {
                    Some(c) => c.2[ch] = true,
                    None => chips.push((r.slot, r.label.to_string(), [ch == 0, ch == 1])),
                }
            }
            let swatches = chips
                .into_iter()
                .map(|(slot, label, spins)| {
                    let c = providers::orbitals::ORBITAL_PALETTE[slot as usize % providers::orbitals::ORBITAL_PALETTE.len()];
                    let name = match spins {
                        [true, false] => format!("{label}↑"),
                        [false, true] => format!("{label}↓"),
                        _ => label,
                    };
                    (c, name)
                })
                .collect();
            let text = match look.enclose {
                Some(pct) => format!("each orbital's boundary surface: the region holding {pct} % of it{}", if look.shell { ", drawn as a shell" } else { "" }),
                None if *physical => "physical scale: hue = which orbital dominates, opacity = the density".into(),
                None => "each orbital scaled to its own peak".into(),
            };
            let warn = ot::touching_line(&drawn);
            Legend { swatches, text, tip: format!("{} One colour per spatial orbital: a ↓ orbital shares its ↑ partner's. Slots repeat beyond 12 orbitals.", ot::KS_NOTE), warn }
        }
        _ => return None,
    })
}

/// The legend chip (bottom-left zone, above the view switch and its sliders). Its caption folds
/// to one line past two (form 0) or always (form 1), with its ⓘ.
fn legend(ctx: &egui::Context, view: Rect, lay: &ViewLayout, st: &UiState, input: &PanelInput<'_>) -> Option<Rect> {
    let l = legend_of(st, input)?;
    let rows = if lay.form(Slot::OrbitalLegend) == 0 { 2 } else { 1 };
    let r = area("orbital-legend", Slot::OrbitalLegend, lay).show(ctx, |ui| {
        theme::card()
            .show(ui, |ui| {
                ui.set_max_width((view.width() * 0.45).clamp(200.0, 520.0));
                ui.horizontal_wrapped(|ui| {
                    for (c, name) in &l.swatches {
                        let (rect, _) = ui.allocate_exact_size(Vec2::splat(11.0), egui::Sense::hover());
                        ui.painter().rect_filled(rect, 3.0, dock::orbitals::swatch(*c));
                        ui.label(RichText::new(name).size(11.5));
                    }
                });
                help_line(ui, &l.text, 11.0, theme::INK_DIM, rows);
                if let Some(w) = &l.warn {
                    ui.add(egui::Label::new(RichText::new(w).size(11.0).color(super::AMBER)).wrap()).on_hover_text(dock::orbitals::TOUCHES_BOX_TIP);
                }
            })
            .response
            .on_hover_text(&l.tip);
    });
    Some(r.response.rect)
}

/// The camera's x (red), y (green), z (blue) axes, seen from the eye.
fn gizmo(ui: &mut egui::Ui, input: &PanelInput<'_>) {
    let (rect, r) = ui.allocate_exact_size(Vec2::splat(46.0), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    p.circle(c, 21.0, Color32::from_rgba_premultiplied(18, 17, 24, 200), Stroke::new(1.0, theme::LINE));
    let view = input.camera.view();
    let mut axes = [
        (glam::Vec3::X, Color32::from_rgb(232, 96, 90), "x"),
        (glam::Vec3::Y, Color32::from_rgb(120, 210, 126), "y"),
        (glam::Vec3::Z, Color32::from_rgb(110, 168, 255), "z"),
    ]
    .map(|(a, col, l)| (view.transform_vector3(a), col, l));
    // far axes first
    axes.sort_by(|a, b| a.0.z.total_cmp(&b.0.z));
    for (v, col, l) in axes {
        let tip: Pos2 = c + Vec2::new(v.x, -v.y) * 15.0;
        p.line_segment([c, tip], Stroke::new(2.0, col));
        p.text(c + Vec2::new(v.x, -v.y) * 19.0, Align2::CENTER_CENTER, l, egui::FontId::monospace(8.5), col);
    }
    r.on_hover_text("the world axes as the camera sees them: x red, y green (up), z blue (View › Display aids › axes)");
}

/// What the bottom-right corner shows this frame, worked out before the layout: the bond
/// references' progress card in place of the pair while they are solved, else the pair's two
/// states; the Orbitals card's progress while its solve runs; the stale bonds' line (the
/// bottom-centre label leaves it out).
struct Corner {
    /// The bond references being solved: the species, the bar's fraction and the time left.
    refs: Option<(u8, f32, Option<f64>)>,
    bonds: BtnState,
    orbitals: BtnState,
    /// The Orbitals card while its solve runs: the bar's fraction and its words.
    card: Option<(f32, String)>,
    line: Option<String>,
}

/// The widest the stale bonds' line over "Bonds now" gets (points): inside the 280-point margin
/// the bottom-centre label leaves at each side, so the two never meet on a wide view.
pub const STALE_LINE_WIDTH: f32 = 260.0;
/// The narrowest the stale line gets to stay beside the view switch (points): three short lines;
/// narrower it would wrap to four or more, so it keeps its width and rises past the switch.
const STALE_LINE_MIN_BESIDE: f32 = 160.0;

/// The estimated view's "Bonds now" (bottom right; user request 2026-10-19): the surrogate's steps
/// carry no density, so the bonds and angles are an earlier DFT step's; this asks for one teacher
/// solve between steps (`Action::RefreshBonds`) and shows the bonds, lengths and angles. It spins
/// until the graph is fresh (at most 20 s). While the bonds on screen are stale (and shown), a
/// small quiet line over it says whose they are (`ui::bonds_label`; user request 2026-11-13).
fn corner(ctx: &egui::Context, st: &mut UiState, input: &PanelInput<'_>) -> Option<Corner> {
    if st.backend != crate::sim::Backend::Estimated {
        return None;
    }
    let s = input.snapshot?;
    if s.scene.nuclei.len() < 2 {
        return None;
    }
    if s.dynamics.bonds_pending.is_empty() {
        st.ref_seen = None;
    }
    let stale = s.dynamics.bonds_stale || s.dynamics.bonds.is_none();
    let busy = st.bonds_refresh.is_some_and(|t| t.elapsed().as_secs_f64() < 20.0) && (stale || st.bonds_refresh.is_some_and(|t| t.elapsed().as_secs_f64() < 0.6));
    if !busy {
        st.bonds_refresh = None;
    }
    // experience's line when it draws over the stale bonds (`crate::bond_experience`), else whose
    // they are
    let line = st.show_bonds.then(|| input.stale_bonds_line.clone().or_else(|| super::bonds_label(&s.dynamics))).flatten().filter(|_| s.dynamics.bonds_pending.is_empty());
    // the bond references come first (an isolated-atom solve per species): the graph cannot be
    // read without them, and a press waits for them in the driver
    let pending = &s.dynamics.bonds_pending;
    if !pending.is_empty() {
        let r = s.dynamics.bond_reference.as_ref();
        let z = r.map_or(pending[0], |r| r.z);
        // smooth: the solves done so far set the pace, the time since the last report fills in
        // between them (never past the next solve's share); the ETA from the same pace
        let now = ctx.input(|i| i.time);
        let seen = *st.ref_seen.get_or_insert((now, r.map_or(0.0, |r| r.elapsed_s)));
        if r.is_some_and(|r| (r.elapsed_s - seen.1).abs() > 1e-9) {
            st.ref_seen = Some((now, r.map_or(0.0, |r| r.elapsed_s)));
        }
        let (t0, e0) = st.ref_seen.unwrap_or(seen);
        let (frac, eta) = match r.filter(|r| r.total > 0) {
            Some(r) => {
                let elapsed = e0 + (now - t0);
                let step = 1.0 / r.total as f64;
                let done = r.done as f64 * step;
                if r.done > 0 {
                    let per = r.elapsed_s / r.done as f64;
                    let frac = (elapsed / (per * r.total as f64)).clamp(done, done + step * 0.98);
                    (frac as f32, Some((per * r.total as f64 - elapsed).max(0.0)))
                } else {
                    (0.0, None)
                }
            }
            None => (0.0, None),
        };
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        return Some(Corner { refs: Some((z, frac, eta)), bonds: BtnState::Off, orbitals: BtnState::Off, card: None, line });
    }
    // two connected icon buttons (user 2026-11-22): Solve bonds | Solve orbitals, each raised and
    // lit when a press would do something, flat with a spinner while its solve runs, flat and
    // greyed when there is nothing to do (or a sampling pass holds the teacher)
    let here: Vec<[f64; 3]> = input.scene.nuclei.iter().map(|n| n.pos).collect();
    let side_running = input.probe.as_ref().is_some_and(|p| p.side.running);
    let orbitals_busy = st.orbitals_press.as_ref().is_some_and(|(t, _)| {
        let e = t.elapsed().as_secs_f64();
        e < 0.6 || (side_running && e < 300.0)
    });
    let orbitals_done = !orbitals_busy && st.orbitals_press.as_ref().is_some_and(|(_, at)| *at == here);
    if !orbitals_busy && !orbitals_done {
        st.orbitals_press = None;
    }
    let sampling = input.sampling();
    let bonds = if sampling { BtnState::Off } else if busy { BtnState::Busy } else if stale { BtnState::Ready } else { BtnState::Off };
    let orbitals = if sampling { BtnState::Off } else if orbitals_busy { BtnState::Busy } else if orbitals_done { BtnState::Off } else { BtnState::Ready };
    let card = (orbitals == BtnState::Busy).then(|| {
        let live = input.probe.as_ref().filter(|p| p.side.running).and_then(|p| p.side.live);
        // Δρ falls from ~1 to the tol on a log scale: the bar's fraction (capped short of full)
        let frac = live.filter(|l| l.tol > 0.0 && l.delta_rho > 0.0).map_or(0.0, |l| ((1.0 / l.delta_rho).ln() / (1.0 / l.tol).ln()).clamp(0.0, 0.97) as f32);
        let words = match live {
            Some(l) => format!("iteration {}", l.iteration),
            None => "starting".into(),
        };
        (frac, words)
    });
    Some(Corner { refs: None, bonds, orbitals, card, line })
}

/// A progress card of the bottom-right corner (the bond references', the Orbitals solve's).
fn corner_card(ui: &mut Ui, title: &str, words: &str, frac: f32) -> egui::Response {
    egui::Frame::new()
        .fill(theme::PANEL_3)
        .stroke(egui::Stroke::new(1.0, theme::LINE))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(14, 9))
        .show(ui, |ui| {
            ui.set_width(220.0);
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(14.0).color(theme::INK));
                ui.label(egui::RichText::new(title).strong().size(14.0).color(theme::INK));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(words).size(12.0).color(theme::INK_DIM));
                });
            });
            ui.add_space(4.0);
            ui.add(egui::ProgressBar::new(frac).desired_height(6.0).fill(theme::ACCENT));
        })
        .response
}

/// The corner's pair (Solve bonds | Solve orbitals) at the view's bottom-right corner, or the bond
/// references' progress card in its place.
fn bonds_pair(ctx: &egui::Context, lay: &ViewLayout, st: &mut UiState, input: &PanelInput<'_>, c: &Corner, acts: &mut Vec<Action>) -> Rect {
    area("bonds-now", Slot::BondsPair, lay)
        .show(ctx, |ui| {
            if let Some((z, frac, eta)) = c.refs {
                let words = match eta {
                    Some(e) if e >= 1.0 => format!("≈ {}:{:02} left", (e / 60.0) as u64, (e % 60.0) as u64),
                    Some(_) => "almost done".into(),
                    None => "starting".into(),
                };
                corner_card(ui, &format!("Bond references: {}", units::elements::symbol(z)), &words, frac)
                    .on_hover_text("the bonds are read against isolated-atom references, one per element, solved in the background first; a press made now runs once they are done");
                return;
            }
            let sampling = input.sampling();
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.horizontal(|ui| {
                let bonds_tip = match c.bonds {
                    BtnState::Busy => "Solving bonds: one DFT solve at the current positions, then the bonds and angles",
                    BtnState::Ready => "Solve bonds: the model's steps carry no density, so the bonds and angles shown are from the last DFT step. This solves the density once at the current positions (between two steps; a running simulation carries on) and shows the bonds, their lengths and the angles",
                    BtnState::Off if sampling => "Solve bonds: not while a sampling pass uses the teacher",
                    BtnState::Off => "Bonds solved at these positions: move an atom (or let the run go on) to solve them again",
                };
                let r = state_button(ui, Icon::Bonds, c.bonds, true).on_hover_text(bonds_tip).on_disabled_hover_text(bonds_tip);
                super::probe("bonds-now", &r);
                if r.clicked() {
                    acts.push(Action::RefreshBonds);
                }
                let orbitals_tip = match c.orbitals {
                    BtnState::Busy => "Solving orbitals: a display-only DFT solve on the background solver at these positions",
                    BtnState::Ready => "Solve orbitals: one display-only DFT solve on the background solver at the current positions, for the density cloud and the orbitals shown (Orbitals tab), even under Model only. It teaches nothing and never delays a step",
                    BtnState::Off if sampling => "Solve orbitals: not while a sampling pass uses the solvers",
                    BtnState::Off => "Orbitals solved at these positions: move an atom (or let the run go on) to solve them again",
                };
                let r = state_button(ui, Icon::Orbitals, c.orbitals, false).on_hover_text(orbitals_tip).on_disabled_hover_text(orbitals_tip);
                super::probe("orbitals-now", &r);
                if r.clicked() {
                    let here: Vec<[f64; 3]> = input.scene.nuclei.iter().map(|n| n.pos).collect();
                    st.orbitals_press = Some((std::time::Instant::now(), here));
                    acts.push(Action::SolveOrbitals);
                }
            });
        })
        .response
        .rect
}

/// The Orbitals card while the display solve runs (above the pair and its stale line).
fn orbitals_card(ctx: &egui::Context, lay: &ViewLayout, c: &Corner) -> Option<Rect> {
    let (frac, words) = c.card.as_ref()?;
    let r = area("orbitals-card", Slot::OrbitalsCard, lay).show(ctx, |ui| {
        corner_card(ui, "Orbitals", words, *frac).on_hover_text("the display-only DFT solve on the background solver: the bar follows its density change down to the tolerance (log scale)");
    });
    Some(r.response.rect)
}

/// The stale bonds' line just over the pair: quiet — small dim text on a faint backing, not
/// interactable (never takes a click from the view), right-aligned with the pair and wrapping on
/// a narrow view — narrower than the room the view switch (`switch_w` wide, bottom-left, level
/// with it) leaves, so it stays just over the pair instead of moving up past the switch.
fn stale_line(ctx: &egui::Context, view: Rect, lay: &ViewLayout, switch_w: f32, text: &str) -> Rect {
    let beside = view.width() - 2.0 * super::zones::INSET_BOTTOM - switch_w - super::zones::GAP;
    let r = area("bonds-stale-line", Slot::StaleLine, lay).interactable(false).show(ctx, |ui| {
        egui::Frame::new()
            .fill(Color32::from_black_alpha(110))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(6, 3))
            .show(ui, |ui| {
                // the frame's 6-point margins inside STALE_LINE_WIDTH (and the view's width)
                // (a view too narrow even for that: full width, and it goes up past the switch)
                let beside = if beside >= STALE_LINE_MIN_BESIDE { beside } else { f32::INFINITY };
                ui.set_max_width((STALE_LINE_WIDTH.min(view.width() - 40.0).min(beside) - 12.0).max(108.0));
                ui.add(egui::Label::new(RichText::new(text).size(11.0).color(theme::INK_DIM)).wrap());
            });
    });
    super::probe("bonds-stale-line", &r.response);
    r.response.rect
}

/// A view button's state (the bonds / orbitals pair, 2026-11-22).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BtnState {
    /// A press would do something: raised and lit.
    Ready,
    /// Its solve runs: flat, a spinner in place of the icon, not pressable.
    Busy,
    /// Nothing to do now (done at these positions, or held): flat and greyed.
    Off,
}

/// One half of the joined pair: `left` rounds the left corners, else the right ones.
fn state_button(ui: &mut Ui, icon: Icon, state: BtnState, left: bool) -> egui::Response {
    let cr = if left { egui::CornerRadius { nw: 8, sw: 8, ne: 0, se: 0 } } else { egui::CornerRadius { nw: 0, sw: 0, ne: 8, se: 8 } };
    let (fill, stroke) = match state {
        BtnState::Ready => (theme::GLOW, Stroke::new(1.5, theme::ACCENT_INK)),
        BtnState::Busy => (theme::PANEL_3, Stroke::new(1.0, theme::ACCENT)),
        BtnState::Off => (theme::PANEL_3, Stroke::new(1.0, theme::LINE)),
    };
    let size = Vec2::new(52.0, 36.0);
    let b = egui::Button::new("").fill(fill).stroke(stroke).corner_radius(cr).min_size(size);
    let r = ui.add_enabled(state == BtnState::Ready, b);
    let inner = Rect::from_center_size(r.rect.center(), Vec2::splat(22.0));
    match state {
        BtnState::Ready => super::icons::paint(ui.painter(), inner, icon, Color32::WHITE),
        BtnState::Busy => {
            egui::Spinner::new().size(18.0).color(theme::INK).paint_at(ui, Rect::from_center_size(r.rect.center(), Vec2::splat(18.0)));
        }
        BtnState::Off => super::icons::paint_disabled(ui.painter(), inner, icon, theme::INK_FAINT),
    }
    if state == BtnState::Busy {
        ui.ctx().request_repaint();
    }
    r.on_hover_cursor(if state == BtnState::Ready { egui::CursorIcon::PointingHand } else { egui::CursorIcon::Default })
}
