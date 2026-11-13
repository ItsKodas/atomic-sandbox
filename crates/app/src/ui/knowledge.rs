//! The knowledge map's view (M8a Task 9; the approved mockup,
//! `docs/superpowers/notes/2026-10-21-m8-mockup.html`, parts (a) and (c)): what the estimated
//! solver's model knows, drawn from `sim::KnowledgeMap` (the worker ships it in
//! `ActivityDelta::knowledge`; `ActivityState::knowledge` keeps the last one). Everything that
//! draws it lives here, self-contained, so the docking redesign can move it without unpicking it:
//!
//! - the **Knowledge panel** ([`tab`]) draws [`body`]: the summary line, the graph (one node per element, its area ∝ the atoms the
//!   fitted samples hold; one line per species pair, coloured by its confidence — green trusted,
//!   amber thin, grey dashed never seen; same-element pairs are loops; line width ∝ log samples),
//!   what the open scene asks (each pair within r_cut, its distances taught / thinly / never), and
//!   the detail of the picked pair (its distance histogram with the "asked now" ticks and the
//!   environment keys behind its verdict) or element (its neighbour counts and bond angles).
//!   The graph zooms (wheel, about the cursor), pans (drag) and fits (its Fit button, a
//!   double-click on empty space, by itself when an element joins); its text keeps its size and
//!   its pair labels sit beside their lines, clear of the nodes ([`graph`]).
//!   A wide panel (a wide leaf, or a detached window) lays the map out in two columns. Its old
//!   pop-out window is gone (docking spec D-5): detaching the panel to an OS window (docking D3)
//!   does that job;
//! - the **3D overlay** (View ▸ Display aids ▸ knowledge overlay, [`overlay_toggle`]): each atom
//!   tinted by `sim::atom_coverage` ([`overlay_coverage`], [`tint_srgb`]: grey 0 never seen → amber
//!   0.5 thin → green 1 well known, the graph's colours) with a legend card ([`legend`]).
//!
//! All of it is display state (`UiState::show_knowledge`, [`KnowledgeView`]): no `Action`, nothing
//! recorded, nothing reaches the worker. It decides nothing: the verdicts are the provider's
//! (`sim::knowledge` module doc, "Confidence"). Distances in Bohr, angles in degrees.
//!
//! **Bins and asked distances** are judged by the same count the overlay uses
//! (`sim::knowledge::atom_coverage`): the histogram's distances in the bin of d and its two
//! neighbours, n, read "taught" when 1 − e^(−n/`COVERAGE_COUNT`) ≥ [`TAUGHT`] (n ≥ 10), "thin" with
//! fewer, "never" with none — so a tick and the bar under it and the atom's tint agree. A display
//! threshold, not the verdict: a trusted pair can have thin distances, and a thin pair's distances
//! can be well sampled (it is thin because an environment holding it is short; the detail lists
//! those environments from the Model status's per-key verdicts).
use super::{dock, theme, PanelInput, UiState, AMBER, GREEN};
use crate::scene::SceneNucleus;
use crate::sim::Backend;
use egui::{pos2, vec2, Align2, Color32, FontId, Id, Order, Pos2, Rect, RichText, Sense, Shape, Stroke, Ui, Vec2};
use sim::knowledge::{angle_bin, first_shell, ElementKnowledge, KnowledgeMap, PairKnowledge, ANGLE_BINS, ANGLE_BIN_DEG, COVERAGE_COUNT};
use sim::Confidence;
use std::collections::BTreeMap;

/// The occupancy (1 − e^(−n/`COVERAGE_COUNT`), the overlay's) at which a distance reads taught.
pub const TAUGHT: f64 = 0.9;

/// What the detail shows: a pair, an element, or (nothing picked) the weakest pair the scene asks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pick {
    #[default]
    None,
    /// (min Z, max Z).
    Pair(u8, u8),
    Element(u8),
}

/// The map's own view state (display only, not an `Action`, not recorded): the pick, and the
/// graph's zoom and pan (user request 2026-11-14; reset on a new scene, `UiState::adopt`).
#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeView {
    pub pick: Pick,
    /// × the base layout, [`ZOOM_MIN`]–[`ZOOM_MAX`]; about the map's centre, then `pan` (points).
    pub zoom: f32,
    pub pan: Vec2,
    /// The elements the view was last fitted to (it fits again when that set changes) and the
    /// zoom that fit gave (zoomed out past it, crowded pair labels fade).
    pub fitted: Option<Vec<u8>>,
    pub fit_zoom: f32,
}

impl Default for KnowledgeView {
    fn default() -> Self {
        KnowledgeView { pick: Pick::None, zoom: 1.0, pan: Vec2::ZERO, fitted: None, fit_zoom: 1.0 }
    }
}

impl KnowledgeView {
    /// A new scene: the graph fits itself again on its next frame (the pick is kept).
    pub fn reset_view(&mut self) {
        *self = KnowledgeView { pick: self.pick, ..Default::default() };
    }
}

/// How a distance is covered (module doc, "Bins and asked distances").
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Taught {
    Never,
    Thin,
    Taught,
}

impl Taught {
    fn of(n: u32) -> Taught {
        if n == 0 {
            Taught::Never
        } else if 1.0 - (-(n as f64) / COVERAGE_COUNT).exp() >= TAUGHT {
            Taught::Taught
        } else {
            Taught::Thin
        }
    }

    fn word(self) -> &'static str {
        match self {
            Taught::Never => "never taught",
            Taught::Thin => "thin",
            Taught::Taught => "taught",
        }
    }

    fn colour(self) -> Color32 {
        match self {
            Taught::Never => theme::INK_FAINT,
            Taught::Thin => AMBER,
            Taught::Taught => GREEN,
        }
    }
}

fn sym(z: u8) -> &'static str {
    render::elements::symbol(z)
}

/// "C–H", "O–H", "C–O", "H–H": H last, else ascending Z.
pub fn pair_name(a: u8, b: u8) -> String {
    let (lo, hi) = (a.min(b), a.max(b));
    if lo == 1 && hi != 1 {
        format!("{}–H", sym(hi))
    } else {
        format!("{}–{}", sym(lo), sym(hi))
    }
}

pub fn confidence_colour(c: Confidence) -> Color32 {
    match c {
        Confidence::Trusted => GREEN,
        Confidence::Thin => AMBER,
        Confidence::Unseen => theme::INK_FAINT,
    }
}

pub fn confidence_word(c: Confidence) -> &'static str {
    match c {
        Confidence::Trusted => "trusted",
        Confidence::Thin => "thin",
        Confidence::Unseen => "never seen",
    }
}

/// The overlay's colour at coverage `c` (sRGB): grey #7a7690 at 0 → amber #e4c044 at ½ → green
/// #6ed278 at 1, linear in between (the mockup's (c)).
pub fn tint_srgb(c: f32) -> [u8; 3] {
    const GREY: [f32; 3] = [122.0, 118.0, 144.0];
    const AMB: [f32; 3] = [228.0, 192.0, 68.0];
    const GRN: [f32; 3] = [110.0, 210.0, 120.0];
    let c = if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 };
    let (a, b, t) = if c < 0.5 { (GREY, AMB, c * 2.0) } else { (AMB, GRN, (c - 0.5) * 2.0) };
    std::array::from_fn(|k| (a[k] + t * (b[k] - a[k])).round() as u8)
}

/// The atom pairs of the histogram around d (the bin of d and its two neighbours): the count
/// `sim::atom_coverage` reads.
fn occupancy(p: &PairKnowledge, d: f64) -> u32 {
    let k = (d / p.hist.bin) as usize;
    (k.saturating_sub(1)..=k + 1).filter_map(|b| p.hist.counts.get(b)).sum()
}

/// The knowledge overlay's per-atom coverage of `nuclei` (empty: overlay off or no map).
pub fn overlay_coverage(st: &UiState, nuclei: &[SceneNucleus]) -> Vec<f32> {
    let Some(map) = st.activity.knowledge.as_deref().filter(|_| st.show_knowledge) else { return Vec::new() };
    let z: Vec<u8> = nuclei.iter().map(|n| n.z).collect();
    let pos: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
    sim::atom_coverage(map, &z, &pos)
}

/// The map shown: the last one shipped, with the estimated solver only.
fn map_of(st: &UiState, input: &PanelInput<'_>) -> Option<std::sync::Arc<KnowledgeMap>> {
    st.activity.knowledge.clone().filter(|_| input.snapshot.is_none_or(|s| s.backend == Backend::Estimated))
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Every species pair the scene has within r_cut, with its distances (ascending).
fn scene_pairs(nuclei: &[SceneNucleus], r_cut: f64) -> BTreeMap<(u8, u8), Vec<f64>> {
    let mut out: BTreeMap<(u8, u8), Vec<f64>> = BTreeMap::new();
    for i in 0..nuclei.len() {
        for j in i + 1..nuclei.len() {
            let d = dist(nuclei[i].pos, nuclei[j].pos);
            if d < r_cut {
                let (a, b) = (nuclei[i].z, nuclei[j].z);
                out.entry((a.min(b), a.max(b))).or_default().push(d);
            }
        }
    }
    for v in out.values_mut() {
        v.sort_by(f64::total_cmp);
    }
    out
}

/// What the scene asks of one pair: per coverage class, (count, shortest, longest), weakest first.
fn asked(map: &KnowledgeMap, z: (u8, u8), ds: &[f64]) -> Vec<(Taught, usize, f64, f64)> {
    let mut by: BTreeMap<Taught, (usize, f64, f64)> = BTreeMap::new();
    for &d in ds {
        let t = map.pair(z.0, z.1).map_or(Taught::Never, |p| Taught::of(occupancy(p, d)));
        let e = by.entry(t).or_insert((0, d, d));
        e.0 += 1;
        e.1 = e.1.min(d);
        e.2 = e.2.max(d);
    }
    by.into_iter().map(|(t, (n, lo, hi))| (t, n, lo, hi)).collect()
}

/// "8 at 2.06 taught · 10 at 4.05–4.11 thin".
fn asked_words(rows: &[(Taught, usize, f64, f64)]) -> Vec<(String, Color32)> {
    rows.iter()
        .map(|&(t, n, lo, hi)| {
            let at = if hi - lo < 0.005 { format!("{lo:.2}") } else { format!("{lo:.2}–{hi:.2}") };
            (format!("{n} at {at} {}", t.word()), t.colour())
        })
        .collect()
}

/// The pair the detail shows when nothing is picked: the weakest the scene asks (unseen, then
/// thin, then the most distances never taught), else the map's first unseen or thin pair.
fn default_pair(map: &KnowledgeMap, asks: &BTreeMap<(u8, u8), Vec<f64>>) -> Option<(u8, u8)> {
    let rank = |c: Confidence| match c {
        Confidence::Unseen => 0,
        Confidence::Thin => 1,
        Confidence::Trusted => 2,
    };
    let conf = |z: (u8, u8)| map.pair(z.0, z.1).map_or(Confidence::Unseen, |p| p.confidence);
    asks.iter()
        .min_by_key(|(z, ds)| (rank(conf(**z)), std::cmp::Reverse(asked(map, **z, ds).iter().filter(|r| r.0 != Taught::Taught).map(|r| r.1).sum::<usize>())))
        .map(|(z, _)| *z)
        .or_else(|| map.pairs.iter().min_by_key(|p| rank(p.confidence)).map(|p| p.z))
}

/// The strip button's hover text.
pub fn peek(st: &UiState) -> String {
    match st.activity.knowledge.as_deref() {
        Some(m) => {
            let n = |c: Confidence| m.pairs.iter().filter(|p| p.confidence == c).count();
            format!(
                "Knowledge\n{} samples · {} · pairs {} trusted · {} thin · {} never seen",
                m.samples,
                m.elements.iter().map(|e| sym(e.z)).collect::<Vec<_>>().join(" "),
                n(Confidence::Trusted),
                n(Confidence::Thin),
                n(Confidence::Unseen)
            )
        }
        None => "Knowledge (estimated solver)\nno map yet: it is built when the model is fitted".into(),
    }
}

/// View ▸ Display aids ▸ knowledge overlay (the View panel, `dock::view`; display only, not
/// recorded; kept in the profile).
pub fn overlay_toggle(ui: &mut Ui, st: &mut UiState) {
    super::probed("show-knowledge", ui.checkbox(&mut st.show_knowledge, "knowledge overlay")).on_hover_text(
        "tints each atom by how well the estimated solver's model knows its surroundings: 0 never seen (grey) … 0.5 thin (amber) … 1 well known (green), the knowledge map's colours. Per atom it is its least-known neighbour within the model's cutoff — that pair's verdict and how many distances like this one were taught. Needs the estimated solver's map (the Knowledge panel). Display only, not recorded",
    );
}

/// The panel at least this wide draws the map in two columns (the old pop-out window's layout).
const WIDE: f32 = 760.0;

/// The Knowledge panel.
pub fn tab(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>) {
    dock::group(ui, st, "d-knowledge", |ui, st| {
        theme::section(ui, "Shared model");
        let wide = ui.available_width() >= WIDE;
        body(ui, st, input, wide);
    });
}

/// The map: summary, graph, what the scene asks and the detail (one column; two when `wide`).
pub fn body(ui: &mut Ui, st: &mut UiState, input: &PanelInput<'_>, wide: bool) {
    let Some(map) = map_of(st, input) else {
        let text = if input.snapshot.is_some_and(|s| s.backend != Backend::Estimated) || st.backend != Backend::Estimated {
            "The knowledge map is the estimated solver's: what its model has learned, by element and pair. Pick Estimated to see it."
        } else {
            "No knowledge map yet: it is built from the model's fitted samples once a model is fitted (a sampling pass, a loaded model or the shared collection)."
        };
        theme::note(ui, text);
        return;
    };
    let map = &*map;
    if map.elements.is_empty() {
        theme::note(ui, "The model's fit holds no samples yet: nothing is known.");
        return;
    }
    let asks = scene_pairs(&input.scene.nuclei, map.r_cut);
    summary(ui, map);
    if wide {
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(460.0);
                graph(ui, st, map, vec2(450.0, 360.0));
                graph_legend(ui);
                scene_asks(ui, st, map, &asks);
            });
            ui.add_space(14.0);
            ui.vertical(|ui| {
                ui.set_min_width(420.0);
                detail(ui, st, map, &asks, input);
            });
        });
    } else {
        let w = ui.available_width().max(240.0);
        graph(ui, st, map, vec2(w, (w * 0.58).clamp(150.0, 280.0)));
        graph_legend(ui);
        scene_asks(ui, st, map, &asks);
        detail(ui, st, map, &asks, input);
    }
}

fn summary(ui: &mut Ui, map: &KnowledgeMap) {
    let slug = if map.slug.is_empty() { "a per-scene model (not a shared collection)".to_string() } else { map.slug.clone() };
    ui.label(RichText::new(slug).monospace().size(10.5).color(theme::INK_FAINT));
    let n = |c: Confidence| map.pairs.iter().filter(|p| p.confidence == c).count();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let mono = |t: String, c: Color32| RichText::new(t).monospace().size(11.0).color(c);
        ui.label(mono(
            format!(
                "{} samples · {} · r_cut {:.1} Bohr ({:.1} Å) · pairs",
                map.samples,
                map.elements.iter().map(|e| sym(e.z)).collect::<Vec<_>>().join(" "),
                map.r_cut,
                units::bohr_to_angstrom(map.r_cut)
            ),
            theme::INK,
        ));
        ui.label(mono(format!("{} trusted", n(Confidence::Trusted)), GREEN));
        ui.label(mono("·".into(), theme::INK_FAINT));
        ui.label(mono(format!("{} thin", n(Confidence::Thin)), AMBER));
        ui.label(mono("·".into(), theme::INK_FAINT));
        ui.label(mono(format!("{} never seen", n(Confidence::Unseen)), theme::INK_DIM));
    });
}

fn graph_legend(ui: &mut Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        for (c, t) in [(GREEN, "trusted"), (AMBER, "thin"), (theme::INK_FAINT, "never seen")] {
            let (r, _) = ui.allocate_exact_size(vec2(14.0, 10.0), Sense::hover());
            ui.painter().line_segment([r.left_center(), r.right_center()], Stroke::new(3.0, c));
            ui.label(RichText::new(t).size(11.0).color(theme::INK_DIM));
        }
        ui.label(RichText::new("· node area = atoms taught · line width = samples · click a line or an element").size(11.0).color(theme::INK_FAINT));
    });
}

/// Where each element's node sits and its radius (points).
fn layout(map: &KnowledgeMap, rect: Rect) -> Vec<(u8, Pos2, f32)> {
    let n = map.elements.len();
    let c = rect.center() + vec2(0.0, 6.0);
    let span = rect.width().min(rect.height());
    let ring = if n <= 1 { 0.0 } else { (0.30 * span).min(0.36 * rect.width()) };
    let most = map.elements.iter().map(|e| e.atoms_seen).max().unwrap_or(1).max(1) as f32;
    let (r_min, r_max) = (9.0, (0.11 * span).max(14.0));
    map.elements
        .iter()
        .enumerate()
        .map(|(k, e)| {
            let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * k as f32 / n.max(1) as f32 + if n == 2 { std::f32::consts::FRAC_PI_2 } else { 0.0 };
            let at = c + vec2(a.cos(), a.sin()) * ring;
            (e.z, at, r_min + (r_max - r_min) * (e.atoms_seen as f32 / most).sqrt())
        })
        .collect()
}

/// A same-element pair's loop: its centre and radius, outside the node, away from the middle.
fn loop_of(node: Pos2, r: f32, middle: Pos2) -> (Pos2, f32) {
    let out = (node - middle).normalized();
    let out = if out.length() > 0.5 { out } else { vec2(0.0, -1.0) };
    let lr = (0.55 * r).max(8.0);
    (node + out * (r + 0.6 * lr), lr)
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}

/// The graph's zoom range (× the base layout).
pub const ZOOM_MIN: f32 = 0.25;
pub const ZOOM_MAX: f32 = 8.0;
/// The zoom factor per wheel line (a pixel wheel's 50 points count as a line).
const WHEEL_STEP: f32 = 1.2;
/// Room (points) Fit leaves round the nodes, and below them for their "atoms" line.
const FIT_MARGIN: f32 = 14.0;
const FIT_BELOW: f32 = 16.0;
/// Fit never magnifies past this (one element alone would fill the map).
const FIT_MAX: f32 = 1.25;
/// The Fit button (top right of the map).
const FIT_BUTTON: Vec2 = vec2(30.0, 18.0);

/// The view's transform: a base-layout point on screen (zoom about the map's centre `o`, then pan).
fn to_screen(v: &KnowledgeView, o: Pos2, q: Pos2) -> Pos2 {
    o + v.pan + (q - o) * v.zoom
}

/// Zooms by `factor` about the screen point `at` (it stays where it is), within the zoom range.
pub(crate) fn zoom_about(v: &mut KnowledgeView, o: Pos2, at: Pos2, factor: f32) {
    let zoom = (v.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    let ratio = zoom / v.zoom;
    v.pan = (at - o) - (at - o - v.pan) * ratio;
    v.zoom = zoom;
}

/// What Fit frames, in base-layout coordinates: every node disc and every same-element loop.
fn extent(map: &KnowledgeMap, base: &[(u8, Pos2, f32)], middle: Pos2) -> Rect {
    let mut r = Rect::NOTHING;
    for &(_, at, rad) in base {
        r = r.union(Rect::from_center_size(at, Vec2::splat(2.0 * rad)));
    }
    for q in map.pairs.iter().filter(|q| q.z.0 == q.z.1) {
        if let Some(&(_, at, rad)) = base.iter().find(|n| n.0 == q.z.0) {
            let (c, lr) = loop_of(at, rad, middle);
            r = r.union(Rect::from_center_size(c, Vec2::splat(2.0 * lr)));
        }
    }
    r
}

/// Fit: the zoom and pan that frame [`extent`] inside `rect` (its margins, the "atoms" line under
/// the nodes), at most [`FIT_MAX`].
fn fit(v: &mut KnowledgeView, ext: Rect, rect: Rect) {
    let o = rect.center();
    let room = vec2(rect.width() - 2.0 * FIT_MARGIN, rect.height() - 2.0 * FIT_MARGIN - FIT_BELOW).max(Vec2::splat(1.0));
    let zoom = (room.x / ext.width().max(1.0)).min(room.y / ext.height().max(1.0)).clamp(ZOOM_MIN, FIT_MAX);
    let target = o - vec2(0.0, FIT_BELOW / 2.0);
    v.zoom = zoom;
    v.pan = target - o - (ext.center() - o) * zoom;
    v.fit_zoom = zoom;
}

/// A pair label's place beside its line `a`–`b`: the first of a few spots along it (the middle
/// first), on either side, whose rect (`size`, moved inside `bounds`) is clear of every disc in
/// `avoid` and every label already placed; else the middle of the line. Deterministic; no force
/// layout. Returns the rect and whether it is clear.
fn beside(a: Pos2, b: Pos2, size: Vec2, bounds: Rect, avoid: &[(Pos2, f32)], placed: &[Rect]) -> (Rect, bool) {
    let d = (b - a).normalized();
    let n = d.rot90();
    let half = size / 2.0;
    let off = n.x.abs() * half.x + n.y.abs() * half.y + 3.0;
    for t in [0.5, 0.38, 0.62, 0.26, 0.74] {
        for side in [1.0, -1.0] {
            let r = within(Rect::from_center_size(a + (b - a) * t + n * side * off, size), bounds);
            if clear(r, avoid, placed) {
                return (r, true);
            }
        }
    }
    (within(Rect::from_center_size(a + (b - a) * 0.5, size), bounds), false)
}

/// A loop's label: outside the loop (`c`, radius `lr`), along `out` (away from its node) or, where
/// that meets a disc or a label, turned from it by 45°, 90°, 135° either way, then a little
/// farther out — the first that is clear; else along `out`. Each spot is moved inside `bounds` first (a loop at the map's edge).
fn outside(c: Pos2, lr: f32, out: Vec2, size: Vec2, bounds: Rect, avoid: &[(Pos2, f32)], placed: &[Rect]) -> (Rect, bool) {
    let half = size / 2.0;
    let at = |d: Vec2, gap: f32| Rect::from_center_size(c + d * (lr + gap + d.x.abs() * half.x + d.y.abs() * half.y), size);
    let turn = |deg: f32| {
        let (s, k) = deg.to_radians().sin_cos();
        vec2(out.x * k - out.y * s, out.x * s + out.y * k)
    };
    for gap in [4.0, 14.0, 24.0] {
        for deg in [0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0] {
            let r = within(at(turn(deg), gap), bounds);
            if clear(r, avoid, placed) {
                return (r, true);
            }
        }
    }
    (within(at(out, 4.0), bounds), false)
}

/// `r` moved (not resized) to lie inside `bounds` where it fits.
fn within(r: Rect, bounds: Rect) -> Rect {
    let dx = (bounds.left() - r.left()).max(0.0) + (bounds.right() - r.right()).min(0.0);
    let dy = (bounds.top() - r.top()).max(0.0) + (bounds.bottom() - r.bottom()).min(0.0);
    r.translate(vec2(dx, dy))
}

fn clear(r: Rect, avoid: &[(Pos2, f32)], placed: &[Rect]) -> bool {
    avoid.iter().all(|&(c, rad)| r.distance_to_pos(c) > rad + 2.0) && placed.iter().all(|p| !p.expand(2.0).intersects(r))
}

/// The graph (module doc): nodes, lines and loops, labels; a click on a line, a label or a node
/// picks it. The view zooms with the wheel about the cursor (only while the pointer is over the
/// map: elsewhere the panel scrolls), pans with a drag (a click without moving still picks), and
/// Fit (its button, a double-click on empty space; by itself when first shown and when an element
/// joins) frames every node. Geometry scales with the zoom; the text keeps its size, and pair
/// labels fade out when zoomed out past the fit so far that they would overlap.
fn graph(ui: &mut Ui, st: &mut UiState, map: &KnowledgeMap, size: Vec2) {
    let (rect, bg) = ui.allocate_exact_size(size, Sense::click_and_drag());
    super::probe("knowledge-map", &bg);
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 6.0, Color32::from_black_alpha(50));
    let o = rect.center();
    let base = layout(map, rect);
    let base_middle = rect.center() + vec2(0.0, 6.0);
    let ext = extent(map, &base, base_middle);
    // auto-fit: first shown, or a new set of elements (not on every refresh of the same map)
    let zs: Vec<u8> = base.iter().map(|n| n.0).collect();
    if st.knowledge.fitted.as_deref() != Some(&zs[..]) {
        fit(&mut st.knowledge, ext, rect);
        st.knowledge.fitted = Some(zs);
    }
    // the wheel, over the map only (taken from the panel's scroll then)
    if ui.rect_contains_pointer(rect) {
        let (lines, at) = ui.input_mut(|i| {
            let lines: f32 = i
                .events
                .iter()
                .map(|e| match e {
                    egui::Event::MouseWheel { unit, delta, .. } => match unit {
                        egui::MouseWheelUnit::Line => delta.y,
                        egui::MouseWheelUnit::Point => delta.y / 50.0,
                        egui::MouseWheelUnit::Page => delta.y * 10.0,
                    },
                    _ => 0.0,
                })
                .sum();
            i.smooth_scroll_delta = Vec2::ZERO;
            (lines, i.pointer.hover_pos())
        });
        if let (true, Some(at)) = (lines != 0.0, at) {
            zoom_about(&mut st.knowledge, o, at, WHEEL_STEP.powf(lines));
        }
    }
    // a drag pans (a click, which moves less than egui's drag threshold, still picks below)
    if bg.dragged() {
        st.knowledge.pan += bg.drag_delta();
    }
    let v = st.knowledge.clone();
    let nodes: Vec<(u8, Pos2, f32, f32)> = base.iter().map(|&(z, at, r)| (z, to_screen(&v, o, at), r * v.zoom, r)).collect();
    let middle = to_screen(&v, o, base_middle);
    let node = |z: u8| nodes.iter().find(|n| n.0 == z).map(|n| (n.1, n.2));
    let most = map.pairs.iter().map(|q| q.samples).max().unwrap_or(1).max(1) as f32;
    let picked = v.pick;
    let font = FontId::monospace(10.0);
    // the lines (and loops) under the nodes; each with its label
    let mut labels: Vec<((u8, u8), String, Color32, Option<(Pos2, Pos2)>, Option<(Pos2, f32, Vec2)>)> = Vec::new();
    let mut shapes: Vec<(u8, u8, Option<(Pos2, Pos2)>, Option<(Pos2, f32)>)> = Vec::new();
    let mut avoid: Vec<(Pos2, f32)> = nodes.iter().map(|n| (n.1, n.2)).collect();
    for q in &map.pairs {
        let (Some((a, ra)), Some((b, _))) = (node(q.z.0), node(q.z.1)) else { continue };
        let colour = confidence_colour(q.confidence);
        let width = if q.samples == 0 { 2.0 } else { 2.0 + 4.5 * (1.0 + q.samples as f32).ln() / (1.0 + most).ln() };
        let stroke = Stroke::new(width, colour);
        let halo = Stroke::new(width + 7.0, theme::ACCENT.gamma_multiply(0.35));
        let on = picked == Pick::Pair(q.z.0, q.z.1);
        let label = format!(
            "{} · {}",
            pair_name(q.z.0, q.z.1),
            match q.confidence {
                Confidence::Unseen => "never".to_string(),
                c => confidence_word(c).to_string(),
            }
        );
        if q.z.0 == q.z.1 {
            let (c, lr) = loop_of(a, ra, middle);
            let pts: Vec<Pos2> = (0..=48).map(|k| c + Vec2::angled(std::f32::consts::TAU * k as f32 / 48.0) * lr).collect();
            if on {
                p.add(Shape::line(pts.clone(), halo));
            }
            if q.confidence == Confidence::Unseen {
                p.extend(Shape::dashed_line(&pts, stroke, 5.0, 4.0));
            } else {
                p.add(Shape::line(pts, stroke));
            }
            let out = (c - a).normalized();
            labels.push((q.z, label, colour, None, Some((c, lr, out))));
            shapes.push((q.z.0, q.z.1, None, Some((c, lr))));
            avoid.push((c, lr));
        } else {
            if on {
                p.line_segment([a, b], halo);
            }
            if q.confidence == Confidence::Unseen {
                p.extend(Shape::dashed_line(&[a, b], stroke, 5.0, 4.0));
            } else {
                p.line_segment([a, b], stroke);
            }
            labels.push((q.z, label, colour, Some((a, b)), None));
            shapes.push((q.z.0, q.z.1, Some((a, b)), None));
        }
    }
    // the nodes (their letters and counts at a fixed size: the base radius's, not the zoom's)
    let mut hits: Vec<(Pick, egui::Response)> = Vec::new();
    for &(z, at, r, r0) in &nodes {
        let [cr, cg, cb] = render::elements::colour_srgb(z);
        let fill = Color32::from_rgb(cr, cg, cb);
        if picked == Pick::Element(z) {
            p.circle_filled(at, r + 5.0, theme::ACCENT.gamma_multiply(0.35));
        }
        p.circle(at, r, fill, Stroke::new(1.0, Color32::from_white_alpha(64)));
        let light = 0.299 * cr as f32 + 0.587 * cg as f32 + 0.114 * cb as f32 > 150.0;
        p.text(at, Align2::CENTER_CENTER, sym(z), FontId::proportional((r0 * 0.9).clamp(11.0, 17.0)), if light { Color32::from_rgb(23, 17, 42) } else { Color32::WHITE });
        let seen = map.element(z).map_or(0, |e| e.atoms_seen);
        p.text(at + vec2(0.0, r + 3.0), Align2::CENTER_TOP, format!("{seen} atoms"), font.clone(), theme::INK_DIM);
        let resp = ui.interact(Rect::from_center_size(at, Vec2::splat(2.0 * r)).intersect(rect), ui.id().with(("knowledge-node", z)), Sense::click());
        super::probe(&format!("knowledge-element-{z}"), &resp);
        super::probe_rect(&format!("knowledge-node-{z}"), Rect::from_center_size(at, Vec2::splat(2.0 * r)));
        hits.push((Pick::Element(z), resp.on_hover_text(format!("{}: {seen} atoms in the fitted samples — click for its neighbours and bond angles", sym(z)))));
    }
    // the labels, clear of the nodes and loops where they can be; zoomed out past the fit so far
    // that they would overlap, they fade out (the lines and the node letters stay)
    let mut placed: Vec<(Rect, (u8, u8), std::sync::Arc<egui::Galley>, Color32)> = Vec::new();
    let mut crowded = false;
    for (z, text, colour, seg, lp) in labels {
        let g = p.layout_no_wrap(text, font.clone(), colour);
        let size = g.size() + vec2(8.0, 4.0);
        let rects: Vec<Rect> = placed.iter().map(|l| l.0).collect();
        let (r, ok) = match (seg, lp) {
            (Some((a, b)), _) => beside(a, b, size, rect.shrink(2.0), &avoid, &rects),
            (_, Some((c, lr, out))) => outside(c, lr, out, size, rect.shrink(2.0), &avoid, &rects),
            _ => continue,
        };
        crowded |= !ok && rects.iter().any(|q| q.intersects(r));
        placed.push((r, z, g, colour));
    }
    let fade = crowded && v.zoom < v.fit_zoom - 1e-3;
    let alpha = ui.ctx().animate_bool_with_time(ui.id().with("knowledge-labels"), !fade, 0.15);
    if alpha > 0.05 {
        for (r, z, g, colour) in placed {
            p.rect_filled(r, 3.0, Color32::from_rgba_unmultiplied(13, 12, 18, 215).gamma_multiply(alpha));
            p.galley(r.min + vec2(4.0, 2.0), g, colour.gamma_multiply(alpha));
            let resp = ui.interact(r.intersect(rect), ui.id().with(("knowledge-pair", z)), Sense::click());
            super::probe(&format!("knowledge-pair-{}-{}", z.0, z.1), &resp);
            hits.push((Pick::Pair(z.0, z.1), resp));
        }
    }
    // Fit: a small button in the top-right corner
    let fit_rect = Rect::from_min_size(pos2(rect.right() - FIT_BUTTON.x - 4.0, rect.top() + 4.0), FIT_BUTTON);
    let fit_resp = ui.interact(fit_rect, ui.id().with("knowledge-fit"), Sense::click());
    super::probe("knowledge-fit", &fit_resp);
    p.rect(fit_rect, 4.0, if fit_resp.hovered() { theme::PANEL_3 } else { theme::PANEL_2 }, Stroke::new(1.0, theme::LINE), egui::StrokeKind::Inside);
    p.text(fit_rect.center(), Align2::CENTER_CENTER, "Fit", FontId::proportional(10.5), theme::INK_DIM);
    if fit_resp.on_hover_text("frame every element (also: double-click empty space); wheel zooms about the cursor, drag pans").clicked() {
        fit(&mut st.knowledge, ext, rect);
    }
    for (pick, r) in &hits {
        if r.clicked() {
            st.knowledge.pick = *pick;
        }
    }
    // a click on a line itself (not its label); a double-click away from every line fits
    let near_line = |at: Pos2| {
        shapes
            .iter()
            .map(|&(a, b, seg, lp)| {
                let d = match (seg, lp) {
                    (Some((p0, p1)), _) => seg_dist(at, p0, p1),
                    (_, Some((c, lr))) => (at.distance(c) - lr).abs(),
                    _ => f32::INFINITY,
                };
                (d, a, b)
            })
            .min_by(|x, y| x.0.total_cmp(&y.0))
            .filter(|x| x.0 < 8.0)
    };
    if bg.clicked() {
        if let Some((_, a, b)) = bg.interact_pointer_pos().and_then(near_line) {
            st.knowledge.pick = Pick::Pair(a, b);
        }
    }
    // (a third quick click counts too: egui reports it as a triple)
    if (bg.double_clicked() || bg.triple_clicked()) && bg.interact_pointer_pos().is_some_and(|at| near_line(at).is_none()) {
        fit(&mut st.knowledge, ext, rect);
    }
}

/// "This scene asks": each pair the open scene has within r_cut, weakest first; a click picks it.
fn scene_asks(ui: &mut Ui, st: &mut UiState, map: &KnowledgeMap, asks: &BTreeMap<(u8, u8), Vec<f64>>) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        theme::section(ui, "This scene asks");
        ui.label(RichText::new(format!("pairs within {:.1} Bohr", map.r_cut)).size(10.5).color(theme::INK_FAINT));
    });
    if asks.is_empty() {
        theme::note(ui, "no two atoms of the open scene are within the model's cutoff: nothing to ask");
        return;
    }
    let conf = |z: (u8, u8)| map.pair(z.0, z.1).map_or(Confidence::Unseen, |p| p.confidence);
    let mut rows: Vec<(&(u8, u8), &Vec<f64>)> = asks.iter().collect();
    rows.sort_by_key(|(z, _)| match conf(**z) {
        Confidence::Unseen => 0,
        Confidence::Thin => 1,
        Confidence::Trusted => 2,
    });
    egui::Grid::new(ui.id().with("knowledge-asks")).num_columns(2).spacing(vec2(10.0, 3.0)).show(ui, |ui| {
        for (z, ds) in rows {
            let c = conf(*z);
            let name = ui.add(egui::Button::selectable(st.knowledge.pick == Pick::Pair(z.0, z.1), RichText::new(pair_name(z.0, z.1)).monospace().size(11.5)));
            if name.on_hover_text("show this pair's distances").clicked() {
                st.knowledge.pick = Pick::Pair(z.0, z.1);
            }
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(RichText::new(confidence_word(c)).size(11.5).color(confidence_colour(c)));
                for (t, colour) in asked_words(&asked(map, *z, ds)) {
                    ui.label(RichText::new("·").size(11.5).color(theme::INK_FAINT));
                    ui.label(RichText::new(t).size(11.5).color(colour));
                }
                ui.label(RichText::new("Bohr").size(10.5).color(theme::INK_FAINT));
            });
            ui.end_row();
        }
    });
}

/// The detail of the pick (or of the weakest pair the scene asks).
fn detail(ui: &mut Ui, st: &mut UiState, map: &KnowledgeMap, asks: &BTreeMap<(u8, u8), Vec<f64>>, input: &PanelInput<'_>) {
    ui.add_space(6.0);
    match st.knowledge.pick {
        Pick::Element(z) => match map.element(z) {
            Some(e) => element_detail(ui, map, e, input),
            None => {
                theme::note(ui, format!("{} is not in the model's fitted samples", sym(z)));
            }
        },
        pick => {
            let (z, why) = match pick {
                Pick::Pair(a, b) => ((a, b), "the pair you picked"),
                _ => match default_pair(map, asks) {
                    Some(z) => (z, if asks.contains_key(&z) { "the weakest pair this scene asks" } else { "the weakest pair the model holds" }),
                    None => return,
                },
            };
            match map.pair(z.0, z.1) {
                Some(p) => pair_detail(ui, map, p, why, asks.get(&z).map_or(&[][..], |v| &v[..]), input),
                None => {
                    theme::note(ui, format!("{} is not a pair of the model's species", pair_name(z.0, z.1)));
                }
            }
        }
    }
}

/// The environment keys behind a pair's verdict, from the Model status's per-key verdicts: every
/// key in which one species sees the other, unlearned first. (class, verdict) pairs.
fn keys_of<'a>(input: &'a PanelInput<'_>, z: (u8, u8)) -> Vec<(&'a str, &'a sim::SpeciesVerdict)> {
    let Some(m) = input.model() else { return Vec::new() };
    let mut v: Vec<(&str, &sim::SpeciesVerdict)> = m
        .class_loo
        .iter()
        .flat_map(|c| c.species.iter().map(move |s| (c.class, s)))
        .filter(|(_, s)| (s.z == z.0 && s.neighbours.contains(&z.1)) || (s.z == z.1 && s.neighbours.contains(&z.0)))
        .collect();
    v.sort_by_key(|(_, s)| s.trusted);
    v
}

fn pair_detail(ui: &mut Ui, map: &KnowledgeMap, p: &PairKnowledge, why: &str, ds: &[f64], input: &PanelInput<'_>) {
    let total: u32 = p.hist.counts.iter().sum();
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(pair_name(p.z.0, p.z.1)).strong().size(14.0));
        ui.label(RichText::new(format!("{why} · {} samples · {total} distances taught", p.samples)).monospace().size(11.0).color(theme::INK_DIM));
    });
    let c = p.confidence;
    let verdict = match c {
        Confidence::Trusted => "trusted: the model answers every environment this pair was met in".to_string(),
        Confidence::Thin => "thin: taught, but not everywhere it was met — the model refuses the environments below".to_string(),
        Confidence::Unseen => format!("never seen: no fitted sample has these two within {:.1} Bohr", map.r_cut),
    };
    ui.label(RichText::new(verdict).size(12.0).color(confidence_colour(c)));
    // why: the model's own trust rule, then the keys
    if c != Confidence::Unseen {
        if let Some(m) = input.model() {
            if !m.trusted {
                let reason = m.untrusted_reason.as_deref().unwrap_or("its trust rule fails");
                ui.add(egui::Label::new(RichText::new(format!("the model as a whole is not trusted yet ({reason}): no pair reads trusted until it is")).size(11.5).color(AMBER)).wrap());
            }
        }
        let keys = keys_of(input, p.z);
        if keys.is_empty() {
            if c == Confidence::Thin {
                theme::note(ui, "the per-environment verdicts are not in this snapshot (Model tab ▸ learned classes)");
            }
        } else {
            let (short, learned): (Vec<_>, Vec<_>) = keys.iter().partition(|(_, s)| !s.trusted);
            for (class, s) in &short {
                ui.add(egui::Label::new(RichText::new(super::activity::words::species_line(class, s)).size(11.5).color(AMBER)).wrap());
            }
            if !learned.is_empty() {
                let names: Vec<String> = learned.iter().map(|(class, s)| format!("{} {}", if *class == "pair" { "pair" } else { "many-body" }, s.words())).collect();
                ui.add(egui::Label::new(RichText::new(format!("learned: {}", names.join("; "))).size(11.0).color(theme::INK_DIM)).wrap())
                    .on_hover_text("the environment keys holding this pair that the model answers: (central atom, its neighbours' species within the cutoff)");
            }
            if !short.is_empty() {
                theme::note(ui, "train scenes with these environments (a sampling pass of a molecule that has them) to teach them");
            }
        }
    }
    // the histogram, with the scene's distances
    let bins = p.hist.counts.len();
    let colour_of = |k: usize| -> Option<Color32> {
        (p.hist.counts[k] > 0).then(|| Taught::of(occupancy(p, (k as f64 + 0.5) * p.hist.bin)).colour())
    };
    let x_max = (bins as f64 * p.hist.bin).max(map.r_cut);
    let labels: Vec<(f64, String)> = (1..=x_max.floor() as usize).map(|x| (x as f64, x.to_string())).collect();
    let w = ui.available_width().clamp(240.0, 640.0);
    histogram(
        ui,
        &p.hist.counts,
        p.hist.bin,
        x_max,
        &colour_of,
        ds,
        &labels,
        ("compressed ←", &format!("→ stretched · r_cut {:.1} Bohr", map.r_cut)),
        vec2(w, if w > 400.0 { 140.0 } else { 86.0 }),
        (total == 0).then_some("nothing taught"),
    );
    if total > 0 {
        let lo = p.hist.counts.iter().position(|&n| n > 0).map_or(0.0, |k| k as f64 * p.hist.bin);
        let hi = p.hist.counts.iter().rposition(|&n| n > 0).map_or(0.0, |k| (k + 1) as f64 * p.hist.bin);
        let mut note = format!("taught {lo:.1}–{hi:.1} Bohr");
        if let Some(r) = first_shell(&p.hist) {
            note += &format!(" · first peak ends at {r:.2} Bohr (the neighbours the bond angles count; for a pair never bonded it is the nearest contact, not a bond)");
        }
        theme::note(ui, note);
    } else {
        theme::note(ui, format!("no distances taught: no fitted sample has these two within {:.1} Bohr. Train a scene where they meet to teach it.", map.r_cut));
    }
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label(RichText::new("asked now (the open scene):").size(11.5).strong());
        if ds.is_empty() {
            ui.label(RichText::new("none").size(11.5).color(theme::INK_FAINT));
        }
        for (k, (t, colour)) in asked_words(&asked(map, p.z, ds)).into_iter().enumerate() {
            if k > 0 {
                ui.label(RichText::new("·").size(11.5).color(theme::INK_FAINT));
            }
            ui.label(RichText::new(t).size(11.5).color(colour));
        }
        if !ds.is_empty() {
            ui.label(RichText::new("Bohr").size(10.5).color(theme::INK_FAINT));
        }
    });
}

fn element_detail(ui: &mut Ui, map: &KnowledgeMap, e: &ElementKnowledge, input: &PanelInput<'_>) {
    let nuclei = &input.scene.nuclei;
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(sym(e.z)).strong().size(14.0));
        ui.label(RichText::new(format!("element · {} atoms in the fitted samples", e.atoms_seen)).monospace().size(11.0).color(theme::INK_DIM));
    });
    // the scene's atoms of this element: neighbour counts and first-shell angles
    let rc = map.r_cut;
    let mine: Vec<usize> = (0..nuclei.len()).filter(|&i| nuclei[i].z == e.z).collect();
    let coord: Vec<(usize, usize)> = mine.iter().map(|&i| (i, (0..nuclei.len()).filter(|&j| j != i && dist(nuclei[i].pos, nuclei[j].pos) < rc).count())).collect();
    let mut no_shell = false;
    let mut angles: Vec<f64> = Vec::new();
    for &i in &mine {
        let shell: Vec<[f64; 3]> = (0..nuclei.len())
            .filter(|&j| j != i)
            .filter_map(|j| {
                let d = dist(nuclei[i].pos, nuclei[j].pos);
                if d >= rc || d <= 0.0 {
                    return None;
                }
                let Some(rs) = map.pair(e.z, nuclei[j].z).and_then(|p| first_shell(&p.hist)) else {
                    no_shell = true;
                    return None;
                };
                (d <= rs).then(|| std::array::from_fn(|k| (nuclei[j].pos[k] - nuclei[i].pos[k]) / d))
            })
            .collect();
        for (a, u) in shell.iter().enumerate() {
            for v in &shell[a + 1..] {
                let c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).clamp(-1.0, 1.0);
                angles.push(c.acos().to_degrees());
            }
        }
    }
    theme::section(ui, &format!("Neighbours within {rc:.1} Bohr"));
    let top = e.coordination.iter().copied().max().unwrap_or(0).max(1) as f32;
    let asked_n = coord.iter().map(|c| c.1).max().unwrap_or(0);
    egui::Grid::new(ui.id().with(("knowledge-coord", e.z))).num_columns(3).spacing(vec2(8.0, 2.0)).show(ui, |ui| {
        for n in 0..e.coordination.len().max(asked_n + 1) {
            let count = e.coordination.get(n).copied().unwrap_or(0);
            let here: Vec<String> = coord.iter().filter(|c| c.1 == n).map(|c| format!("#{}", c.0)).collect();
            if count == 0 && here.is_empty() {
                continue;
            }
            ui.label(RichText::new(format!("{n} neighbour{}", if n == 1 { "" } else { "s" })).size(11.5));
            let (r, _) = ui.allocate_exact_size(vec2(120.0, 8.0), Sense::hover());
            ui.painter().rect_filled(r, 3.0, Color32::from_black_alpha(70));
            let fill = Rect::from_min_size(r.min, vec2(r.width() * count as f32 / top, r.height()));
            ui.painter().rect_filled(fill, 3.0, Taught::of(count).colour());
            ui.horizontal(|ui| {
                ui.label(RichText::new(count.to_string()).monospace().size(11.0));
                if !here.is_empty() {
                    let c = if count == 0 { theme::INK_DIM } else { Taught::of(count).colour() };
                    ui.label(RichText::new(format!("asked: {}", here.join(", "))).size(11.0).color(c));
                }
            });
            ui.end_row();
        }
    });
    theme::note(ui, "the model's neighbours (every atom within its cutoff), not bonds");
    theme::section(ui, &format!("Bond angles at {} · {ANGLE_BIN_DEG:.0}° bins", sym(e.z)));
    let colour_of = |k: usize| -> Option<Color32> { e.angles.get(k).filter(|&&n| n > 0).map(|&n| Taught::of(n).colour()) };
    let labels: Vec<(f64, String)> = (0..=6).map(|k| (k as f64 * 30.0, (k * 30).to_string())).collect();
    let w = ui.available_width().clamp(240.0, 640.0);
    let none = e.angles.iter().all(|&n| n == 0);
    histogram(
        ui,
        &e.angles,
        ANGLE_BIN_DEG,
        ANGLE_BINS as f64 * ANGLE_BIN_DEG,
        &colour_of,
        &angles,
        &labels,
        ("", "bond angle, degrees"),
        vec2(w, if w > 400.0 { 110.0 } else { 80.0 }),
        none.then_some("no angle taught"),
    );
    let mut note = "between two first-shell neighbours (each pair's first histogram peak, measured, never a table)".to_string();
    if none {
        note = format!("no {} atom of the fitted samples had two first-shell neighbours. {note}", sym(e.z));
    }
    if no_shell {
        note += ". Some of the scene's neighbours are of a pair never taught: it has no first shell yet, so they are not in the asked angles";
    }
    theme::note(ui, note);
    if !angles.is_empty() {
        let mut taught = 0;
        for &a in &angles {
            if e.angles.get(angle_bin(a)).is_some_and(|&n| n > 0) {
                taught += 1;
            }
        }
        theme::note(ui, format!("asked now: {} angle{} at {} atom{}, {taught} in a bin taught", angles.len(), if angles.len() == 1 { "" } else { "s" }, mine.len(), if mine.len() == 1 { "" } else { "s" }));
    }
}

/// A log-scale histogram: `counts` in bins of `bin` x-units from 0 to `x_max`, each bar coloured
/// by `colour_of` (`None`: empty, drawn as a grey baseline), white dashed "asked now" ticks at
/// `ticks`, axis labels and the two end captions.
#[allow(clippy::too_many_arguments)]
fn histogram(
    ui: &mut Ui,
    counts: &[u32],
    bin: f64,
    x_max: f64,
    colour_of: &dyn Fn(usize) -> Option<Color32>,
    ticks: &[f64],
    labels: &[(f64, String)],
    ends: (&str, &str),
    size: Vec2,
    empty: Option<&str>,
) {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter_at(rect.expand(2.0));
    let plot = Rect::from_min_max(rect.min + vec2(26.0, 16.0), rect.max - vec2(8.0, 30.0));
    p.rect_filled(plot, 0.0, Color32::from_black_alpha(64));
    let x_of = |x: f64| plot.left() + (x / x_max.max(1e-9)) as f32 * plot.width();
    let top = counts.iter().copied().max().unwrap_or(0) as f32;
    let small = FontId::monospace(8.5);
    for (k, &n) in counts.iter().enumerate() {
        let (x0, x1) = (x_of(k as f64 * bin) + 0.5, x_of((k + 1) as f64 * bin) - 0.5);
        if x0 >= plot.right() {
            break;
        }
        match colour_of(k) {
            Some(c) if n > 0 => {
                let h = (1.0 + n as f32).ln() / (1.0 + top).ln() * plot.height();
                p.rect_filled(Rect::from_min_max(pos2(x0, plot.bottom() - h), pos2(x1.min(plot.right()), plot.bottom())), 0.0, c.gamma_multiply(0.85));
            }
            _ => {
                p.line_segment([pos2(x0, plot.bottom()), pos2(x1.min(plot.right()), plot.bottom())], Stroke::new(2.0, theme::INK_FAINT));
            }
        }
    }
    for (x, t) in labels {
        let at = x_of(*x);
        p.line_segment([pos2(at, plot.bottom()), pos2(at, plot.bottom() + 3.0)], Stroke::new(1.0, theme::INK_FAINT));
        p.text(pos2(at, plot.bottom() + 4.0), Align2::CENTER_TOP, t, small.clone(), theme::INK_FAINT);
    }
    p.text(pos2(plot.left(), rect.bottom() - 2.0), Align2::LEFT_BOTTOM, ends.0, small.clone(), theme::INK_FAINT);
    p.text(pos2(plot.right(), rect.bottom() - 2.0), Align2::RIGHT_BOTTOM, ends.1, small.clone(), theme::INK_FAINT);
    p.text(pos2(plot.left() - 3.0, plot.top() + 2.0), Align2::RIGHT_TOP, "log", small.clone(), theme::INK_FAINT);
    if let Some(t) = empty {
        p.text(plot.center(), Align2::CENTER_CENTER, t, FontId::monospace(10.0), theme::INK_DIM);
    }
    // the asked distances: one tick per distinct position (pixels)
    let mut xs: Vec<f32> = ticks.iter().filter(|&&d| d <= x_max).map(|&d| x_of(d).round()).collect();
    xs.dedup();
    for (k, &x) in xs.iter().enumerate() {
        p.extend(Shape::dashed_line(&[pos2(x, plot.top() - 2.0), pos2(x, plot.bottom())], Stroke::new(1.3, Color32::from_white_alpha(215)), 2.0, 2.0));
        p.add(Shape::convex_polygon(vec![pos2(x - 4.0, plot.top() - 8.0), pos2(x + 4.0, plot.top() - 8.0), pos2(x, plot.top() - 2.0)], Color32::WHITE, Stroke::NONE));
        if k == 0 {
            p.text(pos2(x + 6.0, plot.top() - 9.0), Align2::LEFT_TOP, "asked now", FontId::monospace(9.0), Color32::WHITE);
        }
    }
}

/// The overlay's legend card (bottom left of the 3D view, above the view switch), while the
/// overlay is on.
pub fn legend(ctx: &egui::Context, view: Rect, st: &UiState, input: &PanelInput<'_>) {
    if !st.show_knowledge {
        return;
    }
    // above the view switch, its slider rows, and the orbital legend when that is up
    let lift = 70.0 + 40.0 * (u8::from(st.camera_view) + u8::from(st.volume_mode == super::VOLUME_DOTS)) as f32 + if st.show != super::ShowMode::Density { 64.0 } else { 0.0 };
    let map = map_of(st, input);
    egui::Area::new(Id::new("knowledge-legend"))
        .pivot(Align2::LEFT_BOTTOM)
        .fixed_pos(pos2(view.left() + 14.0, view.bottom() - lift))
        .order(Order::Middle)
        .fade_in(false)
        .show(ctx, |ui| {
            theme::card().show(ui, |ui| {
                ui.set_max_width(260.0);
                ui.label(RichText::new("knowledge overlay").strong().size(11.5));
                let (r, _) = ui.allocate_exact_size(vec2(232.0, 10.0), Sense::hover());
                let mut mesh = egui::Mesh::default();
                for k in 0..=16 {
                    let t = k as f32 / 16.0;
                    let [cr, cg, cb] = tint_srgb(t);
                    let c = Color32::from_rgb(cr, cg, cb);
                    let x = r.left() + t * r.width();
                    mesh.colored_vertex(pos2(x, r.top()), c);
                    mesh.colored_vertex(pos2(x, r.bottom()), c);
                    if k > 0 {
                        let i = 2 * k as u32;
                        mesh.add_triangle(i - 2, i - 1, i);
                        mesh.add_triangle(i - 1, i + 1, i);
                    }
                }
                ui.painter().add(Shape::mesh(mesh));
                // the ticks under their places on the bar: 0, ½, 1
                let (row, _) = ui.allocate_exact_size(vec2(r.width(), 14.0), Sense::hover());
                let font = FontId::proportional(10.5);
                for (x, align, text) in [(row.left(), Align2::LEFT_TOP, "0 never seen"), (row.center().x, Align2::CENTER_TOP, "0.5 thin"), (row.right(), Align2::RIGHT_TOP, "1 well known")] {
                    ui.painter().text(pos2(x, row.top()), align, text, font.clone(), theme::INK_DIM);
                }
                match map.as_deref() {
                    Some(m) => {
                        theme::note(ui, format!("per atom: its least-known neighbour within {:.1} Bohr — that pair's verdict and how many distances like it were taught", m.r_cut));
                    }
                    None => {
                        ui.add(egui::Label::new(RichText::new("no knowledge map yet: the estimated solver builds it when its model is fitted").size(11.0).color(AMBER)).wrap());
                    }
                }
            });
        });
}

#[cfg(test)]
pub(crate) fn fixture_map() -> KnowledgeMap {
    use sim::knowledge::{PairHistogram, BIN};
    let hist = |z: (u8, u8), peaks: &[(usize, u32)]| {
        let mut counts = vec![0u32; 60];
        for &(k, n) in peaks {
            counts[k] = n;
        }
        PairHistogram { z, bin: BIN, counts }
    };
    let pk = |z, samples, peaks: &[(usize, u32)], confidence| PairKnowledge { z, samples, hist: hist(z, peaks), confidence };
    let mut angles = vec![0u32; ANGLE_BINS];
    angles[20] = 40;
    angles[21] = 300;
    angles[22] = 60;
    let el = |z, atoms_seen, coordination: Vec<u32>, angles: Vec<u32>| ElementKnowledge { z, atoms_seen, coordination, angles };
    KnowledgeMap {
        elements: vec![el(1, 2140, vec![0, 1200, 900, 40], vec![0; ANGLE_BINS]), el(6, 610, vec![0, 0, 0, 33, 571, 6], angles.clone()), el(8, 880, vec![0, 34, 846], angles)],
        pairs: vec![
            pk((1, 1), 2140, &[(14, 600), (15, 400), (28, 30), (33, 900), (34, 200), (50, 60)], Confidence::Thin),
            pk((1, 6), 610, &[(20, 1200), (21, 900), (35, 4), (40, 3), (52, 200)], Confidence::Trusted),
            pk((1, 8), 880, &[(17, 800), (18, 700), (30, 90)], Confidence::Trusted),
            pk((6, 6), 0, &[], Confidence::Unseen),
            pk((6, 8), 38, &[(54, 20), (55, 30), (56, 17)], Confidence::Thin),
            pk((8, 8), 420, &[(53, 90), (54, 120), (55, 80)], Confidence::Trusted),
        ],
        slug: "lsda-h0.15-s0.15-tol1e-6-gpu-accepted-3f9a".into(),
        samples: 2140,
        r_cut: 6.0,
    }
}
