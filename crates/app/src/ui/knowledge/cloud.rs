//! The three-atom map's 3D view (spec 2026-11-17 §6.5 "The 3D view"; Phase A Task 8): the same
//! cells as the 2D slice, every slice at once, as a rotatable voxel cloud. Rendering geometry, not
//! physics.
//!
//! - **Axes.** r₁, θ (the 10 cos θ bins, θ upward) and r₂, each normalised to [−½, ½] over the
//!   class cache's cropped extent ([`voxels`]); the angle axis always spans its 10 bins. At yaw 0°
//!   and pitch 0° the screen's x is r₁, its up is θ, and the viewer looks along +r₂ (of two cells
//!   differing only in r₂ the smaller r₂ is in front). Orthographic: yaw turns about the θ axis,
//!   then pitch about the screen's x (positive pitch looks down on the cloud).
//! - **Cells.** A filled cell (taught or thin) is a box at [`CUBE_FILL`] of its cell, its ≤ 3
//!   faces that face the viewer filled in its class's colour, shaded by [`SHADES`] (θ-face,
//!   r₁-face, r₂-face) and drawn back to front ([`project`]: a painter's sort on the rotated
//!   centre's depth, equal depths by ascending cell). A missing cell within reach is a small
//!   `INK_FAINT` outlined square at its centre (the "missing" checkbox hides them); a new cell
//!   (pending > 0) a dashed square; the open scene's arrangements of the type white crosses.
//! - **Input.** A drag rotates ([`rotate`], pitch held within ±[`PITCH_LIMIT_DEG`]), the wheel
//!   zooms (over the cloud only; elsewhere the panel scrolls), a double-click or Reset returns to
//!   the default angles ([`reset`]). A click picks the front-most cell whose projected outline
//!   holds the pointer ([`hit`], spec U3) — a cube or a square — and opens the same cell card as
//!   the 2D slice; empty space clears it. Hover reads the slice's words.
//! - **Cost** (spec §6.4 "UI cost"). The drawn cells and their voxels are found once per (map,
//!   type) and kept beside the classes ([`Cloud`], in `ClassCache::cloud`); the projection and the
//!   depth sort run again only when the angles, the zoom, the rect or the checkbox change.
//!
//! The camera (`TripleView::yaw_deg`, `pitch_deg`, `zoom`, `show_missing`) is display state, not
//! recorded; nothing here emits an `Action`. Distances in Bohr; degrees only in the labels.
use super::triples::{cell_card, cell_hover, drawn, nothing_drawn, scene_instances, ClassCache};
use super::{pair_name, sym, TripleView, NEW_INK, WHEEL_STEP};
use crate::ui::{keyed, probe, probe_rect, theme, PanelInput, AMBER, GREEN};
use egui::{pos2, vec2, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, Ui, Vec2};
use sim::knowledge::{CellClass, TripleGrid, TripleKnowledge, TripleMap};
use std::ops::Range;

/// The default rotation (degrees): a three-quarter view from a little above.
pub const DEFAULT_YAW_DEG: f32 = 35.0;
pub const DEFAULT_PITCH_DEG: f32 = 20.0;
/// The pitch is held within ± this (degrees), short of looking straight down an axis.
pub const PITCH_LIMIT_DEG: f32 = 89.0;
/// A cube is 0.8 of its cell.
pub const CUBE_FILL: f32 = 0.8;
/// The visible faces' shading: θ-face, r₁-face, r₂-face.
pub const SHADES: [f32; 3] = [1.0, 0.85, 0.7];
/// Degrees of rotation per point of drag.
pub const DEG_PER_PT: f32 = 0.5;
/// The zoom range (× the fitted cloud).
pub const ZOOM_MIN: f32 = 0.5;
pub const ZOOM_MAX: f32 = 6.0;
/// A missing or new cell's square: this fraction of its cell's smallest side on screen, at least
/// [`SQUARE_MIN_PX`] across.
pub const SQUARE_FILL: f32 = 0.4;
pub const SQUARE_MIN_PX: f32 = 4.0;
/// A face whose normal is this close to the screen's plane is edge-on and not drawn.
const EDGE_ON: f32 = 1e-4;

/// One cell placed in the cloud.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voxel {
    pub cell: usize,
    /// Its centre in [−½, ½]³ over the extent: (r₁, θ, r₂).
    pub at: [f32; 3],
    /// A filled cell's cube; `false`: a missing (or new-only) cell's square.
    pub cube: bool,
}

/// One cell as drawn: its depth (larger is farther), its visible faces with their shade (none for
/// a square), and its outline on screen (convex), which the hit test reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawn {
    pub cell: usize,
    pub depth: f32,
    pub faces: Vec<([Pos2; 4], f32)>,
    pub hull: Vec<Pos2>,
}

/// The voxels of `cells` (the views' drawn cells), each at its centre normalised over `extent`'s
/// (r₁ rows, r₂ slices) and the grid's cos bins; cells outside the extent are left out.
pub fn voxels(grid: &TripleGrid, cells: &[usize], extent: &(Range<usize>, Range<usize>), cube: impl Fn(usize) -> bool) -> Vec<Voxel> {
    let (rows, slices) = extent;
    let norm = |k: usize, r: &Range<usize>| ((k - r.start) as f32 + 0.5) / r.len() as f32 - 0.5;
    cells
        .iter()
        .filter_map(|&c| {
            let (k1, k2, q) = grid.coords(c);
            (rows.contains(&k1) && slices.contains(&k2)).then(|| Voxel { cell: c, at: [norm(k1, rows), ((q as f32 + 0.5) / grid.cos_bins as f32) - 0.5, norm(k2, slices)], cube: cube(c) })
        })
        .collect()
}

/// One cell's size per axis in the normalised cloud: (1/rows, 1/cos bins, 1/slices).
pub fn cell_size(grid: &TripleGrid, extent: &(Range<usize>, Range<usize>)) -> [f32; 3] {
    [1.0 / extent.0.len().max(1) as f32, 1.0 / grid.cos_bins.max(1) as f32, 1.0 / extent.1.len().max(1) as f32]
}

/// The rotation by yaw (about θ) then pitch (about the screen's x).
#[derive(Clone, Copy)]
struct Rot {
    cy: f32,
    sy: f32,
    cp: f32,
    sp: f32,
}

impl Rot {
    fn new(yaw_deg: f32, pitch_deg: f32) -> Rot {
        let (sy, cy) = yaw_deg.to_radians().sin_cos();
        let (sp, cp) = pitch_deg.to_radians().sin_cos();
        Rot { cy, sy, cp, sp }
    }

    /// (screen x, screen up, depth along the view).
    fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let (x1, y1, z1) = (p[0] * self.cy + p[2] * self.sy, p[1], -p[0] * self.sy + p[2] * self.cy);
        [x1, y1 * self.cp + z1 * self.sp, -y1 * self.sp + z1 * self.cp]
    }
}

/// Points per normalised unit: the cube's bounding sphere fits `rect` at zoom 1, at any angle.
fn scale(rect: Rect, zoom: f32) -> f32 {
    rect.width().min(rect.height()) / 3f32.sqrt() * zoom
}

/// A normalised point on screen.
pub fn to_screen(p: [f32; 3], yaw_deg: f32, pitch_deg: f32, zoom: f32, rect: Rect) -> Pos2 {
    let q = Rot::new(yaw_deg, pitch_deg).apply(p);
    let s = scale(rect, zoom);
    pos2(rect.center().x + q[0] * s, rect.center().y - q[1] * s)
}

/// The voxels' own cell size per axis: the smallest gap between two of their centres along it (a
/// single layer fills its axis). [`project`] uses it; the panel knows the extent and passes
/// [`cell_size`] to [`project_sized`].
fn lattice(voxels: &[Voxel]) -> [f32; 3] {
    std::array::from_fn(|a| {
        let mut xs: Vec<f32> = voxels.iter().map(|v| v.at[a]).collect();
        xs.sort_by(f32::total_cmp);
        xs.windows(2).map(|w| w[1] - w[0]).filter(|&d| d > 1e-6).fold(1.0f32, f32::min)
    })
}

/// The cloud drawn back to front (equal depth: ascending cell), each cube at [`CUBE_FILL`] of the
/// voxels' own spacing ([`project_sized`] with their lattice).
pub fn project(voxels: &[Voxel], yaw_deg: f32, pitch_deg: f32, zoom: f32, rect: Rect) -> Vec<Drawn> {
    project_sized(voxels, lattice(voxels), yaw_deg, pitch_deg, zoom, rect)
}

/// [`project`] with the cell's size per axis given (normalised units, [`cell_size`]).
pub fn project_sized(voxels: &[Voxel], size: [f32; 3], yaw_deg: f32, pitch_deg: f32, zoom: f32, rect: Rect) -> Vec<Drawn> {
    let rot = Rot::new(yaw_deg, pitch_deg);
    let s = scale(rect, zoom);
    let o = rect.center();
    let screen = |p: [f32; 3]| {
        let q = rot.apply(p);
        pos2(o.x + q[0] * s, o.y - q[1] * s)
    };
    // the faces that face the viewer (it looks along +depth): per axis in SHADES' order, the side
    // whose outward normal has a negative depth; an edge-on axis shows no face
    let faces: Vec<(usize, f32, f32)> = [1usize, 0, 2]
        .into_iter()
        .zip(SHADES)
        .filter_map(|(a, shade)| {
            let mut e = [0.0; 3];
            e[a] = 1.0;
            let nd = rot.apply(e)[2];
            (nd.abs() > EDGE_ON).then_some((a, if nd < 0.0 { 1.0 } else { -1.0 }, shade))
        })
        .collect();
    let half: [f32; 3] = std::array::from_fn(|a| 0.5 * CUBE_FILL * size[a]);
    let square = (0.5 * SQUARE_FILL * size.iter().copied().fold(f32::INFINITY, f32::min) * s).max(SQUARE_MIN_PX / 2.0);
    let mut out: Vec<Drawn> = voxels
        .iter()
        .map(|v| {
            // + 0.0: a −0.0 depth ties with 0.0
            let depth = rot.apply(v.at)[2] + 0.0;
            if !v.cube {
                let c = screen(v.at);
                let hull = vec![c + vec2(-square, -square), c + vec2(square, -square), c + vec2(square, square), c + vec2(-square, square)];
                return Drawn { cell: v.cell, depth, faces: Vec::new(), hull };
            }
            let corner = |sx: f32, sy: f32, sz: f32| screen([v.at[0] + sx * half[0], v.at[1] + sy * half[1], v.at[2] + sz * half[2]]);
            let quads = faces
                .iter()
                .map(|&(a, side, shade)| {
                    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                    let quad = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(u, w)| {
                        let mut sgn = [0.0f32; 3];
                        sgn[a] = side;
                        sgn[b] = u;
                        sgn[c] = w;
                        corner(sgn[0], sgn[1], sgn[2])
                    });
                    (quad, shade)
                })
                .collect();
            let mut corners = Vec::with_capacity(8);
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        corners.push(corner(sx, sy, sz));
                    }
                }
            }
            Drawn { cell: v.cell, depth, faces: quads, hull: hull(corners) }
        })
        .collect();
    out.sort_by(|a, b| b.depth.total_cmp(&a.depth).then(a.cell.cmp(&b.cell)));
    out
}

fn cross(o: Pos2, a: Pos2, b: Pos2) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// The convex hull of `pts` (monotone chain).
fn hull(mut pts: Vec<Pos2>) -> Vec<Pos2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut lower: Vec<Pos2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Pos2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// `p` inside the convex polygon `poly` or on its edge.
fn inside(poly: &[Pos2], p: Pos2) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let (mut pos, mut neg) = (false, false);
    for (i, &a) in poly.iter().enumerate() {
        let c = cross(a, poly[(i + 1) % poly.len()], p);
        pos |= c > 0.0;
        neg |= c < 0.0;
        if pos && neg {
            return false;
        }
    }
    true
}

/// The front-most cell whose outline holds `p` (`drawn` is back to front).
pub fn hit(drawn: &[Drawn], p: Pos2) -> Option<usize> {
    drawn.iter().rev().find(|d| inside(&d.hull, p)).map(|d| d.cell)
}

/// A drag of `drag` points: right turns the front of the cloud right (yaw), down tilts it down
/// (pitch, held within ±[`PITCH_LIMIT_DEG`]); yaw kept in [−180°, 180°).
pub fn rotate(view: &mut TripleView, drag: Vec2) {
    view.yaw_deg = (view.yaw_deg - drag.x * DEG_PER_PT + 180.0).rem_euclid(360.0) - 180.0;
    view.pitch_deg = (view.pitch_deg + drag.y * DEG_PER_PT).clamp(-PITCH_LIMIT_DEG, PITCH_LIMIT_DEG);
}

/// The default angles, zoom 1.
pub fn reset(view: &mut TripleView) {
    view.yaw_deg = DEFAULT_YAW_DEG;
    view.pitch_deg = DEFAULT_PITCH_DEG;
    view.zoom = 1.0;
}

/// What a projection was made for: it is made again only when one of these changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub zoom: f32,
    pub rect: Rect,
    pub show_missing: bool,
}

/// One type's cloud for one map (kept in `ClassCache::cloud`, so a new map or type starts over):
/// its voxels, found once, and the last projection.
#[derive(Clone, Debug, PartialEq)]
pub struct Cloud {
    /// Every drawn cell of the extent (the 2D views' rule, `triples::drawn`).
    pub voxels: Vec<Voxel>,
    /// [`cell_size`] of the extent.
    pub size: [f32; 3],
    pub key: Option<Key>,
    /// The last projection, back to front.
    pub drawn: Vec<Drawn>,
}

impl Cloud {
    /// The cloud of `tk` over `extent`: cubes for the filled cells, squares for the rest.
    pub fn of(grid: &TripleGrid, tk: &TripleKnowledge, classes: &[CellClass], extent: &(Range<usize>, Range<usize>)) -> Cloud {
        let cells: Vec<usize> = (0..grid.cells().min(classes.len())).filter(|&c| drawn(tk, classes[c], c)).collect();
        let voxels = voxels(grid, &cells, extent, |c| matches!(classes[c], CellClass::Taught | CellClass::Thin));
        Cloud { voxels, size: cell_size(grid, extent), key: None, drawn: Vec::new() }
    }

    /// Projects the voxels `keep` admits for `key`, unless the last projection was for it.
    pub fn update(&mut self, key: Key, keep: impl Fn(usize) -> bool) {
        if self.key == Some(key) {
            return;
        }
        let v: Vec<Voxel> = self.voxels.iter().filter(|v| keep(v.cell)).copied().collect();
        self.drawn = project_sized(&v, self.size, key.yaw_deg, key.pitch_deg, key.zoom, key.rect);
        self.key = Some(key);
    }
}

/// A premultiplied colour darkened by `shade`, its alpha kept.
fn shaded(c: Color32, shade: f32) -> Color32 {
    let [r, g, b, a] = c.to_array();
    let f = |x: u8| (x as f32 * shade).round() as u8;
    Color32::from_rgba_premultiplied(f(r), f(g), f(b), a)
}

/// The wheel's lines this frame (a pixel wheel's 50 points count as one), taken from the panel's
/// scroll.
fn wheel_lines(ui: &Ui) -> f32 {
    ui.input_mut(|i| {
        let lines = i
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
        lines
    })
}

/// The 3D view of the shown type (module doc): its controls, the cloud with its input, the caption
/// and the picked cell's card.
pub fn show(ui: &mut Ui, view: &mut TripleView, cache: &mut ClassCache, input: &PanelInput<'_>, map: &TripleMap, tk: &TripleKnowledge) {
    let t = tk.t;
    let Some(extent) = cache.extent.clone() else {
        nothing_drawn(ui, t);
        return;
    };
    let (g, folded) = (&map.grid, t.folded());
    // the controls: the "missing" checkbox and Reset
    ui.horizontal_wrapped(|ui| {
        let mut on = view.show_missing;
        let r = keyed(ui, "knowledge-triple-missing", |ui| ui.checkbox(&mut on, RichText::new("missing").size(11.0)))
            .on_hover_text("the empty cells within reach: small grey squares");
        probe("knowledge-triple-missing", &r);
        if r.changed() {
            view.show_missing = on;
        }
        let r = keyed(ui, "knowledge-triple-reset", |ui| ui.small_button("Reset")).on_hover_text("the default angle and zoom (also: double-click the cloud)");
        probe("knowledge-triple-reset", &r);
        if r.clicked() {
            reset(view);
        }
        ui.label(RichText::new("drag rotates · wheel zooms").size(10.0).color(theme::INK_FAINT));
    });
    // the cloud and its input
    let w = ui.available_width();
    let size = vec2(w, (w * 0.8).clamp(160.0, 320.0));
    let resp = keyed(ui, "knowledge-triple-cloud", |ui| ui.allocate_exact_size(size, Sense::click_and_drag()).1);
    probe("knowledge-triple-cloud", &resp);
    let rect = resp.rect;
    if ui.rect_contains_pointer(rect) {
        let lines = wheel_lines(ui);
        if lines != 0.0 {
            view.zoom = (view.zoom * WHEEL_STEP.powf(lines)).clamp(ZOOM_MIN, ZOOM_MAX);
        }
    }
    if resp.dragged() {
        rotate(view, resp.drag_delta());
    }
    if resp.double_clicked() {
        reset(view);
    }
    let classes = &cache.classes;
    let pending = |c: usize| tk.pending.get(c).copied().unwrap_or(0);
    let missing = |c: usize| classes[c] == CellClass::Missing && pending(c) == 0;
    let cloud = cache.cloud.get_or_insert_with(|| Cloud::of(g, tk, classes, &extent));
    let key = Key { yaw_deg: view.yaw_deg, pitch_deg: view.pitch_deg, zoom: view.zoom, rect, show_missing: view.show_missing };
    cloud.update(key, |c| view.show_missing || !missing(c));
    let drawn = &cloud.drawn;
    // the box and its axes, behind the cells
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, Color32::from_black_alpha(64));
    let at = |q: [f32; 3]| to_screen(q, view.yaw_deg, view.pitch_deg, view.zoom, rect);
    let origin = [-0.5, -0.5, -0.5];
    let small = FontId::monospace(9.0);
    for (a, label) in ["r₁", "θ", "r₂"].into_iter().enumerate() {
        let mut end = origin;
        end[a] = 0.55;
        let (from, to) = (at(origin), at(end));
        p.line_segment([from, to], Stroke::new(1.0, theme::INK_FAINT));
        let dir = (to - from).normalized();
        p.text(to + dir * 7.0, Align2::CENTER_CENTER, label, small.clone(), theme::INK_DIM);
    }
    // the cells, back to front
    let top = cache.top;
    let opacity = |n: u32| 0.35 + 0.65 * ((1.0 + n as f32).ln() / (1.0 + top as f32).ln().max(1e-6)).min(1.0);
    let edge = Stroke::new(0.5, Color32::from_black_alpha(110));
    let centre = |d: &Drawn| Rect::from_points(&d.hull).center();
    let square = |d: &Drawn| -> f32 {
        let h = Rect::from_points(&d.hull);
        if d.faces.is_empty() { h.width() / 2.0 } else { (0.5 * SQUARE_FILL * h.width().min(h.height())).max(SQUARE_MIN_PX / 2.0) }
    };
    for d in drawn {
        let c = d.cell;
        if d.faces.is_empty() {
            if classes[c] == CellClass::Missing {
                p.add(Shape::closed_line(d.hull.clone(), Stroke::new(1.0, theme::INK_FAINT)));
            }
        } else {
            let base = match classes[c] {
                CellClass::Taught => GREEN,
                CellClass::Thin => AMBER,
                _ => theme::INK_DIM,
            }
            .gamma_multiply(opacity(tk.counts.get(c).copied().unwrap_or(0)));
            for (quad, shade) in &d.faces {
                p.add(Shape::convex_polygon(quad.to_vec(), shaded(base, *shade), edge));
            }
        }
        if pending(c) > 0 {
            let r = Rect::from_center_size(centre(d), Vec2::splat(2.0 * square(d)));
            p.extend(Shape::dashed_line(&[r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()], Stroke::new(1.0, NEW_INK), 2.0, 2.0));
        }
    }
    // the open scene's arrangements: white crosses (those inside the extent)
    let (rows, slices) = &extent;
    let instances = scene_instances(&input.scene.nuclei, t, map.r_cut);
    let norm = |r: f64, span: &Range<usize>| ((r / g.r_bin) as f32 - span.start as f32) / span.len() as f32 - 0.5;
    let mut first = true;
    for inst in &instances {
        let q = [norm(inst.r1, rows), (-inst.u / 2.0) as f32, norm(inst.r2, slices)];
        if !(q[0].abs() <= 0.5 && q[2].abs() <= 0.5) {
            continue;
        }
        let x = at(q);
        let stroke = Stroke::new(1.5, Color32::WHITE);
        p.line_segment([x + vec2(-4.0, -4.0), x + vec2(4.0, 4.0)], stroke);
        p.line_segment([x + vec2(-4.0, 4.0), x + vec2(4.0, -4.0)], stroke);
        if first {
            probe_rect("knowledge-triple-cross", Rect::from_center_size(x, vec2(8.0, 8.0)));
            first = false;
        }
    }
    // the picked cell and the hovered one, outlined over the rest
    if let Some(d) = view.cell.and_then(|(k1, k2, q)| drawn.iter().find(|d| d.cell == g.index(folded, k1, k2, q))) {
        p.add(Shape::closed_line(d.hull.clone(), Stroke::new(2.0, theme::ACCENT)));
    }
    let hovered = resp.hover_pos().and_then(|at| hit(drawn, at));
    let resp = match hovered.and_then(|c| drawn.iter().find(|d| d.cell == c)) {
        Some(d) => {
            p.add(Shape::closed_line(d.hull.clone(), Stroke::new(1.0, Color32::from_white_alpha(200))));
            let mut text = cell_hover(map, tk, d.cell);
            let mine = instances.iter().filter(|i| g.cell(folded, i.r1, i.r2, i.u) == Some(d.cell)).count();
            if mine > 0 {
                let what = if mine == 1 { "arrangement".to_string() } else { format!("{mine} arrangements") };
                text.push_str(&format!("\ncross: the open scene's {what} here"));
            }
            resp.on_hover_ui_at_pointer(|ui| {
                ui.label(RichText::new(text).size(11.0));
            })
        }
        None => resp,
    };
    if resp.clicked() {
        view.cell = resp.interact_pointer_pos().and_then(|at| hit(drawn, at)).map(|c| g.coords(c));
    }
    let ((a1, b1), (a2, b2)) = ((g.r_edges(rows.start).0, g.r_edges(rows.end - 1).1), (g.r_edges(slices.start).0, g.r_edges(slices.end - 1).1));
    let caption = format!(
        "r₁ ({}) {a1:.1}–{b1:.1} · r₂ ({}) {a2:.1}–{b2:.1} Bohr · θ 0–180°, the angle at {} (cos θ bins)",
        pair_name(t.centre, t.a),
        pair_name(t.centre, t.b),
        sym(t.centre)
    );
    ui.add(egui::Label::new(RichText::new(caption).size(10.0).color(theme::INK_FAINT)).wrap());
    cell_card(ui, view, map, tk);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::knowledge::TripleView;
    use egui::{pos2, vec2, Rect};

    /// Spec U3.
    #[test]
    fn the_cloud_hit_test_picks_the_front_cube() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 400.0));
        let v = [Voxel { cell: 7, at: [0.0, 0.0, -0.3], cube: true }, Voxel { cell: 3, at: [0.0, 0.0, 0.3], cube: true }];
        let d = project(&v, 0.0, 0.0, 1.0, rect);
        assert_eq!(d.iter().map(|x| x.cell).collect::<Vec<_>>(), vec![3, 7], "back to front");
        assert_eq!(hit(&d, rect.center()), Some(7), "the front-most");
        assert_eq!(hit(&project(&v, 180.0, 0.0, 1.0, rect), rect.center()), Some(3));
        assert_eq!(hit(&d, pos2(1.0, 1.0)), None);
        let side = [Voxel { cell: 9, at: [-0.3, 0.0, 0.0], cube: true }, Voxel { cell: 2, at: [0.3, 0.0, 0.0], cube: true }];
        assert_eq!(project(&side, 0.0, 0.0, 1.0, rect).iter().map(|x| x.cell).collect::<Vec<_>>(), vec![2, 9], "equal depth: by cell");
        let a = project(&v, 35.0, 20.0, 1.0, rect);
        assert!(a.iter().all(|x| x.faces.len() <= 3));
        assert_eq!(a.iter().map(|x| x.cell).collect::<Vec<_>>(), project(&v, 35.0, 20.0, 1.0, rect).iter().map(|x| x.cell).collect::<Vec<_>>(), "stable");
        let mut view = TripleView::default();
        rotate(&mut view, vec2(0.0, -10_000.0));
        assert_eq!(view.pitch_deg.abs(), PITCH_LIMIT_DEG);
        reset(&mut view);
        assert_eq!((view.yaw_deg, view.pitch_deg, view.zoom), (DEFAULT_YAW_DEG, DEFAULT_PITCH_DEG, 1.0));
    }
}
