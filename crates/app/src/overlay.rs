//! Reference geometry for the 3D view: the simulation cell, a floor grid with axes (Blender-style,
//! the xz plane with +y up), in Place mode the placement plane the next click lands on, and the
//! bonds read off the density (M6 spec §3.4: [`bond_lines`]). Pure: turns the display toggles and
//! the bond graph into line vertices; the painter draws them depth-tested, so a bond shows between
//! the spheres' surfaces. A bond's line encodes its strength (user request, M6): width and opacity
//! follow log ρ_b ([`strength_fraction`]), the pattern its QTAIM kind (`sim::BondKind`: solid
//! shared, dashed closed-shell, dotted indeterminate), the colour a multi-centre ring.
use crate::scene::Scene;
use crate::ui::{ToolMode, UiState};
use glam::Vec3;
use render::camera::OrbitCamera;
use render::lines::{self, LineVertex};

const CELL: [f32; 4] = [0.95, 0.80, 0.35, 0.55];
/// The container walls' face: faint, cooler than the cell.
const WALLS: [f32; 4] = [0.55, 0.85, 1.0, 0.30];
const FLOOR_MINOR: [f32; 4] = [1.0, 1.0, 1.0, 0.09];
const FLOOR_MAJOR: [f32; 4] = [1.0, 1.0, 1.0, 0.22];
const PLACE_MINOR: [f32; 4] = [0.45, 0.80, 1.0, 0.16];
const PLACE_MAJOR: [f32; 4] = [0.45, 0.80, 1.0, 0.40];
/// Major line every this many minor spacings.
const MAJOR_EVERY: i32 = 5;
/// Half-size of the placement-plane grid (Bohr): big enough to frame a molecule, small enough
/// not to read as a floor.
const PLACE_HALF_EXTENT: f32 = 6.0;

/// A bond's colour (warm white) and a multi-centre ring bond's (violet: the three-centre bond of
/// H₃⁺ is one bond shared by three nuclei, not three ordinary ones).
const BOND: [f32; 3] = [0.93, 0.91, 0.84];
const RING_BOND: [f32; 3] = [0.74, 0.62, 1.0];
/// Alpha of a bond from the current evaluation, and of one from an earlier DFT step drawn between
/// the current positions (a surrogate step has no electrons: the faded graph, spec §3.4).
pub const BOND_ALPHA: f32 = 0.9;
pub const STALE_BOND_ALPHA: f32 = 0.3;
/// A bond is drawn as a bundle of this many thin lines on a cylinder of this radius (Bohr) at full
/// strength: one pixel-wide line reads as a hairline next to the spheres (display choice, not
/// physics).
pub const BOND_STRANDS: usize = 6;
pub const BOND_RADIUS: f32 = 0.045;
/// The display scale of bond strength — **a display scale, not physics**: log ρ_b mapped from the
/// detector's floor `sim::bonds::RHO_FLOOR` (0.01 e/Bohr³, nothing weaker is a bond) to
/// `RHO_STRONG` = 0.3 e/Bohr³ (the upper end of covalent ρ_b at a bond point — H₂'s is ≈ 0.27
/// near the exact density; this solver's at the preset 1.45 Bohr is 0.232), clamped.
pub const RHO_STRONG: f64 = 0.3;
/// At the floor a bond is drawn at these fractions of the full radius and alpha.
pub const MIN_WIDTH: f32 = 0.3;
pub const MIN_ALPHA: f32 = 0.35;
/// A closed-shell contact's dash and gap (Bohr), and an indeterminate one's dot and gap.
const DASH: f32 = 0.22;
const GAP: f32 = 0.14;
const DOT: f32 = 0.05;
const DOT_GAP: f32 = 0.12;

/// Where ρ_b sits on the display scale: ln(ρ_b/ρ_floor)/ln(ρ_strong/ρ_floor), clamped to [0, 1]
/// (0 for a non-finite ρ_b).
pub fn strength_fraction(rho_b: f64) -> f32 {
    let f = (rho_b / sim::bonds::RHO_FLOOR).ln() / (RHO_STRONG / sim::bonds::RHO_FLOOR).ln();
    if f.is_finite() { f.clamp(0.0, 1.0) as f32 } else { 0.0 }
}

/// The dash pattern of a kind (Bohr; `None` = solid).
fn pattern(kind: sim::BondKind) -> Option<(f32, f32)> {
    match kind {
        sim::BondKind::Shared => None,
        sim::BondKind::ClosedShell => Some((DASH, GAP)),
        sim::BondKind::Indeterminate => Some((DOT, DOT_GAP)),
    }
}

/// The bonds of `graph` between the nuclei at `pos` (the positions drawn now — the topology of the
/// last DFT graph between the CURRENT nuclei, spec §3.4), faded when `stale`. Nothing when the
/// graph's nucleus count is not `pos.len()` (a graph with wrong indices is never drawn).
pub fn bond_lines(pos: &[[f64; 3]], graph: &sim::BondGraph, stale: bool) -> Vec<LineVertex> {
    let mut out = Vec::new();
    if graph.n_nuclei != pos.len() {
        return out;
    }
    let base = if stale { STALE_BOND_ALPHA } else { BOND_ALPHA };
    for b in &graph.bonds {
        let (Some(&a), Some(&c)) = (pos.get(b.i), pos.get(b.j)) else { continue };
        let (a, c) = (Vec3::from_array(a.map(|x| x as f32)), Vec3::from_array(c.map(|x| x as f32)));
        let axis = c - a;
        let len = axis.length();
        if len < 1e-6 {
            continue;
        }
        let rgb = if b.multi_centre { RING_BOND } else { BOND };
        let f = strength_fraction(b.rho_bcp);
        let colour = [rgb[0], rgb[1], rgb[2], base * (MIN_ALPHA + (1.0 - MIN_ALPHA) * f)];
        let radius = BOND_RADIUS * (MIN_WIDTH + (1.0 - MIN_WIDTH) * f);
        let (u, v) = plane_basis(axis / len);
        for k in 0..BOND_STRANDS {
            let t = k as f32 * std::f32::consts::TAU / BOND_STRANDS as f32;
            let off = radius * (t.cos() * u + t.sin() * v);
            match pattern(b.kind()) {
                Some((on, gap)) => {
                    let mut s = 0.0;
                    while s < len {
                        let e = (s + on).min(len);
                        lines::segment(&mut out, (a + off + axis * (s / len)).to_array(), (a + off + axis * (e / len)).to_array(), colour);
                        s = e + gap;
                    }
                }
                None => lines::segment(&mut out, (a + off).to_array(), (c + off).to_array(), colour),
            }
        }
    }
    out
}

/// The tint of a core-overlap pair (M7, spec §5.6): red, over the bond lines.
pub const CORE_OVERLAP: [f32; 4] = [1.0, 0.35, 0.3, 0.8];

/// Each pair (I, J, R) of `pairs` whose frozen cores overlap (`EvalStats::core_overlaps`) as a red strand bundle
/// between the nuclei at `pos` (the positions drawn now): the frozen-core approximation fails there, the step is
/// still taken, and the answer is never a training sample. Nothing for an index outside `pos`.
pub fn core_overlap_lines(pos: &[[f64; 3]], pairs: &[(usize, usize, f64)]) -> Vec<LineVertex> {
    let mut out = Vec::new();
    for &(i, j, _) in pairs {
        let (Some(&a), Some(&c)) = (pos.get(i), pos.get(j)) else { continue };
        let (a, c) = (Vec3::from_array(a.map(|x| x as f32)), Vec3::from_array(c.map(|x| x as f32)));
        let axis = c - a;
        let len = axis.length();
        if len < 1e-6 {
            continue;
        }
        let (u, v) = plane_basis(axis / len);
        for k in 0..BOND_STRANDS {
            let t = k as f32 * std::f32::consts::TAU / BOND_STRANDS as f32;
            let off = 1.6 * BOND_RADIUS * (t.cos() * u + t.sin() * v);
            lines::segment(&mut out, (a + off).to_array(), (c + off).to_array(), CORE_OVERLAP);
        }
    }
    out
}

/// Grid spacing in Bohr for the panel's unit choice.
pub fn spacing(in_angstrom: bool) -> f32 {
    if in_angstrom { units::angstrom_to_bohr(1.0) as f32 } else { 1.0 }
}

/// In-plane orthonormal basis (u, v) of the plane with unit normal `n`: u horizontal where possible.
fn plane_basis(n: Vec3) -> (Vec3, Vec3) {
    let mut u = Vec3::Y.cross(n);
    if u.length_squared() < 1e-6 { u = Vec3::X.cross(n); }
    let u = u.normalize();
    (u, n.cross(u).normalize())
}

pub fn build(ui: &UiState, scene: &Scene, camera: &OrbitCamera, mode: ToolMode) -> Vec<LineVertex> {
    let mut out = Vec::new();
    // the cell is centred on the origin: samples at origin + i·h, origin = −(n−1)h/2 (qm::Grid::cube)
    let half = (scene.grid_n as f32 - 1.0) * scene.grid_h as f32 / 2.0;
    let s = spacing(ui.grid_in_angstrom);
    if ui.show_cell {
        lines::wire_box(&mut out, [-half; 3], [half; 3], CELL);
    }
    if let Some(w) = ui.walls() {
        let l = w.half_width as f32;
        lines::wire_box(&mut out, [-l; 3], [l; 3], WALLS);
    }
    if ui.show_floor {
        // cover the cell footprint in whole spacings
        let ext = (half / s).ceil() * s;
        lines::grid(&mut out, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], ext, s, MAJOR_EVERY, FLOOR_MINOR, FLOOR_MAJOR);
    }
    if ui.show_axes {
        lines::axes(&mut out, [0.0; 3], (2.0 * s).min(half));
    }
    if ui.show_place_grid && mode == ToolMode::Place {
        let (p, n) = camera.focus_plane();
        let (u, v) = plane_basis(n);
        lines::grid(&mut out, p.to_array(), u.to_array(), v.to_array(), PLACE_HALF_EXTENT, s, MAJOR_EVERY, PLACE_MINOR, PLACE_MAJOR);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ui(scene: &Scene) -> UiState {
        let mut ui = UiState::new("h2", scene);
        ui.show_cell = false;
        ui.show_floor = false;
        ui.show_axes = false;
        ui.show_place_grid = false;
        // the container walls are on by default in the app (they draw an inner box): off here,
        // explicitly — the tests below switch on what each one draws
        ui.walls_on = false;
        ui
    }

    #[test]
    fn everything_off_draws_nothing() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        assert!(build(&ui(&scene), &scene, &cam, ToolMode::Orbit).is_empty());
    }

    #[test]
    fn cell_box_matches_the_grid_extent() {
        let scene = Scene::preset("h2").unwrap(); // 49 × 0.25 → half = 6.0
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let mut u = ui(&scene);
        u.show_cell = true;
        let v = build(&u, &scene, &cam, ToolMode::Orbit);
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|p| p.pos.iter().all(|c| c.abs() == 6.0)));
    }

    #[test]
    fn container_walls_draw_an_inner_box() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let mut u = ui(&scene);
        u.walls_on = true;
        u.wall_half_width = 4.0;
        let v = build(&u, &scene, &cam, ToolMode::Orbit);
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|p| p.pos.iter().all(|c| c.abs() == 4.0) && p.colour == WALLS));
    }

    #[test]
    fn placement_grid_only_in_place_mode_and_lies_in_the_focus_plane() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::new(0.3, -0.2, 0.1), 10.0);
        let mut u = ui(&scene);
        u.show_place_grid = true;
        assert!(build(&u, &scene, &cam, ToolMode::Select).is_empty());
        let v = build(&u, &scene, &cam, ToolMode::Place);
        assert!(!v.is_empty());
        let (p, n) = cam.focus_plane();
        for q in &v {
            let d = (Vec3::from_array(q.pos) - p).dot(n);
            assert!(d.abs() < 1e-4, "vertex off the focus plane by {d}");
        }
    }

    fn graph(n: usize, bonds: &[(usize, usize, bool)]) -> sim::BondGraph {
        graph_with(n, bonds, RHO_STRONG, -1.0)
    }

    /// Bonds of density `rho` at the bond point and Laplacian `lap` (band 0.01).
    fn graph_with(n: usize, bonds: &[(usize, usize, bool)], rho: f64, lap: f64) -> sim::BondGraph {
        sim::BondGraph {
            bonds: bonds.iter().map(|&(i, j, m)| sim::Bond { i, j, rho_bcp: rho, enhancement: 1.3, s: 0.5, route: sim::Route::Saddle, multi_centre: m, laplacian_bcp: lap, laplacian_noise: 0.01 }).collect(),
            components: vec![(0..n).collect()],
            step: 0,
            sim_time: 0.0,
            n_nuclei: n,
        }
    }

    /// Spec §3.4 / plan Task 10 step 1: a bond is drawn between the CURRENT positions (a strand
    /// bundle of radius `BOND_RADIUS` around the segment between them), at alpha 0.3 when the graph
    /// is stale and full alpha when not; a graph whose nucleus count is not the scene's draws
    /// nothing.
    #[test]
    fn bond_lines_follow_current_positions_with_the_stale_fade() {
        let g = graph(2, &[(0, 1, false)]);
        // positions that moved since the graph's evaluation: the lines follow them
        let pos = [[-1.0, 0.2, 0.0], [1.5, 0.2, 0.3]];
        for (stale, alpha) in [(true, STALE_BOND_ALPHA), (false, BOND_ALPHA)] {
            let v = bond_lines(&pos, &g, stale);
            assert_eq!(v.len(), 2 * BOND_STRANDS, "one bond, {BOND_STRANDS} strands");
            assert!(v.iter().all(|p| p.colour[3] == alpha));
            for pair in v.chunks(2) {
                for (p, end) in pair.iter().zip(pos) {
                    let d = (Vec3::from_array(p.pos) - Vec3::from_array(end.map(|x| x as f32))).length();
                    assert!((d - BOND_RADIUS).abs() < 1e-5, "a strand ends {d} from its nucleus");
                }
            }
        }
        assert_eq!(STALE_BOND_ALPHA, 0.3);
        // a nucleus count change: nothing (never a graph with wrong indices)
        assert!(bond_lines(&pos[..1], &g, false).is_empty());
        assert!(bond_lines(&[[0.0; 3]; 3], &g, false).is_empty());
        // a ring bond is told apart by its colour (its pattern is that of its kind)
        let ring = bond_lines(&pos, &graph(2, &[(0, 1, true)]), false);
        assert_eq!(ring.len(), 2 * BOND_STRANDS, "a shared ring bond is solid");
        assert_ne!(ring[0].colour, bond_lines(&pos, &g, false)[0].colour, "ring bonds are told apart");
    }

    /// The strength display (user request, M6): width and alpha follow log ρ_b between the floor
    /// and `RHO_STRONG`; shared is solid, closed-shell dashed, indeterminate dotted (more, shorter
    /// segments than the dashes).
    #[test]
    fn bond_lines_encode_strength_and_kind() {
        assert_eq!(strength_fraction(sim::bonds::RHO_FLOOR), 0.0);
        assert_eq!(strength_fraction(RHO_STRONG), 1.0);
        assert_eq!(strength_fraction(1e-4), 0.0);
        assert_eq!(strength_fraction(0.9), 1.0);
        assert_eq!(strength_fraction(f64::NAN), 0.0);
        assert!((strength_fraction((sim::bonds::RHO_FLOOR * RHO_STRONG).sqrt()) - 0.5).abs() < 1e-6, "log scale: the geometric mean is halfway");
        let pos = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let radius = |v: &[LineVertex]| (Vec3::from_array(v[0].pos) - Vec3::from_array([-1.5f32, 0.0, 0.0])).length();
        let strong = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.27, -1.0), false);
        let weak = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.02, -1.0), false);
        assert_eq!((strong.len(), weak.len()), (2 * BOND_STRANDS, 2 * BOND_STRANDS), "shared: solid");
        assert!(radius(&weak) < radius(&strong) && weak[0].colour[3] < strong[0].colour[3], "weaker is thinner and fainter");
        assert!((radius(&weak) - BOND_RADIUS * (MIN_WIDTH + (1.0 - MIN_WIDTH) * strength_fraction(0.02))).abs() < 1e-5);
        let dashed = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.02, 0.5), false);
        let dotted = bond_lines(&pos, &graph_with(2, &[(0, 1, false)], 0.02, 0.005), false);
        assert!(dashed.len() > 2 * BOND_STRANDS && dashed.len() % (2 * BOND_STRANDS) == 0, "closed-shell: dashed ({})", dashed.len());
        assert!(dotted.len() > dashed.len() && dotted.len() % (2 * BOND_STRANDS) == 0, "indeterminate: dotted ({} vs {})", dotted.len(), dashed.len());
        let seg = |v: &[LineVertex]| (Vec3::from_array(v[1].pos) - Vec3::from_array(v[0].pos)).length();
        assert!((seg(&dashed) - DASH).abs() < 1e-5 && (seg(&dotted) - DOT).abs() < 1e-5);
    }

    /// M7 Task 9 (spec §5.6): an overlapping pair is a red strand bundle between the current positions; an
    /// index outside the scene draws nothing.
    #[test]
    fn core_overlap_pairs_are_red_segments_between_the_current_positions() {
        let pos = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let v = core_overlap_lines(&pos, &[(0, 1, 1.0)]);
        assert_eq!(v.len(), 2 * BOND_STRANDS);
        assert!(v.iter().all(|p| p.colour == CORE_OVERLAP));
        assert!(core_overlap_lines(&pos, &[(0, 7, 1.0)]).is_empty());
        assert!(core_overlap_lines(&pos, &[]).is_empty());
    }

    #[test]
    fn angstrom_spacing_changes_the_floor_line_count() {
        let scene = Scene::preset("h2").unwrap();
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let mut u = ui(&scene);
        u.show_floor = true;
        let bohr = build(&u, &scene, &cam, ToolMode::Orbit).len();
        u.grid_in_angstrom = true;
        let ang = build(&u, &scene, &cam, ToolMode::Orbit).len();
        assert!(ang < bohr, "1 Å = 1.89 Bohr → fewer lines ({ang} vs {bohr})");
    }
}
