//! Reference geometry for the 3D view: the simulation cell, a floor grid with axes (Blender-style,
//! the xz plane with +y up), in Place mode the placement plane the next click lands on, and the
//! bonds read off the density (M6 spec §3.4: [`bond_lines`]). Pure: turns the display toggles and
//! the bond graph into line vertices; the painter draws them depth-tested, so a bond shows between
//! the spheres' surfaces.
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

/// A bond's colour (warm white) and a multi-centre ring bond's (violet, dashed: the three-centre
/// bond of H₃⁺ is one bond shared by three nuclei, not three ordinary ones).
const BOND: [f32; 3] = [0.93, 0.91, 0.84];
const RING_BOND: [f32; 3] = [0.74, 0.62, 1.0];
/// Alpha of a bond from the current evaluation, and of one from an earlier DFT step drawn between
/// the current positions (a surrogate step has no electrons: the faded graph, spec §3.4).
pub const BOND_ALPHA: f32 = 0.9;
pub const STALE_BOND_ALPHA: f32 = 0.3;
/// A bond is drawn as a bundle of this many thin lines on a cylinder of this radius (Bohr): one
/// pixel-wide line reads as a hairline next to the spheres (display choice, not physics).
pub const BOND_STRANDS: usize = 6;
pub const BOND_RADIUS: f32 = 0.045;
/// A ring bond's dash and gap (Bohr).
const DASH: f32 = 0.22;
const GAP: f32 = 0.14;

/// The bonds of `graph` between the nuclei at `pos` (the positions drawn now — the topology of the
/// last DFT graph between the CURRENT nuclei, spec §3.4), faded when `stale`. Nothing when the
/// graph's nucleus count is not `pos.len()` (a graph with wrong indices is never drawn).
pub fn bond_lines(pos: &[[f64; 3]], graph: &sim::BondGraph, stale: bool) -> Vec<LineVertex> {
    let mut out = Vec::new();
    if graph.n_nuclei != pos.len() {
        return out;
    }
    let alpha = if stale { STALE_BOND_ALPHA } else { BOND_ALPHA };
    for b in &graph.bonds {
        let (Some(&a), Some(&c)) = (pos.get(b.i), pos.get(b.j)) else { continue };
        let (a, c) = (Vec3::from_array(a.map(|x| x as f32)), Vec3::from_array(c.map(|x| x as f32)));
        let axis = c - a;
        let len = axis.length();
        if len < 1e-6 {
            continue;
        }
        let rgb = if b.multi_centre { RING_BOND } else { BOND };
        let colour = [rgb[0], rgb[1], rgb[2], alpha];
        let (u, v) = plane_basis(axis / len);
        for k in 0..BOND_STRANDS {
            let t = k as f32 * std::f32::consts::TAU / BOND_STRANDS as f32;
            let off = BOND_RADIUS * (t.cos() * u + t.sin() * v);
            if b.multi_centre {
                let mut s = 0.0;
                while s < len {
                    let e = (s + DASH).min(len);
                    lines::segment(&mut out, (a + off + axis * (s / len)).to_array(), (a + off + axis * (e / len)).to_array(), colour);
                    s = e + GAP;
                }
            } else {
                lines::segment(&mut out, (a + off).to_array(), (c + off).to_array(), colour);
            }
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
        sim::BondGraph {
            bonds: bonds.iter().map(|&(i, j, m)| sim::Bond { i, j, rho_bcp: 0.2, enhancement: 1.3, s: 0.5, route: sim::Route::Saddle, multi_centre: m }).collect(),
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
        // a ring bond is dashed: more, shorter segments, all on the same cylinder
        let ring = bond_lines(&pos, &graph(2, &[(0, 1, true)]), false);
        assert!(ring.len() > 2 * BOND_STRANDS && ring.len() % (2 * BOND_STRANDS) == 0, "{}", ring.len());
        assert_ne!(ring[0].colour, bond_lines(&pos, &g, false)[0].colour, "ring bonds are told apart");
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
