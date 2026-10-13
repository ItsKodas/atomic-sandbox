//! Reference geometry for the 3D view: the simulation cell, a floor grid with axes (Blender-style,
//! the xz plane with +y up), and in Place mode the placement plane the next click lands on.
//! Pure: turns the display toggles into line vertices; the painter draws them depth-tested.
use crate::scene::Scene;
use crate::ui::{ToolMode, UiState};
use glam::Vec3;
use render::camera::OrbitCamera;
use render::lines::{self, LineVertex};

const CELL: [f32; 4] = [0.95, 0.80, 0.35, 0.55];
const FLOOR_MINOR: [f32; 4] = [1.0, 1.0, 1.0, 0.09];
const FLOOR_MAJOR: [f32; 4] = [1.0, 1.0, 1.0, 0.22];
const PLACE_MINOR: [f32; 4] = [0.45, 0.80, 1.0, 0.16];
const PLACE_MAJOR: [f32; 4] = [0.45, 0.80, 1.0, 0.40];
/// Major line every this many minor spacings.
const MAJOR_EVERY: i32 = 5;
/// Half-size of the placement-plane grid (Bohr): big enough to frame a molecule, small enough
/// not to read as a floor.
const PLACE_HALF_EXTENT: f32 = 6.0;

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
