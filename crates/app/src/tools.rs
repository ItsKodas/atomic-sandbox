//! Mouse tools in world space: picking, placement on the focus plane, and plane-constrained drag.
//! Pixel coordinates are physical pixels, origin top-left (the same frame as `OrbitCamera::pixel_ray`).
use crate::scene::SceneNucleus;
use glam::Vec3;
use render::camera::{ray_plane, ray_sphere, OrbitCamera};
use render::elements;

#[derive(Default)]
pub struct ToolState {
    pub selected: Option<usize>,
    /// A selected nucleus is being dragged (left button held after picking it).
    pub dragging: bool,
    /// (point, unit normal) of the plane the dragged nucleus moves in.
    pub drag_plane: Option<(Vec3, Vec3)>,
}

pub fn v3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

fn a3(v: Vec3) -> [f64; 3] {
    [v.x as f64, v.y as f64, v.z as f64]
}

/// Nearest nucleus along the pixel ray (ray_sphere against display radii), ties broken by smallest t.
pub fn pick(
    cam: &OrbitCamera,
    px: f32,
    py: f32,
    w: f32,
    h: f32,
    nuclei: &[SceneNucleus],
) -> Option<usize> {
    let (o, d) = cam.pixel_ray(px, py, w, h);
    nuclei
        .iter()
        .enumerate()
        .filter_map(|(i, n)| ray_sphere(o, d, v3(n.pos), elements::radius(n.z)).map(|t| (t, i)))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, i)| i)
}

/// Placement point: intersection of the pixel ray with the camera's focus plane.
pub fn place_point(cam: &OrbitCamera, px: f32, py: f32, w: f32, h: f32) -> Option<[f64; 3]> {
    let (o, d) = cam.pixel_ray(px, py, w, h);
    let (p, n) = cam.focus_plane();
    ray_plane(o, d, p, n).map(a3)
}

/// Drag: intersection with the plane through the nucleus's original position, parallel to the focus plane.
pub fn drag_point(
    cam: &OrbitCamera,
    px: f32,
    py: f32,
    w: f32,
    h: f32,
    plane: (Vec3, Vec3),
) -> Option<[f64; 3]> {
    let (o, d) = cam.pixel_ray(px, py, w, h);
    ray_plane(o, d, plane.0, plane.1).map(a3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneNucleus;
    use glam::Vec3;
    use render::camera::OrbitCamera;

    fn two_overlapping() -> Vec<SceneNucleus> {
        // both project onto the screen centre; the nearer one (closer to the eye) must win
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let eye = cam.eye();
        let dir = (Vec3::ZERO - eye).normalize();
        let near = eye + dir * 6.0;
        let far = eye + dir * 12.0;
        vec![
            SceneNucleus {
                z: 1,
                pos: [far.x as f64, far.y as f64, far.z as f64],
                vel: [0.0; 3],
            },
            SceneNucleus {
                z: 1,
                pos: [near.x as f64, near.y as f64, near.z as f64],
                vel: [0.0; 3],
            },
        ]
    }

    #[test]
    fn pick_returns_the_nearest_of_two_overlapping_nuclei() {
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let nuclei = two_overlapping();
        assert_eq!(pick(&cam, 400.0, 300.0, 800.0, 600.0, &nuclei), Some(1));
    }

    #[test]
    fn pick_misses_empty_space() {
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let nuclei = vec![SceneNucleus {
            z: 1,
            pos: [0.0; 3],
            vel: [0.0; 3],
        }];
        assert_eq!(pick(&cam, 5.0, 5.0, 800.0, 600.0, &nuclei), None);
    }

    #[test]
    fn place_point_lies_on_the_focus_plane() {
        let cam = OrbitCamera::new(Vec3::new(1.0, 0.0, 0.0), 10.0);
        let p = place_point(&cam, 500.0, 200.0, 800.0, 600.0).unwrap();
        let (pt, n) = cam.focus_plane();
        let v = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) - pt;
        assert!(v.dot(n).abs() < 1e-4);
    }

    #[test]
    fn drag_keeps_the_nucleus_in_its_own_plane() {
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let (_, n) = cam.focus_plane();
        let plane = (Vec3::new(0.0, 0.0, 2.0), n);
        let p = drag_point(&cam, 450.0, 320.0, 800.0, 600.0, plane).unwrap();
        let v = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) - plane.0;
        assert!(v.dot(n).abs() < 1e-4);
    }
}
