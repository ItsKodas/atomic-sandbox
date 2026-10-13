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
    /// Last cursor position over the 3D view (physical pixels); `None` while it is over the panel.
    pub cursor: Option<(f32, f32)>,
    /// Size of the 3D view in physical pixels, as of the last cursor event.
    pub view_px: (f32, f32),
}

/// Nuclei within this distance of the guide origin get a distance line (Bohr).
pub const GUIDE_RADIUS: f64 = 6.0;
/// At most this many lines, nearest first, so a crowded scene stays readable.
pub const GUIDE_MAX: usize = 8;

/// One distance line: the nucleus index it points at and the centre-to-centre distance (Bohr).
#[derive(Clone, Debug, PartialEq)]
pub struct Guide {
    pub index: usize,
    pub distance: f64,
}

/// Distance guides from `origin` to the nuclei within [`GUIDE_RADIUS`], nearest first, at most
/// [`GUIDE_MAX`]; `exclude` is the nucleus being dragged (its distance to itself is not a guide).
pub fn distance_guides(origin: [f64; 3], nuclei: &[SceneNucleus], exclude: Option<usize>) -> Vec<Guide> {
    let mut guides: Vec<Guide> = nuclei
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != exclude)
        .map(|(index, n)| {
            let d = (0..3).map(|k| (n.pos[k] - origin[k]).powi(2)).sum::<f64>().sqrt();
            Guide { index, distance: d }
        })
        .filter(|g| g.distance <= GUIDE_RADIUS)
        .collect();
    guides.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    guides.truncate(GUIDE_MAX);
    guides
}

/// World point to physical pixel (origin top-left, +y down), the inverse of `pixel_ray`;
/// `None` for points behind the eye.
pub fn project(cam: &OrbitCamera, p: Vec3, w: f32, h: f32) -> Option<(f32, f32)> {
    let clip = cam.view_proj(w / h) * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(((ndc.x + 1.0) * 0.5 * w, (1.0 - ndc.y) * 0.5 * h))
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
    fn distance_guides_filter_sort_and_exclude_the_dragged_nucleus() {
        let n = |x: f64| SceneNucleus {
            z: 1,
            pos: [x, 0.0, 0.0],
            vel: [0.0; 3],
        };
        // origin is nucleus 0 at x = 0; 1 is at 3 Bohr, 2 at 1.5 Bohr, 3 is out of range
        let nuclei = vec![n(0.0), n(3.0), n(1.5), n(GUIDE_RADIUS + 0.1)];
        let g = distance_guides(nuclei[0].pos, &nuclei, Some(0));
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].index, 2);
        assert!((g[0].distance - 1.5).abs() < 1e-12);
        assert_eq!(g[1].index, 1);
        assert!((g[1].distance - 3.0).abs() < 1e-12);
        // without an exclusion the origin's own nucleus is a zero-length guide
        assert_eq!(distance_guides(nuclei[0].pos, &nuclei, None)[0].distance, 0.0);
    }

    #[test]
    fn distance_guides_are_capped_nearest_first() {
        let nuclei: Vec<SceneNucleus> = (1..=12)
            .map(|k| SceneNucleus {
                z: 1,
                pos: [k as f64 * 0.4, 0.0, 0.0],
                vel: [0.0; 3],
            })
            .collect();
        let g = distance_guides([0.0; 3], &nuclei, None);
        assert_eq!(g.len(), GUIDE_MAX);
        assert!(g.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn project_inverts_pixel_ray() {
        let cam = OrbitCamera::new(Vec3::new(0.5, -0.2, 0.0), 8.0);
        let (px, py, w, h) = (123.0, 456.0, 800.0, 600.0);
        let (o, d) = cam.pixel_ray(px, py, w, h);
        let p = o + d * 5.0;
        let (qx, qy) = project(&cam, p, w, h).unwrap();
        assert!((qx - px).abs() < 0.05 && (qy - py).abs() < 0.05, "({qx}, {qy})");
        // a point behind the eye has no pixel
        assert_eq!(project(&cam, o - d * 5.0, w, h), None);
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
