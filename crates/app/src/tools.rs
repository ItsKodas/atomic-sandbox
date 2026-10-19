//! Mouse tools in world space: picking, placement on the focus plane, and plane-constrained drag.
//! Pixel coordinates are window physical pixels, origin top-left; every function takes the frame's
//! [`Viewport`] (the central rect the scene is drawn into) and maps through it, so the mouse and the
//! render agree wherever the panels put the 3D view.
use crate::scene::{Scene, SceneNucleus};
use crate::sim::Command;
use glam::Vec3;
use render::camera::{ray_plane, ray_sphere, OrbitCamera, Viewport};
use sim::DynCommand;
use render::elements;

#[derive(Default)]
pub struct ToolState {
    pub selected: Option<usize>,
    /// A selected nucleus is being dragged (left button held after picking it).
    pub dragging: bool,
    /// The drag is currently the spring (`DragStart` sent, `DragEnd` due) rather than a teleport
    /// (`MoveNucleus`, paused only); `drag_commands` switches it with the run's state.
    pub spring: bool,
    /// (point, unit normal) of the plane the dragged nucleus moves in.
    pub drag_plane: Option<(Vec3, Vec3)>,
    /// Last cursor position over the 3D view (window physical pixels); `None` while it is over the
    /// chrome or outside the viewport.
    pub cursor: Option<(f32, f32)>,
    /// The Edit tool's selection (user request 2026-10-20): every selected nucleus, `selected` the
    /// last one clicked. Read through [`ToolState::set`].
    pub group: Vec<usize>,
    /// The Edit tool's handle being held.
    pub gizmo: Option<GizmoDrag>,
    /// The Edit tool's box selection: where the left button went down on empty space.
    pub boxing: Option<(f32, f32)>,
}

impl ToolState {
    /// The selected nuclei: the Edit tool's group while it still holds the primary selection (a
    /// click with another tool replaces it), else the one selected nucleus; indices below `n`.
    pub fn set(&self, n: usize) -> Vec<usize> {
        let group_ok = !self.group.is_empty() && self.selected.is_none_or(|s| self.group.contains(&s));
        let v: Vec<usize> = if group_ok { self.group.clone() } else { self.selected.into_iter().collect() };
        v.into_iter().filter(|&i| i < n).collect()
    }

    /// Clears the selection (Esc, empty-space clicks).
    pub fn clear_selection(&mut self) {
        self.selected = None;
        self.group.clear();
    }
}

/// An Edit-tool handle: an axis arrow (0 = x, 1 = y, 2 = z), or the square of the plane normal to
/// that axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    Axis(usize),
    Plane(usize),
}

/// A held Edit handle: the selection's positions when it was grabbed, the point on the handle's
/// line or plane under the cursor then, and whether the run was playing (it resumes on release).
#[derive(Clone, Debug)]
pub struct GizmoDrag {
    pub handle: Handle,
    pub base: Vec<(usize, [f64; 3])>,
    pub centre: Vec3,
    pub start: Vec3,
    pub resume: bool,
    /// The offset applied so far (Bohr).
    pub delta: [f64; 3],
}

/// The handles' arrow length on screen (window physical pixels).
pub const GIZMO_PX: f32 = 110.0;
/// The plane squares span this fraction of the arrow length from the centre.
pub const PLANE_SPAN: (f32, f32) = (0.22, 0.48);

pub fn axis(k: usize) -> Vec3 {
    [Vec3::X, Vec3::Y, Vec3::Z][k]
}

/// The world length that projects to [`GIZMO_PX`] at `c` (an arrow's length).
pub fn gizmo_length(cam: &OrbitCamera, c: Vec3, vp: Viewport) -> Option<f32> {
    let v = (c - cam.eye()).normalize_or_zero();
    let mut u = v.cross(Vec3::Y);
    if u.length_squared() < 1e-6 {
        u = v.cross(Vec3::X);
    }
    let (a, b) = (project(cam, c, vp)?, project(cam, c + u.normalize(), vp)?);
    let ppu = (b.0 - a.0).hypot(b.1 - a.1);
    (ppu > 1e-3).then(|| GIZMO_PX / ppu)
}

/// The corners of plane `k`'s square (world), around the centre `c` with arrow length `l`.
pub fn plane_quad(c: Vec3, l: f32, k: usize) -> [Vec3; 4] {
    let (a, b) = (axis((k + 1) % 3), axis((k + 2) % 3));
    let (s1, s2) = (PLANE_SPAN.0 * l, PLANE_SPAN.1 * l);
    [c + a * s1 + b * s1, c + a * s2 + b * s1, c + a * s2 + b * s2, c + a * s1 + b * s2]
}

fn seg_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}

fn in_quad(p: (f32, f32), q: &[(f32, f32); 4]) -> bool {
    let side = |a: (f32, f32), b: (f32, f32)| (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
    let s: Vec<f32> = (0..4).map(|i| side(q[i], q[(i + 1) % 4])).collect();
    s.iter().all(|&x| x >= 0.0) || s.iter().all(|&x| x <= 0.0)
}

/// The handle under the pixel: an arrow within 8 px of its shaft (the nearest), else a plane square.
pub fn gizmo_hit(cam: &OrbitCamera, vp: Viewport, c: Vec3, px: f32, py: f32) -> Option<Handle> {
    let l = gizmo_length(cam, c, vp)?;
    let c_px = project(cam, c, vp)?;
    let axis_hit = (0..3)
        .filter_map(|k| {
            let tip = project(cam, c + axis(k) * l, vp)?;
            let from = (c_px.0 + 0.15 * (tip.0 - c_px.0), c_px.1 + 0.15 * (tip.1 - c_px.1));
            let d = seg_dist((px, py), from, tip);
            (d < 8.0).then_some((d, k))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, k)) = axis_hit {
        return Some(Handle::Axis(k));
    }
    (0..3).find_map(|k| {
        let q = plane_quad(c, l, k);
        let q_px = [project(cam, q[0], vp)?, project(cam, q[1], vp)?, project(cam, q[2], vp)?, project(cam, q[3], vp)?];
        in_quad((px, py), &q_px).then_some(Handle::Plane(k))
    })
}

/// The point under the pixel on the handle's line (the closest point of the axis through `c` to
/// the pixel ray) or plane (through `c`); `None` when the ray runs along the axis or in the plane.
pub fn handle_point(cam: &OrbitCamera, vp: Viewport, px: f32, py: f32, h: Handle, c: Vec3) -> Option<Vec3> {
    let (o, d) = vp.ray(cam, px, py);
    match h {
        Handle::Axis(k) => {
            let e = axis(k);
            let w0 = c - o;
            let b = e.dot(d);
            let denom = 1.0 - b * b;
            if denom < 1e-4 {
                return None;
            }
            let t = (b * d.dot(w0) - e.dot(w0)) / denom;
            Some(c + e * t)
        }
        Handle::Plane(k) => {
            let n = axis(k);
            if d.dot(n).abs() < 0.03 {
                return None;
            }
            ray_plane(o, d, c, n)
        }
    }
}

/// The handle's offset from `start` to `now`, kept to its axis or plane.
pub fn handle_delta(h: Handle, start: Vec3, now: Vec3) -> [f64; 3] {
    let mut d = a3(now - start);
    match h {
        Handle::Axis(k) => (0..3).filter(|&j| j != k).for_each(|j| d[j] = 0.0),
        Handle::Plane(k) => d[k] = 0.0,
    }
    d
}

/// A rigid offset limited so every nucleus of `base` stays inside the cell's margin
/// ([`clamp_into_cell`]'s limit), per axis: the group keeps its shape at the wall.
pub fn clamp_delta(base: &[(usize, [f64; 3])], delta: [f64; 3], scene: &Scene) -> [f64; 3] {
    let cell = sim::Cell { n: scene.grid_n, h: scene.grid_h };
    let lim = (cell.max_coord() - 0.5 * cell.h).max(0.0);
    std::array::from_fn(|k| {
        let lo = base.iter().map(|(_, p)| -lim - p[k]).fold(f64::NEG_INFINITY, f64::max);
        let hi = base.iter().map(|(_, p)| lim - p[k]).fold(f64::INFINITY, f64::min);
        delta[k].max(lo.min(0.0)).min(hi.max(0.0))
    })
}

/// The nuclei whose centres project inside the screen rectangle with corners `a` and `b`.
pub fn box_select(cam: &OrbitCamera, vp: Viewport, nuclei: &[SceneNucleus], a: (f32, f32), b: (f32, f32)) -> Vec<usize> {
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    nuclei
        .iter()
        .enumerate()
        .filter_map(|(i, n)| project(cam, v3(n.pos), vp).filter(|p| (x0..=x1).contains(&p.0) && (y0..=y1).contains(&p.1)).map(|_| i))
        .collect()
}

/// Clamps a drag target, teleport or placement into the cell minus the providers' margin
/// (`sim::Cell::max_coord`: 3h from every face, ruling R12) less half a grid step of slack, so a
/// position the mouse produces never sits on — or beyond — the validation boundary.
pub fn clamp_into_cell(pos: [f64; 3], scene: &Scene) -> [f64; 3] {
    let cell = sim::Cell {
        n: scene.grid_n,
        h: scene.grid_h,
    };
    let lim = (cell.max_coord() - 0.5 * cell.h).max(0.0);
    pos.map(|x| x.clamp(-lim, lim))
}

/// The commands one motion event of a Select-drag sends, decided from the run's state at that
/// event, not at the press (review I1): while playing, the drag is the spring — `DragStart` at
/// the first playing event, `DragMove` after; while paused it is a teleport, `MoveNucleus` —
/// ending a spring first if the run was paused mid-drag. A teleport is never sent while
/// playing: the driver would refuse it, and it aborts the step in flight (R11).
pub fn drag_commands(spring: &mut bool, playing: bool, index: usize, pos: [f64; 3]) -> Vec<Command> {
    match (playing, *spring) {
        (true, false) => {
            *spring = true;
            vec![Command::Dyn(DynCommand::DragStart { index, target: pos })]
        }
        (true, true) => vec![Command::Dyn(DynCommand::DragMove { target: pos })],
        (false, true) => {
            *spring = false;
            vec![Command::Dyn(DynCommand::DragEnd), Command::MoveNucleus { index, pos }]
        }
        (false, false) => vec![Command::MoveNucleus { index, pos }],
    }
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

/// World point to window physical pixel (origin top-left, +y down) through the viewport, the inverse
/// of [`Viewport::ray`]; `None` for points behind the eye.
pub fn project(cam: &OrbitCamera, p: Vec3, vp: Viewport) -> Option<(f32, f32)> {
    vp.project(cam, p)
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
    vp: Viewport,
    nuclei: &[SceneNucleus],
) -> Option<usize> {
    let (o, d) = vp.ray(cam, px, py);
    nuclei
        .iter()
        .enumerate()
        .filter_map(|(i, n)| ray_sphere(o, d, v3(n.pos), elements::radius(n.z)).map(|t| (t, i)))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, i)| i)
}

/// Placement point: intersection of the pixel ray with the camera's focus plane.
pub fn place_point(cam: &OrbitCamera, px: f32, py: f32, vp: Viewport) -> Option<[f64; 3]> {
    let (o, d) = vp.ray(cam, px, py);
    let (p, n) = cam.focus_plane();
    ray_plane(o, d, p, n).map(a3)
}

/// Drag: intersection with the plane through the nucleus's original position, parallel to the focus plane.
pub fn drag_point(
    cam: &OrbitCamera,
    px: f32,
    py: f32,
    vp: Viewport,
    plane: (Vec3, Vec3),
) -> Option<[f64; 3]> {
    let (o, d) = vp.ray(cam, px, py);
    ray_plane(o, d, plane.0, plane.1).map(a3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneNucleus;
    use glam::Vec3;
    use render::camera::{OrbitCamera, Viewport};

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
                a: None,
                pos: [far.x as f64, far.y as f64, far.z as f64],
                vel: [0.0; 3],
            },
            SceneNucleus {
                z: 1,
                a: None,
                pos: [near.x as f64, near.y as f64, near.z as f64],
                vel: [0.0; 3],
            },
        ]
    }

    #[test]
    fn pick_returns_the_nearest_of_two_overlapping_nuclei() {
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let nuclei = two_overlapping();
        assert_eq!(pick(&cam, 400.0, 300.0, Viewport::full(800.0, 600.0), &nuclei), Some(1));
    }

    #[test]
    fn pick_misses_empty_space() {
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let nuclei = vec![SceneNucleus {
            z: 1,
            a: None,
            pos: [0.0; 3],
            vel: [0.0; 3],
        }];
        assert_eq!(pick(&cam, 5.0, 5.0, Viewport::full(800.0, 600.0), &nuclei), None);
    }

    #[test]
    fn place_point_lies_on_the_focus_plane() {
        let cam = OrbitCamera::new(Vec3::new(1.0, 0.0, 0.0), 10.0);
        let p = place_point(&cam, 500.0, 200.0, Viewport::full(800.0, 600.0)).unwrap();
        let (pt, n) = cam.focus_plane();
        let v = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) - pt;
        assert!(v.dot(n).abs() < 1e-4);
    }

    #[test]
    fn distance_guides_filter_sort_and_exclude_the_dragged_nucleus() {
        let n = |x: f64| SceneNucleus {
            z: 1,
            a: None,
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
                a: None,
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
        let (qx, qy) = project(&cam, p, Viewport::full(w, h)).unwrap();
        assert!((qx - px).abs() < 0.05 && (qy - py).abs() < 0.05, "({qx}, {qy})");
        // a point behind the eye has no pixel
        assert_eq!(project(&cam, o - d * 5.0, Viewport::full(w, h)), None);
    }

    /// The viewer's central viewport (offset by the top bar and with the dock on the right): the
    /// camera target projects to the rect's centre, and the ray back through that pixel is the same
    /// ray (through the target, from the eye); the placement point there is the target.
    #[test]
    fn a_point_projected_at_the_viewport_centre_unprojects_to_the_same_ray() {
        let cam = OrbitCamera::new(Vec3::new(0.4, 0.3, -0.2), 9.0);
        let vp = Viewport { origin: [0.0, 38.0], size: [972.0, 778.0] };
        let (cx, cy) = vp.centre();
        let (px, py) = project(&cam, cam.target, vp).unwrap();
        assert!((px - cx).abs() < 0.05 && (py - cy).abs() < 0.05, "({px}, {py}) vs ({cx}, {cy})");
        let (o, d) = vp.ray(&cam, px, py);
        let to_target = (cam.target - o).normalize();
        assert!(o.distance(cam.eye()) < 1e-4 && d.dot(to_target) > 1.0 - 1e-6, "{d:?} vs {to_target:?}");
        let p = place_point(&cam, cx, cy, vp).unwrap();
        assert!(v3(p).distance(cam.target) < 1e-3, "{p:?}");
        // the same pixel in a full-window frame is a different ray: the offset matters
        let (_, d_full) = Viewport::full(1400.0, 900.0).ray(&cam, cx, cy);
        assert!(d_full.dot(to_target) < 1.0 - 1e-4);
    }

    /// Picking at the viewport's centre selects the nucleus on the camera axis, and misses it at
    /// the window's centre (where the old full-window frame would have looked).
    #[test]
    fn picking_at_the_viewport_centre_selects_the_nucleus_on_the_camera_axis() {
        let cam = OrbitCamera::new(Vec3::new(1.0, 0.5, 0.0), 10.0);
        let vp = Viewport { origin: [0.0, 38.0], size: [972.0, 778.0] };
        let on_axis = cam.eye() + (cam.target - cam.eye()).normalize() * 7.0;
        let nuclei = vec![
            SceneNucleus { z: 1, a: None, pos: [on_axis.x as f64 + 3.0, on_axis.y as f64, on_axis.z as f64], vel: [0.0; 3] },
            SceneNucleus { z: 1, a: None, pos: [on_axis.x as f64, on_axis.y as f64, on_axis.z as f64], vel: [0.0; 3] },
        ];
        let (cx, cy) = vp.centre();
        assert_eq!(pick(&cam, cx, cy, vp, &nuclei), Some(1));
        assert_eq!(pick(&cam, 700.0, 450.0, vp, &nuclei[1..]), None, "the window centre is off the axis");
    }

    /// Review I1: the drag mode follows the run at every motion event — a teleport drag that
    /// sees Play becomes the spring (never a `MoveNucleus` while playing), and a spring drag that
    /// sees Pause ends the spring and teleports.
    #[test]
    fn drag_mode_follows_the_play_state_at_each_motion() {
        let mut spring = false;
        let p = [0.5, 0.0, 0.0];
        let paused = drag_commands(&mut spring, false, 1, p);
        assert!(matches!(paused.as_slice(), [Command::MoveNucleus { index: 1, .. }]) && !spring);
        let mut sent_while_playing = Vec::new();
        for k in 0..5 {
            sent_while_playing.extend(drag_commands(&mut spring, true, 1, [0.5 + 0.1 * k as f64, 0.0, 0.0]));
        }
        assert!(spring);
        assert!(matches!(sent_while_playing[0], Command::Dyn(DynCommand::DragStart { index: 1, target }) if target == p));
        assert!(sent_while_playing[1..].iter().all(|c| matches!(c, Command::Dyn(DynCommand::DragMove { .. }))));
        assert!(!sent_while_playing.iter().any(|c| matches!(c, Command::MoveNucleus { .. })), "no teleport while playing");
        let paused_again = drag_commands(&mut spring, false, 1, p);
        assert!(matches!(paused_again.as_slice(), [Command::Dyn(DynCommand::DragEnd), Command::MoveNucleus { index: 1, .. }]) && !spring);
    }

    /// Review I2: positions are clamped into the cell minus 3h and half a step (49³ at 0.25:
    /// half-edge 6, limit 6 − 0.75 − 0.125 = 5.125 Bohr).
    #[test]
    fn clamp_keeps_positions_inside_the_validation_margin() {
        let s = Scene::preset("h2").unwrap();
        assert_eq!(clamp_into_cell([10.0, -7.0, 1.0], &s), [5.125, -5.125, 1.0]);
        assert_eq!(clamp_into_cell([0.8, 0.0, 0.0], &s), [0.8, 0.0, 0.0]);
    }

    /// The Edit tool: an arrow is hit along its shaft, the axis drag follows the cursor along
    /// its axis only, and the group's offset stops at the cell's margin without changing its shape.
    #[test]
    fn edit_handles_hit_follow_and_clamp() {
        let cam = OrbitCamera::new(Vec3::new(0.3, 0.2, 0.1), 10.0);
        let vp = Viewport::full(800.0, 600.0);
        let c = Vec3::ZERO;
        let l = gizmo_length(&cam, c, vp).unwrap();
        for k in 0..3 {
            let mid = project(&cam, c + axis(k) * 0.7 * l, vp).unwrap();
            assert_eq!(gizmo_hit(&cam, vp, c, mid.0, mid.1), Some(Handle::Axis(k)), "axis {k}");
        }
        let c_px = project(&cam, c, vp).unwrap();
        let start = handle_point(&cam, vp, c_px.0, c_px.1, Handle::Axis(0), c).unwrap();
        let tip = project(&cam, c + Vec3::X * 2.0, vp).unwrap();
        let now = handle_point(&cam, vp, tip.0, tip.1, Handle::Axis(0), c).unwrap();
        let d = handle_delta(Handle::Axis(0), start, now);
        assert!((d[0] - 2.0).abs() < 1e-3 && d[1] == 0.0 && d[2] == 0.0, "{d:?}");
        // 49³ × 0.25: limit 5.125 Bohr; the group's right-most nucleus stops there
        let s = Scene::preset("h2").unwrap();
        let base = vec![(0, [4.0, 0.0, 0.0]), (1, [3.0, 1.0, 0.0])];
        assert_eq!(clamp_delta(&base, [3.0, -0.5, 0.0], &s), [1.125, -0.5, 0.0]);
        let one = [SceneNucleus { z: 1, a: None, pos: [0.0; 3], vel: [0.0; 3] }];
        assert_eq!(box_select(&cam, vp, &one, (c_px.0 - 10.0, c_px.1 - 10.0), (c_px.0 + 10.0, c_px.1 + 10.0)), vec![0]);
    }

    #[test]
    fn drag_keeps_the_nucleus_in_its_own_plane() {
        let cam = OrbitCamera::new(Vec3::ZERO, 10.0);
        let (_, n) = cam.focus_plane();
        let plane = (Vec3::new(0.0, 0.0, 2.0), n);
        let p = drag_point(&cam, 450.0, 320.0, Viewport::full(800.0, 600.0), plane).unwrap();
        let v = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) - plane.0;
        assert!(v.dot(n).abs() < 1e-4);
    }
}
