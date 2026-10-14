//! Orbit camera in world units (Bohr). Right-handed, +y up; wgpu clip space (depth 0..1).
use glam::camera::rh::{proj::directx, view};
use glam::{Mat4, Vec3};

/// Camera orbiting `target` at `distance`, parameterised by yaw (about +y) and pitch (elevation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32) -> OrbitCamera {
        OrbitCamera {
            target,
            yaw: 0.6,
            pitch: 0.35,
            distance,
            fov_y: 50f32.to_radians(),
            near: 0.05,
            far: 500.0,
        }
    }

    /// Unit direction from the eye to the target.
    fn forward(&self) -> Vec3 {
        -Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }

    pub fn eye(&self) -> Vec3 {
        self.target - self.forward() * self.distance
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Right-handed perspective projection with wgpu depth range 0..1 (y up).
    /// glam 0.34.1 moved `Mat4::perspective_rh` to `camera::rh::proj::directx::perspective`
    /// (the DirectX/WebGPU NDC convention).
    pub fn proj(&self, aspect: f32) -> Mat4 {
        directx::perspective(self.fov_y, aspect, self.near, self.far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }

    /// Rotate about the target: 0.005 rad per pixel; pitch clamped to ±1.5 rad (keeps clear of the +y pole).
    pub fn orbit(&mut self, dx_px: f32, dy_px: f32) {
        self.yaw -= dx_px * 0.005;
        self.pitch = (self.pitch + dy_px * 0.005).clamp(-1.5, 1.5);
    }

    /// Dolly: `distance *= 0.9^lines` (positive lines zoom in), never below 0.5.
    pub fn zoom(&mut self, scroll_lines: f32) {
        self.distance = (self.distance * 0.9f32.powf(scroll_lines)).max(0.5);
    }

    /// Move the target in the view plane so the scene follows the cursor.
    /// One screen height corresponds to `distance · 2 · tan(fov_y / 2)` world units.
    pub fn pan(&mut self, dx_px: f32, dy_px: f32, height_px: f32) {
        let world_per_px = 2.0 * self.distance * (self.fov_y / 2.0).tan() / height_px;
        let f = self.forward();
        let right = f.cross(Vec3::Y).normalize();
        let up = right.cross(f).normalize();
        self.target += (-right * dx_px + up * dy_px) * world_per_px;
    }

    pub fn focus(&mut self, point: Vec3) {
        self.target = point;
    }

    /// World-space ray through a pixel (origin, unit direction).
    ///
    /// Pixel coordinates have the origin at the top-left, +y down. The direction is taken from the
    /// unprojected near/far points on that pixel's line of sight; for a perspective projection that
    /// line passes through the eye, which is returned as the origin.
    pub fn pixel_ray(&self, px: f32, py: f32, width_px: f32, height_px: f32) -> (Vec3, Vec3) {
        let ndc_x = 2.0 * px / width_px - 1.0;
        let ndc_y = 1.0 - 2.0 * py / height_px;
        let inv = self.view_proj(width_px / height_px).inverse();
        let near = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        (self.eye(), (far - near).normalize())
    }

    /// Plane through `target` perpendicular to the view direction (the placement plane of spec §8).
    /// Returns (point, unit normal toward the eye).
    pub fn focus_plane(&self) -> (Vec3, Vec3) {
        (self.target, -self.forward())
    }
}

/// The part of the render target the 3D scene is drawn into, in physical pixels (origin top-left,
/// +y down): the viewer's central rect, left by its top bar, toolbar and dock. One value per frame is
/// shared by the renderer (render-pass viewport and scissor, the camera's aspect and the shaders'
/// pixel → NDC mapping) and by every mouse ↔ world path (picking, drag and placement planes,
/// projected overlays), which all take window pixel coordinates and go through [`Viewport::ray`] /
/// [`Viewport::project`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub origin: [f32; 2],
    pub size: [f32; 2],
}

impl Viewport {
    /// The whole `w × h` target.
    pub fn full(w: f32, h: f32) -> Viewport {
        Viewport { origin: [0.0, 0.0], size: [w.max(1.0), h.max(1.0)] }
    }

    /// Width / height (the camera's aspect).
    pub fn aspect(&self) -> f32 {
        self.size[0] / self.size[1]
    }

    pub fn centre(&self) -> (f32, f32) {
        (self.origin[0] + self.size[0] / 2.0, self.origin[1] + self.size[1] / 2.0)
    }

    /// Whether the window pixel lies inside the viewport.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.origin[0] && py >= self.origin[1] && px < self.origin[0] + self.size[0] && py < self.origin[1] + self.size[1]
    }

    /// World-space ray through window pixel (px, py) (origin, unit direction).
    pub fn ray(&self, cam: &OrbitCamera, px: f32, py: f32) -> (Vec3, Vec3) {
        cam.pixel_ray(px - self.origin[0], py - self.origin[1], self.size[0], self.size[1])
    }

    /// World point to window pixel, the inverse of [`Viewport::ray`]; `None` behind the eye.
    pub fn project(&self, cam: &OrbitCamera, p: Vec3) -> Option<(f32, f32)> {
        let clip = cam.view_proj(self.aspect()) * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some((
            self.origin[0] + (ndc.x + 1.0) * 0.5 * self.size[0],
            self.origin[1] + (1.0 - ndc.y) * 0.5 * self.size[1],
        ))
    }

    /// The scissor rect (x, y, w, h) inside a `target` of the given size, or `None` when empty.
    pub fn scissor(&self, target: [u32; 2]) -> Option<[u32; 4]> {
        let x0 = (self.origin[0].max(0.0).round() as u32).min(target[0]);
        let y0 = (self.origin[1].max(0.0).round() as u32).min(target[1]);
        let x1 = ((self.origin[0] + self.size[0]).round().max(0.0) as u32).min(target[0]);
        let y1 = ((self.origin[1] + self.size[1]).round().max(0.0) as u32).min(target[1]);
        (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
    }
}

/// Intersection of the ray `origin + t·dir` (t ≥ 0) with the plane through `point` with `normal`.
/// `None` if the ray is parallel to the plane or the plane lies behind the origin.
pub fn ray_plane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = dir.dot(normal);
    if denom.abs() < 1e-8 {
        return None;
    }
    let t = (point - origin).dot(normal) / denom;
    if t < 0.0 {
        None
    } else {
        Some(origin + dir * t)
    }
}

/// Nearest `t ≥ 0` at which the ray `origin + t·dir` meets the sphere; `dir` must be unit length.
/// If the origin is inside the sphere, returns the exit distance.
pub fn ray_sphere(origin: Vec3, dir: Vec3, centre: Vec3, radius: f32) -> Option<f32> {
    let oc = origin - centre;
    let b = oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t0 = -b - s;
    let t1 = -b + s;
    if t0 >= 0.0 {
        Some(t0)
    } else if t1 >= 0.0 {
        Some(t1)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use glam::{Vec3, Vec4};

    #[test]
    fn eye_is_distance_from_target_and_looks_at_it() {
        let c = OrbitCamera::new(Vec3::new(1.0, 2.0, 3.0), 10.0);
        assert_relative_eq!(c.eye().distance(c.target), 10.0, epsilon = 1e-5);
        let v = c.view();
        let t = v * Vec4::new(1.0, 2.0, 3.0, 1.0);
        // target sits on the −z axis in view space at distance 10
        assert_relative_eq!(t.x, 0.0, epsilon = 1e-4);
        assert_relative_eq!(t.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(t.z, -10.0, epsilon = 1e-4);
    }

    #[test]
    fn pixel_ray_through_centre_hits_target() {
        let c = OrbitCamera::new(Vec3::ZERO, 8.0);
        let (o, d) = c.pixel_ray(400.0, 300.0, 800.0, 600.0);
        let (p, n) = c.focus_plane();
        let hit = ray_plane(o, d, p, n).unwrap();
        assert!(hit.length() < 1e-3, "{hit:?}");
    }

    #[test]
    fn pixel_ray_offsets_map_to_plane_offsets() {
        let c = OrbitCamera::new(Vec3::ZERO, 8.0);
        let (o, d) = c.pixel_ray(400.0 + 100.0, 300.0, 800.0, 600.0);
        let (p, n) = c.focus_plane();
        let hit = ray_plane(o, d, p, n).unwrap();
        // 100 px right of centre at height 600 ⇒ world offset = (100/300)·distance·tan(fov/2) to the camera's right
        let expect = (100.0 / 300.0) * 8.0 * (c.fov_y / 2.0).tan();
        assert_relative_eq!(hit.length(), expect, epsilon = 1e-3);
        assert!(hit.dot(n).abs() < 1e-4);
    }

    /// An offset viewport: the projected target lands on the rect's centre, and the ray through the
    /// rect's centre passes through the target (the window-pixel frame, not the viewport's own).
    #[test]
    fn viewport_projection_and_rays_are_offset_by_its_origin() {
        let c = OrbitCamera::new(Vec3::new(0.5, -0.2, 0.3), 8.0);
        let vp = Viewport { origin: [120.0, 40.0], size: [900.0, 600.0] };
        let (cx, cy) = vp.centre();
        let (px, py) = vp.project(&c, c.target).unwrap();
        assert_relative_eq!(px, cx, epsilon = 1e-2);
        assert_relative_eq!(py, cy, epsilon = 1e-2);
        let (o, d) = vp.ray(&c, cx, cy);
        let (p, n) = c.focus_plane();
        let hit = ray_plane(o, d, p, n).unwrap();
        assert!(hit.distance(c.target) < 1e-3, "{hit:?}");
        // an off-centre pixel round-trips through ray → plane → project
        let (o, d) = vp.ray(&c, cx + 200.0, cy - 75.0);
        let q = vp.project(&c, ray_plane(o, d, p, n).unwrap()).unwrap();
        assert_relative_eq!(q.0, cx + 200.0, epsilon = 1e-2);
        assert_relative_eq!(q.1, cy - 75.0, epsilon = 1e-2);
        assert!(vp.contains(cx, cy) && !vp.contains(100.0, cy) && !vp.contains(cx, 650.0));
        assert_eq!(vp.scissor([1400, 900]), Some([120, 40, 900, 600]));
        assert_eq!(vp.scissor([1000, 500]), Some([120, 40, 880, 460]));
    }

    #[test]
    fn ray_sphere_nearest_hit() {
        let t = ray_sphere(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ZERO,
            1.0,
        )
        .unwrap();
        assert_relative_eq!(t, 4.0, epsilon = 1e-6);
        assert!(ray_sphere(
            Vec3::new(0.0, 3.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ZERO,
            1.0
        )
        .is_none());
        assert!(ray_sphere(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ZERO,
            1.0
        )
        .is_none()); // behind
    }

    #[test]
    fn pitch_is_clamped_and_zoom_bounded() {
        let mut c = OrbitCamera::new(Vec3::ZERO, 1.0);
        c.orbit(0.0, 10_000.0);
        assert!(c.pitch <= 1.5);
        c.zoom(1000.0);
        assert!(c.distance >= 0.5);
    }
}
