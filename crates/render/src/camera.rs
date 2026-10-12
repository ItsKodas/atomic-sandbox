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
