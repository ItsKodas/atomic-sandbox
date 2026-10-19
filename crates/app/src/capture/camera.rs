//! The capture's frame and camera (display only): the frame's aspect ratio and size, the camera it
//! is drawn from — the window's view (cropped to the aspect: the guide frame drawn over the view
//! is exactly what is captured), a fixed angle, or an orbit about the scene — the gizmo that shows
//! a fixed or orbiting capture camera in the editor view, and the distance scale bar.
use super::Resolution;
use render::camera::OrbitCamera;
use render::lines::{self, LineVertex};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aspect {
    /// The 3D view's own.
    View,
    Wide,
    Classic,
    Square,
    Portrait,
    Cinema,
}

impl Aspect {
    pub const ALL: [Aspect; 6] = [Aspect::View, Aspect::Wide, Aspect::Classic, Aspect::Square, Aspect::Portrait, Aspect::Cinema];

    pub fn label(self) -> &'static str {
        match self {
            Aspect::View => "view",
            Aspect::Wide => "16:9",
            Aspect::Classic => "4:3",
            Aspect::Square => "1:1",
            Aspect::Portrait => "9:16",
            Aspect::Cinema => "21:9",
        }
    }

    /// Width / height for a view of `view` pixels.
    pub fn ratio(self, view: (f32, f32)) -> f32 {
        match self {
            Aspect::View => view.0.max(1.0) / view.1.max(1.0),
            Aspect::Wide => 16.0 / 9.0,
            Aspect::Classic => 4.0 / 3.0,
            Aspect::Square => 1.0,
            Aspect::Portrait => 9.0 / 16.0,
            Aspect::Cinema => 21.0 / 9.0,
        }
    }
}

/// The frame's size in pixels: the view's height (the view's size itself at its own aspect), or
/// the named resolution as the short side (1080p 9:16 is 1080 × 1920). Even, for yuv420p.
pub fn frame_size(res: Resolution, aspect: Aspect, view: (u32, u32)) -> (u32, u32) {
    let view = (view.0.max(16), view.1.max(16));
    let even = |x: f32| ((x.round() as u32) & !1).max(16);
    if res == Resolution::View && aspect == Aspect::View {
        return (even(view.0 as f32), even(view.1 as f32));
    }
    let a = aspect.ratio((view.0 as f32, view.1 as f32));
    let (w, h) = match res.short_side() {
        None => (view.1 as f32 * a, view.1 as f32),
        Some(s) if a >= 1.0 => (s * a, s),
        Some(s) => (s, s / a),
    };
    (even(w), even(h))
}

/// `size` scaled down (aspect kept, even) so its longer side is at most `max_side` (the GPU's
/// texture limit: 8192 px, or 4096 in the Camera view, whose glow pass supersamples by 2).
pub fn fit(size: (u32, u32), max_side: u32) -> (u32, u32) {
    let long = size.0.max(size.1);
    if long <= max_side || max_side == 0 {
        return size;
    }
    let k = max_side as f64 / long as f64;
    let even = |x: u32| (((x as f64 * k).floor() as u32) & !1).max(16);
    (even(size.0), even(size.1))
}

/// The largest rect of aspect `ratio` centred in `(x, y, w, h)`: what a capture from the window's
/// view takes (the guide frame).
pub fn inscribed(view: (f32, f32, f32, f32), ratio: f32) -> (f32, f32, f32, f32) {
    let (x, y, w, h) = view;
    let (gw, gh) = if w / h > ratio { (h * ratio, h) } else { (w, w / ratio) };
    (x + (w - gw) / 2.0, y + (h - gh) / 2.0, gw, gh)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraMode {
    /// The window's camera (cropped to the frame's aspect).
    View,
    /// A fixed angle and distance about the window camera's target.
    Fixed,
    /// Round the target at a steady rate (per second of video), from the fixed angle.
    Orbit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraSettings {
    pub mode: CameraMode,
    /// The angle round the vertical (degrees; the orbit camera's yaw).
    pub azimuth_deg: f32,
    /// Above (+) or below (−) the target's horizontal plane (degrees).
    pub elevation_deg: f32,
    /// From the target (Bohr).
    pub distance: f32,
    /// The orbit's rate (degrees per second of video; negative turns the other way).
    pub orbit_deg_per_s: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        CameraSettings { mode: CameraMode::View, azimuth_deg: 34.0, elevation_deg: 20.0, distance: 14.0, orbit_deg_per_s: 20.0 }
    }
}

/// The elevation's limit (the orbit camera keeps clear of the poles).
pub const MAX_ELEVATION: f32 = 85.0;

impl CameraSettings {
    /// Takes the window camera's angle and distance (switching to a fixed or orbiting camera starts
    /// from what is on screen).
    pub fn from_view(&mut self, cam: &OrbitCamera) {
        self.azimuth_deg = wrap_deg(cam.yaw.to_degrees());
        self.elevation_deg = cam.pitch.to_degrees().clamp(-MAX_ELEVATION, MAX_ELEVATION);
        self.distance = cam.distance;
    }
}

/// An angle in (−180, 180].
pub fn wrap_deg(a: f32) -> f32 {
    let w = (a + 180.0).rem_euclid(360.0) - 180.0;
    if w == -180.0 { 180.0 } else { w }
}

/// The camera a frame is drawn from: `window` is the window's, `view_ratio` its view's aspect,
/// `frame_ratio` the frame's, `t_video` the frame's time in the video (s; the orbit's phase).
pub fn capture_camera(window: &OrbitCamera, view_ratio: f32, frame_ratio: f32, s: &CameraSettings, t_video: f64) -> OrbitCamera {
    match s.mode {
        CameraMode::View => {
            // the guide frame's share of the view's height narrows the vertical field of view
            let k = (view_ratio / frame_ratio).min(1.0);
            OrbitCamera { fov_y: 2.0 * ((window.fov_y / 2.0).tan() * k).atan(), ..*window }
        }
        CameraMode::Fixed | CameraMode::Orbit => {
            let turn = if s.mode == CameraMode::Orbit { s.orbit_deg_per_s as f64 * t_video } else { 0.0 };
            OrbitCamera {
                yaw: (s.azimuth_deg as f64 + turn).to_radians() as f32,
                pitch: s.elevation_deg.clamp(-MAX_ELEVATION, MAX_ELEVATION).to_radians(),
                distance: s.distance.max(0.5),
                ..*window
            }
        }
    }
}

/// The capture camera drawn in the editor view: its frustum (the frame's aspect), the line to
/// the target and, orbiting, its path. Display only; never in a capture.
pub fn gizmo(cam: &OrbitCamera, frame_ratio: f32, orbit: bool) -> Vec<LineVertex> {
    const COLOUR: [f32; 4] = [1.0, 0.72, 0.28, 0.95];
    const FAINT: [f32; 4] = [1.0, 0.72, 0.28, 0.35];
    let mut out = Vec::new();
    let eye = cam.eye();
    let f = (cam.target - eye).normalize_or(Vec3::NEG_Z);
    let right = f.cross(Vec3::Y).normalize_or(Vec3::X);
    let up = right.cross(f).normalize_or(Vec3::Y);
    let l = (cam.distance * 0.18).clamp(0.6, 3.0);
    let ty = (cam.fov_y / 2.0).tan() * l;
    let tx = ty * frame_ratio;
    let c = eye + f * l;
    let corners = [c - right * tx - up * ty, c + right * tx - up * ty, c + right * tx + up * ty, c - right * tx + up * ty];
    for k in 0..4 {
        lines::segment(&mut out, eye.to_array(), corners[k].to_array(), COLOUR);
        lines::segment(&mut out, corners[k].to_array(), corners[(k + 1) % 4].to_array(), COLOUR);
    }
    // the frame's up: a small triangle on its top edge
    let top = c + up * ty;
    lines::segment(&mut out, (top - right * tx * 0.3 + up * 0.05).to_array(), (top + up * ty * 0.45).to_array(), COLOUR);
    lines::segment(&mut out, (top + right * tx * 0.3 + up * 0.05).to_array(), (top + up * ty * 0.45).to_array(), COLOUR);
    // dashed to the target
    let n = 16;
    for k in (0..n).step_by(2) {
        let a = eye.lerp(cam.target, k as f32 / n as f32);
        let b = eye.lerp(cam.target, (k + 1) as f32 / n as f32);
        lines::segment(&mut out, a.to_array(), b.to_array(), FAINT);
    }
    if orbit {
        let (r, y) = (cam.distance * cam.pitch.cos(), cam.target.y + cam.distance * cam.pitch.sin());
        let m = 72;
        for k in 0..m {
            let p = |k: usize| {
                let a = std::f32::consts::TAU * k as f32 / m as f32;
                [cam.target.x + r * a.sin(), y, cam.target.z + r * a.cos()]
            };
            lines::segment(&mut out, p(k), p(k + 1), FAINT);
        }
    }
    out
}

/// The scale bar for a frame `height_pt` points tall drawn by `cam` (at the target's distance):
/// the longest of 0.1, 0.2, 0.5, 1, 2, 5 … Å that fits in `max_pt`, and its length in points.
pub fn scale_bar(cam: &OrbitCamera, height_pt: f32, max_pt: f32) -> (f64, f32) {
    let pt_per_bohr = height_pt / (2.0 * cam.distance * (cam.fov_y / 2.0).tan());
    let pt_per_angstrom = pt_per_bohr * units::angstrom_to_bohr(1.0) as f32;
    let steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
    let pick = steps.iter().rev().find(|&&a| a as f32 * pt_per_angstrom <= max_pt).copied().unwrap_or(steps[0]);
    (pick, pick as f32 * pt_per_angstrom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sizes_follow_the_aspect_and_the_short_side() {
        assert_eq!(frame_size(Resolution::View, Aspect::View, (1001, 701)), (1000, 700));
        assert_eq!(frame_size(Resolution::P1080, Aspect::Wide, (800, 800)), (1920, 1080));
        assert_eq!(frame_size(Resolution::P1080, Aspect::Square, (800, 600)), (1080, 1080));
        assert_eq!(frame_size(Resolution::P1080, Aspect::Portrait, (800, 600)), (1080, 1920));
        assert_eq!(frame_size(Resolution::P2160, Aspect::Wide, (1, 1)), (3840, 2160));
        // the view's height at another aspect
        assert_eq!(frame_size(Resolution::View, Aspect::Square, (1200, 800)), (800, 800));
        // a named resolution at the view's own aspect
        assert_eq!(frame_size(Resolution::P1080, Aspect::View, (1200, 800)), (1620, 1080));
        // 4K 21:9 in the Camera view: 5040 × 2160 scaled into 4096
        assert_eq!(fit(frame_size(Resolution::P2160, Aspect::Cinema, (1, 1)), 4096), (4096, 1754));
        assert_eq!(fit((3840, 2160), 8192), (3840, 2160));
    }

    #[test]
    fn the_guide_is_inscribed_and_the_view_camera_sees_exactly_it() {
        let g = inscribed((100.0, 50.0, 1200.0, 800.0), 1.0);
        assert_eq!(g, (300.0, 50.0, 800.0, 800.0));
        let g = inscribed((0.0, 0.0, 1200.0, 800.0), 21.0 / 9.0);
        assert!((g.2 - 1200.0).abs() < 1e-3 && (g.3 - 1200.0 * 9.0 / 21.0).abs() < 1e-3);
        // a wide frame from a 3:2 view: its half-width angle is the view's, its height the guide's
        let w = OrbitCamera::new(Vec3::ZERO, 10.0);
        let (vr, fr) = (1.5, 21.0 / 9.0);
        let c = capture_camera(&w, vr, fr, &CameraSettings::default(), 0.0);
        let half_w = |cam: &OrbitCamera, r: f32| (cam.fov_y / 2.0).tan() * r;
        assert!((half_w(&c, fr) - half_w(&w, vr)).abs() < 1e-5);
        // a square frame keeps the view's height
        assert_eq!(capture_camera(&w, vr, 1.0, &CameraSettings::default(), 0.0).fov_y, w.fov_y);
    }

    #[test]
    fn fixed_and_orbit_cameras() {
        let w = OrbitCamera::new(Vec3::new(1.0, 0.0, 0.0), 10.0);
        let s = CameraSettings { mode: CameraMode::Fixed, azimuth_deg: 90.0, elevation_deg: 0.0, distance: 20.0, orbit_deg_per_s: 30.0 };
        let c = capture_camera(&w, 1.5, 1.5, &s, 5.0);
        assert!((c.eye() - Vec3::new(21.0, 0.0, 0.0)).length() < 1e-4, "{:?}", c.eye());
        let o = capture_camera(&w, 1.5, 1.5, &CameraSettings { mode: CameraMode::Orbit, ..s.clone() }, 3.0);
        assert!((o.yaw.to_degrees() - 180.0).abs() < 1e-3, "90° + 30°/s × 3 s");
        assert!(!gizmo(&o, 16.0 / 9.0, true).is_empty());
        let mut t = s.clone();
        t.from_view(&OrbitCamera { yaw: 7.0, pitch: 1.4, ..w });
        assert!((t.azimuth_deg - wrap_deg(7f32.to_degrees())).abs() < 1e-3 && t.elevation_deg <= MAX_ELEVATION && t.distance == 10.0);
        assert_eq!(wrap_deg(190.0), -170.0);
        assert_eq!(wrap_deg(-180.0), 180.0);
    }

    #[test]
    fn the_scale_bar_picks_a_round_length() {
        let c = OrbitCamera::new(Vec3::ZERO, 14.0);
        let (a, pt) = scale_bar(&c, 900.0, 200.0);
        assert!(pt <= 200.0 && [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0].contains(&a), "{a} Å, {pt} pt");
        let (b, _) = scale_bar(&c, 900.0, 2.0 * pt + 1.0);
        assert!(b > a);
    }
}
