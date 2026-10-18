//! "Show what training is solving" (View ▸ Display aids): while a sampling pass runs, the 3D view
//! replaces the user's scene with the geometry the teacher is solving right now — a wireframe
//! ghost, so it cannot be mistaken for the scene — and brings the scene back when the pass ends,
//! is cancelled or fails. **Display only**: it reads `DynSnapshot::ghost` (`sim::PassGhost`: the
//! job's positions and Z, its stage and what it is, the last finished job's outcome — a few dozen
//! numbers from the pass's own queue, no density, no GPU readback) and draws it with
//! `render::lines`; nothing here feeds the solve or the scene.
//!
//! What is drawn, per stage (the labels are [`label`] / [`stage_line`]):
//! - **references**: the lone atom, centred as solved ("reference: isolated O");
//! - **perturbations**: the nudged geometry, with an arrow from each atom's unperturbed (pass
//!   base) position to its nudged one;
//! - **scans**: the scanned pair brighter and thicker, linked by a bright bundle, with the live
//!   "O–H · 1.50 × r₀ · 2.71 Bohr";
//! - **hot burst**: the moving geometry with a short fading trail of the last solved ones.
//!
//! Every atom is an outlined sphere in its element colour (`render::elements::colour`, linear):
//! the camera-facing silhouette and three world-axis great circles. Atoms closer than
//! [`LINK_FACTOR`] × the sum of their display radii are joined by a thin line (display only: the
//! factor is 2.2, not the 1.3 one might guess — the display radii are legibility choices, 0.35 Bohr
//! for H, and H₂'s own bond, 1.45 Bohr, is 2.07 × their sum). When a job finishes the *finished*
//! geometry is drawn once more in green (kept), red (failed, stalled or the SCF cap) or grey
//! (skipped), fading over [`FLASH_SECS`] — the solve log's outcome words.
use glam::Vec3;
use render::elements;
use render::lines::{self, LineVertex};
use sim::sampling::{GhostAtom, GhostJob, GhostOutcome, PassGhost};
use std::time::Instant;

/// The flash of a finished job fades out over this long.
pub const FLASH_SECS: f32 = 0.5;
/// How many finished samples stay on screen behind the one being solved (user request
/// 2026-10-18), newest first, and their alphas: gradually more transparent, in the outcome colour.
pub const HISTORY: usize = 3;
pub const HISTORY_ALPHA: [f32; HISTORY] = [0.45, 0.25, 0.12];
/// Two atoms are joined by a thin line when closer than this × the sum of their display radii.
pub const LINK_FACTOR: f32 = 2.2;
/// Segments of a circle.
const SEGMENTS: usize = 48;
/// The silhouette's alpha; the great circles are fainter.
const RING_ALPHA: f32 = 0.95;
const GREAT_ALPHA: f32 = 0.4;
/// Linear flash colours (the lines go to an sRGB target, like the element colours).
pub const KEPT: [f32; 3] = [0.10, 0.80, 0.18];
pub const FAILED: [f32; 3] = [0.95, 0.07, 0.07];
pub const SKIPPED: [f32; 3] = [0.40, 0.40, 0.40];
/// The scanned pair's bright link and the nudge arrows.
const PAIR_LINK: [f32; 4] = [1.0, 0.93, 0.55, 0.95];
const LINK: [f32; 4] = [0.85, 0.88, 0.95, 0.5];
const ARROW: [f32; 4] = [1.0, 1.0, 1.0, 0.9];
/// Strands and radius (Bohr) of the scanned pair's link bundle.
const PAIR_STRANDS: usize = 6;
const PAIR_RADIUS: f32 = 0.05;

/// The label over the ghost: what is being solved ("reference: isolated O", "random nudge 3 of 12",
/// "O–H · 1.50 × r₀ · 2.71 Bohr", "hot MD burst · step 17 of 40").
pub fn label(g: &PassGhost) -> String {
    let sym = |i: usize| g.atoms.get(i).map_or("?", |a| elements::symbol(a.z));
    match g.job {
        GhostJob::Reference { z } => format!("reference: isolated {}", elements::symbol(z)),
        GhostJob::Perturbation { index, of } => format!("random nudge {index} of {of}"),
        GhostJob::Scan { i, j, ratio, r, .. } => format!("{}–{} · {ratio:.2} × r₀ · {r:.2} Bohr", sym(i), sym(j)),
        GhostJob::HotBurst { step, of } => format!("hot MD burst · step {step} of {of}"),
    }
}

/// The first line of the label card: the pass stage in the Activity tab's words, with the scan
/// point ("bond scan · point 9 of 12").
pub fn stage_line(g: &PassGhost) -> String {
    match g.job {
        GhostJob::Reference { .. } => "reference".into(),
        GhostJob::Perturbation { .. } => "random nudges".into(),
        GhostJob::Scan { k, of, .. } => format!("bond scan · point {k} of {of}"),
        GhostJob::HotBurst { .. } => "hot MD burst".into(),
    }
}

/// The flash colour of an outcome: green kept, red failed or stalled, grey skipped.
pub fn outcome_colour(o: GhostOutcome) -> [f32; 3] {
    match o {
        GhostOutcome::Kept => KEPT,
        GhostOutcome::Failed(_) => FAILED,
        GhostOutcome::Skipped => SKIPPED,
    }
}

/// The flash's strength `age` seconds after the job finished: 1 at once, 0 from [`FLASH_SECS`] on.
pub fn flash_alpha(age: f32) -> f32 {
    if !(0.0..FLASH_SECS).contains(&age) {
        return 0.0;
    }
    1.0 - age / FLASH_SECS
}

/// A finished job's flash, as the viewer saw it arrive.
#[derive(Clone, Debug, PartialEq)]
pub struct Flash {
    pub at: Instant,
    pub outcome: GhostOutcome,
    pub atoms: Vec<GhostAtom>,
}

/// Follows the snapshots' ghosts: a new `finished.serial` starts a flash; no ghost (the pass
/// ended, was cancelled or failed) drops it, with the memory of the serial (the next pass counts
/// from 1 again).
#[derive(Clone, Debug, Default)]
pub struct Tracker {
    seen: Option<u64>,
    flash: Option<Flash>,
    /// The last [`HISTORY`] finished samples, newest first.
    recent: std::collections::VecDeque<Flash>,
}

impl Tracker {
    pub fn follow(&mut self, ghost: Option<&PassGhost>, now: Instant) {
        let Some(g) = ghost else {
            *self = Tracker::default();
            return;
        };
        let Some(f) = &g.finished else { return };
        if self.seen != Some(f.serial) {
            self.seen = Some(f.serial);
            let fl = Flash { at: now, outcome: f.outcome, atoms: f.atoms.clone() };
            self.recent.push_front(fl.clone());
            self.recent.truncate(HISTORY);
            self.flash = Some(fl);
        }
    }

    /// The last finished samples with their alphas ([`HISTORY_ALPHA`]), newest first; they stay
    /// until the pass ends.
    pub fn history(&self) -> Vec<(&Flash, f32)> {
        self.recent.iter().zip(HISTORY_ALPHA).collect()
    }

    /// The flash at `now` with its strength; `None` once it has faded.
    pub fn flash(&self, now: Instant) -> Option<(&Flash, f32)> {
        let f = self.flash.as_ref()?;
        let a = flash_alpha(now.saturating_duration_since(f.at).as_secs_f32());
        (a > 0.0).then_some((f, a))
    }

    /// A flash is still fading (the view needs another frame).
    pub fn animating(&self, now: Instant) -> bool {
        self.flash(now).is_some()
    }
}

fn v3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

/// A circle of `radius` about `c` in the plane spanned by the unit vectors `u`, `v`.
fn circle(out: &mut Vec<LineVertex>, c: Vec3, u: Vec3, v: Vec3, radius: f32, colour: [f32; 4]) {
    let at = |k: usize| {
        let t = k as f32 * std::f32::consts::TAU / SEGMENTS as f32;
        (c + radius * (t.cos() * u + t.sin() * v)).to_array()
    };
    for k in 0..SEGMENTS {
        lines::segment(out, at(k), at(k + 1), colour);
    }
}

/// An outlined sphere: the camera-facing silhouette (`right`, `up`) at full strength and the three
/// world-axis great circles fainter; `strands` > 1 thickens the silhouette with concentric rings
/// 3 % apart.
fn sphere_outline(out: &mut Vec<LineVertex>, c: Vec3, radius: f32, rgb: [f32; 3], alpha: f32, cam: &CamBasis, strands: usize) {
    let ring = |a: f32| [rgb[0], rgb[1], rgb[2], a];
    for s in 0..strands {
        let k = s as f32 - 0.5 * (strands as f32 - 1.0);
        circle(out, c, cam.right, cam.up, radius * (1.0 + 0.03 * k), ring(RING_ALPHA * alpha));
    }
    let g = ring(GREAT_ALPHA * alpha);
    circle(out, c, Vec3::X, Vec3::Y, radius, g);
    circle(out, c, Vec3::Y, Vec3::Z, radius, g);
    circle(out, c, Vec3::Z, Vec3::X, radius, g);
}

/// The camera's right and up (world space) and the direction towards it.
#[derive(Clone, Copy, Debug)]
pub struct CamBasis {
    pub right: Vec3,
    pub up: Vec3,
}

impl CamBasis {
    pub fn of(camera: &render::camera::OrbitCamera) -> CamBasis {
        let m = camera.view().inverse();
        CamBasis { right: m.x_axis.truncate().normalize_or_zero(), up: m.y_axis.truncate().normalize_or_zero() }
    }

    fn toward(&self) -> Vec3 {
        self.right.cross(self.up)
    }
}

/// A unit vector perpendicular to `dir` in the plane facing the camera.
fn facing_perp(dir: Vec3, cam: &CamBasis) -> Vec3 {
    let p = dir.cross(cam.toward());
    if p.length_squared() > 1e-8 { p.normalize() } else { cam.up }
}

/// The pairs of `atoms` that get a thin link (closer than [`LINK_FACTOR`] × the sum of their
/// display radii), plus the scanned pair, which always has one.
pub fn links(atoms: &[GhostAtom], scanned: Option<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..atoms.len() {
        for j in i + 1..atoms.len() {
            let d = (v3(atoms[j].pos) - v3(atoms[i].pos)).length();
            if scanned == Some((i, j)) || d < LINK_FACTOR * (elements::radius(atoms[i].z) + elements::radius(atoms[j].z)) {
                out.push((i, j));
            }
        }
    }
    out
}

/// The ghost's line list for a frame: outlines, links, the stage's extras (arrows, the pair
/// bundle, the trail) and, while it fades, the last finished job's flash.
pub fn build(g: &PassGhost, cam: &CamBasis, flash: Option<(&Flash, f32)>, history: &[(&Flash, f32)]) -> Vec<LineVertex> {
    let mut out = Vec::new();
    // the last few finished samples, under the current one, fading with age (outcome colours)
    for (f, alpha) in history {
        let c = outcome_colour(f.outcome);
        for a in &f.atoms {
            sphere_outline(&mut out, v3(a.pos), elements::radius(a.z), c, *alpha, cam, 1);
        }
        for (i, j) in links(&f.atoms, None) {
            let (a, b) = (v3(f.atoms[i].pos), v3(f.atoms[j].pos));
            let (ra, rb) = (elements::radius(f.atoms[i].z), elements::radius(f.atoms[j].z));
            let axis = b - a;
            let len = axis.length();
            if len > ra + rb {
                let dir = axis / len;
                lines::segment(&mut out, (a + dir * ra).to_array(), (b - dir * rb).to_array(), [c[0], c[1], c[2], *alpha]);
            }
        }
    }
    let scanned = match g.job {
        GhostJob::Scan { i, j, .. } => Some((i.min(j), i.max(j))),
        _ => None,
    };
    // the hot burst's trail, under everything: each atom's last positions, fading in
    if !g.trail.is_empty() {
        for (i, a) in g.atoms.iter().enumerate() {
            let c = elements::colour(a.z);
            let pts: Vec<Vec3> = g.trail.iter().filter_map(|t| t.get(i).copied()).map(v3).chain(std::iter::once(v3(a.pos))).collect();
            let n = pts.len();
            for (k, w) in pts.windows(2).enumerate() {
                let alpha = 0.12 + 0.5 * (k + 1) as f32 / (n - 1).max(1) as f32;
                lines::segment(&mut out, w[0].to_array(), w[1].to_array(), [c[0], c[1], c[2], alpha]);
            }
        }
    }
    for (i, a) in g.atoms.iter().enumerate() {
        let c = elements::colour(a.z);
        let hot = scanned.is_some_and(|(p, q)| i == p || i == q);
        sphere_outline(&mut out, v3(a.pos), elements::radius(a.z), c, 1.0, cam, if hot { 3 } else { 1 });
    }
    for (i, j) in links(&g.atoms, scanned) {
        let (a, b) = (v3(g.atoms[i].pos), v3(g.atoms[j].pos));
        let (ra, rb) = (elements::radius(g.atoms[i].z), elements::radius(g.atoms[j].z));
        let axis = b - a;
        let len = axis.length();
        if len <= ra + rb {
            continue; // the outlines overlap: no gap to bridge
        }
        let dir = axis / len;
        let (from, to) = (a + dir * ra, b - dir * rb);
        if scanned == Some((i, j)) {
            let u = facing_perp(dir, cam);
            let w = dir.cross(u).normalize_or_zero();
            for k in 0..PAIR_STRANDS {
                let t = k as f32 * std::f32::consts::TAU / PAIR_STRANDS as f32;
                let off = PAIR_RADIUS * (t.cos() * u + t.sin() * w);
                lines::segment(&mut out, (from + off).to_array(), (to + off).to_array(), PAIR_LINK);
            }
        } else {
            lines::segment(&mut out, from.to_array(), to.to_array(), LINK);
        }
    }
    // perturbations: unperturbed -> nudged
    if matches!(g.job, GhostJob::Perturbation { .. }) {
        for (a, b) in g.base.iter().zip(&g.atoms) {
            arrow(&mut out, v3(*a), v3(b.pos), cam);
        }
    }
    if let Some((f, alpha)) = flash {
        let c = outcome_colour(f.outcome);
        for a in &f.atoms {
            sphere_outline(&mut out, v3(a.pos), elements::radius(a.z) * 1.08, c, alpha, cam, 3);
        }
    }
    out
}

/// A small arrow from `from` to `to` (nothing under 0.01 Bohr): the shaft and two head strokes in
/// the plane facing the camera.
fn arrow(out: &mut Vec<LineVertex>, from: Vec3, to: Vec3, cam: &CamBasis) {
    let axis = to - from;
    let len = axis.length();
    if len < 0.01 {
        return;
    }
    let dir = axis / len;
    lines::segment(out, from.to_array(), to.to_array(), ARROW);
    let head = (0.3 * len).min(0.12);
    let perp = facing_perp(dir, cam);
    for s in [-1.0f32, 1.0] {
        let tip_back = to - dir * head * 0.9 + perp * (s * head * 0.45);
        lines::segment(out, to.to_array(), tip_back.to_array(), ARROW);
    }
}

#[cfg(test)]
mod tests;
