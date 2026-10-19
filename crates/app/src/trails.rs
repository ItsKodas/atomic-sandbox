//! Atom trails (user request 2026-10-17): a fading single line behind each atom showing where it
//! has been. **Display only** — the viewer keeps a per-atom history of (simulated time, position)
//! from the snapshots it receives while View ▸ Display aids ▸ Trails is on, and draws it through
//! `render::lines` like the bonds. Nothing here reaches the worker or the solve; with Trails off no
//! history is collected at all.
//!
//! The *length* is simulated time (fs): a point stays visible while its age — the newest recorded
//! time minus its own — is within the length, so the trail covers the same motion at any play speed.
//! The *opacity* is the alpha of the newest end; it fades linearly to zero at age = length
//! ([`alpha_at`]). The history is capped at [`MAX_POINTS`] per atom; past it the oldest half is
//! thinned to every other point, so a long run at high speed cannot grow without bound.
//!
//! Breaks (a trail is never drawn across one): a teleport clears that atom's history, a change of
//! the nucleus count, a replaced scene, a relax and time going backwards clear everything
//! ([`Trails::on_command`], [`Trails::record`]); a drag keeps the trail.
use crate::scene::SceneNucleus;
use crate::sim::Command;
use render::lines::LineVertex;

/// The Length slider's range and default, in simulated femtoseconds.
pub const LENGTH_MIN_FS: f64 = 5.0;
pub const LENGTH_MAX_FS: f64 = 10_000.0;
pub const LENGTH_DEFAULT_FS: f64 = 100.0;
/// The newest end's alpha by default (the Opacity slider is 0–100 %).
pub const OPACITY_DEFAULT: f32 = 0.7;
/// At most this many points per atom; past it the oldest half is thinned to every other point.
pub const MAX_POINTS: usize = 4096;

/// The Length slider's position (0 at [`LENGTH_MIN_FS`], 1 at [`LENGTH_MAX_FS`]) for a length in
/// fs: logarithmic, so 5 fs … 10 ps gets the same room per decade.
pub fn length_to_slider(fs: f64) -> f64 {
    let f = fs.clamp(LENGTH_MIN_FS, LENGTH_MAX_FS);
    ((f / LENGTH_MIN_FS).ln() / (LENGTH_MAX_FS / LENGTH_MIN_FS).ln()).clamp(0.0, 1.0)
}

/// The length (fs) at a slider position in [0, 1]: the inverse of [`length_to_slider`].
pub fn slider_to_length(x: f64) -> f64 {
    LENGTH_MIN_FS * (LENGTH_MAX_FS / LENGTH_MIN_FS).powf(x.clamp(0.0, 1.0))
}

/// A trail vertex's alpha at `age_fs` for a trail of `length_fs` whose newest end has `opacity`:
/// linear from `opacity` at age 0 to 0 at age = length (and beyond).
pub fn alpha_at(age_fs: f64, length_fs: f64, opacity: f32) -> f32 {
    if !(length_fs > 0.0) || !age_fs.is_finite() {
        return 0.0;
    }
    opacity.clamp(0.0, 1.0) * (1.0 - age_fs / length_fs).clamp(0.0, 1.0) as f32
}

/// The Length slider's text: "5 fs", "120 fs", "1.5 ps", "10 ps".
pub fn length_text(fs: f64) -> String {
    if fs >= 1000.0 {
        let ps = fs / 1000.0;
        if (ps - ps.round()).abs() < 0.05 { format!("{} ps", ps.round()) } else { format!("{ps:.1} ps") }
    } else if fs >= 100.0 {
        format!("{fs:.0} fs")
    } else {
        format!("{fs:.1} fs").replace(".0 fs", " fs")
    }
}

/// A length typed into the slider's box, in fs: "200", "200 fs", "1.5 ps" (a bare number is fs).
pub fn parse_length(text: &str) -> Option<f64> {
    let t = text.trim().to_lowercase();
    let (num, scale) = match (t.strip_suffix("ps"), t.strip_suffix("fs")) {
        (Some(n), _) => (n, 1000.0),
        (_, Some(n)) => (n, 1.0),
        _ => (t.as_str(), 1.0),
    };
    num.trim().parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0).map(|x| x * scale)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    /// Simulated time (fs).
    t: f64,
    /// Bohr.
    pos: [f64; 3],
}

/// The per-atom position history.
#[derive(Clone, Debug)]
pub struct Trails {
    atoms: Vec<Vec<Point>>,
    /// The newest simulated time recorded (fs); `-inf` when nothing is.
    newest: f64,
}

impl Default for Trails {
    fn default() -> Trails {
        Trails::new()
    }
}

impl Trails {
    pub fn new() -> Trails {
        Trails { atoms: Vec::new(), newest: f64::NEG_INFINITY }
    }

    /// Forgets every trail.
    pub fn clear(&mut self) {
        self.atoms.clear();
        self.newest = f64::NEG_INFINITY;
    }

    /// No atom has a recorded point.
    pub fn is_empty(&self) -> bool {
        self.atoms.iter().all(|a| a.is_empty())
    }

    /// The recorded points of atom `i` (tests).
    #[cfg(test)]
    pub fn points(&self, i: usize) -> usize {
        self.atoms.get(i).map_or(0, |a| a.len())
    }

    /// Starts a fresh trail for atom `i` (a teleport).
    pub fn break_atom(&mut self, i: usize) {
        if let Some(a) = self.atoms.get_mut(i) {
            a.clear();
        }
    }

    /// What a command sent to the worker does to the trails: a teleport breaks that atom's, and
    /// anything that adds or removes a nucleus, replaces the scene (time restarts), or moves the
    /// atoms without the clock (a relax) clears them all. Drags and time control keep them.
    pub fn on_command(&mut self, cmd: &Command) {
        match cmd {
            Command::MoveNucleus { index, .. } => self.break_atom(*index),
            Command::SetScene(_)
            | Command::AddNucleus { .. }
            | Command::RemoveNucleus(_)
            | Command::SetCharge(_)
            | Command::SetMultiplicity(_)
            | Command::SetGrid { .. }
            | Command::SetSigma(_)
            | Command::Dyn(sim::DynCommand::Relax) => self.clear(),
            _ => {}
        }
    }

    /// Records the positions (Bohr) at simulated time `t_fs` and drops the points older than
    /// `length_fs`. A different nucleus count, or a time earlier than the newest recorded (a
    /// rewind, a reset), starts every trail afresh; an atom already holding a point at exactly
    /// `t_fs` (the clock did not move: paused, a relax) gets that point moved, not a new one.
    pub fn record(&mut self, t_fs: f64, pos: &[[f64; 3]], length_fs: f64) {
        if !t_fs.is_finite() {
            return;
        }
        if pos.len() != self.atoms.len() {
            self.atoms = vec![Vec::new(); pos.len()];
            self.newest = f64::NEG_INFINITY;
        } else if t_fs < self.newest {
            self.atoms.iter_mut().for_each(Vec::clear);
            self.newest = f64::NEG_INFINITY;
        }
        self.newest = self.newest.max(t_fs);
        let cutoff = self.newest - length_fs;
        for (a, &p) in self.atoms.iter_mut().zip(pos) {
            if a.last().is_some_and(|l| l.t == t_fs) {
                a.last_mut().expect("just seen").pos = p;
            } else {
                a.push(Point { t: t_fs, pos: p });
            }
            let old = a.iter().take_while(|q| q.t < cutoff).count();
            a.drain(..old);
            if a.len() > MAX_POINTS {
                thin_oldest_half(a);
            }
        }
    }

    /// The trail lines for the nuclei drawn now: for each atom a polyline from its oldest point
    /// within `length_fs` to its current drawn position, in its element colour with the alpha
    /// fading from `opacity` at the newest end to 0 at age = length. An atom with no history (just
    /// teleported, just switched on) draws nothing; the current position counts as age 0.
    pub fn lines(&self, nuclei: &[SceneNucleus], length_fs: f64, opacity: f32) -> Vec<LineVertex> {
        let mut out = Vec::new();
        if self.atoms.len() != nuclei.len() {
            return out;
        }
        for (hist, n) in self.atoms.iter().zip(nuclei) {
            let cutoff = self.newest - length_fs;
            let mut pts: Vec<(f64, [f32; 3])> =
                hist.iter().filter(|p| p.t >= cutoff).map(|p| (p.t, p.pos.map(|x| x as f32))).collect();
            if pts.is_empty() {
                continue;
            }
            let here = n.pos.map(|x| x as f32);
            // smooth motion draws the sphere between the last two snapshots: the newest point is
            // then ahead of it, so end the line at the sphere instead of doubling back
            if pts.len() >= 2 && pts[pts.len() - 1].1 != here {
                pts.pop();
            }
            pts.push((self.newest, here));
            let rgb = render::elements::colour(n.z);
            let vertex = |&(t, pos): &(f64, [f32; 3])| LineVertex { pos, colour: [rgb[0], rgb[1], rgb[2], alpha_at(self.newest - t, length_fs, opacity)] };
            for w in pts.windows(2) {
                if w[0].1 != w[1].1 {
                    out.push(vertex(&w[0]));
                    out.push(vertex(&w[1]));
                }
            }
        }
        out
    }
}

/// Keeps the newest half of `a` and every other point of the older half.
fn thin_oldest_half(a: &mut Vec<Point>) {
    let half = a.len() / 2;
    let mut i = 0;
    a.retain(|_| {
        let keep = i >= half || i % 2 == 0;
        i += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nuc(z: u8, x: f64) -> SceneNucleus {
        SceneNucleus { z, a: None, pos: [x, 0.0, 0.0], vel: [0.0; 3] }
    }

    fn walk(tr: &mut Trails, from: u32, to: u32, length: f64) {
        for k in from..=to {
            tr.record(k as f64, &[[k as f64 * 0.01, 0.0, 0.0], [0.0, k as f64 * 0.01, 0.0]], length);
        }
    }

    #[test]
    fn trail_slider_maps_log_and_round_trips() {
        assert!((slider_to_length(0.0) - LENGTH_MIN_FS).abs() < 1e-12);
        assert!((slider_to_length(1.0) - LENGTH_MAX_FS).abs() < 1e-9);
        assert_eq!(length_to_slider(LENGTH_MIN_FS), 0.0);
        assert!((length_to_slider(LENGTH_MAX_FS) - 1.0).abs() < 1e-12);
        // logarithmic: the geometric middle of the range is the middle of the slider
        let mid = (LENGTH_MIN_FS * LENGTH_MAX_FS).sqrt();
        assert!((length_to_slider(mid) - 0.5).abs() < 1e-12);
        for fs in [5.0, 7.3, 20.0, 100.0, 333.0, 2500.0, 10_000.0] {
            assert!((slider_to_length(length_to_slider(fs)) - fs).abs() < 1e-9 * fs, "{fs}");
        }
        // out of range clamps
        assert_eq!(length_to_slider(1.0), 0.0);
        assert_eq!(length_to_slider(1e9), 1.0);
        assert!((slider_to_length(2.0) - LENGTH_MAX_FS).abs() < 1e-9);
        assert!((0.0..=1.0).contains(&length_to_slider(LENGTH_DEFAULT_FS)));
        assert_eq!(length_text(5.0), "5 fs");
        assert_eq!(length_text(100.0), "100 fs");
        assert_eq!(length_text(1500.0), "1.5 ps");
        assert_eq!(length_text(10_000.0), "10 ps");
        assert_eq!(parse_length("200"), Some(200.0));
        assert_eq!(parse_length(" 200 fs "), Some(200.0));
        assert_eq!(parse_length("1.5 ps"), Some(1500.0));
        assert_eq!(parse_length("abc"), None);
        assert_eq!(parse_length("-3"), None);
    }

    #[test]
    fn trail_alpha_fades_linearly_from_opacity_to_zero_at_the_length() {
        assert_eq!(alpha_at(0.0, 100.0, 0.7), 0.7);
        assert!((alpha_at(50.0, 100.0, 0.7) - 0.35).abs() < 1e-6);
        assert_eq!(alpha_at(100.0, 100.0, 0.7), 0.0);
        assert_eq!(alpha_at(150.0, 100.0, 0.7), 0.0);
        assert_eq!(alpha_at(-1.0, 100.0, 0.7), 0.7, "never brighter than the opacity");
        assert_eq!(alpha_at(10.0, 100.0, 0.0), 0.0);
        assert_eq!(alpha_at(0.0, 0.0, 0.7), 0.0);
    }

    #[test]
    fn trail_points_older_than_the_length_are_pruned() {
        let mut tr = Trails::new();
        walk(&mut tr, 0, 100, 30.0);
        // times 70..=100 stay (age <= 30)
        assert_eq!(tr.points(0), 31);
        assert_eq!(tr.points(1), 31);
        assert_eq!(tr.atoms[0][0].t, 70.0);
        // a repeat at the same time (paused) adds nothing
        tr.record(100.0, &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 30.0);
        assert_eq!(tr.points(0), 31);
        // a shorter length prunes at the next record and in the drawn lines at once
        let nuclei = [nuc(1, 1.0), nuc(1, 0.0)];
        let long = tr.lines(&nuclei, 30.0, 1.0).len();
        let short = tr.lines(&nuclei, 10.0, 1.0).len();
        assert!(short < long, "{short} < {long}");
        tr.record(101.0, &[[1.01, 0.0, 0.0], [0.0, 1.01, 0.0]], 10.0);
        assert_eq!(tr.points(0), 11);
        // the clock not moving (paused, a relax) moves the newest point instead of adding one
        tr.record(101.0, &[[2.0, 0.0, 0.0], [0.0, 1.01, 0.0]], 10.0);
        assert_eq!(tr.points(0), 11);
        assert_eq!(tr.atoms[0].last().unwrap().pos, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn trail_history_is_capped_and_thins_the_oldest_half() {
        let mut tr = Trails::new();
        let n = MAX_POINTS as u32 + 1;
        for k in 0..n {
            tr.record(k as f64, &[[k as f64, 0.0, 0.0]], 1e9);
        }
        let a = &tr.atoms[0];
        assert!(a.len() < MAX_POINTS && a.len() > MAX_POINTS / 2, "{}", a.len());
        // the newest half is untouched, the oldest half has every other point, order kept
        let half = (MAX_POINTS + 1) / 2;
        assert_eq!(a.len(), (MAX_POINTS + 1) - half / 2);
        assert_eq!(a[0].t, 0.0);
        assert_eq!(a[1].t, 2.0);
        assert!(a.windows(2).all(|w| w[0].t < w[1].t));
        assert_eq!(a.last().unwrap().t, (n - 1) as f64);
        // it keeps growing without ever exceeding the cap
        for k in n..n * 20 {
            tr.record(k as f64, &[[k as f64, 0.0, 0.0]], 1e9);
            assert!(tr.points(0) <= MAX_POINTS);
        }
        assert_eq!(tr.atoms[0].last().unwrap().t, (n * 20 - 1) as f64);
    }

    #[test]
    fn trail_lines_fade_in_the_element_colour_to_the_current_position() {
        let mut tr = Trails::new();
        for k in 0..=10 {
            tr.record(k as f64 * 10.0, &[[k as f64, 0.0, 0.0]], 100.0);
        }
        let v = tr.lines(&[nuc(8, 10.0)], 100.0, 0.7);
        // ten segments from x = 0 to x = 10, oldest first
        assert_eq!(v.len(), 20);
        assert_eq!(v[0].pos, [0.0; 3]);
        assert_eq!(v.last().unwrap().pos, [10.0, 0.0, 0.0]);
        let rgb = render::elements::colour(8);
        assert!(v.iter().all(|p| p.colour[..3] == rgb));
        // alpha: opacity at the newest end, 0 at age = length, half at half the length
        assert_eq!(v.last().unwrap().colour[3], 0.7);
        assert_eq!(v[0].colour[3], 0.0);
        let half = v.iter().find(|p| p.pos == [5.0, 0.0, 0.0]).unwrap();
        assert!((half.colour[3] - 0.35).abs() < 1e-6);
        // a sphere drawn between snapshots ends the line at the sphere, not at the newest point
        let v = tr.lines(&[nuc(8, 9.5)], 100.0, 0.7);
        assert_eq!(v.last().unwrap().pos, [9.5, 0.0, 0.0]);
        assert!(v.iter().all(|p| p.pos[0] <= 9.5));
        // a different count draws nothing
        assert!(tr.lines(&[nuc(8, 10.0), nuc(1, 0.0)], 100.0, 0.7).is_empty());
    }

    #[test]
    fn trail_breaks_on_teleport_count_change_and_time_going_backwards() {
        // a teleport clears that atom's history only (a drag keeps it)
        let mut tr = Trails::new();
        walk(&mut tr, 0, 20, 100.0);
        tr.on_command(&Command::Dyn(sim::DynCommand::DragEnd));
        assert_eq!((tr.points(0), tr.points(1)), (21, 21));
        tr.on_command(&Command::MoveNucleus { index: 0, pos: [5.0, 0.0, 0.0] });
        assert_eq!((tr.points(0), tr.points(1)), (0, 21));
        let v = tr.lines(&[nuc(1, 5.0), nuc(1, 0.0)], 100.0, 1.0);
        assert!(!v.is_empty() && v.iter().all(|p| p.pos[0] == 0.0), "only atom 1's trail, nothing joins the old and new places");
        // the first point after the teleport starts a new trail (same time: a paused teleport)
        tr.record(20.0, &[[5.0, 0.0, 0.0], [0.0, 0.2, 0.0]], 100.0);
        assert_eq!((tr.points(0), tr.points(1)), (1, 21));
        tr.record(21.0, &[[5.01, 0.0, 0.0], [0.0, 0.21, 0.0]], 100.0);
        assert_eq!(tr.points(0), 2);
        let v = tr.lines(&[nuc(1, 5.01), nuc(1, 0.0)], 100.0, 1.0);
        assert!(v.iter().all(|p| p.pos[0] == 0.0 || p.pos[0] >= 5.0));

        // the commands that change the scene clear everything
        let scene = crate::scene::Scene::preset("h2").unwrap();
        for cmd in [
            Command::SetScene(scene),
            Command::AddNucleus { z: 1, a: None, pos: [0.0; 3] },
            Command::RemoveNucleus(0),
            Command::SetCharge(1),
            Command::Dyn(sim::DynCommand::Relax),
        ] {
            let mut tr = Trails::new();
            walk(&mut tr, 0, 10, 100.0);
            tr.on_command(&cmd);
            assert!(tr.is_empty(), "{cmd:?}");
        }
        // time control does not
        let mut tr = Trails::new();
        walk(&mut tr, 0, 10, 100.0);
        tr.on_command(&Command::Dyn(sim::DynCommand::Pause));
        assert_eq!(tr.points(0), 11);

        // a snapshot with another nucleus count starts afresh
        tr.record(11.0, &[[0.0; 3]], 100.0);
        assert_eq!((tr.atoms.len(), tr.points(0)), (1, 1));
        // time going backwards (a rewind, a reset) clears the history
        let mut tr = Trails::new();
        walk(&mut tr, 0, 10, 100.0);
        tr.record(4.0, &[[0.0; 3], [0.0; 3]], 100.0);
        assert_eq!((tr.points(0), tr.points(1)), (1, 1));
        assert_eq!(tr.atoms[0][0].t, 4.0);
        assert!(tr.lines(&[nuc(1, 0.0), nuc(1, 0.0)], 100.0, 1.0).is_empty(), "one point and the atom's place: nothing to draw");
    }

    #[test]
    fn trail_collects_nothing_until_recorded_and_clear_forgets() {
        let tr = Trails::new();
        assert!(tr.is_empty());
        assert!(tr.lines(&[], 100.0, 0.7).is_empty());
        assert!(tr.lines(&[nuc(1, 0.0)], 100.0, 0.7).is_empty());
        let mut tr = Trails::new();
        walk(&mut tr, 0, 5, 100.0);
        assert!(!tr.is_empty());
        tr.clear();
        assert!(tr.is_empty());
        assert!(tr.lines(&[nuc(1, 0.0), nuc(1, 0.0)], 100.0, 0.7).is_empty());
    }
}
