//! When a recording takes its frames (display only: nothing here touches the run).
//!
//! **Simulated time** ([`SimClock`], the default): one frame per `slice` fs of simulated time, so a
//! slow DFT run (a few fs per wall second) becomes a smooth time-lapse and a fast estimated run is
//! evenly sampled. The viewer feeds every adopted playing snapshot ([`SimClock::push`]); a frame
//! due between two snapshots gets their positions interpolated linearly by simulated time (the
//! viewer's smooth motion does the same by wall time). A pause, a scene edit or a step counter that
//! went back breaks the interpolation ([`SimClock::pause`]): frames resume at the next snapshot,
//! never morphing across an edit. When the snapshots are farther apart than [`COARSE_SLICES`]
//! slices there is nothing honest to interpolate (the estimated solver at full speed publishes
//! every 10 000 steps): the frame is taken at the snapshot and counted as `coarse`. When the
//! renderer or the writer cannot keep up ([`SimClock::fell_behind`]) the slice doubles — the video
//! stays evenly spaced in simulated time, it just plays faster from there — and the caller says so.
//!
//! **As seen** ([`WallClock`]): one slot every 1/30 s of wall time; a slot no frame landed in (a
//! slow frame, a dropped write) is filled with the previous frame when the recording is finished,
//! so the video plays in real time.
use std::collections::VecDeque;
use std::time::Instant;

/// The video's frame rate (both modes).
pub const FPS: f64 = 30.0;
/// The default simulated time per frame: an O–H stretch (≈ 9 fs) gets ~18 frames.
pub const DEFAULT_FS_PER_FRAME: f64 = 0.5;
/// Snapshots farther apart than this many slices are not interpolated (one frame at the snapshot).
pub const COARSE_SLICES: f64 = 4.0;
/// Snapshots kept for interpolation; more pending means the renderer fell behind.
pub const HISTORY: usize = 64;
/// Float slack in comparisons of simulated times (fs).
const EPS: f64 = 1e-9;

/// One adopted playing snapshot: simulated time, step and the nuclei's positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub t_fs: f64,
    pub step: u64,
    pub pos: Vec<[f64; 3]>,
}

/// A frame the recorder wants rendered: its index in the video, its simulated time, the positions
/// and where it falls between the two snapshots around it (0 = the earlier, 1 = the later; the
/// density crossfade's weight).
#[derive(Clone, Debug, PartialEq)]
pub struct SimFrame {
    pub index: u64,
    pub t_fs: f64,
    pub pos: Vec<[f64; 3]>,
    pub alpha: f64,
}

#[derive(Debug)]
pub struct SimClock {
    /// Simulated fs per frame (doubles when the recording falls behind).
    pub slice: f64,
    /// The snapshots since the last break, oldest first.
    history: VecDeque<Sample>,
    /// The next frame's simulated time (`None` until the first snapshot).
    next: Option<f64>,
    /// The next frame's index.
    index: u64,
    /// Frames taken at a snapshot because the snapshots were too far apart to interpolate.
    pub coarse: u64,
    /// Times the slice was doubled.
    pub coarsened: u32,
    /// The simulated times of the first and the last frame taken.
    pub first_fs: Option<f64>,
    pub last_fs: Option<f64>,
}

impl SimClock {
    pub fn new(slice: f64) -> SimClock {
        SimClock { slice: slice.max(1e-3), history: VecDeque::new(), next: None, index: 0, coarse: 0, coarsened: 0, first_fs: None, last_fs: None }
    }

    /// Frames taken so far.
    pub fn frames(&self) -> u64 {
        self.index
    }

    /// A playing snapshot. One that does not continue the last (step not after it, another atom
    /// count) starts a new stretch: nothing is interpolated across it.
    pub fn push(&mut self, s: Sample) {
        let continues = self.history.back().is_some_and(|b| s.step > b.step && s.t_fs > b.t_fs && s.pos.len() == b.pos.len());
        if !continues {
            self.history.clear();
        }
        self.history.push_back(s);
        if self.history.len() > HISTORY {
            self.history.pop_front();
        }
    }

    /// Paused (or the scene was edited): the next snapshot starts a new stretch.
    pub fn pause(&mut self) {
        self.history.clear();
    }

    /// More than [`HISTORY`] / 2 snapshots wait: the renderer is behind the run.
    pub fn backlog(&self) -> bool {
        let Some(next) = self.next else { return false };
        self.history.iter().filter(|s| s.t_fs > next + EPS).count() > HISTORY / 2
    }

    /// The renderer or the writer could not keep up: twice the simulated time per frame from now on.
    pub fn fell_behind(&mut self) {
        self.slice *= 2.0;
        self.coarsened += 1;
    }

    /// The frames due now, at most `max` (the rest stay due for the next call).
    pub fn due(&mut self, max: usize) -> Vec<SimFrame> {
        let mut out = Vec::new();
        while out.len() < max {
            let (Some(first), Some(last)) = (self.history.front(), self.history.back()) else { break };
            let next = *self.next.get_or_insert(first.t_fs);
            if next > last.t_fs + EPS {
                break; // not there yet
            }
            // the snapshots around `next`: the last one at or before it and the one after
            let after = self.history.iter().position(|s| s.t_fs >= next - EPS).expect("the last one is at or after next");
            let frame = if after == 0 {
                // nothing before it in this stretch (the first snapshot, or after a break or a gap
                // the history no longer holds): the frame is the snapshot itself
                let s = &self.history[0];
                SimFrame { index: self.index, t_fs: s.t_fs, pos: s.pos.clone(), alpha: 1.0 }
            } else {
                let (a, b) = (&self.history[after - 1], &self.history[after]);
                let gap = b.t_fs - a.t_fs;
                if gap > COARSE_SLICES * self.slice {
                    self.coarse += 1;
                    SimFrame { index: self.index, t_fs: b.t_fs, pos: b.pos.clone(), alpha: 1.0 }
                } else {
                    let alpha = ((next - a.t_fs) / gap).clamp(0.0, 1.0);
                    let pos = a.pos.iter().zip(&b.pos).map(|(p, q)| std::array::from_fn(|k| p[k] + alpha * (q[k] - p[k]))).collect();
                    SimFrame { index: self.index, t_fs: next, pos, alpha }
                }
            };
            // the next frame one slice later — after this frame's time, which a snapshot frame
            // may have moved past `next`
            self.next = Some(if frame.t_fs > next + EPS { frame.t_fs + self.slice } else { next + self.slice });
            self.index += 1;
            self.first_fs.get_or_insert(frame.t_fs);
            self.last_fs = Some(frame.t_fs);
            // what is before the bracket is no longer needed
            let keep_from = after.saturating_sub(1);
            self.history.drain(..keep_from);
            out.push(frame);
        }
        out
    }

    /// The simulated time the frames cover.
    pub fn covered_fs(&self) -> f64 {
        match (self.first_fs, self.last_fs) {
            (Some(a), Some(b)) => b - a,
            _ => 0.0,
        }
    }
}

/// As seen: a slot every 1/[`FPS`] s of wall time since the first frame.
#[derive(Debug, Default)]
pub struct WallClock {
    start: Option<Instant>,
    next_slot: u64,
    /// Slots skipped (filled with the previous frame when the recording is finished).
    pub repeated: u64,
}

impl WallClock {
    /// The slot to capture now, if a new one began since the last capture.
    pub fn due(&mut self, now: Instant) -> Option<u64> {
        let start = *self.start.get_or_insert(now);
        let slot = (now.saturating_duration_since(start).as_secs_f64() * FPS).floor() as u64;
        if slot < self.next_slot {
            return None;
        }
        self.repeated += slot - self.next_slot;
        self.next_slot = slot + 1;
        Some(slot)
    }

    /// Slots so far (the video's length in frames).
    pub fn frames(&self) -> u64 {
        self.next_slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn s(t: f64, step: u64, x: f64) -> Sample {
        Sample { t_fs: t, step, pos: vec![[x, 0.0, 0.0]] }
    }

    #[test]
    fn frames_are_evenly_spaced_in_simulated_time_and_interpolated() {
        let mut c = SimClock::new(0.5);
        assert!(c.due(8).is_empty(), "nothing before the first snapshot");
        c.push(s(10.0, 40, 0.0));
        let f = c.due(8);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].index, f[0].t_fs, f[0].pos[0][0]), (0, 10.0, 0.0));
        assert!(c.due(8).is_empty(), "the next frame is at 10.5 fs");
        // a DFT step of 1.25 fs: frames at 10.5 and 11.0, interpolated
        c.push(s(11.25, 45, 1.25));
        let f = c.due(8);
        assert_eq!(f.iter().map(|f| f.t_fs).collect::<Vec<_>>(), vec![10.5, 11.0]);
        assert!((f[0].pos[0][0] - 0.5).abs() < 1e-12 && (f[1].pos[0][0] - 1.0).abs() < 1e-12);
        assert!((f[0].alpha - 0.4).abs() < 1e-12);
        assert_eq!(f[1].index, 2);
        assert_eq!((c.frames(), c.covered_fs()), (3, 1.0));
    }

    #[test]
    fn many_small_steps_give_one_frame_per_slice() {
        let mut c = SimClock::new(0.5);
        let mut n = 0;
        for k in 0..=100u64 {
            c.push(s(k as f64 * 0.1, k, k as f64));
            n += c.due(8).len();
        }
        // 0, 0.5, …, 10.0 fs
        assert_eq!(n, 21);
        assert!((c.covered_fs() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_cap_leaves_frames_due_for_the_next_call() {
        let mut c = SimClock::new(0.1);
        c.push(s(0.0, 0, 0.0));
        assert_eq!(c.due(8).len(), 1);
        c.push(s(0.35, 1, 3.5)); // gap 3.5 slices: interpolated, three frames due
        assert_eq!(c.due(2).len(), 2);
        let rest = c.due(2);
        assert_eq!(rest.len(), 1);
        assert!((rest[0].t_fs - 0.3).abs() < 1e-9 && (rest[0].pos[0][0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn far_apart_snapshots_are_taken_as_they_are() {
        let mut c = SimClock::new(0.5);
        c.push(s(0.0, 0, 0.0));
        c.due(8);
        c.push(s(500.0, 1000, 7.0)); // the estimated solver at full speed
        let f = c.due(8);
        assert_eq!(f.len(), 1, "one frame, not a thousand interpolated across 500 fs");
        assert_eq!((f[0].t_fs, f[0].pos[0][0], c.coarse), (500.0, 7.0, 1));
        c.push(s(1000.0, 2000, 8.0));
        assert_eq!(c.due(8).len(), 1);
    }

    #[test]
    fn a_pause_never_interpolates_across_an_edit() {
        let mut c = SimClock::new(0.5);
        c.push(s(0.0, 0, 0.0));
        c.push(s(0.25, 1, 0.25));
        assert_eq!(c.due(8).len(), 1);
        c.pause();
        // dragged while paused, then played: the next snapshot at 0.5 fs is far from the last
        c.push(s(0.5, 2, 9.0));
        let f = c.due(8);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].t_fs, f[0].pos[0][0]), (0.5, 9.0), "the snapshot, not a morph");
        // a replaced scene restarts the step counter: a break too
        c.push(s(0.0, 0, 5.0));
        assert!(c.due(8).is_empty(), "time went back: the next frame is still at 1.0 fs");
        c.push(s(1.0, 4, 6.0));
        assert_eq!(c.due(8).len(), 1);
    }

    #[test]
    fn falling_behind_doubles_the_slice() {
        let mut c = SimClock::new(0.5);
        c.push(s(0.0, 0, 0.0));
        c.due(8);
        for k in 1..=40u64 {
            c.push(s(k as f64 * 0.5, k, 0.0));
        }
        assert!(c.backlog(), "40 snapshots waiting");
        c.fell_behind();
        assert_eq!((c.slice, c.coarsened), (1.0, 1));
        let f = c.due(100);
        assert_eq!(f.first().map(|f| f.t_fs), Some(0.5), "the frame already due stays");
        assert!(f.windows(2).skip(1).all(|w| (w[1].t_fs - w[0].t_fs - 1.0).abs() < 1e-9), "then every 1 fs");
        assert!(!c.backlog());
    }

    #[test]
    fn the_wall_clock_counts_skipped_slots() {
        let mut w = WallClock::default();
        let t0 = Instant::now();
        assert_eq!(w.due(t0), Some(0));
        assert_eq!(w.due(t0 + Duration::from_millis(10)), None);
        assert_eq!(w.due(t0 + Duration::from_millis(40)), Some(1));
        assert_eq!(w.due(t0 + Duration::from_millis(140)), Some(4));
        assert_eq!((w.repeated, w.frames()), (2, 5));
    }
}
