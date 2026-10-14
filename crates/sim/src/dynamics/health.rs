//! The energy history and the NVE health number: the least-squares slope of the conserved
//! energy over the history tail (spec §7, "NVE drift per ps displayed as the primary health
//! indicator"). Atomic units in, Ha/ps out (the one unit conversion, at the readout).
use crate::provider::Source;
use crate::scene::neumaier;
use std::collections::VecDeque;

/// History tail length (steps) — the drift fit window and the sparkline.
pub const HISTORY_LEN: usize = 2000;
/// Fewer points than this give no drift estimate (`None`).
pub const DRIFT_MIN_POINTS: usize = 100;

/// One step's bookkeeping, recorded after the thermostat (spec §7 order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Point {
    /// Simulated time (atomic units).
    pub t: f64,
    /// E_total = K + E_KS (Hartree).
    pub e_total: f64,
    /// The invariant (`DynSnapshot::energy_conserved`, Hartree).
    pub e_conserved: f64,
    /// Kinetic temperature (K).
    pub temperature: f64,
    /// Who answered this step's evaluation (the sparkline colours each point by it).
    pub source: Source,
    /// The answer was a draft-mode guess (`EvalStats::extrapolation`).
    pub guess: bool,
}

/// The last ≤ [`HISTORY_LEN`] points, oldest first.
#[derive(Clone, Debug, Default)]
pub(crate) struct History {
    points: VecDeque<Point>,
}

impl History {
    pub fn push(&mut self, p: Point) {
        if self.points.len() == HISTORY_LEN {
            self.points.pop_front();
        }
        self.points.push_back(p);
    }

    pub fn clear(&mut self) {
        self.points.clear();
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `(t, E_total, T)` per point, oldest first — the snapshot's `history`.
    pub fn tail(&self) -> Vec<(f64, f64, f64)> {
        self.points.iter().map(|p| (p.t, p.e_total, p.temperature)).collect()
    }

    /// Each point's evaluation source, parallel to [`History::tail`] — the snapshot's
    /// `history_source`.
    pub fn sources(&self) -> Vec<Source> {
        self.points.iter().map(|p| p.source).collect()
    }

    /// Whether each point was a draft-mode guess, parallel to [`History::tail`] — the snapshot's
    /// `history_guess`.
    pub fn guesses(&self) -> Vec<bool> {
        self.points.iter().map(|p| p.guess).collect()
    }

    /// Least-squares slope of `e_conserved` against `t` over the whole tail, in Ha/ps; `None`
    /// below [`DRIFT_MIN_POINTS`] points (or for a degenerate time axis). Centred sums
    /// (compensated, CLAUDE.md rule 4): the drift is a tiny slope on an O(1) Ha offset, so the
    /// means are subtracted before any product is formed.
    pub fn drift_per_ps(&self) -> Option<f64> {
        let n = self.points.len();
        if n < DRIFT_MIN_POINTS {
            return None;
        }
        let tm = neumaier(self.points.iter().map(|p| p.t)) / n as f64;
        let em = neumaier(self.points.iter().map(|p| p.e_conserved)) / n as f64;
        let stt = neumaier(self.points.iter().map(|p| (p.t - tm) * (p.t - tm)));
        let ste = neumaier(self.points.iter().map(|p| (p.t - tm) * (p.e_conserved - em)));
        if stt.is_nan() || stt <= 0.0 {
            return None;
        }
        // Ha per atomic time unit → Ha/ps
        Some(ste / stt * units::fs_to_au_time(1000.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize, slope_au: f64, offset: f64) -> History {
        let mut h = History::default();
        for i in 0..n {
            let t = 10.0 * i as f64;
            h.push(Point { t, e_total: 0.0, e_conserved: offset + slope_au * t, temperature: 0.0, source: Source::Dft, guess: false });
        }
        h
    }

    #[test]
    fn slope_of_a_line_in_ha_per_ps_and_the_window() {
        // 1e-3 Ha/ps on a −76 Ha offset (an H₂O-sized energy)
        let per_au = 1e-3 / units::fs_to_au_time(1000.0);
        let h = line(500, per_au, -76.0);
        let d = h.drift_per_ps().unwrap();
        assert!((d - 1e-3).abs() < 1e-9, "{d}");
        assert_eq!(line(99, per_au, 0.0).drift_per_ps(), None);
        assert!(line(100, per_au, 0.0).drift_per_ps().is_some());
        let long = line(HISTORY_LEN + 37, per_au, 0.0);
        assert_eq!(long.len(), HISTORY_LEN);
        assert_eq!(long.tail()[0].0, 370.0, "oldest points dropped first");
        let mut c = long.clone();
        c.clear();
        assert_eq!(c.drift_per_ps(), None);
    }
}
