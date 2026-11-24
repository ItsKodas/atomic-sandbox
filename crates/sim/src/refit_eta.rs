//! How long a refit will take (2026-11-15, the user's idea: "track how long each refit takes to give
//! an actual estimate instead of just 'loading…'"). Bookkeeping only: nothing here reaches a fit or
//! a verdict. Since 2026-11-25 the display words come from the running stage's own counters
//! ([`crate::refit_live`], the refit picture: the spec has the estimate from the stage counters, not
//! from the last refits' walls); the history is still recorded and computed here.
//!
//! # The history
//!
//! Every refit that ends with a fit leaves a [`RefitRecord`]: its wall, the samples and species it
//! fitted, whether it ran in the background or in the tick, its kind ([`RefitKind`]: plain, or
//! with the trust recovery — the degree-cap retries and quarantine rounds, several fits and
//! leave-one-out passes each) and how many fits and leave-one-out passes it ran (its stages). The
//! provider keeps the newest [`REFIT_HISTORY_CAP`]; with a shared collection attached they are
//! the collection's `refit_times.csv` (`surrogate::Collection::save_refit_times`; text, one record per line — [`encode`] /
//! [`decode`]: a line that does not read is skipped, never fatal), in model-file mode the
//! session's (the probe's list, which outlives a provider).
//!
//! # The estimate ([`estimate`])
//!
//! The comparable records are those of the refit's species count (all of them when none has it)
//! and of its kind: the trust recovery once the running refit entered it, else the kind of the
//! latest comparable refit (a collection's misfit set persists from one refit to the next, so a
//! refit is most likely what the last one was); records of the other kind only when none has it.
//! The time scales with the sample count n as t ∝ n^p: **p = 1** (the fit's statistics, the
//! residual and leave-one-out passes are all linear in the samples at a fixed basis — the first
//! model), unless the comparable records span n by 1.5× or more over at least 3 refits and the
//! least-squares slope of ln t on ln n differs from 1 by more than [`EXPONENT_MARGIN`]: then that
//! slope clamped to [1, 2]. The estimate is the mean of the last ≤ [`ESTIMATE_FROM`] comparable
//! refits' walls, each scaled to n: t̂ = mean_i t_i (n / n_i)^p. The expected stage counts are the
//! medians of the same records'.
//!
//! # The countdown ([`eta`])
//!
//! Stages ([`RefitStages`]) count the fits and leave-one-out passes as they start; a stage's start
//! is the previous one's end, so at the k-th start k − 1 stages are done, at `at_s` seconds. Of the
//! expected stages S that is a fraction f = (k − 1)/S (below 0.95; a refit running more stages than
//! expected counts S = k), and it implies a total `at_s / f`. The total is blended,
//! T = (1 − f)·t̂ + f·at_s/f — the history early on, the stages as they accumulate — and the time
//! left is T − elapsed: between two stage starts it counts down second by second, it moves only
//! when a stage starts. Below one second, or past the estimate, it reads "finishing…", never a
//! negative time. Without a history there is no estimate ("estimating…").

/// The records kept (newest last).
pub const REFIT_HISTORY_CAP: usize = 50;
/// The estimate's comparable refits, at most.
pub const ESTIMATE_FROM: usize = 5;
/// How far from 1 the fitted exponent must be to replace it (module doc).
pub const EXPONENT_MARGIN: f64 = 0.25;
/// The stage fraction's ceiling (module doc).
const STAGE_FRACTION_MAX: f64 = 0.95;
/// Below this many seconds left the countdown reads "finishing…".
pub const FINISHING_S: f64 = 1.0;

const HEADER: &str = "# refit times v1: unix_s,wall_s,samples,species,origin,kind,fits,loo";

/// A refit with or without the trust recovery (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefitKind {
    Plain,
    Recovery,
}

/// One finished refit (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefitRecord {
    /// When it ended (seconds since the Unix epoch; 0 unknown).
    pub at_unix_s: u64,
    /// Its wall (seconds).
    pub wall_s: f64,
    /// Samples held when it started.
    pub samples: u32,
    /// Species of those samples.
    pub species: u32,
    /// Ran on its own thread (else in the tick).
    pub background: bool,
    pub kind: RefitKind,
    /// Fits it ran.
    pub fits: u32,
    /// Leave-one-out passes it ran.
    pub loo: u32,
}

/// The running refit's stages so far (module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefitStages {
    /// Fits started.
    pub fits: u32,
    /// Leave-one-out passes started.
    pub loo: u32,
    /// The trust recovery started.
    pub recovery: bool,
    /// Seconds from the refit's start to the latest stage start.
    pub at_s: f64,
}

/// A stage that starts ([`crate::ActivityProbe::refit_stage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefitStage {
    Fit,
    LeaveOneOut,
    Recovery,
}

/// What the history predicts for a refit (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefitEstimate {
    /// The predicted wall (seconds).
    pub total_s: f64,
    /// Comparable refits it is from.
    pub from: u32,
    /// The exponent p of t ∝ n^p.
    pub exponent: f64,
    /// The comparable refits' kind.
    pub kind: RefitKind,
    /// The expected fits and leave-one-out passes (medians).
    pub fits: u32,
    pub loo: u32,
}

/// The time left (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RefitEta {
    /// No history yet.
    Estimating,
    /// About `remaining_s` left; `fraction` of the estimated total elapsed (0–1).
    Left { remaining_s: f64, fraction: f64 },
    /// Under a second left, or past the estimate.
    Finishing,
}

impl RefitEta {
    /// The progress bar's fill (`None`: no estimate).
    pub fn fraction(&self) -> Option<f64> {
        match *self {
            RefitEta::Estimating => None,
            RefitEta::Left { fraction, .. } => Some(fraction.clamp(0.0, 1.0)),
            RefitEta::Finishing => Some(1.0),
        }
    }
}

/// Appends `r`, keeping the newest [`REFIT_HISTORY_CAP`].
pub fn push(history: &mut Vec<RefitRecord>, r: RefitRecord) {
    history.push(r);
    if history.len() > REFIT_HISTORY_CAP {
        let cut = history.len() - REFIT_HISTORY_CAP;
        history.drain(..cut);
    }
}

/// The history as the file's text (module doc): a header line, then one record per line.
pub fn encode(history: &[RefitRecord]) -> String {
    let mut s = String::from(HEADER);
    s.push('\n');
    for r in history {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            r.at_unix_s,
            r.wall_s,
            r.samples,
            r.species,
            if r.background { "background" } else { "tick" },
            match r.kind {
                RefitKind::Plain => "plain",
                RefitKind::Recovery => "recovery",
            },
            r.fits,
            r.loo
        ));
    }
    s
}

/// The records of a file's text, oldest first and at most the newest [`REFIT_HISTORY_CAP`], and
/// how many non-empty, non-comment lines did not read (skipped).
pub fn decode(text: &str) -> (Vec<RefitRecord>, usize) {
    let mut out = Vec::new();
    let mut bad = 0;
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        match parse_line(line) {
            Some(r) => push(&mut out, r),
            None => bad += 1,
        }
    }
    (out, bad)
}

fn parse_line(line: &str) -> Option<RefitRecord> {
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    let [at, wall, samples, species, origin, kind, fits, loo] = f.as_slice() else { return None };
    let wall_s: f64 = wall.parse().ok()?;
    if !(wall_s.is_finite() && wall_s >= 0.0) {
        return None;
    }
    Some(RefitRecord {
        at_unix_s: at.parse().ok()?,
        wall_s,
        samples: samples.parse().ok()?,
        species: species.parse().ok()?,
        background: match *origin {
            "background" => true,
            "tick" => false,
            _ => return None,
        },
        kind: match *kind {
            "plain" => RefitKind::Plain,
            "recovery" => RefitKind::Recovery,
            _ => return None,
        },
        fits: fits.parse().ok()?,
        loo: loo.parse().ok()?,
    })
}

/// The exponent p of t ∝ n^p over `records` (module doc): 1 unless they show a clearly different
/// one.
pub fn exponent(records: &[&RefitRecord]) -> f64 {
    let pts: Vec<(f64, f64)> = records.iter().filter(|r| r.samples > 0 && r.wall_s > 0.0).map(|r| ((r.samples as f64).ln(), r.wall_s.ln())).collect();
    if pts.len() < 3 {
        return 1.0;
    }
    let (lo, hi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.0), hi.max(p.0)));
    if hi - lo < 1.5f64.ln() {
        return 1.0;
    }
    let m = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / m, pts.iter().map(|p| p.1).sum::<f64>() / m);
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let p = sxy / sxx;
    if !p.is_finite() || (p - 1.0).abs() <= EXPONENT_MARGIN {
        1.0
    } else {
        p.clamp(1.0, 2.0)
    }
}

fn median(mut v: Vec<u32>) -> u32 {
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or(0)
}

/// The history's prediction for a refit of `samples` samples and `species` species (module doc);
/// `recovery`: the running refit entered the trust recovery. `None` without a history.
pub fn estimate(history: &[RefitRecord], samples: u32, species: u32, recovery: bool) -> Option<RefitEstimate> {
    let same_species: Vec<&RefitRecord> = history.iter().filter(|r| r.species == species).collect();
    let pool: Vec<&RefitRecord> = if same_species.is_empty() { history.iter().collect() } else { same_species };
    let latest = pool.last()?;
    let kind = if recovery { RefitKind::Recovery } else { latest.kind };
    let of_kind: Vec<&RefitRecord> = pool.iter().copied().filter(|r| r.kind == kind).collect();
    let of_kind = if !of_kind.is_empty() {
        of_kind
    } else {
        // none of that kind at this species count: that kind at any, else this count's any kind
        let any: Vec<&RefitRecord> = history.iter().filter(|r| r.kind == kind).collect();
        if any.is_empty() { pool } else { any }
    };
    let p = exponent(&of_kind);
    let recent = &of_kind[of_kind.len().saturating_sub(ESTIMATE_FROM)..];
    let n = samples.max(1) as f64;
    let total_s = recent.iter().map(|r| r.wall_s * (n / r.samples.max(1) as f64).powf(p)).sum::<f64>() / recent.len() as f64;
    Some(RefitEstimate {
        total_s,
        from: recent.len() as u32,
        exponent: p,
        kind: recent.last().map_or(kind, |r| r.kind),
        fits: median(recent.iter().map(|r| r.fits).collect()),
        loo: median(recent.iter().map(|r| r.loo).collect()),
    })
}

/// The time left after `elapsed_s` seconds (module doc).
pub fn eta(elapsed_s: f64, est: Option<&RefitEstimate>, stages: &RefitStages) -> RefitEta {
    let Some(e) = est else { return RefitEta::Estimating };
    let started = stages.fits + stages.loo;
    let done = started.saturating_sub(1);
    let expected = (e.fits + e.loo).max(started);
    let total = if done > 0 && expected > 0 && stages.at_s > 0.0 {
        let f = (done as f64 / expected as f64).min(STAGE_FRACTION_MAX);
        (1.0 - f) * e.total_s + stages.at_s
    } else {
        e.total_s
    };
    let remaining = total - elapsed_s;
    if !remaining.is_finite() || remaining < FINISHING_S {
        RefitEta::Finishing
    } else {
        RefitEta::Left { remaining_s: remaining, fraction: (elapsed_s / total).clamp(0.0, 1.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(samples: u32, wall_s: f64, kind: RefitKind) -> RefitRecord {
        RefitRecord { at_unix_s: 1_763_000_000, wall_s, samples, species: 3, background: true, kind, fits: 4, loo: 2 }
    }

    /// Linear scaling: t̂ = mean of t_i · n / n_i over the last five comparable refits.
    #[test]
    fn the_estimate_scales_the_last_five_comparable_refits_by_the_sample_count() {
        let mut h: Vec<RefitRecord> = (0..8).map(|k| rec(1000 + 10 * k, 100.0, RefitKind::Plain)).collect();
        // another species count and the other kind: not comparable
        h.push(RefitRecord { species: 2, wall_s: 1.0, ..rec(1000, 1.0, RefitKind::Plain) });
        let e = estimate(&h, 2000, 3, false).unwrap();
        assert_eq!((e.from, e.exponent, e.kind), (5, 1.0, RefitKind::Plain));
        let want = [1030.0, 1040.0, 1050.0, 1060.0, 1070.0].iter().map(|n: &f64| 100.0 * 2000.0 / n).sum::<f64>() / 5.0;
        assert!((e.total_s - want).abs() < 1e-9, "{} vs {want}", e.total_s);
        assert_eq!((e.fits, e.loo), (4, 2));
        // a species count no record has: all of them (the latest is the 2-species one)
        let e = estimate(&h, 1000, 4, false).unwrap();
        assert_eq!(e.from, 5);
        assert!(e.total_s < 100.0);
    }

    /// The kind: the latest comparable refit's, the recovery once the running refit entered it;
    /// the other kind only when none has it.
    #[test]
    fn the_kind_follows_the_latest_refit_and_the_recovery() {
        let h = vec![rec(1000, 150.0, RefitKind::Recovery), rec(1000, 10.0, RefitKind::Plain)];
        let plain = estimate(&h, 1000, 3, false).unwrap();
        assert_eq!((plain.kind, plain.total_s, plain.from), (RefitKind::Plain, 10.0, 1));
        let rec_ = estimate(&h, 1000, 3, true).unwrap();
        assert_eq!((rec_.kind, rec_.total_s), (RefitKind::Recovery, 150.0));
        let only_plain = vec![rec(1000, 10.0, RefitKind::Plain)];
        assert_eq!(estimate(&only_plain, 1000, 3, true).unwrap().total_s, 10.0, "no recovery on record: the plain ones");
    }

    #[test]
    fn no_history_no_estimate() {
        assert_eq!(estimate(&[], 2000, 3, false), None);
        assert_eq!(eta(5.0, None, &RefitStages::default()), RefitEta::Estimating);
        assert_eq!(RefitEta::Estimating.fraction(), None);
    }

    /// The exponent: 1 for a narrow spread or a slope near 1; a clearly different slope, clamped
    /// to [1, 2].
    #[test]
    fn the_exponent_is_fitted_only_when_clearly_different_and_clamped() {
        let rs = |p: f64, ns: &[u32]| ns.iter().map(|&n| rec(n, 1e-3 * (n as f64).powf(p), RefitKind::Plain)).collect::<Vec<_>>();
        fn refs(v: &[RefitRecord]) -> Vec<&RefitRecord> {
            v.iter().collect()
        }
        let wide = [500, 1000, 2000, 4000];
        assert_eq!(exponent(&refs(&rs(1.1, &wide))), 1.0, "within the margin");
        assert!((exponent(&refs(&rs(1.5, &wide))) - 1.5).abs() < 1e-9);
        assert_eq!(exponent(&refs(&rs(3.0, &wide))), 2.0, "clamped above");
        assert_eq!(exponent(&refs(&rs(0.2, &wide))), 1.0, "clamped below");
        assert_eq!(exponent(&refs(&rs(1.5, &[1000, 1100, 1200, 1300]))), 1.0, "a spread below 1.5×");
        assert_eq!(exponent(&refs(&rs(1.5, &[500, 4000]))), 1.0, "fewer than 3 refits");
        // used by the estimate: quadratic growth predicts 4× at 2× the samples
        let e = estimate(&rs(2.0, &[500, 1000, 2000, 4000]), 8000, 3, false).unwrap();
        assert_eq!(e.exponent, 2.0);
        assert!((e.total_s - 1e-3 * 8000f64.powi(2)).abs() < 1e-6 * e.total_s);
    }

    /// Append, the cap of 50 (newest kept), a corrupt line skipped, the round trip bit for bit.
    #[test]
    fn the_history_file_round_trips_caps_and_skips_a_corrupt_line() {
        let mut h = Vec::new();
        for k in 0..60u32 {
            push(&mut h, RefitRecord { at_unix_s: k as u64, wall_s: 0.1 + k as f64 / 3.0, samples: k, species: 1 + k % 3, background: k % 2 == 0, kind: if k % 4 == 0 { RefitKind::Recovery } else { RefitKind::Plain }, fits: k, loo: k / 2 });
        }
        assert_eq!(h.len(), REFIT_HISTORY_CAP);
        assert_eq!(h[0].at_unix_s, 10, "the oldest ten dropped");
        let text = encode(&h);
        let (back, bad) = decode(&text);
        assert_eq!((back.clone(), bad), (h.clone(), 0));
        // a corrupt line, a truncated one, a negative wall and an unknown kind: skipped
        let mut lines: Vec<&str> = text.lines().collect();
        lines.insert(3, "garbage");
        lines.insert(5, "12,3.5,100");
        lines.insert(7, "12,-3.5,100,2,tick,plain,1,1");
        lines.insert(9, "12,3.5,100,2,tick,weird,1,1");
        let (back, bad) = decode(&lines.join("\n"));
        assert_eq!((back, bad), (h.clone(), 4));
        // more than 50 lines in a file: the newest 50
        let mut long = h.clone();
        long.extend(h[..5].iter().map(|r| RefitRecord { at_unix_s: 1000 + r.at_unix_s, ..*r }));
        let (back, _) = decode(&format!("{HEADER}\n{}", encode(&long).lines().skip(1).collect::<Vec<_>>().join("\n")));
        assert_eq!(back.len(), REFIT_HISTORY_CAP);
        assert_eq!(back.last().unwrap().at_unix_s, 1014);
    }

    /// The countdown: from the history alone it counts down second by second, past the estimate
    /// it reads Finishing, never negative; the stages pull the total toward elapsed / fraction.
    #[test]
    fn the_countdown_never_goes_negative_and_the_stages_refine_it() {
        let e = RefitEstimate { total_s: 100.0, from: 3, exponent: 1.0, kind: RefitKind::Plain, fits: 6, loo: 4 };
        let none = RefitStages::default();
        assert_eq!(eta(0.0, Some(&e), &none), RefitEta::Left { remaining_s: 100.0, fraction: 0.0 });
        assert_eq!(eta(40.0, Some(&e), &none), RefitEta::Left { remaining_s: 60.0, fraction: 0.4 });
        assert_eq!(eta(99.5, Some(&e), &none), RefitEta::Finishing);
        assert_eq!(eta(500.0, Some(&e), &none), RefitEta::Finishing);
        assert_eq!(RefitEta::Finishing.fraction(), Some(1.0));
        // 5 of 10 stages done at 20 s (this refit runs fast): T = 0.5·100 + 20 = 70
        let st = RefitStages { fits: 4, loo: 2, recovery: false, at_s: 20.0 };
        assert_eq!(eta(30.0, Some(&e), &st), RefitEta::Left { remaining_s: 40.0, fraction: 30.0 / 70.0 });
        // between two stage starts it counts down second by second
        assert_eq!(eta(31.0, Some(&e), &st), RefitEta::Left { remaining_s: 39.0, fraction: 31.0 / 70.0 });
        // the first stage alone says nothing yet
        let first = RefitStages { fits: 1, loo: 0, recovery: false, at_s: 0.0 };
        assert_eq!(eta(10.0, Some(&e), &first), RefitEta::Left { remaining_s: 90.0, fraction: 0.1 });
        // more stages than expected: S = k, f below 0.95
        let over = RefitStages { fits: 30, loo: 10, recovery: true, at_s: 300.0 };
        match eta(301.0, Some(&e), &over) {
            RefitEta::Left { remaining_s, .. } => assert!(remaining_s > 0.0 && remaining_s <= 0.05 * 100.0 + 1e-9),
            RefitEta::Finishing => {}
            x => panic!("{x:?}"),
        }
        for t in [0.0, 1.0, 50.0, 99.0, 100.0, 1e4] {
            for s in [none, st, first, over] {
                if let RefitEta::Left { remaining_s, fraction } = eta(t, Some(&e), &s) {
                    assert!(remaining_s >= FINISHING_S && (0.0..=1.0).contains(&fraction));
                }
            }
        }
    }
}
