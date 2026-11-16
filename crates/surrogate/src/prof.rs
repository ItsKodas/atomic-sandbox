//! Stage timers for profiling a refit (the refit-profile measurement, 2026-11-17): spans in the
//! fit (`fit`, `loo`, `grade`, `novelty`) and in the provider's refit path, read by the
//! `#[ignore]`d `providers` harness `surrogate::tests::refit_profile`.
//!
//! **Compiled in only with the `profile` cargo feature** (off by default; `providers` forwards
//! it): `cargo test -p providers --release --features profile --lib refit_profile -- --ignored`.
//! Without it [`Span`] is zero-sized, [`span`] and [`count`] do nothing and [`take`] returns empty
//! tables, so a build carries no timing code. With it, a process-wide table of (calls, seconds)
//! per named stage; [`span`] times a scope, nested spans each count their own wall (a parent
//! includes its children), ~100 ns a span. Timing only — nothing here changes a number.

#[cfg(feature = "profile")]
mod imp {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::Instant;

    static TABLE: Mutex<BTreeMap<&'static str, (u64, f64)>> = Mutex::new(BTreeMap::new());
    static COUNTS: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

    /// A running span: its time is added to the table when dropped.
    pub struct Span(&'static str, Instant);

    impl Drop for Span {
        fn drop(&mut self) {
            let dt = self.1.elapsed().as_secs_f64();
            let mut t = TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let e = t.entry(self.0).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += dt;
        }
    }

    /// Times the enclosing scope under `name`.
    pub fn span(name: &'static str) -> Span {
        Span(name, Instant::now())
    }

    /// Counts an event under `name`, `n` times.
    pub fn count(name: &'static str, n: u64) {
        *COUNTS.lock().unwrap_or_else(|e| e.into_inner()).entry(name).or_insert(0) += n;
    }

    /// Clears the tables.
    pub fn reset() {
        TABLE.lock().unwrap_or_else(|e| e.into_inner()).clear();
        COUNTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// The tables: (stage, calls, seconds), and the counters; cleared.
    #[allow(clippy::type_complexity)]
    pub fn take() -> (Vec<(&'static str, u64, f64)>, Vec<(&'static str, u64)>) {
        let t = std::mem::take(&mut *TABLE.lock().unwrap_or_else(|e| e.into_inner()));
        let c = std::mem::take(&mut *COUNTS.lock().unwrap_or_else(|e| e.into_inner()));
        (t.into_iter().map(|(k, (n, s))| (k, n, s)).collect(), c.into_iter().collect())
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    /// Zero-sized without the `profile` feature (module doc).
    pub struct Span;

    /// A no-op without the `profile` feature.
    #[inline(always)]
    pub fn span(_name: &'static str) -> Span {
        Span
    }

    /// A no-op without the `profile` feature.
    #[inline(always)]
    pub fn count(_name: &'static str, _n: u64) {}

    /// A no-op without the `profile` feature.
    #[inline(always)]
    pub fn reset() {}

    /// Empty without the `profile` feature.
    #[allow(clippy::type_complexity)]
    pub fn take() -> (Vec<(&'static str, u64, f64)>, Vec<(&'static str, u64)>) {
        (Vec::new(), Vec::new())
    }
}

pub use imp::{count, reset, span, take, Span};

#[cfg(test)]
mod tests {
    /// Without the feature a span is zero-sized and the tables stay empty; with it they fill.
    #[test]
    fn spans_cost_nothing_without_the_profile_feature() {
        super::reset();
        {
            let _s = super::span("prof test");
            super::count("prof test", 2);
        }
        let (rows, counts) = super::take();
        if cfg!(feature = "profile") {
            assert!(rows.iter().any(|r| r.0 == "prof test" && r.1 == 1));
            assert!(counts.contains(&("prof test", 2)));
        } else {
            assert_eq!(std::mem::size_of::<super::Span>(), 0);
            assert!(rows.is_empty() && counts.is_empty());
        }
    }
}
