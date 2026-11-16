//! TEMPORARY stage timers for the refit profile (branch refit-profile; measurement only, never
//! merged). A process-wide table of (calls, seconds) per named stage; [`span`] times a scope.
//! Nested spans each count their own wall (a parent includes its children).
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

static TABLE: Mutex<BTreeMap<&'static str, (u64, f64)>> = Mutex::new(BTreeMap::new());
static COUNTS: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

pub struct Span(&'static str, Instant);

impl Drop for Span {
    fn drop(&mut self) {
        let dt = self.1.elapsed().as_secs_f64();
        let mut t = TABLE.lock().unwrap();
        let e = t.entry(self.0).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += dt;
    }
}

/// Times the enclosing scope under `name`.
pub fn span(name: &'static str) -> Span {
    Span(name, Instant::now())
}

/// Counts an event (e.g. a Cholesky) under `name`, `n` times.
pub fn count(name: &'static str, n: u64) {
    *COUNTS.lock().unwrap().entry(name).or_insert(0) += n;
}

/// Clears the table.
pub fn reset() {
    TABLE.lock().unwrap().clear();
    COUNTS.lock().unwrap().clear();
}

/// The table: (stage, calls, seconds), and the counters.
pub fn take() -> (Vec<(&'static str, u64, f64)>, Vec<(&'static str, u64)>) {
    let t = std::mem::take(&mut *TABLE.lock().unwrap());
    let c = std::mem::take(&mut *COUNTS.lock().unwrap());
    (t.into_iter().map(|(k, (n, s))| (k, n, s)).collect(), c.into_iter().collect())
}
