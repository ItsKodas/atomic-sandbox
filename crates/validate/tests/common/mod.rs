//! Helpers shared by the GPU test binaries (`gpu_vs_cpu.rs`, `dynamics.rs`). Each integration
//! test is its own crate, so this is pulled in with `mod common;` and every item is allowed to be
//! unused by one of them.
#![allow(dead_code)]
use gpu::GpuContext;
use std::sync::Mutex;
use std::time::Duration;

/// Serialises the GPU sections of concurrently running tests within one binary (poisoning is
/// harmless here). Distinct test binaries never run at the same time under `cargo test`.
pub static GPU: Mutex<()> = Mutex::new(());

/// A headless device, or `None` (with a note naming the suite) when there is no adapter.
pub fn gpu(suite: &str) -> Option<GpuContext> {
    let ctx = GpuContext::new_headless();
    if ctx.is_none() { eprintln!("{suite}: no GPU adapter — skipped"); }
    ctx
}

pub fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

/// Today's date from the system clock without a date crate: days since the Unix epoch →
/// civil date (Howard Hinnant's algorithm), UTC.
pub fn chrono_free_date() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
