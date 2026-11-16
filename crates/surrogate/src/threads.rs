//! The fit's and the leave-one-out's worker threads (2026-11-15: a refit at ~2,000 samples kept
//! every hardware thread busy at normal priority for ~2.5 min and the user's PC became laggy).
//!
//! * **How many:** [`worker_threads`] = max(1, ⌊[`FIT_CPU_SHARE`] × available parallelism⌋), at
//!   most 16 — 12 of the user's 16 hardware threads, four left for the viewer, the worker and the
//!   rest of the machine.
//! * **At what priority:** every worker thread calls [`lower_priority`] first: below normal on
//!   Windows (`SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)`), a no-op
//!   elsewhere. The providers' background refit thread does the same.
//!
//! Neither changes a number: every threaded site (`Accumulator::add_many`'s design rows and Gram
//! rows, `batch_update`'s Gram rows, `loo::leave_one_out_many`) writes each output entry from one
//! thread, summing in sample order whatever the partition, so the thread count only changes the
//! wall (pinned by `fit::tests::the_fit_is_the_same_on_one_thread_and_on_the_capped_count` and
//! `loo::tests::leave_one_out_many_is_the_serial_loop_bit_for_bit`).

/// The share of the logical CPUs a fit or leave-one-out pass uses: the user's choice (2026-11-15).
pub const FIT_CPU_SHARE: f64 = 0.8;

/// The most worker threads a fit or leave-one-out pass uses.
pub const MAX_THREADS: usize = 16;

/// Worker threads for `available` hardware threads (module doc): max(1, ⌊0.8 × available⌋), ≤ 16.
pub fn threads_for(available: usize) -> usize {
    ((FIT_CPU_SHARE * available as f64).floor() as usize).clamp(1, MAX_THREADS)
}

/// Worker threads for the fit's and the leave-one-out's threaded parts (module doc).
pub fn worker_threads() -> usize {
    #[cfg(test)]
    if let Some(n) = OVERRIDE.with(|o| o.get()) {
        return n;
    }
    threads_for(std::thread::available_parallelism().map_or(1, |n| n.get()))
}

#[cfg(test)]
thread_local! {
    static OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Tests: runs `f` with [`worker_threads`] answering `n` on this thread.
#[cfg(test)]
pub(crate) fn with_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
    let old = OVERRIDE.with(|o| o.replace(Some(n.max(1))));
    let r = f();
    OVERRIDE.with(|o| o.set(old));
    r
}

/// Lowers the calling thread's OS scheduling priority to below normal (module doc); a no-op on
/// other systems than Windows, and when the call fails (the thread then runs at its priority).
pub fn lower_priority() {
    #[cfg(windows)]
    {
        type Handle = *mut core::ffi::c_void;
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> Handle;
            fn SetThreadPriority(thread: Handle, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        // SAFETY: GetCurrentThread returns a pseudo-handle for the calling thread (never closed);
        // SetThreadPriority only reads it
        unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
}

/// The calling thread's OS priority (Windows; `None` elsewhere): tests.
#[cfg(test)]
fn current_priority() -> Option<i32> {
    #[cfg(windows)]
    {
        type Handle = *mut core::ffi::c_void;
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> Handle;
            fn GetThreadPriority(thread: Handle) -> i32;
        }
        // SAFETY: as in `lower_priority`
        Some(unsafe { GetThreadPriority(GetCurrentThread()) })
    }
    #[cfg(not(windows))]
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eighty_percent_of_the_logical_cpus_at_least_one_at_most_16() {
        assert_eq!(threads_for(0), 1);
        assert_eq!(threads_for(1), 1);
        assert_eq!(threads_for(2), 1);
        assert_eq!(threads_for(4), 3);
        assert_eq!(threads_for(8), 6);
        assert_eq!(threads_for(16), 12, "the Ryzen 7 9800X3D: 16 hardware threads, 4 left free");
        assert_eq!(threads_for(20), 16);
        assert_eq!(threads_for(64), 16);
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(worker_threads(), threads_for(n));
        assert_eq!(with_threads(3, worker_threads), 3);
        assert_eq!(with_threads(0, worker_threads), 1);
        assert_eq!(worker_threads(), threads_for(n), "the override ends with the closure");
    }

    #[test]
    fn a_worker_thread_runs_below_normal() {
        let (before, after) = std::thread::spawn(|| {
            let before = current_priority();
            lower_priority();
            (before, current_priority())
        })
        .join()
        .unwrap();
        if cfg!(windows) {
            assert_eq!(before, Some(0), "THREAD_PRIORITY_NORMAL");
            assert_eq!(after, Some(-1), "THREAD_PRIORITY_BELOW_NORMAL");
        } else {
            assert_eq!((before, after), (None, None));
        }
    }
}
