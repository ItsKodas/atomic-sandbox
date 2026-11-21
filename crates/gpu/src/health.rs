//! Whether the shared device can still be trusted (the gpu-fault fix, 2026-11-21).
//!
//! # Why
//! wgpu reports a failure in two ways, and before this module the app listened to neither:
//! - an **uncaptured error** — an out-of-memory on a buffer or texture creation, or any validation or internal error
//!   no error scope caught — goes to the device's uncaptured-error handler; wgpu 30's default handler panics
//!   ("wgpu error: …", `wgpu::backend::wgpu_core::default_error_handler`) on whichever thread made the call;
//! - a **lost device** (an out-of-memory inside a submit, a map or a descriptor allocation — wgpu-core's
//!   `Device::handle_hal_error` loses the device for those — or a driver reset) only calls the device-lost callback;
//!   every later error of the lost device is dropped silently (`ErrorType::DeviceLost => return`), and the next
//!   buffer created with `mapped_at_creation` panics on its invalid mapping (measured: a `GpuKs` evaluation on a
//!   destroyed device panics within 0.1–10 ms, before or during its solve).
//!
//! Either way nothing recorded that the device was gone, so every solver of the session — the stepping teacher, the
//! background auditor, the side solver, a sampling pass — kept using it, one fast failure (or one answer built from
//! invalid objects) after another.
//!
//! # What
//! [`GpuHealth`] is the session's record: the first failure, in words. [`GpuContext`](crate::GpuContext) installs the
//! handler and the callback when it creates the device (`GpuContext::request`); both only record (no panic: the
//! thread that hit the error carries on with invalid objects, and the solvers check the record before they trust
//! anything they computed — `providers::ks::GpuKs`). A failure is permanent: the device is not recovered, the app is
//! restarted (the user's message says so).
//!
//! The error scopes the viewer pushes (`SurfaceState::resize`, the pop-out and detached windows) still catch their
//! own validation errors first: wgpu hands an error to the uncaptured handler only when no scope of its kind is open
//! on the calling thread.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// The device's failure record (module doc). Shared by every clone of a [`GpuContext`](crate::GpuContext).
#[derive(Debug, Default)]
pub struct GpuHealth {
    failed: AtomicBool,
    /// The first failure, in words (later ones are dropped: the first is the cause).
    why: Mutex<Option<String>>,
}

impl GpuHealth {
    /// Records a failure; the first one's words are kept. Also the test hook for a device that is "marked lost".
    pub fn fail(&self, why: impl Into<String>) {
        let why = why.into();
        let mut slot = self.why.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            eprintln!("gpu: the device failed — {why}; every later solve on it is refused");
            *slot = Some(why);
        }
        self.failed.store(true, Ordering::Release);
    }

    /// The first recorded failure, or `None` while the device is trusted.
    pub fn failure(&self) -> Option<String> {
        if !self.failed.load(Ordering::Acquire) {
            return None;
        }
        Some(self.why.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| "the GPU failed".into()))
    }

    /// The words for an uncaptured wgpu error.
    pub fn describe_error(e: &wgpu::Error) -> String {
        match e {
            wgpu::Error::OutOfMemory { .. } => format!("{OUT_OF_MEMORY_WORDS} ({e})"),
            wgpu::Error::Validation { description, .. } => format!("{UNCAUGHT_ERROR_WORDS} ({})", first_line(description)),
            wgpu::Error::Internal { description, .. } => format!("{INTERNAL_ERROR_WORDS} ({})", first_line(description)),
        }
    }

    /// The words for a lost device.
    pub fn describe_loss(reason: wgpu::DeviceLostReason, message: &str) -> String {
        match (reason, message.trim()) {
            (wgpu::DeviceLostReason::Destroyed, "") => DESTROYED_WORDS.into(),
            (wgpu::DeviceLostReason::Destroyed, m) => format!("{DESTROYED_WORDS} ({m})"),
            (_, "") => format!("{LOST_WORDS} (a driver reset or out of memory)"),
            (_, m) => format!("{LOST_WORDS} ({m})"),
        }
    }
}

/// How a recorded failure began ([`GpuHealth::describe_error`] / [`GpuHealth::describe_loss`] start their words
/// with these), so the user's message can say what is known and no more ([`FailureKind::of`]).
pub const OUT_OF_MEMORY_WORDS: &str = "out of GPU memory";
/// A validation error no error scope caught: a bug in the app's GPU use, or a limit the device enforces.
pub const UNCAUGHT_ERROR_WORDS: &str = "a GPU error no scope caught";
/// wgpu's internal error.
pub const INTERNAL_ERROR_WORDS: &str = "an internal GPU error";
/// The device was lost (an out-of-memory inside a submit or a map, or a driver reset: wgpu does not say which).
pub const LOST_WORDS: &str = "the GPU device was lost";
/// The device was destroyed (`Device::destroy`: the app's own shutdown, or a test).
pub const DESTROYED_WORDS: &str = "the GPU device was destroyed";

/// What a recorded failure is known to be, read from its words (the gpu-fault review, 2026-11-21: the message must
/// not say "out of memory" for a validation error).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// wgpu reported an out-of-memory.
    OutOfMemory,
    /// The device was lost: a driver reset or an out-of-memory, not told apart.
    Lost,
    /// The device was destroyed.
    Destroyed,
    /// Another error (validation, internal) or words this module did not write: the cause is not known to be memory.
    Other,
}

impl FailureKind {
    /// The kind of a failure recorded with these words.
    pub fn of(why: &str) -> FailureKind {
        if why.starts_with(OUT_OF_MEMORY_WORDS) {
            FailureKind::OutOfMemory
        } else if why.starts_with(LOST_WORDS) {
            FailureKind::Lost
        } else if why.starts_with(DESTROYED_WORDS) {
            FailureKind::Destroyed
        } else {
            FailureKind::Other
        }
    }
}

/// The first non-empty line of a wgpu description, short enough for a status line.
fn first_line(s: &str) -> String {
    let line = s.lines().map(str::trim).find(|l| !l.is_empty() && *l != "Validation Error" && *l != "Caused by:").unwrap_or("");
    line.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuContext;

    /// Each description is read back as its own kind; anything else is `Other`.
    #[test]
    fn a_failure_is_read_back_as_its_kind() {
        assert_eq!(FailureKind::of(&GpuHealth::describe_loss(wgpu::DeviceLostReason::Destroyed, "")), FailureKind::Destroyed);
        assert_eq!(FailureKind::of(&GpuHealth::describe_loss(wgpu::DeviceLostReason::Unknown, "")), FailureKind::Lost);
        assert_eq!(FailureKind::of(&GpuHealth::describe_loss(wgpu::DeviceLostReason::Unknown, "reset")), FailureKind::Lost);
        assert_eq!(FailureKind::of(&format!("{OUT_OF_MEMORY_WORDS} (Out of Memory)")), FailureKind::OutOfMemory);
        assert_eq!(FailureKind::of(&format!("{UNCAUGHT_ERROR_WORDS} (Buffer size exceeds the limit)")), FailureKind::Other);
        assert_eq!(FailureKind::of(&format!("{INTERNAL_ERROR_WORDS} (x)")), FailureKind::Other);
        assert_eq!(FailureKind::of("injected: something"), FailureKind::Other);
    }

    #[test]
    fn the_first_failure_is_kept() {
        let h = GpuHealth::default();
        assert_eq!(h.failure(), None);
        h.fail("first");
        h.fail("second");
        assert_eq!(h.failure().as_deref(), Some("first"));
    }

    /// An uncaptured error (here a buffer over the device's size limit: a validation error, which takes the same
    /// path as an out-of-memory on creation) no longer panics: it is recorded on the context and every clone sees it.
    #[test]
    fn an_uncaptured_error_marks_the_context_failed_instead_of_panicking() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("skipped: no GPU adapter"); return };
        let clone = ctx.clone();
        assert_eq!(ctx.failure(), None);
        let too_big = ctx.device.limits().max_buffer_size + 4;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("too big"), size: too_big, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
        }));
        assert!(r.is_ok(), "the uncaptured error panicked");
        let why = clone.failure().expect("the error is recorded");
        assert!(why.contains("no scope caught"), "{why}");
    }

    /// A scope of the error's kind still catches it first: nothing is recorded.
    #[test]
    fn a_scoped_error_is_not_a_failure() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("skipped: no GPU adapter"); return };
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _b = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("too big"), size: ctx.device.limits().max_buffer_size + 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        assert!(pollster::block_on(scope.pop()).is_some());
        assert_eq!(ctx.failure(), None);
    }

    /// A lost device (`destroy`, the same `valid = false` wgpu-core's `lose` sets) is recorded once the device is
    /// polled — what `GpuContext::failure` does.
    #[test]
    fn a_lost_device_marks_the_context_failed() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("skipped: no GPU adapter"); return };
        ctx.device.destroy();
        let why = ctx.failure().expect("a destroyed device is a failure");
        assert!(why.contains("destroyed"), "{why}");
    }
}
