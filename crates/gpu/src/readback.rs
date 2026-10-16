//! GPU → CPU buffer readbacks that fail with an error instead of a panic.
//!
//! A readback is `map_async`, a blocking `Device::poll`, and a wait for the map callback's
//! result. On a device shared between threads (the viewer's render thread, the sim worker's
//! solver, other processes' work on the same card) any of the three can fail: the poll can time
//! out or report a lost device, and the callback's sender can be dropped without ever being
//! called (the channel's `RecvError` that killed the user's window on 2026-10-17, while the
//! surface was being reconfigured under a running solve). Those are conditions to report and
//! retry, not invariants, so [`read_buffer`] returns [`ReadbackError`] and every caller either
//! passes it on (`Result`) or, where the signature is infallible by construction (the solver's
//! reductions), latches it (`qm_gpu::grid::take_readback_fault`) for the solver to raise at its
//! next check.
//!
//! A failed readback leaves the buffer unmapped (`Buffer::unmap` cancels a map that is still
//! pending), so a persistent staging buffer can be reused by the next attempt.
use crate::GpuContext;
use std::cell::Cell;
use std::ops::Range;
use std::sync::mpsc::Receiver;

/// Why a readback produced no data.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReadbackError {
    /// `Device::poll` failed (a timeout under contention, or a lost device).
    #[error("waiting for the GPU failed ({0})")]
    Poll(String),
    /// The map callback never reported: its sender was dropped (`RecvError`).
    #[error("the GPU never answered the buffer map (callback dropped)")]
    CallbackDropped,
    /// The map itself failed (`BufferAsyncError`), or the mapped range could not be taken.
    #[error("mapping the buffer failed ({0})")]
    Map(String),
}

thread_local! {
    /// Test hook (see [`fail_readbacks_after`]): (readbacks to let through, readbacks to fail).
    static INJECT: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

/// Test hook: on the calling thread, let `skip` readbacks through, then make the next `count`
/// fail exactly as a dropped map-callback sender does ([`ReadbackError::CallbackDropped`]). It
/// drives the real code paths (the same [`finish_map`] a real failure reaches); there is no
/// other way to provoke a `RecvError` on demand. Per thread, so parallel tests do not see each
/// other's injections; `fail_readbacks_after(0, 0)` clears it.
#[doc(hidden)]
pub fn fail_readbacks_after(skip: usize, count: usize) {
    INJECT.with(|c| c.set((skip, count)));
}

/// True when the calling thread's injected failure (if any) is due for this readback.
fn injected() -> bool {
    INJECT.with(|c| {
        let (skip, count) = c.get();
        if count == 0 { return false; }
        if skip > 0 { c.set((skip - 1, count)); return false; }
        c.set((0, count - 1));
        true
    })
}

/// The tail of a readback: the poll's outcome, then the callback's. Separate from the wgpu calls
/// so a dropped sender can be tested without a GPU.
pub fn finish_map(poll: Result<(), String>, rx: &Receiver<Result<(), wgpu::BufferAsyncError>>) -> Result<(), ReadbackError> {
    poll.map_err(ReadbackError::Poll)?;
    match rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(ReadbackError::Map(e.to_string())),
        Err(_) => Err(ReadbackError::CallbackDropped),
    }
}

/// Maps `range` of `buffer` (usage `MAP_READ`, the copy into it already submitted), waits for the
/// GPU, hands the bytes to `f` and unmaps — also on every error path.
pub fn read_buffer<T>(ctx: &GpuContext, buffer: &wgpu::Buffer, range: Range<u64>, f: impl FnOnce(&[u8]) -> T) -> Result<T, ReadbackError> {
    let slice = buffer.slice(range);
    let (tx, rx) = std::sync::mpsc::channel();
    let inject = injected();
    slice.map_async(wgpu::MapMode::Read, move |r| { if !inject { let _ = tx.send(r); } });
    let poll = ctx.device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| e.to_string());
    let waited = finish_map(poll, &rx);
    let out = waited.and_then(|()| {
        let view = slice.get_mapped_range().map_err(|e| ReadbackError::Map(e.to_string()))?;
        let v = f(&view);
        drop(view);
        Ok(v)
    });
    buffer.unmap(); // a no-op for an unmapped buffer; cancels a map still pending after a failure
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dropped_sender_is_an_error_not_a_panic() {
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        drop(tx);
        assert_eq!(finish_map(Ok(()), &rx), Err(ReadbackError::CallbackDropped));
    }

    #[test]
    fn a_failed_poll_is_reported_before_the_callback_is_awaited() {
        // the sender is alive and silent: waiting on it would hang, so the poll error must win
        let (_tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        assert_eq!(finish_map(Err("timeout".into()), &rx), Err(ReadbackError::Poll("timeout".into())));
    }

    #[test]
    fn a_delivered_map_is_ok() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok(())).unwrap();
        assert_eq!(finish_map(Ok(()), &rx), Ok(()));
    }

    #[test]
    fn injection_skips_then_fails_then_clears() {
        fail_readbacks_after(1, 2);
        assert_eq!([injected(), injected(), injected(), injected()], [false, true, true, false]);
        fail_readbacks_after(0, 0);
        assert!(!injected());
    }

    #[test]
    fn an_injected_failure_reaches_a_real_readback_as_an_error() {
        let Some(ctx) = GpuContext::new_headless() else { eprintln!("skipped: no GPU adapter"); return };
        let buf = |ctx: &GpuContext| ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("t"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let b = buf(&ctx);
        fail_readbacks_after(0, 1);
        assert_eq!(read_buffer(&ctx, &b, 0..16, |x| x.len()), Err(ReadbackError::CallbackDropped));
        // the same buffer is usable by the next attempt
        assert_eq!(read_buffer(&ctx, &b, 0..16, |x| x.len()), Ok(16));
    }
}
