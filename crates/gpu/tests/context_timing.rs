//! Diagnostic (Task 9 item 5): what one headless `GpuContext` costs to create, and which adapter
//! it picks. `cargo test -p gpu --release --test context_timing -- --ignored --nocapture`.
use std::time::Instant;

#[test]
#[ignore = "diagnostic: timing of GpuContext::new_headless"]
fn diag_context_creation_time() {
    for i in 0..3 {
        let t = Instant::now();
        let Some(ctx) = gpu::GpuContext::new_headless() else {
            eprintln!("no adapter");
            return;
        };
        let info = ctx.adapter.get_info();
        eprintln!("context {i}: {:.0} ms — {} ({:?})", t.elapsed().as_secs_f64() * 1e3, info.name, info.backend);
    }
}
