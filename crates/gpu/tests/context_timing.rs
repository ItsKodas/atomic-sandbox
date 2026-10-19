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

/// The card's memory as the GPU memory guard reads it (`GpuContext::device_local_memory_bytes`):
/// on a Vulkan adapter the largest device-local heap, which a discrete card has well above 1 GB;
/// no adapter, or another backend, says nothing.
#[test]
fn the_vulkan_adapter_reports_its_device_local_memory() {
    let Some(ctx) = gpu::GpuContext::new_headless() else {
        eprintln!("no adapter");
        return;
    };
    let info = ctx.adapter.get_info();
    let mem = ctx.device_local_memory_bytes();
    eprintln!("{} ({:?}, {:?}): device-local {:?} bytes", info.name, info.backend, info.device_type, mem);
    if info.backend == wgpu::Backend::Vulkan && info.device_type == wgpu::DeviceType::DiscreteGpu {
        assert!(mem.is_some_and(|b| b > 1 << 30), "{mem:?}");
    } else if info.backend != wgpu::Backend::Vulkan {
        assert_eq!(mem, None);
    }
}
