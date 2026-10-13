//! `gpu::SurfaceState::resize` while another thread submits on the shared device.
//!
//! The viewer's sim worker and its render thread share one `wgpu::Device` (the GPU solver's
//! density texture is bound by the renderer without a copy). `Surface::configure` waits for the
//! device to go idle and then, in wgpu 30, reports a *validation* error — "Failed to wait for GPU
//! to come idle before reconfiguring the Surface" (`ConfigureSurfaceError::GpuWaitTimeout`, raised
//! when the queue is non-empty again after the wait, i.e. another thread submitted meanwhile) —
//! which wgpu's default handler turns into a panic. Seen once in the viewer ~2 s after launch with
//! the worker sampling, on the first Resized/Outdated event.
//!
//! This test recreates the race on purpose: a worker thread submits buffer copies back to back
//! while the test thread reconfigures a hidden window's surface 100 times. It passes when no
//! configure panics, every refused configure leaves `config` at the last accepted size, and a
//! later configure succeeds. Whether the race was actually hit is printed (`refused n`); on the
//! RX 7900 XTX about one in six is (4 MiB copies; 64 MiB copies refuse more but take a minute). Windows only: the event loop is built off the main thread
//! with `with_any_thread`. Skips without a GPU adapter.
#![cfg(target_os = "windows")]

use gpu::{GpuContext, SurfaceState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::Window;

#[test]
fn reconfigure_while_a_worker_submits_never_panics() {
    let event_loop = match EventLoop::builder().with_any_thread(true).build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipped: no event loop ({e})");
            return;
        }
    };
    let attrs = Window::default_attributes()
        .with_visible(false)
        .with_inner_size(PhysicalSize::new(640u32, 480u32));
    #[allow(deprecated)]
    let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
    let Some((ctx, surface)) = GpuContext::new_for_surface(wgpu::SurfaceTarget::from(window.clone())) else {
        eprintln!("skipped: no GPU adapter");
        return;
    };
    let mut surface = SurfaceState::new(&ctx, surface, 640, 480);

    // The stand-in for the sim worker: keeps the shared queue non-empty.
    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let ctx = ctx.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            let size = 1u64 << 20;
            let make = |usage, label| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label), size, usage, mapped_at_creation: false,
                })
            };
            let src = make(wgpu::BufferUsages::COPY_SRC, "race src");
            let dst = make(wgpu::BufferUsages::COPY_DST, "race dst");
            let mut submits = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(&src, 0, &dst, 0, size);
                let idx = ctx.queue.submit([enc.finish()]);
                submits += 1;
                // Like the real worker (a readback wait every step, CPU work between steps):
                // bounded in-flight work and short idle gaps. Unbounded back-to-back submission
                // makes every configure drain a long backlog (up to ~0.5 s each) and the run time
                // swing between 10 s and a minute.
                if submits % 8 == 0 {
                    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None });
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }
            submits
        })
    };

    let (mut accepted, mut refused) = (0u32, 0u32);
    let (mut t_ok, mut t_err) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let mut last_ok = (640u32, 480u32);
    for i in 0..100u32 {
        let want = if i % 2 == 0 { (800, 600) } else { (640, 480) };
        let t0 = std::time::Instant::now();
        let r = surface.resize(&ctx, want.0, want.1);
        let dt = t0.elapsed();
        match r {
            Ok(()) => {
                accepted += 1;
                t_ok += dt;
                last_ok = want;
            }
            Err(e) => {
                refused += 1;
                t_err += dt;
                let text = e.to_string();
                assert!(text.contains("GPU to come idle"), "unexpected configure error: {text}");
            }
        }
        assert_eq!((surface.config.width, surface.config.height), last_ok, "config must track the last accepted size");
    }
    stop.store(true, Ordering::Relaxed);
    let submits = worker.join().expect("worker");
    eprintln!(
        "surface reconfigure under load: accepted {accepted} ({:.1} ms each), refused {refused} ({:.1} ms each), worker submits {submits}",
        t_ok.as_secs_f64() * 1e3 / accepted.max(1) as f64,
        t_err.as_secs_f64() * 1e3 / refused.max(1) as f64
    );
    assert_eq!(accepted + refused, 100);
    // The viewer's retry: once the worker is quiet the same reconfigure goes through.
    let want = if last_ok == (640, 480) { (800, 600) } else { (640, 480) };
    surface.resize(&ctx, want.0, want.1).expect("reconfigure with the device quiet");
    assert_eq!((surface.config.width, surface.config.height), want);
}
