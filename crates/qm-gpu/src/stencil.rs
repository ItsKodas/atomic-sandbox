//! 8th-order Laplacian / kinetic operator on padded fields — the GPU counterpart of
//! `qm::stencil::{laplacian, kinetic}` (same coefficients, Dirichlet f = 0 outside the box).
//!
//! The kernel (`shaders/stencil.wgsl`) runs one thread per interior point in (8, 8, 4)
//! workgroups and reads the ±4 neighbours along each axis straight from the padded buffer: the
//! zero ghost cells are the Dirichlet boundary, so there are no bounds checks in the stencil.
//! Only the interior of `out` is written (its ghosts stay 0). Laplacian and kinetic are the same
//! entry point with a different final scale (1/h² vs −1/(2h²)), so `kinetic` is bitwise
//! −½·`laplacian`. `*_rec` forms record into a caller-owned encoder (see `kernels::encoder`).
use crate::grid::{Field, GpuGrid};
use crate::kernels::{immediate, uniform, Kernels};
use crate::pointwise::{check, no_alias};
use gpu::GpuContext;
use wgpu::CommandEncoder;

/// Workgroup shape of the stencil kernel (must match `@workgroup_size` in stencil.wgsl).
pub const STENCIL_WORKGROUP: [u32; 3] = [8, 8, 4];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct StencilParams { scale: f32, p: u32, n: u32, zero: u32 /* must be 0: the shader's reassociation barrier */ }

/// 1/h² in f32.
fn inv_h2(g: &GpuGrid) -> f32 { (1.0 / (g.h * g.h)) as f32 }

#[allow(clippy::too_many_arguments)]
fn stencil_rec(enc: &mut CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, op: &str, f: &Field, out: &Field, scale: f32) {
    check(g, &[f, out]);
    no_alias(op, out, f);
    let u = uniform(ctx, op, &StencilParams { scale, p: g.p as u32, n: g.n as u32, zero: 0 });
    let bg = k.bind_group(ctx, "laplacian", &[u.as_entire_binding(), out.buffer.as_entire_binding(), f.buffer.as_entire_binding()]);
    let n = g.n as u32;
    let [wx, wy, wz] = STENCIL_WORKGROUP;
    k.record_3d(enc, "laplacian", &bg, [n.div_ceil(wx), n.div_ceil(wy), n.div_ceil(wz)]);
}

/// out = ∇²f on the interior (recorded). Ghosts of `f` must be 0 (they are the Dirichlet wall).
pub fn laplacian_rec(enc: &mut CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    stencil_rec(enc, ctx, k, g, "laplacian", f, out, inv_h2(g));
}

/// out = ∇²f on the interior; reads the zero ghosts → Dirichlet for free.
pub fn laplacian(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    immediate(ctx, "laplacian", |enc| laplacian_rec(enc, ctx, k, g, f, out));
}

/// out = −½∇²f (recorded). Scale −½·(1/h²) is exact in f32, so this is bitwise −½·laplacian.
pub fn kinetic_rec(enc: &mut CommandEncoder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    stencil_rec(enc, ctx, k, g, "kinetic", f, out, -0.5 * inv_h2(g));
}

/// out = −½∇²f (kinetic-energy operator in atomic units).
pub fn kinetic(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    immediate(ctx, "kinetic", |enc| kinetic_rec(enc, ctx, k, g, f, out));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};
    use crate::kernels::{encoder, Kernels};
    use crate::pointwise::{axpy, axpy_rec};

    fn gaussian(cg: &qm::Grid) -> Vec<f64> { cg.sample(|[x, y, z]| (-(x * x + y * y + z * z) / 2.0).exp()) }

    fn upload(ctx: &GpuContext, g: &GpuGrid, data: &[f64], label: &str) -> Field {
        let f = Field::new(ctx, g, label);
        f.upload_interior(ctx, g, data);
        f
    }

    fn cpu_lap(cg: &qm::Grid, f: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; cg.len()];
        qm::stencil::laplacian(cg, f, &mut out);
        out
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn laplacian_matches_cpu_stencil() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(33, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let f = gaussian(&cg);
        let (ff, out) = (upload(&ctx, &g, &f, "f"), Field::new(&ctx, &g, "lap"));
        laplacian(&ctx, &k, &g, &ff, &out);
        let gpu = out.read_interior(&ctx, &g);
        // end to end: f64 input, f64 CPU stencil
        let err = max_abs_diff(&gpu, &cpu_lap(&cg, &f));
        // kernel arithmetic only: the CPU stencil applied to the f32-rounded field the GPU sees
        let f32_in: Vec<f64> = f.iter().map(|v| *v as f32 as f64).collect();
        let err_kernel = max_abs_diff(&gpu, &cpu_lap(&cg, &f32_in));
        println!("laplacian 33³ h=0.25: max |gpu − cpu| = {err:.2e} (f64 input), {err_kernel:.2e} (f32-rounded input)");
        // Achieved (RX 7900 XTX): 5.8e-6 end to end — the f64→f32 rounding of the input (≤ 3e-8·f)
        // amplified by Σ|c|/h² ≈ 300; inherent to f32 storage — and 1.6e-7 from the kernel's own
        // arithmetic (5e-8 relative to max|∇²f| ≈ 3). The kernel bound is tightened 1e-5 → 1e-6:
        // without the reassociation barrier in stencil.wgsl the driver's fast math gives 1.2e-5.
        assert!(err < 1e-5, "max abs diff {err}");
        assert!(err_kernel < 1e-6, "max abs diff (f32-rounded input) {err_kernel}");
    }

    #[test]
    fn dirichlet_boundary_via_zero_ghosts() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(13, 0.5);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let ones = vec![1.0; cg.len()];
        let (ff, out) = (upload(&ctx, &g, &ones, "ones"), Field::new(&ctx, &g, "lap"));
        laplacian(&ctx, &k, &g, &ff, &out);
        let gpu = out.read_interior(&ctx, &g);
        let cpu = cpu_lap(&cg, &ones);
        let scale = qm::stencil::COEFF[0].abs() * 3.0 / (cg.h * cg.h);
        // every point: corners (three faces cut), edges, faces, and the deep interior (exactly 0)
        for kk in 0..cg.n[2] { for j in 0..cg.n[1] { for i in 0..cg.n[0] {
            let id = cg.idx(i, j, kk);
            assert!((gpu[id] - cpu[id]).abs() < 1e-6 * scale, "({i},{j},{kk}): gpu {} cpu {}", gpu[id], cpu[id]);
        }}}
        let corner = cg.idx(0, 0, 0);
        assert!(cpu[corner] < -1.0, "corner sees the Dirichlet wall: {}", cpu[corner]);
        assert_eq!(gpu[cg.idx(6, 6, 6)], 0.0, "constant field away from the walls has zero Laplacian");
    }

    #[test]
    fn kinetic_is_minus_half_laplacian() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(21, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let f = cg.sample(|[x, y, z]| (-(x * x + 0.5 * y * y + 2.0 * z * z)).exp() * (1.0 + 0.3 * x));
        let ff = upload(&ctx, &g, &f, "f");
        let (l, t) = (Field::new(&ctx, &g, "l"), Field::new(&ctx, &g, "t"));
        laplacian(&ctx, &k, &g, &ff, &l);
        kinetic(&ctx, &k, &g, &ff, &t);
        let (lr, tr) = (l.read_padded_raw(&ctx), t.read_padded_raw(&ctx));
        assert!(lr.iter().any(|v| *v != 0.0));
        // interior only: ghosts are never written, so both hold +0 there (and −½·(+0) is −0)
        for kk in 0..g.n { for j in 0..g.n { for i in 0..g.n {
            let q = g.pidx(i, j, kk);
            assert_eq!(tr[q].to_bits(), (-0.5 * lr[q]).to_bits(), "({i},{j},{kk}): t {} vs −½·l {}", tr[q], -0.5 * lr[q]);
        }}}
    }

    #[test]
    fn ghosts_stay_zero_after_stencil() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.4);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let f = cg.sample(|[x, y, z]| 1.0 + x - 0.5 * y * z); // non-zero right up to the walls
        let ff = upload(&ctx, &g, &f, "f");
        let out = Field::new(&ctx, &g, "out");
        laplacian(&ctx, &k, &g, &ff, &out);
        assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
        kinetic(&ctx, &k, &g, &ff, &out);
        assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
        assert_ghosts_zero(&g, &ff.read_padded_raw(&ctx)); // the input is untouched too
        assert_eq!(ff.read_interior(&ctx, &g).iter().zip(&f).filter(|(a, b)| (*a - *b).abs() > 1e-6).count(), 0);
    }

    #[test]
    #[should_panic(expected = "alias")]
    fn laplacian_rejects_aliasing() {
        let Some(ctx) = GpuContext::new_headless() else { panic!("alias (no adapter; forced)") };
        let g = GpuGrid::new(&qm::Grid::cube(5, 1.0));
        let k = Kernels::new(&ctx);
        let f = Field::new(&ctx, &g, "f");
        laplacian(&ctx, &k, &g, &f, &f);
    }

    /// Timing probe for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture bench_97_stencil`.
    /// Per-op cost of 25×(laplacian + axpy) = 50 ops, immediate (one submit per op) vs batched
    /// (one encoder, one submit), and the GPU time of the stencil kernel itself.
    #[test]
    #[ignore]
    fn bench_97_stencil_batched_vs_immediate() {
        use std::time::Instant;
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(97, 0.2);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let x0 = gaussian(&cg);
        let (x, l) = (upload(&ctx, &g, &x0, "x"), Field::new(&ctx, &g, "l"));
        let sync = || { ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll"); };
        let tau = 1e-4f32;
        let (rounds, ops) = (20, 50);
        let per_op = |t: Instant| t.elapsed().as_secs_f64() * 1e3 / (rounds * ops) as f64;

        let immediate_round = || { for _ in 0..ops / 2 { laplacian(&ctx, &k, &g, &x, &l); axpy(&ctx, &k, &g, &x, tau, &l); } };
        let batched_round = || {
            let mut enc = encoder(&ctx, "bench");
            for _ in 0..ops / 2 { laplacian_rec(&mut enc, &ctx, &k, &g, &x, &l); axpy_rec(&mut enc, &ctx, &k, &g, &x, tau, &l); }
            ctx.queue.submit([enc.finish()]);
        };
        for _ in 0..3 { immediate_round(); batched_round(); }
        sync();

        let t = Instant::now();
        for _ in 0..rounds { immediate_round(); sync(); }
        let imm = per_op(t);
        let t = Instant::now();
        for _ in 0..rounds { batched_round(); sync(); }
        let bat = per_op(t);

        // GPU time per kernel: record 200, then time submit → idle
        let gpu_time = |name: &str, rec: &dyn Fn(&mut CommandEncoder)| {
            let reps = 200;
            let mut enc = encoder(&ctx, name);
            let t = Instant::now();
            for _ in 0..reps { rec(&mut enc); }
            let cb = enc.finish();
            let record_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let t = Instant::now();
            ctx.queue.submit([cb]);
            sync();
            let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!("97³ {name}: {gpu_ms:.3} ms per dispatch submit→idle, {record_us:.1} µs CPU to record+finish");
        };
        gpu_time("laplacian", &|enc| laplacian_rec(enc, &ctx, &k, &g, &x, &l));
        gpu_time("axpy", &|enc| axpy_rec(enc, &ctx, &k, &g, &l, tau, &x));
        { // reference for later tasks: 200 laplacian dispatches in ONE compute pass (shared uniform,
          // bind group per dispatch) — isolates the per-pass cost of the one-pass-per-op `*_rec` API
            let reps = 200;
            let u = crate::kernels::uniform(&ctx, "u", &StencilParams { scale: 1.0, p: g.p as u32, n: g.n as u32, zero: 0 });
            let t = Instant::now();
            let mut enc = encoder(&ctx, "one pass");
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(k.pipeline("laplacian"));
                for _ in 0..reps {
                    let bg = k.bind_group(&ctx, "laplacian", &[u.as_entire_binding(), l.buffer.as_entire_binding(), x.buffer.as_entire_binding()]);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(13, 13, 25);
                }
            }
            let cb = enc.finish();
            let rec_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let t = Instant::now();
            ctx.queue.submit([cb]);
            sync();
            println!("97³ laplacian, one shared pass: {:.3} ms per dispatch submit→idle, {rec_us:.1} µs CPU to record+finish", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }

        let mut cpu_out = vec![0.0; cg.len()];
        let t = Instant::now();
        qm::stencil::laplacian(&cg, &x0, &mut cpu_out);
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        println!("97³ per op (25×(laplacian+axpy), wall incl. sync): immediate {imm:.3} ms, batched {bat:.3} ms ({:.1}× faster); CPU f64 laplacian {cpu_ms:.1} ms", imm / bat);
    }

    /// 25 explicit heat-equation steps x ← x + τ∇²x = 50 dispatches (stencil, axpy, stencil, …)
    /// recorded into ONE encoder and submitted once. Each step reads what the previous one
    /// wrote, so any missing ordering between recorded passes shows up as a mismatch vs the CPU.
    #[test]
    fn batched_encoder_matches_cpu_and_immediate() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(25, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let x0 = gaussian(&cg);
        let steps = 25;
        // a different τ per step, so every recorded dispatch needs its own uniform;
        // explicit stability needs τ ≲ h²/9 for this stencil
        let taus: Vec<f32> = (0..steps).map(|s| ((0.02 + 0.003 * s as f64) * cg.h * cg.h) as f32).collect();

        let mut cpu = x0.clone();
        for &tau in &taus {
            let l = cpu_lap(&cg, &cpu);
            for (c, d) in cpu.iter_mut().zip(&l) { *c += tau as f64 * d; }
        }

        let (x, l) = (upload(&ctx, &g, &x0, "x"), Field::new(&ctx, &g, "l"));
        let mut enc = encoder(&ctx, "batched heat steps");
        for &tau in &taus {
            laplacian_rec(&mut enc, &ctx, &k, &g, &x, &l);
            axpy_rec(&mut enc, &ctx, &k, &g, &x, tau, &l);
        }
        ctx.queue.submit([enc.finish()]);
        let batched = x.read_interior(&ctx, &g);

        let (xi, li) = (upload(&ctx, &g, &x0, "xi"), Field::new(&ctx, &g, "li"));
        for &tau in &taus {
            laplacian(&ctx, &k, &g, &xi, &li);
            axpy(&ctx, &k, &g, &xi, tau, &li);
        }
        let immediate = xi.read_interior(&ctx, &g);

        let err = max_abs_diff(&batched, &cpu);
        let moved = max_abs_diff(&cpu, &x0);
        println!("batched 50 ops 25³: max |gpu − cpu| = {err:.2e}, solution moved by {moved:.2e}");
        assert!(moved > 1e-2, "test is vacuous: field barely changed ({moved})");
        assert!(err < 1e-6, "batched vs cpu: {err}");
        assert_eq!(batched, immediate, "batched and immediate must be bit-identical");
        assert_ghosts_zero(&g, &x.read_padded_raw(&ctx));
    }
}
