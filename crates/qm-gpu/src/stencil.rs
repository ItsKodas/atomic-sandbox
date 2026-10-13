//! 8th-order Laplacian / kinetic operator on padded fields — the GPU counterpart of
//! `qm::stencil::{laplacian, kinetic}` (same coefficients, Dirichlet f = 0 outside the box).
//!
//! The kernel (`shaders/stencil.wgsl`) runs one thread per interior point in (8, 8, 4)
//! workgroups and reads the ±4 neighbours along each axis straight from the padded buffer: the
//! zero ghost cells are the Dirichlet boundary, so there are no bounds checks in the stencil.
//! Only the interior of `out` is written (its ghosts stay 0). Laplacian and kinetic are the same
//! entry point with a different final scale (1/h² vs −1/(2h²)), so `kinetic` is bitwise
//! −½·`laplacian`. `*_rec` forms record into any `Recorder` (a caller-owned encoder or a `Batch`),
//! `*_in` forms into a `Batch` (one shared compute pass).
//!
//! Fused entries for the eigensolver (one dispatch instead of two or three): `hamiltonian`
//! (−½∇²f + V·f) and `jacobi_step` (one damped Jacobi sweep on (H − λ)w = r with H applied
//! inside the kernel). Their kinetic part is the same `d2` code as `laplacian`.
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid};
use crate::kernels::{immediate, uniform, Kernels};
use crate::pointwise::{check, no_alias};
use gpu::GpuContext;

/// Workgroup shape of the stencil kernel (must match `@workgroup_size` in stencil.wgsl).
pub const STENCIL_WORKGROUP: [u32; 3] = [8, 8, 4];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct StencilParams { scale: f32, p: u32, n: u32, zero: u32 /* must be 0: the shader's reassociation barrier */ }

/// Uniform of the fused `hamiltonian`/`jacobi_step` entries (binding 4 in stencil.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct HamParams { scale: f32, p: u32, n: u32, zero: u32, omega: f32, lambda: f32, diag_t: f32, pad: f32 }

/// 1/h² in f32.
fn inv_h2(g: &GpuGrid) -> f32 { (1.0 / (g.h * g.h)) as f32 }

fn stencil_workgroups(g: &GpuGrid) -> [u32; 3] {
    let n = g.n as u32;
    let [wx, wy, wz] = STENCIL_WORKGROUP;
    [n.div_ceil(wx), n.div_ceil(wy), n.div_ceil(wz)]
}

/// out = −½∇²f + v·f on the interior in one dispatch (recorded). `out` must alias neither input.
pub fn hamiltonian_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, v: &Field, out: &Field) {
    check(g, &[f, v, out]);
    no_alias("hamiltonian", out, f);
    no_alias("hamiltonian", out, v);
    let u = uniform(ctx, "hamiltonian", &HamParams { scale: -0.5 * inv_h2(g), p: g.p as u32, n: g.n as u32, zero: 0, ..Default::default() });
    let bg = k.bind_group(ctx, "hamiltonian", &[u.as_entire_binding(), out.buffer.as_entire_binding(), f.buffer.as_entire_binding(), v.buffer.as_entire_binding()]);
    r.dispatch(k, "hamiltonian", &bg, stencil_workgroups(g));
}

/// out = −½∇²f + v·f (into a `Batch`).
pub fn hamiltonian_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, f: &Field, v: &Field, out: &Field) {
    let ctx = b.ctx();
    hamiltonian_rec(b, ctx, k, g, f, v, out);
}

/// One damped Jacobi sweep on (H − λ)w = r, H = −½∇² + v, D = max(diag_t + v − λ, 0.1):
/// out = w + ω·D⁻¹·(r − (H w − λ w)) (recorded). `out` must alias none of the inputs (sweeps
/// ping-pong between two fields).
#[allow(clippy::too_many_arguments)]
pub fn jacobi_step_rec(rec: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, w: &Field, v: &Field, r: &Field, out: &Field, omega: f64, lambda: f64, diag_t: f64) {
    check(g, &[w, v, r, out]);
    for f in [w, v, r] { no_alias("jacobi_step", out, f); }
    let u = uniform(ctx, "jacobi_step", &HamParams {
        scale: -0.5 * inv_h2(g), p: g.p as u32, n: g.n as u32, zero: 0,
        omega: omega as f32, lambda: lambda as f32, diag_t: diag_t as f32, pad: 0.0,
    });
    let bg = k.bind_group(ctx, "jacobi_step", &[
        u.as_entire_binding(), out.buffer.as_entire_binding(), w.buffer.as_entire_binding(), v.buffer.as_entire_binding(), r.buffer.as_entire_binding(),
    ]);
    rec.dispatch(k, "jacobi_step", &bg, stencil_workgroups(g));
}

/// `jacobi_step_rec` into a `Batch`.
#[allow(clippy::too_many_arguments)]
pub fn jacobi_step_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, w: &Field, v: &Field, r: &Field, out: &Field, omega: f64, lambda: f64, diag_t: f64) {
    let ctx = b.ctx();
    jacobi_step_rec(b, ctx, k, g, w, v, r, out, omega, lambda, diag_t);
}

#[allow(clippy::too_many_arguments)]
fn stencil_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, op: &str, f: &Field, out: &Field, scale: f32) {
    check(g, &[f, out]);
    no_alias(op, out, f);
    let u = uniform(ctx, op, &StencilParams { scale, p: g.p as u32, n: g.n as u32, zero: 0 });
    let bg = k.bind_group(ctx, "laplacian", &[u.as_entire_binding(), out.buffer.as_entire_binding(), f.buffer.as_entire_binding()]);
    r.dispatch(k, "laplacian", &bg, stencil_workgroups(g));
}

/// out = ∇²f on the interior (recorded). Ghosts of `f` must be 0 (they are the Dirichlet wall).
pub fn laplacian_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    stencil_rec(r, ctx, k, g, "laplacian", f, out, inv_h2(g));
}

/// out = ∇²f (into a `Batch`).
pub fn laplacian_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    let ctx = b.ctx();
    laplacian_rec(b, ctx, k, g, f, out);
}

/// out = ∇²f on the interior; reads the zero ghosts → Dirichlet for free.
pub fn laplacian(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    immediate(ctx, "laplacian", |enc| laplacian_rec(enc, ctx, k, g, f, out));
}

/// out = −½∇²f (recorded). Scale −½·(1/h²) is exact in f32, so this is bitwise −½·laplacian.
pub fn kinetic_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    stencil_rec(r, ctx, k, g, "kinetic", f, out, -0.5 * inv_h2(g));
}

/// out = −½∇²f (into a `Batch`).
pub fn kinetic_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, f: &Field, out: &Field) {
    let ctx = b.ctx();
    kinetic_rec(b, ctx, k, g, f, out);
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
    use wgpu::CommandEncoder;

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

    /// The fused `hamiltonian` equals `kinetic` + V·f (the two-dispatch form) to f32 rounding,
    /// and `jacobi_step` equals the f64 sweep w + ω D⁻¹(r − (H w − λ w)) with D = max(diag_t + V − λ, 0.1)
    /// evaluated on the same f32-rounded inputs (the floor of D is exercised: V − λ < −diag_t at
    /// the centre of the well).
    #[test]
    fn fused_hamiltonian_and_jacobi_step_match_cpu() {
        use crate::pointwise::mul_add;
        use crate::stencil::{hamiltonian_in, jacobi_step_in};
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(21, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let w0 = cg.sample(|[x, y, z]| (-(x * x + 0.5 * y * y + 2.0 * z * z)).exp() * (1.0 + 0.3 * x));
        let v0 = cg.sample(|[x, y, z]| 0.5 * (x * x + y * y + z * z) - 80.0); // deep enough that D hits its 0.1 floor
        let r0 = cg.sample(|[x, y, z]| (x - 0.2 * y + 0.1 * z).sin() * (-(x * x + y * y + z * z) / 3.0).exp());
        let (w, v, r, out) = (upload(&ctx, &g, &w0, "w"), upload(&ctx, &g, &v0, "v"), upload(&ctx, &g, &r0, "r"), Field::new(&ctx, &g, "out"));
        let (omega, lambda) = (qm::eigen::jacobi_omega(), -20.0);
        let diag_t = -1.5 * qm::stencil::COEFF[0] / (cg.h * cg.h);
        // reference on the f32-rounded inputs, f64 arithmetic
        let r32 = |d: &[f64]| -> Vec<f64> { d.iter().map(|x| *x as f32 as f64).collect() };
        let (w32, v32, r32v) = (r32(&w0), r32(&v0), r32(&r0));
        let mut hw = vec![0.0; cg.len()];
        qm::stencil::kinetic(&cg, &w32, &mut hw);
        for i in 0..cg.len() { hw[i] += v32[i] * w32[i]; }
        let mut b = crate::batch::Batch::new(&ctx, "fused");
        hamiltonian_in(&mut b, &k, &g, &w, &v, &out);
        b.submit();
        let got = out.read_interior(&ctx, &g);
        let scale = hw.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let err_h = max_abs_diff(&got, &hw);
        // the two-dispatch form on the same inputs
        let two = Field::new(&ctx, &g, "two");
        kinetic(&ctx, &k, &g, &w, &two);
        mul_add(&ctx, &k, &g, &two, &v, &w);
        let err_two = max_abs_diff(&got, &two.read_interior(&ctx, &g));
        let want: Vec<f64> = (0..cg.len()).map(|i| {
            let d = (diag_t + v32[i] - lambda).max(0.1);
            w32[i] + omega / d * (r32v[i] - (hw[i] - lambda * w32[i]))
        }).collect();
        assert!((0..cg.len()).any(|i| diag_t + v32[i] - lambda < 0.1), "test assumption: the 0.1 floor of D is exercised");
        let out2 = Field::new(&ctx, &g, "out2");
        let mut b = crate::batch::Batch::new(&ctx, "fused");
        jacobi_step_in(&mut b, &k, &g, &w, &v, &r, &out2, omega, lambda, diag_t);
        b.submit();
        let got2 = out2.read_interior(&ctx, &g);
        let scale2 = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let err_j = max_abs_diff(&got2, &want);
        println!("fused 21³: hamiltonian max|Δ| {err_h:.2e} vs f64 (max|Hw| {scale:.2}), {err_two:.2e} vs kinetic+mul_add; jacobi_step max|Δ| {err_j:.2e} (max {scale2:.2})");
        assert!(err_h < 1e-6 * scale, "hamiltonian: {err_h:.3e}");
        assert!(err_two < 1e-6 * scale, "hamiltonian vs two dispatches: {err_two:.3e}");
        assert!(err_j < 1e-6 * scale2, "jacobi_step: {err_j:.3e}");
        assert_ghosts_zero(&g, &out.read_padded_raw(&ctx));
        assert_ghosts_zero(&g, &out2.read_padded_raw(&ctx));
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
