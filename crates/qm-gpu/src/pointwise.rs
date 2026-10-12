//! Pointwise vector ops on padded fields. All write the interior only (ghosts stay 0), record
//! into a fresh encoder and submit without waiting. Outputs must not alias inputs (asserted):
//! a buffer bound read_write and read-only in one dispatch is a wgpu validation error.
use crate::grid::{Field, GpuGrid};
use crate::kernels::{uniform, workgroups_for, Kernels, LINCOMB_MAX_TERMS};
use gpu::GpuContext;

/// Uniform shared by the pointwise and reduction kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params { pub a: f32, pub b: f32, pub p: u32, pub n: u32 }

impl Params {
    pub fn new(g: &GpuGrid, a: f32, b: f32) -> Params { Params { a, b, p: g.p as u32, n: g.n as u32 } }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LincombParams { n: u32, p: u32, count: u32, accumulate: u32, c: [f32; 8] }

fn check(g: &GpuGrid, fields: &[&Field]) {
    for f in fields { assert_eq!(f.len, g.padded_len(), "pointwise: field/grid size mismatch"); }
}

fn no_alias(op: &str, out: &Field, input: &Field) {
    assert!(out.buffer != input.buffer, "{op}: output must not alias an input");
}

fn run(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, name: &str, params: Params, fields: &[&Field]) {
    check(g, fields);
    let u = uniform(ctx, name, &params);
    let mut res = vec![u.as_entire_binding()];
    res.extend(fields.iter().map(|f| f.buffer.as_entire_binding()));
    k.run(ctx, name, &res, workgroups_for(g.interior_len()));
}

/// y += a·x
pub fn axpy(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32, x: &Field) {
    no_alias("axpy", y, x);
    run(ctx, k, g, "axpy", Params::new(g, a, 0.0), &[y, x]);
}

/// y *= a
pub fn scale(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    run(ctx, k, g, "scale", Params::new(g, a, 0.0), &[y]);
}

/// rho += psi² (density accumulate)
pub fn mul_add_sq(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho: &Field, psi: &Field) {
    no_alias("mul_add_sq", rho, psi);
    run(ctx, k, g, "mul_add_sq", Params::new(g, 0.0, 0.0), &[rho, psi]);
}

/// out = a·b (e.g. V·ψ). `a` and `b` may be the same field.
pub fn mul(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, a: &Field, b: &Field) {
    no_alias("mul", out, a);
    no_alias("mul", out, b);
    run(ctx, k, g, "mul", Params::new(g, 0.0, 0.0), &[out, a, b]);
}

/// Zeroes the whole padded buffer (interior and ghosts).
pub fn zero(ctx: &GpuContext, _k: &Kernels, g: &GpuGrid, f: &Field) {
    check(g, &[f]);
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero") });
    enc.clear_buffer(&f.buffer, 0, None);
    ctx.queue.submit([enc.finish()]);
}

/// out = Σ cᵢ xᵢ. Any number of terms: chunks of `LINCOMB_MAX_TERMS` are chained (first chunk
/// overwrites, later chunks accumulate). Terms equal to `out` itself are allowed: their
/// coefficients are summed and applied first with `scale`, the rest then accumulates onto it.
/// No terms ⇒ out = 0.
pub fn lincomb(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, terms: &[(f32, &Field)]) {
    check(g, &[out]);
    let mut self_coef: Option<f32> = None;
    let mut others: Vec<(f32, &Field)> = Vec::with_capacity(terms.len());
    for &(c, f) in terms {
        check(g, &[f]);
        if f.buffer == out.buffer { *self_coef.get_or_insert(0.0) += c; } else { others.push((c, f)); }
    }
    let mut accumulate = match self_coef {
        Some(c) => { scale(ctx, k, g, out, c); true }
        None => false,
    };
    if others.is_empty() {
        if !accumulate { zero(ctx, k, g, out); }
        return;
    }
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lincomb") });
    let wg = workgroups_for(g.interior_len());
    // keep uniforms alive until the submit
    let mut uniforms = Vec::new();
    for chunk in others.chunks(LINCOMB_MAX_TERMS) {
        let mut c = [0.0f32; 8];
        for (slot, (coef, _)) in c.iter_mut().zip(chunk) { *slot = *coef; }
        let lp = LincombParams { n: g.n as u32, p: g.p as u32, count: chunk.len() as u32, accumulate: accumulate as u32, c };
        uniforms.push(uniform(ctx, "lincomb", &lp));
        let u = uniforms.last().unwrap();
        let mut res = vec![u.as_entire_binding(), out.buffer.as_entire_binding()];
        // unused slots re-bind the chunk's first input (read-only; never read, count guards them)
        res.extend((0..LINCOMB_MAX_TERMS).map(|t| chunk.get(t).unwrap_or(&chunk[0]).1.buffer.as_entire_binding()));
        let bg = k.bind_group(ctx, "lincomb", &res);
        k.record(&mut enc, "lincomb", &bg, wg);
        accumulate = true;
    }
    ctx.queue.submit([enc.finish()]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};

    /// |got − want| ≤ 1e-6 · mag elementwise, where `mag` is the magnitude of the terms that
    /// produced `want` (so cancellation to ~0 is judged against the inputs, not against 0).
    fn check(name: &str, got: &[f64], want: &[f64], mag: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (idx, ((g, w), m)) in got.iter().zip(want).zip(mag).enumerate() {
            assert!((g - w).abs() <= 1e-6 * m.max(1e-6), "{name}[{idx}]: gpu {g} cpu {w} (mag {m})");
        }
    }

    #[test]
    fn axpy_scale_lincomb_mul_density_match_cpu() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let x = cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let y = cg.sample(|[x, y, z]| 1.0 + 0.1 * x - 0.05 * y * z);
        let z = cg.sample(|[x, y, z]| (x - 0.5 * y + 0.3 * z).sin());
        let up = |d: &[f64], l: &str| { let f = Field::new(&ctx, &g, l); f.upload_interior(&ctx, &g, d); f };
        let (fx, fy, fz) = (up(&x, "x"), up(&y, "y"), up(&z, "z"));
        let read = |f: &Field| { assert_ghosts_zero(&g, &f.read_padded_raw(&ctx)); f.read_interior(&ctx, &g) };
        let zip2 = |a: &[f64], b: &[f64], op: &dyn Fn(f64, f64) -> f64| -> Vec<f64> { a.iter().zip(b).map(|(p, q)| op(*p, *q)).collect() };

        // axpy: y += a·x
        let out = up(&y, "out");
        axpy(&ctx, &k, &g, &out, 0.7, &fx);
        check("axpy", &read(&out), &zip2(&y, &x, &|p, q| p + 0.7 * q), &zip2(&y, &x, &|p, q| p.abs() + 0.7 * q.abs()));

        // scale: y *= a
        let out = up(&y, "out");
        scale(&ctx, &k, &g, &out, -1.3);
        check("scale", &read(&out), &y.iter().map(|p| -1.3 * p).collect::<Vec<_>>(), &y.iter().map(|p| 1.3 * p.abs()).collect::<Vec<_>>());

        // mul: out = a·b (out starts non-zero to prove it is overwritten)
        let out = up(&z, "out");
        mul(&ctx, &k, &g, &out, &fx, &fy);
        check("mul", &read(&out), &zip2(&x, &y, &|p, q| p * q), &zip2(&x, &y, &|p, q| (p * q).abs()));

        // mul_add_sq: rho += psi², twice
        let rho = up(&y, "rho");
        mul_add_sq(&ctx, &k, &g, &rho, &fz);
        mul_add_sq(&ctx, &k, &g, &rho, &fx);
        let want: Vec<f64> = (0..y.len()).map(|i| y[i] + z[i] * z[i] + x[i] * x[i]).collect();
        let mag: Vec<f64> = (0..y.len()).map(|i| y[i].abs() + z[i] * z[i] + x[i] * x[i]).collect();
        check("mul_add_sq", &read(&rho), &want, &mag);

        // lincomb: 3 terms (one dispatch), out starts non-zero
        let out = up(&z, "out");
        lincomb(&ctx, &k, &g, &out, &[(0.5, &fx), (-2.0, &fy), (1.5, &fz)]);
        let want: Vec<f64> = (0..y.len()).map(|i| 0.5 * x[i] - 2.0 * y[i] + 1.5 * z[i]).collect();
        let mag: Vec<f64> = (0..y.len()).map(|i| 0.5 * x[i].abs() + 2.0 * y[i].abs() + 1.5 * z[i].abs()).collect();
        check("lincomb3", &read(&out), &want, &mag);

        // lincomb: 17 terms (chained dispatches) including `out` itself as an input (aliasing)
        let out = up(&y, "out");
        let fs = [&fx, &fy, &fz];
        let cpu = [&x, &y, &z];
        let mut terms: Vec<(f32, &Field)> = (0..16).map(|t| (0.1 * t as f32 - 0.7, fs[t % 3])).collect();
        terms.push((0.25, &out));
        lincomb(&ctx, &k, &g, &out, &terms);
        let want: Vec<f64> = (0..y.len()).map(|i| {
            (0..16).map(|t| (0.1f32 * t as f32 - 0.7) as f64 * cpu[t % 3][i]).sum::<f64>() + 0.25 * y[i]
        }).collect();
        let mag: Vec<f64> = (0..y.len()).map(|i| {
            (0..16).map(|t| ((0.1f32 * t as f32 - 0.7) as f64 * cpu[t % 3][i]).abs()).sum::<f64>() + 0.25 * y[i].abs()
        }).collect();
        check("lincomb17", &read(&out), &want, &mag);

        // lincomb with no terms zeroes the output
        let out = up(&y, "out");
        lincomb(&ctx, &k, &g, &out, &[]);
        assert!(read(&out).iter().all(|v| *v == 0.0));

        // zero
        let out = up(&y, "out");
        zero(&ctx, &k, &g, &out);
        assert!(out.read_padded_raw(&ctx).iter().all(|v| *v == 0.0));

        // inputs untouched by all of the above
        check("x unchanged", &read(&fx), &x, &x.iter().map(|v| v.abs()).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "alias")]
    fn axpy_rejects_aliasing() {
        let Some(ctx) = gpu::GpuContext::new_headless() else { panic!("alias (no adapter; forced)") };
        let g = GpuGrid::new(&qm::Grid::cube(5, 1.0));
        let k = Kernels::new(&ctx);
        let f = Field::new(&ctx, &g, "f");
        axpy(&ctx, &k, &g, &f, 1.0, &f);
    }
}
