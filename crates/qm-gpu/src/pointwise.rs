//! Pointwise vector ops on padded fields. All write the interior only (ghosts stay 0). Each op
//! has three forms:
//! - `op_rec(r, ctx, …)` records its dispatch(es) into any `Recorder`: a caller-owned
//!   `CommandEncoder` (one compute pass per dispatch; see `kernels::encoder`) or a `Batch`;
//! - `op_in(batch, …)` = `op_rec` into a `Batch` (one shared compute pass, the fast path);
//! - `op(ctx, …)` = fresh encoder → `op_rec` → submit, without waiting.
//!
//! Outputs must not alias inputs (asserted): a buffer bound read_write and read-only in one
//! dispatch is a wgpu validation error.
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid};
use crate::kernels::{immediate, workgroups_for, Kernels, DIIS_MAX_TERMS, LINCOMB_MAX_TERMS};
use gpu::GpuContext;
use wgpu::CommandEncoder;

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DiisParams { n: u32, p: u32, count: u32, accumulate: u32, sub_mask: u32, clamp: u32, zero: u32, pad: u32, c: [f32; 8] }

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct UnpadParams { n: u32, p: u32, stride: u32, pad: u32 }

pub(crate) fn check(g: &GpuGrid, fields: &[&Field]) {
    for f in fields { assert_eq!(f.len, g.padded_len(), "field/grid size mismatch"); }
}

pub(crate) fn no_alias(op: &str, out: &Field, input: &Field) {
    assert!(out.buffer != input.buffer, "{op}: output must not alias an input");
}

/// Records one grid-stride dispatch of `name` with a fresh uniform (the bind group keeps it
/// alive until the work has executed, so recorded dispatches never share mutable state).
fn rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, name: &str, params: Params, fields: &[&Field]) {
    check(g, fields);
    let u = r.uniform(ctx, k, name, &params);
    let mut res = vec![u.binding()];
    res.extend(fields.iter().map(|f| f.buffer.as_entire_binding()));
    let bg = k.bind_group(ctx, name, &res);
    r.dispatch(k, name, &bg, [workgroups_for(g.interior_len()), 1, 1]);
}

/// y += a·x (recorded)
pub fn axpy_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32, x: &Field) {
    no_alias("axpy", y, x);
    rec(r, ctx, k, g, "axpy", Params::new(g, a, 0.0), &[y, x]);
}

/// y += a·x (into a `Batch`)
pub fn axpy_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, y: &Field, a: f32, x: &Field) {
    let ctx = b.ctx();
    axpy_rec(b, ctx, k, g, y, a, x);
}

/// y += a·x
pub fn axpy(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32, x: &Field) {
    immediate(ctx, "axpy", |enc| axpy_rec(enc, ctx, k, g, y, a, x));
}

/// y *= a (recorded)
pub fn scale_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    rec(r, ctx, k, g, "scale", Params::new(g, a, 0.0), &[y]);
}

/// y *= a (into a `Batch`)
pub fn scale_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    let ctx = b.ctx();
    scale_rec(b, ctx, k, g, y, a);
}

/// y *= a
pub fn scale(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    immediate(ctx, "scale", |enc| scale_rec(enc, ctx, k, g, y, a));
}

/// y = value on the interior (recorded). Ghosts are not touched (they stay 0). This is the
/// in-pass replacement for `zero` inside a `Batch` (a buffer clear cannot be recorded in a
/// compute pass).
pub fn fill_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, value: f32) {
    rec(r, ctx, k, g, "fill", Params::new(g, value, 0.0), &[y]);
}

/// y = value on the interior (into a `Batch`); `fill_in(b, …, 0.0)` is the batched `zero`.
pub fn fill_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, y: &Field, value: f32) {
    let ctx = b.ctx();
    fill_rec(b, ctx, k, g, y, value);
}

/// y = value on the interior (ghosts untouched).
pub fn fill(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, value: f32) {
    immediate(ctx, "fill", |enc| fill_rec(enc, ctx, k, g, y, value));
}

/// y = max(y, a) on the interior (recorded).
pub fn clamp_min_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    rec(r, ctx, k, g, "clamp_min", Params::new(g, a, 0.0), &[y]);
}

/// y = max(y, a) (into a `Batch`)
pub fn clamp_min_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    let ctx = b.ctx();
    clamp_min_rec(b, ctx, k, g, y, a);
}

/// y = max(y, a)
pub fn clamp_min(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: f32) {
    immediate(ctx, "clamp_min", |enc| clamp_min_rec(enc, ctx, k, g, y, a));
}

/// rho += psi² (density accumulate, recorded)
pub fn mul_add_sq_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho: &Field, psi: &Field) {
    no_alias("mul_add_sq", rho, psi);
    rec(r, ctx, k, g, "mul_add_sq", Params::new(g, 0.0, 0.0), &[rho, psi]);
}

/// rho += psi² (into a `Batch`)
pub fn mul_add_sq_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, rho: &Field, psi: &Field) {
    let ctx = b.ctx();
    mul_add_sq_rec(b, ctx, k, g, rho, psi);
}

/// rho += psi² (density accumulate)
pub fn mul_add_sq(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho: &Field, psi: &Field) {
    immediate(ctx, "mul_add_sq", |enc| mul_add_sq_rec(enc, ctx, k, g, rho, psi));
}

/// rho += f·psi² (the occupation-weighted density of `qm::scf` "Occupations", recorded)
pub fn mul_add_sq_w_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho: &Field, psi: &Field, f: f32) {
    no_alias("mul_add_sq_w", rho, psi);
    rec(r, ctx, k, g, "mul_add_sq_w", Params::new(g, f, 0.0), &[rho, psi]);
}

/// rho += f·psi² (into a `Batch`)
pub fn mul_add_sq_w_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, rho: &Field, psi: &Field, f: f32) {
    let ctx = b.ctx();
    mul_add_sq_w_rec(b, ctx, k, g, rho, psi, f);
}

/// out = a·b (recorded). `a` and `b` may be the same field.
pub fn mul_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, a: &Field, b: &Field) {
    no_alias("mul", out, a);
    no_alias("mul", out, b);
    rec(r, ctx, k, g, "mul", Params::new(g, 0.0, 0.0), &[out, a, b]);
}

/// out = a·b (into a `Batch`)
pub fn mul_in(batch: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field, a: &Field, b: &Field) {
    let ctx = batch.ctx();
    mul_rec(batch, ctx, k, g, out, a, b);
}

/// out = a·b (e.g. V·ψ). `a` and `b` may be the same field.
pub fn mul(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, a: &Field, b: &Field) {
    immediate(ctx, "mul", |enc| mul_rec(enc, ctx, k, g, out, a, b));
}

/// y += a·b (recorded). `a` and `b` may be the same field; neither may be `y`.
pub fn mul_add_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: &Field, b: &Field) {
    no_alias("mul_add", y, a);
    no_alias("mul_add", y, b);
    rec(r, ctx, k, g, "mul_add", Params::new(g, 0.0, 0.0), &[y, a, b]);
}

/// y += a·b (into a `Batch`)
pub fn mul_add_in(batch: &mut Batch, k: &Kernels, g: &GpuGrid, y: &Field, a: &Field, b: &Field) {
    let ctx = batch.ctx();
    mul_add_rec(batch, ctx, k, g, y, a, b);
}

/// y += a·b (e.g. the V·ψ term of H ψ on top of `kinetic`).
pub fn mul_add(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, y: &Field, a: &Field, b: &Field) {
    immediate(ctx, "mul_add", |enc| mul_add_rec(enc, ctx, k, g, y, a, b));
}

/// Zeroes the whole padded buffer, interior and ghosts (recorded `clear_buffer`). Encoder only:
/// a clear cannot be recorded inside a compute pass, so a `Batch` uses `fill_in(…, 0.0)`.
pub fn zero_rec(enc: &mut CommandEncoder, _ctx: &GpuContext, _k: &Kernels, g: &GpuGrid, f: &Field) {
    check(g, &[f]);
    enc.clear_buffer(&f.buffer, 0, None);
}

/// Zeroes the whole padded buffer (interior and ghosts).
pub fn zero(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field) {
    immediate(ctx, "zero", |enc| zero_rec(enc, ctx, k, g, f));
}

/// out = Σ cᵢ xᵢ (recorded). Any number of terms: chunks of `LINCOMB_MAX_TERMS` are chained
/// (first chunk overwrites, later chunks accumulate). Terms equal to `out` itself are allowed:
/// their coefficients are summed and applied first with `scale` (skipped when the sum is exactly
/// 1), the rest then accumulates onto it. No terms ⇒ out = 0 on the interior (`fill` with 0;
/// ghosts are 0 already).
pub fn lincomb_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, terms: &[(f32, &Field)]) {
    check(g, &[out]);
    let mut self_coef: Option<f32> = None;
    let mut others: Vec<(f32, &Field)> = Vec::with_capacity(terms.len());
    for &(c, f) in terms {
        check(g, &[f]);
        if f.buffer == out.buffer { *self_coef.get_or_insert(0.0) += c; } else { others.push((c, f)); }
    }
    let mut accumulate = match self_coef {
        Some(c) => { if c != 1.0 { scale_rec(r, ctx, k, g, out, c); } true }
        None => false,
    };
    if others.is_empty() {
        if !accumulate { fill_rec(r, ctx, k, g, out, 0.0); }
        return;
    }
    let wg = workgroups_for(g.interior_len());
    for chunk in others.chunks(LINCOMB_MAX_TERMS) {
        let mut c = [0.0f32; 8];
        for (slot, (coef, _)) in c.iter_mut().zip(chunk) { *slot = *coef; }
        let lp = LincombParams { n: g.n as u32, p: g.p as u32, count: chunk.len() as u32, accumulate: accumulate as u32, c };
        let u = r.uniform(ctx, k, "lincomb", &lp);
        let mut res = vec![u.binding(), out.buffer.as_entire_binding()];
        // unused slots re-bind the chunk's first input (read-only; never read, count guards them)
        res.extend((0..LINCOMB_MAX_TERMS).map(|t| chunk.get(t).unwrap_or(&chunk[0]).1.buffer.as_entire_binding()));
        let bg = k.bind_group(ctx, "lincomb", &res);
        r.dispatch(k, "lincomb", &bg, [wg, 1, 1]);
        accumulate = true;
    }
}

/// out = Σ cᵢ xᵢ (into a `Batch`); see `lincomb_rec`.
pub fn lincomb_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field, terms: &[(f32, &Field)]) {
    let ctx = b.ctx();
    lincomb_rec(b, ctx, k, g, out, terms);
}

/// out = Σ cᵢ xᵢ; see `lincomb_rec`.
pub fn lincomb(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, terms: &[(f32, &Field)]) {
    immediate(ctx, "lincomb", |enc| lincomb_rec(enc, ctx, k, g, out, terms));
}

/// One term of `diis_mix`: coefficient, field, and whether `base` is subtracted from the field
/// before scaling (`true` for a history density, `false` for a residual).
pub type DiisTerm<'a> = (f32, &'a Field, bool);

/// out = base + Σ_t c_t·(sub_t ? (x_t − base) : x_t), then max(out, 0) if `clamp` (recorded).
/// The re-centred evaluation of the Pulay mix (see `diis_mix` in pointwise.wgsl for why it is
/// not a `lincomb`): the differences are formed exactly and the sum of small terms is added to
/// `base` once per chunk of `DIIS_MAX_TERMS`, so the result carries ~1 ulp of ρ-sized rounding
/// instead of |c|·ρ-sized. `out` must not alias `base` or any term.
#[allow(clippy::too_many_arguments)]
pub fn diis_mix_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, base: &Field, terms: &[DiisTerm], clamp: bool) {
    check(g, &[out, base]);
    no_alias("diis_mix", out, base);
    for &(_, f, _) in terms { check(g, &[f]); no_alias("diis_mix", out, f); }
    let chunks = terms.len().div_ceil(DIIS_MAX_TERMS).max(1);
    let wg = workgroups_for(g.interior_len());
    for ci in 0..chunks {
        let chunk = &terms[(ci * DIIS_MAX_TERMS).min(terms.len())..((ci + 1) * DIIS_MAX_TERMS).min(terms.len())];
        let mut c = [0.0f32; 8];
        let mut sub_mask = 0u32;
        for (t, (slot, &(coef, _, sub))) in c.iter_mut().zip(chunk).enumerate() {
            *slot = coef;
            if sub { sub_mask |= 1 << t; }
        }
        let params = DiisParams {
            n: g.n as u32, p: g.p as u32, count: chunk.len() as u32, accumulate: (ci > 0) as u32,
            sub_mask, clamp: (clamp && ci + 1 == chunks) as u32, zero: 0, pad: 0, c,
        };
        let u = r.uniform(ctx, k, "diis_mix", &params);
        let mut res = vec![u.binding(), out.buffer.as_entire_binding(), base.buffer.as_entire_binding()];
        // unused slots re-bind `base` (read-only; never read, count guards them)
        res.extend((0..DIIS_MAX_TERMS).map(|t| chunk.get(t).map_or(base, |term| term.1).buffer.as_entire_binding()));
        let bg = k.bind_group(ctx, "diis_mix", &res);
        r.dispatch(k, "diis_mix", &bg, [wg, 1, 1]);
    }
}

/// out = base + Σ c_t·(sub_t ? (x_t − base) : x_t), clamped at 0 if `clamp`; see `diis_mix_rec`.
pub fn diis_mix(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, base: &Field, terms: &[DiisTerm], clamp: bool) {
    immediate(ctx, "diis_mix", |enc| diis_mix_rec(enc, ctx, k, g, out, base, terms, clamp));
}

/// `diis_mix` into a `Batch`; see `diis_mix_rec`.
pub fn diis_mix_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field, base: &Field, terms: &[DiisTerm], clamp: bool) {
    let ctx = b.ctx();
    diis_mix_rec(b, ctx, k, g, out, base, terms, clamp);
}

/// Floats per row of the row-padded buffer `unpad_rec` writes: the smallest multiple of 64
/// (256 bytes, wgpu's `COPY_BYTES_PER_ROW_ALIGNMENT`) that holds n floats.
pub fn unpad_row_stride(n: usize) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize / 4;
    (n.div_ceil(align) * align) as u32
}

/// Interior of `f` → `rows` (a STORAGE buffer of at least `stride·n·n` floats) in the
/// row-padded layout `i + stride·(j + n·k)` that `copy_buffer_to_texture` reads with
/// `bytes_per_row = 4·stride`, `rows_per_image = n` (recorded). `stride` from `unpad_row_stride`.
pub fn unpad_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, f: &Field, rows: &wgpu::Buffer, stride: u32) {
    check(g, &[f]);
    assert!(stride as usize >= g.n && stride.is_multiple_of(64), "unpad: stride {stride} must be a multiple of 64 floats ≥ n = {}", g.n);
    assert!(rows.size() >= (stride as u64) * (g.n * g.n * 4) as u64, "unpad: row buffer too small");
    assert!(*rows != f.buffer, "unpad: output must not alias the field");
    let u = r.uniform(ctx, k, "unpad", &UnpadParams { n: g.n as u32, p: g.p as u32, stride, pad: 0 });
    let bg = k.bind_group(ctx, "unpad", &[u.binding(), rows.as_entire_binding(), f.buffer.as_entire_binding()]);
    r.dispatch(k, "unpad", &bg, [workgroups_for(g.interior_len()), 1, 1]);
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

        // mul_add: y += a·b (out starts as y; a and b may coincide)
        let out = up(&y, "out");
        mul_add(&ctx, &k, &g, &out, &fx, &fz);
        mul_add(&ctx, &k, &g, &out, &fz, &fz);
        let want: Vec<f64> = (0..y.len()).map(|i| y[i] + x[i] * z[i] + z[i] * z[i]).collect();
        let mag: Vec<f64> = (0..y.len()).map(|i| y[i].abs() + (x[i] * z[i]).abs() + z[i] * z[i]).collect();
        check("mul_add", &read(&out), &want, &mag);

        // lincomb with a self coefficient of exactly 1 skips the scale: y = y + 0.5·x
        let out = up(&y, "out");
        lincomb(&ctx, &k, &g, &out, &[(1.0, &out), (0.5, &fx)]);
        check("lincomb self 1", &read(&out), &zip2(&y, &x, &|p, q| p + 0.5 * q), &zip2(&y, &x, &|p, q| p.abs() + 0.5 * q.abs()));

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

        // fill: interior = value, ghosts stay 0
        let out = up(&y, "out");
        fill(&ctx, &k, &g, &out, -2.5);
        assert!(read(&out).iter().all(|v| *v == -2.5));

        // clamp_min: y = max(y, 0.3)
        let out = up(&z, "out");
        clamp_min(&ctx, &k, &g, &out, 0.3);
        check("clamp_min", &read(&out), &z.iter().map(|p| p.max(0.3)).collect::<Vec<_>>(), &z.iter().map(|p| p.abs().max(0.3)).collect::<Vec<_>>());

        // diis_mix: out = base + Σ c_t·(sub ? x_t − base : x_t), 8 terms (two chunks of 6), clamped at 0.
        // base = y (positive-ish), differences against x and z, residual terms of both signs.
        let out = up(&z, "out");
        let terms: Vec<DiisTerm> = vec![(0.7, &fx, true), (-0.4, &fz, true), (0.3, &fx, false), (-1.2, &fz, false), (0.25, &fy, true), (0.5, &fz, true), (-0.6, &fx, false), (0.9, &fz, false)];
        diis_mix(&ctx, &k, &g, &out, &fy, &terms, true);
        let cpu_terms: [(f64, &Vec<f64>, bool); 8] = [(0.7, &x, true), (-0.4, &z, true), (0.3, &x, false), (-1.2, &z, false), (0.25, &y, true), (0.5, &z, true), (-0.6, &x, false), (0.9, &z, false)];
        let want: Vec<f64> = (0..y.len()).map(|i| {
            (y[i] + cpu_terms.iter().map(|(c, f, sub)| c * if *sub { f[i] - y[i] } else { f[i] }).sum::<f64>()).max(0.0)
        }).collect();
        let mag: Vec<f64> = (0..y.len()).map(|i| y[i].abs() + cpu_terms.iter().map(|(c, f, sub)| (c * if *sub { f[i] - y[i] } else { f[i] }).abs()).sum::<f64>()).collect();
        check("diis_mix", &read(&out), &want, &mag);
        assert!(want.contains(&0.0), "test must exercise the clamp");
        // no terms: out = max(base, 0)
        let out = up(&z, "out");
        diis_mix(&ctx, &k, &g, &out, &fz, &[], true);
        check("diis_mix empty", &read(&out), &z.iter().map(|p| p.max(0.0)).collect::<Vec<_>>(), &z.iter().map(|p| p.abs()).collect::<Vec<_>>());

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
