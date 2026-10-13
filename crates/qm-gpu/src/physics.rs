//! Physics kernels: the analytic nuclear potential (`shaders/nuclear.wgsl`) and LSDA
//! (Dirac + PW92) exchange-correlation (`shaders/xc.wgsl`), the GPU counterparts of
//! `qm::nuclear::nuclear_potential` and `qm::xc::lsda_pw92`. Both are pointwise, write the
//! interior only (ghosts stay 0) and come in the usual three forms (`_rec` into any `Recorder`,
//! `_in` into a `Batch`, immediate). Achieved precision vs the f64 CPU reference is pinned by
//! the tests below.
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid};
use crate::kernels::{immediate, workgroups_for, Kernels};
use crate::pointwise::{check, no_alias};
use gpu::GpuContext;
use qm::Nucleus;
use wgpu::util::DeviceExt;

/// Most nuclei one `nuclear_potential` dispatch accepts (each grid point loops over all of them).
pub const MAX_NUCLEI: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NucParams { p: u32, n: u32, count: u32, zero: u32, h: f32, inv_s2: f32, pad: [f32; 2] }

/// A nucleus as the kernel sees it: its position relative to the grid origin in units of h,
/// split into an integer cell and an f32 fraction, plus Z (in `frac[3]`). The kernel forms the
/// displacement from grid point i as (i − cell) − frac: an exact integer minus a fraction, so
/// the distance keeps *relative* f32 precision even right next to the nucleus.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NucleusRecord { pub cell: [i32; 4], pub frac: [f32; 4] }

impl NucleusRecord {
    pub(crate) fn new(g: &GpuGrid, nuc: &Nucleus) -> NucleusRecord {
        let (mut cell, mut frac) = ([0i32; 4], [0f32; 4]);
        for a in 0..3 {
            let c = (nuc.pos[a] - g.origin[a]) / g.h;
            let fl = c.floor();
            cell[a] = fl as i32;
            frac[a] = (c - fl) as f32;
        }
        frac[3] = nuc.z as f32;
        NucleusRecord { cell, frac }
    }
}

/// V_nuc(r) = −Σ_I Z_I erf(d/(σ√2))/d on the interior, d → 0 limit −Z√(2/π)/σ (recorded).
/// At most `MAX_NUCLEI` nuclei; none ⇒ out = 0. erf: Maclaurin series of erf(x)/x for
/// x = d/(σ√2) < 1, a Chebyshev/asymptotic erfc (|Δerf| ≤ 6.1e-9) above — see the shader.
#[allow(clippy::too_many_arguments)]
pub fn nuclear_potential_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, nuclei: &[Nucleus], sigma: f64) {
    check(g, &[out]);
    assert!(nuclei.len() <= MAX_NUCLEI, "nuclear_potential: {} nuclei > MAX_NUCLEI = {MAX_NUCLEI}", nuclei.len());
    assert!(sigma > 0.0, "nuclear_potential: sigma must be positive, got {sigma}");
    let mut records: Vec<NucleusRecord> = nuclei.iter().map(|n| NucleusRecord::new(g, n)).collect();
    if records.is_empty() { records.push(NucleusRecord::default()); } // bindings cannot be empty; count = 0
    let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nuclei"), contents: bytemuck::cast_slice(&records), usage: wgpu::BufferUsages::STORAGE,
    });
    let params = NucParams {
        p: g.p as u32, n: g.n as u32, count: nuclei.len() as u32, zero: 0,
        h: g.h as f32, inv_s2: (1.0 / (sigma * std::f64::consts::SQRT_2)) as f32, pad: [0.0; 2],
    };
    let u = r.uniform(ctx, k, "nuclear_potential", &params);
    let bg = k.bind_group(ctx, "nuclear_potential", &[u.binding(), out.buffer.as_entire_binding(), buf.as_entire_binding()]);
    r.dispatch(k, "nuclear_potential", &bg, [workgroups_for(g.interior_len()), 1, 1]);
}

/// `nuclear_potential` into a `Batch`.
pub fn nuclear_potential_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, out: &Field, nuclei: &[Nucleus], sigma: f64) {
    let ctx = b.ctx();
    nuclear_potential_rec(b, ctx, k, g, out, nuclei, sigma);
}

/// V_nuc(r) = −Σ_I Z_I erf(d/(σ√2))/d on the interior, d → 0 limit −Z√(2/π)/σ. Nuclei are
/// passed to the kernel as a storage buffer (cell + fraction + Z, 32 bytes each), ≤ 256.
pub fn nuclear_potential(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, out: &Field, nuclei: &[Nucleus], sigma: f64) {
    immediate(ctx, "nuclear_potential", |enc| nuclear_potential_rec(enc, ctx, k, g, out, nuclei, sigma));
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct XcParams { p: u32, n: u32, zero: u32, pad: u32 }

/// From (ρ↑, ρ↓): energy density n·ε_xc, v↑, v↓ — `qm::xc` (Dirac + PW92) in f32 (recorded).
/// `rho_up` and `rho_dn` may be the same field (unpolarised: pass ρ/2 twice); the three
/// outputs must be distinct and must not alias the inputs.
#[allow(clippy::too_many_arguments)]
pub fn xc_lsda_rec(r: &mut impl Recorder, ctx: &GpuContext, k: &Kernels, g: &GpuGrid,
                   rho_up: &Field, rho_dn: &Field, exc: &Field, v_up: &Field, v_dn: &Field) {
    check(g, &[rho_up, rho_dn, exc, v_up, v_dn]);
    for out in [exc, v_up, v_dn] {
        no_alias("xc_lsda", out, rho_up);
        no_alias("xc_lsda", out, rho_dn);
    }
    no_alias("xc_lsda (exc/v_up)", exc, v_up);
    no_alias("xc_lsda (exc/v_dn)", exc, v_dn);
    no_alias("xc_lsda (v_up/v_dn)", v_up, v_dn);
    let u = r.uniform(ctx, k, "xc_lsda", &XcParams { p: g.p as u32, n: g.n as u32, zero: 0, pad: 0 });
    let res: Vec<wgpu::BindingResource> = [rho_up, rho_dn, exc, v_up, v_dn].iter().map(|f| f.buffer.as_entire_binding()).collect();
    let bg = k.bind_group(ctx, "xc_lsda", &[&[u.binding()][..], &res].concat());
    r.dispatch(k, "xc_lsda", &bg, [workgroups_for(g.interior_len()), 1, 1]);
}

/// `xc_lsda` into a `Batch`.
#[allow(clippy::too_many_arguments)]
pub fn xc_lsda_in(b: &mut Batch, k: &Kernels, g: &GpuGrid, rho_up: &Field, rho_dn: &Field, exc: &Field, v_up: &Field, v_dn: &Field) {
    let ctx = b.ctx();
    xc_lsda_rec(b, ctx, k, g, rho_up, rho_dn, exc, v_up, v_dn);
}

/// From (ρ↑, ρ↓): energy density n·ε_xc, v↑, v↓ — the PW92/Dirac formulas of `qm::xc`
/// transcribed to WGSL in f32. Vacuum (n < 1e-18) gives exactly 0 in all three.
#[allow(clippy::too_many_arguments)]
pub fn xc_lsda(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, rho_up: &Field, rho_dn: &Field, exc: &Field, v_up: &Field, v_dn: &Field) {
    immediate(ctx, "xc_lsda", |enc| xc_lsda_rec(enc, ctx, k, g, rho_up, rho_dn, exc, v_up, v_dn));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};
    use crate::kernels::Kernels;
    use gpu::GpuContext;
    use qm::Nucleus;

    fn max_abs(v: &[f64]) -> f64 { v.iter().fold(0.0, |m, x| m.max(x.abs())) }
    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }

    /// CPU reference evaluated on exactly the f32 inputs the kernel sees (positions relative to
    /// the grid as f32 fractions of a cell, f32 h and f32 1/(σ√2)), in f64: isolates the
    /// kernel's own error from the rounding of its inputs.
    fn cpu_on_f32_inputs(cg: &qm::Grid, g: &GpuGrid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64> {
        let h32 = g.h as f32 as f64;
        let inv_s2 = (1.0 / (sigma * std::f64::consts::SQRT_2)) as f32 as f64;
        let rounded: Vec<Nucleus> = nuclei.iter().map(|n| {
            let rec = NucleusRecord::new(g, n);
            let pos = std::array::from_fn(|a| cg.origin[a] + (rec.cell[a] as f64 + rec.frac[a] as f64) * h32);
            Nucleus { pos, z: n.z }
        }).collect();
        let grid = qm::Grid { n: cg.n, h: h32, origin: cg.origin };
        qm::nuclear::nuclear_potential(&grid, &rounded, 1.0 / (inv_s2 * std::f64::consts::SQRT_2))
    }

    fn gpu_vnuc(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, nuclei: &[Nucleus], sigma: f64) -> Vec<f64> {
        let out = Field::new(ctx, g, "vnuc");
        nuclear_potential(ctx, k, g, &out, nuclei, sigma);
        assert_ghosts_zero(g, &out.read_padded_raw(ctx));
        out.read_interior(ctx, g)
    }

    #[test]
    fn nuclear_potential_matches_cpu_erf_form() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(33, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let nuc = [Nucleus { pos: [0.7, 0.3, -0.2], z: 1 }, Nucleus { pos: [-0.7, 0.0, 0.0], z: 2 }];
        let sigma = 0.25;
        let gpu = gpu_vnuc(&ctx, &k, &g, &nuc, sigma);
        let cpu = qm::nuclear::nuclear_potential(&cg, &nuc, sigma);
        let vmax = max_abs(&cpu);
        let err = max_abs_diff(&gpu, &cpu);
        let err_kernel = max_abs_diff(&gpu, &cpu_on_f32_inputs(&cg, &g, &nuc, sigma));
        // pointwise relative error (V < 0 everywhere, |V| ≥ 3/(√3·4) far out)
        let rel = gpu.iter().zip(&cpu).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
        println!("nuclear 33³ σ=0.25: max|V| = {vmax:.4}, max |Δ| = {err:.2e} ({:.2e} of max|V|), kernel-only {err_kernel:.2e}, max pointwise rel {rel:.2e}", err / vmax);
        // Brief bound 2e-6·max|V|, tightened to 5e-7 (Task 7: 1.3e-7·max|V| with A&S) and to 3e-7
        // with Task 10's erfc fit: 1.02e-7·max|V| (7.1e-7 absolute, the same kernel-only: input
        // rounding is negligible) and ≤ 2.3e-7 pointwise relative — f32 rounding, no systematic
        // part left (A&S's 1.4e-7 absolute erf error was smooth in x).
        assert!(err < 3e-7 * vmax, "max |gpu − cpu| = {err:.3e}, max|V| = {vmax}");
        assert!(err_kernel < 3e-7 * vmax, "kernel-only max |gpu − cpu| = {err_kernel:.3e}, max|V| = {vmax}");
        assert!(rel < 5e-7, "max pointwise relative error {rel:.3e}");
    }

    #[test]
    fn nuclear_potential_on_site_limit() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.25);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let sigma = 0.25;
        // exactly on grid point (6, 9, 8)
        let at = cg.idx(6, 9, 8);
        let nuc = [Nucleus { pos: cg.point(at), z: 3 }];
        let gpu = gpu_vnuc(&ctx, &k, &g, &nuc, sigma);
        let limit = -3.0 * (2.0 / std::f64::consts::PI).sqrt() / sigma;
        println!("on-site: gpu {} limit {limit} rel {:.2e}", gpu[at], ((gpu[at] - limit) / limit).abs());
        // brief: within 1e-5 (abs, ≈ 1e-6 relative here); achieved 2e-8 relative
        assert!((gpu[at] - limit).abs() < 1e-6 * limit.abs(), "on-site {} vs {limit}", gpu[at]);
        let cpu = qm::nuclear::nuclear_potential(&cg, &nuc, sigma);
        assert!(max_abs_diff(&gpu, &cpu) < 1e-6 * limit.abs());

        // 1e-3 Bohr off a grid point: d/(σ√2) ≈ 3e-3, where 1 − erfc evaluated there would lose
        // (its absolute erf error)/x relative (5e-5 with A&S); the small-x series keeps it at f32
        // rounding (achieved 2.4e-7; bound 1e-6 → 5e-7 in Task 10).
        let p = cg.point(at);
        let nuc = [Nucleus { pos: [p[0] + 1e-3, p[1] - 2e-4, p[2] + 5e-4], z: 3 }];
        let gpu = gpu_vnuc(&ctx, &k, &g, &nuc, sigma);
        let cpu = qm::nuclear::nuclear_potential(&cg, &nuc, sigma);
        let rel = gpu.iter().zip(&cpu).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
        println!("near-site (1.1e-3 Bohr): nearest point gpu {} cpu {}, max pointwise rel {rel:.2e}", gpu[at], cpu[at]);
        assert!(rel < 5e-7, "near-site max pointwise relative error {rel:.3e}");
    }

    #[test]
    fn nuclear_potential_with_no_nuclei_is_zero() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let g = GpuGrid::new(&qm::Grid::cube(9, 0.5));
        let k = Kernels::new(&ctx);
        let out = Field::new(&ctx, &g, "v");
        out.upload_interior(&ctx, &g, &vec![1.0; g.interior_len()]);
        nuclear_potential(&ctx, &k, &g, &out, &[], 0.3);
        assert!(out.read_padded_raw(&ctx).iter().all(|v| *v == 0.0));
    }

    /// Worst errors of `got` vs `want` under the brief's rule: relative where |want| > 1e-3,
    /// absolute elsewhere. Returns (max rel, max abs, index of max rel).
    fn xc_err(got: &[f64], want: &[f64]) -> (f64, f64, usize) {
        let (mut rel, mut abs, mut at) = (0.0f64, 0.0f64, 0);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "non-finite GPU value at {i}: {g}");
            if w.abs() > 1e-3 {
                let r = ((g - w) / w).abs();
                if r > rel { rel = r; at = i; }
            } else {
                abs = abs.max((g - w).abs());
            }
        }
        (rel, abs, at)
    }

    struct XcOut { exc: Vec<f64>, v_up: Vec<f64>, v_dn: Vec<f64> }

    fn gpu_xc(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, up: &[f64], dn: &[f64]) -> XcOut {
        let (fu, fd) = (Field::new(ctx, g, "up"), Field::new(ctx, g, "dn"));
        fu.upload_interior(ctx, g, up);
        fd.upload_interior(ctx, g, dn);
        let (e, vu, vd) = (Field::new(ctx, g, "exc"), Field::new(ctx, g, "vu"), Field::new(ctx, g, "vd"));
        xc_lsda(ctx, k, g, &fu, &fd, &e, &vu, &vd);
        for f in [&e, &vu, &vd] { assert_ghosts_zero(g, &f.read_padded_raw(ctx)); }
        XcOut { exc: e.read_interior(ctx, g), v_up: vu.read_interior(ctx, g), v_dn: vd.read_interior(ctx, g) }
    }

    /// Compares GPU XC to `qm::xc::lsda_pw92` end to end (f64 inputs) and kernel-only (f32-rounded
    /// inputs). Brief bound: 1e-5 relative where |ref| > 1e-3, 1e-7 absolute elsewhere; tightened
    /// to 3e-6 / 1e-8. Achieved (RX 7900 XTX): ≤ 2.2e-7 in n·ε_xc, ≤ 1.7e-7 in v on smooth densities,
    /// ≤ 1.5e-6 in v for an empty spin channel at |ζ| = 1 and at n ~ 1e-8 (f32 conditioning of
    /// PW92's ζ interpolation, see xc.wgsl); absolute ≤ 1.6e-9. End-to-end ≈ kernel-only: the f32
    /// rounding of the densities moves v by only ~2e-8 relative.
    fn check_xc(name: &str, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, up: &[f64], dn: &[f64]) -> XcOut {
        let gpu = gpu_xc(ctx, k, g, up, dn);
        let r32 = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| *x as f32 as f64).collect() };
        for (label, cpu) in [("f64 in", qm::xc::lsda_pw92(up, dn)), ("f32 in", qm::xc::lsda_pw92(&r32(up), &r32(dn)))] {
            for (q, got, want) in [("exc", &gpu.exc, &cpu.energy_density), ("v_up", &gpu.v_up, &cpu.v_up), ("v_dn", &gpu.v_dn, &cpu.v_dn)] {
                let (rel, abs, at) = xc_err(got, want);
                println!("{name} [{label}] {q}: max rel {rel:.2e} (at n↑={:.3e}, n↓={:.3e}), max abs (|ref| ≤ 1e-3) {abs:.2e}", up[at], dn[at]);
                assert!(rel <= 3e-6, "{name} {q}: relative error {rel:.3e} at {at} (gpu {}, cpu {})", got[at], want[at]);
                assert!(abs <= 1e-8, "{name} {q}: absolute error {abs:.3e}");
            }
        }
        gpu
    }

    #[test]
    fn xc_matches_cpu_pointwise() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let gauss = cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let mut up: Vec<f64> = gauss.iter().map(|v| 0.6 * v).collect();
        let mut dn: Vec<f64> = gauss.iter().map(|v| 0.4 * v).collect();
        let zero_row: Vec<usize> = (0..cg.n[0]).map(|i| cg.idx(i, 8, 8)).collect(); // through the centre
        for &i in &zero_row { up[i] = 0.0; dn[i] = 0.0; }
        let gpu = check_xc("xc 17³ ζ=0.2", &ctx, &k, &g, &up, &dn);
        for &i in &zero_row {
            assert_eq!((gpu.exc[i], gpu.v_up[i], gpu.v_dn[i]), (0.0, 0.0, 0.0), "vacuum point {i} not exactly zero");
        }
    }

    #[test]
    fn xc_polarised_channel_zero_density() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let up = cg.sample(|[x, y, z]| (-(x * x + y * y + z * z)).exp());
        let dn = vec![0.0; cg.len()];
        let gpu = check_xc("xc 17³ ζ=1", &ctx, &k, &g, &up, &dn);
        assert!(gpu.v_dn.iter().all(|v| v.is_finite()) && gpu.v_dn.iter().any(|v| *v != 0.0), "v↓ must be finite and non-trivial");
    }

    /// Densities 1e-16 … 3e3 (log-spaced) × polarisations ζ ∈ {−1, −0.6, 0, 0.3, 0.999, 1}:
    /// every regime of the PW92 fit (r_s from 0.04 to 1e5) and the ζ interpolation.
    #[test]
    fn xc_density_sweep() {
        let Some(ctx) = GpuContext::new_headless() else { return };
        let cg = qm::Grid::cube(17, 0.3);
        let g = GpuGrid::new(&cg);
        let k = Kernels::new(&ctx);
        let zetas = [-1.0, -0.6, 0.0, 0.3, 0.999, 1.0];
        let per = cg.len() / zetas.len();
        let (mut up, mut dn) = (vec![0.0; cg.len()], vec![0.0; cg.len()]);
        for (zi, z) in zetas.iter().enumerate() {
            for t in 0..per {
                let n = 10f64.powf(-16.0 + 19.5 * t as f64 / per as f64);
                up[zi * per + t] = n * (1.0 + z) / 2.0;
                dn[zi * per + t] = n * (1.0 - z) / 2.0;
            }
        }
        // f32 denormals (a nearly empty spin channel's Σψ² tails): pow(denormal, 1/3) may flush to
        // 0, so cbrt must not divide by it. The rest of the grid (5 points) stays vacuum.
        let tail = zetas.len() * per;
        for (i, (u, d)) in [(0.1, 1e-40), (1e-40, 0.1), (1e-40, 1e-40), (2e-3, 3e-39)].into_iter().enumerate() {
            up[tail + i] = u;
            dn[tail + i] = d;
        }
        let gpu = check_xc("xc sweep", &ctx, &k, &g, &up, &dn);
        for i in tail..tail + 4 {
            assert!(gpu.exc[i].is_finite() && gpu.v_up[i].is_finite() && gpu.v_dn[i].is_finite(), "denormal row {i} not finite");
        }
    }
}
