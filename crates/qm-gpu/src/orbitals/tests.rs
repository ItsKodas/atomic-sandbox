//! Parity of the GPU display products with the `qm::orbitals` f64 references on the same f32
//! data, the single readback, and the no-impact guarantees (plan Task 2). GPU tests skip
//! cleanly without an adapter; the ones that run an SCF are release-only (as the oracles).
use super::*;
use crate::scf::GpuSolver;
use qm::orbitals as cpu;
use qm::{Grid, Nucleus, ScfSettings, System};


fn ctx() -> Option<GpuContext> {
    let c = GpuContext::new_headless();
    if c.is_none() { eprintln!("no GPU adapter: skipping"); }
    c
}

fn norm(grid: &Grid, mut v: Vec<f64>) -> Vec<f64> {
    let s = grid.dot(&v, &v).sqrt();
    for x in v.iter_mut() { *x /= s; }
    v
}

/// Hydrogenic-shaped functions about `c` with exponent scale `z` (normalised on the grid).
fn hydrogenic(grid: &Grid, c: [f64; 3], z: f64, which: &str) -> Vec<f64> {
    let f = grid.sample(|p| {
        let (x, y, w) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        let r = (x * x + y * y + w * w).sqrt();
        match which {
            "1s" => (-z * r).exp(),
            "2s" => (2.0 - z * r) * (-z * r / 2.0).exp(),
            "2px" => x * (-z * r / 2.0).exp(),
            "2py" => y * (-z * r / 2.0).exp(),
            "2pz" => w * (-z * r / 2.0).exp(),
            "3s" => (27.0 - 18.0 * z * r + 2.0 * z * z * r * r) * (-z * r / 3.0).exp(),
            "3dxy" => x * y * (-z * r / 3.0).exp(),
            other => panic!("unknown shape {other}"),
        }
    });
    norm(grid, f)
}

/// A converged-looking `GpuState` from f64 orbitals uploaded as f32: densities Σψ² (from the
/// f32-rounded data), V_H and V_nuc zero. Returns the state and the f32-rounded orbitals as f64
/// (the data the CPU references see).
fn synthetic_state(s: &GpuSolver, up: &[Vec<f64>], dn: &[Vec<f64>], e_up: &[f64], e_dn: &[f64]) -> (GpuState, [Vec<Vec<f64>>; 2]) {
    let grid = s.grid().clone();
    let (c, g) = (&s.ctx, &s.g);
    let upload = |d: &[f64], l: &str| { let f = Field::new(c, g, l); f.upload_interior(c, g, d); f };
    let fields = |set: &[Vec<f64>]| -> Vec<Field> { set.iter().map(|d| upload(d, "orbital")).collect() };
    let (fu, fd) = (fields(up), fields(dn));
    let rounded = |fs: &[Field]| -> Vec<Vec<f64>> { fs.iter().map(|f| f.read_interior(c, g)).collect() };
    let raw = [rounded(&fu), rounded(&fd)];
    let dens = |set: &[Vec<f64>]| -> Vec<f64> {
        let mut r = vec![0.0; grid.len()];
        for o in set { for (a, v) in r.iter_mut().zip(o) { *a += v * v; } }
        r
    };
    let (ru, rd) = (dens(&raw[0]), dens(&raw[1]));
    let rt: Vec<f64> = ru.iter().zip(&rd).map(|(a, b)| a + b).collect();
    let state = GpuState {
        grid: grid.clone(), sigma: grid.h,
        orbitals_up: fu, orbitals_dn: fd, evals_up: e_up.to_vec(), evals_dn: e_dn.to_vec(),
        rho: upload(&rt, "rho"), rho_up: upload(&ru, "rho up"), rho_dn: upload(&rd, "rho dn"),
        v_h: Field::new(c, g, "v_h"), v_nuc: Field::new(c, g, "v_nuc"),
    };
    (state, raw)
}

fn nucleus(pos: [f64; 3], z: u8) -> Nucleus { Nucleus { pos, z } }

/// Σ|ψ·w|·dv over the grid: the natural scale of an f32 partial-sum error of ⟨ψ|w⟩.
fn abs_scale(grid: &Grid, psi: &[f64], w: impl Fn([f64; 3]) -> f64) -> f64 {
    (0..grid.len()).map(|p| (psi[p] * w(grid.point(p))).abs()).sum::<f64>() * grid.dv()
}

fn solid(d: [f64; 3]) -> [f64; 16] {
    let mut s = [0.0; qm::poisson::N_MOMENTS];
    qm::poisson::solid_harmonics(d, &mut s);
    std::array::from_fn(|i| s[i])
}

fn unit_y(d: [f64; 3]) -> [f64; 16] {
    let s = solid(d);
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let mut y = [0.0; 16];
    if r == 0.0 { y[0] = 1.0 / (4.0 * std::f64::consts::PI).sqrt(); return y; }
    for (i, v) in y.iter_mut().enumerate() {
        let l = (i as f64).sqrt().floor() as i32;
        *v = ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt() * s[i] / r.powi(l);
    }
    y
}

#[test]
fn abs_max_reduction_is_exact() {
    let Some(ctx) = ctx() else { return };
    let k = Kernels::new(&ctx);
    for n in [13usize, 29] { // 29: more points than one workgroup per partial; both odd
        let cg = Grid::cube(n, 0.3);
        let g = GpuGrid::new(&cg);
        let r = Reducer::new(&ctx, &k, &g);
        // a smooth field with its extremes on padding-adjacent interior points: the most
        // negative at the (0, 0, 0) corner, the largest positive at (n−1, n−1, n−1), and a
        // second, slightly smaller, negative at (n−1, 0, 0)
        let mut a = cg.sample(|[x, y, z]| 0.3 * (x + 0.5 * y - 0.2 * z).sin());
        a[cg.idx(0, 0, 0)] = -7.25;
        a[cg.idx(n - 1, n - 1, n - 1)] = 7.0;
        a[cg.idx(n - 1, 0, 0)] = -6.5;
        let mut b = a.clone(); // positive extreme dominates
        b[cg.idx(n - 1, n - 1, n - 1)] = 7.75;
        let up = |d: &[f64]| { let f = Field::new(&ctx, &g, "a"); f.upload_interior(&ctx, &g, d); f };
        let (fa, fb) = (up(&a), up(&b));
        let v = r.many(&ctx, &k, &g, &[Reduction::AbsMax(&fa), Reduction::AbsMax(&fb), Reduction::Max(&fa)]);
        let exact = |d: &[f64]| d.iter().map(|x| (*x as f32).abs()).fold(0.0f32, f32::max) as f64;
        println!("abs max {n}³: {} / {} (cpu {} / {}), max {}", v[0], v[1], exact(&a), exact(&b), v[2]);
        assert_eq!(v[0], exact(&a), "AbsMax with the extreme negative on the corner");
        assert_eq!(v[1], exact(&b), "AbsMax with the extreme positive on the far corner");
        assert_eq!(v[0], 7.25);
        assert_eq!(v[2], 7.0, "Max is unchanged by the new variant");
    }
}

/// b, c, the nine windowed first-sign functionals and the outer-L/6 weight of hydrogenic 1s,
/// 2p_{x,y,z} and 3d_xy at an off-grid centre (lone atom), against the `qm::orbitals` f64
/// functions on the same f32 data. Errors are relative to Σ|ψ·w|·dv, the scale an f32 partial
/// sum is accurate to (a vanishing moment has no meaningful relative error of its own).
#[test]
fn moments_match_the_cpu_reference() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(29, 0.35);
    let h = grid.h;
    let centre = [0.37 * h, -0.21 * h, 0.12 * h];
    let s = GpuSolver::new(ctx, &grid);
    let shapes = ["1s", "2px", "2py", "2pz", "3dxy"];
    let orbs: Vec<Vec<f64>> = shapes.iter().map(|w| hydrogenic(&grid, centre, 2.0, w)).collect();
    let evals = [-2.0, -0.5, -0.5, -0.5, -0.22];
    let (state, raw) = synthetic_state(&s, &orbs, &[], &evals, &[]);
    let nuclei = [nucleus(centre, 2)];
    let mut view = OrbitalView::new(&s);
    let rb = view.read_scalars(&s, &state, &nuclei);
    let gm = &rb.moments[0];
    assert_eq!(gm.len(), shapes.len());
    let (mut worst_b, mut worst_c, mut worst_f, mut worst_o) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (o, name) in shapes.iter().enumerate() {
        let psi = &raw[0][o];
        let m = cpu::moments(&grid, psi, centre);
        for lm in 0..16 {
            let sb = abs_scale(&grid, psi, |p| solid([p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]])[lm]);
            let sc = abs_scale(&grid, psi, |p| unit_y([p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]])[lm]);
            worst_b = worst_b.max((gm[o].b[lm] - m.b[lm]).abs() / sb);
            worst_c = worst_c.max((gm[o].c[lm] - m.c[lm]).abs() / sc);
        }
        let f = cpu::first_sign_functionals(&grid, psi, &[centre]);
        for (i, (fg, fc)) in rb.first_sign[0][o].iter().zip(&f).enumerate() {
            // scale: Σ|ψ w_i| dv / ‖w_i‖
            let w = |p: [f64; 3]| -> f64 {
                let d = [p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]];
                let g = (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).exp();
                match i { 0 => g, 1..=3 => d[i - 1] * g, _ => solid(d)[i] * g }
            };
            let wn = (grid.sample(w).iter().map(|v| v * v).sum::<f64>() * grid.dv()).sqrt();
            worst_f = worst_f.max((fg - fc).abs() / (abs_scale(&grid, psi, w) / wn));
        }
        let ow = cpu::outer_weight(&grid, psi);
        let og = rb.outer_weight[0][o];
        worst_o = worst_o.max((og - ow).abs() / ow);
        println!("{name}: W_l gpu {:?} cpu {:?}; outer {og:.6e} vs {ow:.6e}; sign fns {:?}",
                 cpu::character(&gm[o]), cpu::character(&m), rb.first_sign[0][o].map(|v| (v * 1e4).round() / 1e4));
        // the angular character the labels use agrees to far better than the 0.9 threshold
        for (a, b) in cpu::character(&gm[o]).iter().zip(cpu::character(&m)) { assert!((a - b).abs() < 1e-6, "{name} W_l"); }
    }
    // f32 bound: (points per thread + tree depth 8 + harmonic rounding ~8) · u, here
    // ≈ (24389/(96·256) + 16) · 6e-8 ≈ 1e-6; the brief's 1e-5 has 10× headroom.
    println!("moments 29³ max err / Σ|terms|: b {worst_b:.2e}, c {worst_c:.2e}, first-sign {worst_f:.2e}; outer rel {worst_o:.2e}");
    for (what, e) in [("b", worst_b), ("c", worst_c), ("first-sign", worst_f), ("outer weight", worst_o)] {
        assert!(e <= 1e-5, "{what}: {e:.3e} > 1e-5");
    }
}

/// The radial-shell profiles (lone atom; ruling 1): ν exactly, R̄_k within the f32 tolerance on
/// bins whose point count agrees (f32 r/dr may move a boundary point across a bin edge, so the
/// kernel's own counts are used for the GPU's R̄_k, as the plan says).
#[test]
fn radial_profiles_match_the_cpu_reference() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(41, 0.3);
    let h = grid.h;
    let centre = [0.37 * h, -0.21 * h, 0.12 * h];
    let s = GpuSolver::new(ctx, &grid);
    let specs = [("1s", 2.0, 0u8), ("2s", 2.0, 0), ("2px", 2.0, 1), ("3s", 3.0, 0), ("3dxy", 3.0, 2)];
    let orbs: Vec<Vec<f64>> = specs.iter().map(|(w, z, _)| hydrogenic(&grid, centre, *z, w)).collect();
    let (state, raw) = synthetic_state(&s, &orbs, &[], &[-2.0, -0.5, -0.5, -0.4, -0.3], &[]);
    let mut view = OrbitalView::new(&s);
    let rb = view.read_scalars(&s, &state, &[nucleus(centre, 3)]);
    let gp = &rb.radial.as_ref().expect("lone atom: radial profiles")[0];
    let mut moved = 0usize;
    let mut worst = 0.0f64;
    for (o, (name, _, l)) in specs.iter().enumerate() {
        let cp = cpu::radial_profile(&grid, &raw[0][o], centre);
        let cm = cpu::moments(&grid, &raw[0][o], centre);
        let gm = &rb.moments[0][o];
        assert_eq!(gp[o].a.len(), cp.a.len(), "{name}: bin count");
        assert_eq!(gp[o].dr, cp.dr);
        let nu_c = cpu::profile_nodes(&cp, &cm, *l);
        let nu_g = cpu::profile_nodes(&gp[o], gm, *l);
        // R̄_k with each side's own c direction and counts
        let rbar = |p: &cpu::RadialProfile, m: &cpu::Moments| -> Vec<f64> {
            let lo = (*l as usize).pow(2);
            let d = 2 * *l as usize + 1;
            let nrm = m.c[lo..lo + d].iter().map(|x| x * x).sum::<f64>().sqrt();
            p.a.iter().zip(&p.count).map(|(a, &n)| if n == 0 { 0.0 } else { (0..d).map(|i| m.c[lo + i] / nrm * a[lo + i]).sum::<f64>() / n as f64 }).collect()
        };
        let (rg, rc) = (rbar(&gp[o], gm), rbar(&cp, &cm));
        let max = rc.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for kk in 0..rc.len() {
            if gp[o].count[kk] != cp.count[kk] { if o == 0 { moved += gp[o].count[kk].abs_diff(cp.count[kk]) as usize; } continue; }
            worst = worst.max((rg[kk] - rc[kk]).abs() / max);
        }
        println!("{name}: ν gpu {nu_g:?} cpu {nu_c:?}; {} bins", cp.a.len());
        assert_eq!(nu_g, nu_c, "{name}: radial node count");
        assert!(nu_c.is_some());
    }
    let total_c: u32 = cpu::radial_profile(&grid, &raw[0][0], centre).count.iter().sum();
    let total_g: u32 = gp[0].count.iter().sum();
    println!("radial 41³: max |ΔR̄|/max|R̄| = {worst:.2e} on bins with equal counts; boundary points that changed bin: {moved}; points inside: gpu {total_g} cpu {total_c}");
    assert!(worst <= 1e-5, "R̄_k: {worst:.3e}");
    assert!(moved <= 8, "{moved} points changed bin (f32 r/dr at bin edges)");
    assert!(total_g.abs_diff(total_c) <= 4);
    // the expected nodes: 1s 0, 2s 1, 2p 0, 3s 2, 3d 0
    let nus: Vec<Option<usize>> = specs.iter().enumerate().map(|(o, (_, _, l))| cpu::profile_nodes(&gp[o], &rb.moments[0][o], *l)).collect();
    assert_eq!(nus, vec![Some(0), Some(1), Some(0), Some(2), Some(0)]);
}

/// Task 2 review: the bin partition at 97³, where `bin_split` = 10 workgroups share each bin's
/// columns (the 41³ test above has 2). Σ n_k — the points inside the inscribed sphere — must be
/// the CPU's exactly (no point dropped or counted twice by the column split), and the per-bin
/// counts may differ only by points within f32 rounding of a bin edge.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn radial_bin_counts_partition_the_sphere_exactly_at_97() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(97, 0.25);
    let h = grid.h;
    let centre = [0.37 * h, -0.21 * h, 0.12 * h];
    let s = GpuSolver::new(ctx, &grid);
    let orb = hydrogenic(&grid, centre, 1.0, "1s");
    let (state, raw) = synthetic_state(&s, &[orb], &[], &[-0.5], &[]);
    let mut view = OrbitalView::new(&s);
    let rb = view.read_scalars(&s, &state, &[nucleus(centre, 1)]);
    let gp = &rb.radial.as_ref().expect("lone atom: radial profiles")[0][0];
    let cp = cpu::radial_profile(&grid, &raw[0][0], centre);
    assert_eq!(gp.count.len(), cp.count.len(), "bin count");
    let (tg, tc): (u64, u64) = (gp.count.iter().map(|&c| c as u64).sum(), cp.count.iter().map(|&c| c as u64).sum());
    let moved: u64 = gp.count.iter().zip(&cp.count).map(|(a, b)| a.abs_diff(*b) as u64).sum();
    let n_bins = cp.count.len();
    println!("97³ radial bins: {n_bins} bins; Σn_k gpu {tg} cpu {tc}; Σ|Δn_k| = {moved} (boundary points that changed bin)");
    assert!(n_bins > 90, "the 97³ inscribed sphere has ~96 bins of h/2: {n_bins}");
    assert_eq!(tg, tc, "Σ n_k: the column split must partition the sphere exactly");
    assert!(moved <= 16, "{moved} points changed bin (f32 r/dr at bin edges)");
}

/// 6 synthetic orbitals (3 ↑, 3 ↓, a molecule): after one update has set the displayed set,
/// the overlaps of a perturbed set with it, the cross-channel overlaps and max|ψ| against the
/// CPU on the same f32 data.
#[test]
fn overlaps_and_max_match_the_cpu() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(27, 0.35);
    let s = GpuSolver::new(ctx, &grid);
    let (a, b) = ([0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]);
    let nuclei = [nucleus(a, 1), nucleus(b, 1)];
    let set = |shift: f64| -> Vec<Vec<f64>> {
        let a2 = [a[0] + shift, 0.1, 0.0];
        vec![hydrogenic(&grid, a2, 1.5, "1s"), hydrogenic(&grid, b, 1.5, "2px"), hydrogenic(&grid, a2, 1.2, "2py")]
    };
    let (s0, _) = synthetic_state(&s, &set(0.0), &set(0.05), &[-0.6, -0.3, -0.2], &[-0.6, -0.3, -0.2]);
    let mut view = OrbitalView::new(&s);
    let first = view.update(&s, &s0, &nuclei, 1e-6, Product::None);
    assert_eq!(first.readbacks, 1);
    let shown: [Vec<Vec<f64>>; 2] = [0, 1].map(|ch| view.prev[ch].iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect());
    let (s1, raw) = synthetic_state(&s, &set(0.1), &set(0.13), &[-0.6, -0.3, -0.2], &[-0.6, -0.3, -0.2]);
    let rb = view.read_scalars(&s, &s1, &nuclei);
    let mut worst = 0.0f64;
    for ch in 0..2 {
        let o = rb.overlaps_prev[ch].as_ref().expect("a previous display exists");
        assert_eq!((o.nrows(), o.ncols()), (3, 3));
        for i in 0..3 { for j in 0..3 {
            worst = worst.max((o[(i, j)] - grid.dot(&raw[ch][i], &shown[ch][j])).abs());
        }}
        for (i, m) in rb.max_abs[ch].iter().enumerate() {
            let exact = raw[ch][i].iter().fold(0.0f64, |a, v| a.max(v.abs()));
            assert_eq!(*m, exact, "max|ψ| ch {ch} orbital {i}");
        }
    }
    let x = rb.cross.as_ref().expect("both channels occupied");
    for i in 0..3 { for j in 0..3 { worst = worst.max((x[(i, j)] - grid.dot(&raw[0][i], &raw[1][j])).abs()); } }
    println!("overlaps 27³: max |O_gpu − O_cpu| = {worst:.2e}");
    assert!(worst <= 1e-6, "{worst:.3e}");
    assert!(rb.radial.is_none(), "a molecule has no radial profiles");
}

/// A lone atom's 1s and a rotated, sign-flipped 2p set: the aligned display copied to the
/// texture equals `qm::orbitals::apply_rotation` of the same f32 data.
#[test]
fn display_copy_matches_cpu_reference() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(31, 0.3);
    let h = grid.h;
    let centre = [0.37 * h, -0.21 * h, 0.12 * h];
    let s = GpuSolver::new(ctx, &grid);
    let p: Vec<Vec<f64>> = ["2px", "2py", "2pz"].iter().map(|w| hydrogenic(&grid, centre, 2.0, w)).collect();
    // an orthogonal R with max|R − Rᵀ| well away from 0, and a sign flip on its second member
    let (ca, sa, cb, sb) = (0.6f64.cos(), 0.6f64.sin(), 1.1f64.cos(), 1.1f64.sin());
    let rz = [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]];
    let r: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|q| rz[i][q] * rx[q][j]).sum()));
    let rotated: Vec<Vec<f64>> = (0..3).map(|j| {
        let sign = if j == 1 { -1.0 } else { 1.0 };
        (0..grid.len()).map(|q| sign * (0..3).map(|a| r[j][a] * p[a][q]).sum::<f64>()).collect()
    }).collect();
    let mut up = vec![hydrogenic(&grid, centre, 2.0, "1s")];
    up.extend(rotated);
    let (state, raw) = synthetic_state(&s, &up, &[], &[-2.0, -0.5, -0.5 + 1e-4, -0.5 + 3e-4], &[]);
    let nuclei = [nucleus(centre, 2)];
    let mut view = OrbitalView::new(&s);
    let vol = Volume3D::new(&s.ctx, [grid.n[0] as u32; 3]);
    let probe = view.update(&s, &state, &nuclei, 1e-6, Product::None);
    assert!(matches!(probe.out.lone_atom, cpu::LoneAtom::Labelled { .. }), "{:?}", probe.out.lone_atom);
    let (rng, u) = &probe.out.rotations[0][1];
    assert_eq!(rng.clone(), 1..4, "the p set is one aligned shell");
    let off = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| (u[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
    assert!(off > 0.3, "the injected rotation must be undone by a non-trivial U (max|U − I| = {off})");
    let mut worst = 0.0f64;
    for row in probe.out.rows.iter().filter(|r| r.spin == Spin::Up) {
        let upd = view.update(&s, &state, &nuclei, 1e-6, Product::Orbital { spin: Spin::Up, track: row.track, squared: false, out: &vol });
        assert_eq!(upd.readbacks, 1);
        let want = &cpu::apply_rotation(&raw[0], &upd.out, Spin::Up)[row.index];
        let got = vol.read_back(&s.ctx).expect("readback");
        let mx = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let d = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max) / mx;
        assert!((upd.scale as f64 - upd.out.scale[0][row.index]).abs() <= 1e-7 * upd.out.scale[0][row.index]);
        println!("{}: max |texture − apply_rotation| / max|ψ| = {d:.2e}", row.label);
        worst = worst.max(d);
    }
    let labels: Vec<String> = probe.out.rows.iter().map(|r| r.label.to_string()).collect();
    assert_eq!(labels, ["1s", "2p_x", "2p_y", "2p_z"]);
    assert!(worst <= 1e-6, "{worst:.3e}");
}

/// Σ (signed texture)² h³ = 1 and Σ (squared texture) h³ = 1 for the H atom's 1s (spec §11).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn h_atom_display_texture_is_normalised() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(49, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let mut s = GpuSolver::new(ctx, &grid);
    let nuclei = vec![nucleus([0.0; 3], 1)];
    s.solve(&System::new(nuclei.clone(), 0, None), &settings, false).expect("H converges");
    let state = s.state().expect("converged");
    let mut view = OrbitalView::new(&s);
    let vol = Volume3D::new(&s.ctx, [49; 3]);
    let probe = view.update(&s, state, &nuclei, settings.tol, Product::None);
    let track = probe.out.rows[0].track;
    let dv = grid.dv();
    let phase = view.update(&s, state, &nuclei, settings.tol, Product::Orbital { spin: Spin::Up, track, squared: false, out: &vol });
    let t = vol.read_back(&s.ctx).expect("readback");
    let n2 = t.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() * dv;
    let max = t.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let sq = view.update(&s, state, &nuclei, settings.tol, Product::Orbital { spin: Spin::Up, track, squared: true, out: &vol });
    let ts = vol.read_back(&s.ctx).expect("readback");
    let n1 = ts.iter().map(|v| *v as f64).sum::<f64>() * dv;
    println!("H 49³: Σψ̃²h³ = {n2:.9}, Σ(ψ̃²)h³ = {n1:.9}; scale {} (texture max {max}), squared scale {}; label {}",
             phase.scale, sq.scale, phase.out.rows[0].label);
    assert!((n2 - 1.0).abs() <= 1e-4, "{n2}");
    assert!((n1 - 1.0).abs() <= 1e-4, "{n1}");
    assert_eq!(phase.scale, max, "phase scale is max|ψ| of the (single-member) set");
    assert_eq!(sq.scale, phase.scale * phase.scale);
    assert!(t.iter().sum::<f32>() > 0.0, "sign rule: the 1s is positive");
}

#[test]
#[cfg_attr(debug_assertions, ignore)]
fn spin_texture_integrates_to_the_unpaired_count() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(49, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let mut s = GpuSolver::new(ctx, &grid);
    let nuclei = vec![nucleus([0.0; 3], 1)];
    s.solve(&System::new(nuclei.clone(), 0, None), &settings, false).expect("H converges");
    let state = s.state().expect("converged");
    let mut view = OrbitalView::new(&s);
    let vol = Volume3D::new(&s.ctx, [49; 3]);
    let upd = view.update(&s, state, &nuclei, settings.tol, Product::Spin { out: &vol });
    let t = vol.read_back(&s.ctx).expect("readback");
    let sum = t.iter().map(|v| *v as f64).sum::<f64>() * grid.dv();
    let abs = t.iter().map(|v| v.abs() as f64).sum::<f64>() * grid.dv();
    let max = t.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    println!("H spin: texture ∫m {sum:.9}, ∫|m| {abs:.9}, max|m| {max}; summary {:?}, scale {}", upd.spin, upd.scale);
    assert!((sum - 1.0).abs() <= 1e-4, "{sum}");
    assert!((upd.spin.integral - sum).abs() <= 1e-6, "{} vs {sum}", upd.spin.integral);
    assert!((upd.spin.abs_integral - abs).abs() <= 1e-6);
    assert_eq!(upd.spin.max_abs, max as f64);
    assert_eq!((upd.spin.n_up, upd.spin.n_dn), (1, 0));
    assert_eq!(upd.scale, upd.spin.scale() as f32);
}

/// Two orbitals, per-orbital and physical scaling: the rgba16float volume against
/// `qm::orbitals::colour_volume` of the displayed (f32) orbitals.
#[test]
fn colour_volume_matches_the_cpu_reference() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(25, 0.35);
    let s = GpuSolver::new(ctx, &grid);
    let (a, b) = ([0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]);
    let nuclei = [nucleus(a, 1), nucleus(b, 1)];
    let up = vec![hydrogenic(&grid, a, 1.6, "1s"), hydrogenic(&grid, b, 1.3, "2pz")];
    let (state, _) = synthetic_state(&s, &up, &[], &[-0.6, -0.2], &[]);
    let mut view = OrbitalView::new(&s);
    let vol = ColourVolume3D::new(&s.ctx, [25; 3]);
    let probe = view.update(&s, &state, &nuclei, 1e-6, Product::None);
    let include: Vec<u32> = probe.out.rows.iter().map(|r| r.track).collect();
    for physical in [false, true] {
        let upd = view.update(&s, &state, &nuclei, 1e-6, Product::Colour { include: &include, physical, out: &vol });
        assert_eq!(upd.readbacks, 1);
        let shown: Vec<Vec<f64>> = view.prev[0].iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect();
        let rho_max = upd.spin.rho_max;
        let terms: Vec<(&[f64], f64, [f64; 3], f64)> = upd.out.rows.iter().map(|r| {
            let c = ORBITAL_PALETTE[r.slot as usize].map(|v| v as f64);
            let pk = if physical { rho_max } else { upd.out.scale[0][r.index].powi(2) };
            (shown[r.index].as_slice(), r.occupation, c, pk)
        }).collect();
        let want = cpu::colour_volume(&grid, &terms);
        let got = vol.read_back(&s.ctx).expect("readback");
        let (mut worst, mut checked) = (0.0f64, 0usize);
        for (g4, w4) in got.iter().zip(&want) {
            if w4[3] <= 1e-3 { continue; }
            checked += 1;
            for c in 0..4 { worst = worst.max((g4[c] as f64 - w4[c]).abs() / (w4[c].abs() + 1e-7 / 2e-3)); }
        }
        let amax = want.iter().fold(0.0f64, |m, v| m.max(v[3]));
        println!("colour 25³ physical={physical}: max rel err {worst:.2e} over {checked} points with a > 1e-3 (max a {amax:.3}), scale {}", upd.scale);
        assert!(checked > 100);
        assert!(worst <= 2e-3, "{worst:.3e}");
        if !physical { assert!((amax - 1.0).abs() < 0.05, "per-orbital: each orbital peaks near 1, max a = {amax}"); }
    }
}

/// One scalar readback per update, for every product, lone atom and molecule.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn one_readback_per_update() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(33, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let mut s = GpuSolver::new(ctx, &grid);
    let vol = Volume3D::new(&s.ctx, [33; 3]);
    let cvol = ColourVolume3D::new(&s.ctx, [33; 3]);
    for nuclei in [vec![nucleus([0.1, 0.0, 0.0], 3)], vec![nucleus([0.7, 0.0, 0.0], 1), nucleus([-0.7, 0.0, 0.0], 1)]] {
        s.solve(&System::new(nuclei.clone(), 0, None), &settings, false).expect("converges");
        let state = s.state().expect("converged");
        let mut view = OrbitalView::new(&s);
        let first = view.update(&s, state, &nuclei, settings.tol, Product::None);
        let tracks: Vec<u32> = first.out.rows.iter().map(|r| r.track).collect();
        let (spin, track) = (first.out.rows[0].spin, first.out.rows[0].track);
        let products = [Product::None, Product::Orbital { spin, track, squared: false, out: &vol }, Product::Orbital { spin, track, squared: true, out: &vol },
                        Product::Colour { include: &tracks, physical: false, out: &cvol }, Product::Colour { include: &tracks, physical: true, out: &cvol },
                        Product::Spin { out: &vol }];
        for p in products {
            let what = format!("{p:?}");
            let u = view.update(&s, state, &nuclei, settings.tol, p);
            assert_eq!(u.readbacks, 1, "{} nuclei, {what}", nuclei.len());
        }
        println!("{} nuclei: {} rows, lone atom {:?}", nuclei.len(), first.out.rows.len(), first.out.lone_atom);
    }
}

/// Every `GpuState` field is byte-identical after three updates of each product.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn update_does_not_write_solver_fields() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(33, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let mut s = GpuSolver::new(ctx, &grid);
    let vol = Volume3D::new(&s.ctx, [33; 3]);
    let cvol = ColourVolume3D::new(&s.ctx, [33; 3]);
    for nuclei in [vec![nucleus([0.7, 0.0, 0.0], 1), nucleus([-0.7, 0.0, 0.0], 1)], vec![nucleus([0.1, -0.05, 0.0], 3)]] {
        s.solve(&System::new(nuclei.clone(), 0, None), &settings, false).expect("converges");
        let state = s.state().expect("converged");
        let snapshot = |st: &GpuState| -> Vec<Vec<f32>> {
            st.orbitals_up.iter().chain(&st.orbitals_dn).chain([&st.rho, &st.rho_up, &st.rho_dn, &st.v_h, &st.v_nuc])
                .map(|f| f.read_padded_raw(&s.ctx)).collect()
        };
        let before = snapshot(state);
        let evals = (state.evals_up.clone(), state.evals_dn.clone());
        let mut view = OrbitalView::new(&s);
        for _ in 0..3 {
            let first = view.update(&s, state, &nuclei, settings.tol, Product::None);
            let tracks: Vec<u32> = first.out.rows.iter().map(|r| r.track).collect();
            let (spin, track) = (first.out.rows[0].spin, first.out.rows[0].track);
            for p in [Product::Orbital { spin, track, squared: false, out: &vol }, Product::Orbital { spin, track, squared: true, out: &vol },
                      Product::Colour { include: &tracks, physical: false, out: &cvol }, Product::Colour { include: &tracks, physical: true, out: &cvol },
                      Product::Spin { out: &vol }] {
                view.update(&s, state, &nuclei, settings.tol, p);
            }
        }
        gpu::wait_idle(&s.ctx).expect("wait idle");
        let after = snapshot(state);
        assert_eq!(before.len(), after.len());
        for (i, (a, b)) in before.iter().zip(&after).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{} nuclei: solver field {i} changed", nuclei.len());
        }
        assert_eq!(evals, (state.evals_up.clone(), state.evals_dn.clone()));
        println!("{} nuclei: {} solver fields byte-identical after 18 updates", nuclei.len(), before.len());
    }
}

/// Display products between SCF solves leave the SCF bitwise unchanged: two fresh solvers on
/// the same three-geometry warm trajectory, one with every product run after each solve,
/// give identical energies, Δρ, step counts and forces (`==` on the bits; `solve_is_deterministic`
/// makes the comparison meaningful).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn display_updates_between_solves_leave_the_scf_bitwise_unchanged() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(33, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let geometry = |t: usize| -> Vec<Nucleus> { let d = 0.7 + 0.03 * t as f64; vec![nucleus([d, 0.01 * t as f64, 0.0], 1), nucleus([-d, 0.0, 0.0], 1)] };
    let run = |with_view: bool| -> Vec<(u64, u64, usize, Vec<[u64; 3]>)> {
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        let vol = Volume3D::new(&s.ctx, [33; 3]);
        let cvol = ColourVolume3D::new(&s.ctx, [33; 3]);
        let mut view: Option<OrbitalView> = None;
        let mut out = Vec::new();
        for t in 0..3 {
            let nuclei = geometry(t);
            let (_, st) = s.solve(&System::new(nuclei.clone(), 0, None), &settings, t > 0).expect("converges");
            let state = s.state().expect("converged");
            let f = s.forces(state, &nuclei, settings.sigma).expect("forces");
            out.push((st.energy.total.to_bits(), st.delta_rho.to_bits(), st.iterations, f.iter().map(|v| v.map(f64::to_bits)).collect()));
            if with_view {
                let v = view.get_or_insert_with(|| OrbitalView::new(&s));
                let first = v.update(&s, state, &nuclei, settings.tol, Product::None);
                let tracks: Vec<u32> = first.out.rows.iter().map(|r| r.track).collect();
                let track = first.out.rows[0].track;
                v.update(&s, state, &nuclei, settings.tol, Product::Orbital { spin: Spin::Up, track, squared: true, out: &vol });
                v.update(&s, state, &nuclei, settings.tol, Product::Colour { include: &tracks, physical: true, out: &cvol });
                v.update(&s, state, &nuclei, settings.tol, Product::Spin { out: &vol });
            }
        }
        out
    };
    let (off, on) = (run(false), run(true));
    for (t, (a, b)) in off.iter().zip(&on).enumerate() {
        println!("step {t}: E bits {:x} / {:x}, {} / {} SCF steps", a.0, b.0, a.2, b.2);
        assert_eq!(a, b, "step {t}: the SCF changed with the view on");
    }
}

/// Timing for the report (contended GPU; numbers reported, never asserted):
/// `cargo test -p qm-gpu --release -- --ignored --nocapture bench_orbital_update`.
#[test]
#[ignore]
fn bench_orbital_update() {
    let Some(ctx) = ctx() else { return };
    for n in [49usize, 97] {
        let grid = Grid::cube(n, 0.25);
        let s = GpuSolver::new(ctx.clone(), &grid);
        let c = [0.37 * grid.h, 0.0, 0.0];
        let shapes = [("1s", 8.0), ("2s", 8.0), ("2px", 8.0), ("2py", 8.0), ("2pz", 8.0)];
        let set: Vec<Vec<f64>> = shapes.iter().map(|(w, z)| hydrogenic(&grid, c, *z, w)).collect();
        let e = [-20.0, -1.0, -0.8, -0.8, -0.8];
        let (state, _) = synthetic_state(&s, &set, &set, &e, &e);
        let vol = Volume3D::new(&s.ctx, [n as u32; 3]);
        let cvol = ColourVolume3D::new(&s.ctx, [n as u32; 3]);
        for (what, nuclei) in [("lone atom (radial profiles)", vec![nucleus(c, 10)]), ("molecule", vec![nucleus(c, 5), nucleus([-1.0, 0.0, 0.0], 5)])] {
            let mut view = OrbitalView::new(&s);
            let first = view.update(&s, &state, &nuclei, 1e-5, Product::None);
            let tracks: Vec<u32> = first.out.rows.iter().map(|r| r.track).collect();
            let track = first.out.rows[2].track;
            let sync = || { gpu::wait_idle(&s.ctx).expect("wait idle"); };
            for (pname, p) in [("none", Product::None), ("orbital", Product::Orbital { spin: Spin::Up, track, squared: false, out: &vol }),
                               ("colour", Product::Colour { include: &tracks, physical: false, out: &cvol }), ("spin", Product::Spin { out: &vol })] {
                for _ in 0..3 { view.update(&s, &state, &nuclei, 1e-5, p); sync(); }
                let reps = 10;
                let t = std::time::Instant::now();
                for _ in 0..reps { view.update(&s, &state, &nuclei, 1e-5, p); sync(); }
                println!("{n}³ K = 10, {what}, product {pname}: {:.2} ms per update (incl. GPU completion)", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
            }
        }
    }
}

/// The CPU step fed with the GPU scalars gives what the whole `qm::orbitals` f64 pipeline gives
/// on the same f32 orbitals (read back): real SCF states of a molecule (H₂), an open-shell lone
/// atom (Li) and a labelled lone atom with a p shell (Ne, 41³). Same lone-atom verdict, labels,
/// tracks and slots; rotations to 1e-6; scales exact.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn cpu_step_on_gpu_scalars_matches_the_cpu_pipeline() {
    let Some(ctx) = ctx() else { return };
    for (what, n, nuclei) in [("H2", 33usize, vec![nucleus([0.7, 0.0, 0.0], 1), nucleus([-0.7, 0.0, 0.0], 1)]),
                              ("Li", 33, vec![nucleus([0.1, 0.0, 0.0], 3)]),
                              ("Ne", 41, vec![nucleus([0.37 * 0.25, -0.21 * 0.25, 0.12 * 0.25], 10)])] {
        let grid = Grid::cube(n, 0.25);
        let settings = ScfSettings::for_grid(&grid);
        let mut s = GpuSolver::new(ctx.clone(), &grid);
        if let Err(e) = s.solve(&System::new(nuclei.clone(), 0, None), &settings, false) { panic!("{what}: {e:?}"); }
        let state = s.state().expect("converged");
        let mut view = OrbitalView::new(&s);
        let gpu = view.update(&s, state, &nuclei, settings.tol, Product::None);
        // the CPU pipeline on the read-back orbitals
        let raw = [state.orbitals_up.iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect::<Vec<_>>(),
                   state.orbitals_dn.iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect::<Vec<_>>()];
        let pos: Vec<[f64; 3]> = nuclei.iter().map(|n| n.pos).collect();
        let lone = nuclei.len() == 1;
        let refs = |ch: usize| raw[ch].iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        let ms = [0, 1].map(|ch| cpu::moments_many(&grid, &refs(ch), pos[0]));
        let rp = [0, 1].map(|ch| cpu::radial_profiles(&grid, &refs(ch), pos[0]));
        let fs = [0, 1].map(|ch| raw[ch].iter().map(|p| cpu::first_sign_functionals(&grid, p, &pos)).collect::<Vec<_>>());
        let mx = [0, 1].map(|ch| raw[ch].iter().map(|p| p.iter().fold(0.0f64, |a, v| a.max(v.abs()))).collect::<Vec<_>>());
        let ow = [0, 1].map(|ch| raw[ch].iter().map(|p| cpu::outer_weight(&grid, p)).collect::<Vec<_>>());
        let cross = (!raw[1].is_empty()).then(|| nalgebra::DMatrix::from_fn(raw[0].len(), raw[1].len(), |i, j| grid.dot(&raw[0][i], &raw[1][j])));
        let n_occ = [raw[0].len(), raw[1].len()];
        let occ = [vec![1.0; n_occ[0]], vec![1.0; n_occ[1]]];
        let input = cpu::UpdateInput {
            evals: [&state.evals_up, &state.evals_dn], occupations: [&occ[0], &occ[1]], overlaps_prev: [None, None], cross: cross.as_ref(),
            moments: lone.then(|| [&ms[0][..], &ms[1][..]]), radial: lone.then(|| [&rp[0][..], &rp[1][..]]),
            first_sign: [&fs[0], &fs[1]], max_abs: [&mx[0], &mx[1]], outer_weight: [&ow[0], &ow[1]],
            n_occ, n_computed: n_occ, n_core: [0; 4], z_sigma: lone.then(|| nuclei[0].z as f64 * state.sigma), tol_scf: settings.tol, unoccupied: None, face_limit: f64::INFINITY,
        };
        let want = cpu::Tracker::new().update(&input);
        let labels = |o: &cpu::UpdateOutput| o.rows.iter().map(|r| format!("{}{}", r.label, if r.spin == Spin::Up { "↑" } else { "↓" })).collect::<Vec<_>>();
        println!("{what}: gpu {:?} {:?}; cpu {:?} {:?}; ε↑ {:.4?}; W_l per ↑ orbital {:?}", gpu.out.lone_atom, labels(&gpu.out), want.lone_atom, labels(&want),
                 state.evals_up, view_chars(&ms[0]));
        assert_eq!(gpu.out.lone_atom, want.lone_atom, "{what}");
        assert_eq!(gpu.out.rows.iter().map(|r| (r.spin, r.index, r.track, r.slot, r.label.clone(), r.rotated)).collect::<Vec<_>>(),
                   want.rows.iter().map(|r| (r.spin, r.index, r.track, r.slot, r.label.clone(), r.rotated)).collect::<Vec<_>>(), "{what}");
        for ch in 0..2 {
            assert_eq!(gpu.out.scale[ch], want.scale[ch], "{what}: scales (max|ψ| is exact)");
            for ((ra, ua), (rb, ub)) in gpu.out.rotations[ch].iter().zip(&want.rotations[ch]) {
                assert_eq!(ra, rb);
                let d = (ua - ub).abs().max();
                assert!(d <= 1e-6, "{what}: rotation {ra:?} differs by {d:.2e}");
            }
        }
        if what == "Ne" { assert!(matches!(gpu.out.lone_atom, cpu::LoneAtom::Labelled { .. }), "Ne is labelled: {:?}", gpu.out.lone_atom); }
    }
}

fn view_chars(ms: &[cpu::Moments]) -> Vec<String> {
    ms.iter().map(|m| format!("{:.3?}", cpu::character(m))).collect()
}

/// More than `COLOUR_MAX_TERMS` included orbitals chain through the accumulator: 8 orbitals
/// (4 up, 4 down) against the CPU reference.
#[test]
fn colour_volume_chains_more_than_six_orbitals() {
    let Some(ctx) = ctx() else { return };
    let grid = Grid::cube(23, 0.35);
    let s = GpuSolver::new(ctx, &grid);
    let (a, b) = ([0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]);
    let nuclei = [nucleus(a, 4), nucleus(b, 4)];
    let set = |c: [f64; 3]| vec![hydrogenic(&grid, c, 2.0, "1s"), hydrogenic(&grid, c, 1.5, "2s"), hydrogenic(&grid, c, 1.4, "2px"), hydrogenic(&grid, c, 1.3, "2pz")];
    let e = [-2.0, -0.5, -0.3, -0.25];
    let (state, _) = synthetic_state(&s, &set(a), &set(b), &e, &e);
    let mut view = OrbitalView::new(&s);
    let vol = ColourVolume3D::new(&s.ctx, [23; 3]);
    let probe = view.update(&s, &state, &nuclei, 1e-6, Product::None);
    let include: Vec<u32> = probe.out.rows.iter().map(|r| r.track).collect();
    assert_eq!(include.len(), 8);
    let upd = view.update(&s, &state, &nuclei, 1e-6, Product::Colour { include: &include, physical: false, out: &vol });
    assert!(view.colour_acc.is_some(), "8 orbitals need the accumulator");
    let shown: [Vec<Vec<f64>>; 2] = [0, 1].map(|ch| view.prev[ch].iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect());
    let terms: Vec<(&[f64], f64, [f64; 3], f64)> = upd.out.rows.iter().map(|r| {
        let ch = r.spin.index();
        (shown[ch][r.index].as_slice(), r.occupation, ORBITAL_PALETTE[r.slot as usize].map(|v| v as f64), upd.out.scale[ch][r.index].powi(2))
    }).collect();
    let want = cpu::colour_volume(&grid, &terms);
    let got = vol.read_back(&s.ctx).expect("readback");
    let mut worst = 0.0f64;
    for (g4, w4) in got.iter().zip(&want) {
        if w4[3] <= 1e-3 { continue; }
        for c in 0..4 { worst = worst.max((g4[c] as f64 - w4[c]).abs() / (w4[c].abs() + 5e-5)); }
    }
    println!("colour 23³, 8 orbitals in 2 chained dispatches: max rel err {worst:.2e}; slots {:?}", upd.out.rows.iter().map(|r| r.slot).collect::<Vec<_>>());
    assert!(worst <= 2e-3, "{worst:.3e}");
    // a subset that fits one dispatch afterwards still matches (the accumulator is not read)
    let upd = view.update(&s, &state, &nuclei, 1e-6, Product::Colour { include: &include[..2], physical: true, out: &vol });
    let got = vol.read_back(&s.ctx).expect("readback");
    let first: Vec<(&[f64], f64, [f64; 3], f64)> = upd.out.rows.iter().filter(|r| include[..2].contains(&r.track)).map(|r| {
        (shown[r.spin.index()][r.index].as_slice(), r.occupation, ORBITAL_PALETTE[r.slot as usize].map(|v| v as f64), upd.spin.rho_max)
    }).collect();
    let want = cpu::colour_volume(&grid, &first);
    let worst = got.iter().zip(&want).filter(|(_, w)| w[3] > 1e-3)
        .map(|(g4, w4)| (0..4).map(|c| (g4[c] as f64 - w4[c]).abs() / (w4[c].abs() + 5e-5)).fold(0.0, f64::max)).fold(0.0, f64::max);
    assert!(worst <= 2e-3, "single chunk after chained: {worst:.3e}");
}

/// `reset` drops the display fields and restarts the tracks (fresh ids); a solver on another grid
/// rebuilds the view's grid-sized resources and resets it.
#[test]
fn reset_and_grid_change_rebuild_the_display() {
    let Some(ctx) = ctx() else { return };
    let (g1, g2) = (Grid::cube(17, 0.4), Grid::cube(21, 0.35));
    let (s1, s2) = (GpuSolver::new(ctx.clone(), &g1), GpuSolver::new(ctx, &g2));
    let nuclei = [nucleus([0.1, 0.0, 0.0], 2)];
    let st = |s: &GpuSolver, g: &Grid| synthetic_state(s, &[hydrogenic(g, nuclei[0].pos, 2.0, "1s")], &[hydrogenic(g, nuclei[0].pos, 2.0, "1s")], &[-1.0], &[-1.0]).0;
    let (a, b) = (st(&s1, &g1), st(&s2, &g2));
    let mut view = OrbitalView::new(&s1);
    let u1 = view.update(&s1, &a, &nuclei, 1e-6, Product::None);
    let u1b = view.update(&s1, &a, &nuclei, 1e-6, Product::None);
    assert_eq!(u1.out.rows[0].track, u1b.out.rows[0].track, "continuity keeps the track");
    assert_eq!(view.prev[0].len() + view.prev[1].len(), 2);
    view.reset();
    assert!(view.prev[0].is_empty() && view.prev[1].is_empty() && view.pool.is_empty() && view.scratch.is_none());
    let u2 = view.update(&s1, &a, &nuclei, 1e-6, Product::None);
    assert!(u2.out.rows[0].track > u1b.out.rows[1].track, "fresh tracks after reset");
    let vol = Volume3D::new(&s2.ctx, [21; 3]);
    let u3 = view.update(&s2, &b, &nuclei, 1e-6, Product::Orbital { spin: Spin::Up, track: u2.out.rows[0].track, squared: false, out: &vol });
    assert_eq!(view.grid, g2);
    assert_eq!(u3.scale, 0.0, "the grid change reset the tracks, so the old track is gone");
    let u4 = view.update(&s2, &b, &nuclei, 1e-6, Product::Orbital { spin: Spin::Up, track: u3.out.rows[0].track, squared: false, out: &vol });
    assert!(u4.scale > 0.0);
    let t = vol.read_back(&s2.ctx).expect("readback");
    let n2 = t.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() * g2.dv();
    assert!((n2 - 1.0).abs() < 1e-5, "{n2}");
}

/// Phase breakdown of one update at 49³, 57³ and 97³ (contended GPU; reported, never asserted):
/// `cargo test -p qm-gpu --release -- --ignored --nocapture diag_orbital_update_phases`. The last
/// column is the whole update with the signed-orbital product written to its texture (the paused
/// path of the viewer: the readback, the lincombs and the copy).
#[test]
#[ignore]
fn diag_orbital_update_phases() {
    let Some(ctx) = ctx() else { return };
    for n in [49usize, 57, 97] {
        diag_orbital_update_phases_at(&ctx, n);
    }
}

fn diag_orbital_update_phases_at(ctx: &GpuContext, n: usize) {
    let grid = Grid::cube(n, 0.25);
    let s = GpuSolver::new(ctx.clone(), &grid);
    let c = [0.37 * grid.h, 0.0, 0.0];
    let shapes = [("1s", 8.0), ("2s", 8.0), ("2px", 8.0), ("2py", 8.0), ("2pz", 8.0)];
    let set: Vec<Vec<f64>> = shapes.iter().map(|(w, z)| hydrogenic(&grid, c, *z, w)).collect();
    let e = [-20.0, -1.0, -0.8, -0.8, -0.8];
    let (state, _) = synthetic_state(&s, &set, &set, &e, &e);
    let sync = || { gpu::wait_idle(&s.ctx).expect("wait idle"); };
    for (what, nuclei) in [("lone atom", vec![nucleus(c, 10)]), ("molecule", vec![nucleus(c, 5), nucleus([-1.0, 0.0, 0.0], 5)])] {
        let mut view = OrbitalView::new(&s);
        for _ in 0..3 { view.update(&s, &state, &nuclei, 1e-5, Product::None); sync(); }
        let reps = 20;
        let (mut t_sc, mut t_tr, mut t_all) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let sc = view.read_scalars(&s, &state, &nuclei);
            t_sc = t_sc.min(t.elapsed().as_secs_f64());
            let occ = [vec![1.0; 5], vec![1.0; 5]];
            let lone = nuclei.len() == 1;
            let input = cpu::UpdateInput {
                evals: [&state.evals_up, &state.evals_dn], occupations: [&occ[0], &occ[1]],
                overlaps_prev: [sc.overlaps_prev[0].as_ref(), sc.overlaps_prev[1].as_ref()], cross: sc.cross.as_ref(),
                moments: lone.then(|| [&sc.moments[0][..], &sc.moments[1][..]]), radial: sc.radial.as_ref().map(|r| [&r[0][..], &r[1][..]]),
                first_sign: [&sc.first_sign[0], &sc.first_sign[1]], max_abs: [&sc.max_abs[0], &sc.max_abs[1]], outer_weight: [&sc.outer_weight[0], &sc.outer_weight[1]],
                n_occ: [5, 5], n_computed: [5, 5], n_core: [0; 4], z_sigma: None, tol_scf: 1e-5, unoccupied: None, face_limit: f64::INFINITY,
            };
            let mut tr = cpu::Tracker::new();
            let t = std::time::Instant::now();
            std::hint::black_box(tr.update(&input));
            t_tr = t_tr.min(t.elapsed().as_secs_f64());
            let t = std::time::Instant::now();
            view.update(&s, &state, &nuclei, 1e-5, Product::None);
            sync();
            t_all = t_all.min(t.elapsed().as_secs_f64());
        }
        // the whole update with the signed orbital product written to a texture
        let vol = Volume3D::new(&s.ctx, [n as u32; 3]);
        let track = view.update(&s, &state, &nuclei, 1e-5, Product::None).out.rows[0].track;
        let mut t_prod = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let u = view.update(&s, &state, &nuclei, 1e-5, Product::Orbital { spin: Spin::Up, track, squared: false, out: &vol });
            sync();
            t_prod = t_prod.min(t.elapsed().as_secs_f64());
            assert_eq!(u.readbacks, 1, "one readback per update");
        }
        let ms = |t: f64| t * 1e3;
        println!("{n}³ K = 10 {what} (min of {reps}): read_scalars {:.2} ms, Tracker::update {:.3} ms (CPU), whole update (product none) {:.2} ms, with the signed orbital texture {:.2} ms, 1 readback", ms(t_sc), ms(t_tr), ms(t_all), ms(t_prod));
    }
}

// ───────── unoccupied orbitals: the display eigensolve (`qm::orbitals` §8; plan Task 4) ─────────

/// `providers::surrogate::box_affected_face_density(0.25)` (2e-5 e/Bohr³ at h = 0.25): the box
/// rule's face limit the provider passes (`qm-gpu` cannot name `providers`).
const FACE_LIMIT_H025: f64 = 2e-5;

/// A converged GPU solve of `nuclei` on n³ at h = 0.25 (σ = h, tol 1e-6).
fn solved(ctx: &GpuContext, n: usize, nuclei: &[Nucleus]) -> (GpuSolver, ScfSettings) {
    let grid = Grid::cube(n, 0.25);
    let settings = ScfSettings::for_grid(&grid);
    let mut s = GpuSolver::new(ctx.clone(), &grid);
    s.solve(&System::new(nuclei.to_vec(), 0, None), &settings, false).expect("the SCF converges");
    (s, settings)
}

fn h2_nuclei() -> Vec<Nucleus> { vec![nucleus([0.7, 0.0, 0.0], 1), nucleus([-0.7, 0.0, 0.0], 1)] }
fn h2o_nuclei() -> Vec<Nucleus> { vec![nucleus([0.0; 3], 8), nucleus([1.43, 1.11, 0.0], 1), nucleus([-1.43, 1.11, 0.0], 1)] }

/// One line per channel: iterations, cap, |Δε_occ|, and (ε, residual, converged) per vector.
fn report(what: &str, u: &cpu::Unoccupied, wall: f64) {
    println!("{what}: tol {:.3e}, wall {wall:.2} s, rejected {:?}", u.tol, u.rejected);
    for ch in 0..2 {
        let rows: Vec<String> = (0..u.evals[ch].len()).map(|i| format!("{:+.5} ({:.1e}{})", u.evals[ch][i], u.residuals[ch][i], if u.converged[ch][i] { "" } else { " NOT" })).collect();
        println!("  {}: {} it{}, |Δε_occ| {:.1e}, hidden {}: {}", if ch == 0 { "↑" } else { "↓" }, u.iterations[ch], if u.at_cap[ch] { " (cap)" } else { "" },
                 u.occupied_error[ch], u.hidden[ch], rows.join(", "));
    }
}

/// The display eigensolve reads the solver's state and writes none of it (rule 6 at the solver
/// level; the 20-step MD bitwise test is `providers`' `unoccupied_solve_leaves_scf_state_untouched`):
/// orbitals, eigenvalues, densities and V_H read back bit-identical, and the next warm solve is
/// bit-identical to one on a twin solver that never ran it — even though the display solve grew the
/// shared `LobpcgWork` (and a later release drops it).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn unoccupied_solve_does_not_write_solver_fields() {
    let Some(ctx) = ctx() else { return };
    let nuclei = h2_nuclei();
    let (mut a, settings) = solved(&ctx, 33, &nuclei);
    let (mut b, _) = solved(&ctx, 33, &nuclei);
    let snap = |s: &GpuSolver| {
        let st = s.state().unwrap();
        let rd = |f: &Field| f.read_interior_f32(&s.ctx, &s.g).into_iter().map(f32::to_bits).collect::<Vec<_>>();
        (st.orbitals_up.iter().chain(&st.orbitals_dn).map(rd).collect::<Vec<_>>(), st.evals_up.clone(), st.evals_dn.clone(),
         rd(&st.rho), rd(&st.rho_up), rd(&st.rho_dn), rd(&st.v_h), rd(&st.v_nuc))
    };
    let before = snap(&a);
    let mut view = OrbitalView::new(&a);
    let u = view.unoccupied(&mut a, &nuclei, 4, &settings, &mut || false).expect("not aborted");
    assert!(u.rejected.is_none(), "{:?}", u.rejected);
    assert!(a.work_capacity() >= 1 + 4 + cpu::UNOCC_GUARD, "the shared work space grew to the display block");
    assert!(snap(&a) == before, "the display eigensolve wrote a solver field");
    let moved = System::new(vec![nucleus([0.72, 0.0, 0.0], 1), nucleus([-0.7, 0.0, 0.0], 1)], 0, None);
    let ea = a.solve(&moved, &settings, true).unwrap().1.energy.total;
    let eb = b.solve(&moved, &settings, true).unwrap().1.energy.total;
    assert_eq!(ea.to_bits(), eb.to_bits(), "warm solve after a display eigensolve: {ea} vs {eb}");
    view.release(&mut a);
    let ea = a.solve(&System::new(nuclei.clone(), 0, None), &settings, true).unwrap().1.energy.total;
    let eb = b.solve(&System::new(nuclei.clone(), 0, None), &settings, true).unwrap().1.energy.total;
    assert_eq!(ea.to_bits(), eb.to_bits(), "after the work space was released: {ea} vs {eb}");
}

/// Review MUST 4 / Task 0: the display solve runs at the SCF's **effective** tolerance
/// max(eig_tol, 3 × the f32 floor) — 5.42e-6 at h = 0.25, never the raw 1e-6 that sits below the
/// floor — with the SCF's 200-iteration cap. H₂, H₂O and Li at 49³, m = 4, cold.
///
/// The cap (controller ruling on Task 4 review I3, which replaces the brief's "none ends on the
/// cap"): it is accepted with honest tags (Task 0 option 1). The block's guard vectors, and at
/// times the top shown row, end at the 200 cap above the tol (H₂ and H₂O here: 200/200). Asserted
/// instead: **every SHOWN unoccupied row is at or below its tolerance or tagged "not converged"**
/// (a row's residual is its set's largest; the guards are never shown), the flag is the residual
/// against the tol, and every bound (ε < 0) row is converged. LSDA H₂ and H₂O bind no unoccupied
/// state in this 12 Bohr box (every row is the box's), so Li — whose 2p is bound, ↑ ≈ −0.040 Ha —
/// is what makes the bound-row check bite.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn display_solve_converges_at_the_effective_tolerance() {
    let Some(ctx) = ctx() else { return };
    let mut bound_total = 0;
    for (what, nuclei) in [("H₂ 49³", h2_nuclei()), ("H₂O 49³", h2o_nuclei()), ("Li 49³", vec![nucleus([0.0; 3], 3)])] {
        let (mut s, settings) = solved(&ctx, 49, &nuclei);
        let eff = s.effective_eig_tol(&settings);
        let floor = crate::eigen::residual_floor_estimate(&s.g);
        let mut view = OrbitalView::new(&s);
        let t = std::time::Instant::now();
        let u = view.unoccupied(&mut s, &nuclei, 4, &settings, &mut || false).expect("not aborted");
        report(what, &u, t.elapsed().as_secs_f64());
        println!("  effective tol {eff:.3e} = max(eig_tol {:.0e}, 3 × floor {floor:.3e})", settings.eig_tol);
        assert_eq!(u.tol, eff, "the display solve's tol is the SCF's effective one");
        assert!(eff > settings.eig_tol && (eff - 3.0 * floor).abs() <= 1e-15, "at h = 0.25 the floor decides: the raw 1e-6 is never the stop criterion");
        assert!(u.rejected.is_none(), "{what}: {:?}", u.rejected);
        let mut bound = 0;
        for ch in 0..2 {
            assert_eq!(u.evals[ch].len(), 4 + cpu::UNOCC_GUARD);
            for i in 0..u.evals[ch].len() {
                assert_eq!(u.converged[ch][i], u.residuals[ch][i] <= u.tol, "the flag is the residual against the tol");
                if u.evals[ch][i] < 0.0 {
                    bound += 1;
                    assert!(u.converged[ch][i], "{what}: bound unoccupied ε = {} ended at residual {:.2e} > {:.2e}", u.evals[ch][i], u.residuals[ch][i], u.tol);
                }
            }
        }
        println!("  bound unoccupied rows: {bound}");
        bound_total += bound;
        // the shown rows: at or below the tol, or tagged; never a guard
        let st = s.state().unwrap();
        let out = view.update(&s, st, &nuclei, settings.tol, Product::None).out;
        let (mut shown, mut tagged) = (0, 0);
        for r in out.rows.iter().filter(|r| r.occupation == 0.0) {
            let res = r.residual.expect("an unoccupied row's residual");
            assert!(res <= u.tol || r.not_converged, "{what}: shown row #{} at {res:.2e} > {:.2e} untagged", r.index + 1, u.tol);
            let n_occ = if r.spin == Spin::Up { st.orbitals_up.len() } else { st.orbitals_dn.len() };
            assert!(r.index < n_occ + 4 + cpu::UNOCC_GUARD - 1, "{what}: the last computed vector is never shown");
            shown += 1;
            tagged += usize::from(r.not_converged);
        }
        println!("  shown unoccupied rows {shown}, tagged not converged {tagged}");
    }
    assert!(bound_total >= 3, "Li's bound 2p rows were checked ({bound_total})");
}

/// §8's bound: the display solve's occupied eigenvalues (of V[ρ_out]) agree with the SCF's (of
/// V[ρ_in]) to τ(tol_scf) = 1e-4 Ha at tol 1e-6 (derived there: C·Δρ, C ≈ 1–5 Ha/e). H₂ and Li
/// (2↑ 1↓, an empty-shell ↓ channel above its occupied 1s) at 49³, m = 4; measured printed.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn occupied_eigenvalues_agree_with_the_scf() {
    let Some(ctx) = ctx() else { return };
    for (what, nuclei) in [("H₂", h2_nuclei()), ("Li", vec![nucleus([0.0; 3], 3)])] {
        let (mut s, settings) = solved(&ctx, 49, &nuclei);
        let mut view = OrbitalView::new(&s);
        let t = std::time::Instant::now();
        let u = view.unoccupied(&mut s, &nuclei, 4, &settings, &mut || false).expect("not aborted");
        report(what, &u, t.elapsed().as_secs_f64());
        let tau = cpu::tau(settings.tol);
        assert_eq!(tau, 1e-4);
        for ch in 0..2 {
            assert!(u.occupied_error[ch] <= tau, "{what} ch {ch}: |Δε_occ| = {:.3e} > τ", u.occupied_error[ch]);
        }
        assert!(u.rejected.is_none());
    }
}

/// The GPU display solve against the f64 CPU reference (`qm::orbitals::unoccupied_cpu`) on the
/// same discrete problem: H₂ 49³, σ = h, tol 1e-6, m = 2. Rows both sides converged agree to
/// 1e-4 Ha; every residual is ≤ its own tolerance (GPU effective, CPU eig_tol) or flagged.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn unoccupied_gpu_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let nuclei = h2_nuclei();
    let (mut s, settings) = solved(&ctx, 49, &nuclei);
    let mut view = OrbitalView::new(&s);
    let t = std::time::Instant::now();
    let g = view.unoccupied(&mut s, &nuclei, 2, &settings, &mut || false).expect("not aborted");
    report("GPU H₂ 49³ m = 2", &g, t.elapsed().as_secs_f64());
    let grid = s.grid().clone();
    let (st, _) = qm::scf(&grid, &System::new(nuclei.clone(), 0, None), &settings, None).expect("CPU SCF");
    let t = std::time::Instant::now();
    let (c, _) = cpu::unoccupied_cpu(&st, &nuclei, 2, &settings, None, &mut || false).expect("not aborted");
    report("CPU H₂ 49³ m = 2", &c, t.elapsed().as_secs_f64());
    assert!(g.rejected.is_none() && c.rejected.is_none());
    let mut compared = 0;
    for ch in 0..2 {
        assert_eq!(g.evals[ch].len(), c.evals[ch].len());
        for i in 0..g.evals[ch].len() {
            for u in [&g, &c] { assert!(u.residuals[ch][i] <= u.tol || !u.converged[ch][i], "a residual above tol must be flagged"); }
            if g.converged[ch][i] && c.converged[ch][i] {
                let d = (g.evals[ch][i] - c.evals[ch][i]).abs();
                println!("  ch {ch} #{i}: GPU {:+.7} CPU {:+.7} Δ {d:.1e}", g.evals[ch][i], c.evals[ch][i]);
                assert!(d <= 1e-4, "ch {ch} #{i}: GPU {} CPU {}", g.evals[ch][i], c.evals[ch][i]);
                compared += 1;
            }
        }
    }
    assert!(compared >= 4, "at least the first two per channel converged on both sides ({compared})");
}

/// The shown set is orthonormal (§8: the unoccupied vectors are projected onto the complement of
/// the SCF's occupied orbitals): |⟨ψ_occ|φ⟩| ≤ 1e-5 on the GPU's f32 vectors (read back, dotted in
/// f64). The CPU's 1e-10 is `qm`'s `unoccupied_cpu_is_a_display_solve_on_the_output_potential`.
/// H₂ and Li at 49³, m = 4.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn unoccupied_are_orthogonal_to_occupied() {
    let Some(ctx) = ctx() else { return };
    for (what, nuclei) in [("H₂", h2_nuclei()), ("Li", vec![nucleus([0.0; 3], 3)])] {
        let (mut s, settings) = solved(&ctx, 49, &nuclei);
        let mut view = OrbitalView::new(&s);
        let u = view.unoccupied(&mut s, &nuclei, 4, &settings, &mut || false).expect("not aborted");
        assert!(u.rejected.is_none());
        let grid = s.grid().clone();
        let st = s.state().unwrap();
        let (mut worst_occ, mut worst_self) = (0.0f64, 0.0f64);
        for ch in 0..2 {
            let occ: Vec<Vec<f64>> = if ch == 0 { &st.orbitals_up } else { &st.orbitals_dn }.iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect();
            let phis: Vec<Vec<f64>> = view.unocc[ch].iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect();
            for phi in &phis {
                for psi in &occ { worst_occ = worst_occ.max(grid.dot(psi, phi).abs()); }
                for other in &phis {
                    let d = grid.dot(phi, other);
                    worst_self = worst_self.max(if std::ptr::eq(phi, other) { (d - 1.0).abs() } else { d.abs() });
                }
            }
        }
        println!("{what}: max |⟨ψ_occ|φ⟩| {worst_occ:.2e}, max |⟨φ|φ′⟩ − δ| {worst_self:.2e}");
        assert!(worst_occ <= 1e-5, "{what}: {worst_occ:.2e}");
    }
}

/// Review MUST 3 on the real spectrum: He at a grid point of 49³ (an O_h site: its box states come
/// in 1-, 3- and 6-fold sets), m = 1 … 8, warm from the previous m's vectors. No shown cluster ever
/// contains the last computed vector (index n_occ + m + UNOCC_GUARD − 1), every shown set is whole,
/// and the "k shown" per m is printed.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn block_edge_hides_a_cut_set() {
    let Some(ctx) = ctx() else { return };
    let nuclei = vec![nucleus([0.0; 3], 2)];
    let (mut s, settings) = solved(&ctx, 49, &nuclei);
    let tau = cpu::tau(settings.tol);
    let mut view = OrbitalView::new(&s);
    view.set_face_limit(FACE_LIMIT_H025);
    for m in 1..=8usize {
        let t = std::time::Instant::now();
        let u = view.unoccupied(&mut s, &nuclei, m, &settings, &mut || false).expect("not aborted");
        let wall = t.elapsed().as_secs_f64();
        assert!(u.rejected.is_none(), "m = {m}: {:?}", u.rejected);
        let st = s.state().unwrap();
        let out = view.update(&s, st, &nuclei, settings.tol, Product::None).out;
        let mut line = Vec::new();
        for (ch, spin) in [(0usize, Spin::Up), (1, Spin::Down)] {
            let n_occ = 1;
            let last = n_occ + m + cpu::UNOCC_GUARD - 1;
            let unocc: Vec<&cpu::Row> = out.rows.iter().filter(|r| r.spin == spin && r.index >= n_occ).collect();
            assert!(unocc.iter().all(|r| r.index < last), "m = {m}: the last computed vector is shown");
            let evals = &u.evals[ch];
            let clusters = cpu::clusters(evals, tau);
            for r in &unocc {
                let c = clusters.iter().find(|c| c.contains(&(r.index - n_occ))).unwrap();
                assert!(c.end - 1 + n_occ < last, "m = {m}: a shown row's cluster {c:?} reaches the last vector");
                assert!(c.clone().all(|j| unocc.iter().any(|q| q.index == j + n_occ)), "m = {m}: the shown set {c:?} is whole");
            }
            assert_eq!(u.hidden[ch], out.hidden_unoccupied[ch]);
            let sizes: Vec<usize> = clusters.iter().map(|c| c.len()).collect();
            line.push(format!("{} {} shown, {} hidden, sets {sizes:?}", if ch == 0 { "↑" } else { "↓" }, unocc.len(), u.hidden[ch]));
        }
        println!("He 49³ m = {m}: {} ({wall:.1} s, it {:?})", line.join("; "), u.iterations);
    }
}

/// Spec §5.7 + §8 on the real spectrum, 49³, m = 4: every unoccupied row whose set has a member
/// with ε ≥ 0, an outer-L/6 weight > 0.05 or a face density above the limit is tagged a box state,
/// and no other row is; (ε, outer weight, face density, residual) per vector printed for
/// `references.md`. He: LSDA binds no unoccupied state, so every row is the box's. Li: its 2p is
/// bound (ε < 0), so the outer-weight and face criteria decide.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn box_states_are_flagged() {
    let Some(ctx) = ctx() else { return };
    for (what, z) in [("He", 2u8), ("Li", 3)] {
        box_states_of(&ctx, what, z);
    }
}

fn box_states_of(ctx: &GpuContext, what: &str, z: u8) {
    let nuclei = vec![nucleus([0.0; 3], z)];
    let (mut s, settings) = solved(ctx, 49, &nuclei);
    let tau = cpu::tau(settings.tol);
    let mut view = OrbitalView::new(&s);
    view.set_face_limit(FACE_LIMIT_H025);
    let u = view.unoccupied(&mut s, &nuclei, 4, &settings, &mut || false).expect("not aborted");
    let grid = s.grid().clone();
    let st = s.state().unwrap();
    let out = view.update(&s, st, &nuclei, settings.tol, Product::None).out;
    let n_occ = [st.orbitals_up.len(), st.orbitals_dn.len()];
    for (ch, spin) in [(0usize, Spin::Up), (1, Spin::Down)] {
        let phis: Vec<Vec<f64>> = view.unocc[ch].iter().map(|f| f.read_interior(&s.ctx, &s.g)).collect();
        let ow: Vec<f64> = phis.iter().map(|p| cpu::outer_weight(&grid, p)).collect();
        let boxy = |i: usize| cpu::box_state(u.evals[ch][i], ow[i], u.face_density[ch][i], FACE_LIMIT_H025);
        for i in 0..phis.len() {
            println!("{what} {} #{}: ε {:+.5} Ha, outer weight {:.3}, face density {:.2e}, residual {:.1e}{}", if ch == 0 { "↑" } else { "↓" }, i + n_occ[ch] + 1,
                     u.evals[ch][i], ow[i], u.face_density[ch][i], u.residuals[ch][i], if boxy(i) { " → box state" } else { "" });
            let set = cpu::clusters(&u.evals[ch], tau).into_iter().find(|c| c.contains(&i)).unwrap();
            let members: Vec<&[f64]> = phis[set].iter().map(|v| v.as_slice()).collect();
            let want = cpu::face_density_set(&grid, &members);
            assert!((u.face_density[ch][i] - want).abs() <= 1e-6 * want.max(1e-12), "set face density GPU {} vs CPU {want} on the same data", u.face_density[ch][i]);
            assert!(u.face_density[ch][i] >= cpu::face_density(&grid, &phis[i]) * (1.0 - 1e-6), "≥ the member's own");
        }
        let clusters = cpu::clusters(&u.evals[ch], tau);
        for r in out.rows.iter().filter(|r| r.spin == spin && r.index >= n_occ[ch]) {
            let c = clusters.iter().find(|c| c.contains(&(r.index - n_occ[ch]))).unwrap();
            assert_eq!(r.box_state, c.clone().any(boxy), "{what} row #{}: box_state", r.index + 1);
        }
        for r in out.rows.iter().filter(|r| r.spin == spin && r.index < n_occ[ch]) { assert!(!r.box_state, "occupied rows are never box states"); }
        if z == 2 {
            assert!(out.rows.iter().filter(|r| r.spin == spin && r.index >= 1).all(|r| r.box_state), "LSDA He: every unoccupied row is a box state");
        }
    }
}

/// Task 4 review I2: the display eigensolve yields. With an abort raised after a few LOBPCG
/// iterations it returns `None` promptly and changes nothing: the previous solve's result and its
/// vectors (read back bit for bit) stay; a following unaborted solve completes as before.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn the_display_eigensolve_aborts_and_keeps_the_previous_vectors() {
    let Some(ctx) = ctx() else { return };
    let nuclei = h2_nuclei();
    let (mut s, settings) = solved(&ctx, 33, &nuclei);
    let mut view = OrbitalView::new(&s);
    let first = view.unoccupied(&mut s, &nuclei, 2, &settings, &mut || false).expect("not aborted");
    let read = |v: &OrbitalView, s: &GpuSolver| -> Vec<Vec<u32>> {
        v.unocc.iter().flatten().map(|f| f.read_interior_f32(&s.ctx, &s.g).into_iter().map(f32::to_bits).collect()).collect()
    };
    let before = read(&view, &s);
    // a fresh geometry so the warm solve must iterate; abort on the 5th poll
    let moved = System::new(vec![nucleus([0.75, 0.0, 0.0], 1), nucleus([-0.7, 0.0, 0.0], 1)], 0, None);
    s.solve(&moved, &settings, true).unwrap();
    let mut polls = 0;
    let t = std::time::Instant::now();
    let r = view.unoccupied(&mut s, &moved.nuclei, 2, &settings, &mut || { polls += 1; polls >= 5 });
    let wall = t.elapsed().as_secs_f64();
    println!("aborted after {polls} polls in {:.0} ms (a full solve took {:?} iterations)", wall * 1e3, first.iterations);
    assert!(r.is_none() && polls == 5, "aborted on the 5th poll");
    assert_eq!(view.unoccupied_info(), Some(&first), "the previous result stays");
    assert!(read(&view, &s) == before, "the previous vectors stay, bit for bit");
    let again = view.unoccupied(&mut s, &moved.nuclei, 2, &settings, &mut || false).expect("not aborted");
    assert!(again.rejected.is_none() && again.evals[0].len() == 2 + cpu::UNOCC_GUARD);
}
