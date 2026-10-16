//! Orbital viewer, Task 0 spike (measure only; plan
//! `docs/superpowers/plans/2026-10-15-orbital-viewer.md` Task 0, numbers and conclusions in
//! `docs/superpowers/notes/2026-10-15-orbital-viewer-task0-spike.md`). Every test is an
//! `#[ignore]`d `diag_` measurement that prints a table and asserts only that it ran:
//!
//! `cargo test -p validate --release --test orbitals_spike -- --ignored --nocapture --test-threads=1`
//!
//! - `diag_2p_spread_vs_position`: lone Ne and lone O (singlet) on the GPU solver, σ = h, at
//!   h = 0.25 (57³) and h = 0.15 (95³) — the same ~14 Bohr box — at four nuclear positions (a
//!   grid point, a cube centre, the off-grid point (0.37h, −0.21h, 0.12h) and (h/2, 0, 0)),
//!   at the app's `INTERACTIVE_TOL` 1e-5 and the validate tol 1e-6: every occupied eigenvalue
//!   per channel, the spread of the occupied p-like set, the 1s/2s shift between sites, SCF
//!   steps / wall / convergence, and the closed-shell spin noise. Plus one O triplet attempt.
//! - `diag_closed_shell_spin_noise`: the H₂ singlet at 1.4 and 1.6 Bohr, 49³:
//!   max|ρ↑ − ρ↓| / max ρ, ∫|m| and ∫m, cold and warm.
//! - `diag_display_eigensolve_cost`: `qm_gpu::eigen::lobpcg_with_guess_tol` on the converged
//!   V_σ[ρ_out] = v_nuc + v_h + v_xc,σ[ρ_out↑, ρ_out↓] built from the `GpuState`, block
//!   n_σ + m + 2 (m = 4, 8), cold (unoccupied guess from `initial_orbitals`, seeds 3/4) and warm
//!   (the previous call's unoccupied vectors after a 0.01 Bohr move), at the SCF's effective tol
//!   max(eig_tol, 3 × `residual_floor_estimate`), for H₂ and H₂O at 49³, 81³ and 97³ (h = 0.25),
//!   against the warm SCF wall at the same size, plus one cold m = 4 solve with a 1000-iteration
//!   cap; every vector's own residual is recomputed afterwards (untimed). Cost numbers are wall times on whatever the
//!   GPU is doing at the time: the notes say whether they were measured on a quiet GPU.
//!
//! Nothing here touches library code: the display potential and eigensolve are assembled from
//! the public `qm_gpu` API with the test's own `Kernels`/`Reducer`/`LobpcgWork`.

use gpu::GpuContext;
use qm::{Grid, Nucleus, ScfSettings, System};
use qm_gpu::eigen::{lobpcg_with_guess_tol, residual_floor_estimate};
use qm_gpu::physics::xc_lsda;
use qm_gpu::pointwise::{lincomb, lincomb_in};
use qm_gpu::{Batch, EigenStats, Field, GpuGrid, GpuHamiltonian, GpuSolver, GpuState, Kernels, LobpcgWork, Reducer, Reduction};
use std::time::{Duration, Instant};

mod common;
use common::{ms, GPU};

fn gpu() -> Option<GpuContext> { common::gpu("orbitals_spike") }

/// The app's DFT display tolerance (`crates/app/src/sim.rs` `INTERACTIVE_TOL`) and the validate tol.
const INTERACTIVE_TOL: f64 = 1e-5;
const VALIDATE_TOL: f64 = 1e-6;
/// Copies of `qm_gpu::scf`'s private constants (scf.rs: `EIGEN_MAX_ITER`, `EIG_TOL_FLOOR_FACTOR`,
/// `EIG_GUESS_FACTOR`), so the display solve runs exactly as the SCF's eigensolves do.
const EIGEN_MAX_ITER: usize = 200;
const EIG_TOL_FLOOR_FACTOR: f64 = 3.0;
const EIG_GUESS_FACTOR: f64 = 1.5;
/// Guard vectors of the display eigensolve (spec §4.5 `UNOCC_GUARD`).
const UNOCC_GUARD: usize = 2;

fn positions(h: f64) -> [(&'static str, [f64; 3]); 4] {
    [("grid point", [0.0; 3]), ("cube centre", [h / 2.0; 3]), ("off-grid", [0.37 * h, -0.21 * h, 0.12 * h]), ("(h/2,0,0)", [h / 2.0, 0.0, 0.0])]
}

fn spread(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max) - v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Spin noise of a converged state, finished in f64 on the CPU from the f32 output densities:
/// (max|ρ↑ − ρ↓|, max ρ, ∫|m|, ∫m).
fn spin_stats(ctx: &GpuContext, g: &GpuGrid, s: &GpuState) -> (f64, f64, f64, f64) {
    let up = s.rho_up.read_interior_f32(ctx, g);
    let dn = s.rho_dn.read_interior_f32(ctx, g);
    let dv = g.h * g.h * g.h;
    let (mut max_m, mut max_rho) = (0.0f64, 0.0f64);
    let (mut abs_m, mut m) = (Vec::with_capacity(up.len()), Vec::with_capacity(up.len()));
    for (a, b) in up.iter().zip(&dn) {
        let (a, b) = (*a as f64, *b as f64);
        max_m = max_m.max((a - b).abs());
        max_rho = max_rho.max(a + b);
        abs_m.push((a - b).abs());
        m.push(a - b);
    }
    (max_m, max_rho, qm_gpu::reduce::neumaier(abs_m.into_iter()) * dv, qm_gpu::reduce::neumaier(m.into_iter()) * dv)
}

fn fmt_evals(v: &[f64]) -> String { v.iter().map(|e| format!("{e:.6}")).collect::<Vec<_>>().join(" ") }

/// Real unit spherical harmonics Y_lm(r̂) for l ≤ 3 (16 values; 0 at r = 0 for l > 0), the
/// spec §5.2 angular-character basis. Written out here (display-only measurement helper).
fn ylm(d: [f64; 3]) -> [f64; 16] {
    use std::f64::consts::PI;
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let mut y = [0.0; 16];
    y[0] = 0.5 / PI.sqrt();
    if r2 == 0.0 { return y; }
    let r = r2.sqrt();
    let (x, yy, z) = (d[0] / r, d[1] / r, d[2] / r);
    let c1 = (3.0 / (4.0 * PI)).sqrt();
    y[1] = c1 * yy; y[2] = c1 * z; y[3] = c1 * x;
    let c2 = (15.0 / (4.0 * PI)).sqrt();
    y[4] = c2 * x * yy; y[5] = c2 * yy * z; y[6] = (5.0 / (16.0 * PI)).sqrt() * (3.0 * z * z - 1.0);
    y[7] = c2 * x * z; y[8] = (15.0 / (16.0 * PI)).sqrt() * (x * x - yy * yy);
    y[9] = (35.0 / (32.0 * PI)).sqrt() * yy * (3.0 * x * x - yy * yy);
    y[10] = (105.0 / (4.0 * PI)).sqrt() * x * yy * z;
    y[11] = (21.0 / (32.0 * PI)).sqrt() * yy * (5.0 * z * z - 1.0);
    y[12] = (7.0 / (16.0 * PI)).sqrt() * z * (5.0 * z * z - 3.0);
    y[13] = (21.0 / (32.0 * PI)).sqrt() * x * (5.0 * z * z - 1.0);
    y[14] = (105.0 / (16.0 * PI)).sqrt() * z * (x * x - yy * yy);
    y[15] = (35.0 / (32.0 * PI)).sqrt() * x * (x * x - 3.0 * yy * yy);
    y
}

/// Angular character W_l (l = 0..3) of each orbital about `centre` from the linear moments
/// c_lm = ⟨ψ|Y_lm⟩ (spec §5.2), f64 on the CPU from the f32 readback.
fn characters(ctx: &GpuContext, g: &GpuGrid, grid: &Grid, orbs: &[Field], centre: [f64; 3]) -> Vec<[f64; 4]> {
    let ys: Vec<[f64; 16]> = (0..grid.len()).map(|i| { let p = grid.point(i); ylm([p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]]) }).collect();
    orbs.iter().map(|o| {
        let psi = o.read_interior_f32(ctx, g);
        let mut c = [0.0f64; 16];
        for (v, y) in psi.iter().zip(&ys) { for m in 0..16 { c[m] += *v as f64 * y[m]; } }
        let w = [c[0] * c[0], c[1..4].iter().map(|a| a * a).sum::<f64>(), c[4..9].iter().map(|a| a * a).sum::<f64>(), c[9..16].iter().map(|a| a * a).sum::<f64>()];
        let tot: f64 = w.iter().sum();
        w.map(|a| a / tot)
    }).collect()
}

/// Dominant l (W_l ≥ 0.9, spec `L_CHAR_MIN`) or None.
fn dominant(w: &[f64; 4]) -> Option<usize> { (0..4).find(|&l| w[l] >= 0.9) }

/// One channel's classified orbitals: (ε, W).
type Classified = Vec<(f64, [f64; 4])>;

fn of_l(c: &Classified, l: usize) -> Vec<f64> { c.iter().filter(|(_, w)| dominant(w) == Some(l)).map(|(e, _)| *e).collect() }

fn fmt_channel(c: &Classified) -> String {
    let name = ["s", "p", "d", "f"];
    c.iter().map(|(e, w)| match dominant(w) {
        Some(l) => format!("{e:.6}({} {:.4})", name[l], w[l]),
        None => format!("{e:.6}(? {:.2}/{:.2}/{:.2}/{:.2})", w[0], w[1], w[2], w[3]),
    }).collect::<Vec<_>>().join(" ")
}

/// One lone-atom solve's record.
struct AtomRow { pos: &'static str, ok: bool, steps: usize, wall: Duration, delta: f64, up: Classified, dn: Classified, noise: f64, abs_m: f64, int_m: f64, err: String }

impl AtomRow {
    /// The worst p-set spread over the two channels.
    fn p_spread(&self) -> f64 { spread(&of_l(&self.up, 1)).max(spread(&of_l(&self.dn, 1))) }
}

#[test]
#[ignore = "Task 0 spike: measurement only"]
fn diag_2p_spread_vs_position() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let grids = [(57usize, 0.25f64), (95, 0.15)];
    // (label, Z, multiplicity); the O singlet: the triplet does not converge (spec §11)
    let atoms: [(&str, u8, Option<u32>); 2] = [("Ne", 10, None), ("O singlet", 8, Some(1))];
    let mut summary = String::new();
    for &(label, z, mult) in &atoms {
        for &(n, h) in &grids {
            let grid = Grid::cube(n, h);
            let g = GpuGrid::new(&grid);
            for tol in [INTERACTIVE_TOL, VALIDATE_TOL] {
                println!("\n=== {label} (Z = {z}) {n}³ h = {h} (box {:.2} Bohr) σ = h, tol {tol:.0e} ===", (n - 1) as f64 * h);
                let mut rows = Vec::new();
                for (pos, r) in positions(h) {
                    let sys = System::new(vec![Nucleus { pos: r, z }], 0, mult);
                    let mut settings = ScfSettings::for_grid(&grid);
                    settings.tol = tol;
                    let mut solver = GpuSolver::new(ctx.clone(), &grid);
                    let t = Instant::now();
                    let res = solver.solve(&sys, &settings, false).map(|(_, st)| st);
                    let wall = t.elapsed();
                    let prof = solver.profile().clone();
                    let row = match res {
                        Ok(st) => {
                            let s = solver.state().expect("converged");
                            let (max_m, max_rho, abs_m, int_m) = spin_stats(&ctx, &g, s);
                            if !st.warnings.is_empty() { println!("  {pos}: warnings {:?}", st.warnings); }
                            let up = s.evals_up.iter().copied().zip(characters(&ctx, &g, &grid, &s.orbitals_up, r)).collect();
                            let dn = s.evals_dn.iter().copied().zip(characters(&ctx, &g, &grid, &s.orbitals_dn, r)).collect();
                            AtomRow { pos, ok: true, steps: st.iterations, wall, delta: st.delta_rho, up, dn, noise: max_m / max_rho, abs_m, int_m, err: String::new() }
                        }
                        Err(e) => AtomRow { pos, ok: false, steps: prof.steps, wall, delta: prof.delta_rho_steps.last().copied().unwrap_or(f64::NAN),
                                            up: Vec::new(), dn: Vec::new(), noise: f64::NAN, abs_m: f64::NAN, int_m: f64::NAN, err: e.to_string() },
                    };
                    if row.ok {
                        println!("  {:<12} converged {:>3} steps {:>7.0} ms Δρ {:.2e} | p spread ↑{:.2e} ↓{:.2e} | s ↑[{}] ↓[{}] | max|m|/maxρ {:.2e} ∫|m| {:.2e} ∫m {:+.1e}",
                                 row.pos, row.steps, ms(row.wall), row.delta, spread(&of_l(&row.up, 1)), spread(&of_l(&row.dn, 1)),
                                 fmt_evals(&of_l(&row.up, 0)), fmt_evals(&of_l(&row.dn, 0)), row.noise, row.abs_m, row.int_m);
                        println!("      ↑ {}", fmt_channel(&row.up));
                        println!("      ↓ {}", fmt_channel(&row.dn));
                    } else {
                        println!("  {:<12} NOT converged: {} ({} steps, {:.0} ms, last Δρ {:.2e})", row.pos, row.err, row.steps, ms(row.wall), row.delta);
                    }
                    rows.push(row);
                }
                let ok: Vec<&AtomRow> = rows.iter().filter(|r| r.ok).collect();
                if !ok.is_empty() {
                    // 1s/2s = the first/second s-dominant ↑ orbital (by energy)
                    let s_k = |k: usize| ok.iter().filter_map(|r| of_l(&r.up, 0).get(k).copied()).collect::<Vec<f64>>();
                    let (s1, s2) = (s_k(0), s_k(1));
                    let worst_p = ok.iter().map(|r| r.p_spread()).fold(0.0, f64::max);
                    let line = format!("{label} {n}³ h {h} tol {tol:.0e}: converged {}/4 (steps {}); 1s(↑) across sites [{}] shift {:.2e} Ha; 2s(↑) [{}] shift {:.2e} Ha; p spread per site (max ↑↓): {}; worst {:.2e} Ha; noise max|m|/maxρ ≤ {:.1e}, ∫|m| ≤ {:.1e}",
                                       ok.len(), ok.iter().map(|r| r.steps.to_string()).collect::<Vec<_>>().join("/"),
                                       fmt_evals(&s1), spread(&s1), fmt_evals(&s2), spread(&s2),
                                       ok.iter().map(|r| format!("{} {:.2e}", r.pos, r.p_spread())).collect::<Vec<_>>().join(", "), worst_p,
                                       ok.iter().map(|r| r.noise).fold(0.0, f64::max), ok.iter().map(|r| r.abs_m).fold(0.0, f64::max));
                    println!("  summary: {line}");
                    summary.push_str(&line);
                    summary.push('\n');
                } else {
                    let line = format!("{label} {n}³ h {h} tol {tol:.0e}: converged 0/4\n");
                    print!("  summary: {line}");
                    summary.push_str(&line);
                }
            }
        }
    }
    // the O triplet, once (it is known not to converge at h = 0.25; recorded, not assumed)
    let grid = Grid::cube(57, 0.25);
    let sys = System::new(vec![Nucleus { pos: [0.0; 3], z: 8 }], 0, Some(3));
    let mut settings = ScfSettings::for_grid(&grid);
    settings.tol = INTERACTIVE_TOL;
    let mut solver = GpuSolver::new(ctx.clone(), &grid);
    let t = Instant::now();
    let line = match solver.solve(&sys, &settings, false) {
        Ok((s, st)) => format!("O triplet 57³ grid point tol 1e-5: converged {} steps {:.0} ms, ↑[{}] ↓[{}]", st.iterations, ms(t.elapsed()), fmt_evals(&s.evals_up), fmt_evals(&s.evals_dn)),
        Err(e) => {
            let p = solver.profile();
            let tail: Vec<String> = p.delta_rho_steps.iter().rev().take(5).rev().map(|d| format!("{d:.1e}")).collect();
            format!("O triplet 57³ grid point tol 1e-5: NOT converged ({e}); {} steps {:.0} ms, last Δρ [{}], min Δρ {:.2e}", p.steps, ms(t.elapsed()), tail.join(" "),
                    p.delta_rho_steps.iter().copied().fold(f64::INFINITY, f64::min))
        }
    };
    println!("\n{line}");
    summary.push_str(&line);
    println!("\n=== SUMMARY ===\n{summary}");
}

fn h2_sys(r: f64, shift: f64) -> System {
    System::new(vec![Nucleus { pos: [r / 2.0 + shift, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0 + shift, 0.0, 0.0], z: 1 }], 0, None)
}

#[test]
#[ignore = "Task 0 spike: measurement only"]
fn diag_closed_shell_spin_noise() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::cube(49, 0.25);
    let g = GpuGrid::new(&grid);
    println!("\n=== H₂ singlet spin noise, 49³ h = 0.25 σ = h ===");
    for r in [1.4, 1.6] {
        for tol in [INTERACTIVE_TOL, VALIDATE_TOL] {
            let mut settings = ScfSettings::for_grid(&grid);
            settings.tol = tol;
            let mut solver = GpuSolver::new(ctx.clone(), &grid);
            for (what, shift, warm) in [("cold", 0.0, false), ("warm +0.01", 0.01, true), ("warm +0.02", 0.02, true)] {
                let st = solver.solve(&h2_sys(r, shift), &settings, warm).expect("H₂ converges").1;
                let (max_m, max_rho, abs_m, int_m) = spin_stats(&ctx, &g, solver.state().expect("converged"));
                let s = solver.state().unwrap();
                println!("  R {r} tol {tol:.0e} {what:<11}: {:>2} steps, ε ↑{:.7} ↓{:.7} (Δ {:.1e}) | max|m| {:.2e} max ρ {:.4} → max|m|/maxρ {:.2e} | ∫|m| {:.2e} ∫m {:+.1e}",
                         st.iterations, s.evals_up[0], s.evals_dn[0], (s.evals_up[0] - s.evals_dn[0]).abs(), max_m, max_rho, max_m / max_rho, abs_m, int_m);
            }
        }
    }
}

/// The display eigensolve's own fields: the test's kernels/reducer and the potential scratch.
struct Display { k: Kernels, r: Reducer, g: GpuGrid, exc: Field, xc_up: Field, xc_dn: Field, v: [Field; 2], work: LobpcgWork }

impl Display {
    fn new(ctx: &GpuContext, grid: &Grid, kmax: usize) -> Display {
        let g = GpuGrid::new(grid);
        let k = Kernels::new(ctx);
        let r = Reducer::new(ctx, &k, &g);
        let f = |l: &str| Field::new(ctx, &g, l);
        Display { exc: f("exc"), xc_up: f("xc up"), xc_dn: f("xc dn"), v: [f("v up"), f("v dn")], work: LobpcgWork::new(ctx, &g, kmax), k, r, g }
    }
}

/// One channel's display solve record: LOBPCG's stats, the eigenvalues, the largest deviation
/// of the returned occupied eigenvalues from the SCF's, and every vector's own residual
/// ‖Hx − εx‖/‖x‖ recomputed after the timed solve.
struct Solve { wall: Duration, stats: EigenStats, evals: Vec<f64>, occ_dev: f64, res: Vec<f64> }

/// ‖H xᵢ − εᵢ xᵢ‖/‖xᵢ‖ per vector (one readback each; outside the timing).
fn residuals(ctx: &GpuContext, d: &Display, h: &GpuHamiltonian, x: &[Field], evals: &[f64]) -> Vec<f64> {
    let (hx, r) = (Field::new(ctx, &d.g, "res hx"), Field::new(ctx, &d.g, "res r"));
    x.iter().zip(evals).map(|(xi, e)| {
        let mut b = Batch::new(ctx, "residual");
        h.apply_in(&mut b, &d.k, &d.g, xi, &hx);
        lincomb_in(&mut b, &d.k, &d.g, &r, &[(1.0, &hx), (-(*e as f32), xi)]);
        let v = d.r.many_in(&mut b, &d.k, &d.g, &[Reduction::Dot(&r, &r), Reduction::Dot(xi, xi)]);
        (v[0] / v[1]).sqrt()
    }).collect()
}

/// The display eigensolve on `s` for both channels: V_σ[ρ_out] built from the state, block =
/// the occupied copies + `unocc` (m + guard) vectors (taken as the guess and returned as the
/// next guess). Each channel is timed from the start of its potential build/guess copy to the
/// end of LOBPCG (which ends on a readback).
fn display_solve(ctx: &GpuContext, d: &mut Display, s: &GpuState, unocc: [Vec<Field>; 2], tol: f64, max_iter: usize) -> ([Solve; 2], [Vec<Field>; 2], Duration) {
    let t_xc = Instant::now();
    xc_lsda(ctx, &d.k, &d.g, &s.rho_up, &s.rho_dn, &d.exc, &d.xc_up, &d.xc_dn);
    let xc_wall = t_xc.elapsed();
    let mut out: Vec<Solve> = Vec::new();
    let mut back: Vec<Vec<Field>> = Vec::new();
    for (ch, unocc) in unocc.into_iter().enumerate() {
        let t = Instant::now();
        let (occ, evals_occ, xc) = if ch == 0 { (&s.orbitals_up, &s.evals_up, &d.xc_up) } else { (&s.orbitals_dn, &s.evals_dn, &d.xc_dn) };
        lincomb(ctx, &d.k, &d.g, &d.v[ch], &[(1.0, &s.v_nuc), (1.0, &s.v_h), (1.0, xc)]);
        let mut x: Vec<Field> = occ.iter().map(|o| { let f = Field::new(ctx, &d.g, "display occ"); lincomb(ctx, &d.k, &d.g, &f, &[(1.0, o)]); f }).collect();
        let n_occ = x.len();
        x.extend(unocc);
        let mut evals = vec![0.0; x.len()];
        let stats = {
            let h = GpuHamiltonian { v: &d.v[ch], nl: None };
            lobpcg_with_guess_tol(ctx, &d.k, &d.g, &d.r, &h, &mut x, &mut evals, &mut d.work, tol, EIG_GUESS_FACTOR * tol, max_iter, None).expect("display eigensolve")
        };
        let wall = t.elapsed();
        let res = residuals(ctx, d, &GpuHamiltonian { v: &d.v[ch], nl: None }, &x, &evals);
        let occ_dev = evals.iter().zip(evals_occ).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        back.push(x.split_off(n_occ));
        out.push(Solve { wall, stats, evals, occ_dev, res });
    }
    let mut o = out.into_iter();
    let mut b = back.into_iter();
    ([o.next().unwrap(), o.next().unwrap()], [b.next().unwrap(), b.next().unwrap()], xc_wall)
}

/// Cold unoccupied guess: `initial_orbitals` with seeds 3 (↑) and 4 (↓), uploaded as f32.
fn cold_guess(ctx: &GpuContext, d: &Display, grid: &Grid, nuclei: &[Nucleus], count: usize) -> ([Vec<Field>; 2], Duration) {
    let t = Instant::now();
    let up = |seed: u64| qm::scf::initial_orbitals(grid, nuclei, count, seed).iter().map(|v| { let f = Field::new(ctx, &d.g, "display unocc"); f.upload_interior(ctx, &d.g, v); f }).collect::<Vec<_>>();
    let r = [up(3), up(4)];
    (r, t.elapsed())
}

fn h2o_sys(shift: f64) -> System {
    System::new(vec![Nucleus { pos: [shift, 0.0, 0.0], z: 8 }, Nucleus { pos: [1.43, 1.11, 0.0], z: 1 }, Nucleus { pos: [-1.43, 1.11, 0.0], z: 1 }], 0, None)
}

/// Prints one display solve (both channels) and returns its summary line.
#[allow(clippy::too_many_arguments)]
fn report(label: &str, n: usize, what: &str, m: usize, n_occ: [usize; 2], res: &[Solve; 2], extra: Duration, scf: (Duration, usize, &str), tol: f64) -> String {
    for (ch, r) in res.iter().enumerate() {
        let shown = n_occ[ch] + m;
        let max_shown = r.res[..shown].iter().copied().fold(0.0, f64::max);
        let max_guard = r.res[shown..].iter().copied().fold(0.0, f64::max);
        println!("  display {what} m {m} {} block {:>2}: {:>6.0} ms, {:>4} it, {:>4} readbacks, {} MGS fallbacks, α worst {:.2}, stop {:?}, LOBPCG max res {:.2e}; own res: shown (n+m) ≤ {:.2e} {}, guards ≤ {:.2e}; |Δε_occ| {:.1e}{}",
                 ["↑", "↓"][ch], r.evals.len(), ms(r.wall), r.stats.iterations, r.stats.readbacks, r.stats.fallbacks, r.stats.alpha_worst, r.stats.stop, r.stats.max_residual,
                 max_shown, if max_shown <= tol { "(≤ tol)" } else { "(> tol)" }, max_guard, r.occ_dev,
                 if r.occ_dev > 1e-4 { " ← OCCUPIED EIGENPAIR LOST" } else { "" });
        println!("      all ε [{}]", fmt_evals(&r.evals));
        println!("      unocc ε [{}]  res [{}]", fmt_evals(&r.evals[n_occ[ch]..]), r.res[n_occ[ch]..].iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" "));
    }
    let total = res[0].wall + res[1].wall + extra;
    let shown_ok = (0..2).all(|ch| res[ch].res[..n_occ[ch] + m].iter().all(|&v| v <= tol));
    let occ_lost = res.iter().any(|r| r.occ_dev > 1e-4);
    let (w, steps, how) = scf;
    let line = format!("{label} {n}³ m {m} {what}: {:.0} ms = {:.2} × {how} SCF ({:.0} ms, {} steps) = {:.1} × one SCF step; it ↑{} ↓{}, stop ↑{:?} ↓{:?}; shown rows within tol: {}",
                       ms(total), total.as_secs_f64() / w.as_secs_f64(), ms(w), steps, total.as_secs_f64() / (w.as_secs_f64() / steps as f64),
                       res[0].stats.iterations, res[1].stats.iterations, res[0].stats.stop, res[1].stats.stop, if shown_ok { "yes" } else { "NO" }) + if occ_lost { "; OCCUPIED EIGENPAIR LOST (|Δε_occ| > 1e-4)" } else { "" };
    println!("  -> {line}");
    line
}

#[test]
#[ignore = "Task 0 spike: measurement only"]
fn diag_display_eigensolve_cost() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let ms_list = [4usize, 8];
    /// A longer cap for one extra cold m = 4 solve: does it converge at all, and when?
    const LONG_CAP: usize = 1000;
    let mut summary = String::new();
    // H₂ at the 1.4 Bohr bond (moved rigidly along x), H₂O at the h2o preset geometry (O moved along x)
    let systems: [(&str, fn(f64) -> System); 2] = [("H2", |s| h2_sys(1.4, s)), ("H2O", h2o_sys)];
    for &(label, make) in &systems {
        for n in [49usize, 81, 97] {
            let grid = Grid::cube(n, 0.25);
            let mut settings = ScfSettings::for_grid(&grid);
            settings.tol = INTERACTIVE_TOL;
            let mut solver = GpuSolver::new(ctx.clone(), &grid);
            // the display solve's effective tol: the SCF's own `eig_tol` of `run`
            let g = GpuGrid::new(&grid);
            let floor = residual_floor_estimate(&g);
            let tol = settings.eig_tol.max(EIG_TOL_FLOOR_FACTOR * floor);
            let sys0 = make(0.0);
            let n_occ = [sys0.n_up, sys0.n_dn];
            let kmax = n_occ[0].max(n_occ[1]) + ms_list.iter().max().unwrap() + UNOCC_GUARD;
            let mut d = Display::new(&ctx, &grid, kmax);
            println!("\n=== {label} {n}³ h = 0.25 (box {:.1} Bohr), SCF tol {:.0e}, n↑ {} n↓ {}; residual floor {floor:.3e}, effective eig tol max({:.0e}, 3·floor) = {tol:.3e} ===",
                     (n - 1) as f64 * 0.25, settings.tol, n_occ[0], n_occ[1], settings.eig_tol);
            let t = Instant::now();
            let st = solver.solve(&sys0, &settings, false).expect("cold SCF").1;
            println!("  SCF cold: {} steps, {:.0} ms", st.iterations, ms(t.elapsed()));
            // a warm SCF after the move; if it fails (recorded), a cold solve at the same geometry
            let scf = |solver: &mut GpuSolver, shift: f64, summary: &mut String| -> (Duration, usize, &'static str) {
                let t = Instant::now();
                match solver.solve(&make(shift), &settings, true) {
                    Ok((_, st)) => {
                        let w = t.elapsed();
                        let p = solver.profile().clone();
                        println!("  SCF warm (+{shift:.2} Bohr): {} steps, {:.0} ms ({:.0} ms/step; Poisson {:.0} ms, LOBPCG {:.0} ms, {} eigen it)",
                                 st.iterations, ms(w), ms(w) / st.iterations as f64, ms(p.poisson_wall), ms(p.eigen_wall), p.eigen_iterations);
                        (w, st.iterations, "warm")
                    }
                    Err(e) => {
                        let p = solver.profile().clone();
                        let tail: Vec<String> = p.delta_rho_steps.iter().rev().take(6).rev().map(|v| format!("{v:.1e}")).collect();
                        let line = format!("{label} {n}³: WARM SCF FAILED after +{shift:.2} Bohr: {e} ({:.0} ms; Δρ head [{}] tail [{}]; min Δρ {:.2e}; notes {:?})", ms(t.elapsed()),
                                           p.delta_rho_steps.iter().take(8).map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" "), tail.join(" "),
                                           p.delta_rho_steps.iter().copied().fold(f64::INFINITY, f64::min), p.notes);
                        println!("  {line}");
                        summary.push_str(&line);
                        summary.push('\n');
                        let t = Instant::now();
                        let st = solver.solve(&make(shift), &settings, false).expect("cold SCF after the failed warm one").1;
                        let w = t.elapsed();
                        println!("  SCF cold instead (+{shift:.2} Bohr): {} steps, {:.0} ms", st.iterations, ms(w));
                        (w, st.iterations, "cold (warm failed)")
                    }
                }
            };
            let scf1 = scf(&mut solver, 0.01, &mut summary);
            let mut saved: Vec<[Vec<Field>; 2]> = Vec::new();
            for &m in &ms_list {
                let (guess, gen) = cold_guess(&ctx, &d, &grid, &make(0.01).nuclei, m + UNOCC_GUARD);
                let (res, next, xc) = display_solve(&ctx, &mut d, solver.state().expect("converged"), guess, tol, EIGEN_MAX_ITER);
                println!("  (cold guess generation {:.0} ms, xc {:.1} ms; included)", ms(gen), ms(xc));
                summary.push_str(&report(label, n, "cold", m, n_occ, &res, gen + xc, scf1, tol));
                summary.push('\n');
                saved.push(next);
            }
            {
                let m = 4;
                let (guess, gen) = cold_guess(&ctx, &d, &grid, &make(0.01).nuclei, m + UNOCC_GUARD);
                let (res, _, xc) = display_solve(&ctx, &mut d, solver.state().expect("converged"), guess, tol, LONG_CAP);
                summary.push_str(&report(label, n, "cold cap 1000", m, n_occ, &res, gen + xc, scf1, tol));
                summary.push('\n');
            }
            let scf2 = scf(&mut solver, 0.02, &mut summary);
            for (&m, prev) in ms_list.iter().zip(saved) {
                let (res, _, xc) = display_solve(&ctx, &mut d, solver.state().expect("converged"), prev, tol, EIGEN_MAX_ITER);
                summary.push_str(&report(label, n, "warm", m, n_occ, &res, xc, scf2, tol));
                summary.push('\n');
            }
        }
    }
    println!("\n=== SUMMARY ===\n{summary}");
}
