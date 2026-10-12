//! Jacobi-CG Poisson solver on the GPU with the solid-harmonic multipole boundary — the port
//! of `qm::poisson` (read its module doc first: the equation ∇²V = −4πρ, the split
//! L V = L_int V + L_bc, A = −L_int SPD, the Racah-normalised real solid harmonics R_lm to
//! l = 6 and the ghost values V_bc(r) = Σ_lm Q_lm R_lm(r)/|r|^{2l+1}, Q_lm = ∫ρ R_lm). The
//! algorithm is unchanged; this doc covers only what the GPU changes.
//!
//! # Where f64 remains
//! Moments Q_lm, the CG scalars (α, β, ‖r‖, ‖b‖) and every reduction finish are f64 on the
//! CPU: a reduction is f32 workgroup partials → Neumaier f64 (`Reducer`). The 49 R_lm fields
//! and the ghost table are evaluated in f64 by `qm::poisson::solid_harmonics` and rounded to
//! f32 once per grid. Everything on the grid (ρ, V, r, p, A p, b) is f32.
//!
//! # Moments on the GPU
//! Q_lm = `Reducer::gram(&[ρ], &R_lm[..49])`: one submission, 49 dot products, each a sum of
//! f32 products ρ·R_lm accumulated per thread (≤ 4 points at 97³), tree-reduced per workgroup
//! in f32 and finished in f64. |R_lm| ≤ √2|r|^l grows to ~5e4 at l = 6, |r| = 6 Bohr, but f32
//! keeps *relative* precision there (6e-8·|R_lm|), so the error of Q_lm is ≲ 1e-7·S_l with
//! S_l = ∫ρ|r|^l the natural scale of an l-th moment: the f32 products and the 8-level f32
//! tree reduce add at most ~(1 + log₂ 256)·ε·Σ|ρ R_lm| on top of the input rounding (measured
//! 2e-8·S_l at 49³, see `moments_match_cpu_to_f32_precision`). Moments that nearly cancel
//! (|Q_lm| ≪ S_l) are therefore only absolutely accurate to that, as on the CPU with f32-rounded ρ.
//!
//! # Ghost fill
//! The stencil reads a ghost cell only along an axis line from an interior point, so only the
//! six face slabs of the 4-deep padding are ever read: 6·4·n² cells (edge and corner regions
//! stay 0 and are never read). Once per grid the CPU tabulates, for each such cell, its padded
//! index and the 49 values R_lm(r)/|r|^{2l+1} (f64 → f32; 45 MB at 97³). Per solve the
//! `ghost_fill` kernel writes V_bc = Σ_lm Q_lm·table[cell][lm] (one 49-term f32 dot, summed
//! from l = 6 down to l = 0 so the small terms go in first) into the ghost cells of a scratch
//! field whose interior is 0. `laplacian` of that field then gives exactly L_bc on the
//! interior — the GPU stencil has no bounds checks, it simply reads what the ghosts hold — and
//! b = 4πρ + L_bc by `lincomb`. The CG itself runs on fields whose ghosts are all 0, so
//! `stencil::laplacian` *is* L_int and A p = −L p.
//!
//! # CG loop
//! Per iteration: `laplacian` (A p), `dot` (p·Ap), two `axpy` (V += αp, r += α·Lp), `dot`
//! (r·r), `lincomb` (p = inv_diag·r + βp). With the constant Jacobi diagonal, z = inv_diag·r is
//! never stored: r·z = inv_diag·(r·r) uses the ‖r‖² that the stopping test needs anyway, so
//! the loop has two readbacks per iteration (α needs p·Ap before V and r can move; β needs the
//! new r·r). Each iteration is one `Batch` segment between the two sync points.
//!
//! # The f32 residual floor
//! CG updates the residual by the recurrence r ← r − α A p. In f32 the recurrence keeps
//! decreasing well below the *true* residual ‖b − A V‖ of the stored V (Greenbaum 1997): each
//! step's rounding of V += αp and of A p drops out of the recurrence but accumulates in the gap
//! r_rec − (b − A V), so what floors is the consistency of the unrestarted recurrence with the
//! stored V, not V's own representability — a restart from the floored f32 V (fresh r = b − A V)
//! reaches a true residual ~5× lower (`warm_start_reduces_iterations`: 5.1e-6 → 1.2e-6 in two
//! iterations). A crude scale for the gap: per-point rounding ~3e-8·|V| amplified by the stencil
//! (√(9c₀² + 6Σc_m²)/h² ≈ 150 at h = 0.25) gives ‖b − A V‖/‖b‖ ~ 5e-6·‖V‖/‖b‖, which is where the
//! unrestarted recurrence lands. Residual replacement at the check points (van der Vorst & Ye
//! 1999: r ← b − A V and continue) would push the floor down to the restart level; it is the
//! follow-up, not done here.
//!
//! Measured (RX 7900 XTX, off-centre Gaussian q = 1, σ = 0.5): the recurrence and the true
//! residual agree to four digits all the way down, then the true one floors while the
//! recurrence keeps falling — 33³ h = 0.25: floor 6.2e-6; 49³ h = 0.25: 9.8e-6 (recurrence
//! 1.6e-7 at iteration 200); 97³ h = 0.2: 1.9e-5 (recurrence 1.1e-6 at 350). The scale above
//! gives 7e-6 / 8e-6 / 1.5e-5 (‖V‖ grows with the box, the stencil factor with 1/h²).
//!
//! The solver uses the recurrence residual for the stopping test (as the CPU does, so iteration
//! counts are comparable) and every `CHECK_EVERY` iterations recomputes the true residual from
//! scratch (one `laplacian`, one `axpy`, one `dot`). Two signatures end a solve early, both
//! requiring the true residual to have *detached* from the recurrence (before the floor the
//! two agree to ~1e-3 relative, so a detached pair is unambiguous evidence of accumulated
//! rounding; see `classify`):
//! - `Stop::Floor`: true ≥ 4× recurrence. Once the gap F saturates, true ≈ √(rec² + F²), so 4×
//!   means the true residual is within 3% of F and the recurrence — which still tracks the
//!   low-frequency error the gap hides — is below F/4.
//! - `Stop::Stagnated`: true ≥ 2× recurrence *and* not decreased since the last check — the
//!   recurrence itself has stalled while detached. A non-decrease alone is deliberately not a
//!   trigger: CG residual norms plateau in healthy solves (97³: −7% between iterations 50 and
//!   75, true and recurrence still equal), and a plateau with no detachment is just CG.
//!
//! `PoissonStats::residual` is always the *true* residual ‖b − A V‖/‖b‖ of the returned V (one
//! final recomputation), never the recurrence value, so a caller asking for a tolerance below
//! the floor gets the floor reported, not a number the solution does not satisfy. Pinned in
//! `f32_residual_floor_is_documented`.
use crate::batch::{Batch, Recorder};
use crate::grid::{Field, GpuGrid, GHOST};
use crate::kernels::{uniform, workgroups_for, Kernels};
use crate::pointwise::{axpy_in, check, lincomb_in};
use crate::reduce::Reducer;
use crate::stencil::laplacian_in;
use gpu::GpuContext;
use qm::poisson::{lm_index, solid_harmonics, L_MAX, N_MOMENTS};
use qm::stencil::COEFF;
use std::f64::consts::PI;
use wgpu::util::DeviceExt;

/// Iterations between recomputations of the true residual ‖b − A V‖ (floor detection, see the
/// module doc). Each check costs one laplacian + one axpy + one dot ≈ one third of an iteration.
pub const CHECK_EVERY: usize = 25;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GhostParams { count: u32, pad: [u32; 3] }

/// Jacobi-preconditioned CG Poisson solver on the GPU with preallocated work fields, the 49
/// solid-harmonic fields and the ghost table for one grid (`new`; `solve` asserts the grid).
pub struct GpuPoisson {
    r: Field, p: Field, ap: Field, b: Field,
    /// Interior 0 (never written); ghosts receive V_bc each solve, so `laplacian(bc)` = L_bc.
    bc: Field,
    /// R_lm on the interior, index `lm_index(l, m)`.
    harm: Vec<Field>,
    /// Padded index of every face-slab ghost cell (6·4·n² of them) …
    ghost_cell: wgpu::Buffer,
    /// … and its 49 values R_lm(r)/|r|^{2l+1}, lm-major per cell.
    ghost_table: wgpu::Buffer,
    ghost_count: u32,
    grid: GpuGrid,
}

/// Why `solve` returned. `residual` is the true residual in every case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The recurrence residual reached `tol`. `residual` is ≤ tol unless the floor sits just
    /// above tol (then it is the floor; treat as converged).
    Converged,
    /// The f32 floor (module doc): the true residual detached ≥ 4× from the recurrence. The
    /// solution is as good as this solver makes it; treat as converged at `residual`.
    Floor,
    /// Detached ≥ 2× and the true residual did not decrease over `CHECK_EVERY` iterations: the
    /// recurrence stalled. Expected never on an SPD Poisson problem; treat as a warning.
    Stagnated,
    /// `max_iter` reached with the recurrence residual above `tol` and no floor signature.
    /// Treat as a warning (raise `max_iter`, or `tol` is unreachable for other reasons).
    MaxIter,
}

#[derive(Debug, Clone, Copy)]
pub struct PoissonStats {
    pub iterations: usize,
    /// True residual ‖b − A V‖/‖b‖ of the returned V, recomputed from scratch (not the CG recurrence).
    pub residual: f64,
    pub stop: Stop,
    /// GPU→CPU readbacks (sync points) in this call: moments, ‖b‖, ‖r‖, 2 per iteration, 1 per
    /// true-residual check, 1 for the final true residual.
    pub readbacks: usize,
}

/// The early-stop decision at a check point (module doc): `t` true residual, `rec` recurrence
/// residual, `last` true residual at the previous check (∞ at the first).
fn classify(t: f64, rec: f64, last: f64) -> Option<Stop> {
    if t >= 4.0 * rec { Some(Stop::Floor) } else if t >= 2.0 * rec && t >= last { Some(Stop::Stagnated) } else { None }
}

impl GpuPoisson {
    /// Samples R_lm on the grid and tabulates the ghost shell (both f64 → f32), uploads once.
    pub fn new(ctx: &GpuContext, _k: &Kernels, g: &GpuGrid) -> GpuPoisson {
        let (n, p) = (g.n, g.p);
        let point = |c: [i64; 3]| -> [f64; 3] { std::array::from_fn(|a| g.origin[a] + c[a] as f64 * g.h) };
        let mut h = [0.0; N_MOMENTS];

        // R_lm fields: one padded f32 buffer per moment, ghosts 0
        let mut padded: Vec<Vec<f32>> = (0..N_MOMENTS).map(|_| vec![0.0f32; g.padded_len()]).collect();
        for k in 0..n { for j in 0..n { for i in 0..n {
            solid_harmonics(point([i as i64, j as i64, k as i64]), &mut h);
            let q = g.pidx(i, j, k);
            for (lm, value) in h.iter().enumerate() { padded[lm][q] = *value as f32; }
        }}}
        let harm: Vec<Field> = padded.iter().map(|data| {
            let f = Field::new(ctx, g, "R_lm");
            ctx.queue.write_buffer(&f.buffer, 0, bytemuck::cast_slice(data));
            f
        }).collect();
        drop(padded);

        // ghost table: the 4 cells beyond each face along each axis line, for every (u, v) on the face
        let count = 6 * GHOST * n * n;
        let mut cells: Vec<u32> = Vec::with_capacity(count);
        let mut table: Vec<f32> = Vec::with_capacity(count * N_MOMENTS);
        for axis in 0..3 {
            for high in [false, true] {
                for m in 1..=GHOST as i64 {
                    for v in 0..n { for u in 0..n {
                        let mut c = [0i64; 3];
                        let (ua, va) = ((axis + 1) % 3, (axis + 2) % 3);
                        c[ua] = u as i64;
                        c[va] = v as i64;
                        c[axis] = if high { n as i64 - 1 + m } else { -m };
                        let pc: Vec<i64> = c.iter().map(|x| x + GHOST as i64).collect();
                        debug_assert!(pc.iter().all(|x| (0..p as i64).contains(x)));
                        cells.push((pc[0] + p as i64 * (pc[1] + p as i64 * pc[2])) as u32);
                        let r = point(c);
                        let d2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                        assert!(d2 > 0.0, "GpuPoisson::new: a ghost cell sits at the expansion origin");
                        solid_harmonics(r, &mut h);
                        let mut inv_pow = 1.0 / d2.sqrt(); // |r|^{-(2l+1)}
                        for l in 0..=L_MAX {
                            for mm in -(l as i32)..=(l as i32) { table.push((h[lm_index(l, mm)] * inv_pow) as f32); }
                            inv_pow /= d2;
                        }
                    }}
                }
            }
        }
        debug_assert_eq!(cells.len(), count);
        let ghost_cell = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poisson ghost cells"), contents: bytemuck::cast_slice(&cells), usage: wgpu::BufferUsages::STORAGE,
        });
        let ghost_table = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poisson ghost table"), contents: bytemuck::cast_slice(&table), usage: wgpu::BufferUsages::STORAGE,
        });
        let field = |l: &str| Field::new(ctx, g, l);
        GpuPoisson {
            r: field("poisson r"), p: field("poisson p"), ap: field("poisson Ap"), b: field("poisson b"), bc: field("poisson bc"),
            harm, ghost_cell, ghost_table, ghost_count: count as u32, grid: *g,
        }
    }

    fn check_grid(&self, g: &GpuGrid) {
        assert!(*g == self.grid, "GpuPoisson was built for a different grid: built for {:?}, called with {:?}", self.grid, g);
    }

    /// Q_lm = ∫ρ R_lm for all l ≤ 6 (index `lm_index`): 49 dots in one submission, f64 finish.
    pub fn moments(&self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, r: &Reducer, rho: &Field) -> Vec<f64> {
        self.check_grid(g);
        check(g, &[rho]);
        let harm: Vec<&Field> = self.harm.iter().collect();
        r.gram(ctx, k, g, &[rho], &harm)
    }

    /// Records: ghost cells of `bc` ← V_bc from the moments `q` (f32 storage buffer).
    fn ghost_fill_in(&self, batch: &mut Batch, k: &Kernels, q: &[f64]) {
        let ctx = batch.ctx();
        let q32: Vec<f32> = q.iter().map(|x| *x as f32).collect();
        let qbuf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("poisson moments"), contents: bytemuck::cast_slice(&q32), usage: wgpu::BufferUsages::STORAGE,
        });
        let u = uniform(ctx, "ghost_fill", &GhostParams { count: self.ghost_count, pad: [0; 3] });
        let bg = k.bind_group(ctx, "ghost_fill", &[
            u.as_entire_binding(), self.bc.buffer.as_entire_binding(), self.ghost_cell.as_entire_binding(),
            self.ghost_table.as_entire_binding(), qbuf.as_entire_binding(),
        ]);
        batch.dispatch(k, "ghost_fill", &bg, [workgroups_for(self.ghost_count as usize), 1, 1]);
    }

    /// ‖b − A V‖/‖b‖ from scratch (clobbers `ap`; a sync point).
    fn true_residual(&self, batch: &mut Batch, k: &Kernels, g: &GpuGrid, r: &Reducer, v: &Field, b_norm: f64) -> f64 {
        laplacian_in(batch, k, g, v, &self.ap);           // ap = L_int V = −A V
        axpy_in(batch, k, g, &self.ap, 1.0, &self.b);     // ap = b − A V
        r.dot_in(batch, k, g, &self.ap, &self.ap).sqrt() / b_norm
    }

    /// Solves ∇²V = −4πρ with V(ghost) = Σ_{l≤6,m} Q_lm R_lm(r)/|r|^{2l+1}, Q_lm = ∫ρ R_lm.
    /// `v` is the initial guess and the output; its ghosts must be 0 (true for any field
    /// produced by this crate). Stops when the CG recurrence residual ‖r‖/‖b‖ ≤ `tol`
    /// (`Stop::Converged`), after `max_iter` iterations (`Stop::MaxIter`), or at a check point
    /// every `CHECK_EVERY` iterations on a floor signature (`Stop::Floor` / `Stop::Stagnated`,
    /// module doc). The returned `residual` is the true ‖b − A V‖/‖b‖ of the output, which can
    /// exceed `tol` only at the floor (~6e-6 at 33³, 1e-5 at 49³, 2e-5 at 97³ for a unit charge).
    #[allow(clippy::too_many_arguments)]
    pub fn solve(&mut self, ctx: &GpuContext, k: &Kernels, g: &GpuGrid, r: &Reducer, rho: &Field, v: &Field, tol: f64, max_iter: usize) -> PoissonStats {
        self.check_grid(g);
        check(g, &[rho, v]);
        assert!(rho.buffer != v.buffer, "GpuPoisson::solve: rho and v alias");
        let q = self.moments(ctx, k, g, r, rho);
        let mut batch = Batch::new(ctx, "poisson");
        // b = 4πρ + L_bc
        self.ghost_fill_in(&mut batch, k, &q);
        laplacian_in(&mut batch, k, g, &self.bc, &self.ap);
        lincomb_in(&mut batch, k, g, &self.b, &[((4.0 * PI) as f32, rho), (1.0, &self.ap)]);
        // r = b − A v = b + L_int v
        laplacian_in(&mut batch, k, g, v, &self.ap);
        lincomb_in(&mut batch, k, g, &self.r, &[(1.0, &self.b), (1.0, &self.ap)]);
        let b_norm = r.dot_in(&mut batch, k, g, &self.b, &self.b).sqrt().max(1e-300);
        let mut rr = r.dot_in(&mut batch, k, g, &self.r, &self.r);
        let inv_diag = -(g.h * g.h) / (3.0 * COEFF[0]);
        let mut rz = inv_diag * rr;
        let mut residual = rr.sqrt() / b_norm;
        let mut beta = 0.0;
        let mut iterations = 0;
        let mut last_true = f64::INFINITY;
        let mut stop = None;
        let mut readbacks = 3usize; // moments, ‖b‖, ‖r‖
        while residual > tol && iterations < max_iter {
            // p = z + βp, z = inv_diag·r
            if iterations == 0 {
                lincomb_in(&mut batch, k, g, &self.p, &[(inv_diag as f32, &self.r)]);
            } else {
                lincomb_in(&mut batch, k, g, &self.p, &[(inv_diag as f32, &self.r), (beta as f32, &self.p)]);
            }
            laplacian_in(&mut batch, k, g, &self.p, &self.ap);           // ap = L_int p = −A p
            let pap = -r.dot_in(&mut batch, k, g, &self.p, &self.ap);
            assert!(pap > 0.0, "A lost positive-definiteness: p·Ap = {pap}");
            let alpha = rz / pap;
            axpy_in(&mut batch, k, g, v, alpha as f32, &self.p);         // v += αp
            axpy_in(&mut batch, k, g, &self.r, alpha as f32, &self.ap);  // r −= αAp
            rr = r.dot_in(&mut batch, k, g, &self.r, &self.r);
            let rz_new = inv_diag * rr;
            beta = rz_new / rz;
            rz = rz_new;
            residual = rr.sqrt() / b_norm;
            iterations += 1;
            readbacks += 2;
            if iterations % CHECK_EVERY == 0 && residual > tol {
                let t = self.true_residual(&mut batch, k, g, r, v, b_norm);
                readbacks += 1;
                stop = classify(t, residual, last_true);
                if stop.is_some() { break; }
                last_true = t;
            }
        }
        let stop = stop.unwrap_or(if residual <= tol { Stop::Converged } else { Stop::MaxIter });
        let residual = self.true_residual(&mut batch, k, g, r, v, b_norm);
        readbacks += 1;
        drop(batch); // empty after the last sync point
        PoissonStats { iterations, residual, stop, readbacks }
    }
}

/// Hartree energy ½∫ρV (Hartree); one `dot`.
pub fn hartree_energy(ctx: &GpuContext, k: &Kernels, g: &GpuGrid, r: &Reducer, rho: &Field, v: &Field) -> f64 {
    0.5 * r.dot(ctx, k, g, rho, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{assert_ghosts_zero, Field, GpuGrid};
    use crate::kernels::Kernels;
    use crate::reduce::Reducer;
    use gpu::GpuContext;
    use std::f64::consts::PI;

    /// Normalised Gaussian charge q, width σ, centre c: ρ = q (2πσ²)^{-3/2} exp(−|r−c|²/2σ²);
    /// exact V = q erf(|r−c|/(σ√2))/|r−c|. Same case as the `qm::poisson` tests.
    fn gaussian_case(n: usize, h: f64, q: f64, sigma: f64, centre: [f64; 3]) -> (qm::Grid, Vec<f64>, Vec<f64>) {
        let g = qm::Grid::cube(n, h);
        let norm = q / (2.0 * PI * sigma * sigma).powf(1.5);
        let rho = g.sample(|[x, y, z]| {
            let r2 = (x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2);
            norm * (-r2 / (2.0 * sigma * sigma)).exp()
        });
        let exact = g.sample(|[x, y, z]| {
            let r = ((x - centre[0]).powi(2) + (y - centre[1]).powi(2) + (z - centre[2]).powi(2)).sqrt();
            if r < 1e-12 { q * (2.0 / PI).sqrt() / sigma } else { q * libm::erf(r / (sigma * 2f64.sqrt())) / r }
        });
        (g, rho, exact)
    }

    struct Rig { ctx: GpuContext, k: Kernels, g: GpuGrid, r: Reducer, solver: GpuPoisson }

    fn rig(cg: &qm::Grid) -> Option<Rig> {
        let ctx = GpuContext::new_headless()?;
        let g = GpuGrid::new(cg);
        let k = Kernels::new(&ctx);
        let r = Reducer::new(&ctx, &k, &g);
        let solver = GpuPoisson::new(&ctx, &k, &g);
        Some(Rig { ctx, k, g, r, solver })
    }

    fn upload(rig: &Rig, data: &[f64], label: &str) -> Field {
        let f = Field::new(&rig.ctx, &rig.g, label);
        f.upload_interior(&rig.ctx, &rig.g, data);
        f
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) }

    /// True residual ‖b − A V‖/‖b‖ recomputed on the CPU in f64 from the GPU's f32 V and the
    /// CPU's f64 b: the oracle for the `residual` the solver reports.
    fn cpu_true_residual(cg: &qm::Grid, rho: &[f64], v: &[f64]) -> f64 {
        let mut cpu = qm::poisson::PoissonSolver::new(cg);
        let mut v_cpu = v.to_vec();
        // one CG step from V with tol = ∞ does nothing but we need b: re-derive it via a zero-iteration solve
        let stats = cpu.solve(cg, rho, &mut v_cpu, f64::INFINITY, 0);
        assert_eq!(stats.iterations, 0);
        stats.residual
    }

    #[test]
    fn gaussian_potential_matches_cpu_solver() {
        let c = [1.3, -0.7, 0.4];
        let (cg, rho, exact) = gaussian_case(33, 0.25, 1.0, 0.5, c);
        let Some(mut rig) = rig(&cg) else { return };
        let tol = 1e-6;
        let mut v_cpu = vec![0.0; cg.len()];
        let cpu = qm::poisson::PoissonSolver::new(&cg).solve(&cg, &rho, &mut v_cpu, tol, 2000);
        let (frho, fv) = (upload(&rig, &rho, "rho"), Field::new(&rig.ctx, &rig.g, "v"));
        let Rig { ctx, k, g, r, solver } = &mut rig;
        let gpu = solver.solve(ctx, k, g, r, &frho, &fv, tol, 2000);
        let v_gpu = fv.read_interior(ctx, g);
        assert_ghosts_zero(g, &fv.read_padded_raw(ctx));
        let diff = max_abs_diff(&v_gpu, &v_cpu);
        let err_gpu = max_abs_diff(&v_gpu, &exact);
        let err_cpu = max_abs_diff(&v_cpu, &exact);
        let true_res = cpu_true_residual(&cg, &rho, &v_gpu);
        println!("33³ off-centre Gaussian, tol {tol:.0e}: cpu {} it (res {:.2e}), gpu {} it ({:?}, res {:.2e}, f64-recomputed {true_res:.2e}); max|V_gpu − V_cpu| = {diff:.2e}; vs exact: gpu {err_gpu:.2e} cpu {err_cpu:.2e}",
                 cpu.iterations, cpu.residual, gpu.iterations, gpu.stop, gpu.residual);
        // plan bound 1e-5, tightened to 3e-6: achieved 1.0e-6 (RX 7900 XTX); iterations 125 vs 127
        // (the GPU stops on the floor signature at the check point 125; its recurrence is at 1.2e-6)
        assert!(diff < 3e-6, "max |V_gpu − V_cpu| = {diff:.3e}");
        assert!(matches!(gpu.stop, Stop::Converged | Stop::Floor), "stop {:?}", gpu.stop);
        assert!((gpu.iterations as i64 - cpu.iterations as i64).abs() <= 5, "iterations: gpu {} cpu {}", gpu.iterations, cpu.iterations);
        assert!(gpu.residual <= 1e-5, "reported residual {:.3e}", gpu.residual);
        assert!((gpu.residual - true_res).abs() <= 1e-6, "reported residual {:.3e} vs f64 recomputation {true_res:.3e}", gpu.residual);
    }

    /// The 49 GPU moments vs the CPU's compensated f64 ones, for an off-centre Gaussian (every
    /// moment non-zero: Q_lm = q·R_lm(d) by the mean-value property). Brief rule: 1e-6 relative
    /// for |Q| > 1e-6. Also printed: the error against the CPU moments of the f32-rounded ρ
    /// (kernel-only) and the per-l scale S_l = ∫ρ|r|^l the absolute error is judged against.
    #[test]
    fn moments_match_cpu_to_f32_precision() {
        let d = [1.3, -0.7, 0.4];
        let (cg, rho, _) = gaussian_case(49, 0.25, 1.0, 0.5, d);
        let Some(mut rig) = rig(&cg) else { return };
        let cpu_solver = qm::poisson::PoissonSolver::new(&cg);
        let q_cpu = cpu_solver.multipole_moments(&cg, &rho);
        let rho32: Vec<f64> = rho.iter().map(|v| *v as f32 as f64).collect();
        let q_cpu32 = cpu_solver.multipole_moments(&cg, &rho32);
        let frho = upload(&rig, &rho, "rho");
        let Rig { ctx, k, g, r, solver } = &mut rig;
        let q_gpu = solver.moments(ctx, k, g, r, &frho);
        assert_eq!(q_gpu.len(), N_MOMENTS);
        let mut h = [0.0; N_MOMENTS];
        solid_harmonics(d, &mut h);
        let (mut worst_rel, mut worst_abs_scaled) = (0.0f64, 0.0f64);
        for l in 0..=L_MAX {
            let s_l = cg.dot(&rho, &cg.sample(|[x, y, z]| (x * x + y * y + z * z).sqrt().powi(l as i32)));
            for m in -(l as i32)..=(l as i32) {
                let i = lm_index(l, m);
                let (gq, cq, cq32) = (q_gpu[i], q_cpu[i], q_cpu32[i]);
                let rel = ((gq - cq) / cq).abs();
                let scaled = (gq - cq).abs() / s_l;
                println!("Q[{l},{m:+}] gpu {gq:+.9e} cpu {cq:+.9e} rel {rel:.2e}; vs f32-ρ cpu rel {:.2e}; |Δ|/S_l {scaled:.2e}; exact q·R_lm(d) {:+.6e}",
                         ((gq - cq32) / cq32).abs(), h[i]);
                assert!(cq.abs() > 1e-6, "test assumption: every moment of the off-centre Gaussian is non-trivial");
                worst_rel = worst_rel.max(rel);
                worst_abs_scaled = worst_abs_scaled.max(scaled);
            }
        }
        println!("moments 49³: worst relative {worst_rel:.2e}, worst |Δ|/S_l {worst_abs_scaled:.2e}");
        for i in 0..N_MOMENTS {
            if q_cpu[i].abs() > 1e-6 {
                let rel = ((q_gpu[i] - q_cpu[i]) / q_cpu[i]).abs();
                assert!(rel <= 1e-6, "Q[{i}]: gpu {} cpu {} rel {rel:.3e}", q_gpu[i], q_cpu[i]);
            }
        }
    }

    #[test]
    fn hartree_energy_of_gaussian() {
        // E_H = q² / (2 σ √π), as in qm::poisson::tests::hartree_energy_of_gaussian
        let (cg, rho, _) = gaussian_case(49, 0.25, 1.0, 0.5, [0.0; 3]);
        let Some(mut rig) = rig(&cg) else { return };
        let (frho, fv) = (upload(&rig, &rho, "rho"), Field::new(&rig.ctx, &rig.g, "v"));
        let Rig { ctx, k, g, r, solver } = &mut rig;
        let stats = solver.solve(ctx, k, g, r, &frho, &fv, 1e-6, 2000);
        let e_h = hartree_energy(ctx, k, g, r, &frho, &fv);
        let exact = 1.0 / (2.0 * 0.5 * PI.sqrt());
        println!("E_H 49³: gpu {e_h:.8} exact {exact:.8} (Δ {:.2e}); {} it, residual {:.2e}", e_h - exact, stats.iterations, stats.residual);
        // qm bound 1e-4, tightened to 1e-5: achieved 9.5e-7 (the h = 0.25 discretisation error)
        assert!((e_h - exact).abs() < 1e-5, "E_H {e_h} vs {exact}");
    }

    /// Warm start: the same ρ again converges in far fewer iterations (as on the CPU), and for
    /// a perturbed ρ₂ (centre moved by 0.06 Bohr) starting from V(ρ₁) beats starting from 0 on
    /// the *same* ρ₂. (The off-centre ρ₂ is a harder problem than the centred ρ₁ — cold it takes
    /// 105 iterations on the CPU vs 85 — so the cold count of ρ₁ is not the right baseline.
    /// CPU counts: cold ρ₁ 85, same again 0, ρ₂ warm 92, ρ₂ cold 105; the GPU reproduces them.)
    #[test]
    fn warm_start_reduces_iterations() {
        let (cg, rho, _) = gaussian_case(33, 0.25, 1.0, 0.5, [0.0; 3]);
        let (_, rho2, _) = gaussian_case(33, 0.25, 1.0, 0.5, [0.05, -0.02, 0.03]);
        let Some(mut rig) = rig(&cg) else { return };
        let (frho, frho2) = (upload(&rig, &rho, "rho"), upload(&rig, &rho2, "rho2"));
        let (fv, fv2) = (Field::new(&rig.ctx, &rig.g, "v"), Field::new(&rig.ctx, &rig.g, "v2"));
        let Rig { ctx, k, g, r, solver } = &mut rig;
        let tol = 1e-6;
        let cold = solver.solve(ctx, k, g, r, &frho, &fv, tol, 2000);
        let again = solver.solve(ctx, k, g, r, &frho, &fv, tol, 2000);
        let warm2 = solver.solve(ctx, k, g, r, &frho2, &fv, tol, 2000);
        let cold2 = solver.solve(ctx, k, g, r, &frho2, &fv2, tol, 2000);
        println!("33³ warm start: ρ₁ cold {} it, same ρ₁ again {} it; ρ₂ warm from V(ρ₁) {} it vs cold {} it (true residuals {:.2e} / {:.2e} / {:.2e} / {:.2e}; stops {:?} / {:?} / {:?} / {:?})",
                 cold.iterations, again.iterations, warm2.iterations, cold2.iterations, cold.residual, again.residual, warm2.residual, cold2.residual,
                 cold.stop, again.stop, warm2.stop, cold2.stop);
        // the restart from the floored V: the fresh recurrence reaches tol in 2 iterations and the
        // true residual drops below the unrestarted floor (module doc: the floor is the gap, not V)
        assert_eq!(again.stop, Stop::Converged);
        assert!(again.residual < 0.5 * cold.residual, "restart {:.2e} vs floored {:.2e}", again.residual, cold.residual);
        assert!(again.iterations < cold.iterations / 4, "cold {} again {}", cold.iterations, again.iterations);
        assert!(warm2.iterations < cold2.iterations, "ρ₂: warm {} vs cold {}", warm2.iterations, cold2.iterations);
        let diff = max_abs_diff(&fv.read_interior(ctx, g), &fv2.read_interior(ctx, g));
        println!("ρ₂ warm vs cold solutions differ by {diff:.2e}");
        assert!(diff < 1e-5, "warm and cold solutions of ρ₂ differ by {diff:.3e}");
    }

    /// The early-stop gating on measured trajectories (module doc). A synthetic CG plateau solve
    /// is not cheaply constructible on the GPU, so the decision function is pinned directly:
    /// the 97³ plateau (iterations 50 → 75, true = recurrence) must not stop, nor any
    /// non-decrease without detachment; detachment ≥ 4× is `Floor`; 2–4× with no progress is
    /// `Stagnated`; 2–4× with progress continues.
    #[test]
    fn early_stop_requires_detachment() {
        // 97³ healthy plateau: last 6.300e-2 → now 5.886e-2, recurrence equal to the true residual
        assert_eq!(classify(5.886e-2, 5.886e-2, 6.300e-2), None);
        // a true non-decrease with no detachment is a CG plateau, not a floor
        assert_eq!(classify(6.4e-2, 6.4e-2, 6.3e-2), None);
        assert_eq!(classify(6.4e-2, 3.3e-2, 6.3e-2), None); // 1.94×: still below the stagnation gate
        // first check (last = ∞): detachment alone decides
        assert_eq!(classify(1.0e-5, 1.0e-5, f64::INFINITY), None);
        assert_eq!(classify(1.0e-5, 2.5e-6, f64::INFINITY), Some(Stop::Floor));
        assert_eq!(classify(1.0e-5, 4.0e-6, f64::INFINITY), None); // 2.5× detached, but progress vs ∞ → continue
        // 49³ iteration 175: true 9.84e-6, recurrence 2.97e-6 (3.3×), last 2.70e-5 → progress, continue
        assert_eq!(classify(9.84e-6, 2.97e-6, 2.70e-5), None);
        // 49³ iteration 200: true 9.98e-6, recurrence 1.61e-7 (62×) → Floor (regardless of progress)
        assert_eq!(classify(9.98e-6, 1.61e-7, 9.84e-6), Some(Stop::Floor));
        // detached 2–4× and no progress → Stagnated
        assert_eq!(classify(9.9e-6, 3.0e-6, 9.8e-6), Some(Stop::Stagnated));
        assert_eq!(classify(9.9e-6, 3.0e-6, 9.9e-6), Some(Stop::Stagnated));
    }

    /// Timing for the report: `cargo test -p qm-gpu --release -- --ignored --nocapture bench_poisson`.
    /// Setup (R_lm fields + ghost table), a cold solve to 1e-6, a floor solve (tol 1e-10) at 49³
    /// (h = 0.25) and 97³ (h = 0.2), each the off-centre Gaussian.
    #[test]
    #[ignore]
    fn bench_poisson_49_97() {
        use std::time::Instant;
        let Some(ctx) = GpuContext::new_headless() else { return };
        let k = Kernels::new(&ctx);
        for (n, h) in [(49, 0.25), (97, 0.2)] {
            let (cg, rho, exact) = gaussian_case(n, h, 1.0, 0.5, [1.3, -0.7, 0.4]);
            let g = GpuGrid::new(&cg);
            let r = Reducer::new(&ctx, &k, &g);
            let t = Instant::now();
            let mut solver = GpuPoisson::new(&ctx, &k, &g);
            ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            let setup_ms = t.elapsed().as_secs_f64() * 1e3;
            let frho = Field::new(&ctx, &g, "rho");
            frho.upload_interior(&ctx, &g, &rho);
            let fv = Field::new(&ctx, &g, "v");
            let t = Instant::now();
            let q = solver.moments(&ctx, &k, &g, &r, &frho);
            let moments_ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(q.len(), N_MOMENTS);
            for tol in [1e-6, 1e-10] {
                crate::pointwise::zero(&ctx, &k, &g, &fv);
                let t = Instant::now();
                let s = solver.solve(&ctx, &k, &g, &r, &frho, &fv, tol, 2000);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let err = max_abs_diff(&fv.read_interior(&ctx, &g), &exact);
                println!("{n}³ h={h}: tol {tol:.0e}: {} it ({:?}) in {ms:.1} ms ({:.3} ms/it), true residual {:.2e}, max|V − exact| {err:.2e} [setup {setup_ms:.0} ms, moments {moments_ms:.2} ms]",
                         s.iterations, s.stop, ms / s.iterations.max(1) as f64, s.residual);
            }
        }
    }

    /// Asking for tol = 1e-10 must stop at the f32 floor and report it, not spin to max_iter.
    /// The reported residual is the true ‖b − A V‖/‖b‖ (checked against an f64 recomputation).
    #[test]
    fn f32_residual_floor_is_documented() {
        let (cg, rho, exact) = gaussian_case(33, 0.25, 1.0, 0.5, [1.3, -0.7, 0.4]);
        let Some(mut rig) = rig(&cg) else { return };
        let (frho, fv) = (upload(&rig, &rho, "rho"), Field::new(&rig.ctx, &rig.g, "v"));
        let Rig { ctx, k, g, r, solver } = &mut rig;
        let max_iter = 2000;
        let stats = solver.solve(ctx, k, g, r, &frho, &fv, 1e-10, max_iter);
        let v_gpu = fv.read_interior(ctx, g);
        let true_res = cpu_true_residual(&cg, &rho, &v_gpu);
        let err = max_abs_diff(&v_gpu, &exact);
        println!("33³ tol 1e-10: stopped after {} it ({:?}): reported residual {:.2e}, f64 recomputation {true_res:.2e}; max |V − exact| = {err:.2e}",
                 stats.iterations, stats.stop, stats.residual);
        // achieved: Stop::Floor at the check point 125, floor 6.2e-6 (the recurrence alone would run on to 1e-10)
        assert_eq!(stats.stop, Stop::Floor, "expected the floor stop, got {:?}", stats.stop);
        assert!(stats.iterations < max_iter, "looped to max_iter ({})", stats.iterations);
        assert!(stats.residual < 1e-5, "floor {:.3e} is above 1e-5", stats.residual);
        assert!(stats.residual > 1e-8, "a 'floor' of {:.3e} means the reported residual is not the true one", stats.residual);
        assert!((stats.residual - true_res).abs() <= 0.1 * true_res, "reported {:.3e} vs f64 recomputation {true_res:.3e}", stats.residual);
    }
}
