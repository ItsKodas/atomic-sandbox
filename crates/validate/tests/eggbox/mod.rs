//! The spec §6.6 egg-box instrument for the M7 oracle suite (`pseudo_oracles.rs`; Task 10, Task 11 extends it).
//!
//! Duplicated from `pseudo.rs` ("The O₂ egg-box study": `Wall`, `occupied`, `ks_energy_f64`, `gate_keys`, `estimator`,
//! `transfer_check`, the verdict), without the study's remedy screens, and generic over the molecule (`at`), its setup
//! and multiplicity. `pseudo.rs` keeps its own copy (its gate runs carry the screens); the two are the same formulas.
//!
//! # What is measured (atomic units: Hartree, Bohr, Ha/Bohr)
//! A rigid shift s of every nucleus. On the fixed grid Σ_I F_I(s) = −∇_s E(s) exactly (HF is the derivative of the
//! discrete energy), and E is variational in the state, so the gate's number comes from **energies**, never from a GPU
//! frozen Hellmann–Feynman ΣF (first order in an SCF residual the f32 Δρ stop cannot see; Task 8a/8b reviews; the
//! GPU HF ΣF of O₂ is NOT QUALIFIED, u_HF 2.40e-5):
//! - E is the f64 re-evaluation of the cold GPU state (`ks_energy_f64`): occupied orbitals Löwdin-orthonormalised in
//!   f64, ρ_σ = Σ_i f_iσ|ψ_iσ|², V_H solved on the CPU to a residual ≤ 1e-11, and
//!   E = T + ∫ρV_loc + E_NL + E_H + E_xc[ρ_σ + ½ρ_pc] + E_ii with compensated sums. Error second order in the state.
//! - The Dirichlet box wall (Task 6 review, ruling C): the egg-box is periodic in s with period h, so the lattice
//!   translates ±h·ê_b differ from s = 0 only through the faces. Per axis E_w,b(s) = A₊(e^{2κs} − 1) + A₋(e^{−2κs} − 1),
//!   κ = √(−2ε_max) of the highest occupied level, A± solved from E(±h·ê_b) − E(0);
//!   F_w,b(s) = −2κ(A₊e^{2κs} − A₋e^{−2κs}). Ẽ = E − E_wall.
//! - The estimator at the eight probes p ∈ {±h/4}³ (units q = h/4): ΣF_a(p) = −(π/h)[Ẽ(p + qê_a) − Ẽ(p − qê_a)], 36 cold
//!   solves (coordinate a ∈ {0, ±h/2}, the others ±h/4) plus E(0) and the six translates. Writing Ẽ along a as
//!   Σ_n a_n cos(2πns_a/h) + b_n sin(2πns_a/h), it is exact for a₁, b₁, a₂, b₃; its systematic error is
//!   (4π/h)|b₂| + (8π/h)|a₃|.
//! - G = max_p |ΣF(p)| (wall-subtracted vector norm). Its uncertainty u_I = (π√2/h)σ_E + (4π/h)|b₂| + (8π/h)|a₃|
//!   (`transfer_check`: σ_E from a warm-from-0.02-Bohr solve at one gate state; b₂, a₃ from an eight-point scan
//!   s_a = kh/8 along the decisive probe's largest axis). The instrument qualifies when u_I ≤ 5e-6.
//!   **Assumption (physics review Minor-1):** the series is truncated at n ≤ 3. b₄ also enters the estimator, at
//!   (8π/h)|b₄|, and the eight-point scan cannot see it (sin(πk) = 0 at every sample); the harmonics decay fast, and
//!   (8π/h)|a₄| from the same scan (`coeffs[4]`) is printed as its order-of-magnitude proxy, not added to u_I.
//! - Verdict: PASS when G + 2u_I < 3e-5, MISS when G − 2u_I ≥ 3e-5, otherwise the CPU f64 polish decides (spec §6.6).
#![allow(dead_code)]
use crate::common::GPU;
use qm::ionic::IonicSetup;
use qm::nonlocal::NonlocalOp;
use qm::nuclear::{ionic_local_potential, ionic_repulsion, partial_core_field};
use qm::{Grid, KsState, Nucleus, Occupations, ScfSettings, System, DEGENERACY_TAU};
use qm_gpu::GpuSolver;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The cold GPU solves of the probe (`pseudo.rs` `GPU_REPORT_TOL`): tol 1e-6, eig 1e-7 (the GPU raises it to 3× its f32
/// floor), `DegenerateShells`.
pub const GPU_TOL: f64 = 1e-6;
/// The instrument qualifies when u_I ≤ this (spec §6.6, "Instrument").
pub const QUALIFY: f64 = 5e-6;
/// The §6.6 gate on max_s |Σ_I F_I(s) − F_wall(s)| (the brief's 10× below the all-electron figure of record).
pub const GATE: f64 = 3e-5;
/// The box must keep the wall's force ≤ 10× the gate (Task 6 review ruling C), else it is enlarged.
pub const WALL_MAX: f64 = 3e-4;
/// The all-electron figure of record: |Σ_I F_I,y| = 3.4e-4 Ha/Bohr at one H₂O geometry (raw, σ = h = 0.25, 12 Bohr;
/// references.md "GPU vs CPU at fixed σ") — the achieved factor is quoted against it, indicative, not like-for-like.
pub const AE_FIGURE: f64 = 3.4e-4;
/// The floor of u_I in the ratchet (spec §6.6 "Gate and fallback": each u_I ≥ 1.15e-7, the O₂ qualification at 0.20).
pub const U_FLOOR: f64 = 1.15e-7;

pub fn norm(v: [f64; 3]) -> f64 { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() }

/// Neumaier-compensated sum.
pub fn nsum(it: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in it {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    s + c
}

/// S^{−1/2} of a symmetric positive-definite k×k matrix near the identity: the coupled Newton–Schulz iteration (Higham,
/// *Functions of Matrices*, SIAM 2008, eq. 6.35).
fn inv_sqrt_spd(s: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = s.len();
    let mul = |a: &[Vec<f64>], b: &[Vec<f64>]| -> Vec<Vec<f64>> { (0..k).map(|i| (0..k).map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum()).collect()).collect() };
    let eye: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let (mut y, mut z) = (s.to_vec(), eye.clone());
    for _ in 0..20 {
        let zy = mul(&z, &y);
        let t: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| 1.5 * eye[i][j] - 0.5 * zy[i][j]).collect()).collect();
        y = mul(&y, &t);
        z = mul(&t, &z);
    }
    z
}

/// The occupied orbitals (f > 0) of a state, Löwdin-orthonormalised per spin in f64, their occupations and densities;
/// `dev` = the largest |S_ab − δ_ab| before (the f32 orthonormality error).
pub struct Occupied { pub orbs: [Vec<Vec<f64>>; 2], pub f: [Vec<f64>; 2], pub rho: [Vec<f64>; 2], pub dev: f64 }

pub fn occupied(g: &Grid, st: &KsState) -> Occupied {
    let n = g.len();
    let mut out = Occupied { orbs: [Vec::new(), Vec::new()], f: [Vec::new(), Vec::new()], rho: [vec![0.0; n], vec![0.0; n]], dev: 0.0 };
    for (sp, (orbs, f)) in [(&st.orbitals_up, &st.occ_up), (&st.orbitals_dn, &st.occ_dn)].into_iter().enumerate() {
        let idx: Vec<usize> = (0..orbs.len()).filter(|&i| f[i] > 0.0).collect();
        if idx.is_empty() { continue; }
        let s: Vec<Vec<f64>> = idx.iter().map(|&i| idx.iter().map(|&j| g.dot(&orbs[i], &orbs[j])).collect()).collect();
        for (a, row) in s.iter().enumerate() { for (b, &x) in row.iter().enumerate() { out.dev = out.dev.max((x - if a == b { 1.0 } else { 0.0 }).abs()); } }
        let z = inv_sqrt_spd(&s);
        for a in 0..idx.len() {
            let mut v = vec![0.0; n];
            for (b, &j) in idx.iter().enumerate() { let c = z[b][a]; for (vp, op) in v.iter_mut().zip(&orbs[j]) { *vp += c * op; } }
            let fa = f[idx[a]];
            for (r, x) in out.rho[sp].iter_mut().zip(&v) { *r += fa * x * x; }
            out.orbs[sp].push(v);
            out.f[sp].push(fa);
        }
    }
    out
}

/// The f64 Kohn–Sham energy of a state and its terms (Hartree).
#[derive(Clone, Copy, Debug)]
pub struct EnergyF64 { pub total: f64, pub kinetic: f64, pub local: f64, pub nl: f64, pub hartree: f64, pub xc: f64, pub e_ii: f64 }

fn xc_with_core(g: &Grid, rho: &[Vec<f64>; 2], pc: &[f64]) -> f64 {
    let u: Vec<f64> = rho[0].iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
    let d: Vec<f64> = rho[1].iter().zip(pc).map(|(a, c)| a + 0.5 * c).collect();
    g.integrate(&qm::xc::lsda_pw92(&u, &d).energy_density)
}

/// The instrument's energy (module doc): E = T + ∫ρV_loc + E_NL + E_H + E_xc[ρ_σ + ½ρ_pc] + E_ii in f64.
pub fn ks_energy_f64(g: &Grid, occ: &Occupied, v_h_guess: &[f64], s: &IonicSetup, nuc: &[Nucleus], poisson: &mut qm::poisson::PoissonSolver) -> (EnergyF64, qm::poisson::PoissonStats) {
    let n = g.len();
    let rho: Vec<f64> = (0..n).map(|i| occ.rho[0][i] + occ.rho[1][i]).collect();
    let mut t = vec![0.0; n];
    let mut kin = Vec::new();
    for sp in 0..2 { for (psi, &f) in occ.orbs[sp].iter().zip(&occ.f[sp]) { qm::stencil::kinetic(g, psi, &mut t); kin.push(f * g.dot(psi, &t)); } }
    let kinetic = nsum(kin.into_iter());
    let local = g.dot(&rho, &ionic_local_potential(g, nuc, s));
    let op = NonlocalOp::build(g, nuc, s);
    let nl = op.energy_weighted(&occ.orbs[0], &occ.f[0]) + op.energy_weighted(&occ.orbs[1], &occ.f[1]);
    let mut v = v_h_guess.to_vec();
    let ps = poisson.solve(g, &rho, &mut v, 1e-11, 20000);
    let hartree = qm::poisson::hartree_energy(g, &rho, &v);
    let xc = xc_with_core(g, &occ.rho, &partial_core_field(g, nuc, s));
    let e_ii = ionic_repulsion(nuc, s).expect("no coincident nuclei");
    (EnergyF64 { total: nsum([kinetic, local, nl, hartree, xc, e_ii].into_iter()), kinetic, local, nl, hartree, xc, e_ii }, ps)
}

/// The box-wall term of a rigid-shift probe (module doc; `pseudo.rs` `Wall`).
#[derive(Clone, Debug)]
pub struct Wall { pub kappa: f64, pub a: [[f64; 2]; 3] }

impl Wall {
    pub fn fit(h: f64, kappa: f64, e0: f64, e_plus: [f64; 3], e_minus: [f64; 3]) -> Wall {
        let (p, m) = ((2.0 * kappa * h).exp() - 1.0, (-2.0 * kappa * h).exp() - 1.0);
        let det = p * p - m * m;
        Wall { kappa, a: [0, 1, 2].map(|b| { let (dp, dm) = (e_plus[b] - e0, e_minus[b] - e0); [(p * dp - m * dm) / det, (p * dm - m * dp) / det] }) }
    }
    pub fn energy(&self, s: [f64; 3]) -> f64 {
        (0..3).map(|b| { let x = 2.0 * self.kappa * s[b]; self.a[b][0] * (x.exp() - 1.0) + self.a[b][1] * ((-x).exp() - 1.0) }).sum()
    }
    pub fn force(&self, s: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|b| { let x = 2.0 * self.kappa * s[b]; -2.0 * self.kappa * (self.a[b][0] * x.exp() - self.a[b][1] * (-x).exp()) })
    }
}

/// The highest occupied level of a state (either spin, f > 0), for the wall's decay κ = √(−2ε).
pub fn highest_occupied(st: &KsState) -> f64 {
    st.evals_up.iter().zip(&st.occ_up).chain(st.evals_dn.iter().zip(&st.occ_dn)).filter(|(_, &f)| f > 0.0).map(|(&e, _)| e).fold(f64::MIN, f64::max)
}

/// One GPU solve (`GPU_TOL`, eig 1e-7, `DegenerateShells`), cold or warm from the solver's last state, under the suite's
/// GPU lock; the state downloaded as a `KsState`. Panics on a non-converged solve (an unconverged state is never a probe).
pub fn gpu_solve(solver: &mut GpuSolver, setup: Arc<IonicSetup>, nuclei: Vec<Nucleus>, mult: Option<u32>, warm: bool) -> (f64, KsState, qm::ScfStats, Duration) {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let t0 = Instant::now();
    let g = solver.grid().clone();
    let sys = System::with_ionic(nuclei, 0, mult, setup);
    let mut s = ScfSettings::for_grid(&g);
    s.eig_tol = 1e-7;
    s.tol = GPU_TOL;
    s.occupations = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
    let (_, stats) = solver.solve(&sys, &s, warm).unwrap_or_else(|e| panic!("GPU SCF: {e}"));
    let st = solver.ks_state(solver.state().expect("a converged state"));
    (stats.energy.total, st, stats, t0.elapsed())
}

/// The 43 shifts of the estimator in units of q = h/4: E(0), the translates ±4ê_b, and the 36 points (coordinate
/// a ∈ {0, ±2}, the others ±1).
pub fn gate_keys() -> Vec<[i32; 3]> {
    let mut keys: Vec<[i32; 3]> = vec![[0, 0, 0]];
    for b in 0..3 { for sgn in [4, -4] { let mut k = [0; 3]; k[b] = sgn; keys.push(k); } }
    for a in 0..3 { for ca in [-2, 0, 2] { for s1 in [-1, 1] { for s2 in [-1, 1] {
        let mut k = [0; 3];
        k[a] = ca;
        let others: Vec<usize> = (0..3).filter(|&b| b != a).collect();
        (k[others[0]], k[others[1]]) = (s1, s2);
        if !keys.contains(&k) { keys.push(k); }
    }}}}
    keys
}

/// The estimator from energies at `gate_keys` (module doc): ΣF_a(p) = −(π/h)[Ẽ(p + ê_a) − Ẽ(p − ê_a)] at p ∈ {±1}³.
pub fn estimator(h: f64, kappa: f64, e: &dyn Fn([i32; 3]) -> f64) -> (Vec<([i32; 3], [f64; 3])>, Wall) {
    let q = h / 4.0;
    let e0 = e([0, 0, 0]);
    let wall = Wall::fit(h, kappa, e0, [e([4, 0, 0]), e([0, 4, 0]), e([0, 0, 4])], [e([-4, 0, 0]), e([0, -4, 0]), e([0, 0, -4])]);
    let et = |k: [i32; 3]| e(k) - wall.energy(k.map(|c| c as f64 * q));
    let mut out = Vec::new();
    for sx in [-1, 1] { for sy in [-1, 1] { for sz in [-1, 1] {
        let p = [sx, sy, sz];
        out.push((p, [0, 1, 2].map(|a| { let (mut up, mut dn) = (p, p); up[a] += 1; dn[a] -= 1; -(std::f64::consts::PI / h) * (et(up) - et(dn)) })));
    }}}
    (out, wall)
}

/// The wall's largest force over the eight probes.
pub fn wall_max(w: &Wall, h: f64) -> f64 {
    let q = h / 4.0;
    let mut m = 0.0f64;
    for x in [-q, q] { for y in [-q, q] { for z in [-q, q] { m = m.max(norm(w.force([x, y, z]))); } } }
    m
}

pub fn gmax(p: &[([i32; 3], [f64; 3])]) -> f64 { p.iter().map(|(_, f)| norm(*f)).fold(0.0, f64::max) }

/// One gate state: the GPU's E, the f64 energy, ε_max, SCF steps, Δρ, the f32 orthonormality error and the f64 Poisson
/// solve (iterations, residual).
pub struct GateState { pub e_gpu: f64, pub e: EnergyF64, pub eps_max: f64, pub steps: usize, pub delta_rho: f64, pub dev: f64, pub poisson: (usize, f64) }

/// The 43 cold states of one probe run.
pub struct GateRun { pub h: f64, pub n: usize, pub states: HashMap<[i32; 3], GateState>, pub secs: f64, pub gpu: Duration, pub solves: usize, pub scf_steps: usize, pub warnings: Vec<String> }

/// The molecule rigidly shifted by s (Bohr), the pseudised setup at h, the multiplicity.
pub type At<'a> = &'a dyn Fn([f64; 3]) -> Vec<Nucleus>;

/// The 43 cold GPU solves at h in a `box_bohr` cube (odd n, a grid point at the origin), each state downloaded and
/// re-evaluated in f64.
pub fn gate_run(ctx: &gpu::GpuContext, h: f64, box_bohr: f64, s: Arc<IonicSetup>, at: At, mult: Option<u32>) -> GateRun {
    let n = 2 * (box_bohr / (2.0 * h)).round() as usize + 1;
    let g = Grid::cube(n, h);
    let mut solver = { let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner()); GpuSolver::new(ctx.clone(), &g) };
    let mut poisson = qm::poisson::PoissonSolver::new(&g);
    let q = h / 4.0;
    let t0 = Instant::now();
    let mut run = GateRun { h, n, states: HashMap::new(), secs: 0.0, gpu: Duration::ZERO, solves: 0, scf_steps: 0, warnings: Vec::new() };
    for k in gate_keys() {
        let nuc = at(k.map(|c| c as f64 * q));
        let (e_gpu, st, stats, dt) = gpu_solve(&mut solver, s.clone(), nuc.clone(), mult, false);
        run.gpu += dt;
        run.solves += 1;
        run.scf_steps += stats.iterations;
        for w in &stats.warnings { if !run.warnings.contains(w) { run.warnings.push(w.clone()); } }
        let occ = occupied(&g, &st);
        let (e, ps) = ks_energy_f64(&g, &occ, &st.v_h, &s, &nuc, &mut poisson);
        run.states.insert(k, GateState { e_gpu, e, eps_max: highest_occupied(&st), steps: stats.iterations, delta_rho: stats.delta_rho, dev: occ.dev, poisson: (ps.iterations, ps.residual) });
    }
    run.secs = t0.elapsed().as_secs_f64();
    run
}

impl GateRun {
    pub fn kappa(&self) -> f64 { (-2.0 * self.states[&[0, 0, 0]].eps_max).sqrt() }
    /// The estimator on the f64 energies.
    pub fn probes(&self) -> (Vec<([i32; 3], [f64; 3])>, Wall) { estimator(self.h, self.kappa(), &|k| self.states[&k].e.total) }
    /// The estimator on the GPU's own energies (printed only: f32 evaluation noise, spec §6.6).
    pub fn probes_gpu(&self) -> (Vec<([i32; 3], [f64; 3])>, Wall) { estimator(self.h, self.kappa(), &|k| self.states[&k].e_gpu) }

    /// The run's instrument line and per-probe vectors; returns G.
    pub fn print_summary(&self, label: &str) -> f64 {
        let h = self.h;
        let d: Vec<f64> = self.states.values().map(|s| s.e_gpu - s.e.total).collect();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (d.len() - 1) as f64).sqrt();
        let steps: Vec<usize> = self.states.values().map(|s| s.steps).collect();
        let e0 = &self.states[&[0, 0, 0]].e;
        eprintln!("{label} h = {h}, {}³: E_GPU − E_f64 mean {mean:+.3e}, sd {sd:.2e} Ha (GPU-energy estimator noise (π√2/h)·sd = {:.1e}); f32 orthonormality ≤ {:.1e}; \
                   f64 Poisson ≤ {} iterations, residual ≤ {:.1e}; SCF steps {}–{}, Δρ ≤ {:.1e}; E(0) f64 {:.9} (T {:.6}, loc {:.6}, NL {:.6}, H {:.6}, xc {:.6}, ii {:.6}); κ {:.4}; {:.0} s",
                  self.n, std::f64::consts::PI * 2f64.sqrt() / h * sd, self.states.values().map(|s| s.dev).fold(0.0, f64::max),
                  self.states.values().map(|s| s.poisson.0).max().unwrap(), self.states.values().map(|s| s.poisson.1).fold(0.0, f64::max),
                  steps.iter().min().unwrap(), steps.iter().max().unwrap(), self.states.values().map(|s| s.delta_rho).fold(0.0, f64::max),
                  e0.total, e0.kinetic, e0.local, e0.nl, e0.hartree, e0.xc, e0.e_ii, self.kappa(), self.secs);
        let (base, wall) = self.probes();
        let q = h / 4.0;
        for (p, f) in &base {
            let fw = wall.force(p.map(|c| c as f64 * q));
            eprintln!("    {label} probe {p:?}: ΣF − F_wall ({:+.3e}, {:+.3e}, {:+.3e}) |{:.3e}|; raw |ΣF| {:.3e}; F_wall ({:+.2e}, {:+.2e}, {:+.2e})",
                      f[0], f[1], f[2], norm(*f), norm([f[0] + fw[0], f[1] + fw[1], f[2] + fw[2]]), fw[0], fw[1], fw[2]);
        }
        let (gp, _) = self.probes_gpu();
        eprintln!("  {label} h = {h}: G (f64 energies) {:.3e}; G on the GPU's own energies {:.3e} (printed only); wall max {:.2e}", gmax(&base), gmax(&gp), wall_max(&wall, h));
        gmax(&base)
    }
}

/// The transfer of the instrument to this h and molecule (module doc; `pseudo.rs` `transfer_check`).
pub struct Transfer { pub u: f64, pub sigma_e: f64, pub b2: f64, pub a3: f64, pub probe: [i32; 3], pub axis: usize, pub coeffs: Vec<(f64, f64)>, pub periodicity: f64 }

pub fn transfer_check(ctx: &gpu::GpuContext, run: &GateRun, box_bohr: f64, s: Arc<IonicSetup>, at: At, mult: Option<u32>) -> Transfer {
    let h = run.h;
    let q = h / 4.0;
    let (probes, wall) = run.probes();
    let (probe, f) = probes.iter().copied().max_by(|a, b| norm(a.1).total_cmp(&norm(b.1))).expect("eight probes");
    let axis = (0..3).max_by(|&a, &b| f[a].abs().total_cmp(&f[b].abs())).unwrap();
    let g = Grid::cube(run.n, h);
    debug_assert_eq!(run.n, 2 * (box_bohr / (2.0 * h)).round() as usize + 1);
    let mut solver = { let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner()); GpuSolver::new(ctx.clone(), &g) };
    let mut poisson = qm::poisson::PoissonSolver::new(&g);
    let mut e_f64 = |solver: &mut GpuSolver, shift: [f64; 3], warm: bool| -> f64 {
        let nuc = at(shift);
        let (_, st, _, _) = gpu_solve(solver, s.clone(), nuc.clone(), mult, warm);
        ks_energy_f64(&g, &occupied(&g, &st), &st.v_h, &s, &nuc, &mut poisson).0.total
    };
    let base = probe.map(|c| c as f64 * q);
    let along = |sa: f64| { let mut p = base; p[axis] = sa; p };
    let gate_e = |ka: i32| { let mut k = probe; k[axis] = ka; run.states[&k].e.total };
    let mut et = Vec::new();
    for k in -3i32..=4 {
        let p = along(k as f64 * h / 8.0);
        let e = match k { 0 => gate_e(0), 4 => gate_e(2), _ => e_f64(&mut solver, p, false) };
        et.push(e - wall.energy(p));
    }
    let periodicity = (gate_e(-2) - wall.energy(along(-h / 2.0))) - et[7];
    let coeffs: Vec<(f64, f64)> = (0..=4).map(|n| {
        let (c, sn) = (-3i32..=4).zip(&et).fold((0.0, 0.0), |(c, sn), (k, e)| {
            let x = 2.0 * std::f64::consts::PI * (n * k) as f64 / 8.0;
            (c + e * x.cos(), sn + e * x.sin())
        });
        let w = if n == 0 || n == 4 { 1.0 / 8.0 } else { 2.0 / 8.0 };
        (w * c, w * sn)
    }).collect();
    // σ_E at the gate state p* + (h/4)ê_a*: cold (the gate's) against warm from 0.02 Bohr away
    let mut ke = probe;
    ke[axis] += 1;
    let pe = ke.map(|c| c as f64 * q);
    let mut near = pe;
    near[axis] += 0.02;
    e_f64(&mut solver, near, false);
    let sigma_e = (e_f64(&mut solver, pe, true) - run.states[&ke].e.total).abs();
    let (b2, a3) = (coeffs[2].1, coeffs[3].0);
    let pi = std::f64::consts::PI;
    let u = pi * 2f64.sqrt() / h * sigma_e + 4.0 * pi / h * b2.abs() + 8.0 * pi / h * a3.abs();
    Transfer { u, sigma_e, b2, a3, probe, axis, coeffs, periodicity }
}

/// Spec §6.6's verdict from (G, u_I).
pub fn verdict(big_g: f64, u: f64) -> &'static str {
    if u > QUALIFY { "NOT QUALIFIED — every probe goes to the CPU f64 polish (spec §6.6)" }
    else if big_g + 2.0 * u < GATE { "PASS" }
    else if big_g - 2.0 * u >= GATE { "MISS" }
    else { "inside the ±2u_I band — the decisive probes go to the CPU f64 polish (spec §6.6)" }
}

/// What one probe measured.
#[derive(Clone, Debug)]
pub struct Probe { pub h: f64, pub n: usize, pub g: f64, pub u: f64, pub wall_max: f64, pub fy_max: f64, pub raw_max: f64, pub verdict: &'static str, pub secs: f64, pub solves: usize, pub scf_steps: usize, pub gpu: Duration, pub warnings: Vec<String>,
                   /// The fitted box wall (for `ripple`'s wall-subtracted energies).
                   pub wall: Wall }

/// One §6.6 probe of `label` (gate run + transfer check), with its record when it has one: (G_rec, u_I,rec, PASS).
/// Asserted: the instrument (u_I ≤ `QUALIFY`, wall ≤ `WALL_MAX`); with a PASS record G + 2u_I < `GATE`; with a MISS
/// record the ratchet G ≤ G_rec + 2(u_I + u_I,rec), each u_I ≥ `U_FLOOR` (spec §6.6 "Gate and fallback"). Without a
/// record: report only.
pub fn probe(ctx: &gpu::GpuContext, label: &str, h: f64, box_bohr: f64, s: Arc<IonicSetup>, at: At, mult: Option<u32>, record: Option<(f64, f64, bool)>) -> Probe {
    let t0 = Instant::now();
    let run = gate_run(ctx, h, box_bohr, s.clone(), at, mult);
    let big_g = run.print_summary(label);
    let tr = transfer_check(ctx, &run, box_bohr, s, at, mult);
    let pi = std::f64::consts::PI;
    let (probes, wall) = run.probes();
    let q = h / 4.0;
    let raw_max = probes.iter().map(|(p, f)| { let fw = wall.force(p.map(|c| c as f64 * q)); norm([f[0] + fw[0], f[1] + fw[1], f[2] + fw[2]]) }).fold(0.0, f64::max);
    let (wmax, fy) = (wall_max(&wall, h), probes.iter().map(|(_, f)| f[1].abs()).fold(0.0f64, f64::max));
    let v = verdict(big_g, tr.u);
    eprintln!("{label} at h = {h}, {box_bohr} Bohr: G = {big_g:.3e} Ha/Bohr (raw max {raw_max:.3e}, wall max {wmax:.2e}); transfer: decisive probe {:?} axis {}; σ_E {:.2e} Ha → {:.2e}, b₂ {:+.2e} Ha → {:.2e}, a₃ {:+.2e} Ha → {:.2e}; \
               u_I {:.2e} Ha/Bohr ({}; the n = 4 proxy (8π/h)|a₄| {:.1e}, not in u_I); §6.6 verdict: {v}; max |ΣF_y| {fy:.2e} (the all-electron figure of record 3.4e-4: {:.1}× vector, {:.1}× on y); periodicity Ẽ(−h/2) − Ẽ(h/2) {:+.2e} Ha; scan (a_n, b_n) [{}]; {:.0} s",
              tr.probe, tr.axis, tr.sigma_e, pi * 2f64.sqrt() / h * tr.sigma_e, tr.b2, 4.0 * pi / h * tr.b2.abs(), tr.a3, 8.0 * pi / h * tr.a3.abs(), tr.u,
              if tr.u <= QUALIFY { "transfers" } else { "does not transfer" }, 8.0 * pi / h * tr.coeffs[4].0.abs(), AE_FIGURE / big_g, AE_FIGURE / fy, tr.periodicity,
              tr.coeffs.iter().map(|(a, b)| format!("({a:+.3e}, {b:+.3e})")).collect::<Vec<_>>().join(", "), t0.elapsed().as_secs_f64());
    assert!(tr.u <= QUALIFY && wmax <= WALL_MAX, "{label} at h = {h}: the instrument fails (u_I {:.2e} vs {QUALIFY:.0e}, wall {wmax:.2e} vs {WALL_MAX:.0e})", tr.u);
    match record {
        Some((_, _, true)) => {
            eprintln!("{label} at h = {h}: PASS asserted, G + 2u_I = {:.3e} < {GATE:.0e}", big_g + 2.0 * tr.u);
            assert!(big_g + 2.0 * tr.u < GATE, "{label} at h = {h}: G {big_g:.3e} + 2u_I {:.2e} misses the 3e-5 gate it passed at record", tr.u);
        }
        Some((g_rec, u_rec, false)) => {
            let bound = g_rec + 2.0 * (tr.u.max(U_FLOOR) + u_rec.max(U_FLOOR));
            eprintln!("{label} at h = {h}: MISS on record (spec §6.6 / §7 risk 1); ratchet G ≤ {bound:.3e}");
            assert!(big_g <= bound, "{label} at h = {h}: G {big_g:.3e} regressed past the record {g_rec:.3e} (ratchet {bound:.3e})");
        }
        None => eprintln!("{label} at h = {h}: no record at this spacing (report only; verdict {v})"),
    }
    Probe { h, n: run.n, g: big_g, u: tr.u, wall_max: wmax, fy_max: fy, raw_max, verdict: v, secs: t0.elapsed().as_secs_f64(), solves: run.solves + 8, scf_steps: run.scf_steps, gpu: run.gpu, warnings: run.warnings.clone(), wall }
}

/// The energy ripple over the eight rigid shifts {0, h/2}³ (spec §6.6: for an inversion-symmetric molecule the eight ±h/4
/// energies are blind to the first reciprocal shell, cos(2πR_a/h) = 0 there; the {0, h/2}³ energies are not): f64 energies
/// of cold GPU states, raw and with `p.wall`'s energy removed. Returns (rms raw, rms wall-subtracted, the eight wall-subtracted
/// energies relative to their mean), Hartree.
pub fn ripple(ctx: &gpu::GpuContext, p: &Probe, box_bohr: f64, s: Arc<IonicSetup>, at: At, mult: Option<u32>) -> (f64, f64, Vec<f64>) {
    let h = p.h;
    let g = Grid::cube(p.n, h);
    debug_assert_eq!(p.n, 2 * (box_bohr / (2.0 * h)).round() as usize + 1);
    let mut solver = { let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner()); GpuSolver::new(ctx.clone(), &g) };
    let mut poisson = qm::poisson::PoissonSolver::new(&g);
    let half = h / 2.0;
    let (mut raw, mut cor) = (Vec::new(), Vec::new());
    for sx in [0.0, half] { for sy in [0.0, half] { for sz in [0.0, half] {
        let shift = [sx, sy, sz];
        let nuc = at(shift);
        let (_, st, _, _) = gpu_solve(&mut solver, s.clone(), nuc.clone(), mult, false);
        let e = ks_energy_f64(&g, &occupied(&g, &st), &st.v_h, &s, &nuc, &mut poisson).0.total;
        raw.push(e);
        cor.push(e - p.wall.energy(shift));
    }}}
    let rms = |v: &[f64]| { let m = v.iter().sum::<f64>() / v.len() as f64; (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt() };
    let m = cor.iter().sum::<f64>() / cor.len() as f64;
    (rms(&raw), rms(&cor), cor.iter().map(|x| x - m).collect())
}
