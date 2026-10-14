//! The spherical all-electron Kohn–Sham atom, LSDA, on the universal log grid (spec §4.1). CPU, f64.
//! Atomic units throughout; V is the electron's potential energy (Hartree); u = r·R (u-form),
//! ∫u² dr = 1; ρ_σ is the 3-D spin density (e/Bohr³); spin index 0 = ↑, 1 = ↓.
//!
//! # Equations
//! For each occupied (n, l, σ): −½u″ + [l(l+1)/2r² + V_σ(r)]u = εu (`numerov`), with
//!   V_σ = −Z/r + V_H[ρ] + v_xc,σ[ρ↑, ρ↓],  ρ_σ(r) = Σ_nl f_nlσ u_nlσ(r)² / (4πr²),
//!   V_H(r) = Q(r)/r + ∫_r^∞ 4πr′ρ(r′) dr′,  Q(r) = ∫_0^r 4πr′²ρ(r′) dr′  (Gauss's law, `hartree_potential`).
//! v_xc is the per-point `XcFn` — `qm::xc::exc_point` (Dirac + PW92) unless a test passes another.
//!
//! # Occupations — derived, never tabulated (rule 1)
//! The integer aufbau of the 3-D solver applied radially: each SCF step fills, per spin channel, the
//! lowest eigenvalues (ties to the lower l), at most 2l+1 electrons per (n, l) per spin (2(2l+1) in the
//! restricted mode), the last shell's electrons spread evenly over m (the spherical average). The
//! electron split comes from the multiplicity M = N↑ − N↓ + 1; `ground_state` runs ruling S9's scan
//! (`allowed_multiplicities`: the parity of N, up to `LEDGER.multiplicity_max`, while M − 1 ≤ N) and
//! keeps the lowest total energy. Near-degenerate shells: every pattern that moves one electron (one
//! spin channel; ½ + ½ restricted) from the highest occupied shell to a partially filled or empty
//! shell of another l whose eigenvalue lies within Δ_scan = `LEDGER.delta_scan` of it is solved with
//! fixed occupations; the lowest total energy wins. For Z ≤ 18 the scan is a no-op (tested).
//!
//! # SCF
//! Pulay/DIIS on the stacked [ρ↑, ρ↓] (the 3-D `scf.rs` algorithm on 1-D vectors, inner product
//! ∫ a b r² dr), history `LEDGER.numerics.diis_history`, step α = `mixing`; converged when
//! Σ_σ ∫|ρ_σ^out − ρ_σ^in| r² dr < `scf_tol` (1e-10). The restricted mode is the same code with
//! ρ↑ = ρ↓ (each spin entry holds half of a shell's occupation) — the reference of §4.3.
//!
//! # Energy (the `scf.rs` discipline: output density and orbitals)
//!   T_s = Σ f ε − Σ_σ ∫ρ_σ V_σ^in d³r,  E_en = −Z ∫ρ/r d³r,  E_H = ½∫ρ V_H[ρ] d³r,
//!   E_xc = ∫ n ε_xc(ρ↑, ρ↓) d³r,  E = T_s + E_en + E_H + E_xc.
//! All integrals are trapezoids in x = ln r with Neumaier summation (`LogGrid::integrate`).
//!
//! # The LSDA scaling identity (spec §4.1, review M2) — the virial check that is actually true
//! Under ψ_λ(r) = λ^{3/2}ψ(λr), ρ_λ(r) = λ³ρ(λr): T_s → λ²T_s, E_en → λE_en, E_H → λE_H, while
//! E_xc^{LDA}[ρ_λ] = ∫ρ(r′) ε_xc(λ³ρ↑(r′), λ³ρ↓(r′)) d³r′ has λ-derivative at 1 equal to
//! 3∫ρ Σ_σ ρ_σ ∂ε_xc/∂ρ_σ = 3[Σ_σ∫ρ_σ v_xc,σ − E_xc] (with v_xc,σ = ε_xc + ρ ∂ε_xc/∂ρ_σ).
//! Stationarity of the total energy at λ = 1 gives
//!   2T_s + E_en + E_H + 3[Σ_σ ∫ρ_σ v_xc,σ d³r − E_xc] = 0.
//! For Dirac exchange ρ ∂ε_x/∂ρ = ε_x/3, so the exchange part of the bracket is E_x and the textbook
//! 2T + V = 0 is recovered; PW92 correlation breaks the textbook form at the 10⁻³–10⁻² level, so the
//! generalised identity is what `scaling_identity_residual` measures (divided by T_s).
//!
//! # Outputs
//! ε_nlσ, u_nlσ, ρ_σ, V_σ (of the output density), V_H, the energy components, and per shell the
//! outermost maximum r_peak of |u| and the outermost node r_node (of the ↑ entry, or ↓ when ↑ is empty).
use super::grid::LogGrid;
use super::ledger::LEDGER;
use super::numerov::{bound_state, BoundState};
use crate::grid::neumaier;
use crate::xc::exc_point;
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;

pub type XcFn = fn(f64, f64) -> (f64, f64, f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinMode { Polarised, Restricted }

#[derive(Clone, Debug, PartialEq)]
pub struct ShellOcc { pub n: u8, pub l: u8, pub occ: [f64; 2] }

#[derive(Clone, Debug, PartialEq)]
pub enum Occupations { Aufbau { multiplicity: u32 }, Fixed(Vec<ShellOcc>) }

#[derive(Clone, Debug)]
pub struct AtomOptions { pub spin: SpinMode, pub occupations: Occupations, pub charge: i32, pub xc: XcFn }

impl AtomOptions {
    pub fn aufbau(spin: SpinMode, multiplicity: u32) -> AtomOptions {
        AtomOptions { spin, occupations: Occupations::Aufbau { multiplicity }, charge: 0, xc: exc_point }
    }
    pub fn fixed(spin: SpinMode, shells: Vec<ShellOcc>) -> AtomOptions {
        AtomOptions { spin, occupations: Occupations::Fixed(shells), charge: 0, xc: exc_point }
    }
}

#[derive(Clone, Debug)]
pub struct Orbital { pub n: u8, pub l: u8, pub eps: [f64; 2], pub occ: [f64; 2], pub u: [Vec<f64>; 2], pub r_peak: f64, pub r_node: Option<f64> }

#[derive(Clone, Copy, Debug)]
pub struct AtomEnergy { pub total: f64, pub kinetic: f64, pub nuclear: f64, pub hartree: f64, pub xc: f64, pub xc_potential: f64 }

#[derive(Clone, Debug)]
pub struct AtomResult {
    pub z: u8, pub spin: SpinMode, pub multiplicity: u32, pub orbitals: Vec<Orbital>,
    pub rho: [Vec<f64>; 2], pub v: [Vec<f64>; 2], pub v_h: Vec<f64>, pub energy: AtomEnergy,
    pub iterations: usize, pub grid: LogGrid,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RadialError {
    #[error("radial SCF for Z = {z} did not converge after {iterations} steps (Δρ = {delta:.2e})")]
    NotConverged { z: u8, iterations: usize, delta: f64 },
    #[error("Z = {z}: shell n = {n}, l = {l} is not bound in this potential")]
    Unbound { z: u8, n: u8, l: u8 },
    #[error("impossible occupation: {0}")]
    Occupation(String),
}

/// Ruling S9's rule (`sim::sampling::allowed_multiplicities`): the parity of N, from the lowest up
/// to `LEDGER.multiplicity_max`, while 2S ≤ N.
pub fn allowed_multiplicities(electrons: u32) -> Vec<u32> {
    (1 + electrons % 2..=LEDGER.multiplicity_max).step_by(2).filter(|&m| m - 1 <= electrons).collect()
}

/// V_H(r) = Q(r)/r + ∫_r^∞ 4πr′ρ dr′ (Gauss's law; both integrals cumulative, 4th order).
pub fn hartree_potential(grid: &LogGrid, rho: &[f64]) -> Vec<f64> {
    let q = grid.cumulative(&grid.r.iter().zip(rho).map(|(r, p)| 4.0 * PI * r * r * p).collect::<Vec<_>>());
    let s = grid.cumulative(&grid.r.iter().zip(rho).map(|(r, p)| 4.0 * PI * r * p).collect::<Vec<_>>());
    let s_tot = s[s.len() - 1];
    grid.r.iter().enumerate().map(|(i, r)| q[i] / r + (s_tot - s[i])).collect()
}

/// (2T_s + E_en + E_H + 3[Σ_σ∫ρ_σ v_xc,σ − E_xc]) / T_s — zero for a converged LSDA atom (spec §4.1).
pub fn scaling_identity_residual(a: &AtomResult) -> f64 {
    let e = &a.energy;
    (2.0 * e.kinetic + e.nuclear + e.hartree + 3.0 * (e.xc_potential - e.xc)) / e.kinetic
}

fn spin_counts(n: u32, m: u32) -> Result<(u32, u32), RadialError> {
    if m == 0 || m - 1 > n || (n - (m - 1)) % 2 != 0 {
        return Err(RadialError::Occupation(format!("multiplicity {m} impossible for {n} electrons")));
    }
    Ok(((n + m - 1) / 2, (n + 1 - m) / 2))
}

type Shell = (u8, u8, f64, BoundState);

/// One spin channel's integer aufbau over eigenvalues (ties to the lower l), spherically averaged:
/// the last shell's occupation is spread over its m. `cap(l)`: 2l+1 per spin channel, 2(2l+1) restricted.
fn aufbau_channel(grid: &LogGrid, v: &[f64], z: f64, electrons: f64, cap: impl Fn(u8) -> f64, prev: &[(u8, u8, f64)]) -> Result<Vec<Shell>, RadialError> {
    let guess = |n: u8, l: u8| prev.iter().find(|p| p.0 == n && p.1 == l).map(|p| p.2).filter(|e| e.is_finite());
    let mut next: Vec<Option<(u8, BoundState)>> =
        (0..=3u8).map(|l| bound_state(grid, v, l, 0, z, guess(l + 1, l)).map(|b| (l + 1, b))).collect();
    let mut out = Vec::new();
    let mut left = electrons;
    while left > 1e-12 {
        let (l, _) = next.iter().enumerate()
            .filter_map(|(l, s)| s.as_ref().map(|(_, b)| (l, b.e)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .ok_or_else(|| RadialError::Occupation(format!("{left} electrons left and no bound shell")))?;
        let (n, b) = next[l].take().expect("picked above");
        let occ = left.min(cap(l as u8));
        left -= occ;
        out.push((n, l as u8, occ, b));
        next[l] = bound_state(grid, v, l as u8, (n - l as u8) as usize, z, guess(n + 1, l as u8)).map(|b| (n + 1, b));
    }
    Ok(out)
}

fn density(grid: &LogGrid, shells: &[Shell]) -> Vec<f64> {
    (0..grid.len()).map(|i| {
        let r = grid.r[i];
        shells.iter().map(|s| s.2 * s.3.u[i] * s.3.u[i]).sum::<f64>() / (4.0 * PI * r * r)
    }).collect()
}

fn potentials(grid: &LogGrid, z: f64, rho: &[Vec<f64>; 2], xc: XcFn) -> [Vec<f64>; 2] {
    let n = grid.len();
    let total: Vec<f64> = (0..n).map(|i| rho[0][i] + rho[1][i]).collect();
    let v_h = hartree_potential(grid, &total);
    let mut v = [vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (_, vu, vd) = xc(rho[0][i], rho[1][i]);
        let c = -z / grid.r[i] + v_h[i];
        v[0][i] = c + vu;
        v[1][i] = c + vd;
    }
    v
}

fn merge(channels: &[Vec<Shell>; 2]) -> Vec<Orbital> {
    let mut keys: Vec<(u8, u8)> = channels.iter().flatten().map(|s| (s.0, s.1)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter().map(|(n, l)| {
        let get = |s: usize| channels[s].iter().find(|x| x.0 == n && x.1 == l);
        let (a, b) = (get(0), get(1));
        let peak_src = a.or(b).expect("present in one channel");
        Orbital {
            n, l,
            eps: [a.map_or(f64::NAN, |x| x.3.e), b.map_or(f64::NAN, |x| x.3.e)],
            occ: [a.map_or(0.0, |x| x.2), b.map_or(0.0, |x| x.2)],
            u: [a.map_or(Vec::new(), |x| x.3.u.clone()), b.map_or(Vec::new(), |x| x.3.u.clone())],
            r_peak: peak_src.3.r_peak, r_node: peak_src.3.r_node,
        }
    }).collect()
}

type Occupied = (Vec<Orbital>, [Vec<f64>; 2]);

fn occupy(grid: &LogGrid, z: u8, v: &[Vec<f64>; 2], opts: &AtomOptions, prev: &[Orbital]) -> Result<Occupied, RadialError> {
    let zf = z as f64;
    let prev_eps = |s: usize| prev.iter().map(|o| (o.n, o.l, o.eps[s])).collect::<Vec<_>>();
    let electrons = z as i32 - opts.charge;
    if electrons < 0 {
        return Err(RadialError::Occupation(format!("charge {} exceeds Z = {z}", opts.charge)));
    }
    let channels: [Vec<Shell>; 2] = match (&opts.occupations, opts.spin) {
        (Occupations::Aufbau { .. }, SpinMode::Restricted) => {
            let ch = aufbau_channel(grid, &v[0], zf, electrons as f64, |l| 2.0 * (2 * l + 1) as f64, &prev_eps(0))?;
            let half: Vec<Shell> = ch.into_iter().map(|(n, l, o, b)| (n, l, 0.5 * o, b)).collect();
            [half.clone(), half]
        }
        (Occupations::Aufbau { multiplicity }, SpinMode::Polarised) => {
            let (up, dn) = spin_counts(electrons as u32, *multiplicity)?;
            [aufbau_channel(grid, &v[0], zf, up as f64, |l| (2 * l + 1) as f64, &prev_eps(0))?,
             aufbau_channel(grid, &v[1], zf, dn as f64, |l| (2 * l + 1) as f64, &prev_eps(1))?]
        }
        (Occupations::Fixed(shells), spin) => {
            let mut ch: [Vec<Shell>; 2] = [Vec::new(), Vec::new()];
            for s in shells {
                for sp in 0..2 {
                    if s.occ[sp] <= 0.0 { continue; }
                    let src = if spin == SpinMode::Restricted { 0 } else { sp };
                    let g = prev.iter().find(|o| o.n == s.n && o.l == s.l).map(|o| o.eps[src]).filter(|e| e.is_finite());
                    let b = bound_state(grid, &v[src], s.l, (s.n - s.l - 1) as usize, zf, g)
                        .ok_or(RadialError::Unbound { z, n: s.n, l: s.l })?;
                    ch[sp].push((s.n, s.l, s.occ[sp], b));
                }
            }
            ch
        }
    };
    let rho = [density(grid, &channels[0]), density(grid, &channels[1])];
    Ok((merge(&channels), rho))
}

pub fn solve_atom(z: u8, opts: &AtomOptions) -> Result<AtomResult, RadialError> {
    let grid = LogGrid::universal();
    let n = grid.len();
    let zf = z as f64;
    let num = &LEDGER.numerics;
    // Start from the bare nucleus (ρ_in = 0): the first output is the hydrogenic density, and the
    // first mixing step (one history entry = linear mixing) feeds in only α of it. Starting from the
    // full hydrogenic density instead over-screens: those orbitals are Z-times too compact, so their
    // Hartree potential cancels −Z/r outside ~1 Bohr and unbinds the outer shells (C: no bound 2p).
    let mut orbitals: Vec<Orbital> = Vec::new();
    let mut rho_in = [vec![0.0; n], vec![0.0; n]];
    // the spec's Δρ measure ∫|Δρ| r² dr, and the DIIS inner product ∫ a b r² dr, on the log grid
    let w: Vec<f64> = grid.r.iter().map(|r| r * r * r * grid.delta).collect();
    let mut diis = Diis::new(num.diis_history);
    let mut delta = f64::INFINITY;
    for it in 1..=num.scf_max_iter {
        let v = potentials(&grid, zf, &rho_in, opts.xc);
        let (o, rho_out) = occupy(&grid, z, &v, opts, &orbitals)?;
        orbitals = o;
        delta = (0..2).map(|s| (0..n).map(|i| (rho_out[s][i] - rho_in[s][i]).abs() * w[i]).sum::<f64>()).sum();
        if delta < num.scf_tol {
            return Ok(finish(z, grid, opts, orbitals, rho_out, &v, it));
        }
        rho_in = diis.mix(rho_in, rho_out, &w, num.mixing);
    }
    Err(RadialError::NotConverged { z, iterations: num.scf_max_iter, delta })
}

/// The energy from the output density and orbitals (the `scf.rs` discipline): T_s = Σfε − Σ_σ∫ρ_σ V_σ^in.
fn finish(z: u8, grid: LogGrid, opts: &AtomOptions, orbitals: Vec<Orbital>, rho: [Vec<f64>; 2], v_in: &[Vec<f64>; 2], iterations: usize) -> AtomResult {
    let n = grid.len();
    let zf = z as f64;
    let shell = |f: &[f64]| grid.integrate(&f.iter().zip(&grid.r).map(|(a, r)| 4.0 * PI * r * r * a).collect::<Vec<_>>());
    let total: Vec<f64> = (0..n).map(|i| rho[0][i] + rho[1][i]).collect();
    let band = neumaier(orbitals.iter().flat_map(|o| (0..2).filter(|&s| o.occ[s] > 0.0).map(|s| o.occ[s] * o.eps[s])));
    let pot = neumaier((0..2).map(|s| shell(&(0..n).map(|i| rho[s][i] * v_in[s][i]).collect::<Vec<_>>())));
    let kinetic = band - pot;
    let nuclear = -zf * grid.integrate(&(0..n).map(|i| 4.0 * PI * grid.r[i] * total[i]).collect::<Vec<_>>());
    let v_h = hartree_potential(&grid, &total);
    let hartree = 0.5 * shell(&(0..n).map(|i| total[i] * v_h[i]).collect::<Vec<_>>());
    let (mut exc, mut vr) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let (e, vu, vd) = (opts.xc)(rho[0][i], rho[1][i]);
        exc[i] = e;
        vr[i] = rho[0][i] * vu + rho[1][i] * vd;
    }
    let (xc, xc_potential) = (shell(&exc), shell(&vr));
    let up: f64 = orbitals.iter().map(|o| o.occ[0]).sum();
    let dn: f64 = orbitals.iter().map(|o| o.occ[1]).sum();
    let v = potentials(&grid, zf, &rho, opts.xc);
    AtomResult {
        z, spin: opts.spin, multiplicity: ((up - dn).abs().round() as u32) + 1, orbitals, rho, v, v_h,
        energy: AtomEnergy { total: neumaier([kinetic, nuclear, hartree, xc].into_iter()), kinetic, nuclear, hartree, xc, xc_potential },
        iterations, grid,
    }
}

/// Pulay/DIIS on the stacked [ρ↑, ρ↓] (the 3-D `scf.rs` algorithm on 1-D vectors, inner product
/// ∫ a b r² dr): returns Σc_i(ρ_in,i + α r_i), c minimising ‖Σ c_i r_i‖ with Σ c_i = 1; B scaled by
/// its largest diagonal plus a 1e-12 relative regularisation; a failed solve drops the oldest entry.
struct Diis { m: usize, hist: Vec<(Vec<f64>, Vec<f64>)> }

impl Diis {
    fn new(m: usize) -> Diis { Diis { m: m.max(1), hist: Vec::new() } }
    fn mix(&mut self, rho_in: [Vec<f64>; 2], rho_out: [Vec<f64>; 2], w: &[f64], alpha: f64) -> [Vec<f64>; 2] {
        let n = w.len();
        let x: Vec<f64> = rho_in[0].iter().chain(rho_in[1].iter()).copied().collect();
        let y: Vec<f64> = rho_out[0].iter().chain(rho_out[1].iter()).copied().collect();
        let r: Vec<f64> = y.iter().zip(&x).map(|(a, b)| a - b).collect();
        self.hist.push((x, r));
        if self.hist.len() > self.m { self.hist.remove(0); }
        loop {
            let k = self.hist.len();
            let mut b = DMatrix::<f64>::zeros(k + 1, k + 1);
            for i in 0..k {
                for j in 0..k {
                    b[(i, j)] = (0..2 * n).map(|p| self.hist[i].1[p] * self.hist[j].1[p] * w[p % n]).sum();
                }
            }
            let scale = (0..k).map(|i| b[(i, i)]).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
            for i in 0..k {
                for j in 0..k { b[(i, j)] /= scale; }
                b[(i, i)] += 1e-12;
                b[(i, k)] = 1.0;
                b[(k, i)] = 1.0;
            }
            let mut rhs = DVector::<f64>::zeros(k + 1);
            rhs[k] = 1.0;
            match b.lu().solve(&rhs) {
                Some(c) if c.iter().all(|v| v.is_finite()) => {
                    let mut out = vec![0.0; 2 * n];
                    for i in 0..k {
                        for p in 0..2 * n { out[p] += c[i] * (self.hist[i].0[p] + alpha * self.hist[i].1[p]); }
                    }
                    for v in &mut out { *v = v.max(0.0); }
                    return [out[..n].to_vec(), out[n..].to_vec()];
                }
                _ if k > 1 => { self.hist.remove(0); }
                _ => unreachable!("one entry is linear mixing and always solves"),
            }
        }
    }
}

/// The lowest unoccupied state of angular momentum l in spin channel `spin` of a converged atom.
pub fn next_unoccupied(a: &AtomResult, l: u8, spin: usize) -> Option<(u8, BoundState)> {
    let filled = a.orbitals.iter().filter(|o| o.l == l && o.occ[spin] > 0.0).map(|o| o.n).max();
    let n = filled.map_or(l + 1, |n| n + 1);
    bound_state(&a.grid, &a.v[spin], l, (n - l - 1) as usize, a.z as f64, None).map(|b| (n, b))
}

pub fn ground_state(z: u8, spin: SpinMode) -> Result<AtomResult, RadialError> { ground_state_with(z, spin, exc_point) }

/// Restricted: the aufbau solve plus the Δ_scan scan. Polarised: that for every allowed multiplicity
/// (ruling S9); the lowest total energy wins.
pub fn ground_state_with(z: u8, spin: SpinMode, xc: XcFn) -> Result<AtomResult, RadialError> {
    let with = |m| AtomOptions { xc, ..AtomOptions::aufbau(spin, m) };
    match spin {
        SpinMode::Restricted => scanned(z, &with(1)),
        SpinMode::Polarised => {
            let (mut best, mut err): (Option<AtomResult>, Option<RadialError>) = (None, None);
            for m in allowed_multiplicities(z as u32) {
                match scanned(z, &with(m)) {
                    Ok(a) if best.as_ref().is_none_or(|b| a.energy.total < b.energy.total) => best = Some(a),
                    Ok(_) => {}
                    Err(e) => err = Some(e),
                }
            }
            best.ok_or_else(|| err.expect("at least one multiplicity was tried"))
        }
    }
}

/// Spec §4.1 "near-degenerate shells": from the aufbau solution, every pattern moving one electron
/// (in one spin channel; ½ + ½ when restricted) from the highest occupied shell to a partially filled
/// or empty shell of another l whose eigenvalue is within LEDGER.delta_scan of it is solved with
/// fixed occupations; the lowest total energy wins. No configuration table exists (rule 1).
fn scanned(z: u8, opts: &AtomOptions) -> Result<AtomResult, RadialError> {
    let base = solve_atom(z, opts)?;
    let mut best = base.clone();
    for alt in scan_alternatives(&base) {
        let o = AtomOptions { occupations: Occupations::Fixed(alt), ..opts.clone() };
        if let Ok(a) = solve_atom(z, &o) {
            if a.energy.total < best.energy.total { best = a; }
        }
    }
    Ok(best)
}

fn scan_alternatives(a: &AtomResult) -> Vec<Vec<ShellOcc>> {
    let restricted = a.spin == SpinMode::Restricted;
    let base: Vec<ShellOcc> = a.orbitals.iter().map(|o| ShellOcc { n: o.n, l: o.l, occ: o.occ }).collect();
    let mut out = Vec::new();
    for s in 0..(if restricted { 1 } else { 2 }) {
        let cap = |l: u8| (2 * l + 1) as f64 * if restricted { 2.0 } else { 1.0 };
        let total = |o: &ShellOcc| if restricted { o.occ[0] + o.occ[1] } else { o.occ[s] };
        let Some(homo) = a.orbitals.iter().filter(|o| o.occ[s] > 0.0).max_by(|x, y| x.eps[s].total_cmp(&y.eps[s])) else { continue };
        for l in (0..=3u8).filter(|&l| l != homo.l) {
            let partial = a.orbitals.iter().find(|o| o.l == l && total(&ShellOcc { n: o.n, l, occ: o.occ }) < cap(l) && o.occ[s] > 0.0);
            let (n, eps) = match partial {
                Some(o) => (o.n, o.eps[s]),
                None => match next_unoccupied(a, l, s) { Some((n, b)) => (n, b.e), None => continue },
            };
            if (eps - homo.eps[s]).abs() >= LEDGER.delta_scan { continue; }
            let mut alt = base.clone();
            let take = if restricted { [0.5, 0.5] } else if s == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
            if let Some(h) = alt.iter_mut().find(|x| x.n == homo.n && x.l == homo.l) {
                for k in 0..2 { h.occ[k] -= take[k]; }
            }
            match alt.iter_mut().find(|x| x.n == n && x.l == l) {
                Some(t) => for k in 0..2 { t.occ[k] += take[k]; },
                None => alt.push(ShellOcc { n, l, occ: take }),
            }
            alt.retain(|x| x.occ[0] + x.occ[1] > 0.0);
            out.push(alt);
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::numerov::bound_state;

    /// The functional of the NIST LSD column, like-for-like (Spec question 1). Step 1 found that NIST
    /// used Vosko–Wilk–Nusair (the procedure page, "The local-density functional"), so VWN5 is kept.
    const NIST_XC: XcFn = vwn5::exc_point;

    fn lsd(z: u8) -> f64 { ground_state_with(z, SpinMode::Polarised, NIST_XC).expect("converges").energy.total }

    /// Spec §4.1 / §8 row 1, the first failing test.
    #[test]
    fn h_atom_radial_lsd() {
        let e = lsd(1);
        assert!((e - (-0.478671)).abs() < 1e-5, "H LSD = {e:.7} vs NIST −0.478671");
    }

    #[test]
    fn he_atom_radial_lsd() {
        let e = lsd(2);
        assert!((e - (-2.834836)).abs() < 1e-5, "He LSD = {e:.7} vs NIST −2.834836");
    }

    /// NIST's Li page prints LDA −7.335195 and LSD −7.343957 (read 2026-10-15): the brief's and
    /// references.md's "Li LSD −7.335195" is the LDA column. Like-for-like here: the spin-polarised
    /// atom against the LSD column; the brief's digits are tested where they belong, in
    /// `restricted_radial_lda_matches_nist_lda`.
    #[test]
    fn li_atom_radial_lsd() {
        let e = lsd(3);
        assert!((e - (-7.343957)).abs() < 1e-5, "Li LSD = {e:.7} vs NIST −7.343957");
    }

    /// The spin-restricted mode (the §4.3 reference) against NIST's LDA column — spherical,
    /// unpolarised, VWN, the same occupations (C, N, O: 2p spread evenly over m and spin). Read from
    /// the element pages on 2026-10-15; cited in references.md "M7 radial atom (Task 1)".
    #[test]
    fn restricted_radial_lda_matches_nist_lda() {
        let nist_lda = [(1u8, -0.445671), (2, -2.834836), (3, -7.335195), (6, -37.425749), (7, -54.025016),
                        (8, -74.473077), (10, -128.233481)];
        for (z, nist) in nist_lda {
            let e = ground_state_with(z, SpinMode::Restricted, NIST_XC).expect("converges").energy.total;
            let tol = if z <= 3 { 1e-5 } else { 1e-4 };
            assert!((e - nist).abs() < tol, "Z = {z}: LDA = {e:.7} vs NIST {nist:.6}");
        }
    }

    // NIST LSD totals of C, N, O, Ne, read in Step 1 (spec §4.1: "read off at task time"; cited in
    // references.md "M7 radial atom"): the LSD column of the element pages of Kotochigova et al. 1997.
    const NIST_LSD_C: f64 = -37.470031;
    const NIST_LSD_N: f64 = -54.136799;
    const NIST_LSD_O: f64 = -74.527410;
    const NIST_LSD_NE: f64 = -128.233481;

    /// The second-row LSD totals to 1e-4 (spec §4.1).
    #[test]
    fn second_row_radial_lsd() {
        for (z, nist) in [(6u8, NIST_LSD_C), (7, NIST_LSD_N), (8, NIST_LSD_O), (10, NIST_LSD_NE)] {
            let e = lsd(z);
            assert!((e - nist).abs() < 1e-4, "Z = {z}: LSD = {e:.6} vs NIST {nist:.6}");
        }
    }

    /// Spec §4.1: Hund's triplet is the lowest of M = 1, 3, 5 for O in the spherical LSDA atom (PW92).
    #[test]
    fn o_triplet_is_lowest() {
        let e = |m| solve_atom(8, &AtomOptions::aufbau(SpinMode::Polarised, m)).expect("converges").energy.total;
        let (e1, e3, e5) = (e(1), e(3), e(5));
        assert!(e3 < e1 && e3 < e5, "O: E(M=1) = {e1:.6}, E(3) = {e3:.6}, E(5) = {e5:.6}");
        assert_eq!(ground_state(8, SpinMode::Polarised).unwrap().multiplicity, 3);
    }

    /// Spec §4.1 (review M2): 2T_s + E_en + E_H + 3[Σ_σ∫ρ_σ v_xc,σ − E_xc] = 0 to 1e-5 relative to T_s,
    /// for every converged atom, both spin modes (PW92, the generator's functional).
    #[test]
    fn lsda_scaling_identity_holds() {
        for z in [1u8, 2, 3, 6, 7, 8, 10] {
            for spin in [SpinMode::Polarised, SpinMode::Restricted] {
                let a = ground_state(z, spin).expect("converges");
                let r = scaling_identity_residual(&a);
                assert!(r.abs() < 1e-5, "Z = {z} {spin:?}: residual {r:.2e} (T_s {:.6})", a.energy.kinetic);
            }
        }
    }

    /// The integrator alone (V_H and v_xc off): |ε_nl − (−Z²/2n²)| < 1e-10·Z² — spec revision M1
    /// (plan review 2026-10-15): the error is relative, ∝ δ⁴, so the absolute bound scales as Z² (the
    /// log grid is scale-free). Measured at δ = 0.008 (2026-10-15): Δε/Z² = −4.0e-12 … −4.0e-11 (worst
    /// Z = 2 3s, −1.59e-10 = −3.98e-11·Z², margin 2.5×; Z = 8: 1s −1.11e-9, 2s −1.55e-9). Plus the order
    /// check of M1: the Z = 1 1s error ratio between δ = 0.016 and 0.008 lies in [14, 18] (measured
    /// 16.05), which catches a wrong Numerov weight that one absolute bound would let through.
    #[test]
    fn hydrogenic_eigenvalues_are_exact() {
        let g = LogGrid::universal();
        let shells: [(f64, &[(u8, u8)]); 4] = [
            (1.0, &[(1, 0), (2, 0), (2, 1), (3, 2)]),
            (2.0, &[(1, 0), (2, 0), (2, 1), (3, 0), (3, 2), (4, 3)]),
            (3.0, &[(1, 0), (2, 1), (3, 1), (4, 3)]),
            (8.0, &[(1, 0), (2, 0), (2, 1), (3, 2), (4, 3)]),
        ];
        for (z, list) in shells {
            let v: Vec<f64> = g.r.iter().map(|r| -z / r).collect();
            for &(n, l) in list {
                let b = bound_state(&g, &v, l, (n - l - 1) as usize, z, None).expect("bound");
                let exact = -z * z / (2.0 * (n as f64).powi(2));
                assert!((b.e - exact).abs() < 1e-10 * z * z, "Z = {z}, n = {n}, l = {l}: ε = {:.13} vs {exact:.13} (Δ {:.2e})", b.e, b.e - exact);
            }
        }
        let err = |delta: f64| {
            let g = LogGrid::new(LEDGER.r0, LEDGER.r_max, delta);
            let v: Vec<f64> = g.r.iter().map(|r| -1.0 / r).collect();
            bound_state(&g, &v, 0, 0, 1.0, None).expect("bound").e + 0.5
        };
        let ratio = err(0.016) / err(0.008);
        assert!((14.0..=18.0).contains(&ratio), "Z = 1 1s: error ratio δ 0.016 → 0.008 = {ratio:.3} (δ⁴: 16)");
    }

    /// Spec §4.1: "For Z ≤ 18 the aufbau order is unambiguous and the scan is a no-op."
    #[test]
    fn occupation_scan_is_a_noop_below_z19() {
        for z in 1u8..=18 {
            let plain = solve_atom(z, &AtomOptions::aufbau(SpinMode::Restricted, 1)).expect("converges");
            let scanned = ground_state(z, SpinMode::Restricted).expect("converges");
            let occ = |a: &AtomResult| a.orbitals.iter().map(|o| (o.n, o.l, (o.occ[0] * 1e6).round() as i64)).collect::<Vec<_>>();
            assert_eq!(occ(&plain), occ(&scanned), "Z = {z}");
        }
    }

    /// The open-shell eigenvalues of both spin channels against NIST's LSD column for O (the
    /// transferability centrepiece): NIST lists the majority spin first ("down" in their convention;
    /// our index 0). Read from the O page on 2026-10-15.
    #[test]
    fn o_lsd_eigenvalues_match_nist() {
        let a = ground_state_with(8, SpinMode::Polarised, NIST_XC).expect("converges");
        let nist = [((1u8, 0u8), [-18.766963, -18.713693]), ((2, 0), [-0.915554, -0.801638]), ((2, 1), [-0.381133, -0.272349])];
        for ((n, l), eps) in nist {
            let o = a.orbitals.iter().find(|o| o.n == n && o.l == l).expect("occupied");
            for s in 0..2 {
                assert!((o.eps[s] - eps[s]).abs() < 1e-5, "O ({n}, {l}) spin {s}: ε = {:.7} vs NIST {:.6}", o.eps[s], eps[s]);
            }
        }
    }

    /// The NIST-functional comparison is like-for-like; the generator's PW92 totals are reported against
    /// it (rule 7), never asserted.
    #[test]
    fn report_pw92_against_nist() {
        for z in [1u8, 2, 3, 6, 7, 8, 10] {
            let pw = ground_state(z, SpinMode::Polarised).unwrap().energy.total;
            let nist = lsd(z);
            eprintln!("Z = {z:2}: PW92 {pw:.6}  NIST-functional {nist:.6}  Δ {:+.2e}", pw - nist);
        }
    }

    /// The test-only VWN5 potentials are the derivatives of its energy density (central differences
    /// where they are well conditioned: both spin densities of the same order).
    #[test]
    fn vwn5_potential_is_the_derivative() {
        for &(nu, nd) in &[(0.3, 0.2), (1e-3, 4e-3), (2.0, 1.9), (0.05, 0.01), (10.0, 3.0)] {
            let (_, vu, vd) = vwn5::exc_point(nu, nd);
            let h = 1e-5;
            let e = |a: f64, b: f64| vwn5::exc_point(a, b).0;
            let fu = (e(nu * (1.0 + h), nd) - e(nu * (1.0 - h), nd)) / (2.0 * h * nu);
            let fd = (e(nu, nd * (1.0 + h)) - e(nu, nd * (1.0 - h))) / (2.0 * h * nd);
            assert!((vu - fu).abs() < 1e-8 * vu.abs(), "({nu}, {nd}): v↑ {vu:.12} vs FD {fu:.12}");
            assert!((vd - fd).abs() < 1e-8 * vd.abs(), "({nu}, {nd}): v↓ {vd:.12} vs FD {fd:.12}");
        }
    }
}

/// Vosko, Wilk & Nusair, Can. J. Phys. 58, 1200 (1980), the Ceperley–Alder fit ("VWN5") — TEST-ONLY:
/// the functional of the NIST LSD column, so the radial solver is compared like-for-like (Spec
/// question 1). The twelve constants were checked at task time (2026-10-15) against the table on
/// NIST's "Atomic Reference Data … The local-density functional / The correlation term" page, which
/// is the parametrisation NIST itself ran (P: 0.0310907, −0.10498, 3.72744, 12.9352; F: 0.01554535,
/// −0.32500, 7.06042, 18.0578; α_c: −1/(6π²), −0.0047584, 1.13107, 13.0045). libxc's source was not
/// reachable from the task environment, so it is not the cross-check.
///
/// Exchange: Dirac, spin-scaled (formulas in `qm::xc`'s module doc). Correlation (NIST eqs. 6–10):
///   ε_c(r_s, ζ) = P + α_c f(ζ)/f″(0)·(1 − ζ⁴) + (F − P) f(ζ) ζ⁴,  P, F, α_c = G(x; A, x₀, b, c), x = √r_s,
///   G = A{ ln(x²/X) + (2b/Q) atan(Q/(2x+b)) − (b x₀/X(x₀))[ ln((x−x₀)²/X) + (2(b+2x₀)/Q) atan(Q/(2x+b)) ] },
///   X(x) = x² + bx + c, Q = √(4c − b²).
/// Potentials analytically (not by differences in n_σ: a step ∝ n_σ is ill-conditioned where n_σ ≪ n,
/// relative roundoff ≈ 1e-10·n/n_σ, i.e. the whole tail of every open-shell atom):
///   d atan(Q/(2x+b))/dx = −2Q/((2x+b)² + Q²) = −Q/(2X)  (since (2x+b)² + Q² = 4X), so
///   dG/dx = A{ 2/x − (2x+2b)/X − (b x₀/X(x₀))[2/(x−x₀) − (2x+2b+2x₀)/X] }
///         = 2A[c(x − x₀) − b x₀ x] / [x X (x − x₀)]      (using X − x(x+b) = c, X − (x−x₀)(x+b+x₀) = X(x₀)),
///   dG/dr_s = (dG/dx)/(2x);
///   v_c,σ = ε_c − (r_s/3) ∂ε_c/∂r_s − (ζ − s_σ) ∂ε_c/∂ζ (s↑ = +1, s↓ = −1; derivation in `qm::xc`),
///   ∂ε_c/∂ζ = (α_c/f″(0))[f′(1 − ζ⁴) − 4ζ³f] + (F − P)[f′ζ⁴ + 4ζ³f].
#[cfg(test)]
pub(crate) mod vwn5 {
    use std::f64::consts::PI;

    /// (G, dG/dr_s) at x = √r_s.
    fn g(x: f64, a: f64, x0: f64, b: f64, c: f64) -> (f64, f64) {
        let xx = |y: f64| y * y + b * y + c;
        let q = (4.0 * c - b * b).sqrt();
        let at = (q / (2.0 * x + b)).atan();
        let val = a * ((x * x / xx(x)).ln() + 2.0 * b / q * at
            - b * x0 / xx(x0) * (((x - x0).powi(2) / xx(x)).ln() + 2.0 * (b + 2.0 * x0) / q * at));
        let dgdx = 2.0 * a * (c * (x - x0) - b * x0 * x) / (x * xx(x) * (x - x0));
        (val, dgdx / (2.0 * x))
    }

    pub fn exc_point(nu: f64, nd: f64) -> (f64, f64, f64) {
        let (nu, nd) = (nu.max(0.0), nd.max(0.0));
        let n = nu + nd;
        if n < 1e-20 { return (0.0, 0.0, 0.0); }
        let c = 0.75 * (3.0 / PI).powf(1.0 / 3.0) * 2f64.powf(1.0 / 3.0);
        let ex = -c * (nu.powf(4.0 / 3.0) + nd.powf(4.0 / 3.0));
        let (vxu, vxd) = (-(6.0 * nu / PI).powf(1.0 / 3.0), -(6.0 * nd / PI).powf(1.0 / 3.0));

        let rs = (3.0 / (4.0 * PI * n)).powf(1.0 / 3.0);
        let zeta = ((nu - nd) / n).clamp(-1.0, 1.0);
        let x = rs.sqrt();
        let (p, dp) = g(x, 0.0310907, -0.10498, 3.72744, 12.9352);
        let (f, df) = g(x, 0.01554535, -0.32500, 7.06042, 18.0578);
        let (al, dal) = g(x, -1.0 / (6.0 * PI * PI), -0.0047584, 1.13107, 13.0045);
        let d = 2f64.powf(4.0 / 3.0) - 2.0;
        let fz = ((1.0 + zeta).powf(4.0 / 3.0) + (1.0 - zeta).powf(4.0 / 3.0) - 2.0) / d;
        let dfz = 4.0 / 3.0 * ((1.0 + zeta).powf(1.0 / 3.0) - (1.0 - zeta).powf(1.0 / 3.0)) / d;
        let fpp0 = 8.0 / (9.0 * d);
        let (z3, z4) = (zeta.powi(3), zeta.powi(4));
        let ec = p + al * fz / fpp0 * (1.0 - z4) + (f - p) * fz * z4;
        let dec_drs = dp + dal * fz / fpp0 * (1.0 - z4) + (df - dp) * fz * z4;
        let dec_dz = al / fpp0 * (dfz * (1.0 - z4) - 4.0 * z3 * fz) + (f - p) * (dfz * z4 + 4.0 * z3 * fz);
        let base = ec - rs / 3.0 * dec_drs;
        let (vcu, vcd) = (base - (zeta - 1.0) * dec_dz, base - (zeta + 1.0) * dec_dz);
        (ex + n * ec, vxu + vcu, vxd + vcd)
    }
}

