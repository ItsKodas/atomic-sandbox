//! Live orbitals, Phase A: the Harris display solve's validation (spec
//! `docs/superpowers/specs/2026-11-14-live-orbitals-design.md` §7, V-A1–V-A3; plan task 8). Display
//! only — nothing here reaches forces, energies or the surrogate — but rule 7 needs measured numbers.
//! Numbers of record: `crates/validate/references.md`, "Live orbitals (Harris display; spec
//! 2026-11-14)".
//!
//! `cargo test -p validate --release --test harris -- --nocapture --test-threads=1` (one heavy GPU
//! job; each test is its own ≤ 15 min GPU run: `--test harris va1`, `va2`, `va3`).
//!
//! # What is compared (atomic units: Hartree, Bohr; ħ = mₑ = e = 1)
//! The Harris solve (`qm_gpu::harris` module doc) builds, from an input density ρ_in,σ,
//!
//! ```text
//!   V_σ[ρ_in] = v_ion + V_H[ρ_in] + v_xc,σ[ρ_in↑ + ½ρ_pc, ρ_in↓ + ½ρ_pc]      (∇²V_H = −4πρ_in)
//!   H_σ ψ^H_iσ = ε^H_iσ ψ^H_iσ,   f^H_iσ = qm::one_shot_occupations(ε^H_σ, n_σ, rule)
//!   ρ_out,σ = Σ_i f^H_iσ |ψ^H_iσ|²
//! ```
//!
//! once, with no SCF loop. The SCF's converged state (ψ^S, ε^S, f^S, ρ_SCF) is a fixed point of the
//! same map: ρ_SCF = ρ_out[V[ρ_SCF]] to its tolerance.
//!
//! **The subspace overlap** (`overlap`), per spin channel σ, over the levels with f > 0:
//!
//! ```text
//!   C_ij = h³ Σ_r ψ^H_i(r) ψ^S_j(r)                 (i ∈ occ_H, j ∈ occ_S; f64, Neumaier)
//!   O_σ  = ‖C‖_F² / max(n_occ^H, n_occ^S)
//! ```
//!
//! ‖C‖_F² = Tr(P_H P_S) = Σ_k cos²θ_k (θ_k the principal angles between the two occupied
//! subspaces), so O_σ = 1 iff the subspaces coincide; 1 − O_σ = (1/n)Σ sin²θ_k is **quadratic** in
//! the rotation between them. The brief writes the normaliser as n_σ (the electrons of the
//! channel): that is the same number whenever the f > 0 levels are n_σ whole orbitals (every case
//! here but pseudo-O). Pseudo-O's auto multiplicity is the singlet (Z_v = 6, n↑ = n↓ = 3) and
//! `DegenerateShells` shares the 2p shell, f = 1, ⅔, ⅔, ⅔: four f > 0 levels for n_σ = 3. Dividing
//! by n_σ = 3 would make the identity read 4/3 and let a 25 % mismatch pass; max(n_occ^H, n_occ^S)
//! is the strict normaliser (a count mismatch alone keeps O < 1).
//!
//! **V-A1 (gate): identity on the converged density.** ρ_in = ρ_SCF (the state's ρ↑/ρ↓). The SCF's
//! last step solved in V[ρ_in,last] with ‖ρ_SCF − ρ_in,last‖₁ ≤ tol, so the Harris operator differs
//! from the SCF's by δV = f_Hxc δρ with ‖δρ‖₁ ≤ tol (1e-6, or the floor-aware GPU tol under a
//! setup, ≤ 8e-7 here): |Δε_i| ≈ |⟨ψ_i|δV|ψ_i⟩| ~ 1e-6 Ha, and 1 − O ≈ (1/n)Σ_{i,a}
//! |⟨ψ_a|δV|ψ_i⟩|²/(ε_a − ε_i)² ~ 1e-11. Gates (spec §7): every occupied |Δε| ≤ 1e-5 Ha, O ≥ 1 −
//! 1e-6 per channel. This proves the potential assembly (v_ion with V_loc and the KB projectors,
//! V_H of ρ without ρ_pc, v_xc with ½ρ_pc per spin, the spin assignment) is the SCF's. Cases: H₂,
//! H₂O and a pseudo-O atom.
//!
//! **V-A2 (record + gate): superposed atoms against the SCF.** ρ_in = the renormalised, rescaled
//! superposition of this program's own isolated-atom profiles (`providers::harris::superposition`
//! from `atom_profile`: the app's Atom-job path). With δρ = ρ_in − ρ_SCF, the Harris operator is
//! H[ρ_SCF] + f_Hxc δρ + O(δρ²), so (spec §5.3, corrected 2026-11-14) the orbitals and their ε are
//! **first order** in δρ — δε_i ≈ ⟨ψ_i|f_Hxc δρ|ψ_i⟩ = ⟨ψ_i|V_H[δρ] + f_xc δρ|ψ_i⟩ — and so is
//! ρ_out − ρ_SCF (≈ χ₀ f_Hxc δρ, linear response); only the Harris *total energy* (not computed
//! here) is second order. The overlap deficit 1 − O is quadratic in the first-order rotation.
//! Recorded per case: O per channel, ε^H − ε^S per occupied level, ∫|ρ_in − ρ_SCF|/N and
//! ∫|ρ_out − ρ_SCF|/N (h³ Σ, Neumaier; N = n↑ + n↓). Gate: O ≥ 0.95 per channel for H₂, H₂O, NH₃ and
//! the neutral H₂ + H doublet; H₃⁺ (the charged-rescale case, s = 2/3) recorded only.
//!
//! **V-A3 (record): warm refresh cost** through `GpuKs::harris_display` (the auditor teacher's
//! path) on one teacher: a cold call, then five warm calls, atom 0 moved +0.01 Bohr along x before
//! each. Wall times, each call:
//! - **total**: `OrbitalKind::Harris::wall_s` (problem set-up, superposition, solve, payload);
//! - **display**: the probe's display-solve record (`ActivityProbe::take_display_solves`: the
//!   Activity row's number — the solve and the payload);
//! - **superposition**: `providers::superposition` on the same inputs, timed here (CPU, the same work
//!   `harris_once` does);
//! - **solve**: `HarrisSolve::wall` of a replica `GpuSolver::harris` with the identical input
//!   sequence (same ρ_in, settings, m = 0, cold then warm), interleaved call by call — `GpuKs` does not
//!   expose its inner solve's time;
//! - **payload** ≈ display − solve.
//!
//! Target (spec §7): ≤ 0.3 s warm for H₂ and H₂ + H at 81³; recorded, never asserted.
//!
//! # Geometries and grids (brief, verbatim; cited in `references.md`)
//! H₂ R = 1.4 on x centred at 0; H₂O O–H 1.809 Bohr, ∠HOH 104.5°; NH₃ N–H 1.913 Bohr, ∠HNH 106.7°;
//! H₂ + H: H₂ as above, H at (3.0, 0, 0), multiplicity auto (doublet); H₃⁺ equilateral, side 2.1,
//! charge +1; pseudo-O: one O at the origin (multiplicity auto). H₂, H₂ + H, H₃⁺ at 81³ × 0.15; H₂O,
//! NH₃, pseudo-O at 97³ × 0.15; σ = h; ionic `Accepted` (the app default); tol 1e-6 (raised to the
//! GPU's floor under a setup, as `GpuKs` does: `providers::pseudo::effective_gpu_tol`).
//!
//! # Precision
//! Grid values are the GPU's f32 widened to f64 (`GpuSolver::ks_state`); every overlap and integral
//! here is an f64 Neumaier sum.

use gpu::GpuContext;
use providers::GpuKs;
use qm::{KsState, ScfSettings, System};
use qm_gpu::reduce::neumaier;
use qm_gpu::{GpuSolver, HarrisInput, HarrisSolve, HarrisState};
use sim::{AtomProfile, Cell, ForceProvider, IonicPolicy, Nucleus, Scene};
use std::sync::Arc;
use std::time::Instant;

mod common;
use common::GPU;

fn gpu() -> Option<GpuContext> { common::gpu("harris") }

/// The SCF tolerance on Δρ (brief: tol 1e-6).
const TOL: f64 = 1e-6;
/// The app's default Accurate spacing (and σ = h).
const H: f64 = 0.15;
/// V-A1 gates (spec §7).
const IDENTITY_EPS: f64 = 1e-5;
const IDENTITY_OVERLAP: f64 = 1.0 - 1e-6;
/// V-A2 gate (spec §7): every neutral case.
const SUPERPOSED_OVERLAP: f64 = 0.95;
/// V-A3 target (spec §7), recorded only.
const WARM_TARGET_S: f64 = 0.3;

// ---- geometries (module doc) ----

fn scene(n: usize, nuclei: Vec<Nucleus>, charge: i32) -> Scene {
    Scene { nuclei, cell: Cell { n, h: H }, charge, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: IonicPolicy::Accepted, train_mask: None }
}

fn h2() -> Scene {
    scene(81, vec![Nucleus::new(1, [0.7, 0.0, 0.0]), Nucleus::new(1, [-0.7, 0.0, 0.0])], 0)
}

fn h2_plus_h() -> Scene {
    scene(81, vec![Nucleus::new(1, [0.7, 0.0, 0.0]), Nucleus::new(1, [-0.7, 0.0, 0.0]), Nucleus::new(1, [3.0, 0.0, 0.0])], 0)
}

/// Equilateral, side 2.1, in the xy-plane centred on the origin (`common::h3_plus`'s layout).
fn h3_plus() -> Scene {
    let rad = 2.1 / 3f64.sqrt();
    let nuclei = [90.0f64, 210.0, 330.0].iter().map(|d| { let a = d.to_radians(); Nucleus::new(1, [rad * a.cos(), rad * a.sin(), 0.0]) }).collect();
    scene(81, nuclei, 1)
}

/// O at the origin, the H's in the xy-plane at ±r sin(θ/2) on x, r cos(θ/2) on y.
fn h2o() -> Scene {
    let (r, half) = (1.809, 0.5 * 104.5f64.to_radians());
    scene(97, vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [r * half.sin(), r * half.cos(), 0.0]), Nucleus::new(1, [-r * half.sin(), r * half.cos(), 0.0])], 0)
}

/// N at the origin, the C₃ axis along z: the H's on a circle of radius ρ = d/√3 (d = 2r sin(θ/2)
/// the H–H distance) at z = −√(r² − ρ²), 120° apart.
fn nh3() -> Scene {
    let (r, theta) = (1.913, 106.7f64.to_radians());
    let rho = 2.0 * r * (0.5 * theta).sin() / 3f64.sqrt();
    let z = -(r * r - rho * rho).sqrt();
    let mut nuclei = vec![Nucleus::new(7, [0.0; 3])];
    for k in 0..3 {
        let phi = (120.0 * k as f64).to_radians();
        nuclei.push(Nucleus::new(1, [rho * phi.cos(), rho * phi.sin(), z]));
    }
    scene(97, nuclei, 0)
}

fn pseudo_o() -> Scene {
    scene(97, vec![Nucleus::new(8, [0.0; 3])], 0)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 { ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() }

fn angle(c: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let (u, v) = ([a[0] - c[0], a[1] - c[1], a[2] - c[2]], [b[0] - c[0], b[1] - c[1], b[2] - c[2]]);
    ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (dist(a, c) * dist(b, c))).acos().to_degrees()
}

// ---- the problem as `GpuKs` poses it ----

/// `providers::ks::problem` at `TOL`, the tolerance raised to the GPU floor under a setup (as
/// `GpuKs::evaluate` and `harris_once` do), with the setup's notes.
fn posed(scene: &Scene) -> (qm::Grid, System, ScfSettings, Vec<String>) {
    let mut notes = Vec::new();
    let (grid, system, mut settings) = providers::ks::problem(scene, TOL, &mut notes).unwrap_or_else(|e| panic!("posed: {e}"));
    if system.ionic.is_some() {
        settings.tol = settings.tol.max(providers::pseudo::effective_gpu_tol(TOL, (system.n_up + system.n_dn) as i64));
    }
    (grid, system, settings, notes)
}

/// The electrons each nucleus brings (Z, or Z_v under a pseudopotential): `harris_once`'s `counted`.
fn counted(system: &System) -> Vec<f64> {
    system.nuclei.iter().map(|nu| system.ionic.as_deref().map_or(nu.z as f64, |s| s.valence_charge(nu.z))).collect()
}

/// The isolated neutral atom Z on `scene`'s cell (σ, ionic policy; at the origin, lowest
/// multiplicity) solved by a `GpuKs` with the display on and the partial core off, its display
/// texture read back and binned by `radial_profile` — the app's Atom-job path (the auditor's
/// `isolated_atom` scene, `set_display_partial_core(false)` for atoms; `worker::cloud_jobs`).
fn atom_profile(ctx: &GpuContext, z: u8, scene: &Scene) -> AtomProfile {
    let atom = Scene { nuclei: vec![Nucleus::new(z, [0.0; 3])], cell: scene.cell.clone(), charge: 0, multiplicity: None, sigma: scene.sigma,
                       external_field: [0.0; 3], ionic: scene.ionic, train_mask: None };
    let mut g = GpuKs::new(ctx.clone(), TOL);
    g.set_display_partial_core(false);
    let t = Instant::now();
    let r = g.evaluate(&atom, &mut || false).unwrap_or_else(|e| panic!("the isolated Z = {z} on {}³ × {}: {e}", scene.cell.n, scene.cell.h));
    let Some(sim::Display::Gpu { volume, origin, cell, .. }) = r.display else { panic!("Z = {z}: no GPU display texture") };
    let values = volume.read_back(ctx).unwrap_or_else(|e| panic!("Z = {z}: the density readback failed: {e:?}"));
    let p = sim::radial_profile(z, &values, cell.n, cell.h, origin, scene.sigma, scene.ionic);
    println!("  profile Z = {z} on {}³ × {} ({:?}): {} SCF steps, E {:+.6} Ha, {} shells to {:.2} Bohr, {:.5} electrons; {:.1} s; notes {:?}",
             cell.n, cell.h, scene.ionic, r.stats.scf_steps, r.energy, p.rho.len(), p.reach(), p.electrons, t.elapsed().as_secs_f64(), r.stats.notes);
    p
}

/// One profile per species of `scene` (on its cell), from `cache` or solved.
fn profiles_for(ctx: &GpuContext, scene: &Scene, cache: &mut Vec<AtomProfile>) -> Vec<AtomProfile> {
    let mut species: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
    species.sort_unstable();
    species.dedup();
    species.iter().map(|&z| {
        if let Some(p) = cache.iter().find(|p| p.z == z && p.fits(scene.cell.n, scene.cell.h, scene.sigma, scene.ionic)) {
            return p.clone();
        }
        let p = atom_profile(ctx, z, scene);
        cache.push(p.clone());
        p
    }).collect()
}

// ---- the measures (module doc) ----

/// h³⟨a|b⟩ in f64 (Neumaier).
fn dot(a: &[f64], b: &[f64], dv: f64) -> f64 { neumaier(a.iter().zip(b).map(|(x, y)| x * y)) * dv }

/// The f > 0 levels of one channel: (ε, f, ψ), ascending in ε.
fn occupied<'a>(psi: &'a [Vec<f64>], eps: &[f64], f: &[f64]) -> Vec<(f64, f64, &'a [f64])> {
    assert!(psi.len() == eps.len() && eps.len() == f.len(), "one ε and f per stored orbital ({} / {} / {})", psi.len(), eps.len(), f.len());
    let mut v: Vec<(f64, f64, &[f64])> = psi.iter().zip(eps).zip(f).filter(|(_, fi)| **fi > 0.0).map(|((p, e), fi)| (*e, *fi, p.as_slice())).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v
}

fn channel(s: &KsState, ch: usize) -> Vec<(f64, f64, &[f64])> {
    if ch == 0 { occupied(&s.orbitals_up, &s.evals_up, &s.occ_up) } else { occupied(&s.orbitals_dn, &s.evals_dn, &s.occ_dn) }
}

/// Per channel O_σ = ‖C‖_F² / max(n_occ^H, n_occ^S), C_ij = h³⟨ψ^H_i|ψ^S_j⟩ over the f > 0 levels
/// (module doc; 1 for two empty channels).
fn overlap(h: &KsState, s: &KsState) -> [f64; 2] {
    let dv = s.grid.dv();
    [0, 1].map(|ch| {
        let (a, b) = (channel(h, ch), channel(s, ch));
        let n = a.len().max(b.len());
        if n == 0 { return 1.0; }
        neumaier(a.iter().flat_map(|x| b.iter().map(move |y| dot(x.2, y.2, dv).powi(2)))) / n as f64
    })
}

/// ∫|ρ_a − ρ_b| / N (h³ Σ, Neumaier), ρ = ρ↑ + ρ↓.
fn l1_per_electron(grid: &qm::Grid, a: [&[f64]; 2], b: [&[f64]; 2], n_el: usize) -> f64 {
    let d = (0..grid.len()).map(|p| (a[0][p] + a[1][p] - b[0][p] - b[1][p]).abs());
    neumaier(d) * grid.dv() / n_el as f64
}

/// One channel's occupied levels matched by rank, (ε_H, f_H, ε_SCF, f_SCF) ascending, and the two
/// f > 0 counts (Harris, SCF).
type Levels = (Vec<(f64, f64, f64, f64)>, usize, usize);

/// Per channel: the occupied ε pairs (Harris, SCF) in ascending order, matched by rank, and the
/// two f > 0 counts.
fn levels(h: &KsState, s: &KsState) -> [Levels; 2] {
    [0, 1].map(|ch| {
        let (a, b) = (channel(h, ch), channel(s, ch));
        (a.iter().zip(&b).map(|(x, y)| (x.0, x.1, y.0, y.1)).collect(), a.len(), b.len())
    })
}

/// The SCF of `system` on a fresh solver (cold), its state on the CPU, and the solver (reused for
/// the Harris solve: `harris` never touches the SCF's state).
fn scf(ctx: &GpuContext, name: &str, grid: &qm::Grid, system: &System, settings: &ScfSettings) -> (GpuSolver, KsState) {
    let t = Instant::now();
    let mut s = GpuSolver::new(ctx.clone(), grid);
    let stats = s.solve(system, settings, false).unwrap_or_else(|e| panic!("{name}: the SCF: {e}")).1;
    let ks = s.ks_state(s.state().expect("a converged state"));
    println!("  SCF: {} steps, Δρ {:.1e} (tol {:.1e}), E {:+.8} Ha, n↑ {} n↓ {}, {:.1} s",
             stats.iterations, stats.delta_rho, settings.tol, stats.energy.total, system.n_up, system.n_dn, t.elapsed().as_secs_f64());
    (s, ks)
}

fn harris(s: &mut GpuSolver, name: &str, system: &System, settings: &ScfSettings, rho_in: [&[f64]; 2]) -> (HarrisSolve, KsState) {
    let mut hs = HarrisState::new();
    let input = HarrisInput { system, settings, rho_in, unoccupied: 0 };
    let out = s.harris(&mut hs, &input, &mut || false).unwrap_or_else(|e| panic!("{name}: Harris: {e}")).expect("never aborted");
    println!("  Harris: cold {}, LOBPCG iterations {:?} in {:?} solves, at cap {:?}, n_occ {:?}, {:.3} s, notes {:?}",
             out.cold, out.iterations, out.solves, out.at_cap, out.n_occ, out.wall.as_secs_f64(), out.notes);
    let ks = s.ks_state(hs.state().expect("a completed Harris solve"));
    (out, ks)
}

const SPIN: [&str; 2] = ["↑", "↓"];

// ---- V-A1 ----

/// V-A1 (gate; module doc): Harris on ρ_in = the converged SCF density reproduces the SCF —
/// every occupied |ε^H − ε^S| ≤ 1e-5 Ha and O_σ ≥ 1 − 1e-6 — for H₂ (81³), H₂O and pseudo-O (97³),
/// all under `Accepted`. Measured (2026-11-14): max |Δε| 8.9e-8 / 2.1e-7 / 2.4e-7 Ha, 1 − O ≤ 8.5e-8
/// (pseudo-O with the SCF's f = 1, ⅔, ⅔, ⅔); `references.md` "Live orbitals".
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va1_identity_on_the_converged_density() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let t_all = Instant::now();
    let mut failures = Vec::new();
    for (name, sc) in [("H₂", h2()), ("H₂O", h2o()), ("pseudo-O", pseudo_o())] {
        let t = Instant::now();
        let (grid, system, settings, notes) = posed(&sc);
        println!("V-A1 {name}: {}³ × {H}, σ = h, {:?} (setup {}), tol {:.1e}, occupations {:?}; setup notes {notes:?}",
                 grid.n[0], sc.ionic, if system.ionic.is_some() { "pseudised" } else { "all-electron" }, settings.tol, settings.occupations);
        let (mut s, ks) = scf(&ctx, name, &grid, &system, &settings);
        let (_, hk) = harris(&mut s, name, &system, &settings, [&ks.rho_up, &ks.rho_dn]);
        let o = overlap(&hk, &ks);
        for (ch, (pairs, nh, ns)) in levels(&hk, &ks).into_iter().enumerate() {
            if nh != ns {
                failures.push(format!("{name} {}: {nh} Harris f > 0 levels vs {ns} SCF", SPIN[ch]));
            }
            let mut worst = 0.0f64;
            for (i, (eh, fh, es, fs)) in pairs.iter().enumerate() {
                let d = (eh - es).abs();
                worst = worst.max(d);
                println!("    {} #{i}: ε_H {eh:+.8} (f {fh:.4}) ε_SCF {es:+.8} (f {fs:.4}) |Δ| {d:.2e}", SPIN[ch]);
            }
            println!("  {name} {}: max occupied |Δε| = {worst:.2e} Ha (gate {IDENTITY_EPS:.0e}); O = 1 − {:.2e} (gate 1 − 1e-6)", SPIN[ch], 1.0 - o[ch]);
            if worst > IDENTITY_EPS { failures.push(format!("{name} {}: max |Δε| = {worst:.3e} > {IDENTITY_EPS:.0e} Ha", SPIN[ch])); }
            if o[ch] < IDENTITY_OVERLAP { failures.push(format!("{name} {}: O = {:.9} < 1 − 1e-6", SPIN[ch], o[ch])); }
        }
        println!("  {name}: {:.1} s\n", t.elapsed().as_secs_f64());
    }
    println!("V-A1: {:.1} s", t_all.elapsed().as_secs_f64());
    assert!(failures.is_empty(), "V-A1 failed: {failures:#?}");
}

// ---- V-A2 ----

/// V-A2 (record + gate; module doc): superposed isolated atoms (`atom_profile` →
/// `providers::superposition`, renormalised and rescaled as the app does) against the SCF for H₂,
/// H₂O, NH₃, H₂ + H (doublet) and H₃⁺ (charge +1). Prints O per channel, ε^H − ε^S per occupied
/// level, ∫|ρ_in − ρ_SCF|/N and ∫|ρ_out − ρ_SCF|/N; asserts O ≥ 0.95 per channel for every case
/// but H₃⁺ (recorded only). Measured (2026-11-14): O = 0.9969 (H₂), 0.9919 (H₂O), 0.9879 (NH₃),
/// 0.9962 / 0.9987 (H₂ + H ↑ / ↓), 0.9981 (H₃⁺); ε_H − ε_SCF −0.03 to −0.19 Ha on every level (first
/// order in ρ_in − ρ_SCF; the documented gap, `references.md` "Live orbitals").
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va2_superposed_atoms_against_scf() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let t_all = Instant::now();
    let mut cache = Vec::new();
    let mut failures = Vec::new();
    let mut table = Vec::new();
    for (name, sc, gated) in [("H₂", h2(), true), ("H₂O", h2o(), true), ("NH₃", nh3(), true), ("H₂ + H", h2_plus_h(), true), ("H₃⁺", h3_plus(), false)] {
        let t = Instant::now();
        let (grid, system, settings, notes) = posed(&sc);
        let geo: Vec<String> = match name {
            "H₂O" | "NH₃" => vec![format!("r {:.4}", dist(sc.nuclei[0].pos, sc.nuclei[1].pos)), format!("∠ {:.2}°", angle(sc.nuclei[0].pos, sc.nuclei[1].pos, sc.nuclei[2].pos))],
            _ => (0..sc.nuclei.len()).flat_map(|a| ((a + 1)..sc.nuclei.len()).map(move |b| (a, b))).map(|(a, b)| format!("r{a}{b} {:.4}", dist(sc.nuclei[a].pos, sc.nuclei[b].pos))).collect(),
        };
        println!("V-A2 {name}: {}³ × {H}, σ = h, {:?}, charge {:+}, n↑ {} n↓ {}, tol {:.1e}; geometry {geo:?}; setup notes {notes:?}",
                 grid.n[0], sc.ionic, sc.charge, system.n_up, system.n_dn, settings.tol);
        let profiles = profiles_for(&ctx, &sc, &mut cache);
        let c = counted(&system);
        let rho = providers::superposition(&grid, &system.nuclei, &profiles, &c, system.n_up, system.n_dn).unwrap_or_else(|e| panic!("{name}: superposition: {e}"));
        println!("  ρ_in: counted {c:?}, on-grid S_A {:.5?}, rescale {:.6}, notes {:?}", rho.per_atom, rho.rescale, rho.notes);
        let (mut s, ks) = scf(&ctx, name, &grid, &system, &settings);
        let (_, hk) = harris(&mut s, name, &system, &settings, [&rho.up, &rho.dn]);
        let n_el = system.n_up + system.n_dn;
        let l1_in = l1_per_electron(&grid, [&rho.up, &rho.dn], [&ks.rho_up, &ks.rho_dn], n_el);
        let l1_out = l1_per_electron(&grid, [&hk.rho_up, &hk.rho_dn], [&ks.rho_up, &ks.rho_dn], n_el);
        let o = overlap(&hk, &ks);
        for (ch, (pairs, nh, ns)) in levels(&hk, &ks).into_iter().enumerate() {
            for (i, (eh, fh, es, fs)) in pairs.iter().enumerate() {
                println!("    {} #{i}: ε_H {eh:+.6} (f {fh:.4}) ε_SCF {es:+.6} (f {fs:.4}) ε_H − ε_SCF {:+.2e}", SPIN[ch], eh - es);
            }
            if nh != ns { println!("    {}: {nh} Harris f > 0 levels vs {ns} SCF", SPIN[ch]); }
            let verdict = if gated { if o[ch] >= SUPERPOSED_OVERLAP { "≥ 0.95" } else { "FAILS 0.95" } } else { "recorded only" };
            println!("  {name} {}: O = {:.6} (1 − O = {:.2e}; {verdict})", SPIN[ch], o[ch], 1.0 - o[ch]);
            if gated && o[ch] < SUPERPOSED_OVERLAP { failures.push(format!("{name} {}: O = {:.6} < {SUPERPOSED_OVERLAP}", SPIN[ch], o[ch])); }
        }
        println!("  {name}: ∫|ρ_in − ρ_SCF|/N = {l1_in:.4e}, ∫|ρ_out − ρ_SCF|/N = {l1_out:.4e}; {:.1} s\n", t.elapsed().as_secs_f64());
        table.push(format!("{name}: O {:.6} / {:.6}, L1_in/N {l1_in:.3e}, L1_out/N {l1_out:.3e}, rescale {:.4}", o[0], o[1], rho.rescale));
    }
    println!("V-A2 summary ({:.1} s):", t_all.elapsed().as_secs_f64());
    for r in &table { println!("  {r}"); }
    assert!(failures.is_empty(), "V-A2 failed (investigate, never loosen): {failures:#?}");
}

// ---- V-A3 ----

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

fn min(v: &[f64]) -> f64 { v.iter().copied().fold(f64::INFINITY, f64::min) }

/// V-A3 (record; module doc): one teacher's `harris_display` (All orbitals coloured, m = 0), a cold
/// call then five warm ones each after atom 0 moves +0.01 Bohr; min/median wall of the total, the
/// display record, the superposition, the solve (replica) and the payload, for H₂ and H₂ + H at 81³
/// and H₂O at 97³, against the ≤ 0.3 s target (no assert). Measured (2026-11-14), warm median total:
/// H₂ 0.197 s, H₂ + H 0.263 s (target met), H₂O 1.31 s; cold 0.8 / 1.0 / 2.7 s.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va3_warm_refresh_cost() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut cache = Vec::new();
    let mut summary = Vec::new();
    for (name, sc) in [("H₂", h2()), ("H₂ + H", h2_plus_h()), ("H₂O", h2o())] {
        println!("V-A3 {name}: {}³ × {H}, {:?}", sc.cell.n, sc.ionic);
        let atoms = Arc::new(sim::HarrisAtoms { profiles: profiles_for(&ctx, &sc, &mut cache) });
        let probe = sim::ActivityProbe::new();
        let mut g = GpuKs::new(ctx.clone(), TOL).with_probe(probe.clone(), sim::Slot::Side);
        g.set_orbital_view(sim::OrbitalViewRequest::new(sim::OrbitalRequest::AllColoured { include: None, physical: false, look: sim::ColourLook::default() }));
        let (grid0, _, _, _) = posed(&sc);
        let mut replica = GpuSolver::new(ctx.clone(), &grid0);
        let mut hs = HarrisState::new();
        let mut moved = sc.clone();
        // per call: total, display, superposition, solve, payload (s)
        let mut rows: Vec<[f64; 5]> = Vec::new();
        for call in 0..6 {
            if call > 0 { moved.nuclei[0].pos[0] += 0.01; }
            let (grid, system, settings, _) = posed(&moved);
            let c = counted(&system);
            let t = Instant::now();
            let rho = providers::superposition(&grid, &system.nuclei, &atoms.profiles, &c, system.n_up, system.n_dn).expect("superposition");
            let sup = t.elapsed().as_secs_f64();
            probe.take_display_solves();
            let payload = match g.harris_display(&moved, &atoms, &mut || false) {
                Some(sim::HarrisAnswer::Solved(Ok(p))) => p,
                other => panic!("{name} call {call}: {other:?}"),
            };
            let sim::OrbitalKind::Harris { wall_s, rescale, .. } = payload.kind else { panic!("{name}: not a Harris payload") };
            let display = probe.take_display_solves().last().map(|d| d.ms * 1e-3).expect("the probe's display record");
            let input = HarrisInput { system: &system, settings: &settings, rho_in: [&rho.up, &rho.dn], unoccupied: 0 };
            let rs = replica.harris(&mut hs, &input, &mut || false).expect("replica").expect("never aborted");
            let solve = rs.wall.as_secs_f64();
            println!("  call {call} ({}; x₀ = {:+.2}): total {wall_s:.3} s, display {display:.3} s, superposition {sup:.3} s, solve {solve:.3} s (replica: iterations {:?}, solves {:?}, at cap {:?}), payload ≈ {:.3} s; {} rows, rescale {rescale}",
                     if rs.cold { "cold" } else { "warm" }, moved.nuclei[0].pos[0], rs.iterations, rs.solves, rs.at_cap, display - solve, payload.rows.len());
            rows.push([wall_s, display, sup, solve, display - solve]);
        }
        let warm: Vec<[f64; 5]> = rows[1..].to_vec();
        let col = |k: usize| -> Vec<f64> { warm.iter().map(|r| r[k]).collect() };
        let names = ["total", "display", "superposition", "solve", "payload"];
        let stats: Vec<String> = (0..5).map(|k| format!("{} min {:.3} / median {:.3} s", names[k], min(&col(k)), median(&col(k)))).collect();
        let met = median(&col(0)) <= WARM_TARGET_S;
        println!("  {name} warm (5 calls): {}; cold total {:.3} s; target ≤ {WARM_TARGET_S} s {}\n", stats.join(", "), rows[0][0], if met { "met" } else { "NOT met" });
        summary.push(format!("{name} {}³: cold {:.3} s; warm {}; target {}", sc.cell.n, rows[0][0], stats.join(", "), if met { "met" } else { "not met" }));
    }
    println!("V-A3 summary:");
    for r in &summary { println!("  {r}"); }
}
