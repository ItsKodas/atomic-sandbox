//! Live orbitals, Phase A: the Harris display solve's validation (spec
//! `docs/superpowers/specs/2026-11-14-live-orbitals-design.md` §7, V-A1–V-A4; plan tasks 8, 9). Display
//! only — nothing here reaches forces, energies or the surrogate — but rule 7 needs measured numbers.
//! Numbers of record: `crates/validate/references.md`, "Live orbitals (Harris display; spec
//! 2026-11-14)".
//!
//! `cargo test -p validate --release --test harris -- --nocapture --test-threads=1` (one heavy GPU
//! job; each test is its own ≤ 15 min GPU run: `--test harris va1`, `va2`, `va3`, and V-A4's three
//! frame sets `va4_harris_bonds_against_scf_and_experience_h2o`, `…_h2_scan`, `…_h2_plus_h`).
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
//! **The subspace overlap** (`overlap`), per spin channel σ, over the occupied levels — f > 0 under `Aufbau` and
//! `DegenerateShells`, f ≥ `qm::PARTIAL_OCCUPATION_MIN` (1e-6) under `FermiDirac` (the scf-stall ← main merge: every
//! scene's occupations; plan 2026-11-13 Task 9's display rule — the SCF state's witness columns carry f ≤ e^−36 > 0 and
//! are never occupied; the threshold is the same set as f > 0 under the integer and shared rules, whose f are 0 or a share ≥ 1/k of the k computed levels):
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
//! 1e-6 per channel, and every f^H = f^S to 1e-12 (under `FermiDirac`, a partly filled level to the ε gate propagated through ∂f/∂ε, f(1 − f)/kT · 1e-5: `fd_occ_gate`). This proves the potential assembly (v_ion with
//! V_loc and the KB projectors, V_H of ρ without ρ_pc, v_xc with ½ρ_pc per spin) is the SCF's, and —
//! through the H₂ + H doublet, whose ↑ and ↓ potentials differ by the exchange splitting — the spin
//! assignment (v_xc,↑ on ↑). Cases: H₂, H₂ + H (doublet), H₂O and a pseudo-O atom.
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
//! each, then one warm call after a 0.1 Bohr move (a playing run's step between two refreshes, or a
//! drag; recorded on its own row). Wall times, each call:
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
//! **V-A4 (record only; measurement for spec §9 Q2, not a gate; plan task 9): bonds from the Harris
//! density.** Would bonds and partial charges read from the Harris ρ_out beat the viewer's bond
//! experience (`sim::bond_experience`) between DFT solves? Three frame sets (`va4_…_h2o`,
//! `va4_…_h2_scan`, `va4_…_h2_plus_h`; brief, verbatim): (a) a sampling pass on H₂O (97³; 60
//! perturbations σ 0.15, no burst, no scans; `common::surrogate::student` + `sample_and_fit_with_floor`,
//! the sampler's time limit off — in one process a solve may wait minutes for the suite's GPU lock), its
//! samples in store order, the first 40 train experience and the last 20 are held out; (b) H₂ at R =
//! 0.8, 1.0, …, 6.0 (81³); (c) H₂ + H, the third H at d = 6.0, 5.75, …, 2.0 Bohr from H₂'s centre on
//! its axis (81³, the chain centred in the box: at d = 6 a third H measured from an H₂ at the origin
//! would sit on the box face). In (b) and (c) even-index frames train, odd-index frames are held out.
//! Every frame: the SCF (warm from the set's previous frame) and its graph through the driver's
//! detector — `sim::Detector` with its constants unchanged, against the driver's bond references
//! (`sim::solve_atom` on each cell, a display-off `GpuKs` at tol 1e-6: the driver's side solver),
//! the provider's `sample_density` replaced by `sim::bonds::trilinear_blend` of the density read back
//! (the GPU sampler's operator, in f64 on the f32 values). Training frames: `examples` of the SCF
//! graph into one fresh `Store` per set (as the viewer records a fresh graph). Held-out frames: the
//! Harris ρ_out from V-A2's superposed atoms (warm `HarrisState` per set, m = 0) through the same
//! detector and references; Hirshfeld charges (`sim::charges::hirshfeld`, the driver's free atoms and
//! valence counts) of both; `Store::predict(pair_key(..), R)` per pair within `R_MAX` once every
//! training example is in. **Scored per pair against the SCF:** bonded / not bonded agreement; kind
//! agreement (`Bond::kind`, over the pairs both densities call bonded); |Δρ_b| and |Δ pile-up| — ρ̂ and
//! q = ρ/ρ_pro at each detector's own bond point (`PairReport::rho_min`, `enhancement`), for every pair
//! within `R_MAX`, bonded or not; |Δq| — the mean of the pair's two atoms' Hirshfeld errors (and per
//! atom over the set); experience coverage (answers at all: ≥ `MIN_AGREE` agreeing examples within
//! `MATCH_R`) and its agreement with the SCF where it answers. **Split** (a display split, brief):
//! "near the bond length" = SCF bonded with ρ_b ≥ 0.1 e/Bohr³; "stretched/borderline" = every other
//! pair within `R_MAX` (non-bonded pairs included). Asserted: only that every frame solved.
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
/// V-A1: the one-shot occupations equal the SCF's (the same rule on ε equal to ~1e-7 Ha; exact fractions) — for
/// every level under `Aufbau`/`DegenerateShells`, and under `FermiDirac` for a level within `PARTIAL_OCCUPATION_MIN`
/// of 0 or 1. A partly filled FD level is gated by `fd_occ_gate` instead.
const OCC_EPS: f64 = 1e-12;

/// V-A1's f gate for a partly filled `FermiDirac` level (1e-6 < f < 1 − 1e-6; controller ruling (1) on the scf-stall
/// ← main merge, 2026-11-16). The 1e-12 gate presumed `DegenerateShells`' exact fractions (f = (N − n_below)/|C|, a
/// constant of the level count). Under FD, f = 1/(1 + e^{(ε − μ)/kT}) is a smooth function of ε with
///   ∂f/∂ε = −f(1 − f)/kT,
/// so an ε that passes the ε gate (|Δε| ≤ `IDENTITY_EPS`) moves f by at most f(1 − f)/kT · `IDENTITY_EPS` to first
/// order. The f gate is that propagated ε gate, |Δf| ≤ f(1 − f)/kT · `IDENTITY_EPS` (4.4e-4 at f = ⅔, kT = 5 mHa) —
/// derived from an existing gate, not loosened to pass; `references.md` "Harris under Fermi–Dirac".
fn fd_occ_gate(f: f64, kt: f64) -> f64 { f * (1.0 - f) / kt * IDENTITY_EPS }

/// The f gate of one occupied level of SCF occupation `fs` under `occupations` (`OCC_EPS`, or `fd_occ_gate` for a
/// partly filled FD level).
fn occ_gate(fs: f64, occupations: qm::Occupations) -> f64 {
    match occupations {
        qm::Occupations::FermiDirac { kt } if fs.min(1.0 - fs) > qm::PARTIAL_OCCUPATION_MIN => fd_occ_gate(fs, kt),
        _ => OCC_EPS,
    }
}
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

/// The occupied levels of one channel (module doc: f ≥ `PARTIAL_OCCUPATION_MIN`, which is f > 0 under the integer
/// and shared rules and excludes the FD witness columns): (ε, f, ψ), ascending in ε.
fn occupied<'a>(psi: &'a [Vec<f64>], eps: &[f64], f: &[f64]) -> Vec<(f64, f64, &'a [f64])> {
    assert!(psi.len() == eps.len() && eps.len() == f.len(), "one ε and f per stored orbital ({} / {} / {})", psi.len(), eps.len(), f.len());
    let mut v: Vec<(f64, f64, &[f64])> = psi.iter().zip(eps).zip(f).filter(|(_, fi)| **fi >= qm::PARTIAL_OCCUPATION_MIN).map(|((p, e), fi)| (*e, *fi, p.as_slice())).collect();
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
/// every occupied |ε^H − ε^S| ≤ 1e-5 Ha, |f^H − f^S| ≤ 1e-12 (a partly filled FD level: `fd_occ_gate`) and O_σ ≥ 1 − 1e-6 — for H₂ and the
/// H₂ + H doublet (81³: the spin assignment), H₂O and pseudo-O (97³), all under `Accepted`. Measured (2026-11-15): max |Δε| 9.1e-8 / 4.9e-8 / 1.3e-7 / 2.2e-7 Ha, 1 − O ≤ 1.2e-7,
/// every f equal (pseudo-O 1, ⅔, ⅔, ⅔); `references.md` "Live orbitals".
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va1_identity_on_the_converged_density() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let t_all = Instant::now();
    let mut failures = Vec::new();
    for (name, sc) in [("H₂", h2()), ("H₂ + H", h2_plus_h()), ("H₂O", h2o()), ("pseudo-O", pseudo_o())] {
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
                let gate = occ_gate(*fs, settings.occupations);
                if (fh - fs).abs() > gate { failures.push(format!("{name} {} #{i}: f_H {fh} vs f_SCF {fs} (|Δf| {:.2e} > gate {gate:.2e})", SPIN[ch], (fh - fs).abs())); }
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
/// and H₂O at 97³, against the ≤ 0.3 s target (no assert); then one warm call after a 0.1 Bohr move
/// (its own row). Measured (2026-11-15, after task 8b), warm median total: H₂ 0.115 s, H₂ + H 0.147 s
/// (target met), H₂O 0.460 s; the 0.1 Bohr step 0.138 / 0.186 / 0.585 s; cold 0.68 / 0.67 / 1.52 s
/// (2026-11-14: warm 0.197 / 0.263 / 1.31 s).
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
        for call in 0..7 {
            // calls 1–5: +0.01 Bohr each; call 6: +0.1 Bohr (its own row)
            if call > 0 { moved.nuclei[0].pos[0] += if call == 6 { 0.1 } else { 0.01 }; }
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
        let warm: Vec<[f64; 5]> = rows[1..6].to_vec();
        let step = rows[6];
        let col = |k: usize| -> Vec<f64> { warm.iter().map(|r| r[k]).collect() };
        let names = ["total", "display", "superposition", "solve", "payload"];
        let stats: Vec<String> = (0..5).map(|k| format!("{} min {:.3} / median {:.3} s", names[k], min(&col(k)), median(&col(k)))).collect();
        let met = median(&col(0)) <= WARM_TARGET_S;
        println!("  {name} warm (5 calls): {}; cold total {:.3} s; 0.1 Bohr step: total {:.3} s, solve {:.3} s; target ≤ {WARM_TARGET_S} s {}\n",
                 stats.join(", "), rows[0][0], step[0], step[3], if met { "met" } else { "NOT met" });
        summary.push(format!("{name} {}³: cold {:.3} s; warm {}; 0.1 Bohr step total {:.3} s (solve {:.3} s); target {}",
                             sc.cell.n, rows[0][0], stats.join(", "), step[0], step[3], if met { "met" } else { "not met" }));
    }
    println!("V-A3 summary:");
    for r in &summary { println!("  {r}"); }
}

// ---- V-A4 ----

/// One frame of a V-A4 set: its label, scene, and whether it trains experience (else held out).
struct Frame {
    label: String,
    scene: Scene,
    train: bool,
}

/// One held-out pair within `R_MAX`, scored (module doc, "V-A4").
struct PairRow {
    /// SCF bonded with ρ_b ≥ [`NEAR_RHO_B`] (else "stretched/borderline").
    near: bool,
    scf_bonded: bool,
    harris_bonded: bool,
    /// The two kinds (SCF, Harris), when both densities call the pair bonded.
    kinds: Option<(sim::BondKind, sim::BondKind)>,
    /// |ρ̂_b^H − ρ̂_b^S| and |q_b^H − q_b^S| at each detector's own bond point (`PairReport::rho_min`,
    /// `enhancement`).
    d_rho_b: f64,
    d_pile: f64,
    /// ½(|Δq_i| + |Δq_j|), the pair's two atoms' Hirshfeld charge errors.
    dq: f64,
    exp: sim::bond_experience::Verdict,
}

/// The V-A4 display split (brief): "near the bond length" is SCF bonded with ρ_b ≥ 0.1 e/Bohr³.
const NEAR_RHO_B: f64 = 0.1;

/// The driver's bond reference of species `z` on `cell` (`sim::references::solve_atom`: every
/// allowed multiplicity solved cold, the lowest energy's shell-binned density kept) through a
/// `GpuKs` at `TOL` with the display off (the driver's `side_solver`), σ = h, `Accepted`; cached
/// per (Z, n).
fn bond_reference(ctx: &GpuContext, cell: &Cell, z: u8, cache: &mut Vec<(u8, usize, sim::RadialDensity)>) -> sim::RadialDensity {
    if let Some((_, _, t)) = cache.iter().find(|(zz, n, _)| *zz == z && *n == cell.n) {
        return t.clone();
    }
    let t = Instant::now();
    let mut g = GpuKs::new(ctx.clone(), TOL);
    g.set_display(false);
    let mass = Nucleus::new(z, [0.0; 3]).mass;
    let r = sim::solve_atom(&mut g, cell, None, IonicPolicy::Accepted, z, mass, &mut || false).unwrap_or_else(|e| panic!("the Z = {z} bond reference on {}³: {e}", cell.n));
    println!("  bond reference Z = {z} on {}³ × {}: multiplicity {:?}, E {:+.6} Ha, tol {:.1e}, {} entries, {:.4} electrons; {:.1} s",
             cell.n, cell.h, r.multiplicity, r.energy, r.tol, r.radial.r.len(), r.radial.electrons(), t.elapsed().as_secs_f64());
    cache.push((z, cell.n, r.radial.clone()));
    r.radial
}

/// The detector as the driver runs it (`Dynamics::refresh_bonds`: rounds of points until `feed`
/// returns `None`, then the graph and the reports), the provider's `sample_density` replaced by
/// `sim::bonds::trilinear_blend` of `rho` (ρ↑ + ρ↓ at `cell`'s n³ nodes, `qm::Grid` order) — the GPU
/// sampler's operator, in f64 on the f32 values read back.
fn detect(cell: &Cell, pos: &[[f64; 3]], pro: &sim::Promolecule<'_>, rho: &[f64]) -> (sim::BondGraph, Vec<sim::PairReport>) {
    let n = cell.n;
    assert_eq!(rho.len(), n * n * n);
    let sample = |p: [f64; 3]| sim::bonds::trilinear_blend(cell, p, |[i, j, k]| rho[i + n * (j + n * k)]);
    let mut det = sim::Detector::new(pos);
    let mut pts = det.points().to_vec();
    while !pts.is_empty() {
        let v: Vec<f64> = pts.iter().map(|&p| sample(p)).collect();
        match det.feed(&v, pro, cell.h) {
            Some(next) => pts = next.to_vec(),
            None => break,
        }
    }
    (det.graph(pos.len(), 0, 0.0), det.reports().to_vec())
}

/// The SCF of one frame on the set's solver, warm from the previous frame (`GpuSolver::solve`
/// takes another geometry's state as the start), cold if the warm solve fails.
fn scf_warm(s: &mut GpuSolver, label: &str, system: &System, settings: &ScfSettings) -> (KsState, String) {
    let t = Instant::now();
    // `warm` with no state held (the set's first frame) runs the cold path
    let first = s.state().is_none();
    let (stats, how) = match s.solve(system, settings, true) {
        Ok((_, st)) => (st, if first { "cold" } else { "warm" }),
        Err(e) => {
            println!("    {label}: the warm SCF failed ({e}); cold");
            (s.solve(system, settings, false).unwrap_or_else(|e| panic!("{label}: the SCF: {e}")).1, "cold")
        }
    };
    let ks = s.ks_state(s.state().expect("a converged state"));
    (ks, format!("SCF {how} {} steps Δρ {:.1e} E {:+.6} {:.1} s", stats.iterations, stats.delta_rho, stats.energy.total, t.elapsed().as_secs_f64()))
}

fn total(up: &[f64], dn: &[f64]) -> Vec<f64> { up.iter().zip(dn).map(|(a, b)| a + b).collect() }

fn verdict_word(r: &sim::PairReport) -> String {
    match r.verdict {
        sim::Verdict::Bonded { .. } => format!("bonded ({:?})", r.kind().unwrap_or(sim::BondKind::Indeterminate)),
        v => format!("{v:?}"),
    }
}

/// The mean (f64, Neumaier) and the count; NaN for none.
fn mean(v: impl Iterator<Item = f64>) -> (f64, usize) {
    let v: Vec<f64> = v.collect();
    if v.is_empty() { (f64::NAN, 0) } else { (neumaier(v.iter().copied()) / v.len() as f64, v.len()) }
}

fn pct(k: usize, n: usize) -> String { if n == 0 { "—".into() } else { format!("{:.0} %", 100.0 * k as f64 / n as f64) } }

/// One table row (brief): pairs, Harris agreement, kind agreement (over the pairs both call
/// bonded), mean |Δρ_b|, mean |Δ pile-up|, mean |Δq| (the pairs' atoms), experience coverage
/// (answers at all) and agreement where it answers.
fn table_row(split: &str, rows: &[&PairRow]) -> String {
    use sim::bond_experience::Verdict as E;
    let n = rows.len();
    let agree = rows.iter().filter(|r| r.scf_bonded == r.harris_bonded).count();
    let both: Vec<_> = rows.iter().filter_map(|r| r.kinds).collect();
    let kind_agree = both.iter().filter(|(a, b)| a == b).count();
    let (d_rho, _) = mean(rows.iter().map(|r| r.d_rho_b));
    let (d_pile, _) = mean(rows.iter().map(|r| r.d_pile));
    let (dq, _) = mean(rows.iter().map(|r| r.dq));
    let answered: Vec<_> = rows.iter().filter(|r| r.exp != E::Unknown).collect();
    let exp_agree = answered.iter().filter(|r| (r.exp == E::Bonded) == r.scf_bonded).count();
    format!("| {split} | {n} | {} ({agree}/{n}) | {} ({kind_agree}/{}) | {d_rho:.2e} | {d_pile:.3} | {dq:.2e} | {} ({}/{n}) | {} ({exp_agree}/{}) |",
            pct(agree, n), pct(kind_agree, both.len()), both.len(), pct(answered.len(), n), answered.len(), pct(exp_agree, answered.len()), answered.len())
}

/// A held-out frame's pairs before experience is asked: (i, j, R, row).
struct Held {
    label: String,
    z: Vec<u8>,
    pos: Vec<[f64; 3]>,
    pairs: Vec<(usize, usize, f64, PairRow)>,
    dq_atoms: Vec<f64>,
}

/// V-A4 on one frame set (module doc): every frame's SCF and its bond graph; the training
/// frames' graphs into a fresh experience store (`sim::bond_experience::examples`, as the viewer
/// records a fresh graph); per held-out frame the Harris ρ_out from superposed atoms, both
/// densities through the detector and Hirshfeld, and experience's verdict per pair. Prints one
/// line per held-out pair and the set's table (returned); panics if a frame does not solve.
fn va4_set(ctx: &GpuContext, set: &str, frames: &[Frame]) -> Vec<String> {
    use sim::bond_experience::{examples, neighbour_counts, pair_key, Store};
    let t_set = Instant::now();
    let cell = frames[0].scene.cell.clone();
    let (grid0, _, _, _) = posed(&frames[0].scene);
    let mut s = GpuSolver::new(ctx.clone(), &grid0);
    let mut hs = HarrisState::new();
    let mut refs = Vec::new();
    let mut profiles = Vec::new();
    let mut store = Store::default();
    let mut held: Vec<Held> = Vec::new();
    let (mut harris_at_cap, mut harris_solves) = (0usize, 0usize);
    for f in frames {
        let sc = &f.scene;
        assert_eq!(sc.cell, cell, "one cell per set");
        let (grid, system, settings, _) = posed(sc);
        let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
        let tables: Vec<sim::RadialDensity> = z.iter().map(|&zz| bond_reference(ctx, &cell, zz, &mut refs)).collect();
        let pro = sim::Promolecule::new(tables.iter().collect(), &pos, Some(&cell));
        let (ks, scf_line) = scf_warm(&mut s, &f.label, &system, &settings);
        let rho_s = total(&ks.rho_up, &ks.rho_dn);
        let (g_s, rep_s) = detect(&cell, &pos, &pro, &rho_s);
        if f.train {
            let ex = examples(&z, &pos, &g_s);
            println!("  {set} {} (train): {scf_line}; {} bonds, {} examples", f.label, g_s.bonds.len(), ex.len());
            ex.into_iter().for_each(|e| store.add(e));
            continue;
        }
        // held out: the Harris ρ_out from this program's superposed atoms (V-A2's ρ_in), warm
        // from the set's previous held-out frame (as the app's refreshes are)
        let prof = profiles_for(ctx, sc, &mut profiles);
        let c = counted(&system);
        let rho_in = providers::superposition(&grid, &system.nuclei, &prof, &c, system.n_up, system.n_dn).unwrap_or_else(|e| panic!("{}: superposition: {e}", f.label));
        let input = HarrisInput { system: &system, settings: &settings, rho_in: [&rho_in.up, &rho_in.dn], unoccupied: 0 };
        let out = s.harris(&mut hs, &input, &mut || false).unwrap_or_else(|e| panic!("{}: Harris: {e}", f.label)).expect("never aborted");
        harris_solves += 1;
        harris_at_cap += out.at_cap.iter().any(|&c| c) as usize;
        let hk = s.ks_state(hs.state().expect("a completed Harris solve"));
        let rho_h = total(&hk.rho_up, &hk.rho_dn);
        let (_, rep_h) = detect(&cell, &pos, &pro, &rho_h);
        // Hirshfeld charges of both against the same free atoms (the driver's `refresh_bonds`)
        let valence = providers::pseudo::valence_charges(sc).expect("valence charges");
        let (q_s, n_s) = sim::charges::hirshfeld(pro.tables(), &pos, &valence, &cell, &rho_s);
        let (q_h, n_h) = sim::charges::hirshfeld(pro.tables(), &pos, &valence, &cell, &rho_h);
        let dq_atoms: Vec<f64> = q_s.iter().zip(&q_h).map(|(a, b)| (a - b).abs()).collect();
        println!("  {set} {} (held out): {scf_line}; Harris {} iterations {:?} at cap {:?} {:.2} s, rescale {:.4}; electrons SCF {n_s:.4} Harris {n_h:.4}; q_SCF {:+.4?} q_Harris {:+.4?}",
                 f.label, if out.cold { "cold" } else { "warm" }, out.iterations, out.at_cap, out.wall.as_secs_f64(), rho_in.rescale, q_s, q_h);
        let mut pairs = Vec::new();
        for (a, b) in rep_s.iter().zip(&rep_h) {
            assert_eq!((a.i, a.j), (b.i, b.j), "the same pair order");
            if a.verdict == sim::Verdict::TooFar {
                continue;
            }
            let r = dist(pos[a.i], pos[a.j]);
            let scf_bonded = matches!(a.verdict, sim::Verdict::Bonded { .. });
            let harris_bonded = matches!(b.verdict, sim::Verdict::Bonded { .. });
            let (rb_s, rb_h) = (a.rho_min.expect("a chord point"), b.rho_min.expect("a chord point"));
            let (qb_s, qb_h) = (a.enhancement.expect("a chord point"), b.enhancement.expect("a chord point"));
            let kinds = (scf_bonded && harris_bonded).then(|| (a.kind().unwrap_or(sim::BondKind::Indeterminate), b.kind().unwrap_or(sim::BondKind::Indeterminate)));
            println!("    {}–{} R {r:.3}: SCF {} ρ_b {rb_s:.4} q {qb_s:.3} s {:.3} | Harris {} ρ_b {rb_h:.4} q {qb_h:.3} s {:.3}",
                     a.i, a.j, verdict_word(a), a.s.unwrap_or(f64::NAN), verdict_word(b), b.s.unwrap_or(f64::NAN));
            let row = PairRow { near: scf_bonded && rb_s >= NEAR_RHO_B, scf_bonded, harris_bonded, kinds, d_rho_b: (rb_h - rb_s).abs(), d_pile: (qb_h - qb_s).abs(),
                                dq: 0.5 * (dq_atoms[a.i] + dq_atoms[a.j]), exp: sim::bond_experience::Verdict::Unknown };
            pairs.push((a.i, a.j, r, row));
        }
        held.push(Held { label: f.label.clone(), z, pos, pairs, dq_atoms });
    }
    // experience, trained on every training frame, asked about the held-out pairs
    store.prepare();
    println!("  {set}: experience holds {} examples from {} training frames", store.len(), frames.iter().filter(|f| f.train).count());
    let mut rows: Vec<PairRow> = Vec::new();
    let mut dq_all = Vec::new();
    for h in held {
        let counts = neighbour_counts(&h.pos);
        let mut said = Vec::new();
        for (i, j, r, mut row) in h.pairs {
            row.exp = store.predict(pair_key(h.z[i], counts[i], h.z[j], counts[j]), r);
            said.push(format!("{i}–{j} {:?}", row.exp));
            rows.push(row);
        }
        println!("    {} experience: {}", h.label, said.join(", "));
        dq_all.extend(h.dq_atoms);
    }
    let near: Vec<&PairRow> = rows.iter().filter(|r| r.near).collect();
    let rest: Vec<&PairRow> = rows.iter().filter(|r| !r.near).collect();
    let all: Vec<&PairRow> = rows.iter().collect();
    let (dq_mean, n_atoms) = mean(dq_all.iter().copied());
    let dq_max = dq_all.iter().copied().fold(0.0f64, f64::max);
    let mut table = vec![
        format!("{set}: {} held-out frames, {} training; Harris solves at the LOBPCG cap: {harris_at_cap} of {harris_solves}; per-atom |Δq| mean {dq_mean:.2e}, max {dq_max:.2e} over {n_atoms} atoms; {:.0} s",
                frames.iter().filter(|f| !f.train).count(), frames.iter().filter(|f| f.train).count(), t_set.elapsed().as_secs_f64()),
        "| split | pairs | Harris agreement | kind agreement | mean \\|Δρ_b\\| | mean \\|Δ pile-up\\| | mean \\|Δq\\| | experience coverage | experience agreement |".into(),
        "|---|---|---|---|---|---|---|---|---|".into(),
    ];
    table.push(table_row("near the bond length", &near));
    table.push(table_row("stretched/borderline", &rest));
    table.push(table_row("all", &all));
    println!("V-A4 {}", table.join("\n"));
    table
}

/// The H₂ + H frame with the third H `d` Bohr from H₂'s centre on its axis (H₂ R = 1.4 on x), the
/// chain centred in the box (its extent's midpoint at the origin: at d = 6 the third H would
/// otherwise sit on the box face at x = 6.0).
fn h2_plus_h_at(d: f64) -> Scene {
    let a = -0.5 * (d - 0.7);
    scene(81, vec![Nucleus::new(1, [a - 0.7, 0.0, 0.0]), Nucleus::new(1, [a + 0.7, 0.0, 0.0]), Nucleus::new(1, [a + d, 0.0, 0.0])], 0)
}

/// V-A4 (record only; module doc), frame set (a): a sampling pass on H₂O (97³, `Accepted`; 60
/// perturbations, no burst, no scans) through `common::surrogate::student` + `sample_and_fit_with_floor`
/// (the sampler's time limit off; every job must solve); its
/// samples in store order, the first 40 train experience, the last 20 are held out. Measured
/// (2026-11-16, 997 s; `references.md` "Live orbitals", V-A4): Harris bonded agreement 39/39 near the
/// bond length, 20/21 stretched (the miss an SCF-side detector miss next to the pseudised O), kinds
/// 33/39, per-atom |Δq| 0.20 (q_O ≈ 2× the SCF's); experience coverage 42/60, all 42 right.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va4_harris_bonds_against_scf_and_experience_h2o() {
    let Some(ctx) = gpu() else { return };
    let t = Instant::now();
    // the student's teacher takes the GPU lock per evaluation: the pass runs without ours. The
    // other harris tests hold that lock for their whole bodies (75–116 s each), so a solve here
    // can wait minutes for it; the sampler's time limit would count that wait as a failed solve.
    // It is off (an infinite floor): it is a viewer-responsiveness device, not part of this
    // measurement, and a real SCF stall still counts as failed (final review I1).
    let (mut st, _log) = common::surrogate::student(&ctx, 0);
    let plan = sim::SamplingPlan { perturbations: 60, hot_burst_steps: 0, scan_points_per_pair: 0, ..Default::default() };
    let mut out = String::new();
    let (_, _, p) = common::surrogate::sample_and_fit_with_floor(&mut st, &h2o(), &plan, Some(f64::INFINITY), &mut out);
    print!("V-A4 H₂O sampling pass ({:.0} s):\n{out}", t.elapsed().as_secs_f64());
    assert_eq!((p.failed, p.skipped), (0, 0), "sampling pass: {p:?}");
    let samples: Vec<_> = st.store().expect("samples").samples.iter().filter(|s| s.z.len() == 3).cloned().collect();
    drop(st);
    println!("  {} three-atom samples in store order", samples.len());
    assert!(samples.len() > 20, "the pass produced {} samples", samples.len());
    let n_train = samples.len() - 20;
    let frames: Vec<Frame> = samples.iter().enumerate().map(|(k, s)| {
        let nuclei = s.z.iter().zip(&s.pos).map(|(&z, &p)| Nucleus::new(z, p)).collect();
        let label = format!("#{k} r {:.3}/{:.3} ∠ {:.1}°", dist(s.pos[0], s.pos[1]), dist(s.pos[0], s.pos[2]), angle(s.pos[0], s.pos[1], s.pos[2]));
        Frame { label, scene: scene(97, nuclei, 0), train: k < n_train }
    }).collect();
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let table = va4_set(&ctx, "H₂O sampling pass (97³)", &frames);
    println!("V-A4 H₂O: {:.0} s\n{}", t.elapsed().as_secs_f64(), table.join("\n"));
}

/// V-A4 (record only; module doc), frame set (b): the H₂ scan R = 0.8, 1.0, …, 6.0 (81³, centred on
/// x); even-index frames train experience, odd-index frames are held out. Measured (2026-11-16, 90 s):
/// Harris agreement 12/13 (R = 5.0 crosses the floor), kinds 9/10; experience coverage 0/13 (each
/// held-out R is 0.2 Bohr from the nearest training R, beyond `MATCH_R`).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va4_harris_bonds_against_scf_and_experience_h2_scan() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let frames: Vec<Frame> = (0..=26).map(|k| {
        let r = (8 + 2 * k) as f64 / 10.0;
        Frame { label: format!("R {r:.1}"), scene: scene(81, vec![Nucleus::new(1, [0.5 * r, 0.0, 0.0]), Nucleus::new(1, [-0.5 * r, 0.0, 0.0])], 0), train: k % 2 == 0 }
    }).collect();
    let table = va4_set(&ctx, "H₂ scan (81³)", &frames);
    println!("{}", table.join("\n"));
}

/// V-A4 (record only; module doc), frame set (c): H₂ + H, the third H at d = 6.0, 5.75, …, 2.0 Bohr
/// from H₂'s centre on its axis (81³, the chain centred: `h2_plus_h_at`); even-index frames train
/// experience, odd-index frames are held out. Measured (2026-11-16, 75 s): Harris agreement 23/24
/// (the 2.55 Bohr contact at d = 3.25 crosses 1 + δ), kinds 10/10, per-atom |Δq| 0.021; experience
/// coverage 8/24 (the fixed H₂ pair only), all 8 right.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn va4_harris_bonds_against_scf_and_experience_h2_plus_h() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let frames: Vec<Frame> = (0..=16).map(|k| {
        let d = 6.0 - 0.25 * k as f64;
        Frame { label: format!("d {d:.2}"), scene: h2_plus_h_at(d), train: k % 2 == 0 }
    }).collect();
    let table = va4_set(&ctx, "H₂ + H approach (81³)", &frames);
    println!("{}", table.join("\n"));
}
