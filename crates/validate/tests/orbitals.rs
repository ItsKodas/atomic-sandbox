//! Orbital viewer, the display oracles (plan `docs/superpowers/plans/2026-10-15-orbital-viewer.md`
//! Task 5; spec `docs/superpowers/specs/2026-10-15-orbital-viewer-design.md` §11). Physics checks
//! on what the viewer *shows* — labels, shells, node counts, signs, spin, unoccupied rows — on
//! real LSDA solves of the GPU solver (σ = h), through the display products the app uses
//! (`qm_gpu::orbitals::OrbitalView`, `qm::orbitals`). A display suite: it gates the orbital-view
//! code, not solver changes (spec §11). Numbers of record: `crates/validate/references.md`,
//! "Orbital viewer".
//!
//! `cargo test -p validate --release --test orbitals -- --nocapture --test-threads=1`
//!
//! All GPU, release-only (`#[cfg_attr(debug_assertions, ignore)]`, as the oracles), skipping
//! cleanly without an adapter; GPU sections serialise on `common::GPU`. The injected rotations
//! are seeded (`ORBITALS_SEED` overrides).
//!
//! What each test pins (atomic units; ψ normalised Σψ²h³ = 1; m = ρ↑ − ρ↓):
//! - `h_atom_one_s`: Σψ̃²h³ = 1 ± 1e-4 on the display texture; label "1s"; ε_1s↑ printed next to
//!   the NIST LSD eigenvalue (−0.268975 Ha, majority spin) — a documented gap (smeared nucleus,
//!   box; rule 7), not asserted.
//! - `ne_2p_aligns_to_the_cell_axes`: a seeded O(3) rotation injected into Ne's three 2p fields
//!   per channel (dominance before < 0.9, so the test is not vacuous) is undone by the display:
//!   ⟨p_k|x_k⟩/‖⟨p_k|x⟩‖ > 0.99 and ⟨p_k|x_k⟩ > 0 for k = x, y, z, at a grid point, a cube centre
//!   and off-grid. The grid's 2p splitting is printed and must be what the row tag carries
//!   (`grid_split` = the shell's eigenvalue spread), never bounded (review MUST 1).
//! - `ne_labels_are_1s_2s_2p`: real Ne at four sites, both channels: the measured level order
//!   1s, 2p_x, 2p_y, 2p_z, 2s (Task 0 §1b: the smeared nucleus puts 2s above 2p) labelled by
//!   character and radial nodes, never energy order (ruling 1); `shell_order_inverted: true` (the
//!   documented rule-7 gap, asserted as the outcome, not hidden); 1s tagged smeared (Zσ = 2.5).
//! - `h2_signs_and_tracks_hold_for_100_steps`: 100 NVE steps of H₂ from 1.6 Bohr, the unoccupied
//!   eigensolve (m = 2) at every step: consecutive displayed overlaps ≥ 0.99 for σ_g↑, σ_g↓ and the
//!   lowest unoccupied ↑ (and the lowest ungerade one, σ_u-like, when another); tracks and slots
//!   constant; ↓ shares ↑'s slot; the raw solver's sign flips counted (the rule holds whatever
//!   that count is).
//! - `spin_density_integrals`: ∫m = 1 ± 1e-4 for H and Li (2↑1↓); ∫|m| printed.
//! - `o_triplet_spin_density_integrates_to_two`: ignored until M7 pseudises O (all-electron open-
//!   shell O does not converge on these grids).
//! - `h2_singlet_spin_density_stays_below_the_floor`: max|m| < 0.1 · `SPIN_FLOOR` · max ρ.
//! - `li_labels_and_open_channel`: Li ↑ 1s, 2s; ↓ 1s; the ↑/↓ 1s share a slot (cross overlap
//!   ≥ 0.9); labelled (all s: closed by the occupation rule).
//! - `diag_unoccupied_cost_at_the_default_grid` (`#[ignore]`d measurement): the display
//!   eigensolve's cost and convergence at the app's default h = 0.15 (Task 4 measured h = 0.25).

use gpu::{GpuContext, Volume3D};
use qm::orbitals::{self as qo, LoneAtom, Spin, Row, SPIN_FLOOR};
use qm::{Grid, Nucleus, ScfSettings, System};
use qm_gpu::orbitals::{OrbitalView, Product, ViewUpdate};
use qm_gpu::reduce::neumaier;
use qm_gpu::{Field, GpuGrid, GpuSolver, GpuState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

mod common;
use common::{ms, GPU};

fn gpu() -> Option<GpuContext> { common::gpu("orbitals") }

/// The app's DFT display tolerance and the validate tolerance. `INTERACTIVE_TOL` copies
/// `app::worker::INTERACTIVE_TOL` (private to the `app` binary crate, which cannot be imported
/// here): change the two together.
const INTERACTIVE_TOL: f64 = 1e-5;
const VALIDATE_TOL: f64 = 1e-6;
/// NIST Standard Reference Database 141 (Kotochigova, Levine, Shirley, Stiles, Clark, "Atomic
/// Reference Data for Electronic Structure Calculations"), hydrogen, LSD column, read off the
/// source on 2026-10-15. The LSD column lists two eigenvalues per (nl), and the key to notation
/// says "the first entry for each (nl) corresponds to spin down … 'down' is the direction of the
/// majority spin": the occupied (majority) 1s is −0.268975 Ha, the empty minority 1s −0.100175 Ha.
/// (Same page: LSD E_tot −0.478671 Ha, the value `h_atom_lsda` uses.)
const NIST_LSD_H_1S_MAJORITY: f64 = -0.268975;
const NIST_LSD_H_1S_MINORITY: f64 = -0.100175;
/// The same table, lithium, LSD: 2s majority (occupied) −0.116305 Ha, 1s majority −1.874926 Ha.
const NIST_LSD_LI_2S_MAJORITY: f64 = -0.116305;
const NIST_LSD_LI_1S_MAJORITY: f64 = -1.874926;
/// Task 0 §2: the H₂ singlet's max|m|/max ρ at 49³ (cold solves; 1.4 and 1.6 Bohr).
const TASK0_H2_NOISE: [(f64, f64); 2] = [(1.4, 2.00e-7), (1.6, 2.35e-7)];

const SEED: u64 = 0x0B17_A15E;

/// `ORBITALS_SEED`: decimal, or hex with a `0x` prefix (the tests print their seed as `{:#x}`, so a
/// printed seed pastes back in). An unparsable value stops the run rather than silently re-running
/// the default.
fn seed() -> u64 {
    match std::env::var("ORBITALS_SEED") {
        Err(_) => SEED,
        Ok(v) => {
            let v = v.trim();
            let parsed = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => v.parse(),
            };
            parsed.unwrap_or_else(|e| panic!("ORBITALS_SEED={v:?} is neither decimal nor 0x-hex: {e}"))
        }
    }
}

fn nucleus(pos: [f64; 3], z: u8) -> Nucleus { Nucleus { pos, z } }

/// A cold, converged GPU solve of `nuclei` on n³ × h (σ = h) at `tol`.
fn solve(ctx: &GpuContext, n: usize, h: f64, nuclei: &[Nucleus], tol: f64, multiplicity: Option<u32>) -> (GpuSolver, ScfSettings, usize, f64) {
    let grid = Grid::cube(n, h);
    let mut settings = ScfSettings::for_grid(&grid);
    settings.tol = tol;
    let mut s = GpuSolver::new(ctx.clone(), &grid);
    let t = Instant::now();
    let (_, st) = s.solve(&System::new(nuclei.to_vec(), 0, multiplicity), &settings, false)
        .unwrap_or_else(|e| panic!("{n}³ h {h} {:?}: the SCF did not converge ({e})", nuclei.iter().map(|n| n.z).collect::<Vec<_>>()));
    (s, settings, st.iterations, ms(t.elapsed()))
}

/// ∫ a b d³r of two f32 grid fields, finished in f64 (Neumaier; rule 4).
fn dot32(a: &[f32], b: &[f32], dv: f64) -> f64 {
    neumaier(a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64)) * dv
}

/// ⟨ψ|r_a − c_a⟩ for a = x, y, z (f64 Neumaier on the f32 values).
fn dipole(grid: &Grid, psi: &[f32], c: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|a| neumaier((0..grid.len()).map(|i| psi[i] as f64 * (grid.point(i)[a] - c[a]))) * grid.dv())
}

fn norm3(d: [f64; 3]) -> f64 { (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() }

/// The rows of one channel in index (energy) order.
fn channel(rows: &[Row], spin: Spin) -> Vec<&Row> {
    let mut r: Vec<&Row> = rows.iter().filter(|r| r.spin == spin).collect();
    r.sort_by_key(|r| r.index);
    r
}

fn labels(rows: &[&Row]) -> Vec<String> { rows.iter().map(|r| r.label.to_string()).collect() }

/// One update whose product is the displayed orbital `track` of `spin` (ψ̃, signed), and the
/// texture read back.
fn shown(view: &mut OrbitalView, s: &GpuSolver, state: &GpuState, nuclei: &[Nucleus], tol: f64, spin: Spin, track: u32, vol: &Volume3D, ctx: &GpuContext)
         -> (ViewUpdate, Vec<f32>) {
    let u = view.update(s, state, nuclei, tol, Product::Orbital { spin, track, squared: false, out: vol });
    assert!(u.scale > 0.0, "track {track} {spin:?} is shown by this update");
    (u, vol.read_back(ctx).expect("readback"))
}

/// A padded field with `data` (f64 → f32) on the solver's grid.
fn upload(ctx: &GpuContext, g: &GpuGrid, data: &[f64]) -> Field {
    let f = Field::new(ctx, g, "oracle orbital");
    f.upload_interior(ctx, g, data);
    f
}

fn copy(ctx: &GpuContext, g: &GpuGrid, src: &Field) -> Field {
    let f = Field::new(ctx, g, "oracle copy");
    f.copy_from(ctx, src);
    f
}

/// `state` with other orbital fields (same eigenvalues, densities and potentials, copied): what the
/// display sees when the eigensolver hands back another basis of the same occupied space.
fn with_orbitals(ctx: &GpuContext, g: &GpuGrid, st: &GpuState, up: Vec<Field>, dn: Vec<Field>) -> GpuState {
    GpuState {
        grid: st.grid.clone(), sigma: st.sigma, orbitals_up: up, orbitals_dn: dn, evals_up: st.evals_up.clone(), evals_dn: st.evals_dn.clone(),
        rho: copy(ctx, g, &st.rho), rho_up: copy(ctx, g, &st.rho_up), rho_dn: copy(ctx, g, &st.rho_dn), v_h: copy(ctx, g, &st.v_h), v_nuc: copy(ctx, g, &st.v_nuc),
    }
}

/// A Haar-random orthogonal 3 × 3 matrix (Gram–Schmidt of a Gaussian matrix; det ±1, so a
/// reflection — a sign — is as likely as not).
fn random_orthogonal(rng: &mut StdRng) -> [[f64; 3]; 3] {
    let normal = |rng: &mut StdRng| {
        let u1: f64 = rng.gen::<f64>().max(1e-300);
        let u2: f64 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut q = [[0.0; 3]; 3];
    for i in 0..3 {
        let mut v: [f64; 3] = std::array::from_fn(|_| normal(rng));
        for qj in q.iter().take(i) {
            let d: f64 = (0..3).map(|a| v[a] * qj[a]).sum();
            for a in 0..3 { v[a] -= d * qj[a]; }
        }
        let n = norm3(v);
        q[i] = v.map(|x| x / n);
    }
    q
}

/// The Ne sites of Task 0 at spacing h.
fn sites(h: f64) -> [(&'static str, [f64; 3]); 4] {
    [("grid point", [0.0; 3]), ("cube centre", [h / 2.0; 3]), ("off-grid", [0.37 * h, -0.21 * h, 0.12 * h]), ("(h/2,0,0)", [h / 2.0, 0.0, 0.0])]
}

fn spread(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max) - v.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Spec §11 normalisation and the H atom's label; ε_1s↑ against NIST's LSD eigenvalue (rule 7:
/// printed with the gap, not asserted — the σ = h nucleus is 11 % shallower in one-electron terms
/// at Zσ = 0.25 and the 12 Bohr box confines; neither is a display matter).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn h_atom_one_s() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let nuclei = [nucleus([0.0; 3], 1)];
    let (s, settings, steps, wall) = solve(&ctx, 49, 0.25, &nuclei, VALIDATE_TOL, None);
    let state = s.state().expect("converged");
    let grid = s.grid().clone();
    let mut view = OrbitalView::new(&s);
    let vol = Volume3D::new(&ctx, [49; 3]);
    let probe = view.update(&s, state, &nuclei, settings.tol, Product::None);
    assert_eq!(probe.out.rows.len(), 1, "one occupied orbital, ↑; the ↓ channel is empty");
    let row = &probe.out.rows[0];
    let (u, t) = shown(&mut view, &s, state, &nuclei, settings.tol, Spin::Up, row.track, &vol, &ctx);
    let norm = dot32(&t, &t, grid.dv());
    let monopole = neumaier(t.iter().map(|v| *v as f64)) * grid.dv();
    let e = state.evals_up[0];
    println!("H 49³ σ = h = 0.25, tol {VALIDATE_TOL:.0e}: {steps} SCF steps, {wall:.0} ms");
    println!("  display texture: Σψ̃²h³ = {norm:.9} (|1 − Σ| = {:.2e}); label {}; lone atom {:?}; smeared 1s {}; ∫ψ̃ = {monopole:+.6}",
             (1.0 - norm).abs(), row.label, probe.out.lone_atom, row.smeared_1s);
    println!("  ε_1s↑ = {e:.6} Ha; NIST LSD (point nucleus, no box) 1s majority {NIST_LSD_H_1S_MAJORITY:.6} Ha (minority, empty, {NIST_LSD_H_1S_MINORITY:.6}); \
              gap ours − NIST = {:+.6} Ha (rule 7, not asserted): +0.02950 Ha of it is the σ = h erf nucleus (radial LSDA solve, isolated: ε_1s↑ = −0.23952 Ha, Task 5 review P1) and +0.00217 Ha the 12 Bohr box with the h = 0.25 grid (bracketed by the Dirichlet spheres, −0.23246 inscribed / −0.23948 circumscribed); exact −0.5 Ha (LSDA self-interaction)",
             e - NIST_LSD_H_1S_MAJORITY);
    assert!((norm - 1.0).abs() <= 1e-4, "Σψ̃²h³ = {norm}");
    assert_eq!(row.label.to_string(), "1s");
    assert_eq!(probe.out.lone_atom, LoneAtom::Labelled { shell_order_inverted: false });
    assert!(!row.smeared_1s, "Zσ = 0.25 < SMEARED_ZSIGMA");
    assert!(monopole > 0.0, "first appearance: the 1s is shown positive");
    assert_eq!(u.out.rows[0].track, row.track);
}

/// Review MUST 1 / Review Focus 2–3 on the real solver: the display undoes any rotation of Ne's
/// 2p set the eigensolver might hand back, at symmetric and egg-box-split sites, both channels.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ne_2p_aligns_to_the_cell_axes() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (n, h) = (57usize, 0.25);
    let mut rng = StdRng::seed_from_u64(seed());
    println!("Ne 57³ σ = h = 0.25, tol {INTERACTIVE_TOL:.0e}, seed {:#x}", seed());
    for (site, pos) in sites(h).into_iter().take(3) {
        let nuclei = [nucleus(pos, 10)];
        let (s, settings, steps, wall) = solve(&ctx, n, h, &nuclei, INTERACTIVE_TOL, None);
        let state = s.state().expect("converged");
        let grid = s.grid().clone();
        let g = GpuGrid::new(&grid);
        let vol = Volume3D::new(&ctx, [n as u32; 3]);
        println!("  {site} {pos:?}: {steps} SCF steps, {wall:.0} ms");
        let mut fields: [Vec<Field>; 2] = [Vec::new(), Vec::new()];
        for (ch, (orbs, evals)) in [(&state.orbitals_up, &state.evals_up), (&state.orbitals_dn, &state.evals_dn)].into_iter().enumerate() {
            let raw: Vec<Vec<f64>> = orbs.iter().map(|f| f.read_interior(&ctx, &g)).collect();
            let p: Vec<usize> = (0..raw.len()).filter(|&i| qo::dominant_l(&qo::moments(&grid, &raw[i], pos)) == Some(1)).collect();
            assert_eq!(p.len(), 3, "{site}: three p-dominant occupied orbitals per channel");
            let p_evals: Vec<f64> = p.iter().map(|&i| evals[i]).collect();
            // the injected rotation: drawn until the fixture is not already aligned (min dominance < 0.9)
            let (mut draws, mut rotated, mut before) = (0, Vec::new(), 1.0f64);
            while before >= 0.9 {
                draws += 1;
                assert!(draws <= 100, "100 rotations all left the set aligned");
                let r = random_orthogonal(&mut rng);
                rotated = (0..3).map(|j| (0..grid.len()).map(|q| (0..3).map(|a| r[j][a] * raw[p[a]][q]).sum::<f64>()).collect::<Vec<f64>>()).collect();
                before = rotated.iter().map(|v| {
                    let d = dipole(&grid, &v.iter().map(|x| *x as f32).collect::<Vec<f32>>(), pos);
                    d.iter().fold(0.0f64, |m, x| m.max(x.abs())) / norm3(d)
                }).fold(1.0, f64::min);
            }
            let mut set: Vec<Field> = Vec::new();
            for i in 0..raw.len() {
                set.push(match p.iter().position(|&k| k == i) { Some(j) => upload(&ctx, &g, &rotated[j]), None => copy(&ctx, &g, &orbs[i]) });
            }
            println!("    {}: 2p ε {:?} (grid split {:.3e} Ha); injected rotation after {draws} draw(s): min dominance before {before:.3}",
                     if ch == 0 { "↑" } else { "↓" }, p_evals.iter().map(|e| format!("{e:.6}")).collect::<Vec<_>>(), spread(&p_evals));
            fields[ch] = set;
        }
        let [up, dn] = fields;
        let injected = with_orbitals(&ctx, &g, state, up, dn);
        let mut view = OrbitalView::new(&s);
        let probe = view.update(&s, &injected, &nuclei, settings.tol, Product::None);
        assert!(matches!(probe.out.lone_atom, LoneAtom::Labelled { .. }), "{site}: {:?}", probe.out.lone_atom);
        for spin in [Spin::Up, Spin::Down] {
            let rows = channel(&probe.out.rows, spin);
            let evals = if spin == Spin::Up { &state.evals_up } else { &state.evals_dn };
            for (k, axis) in ["x", "y", "z"].into_iter().enumerate() {
                let want = format!("2p_{axis}");
                let row = rows.iter().find(|r| r.label.to_string() == want).unwrap_or_else(|| panic!("{site} {spin:?}: no {want} in {:?}", labels(&rows)));
                let (_, t) = shown(&mut view, &s, &injected, &nuclei, settings.tol, spin, row.track, &vol, &ctx);
                let d = dipole(&grid, &t, pos);
                let dom = d[k].abs() / norm3(d);
                // the tag carries the shell's measured eigenvalue spread, however large (MUST 1)
                let p_evals: Vec<f64> = rows.iter().filter(|r| r.label.to_string().starts_with("2p")).map(|r| evals[r.index]).collect();
                let split = row.grid_split.unwrap_or_else(|| panic!("{site} {spin:?} {want}: the rotated 2p row carries its grid split"));
                println!("      {spin:?} {want}: ⟨p|x_k⟩ = {:+.5} of ‖⟨p|x⟩‖ {:.5} → dominance {dom:.8} (1 − {:.1e}); ε̄ {:.6}; tag: split by the grid by {:.4} mHa",
                         d[k], norm3(d), 1.0 - dom, row.energy, split * 1e3);
                assert!(dom > 0.99, "{site} {spin:?} {want}: dominance {dom}");
                assert!(d[k] > 0.0, "{site} {spin:?} {want}: the positive lobe on +{axis} (built-in sign)");
                assert!(row.rotated, "{site} {spin:?} {want}: tagged rotated for display");
                assert!((split - spread(&p_evals)).abs() <= 1e-15, "{site}: the tag is the measured spread ({split} vs {})", spread(&p_evals));
            }
        }
    }
}

/// Ruling 1 on the real atom: labels from angular character and radial node counts at four sites,
/// both channels, in the measured level order with 2s above 2p (the smeared nucleus: documented
/// rule-7 outcome, asserted as `shell_order_inverted: true`, not hidden).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn ne_labels_are_1s_2s_2p() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (n, h) = (57usize, 0.25);
    println!("Ne 57³ σ = h = 0.25 (Zσ = 2.5), tol {INTERACTIVE_TOL:.0e}");
    for (site, pos) in sites(h) {
        let nuclei = [nucleus(pos, 10)];
        let (s, settings, steps, wall) = solve(&ctx, n, h, &nuclei, INTERACTIVE_TOL, None);
        let state = s.state().expect("converged");
        let mut view = OrbitalView::new(&s);
        let sc = view.read_scalars(&s, state, &nuclei);
        let radial = sc.radial.as_ref().expect("a lone atom has radial profiles");
        let out = view.update(&s, state, &nuclei, settings.tol, Product::None).out;
        println!("  {site} {pos:?}: {steps} SCF steps, {wall:.0} ms; {:?}", out.lone_atom);
        assert_eq!(out.lone_atom, LoneAtom::Labelled { shell_order_inverted: true }, "{site}: labelled, with the inversion reported");
        for spin in [Spin::Up, Spin::Down] {
            let ch = spin.index();
            let evals = if spin == Spin::Up { &state.evals_up } else { &state.evals_dn };
            let nodes: Vec<String> = (0..evals.len()).map(|i| {
                let m = &sc.moments[ch][i];
                let l = qo::dominant_l(m);
                let nu = l.and_then(|l| qo::profile_nodes(&radial[ch][i], m, l));
                format!("{:.6} (l {} ν {})", evals[i], l.map_or("-".into(), |l| l.to_string()), nu.map_or("-".into(), |v| v.to_string()))
            }).collect();
            let rows = channel(&out.rows, spin);
            println!("    {spin:?}: {:?}; raw ε (l, radial nodes): {}", labels(&rows), nodes.join(", "));
            assert_eq!(labels(&rows), ["1s", "2p_x", "2p_y", "2p_z", "2s"], "{site} {spin:?}: the measured order, labelled by character and nodes");
            for r in &rows {
                assert_eq!(r.smeared_1s, r.label.to_string() == "1s", "{site} {spin:?} {}: smeared-1s tag on the 1s only", r.label);
            }
            assert!(rows[4].energy > rows[1].energy, "{site}: 2s above 2p (Task 0 §1b)");
        }
    }
}

/// Review Focus 4 / SHOULD 13 on the real solver: 100 NVE steps of a vibrating H₂ with the
/// unoccupied eigensolve (m = 2) at every step, as the provider runs it paused (`GpuKs` with
/// `force_every_step`). Driven through `GpuSolver` + `OrbitalView` — `GpuKs`'s own parts in its
/// order, with its density extrapolation on (`set_density_extrapolation(true)`, as `GpuKs` sets
/// it) — rather than through `GpuKs`, because the raw solver's sign (the quantity whose flips the
/// rule must absorb) is not visible through the provider. `GpuKs` also wraps each solve in
/// `retry_cold_on_stall`, which this driver does not (no stall occurs on this trajectory). It
/// starts at R = 1.6 Bohr, the σ = h = 0.15 minimum (the `h2` preset file starts at 1.45), which
/// is also the case with the larger amplitude per step it was written for.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn h2_signs_and_tracks_hold_for_100_steps() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    const STEPS: usize = 100;
    const M: usize = 2;
    let r0 = 1.6;
    let mut scene = common::h2(r0); // 49³ × 0.25, at rest
    let qnuc = |sc: &sim::Scene| -> Vec<Nucleus> { sc.nuclei.iter().map(|n| nucleus(n.pos, n.z)).collect() };
    let grid = Grid::cube(scene.cell.n, scene.cell.h);
    let dv = grid.dv();
    let mut settings = ScfSettings::for_grid(&grid);
    settings.tol = INTERACTIVE_TOL;
    let mut s = GpuSolver::new(ctx.clone(), &grid);
    s.set_density_extrapolation(true);
    let mut view = OrbitalView::new(&s);
    view.set_face_limit(providers::box_affected_face_density(grid.h));
    let vol = Volume3D::new(&ctx, [scene.cell.n as u32; 3]);
    let dt = sim::Integrator::new_fs(common::DT_FS).dt;
    // one step's work: a warm solve, the forces, the forced unoccupied eigensolve, the updates
    // (`out`: the last update's rows and rotations; `fields`: the followed orbitals' ψ̃)
    struct Frame { e: f64, f: Vec<[f64; 3]>, out: qo::UpdateOutput, fields: Vec<Vec<f32>>, eig_s: f64, iters: [usize; 2] }
    let mut tracks: Option<(u32, u32, u32)> = None;
    let frame = |s: &mut GpuSolver, view: &mut OrbitalView, sc: &sim::Scene, warm: bool, tracks: &mut Option<(u32, u32, u32)>| -> Frame {
        let nuclei = qnuc(sc);
        let (_, st) = s.solve(&System::new(nuclei.clone(), 0, None), &settings, warm).expect("H₂ converges");
        let e = st.energy.total;
        let f = s.forces(s.state().expect("converged"), &nuclei, settings.sigma).expect("forces");
        let t = Instant::now();
        let u = view.unoccupied(s, &nuclei, M, &settings, &mut || false).expect("not interrupted");
        let eig_s = t.elapsed().as_secs_f64();
        assert!(u.rejected.is_none(), "the display solve agrees with the SCF: {:?}", u.rejected);
        let state = s.state().expect("converged");
        let probe = view.update(s, state, &nuclei, settings.tol, Product::None).out;
        if tracks.is_none() {
            let up = channel(&probe.rows, Spin::Up);
            let dn = channel(&probe.rows, Spin::Down);
            let unocc: Vec<&&Row> = up.iter().filter(|r| r.occupation == 0.0).collect();
            assert!(!unocc.is_empty(), "m = {M}: at least one unoccupied ↑ row shown");
            *tracks = Some((up[0].track, dn[0].track, unocc[0].track));
        }
        let (tg_up, tg_dn, tu) = tracks.expect("set");
        let mut fields = Vec::new();
        let mut out = probe;
        for (spin, track) in [(Spin::Up, tg_up), (Spin::Down, tg_dn), (Spin::Up, tu)] {
            let (u, t) = shown(view, s, state, &nuclei, settings.tol, spin, track, &vol, &ctx);
            out = u.out;
            fields.push(t);
        }
        Frame { e, f, out, fields, eig_s, iters: u.iterations }
    };
    let mut prev = frame(&mut s, &mut view, &scene, false, &mut tracks);
    let (tg_up, tg_dn, tu) = tracks.expect("set");
    // the first frame's shown unoccupied ↑ rows and their mirror parities (σ_u: odd in x, even in y, z)
    let n = scene.cell.n;
    let mirror = |f: &[f32], axis: usize| -> f64 {
        let idx = |i: [usize; 3]| i[0] + n * (i[1] + n * i[2]);
        neumaier((0..grid.len()).map(|p| {
            let i = [p % n, (p / n) % n, p / (n * n)];
            let mut j = i;
            j[axis] = n - 1 - i[axis];
            f[p] as f64 * f[idx(j)] as f64
        })) * dv
    };
    let up_rows = channel(&prev.out.rows, Spin::Up);
    let unocc_up: Vec<&&Row> = up_rows.iter().filter(|r| r.occupation == 0.0).collect();
    let mut parity_line = Vec::new();
    let mut ungerade: Option<u32> = None;
    for r in &unocc_up {
        let (_, t) = shown(&mut view, &s, s.state().expect("converged"), &qnuc(&scene), settings.tol, Spin::Up, r.track, &vol, &ctx);
        let p = [mirror(&t, 0), mirror(&t, 1), mirror(&t, 2)];
        parity_line.push(format!("{} ε {:+.5} box {} (Px {:+.3} Py {:+.3} Pz {:+.3})", r.label, r.energy, r.box_state, p[0], p[1], p[2]));
        if ungerade.is_none() && p[0] < -0.99 && p[1] > 0.99 && p[2] > 0.99 {
            ungerade = Some(r.track);
        }
    }
    println!("H₂ from R = {r0} Bohr, 49³, σ = h = 0.25, tol {INTERACTIVE_TOL:.0e}, dt {} fs, {STEPS} NVE steps, m = {M} (eigensolve every step)", common::DT_FS);
    println!("  shown unoccupied ↑ at t = 0: {}", parity_line.join("; "));
    let sigma_u = ungerade.expect("a σ_u-like (odd in x, even in y and z) unoccupied ↑ row among the shown ones (review SHOULD 13)");
    let extra = (sigma_u != tu).then_some(sigma_u);
    // the σ_u-like row is followed too when it is not the lowest unoccupied
    let mut followed: Vec<(Spin, u32, &str)> = vec![(Spin::Up, tg_up, "σ_g ↑"), (Spin::Down, tg_dn, "σ_g ↓"), (Spin::Up, tu, "lowest unoccupied ↑")];
    if let Some(t) = extra { followed.push((Spin::Up, t, "σ_u-like ↑")); }
    let read_extra = |view: &mut OrbitalView, s: &GpuSolver, sc: &sim::Scene| -> Vec<f32> {
        extra.map(|t| shown(view, s, s.state().expect("converged"), &qnuc(sc), settings.tol, Spin::Up, t, &vol, &ctx).1).unwrap_or_default()
    };
    if extra.is_some() { prev.fields.push(read_extra(&mut view, &s, &scene)); }
    let slots0: Vec<(Spin, usize, u32, u8)> = prev.out.rows.iter().filter(|r| r.occupation == 1.0).map(|r| (r.spin, r.index, r.track, r.slot)).collect();
    let slot_of = |rows: &[Row], spin: Spin, track: u32| rows.iter().find(|r| r.spin == spin && r.track == track).map(|r| r.slot);
    assert_eq!(slot_of(&prev.out.rows, Spin::Down, tg_dn), slot_of(&prev.out.rows, Spin::Up, tg_up), "↓ σ_g shares ↑ σ_g's colour slot");
    let unocc_slot = slot_of(&prev.out.rows, Spin::Up, tu);
    // the applied sign of a 1 × 1 set (ψ̃ = s ψ_raw): a raw flip is a change of s between steps
    let sign_of = |out: &qo::UpdateOutput, ch: usize, index: usize| -> Option<f64> {
        out.rotations[ch].iter().find(|(r, _)| r.contains(&index)).and_then(|(r, m)| (r.len() == 1).then(|| m[(0, 0)].signum()))
    };
    let index_of = |rows: &[Row], spin: Spin, track: u32| rows.iter().find(|r| r.spin == spin && r.track == track).map(|r| r.index).expect("tracked row present");
    let lowest_unoccupied = |rows: &[Row]| channel(rows, Spin::Up).into_iter().find(|r| r.occupation == 0.0).map(|r| r.track);
    let mut flips = vec![0usize; followed.len()];
    let mut worst = vec![f64::INFINITY; followed.len()];
    let mut unrotatable = vec![0usize; followed.len()];
    let (mut f_now, mut e0, mut e_lo, mut e_hi) = (prev.f.clone(), f64::NAN, f64::INFINITY, f64::NEG_INFINITY);
    let (mut eig_total, mut iters_max, mut reordered) = (prev.eig_s, prev.iters, 0usize);
    let t_md = Instant::now();
    for step in 1..=STEPS {
        sim::Integrator::half_kick(&mut scene, &f_now, dt);
        sim::Integrator::drift(&mut scene, dt);
        let mut cur = frame(&mut s, &mut view, &scene, true, &mut tracks);
        if extra.is_some() { cur.fields.push(read_extra(&mut view, &s, &scene)); }
        f_now = cur.f.clone();
        sim::Integrator::half_kick(&mut scene, &f_now, dt);
        let e_tot = cur.e + scene.kinetic_energy();
        if step == 1 { e0 = e_tot; }
        e_lo = e_lo.min(e_tot);
        e_hi = e_hi.max(e_tot);
        eig_total += cur.eig_s;
        iters_max = [iters_max[0].max(cur.iters[0]), iters_max[1].max(cur.iters[1])];
        for (k, &(spin, track, what)) in followed.iter().enumerate() {
            let o = dot32(&cur.fields[k], &prev.fields[k], dv);
            worst[k] = worst[k].min(o);
            assert!(o >= 0.99, "step {step}: {what} ⟨ψ̃_t|ψ̃_t−1⟩ = {o}");
            let (i0, i1) = (index_of(&prev.out.rows, spin, track), index_of(&cur.out.rows, spin, track));
            let ch = spin.index();
            match (sign_of(&prev.out, ch, i0), sign_of(&cur.out, ch, i1)) {
                (Some(a), Some(b)) => if a != b { flips[k] += 1 },
                _ => unrotatable[k] += 1,
            }
        }
        // tracks and slots constant: every occupied row keeps its (spin, index) → (track, slot)
        let slots: Vec<(Spin, usize, u32, u8)> = cur.out.rows.iter().filter(|r| r.occupation == 1.0).map(|r| (r.spin, r.index, r.track, r.slot)).collect();
        assert_eq!(slots, slots0, "step {step}: occupied tracks and slots constant");
        assert_eq!(slot_of(&cur.out.rows, Spin::Down, tg_dn), slot_of(&cur.out.rows, Spin::Up, tg_up), "step {step}: ↓ shares ↑'s slot");
        // the followed unoccupied orbital keeps its track and slot (the selection follows the
        // orbital, so a level crossing would move it off "lowest" — counted, not a failure)
        assert_eq!(slot_of(&cur.out.rows, Spin::Up, tu), unocc_slot, "step {step}: the followed unoccupied ↑ keeps its track and slot");
        if lowest_unoccupied(&cur.out.rows) != Some(tu) { reordered += 1; }
        if let Some(t) = extra { assert!(slot_of(&cur.out.rows, Spin::Up, t).is_some(), "step {step}: the σ_u-like track is still shown"); }
        prev = cur;
    }
    let bond = common::bond(&scene);
    println!("  {STEPS} steps in {:.1} s ({:.2} s of display eigensolves; LOBPCG iterations ≤ {:?}); R(end) = {bond:.4} Bohr; E_total span {:.2e} Ha (E₀ {e0:.6}); \
              steps where the followed unoccupied ↑ was not the lowest: {reordered}",
             t_md.elapsed().as_secs_f64(), eig_total, iters_max, e_hi - e_lo);
    for (k, (_, _, what)) in followed.iter().enumerate() {
        println!("  {what}: worst consecutive ⟨ψ̃_t|ψ̃_t−1⟩ = {:.6}; raw solver sign flips absorbed: {}{}", worst[k], flips[k],
                 if unrotatable[k] > 0 { format!(" (+{} steps in a degenerate set: rotated, not counted)", unrotatable[k]) } else { String::new() });
    }
}

/// Spec §11 "Spin": ∫m = n↑ − n↓ to 1e-4 for H (1↑) and Li (2↑ 1↓) — the viewer's spin summary
/// (the reduction the readout shows) and the spin texture integrated in f64; ∫|m| printed.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn spin_density_integrals() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (name, z, n_up, n_dn) in [("H", 1u8, 1usize, 0usize), ("Li", 3, 2, 1)] {
        let nuclei = [nucleus([0.0; 3], z)];
        let (s, settings, steps, wall) = solve(&ctx, 49, 0.25, &nuclei, VALIDATE_TOL, None);
        let state = s.state().expect("converged");
        let dv = s.grid().dv();
        let mut view = OrbitalView::new(&s);
        let vol = Volume3D::new(&ctx, [49; 3]);
        let u = view.update(&s, state, &nuclei, settings.tol, Product::Spin { out: &vol });
        let t = vol.read_back(&ctx).expect("readback");
        let tex = neumaier(t.iter().map(|v| *v as f64)) * dv;
        let tex_abs = neumaier(t.iter().map(|v| v.abs() as f64)) * dv;
        let sp = u.spin;
        println!("{name} 49³ σ = h = 0.25, tol {VALIDATE_TOL:.0e} ({steps} steps, {wall:.0} ms): n↑ {} n↓ {}; ∫m = {:.9} (texture {tex:.9}); ∫|m| = {:.9} (texture {tex_abs:.9}); max|m| {:.4e}, max ρ {:.4e}, scale {:.4e}",
                 sp.n_up, sp.n_dn, sp.integral, sp.abs_integral, sp.max_abs, sp.rho_max, sp.scale());
        assert_eq!((sp.n_up, sp.n_dn), (n_up, n_dn));
        let want = (n_up - n_dn) as f64;
        assert!((sp.integral - want).abs() <= 1e-4, "{name}: ∫m = {} (want {want})", sp.integral);
        assert!((tex - want).abs() <= 1e-4, "{name}: texture ∫m = {tex}");
        assert_eq!(u.scale, sp.scale() as f32);
        if z == 3 {
            // core spin polarisation (radial LSDA: 4–8e-4, against f32 noise of ~1e-7): the view
            // resolves a real feature of the functional, not just ∫m = n↑ − n↓
            assert!(sp.abs_integral - sp.integral > 1e-4, "Li: ∫|m| − ∫m = {:.3e}", sp.abs_integral - sp.integral);
        }
    }
}

/// Spec §11: the O triplet's ∫m = 2. Ignored with its cause cited (spec §11, plan Task 5): the
/// all-electron open-shell O does not converge on these grids — Task 0: 200 steps, Δρ oscillating
/// near 1e-4 at 57³ (`references.md` "The O reference is slow and keeps the singlet"). Run with
/// `--ignored` it fails on that. Un-ignored only under an M7 policy that pseudises O, at M7's
/// Hund gate; the body then solves the pseudised O instead.
#[test]
#[ignore = "the open-shell all-electron O does not converge at default grids (references.md: 'The O reference is slow and keeps the singlet'); un-ignored only under an M7 policy that pseudises O, at M7's Hund gate"]
fn o_triplet_spin_density_integrates_to_two() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let nuclei = [nucleus([0.0; 3], 8)];
    let (s, settings, _, _) = solve(&ctx, 57, 0.25, &nuclei, VALIDATE_TOL, Some(3));
    let state = s.state().expect("converged");
    let mut view = OrbitalView::new(&s);
    let u = view.update(&s, state, &nuclei, settings.tol, Product::None);
    println!("O triplet: ∫m = {:.9}, ∫|m| = {:.9}", u.spin.integral, u.spin.abs_integral);
    assert!((u.spin.integral - 2.0).abs() <= 1e-4, "∫m = {}", u.spin.integral);
}

/// Review SHOULD 8 (spec §5.5): the closed-shell spin noise stays under the display floor's
/// requirement, max|m| < 0.1 · `SPIN_FLOOR` · max ρ = 1e-4 · max ρ, so the default 0.1 iso never
/// paints it. H₂ at 1.4 and 1.6 Bohr, both tolerances; the ratio printed next to Task 0's.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn h2_singlet_spin_density_stays_below_the_floor() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let limit = 0.1 * SPIN_FLOOR;
    for (r, task0) in TASK0_H2_NOISE {
        for tol in [INTERACTIVE_TOL, VALIDATE_TOL] {
            let nuclei = [nucleus([r / 2.0, 0.0, 0.0], 1), nucleus([-r / 2.0, 0.0, 0.0], 1)];
            let (s, settings, steps, _) = solve(&ctx, 49, 0.25, &nuclei, tol, None);
            let state = s.state().expect("converged");
            let mut view = OrbitalView::new(&s);
            let sp = view.update(&s, state, &nuclei, settings.tol, Product::None).spin;
            let ratio = sp.max_abs / sp.rho_max;
            println!("H₂ R = {r} 49³ tol {tol:.0e} ({steps} steps): max|m|/max ρ = {ratio:.3e} (Task 0: {task0:.2e}; limit {limit:.0e}, margin {:.0}×); ∫m {:+.2e}, ∫|m| {:.2e}",
                     limit / ratio.max(f64::MIN_POSITIVE), sp.integral, sp.abs_integral);
            assert_eq!((sp.n_up, sp.n_dn), (1, 1));
            assert!(ratio < limit, "R = {r}, tol {tol}: max|m|/max ρ = {ratio:.3e} ≥ {limit:.0e}");
        }
    }
}

/// Spec §5.3 items 4–5 on an open-shell s atom: Li ↑ 1s, 2s; ↓ 1s — labelled, because every
/// occupied l-set is complete (all s); the ↑ and ↓ 1s are one spatial orbital (cross overlap
/// ≥ `SHARE_COLOUR_MIN` = 0.9) and share a colour slot; both 1s tagged smeared (Zσ = 0.75).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn li_labels_and_open_channel() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let nuclei = [nucleus([0.0; 3], 3)];
    let (s, settings, steps, wall) = solve(&ctx, 49, 0.25, &nuclei, VALIDATE_TOL, None);
    let state = s.state().expect("converged");
    let mut view = OrbitalView::new(&s);
    let vol = Volume3D::new(&ctx, [49; 3]);
    let sc = view.read_scalars(&s, state, &nuclei);
    let radial = sc.radial.as_ref().expect("a lone atom");
    let out = view.update(&s, state, &nuclei, settings.tol, Product::None).out;
    let (up, dn) = (channel(&out.rows, Spin::Up), channel(&out.rows, Spin::Down));
    let nodes = |ch: usize, i: usize| qo::profile_nodes(&radial[ch][i], &sc.moments[ch][i], 0);
    println!("Li 49³ σ = h = 0.25, tol {VALIDATE_TOL:.0e} ({steps} steps, {wall:.0} ms): {:?}", out.lone_atom);
    println!("  ↑ {:?} ε {:.6} {:.6} (radial nodes {:?} {:?}); ↓ {:?} ε {:.6} ({:?})", labels(&up), state.evals_up[0], state.evals_up[1], nodes(0, 0), nodes(0, 1),
             labels(&dn), state.evals_dn[0], nodes(1, 0));
    println!("  NIST LSD (point nucleus): 1s majority {NIST_LSD_LI_1S_MAJORITY:.6}, 2s majority {NIST_LSD_LI_2S_MAJORITY:.6} Ha — ours differ by the σ = h nucleus (Zσ = 0.75) and the box (rule 7; not asserted)");
    assert_eq!(out.lone_atom, LoneAtom::Labelled { shell_order_inverted: false });
    assert_eq!(labels(&up), ["1s", "2s"]);
    assert_eq!(labels(&dn), ["1s"]);
    let (_, t_up) = shown(&mut view, &s, state, &nuclei, settings.tol, Spin::Up, up[0].track, &vol, &ctx);
    let (u, t_dn) = shown(&mut view, &s, state, &nuclei, settings.tol, Spin::Down, dn[0].track, &vol, &ctx);
    let cross = dot32(&t_up, &t_dn, s.grid().dv());
    let rows = &u.out.rows;
    let slot = |spin: Spin, i: usize| rows.iter().find(|r| r.spin == spin && r.index == i).map(|r| r.slot);
    println!("  ⟨1s↑|1s↓⟩ (displayed) = {cross:.6}; slots ↑1s {:?} ↑2s {:?} ↓1s {:?}", slot(Spin::Up, 0), slot(Spin::Up, 1), slot(Spin::Down, 0));
    assert!(cross >= qo::SHARE_COLOUR_MIN, "⟨1s↑|1s↓⟩ = {cross}");
    assert_eq!(slot(Spin::Down, 0), slot(Spin::Up, 0), "↓ 1s shares ↑ 1s's slot");
    assert_ne!(slot(Spin::Up, 1), slot(Spin::Up, 0), "2s has its own");
    for r in rows.iter() {
        assert_eq!(r.smeared_1s, r.label.to_string() == "1s", "{} {:?}", r.label, r.spin);
    }
}

/// Task 4 re-review: the display eigensolve's cost and convergence at the app's default grid,
/// h = σ = 0.15 (Accurate: 81³ for the 12 Bohr H₂ preset, 95³ for the 14 Bohr H₂O preset), m = 4,
/// the app's tol 1e-5 — cold, then warm after a 0.01 Bohr move — against the warm SCF wall, plus
/// one display update. A measurement: prints, asserts only that it ran.
/// `cargo test -p validate --release --test orbitals -- --ignored --nocapture diag_unoccupied_cost`
#[test]
#[ignore = "measurement (references.md 'Orbital viewer', Task 5): the display eigensolve at h = 0.15"]
fn diag_unoccupied_cost_at_the_default_grid() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    const M: usize = 4;
    let h = 0.15;
    let systems: [(&str, usize, Vec<Nucleus>); 2] = [
        ("H₂ (1.45 Bohr)", 81, vec![nucleus([0.725, 0.0, 0.0], 1), nucleus([-0.725, 0.0, 0.0], 1)]),
        ("H₂O (preset)", 95, vec![nucleus([0.0; 3], 8), nucleus([1.43, 1.11, 0.0], 1), nucleus([-1.43, 1.11, 0.0], 1)]),
    ];
    println!("display eigensolve at h = σ = {h}, m = {M}, SCF tol {INTERACTIVE_TOL:.0e} (contended GPU possible; walls reported, never asserted)");
    for (name, n, nuclei) in systems {
        let (mut s, settings, steps, wall) = solve(&ctx, n, h, &nuclei, INTERACTIVE_TOL, None);
        let mut view = OrbitalView::new(&s);
        view.set_face_limit(providers::box_affected_face_density(h));
        let line = |what: &str, s: &mut GpuSolver, view: &mut OrbitalView, nuclei: &[Nucleus], scf: (usize, f64)| {
            let t = Instant::now();
            let u = view.unoccupied(s, nuclei, M, &settings, &mut || false).expect("not interrupted");
            let eig = t.elapsed().as_secs_f64();
            let state = s.state().expect("converged");
            let t = Instant::now();
            let up = view.update(s, state, nuclei, settings.tol, Product::None);
            let upd = ms(t.elapsed());
            let unocc: Vec<&Row> = up.out.rows.iter().filter(|r| r.occupation == 0.0).collect();
            let tagged = unocc.iter().filter(|r| r.not_converged).count();
            let bound = unocc.iter().filter(|r| r.energy < 0.0).count();
            let lowest = unocc.iter().map(|r| r.energy).fold(f64::INFINITY, f64::min);
            println!("  {name} {n}³ {what}: SCF {} steps {:.0} ms | eigensolve {eig:.2} s = {:.1}× the SCF, iterations ↑/↓ {:?}, cap {:?}, tol {:.3e}, |Δε_occ| {:.1e}/{:.1e}, \
                      {} shown ({tagged} not converged, {bound} bound; lowest ε {lowest:+.5}), hidden {:?}{} | update {upd:.1} ms",
                     scf.0, scf.1, eig * 1e3 / scf.1, u.iterations, u.at_cap, u.tol, u.occupied_error[0], u.occupied_error[1], unocc.len(), up.out.hidden_unoccupied,
                     u.rejected.as_ref().map_or(String::new(), |r| format!(" REJECTED: {r}")));
        };
        line("cold", &mut s, &mut view, &nuclei, (steps, wall));
        let mut moved = nuclei.clone();
        moved[0].pos[0] += 0.01;
        let t = Instant::now();
        let (_, st) = s.solve(&System::new(moved.clone(), 0, None), &settings, true).expect("warm solve converges");
        let warm = (st.iterations, ms(t.elapsed()));
        line("warm (+0.01 Bohr)", &mut s, &mut view, &moved, warm);
    }
}

/// The scene box of an app preset at spacing h: n = the smallest odd n with (n − 1)·h ≥ box
/// (`app::quality::grid_n`, private to the `app` binary crate: the same rule here).
fn grid_n_for(box_len: f64, h: f64) -> usize {
    let k = (box_len / h * (1.0 - 1e-9)).ceil().max(2.0) as usize;
    (if k % 2 == 1 { k + 1 } else { k }) + 1
}

/// The spin view's noise verdict (`SpinSummary::is_noise`, derivation in `qm::orbitals`) at the
/// app's Fast, Accurate (default) and Precise grids, tol 1e-5 as the app runs, cold and over 8 warm
/// MD-like steps with density extrapolation: the closed shells H₂ and LiH are noise at every grid
/// and every step (so the view draws nothing and says "no net spin"); a lone Li (∫m = 1) and the
/// H₂ triplet (∫m = 2) never are. The margins are printed: the closed shell's ∫|m| against its
/// budget, the real spin's against its budget.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn spin_view_hides_closed_shell_noise_and_keeps_real_spin() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let h2 = [nucleus([0.725, 0.0, 0.0], 1), nucleus([-0.725, 0.0, 0.0], 1)];
    let lih = [nucleus([-0.8, 0.0, 0.0], 3), nucleus([2.2, 0.0, 0.0], 1)];
    let li = [nucleus([0.0; 3], 3)];
    // (name, box length, nuclei, multiplicity, ∫m expected: 0 = closed shell)
    let cases: [(&str, f64, &[Nucleus], Option<u32>, f64); 4] =
        [("H₂", 12.0, &h2, None, 0.0), ("LiH", 14.0, &lih, None, 0.0), ("Li", 12.0, &li, None, 1.0), ("H₂ triplet", 12.0, &h2, Some(3), 2.0)];
    for (name, box_len, nuclei, mult, net) in cases {
        for h in [0.30, 0.15, 0.10] {
            let n = grid_n_for(box_len, h);
            let (mut s, settings, _, _) = solve(&ctx, n, h, nuclei, INTERACTIVE_TOL, mult);
            s.set_density_extrapolation(true);
            let mut view = OrbitalView::new(&s);
            let mut moved = nuclei.to_vec();
            let (mut worst, mut least_budget) = (0.0f64, f64::INFINITY);
            for step in 0..9 {
                if step > 0 {
                    moved[0].pos[0] += 0.004 * (step as f64 * 0.3).sin() + 0.002;
                    s.solve(&System::new(moved.clone(), 0, mult), &settings, true).expect("warm solve converges");
                }
                let state = s.state().expect("converged");
                let sp = view.update(&s, state, &moved, settings.tol, Product::None).spin;
                let eig = s.profile().eig_tol_used;
                let budget = sp.noise_l1(settings.tol, eig);
                worst = worst.max(sp.abs_integral);
                least_budget = least_budget.min(budget);
                if net == 0.0 {
                    assert!(sp.is_noise(settings.tol, eig), "{name} {n}³ h {h} step {step}: closed-shell ∫|m| = {:.2e} exceeds the budget {budget:.2e}", sp.abs_integral);
                } else {
                    assert!(!sp.is_noise(settings.tol, eig), "{name} {n}³ h {h} step {step}: real spin ∫|m| = {:.4} was judged noise (budget {budget:.2e})", sp.abs_integral);
                    assert!((sp.integral - net).abs() < 1e-3, "{name} {n}³ h {h}: ∫m = {} (expected {net})", sp.integral);
                }
            }
            println!("{name:>10} {n:>3}³ h {h:.2}: {} ∫|m| {worst:.2e} vs budget ≥ {least_budget:.2e} ({:.1}×)", if net == 0.0 { "closed shell: noise," } else { "real spin, not noise," },
                     if net == 0.0 { least_budget / worst.max(f64::MIN_POSITIVE) } else { worst / least_budget });
        }
    }
}

/// Measurement: the closed-shell spin noise over a 60-step warm trajectory (density extrapolation
/// on, as the app runs it), per quality: the tail of ∫|m| / tol and of max|m| / max ρ.
/// `cargo test -p validate --release --test orbitals -- --ignored --nocapture diag_spin_noise_tail`
#[test]
#[ignore = "measurement: tail of the closed-shell spin noise over an MD-like warm trajectory"]
fn diag_spin_noise_tail() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let h2 = [nucleus([0.725, 0.0, 0.0], 1), nucleus([-0.725, 0.0, 0.0], 1)];
    let lih = [nucleus([-0.8, 0.0, 0.0], 3), nucleus([2.2, 0.0, 0.0], 1)];
    for (name, box_len, nuclei) in [("H₂", 12.0, &h2[..]), ("LiH", 14.0, &lih[..])] {
        for h in [0.30, 0.25, 0.15, 0.10] {
            let n = grid_n_for(box_len, h);
            let (mut s, settings, _, _) = solve(&ctx, n, h, nuclei, INTERACTIVE_TOL, None);
            s.set_density_extrapolation(true);
            let mut view = OrbitalView::new(&s);
            let mut moved = nuclei.to_vec();
            let (mut worst_l1, mut worst_ratio, mut worst_max) = (0.0f64, 0.0f64, 0.0f64);
            for step in 0..60 {
                moved[0].pos[0] += 0.004 * (step as f64 * 0.3).sin() + 0.002;
                s.solve(&System::new(moved.clone(), 0, None), &settings, true).expect("warm solve converges");
                let state = s.state().expect("converged");
                let sp = view.update(&s, state, &moved, settings.tol, Product::None).spin;
                worst_l1 = worst_l1.max(sp.abs_integral);
                worst_ratio = worst_ratio.max(sp.max_abs / sp.rho_max);
                worst_max = worst_max.max(sp.max_abs);
            }
            println!("{name:>4} {n:>3}³ h {h:.2}: worst ∫|m| {worst_l1:.2e} ({:.2}× tol), worst max|m|/max ρ {worst_ratio:.2e}, worst max|m| {worst_max:.2e}", worst_l1 / INTERACTIVE_TOL);
        }
    }
}

/// Measurement: the spin density of closed and open shells at the app's quality grids.
/// `cargo test -p validate --release --test orbitals -- --ignored --nocapture diag_spin_noise`
#[test]
#[ignore = "measurement: spin-density noise of closed shells vs real spin, per quality"]
fn diag_spin_noise_by_quality() {
    let Some(ctx) = gpu() else { return };
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let h2 = [nucleus([0.725, 0.0, 0.0], 1), nucleus([-0.725, 0.0, 0.0], 1)];
    let lih = [nucleus([-0.8, 0.0, 0.0], 3), nucleus([2.2, 0.0, 0.0], 1)];
    let li = [nucleus([0.0; 3], 3)];
    let cases: [(&str, f64, &[Nucleus], Option<u32>); 4] = [("H₂", 12.0, &h2, None), ("LiH", 14.0, &lih, None), ("Li", 12.0, &li, None), ("H₂ triplet", 12.0, &h2, Some(3))];
    for (name, box_len, nuclei, mult) in cases {
        for h in [0.30, 0.25, 0.15, 0.10] {
            let n = grid_n_for(box_len, h);
            for tol in [INTERACTIVE_TOL] {
                let (mut s, settings, steps, wall) = solve(&ctx, n, h, nuclei, tol, mult);
                let report = |what: &str, s: &GpuSolver, steps: usize| {
                    let state = s.state().expect("converged");
                    let mut view = OrbitalView::new(s);
                    let sp = view.update(s, state, nuclei, settings.tol, Product::None).spin;
                    println!("{name:>10} {n:>3}³ h {h:.2} {what:>5} tol {tol:.0e}: {steps:>3} steps, eig tol {:.1e}: max|m|/max ρ {:.2e}, max ρ {:.3}, max|m| {:.2e}, ∫|m| {:.2e}, ∫m {:+.3}",
                             s.profile().eig_tol_used, sp.max_abs / sp.rho_max, sp.rho_max, sp.max_abs, sp.abs_integral, sp.integral);
                };
                report("cold", &s, steps);
                let mut moved = nuclei.to_vec();
                for step in 1..=3 {
                    moved[0].pos[0] += 0.01;
                    let (_, st) = s.solve(&System::new(moved.clone(), 0, mult), &settings, true).expect("warm solve converges");
                    report(&format!("warm{step}"), &s, st.iterations);
                }
                let _ = wall;
            }
        }
    }
}
