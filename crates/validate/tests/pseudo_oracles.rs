//! M7 Task 10: the validation oracles of the self-generated pseudopotentials on the GPU solver (spec §6.3–§6.7, §8
//! row 10). Atomic units inside (Hartree, Bohr, Ha/Bohr; Å and degrees only in the literature constants and the
//! printout). Release-only: `cargo test -p validate --release --test pseudo_oracles` (skips cleanly without a GPU
//! adapter; `--nocapture` prints every number; each test appends its timing row to the gitignored
//! `last-run-pseudo.md`). Numbers of record: `references.md` "M7 oracles (Task 10)". Task 11 extends this file.
//!
//! # Protocol
//! - GPU: `common::Locked` (`GpuKs` behind the suite's GPU lock, side solvers off), SCF tol 1e-6 (the floor-aware stop
//!   `providers::pseudo::effective_gpu_tol` keeps it for N ≤ 10 valence electrons; O₂'s N = 12 runs at 1.2e-6).
//! - The tables are the providers' cache (`providers::pseudo::table_for`, the ones the solves use — plan review M6;
//!   from Task 11 the dimer-gated H). Recipe `m7-tm-kb-nlcc-4` (c = ¼; NLCC ruling: the geometry oracles move by
//!   δθ ≈ 0.03°, δr ≈ 1e-4 Bohr against -3, far inside the bounds).
//! - Grids: h = 0.25 when every species' h_req allows it, else the finest h_req (`h_for`; O and N: 0.20), plus
//!   h = 0.20 reported where that differs (CH₄); a 14 Bohr cube with a grid point at the origin (`pseudo_scene`).
//! - Relaxation: the quenched velocity Verlet of `common::relax` (the M6 relax tool's quench, spec §6.4), Δt 0.5 fs, run
//!   to max_I |F_I| < `ORACLE_FTOL` = 2e-4 Ha/Bohr (at most `ORACLE_MAX_STEPS`) instead of the suite-wide 1e-3, because
//!   the angle a residual force leaves is a third of the H₂O bound and more than the CH₄ one (derivation at the
//!   constant). The geometry is cached per (molecule, policy, h) for the binary, so the egg-box record probes the same
//!   relaxed water the angle test measured.
//! - **Spec question 8** (plan review, accepted): `All` is usable from Task 6, so the geometries are gated under `All`
//!   with H at its atomic-gate f (the provisional H table; Task 11's dimer gate may move it and re-runs these).
//!   `CoresOnly` (pseudo-O/N/C beside an all-electron H smeared at σ = h) is recorded, not gated: its miss is the
//!   measured cost of the remaining smearing (spec §6.4, review S8).
//!
//! # Literature (Step 1; `references.md` "M7 oracles (Task 10)" has the access record)
//! NIST CCCBDB, Release 22 (May 2022), "Calculated geometries", method **LSDA**, basis **aug-cc-pVQZ** (the largest of
//! the aug-cc-pVTZ/QZ class listed; a grid solver is at the basis-set limit). CCCBDB's glossary: "In the Gaussian
//! programs 'LSDA' is equivalent to the 'SVWN' keyword", and Gaussian 16's manual: SVWN's VWN is "functional (III)
//! fitting the RPA solution" — so these are **SVWN = Slater + VWN-RPA (VWN3)**, not VWN5. This solver's LSDA is PW92
//! (≈ VWN5); the VWN3/VWN5 geometry difference (~1e-3 Å, ~0.1°) is far inside ±0.02 Å / ±2°. Reported alongside, not
//! pinned: CCCBDB LSDA/aug-cc-pVTZ and LSDA/6-31G* (the method and basis of Johnson, Gill & Pople, *J. Chem. Phys.* 98,
//! 5612 (1993) — whose own table could not be read, see references.md; CCCBDB's 6-31G* row stands in for it).
//!
//! # Tests
//! - `h2o_angle_within_2deg_of_lsda` (spec §8 row 10's first failing test): water relaxed under `All` within ±2° and
//!   ±0.02 Å of SVWN/aug-cc-pVQZ; `CoresOnly` recorded. Control: `h2o_under_none_is_the_smeared_177` (ignored).
//! - `nh3_angle_within_2deg_of_lsda`, `ch4_angle_is_tetrahedral_within_half_degree` (spec §6.4).
//! - `ne_h_convergence_is_monotone` (spec §6.3): pseudo-Ne over the whole ladder 0.35 … 0.15 through
//!   `IonicSetup::from_tables` (Ne is certified only at 0.15, so the brief's filter h ≤ h_req would leave one point
//!   and test nothing; spec §6.3 names the five spacings and the `from_tables` route), energies re-evaluated in f64.
//! - `h2o_cores_only_eggbox_is_recorded` (spec §6.6): the relaxed `CoresOnly` water's rigid-shift probe, f64-energy
//!   instrument (`eggbox`), wall-subtracted, at O's h_req 0.20 and the app default 0.15, "per the outcome" (NLCC ruling):
//!   PASS asserted where recorded, the ratchet where a MISS is recorded, report only before a record.
//!   `nh3_ch4_eggbox_records` (ignored): the C and N molecule records under `All` (Task 6 review: records, not gates),
//!   with 0.15 and, where the app allows Fast, 0.30.
//! - `pseudo_o2_nve_drift_is_bounded`, `pseudo_o2_csvr_reaches_target` (spec §6.7): O₂ under `CoresOnly` (all-pseudo),
//!   triplet, at the dynamics suite's bounds; the momentum budget from the recorded O₂ egg-box (NLCC ruling "Task 10";
//!   task-8b review "What Task 10 must carry"), the COM correction on under the thermostat.
//!
//! The GPU work of concurrently running tests is serialised through `common::GPU` (per evaluation; the egg-box
//! instrument per solve), so tests interleave and the GPU times in the rows are uncontended.
mod common;
mod eggbox;

use common::*;
use gpu::GpuContext;
use sim::{Cell, ForceProvider, IonicPolicy, Nucleus, Scene};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// NIST CCCBDB (Release 22), method LSDA = Gaussian SVWN (Slater + VWN-RPA, "VWN3"), basis aug-cc-pVQZ: geometry of H₂O,
/// r(O–H) = 0.9690 Å (accessed 2026-10-18; references.md "M7 oracles (Task 10)"): the basis-set-limit LSDA water.
const LSDA_H2O_ROH_ANGSTROM: f64 = 0.9690;
/// Same source and row: ∠HOH = 104.994°.
const LSDA_H2O_ANGLE_DEG: f64 = 104.994;
/// Same source, NH₃, LSDA/aug-cc-pVQZ: ∠HNH = 107.315° (r(N–H) 1.0200 Å, reported).
const LSDA_NH3_ANGLE_DEG: f64 = 107.315;
/// CH₄: the tetrahedral angle acos(−1/3) — a symmetry, functional-independent (spec §6.4).
const TETRAHEDRAL_DEG: f64 = 109.471_220_634_490_7;
/// Reported alongside, not pinned (same source, Å / degrees): (row, r(X–H), ∠). The LSDA/6-31G* row is the method and
/// basis of Johnson, Gill & Pople 1993 (review S8), as computed by CCCBDB.
const REPORTED_H2O: &[(&str, f64, f64)] = &[("SVWN/aug-cc-pVTZ", 0.9703, 104.954), ("SVWN/6-31G*", 0.9747, 103.730)];
const REPORTED_NH3: &[(&str, f64, f64)] = &[("SVWN/aug-cc-pVQZ", 1.0200, 107.315), ("SVWN/aug-cc-pVTZ", 1.0209, 107.292), ("SVWN/6-31G*", 1.0251, 106.154)];
const REPORTED_CH4_RCH_ANGSTROM: f64 = 1.0947; // SVWN/aug-cc-pVQZ

/// The relaxation's force criterion here (Ha/Bohr, per-atom vector norm as `common::relax`). The suite-wide 1e-3
/// (`RELAX_FTOL`) leaves a bend error: moving one H by dx ⊥ O–H changes θ by dx/r, so a residual F ⊥ O–H on it is
/// k_θ·δθ/r and δθ = F·r/k_θ. For water k_θ = ω_b²/G_θθ with ω_b = 1595 cm⁻¹ (the bend; an estimate used only to size the
/// criterion) and G_θθ ≈ 2/(m_H r²) + (2 − 2cos θ)/(m_O r²) = 3.6e-4 a.u. at r = 1.81 Bohr: k_θ ≈ 0.15 Ha/rad², and
/// F = 1e-3 Ha/Bohr leaves δθ ≈ 0.012 rad = 0.7° — a third of the H₂O bound and more than the CH₄ bound (±0.5°; its
/// bends are as stiff). 2e-4 brings that to ≈ 0.14°
/// while staying 4× above the GPU pseudo path's force noise (≈ 5e-5 per component, warm chain; task-8b review ruling (b))
/// and the egg-box (O₂ G ≤ 4.3e-5 at 0.20). Reported: the final max|F|.
const ORACLE_FTOL: f64 = 2e-4;
/// At most this many quench steps (the H₂O relax under `None` took 74 to 1e-3, M4).
const ORACLE_MAX_STEPS: usize = 600;
/// The molecules' cube (Bohr): spec §6.4/§6.6's 14 Bohr.
const BOX: f64 = 14.0;
/// The SCF tolerance of the geometry and atom oracles (spec §6.4; the dynamics/surrogate suites' TOL_H2).
const TOL: f64 = 1e-6;

/// The recorded O₂ egg-box at O's h_req (references.md "M7 forces", `pseudo.rs` `EGGBOX_RECORD`: G_rec = 4.262e-5 Ha/Bohr
/// at 0.20, `CoresOnly`, 14 Bohr, f64-energy instrument) and the GPU state's net-force error per component (u_HF of
/// `o2_gpu_hf_sum_force_qualification`, 2.40e-5, of which the kernels ≤ 3.8e-7: "ΣF ≠ 0 up to ~2.4e-5 per component from
/// the state", task-8b review). The momentum budget of the pseudo dynamics tests is built from these, never from 3e-5
/// (NLCC ruling, "Task 10").
const O2_G_REC: f64 = 4.262e-5;
const STATE_SUM_F_PER_COMPONENT: f64 = 2.4e-5;
/// The dynamics suite's bounds (`dynamics.rs`): |slope of E_total| ≤ 2e-4 Ha/ps over 2000 steps; CSVR T = 300 K, τ = 5 fs,
/// 2800 steps, settle 100, |⟨T⟩/T − 1| ≤ 0.2; seed 7.
const DRIFT_BOUND: f64 = 2e-4;
const NVE_STEPS: usize = 2000;
const T_TARGET: f64 = 300.0;
const THERMO_TAU_FS: f64 = 5.0;
const THERMO_STEPS: usize = 2800;
const THERMO_SETTLE_STEPS: usize = 100;
const T_BOUND: f64 = 0.2;
const SEED: u64 = 7;

/// The §6.6 records of this file (molecule label, h, G_rec, u_I,rec, PASS achieved), measured by Task 10 on recipe
/// `m7-tm-kb-nlcc-4` (references.md "M7 oracles (Task 10)"); the relaxed `CoresOnly` water at O's h_req and 0.15
/// (run of record 2026-10-18, RX 7900 XTX: MISS at 0.20, G 3.997e-5 — the preset geometry's is 3.033e-5, `pseudo.rs` —;
/// PASS at 0.15, G 4.428e-6). A version that lowers G re-records G_rec (spec §6.6 "Gate and fallback").
const EGGBOX_RECORD: &[(&str, f64, f64, f64, bool)] = &[
    ("h2o-cores-relaxed", 0.20, 3.997e-5, 9.65e-9, false), ("h2o-cores-relaxed", 0.15, 4.428e-6, 1.61e-8, true),
];

fn record_for(label: &str, h: f64) -> Option<(f64, f64, bool)> {
    EGGBOX_RECORD.iter().find(|r| r.0 == label && (r.1 - h).abs() < 1e-12).map(|r| (r.2, r.3, r.4))
}

// ---------------------------------------------------------------- the brief's helpers

/// The finest of `wanted` and every species' h_req, from the providers' cache: the table the solves themselves use
/// (plan review M6 — `qm::radial::generate` would return the ungated, provisional H).
fn h_for(zs: &[u8], wanted: f64) -> f64 {
    zs.iter().filter_map(|&z| providers::pseudo::table_for(z).ok()?.record.h_req).fold(wanted, f64::min)
}

/// A neutral scene of `nuclei` on the odd cube of ≈ `box_bohr` at spacing h (a grid point at the origin).
fn pseudo_scene(nuclei: Vec<Nucleus>, h: f64, box_bohr: f64, policy: IonicPolicy) -> Scene {
    let n = 2 * (box_bohr / (2.0 * h)).round() as usize + 1;
    Scene { nuclei, cell: Cell { n, h }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: policy }
}

/// The angle at nucleus `v` between the bonds to `a` and `b` (degrees).
fn angle_deg(s: &Scene, v: usize, a: usize, b: usize) -> f64 {
    let d = |i: usize| [0, 1, 2].map(|k| s.nuclei[i].pos[k] - s.nuclei[v].pos[k]);
    let (x, y) = (d(a), d(b));
    let dot: f64 = (0..3).map(|k| x[k] * y[k]).sum();
    let n = |u: [f64; 3]| u.iter().map(|c| c * c).sum::<f64>().sqrt();
    (dot / (n(x) * n(y))).clamp(-1.0, 1.0).acos().to_degrees()
}

fn dist_angstrom(s: &Scene, i: usize, j: usize) -> f64 {
    units::bohr_to_angstrom((0..3).map(|k| (s.nuclei[i].pos[k] - s.nuclei[j].pos[k]).powi(2)).sum::<f64>().sqrt())
}

/// A start away from the answer (112°, 1.85 Bohr), so the relax has to find the minimum.
fn water(policy: IonicPolicy, h: f64) -> Scene {
    let (r, half) = (1.85, 56f64.to_radians());
    pseudo_scene(vec![Nucleus::new(8, [0.0, 0.0, 0.0]), Nucleus::new(1, [r * half.sin(), r * half.cos(), 0.0]),
                      Nucleus::new(1, [-r * half.sin(), r * half.cos(), 0.0])], h, BOX, policy)
}

/// NH₃ from a flattened pyramid, ∠HNH = 114° and r(N–H) = 1.85 Bohr (LSDA: 107.3°, 1.93 Bohr), N at the origin and the
/// C₃ axis along z. With β the H's angle from the axis, cos ∠HNH = 1 − (3/2)sin²β. (The brief's start — r = 1.95, the
/// z-height scaled by 0.6 — is at 106.8° with r = 1.74 Bohr, i.e. already at the answer's angle; it would not show that
/// the relax finds the angle.)
fn ammonia(policy: IonicPolicy, h: f64) -> Scene {
    let (r, alpha) = (1.85, 114f64.to_radians());
    let beta = ((2.0 / 3.0) * (1.0 - alpha.cos())).sqrt().asin();
    let mut nuc = vec![Nucleus::new(7, [0.0; 3])];
    nuc.extend((0..3).map(|k| { let phi = k as f64 * 2.0 * std::f64::consts::PI / 3.0;
        Nucleus::new(1, [r * beta.sin() * phi.cos(), r * beta.sin() * phi.sin(), -r * beta.cos()]) }));
    pseudo_scene(nuc, h, BOX, policy)
}

/// CH₄ from a distorted tetrahedron (C–H 2.06 Bohr before a 3–5 % stretch of one coordinate of three H).
fn methane(policy: IonicPolicy, h: f64) -> Scene {
    let a = 2.06 / 3f64.sqrt();
    pseudo_scene(vec![Nucleus::new(6, [0.0; 3]), Nucleus::new(1, [a * 1.05, a, a]), Nucleus::new(1, [-a, -a * 0.97, a]),
                      Nucleus::new(1, [-a, a, -a * 1.03]), Nucleus::new(1, [a, -a, -a])], h, BOX, policy)
}

// ---------------------------------------------------------------- relaxation, cached per binary

/// A relaxed geometry and how it was reached (`log`: the relax's lines, printed when it was made).
#[allow(dead_code)]
struct Relaxed { scene: Scene, steps: usize, max_f: f64, energy: f64, start_angle: Option<f64>, log: String, eval: EvalLog, wall: Duration }

/// `common::relax_with` (quenched velocity Verlet: velocities zeroed whenever Σ F·v < 0; Δt `RELAX_DT_FS`) to `ftol`
/// in at most `max_steps`. Returns the relaxed scene at rest, its energy, the final max|F| and the steps.
fn quench(p: &mut dyn ForceProvider, mut s: Scene, ftol: f64, max_steps: usize, log: &mut String) -> (Scene, f64, f64, usize) {
    let integ = sim::Integrator::new_fs(RELAX_DT_FS);
    let maxf = |f: &[[f64; 3]]| sim::relax::max_norm(f);
    let r = solve(p, &s, "relax start");
    let mut f = r.forces;
    let _ = writeln!(log, "  relax start: E = {:+.6} Ha, max|F| = {:.2e}", r.energy, maxf(&f));
    let (mut steps, mut quenches, mut confirmations) = (0, 0, 0);
    // the stop is confirmed by a solve at rest at the final geometry; the GPU's warm-path force scatter (≈ 5e-5 per
    // component, task-8b review) can put that solve back above ftol (first run: 2.26e-4 after a step's < 2e-4), and then
    // the quench simply continues from it, at rest, within the same step budget
    let r = loop {
        while maxf(&f) >= ftol && steps < max_steps {
            let (f_new, st) = integ.step(&mut s, &f, p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("relax step {steps}: {e}"));
            f = f_new;
            steps += 1;
            let power: f64 = s.nuclei.iter().zip(&f).map(|(n, f)| n.vel[0] * f[0] + n.vel[1] * f[1] + n.vel[2] * f[2]).sum();
            if power < 0.0 {
                for n in &mut s.nuclei { n.vel = [0.0; 3]; }
                quenches += 1;
            }
            if steps % 25 == 0 { let _ = writeln!(log, "  relax {steps}: E_KS = {:+.6}, max|F| = {:.2e}, {quenches} quenches", st.energy_ks, maxf(&f)); }
        }
        for n in &mut s.nuclei { n.vel = [0.0; 3]; }
        let r = solve(p, &s, "relaxed");
        confirmations += 1;
        if maxf(&r.forces) < ftol || steps >= max_steps { break r; }
        let _ = writeln!(log, "  confirmation solve at step {steps}: max|F| = {:.2e} ≥ {ftol:.0e} (the step's own {:.2e}) — the quench continues at rest", maxf(&r.forces), maxf(&f));
        f = r.forces;
    };
    let _ = writeln!(log, "  relaxed after {steps} steps ({quenches} quenches, {confirmations} confirmation solves): E = {:+.6} Ha, max|F| = {:.2e} (target < {ftol:.0e})", r.energy, maxf(&r.forces));
    assert!(maxf(&r.forces) < ftol, "relaxation did not converge in {max_steps} steps: max|F| = {:.2e}\n{log}", maxf(&r.forces));
    (s, r.energy, maxf(&r.forces), steps)
}

/// The relaxed geometry of `key` (molecule/policy/h), relaxed once per binary from `start` on a fresh `Locked` at `TOL`.
fn relaxed(ctx: &GpuContext, key: &str, start: impl FnOnce() -> Scene) -> Arc<Relaxed> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<OnceLock<Arc<Relaxed>>>>>> = OnceLock::new();
    let cell = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()).entry(key.to_string()).or_default().clone();
    cell.get_or_init(|| {
        let t0 = Instant::now();
        let (mut p, eval) = Locked::new(ctx.clone(), TOL);
        let s0 = start();
        let start_angle = (s0.nuclei.len() >= 3).then(|| angle_deg(&s0, 0, 1, 2));
        let mut log = format!("{key}: {}³ × {}, start ∠(1,0,2) {:?}°\n", s0.cell.n, s0.cell.h, start_angle);
        let (scene, energy, max_f, steps) = quench(&mut p, s0, ORACLE_FTOL, ORACLE_MAX_STEPS, &mut log);
        let eval = eval.lock().unwrap().clone();
        let _ = writeln!(log, "  {}", eval.summary());
        eprintln!("{log}");
        Arc::new(Relaxed { scene, steps, max_f, energy, start_angle, log, eval, wall: t0.elapsed() })
    }).clone()
}

/// No solve of a gated geometry may carry the box warning (a truncated density biases the geometry).
fn assert_no_box_warning(what: &str, r: &Relaxed) {
    assert!(r.eval.warnings.iter().all(|w| !w.contains("box too small")), "{what}: {:?}", r.eval.warnings);
}

// ---------------------------------------------------------------- timing rows

static TIMING: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Appends a row and rewrites `last-run-pseudo.md`.
fn record(ctx: &GpuContext, row: String) {
    let mut rows = TIMING.lock().unwrap_or_else(|e| e.into_inner());
    rows.push(row);
    let adapter = ctx.adapter.get_info();
    let mut md = String::new();
    let _ = writeln!(md, "# Pseudo oracles timing (`validate::pseudo_oracles`, {} UTC)\n", chrono_free_date());
    let _ = writeln!(md, "Adapter: {} ({:?}, {:?}); release build. Tests run concurrently with their GPU work serialised, so \"GPU\" is the uncontended sum of this test's solves and \"wall\" includes waiting for the others. Reported, never asserted.\n", adapter.name, adapter.backend, adapter.device_type);
    let _ = writeln!(md, "| test | system | grid | tol | solves | GPU s | ms/solve | SCF steps/solve | readbacks/solve | test wall s | result |\n|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rows.iter() { md += r; md.push('\n'); }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-pseudo.md");
    std::fs::write(path, &md).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}

/// A timing row for a cached relaxation.
fn relax_row(ctx: &GpuContext, test: &str, system: &str, r: &Relaxed, result: &str) {
    record(ctx, timing_row(test, system, r.scene.cell.n, TOL, &r.eval, r.wall, &format!("{result}; {} relax steps, max|F| {:.1e}", r.steps, r.max_f)));
}

// ---------------------------------------------------------------- geometries (spec §6.4)

/// Spec §8 row 10 first failing test (§6.4; review S8): relaxed water under `All` — ∠HOH within ±2° and r(O–H) within
/// ±0.02 Å of the basis-set-limit LSDA geometry (SVWN/aug-cc-pVQZ); the `CoresOnly` numbers (σ = h H beside pseudo-O)
/// recorded, not gated: their miss is the measured cost of the remaining smearing. The all-electron smeared surface
/// relaxes to ≈ 177° (`h2o_under_none_is_the_smeared_177`). H's table is the atomic-gate one (Spec question 8).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_angle_within_2deg_of_lsda() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let h = h_for(&[8, 1], 0.25);
    let all = relaxed(&ctx, &format!("h2o/all/{h}"), || water(IonicPolicy::All, h));
    let core = relaxed(&ctx, &format!("h2o/cores/{h}"), || water(IonicPolicy::CoresOnly, h));
    let (ang, roh) = (angle_deg(&all.scene, 0, 1, 2), dist_angstrom(&all.scene, 0, 1));
    let (ang_c, roh_c) = (angle_deg(&core.scene, 0, 1, 2), dist_angstrom(&core.scene, 0, 1));
    let h_table = providers::pseudo::table_for(1).expect("H has a table");
    eprintln!("H₂O at h = {h} ({}³, {BOX} Bohr): All ∠HOH {ang:.3}°, r(O–H) {roh:.4} / {:.4} Å, E {:.6} Ha ({} relax steps, max|F| {:.1e}); CoresOnly {ang_c:.3}°, {roh_c:.4} / {:.4} Å, E {:.6} Ha ({} steps; recorded); \
               LSDA (SVWN/aug-cc-pVQZ) {LSDA_H2O_ANGLE_DEG}°, {LSDA_H2O_ROH_ANGSTROM} Å → All Δ {:+.3}°, {:+.4} Å; CoresOnly Δ {:+.3}°, {:+.4} Å; reported alongside {REPORTED_H2O:?}; H table f = {} {} (Spec question 8: atomic-gate H)",
              all.scene.cell.n, dist_angstrom(&all.scene, 0, 2), all.energy, all.steps, all.max_f, dist_angstrom(&core.scene, 0, 2), core.energy, core.steps,
              ang - LSDA_H2O_ANGLE_DEG, roh - LSDA_H2O_ROH_ANGSTROM, ang_c - LSDA_H2O_ANGLE_DEG, roh_c - LSDA_H2O_ROH_ANGSTROM, h_table.f, h_table.record.local);
    relax_row(&ctx, "h2o_angle (All)", "H₂O", &all, &format!("∠HOH {ang:.2}°, r(O–H) {roh:.4} Å"));
    relax_row(&ctx, "h2o_angle (CoresOnly, recorded)", "H₂O", &core, &format!("∠HOH {ang_c:.2}°, r(O–H) {roh_c:.4} Å"));
    eprintln!("h2o_angle_within_2deg_of_lsda: {:.0} s", t0.elapsed().as_secs_f64());
    assert_no_box_warning("H₂O All", &all);
    assert!((ang - LSDA_H2O_ANGLE_DEG).abs() <= 2.0, "∠HOH {ang:.2}° vs {LSDA_H2O_ANGLE_DEG}°");
    assert!((roh - LSDA_H2O_ROH_ANGSTROM).abs() <= 0.02, "r(O–H) {roh:.4} Å vs {LSDA_H2O_ROH_ANGSTROM} Å");
    assert!((dist_angstrom(&all.scene, 0, 2) - LSDA_H2O_ROH_ANGSTROM).abs() <= 0.02, "r(O–H') {:.4} Å", dist_angstrom(&all.scene, 0, 2));
}

/// The discriminating control: the same start relaxed under `None` (σ = h = 0.25, all-electron O) lands on the smeared
/// near-linear minimum (spec §1.2: 177.2°; M4: 177° at 57³) — `h2o_angle_within_2deg_of_lsda` would fail there.
/// Ignored (minutes; a record). Relaxed with `common::relax` (1e-3: the control needs no finer angle).
#[test]
#[ignore = "control for h2o_angle_within_2deg_of_lsda: the all-electron σ = 0.25 surface is near-linear (spec §1.2)"]
fn h2o_under_none_is_the_smeared_177() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let (mut p, eval) = Locked::new(ctx.clone(), TOL);
    let mut log = String::new();
    let (s, _, steps) = relax(&mut p, water(IonicPolicy::None, 0.25), &mut log);
    let ang = angle_deg(&s, 0, 1, 2);
    eprintln!("{log}H₂O under None (σ = h = 0.25, {}³): ∠HOH {ang:.2}°, r(O–H) {:.4} Å after {steps} relax steps", s.cell.n, dist_angstrom(&s, 0, 1));
    let eval = eval.lock().unwrap().clone();
    record(&ctx, timing_row("h2o_under_none (control)", "H₂O", s.cell.n, TOL, &eval, t0.elapsed(), &format!("∠HOH {ang:.2}°")));
    assert!(ang > 170.0, "∠HOH {ang:.2}°: the smeared surface is near-linear");
}

/// Spec §6.4: NH₃ ∠HNH within ±2° of the LSDA value (SVWN/aug-cc-pVQZ, 107.315°), under `All`, from a flattened start.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn nh3_angle_within_2deg_of_lsda() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[7, 1], 0.25);
    let r = relaxed(&ctx, &format!("nh3/all/{h}"), || ammonia(IonicPolicy::All, h));
    let s = &r.scene;
    let angles = [angle_deg(s, 0, 1, 2), angle_deg(s, 0, 2, 3), angle_deg(s, 0, 1, 3)];
    let lengths = [dist_angstrom(s, 0, 1), dist_angstrom(s, 0, 2), dist_angstrom(s, 0, 3)];
    eprintln!("NH₃ (All) at h = {h} ({}³): ∠HNH {angles:.3?}° (start {:.1?}°), r(N–H) {lengths:.4?} Å, E {:.6} Ha, {} relax steps, max|F| {:.1e}; LSDA {LSDA_NH3_ANGLE_DEG}° → Δ {:+.3?}; reported {REPORTED_NH3:?}",
              s.cell.n, r.start_angle, r.energy, r.steps, r.max_f, angles.map(|a| a - LSDA_NH3_ANGLE_DEG));
    relax_row(&ctx, "nh3_angle (All)", "NH₃", &r, &format!("∠HNH {:.2}/{:.2}/{:.2}°", angles[0], angles[1], angles[2]));
    assert_no_box_warning("NH₃ All", &r);
    for a in angles { assert!((a - LSDA_NH3_ANGLE_DEG).abs() <= 2.0, "∠HNH {a:.2}° vs {LSDA_NH3_ANGLE_DEG}°"); }
}

/// Spec §6.4: CH₄ relaxed from a distorted start under `All` — every ∠HCH within ±0.5° of 109.47° at h = 0.25 (C's
/// h_req 0.30 and H's 0.35 allow it); h = 0.20 reported (spec §6.4 "and h = 0.20 reported").
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn ch4_angle_is_tetrahedral_within_half_degree() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[6, 1], 0.25);
    let pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
    let mut gated = Vec::new();
    for (hh, gate) in [(h, true), (0.20, false)] {
        if !gate && (hh - h).abs() < 1e-12 { continue; }
        let r = relaxed(&ctx, &format!("ch4/all/{hh}"), || methane(IonicPolicy::All, hh));
        let s = &r.scene;
        let angles: Vec<f64> = pairs.iter().map(|&(i, j)| angle_deg(s, 0, i, j)).collect();
        let lengths: Vec<f64> = (1..=4).map(|i| dist_angstrom(s, 0, i)).collect();
        let worst = angles.iter().map(|a| (a - TETRAHEDRAL_DEG).abs()).fold(0.0, f64::max);
        eprintln!("CH₄ (All) at h = {hh} ({}³, {}): ∠HCH {angles:.3?}° (worst |Δ| {worst:.3}°), r(C–H) {lengths:.4?} Å (SVWN/aug-cc-pVQZ {REPORTED_CH4_RCH_ANGSTROM} Å), E {:.6} Ha, {} relax steps, max|F| {:.1e}",
                  s.cell.n, if gate { "gated" } else { "reported" }, r.energy, r.steps, r.max_f);
        relax_row(&ctx, &format!("ch4_angle (All, {})", if gate { "gated" } else { "reported" }), "CH₄", &r, &format!("worst |∠HCH − 109.47°| {worst:.3}°"));
        if gate { assert_no_box_warning("CH₄ All", &r); gated = angles; }
    }
    for (k, (i, j)) in pairs.iter().enumerate() {
        assert!((gated[k] - TETRAHEDRAL_DEG).abs() <= 0.5, "∠H{i}CH{j} = {:.3}°", gated[k]);
    }
}

// ---------------------------------------------------------------- atoms (spec §6.3)

/// Spec §6.3: the h-convergence of pseudo-Ne on the GPU over the whole ladder (`LEDGER.h_ladder`, 0.35 … 0.15), 12 Bohr,
/// each rung on Ne's table filtered at that h through `IonicSetup::from_tables` (Ne is certified only at 0.15 — its
/// h_req — so the providers would refuse the coarser rungs; the spec names this route). Monotone: the successive
/// differences E(h_{k+1}) − E(h_k) all have one sign; that sign is reported, not assumed (plan review S5: the 8th-order
/// FD symbol underestimates q², pulling coarse-h energies down, while the per-h filter smooths V_SR, pushing them up —
/// the net sign is not derivable a priori). The energies are the f64 re-evaluation of each cold GPU state (`eggbox`
/// instrument: no f32 evaluation noise in a 1e-6 comparison); the GPU's own E printed beside them. Reported:
/// |E(0.25) − E(0.15)| (spec §6.3: "the honest statement of what h = 0.25 costs") and E_3D − E_radial at every rung
/// (the radial KB atom on the same filtered table; the < 2 mHa gate at 0.15 is the CPU's `pseudo_ne_3d_matches_radial`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn ne_h_convergence_is_monotone() {
    use qm::ionic::{Ionic, IonicPolicy as QmPolicy, IonicSetup};
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let t = providers::pseudo::table_for(10).expect("Ne is accepted");
    let mut es: Vec<(f64, f64)> = Vec::new();
    let mut log = EvalLog::default();
    let mut lines = String::new();
    for &h in qm::radial::LEDGER.h_ladder.iter() {
        let ft = Arc::new(qm::radial::filter::filtered(&t, h));
        let setup = Arc::new(IonicSetup::from_tables(QmPolicy::CoresOnly, h, h, BTreeMap::from([(10, Ionic::Pseudo(ft.clone()))])));
        let n = 2 * (12.0 / (2.0 * h)).round() as usize + 1;
        let g = qm::Grid::cube(n, h);
        let mut solver = { let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner()); qm_gpu::GpuSolver::new(ctx.clone(), &g) };
        let nuc = vec![qm::Nucleus { pos: [0.0; 3], z: 10 }];
        let (e_gpu, st, stats, dt) = eggbox::gpu_solve(&mut solver, setup.clone(), nuc.clone(), Some(1), false);
        let mut poisson = qm::poisson::PoissonSolver::new(&g);
        let (e, _) = eggbox::ks_energy_f64(&g, &eggbox::occupied(&g, &st), &st.v_h, &setup, &nuc, &mut poisson);
        let radial = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &qm::radial::kb::reference_config(&t.partition), qm::radial::SpinMode::Restricted).expect("the radial KB Ne").energy;
        (log.calls, log.gpu, log.scf_steps) = (log.calls + 1, log.gpu + dt, log.scf_steps + stats.iterations);
        let _ = writeln!(lines, "  h = {h} ({n}³): E_f64 {:.7} (GPU {e_gpu:.7}), radial KB {radial:.7}, E_3D − radial {:+.3} mHa; {} SCF steps, Δρ {:.1e}, warnings {:?}",
                         e.total, (e.total - radial) * 1e3, stats.iterations, stats.delta_rho, stats.warnings);
        es.push((h, e.total));
    }
    let diffs: Vec<f64> = es.windows(2).map(|w| w[1].1 - w[0].1).collect();
    // 1e-6: the SCF tolerance, as the original one-sided check allowed
    let sign = if diffs.iter().all(|&d| d <= 1e-6) { "downward (E falls as h refines)" } else if diffs.iter().all(|&d| d >= -1e-6) { "upward (E rises as h refines)" } else { "mixed" };
    let at = |h: f64| es.iter().find(|(x, _)| (x - h).abs() < 1e-12).map(|e| e.1).expect("a ladder rung");
    eprintln!("pseudo-Ne E(h), 12 Bohr, CoresOnly via from_tables (h_req {:?}):\n{lines}  successive differences [{}] Ha: {sign}; |E(0.25) − E(0.15)| = {:.3} mHa",
              t.record.h_req, diffs.iter().map(|d| format!("{d:+.3e}")).collect::<Vec<_>>().join(", "), (at(0.25) - at(0.15)).abs() * 1e3);
    record(&ctx, timing_row("ne_h_convergence", "Ne", 81, eggbox::GPU_TOL, &log, t0.elapsed(), &format!("{sign}; |E(0.25) − E(0.15)| {:.2} mHa", (at(0.25) - at(0.15)).abs() * 1e3)));
    assert_eq!(es.len(), 5, "the ladder");
    assert_ne!(sign, "mixed", "not monotone: {es:?}");
}

// ---------------------------------------------------------------- egg-box (spec §6.6)

/// `relaxed`'s water as the egg-box instrument's molecule: its nuclei shifted rigidly by s (qm form).
fn rigid(s: &Scene) -> impl Fn([f64; 3]) -> Vec<qm::Nucleus> + '_ {
    move |d| s.nuclei.iter().map(|n| qm::Nucleus { pos: [n.pos[0] + d[0], n.pos[1] + d[1], n.pos[2] + d[2]], z: n.z }).collect()
}

/// One §6.6 probe of a relaxed scene at spacing h (the providers' certified setup at that h, σ = h for an all-electron
/// species, the scene's multiplicity), or the providers' refusal at that h.
fn probe_at(ctx: &GpuContext, label: &str, base: &Scene, h: f64, mult: Option<u32>) -> Result<eggbox::Probe, String> {
    let at_h = pseudo_scene(base.nuclei.clone(), h, BOX, base.ionic);
    let mut notes = Vec::new();
    let setup = providers::pseudo::setup_for(&at_h, h, &mut notes).map_err(|e| e.to_string())?.ok_or("nothing pseudised")?;
    Ok(eggbox::probe(ctx, label, h, BOX, setup, &rigid(base), mult, record_for(label, h)))
}

fn probe_row(ctx: &GpuContext, test: &str, system: &str, p: &eggbox::Probe) {
    let log = EvalLog { calls: p.solves, gpu: p.gpu, scf_steps: p.scf_steps, ..Default::default() };
    record(ctx, timing_row(test, system, p.n, eggbox::GPU_TOL, &log, Duration::from_secs_f64(p.secs),
                           &format!("h {}: G {:.2e} (u_I {:.1e}, wall {:.1e}, raw {:.1e}, |ΣF_y| {:.1e}) {}", p.h, p.g, p.u, p.wall_max, p.raw_max, p.fy_max, p.verdict)));
}

/// Spec §6.6 (review M5): H₂O under `CoresOnly` keeps two smeared H's. Recorded against the smeared-H egg-box ≈ 5e-7 per
/// H (Task 6 review; the earlier ≤ 4e-5 was the wall term). The relaxed `CoresOnly` water of
/// `h2o_angle_within_2deg_of_lsda` (O's h_req 0.20), rigid-shift probe with the f64-energy instrument and its transfer
/// check (`eggbox`), wall-subtracted, at 0.20 and at the app default 0.15 (reported beside it, never moved as the gate's h;
/// Task 6 review C.3); the max |ΣF_y| is reported next to the all-electron figure of record −3.4e-4 (like for like 10×).
/// Per the outcome (NLCC ruling, "Task 10"): the instrument is asserted (u_I ≤ 5e-6, wall ≤ 3e-4); a PASS on record is a
/// hard PASS, a MISS on record the ratchet (`EGGBOX_RECORD`); before a record, report only. (Under `CoresOnly` the water
/// is not a §6.6 gate — the pseudopotential is not its only egg-box source; the preset-geometry water is `pseudo.rs`'s.)
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_cores_only_eggbox_is_recorded() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8, 1], 0.25);
    let base = relaxed(&ctx, &format!("h2o/cores/{h}"), || water(IonicPolicy::CoresOnly, h));
    eprintln!("H₂O CoresOnly egg-box at the relaxed geometry (∠HOH {:.3}°, r(O–H) {:.4} Å; relaxed at h = {h})", angle_deg(&base.scene, 0, 1, 2), dist_angstrom(&base.scene, 0, 1));
    for hp in [h, 0.15] {
        let p = probe_at(&ctx, "h2o-cores-relaxed", &base.scene, hp, Some(1)).unwrap_or_else(|e| panic!("H₂O CoresOnly at h = {hp}: {e}"));
        probe_row(&ctx, "h2o_cores_only_eggbox", "H₂O (CoresOnly)", &p);
        assert!(p.g.is_finite());
    }
}

/// The C and N molecule egg-box records (Task 6 review: "CH₄ and NH₃ … the C and N floors are unmeasured (Task 10). They
/// are records, not §6.6 gates"): NH₃ and CH₄ relaxed under `All` (the geometry oracles' molecules), probed at their
/// gate h, at 0.15 (the app default) and at 0.30 (Fast, which the app permits for C and H scenes; NH₃ is refused there,
/// N's h_req being 0.20 at this recipe — printed). Report only (`--ignored`, ≈ 30 min).
#[test]
#[ignore = "record (≈ 30 min): cargo test -p validate --release --test pseudo_oracles -- --ignored --nocapture nh3_ch4_eggbox_records"]
fn nh3_ch4_eggbox_records() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    for (label, zs, build) in [("nh3-all-relaxed", [7u8, 1], ammonia as fn(IonicPolicy, f64) -> Scene), ("ch4-all-relaxed", [6, 1], methane)] {
        let h = h_for(&zs, 0.25);
        let base = relaxed(&ctx, &format!("{}/all/{h}", &label[..3]), || build(IonicPolicy::All, h));
        for hp in [h, 0.15, 0.30] {
            match probe_at(&ctx, label, &base.scene, hp, Some(1)) {
                Ok(p) => probe_row(&ctx, "nh3_ch4_eggbox_records", label, &p),
                Err(e) => eprintln!("{label} at h = {hp}: refused by the providers ({e}) — not probed"),
            }
        }
    }
}

// ---------------------------------------------------------------- dynamics (spec §6.7)

/// Σ_I m_I v_I (a.u.).
fn momentum(s: &Scene) -> [f64; 3] {
    [0, 1, 2].map(|k| s.nuclei.iter().map(|n| n.mass * n.vel[k]).sum())
}

/// Spec §6.7: NVE drift on a pseudo system (O₂ under `CoresOnly` = all-pseudo, triplet, R = 2.34 Bohr from rest, 10 Bohr
/// box at O's h_req) at the dynamics suite's bound — |slope of E_total| ≤ 2e-4 Ha/ps (`dynamics.rs` DRIFT_BOUND) over
/// 2000 steps of 0.25 fs, COM correction off (the plain total is the invariant).
///
/// Momentum (NLCC ruling "Task 10"; task-8b review "What Task 10 must carry"): with no tools the only net force is
/// Σ_I F_I = egg-box + the GPU state's error, so |P(t) − P(0)| ≤ (G_rec + √3·u_state)·t with G_rec the recorded O₂
/// egg-box at 0.20 (4.262e-5; the per-probe maximum over rigid shifts) and u_state = 2.4e-5 per component (the GPU
/// state's ΣF error, `o2_gpu_hf_sum_force_qualification`) — a worst case (constant, aligned net force), asserted at every
/// step. Here the molecule is centred on a grid point and stays inversion-symmetric, so the egg-box part vanishes by
/// symmetry and what P measures is the f32 asymmetry of the state. The drift budget of the warm chain's first-order
/// force error (≈ 5e-5 per component, task-8b review) is printed beside the bound: aligned with v it would be
/// 2·5e-5·⟨|v_I|⟩ per atom and component.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_o2_nve_drift_is_bounded() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let h = h_for(&[8], 0.25);
    let mut s = pseudo_scene(vec![Nucleus::new(8, [-1.17, 0.0, 0.0]), Nucleus::new(8, [1.17, 0.0, 0.0])], h, 10.0, IonicPolicy::CoresOnly);
    s.multiplicity = Some(3);
    let n = s.cell.n;
    let (p, log) = Locked::new(ctx.clone(), TOL);
    let mut d = sim::Dynamics::new(s, Box::new(p), DT_FS, SEED);
    let s0 = d.snapshot();
    assert!(s0.last_error.is_none(), "{:?}", s0.last_error);
    let p_rate = O2_G_REC + 3f64.sqrt() * STATE_SUM_F_PER_COMPONENT;
    let (mut t, mut e, mut worst_p, mut speed) = (Vec::new(), Vec::new(), 0.0f64, 0.0f64);
    for i in 1..=NVE_STEPS {
        d.apply(sim::DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert!(snap.last_error.is_none(), "step {i}: {:?}", snap.last_error);
        assert_eq!(snap.energy_conserved.to_bits(), snap.total.to_bits(), "NVE with no tools: the invariant is the plain total");
        t.push(snap.sim_time / au_per_ps());
        e.push(snap.total);
        let pm = eggbox::norm(momentum(&snap.scene));
        worst_p = worst_p.max(pm / (p_rate * snap.sim_time));
        speed += snap.scene.nuclei.iter().map(|n| n.vel.iter().map(|v| v * v).sum::<f64>().sqrt()).sum::<f64>() / 2.0;
        assert!(pm <= p_rate * snap.sim_time, "step {i}: |P| {pm:.3e} exceeds the budget (G_rec + √3·u_state)·t = {:.3e}", p_rate * snap.sim_time);
    }
    let last = d.snapshot();
    let (slope, fluct, e_mean) = fit_line(&t, &e);
    let mean_speed = speed / NVE_STEPS as f64;
    let aligned = 2.0 * 3f64.sqrt() * 5e-5 * mean_speed * au_per_ps();
    let log = log.lock().unwrap().clone();
    eprintln!("pseudo-O₂ NVE (CoresOnly, M = 3, h = {h}, {n}³ = 10 Bohr, tol {TOL:.0e}, dt {DT_FS} fs, {NVE_STEPS} steps): drift {slope:+.3e} Ha/ps (driver {:+.3e}; bound {DRIFT_BOUND:.0e}); detrended peak-to-peak {fluct:.2e} Ha, mean E {e_mean:+.6}; \
               worst |P|/budget {worst_p:.2e} (budget rate {p_rate:.2e} Ha/Bohr); ⟨|v|⟩ {mean_speed:.2e} a.u. → an aligned 5e-5 first-order force error would drift {aligned:.1e} Ha/ps; final R {:.4} Bohr; {}; warnings {:?}; {:.0} s",
              last.drift_per_ps.unwrap_or(f64::NAN), bond(&last.scene), log.summary(), log.warnings, t0.elapsed().as_secs_f64());
    record(&ctx, timing_row("pseudo_o2_nve", "O₂ (CoresOnly)", n, TOL, &log, t0.elapsed(), &format!("drift {slope:+.2e} Ha/ps, |P|/budget ≤ {worst_p:.1e}")));
    assert!(slope.abs() <= DRIFT_BOUND, "drift {slope:+.3e} Ha/ps (bound {DRIFT_BOUND:.0e})");
}

/// Spec §6.7: CSVR on a pseudo system reaches its target at the dynamics suite's bounds (T = 300 K, τ = 5 fs, 2800 steps,
/// settle 100, |⟨T⟩/T − 1| ≤ 0.2, seed 7, tol 1e-5 — the interactive tolerance the thermostat test runs at) — O₂ under
/// `CoresOnly`, triplet, from rest at R = 2.28 Bohr; the COM correction on (NLCC ruling "Task 10": a thermostatted pseudo
/// run turns it on rather than budgeting the egg-box momentum). N_f = 3N − 3 = 3 (one vibration, two rotations).
///
/// What the 20 % bound means here (the dynamics suite's derivation, `dynamics.rs` module doc): the window mean of T is
/// governed by S = Σ_vib E_i/2 + Σ_rot K_j with S̄ = (n_v + n_r)k_BT/2 and Var S = (n_v/4 + n_r/2)(k_BT)², relaxed with
/// time constant τ, so σ_mean/T ≈ (√Var S/S̄)·√(2τ/T_w). For O₂ (n_v = 1, n_r = 2) √Var S/S̄ = 0.745 (H₃⁺: 0.50), giving
/// ≈ 9 % over the 675 fs window — the bound is ≈ 2.2σ for uncoupled modes, not the H₃⁺ test's 3.3σ, and the O₂ vibration
/// (period ≈ 21 fs ≫ τ) decorrelates more slowly than τ. Printed; not asserted as a design guard (it would fail at the
/// brief's window): a gross-error detector (a DOF or K̄ error of 2× fails it), deterministic on one adapter — the seed is
/// fixed (`SEED` = 7), so the test is one pinned trajectory, not a fresh 2.2σ draw. Coordinator ruling (2026-10-18): the
/// window is lengthened to meet σ ≤ bound/3 only if the extra steps cost ≤ 10 min; the run prints that cost.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_o2_csvr_reaches_target() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let h = h_for(&[8], 0.25);
    let mut s = pseudo_scene(vec![Nucleus::new(8, [-1.14, 0.0, 0.0]), Nucleus::new(8, [1.14, 0.0, 0.0])], h, 10.0, IonicPolicy::CoresOnly);
    s.multiplicity = Some(3);
    let n = s.cell.n;
    let (p, log) = Locked::new(ctx.clone(), 1e-5);
    let mut d = sim::Dynamics::new(s, Box::new(p), DT_FS, SEED);
    d.apply(sim::DynCommand::SetComCorrection(true));
    d.apply(sim::DynCommand::SetThermostat(Some((T_TARGET, THERMO_TAU_FS))));
    let s1 = d.snapshot();
    assert!(s1.last_error.is_none() && s1.last_rejection.is_none(), "{:?} {:?}", s1.last_error, s1.last_rejection);
    assert_eq!(s1.scene.dof(), 3);
    let mut temps = Vec::new();
    for i in 1..=THERMO_STEPS {
        d.apply(sim::DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert!(snap.last_error.is_none(), "step {i}: {:?}", snap.last_error);
        if i > THERMO_SETTLE_STEPS { temps.push(snap.temperature); }
    }
    let last = d.snapshot();
    let mean = temps.iter().sum::<f64>() / temps.len() as f64;
    let t_w = temps.len() as f64 * DT_FS;
    let c = (0.25f64 + 2.0 * 0.5).sqrt() / 1.5;
    let sigma = c * (2.0 * THERMO_TAU_FS / t_w * (1.0 - THERMO_TAU_FS / t_w * (1.0 - (-t_w / THERMO_TAU_FS).exp()))).sqrt();
    let block = 100;
    let blocks: Vec<f64> = temps.chunks_exact(block).map(|c| c.iter().sum::<f64>() / block as f64).collect();
    let log = log.lock().unwrap().clone();
    // the design check of the dynamics suite (σ ≤ bound/3), printed: the window it would need and what that costs here
    // (coordinator ruling: lengthen the window instead when the extra steps cost ≤ 10 min)
    let need_tw = 2.0 * THERMO_TAU_FS * (c / (T_BOUND / 3.0)).powi(2);
    let need_steps = THERMO_SETTLE_STEPS + (need_tw / DT_FS).ceil() as usize;
    let s_per_step = t0.elapsed().as_secs_f64() / THERMO_STEPS as f64;
    eprintln!("pseudo-O₂ CSVR design check: σ_mean/T {:.1} % vs bound/3 = {:.1} % → {}; σ ≤ bound/3 needs T_w ≥ {need_tw:.0} fs = {need_steps} steps, {} more at {:.2} s/step ≈ {:.1} min",
              100.0 * sigma, 100.0 * T_BOUND / 3.0, if sigma <= T_BOUND / 3.0 { "met" } else { "not met (a gross-error detector at this window)" },
              need_steps.saturating_sub(THERMO_STEPS), s_per_step, need_steps.saturating_sub(THERMO_STEPS) as f64 * s_per_step / 60.0);
    eprintln!("pseudo-O₂ CSVR (CoresOnly, M = 3, h = {h}, {n}³, tol 1e-5, τ = {THERMO_TAU_FS} fs, seed {SEED}, COM correction on): ⟨T⟩ = {mean:.1} K ({:+.1} %, bound ±{:.0} %); analytic σ_mean/T ≈ {:.1} % (the bound is {:.1}σ); \
               block means ({block} steps) {}; E_conserved drift {:+.2e} Ha/ps; final R {:.4} Bohr; {}; warnings {:?}; {:.0} s",
              100.0 * (mean / T_TARGET - 1.0), 100.0 * T_BOUND, 100.0 * sigma, T_BOUND / sigma, blocks.iter().map(|b| format!("{b:.0}")).collect::<Vec<_>>().join(" "),
              last.drift_per_ps.unwrap_or(f64::NAN), bond(&last.scene), log.summary(), log.warnings, t0.elapsed().as_secs_f64());
    record(&ctx, timing_row("pseudo_o2_csvr", "O₂ (CoresOnly)", n, 1e-5, &log, t0.elapsed(), &format!("⟨T⟩ {mean:.0} K ({:+.1} %)", 100.0 * (mean / T_TARGET - 1.0))));
    assert!((mean / T_TARGET - 1.0).abs() <= T_BOUND, "⟨T⟩ = {mean:.1} K vs {T_TARGET} K (bound {:.0} %)", 100.0 * T_BOUND);
}
