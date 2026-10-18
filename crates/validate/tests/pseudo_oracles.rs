//! M7 Tasks 10 and 11: the validation oracles of the self-generated pseudopotentials on the GPU solver (spec §6.3–§6.7,
//! §8 rows 10 and 11). Atomic units inside (Hartree, Bohr, Ha/Bohr; Å, degrees and cm⁻¹ only in the literature constants
//! and the printout). Release-only: `cargo test -p validate --release --test pseudo_oracles` (skips cleanly without a GPU
//! adapter; `--nocapture` prints every number; each test appends its timing row to the gitignored
//! `last-run-pseudo.md`). Numbers of record: `references.md` "M7 oracles (Task 10)" and "M7 `All` (Task 11)".
//!
//! # Protocol
//! - GPU: `common::Locked` (`GpuKs` behind the suite's GPU lock, side solvers off), SCF tol 1e-6 (the floor-aware stop
//!   `providers::pseudo::effective_gpu_tol` keeps it for N ≤ 10 valence electrons; O₂'s N = 12 runs at 1.2e-6).
//! - The tables are the providers' cache (`providers::pseudo::table_for`, the ones the solves use — plan review M6):
//!   since Task 11 H and He come through the dimer gate on f (`providers::pseudo::GridDimerGate`, on this suite's GPU:
//!   `common::gpu` sets the backend). Recipe `m7-tm-kb-nlcc-4` + `DIMER_PROCEDURE` (the gate moved only H and He; c = ¼ since
//!   -4, NLCC ruling: the geometry oracles moved by δθ ≈ 0.03°, δr ≈ 1e-4 Bohr against -3).
//! - Grids: h = 0.25 when every species' h_req allows it, else the finest h_req (`h_for`; O and N: 0.20), plus
//!   h = 0.20 reported where that differs (CH₄); a 14 Bohr cube with a grid point at the origin (`pseudo_scene`).
//! - Relaxation: the quenched velocity Verlet of `common::relax` (the M6 relax tool's quench, spec §6.4), Δt 0.5 fs, run
//!   to max_I |F_I| < `ORACLE_FTOL` = 2e-4 Ha/Bohr (at most `ORACLE_MAX_STEPS`) instead of the suite-wide 1e-3, because
//!   the angle a residual force leaves is a third of the H₂O bound and more than the CH₄ one (derivation at the
//!   constant). The geometry is cached per (molecule, policy, h) for the binary, so the egg-box record probes the same
//!   relaxed water the angle test measured.
//! - **Spec question 8** (plan review, accepted): the geometries are gated under `All` — since Task 11 with the
//!   dimer-gated H (Task 10's runs used H at its atomic-gate f; their re-run with the gated H is what references.md keeps).
//!   `CoresOnly` (pseudo-O/N/C beside an all-electron H smeared at σ = h) is recorded, not gated: its miss is the
//!   measured cost of the remaining smearing (spec §6.4, review S8). **Stop rule (Task 11 carry):** an H₂O O–H past
//!   0.02 Å of the reference is a stop and a report, never a bound revision; the O table's own bond-length error (O₂
//!   r_e +0.0097 Å against SVWN/aug-cc-pVQZ, `geometry_attribution_report`) stays recorded as a known limitation.
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
//! - `ne_h_convergence_is_monotone` (spec §6.3; fix round, controller ruling): pseudo-Ne over the ladder 0.35 … 0.15 and
//!   0.125, 0.10 through `IonicSetup::from_tables`, energies re-evaluated in f64; gated over the spacings each certified
//!   by its own T8 row (≥ 3; Ne's h_req is 0.15), the uncertified ones reported; the 0.15 3-D-vs-radial gap gated
//!   against the derived stencil term (`stencil_kinetic_shift`).
//! - `h2o_cores_only_eggbox_is_recorded` (spec §6.6): the relaxed `CoresOnly` water's rigid-shift probe, f64-energy
//!   instrument (`eggbox`), wall-subtracted, at O's h_req 0.20 and the app default 0.15, "per the outcome" (NLCC ruling):
//!   PASS asserted where recorded, the ratchet where a MISS is recorded, report only before a record.
//!   `nh3_ch4_eggbox_records` (ignored): the C and N molecule records under `All` (Task 6 review: records, not gates),
//!   with 0.15 and, where the app allows Fast, 0.30.
//! - `pseudo_o2_nve_drift_is_bounded` (spec §6.7): O₂ under `CoresOnly` (all-pseudo), triplet, at the dynamics suite's
//!   bound; the momentum budget from the recorded O₂ egg-box (NLCC ruling "Task 10"; task-8b review "What Task 10 must
//!   carry"). `pseudo_ch4_csvr_reaches_target`: CSVR on CH₄ under `All` (N_f = 12) with the COM correction on and the
//!   suite's pre-registered design check asserted; the first design's O₂ (N_f = 3, which fails that check) is the
//!   `--ignored` report `pseudo_o2_csvr_report` (fix round, controller ruling).
//!
//! - Task 11 (spec §6.4, §6.6, §6.7, §8 row 11): `pseudo_h2_omega_within_5pct_of_sigma0_curve` (pseudo-H₂'s R_e and ω
//!   against this solver's own σ → 0 all-electron curve, ω as the step-halved Richardson frequency on both sides, computed
//!   live — independent of the gate's δ = h number, review SHOULD-1); `h2_all_eggbox_passes` (the core-free system: the hard §6.6
//!   PASS); `h2o_all_eggbox_recorded` (the relaxed water under `All`, f64-energy instrument, wall-subtracted, "per the
//!   outcome" — NLCC ruling Ruling 6: the brief's `h2o_all_max_shift_sum_of_forces_below_3e_5` was kept only if it passed;
//!   it MISSED at 0.20, so the ratchet), with the {0, h/2}³ energy ripple and the 0.15 probe beside it;
//!   `h2o_preset_dynamics_gate` (spec §6.7: `thermostat_h2o_preset_known_limitation` becomes a gate under `All`).
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
/// while staying 2.3× above the GPU pseudo path's force noise as max|F| sees it (a per-atom vector norm: √3 · 5e-5 =
/// 8.7e-5 for ≈ 5e-5 per component, warm chain; task-8b review ruling (b)) and above the egg-box (O₂ G ≤ 4.3e-5 at 0.20)
/// — which is why the stop needs the confirmation loop of `quench`. The honest geometric uncertainty is the
/// 2e-4 + √3·5e-5 ball: ≈ ±0.2° in angle, ≈ ±3e-4 Å in length (physics review). Reported: the final max|F|.
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
    // Task 11 (run of record 2026-10-18, RX 7900 XTX; dimer-gated H f = 1.2): the relaxed water under `All` MISSES at O's
    // h_req 0.20 — G 4.153e-5, u_I 1.91e-8 (8.2× vector, 15.9× on y against 3.4e-4) — so the ratchet (NLCC ruling 6)
    ("h2o-all-relaxed", 0.20, 4.153e-5, 1.91e-8, false),
    // and PASSES at the app default 0.15: G 4.720e-6, u_I 1.56e-8 (G + 2u_I 4.75e-6) — the hard PASS from here on
    ("h2o-all-relaxed", 0.15, 4.720e-6, 1.56e-8, true),
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
/// relaxes to ≈ 177° (`h2o_under_none_is_the_smeared_177`). H's table is the dimer-gated one since Task 11; Task 10 measured
/// with the atomic-gate H (+0.0163 Å on O–H at 0.20). An O–H past ±0.02 Å is a stop (Task 11 carry, rule 6).
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
               LSDA (SVWN/aug-cc-pVQZ) {LSDA_H2O_ANGLE_DEG}°, {LSDA_H2O_ROH_ANGSTROM} Å → All Δ {:+.3}°, {:+.4} Å; CoresOnly Δ {:+.3}°, {:+.4} Å; reported alongside {REPORTED_H2O:?}; H table f = {} {} (dimer gate: {})",
              all.scene.cell.n, dist_angstrom(&all.scene, 0, 2), all.energy, all.steps, all.max_f, dist_angstrom(&core.scene, 0, 2), core.energy, core.steps,
              ang - LSDA_H2O_ANGLE_DEG, roh - LSDA_H2O_ROH_ANGSTROM, ang_c - LSDA_H2O_ANGLE_DEG, roh_c - LSDA_H2O_ROH_ANGSTROM, h_table.f, h_table.record.local, h_table.record.dimer);
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

/// NIST CCCBDB (Release 22), LSDA (= Gaussian SVWN, VWN-RPA) / aug-cc-pVQZ, O₂ ³Σg⁻: r_e = 1.2021 Å (aug-cc-pVTZ 1.2037 Å;
/// accessed 2026-10-18; references.md "M7 oracles (Task 10)"). Reported against, not pinned (physics review SHOULD-1).
const LSDA_O2_RE_ANGSTROM: f64 = 1.2021;

/// Report (`--ignored`; physics review SHOULD-1 and Minor-6, Task 10 fix round): what the water's O–H excess under `All`
/// (+0.0163 Å at 0.20) comes from, and the `CoresOnly` geometries spec §6.4 asks to record for every molecule.
/// - H₂O under `All` at h = 0.15 (95³): separates the grid from the tables (an O–H that does not move is not the grid).
/// - O₂ under `CoresOnly` (all-pseudo, triplet, 14 Bohr, O's h_req 0.20) relaxed r_e against CCCBDB SVWN/aug-cc-pVQZ
///   (`LSDA_O2_RE_ANGSTROM`): the O table without H. If O₂ is close and 0.15 does not move the O–H, the residual is H or
///   the O–H pairing (Task 11's dimer gate is where to look); if O₂ is long too, it is the O table.
/// - NH₃ and CH₄ under `CoresOnly` (pseudo-N/C beside the smeared H) at their gate h: recorded like the water's.
/// Nothing asserted beyond the relax converging; GPU ≈ 10 min.
#[test]
#[ignore = "report (≈ 10 min): cargo test -p validate --release --test pseudo_oracles -- --ignored --nocapture geometry_attribution_report"]
fn geometry_attribution_report() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let w = relaxed(&ctx, "h2o/all/0.15", || water(IonicPolicy::All, 0.15));
    let (ang, roh) = (angle_deg(&w.scene, 0, 1, 2), dist_angstrom(&w.scene, 0, 1));
    eprintln!("H₂O (All) at h = 0.15 ({}³): ∠HOH {ang:.3}° (Δ {:+.3}°), r(O–H) {roh:.4} / {:.4} Å (Δ {:+.4} Å), E {:.6} Ha, {} relax steps, max|F| {:.1e}",
              w.scene.cell.n, ang - LSDA_H2O_ANGLE_DEG, dist_angstrom(&w.scene, 0, 2), roh - LSDA_H2O_ROH_ANGSTROM, w.energy, w.steps, w.max_f);
    relax_row(&ctx, "attribution: H₂O All 0.15", "H₂O", &w, &format!("∠HOH {ang:.2}°, r(O–H) {roh:.4} Å"));
    let h = h_for(&[8], 0.25);
    let o2 = relaxed(&ctx, &format!("o2/cores/{h}"), || {
        let mut s = pseudo_scene(vec![Nucleus::new(8, [-1.17, 0.0, 0.0]), Nucleus::new(8, [1.17, 0.0, 0.0])], h, BOX, IonicPolicy::CoresOnly);
        s.multiplicity = Some(3);
        s
    });
    let r = dist_angstrom(&o2.scene, 0, 1);
    eprintln!("O₂ (CoresOnly, M = 3) at h = {h} ({}³): r_e {r:.4} Å against SVWN/aug-cc-pVQZ {LSDA_O2_RE_ANGSTROM} Å (Δ {:+.4} Å; aug-cc-pVTZ 1.2037), E {:.6} Ha, {} relax steps, max|F| {:.1e}; warnings {:?}",
              o2.scene.cell.n, r - LSDA_O2_RE_ANGSTROM, o2.energy, o2.steps, o2.max_f, o2.eval.warnings);
    relax_row(&ctx, "attribution: O₂ CoresOnly", "O₂", &o2, &format!("r_e {r:.4} Å (SVWN/aQZ {LSDA_O2_RE_ANGSTROM})"));
    let hn = h_for(&[7, 1], 0.25);
    let n = relaxed(&ctx, &format!("nh3/cores/{hn}"), || ammonia(IonicPolicy::CoresOnly, hn));
    let na = [angle_deg(&n.scene, 0, 1, 2), angle_deg(&n.scene, 0, 2, 3), angle_deg(&n.scene, 0, 1, 3)];
    eprintln!("NH₃ (CoresOnly) at h = {hn}: ∠HNH {na:.3?}° (LSDA {LSDA_NH3_ANGLE_DEG}°), r(N–H) {:.4} Å, E {:.6} Ha, {} relax steps", dist_angstrom(&n.scene, 0, 1), n.energy, n.steps);
    relax_row(&ctx, "attribution: NH₃ CoresOnly (recorded)", "NH₃", &n, &format!("∠HNH {:.2}°, r(N–H) {:.4} Å", na[0], dist_angstrom(&n.scene, 0, 1)));
    let hc = h_for(&[6, 1], 0.25);
    let c = relaxed(&ctx, &format!("ch4/cores/{hc}"), || methane(IonicPolicy::CoresOnly, hc));
    let worst = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)].iter().map(|&(i, j)| (angle_deg(&c.scene, 0, i, j) - TETRAHEDRAL_DEG).abs()).fold(0.0, f64::max);
    eprintln!("CH₄ (CoresOnly) at h = {hc}: worst |∠HCH − 109.47°| {worst:.3}°, r(C–H) {:.4} Å, E {:.6} Ha, {} relax steps", dist_angstrom(&c.scene, 0, 1), c.energy, c.steps);
    relax_row(&ctx, "attribution: CH₄ CoresOnly (recorded)", "CH₄", &c, &format!("worst Δ∠ {worst:.3}°, r(C–H) {:.4} Å", dist_angstrom(&c.scene, 0, 1)));
}

// ---------------------------------------------------------------- atoms (spec §6.3)

/// The spacings of the pseudo-Ne series: the ladder (0.35 … 0.15) and two finer than it, 0.125 and 0.10, each certified
/// (or not) by its own T8 row (`qm::ionic` "Build rules": a scene finer than the ladder is certified at its own h).
const NE_SPACINGS: [f64; 7] = [0.35, 0.30, 0.25, 0.20, 0.15, 0.125, 0.10];
/// The house margin on a derived leading term (the 3× rule of the gpu_vs_cpu and Task 8 bounds).
const DERIVED_MARGIN: f64 = 3.0;

/// The first-order kinetic error of the 8th-order stencil (`qm::stencil`, c = [−205/72, 8/5, −1/5, 8/315, −1/560]) on the
/// radial KB atom's own orbitals, Hartree — the derived leading term of E_3D − E_radial on one table.
///
/// For a plane wave the stencil's −∇² is s(k) = −(1/h²)[c₀ + 2Σ_m c_m cos(mkh)] ≤ k² per axis, so ⟨ψ|T_h − T|ψ⟩ =
/// ∫ d³q/(2π)³ |ψ̂(q)|² ½[Σ_a s(q_a) − q²]. A closed shell (Ne) is spherical after the sum over m:
/// Σ_m|ψ̂_lm(q)|²/(2l+1) = |φ̂_l(q)|²/4π with φ̂_l(q) = 4π∫ j_l(qr) u_l(r) r dr (u = rR, ∫u²dr = 1; Parseval:
/// ∫ q²|φ̂|² dq/(2π)³ = 1), and the angular average of Σ_a cos(m q_a h) is 3 sin(mqh)/(mqh). Hence
///   δE_T = Σ_{l,σ} f_lσ ∫_0^{π/h} q² dq/(2π)³ |φ̂_l(q)|² · ½A(q),  A(q) = −(3/h²)[c₀ + 2Σ_m c_m sin(mqh)/(mqh)] − q² < 0,
/// over the sphere inside the first Brillouin zone; the spectral weight beyond π/h (aliasing, not this term) is
/// returned beside it. First order in the operator change (Hellmann–Feynman on the variational energy), negative: the
/// stencil underestimates q², so the grid energy lies below the radial one. The other grid terms (the FD Poisson's
/// Hartree, pointwise XC, the projector and V_SR quadratures, the box) are not in it.
fn stencil_kinetic_shift(ft: &qm::radial::FilteredTable, atom: &qm::radial::kb::KbAtom, occ: &qm::radial::kb::ValenceConfig, h: f64) -> (f64, f64) {
    let c = qm::stencil::COEFF;
    let g = &ft.grid;
    let q_max = std::f64::consts::PI / h;
    let nq = 2000;
    let dq = q_max / nq as f64;
    let a = |q: f64| -> f64 {
        if q == 0.0 { return 0.0; }
        let s: f64 = (1..=4).map(|m| { let x = m as f64 * q * h; c[m] * x.sin() / x }).sum();
        -(3.0 / (h * h)) * (c[0] + 2.0 * s) - q * q
    };
    let (mut de, mut w_out) = (0.0, 0.0);
    for (l, f) in occ.occ.iter().enumerate() {
        for sp in 0..2 {
            if f[sp] <= 0.0 { continue; }
            let u = if atom.u[l][sp].is_empty() { &atom.u[l][0] } else { &atom.u[l][sp] };
            let norm = g.integrate(&u.iter().map(|x| x * x).collect::<Vec<_>>());
            let phi = |q: f64| 4.0 * std::f64::consts::PI * g.integrate(&g.r.iter().zip(u).map(|(&r, &x)| qm::radial::sph_bessel(l as u8, q * r) * x * r).collect::<Vec<_>>()) / norm.sqrt();
            // Simpson on [0, π/h]
            let (mut sum_e, mut sum_w) = (0.0, 0.0);
            for k in 0..=nq {
                let q = k as f64 * dq;
                let wk = if k == 0 || k == nq { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
                let p2 = phi(q).powi(2) * q * q / (2.0 * std::f64::consts::PI).powi(3);
                sum_e += wk * p2 * 0.5 * a(q);
                sum_w += wk * p2;
            }
            de += f[sp] * sum_e * dq / 3.0;
            w_out = f64::max(w_out, 1.0 - sum_w * dq / 3.0);
        }
    }
    (de, w_out)
}

/// Spec §6.3 as amended (controller ruling and physics review MUST-1, Task 10 fix round): the h-convergence of pseudo-Ne
/// on the GPU is gated on the **grid part** at the **certified** spacings. `NE_SPACINGS` (the ladder plus 0.125 and
/// 0.10), 12 Bohr, `CoresOnly`, each spacing on Ne's table filtered at that h through `IonicSetup::from_tables`; certified
/// iff `IonicSetup::build` accepts it (its own T8 row; Ne's h_req is 0.15). Energies: the f64 re-evaluation of each cold
/// GPU state (`eggbox` instrument); the GPU's own E printed beside them.
///
/// E_3D(h) = E_rad(h) + [E_3D(h) − E_rad(h)]: the **filter part** E_rad(h), the radial KB atom on the table filtered at
/// h (a different potential at every h; T8(a) bounds it to 1 mHa of the unfiltered atom at a certified h, with no
/// derivable sign), and the **grid part**, the 3-D solver's integration error on one fixed table, which must vanish as h
/// refines. Only the grid part is a property of the 3-D solver (the §6.3 heading's question).
/// - Gated: ≥ 3 certified spacings (0.15 among them); at every certified spacing |E_3D − E_rad| < 2 mHa (spec §6.3's
///   bound, now at every certified rung); over them |E_3D − E_rad| non-increasing as h refines (1e-6 slack, the SCF
///   tolerance), the sign of the grid part reported.
/// - Gated at 0.15: |E_3D − E_rad| ≤ `DERIVED_MARGIN`·|δE_T|, δE_T the derived leading term (`stencil_kinetic_shift`):
///   the gap must be what the stencil explains.
/// - **Known, unexplained remainder (Task 10 re-review R-1).** The grid part minus δE_T is a nearly constant
///   +0.058 / +0.051 / +0.049 mHa at 0.15 / 0.125 / 0.10 (≈ 5e-5 Ha, 50× the SCF tolerance): an h-independent term not
///   yet attributed (the Poisson boundary and the radial reference are the candidates). It is why the 0.10 grid part
///   is +0.005 mHa (δE_T −0.044 + 0.049), a sign flip at the noise level with |.| still decreasing. A finer certified
///   spacing (≈ 0.08, where δE_T falls below ≈ 0.025 mHa) would make |grid part| ≈ the remainder and **rise**, failing
///   the non-increasing gate for this known reason, not a solver regression — do not add one without attributing it.
/// - Reported: the raw E(h) series with its sign (certified and all), the filter part (E_rad(h) and T8(a)'s ΔE) and the
///   grid part at every spacing, |E(0.25) − E(0.15)| split into the two (first run, 2026-10-18: over the ladder the raw
///   series was mixed, −14.8, −166.1, −130.7, +16.5 mHa, the radial energy moving by up to 0.32 Ha — spec §6.3's
///   "monotone" and "expected mHa" premise falsified for Ne at `m7-tm-kb-nlcc-4`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn ne_h_convergence_is_monotone() {
    use qm::ionic::{Ionic, IonicPolicy as QmPolicy, IonicSetup};
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let t = providers::pseudo::table_for(10).expect("Ne is accepted");
    let occ = qm::radial::kb::reference_config(&t.partition);
    // (h, certified, E_3D, E_rad, δE_T, T8(a) ΔE)
    let mut rows: Vec<(f64, bool, f64, f64, f64, f64)> = Vec::new();
    let mut log = EvalLog::default();
    let mut lines = String::new();
    for &h in NE_SPACINGS.iter() {
        let ft = Arc::new(qm::radial::filter::filtered(&t, h));
        let certified = IonicSetup::build(QmPolicy::CoresOnly, h, h, &[10], &mut |z| qm::radial::has_core(z), &mut |_| Ok(ft.clone()));
        let setup = Arc::new(IonicSetup::from_tables(QmPolicy::CoresOnly, h, h, BTreeMap::from([(10, Ionic::Pseudo(ft.clone()))])));
        let n = 2 * (12.0 / (2.0 * h)).round() as usize + 1;
        let g = qm::Grid::cube(n, h);
        let mut solver = { let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner()); qm_gpu::GpuSolver::new(ctx.clone(), &g) };
        let nuc = vec![qm::Nucleus { pos: [0.0; 3], z: 10 }];
        let (e_gpu, st, stats, dt) = eggbox::gpu_solve(&mut solver, setup.clone(), nuc.clone(), Some(1), false);
        drop(solver);
        let mut poisson = qm::poisson::PoissonSolver::new(&g);
        let (e, _) = eggbox::ks_energy_f64(&g, &eggbox::occupied(&g, &st), &st.v_h, &setup, &nuc, &mut poisson);
        let atom = qm::radial::kb::solve_kb_atom(&ft.grid, &ft.input(), &occ, qm::radial::SpinMode::Restricted).expect("the radial KB Ne");
        let (de_t, w_out) = stencil_kinetic_shift(&ft, &atom, &occ, h);
        (log.calls, log.gpu, log.scf_steps) = (log.calls + 1, log.gpu + dt, log.scf_steps + stats.iterations);
        let c = &ft.check;
        let _ = writeln!(lines, "  h = {h} ({n}³, {}): E_3D (f64) {:.7} (GPU {e_gpu:.7}); filter part E_rad {:.7} (T8(a) ΔE {:+.3} mHa); grid part E_3D − E_rad {:+.3} mHa, derived stencil term δE_T {:+.3} mHa (weight beyond π/h {w_out:.1e}); \
                                T8 row Δε {:.3} mHa, T3 {:.3} mHa, clamped {}, F_table {:.2e}; {} SCF steps, Δρ {:.1e}, warnings {:?}",
                         match &certified { Ok(_) => "certified".to_string(), Err(e) => format!("not certified: {e}") },
                         e.total, atom.energy, c.de * 1e3, (e.total - atom.energy) * 1e3, de_t * 1e3, c.max_eig * 1e3, c.t3_max * 1e3, c.clamped,
                         c.f_table.iter().sum::<f64>(), stats.iterations, stats.delta_rho, stats.warnings);
        rows.push((h, certified.is_ok(), e.total, atom.energy, de_t, c.de));
    }
    let sign_of = |es: &[f64]| -> (&'static str, Vec<f64>) {
        let d: Vec<f64> = es.windows(2).map(|w| w[1] - w[0]).collect();
        // 1e-6: the SCF tolerance, as the original one-sided check allowed
        let s = if d.iter().all(|&x| x <= 1e-6) { "downward (falls as h refines)" } else if d.iter().all(|&x| x >= -1e-6) { "upward (rises as h refines)" } else { "mixed" };
        (s, d)
    };
    let fmt = |d: &[f64]| d.iter().map(|x| format!("{:+.3}", x * 1e3)).collect::<Vec<_>>().join(", ");
    let cert: Vec<_> = rows.iter().filter(|r| r.1).collect();
    let grid: Vec<f64> = cert.iter().map(|r| r.2 - r.3).collect();
    let (grid_sign, grid_diffs) = sign_of(&grid.iter().map(|x| x.abs()).collect::<Vec<_>>());
    let (raw_sign, raw_diffs) = sign_of(&cert.iter().map(|r| r.2).collect::<Vec<_>>());
    let (all_sign, all_diffs) = sign_of(&rows.iter().map(|r| r.2).collect::<Vec<_>>());
    let at = |h: f64| *rows.iter().find(|r| (r.0 - h).abs() < 1e-12).expect("a spacing");
    let (_, _, e15, r15, t15, _) = at(0.15);
    let (_, _, e25, r25, _, _) = at(0.25);
    let gap15 = e15 - r15;
    eprintln!("pseudo-Ne E(h), 12 Bohr, CoresOnly via from_tables (h_req {:?}):\n{lines}  certified spacings {:?}: grid part [{}] mHa, |grid part| successive differences [{}] mHa: {grid_sign}; \
               raw E successive differences (reported) [{}] mHa: {raw_sign}; all spacings (reported) [{}] mHa: {all_sign}; \
               |E(0.25) − E(0.15)| = {:.3} mHa = filter part {:+.3} + grid part {:+.3} mHa; at 0.15 the grid part {:+.3} mHa against the derived stencil term {:+.3} mHa (bound {DERIVED_MARGIN}× = {:.3} mHa)",
              t.record.h_req, cert.iter().map(|r| r.0).collect::<Vec<_>>(), fmt(&grid), fmt(&grid_diffs), fmt(&raw_diffs), fmt(&all_diffs),
              (e25 - e15).abs() * 1e3, (r25 - r15) * 1e3, ((e25 - r25) - gap15) * 1e3, gap15 * 1e3, t15 * 1e3, DERIVED_MARGIN * t15.abs() * 1e3);
    record(&ctx, timing_row("ne_h_convergence", "Ne (35³–121³)", 121, eggbox::GPU_TOL, &log, t0.elapsed(),
                            &format!("certified |grid part| {grid_sign}; 0.15 grid part {:+.2} mHa (δE_T {:+.2})", gap15 * 1e3, t15 * 1e3)));
    assert!(cert.len() >= 3, "fewer than 3 certified spacings: {:?}", cert.iter().map(|r| r.0).collect::<Vec<_>>());
    assert!(cert.iter().any(|r| (r.0 - 0.15).abs() < 1e-12), "Ne must be certified at its h_req 0.15");
    for r in &cert { assert!((r.2 - r.3).abs() < 2e-3, "h = {}: grid part E_3D − E_rad = {:+.3} mHa (spec §6.3: < 2 mHa)", r.0, (r.2 - r.3) * 1e3); }
    assert_eq!(grid_sign, "downward (falls as h refines)", "|grid part| not non-increasing over the certified spacings: {grid:?}");
    assert!(gap15.abs() <= DERIVED_MARGIN * t15.abs(), "grid part at 0.15 = {:+.3} mHa is not the stencil's {:+.3} mHa (≤ {DERIVED_MARGIN}×)", gap15 * 1e3, t15 * 1e3);
}

// ---------------------------------------------------------------- egg-box (spec §6.6)

/// `relaxed`'s water as the egg-box instrument's molecule: its nuclei shifted rigidly by s (qm form).
fn rigid(s: &Scene) -> impl Fn([f64; 3]) -> Vec<qm::Nucleus> + '_ {
    move |d| s.nuclei.iter().map(|n| qm::Nucleus { pos: [n.pos[0] + d[0], n.pos[1] + d[1], n.pos[2] + d[2]], z: n.z }).collect()
}

/// One §6.6 probe of a relaxed scene at spacing h (the providers' certified setup at that h, σ = h for an all-electron
/// species, the scene's multiplicity), or the providers' refusal at that h.
fn probe_at(ctx: &GpuContext, label: &str, base: &Scene, h: f64, mult: Option<u32>) -> Result<eggbox::Probe, String> {
    probe_with(ctx, label, base, h, mult, record_for(label, h)).map(|(p, _)| p)
}

/// `probe_at` with the record given (a hard PASS is `Some((NaN, NaN, true))`), returning the setup too.
fn probe_with(ctx: &GpuContext, label: &str, base: &Scene, h: f64, mult: Option<u32>, rec: Option<(f64, f64, bool)>) -> Result<(eggbox::Probe, Arc<qm::IonicSetup>), String> {
    let at_h = pseudo_scene(base.nuclei.clone(), h, BOX, base.ionic);
    let mut notes = Vec::new();
    let setup = providers::pseudo::setup_for(&at_h, h, &mut notes).map_err(|e| e.to_string())?.ok_or("nothing pseudised")?;
    Ok((eggbox::probe(ctx, label, h, BOX, setup.clone(), &rigid(base), mult, rec), setup))
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

/// What one CSVR run measured.
struct CsvrRun { mean: f64, sigma: f64, line: String, warnings: std::collections::BTreeSet<String> }

/// The analytic σ_mean/T of the window mean of T (the dynamics suite's derivation, `csvr_run` doc) for a molecule with
/// `n_v` vibrations and `n_r` rotations at `THERMO_TAU_FS` over the window after `THERMO_SETTLE_STEPS`.
fn csvr_sigma(n_v: usize, n_r: usize) -> f64 {
    let t_w = (THERMO_STEPS - THERMO_SETTLE_STEPS) as f64 * DT_FS;
    let c = (n_v as f64 / 4.0 + n_r as f64 / 2.0).sqrt() / ((n_v + n_r) as f64 / 2.0);
    c * (2.0 * THERMO_TAU_FS / t_w * (1.0 - THERMO_TAU_FS / t_w * (1.0 - (-t_w / THERMO_TAU_FS).exp()))).sqrt()
}

/// One thermostatted run through the driver at the dynamics suite's settings (T = 300 K, τ = 5 fs, `THERMO_STEPS`,
/// settle `THERMO_SETTLE_STEPS`, seed 7, tol 1e-5 — the interactive tolerance the thermostat test runs at), the COM
/// correction on (NLCC ruling "Task 10": a thermostatted pseudo run turns it on rather than budgeting the egg-box
/// momentum), from `start` at rest (the driver's Maxwell–Boltzmann draw, R7). `n_v`, `n_r`: the molecule's vibrational
/// and rotational DOF for the design check.
///
/// The design check (the dynamics suite's, `dynamics.rs` module doc): the window mean of T is governed by
/// S = Σ_vib E_i/2 + Σ_rot K_j, S̄ = (n_v + n_r)k_BT/2, Var S = (n_v/4 + n_r/2)(k_BT)², relaxed with time constant τ, so
/// σ_mean/T ≈ (√Var S/S̄)·√(2τ/T_w·(1 − (τ/T_w)(1 − e^{−T_w/τ}))); pre-registered: σ ≤ bound/3.
fn csvr_run(ctx: &GpuContext, label: &str, start: Scene, n_v: usize, n_r: usize) -> CsvrRun {
    let t0 = Instant::now();
    let n = start.cell.n;
    let h = start.cell.h;
    let (p, log) = Locked::new(ctx.clone(), 1e-5);
    let mut d = sim::Dynamics::new(start, Box::new(p), DT_FS, SEED);
    d.apply(sim::DynCommand::SetComCorrection(true));
    d.apply(sim::DynCommand::SetThermostat(Some((T_TARGET, THERMO_TAU_FS))));
    let s1 = d.snapshot();
    assert!(s1.last_error.is_none() && s1.last_rejection.is_none(), "{label}: {:?} {:?}", s1.last_error, s1.last_rejection);
    assert_eq!(s1.scene.dof(), n_v + n_r, "{label}: N_f");
    let mut temps = Vec::new();
    for i in 1..=THERMO_STEPS {
        d.apply(sim::DynCommand::Step(1));
        let snap = d.tick(f64::INFINITY, &mut never());
        assert!(snap.last_error.is_none(), "{label} step {i}: {:?}", snap.last_error);
        if i > THERMO_SETTLE_STEPS { temps.push(snap.temperature); }
    }
    let last = d.snapshot();
    let mean = temps.iter().sum::<f64>() / temps.len() as f64;
    let c = (n_v as f64 / 4.0 + n_r as f64 / 2.0).sqrt() / ((n_v + n_r) as f64 / 2.0);
    let sigma = csvr_sigma(n_v, n_r);
    let block = 100;
    let blocks: Vec<f64> = temps.chunks_exact(block).map(|c| c.iter().sum::<f64>() / block as f64).collect();
    let log = log.lock().unwrap().clone();
    let need_tw = 2.0 * THERMO_TAU_FS * (c / (T_BOUND / 3.0)).powi(2);
    let line = format!("{label} CSVR (h = {h}, {n}³, tol 1e-5, τ = {THERMO_TAU_FS} fs, {THERMO_STEPS} steps, seed {SEED}, COM correction on, N_f = {}): ⟨T⟩ = {mean:.1} K ({:+.1} %, bound ±{:.0} %); \
                        design check σ_mean/T {:.1} % vs bound/3 = {:.1} % → {} (σ ≤ bound/3 needs T_w ≥ {need_tw:.0} fs); block means ({block} steps) {}; E_conserved drift {:+.2e} Ha/ps; {}; warnings {:?}; {:.0} s",
                       n_v + n_r, 100.0 * (mean / T_TARGET - 1.0), 100.0 * T_BOUND, 100.0 * sigma, 100.0 * T_BOUND / 3.0, if sigma <= T_BOUND / 3.0 { "met" } else { "NOT met" },
                       blocks.iter().map(|b| format!("{b:.0}")).collect::<Vec<_>>().join(" "), last.drift_per_ps.unwrap_or(f64::NAN), log.summary(), log.warnings, t0.elapsed().as_secs_f64());
    eprintln!("{line}");
    record(ctx, timing_row(&format!("csvr ({label})"), label, n, 1e-5, &log, t0.elapsed(), &format!("⟨T⟩ {mean:.0} K ({:+.1} %), σ {:.1} %", 100.0 * (mean / T_TARGET - 1.0), 100.0 * sigma)));
    CsvrRun { mean, sigma, line, warnings: log.warnings.clone() }
}

/// Spec §6.7: CSVR on a pseudo system reaches its target at the dynamics suite's bounds — **CH₄ under `All`** (pseudo-C,
/// pseudo-H; N_f = 3N − 3 = 12 under the COM correction: n_v = 3N − 6 = 9 vibrations, n_r = 3 rotations), relaxed at
/// h = 0.25 (`ch4_angle_is_tetrahedral_within_half_degree`'s geometry), 14 Bohr, no box warning; |⟨T⟩/T − 1| ≤ 0.2.
///
/// The design check is pre-registered and asserted **before** the run: from CH₄'s own modes √Var S/S̄ = √(9/4 + 3/2)/6
/// = 0.3227, so σ_mean/T = 3.9 % over the 675 fs window (4.95 % if the vibrational energies relax at ≈ 2τ, the physics
/// review's correction), against bound/3 = 6.7 %: the 20 % bound is a 4.0σ (2.5σ with that correction) statement
/// analytically — at parity with or better than the dynamics suite's H₃⁺ gate (3.3σ), whose two seeds showed the real
/// window mean wider than the uncoupled-mode σ (`dynamics.rs` module doc). One run of record, seed 7, no other seed tried.
///
/// Why not O₂ (controller ruling, Task 10 fix round; not a loosening): the first design ran O₂ (N_f = 3), whose
/// σ_mean/T = 9.0 % (9.9 % with the 2τ correction) at this window violates that pre-registered check — its 20 % bound was
/// a ≈ 2.2σ statement — and its run of record read ⟨T⟩ = 361.2 K (+20.4 %, +2.3σ, E_conserved drift 1.3e-6 Ha/ps), a run
/// against a gate whose own design check failed. The gate moves to a molecule the check admits, at the same window, bound,
/// settle and seed; O₂ stays a report (`pseudo_o2_csvr_report`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_ch4_csvr_reaches_target() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[6, 1], 0.25);
    let sigma = csvr_sigma(9, 3);
    assert!(sigma <= T_BOUND / 3.0, "test design: σ_mean/T {:.1} % must be ≤ a third of the bound", 100.0 * sigma);
    let base = relaxed(&ctx, &format!("ch4/all/{h}"), || methane(IonicPolicy::All, h));
    assert_eq!(base.scene.cell.n, 2 * (BOX / (2.0 * h)).round() as usize + 1, "the 14 Bohr box");
    let r = csvr_run(&ctx, "CH₄ (All)", base.scene.clone(), 9, 3);
    assert!(r.warnings.iter().all(|w| !w.contains("box too small")), "CH₄ CSVR: {:?}", r.warnings);
    assert!((r.mean / T_TARGET - 1.0).abs() <= T_BOUND, "⟨T⟩ = {:.1} K vs {T_TARGET} K (bound {:.0} %)\n{}", r.mean, 100.0 * T_BOUND, r.line);
}

/// Report (`--ignored`, ≈ 1 h): the first CSVR design, O₂ under `CoresOnly`, triplet, from rest at R = 2.28 Bohr, 10 Bohr
/// box — N_f = 3, σ_mean/T 9.0–9.9 % > bound/3 = 6.7 % (see `pseudo_ch4_csvr_reaches_target`). Run of record 2026-10-18:
/// ⟨T⟩ = 361.2 K (+20.4 %), a run against a gate whose own design check failed. Nothing asserted on T.
#[test]
#[ignore = "report (≈ 1 h): cargo test -p validate --release --test pseudo_oracles -- --ignored --nocapture pseudo_o2_csvr_report"]
fn pseudo_o2_csvr_report() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8], 0.25);
    let mut s = pseudo_scene(vec![Nucleus::new(8, [-1.14, 0.0, 0.0]), Nucleus::new(8, [1.14, 0.0, 0.0])], h, 10.0, IonicPolicy::CoresOnly);
    s.multiplicity = Some(3);
    let r = csvr_run(&ctx, "O₂ (CoresOnly)", s, 1, 2);
    eprintln!("pseudo_o2_csvr_report: ⟨T⟩ {:.1} K, σ {:.1} % (report only)", r.mean, 100.0 * r.sigma);
}

// ---------------------------------------------------------------- Task 11: IonicPolicy::All (spec §6.4, §6.6, §6.7)

/// This solver's own all-electron H₂ ω at σ → 0, cm⁻¹, by the dimer gate's procedure (`providers::pseudo`: h = 0.15, 81³,
/// 12 Bohr, F(R; σ) at σ = h·{1, 1.2, 1.4, 1.6} extrapolated σ → 0 per R, R_e by bisection, k = −[F(R_e + h) − F(R_e − h)]/(2h)):
/// k = 0.3458 Ha/Bohr², ω = 0.019402 a.u. (M7 Task 11 run of record, 2026-10-18, RX 7900 XTX; R_e 1.4489 Bohr). Reported
/// beside the gated quantity below (the gate's restatement), no longer the oracle's reference (review SHOULD-1).
const AE_SIGMA0_OMEGA_CM: f64 = 4258.2;
/// This solver's own all-electron H₂ R_e at σ → 0 (references.md "Oracle results", `h2_bond_length`, CPU f64): 1.451 Bohr
/// (the gate's GPU procedure gives 1.4489, within its ≈ 1e-3 Bohr extrapolation noise).
const AE_SIGMA0_RE_BOHR: f64 = 1.451;
/// CCCBDB LSDA (SVWN) H₂ ω_e, cm⁻¹ (the dynamics suite's `OMEGA_E_LSDA_CM`), and experiment (NIST, ω_e) — reported.
const LSDA_H2_OMEGA_CM: f64 = 4201.0;
const EXPT_H2_OMEGA_CM: f64 = 4401.21;

/// Spec §8 row 11 first failing test (§6.4): pseudo-H₂ under `All` against this solver's own σ → 0 all-electron curve —
/// R_e within 0.02 Bohr of the CPU oracle's 1.451, ω within 5 % (the bound unchanged). The table is the providers'
/// (dimer-gated) H — the one every solve uses.
///
/// **Independent of the gate (Task 11 review SHOULD-1, controller ruling).** The gate chose f by "ω within 5 % at the step
/// δ = h"; asserting that same number again would pass by construction. So ω here is the **step-halved Richardson**
/// frequency on both sides — k₀ = [4k(h/2) − k(h)]/3 removes the O(δ²) bias k(δ) = k + E⁗δ²/6 of the central difference
/// with no curve model (`providers::pseudo::DimerFit`) — for the pseudised dimer and for the all-electron σ → 0 dimer computed
/// here, live, and the ratio of the two is asserted within 5 %. Reported: each bias-free ω against LSDA (4201) and
/// experiment (4401), and the gate's same-procedure ratio (the gate's own −3.45 %; reference 4258).
///
/// **Noise (Task 11 review N-2).** k(h/2) uses a 0.15 Bohr difference, so the AE k noise is √2·3e-4/0.15 ≈ 2.8e-3 Ha/Bohr²
/// (0.8 %). The Richardson weights (4/3, −1/3) amplify that to ≈ 1.1 % in k₀, i.e. ≈ 0.55 % in the AE ω₀; the PP noise is
/// negligible (no σ extrapolation). The ratio's −3.67 % therefore sits ≈ 2.4σ inside the 5 % bound: deterministic on one
/// adapter, but not a wide margin on a fresh draw (the AE ω₀ against LSDA, −0.57 %, is ≈ 1σ, consistent). A future miss near
/// 5 % is read as noise first, investigated, and never met by loosening the bound.
///
/// The reference chain (controller ruling; references.md "M7 `All` (Task 11)"): the brief's 4405 ± 60 (a parabola at 0.15
/// Bohr spacing, biased high) was superseded by this solver's own AE ω — 4258 cm⁻¹ at the gate's step, whose Morse bias
/// 7a²h²/6 (+1.38 % in ω) puts the bias-free value at ≈ 4200 (LSDA 4201) — which Richardson now measures directly.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn pseudo_h2_omega_within_5pct_of_sigma0_curve() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let t = providers::pseudo::table_for(1).expect("pseudo-H generates (with its dimer gate)");
    let t_gen = t0.elapsed().as_secs_f64();
    assert!(!t.provisional, "the providers' H is the gated one");
    let backend = providers::pseudo::DimerBackend::Gpu(ctx.clone());
    let (pp, ae) = {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        (providers::pseudo::pseudo_dimer_fit(&t, &backend).expect("pseudo-H₂ is bound"), providers::pseudo::all_electron_dimer_fit(&t, &backend).expect("AE H₂ is bound"))
    };
    let cm = |w: f64| w * HARTREE_IN_CM;
    let ratio = pp.omega_richardson / ae.omega_richardson;
    eprintln!("pseudo-H₂ (f = {}, {}, h_req {:?}, h_eb {:?}): R_e {:.4} Bohr (Δ {:+.4} against AE σ→0 {AE_SIGMA0_RE_BOHR}; this run's AE {:.4}); \
               ω at δ = h / h/2 / Richardson: PP {:.0} / {:.0} / {:.0} cm⁻¹, AE σ→0 {:.0} / {:.0} / {:.0} cm⁻¹ → bias-free ratio − 1 {:+.2} % (bound 5 %); \
               the gate's same-procedure ratio {:+.2} % (AE at δ = h {:.0} vs the recorded {AE_SIGMA0_OMEGA_CM}); bias-free PP vs LSDA {LSDA_H2_OMEGA_CM} {:+.2} %, expt {EXPT_H2_OMEGA_CM} {:+.2} %; \
               bias-free AE vs LSDA {:+.2} %; dimer gate: {}; generation {t_gen:.0} s, test {:.0} s",
              t.f, t.record.local, t.record.h_req, t.record.h_eb, pp.r_e, pp.r_e - AE_SIGMA0_RE_BOHR, ae.r_e,
              cm(pp.omega), cm(pp.omega_half), cm(pp.omega_richardson), cm(ae.omega), cm(ae.omega_half), cm(ae.omega_richardson), 100.0 * (ratio - 1.0),
              100.0 * (pp.omega / ae.omega - 1.0), cm(ae.omega), 100.0 * (cm(pp.omega_richardson) / LSDA_H2_OMEGA_CM - 1.0), 100.0 * (cm(pp.omega_richardson) / EXPT_H2_OMEGA_CM - 1.0),
              100.0 * (cm(ae.omega_richardson) / LSDA_H2_OMEGA_CM - 1.0), t.record.dimer, t0.elapsed().as_secs_f64());
    record(&ctx, timing_row("pseudo_h2_omega", "H₂ (All)", 81, 1e-6, &EvalLog::default(), t0.elapsed(),
                            &format!("R_e {:.4} Bohr, ω₀ PP/AE − 1 {:+.2} %; H f = {}", pp.r_e, 100.0 * (ratio - 1.0), t.f)));
    assert!((pp.r_e - AE_SIGMA0_RE_BOHR).abs() <= 0.02, "R_e {:.4}", pp.r_e);
    assert!((ratio - 1.0).abs() <= 0.05, "bias-free ω ratio PP/AE {ratio:.4}");
}

/// H₂ along x at bond length r, centred on the origin, under `policy`, on the 14 Bohr cube at h.
fn h2_scene(r: f64, h: f64, policy: IonicPolicy) -> Scene {
    pseudo_scene(vec![Nucleus::new(1, [-0.5 * r, 0.0, 0.0]), Nucleus::new(1, [0.5 * r, 0.0, 0.0])], h, BOX, policy)
}

/// Spec §6.6 (NLCC ruling, Task 11: "H₂ under All gets the hard §6.6 gate"; single-H floor 4.1e-7 Ha/Bohr, Task 6): the
/// relaxed pseudo-H₂ (no core, no partial core: the tables are the only egg-box source), f64-energy instrument,
/// wall-subtracted, at the coarsest h H certifies capped at 0.25 — PASS asserted (G + 2u_I < 3e-5) with the instrument's
/// qualification; the app default 0.15 reported beside it.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2_all_eggbox_passes() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[1], 0.25);
    let base = relaxed(&ctx, &format!("h2/all/{h}"), || h2_scene(1.40, h, IonicPolicy::All));
    eprintln!("H₂ All egg-box at the relaxed geometry (R {:.4} Bohr; relaxed at h = {h})", units::angstrom_to_bohr(dist_angstrom(&base.scene, 0, 1)));
    for (hp, gate) in [(h, true), (0.15, false)] {
        if !gate && (hp - h).abs() < 1e-12 { continue; }
        // a hard PASS at the gate h (core-free: spec §6.6 "Core-free systems: PASS is asserted"); report at 0.15
        let rec = gate.then_some((f64::NAN, f64::NAN, true));
        let (p, _) = probe_with(&ctx, "h2-all-relaxed", &base.scene, hp, None, rec).unwrap_or_else(|e| panic!("H₂ All at h = {hp}: {e}"));
        probe_row(&ctx, &format!("h2_all_eggbox ({})", if gate { "gated" } else { "reported" }), "H₂ (All)", &p);
        assert!(p.g.is_finite());
    }
}

/// Spec §6.6 (review M5) under `All`, the brief's test (Task 6 review addition: wall subtraction and its assertions): the
/// relaxed H₂O of `h2o_angle_within_2deg_of_lsda`, the rigid-shift probe with the f64-energy instrument (`eggbox`; never the
/// GPU HF ΣF, which is NOT QUALIFIED, spec §6.6 "Instrument"), wall-subtracted — raw, wall and corrected printed, the box
/// wall ≤ 3e-4 and u_I ≤ 5e-6 asserted — at the coarsest h every species certifies capped at 0.25 (O: 0.20) and at the app
/// default 0.15 beside it; the y-component reported against today's all-electron −3.4e-4 (like for like 10×); the energy
/// ripple from the 8 shifts {0, h/2}³, wall-subtracted (plan review M4).
/// Per the outcome (NLCC ruling Ruling 6: H₂O under `All` has O's partial core; "keep
/// `h2o_all_max_shift_sum_of_forces_below_3e_5` only if it passes; otherwise `h2o_all_eggbox_recorded` with the ratchet"):
/// the first run (Task 11, 2026-10-18) MISSED at 0.20 — G 4.153e-5 Ha/Bohr, u_I 1.91e-8, wall max 4.22e-5, raw 7.61e-5 — so
/// this is the renamed test: the instrument and the ratchet G ≤ G_rec + 2(u_I + u_I,rec) at 0.20 asserted; at 0.15 it
/// PASSED (G 4.720e-6, u_I 1.56e-8; the {0, h/2}³ ripple 1.1e-7 Ha wall-subtracted), asserted as the hard PASS since
/// (`EGGBOX_RECORD`). What remains at 0.20 is O's (the CoresOnly water misses there too,
/// 3.997e-5; the core-free H₂ under `All` passes with G 2.8e-6, `h2_all_eggbox_passes`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_all_eggbox_recorded() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let h = h_for(&[8, 1], 0.25);
    let base = relaxed(&ctx, &format!("h2o/all/{h}"), || water(IonicPolicy::All, h));
    eprintln!("H₂O All egg-box at the relaxed geometry (∠HOH {:.3}°, r(O–H) {:.4} Å; relaxed at h = {h})", angle_deg(&base.scene, 0, 1, 2), dist_angstrom(&base.scene, 0, 1));
    for hp in [h, 0.15] {
        let (p, setup) = probe_with(&ctx, "h2o-all-relaxed", &base.scene, hp, Some(1), record_for("h2o-all-relaxed", hp))
            .unwrap_or_else(|e| panic!("H₂O All at h = {hp}: {e}"));
        let (raw, cor, es) = eggbox::ripple(&ctx, &p, BOX, setup, &rigid(&base.scene), Some(1));
        eprintln!("H₂O All at h = {hp}: G {:.3e} (raw max {:.3e}, wall max {:.2e}; u_I {:.2e}) — {}; max |ΣF_y| {:.2e} (today's all-electron ΣF_y −3.4e-4: {:.1}×); E ripple over {{0, h/2}}³ rms raw {raw:.2e}, wall-subtracted {cor:.2e} Ha ({})",
                  p.g, p.raw_max, p.wall_max, p.u, p.verdict, p.fy_max, eggbox::AE_FIGURE / p.fy_max, es.iter().map(|x| format!("{x:+.2e}")).collect::<Vec<_>>().join(", "));
        probe_row(&ctx, "h2o_all_eggbox", "H₂O (All)", &p);
        assert!(p.g.is_finite() && cor.is_finite());
    }
}

/// Spec §6.7: `thermostat_h2o_preset_known_limitation` (dynamics suite, ignored under `None`) becomes a gate under `All` —
/// the `h2o` preset (physical geometry: O–H 1.81 Bohr, 104.5°) is within 0.02 Ha of the relaxed minimum (the relaxed water of
/// `h2o_angle_within_2deg_of_lsda`, same h, box and tolerance), 600 NVE steps of 0.25 fs from rest stay below 1000 K, and no
/// warm SCF needed a cold retry (the 0.34 Ha/Bohr preset forces were the smeared surface's). NVE at the interactive
/// tolerance 1e-5 (the dynamics suite's).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test pseudo_oracles")]
fn h2o_preset_dynamics_gate() {
    let Some(ctx) = gpu("pseudo_oracles") else { return };
    let t0 = Instant::now();
    let h = h_for(&[8, 1], 0.25);
    let preset = pseudo_scene(vec![Nucleus::new(8, [0.0, 0.0, 0.0]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])], h, BOX, IonicPolicy::All);
    let min = relaxed(&ctx, &format!("h2o/all/{h}"), || water(IonicPolicy::All, h));
    let (mut p, _) = Locked::new(ctx.clone(), TOL);
    let r0 = solve(&mut p, &preset, "preset");
    let de = r0.energy - min.energy;
    let fmax = sim::relax::max_norm(&r0.forces);
    eprintln!("h2o preset under All at h = {h} ({}³): E − E_min = {de:.5} Ha (E {:.6}, E_min {:.6}), max|F| {fmax:.3e} Ha/Bohr at the preset (the smeared surface's: 0.34)", preset.cell.n, r0.energy, min.energy);
    assert!(de <= 0.02, "E_preset − E_min = {de:.4} Ha");
    let (p, log) = Locked::new(ctx.clone(), 1e-5);
    let mut d = sim::Dynamics::new(preset, Box::new(p), DT_FS, SEED);
    let s0 = d.snapshot();
    assert!(s0.last_error.is_none(), "{:?}", s0.last_error);
    let (mut t_max, mut e) = (0.0f64, Vec::new());
    for i in 1..=600 {
        d.apply(sim::DynCommand::Step(1));
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(s.last_error.is_none(), "step {i}: {:?}", s.last_error);
        assert!(!s.eval.notes.iter().any(|n| n.contains("retried cold")), "step {i}: a warm SCF stalled: {:?}", s.eval.notes);
        t_max = t_max.max(s.temperature);
        e.push(s.total);
    }
    let last = d.snapshot();
    let log = log.lock().unwrap().clone();
    let spread = e.iter().cloned().fold(f64::MIN, f64::max) - e.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("h2o preset NVE from rest (All, h = {h}, tol 1e-5, {DT_FS} fs × 600): max T {t_max:.0} K (bound 1000), E_total spread {spread:.2e} Ha, final ∠HOH {:.2}°, r(O–H) {:.4} Å; {}; {:.0} s",
              angle_deg(&last.scene, 0, 1, 2), dist_angstrom(&last.scene, 0, 1), log.summary(), t0.elapsed().as_secs_f64());
    record(&ctx, timing_row("h2o_preset_dynamics_gate", "H₂O preset (All)", last.scene.cell.n, 1e-5, &log, t0.elapsed(), &format!("ΔE {de:.4} Ha, max T {t_max:.0} K")));
    assert!(log.cold_retries.is_empty(), "cold retries: {:?}", log.cold_retries);
    assert!(t_max < 1000.0, "max T {t_max:.0} K");
}
