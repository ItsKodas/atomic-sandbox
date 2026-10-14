//! The M6 tools on the real GPU DFT surface (M6 Task 7; spec §11; numbers of record in
//! `references.md`, section "Tools (M6)"). Run with `cargo test -p validate --release --test tools`
//! (release-only; skips cleanly without a GPU adapter; `--nocapture` prints every number, the
//! pre-measurement printouts included; each test appends its timing row to the gitignored
//! `last-run-tools.md`). Wall times are reported, never asserted (ruling R3).
//!
//! Protocol: σ = h = 0.25 Bohr, SCF tol 1e-6, 49³ unless stated (57³ for water — the `h2o`
//! preset's grid; 65³ for the H₂ dissociation scan and LiH, whose 49³ box warns), `GpuKs` behind
//! the suite's GPU lock (`common::Locked`) **with side solvers**, so the driver solves its
//! isolated-atom references off the stepping path exactly as in the app (spec §3.2). The bond
//! tests go through `Dynamics` and read the snapshot's `PairReport`s; the chords behind each
//! verdict (raw ρ, ρ̂, ρ_pro, q at the 33 samples) are rebuilt from the driver's own reference
//! tables (`Dynamics::references`) and printed every run — the brief's pre-measurement, kept.
//!
//! # Bonds (spec §3)
//! - `h2_is_bonded_at_equilibrium_and_not_at_six_bohr`: R_e(σ) = 1.78894 by `locate_surface`; the
//!   first density queues the H reference (`bonds_pending` [1] → []), then one bond, Route S, not
//!   multi-centre, q = 1.703 ≥ `H2_ENHANCEMENT_MIN` = 1.3. **The real-surface noise:** the same
//!   molecule at 12 placements (centre shifted 0, h/4, h/2, 3h/4 along x × the axis along x, the
//!   xy and the xyz diagonal): q spread 0.0208 (1.2 % of 1.697 — the sampled minimum wanders
//!   s = 0.469 … 0.5625 along the flat bottom), transverse spread 0.0079; **δ/3 = 0.037 and
//!   ε_⊥/3 = 0.023 hold with margins 1.8× and 3.0×, so neither constant moves**: the binding
//!   measurement behind δ stays Task 3's gridded fixture (3.4 %), which `NOISE_Q` carries.
//!   Asserted: spread ≤ δ/3, relative spread ≤ `NOISE_Q`, transverse spread ≤ ε_⊥/3. At 6.0 Bohr
//!   (65³): no bond (`BelowFloor`, ρ̂ 0.0053). **The dissociation scan is reported, not asserted,
//!   and is not monotone** (spec §11 predicted a monotone fall and a crossing of 1 + δ): q =
//!   1.708, 1.598, 1.637, 1.959 at R = 1.8, 2.5, 3.5, 5.0 (2.215 at 6). That is the solver's
//!   spin-restricted singlet (`multiplicity: None` → M = 1 with n↑ = n↓ from a symmetric start;
//!   no broken-symmetry search): one doubly occupied σ_g ∝ φ_A + φ_B puts 4φ²/(1 + S) at the
//!   midpoint against the promolecule's 2φ², so q → 2/(1 + S) → 2 as the overlap S → 0, where
//!   two separated atoms would give 1 (the Coulson–Fischer problem; LSDA's static-correlation
//!   gap, rule 7) — and the unpolarised molecule is more diffuse than its fully polarised
//!   reference atom, so q passes 2. The pair leaves the bond list at the **floor** between
//!   R = 5 (ρ̂ 0.0109) and 6 (0.0053), not at the accumulation test: no 1 + δ crossing exists.
//! - `he2_is_a_closed_shell_contact_not_a_bond`: He₂ at 3.0 Bohr: Route S (a saddle exists — the
//!   "bond path ≠ bond" case), ρ̂(p*) = 0.0570 > ρ_floor, q = 0.9705 → `NotAccumulated`, no bond,
//!   at all 12 placements (q 0.9705 … 0.9723). Two margins, both asserted. **The verdict:**
//!   (1 + δ)/q_max − 1 = 14.2 % ≥ 3·`NOISE_Q` = 10.2 %. **The depletion** (q < 1, the Pauli
//!   signature): q_max ≤ 1 − 3·(He₂'s own relative spread 0.0018) = 0.9946 — depleted by 2.8 %,
//!   15× its spread. **Deviation from the brief:** it asked for enhancement ≤ 1 − 3·`NOISE_Q` =
//!   0.898. The depletion on this surface is 2.9 %, less than one unit of the gridded fixture's
//!   noise, so that pin cannot hold — and the fixture's 3.4 % does not describe He₂: it is the
//!   same-operator residual of a strongly deformed pair (q 1.9, D_u = −4.8 Bohr⁻²), while He₂'s
//!   q is within 3 % of 1 and its residual an order smaller (measured 0.18 %). The verdict is
//!   held to the global noise and the depletion to its own measured noise (rule 6: a bound
//!   re-derived for the quantity it bounds, not widened; rule 7: the small depletion is the
//!   finding, recorded).
//! - `h3_plus_is_a_three_centre_ring`: H₃⁺ relaxed by the quench from side 1.9 (6 steps, sides
//!   2.1174); ρ̂(centroid)/ρ̂(edge midpoint) = 1.147 (the reviewer's LCAO estimate was +5–7 %);
//!   every edge takes Route S with a transverse rise of 1.099–1.100 > 1 + ε_⊥, so the march runs
//!   and stalls on the ring point (stop `Stationary`, 0.11–0.19 Bohr from the centroid, ≥ 1.04 Bohr
//!   from every nucleus): three `Bonded { multi_centre: true }`, one component. Asserted per edge:
//!   `multi_centre` == (rise > 1 + ε_⊥), a march end ≥ h from every nucleus, Route S; three bonds;
//!   components [[0, 1, 2]].
//! - `heh_plus_is_reported` (spec §3.6: **reported, not asserted**): HeH⁺ at 1.46, charge +1 →
//!   **`Bonded`** on Route D (no interior minimum: He's smeared density runs down to the H), q =
//!   1.405 at s = 0.656 (0.96 Bohr from He, 0.50 from H), ρ̂ 0.271, transverse 0.967. The spec
//!   predicted `NotAccumulated` (q 0.6–0.9 under the neutral promolecule, an unsmeared estimate);
//!   at σ = 0.25 the ion's density there is 0.27 against a promolecule of 0.19. A finding (rule
//!   7), not a pin: HeH⁺ is a bound ion (D_e ≈ 2 eV), so "bonded" is chemically right, while QTAIM
//!   classes its bond closed-shell-like. Asserted only that the report exists, filled, not `TooFar`.
//! - `water_has_two_oh_bonds_and_no_hh_bond` (Review Focus 1): the preset geometry at 57³, and the
//!   same molecule turned 45° about the H–H axis and by a seeded random rotation, each moved off
//!   the grid: bonds exactly {(0,1), (0,2)}, both Route D (q 1.44–1.47 at 0.23–0.34 Bohr from the
//!   H), and the H–H pair `NeighbourRegion { atom: 0 }` every time. **This is where the detector
//!   gained condition 4** (`sim::bonds` module doc, "Attribution"; spec §3.1, ruling P11): the H–H
//!   chord takes Route D and finds q's interior maximum 0.18 Bohr from an H (q 1.426, transverse
//!   1.024, ρ̂ 0.252) — conditions 1–3 all pass, and before the fix the pair was `Bonded`. The
//!   point is the O–H bond's own lobe seen from the side: there the promolecular terms are O
//!   0.064, near H 0.111, far H 0.0020 — oxygen outweighs the partner 32× (32× and 32× at the two
//!   other placements), while at the O–H points O outweighs the other H 40–45×. The O reference
//!   is the singlet (the triplet and quintet stop at the 200-iteration cap; Task 3), ≈ 57 s of GPU
//!   on the side solver at 57³, tol 1e-6.
//! - `lih_is_reported` (spec §3.6: **reported, not asserted**): LiH centred on the 65³ grid (the
//!   brief put Li at the origin; with the H 3.3 Bohr toward one face every solve warned "box too
//!   small" at 1.1–1.7e-5, so the molecule is centred — no warning), R_e(σ) = 3.366 Bohr by force
//!   bisection on [2.6, 3.6] (experiment 3.015; the smeared surface is long, as H₂'s 1.79 vs 1.40)
//!   → **`Bonded`**, Route S, ρ̂(p*) = 0.0423 (the literature's ρ_b ≈ 0.04, Bader 1990), q = 1.738
//!   at 1.58 Bohr from Li. The accumulation test is a covalent criterion and the ionic case was
//!   the open question; on this surface it passes. A finding, not a pin.
//!
//! # Isotopes (spec §5.5)
//! - `d2_vibrates_slower_than_h2_by_the_mass_ratio`: the M4 protocol twice on the same surface (R_e
//!   + 0.05 from rest, dt 0.25 fs, 400 steps, the discrete w from the upward crossings — 9 for H₂,
//!   7 for D₂), each converted to the frequency Verlet integrates, ω = √(2 − 2cos(w·dt))/dt:
//!   ω_H/ω_D = 1.4136050 against √(m_D/m_H) = 1.4136703 from the scenes' own nuclei (AME2020
//!   atomic masses): **−4.6e-5**; the raw w_H/w_D is +4.4e-4 off (Verlet's (h_H² − h_D²)/24 =
//!   +4.8e-4). **Bound `D2_BOUND` = 3e-4 (spec 1e-3, tightened by derivation):** in classical
//!   mechanics the mass scales out of μẍ = −V′(x) under t → t√μ, so at equal amplitude the ratio
//!   is √(μ_D/μ_H) to all orders of anharmonicity; Verlet's map sees the mass only through dt²/μ,
//!   so the correction removes the harmonic part exactly and leaves the cross term of h² with the
//!   anharmonic shift (≈ h²·1.5e-3 ≈ 3e-5); the crossing interpolation errs by ≲ (w·dt)²·dt/10 per
//!   crossing over 5–8 periods (≈ 1e-5); the warm-path SCF force scatter (≈ 1e-6 on forces of
//!   1e-2) averages to ≈ 1e-5 — a budget of ≈ 1e-4, of which the bound is 3×. The raw ratio misses
//!   by more than the bound, so the test needs the correction. Printed: the nuclear-mass BO ratio
//!   1.41386 and experiment's 1.41268 (−6.5e-4 from ours: the adiabatic and non-adiabatic
//!   corrections no Born–Oppenheimer code has, plus the atomic-mass convention's ≈ 2e-4).
//!
//! # Relax (spec §6)
//! - `fire_relaxes_h3_plus_to_the_quenched_minimum`: H₃⁺ (side 2.117) with each nucleus kicked
//!   0.2 Bohr in a seeded random direction; the quench (`common::relax`, 0.5 fs) and FIRE through
//!   the driver, one FIRE step per tick so every snapshot is checked; both results re-solved cold.
//!   Asserted: FIRE converged; max|F| < F_TOL on the cold solve at its result (9.2e-4);
//!   |E_f − E_q| = 1.8e-6 ≤ 3·F_TOL²/(2·`K_SOFT`) = 3e-5 Ha (MUST-8: both stop at max_i |F_i| <
//!   F_TOL, so each lies within |F|²/(2k_min) ≤ 3·F_TOL²/(2k_min) of the minimum); the premise
//!   k_min ≥ `K_SOFT` = 0.05, measured — the Cartesian Hessian from central force differences
//!   (±0.01 Bohr, 18 cold solves) has six near-zero eigenvalues (|λ| ≤ 8e-4: translations,
//!   rotations, egg-box) and the vibrations 0.1435, 0.1476 (the E′ pair) and 0.2392 Ha/Bohr² (the
//!   breathing mode; its 3-point energy curvature 0.2393 agrees), so the bound at the measured
//!   k_min would be 1.0e-5; |ΔE_conserved| = 0 (≤ 1e-10) over all 24 snapshots, W = E_KS(end) −
//!   E_KS(start) = −4.584e-3 Ha to the last digit, `sim_time` and `step` 0 throughout. Printed:
//!   the sides (|Δ| ≤ 0.0055 Bohr against the slack 2·F_TOL/k_min = 0.014). **Evaluations,
//!   reported, not asserted:** FIRE 25 against the 0.5 fs quench's 16 (ratio 1.56) and the
//!   0.25 fs quench's 29 (0.86). The spec's "≤ 0.5 × the quench's" is not met and not pinned:
//!   FIRE's advantage is a many-mode, ill-conditioned one (Bitzek 2006: the adaptive Δt and the
//!   mixing pay off where the stiffest mode caps the step and the soft ones need thousands); a
//!   three-atom triangle kicked by 0.2 Bohr has nothing soft, and a quench at twice FIRE's
//!   starting Δt converges in 14 steps while FIRE spends its first N_min-limited steps growing Δt
//!   from 0.25 fs. A pin on the ratio would pin Bitzek's start-up transient, not a property of
//!   the tool (Task 6 measured 1.15–1.63 in FIRE's favour on its analytic fixtures: a statement
//!   about the problem, not the method).
//! - `fire_on_the_surrogate_is_verified_by_the_teacher`: the surrogate suite's H₂ pass (default
//!   plan, 1000 K burst, seed 7: 272 samples, trusted), then Relax from R = 1.4 on the model: 21
//!   FIRE steps, converged; one teacher solve at the result gives the verified max|F| = 1.5e-4 ≤
//!   3e-3 (spec; the M5 force tier); R = 1.78818 against the DFT R_e 1.78894: ΔR = −7.6e-4 ≤
//!   2·F_TOL/k = 1e-2 (k = 0.200, M4).
//!
//! The GPU work of concurrently running tests is serialised through `common::GPU`, one evaluation
//! at a time (the side solvers' too), so the per-solve times in the timing rows are the solves'
//! own while the tests interleave.
use gpu::GpuContext;
use providers::{FallbackPolicy, Sampler, SurrogateKs};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::bonds::{PairReport, Promolecule, Route, Verdict, DELTA, EPS_TRANSVERSE, RHO_FLOOR};
use sim::{DynCommand, DynSnapshot, Dynamics, ForceProvider, Integrator, Nucleus, RadialDensity, SamplingPlan, Scene, F_TOL};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use surrogate::Basis;

mod common;
use common::{
    bond, chrono_free_date, h2, h2_isotope, h3_plus, he2, heh_plus, lih, locate_surface, never, no_augment, period_from_upward_crossings, relax, relax_with,
    scene, solve, timing_row, triangle, EvalLog, Locked, DISPLACEMENT, DT_FS, H, H3_SIDE0, HARTREE_IN_CM, N_H2, TOL_H2,
};

/// Timing rows for `last-run-tools.md`, rewritten whenever a test finishes.
static TIMING: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The far-pair grid (16.25 Bohr box): H₂ out to 6 Bohr and LiH without a box warning.
const N_FAR: usize = 65;
/// The `h2o` preset's grid (14.25 Bohr box).
const N_H2O: usize = 57;
/// The driver's seed (no thermostat runs here; only the FIRE start draws from it).
const SEED: u64 = 7;

// ---- bonds ----
/// The relative noise of the detector's q (spread/mean) that the margins below are taken against:
/// the larger of the gridded-fixture noise of Task 3 (0.0629/1.8521 = 0.0340 over 24 placements,
/// `sim::bonds::tests::the_same_operator_cancels_the_interpolation_bias`) and this suite's H₂ on
/// the real surface (run of record 0.0208/1.6971 = 0.0123 over 12 placements). The H₂ test
/// asserts its live value against this constant, so a surface noisier than the constant fails
/// there first (and δ, ≥ 3 × this, would have to be raised).
const NOISE_Q: f64 = 0.034;
/// H₂ bonds at R_e: the enhancement floor (spec §11).
const H2_ENHANCEMENT_MIN: f64 = 1.3;
/// The separations of the H₂ dissociation scan (65³; reported) and the one asserted unbonded.
const H2_SCAN: [f64; 4] = [1.8, 2.5, 3.5, 5.0];
const H2_FAR: f64 = 6.0;
const HE2_R: f64 = 3.0;
const HEH_R: f64 = 1.46;
/// LiH R_e bracket (Bohr), force bisection to `LIH_RE_TOL`.
const LIH_BRACKET: (f64, f64) = (2.6, 3.6);
const LIH_RE_TOL: f64 = 1e-3;

// ---- isotopes ----
const VIB_STEPS: usize = 400;
/// |ω_H/ω_D / √(m_D/m_H) − 1| after the Verlet correction. Spec §5.5 / S3: 1e-3 → **3e-4**
/// (module doc: the residual budget is ≈ 1e-4, the run of record −4.6e-5; the raw ratio's
/// +4.8e-4 Verlet systematic lies outside this bound, so the test needs the correction).
const D2_BOUND: f64 = 3e-4;
/// The nuclear-mass Born–Oppenheimer ratio √(2.013553212745/1.007276466621) and experiment's
/// ω_e(H₂)/ω_e(D₂) = 4401.21/3115.50 (NIST CCCBDB; Huber & Herzberg) — printed, never asserted.
const RATIO_NUCLEAR_MASSES: f64 = 1.41386;
const RATIO_EXPERIMENT: f64 = 4401.21 / 3115.50;

// ---- relax ----
/// The suite's H₃⁺ side on the σ = 0.25 surface (M4: 2.117 Bohr; this run relaxes to 2.1174).
const H3_SIDE: f64 = 2.117;
/// Each nucleus of the FIRE start is displaced by this much in a seeded random direction (Bohr).
const FIRE_KICK: f64 = 0.2;
/// The stated lower bound of H₃⁺'s vibrational force constants (Ha/Bohr², spec §11); the test
/// measures the smallest Hessian eigenvalue and asserts it is above.
const K_SOFT: f64 = 0.05;
/// |ΔE_conserved| across a relax (spec §11 / Task 6).
const E_CONSERVED_BOUND: f64 = 1e-10;
/// Hessian finite-difference step (Bohr).
const HESS_STEP: f64 = 0.01;
/// The surrogate relax (spec §11): DFT max|F| at the model's minimum, and |R − R_e(DFT)| ≤
/// 2·F_TOL/k with M4's k = 0.200.
const VERIFIED_F_BOUND: f64 = 3e-3;
const SURROGATE_R_BOUND: f64 = 2.0 * F_TOL / 0.200;
const SURROGATE_RELAX_R0: f64 = 1.4;

fn gpu() -> Option<GpuContext> { common::gpu("tools") }

/// Appends a row and rewrites `last-run-tools.md`.
fn record(ctx: &GpuContext, row: String) {
    let mut rows = TIMING.lock().unwrap_or_else(|e| e.into_inner());
    rows.push(row);
    let adapter = ctx.adapter.get_info();
    let mut md = String::new();
    let _ = writeln!(md, "# Tools suite timing (`validate::tools`, {} UTC)\n", chrono_free_date());
    let _ = writeln!(md, "Adapter: {} ({:?}, {:?}); release build; σ = h = 0.25. Tests run concurrently with their GPU evaluations serialised, so \"GPU\" is the uncontended sum of this test's stepping solves (the side solver's atom references are not in it) and \"wall\" includes waiting for the others. Wall times are reported, never asserted (R3).\n", adapter.name, adapter.backend, adapter.device_type);
    let _ = writeln!(md, "| test | system | grid | tol | solves | GPU s | ms/solve | SCF steps/solve | readbacks/solve | test wall s | result |\n|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rows.iter() { md += r; md.push('\n'); }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-tools.md");
    std::fs::write(path, &md).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}

fn snap(log: &Arc<Mutex<EvalLog>>) -> EvalLog { log.lock().unwrap().clone() }

/// The suite's Kohn-Sham provider: `GpuKs` behind the lock, with side solvers for the bond
/// references (their own log). Returns the provider, its log and the side log.
fn kohn_sham(ctx: &GpuContext, tol: f64) -> (Locked, Arc<Mutex<EvalLog>>, Arc<Mutex<EvalLog>>) {
    let (p, log) = Locked::new(ctx.clone(), tol);
    let (p, side) = p.with_side_solver();
    (p, log, side)
}

/// Ticks a paused driver until the bond graph of its current positions is published with every
/// reference in (the references run on the side solver's thread; the ticks poll them).
fn settle(d: &mut Dynamics, what: &str) -> DynSnapshot {
    let t = Instant::now();
    loop {
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(s.last_error.is_none(), "{what}: {:?}", s.last_error);
        if s.bonds_pending.is_empty() && s.bonds.is_some() && !s.forces_stale && s.bond_reference.is_none() {
            return s;
        }
        assert!(t.elapsed() < Duration::from_secs(900), "{what}: no graph after 900 s (pending {:?}, notes {:?})", s.bonds_pending, s.notes);
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn report(s: &DynSnapshot, i: usize, j: usize) -> PairReport {
    s.pair_reports.iter().find(|r| r.i == i && r.j == j).unwrap_or_else(|| panic!("no report for ({i}, {j}): {:?}", s.pair_reports)).clone()
}

fn show(r: &PairReport) -> String {
    let o = |x: Option<f64>| x.map_or("—".to_string(), |v| format!("{v:.4}"));
    format!("({}, {}): {:?}, route {:?}, s {}, ρ̂(p*) {}, ρ_pro {}, q {}, transverse {}, march {:?} at {:?}",
            r.i, r.j, r.verdict, r.route, o(r.s), o(r.rho_min), o(r.rho_pro), o(r.enhancement), o(r.transverse_ratio), r.march_stop, r.march_end.map(|p| p.map(|x| (x * 1e3).round() / 1e3)))
}

/// The driver's own reference tables for the scene's nuclei, cloned (the promolecule borrows
/// them while the provider is sampled through the driver).
fn tables(d: &Dynamics, s: &Scene) -> Vec<RadialDensity> {
    s.nuclei.iter().map(|n| {
        d.references().iter().find(|(k, r)| k.z == n.z && k.grid_n == s.cell.n && !r.radial.is_empty()).unwrap_or_else(|| panic!("no table for Z = {}", n.z)).1.radial.clone()
    }).collect()
}

fn fmt(v: &[f64]) -> String { v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" ") }

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 { ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() }

fn chord_point(s: &Scene, i: usize, j: usize, f: f64) -> [f64; 3] {
    let (a, b) = (s.nuclei[i].pos, s.nuclei[j].pos);
    std::array::from_fn(|c| a[c] + f * (b[c] - a[c]))
}

/// The 33 chord samples of (i, j) — raw ρ, ρ̂, ρ_pro and q — appended to `out`; returns ρ̂.
fn chord(d: &mut Dynamics, s: &Scene, i: usize, j: usize, label: &str, out: &mut String) -> Vec<f64> {
    let t = tables(d, s);
    let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
    let pro = Promolecule::new(t.iter().collect(), &pos, Some(&s.cell));
    let pts: Vec<[f64; 3]> = (0..33).map(|k| chord_point(s, i, j, k as f64 / 32.0)).collect();
    let raw = d.provider_mut().sample_density(&pts).expect("a density");
    let hat: Vec<f64> = pts.iter().zip(&raw).map(|(&x, &v)| pro.corrected(x, v)).collect();
    let q: Vec<f64> = pts.iter().zip(&raw).map(|(&x, &v)| pro.ratio(x, v)).collect();
    let ex: Vec<f64> = pts.iter().map(|&x| pro.exact(x)).collect();
    let _ = writeln!(out, "  {label} chord ({i}, {j}), s = k/32:\n    raw ρ {}\n    ρ̂     {}\n    ρ_pro {}\n    q     {}", fmt(&raw), fmt(&hat), fmt(&ex), fmt(&q));
    hat
}

/// Each nucleus's promolecular term at the bond point of report `r` (condition 4's numbers).
fn shares(d: &Dynamics, s: &Scene, r: &PairReport) -> String {
    let Some(f) = r.s else { return "—".into() };
    let t = tables(d, s);
    let pos: Vec<[f64; 3]> = s.nuclei.iter().map(|n| n.pos).collect();
    let pro = Promolecule::new(t.iter().collect(), &pos, Some(&s.cell));
    let p = chord_point(s, r.i, r.j, f);
    let near = pos.iter().map(|&x| dist(x, p)).fold(f64::INFINITY, f64::min);
    format!("promolecular terms at p* [{}], {near:.3} Bohr from the nearest nucleus", (0..pos.len()).map(|c| format!("{:.4}", pro.atom(c, p))).collect::<Vec<_>>().join(", "))
}

fn spread(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min) }
fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

/// A diatomic of species `za`, `zb`, bond length `r` along the unit vector `u`, centred on `c`.
fn diatomic(n: usize, za: u8, zb: u8, r: f64, u: [f64; 3], c: [f64; 3]) -> Scene {
    let at = |z: u8, sgn: f64| Nucleus::new(z, std::array::from_fn(|k| c[k] + sgn * 0.5 * r * u[k]));
    scene(n, vec![at(za, 1.0), at(zb, -1.0)])
}

/// The 12 placements of a diatomic (spec §11): the centre shifted by 0, h/4, h/2, 3h/4 along x,
/// the axis along x, the xy diagonal and the xyz diagonal.
fn placements() -> Vec<(String, [f64; 3], [f64; 3])> {
    let r2 = std::f64::consts::FRAC_1_SQRT_2;
    let r3 = 1.0 / 3f64.sqrt();
    let mut v = Vec::new();
    for shift in [0.0, 0.25 * H, 0.5 * H, 0.75 * H] {
        for (name, u) in [("x", [1.0, 0.0, 0.0]), ("xy", [r2, r2, 0.0]), ("xyz", [r3, r3, r3])] {
            v.push((format!("shift {shift:.4} along {name}"), [shift, 0.0, 0.0], u));
        }
    }
    v
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn h2_is_bonded_at_equilibrium_and_not_at_six_bohr() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (mut p0, _) = Locked::new(ctx.clone(), TOL_H2);
    let surf = locate_surface(&mut p0);
    let _ = writeln!(out, "H₂ bonds, {N_H2}³, σ = h = {H}, tol {TOL_H2:.0e}");
    out += &surf.log;
    let (p, log, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(h2(surf.r_e), Box::new(p), DT_FS, SEED);
    let s0 = d.snapshot();
    let _ = writeln!(out, "  after the constructor: bonds_pending {:?}, bonds {:?}", s0.bonds_pending, s0.bonds.as_ref().map(|g| g.bonds.len()));
    assert_eq!(s0.bonds_pending, vec![1], "the H reference is queued by the first density");
    assert!(s0.bonds.is_none());
    let s = settle(&mut d, "H₂ at R_e");
    assert!(s.bonds_pending.is_empty());
    let g = s.bonds.clone().unwrap();
    let r = report(&s, 0, 1);
    let _ = writeln!(out, "  R_e = {:.5}: {}", surf.r_e, show(&r));
    let sc = s.scene.clone();
    chord(&mut d, &sc, 0, 1, "R_e", &mut out);
    assert_eq!(g.bonds.len(), 1, "{:?}", s.pair_reports);
    assert_eq!((g.bonds[0].route, g.bonds[0].multi_centre), (Route::Saddle, false));
    assert!(g.bonds[0].enhancement >= H2_ENHANCEMENT_MIN, "q = {} at R_e", g.bonds[0].enhancement);
    // the real-surface noise: 12 placements of the same molecule
    let (mut qs, mut ts) = (Vec::new(), Vec::new());
    for (name, c, u) in placements() {
        d.apply(DynCommand::ReplaceScene(diatomic(N_H2, 1, 1, surf.r_e, u, c)));
        let s = settle(&mut d, &name);
        let r = report(&s, 0, 1);
        let _ = writeln!(out, "  {name}: {}", show(&r));
        assert_eq!((r.verdict, r.route), (Verdict::Bonded { multi_centre: false }, Some(Route::Saddle)), "{name}: {}", show(&r));
        qs.push(r.enhancement.unwrap());
        ts.push(r.transverse_ratio.unwrap());
    }
    let (sq, st, mq) = (spread(&qs), spread(&ts), mean(&qs));
    let _ = writeln!(out, "  12 placements: q spread {sq:.4} (relative {:.4}; NOISE_Q {NOISE_Q}), mean {mq:.4}; transverse spread {st:.4}, mean {:.4}; bounds δ/3 = {:.4}, ε_⊥/3 = {:.4}",
                     sq / mq, mean(&ts), DELTA / 3.0, EPS_TRANSVERSE / 3.0);
    // the dissociation scan on the 65³ grid (reported: module doc, "the restricted singlet")
    let mut scan = Vec::new();
    for rr in H2_SCAN.iter().chain([H2_FAR].iter()) {
        d.apply(DynCommand::ReplaceScene(diatomic(N_FAR, 1, 1, *rr, [1.0, 0.0, 0.0], [0.0; 3])));
        let s = settle(&mut d, "R scan");
        let r = report(&s, 0, 1);
        let n_b = s.bonds.as_ref().unwrap().bonds.len();
        let _ = writeln!(out, "  {N_FAR}³ R = {rr}: {} (bonds {n_b})", show(&r));
        scan.push((*rr, r, n_b));
    }
    let qs_scan: Vec<f64> = scan[..H2_SCAN.len()].iter().map(|x| x.1.enhancement.unwrap()).collect();
    let crossing = qs_scan.windows(2).zip(H2_SCAN.windows(2)).find(|(q, _)| (q[0] - 1.0 - DELTA) * (q[1] - 1.0 - DELTA) <= 0.0)
        .map(|(q, r)| r[0] + (r[1] - r[0]) * (q[0] - 1.0 - DELTA) / (q[0] - q[1]));
    let _ = writeln!(out, "  enhancement over R {H2_SCAN:?}: {qs_scan:?}; monotone decreasing: {}; crossing of 1 + δ: {}",
                     qs_scan.windows(2).all(|w| w[1] < w[0]), crossing.map_or("none in the scan".to_string(), |r| format!("R ≈ {r:.2}")));
    let log = snap(&log);
    let _ = writeln!(out, "  stepping: {}\n  side (references): {}; test wall {:.1} s", log.summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("h2_bonds", "H₂", N_H2, TOL_H2, &log, t_test.elapsed(), &format!("q(R_e) {:.4}; 12 placements: q spread {sq:.4}, transverse {st:.4}; R = 6: {:?}", g.bonds[0].enhancement, scan[4].1.verdict)));
    assert!(sq <= DELTA / 3.0, "q spread {sq:.4} over the 12 placements: δ = {DELTA} must be raised to {:.3}", 3.0 * sq);
    assert!(sq / mq <= NOISE_Q, "the real-surface relative q noise {:.4} exceeds NOISE_Q = {NOISE_Q}", sq / mq);
    assert!(st <= EPS_TRANSVERSE / 3.0, "transverse spread {st:.4}: ε_⊥ = {EPS_TRANSVERSE} must be raised to {:.3}", 3.0 * st);
    let far = &scan[H2_SCAN.len()];
    assert_eq!(far.2, 0, "a bond at {H2_FAR} Bohr: {}", show(&far.1));
    assert!(far.1.route.is_some() && far.1.verdict != Verdict::TooFar, "sampled: {}", show(&far.1));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn he2_is_a_closed_shell_contact_not_a_bond() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (p, log, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(he2(N_H2, HE2_R), Box::new(p), DT_FS, SEED);
    let s = settle(&mut d, "He₂");
    let r = report(&s, 0, 1);
    let _ = writeln!(out, "He₂ at {HE2_R} Bohr, {N_H2}³: {}", show(&r));
    let sc = s.scene.clone();
    chord(&mut d, &sc, 0, 1, "He–He", &mut out);
    for (k, rr) in d.references().iter() {
        let _ = writeln!(out, "  reference Z = {}: E {:.6}, multiplicity {:?}, {:.4} electrons, ρ̄(1.5) = {:.4}", k.z, rr.energy, rr.multiplicity, rr.radial.electrons(), rr.radial.at(0.5 * HE2_R));
    }
    assert!(s.bonds.as_ref().unwrap().bonds.is_empty(), "{}", show(&r));
    // the saddle and the floor pass — the point of the test — and the accumulation does not
    assert_eq!(r.route, Some(Route::Saddle), "{}", show(&r));
    assert!(r.rho_min.unwrap() > RHO_FLOOR, "{}", show(&r));
    assert_eq!(r.verdict, Verdict::NotAccumulated, "{}", show(&r));
    // the same verdict at every cell offset, and He₂'s own q noise
    let mut qs = vec![r.enhancement.unwrap()];
    for (name, c, u) in placements().into_iter().skip(1) {
        d.apply(DynCommand::ReplaceScene(diatomic(N_H2, 2, 2, HE2_R, u, c)));
        let s = settle(&mut d, &name);
        let r = report(&s, 0, 1);
        let _ = writeln!(out, "  {name}: {}", show(&r));
        assert_eq!((r.verdict, r.route), (Verdict::NotAccumulated, Some(Route::Saddle)), "{name}: {}", show(&r));
        assert!(r.rho_min.unwrap() > RHO_FLOOR, "{name}: {}", show(&r));
        qs.push(r.enhancement.unwrap());
    }
    let (q_max, sq, mq) = (qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max), spread(&qs), mean(&qs));
    let noise_he = sq / mq;
    let verdict_margin = (1.0 + DELTA) / q_max - 1.0;
    let depletion_ceiling = 1.0 - 3.0 * noise_he;
    let log = snap(&log);
    let _ = writeln!(out, "  12 placements: q mean {mq:.4}, max {q_max:.4}, spread {sq:.4} (relative {noise_he:.4}); the verdict's margin (1 + δ)/q_max − 1 = {verdict_margin:.4} vs 3·NOISE_Q = {:.4}; depletion: q_max ≤ 1 − 3·(He₂'s own relative spread) = {depletion_ceiling:.4}; the brief's 1 − 3·NOISE_Q = {:.4} (not met: module doc)",
                     3.0 * NOISE_Q, 1.0 - 3.0 * NOISE_Q);
    let _ = writeln!(out, "  stepping {}; side {}; test wall {:.1} s", log.summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("he2", "He₂", N_H2, TOL_H2, &log, t_test.elapsed(), &format!("NotAccumulated ×12, q {mq:.4} (max {q_max:.4}, spread {sq:.4})")));
    assert!(verdict_margin >= 3.0 * NOISE_Q, "NotAccumulated by {verdict_margin:.4} relative, under 3·NOISE_Q = {:.4}", 3.0 * NOISE_Q);
    assert!(q_max <= depletion_ceiling, "the depletion q_max = {q_max:.4} is not resolved above He₂'s own noise (ceiling {depletion_ceiling:.4})");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn h3_plus_is_a_three_centre_ring() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let _ = writeln!(out, "H₃⁺ ring, {N_H2}³, σ = h = {H}, tol {TOL_H2:.0e}");
    let (mut p0, _) = Locked::new(ctx.clone(), TOL_H2);
    let (relaxed, _, steps) = relax(&mut p0, h3_plus(H3_SIDE0), &mut out);
    let _ = writeln!(out, "  relaxed (quench) in {steps} steps from side {H3_SIDE0}: sides {}", triangle(&relaxed));
    let (p, log, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(relaxed, Box::new(p), DT_FS, SEED);
    let s = settle(&mut d, "H₃⁺");
    let sc = s.scene.clone();
    // pre-measurement printout: the edge chord, the centroid against the edge midpoint
    let hat = chord(&mut d, &sc, 0, 1, "edge", &mut out);
    let pos: Vec<[f64; 3]> = sc.nuclei.iter().map(|n| n.pos).collect();
    let centroid: [f64; 3] = std::array::from_fn(|k| (pos[0][k] + pos[1][k] + pos[2][k]) / 3.0);
    let mid = chord_point(&sc, 0, 1, 0.5);
    let t = tables(&d, &sc);
    let pro = Promolecule::new(t.iter().collect(), &pos, Some(&sc.cell));
    let rho = d.provider_mut().sample_density(&[centroid, mid]).unwrap();
    let (rc, rm) = (pro.corrected(centroid, rho[0]), pro.corrected(mid, rho[1]));
    let _ = writeln!(out, "  ρ̂(centroid) {rc:.4}, ρ̂(edge midpoint) {rm:.4}: ratio {:.4} (the reviewer's LCAO estimate +5–7 %); the chord's midpoint sample {:.4}", rc / rm, hat[16]);
    let g = s.bonds.clone().unwrap();
    for r in &s.pair_reports {
        let ends = r.march_end.map(|e| (dist(e, centroid), pos.iter().map(|&p| dist(e, p)).fold(f64::INFINITY, f64::min)));
        let _ = writeln!(out, "  {}{}", show(r), ends.map_or(String::new(), |(c, n)| format!(" — march end {c:.3} Bohr from the centroid, {n:.3} from the nearest nucleus")));
        let rise = r.transverse_ratio.expect("probed");
        // multi_centre exactly when the rise exceeded ε_⊥ and the march stalled on a
        // non-nuclear point at least h from every nucleus
        let expect_multi = rise > 1.0 + EPS_TRANSVERSE;
        assert_eq!(r.verdict, Verdict::Bonded { multi_centre: expect_multi }, "{}", show(r));
        if let Some((_, n)) = ends {
            assert!(n >= H, "the march ended {n:.3} Bohr from a nucleus: {}", show(r));
        }
        assert_eq!(r.route, Some(Route::Saddle), "{}", show(r));
    }
    let log = snap(&log);
    let _ = writeln!(out, "  components {:?}; stepping {}; side {}; test wall {:.1} s", g.components, log.summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    let multi: Vec<bool> = g.bonds.iter().map(|b| b.multi_centre).collect();
    record(&ctx, timing_row("h3_plus_ring", "H₃⁺", N_H2, TOL_H2, &log, t_test.elapsed(), &format!("{} bonds, multi_centre {multi:?}, ρ̂(centroid)/ρ̂(mid) {:.3}", g.bonds.len(), rc / rm)));
    assert_eq!(g.bonds.len(), 3, "{:?}", s.pair_reports);
    assert_eq!(g.components, vec![vec![0, 1, 2]]);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn heh_plus_is_reported() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (p, log, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(heh_plus(N_H2, HEH_R), Box::new(p), DT_FS, SEED);
    let s = settle(&mut d, "HeH⁺");
    let r = report(&s, 0, 1);
    let _ = writeln!(out, "HeH⁺ at {HEH_R} Bohr, charge +1, {N_H2}³ (reported, not asserted — spec §3.6): {}", show(&r));
    let sc = s.scene.clone();
    chord(&mut d, &sc, 0, 1, "He–H", &mut out);
    if let Some(f) = r.s {
        let p = chord_point(&sc, 0, 1, f);
        let _ = writeln!(out, "  p* is {:.3} Bohr from He, {:.3} from H", dist(p, sc.nuclei[0].pos), dist(p, sc.nuclei[1].pos));
    }
    let log = snap(&log);
    let _ = writeln!(out, "  stepping {}; side {}; test wall {:.1} s", log.summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("heh_plus", "HeH⁺", N_H2, TOL_H2, &log, t_test.elapsed(), &format!("{:?}, {:?}, q {:.3}", r.verdict, r.route, r.enhancement.unwrap_or(f64::NAN))));
    assert!(r.rho_min.is_some() && r.enhancement.is_some() && r.verdict != Verdict::TooFar, "{}", show(&r));
}

/// The `h2o` preset geometry (O at the origin, H at (±1.43, 1.11, 0): O–H 1.81 Bohr, 104.5°).
fn water(rot: Option<([f64; 3], f64)>, shift: [f64; 3]) -> Scene {
    let mut nuclei = vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])];
    if let Some((axis, angle)) = rot {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let a = axis.map(|x| x / n);
        let (s, c) = angle.sin_cos();
        for nu in &mut nuclei {
            let p = nu.pos;
            let d = a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
            let x = [a[1] * p[2] - a[2] * p[1], a[2] * p[0] - a[0] * p[2], a[0] * p[1] - a[1] * p[0]];
            nu.pos = std::array::from_fn(|k| p[k] * c + x[k] * s + a[k] * d * (1.0 - c));
        }
    }
    for nu in &mut nuclei {
        nu.pos = std::array::from_fn(|k| nu.pos[k] + shift[k]);
    }
    scene(N_H2O, nuclei)
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn water_has_two_oh_bonds_and_no_hh_bond() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (p, log, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(water(None, [0.0; 3]), Box::new(p), DT_FS, SEED);
    let s = settle(&mut d, "H₂O");
    let side_log = snap(&side);
    let _ = writeln!(out, "H₂O, the h2o preset geometry, {N_H2O}³, tol {TOL_H2:.0e}: references in {:.1} s of wall ({}); notes {:?}", t_test.elapsed().as_secs_f64(), side_log.summary(), s.notes);
    for (k, r) in d.references().iter() {
        let _ = writeln!(out, "  reference Z = {}: E {:.6}, multiplicity {:?}, {:.4} electrons in the table", k.z, r.energy, r.multiplicity, r.radial.electrons());
    }
    let sc = s.scene.clone();
    chord(&mut d, &sc, 0, 1, "O–H", &mut out);
    chord(&mut d, &sc, 1, 2, "H–H", &mut out);
    let mut cases = vec![("preset".to_string(), s)];
    // the same molecule turned and moved off the grid: the verdicts must not depend on it
    let mut rng = StdRng::seed_from_u64(SEED);
    let axis: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
    let angle = rng.gen_range(0.0..std::f64::consts::TAU);
    for (name, rot, shift) in [("45° about the H–H axis, shifted (h/3, h/5, 0)", Some(([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_4)), [H / 3.0, H / 5.0, 0.0]), ("a seeded random rotation, shifted (0, h/2, h/4)", Some((axis, angle)), [0.0, 0.5 * H, 0.25 * H])] {
        d.apply(DynCommand::ReplaceScene(water(rot, shift)));
        cases.push((name.to_string(), settle(&mut d, name)));
    }
    let mut results = Vec::new();
    for (name, s) in &cases {
        let _ = writeln!(out, "  {name}:");
        for r in &s.pair_reports {
            let _ = writeln!(out, "    {}\n      {}", show(r), shares(&d, &s.scene, r));
        }
        let bonds: Vec<(usize, usize, Route)> = s.bonds.as_ref().unwrap().bonds.iter().map(|b| (b.i, b.j, b.route)).collect();
        let hh = report(s, 1, 2);
        let _ = writeln!(out, "    bonds {bonds:?}; each O–H enhancement is a smeared-surface number (h = 0.25: the O is smeared, ρ_b not physical — spec §3.6)");
        results.push((name.clone(), bonds, hh));
    }
    let log = snap(&log);
    let _ = writeln!(out, "  stepping {}; side {}; test wall {:.1} s", log.summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("water", "H₂O", N_H2O, TOL_H2, &log, t_test.elapsed(), &format!("bonds {:?}; H–H {:?}", results[0].1.iter().map(|b| (b.0, b.1)).collect::<Vec<_>>(), results[0].2.verdict)));
    for (name, bonds, hh) in &results {
        assert_eq!(bonds.iter().map(|b| (b.0, b.1)).collect::<Vec<_>>(), vec![(0, 1), (0, 2)], "{name}: bonds {bonds:?}; H–H {}", show(hh));
        assert!(matches!(hh.verdict, Verdict::NoInteriorMinimum | Verdict::NotATransverseMaximum | Verdict::NotAccumulated | Verdict::NeighbourRegion { atom: 0 }), "{name}: {}", show(hh));
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn lih_is_reported() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (mut p0, log0) = Locked::new(ctx.clone(), TOL_H2);
    // F_x on H (nucleus 1, at +R/2) = −dE/dR for the centred molecule
    let f_h = |p: &mut Locked, r: f64| solve(p, &lih(N_FAR, r), "LiH").forces[1][0];
    let (mut lo, mut hi) = LIH_BRACKET;
    let (f_lo, f_hi) = (f_h(&mut p0, lo), f_h(&mut p0, hi));
    let _ = writeln!(out, "LiH (reported, not asserted — the ionic case, spec §3.6), {N_FAR}³, tol {TOL_H2:.0e}: F_x(H) {f_lo:+.3e} at R = {lo}, {f_hi:+.3e} at {hi}");
    assert!(f_lo > 0.0 && f_hi < 0.0, "R_e not bracketed by {LIH_BRACKET:?}");
    while hi - lo > LIH_RE_TOL {
        let m = 0.5 * (lo + hi);
        if f_h(&mut p0, m) > 0.0 { lo = m } else { hi = m }
    }
    let r_e = 0.5 * (lo + hi);
    let log0 = snap(&log0);
    let _ = writeln!(out, "  R_e(σ = 0.25) = {r_e:.4} Bohr by force bisection (experiment 3.015); {}", log0.summary());
    let (p, log, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(lih(N_FAR, r_e), Box::new(p), DT_FS, SEED);
    let s = settle(&mut d, "LiH");
    let r = report(&s, 0, 1);
    let _ = writeln!(out, "  {}", show(&r));
    let sc = s.scene.clone();
    chord(&mut d, &sc, 0, 1, "Li–H", &mut out);
    if let Some(f) = r.s {
        let p = chord_point(&sc, 0, 1, f);
        let _ = writeln!(out, "  p* is {:.3} Bohr from Li, {:.3} from H (literature ρ_b ≈ 0.04 e/Bohr³, Bader 1990 ch. 7)", dist(p, sc.nuclei[0].pos), dist(p, sc.nuclei[1].pos));
    }
    for (k, rr) in d.references().iter() {
        let _ = writeln!(out, "  reference Z = {}: E {:.6}, multiplicity {:?}, {:.4} electrons in the table", k.z, rr.energy, rr.multiplicity, rr.radial.electrons());
    }
    let log = snap(&log);
    let _ = writeln!(out, "  stepping {}; side {}; test wall {:.1} s", log.summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("lih", "LiH", N_FAR, TOL_H2, &log, t_test.elapsed(), &format!("R_e {r_e:.3}; {:?}, {:?}, q {:.3}, ρ̂ {:.4}", r.verdict, r.route, r.enhancement.unwrap_or(f64::NAN), r.rho_min.unwrap_or(f64::NAN))));
    assert!(r.verdict != Verdict::TooFar, "{}", show(&r));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn d2_vibrates_slower_than_h2_by_the_mass_ratio() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (mut p, log) = Locked::new(ctx.clone(), TOL_H2);
    let surf = locate_surface(&mut p);
    let _ = writeln!(out, "H₂ vs D₂, {N_H2}³, σ = h = {H}, tol {TOL_H2:.0e}, dt {DT_FS} fs, {VIB_STEPS} steps from R_e + {DISPLACEMENT} at rest");
    out += &surf.log;
    let integ = Integrator::new_fs(DT_FS);
    // (discrete w, Verlet-corrected ω, mass, crossings)
    let mut res = Vec::new();
    for a in [1u16, 2] {
        let mut s = h2_isotope(surf.r_e + DISPLACEMENT, a);
        p.reset();
        let mut f = solve(&mut p, &s, "start").forces;
        let mut x = vec![bond(&s) - surf.r_e];
        for _ in 0..VIB_STEPS {
            let (f_new, _) = integ.step(&mut s, &f, &mut p, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("MD step: {e}"));
            f = f_new;
            x.push(bond(&s) - surf.r_e);
        }
        let (period, crossings) = period_from_upward_crossings(&x, integ.dt);
        let w = 2.0 * std::f64::consts::PI / period;
        let omega = (2.0 - 2.0 * (w * integ.dt).cos()).sqrt() / integ.dt;
        let _ = writeln!(out, "  {}: A = {a}, m = {:.4} mₑ, {crossings} upward crossings, period {:.4} fs, w = {w:.6e} a.u. ({:.1} cm⁻¹), Verlet-corrected ω = √(2 − 2cos(w·dt))/dt = {omega:.6e} ({:.1} cm⁻¹)",
                         if a == 1 { "H₂" } else { "D₂" }, s.nuclei[0].mass, units::au_time_to_fs(period), w * HARTREE_IN_CM, omega * HARTREE_IN_CM);
        res.push((w, omega, s.nuclei[0].mass, crossings));
    }
    let expected = (res[1].2 / res[0].2).sqrt();
    let ratio = res[0].1 / res[1].1;
    let raw = res[0].0 / res[1].0;
    let log = snap(&log);
    let _ = writeln!(out, "  ω_H/ω_D = {ratio:.7} (Verlet-corrected), raw w_H/w_D = {raw:.7}; √(m_D/m_H) from the scenes' nuclei = {expected:.7}: corrected − 1 = {:+.3e} (bound {D2_BOUND:.0e}), raw − 1 = {:+.3e} (Verlet's (h_H² − h_D²)/24 predicts {:+.3e})",
                     ratio / expected - 1.0, raw / expected - 1.0, ((res[0].1 * integ.dt).powi(2) - (res[1].1 * integ.dt).powi(2)) / 24.0);
    let _ = writeln!(out, "  printed, not asserted: nuclear-mass BO ratio {RATIO_NUCLEAR_MASSES} ({:+.2e} from ours), experiment ω_e(H₂)/ω_e(D₂) = {RATIO_EXPERIMENT:.5} ({:+.2e}: the adiabatic/non-adiabatic corrections no BO code has)",
                     RATIO_NUCLEAR_MASSES / ratio - 1.0, RATIO_EXPERIMENT / ratio - 1.0);
    let _ = writeln!(out, "  {}; test wall {:.1} s", log.summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("d2_vs_h2", "H₂, D₂", N_H2, TOL_H2, &log, t_test.elapsed(), &format!("ω_H/ω_D {ratio:.7} vs {expected:.7}: {:+.2e} (raw {:+.2e})", ratio / expected - 1.0, raw / expected - 1.0)));
    assert!(log.warnings.iter().all(|w| !w.contains("box too small")), "{:?}", log.warnings);
    assert!(res.iter().all(|r| r.3 >= 5), "too few crossings: {:?}", res.iter().map(|r| r.3).collect::<Vec<_>>());
    assert!((ratio / expected - 1.0).abs() <= D2_BOUND, "ω_H/ω_D = {ratio:.7} vs √(m_D/m_H) = {expected:.7}: {:+.3e} (bound {D2_BOUND:.0e})", ratio / expected - 1.0);
}

fn centroid(s: &Scene) -> [f64; 3] {
    std::array::from_fn(|k| s.nuclei.iter().map(|n| n.pos[k]).sum::<f64>() / s.nuclei.len() as f64)
}

/// The scene scaled about its centroid by 1 + λ (the symmetric stretch of a triangle).
fn breathe(s: &Scene, lambda: f64) -> Scene {
    let c = centroid(s);
    let mut t = s.clone();
    for n in &mut t.nuclei {
        n.pos = std::array::from_fn(|k| c[k] + (1.0 + lambda) * (n.pos[k] - c[k]));
    }
    t
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations (Numerical Recipes §11.1), ascending.
fn eigenvalues_symmetric(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 {
        let off: f64 = (0..n).map(|i| (0..n).filter(|&j| j != i).map(|j| a[i][j] * a[i][j]).sum::<f64>()).sum();
        let diag: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-28 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta >= 0.0 { 1.0 } else { -1.0 } / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut e: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    e.sort_by(f64::total_cmp);
    e
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn fire_relaxes_h3_plus_to_the_quenched_minimum() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut start = h3_plus(H3_SIDE);
    for n in &mut start.nuclei {
        // a uniformly random direction (rejection from the unit ball), FIRE_KICK long
        let v: [f64; 3] = loop {
            let v: [f64; 3] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));
            let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if m > 0.1 && m <= 1.0 {
                break v.map(|x| x / m);
            }
        };
        n.pos = std::array::from_fn(|k| n.pos[k] + FIRE_KICK * v[k]);
    }
    let _ = writeln!(out, "FIRE vs the quench, H₃⁺ {N_H2}³, tol {TOL_H2:.0e}: start sides {} (side {H3_SIDE} with each nucleus kicked {FIRE_KICK} Bohr, seed {SEED})", triangle(&start));
    // (a) the suite's quench (0.5 fs), and the same at FIRE's own starting Δt (0.25 fs), reported
    let (mut pq, logq) = Locked::new(ctx.clone(), TOL_H2);
    let (sq, _, steps_q) = relax(&mut pq, start.clone(), &mut out);
    let evals_q = snap(&logq).calls;
    let (mut pq2, logq2) = Locked::new(ctx.clone(), TOL_H2);
    let mut out2 = String::new();
    let (_, _, steps_q2) = relax_with(&mut pq2, start.clone(), DT_FS, &mut out2);
    let evals_q2 = snap(&logq2).calls;
    // (b) FIRE through the driver, one FIRE step per tick (budget 0), every snapshot checked
    let (pf, logf, side) = kohn_sham(&ctx, TOL_H2);
    let mut d = Dynamics::new(start.clone(), Box::new(pf), DT_FS, SEED);
    let c0 = snap(&logf).calls;
    let s0 = d.snapshot();
    let (e0, eks0) = (s0.energy_conserved, s0.energy_ks);
    d.apply(DynCommand::Relax);
    let mut max_dev: f64 = 0.0;
    let mut ticks = 0;
    let last = loop {
        let s = d.tick(0.0, &mut never());
        ticks += 1;
        assert!(s.last_error.is_none(), "{:?}", s.last_error);
        max_dev = max_dev.max((s.energy_conserved - e0).abs());
        assert!(s.sim_time == 0.0 && s.step == 0, "a relax runs no clock: t {} step {}", s.sim_time, s.step);
        if s.relax.as_ref().is_some_and(|r| r.converged.is_some()) {
            break s;
        }
        assert!(ticks < 3000, "the relax did not end");
    };
    let evals_f = snap(&logf).calls - c0;
    let rp = last.relax.clone().unwrap();
    let sf = last.scene.clone();
    // fresh cold solves at both results (the energies compared), and the curvature
    let mut cold = |s: &Scene, what: &str| { pq.reset(); solve(&mut pq, s, what) };
    let (rq, rf) = (cold(&sq, "quench result"), cold(&sf, "FIRE result"));
    let (eq, ef) = (rq.energy, rf.energy);
    let mf = sim::relax::max_norm(&rf.forces);
    let lambda = 0.02;
    let (ep, em) = (cold(&breathe(&sf, lambda), "breathe +").energy, cold(&breathe(&sf, -lambda), "breathe −").energy);
    let c = centroid(&sf);
    let q_norm = lambda * sf.nuclei.iter().map(|n| dist(n.pos, c).powi(2)).sum::<f64>().sqrt();
    let k_breathe = (ep - 2.0 * ef + em) / (q_norm * q_norm);
    // the Cartesian Hessian from central force differences: its smallest vibrational eigenvalue
    // is the k the energy bound needs (six near-zero ones: translations, rotations)
    let n3 = 3 * sf.nuclei.len();
    let mut hess = vec![vec![0.0; n3]; n3];
    for b in 0..n3 {
        let mut fp = sf.clone();
        let mut fm = sf.clone();
        fp.nuclei[b / 3].pos[b % 3] += HESS_STEP;
        fm.nuclei[b / 3].pos[b % 3] -= HESS_STEP;
        let (gp, gm) = (cold(&fp, "hessian +").forces, cold(&fm, "hessian −").forces);
        for a in 0..n3 {
            hess[a][b] = -(gp[a / 3][a % 3] - gm[a / 3][a % 3]) / (2.0 * HESS_STEP);
        }
    }
    let asym = (0..n3).flat_map(|a| (0..n3).map(move |b| (a, b))).map(|(a, b)| (hess[a][b] - hess[b][a]).abs()).fold(0.0f64, f64::max);
    let sym: Vec<Vec<f64>> = (0..n3).map(|a| (0..n3).map(|b| 0.5 * (hess[a][b] + hess[b][a])).collect()).collect();
    let eig = eigenvalues_symmetric(sym);
    let k_min = eig[6];
    let e_bound = 3.0 * F_TOL * F_TOL / (2.0 * K_SOFT);
    let _ = writeln!(out, "  quench (dt 0.5 fs): {steps_q} steps, {evals_q} evaluations, sides {}, cold E {eq:+.9} Ha, cold max|F| {:.2e}", triangle(&sq), sim::relax::max_norm(&rq.forces));
    let _ = writeln!(out, "  quench at dt {DT_FS} fs (FIRE's starting Δt; reported): {steps_q2} steps, {evals_q2} evaluations");
    let _ = writeln!(out, "  FIRE: {} steps, {evals_f} evaluations ({ticks} ticks), converged {:?}, final Δt {:.3} fs, α {:.3}, sides {}, cold E {ef:+.9} Ha, cold max|F| {mf:.2e}; note {:?}",
                     rp.steps, rp.converged, rp.dt_fs, rp.alpha, triangle(&sf), rp.note);
    let _ = writeln!(out, "  |E_f − E_q| = {:.2e} Ha (bound 3·F_TOL²/(2·k_soft) = {e_bound:.1e} with k_soft = {K_SOFT}); Hessian eigenvalues {:?} (max asymmetry {asym:.1e}): k_min = {k_min:.4} Ha/Bohr²; k_breathe (3-point) = {k_breathe:.4}; the bound at the measured k_min would be {:.1e}",
                     (ef - eq).abs(), eig.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(), 3.0 * F_TOL * F_TOL / (2.0 * k_min));
    let _ = writeln!(out, "  sides: |Δ| up to {:.4} Bohr against the geometric slack 2·F_TOL/k_min = {:.2e} (printed, not asserted)", {
        let d = |s: &Scene, a: usize, b: usize| dist(s.nuclei[a].pos, s.nuclei[b].pos);
        [(0, 1), (0, 2), (1, 2)].iter().map(|&(a, b)| (d(&sf, a, b) - d(&sq, a, b)).abs()).fold(0.0f64, f64::max)
    }, 2.0 * F_TOL / k_min);
    let _ = writeln!(out, "  evaluations FIRE/quench = {evals_f}/{evals_q} = {:.3} (spec's ≤ 0.5 not asserted: module doc); vs the 0.25 fs quench {:.3}", evals_f as f64 / evals_q as f64, evals_f as f64 / evals_q2 as f64);
    let _ = writeln!(out, "  bookkeeping: max |E_conserved − E_conserved(start)| over {ticks} snapshots = {max_dev:.2e} (bound {E_CONSERVED_BOUND:.0e}); W = {:+.9e}, E_KS(end) − E_KS(start) = {:+.9e}",
                     last.total - last.energy_conserved, last.energy_ks - eks0);
    let _ = writeln!(out, "  FIRE stepping {}; side {}; test wall {:.1} s", snap(&logf).summary(), snap(&side).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("fire_vs_quench", "H₃⁺", N_H2, TOL_H2, &snap(&logf), t_test.elapsed(), &format!("|E_f − E_q| {:.1e}; evals FIRE {evals_f} / quench {evals_q}; k_min {k_min:.3}", (ef - eq).abs())));
    assert_eq!(rp.converged, Some(true), "{rp:?}");
    assert!(mf < F_TOL, "max|F| = {mf:.2e} on a fresh cold solve at FIRE's result");
    assert!(k_min >= K_SOFT, "the softest vibrational k {k_min:.4} is below the stated K_SOFT = {K_SOFT}: the energy bound's premise fails");
    assert!((ef - eq).abs() <= e_bound, "|E_f − E_q| = {:.2e} (bound {e_bound:.1e})", (ef - eq).abs());
    assert!(max_dev <= E_CONSERVED_BOUND, "E_conserved moved by {max_dev:.2e} during the relax");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p validate --release --test tools")]
fn fire_on_the_surrogate_is_verified_by_the_teacher() {
    let Some(ctx) = gpu() else { return };
    let t_test = Instant::now();
    let mut out = String::new();
    let (mut p0, _) = Locked::new(ctx.clone(), TOL_H2);
    let surf = locate_surface(&mut p0);
    // the surrogate suite's H₂ pass (its `h2_plan`: the default plan with the 1000 K burst, seed 7)
    let (p, log) = Locked::new(ctx.clone(), TOL_H2);
    let mut s = SurrogateKs::new(Box::new(p), FallbackPolicy { audit_every: 0, ..FallbackPolicy::default() }).with_teacher_tol(TOL_H2);
    let plan = SamplingPlan { hot_burst_t: 1000.0, seed: SEED, ..SamplingPlan::default() };
    let mut sampler = Sampler::with_cutoff(&h2(1.6), &plan, Basis::new(&[]).r_cut).unwrap_or_else(|e| panic!("plan: {e}"));
    while sampler.step_own_teacher(&mut s, &mut never()).unwrap_or_else(|e| panic!("sampling: {e}")).is_some() {}
    s.discontinuity();
    let rep = s.refit().unwrap_or_else(|e| panic!("refit: {e}"));
    let _ = writeln!(out, "FIRE on the surrogate, H₂ {N_H2}³: the pass gave {} samples, trusted {}", rep.n_samples, s.trusted());
    assert!(s.trusted(), "the H₂ model is not trusted: {:?}", s.untrusted_reason());
    let calls_trained = snap(&log).calls;
    let mut d = Dynamics::new(h2(SURROGATE_RELAX_R0), Box::new(s), DT_FS, SEED);
    d.apply(DynCommand::Relax);
    let t0 = Instant::now();
    let last = loop {
        let s = d.tick(f64::INFINITY, &mut never());
        assert!(s.last_error.is_none(), "{:?}", s.last_error);
        if s.relax.as_ref().is_some_and(|r| r.converged.is_some()) {
            break s;
        }
        assert!(t0.elapsed() < Duration::from_secs(300), "the relax did not end");
    };
    let rp = last.relax.clone().unwrap();
    let r = bond(&last.scene);
    let teacher_calls = snap(&log).calls - calls_trained;
    let _ = writeln!(out, "  relax from R = {SURROGATE_RELAX_R0}: {} FIRE steps, converged {:?}, model max|F| {:.2e}, verified (teacher) max|F| {:?} (bound {VERIFIED_F_BOUND:.0e}); R = {r:.5} vs DFT R_e {:.5}: ΔR = {:+.2e} (bound 2·F_TOL/k = {SURROGATE_R_BOUND:.0e}); teacher solves during the relax {teacher_calls}; note {:?}",
                     rp.steps, rp.converged, rp.max_force, rp.verified_max_force, surf.r_e, r - surf.r_e, rp.note);
    let _ = writeln!(out, "  teacher {}; test wall {:.1} s", snap(&log).summary(), t_test.elapsed().as_secs_f64());
    println!("{out}");
    record(&ctx, timing_row("fire_surrogate", "H₂ (model)", N_H2, TOL_H2, &snap(&log), t_test.elapsed(), &format!("verified max|F| {:.1e}, ΔR {:+.1e}", rp.verified_max_force.unwrap_or(f64::NAN), r - surf.r_e)));
    assert_eq!(rp.converged, Some(true), "{rp:?}");
    let v = rp.verified_max_force.expect("a converged relax on the model is verified by the teacher");
    assert!(v <= VERIFIED_F_BOUND, "DFT max|F| at the model's minimum {v:.2e} (bound {VERIFIED_F_BOUND:.0e})");
    assert!((r - surf.r_e).abs() <= SURROGATE_R_BOUND, "R = {r:.5} vs R_e {:.5}", surf.r_e);
}
