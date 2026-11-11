//! Transfer and reach of the shared model (M8a Task 6; spec §4.3; numbers of record in
//! `references.md`, "M8a transfer and reach"). GPU, long, `#[ignore]`d — run alone:
//! `cargo test -p validate --release --test transfer -- --ignored --nocapture --test-threads 1`
//! (`fragments_transfer_to_propane_and_the_water_trimer`; the probe and the cluster relaxation are
//! separate `#[ignore]`d tests, named below). Atomic units throughout (Hartree, Bohr).
//!
//! **Measured cost (2026-11-02, RX 7900 XTX):** the hydrocarbon case took 5.0 h of GPU — the CH₄ +
//! C₂H₆ Standard passes ≈ 4.0 h (984 jobs; ≈ 20 s per C₂H₆ solve against the probe's 8 s warm
//! solve), the propane references, confinement solves and 20 held-out geometries 1.0 h (179 s per
//! held-out solve). `TRANSFER_REUSE=1` attaches the collection a previous run taught (under the
//! target's tmp dir) and skips the passes; `TRANSFER_CONTROL_ONLY=1` stops after the in-domain
//! control (CPU only); `TRANSFER_SYSTEMS` (`hydrocarbons,water`) and `TRANSFER_REACHES_<SYSTEM>`
//! select. The water case needs `scenes/h2o_dimer.scene` and `scenes/h2o_trimer.scene` from
//! `relax_the_water_clusters` and has not been run (its estimated cost: `references.md`).
//!
//! # What is claimed
//!
//! The model is local: E = Σ_i E_ref(Z_i) + Σ_i ε(environment of i within r_cut), so a collection
//! taught on fragments answers a larger molecule built from the same environments. The test
//! teaches one shared collection (body-4 basis, `SurrogateKs::attach_collection_at`) the Standard
//! sampling pass (`SamplingPlan::default()`: 60 perturbations σ 0.15, 12-point pair scans
//! 0.7–1.8 r₀ up to 5/6 r_cut, a 200-step 2000 K burst) of the fragments on the app's default
//! surface — Accurate grid (h = σ = 0.15 Bohr), `ionic accepted`, the GPU teacher at the sampling
//! tolerance 1e-6 — and asks it about 20 held-out geometries of the larger molecule (Gaussian
//! σ = 0.15 Bohr per coordinate about the shipped geometry, a seed no pass used):
//!
//! - **hydrocarbons:** CH₄ (`scenes/ch4.scene`, 81³) and C₂H₆ (`scenes/c2h6.scene`, 95³) →
//!   propane (`scenes/c3h8.scene`, 107³);
//! - **water:** H₂O (the app's h2o preset geometry, 95³ = its 14 Bohr box) and (H₂O)₂
//!   (`scenes/h2o_dimer.scene`) → (H₂O)₃ (`scenes/h2o_trimer.scene`).
//!
//! Per held-out geometry the provider's strict verdict is taken **per atom**
//! (`SurrogateKs::atom_verdicts`: rules 5a–5e, 5d's force rows, 4b's environment key and rule 6 per
//! atom; the gate, 5d's energy row, rule 7 and the fences for the whole geometry — exactly
//! `would_answer` when every atom is trusted). Asserted at the collection's reach
//! (`surrogate::COLLECTION_R_CUT`): the per-component force RMSE over the atoms it calls trusted
//! ≤ `TRANSFER_FACTOR` (2) × the collection's own leave-one-out force RMSE, and the RMS of |ΔE|/N
//! over the geometries it answers whole ≤ 2 × its leave-one-out energy-per-atom RMSE
//! (`SurrogateKs::loo_rms`, the fit's own members). The rest is refused (the teacher answers it
//! in strict mode): the refused fraction is reported with the errors the model would have made
//! there, so a refusal can be seen to be warranted or conservative. The other reach is a
//! measurement (reported, the plan's reach rule decides `COLLECTION_R_CUT` from it, `references.md`).
//!
//! **In-domain control** (CPU, before the held-out solves): the same verdicts on 20 perturbed
//! geometries of each fragment — whether the collection trusts the environments it was taught on,
//! which tells a refusal of the target for being new from a refusal of everything.
//!
//! # Reach, and what is reused between reaches
//!
//! The collection's samples are taught once, at the first reach of `REACHES`, and **re-fitted** at
//! the other reach from the same stored samples (a second provider with
//! `SurrogateKs::with_collection_r_cut` attaches the same folder: the samples are re-described, no
//! solve is repeated), and the held-out teacher answers are shared — the DFT does not depend on the
//! descriptor. What that leaves out: a pass run at r_cut = 9 would scan pairs out to 5/6 · 9 =
//! 7.5 Bohr (here: to 5 Bohr) and scan the pairs between 6 and 9 Bohr apart (none in CH₄; the
//! water dimer's far H–H), so the 9 Bohr fit sees the band 5–9 Bohr only through the
//! perturbations and the burst — a statement about this measurement, reported with it.
//!
//! # Boxes (the references-per-box ruling, Task 4a)
//!
//! The fragments and the targets sit in different boxes on purpose (81³/95³ → 107³; 95³ and the dimer's
//! 5 Bohr-margin box → the trimer's 4 Bohr-margin box). Each sample's fit target is E − Σ ref(Z, own box), which removes each atom's own
//! confinement to first order; what it cannot remove is the **molecule-vs-atom confinement
//! residual**: [E_mol(box A) − Σ ref(A)] − [E_mol(box B) − Σ ref(B)] for the same geometry. It is
//! measured here for each fragment between its training box and the target's box (the fragment's
//! own geometry solved in both) and reported per atom beside the leave-one-out RMSE it would enter.
//!
//! # The other tests in this file
//!
//! - `transfer_cost_probe` (`#[ignore]`, minutes): one Accurate solve of each system and the Standard
//!   pass's job counts, for the cost estimate of the full run.
//! - `relax_the_water_clusters` (`#[ignore]`, run once to (re)make the scene files): (H₂O)₂ and
//!   (H₂O)₃ relaxed by FIRE on this surface from the cited start geometries, written to
//!   `scenes/h2o_dimer.scene` and `scenes/h2o_trimer.scene`.
use gpu::GpuContext;
use providers::{AtomVerdicts, PlanSolves, SurrogateKs};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::{Cell, Fire, FireOutcome, FireParams, ForceProvider, IonicPolicy, Nucleus, SamplingPlan, Scene};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

mod common;
use common::surrogate::{class_summary, fit_summary, read_scene, sample_and_fit_counted, scene_text, scenes_dir, student};
use common::{never, no_augment, solve, Locked, DT_FS, TOL_H2};

/// The reaches compared (Bohr): today's 6 and the spec's ≈ 9 (spec §4.3).
const REACHES: [f64; 2] = [6.0, 9.0];
const HELD_OUT: usize = 20;
const HELD_OUT_SIGMA: f64 = 0.15;
/// XORed into the plan's seed for the held-out draws, so no pass seed reproduces them.
const HELD_OUT_SALT: u64 = 0x7a2a_5f3e;
/// Held-out error over the collection's own leave-one-out RMSE (spec §4.3: "at most 2×").
const TRANSFER_FACTOR: f64 = 2.0;
/// The Accurate grid spacing (= σ) of the app's default surface.
const H_ACC: f64 = 0.15;
/// The box margin of a target cluster (Bohr): the builder's "enlarge the box" margin, about the
/// shipped propane's (16 Bohr box, |x| ≤ 4.1).
const TARGET_MARGIN: f64 = 4.0;
/// The box margin of a training cluster (Bohr): about the shipped fragments' (CH₄ 4.8, C₂H₆ 4.9,
/// the h2o preset 5.6), so the pass's stretched and hot geometries stay below the box rule's face
/// density (7.2e-6 e/Bohr³ at h = 0.15; the dimer at margin 4 sits at 5.1e-6 already at rest).
const TRAINING_MARGIN: f64 = 5.0;
/// Repetitions of the model evaluation timing per geometry.
const EVAL_REPS: usize = 20;

fn gpu() -> Option<GpuContext> {
    common::gpu("transfer")
}

fn accurate(n: usize, nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: Cell { n, h: H_ACC }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: IonicPolicy::Accepted, train_mask: None }
}

/// The app's h2o preset geometry (O–H 1.81 Bohr, 104.5°) in its 14 Bohr box at the Accurate grid.
fn water() -> Scene {
    accurate(95, vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.43, 1.11, 0.0]), Nucleus::new(1, [-1.43, 1.11, 0.0])])
}

/// The smallest odd n with (n − 1)·h ≥ 2(max |x| + `margin`), the nuclei centred first (the
/// bounding box's centre at the origin).
fn boxed(mut nuclei: Vec<Nucleus>, margin: f64) -> Scene {
    for k in 0..3 {
        let (lo, hi) = nuclei.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), n| (l.min(n.pos[k]), h.max(n.pos[k])));
        let c = 0.5 * (lo + hi);
        for n in &mut nuclei {
            n.pos[k] -= c;
        }
    }
    let m = nuclei.iter().flat_map(|n| n.pos).fold(0.0f64, |m, x| m.max(x.abs()));
    let mut n = ((2.0 * (m + margin)) / H_ACC).ceil() as usize + 1;
    if n % 2 == 0 {
        n += 1;
    }
    accurate(n, nuclei)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt())
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Gas-phase monomer: r(O–H) 0.9572 Å, ∠HOH 104.52° (Hoy & Bunker, J. Mol. Spectrosc. 74, 1 (1979)).
const R_OH_ANGSTROM: f64 = 0.9572;
const HOH_DEG: f64 = 104.52;

/// The water dimer's start (Cs, trans-linear): R(O···O) = 2.912 Å, the CCSD(T) equilibrium of
/// Klopper, van Duijneveldt-van de Rijdt & van Duijneveldt, Phys. Chem. Chem. Phys. 2, 2227 (2000);
/// the donor's bonded H 5° off the O···O axis and the acceptor's bisector 57° from the axis (the
/// usual Cs dimer angles, rounded); monomers at the gas-phase geometry above. A start, not a
/// literature Cartesian set: FIRE on this surface decides the geometry.
fn dimer_start() -> Scene {
    let r_oo = units::angstrom_to_bohr(2.912);
    let r_oh = units::angstrom_to_bohr(R_OH_ANGSTROM);
    let a = HOH_DEG.to_radians();
    let od = [0.0; 3];
    let hb_dir = [5f64.to_radians().cos(), 5f64.to_radians().sin(), 0.0];
    let hf_dir = [(5f64.to_radians() + a).cos(), (5f64.to_radians() + a).sin(), 0.0];
    let oa = [r_oo, 0.0, 0.0];
    let tilt = 57f64.to_radians();
    let bis = [tilt.cos(), -tilt.sin(), 0.0];
    let half = 0.5 * a;
    let ha1 = add(oa, scale(add(scale(bis, half.cos()), [0.0, 0.0, half.sin()]), r_oh));
    let ha2 = add(oa, scale(add(scale(bis, half.cos()), [0.0, 0.0, -half.sin()]), r_oh));
    boxed(vec![
        Nucleus::new(8, od),
        Nucleus::new(1, scale(hb_dir, r_oh)),
        Nucleus::new(1, scale(hf_dir, r_oh)),
        Nucleus::new(8, oa),
        Nucleus::new(1, ha1),
        Nucleus::new(1, ha2),
    ], TRAINING_MARGIN)
}

/// The cyclic (uud) water trimer's start: the O atoms on an equilateral triangle of side
/// R(O···O) = 2.80 Å (≈ the MP2 cyclic-trimer value, Xantheas & Dunning, J. Chem. Phys. 99, 8774
/// (1993)); each monomer donates its bonded H along the O→O edge to the next (a linear H bond) and
/// its free H points 60° out of the ring plane, up, up, down, rotated from the bonded H by ∠HOH;
/// monomers at the gas-phase geometry. A start, not a literature Cartesian set.
fn trimer_start() -> Scene {
    let r_oo = units::angstrom_to_bohr(2.80);
    let r_oh = units::angstrom_to_bohr(R_OH_ANGSTROM);
    let rad = r_oo / 3f64.sqrt();
    let o: Vec<[f64; 3]> = (0..3).map(|k| {
        let t = (90.0 + 120.0 * k as f64).to_radians();
        [rad * t.cos(), rad * t.sin(), 0.0]
    }).collect();
    let a = HOH_DEG.to_radians();
    let mut nuclei = Vec::new();
    for k in 0..3 {
        let d1 = unit(add(o[(k + 1) % 3], scale(o[k], -1.0)));
        // outward from the ring centre, made perpendicular to d1
        let out = unit(o[k]);
        let w0 = unit(add(out, scale(d1, -dot(out, d1))));
        let up = if k == 2 { -1.0 } else { 1.0 };
        let w = unit(add(scale(w0, 60f64.to_radians().cos()), [0.0, 0.0, up * 60f64.to_radians().sin()]));
        let d2 = add(scale(d1, a.cos()), scale(w, a.sin()));
        nuclei.push(Nucleus::new(8, o[k]));
        nuclei.push(Nucleus::new(1, add(o[k], scale(d1, r_oh))));
        nuclei.push(Nucleus::new(1, add(o[k], scale(unit(d2), r_oh))));
    }
    boxed(nuclei, TARGET_MARGIN)
}

fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `HELD_OUT` geometries of `sc`: every coordinate displaced by N(0, σ²), seeded; a draw that
/// leaves the cell's 3h margin is redrawn (none expected at σ = 0.15 in these boxes).
fn held_out(sc: &Scene, seed: u64) -> Vec<Scene> {
    let mut rng = StdRng::seed_from_u64(seed ^ HELD_OUT_SALT);
    let mut out = Vec::new();
    while out.len() < HELD_OUT {
        let mut g = sc.clone();
        for n in &mut g.nuclei {
            for k in 0..3 {
                n.pos[k] += HELD_OUT_SIGMA * normal(&mut rng);
            }
        }
        if g.check_inside_cell().is_ok() {
            out.push(g);
        }
    }
    out
}

fn species(sc: &Scene) -> Vec<u8> {
    let mut z: Vec<u8> = sc.nuclei.iter().map(|n| n.z).collect();
    z.sort_unstable();
    z.dedup();
    z
}

/// The isolated-atom reference of `z` in `cell` (`sim::solve_atom`: every allowed multiplicity
/// cold, the lowest kept — the sampling pass's rule) and its face density.
fn atom_reference(t: &mut dyn ForceProvider, cell: Cell, z: u8) -> (f64, Option<f64>) {
    let mass = Nucleus::new(z, [0.0; 3]).mass;
    let r = sim::solve_atom(t, &cell, None, IonicPolicy::Accepted, z, mass, &mut never()).unwrap_or_else(|e| panic!("reference Z = {z}, {}³: {e}", cell.n));
    (r.energy, r.face_density)
}

/// Prints the log so far (a run of hours must not lose it to a later panic) and clears it.
fn flush(out: &mut String) {
    print!("{out}");
    out.clear();
}

/// What one system's run measures at one reach.
struct Row {
    system: &'static str,
    r_cut: f64,
    /// (E per atom, F per component) leave-one-out RMSE of the collection's own fit.
    loo: (f64, f64),
    fitted: usize,
    q_prime: usize,
    /// Force RMSE over trusted atoms, and their count.
    f_trusted: f64,
    atoms_trusted: usize,
    atoms_total: usize,
    /// RMS |ΔE|/N over geometries answered whole, and their count.
    e_answered: f64,
    answered: usize,
    /// The same over every held-out geometry/atom, trusted or not (what refusing avoided).
    f_all: f64,
    e_all: f64,
    refusals: BTreeMap<String, usize>,
    /// Model evaluation per geometry, µs: energy + forces; with the variance; the per-atom verdict.
    eval_us: f64,
    eval_var_us: f64,
    verdict_us: f64,
}

impl Row {
    fn line(&self) -> String {
        format!(
            "{} r_cut {:.0}: fit {} samples (Q′ {}), LOO RMSE E {:.2e} Ha/atom F {:.2e} Ha/Bohr → bounds {:.2e} / {:.2e}; trusted atoms {}/{} ({:.0} % refused), force RMSE on them {:.2e} ({:.2}× LOO); answered geometries {}/{HELD_OUT}, RMS |ΔE|/N {:.2e} ({:.2}× LOO); all atoms/geometries: F {:.2e}, |ΔE|/N {:.2e}; refusals {:?}; eval {:.0} µs/geometry ({:.0} with variance, verdict {:.0})",
            self.system,
            self.r_cut,
            self.fitted,
            self.q_prime,
            self.loo.0,
            self.loo.1,
            TRANSFER_FACTOR * self.loo.0,
            TRANSFER_FACTOR * self.loo.1,
            self.atoms_trusted,
            self.atoms_total,
            100.0 * (1.0 - self.atoms_trusted as f64 / self.atoms_total.max(1) as f64),
            self.f_trusted,
            self.f_trusted / self.loo.1,
            self.answered,
            self.e_answered,
            self.e_answered / self.loo.0,
            self.f_all,
            self.e_all,
            self.refusals,
            self.eval_us,
            self.eval_var_us,
            self.verdict_us
        )
    }
}

/// The teacher's answer at a held-out geometry (shared between reaches).
struct Dft {
    energy: f64,
    forces: Vec<[f64; 3]>,
}

/// One system: teach the fragments at `reaches[0]`, solve the held-out targets and the target
/// box's references, measure the confinement residual, then judge every reach (re-fitting the
/// stored samples at each later one). Returns the rows and appends the log.
fn run_system(ctx: &GpuContext, system: &'static str, fragments: &[Scene], target: &Scene, reaches: &[f64], out: &mut String) -> Vec<Row> {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("transfer").join(system);
    // `TRANSFER_REUSE=1`: a collection this test taught before (its folder under the target's tmp
    // dir) is attached as it is and no pass runs — the passes are the hours of this test
    let reuse = std::env::var("TRANSFER_REUSE").is_ok_and(|v| v == "1") && root.exists();
    if !reuse {
        let _ = std::fs::remove_dir_all(&root);
    }
    std::fs::create_dir_all(&root).expect("collection root");
    let plan = SamplingPlan::default();
    let t_sys = Instant::now();
    let _ = writeln!(out, "\n== {system}: fragments {} → target {} atoms in {}³; reaches {reaches:?} ==",
                     fragments.iter().map(|f| format!("{} atoms in {}³", f.nuclei.len(), f.cell.n)).collect::<Vec<_>>().join(", "), target.nuclei.len(), target.cell.n);

    // ---- teach at the first reach ----
    let (s, log) = student(ctx, 0);
    let mut s = s.with_collection_r_cut(reaches[0]);
    s.attach_collection_at(&root, &fragments[0]).unwrap_or_else(|e| panic!("attach: {e}"));
    let mut pass_s = 0.0;
    if reuse {
        let c = s.collection().expect("attached");
        let _ = writeln!(out, "  TRANSFER_REUSE: the collection taught before is attached ({} samples, references {:?}); no pass runs",
                         c.samples.len(), c.references.keys().collect::<Vec<_>>());
    }
    for f in fragments.iter().filter(|_| !reuse) {
        let (_, pass, p) = sample_and_fit_counted(&mut s, f, &plan, out);
        pass_s += pass.as_secs_f64();
        // failed solves are a pass's normal outcome (the time limit, the stall stop): reported
        let _ = writeln!(out, "  pass on {} atoms ({}³): {} jobs, {} done, {} skipped, {} failed, {:.0} s", f.nuclei.len(), f.cell.n, p.total, p.done, p.skipped, p.failed, pass.as_secs_f64());
        flush(out);
    }
    let teach_log = log.lock().unwrap().clone();
    let _ = writeln!(out, "  teacher for the passes: {} (wall {pass_s:.0} s)", teach_log.summary());
    flush(out);

    // ---- in-domain control (CPU, no teacher solve): the verdicts on the fragments' own
    // perturbed geometries — what the collection says where it was taught ----
    let control_refit = s.refit().map(|r| r.q_prime);
    for f in fragments {
        let (mut trusted, mut total, mut answered) = (0usize, 0usize, 0usize);
        let mut rules: BTreeMap<String, usize> = BTreeMap::new();
        for g in held_out(f, plan.seed ^ 0x00c0_7701) {
            let v = s.atom_verdicts(&g).expect("atoms");
            answered += v.answers() as usize;
            if let Some(r) = v.geometry {
                *rules.entry(format!("geometry {r:?}")).or_default() += 1;
            }
            for a in &v.atoms {
                total += 1;
                match a {
                    None if v.geometry.is_none() => trusted += 1,
                    None => {}
                    Some(r) => *rules.entry(format!("{r:?}")).or_default() += 1,
                }
            }
        }
        let _ = writeln!(out, "  in-domain control at r_cut {} (fit Q′ {control_refit:?}): {} perturbed {}-atom fragments ({}³): trusted atoms {trusted}/{total}, answered {answered}/{HELD_OUT}; refusals {rules:?}",
                         reaches[0], HELD_OUT, f.nuclei.len(), f.cell.n);
    }
    let _ = writeln!(out, "  classes: {}", class_summary(&s));
    flush(out);
    if std::env::var("TRANSFER_CONTROL_ONLY").is_ok_and(|v| v == "1") {
        return Vec::new();
    }

    // ---- the target box's references, the held-out answers, the confinement residual ----
    let t = Instant::now();
    let mut ref_target: BTreeMap<u8, f64> = BTreeMap::new();
    for z in species(target) {
        let (e, face) = atom_reference(s.teacher_mut(), target.cell.clone(), z);
        ref_target.insert(z, e);
        s.set_reference_solved(z, e, target.cell.n, face);
        let _ = writeln!(out, "  reference Z = {z} in the target's {}³ box: {e:.6} Ha (face density {face:?})", target.cell.n);
    }
    let sum_ref = |refs: &BTreeMap<u8, f64>, sc: &Scene| -> f64 { sc.nuclei.iter().map(|n| refs[&n.z]).sum() };
    // the molecule-vs-atom confinement residual of each fragment between its box and the target's
    let coll_ref = |s: &SurrogateKs, z: u8, n: usize| s.collection().and_then(|c| c.references.get(&(z, n))).map(|r| r.energy).unwrap_or_else(|| panic!("no Z = {z} reference of the {n}³ box"));
    for f in fragments {
        s.teacher_mut().reset();
        let own = solve(s.teacher_mut(), f, "fragment in its box");
        let mut moved = f.clone();
        moved.cell = target.cell.clone();
        s.teacher_mut().reset();
        let there = solve(s.teacher_mut(), &moved, "fragment in the target's box");
        let a_own = own.energy - f.nuclei.iter().map(|n| coll_ref(&s, n.z, f.cell.n)).sum::<f64>();
        let a_there = there.energy - sum_ref(&ref_target, f);
        let fd: f64 = own.forces.iter().zip(&there.forces).flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).powi(2))).sum::<f64>();
        let _ = writeln!(out, "  confinement residual of the {}-atom fragment, {}³ → {}³: Δ(E − Σ ref) = {:+.2e} Ha ({:+.2e} Ha/atom), force rms Δ {:.2e} Ha/Bohr (raw ΔE {:+.2e} Ha, face densities {:?} / {:?})",
                         f.nuclei.len(), f.cell.n, target.cell.n, a_own - a_there, (a_own - a_there) / f.nuclei.len() as f64,
                         (fd / (3 * f.nuclei.len()) as f64).sqrt(), own.energy - there.energy, own.stats.face_density, there.stats.face_density);
    }
    let geoms = held_out(target, plan.seed);
    let mut dft = Vec::new();
    let solves_before = log.lock().unwrap().clone();
    for (i, g) in geoms.iter().enumerate() {
        let r = solve(s.teacher_mut(), g, &format!("held-out {i}"));
        if !r.stats.warnings.is_empty() {
            let _ = writeln!(out, "  held-out {i}: teacher warnings {:?}", r.stats.warnings);
        }
        dft.push(Dft { energy: r.energy, forces: r.forces });
    }
    let after = log.lock().unwrap().clone();
    let held_gpu = (after.gpu - solves_before.gpu).as_secs_f64();
    let _ = writeln!(out, "  teacher for {HELD_OUT} held-out geometries: {:.1} s GPU ({:.2} s/geometry); references + residuals + held-out wall {:.0} s",
                     held_gpu, held_gpu / HELD_OUT as f64, t.elapsed().as_secs_f64());
    flush(out);

    // ---- judge each reach ----
    let mut rows = Vec::new();
    let mut s = Some(s);
    for (k, &r_cut) in reaches.iter().enumerate() {
        let mut sk = match (k, s.take()) {
            (0, Some(s0)) => s0,
            (_, prev) => {
                drop(prev); // the writable handle released before the next opens the folder
                let (sn, _) = student(ctx, 0);
                let mut sn = sn.with_collection_r_cut(r_cut);
                sn.attach_collection_at(&root, &fragments[0]).unwrap_or_else(|e| panic!("attach at r_cut {r_cut}: {e}"));
                sn
            }
        };
        let t = Instant::now();
        // the first reach refits after the target box's references arrived (a rebase); a later
        // reach's attach fitted already when it could
        let rep = if k == 0 || sk.model().is_none() || sk.pending_refit() {
            sk.refit().unwrap_or_else(|e| panic!("refit at r_cut {r_cut}: {e}"))
        } else {
            sk.report().cloned().expect("the attach fitted")
        };
        let b = sk.basis().expect("fitted");
        assert_eq!(b.r_cut, r_cut, "the collection's fit is at the requested reach");
        let _ = writeln!(out, "  r_cut {r_cut}: refit {:.1} s, basis body3 {} body4 {} species {:?} Q′ {}: {}", t.elapsed().as_secs_f64(), b.body3, b.body4, b.species, b.n_independent_columns(), fit_summary(&rep, &sk));
        for n in sim::Trainable::take_notes(&mut sk) {
            let _ = writeln!(out, "    note: {n}");
        }
        let loo = sk.loo_rms().expect("a fitted model has leave-one-out residuals");
        rows.push(judge(&sk, system, r_cut, &geoms, &dft, (loo.energy_per_atom, loo.force), loo.samples, rep.q_prime, out));
        let _ = writeln!(out, "  {}", rows.last().unwrap().line());
        let _ = writeln!(out, "  classes: {}", class_summary(&sk));
        flush(out);
        s = Some(sk);
    }
    let _ = writeln!(out, "  {system} wall {:.0} s", t_sys.elapsed().as_secs_f64());
    rows
}

/// The model against the shared teacher answers at the held-out geometries (module doc).
#[allow(clippy::too_many_arguments)]
fn judge(s: &SurrogateKs, system: &'static str, r_cut: f64, geoms: &[Scene], dft: &[Dft], loo: (f64, f64), fitted: usize, q_prime: usize, out: &mut String) -> Row {
    let model = s.model().expect("fitted");
    let (mut sf_t, mut nf_t, mut se_a, mut n_a, mut sf_all, mut nf_all, mut se_all) = (0.0, 0usize, 0.0, 0usize, 0.0, 0usize, 0.0);
    let mut atoms_trusted = 0;
    let mut atoms_total = 0;
    let mut refusals: BTreeMap<String, usize> = BTreeMap::new();
    let (mut t_eval, mut t_var, mut t_verdict) = (0.0, 0.0, 0.0);
    for (i, (g, d)) in geoms.iter().zip(dft).enumerate() {
        let pos: Vec<[f64; 3]> = g.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = g.nuclei.iter().map(|n| n.z).collect();
        let n = pos.len();
        let t = Instant::now();
        for _ in 0..EVAL_REPS {
            std::hint::black_box(model.predict(&pos, &z, false).expect("predict"));
        }
        t_eval += t.elapsed().as_secs_f64() / EVAL_REPS as f64;
        let t = Instant::now();
        let pred = model.predict(&pos, &z, true).expect("predict");
        t_var += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let v: AtomVerdicts = s.atom_verdicts(g).expect("atoms");
        t_verdict += t.elapsed().as_secs_f64();
        // energies on one surface: the model's references are the target box's (set before the
        // refit; the canonical box is the largest), the teacher's raw answer is in that box
        let de = (pred.energy - d.energy) / n as f64;
        se_all += de * de;
        if v.answers() {
            se_a += de * de;
            n_a += 1;
        }
        if let Some(r) = v.geometry {
            *refusals.entry(format!("geometry {r:?}")).or_default() += 1;
        }
        let mut worst = (0.0f64, 0usize);
        for j in 0..n {
            let e2: f64 = (0..3).map(|k| (pred.forces[j][k] - d.forces[j][k]).powi(2)).sum();
            sf_all += e2;
            nf_all += 3;
            atoms_total += 1;
            if v.trusted(j) {
                sf_t += e2;
                nf_t += 3;
                atoms_trusted += 1;
                if e2.sqrt() > worst.0 {
                    worst = (e2.sqrt(), j);
                }
            } else if let Some(r) = v.atoms[j] {
                *refusals.entry(format!("{r:?}")).or_default() += 1;
            }
        }
        let _ = writeln!(out, "    r_cut {r_cut} held-out {i}: ΔE/N {de:+.2e}, trusted {}/{n}{}, worst trusted ‖ΔF‖ {:.2e} (atom {}), verdicts {:?}{}",
                         v.atoms.iter().filter(|a| a.is_none()).count(), if v.answers() { " (answered)" } else { "" }, worst.0, worst.1,
                         v.atoms.iter().map(|a| a.map_or("·".to_string(), |r| format!("{r:?}"))).collect::<Vec<_>>(),
                         v.geometry.map_or(String::new(), |r| format!(", geometry {r:?}")));
    }
    let k = geoms.len() as f64;
    Row {
        system,
        r_cut,
        loo,
        fitted,
        q_prime,
        f_trusted: if nf_t > 0 { (sf_t / nf_t as f64).sqrt() } else { f64::NAN },
        atoms_trusted,
        atoms_total,
        e_answered: if n_a > 0 { (se_a / n_a as f64).sqrt() } else { f64::NAN },
        answered: n_a,
        f_all: (sf_all / nf_all.max(1) as f64).sqrt(),
        e_all: (se_all / k).sqrt(),
        refusals,
        eval_us: 1e6 * t_eval / k,
        eval_var_us: 1e6 * t_var / k,
        verdict_us: 1e6 * t_verdict / k,
    }
}

/// The reaches each system is run at: both, by default — a later reach re-fits the stored samples
/// and reuses the held-out answers, so it costs CPU (the refit, the evaluations), no teacher solve.
fn reaches_for(system: &str) -> Vec<f64> {
    let key = format!("TRANSFER_REACHES_{}", system.to_uppercase());
    match std::env::var(&key) {
        Ok(v) => v.split(',').map(|x| x.trim().parse::<f64>().unwrap_or_else(|_| panic!("{key}: '{x}'"))).collect(),
        Err(_) => REACHES.to_vec(),
    }
}

/// The transfer test (module doc). `TRANSFER_SYSTEMS` (`hydrocarbons,water`, the default) picks the
/// systems; `TRANSFER_REACHES_<SYSTEM>` (comma-separated Bohr) overrides a system's reaches.
#[test]
#[ignore = "GPU, hours (hydrocarbons measured 5.0 h; TRANSFER_REUSE=1 skips the passes): cargo test -p validate --release --test transfer -- --ignored --nocapture --test-threads 1 fragments_transfer"]
fn fragments_transfer_to_propane_and_the_water_trimer() {
    let Some(ctx) = gpu() else { return };
    let systems = std::env::var("TRANSFER_SYSTEMS").unwrap_or_else(|_| "hydrocarbons,water".into());
    let mut out = String::new();
    let mut rows = Vec::new();
    let t = Instant::now();
    for system in systems.split(',').map(str::trim) {
        let (name, fragments, target): (&'static str, Vec<Scene>, Scene) = match system {
            "hydrocarbons" => ("hydrocarbons", vec![read_scene("ch4.scene"), read_scene("c2h6.scene")], read_scene("c3h8.scene")),
            "water" => {
                assert!(scenes_dir().join("h2o_trimer.scene").exists() && scenes_dir().join("h2o_dimer.scene").exists(),
                        "the water case needs scenes/h2o_dimer.scene and scenes/h2o_trimer.scene: run relax_the_water_clusters first (GPU, hours), or TRANSFER_SYSTEMS=hydrocarbons");
                ("water", vec![water(), read_scene("h2o_dimer.scene")], read_scene("h2o_trimer.scene"))
            }
            other => panic!("TRANSFER_SYSTEMS: unknown system '{other}'"),
        };
        rows.extend(run_system(&ctx, name, &fragments, &target, &reaches_for(name), &mut out));
        println!("{out}");
        out.clear();
    }
    println!("\n== transfer and reach (bound: {TRANSFER_FACTOR}× the collection's own LOO RMSE; asserted at COLLECTION_R_CUT = {}) ==", surrogate::COLLECTION_R_CUT);
    for r in &rows {
        println!("{}", r.line());
        if r.atoms_trusted == 0 {
            println!("  VACUOUS at r_cut {}: the collection trusts no atom of the held-out target — the 2× criterion constrains nothing here (every atom refused; the refused errors above show whether that was warranted)", r.r_cut);
        }
    }
    println!("total wall {:.0} s", t.elapsed().as_secs_f64());
    let mut failures = Vec::new();
    for r in rows.iter().filter(|r| r.r_cut == surrogate::COLLECTION_R_CUT) {
        if r.atoms_trusted > 0 && !(r.f_trusted <= TRANSFER_FACTOR * r.loo.1) {
            failures.push(format!("{} r_cut {}: force RMSE on trusted atoms {:.2e} > {TRANSFER_FACTOR}× LOO {:.2e}", r.system, r.r_cut, r.f_trusted, r.loo.1));
        }
        if r.answered > 0 && !(r.e_answered <= TRANSFER_FACTOR * r.loo.0) {
            failures.push(format!("{} r_cut {}: RMS |ΔE|/N on answered geometries {:.2e} > {TRANSFER_FACTOR}× LOO {:.2e}", r.system, r.r_cut, r.e_answered, r.loo.0));
        }
    }
    assert!(failures.is_empty(), "transfer: {failures:#?}");
}

/// Cost probe (Task 6 Step 0): one Accurate solve of each system and the Standard pass's job
/// counts, to estimate the full test's wall time before running it.
#[test]
#[ignore = "GPU, minutes: cargo test -p validate --release --test transfer -- --ignored --nocapture transfer_cost_probe"]
fn transfer_cost_probe() {
    let Some(ctx) = gpu() else { return };
    let (mut t, log) = Locked::new(ctx.clone(), TOL_H2);
    let plan = SamplingPlan::default();
    let dimer = std::fs::metadata(scenes_dir().join("h2o_dimer.scene")).map(|_| read_scene("h2o_dimer.scene")).unwrap_or_else(|_| dimer_start());
    let trimer = std::fs::metadata(scenes_dir().join("h2o_trimer.scene")).map(|_| read_scene("h2o_trimer.scene")).unwrap_or_else(|_| trimer_start());
    for (name, sc) in [("CH4", read_scene("ch4.scene")), ("C2H6", read_scene("c2h6.scene")), ("C3H8", read_scene("c3h8.scene")), ("H2O", water()), ("(H2O)2", dimer), ("(H2O)3", trimer)] {
        let jobs = plan.solves(&sc).unwrap_or_else(|e| panic!("{name}: {e}"));
        t.reset();
        let before = log.lock().unwrap().clone();
        let w = Instant::now();
        let cold = solve(&mut t, &sc, name);
        let cold_s = w.elapsed().as_secs_f64();
        // a warm solve after a small displacement (what a pass's next job looks like)
        let mut moved = sc.clone();
        moved.nuclei[0].pos[0] += 0.05;
        let w = Instant::now();
        let warm = solve(&mut t, &moved, name);
        let warm_s = w.elapsed().as_secs_f64();
        let after = log.lock().unwrap().clone();
        println!("{name}: {} atoms, {}³, Standard pass {jobs} jobs; cold solve {cold_s:.2} s ({} SCF), warm {warm_s:.2} s ({} SCF); E {:.6} Ha; face density {:?}; warnings {:?}; GPU {:.2} s",
                 sc.nuclei.len(), sc.cell.n, cold.stats.scf_steps, warm.stats.scf_steps, cold.energy, cold.stats.face_density, after.warnings, (after.gpu - before.gpu).as_secs_f64());
    }
}

/// (H₂O)₂ and (H₂O)₃ relaxed on the app's default surface (Accurate, `ionic accepted`, GPU teacher
/// at 1e-6) by FIRE (`sim::Fire`, the app's Relax: Δt from the 0.25 fs MD step, F_TOL 1e-3
/// Ha/Bohr on every atom's force norm) from the cited starts; the relaxed geometry is re-centred
/// in its box and written to `scenes/` (P7 of the M8a ledger). Run once; the transfer test reads
/// the files.
#[test]
#[ignore = "GPU, hours (estimated 0.5 h dimer, 2–4 h trimer; not yet run): cargo test -p validate --release --test transfer -- --ignored --nocapture relax_the_water_clusters"]
fn relax_the_water_clusters() {
    let Some(ctx) = gpu() else { return };
    let which = std::env::var("RELAX_CLUSTERS").unwrap_or_else(|_| "dimer,trimer".into());
    for name in which.split(',').map(str::trim) {
        let (start, file, margin, cite): (Scene, &str, f64, Vec<String>) = match name {
            "dimer" => (dimer_start(), "h2o_dimer.scene", TRAINING_MARGIN, vec![
                "Water dimer (Cs, trans-linear), relaxed on this solver's default surface: FIRE (the app's Relax, F_TOL 1e-3 Ha/Bohr".into(),
                "per atom) at the Accurate grid (h = sigma = 0.15 Bohr) under the accepted pseudopotentials, GPU teacher at tol 1e-6.".into(),
                "Start: R(O...O) = 2.912 A, the CCSD(T) equilibrium of Klopper, van Duijneveldt-van de Rijdt & van Duijneveldt,".into(),
                "Phys. Chem. Chem. Phys. 2, 2227 (2000); donor H 5 degrees off the O...O axis, acceptor bisector 57 degrees from it;".into(),
                "monomers r(OH) 0.9572 A, HOH 104.52 degrees (Hoy & Bunker, J. Mol. Spectrosc. 74, 1 (1979)). The start is built".into(),
                "from these parameters (not a literature Cartesian set); the geometry below is this solver's minimum, not experiment.".into(),
            ]),
            "trimer" => (trimer_start(), "h2o_trimer.scene", TARGET_MARGIN, vec![
                "Cyclic water trimer (uud), relaxed on this solver's default surface: FIRE (the app's Relax, F_TOL 1e-3 Ha/Bohr".into(),
                "per atom) at the Accurate grid (h = sigma = 0.15 Bohr) under the accepted pseudopotentials, GPU teacher at tol 1e-6.".into(),
                "Start: O atoms on a triangle of side R(O...O) = 2.80 A (about the MP2 cyclic-trimer value of Xantheas & Dunning,".into(),
                "J. Chem. Phys. 99, 8774 (1993)), linear H bonds along the ring, free H 60 degrees out of the ring plane (up, up,".into(),
                "down); monomers r(OH) 0.9572 A, HOH 104.52 degrees (Hoy & Bunker 1979). The start is built from these parameters".into(),
                "(not a literature Cartesian set); the geometry below is this solver's minimum, not experiment.".into(),
            ]),
            other => panic!("RELAX_CLUSTERS: unknown cluster '{other}'"),
        };
        let (mut t, log) = Locked::new(ctx.clone(), TOL_H2);
        let wall = Instant::now();
        let mut sc = start.clone();
        let r0 = solve(&mut t, &sc, "relax start");
        let mut forces = r0.forces;
        let mut fire = Fire::new(FireParams::for_dt(units::fs_to_au_time(DT_FS)), r0.energy);
        let pins = vec![false; sc.nuclei.len()];
        let outcome = loop {
            let st = fire.step(&mut sc, &mut forces, &pins, &mut t, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("{name}: FIRE step {}: {e}", fire.steps()));
            if fire.steps() % 25 == 0 {
                println!("  {name} FIRE {}: E {:.6} Ha, max|F| {:.2e}, Δt {:.2} fs", fire.steps(), st.energy, st.max_force_augmented, units::au_time_to_fs(fire.dt()));
            }
            if st.outcome != FireOutcome::Running {
                break (st.outcome, st.energy, st.max_force_augmented);
            }
        };
        for n in &mut sc.nuclei {
            n.vel = [0.0; 3];
        }
        let relaxed = boxed(sc.nuclei.clone(), margin);
        let check = solve(&mut t, &relaxed, "relaxed, re-centred");
        let fmax = sim::relax::max_norm(&check.forces);
        let o: Vec<[f64; 3]> = relaxed.nuclei.iter().filter(|n| n.z == 8).map(|n| n.pos).collect();
        let oo: Vec<String> = (0..o.len()).flat_map(|i| (i + 1..o.len()).map(move |j| (i, j))).map(|(i, j)| {
            let d = ((0..3).map(|k| (o[i][k] - o[j][k]).powi(2)).sum::<f64>()).sqrt();
            format!("{d:.3} Bohr ({:.3} Å)", units::bohr_to_angstrom(d))
        }).collect();
        println!("{name}: {:?} after {} FIRE steps in {:.0} s: E {:.6} → {:.6} Ha (re-centred in {}³: E {:.6}, max|F| {fmax:.2e}, face density {:?}); O···O {oo:?}; {}",
                 outcome.0, fire.steps(), wall.elapsed().as_secs_f64(), r0.energy, outcome.1, relaxed.cell.n, check.energy, check.stats.face_density, log.lock().unwrap().summary());
        assert_eq!(outcome.0, FireOutcome::Converged, "{name}: FIRE ended {:?} at max|F| {:.2e}", outcome.0, outcome.2);
        let mut comment = cite;
        comment.push(format!("Relaxed: {} FIRE steps, E = {:.6} Ha, max|F| = {fmax:.1e} Ha/Bohr re-centred; O...O {}.", fire.steps(), check.energy, oo.join(", ")));
        comment.push(format!("Box: the nuclei's extent + {margin} Bohr per side at h = {H_ACC} ({}^3). Written by validate/tests/transfer.rs relax_the_water_clusters.", relaxed.cell.n));
        let path = scenes_dir().join(file);
        std::fs::write(&path, scene_text(&relaxed, &comment)).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("wrote {}", path.display());
    }
}
