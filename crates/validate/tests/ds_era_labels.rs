//! The one-off generator of `fixtures/ds_era_labels.txt` (plan 2026-11-13 Task 10 Step 1, preflight F13): the labels
//! the last `DegenerateShells` build's GPU teacher gave three V7 scenes, captured on the pre-deletion tree (1d816c4)
//! before `Occupations::DegenerateShells` was deleted. It cannot compile after Task 10 and was removed there; it is
//! kept in git history as the record of how the fixture was made.
//!
//! The teacher as the DS-era `GpuKs` posed it: `providers::ks::problem` under `IonicPolicy::Accepted` (the app's
//! default; with the GPU dimer gate H is pseudised, so the surface is `All`'s and the legacy tag
//! `all/30a74d9bda495668/shared1e-3/1e-1/stop2x5e-8`), the sample tolerance 1e-6 raised to the GPU floor
//! (`effective_gpu_tol`), occupations `DegenerateShells { tau: DEGENERACY_TAU }`, multiplicity auto, a cold solve
//! (bit-reproducible), the forces from the converged state. The H₄ aspect is scanned until the internal energy moves by
//! |ΔE| ≥ 3ε_E = 3e-5 Ha between the DS solve and an FD solve at `ELECTRON_KT` (class S outside the false-S band).
//!
//! Run: `cargo test -p validate --release --test ds_era_labels -- --ignored --nocapture`.
mod common;

use qm::{Occupations, DEGENERACY_TAU, ELECTRON_KT};
use sim::{Cell, IonicPolicy, Nucleus, Scene};
use std::fmt::Write as _;

const N: usize = 61;
const H: f64 = 0.20;
const TOL: f64 = 1e-6;

fn scene(nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: Cell { n: N, h: H }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3],
            ionic: IonicPolicy::Accepted, train_mask: None }
}

struct Label { iterations: usize, e: f64, internal: f64, ts: f64, forces: Vec<[f64; 3]>, m: usize, note: String }

fn solve(ctx: &gpu::GpuContext, s: &Scene, occ: Occupations) -> Label {
    let mut notes = Vec::new();
    let (grid, system, mut settings) = providers::ks::problem(s, TOL, &mut notes).expect("posed");
    assert!(system.ionic.is_some(), "Accepted with the GPU gate pseudises H and O");
    settings.tol = settings.tol.max(providers::pseudo::effective_gpu_tol(TOL, (system.n_up + system.n_dn) as i64));
    settings.occupations = occ;
    let mut solver = qm_gpu::GpuSolver::new(ctx.clone(), &grid);
    let st = solver.solve(&system, &settings, false).unwrap_or_else(|e| panic!("solve: {e}")).1;
    let state = solver.state().expect("converged");
    let forces = solver.forces(state, &system.nuclei, settings.sigma).expect("forces");
    let note = format!("tol {:e}, {} steps, Δρ {:.2e}, fractional {:?}, released {:?}, partial {:?}, notes {:?}",
                       settings.tol, st.iterations, st.delta_rho, st.fractional, st.released, st.partial, notes);
    Label { iterations: st.iterations, e: st.energy.total, internal: st.energy.internal, ts: st.energy.ts, forces, m: system.n_up - system.n_dn + 1, note }
}

fn h4(a: f64, b: f64) -> Vec<Nucleus> {
    let (x, y) = (0.5 * a, 0.5 * b);
    [[x, y], [-x, y], [-x, -y], [x, -y]].iter().map(|&[p, q]| Nucleus::new(1, [p, q, 0.0])).collect()
}

#[test]
#[ignore = "one-off fixture generator (plan Task 10 Step 1); run with --ignored on the pre-deletion tree"]
fn write_ds_era_labels() {
    let Some(ctx) = common::gpu("ds_era_labels") else { panic!("needs the GPU") };
    let ds = Occupations::DegenerateShells { tau: DEGENERACY_TAU };
    let fd = Occupations::FermiDirac { kt: ELECTRON_KT };
    // the H₄ aspect scan: b = a·(1 + c), a = 1.8
    let a = 1.8;
    let mut chosen = None;
    for c in [0.005, 0.01, 0.02, 0.04, 0.083] {
        let s = scene(h4(a, a * (1.0 + c)));
        let d = solve(&ctx, &s, ds);
        let f = solve(&ctx, &s, fd);
        let de = f.internal - d.e;
        println!("H₄ {a} × {:.4} (c {c}): DS E {:+.10} [{}]; FD F {:+.10} E {:+.10} TS {:.3e} [{}]; E_FD − E_DS {de:+.3e}",
                 a * (1.0 + c), d.e, d.note, f.e, f.internal, f.ts, f.note);
        if chosen.is_none() && de.abs() >= 1e-4 { chosen = Some((c, d, f)); }
    }
    let (c, h4_ds, h4_fd) = chosen.expect("an aspect with |ΔE| ≥ 1e-4 (3ε_E = 3e-5 with margin)");
    let h2 = scene(vec![Nucleus::new(1, [-0.7, 0.0, 0.0]), Nucleus::new(1, [0.7, 0.0, 0.0])]);
    let oh = scene(vec![Nucleus::new(8, [0.0; 3]), Nucleus::new(1, [1.83, 0.0, 0.0])]);
    let h4s = scene(h4(a, a * (1.0 + c)));
    let mut out = String::new();
    writeln!(out, "# DS-era labels for V7 (plan 2026-11-13 Task 10 Step 1, F13), captured on 1d816c4 before DegenerateShells was deleted.").unwrap();
    writeln!(out, "# Teacher: providers::ks::problem, cold GPU solve, occupations DegenerateShells {{ tau: 1e-3 }}, multiplicity auto,").unwrap();
    writeln!(out, "# the sample tolerance raised to the GPU floor (effective_gpu_tol), sigma = h; forces from the converged state.").unwrap();
    writeln!(out, "# Units: Hartree, Bohr, Ha/Bohr. f64 values are Rust's shortest round-trip form. Generator: crates/validate/tests/ds_era_labels.rs (git history).").unwrap();
    writeln!(out, "policy accepted").unwrap();
    writeln!(out, "legacy_tag {}", providers::pseudo::legacy_tag_of(IonicPolicy::Accepted)).unwrap();
    writeln!(out, "grid_n {N}").unwrap();
    writeln!(out, "h {H:?}").unwrap();
    writeln!(out, "box {:?}", (N - 1) as f64 * H).unwrap();
    writeln!(out, "tol {TOL:e}").unwrap();
    for (name, s, l) in [("h2", &h2, solve(&ctx, &h2, ds)), ("oh", &oh, solve(&ctx, &oh, ds)), ("h4", &h4s, h4_ds)] {
        writeln!(out, "case {name}").unwrap();
        for (nu, f) in s.nuclei.iter().zip(&l.forces) {
            writeln!(out, "atom {} {:?} {:?} {:?} force {:?} {:?} {:?}", nu.z, nu.pos[0], nu.pos[1], nu.pos[2], f[0], f[1], f[2]).unwrap();
        }
        writeln!(out, "multiplicity {}", l.m).unwrap();
        writeln!(out, "e_ds {:?}", l.e).unwrap();
        writeln!(out, "# {name}: {} steps; {}", l.iterations, l.note).unwrap();
        if name == "h4" {
            writeln!(out, "aspect {:?}", 1.0 + c).unwrap();
            writeln!(out, "fd_internal {:?}", h4_fd.internal).unwrap();
            writeln!(out, "fd_ts {:?}", h4_fd.ts).unwrap();
            writeln!(out, "delta_e {:?}", h4_fd.internal - l.e).unwrap();
            writeln!(out, "# h4 FD at ELECTRON_KT: {} steps; {}", h4_fd.iterations, h4_fd.note).unwrap();
        }
    }
    print!("{out}");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ds_era_labels.txt");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, out).unwrap();
    println!("written {}", path.display());
}
