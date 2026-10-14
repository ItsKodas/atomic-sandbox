//! Generates the H₃⁺ training fixture of the surrogate crate (M5 Task 6):
//! `crates/surrogate/fixtures/h3plus_33.json`. Run ONCE on the GPU and commit the output:
//!
//! ```text
//! cargo run -p validate --release --example gen_h3plus [--randomise-rigid] [--out <path>]
//! ```
//!
//! `--randomise-rigid` applies a random rigid translation (uniform in ±h/2 per axis) and a
//! Haar-random rotation to every geometry, so the teacher's egg-box error is independent
//! between geometries and a held-out rmse honestly contains the floor (judge the tier as
//! √(rmse² − floor²)). The committed `h3plus_33.json` was generated WITHOUT it (centroid at
//! the origin, no rotation): on that fixture the egg-box ripple is a learnable function of the
//! internal coordinates, which is why its window energy rmse sits below the floor (Task 6
//! review); it is not regenerated, because its numbers are pinned by `tests/body3.rs`. Use the
//! option for the next fixture (h ≤ 0.2). `--out` overrides the output path.
//!
//! What it does, all with `providers::GpuKs` at 33³, h = σ = 0.25 Bohr, tol 1e-6, every solve cold
//! (so the file does not depend on the order of the geometries):
//!
//! 1. the isolated H reference (neutral, multiplicity 2, cell centre) — the `ref_energy` the
//!    fit subtracts;
//! 2. a scan of the equilateral side a (in-plane triangle centred on the origin) to locate the
//!    teacher's own H₃⁺ minimum a₀ at this grid and σ (parabola through the three lowest
//!    points) — the fixture's geometries are perturbations of THAT minimum, not of experiment;
//! 3. an egg-box probe: the a₀ triangle rigidly shifted by eight sub-grid vectors — the rms
//!    scatter of E and of the force components is the teacher's own noise floor at 33³, which
//!    bounds what any fit of these data can reach (recorded in the header, CLAUDE.md rule 7);
//! 4. 80 geometries: 10 equilateral (a₀ ± 0.3), 20 isosceles (apex along the symmetry axis
//!    ± 0.3), 40 scalene (every atom displaced by a random vector of length ≤ 0.3 Bohr, out of
//!    plane included), 10 stretched (one atom pulled 0.4–1.0 Bohr outward, the others jittered
//!    ≤ 0.1); pairs closer than 0.9 Bohr are redrawn. Charge +1, multiplicity 1.
//!
//! The output is minimal JSON written by hand (the surrogate crate has no serde): a
//! `provenance` object, `reference_h` (Hartree), then `geometries`, each with `kind`, `pos`
//! (Bohr, 3×3), `energy` (Hartree, total Kohn-Sham) and `forces` (Hartree/Bohr, 3×3). Numbers
//! are printed with 13 significant digits (the teacher's noise floor is ~1e-5; file ≤ 50 KB).
//! The reader in `crates/surrogate/tests/body3.rs` relies on `reference_h` and every
//! provenance number coming BEFORE the `geometries` key and on 19 numbers per geometry in the
//! order pos, energy, forces.

use gpu::GpuContext;
use providers::GpuKs;
use sim::{Cell, ForceProvider, Nucleus, Scene};
use std::fmt::Write as _;
use std::time::Instant;

const N_GRID: usize = 33;
const H_GRID: f64 = 0.25;
const SIGMA: f64 = 0.25;
const TOL: f64 = 1e-6;
const SEED: u64 = 0x2026_1014_4833;

/// xorshift64* — a self-contained, reproducible generator (no `rand` dependency in `validate`).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    /// Haar-random rotation from a uniformly random unit quaternion.
    fn rotation(&mut self) -> [[f64; 3]; 3] {
        let q = loop {
            let q = [self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0)];
            let n2: f64 = q.iter().map(|c| c * c).sum();
            if n2 > 0.1 && n2 <= 1.0 {
                let n = n2.sqrt();
                break q.map(|c| c / n);
            }
        };
        let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    }

    /// Uniform in the ball of radius `r`.
    fn ball(&mut self, r: f64) -> [f64; 3] {
        loop {
            let v = [self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0)];
            let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if n2 <= 1.0 {
                return [r * v[0], r * v[1], r * v[2]];
            }
        }
    }
}

fn scene(pos: &[[f64; 3]], z: &[u8], charge: i32, multiplicity: u32) -> Scene {
    Scene {
        nuclei: pos.iter().zip(z).map(|(&p, &zz)| Nucleus::new(zz, p)).collect(),
        cell: Cell { n: N_GRID, h: H_GRID },
        charge,
        multiplicity: Some(multiplicity),
        sigma: Some(SIGMA),
        external_field: [0.0; 3],
    }
}

/// Equilateral triangle of side `a` in the xy-plane, centroid at the origin.
fn equilateral(a: f64) -> Vec<[f64; 3]> {
    let r = a / 3f64.sqrt();
    (0..3)
        .map(|k| {
            let th = std::f64::consts::FRAC_PI_2 + k as f64 * 2.0 * std::f64::consts::PI / 3.0;
            [r * th.cos(), r * th.sin(), 0.0]
        })
        .collect()
}

fn min_pair_distance(pos: &[[f64; 3]]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            let d: f64 = (0..3).map(|a| (pos[i][a] - pos[j][a]).powi(2)).sum::<f64>().sqrt();
            m = m.min(d);
        }
    }
    m
}

struct Teacher {
    ks: GpuKs,
    solves: usize,
    wall: f64,
}

impl Teacher {
    /// One cold solve of H₃⁺ (charge +1, singlet) or, with `charge = 0`, of whatever is passed.
    fn solve(&mut self, pos: &[[f64; 3]], charge: i32, multiplicity: u32) -> (f64, Vec<[f64; 3]>) {
        let z = vec![1u8; pos.len()];
        let sc = scene(pos, &z, charge, multiplicity);
        self.ks.reset();
        let t = Instant::now();
        let r = self.ks.evaluate(&sc, &mut || false).unwrap_or_else(|e| panic!("teacher failed at {pos:?}: {e}"));
        self.wall += t.elapsed().as_secs_f64();
        self.solves += 1;
        assert!(r.energy.is_finite() && r.forces.iter().flatten().all(|f| f.is_finite()), "non-finite teacher result");
        (r.energy, r.forces)
    }
}

fn num(x: f64) -> String {
    format!("{x:.12e}")
}

fn vec3(v: &[f64; 3]) -> String {
    format!("[{},{},{}]", num(v[0]), num(v[1]), num(v[2]))
}

fn mat(m: &[[f64; 3]]) -> String {
    format!("[{}]", m.iter().map(vec3).collect::<Vec<_>>().join(","))
}

/// `git rev-parse --short HEAD` of the working tree, for the provenance header.
fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let randomise_rigid = args.iter().any(|a| a == "--randomise-rigid");
    let out_path = args.iter().position(|a| a == "--out").map(|i| std::path::PathBuf::from(&args[i + 1]));
    let Some(ctx) = GpuContext::new_headless() else {
        eprintln!("gen_h3plus: no GPU adapter");
        std::process::exit(2);
    };
    let adapter = ctx.adapter.get_info();
    let mut ks = GpuKs::new(ctx, TOL);
    ks.set_display(false);
    let mut teacher = Teacher { ks, solves: 0, wall: 0.0 };
    let t_all = Instant::now();

    // 1. the H reference (neutral doublet at the cell centre)
    let (e_h, _) = teacher.solve(&[[0.0; 3]], 0, 2);
    eprintln!("H reference (multiplicity 2): {e_h:.8} Ha");

    // 2. the equilateral scan
    let sides: Vec<f64> = (0..15).map(|k| 1.60 + 0.10 * k as f64).collect();
    let mut scan = Vec::with_capacity(sides.len());
    for &a in &sides {
        let (e, _) = teacher.solve(&equilateral(a), 1, 1);
        eprintln!("  a = {a:.3}: E = {e:.8}");
        scan.push((a, e));
    }
    let i_min = (0..scan.len()).min_by(|&i, &j| scan[i].1.partial_cmp(&scan[j].1).unwrap()).unwrap();
    assert!(i_min > 0 && i_min + 1 < scan.len(), "scan minimum on the edge: widen the scan");
    let (x0, y0) = scan[i_min - 1];
    let (x1, y1) = scan[i_min];
    let (x2, y2) = scan[i_min + 1];
    // parabola through three points: vertex
    let d = (x0 - x1) * (x0 - x2) * (x1 - x2);
    let a_coef = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d;
    let b_coef = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d;
    let a0 = -b_coef / (2.0 * a_coef);
    let (e0, f0) = teacher.solve(&equilateral(a0), 1, 1);
    let f0_max = f0.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    eprintln!("equilateral minimum a0 = {a0:.5} Bohr (E = {e0:.8} Ha, max |F| = {f0_max:.2e})");

    // 3. the egg-box probe: rigid sub-grid shifts of the minimum
    let mut rng = Rng(SEED);
    let base = equilateral(a0);
    let mut es = vec![e0];
    let mut fs = vec![f0.clone()];
    for _ in 0..7 {
        let shift = [rng.range(0.0, H_GRID), rng.range(0.0, H_GRID), rng.range(0.0, H_GRID)];
        let pos: Vec<[f64; 3]> = base.iter().map(|p| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]).collect();
        let (e, f) = teacher.solve(&pos, 1, 1);
        es.push(e);
        fs.push(f);
    }
    let n_shift = es.len() as f64;
    let e_mean = es.iter().sum::<f64>() / n_shift;
    let eggbox_e = (es.iter().map(|e| (e - e_mean).powi(2)).sum::<f64>() / (n_shift - 1.0)).sqrt();
    // force scatter: the rigid shift leaves the true forces unchanged, so the rms of F about
    // the shift-mean per component is the egg-box force noise
    let mut ss = 0.0;
    let mut cnt = 0.0;
    for j in 0..3 {
        for a in 0..3 {
            let m = fs.iter().map(|f| f[j][a]).sum::<f64>() / n_shift;
            ss += fs.iter().map(|f| (f[j][a] - m).powi(2)).sum::<f64>();
            cnt += n_shift - 1.0;
        }
    }
    let eggbox_f = (ss / cnt).sqrt();
    eprintln!("egg-box scatter over {} rigid shifts: E {eggbox_e:.2e} Ha, F {eggbox_f:.2e} Ha/Bohr", es.len());

    // 4. the geometries
    let mut geoms: Vec<(&'static str, Vec<[f64; 3]>)> = Vec::with_capacity(80);
    let push = |kind: &'static str, pos: Vec<[f64; 3]>, geoms: &mut Vec<(&'static str, Vec<[f64; 3]>)>| {
        assert!(min_pair_distance(&pos) > 0.9, "{kind}: pair closer than 0.9 Bohr");
        geoms.push((kind, pos));
    };
    for k in 0..10 {
        let da = -0.3 + 0.6 * k as f64 / 9.0;
        push("equilateral", equilateral(a0 + da), &mut geoms);
    }
    for k in 0..20 {
        let dz = -0.3 + 0.6 * k as f64 / 19.0;
        let mut pos = equilateral(a0);
        // apex is vertex 0 (on +y); move it along the symmetry axis
        pos[0][1] += dz;
        push("isosceles", pos, &mut geoms);
    }
    let mut n_scalene = 0;
    while n_scalene < 40 {
        let pos: Vec<[f64; 3]> = base
            .iter()
            .map(|p| {
                let d = rng.ball(0.3);
                [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
            })
            .collect();
        if min_pair_distance(&pos) > 0.9 {
            push("scalene", pos, &mut geoms);
            n_scalene += 1;
        }
    }
    for k in 0..10 {
        let pull = 0.4 + 0.6 * k as f64 / 9.0;
        let which = k % 3;
        let mut pos: Vec<[f64; 3]> = base
            .iter()
            .map(|p| {
                let d = rng.ball(0.1);
                [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
            })
            .collect();
        let r = (base[which][0].powi(2) + base[which][1].powi(2)).sqrt();
        pos[which][0] += pull * base[which][0] / r;
        pos[which][1] += pull * base[which][1] / r;
        push("stretched", pos, &mut geoms);
    }
    assert_eq!(geoms.len(), 80);
    if randomise_rigid {
        for (_, pos) in geoms.iter_mut() {
            let rot = rng.rotation();
            let t = [rng.range(-0.5 * H_GRID, 0.5 * H_GRID), rng.range(-0.5 * H_GRID, 0.5 * H_GRID), rng.range(-0.5 * H_GRID, 0.5 * H_GRID)];
            for p in pos.iter_mut() {
                let v = [
                    rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2],
                    rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2],
                    rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2],
                ];
                *p = [v[0] + t[0], v[1] + t[1], v[2] + t[2]];
            }
        }
    }

    let mut rows = Vec::with_capacity(80);
    for (i, (kind, pos)) in geoms.iter().enumerate() {
        let (e, f) = teacher.solve(pos, 1, 1);
        eprintln!("  [{i:2}] {kind:<11} E = {e:.8}  max|F| = {:.3e}", f.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())));
        rows.push((kind, pos, e, f));
    }

    // 5. write
    let mut out = String::new();
    let _ = writeln!(out, "{{");
    let _ = writeln!(out, "\"provenance\":{{");
    let _ = writeln!(out, "  \"generator\":\"crates/validate/examples/gen_h3plus.rs (M5 Task 6)\",\"commit\":\"{}\",\"randomise_rigid\":{randomise_rigid},", git_commit());
    let _ = writeln!(out, "  \"teacher\":\"{}\",", teacher.ks.name());
    let _ = writeln!(out, "  \"adapter\":\"{} ({:?})\",", adapter.name, adapter.backend);
    let _ = writeln!(out, "  \"grid_n\":{N_GRID},\"h\":{H_GRID},\"sigma\":{SIGMA},\"tol\":{TOL:e},");
    let _ = writeln!(out, "  \"charge\":1,\"multiplicity\":1,\"solves\":\"every geometry cold\",");
    let _ = writeln!(out, "  \"reference_h_multiplicity\":2,");
    let _ = writeln!(out, "  \"equilateral_scan\":[{}],", scan.iter().map(|(a, e)| format!("[{a:.3},{}]", num(*e))).collect::<Vec<_>>().join(","));
    let _ = writeln!(out, "  \"a0\":{},\"e_min\":{},", num(a0), num(e0));
    let _ = writeln!(out, "  \"eggbox_rms_energy\":{},\"eggbox_rms_force\":{},\"eggbox_shifts\":{},", num(eggbox_e), num(eggbox_f), es.len());
    let _ = writeln!(out, "  \"seed\":\"{SEED:#x}\",\"n_geometries\":80,");
    let _ = writeln!(out, "  \"kinds\":\"10 equilateral (a0 ± 0.3), 20 isosceles (apex ± 0.3 on the axis), 40 scalene (each atom ≤ 0.3 in a ball), 10 stretched (one atom pulled 0.4–1.0 outward, others ≤ 0.1)\",");
    let _ = writeln!(out, "  \"units\":\"Bohr, Hartree, Hartree/Bohr\"");
    let _ = writeln!(out, "}},");
    let _ = writeln!(out, "\"reference_h\":{},", num(e_h));
    let _ = writeln!(out, "\"geometries\":[");
    for (i, (kind, pos, e, f)) in rows.iter().enumerate() {
        let comma = if i + 1 < rows.len() { "," } else { "" };
        let _ = writeln!(out, "{{\"kind\":\"{kind}\",\"pos\":{},\"energy\":{},\"forces\":{}}}{comma}", mat(pos), num(*e), mat(f));
    }
    let _ = writeln!(out, "]}}");

    let path = out_path.unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../surrogate/fixtures/h3plus_33.json"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &out).unwrap();
    eprintln!(
        "wrote {} ({} bytes); {} solves, {:.1} s solving, {:.1} s total",
        path.display(),
        out.len(),
        teacher.solves,
        teacher.wall,
        t_all.elapsed().as_secs_f64()
    );
}
