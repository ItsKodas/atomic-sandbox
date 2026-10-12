//! First pictures from the CPU Kohn-Sham solver: the H₂ binding curve E(R) and the electron
//! density, computed from nothing but Z = 1 for each nucleus.
//!
//! Run from the repo root:
//!   cargo run -p qm --release --example h2_picture
//!   python tools/plot_h2.py
//!
//! Writes (relative to the CWD, `out/` is git-ignored):
//!   out/h2_energy_curve.csv            R_bohr, R_angstrom, E_hartree, E_minus_Einf_eV, scf_steps
//!   out/h2_density_z0.csv              x, y, rho on the z = 0 plane at R = 1.4 Bohr
//!   out/h2_density_z0_stretched.csv    same at R = 3.2 Bohr
//!   out/h2_density_49x49x49.f32        full 3D density at R = 1.4, raw little-endian f32,
//!                                      layout idx = i + n*(j + n*k) (x fastest)
//!   out/h2_density.meta                one line: n h origin_x origin_y origin_z (Bohr)
//!
//! Physics notes (see crates/validate/references.md):
//! - At σ = h = 0.25 Bohr the Gaussian-smeared nuclei shift the minimum to R ≈ 1.8 Bohr rather
//!   than the physical 1.45 Bohr (LSDA, point nuclei). That is the documented smearing shift on
//!   this coarse grid, not a bug; the validate suite extrapolates σ → 0 on finer grids.
//! - The scan is spin-restricted (singlet, n_up = n_dn = 1), so at large R it does not
//!   dissociate to two H atoms (the well-known restricted-KS static-correlation error).
//!   E(5.0) is therefore only a convenient "separated" reference for the eV column, not
//!   2·E(H).

use qm::{scf, Grid, KsState, Nucleus, ScfSettings, System};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

const N: usize = 49;
const H: f64 = 0.25;
const R_SCAN: [f64; 12] = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 4.0, 5.0];
const R_BONDED: f64 = 1.4;
const R_STRETCHED: f64 = 3.2;

fn h2(r: f64) -> System {
    System::new(
        vec![Nucleus { pos: [r / 2.0, 0.0, 0.0], z: 1 }, Nucleus { pos: [-r / 2.0, 0.0, 0.0], z: 1 }],
        0,
        None,
    )
}

/// Density on the k = n/2 plane (z = 0 exactly for odd n) as x, y, rho rows.
fn write_plane(g: &Grid, rho: &[f64], path: &str) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "x,y,rho")?;
    let k = g.n[2] / 2;
    for j in 0..g.n[1] {
        for i in 0..g.n[0] {
            let idx = g.idx(i, j, k);
            let p = g.point(idx);
            writeln!(w, "{:.6},{:.6},{:.9e}", p[0], p[1], rho[idx])?;
        }
    }
    Ok(())
}

fn write_volume(g: &Grid, rho: &[f64]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("out/h2_density_{0}x{0}x{0}.f32", g.n[0]))?);
    for &v in rho {
        w.write_all(&(v as f32).to_le_bytes())?;
    }
    let mut m = File::create("out/h2_density.meta")?;
    writeln!(m, "{} {} {} {} {}", g.n[0], g.h, g.origin[0], g.origin[1], g.origin[2])?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all("out")?;
    let g = Grid::cube(N, H);
    let settings = ScfSettings::for_grid(&g);
    let t_all = Instant::now();

    let mut rows: Vec<(f64, f64, usize, f64)> = Vec::new(); // (R, E, steps, seconds)
    let mut prev: Option<KsState> = None;
    for &r in &R_SCAN {
        let t = Instant::now();
        let (state, stats) = scf(&g, &h2(r), &settings, prev.as_ref())?;
        let secs = t.elapsed().as_secs_f64();
        for w in &stats.warnings {
            eprintln!("  R = {r:.2}: warning: {w}");
        }
        println!("R = {r:4.2} Bohr  E = {:+.6} Ha  steps = {:3}  ({secs:.1} s)", stats.energy.total, stats.iterations);
        rows.push((r, stats.energy.total, stats.iterations, secs));

        if r == R_BONDED {
            let rho = state.rho();
            println!("  ∫ρ = {:.6} electrons", g.integrate(&rho));
            write_plane(&g, &rho, "out/h2_density_z0.csv")?;
            write_volume(&g, &rho)?;
        }
        if r == R_STRETCHED {
            write_plane(&g, &state.rho(), "out/h2_density_z0_stretched.csv")?;
        }
        prev = Some(state);
    }

    let e_inf = rows.last().unwrap().1;
    let mut w = BufWriter::new(File::create("out/h2_energy_curve.csv")?);
    writeln!(w, "R_bohr,R_angstrom,E_hartree,E_minus_Einf_eV,scf_steps")?;
    for &(r, e, steps, _) in &rows {
        writeln!(w, "{r},{:.6},{e:.9},{:.6},{steps}", units::bohr_to_angstrom(r), units::hartree_to_ev(e - e_inf))?;
    }
    w.flush()?;

    println!();
    println!("  R (Bohr)  R (Å)    E (Ha)       E−E(5.0) (eV)  steps   time");
    for &(r, e, steps, secs) in &rows {
        println!("  {r:7.2}  {:6.3}  {e:+.6}   {:+9.4}      {steps:4}  {secs:5.1} s",
            units::bohr_to_angstrom(r), units::hartree_to_ev(e - e_inf));
    }
    let (r_min, e_min, _, _) = rows.iter().cloned().fold((0.0, f64::INFINITY, 0, 0.0), |a, b| if b.1 < a.1 { b } else { a });
    println!();
    println!("Scan minimum: R = {r_min} Bohr ({:.3} Å), depth {:.3} eV below E(5.0)",
        units::bohr_to_angstrom(r_min), units::hartree_to_ev(e_inf - e_min));
    println!("(σ = h = {H} Bohr smeared nuclei: minimum is shifted from the physical 1.45 Bohr.)");
    println!("Total wall time: {:.1} s", t_all.elapsed().as_secs_f64());
    Ok(())
}
