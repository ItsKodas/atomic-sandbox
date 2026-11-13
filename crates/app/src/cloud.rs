//! The estimated solver's cloud between DFT densities (M5b, "Density in estimated mode"): an
//! **atomic superposition** — the isolated-atom densities of the solver's own teacher (one solve
//! per species on the scene's cell, run by the surrogate's background auditor:
//! `providers::SurrogateKs::request_atom_density`), spherically averaged into radial profiles and placed
//! at the current nuclei. It is display only and labelled "atomic superposition — not solved": it
//! shows where the atoms are, not their bonds (no charge moves between atoms), and the viewer
//! blends it with the last real density by how far the atoms have moved since that one was solved
//! ([`staleness`]). Nothing here is per element beyond Z (rule 1): the profiles are the teacher's
//! answers. Atomic units (Bohr, e/Bohr³). The isolated atom is solved at the solver's lowest
//! multiplicity (`multiplicity: None`), not the lowest-energy one the sampling pass's references
//! use (ruling S9): identical for H, He, Li; for N or O it may be another spin state — display
//! only (review M5).

pub use sim::{radial_profile, AtomProfile};

/// The superposition Σ_atoms ρ_Z(|r − R|) on the centred n³ grid of spacing h (first point at
/// `origin`; values in `qm::Grid` layout) — `None` when a nucleus's species has no profile in
/// `profiles`. Not renormalised: an atom near the face loses what falls outside, like a real
/// density would.
pub fn superpose(profiles: &[AtomProfile], nuclei: &[(u8, [f64; 3])], n: usize, h: f64, origin: [f64; 3]) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; n * n * n];
    for &(z, pos) in nuclei {
        let p = profiles.iter().find(|p| p.z == z)?;
        let reach = p.reach();
        let range = |a: usize| {
            let lo = ((pos[a] - reach - origin[a]) / h).floor().max(0.0) as usize;
            let hi = (((pos[a] + reach - origin[a]) / h).ceil().max(0.0) as usize).min(n - 1);
            lo..=hi
        };
        let (rx, ry, rz) = (range(0), range(1), range(2));
        for k in rz {
            let dz = origin[2] + k as f64 * h - pos[2];
            for j in ry.clone() {
                let dy = origin[1] + j as f64 * h - pos[1];
                for i in rx.clone() {
                    let dx = origin[0] + i as f64 * h - pos[0];
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r <= reach {
                        out[i + n * (j + n * k)] += p.at(r) as f32;
                    }
                }
            }
        }
    }
    Some(out)
}

/// How far the atoms have moved from where a real density was solved, as the superposition's
/// weight in the blend: max_i |r_i − r_i⁰| / [`BLEND_DISTANCE`], clamped to [0, 1]; 1 when the
/// atoms are not the same ones (count changed). A display heuristic, not physics.
pub fn staleness(solved_at: &[[f64; 3]], now: &[[f64; 3]]) -> f32 {
    if solved_at.len() != now.len() {
        return 1.0;
    }
    let worst = solved_at
        .iter()
        .zip(now)
        .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    (worst / BLEND_DISTANCE).clamp(0.0, 1.0) as f32
}

/// The displacement (Bohr) at which a real density has fully given way to the superposition: a
/// bond-length-scale move makes a solved cloud visibly wrong.
pub const BLEND_DISTANCE: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    /// A centred grid's first point.
    fn origin(n: usize, h: f64) -> [f64; 3] {
        [-(n as f64 - 1.0) * h / 2.0; 3]
    }

    /// The hydrogen 1s density e^{−2r}/π at the origin of an n³ grid.
    fn h1s(n: usize, h: f64) -> Vec<f32> {
        let o = origin(n, h);
        let mut v = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = [o[0] + i as f64 * h, o[1] + j as f64 * h, o[2] + k as f64 * h];
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    v.push(((-2.0 * r).exp() / std::f64::consts::PI) as f32);
                }
            }
        }
        v
    }

    /// The superposition of two analytic 1s atoms (their profile: `sim::profiles`'
    /// `a_1s_profile_reproduces_the_density_and_its_electrons`) at off-grid positions integrates
    /// to the sum of the profiles' electrons (the splat neither loses nor invents charge) — within
    /// the grid's quadrature of a cusped density — and peaks at the nuclei.
    #[test]
    fn superposition_integrates_to_the_electrons_and_peaks_at_the_nuclei() {
        let (n, h) = (61usize, 0.2);
        let p = radial_profile(1, &h1s(n, h), n, h, origin(n, h), None, sim::IonicPolicy::None);
        // two atoms, off the grid points, well inside a 12 Bohr box
        let (m, o) = (81usize, origin(81, 0.2));
        let nuclei = [(1u8, [-0.73, 0.05, 0.0]), (1u8, [0.71, -0.03, 0.02])];
        let rho = superpose(std::slice::from_ref(&p), &nuclei, m, h, o).expect("H has a profile");
        let total: f64 = rho.iter().map(|&v| v as f64).sum::<f64>() * h * h * h;
        println!("superposition of two: Σρh³ = {total:.5} (2 × profile {:.5})", p.electrons);
        assert!((total / (2.0 * p.electrons) - 1.0).abs() < 5e-3, "{total}");
        assert!((total - 2.0).abs() < 2e-2, "{total}");
        // peaks at the nuclei: the nearest grid point to each beats everything 0.5 Bohr from both
        let at = |q: [f64; 3]| {
            let idx = |a: usize| ((q[a] - o[a]) / h).round() as usize;
            rho[idx(0) + m * (idx(1) + m * idx(2))]
        };
        let far_max = (0..m * m * m)
            .filter(|&l| {
                let q = [o[0] + (l % m) as f64 * h, o[1] + ((l / m) % m) as f64 * h, o[2] + (l / (m * m)) as f64 * h];
                nuclei.iter().all(|(_, r)| ((q[0] - r[0]).powi(2) + (q[1] - r[1]).powi(2) + (q[2] - r[2]).powi(2)).sqrt() > 0.5)
            })
            .map(|l| rho[l])
            .fold(0.0f32, f32::max);
        for (_, r) in nuclei {
            assert!(at(r) > far_max, "{} vs {far_max}", at(r));
        }
        // a species without a profile: no superposition
        assert!(superpose(&[p], &[(8, [0.0; 3])], m, h, o).is_none());
    }

    #[test]
    fn staleness_grows_with_the_displacement_and_saturates() {
        let a = [[0.0; 3], [1.4, 0.0, 0.0]];
        assert_eq!(staleness(&a, &a), 0.0);
        assert!((staleness(&a, &[[0.0; 3], [1.65, 0.0, 0.0]]) - 0.5).abs() < 1e-6);
        assert_eq!(staleness(&a, &[[0.0; 3], [3.0, 0.0, 0.0]]), 1.0);
        assert_eq!(staleness(&a, &a[..1]), 1.0, "other atoms");
    }
}
