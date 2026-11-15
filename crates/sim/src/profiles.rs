//! Isolated-atom radial density profiles (moved from the app's cloud, live-orbitals Task 3): one
//! spherically averaged density per species, solved by the provider's own teacher on the scene's
//! grid. The app's atomic-superposition cloud (`app::cloud`) splats them at the nuclei for display,
//! and a Harris display solve ([`crate::provider::HarrisAtoms`]) superposes them into its input
//! density. Nothing here is per element beyond Z (rule 1): the profiles are the solver's answers.
//! Atomic units (Bohr, e/Bohr³).
use crate::scene::IonicPolicy;

/// A spherically averaged isolated-atom density (shells of width h/2; e/Bohr³), solved on an
/// n³ grid of spacing h with nuclear smearing σ (`None`: h) — the key it is valid for.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomProfile {
    pub z: u8,
    pub n: usize,
    pub h: f64,
    pub sigma: Option<f64>,
    /// The ionic policy the atom was solved under (M7): an O profile of the all-electron atom (8 electrons, a σ-wide core)
    /// is not the pseudo-O's (6) — a policy switch drops the profiles of the other one. The policy decides which species
    /// are pseudised, and so the provenance tag of the lone atom, for a given species.
    pub ionic: IonicPolicy,
    /// The shells' mean radii (Bohr, increasing) and mean densities (e/Bohr³).
    pub r: Vec<f64>,
    pub rho: Vec<f64>,
    /// Σ ρ h³ over the grid points within the profile's reach: the electrons it carries.
    pub electrons: f64,
}

/// Profile values below this fraction of the peak are dropped (the splat's reach; what they
/// carry is < 1e-4 of an electron for the atoms here).
const TAIL_FRACTION: f64 = 1e-6;

/// The radial profile of an atom solved at the origin of a centred n³ grid of spacing h whose
/// first point is at `origin` (values in `qm::Grid` layout, x fastest): the grid points binned in
/// shells of width h/2 around the origin, each shell's mean radius and mean density (the mean
/// radius, not the shell's centre: more points sit at its outer edge), out to the largest sphere
/// inside the cell, the tail below [`TAIL_FRACTION`] of the peak cut.
pub fn radial_profile(z: u8, rho: &[f32], n: usize, h: f64, origin: [f64; 3], sigma: Option<f64>, ionic: IonicPolicy) -> AtomProfile {
    assert_eq!(rho.len(), n * n * n, "radial_profile: {} values for {n}³", rho.len());
    let dr = 0.5 * h;
    let r_max = (n as f64 - 1.0) * h / 2.0;
    let bins = (r_max / dr).floor() as usize + 1;
    let (mut sum, mut rsum, mut count) = (vec![0.0f64; bins], vec![0.0f64; bins], vec![0usize; bins]);
    for k in 0..n {
        let zc = origin[2] + k as f64 * h;
        for j in 0..n {
            let yc = origin[1] + j as f64 * h;
            for i in 0..n {
                let xc = origin[0] + i as f64 * h;
                let r = (xc * xc + yc * yc + zc * zc).sqrt();
                let b = (r / dr).floor() as usize;
                if r <= r_max && b < bins {
                    sum[b] += rho[i + n * (j + n * k)] as f64;
                    rsum[b] += r;
                    count[b] += 1;
                }
            }
        }
    }
    let (mut rs, mut vs) = (Vec::with_capacity(bins), Vec::with_capacity(bins));
    let mut kept = Vec::with_capacity(bins);
    for b in (0..bins).filter(|&b| count[b] > 0) {
        rs.push(rsum[b] / count[b] as f64);
        vs.push((sum[b] / count[b] as f64).max(0.0));
        kept.push(b);
    }
    let peak = vs.iter().cloned().fold(0.0f64, f64::max);
    while vs.len() > 2 && vs.last().is_some_and(|&v| v < TAIL_FRACTION * peak) {
        vs.pop();
        rs.pop();
        kept.pop();
    }
    let electrons = kept.iter().map(|&b| sum[b]).sum::<f64>() * h * h * h;
    AtomProfile { z, n, h, sigma, ionic, r: rs, rho: vs, electrons }
}

impl AtomProfile {
    /// ρ(r) by linear interpolation between the shells' mean radii (the innermost value inside
    /// the first); 0 beyond the last shell.
    pub fn at(&self, r: f64) -> f64 {
        if r <= self.r[0] {
            return self.rho[0];
        }
        if r > self.reach() {
            return 0.0;
        }
        let k = self.r.partition_point(|&x| x < r).max(1);
        let f = (r - self.r[k - 1]) / (self.r[k] - self.r[k - 1]);
        self.rho[k - 1] * (1.0 - f) + self.rho[k] * f
    }

    /// The profile's reach (Bohr): the last shell's mean radius.
    pub fn reach(&self) -> f64 {
        *self.r.last().expect("a profile has shells")
    }

    /// Valid for a scene solved on this grid, smearing and ionic policy.
    pub fn fits(&self, n: usize, h: f64, sigma: Option<f64>, ionic: IonicPolicy) -> bool {
        self.n == n && self.h == h && self.sigma == sigma && self.ionic == ionic
    }
}

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

    /// The profile of an analytic 1s atom reproduces it (shell means of e^{−2r}/π) and its
    /// electron count (the profile half of the app cloud's
    /// `superposition_integrates_to_the_electrons_and_peaks_at_the_nuclei`, moved with it).
    #[test]
    fn a_1s_profile_reproduces_the_density_and_its_electrons() {
        let (n, h) = (61usize, 0.2);
        let p = radial_profile(1, &h1s(n, h), n, h, origin(n, h), None, IonicPolicy::None);
        println!("1s profile: {} shells to {:.2} Bohr, ρ(0) {:.4} (1/π = {:.4}), ρ(1) {:.5} (e⁻²/π = {:.5}), electrons within its reach {:.5}",
                 p.rho.len(), p.reach(), p.at(0.0), 1.0 / std::f64::consts::PI, p.at(1.0), (-2.0f64).exp() / std::f64::consts::PI, p.electrons);
        // shell means at the mean radius: the convexity of e^{−2r} over a h/2 shell, ≤ 1 %
        assert!((p.at(1.0) / ((-2.0f64).exp() / std::f64::consts::PI) - 1.0).abs() < 1e-2, "{}", p.at(1.0));
        // the grid's quadrature of the cusped 1s (0.4 % low at h = 0.2), and e^{−12}·85 beyond 6 Bohr
        assert!((p.electrons - 1.0).abs() < 1e-2, "{}", p.electrons);
    }

    /// M7 Task 9 review I1: a profile is valid for the ionic policy it was solved under — a switch of Physics ▸ Ionic
    /// potential on the same grid drops it (the O atom of 8 electrons is not the pseudo-O of 6).
    #[test]
    fn a_profile_fits_only_its_own_ionic_policy() {
        let (n, h) = (33usize, 0.25);
        for solved in IonicPolicy::ALL {
            let p = radial_profile(1, &h1s(n, h), n, h, origin(n, h), None, solved);
            for asked in IonicPolicy::ALL {
                assert_eq!(p.fits(n, h, None, asked), solved == asked, "{solved:?} vs {asked:?}");
            }
            assert!(!p.fits(n + 2, h, None, solved) && !p.fits(n, h, Some(0.2), solved));
        }
    }
}
