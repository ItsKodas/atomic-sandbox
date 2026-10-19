//! Partial charges by the **Hirshfeld (stockholder) partition** of the solved density — display
//! only: computed after a converged density (the driver after an evaluation, the app's worker
//! from a side density in estimated mode), shipped in the snapshot, never fed to forces or the
//! solve. Nothing here is per element (CLAUDE.md rule 1): the free atoms are this solver's own
//! isolated-atom references (`crate::references`), the counted charges are the provider's.
//!
//! # Equations (atomic units: Bohr, e; e/Bohr³)
//!
//! With ρ_A⁰ the shell-binned density of the isolated neutral atom of nucleus A's species (its
//! reference table, solved on the scene's own grid, σ and ionic surface — module doc of
//! `crate::references`; the bin value is the geometric mean of ρ over the shell, the spherical
//! average for a spherical atom: closed shells, and the pseudo-atoms, solved with shared degenerate
//! shells; an open-shell all-electron N/O/F under the policy `None` is solved Aufbau and is
//! aspherical, and its geometric mean sits O(ε²/10) below the arithmetic average of an ε P₂
//! anisotropy — ≲ 1 % in its own p-shell region, where its weight is ≈ 1 anyway; physics review
//! SHOULD-2) placed
//! at R_A, the promolecule ρ⁰(r) = Σ_B ρ_B⁰(|r − R_B|) shares every point between the atoms by
//! the stockholder weights (Hirshfeld, Theor. Chim. Acta 44, 129 (1977))
//!
//! ```text
//! w_A(r) = ρ_A⁰(|r − R_A|) / ρ⁰(r),        Σ_A w_A(r) = 1,
//! q_A    = Z_A − ∫ w_A(r) ρ(r) d³r ≈ Z_A − h³ Σ_nodes w_A(r_n) ρ(r_n),
//! ```
//!
//! ρ the converged ρ↑ + ρ↓ at the cell's n³ nodes (the solver's own quadrature: Σ ρ h³ is its
//! electron count; a node is read through `sample_density` at its coordinate, which returns the
//! stored value to rounding — at h = 0.15 a quarter of the nodes resolve to the lower cell with
//! t = 1 − 7e-15, within one f32 ulp on the GPU: ≤ 1e-7 e on N, physics review NIT-1), Z_A the charge A's electrons are counted against — Z for an all-electron
//! nucleus, the valence charge Z_v for a pseudised one (`ForceProvider::valence_charges`; the
//! density of a pseudised solve is the valence density, and its reference atom is the pseudo-atom
//! of Z_v electrons). Because Σ_A w_A = 1 at every node,
//!
//! ```text
//! Σ_A q_A = Σ_A Z_A − h³ Σ_nodes ρ = Σ_A Z_A − N = the system charge,
//! ```
//!
//! exactly up to rounding — and **checked** ([`PartialCharges::sum_error`], [`SUM_TOL`]): a density
//! that does not integrate to the solve's electron count (the wrong ρ, a display field with the
//! partial core added) shows as a sum that misses the scene's charge, and such charges are not
//! shown. It cannot see a density of the right electron count at another geometry: that is
//! [`PartialCharges::is_current`]'s (the viewer fades those). A node where every free atom's tail has underflowed to 0 (ρ⁰ = 0) is given to the
//! nearest nucleus, so no electron is dropped.
//!
//! **What it is not.** One partition scheme of several (Mulliken, Bader/QTAIM, CHELPG, …); the
//! numbers depend on the scheme, and Hirshfeld's are known to be small in magnitude (the free-atom
//! weights pull the partition toward neutral atoms; Davidson & Chakravorty, Theor. Chim. Acta 83,
//! 319 (1992); the iterative Hirshfeld-I of Bultinck et al., J. Chem. Phys. 126, 144111 (2007),
//! is the usual correction). They are measured from this solver's LSDA density at its grid and
//! smearing, so they share its approximations (rule 7: the viewer says so). Further, from the
//! physics review:
//! - **Neutral free atoms, also for ions** (SHOULD-3): the references are the neutral atoms
//!   (ruling S9), so in H₃⁺ or any charged scene the net charge is shared by neutral-atom weights —
//!   the reference-state problem Bultinck et al. raise against Hirshfeld. Σq is still the charge;
//!   its distribution follows the neutral weights.
//! - **Valence charges under pseudopotentials** (SHOULD-4a): the core electrons belong to their
//!   nucleus by construction (frozen core) and the weights are pseudo-valence densities — nodeless
//!   and smaller than the all-electron valence inside r_c — so the pseudo and all-electron charges
//!   of the same molecule are not expected to agree (Σq holds for both).
//! - **At this rung** (SHOULD-4c): molecule and free atoms carry the same σ and h, so the partition
//!   is self-consistent on the smeared surface; no σ → 0 extrapolation is applied.
//! - **Estimated mode** (SHOULD-4b): between teacher steps the charges come from the side solver's
//!   density (an audit or a display refresh) at that solve's own geometry and time, never from the
//!   model's step; their `step` is the driver's step when they arrived.
//!
//! **The free-atom lookup.** ρ_A⁰ at a node is read from a uniform table of the reference's
//! `RadialDensity::at` on r = 0, dr, 2dr, … (dr = h/16) out to the cell's diagonal, linearly
//! interpolated in ρ — the weights only, never the sum rule (Σ w = 1 holds for any weights). Its
//! relative error is ≤ dr²|ρ''|/(8ρ): (h/16)²κ²/2 for an e^{−2κr} tail, 1.2e-4 for H at h = 0.25;
//! ≲ 2e-3 in a smeared core (κ ≲ 1/σ), where w_A ≈ 1 (physics review NIT-2). One table per
//! distinct reference (pointer identity: `Promolecule::from_refs` borrows one cache entry per
//! species; a copy per nucleus would only cost time).
use crate::provider::SideKind;
use crate::references::RadialDensity;
use crate::scene::{neumaier, Cell, Scene};

/// |Σ q − system charge| above which the charges are not shown (e). Measured |Σq − charge| on the
/// CPU solver (`validate --test charges`, `references.md` "Partial charges"): see there; the
/// GPU's f32 storage of ρ bounds its sum's rounding at ~1e-7 relative per value, ≪ 1e-3 e for the
/// presets' ≤ 18 electrons. A partial core added to the display density — the smallest
/// wrong-density case the check is for — is ≥ 2 e (an O 1s² core).
pub const SUM_TOL: f64 = 1e-3;

/// Steps of the free-atom lookup table per grid spacing (module doc, "The free-atom lookup").
const LOOKUP_PER_H: f64 = 16.0;

/// Where a set of charges was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChargeSource {
    /// The stepping provider's (or the estimated solver's teacher's) density after an evaluation.
    Step,
    /// A density solved off the stepping path (estimated mode: an audit or a display refresh).
    Side(SideKind),
}

/// One partition's result (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct PartialCharges {
    /// q_A per nucleus, in scene order (e; positive = fewer electrons than the free atom).
    pub q: Vec<f64>,
    /// The charge each nucleus's electrons were counted against (Z, or Z_v when pseudised).
    pub counted: Vec<f64>,
    /// The nuclei's species and positions (Bohr) the density was solved at.
    pub z: Vec<u8>,
    pub positions: Vec<[f64; 3]>,
    /// h³ Σ ρ over the nodes: the electrons the density holds.
    pub electrons: f64,
    /// The scene's net charge when it was solved.
    pub system_charge: i32,
    /// Simulated time (a.u.) and step of the geometry.
    pub t: f64,
    pub step: u64,
    pub source: ChargeSource,
    /// The ionic surface's tag (`ForceProvider::ionic_tag`): `"none"` = all-electron.
    pub ionic: String,
    /// The electric dipole μ = Σ_A Z_A R_A − ∫ ρ r d³r of the same density ([`dipole`]; e·Bohr,
    /// about the cell centre — origin-dependent for a charged scene). Display only (the reference
    /// comparison, `app::reference`).
    pub dipole: [f64; 3],
}

impl PartialCharges {
    /// Σ_A q_A (compensated).
    pub fn sum(&self) -> f64 {
        neumaier(self.q.iter().copied())
    }

    /// Σ q − the system charge: rounding only for a density of the solve's electron count.
    pub fn sum_error(&self) -> f64 {
        self.sum() - self.system_charge as f64
    }

    /// The sum rule holds to [`SUM_TOL`] (module doc): the charges may be shown.
    pub fn balanced(&self) -> bool {
        self.sum_error().abs() <= SUM_TOL
    }

    pub fn t_fs(&self) -> f64 {
        units::au_time_to_fs(self.t)
    }

    /// Measured at `scene`'s current geometry: the same species in the same order at the same
    /// positions (bit for bit — a paused scene does not move, and a step publishes the geometry it
    /// evaluated).
    pub fn is_current(&self, scene: &Scene) -> bool {
        self.fits(scene) && scene.nuclei.iter().zip(&self.positions).all(|(n, p)| n.pos == *p)
    }

    /// The same nuclei as `scene` (species, in order): the charges can be drawn on its atoms.
    pub fn fits(&self, scene: &Scene) -> bool {
        self.z.len() == scene.nuclei.len() && scene.nuclei.iter().zip(&self.z).all(|(n, &z)| n.z == z)
    }
}

/// A free atom's density on a uniform r table (module doc, "The free-atom lookup").
struct Lookup {
    rho: Vec<f64>,
    inv_dr: f64,
}

impl Lookup {
    fn new(t: &RadialDensity, dr: f64, r_max: f64) -> Lookup {
        let n = (r_max / dr).ceil() as usize + 2;
        Lookup { rho: (0..n).map(|k| t.at(k as f64 * dr)).collect(), inv_dr: 1.0 / dr }
    }

    #[inline]
    fn at(&self, r: f64) -> f64 {
        let x = r * self.inv_dr;
        let i = x as usize;
        if i + 1 >= self.rho.len() {
            return *self.rho.last().unwrap_or(&0.0);
        }
        let f = x - i as f64;
        self.rho[i] + (self.rho[i + 1] - self.rho[i]) * f
    }
}

/// A compensated running sum (Neumaier), for the n³-term node sums (rule 4).
#[derive(Clone, Copy, Default)]
struct Acc {
    sum: f64,
    comp: f64,
}

impl Acc {
    #[inline]
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() { self.comp += (self.sum - t) + x } else { self.comp += (x - t) + self.sum }
        self.sum = t;
    }

    fn value(&self) -> f64 {
        self.sum + self.comp
    }
}

/// The Hirshfeld charges (module doc) of the density `rho` — ρ↑ + ρ↓ at `cell`'s n³ nodes in
/// `qm::Grid` order (x fastest; node (i, j, k) at −half + (i, j, k)·h) — with free atom
/// `free[a]` at `pos[a]` and counted charge `counted[a]` per nucleus. Returns (q per nucleus,
/// h³ Σ ρ).
pub fn hirshfeld(free: &[&RadialDensity], pos: &[[f64; 3]], counted: &[f64], cell: &Cell, rho: &[f64]) -> (Vec<f64>, f64) {
    let atoms = pos.len();
    assert_eq!(free.len(), atoms, "hirshfeld: {} free atoms for {atoms} nuclei", free.len());
    assert_eq!(counted.len(), atoms, "hirshfeld: {} counted charges for {atoms} nuclei", counted.len());
    let n = cell.n;
    assert_eq!(rho.len(), n * n * n, "hirshfeld: {} values for {n}³ nodes", rho.len());
    if atoms == 0 {
        return (Vec::new(), cell.h.powi(3) * neumaier(rho.iter().copied()));
    }
    let (h, half) = (cell.h, cell.half_extent());
    // one lookup per distinct table (species), reaching every node from every nucleus inside the cell
    let dr = h / LOOKUP_PER_H;
    let r_max = 2.0 * 3f64.sqrt() * half + h;
    let mut tables: Vec<(*const RadialDensity, Lookup)> = Vec::new();
    let mut which = Vec::with_capacity(atoms);
    for t in free {
        let key = *t as *const RadialDensity;
        let k = match tables.iter().position(|(p, _)| *p == key) {
            Some(k) => k,
            None => {
                tables.push((key, Lookup::new(t, dr, r_max)));
                tables.len() - 1
            }
        };
        which.push(k);
    }
    let mut own = vec![Acc::default(); atoms];
    let mut total = Acc::default();
    let mut w = vec![0.0f64; atoms];
    let coord = |i: usize| -half + i as f64 * h;
    for kz in 0..n {
        let z = coord(kz);
        for jy in 0..n {
            let y = coord(jy);
            let row = n * (jy + n * kz);
            for ix in 0..n {
                let v = rho[row + ix];
                if v == 0.0 {
                    continue;
                }
                total.add(v);
                let x = coord(ix);
                let mut sum = 0.0;
                let (mut nearest, mut best) = (0usize, f64::INFINITY);
                for a in 0..atoms {
                    let d = [x - pos[a][0], y - pos[a][1], z - pos[a][2]];
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 < best {
                        (nearest, best) = (a, r2);
                    }
                    w[a] = tables[which[a]].1.at(r2.sqrt());
                    sum += w[a];
                }
                if sum > 0.0 {
                    let s = v / sum;
                    for a in 0..atoms {
                        own[a].add(w[a] * s);
                    }
                } else {
                    own[nearest].add(v);
                }
            }
        }
    }
    let h3 = h * h * h;
    let q = (0..atoms).map(|a| counted[a] - h3 * own[a].value()).collect();
    (q, h3 * total.value())
}

/// The electric dipole (e·Bohr, about the cell centre — the coordinates' origin) of point charges
/// `counted[a]` at `pos[a]` and the electron density `rho` at `cell`'s n³ nodes in `qm::Grid`
/// order (as [`hirshfeld`]): μ = Σ_A Z_A R_A − h³ Σ_n ρ(r_n) r_n, the convention of μ pointing from
/// the negative to the positive charge. Under pseudopotentials `counted` is Z_v and `rho` the
/// valence density: a frozen core is a neutral (Z − Z_v nucleus + core electrons) charge at R_A,
/// so μ is the all-electron dipole up to the core's polarisation. The box truncates the density's
/// tail (the cell is finite) — the same truncation the solve has. Display only (rule 4: compensated
/// sums over the n³ nodes).
pub fn dipole(pos: &[[f64; 3]], counted: &[f64], cell: &Cell, rho: &[f64]) -> [f64; 3] {
    let n = cell.n;
    assert_eq!(rho.len(), n * n * n, "dipole: ρ has {} values for {n}³ nodes", rho.len());
    let (h, half) = (cell.h, cell.half_extent());
    let coord = |i: usize| -half + i as f64 * h;
    let mut el = [Acc::default(); 3];
    for kz in 0..n {
        let z = coord(kz);
        for jy in 0..n {
            let y = coord(jy);
            let row = n * (jy + n * kz);
            for ix in 0..n {
                let v = rho[row + ix];
                if v != 0.0 {
                    el[0].add(v * coord(ix));
                    el[1].add(v * y);
                    el[2].add(v * z);
                }
            }
        }
    }
    let h3 = h * h * h;
    std::array::from_fn(|c| neumaier(pos.iter().zip(counted).map(|(p, q)| q * p[c])) - h3 * el[c].value())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Nucleus;

    /// Two unit charges with their electrons displaced: μ = Σ Z R − Σ N_e r_e. A neutral atom whose
    /// electron sits on it has μ = 0; an electron moved by d gives μ = −d per electron.
    #[test]
    fn the_dipole_of_displaced_gaussians() {
        let cell = Cell { n: 49, h: 0.25 };
        let g = gauss(0.6);
        let pos = [[-1.0, 0.0, 0.5], [1.2, 0.3, 0.0]];
        let rho = on_nodes(&cell, &[(1.0, pos[0], &g), (1.0, pos[1], &g)]);
        let mu = dipole(&pos, &[1.0, 1.0], &cell, &rho);
        assert!(mu.iter().all(|m| m.abs() < 1e-8), "{mu:?}");
        // atom 1's electron moved to atom 0: μ = R₁ − R₀ (one electron's charge from R₁ to R₀)
        let rho = on_nodes(&cell, &[(2.0, pos[0], &g)]);
        let mu = dipole(&pos, &[1.0, 1.0], &cell, &rho);
        for c in 0..3 {
            assert!((mu[c] - (pos[1][c] - pos[0][c])).abs() < 1e-8, "{mu:?}");
        }
    }

    /// The density Σ_A c_A g(|r − R_A|) at the nodes of `cell`.
    fn on_nodes(cell: &Cell, terms: &[(f64, [f64; 3], &dyn Fn(f64) -> f64)]) -> Vec<f64> {
        let n = cell.n;
        let half = cell.half_extent();
        let mut v = Vec::with_capacity(n * n * n);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = [-half + i as f64 * cell.h, -half + j as f64 * cell.h, -half + k as f64 * cell.h];
                    v.push(terms.iter().map(|(c, r, f)| c * f(((p[0] - r[0]).powi(2) + (p[1] - r[1]).powi(2) + (p[2] - r[2]).powi(2)).sqrt())).sum());
                }
            }
        }
        v
    }

    /// A normalised (one-electron) Gaussian of width w.
    fn gauss(w: f64) -> impl Fn(f64) -> f64 {
        let norm = (2.0 * std::f64::consts::PI * w * w).powf(-1.5);
        move |r: f64| norm * (-r * r / (2.0 * w * w)).exp()
    }

    /// A symmetric dimer of identical atoms whose density is exactly the promolecule: both charges
    /// 0 (to the grid's quadrature of the free atom, which is the same for both) and the sum 0.
    #[test]
    fn a_symmetric_dimer_is_neutral_and_the_sum_is_the_system_charge() {
        let cell = Cell { n: 49, h: 0.25 };
        let g = gauss(0.7);
        let atom = RadialDensity::from_fn(&g, 6.0, 0.125);
        let pos = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
        // the molecule: 0.15 e of each atom moved into a bond bump at the midpoint (still symmetric)
        let bump = gauss(0.4);
        let rho = on_nodes(&cell, &[(0.85, pos[0], &g), (0.85, pos[1], &g), (0.3, [0.0; 3], &bump)]);
        let (q, electrons) = hirshfeld(&[&atom, &atom], &pos, &[1.0, 1.0], &cell, &rho);
        println!("symmetric dimer: q = {q:?}, electrons {electrons:.8}");
        assert!((q[0] - q[1]).abs() < 1e-12, "{q:?}");
        assert!(q[0].abs() < 1e-6, "{q:?}");
        assert!((electrons - 2.0).abs() < 1e-6, "{electrons}");
        assert!((q[0] + q[1] - (2.0 - electrons)).abs() < 1e-12);
    }

    /// Density moved from atom 1 to atom 0 shows as q₀ < 0 < q₁ (neutral); the sum stays ΣZ − N for a
    /// neutral and a charged scene alike, and a node beyond every free atom's reach is not lost.
    #[test]
    fn transferred_density_has_the_expected_signs_and_the_sum_rule_holds() {
        let cell = Cell { n: 49, h: 0.25 };
        let (ga, gb) = (gauss(0.6), gauss(0.9));
        let (a, b) = (RadialDensity::from_fn(&ga, 6.0, 0.125), RadialDensity::from_fn(&gb, 6.0, 0.125));
        let pos = [[-1.2, 0.3, 0.0], [1.5, -0.2, 0.4]];
        // "HF"-like: atom 0 holds 1.3 of its free density, atom 1 0.7 — 0.3 e moved from 1 to 0
        for (charge, extra) in [(0i32, 0.0), (1, -1.0), (-1, 1.0)] {
            let rho = on_nodes(&cell, &[(1.3 + extra, pos[0], &ga), (0.7, pos[1], &gb)]);
            let counted = [1.0, 1.0];
            let (q, electrons) = hirshfeld(&[&a, &b], &pos, &counted, &cell, &rho);
            let sum: f64 = q.iter().sum();
            println!("charge {charge}: q = {q:?}, Σq = {sum:.8}, N = {electrons:.8}");
            if charge == 0 {
                assert!(q[0] < -0.1 && q[1] > 0.1, "{q:?}");
            }
            // Σq = ΣZ − N exactly; N = 2 − charge to the quadrature of the Gaussians on the grid
            assert!((sum - (2.0 - electrons)).abs() < 1e-12, "{sum} vs {}", 2.0 - electrons);
            assert!((sum - charge as f64).abs() < 1e-6, "{sum} vs {charge}");
            let pc = PartialCharges {
                q: q.clone(),
                counted: counted.to_vec(),
                z: vec![1, 1],
                positions: pos.to_vec(),
                electrons,
                system_charge: charge,
                t: 0.0,
                step: 0,
                source: ChargeSource::Step,
                ionic: "none".into(),
                dipole: [0.0; 3],
            };
            assert!(pc.balanced(), "{}", pc.sum_error());
        }
        // a free atom with no tail (zero beyond 1 Bohr) and density beyond it: given to the nearest
        let short = RadialDensity { r: vec![0.0, 0.5, 1.0], log_rho: vec![0.0, -1.0, 1.0] }; // the last segment rises: cut to 0 beyond
        let rho = on_nodes(&cell, &[(1.0, pos[0], &ga), (1.0, pos[1], &gb)]);
        let (q, electrons) = hirshfeld(&[&short, &short], &pos, &[1.0, 1.0], &cell, &rho);
        assert!((q.iter().sum::<f64>() - (2.0 - electrons)).abs() < 1e-12, "{q:?}");
    }

    /// The partition's cost at the default Accurate grid (81³) for water and for 21 atoms (aspirin's
    /// count) at 95³: reported, never asserted (`--ignored --nocapture`).
    #[test]
    #[ignore = "a measurement"]
    fn partition_cost_is_reported() {
        let g = gauss(0.6);
        let atom = RadialDensity::from_fn(&g, 6.0, 0.075);
        for (n, atoms) in [(81usize, 3usize), (95, 21)] {
            let cell = Cell { n, h: 0.15 };
            let pos: Vec<[f64; 3]> = (0..atoms).map(|k| { let a = k as f64 * 2.4; [2.5 * a.cos(), 2.5 * a.sin(), 0.3 * k as f64 - 3.0] }).collect();
            let rho = on_nodes(&cell, &pos.iter().map(|&p| (1.0, p, &g as &dyn Fn(f64) -> f64)).collect::<Vec<_>>());
            let free = vec![&atom; atoms];
            let t = std::time::Instant::now();
            let (q, _) = hirshfeld(&free, &pos, &vec![1.0; atoms], &cell, &rho);
            println!("{n}³, {atoms} atoms: {:.1} ms (Σq = {:+.2e})", t.elapsed().as_secs_f64() * 1e3, q.iter().sum::<f64>());
        }
    }

    /// A density that does not integrate to the electrons the counted charges imply (a partial core
    /// added) misses the system charge and is not balanced.
    #[test]
    fn a_density_with_extra_electrons_is_not_balanced() {
        let cell = Cell { n: 33, h: 0.25 };
        let g = gauss(0.6);
        let atom = RadialDensity::from_fn(&g, 4.0, 0.125);
        let pos = [[0.0; 3]];
        let rho = on_nodes(&cell, &[(3.0, pos[0], &g)]); // 3 electrons against Z_v = 1
        let (q, electrons) = hirshfeld(&[&atom], &pos, &[1.0], &cell, &rho);
        let pc = PartialCharges { q, counted: vec![1.0], z: vec![1], positions: pos.to_vec(), electrons, system_charge: 0, t: 0.0, step: 0, source: ChargeSource::Step, ionic: "none".into(), dipole: [0.0; 3] };
        assert!(!pc.balanced() && (pc.sum_error() + 2.0).abs() < 1e-3, "{}", pc.sum_error());
        let scene = Scene { nuclei: vec![Nucleus::new(1, [0.0; 3])], cell, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: crate::scene::IonicPolicy::None };
        assert!(pc.is_current(&scene));
        let mut moved = scene.clone();
        moved.nuclei[0].pos[0] = 0.1;
        assert!(!pc.is_current(&moved) && pc.fits(&moved));
    }
}
