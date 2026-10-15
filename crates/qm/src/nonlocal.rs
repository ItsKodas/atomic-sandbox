//! The Kleinman–Bylander nonlocal operator on the real-space grid (M7; spec §4.6, §5.2, §5.3). CPU, f64,
//! atomic units (Hartree, Bohr).
//!
//! # The operator (spec §4.6, §5.3; review S1)
//! For every pseudised ion I at R_I and every projector l of its table (one per channel other than the local
//! one, l ≤ 2 for Z ≤ 18),
//!   V_NL = Σ_I Σ_l Σ_{m=−l..l} E_l^KB |β_lm^I⟩⟨β_lm^I|,
//!   β_lm^I(r) = t_l(d)·N_l·R_lm(d),   d = r − R_I, d = |d|,   N_l = √((2l+1)/4π),
//! where R_lm(d) = d^l S_lm(d̂) is `qm::poisson`'s real solid harmonic with Racah normalisation
//! (∫|S_lm|²dΩ = 4π/(2l+1)), so N_l S_lm = Y_lm is the orthonormal real spherical harmonic, and
//! t_l = F_l/d^l with F_l = β_u/d = δV_l u_l/d the filtered radial projector (filter.rs "The u-form ↔ 3-D
//! convention"; finite at 0 since u_l ∝ d^{l+1}). Hence β_lm = F_l(d) Y_lm(d̂) and, for ψ = (u/d)Y_l'm',
//!   ⟨β_lm|ψ⟩ = δ_ll' δ_mm' ∫₀^∞ F_l (u/d) d² dd = δ_ll' δ_mm' ∫ β_u u dd = δ_ll' δ_mm' / E_l^KB
//! on the reference pseudo-orbital (E_KB = 1/⟨φ_l|δV_l|φ_l⟩ radially), so V_NL φ_l0 = β_l0 = δV_l φ_l0: the
//! semilocal operator on the reference state, exactly (the S1 identity test below). Using R_lm without N_l
//! would multiply V_NL by 4π/(2l+1) ≥ 4.19 (l ≤ 2) — the factor that test exists to catch.
//!
//! # On the grid
//! The projections are the grid quadrature c_lm[ψ] = h³ Σ_{p ∈ sphere} β_lm(r_p) ψ(r_p), and
//!   (V_NL ψ)_p = Σ_{I,l,m} E_l^KB c_lm[ψ] β_lm(r_p),   E_NL = Σ_{iσ} Σ_{I,lm} E_l^KB c_lm[ψ_iσ]²
//! (integer occupations: f = 1 per spin orbital). The sphere of an ion's projector l is the set of grid points
//! within the filtered table's radius γ_r r_β + 8h (`FilteredProjector::radius`), r_β = max(r_c, r_nlc) the
//! derived support (spec §4.6, Task 3 review; never a 1e-12-relative rule): beyond it t_l is exactly 0 (the
//! band completion's outer step), so the list loses nothing. t_l is read only from the uniform Δ = h/256 table
//! (`FilteredTable::beta_radial`, spec §5.1 "Sampling"), never from the log grid. The sphere's points and the
//! (2l+1) β_lm values on them are built once per geometry ([`NonlocalOp::build`]).
//!
//! # Symmetry, and spheres clipped by the box (Review Focus 3)
//! In the grid inner product ⟨a, b⟩ = h³Σ a_p b_p (`Grid::dot`), ⟨a, V_NL b⟩ = Σ E_KB c_lm[a] c_lm[b] =
//! ⟨V_NL a, b⟩: symmetric whatever the point list, because the same list and the same β values enter both the
//! projection and the scatter. A sphere cut by a box face simply has fewer points: points outside the box do not
//! exist (ψ = 0 there under the Dirichlet wall), so the clipped operator is the exact restriction of the
//! unclipped one to the box, and it stays symmetric. LOBPCG (`eigen`) requires exactly this symmetry.
//!
//! # Precision (rule 4)
//! c_lm is a Neumaier-compensated f64 sum over the sphere times h³; E_NL a Neumaier sum over every (orbital,
//! sphere, m) term. The scatter is pointwise.
//!
//! # Cost (spec §5.3)
//! Per orbital per sphere, 2(2l+1)·n_pts multiply-adds (projection and scatter), n_pts ≈ (4/3)π(radius/h)³ —
//! measured against the local Hamiltonian in Task 6's report.
use crate::eigen::Operator;
use crate::grid::{neumaier, Grid};
use crate::ionic::{Ionic, IonicSetup};
use crate::nuclear::{points_within, Nucleus};
use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
use crate::stencil::kinetic;
use std::f64::consts::PI;

/// One projector sphere: ion index, Z, l, the grid points within the table radius, β_lm on them ([m + l][point])
/// and E_KB (Ha).
#[derive(Clone, Debug)]
pub struct Sphere { pub ion: usize, pub z: u8, pub l: u8, pub points: Vec<usize>, pub beta: Vec<Vec<f64>>, pub e_kb: f64 }

/// V_NL on a grid for one geometry (module doc); `dv` = h³. Empty when no pseudised species has a projector.
#[derive(Clone, Debug, Default)]
pub struct NonlocalOp { pub spheres: Vec<Sphere>, pub dv: f64 }

impl NonlocalOp {
    /// The spheres of every pseudised ion's projectors at these nuclear positions (module doc).
    pub fn build(grid: &Grid, nuclei: &[Nucleus], setup: &IonicSetup) -> NonlocalOp {
        let mut spheres = Vec::new();
        for (ion, n) in nuclei.iter().enumerate() {
            let Ionic::Pseudo(t) = setup.ionic(n.z) else { continue };
            for (k, p) in t.projectors.iter().enumerate() {
                assert!(p.l <= 2, "projectors up to l = 2 (Z ≤ 18; Task 7's gradients are explicit to l = 2)");
                let points = points_within(grid, n.pos, p.radius);
                let norm = ((2 * p.l + 1) as f64 / (4.0 * PI)).sqrt();
                let li = p.l as i32;
                let mut beta = vec![Vec::with_capacity(points.len()); 2 * p.l as usize + 1];
                for &q in &points {
                    let r = grid.point(q);
                    let d = [r[0] - n.pos[0], r[1] - n.pos[1], r[2] - n.pos[2]];
                    let mut sh = [0.0; N_MOMENTS];
                    solid_harmonics(d, &mut sh);
                    let tn = t.beta_radial(k, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()) * norm;
                    for m in -li..=li { beta[(m + li) as usize].push(tn * sh[lm_index(p.l as usize, m)]); }
                }
                spheres.push(Sphere { ion, z: n.z, l: p.l, points, beta, e_kb: p.e_kb });
            }
        }
        NonlocalOp { spheres, dv: grid.dv() }
    }

    pub fn is_empty(&self) -> bool { self.spheres.is_empty() }

    /// c[sphere][m + l] = h³ Σ_p β_lm(r_p) ψ(r_p) (Neumaier).
    pub fn project(&self, psi: &[f64]) -> Vec<Vec<f64>> {
        self.spheres.iter().map(|s| s.beta.iter().map(|b| neumaier(s.points.iter().zip(b).map(|(&p, &bv)| bv * psi[p])) * self.dv).collect()).collect()
    }

    /// out += V_NL ψ = Σ E_KB c_lm β_lm.
    pub fn apply_add(&self, psi: &[f64], out: &mut [f64]) {
        for (s, c) in self.spheres.iter().zip(self.project(psi)) {
            for (b, cm) in s.beta.iter().zip(c) {
                let f = s.e_kb * cm;
                for (&p, &bv) in s.points.iter().zip(b) { out[p] += f * bv; }
            }
        }
    }

    /// E_NL = Σ_i Σ_{sphere, m} E_KB c_lm[ψ_i]² over the given (singly occupied) orbitals (Neumaier).
    pub fn energy(&self, orbitals: &[Vec<f64>]) -> f64 {
        let mut terms = Vec::new();
        for psi in orbitals {
            for (s, c) in self.spheres.iter().zip(self.project(psi)) {
                terms.extend(c.into_iter().map(|cm| s.e_kb * cm * cm));
            }
        }
        neumaier(terms.into_iter())
    }

    /// E_NL = Σ_i f_i Σ_{sphere, m} E_KB c_lm[ψ_i]² with occupations f (spec §5.7; Neumaier).
    pub fn energy_weighted(&self, orbitals: &[Vec<f64>], f: &[f64]) -> f64 {
        assert_eq!(orbitals.len(), f.len(), "one occupation per orbital");
        let mut terms = Vec::new();
        for (psi, &fi) in orbitals.iter().zip(f) {
            if fi == 0.0 { continue; }
            for (s, c) in self.spheres.iter().zip(self.project(psi)) {
                terms.extend(c.into_iter().map(|cm| fi * s.e_kb * cm * cm));
            }
        }
        neumaier(terms.into_iter())
    }
}

/// H = −½∇² + V + V_NL (V the local effective potential of one spin channel).
pub struct PseudoHamiltonian<'a> { pub grid: &'a Grid, pub v: &'a [f64], pub nl: &'a NonlocalOp }

impl Operator for PseudoHamiltonian<'_> {
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        kinetic(self.grid, x, y);
        for i in 0..y.len() { y[i] += self.v[i] * x[i]; }
        self.nl.apply_add(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::ionic::{Ionic, IonicPolicy, IonicSetup};
    use crate::nuclear::Nucleus;
    use crate::poisson::{lm_index, solid_harmonics, N_MOMENTS};
    use crate::radial::filter::FilteredTable;
    use crate::radial::test_cache::{filtered_at, table};
    use std::collections::BTreeMap;
    use std::f64::consts::PI;
    use std::sync::Arc;

    fn o_setup(ft: Arc<FilteredTable>) -> IonicSetup {
        let h = ft.h;
        IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, Ionic::Pseudo(ft))]))
    }

    /// Spec §8 row 6 first failing test (review S1): on the grid, V_NL applied to the sampled reference
    /// pseudo-orbital φ_l0 = (u_l/d)·Y_l0 equals δV_l φ_l0 pointwise to 1e-6 of max|δV φ|, and ⟨φ|V_NL|φ⟩
    /// equals ⟨φ|δV|φ⟩ — which catches a missing √((2l+1)/4π) (a factor 4π/(2l+1) ≥ 4.19 in V_NL).
    /// Separately, the grid quadrature of the projection: |E_KB⟨β|φ⟩_grid − 1| < 1e-4. Plan review S8: an
    /// absolute pointwise 1e-6 mixed that quadrature floor into an operator-normalisation test; the relative
    /// bound keeps the discrimination. Unfiltered table; O off a grid point.
    ///
    /// Deviation (Task 6, for physics review): the bounds are run at h = 0.05, not the plan's 0.1. With
    /// ε = E_KB⟨β|φ⟩_grid − 1, V_NLφ − δVφ = ε·δVφ + (m ≠ 0 terms) pointwise and ⟨φ|V_NL|φ⟩ = (1 + ε)⟨φ|δV|φ⟩,
    /// so all three assertions measure the one quadrature ε. S8 estimated it at ~1e-7 at h = 0.1. The TM junction
    /// at r_c is C² (V‴ jumps), so the transform of δV·φ² falls as q⁻⁵ and ε ∝ h⁵, with an offset-dependent sign
    /// (aliasing). The lattice sum alone, offsets (0.013, −0.021, 0.017) / (h/2)³, measured:
    /// - h = 0.2: +8.7e-5 / −1.2e-4;
    /// - h = 0.1: −2.7e-6 / +3.8e-6;
    /// - h = 0.05: −4.9e-8 / −1.2e-7 (32× per halving).
    /// A correct operator cannot meet 1e-6 at h = 0.1, while a normalisation error is h-independent. So h = 0.1
    /// keeps S8's separate 1e-4 and must show ε falling with h; h = 0.05 carries the plan's three bounds unchanged.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn v_nl_on_grid_is_the_operator_identity() {
        let t = table(8);
        let nuc = [Nucleus { pos: [0.013, -0.021, 0.017], z: 8 }];
        let mut eps_by_h = Vec::new();
        for (h, n) in [(0.1, 61), (0.05, 121)] {             // the same 6 Bohr box
            let setup = o_setup(Arc::new(FilteredTable::unfiltered(t, h)));
            let grid = Grid::cube(n, h);
            let nl = NonlocalOp::build(&grid, &nuc, &setup);
            assert!(!nl.is_empty(), "O has a projector");
            for p in &t.kb.projectors {
                let y = ((2 * p.l + 1) as f64 / (4.0 * PI)).sqrt();
                let at = |r: [f64; 3]| { let d = [r[0] - nuc[0].pos[0], r[1] - nuc[0].pos[1], r[2] - nuc[0].pos[2]]; let dd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12); (d, dd) };
                let phi = grid.sample(|r| { let (d, dd) = at(r); let mut sh = [0.0; N_MOMENTS]; solid_harmonics(d, &mut sh);
                                             t.grid.interp(&p.u, dd) / dd.powi(p.l as i32 + 1) * y * sh[lm_index(p.l as usize, 0)] });
                let expect: Vec<f64> = (0..grid.len()).map(|i| { let (_, dd) = at(grid.point(i)); t.grid.interp(&p.dv, dd) * phi[i] }).collect();
                let mut got = vec![0.0; grid.len()];
                nl.apply_add(&phi, &mut got);
                let worst = got.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                let scale = expect.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let (lhs, rhs) = (grid.dot(&phi, &got), grid.dot(&phi, &expect));
                // the projection's own quadrature: E_KB·⟨β_l0|φ_l0⟩_grid = 1 (radially ⟨β|φ⟩ = 1/E_KB exactly)
                let k = nl.spheres.iter().position(|s| s.l == p.l).expect("a sphere for this projector");
                let ekb_c = nl.spheres[k].e_kb * nl.project(&phi)[k][p.l as usize];   // m = 0 sits at index l
                eprintln!("h = {h}, l = {}: max |V_NL φ − δV φ| = {worst:.3e} of max |δV φ| {scale:.3e} ({:.2e} relative); ⟨φ|V_NL|φ⟩ {lhs:.10} vs ⟨φ|δV|φ⟩ {rhs:.10}; ε = E_KB⟨β|φ⟩_grid − 1 = {:+.3e}",
                          p.l, worst / scale, ekb_c - 1.0);
                assert!((ekb_c - 1.0).abs() < 1e-4, "h = {h}, l = {}: E_KB⟨β|φ⟩_grid = {ekb_c:.8}", p.l);
                eps_by_h.push((ekb_c - 1.0).abs());
                if h == 0.05 {
                    assert!(worst <= 1e-6 * scale, "l = {}: max |V_NL φ − δV φ| = {worst:.3e} of max |δV φ| {scale:.3e}", p.l);
                    assert!((lhs - rhs).abs() < 1e-6 * rhs.abs(), "l = {}: ⟨φ|V_NL|φ⟩ {lhs:.9} vs ⟨φ|δV|φ⟩ {rhs:.9}", p.l);
                }
            }
        }
        // a normalisation error is h-independent; the quadrature floor falls (∝ h⁵; measured 34×, ε −2.69e-6 → −8.0e-8)
        let np = t.kb.projectors.len();
        for k in 0..np { assert!(eps_by_h[np + k] < 0.25 * eps_by_h[k], "ε does not fall with h: {eps_by_h:?}"); }
    }

    /// Review Focus 3: a sphere clipped by the box face keeps V_NL symmetric (the clipped points carry
    /// ψ = 0 under the Dirichlet wall, so the clipped operator is the exact restriction).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn projector_sphere_clipped_by_the_box_stays_symmetric() {
        let t = table(8);
        let h = t.record.h_req.unwrap();
        let setup = o_setup(filtered_at(8, h));
        let grid = Grid::cube(33, h);
        let edge = grid.origin[0] + 0.4;                         // 0.4 Bohr inside the −x face: the sphere is clipped
        let nl = NonlocalOp::build(&grid, &[Nucleus { pos: [edge, 0.1, -0.2], z: 8 }], &setup);
        let whole = NonlocalOp::build(&grid, &[Nucleus { pos: [0.0, 0.1, -0.2], z: 8 }], &setup);
        assert!(4 * nl.spheres[0].points.len() < 3 * whole.spheres[0].points.len(), "the sphere is clipped: {} of {} points",
                nl.spheres[0].points.len(), whole.spheres[0].points.len());
        let mut seed = 0x5EEDu64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
        let a: Vec<f64> = (0..grid.len()).map(|_| rnd()).collect();
        let b: Vec<f64> = (0..grid.len()).map(|_| rnd()).collect();
        let (mut na, mut nb) = (vec![0.0; grid.len()], vec![0.0; grid.len()]);
        nl.apply_add(&a, &mut na);
        nl.apply_add(&b, &mut nb);
        let (x, y) = (grid.dot(&a, &nb), grid.dot(&na, &b));
        assert!((x - y).abs() <= 1e-12 * x.abs().max(y.abs()), "⟨a|V_NL b⟩ {x} vs ⟨V_NL a|b⟩ {y}");
    }
}
