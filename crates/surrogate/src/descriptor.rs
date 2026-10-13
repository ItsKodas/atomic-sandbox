//! Invariant descriptors of an atom's neighbourhood: the body-2 (l = 0) atomic basis on species
//! channels, with analytic position gradients. Pure f64, CPU. Body-3 (l ≥ 1 contractions under
//! the degree cap) is Task 6's block; every `match`/branch on [`Basis::body3`] below is where it
//! plugs in without reshaping the body-2 indices.
//!
//! # The atomic basis for l = 0 and the species channel
//!
//! For centre atom i and a neighbour species t (an index into the sorted species list, not a
//! number derived from Z — CLAUDE.md rule 1), the l = 0 one-particle basis summed over the
//! neighbours of that species is
//!
//!   A_{n,t}(i) = Σ_{j ≠ i, s_j = t, r_ij < r_cut} R_n(r_ij) f_c(r_ij),   n = 0..n_max,
//!
//! with R_n f_c from [`crate::basis::radial`] (S_00 = 1, so no angular factor). Because it depends
//! on the distances r_ij = |r_j − r_i| only, every A_{n,t} is invariant under translations and
//! rotations of the whole cluster, and under permutations of atoms it follows its centre (the
//! sum over j is order-free). These A are already invariants, so the body-2 features are
//! B_{i,k} = A_{n,t}(i) with the flat index k = t·n_max + n — neighbour species major, radial
//! index minor, so the n_max coefficients of one (centre, neighbour) species pair are contiguous.
//! Feature count: n_max·S per centre species (ruling S3), the same for every centre species; the
//! centre species selects the coefficient row, not the feature layout.
//!
//! # Gradients
//!
//! With u_ij = (r_j − r_i)/r_ij the unit vector from i to j, ∂r_ij/∂r_j = u_ij and
//! ∂r_ij/∂r_i = −u_ij, so for g_n(r) = d[R_n f_c]/dr (the `d_out` of `radial`)
//!
//!   ∂B_{i,(t,n)}/∂r_j = δ_{s_j t} g_n(r_ij) u_ij   (j ≠ i),
//!   ∂B_{i,(t,n)}/∂r_i = −Σ_{j: s_j = t} g_n(r_ij) u_ij,
//!
//! i.e. the self gradient is minus the sum of the neighbour gradients (translation invariance:
//! Σ_j ∂B_i/∂r_j = 0). Each pair (i, j) inside the cutoff is visited once; it adds R_n f_c to
//! B_i's (s_j, n) features and to B_j's (s_i, n) features, and the gradient ±g_n u_ij to the four
//! affected entries (B_i w.r.t. r_j and r_i, B_j w.r.t. r_i and r_j). At r_cut both R_n f_c and
//! g_n are exactly zero, so a pair leaving the sphere changes nothing discontinuously (E is C¹,
//! F continuous; see `basis`). Coincident atoms (r_ij = 0) have no direction: the value R_n(0) = 1
//! is kept and the gradient contribution is set to zero (the energy has a |r| cusp there, 0 is a
//! valid subgradient and keeps the pair symmetry, hence ΣF = 0).
//!
//! # Neighbour search
//!
//! O(N²) with the r² < r_cut² test, no cell list: the surrogate targets N ≤ 200 atoms, where a
//! 20 000-pair sweep is far below one descriptor evaluation's useful work (a cell list is a later
//! performance item). The square-root and the radial evaluation happen only for pairs inside.

use crate::basis::radial;
use std::fmt;

/// Basis definition: what the descriptor vector of an atom consists of. `species` is the sorted,
/// deduplicated list of Z the model knows; a Z not in it is a [`DescribeError::UnknownSpecies`],
/// never a guess.
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    /// Number of radial functions, n = 0..n_max (half-open).
    pub n_max: usize,
    /// Highest angular momentum of the body-3 block (unused while `body3` is false).
    pub l_max: usize,
    /// Cutoff radius in Bohr.
    pub r_cut: f64,
    /// Total-degree cap n₁ + n₂ + l ≤ degree_cap of the body-3 block (unused while `body3` is false).
    pub degree_cap: usize,
    /// Whether the body-3 block is part of the feature vector (Task 6).
    pub body3: bool,
    /// Sorted, unique nuclear charges.
    pub species: Vec<u8>,
}

impl Basis {
    /// Defaults: n_max 8, l_max 3, r_cut 6.0 Bohr, degree_cap 10, body-2 only. `species` is
    /// sorted and deduplicated.
    pub fn new(species: &[u8]) -> Basis {
        let mut species = species.to_vec();
        species.sort_unstable();
        species.dedup();
        Basis {
            n_max: 8,
            l_max: 3,
            r_cut: 6.0,
            degree_cap: 10,
            body3: false,
            species,
        }
    }

    /// Number of species channels S.
    pub fn n_species(&self) -> usize {
        self.species.len()
    }

    /// Index of Z in the species list, if the basis knows it.
    pub fn species_index(&self, z: u8) -> Option<usize> {
        self.species.binary_search(&z).ok()
    }

    /// Species index of every atom, or the first unknown Z.
    pub fn species_indices(&self, z: &[u8]) -> Result<Vec<usize>, DescribeError> {
        z.iter()
            .map(|&zi| self.species_index(zi).ok_or(DescribeError::UnknownSpecies(zi)))
            .collect()
    }

    /// Size of the body-2 block: n_max·S.
    pub fn n_body2(&self) -> usize {
        self.n_max * self.species.len()
    }

    /// Flat feature index of the body-2 feature (neighbour species t, radial n): t·n_max + n.
    #[inline]
    pub fn body2_index(&self, t: usize, n: usize) -> usize {
        t * self.n_max + n
    }

    /// Length P of the per-atom descriptor vector.
    pub fn n_features(&self) -> usize {
        match self.body3 {
            false => self.n_body2(),
            true => unimplemented!("body-3 descriptors land in Task 6"),
        }
    }

    /// Human-readable name of every feature, in index order (for debugging and the sidecar
    /// header): `b2[Z=<neighbour Z>,n=<n>]`.
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.n_features());
        for &zt in &self.species {
            for n in 0..self.n_max {
                names.push(format!("b2[Z={zt},n={n}]"));
            }
        }
        match self.body3 {
            false => {}
            true => unimplemented!("body-3 descriptors land in Task 6"),
        }
        debug_assert_eq!(names.len(), self.n_features());
        names
    }
}

/// Per-atom descriptor vector B_i (length `n_features`) and its gradient with respect to every
/// atom position that influences it: `grad` lists (j, dB_{i,k}/dr_j for all k), with the self
/// entry (j = i) first and then the neighbours inside r_cut in ascending index order. Atoms not
/// listed have zero gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomDescriptor {
    pub b: Vec<f64>,
    pub grad: Vec<(usize, Vec<[f64; 3]>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescribeError {
    /// A nuclear charge the basis has no species channel for.
    UnknownSpecies(u8),
}

impl fmt::Display for DescribeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DescribeError::UnknownSpecies(z) => {
                write!(f, "species Z = {z} is not in the surrogate basis")
            }
        }
    }
}

impl std::error::Error for DescribeError {}

/// Body-2 descriptors and their gradients for every atom (the fit path; the per-step hot path
/// is [`crate::model::energy_and_forces`], which never materialises these lists).
///
/// # Panics
/// If `pos.len() != z.len()`, or if `basis.body3` is set (Task 6).
pub fn describe(
    basis: &Basis,
    pos: &[[f64; 3]],
    z: &[u8],
) -> Result<Vec<AtomDescriptor>, DescribeError> {
    assert_eq!(pos.len(), z.len(), "describe: pos and z lengths differ");
    let s = basis.species_indices(z)?;
    let p = basis.n_features();
    let n_max = basis.n_max;
    let n = pos.len();
    let mut out: Vec<AtomDescriptor> = (0..n)
        .map(|i| AtomDescriptor {
            b: vec![0.0; p],
            grad: vec![(i, vec![[0.0; 3]; p])],
        })
        .collect();
    let mut rn = vec![0.0; n_max];
    let mut drn = vec![0.0; n_max];
    for i in 0..n {
        for j in i + 1..n {
            let Some((r, u)) = pair_geometry(pos[i], pos[j], basis.r_cut) else {
                continue;
            };
            radial(n_max, r, basis.r_cut, &mut rn, &mut drn);
            // dB_i/dr_j (gi) and dB_j/dr_i (gj) for this pair; the self entries get the negatives.
            let mut gi = vec![[0.0; 3]; p];
            let mut gj = vec![[0.0; 3]; p];
            for nn in 0..n_max {
                let g = [drn[nn] * u[0], drn[nn] * u[1], drn[nn] * u[2]];
                let ki = basis.body2_index(s[j], nn);
                out[i].b[ki] += rn[nn];
                gi[ki] = g;
                let self_i = &mut out[i].grad[0].1[ki];
                for a in 0..3 {
                    self_i[a] -= g[a];
                }
                let kj = basis.body2_index(s[i], nn);
                out[j].b[kj] += rn[nn];
                gj[kj] = [-g[0], -g[1], -g[2]];
                let self_j = &mut out[j].grad[0].1[kj];
                for a in 0..3 {
                    self_j[a] += g[a];
                }
            }
            out[i].grad.push((j, gi));
            out[j].grad.push((i, gj));
        }
    }
    Ok(out)
}

/// Distance and unit vector from `ri` to `rj` if the pair is inside the cutoff; `None` at or
/// beyond r_cut. Coincident atoms give a zero unit vector (see the module doc).
#[inline]
pub(crate) fn pair_geometry(ri: [f64; 3], rj: [f64; 3], r_cut: f64) -> Option<(f64, [f64; 3])> {
    let d = [rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]];
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    if r2 >= r_cut * r_cut {
        return None;
    }
    let r = r2.sqrt();
    if r == 0.0 {
        return Some((0.0, [0.0; 3]));
    }
    let inv = 1.0 / r;
    Some((r, [d[0] * inv, d[1] * inv, d[2] * inv]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_defaults_sort_and_dedup_species_and_lay_out_body2_features() {
        let b = Basis::new(&[8, 1, 8, 6]);
        assert_eq!(b.species, vec![1, 6, 8]);
        assert_eq!((b.n_max, b.l_max, b.r_cut, b.degree_cap, b.body3), (8, 3, 6.0, 10, false));
        assert_eq!(b.n_species(), 3);
        assert_eq!(b.n_features(), 24);
        assert_eq!(b.species_index(6), Some(1));
        assert_eq!(b.species_index(2), None);
        assert_eq!(b.species_indices(&[8, 1]), Ok(vec![2, 0]));
        assert_eq!(b.species_indices(&[8, 3, 1]), Err(DescribeError::UnknownSpecies(3)));
        assert_eq!(b.body2_index(0, 0), 0);
        assert_eq!(b.body2_index(1, 0), 8);
        assert_eq!(b.body2_index(2, 7), 23);
        let names = b.feature_names();
        assert_eq!(names.len(), 24);
        assert_eq!(names[0], "b2[Z=1,n=0]");
        assert_eq!(names[8], "b2[Z=6,n=0]");
        assert_eq!(names[23], "b2[Z=8,n=7]");
        assert_eq!(Basis::new(&[]).n_features(), 0);
    }

    #[test]
    fn pair_geometry_is_half_open_and_handles_coincidence() {
        assert_eq!(pair_geometry([0.0; 3], [6.0, 0.0, 0.0], 6.0), None);
        assert_eq!(pair_geometry([0.0; 3], [7.0, 0.0, 0.0], 6.0), None);
        let (r, u) = pair_geometry([1.0, 0.0, 0.0], [1.0, 0.0, -3.0], 6.0).unwrap();
        assert_eq!((r, u), (3.0, [0.0, 0.0, -1.0]));
        assert_eq!(pair_geometry([2.0; 3], [2.0; 3], 6.0), Some((0.0, [0.0; 3])));
    }

    #[test]
    fn coincident_atoms_count_as_neighbours_with_zero_gradient() {
        let b = Basis::new(&[1]);
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let d = describe(&b, &pos, &[1, 1]).unwrap();
        // R_n(0) f_c(0) = T_n(1) = 1 for every n.
        assert!(d[0].b.iter().all(|&v| v == 1.0));
        assert!(d[1].b.iter().all(|&v| v == 1.0));
        for atom in &d {
            assert_eq!(atom.grad.len(), 2);
            for (_, g) in &atom.grad {
                assert!(g.iter().all(|v| *v == [0.0; 3]));
            }
        }
    }

    #[test]
    #[should_panic(expected = "pos and z lengths differ")]
    fn describe_rejects_mismatched_lengths() {
        let _ = describe(&Basis::new(&[1]), &[[0.0; 3]], &[1, 1]);
    }
}
