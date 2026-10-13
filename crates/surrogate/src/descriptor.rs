//! Invariant descriptors of an atom's neighbourhood: the body-2 (l = 0) atomic basis on species
//! channels and the body-3 real-harmonic contractions under a total-degree cap, with analytic
//! position gradients. Pure f64, CPU.
//!
//! # The atomic basis on species channels
//!
//! For centre atom i, a neighbour species t (an index into the sorted species list, not a
//! number derived from Z — CLAUDE.md rule 1), radial index n and angular index (l, m), the
//! one-particle basis summed over the neighbours of that species is
//!
//!   A_{(t,n),lm}(i) = Σ_{j ≠ i, s_j = t, r_ij < r_cut} φ_{n,lm}(r_j − r_i),
//!   φ_{n,lm}(d) = R_n(|d|) f_c(|d|) S_lm(d / r_cut),
//!
//! with R_n f_c from [`crate::basis::radial`] and S_lm the real **solid** harmonics of
//! [`crate::basis::solid_harmonics`] (Racah normalised, S_00 = 1). The pair (t, n) is one
//! *channel*, c = t·n_max + n, of which there are C = S·n_max. The A are linear in the
//! neighbour density, so each costs one pass over the neighbours.
//!
//! **Undivided solid harmonics (the |r|^l question).** The spec writes the angular factor as
//! Y_lm(r̂) = S_lm(r)/|r|^l. This module keeps S_lm undivided, evaluated at the dimensionless
//! d/r_cut (S_lm is homogeneous of degree l, so S_lm(d/r_cut) = S_lm(d)/r_cut^l): the effective
//! radial function of the l channel is (r/r_cut)^l R_n(r) f_c(r). Both choices give rotation-invariant contractions
//! (the (r/r_cut)^l factor is a scalar per neighbour); the undivided form is chosen because φ
//! is then a polynomial in d times a smooth function of |d|: finite with a finite, continuous
//! gradient at every d including d = 0 — whereas Y_lm has an O(1/r) gradient and is 0/0 at
//! r = 0 (Task 1 review), which an interactive drag can produce before any fallback engages.
//! The price is a different (not smaller) radial span per l: {(r/r_cut)^l T_n(x) f_c} for
//! n < n_max instead of {T_n(x) f_c} — the same dimension, a subspace that trades small-r
//! flexibility in the high-l channels (on r/r_cut ∈ [0.17, 1] a degree-7 polynomial misses
//! 1/r^l by ~ρ^{−n_max} ≈ 1e-3 relative for l = 3, Bernstein parameter ρ ≈ 2.4 for the pole at
//! r = 0; the l = 3 term at a 1 Bohr bond is (1/6)³ of its value at r_cut, so short-range
//! angular structure costs larger coefficients, which the ridge dislikes) — the regular-solid-
//! harmonic form of Gaussian-type orbitals, r^l Y_lm × radial. In return the cap n₁ + n₂ + l ≤ D
//! is an honest total polynomial degree in the coordinates. The n_max convergence on a
//! short-bond case (H₂O) should be re-checked with body-3 on (Task 7/8; a synthetic O–H₂ probe
//! is in `tests/body3.rs`). The fit standardises every feature anyway. The 1/r_cut
//! is a fixed gauge, not physics: it keeps |S_lm| ≤ 1 inside the sphere so every A is O(number
//! of neighbours) and the products B carry no 1e4-sized terms whose cancellation would cost
//! the 1e-12 invariance (without it the l = 3 terms reach 216 at r_cut and the rotation test
//! sat at 1.4e-12 relative). At d = 0, S_lm(0) = 0 for l ≥ 1 and ∇S_1m(0) = ê_m/r_cut, so a
//! coincident neighbour contributes R_n(0) f_c(0) = 1 to the l = 0 channels only and a finite
//! ∇φ_{n,1m} = ê_m/r_cut to the gradient (`pair_geometry` returns u = 0 there, which the
//! formula below handles with no special case).
//!
//! # Body-2 features
//!
//! For l = 0 the A are already invariants (they depend on the distances only): the body-2
//! features are B_{i,k} = A_{(t,n),00}(i) with the flat index k = c = t·n_max + n — neighbour
//! species major, radial index minor, so the n_max coefficients of one (centre, neighbour)
//! species pair are contiguous. Count n_max·S per centre species (ruling S3), the same for
//! every centre species; the centre species selects the coefficient row, not the layout.
//!
//! # Body-3 features: the real-harmonic contraction
//!
//! Under a rotation Q the vector of real solid harmonics of fixed l transforms by a real
//! orthogonal (2l+1)×(2l+1) matrix D^l(Q) (the real Wigner matrix): S_l(Qd) = D^l(Q) S_l(d).
//! Hence A_l(i) → D^l A_l(i) for every channel, and for two channels c₁, c₂
//!
//!   Σ_m A_{c₁,lm} A_{c₂,lm} = A_{c₁,l}ᵀ A_{c₂,l} → A_{c₁,l}ᵀ D^lᵀ D^l A_{c₂,l} = A_{c₁,l}ᵀ A_{c₂,l}
//!
//! is invariant because D^l is orthogonal. Equivalently, by the addition theorem of the Racah
//! harmonics (`basis` docs), Σ_m S_lm(a) S_lm(b) = |a|^l |b|^l P_l(â·b̂), so
//!
//!   B_{i,(c₁,c₂,l)} = Σ_{j∈t₁} Σ_{k∈t₂} R_{n₁}f_c(r_ij) R_{n₂}f_c(r_ik) (r_ij r_ik / r_cut²)^l P_l(cos θ_jik)
//!
//! — a sum over neighbour PAIRS of a function of the two distances and the angle at the
//! centre: the body-3 (angular) information. The form Σ_m A_{lm} A_{l,−m} is the invariant of
//! the COMPLEX harmonics (where Y_{l,−m} = (−1)^m Y_lm*); with real harmonics it pairs cos(mφ)
//! with sin(mφ) partners and is not invariant (for l = 1 it is 2A_xA_y + A_z²). The
//! `every_body3_feature_is_invariant_under_random_rotations_and_permutations` test checks
//! both statements. The j = k terms of the double sum are included (they are body-2 functions
//! of another radial shape, R_{n₁}R_{n₂}f_c² (r/r_cut)^{2l}, as in every ACE-type basis); the l = 0
//! terms are the products of two body-2 sums, which are not in the span of the body-2 block
//! (quadratic in the neighbour sums), so they are kept too.
//!
//! The product is symmetric in (c₁, c₂), so unordered channel pairs c₁ ≤ c₂ are stored once;
//! together with the total-degree cap n₁ + n₂ + l ≤ D (`degree_cap`, default 4) the block
//! holds, per centre species,
//!
//!   P₃ = ½ [ S² Σ_k N(k) c(k) + S Σ_n c(2n) ],   N(k) = #{(n₁, n₂) ∈ [0, n_max)² : n₁ + n₂ = k},
//!   c(k) = max(0, min(l_max, D − k) + 1)
//!
//! terms (the first sum counts ordered channel pairs, the second the diagonal c₁ = c₂, and the
//! half removes the double count of c₁ ≠ c₂). For n_max 8, l_max 3, D = 10: Σ_k N(k)c(k) = 182
//! and Σ_n c(2n) = 20, so P₃ = 101 (S = 1), 384 (S = 2), 1 496 (S = 4), i.e. P = n_max·S + P₃
//! = 109, 400, 1 528; without the cap (D ≥ 17) P = 144, 560, 2 144 — the spec review's
//! uncapped estimates. The Q′ independent columns (below) are then 109, 792 and 6 064.
//!
//! The cap's default is D = 4, chosen on the H₃⁺ fixture (Task 6 fix round, ruling S12: the
//! smallest D whose held-out window force rmse keeps ≥ 1.5× margin to 2e-3 Ha/Bohr and whose
//! energy rmse is within 10 % of D = 10's on both splits; the energy tier itself is not
//! measurable there, the teacher's egg-box floor 1.67e-4 Ha being above it):
//!
//! | D | P₃ S=1/2/4 | Q′ S=1/2/4 | window E / F (split 3) | window E / F (split 0) | all-20 E / F (3 / 0) | N=20 S=2 describe / forces |
//! |---|---|---|---|---|---|---|
//! | 4 | 21 / 76 / 288 | 29 / 176 / 1 232 | 8.26e-5 / 2.74e-4 | 7.90e-5 / 2.57e-4 | 6.43e-4 / 8.60e-4, 2.06e-4 / 4.01e-4 | 0.41 / 0.112 ms |
//! | 5 | 31 / 114 / 436 | 39 / 252 / 1 824 | 8.25e-5 / 3.15e-4 | 8.00e-5 / 2.68e-4 | 6.42e-4 / 9.01e-4, 2.10e-4 / 4.09e-4 | 0.45 / 0.096 ms |
//! | 6 | 43 / 160 / 616 | 51 / 344 / 2 544 | 8.14e-5 / 3.07e-4 | 8.39e-5 / 2.49e-4 | 6.38e-4 / 8.84e-4, 2.05e-4 / 4.09e-4 | 0.64 / 0.102 ms |
//! | 8 | 72 / 272 / 1 056 | 80 / 568 / 4 304 | 8.17e-5 / 3.10e-4 | 8.31e-5 / 2.52e-4 | 6.26e-4 / 8.44e-4, 1.99e-4 / 4.13e-4 | 1.08 / 0.115 ms |
//! | 10 | 101 / 384 / 1 496 | 109 / 792 / 6 064 | 8.60e-5 / 3.65e-4 | 8.28e-5 / 2.56e-4 | 6.17e-4 / 8.35e-4, 1.94e-4 / 4.16e-4 | 1.16 / 0.127 ms |
//!
//! (Ha and Ha/Bohr; the all-20 numbers include the three sparsely sampled stretched geometries.)
//! Three internal degrees of freedom and one angle per centre make H₃⁺ insensitive to D —
//! the evidence uses γ ≈ 13–18 directions whatever the basis holds — so D = 4 is the fixture's
//! answer, not a universal one (ruling S13: the smallest Q′ makes trust reachable with the
//! fewest samples); what the cap removes is sharp radial structure on the high-l terms
//! (l ≤ D − n₁ − n₂). **Setting for bent molecules with wide bond excursions: `degree_cap = 6`.**
//! The synthetic O–H₂ probe in `tests/body3.rs` (Morse pairs + harmonic bend, O–H 1.5–2.1 Bohr,
//! angle 85–125°, n_max 8, held-out E Ha / F Ha/Bohr) measures D = 4: 9.3e-7 / 7.9e-6,
//! D = 6: 2.8e-7 / 2.8e-6, D = 10: 4.0e-8 / 5.9e-7, uncapped: 2.3e-8 / 3.0e-7 — a factor 3
//! from D = 4 to 6 at a level 100× under the tier for that smooth teacher, with Q′ 176 → 344 at
//! S = 2.
//!
//! Layout: the body-2 block first (unchanged), then the terms of [`Basis::body3_terms`] in
//! increasing (c₁, c₂, l) order: all l of one channel pair contiguous, pairs in increasing
//! (c₁, c₂) with c₁ ≤ c₂. Feature index n_max·S + q for the q-th term.
//!
//! # Gradients
//!
//! φ depends on d_ij = r_j − r_i only, so ∂φ/∂r_j = −∂φ/∂r_i = ∇φ with, for u = d/|d| and
//! g_n = d[R_n f_c]/dr,
//!
//!   ∇φ_{n,lm}(d) = g_n(r) S_lm(d/r_cut) u + R_n(r) f_c(r) (∇S_lm)(d/r_cut) / r_cut.
//!
//! Body-2 (S_00 = 1, ∇S_00 = 0): ∂B_{i,(t,n)}/∂r_j = δ_{s_j t} g_n(r_ij) u_ij for j ≠ i and
//! ∂B_i/∂r_i = −Σ_j of that. Body-3, by the product rule through both factors:
//!
//!   ∂B_{i,(c₁,c₂,l)}/∂r_j = Σ_m [ δ_{s_j t₁} ∇φ_{n₁,lm}(d_ij) A_{c₂,lm}(i)
//!                                + A_{c₁,lm}(i) δ_{s_j t₂} ∇φ_{n₂,lm}(d_ij) ],
//!
//! for c₁ = c₂ both terms are present and equal (2 A ∇φ), and again ∂B_i/∂r_i = −Σ_j ∂B_i/∂r_j
//! (translation invariance: Σ_j ∂B_i/∂r_j = 0 exactly, so ΣF = 0 by construction in every
//! consumer). Rotation invariance of each B_i gives Σ_j d_ij × ∂B_i/∂r_j = 0, hence zero total
//! torque — analytically; numerically the body-3 torque cancels between different neighbours
//! of a centre (the per-neighbour body-3 force is not central), so it is zero to rounding of
//! those terms rather than identically. At r_cut both R_n f_c and g_n vanish, so a pair leaving
//! the sphere changes nothing discontinuously (E is C¹, F continuous).
//!
//! Cost of [`describe`] per atom: one `radial` + one `solid_harmonics` per neighbour, then
//! O(neighbours · P₃ · (2l+1)) for the gradients; the per-step hot path
//! ([`crate::model::energy_and_forces`]) contracts the coefficients first (an adjoint
//! W_{c,lm} = ∂E_i/∂A_{c,lm}) and is O(neighbours · n_max · (l_max+1)²) per atom after
//! O(P₃ · (2l+1)) for W — no per-feature gradient is ever formed there.
//!
//! # Column classes
//!
//! Hetero-pair body-2 columns are tied across centres (the same pair function counted from
//! either end, Task 3). A body-3 column is not: the centre is the vertex of the angle, so
//! (centre s, (t₁n₁, t₂n₂, l)) and (centre t₁, (s n₁, t₂n₂, l)) are different functions; the
//! only within-centre symmetry is c₁ ↔ c₂, removed by storing each unordered pair once. Each
//! body-3 column is therefore its own class (`body3_columns_are_their_own_classes_lambda_zero_rank_test`
//! pins it with a λ = 0 Gram on noise-free data).
//!
//! # Neighbour search
//!
//! O(N²) with the r² < r_cut² test, no cell list: the surrogate targets N ≤ 200 atoms, where a
//! 20 000-pair sweep is far below one descriptor evaluation's useful work (a cell list is a later
//! performance item). The square root and the basis functions are evaluated for pairs inside only.

use crate::basis::{radial, solid_harmonics};
use std::fmt;

/// Basis definition: what the descriptor vector of an atom consists of. `species` is the sorted,
/// deduplicated list of Z the model knows; a Z not in it is a [`DescribeError::UnknownSpecies`],
/// never a guess.
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    /// Number of radial functions, n = 0..n_max (half-open).
    pub n_max: usize,
    /// Highest angular momentum of the body-3 block (unused while `body3` is false;
    /// ≤ [`crate::basis::L_MAX_SUPPORTED`]).
    pub l_max: usize,
    /// Cutoff radius in Bohr.
    pub r_cut: f64,
    /// Total-degree cap n₁ + n₂ + l ≤ degree_cap of the body-3 block (unused while `body3` is false).
    pub degree_cap: usize,
    /// Whether the body-3 block is part of the feature vector.
    pub body3: bool,
    /// Sorted, unique nuclear charges.
    pub species: Vec<u8>,
}

/// One body-3 feature: the contraction Σ_m A_{(t₁,n₁),lm} A_{(t₂,n₂),lm} of two channels with
/// (t₁, n₁) ≤ (t₂, n₂) in channel order (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Body3Term {
    pub t1: usize,
    pub n1: usize,
    pub t2: usize,
    pub n2: usize,
    pub l: usize,
}

impl Basis {
    /// Defaults: n_max 8, l_max 3, r_cut 6.0 Bohr, body-3 on with degree_cap 4. `species` is
    /// sorted and deduplicated.
    ///
    /// Why body-3 (ruling S11): on the H₃⁺ fixture (`tests/body3.rs`) body-2 reaches 3.8e-4 Ha /
    /// 1.5e-3 Ha/Bohr held-out on the ±0.3 Bohr window — it misses the 1e-4 Ha energy tier —
    /// while body-3 reaches 8e-5 / 3e-4. Why D = 4 (ruling S12, table in the module doc): every
    /// D ∈ {4, 5, 6, 8, 10} gives the same held-out numbers on that fixture within 5 %, so the
    /// smallest D with ≥ 1.5× force margin wins; it also keeps Q′ (= the trust rule's sample
    /// count) at 29 / 176 / 1 232 for S = 1 / 2 / 4. A short-bond synthetic O–H₂ probe
    /// (`n_max_short_bond_probe`) shows D = 4 losing a factor ~5 against D ≥ 6 at a level 60×
    /// below the tier — raise `degree_cap` to 6 for bent molecules with wide bond-length
    /// excursions if a real fixture shows it.
    pub fn new(species: &[u8]) -> Basis {
        let mut species = species.to_vec();
        species.sort_unstable();
        species.dedup();
        Basis {
            n_max: 8,
            l_max: 3,
            r_cut: 6.0,
            degree_cap: 4,
            body3: true,
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

    /// Size of the body-2 block: n_max·S (also the number of channels C).
    pub fn n_body2(&self) -> usize {
        self.n_max * self.species.len()
    }

    /// Flat feature index of the body-2 feature (neighbour species t, radial n): t·n_max + n —
    /// also the channel index of (t, n) in the atomic basis.
    #[inline]
    pub fn body2_index(&self, t: usize, n: usize) -> usize {
        t * self.n_max + n
    }

    /// Number of l values a channel pair with n₁ + n₂ = k carries: max(0, min(l_max, D − k) + 1).
    #[inline]
    fn l_count(&self, k: usize) -> usize {
        if k > self.degree_cap {
            0
        } else {
            self.l_max.min(self.degree_cap - k) + 1
        }
    }

    /// Size of the body-3 block P₃ (module doc); 0 while `body3` is off. Closed form in
    /// O(n_max): ½[S² Σ_k N(k) c(k) + S Σ_n c(2n)] — called from `n_features()` on every
    /// hot-path shape check, so not the O(C²) enumeration (`body3_terms().len()` cross-checks it
    /// in the tests).
    pub fn n_body3(&self) -> usize {
        if !self.body3 {
            return 0;
        }
        let s = self.n_species();
        let n_max = self.n_max;
        // Σ_k N(k) c(k) over ordered (n₁, n₂), and the diagonal Σ_n c(2n).
        let mut ordered = 0;
        for n1 in 0..n_max {
            for n2 in 0..n_max {
                ordered += self.l_count(n1 + n2);
            }
        }
        let diagonal: usize = (0..n_max).map(|n| self.l_count(2 * n)).sum();
        (s * s * ordered + s * diagonal) / 2
    }

    /// The body-3 terms in feature order (module doc, "Layout"); empty while `body3` is off.
    pub fn body3_terms(&self) -> Vec<Body3Term> {
        let mut terms = Vec::with_capacity(self.n_body3());
        if !self.body3 {
            return terms;
        }
        let c = self.n_body2();
        for c1 in 0..c {
            let (t1, n1) = (c1 / self.n_max, c1 % self.n_max);
            for c2 in c1..c {
                let (t2, n2) = (c2 / self.n_max, c2 % self.n_max);
                for l in 0..self.l_count(n1 + n2) {
                    terms.push(Body3Term { t1, n1, t2, n2, l });
                }
            }
        }
        terms
    }

    /// Independent coefficient columns of the stacked layout q = s·P + k (centre species s,
    /// feature k): one class per representative, listing the full columns tied to it, in
    /// increasing representative order. Body-2 feature (t, n) of centre species s and feature
    /// (s, n) of centre species t are the SAME function of the positions
    /// (Σ_{i∈s, j∈t} R_n f_c(r_ij) counted from either end), so the energy depends only on the
    /// sum of their two coefficients; the fit solves for the sum and splits it equally (`fit`
    /// docs). The representative is the column with s ≤ t. Body-3 columns are all independent
    /// (the centre is the vertex of the angle, and unordered channel pairs are stored once) and
    /// each forms its own class.
    pub fn column_classes(&self) -> Vec<Vec<usize>> {
        let p = self.n_features();
        let p2 = self.n_body2();
        let s_n = self.n_species();
        let mut classes = Vec::with_capacity(self.n_independent_columns());
        for s in 0..s_n {
            for t in 0..s_n {
                for n in 0..self.n_max {
                    let q = s * p + self.body2_index(t, n);
                    if t == s {
                        classes.push(vec![q]);
                    } else if s < t {
                        classes.push(vec![q, t * p + self.body2_index(s, n)]);
                    }
                }
            }
            for k in p2..p {
                classes.push(vec![s * p + k]);
            }
        }
        classes
    }

    /// Number of independent coefficient columns Q′ (see [`Basis::column_classes`]):
    /// n_max·S(S+1)/2 body-2 classes plus S·P₃ body-3 columns.
    pub fn n_independent_columns(&self) -> usize {
        let s = self.n_species();
        self.n_max * s * (s + 1) / 2 + s * self.n_body3()
    }

    /// Length P of the per-atom descriptor vector: n_max·S + P₃.
    pub fn n_features(&self) -> usize {
        self.n_body2() + self.n_body3()
    }

    /// Human-readable name of every feature, in index order (for debugging and the sidecar
    /// header): `b2[Z=<neighbour Z>,n=<n>]`, then `b3[Z=<Z₁>,n=<n₁>;Z=<Z₂>,n=<n₂>;l=<l>]`.
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.n_features());
        for &zt in &self.species {
            for n in 0..self.n_max {
                names.push(format!("b2[Z={zt},n={n}]"));
            }
        }
        for t in self.body3_terms() {
            names.push(format!(
                "b3[Z={},n={};Z={},n={};l={}]",
                self.species[t.t1], t.n1, self.species[t.t2], t.n2, t.l
            ));
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

/// Per-neighbour pieces of the atomic basis of one centre: the values φ_{n,lm} are summed into
/// `a` as they are produced; the gradients ∇φ_{n,lm} are kept per neighbour slot for the
/// gradient pass. Allocated once per call, sized for the worst case of N − 1 neighbours.
pub(crate) struct AtomicBasisScratch {
    pub n_max: usize,
    pub nlm: usize,
    /// A_{c,lm}, row-major [channel][lm].
    pub a: Vec<f64>,
    /// ∇φ_{n,lm} per neighbour slot: [slot][n][lm].
    pub dphi: Vec<[f64; 3]>,
    /// Atom index of every neighbour slot, ascending.
    pub neighbours: Vec<usize>,
    rn: Vec<f64>,
    drn: Vec<f64>,
    sh: Vec<f64>,
    dsh: Vec<[f64; 3]>,
}

impl AtomicBasisScratch {
    pub fn new(basis: &Basis, n_atoms: usize) -> AtomicBasisScratch {
        let n_max = basis.n_max;
        let nlm = (basis.l_max + 1) * (basis.l_max + 1);
        AtomicBasisScratch {
            n_max,
            nlm,
            a: vec![0.0; basis.n_body2() * nlm],
            dphi: vec![[0.0; 3]; n_atoms.saturating_sub(1) * n_max * nlm],
            neighbours: Vec::with_capacity(n_atoms),
            rn: vec![0.0; n_max],
            drn: vec![0.0; n_max],
            sh: vec![0.0; nlm],
            dsh: vec![[0.0; 3]; nlm],
        }
    }

    /// Start a new centre.
    #[inline]
    pub fn clear(&mut self) {
        self.a.iter_mut().for_each(|v| *v = 0.0);
        self.neighbours.clear();
    }

    /// Add neighbour `j` of species `t` at d = r_j − r_i (|d| = r < r_cut, u = d/r or 0): its
    /// φ_{n,lm} into A_{(t,n),lm} and its ∇φ_{n,lm} into the next slot.
    #[inline]
    pub fn add_neighbour(&mut self, basis: &Basis, j: usize, t: usize, d: [f64; 3], r: f64, u: [f64; 3]) {
        let (n_max, nlm) = (self.n_max, self.nlm);
        radial(n_max, r, basis.r_cut, &mut self.rn, &mut self.drn);
        let inv_rc = 1.0 / basis.r_cut;
        solid_harmonics(basis.l_max, [d[0] * inv_rc, d[1] * inv_rc, d[2] * inv_rc], &mut self.sh, &mut self.dsh);
        for ds in self.dsh.iter_mut() {
            *ds = [ds[0] * inv_rc, ds[1] * inv_rc, ds[2] * inv_rc];
        }
        let slot = self.neighbours.len();
        self.neighbours.push(j);
        let base = slot * n_max * nlm;
        for nn in 0..n_max {
            let (rn, g) = (self.rn[nn], self.drn[nn]);
            let a_row = &mut self.a[(t * n_max + nn) * nlm..(t * n_max + nn + 1) * nlm];
            let d_row = &mut self.dphi[base + nn * nlm..base + (nn + 1) * nlm];
            for lm in 0..nlm {
                let s = self.sh[lm];
                let ds = self.dsh[lm];
                a_row[lm] += rn * s;
                d_row[lm] = [g * s * u[0] + rn * ds[0], g * s * u[1] + rn * ds[1], g * s * u[2] + rn * ds[2]];
            }
        }
    }

    /// ∇φ_{n,lm} of neighbour slot `slot`, radial index `n`, for lm = 0..nlm.
    #[inline]
    pub fn dphi(&self, slot: usize, n: usize) -> &[[f64; 3]] {
        let base = (slot * self.n_max + n) * self.nlm;
        &self.dphi[base..base + self.nlm]
    }

    /// A_{c,lm} of channel `c`, lm = 0..nlm.
    #[inline]
    pub fn a(&self, c: usize) -> &[f64] {
        &self.a[c * self.nlm..(c + 1) * self.nlm]
    }

    /// Position of the first m of degree l in an lm row (lm_index(l, −l)) and the row length 2l+1.
    #[inline]
    pub fn l_range(l: usize) -> std::ops::Range<usize> {
        l * l..(l + 1) * (l + 1)
    }
}

/// Descriptors and their gradients for every atom (the fit path; the per-step hot path is
/// [`crate::model::energy_and_forces`], which never materialises these lists).
///
/// # Panics
/// If `pos.len() != z.len()`, or `basis.body3` with `l_max` above [`crate::basis::L_MAX_SUPPORTED`].
pub fn describe(
    basis: &Basis,
    pos: &[[f64; 3]],
    z: &[u8],
) -> Result<Vec<AtomDescriptor>, DescribeError> {
    assert_eq!(pos.len(), z.len(), "describe: pos and z lengths differ");
    let s = basis.species_indices(z)?;
    let p = basis.n_features();
    let p2 = basis.n_body2();
    let n_max = basis.n_max;
    let n = pos.len();
    let terms = basis.body3_terms();
    let mut out: Vec<AtomDescriptor> = (0..n)
        .map(|i| AtomDescriptor {
            b: vec![0.0; p],
            grad: vec![(i, vec![[0.0; 3]; p])],
        })
        .collect();
    let mut rn = vec![0.0; n_max];
    let mut drn = vec![0.0; n_max];
    let mut scratch = basis.body3.then(|| AtomicBasisScratch::new(basis, n));
    for i in 0..n {
        if let Some(sc) = scratch.as_mut() {
            sc.clear();
        }
        // Pass 1: neighbours in ascending order — body-2 values and gradients, atomic basis.
        for j in 0..n {
            if j == i {
                continue;
            }
            let Some((r, u)) = pair_geometry(pos[i], pos[j], basis.r_cut) else {
                continue;
            };
            radial(n_max, r, basis.r_cut, &mut rn, &mut drn);
            let mut gj = vec![[0.0; 3]; p];
            let (bi, self_g) = {
                let atom = &mut out[i];
                let (b, grad) = (&mut atom.b, &mut atom.grad);
                (b, &mut grad[0].1)
            };
            for nn in 0..n_max {
                let k = basis.body2_index(s[j], nn);
                bi[k] += rn[nn];
                let g = [drn[nn] * u[0], drn[nn] * u[1], drn[nn] * u[2]];
                gj[k] = g;
                for a in 0..3 {
                    self_g[k][a] -= g[a];
                }
            }
            if let Some(sc) = scratch.as_mut() {
                let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
                sc.add_neighbour(basis, j, s[j], d, r, u);
            }
            out[i].grad.push((j, gj));
        }
        let Some(sc) = scratch.as_ref() else {
            continue;
        };
        // Body-3 values: B = Σ_m A_{c₁,lm} A_{c₂,lm}.
        for (q, t) in terms.iter().enumerate() {
            let (a1, a2) = (sc.a(basis.body2_index(t.t1, t.n1)), sc.a(basis.body2_index(t.t2, t.n2)));
            let mut b = 0.0;
            for lm in AtomicBasisScratch::l_range(t.l) {
                b += a1[lm] * a2[lm];
            }
            out[i].b[p2 + q] = b;
        }
        // Body-3 gradients: the product rule through both factors, per neighbour slot; the self
        // entry is minus the sum.
        let atom = &mut out[i];
        for (slot, &j) in sc.neighbours.iter().enumerate() {
            let tj = s[j];
            // grad[0] is the self entry; the neighbour entries follow in slot order.
            let (self_entry, rest) = atom.grad.split_at_mut(1);
            let gj = &mut rest[slot].1;
            debug_assert_eq!(rest[slot].0, j);
            let self_g = &mut self_entry[0].1;
            for (q, t) in terms.iter().enumerate() {
                if t.t1 != tj && t.t2 != tj {
                    continue;
                }
                let mut g = [0.0; 3];
                let range = AtomicBasisScratch::l_range(t.l);
                if t.t1 == tj {
                    let a2 = sc.a(basis.body2_index(t.t2, t.n2));
                    let dp = sc.dphi(slot, t.n1);
                    for lm in range.clone() {
                        for a in 0..3 {
                            g[a] += dp[lm][a] * a2[lm];
                        }
                    }
                }
                if t.t2 == tj {
                    let a1 = sc.a(basis.body2_index(t.t1, t.n1));
                    let dp = sc.dphi(slot, t.n2);
                    for lm in range {
                        for a in 0..3 {
                            g[a] += a1[lm] * dp[lm][a];
                        }
                    }
                }
                gj[p2 + q] = g;
                for a in 0..3 {
                    self_g[p2 + q][a] -= g[a];
                }
            }
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
        assert_eq!((b.n_max, b.l_max, b.r_cut, b.degree_cap, b.body3), (8, 3, 6.0, 4, true));
        assert_eq!(b.n_species(), 3);
        assert_eq!(b.n_body2(), 24);
        // P₃ at D = 4, S = 3: ½[9·34 + 3·8] = 165 (Σ_k N(k)c(k) = 34, Σ_n c(2n) = 8).
        assert_eq!(b.n_body3(), 165);
        assert_eq!(b.n_features(), 189);
        assert_eq!(b.body3_terms().len(), b.n_body3(), "closed form vs enumeration");
        assert_eq!(b.n_independent_columns(), 8 * 6 + 3 * 165);
        assert_eq!(b.species_index(6), Some(1));
        assert_eq!(b.species_index(2), None);
        assert_eq!(b.species_indices(&[8, 1]), Ok(vec![2, 0]));
        assert_eq!(b.species_indices(&[8, 3, 1]), Err(DescribeError::UnknownSpecies(3)));
        assert_eq!(b.body2_index(0, 0), 0);
        assert_eq!(b.body2_index(1, 0), 8);
        assert_eq!(b.body2_index(2, 7), 23);
        let names = b.feature_names();
        assert_eq!(names.len(), b.n_features());
        assert_eq!(names[0], "b2[Z=1,n=0]");
        assert_eq!(names[8], "b2[Z=6,n=0]");
        assert_eq!(names[23], "b2[Z=8,n=7]");
        assert_eq!(Basis::new(&[]).n_features(), 0);
        let mut b2 = b.clone();
        b2.body3 = false;
        assert_eq!(b2.n_features(), 24);
    }

    #[test]
    fn column_classes_tie_the_two_orderings_of_a_hetero_pair() {
        let mut b = Basis::new(&[1, 8]);
        b.body3 = false;
        b.n_max = 2; // P = 4: block H = k 0,1 (neighbour H), k 2,3 (neighbour O); Q = 8
        let classes = b.column_classes();
        // Centre H: (H,0),(H,1) alone; (O,0),(O,1) tied to centre O's (H,0),(H,1) = q 4, 5.
        // Centre O: (O,0),(O,1) = q 6, 7 alone.
        assert_eq!(
            classes,
            vec![vec![0], vec![1], vec![2, 4], vec![3, 5], vec![6], vec![7]]
        );
        assert_eq!(classes.len(), b.n_independent_columns());
        let mut three = Basis::new(&[1, 6, 8]);
        three.body3 = false;
        assert_eq!(three.column_classes().len(), three.n_independent_columns());
        assert_eq!(three.n_independent_columns(), 8 * 6);
        let mut one = Basis::new(&[1]);
        one.body3 = false;
        assert_eq!(one.column_classes(), (0..8).map(|q| vec![q]).collect::<Vec<_>>());
    }

    #[test]
    fn column_classes_with_body3_keep_the_body2_ties_and_add_one_class_per_body3_column() {
        let mut b = Basis::new(&[1, 8]);
        b.body3 = true;
        b.n_max = 2;
        b.l_max = 1;
        // Channels: (H,0) (H,1) (O,0) (O,1); 10 unordered pairs × 2 l (cap 10 never binds) = 20.
        assert_eq!(b.n_body3(), 20);
        let p = b.n_features();
        assert_eq!(p, 24);
        let classes = b.column_classes();
        assert_eq!(classes.len(), b.n_independent_columns());
        assert_eq!(classes.len(), 6 + 2 * 20);
        assert_eq!(&classes[..4], &[vec![0], vec![1], vec![2, 4 + 20], vec![3, 5 + 20]]);
        // Representatives strictly increasing; every body-3 column alone.
        for w in classes.windows(2) {
            assert!(w[0][0] < w[1][0]);
        }
        for c in &classes {
            if c[0] % p >= 4 {
                assert_eq!(c.len(), 1);
            }
        }
        let terms = b.body3_terms();
        assert_eq!(terms[0], Body3Term { t1: 0, n1: 0, t2: 0, n2: 0, l: 0 });
        assert_eq!(terms[2], Body3Term { t1: 0, n1: 0, t2: 0, n2: 1, l: 0 });
        assert_eq!(terms[19], Body3Term { t1: 1, n1: 1, t2: 1, n2: 1, l: 1 });
        assert_eq!(b.feature_names()[4], "b3[Z=1,n=0;Z=1,n=0;l=0]");
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
        let mut b = Basis::new(&[1]);
        b.body3 = false;
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

    /// Pins the species-channel SEMANTICS: block t of B_i is filled by neighbours of species t
    /// (species index into the sorted list), never by the centre's own species. O at the origin,
    /// H1 at +3.5 x̂ and H2 at −2.5 x̂: the H–H distance is exactly r_cut (excluded), so each H
    /// sees only O and O sees both H.
    #[test]
    fn species_channel_blocks_follow_the_neighbour_species() {
        let mut b = Basis::new(&[8, 1]); // sorted → [1, 8]: t(H) = 0, t(O) = 1
        b.body3 = false;
        let (t_h, t_o) = (0usize, 1usize);
        let n_max = b.n_max;
        let pos = [[0.0; 3], [3.5, 0.0, 0.0], [-2.5, 0.0, 0.0]];
        let z = [8, 1, 1];
        let d = describe(&b, &pos, &z).unwrap();
        let mut rn1 = vec![0.0; n_max];
        let mut rn2 = vec![0.0; n_max];
        let mut dr = vec![0.0; n_max];
        radial(n_max, 3.5, b.r_cut, &mut rn1, &mut dr);
        radial(n_max, 2.5, b.r_cut, &mut rn2, &mut dr);
        for nn in 0..n_max {
            // Centre O: neighbour-H block holds both H, neighbour-O block is empty.
            assert!((d[0].b[b.body2_index(t_h, nn)] - (rn1[nn] + rn2[nn])).abs() < 1e-15);
            assert_eq!(d[0].b[b.body2_index(t_o, nn)], 0.0);
            // Centre H1 / H2: neighbour-O block holds their own O distance, neighbour-H block empty.
            assert_eq!(d[1].b[b.body2_index(t_o, nn)], rn1[nn]);
            assert_eq!(d[1].b[b.body2_index(t_h, nn)], 0.0);
            assert_eq!(d[2].b[b.body2_index(t_o, nn)], rn2[nn]);
            assert_eq!(d[2].b[b.body2_index(t_h, nn)], 0.0);
        }
        // Gradient lists: O depends on itself and both H; each H on itself and O only.
        let partners = |i: usize| d[i].grad.iter().map(|(j, _)| *j).collect::<Vec<_>>();
        assert_eq!(partners(0), vec![0, 1, 2]);
        assert_eq!(partners(1), vec![1, 0]);
        assert_eq!(partners(2), vec![2, 0]);
        assert!(d[1].b.iter().zip(&d[2].b).any(|(a, c)| a != c), "fixture is asymmetric");

        // Hot-path semantics: coefficient row = CENTRE species, feature block = NEIGHBOUR species.
        // A unit coefficient on (centre H, neighbour H) sees no pair here; on (centre O,
        // neighbour H) it sees both O–H pairs once; (centre O, neighbour O) sees nothing.
        let unit_model = |row: usize, block: usize| {
            let mut m = crate::model::Model {
                basis: b.clone(),
                coeffs: vec![vec![0.0; b.n_features()]; 2],
                mean: vec![0.0; b.n_features()],
                scale: vec![1.0; b.n_features()],
                ref_energy: vec![0.0, 0.0],
                sigma_e: 0.0,
                sigma_f: 0.0,
                energy_per_atom: true,
                lambda: 0.0,
                sigma_inv_chol: None,
                n_samples: 0,
                provenance: crate::model::Provenance {
                    grid_n: 1,
                    h: 1.0,
                    sigma: 1.0,
                    tol: 1.0,
                    charge: 0,
                    multiplicity: None,
                    teacher: String::new(),
                },
                representatives: std::sync::OnceLock::new(),
            };
            m.coeffs[row][b.body2_index(block, 0)] = 1.0; // n = 0: R_0 f_c = f_c
            m
        };
        let energy = |m: &crate::model::Model, pos: &[[f64; 3]]| {
            crate::model::energy_and_forces(m, pos, &z).unwrap().0
        };
        let fc = |r: f64| crate::basis::cutoff(r, b.r_cut).0;
        assert_eq!(energy(&unit_model(t_h, t_h), &pos), 0.0);
        assert_eq!(energy(&unit_model(t_o, t_o), &pos), 0.0);
        assert!((energy(&unit_model(t_o, t_h), &pos) - (fc(3.5) + fc(2.5))).abs() < 1e-15);
        assert!((energy(&unit_model(t_h, t_o), &pos) - (fc(3.5) + fc(2.5))).abs() < 1e-15);
        // With the H–H pair inside the cutoff, (centre H, neighbour H) counts it from both ends.
        let close = [[0.0; 3], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0]];
        let r_hh = (2.0f64 * 2.0 + 2.5 * 2.5).sqrt();
        assert!((energy(&unit_model(t_h, t_h), &close) - 2.0 * fc(r_hh)).abs() < 1e-15);
        assert_eq!(energy(&unit_model(t_o, t_o), &close), 0.0);
    }

    /// The body-3 block of a four-atom cluster against the double sum over neighbour pairs
    /// with the Legendre polynomial (module doc): Σ_{j,k} R_{n₁}f_c R_{n₂}f_c (r_j r_k/r_cut²)^l P_l(cos θ).
    #[test]
    fn body3_features_equal_the_neighbour_pair_sum_with_legendre_polynomials() {
        let mut b = Basis::new(&[1, 8]);
        b.body3 = true;
        let pos = [[0.1, -0.2, 0.3], [1.9, 0.4, -0.5], [-0.7, 2.2, 1.1], [1.0, -1.5, -2.0]];
        let z = [8, 1, 1, 8];
        let d = describe(&b, &pos, &z).unwrap();
        let p2 = b.n_body2();
        let legendre = |l: usize, x: f64| match l {
            0 => 1.0,
            1 => x,
            2 => 0.5 * (3.0 * x * x - 1.0),
            3 => 0.5 * (5.0 * x * x * x - 3.0 * x),
            _ => unreachable!(),
        };
        let mut rn = vec![0.0; b.n_max];
        let mut drn = vec![0.0; b.n_max];
        let mut worst: f64 = 0.0;
        for i in 0..4 {
            // per neighbour: species, r, unit vector, R_n f_c
            let nb: Vec<(usize, f64, [f64; 3], Vec<f64>)> = (0..4)
                .filter(|&j| j != i)
                .filter_map(|j| {
                    let (r, u) = pair_geometry(pos[i], pos[j], b.r_cut)?;
                    radial(b.n_max, r, b.r_cut, &mut rn, &mut drn);
                    Some((b.species_index(z[j]).unwrap(), r, u, rn.clone()))
                })
                .collect();
            for (q, t) in b.body3_terms().iter().enumerate() {
                let mut expect = 0.0;
                for (tj, rj, uj, rnj) in &nb {
                    if *tj != t.t1 {
                        continue;
                    }
                    for (tk, rk, uk, rnk) in &nb {
                        if *tk != t.t2 {
                            continue;
                        }
                        let cos = uj[0] * uk[0] + uj[1] * uk[1] + uj[2] * uk[2];
                        expect += rnj[t.n1] * rnk[t.n2] * (rj * rk / (b.r_cut * b.r_cut)).powi(t.l as i32) * legendre(t.l, cos);
                    }
                }
                let got = d[i].b[p2 + q];
                let err = (got - expect).abs() / expect.abs().max(1.0);
                worst = worst.max(err);
                assert!(err < 1e-12, "atom {i} term {t:?}: {got} vs {expect}");
            }
        }
        assert!(d.iter().any(|a| a.b[p2..].iter().any(|v| *v != 0.0)));
        eprintln!("body-3 vs Legendre double sum: worst {worst:.2e}");
    }

    #[test]
    #[should_panic(expected = "pos and z lengths differ")]
    fn describe_rejects_mismatched_lengths() {
        let _ = describe(&Basis::new(&[1]), &[[0.0; 3]], &[1, 1]);
    }
}
