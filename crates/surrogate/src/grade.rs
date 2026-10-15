//! The extrapolation grade of a linear model, per atom (M6 Task 11 round 4; the provider's rule
//! 5e). The D-optimality / MaxVol criterion of Podryabinkin & Shapeev, "Active learning of
//! linearly parametrized interatomic potentials", Comput. Mater. Sci. 140, 171 (2017), grades
//! whole configurations; the per-atom (neighbourhood) grade used here is that of Gubaev,
//! Podryabinkin & Shapeev, J. Chem. Phys. 148, 241727 (2018), and of the MLIP package (Novikov,
//! Gubaev, Podryabinkin, Shapeev, Mach. Learn.: Sci. Technol. 2, 025002 (2021)). Per atom is the
//! right object for a model that is a sum of per-atom terms: the grade does not grow or shrink
//! with the system size, and one extrapolating atom inside a sum of interpolating ones is caught
//! (a configuration's summed row can sit inside the training rows while one of its atoms does
//! not).
//!
//! # The grade
//!
//! The model is linear in the per-atom standardised descriptor x (`model` docs): E_i = c_s·x_i
//! for an atom of species s. Of the training atoms of species s (their rows X_s), MaxVol keeps an
//! **active set** A_s of r rows with (locally) maximal |det|: every other training row is then a
//! combination of the active rows with coefficients of magnitude ≤ 1 + δ. A query row x is
//! written the same way, xᵀ = aᵀA_s, and its grade is
//!
//!   γ = max_j |a_j| = max_j |(xᵀ A_s⁻¹)_j|.
//!
//! γ ≤ 1 means x lies inside the parallelotope the active rows span (interpolation); γ > 1 is
//! extrapolation by that factor along some active direction. The grade is invariant to any
//! invertible linear map of the features (the standardisation included). **What it does and does
//! not bound.** For the per-atom energy the model's value at x is aᵀ(its values at the active
//! rows) exactly, so |a_j| ≤ γ limits how far the linear form is carried beyond them — but the
//! per-atom energies at the active rows are never observed (only sums of them are), and the force
//! rows ∂x/∂r are not graded at all; so as a bound on force errors the grade is a **heuristic**,
//! supported by measurement: the 40-geometry transition test
//! (`providers` `h2_plus_h_falls_back_on_the_grade_until_enough_is_learned`) — with 1–3 learned
//! H₃ samples the model's forces on held-out H₂ + H were 33–65× their bound and γ refused every
//! one; every geometry it let through was ≤ 0.03× — and the GPU repro's hot run (round 3: a
//! model that had kept ~22 H₃ fallbacks answered three-body geometries 20× off in force under a
//! confident posterior).
//!
//! # Rank deficiency: the span first
//!
//! The training rows need not span all P features (an H₂-only store's per-atom rows span 12 of
//! 29 directions, measured). MaxVol runs in the span: U_s, the eigenvectors of X_sᵀX_s with
//! eigenvalue above τ_s = P·ε·tr (the rounding threshold of `novelty`, the same derivation), give
//! coordinates y = U_sᵀx of rank r_s; a query's component outside the span, ‖x − U_sU_sᵀx‖² >
//! N_null·τ_s, is γ = ∞. This is the per-atom analogue of rule 5d (`novelty`), **not the same
//! test**: 5d checks the *design rows* — a configuration's energy row (the sum of its atoms' x)
//! and its force rows (−∂Σx/∂r) — against the fit's own rows; a query's force rows can leave the
//! design span while every x_i stays inside the per-atom span, and an x_i can leave the per-atom
//! span while the summed row does not. Both are kept.
//!
//! # MaxVol
//!
//! Initial rows by greedy pivoted Gram–Schmidt (the row of largest residual norm, r times);
//! then swaps (Goreinov, Oseledets, Savostyanov, Tyrtyshnikov, Zamarashkin, Zagoruyko, "How to
//! find a good submatrix", 2010): B = Y A⁻¹; while max |B_ij| > 1 + δ, row i replaces active row
//! j (|det| grows by |B_ij|), δ = [`MAXVOL_TOL`], at most [`MAXVOL_MAX_SWAPS`] swaps. Converged,
//! each training row has γ ≤ 1 + δ; **not converged** (the cap reached) that guarantee is void,
//! so the species is marked `converged = false` and every query of it grades ∞ (rule 5e falls
//! back; the provider notes it) until a refit converges.
//! Cost: O(n r²) per swap with A⁻¹ recomputed (Gauss–Jordan, partial pivoting), a few swaps per
//! refit; per query O(P·r + r²) — the query's projection and one row–matrix product.

use crate::descriptor::Basis;
use crate::linalg::symmetric_eigen;

/// MaxVol stops when every training row's coefficients are within 1 + this.
pub const MAXVOL_TOL: f64 = 1e-3;

/// MaxVol gives up after this many swaps (module doc: the species is then not converged).
pub const MAXVOL_MAX_SWAPS: usize = 1000;

/// One species' active set (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesActive {
    /// Span basis: r rows of length P (orthonormal, standardised feature space).
    pub u: Vec<Vec<f64>>,
    /// A⁻¹ (r×r, row-major) of the active rows' span coordinates.
    pub a_inv: Vec<f64>,
    /// Which training rows are active (indices into the rows the set was built from).
    pub active: Vec<usize>,
    /// The unspanned-energy threshold N_null·τ (module doc); ∞ when everything is spanned.
    pub null_threshold: f64,
    /// MaxVol converged within [`MAXVOL_MAX_SWAPS`]; when not, every query grades ∞.
    pub converged: bool,
}

/// The extrapolation grade's active sets, one per species (module doc).
#[derive(Clone, Debug, PartialEq)]
pub struct ActiveSet {
    pub basis: Basis,
    /// 1/scale per feature (the model's standardisation).
    pub inv_scale: Vec<f64>,
    /// `None` for a species with no training atom.
    pub species: Vec<Option<SpeciesActive>>,
}

/// Inverse of a square matrix by Gauss–Jordan with partial pivoting; `None` if singular.
fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| m[x * n + col].abs().total_cmp(&m[y * n + col].abs()))?;
        if m[piv * n + col] == 0.0 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                m.swap(piv * n + k, col * n + k);
                inv.swap(piv * n + k, col * n + k);
            }
        }
        let p = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= p;
            inv[col * n + k] /= p;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[col * n + k];
                        inv[r * n + k] -= f * inv[col * n + k];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// The coefficients yᵀA⁻¹ of a span-coordinate row.
fn coefficients(y: &[f64], a_inv: &[f64], r: usize) -> Vec<f64> {
    (0..r).map(|j| (0..r).map(|k| y[k] * a_inv[k * r + j]).sum()).collect()
}

/// MaxVol on the rows `y` (each of length r), at most `max_swaps` swaps: the active row
/// indices, A⁻¹ and whether every row's coefficients ended within 1 + [`MAXVOL_TOL`].
fn maxvol(y: &[Vec<f64>], r: usize, max_swaps: usize) -> (Vec<usize>, Vec<f64>, bool) {
    // greedy pivoted Gram–Schmidt for the start
    let mut resid: Vec<Vec<f64>> = y.to_vec();
    let mut active: Vec<usize> = Vec::with_capacity(r);
    for _ in 0..r {
        let (best, _) = resid
            .iter()
            .enumerate()
            .filter(|(i, _)| !active.contains(i))
            .map(|(i, v)| (i, v.iter().map(|x| x * x).sum::<f64>()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least r rows");
        let q: Vec<f64> = {
            let n = resid[best].iter().map(|x| x * x).sum::<f64>().sqrt();
            resid[best].iter().map(|x| x / n).collect()
        };
        active.push(best);
        for v in resid.iter_mut() {
            let d: f64 = v.iter().zip(&q).map(|(a, b)| a * b).sum();
            for (x, qk) in v.iter_mut().zip(&q) {
                *x -= d * qk;
            }
        }
    }
    let build = |active: &[usize]| -> Vec<f64> { active.iter().flat_map(|&i| y[i].iter().copied()).collect() };
    let mut a_inv = invert(&build(&active), r).expect("the greedy rows are independent");
    let mut swaps = 0;
    loop {
        let mut worst = (0.0f64, 0usize, 0usize);
        for (i, yi) in y.iter().enumerate() {
            for (j, c) in coefficients(yi, &a_inv, r).into_iter().enumerate() {
                if c.abs() > worst.0 {
                    worst = (c.abs(), i, j);
                }
            }
        }
        if worst.0 <= 1.0 + MAXVOL_TOL {
            return (active, a_inv, true);
        }
        if swaps == max_swaps {
            return (active, a_inv, false);
        }
        active[worst.2] = worst.1;
        a_inv = invert(&build(&active), r).expect("a swap keeps the rows independent (|det| grew)");
        swaps += 1;
    }
}

impl ActiveSet {
    /// The active sets of the training atoms `atoms`: (species index, raw per-atom descriptor),
    /// standardised by the model's `scale` (0 means 1).
    pub fn build<'a>(basis: &Basis, scale: &[f64], atoms: impl IntoIterator<Item = (usize, &'a [f64])>) -> ActiveSet {
        ActiveSet::build_capped(basis, scale, atoms, MAXVOL_MAX_SWAPS)
    }

    /// [`build`](Self::build) with the caller's swap cap (the non-convergence test).
    pub fn build_capped<'a>(basis: &Basis, scale: &[f64], atoms: impl IntoIterator<Item = (usize, &'a [f64])>, max_swaps: usize) -> ActiveSet {
        let p = basis.n_features();
        let inv_scale: Vec<f64> = scale.iter().map(|&s| if s > 0.0 { 1.0 / s } else { 1.0 }).collect();
        let mut rows: Vec<Vec<Vec<f64>>> = vec![Vec::new(); basis.n_species()];
        for (s, b) in atoms {
            rows[s].push(b.iter().zip(&inv_scale).map(|(x, d)| x * d).collect());
        }
        let species = rows
            .into_iter()
            .map(|xs| {
                if xs.is_empty() {
                    return None;
                }
                // the span (module doc): X_sᵀX_s, compensated like the fit's Gram
                let mut g = vec![0.0; p * p];
                let mut c = vec![0.0; p * p];
                for x in &xs {
                    for i in 0..p {
                        if x[i] == 0.0 {
                            continue;
                        }
                        for j in 0..p {
                            let v = x[i] * x[j];
                            let k = i * p + j;
                            let t = g[k] + v;
                            c[k] += if g[k].abs() >= v.abs() { (g[k] - t) + v } else { (v - t) + g[k] };
                            g[k] = t;
                        }
                    }
                }
                for (gk, ck) in g.iter_mut().zip(&c) {
                    *gk += ck;
                }
                let tr: f64 = (0..p).map(|i| g[i * p + i]).sum();
                let tau = p as f64 * f64::EPSILON * tr;
                let (ev, v) = symmetric_eigen(&g, p);
                let keep: Vec<usize> = (0..p).filter(|&i| ev[i] > tau).collect();
                let n_null = p - keep.len();
                let u: Vec<Vec<f64>> = keep.iter().map(|&i| (0..p).map(|k| v[k * p + i]).collect()).collect();
                let r = u.len();
                let y: Vec<Vec<f64>> = xs.iter().map(|x| u.iter().map(|ui| ui.iter().zip(x).map(|(a, b)| a * b).sum()).collect()).collect();
                let (active, a_inv, converged) = maxvol(&y, r, max_swaps);
                Some(SpeciesActive { u, a_inv, active, null_threshold: if n_null > 0 { n_null as f64 * tau } else { f64::INFINITY }, converged })
            })
            .collect();
        ActiveSet { basis: basis.clone(), inv_scale, species }
    }

    /// Whether MaxVol converged for every species with training atoms (module doc).
    pub fn converged(&self) -> bool {
        self.species.iter().flatten().all(|s| s.converged)
    }

    /// The grade of one atom of species index `s` with raw descriptor `b` (module doc): ∞ for a
    /// species without training atoms, a component outside the training span, or a species
    /// whose MaxVol did not converge.
    pub fn grade(&self, s: usize, b: &[f64]) -> f64 {
        let Some(sa) = self.species.get(s).and_then(|x| x.as_ref()) else { return f64::INFINITY };
        if !sa.converged {
            return f64::INFINITY;
        }
        let x: Vec<f64> = b.iter().zip(&self.inv_scale).map(|(v, d)| v * d).collect();
        let y: Vec<f64> = sa.u.iter().map(|ui| ui.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
        let norm2: f64 = x.iter().map(|v| v * v).sum();
        let span2: f64 = y.iter().map(|v| v * v).sum();
        if norm2 - span2 > sa.null_threshold {
            return f64::INFINITY;
        }
        coefficients(&y, &sa.a_inv, y.len()).into_iter().fold(0.0f64, |m, c| m.max(c.abs()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MaxVol on random rows: every row's coefficients within 1 + δ, the active rows exactly 1.
    #[test]
    fn maxvol_bounds_every_row() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let r = 6;
        let y: Vec<Vec<f64>> = (0..200).map(|_| (0..r).map(|_| rnd()).collect()).collect();
        let (active, a_inv, converged) = maxvol(&y, r, MAXVOL_MAX_SWAPS);
        assert!(converged);
        for yi in &y {
            let g = coefficients(yi, &a_inv, r).into_iter().fold(0.0f64, |m, c| m.max(c.abs()));
            assert!(g <= 1.0 + MAXVOL_TOL + 1e-12, "{g}");
        }
        for &i in &active {
            let g = coefficients(&y[i], &a_inv, r).into_iter().fold(0.0f64, |m, c| m.max(c.abs()));
            assert!((g - 1.0).abs() < 1e-10);
        }
        // the cap: with no swap allowed the greedy start is not a MaxVol set here, and the flag
        // says so; a not-converged species grades every query ∞
        // heavy-tailed rows, where the greedy start alone is not a MaxVol set
        let y: Vec<Vec<f64>> = (0..200).map(|_| (0..r).map(|_| rnd() * rnd().abs().powi(3)).collect()).collect();
        let (_, _, conv0) = maxvol(&y, r, 0);
        assert!(!conv0, "test premise: the greedy start alone is not converged");
        let mut basis = Basis::new(&[1]);
        basis.body3 = false;
        basis.n_max = r;
        let capped = ActiveSet::build_capped(&basis, &vec![1.0; r], y.iter().map(|x| (0usize, &x[..])), 0);
        assert!(!capped.converged());
        assert_eq!(capped.grade(0, &y[0]), f64::INFINITY);
        let full = ActiveSet::build(&basis, &vec![1.0; r], y.iter().map(|x| (0usize, &x[..])));
        assert!(full.converged() && full.grade(0, &y[0]) <= 1.0 + MAXVOL_TOL + 1e-12);
    }
}
