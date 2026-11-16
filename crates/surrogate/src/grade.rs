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
//!
//! # The large basis (2026-11-17, the refit speed-up)
//!
//! A body-4 basis (a shared collection's, P ≈ 800 per species) takes its span from faer's
//! symmetric eigensolver (`dense::symmetric_eigen`: blocked, threaded) instead of Jacobi — the
//! same O(ε)‖G‖ backward error the threshold τ rests on, minutes faster at that size (Jacobi
//! sweeps cost 10P³ each); every other basis keeps Jacobi bit for bit, as `novelty` does. The Gram,
//! the projection onto the span and MaxVol's scan of every row's coefficients run on
//! [`crate::threads::worker_threads`] threads, each output written by one thread in the serial
//! loop's order, so they change no bit.
//!
//! **Why another eigensolver gives the same grade.** The eigenvectors of a cluster of (nearly)
//! equal eigenvalues are only defined up to a rotation within their span, and two solvers return
//! different ones; but the grade does not depend on the basis U chooses for the kept span. For
//! U' = U R (R orthogonal r×r) every row's coordinates become y' = Rᵀy, the active matrix
//! A' = A R and A'⁻¹ = Rᵀ A⁻¹, so the coefficients y'ᵀA'⁻¹ = yᵀR Rᵀ A⁻¹ = yᵀA⁻¹ are the same
//! numbers, and so are the greedy start (it ranks rows by residual norms, which R preserves) and
//! every swap (it ranks coefficients); the null test ‖x‖² − ‖Uᵀx‖² is the projection's, not the
//! basis's. Only the kept SPAN matters, and that is where the threshold comes in
//! (`a_rotation_within_the_kept_span_changes_no_grade`).
//!
//! **Eigenvalues within rounding of τ** (the kept span's edge). Two backward-stable solvers
//! return an eigenvalue to within a few ε‖G‖₂ of each other — the rigorous bound is O(P)·ε‖G‖₂
//! (Golub & Van Loan §8.1), the realistic one, rounding errors adding like a random walk,
//! √P·ε‖G‖₂ (Higham, *Accuracy and Stability*, §2.8) — so an eigenvalue in (τ, τ + δ],
//! δ = √P·ε·‖G‖₂, is spanned or null by the solver's rounding, not by the data. Such an
//! eigenvalue is counted **null** ([`span_split`]): its direction leaves MaxVol's coordinates and
//! joins the null test, the conservative side (a query reaching along it by more than N_null·τ
//! grades ∞ and falls back, where kept it would be graded against active rows that barely span
//! it). The rule is a fixed function of the computed spectrum, the same on any thread count
//! (faer's results do not depend on it, `dense` module doc). Measured on the user's 1 588-sample
//! H/O/P collection (2026-11-17; `providers` `refit_golden::active_span_margins`): δ/τ =
//! 0.015–0.034 per species, Jacobi and faer agree on every eigenvalue within (0.1τ, 10τ) to
//! ≤ 1.0e-3·τ, and the nearest eigenvalues sit at 1.055τ above and 0.89τ below — outside the band,
//! so the kept counts (194, 27, 48) and every verdict are Jacobi's. **τ sits in no spectral gap
//! there** (unlike the dimer stores of `novelty`, two decades either side): the span's edge is
//! set by the data to within 6 %, so a few new samples can move an eigenvalue across τ whatever
//! the solver (the band rule pins only the solver's part).

use crate::descriptor::Basis;
use crate::linalg::symmetric_eigen;
use std::ops::Range;

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

impl SpeciesActive {
    /// One species' active set from its standardised training rows `xs` and an orthonormal basis
    /// `u` (r rows of length P) of their span, with the null count and τ of the split: the rows'
    /// span coordinates (threaded, each row's own sums) and MaxVol on them.
    pub fn from_span(xs: &[Vec<f64>], u: Vec<Vec<f64>>, n_null: usize, tau: f64, max_swaps: usize) -> SpeciesActive {
        let _pm = crate::prof::span("grade: active set projection + MaxVol");
        let r = u.len();
        let y: Vec<Vec<f64>> = par_ranges(xs.len(), |rows| rows.map(|i| u.iter().map(|ui| ui.iter().zip(&xs[i]).map(|(a, b)| a * b).sum()).collect::<Vec<f64>>()).collect::<Vec<_>>()).concat();
        let (active, a_inv, converged) = maxvol(&y, r, max_swaps);
        SpeciesActive { u, a_inv, active, null_threshold: if n_null > 0 { n_null as f64 * tau } else { f64::INFINITY }, converged }
    }
}

/// One species' active set from its standardised training rows (each of length p): the span
/// (module doc) — X_sᵀX_s compensated like the fit's Gram, its eigenvectors above τ — and MaxVol
/// in it. `large`: faer's eigensolver with the rounding band at τ counted null (module doc, "The
/// large basis"; a body-4 basis); otherwise Jacobi and the bare comparison, bit for bit.
fn species_active(xs: &[Vec<f64>], p: usize, large: bool, max_swaps: usize) -> SpeciesActive {
    species_active_with(xs, p, large, max_swaps, crate::dense::symmetric_eigen)
}

/// The large path's eigensolver result: `solver`'s (faer's in a build), or Jacobi's when its QL
/// does not converge (`dense::symmetric_eigen`'s `Err`, never seen) — the active set is then
/// built as before the speed-up, slower, instead of the refit panicking. The band rule applies to
/// either spectrum.
type EigenResult = Result<(Vec<f64>, Vec<f64>), crate::linalg::LinalgError>;
fn large_span_eigen(g: &[f64], p: usize, solver: impl Fn(&[f64], usize) -> EigenResult) -> (Vec<f64>, Vec<f64>) {
    solver(g, p).unwrap_or_else(|_| {
        crate::prof::count("active set: QL did not converge, Jacobi instead", 1);
        symmetric_eigen(g, p)
    })
}

/// [`species_active`] with the large path's eigensolver given (the fallback test).
fn species_active_with(xs: &[Vec<f64>], p: usize, large: bool, max_swaps: usize, solver: impl Fn(&[f64], usize) -> EigenResult) -> SpeciesActive {
    crate::prof::count("active set: species blocks", 1);
    crate::prof::count("active set: features p (summed over species)", p as u64);
    crate::prof::count("active set: atoms (rows)", xs.len() as u64);
    let _pg = crate::prof::span("grade: active set Gram (threaded, compensated)");
    let g = gram(xs, p);
    let tr: f64 = (0..p).map(|i| g[i * p + i]).sum();
    let tau = p as f64 * f64::EPSILON * tr;
    drop(_pg);
    let _pe = crate::prof::span("grade: active set eigen (faer for body-4, else Jacobi)");
    let (v, (keep, n_null)) = if large {
        let (ev, v) = large_span_eigen(&g, p, solver);
        (v, span_split(&ev, tau))
    } else {
        let (ev, v) = symmetric_eigen(&g, p);
        let keep: Vec<usize> = (0..p).filter(|&i| ev[i] > tau).collect();
        let n_null = p - keep.len();
        (v, (keep, n_null))
    };
    drop(_pe);
    let u: Vec<Vec<f64>> = keep.iter().map(|&i| (0..p).map(|k| v[k * p + i]).collect()).collect();
    SpeciesActive::from_span(xs, u, n_null, tau, max_swaps)
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

/// Runs `f` over `n` items split into contiguous ranges on [`crate::threads::worker_threads`]
/// threads (below normal priority), the ranges' results in order.
fn par_ranges<R: Send>(n: usize, f: impl Fn(Range<usize>) -> R + Sync) -> Vec<R> {
    let threads = crate::threads::worker_threads().min(n.max(1));
    if threads <= 1 {
        return vec![f(0..n)];
    }
    let per = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(per)
            .map(|a| {
                scope.spawn(move || {
                    crate::threads::lower_priority();
                    f(a..(a + per).min(n))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("an active-set thread panicked")).collect()
    })
}

/// MaxVol's scan: the largest |(yᵢᵀA⁻¹)_j| over every row i and column j, and where — the first
/// (i, j) in row-major order attaining it, as the serial loop over [`coefficients`] finds it. The
/// rows are split among threads, each coefficient summed over k in order exactly as
/// `coefficients` sums it (row-oriented, so A⁻¹ is read along its rows), and the ranges' maxima
/// combined in order with a strict comparison: the same bits and the same indices on any thread
/// count.
fn worst_coefficient(y: &[Vec<f64>], a_inv: &[f64], r: usize) -> (f64, usize, usize) {
    let parts = par_ranges(y.len(), |rows| {
        let mut worst = (0.0f64, 0usize, 0usize);
        // `Sum for f64` folds from −0.0: the same start keeps every bit (a sign of zero only)
        let mut acc = vec![-0.0f64; r];
        for i in rows {
            acc.iter_mut().for_each(|a| *a = -0.0);
            for (k, &yk) in y[i].iter().enumerate() {
                for (a, &b) in acc.iter_mut().zip(&a_inv[k * r..(k + 1) * r]) {
                    *a += yk * b;
                }
            }
            for (j, c) in acc.iter().enumerate() {
                if c.abs() > worst.0 {
                    worst = (c.abs(), i, j);
                }
            }
        }
        worst
    });
    parts.into_iter().fold((0.0f64, 0, 0), |best, w| if w.0 > best.0 { w } else { best })
}

/// X_sᵀX_s over the rows `xs` (each of length p), compensated (Neumaier) like the fit's Gram: each
/// row of G accumulated by one thread over the rows of X in order, so the same bits on any thread
/// count as the serial loop.
fn gram(xs: &[Vec<f64>], p: usize) -> Vec<f64> {
    let parts = par_ranges(p, |is| {
        let mut g = vec![0.0; is.len() * p];
        let mut c = vec![0.0; is.len() * p];
        for x in xs {
            for (li, i) in is.clone().enumerate() {
                if x[i] == 0.0 {
                    continue;
                }
                for j in 0..p {
                    let v = x[i] * x[j];
                    let k = li * p + j;
                    let t = g[k] + v;
                    c[k] += if g[k].abs() >= v.abs() { (g[k] - t) + v } else { (v - t) + g[k] };
                    g[k] = t;
                }
            }
        }
        for (gk, ck) in g.iter_mut().zip(&c) {
            *gk += ck;
        }
        g
    });
    parts.concat()
}

/// Which eigenvalues span (module doc, "Eigenvalues within rounding of τ"): the indices of those
/// above τ + δ, δ = √P·ε·max|λ| (the solvers' rounding of an eigenvalue, ‖G‖₂ = max|λ| for the
/// symmetric G), in the solver's order; every other one — at or below τ, or within δ above it —
/// is null. Returns (kept, null count).
pub fn span_split(ev: &[f64], tau: f64) -> (Vec<usize>, usize) {
    let p = ev.len();
    let norm = ev.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let delta = (p as f64).sqrt() * f64::EPSILON * norm;
    let keep: Vec<usize> = (0..p).filter(|&i| ev[i] > tau + delta).collect();
    let n_null = p - keep.len();
    (keep, n_null)
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
        let worst = worst_coefficient(y, &a_inv, r);
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
        let species = rows.into_iter().map(|xs| (!xs.is_empty()).then(|| species_active(&xs, p, basis.body4, max_swaps))).collect();
        ActiveSet { basis: basis.clone(), inv_scale, species }
    }

    /// Measurement (`providers` `refit_golden::active_span_margins`): per species with training
    /// atoms, the spectrum of its standardised Gram X_sᵀX_s from Jacobi and from faer (each sorted)
    /// and τ — where the kept span's edge sits against the rounding band (module doc).
    #[doc(hidden)]
    pub fn species_spectra<'a>(basis: &Basis, scale: &[f64], atoms: impl IntoIterator<Item = (usize, &'a [f64])>) -> Vec<Option<(Vec<f64>, Vec<f64>, f64)>> {
        let p = basis.n_features();
        let inv_scale: Vec<f64> = scale.iter().map(|&s| if s > 0.0 { 1.0 / s } else { 1.0 }).collect();
        let mut rows: Vec<Vec<Vec<f64>>> = vec![Vec::new(); basis.n_species()];
        for (s, b) in atoms {
            rows[s].push(b.iter().zip(&inv_scale).map(|(x, d)| x * d).collect());
        }
        rows.into_iter()
            .map(|xs| {
                if xs.is_empty() {
                    return None;
                }
                let g = gram(&xs, p);
                let tau = p as f64 * f64::EPSILON * (0..p).map(|i| g[i * p + i]).sum::<f64>();
                let (mut ej, _) = symmetric_eigen(&g, p);
                let (mut ef, _) = crate::dense::symmetric_eigen(&g, p).expect("measurement: faer's QL did not converge");
                ej.sort_by(f64::total_cmp);
                ef.sort_by(f64::total_cmp);
                Some((ej, ef, tau))
            })
            .collect()
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

    /// Rows of a rank-`r` subspace of R^p with every 17th row heavier than the rest.
    fn rank_deficient_rows(seed: u64, n: usize, p: usize, r: usize) -> Vec<Vec<f64>> {
        let mut s = seed;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let basis: Vec<Vec<f64>> = (0..r).map(|_| (0..p).map(|_| rnd()).collect()).collect();
        (0..n)
            .map(|i| {
                let w = if i % 17 == 0 { 3.0 } else { 1.0 };
                let c: Vec<f64> = (0..r).map(|_| w * rnd()).collect();
                (0..p).map(|k| (0..r).map(|t| c[t] * basis[t][k]).sum()).collect()
            })
            .collect()
    }

    /// The grade is invariant to the basis chosen for the kept span (module doc, "Why another
    /// eigensolver gives the same grade"): U and R·U (R a random orthogonal r×r, mixing the span's
    /// basis vectors) give the same active rows, the same MaxVol verdict and the same grade of
    /// every query — training rows, combinations inside the span, and a query with a component
    /// outside it — to rounding.
    #[test]
    fn a_rotation_within_the_kept_span_changes_no_grade() {
        let (n, p, r) = (300, 24, 9);
        let xs = rank_deficient_rows(7, n, p, r);
        let g = gram(&xs, p);
        let tau = p as f64 * f64::EPSILON * (0..p).map(|i| g[i * p + i]).sum::<f64>();
        let (ev, v) = symmetric_eigen(&g, p);
        let keep: Vec<usize> = (0..p).filter(|&i| ev[i] > tau).collect();
        assert_eq!(keep.len(), r, "test premise: the rows span r directions");
        let u: Vec<Vec<f64>> = keep.iter().map(|&i| (0..p).map(|k| v[k * p + i]).collect()).collect();
        // R: Gram–Schmidt of random r-vectors; the rotated basis vectors are Σ_t R_st u_t
        let mut q: Vec<Vec<f64>> = Vec::new();
        for c in rank_deficient_rows(11, r, r, r) {
            let mut c = c;
            for b in &q {
                let d: f64 = c.iter().zip(b).map(|(x, y)| x * y).sum();
                c.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
            }
            let nn = c.iter().map(|x| x * x).sum::<f64>().sqrt();
            q.push(c.into_iter().map(|x| x / nn).collect());
        }
        let u_rot: Vec<Vec<f64>> = (0..r).map(|s| (0..p).map(|k| (0..r).map(|t| q[s][t] * u[t][k]).sum()).collect()).collect();
        let a = SpeciesActive::from_span(&xs, u, p - r, tau, MAXVOL_MAX_SWAPS);
        let b = SpeciesActive::from_span(&xs, u_rot, p - r, tau, MAXVOL_MAX_SWAPS);
        assert!(a.converged && b.converged);
        assert_eq!(a.active, b.active, "the same active rows");
        let mut basis = Basis::new(&[1]);
        basis.body3 = false;
        basis.n_max = p;
        let set = |sa: SpeciesActive| ActiveSet { basis: basis.clone(), inv_scale: vec![1.0; p], species: vec![Some(sa)] };
        let (sa, sb) = (set(a), set(b));
        let mut queries: Vec<Vec<f64>> = xs.iter().step_by(7).cloned().collect();
        queries.extend(rank_deficient_rows(7, 40, p, r).into_iter().map(|x| x.into_iter().map(|v| 1.7 * v).collect::<Vec<_>>()));
        let mut off = xs[3].clone();
        off[0] += 1e-3;
        queries.push(off);
        for x in &queries {
            let (ga, gb) = (sa.grade(0, x), sb.grade(0, x));
            assert!(ga == gb || (ga - gb).abs() <= 1e-10 * ga.abs().max(1.0), "grade {ga} vs {gb}");
        }
        assert_eq!(sa.grade(0, queries.last().unwrap()), f64::INFINITY, "a component outside the span grades ∞ in both");
    }

    /// The body-4 path (faer, the band rule) against Jacobi's on rank-deficient rows at a size past
    /// faer's blocking: the same kept count, the same active rows, the same grades to rounding.
    #[test]
    fn the_faer_span_grades_as_the_jacobi_span() {
        let (n, p, r) = (900, 160, 57);
        let xs = rank_deficient_rows(3, n, p, r);
        let mut basis = Basis::new(&[1]);
        basis.body3 = false;
        basis.n_max = p;
        let set = |large: bool| ActiveSet { basis: basis.clone(), inv_scale: vec![1.0; p], species: vec![Some(species_active(&xs, p, large, MAXVOL_MAX_SWAPS))] };
        let (jac, fae) = (set(false), set(true));
        let (j, f) = (jac.species[0].as_ref().unwrap(), fae.species[0].as_ref().unwrap());
        assert_eq!((j.u.len(), f.u.len()), (r, r));
        assert_eq!(j.active, f.active);
        assert_eq!(j.null_threshold, f.null_threshold);
        for x in xs.iter().step_by(11) {
            let (gj, gf) = (jac.grade(0, x), fae.grade(0, x));
            assert!((gj - gf).abs() <= 1e-9 * gj.abs().max(1.0), "{gj} vs {gf}");
        }
    }

    /// A QL that does not converge on the large path (`dense::symmetric_eigen`'s `Err`) falls back on
    /// Jacobi: no panic, and the same span and active rows as Jacobi's path (the band rule moves
    /// nothing on these rows), to rounding in the grade.
    #[test]
    fn a_failed_ql_falls_back_on_jacobi() {
        let (n, p, r) = (400, 40, 17);
        let xs = rank_deficient_rows(13, n, p, r);
        let g = gram(&xs, p);
        let failing = |_: &[f64], _: usize| -> EigenResult { Err(crate::linalg::LinalgError::NotPositiveDefinite(0)) };
        let (ev, v) = large_span_eigen(&g, p, failing);
        let (ej, vj) = symmetric_eigen(&g, p);
        assert_eq!((ev, v), (ej, vj), "the fallback is Jacobi's decomposition, bit for bit");
        let fell_back = species_active_with(&xs, p, true, MAXVOL_MAX_SWAPS, failing);
        let jacobi = species_active(&xs, p, false, MAXVOL_MAX_SWAPS);
        assert_eq!((fell_back.u.len(), &fell_back.active, fell_back.converged), (r, &jacobi.active, true));
        assert_eq!(fell_back.null_threshold, jacobi.null_threshold);
    }

    /// The band rule (module doc): an eigenvalue within δ = √P·ε·max|λ| above τ is null, one above
    /// τ + δ is kept, one at τ or below is null.
    #[test]
    fn eigenvalues_within_rounding_of_tau_are_null() {
        let tau = 1e-10;
        let delta = 6f64.sqrt() * f64::EPSILON * 5.0;
        let ev = [5.0, 1e-3, tau * 0.5, tau, tau + 0.5 * delta, tau + 2.0 * delta];
        let (keep, n_null) = span_split(&ev, tau);
        assert_eq!(keep, vec![0, 1, 5]);
        assert_eq!(n_null, 3);
    }

    /// The threaded Gram and MaxVol scan are the serial loops bit for bit (module doc).
    #[test]
    fn the_threaded_gram_and_scan_are_the_serial_loops() {
        let (n, p, r) = (500, 30, 30);
        let xs = rank_deficient_rows(5, n, p, r);
        let serial_gram = {
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
            g.iter().zip(&c).map(|(a, b)| a + b).collect::<Vec<f64>>()
        };
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for t in [1, 3, crate::threads::worker_threads()] {
            assert_eq!(bits(&crate::threads::with_threads(t, || gram(&xs, p))), bits(&serial_gram), "{t} threads");
        }
        let a_inv: Vec<f64> = rank_deficient_rows(9, r, r, r).concat();
        let mut serial = (0.0f64, 0usize, 0usize);
        for (i, yi) in xs.iter().enumerate() {
            for (j, c) in coefficients(yi, &a_inv, r).into_iter().enumerate() {
                if c.abs() > serial.0 {
                    serial = (c.abs(), i, j);
                }
            }
        }
        for t in [1, 4, crate::threads::worker_threads()] {
            let w = crate::threads::with_threads(t, || worst_coefficient(&xs, &a_inv, r));
            assert_eq!((w.0.to_bits(), w.1, w.2), (serial.0.to_bits(), serial.1, serial.2), "{t} threads");
        }
    }
}
