//! Exact leave-one-sample-out residuals of the fit at fixed hyperparameters (M6 Task 11 round 6:
//! the provider's class-wise trust).
//!
//! # Derivation
//!
//! The fit (`fit` docs) solves, in the standardised reduced coefficient space (Q′ columns),
//!
//!   A θ = b,   A = Σ_rows w_r φ_r φ_rᵀ + λ_eff I,   b = Σ_rows w_r φ_r y_r,
//!
//! with row weight w_r = (sample weight)/σ_e² for a sample's energy row (φ = Σ_i x_i/N, y =
//! (E − Σ E_ref)/N when the fit is per atom) and (sample weight)/σ_f² for each of its 3N force
//! rows (φ = −∂Σ_i x_i/∂r_{j,a}, y = F_{j,a}); A = L Lᵀ is the model's `sigma_inv_chol`. Leave
//! sample s out — its m = 1 + 3N rows Φ_s (m × Q′), weights W_s, targets y_s — at the **same** λ,
//! σ_e, σ_f and standardisation: A₋ = A − Φ_sᵀW_sΦ_s, b₋ = b − Φ_sᵀW_s y_s, θ₋ = A₋⁻¹b₋. By
//! Woodbury, A₋⁻¹ = A⁻¹ + A⁻¹Φ_sᵀ(W_s⁻¹ − Φ_sA⁻¹Φ_sᵀ)⁻¹Φ_sA⁻¹, and with the in-sample residual
//! e_s = y_s − Φ_sθ and the weighted hat block H_ss = Φ_s A⁻¹ Φ_sᵀ W_s (m × m), the residual of
//! the left-out sample under the fit without it is
//!
//!   r_s = y_s − Φ_s θ₋ = (I − H_ss)⁻¹ e_s,
//!
//! the block form of the familiar e_i/(1 − h_ii) (for m = 1, W = w: r = e/(1 − w φᵀA⁻¹φ)).
//! Proof: with P = Φ_sA⁻¹Φ_sᵀ (symmetric), Woodbury gives A₋⁻¹Φ_sᵀ = A⁻¹Φ_sᵀ(I + (W⁻¹ − P)⁻¹P)
//! = A⁻¹Φ_sᵀ(W⁻¹ − P)⁻¹W⁻¹, hence (transposing) Φ_sA₋⁻¹ = W⁻¹(W⁻¹ − P)⁻¹Φ_sA⁻¹, and
//! Φ_sθ₋ = Φ_sA₋⁻¹(b − Φ_sᵀW y_s) = W⁻¹(W⁻¹ − P)⁻¹(Φ_sθ − P W y_s). So
//! y_s − Φ_sθ₋ = W⁻¹(W⁻¹ − P)⁻¹[(W⁻¹ − P)W y_s − Φ_sθ + P W y_s] = W⁻¹(W⁻¹ − P)⁻¹(y_s − Φ_sθ)
//! = (I − P W)⁻¹ e_s = (I − H_ss)⁻¹ e_s. Verified against an explicit downdate of the model's own
//! normal matrix at fixed hyperparameters (`loo_matches_a_downdate_of_the_fit`, 1e-10) — a check
//! of the Woodbury algebra and the rows, not of the row weights w/σ² against the fit's (a
//! mismatch there would cancel on both sides; final review M7).
//!
//! The hyperparameters are not re-estimated without the sample (the evidence's λ and σ move by
//! O(1/n)); that is the standard closed-form LOO, and what "how well would this fit predict a
//! sample it has not seen" needs.
//!
//! **Masked samples** (M8a Task 10; `fit` module doc, "Masked samples"): the left-out block is the
//! rows the sample put into the fit — its energy row only when `use_energy`, the force rows only
//! of its masked-in (interior) atoms — so m = [use_energy] + 3·n_interior. [`SampleLoo`] then has
//! no energy residual (`None`) for a sample whose energy was not fitted and none for a masked-out
//! atom (a cap or an edge atom is not a training point: every consumer — the class and key
//! statistics, the LOO RMS, an audit's re-judgement — skips it).
//!
//! **The forward-solve form** (2026-11-17, the refit speed-up): with A = L Lᵀ, H_ss's
//! (i, j) entry φ_iᵀA⁻¹φ_j w_j = (L⁻¹φ_i)·(L⁻¹φ_j) w_j, so Y = L⁻¹Φ_sᵀ — one forward
//! substitution per row, no back substitution — gives the whole block as w_j y_i·y_j. The m
//! right-hand sides are substituted together, one pass over L's rows for all of them
//! ([`forward_many`]: L's row i is read once and dotted with every y_t, where the earlier form
//! read L twice per row, the second time down its columns), and a row's leading zeros (a species
//! block the sample does not touch) are skipped — exact, they stay zero. The same numbers to
//! rounding as A⁻¹Φᵀ by two solves (`loo_matches_a_downdate_of_the_fit`); each right-hand side's
//! arithmetic is its own, so the batch changes no bit of any of them.
//!
//! Cost per sample: m forward substitutions of size Q′ (O(m Q′²/2), one pass over L), the m × m
//! block O(m²Q′) and its solve O(m³) — for H₂ (m = 7, Q′ = 29) ≈ 10⁴ flops, for an S = 2 sample
//! of 3 atoms (m = 10, Q′ = 176) ≈ 3·10⁵: a store of a few hundred samples costs milliseconds per
//! refit.

use crate::descriptor::{describe, Basis, DescribeError};
use crate::fit::Sample;
use crate::linalg::dot;
use crate::model::Model;

/// A sample's leave-one-out residuals (y − prediction of the fit without it): the energy per
/// atom (Hartree per atom when the fit is per atom, else Hartree) and the forces (Hartree/Bohr) —
/// of the rows the sample was fitted with: `energy` is `None` when its energy was not fitted,
/// `forces[j]` `None` for an atom its force mask leaves out (module doc, "Masked samples").
#[derive(Clone, Debug, PartialEq)]
pub struct SampleLoo {
    pub energy: Option<f64>,
    pub forces: Vec<Option<[f64; 3]>>,
}

/// The standardised reduced design rows of a geometry exactly as `fit::Accumulator::add` builds
/// them: the energy row (÷ N when `per_atom`) and the 3N force rows, each of length Q′.
pub fn design_rows(model: &Model, pos: &[[f64; 3]], z: &[u8]) -> Result<(Vec<f64>, Vec<Vec<f64>>), DescribeError> {
    let basis: &Basis = &model.basis;
    let p = basis.n_features();
    let classes = basis.column_classes();
    let mut col_index = vec![None; basis.n_species() * p];
    for (c, class) in classes.iter().enumerate() {
        col_index[class[0]] = Some(c);
    }
    let d: Vec<f64> = classes.iter().map(|c| { let s = model.scale[c[0] % p]; if s > 0.0 { 1.0 / s } else { 1.0 } }).collect();
    standardised_rows(basis, &col_index, &d, model.energy_per_atom, pos, z)
}

/// The row builder behind [`design_rows`] and `novelty::NullSpace::novelty` (one copy, M6 final
/// review M7: rules 4b and 5d are only right while both see the fit's rows; this one is checked
/// against the fit through the in-sample residual Φθ = prediction): `col_index` maps a full
/// column to its reduced one (`None` for a tied duplicate), `d` is 1/scale per reduced column.
pub(crate) fn standardised_rows(
    basis: &Basis,
    col_index: &[Option<usize>],
    d: &[f64],
    per_atom: bool,
    pos: &[[f64; 3]],
    z: &[u8],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), DescribeError> {
    let p = basis.n_features();
    let q = d.len();
    let s = basis.species_indices(z)?;
    let desc = describe(basis, pos, z)?;
    let n = pos.len();
    let mut e_row = vec![0.0; q];
    let mut f_rows = vec![vec![0.0; q]; 3 * n];
    for (i, di) in desc.iter().enumerate() {
        let off = s[i] * p;
        for k in 0..p {
            if let Some(c) = col_index[off + k] {
                e_row[c] += di.b[k];
            }
        }
        for (j, g) in &di.grad {
            for k in 0..p {
                if let Some(c) = col_index[off + k] {
                    for a in 0..3 {
                        f_rows[3 * j + a][c] -= g[k][a];
                    }
                }
            }
        }
    }
    // the fit's own row (`crate::fit` module doc, "Design matrix"): per N_e
    let per = if per_atom { 1.0 / crate::energy_row_atoms(basis, pos) as f64 } else { 1.0 };
    for (c, v) in e_row.iter_mut().enumerate() {
        *v *= per * d[c];
    }
    for row in f_rows.iter_mut() {
        for (c, v) in row.iter_mut().enumerate() {
            *v *= d[c];
        }
    }
    Ok((e_row, f_rows))
}

/// Y = L⁻¹Φᵀ for the rows `rows` (each of length q), L the lower factor (row-major, `linalg`'s
/// layout): one forward substitution per row, all of them in one pass over L's rows (module doc,
/// "The forward-solve form"). A row's leading zeros are skipped — y stays zero there exactly —
/// and every right-hand side is substituted with its own arithmetic, so a row's result does not
/// depend on the others in the batch.
pub fn forward_many(l: &[f64], q: usize, rows: &[&[f64]]) -> Vec<Vec<f64>> {
    debug_assert_eq!(l.len(), q * q);
    let mut y: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
    let start: Vec<usize> = y.iter().map(|v| v.iter().position(|&x| x != 0.0).unwrap_or(q)).collect();
    let first = start.iter().copied().min().unwrap_or(q);
    for i in first..q {
        let li = &l[i * q..i * q + i];
        let lii = l[i * q + i];
        for (yt, &st) in y.iter_mut().zip(&start) {
            if i < st {
                continue;
            }
            let s = yt[i] - dot(&li[st..], &yt[st..i]);
            yt[i] = s / lii;
        }
    }
    y
}

/// Solve the small dense system M x = v (m × m, row-major) by Gaussian elimination with partial
/// pivoting; `None` if singular.
fn solve_small(mut m: Vec<f64>, mut v: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| m[x * n + col].abs().total_cmp(&m[y * n + col].abs()))?;
        if m[piv * n + col] == 0.0 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                m.swap(piv * n + k, col * n + k);
            }
            v.swap(piv, col);
        }
        for r in col + 1..n {
            let f = m[r * n + col] / m[col * n + col];
            if f != 0.0 {
                for k in col..n {
                    m[r * n + k] -= f * m[col * n + k];
                }
                v[r] -= f * v[col];
            }
        }
    }
    for col in (0..n).rev() {
        let mut s = v[col];
        for k in col + 1..n {
            s -= m[col * n + k] * v[k];
        }
        v[col] = s / m[col * n + col];
    }
    Some(v)
}

/// [`leave_one_out`] of every sample in `samples`, in order, on [`crate::threads::worker_threads`]
/// threads below normal priority (contiguous chunks). Each sample's residuals depend on the model and that sample alone, so the numbers are
/// bit for bit those of one call at a time; only the wall changes (the provider's class-wise
/// trust pass, O(P · m · Q′²): 4.6 ms per sample at Q′ 1 032 on one thread, `references.md`
/// "Refit cost at scale").
pub fn leave_one_out_many(model: &Model, samples: &[&Sample]) -> Vec<Option<SampleLoo>> {
    let _prof = crate::prof::span("loo: leave_one_out_many");
    let threads = crate::threads::worker_threads();
    if threads == 1 || samples.len() < 2 {
        return samples.iter().map(|s| leave_one_out(model, s)).collect();
    }
    let per = samples.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = samples.chunks(per).map(|chunk| {
                scope.spawn(move || {
                    crate::threads::lower_priority();
                    chunk.iter().map(|s| leave_one_out(model, s)).collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("a leave-one-out thread panicked")).collect()
    })
}

/// The leave-one-out residuals of a training sample under `model` (module doc). `None` when the
/// model carries no posterior factor, the sample's species are unknown, or I − H_ss is singular
/// (a sample that alone determines a direction: its LOO prediction there is undefined).
pub fn leave_one_out(model: &Model, sample: &Sample) -> Option<SampleLoo> {
    let l = model.sigma_inv_chol.as_ref()?;
    let _prof_all = crate::prof::span("loo/sample (cpu-s, threads summed): total");
    let _prof_rows = crate::prof::span("loo/sample (cpu-s):   design_rows + predict (descriptors twice)");
    let (e_row, f_rows) = design_rows(model, &sample.pos, &sample.z).ok()?;
    let q = e_row.len();
    let n = sample.pos.len();
    // the rows the sample was fitted with (module doc, "Masked samples")
    let taught: Vec<usize> = (0..n).filter(|&j| sample.trains_atom(j)).collect();
    let ne = usize::from(sample.use_energy);
    let m = ne + 3 * taught.len();
    if m == 0 {
        return None;
    }
    // in-sample residuals y − Φθ, from the model's own predictions (the same rows, expanded)
    let pred = model.predict(&sample.pos, &sample.z, false).ok()?;
    let per = if model.energy_per_atom { 1.0 / crate::energy_row_atoms(&model.basis, &sample.pos) as f64 } else { 1.0 };
    let mut e = Vec::with_capacity(m);
    if sample.use_energy {
        e.push((sample.energy - pred.energy) * per);
    }
    for &j in &taught {
        for a in 0..3 {
            e.push(sample.forces[j][a] - pred.forces[j][a]);
        }
    }
    drop(_prof_rows);
    let _prof_solve = crate::prof::span("loo/sample (cpu-s):   triangular solves + I-H (Q'^2 per row)");
    let w_e = sample.weight / (model.sigma_e * model.sigma_e);
    let w_f = sample.weight / (model.sigma_f * model.sigma_f);
    let rows: Vec<&[f64]> = std::iter::once(&e_row[..])
        .take(ne)
        .chain(taught.iter().flat_map(|&j| (0..3).map(move |a| 3 * j + a)).map(|r| &f_rows[r][..]))
        .collect();
    let w: Vec<f64> = std::iter::once(w_e).take(ne).chain(std::iter::repeat_n(w_f, 3 * taught.len())).collect();
    // Y = L⁻¹Φᵀ, all m rows in one pass over L (module doc, "The forward-solve form")
    let y = forward_many(l, q, &rows);
    // I − H, H_ij = (y_i · y_j) w_j
    let mut ih = vec![0.0; m * m];
    for i in 0..m {
        for jj in i..m {
            let g = dot(&y[i], &y[jj]);
            ih[i * m + jj] = if i == jj { 1.0 - g * w[jj] } else { -g * w[jj] };
            if jj != i {
                ih[jj * m + i] = -g * w[i];
            }
        }
    }
    let r = solve_small(ih, e, m)?;
    let mut forces = vec![None; n];
    for (t, &j) in taught.iter().enumerate() {
        let b = ne + 3 * t;
        forces[j] = Some([r[b], r[b + 1], r[b + 2]]);
    }
    Some(SampleLoo { energy: sample.use_energy.then(|| r[0]), forces })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{Accumulator, FitOptions, SampleTag};
    use crate::model::Provenance;

    /// The fixture: 40 noisy, weighted samples of a pair-potential teacher on 2- and 3-atom
    /// geometries, with the basis, provenance and pinned hyperparameters they are fitted at.
    fn fixture() -> (Basis, Provenance, Vec<Sample>, FitOptions) {
        let mut basis = Basis::new(&[1]);
        basis.n_max = 4;
        let (prov, samples, opts) = fixture_samples(&|_, _| 1);
        (basis, prov, samples, opts)
    }

    /// [`fixture`]'s samples with atom k of sample i of species `z_of(i, k)` (the teacher is the
    /// same species-blind pair potential; its isolated-atom energy −0.5 per atom whatever the
    /// species — a fit's references are the caller's).
    fn fixture_samples(z_of: &dyn Fn(usize, usize) -> u8) -> (Provenance, Vec<Sample>, FitOptions) {
        let prov = Provenance { grid_n: 33, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "pair".into(), masses: vec![], ionic: "none".into() };
        let mut seed = 5u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        // a pair potential teacher with noise, on 2- and 3-atom geometries
        let pair = |r: f64| -> (f64, f64) { let x = (-(r - 1.4f64)).exp(); (0.17 * ((1.0 - x) * (1.0 - x) - 1.0), 0.34 * (1.0 - x) * x) };
        let mut samples = Vec::new();
        for i in 0..40 {
            let n = if i % 3 == 0 { 3 } else { 2 };
            let pos: Vec<[f64; 3]> = (0..n).map(|k| if k == 0 { [0.0; 3] } else { [1.0 + 1.5 * rnd(), 1.5 * rnd() - 0.75, (k as f64 - 1.0) * (1.0 + rnd())] }).collect();
            let mut e = -0.5 * n as f64;
            let mut f = vec![[0.0; 3]; n];
            for a in 0..n {
                for b in a + 1..n {
                    let d: [f64; 3] = std::array::from_fn(|k| pos[b][k] - pos[a][k]);
                    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let (ep, de) = pair(r);
                    e += ep;
                    for k in 0..3 {
                        f[a][k] += de * d[k] / r;
                        f[b][k] -= de * d[k] / r;
                    }
                }
            }
            e += 1e-4 * (rnd() - 0.5);
            for fa in f.iter_mut() {
                for x in fa.iter_mut() {
                    *x += 1e-3 * (rnd() - 0.5);
                }
            }
            // M8a Task 10: every fourth sample is masked as a segment would be (atom 1 not taught,
            // no energy), every fifth keeps its energy with atom 0 masked out
            let force_mask = match i {
                _ if i % 4 == 1 => Some((0..n).map(|k| k != 1).collect()),
                _ if i % 5 == 2 => Some((0..n).map(|k| k != 0).collect()),
                _ => None,
            };
            let use_energy = i % 4 != 1;
            samples.push(Sample { pos, z: (0..n).map(|k| z_of(i, k)).collect(), energy: e, forces: f, tag: SampleTag::Scan, weight: 0.5 + rnd(), force_mask, use_energy });
        }
        // pinned hyperparameters with a moderate ridge: the brute-force side (an explicit A₋ and a
        // dense solve) is only as good as A₋'s conditioning, so the comparison runs where that is
        // moderate (printed) — the closed form itself does not depend on the choice
        let opts = FitOptions { sigma_e: Some(1e-2), sigma_f: Some(3e-2), lambda: Some(1e3), ..FitOptions::default() };
        (prov, samples, opts)
    }

    fn fit(basis: &Basis, prov: &Provenance, samples: &[Sample], opts: &FitOptions) -> (Model, crate::fit::FitReport) {
        fit_with_refs(basis, prov, samples, opts, &[-0.5])
    }

    fn fit_with_refs(basis: &Basis, prov: &Provenance, samples: &[Sample], opts: &FitOptions, refs: &[f64]) -> (Model, crate::fit::FitReport) {
        let mut acc = Accumulator::new(basis);
        for s in samples {
            acc.add(s, basis, refs, opts).unwrap();
        }
        acc.solve(opts, prov, refs).unwrap()
    }

    /// The closed form against an explicit downdate of the fit at its own hyperparameters (not an
    /// independent refit: the weights come from the same model on both sides): A₋ = L Lᵀ −
    /// Φ_sᵀW_sΦ_s, b = A θ (θ the model's standardised coefficients, S = 1: no ties), θ₋ =
    /// A₋⁻¹(b − Φ_sᵀW_s y_s), r = y_s − Φ_sθ₋, for every sample of a small mixed dimer/trimer
    /// store (noisy, weighted rows, so H_ss is far from 0). Masked samples (M8a Task 10) are left
    /// out by their fitted rows only, and report no energy residual without an energy row and no
    /// force residual for a masked-out atom.
    /// The threaded map is the one-at-a-time loop, bit for bit and in order (the provider's class
    /// rule reads it: `references.md` "Refit cost at scale").
    /// The batched forward substitution (module doc, "The forward-solve form"): each right-hand
    /// side's result does not depend on the batch (bit for bit alone or together), leading zeros
    /// stay exactly zero, and L·y = φ to rounding.
    #[test]
    fn forward_many_is_each_rows_own_substitution() {
        let q = 37;
        let mut seed = 3u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut l = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..i {
                l[i * q + j] = rnd();
            }
            l[i * q + i] = 1.0 + rnd().abs();
        }
        let rows: Vec<Vec<f64>> = (0..5).map(|t| (0..q).map(|k| if k < 4 * t { 0.0 } else { rnd() }).collect()).collect();
        let refs: Vec<&[f64]> = rows.iter().map(|r| &r[..]).collect();
        let all = forward_many(&l, q, &refs);
        for (t, r) in refs.iter().enumerate() {
            let alone = forward_many(&l, q, &[r]);
            assert_eq!(alone[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), all[t].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "row {t}");
            assert!(all[t][..4 * t].iter().all(|&v| v == 0.0), "row {t}: leading zeros");
            for i in 0..q {
                let ly: f64 = (0..=i).map(|k| l[i * q + k] * all[t][k]).sum();
                assert!((ly - r[i]).abs() < 1e-13, "row {t}, {i}: L y = {ly} vs {}", r[i]);
            }
        }
    }

    #[test]
    fn leave_one_out_many_is_the_serial_loop_bit_for_bit() {
        let (basis, prov, samples, opts) = fixture();
        let (model, _) = fit(&basis, &prov, &samples, &opts);
        let refs: Vec<&Sample> = samples.iter().collect();
        let many = leave_one_out_many(&model, &refs);
        let serial: Vec<Option<SampleLoo>> = samples.iter().map(|s| leave_one_out(&model, s)).collect();
        assert_eq!(many.len(), serial.len());
        let bits = |l: &Option<SampleLoo>| l.as_ref().map(|l| (l.energy.map(f64::to_bits), l.forces.iter().map(|f| f.map(|f| f.map(f64::to_bits))).collect::<Vec<_>>()));
        for (a, b) in many.iter().zip(&serial) {
            assert_eq!(bits(a), bits(b));
        }
        assert!(serial.iter().all(Option::is_some));
        assert!(leave_one_out_many(&model, &[]).is_empty());
    }

    #[test]
    fn loo_matches_a_downdate_of_the_fit() {
        let (basis, prov, samples, opts) = fixture();
        let (model, rep) = fit(&basis, &prov, &samples, &opts);
        println!("fixture: κ(A) {:.2e}", rep.condition_estimate);
        // S = 1: no tied columns, θ is the stored coefficients
        downdate_check(&model, &samples, &[-0.5], &model.coeffs[0].clone());
    }

    /// The same downdate on a two-species fit whose body-3 block is on one chemical channel (plan
    /// 2026-11-22-chemical-channels Task 4): the closed form is generic in the rows, and its
    /// in-sample residual comes from `predict` (the hot path's channel adjoint). θ per column class
    /// is the fit's (the representative's coefficient × the class size, the equal split undone).
    #[test]
    fn channel_loo_matches_a_downdate_of_the_fit() {
        let mut basis = Basis::new(&[1, 8]).with_channels(crate::descriptor::ChannelMix { k: 1, w: vec![0.8, -1.1] });
        basis.n_max = 4;
        let (prov, samples, opts) = fixture_samples(&|i, k| if (i + k) % 3 == 0 { 8 } else { 1 });
        assert!(samples.iter().any(|s| s.z.contains(&8) && s.z.contains(&1)));
        let refs = [-0.5, -0.5];
        let (model, rep) = fit_with_refs(&basis, &prov, &samples, &opts, &refs);
        println!("channel fixture: Q′ {}, κ(A) {:.2e}", rep.q_prime, rep.condition_estimate);
        let p = basis.n_features();
        let theta: Vec<f64> = basis.column_classes().iter().map(|c| model.coeffs[c[0] / p][c[0] % p] * c.len() as f64).collect();
        downdate_check(&model, &samples, &refs, &theta);
    }

    /// The body of [`loo_matches_a_downdate_of_the_fit`] (its doc) for `model` fitted to `samples`
    /// with references `refs`, θ its standardised reduced coefficients.
    fn downdate_check(model: &Model, samples: &[Sample], refs: &[f64], theta: &[f64]) {
        let basis = &model.basis;
        let q = basis.n_independent_columns();
        assert_eq!(theta.len(), q);
        let l = model.sigma_inv_chol.as_ref().unwrap();
        let mut a = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..q {
                a[i * q + j] = (0..q).map(|k| l[i * q + k] * l[j * q + k]).sum();
            }
        }
        let b: Vec<f64> = (0..q).map(|i| (0..q).map(|j| a[i * q + j] * theta[j]).sum()).collect();
        let (mut worst, mut max_h) = (0.0f64, 0.0f64);
        let mut masked = 0;
        for s in samples {
            let loo = leave_one_out(model, s).expect("loo");
            let ins = model.predict(&s.pos, &s.z, false).unwrap();
            let j0 = (0..s.pos.len()).find(|&j| s.trains_atom(j)).unwrap();
            max_h = max_h.max((loo.forces[j0].unwrap()[0] - (s.forces[j0][0] - ins.forces[j0][0])).abs());
            assert_eq!(loo.energy.is_some(), s.use_energy, "an energy residual exactly when the energy was fitted");
            for (j, f) in loo.forces.iter().enumerate() {
                assert_eq!(f.is_some(), s.trains_atom(j), "a force residual exactly for a fitted atom");
            }
            masked += usize::from(!s.is_whole());
            let (er, fr) = design_rows(model, &s.pos, &s.z).unwrap();
            let n = s.pos.len();
            // the rows the sample was fitted with
            let rows: Vec<Vec<f64>> = std::iter::once(er).filter(|_| s.use_energy).chain(fr.into_iter().enumerate().filter(|(r, _)| s.trains_atom(r / 3)).map(|(_, row)| row)).collect();
            let e_ref: f64 = basis.species_indices(&s.z).unwrap().iter().map(|&t| refs[t]).sum();
            let y: Vec<f64> = std::iter::once((s.energy - e_ref) / n as f64)
                .filter(|_| s.use_energy)
                .chain(s.forces.iter().enumerate().filter(|(j, _)| s.trains_atom(*j)).flat_map(|(_, f)| f.iter().copied()))
                .collect();
            let w: Vec<f64> = std::iter::once(s.weight / model.sigma_e.powi(2)).filter(|_| s.use_energy).chain(std::iter::repeat_n(s.weight / model.sigma_f.powi(2), 3 * s.trained_atoms())).collect();
            let mut am = a.clone();
            let mut bm = b.clone();
            for (r, (row, (&wr, &yr))) in rows.iter().zip(w.iter().zip(&y)).enumerate() {
                let _ = r;
                for i in 0..q {
                    bm[i] -= wr * row[i] * yr;
                    for j in 0..q {
                        am[i * q + j] -= wr * row[i] * row[j];
                    }
                }
            }
            let th = solve_small(am, bm, q).expect("A₋ invertible");
            let brute: Vec<f64> = rows.iter().zip(&y).map(|(row, &yr)| yr - row.iter().zip(&th).map(|(x, t)| x * t).sum::<f64>()).collect();
            let closed: Vec<f64> = loo.energy.into_iter().chain(loo.forces.iter().flatten().flat_map(|f| f.iter().copied())).collect();
            assert_eq!(closed.len(), brute.len());
            // relative to the sample's largest residual component
            let scale = brute.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            for (c, bf) in closed.iter().zip(&brute) {
                worst = worst.max((c - bf).abs() / scale);
            }
        }
        println!("closed-form LOO vs the explicit downdate ({masked} of {} samples masked): worst relative difference {worst:.2e}; largest LOO − in-sample force residual {max_h:.2e} (the hat block matters)", samples.len());
        assert!(masked >= 10);
        assert!(max_h > 1e-4, "test premise: leaving a sample out changes its residual");
        assert!(worst < 1e-10, "{worst:e}");
    }
    /// The LOO residuals of a rigidly rotated store (positions and forces) at the same pinned
    /// hyperparameters: the energy residual unchanged and every force residual rotated, to
    /// rounding (the descriptors are invariant, so the design rows, A and the hat blocks are).
    #[test]
    fn loo_is_rotation_covariant() {
        let (basis, prov, samples, opts) = fixture();
        let (k, th) = ([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0], 0.9f64);
        let (c, sn) = (th.cos(), th.sin());
        let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
        let r: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| (if i == j { c } else { 0.0 }) + sn * kx[i][j] + (1.0 - c) * k[i] * k[j]));
        let rot = |v: &[f64; 3]| -> [f64; 3] { std::array::from_fn(|i| (0..3).map(|j| r[i][j] * v[j]).sum()) };
        let rotated: Vec<Sample> = samples.iter().map(|s| Sample { pos: s.pos.iter().map(rot).collect(), forces: s.forces.iter().map(rot).collect(), ..s.clone() }).collect();
        let (m0, _) = fit(&basis, &prov, &samples, &opts);
        let (m1, _) = fit(&basis, &prov, &rotated, &opts);
        let mut worst = 0.0f64;
        for (s0, s1) in samples.iter().zip(&rotated) {
            let (l0, l1) = (leave_one_out(&m0, s0).unwrap(), leave_one_out(&m1, s1).unwrap());
            let scale = l0.energy.iter().map(|e| e.abs()).chain(l0.forces.iter().flatten().flatten().map(|x| x.abs())).fold(0.0f64, f64::max);
            assert_eq!(l0.energy.is_some(), l1.energy.is_some());
            if let (Some(e0), Some(e1)) = (l0.energy, l1.energy) {
                worst = worst.max((e0 - e1).abs() / scale);
            }
            for (f0, f1) in l0.forces.iter().zip(&l1.forces) {
                let (Some(f0), Some(f1)) = (f0, f1) else {
                    assert_eq!(f0.is_some(), f1.is_some());
                    continue;
                };
                let rf0 = rot(f0);
                for a in 0..3 {
                    worst = worst.max((rf0[a] - f1[a]).abs() / scale);
                }
            }
        }
        println!("rotated store: worst relative LOO difference {worst:.2e}");
        assert!(worst < 1e-9, "{worst:e}");
    }
}
