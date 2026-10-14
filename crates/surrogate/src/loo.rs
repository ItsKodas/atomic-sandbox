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
//! = (I − P W)⁻¹ e_s = (I − H_ss)⁻¹ e_s. Verified against a brute-force refit at fixed
//! hyperparameters (`loo_matches_a_brute_force_refit`, 1e-10).
//!
//! The hyperparameters are not re-estimated without the sample (the evidence's λ and σ move by
//! O(1/n)); that is the standard closed-form LOO, and what "how well would this fit predict a
//! sample it has not seen" needs. Cost per sample: m triangular solve pairs of size Q′ (A⁻¹Φ_sᵀ,
//! O(m Q′²)), the m × m block and its solve O(m²Q′ + m³) — for H₂ (m = 7, Q′ = 29) ≈ 2·10⁴
//! flops, for an S = 2 sample of 3 atoms (m = 10, Q′ = 176) ≈ 6·10⁵: a store of a few hundred
//! samples costs milliseconds per refit.

use crate::descriptor::{describe, Basis, DescribeError};
use crate::fit::Sample;
use crate::linalg::cholesky_solve;
use crate::model::Model;

/// A sample's leave-one-out residuals (y − prediction of the fit without it): the energy per
/// atom (Hartree per atom when the fit is per atom, else Hartree) and the forces (Hartree/Bohr).
#[derive(Clone, Debug, PartialEq)]
pub struct SampleLoo {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
}

/// The standardised reduced design rows of a geometry exactly as `fit::Accumulator::add` builds
/// them: the energy row (÷ N when `per_atom`) and the 3N force rows, each of length Q′.
pub fn design_rows(model: &Model, pos: &[[f64; 3]], z: &[u8]) -> Result<(Vec<f64>, Vec<Vec<f64>>), DescribeError> {
    let basis: &Basis = &model.basis;
    let p = basis.n_features();
    let classes = basis.column_classes();
    let q = classes.len();
    let mut col_index = vec![None; basis.n_species() * p];
    for (c, class) in classes.iter().enumerate() {
        col_index[class[0]] = Some(c);
    }
    let d: Vec<f64> = classes.iter().map(|c| { let s = model.scale[c[0] % p]; if s > 0.0 { 1.0 / s } else { 1.0 } }).collect();
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
    let per = if model.energy_per_atom { 1.0 / n as f64 } else { 1.0 };
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

/// The leave-one-out residuals of a training sample under `model` (module doc). `None` when the
/// model carries no posterior factor, the sample's species are unknown, or I − H_ss is singular
/// (a sample that alone determines a direction: its LOO prediction there is undefined).
pub fn leave_one_out(model: &Model, sample: &Sample) -> Option<SampleLoo> {
    let l = model.sigma_inv_chol.as_ref()?;
    let (e_row, f_rows) = design_rows(model, &sample.pos, &sample.z).ok()?;
    let q = e_row.len();
    let n = sample.pos.len();
    let m = 1 + 3 * n;
    // in-sample residuals y − Φθ, from the model's own predictions (the same rows, expanded)
    let pred = model.predict(&sample.pos, &sample.z, false).ok()?;
    let per = if model.energy_per_atom { 1.0 / n as f64 } else { 1.0 };
    let mut e = Vec::with_capacity(m);
    e.push((sample.energy - pred.energy) * per);
    for j in 0..n {
        for a in 0..3 {
            e.push(sample.forces[j][a] - pred.forces[j][a]);
        }
    }
    let w_e = sample.weight / (model.sigma_e * model.sigma_e);
    let w_f = sample.weight / (model.sigma_f * model.sigma_f);
    let rows: Vec<&[f64]> = std::iter::once(&e_row[..]).chain(f_rows.iter().map(|r| &r[..])).collect();
    let w: Vec<f64> = std::iter::once(w_e).chain(std::iter::repeat_n(w_f, 3 * n)).collect();
    // Z = A⁻¹Φᵀ, column by column
    let z: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let mut v = r.to_vec();
            cholesky_solve(l, q, &mut v);
            v
        })
        .collect();
    // I − H, H_ij = (φ_i · z_j) w_j
    let mut ih = vec![0.0; m * m];
    for i in 0..m {
        for jj in 0..m {
            let h: f64 = rows[i].iter().zip(&z[jj]).map(|(a, b)| a * b).sum::<f64>() * w[jj];
            ih[i * m + jj] = if i == jj { 1.0 - h } else { -h };
        }
    }
    let r = solve_small(ih, e, m)?;
    Some(SampleLoo { energy: r[0], forces: (0..n).map(|j| [r[1 + 3 * j], r[2 + 3 * j], r[3 + 3 * j]]).collect() })
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
        let prov = Provenance { grid_n: 33, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "pair".into(), masses: vec![] };
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
            samples.push(Sample { pos, z: vec![1; n], energy: e, forces: f, tag: SampleTag::Scan, weight: 0.5 + rnd() });
        }
        // pinned hyperparameters with a moderate ridge: the brute-force side (an explicit A₋ and a
        // dense solve) is only as good as A₋'s conditioning, so the comparison runs where that is
        // moderate (printed) — the closed form itself does not depend on the choice
        let opts = FitOptions { sigma_e: Some(1e-2), sigma_f: Some(3e-2), lambda: Some(1e3), ..FitOptions::default() };
        (basis, prov, samples, opts)
    }

    fn fit(basis: &Basis, prov: &Provenance, samples: &[Sample], opts: &FitOptions) -> (Model, crate::fit::FitReport) {
        let mut acc = Accumulator::new(basis);
        for s in samples {
            acc.add(s, basis, &[-0.5], opts).unwrap();
        }
        acc.solve(opts, prov, &[-0.5]).unwrap()
    }

    /// The closed form against a brute-force refit at the fit's own hyperparameters: A₋ = L Lᵀ −
    /// Φ_sᵀW_sΦ_s, b = A θ (θ the model's standardised coefficients, S = 1: no ties), θ₋ =
    /// A₋⁻¹(b − Φ_sᵀW_s y_s), r = y_s − Φ_sθ₋, for every sample of a small mixed dimer/trimer
    /// store (noisy, weighted rows, so H_ss is far from 0).
    #[test]
    fn loo_matches_a_brute_force_refit() {
        let (basis, prov, samples, opts) = fixture();
        let (model, rep) = fit(&basis, &prov, &samples, &opts);
        println!("fixture: κ(A) {:.2e}", rep.condition_estimate);
        let q = basis.n_independent_columns();
        let l = model.sigma_inv_chol.as_ref().unwrap();
        let mut a = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..q {
                a[i * q + j] = (0..q).map(|k| l[i * q + k] * l[j * q + k]).sum();
            }
        }
        let theta: Vec<f64> = model.coeffs[0].clone();
        let b: Vec<f64> = (0..q).map(|i| (0..q).map(|j| a[i * q + j] * theta[j]).sum()).collect();
        let (mut worst, mut max_h) = (0.0f64, 0.0f64);
        for s in &samples {
            let loo = leave_one_out(&model, s).expect("loo");
            let ins = model.predict(&s.pos, &s.z, false).unwrap();
            max_h = max_h.max((loo.forces[0][0] - (s.forces[0][0] - ins.forces[0][0])).abs());
            let (er, fr) = design_rows(&model, &s.pos, &s.z).unwrap();
            let n = s.pos.len();
            let rows: Vec<Vec<f64>> = std::iter::once(er).chain(fr).collect();
            let y: Vec<f64> = std::iter::once((s.energy + 0.5 * n as f64) / n as f64).chain(s.forces.iter().flat_map(|f| f.iter().copied())).collect();
            let w: Vec<f64> = std::iter::once(s.weight / model.sigma_e.powi(2)).chain(std::iter::repeat_n(s.weight / model.sigma_f.powi(2), 3 * n)).collect();
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
            let closed: Vec<f64> = std::iter::once(loo.energy).chain(loo.forces.iter().flat_map(|f| f.iter().copied())).collect();
            // relative to the sample's largest residual component
            let scale = brute.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            for (c, bf) in closed.iter().zip(&brute) {
                worst = worst.max((c - bf).abs() / scale);
            }
        }
        println!("closed-form LOO vs brute-force refit: worst relative difference {worst:.2e}; largest LOO − in-sample force residual {max_h:.2e} (the hat block matters)");
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
            let scale = std::iter::once(l0.energy.abs()).chain(l0.forces.iter().flatten().map(|x| x.abs())).fold(0.0f64, f64::max);
            worst = worst.max((l0.energy - l1.energy).abs() / scale);
            for (f0, f1) in l0.forces.iter().zip(&l1.forces) {
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
