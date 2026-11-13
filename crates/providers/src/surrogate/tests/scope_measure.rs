//! Scoped-trust study, step 3 (2026-11-13; the study's "Rule 3: measured counterfactual"): a
//! **measurement only**. Nothing here is called by the provider and no trust behaviour changes;
//! the block downdate below lives in this test module on purpose.
//!
//! # Derivation (atomic units; the standardised reduced coefficient space of `surrogate::loo`)
//!
//! The fit solves A θ = b with A = Σ_r w_r φ_r φ_rᵀ + λ_eff I = L Lᵀ (`Model::sigma_inv_chol`),
//! b = Σ_r w_r φ_r y_r; a sample's rows are those `surrogate::loo::design_rows` builds (its energy
//! row only when `use_energy`, the force rows of its trained atoms only), with w = weight/σ_e² on
//! the energy row and weight/σ_f² on each force row. Remove a **block** S of samples — m rows
//! Φ_S (m × Q′), weights W_S (diagonal), targets y_S — at the same λ_eff, σ_e, σ_f and
//! standardisation (the hyperparameters are not re-estimated): A₋ = A − Φ_SᵀW_SΦ_S,
//! b₋ = b − Φ_SᵀW_S y_S, A₋θ₋ = b₋. Then
//!
//!   A₋(θ₋ − θ) = b₋ − A₋θ = −Φ_SᵀW_S(y_S − Φ_Sθ) = −Φ_SᵀW_S e_S,
//!
//! with e_S the in-sample residuals of S under the standing fit, so θ₋ = θ − Δc_S and
//!
//!   Δc_S = A₋⁻¹Φ_SᵀW_S e_S = A⁻¹Φ_SᵀW_S (I − H_SS)⁻¹ e_S,   H_SS = Φ_S A⁻¹ Φ_Sᵀ W_S
//!
//! (Woodbury as in `loo.rs`: A₋⁻¹Φᵀ = A⁻¹Φᵀ(W⁻¹ − P)⁻¹W⁻¹ with P = ΦA⁻¹Φᵀ, and
//! (W⁻¹ − P)⁻¹ = W(I − PW)⁻¹). It is computed in the symmetric form: with Ẑ = L⁻¹Φ_SᵀW^½ (Q′ × m),
//! G = I − ẐᵀẐ = W^½(W⁻¹ − P)W^½ is symmetric and positive definite exactly when A₋ is (its
//! eigenvalues are 1 − the block's leverages, in (0, 1]), and
//!
//!   Δc_S = A⁻¹ Φ_Sᵀ W^½ G⁻¹ W^½ e_S
//!
//! — m forward substitutions of size Q′, one m × m Cholesky, one Q′ solve. A failed Cholesky of G
//! means S alone spans a direction of the fit (the block analogue of ruling R1a): undefined.
//! For S = {s} it is the leave-one-out's algebra, y_s − Φ_sθ₋ = e_s + Φ_sΔc_s = (I − H_ss)⁻¹e_s
//! (checked below against `surrogate::loo::leave_one_out`).
//!
//! **The prediction shift.** The standing fit predicts Φθ (the energy row plus Σ E_ref), so the fit
//! without S predicts, at any geometry with rows Φ, F_j − Φ_{F,j}Δc_S (Ha/Bohr) and
//! E/N − φ_E·Δc_S (Ha/atom, per-atom fit: φ_E already carries 1/N; ÷ N for a whole-energy fit):
//! ΔF_j = −Φ_{F,j}Δc_S, ΔE/N = −φ_E·Δc_S. In bound units, as the trust checks read a force
//! residual (`FallbackPolicy::force_residual`: ‖·‖/√3, the per-component equivalent, against
//! b_j = `force_bound`(|F_j teacher|)): s_F = max_j ‖ΔF_j‖/√3 / b_j, s_E = |ΔE|/N /
//! `energy_std_per_atom`. Gate (ii) at ε passes when both are ≤ ε. (The raw Euclidean ‖ΔF_j‖/b_j
//! is √3 larger; the measurement prints the gate count under it as well.)
//!
//! **The suspect set** (the study's S): the standing fit's members (the accumulated prefix, not
//! excluded) failing in-sample — the energy |E_model − E|/N over `energy_std_per_atom` (energy
//! fitted) or any trained atom's force residual over its bound above 1, as
//! `compute_training_residuals` reads them — plus the leave-one-out offenders by the recovery's
//! rule (`worst_offenders_bound`): the own-bound leave-one-out ratio ρ (`loo_ratio`) > 1 and
//! ≥ ½ of the worst, over members not excluded and not box-affected. While the residuals fail
//! that is exactly `worst_offenders()` (asserted); on a fit whose residuals pass, the code's
//! ranking switches to the absolute tier ρ_σ, and this measurement keeps ρ (Rule 3 scopes the
//! training-residual checks; printed either way).
use super::*;
use super::super::quarantine::origin_word;
use super::super::shared::formula;
use std::time::Instant;
use surrogate::linalg::{axpy, cholesky_in_place_fast, cholesky_solve, dot, forward_substitute};
use surrogate::loo::design_rows;

/// `f` over `items` on up to `threads` threads (contiguous chunks), results in order.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.min(std::thread::available_parallelism().map_or(1, |n| n.get())).max(1);
    if threads == 1 || items.len() < 2 {
        return items.iter().map(&f).collect();
    }
    let per = items.len().div_ceil(threads);
    let f = &f;
    std::thread::scope(|sc| {
        let hs: Vec<_> = items.chunks(per).map(|c| sc.spawn(move || c.iter().map(f).collect::<Vec<_>>())).collect();
        hs.into_iter().flat_map(|h| h.join().expect("a worker thread panicked")).collect()
    })
}

/// A sample's fitted rows under `model`, in `loo.rs`'s order: (rows, weights, targets y,
/// in-sample residuals e = y − Φθ from the model's own prediction).
struct Block {
    rows: Vec<Vec<f64>>,
    w: Vec<f64>,
    y: Vec<f64>,
    e: Vec<f64>,
}

fn fitted_block(model: &Model, s: &Sample) -> Option<Block> {
    let (e_row, f_rows) = design_rows(model, &s.pos, &s.z).ok()?;
    let n = s.pos.len();
    let per = if model.energy_per_atom { 1.0 / n as f64 } else { 1.0 };
    let pred = model.predict(&s.pos, &s.z, false).ok()?;
    let sp = model.basis.species_indices(&s.z).ok()?;
    let e_ref: f64 = sp.iter().map(|&k| model.ref_energy[k]).sum();
    let (w_e, w_f) = (s.weight / (model.sigma_e * model.sigma_e), s.weight / (model.sigma_f * model.sigma_f));
    let mut b = Block { rows: Vec::new(), w: Vec::new(), y: Vec::new(), e: Vec::new() };
    if s.use_energy {
        b.rows.push(e_row);
        b.w.push(w_e);
        b.y.push((s.energy - e_ref) * per);
        b.e.push((s.energy - pred.energy) * per);
    }
    for j in (0..n).filter(|&j| s.trains_atom(j)) {
        for a in 0..3 {
            b.rows.push(f_rows[3 * j + a].clone());
            b.w.push(w_f);
            b.y.push(s.forces[j][a]);
            b.e.push(s.forces[j][a] - pred.forces[j][a]);
        }
    }
    Some(b)
}

/// The block downdate Δc_S (module doc): θ₋ = θ − `dc`. `m` the block's rows, `min_pivot` the
/// smallest Cholesky pivot of G (1 − the largest leverage direction, roughly).
struct Downdate {
    dc: Vec<f64>,
    m: usize,
    min_pivot: f64,
}

fn block_downdate(model: &Model, samples: &[&Sample]) -> Result<Downdate, String> {
    let l = model.sigma_inv_chol.as_ref().ok_or("the model has no posterior factor")?;
    let q = model.basis.n_independent_columns();
    let blocks = par_map(samples, 16, |s| fitted_block(model, s));
    let (mut rows, mut sw, mut e) = (Vec::new(), Vec::new(), Vec::new());
    for b in blocks {
        let b = b.ok_or("a sample's rows could not be built")?;
        sw.extend(b.w.iter().map(|w| w.sqrt()));
        rows.extend(b.rows);
        e.extend(b.e);
    }
    let m = rows.len();
    if m == 0 {
        return Ok(Downdate { dc: vec![0.0; q], m, min_pivot: 1.0 });
    }
    let idx: Vec<usize> = (0..m).collect();
    // Ẑ = L⁻¹ Φᵀ W^½, column by column
    let zh: Vec<Vec<f64>> = par_map(&idx, 16, |&r| {
        let mut z = rows[r].clone();
        forward_substitute(l, q, &mut z);
        z.iter_mut().for_each(|x| *x *= sw[r]);
        z
    });
    // G = I − ẐᵀẐ (full rows: balanced over the threads)
    let g_rows: Vec<Vec<f64>> = par_map(&idx, 16, |&i| (0..m).map(|j| f64::from(u8::from(i == j)) - dot(&zh[i], &zh[j])).collect());
    let mut g: Vec<f64> = g_rows.into_iter().flatten().collect();
    cholesky_in_place_fast(&mut g, m).map_err(|err| format!("G = I − H is not positive definite ({err}): the block alone spans a direction of the fit"))?;
    let min_pivot = (0..m).map(|j| g[j * m + j] * g[j * m + j]).fold(f64::INFINITY, f64::min);
    let mut u: Vec<f64> = e.iter().zip(&sw).map(|(e, s)| e * s).collect();
    cholesky_solve(&g, m, &mut u);
    let mut v = vec![0.0; q];
    for r in 0..m {
        axpy(sw[r] * u[r], &rows[r], &mut v);
    }
    cholesky_solve(l, q, &mut v);
    Ok(Downdate { dc: v, m, min_pivot })
}

/// The prediction shift of the fit without S at a sample's geometry (module doc): (|ΔE|/N in
/// Ha/atom, ΔF_j per atom in Ha/Bohr; the downdated prediction is the standing one plus these).
fn shift(model: &Model, dc: &[f64], pos: &[[f64; 3]], z: &[u8]) -> Option<(f64, Vec<[f64; 3]>)> {
    let (e_row, f_rows) = design_rows(model, pos, z).ok()?;
    let n = pos.len();
    let de = -dot(&e_row, dc) * if model.energy_per_atom { 1.0 } else { 1.0 / n as f64 };
    let df = (0..n).map(|j| std::array::from_fn(|a| -dot(&f_rows[3 * j + a], dc))).collect();
    Some((de, df))
}

/// An explicit refit at the model's own hyperparameters and standardisation (the oracle of the
/// downdate): A = Σ w φφᵀ (+ λ_eff I) built row by row from `fitted_block`, separately over the
/// kept samples and over S, then θ_full = (A_kept + A_S + λI)⁻¹(b_kept + b_S) and
/// θ_rest = (A_kept + λI)⁻¹ b_kept, with the fit's λ_eff = max(λ, 4 Q′ ε max_i A_ii); `a_dev` is
/// max_ij |L Lᵀ − A_all − λ_eff I| / max_ij |L Lᵀ| (the rows, weights and ridge against the fit's
/// normal matrix), `lambda_spread` max_i |(L Lᵀ − A_all)_ii − λ_eff| (absolute).
struct Explicit {
    theta_full: Vec<f64>,
    theta_rest: Vec<f64>,
    lambda_eff: f64,
    lambda_spread: f64,
    a_dev: f64,
}

fn explicit_refit(model: &Model, kept: &[&Sample], s: &[&Sample]) -> Explicit {
    let q = model.basis.n_independent_columns();
    let l = model.sigma_inv_chol.as_ref().unwrap();
    // per-thread normal equations (≤ 8 threads: two Q′² matrices each)
    let gram = |samples: &[&Sample]| -> (Vec<f64>, Vec<f64>) {
        let threads = 8usize.min(std::thread::available_parallelism().map_or(1, |n| n.get())).max(1);
        let per = samples.len().div_ceil(threads).max(1);
        let parts: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = samples
                .chunks(per)
                .map(|chunk| {
                    sc.spawn(move || {
                        let (mut a, mut b) = (vec![0.0; q * q], vec![0.0; q]);
                        for smp in chunk {
                            let blk = fitted_block(model, smp).expect("rows");
                            for ((row, &w), &y) in blk.rows.iter().zip(&blk.w).zip(&blk.y) {
                                axpy(w * y, row, &mut b);
                                for i in 0..q {
                                    let wi = w * row[i];
                                    if wi != 0.0 {
                                        axpy(wi, row, &mut a[i * q..(i + 1) * q]);
                                    }
                                }
                            }
                        }
                        (a, b)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (mut a, mut b) = (vec![0.0; q * q], vec![0.0; q]);
        for (pa, pb) in parts {
            axpy(1.0, &pa, &mut a);
            axpy(1.0, &pb, &mut b);
        }
        (a, b)
    };
    let (a_kept, b_kept) = gram(kept);
    let (a_s, b_s) = gram(s);
    let mut a_all = a_kept.clone();
    axpy(1.0, &a_s, &mut a_all);
    let mut b_all = b_kept.clone();
    axpy(1.0, &b_s, &mut b_all);
    // L Lᵀ, row by row
    let idx: Vec<usize> = (0..q).collect();
    let llt: Vec<Vec<f64>> = par_map(&idx, 16, |&i| (0..q).map(|j| { let k = i.min(j) + 1; dot(&l[i * q..i * q + k], &l[j * q..j * q + k]) }).collect());
    // the fit's own ridge (`fit::Problem::solve`): λ_eff = max(λ, 4 Q′ ε max_i A_ii), read back
    // against the factor's diagonal (cancellation-limited at ≈ ε max A_ii: printed, not used)
    let max_diag = (0..q).map(|i| a_all[i * q + i]).fold(0.0, f64::max);
    let lambda_eff = model.lambda.max(4.0 * q.max(1) as f64 * f64::EPSILON * max_diag);
    let lambda_spread = (0..q).map(|i| (llt[i][i] - a_all[i * q + i] - lambda_eff).abs()).fold(0.0, f64::max);
    let scale = llt.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut a_dev = 0.0f64;
    for i in 0..q {
        for j in 0..q {
            let lam = if i == j { lambda_eff } else { 0.0 };
            a_dev = a_dev.max((llt[i][j] - a_all[i * q + j] - lam).abs() / scale);
        }
    }
    let solve = |mut a: Vec<f64>, mut b: Vec<f64>| -> Vec<f64> {
        for i in 0..q {
            a[i * q + i] += lambda_eff;
        }
        cholesky_in_place_fast(&mut a, q).expect("the explicit normal matrix is positive definite");
        cholesky_solve(&a, q, &mut b);
        b
    };
    Explicit { theta_full: solve(a_all, b_all), theta_rest: solve(a_kept, b_kept), lambda_eff, lambda_spread, a_dev }
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// The unit test's fixture: 48 weighted, noisy samples of a pair-potential teacher on H and O
/// (H₂, H₃, OH, H₂O-like and H₃O-like geometries, so the body-2 H–O columns are tied across the
/// two centre species), a quarter of them masked as segments are, fitted at pinned hyperparameters
/// with a moderate ridge (the explicit side is only as good as A₋'s conditioning, printed).
fn fixture() -> (Model, Vec<Sample>) {
    let mut basis = Basis::new(&[1, 8]);
    basis.n_max = 4;
    let prov = Provenance { grid_n: 33, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "pair".into(), masses: vec![], ionic: "none".into() };
    let mut rng = StdRng::seed_from_u64(1803);
    let pair = |r: f64, k: f64| -> (f64, f64) { let x = (-(r - 1.4f64)).exp(); (k * ((1.0 - x) * (1.0 - x) - 1.0), 2.0 * k * (1.0 - x) * x) };
    let refs = [-0.5, -15.8];
    let mut samples = Vec::new();
    for i in 0..48 {
        let z: Vec<u8> = match i % 4 {
            0 => vec![1, 1],
            1 => vec![1, 1, 1],
            2 => vec![8, 1],
            _ => if i % 8 == 3 { vec![8, 1, 1] } else { vec![8, 1, 1, 1] },
        };
        let n = z.len();
        let pos: Vec<[f64; 3]> = (0..n).map(|k| if k == 0 { [0.0; 3] } else { [rng.gen_range(-1.8..1.8), rng.gen_range(-1.8..1.8), rng.gen_range(0.9..1.9) * if k % 2 == 0 { -1.0 } else { 1.0 }] }).collect();
        let mut e: f64 = z.iter().map(|&zi| if zi == 1 { refs[0] } else { refs[1] }).sum();
        let mut f = vec![[0.0; 3]; n];
        for a in 0..n {
            for b in a + 1..n {
                let d: [f64; 3] = std::array::from_fn(|k| pos[b][k] - pos[a][k]);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let (ep, de) = pair(r, if z[a] == z[b] { 0.17 } else { 0.25 });
                e += ep;
                for k in 0..3 {
                    f[a][k] += de * d[k] / r;
                    f[b][k] -= de * d[k] / r;
                }
            }
        }
        e += 1e-4 * rng.gen_range(-0.5..0.5);
        f.iter_mut().flatten().for_each(|x| *x += 1e-3 * rng.gen_range(-0.5..0.5));
        let force_mask = match i {
            _ if i % 4 == 1 && i % 3 == 0 => Some((0..n).map(|k| k != 1).collect()),
            _ if i % 5 == 2 => Some((0..n).map(|k| k != 0).collect()),
            _ => None,
        };
        let use_energy = !(i % 4 == 1 && i % 3 == 0) && i % 7 != 6;
        samples.push(Sample { pos, z, energy: e, forces: f, tag: SampleTag::Scan, weight: rng.gen_range(0.5..1.5), force_mask, use_energy });
    }
    let opts = FitOptions { sigma_e: Some(1e-2), sigma_f: Some(3e-2), lambda: Some(1e3), ..FitOptions::default() };
    let mut acc = Accumulator::new(&basis);
    for s in &samples {
        acc.add(s, &basis, &refs, &opts).unwrap();
    }
    let (model, rep) = acc.solve(&opts, &prov, &refs).unwrap();
    println!("fixture: Q′ {}, κ(A) {:.2e}, λ {:e}", rep.q_prime, rep.condition_estimate, model.lambda);
    (model, samples)
}

/// The block downdate against an explicit refit without S at the same λ, σ_e, σ_f and
/// standardisation (module doc), on a mixed H/O store with tied columns, masked samples and a
/// sample without its energy row in S; and for S = {s}, the leave-one-out's residual.
#[test]
fn the_block_downdate_is_an_explicit_refit_without_the_block() {
    let (model, samples) = fixture();
    let q = model.basis.n_independent_columns();
    // S: H₂ #0, OH #2 (atom 0 masked), H₂O-like #3, H₃ #5, H₃O-like #7 (atom 0 masked), H₃ #9
    // (atom 1 masked, no energy row), H₃ #13 (no energy row)
    let in_s = |i: usize| [0usize, 2, 3, 5, 7, 9, 13].contains(&i);
    assert!(samples.iter().enumerate().any(|(i, s)| in_s(i) && !s.use_energy), "the premise: S holds a sample without its energy row");
    assert!(samples.iter().enumerate().any(|(i, s)| in_s(i) && s.force_mask.is_some()), "the premise: S holds a masked sample");
    let s_set: Vec<&Sample> = samples.iter().enumerate().filter(|(i, _)| in_s(*i)).map(|(_, s)| s).collect();
    let kept: Vec<&Sample> = samples.iter().enumerate().filter(|(i, _)| !in_s(*i)).map(|(_, s)| s).collect();
    let d = block_downdate(&model, &s_set).expect("defined");
    let ex = explicit_refit(&model, &kept, &s_set);
    println!("explicit: λ_eff {:e} (model λ {:e}, spread {:.1e}); |L Lᵀ − A − λI| / |L Lᵀ| {:.2e}; m {}, min pivot of G {:.3}", ex.lambda_eff, model.lambda, ex.lambda_spread, ex.a_dev, d.m, d.min_pivot);
    assert!(ex.a_dev < 1e-12, "the rows and weights are the fit's: {:e}", ex.a_dev);
    assert_eq!(ex.lambda_eff, model.lambda, "the premise: λ above its floor");
    assert!(ex.lambda_spread < 1e-9 * model.lambda, "the factor's diagonal carries that λ: {:e}", ex.lambda_spread);
    // the explicit full fit predicts what the model predicts (the targets and references)
    let mut pred_dev = 0.0f64;
    for s in &samples {
        let p = model.predict(&s.pos, &s.z, false).unwrap();
        let (er, fr) = design_rows(&model, &s.pos, &s.z).unwrap();
        let sp = model.basis.species_indices(&s.z).unwrap();
        let n = s.pos.len() as f64;
        let e_ref: f64 = sp.iter().map(|&k| model.ref_energy[k]).sum();
        pred_dev = pred_dev.max(((p.energy - e_ref) / n - dot(&er, &ex.theta_full)).abs());
        for (r, row) in fr.iter().enumerate() {
            pred_dev = pred_dev.max((p.forces[r / 3][r % 3] - dot(row, &ex.theta_full)).abs());
        }
    }
    // Δc against θ_full − θ_rest, and θ_rest against θ_full − Δc
    let dc_ex: Vec<f64> = ex.theta_full.iter().zip(&ex.theta_rest).map(|(a, b)| a - b).collect();
    let dc_err = max_abs(&d.dc.iter().zip(&dc_ex).map(|(a, b)| a - b).collect::<Vec<_>>()) / max_abs(&dc_ex);
    let th_err = max_abs(&(0..q).map(|k| ex.theta_full[k] - d.dc[k] - ex.theta_rest[k]).collect::<Vec<_>>()) / max_abs(&ex.theta_rest);
    println!("explicit full fit vs the model's predictions: {pred_dev:.2e} (Ha/atom, Ha/Bohr); |Δc| {:.2e}; Δc vs θ_full − θ_rest {dc_err:.2e} (relative); θ_full − Δc vs θ_rest {th_err:.2e} (relative)", max_abs(&dc_ex));
    assert!(pred_dev < 1e-10, "{pred_dev:e}");
    assert!(max_abs(&dc_ex) > 1e-3 * max_abs(&ex.theta_rest), "the premise: removing S moves the fit");
    assert!(dc_err < 1e-10, "{dc_err:e}");
    assert!(th_err < 1e-10, "{th_err:e}");
    // S = {s}: e_s + Φ_sΔc_s is the leave-one-out residual of `surrogate::loo`
    let mut loo_err = 0.0f64;
    for k in [1usize, 2, 13, 30] {
        let s = &samples[k];
        let d = block_downdate(&model, &[s]).unwrap();
        let blk = fitted_block(&model, s).unwrap();
        let mine: Vec<f64> = blk.rows.iter().zip(&blk.e).map(|(row, e)| e + dot(row, &d.dc)).collect();
        let l = surrogate::loo::leave_one_out(&model, s).unwrap();
        let theirs: Vec<f64> = l.energy.into_iter().chain(l.forces.iter().flatten().flat_map(|f| f.iter().copied())).collect();
        assert_eq!(mine.len(), theirs.len());
        let scale = max_abs(&theirs);
        loo_err = loo_err.max(max_abs(&mine.iter().zip(&theirs).map(|(a, b)| a - b).collect::<Vec<_>>()) / scale);
    }
    println!("S = {{s}}: vs surrogate::loo::leave_one_out {loo_err:.2e} (relative)");
    assert!(loo_err < 1e-10, "{loo_err:e}");
}

/// One stored sample's numbers under a downdate.
struct SampleRow {
    idx: usize,
    formula: String,
    member: bool,
    in_s: bool,
    /// max_j ‖ΔF_j‖/√3 / b_j over all atoms, and the raw ‖ΔF_j‖ / b_j.
    shift_f: f64,
    shift_f_raw: f64,
    /// |ΔE|/N / energy_std_per_atom.
    shift_e: f64,
    /// A member's in-sample worst ratio before (energy, force) and under the downdate (energy per
    /// atom in Ha/atom, force ratio, its d and b and atom).
    before_e: f64,
    before_f: f64,
    after_e: f64,
    after_f: (f64, f64, f64, usize),
}

fn quantiles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "—".into();
    }
    v.sort_by(f64::total_cmp);
    let at = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    format!("min {:.3e} / median {:.3e} / p90 {:.3e} / max {:.3e}", v[0], at(0.5), at(0.9), v[v.len() - 1])
}

/// The measurement on the fit that stands in `s` (`label`); returns |S|. `loo_only`: a contrast
/// outside the study's definition — S is the leave-one-out offenders alone (on the pre-1805 entry
/// fit: #1804 alone), to separate the culprit's own leak from its victims'.
fn measure(s: &mut SurrogateKs, label: &str, loo_only: bool) -> usize {
    let t_total = Instant::now();
    let (policy, tier) = (s.policy.clone(), s.policy.energy_std_per_atom);
    let n_store = s.samples();
    let members: Vec<usize> = (0..s.accumulated.min(n_store)).filter(|&i| !s.excluded(i)).collect();
    let r = s.report().unwrap().clone();
    let t = s.training_residuals().unwrap();
    println!(
        "\n######## {label}: store {n_store}, fitted {} (report n {}), held {} (cap {:.1}); Q′ {}, κ(A) {:.2e}, λ {:.3e}, σ_E {:.3e}, σ_F {:.3e}; worst E/atom {:.3e} (ratio {:.4}), F {:.3e} / {:.3e} (ratio {:.4}); residuals fail {}; trusted {} {:?}",
        members.len(),
        r.n_samples,
        s.quarantined(),
        QUARANTINE_MAX_FRACTION * n_store as f64,
        r.q_prime,
        r.condition_estimate,
        r.lambda,
        r.sigma_e,
        r.sigma_f,
        t.energy_per_atom,
        t.energy_per_atom / tier,
        t.force,
        t.force_bound,
        t.force / t.force_bound,
        s.residuals_fail(),
        s.trusted(),
        s.untrusted_reason()
    );
    // ---- phase 1: the suspect set
    let t1 = Instant::now();
    let in_sample: Vec<(usize, f64)> = {
        let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
        par_map(&members, 16, |&i| {
            let smp = &store.samples[i];
            let Ok(p) = model.predict(&smp.pos, &smp.z, false) else { return (i, f64::INFINITY) };
            let n = smp.pos.len().max(1) as f64;
            let mut r = if smp.use_energy { (p.energy - smp.energy).abs() / n / tier } else { 0.0 };
            for (j, (fm, fs)) in p.forces.iter().zip(&smp.forces).enumerate() {
                if smp.trains_atom(j) {
                    let (d, b) = policy.force_residual(fm, fs);
                    r = super::super::trust::max_nan(r, d / b);
                }
            }
            (i, if r.is_nan() { f64::INFINITY } else { r })
        })
    };
    let loo: Vec<(usize, f64)> = s.loo_ratios().unwrap().to_vec();
    let loo_of = |i: usize| loo.iter().find(|x| x.0 == i).map_or(f64::NAN, |x| x.1);
    let cands: Vec<(usize, f64)> = loo.iter().copied().filter(|&(i, _)| !s.excluded(i) && !s.is_box(i)).collect();
    let worst_loo = cands.iter().map(|x| x.1).fold(0.0, f64::max);
    let loo_off: Vec<usize> = if worst_loo <= 1.0 { Vec::new() } else { cands.iter().filter(|&&(_, r)| r > 1.0 && r >= 0.5 * worst_loo).map(|x| x.0).collect() };
    let code_off = s.worst_offenders();
    if s.residuals_fail() {
        assert_eq!(code_off.0, loo_off, "while the residuals fail this is the recovery's own ranking");
    }
    let failing: Vec<usize> = in_sample.iter().filter(|x| x.1 > 1.0).map(|x| x.0).collect();
    let mut set: Vec<usize> = failing.iter().filter(|_| !loo_only).chain(&loo_off).copied().collect();
    set.sort_unstable();
    set.dedup();
    let wall_suspects = t1.elapsed().as_secs_f64();
    println!(
        "phase 1 (suspects{}) {wall_suspects:.2} s: |S| = {} ({} failing in-sample, {} leave-one-out offenders, worst ρ {worst_loo:.2}; the code's worst_offenders(): {} named, worst {:.2}, mode {}); |S| + held = {} vs cap {:.1}",
        if loo_only { ", LOO offenders only" } else { "" },
        set.len(),
        failing.len(),
        loo_off.len(),
        code_off.0.len(),
        code_off.1,
        if s.residuals_fail() { "own bound" } else { "absolute tier ρ_σ (residuals pass)" },
        set.len() + s.quarantined(),
        QUARANTINE_MAX_FRACTION * n_store as f64
    );
    {
        if loo_only {
            println!("    (contrast: S = the leave-one-out offenders alone; the in-sample failures stay in the fit)");
        }
        let store = s.store.as_ref().unwrap();
        let ins = |i: usize| in_sample.iter().find(|x| x.0 == i).map_or(f64::NAN, |x| x.1);
        let mut rows: Vec<usize> = set.clone();
        rows.sort_by(|&a, &b| loo_of(b).total_cmp(&loo_of(a)));
        for &i in &rows {
            let smp = &store.samples[i];
            println!(
                "    S #{i:<5} {:<5} {:<18} in-sample {:>8.3}  LOO {:>9.3}  {}{}",
                formula(&smp.z),
                origin_word(smp.tag),
                ins(i),
                loo_of(i),
                if failing.contains(&i) { "fails in-sample" } else { "" },
                if loo_off.contains(&i) { if failing.contains(&i) { " + LOO offender" } else { "LOO offender" } } else { "" }
            );
        }
        let mut by: BTreeMap<String, usize> = BTreeMap::new();
        for &i in &set {
            *by.entry(formula(&store.samples[i].z)).or_default() += 1;
        }
        println!("    S by formula: {by:?}");
    }
    if set.is_empty() {
        println!("S is empty: Δc_S = 0, nothing to measure ({:.1} s)", t_total.elapsed().as_secs_f64());
        return 0;
    }
    // ---- phase 2: the downdate
    let t2 = Instant::now();
    let (model, store) = (s.model.as_ref().unwrap(), s.store.as_ref().unwrap());
    let s_samples: Vec<&Sample> = set.iter().map(|&i| &store.samples[i]).collect();
    let d = match block_downdate(model, &s_samples) {
        Ok(d) => d,
        Err(e) => {
            println!("phase 2: the downdate is undefined: {e}");
            return set.len();
        }
    };
    let wall_down = t2.elapsed().as_secs_f64();
    println!("phase 2 (block downdate) {wall_down:.2} s: m {} rows, min pivot of G = I − H {:.3e}; ‖Δc_S‖∞ {:.3e}", d.m, d.min_pivot, max_abs(&d.dc));
    // ---- phase 3: every stored sample's shift, and the members' residuals under it
    let t3 = Instant::now();
    let all: Vec<usize> = (0..n_store).collect();
    let is_member: Vec<bool> = { let mut v = vec![false; n_store]; members.iter().for_each(|&i| v[i] = true); v };
    let is_s: Vec<bool> = { let mut v = vec![false; n_store]; set.iter().for_each(|&i| v[i] = true); v };
    let rows: Vec<Option<SampleRow>> = par_map(&all, 16, |&i| {
        let smp = &store.samples[i];
        let (de, df) = shift(model, &d.dc, &smp.pos, &smp.z)?;
        let p = model.predict(&smp.pos, &smp.z, false).ok()?;
        let n = smp.pos.len().max(1) as f64;
        let (mut shift_f, mut shift_f_raw) = (0.0f64, 0.0f64);
        for (j, dfj) in df.iter().enumerate() {
            // ‖ΔF_j‖/√3 against the bound of the atom's teacher force (force_residual's reading)
            let (dd, b) = policy.force_residual(&std::array::from_fn(|a| smp.forces[j][a] + dfj[a]), &smp.forces[j]);
            shift_f = shift_f.max(dd / b);
            shift_f_raw = shift_f_raw.max(dd * 3f64.sqrt() / b);
        }
        let mut row = SampleRow { idx: i, formula: formula(&smp.z), member: is_member[i], in_s: is_s[i], shift_f, shift_f_raw, shift_e: de.abs() / tier, before_e: 0.0, before_f: 0.0, after_e: 0.0, after_f: (0.0, 0.0, 0.0, 0) };
        if row.member {
            if smp.use_energy {
                row.before_e = (p.energy - smp.energy).abs() / n;
                row.after_e = (p.energy + de * n - smp.energy).abs() / n;
            }
            for j in (0..smp.pos.len()).filter(|&j| smp.trains_atom(j)) {
                let (d0, b0) = policy.force_residual(&p.forces[j], &smp.forces[j]);
                row.before_f = row.before_f.max(d0 / b0);
                let fd: [f64; 3] = std::array::from_fn(|a| p.forces[j][a] + df[j][a]);
                let (d1, b1) = policy.force_residual(&fd, &smp.forces[j]);
                if d1 / b1 > row.after_f.0 {
                    row.after_f = (d1 / b1, d1, b1, j);
                }
            }
        }
        Some(row)
    });
    let rows: Vec<SampleRow> = rows.into_iter().flatten().collect();
    let wall_shift = t3.elapsed().as_secs_f64();
    // gate (i): the members less S, under the downdate
    let rest: Vec<&SampleRow> = rows.iter().filter(|r| r.member && !r.in_s).collect();
    let we = rest.iter().max_by(|a, b| a.after_e.total_cmp(&b.after_e)).unwrap();
    let wf = rest.iter().max_by(|a, b| a.after_f.0.total_cmp(&b.after_f.0)).unwrap();
    let be = rest.iter().map(|r| r.before_e).fold(0.0, f64::max);
    let bf = rest.iter().map(|r| r.before_f).fold(0.0, f64::max);
    let gate_i = we.after_e <= tier && wf.after_f.0 <= 1.0;
    println!(
        "phase 3 (shifts + residuals of {} stored samples) {wall_shift:.2} s\n  gate (i) over the {} members outside S: worst energy {:.3e} Ha/atom = {:.4}× tier (#{} {}; standing fit {:.4}×), worst force {:.3e} / {:.3e} = {:.4}× (#{} {} atom {}; standing fit {:.4}×) → {}",
        rows.len(),
        rest.len(),
        we.after_e,
        we.after_e / tier,
        we.idx,
        we.formula,
        be / tier,
        wf.after_f.1,
        wf.after_f.2,
        wf.after_f.0,
        wf.idx,
        wf.formula,
        wf.after_f.3,
        bf,
        if gate_i { "PASSES" } else { "FAILS" }
    );
    let over: Vec<&&SampleRow> = rest.iter().filter(|r| r.after_f.0 > 1.0 || r.after_e > tier).collect();
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    over.iter().for_each(|r| *by.entry(r.formula.clone()).or_default() += 1);
    println!("  members outside S failing under the downdate: {} {by:?}", over.len());
    for f in ["H2", "H3", "H2O", "H3O", "N2", "HO"] {
        let of: Vec<&SampleRow> = rows.iter().filter(|r| r.formula == f).collect();
        if of.is_empty() {
            continue;
        }
        let pass = |r: &&SampleRow, raw: bool| (if raw { r.shift_f_raw } else { r.shift_f }) <= 0.1 && r.shift_e <= 0.1;
        let outside: Vec<&&SampleRow> = of.iter().filter(|r| !r.in_s).collect();
        println!(
            "  {f:<4} n {:>4} ({} fitted, {} in S)\n       force shift (‖ΔF‖/√3 / b): {}\n       energy shift (|ΔE|/N / tier): {}\n       gate (ii) at ε 0.1: {} of {} pass ({} of {} outside S); with the raw ‖ΔF‖/b: {} of {}",
            of.len(),
            of.iter().filter(|r| r.member).count(),
            of.iter().filter(|r| r.in_s).count(),
            quantiles(of.iter().map(|r| r.shift_f).collect()),
            quantiles(of.iter().map(|r| r.shift_e).collect()),
            of.iter().filter(|r| pass(r, false)).count(),
            of.len(),
            outside.iter().filter(|r| pass(r, false)).count(),
            outside.len(),
            of.iter().filter(|r| pass(r, true)).count(),
            of.len()
        );
        let mut worst: Vec<&&SampleRow> = outside.clone();
        worst.sort_by(|a, b| b.shift_f.max(b.shift_e).total_cmp(&a.shift_f.max(a.shift_e)));
        let w: Vec<String> = worst.iter().take(4).map(|r| format!("#{} F {:.3} E {:.3}", r.idx, r.shift_f, r.shift_e)).collect();
        println!("       largest outside S: {}", w.join(", "));
    }
    // ---- phase 4: the oracle on the real fit (an explicit refit without S)
    if std::env::var("SURROGATE_SCOPE_EXPLICIT").map_or(true, |v| v != "0") {
        let t4 = Instant::now();
        let kept: Vec<&Sample> = members.iter().filter(|&&i| !is_s[i]).map(|&i| &store.samples[i]).collect();
        let ex = explicit_refit(model, &kept, &s_samples);
        let dc_ex: Vec<f64> = ex.theta_full.iter().zip(&ex.theta_rest).map(|(a, b)| a - b).collect();
        let dc_err = max_abs(&d.dc.iter().zip(&dc_ex).map(|(a, b)| a - b).collect::<Vec<_>>()) / max_abs(&dc_ex);
        // the H₂/H₃ shifts under the explicit Δc, against the downdate's, in bound units
        let hh: Vec<usize> = rows.iter().filter(|r| r.formula == "H2" || r.formula == "H3").map(|r| r.idx).collect();
        let dev = par_map(&hh, 16, |&i| {
            let smp = &store.samples[i];
            let (a, fa) = shift(model, &d.dc, &smp.pos, &smp.z).unwrap();
            let (b, fb) = shift(model, &dc_ex, &smp.pos, &smp.z).unwrap();
            let mut m = (a - b).abs() / tier;
            for j in 0..fa.len() {
                let (dd, _) = policy.force_residual(&fa[j], &fb[j]);
                m = m.max(dd / policy.force_bound((smp.forces[j][0].powi(2) + smp.forces[j][1].powi(2) + smp.forces[j][2].powi(2)).sqrt()));
            }
            m
        })
        .into_iter()
        .fold(0.0, f64::max);
        println!(
            "phase 4 (explicit refit without S, the oracle) {:.1} s: λ_eff {:.4e} (model λ {:.4e}, diagonal spread {:.1e}); |L Lᵀ − A − λI|/|L Lᵀ| {:.2e}; Δc vs θ_full − θ_rest {:.2e} (relative to ‖Δc‖∞ {:.3e}); H₂/H₃ shifts, downdate vs explicit: worst difference {:.2e} bound units",
            t4.elapsed().as_secs_f64(),
            ex.lambda_eff,
            model.lambda,
            ex.lambda_spread,
            ex.a_dev,
            dc_err,
            max_abs(&dc_ex),
            dev
        );
    }
    println!("{label}: {:.1} s in all (suspects {wall_suspects:.2}, downdate {wall_down:.2}, shifts {wall_shift:.2})", t_total.elapsed().as_secs_f64());
    set.len()
}

/// A fresh copy of the collection folder `SURROGATE_SCOPE_SNAPSHOT` names, opened (the copy's
/// temp root is `tag`'s; neither the folder nor any user folder is written).
fn scope_copy(tag: &str) -> (surrogate::Collection, CollectionRoot) {
    let src = std::path::PathBuf::from(std::env::var("SURROGATE_SCOPE_SNAPSHOT").expect("SURROGATE_SCOPE_SNAPSHOT names a collection folder copy"));
    let key = surrogate::SurfaceKey::from_text(&std::fs::read_to_string(src.join("key.txt")).unwrap()).unwrap();
    let root = collection_root(tag);
    let dst = root.join(key.slug());
    std::fs::create_dir_all(&dst).unwrap();
    for f in ["key.txt", "references.txt", "samples.bin", "history.bin"] {
        if src.join(f).exists() {
            std::fs::copy(src.join(f), dst.join(f)).unwrap();
        }
    }
    let c = surrogate::Collection::open(&root, &key).unwrap().0;
    (c, root) // the caller keeps the root (removed on drop) while the collection writes into it
}

/// The measurement (module doc): attach a copy of the snapshot and refit as the app does, then
/// measure on the fit that stands; then on the entry fit (the file's codes, no recovery), the
/// fit the refit's recovery judged first — with the study's S, and as a contrast with S = the
/// leave-one-out offenders alone.
///
/// `SURROGATE_SCOPE_SNAPSHOT=<copy>/<slug> cargo test -p providers --release --lib scope_measurement -- --ignored --nocapture --test-threads 1`
/// (`SURROGATE_SCOPE_EXPLICIT=0` skips phase 4, the explicit-refit oracle).
#[test]
#[ignore = "measurement on a copy of a collection snapshot (SURROGATE_SCOPE_SNAPSHOT)"]
fn scope_measurement() {
    let (c, _root) = scope_copy("scope-measure");
    let disk = c.quarantine.clone();
    println!("snapshot: {} samples, {} held on disk", c.samples.len(), disk.iter().filter(|&&q| q == Q_HELD).count());
    let mut s = SurrogateKs::new(Box::new(StubKsGpu), FallbackPolicy::default()).with_teacher_tol(1e-6);
    let t0 = Instant::now();
    s.attach(c).unwrap();
    if s.report().is_none() {
        SurrogateKs::refit(&mut s).unwrap();
    }
    println!("phase 0 (attach + refit, as the app) {:.1} s, {} fits; recovery: {:?}", t0.elapsed().as_secs_f64(), s.fits, s.recovery_note());
    measure(&mut s, "the fit that stands after the attach's refit", false);
    // the entry fit: the file's codes (lone atoms kept out), no recovery
    let lone: Vec<usize> = (0..s.quarantine.len()).filter(|&i| s.quarantine[i] == Q_LONE).collect();
    let t5 = Instant::now();
    s.quarantine = disk.clone();
    s.quarantine.resize(s.samples(), Q_NONE);
    for &i in &lone {
        s.quarantine[i] = Q_LONE;
    }
    s.degree_cap = None;
    s.species_dirty = true;
    s.fit_store_deferred().unwrap();
    println!("\nentry fit (the file's codes, default basis, no recovery): {:.1} s", t5.elapsed().as_secs_f64());
    measure(&mut s, "the entry fit", false);
    measure(&mut s, "the entry fit, contrast: S = the leave-one-out offenders alone", true);
}
