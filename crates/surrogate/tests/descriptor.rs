//! Task 2 acceptance tests for `surrogate::descriptor` and `surrogate::model` (the brief's five,
//! plus the hot path vs descriptor path agreement and an ignored timing probe).
//! Finite differences use a 5-point central stencil (O(h⁴)); the fixtures reject any pair within
//! 0.05 Bohr of r_cut so no stencil straddles the C¹ (not C²) point of the cutoff.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use surrogate::descriptor::{describe, Basis, DescribeError};
use surrogate::model::{energy_and_forces, Model, Provenance};

const H: u8 = 1;
const O: u8 = 8;

fn fd5(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Random cluster of `n` atoms (alternating H/O) in a cube of side `side`, re-drawn until no pair
/// is closer than 0.8 Bohr or within 0.05 Bohr of r_cut (keeps every FD stencil on one side of
/// the cutoff). With side 5 and r_cut 6 some pairs fall outside the cutoff, so the neighbour
/// test is exercised.
fn random_cluster(rng: &mut StdRng, n: usize, side: f64, r_cut: f64) -> (Vec<[f64; 3]>, Vec<u8>) {
    let z: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { H } else { O }).collect();
    loop {
        let pos: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                [
                    rng.gen_range(0.0..side),
                    rng.gen_range(0.0..side),
                    rng.gen_range(0.0..side),
                ]
            })
            .collect();
        let ok = (0..n).all(|i| {
            (i + 1..n).all(|j| {
                let r = dist(pos[i], pos[j]);
                r > 0.8 && (r - r_cut).abs() > 0.05
            })
        });
        if ok {
            return (pos, z);
        }
    }
}

fn provenance() -> Provenance {
    Provenance {
        grid_n: 49,
        h: 0.25,
        sigma: 0.25,
        tol: 1e-6,
        charge: 0,
        multiplicity: None,
        teacher: "test".to_string(),
    }
}

/// Model with random coefficients in [−1, 1], random per-feature mean (unless `zero_mean`) and
/// scale in [0.5, 2] (one feature pinned to scale 0 to exercise the constant-feature rule), and
/// random reference energies.
fn random_model(rng: &mut StdRng, basis: Basis, zero_mean: bool) -> Model {
    let p = basis.n_features();
    let s = basis.species.len();
    let coeffs = (0..s)
        .map(|_| (0..p).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let mean = (0..p)
        .map(|_| if zero_mean { 0.0 } else { rng.gen_range(-1.0..1.0) })
        .collect();
    let mut scale: Vec<f64> = (0..p).map(|_| rng.gen_range(0.5..2.0)).collect();
    scale[p / 2] = 0.0;
    let ref_energy = (0..s).map(|_| rng.gen_range(-80.0..-0.4)).collect();
    Model {
        basis,
        coeffs,
        mean,
        scale,
        ref_energy,
        sigma_e: 0.0,
        sigma_f: 0.0,
        energy_per_atom: true,
        lambda: 0.0,
        sigma_inv_chol: None,
        n_samples: 0,
        provenance: provenance(),
        representatives: std::sync::OnceLock::new(),
    }
}

/// Energy and forces from the per-atom descriptor lists, written out from the model definition
/// E_i = ref[s_i] + Σ_k c_{s_i,k}(B_{i,k} − mean_k)/scale_k independently of the hot path.
fn via_descriptors(model: &Model, pos: &[[f64; 3]], z: &[u8]) -> (f64, Vec<[f64; 3]>) {
    let basis = &model.basis;
    let desc = describe(basis, pos, z).unwrap();
    let mut energy = 0.0;
    let mut forces = vec![[0.0; 3]; pos.len()];
    for (i, d) in desc.iter().enumerate() {
        let s = basis.species_index(z[i]).unwrap();
        energy += model.ref_energy[s];
        for k in 0..basis.n_features() {
            let scale = if model.scale[k] == 0.0 { 1.0 } else { model.scale[k] };
            let c = model.coeffs[s][k] / scale;
            energy += c * (d.b[k] - model.mean[k]);
            for (j, g) in &d.grad {
                for a in 0..3 {
                    forces[*j][a] -= c * g[k][a];
                }
            }
        }
    }
    (energy, forces)
}

#[test]
fn descriptors_are_translation_and_permutation_invariant() {
    let mut rng = StdRng::seed_from_u64(20261013);
    let mut basis = Basis::new(&[O, H]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    let (pos, z) = random_cluster(&mut rng, 5, 5.0, basis.r_cut);
    let d0 = describe(&basis, &pos, &z).unwrap();
    let p = basis.n_features();
    assert_eq!(p, 16, "n_max·S = 8·2");
    assert!(d0.iter().all(|d| d.b.len() == p));
    assert!(
        d0.iter().any(|d| d.b.iter().any(|b| *b != 0.0)),
        "fixture has neighbours"
    );

    // Translation by an arbitrary vector: identical to rounding.
    let t = [3.7, -12.1, 0.42];
    let shifted: Vec<[f64; 3]> = pos
        .iter()
        .map(|r| [r[0] + t[0], r[1] + t[1], r[2] + t[2]])
        .collect();
    let d1 = describe(&basis, &shifted, &z).unwrap();
    let mut worst_t: f64 = 0.0;
    for (a, b) in d0.iter().zip(&d1) {
        for k in 0..p {
            let err = (a.b[k] - b.b[k]).abs() / a.b[k].abs().max(1.0);
            worst_t = worst_t.max(err);
            assert!(err < 1e-14, "translation changed B_k: {} vs {}", a.b[k], b.b[k]);
        }
    }

    // Rotation (distances only enter): identical to rounding as well.
    let (c, s) = (0.6f64, 0.8f64);
    let rotated: Vec<[f64; 3]> = pos
        .iter()
        .map(|r| [c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])
        .collect();
    let d2 = describe(&basis, &rotated, &z).unwrap();
    for (a, b) in d0.iter().zip(&d2) {
        for k in 0..p {
            let err = (a.b[k] - b.b[k]).abs() / a.b[k].abs().max(1.0);
            assert!(err < 1e-13, "rotation changed B_k: {} vs {}", a.b[k], b.b[k]);
        }
    }

    // Permutation: B follows the atom, and so does its gradient list (relabelled).
    let mut perm: Vec<usize> = (0..pos.len()).collect();
    perm.shuffle(&mut rng);
    assert!(perm.iter().enumerate().any(|(i, p)| i != *p));
    let ppos: Vec<[f64; 3]> = perm.iter().map(|&i| pos[i]).collect();
    let pz: Vec<u8> = perm.iter().map(|&i| z[i]).collect();
    let d3 = describe(&basis, &ppos, &pz).unwrap();
    let mut worst_p: f64 = 0.0;
    for (new_i, &old_i) in perm.iter().enumerate() {
        for k in 0..p {
            let err = (d3[new_i].b[k] - d0[old_i].b[k]).abs() / d0[old_i].b[k].abs().max(1.0);
            worst_p = worst_p.max(err);
            assert!(err < 1e-14, "permutation changed B_k");
        }
        // Gradient lists: the same set of (atom, dB/dr) pairs under the relabelling.
        assert_eq!(d3[new_i].grad.len(), d0[old_i].grad.len());
        for (new_j, g_new) in &d3[new_i].grad {
            let old_j = perm[*new_j];
            let (_, g_old) = d0[old_i]
                .grad
                .iter()
                .find(|(j, _)| *j == old_j)
                .expect("permuted gradient entry present");
            for k in 0..p {
                for a in 0..3 {
                    let err = (g_new[k][a] - g_old[k][a]).abs() / g_old[k][a].abs().max(1.0);
                    assert!(err < 1e-14, "permutation changed dB/dr");
                }
            }
        }
    }
    eprintln!("invariance: translation {worst_t:.2e}, permutation {worst_p:.2e}");
}

#[test]
fn descriptor_gradients_match_finite_differences() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut basis = Basis::new(&[H, O]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    let (pos, z) = random_cluster(&mut rng, 5, 5.0, basis.r_cut);
    let desc = describe(&basis, &pos, &z).unwrap();
    let p = basis.n_features();
    let n = pos.len();
    let h = 1e-3;
    let mut worst: f64 = 0.0;
    let mut n_nonzero = 0usize;
    for (i, di) in desc.iter().enumerate() {
        for j in 0..n {
            let analytic: Option<&Vec<[f64; 3]>> = di
                .grad
                .iter()
                .find(|(jj, _)| *jj == j)
                .map(|(_, g)| g);
            for a in 0..3 {
                for k in 0..p {
                    let fd = fd5(
                        |x| {
                            let mut q = pos.clone();
                            q[j][a] = x;
                            describe(&basis, &q, &z).unwrap()[i].b[k]
                        },
                        pos[j][a],
                        h,
                    );
                    let an = analytic.map_or(0.0, |g| g[k][a]);
                    if an != 0.0 {
                        n_nonzero += 1;
                    }
                    let err = (fd - an).abs() / an.abs().max(1.0);
                    worst = worst.max(err);
                    assert!(
                        err < 1e-8,
                        "dB_{{{i},{k}}}/dr_{{{j},{a}}}: analytic {an} vs FD {fd}"
                    );
                }
            }
        }
    }
    assert!(n_nonzero > 100, "fixture exercises gradients ({n_nonzero} nonzero)");
    eprintln!("dB/dr vs FD: worst {worst:.2e} over {n_nonzero} nonzero entries");
}

#[test]
fn forces_are_the_gradient_of_the_energy_and_conserve_momentum_and_torque() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut basis = Basis::new(&[H, O]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    let (pos, z) = random_cluster(&mut rng, 6, 5.0, basis.r_cut);
    let model = random_model(&mut rng, basis, false);
    let pred = model.predict(&pos, &z, true).unwrap();
    assert!(pred.energy_std.is_none() && pred.force_std.is_none());
    let energy = |q: &[[f64; 3]]| model.predict(q, &z, false).unwrap().energy;
    assert_eq!(energy(&pos), pred.energy);

    let h = 1e-3;
    let mut worst: f64 = 0.0;
    let mut fmax: f64 = 0.0;
    for j in 0..pos.len() {
        for a in 0..3 {
            let fd = -fd5(
                |x| {
                    let mut q = pos.clone();
                    q[j][a] = x;
                    energy(&q)
                },
                pos[j][a],
                h,
            );
            let f = pred.forces[j][a];
            fmax = fmax.max(f.abs());
            let err = (fd - f).abs() / f.abs().max(1.0);
            worst = worst.max(err);
            assert!(err < 1e-8, "F_{{{j},{a}}}: analytic {f} vs −dE/dr {fd}");
        }
    }
    assert!(fmax > 0.1, "fixture has forces ({fmax})");

    let mut sum = [0.0f64; 3];
    let mut torque = [0.0f64; 3];
    for (r, f) in pos.iter().zip(&pred.forces) {
        for a in 0..3 {
            sum[a] += f[a];
        }
        torque[0] += r[1] * f[2] - r[2] * f[1];
        torque[1] += r[2] * f[0] - r[0] * f[2];
        torque[2] += r[0] * f[1] - r[1] * f[0];
    }
    let sum_n = (sum[0].powi(2) + sum[1].powi(2) + sum[2].powi(2)).sqrt();
    let torque_n = (torque[0].powi(2) + torque[1].powi(2) + torque[2].powi(2)).sqrt();
    assert!(sum_n < 1e-12, "ΣF = {sum_n:e}");
    assert!(torque_n < 1e-12, "Σ r×F = {torque_n:e}");
    eprintln!("forces vs FD: worst {worst:.2e} (|F|max {fmax:.2}); ΣF {sum_n:.2e}, Σ r×F {torque_n:.2e}");
}

#[test]
fn two_atoms_beyond_r_cut_give_the_reference_energies() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut basis = Basis::new(&[H, O]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    let r_cut = basis.r_cut;
    // Zero feature means: the isolated-atom limit of the standardised form is
    // ref[s] − Σ_k c̃_{s,k} mean_k, which equals ref[s] only for mean = 0 (see the report).
    let model = random_model(&mut rng, basis, true);
    let pos = [[0.0, 0.0, 0.0], [r_cut + 1.0, 0.3, -0.2]];
    let z = [O, H];
    let pred = model.predict(&pos, &z, false).unwrap();
    let expected = model.ref_energy[1] + model.ref_energy[0];
    assert!(
        (pred.energy - expected).abs() < 1e-10,
        "E = {} vs Σ ref = {}",
        pred.energy,
        expected
    );
    assert_eq!(pred.forces, vec![[0.0; 3]; 2], "no pair inside r_cut → F exactly 0");

    // Exactly at r_cut the cutoff is already zero (half-open sphere).
    let at = [[0.0, 0.0, 0.0], [r_cut, 0.0, 0.0]];
    let pred_at = model.predict(&at, &z, false).unwrap();
    assert!((pred_at.energy - expected).abs() < 1e-10);
    assert_eq!(pred_at.forces, vec![[0.0; 3]; 2]);

    // Just inside, the energy moves away from the references (the fixture is not degenerate).
    let inside = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let pred_in = model.predict(&inside, &z, false).unwrap();
    assert!((pred_in.energy - expected).abs() > 1e-3);

    let desc = describe(&model.basis, &pos, &z).unwrap();
    for d in &desc {
        assert!(d.b.iter().all(|b| *b == 0.0));
        assert_eq!(d.grad.len(), 1, "only the self entry");
    }
}

#[test]
fn unknown_species_is_an_error_not_a_guess() {
    let mut rng = StdRng::seed_from_u64(5);
    let mut basis = Basis::new(&[H, O]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    assert_eq!(basis.species_index(11), None);
    let pos = [[0.0; 3], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]];
    let z = [H, 11, O];
    assert_eq!(
        describe(&basis, &pos, &z).unwrap_err(),
        DescribeError::UnknownSpecies(11)
    );
    let model = random_model(&mut rng, basis, false);
    assert_eq!(
        model.predict(&pos, &z, false).unwrap_err(),
        DescribeError::UnknownSpecies(11)
    );
    assert_eq!(
        energy_and_forces(&model, &pos, &z).unwrap_err(),
        DescribeError::UnknownSpecies(11)
    );
}

#[test]
fn hot_path_matches_the_descriptor_path() {
    let mut rng = StdRng::seed_from_u64(13);
    let mut basis = Basis::new(&[H, O]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    let mut worst_e: f64 = 0.0;
    let mut worst_f: f64 = 0.0;
    for n in [2usize, 5, 9] {
        let (pos, z) = random_cluster(&mut rng, n, 5.0, basis.r_cut);
        let model = random_model(&mut rng, basis.clone(), false);
        let (e_hot, f_hot) = energy_and_forces(&model, &pos, &z).unwrap();
        let (e_ref, f_ref) = via_descriptors(&model, &pos, &z);
        let err_e = (e_hot - e_ref).abs() / e_ref.abs().max(1.0);
        worst_e = worst_e.max(err_e);
        assert!(err_e < 1e-14, "N = {n}: E {e_hot} vs {e_ref}");
        for (fh, fr) in f_hot.iter().zip(&f_ref) {
            for a in 0..3 {
                let err = (fh[a] - fr[a]).abs() / fr[a].abs().max(1.0);
                worst_f = worst_f.max(err);
                assert!(err < 1e-14, "N = {n}: F {fh:?} vs {fr:?}");
            }
        }
        let pred = model.predict(&pos, &z, false).unwrap();
        assert_eq!(pred.energy, e_hot);
        assert_eq!(pred.forces, f_hot);
    }
    eprintln!("hot path vs descriptor path: energy {worst_e:.2e}, forces {worst_f:.2e}");
}

/// Timing probe (reported, not asserted): ns per atom pair for `describe` and for
/// `energy_and_forces` at N = 20, S = 2. Run with
/// `cargo test -p surrogate --release --test descriptor -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_probe_n20_s2() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut basis = Basis::new(&[H, O]);
    basis.body3 = false; // body-2 regression fixture (the default is body-3)
    let (pos, z) = random_cluster(&mut rng, 20, 6.0, basis.r_cut);
    let model = random_model(&mut rng, basis.clone(), false);
    let n_pairs = (pos.len() * (pos.len() - 1) / 2) as f64;
    let inside = (0..pos.len())
        .flat_map(|i| (i + 1..pos.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| dist(pos[i], pos[j]) < basis.r_cut)
        .count();
    let reps = 2000;
    let t0 = std::time::Instant::now();
    let mut sink = 0.0;
    for _ in 0..reps {
        let d = describe(&basis, &pos, &z).unwrap();
        sink += d[0].b[0];
    }
    let describe_ns = t0.elapsed().as_nanos() as f64 / reps as f64;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let (e, _) = energy_and_forces(&model, &pos, &z).unwrap();
        sink += e;
    }
    let hot_ns = t0.elapsed().as_nanos() as f64 / reps as f64;
    eprintln!(
        "N = 20, S = 2, {inside} of {n_pairs} pairs inside r_cut (sink {sink:e}):\n  describe: {:.0} ns/call = {:.1} ns/pair\n  energy_and_forces: {:.0} ns/call = {:.1} ns/pair",
        describe_ns,
        describe_ns / n_pairs,
        hot_ns,
        hot_ns / n_pairs
    );
}
