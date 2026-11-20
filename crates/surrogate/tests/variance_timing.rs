//! The cost of `Model::predict` with and without the predictive variance (spec
//! `2026-11-20-variance-support-design.md`, "Measurement"): H₂ at 1.4 Bohr, 2000 calls each. A
//! measurement; nothing is asserted but that the variance is reported.
//!
//! `MODEL_FILE=<a model sidecar> cargo test -p surrogate --release --test variance_timing -- --ignored --nocapture`
//!
//! Without `MODEL_FILE` it measures a synthetic model of the user's BEST shape: species H, O, P,
//! the default basis with body-4 at degree 3 (Q′ = 1902), seeded random coefficients and a
//! posterior factor L = chol(λI + BBᵀ) with B a random Q′ × 64 matrix (the cost does not depend
//! on the values).
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;
use surrogate::{Basis, Model, Provenance};

fn synthetic_best_shape() -> Model {
    let mut rng = StdRng::seed_from_u64(0x5eed_2026_1120);
    let mut basis = Basis::new(&[1, 8, 15]);
    basis.body4 = true;
    basis.body4_degree = 3;
    let p = basis.n_features();
    let s = basis.n_species();
    let q = basis.n_independent_columns();
    let k = 64;
    let b: Vec<f64> = (0..q * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut a = vec![0.0; q * q];
    for i in 0..q {
        for j in 0..=i {
            let d: f64 = (0..k).map(|t| b[i * k + t] * b[j * k + t]).sum();
            a[i * q + j] = d;
            a[j * q + i] = d;
        }
        a[i * q + i] += 1e-3;
    }
    surrogate::linalg::cholesky_in_place_fast(&mut a, q).expect("λI + BBᵀ is positive definite");
    Model {
        basis,
        coeffs: (0..s).map(|_| (0..p).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect(),
        mean: vec![0.0; p],
        scale: (0..p).map(|_| rng.gen_range(0.5..2.0)).collect(),
        ref_energy: vec![-0.478, -74.8, -340.0],
        sigma_e: 1e-4,
        sigma_f: 1e-3,
        energy_per_atom: true,
        lambda: 1e-3,
        sigma_inv_chol: Some(a),
        n_samples: 0,
        provenance: Provenance {
            grid_n: 1,
            h: 1.0,
            sigma: 1.0,
            tol: 1.0,
            charge: 0,
            multiplicity: None,
            teacher: String::new(),
            masses: Vec::new(),
            ionic: "none".into(),
        },
        representatives: std::sync::OnceLock::new(),
        variance_support: surrogate::model::SupportCache::new(),
    }
}

#[test]
#[ignore = "a timing measurement (MODEL_FILE, else a synthetic BEST-shaped model)"]
fn predict_cost_on_a_model_file() {
    let model = match std::env::var_os("MODEL_FILE") {
        Some(path) => surrogate::sidecar::read(std::path::Path::new(&path)).expect("a readable model sidecar").model,
        None => synthetic_best_shape(),
    };
    let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]];
    let z = [1u8, 1];
    println!(
        "species {:?}, Q′ {}, covariance {}",
        model.basis.species,
        model.basis.n_independent_columns(),
        model.sigma_inv_chol.is_some()
    );
    let calls = 2000;
    // warm-up: the first call fills the lazy caches (the representatives, the variance support)
    let t = Instant::now();
    let first = model.predict(&pos, &z, true).unwrap();
    let first_ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(first.energy_std.is_some(), "the model carries a posterior");
    let t = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(model.predict(std::hint::black_box(&pos), &z, false).unwrap());
    }
    let plain = t.elapsed().as_secs_f64() / calls as f64 * 1e6;
    let t = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(model.predict(std::hint::black_box(&pos), &z, true).unwrap());
    }
    let var = t.elapsed().as_secs_f64() / calls as f64 * 1e6;
    let t = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(surrogate::describe(&model.basis, std::hint::black_box(&pos), &z).unwrap());
    }
    let desc = t.elapsed().as_secs_f64() / calls as f64 * 1e6;
    println!("first predict(.., true): {first_ms:.2} ms; describe alone {desc:.1} µs");
    println!("support: {:?}", model.variance_support.support_for(&[0]).map(|s| s.len()));
    println!("predict(.., false): {plain:.1} µs, predict(.., true): {var:.1} µs (mean of {calls})");
    println!("energy_std {:e}, force_std {:?}", first.energy_std.unwrap(), first.force_std.unwrap());
}
