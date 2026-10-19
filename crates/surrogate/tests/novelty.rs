//! The unspanned-direction test (`surrogate::novelty`, M6 Task 11): a Morse dimer fit leaves
//! directions no training row reaches; a third atom reaches them, a dimer does not.
use surrogate::{Accumulator, Basis, FitOptions, NullSpace, Provenance, Sample, SampleTag};

fn dimer_fit() -> (Accumulator, surrogate::Model) {
    let basis = Basis::new(&[1]);
    let prov = Provenance { grid_n: 49, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "morse".into(), masses: vec![], ionic: "none".into() };
    let mut acc = Accumulator::new(&basis);
    let opts = FitOptions::default();
    let mut seed = 12345u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..300 {
        let r = 1.0 + 4.125 * i as f64 / 299.0;
        let (mut d, mut nn) = ([0.0; 3], 0.0);
        while !(1e-3..=1.0).contains(&nn) {
            d = [2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0];
            nn = d.iter().map(|x| x * x).sum::<f64>();
        }
        let u: Vec<f64> = d.iter().map(|x| x / nn.sqrt()).collect();
        let x = (-(r - 1.4f64)).exp();
        let e = -0.956 + 0.17 * ((1.0 - x) * (1.0 - x) - 1.0);
        let de = 0.34 * (1.0 - x) * x;
        let f1 = [-de * u[0], -de * u[1], -de * u[2]];
        let s = Sample { pos: vec![[0.0; 3], [r * u[0], r * u[1], r * u[2]]], z: vec![1, 1], energy: e, forces: vec![[-f1[0], -f1[1], -f1[2]], f1], tag: SampleTag::Scan, weight: 1.0 };
        acc.add(&s, &basis, &[-0.478], &opts).unwrap();
    }
    let (model, _) = acc.solve(&opts, &prov, &[-0.478]).unwrap();
    (acc, model)
}

#[test]
fn novelty_flags_a_third_atom_and_not_a_dimer() {
    let (acc, model) = dimer_fit();
    let ns = NullSpace::of(&acc, &model.scale);
    println!("unspanned directions {} of {}, τ {:.2e}", ns.dimension(), acc.q_prime, ns.tau);
    assert!(ns.dimension() > 0 && ns.dimension() < acc.q_prime);
    // no feature column was zero in training: the novelty is a direction, not a column
    assert!((0..acc.q_prime).all(|k| acc.envelope.max[k] != 0.0 || acc.envelope.min[k] != 0.0));
    let mut worst_dimer = 0.0f64;
    for (r, dir) in [(1.2, [1.0, 0.0, 0.0]), (1.7, [0.3, -0.5, 0.81]), (2.37, [0.0, 0.6, 0.8]), (4.4, [0.5, 0.5, 0.707])] {
        let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] as f64).sqrt();
        let pos = [[0.1, 0.2, 0.3], [0.1 + r * dir[0] / n, 0.2 + r * dir[1] / n, 0.3 + r * dir[2] / n]];
        let (nu, _) = ns.novelty(&pos, &[1, 1]).unwrap();
        worst_dimer = worst_dimer.max(nu);
    }
    let mut least_trimer = f64::INFINITY;
    for d3 in [2.0, 3.0, 5.0] {
        let pos = [[0.0; 3], [1.4, 0.0, 0.0], [0.7, d3, 0.0]];
        let (nu, at) = ns.novelty(&pos, &[1, 1, 1]).unwrap();
        println!("H₂ + H at {d3} Bohr: ν {nu:.2e} (row {at:?})");
        least_trimer = least_trimer.min(nu);
    }
    println!("dimers: ν ≤ {worst_dimer:.2e}; H₂ + H: ν ≥ {least_trimer:.2e}");
    assert!(worst_dimer < 1.0, "a dimer is in the training span: ν {worst_dimer:.2e}");
    assert!(least_trimer > 1.0, "a third atom reaches the unspanned directions: ν {least_trimer:.2e}");
}

/// A dimer store of `n` samples (r 1.0–5.125, random orientations, noise-free Morse).
fn dimer_store_fit(n: usize) -> (Accumulator, Result<surrogate::Model, surrogate::FitError>) {
    let basis = Basis::new(&[1]);
    let prov = Provenance { grid_n: 49, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "morse".into(), masses: vec![], ionic: "none".into() };
    let mut acc = Accumulator::new(&basis);
    let opts = FitOptions::default();
    let mut seed = 777u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..n {
        let r = 1.0 + 4.125 * i as f64 / (n - 1) as f64;
        let (mut d, mut nn) = ([0.0; 3], 0.0);
        while !(1e-3..=1.0).contains(&nn) {
            d = [2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0];
            nn = d.iter().map(|x| x * x).sum::<f64>();
        }
        let u: Vec<f64> = d.iter().map(|x| x / nn.sqrt()).collect();
        let x = (-(r - 1.4f64)).exp();
        let e = -0.956 + 0.17 * ((1.0 - x) * (1.0 - x) - 1.0);
        let de = 0.34 * (1.0 - x) * x;
        let f1 = [-de * u[0], -de * u[1], -de * u[2]];
        let s = Sample { pos: vec![[0.0; 3], [r * u[0], r * u[1], r * u[2]]], z: vec![1, 1], energy: e, forces: vec![[-f1[0], -f1[1], -f1[2]], f1], tag: SampleTag::Scan, weight: 1.0 };
        acc.add(&s, &basis, &[-0.478], &opts).unwrap();
    }
    let model = acc.solve(&opts, &prov, &[-0.478]).map(|(m, _)| m);
    (acc, model)
}

/// The rounding floor of the null eigenvalues against τ = Q′·ε·tr as the store grows (M6 Task 11
/// round 3: the Gram is Neumaier-compensated; a naive sum gave 3.6e-16·tr at 300 dimers and
/// 9.4e-16·tr at 3000 — the reviewer's measurement — a margin falling as the row count grows).
/// Numbers of record in `references.md`, "Null-space margin"; 30 000 is `--ignored`.
fn margin(n: usize) -> (f64, f64, f64) {
    let (acc, model) = dimer_store_fit(n);
    // the standardisation of the fit (the model's scale; a Singular fit is reported, and the
    // scale computed as the fit does)
    let p = acc.basis.n_features();
    let scale: Vec<f64> = match &model {
        Ok(m) => m.scale.clone(),
        Err(_) => acc.sumsq.iter().map(|&ss| if ss > 0.0 { (ss / acc.n_e as f64).sqrt() } else { 1.0 }).collect(),
    };
    let q = acc.q_prime;
    let sums = acc.sums();
    let g: Vec<f64> = (0..q * q).map(|ij| (sums.gram_e[ij] + sums.gram_f[ij]) / (scale[(ij / q) % p] * scale[(ij % q) % p])).collect();
    let tr: f64 = (0..q).map(|i| g[i * q + i]).sum();
    let (ev, _) = surrogate::linalg::symmetric_eigen(&g, q);
    let tau = q as f64 * f64::EPSILON * tr;
    let null_max = ev.iter().filter(|&&l| l <= tau).fold(0.0f64, |m, &l| m.max(l.abs()));
    let spanned_min = ev.iter().filter(|&&l| l > tau).fold(f64::INFINITY, |m, &l| m.min(l));
    println!("{n} dimers: null directions {}, max |λ_null|/tr {:.2e}, min spanned/tr {:.2e}, τ/tr {:.2e}: margin τ/max|λ_null| {:.0}×, spanned/τ {:.0}×; fit {}",
             ev.iter().filter(|&&l| l <= tau).count(), null_max / tr, spanned_min / tr, tau / tr, tau / null_max, spanned_min / tau,
             match &model { Ok(_) => "ok".to_string(), Err(e) => format!("{e}") });
    (null_max / tr, spanned_min / tr, tau / tr)
}

#[test]
fn the_null_eigenvalues_stay_at_rounding_as_the_store_grows() {
    for n in [300, 3000] {
        let (null, spanned, tau) = margin(n);
        assert!(null < tau / 10.0 && spanned > 10.0 * tau, "{n} dimers: null {null:.1e}, spanned {spanned:.1e}, τ {tau:.1e} (relative to tr)");
    }
}

#[test]
#[ignore = "measurement (30 000 dimers, ~30 s): cargo test -p surrogate --release --test novelty -- --ignored --nocapture"]
fn the_null_eigenvalues_at_thirty_thousand_dimers() {
    let (null, spanned, tau) = margin(30_000);
    assert!(null < tau / 10.0 && spanned > 10.0 * tau, "null {null:.1e}, spanned {spanned:.1e}, τ {tau:.1e}");
}
