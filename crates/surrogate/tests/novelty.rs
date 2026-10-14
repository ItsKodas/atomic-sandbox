//! The unspanned-direction test (`surrogate::novelty`, M6 Task 11): a Morse dimer fit leaves
//! directions no training row reaches; a third atom reaches them, a dimer does not.
use surrogate::{Accumulator, Basis, FitOptions, NullSpace, Provenance, Sample, SampleTag};

fn dimer_fit() -> (Accumulator, surrogate::Model) {
    let basis = Basis::new(&[1]);
    let prov = Provenance { grid_n: 49, h: 0.25, sigma: 0.25, tol: 1e-6, charge: 0, multiplicity: None, teacher: "morse".into(), masses: vec![] };
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
