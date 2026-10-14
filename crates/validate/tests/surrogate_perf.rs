//! Surrogate performance (M5 Task 9 item 5): what one answered step and one refit cost, by
//! species count S, atom count N and stored samples P, and what the variance cadence buys.
//! CPU only (no teacher is ever called), reported, never asserted (ruling R3). Run with
//! `cargo test -p validate --release --test surrogate_perf -- --ignored --nocapture`; the table is
//! printed and written into the gitignored `last-run-surrogate.md` ("Performance" section, kept
//! by the surrogate suite's own rewrites). The numbers of record are in `references.md`
//! ("Surrogate (M5)" → speed).
//!
//! The models are fitted to a synthetic teacher — a sum of pairwise Morse terms whose depth
//! depends on the pair's labels, plus a teacher-like scatter — on random clusters: the costs
//! depend on Q′ (the basis size: body-2 + body-3 at the default degree cap D = 4), N and P, not on
//! what is fitted. The policy is relaxed (thresholds 10 Ha, Ha/Bohr; no envelope; no audits) so
//! every step is answered and the loop measures the model alone.
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use providers::{FallbackPolicy, SurrogateKs};
use sim::{Cell, ForceProvider, MorseDimer, Nucleus, Scene};
use std::fmt::Write as _;
use std::time::{Duration, Instant};
use surrogate::{describe, describe_values, Sample, SampleTag};

const SPECIES: [u8; 4] = [1, 6, 7, 8];

/// E = Σ_{i<j} D_ij [(1 − e^{−(r − 1.6)})² − 1], D_ij = 0.05 + 0.01 (Z_i + Z_j mod 5), with the
/// exact forces (Hartree, Bohr). Synthetic: only its smoothness matters here.
fn teacher(pos: &[[f64; 3]], z: &[u8]) -> (f64, Vec<[f64; 3]>) {
    let mut e = 0.0;
    let mut f = vec![[0.0; 3]; pos.len()];
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            let d: [f64; 3] = std::array::from_fn(|a| pos[j][a] - pos[i][a]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let dd = 0.05 + 0.01 * ((z[i] as u32 + z[j] as u32) % 5) as f64;
            let x = (-(r - 1.6)).exp();
            e += dd * ((1.0 - x) * (1.0 - x) - 1.0);
            let de = 2.0 * dd * (1.0 - x) * x;
            for a in 0..3 {
                f[i][a] += de * d[a] / r;
                f[j][a] -= de * d[a] / r;
            }
        }
    }
    (e, f)
}

/// A teacher-like scatter on a sample (1e-5 Ha on the energy, 1e-4 Ha/Bohr per force component,
/// roughly the GPU teacher's egg-box and SCF floor): without it the evidence search drives the
/// noise levels to their floors and a refit takes many more rounds than one on DFT data.
fn noisy(rng: &mut StdRng, e: f64, f: Vec<[f64; 3]>) -> (f64, Vec<[f64; 3]>) {
    let mut g = || (0..12).map(|_| rng.gen_range(0.0..1.0)).sum::<f64>() - 6.0;
    let e = e + 1e-5 * g();
    let f = f.into_iter().map(|v| [v[0] + 1e-4 * g(), v[1] + 1e-4 * g(), v[2] + 1e-4 * g()]).collect();
    (e, f)
}

/// N atoms of the first `s` species (each present when N ≥ s), uniformly in a ball sized for
/// ≈ 1.6 Bohr nearest neighbours, no pair closer than 1.1 Bohr.
fn cluster(rng: &mut StdRng, n: usize, s: usize) -> (Vec<[f64; 3]>, Vec<u8>) {
    let radius = 1.0 + 1.1 * (n as f64).cbrt();
    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(n);
    while pos.len() < n {
        let p = [rng.gen_range(-radius..radius), rng.gen_range(-radius..radius), rng.gen_range(-radius..radius)];
        let inside = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= radius * radius;
        if inside && pos.iter().all(|q| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>() > 1.21) {
            pos.push(p);
        }
    }
    let z = (0..n).map(|i| if i < s { SPECIES[i] } else { SPECIES[rng.gen_range(0..s)] }).collect();
    (pos, z)
}

fn scene_of(pos: &[[f64; 3]], z: &[u8]) -> Scene {
    Scene {
        nuclei: pos.iter().zip(z).map(|(p, &zz)| Nucleus::new(zz, *p)).collect(),
        cell: Cell { n: 97, h: 0.25 },
        charge: 0,
        multiplicity: None,
        sigma: None,
        external_field: [0.0; 3],
    }
}

fn relaxed(cadence: usize) -> FallbackPolicy {
    FallbackPolicy {
        force_std_abs: 10.0,
        force_std_rel: 0.0,
        energy_std_per_atom: 10.0,
        max_descriptor_distance: 1e9,
        variance_cadence: cadence,
        audit_every: 0,
        never_fallback: false,
    }
}

/// One synthetic sample of an `n`-atom cluster over `s` species.
fn sample(rng: &mut StdRng, n: usize, s: usize, tag: SampleTag) -> (Sample, Scene) {
    let (pos, z) = cluster(rng, n, s);
    let (e, f) = teacher(&pos, &z);
    let (e, f) = noisy(rng, e - 0.5 * z.len() as f64, f);
    let sc = scene_of(&pos, &z);
    (Sample { pos, z, energy: e, forces: f, tag, weight: 1.0 }, sc)
}

/// A surrogate with `p` samples of `n_train`-atom clusters over `s` species; the wall of its first
/// refit (accumulate + evidence search + solve + the training-residual pass) and Q′.
fn trained(s: usize, n_train: usize, p: usize, seed: u64) -> (SurrogateKs, Duration, usize) {
    let mut sk = SurrogateKs::new(Box::new(MorseDimer::h2()), relaxed(1)).with_teacher_tol(1e-6);
    for &z in &SPECIES[..s] {
        sk.set_reference_energy(z, -0.5);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..p {
        let (smp, sc) = sample(&mut rng, n_train, s, SampleTag::Scan);
        sk.add_sample(smp, &sc).unwrap();
    }
    let t = Instant::now();
    let rep = sk.refit().expect("refit");
    (sk, t.elapsed(), rep.q_prime)
}

/// Answered steps per second of `SurrogateKs::evaluate` on an N-atom cluster jittered every step
/// (≥ 0.3 s of steps), with the given variance cadence.
fn steps_per_s(sk: &mut SurrogateKs, n: usize, s: usize, cadence: usize, seed: u64) -> f64 {
    sk.set_policy(relaxed(cadence));
    let mut rng = StdRng::seed_from_u64(seed);
    let (pos, z) = cluster(&mut rng, n, s);
    let mut sc = scene_of(&pos, &z);
    let mut never = || false;
    let t = Instant::now();
    let mut steps = 0usize;
    while t.elapsed() < Duration::from_millis(300) || steps < 20 {
        for nu in &mut sc.nuclei {
            for x in &mut nu.pos {
                *x += 1e-4 * rng.gen_range(-1.0..1.0);
            }
        }
        let r = sk.evaluate(&sc, &mut never).expect("answered");
        assert_eq!(r.stats.source, sim::Source::Surrogate, "{:?}", r.stats.notes);
        steps += 1;
    }
    steps as f64 / t.elapsed().as_secs_f64()
}

/// (predict without variance, predict with variance, describe with gradients, values only) in
/// µs on N-atom clusters.
fn costs(sk: &SurrogateKs, n: usize, s: usize, seed: u64) -> (f64, f64, f64, f64) {
    let model = sk.model().unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let geoms: Vec<_> = (0..8).map(|_| cluster(&mut rng, n, s)).collect();
    let time = |f: &mut dyn FnMut()| {
        let t = Instant::now();
        let mut k = 0;
        while t.elapsed() < Duration::from_millis(150) || k < 8 {
            f();
            k += 1;
        }
        t.elapsed().as_secs_f64() * 1e6 / k as f64
    };
    let mut i = 0;
    let plain = time(&mut || {
        let (p, z) = &geoms[i % 8];
        std::hint::black_box(model.predict(p, z, false).unwrap());
        i += 1;
    });
    let with_var = time(&mut || {
        let (p, z) = &geoms[i % 8];
        std::hint::black_box(model.predict(p, z, true).unwrap());
        i += 1;
    });
    let desc = time(&mut || {
        let (p, z) = &geoms[i % 8];
        std::hint::black_box(describe(&model.basis, p, z).unwrap());
        i += 1;
    });
    let values = time(&mut || {
        let (p, z) = &geoms[i % 8];
        std::hint::black_box(describe_values(&model.basis, p, z).unwrap());
        i += 1;
    });
    (plain, with_var, desc, values)
}

/// Writes `section` as the "Performance" section of `last-run-surrogate.md`, keeping the rest.
fn write_section(section: &str) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/last-run-surrogate.md");
    let old = std::fs::read_to_string(path).unwrap_or_default();
    let head = old.split("\n## Performance").next().unwrap_or("").trim_end().to_string();
    std::fs::write(path, format!("{head}\n\n{section}")).unwrap_or_else(|e| panic!("writing {path}: {e}"));
}

#[test]
#[ignore = "timing table (CPU, ~5 min): cargo test -p validate --release --test surrogate_perf -- --ignored --nocapture"]
fn surrogate_perf_table() {
    assert!(!cfg!(debug_assertions), "timings in a debug build mean nothing: run with --release");
    let mut md = String::new();
    let _ = writeln!(md, "## Performance (`surrogate_perf`, {} threads available)\n", std::thread::available_parallelism().map_or(1, |n| n.get()));
    let _ = writeln!(
        md,
        "Synthetic pair-Morse teacher (+ a 1e-5 Ha / 1e-4 Ha/Bohr scatter) on random clusters; body-2 + body-3 at D = 4; the relaxed policy answers every step. \
         Steps/s = `SurrogateKs::evaluate` (predict + the envelope's value-only describe) on an N-atom cluster jittered each step, at variance cadence k; \
         \"+ variance\" = the extra cost of `predict(with_variance)`; \"describe\" with gradients (the fit's), \"values\" without (the envelope's).\n"
    );
    let _ = writeln!(md, "| S | Q′ | P (train N = 6) | first refit ms | N | predict µs | + variance µs | describe µs | values µs | steps/s k=1 | k=4 | k=10 |");
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|---|");
    for s in [1usize, 2, 4] {
        let q_guess = [0, 29, 176, 0, 1232][s];
        let p = q_guess + 60;
        let (mut sk, fit, q) = trained(s, 6, p, 11 + s as u64);
        for n in [2usize, 6, 20] {
            let n = n.max(s);
            let (plain, var, desc, values) = costs(&sk, n, s, 100 + n as u64);
            let k1 = steps_per_s(&mut sk, n, s, 1, 7);
            let k4 = steps_per_s(&mut sk, n, s, 4, 7);
            let k10 = steps_per_s(&mut sk, n, s, 10, 7);
            let row = format!(
                "| {s} | {q} | {p} | {:.0} | {n} | {plain:.1} | {:.1} | {desc:.1} | {values:.1} | {k1:.0} | {k4:.0} | {k10:.0} |",
                fit.as_secs_f64() * 1e3,
                var - plain
            );
            println!("{row}");
            let _ = writeln!(md, "{row}");
        }
    }
    let _ = writeln!(md, "\nRefit wall vs stored samples (S = 1, Q′ 29, H₂-like 2-atom samples; S = 2, Q′ 176, 6-atom samples): the first fit over the store, then an incremental refit after 25 more samples (the viewer's cadence).\n");
    let _ = writeln!(md, "| S | P | first refit ms | +25 samples refit ms |\n|---|---|---|---|");
    for (s, n_train) in [(1usize, 2usize), (2, 6)] {
        for p in [250usize, 500, 1000, 2000] {
            let (mut sk, fit, _) = trained(s, n_train, p, 31);
            let mut rng = StdRng::seed_from_u64(77);
            for _ in 0..25 {
                let (smp, sc) = sample(&mut rng, n_train, s, SampleTag::Fallback);
                sk.add_sample(smp, &sc).unwrap();
            }
            let t = Instant::now();
            sk.refit().unwrap();
            let inc = t.elapsed();
            let row = format!("| {s} | {p} | {:.0} | {:.0} |", fit.as_secs_f64() * 1e3, inc.as_secs_f64() * 1e3);
            println!("{row}");
            let _ = writeln!(md, "{row}");
        }
    }
    write_section(&md);
}
