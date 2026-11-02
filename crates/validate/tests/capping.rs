//! Capped segments train like the whole molecule (M8a Task 10, "Train this segment"; spec §4.5's
//! condition, signed off by the user 2026-10-20: "happy with that if it trains the same behaviour
//! outcomes"). Atomic units throughout (Hartree, Bohr).
//!
//! # What is claimed
//!
//! A segment is a capped cut-out of a larger molecule: the selected atoms, every atom within r_cut
//! of them, and a hydrogen on every bond the cut breaks (ruling P15: the cut grows to a fixed point
//! so that every selected atom is **interior** — every source atom within r_cut of it is in the
//! segment and no cap is). Its teacher answers teach only the interior atoms' forces and no energy
//! (`surrogate::fit`, "Masked samples"; the provider's keep path, `providers::surrogate`, "Masked
//! scenes"). The spec's condition: a collection taught on capped segments of a molecule reproduces
//! fresh DFT points of the *whole* molecule as well as a collection taught on the whole molecule's
//! own samples — errors within **1.5×** of each other — or capping is replaced by the uncapped
//! two-shell cut.
//!
//! The molecule is n-butane (ruling P7: it stands in for the spec's polyethylene; the water trimer
//! is not built, ruling P20), on the app's default surface: Accurate grid (h = σ = 0.15 Bohr),
//! `ionic accepted`, the GPU teacher at the sampling tolerance 1e-6, the shared collection's body-4
//! basis at `surrogate::COLLECTION_R_CUT` (6 Bohr). Butane at 6 Bohr: an inner carbon sees all four
//! carbons, so only an **end carbon** can be the centre of a segment smaller than the molecule —
//! selecting C1 keeps C1–C3 with their hydrogens, caps the cut C3–C4 bond (6.6 Bohr from C1, beyond
//! r_cut) and leaves a propane-like segment whose interior is C1 and the hydrogens whose own
//! neighbourhood is whole.
//!
//! - **Training geometries**: `CAPPING_TRAIN` (default 40) Gaussian displacements of the relaxed
//!   molecule (σ = 0.15 Bohr per coordinate, the sampling pass's perturbation size), seeded.
//! - **Collection A** (the reference method): the whole molecule's DFT answer at each — energy and
//!   every atom's forces.
//! - **Collection B** (capping): at each geometry the two end segments (select C1; select C4), cut
//!   from that geometry, capped, solved, kept through the provider's keep path with the segment
//!   scene's `train_mask` (= interior) — interior forces only, no energy.
//! - **Control** (CPU, no extra solve): the whole molecule's answers with the force mask set to the
//!   **interior class** and no energy — B's rows without the capping, so B against the control
//!   isolates the capping error from the data volume.
//! - **Held out**: `HELD_OUT` (20) more displacements of the whole molecule (another seed).
//! - **Interior class**: the source atoms that are interior in the end cuts of the relaxed geometry
//!   (the atoms Train-this-segment teaches). The criterion: B's per-component force RMSE on the
//!   interior-class atoms of the held-out geometries ≤ `FACTOR` (1.5) × A's. Both are printed with
//!   the control's, A's RMSE over all atoms and the trusted counts (`SurrogateKs::atom_verdicts`).
//!   If A itself trusts nothing, the comparison stands: it measures whether the two **training
//!   methods** are equivalent, not whether either model is trusted (the output says so).
//!
//! The cut is **test-local** (ruling P9): bonds from the molecule's own geometry (each H to its
//! nearest heavy atom, heavy pairs within 1.25 × the shortest heavy–heavy distance) and the cap
//! length from the molecule's own mean C–H distance — the builder's tables never leave the app
//! (`app::segment`, Task 10b, has its own). A dropped H bonded to a kept atom is kept (a cap there
//! would be the same H); a heavy atom bonded to a kept H is kept.
//!
//! # Cost — not yet run (2026-11-02)
//!
//! `capping_cost_probe` (RX 7900 XTX, `ollama` resident and idle, nothing else on the GPU): the
//! start butane in its 125³ box (margin 4 Bohr at h = 0.15) solves in 159 s cold with the
//! pseudopotential generation, 95 s warm after a 0.05 Bohr move (13 SCF), 90 s cold after a reset
//! (15 SCF); an end segment (11 atoms) in the same box 48 s cold, 37 s warm. Estimate of the GPU
//! wall: the gate, `train` × (whole + 2 segments) + 20 held-out, = 1.5 h for 20 training geometries
//! and 2.6 h for 40 at the probe's times (3.8 h / 6.4 h at Task 6's measured 2.5× run-over-probe
//! factor); the relaxation that must come first (ruling P7), ≈ 100–200 FIRE steps × 95 s = 2.6–5.3 h.
//! Over the task's 3 h budget, so the GPU tests are `#[ignore]`d and were not run (Task 10a report).
//! Cheaper variants, not taken without the controller: the segments in their own smaller box (a
//! second reference set; ≈ 107³), fewer training geometries.
//!
//! # Running it
//!
//! GPU, hours, `#[ignore]`d — run alone: first `relax_butane` (writes `scenes/butane.scene`), then
//! `cargo test -p validate --release --test capping -- --ignored --nocapture --test-threads 1 capped_segments`.
//! The answers are kept in collections under the target's tmp dir and a rerun continues where the
//! last one stopped (`CAPPING_REUSE=1`; without it the folders are cleared). `capping_cost_probe`
//! times one solve of each kind for the estimate. The cut itself is checked on the CPU
//! (`the_test_cut_keeps_the_selected_atoms_interior`, not ignored).
use gpu::GpuContext;
use providers::SurrogateKs;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sim::{Cell, Fire, FireOutcome, FireParams, ForceProvider, IonicPolicy, Nucleus, Scene};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;
use surrogate::{Collection, CollectionSample, Sample, SampleTag, COLLECTION_R_CUT};

mod common;
use common::surrogate::{class_summary, fit_summary, read_scene, scene_text, scenes_dir, student};
use common::{never, no_augment, solve, Locked, DT_FS, TOL_H2};

/// The Accurate grid spacing (= σ) of the app's default surface.
const H_ACC: f64 = 0.15;
/// The box margin (Bohr): the shipped propane's (16 Bohr box, |x| ≤ 4.1) and the transfer test's
/// target margin.
const MARGIN: f64 = 4.0;
const HELD_OUT: usize = 20;
/// Displacement per coordinate (Bohr): the sampling pass's perturbation σ.
const SIGMA: f64 = 0.15;
const SEED_TRAIN: u64 = 0xb07a_0001;
const SEED_HELD: u64 = 0xb07a_0002;
/// B's interior force RMSE over A's (spec §4.5).
const FACTOR: f64 = 1.5;
/// The default number of training geometries (`CAPPING_TRAIN`).
const TRAIN_DEFAULT: usize = 40;
/// In-run solve time over the probe's warm solve (Task 6 measured 2.5×: ≈ 20 s per C₂H₆ solve in
/// its passes against an 8 s probe; Task 6 report): the estimate's factor.
const RUN_OVER_PROBE: f64 = 2.5;

fn gpu() -> Option<GpuContext> {
    common::gpu("capping")
}

fn accurate(n: usize, nuclei: Vec<Nucleus>) -> Scene {
    Scene { nuclei, cell: Cell { n, h: H_ACC }, charge: 0, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: IonicPolicy::Accepted, train_mask: None }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] + b[k])
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|x| x * s)
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / norm(a))
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

/// The smallest odd n with (n − 1)·h ≥ 2(max |x| + `margin`), the nuclei centred first.
fn boxed(mut nuclei: Vec<Nucleus>, margin: f64) -> Scene {
    for k in 0..3 {
        let (lo, hi) = nuclei.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), n| (l.min(n.pos[k]), h.max(n.pos[k])));
        let c = 0.5 * (lo + hi);
        for n in &mut nuclei {
            n.pos[k] -= c;
        }
    }
    let m = nuclei.iter().flat_map(|n| n.pos).fold(0.0f64, |m, x| m.max(x.abs()));
    let mut n = ((2.0 * (m + margin)) / H_ACC).ceil() as usize + 1;
    if n % 2 == 0 {
        n += 1;
    }
    accurate(n, nuclei)
}

/// The start of the butane relaxation: anti (C₂h), all staggered, from rounded textbook bond
/// parameters — C–C 1.53 Å and ∠CCC 113° (about the gas-phase electron-diffraction structure of
/// n-butane, Bradford, Fitzwater & Bartell, J. Mol. Struct. 38, 185 (1977)), C–H 1.09 Å, the
/// methyl H at the tetrahedral angle to the C–C bond, ∠HCH 107° on the CH₂. A start built from these
/// parameters, not a literature Cartesian set: FIRE on this surface decides the geometry.
/// Atom order: C1 C2 C3 C4, then the H of C1, C2, C3, C4.
fn butane_start() -> Scene {
    let b = units::angstrom_to_bohr(1.53);
    let r_ch = units::angstrom_to_bohr(1.09);
    let half = ((180.0 - 113.0) / 2.0f64).to_radians();
    let da = [half.cos(), half.sin(), 0.0];
    let db = [half.cos(), -half.sin(), 0.0];
    let c1 = [0.0; 3];
    let c2 = add(c1, scale(da, b));
    let c3 = add(c2, scale(db, b));
    let c4 = add(c3, scale(da, b));
    let mut nuclei: Vec<Nucleus> = [c1, c2, c3, c4].iter().map(|&p| Nucleus::new(6, p)).collect();
    // a methyl on `c` bonded to `nb`, staggered: one H anti to `far` (the next carbon along)
    let methyl = |c: [f64; 3], nb: [f64; 3], far: [f64; 3]| -> Vec<[f64; 3]> {
        let u = unit(sub(nb, c));
        let w = sub(far, nb);
        let e1 = unit(scale(sub(w, scale(u, dot(w, u))), -1.0));
        let e2 = cross(u, e1);
        let (ct, st) = (-1.0 / 3.0, (8.0f64).sqrt() / 3.0);
        (0..3)
            .map(|k| {
                let p = (120.0 * k as f64).to_radians();
                add(c, scale(add(scale(u, ct), scale(add(scale(e1, p.cos()), scale(e2, p.sin())), st)), r_ch))
            })
            .collect()
    };
    // a methylene on `c` bonded to `a` and `bb`: the two H in the plane bisecting ∠(a c bb)
    let methylene = |c: [f64; 3], a: [f64; 3], bb: [f64; 3]| -> Vec<[f64; 3]> {
        let (u1, u2) = (unit(sub(a, c)), unit(sub(bb, c)));
        let bis = unit(scale(add(u1, u2), -1.0));
        let n = unit(cross(u1, u2));
        let h = (107.0f64 / 2.0).to_radians();
        [1.0, -1.0].iter().map(|s| add(c, scale(add(scale(bis, h.cos()), scale(n, s * h.sin())), r_ch))).collect()
    };
    let hs: Vec<[f64; 3]> = [methyl(c1, c2, c3), methylene(c2, c1, c3), methylene(c3, c2, c4), methyl(c4, c3, c2)].concat();
    nuclei.extend(hs.into_iter().map(|p| Nucleus::new(1, p)));
    boxed(nuclei, MARGIN)
}

/// The relaxed butane (`scenes/butane.scene`, written by `relax_butane`).
fn butane() -> Scene {
    let path = scenes_dir().join("butane.scene");
    assert!(path.exists(), "{} is missing: run `cargo test -p validate --release --test capping -- --ignored --nocapture relax_butane` first", path.display());
    read_scene("butane.scene")
}

/// The molecule's bonds from its own geometry (module doc): each H to its nearest heavy atom,
/// heavy pairs within 1.25 × the shortest heavy–heavy distance. Z only (H or not).
fn bonds(sc: &Scene) -> Vec<(usize, usize)> {
    let nuc = &sc.nuclei;
    let heavy: Vec<usize> = (0..nuc.len()).filter(|&i| nuc[i].z > 1).collect();
    let mut out = Vec::new();
    for i in (0..nuc.len()).filter(|&i| nuc[i].z == 1) {
        let j = *heavy.iter().min_by(|&&a, &&b| dist(nuc[i].pos, nuc[a].pos).total_cmp(&dist(nuc[i].pos, nuc[b].pos))).expect("a heavy atom");
        out.push((j.min(i), j.max(i)));
    }
    let shortest = heavy.iter().flat_map(|&a| heavy.iter().filter(move |&&b| b > a).map(move |&b| (a, b))).map(|(a, b)| dist(nuc[a].pos, nuc[b].pos)).fold(f64::INFINITY, f64::min);
    for (ia, &a) in heavy.iter().enumerate() {
        for &b in &heavy[ia + 1..] {
            if dist(nuc[a].pos, nuc[b].pos) <= 1.25 * shortest {
                out.push((a, b));
            }
        }
    }
    out
}

/// The mean heavy-atom–H bond length of the molecule (the cap length, ruling P9).
fn cap_length(sc: &Scene, bonds: &[(usize, usize)]) -> f64 {
    let d: Vec<f64> = bonds.iter().filter(|&&(a, b)| (sc.nuclei[a].z == 1) != (sc.nuclei[b].z == 1)).map(|&(a, b)| dist(sc.nuclei[a].pos, sc.nuclei[b].pos)).collect();
    d.iter().sum::<f64>() / d.len() as f64
}

/// A capped segment (module doc): its scene (the kept atoms in source order, then the caps; the
/// source's box), its source index per atom (`None` for a cap) and its interior flags (the scene's
/// `train_mask`).
struct Cut {
    scene: Scene,
    source: Vec<Option<usize>>,
    interior: Vec<bool>,
}

/// The cut of `sc` around `selected` (module doc; ruling P15): kept = selected ∪ every atom within
/// r_cut of a selected one, grown to a fixed point by (a) a dropped H bonded to a kept atom, (b) a
/// dropped heavy atom bonded to a kept H, (c) a dropped heavy atom whose cap would land within r_cut
/// of a selected atom; then one H cap per cut heavy–heavy bond at `d_cap` along it. Interior: every
/// source atom within r_cut is kept and no cap is within r_cut. Asserts every selected atom is.
fn cut(sc: &Scene, bonds: &[(usize, usize)], selected: &[usize], r_cut: f64, d_cap: f64) -> Cut {
    let nuc = &sc.nuclei;
    let n = nuc.len();
    let mut kept = vec![false; n];
    for &s in selected {
        for i in 0..n {
            if dist(nuc[s].pos, nuc[i].pos) < r_cut {
                kept[i] = true;
            }
        }
    }
    let cap_of = |k: usize, d: usize| add(nuc[k].pos, scale(unit(sub(nuc[d].pos, nuc[k].pos)), d_cap));
    loop {
        let mut grew = false;
        for &(a, b) in bonds {
            for (k, d) in [(a, b), (b, a)] {
                if !kept[k] || kept[d] {
                    continue;
                }
                let keep = nuc[d].z == 1 || nuc[k].z == 1 || selected.iter().any(|&s| dist(nuc[s].pos, cap_of(k, d)) < r_cut);
                if keep {
                    kept[d] = true;
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    let caps: Vec<[f64; 3]> = bonds
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .filter(|&(k, d)| kept[k] && !kept[d])
        .map(|(k, d)| cap_of(k, d))
        .collect();
    let mut nuclei = Vec::new();
    let mut source = Vec::new();
    let mut interior = Vec::new();
    for i in (0..n).filter(|&i| kept[i]) {
        let whole = (0..n).all(|j| kept[j] || dist(nuc[i].pos, nuc[j].pos) >= r_cut);
        let capped = caps.iter().any(|&c| dist(nuc[i].pos, c) < r_cut);
        nuclei.push(Nucleus::new(nuc[i].z, nuc[i].pos));
        source.push(Some(i));
        interior.push(whole && !capped);
    }
    for &c in &caps {
        nuclei.push(Nucleus::new(1, c));
        source.push(None);
        interior.push(false);
    }
    for &s in selected {
        let k = source.iter().position(|&x| x == Some(s)).expect("a selected atom is kept");
        assert!(interior[k], "selected atom {s} is not interior after the fixed point");
    }
    let mut scene = sc.clone();
    scene.nuclei = nuclei;
    scene.train_mask = Some(interior.clone());
    Cut { scene, source, interior }
}

/// The end carbons: heavy atoms with exactly one heavy neighbour (butane's C1 and C4).
fn ends(sc: &Scene, bonds: &[(usize, usize)]) -> Vec<usize> {
    let heavy = |i: usize| sc.nuclei[i].z > 1;
    (0..sc.nuclei.len()).filter(|&i| heavy(i) && bonds.iter().filter(|&&(a, b)| heavy(a) && heavy(b) && (a == i || b == i)).count() == 1).collect()
}

/// The interior class (module doc): the source atoms interior in the end cuts of `sc`.
fn interior_class(sc: &Scene, bonds: &[(usize, usize)], d_cap: f64) -> Vec<bool> {
    let mut class = vec![false; sc.nuclei.len()];
    for e in ends(sc, bonds) {
        let c = cut(sc, bonds, &[e], COLLECTION_R_CUT, d_cap);
        for (s, &int) in c.source.iter().zip(&c.interior) {
            if let (Some(s), true) = (s, int) {
                class[*s] = true;
            }
        }
    }
    class
}

fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `count` displacements of `sc` (N(0, σ²) per coordinate), seeded; a draw leaving the cell's
/// margin is redrawn.
fn displaced(sc: &Scene, count: usize, seed: u64) -> Vec<Scene> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let mut g = sc.clone();
        for n in &mut g.nuclei {
            for k in 0..3 {
                n.pos[k] += SIGMA * normal(&mut rng);
            }
        }
        if g.check_inside_cell().is_ok() {
            out.push(g);
        }
    }
    out
}

/// Prints the log so far (a run of hours must not lose it to a later panic) and clears it.
fn flush(out: &mut String) {
    print!("{out}");
    out.clear();
}

/// The cut on the CPU (no GPU, not ignored): on the start geometry (or the relaxed one when
/// written) each end cut keeps its selected carbon interior, caps exactly the cut C–C bond at the
/// molecule's own C–H length, marks no cap and no atom with a cap or a dropped atom within r_cut
/// interior, and yields a closed-shell segment (even electron count). The interior class is printed.
#[test]
fn the_test_cut_keeps_the_selected_atoms_interior() {
    let sc = if scenes_dir().join("butane.scene").exists() { butane() } else { butane_start() };
    let b = bonds(&sc);
    assert_eq!(b.len(), 13, "butane has 13 bonds (3 C–C, 10 C–H)");
    let d_cap = cap_length(&sc, &b);
    let e = ends(&sc, &b);
    assert_eq!(e.len(), 2, "two end carbons");
    for &end in &e {
        let c = cut(&sc, &b, &[end], COLLECTION_R_CUT, d_cap);
        let caps: Vec<usize> = (0..c.source.len()).filter(|&k| c.source[k].is_none()).collect();
        let carbons = c.scene.nuclei.iter().filter(|n| n.z == 6).count();
        assert_eq!((carbons, caps.len()), (3, 1), "end {end}: a C3 segment with one cap on the cut C–C bond");
        let electrons: u32 = c.scene.nuclei.iter().map(|n| n.z as u32).sum();
        assert_eq!(electrons % 2, 0, "closed shell");
        // the cap at the C–H length from its carbon, on the cut bond
        let cap = c.scene.nuclei[caps[0]].pos;
        let near = c.scene.nuclei.iter().filter(|n| n.z == 6).map(|n| dist(n.pos, cap)).fold(f64::INFINITY, f64::min);
        assert!((near - d_cap).abs() < 1e-12, "cap {near} vs {d_cap}");
        // interior: the definition, checked independently
        for (k, &int) in c.interior.iter().enumerate() {
            let p = c.scene.nuclei[k].pos;
            let cap_near = caps.iter().any(|&q| dist(c.scene.nuclei[q].pos, p) < COLLECTION_R_CUT);
            let dropped_near = c.source[k].is_some() && (0..sc.nuclei.len()).any(|j| !c.source.contains(&Some(j)) && dist(sc.nuclei[j].pos, p) < COLLECTION_R_CUT);
            assert_eq!(int, c.source[k].is_some() && !cap_near && !dropped_near, "atom {k} of the end-{end} segment");
        }
        assert_eq!(c.scene.train_mask.as_deref(), Some(&c.interior[..]));
        let names: Vec<String> = c.source.iter().zip(&c.interior).filter(|(_, &i)| i).map(|(s, _)| format!("{}{}", if sc.nuclei[s.unwrap()].z == 6 { "C" } else { "H" }, s.unwrap())).collect();
        println!("end {end}: segment of {} atoms ({} kept + 1 cap at {d_cap:.4} Bohr), interior {names:?}", c.scene.nuclei.len(), c.scene.nuclei.len() - 1);
    }
    let class = interior_class(&sc, &b, d_cap);
    println!("interior class: {:?}", (0..class.len()).filter(|&i| class[i]).collect::<Vec<_>>());
    assert!(e.iter().all(|&i| class[i]), "the end carbons are interior");
}

/// A teacher answer as a whole sample.
fn sample_of(sc: &Scene, energy: f64, forces: Vec<[f64; 3]>) -> Sample {
    Sample::new(sc.nuclei.iter().map(|n| n.pos).collect(), sc.nuclei.iter().map(|n| n.z).collect(), energy, forces, SampleTag::Perturbation, 1.0)
}

/// What one collection's model does on the held-out geometries.
struct Judged {
    /// Per-component force RMSE over the interior-class atoms, and over every atom.
    f_class: f64,
    f_all: f64,
    /// Interior-class atoms the strict policy trusts, of all interior-class atoms; geometries answered.
    trusted_class: usize,
    class_atoms: usize,
    answered: usize,
}

fn judge(s: &SurrogateKs, held: &[Scene], dft: &[(f64, Vec<[f64; 3]>)], class: &[bool]) -> Judged {
    let model = s.model().expect("fitted");
    let (mut sc_, mut nc, mut sa, mut na, mut trusted, mut atoms, mut answered) = (0.0, 0usize, 0.0, 0usize, 0usize, 0usize, 0usize);
    for (g, (_, f)) in held.iter().zip(dft) {
        let pos: Vec<[f64; 3]> = g.nuclei.iter().map(|n| n.pos).collect();
        let z: Vec<u8> = g.nuclei.iter().map(|n| n.z).collect();
        let (_, fm) = surrogate::energy_and_forces(model, &pos, &z).expect("describe");
        for (i, (a, b)) in fm.iter().zip(f).enumerate() {
            let d2: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum();
            sa += d2;
            na += 3;
            if class[i] {
                sc_ += d2;
                nc += 3;
            }
        }
        if let Some(v) = s.atom_verdicts(g) {
            answered += v.answers() as usize;
            for i in (0..class.len()).filter(|&i| class[i]) {
                atoms += 1;
                trusted += v.trusted(i) as usize;
            }
        }
    }
    Judged { f_class: (sc_ / nc.max(1) as f64).sqrt(), f_all: (sa / na.max(1) as f64).sqrt(), trusted_class: trusted, class_atoms: atoms, answered }
}

/// The gate (module doc).
#[test]
#[ignore = "GPU, hours (not yet run; probe estimate 1.5–2.6 h at 20–40 training geometries, up to 6.4 h at Task 6's 2.5× run factor, after relax_butane): cargo test -p validate --release --test capping -- --ignored --nocapture --test-threads 1 capped_segments"]
fn capped_segments_train_like_the_whole_molecule() {
    let Some(ctx) = gpu() else { return };
    let mut out = String::new();
    let train_n: usize = std::env::var("CAPPING_TRAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(TRAIN_DEFAULT);
    let reuse = std::env::var("CAPPING_REUSE").is_ok_and(|v| v == "1");
    let base = butane();
    let b = bonds(&base);
    let d_cap = cap_length(&base, &b);
    let class = interior_class(&base, &b, d_cap);
    let end_atoms = ends(&base, &b);
    let _ = writeln!(out, "capping gate: butane ({} atoms, {}³ × {H_ACC}), r_cut {COLLECTION_R_CUT}, cap length {d_cap:.4} Bohr (the molecule's mean C–H), interior class {:?}, {train_n} training + {HELD_OUT} held-out geometries (σ {SIGMA}), reuse {reuse}",
                     base.nuclei.len(), base.cell.n, (0..class.len()).filter(|&i| class[i]).collect::<Vec<_>>());
    flush(&mut out);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("capping");
    let roots = ["whole", "segments", "control", "heldout"].map(|n| tmp.join(n));
    if !reuse {
        let _ = std::fs::remove_dir_all(&tmp);
    }
    for r in &roots {
        std::fs::create_dir_all(r).expect("collection root");
    }
    let (a0, log) = student(&ctx, 0);
    let mut a = a0.with_collection_r_cut(COLLECTION_R_CUT);
    a.attach_collection_at(&roots[0], &base).unwrap_or_else(|e| panic!("attach A: {e}"));
    let (bb, _) = student(&ctx, 0);
    let mut bs = bb.with_collection_r_cut(COLLECTION_R_CUT);
    bs.attach_collection_at(&roots[1], &base).unwrap_or_else(|e| panic!("attach B: {e}"));
    let (cc, _) = student(&ctx, 0);
    let mut ctl = cc.with_collection_r_cut(COLLECTION_R_CUT);
    ctl.attach_collection_at(&roots[2], &base).unwrap_or_else(|e| panic!("attach control: {e}"));
    let key = a.provenance_for(&base).surface_key();
    let wall = Instant::now();

    // ---- references: one box for everything (the segments keep their source coordinates) ----
    let refs_have = |s: &SurrogateKs, z: u8| s.collection().is_some_and(|c| c.references.contains_key(&(z, base.cell.n)));
    for z in [1u8, 6] {
        if [&a, &bs, &ctl].iter().all(|s| refs_have(*s, z)) {
            continue;
        }
        let mass = Nucleus::new(z, [0.0; 3]).mass;
        let r = sim::solve_atom(a.teacher_mut(), &base.cell, None, IonicPolicy::Accepted, z, mass, &mut never()).unwrap_or_else(|e| panic!("reference Z = {z}: {e}"));
        for s in [&mut a, &mut bs, &mut ctl] {
            s.set_reference_solved(z, r.energy, base.cell.n, r.face_density);
        }
        let _ = writeln!(out, "  reference Z = {z} in {}³: {:.6} Ha (face density {:?})", base.cell.n, r.energy, r.face_density);
    }
    flush(&mut out);

    // ---- the training answers: whole (A, and the control from the same answer), two segments (B) ----
    let rule = providers::box_affected_face_density(H_ACC);
    let geoms = displaced(&base, train_n, SEED_TRAIN);
    let mut seg_atoms = 0usize;
    let mut seg_interior = 0usize;
    for (i, g) in geoms.iter().enumerate() {
        let done_a = a.collection().map_or(0, |c| c.samples.len());
        if i >= done_a {
            let t = Instant::now();
            a.teacher_mut().reset();
            let r = solve(a.teacher_mut(), g, &format!("whole {i}"));
            assert!(r.stats.face_density.is_none_or(|f| f <= rule), "whole {i}: box-affected (face density {:?} > {rule:e})", r.stats.face_density);
            a.add_sample_answer(sample_of(g, r.energy, r.forces), g, r.stats.face_density, r.stats.multiplicity).unwrap_or_else(|e| panic!("A refused whole {i}: {e}"));
            let _ = writeln!(out, "  whole {i}: E {:.6} Ha, {} SCF, {:.1} s", r.energy, r.stats.scf_steps, t.elapsed().as_secs_f64());
        }
        // the control: the same whole answer, B's rows (the interior class, no energy)
        if ctl.collection().map_or(0, |c| c.samples.len()) <= i {
            let s = a.collection().expect("attached").samples[i].clone();
            let mut gm = g.clone();
            gm.train_mask = Some(class.clone());
            // (its face density was below the box rule when solved — asserted above, not stored)
            ctl.add_sample_answer(s.sample.clone(), &gm, Some(0.0), s.multiplicity).unwrap_or_else(|e| panic!("control {i}: {e}"));
        }
        for (k, &end) in end_atoms.iter().enumerate() {
            let c = cut(g, &b, &[end], COLLECTION_R_CUT, d_cap);
            seg_atoms += c.scene.nuclei.len();
            seg_interior += c.interior.iter().filter(|&&x| x).count();
            if bs.collection().map_or(0, |col| col.samples.len()) > 2 * i + k {
                continue;
            }
            let t = Instant::now();
            a.teacher_mut().reset();
            let r = solve(a.teacher_mut(), &c.scene, &format!("segment {i}/{k}"));
            assert!(r.stats.face_density.is_none_or(|f| f <= rule), "segment {i}/{k}: box-affected");
            // through the provider's keep path: the scene's train_mask masks the sample (ruling P10)
            bs.add_sample_answer(sample_of(&c.scene, r.energy, r.forces), &c.scene, r.stats.face_density, r.stats.multiplicity).unwrap_or_else(|e| panic!("B refused segment {i}/{k}: {e}"));
            let stored = &bs.collection().expect("attached").samples.last().expect("kept").sample;
            assert_eq!((stored.force_mask.as_deref(), stored.use_energy), (Some(&c.interior[..]), false), "the keep path masked the segment");
            let _ = writeln!(out, "  segment {i}/{k}: {} atoms ({} interior), {} SCF, {:.1} s", c.scene.nuclei.len(), c.interior.iter().filter(|&&x| x).count(), r.stats.scf_steps, t.elapsed().as_secs_f64());
        }
        flush(&mut out);
    }
    let _ = writeln!(out, "  training answers: {} (wall so far {:.0} s); segments average {:.1} atoms, {:.2} interior", log.lock().unwrap().summary(), wall.elapsed().as_secs_f64(),
                     seg_atoms as f64 / (2 * train_n) as f64, seg_interior as f64 / (2 * train_n) as f64);
    flush(&mut out);

    // ---- the held-out answers (kept in their own collection, so a rerun reuses them) ----
    let held = displaced(&base, HELD_OUT, SEED_HELD);
    let mut dft: Vec<(f64, Vec<[f64; 3]>)> = Vec::new();
    {
        let (mut hc, _) = Collection::open(&roots[3], &key).unwrap_or_else(|e| panic!("held-out collection: {e}"));
        for (j, g) in held.iter().enumerate() {
            if j >= hc.samples.len() {
                let t = Instant::now();
                a.teacher_mut().reset();
                let r = solve(a.teacher_mut(), g, &format!("held-out {j}"));
                hc.append(CollectionSample { sample: sample_of(g, r.energy, r.forces), grid_n: g.cell.n, multiplicity_forced: false, multiplicity: r.stats.multiplicity, scene: "butane held-out".into() }).unwrap_or_else(|e| panic!("held-out {j}: {e}"));
                let _ = writeln!(out, "  held-out {j}: E {:.6} Ha, {:.1} s", r.energy, t.elapsed().as_secs_f64());
                flush(&mut out);
            }
            let s = &hc.samples[j].sample;
            assert_eq!(s.pos, g.nuclei.iter().map(|n| n.pos).collect::<Vec<_>>(), "held-out {j}: the stored geometry is this draw's");
            dft.push((s.energy, s.forces.clone()));
        }
    }
    let _ = writeln!(out, "  teacher in all: {}; wall {:.0} s", log.lock().unwrap().summary(), wall.elapsed().as_secs_f64());
    flush(&mut out);

    // ---- fit and judge each collection ----
    let mut rows = Vec::new();
    for (name, s) in [("A (whole)", &mut a), ("control (whole answers, interior rows)", &mut ctl), ("B (capped segments)", &mut bs)] {
        let t = Instant::now();
        let rep = s.refit().unwrap_or_else(|e| panic!("{name}: refit: {e}"));
        let acc = s.accumulator().expect("fitted");
        let _ = writeln!(out, "  {name}: {} samples, {} energy rows, {} force rows; refit {:.1} s: {}", acc.n_samples, acc.n_e, acc.n_f, t.elapsed().as_secs_f64(), fit_summary(&rep, s));
        let j = judge(s, &held, &dft, &class);
        let _ = writeln!(out, "    held-out force RMSE: interior class {:.3e} Ha/Bohr, all atoms {:.3e}; trusted interior-class atoms {}/{}, answered geometries {}/{HELD_OUT}; classes: {}",
                         j.f_class, j.f_all, j.trusted_class, j.class_atoms, j.answered, class_summary(s));
        flush(&mut out);
        rows.push(j);
    }
    let (ja, jc, jb) = (&rows[0], &rows[1], &rows[2]);
    let ratio = jb.f_class / ja.f_class;
    let _ = writeln!(out, "capping gate: B/A interior-class force RMSE {:.3e}/{:.3e} = {ratio:.2}× (limit {FACTOR}×); B/control {:.2}× (the capping error alone); control/A {:.2}× (the rows B teaches, uncapped)",
                     jb.f_class, ja.f_class, jb.f_class / jc.f_class, jc.f_class / ja.f_class);
    if ja.trusted_class == 0 {
        let _ = writeln!(out, "capping gate: A trusts none of the interior-class atoms — the comparison is still on force RMSE: it measures whether the two training methods are equivalent, not whether either model is trusted");
    }
    let _ = writeln!(out, "capping gate wall {:.0} s", wall.elapsed().as_secs_f64());
    flush(&mut out);
    assert!(ratio <= FACTOR, "capped segments train {ratio:.2}× worse than the whole molecule on the interior class (limit {FACTOR}×) — spec §4.5: replace capping by the uncapped two-shell cut and rerun");
}

/// Cost probe (Task 10a Step 0): one Accurate solve of the whole molecule and of an end segment,
/// cold and warm after a 0.05 Bohr move, and the estimate of the gate's GPU wall from them.
#[test]
#[ignore = "GPU, minutes: cargo test -p validate --release --test capping -- --ignored --nocapture capping_cost_probe"]
fn capping_cost_probe() {
    let Some(ctx) = gpu() else { return };
    let (mut t, log) = Locked::new(ctx.clone(), TOL_H2);
    let sc = if scenes_dir().join("butane.scene").exists() { butane() } else { butane_start() };
    let b = bonds(&sc);
    let d_cap = cap_length(&sc, &b);
    let seg = cut(&sc, &b, &[ends(&sc, &b)[0]], COLLECTION_R_CUT, d_cap).scene;
    let mut times = Vec::new();
    for (name, g) in [("butane", sc.clone()), ("end segment", seg)] {
        t.reset();
        let w = Instant::now();
        let cold = solve(&mut t, &g, name);
        let cold_s = w.elapsed().as_secs_f64();
        let mut moved = g.clone();
        moved.nuclei[0].pos[0] += 0.05;
        let w = Instant::now();
        let warm = solve(&mut t, &moved, name);
        let warm_s = w.elapsed().as_secs_f64();
        // a displaced geometry after a reset, as the gate solves each (cold by design)
        let mut far = g.clone();
        for n in &mut far.nuclei {
            n.pos[1] += 0.1;
        }
        t.reset();
        let w = Instant::now();
        let again = solve(&mut t, &far, name);
        let again_s = w.elapsed().as_secs_f64();
        println!("{name}: {} atoms, {}³; cold {cold_s:.1} s ({} SCF, incl. any pseudo generation), warm {warm_s:.1} s ({} SCF), cold again {again_s:.1} s ({} SCF); E {:.6}; face density {:?}",
                 g.nuclei.len(), g.cell.n, cold.stats.scf_steps, warm.stats.scf_steps, again.stats.scf_steps, cold.energy, cold.stats.face_density);
        times.push((warm_s, again_s));
    }
    let (whole, seg) = (times[0].1, times[1].1);
    println!("teacher: {}", log.lock().unwrap().summary());
    for n in [20usize, 40] {
        let gate = n as f64 * (whole + 2.0 * seg) + HELD_OUT as f64 * whole;
        println!("estimate, {n} training geometries: gate {:.1} h at the probe's cold solve, {:.1} h at {RUN_OVER_PROBE}× (Task 6's run/probe); relaxation ≈ 100–200 FIRE steps × warm {:.0} s = {:.1}–{:.1} h ({:.1}–{:.1} h at {RUN_OVER_PROBE}×)",
                 gate / 3600.0, RUN_OVER_PROBE * gate / 3600.0, times[0].0, 100.0 * times[0].0 / 3600.0, 200.0 * times[0].0 / 3600.0, RUN_OVER_PROBE * 100.0 * times[0].0 / 3600.0, RUN_OVER_PROBE * 200.0 * times[0].0 / 3600.0);
    }
}

/// n-butane relaxed on the app's default surface (Accurate, `ionic accepted`, GPU teacher at 1e-6)
/// by FIRE (`sim::Fire`, the app's Relax: Δt from the 0.25 fs MD step, F_TOL 1e-3 Ha/Bohr on every
/// atom's force norm) from the cited start; re-centred in its box and written to
/// `scenes/butane.scene` (ruling P7). Run once; the gate reads the file.
#[test]
#[ignore = "GPU, hours (not yet run; probe estimate 100–200 FIRE steps × 95 s = 2.6–5.3 h): cargo test -p validate --release --test capping -- --ignored --nocapture relax_butane"]
fn relax_butane() {
    let Some(ctx) = gpu() else { return };
    let (mut t, log) = Locked::new(ctx.clone(), TOL_H2);
    let wall = Instant::now();
    let mut sc = butane_start();
    let r0 = solve(&mut t, &sc, "relax start");
    let mut forces = r0.forces;
    let mut fire = Fire::new(FireParams::for_dt(units::fs_to_au_time(DT_FS)), r0.energy);
    let pins = vec![false; sc.nuclei.len()];
    let outcome = loop {
        let st = fire.step(&mut sc, &mut forces, &pins, &mut t, &mut no_augment(), &mut never()).unwrap_or_else(|e| panic!("FIRE step {}: {e}", fire.steps()));
        if fire.steps() % 10 == 0 {
            println!("  butane FIRE {}: E {:.6} Ha, max|F| {:.2e}, Δt {:.2} fs, wall {:.0} s", fire.steps(), st.energy, st.max_force_augmented, units::au_time_to_fs(fire.dt()), wall.elapsed().as_secs_f64());
        }
        if st.outcome != FireOutcome::Running {
            break (st.outcome, st.energy, st.max_force_augmented);
        }
    };
    for n in &mut sc.nuclei {
        n.vel = [0.0; 3];
    }
    let relaxed = boxed(sc.nuclei.clone(), MARGIN);
    let check = solve(&mut t, &relaxed, "relaxed, re-centred");
    let fmax = sim::relax::max_norm(&check.forces);
    let bnd = bonds(&relaxed);
    let cc: Vec<String> = bnd.iter().filter(|&&(x, y)| relaxed.nuclei[x].z == 6 && relaxed.nuclei[y].z == 6).map(|&(x, y)| format!("{:.4}", dist(relaxed.nuclei[x].pos, relaxed.nuclei[y].pos))).collect();
    println!("butane: {:?} after {} FIRE steps in {:.0} s: E {:.6} → {:.6} Ha (re-centred in {}³: E {:.6}, max|F| {fmax:.2e}, face density {:?}); C–C {cc:?} Bohr, C–H mean {:.4} Bohr; {}",
             outcome.0, fire.steps(), wall.elapsed().as_secs_f64(), r0.energy, outcome.1, relaxed.cell.n, check.energy, check.stats.face_density, cap_length(&relaxed, &bnd), log.lock().unwrap().summary());
    assert_eq!(outcome.0, FireOutcome::Converged, "FIRE ended {:?} at max|F| {:.2e}", outcome.0, outcome.2);
    let comment = vec![
        "n-Butane (anti, C2h), relaxed on this solver's default surface: FIRE (the app's Relax, F_TOL 1e-3 Ha/Bohr per atom)".to_string(),
        "at the Accurate grid (h = sigma = 0.15 Bohr) under the accepted pseudopotentials, GPU teacher at tol 1e-6.".to_string(),
        "Start: all staggered, C-C 1.53 A, CCC 113 degrees (about the gas-phase electron-diffraction structure, Bradford,".to_string(),
        "Fitzwater & Bartell, J. Mol. Struct. 38, 185 (1977)), C-H 1.09 A, tetrahedral methyl H, HCH 107 degrees on CH2; built".to_string(),
        "from these parameters (not a literature Cartesian set); the geometry below is this solver's minimum, not experiment.".to_string(),
        format!("Relaxed: {} FIRE steps, E = {:.6} Ha, max|F| = {fmax:.1e} Ha/Bohr re-centred; C-C {} Bohr.", fire.steps(), check.energy, cc.join(", ")),
        format!("Box: the nuclei's extent + {MARGIN} Bohr per side at h = {H_ACC} ({}^3). Written by validate/tests/capping.rs relax_butane.", relaxed.cell.n),
    ];
    let path = scenes_dir().join("butane.scene");
    std::fs::write(&path, scene_text(&relaxed, &comment)).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    println!("wrote {}", path.display());
}
