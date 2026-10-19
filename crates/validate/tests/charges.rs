//! Partial charges on real densities (`sim::charges`, Hirshfeld; display only): the driver's
//! charges after a converged CPU solve (`providers::CpuKs`, all-electron, 41³–49³ × 0.25 Bohr —
//! the CPU so that this suite never competes for the GPU), with the free-atom references the bond
//! detector solves on its side solver. What is checked is what the partition guarantees or what any
//! partition must get right, never a Hirshfeld magnitude from the literature (the scheme and the
//! LSDA/smeared density both move it):
//! - H₂ (symmetric): both charges ≈ 0, Σq = 0;
//! - LiH: Li δ+, H δ− (the electropositive metal gives up density), Σq = 0;
//! - HeH⁺ (charge +1): Σq = +1 — the sum is the scene's charge.
//! Release only (`cargo test -p validate --release --test charges`: 645 s on the CPU, the LiH test's
//! Li reference scan nearly all of it; numbers in `references.md` "Partial charges").
use providers::CpuKs;
use sim::{Cell, DynCommand, Dynamics, IonicPolicy, Nucleus, PartialCharges, Scene};
use std::time::{Duration, Instant};

fn scene(nuclei: Vec<Nucleus>, n: usize, charge: i32) -> Scene {
    Scene { nuclei, cell: Cell { n, h: 0.25 }, charge, multiplicity: None, sigma: None, external_field: [0.0; 3], ionic: IonicPolicy::None }
}

/// The driver's charges at the scene's geometry: paused ticks until the references are in and the
/// charges published (module doc of `sim::dynamics`, "Partial charges").
fn charges_of(s: Scene) -> PartialCharges {
    let t0 = Instant::now();
    let mut d = Dynamics::new(s, Box::new(CpuKs::new(1e-6)), 0.5, 1);
    d.apply(DynCommand::Pause);
    loop {
        let snap = d.tick(f64::INFINITY, &mut || false);
        assert_eq!(snap.last_error, None);
        if let Some(c) = snap.charges.as_deref().filter(|c| c.is_current(&snap.scene)) {
            println!("  ({:.1} s, {} notes: {:?})", t0.elapsed().as_secs_f64(), snap.notes.len(), snap.notes.last());
            return c.clone();
        }
        assert!(t0.elapsed() < Duration::from_secs(900), "no charges after 15 min: pending {:?}", snap.bonds_pending);
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn report(what: &str, c: &PartialCharges) {
    let q: Vec<String> = c.z.iter().zip(&c.q).map(|(z, q)| format!("{} {q:+.4}", units::isotopes::symbol(*z))).collect();
    println!("{what}: {} · Σq − charge = {:+.2e} (charge {:+}) · N = {:.6}", q.join(", "), c.sum_error(), c.system_charge, c.electrons);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only: real SCF solves")]
fn symmetric_h2_is_neutral_and_sums_to_zero() {
    let c = charges_of(scene(vec![Nucleus::new(1, [-0.725, 0.0, 0.0]), Nucleus::new(1, [0.725, 0.0, 0.0])], 41, 0));
    report("H₂ (R = 1.45 Bohr, 41³ × 0.25)", &c);
    assert!(c.balanced(), "Σq = {:+e}", c.sum_error());
    assert!(c.sum().abs() < 1e-3, "{}", c.sum());
    assert!((c.q[0] - c.q[1]).abs() < 1e-4, "{:?}", c.q);
    assert!(c.q[0].abs() < 1e-3 && c.q[1].abs() < 1e-3, "{:?}", c.q);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only: real SCF solves")]
fn lih_is_li_plus_h_minus() {
    // R = 3.015 Bohr (experimental r_e), off-centre so nothing is symmetric by construction. A 12 Bohr
    // box (49³ × 0.25): the physics review notes that its Dirichlet wall pinches Li's diffuse 2s (the
    // lih preset uses 14 Bohr) — enough to move the magnitude a little, not the sign this test asserts;
    // the 57³ CPU run would take ≈ 17 min against 11 (the Li reference scan dominates)
    let c = charges_of(scene(vec![Nucleus::new(3, [-1.6, 0.1, 0.0]), Nucleus::new(1, [1.415, 0.1, 0.0])], 49, 0));
    report("LiH (R = 3.015 Bohr, 49³ × 0.25, all-electron)", &c);
    assert!(c.balanced(), "Σq = {:+e}", c.sum_error());
    assert!(c.q[0] > 0.05 && c.q[1] < -0.05, "Li δ+, H δ−: {:?}", c.q);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release only: real SCF solves")]
fn the_sum_is_the_scene_charge_for_a_cation() {
    let c = charges_of(scene(vec![Nucleus::new(2, [-0.73, 0.0, 0.0]), Nucleus::new(1, [0.73, 0.0, 0.0])], 41, 1));
    report("HeH⁺ (R = 1.46 Bohr, 41³ × 0.25, charge +1)", &c);
    assert_eq!(c.system_charge, 1);
    assert!(c.balanced(), "Σq = {:+e}", c.sum_error());
    assert!((c.sum() - 1.0).abs() < 1e-3, "{}", c.sum());
}
