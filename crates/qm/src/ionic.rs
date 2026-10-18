//! The ionic species of a scene (M7; spec §4.7, §5.2, §5.5 `qm` bullet, §5.6): which nuclei the electrons
//! see as all-electron smeared charges and which as norm-conserving pseudo-ions, decided once per scene by a
//! scene-wide [`IonicPolicy`] and frozen into an [`IonicSetup`]. CPU, f64, atomic units (Hartree, Bohr).
//!
//! # The three policies (spec §4.7)
//! The policy is global, never per species: which species actually carry a pseudopotential is an *output* of
//! the recipe (§4.2's partition: a species has a core or not), never an input entered per element
//! (CLAUDE.md rule 1 — the only per-element datum is Z).
//! - `None`: today's all-electron solver throughout: every species is a Gaussian-smeared nucleus of charge Z
//!   and width σ_AE (`nuclear::nuclear_potential`). Inside the SCF this is `System.ionic == None`, the same code
//!   path bit for bit (`scf` module doc, "Ionic species (M7)").
//! - `CoresOnly`: a species is pseudised iff §4.2 gives it a core (`has_core`); H and He stay all-electron at
//!   σ_AE. A recipe refusal (Li, Be, Na, Mg, Al at `m7-tm-kb-nlcc-4`) falls back to all-electron with a note:
//!   the scene runs, with that species' Zσ error, and the note says so (rule 7).
//! - `All`: every species is pseudised by the same recipe, including the no-core ones (H, He: Z_v = Z, no
//!   partial core, r_core = 0 — the explicit no-core branch, spec §4.2 item 4; their tables must have passed the
//!   dimer gate on f, else `Provisional` below). A refusal is an error
//!   ([`IonicError::Refused`]): `All` has no all-electron fallback by definition.
//!
//! # Build rules (spec §4.8 T8, §5.1; Task 5 physics review items 4 and M5/M8)
//! Every table used is the one filtered for *this* grid spacing h (`radial::filter::filtered`), and both
//! pseudo policies require the table to be certified at h:
//! - h ≤ h_req (the coarsest ladder h at which it and every finer ladder h pass T8(a)–(d)), else
//!   [`IonicError::TooCoarse`];
//! - a scene *finer than the whole ladder* (h < min `LEDGER.h_ladder`, e.g. Precise 0.10) is certified at its own
//!   h by the table's own filtered re-run, even when the element has no h_req on the ladder (Ne at this recipe
//!   version: T8(a) misses by 0.35 mHa at 0.15 — an outcome, not a refusal; ruling item 4);
//! - in both cases the table's own T8 row at h (`FilteredTable::check`) must pass `FilterRow::passes()` — the
//!   same predicate `filter::h_req` uses, so the two cannot drift — else [`IonicError::FilteredGate`];
//! - before all of these, a **provisional** table (Z ≤ `LEDGER.dimer_z_max` generated without the dimer gate, M7 Task 11;
//!   plan review M6) is refused, [`IonicError::Provisional`]: only the providers' cache, which runs the gate, makes
//!   usable H and He tables. `from_tables` stays unchecked (tests and comparisons on purpose).
//!
//! # Core overlap (spec §5.6; plan review M3; Task 2 ruling)
//! The frozen-core approximation fails when two ions' *core densities* overlap, not when their pseudisation
//! spheres do (overlapping r_c spheres are the normal regime of norm-conserving potentials). The rule uses the
//! derived core radius r_core,I: the radius enclosing all but ε_core = `LEDGER.eps_core` = 1e-3 of the N_c core
//! electrons of the AE atom (`radial::nlcc::core_radius`), 0 for a species without a core and for every
//! all-electron species. A pair with R_IJ < r_core,I + r_core,J is reported by [`core_overlaps`] and becomes a
//! per-step warning (`scf`, [`CORE_OVERLAP_WARNING`]); the step is still taken (every field is finite and smooth).
//! r_cmp (where V_ion → −Z_v/r) is never used as a core radius (Task 2 ruling).
use crate::nuclear::Nucleus;
use crate::radial::filter::FilteredTable;
use crate::radial::ledger::{recipe_hash, LEDGER};
use crate::radial::Refusal;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Which species carry a pseudopotential (module doc): scene-wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IonicPolicy { None, CoresOnly, All }

/// How the electrons see one species: today's smeared nucleus (charge Z, width `sigma`), or a pseudo-ion
/// (charge Z_v, V_LR at σ_LR, the one V_SR table, KB projectors, partial core) filtered for the scene's h.
#[derive(Clone, Debug)]
pub enum Ionic { AllElectron { sigma: f64 }, Pseudo(Arc<FilteredTable>) }

/// The frozen ionic description of a scene at spacing `h` (module doc). `sigma_ae` is the smearing of every
/// all-electron species (the SCF's σ; `scf` asserts they agree). `notes` records the `CoresOnly` fallbacks.
#[derive(Clone, Debug)]
pub struct IonicSetup { pub policy: IonicPolicy, pub h: f64, pub sigma_ae: f64, pub recipe_hash: u64, pub per_z: BTreeMap<u8, Ionic>, pub notes: Vec<String> }

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum IonicError {
    #[error("{} is refused by the pseudopotential recipe (IonicPolicy::All has no all-electron fallback): {refusal}", units::elements::symbol(*z))]
    Refused { z: u8, refusal: Refusal },
    #[error("h = {h} Bohr is coarser than {}'s pseudopotential allows (h_req = {}): choose a finer grid", units::elements::symbol(*z), h_req.map_or("none on the ladder".to_string(), |x| format!("{x} Bohr")))]
    TooCoarse { z: u8, h: f64, h_req: Option<f64> },
    #[error("{}'s filtered pseudopotential fails the filtered gates at h = {h} Bohr: {why}", units::elements::symbol(*z))]
    FilteredGate { z: u8, h: f64, why: String },
    /// Plan review M6 (spec §4.4, §4.9): a table for Z ≤ `LEDGER.dimer_z_max` generated without the dimer gate.
    #[error("{}'s pseudopotential is provisional: the dimer gate on f (spec §4.4) has not run — tables for Z ≤ {} come from the providers' cache, which runs it", units::elements::symbol(*z), LEDGER.dimer_z_max)]
    Provisional { z: u8 },
}

/// The prefix of the per-step core-overlap warning (spec §5.6): "cores overlap: I–J at R = … Bohr".
pub const CORE_OVERLAP_WARNING: &str = "cores overlap";

/// T8 at the scene's h (module doc, "Build rules"; Task 5 physics review M8), after the provisional test (plan review M6).
fn check(z: u8, t: &FilteredTable, h: f64) -> Result<(), IonicError> {
    if t.provisional { return Err(IonicError::Provisional { z }); }
    assert!((t.h - h).abs() < 1e-15, "a table filtered at h = {} used at h = {h}", t.h);
    let finest = LEDGER.h_ladder.iter().cloned().fold(f64::INFINITY, f64::min);
    match t.record.h_req {
        Some(hr) if h <= hr + 1e-12 => {}
        // finer than the ladder: certified by this h's own filtered re-run below (plan review Q11)
        None if h < finest - 1e-12 => {}
        other => return Err(IonicError::TooCoarse { z, h, h_req: other }),
    }
    let c = &t.check;
    if !c.passes() {
        return Err(IonicError::FilteredGate { z, h, why: format!(
            "ΔE {:.3} mHa, Δε {:.3} mHa, T3 {:.3} mHa, XC clamp at {} points, table egg-box force {:.2e} Ha/Bohr",
            c.de * 1e3, c.max_eig * 1e3, c.t3_max * 1e3, c.clamped, c.f_table.iter().sum::<f64>()) });
    }
    Ok(())
}

impl IonicSetup {
    /// The setup for the species `zs` (duplicates allowed) under `policy` at spacing `h` (module doc, "Build
    /// rules"). `has_core` and `table` are the recipe's (`radial::has_core`, `filtered(&generate(z)?, h)`),
    /// passed in so a caller can cache them; `table` must return the table filtered at this `h`.
    pub fn build(policy: IonicPolicy, h: f64, sigma_ae: f64, zs: &[u8],
                 has_core: &mut dyn FnMut(u8) -> Result<bool, Refusal>,
                 table: &mut dyn FnMut(u8) -> Result<Arc<FilteredTable>, Refusal>) -> Result<IonicSetup, IonicError> {
        let mut species: Vec<u8> = zs.to_vec();
        species.sort_unstable();
        species.dedup();
        let (mut per_z, mut notes) = (BTreeMap::new(), Vec::new());
        let ae = Ionic::AllElectron { sigma: sigma_ae };
        for z in species {
            let sym = units::elements::symbol(z);
            let ionic = match policy {
                IonicPolicy::None => ae.clone(),
                IonicPolicy::CoresOnly => match has_core(z) {
                    Ok(false) => ae.clone(),
                    Ok(true) => match table(z) {
                        Ok(t) => { check(z, &t, h)?; Ionic::Pseudo(t) }
                        Err(r) => { notes.push(format!("{sym}: {r} — kept all-electron (σ = {sigma_ae})")); ae.clone() }
                    },
                    Err(r) => { notes.push(format!("{sym}: {r} — kept all-electron (σ = {sigma_ae})")); ae.clone() }
                },
                IonicPolicy::All => match table(z) {
                    Ok(t) => { check(z, &t, h)?; Ionic::Pseudo(t) }
                    Err(refusal) => return Err(IonicError::Refused { z, refusal }),
                },
            };
            per_z.insert(z, ionic);
        }
        Ok(IonicSetup { policy, h, sigma_ae, recipe_hash: recipe_hash(), per_z, notes })
    }

    /// A setup from given treatments, with no checks (tests, and comparisons that need an uncertified table,
    /// e.g. the radial-vs-3-D Ne test at h = 0.15, where Ne has no h_req).
    pub fn from_tables(policy: IonicPolicy, h: f64, sigma_ae: f64, per_z: BTreeMap<u8, Ionic>) -> IonicSetup {
        IonicSetup { policy, h, sigma_ae, recipe_hash: recipe_hash(), per_z, notes: Vec::new() }
    }

    /// The treatment of Z. Panics for a Z not in `per_z` (a programming error: the setup is built from the scene).
    pub fn ionic(&self, z: u8) -> &Ionic { self.per_z.get(&z).unwrap_or_else(|| panic!("Z = {z} is not in this ionic setup")) }

    /// The charge the electrons see far away: Z for an all-electron species, Z_v for a pseudised one.
    pub fn valence_charge(&self, z: u8) -> f64 {
        match self.ionic(z) { Ionic::AllElectron { .. } => z as f64, Ionic::Pseudo(t) => t.z_v }
    }

    /// The pseudised species, ascending.
    pub fn pseudised(&self) -> Vec<u8> { self.per_z.iter().filter(|(_, i)| matches!(i, Ionic::Pseudo(_))).map(|(z, _)| *z).collect() }

    /// Same policy, recipe, h, σ_AE and the same species with the same treatment: a `KsState` converged under
    /// one is a valid warm start for the other (the setup-equality guard of spec §5.5).
    pub fn same_physics(&self, o: &IonicSetup) -> bool {
        self.policy == o.policy && self.h == o.h && self.sigma_ae == o.sigma_ae && self.recipe_hash == o.recipe_hash
            && self.per_z.len() == o.per_z.len()
            && self.per_z.iter().zip(&o.per_z).all(|((za, a), (zb, b))| za == zb && match (a, b) {
                (Ionic::AllElectron { sigma: x }, Ionic::AllElectron { sigma: y }) => x == y,
                (Ionic::Pseudo(x), Ionic::Pseudo(y)) => x.h == y.h && x.z_v == y.z_v,
                _ => false,
            })
    }

    /// r_core (spec §5.6): the pseudised species' derived core radius; 0 for an all-electron species.
    pub fn r_core(&self, z: u8) -> f64 { match self.ionic(z) { Ionic::Pseudo(t) => t.r_core, Ionic::AllElectron { .. } => 0.0 } }
}

/// Spec §5.6: the pairs (I, J, R_IJ), I < J, with R_IJ < r_core,I + r_core,J (module doc).
pub fn core_overlaps(setup: &IonicSetup, nuclei: &[Nucleus]) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    for i in 0..nuclei.len() {
        for j in i + 1..nuclei.len() {
            let d = (0..3).map(|a| (nuclei[i].pos[a] - nuclei[j].pos[a]).powi(2)).sum::<f64>().sqrt();
            if d < setup.r_core(nuclei[i].z) + setup.r_core(nuclei[j].z) { out.push((i, j, d)); }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radial::filter::FilteredTable;
    use crate::radial::test_cache::{filtered_at, table};
    use crate::radial::{generate, has_core};

    /// The recipe's closures at h, through the test caches (a refused Z is refused by `generate`).
    fn tables(h: f64) -> impl FnMut(u8) -> Result<Arc<FilteredTable>, Refusal> {
        move |z| if z == 3 { generate(z).map(|_| unreachable!("Li is refused")) } else { Ok(filtered_at(z, h)) }
    }

    /// Spec §4.7: None is all-electron throughout; CoresOnly pseudises exactly the species with a core and
    /// falls back to all-electron, with a note, for a refused one (Li); All pseudises H too (the no-core
    /// branch: Z_v = Z, no partial core, r_core = 0) and refuses Li outright.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn build_follows_the_policy() {
        let h = 0.2;
        let zs = [8, 1, 1, 3];
        let none = IonicSetup::build(IonicPolicy::None, h, h, &zs, &mut has_core, &mut tables(h)).unwrap();
        assert!(none.pseudised().is_empty() && none.notes.is_empty() && none.per_z.len() == 3);
        assert_eq!((none.valence_charge(8), none.valence_charge(1)), (8.0, 1.0));

        let cores = IonicSetup::build(IonicPolicy::CoresOnly, h, h, &zs, &mut has_core, &mut tables(h)).unwrap();
        assert_eq!(cores.pseudised(), vec![8]);
        assert!(matches!(cores.ionic(1), Ionic::AllElectron { sigma } if *sigma == h));
        assert!(matches!(cores.ionic(3), Ionic::AllElectron { .. }));
        assert_eq!(cores.notes.len(), 1, "{:?}", cores.notes);
        assert!(cores.notes[0].starts_with("Li: ") && cores.notes[0].contains("kept all-electron"), "{:?}", cores.notes);
        assert_eq!((cores.valence_charge(8), cores.valence_charge(1), cores.valence_charge(3)), (6.0, 1.0, 3.0));

        // the test cache's H is `generate(1)`: ungated, so provisional (plan review M6) and refused by `build`
        match IonicSetup::build(IonicPolicy::All, h, h, &[8, 1], &mut has_core, &mut tables(h)) {
            Err(e @ IonicError::Provisional { z: 1 }) => assert!(e.to_string().starts_with("H's pseudopotential is provisional"), "{e}"),
            other => panic!("All with the ungated H: {other:?}"),
        }
        // the policy's mechanics on an explicit stand-in for the gated H (qm cannot run the 3-D gate; the providers do)
        let stand_in = |z: u8| -> Result<Arc<FilteredTable>, Refusal> {
            if z == 3 { return generate(z).map(|_| unreachable!("Li is refused")); }
            let mut t = (*filtered_at(z, h)).clone();
            t.provisional = false;
            Ok(Arc::new(t))
        };
        let all = IonicSetup::build(IonicPolicy::All, h, h, &[8, 1], &mut has_core, &mut { stand_in }).unwrap();
        assert_eq!(all.pseudised(), vec![1, 8]);
        let Ionic::Pseudo(hy) = all.ionic(1) else { unreachable!() };
        assert_eq!((hy.z_v, hy.pc_radius, hy.r_core, all.r_core(1)), (1.0, 0.0, 0.0, 0.0));
        match IonicSetup::build(IonicPolicy::All, h, h, &zs, &mut has_core, &mut { stand_in }) {
            Err(e @ IonicError::Refused { z: 3, .. }) => assert!(e.to_string().starts_with("Li is refused"), "{e}"),
            other => panic!("All with Li: {other:?}"),
        }
    }

    /// T8 at the scene's h (Task 5 physics review items 4, M5, M8): h above h_req is TooCoarse (O at 0.25,
    /// h_req 0.20); a failing T8 row is FilteredGate whatever h_req says; and the check never runs on a table
    /// filtered for another h.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn check_refuses_coarse_grids_and_failing_rows() {
        assert_eq!(table(8).record.h_req, Some(0.2));
        match IonicSetup::build(IonicPolicy::CoresOnly, 0.25, 0.25, &[8], &mut has_core, &mut tables(0.25)) {
            Err(e @ IonicError::TooCoarse { z: 8, h_req: Some(hr), .. }) if hr == 0.2 => assert!(e.to_string().contains("h_req = 0.2 Bohr"), "{e}"),
            other => panic!("O at 0.25: {other:?}"),
        }
        let mut bad = (*filtered_at(8, 0.2)).clone();
        bad.check.clamped = 1;
        let bad = Arc::new(bad);
        match IonicSetup::build(IonicPolicy::All, 0.2, 0.2, &[8], &mut has_core, &mut |_| Ok(bad.clone())) {
            Err(IonicError::FilteredGate { z: 8, why, .. }) => assert!(why.contains("XC clamp at 1 points"), "{why}"),
            other => panic!("a failing row: {other:?}"),
        }
        let wrong_h = std::panic::catch_unwind(|| check(8, &filtered_at(8, 0.2), 0.25));
        assert!(wrong_h.is_err(), "a table filtered at 0.20 used at 0.25 must panic");
    }

    /// The setup-equality guard (spec §5.5): same policy, h, σ_AE, recipe and per-species treatment.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn same_physics_compares_every_treatment() {
        let h = 0.2;
        let ae = |s: f64| Ionic::AllElectron { sigma: s };
        let o = || Ionic::Pseudo(filtered_at(8, h));
        let a = IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(1, ae(h)), (8, o())]));
        assert!(a.same_physics(&a.clone()));
        assert!(a.same_physics(&IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(1, ae(h)), (8, o())]))));
        for b in [IonicSetup::from_tables(IonicPolicy::All, h, h, BTreeMap::from([(1, ae(h)), (8, o())])),
                  IonicSetup::from_tables(IonicPolicy::CoresOnly, h, 0.25, BTreeMap::from([(1, ae(h)), (8, o())])),
                  IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(1, ae(0.25)), (8, o())])),
                  IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(1, ae(h)), (8, ae(h))])),
                  IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(8, o())]))] {
            assert!(!a.same_physics(&b) && !b.same_physics(&a), "{:?} vs {:?}", a.policy, b.policy);
        }
        let mut other_recipe = a.clone();
        other_recipe.recipe_hash ^= 1;
        assert!(!a.same_physics(&other_recipe));
    }

    /// Spec §5.6: r_core (all but ε_core = 1e-3 of N_c) is silent at bonding distances and loud when cores are
    /// pushed together; all-electron species (r_core = 0) never overlap.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only; run: cargo test -p qm --release")]
    fn core_overlap_uses_the_core_radius() {
        let h = 0.2;
        let s = IonicSetup::from_tables(IonicPolicy::CoresOnly, h, h, BTreeMap::from([(1, Ionic::AllElectron { sigma: h }), (8, Ionic::Pseudo(filtered_at(8, h)))]));
        let r = s.r_core(8);
        eprintln!("O r_core = {r:.4} Bohr (spec §5.6 hydrogenic estimate ≈ 0.7)");
        assert!(r > 0.3 && r < 1.0 && s.r_core(1) == 0.0, "O r_core {r}");
        let at = |x: f64, z: u8| Nucleus { pos: [x, 0.0, 0.0], z };
        assert!(core_overlaps(&s, &[at(0.0, 8), at(2.28, 8), at(-1.81, 1)]).is_empty(), "O₂ and O–H bond lengths are silent");
        // two O cores pushed together, and a proton (all-electron, r_core 0) inside an O core: R < r_core,O + 0
        let close = core_overlaps(&s, &[at(0.0, 8), at(1.9 * r, 8), at(-0.5 * r, 1)]);
        assert_eq!(close.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(), vec![(0, 1), (0, 2)]);
        assert!((close[0].2 - 1.9 * r).abs() < 1e-12 && (close[1].2 - 0.5 * r).abs() < 1e-12);
    }
}
