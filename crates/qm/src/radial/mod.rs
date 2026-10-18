//! Self-generated pseudopotentials (M7), the generator: CPU, f64, no GPU. Spec:
//! docs/superpowers/specs/2026-10-14-pseudopotential-design.md §4–§5.1.
//! Pipeline: atom (§4.1) → partition (§4.2) → nlcc (§4.5) → tm + unscreening (§4.3) → kb + ghost
//! gates (§4.6) → transferability gates T1–T7 (§4.8) → recipe: the f ladder × local order (§4.4) →
//! record + recipe_hash (§4.9) → filter + h_req (§5.1, §4.8 T8). Every universal constant is a field
//! of `ledger::LEDGER`; nothing is indexed by Z (CLAUDE.md rule 1).
pub mod atom;
pub mod filter;
pub mod gates;
pub mod grid;
pub mod kb;
pub mod ledger;
pub mod nlcc;
pub mod numerov;
pub mod partition;
pub mod recipe;
pub mod record;
pub mod tm;

pub use atom::{allowed_multiplicities, ground_state, ground_state_with, hartree_potential, next_unoccupied,
               scaling_identity_residual, solve_atom, AtomEnergy, AtomOptions, AtomResult, Occupations, Orbital,
               RadialError, ShellOcc, SpinMode, XcFn};
pub use filter::{filter_radial, filtered, h_eb, h_req, mask, sph_bessel, sph_bessel_scaled, FilteredProjector, FilteredTable};
pub use grid::LogGrid;
pub use gates::{delta_e_equiv, run_gates, transfer_configs, GateReport, T7Row, TransferConfig, TransferRow};
pub use ledger::{core_recipe_hash, hash_ledger, recipe_hash, species_recipe_hash, Ledger, LocalCandidate, NlccMode, DIMER_GATE, LEDGER};
pub use nlcc::{core_radius, partial_core, partial_core_at, PartialCore};
pub use record::{record_of, records_table, FilterRow, GenerationRecord};
pub use partition::{core_density, has_core, partition, valence_density, Channel, Partition, Refusal, Shell};
pub use tm::{tm_channel, tm_channels, unscreen, TmChannel, Unscreened};
pub use kb::{core_tail, ghost_gates, gss_verdict, kb_count_below, kb_eigenvalues, kb_form, kb_form_with_local, kb_lower_bound,
             kb_regular, kb_state, polynomial_local, reference_config, solve_kb_atom, GhostReport, GssVerdict, KbAtom,
             KbInput, KbPotential, Projector, ValenceConfig};
pub use recipe::{generate, generate_at_crossover, generate_at_crossover_with, generate_with, DimerError, DimerGate, DimerPair, PseudoTable};

/// Test-process caches (`cargo test -p qm`): each element generated once (`generate` runs the whole T8
/// ladder), and each (Z, h) filtered once, shared by every module's tests.
#[cfg(test)]
pub(crate) mod test_cache {
    use super::filter::{filtered, FilteredTable};
    use super::recipe::{generate, PseudoTable};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    pub(crate) fn table(z: u8) -> &'static PseudoTable {
        static CACHE: [OnceLock<PseudoTable>; 19] = [const { OnceLock::new() }; 19];
        CACHE[z as usize].get_or_init(|| generate(z).unwrap_or_else(|e| panic!("Z = {z}: {e}")))
    }

    pub(crate) fn filtered_at(z: u8, h: f64) -> Arc<FilteredTable> {
        static CACHE: OnceLock<Mutex<HashMap<(u8, u64), Arc<FilteredTable>>>> = OnceLock::new();
        let t = table(z);
        let key = (z, h.to_bits());
        if let Some(f) = CACHE.get_or_init(Default::default).lock().unwrap().get(&key) { return f.clone(); }
        let f = Arc::new(filtered(t, h));
        CACHE.get_or_init(Default::default).lock().unwrap().entry(key).or_insert(f).clone()
    }
}
