//! Self-generated pseudopotentials (M7), the generator: CPU, f64, no GPU. Spec:
//! docs/superpowers/specs/2026-10-14-pseudopotential-design.md §4–§5.1.
//! Pipeline: atom (§4.1) → partition (§4.2) → nlcc (§4.5) → tm + unscreening (§4.3) → kb + ghost
//! gates (§4.6) → transferability gates T1–T7 (§4.8) → recipe: the f ladder × local order (§4.4) →
//! record + recipe_hash (§4.9) → filter + h_req (§5.1, §4.8 T8). Every universal constant is a field
//! of `ledger::LEDGER`; nothing is indexed by Z (CLAUDE.md rule 1).
pub mod atom;
pub mod grid;
pub mod ledger;
pub mod numerov;

pub use atom::{allowed_multiplicities, ground_state, ground_state_with, hartree_potential, next_unoccupied,
               scaling_identity_residual, solve_atom, AtomEnergy, AtomOptions, AtomResult, Occupations, Orbital,
               RadialError, ShellOcc, SpinMode, XcFn};
pub use grid::LogGrid;
pub use ledger::{Ledger, LocalCandidate, NlccMode, LEDGER};
