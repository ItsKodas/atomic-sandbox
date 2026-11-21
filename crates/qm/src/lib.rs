//! See docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md

pub mod eigen;
pub mod error;
pub mod forces;
pub mod grid;
pub mod ionic;
pub mod nonlocal;
pub mod nuclear;
pub mod occupations;
pub mod orbitals;
pub mod poisson;
pub mod radial;
pub mod scf;
pub mod stencil;
pub mod xc;

pub use error::QmError;
pub use forces::forces;
pub use grid::Grid;
pub use ionic::{IonicPolicy, IonicSetup};
pub use nuclear::Nucleus;
pub use occupations::{
    fd_entropy, fermi_dirac, grown_extras, partial_levels, step_tail_kt, tight_columns, tight_start, witness_bound_reached, witness_bound_reached_tail, witness_keep, witness_keep_tail, witness_keep_within, witness_reached, FdOccupations, PartialLevel, ELECTRON_KT,
    GPU_PARTIAL_TOL, LOOSE_F_FACTOR, STEP_TAIL_C, STEP_TAIL_MIN_KT, MAX_MU_HALVINGS, MU_BRACKET_KT, PARTIAL_OCCUPATION_MIN, WITNESS_GAP, WITNESS_GUARD_TOL, WITNESS_TAIL_KT,
};
pub use scf::{extrapolate_sigma, one_shot_occupations, scf, scf_observed, valence_electrons, Energy, KsState, OneShot, Occupations, ScfProgress, ScfSettings, ScfStats, StallWatch, System, SCF_STALL_WINDOW};
