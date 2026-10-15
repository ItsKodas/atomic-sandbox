//! See docs/superpowers/specs/2026-10-12-atomic-sandbox-design.md

pub mod eigen;
pub mod error;
pub mod forces;
pub mod grid;
pub mod ionic;
pub mod nonlocal;
pub mod nuclear;
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
pub use scf::{extrapolate_sigma, scf, valence_electrons, Energy, KsState, ScfSettings, ScfStats, System};
