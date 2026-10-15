//! The simulation worker's old path: everything moved to [`crate::worker`] in M6 (Task 1); this
//! re-export keeps every `crate::sim::` path in the app working.
pub use crate::worker::*;
