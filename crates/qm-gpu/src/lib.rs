//! Kohn-Sham solver on wgpu compute (M3). Grid quantities are f32 in padded storage buffers
//! (`grid`), kernels are compiled once per device (`kernels`), reductions are two-level
//! f32 workgroup partials finished in f64 on the CPU (`reduce`), pointwise vector ops in
//! `pointwise`. The CPU `qm` crate is the reference for every kernel.
pub mod grid;
pub mod kernels;
pub mod pointwise;
pub mod reduce;

pub use grid::{Field, GpuGrid};
pub use kernels::Kernels;
pub use pointwise::Params;
pub use reduce::Reducer;
