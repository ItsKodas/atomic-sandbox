//! Kohn-Sham solver on wgpu compute (M3). Grid quantities are f32 in padded storage buffers
//! (`grid`), kernels are compiled once per device (`kernels`), reductions are two-level
//! f32 workgroup partials finished in f64 on the CPU (`reduce`), pointwise vector ops in
//! `pointwise`, the Laplacian in `stencil`, the nuclear potential and LSDA XC in `physics`.
//! Many ops record into one compute pass through a `batch::Batch`. The CPU `qm` crate is the
//! reference for every kernel.
pub mod batch;
pub mod eigen;
pub mod grid;
pub mod kernels;
pub mod physics;
pub mod pointwise;
pub mod poisson;
pub mod reduce;
pub mod stencil;

pub use batch::{Batch, Recorder};
pub use eigen::{lobpcg, EigenStats, GpuHamiltonian, LobpcgWork};
pub use grid::{Field, GpuGrid};
pub use kernels::Kernels;
pub use pointwise::Params;
pub use poisson::{GpuPoisson, PoissonStats, Stop};
pub use reduce::Reducer;
