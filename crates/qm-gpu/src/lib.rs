//! Kohn-Sham solver on wgpu compute (M3). Grid quantities are f32 in padded storage buffers
//! (`grid`), kernels are compiled once per device (`kernels`), reductions are two-level
//! f32 workgroup partials finished in f64 on the CPU (`reduce`), pointwise vector ops in
//! `pointwise`, the Laplacian in `stencil`, the nuclear potential and LSDA XC in `physics`.
//! Many ops record into one compute pass through a `batch::Batch`. The Poisson solver
//! (`poisson`), LOBPCG (`eigen`), the Kohn-Sham SCF (`scf`, the public `GpuSolver`) and the
//! Hellmann-Feynman forces (`forces`, `GpuSolver::forces`) are built on them. The CPU `qm`
//! crate is the reference for every kernel and for the solver.
pub mod batch;
pub mod eigen;
pub mod forces;
pub mod grid;
pub mod kernels;
pub mod physics;
pub mod pointwise;
pub mod poisson;
pub mod reduce;
pub mod scf;
pub mod stencil;

pub use batch::{Batch, Recorder};
pub use eigen::{lobpcg, EigenStats, GpuHamiltonian, LobpcgWork, Stop as EigenStop};
pub use forces::electronic_forces;
pub use grid::{Field, GpuGrid};
pub use kernels::Kernels;
pub use pointwise::Params;
pub use poisson::{GpuPoisson, PoissonStats, Stop};
pub use reduce::{Reducer, Reduction};
pub use scf::{GpuSolver, GpuState, SolveProfile};
