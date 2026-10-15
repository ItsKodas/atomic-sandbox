//! The force providers (M6, Task 1: moved out of `sim`, which keeps the boundary — the
//! `ForceProvider` and `Trainable` traits and their data types — and the driver): the Kohn-Sham
//! adapters ([`ks`]: `CpuKs` on `qm`, `GpuKs` on `qm-gpu`), the surrogate ([`surrogate`]:
//! `SurrogateKs`, a fitted model that hands what it cannot answer to its Kohn-Sham teacher and
//! learns from it, `Trainable`) and its sampling pass ([`sampling`]). `sim` never names these
//! types; the app builds them and hands them to `sim::Dynamics` as `Box<dyn ForceProvider>`.
//! [`orbitals`] holds what the Kohn-Sham adapters share for the orbital display (plan Task 3 of
//! the orbital viewer).
pub mod ks;
pub mod orbitals;
pub mod sampling;
pub mod surrogate;

pub use ks::{field_bytes, gpu_footprint_bytes, orbital_view_footprint_bytes, solve_footprint_bytes, CpuKs, GpuKs};
pub use sampling::{atom_key, run_sampling, PairScan, PlanSolves, Sampler};
pub use surrogate::{box_affected_face_density, FallbackPolicy, SurrogateKs, TrainingResiduals};
