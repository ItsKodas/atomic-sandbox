//! Surrogate force provider: a linear atomic-cluster-expansion (ACE-style) potential trained on
//! this project's own Kohn-Sham solver (spec `docs/superpowers/specs/2026-10-13-m5-surrogate-provider-design.md`).
//!
//! Model, in atomic units (Hartree, Bohr) throughout: E = Σ_i E_i with E_i = E_atom(Z_i) +
//! Σ_k c_{Z_i,k} B_k(neighbourhood of i). The one-particle basis is φ_{nlm}(r_ij, Z_j) =
//! R_n(r_ij)·f_c(r_ij)·S_lm(r_ij)/|r_ij|^l on a species channel Z_j, with R_n Chebyshev
//! polynomials in a scaled distance, f_c the cosine cutoff that takes E and F continuously to
//! zero at r_cut, and S_lm the real solid harmonics (Racah normalised, [`basis`]). The atomic
//! basis A_{nlm,Z} = Σ_{j: Z_j = Z} φ_{nlm} gives the invariants B: body-2 A_{n00,Z}, body-3
//! Σ_m A_{n₁lm,Z₁} A_{n₂lm,Z₂} (the contraction that is rotation invariant for REAL harmonics),
//! under a total-degree cap. Forces are the analytic gradient −∂E/∂r_k through the chain rule;
//! the coefficients c come from a ridge-regularised least-squares fit to energies and forces,
//! whose Bayesian predictive variance drives the fall-back to the DFT teacher. The species
//! channel is a categorical index (CLAUDE.md rule 1: the only per-element inputs are Z and the
//! isotope mass; everything element-specific is learned).

pub mod basis;
pub mod descriptor;
pub mod fit;
pub mod linalg;
pub mod model;

pub use descriptor::{describe, AtomDescriptor, Basis, DescribeError};
pub use fit::{
    fit, Accumulator, FitError, FitOptions, FitReport, Sample, SampleStore, SampleTag, StoreError,
};
pub use model::{energy_and_forces, Model, Prediction, Provenance};
