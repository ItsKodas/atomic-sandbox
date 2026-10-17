//! Error type for the `qm` crate.

#[derive(Debug, thiserror::Error)]
pub enum QmError {
    /// Two nuclei closer than the coincidence threshold (see `nuclear::nuclear_repulsion`);
    /// the point-charge repulsion Z_I Z_J / |R_I − R_J| is undefined.
    #[error("nuclei {0} and {1} coincide")]
    CoincidentNuclei(usize, usize),
    /// The SCF loop hit its iteration cap with the density change Δρ = ∫|ρ_out − ρ_in| still
    /// above tolerance.
    #[error("SCF did not converge after {0} iterations (Δρ = {1:.3e})")]
    ScfNotConverged(usize, f64),
    /// Non-finite or rank-deficient numerics inside a GPU solve (NaN Gram, NaN residual,
    /// p·Ap ≤ 0, a failed Cholesky); the solver discards its state (the next solve is cold).
    /// The message names the check that failed. A caller in a dynamics loop can react (halve
    /// the time step, retry) without rebuilding the solver.
    #[error("numerical failure in the solver: {0}")]
    Numerical(String),
    /// A GPU → CPU readback failed (the device was busy past a wait, or lost): the solve's numbers
    /// cannot be trusted, so it ends here and the solver discards its state like after
    /// `Numerical`. Transient under contention — a caller can retry. The message names the cause.
    #[error("GPU readback failed: {0}")]
    Device(String),
}
