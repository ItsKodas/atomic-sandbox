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
}
