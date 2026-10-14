//! Isolated-atom reference energies (M6 spec §3.2), keyed by the settings the atom was solved at
//! and its Z. **A stub in M6 Task 1**: it carries exactly what the surrogate's reference cache
//! held before the providers moved out of `sim` (the cache that lets a later sampling pass skip a
//! reference scan already solved at the same settings — `providers::sampling`, "References"), in
//! types `sim` owns, so [`Trainable::start_pass`](crate::Trainable::start_pass) can take it.
//! Task 3 makes the driver its owner and adds the radial densities.

/// The settings an isolated atom was solved at — the atom's own provenance: the scene's grid
/// (`grid_n`, `h`), σ, the teacher and its SCF tolerance; neutral, at its lowest-energy
/// multiplicity (ruling S9) — and its Z. Two keys are equal when every field is.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceKey {
    pub grid_n: usize,
    pub h: f64,
    pub sigma: f64,
    pub tol: f64,
    pub teacher: String,
    pub z: u8,
}

/// One isolated-atom reference: the energy (Hartree) and the multiplicity that won the scan
/// (`None` for one carried by a loaded model).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomReference {
    pub energy: f64,
    pub multiplicity: Option<u32>,
}

/// The references solved so far (module doc).
#[derive(Clone, Debug, Default)]
pub struct AtomReferences {
    entries: Vec<(ReferenceKey, AtomReference)>,
}

impl AtomReferences {
    /// The reference stored under `key`, if any.
    pub fn get(&self, key: &ReferenceKey) -> Option<AtomReference> {
        self.entries.iter().find(|(k, _)| k == key).map(|e| e.1)
    }

    /// Stores `r` under `key`, replacing an entry with the same key.
    pub fn insert(&mut self, key: ReferenceKey, r: AtomReference) {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = r,
            None => self.entries.push((key, r)),
        }
    }

    /// Every entry, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &(ReferenceKey, AtomReference)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
