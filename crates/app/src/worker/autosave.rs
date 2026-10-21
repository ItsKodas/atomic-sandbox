//! The session autosave with ownership (the parent module doc, "The estimated solver",
//! "Autosave"): the model is written to the autosave path at shutdown — only once this worker owns
//! the file — or the file removed when there is nothing to keep (samples taught before a fit formed are kept). Split out of `app/src/sim.rs` in M6
//! Task 1.
//!
//! **In collection mode** (ruling P13; M8a Task 4b fix round 1) the collection saves itself, so the
//! autosave holds only the samples it does not — the session-only ones (refused by it, or all of a
//! read-only collection's: `Trainable::save_session_only`) — and is removed when there are none. The
//! next worker puts them back beside the collection it attaches (`model` module doc, "The shared
//! collection"); such a file is never a switch to a model file.

use super::*;

impl Worker {
    /// Writes the surrogate's model to `path` (a fitted model, or the samples taught before a fit
    /// formed: a samples-only file, `surrogate::sidecar` version 6 — the 2026-10-20 PH₃ session
    /// lost minutes of DFT when its P reference did not solve and this removed the unfitted
    /// state), or removes `path` (nothing held); the error message on failure.
    pub(super) fn save_model_to(&mut self, path: &std::path::Path) -> Result<bool, String> {
        // a shared collection saves itself (ruling P13, module doc): its session-only samples are
        // kept here; with none the file is removed (a stale one of a file chosen earlier too)
        let shared = self.in_collection();
        let Some(s) = trainable(&mut self.dy) else { return Ok(false) };
        if shared {
            if s.save_session_only(path).map_err(|e| format!("autosave {}: {e}", path.display()))? > 0 {
                return Ok(true);
            }
        } else if s.fitted() || s.samples() > 0 {
            s.save(path).map_err(|e| format!("autosave {}: {e}", path.display()))?;
            return Ok(true);
        }
        match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("autosave {}: {e}", path.display())),
            _ => Ok(false),
        }
    }

    /// The provider learns into a shared collection (`sim::ModelSource::Collection`; cheap: no
    /// readouts are built — review Minor 4).
    pub(super) fn in_collection(&mut self) -> bool {
        trainable(&mut self.dy).is_some_and(|t| matches!(t.model_source(), sim::ModelSource::Collection { .. }))
    }

    /// The shared collection attached is read-only (another process holds its lock).
    pub(super) fn collection_read_only(&mut self) -> bool {
        trainable(&mut self.dy).is_some_and(|t| matches!(t.model_source(), sim::ModelSource::Collection { read_only: true, .. }))
    }

    /// Shutdown (`worker` module doc, "Autosave"): the model to the autosave path — only if this worker
    /// owns it — then `Retired`.
    pub(super) fn retire(&mut self) {
        if let Some(path) = self.autosave.clone().filter(|_| self.owns_autosave) {
            if let Err(e) = self.save_model_to(&path) {
                self.error(e);
            }
        }
        let _ = self.tx.send(Event::Retired);
    }
}
