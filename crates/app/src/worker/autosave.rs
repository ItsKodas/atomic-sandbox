//! The session autosave with ownership (the parent module doc, "The estimated solver",
//! "Autosave"): the model is written to the autosave path at shutdown — only once this worker owns
//! the file — or the file removed when there is nothing to keep (samples taught before a fit formed are kept). Split out of `app/src/sim.rs` in M6
//! Task 1.

use super::*;

impl Worker {
    /// Writes the surrogate's model to `path` (a fitted model, or the samples taught before a fit
    /// formed: a samples-only file, `surrogate::sidecar` version 6 — the 2026-10-20 PH₃ session
    /// lost minutes of DFT when its P reference did not solve and this removed the unfitted
    /// state), or removes `path` (nothing held); the error message on failure.
    pub(super) fn save_model_to(&mut self, path: &std::path::Path) -> Result<bool, String> {
        let Some(s) = trainable(&mut self.dy) else { return Ok(false) };
        if s.fitted() || s.samples() > 0 {
            s.save(path).map_err(|e| format!("autosave {}: {e}", path.display()))?;
            Ok(true)
        } else {
            match std::fs::remove_file(path) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("autosave {}: {e}", path.display())),
                _ => Ok(false),
            }
        }
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
