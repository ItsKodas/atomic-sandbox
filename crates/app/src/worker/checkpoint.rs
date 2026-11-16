//! Resumable training (user report 2026-10-20: "save the active training to disk, so if it does
//! crash or close … you can restore … and resume training from where it was"). With
//! [`Command::TrainingCheckpoints`] set (the window: `%LOCALAPPDATA%\atomic-sandbox\training`), a
//! running sampling pass is kept in that folder as two files written together after its solves —
//! at most once per [`CHECKPOINT_INTERVAL`], so a crash loses at most that and the solve in
//! flight: [`MODEL_FILE`] (the surrogate's samples, `Trainable::save`, even before a fit formed) and
//! [`CHECKPOINT_FILE`] (`Trainable::pass_checkpoint`, `providers::sampling`'s "Checkpoints"). Each
//! is written to a `.tmp` and renamed, the model first; a resume refuses a pair whose sample counts
//! disagree. When the pass finishes or is cancelled (Cancel, a new scene, Forget) the files are
//! removed; a shutdown or a solver switch leaves them, which is the point. [`Command::ResumeTraining`]
//! loads the samples file and resumes the checkpoint on the pass's own scene (the viewer sets it
//! first, `Viewer::resume_training`). One folder, one interrupted pass: a new pass replaces it.
//!
//! # Paused passes (user request 2026-11-16)
//! A pass the user paused (`DynCommand::PauseSampling`) is checkpointed in the round that publishes
//! the pause, whatever [`CHECKPOINT_INTERVAL`] says: the files then hold exactly where it stands
//! (the abandoned solve's job is back in its queue). Nothing is written while it stays paused.
//! Cancel while paused removes the files as any Cancel does. A new scene while paused (a preset, a
//! loaded scene, a charge, grid, σ or ionic change: `ReplaceScene`) ends the pass in the driver as
//! it would a running one, but its files are **kept** — the worker lets go of them
//! ([`Worker::set_aside_paused_pass`]) — so the pass is an interrupted one, which the Model tab
//! offers to resume on its own scene; a solver switch or a shutdown keeps them as before.
//!
//! # Collection mode (M8a Task 4b; Task 4a review concern 6)
//! A pass that learns into a shared collection has its samples saved by the collection as they are
//! kept, so only [`CHECKPOINT_FILE`] is written (a stale [`MODEL_FILE`] is removed). Its provenance
//! line names the collection and its counts are the collection's (`SurrogateKs::checkpoint_counts`):
//! the resume attaches the scene's collection and goes on when the collection still holds the
//! samples it held at the checkpoint; jobs whose geometry it already holds are not solved again.
//! A **read-only** collection (another process holds its lock) keeps nothing of the pass, so its
//! pass is not resumable: no checkpoint is written (review I1). Samples a writable collection
//! refuses (box-affected ones) stay session-only and are not in the checkpoint; the autosave keeps
//! them at a solver switch or a close (`autosave` module doc), a crash loses them.

use super::*;

/// The samples file of the interrupted pass.
pub const MODEL_FILE: &str = "pass.model";
/// The pass checkpoint next to it.
pub const CHECKPOINT_FILE: &str = "pass.ckpt";
/// Checkpoints are at most this often (module doc).
pub const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(2);

/// The worker's checkpoint state (module doc).
pub(super) struct Checkpoints {
    dir: PathBuf,
    /// When the last checkpoint was written and the pass's `done` then.
    written: Option<(Instant, usize)>,
    /// This worker wrote (or resumed) the files now in the folder: it removes them when its pass ends.
    owns: bool,
    /// A write failed: said once.
    failed: bool,
    /// The running pass is paused and its checkpoint was written at the pause (module doc,
    /// "Paused passes").
    pause_written: bool,
}

impl Checkpoints {
    pub(super) fn new(dir: PathBuf) -> Checkpoints {
        Checkpoints { dir, written: None, owns: false, failed: false, pause_written: false }
    }

    /// The checkpoint folder.
    pub(super) fn dir(&self) -> &std::path::Path {
        &self.dir
    }
}

/// The interrupted pass in `dir`, if there is one that can be read: its checkpoint text and header.
pub fn interrupted_pass(dir: &std::path::Path) -> Option<(String, providers::sampling::CheckpointInfo)> {
    let text = std::fs::read_to_string(dir.join(CHECKPOINT_FILE)).ok()?;
    let info = providers::sampling::checkpoint_info(&text).ok()?;
    // a per-scene pass needs its samples file; a collection's pass has its samples in the collection
    (dir.join(MODEL_FILE).exists() || is_collection_checkpoint(&info)).then_some((text, info))
}

/// The checkpoint was written in collection mode (module doc, "Collection mode").
pub fn is_collection_checkpoint(info: &providers::sampling::CheckpointInfo) -> bool {
    info.provenance.starts_with(providers::surrogate::COLLECTION_CHECKPOINT_PREFIX)
}

/// Removes the interrupted pass's files from `dir` (the offer's Discard; a pass that ended).
pub fn discard_interrupted_pass(dir: &std::path::Path) {
    for f in [MODEL_FILE, CHECKPOINT_FILE] {
        let _ = std::fs::remove_file(dir.join(f));
    }
}

impl Worker {
    /// After a round (module doc): a running pass is checkpointed when it moved since the last
    /// checkpoint and [`CHECKPOINT_INTERVAL`] has passed; an ended pass's files are removed.
    pub(super) fn checkpoint_round(&mut self, snap: &DynSnapshot) {
        let Some(ck) = self.checkpoints.as_mut() else { return };
        let active = sampling_active(snap);
        let paused = pass_paused(snap);
        if !paused {
            ck.pause_written = false;
        }
        if !active {
            if ck.owns {
                discard_interrupted_pass(&ck.dir);
                (ck.owns, ck.written) = (false, None);
            }
            return;
        }
        let done = snap.sampling.as_ref().map_or(0, |p| p.done);
        // a pause is written at once, then nothing while it lasts (module doc, "Paused passes")
        let at_pause = paused && !ck.pause_written;
        if !at_pause && (paused || ck.written.is_some_and(|(at, d)| d == done || at.elapsed() < CHECKPOINT_INTERVAL)) {
            return;
        }
        ck.pause_written = paused;
        let dir = ck.dir.clone();
        let result = self.write_checkpoint(&dir);
        let ck = self.checkpoints.as_mut().expect("checked");
        match result {
            Ok(true) => (ck.written, ck.owns) = (Some((Instant::now(), done)), true),
            Ok(false) => ck.written = Some((Instant::now(), done)), // a pass that cannot be resumed
            Err(e) => {
                ck.written = Some((Instant::now(), done));
                if !std::mem::replace(&mut ck.failed, true) {
                    self.dy.add_note(format!("training checkpoint: {e} — this pass cannot be resumed if the app closes"));
                }
            }
        }
    }

    /// Writes the samples file and the checkpoint (module doc); `Ok(false)` for a pass that is not
    /// resumable (teach-around, references only).
    fn write_checkpoint(&mut self, dir: &std::path::Path) -> Result<bool, String> {
        let shared = self.in_collection();
        if shared && self.collection_read_only() {
            return Ok(false); // nothing of the pass is durable (module doc, "Collection mode")
        }
        let Some(t) = trainable(&mut self.dy) else { return Ok(false) };
        let Some(text) = t.pass_checkpoint() else { return Ok(false) };
        if t.samples() == 0 {
            return Ok(false); // nothing taught yet: nothing to lose (the samples file needs one)
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let (model, ckpt) = (dir.join(MODEL_FILE), dir.join(CHECKPOINT_FILE));
        let tmp = |p: &std::path::Path| p.with_extension(format!("{}.tmp", p.extension().and_then(|x| x.to_str()).unwrap_or("")));
        if shared {
            // the collection holds the samples (module doc, "Collection mode")
            match std::fs::remove_file(&model) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("{}: {e}", model.display())),
                _ => {}
            }
            std::fs::write(tmp(&ckpt), text).map_err(|e| format!("{}: {e}", ckpt.display()))?;
            std::fs::rename(tmp(&ckpt), &ckpt).map_err(|e| format!("{}: {e}", ckpt.display()))?;
            return Ok(true);
        }
        t.save(&tmp(&model)).map_err(|e| format!("{}: {e}", model.display()))?;
        std::fs::write(tmp(&ckpt), text).map_err(|e| format!("{}: {e}", ckpt.display()))?;
        std::fs::rename(tmp(&model), &model).map_err(|e| format!("{}: {e}", model.display()))?;
        std::fs::rename(tmp(&ckpt), &ckpt).map_err(|e| format!("{}: {e}", ckpt.display()))?;
        Ok(true)
    }

    /// A scene replacement ended a paused pass (module doc, "Paused passes"): its files, written
    /// at the pause, are left in the folder as an interrupted pass — never removed by this worker —
    /// and the Log says so. `was_paused`: the pass was paused before the replacement.
    pub(super) fn set_aside_paused_pass(&mut self, was_paused: bool) {
        if !was_paused || sampling_active(&self.dy.snapshot()) {
            return;
        }
        let Some(ck) = self.checkpoints.as_mut().filter(|ck| ck.owns) else { return };
        (ck.owns, ck.written, ck.pause_written) = (false, None, false);
        self.dy.add_note("the paused training pass ended with the scene change: its checkpoint is kept — the Model tab offers to resume it on its own scene".into());
    }

    /// [`Command::ResumeTraining`] (module doc).
    pub(super) fn resume_training(&mut self, dir: &std::path::Path) {
        if trainable(&mut self.dy).is_none() {
            return self.error("resume the interrupted training: the estimated solver is not running".into());
        }
        let text = match std::fs::read_to_string(dir.join(CHECKPOINT_FILE)) {
            Ok(t) => t,
            Err(e) => return self.error(format!("resume the interrupted training: {}: {e}", dir.join(CHECKPOINT_FILE).display())),
        };
        if dir.join(MODEL_FILE).exists() {
            self.dy.apply(DynCommand::LoadModel(dir.join(MODEL_FILE)));
            if self.dy.snapshot().last_rejection.is_some() {
                return;
            }
            self.use_collection = false;
        } else {
            // a collection's pass (module doc, "Collection mode"): its samples are in the collection
            self.use_collection = true;
            self.sync_collection();
        }
        self.owns_autosave = true;
        self.pass_offer = None;
        self.sizing = None;
        self.dy.apply(DynCommand::ResumeSampling(text));
        if self.dy.snapshot().last_rejection.is_none() {
            if let Some(ck) = self.checkpoints.as_mut() {
                // the files are this pass's now: rewritten as it goes, removed when it ends
                ck.owns = true;
                ck.written = self.dy.snapshot().sampling.map(|p| (Instant::now(), p.done));
            }
        }
    }
}
