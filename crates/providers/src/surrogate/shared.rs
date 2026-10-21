//! The shared collection (M8a Task 4a; spec `docs/superpowers/specs/2026-10-20-m8-shared-model-and-sandbox-design.md`
//! §4.2, plan `docs/superpowers/plans/2026-10-21-m8a-shared-model.md` Task 4): the surrogate learns
//! into the one append-only store of its solver surface (`surrogate::Collection`) instead of a
//! per-scene store. Storage and the admission rules are the `surrogate` crate's (`collection`
//! module doc); this module is how the provider uses them. Atomic units throughout.
//!
//! # Collection mode
//!
//! [`SurrogateKs::attach`] (and `Trainable::attach_collection`, which opens the folder of a scene's
//! surface) replaces what the provider held — like a load — with the collection's samples (their
//! latest quarantine codes), its references and its audit history, and fits when a fit can form.
//! From then on:
//!
//! * **The store mirrors the collection.** Its provenance is the collection's (`shared_provenance`:
//!   the key's h, σ, tolerance, teacher and ionic tag, any box — `grid_n` 0 —, neutral, automatic
//!   multiplicity), and a sample's provenance check is the collection's **surface-key check**
//!   (`Collection::admit_provenance`: charge 0, no forced multiplicity, the same key — grid n,
//!   charge and multiplicity of a scene are not compared beyond that, controller ruling). The same
//!   check replaces the full `Provenance` comparison everywhere the model is matched to a scene:
//!   rule 3 of the decision list, an audit's judgement, the refit jump, the readouts. Per-scene
//!   stores keep today's rule unchanged.
//! * **Every kept teacher answer is offered to the collection** — live fallbacks, audits (in the
//!   step and background), pass and teach-around samples — through `Collection::admit` and
//!   `append` (flushed and synced before the step returns), with [`Q_BOX`] passed for an answer
//!   whose face density is above [`box_affected_face_density`]`(h)` even while the provider keeps
//!   it in its own fit at code 0, and with the multiplicity the teacher solved at
//!   (`EvalStats::multiplicity`; `None` when it reports none). An answer the collection refuses
//!   takes **today's per-scene path**: it stays in the provider's store for this session (fitted or
//!   not by the same rules as before) and is never written to the collection; a live step's note
//!   says so ("kept for this session only — …"), a pass's refusals are summed in one note at its
//!   end. Refusal reasons are the collection's own words (box-affected, box status unknown — a
//!   teacher that reports no face density —, quarantined, a duplicate, read-only).
//! * **A draft-mode guess's audit** (`Q_DRAFT`, held out of the strict fit) waits outside the
//!   collection and is offered again if a refit promotes it into the fit (`judge_held_out`).
//! * **Status changes are appended**: whenever the provider's quarantine code of a collection
//!   sample changes (held, superseded, released), a status record is written
//!   (`sync_collection_status`, after every refit and every admitted teacher answer).
//! * **Species growth** is today's `species_dirty` path: a sample with a new element marks the
//!   basis dirty and the next refit rebuilds the accumulator from the whole store — every earlier
//!   sample included — with the enlarged species set (it needs the new species' reference).
//! * **References** (the Task 3 review's physics (c), controller ruling): a solved reference
//!   ([`SurrogateKs::set_reference_solved`], what the sampling pass calls) is offered with its box
//!   and face density (`Collection::offer_reference`: the box rule, odd n — the atom on a grid
//!   node —, the largest passing box stands). The fit uses the collection's reference; one the
//!   collection refuses is used for this session only when the collection has none, with a note.
//!   A reference set without its box (`set_reference_energy`) is session-only.
//! * **The audit history** is the collection's (`history.bin`): installed at attach, written back
//!   (with `README.txt`) after every refit, at detach and when the provider is dropped. The class
//!   verdict's hysteresis memory is not persisted in a collection (a reattached collection starts
//!   without one, like a sidecar of version < 7).
//!
//! **Not in collection mode:** a scene that cannot share the collection (charged, a forced
//! multiplicity, another surface) is refused by rule 3 — the teacher answers, the sample is not
//! kept, the note names the reason (spec: charged scenes keep per-scene models — the caller
//! decides not to attach for them; `Trainable::attach_collection` refuses such a scene). A sampling
//! pass on such a scene is refused. `save` is refused (the collection saves itself as it learns;
//! a specialised model file is plan Task 6's). `load` of a model file **switches** to it (the
//! collection is detached: a specialised, per-scene model); `forget` detaches too and never
//! touches the collection on disk.
//!
//! # Who may open a collection for writing
//!
//! One writable handle per process (`Collection::open`: a second open in the same process is
//! read-only): the provider owns it. Anything else that needs to write — importing a `.model`
//! file — goes through the provider ([`SurrogateKs::import_into_collection`],
//! `Trainable::import_model`); readers (a README viewer, the knowledge map) may open a read-only
//! handle, which never writes and never removes the owner's lock. `attach_collection` on the
//! collection already attached is a no-op, never a second open.

use super::*;
use std::path::{Path, PathBuf};
use surrogate::collection::{CollectionReference, ImportReport, ReferenceOutcome, REFUSE_BOX_UNKNOWN};
use surrogate::{Collection, CollectionSample, SurfaceKey};

/// Where a stored sample stands in the attached collection (parallel to the store).
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Link {
    /// In the collection at this index.
    In(usize),
    /// Kept for this session only: the collection refused it for good (module doc).
    Out,
    /// A held-out draft guess (`Q_DRAFT`): offered again once a refit promotes it into the fit.
    Waiting { grid_n: usize, multiplicity: Option<u32> },
}

/// The attached collection and the provider's bookkeeping of it (module doc).
pub(super) struct SharedCollection {
    pub(super) coll: Collection,
    /// Per stored sample: its place in the collection.
    pub(super) link: Vec<Link>,
    /// The store's provenance in collection mode ([`shared_provenance`]).
    pub(super) prov: Provenance,
    /// Samples appended since the counter was last read (a pass's summary).
    appended: usize,
    /// Samples refused since the counter was last read, by reason.
    refused: BTreeMap<String, usize>,
    /// The audit history's [`history_stamp`] when it was last written.
    history_written: Option<(u64, u64)>,
    /// A write failed (or the collection is read-only): said once.
    write_noted: bool,
}

/// The store's provenance in collection mode: the key's settings, any box (`grid_n` 0), neutral,
/// automatic multiplicity — what `Collection::admit_provenance` accepts, and nothing it does not.
pub(super) fn shared_provenance(key: &SurfaceKey) -> Provenance {
    Provenance {
        grid_n: 0,
        h: key.h,
        sigma: key.sigma,
        tol: key.tol,
        charge: 0,
        multiplicity: None,
        teacher: key.teacher.clone(),
        masses: Vec::new(),
        ionic: key.ionic.clone(),
    }
}

/// What changes when the audit history changes: its recorded points and its unrecorded counts.
fn history_stamp(h: &surrogate::AuditHistory) -> (u64, u64) {
    let u = &h.unrecorded;
    (h.recorded(), u.no_variance + u.box_affected + u.draft + u.unfit)
}

/// A geometry's formula in Hill order (C, then H, then the rest alphabetically; without carbon all
/// alphabetically) — the name a collection sample is filed under (the scene has no name here).
pub(super) fn formula(z: &[u8]) -> String {
    let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    for &zi in z {
        *count.entry(symbol(zi)).or_default() += 1;
    }
    let mut keys: Vec<&str> = count.keys().copied().collect();
    if count.contains_key("C") {
        keys.retain(|&k| k != "C" && k != "H");
        keys.insert(0, "C");
        if count.contains_key("H") {
            keys.insert(1, "H");
        }
    }
    keys.iter().map(|&k| if count[k] == 1 { k.to_string() } else { format!("{k}{}", count[k]) }).collect()
}

impl SurrogateKs {
    /// Learns into `c` (module doc, "Collection mode"): what the provider held is replaced — like a
    /// load — by the collection's samples, references and audit history, and a fit is made when one
    /// can form. `Err`, nothing changed, while a pass runs or for a collection of another teacher or
    /// tolerance than this provider's (its h, σ and ionic tag are the scenes' to match).
    pub fn attach(&mut self, c: Collection) -> Result<(), String> {
        if self.pass.is_some() {
            return Err("shared collection: cancel the running sampling pass first".into());
        }
        if c.key.teacher != self.teacher.name() {
            return Err(format!("shared collection {}: it was taught by '{}', this provider's teacher is '{}'", c.key.slug(), c.key.teacher, self.teacher.name()));
        }
        if c.key.tol != self.teacher_tol {
            return Err(format!("shared collection {}: it was taught at tol {:e}, this provider's teacher solves at {:e}", c.key.slug(), c.key.tol, self.teacher_tol));
        }
        let prov = shared_provenance(&c.key);
        let mut store = SampleStore::new(prov.clone());
        for cs in &c.samples {
            store.push(cs.sample.clone()).map_err(|e| format!("shared collection {}: {e}", c.key.slug()))?;
        }
        let held = self.samples();
        // a clean slate (a previous collection is flushed and released)
        self.forget();
        let n = store.len();
        self.face = vec![f64::NAN; n];
        self.quarantine = c.quarantine.clone();
        self.store = (n > 0).then_some(store);
        for (&z, r) in &c.references {
            self.refs.insert(z, r.energy);
        }
        self.refs_dirty = true;
        self.pending_refit = n > 0;
        // the collection's audit history (module doc); its generations are the writers', so this
        // provider's generation moves past them (as `install_sidecar` does)
        let newest = c.history.iter().map(|p| p.model_gen).max().unwrap_or(0);
        self.audit_history = c.history.clone();
        self.model_gen = self.model_gen.max(newest) + 1;
        let slug = c.key.slug();
        let held_out = c.quarantine.iter().filter(|&&q| q != Q_NONE).count();
        let read_only = c.read_only;
        let stamp = history_stamp(&c.history);
        self.shared = Some(SharedCollection {
            link: (0..n).map(Link::In).collect(),
            coll: c,
            prov,
            appended: 0,
            refused: BTreeMap::new(),
            history_written: Some(stamp),
            write_noted: false,
        });
        self.source_file = None;
        let mut note = format!("shared collection {slug} attached: {n} samples ({held_out} held out of the fit), {} references", self.refs.len());
        if held > 0 {
            note.push_str(&format!(" — the {held} samples held before are not in it (import a saved model file to add them)"));
        }
        if read_only {
            note.push_str(" — read-only: another process holds its lock, so this session's answers are used and not added to it");
        }
        self.user_notes.push(note);
        if self.no_model().is_some_and(|nm| nm.missing.is_empty() && nm.retry) {
            if let Err(e) = SurrogateKs::refit(self) {
                self.last_fit_error = Some(e.to_string());
            }
        }
        Ok(())
    }

    /// Opens the collection of `scene`'s solver surface under `root` and attaches it
    /// (`Trainable::attach_collection`; module doc): a no-op when it is attached already. `Err`
    /// leaves the provider as it was — a collection attached before stays attached (the caller
    /// decides what a charged scene learns into: `forget` or a model file makes it per-scene).
    pub fn attach_collection_at(&mut self, root: &Path, scene: &Scene) -> Result<(), String> {
        let prov = self.scene_provenance(scene);
        // the scene's own reasons first, in the collection's words (a charged scene keeps a
        // per-scene model)
        if prov.charge != 0 || prov.multiplicity.is_some() {
            let why = if prov.charge != 0 { format!("charge {}: collections are neutral until M9", prov.charge) } else { format!("the scene forced multiplicity {}", prov.multiplicity.unwrap_or(0)) };
            return Err(format!("this scene keeps a per-scene model ({why})"));
        }
        let key = prov.surface_key();
        if self.shared.as_ref().is_some_and(|s| s.coll.key == key) {
            return Ok(());
        }
        if self.pass.is_some() {
            return Err("shared collection: cancel the running sampling pass first".into());
        }
        // another surface's collection, if one is attached, is released by `attach` once this one
        // has opened (a different folder, so its lock is no obstacle)
        let (c, report) = Collection::open(root, &key)?;
        self.user_notes.extend(report.notes.into_iter().map(|n| format!("shared collection {}: {n}", key.slug())));
        self.attach(c)
    }

    /// Writes the audit history and the README of the attached collection, then lets it go
    /// (its lock released). `None` when none was attached.
    pub(super) fn detach_collection(&mut self) -> Option<Collection> {
        self.flush_collection();
        self.shared.take().map(|s| s.coll)
    }

    /// The collection attached, if any (a read-only view).
    pub fn collection(&self) -> Option<&Collection> {
        self.shared.as_ref().map(|s| &s.coll)
    }

    /// The collection's link of each stored sample (tests and readouts): `Some(k)` = its index in
    /// the collection, `None` = kept for this session only.
    pub fn collection_index(&self, i: usize) -> Option<usize> {
        match self.shared.as_ref()?.link.get(i)? {
            Link::In(k) => Some(*k),
            _ => None,
        }
    }

    /// Where the model comes from (`TrainableStatus::model_source`).
    pub(super) fn model_source(&self) -> sim::ModelSource {
        match (&self.shared, &self.source_file) {
            (Some(s), _) => sim::ModelSource::Collection { slug: s.coll.key.slug(), samples: s.coll.samples.len(), read_only: s.coll.read_only },
            (None, Some(p)) => sim::ModelSource::File(p.clone()),
            (None, None) => sim::ModelSource::None,
        }
    }

    /// Whether a sample solved at `theirs` belongs to the surface of a store or model at `ours`:
    /// in collection mode both must pass the collection's surface-key check (module doc), else the
    /// full per-scene `Provenance` comparison of before.
    pub(super) fn same_surface(&self, ours: &Provenance, theirs: &Provenance) -> bool {
        match &self.shared {
            Some(s) => s.coll.admit_provenance(ours).is_ok() && s.coll.admit_provenance(theirs).is_ok(),
            None => ours == theirs,
        }
    }

    /// In collection mode, why a scene solved at `prov` cannot share the collection; `None` when it
    /// can, or without a collection.
    pub(super) fn collection_refuses(&self, prov: &Provenance) -> Option<String> {
        let s = self.shared.as_ref()?;
        s.coll.admit_provenance(prov).err().map(|why| format!("the shared collection {} does not take this scene ({why}) — a per-scene model learns it", s.coll.key.slug()))
    }

    /// Into the store in collection mode (`push_sample`'s branch): the surface-key check, then the
    /// store (created with the collection's provenance), then the offer to the collection.
    pub(super) fn push_shared(&mut self, s: Sample, prov: &Provenance, face: f64, multiplicity: Option<u32>) -> Result<(), StoreError> {
        // the note describes this sample only (a pass sample's refusal must not reach a later step)
        self.collection_note = None;
        let sh = self.shared.as_ref().expect("collection mode");
        if sh.coll.admit_provenance(prov).is_err() {
            return Err(StoreError::ProvenanceMismatch);
        }
        let n = self.store.as_ref().map_or(0, |st| st.len());
        if !self.quarantine.is_empty() {
            self.quarantine.resize(n, Q_NONE);
        }
        let shared_prov = sh.prov.clone();
        let store = self.store.get_or_insert_with(|| SampleStore::new(shared_prov));
        store.push(s)?;
        store.provenance.record_masses(&prov.masses);
        self.face.resize(n, f64::NAN);
        self.face.push(face);
        let sh = self.shared.as_mut().expect("collection mode");
        sh.link.resize(n, Link::Out);
        sh.link.push(Link::Out);
        let link = self.offer(n, prov, multiplicity);
        self.shared.as_mut().expect("collection mode").link[n] = link;
        Ok(())
    }

    /// Offers stored sample `i` (solved at `prov`) to the collection: its link afterwards. The
    /// code passed to `admit` is `Q_BOX` for a box-affected answer, `Q_DRAFT` for a draft guess's
    /// audit, else the provider's own; an answer without a face density is refused as box status
    /// unknown (module doc).
    fn offer(&mut self, i: usize, prov: &Provenance, multiplicity: Option<u32>) -> Link {
        let s = self.store.as_ref().expect("stored").samples[i].clone();
        let face = self.face.get(i).copied().unwrap_or(f64::NAN);
        let draft = s.tag == SampleTag::DraftAudit || self.quarantine.get(i) == Some(&Q_DRAFT);
        let code = if face > box_affected_face_density(prov.h) {
            Q_BOX
        } else if draft {
            Q_DRAFT
        } else {
            self.quarantine.get(i).copied().unwrap_or(Q_NONE)
        };
        let sh = self.shared.as_mut().expect("collection mode");
        let verdict = match sh.coll.admit(&s, prov, code) {
            Err(why) => Err(why),
            Ok(()) if !face.is_finite() => Err(REFUSE_BOX_UNKNOWN.to_string()),
            Ok(()) => {
                let k = sh.coll.samples.len();
                let cs = CollectionSample { sample: s.clone(), grid_n: prov.grid_n, multiplicity_forced: false, multiplicity, scene: formula(&s.z) };
                sh.coll.append(cs).map(|()| k)
            }
        };
        match verdict {
            Ok(k) => {
                sh.appended += 1;
                Link::In(k)
            }
            Err(why) => {
                *sh.refused.entry(why.clone()).or_default() += 1;
                self.collection_note = Some(why);
                if code == Q_DRAFT {
                    Link::Waiting { grid_n: prov.grid_n, multiplicity }
                } else {
                    Link::Out
                }
            }
        }
    }

    /// The live step's words for a sample the collection refused (taken once: `keep_sample`).
    pub(super) fn take_collection_note(&mut self) -> Option<String> {
        let why = self.collection_note.take()?;
        let slug = self.shared.as_ref().map_or_else(String::new, |s| s.coll.key.slug());
        Some(format!("surrogate: kept for this session only — the shared collection {slug} did not take it ({why})"))
    }

    /// Writes a status record for every collection sample whose quarantine code changed, and
    /// offers a promoted draft guess again (module doc). A failed write is said once.
    pub(super) fn sync_collection_status(&mut self) {
        let Some(sh) = self.shared.as_mut() else { return };
        let mut failed: Option<String> = None;
        let mut promoted: Vec<(usize, usize, Option<u32>)> = Vec::new();
        for (i, link) in sh.link.iter().enumerate() {
            let code = self.quarantine.get(i).copied().unwrap_or(Q_NONE);
            match *link {
                Link::In(k) => {
                    if sh.coll.quarantine.get(k).is_some_and(|&c| c != code) {
                        if let Err(e) = sh.coll.set_status(k, code) {
                            failed.get_or_insert(e);
                        }
                    }
                }
                Link::Waiting { grid_n, multiplicity } if code == Q_NONE => promoted.push((i, grid_n, multiplicity)),
                _ => {}
            }
        }
        if let Some(e) = failed {
            self.note_write_failure(&e);
        }
        for (i, grid_n, multiplicity) in promoted {
            let mut prov = self.shared.as_ref().expect("collection mode").prov.clone();
            prov.grid_n = grid_n;
            let link = self.offer(i, &prov, multiplicity);
            self.collection_note = None;
            self.shared.as_mut().expect("collection mode").link[i] = link;
        }
    }

    fn note_write_failure(&mut self, e: &str) {
        if let Some(sh) = self.shared.as_mut().filter(|s| !s.write_noted) {
            sh.write_noted = true;
            let slug = sh.coll.key.slug();
            self.user_notes.push(format!("shared collection {slug}: {e} — this session's changes are used, not all are written"));
        }
    }

    /// Writes the audit history (when it grew) and the README of the attached collection.
    pub(super) fn flush_collection(&mut self) {
        let Some(sh) = self.shared.as_mut() else { return };
        if sh.coll.read_only {
            return;
        }
        let stamp = history_stamp(&self.audit_history);
        let mut failed = None;
        if sh.history_written != Some(stamp) {
            sh.coll.history = self.audit_history.clone();
            match sh.coll.save_history() {
                Ok(()) => sh.history_written = Some(stamp),
                Err(e) => failed = Some(e),
            }
        }
        if let Err(e) = sh.coll.write_readme() {
            failed.get_or_insert(e);
        }
        if let Some(e) = failed {
            self.note_write_failure(&e);
        }
    }

    /// The pass's summary of what the collection took (`finish_pass`): `None` without a collection
    /// or when nothing was offered since the last summary.
    pub(super) fn take_collection_summary(&mut self) -> Option<String> {
        let sh = self.shared.as_mut()?;
        let refused: usize = sh.refused.values().sum();
        if sh.appended == 0 && refused == 0 {
            return None;
        }
        let why: Vec<String> = std::mem::take(&mut sh.refused).into_iter().map(|(w, n)| format!("{n} {w}")).collect();
        let appended = std::mem::take(&mut sh.appended);
        let mut text = format!("shared collection {}: {appended} samples added ({} in all)", sh.coll.key.slug(), sh.coll.samples.len());
        if refused > 0 {
            text.push_str(&format!("; {refused} kept for this session only — {}", why.join("; ")));
        }
        Some(text)
    }

    /// A solved isolated-atom reference of species `z` (Hartree) with its box (`grid_n`, the solve's
    /// face density): the sampling pass's setter. Without a collection it is
    /// [`set_reference_energy`](Self::set_reference_energy); in collection mode it is offered to the
    /// collection first and the fit uses the collection's reference (module doc, "References").
    pub fn set_reference_solved(&mut self, z: u8, e: f64, grid_n: usize, face: Option<f64>) {
        let Some(sh) = self.shared.as_mut() else { return self.set_reference_energy(z, e) };
        let limit = box_affected_face_density(sh.coll.key.h);
        let slug = sh.coll.key.slug();
        let name = symbol(z);
        let offered = CollectionReference { energy: e, grid_n: Some(grid_n), face_density: face };
        let (value, note) = match sh.coll.offer_reference(z, offered, limit) {
            Ok(ReferenceOutcome::Stored) => (e, format!("shared collection {slug}: the {name} reference {e:.6} Ha ({grid_n}³) is stored")),
            Ok(ReferenceOutcome::Replaced { previous }) => (
                e,
                format!(
                    "shared collection {slug}: the {name} reference {e:.6} Ha ({grid_n}³) replaces the one from {} ({:.6} Ha, Δ {:+.1e} Ha) — the larger box stands; the fit is rebuilt on it",
                    previous.grid_n.map_or("an unrecorded box".to_string(), |n| format!("{n}³")),
                    previous.energy,
                    e - previous.energy
                ),
            ),
            Ok(ReferenceOutcome::Kept { stored }) => (
                stored.energy,
                format!(
                    "shared collection {slug}: the {name} reference from {} ({:.6} Ha) stands — this {grid_n}³ solve ({e:.6} Ha) is no larger a box (Δ {:+.1e} Ha)",
                    stored.grid_n.map_or("an unrecorded box".to_string(), |n| format!("{n}³")),
                    stored.energy,
                    e - stored.energy
                ),
            ),
            Ok(ReferenceOutcome::Refused(why)) => match sh.coll.references.get(&z) {
                Some(r) => (r.energy, format!("shared collection {slug}: the {name} reference of this {grid_n}³ solve is not taken ({why}); the collection's ({:.6} Ha) stands", r.energy)),
                None => (e, format!("shared collection {slug}: the {name} reference of this {grid_n}³ solve is not taken ({why}) — used for this session only; the collection needs one solved in a box that passes the box rule, with odd n")),
            },
            Err(w) => (
                sh.coll.references.get(&z).map_or(e, |r| r.energy),
                format!("shared collection {slug}: the {name} reference could not be written ({w}) — used for this session only"),
            ),
        };
        self.user_notes.push(note);
        self.set_reference_energy(z, value);
    }

    /// Imports a `.model` file into the attached collection (module doc, "Who may open a
    /// collection"): `Collection::import_sidecar` with the box rule of the collection's h; the
    /// samples it took join the store (the next refit fits them). `Err` without a collection, while
    /// a pass runs, or from the import (an unreadable file, a read-only collection).
    pub fn import_into_collection(&mut self, path: &Path) -> Result<ImportReport, String> {
        if self.pass.is_some() {
            return Err("import: cancel the running sampling pass first".into());
        }
        let sh = self.shared.as_mut().ok_or("import: no shared collection is attached (the Model tab's shared model)")?;
        let limit = box_affected_face_density(sh.coll.key.h);
        let before = sh.coll.samples.len();
        let report = sh.coll.import_sidecar(path, limit)?;
        let new: Vec<Sample> = sh.coll.samples[before..].iter().map(|c| c.sample.clone()).collect();
        let prov = sh.prov.clone();
        for (k, s) in (before..).zip(new) {
            let z = s.z.clone();
            let n = self.store.as_ref().map_or(0, |st| st.len());
            if !self.quarantine.is_empty() {
                self.quarantine.resize(n, Q_NONE);
            }
            let store = self.store.get_or_insert_with(|| SampleStore::new(prov.clone()));
            store.push(s).map_err(|e| format!("import: {e}"))?;
            self.face.resize(n, f64::NAN);
            self.face.push(f64::NAN);
            let sh = self.shared.as_mut().expect("collection mode");
            sh.link.resize(n, Link::Out);
            sh.link.push(Link::In(k));
            if let Some(acc) = &self.accumulator {
                if z.iter().any(|&zi| acc.basis.species_index(zi).is_none()) {
                    self.species_dirty = true;
                }
            }
        }
        if report.imported > 0 {
            self.pending_refit = true;
        }
        let slug = self.shared.as_ref().map_or_else(String::new, |s| s.coll.key.slug());
        self.user_notes.push(format!("shared collection {slug}: {} imported from {}, {} refused", report.imported, path.display(), report.refused.len()));
        self.flush_collection();
        Ok(report)
    }

    /// The model file this provider's model was loaded from (`load`), for the readouts.
    pub(super) fn set_source_file(&mut self, path: Option<PathBuf>) {
        self.source_file = path;
    }
}

impl Drop for SurrogateKs {
    /// The attached collection's audit history and README are written before its lock is released.
    fn drop(&mut self) {
        self.flush_collection();
    }
}
