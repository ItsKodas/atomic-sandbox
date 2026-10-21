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
//! * **References per box** (M8a Task 4a fix round 2, controller ruling; the measurement:
//!   `validate/references.md`, "M8a collection references", series (3) — within the box rule a lone
//!   atom's energy still depends on the box length by up to ≈ 1e-2 Ha). The collection keeps a
//!   reference per (species, box n) (`Collection::offer_reference`: the box rule, odd n), and **each
//!   sample's fit target is E − Σ ref of its own box**. The provider keeps every collection-mode
//!   sample's teacher energy and box (`SharedCollection::raw`, `box_n`) and fits
//!   E_raw − Σ_i (ref(Z_i, its box) − ref_canonical(Z_i)) — with the model's references the
//!   **canonical** ones (each species' largest box: [`SurrogateKs::display_reference`], the
//!   reference for absolute energies), this is exactly E_raw − Σ ref(own box) as the fit's target,
//!   and every model-vs-sample comparison downstream (residuals, leave-one-out, quarantine, fences,
//!   audits) stays on one surface. A teacher answer entering a comparison is put on that surface the
//!   same way ([`SurrogateKs::to_canonical`]; per scene the identity), and so is the step's reported
//!   energy in collection mode (the teacher's own breakdown stays in `EvalStats::energy`): within a
//!   scene the shift is a constant. The collection stores the raw answer. A sample whose box has no
//!   reference for one of its species **waits** out of the fit (`Q_REF_WAIT`, provider-only) until
//!   the reference arrives — from a sampling pass's reference jobs, or from the auditor, which the
//!   provider asks itself when a scene's box lacks one (`request_box_references`; without an
//!   auditor — the CPU teacher — a pass solves it), or with an import; rule 3's gate hands such a
//!   box's steps to the teacher (the model cannot be judged there). When a reference arrives every
//!   energy is rebased and the fit rebuilt (`rebase_collection`). A reference the collection refuses
//!   (even n, box-affected) is used for its box in this session only.
//! * **The audit history** is the collection's (`history.bin`): installed at attach, written back
//!   (with `README.txt`) after every refit, at detach and when the provider is dropped. The class
//!   verdict's hysteresis memory is not persisted in a collection (a reattached collection starts
//!   without one, like a sidecar of version < 7). **The history judges the session's fit** (review
//!   M5): an audit or fallback point compares the teacher with the model standing in this session,
//!   which includes any session-only samples (box-affected ones the provider still fits below its
//!   trust floor). Such a point is written to the collection's `history.bin` like any other; the
//!   difference from the collection-only fit is those few samples' influence (a box-affected sample
//!   leaves the fit once the trust floor is met without it). Recorded, not corrected: a point
//!   measures the model that judged it, and the evidence window ages by model generation.
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
    /// Draft-mode guesses set to wait since the counter was last read (not refusals: module doc).
    waiting: usize,
    /// The audit history's [`history_stamp`] when it was last written.
    history_written: Option<(u64, u64)>,
    /// Per stored sample: the teacher's own energy (Hartree) and the box it was solved in (grid
    /// points per side) — module doc, "References per box".
    pub(super) raw: Vec<f64>,
    pub(super) box_n: Vec<usize>,
    /// The references in use per (Z, box n): the collection's and this session's.
    pub(super) box_refs: BTreeMap<(u8, usize), f64>,
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
    ///
    /// A handle of the collection already attached is dropped and nothing changes (a second open in
    /// this process is read-only: replacing the writable handle with it would lose the lock — review
    /// M6).
    pub fn attach(&mut self, c: Collection) -> Result<(), String> {
        self.attach_inner(c, false)
    }

    /// [`attach`](Self::attach); `restoring`: a collection taken back after a failed load (its
    /// note says so, and says nothing of the samples held before — they are the failed load's).
    pub(super) fn attach_inner(&mut self, c: Collection, restoring: bool) -> Result<(), String> {
        if self.shared.as_ref().is_some_and(|s| s.coll.key == c.key) {
            return Ok(());
        }
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
            prov,
            appended: 0,
            refused: BTreeMap::new(),
            waiting: 0,
            history_written: Some(stamp),
            raw: c.samples.iter().map(|x| x.sample.energy).collect(),
            box_n: c.samples.iter().map(|x| x.grid_n).collect(),
            box_refs: c.references.iter().map(|(&k, r)| (k, r.energy)).collect(),
            write_noted: false,
            coll: c,
        });
        // the fit's energies: each sample against its own box's references (module doc)
        self.rebase_collection();
        self.source_file = None;
        let mut note = format!(
            "shared collection {slug} {}: {n} samples ({held_out} held out of the fit{}), references of {} species in {} (species, box) pairs",
            if restoring { "attached again after the failed load" } else { "attached" },
            match self.waiting_for_reference() {
                0 => String::new(),
                k => format!("; {k} wait for their box's references"),
            },
            self.refs.len(),
            self.shared.as_ref().map_or(0, |s| s.box_refs.len())
        );
        if held > 0 && !restoring {
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

    /// Takes a collection back after a load that failed (review M3): a failure to do so is said.
    pub(super) fn reattach_after_failed_load(&mut self, c: Collection) {
        let slug = c.key.slug();
        if let Err(e) = self.attach_inner(c, true) {
            self.user_notes.push(format!("shared collection {slug}: could not be attached again after the failed load ({e}) — the provider is per-scene now; attach it again to use it"));
        }
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
        self.collection_note.clear();
        let sh = self.shared.as_ref().expect("collection mode");
        if sh.coll.admit_provenance(prov).is_err() {
            return Err(StoreError::ProvenanceMismatch);
        }
        // the same answer is held already (its twin is in the store, mirrored from the collection):
        // a second copy would double its weight and give it a leave-one-out twin (review I1)
        if sh.coll.holds_answer(&s) {
            return Err(StoreError::Duplicate);
        }
        let n = self.store.as_ref().map_or(0, |st| st.len());
        if !self.quarantine.is_empty() {
            self.quarantine.resize(n, Q_NONE);
        }
        let shared_prov = sh.prov.clone();
        // the fit's energy: against its own box's references, or it waits (module doc)
        let raw = s.energy;
        let offset = self.box_offset(&s.z, prov.grid_n);
        let mut s = s;
        if let Some(o) = offset {
            s.energy = raw - o;
        }
        let z_of = s.z.clone();
        let store = self.store.get_or_insert_with(|| SampleStore::new(shared_prov));
        store.push(s)?;
        store.provenance.record_masses(&prov.masses);
        self.face.resize(n, f64::NAN);
        self.face.push(face);
        if offset.is_none() {
            self.quarantine.resize(n + 1, Q_NONE);
            self.quarantine[n] = Q_REF_WAIT;
        }
        let sh = self.shared.as_mut().expect("collection mode");
        sh.link.resize(n, Link::Out);
        sh.link.push(Link::Out);
        sh.raw.resize(n, f64::NAN);
        sh.raw.push(raw);
        sh.box_n.resize(n, 0);
        sh.box_n.push(prov.grid_n);
        let link = self.offer(n, prov, multiplicity);
        self.shared.as_mut().expect("collection mode").link[n] = link;
        if offset.is_none() {
            let missing: Vec<&str> = self.box_refs_missing(&z_of, prov.grid_n).into_iter().map(symbol).collect();
            self.collection_note.push(format!(
                "surrogate: kept, waiting out of the fit for the isolated-atom reference of {} in its {}³ box (its fit target is E − Σ ref of its own box)",
                missing.join(", "),
                prov.grid_n
            ));
        }
        Ok(())
    }

    /// Offers stored sample `i` (solved at `prov`) to the collection: its link afterwards. The
    /// code passed to `admit` is `Q_BOX` for a box-affected answer, `Q_DRAFT` for a draft guess's
    /// audit, else the provider's own; an answer without a face density is refused as box status
    /// unknown (module doc).
    fn offer(&mut self, i: usize, prov: &Provenance, multiplicity: Option<u32>) -> Link {
        let mut s = self.store.as_ref().expect("stored").samples[i].clone();
        // the collection keeps the teacher's own answer (module doc, "References per box")
        if let Some(&raw) = self.shared.as_ref().and_then(|sh| sh.raw.get(i)) {
            s.energy = raw;
        }
        let face = self.face.get(i).copied().unwrap_or(f64::NAN);
        let draft = s.tag == SampleTag::DraftAudit || self.quarantine.get(i) == Some(&Q_DRAFT);
        let code = if face > box_affected_face_density(prov.h) {
            Q_BOX
        } else if draft {
            Q_DRAFT
        } else {
            self.quarantine.get(i).copied().filter(|&q| q != Q_REF_WAIT).unwrap_or(Q_NONE)
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
            Err(_) if code == Q_DRAFT => {
                sh.waiting += 1;
                let slug = sh.coll.key.slug();
                self.collection_note.push(format!(
                    "surrogate: held out of the strict fit, so not in the shared collection {slug} yet — it is added if a refit promotes it into the fit"
                ));
                Link::Waiting { grid_n: prov.grid_n, multiplicity }
            }
            Err(why) => {
                *sh.refused.entry(why.clone()).or_default() += 1;
                let slug = sh.coll.key.slug();
                self.collection_note.push(format!("surrogate: kept for this session only — the shared collection {slug} did not take it ({why})"));
                Link::Out
            }
        }
    }

    /// Writes a status record for every collection sample whose quarantine code changed, and
    /// offers a promoted draft guess again (module doc). A failed write is said once.
    pub(super) fn sync_collection_status(&mut self) {
        let Some(sh) = self.shared.as_mut() else { return };
        let mut failed: Option<String> = None;
        let mut promoted: Vec<(usize, usize, Option<u32>)> = Vec::new();
        for (i, link) in sh.link.iter().enumerate() {
            let code = self.quarantine.get(i).copied().filter(|&q| q != Q_REF_WAIT).unwrap_or(Q_NONE);
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
            self.collection_note.clear();
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

    /// The pass's summary of what the collection took (`finish_pass`; `held`: the pass's jobs not
    /// solved because their geometry was held already): `None` without a collection or when nothing
    /// was offered or skipped since the last summary.
    pub(super) fn take_collection_summary(&mut self, held: usize) -> Option<String> {
        let wait_ref = self.waiting_for_reference();
        let sh = self.shared.as_mut()?;
        let refused: usize = sh.refused.values().sum();
        if sh.appended == 0 && refused == 0 && sh.waiting == 0 && held == 0 && wait_ref == 0 {
            return None;
        }
        let why: Vec<String> = std::mem::take(&mut sh.refused).into_iter().map(|(w, n)| format!("{n} {w}")).collect();
        let appended = std::mem::take(&mut sh.appended);
        let waiting = std::mem::take(&mut sh.waiting);
        let mut text = format!("shared collection {}: {appended} samples added ({} in all)", sh.coll.key.slug(), sh.coll.samples.len());
        if held > 0 {
            text.push_str(&format!("; {held} geometries already in it, not solved again"));
        }
        if refused > 0 {
            text.push_str(&format!("; {refused} kept for this session only — {}", why.join("; ")));
        }
        if waiting > 0 {
            text.push_str(&format!("; {waiting} draft-mode guesses held out of the strict fit wait outside it"));
        }
        let wait_ref = self.waiting_for_reference();
        if wait_ref > 0 {
            text.push_str(&format!("; {wait_ref} samples wait out of the fit for their box's isolated-atom references"));
        }
        Some(text)
    }

    /// Whether the attached collection holds `scene`'s geometry bit for bit (the pass's
    /// skip-before-solving; `crate::sampling`, `held_by_collection`). `false` without a collection
    /// or for a scene that cannot share it.
    pub(crate) fn collection_holds_geometry(&self, scene: &Scene) -> bool {
        let Some(sh) = self.shared.as_ref() else { return false };
        if sh.coll.admit_provenance(&self.scene_provenance(scene)).is_err() {
            return false;
        }
        let z: Vec<u8> = scene.nuclei.iter().map(|n| n.z).collect();
        let pos: Vec<[f64; 3]> = scene.nuclei.iter().map(|n| n.pos).collect();
        sh.coll.holds_geometry(&z, &pos)
    }

    /// A solved isolated-atom reference of species `z` (Hartree) with its box (`grid_n`, the solve's
    /// face density): the sampling pass's setter (and the auditor's answers in collection mode).
    /// Without a collection it is [`set_reference_energy`](Self::set_reference_energy); in collection
    /// mode it is offered to the collection for its (species, box) and used for that box — the
    /// collection's stored one when it has one, else this one (for this session only when the
    /// collection refuses it) — and every energy is rebased (module doc, "References per box").
    pub fn set_reference_solved(&mut self, z: u8, e: f64, grid_n: usize, face: Option<f64>) {
        let Some(sh) = self.shared.as_mut() else { return self.set_reference_energy(z, e) };
        let limit = box_affected_face_density(sh.coll.key.h);
        let slug = sh.coll.key.slug();
        let name = symbol(z);
        let offered = CollectionReference { energy: e, grid_n: Some(grid_n), face_density: face };
        let (value, note) = match sh.coll.offer_reference(z, offered, limit) {
            Ok(ReferenceOutcome::Stored) => (e, Some(format!("shared collection {slug}: the {name} reference of the {grid_n}³ box, {e:.6} Ha, is stored"))),
            Ok(ReferenceOutcome::Kept { stored }) => (
                stored.energy,
                (stored.energy.to_bits() != e.to_bits()).then(|| format!("shared collection {slug}: the {name} reference of the {grid_n}³ box is the stored {:.6} Ha (this solve: Δ {:+.1e} Ha)", stored.energy, e - stored.energy)),
            ),
            Ok(ReferenceOutcome::Refused(why)) => (e, Some(format!("shared collection {slug}: the {name} reference of the {grid_n}³ box is not stored ({why}) — used for this box in this session only"))),
            Err(w) => (e, Some(format!("shared collection {slug}: the {name} reference of the {grid_n}³ box could not be written ({w}) — used for this session only"))),
        };
        let changed = sh.box_refs.insert((z, grid_n), value).is_none_or(|old| old.to_bits() != value.to_bits());
        self.user_notes.extend(note);
        self.failed_refs.remove(&z);
        self.legacy_refs.remove(&z);
        if changed {
            self.rebase_collection();
        }
    }

    /// The canonical reference of each species: its largest box's (module doc, "References per box").
    fn canonical_refs(box_refs: &BTreeMap<(u8, usize), f64>) -> BTreeMap<u8, f64> {
        let mut out = BTreeMap::new();
        for (&(z, _), &e) in box_refs {
            out.insert(z, e); // ascending n: the largest box's comes last
        }
        out
    }

    /// The reference for absolute (display) energies of species `z`: in collection mode its largest
    /// box's (module doc, "References per box"), else the model's reference.
    pub fn display_reference(&self, z: u8) -> Option<f64> {
        self.refs.get(&z).copied()
    }

    /// Σ_i (ref(Z_i, box n) − ref_canonical(Z_i)) over a geometry's atoms: what separates its own
    /// box's fit target from the canonical surface. `Some(0)` without a collection; `None` when the
    /// box lacks a reference of one of the species.
    pub(super) fn box_offset(&self, z: &[u8], n: usize) -> Option<f64> {
        let Some(sh) = self.shared.as_ref() else { return Some(0.0) };
        let mut off = 0.0;
        for &zi in z {
            let own = sh.box_refs.get(&(zi, n))?;
            let canon = sh.box_refs.range((zi, 0)..=(zi, usize::MAX)).next_back().map(|(_, &e)| e)?;
            off += own - canon;
        }
        Some(off)
    }

    /// A teacher energy of a geometry solved in box `n` on the canonical surface (module doc,
    /// "References per box"); unchanged without a collection or when the box lacks a reference.
    pub(super) fn to_canonical(&self, z: &[u8], n: usize, e: f64) -> f64 {
        self.box_offset(z, n).map_or(e, |o| e - o)
    }

    /// The inverse of [`to_canonical`](Self::to_canonical): the teacher's own energy back.
    pub(super) fn to_raw(&self, z: &[u8], n: usize, e: f64) -> f64 {
        self.box_offset(z, n).map_or(e, |o| e + o)
    }

    /// The species of `z` whose isolated-atom reference box `n` lacks (collection mode; empty
    /// otherwise, and for the collection's own any-box provenance, n = 0).
    pub(super) fn box_refs_missing(&self, z: &[u8], n: usize) -> Vec<u8> {
        let Some(sh) = self.shared.as_ref().filter(|_| n != 0) else { return Vec::new() };
        let mut out: Vec<u8> = z.iter().copied().filter(|&zi| !sh.box_refs.contains_key(&(zi, n))).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Samples waiting out of the fit for their box's references (`Q_REF_WAIT`).
    pub fn waiting_for_reference(&self) -> usize {
        self.quarantine.iter().filter(|&&q| q == Q_REF_WAIT).count()
    }

    /// Recomputes every collection-mode sample's fit energy from its teacher energy and its box's
    /// references (module doc, "References per box"): the model's references become the canonical
    /// ones, a sample whose box now has its references leaves the wait, one whose box lacks them
    /// waits; anything changed rebuilds the fit at the next refit.
    pub(super) fn rebase_collection(&mut self) {
        let Some(sh) = self.shared.as_ref() else { return };
        let canon = Self::canonical_refs(&sh.box_refs);
        if canon != self.refs {
            self.refs = canon;
            self.refs_dirty = true;
        }
        let n = self.store.as_ref().map_or(0, |st| st.len());
        let offsets: Vec<Option<f64>> = (0..n)
            .map(|i| {
                let sh = self.shared.as_ref().expect("collection mode");
                let z = &self.store.as_ref().expect("stored").samples[i].z;
                self.box_offset(z, sh.box_n[i])
            })
            .collect();
        let raw = self.shared.as_ref().expect("collection mode").raw.clone();
        let mut changed = false;
        if !self.quarantine.is_empty() || offsets.iter().any(Option::is_none) {
            self.quarantine.resize(n, Q_NONE);
        }
        for (i, off) in offsets.into_iter().enumerate() {
            match off {
                Some(o) => {
                    let e = raw[i] - o;
                    let st = self.store.as_mut().expect("stored");
                    if st.samples[i].energy.to_bits() != e.to_bits() {
                        st.samples[i].energy = e;
                        changed = true;
                    }
                    if self.quarantine.get(i) == Some(&Q_REF_WAIT) {
                        self.quarantine[i] = Q_NONE;
                        changed = true;
                    }
                }
                None => {
                    if self.quarantine.get(i).is_none_or(|&q| q == Q_NONE) {
                        self.quarantine.resize(n, Q_NONE);
                        self.quarantine[i] = Q_REF_WAIT;
                        changed = true;
                    }
                }
            }
        }
        if changed {
            self.species_dirty = true;
            self.pending_refit = n > 0;
        }
    }

    /// In collection mode with an auditor: asks it for every isolated-atom reference `scene`'s box
    /// lacks (module doc, "References per box"); the answers are the provider's own (not handed on
    /// to the driver unless it asks too).
    pub(super) fn request_box_references(&mut self, scene: &Scene) {
        if self.shared.is_none() || self.auditor.is_none() {
            return;
        }
        let n = scene.cell.n;
        let z: Vec<u8> = scene.nuclei.iter().map(|x| x.z).collect();
        for zm in self.box_refs_missing(&z, n) {
            if self.own_ref_requests.contains(&(zm, n)) {
                continue;
            }
            let mass = scene.nuclei.iter().find(|x| x.z == zm).map_or(0.0, |x| x.mass);
            if self.request_atom_reference_job(zm, mass, scene) {
                self.own_ref_requests.push((zm, n));
            }
        }
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
        let new: Vec<(Sample, usize)> = sh.coll.samples[before..].iter().map(|c| (c.sample.clone(), c.grid_n)).collect();
        let prov = sh.prov.clone();
        for (k, (s, grid_n)) in (before..).zip(new) {
            let raw = s.energy;
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
            sh.raw.resize(n, f64::NAN);
            sh.raw.push(raw);
            sh.box_n.resize(n, 0);
            sh.box_n.push(grid_n);
            if let Some(acc) = &self.accumulator {
                if z.iter().any(|&zi| acc.basis.species_index(zi).is_none()) {
                    self.species_dirty = true;
                }
            }
        }
        if report.imported > 0 {
            self.pending_refit = true;
            // each against its own box's references; a box without them waits (module doc)
            self.rebase_collection();
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
